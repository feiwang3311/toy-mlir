//====- LowerToAffineLoops.cpp - Partial lowering from Toy to Affine+Std --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a partial lowering of Toy operations to a combination of
// affine loops, memref operations and standard operations. This lowering
// expects that all calls have been inlined, and all shapes have been resolved.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "toy/Dialect.h"
#include "toy/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Parser/Parser.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>

using namespace mlir;

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns
//===----------------------------------------------------------------------===//

/// Convert the given RankedTensorType into the corresponding MemRefType.
static MemRefType convertTensorToMemRef(RankedTensorType type) {
  return MemRefType::get(type.getShape(), type.getElementType());
}

/// Insert an allocation and deallocation for the given MemRefType.
static Value insertAllocAndDealloc(MemRefType type, Location loc,
                                   PatternRewriter &rewriter) {
  auto alloc = rewriter.create<memref::AllocOp>(loc, type);

  // Make sure to allocate at the beginning of the block.
  auto *parentBlock = alloc->getBlock();
  alloc->moveBefore(&parentBlock->front());

  // Make sure to deallocate this alloc at the end of the block. This is fine
  // as toy functions have no control flow.
  auto dealloc = rewriter.create<memref::DeallocOp>(loc, alloc);
  dealloc->moveBefore(&parentBlock->back());
  return alloc;
}

/// This defines the function type used to process an iteration of a lowered
/// loop. It takes as input an OpBuilder, an range of memRefOperands
/// corresponding to the operands of the input operation, and the range of loop
/// induction variables for the iteration. It returns a value to store at the
/// current index of the iteration.
using LoopIterationFn = function_ref<Value(
    OpBuilder &rewriter, ValueRange memRefOperands, ValueRange loopIvs)>;

static void lowerOpToLoops(Operation *op, ValueRange operands,
                           PatternRewriter &rewriter,
                           LoopIterationFn processIteration) {
  auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));
  auto loc = op->getLoc();

  // Insert an allocation and deallocation for the result of this operation.
  auto memRefType = convertTensorToMemRef(tensorType);
  auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

  // Create a nest of affine loops, with one loop per dimension of the shape.
  // The buildAffineLoopNest function takes a callback that is used to construct
  // the body of the innermost loop given a builder, a location and a range of
  // loop induction variables.
  SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value=*/0);
  SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);
  affine::buildAffineLoopNest(
      rewriter, loc, lowerBounds, tensorType.getShape(), steps,
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
        // Call the processing function with the rewriter, the memref operands,
        // and the loop induction variables. This function will return the value
        // to store at the current index.
        Value valueToStore = processIteration(nestedBuilder, operands, ivs);
        nestedBuilder.create<affine::AffineStoreOp>(loc, valueToStore, alloc,
                                                    ivs);
      });

  // Replace this operation with the generated alloc.
  rewriter.replaceOp(op, alloc);
}

namespace {
//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Binary operations
//===----------------------------------------------------------------------===//

template <typename BinaryOp, typename LoweredBinaryOp>
struct BinaryOpLowering : public ConversionPattern {
  BinaryOpLowering(MLIRContext *ctx)
      : ConversionPattern(BinaryOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    lowerOpToLoops(op, operands, rewriter,
                   [loc](OpBuilder &builder, ValueRange memRefOperands,
                         ValueRange loopIvs) {
                     // Generate an adaptor for the remapped operands of the
                     // BinaryOp. This allows for using the nice named accessors
                     // that are generated by the ODS.
                     typename BinaryOp::Adaptor binaryAdaptor(memRefOperands);

                     // Generate loads for the element of 'lhs' and 'rhs' at the
                     // inner loop.
                     auto loadedLhs = builder.create<affine::AffineLoadOp>(
                         loc, binaryAdaptor.getLhs(), loopIvs);
                     auto loadedRhs = builder.create<affine::AffineLoadOp>(
                         loc, binaryAdaptor.getRhs(), loopIvs);

                     // Create the binary operation performed on the loaded
                     // values.
                     return builder.create<LoweredBinaryOp>(loc, loadedLhs,
                                                            loadedRhs);
                   });
    return success();
  }
};
using AddOpLowering = BinaryOpLowering<toy::AddOp, arith::AddFOp>;
using MulOpLowering = BinaryOpLowering<toy::MulOp, arith::MulFOp>;

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Constant operations
//===----------------------------------------------------------------------===//

struct ConstantOpLowering : public OpRewritePattern<toy::ConstantOp> {
  using OpRewritePattern<toy::ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(toy::ConstantOp op,
                                PatternRewriter &rewriter) const final {
    DenseElementsAttr constantValue = op.getValue();
    Location loc = op.getLoc();

    // When lowering the constant operation, we allocate and assign the constant
    // values to a corresponding memref allocation.
    auto tensorType = llvm::cast<RankedTensorType>(op.getType());
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // We will be generating constant indices up-to the largest dimension.
    // Create these constants up-front to avoid large amounts of redundant
    // operations.
    auto valueShape = memRefType.getShape();
    SmallVector<Value, 8> constantIndices;

    if (!valueShape.empty()) {
      for (auto i : llvm::seq<int64_t>(
               0, *std::max_element(valueShape.begin(), valueShape.end())))
        constantIndices.push_back(
            rewriter.create<arith::ConstantIndexOp>(loc, i));
    } else {
      // This is the case of a tensor of rank 0.
      constantIndices.push_back(
          rewriter.create<arith::ConstantIndexOp>(loc, 0));
    }

    // The constant operation represents a multi-dimensional constant, so we
    // will need to generate a store for each of the elements. The following
    // functor recursively walks the dimensions of the constant shape,
    // generating a store when the recursion hits the base case.
    SmallVector<Value, 2> indices;
    auto valueIt = constantValue.value_begin<FloatAttr>();
    std::function<void(uint64_t)> storeElements = [&](uint64_t dimension) {
      // The last dimension is the base case of the recursion, at this point
      // we store the element at the given index.
      if (dimension == valueShape.size()) {
        rewriter.create<affine::AffineStoreOp>(
            loc, rewriter.create<arith::ConstantOp>(loc, *valueIt++), alloc,
            llvm::ArrayRef(indices));
        return;
      }

      // Otherwise, iterate over the current dimension and add the indices to
      // the list.
      for (uint64_t i = 0, e = valueShape[dimension]; i != e; ++i) {
        indices.push_back(constantIndices[i]);
        storeElements(dimension + 1);
        indices.pop_back();
      }
    };

    // Start the element storing recursion from the first dimension.
    storeElements(/*dimension=*/0);

    // Replace this operation with the generated alloc.
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Func operations
//===----------------------------------------------------------------------===//

struct FuncOpLowering : public OpConversionPattern<toy::FuncOp> {
  using OpConversionPattern<toy::FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(toy::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // We only lower the main function as we expect that all other functions
    // have been inlined.
    if (op.getName() != "main")
      return failure();

    // Verify that the given main has no inputs and results.
    if (op.getNumArguments() || op.getFunctionType().getNumResults()) {
      return rewriter.notifyMatchFailure(op, [](Diagnostic &diag) {
        diag << "expected 'main' to have 0 inputs and 0 results";
      });
    }

    // Create a new non-toy function, with the same region.
    auto func = rewriter.create<mlir::func::FuncOp>(op.getLoc(), op.getName(),
                                                    op.getFunctionType());
    rewriter.inlineRegionBefore(op.getRegion(), func.getBody(), func.end());
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Print operations
//===----------------------------------------------------------------------===//

struct PrintOpLowering : public OpConversionPattern<toy::PrintOp> {
  using OpConversionPattern<toy::PrintOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(toy::PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // We don't lower "toy.print" in this pass, but we need to update its
    // operands.
    rewriter.updateRootInPlace(op,
                               [&] { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Return operations
//===----------------------------------------------------------------------===//

struct ReturnOpLowering : public OpRewritePattern<toy::ReturnOp> {
  using OpRewritePattern<toy::ReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(toy::ReturnOp op,
                                PatternRewriter &rewriter) const final {
    // During this lowering, we expect that all function calls have been
    // inlined.
    if (op.hasOperand())
      return failure();

    // We lower "toy.return" directly to "func.return".
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Transpose operations
//===----------------------------------------------------------------------===//

struct TransposeOpLowering : public ConversionPattern {
  TransposeOpLowering(MLIRContext *ctx)
      : ConversionPattern(toy::TransposeOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    lowerOpToLoops(op, operands, rewriter,
                   [loc](OpBuilder &builder, ValueRange memRefOperands,
                         ValueRange loopIvs) {
                     // Generate an adaptor for the remapped operands of the
                     // TransposeOp. This allows for using the nice named
                     // accessors that are generated by the ODS.
                     toy::TransposeOpAdaptor transposeAdaptor(memRefOperands);
                     Value input = transposeAdaptor.getInput();

                     // Transpose the elements by generating a load from the
                     // reverse indices.
                     SmallVector<Value, 2> reverseIvs(llvm::reverse(loopIvs));
                     return builder.create<affine::AffineLoadOp>(loc, input,
                                                                 reverseIvs);
                   });
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: TransposeMul optimization
//===----------------------------------------------------------------------===//

/// Pattern to match a toy.transpose followed by toy.mul with the same operand
struct TransposeMulPattern : public OpRewritePattern<toy::MulOp> {
  using OpRewritePattern<toy::MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(toy::MulOp mulOp, PatternRewriter &rewriter) const override {
    llvm::errs() << "[TransposeMulPattern] Checking MulOp at " << mulOp.getLoc() << "\n";

    // Check if both operands of mul are the same
    if (mulOp.getLhs() != mulOp.getRhs()) {
      llvm::errs() << "[TransposeMulPattern] Operands are different - skipping\n";
      return failure();
    }
    llvm::errs() << "[TransposeMulPattern] Both operands are the same!\n";

    // Check if the operand is a transpose operation
    auto transposeOp = mulOp.getLhs().getDefiningOp<toy::TransposeOp>();
    if (!transposeOp) {
      llvm::errs() << "[TransposeMulPattern] Operand is not a transpose - skipping\n";
      return failure();
    }
    llvm::errs() << "[TransposeMulPattern] Operand is a transpose! Pattern matches!\n";

    // Get the module operation
    ModuleOp moduleOp = mulOp->getParentOfType<ModuleOp>();
    if (!moduleOp) {
      llvm::errs() << "[TransposeMulPattern] Cannot find module - skipping\n";
      return failure();
    }

    // Create a new function name
    std::string funcName = "transpose_mul_opt";

    // Check if the function already exists in the module
    toy::FuncOp existingFunc = moduleOp.lookupSymbol<toy::FuncOp>(funcName);
    if (!existingFunc) {
      llvm::errs() << "[TransposeMulPattern] Creating new function: " << funcName << "\n";
      // Save the current insertion point
      OpBuilder::InsertionGuard guard(rewriter);

      // Set insertion point to the end of the module to insert the new function
      rewriter.setInsertionPointToEnd(moduleOp.getBody());

      // Create the function type (takes one tensor, returns one tensor)
      auto inputType = transposeOp.getInput().getType();
      FunctionType funcType = rewriter.getFunctionType({inputType}, {inputType});

      // Create the function
      existingFunc = rewriter.create<toy::FuncOp>(mulOp.getLoc(), funcName, funcType);
      existingFunc.setPrivate();

      // Get or create the function body
      Block *block;
      if (existingFunc.getBody().empty()) {
        block = existingFunc.addEntryBlock();
      } else {
        block = &existingFunc.getBody().front();
      }
      rewriter.setInsertionPointToStart(block);

      // Add the transpose and mul operations
      auto arg = existingFunc.getArgument(0);
      auto newTranspose = rewriter.create<toy::TransposeOp>(mulOp.getLoc(), arg);
      auto newMul = rewriter.create<toy::MulOp>(mulOp.getLoc(), newTranspose.getResult(), newTranspose.getResult());

      // Add the return operation
      rewriter.create<toy::ReturnOp>(mulOp.getLoc(), newMul.getResult());
      llvm::errs() << "[TransposeMulPattern] Function created successfully!\n";
    } else {
      llvm::errs() << "[TransposeMulPattern] Function already exists, reusing it\n";
    }

    // Replace the original mul operation with a call to the new function
    llvm::errs() << "[TransposeMulPattern] Creating GenericCallOp to " << funcName << "\n";
    SmallVector<Value, 1> callOperands;
    callOperands.push_back(transposeOp.getInput());
    auto callOp = rewriter.create<toy::GenericCallOp>(
        mulOp.getLoc(),
        mulOp.getType(),
        mlir::SymbolRefAttr::get(rewriter.getContext(), funcName),
        callOperands);

    // Replace the mul operation with the call result
    llvm::errs() << "[TransposeMulPattern] Replacing MulOp with call\n";
    rewriter.replaceOp(mulOp, callOp.getResult());

    llvm::errs() << "[TransposeMulPattern] Pattern applied successfully!\n";
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern File Parsing Infrastructure
//===----------------------------------------------------------------------===//

namespace {

// Structure to hold parsed pattern information
struct PatternInfo {
  SmallVector<std::string> operationNames;  // Operation names in the pattern
  size_t numInputs;                          // Number of inputs
  size_t numOutputs;                         // Number of outputs
  std::string name;                          // Pattern name
  OwningOpRef<ModuleOp> patternModule;      // Keep the parsed module alive

  void dump() const {
    llvm::errs() << "=== Pattern: " << name << " ===\n";
    llvm::errs() << "Operations (" << operationNames.size() << "):\n";
    for (const auto &opName : operationNames) {
      llvm::errs() << "  " << opName << "\n";
    }
    llvm::errs() << "Inputs: " << numInputs << "\n";
    llvm::errs() << "Outputs: " << numOutputs << "\n";
    llvm::errs() << "===================\n";
  }
};

// Parse a pattern file and extract pattern information
LogicalResult parsePatternFile(StringRef filePath, MLIRContext *context,
                               PatternInfo &patternInfo) {
  llvm::errs() << "[PatternParser] Reading pattern file: " << filePath << "\n";

  // Read the pattern file
  auto fileOrErr = llvm::MemoryBuffer::getFile(filePath);
  if (!fileOrErr) {
    llvm::errs() << "[PatternParser] ERROR: Could not read file: "
                 << fileOrErr.getError().message() << "\n";
    return failure();
  }

  llvm::errs() << "[PatternParser] File contents:\n"
               << fileOrErr.get()->getBuffer() << "\n";

  // Parse the MLIR snippet as a module
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());

  patternInfo.patternModule = parseSourceFile<ModuleOp>(sourceMgr, context);
  if (!patternInfo.patternModule) {
    llvm::errs() << "[PatternParser] ERROR: Failed to parse pattern file as module\n";
    return failure();
  }

  llvm::errs() << "[PatternParser] Successfully parsed module\n";

  // Find the pattern function
  toy::FuncOp patternFunc = nullptr;
  patternInfo.patternModule->walk([&](toy::FuncOp func) {
    if (func.getName() == "pattern") {
      patternFunc = func;
    }
  });

  if (!patternFunc) {
    llvm::errs() << "[PatternParser] ERROR: No toy.func @pattern found\n";
    return failure();
  }

  llvm::errs() << "[PatternParser] Found pattern function with "
               << patternFunc.getNumArguments() << " arguments\n";

  // Get the function body
  Block *patternBlock = &patternFunc.getBody().front();

  llvm::errs() << "[PatternParser] Function body has "
               << patternBlock->getOperations().size() << " operations\n";

  // Extract operation names (except return)
  Operation *returnOp = nullptr;
  for (auto &op : patternBlock->getOperations()) {
    if (isa<func::ReturnOp>(&op) || isa<toy::ReturnOp>(&op)) {
      returnOp = &op;
      llvm::errs() << "[PatternParser] Found return op\n";
    } else {
      patternInfo.operationNames.push_back(op.getName().getStringRef().str());
      llvm::errs() << "[PatternParser] Added operation: " << op.getName() << "\n";
    }
  }

  if (!returnOp) {
    llvm::errs() << "[PatternParser] ERROR: No return operation found\n";
    return failure();
  }

  // Count outputs from return operands
  patternInfo.numOutputs = returnOp->getNumOperands();
  llvm::errs() << "[PatternParser] Outputs: " << patternInfo.numOutputs << "\n";

  // Count inputs: function arguments
  patternInfo.numInputs = patternFunc.getNumArguments();
  llvm::errs() << "[PatternParser] Inputs: " << patternInfo.numInputs << "\n";

  llvm::errs() << "[PatternParser] Pattern parsing complete!\n";
  return success();
}

//===----------------------------------------------------------------------===//
// Pattern Matching Infrastructure
//===----------------------------------------------------------------------===//

// Structure to hold a single pattern match result
struct PatternMatch {
  SmallVector<Operation*> matchedOps;     // Operations that matched
  DenseMap<Value, Value> valueBindings;   // Pattern value -> IR value
  Location loc;                            // Location of the match

  PatternMatch(Location loc) : loc(loc) {}
};

class PatternMatcher {
public:
  PatternMatcher(PatternInfo &pattern) : pattern(pattern) {
    // Get the pattern function to access its operations
    toy::FuncOp patternFunc = nullptr;
    pattern.patternModule->walk([&](toy::FuncOp func) {
      if (func.getName() == "pattern") {
        patternFunc = func;
      }
    });

    if (patternFunc) {
      Block *patternBlock = &patternFunc.getBody().front();
      for (auto &op : patternBlock->getOperations()) {
        if (!isa<func::ReturnOp>(&op) && !isa<toy::ReturnOp>(&op)) {
          patternOps.push_back(&op);
        }
      }
    }
  }

  // Find all matches in the module
  SmallVector<PatternMatch> findMatches(ModuleOp module) {
    SmallVector<PatternMatch> matches;

    llvm::errs() << "[PatternMatcher] Searching for matches...\n";

    // Walk all operations in the module
    module.walk([&](Operation *op) {
      // Try to match starting from this operation
      bindings.clear();
      matchedOps.clear();

      if (matchSequence(op, 0)) {
        llvm::errs() << "[PatternMatcher] ✓ Found match at " << op->getLoc() << "\n";

        PatternMatch match(op->getLoc());
        match.matchedOps = matchedOps;
        match.valueBindings = bindings;
        matches.push_back(match);
      }
    });

    return matches;
  }

private:
  PatternInfo &pattern;
  SmallVector<Operation*> patternOps;              // Pattern operations
  DenseMap<Value, Value> bindings;                 // Pattern value -> IR value
  SmallVector<Operation*> matchedOps;              // Currently matched ops

  // Match a value, respecting existing bindings
  bool matchValue(Value patternVal, Value irVal) {
    // Check if this pattern value is already bound
    if (bindings.count(patternVal)) {
      bool matches = (bindings[patternVal] == irVal);
      if (!matches) {
        llvm::errs() << "  [matchValue] ✗ Binding conflict: pattern value already bound to different IR value\n";
      }
      return matches;
    }

    // First time seeing this pattern value - bind it
    bindings[patternVal] = irVal;
    llvm::errs() << "  [matchValue] ✓ Bound pattern value to IR value\n";
    return true;
  }

  // Match a single operation
  bool matchOp(Operation *patternOp, Operation *irOp) {
    llvm::errs() << "  [matchOp] Trying to match " << patternOp->getName()
                 << " with " << irOp->getName() << "\n";

    // Match operation name (ignore types)
    if (patternOp->getName() != irOp->getName()) {
      llvm::errs() << "  [matchOp] ✗ Operation names don't match\n";
      return false;
    }

    // Special handling for toy.generic_call: match by callee prefix
    if (auto patternCall = dyn_cast<toy::GenericCallOp>(patternOp)) {
      if (auto irCall = dyn_cast<toy::GenericCallOp>(irOp)) {
        StringRef patternCallee = patternCall.getCallee();
        StringRef irCallee = irCall.getCallee();

        llvm::errs() << "  [matchOp] Matching call: pattern callee='" << patternCallee
                     << "' vs IR callee='" << irCallee << "'\n";

        // Remove _N suffix from IR callee (e.g., "transpose_mul_0" -> "transpose_mul")
        // Split on last '_' and check if suffix is a number
        auto splitResult = irCallee.rsplit('_');
        StringRef irCalleePrefix = splitResult.first;
        StringRef suffix = splitResult.second;

        // Only strip suffix if it's actually a number
        bool hasSuffix = !suffix.empty() &&
                         std::all_of(suffix.begin(), suffix.end(),
                                   [](char c) { return std::isdigit(c); });

        if (hasSuffix) {
          llvm::errs() << "  [matchOp] IR callee has numeric suffix, using prefix: '"
                       << irCalleePrefix << "'\n";
          // Check if IR callee prefix matches pattern callee exactly
          if (irCalleePrefix != patternCallee) {
            llvm::errs() << "  [matchOp] ✗ Callee prefix doesn't match\n";
            return false;
          }
        } else {
          // No numeric suffix, require exact match
          if (irCallee != patternCallee) {
            llvm::errs() << "  [matchOp] ✗ Callee doesn't match (no suffix)\n";
            return false;
          }
        }

        llvm::errs() << "  [matchOp] ✓ Callee matched!\n";
        // Continue to match operands below
      }
    }

    // Match operand count
    if (patternOp->getNumOperands() != irOp->getNumOperands()) {
      llvm::errs() << "  [matchOp] ✗ Operand counts don't match ("
                   << patternOp->getNumOperands() << " vs "
                   << irOp->getNumOperands() << ")\n";
      return false;
    }

    // Match operands (with SSA binding)
    for (auto [patternOperand, irOperand] :
         llvm::zip(patternOp->getOperands(), irOp->getOperands())) {
      if (!matchValue(patternOperand, irOperand)) {
        return false;
      }
    }

    llvm::errs() << "  [matchOp] ✓ Operation matched successfully\n";
    return true;
  }

  // Match all pattern operations starting from a candidate operation
  // This handles DAG patterns by matching based on data dependencies
  bool matchSequence(Operation *startOp, size_t patternIndex) {
    if (patternIndex >= patternOps.size()) {
      // Successfully matched all pattern operations
      return true;
    }

    Operation *patternOp = patternOps[patternIndex];

    llvm::errs() << "[matchDAG] Matching pattern op " << patternIndex
                 << " (" << patternOp->getName() << ")\n";

    // Try to match this pattern operation with startOp
    if (!matchOp(patternOp, startOp)) {
      return false;
    }

    // Add to matched operations
    matchedOps.push_back(startOp);

    // Bind pattern result to IR result
    if (patternOp->getNumResults() == 1 && startOp->getNumResults() == 1) {
      bindings[patternOp->getResult(0)] = startOp->getResult(0);
    }

    // Try to match remaining pattern operations
    // Search for them based on data dependencies, not sequential order
    if (matchRemainingOps(patternIndex + 1)) {
      return true;
    }

    // Backtrack
    matchedOps.pop_back();
    return false;
  }

  // Match remaining pattern operations by searching the IR based on data flow
  bool matchRemainingOps(size_t startIndex) {
    if (startIndex >= patternOps.size()) {
      return true;  // All matched
    }

    Operation *patternOp = patternOps[startIndex];

    llvm::errs() << "[matchRemaining] Looking for pattern op " << startIndex
                 << " (" << patternOp->getName() << ")\n";

    // Find candidate operations in the IR that could match this pattern op
    // Strategy: look for operations that:
    // 1. Have the same name
    // 2. Are in the same block as already matched ops
    // 3. Use values we've already bound

    Block *searchBlock = matchedOps.empty() ? nullptr : matchedOps[0]->getBlock();

    if (!searchBlock) {
      llvm::errs() << "[matchRemaining] ✗ No search block\n";
      return false;
    }

    // Search the block for matching operation
    for (auto &candidateOp : searchBlock->getOperations()) {
      // Skip if already matched
      if (std::find(matchedOps.begin(), matchedOps.end(), &candidateOp) != matchedOps.end()) {
        continue;
      }

      // Try to match this candidate
      if (matchOp(patternOp, &candidateOp)) {
        matchedOps.push_back(&candidateOp);

        // Bind results
        if (patternOp->getNumResults() == 1 && candidateOp.getNumResults() == 1) {
          bindings[patternOp->getResult(0)] = candidateOp.getResult(0);
        }

        // Recursively match remaining operations
        if (matchRemainingOps(startIndex + 1)) {
          return true;
        }

        // Backtrack
        matchedOps.pop_back();
        // Note: bindings are not removed here, which is OK because we'll clear
        // them when we try a new match from scratch
      }
    }

    llvm::errs() << "[matchRemaining] ✗ Could not find match for pattern op " << startIndex << "\n";
    return false;
  }
};

//===----------------------------------------------------------------------===//
// Pattern Extraction and Replacement
//===----------------------------------------------------------------------===//

class FunctionExtractor {
public:
  // Extract matched pattern to a new toy.func and replace with call
  void extractAndReplace(PatternMatch &match, PatternInfo &pattern,
                         StringRef funcName, ModuleOp module, OpBuilder &builder) {
    llvm::errs() << "\n[FunctionExtractor] Extracting match to function: " << funcName << "\n";

    // Get the pattern function to determine inputs/outputs
    toy::FuncOp patternFunc = nullptr;
    pattern.patternModule->walk([&](toy::FuncOp func) {
      if (func.getName() == "pattern") {
        patternFunc = func;
      }
    });

    if (!patternFunc) {
      llvm::errs() << "[FunctionExtractor] ERROR: Pattern function not found\n";
      return;
    }

    // Compute inputs: values used but not produced by matched ops
    SmallVector<Value> inputs;
    llvm::SmallPtrSet<Value, 8> producedValues;

    for (auto *op : match.matchedOps) {
      for (auto result : op->getResults()) {
        producedValues.insert(result);
      }
    }

    llvm::SmallPtrSet<Value, 8> inputSet;
    for (auto *op : match.matchedOps) {
      for (auto operand : op->getOperands()) {
        if (!producedValues.contains(operand) && !inputSet.contains(operand)) {
          inputs.push_back(operand);
          inputSet.insert(operand);
        }
      }
    }

    llvm::errs() << "[FunctionExtractor] Found " << inputs.size() << " inputs\n";

    // Compute outputs: results of matched ops used outside the match
    SmallVector<Value> outputs;
    for (auto *op : match.matchedOps) {
      for (auto result : op->getResults()) {
        for (auto user : result.getUsers()) {
          if (std::find(match.matchedOps.begin(), match.matchedOps.end(), user) ==
              match.matchedOps.end()) {
            // Used outside the matched region
            outputs.push_back(result);
            break;
          }
        }
      }
    }

    llvm::errs() << "[FunctionExtractor] Found " << outputs.size() << " outputs\n";

    // Create function type
    SmallVector<Type> inputTypes;
    for (auto input : inputs) {
      inputTypes.push_back(input.getType());
    }

    SmallVector<Type> outputTypes;
    for (auto output : outputs) {
      outputTypes.push_back(output.getType());
    }

    auto funcType = builder.getFunctionType(inputTypes, outputTypes);

    // Create the function at module level
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(module.getBody());

    auto func = builder.create<toy::FuncOp>(match.loc, funcName, funcType);
    func.setPrivate();

    llvm::errs() << "[FunctionExtractor] Created function with " << inputTypes.size()
                 << " inputs and " << outputTypes.size() << " outputs\n";

    // Get or create function body
    Block *funcBody;
    if (func.getBody().empty()) {
      funcBody = func.addEntryBlock();
    } else {
      funcBody = &func.getBody().front();
    }
    builder.setInsertionPointToStart(funcBody);

    // Clone matched operations into function
    IRMapping mapper;
    for (auto [input, arg] : llvm::zip(inputs, funcBody->getArguments())) {
      mapper.map(input, arg);
    }

    llvm::errs() << "[FunctionExtractor] Cloning " << match.matchedOps.size()
                 << " operations into function body\n";

    for (auto *op : match.matchedOps) {
      builder.clone(*op, mapper);
    }

    // Add return statement
    SmallVector<Value> returnVals;
    for (auto output : outputs) {
      returnVals.push_back(mapper.lookup(output));
    }
    builder.create<toy::ReturnOp>(match.loc, returnVals);

    llvm::errs() << "[FunctionExtractor] Function created successfully\n";

    // Replace matched operations with a call
    builder.setInsertionPoint(match.matchedOps.front());

    auto call = builder.create<toy::GenericCallOp>(
        match.loc,
        outputs.empty() ? Type() : outputs[0].getType(),  // GenericCallOp returns single value
        mlir::SymbolRefAttr::get(builder.getContext(), funcName),
        inputs);

    llvm::errs() << "[FunctionExtractor] Created call operation\n";

    // Replace uses of matched op results with call result
    if (!outputs.empty()) {
      outputs[0].replaceAllUsesWith(call.getResult());
    }

    llvm::errs() << "[FunctionExtractor] Replaced " << outputs.size() << " values\n";

    // Erase matched operations
    for (auto *op : llvm::reverse(match.matchedOps)) {
      llvm::errs() << "[FunctionExtractor] Erasing " << op->getName() << "\n";
      op->erase();
    }

    llvm::errs() << "[FunctionExtractor] Extraction complete!\n";
  }

  // Inline all toy.generic_call operations within a function
  void inlineCalledFunctions(toy::FuncOp func, ModuleOp module) {
    llvm::errs() << "\n[FunctionInliner] Starting inlining for function: " << func.getName() << "\n";

    // Keep inlining until no more calls remain
    bool changed = true;
    int iteration = 0;
    while (changed) {
      changed = false;
      iteration++;
      llvm::errs() << "[FunctionInliner] Iteration " << iteration << "\n";

      // Find all generic_call operations
      SmallVector<toy::GenericCallOp> callsToInline;
      func.walk([&](toy::GenericCallOp callOp) {
        callsToInline.push_back(callOp);
      });

      llvm::errs() << "[FunctionInliner] Found " << callsToInline.size() << " calls to inline\n";

      for (auto callOp : callsToInline) {
        StringRef callee = callOp.getCallee();
        llvm::errs() << "[FunctionInliner] Inlining call to: " << callee << "\n";

        // Find the called function in the module
        auto calledFunc = module.lookupSymbol<toy::FuncOp>(callee);
        if (!calledFunc) {
          llvm::errs() << "[FunctionInliner] ✗ Function not found: " << callee << "\n";
          continue;
        }

        llvm::errs() << "[FunctionInliner] ✓ Found function to inline\n";

        // Create builder at call site
        OpBuilder builder(callOp);
        IRMapping mapping;

        // Map function arguments to call operands
        for (auto [arg, operand] : llvm::zip(
               calledFunc.getArguments(), callOp.getOperands())) {
          mapping.map(arg, operand);
          llvm::errs() << "[FunctionInliner] Mapped argument to operand\n";
        }

        // Clone all operations from function body (except return)
        Value inlinedResult;
        Block &calledBody = calledFunc.getBody().front();

        llvm::errs() << "[FunctionInliner] Cloning "
                     << calledBody.getOperations().size() << " operations\n";

        for (Operation &op : calledBody.getOperations()) {
          if (auto returnOp = dyn_cast<toy::ReturnOp>(&op)) {
            // Get the returned value (will replace call result)
            if (returnOp.getNumOperands() > 0) {
              inlinedResult = mapping.lookup(returnOp.getOperand(0));
              llvm::errs() << "[FunctionInliner] Found return value\n";
            }
          } else {
            // Clone the operation
            Operation *cloned = builder.clone(op, mapping);
            llvm::errs() << "[FunctionInliner] Cloned: " << cloned->getName() << "\n";
          }
        }

        // Replace call with inlined result
        if (inlinedResult) {
          callOp.getResult().replaceAllUsesWith(inlinedResult);
          llvm::errs() << "[FunctionInliner] Replaced call result with inlined value\n";
        }

        // Erase the call
        callOp.erase();
        llvm::errs() << "[FunctionInliner] Erased call operation\n";

        changed = true;
      }
    }

    llvm::errs() << "[FunctionInliner] Inlining complete after " << iteration << " iterations\n";
  }

  // Remove dead (unused) functions from the module
  // Keeps main function and any functions that are called
  static void removeDeadFunctions(ModuleOp module) {
    llvm::errs() << "\n[DeadFunctionElimination] Starting dead function elimination\n";

    // Collect all function calls in the module
    DenseSet<StringRef> calledFunctions;
    module.walk([&](toy::GenericCallOp callOp) {
      calledFunctions.insert(callOp.getCallee());
      llvm::errs() << "[DeadFunctionElimination] Function is called: " << callOp.getCallee() << "\n";
    });

    // Find functions to remove (private functions that are never called)
    SmallVector<toy::FuncOp> toErase;
    module.walk([&](toy::FuncOp func) {
      StringRef funcName = func.getName();

      // Keep main function
      if (funcName == "main") {
        llvm::errs() << "[DeadFunctionElimination] Keeping main function\n";
        return;
      }

      // Keep public functions
      if (!func.isPrivate()) {
        llvm::errs() << "[DeadFunctionElimination] Keeping public function: " << funcName << "\n";
        return;
      }

      // Check if function is called
      if (!calledFunctions.contains(funcName)) {
        llvm::errs() << "[DeadFunctionElimination] ✗ Marking for removal: " << funcName << " (unused)\n";
        toErase.push_back(func);
      } else {
        llvm::errs() << "[DeadFunctionElimination] ✓ Keeping function: " << funcName << " (used)\n";
      }
    });

    // Remove dead functions
    llvm::errs() << "[DeadFunctionElimination] Removing " << toErase.size() << " dead functions\n";
    for (auto func : toErase) {
      llvm::errs() << "[DeadFunctionElimination] Erasing function: " << func.getName() << "\n";
      func.erase();
    }

    llvm::errs() << "[DeadFunctionElimination] Complete!\n";
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// ToyToAffineLoweringPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to affine loops of the toy operations that are
/// computationally intensive (like matmul for example...) while keeping the
/// rest of the code in the Toy dialect.
namespace {
struct ToyToAffineLoweringPass
    : public PassWrapper<ToyToAffineLoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ToyToAffineLoweringPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, func::FuncDialect,
                    memref::MemRefDialect>();
  }
  void runOnOperation() final;
};
} // namespace

void ToyToAffineLoweringPass::runOnOperation() {
  // List of pattern files to try
  SmallVector<std::pair<std::string, std::string>> patternFiles = {
    {"transpose_mul", "patterns/transpose_mul.mlir"},
    {"transpose_dag", "patterns/transpose_dag.mlir"},
    {"double_transpose", "patterns/double_transpose.mlir"},
    {"add_mul_chain", "patterns/add_mul_chain.mlir"},
    {"transpose_mul_add", "patterns/transpose_mul_add.mlir"}
  };

  llvm::errs() << "\n========== TESTING MULTIPLE PATTERNS ==========\n";
  llvm::errs() << "Will try " << patternFiles.size() << " patterns\n\n";

  // Try each pattern
  for (auto &[patternName, patternPath] : patternFiles) {
    llvm::errs() << "\n********** Pattern: " << patternName << " **********\n";
    llvm::errs() << "File: " << patternPath << "\n";

    // ========== STEP 1: Parse pattern file ==========
    PatternInfo patternInfo;
    patternInfo.name = patternName;

    if (failed(parsePatternFile(patternPath, &getContext(), patternInfo))) {
      llvm::errs() << "[LowerToAffine] ERROR: Pattern parsing failed for "
                   << patternName << "\n";
      llvm::errs() << "**********************************************\n\n";
      continue;
    }

    llvm::errs() << "[LowerToAffine] Pattern parsing succeeded!\n";
    patternInfo.dump();

    // ========== STEP 2: Pattern Matching ==========
    llvm::errs() << "\n--- Pattern Matching ---\n";

    if (patternInfo.patternModule) {
      PatternMatcher matcher(patternInfo);
      SmallVector<PatternMatch> matches = matcher.findMatches(getOperation());

      llvm::errs() << "\n[LowerToAffine] Found " << matches.size() << " matches for "
                   << patternName << "\n";

      if (!matches.empty()) {
        for (size_t i = 0; i < matches.size(); ++i) {
          llvm::errs() << "\n  Match " << i << ":\n";
          llvm::errs() << "    Location: " << matches[i].loc << "\n";
          llvm::errs() << "    Matched operations: " << matches[i].matchedOps.size() << "\n";
          for (auto *op : matches[i].matchedOps) {
            llvm::errs() << "      - " << op->getName() << "\n";
          }
        }

        // ========== Extract matches to functions ==========
        llvm::errs() << "\n--- Extracting to functions ---\n";

        FunctionExtractor extractor;
        OpBuilder builder(&getContext());

        SmallVector<toy::FuncOp> extractedFuncs;
        for (size_t i = 0; i < matches.size(); ++i) {
          std::string funcName = patternInfo.name + "_" + std::to_string(i);
          extractor.extractAndReplace(matches[i], patternInfo, funcName,
                                       getOperation(), builder);

          // Find the newly created function
          auto newFunc = getOperation().lookupSymbol<toy::FuncOp>(funcName);
          if (newFunc) {
            extractedFuncs.push_back(newFunc);
          }
        }

        // ========== Inline called functions ==========
        llvm::errs() << "\n--- Inlining called functions ---\n";
        for (auto func : extractedFuncs) {
          extractor.inlineCalledFunctions(func, getOperation());
        }
      } else {
        llvm::errs() << "[LowerToAffine] ✗ No matches found for " << patternName << "\n";
      }
    }

    llvm::errs() << "**********************************************\n\n";
  }

  llvm::errs() << "============ ALL PATTERNS TESTED ============\n\n";

  // ========== Dead Function Elimination ==========
  llvm::errs() << "========== Dead Function Elimination ==========\n";
  FunctionExtractor::removeDeadFunctions(getOperation());
  llvm::errs() << "================================================\n\n";

  // COMMENTED OUT: Old TransposeMulPattern - testing new pattern matcher instead
  // // First, apply the TransposeMulPattern as a greedy rewrite
  // // This runs before the conversion framework, so it works on legal ops
  // llvm::errs() << "[LowerToAffine] Running TransposeMulPattern first...\n";
  // RewritePatternSet patterns(&getContext());
  // patterns.add<TransposeMulPattern>(&getContext());
  // if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
  //   llvm::errs() << "[LowerToAffine] Greedy rewrite failed\n";
  // } else {
  //   llvm::errs() << "[LowerToAffine] Greedy rewrite completed\n";
  // }

  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering. In our case, we are lowering to a combination of the
  // `Affine`, `Arith`, `Func`, and `MemRef` dialects.
  target.addLegalDialect<affine::AffineDialect, BuiltinDialect,
                         arith::ArithDialect, func::FuncDialect,
                         memref::MemRefDialect>();

  // We also define the Toy dialect as Illegal so that the conversion will fail
  // if any of these operations are *not* converted. Given that we actually want
  // a partial lowering, we explicitly mark the Toy operations that don't want
  // to lower, `toy.print`, as `legal`. `toy.print` will still need its operands
  // to be updated though (as we convert from TensorType to MemRefType), so we
  // only treat it as `legal` if its operands are legal.
  target.addIllegalDialect<toy::ToyDialect>();
  target.addDynamicallyLegalOp<toy::PrintOp>([](toy::PrintOp op) {
    return llvm::none_of(op->getOperandTypes(),
                         [](Type type) { return llvm::isa<TensorType>(type); });
  });

  // Mark operations as legal to test TransposeMulPattern in isolation
  target.addLegalOp<toy::AddOp, toy::ConstantOp, toy::TransposeOp, toy::MulOp, toy::PrintOp>();
  target.addLegalOp<toy::FuncOp, toy::ReturnOp>(); // Keep toy functions and returns as-is
  target.addLegalOp<toy::GenericCallOp>(); // Allow the generated call

  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the Toy operations.
  RewritePatternSet conversionPatterns(&getContext());
  // Only keep PrintOpLowering to update operands from tensor to memref
  // All other toy ops are marked legal for testing TransposeMulPattern in isolation
  // Note: TransposeMulPattern is applied separately as a greedy rewrite above
  conversionPatterns.add<PrintOpLowering>(&getContext());

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(conversionPatterns))))
    signalPassFailure();
}

/// Create a pass for lowering operations in the `Affine` and `Std` dialects,
/// for a subset of the Toy IR (e.g. matmul).
std::unique_ptr<Pass> mlir::toy::createLowerToAffinePass() {
  return std::make_unique<ToyToAffineLoweringPass>();
}
