# MLIR Pattern Rewriter Plan - Analysis & Recommendations

## Overview

This document provides a detailed analysis of the MLIR pattern rewriter plan in `mlir_pattern_rewriter_plan.md`, identifying strengths, critical issues, and providing recommendations for implementation.

---

## ✅ Strengths of Your Plan

1. **Good architectural separation**: Parse → Match → Rewrite is the right flow
2. **Leverages MLIR infrastructure**: Using `parseSourceString` and existing APIs is smart
3. **Clear example workflow**: Shows the transformation concretely
4. **Extensible design**: YAML/JSON front-end is a good future direction

---

## ⚠️ Critical Issues & Design Decisions

### 1. **Input/Output Boundary Detection** ✅ DECIDED

**Decision:** Use a simple, explicit approach:
- **Inputs**: Any tensor in the pattern that is NOT produced by operations in the pattern (e.g., `%arg0`, `%arg1`) becomes a function argument
- **Outputs**: Use an explicit `return` operation in the pattern to mark which tensors are returned

**Pattern Format:**
```mlir
// Pattern file: transpose_mul.mlir
%0 = toy.transpose(%arg0)
%1 = toy.mul %0, %0
toy.return %1
```

**Implementation:**
```cpp
struct PatternMatch {
  SmallVector<Operation*> matchedOps;
  SmallVector<Value> inputs;   // Values used but not defined in pattern
  SmallVector<Value> outputs;  // Values in the return operation
};

void computeBoundary(Operation *patternRoot) {
  // Walk pattern to find inputs
  patternRoot->walk([&](Operation *op) {
    for (auto operand : op->getOperands()) {
      if (!isDefinedInPattern(operand)) {
        inputs.push_back(operand);  // This is an argument
      }
    }
  });

  // Find return op to get outputs
  auto returnOp = patternRoot->walk([](Operation *op) {
    return isa<ReturnOp>(op);
  });
  outputs = returnOp.getOperands();
}
```

### 2. **SSA Value Binding** ✅ DECIDED

**Decision:** Use a binding map to ensure SSA values are matched consistently.

**Challenge:**
```mlir
// Pattern:
%0 = toy.transpose(%arg0)
%1 = toy.mul %0, %0    // Same %0 used twice!

// Should match:
%t = toy.transpose(%a)
%m = toy.mul %t, %t    // ✓ Correct - same value used twice

// Should NOT match:
%t1 = toy.transpose(%a)
%t2 = toy.transpose(%a)
%m = toy.mul %t1, %t2  // ✗ Different values, even if semantically equal
```

**Implementation:** Use a binding map to track pattern value → IR value mappings:

```cpp
class PatternMatcher {
  DenseMap<Value, Value> bindings;  // pattern value -> IR value

  bool matchValue(Value patternVal, Value irVal) {
    if (bindings.count(patternVal)) {
      // This pattern value was already bound - must match the same IR value
      return bindings[patternVal] == irVal;
    }
    // First time seeing this pattern value - bind it
    bindings[patternVal] = irVal;
    return true;
  }
};
```

### 3. **Type Generalization** ✅ DECIDED

**Decision:** Ignore types completely - no type checking during pattern matching.

**Rationale:**
- Pattern matching only cares about operation structure and SSA value bindings
- Types (shapes and element types) don't matter for identifying the pattern
- This simplifies the matcher significantly

**Implementation:**
```cpp
bool matchOp(Operation *patternOp, Operation *irOp) {
  // Match operation name
  if (patternOp->getName() != irOp->getName())
    return false;

  // Match operand count
  if (patternOp->getNumOperands() != irOp->getNumOperands())
    return false;

  // Match operands (ignoring types)
  for (auto [patternOperand, irOperand] :
       llvm::zip(patternOp->getOperands(), irOp->getOperands())) {
    if (!matchValue(patternOperand, irOperand))
      return false;
  }

  // NO type checking needed!
  return true;
}
```

### 4. **DAG vs Chain Patterns** ✅ DECIDED

**Decision:** Support full DAG patterns (not just linear chains) by matching connected components.

**Example DAG Pattern:**
```mlir
// Pattern with DAG structure - %0 is used in multiple places
%0 = toy.transpose(%arg0)
%1 = toy.transpose(%arg1)
%2 = toy.mul %0, %1
%3 = toy.add %0, %2    // %0 used in both mul and add
return %3
```

**Implementation Strategy:**
1. Parse the entire pattern block (all operations between start and return)
2. Match the subgraph structure, not just a chain
3. Use the SSA binding map to ensure proper value correspondence
4. All operations in the pattern must match (as a connected component)

```cpp
// Match entire pattern block as a connected subgraph
bool matchPattern(Block *patternBlock, Operation *rootCandidate) {
  // Try matching from each operation in the pattern
  for (auto &patternOp : patternBlock->getOperations()) {
    bindings.clear();
    if (matchSubgraph(&patternOp, rootCandidate)) {
      // Found a match starting from this op
      return true;
    }
  }
  return false;
}
```

### 5. **Multiple Matches & Deduplication** ✅ DECIDED

**Decision:** Create separate functions for each match instance (`@pattern_0`, `@pattern_1`, etc.)

**Example:**
```mlir
// Original IR with pattern matching twice:
%a = toy.transpose(%x)
%b = toy.mul %a, %a    // Match 1

%c = toy.transpose(%y)
%d = toy.mul %c, %c    // Match 2
```

**After transformation:**
```mlir
// Two separate functions created
toy.func @pattern_0(%arg0: tensor<*xf64>) -> tensor<*xf64> {
  %0 = toy.transpose(%arg0) : tensor<*xf64>
  %1 = toy.mul %0, %0 : tensor<*xf64>
  return %1
}

toy.func @pattern_1(%arg0: tensor<*xf64>) -> tensor<*xf64> {
  %0 = toy.transpose(%arg0) : tensor<*xf64>
  %1 = toy.mul %0, %0 : tensor<*xf64>
  return %1
}

// Original code replaced with calls
%b = toy.generic_call @pattern_0(%x)
%d = toy.generic_call @pattern_1(%y)
```

**Implementation:**
```cpp
void extractAllMatches(ModuleOp module, StringRef patternName) {
  auto matches = findAllMatches(module, pattern);

  int counter = 0;
  for (auto &match : matches) {
    std::string funcName = patternName.str() + "_" + std::to_string(counter++);
    auto func = extractToFunction(match, funcName);
    replaceWithCall(match, func);
  }
}
```

**Note:** Future optimization could deduplicate identical functions, but starting simple is better.

---

## 🎯 Finalized Architecture

Based on the design decisions above, here's the **three-phase approach**:

### Phase 1: Pattern Definition (Simple MLIR Format)

Pattern files are plain MLIR snippets with operations and a return statement:

```mlir
// File: transpose_mul.mlir
// Pattern name is derived from filename: "transpose_mul"

%0 = toy.transpose(%arg0) : tensor<*xf64>
%1 = toy.mul %0, %0 : tensor<*xf64>
return %1
```

Key points:
- **Arguments**: Any value not produced by an operation (e.g., `%arg0`) is an input
- **Return**: The `return` statement marks outputs
- **No type checking**: Types can be anything, ignored during matching
- **SSA binding**: Same value names (e.g., `%0` used twice) must match same IR values

### Phase 2: Matching Engine

```cpp
class PatternMatcher {
public:
  // Parse pattern from MLIR file
  PatternMatcher(StringRef patternFile, MLIRContext *ctx) {
    // Read file and parse as MLIR
    auto fileOrErr = MemoryBuffer::getFile(patternFile);
    auto parsedModule = parseSourceString<ModuleOp>(
        fileOrErr.get()->getBuffer(), ctx);

    // Extract the operations (everything except return)
    patternOps = extractOpsFromModule(parsedModule);

    // Extract arguments from pattern
    patternInputs = findUnproducedValues(patternOps);

    // Extract return op to get outputs
    returnOp = findReturnOp(parsedModule);
    patternOutputs = returnOp.getOperands();
  }

  // Find all matches in module
  SmallVector<PatternMatch> findMatches(ModuleOp module) {
    SmallVector<PatternMatch> matches;

    module.walk([&](Operation *op) {
      bindings.clear();
      if (matchSubgraph(patternOps, op)) {
        matches.push_back(createMatch());
      }
    });

    return matches;
  }

private:
  SmallVector<Operation*> patternOps;
  SmallVector<Value> patternInputs;
  SmallVector<Value> patternOutputs;
  DenseMap<Value, Value> bindings;  // Pattern value -> IR value

  bool matchOp(Operation *patternOp, Operation *irOp);
  bool matchValue(Value patternVal, Value irVal);
  bool matchSubgraph(ArrayRef<Operation*> pattern, Operation *root);
};
```

### Phase 3: Function Extraction

```cpp
class FunctionExtractor {
public:
  // Extract all matches and create separate functions
  void extractAllMatches(ModuleOp module,
                         SmallVector<PatternMatch> &matches,
                         StringRef patternName) {
    int counter = 0;
    for (auto &match : matches) {
      std::string funcName = patternName.str() + "_" +
                            std::to_string(counter++);

      auto func = extractToFunction(match, funcName);
      replaceWithCall(match, func);
    }
  }

  // Extract single matched pattern to a toy.func
  toy::FuncOp extractToFunction(PatternMatch &match, StringRef name) {
    OpBuilder builder(match.moduleOp);

    // Create function type from inputs/outputs
    auto funcType = builder.getFunctionType(
        match.inputs.getTypes(),
        match.outputs.getTypes());

    // Create toy.func at module level
    auto func = builder.create<toy::FuncOp>(
        match.loc, name, funcType);
    func.setPrivate();

    // Add entry block
    Block *entry = func.addEntryBlock();
    builder.setInsertionPointToStart(entry);

    // Clone matched operations into function body
    BlockAndValueMapping mapper;
    for (auto [input, arg] : llvm::zip(match.inputs, entry->getArguments())) {
      mapper.map(input, arg);
    }

    for (auto *op : match.matchedOps) {
      builder.clone(*op, mapper);
    }

    // Add return
    SmallVector<Value> returnVals;
    for (auto output : match.outputs) {
      returnVals.push_back(mapper.lookup(output));
    }
    builder.create<toy::ReturnOp>(match.loc, returnVals);

    return func;
  }

  // Replace matched operations with call
  void replaceWithCall(PatternMatch &match, toy::FuncOp func) {
    OpBuilder builder(match.matchedOps.front());

    auto call = builder.create<toy::GenericCallOp>(
        match.loc, func.getName(), match.inputs);

    // Replace all output uses with call results
    for (auto [output, callResult] :
         llvm::zip(match.outputs, call.getResults())) {
      output.replaceAllUsesWith(callResult);
    }

    // Erase matched operations
    for (auto *op : llvm::reverse(match.matchedOps)) {
      op->erase();
    }
  }
};
```

---

## 📝 Implementation Roadmap

Based on the finalized design decisions, here's the step-by-step implementation plan:

### Suggested Incremental Approach

#### Step 1: Basic Infrastructure (Week 1)
- Create pass skeleton with command-line option for pattern file
- Implement pattern file parser (read MLIR snippet from file)
- Parse pattern into operations list
- Extract inputs (unproduced values) and outputs (return operands)
- **Test**: Can parse `transpose_mul.mlir` pattern file

#### Step 2: Exact Matching (Week 2)
- Implement operation name matching (ignore types)
- Implement SSA value binding with `DenseMap<Value, Value>`
- Test with simple 2-op linear pattern (transpose → mul)
- **Test**: Can match `%0 = transpose(%x); %1 = mul %0, %0` in IR

#### Step 3: DAG Pattern Matching (Week 3)
- Extend matcher to handle DAG patterns (values used multiple times)
- Implement connected component matching
- Handle patterns with multiple roots
- **Test**: Match patterns where same value used in multiple operations

#### Step 4: Function Extraction (Week 4)
- Implement `extractToFunction()` - clone ops to new `toy.func`
- Create function signature from matched inputs/outputs
- Use `BlockAndValueMapping` for value remapping
- **Test**: Can create valid `toy.func` from matched pattern

#### Step 5: Pattern Replacement (Week 5)
- Implement `replaceWithCall()` - create `toy.generic_call`
- Replace matched ops with function call
- Properly handle value uses and erasure
- **Test**: IR transformation produces valid, equivalent code

#### Step 6: Multi-Match Support (Week 6)
- Find all pattern occurrences in module
- Create separate function for each match (`@pattern_0`, `@pattern_1`, ...)
- Handle non-overlapping matches
- **Test**: Multiple pattern instances → multiple functions

#### Step 7: Polish & Testing (Week 7+)
- Add error handling (malformed patterns, parse errors)
- Add debug output (show matches found, functions created)
- Test on complex examples
- Integration with existing lowering passes

---

## 🔍 Existing MLIR Tools to Consider

Before building from scratch, check these:

### 1. **PDL (Pattern Descriptor Language)**
MLIR's built-in pattern language that can express complex patterns declaratively.

- **Location**: `mlir/include/mlir/Dialect/PDL`
- **Capabilities**: Define patterns in a declarative way, compile to C++
- **Pros**: Already integrated into MLIR, well-tested
- **Cons**: Learning curve, may be overkill for simple cases

**Example PDL Pattern:**
```mlir
pdl.pattern @transpose_mul : benefit(1) {
  %type = pdl.type : tensor<*xf64>
  %transpose = pdl.operation "toy.transpose"(%arg0 : !pdl.value) -> (%type : !pdl.type)
  %result = pdl.result 0 of %transpose
  %mul = pdl.operation "toy.mul"(%result, %result : !pdl.value, !pdl.value)
  pdl.rewrite %mul with "extractToFunction"(%mul : !pdl.operation)
}
```

### 2. **Existing Pattern Matchers in MLIR**

Study these for reference:
- `mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp`
- `mlir/include/mlir/IR/PatternMatch.h`
- Dialect-specific pattern files (e.g., `Affine/Transforms/AffinePatterns.td`)

### 3. **SubgraphMatcher**

Check if MLIR has subgraph matching utilities:
- Search codebase for "subgraph" or "graph matching"
- May exist in analysis or transform utilities

---

## 🚦 Decision Points

### Option A: Build Custom Matcher
**Pros:**
- Full control over matching logic
- Can optimize for your specific use case
- Easier to debug and understand

**Cons:**
- More implementation work
- May miss edge cases
- Reinventing the wheel

### Option B: Use PDL
**Pros:**
- Standard MLIR approach
- Well-tested, handles edge cases
- Documentation and examples available

**Cons:**
- Steeper learning curve
- May be harder to express some patterns
- Less flexible for custom logic

### Recommendation: Hybrid Approach
1. **Start with custom matcher** for learning and prototyping
2. **Migrate to PDL** once you understand requirements
3. **Use PDL for pattern matching**, custom code for function extraction

---

## 🎓 Learning Resources

1. **MLIR PDL Documentation**:
   - https://mlir.llvm.org/docs/Dialects/PDLOps/
   - https://mlir.llvm.org/docs/DeclarativeRewrites/

2. **Pattern Rewriting Guide**:
   - https://mlir.llvm.org/docs/PatternRewriter/

3. **Example Passes**:
   - Study `mlir/lib/Dialect/Affine/Transforms/LoopFusion.cpp`
   - Study `mlir/lib/Dialect/Linalg/Transforms/ElementwiseOpFusion.cpp`

4. **MLIR Tutorials**:
   - LLVM/MLIR documentation on pattern matching
   - MLIR developer meetings talks on PDL

---

## 💡 Concrete Next Steps

### Option 1: Simple Prototype
I can help you implement a basic version with:
- Exact matching for linear chains
- Manual input/output specification
- Single pattern support
- Test with your Toy dialect

### Option 2: PDL Exploration
I can help you:
- Write PDL patterns for your use case
- Create custom PDL rewrite methods
- Integrate with your existing pass

### Option 3: Hybrid Proof-of-Concept
I can help you:
- Build simple matcher for one pattern
- Show how to extract to function
- Demonstrate on your `transpose_mul` example
- Identify what features you actually need

---

## Design Questions - ANSWERED ✅

1. **Pattern Complexity**: Support both linear chains AND complex DAGs
2. **Type Flexibility**: Ignore types completely - no type checking
3. **Multiple Matches**: Create separate functions for each match (`@pattern_0`, `@pattern_1`, ...)
4. **Pattern Count**: Support arbitrary number of pattern files
5. **Performance**: Offline optimization (not JIT)
6. **Integration**: Integrated as an MLIR pass (can be run standalone via mlir-opt)

---

## Summary

### ✅ Finalized Design Decisions

1. **Input/Output Detection**: Simple rule-based (unproduced values are inputs, return marks outputs)
2. **SSA Value Binding**: Use `DenseMap<Value, Value>` to ensure consistency
3. **Type Matching**: Ignore types completely - match only on operation structure
4. **Pattern Structure**: Support full DAG patterns (connected components)
5. **Multiple Matches**: Create separate functions for each match instance

### 🎯 Implementation Strategy

**Start simple and iterate:**
1. Begin with 2-op linear patterns (transpose → mul)
2. Add SSA binding map for correctness
3. Extend to DAG patterns
4. Implement function extraction
5. Support multiple matches
6. Polish and test

**Key Simplifications:**
- No type checking (major simplification!)
- Pattern files are plain MLIR (no DSL to design)
- Explicit return for outputs (no heuristics needed)
- Separate functions per match (no deduplication complexity initially)

### 🚀 Ready to Implement

The plan is now concrete and actionable. The design is simplified but complete:

**Core Components:**
1. Pattern parser (parse MLIR file)
2. Pattern matcher (recursive matching with SSA bindings)
3. Function extractor (clone ops + create toy.func)
4. Pass infrastructure (find matches, extract, replace)

**Next Step:** Start with Step 1 (Basic Infrastructure) and implement the pattern file parser.

Would you like help implementing any specific component?
