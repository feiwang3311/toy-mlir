# toy-mlir

This is an excerpt of the toy-language tutorial MLIR within LLVM. This
repository (and commits, history) is alternate _view_ of the toy language
tutorial.  I created this to have a smaller consolidated source and to follow
what exactly is happening in source between chapters (by means of a _diff_
view).

LLVM organizes this as different folders:
[ch1](https://github.com/llvm/llvm-project/tree/main/mlir/examples/toy/ch1),
[ch2](https://github.com/llvm/llvm-project/tree/main/mlir/examples/toy/ch2)...

The following provides diffs corresponding to what's changing between chapters
as opposed to the folder view, and for me is easier to follow and figure out.

1. Initial setup for standalone builds + Chapter 1 [847d15...716325](https://github.com/jerinphilip/toy-mlir/compare/847d15...716325)
2. [Chapter 2: Emitting Basic MLIR](https://github.com/jerinphilip/toy-mlir/commit/1ea795a8741ea63b901152b4c5d40011aabf9420)
3. [Chapter 3: High-level Language-Specific Analysis and Transformation](https://github.com/jerinphilip/toy-mlir/commit/bbc7bc9b063669728ba26f343c16b6878ca0d35d)
4. [Chapter 4: Enabling Generic Transformation with Interfaces](https://github.com/jerinphilip/toy-mlir/commit/ac792399cc427e48cba4601b4bce83e87c12fff3)
5. [Chapter 5: Partial Lowering to Lower-Level Dialects for Optimization](https://github.com/jerinphilip/toy-mlir/commit/c9cdd55b60c29f1be0a8f52eaaac1fe69c933c18)
6. [Chapter 6: Lowering to LLVM and CodeGeneration](https://github.com/jerinphilip/toy-mlir/commit/58c24b26489784c960c7085946ebc801aa20bb17)
7. [Chapter 7: Adding a Composite Type to Toy](https://github.com/jerinphilip/toy-mlir/commit/a08fe58ada8833baeffa107f6b84efead03a3050)

Please be warned there could be errors, and this can go out-of-date in the
future, the following pin marks an LLVM commit where this worked:

* [llvm-project#555a71b](https://github.com/llvm/llvm-project/commit/555a71be457f351411b89c6a6a66aeecf7ca5291)


### Build instructions

I build standalone locally using:

```bash
# Clone LLVM (for building MLIR)
LLVM_SOURCE_DIR=$HOME/code/llvm-project
git clone https://github.com/llvm/llvm-project $LLVM_SOURCE_DIR

# Move to a known working commit (?)
git -C $LLVM_SOURCE_DIR checkout 555a71b

# Clone this repository, and switch to source-root
git clone https://github.com/jerinphilip/toy-mlir
cd toy-mlir

# Build MLIR to $LLVM_SOURCE_DIR/build-mlir
bash scripts/build-llvm-mlir.sh

# Build ToyC.
bash scripts/standalone-build.sh
```

The shell-scripts above assume the LLVM source is at `$HOME/code/llvm-project`.
You may look within the scripts and adapt for your use-case accordingly.

---

# MLIR Pattern Rewriter with Function Composition

A general-purpose pattern matching and rewriting system for MLIR that extracts patterns into functions and supports composable patterns with automatic inlining.

## Overview

This implementation adds a flexible pattern rewriter to the Toy MLIR dialect that can:
- Match user-defined patterns in MLIR IR
- Extract matched code into reusable functions
- Compose patterns by matching on extracted functions
- Automatically inline nested function calls
- Clean up unused functions

## Architecture: Three Main Components

### PART 1: Pattern Loading
**Location:** `toy/LowerToAffineLoops.cpp` (Lines 421-510)

Parses pattern files (MLIR code snippets) and extracts structural information.

**Key Features:**
- Patterns are defined as `toy.func @pattern(...)` in `.mlir` files
- Parser extracts: operation names, number of inputs/outputs
- Stores parsed patterns in `PatternInfo` structures for matching

**Example Pattern File (`patterns/transpose_mul.mlir`):**
```mlir
toy.func @pattern(%arg0: tensor<2x3xf64>) -> tensor<3x2xf64> {
  %0 = toy.transpose(%arg0 : tensor<2x3xf64>) to tensor<3x2xf64>
  %1 = toy.mul %0, %0 : tensor<3x2xf64>
  toy.return %1 : tensor<3x2xf64>
}
```

### PART 2: Pattern Matching
**Location:** `toy/LowerToAffineLoops.cpp` (Lines 512-722)

Implements SSA value binding algorithm to find pattern matches in IR.

**📖 For a detailed deep-dive explanation of this part, see [PATTERN_MATCHING_EXPLAINED.md](PATTERN_MATCHING_EXPLAINED.md)**

**Key Features:**
- **SSA Value Binding:** Pattern values consistently map to IR values
- **DAG Pattern Support:** Handles values used multiple times (not just linear chains)
- **Call Prefix Matching:** Pattern `@foo` matches `@foo_0`, `@foo_1`, etc.
- **Backtracking Search:** Finds all non-overlapping matches

**Matching Algorithm:**
1. Walk all operations in the module
2. For each operation, try to match the first pattern operation
3. Use data-flow based search to match remaining operations
4. Bind SSA values consistently (same pattern value → same IR value)
5. Backtrack if constraints violated

### PART 3: Function Transformations
**Location:** `toy/LowerToAffineLoops.cpp` (Lines 724-914)

Transforms matched patterns into functions and performs optimizations.

**Key Features:**
1. **Function Extraction:** Create `toy.func` from matched operations
2. **Call Generation:** Replace matches with `toy.generic_call`
3. **Function Inlining:** Inline called functions to avoid nesting
4. **Dead Code Elimination:** Remove unused functions

**Transformation Pipeline:**
```
Input IR              Pattern Match         Extract to Function      Inline & Cleanup
─────────            ──────────────        ───────────────────      ────────────────
%0 = transpose(x)    [Found match!]   →   @foo_0(%x) {        →    @foo_0(%x) {
%1 = mul(%0, %0)                             ...                       (inlined body)
                                           }                          }
                                           %2 = call @foo_0(x)
```

## Usage

### Building
```bash
cmake --build build --target toyc
```

### Running Pattern Matching
```bash
./build/toy/toyc <input.mlir> -emit=mlir-affine
```

### Adding New Patterns

1. Create a pattern file in `patterns/`:
```mlir
// patterns/my_pattern.mlir
toy.func @pattern(%arg0: tensor<...>) -> tensor<...> {
  // Define your pattern operations here
  toy.return ...
}
```

2. Add pattern to the pass in `toy/LowerToAffineLoops.cpp`:
```cpp
SmallVector<std::pair<std::string, std::string>> patternFiles = {
  // ... existing patterns ...
  {"my_pattern", "patterns/my_pattern.mlir"}
};
```

3. Rebuild and run!

## Pattern Composition Example

**Step 1:** Define base pattern (`transpose_mul.mlir`)
```mlir
toy.func @pattern(%arg0: tensor<2x3xf64>) -> tensor<3x2xf64> {
  %0 = toy.transpose(%arg0)
  %1 = toy.mul %0, %0
  toy.return %1
}
```

**Step 2:** First pass extracts base pattern
```mlir
// Before
%0 = toy.transpose(%x)
%1 = toy.mul %0, %0

// After
%1 = toy.generic_call @transpose_mul_0(%x)
```

**Step 3:** Define composition pattern (`transpose_mul_add.mlir`)
```mlir
toy.func @pattern(%arg0: tensor<2x3xf64>, %arg1: tensor<3x2xf64>) -> tensor<3x2xf64> {
  %0 = toy.generic_call @transpose_mul(%arg0)  // Matches @transpose_mul_0
  %1 = toy.add %0, %arg1
  toy.return %1
}
```

**Step 4:** Second pass composes and inlines
```mlir
// Before
%1 = toy.generic_call @transpose_mul_0(%x)
%2 = toy.add %1, %y

// After (composed and inlined)
@transpose_mul_add_0(%x, %y) {
  %0 = toy.transpose(%x)      // Inlined from transpose_mul_0
  %1 = toy.mul %0, %0         // Inlined from transpose_mul_0
  %2 = toy.add %1, %y         // From composition pattern
  toy.return %2
}
// transpose_mul_0 removed (dead function elimination)
```

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Ignore types during matching** | Simplifies matcher, types verified by MLIR |
| **Support DAG patterns** | Real code has values used multiple times |
| **Prefix matching for calls** | Enable composable patterns (@foo matches @foo_N) |
| **Automatic inlining** | Avoid nested calls, produce flat functions |
| **Dead function elimination** | Keep IR clean after composition |

## Files Modified/Added

### Modified
- `toy/LowerToAffineLoops.cpp` - Main implementation (3 parts above)

### Added Pattern Files
- `patterns/transpose_mul.mlir` - Base: transpose + mul
- `patterns/transpose_dag.mlir` - DAG: transpose → add, mul
- `patterns/double_transpose.mlir` - DAG: 2 transposes → mul
- `patterns/add_mul_chain.mlir` - DAG: add → 2 muls
- `patterns/transpose_mul_add.mlir` - Composition: transpose_mul + add

### Added Test Files
- `examples/dag_test*.mlir` - DAG pattern tests
- `examples/composition_test1.mlir` - Composition + inlining test
- `examples/multi_match_test.mlir` - Multiple matches test
- `examples/negative_test*.mlir` - Precision tests (should NOT match)

---

# Testing Guide

## Test Files Overview

### Pattern Files (in `patterns/`)
1. **transpose_mul.mlir** - Base pattern: transpose followed by mul with same operand
2. **transpose_dag.mlir** - DAG pattern: transpose result used in both add and mul
3. **double_transpose.mlir** - DAG pattern: two transposes multiplied together
4. **add_mul_chain.mlir** - DAG pattern: add result used in two different muls
5. **transpose_mul_add.mlir** - Composition pattern: call to transpose_mul followed by add

### Positive Test Files (in `examples/`)
Tests that **should match** and extract functions

### Negative Test Files (in `examples/`)
Tests that **should NOT match** (pattern matcher precision tests)

---

## Positive Tests

### 1. Basic Linear Pattern Test
**Pattern:** transpose_mul
**Test File:** `examples/affine-lowering.mlir`
**Command:**
```bash
./build/toy/toyc examples/affine-lowering.mlir -emit=mlir-affine
```
**Expected:**
- Pattern 'transpose_mul': Found 1 match(es)
- Creates function `@transpose_mul_0` containing transpose + mul
- Original transpose and mul operations replaced with call to `@transpose_mul_0`

---

### 2. DAG Pattern Test - Transpose with Multiple Uses
**Pattern:** transpose_dag
**Test File:** `examples/dag_test1.mlir`
**Command:**
```bash
./build/toy/toyc examples/dag_test1.mlir -emit=mlir-affine
```
**Expected:**
- Pattern 'transpose_dag': Found 1 match(es)
- Creates function `@transpose_dag_0` containing transpose, add, mul
- Transpose result correctly used in both add and mul operations

**Test File Content:**
```mlir
// transpose result used in both add and mul
%2 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
%3 = toy.add %2, %1 : tensor<3x2xf64>
%4 = toy.mul %2, %3 : tensor<3x2xf64>
```

---

### 3. DAG Pattern Test - Double Transpose
**Pattern:** double_transpose
**Test File:** `examples/dag_test2.mlir`
**Command:**
```bash
./build/toy/toyc examples/dag_test2.mlir -emit=mlir-affine
```
**Expected:**
- Pattern 'double_transpose': Found 1 match(es)
- Creates function `@double_transpose_0` containing 2 transposes + 1 mul
- Both transpose results used as operands to mul

**Test File Content:**
```mlir
// Two transposes multiplied together (diamond pattern)
%2 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
%3 = toy.transpose(%1 : tensor<2x3xf64>) to tensor<3x2xf64>
%4 = toy.mul %2, %3 : tensor<3x2xf64>
```

---

### 4. DAG Pattern Test - Add-Mul Chain
**Pattern:** add_mul_chain
**Test File:** `examples/dag_test3.mlir`
**Command:**
```bash
./build/toy/toyc examples/dag_test3.mlir -emit=mlir-affine
```
**Expected:**
- Pattern 'add_mul_chain': Found 1 match(es)
- Creates function `@add_mul_chain_0` containing add, 2 muls
- Add result used in both mul operations

**Test File Content:**
```mlir
// Add result used in two different muls
%3 = toy.add %0, %1 : tensor<3x2xf64>
%4 = toy.mul %3, %2 : tensor<3x2xf64>
%5 = toy.mul %3, %4 : tensor<3x2xf64>
```

---

### 5. Composition Pattern Test with Inlining
**Pattern:** transpose_mul_add
**Test File:** `examples/composition_test1.mlir`
**Command:**
```bash
./build/toy/toyc examples/composition_test1.mlir -emit=mlir-affine
```
**Expected:**
- Pattern 'transpose_mul_add': Found 1 match(es)
- Creates function `@transpose_mul_add_0`
- **Inlines** `@transpose_mul_0` body into `@transpose_mul_add_0`
- Final `@transpose_mul_add_0` contains: transpose, mul, add (no nested calls)
- **Dead function elimination**: `@transpose_mul_0` is removed (unused after inlining)
- Final IR contains only `@main` and `@transpose_mul_add_0`

**Test File Content:**
```mlir
// Call to extracted function followed by add
%2 = toy.generic_call @transpose_mul_0(%0) : (tensor<2x3xf64>) -> tensor<3x2xf64>
%3 = toy.add %2, %1 : tensor<3x2xf64>
```

**Key Features Tested:**
- ✓ Prefix matching: pattern `@transpose_mul` matches IR call `@transpose_mul_0`
- ✓ Function inlining: nested call is inlined
- ✓ Dead function elimination: unused functions removed

---

### 6. Multi-Match Test
**Pattern:** transpose_mul
**Test File:** `examples/multi_match_test.mlir`
**Command:**
```bash
./build/toy/toyc examples/multi_match_test.mlir -emit=mlir-affine
```
**Expected:**
- Pattern 'transpose_mul': Found 3 match(es)
- Creates 3 functions: `@transpose_mul_0`, `@transpose_mul_1`, `@transpose_mul_2`
- Each instance of the pattern is extracted to a separate function
- All 3 functions appear in final IR

**Test File Content:**
```mlir
// Three instances of transpose-mul pattern
// Instance 1: %2 = transpose(%0); %3 = mul(%2, %2)
// Instance 2: %4 = transpose(%1); %5 = mul(%4, %4)
// Instance 3: %6 = transpose(%3); %7 = mul(%6, %6)
```

---

## Negative Tests

These tests verify the pattern matcher is **precise** and does **not** produce false positives.

### 7. Negative Test - Different Operands
**Pattern:** transpose_mul (expects mul with same operands)
**Test File:** `examples/negative_test1.mlir`
**Command:**
```bash
./build/toy/toyc examples/negative_test1.mlir -emit=mlir-affine
```
**Expected:**
- Pattern 'transpose_mul': Found 0 match(es)
- No functions created
- IR unchanged (only constants and print)

**Test File Content:**
```mlir
// transpose followed by mul, but mul uses DIFFERENT operands
%2 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
%3 = toy.mul %2, %1 : tensor<3x2xf64>  // %1 is different from %2!
```

**Why it shouldn't match:** Pattern expects `mul(%t, %t)` but IR has `mul(%t, %other)`

---

### 8. Negative Test - Partial Pattern
**Pattern:** transpose_mul (expects transpose + mul)
**Test File:** `examples/negative_test2.mlir`
**Command:**
```bash
./build/toy/toyc examples/negative_test2.mlir -emit=mlir-affine
```
**Expected:**
- Pattern 'transpose_mul': Found 0 match(es)
- No functions created
- IR unchanged

**Test File Content:**
```mlir
// Only transpose, no mul follows
%1 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
```

**Why it shouldn't match:** Pattern has 2 ops (transpose + mul) but IR only has 1 op

---

### 9. Negative Test - Wrong Operation Type
**Pattern:** transpose_mul (expects mul after transpose)
**Test File:** `examples/negative_test3.mlir`
**Command:**
```bash
./build/toy/toyc examples/negative_test3.mlir -emit=mlir-affine
```
**Expected:**
- Pattern 'transpose_mul': Found 0 match(es)
- No functions created
- IR unchanged

**Test File Content:**
```mlir
// transpose followed by ADD instead of MUL
%1 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
%2 = toy.add %1, %1 : tensor<3x2xf64>  // ADD not MUL!
```

**Why it shouldn't match:** Pattern expects `toy.mul` but IR has `toy.add`

---

### 10. Negative Test - Wrong Data Flow
**Pattern:** transpose_mul (expects mul to use transpose result)
**Test File:** `examples/negative_test4.mlir`
**Command:**
```bash
./build/toy/toyc examples/negative_test4.mlir -emit=mlir-affine
```
**Expected:**
- Pattern 'transpose_mul': Found 0 match(es)
- No functions created
- IR unchanged

**Test File Content:**
```mlir
// transpose and mul both exist, but mul doesn't use transpose result
%2 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
%3 = toy.mul %1, %1 : tensor<3x2xf64>  // Uses %1, not %2!
```

**Why it shouldn't match:** Pattern expects mul to use transpose's result, but data flow is disconnected

---

## Quick Test Suite

Run all tests to verify full functionality:

```bash
# Positive tests
echo "=== Positive Tests ==="
./build/toy/toyc examples/affine-lowering.mlir -emit=mlir-affine 2>&1 | grep "Found.*match"
./build/toy/toyc examples/dag_test1.mlir -emit=mlir-affine 2>&1 | grep "Found.*match"
./build/toy/toyc examples/dag_test2.mlir -emit=mlir-affine 2>&1 | grep "Found.*match"
./build/toy/toyc examples/dag_test3.mlir -emit=mlir-affine 2>&1 | grep "Found.*match"
./build/toy/toyc examples/composition_test1.mlir -emit=mlir-affine 2>&1 | grep "Found.*match"
./build/toy/toyc examples/multi_match_test.mlir -emit=mlir-affine 2>&1 | grep "Found.*match"

# Negative tests (all should show 0 matches)
echo ""
echo "=== Negative Tests (should all be 0) ==="
./build/toy/toyc examples/negative_test1.mlir -emit=mlir-affine 2>&1 | grep "transpose_mul.*Found.*match"
./build/toy/toyc examples/negative_test2.mlir -emit=mlir-affine 2>&1 | grep "transpose_mul.*Found.*match"
./build/toy/toyc examples/negative_test3.mlir -emit=mlir-affine 2>&1 | grep "transpose_mul.*Found.*match"
./build/toy/toyc examples/negative_test4.mlir -emit=mlir-affine 2>&1 | grep "transpose_mul.*Found.*match"
```

---

## Summary of Pattern Matcher Features Tested

| Feature | Test File(s) |
|---------|-------------|
| ✓ Linear pattern matching | affine-lowering.mlir |
| ✓ DAG pattern matching | dag_test1.mlir, dag_test2.mlir, dag_test3.mlir |
| ✓ SSA value binding | All positive tests |
| ✓ Multiple pattern support | All tests (5 patterns tried) |
| ✓ Multiple matches per pattern | multi_match_test.mlir |
| ✓ Call prefix matching (@foo_0 matches @foo) | composition_test1.mlir |
| ✓ Function inlining | composition_test1.mlir |
| ✓ Dead function elimination | composition_test1.mlir |
| ✓ Precise matching (no false positives) | negative_test*.mlir |
| ✓ Operation name matching | negative_test3.mlir |
| ✓ Operand binding consistency | negative_test1.mlir |
| ✓ Data flow verification | negative_test4.mlir |
| ✓ Complete pattern matching | negative_test2.mlir |

---

## Expected Output Format

Clean, concise output per pattern:
```
Pattern 'pattern_name': N ops, N inputs, N outputs
  Found N match(es)
```

Example successful run (composition_test1.mlir):
```
Pattern 'transpose_mul': 2 ops, 1 inputs, 1 outputs
  Found 0 match(es)
Pattern 'transpose_dag': 3 ops, 2 inputs, 1 outputs
  Found 0 match(es)
Pattern 'double_transpose': 3 ops, 2 inputs, 1 outputs
  Found 0 match(es)
Pattern 'add_mul_chain': 3 ops, 3 inputs, 1 outputs
  Found 0 match(es)
Pattern 'transpose_mul_add': 2 ops, 2 inputs, 1 outputs
  Found 1 match(es)

[Final IR shows @main and @transpose_mul_add_0, with @transpose_mul_0 removed]
```

---

### License

Modifications and additions in this repo are licensed [MIT](./LICENSE). The original
toy-mlir example code is a derivative of LLVM code, which is under [Apache with
LLVM exceptions](./LICENSE.apache-llvm-exceptions.txt).
