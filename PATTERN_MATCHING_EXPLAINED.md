# Pattern Matching Deep Dive: SSA Value Binding and DAG Matching

This document provides a detailed, instructive explanation of **Part 2: Pattern Matching** from the MLIR pattern rewriter implementation. This is the most complex part of the code, implementing SSA value binding and DAG (Directed Acyclic Graph) pattern matching.

---

## Table of Contents
1. [The Challenge](#the-challenge)
2. [Key Data Structures](#key-data-structures)
3. [The Matching Algorithm](#the-matching-algorithm)
4. [SSA Value Binding](#ssa-value-binding)
5. [Operation Matching](#operation-matching)
6. [DAG Pattern Matching](#dag-pattern-matching)
7. [Call Prefix Matching](#call-prefix-matching)
8. [Backtracking](#backtracking)
9. [Complete Walkthrough Example](#complete-walkthrough-example)

---

## The Challenge

Pattern matching in MLIR is challenging because:

1. **SSA Form**: Values are defined once and used multiple times. We need to ensure consistent binding.
2. **DAG Patterns**: Operations form a graph, not just a linear sequence. A value can be used by multiple operations.
3. **Operation Attributes**: Need to match not just operation types, but also operand relationships.
4. **Composability**: Patterns should match previously extracted functions (e.g., `@foo` matches `@foo_0`).

**Example Challenge:**
```mlir
// Pattern (DAG structure):
%0 = transpose(%arg0)
%1 = add %0, %arg1      // %0 used here
%2 = mul %0, %1         // %0 also used here!

// This is NOT a linear chain - it's a DAG!
```

---

## Key Data Structures

### PatternMatch
```cpp
struct PatternMatch {
  SmallVector<Operation*> matchedOps;     // Operations that matched the pattern
  DenseMap<Value, Value> valueBindings;   // Pattern value -> IR value mapping
  Location loc;                            // Source location of the match
};
```

**Purpose**: Holds a single successful match result.

**Why `valueBindings`?**
- Pattern SSA values (like `%0` in the pattern) must consistently map to IR values
- Example: If pattern `%0` binds to IR `%5`, all uses of pattern `%0` must match IR `%5`

---

### PatternMatcher Class

```cpp
class PatternMatcher {
public:
  PatternMatcher(PatternInfo &pattern);
  SmallVector<PatternMatch> findMatches(ModuleOp module);

private:
  PatternInfo &pattern;                    // The pattern to match
  SmallVector<Operation*> patternOps;      // Pattern operations (extracted from pattern function)
  DenseMap<Value, Value> bindings;         // Current SSA value bindings
  SmallVector<Operation*> matchedOps;      // Currently matched operations

  bool matchValue(Value patternVal, Value irVal);
  bool matchOp(Operation *patternOp, Operation *irOp);
  bool matchSequence(Operation *startOp, size_t patternIndex);
  bool matchRemainingOps(size_t startIndex);
};
```

**Key Design**:
- `bindings`: Mutable state tracking current SSA value bindings during matching
- `matchedOps`: Accumulates matched operations as we search
- Backtracking-based search with state management

---

## The Matching Algorithm

### High-Level Flow

```
For each operation in the module:
  1. Try to match it as the FIRST pattern operation
  2. If match succeeds:
     - Record this operation as matched
     - Recursively find remaining pattern operations in the block
     - Use data-flow dependencies to guide search (DAG matching)
  3. If all pattern operations matched:
     - Success! Record the match
  4. Otherwise:
     - Backtrack and try next operation
```

### Entry Point: `findMatches()`

```cpp
SmallVector<PatternMatch> findMatches(ModuleOp module) {
  SmallVector<PatternMatch> matches;

  // Walk all operations in the module
  module.walk([&](Operation *op) {
    // Try to match starting from this operation
    bindings.clear();        // Reset SSA bindings
    matchedOps.clear();      // Reset matched operations

    if (matchSequence(op, 0)) {  // Try to match pattern starting here
      PatternMatch match(op->getLoc());
      match.matchedOps = matchedOps;
      match.valueBindings = bindings;
      matches.push_back(match);
    }
  });

  return matches;
}
```

**Key Points**:
- Try every operation as a potential starting point
- Clear state (`bindings`, `matchedOps`) before each attempt
- Capture successful matches with their bindings

---

## SSA Value Binding

### The Core Challenge

In SSA form, a value can be used multiple times:
```mlir
%0 = toy.transpose(%arg0)
%1 = toy.add %0, %arg1    // %0 used here
%2 = toy.mul %0, %1       // %0 used here again!
```

**Question**: How do we ensure the SAME pattern value always maps to the SAME IR value?

### Solution: `matchValue()`

```cpp
bool matchValue(Value patternVal, Value irVal) {
  // Check if this pattern value is already bound
  if (bindings.count(patternVal)) {
    return bindings[patternVal] == irVal;  // Must match existing binding!
  }

  // First time seeing this pattern value - bind it
  bindings[patternVal] = irVal;
  return true;
}
```

**How it works**:

1. **First encounter**: `bindings[patternVal] = irVal` (create binding)
2. **Subsequent encounters**: Check `bindings[patternVal] == irVal` (verify consistency)

**Example**:
```cpp
// Pattern: %p0 = transpose(%arg0); %p1 = mul %p0, %p0
// IR:      %5 = transpose(%3);     %6 = mul %5, %5

// First mul operand:
matchValue(%p0, %5)  // bindings[%p0] = %5, return true

// Second mul operand:
matchValue(%p0, %5)  // bindings[%p0] == %5? YES! return true

// If IR was: %6 = mul %5, %7
matchValue(%p0, %7)  // bindings[%p0] == %7? NO! return false
```

**This ensures**: Same pattern value → Same IR value (SSA consistency)

---

## Operation Matching

### Basic Operation Matching: `matchOp()`

```cpp
bool matchOp(Operation *patternOp, Operation *irOp) {
  // 1. Match operation name (ignore types)
  if (patternOp->getName() != irOp->getName()) {
    return false;
  }

  // 2. Special handling for toy.generic_call (explained later)
  if (auto patternCall = dyn_cast<toy::GenericCallOp>(patternOp)) {
    // ... prefix matching for calls ...
  }

  // 3. Match operand count
  if (patternOp->getNumOperands() != irOp->getNumOperands()) {
    return false;
  }

  // 4. Match operands (with SSA binding)
  for (auto [patternOperand, irOperand] :
       llvm::zip(patternOp->getOperands(), irOp->getOperands())) {
    if (!matchValue(patternOperand, irOperand)) {
      return false;
    }
  }

  return true;
}
```

**Why ignore types?**
- Simplifies matching (types are verified by MLIR's type system anyway)
- Focus on structural matching: operation names + operand bindings

**Critical Step 4**: Matching operands uses `matchValue()` to enforce SSA binding consistency!

**Example**:
```mlir
// Pattern: toy.mul %p0, %p0
// IR:      toy.mul %5, %5

matchOp(pattern_mul, ir_mul):
  1. Names match? toy.mul == toy.mul ✓
  2. Operand count? 2 == 2 ✓
  3. Match operand 0: matchValue(%p0, %5) → bind %p0 to %5 ✓
  4. Match operand 1: matchValue(%p0, %5) → check %p0 == %5 ✓
  Result: MATCH!

// But if IR was: toy.mul %5, %7
  3. Match operand 0: matchValue(%p0, %5) → bind %p0 to %5 ✓
  4. Match operand 1: matchValue(%p0, %7) → check %p0 == %7? NO! ✗
  Result: NO MATCH
```

---

## DAG Pattern Matching

### The Problem with Sequential Matching

**Linear (chain) pattern**:
```mlir
%0 = op1(...)
%1 = op2(%0)    // Sequential: %1 immediately follows %0
```

**DAG pattern**:
```mlir
%0 = op1(...)
%1 = op2(%0)    // %0 used here
%2 = op3(%0)    // %0 ALSO used here (not sequential!)
```

In a DAG pattern, operations are not in strict sequential order. We need to **search the entire block** based on data dependencies.

### Solution: Two-Phase Matching

#### Phase 1: `matchSequence()` - Match First Operation

```cpp
bool matchSequence(Operation *startOp, size_t patternIndex) {
  if (patternIndex >= patternOps.size()) {
    return true;  // All pattern operations matched!
  }

  Operation *patternOp = patternOps[patternIndex];

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
  if (matchRemainingOps(patternIndex + 1)) {
    return true;
  }

  // Backtrack
  matchedOps.pop_back();
  return false;
}
```

**Key Points**:
- Matches the first pattern operation with `startOp`
- Records the result binding (critical for DAG matching!)
- Delegates to `matchRemainingOps()` for the rest

#### Phase 2: `matchRemainingOps()` - DAG Search

```cpp
bool matchRemainingOps(size_t startIndex) {
  if (startIndex >= patternOps.size()) {
    return true;  // All matched!
  }

  Operation *patternOp = patternOps[startIndex];
  Block *searchBlock = matchedOps.empty() ? nullptr : matchedOps[0]->getBlock();

  if (!searchBlock) {
    return false;
  }

  // Search the ENTIRE block for matching operation
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
        return true;  // Success!
      }

      // Backtrack
      matchedOps.pop_back();
    }
  }

  return false;  // No match found
}
```

**How DAG Matching Works**:

1. **Search entire block**: Don't assume operations are sequential
2. **Skip already matched**: Avoid matching the same operation twice
3. **Use SSA bindings to guide search**: `matchOp()` will fail if operand bindings don't match
4. **Backtrack on failure**: Try different operation orderings

**Example DAG Match**:
```mlir
// Pattern:
%p0 = transpose(%arg0)
%p1 = add %p0, %arg1
%p2 = mul %p0, %p1

// IR (operations in different order!):
%0 = constant ...
%1 = constant ...
%2 = transpose(%0)    // Match pattern op 0
%3 = mul %2, %4       // This will be matched LAST (pattern op 2)
%4 = add %2, %1       // Match pattern op 1

// Matching sequence:
1. matchSequence(%2, 0): Match transpose ✓
   - Bind %p0 → %2
2. matchRemainingOps(1): Search for add
   - Try %3 (mul): No, looking for add ✗
   - Try %4 (add): Yes! Check operands:
     - matchValue(%p0, %2): %p0 already bound to %2 ✓
     - matchValue(%arg1, %1): Bind %arg1 → %1 ✓
   - Bind %p1 → %4
3. matchRemainingOps(2): Search for mul
   - Try %3 (mul): Check operands:
     - matchValue(%p0, %2): %p0 already bound to %2 ✓
     - matchValue(%p1, %4): %p1 already bound to %4 ✓
   - SUCCESS! All matched!
```

**Why this works**:
- SSA bindings enforce data-flow correctness
- We don't care about operation order, only data dependencies
- Backtracking explores all possible matchings

---

## Call Prefix Matching

### The Composition Challenge

When patterns compose (one pattern references extracted functions), we need:
```mlir
// Pattern defines:
toy.generic_call @transpose_mul(...)

// But IR has:
toy.generic_call @transpose_mul_0(...)  // Note the _0 suffix!
```

**Challenge**: Pattern references `@transpose_mul`, but IR has `@transpose_mul_0`, `@transpose_mul_1`, etc.

### Solution: Prefix Matching for Calls

```cpp
// Inside matchOp(), special handling for toy.generic_call:
if (auto patternCall = dyn_cast<toy::GenericCallOp>(patternOp)) {
  if (auto irCall = dyn_cast<toy::GenericCallOp>(irOp)) {
    StringRef patternCallee = patternCall.getCallee();  // e.g., "transpose_mul"
    StringRef irCallee = irCall.getCallee();            // e.g., "transpose_mul_0"

    // Remove _N suffix from IR callee
    auto splitResult = irCallee.rsplit('_');
    StringRef irCalleePrefix = splitResult.first;       // "transpose_mul"
    StringRef suffix = splitResult.second;              // "0"

    // Only strip suffix if it's actually a number
    bool hasSuffix = !suffix.empty() &&
                     std::all_of(suffix.begin(), suffix.end(),
                               [](char c) { return std::isdigit(c); });

    if (hasSuffix) {
      // Compare prefix: "transpose_mul" == "transpose_mul"?
      if (irCalleePrefix != patternCallee) {
        return false;
      }
    } else {
      // No numeric suffix, require exact match
      if (irCallee != patternCallee) {
        return false;
      }
    }
    // Continue to match operands...
  }
}
```

**How it works**:

1. **Extract prefix**: Split IR callee on last `_`, get first part
2. **Check if numeric suffix**: Is the part after `_` all digits?
3. **Compare prefix**: Pattern callee must match IR callee prefix

**Examples**:
```cpp
Pattern: @transpose_mul
IR: @transpose_mul_0    → Prefix: "transpose_mul", Suffix: "0" → MATCH ✓
IR: @transpose_mul_1    → Prefix: "transpose_mul", Suffix: "1" → MATCH ✓
IR: @transpose_mul_add  → No numeric suffix → NO MATCH ✗
IR: @foo_0              → Prefix: "foo" → NO MATCH ✗
IR: @transpose_mul      → No suffix, exact match → MATCH ✓
```

**Why this matters**:
- Enables composable patterns
- Pattern `@transpose_mul + add` can match any instance `@transpose_mul_0`, `@transpose_mul_1`, etc.
- Critical for multi-level pattern composition

---

## Backtracking

### Why Backtracking?

Pattern matching is a **search problem** with multiple possible solutions. Backtracking explores all possibilities.

### Where Backtracking Happens

#### In `matchSequence()`:
```cpp
// Try to match remaining pattern operations
if (matchRemainingOps(patternIndex + 1)) {
  return true;
}

// BACKTRACK: Remove this operation from matched set
matchedOps.pop_back();
return false;
```

#### In `matchRemainingOps()`:
```cpp
for (auto &candidateOp : searchBlock->getOperations()) {
  if (matchOp(patternOp, &candidateOp)) {
    matchedOps.push_back(&candidateOp);

    if (matchRemainingOps(startIndex + 1)) {
      return true;  // Found valid match!
    }

    // BACKTRACK: This candidate didn't lead to a complete match
    matchedOps.pop_back();
  }
}
```

### Backtracking Example

```mlir
// Pattern: op1, op2, op3
// IR: opA, opB, opC, opD

// Try 1: opA matches op1
//   Try: opB matches op2? No → Backtrack
//   Try: opC matches op2? Yes
//     Try: opD matches op3? No → Backtrack
//   Try: opD matches op2? No → Backtrack
// Try 2: opB matches op1? No
// Try 3: opC matches op1? Yes
//   ... and so on
```

**Key Points**:
- `matchedOps.pop_back()` undoes the match
- **Note**: We don't explicitly undo `bindings` - we rely on `bindings.clear()` at the start of each top-level match attempt
- This works because bindings are only valid within a single match attempt

---

## Complete Walkthrough Example

Let's walk through matching the `transpose_dag` pattern:

### Pattern Definition
```mlir
// patterns/transpose_dag.mlir
toy.func @pattern(%arg0: tensor<2x3xf64>, %arg1: tensor<3x2xf64>) -> tensor<3x2xf64> {
  %0 = toy.transpose(%arg0 : tensor<2x3xf64>) to tensor<3x2xf64>
  %1 = toy.add %0, %arg1 : tensor<3x2xf64>
  %2 = toy.mul %0, %1 : tensor<3x2xf64>
  toy.return %2 : tensor<3x2xf64>
}
```

**Pattern operations**: `[transpose, add, mul]`

### Input IR
```mlir
toy.func @main() {
  %0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
  %1 = toy.constant dense<[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]> : tensor<3x2xf64>

  %2 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
  %3 = toy.add %2, %1 : tensor<3x2xf64>
  %4 = toy.mul %2, %3 : tensor<3x2xf64>

  toy.print %4 : tensor<3x2xf64>
  toy.return
}
```

### Step-by-Step Matching

#### Step 1: Walk operations, try %0 (first constant)
```cpp
bindings.clear()
matchedOps.clear()
matchSequence(%0, 0)  // Try to match pattern op 0 (transpose)
  matchOp(pattern_transpose, ir_constant)
    Names don't match: transpose != constant
  return false
```
❌ No match

#### Step 2: Try %1 (second constant)
```cpp
bindings.clear()
matchedOps.clear()
matchSequence(%1, 0)
  matchOp(pattern_transpose, ir_constant)
    Names don't match
  return false
```
❌ No match

#### Step 3: Try %2 (transpose) ⭐
```cpp
bindings.clear()
matchedOps.clear()
matchSequence(%2, 0)
  matchOp(pattern_transpose, ir_transpose)
    ✓ Names match: transpose == transpose
    ✓ Operand count: 1 == 1
    matchValue(pattern_%arg0, ir_%0)
      ✓ bindings[pattern_%arg0] = ir_%0
  ✓ matchedOps = [%2]
  ✓ bindings[pattern_%0] = ir_%2  // Bind result!

  matchRemainingOps(1)  // Find pattern op 1 (add)
    Search block for add operation...

    Try %3 (add):
      matchOp(pattern_add, ir_add)
        ✓ Names match: add == add
        ✓ Operand count: 2 == 2
        matchValue(pattern_%0, ir_%2)
          ✓ pattern_%0 already bound to ir_%2, check: %2 == %2 ✓
        matchValue(pattern_%arg1, ir_%1)
          ✓ bindings[pattern_%arg1] = ir_%1
      ✓ matchedOps = [%2, %3]
      ✓ bindings[pattern_%1] = ir_%3

      matchRemainingOps(2)  // Find pattern op 2 (mul)
        Search block for mul operation...

        Try %4 (mul):
          matchOp(pattern_mul, ir_mul)
            ✓ Names match: mul == mul
            ✓ Operand count: 2 == 2
            matchValue(pattern_%0, ir_%2)
              ✓ pattern_%0 already bound to ir_%2, check: %2 == %2 ✓
            matchValue(pattern_%1, ir_%3)
              ✓ pattern_%1 already bound to ir_%3, check: %3 == %3 ✓
          ✓ matchedOps = [%2, %3, %4]
          ✓ bindings[pattern_%2] = ir_%4

          matchRemainingOps(3)
            3 >= 3, all matched!
            return true
          return true
        return true
      return true
    return true
  return true
```

✅ **MATCH FOUND!**

**Final bindings**:
```
pattern_%arg0 → ir_%0
pattern_%arg1 → ir_%1
pattern_%0    → ir_%2
pattern_%1    → ir_%3
pattern_%2    → ir_%4
```

**Matched operations**: `[%2, %3, %4]`

### Key Observations

1. **SSA bindings enforce correctness**: `pattern_%0` bound to `ir_%2` on first use, verified on subsequent uses
2. **DAG search worked**: Operations matched in order [transpose, add, mul] even though they could be reordered
3. **Data-flow guided search**: `matchOp()` only succeeds if operand bindings are consistent
4. **No false positives**: If IR was `%4 = toy.mul %3, %3`, it would fail because `pattern_%0` bound to `%2`, not `%3`

---

## Why This Design Works

### 1. SSA Value Binding Ensures Correctness
- Same pattern value → same IR value (enforced by `matchValue()`)
- Prevents incorrect matches where data flow doesn't match

### 2. DAG Search Handles Non-Sequential Patterns
- `matchRemainingOps()` searches entire block
- Not limited to sequential operation order
- Critical for real-world patterns

### 3. Backtracking Explores All Possibilities
- Try different operation combinations
- Find valid matches even with multiple candidates

### 4. Prefix Matching Enables Composition
- Patterns can reference extracted functions
- Multi-level pattern composition becomes possible

### 5. Stateless Per-Attempt
- `bindings.clear()` and `matchedOps.clear()` before each attempt
- No cross-contamination between different match attempts

---

## Common Pitfalls and Solutions

### Pitfall 1: Forgetting to Bind Results
```cpp
// WRONG: Forget to bind result
matchedOps.push_back(startOp);
// Pattern uses %0 as operand later, but it's not bound!

// RIGHT: Always bind results
if (patternOp->getNumResults() == 1 && startOp->getNumResults() == 1) {
  bindings[patternOp->getResult(0)] = startOp->getResult(0);
}
```

### Pitfall 2: Not Checking Existing Bindings
```cpp
// WRONG: Always create new binding
bindings[patternVal] = irVal;

// RIGHT: Check if already bound
if (bindings.count(patternVal)) {
  return bindings[patternVal] == irVal;  // Verify consistency
}
```

### Pitfall 3: Sequential-Only Matching
```cpp
// WRONG: Only check next operation
Operation *nextOp = currentOp->getNextNode();
if (matchOp(patternOp, nextOp)) { ... }

// RIGHT: Search entire block
for (auto &candidateOp : searchBlock->getOperations()) {
  if (matchOp(patternOp, &candidateOp)) { ... }
}
```

### Pitfall 4: Not Handling Call Prefixes
```cpp
// WRONG: Exact match only
if (patternCallee != irCallee) return false;

// RIGHT: Prefix match for numeric suffixes
StringRef prefix = irCallee.rsplit('_').first;
if (hasNumericSuffix && prefix != patternCallee) return false;
```

---

## Performance Considerations

### Current Algorithm: O(n * m * k)
- **n**: Number of operations in IR
- **m**: Number of pattern operations
- **k**: Average number of candidate operations per pattern operation

### Optimizations (Not Implemented)
1. **Operation name indexing**: Build index of operations by name
2. **Early pruning**: Fail fast if required operations don't exist
3. **Dominance checking**: Use SSA dominance to prune invalid candidates
4. **Memoization**: Cache failed match attempts

**Why not optimized yet?**
- Current implementation prioritizes correctness and clarity
- Performance is acceptable for typical pattern sizes (2-5 operations)
- Optimizations can be added later without changing the algorithm

---

## Summary

The pattern matching algorithm combines:
1. **SSA Value Binding**: Ensures data-flow correctness
2. **DAG Search**: Handles non-sequential operation patterns
3. **Backtracking**: Explores all possible matches
4. **Prefix Matching**: Enables pattern composition

**Core Insight**: Pattern matching is a constraint satisfaction problem where:
- **Constraints**: Operation names, operand bindings, SSA consistency
- **Search strategy**: Backtracking with pruning via SSA bindings
- **Success criteria**: All pattern operations matched with consistent bindings

This design is **general**, **correct**, and **composable** - suitable for building complex pattern rewriting systems in MLIR!
