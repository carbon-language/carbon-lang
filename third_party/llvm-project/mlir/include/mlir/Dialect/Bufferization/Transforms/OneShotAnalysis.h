//===- OneShotAnalysis.h - One-Shot (Single Pass) Analysis ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_ONESHOTANALYSIS_H
#define MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_ONESHOTANALYSIS_H

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "llvm/ADT/EquivalenceClasses.h"

namespace mlir {
namespace bufferization {

struct OneShotBufferizationOptions;
class BufferizationAliasInfo;
class OneShotAnalysisState;

/// Options for analysis-enabled bufferization.
struct OneShotBufferizationOptions : public BufferizationOptions {
  OneShotBufferizationOptions() = default;

  /// Specifies whether returning newly allocated memrefs should be allowed.
  /// Otherwise, a pass failure is triggered.
  bool allowReturnAllocs = false;
};

/// The BufferizationAliasInfo class maintains a list of buffer aliases and
/// equivalence classes to support bufferization.
class BufferizationAliasInfo {
public:
  explicit BufferizationAliasInfo(Operation *rootOp);

  // BufferizationAliasInfo should be passed as a reference.
  BufferizationAliasInfo(const BufferizationAliasInfo &) = delete;

  /// Add a new entry for `v` in the `aliasInfo` and `equivalentInfo`. In the
  /// beginning the alias and equivalence sets only contain `v` itself.
  void createAliasInfoEntry(Value v);

  /// Insert an info entry for `newValue` and merge its alias set with that of
  /// `alias`.
  void insertNewBufferAlias(Value newValue, Value alias);

  /// Insert an info entry for `newValue` and merge its alias set with that of
  /// `alias`. Additionally, merge their equivalence classes.
  void insertNewBufferEquivalence(Value newValue, Value alias);

  /// Set the inPlace bufferization spec to true.
  /// Merge result's and operand's aliasing sets and iterate to a fixed point.
  void bufferizeInPlace(OpOperand &operand, AnalysisState &state);

  /// Set the inPlace bufferization spec to false.
  void bufferizeOutOfPlace(OpOperand &operand);

  /// Return true if `v1` and `v2` may bufferize to aliasing buffers.
  bool areAliasingBufferizedValues(Value v1, Value v2) const {
    return aliasInfo.isEquivalent(v1, v2);
  }

  /// Return true if `v1` and `v2` bufferize to equivalent buffers.
  bool areEquivalentBufferizedValues(Value v1, Value v2) const {
    return equivalentInfo.isEquivalent(v1, v2);
  }

  /// Union the alias sets of `v1` and `v2`.
  void unionAliasSets(Value v1, Value v2) { aliasInfo.unionSets(v1, v2); }

  /// Union the equivalence classes of `v1` and `v2`.
  void unionEquivalenceClasses(Value v1, Value v2) {
    equivalentInfo.unionSets(v1, v2);
  }

  /// Apply `fun` to all the members of the equivalence class of `v`.
  void applyOnEquivalenceClass(Value v, function_ref<void(Value)> fun) const;

  /// Apply `fun` to all aliases of `v`.
  void applyOnAliases(Value v, function_ref<void(Value)> fun) const;

  /// Mark a value as in-place bufferized.
  void markInPlace(OpOperand &o) { inplaceBufferized.insert(&o); }

  /// Return `true` if a value was marked as in-place bufferized.
  bool isInPlace(OpOperand &opOperand) const;

private:
  /// llvm::EquivalenceClasses wants comparable elements. This comparator uses
  /// uses pointer comparison on the defining op. This is a poor man's
  /// comparison but it's not like UnionFind needs ordering anyway.
  struct ValueComparator {
    bool operator()(const Value &lhs, const Value &rhs) const {
      return lhs.getImpl() < rhs.getImpl();
    }
  };

  using EquivalenceClassRangeType = llvm::iterator_range<
      llvm::EquivalenceClasses<Value, ValueComparator>::member_iterator>;
  /// Check that aliasInfo for `v` exists and return a reference to it.
  EquivalenceClassRangeType getAliases(Value v) const;

  /// Set of all OpResults that were decided to bufferize in-place.
  llvm::DenseSet<OpOperand *> inplaceBufferized;

  /// Auxiliary structure to store all the values a given value may alias with.
  /// Alias information is "may be" conservative: In the presence of branches, a
  /// value may alias with one of multiple other values. The concrete aliasing
  /// value may not even be known at compile time. All such values are
  /// considered to be aliases.
  llvm::EquivalenceClasses<Value, ValueComparator> aliasInfo;

  /// Auxiliary structure to store all the equivalent buffer classes. Equivalent
  /// buffer information is "must be" conservative: Only if two values are
  /// guaranteed to be equivalent at runtime, they said to be equivalent. It is
  /// possible that, in the presence of branches, it cannot be determined
  /// statically if two values are equivalent. In that case, the values are
  /// considered to be not equivalent.
  llvm::EquivalenceClasses<Value, ValueComparator> equivalentInfo;
};

/// State for analysis-enabled bufferization. This class keeps track of alias
/// (via BufferizationAliasInfo) to decide if tensor OpOperands should bufferize
/// in-place.
class OneShotAnalysisState : public AnalysisState {
public:
  OneShotAnalysisState(Operation *op,
                       const OneShotBufferizationOptions &options);

  OneShotAnalysisState(const OneShotAnalysisState &) = delete;

  virtual ~OneShotAnalysisState() = default;

  /// Return a reference to the BufferizationAliasInfo.
  BufferizationAliasInfo &getAliasInfo() { return aliasInfo; }

  /// Return `true` if the given OpResult has been decided to bufferize inplace.
  bool isInPlace(OpOperand &opOperand) const override;

  /// Return true if `v1` and `v2` bufferize to equivalent buffers.
  bool areEquivalentBufferizedValues(Value v1, Value v2) const override;

  /// Return true if `v1` and `v2` may bufferize to aliasing buffers.
  bool areAliasingBufferizedValues(Value v1, Value v2) const override;

  /// Return `true` if the given tensor has undefined contents.
  bool hasUndefinedContents(OpOperand *opOperand) const override;

  /// Return true if the given tensor (or an aliasing tensor) is yielded from
  /// the containing block. Also include all aliasing tensors in the same block.
  bool isTensorYielded(Value tensor) const override;

  /// Find all tensor values in the given operation that have undefined contents
  /// and store them in `undefinedTensorUses`.
  void gatherUndefinedTensorUses(Operation *op);

  /// Find all tensors that are yielded/returned from a block and store them in
  /// `yieldedTensors`. Also include all aliasing tensors in the same block.
  void gatherYieldedTensors(Operation *op);

  /// Return true if the buffer of the given tensor value is written to. Must
  /// not be called for values inside not yet analyzed functions.
  bool isValueWritten(Value value) const;

  /// Return true if the buffer of the given tensor value is writable.
  bool isWritable(Value value) const;

private:
  /// `aliasInfo` keeps track of aliasing and equivalent values. Only internal
  /// functions and `runOneShotBufferize` may access this object.
  BufferizationAliasInfo aliasInfo;

  /// A set of all tensors (and maybe aliasing tensors) that yielded from a
  /// block.
  DenseSet<Value> yieldedTensors;

  /// A set of uses of tensors that have undefined contents.
  DenseSet<OpOperand *> undefinedTensorUses;
};

/// Analyze `op` and its nested ops. Bufferization decisions are stored in
/// `state`.
LogicalResult analyzeOp(Operation *op, OneShotAnalysisState &state);

/// Run One-Shot Bufferize on the given op: Analysis + Bufferization
LogicalResult runOneShotBufferize(Operation *op,
                                  const OneShotBufferizationOptions &options);

} // namespace bufferization
} // namespace mlir

#endif // MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_ONESHOTANALYSIS_H
