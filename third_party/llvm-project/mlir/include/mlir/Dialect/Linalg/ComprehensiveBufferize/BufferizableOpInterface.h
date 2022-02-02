//===- BufferizableOpInterface.h - Comprehensive Bufferize ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_COMPREHENSIVEBUFFERIZE_BUFFERIZABLEOPINTERFACE_H_
#define MLIR_DIALECT_LINALG_COMPREHENSIVEBUFFERIZE_BUFFERIZABLEOPINTERFACE_H_

#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SetVector.h"

namespace mlir {
class BlockAndValueMapping;
class DominanceInfo;
class FuncOp;

namespace linalg {
namespace comprehensive_bufferize {

// TODO: from some HW description.
static constexpr int64_t kBufferAlignments = 128;

class BufferizationAliasInfo;
class BufferizableOpInterface;
struct BufferizationOptions;
class BufferizationState;
struct PostAnalysisStep;

/// Callback functions that are used to allocate/deallocate/copy memory buffers.
/// Comprehensive Bufferize provides default implementations of these functions.
// TODO: Could be replaced with a "bufferization strategy" object with virtual
// functions in the future.
struct AllocationCallbacks {
  using AllocationFn = std::function<FailureOr<Value>(
      OpBuilder &, Location, MemRefType, ArrayRef<Value>)>;
  using DeallocationFn = std::function<void(OpBuilder &, Location, Value)>;
  using MemCpyFn = std::function<void(OpBuilder &, Location, Value, Value)>;

  AllocationCallbacks(AllocationFn allocFn, DeallocationFn deallocFn,
                      MemCpyFn copyFn)
      : allocationFn(allocFn), deallocationFn(deallocFn), memCpyFn(copyFn) {}

  /// A function that allocates memory.
  AllocationFn allocationFn;

  /// A function that deallocated memory. Must be allocated by `allocationFn`.
  DeallocationFn deallocationFn;

  /// A function that copies memory between two allocations.
  MemCpyFn memCpyFn;
};

/// Return default allocation callbacks.
std::unique_ptr<AllocationCallbacks> defaultAllocationCallbacks();

/// PostAnalysisSteps can be registered with `BufferizationOptions` and are
/// executed after the analysis, but before bufferization. They can be used to
/// implement custom dialect-specific optimizations.
struct PostAnalysisStep {
  virtual ~PostAnalysisStep() {}

  /// Run the post analysis step. This function may modify the IR, but must keep
  /// `aliasInfo` consistent. Newly created operations and operations that
  /// should be re-analyzed must be added to `newOps`.
  virtual LogicalResult run(Operation *op, BufferizationState &state,
                            BufferizationAliasInfo &aliasInfo,
                            SmallVector<Operation *> &newOps) = 0;
};

using PostAnalysisStepList = std::vector<std::unique_ptr<PostAnalysisStep>>;

/// Options for ComprehensiveBufferize.
struct BufferizationOptions {
  BufferizationOptions();

  // BufferizationOptions cannot be copied.
  BufferizationOptions(const BufferizationOptions &other) = delete;

  /// Register a "post analysis" step. Such steps are executed after the
  /// analysis, but before bufferization.
  template <typename Step, typename... Args>
  void addPostAnalysisStep(Args... args) {
    postAnalysisSteps.emplace_back(
        std::make_unique<Step>(std::forward<Args>(args)...));
  }

  /// Return `true` if the op is allowed to be bufferized.
  bool isOpAllowed(Operation *op) const {
    if (!dialectFilter.hasValue())
      return true;
    return dialectFilter->contains(op->getDialect()->getNamespace());
  }

  /// Allow-list the given dialects in the dialect filter. Only ops from
  /// allow-listed dialects will be bufferized. If no dialect is added, ops from
  /// any dialect will be bufferized.
  template <typename... DialectTs>
  void addToDialectFilter() {
    // The following expands a call to addToDialectFilterImpl for each dialect
    // in 'DialectTs'. This magic is necessary due to a limitation in the places
    // that a parameter pack can be expanded in c++11.
    // FIXME: In c++17 this can be simplified by using 'fold expressions'.
    (void)std::initializer_list<int>{
        0, (addToDialectFilterImpl<DialectTs>(), 0)...};
  }

  /// Try to cast the given op to BufferizableOpInterface if the op is allow
  /// listed.
  BufferizableOpInterface dynCastBufferizableOp(Operation *op) const;

  /// Try to cast the given value to BufferizableOpInterface if the op is allow
  /// listed.
  BufferizableOpInterface dynCastBufferizableOp(Value value) const;

  /// Helper functions for allocation, deallocation, memory copying.
  std::unique_ptr<AllocationCallbacks> allocationFns;

  /// Specifies whether returning newly allocated memrefs should be allowed.
  /// Otherwise, a pass failure is triggered.
  bool allowReturnMemref = false;

  /// Specifies whether not bufferizable ops are allowed in the input. If so,
  /// bufferization.to_memref and bufferization.to_tensor ops are inserted at
  /// the boundaries.
  bool allowUnknownOps = false;

  /// Specifies whether dealloc ops should be generated along with alloc ops. If
  /// not, new memory allocations will leak.
  bool createDeallocs = true;

  /// Seed for the analysis fuzzer. If set to `0`, the fuzzer is deactivated.
  /// Should be used only with `testAnalysisOnly = true`.
  unsigned analysisFuzzerSeed = 0;

  /// If set to `true`, does not modify the IR apart from adding attributes (for
  /// checking the results of the analysis) and post analysis steps.
  bool testAnalysisOnly = false;

  /// If set to `true`, the IR is annotated with details about RaW conflicts.
  /// For debugging only. Should be used together with `testAnalysisOnly`.
  bool printConflicts = false;

  /// Registered post analysis steps.
  PostAnalysisStepList postAnalysisSteps;

  /// Only bufferize ops from dialects that are allowed-listed by the filter.
  /// All other ops are ignored. This option controls the scope of partial
  /// bufferization.
  ///
  /// Note: If no filter is specified, all ops are bufferized (as long as they
  /// implement BufferizableOpInterface). If a filter is specified,
  /// `allowUnknownOps` should be enabled. Otherwise, bufferization would fail
  /// when encountering an op that is forbidden by the filter.
  Optional<DenseSet<StringRef>> dialectFilter;

private:
  /// Allow-list a dialect in the dialect filter.
  template <typename DialectT>
  void addToDialectFilterImpl() {
    if (!dialectFilter.hasValue())
      dialectFilter.emplace();
    dialectFilter->insert(DialectT::getDialectNamespace());
  }
};

/// Specify fine-grain relationship between buffers to enable more analysis.
enum class BufferRelation {
  None,
  // TODO: ResultContainsOperand,
  // TODO: OperandContainsResult,
  Equivalent
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
  void bufferizeInPlace(OpOperand &operand, BufferizationState &state);

  /// Set the inPlace bufferization spec to false.
  void bufferizeOutOfPlace(OpOperand &operand);

  /// Return true if `v1` and `v2` bufferize to equivalent buffers.
  bool areEquivalentBufferizedValues(Value v1, Value v2) const {
    return equivalentInfo.isEquivalent(v1, v2);
  }

  /// Return true if `v1` and `v2` bufferize to aliasing buffers.
  bool areAliasingBufferizedValues(Value v1, Value v2) const {
    return aliasInfo.isEquivalent(v1, v2);
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

/// Return `true` if the given value is a BlockArgument of a FuncOp.
bool isFunctionArgument(Value value);

/// Dialect-specific bufferization state. Analysis/bufferization information
/// that is specific to ops from a certain dialect can be stored in derived
/// variants of this struct.
struct DialectBufferizationState {
  DialectBufferizationState() = default;

  virtual ~DialectBufferizationState() = default;

  // Copying state is forbidden. Always pass as reference.
  DialectBufferizationState(const DialectBufferizationState &) = delete;
};

/// BufferizationState provides a variety of helper functions for dealing with
/// tensor values and memref buffers.
class BufferizationState {
public:
  BufferizationState(Operation *op, const BufferizationOptions &options);

  // BufferizationState should be passed as a reference.
  BufferizationState(const BufferizationState &) = delete;

  /// Determine which OpOperand* will alias with `result` if the op is
  /// bufferized in place. Return an empty vector if the op is not bufferizable.
  SmallVector<OpOperand *> getAliasingOpOperand(OpResult result) const;

  /// Determine which OpResult will alias with `opOperand` if the op is
  /// bufferized in place. Return an empty OpResult if the op is not
  /// bufferizable.
  OpResult getAliasingOpResult(OpOperand &opOperand) const;

  /// Return true if `opOperand` bufferizes to a memory read. Return `true` if
  /// the op is not bufferizable.
  bool bufferizesToMemoryRead(OpOperand &opOperand) const;

  /// Return true if `opOperand` bufferizes to a memory write. Return true` if
  /// the op is not bufferizable.
  bool bufferizesToMemoryWrite(OpOperand &opOperand) const;

  /// Return true if `opOperand` does neither read nor write but bufferizes to
  /// an alias. Return false if the op is not bufferizable.
  bool bufferizesToAliasOnly(OpOperand &opOperand) const;

  /// Return true if the given value is read by an op that bufferizes to a
  /// memory read. Also takes into account ops that create an alias but do not
  /// read by themselves (e.g., ExtractSliceOp).
  bool isValueRead(Value value) const;

  /// Starting from `value`, follow the use-def chain in reverse, always
  /// selecting the aliasing OpOperands. Find and return Values for which
  /// `condition` evaluates to true. OpOperands of such matching Values are not
  /// traversed any further.
  ///
  /// When reaching the end of a chain (BlockArgument or Value without aliasing
  /// OpOperands), also return the last Value of that chain.
  ///
  /// Example:
  ///
  ///                               8
  ///                               |
  ///   6*         7*         +-----+----+
  ///   |          |          |          |
  ///   2*         3          4*         5
  ///   |          |          |          |
  ///   +----------+----------+----------+
  ///              |
  ///              1
  ///
  /// In the above example, Values with a star satisfy the condition. When
  /// starting the traversal from Value 1, the resulting SetVector is:
  /// { 2, 7, 8, 5 }
  SetVector<Value> findValueInReverseUseDefChain(
      Value value, llvm::function_ref<bool(Value)> condition) const;

  /// Find the Values of the last preceding write of a given Value.
  ///
  /// Note: Unknown ops are handled conservatively and assumed to be writes.
  /// Furthermore, BlockArguments are also assumed to be writes. There is no
  /// analysis across block boundaries.
  ///
  /// Note: When reaching an end of the reverse SSA use-def chain, that value
  /// is returned regardless of whether it is a memory write or not.
  SetVector<Value> findLastPrecedingWrite(Value value) const;

  /// Creates a memref allocation.
  FailureOr<Value> createAlloc(OpBuilder &b, Location loc, MemRefType type,
                               ArrayRef<Value> dynShape) const;

  /// Creates a memref allocation for the given shaped value. This function may
  /// perform additional optimizations such as buffer allocation hoisting. If
  /// `createDealloc`, a deallocation op is inserted at the point where the
  /// allocation goes out of scope.
  FailureOr<Value> createAlloc(OpBuilder &b, Location loc, Value shapedValue,
                               bool deallocMemref) const;

  /// Creates a memref deallocation. The given memref buffer must have been
  /// allocated using `createAlloc`.
  void createDealloc(OpBuilder &b, Location loc, Value allocatedBuffer) const;

  /// Creates a memcpy between two given buffers.
  void createMemCpy(OpBuilder &b, Location loc, Value from, Value to) const;

  /// Return `true` if the given OpResult has been decided to bufferize inplace.
  bool isInPlace(OpOperand &opOperand) const;

  /// Return the buffer (memref) for a given OpOperand (tensor). Allocate
  /// a new buffer and copy over data from the existing buffer if out-of-place
  /// bufferization was decided.
  FailureOr<Value> getBuffer(RewriterBase &rewriter, OpOperand &opOperand,
                             bool forceInPlace = false) const;

  /// Return dialect-specific bufferization state.
  template <typename StateT>
  Optional<const StateT *> getDialectState(StringRef name) const {
    auto it = dialectState.find(name);
    if (it == dialectState.end())
      return None;
    return static_cast<const StateT *>(it->getSecond().get());
  }

  /// Return dialect-specific bufferization state or create one if none exists.
  template <typename StateT> StateT &getOrCreateDialectState(StringRef name) {
    // Create state if it does not exist yet.
    if (!dialectState.count(name))
      dialectState[name] = std::make_unique<StateT>();
    return static_cast<StateT &>(*dialectState[name]);
  }

  /// Return a reference to the BufferizationOptions.
  const BufferizationOptions &getOptions() const { return options; }

  /// Return a reference to the BufferizationAliasInfo.
  BufferizationAliasInfo &getAliasInfo() { return aliasInfo; }

private:
  /// `aliasInfo` keeps track of aliasing and equivalent values. Only internal
  /// functions and `runComprehensiveBufferize` may access this object.
  BufferizationAliasInfo aliasInfo;

  /// Dialect-specific bufferization state.
  DenseMap<StringRef, std::unique_ptr<DialectBufferizationState>> dialectState;

  /// A reference to current bufferization options.
  const BufferizationOptions &options;
};

/// Replace an op with replacement values. The op is deleted. Tensor OpResults
/// must be replaced with memref values.
void replaceOpWithBufferizedValues(RewriterBase &rewriter, Operation *op,
                                   ValueRange values);

/// Replace an op with a new op. The new op must have the same number of
/// results as the replaced op. The new op may not return any tensor values.
template <typename OpTy, typename... Args>
OpTy replaceOpWithNewBufferizedOp(RewriterBase &rewriter, Operation *op,
                                  Args &&...args) {
  auto newOp = rewriter.create<OpTy>(op->getLoc(), std::forward<Args>(args)...);
  replaceOpWithBufferizedValues(rewriter, op, newOp->getResults());
  return newOp;
}

/// Return a contiguous MemRefType (i.e. with canonical/empty layout map)
/// with the same shape as `shapedType` and specified `layout` and
/// `addressSpace`.
MemRefType getContiguousMemRefType(ShapedType shapedType,
                                   MemRefLayoutAttrInterface layout = {},
                                   Attribute memorySpace = {});

/// Return an UnrankedMemRefType with the given element type and memory space.
UnrankedMemRefType getUnrankedMemRefType(Type elementType,
                                         Attribute memorySpace = {});

/// Return a MemRefType to which the `tensorType` can be bufferized in a
/// composable fashion. The layout must be the most dynamic possible and
/// canonicalize away once bufferization is finished.
MemRefType getDynamicMemRefType(RankedTensorType tensorType,
                                unsigned addressSpace = 0);

} // namespace comprehensive_bufferize
} // namespace linalg
} // namespace mlir

#include "mlir/Dialect/Linalg/ComprehensiveBufferize/BufferizableOpInterface.h.inc"

namespace mlir {
namespace linalg {
namespace comprehensive_bufferize {

/// AllocationHoistingBarrierOnly is an external implementation of
/// BufferizableOpInterface for ops that are (not yet) bufferizable, but are
/// known to be allocation hoisting barriers. All interface methods (except for
/// `isAllocationHoistingBarrier`) are implemented conservatively.
template <typename OpTy>
struct AllocationHoistingBarrierOnly
    : public BufferizableOpInterface::ExternalModel<
          AllocationHoistingBarrierOnly<OpTy>, OpTy> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const BufferizationState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const BufferizationState &state) const {
    return true;
  }

  SmallVector<OpOperand *>
  getAliasingOpOperand(Operation *op, OpResult opResult,
                       const BufferizationState &state) const {
    return {};
  }

  OpResult getAliasingOpResult(Operation *op, OpOperand &opOperand,
                               const BufferizationState &state) const {
    return OpResult();
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const BufferizationAliasInfo &aliasInfo,
                                const BufferizationState &state) const {
    return BufferRelation::None;
  }

  bool isWritable(Operation *op, Value value,
                  const BufferizationState &state) const {
    return false;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationState &state) const {
    return failure();
  }

  bool isAllocationHoistingBarrier(Operation *op) const { return true; }
};

} // namespace comprehensive_bufferize
} // namespace linalg
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_COMPREHENSIVEBUFFERIZE_BUFFERIZABLEOPINTERFACE_H_
