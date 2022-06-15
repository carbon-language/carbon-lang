//===- TransformInterfaces.h - Transform Dialect Interfaces -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TRANSFORM_IR_TRANSFORMINTERFACES_H
#define MLIR_DIALECT_TRANSFORM_IR_TRANSFORMINTERFACES_H

#include "mlir/IR/OpDefinition.h"

#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {

/// The result of a transform IR operation application. This can have one of the
/// three states:
///   - success;
///   - silencable (recoverable) failure with yet-unreported diagnostic;
///   - definite failure.
/// Silencable failure is intended to communicate information about
/// transformations that did not apply but in a way that supports recovery,
/// for example, they did not modify the payload IR or modified it in some
/// predictable way. They are associated with a Diagnostic that provides more
/// details on the failure. Silencable failure can be discarded, turning the
/// result into success, or "reported", emitting the diagnostic and turning the
/// result into definite failure. Transform IR operations containing other
/// operations are allowed to do either with the results of the nested
/// transformations, but must propagate definite failures as their diagnostics
/// have been already reported to the user.
class LLVM_NODISCARD DiagnosedSilencableFailure {
public:
  explicit DiagnosedSilencableFailure(LogicalResult result) : result(result) {}
  DiagnosedSilencableFailure(const DiagnosedSilencableFailure &) = delete;
  DiagnosedSilencableFailure &
  operator=(const DiagnosedSilencableFailure &) = delete;
  DiagnosedSilencableFailure(DiagnosedSilencableFailure &&) = default;
  DiagnosedSilencableFailure &
  operator=(DiagnosedSilencableFailure &&) = default;

  /// Constructs a DiagnosedSilencableFailure in the success state.
  static DiagnosedSilencableFailure success() {
    return DiagnosedSilencableFailure(::mlir::success());
  }

  /// Constructs a DiagnosedSilencableFailure in the failure state. Typically,
  /// a diagnostic has been emitted before this.
  static DiagnosedSilencableFailure definiteFailure() {
    return DiagnosedSilencableFailure(::mlir::failure());
  }

  /// Constructs a DiagnosedSilencableFailure in the silencable failure state,
  /// ready to emit the given diagnostic. This is considered a failure
  /// regardless of the diagnostic severity.
  static DiagnosedSilencableFailure silencableFailure(Diagnostic &&diag) {
    return DiagnosedSilencableFailure(std::forward<Diagnostic>(diag));
  }

  /// Converts all kinds of failure into a LogicalResult failure, emitting the
  /// diagnostic if necessary. Must not be called more than once.
  LogicalResult checkAndReport() {
#if LLVM_ENABLE_ABI_BREAKING_CHECKS
    assert(!reported && "attempting to report a diagnostic more than once");
    reported = true;
#endif // LLVM_ENABLE_ABI_BREAKING_CHECKS
    if (diagnostic) {
      diagnostic->getLocation().getContext()->getDiagEngine().emit(
          std::move(*diagnostic));
      diagnostic.reset();
      result = ::mlir::failure();
    }
    return result;
  }

  /// Returns `true` if this is a silencable failure.
  bool isSilencableFailure() const { return diagnostic.hasValue(); }

  /// Returns `true` if this is a success.
  bool succeeded() const {
    return !diagnostic.hasValue() && ::mlir::succeeded(result);
  }

  /// Returns the diagnostic message without emitting it. Expects this object
  /// to be a silencable failure.
  std::string getMessage() const { return diagnostic->str(); }

  /// Converts silencable failure into LogicalResult success without reporting
  /// the diagnostic, preserves the other states.
  LogicalResult silence() {
    if (diagnostic) {
      diagnostic.reset();
      result = ::mlir::success();
    }
    return result;
  }

  /// Streams the given values into the diagnotic. Expects this object to be a
  /// silencable failure.
  template <typename T> DiagnosedSilencableFailure &operator<<(T &&value) & {
    assert(isSilencableFailure() &&
           "can only append output in silencable failure state");
    *diagnostic << std::forward<T>(value);
    return *this;
  }
  template <typename T> DiagnosedSilencableFailure &&operator<<(T &&value) && {
    return std::move(this->operator<<(std::forward<T>(value)));
  }

  /// Attaches a note to the diagnostic. Expects this object to be a silencable
  /// failure.
  Diagnostic &attachNote(Optional<Location> loc = llvm::None) {
    assert(isSilencableFailure() &&
           "can only attach notes to silencable failures");
    return diagnostic->attachNote(loc);
  }

private:
  explicit DiagnosedSilencableFailure(Diagnostic &&diagnostic)
      : diagnostic(std::move(diagnostic)), result(failure()) {}

  /// The diagnostic associated with this object. If present, the object is
  /// considered to be in the silencable failure state regardless of the
  /// `result` field.
  Optional<Diagnostic> diagnostic;

  /// The "definite" logical state, either success or failure. Ignored if the
  /// diagnostic message is present.
  LogicalResult result;

#if LLVM_ENABLE_ABI_BREAKING_CHECKS
  /// Whther the associated diagnostic have been reported. Diagnostic reporting
  /// consumes the diagnostic, so we need a mechanism to differentiate a
  /// reported diagnostic from a state where it was never created.
  bool reported = false;
#endif // LLVM_ENABLE_ABI_BREAKING_CHECKS
};

namespace transform {

class TransformOpInterface;

/// Options controlling the application of transform operations by the
/// TransformState.
class TransformOptions {
public:
  TransformOptions() {}

  /// Requests computationally expensive checks of the transform and payload IR
  /// well-formedness to be performed before each transformation. In particular,
  /// these ensure that the handles still point to valid operations when used.
  TransformOptions &enableExpensiveChecks(bool enable = true) {
    expensiveChecksEnabled = enable;
    return *this;
  }

  /// Returns true if the expensive checks are requested.
  bool getExpensiveChecksEnabled() const { return expensiveChecksEnabled; }

private:
  bool expensiveChecksEnabled = true;
};

/// The state maintained across applications of various ops implementing the
/// TransformOpInterface. The operations implementing this interface and the
/// surrounding structure are referred to as transform IR. The operations to
/// which transformations apply are referred to as payload IR. The state thus
/// contains the mapping between values defined in the transform IR ops and
/// payload IR ops. It assumes that each value in the transform IR can be used
/// at most once (since transformations are likely to change the payload IR ops
/// the value corresponds to). Checks that transform IR values correspond to
/// disjoint sets of payload IR ops throughout the transformation.
///
/// A reference to this class is passed as an argument to "apply" methods of the
/// transform op interface. Thus the "apply" method can call
/// `state.getPayloadOps( getSomeOperand() )` to obtain the list of operations
/// associated with its operand and subject to transformation. The method is
/// expected to populate the `TransformResults` class instance in order to
/// update the mapping. The `applyTransform` method takes care of propagating
/// the state of `TransformResults` into the instance of this class.
///
/// When applying transform IR operations with regions, the client is expected
/// to create a RegionScope RAII object to create a new "stack frame" for
/// values defined inside the region. The mappings from and to these values will
/// be automatically dropped when the object goes out of scope, typically at the
/// end of the "apply" function of the parent operation. If a region contains
/// blocks with arguments, the client can map those arguments to payload IR ops
/// using "mapBlockArguments".
class TransformState {
  /// Mapping between a Value in the transform IR and the corresponding set of
  /// operations in the payload IR.
  using TransformOpMapping = DenseMap<Value, SmallVector<Operation *>>;

  /// Mapping between a payload IR operation and the transform IR value it is
  /// currently associated with.
  using TransformOpReverseMapping = DenseMap<Operation *, Value>;

  /// Bidirectional mappings between transform IR values and payload IR
  /// operations.
  struct Mappings {
    TransformOpMapping direct;
    TransformOpReverseMapping reverse;
  };

public:
  /// Creates a state for transform ops living in the given region. The parent
  /// operation of the region. The second argument points to the root operation
  /// in the payload IR beind transformed, which may or may not contain the
  /// region with transform ops. Additional options can be provided through the
  /// trailing configuration object.
  TransformState(Region &region, Operation *root,
                 const TransformOptions &options = TransformOptions());

  /// Returns the op at which the transformation state is rooted. This is
  /// typically helpful for transformations that apply globally.
  Operation *getTopLevel() const;

  /// Returns the list of ops that the given transform IR value corresponds to.
  /// This is helpful for transformations that apply to a particular handle.
  ArrayRef<Operation *> getPayloadOps(Value value) const;

  /// Returns the Transform IR handle for the given Payload IR op if it exists
  /// in the state, null otherwise.
  Value getHandleForPayloadOp(Operation *op) const;

  /// Applies the transformation specified by the given transform op and updates
  /// the state accordingly.
  DiagnosedSilencableFailure applyTransform(TransformOpInterface transform);

  /// Records the mapping between a block argument in the transform IR and a
  /// list of operations in the payload IR. The arguments must be defined in
  /// blocks of the currently processed transform IR region, typically after a
  /// region scope is defined.
  LogicalResult mapBlockArguments(BlockArgument argument,
                                  ArrayRef<Operation *> operations) {
#if LLVM_ENABLE_ABI_BREAKING_CHECKS
    assert(argument.getParentRegion() == regionStack.back() &&
           "mapping block arguments from a region other than the active one");
#endif // LLVM_ENABLE_ABI_BREAKING_CHECKS
    return setPayloadOps(argument, operations);
  }

  // Forward declarations to support limited visibility.
  class RegionScope;

  /// Creates a new region scope for the given region. The region is expected to
  /// be nested in the currently processed region.
  // Implementation note: this method is inline but implemented outside of the
  // class body to comply with visibility and full-declaration requirements.
  inline RegionScope make_region_scope(Region &region);

  /// A RAII object maintaining a "stack frame" for a transform IR region. When
  /// applying a transform IR operation that contains a region, the caller is
  /// expected to create a RegionScope before applying the ops contained in the
  /// region. This ensures that the mappings between values defined in the
  /// transform IR region and payload IR operations are cleared when the region
  /// processing ends; such values cannot be accessed outside the region.
  class RegionScope {
  public:
    /// Forgets the mapping from or to values defined in the associated
    /// transform IR region.
    ~RegionScope() {
      state.mappings.erase(region);
#if LLVM_ENABLE_ABI_BREAKING_CHECKS
      state.regionStack.pop_back();
#endif // LLVM_ENABLE_ABI_BREAKING_CHECKS
    }

  private:
    /// Creates a new scope for mappings between values defined in the given
    /// transform IR region and payload IR operations.
    RegionScope(TransformState &state, Region &region)
        : state(state), region(&region) {
      auto res = state.mappings.try_emplace(this->region);
      assert(res.second && "the region scope is already present");
      (void)res;
#if LLVM_ENABLE_ABI_BREAKING_CHECKS
      assert(state.regionStack.back()->isProperAncestor(&region) &&
             "scope started at a non-nested region");
      state.regionStack.push_back(&region);
#endif // LLVM_ENABLE_ABI_BREAKING_CHECKS
    }

    /// Back-reference to the transform state.
    TransformState &state;

    /// The region this scope is associated with.
    Region *region;

    friend RegionScope TransformState::make_region_scope(Region &);
  };
  friend class RegionScope;

  /// Base class for TransformState extensions that allow TransformState to
  /// contain user-specified information in the state object. Clients are
  /// expected to derive this class, add the desired fields, and make the
  /// derived class compatible with the MLIR TypeID mechanism:
  ///
  /// ```mlir
  /// class MyExtension final : public TransformState::Extension {
  /// public:
  ///   MyExtension(TranfsormState &state, int myData)
  ///     : Extension(state) {...}
  /// private:
  ///   int mySupplementaryData;
  /// };
  /// ```
  ///
  /// Instances of this and derived classes are not expected to be created by
  /// the user, instead they are directly constructed within a TransformState. A
  /// TransformState can only contain one extension with the given TypeID.
  /// Extensions can be obtained from a TransformState instance, and can be
  /// removed when they are no longer required.
  ///
  /// ```mlir
  /// transformState.addExtension<MyExtension>(/*myData=*/42);
  /// MyExtension *ext = transformState.getExtension<MyExtension>();
  /// ext->doSomething();
  /// ```
  class Extension {
    // Allow TransformState to allocate Extensions.
    friend class TransformState;

  public:
    /// Base virtual destructor.
    // Out-of-line definition ensures symbols are emitted in a single object
    // file.
    virtual ~Extension();

  protected:
    /// Constructs an extension of the given TransformState object.
    Extension(TransformState &state) : state(state) {}

    /// Provides read-only access to the parent TransformState object.
    const TransformState &getTransformState() const { return state; }

    /// Replaces the given payload op with another op. If the replacement op is
    /// null, removes the association of the payload op with its handle.
    LogicalResult replacePayloadOp(Operation *op, Operation *replacement);

  private:
    /// Back-reference to the state that is being extended.
    TransformState &state;
  };

  /// Adds a new Extension of the type specified as template parameter,
  /// constructing it with the arguments provided. The extension is owned by the
  /// TransformState. It is expected that the state does not already have an
  /// extension of the same type. Extension constructors are expected to take
  /// a reference to TransformState as first argument, automatically supplied
  /// by this call.
  template <typename Ty, typename... Args>
  Ty &addExtension(Args &&...args) {
    static_assert(
        std::is_base_of<Extension, Ty>::value,
        "only an class derived from TransformState::Extension is allowed here");
    auto ptr = std::make_unique<Ty>(*this, std::forward<Args>(args)...);
    auto result = extensions.try_emplace(TypeID::get<Ty>(), std::move(ptr));
    assert(result.second && "extension already added");
    return *static_cast<Ty *>(result.first->second.get());
  }

  /// Returns the extension of the specified type.
  template <typename Ty>
  Ty *getExtension() {
    static_assert(
        std::is_base_of<Extension, Ty>::value,
        "only an class derived from TransformState::Extension is allowed here");
    auto iter = extensions.find(TypeID::get<Ty>());
    if (iter == extensions.end())
      return nullptr;
    return static_cast<Ty *>(iter->second.get());
  }

  /// Removes the extension of the specified type.
  template <typename Ty>
  void removeExtension() {
    static_assert(
        std::is_base_of<Extension, Ty>::value,
        "only an class derived from TransformState::Extension is allowed here");
    extensions.erase(TypeID::get<Ty>());
  }

private:
  /// Identifier for storing top-level value in the `operations` mapping.
  static constexpr Value kTopLevelValue = Value();

  /// Returns the mappings frame for the reigon in which the value is defined.
  const Mappings &getMapping(Value value) const {
    return const_cast<TransformState *>(this)->getMapping(value);
  }
  Mappings &getMapping(Value value) {
    auto it = mappings.find(value.getParentRegion());
    assert(it != mappings.end() &&
           "trying to find a mapping for a value from an unmapped region");
    return it->second;
  }

  /// Returns the mappings frame for the region in which the operation resides.
  const Mappings &getMapping(Operation *operation) const {
    return const_cast<TransformState *>(this)->getMapping(operation);
  }
  Mappings &getMapping(Operation *operation) {
    auto it = mappings.find(operation->getParentRegion());
    assert(it != mappings.end() &&
           "trying to find a mapping for an operation from an unmapped region");
    return it->second;
  }

  /// Sets the payload IR ops associated with the given transform IR value.
  /// Fails if this would result in multiple transform IR values with uses
  /// corresponding to the same payload IR ops. For example, a hypothetical
  /// "find function by name" transform op would (indirectly) call this
  /// function for its result. Having two such calls in a row with for different
  /// values, e.g. coming from different ops:
  ///
  ///   %0 = transform.find_func_by_name { name = "myfunc" }
  ///   %1 = transform.find_func_by_name { name = "myfunc" }
  ///
  /// would lead to both values pointing to the same operation. The second call
  /// to setPayloadOps will fail, unless the association with the %0 value is
  /// removed first by calling update/removePayloadOps.
  LogicalResult setPayloadOps(Value value, ArrayRef<Operation *> targets);

  /// Forgets the payload IR ops associated with the given transform IR value.
  void removePayloadOps(Value value);

  /// Updates the payload IR ops associated with the given transform IR value.
  /// The callback function is called once per associated operation and is
  /// expected to return the modified operation or nullptr. In the latter case,
  /// the corresponding operation is no longer associated with the transform IR
  /// value. May fail if the operation produced by the update callback is
  /// already associated with a different Transform IR handle value.
  LogicalResult
  updatePayloadOps(Value value,
                   function_ref<Operation *(Operation *)> callback);

  /// Attempts to record the mapping between the given Payload IR operation and
  /// the given Transform IR handle. Fails and reports an error if the operation
  /// is already tracked by another handle.
  static LogicalResult tryEmplaceReverseMapping(Mappings &map, Operation *op,
                                                Value handle);

  /// If the operand is a handle consumed by the operation, i.e. has the "free"
  /// memory effect associated with it, identifies other handles that are
  /// pointing to payload IR operations nested in the operations pointed to by
  /// the consumed handle. Marks all such handles as invalidated so trigger
  /// errors if they are used.
  void recordHandleInvalidation(OpOperand &handle);

  /// Checks that the operation does not use invalidated handles as operands.
  /// Reports errors and returns failure if it does. Otherwise, invalidates the
  /// handles consumed by the operation as well as any handles pointing to
  /// payload IR operations nested in the operations associated with the
  /// consumed handles.
  LogicalResult
  checkAndRecordHandleInvalidation(TransformOpInterface transform);

  /// The mappings between transform IR values and payload IR ops, aggregated by
  /// the region in which the transform IR values are defined.
  llvm::SmallDenseMap<Region *, Mappings> mappings;

  /// Extensions attached to the TransformState, identified by the TypeID of
  /// their type. Only one extension of any given type is allowed.
  DenseMap<TypeID, std::unique_ptr<Extension>> extensions;

  /// The top-level operation that contains all payload IR, typically a module.
  Operation *topLevel;

  /// Additional options controlling the transformation state behavior.
  TransformOptions options;

  /// The mapping from invalidated handles to the error-reporting functions that
  /// describe when the handles were invalidated. Calling such a function emits
  /// a user-visible diagnostic.
  DenseMap<Value, std::function<void()>> invalidatedHandles;

#if LLVM_ENABLE_ABI_BREAKING_CHECKS
  /// A stack of nested regions that are being processed in the transform IR.
  /// Each region must be an ancestor of the following regions in this list.
  /// These are also the keys for "mappings".
  SmallVector<Region *> regionStack;
#endif // LLVM_ENABLE_ABI_BREAKING_CHECKS
};

/// Local mapping between values defined by a specific op implementing the
/// TransformOpInterface and the payload IR ops they correspond to.
class TransformResults {
  friend class TransformState;

public:
  /// Indicates that the result of the transform IR op at the given position
  /// corresponds to the given list of payload IR ops. Each result must be set
  /// by the transformation exactly once.
  void set(OpResult value, ArrayRef<Operation *> ops);

private:
  /// Creates an instance of TransformResults that expects mappings for
  /// `numSegments` values.
  explicit TransformResults(unsigned numSegments);

  /// Gets the list of operations associated with the result identified by its
  /// number in the list of operation results.
  ArrayRef<Operation *> get(unsigned resultNumber) const;

  /// Storage for pointers to payload IR ops that are associated with results of
  /// a transform IR op. `segments` contains as many entries as the transform IR
  /// op has results. Each entry is a reference to a contiguous segment in
  /// the `operations` list that contains the pointers to operations. This
  /// allows for operations to be stored contiguously without nested vectors and
  /// for different segments to be set in any order.
  SmallVector<ArrayRef<Operation *>, 2> segments;
  SmallVector<Operation *> operations;
};

TransformState::RegionScope TransformState::make_region_scope(Region &region) {
  return RegionScope(*this, region);
}

namespace detail {
/// Maps the only block argument of the op with PossibleTopLevelTransformOpTrait
/// to either the list of operations associated with its operand or the root of
/// the payload IR, depending on what is available in the context.
LogicalResult
mapPossibleTopLevelTransformOpBlockArguments(TransformState &state,
                                             Operation *op, unsigned region);

/// Verification hook for PossibleTopLevelTransformOpTrait.
LogicalResult verifyPossibleTopLevelTransformOpTrait(Operation *op);
} // namespace detail

/// This trait is supposed to be attached to Transform dialect operations that
/// can be standalone top-level transforms. Such operations typically contain
/// other Transform dialect operations that can be executed following some
/// control flow logic specific to the current operation. The operations with
/// this trait are expected to have at least one single-block region with one
/// argument of PDL Operation type. The operations are also expected to be valid
/// without operands, in which case they are considered top-level, and with one
/// or more arguments, in which case they are considered nested. Top-level
/// operations have the block argument of the entry block in the Transform IR
/// correspond to the root operation of Payload IR. Nested operations have the
/// block argument of the entry block in the Transform IR correspond to a list
/// of Payload IR operations mapped to the first operand of the Transform IR
/// operation. The operation must implement TransformOpInterface.
template <typename OpTy>
class PossibleTopLevelTransformOpTrait
    : public OpTrait::TraitBase<OpTy, PossibleTopLevelTransformOpTrait> {
public:
  /// Verifies that `op` satisfies the invariants of this trait. Not expected to
  /// be called directly.
  static LogicalResult verifyTrait(Operation *op) {
    return detail::verifyPossibleTopLevelTransformOpTrait(op);
  }

  /// Returns the single block of the given region.
  Block *getBodyBlock(unsigned region = 0) {
    return &this->getOperation()->getRegion(region).front();
  }

  /// Sets up the mapping between the entry block of the given region of this op
  /// and the relevant list of Payload IR operations in the given state. The
  /// state is expected to be already scoped at the region of this operation.
  /// Returns failure if the mapping failed, e.g., the value is already mapped.
  LogicalResult mapBlockArguments(TransformState &state, unsigned region = 0) {
    return detail::mapPossibleTopLevelTransformOpBlockArguments(
        state, this->getOperation(), region);
  }
};

/// Trait implementing the TransformOpInterface for operations applying a
/// transformation to a single operation handle and producing a single operation
/// handle. The op must implement a method with one of the following signatures:
///   - FailureOr<convertible-to-Operation*> applyToOne(OpTy)
///   - LogicalResult applyToOne(OpTy)
/// to perform a transformation that is applied in turn to all payload IR
/// operations that correspond to the handle of the transform IR operation.
/// In the functions above, OpTy is either Operation * or a concrete payload IR
/// Op class that the transformation is applied to (NOT the class of the
/// transform IR op). The op is expected to have one operand and zero or one
/// results.
template <typename OpTy>
class TransformEachOpTrait
    : public OpTrait::TraitBase<OpTy, TransformEachOpTrait> {
public:
  /// Calls `applyToOne` for every payload operation associated with the operand
  /// of this transform IR op. If `applyToOne` returns ops, associates them with
  /// the result of this transform op.
  DiagnosedSilencableFailure apply(TransformResults &transformResults,
                                   TransformState &state);

  /// Checks that the op matches the expectations of this trait.
  static LogicalResult verifyTrait(Operation *op);
};

/// Side effect resource corresponding to the mapping between Transform IR
/// values and Payload IR operations. An Allocate effect from this resource
/// means creating a new mapping entry, it is always accompanied by a Write
/// effet. A Read effect from this resource means accessing the mapping. A Free
/// effect on this resource indicates the removal of the mapping entry,
/// typically after a transformation that modifies the Payload IR operations
/// associated with one of the Transform IR operation's operands. It is always
/// accompanied by a Read effect. Read-after-Free and double-Free are not
/// allowed (they would be problematic with "regular" memory effects too) as
/// they indicate an attempt to access Payload IR operations that have been
/// modified, potentially erased, by the previous tranfsormations.
// TODO: consider custom effects if these are not enabling generic passes such
// as CSE/DCE to work.
struct TransformMappingResource
    : public SideEffects::Resource::Base<TransformMappingResource> {
  StringRef getName() override { return "transform.mapping"; }
};

/// Side effect resource corresponding to the Payload IR itself. Only Read and
/// Write effects are expected on this resource, with Write always accompanied
/// by a Read (short of fully replacing the top-level Payload IR operation, one
/// cannot modify the Payload IR without reading it first). This is intended
/// to disallow reordering of Transform IR operations that mutate the Payload IR
/// while still allowing the reordering of those that only access it.
struct PayloadIRResource
    : public SideEffects::Resource::Base<PayloadIRResource> {
  StringRef getName() override { return "transform.payload_ir"; }
};

/// Trait implementing the MemoryEffectOpInterface for operations that "consume"
/// their operands and produce new results.
template <typename OpTy>
class FunctionalStyleTransformOpTrait
    : public OpTrait::TraitBase<OpTy, FunctionalStyleTransformOpTrait> {
public:
  /// This op "consumes" the operands by reading and freeing then, "produces"
  /// the results by allocating and writing it and reads/writes the payload IR
  /// in the process.
  void getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
    for (Value operand : this->getOperation()->getOperands()) {
      effects.emplace_back(MemoryEffects::Read::get(), operand,
                           TransformMappingResource::get());
      effects.emplace_back(MemoryEffects::Free::get(), operand,
                           TransformMappingResource::get());
    }
    for (Value result : this->getOperation()->getResults()) {
      effects.emplace_back(MemoryEffects::Allocate::get(), result,
                           TransformMappingResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), result,
                           TransformMappingResource::get());
    }
    effects.emplace_back(MemoryEffects::Read::get(), PayloadIRResource::get());
    effects.emplace_back(MemoryEffects::Write::get(), PayloadIRResource::get());
  }

  /// Checks that the op matches the expectations of this trait.
  static LogicalResult verifyTrait(Operation *op) {
    if (!op->getName().getInterface<MemoryEffectOpInterface>()) {
      op->emitError()
          << "FunctionalStyleTransformOpTrait should only be attached to ops "
             "that implement MemoryEffectOpInterface";
    }
    return success();
  }
};

/// Trait implementing the MemoryEffectOpInterface for single-operand
/// single-result operations that use their operand without consuming and
/// without modifying the Payload IR to produce a new handle.
template <typename OpTy>
class NavigationTransformOpTrait
    : public OpTrait::TraitBase<OpTy, NavigationTransformOpTrait> {
public:
  /// This op produces handles to the Payload IR without consuming the original
  /// handles and without modifying the IR itself.
  void getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
    effects.emplace_back(MemoryEffects::Read::get(),
                         this->getOperation()->getOperand(0),
                         TransformMappingResource::get());
    effects.emplace_back(MemoryEffects::Allocate::get(),
                         this->getOperation()->getResult(0),
                         TransformMappingResource::get());
    effects.emplace_back(MemoryEffects::Write::get(),
                         this->getOperation()->getResult(0),
                         TransformMappingResource::get());
    effects.emplace_back(MemoryEffects::Read::get(), PayloadIRResource::get());
  }

  /// Checks that the op matches the expectation of this trait.
  static LogicalResult verifyTrait(Operation *op) {
    static_assert(OpTy::template hasTrait<OpTrait::OneOperand>(),
                  "expected single-operand op");
    static_assert(OpTy::template hasTrait<OpTrait::OneResult>(),
                  "expected single-result op");
    if (!op->getName().getInterface<MemoryEffectOpInterface>()) {
      op->emitError() << "NavigationTransformOpTrait should only be attached "
                         "to ops that implement MemoryEffectOpInterface";
    }
    return success();
  }
};

} // namespace transform
} // namespace mlir

#include "mlir/Dialect/Transform/IR/TransformInterfaces.h.inc"

namespace mlir {
namespace transform {
namespace detail {
/// Appends `result` to the vector assuming it corresponds to the success state
/// in `FailureOr<convertible-to-Operation*>`. If `result` is just a
/// `LogicalResult`, does nothing.
template <typename Ty>
std::enable_if_t<std::is_same<Ty, LogicalResult>::value, LogicalResult>
appendTransformResultToVector(Ty result,
                              SmallVectorImpl<Operation *> &results) {
  return result;
}
template <typename Ty>
std::enable_if_t<!std::is_same<Ty, LogicalResult>::value, LogicalResult>
appendTransformResultToVector(Ty result,
                              SmallVectorImpl<Operation *> &results) {
  static_assert(
      std::is_convertible<typename Ty::value_type, Operation *>::value,
      "expected transform function to return operations");
  if (failed(result))
    return failure();

  results.push_back(*result);
  return success();
}

/// Applies a one-to-one transform to each of the given targets. Puts the
/// results of transforms, if any, in `results` in the same order. Fails if any
/// of the application fails. Individual transforms must be callable with
/// one of the following signatures:
///   - FailureOr<convertible-to-Operation*>(OpTy)
///   - LogicalResult(OpTy)
/// where OpTy is either
///   - Operation *, in which case the transform is always applied;
///   - a concrete Op class, in which case a check is performed whether
///   `targets` contains operations of the same class and a silencable failure
///   is reported if it does not.
template <typename FnTy>
DiagnosedSilencableFailure
applyTransformToEach(ArrayRef<Operation *> targets,
                     SmallVectorImpl<Operation *> &results, FnTy transform) {
  using OpTy = typename llvm::function_traits<FnTy>::template arg_t<0>;
  static_assert(std::is_convertible<OpTy, Operation *>::value,
                "expected transform function to take an operation");
  using RetTy = typename llvm::function_traits<FnTy>::result_t;
  static_assert(std::is_convertible<RetTy, LogicalResult>::value,
                "expected transform function to return LogicalResult or "
                "FailureOr<convertible-to-Operation*>");
  for (Operation *target : targets) {
    auto specificOp = dyn_cast<OpTy>(target);
    if (!specificOp) {
      Diagnostic diag(target->getLoc(), DiagnosticSeverity::Error);
      diag << "attempted to apply transform to the wrong op kind";
      return DiagnosedSilencableFailure::silencableFailure(std::move(diag));
    }

    auto result = transform(specificOp);
    if (failed(appendTransformResultToVector(result, results)))
      return DiagnosedSilencableFailure::definiteFailure();
  }
  return DiagnosedSilencableFailure::success();
}
} // namespace detail
} // namespace transform
} // namespace mlir

template <typename OpTy>
mlir::DiagnosedSilencableFailure
mlir::transform::TransformEachOpTrait<OpTy>::apply(
    TransformResults &transformResults, TransformState &state) {
  using TransformOpType = typename llvm::function_traits<
      decltype(&OpTy::applyToOne)>::template arg_t<0>;
  ArrayRef<Operation *> targets =
      state.getPayloadOps(this->getOperation()->getOperand(0));
  SmallVector<Operation *> results;
  DiagnosedSilencableFailure result = detail::applyTransformToEach(
      targets, results, [&](TransformOpType specificOp) {
        return static_cast<OpTy *>(this)->applyToOne(specificOp);
      });
  if (!result.succeeded())
    return result;

  if (OpTy::template hasTrait<OpTrait::OneResult>()) {
    transformResults.set(
        this->getOperation()->getResult(0).template cast<OpResult>(), results);
  }
  return DiagnosedSilencableFailure::success();
}

template <typename OpTy>
mlir::LogicalResult
mlir::transform::TransformEachOpTrait<OpTy>::verifyTrait(Operation *op) {
  static_assert(OpTy::template hasTrait<OpTrait::OneOperand>(),
                "expected single-operand op");
  static_assert(OpTy::template hasTrait<OpTrait::OneResult>() ||
                    OpTy::template hasTrait<OpTrait::ZeroResults>(),
                "expected zero- or single-result op");
  if (!op->getName().getInterface<TransformOpInterface>()) {
    return op->emitError() << "TransformEachOpTrait should only be attached to "
                              "ops that implement TransformOpInterface";
  }

  return success();
}

#endif // DIALECT_TRANSFORM_IR_TRANSFORMINTERFACES_H
