//===- Operation.h - MLIR Operation Class -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Operation class.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_OPERATION_H
#define MLIR_IR_OPERATION_H

#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Region.h"
#include "llvm/ADT/Twine.h"

namespace mlir {
/// Operation is a basic unit of execution within MLIR. Operations can
/// be nested within `Region`s held by other operations effectively forming a
/// tree. Child operations are organized into operation blocks represented by a
/// 'Block' class.
class alignas(8) Operation final
    : public llvm::ilist_node_with_parent<Operation, Block>,
      private llvm::TrailingObjects<Operation, detail::OperandStorage,
                                    BlockOperand, Region, OpOperand> {
public:
  /// Create a new Operation with the specific fields.
  static Operation *create(Location location, OperationName name,
                           TypeRange resultTypes, ValueRange operands,
                           ArrayRef<NamedAttribute> attributes,
                           BlockRange successors, unsigned numRegions);

  /// Overload of create that takes an existing DictionaryAttr to avoid
  /// unnecessarily uniquing a list of attributes.
  static Operation *create(Location location, OperationName name,
                           TypeRange resultTypes, ValueRange operands,
                           DictionaryAttr attributes, BlockRange successors,
                           unsigned numRegions);

  /// Create a new Operation from the fields stored in `state`.
  static Operation *create(const OperationState &state);

  /// Create a new Operation with the specific fields.
  static Operation *create(Location location, OperationName name,
                           TypeRange resultTypes, ValueRange operands,
                           DictionaryAttr attributes,
                           BlockRange successors = {},
                           RegionRange regions = {});

  /// The name of an operation is the key identifier for it.
  OperationName getName() { return name; }

  /// If this operation has a registered operation description, return it.
  /// Otherwise return None.
  Optional<RegisteredOperationName> getRegisteredInfo() {
    return getName().getRegisteredInfo();
  }

  /// Returns true if this operation has a registered operation description,
  /// otherwise false.
  bool isRegistered() { return getName().isRegistered(); }

  /// Remove this operation from its parent block and delete it.
  void erase();

  /// Remove the operation from its parent block, but don't delete it.
  void remove();

  /// Class encompassing various options related to cloning an operation. Users
  /// of this class should pass it to Operation's 'clone' methods.
  /// Current options include:
  /// * Whether cloning should recursively traverse into the regions of the
  ///   operation or not.
  /// * Whether cloning should also clone the operands of the operation.
  class CloneOptions {
  public:
    /// Default constructs an option with all flags set to false. That means all
    /// parts of an operation that may optionally not be cloned, are not cloned.
    CloneOptions();

    /// Constructs an instance with the clone regions and clone operands flags
    /// set accordingly.
    CloneOptions(bool cloneRegions, bool cloneOperands);

    /// Returns an instance with all flags set to true. This is the default
    /// when using the clone method and clones all parts of the operation.
    static CloneOptions all();

    /// Configures whether cloning should traverse into any of the regions of
    /// the operation. If set to true, the operation's regions are recursively
    /// cloned. If set to false, cloned operations will have the same number of
    /// regions, but they will be empty.
    /// Cloning of nested operations in the operation's regions are currently
    /// unaffected by other flags.
    CloneOptions &cloneRegions(bool enable = true);

    /// Returns whether regions of the operation should be cloned as well.
    bool shouldCloneRegions() const { return cloneRegionsFlag; }

    /// Configures whether operation' operands should be cloned. Otherwise the
    /// resulting clones will simply have zero operands.
    CloneOptions &cloneOperands(bool enable = true);

    /// Returns whether operands should be cloned as well.
    bool shouldCloneOperands() const { return cloneOperandsFlag; }

  private:
    /// Whether regions should be cloned.
    bool cloneRegionsFlag : 1;
    /// Whether operands should be cloned.
    bool cloneOperandsFlag : 1;
  };

  /// Create a deep copy of this operation, remapping any operands that use
  /// values outside of the operation using the map that is provided (leaving
  /// them alone if no entry is present).  Replaces references to cloned
  /// sub-operations to the corresponding operation that is copied, and adds
  /// those mappings to the map.
  /// Optionally, one may configure what parts of the operation to clone using
  /// the options parameter.
  ///
  /// Calling this method from multiple threads is generally safe if through the
  /// process of cloning no new uses of 'Value's from outside the operation are
  /// created. Cloning an isolated-from-above operation with no operands, such
  /// as top level function operations, is therefore always safe. Using the
  /// mapper, it is possible to avoid adding uses to outside operands by
  /// remapping them to 'Value's owned by the caller thread.
  Operation *clone(BlockAndValueMapping &mapper,
                   CloneOptions options = CloneOptions::all());
  Operation *clone(CloneOptions options = CloneOptions::all());

  /// Create a partial copy of this operation without traversing into attached
  /// regions. The new operation will have the same number of regions as the
  /// original one, but they will be left empty.
  /// Operands are remapped using `mapper` (if present), and `mapper` is updated
  /// to contain the results.
  Operation *cloneWithoutRegions(BlockAndValueMapping &mapper);

  /// Create a partial copy of this operation without traversing into attached
  /// regions. The new operation will have the same number of regions as the
  /// original one, but they will be left empty.
  Operation *cloneWithoutRegions();

  /// Returns the operation block that contains this operation.
  Block *getBlock() { return block; }

  /// Return the context this operation is associated with.
  MLIRContext *getContext() { return location->getContext(); }

  /// Return the dialect this operation is associated with, or nullptr if the
  /// associated dialect is not loaded.
  Dialect *getDialect() { return getName().getDialect(); }

  /// The source location the operation was defined or derived from.
  Location getLoc() { return location; }

  /// Set the source location the operation was defined or derived from.
  void setLoc(Location loc) { location = loc; }

  /// Returns the region to which the instruction belongs. Returns nullptr if
  /// the instruction is unlinked.
  Region *getParentRegion() { return block ? block->getParent() : nullptr; }

  /// Returns the closest surrounding operation that contains this operation
  /// or nullptr if this is a top-level operation.
  Operation *getParentOp() { return block ? block->getParentOp() : nullptr; }

  /// Return the closest surrounding parent operation that is of type 'OpTy'.
  template <typename OpTy>
  OpTy getParentOfType() {
    auto *op = this;
    while ((op = op->getParentOp()))
      if (auto parentOp = dyn_cast<OpTy>(op))
        return parentOp;
    return OpTy();
  }

  /// Returns the closest surrounding parent operation with trait `Trait`.
  template <template <typename T> class Trait>
  Operation *getParentWithTrait() {
    Operation *op = this;
    while ((op = op->getParentOp()))
      if (op->hasTrait<Trait>())
        return op;
    return nullptr;
  }

  /// Return true if this operation is a proper ancestor of the `other`
  /// operation.
  bool isProperAncestor(Operation *other);

  /// Return true if this operation is an ancestor of the `other` operation. An
  /// operation is considered as its own ancestor, use `isProperAncestor` to
  /// avoid this.
  bool isAncestor(Operation *other) {
    return this == other || isProperAncestor(other);
  }

  /// Replace any uses of 'from' with 'to' within this operation.
  void replaceUsesOfWith(Value from, Value to);

  /// Replace all uses of results of this operation with the provided 'values'.
  template <typename ValuesT>
  void replaceAllUsesWith(ValuesT &&values) {
    getResults().replaceAllUsesWith(std::forward<ValuesT>(values));
  }

  /// Destroys this operation and its subclass data.
  void destroy();

  /// This drops all operand uses from this operation, which is an essential
  /// step in breaking cyclic dependences between references when they are to
  /// be deleted.
  void dropAllReferences();

  /// Drop uses of all values defined by this operation or its nested regions.
  void dropAllDefinedValueUses();

  /// Unlink this operation from its current block and insert it right before
  /// `existingOp` which may be in the same or another block in the same
  /// function.
  void moveBefore(Operation *existingOp);

  /// Unlink this operation from its current block and insert it right before
  /// `iterator` in the specified block.
  void moveBefore(Block *block, llvm::iplist<Operation>::iterator iterator);

  /// Unlink this operation from its current block and insert it right after
  /// `existingOp` which may be in the same or another block in the same
  /// function.
  void moveAfter(Operation *existingOp);

  /// Unlink this operation from its current block and insert it right after
  /// `iterator` in the specified block.
  void moveAfter(Block *block, llvm::iplist<Operation>::iterator iterator);

  /// Given an operation 'other' that is within the same parent block, return
  /// whether the current operation is before 'other' in the operation list
  /// of the parent block.
  /// Note: This function has an average complexity of O(1), but worst case may
  /// take O(N) where N is the number of operations within the parent block.
  bool isBeforeInBlock(Operation *other);

  void print(raw_ostream &os, const OpPrintingFlags &flags = llvm::None);
  void print(raw_ostream &os, AsmState &state);
  void dump();

  //===--------------------------------------------------------------------===//
  // Operands
  //===--------------------------------------------------------------------===//

  /// Replace the current operands of this operation with the ones provided in
  /// 'operands'.
  void setOperands(ValueRange operands);

  /// Replace the operands beginning at 'start' and ending at 'start' + 'length'
  /// with the ones provided in 'operands'. 'operands' may be smaller or larger
  /// than the range pointed to by 'start'+'length'.
  void setOperands(unsigned start, unsigned length, ValueRange operands);

  /// Insert the given operands into the operand list at the given 'index'.
  void insertOperands(unsigned index, ValueRange operands);

  unsigned getNumOperands() {
    return LLVM_LIKELY(hasOperandStorage) ? getOperandStorage().size() : 0;
  }

  Value getOperand(unsigned idx) { return getOpOperand(idx).get(); }
  void setOperand(unsigned idx, Value value) {
    return getOpOperand(idx).set(value);
  }

  /// Erase the operand at position `idx`.
  void eraseOperand(unsigned idx) { eraseOperands(idx); }

  /// Erase the operands starting at position `idx` and ending at position
  /// 'idx'+'length'.
  void eraseOperands(unsigned idx, unsigned length = 1) {
    getOperandStorage().eraseOperands(idx, length);
  }

  /// Erases the operands that have their corresponding bit set in
  /// `eraseIndices` and removes them from the operand list.
  void eraseOperands(const BitVector &eraseIndices) {
    getOperandStorage().eraseOperands(eraseIndices);
  }

  // Support operand iteration.
  using operand_range = OperandRange;
  using operand_iterator = operand_range::iterator;

  operand_iterator operand_begin() { return getOperands().begin(); }
  operand_iterator operand_end() { return getOperands().end(); }

  /// Returns an iterator on the underlying Value's.
  operand_range getOperands() {
    MutableArrayRef<OpOperand> operands = getOpOperands();
    return OperandRange(operands.data(), operands.size());
  }

  MutableArrayRef<OpOperand> getOpOperands() {
    return LLVM_LIKELY(hasOperandStorage) ? getOperandStorage().getOperands()
                                          : MutableArrayRef<OpOperand>();
  }

  OpOperand &getOpOperand(unsigned idx) {
    return getOperandStorage().getOperands()[idx];
  }

  // Support operand type iteration.
  using operand_type_iterator = operand_range::type_iterator;
  using operand_type_range = operand_range::type_range;
  operand_type_iterator operand_type_begin() { return operand_begin(); }
  operand_type_iterator operand_type_end() { return operand_end(); }
  operand_type_range getOperandTypes() { return getOperands().getTypes(); }

  //===--------------------------------------------------------------------===//
  // Results
  //===--------------------------------------------------------------------===//

  /// Return the number of results held by this operation.
  unsigned getNumResults() { return numResults; }

  /// Get the 'idx'th result of this operation.
  OpResult getResult(unsigned idx) { return OpResult(getOpResultImpl(idx)); }

  /// Support result iteration.
  using result_range = ResultRange;
  using result_iterator = result_range::iterator;

  result_iterator result_begin() { return getResults().begin(); }
  result_iterator result_end() { return getResults().end(); }
  result_range getResults() {
    return numResults == 0 ? result_range(nullptr, 0)
                           : result_range(getInlineOpResult(0), numResults);
  }

  result_range getOpResults() { return getResults(); }
  OpResult getOpResult(unsigned idx) { return getResult(idx); }

  /// Support result type iteration.
  using result_type_iterator = result_range::type_iterator;
  using result_type_range = result_range::type_range;
  result_type_iterator result_type_begin() { return getResultTypes().begin(); }
  result_type_iterator result_type_end() { return getResultTypes().end(); }
  result_type_range getResultTypes() { return getResults().getTypes(); }

  //===--------------------------------------------------------------------===//
  // Attributes
  //===--------------------------------------------------------------------===//

  // Operations may optionally carry a list of attributes that associate
  // constants to names.  Attributes may be dynamically added and removed over
  // the lifetime of an operation.

  /// Return all of the attributes on this operation.
  ArrayRef<NamedAttribute> getAttrs() { return attrs.getValue(); }

  /// Return all of the attributes on this operation as a DictionaryAttr.
  DictionaryAttr getAttrDictionary() { return attrs; }

  /// Set the attribute dictionary on this operation.
  void setAttrs(DictionaryAttr newAttrs) {
    assert(newAttrs && "expected valid attribute dictionary");
    attrs = newAttrs;
  }
  void setAttrs(ArrayRef<NamedAttribute> newAttrs) {
    setAttrs(DictionaryAttr::get(getContext(), newAttrs));
  }

  /// Return the specified attribute if present, null otherwise.
  Attribute getAttr(StringAttr name) { return attrs.get(name); }
  Attribute getAttr(StringRef name) { return attrs.get(name); }

  template <typename AttrClass>
  AttrClass getAttrOfType(StringAttr name) {
    return getAttr(name).dyn_cast_or_null<AttrClass>();
  }
  template <typename AttrClass>
  AttrClass getAttrOfType(StringRef name) {
    return getAttr(name).dyn_cast_or_null<AttrClass>();
  }

  /// Return true if the operation has an attribute with the provided name,
  /// false otherwise.
  bool hasAttr(StringAttr name) { return attrs.contains(name); }
  bool hasAttr(StringRef name) { return attrs.contains(name); }
  template <typename AttrClass, typename NameT>
  bool hasAttrOfType(NameT &&name) {
    return static_cast<bool>(
        getAttrOfType<AttrClass>(std::forward<NameT>(name)));
  }

  /// If the an attribute exists with the specified name, change it to the new
  /// value. Otherwise, add a new attribute with the specified name/value.
  void setAttr(StringAttr name, Attribute value) {
    NamedAttrList attributes(attrs);
    if (attributes.set(name, value) != value)
      attrs = attributes.getDictionary(getContext());
  }
  void setAttr(StringRef name, Attribute value) {
    setAttr(StringAttr::get(getContext(), name), value);
  }

  /// Remove the attribute with the specified name if it exists. Return the
  /// attribute that was erased, or nullptr if there was no attribute with such
  /// name.
  Attribute removeAttr(StringAttr name) {
    NamedAttrList attributes(attrs);
    Attribute removedAttr = attributes.erase(name);
    if (removedAttr)
      attrs = attributes.getDictionary(getContext());
    return removedAttr;
  }
  Attribute removeAttr(StringRef name) {
    return removeAttr(StringAttr::get(getContext(), name));
  }

  /// A utility iterator that filters out non-dialect attributes.
  class dialect_attr_iterator
      : public llvm::filter_iterator<ArrayRef<NamedAttribute>::iterator,
                                     bool (*)(NamedAttribute)> {
    static bool filter(NamedAttribute attr) {
      // Dialect attributes are prefixed by the dialect name, like operations.
      return attr.getName().strref().count('.');
    }

    explicit dialect_attr_iterator(ArrayRef<NamedAttribute>::iterator it,
                                   ArrayRef<NamedAttribute>::iterator end)
        : llvm::filter_iterator<ArrayRef<NamedAttribute>::iterator,
                                bool (*)(NamedAttribute)>(it, end, &filter) {}

    // Allow access to the constructor.
    friend Operation;
  };
  using dialect_attr_range = iterator_range<dialect_attr_iterator>;

  /// Return a range corresponding to the dialect attributes for this operation.
  dialect_attr_range getDialectAttrs() {
    auto attrs = getAttrs();
    return {dialect_attr_iterator(attrs.begin(), attrs.end()),
            dialect_attr_iterator(attrs.end(), attrs.end())};
  }
  dialect_attr_iterator dialect_attr_begin() {
    auto attrs = getAttrs();
    return dialect_attr_iterator(attrs.begin(), attrs.end());
  }
  dialect_attr_iterator dialect_attr_end() {
    auto attrs = getAttrs();
    return dialect_attr_iterator(attrs.end(), attrs.end());
  }

  /// Set the dialect attributes for this operation, and preserve all dependent.
  template <typename DialectAttrT>
  void setDialectAttrs(DialectAttrT &&dialectAttrs) {
    NamedAttrList attrs;
    attrs.append(std::begin(dialectAttrs), std::end(dialectAttrs));
    for (auto attr : getAttrs())
      if (!attr.getName().strref().contains('.'))
        attrs.push_back(attr);
    setAttrs(attrs.getDictionary(getContext()));
  }

  //===--------------------------------------------------------------------===//
  // Blocks
  //===--------------------------------------------------------------------===//

  /// Returns the number of regions held by this operation.
  unsigned getNumRegions() { return numRegions; }

  /// Returns the regions held by this operation.
  MutableArrayRef<Region> getRegions() {
    auto *regions = getTrailingObjects<Region>();
    return {regions, numRegions};
  }

  /// Returns the region held by this operation at position 'index'.
  Region &getRegion(unsigned index) {
    assert(index < numRegions && "invalid region index");
    return getRegions()[index];
  }

  //===--------------------------------------------------------------------===//
  // Successors
  //===--------------------------------------------------------------------===//

  MutableArrayRef<BlockOperand> getBlockOperands() {
    return {getTrailingObjects<BlockOperand>(), numSuccs};
  }

  // Successor iteration.
  using succ_iterator = SuccessorRange::iterator;
  succ_iterator successor_begin() { return getSuccessors().begin(); }
  succ_iterator successor_end() { return getSuccessors().end(); }
  SuccessorRange getSuccessors() { return SuccessorRange(this); }

  bool hasSuccessors() { return numSuccs != 0; }
  unsigned getNumSuccessors() { return numSuccs; }

  Block *getSuccessor(unsigned index) {
    assert(index < getNumSuccessors());
    return getBlockOperands()[index].get();
  }
  void setSuccessor(Block *block, unsigned index);

  //===--------------------------------------------------------------------===//
  // Accessors for various properties of operations
  //===--------------------------------------------------------------------===//

  /// Attempt to fold this operation with the specified constant operand values
  /// - the elements in "operands" will correspond directly to the operands of
  /// the operation, but may be null if non-constant. If folding is successful,
  /// this fills in the `results` vector. If not, `results` is unspecified.
  LogicalResult fold(ArrayRef<Attribute> operands,
                     SmallVectorImpl<OpFoldResult> &results);

  /// Returns true if the operation was registered with a particular trait, e.g.
  /// hasTrait<OperandsAreSignlessIntegerLike>().
  template <template <typename T> class Trait>
  bool hasTrait() {
    return name.hasTrait<Trait>();
  }

  /// Returns true if the operation *might* have the provided trait. This
  /// means that either the operation is unregistered, or it was registered with
  /// the provide trait.
  template <template <typename T> class Trait>
  bool mightHaveTrait() {
    return name.mightHaveTrait<Trait>();
  }

  //===--------------------------------------------------------------------===//
  // Operation Walkers
  //===--------------------------------------------------------------------===//

  /// Walk the operation by calling the callback for each nested operation
  /// (including this one), block or region, depending on the callback provided.
  /// Regions, blocks and operations at the same nesting level are visited in
  /// lexicographical order. The walk order for enclosing regions, blocks and
  /// operations with respect to their nested ones is specified by 'Order'
  /// (post-order by default). A callback on a block or operation is allowed to
  /// erase that block or operation if either:
  ///   * the walk is in post-order, or
  ///   * the walk is in pre-order and the walk is skipped after the erasure.
  ///
  /// The callback method can take any of the following forms:
  ///   void(Operation*) : Walk all operations opaquely.
  ///     * op->walk([](Operation *nestedOp) { ...});
  ///   void(OpT) : Walk all operations of the given derived type.
  ///     * op->walk([](ReturnOp returnOp) { ...});
  ///   WalkResult(Operation*|OpT) : Walk operations, but allow for
  ///                                interruption/skipping.
  ///     * op->walk([](... op) {
  ///         // Skip the walk of this op based on some invariant.
  ///         if (some_invariant)
  ///           return WalkResult::skip();
  ///         // Interrupt, i.e cancel, the walk based on some invariant.
  ///         if (another_invariant)
  ///           return WalkResult::interrupt();
  ///         return WalkResult::advance();
  ///       });
  template <WalkOrder Order = WalkOrder::PostOrder, typename FnT,
            typename RetT = detail::walkResultType<FnT>>
  typename std::enable_if<
      llvm::function_traits<std::decay_t<FnT>>::num_args == 1, RetT>::type
  walk(FnT &&callback) {
    return detail::walk<Order>(this, std::forward<FnT>(callback));
  }

  /// Generic walker with a stage aware callback. Walk the operation by calling
  /// the callback for each nested operation (including this one) N+1 times,
  /// where N is the number of regions attached to that operation.
  ///
  /// The callback method can take any of the following forms:
  ///   void(Operation *, const WalkStage &) : Walk all operation opaquely
  ///     * op->walk([](Operation *nestedOp, const WalkStage &stage) { ...});
  ///   void(OpT, const WalkStage &) : Walk all operations of the given derived
  ///                                  type.
  ///     * op->walk([](ReturnOp returnOp, const WalkStage &stage) { ...});
  ///   WalkResult(Operation*|OpT, const WalkStage &stage) : Walk operations,
  ///          but allow for interruption/skipping.
  ///     * op->walk([](... op, const WalkStage &stage) {
  ///         // Skip the walk of this op based on some invariant.
  ///         if (some_invariant)
  ///           return WalkResult::skip();
  ///         // Interrupt, i.e cancel, the walk based on some invariant.
  ///         if (another_invariant)
  ///           return WalkResult::interrupt();
  ///         return WalkResult::advance();
  ///       });
  template <typename FnT, typename RetT = detail::walkResultType<FnT>>
  typename std::enable_if<
      llvm::function_traits<std::decay_t<FnT>>::num_args == 2, RetT>::type
  walk(FnT &&callback) {
    return detail::walk(this, std::forward<FnT>(callback));
  }

  //===--------------------------------------------------------------------===//
  // Uses
  //===--------------------------------------------------------------------===//

  /// Drop all uses of results of this operation.
  void dropAllUses() {
    for (OpResult result : getOpResults())
      result.dropAllUses();
  }

  using use_iterator = result_range::use_iterator;
  using use_range = result_range::use_range;

  use_iterator use_begin() { return getResults().use_begin(); }
  use_iterator use_end() { return getResults().use_end(); }

  /// Returns a range of all uses, which is useful for iterating over all uses.
  use_range getUses() { return getResults().getUses(); }

  /// Returns true if this operation has exactly one use.
  bool hasOneUse() { return llvm::hasSingleElement(getUses()); }

  /// Returns true if this operation has no uses.
  bool use_empty() { return getResults().use_empty(); }

  /// Returns true if the results of this operation are used outside of the
  /// given block.
  bool isUsedOutsideOfBlock(Block *block) {
    return llvm::any_of(getOpResults(), [block](OpResult result) {
      return result.isUsedOutsideOfBlock(block);
    });
  }

  //===--------------------------------------------------------------------===//
  // Users
  //===--------------------------------------------------------------------===//

  using user_iterator = ValueUserIterator<use_iterator, OpOperand>;
  using user_range = iterator_range<user_iterator>;

  user_iterator user_begin() { return user_iterator(use_begin()); }
  user_iterator user_end() { return user_iterator(use_end()); }

  /// Returns a range of all users.
  user_range getUsers() { return {user_begin(), user_end()}; }

  //===--------------------------------------------------------------------===//
  // Other
  //===--------------------------------------------------------------------===//

  /// Emit an error with the op name prefixed, like "'dim' op " which is
  /// convenient for verifiers.
  InFlightDiagnostic emitOpError(const Twine &message = {});

  /// Emit an error about fatal conditions with this operation, reporting up to
  /// any diagnostic handlers that may be listening.
  InFlightDiagnostic emitError(const Twine &message = {});

  /// Emit a warning about this operation, reporting up to any diagnostic
  /// handlers that may be listening.
  InFlightDiagnostic emitWarning(const Twine &message = {});

  /// Emit a remark about this operation, reporting up to any diagnostic
  /// handlers that may be listening.
  InFlightDiagnostic emitRemark(const Twine &message = {});

private:
  //===--------------------------------------------------------------------===//
  // Ordering
  //===--------------------------------------------------------------------===//

  /// This value represents an invalid index ordering for an operation within a
  /// block.
  static constexpr unsigned kInvalidOrderIdx = -1;

  /// This value represents the stride to use when computing a new order for an
  /// operation.
  static constexpr unsigned kOrderStride = 5;

  /// Update the order index of this operation of this operation if necessary,
  /// potentially recomputing the order of the parent block.
  void updateOrderIfNecessary();

  /// Returns true if this operation has a valid order.
  bool hasValidOrder() { return orderIndex != kInvalidOrderIdx; }

private:
  Operation(Location location, OperationName name, unsigned numResults,
            unsigned numSuccessors, unsigned numRegions,
            DictionaryAttr attributes, bool hasOperandStorage);

  // Operations are deleted through the destroy() member because they are
  // allocated with malloc.
  ~Operation();

  /// Returns the additional size necessary for allocating the given objects
  /// before an Operation in-memory.
  static size_t prefixAllocSize(unsigned numOutOfLineResults,
                                unsigned numInlineResults) {
    return sizeof(detail::OutOfLineOpResult) * numOutOfLineResults +
           sizeof(detail::InlineOpResult) * numInlineResults;
  }
  /// Returns the additional size allocated before this Operation in-memory.
  size_t prefixAllocSize() {
    unsigned numResults = getNumResults();
    unsigned numOutOfLineResults = OpResult::getNumTrailing(numResults);
    unsigned numInlineResults = OpResult::getNumInline(numResults);
    return prefixAllocSize(numOutOfLineResults, numInlineResults);
  }

  /// Returns the operand storage object.
  detail::OperandStorage &getOperandStorage() {
    assert(hasOperandStorage && "expected operation to have operand storage");
    return *getTrailingObjects<detail::OperandStorage>();
  }

  /// Returns a pointer to the use list for the given out-of-line result.
  detail::OutOfLineOpResult *getOutOfLineOpResult(unsigned resultNumber) {
    // Out-of-line results are stored in reverse order after (before in memory)
    // the inline results.
    return reinterpret_cast<detail::OutOfLineOpResult *>(getInlineOpResult(
               detail::OpResultImpl::getMaxInlineResults() - 1)) -
           ++resultNumber;
  }

  /// Returns a pointer to the use list for the given inline result.
  detail::InlineOpResult *getInlineOpResult(unsigned resultNumber) {
    // Inline results are stored in reverse order before the operation in
    // memory.
    return reinterpret_cast<detail::InlineOpResult *>(this) - ++resultNumber;
  }

  /// Returns a pointer to the use list for the given result, which may be
  /// either inline or out-of-line.
  detail::OpResultImpl *getOpResultImpl(unsigned resultNumber) {
    unsigned maxInlineResults = detail::OpResultImpl::getMaxInlineResults();
    if (resultNumber < maxInlineResults)
      return getInlineOpResult(resultNumber);
    return getOutOfLineOpResult(resultNumber - maxInlineResults);
  }

  /// Provide a 'getParent' method for ilist_node_with_parent methods.
  /// We mark it as a const function because ilist_node_with_parent specifically
  /// requires a 'getParent() const' method. Once ilist_node removes this
  /// constraint, we should drop the const to fit the rest of the MLIR const
  /// model.
  Block *getParent() const { return block; }

  /// The operation block that contains this operation.
  Block *block = nullptr;

  /// This holds information about the source location the operation was defined
  /// or derived from.
  Location location;

  /// Relative order of this operation in its parent block. Used for
  /// O(1) local dominance checks between operations.
  mutable unsigned orderIndex = 0;

  const unsigned numResults;
  const unsigned numSuccs;
  const unsigned numRegions : 31;

  /// This bit signals whether this operation has an operand storage or not. The
  /// operand storage may be elided for operations that are known to never have
  /// operands.
  bool hasOperandStorage : 1;

  /// This holds the name of the operation.
  OperationName name;

  /// This holds general named attributes for the operation.
  DictionaryAttr attrs;

  // allow ilist_traits access to 'block' field.
  friend struct llvm::ilist_traits<Operation>;

  // allow block to access the 'orderIndex' field.
  friend class Block;

  // allow value to access the 'ResultStorage' methods.
  friend class Value;

  // allow ilist_node_with_parent to access the 'getParent' method.
  friend class llvm::ilist_node_with_parent<Operation, Block>;

  // This stuff is used by the TrailingObjects template.
  friend llvm::TrailingObjects<Operation, detail::OperandStorage, BlockOperand,
                               Region, OpOperand>;
  size_t numTrailingObjects(OverloadToken<detail::OperandStorage>) const {
    return hasOperandStorage ? 1 : 0;
  }
  size_t numTrailingObjects(OverloadToken<BlockOperand>) const {
    return numSuccs;
  }
  size_t numTrailingObjects(OverloadToken<Region>) const { return numRegions; }
};

inline raw_ostream &operator<<(raw_ostream &os, const Operation &op) {
  const_cast<Operation &>(op).print(os, OpPrintingFlags().useLocalScope());
  return os;
}

} // namespace mlir

namespace llvm {
/// Cast from an (const) Operation * to a derived operation type.
template <typename T>
struct CastInfo<T, ::mlir::Operation *>
    : public ValueFromPointerCast<T, ::mlir::Operation,
                                  CastInfo<T, ::mlir::Operation *>> {
  static bool isPossible(::mlir::Operation *op) { return T::classof(op); }
};
template <typename T>
struct CastInfo<T, const ::mlir::Operation *>
    : public ConstStrippingForwardingCast<T, const ::mlir::Operation *,
                                          CastInfo<T, ::mlir::Operation *>> {};

/// Cast from an (const) Operation & to a derived operation type.
template <typename T>
struct CastInfo<T, ::mlir::Operation>
    : public NullableValueCastFailed<T>,
      public DefaultDoCastIfPossible<T, ::mlir::Operation &,
                                     CastInfo<T, ::mlir::Operation>> {
  // Provide isPossible here because here we have the const-stripping from
  // ConstStrippingCast.
  static bool isPossible(::mlir::Operation &val) { return T::classof(&val); }
  static T doCast(::mlir::Operation &val) { return T(&val); }
};
template <typename T>
struct CastInfo<T, const ::mlir::Operation>
    : public ConstStrippingForwardingCast<T, const ::mlir::Operation,
                                          CastInfo<T, ::mlir::Operation>> {};

/// Cast (const) Operation * to itself. This is helpful to avoid SFINAE in
/// templated implementations that should work on both base and derived
/// operation types.
template <>
struct CastInfo<::mlir::Operation *, ::mlir::Operation *>
    : public NullableValueCastFailed<::mlir::Operation *>,
      public DefaultDoCastIfPossible<
          ::mlir::Operation *, ::mlir::Operation *,
          CastInfo<::mlir::Operation *, ::mlir::Operation *>> {
  static bool isPossible(::mlir::Operation *op) { return true; }
  static ::mlir::Operation *doCast(::mlir::Operation *op) { return op; }
};
template <>
struct CastInfo<const ::mlir::Operation *, const ::mlir::Operation *>
    : public ConstStrippingForwardingCast<
          const ::mlir::Operation *, const ::mlir::Operation *,
          CastInfo<::mlir::Operation *, ::mlir::Operation *>> {};
} // namespace llvm

#endif // MLIR_IR_OPERATION_H
