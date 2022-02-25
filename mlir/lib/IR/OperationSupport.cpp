//===- OperationSupport.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains out-of-line implementations of the support types that
// Operation and related classes build on top of.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/BitVector.h"
#include <numeric>

using namespace mlir;

//===----------------------------------------------------------------------===//
// NamedAttrList
//===----------------------------------------------------------------------===//

NamedAttrList::NamedAttrList(ArrayRef<NamedAttribute> attributes) {
  assign(attributes.begin(), attributes.end());
}

NamedAttrList::NamedAttrList(DictionaryAttr attributes)
    : NamedAttrList(attributes ? attributes.getValue()
                               : ArrayRef<NamedAttribute>()) {
  dictionarySorted.setPointerAndInt(attributes, true);
}

NamedAttrList::NamedAttrList(const_iterator in_start, const_iterator in_end) {
  assign(in_start, in_end);
}

ArrayRef<NamedAttribute> NamedAttrList::getAttrs() const { return attrs; }

Optional<NamedAttribute> NamedAttrList::findDuplicate() const {
  Optional<NamedAttribute> duplicate =
      DictionaryAttr::findDuplicate(attrs, isSorted());
  // DictionaryAttr::findDuplicate will sort the list, so reset the sorted
  // state.
  if (!isSorted())
    dictionarySorted.setPointerAndInt(nullptr, true);
  return duplicate;
}

DictionaryAttr NamedAttrList::getDictionary(MLIRContext *context) const {
  if (!isSorted()) {
    DictionaryAttr::sortInPlace(attrs);
    dictionarySorted.setPointerAndInt(nullptr, true);
  }
  if (!dictionarySorted.getPointer())
    dictionarySorted.setPointer(DictionaryAttr::getWithSorted(context, attrs));
  return dictionarySorted.getPointer().cast<DictionaryAttr>();
}

/// Add an attribute with the specified name.
void NamedAttrList::append(StringRef name, Attribute attr) {
  append(Identifier::get(name, attr.getContext()), attr);
}

/// Replaces the attributes with new list of attributes.
void NamedAttrList::assign(const_iterator in_start, const_iterator in_end) {
  DictionaryAttr::sort(ArrayRef<NamedAttribute>{in_start, in_end}, attrs);
  dictionarySorted.setPointerAndInt(nullptr, true);
}

void NamedAttrList::push_back(NamedAttribute newAttribute) {
  assert(newAttribute.second && "unexpected null attribute");
  if (isSorted())
    dictionarySorted.setInt(
        attrs.empty() ||
        strcmp(attrs.back().first.data(), newAttribute.first.data()) < 0);
  dictionarySorted.setPointer(nullptr);
  attrs.push_back(newAttribute);
}

/// Helper function to find attribute in possible sorted vector of
/// NamedAttributes.
template <typename T>
static auto *findAttr(SmallVectorImpl<NamedAttribute> &attrs, T name,
                      bool sorted) {
  if (!sorted) {
    return llvm::find_if(
        attrs, [name](NamedAttribute attr) { return attr.first == name; });
  }

  auto *it = llvm::lower_bound(attrs, name);
  if (it == attrs.end() || it->first != name)
    return attrs.end();
  return it;
}

/// Return the specified attribute if present, null otherwise.
Attribute NamedAttrList::get(StringRef name) const {
  auto *it = findAttr(attrs, name, isSorted());
  return it != attrs.end() ? it->second : nullptr;
}

/// Return the specified attribute if present, null otherwise.
Attribute NamedAttrList::get(Identifier name) const {
  auto *it = findAttr(attrs, name, isSorted());
  return it != attrs.end() ? it->second : nullptr;
}

/// Return the specified named attribute if present, None otherwise.
Optional<NamedAttribute> NamedAttrList::getNamed(StringRef name) const {
  auto *it = findAttr(attrs, name, isSorted());
  return it != attrs.end() ? *it : Optional<NamedAttribute>();
}
Optional<NamedAttribute> NamedAttrList::getNamed(Identifier name) const {
  auto *it = findAttr(attrs, name, isSorted());
  return it != attrs.end() ? *it : Optional<NamedAttribute>();
}

/// If the an attribute exists with the specified name, change it to the new
/// value.  Otherwise, add a new attribute with the specified name/value.
Attribute NamedAttrList::set(Identifier name, Attribute value) {
  assert(value && "attributes may never be null");

  // Look for an existing value for the given name, and set it in-place.
  auto *it = findAttr(attrs, name, isSorted());
  if (it != attrs.end()) {
    // Only update if the value is different from the existing.
    Attribute oldValue = it->second;
    if (oldValue != value) {
      dictionarySorted.setPointer(nullptr);
      it->second = value;
    }
    return oldValue;
  }

  // Otherwise, insert the new attribute into its sorted position.
  it = llvm::lower_bound(attrs, name);
  dictionarySorted.setPointer(nullptr);
  attrs.insert(it, {name, value});
  return Attribute();
}
Attribute NamedAttrList::set(StringRef name, Attribute value) {
  assert(value && "setting null attribute not supported");
  return set(mlir::Identifier::get(name, value.getContext()), value);
}

Attribute
NamedAttrList::eraseImpl(SmallVectorImpl<NamedAttribute>::iterator it) {
  if (it == attrs.end())
    return nullptr;

  // Erasing does not affect the sorted property.
  Attribute attr = it->second;
  attrs.erase(it);
  dictionarySorted.setPointer(nullptr);
  return attr;
}

Attribute NamedAttrList::erase(Identifier name) {
  return eraseImpl(findAttr(attrs, name, isSorted()));
}

Attribute NamedAttrList::erase(StringRef name) {
  return eraseImpl(findAttr(attrs, name, isSorted()));
}

NamedAttrList &
NamedAttrList::operator=(const SmallVectorImpl<NamedAttribute> &rhs) {
  assign(rhs.begin(), rhs.end());
  return *this;
}

NamedAttrList::operator ArrayRef<NamedAttribute>() const { return attrs; }

//===----------------------------------------------------------------------===//
// OperationState
//===----------------------------------------------------------------------===//

OperationState::OperationState(Location location, StringRef name)
    : location(location), name(name, location->getContext()) {}

OperationState::OperationState(Location location, OperationName name)
    : location(location), name(name) {}

OperationState::OperationState(Location location, StringRef name,
                               ValueRange operands, TypeRange types,
                               ArrayRef<NamedAttribute> attributes,
                               BlockRange successors,
                               MutableArrayRef<std::unique_ptr<Region>> regions)
    : location(location), name(name, location->getContext()),
      operands(operands.begin(), operands.end()),
      types(types.begin(), types.end()),
      attributes(attributes.begin(), attributes.end()),
      successors(successors.begin(), successors.end()) {
  for (std::unique_ptr<Region> &r : regions)
    this->regions.push_back(std::move(r));
}

void OperationState::addOperands(ValueRange newOperands) {
  operands.append(newOperands.begin(), newOperands.end());
}

void OperationState::addSuccessors(BlockRange newSuccessors) {
  successors.append(newSuccessors.begin(), newSuccessors.end());
}

Region *OperationState::addRegion() {
  regions.emplace_back(new Region);
  return regions.back().get();
}

void OperationState::addRegion(std::unique_ptr<Region> &&region) {
  regions.push_back(std::move(region));
}

void OperationState::addRegions(
    MutableArrayRef<std::unique_ptr<Region>> regions) {
  for (std::unique_ptr<Region> &region : regions)
    addRegion(std::move(region));
}

//===----------------------------------------------------------------------===//
// OperandStorage
//===----------------------------------------------------------------------===//

detail::OperandStorage::OperandStorage(Operation *owner, ValueRange values)
    : inlineStorage() {
  auto &inlineStorage = getInlineStorage();
  inlineStorage.numOperands = inlineStorage.capacity = values.size();
  auto *operandPtrBegin = getTrailingObjects<OpOperand>();
  for (unsigned i = 0, e = inlineStorage.numOperands; i < e; ++i)
    new (&operandPtrBegin[i]) OpOperand(owner, values[i]);
}

detail::OperandStorage::~OperandStorage() {
  // Destruct the current storage container.
  if (isDynamicStorage()) {
    TrailingOperandStorage &storage = getDynamicStorage();
    storage.~TrailingOperandStorage();
    // Work around -Wfree-nonheap-object false positive fixed by D102728.
    auto *mem = &storage;
    free(mem);
  } else {
    getInlineStorage().~TrailingOperandStorage();
  }
}

/// Replace the operands contained in the storage with the ones provided in
/// 'values'.
void detail::OperandStorage::setOperands(Operation *owner, ValueRange values) {
  MutableArrayRef<OpOperand> storageOperands = resize(owner, values.size());
  for (unsigned i = 0, e = values.size(); i != e; ++i)
    storageOperands[i].set(values[i]);
}

/// Replace the operands beginning at 'start' and ending at 'start' + 'length'
/// with the ones provided in 'operands'. 'operands' may be smaller or larger
/// than the range pointed to by 'start'+'length'.
void detail::OperandStorage::setOperands(Operation *owner, unsigned start,
                                         unsigned length, ValueRange operands) {
  // If the new size is the same, we can update inplace.
  unsigned newSize = operands.size();
  if (newSize == length) {
    MutableArrayRef<OpOperand> storageOperands = getOperands();
    for (unsigned i = 0, e = length; i != e; ++i)
      storageOperands[start + i].set(operands[i]);
    return;
  }
  // If the new size is greater, remove the extra operands and set the rest
  // inplace.
  if (newSize < length) {
    eraseOperands(start + operands.size(), length - newSize);
    setOperands(owner, start, newSize, operands);
    return;
  }
  // Otherwise, the new size is greater so we need to grow the storage.
  auto storageOperands = resize(owner, size() + (newSize - length));

  // Shift operands to the right to make space for the new operands.
  unsigned rotateSize = storageOperands.size() - (start + length);
  auto rbegin = storageOperands.rbegin();
  std::rotate(rbegin, std::next(rbegin, newSize - length), rbegin + rotateSize);

  // Update the operands inplace.
  for (unsigned i = 0, e = operands.size(); i != e; ++i)
    storageOperands[start + i].set(operands[i]);
}

/// Erase an operand held by the storage.
void detail::OperandStorage::eraseOperands(unsigned start, unsigned length) {
  TrailingOperandStorage &storage = getStorage();
  MutableArrayRef<OpOperand> operands = storage.getOperands();
  assert((start + length) <= operands.size());
  storage.numOperands -= length;

  // Shift all operands down if the operand to remove is not at the end.
  if (start != storage.numOperands) {
    auto *indexIt = std::next(operands.begin(), start);
    std::rotate(indexIt, std::next(indexIt, length), operands.end());
  }
  for (unsigned i = 0; i != length; ++i)
    operands[storage.numOperands + i].~OpOperand();
}

void detail::OperandStorage::eraseOperands(
    const llvm::BitVector &eraseIndices) {
  TrailingOperandStorage &storage = getStorage();
  MutableArrayRef<OpOperand> operands = storage.getOperands();
  assert(eraseIndices.size() == operands.size());

  // Check that at least one operand is erased.
  int firstErasedIndice = eraseIndices.find_first();
  if (firstErasedIndice == -1)
    return;

  // Shift all of the removed operands to the end, and destroy them.
  storage.numOperands = firstErasedIndice;
  for (unsigned i = firstErasedIndice + 1, e = operands.size(); i < e; ++i)
    if (!eraseIndices.test(i))
      operands[storage.numOperands++] = std::move(operands[i]);
  for (OpOperand &operand : operands.drop_front(storage.numOperands))
    operand.~OpOperand();
}

/// Resize the storage to the given size. Returns the array containing the new
/// operands.
MutableArrayRef<OpOperand> detail::OperandStorage::resize(Operation *owner,
                                                          unsigned newSize) {
  TrailingOperandStorage &storage = getStorage();

  // If the number of operands is less than or equal to the current amount, we
  // can just update in place.
  unsigned &numOperands = storage.numOperands;
  MutableArrayRef<OpOperand> operands = storage.getOperands();
  if (newSize <= numOperands) {
    // If the number of new size is less than the current, remove any extra
    // operands.
    for (unsigned i = newSize; i != numOperands; ++i)
      operands[i].~OpOperand();
    numOperands = newSize;
    return operands.take_front(newSize);
  }

  // If the new size is within the original inline capacity, grow inplace.
  if (newSize <= storage.capacity) {
    OpOperand *opBegin = operands.data();
    for (unsigned e = newSize; numOperands != e; ++numOperands)
      new (&opBegin[numOperands]) OpOperand(owner);
    return MutableArrayRef<OpOperand>(opBegin, newSize);
  }

  // Otherwise, we need to allocate a new storage.
  unsigned newCapacity =
      std::max(unsigned(llvm::NextPowerOf2(storage.capacity + 2)), newSize);
  auto *newStorageMem =
      malloc(TrailingOperandStorage::totalSizeToAlloc<OpOperand>(newCapacity));
  auto *newStorage = ::new (newStorageMem) TrailingOperandStorage();
  newStorage->numOperands = newSize;
  newStorage->capacity = newCapacity;

  // Move the current operands to the new storage.
  MutableArrayRef<OpOperand> newOperands = newStorage->getOperands();
  std::uninitialized_copy(std::make_move_iterator(operands.begin()),
                          std::make_move_iterator(operands.end()),
                          newOperands.begin());

  // Destroy the original operands.
  for (auto &operand : operands)
    operand.~OpOperand();

  // Initialize any new operands.
  for (unsigned e = newSize; numOperands != e; ++numOperands)
    new (&newOperands[numOperands]) OpOperand(owner);

  // If the current storage is also dynamic, free it.
  if (isDynamicStorage()) {
    // Work around -Wfree-nonheap-object false positive fixed by D102728.
    auto *mem = &storage;
    free(mem);
  }

  // Update the storage representation to use the new dynamic storage.
  dynamicStorage.setPointerAndInt(newStorage, true);
  return newOperands;
}

//===----------------------------------------------------------------------===//
// Operation Value-Iterators
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// OperandRange

OperandRange::OperandRange(Operation *op)
    : OperandRange(op->getOpOperands().data(), op->getNumOperands()) {}

unsigned OperandRange::getBeginOperandIndex() const {
  assert(!empty() && "range must not be empty");
  return base->getOperandNumber();
}

OperandRangeRange OperandRange::split(ElementsAttr segmentSizes) const {
  return OperandRangeRange(*this, segmentSizes);
}

//===----------------------------------------------------------------------===//
// OperandRangeRange

OperandRangeRange::OperandRangeRange(OperandRange operands,
                                     Attribute operandSegments)
    : OperandRangeRange(OwnerT(operands.getBase(), operandSegments), 0,
                        operandSegments.cast<DenseElementsAttr>().size()) {}

OperandRange OperandRangeRange::join() const {
  const OwnerT &owner = getBase();
  auto sizeData = owner.second.cast<DenseElementsAttr>().getValues<uint32_t>();
  return OperandRange(owner.first,
                      std::accumulate(sizeData.begin(), sizeData.end(), 0));
}

OperandRange OperandRangeRange::dereference(const OwnerT &object,
                                            ptrdiff_t index) {
  auto sizeData = object.second.cast<DenseElementsAttr>().getValues<uint32_t>();
  uint32_t startIndex =
      std::accumulate(sizeData.begin(), sizeData.begin() + index, 0);
  return OperandRange(object.first + startIndex, *(sizeData.begin() + index));
}

//===----------------------------------------------------------------------===//
// MutableOperandRange

/// Construct a new mutable range from the given operand, operand start index,
/// and range length.
MutableOperandRange::MutableOperandRange(
    Operation *owner, unsigned start, unsigned length,
    ArrayRef<OperandSegment> operandSegments)
    : owner(owner), start(start), length(length),
      operandSegments(operandSegments.begin(), operandSegments.end()) {
  assert((start + length) <= owner->getNumOperands() && "invalid range");
}
MutableOperandRange::MutableOperandRange(Operation *owner)
    : MutableOperandRange(owner, /*start=*/0, owner->getNumOperands()) {}

/// Slice this range into a sub range, with the additional operand segment.
MutableOperandRange
MutableOperandRange::slice(unsigned subStart, unsigned subLen,
                           Optional<OperandSegment> segment) const {
  assert((subStart + subLen) <= length && "invalid sub-range");
  MutableOperandRange subSlice(owner, start + subStart, subLen,
                               operandSegments);
  if (segment)
    subSlice.operandSegments.push_back(*segment);
  return subSlice;
}

/// Append the given values to the range.
void MutableOperandRange::append(ValueRange values) {
  if (values.empty())
    return;
  owner->insertOperands(start + length, values);
  updateLength(length + values.size());
}

/// Assign this range to the given values.
void MutableOperandRange::assign(ValueRange values) {
  owner->setOperands(start, length, values);
  if (length != values.size())
    updateLength(/*newLength=*/values.size());
}

/// Assign the range to the given value.
void MutableOperandRange::assign(Value value) {
  if (length == 1) {
    owner->setOperand(start, value);
  } else {
    owner->setOperands(start, length, value);
    updateLength(/*newLength=*/1);
  }
}

/// Erase the operands within the given sub-range.
void MutableOperandRange::erase(unsigned subStart, unsigned subLen) {
  assert((subStart + subLen) <= length && "invalid sub-range");
  if (length == 0)
    return;
  owner->eraseOperands(start + subStart, subLen);
  updateLength(length - subLen);
}

/// Clear this range and erase all of the operands.
void MutableOperandRange::clear() {
  if (length != 0) {
    owner->eraseOperands(start, length);
    updateLength(/*newLength=*/0);
  }
}

/// Allow implicit conversion to an OperandRange.
MutableOperandRange::operator OperandRange() const {
  return owner->getOperands().slice(start, length);
}

MutableOperandRangeRange
MutableOperandRange::split(NamedAttribute segmentSizes) const {
  return MutableOperandRangeRange(*this, segmentSizes);
}

/// Update the length of this range to the one provided.
void MutableOperandRange::updateLength(unsigned newLength) {
  int32_t diff = int32_t(newLength) - int32_t(length);
  length = newLength;

  // Update any of the provided segment attributes.
  for (OperandSegment &segment : operandSegments) {
    auto attr = segment.second.second.cast<DenseIntElementsAttr>();
    SmallVector<int32_t, 8> segments(attr.getValues<int32_t>());
    segments[segment.first] += diff;
    segment.second.second = DenseIntElementsAttr::get(attr.getType(), segments);
    owner->setAttr(segment.second.first, segment.second.second);
  }
}

//===----------------------------------------------------------------------===//
// MutableOperandRangeRange

MutableOperandRangeRange::MutableOperandRangeRange(
    const MutableOperandRange &operands, NamedAttribute operandSegmentAttr)
    : MutableOperandRangeRange(
          OwnerT(operands, operandSegmentAttr), 0,
          operandSegmentAttr.second.cast<DenseElementsAttr>().size()) {}

MutableOperandRange MutableOperandRangeRange::join() const {
  return getBase().first;
}

MutableOperandRangeRange::operator OperandRangeRange() const {
  return OperandRangeRange(getBase().first,
                           getBase().second.second.cast<DenseElementsAttr>());
}

MutableOperandRange MutableOperandRangeRange::dereference(const OwnerT &object,
                                                          ptrdiff_t index) {
  auto sizeData =
      object.second.second.cast<DenseElementsAttr>().getValues<uint32_t>();
  uint32_t startIndex =
      std::accumulate(sizeData.begin(), sizeData.begin() + index, 0);
  return object.first.slice(
      startIndex, *(sizeData.begin() + index),
      MutableOperandRange::OperandSegment(index, object.second));
}

//===----------------------------------------------------------------------===//
// ResultRange

ResultRange::use_range ResultRange::getUses() const {
  return {use_begin(), use_end()};
}
ResultRange::use_iterator ResultRange::use_begin() const {
  return use_iterator(*this);
}
ResultRange::use_iterator ResultRange::use_end() const {
  return use_iterator(*this, /*end=*/true);
}
ResultRange::user_range ResultRange::getUsers() {
  return {user_begin(), user_end()};
}
ResultRange::user_iterator ResultRange::user_begin() {
  return user_iterator(use_begin());
}
ResultRange::user_iterator ResultRange::user_end() {
  return user_iterator(use_end());
}

ResultRange::UseIterator::UseIterator(ResultRange results, bool end)
    : it(end ? results.end() : results.begin()), endIt(results.end()) {
  // Only initialize current use if there are results/can be uses.
  if (it != endIt)
    skipOverResultsWithNoUsers();
}

ResultRange::UseIterator &ResultRange::UseIterator::operator++() {
  // We increment over uses, if we reach the last use then move to next
  // result.
  if (use != (*it).use_end())
    ++use;
  if (use == (*it).use_end()) {
    ++it;
    skipOverResultsWithNoUsers();
  }
  return *this;
}

void ResultRange::UseIterator::skipOverResultsWithNoUsers() {
  while (it != endIt && (*it).use_empty())
    ++it;

  // If we are at the last result, then set use to first use of
  // first result (sentinel value used for end).
  if (it == endIt)
    use = {};
  else
    use = (*it).use_begin();
}

//===----------------------------------------------------------------------===//
// ValueRange

ValueRange::ValueRange(ArrayRef<Value> values)
    : ValueRange(values.data(), values.size()) {}
ValueRange::ValueRange(OperandRange values)
    : ValueRange(values.begin().getBase(), values.size()) {}
ValueRange::ValueRange(ResultRange values)
    : ValueRange(values.getBase(), values.size()) {}

/// See `llvm::detail::indexed_accessor_range_base` for details.
ValueRange::OwnerT ValueRange::offset_base(const OwnerT &owner,
                                           ptrdiff_t index) {
  if (const auto *value = owner.dyn_cast<const Value *>())
    return {value + index};
  if (auto *operand = owner.dyn_cast<OpOperand *>())
    return {operand + index};
  return owner.get<detail::OpResultImpl *>()->getNextResultAtOffset(index);
}
/// See `llvm::detail::indexed_accessor_range_base` for details.
Value ValueRange::dereference_iterator(const OwnerT &owner, ptrdiff_t index) {
  if (const auto *value = owner.dyn_cast<const Value *>())
    return value[index];
  if (auto *operand = owner.dyn_cast<OpOperand *>())
    return operand[index].get();
  return owner.get<detail::OpResultImpl *>()->getNextResultAtOffset(index);
}

//===----------------------------------------------------------------------===//
// Operation Equivalency
//===----------------------------------------------------------------------===//

llvm::hash_code OperationEquivalence::computeHash(
    Operation *op, function_ref<llvm::hash_code(Value)> hashOperands,
    function_ref<llvm::hash_code(Value)> hashResults, Flags flags) {
  // Hash operations based upon their:
  //   - Operation Name
  //   - Attributes
  //   - Result Types
  llvm::hash_code hash = llvm::hash_combine(
      op->getName(), op->getAttrDictionary(), op->getResultTypes());

  //   - Operands
  for (Value operand : op->getOperands())
    hash = llvm::hash_combine(hash, hashOperands(operand));
  //   - Operands
  for (Value result : op->getResults())
    hash = llvm::hash_combine(hash, hashResults(result));
  return hash;
}

static bool
isRegionEquivalentTo(Region *lhs, Region *rhs,
                     function_ref<LogicalResult(Value, Value)> mapOperands,
                     function_ref<LogicalResult(Value, Value)> mapResults,
                     OperationEquivalence::Flags flags) {
  DenseMap<Block *, Block *> blocksMap;
  auto blocksEquivalent = [&](Block &lBlock, Block &rBlock) {
    // Check block arguments.
    if (lBlock.getNumArguments() != rBlock.getNumArguments())
      return false;

    // Map the two blocks.
    auto insertion = blocksMap.insert({&lBlock, &rBlock});
    if (insertion.first->getSecond() != &rBlock)
      return false;

    for (auto argPair :
         llvm::zip(lBlock.getArguments(), rBlock.getArguments())) {
      Value curArg = std::get<0>(argPair);
      Value otherArg = std::get<1>(argPair);
      if (curArg.getType() != otherArg.getType())
        return false;
      if (!(flags & OperationEquivalence::IgnoreLocations) &&
          curArg.getLoc() != otherArg.getLoc())
        return false;
      // Check if this value was already mapped to another value.
      if (failed(mapOperands(curArg, otherArg)))
        return false;
    }

    auto opsEquivalent = [&](Operation &lOp, Operation &rOp) {
      // Check for op equality (recursively).
      if (!OperationEquivalence::isEquivalentTo(&lOp, &rOp, mapOperands,
                                                mapResults, flags))
        return false;
      // Check successor mapping.
      for (auto successorsPair :
           llvm::zip(lOp.getSuccessors(), rOp.getSuccessors())) {
        Block *curSuccessor = std::get<0>(successorsPair);
        Block *otherSuccessor = std::get<1>(successorsPair);
        auto insertion = blocksMap.insert({curSuccessor, otherSuccessor});
        if (insertion.first->getSecond() != otherSuccessor)
          return false;
      }
      return true;
    };
    return llvm::all_of_zip(lBlock, rBlock, opsEquivalent);
  };
  return llvm::all_of_zip(*lhs, *rhs, blocksEquivalent);
}

bool OperationEquivalence::isEquivalentTo(
    Operation *lhs, Operation *rhs,
    function_ref<LogicalResult(Value, Value)> mapOperands,
    function_ref<LogicalResult(Value, Value)> mapResults, Flags flags) {
  if (lhs == rhs)
    return true;

  // Compare the operation properties.
  if (lhs->getName() != rhs->getName() ||
      lhs->getAttrDictionary() != rhs->getAttrDictionary() ||
      lhs->getNumRegions() != rhs->getNumRegions() ||
      lhs->getNumSuccessors() != rhs->getNumSuccessors() ||
      lhs->getNumOperands() != rhs->getNumOperands() ||
      lhs->getNumResults() != rhs->getNumResults())
    return false;
  if (!(flags & IgnoreLocations) && lhs->getLoc() != rhs->getLoc())
    return false;

  auto checkValueRangeMapping =
      [](ValueRange lhs, ValueRange rhs,
         function_ref<LogicalResult(Value, Value)> mapValues) {
        for (auto operandPair : llvm::zip(lhs, rhs)) {
          Value curArg = std::get<0>(operandPair);
          Value otherArg = std::get<1>(operandPair);
          if (curArg.getType() != otherArg.getType())
            return false;
          if (failed(mapValues(curArg, otherArg)))
            return false;
        }
        return true;
      };
  // Check mapping of operands and results.
  if (!checkValueRangeMapping(lhs->getOperands(), rhs->getOperands(),
                              mapOperands))
    return false;
  if (!checkValueRangeMapping(lhs->getResults(), rhs->getResults(), mapResults))
    return false;
  for (auto regionPair : llvm::zip(lhs->getRegions(), rhs->getRegions()))
    if (!isRegionEquivalentTo(&std::get<0>(regionPair),
                              &std::get<1>(regionPair), mapOperands, mapResults,
                              flags))
      return false;
  return true;
}
