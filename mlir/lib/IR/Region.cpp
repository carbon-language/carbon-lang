//===- Region.cpp - MLIR Region Class -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Region.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Operation.h"
using namespace mlir;

Region::Region(Operation *container) : container(container) {}

Region::~Region() {
  // Operations may have cyclic references, which need to be dropped before we
  // can start deleting them.
  dropAllReferences();
}

/// Return the context this region is inserted in. The region must have a valid
/// parent container.
MLIRContext *Region::getContext() {
  assert(container && "region is not attached to a container");
  return container->getContext();
}

/// Return a location for this region. This is the location attached to the
/// parent container. The region must have a valid parent container.
Location Region::getLoc() {
  assert(container && "region is not attached to a container");
  return container->getLoc();
}

/// Return a range containing the types of the arguments for this region.
auto Region::getArgumentTypes() -> ValueTypeRange<BlockArgListType> {
  return ValueTypeRange<BlockArgListType>(getArguments());
}

/// Add one argument to the argument list for each type specified in the list.
iterator_range<Region::args_iterator> Region::addArguments(TypeRange types) {
  return front().addArguments(types);
}

Region *Region::getParentRegion() {
  assert(container && "region is not attached to a container");
  return container->getParentRegion();
}

bool Region::isProperAncestor(Region *other) {
  if (this == other)
    return false;

  while ((other = other->getParentRegion())) {
    if (this == other)
      return true;
  }
  return false;
}

/// Return the number of this region in the parent operation.
unsigned Region::getRegionNumber() {
  // Regions are always stored consecutively, so use pointer subtraction to
  // figure out what number this is.
  return this - &getParentOp()->getRegions()[0];
}

/// Clone the internal blocks from this region into `dest`. Any
/// cloned blocks are appended to the back of dest.
void Region::cloneInto(Region *dest, BlockAndValueMapping &mapper) {
  assert(dest && "expected valid region to clone into");
  cloneInto(dest, dest->end(), mapper);
}

/// Clone this region into 'dest' before the given position in 'dest'.
void Region::cloneInto(Region *dest, Region::iterator destPos,
                       BlockAndValueMapping &mapper) {
  assert(dest && "expected valid region to clone into");
  assert(this != dest && "cannot clone region into itself");

  // If the list is empty there is nothing to clone.
  if (empty())
    return;

  for (Block &block : *this) {
    Block *newBlock = new Block();
    mapper.map(&block, newBlock);

    // Clone the block arguments. The user might be deleting arguments to the
    // block by specifying them in the mapper. If so, we don't add the
    // argument to the cloned block.
    for (auto arg : block.getArguments())
      if (!mapper.contains(arg))
        mapper.map(arg, newBlock->addArgument(arg.getType()));

    // Clone and remap the operations within this block.
    for (auto &op : block)
      newBlock->push_back(op.clone(mapper));

    dest->getBlocks().insert(destPos, newBlock);
  }

  // Now that each of the blocks have been cloned, go through and remap the
  // operands of each of the operations.
  auto remapOperands = [&](Operation *op) {
    for (auto &operand : op->getOpOperands())
      if (auto mappedOp = mapper.lookupOrNull(operand.get()))
        operand.set(mappedOp);
    for (auto &succOp : op->getBlockOperands())
      if (auto *mappedOp = mapper.lookupOrNull(succOp.get()))
        succOp.set(mappedOp);
  };

  for (iterator it(mapper.lookup(&front())); it != destPos; ++it)
    it->walk(remapOperands);
}

/// Returns 'block' if 'block' lies in this region, or otherwise finds the
/// ancestor of 'block' that lies in this region. Returns nullptr if the latter
/// fails.
Block *Region::findAncestorBlockInRegion(Block &block) {
  Block *currBlock = &block;
  while (currBlock->getParent() != this) {
    Operation *parentOp = currBlock->getParentOp();
    if (!parentOp || !parentOp->getBlock())
      return nullptr;
    currBlock = parentOp->getBlock();
  }
  return currBlock;
}

/// Returns 'op' if 'op' lies in this region, or otherwise finds the
/// ancestor of 'op' that lies in this region. Returns nullptr if the
/// latter fails.
Operation *Region::findAncestorOpInRegion(Operation &op) {
  Operation *curOp = &op;
  while (Region *opRegion = curOp->getParentRegion()) {
    if (opRegion == this)
      return curOp;

    curOp = opRegion->getParentOp();
    if (!curOp)
      return nullptr;
  }
  return nullptr;
}

void Region::dropAllReferences() {
  for (Block &b : *this)
    b.dropAllReferences();
}

Region *llvm::ilist_traits<::mlir::Block>::getParentRegion() {
  size_t Offset(
      size_t(&((Region *)nullptr->*Region::getSublistAccess(nullptr))));
  iplist<Block> *Anchor(static_cast<iplist<Block> *>(this));
  return reinterpret_cast<Region *>(reinterpret_cast<char *>(Anchor) - Offset);
}

/// This is a trait method invoked when a basic block is added to a region.
/// We keep the region pointer up to date.
void llvm::ilist_traits<::mlir::Block>::addNodeToList(Block *block) {
  assert(!block->getParent() && "already in a region!");
  block->parentValidOpOrderPair.setPointer(getParentRegion());
}

/// This is a trait method invoked when an operation is removed from a
/// region.  We keep the region pointer up to date.
void llvm::ilist_traits<::mlir::Block>::removeNodeFromList(Block *block) {
  assert(block->getParent() && "not already in a region!");
  block->parentValidOpOrderPair.setPointer(nullptr);
}

/// This is a trait method invoked when an operation is moved from one block
/// to another.  We keep the block pointer up to date.
void llvm::ilist_traits<::mlir::Block>::transferNodesFromList(
    ilist_traits<Block> &otherList, block_iterator first, block_iterator last) {
  // If we are transferring operations within the same function, the parent
  // pointer doesn't need to be updated.
  auto *curParent = getParentRegion();
  if (curParent == otherList.getParentRegion())
    return;

  // Update the 'parent' member of each Block.
  for (; first != last; ++first)
    first->parentValidOpOrderPair.setPointer(curParent);
}

//===----------------------------------------------------------------------===//
// Region::OpIterator
//===----------------------------------------------------------------------===//

Region::OpIterator::OpIterator(Region *region, bool end)
    : region(region), block(end ? region->end() : region->begin()) {
  if (!region->empty())
    skipOverBlocksWithNoOps();
}

Region::OpIterator &Region::OpIterator::operator++() {
  // We increment over operations, if we reach the last use then move to next
  // block.
  if (operation != block->end())
    ++operation;
  if (operation == block->end()) {
    ++block;
    skipOverBlocksWithNoOps();
  }
  return *this;
}

void Region::OpIterator::skipOverBlocksWithNoOps() {
  while (block != region->end() && block->empty())
    ++block;

  // If we are at the last block, then set the operation to first operation of
  // next block (sentinel value used for end).
  if (block == region->end())
    operation = {};
  else
    operation = block->begin();
}

//===----------------------------------------------------------------------===//
// RegionRange
//===----------------------------------------------------------------------===//

RegionRange::RegionRange(MutableArrayRef<Region> regions)
    : RegionRange(regions.data(), regions.size()) {}
RegionRange::RegionRange(ArrayRef<std::unique_ptr<Region>> regions)
    : RegionRange(regions.data(), regions.size()) {}

/// See `llvm::detail::indexed_accessor_range_base` for details.
RegionRange::OwnerT RegionRange::offset_base(const OwnerT &owner,
                                             ptrdiff_t index) {
  if (auto *operand = owner.dyn_cast<const std::unique_ptr<Region> *>())
    return operand + index;
  return &owner.get<Region *>()[index];
}
/// See `llvm::detail::indexed_accessor_range_base` for details.
Region *RegionRange::dereference_iterator(const OwnerT &owner,
                                          ptrdiff_t index) {
  if (auto *operand = owner.dyn_cast<const std::unique_ptr<Region> *>())
    return operand[index].get();
  return &owner.get<Region *>()[index];
}
