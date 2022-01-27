//===- InliningUtils.cpp ---- Misc utilities for inlining -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements miscellaneous inlining utilities.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/InliningUtils.h"

#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "inlining"

using namespace mlir;

/// Remap locations from the inlined blocks with CallSiteLoc locations with the
/// provided caller location.
static void
remapInlinedLocations(iterator_range<Region::iterator> inlinedBlocks,
                      Location callerLoc) {
  DenseMap<Location, Location> mappedLocations;
  auto remapOpLoc = [&](Operation *op) {
    auto it = mappedLocations.find(op->getLoc());
    if (it == mappedLocations.end()) {
      auto newLoc = CallSiteLoc::get(op->getLoc(), callerLoc);
      it = mappedLocations.try_emplace(op->getLoc(), newLoc).first;
    }
    op->setLoc(it->second);
  };
  for (auto &block : inlinedBlocks)
    block.walk(remapOpLoc);
}

static void remapInlinedOperands(iterator_range<Region::iterator> inlinedBlocks,
                                 BlockAndValueMapping &mapper) {
  auto remapOperands = [&](Operation *op) {
    for (auto &operand : op->getOpOperands())
      if (auto mappedOp = mapper.lookupOrNull(operand.get()))
        operand.set(mappedOp);
  };
  for (auto &block : inlinedBlocks)
    block.walk(remapOperands);
}

//===----------------------------------------------------------------------===//
// InlinerInterface
//===----------------------------------------------------------------------===//

bool InlinerInterface::isLegalToInline(Operation *call, Operation *callable,
                                       bool wouldBeCloned) const {
  if (auto *handler = getInterfaceFor(call))
    return handler->isLegalToInline(call, callable, wouldBeCloned);
  return false;
}

bool InlinerInterface::isLegalToInline(
    Region *dest, Region *src, bool wouldBeCloned,
    BlockAndValueMapping &valueMapping) const {
  // Regions can always be inlined into functions.
  if (isa<FuncOp>(dest->getParentOp()))
    return true;

  if (auto *handler = getInterfaceFor(dest->getParentOp()))
    return handler->isLegalToInline(dest, src, wouldBeCloned, valueMapping);
  return false;
}

bool InlinerInterface::isLegalToInline(
    Operation *op, Region *dest, bool wouldBeCloned,
    BlockAndValueMapping &valueMapping) const {
  if (auto *handler = getInterfaceFor(op))
    return handler->isLegalToInline(op, dest, wouldBeCloned, valueMapping);
  return false;
}

bool InlinerInterface::shouldAnalyzeRecursively(Operation *op) const {
  auto *handler = getInterfaceFor(op);
  return handler ? handler->shouldAnalyzeRecursively(op) : true;
}

/// Handle the given inlined terminator by replacing it with a new operation
/// as necessary.
void InlinerInterface::handleTerminator(Operation *op, Block *newDest) const {
  auto *handler = getInterfaceFor(op);
  assert(handler && "expected valid dialect handler");
  handler->handleTerminator(op, newDest);
}

/// Handle the given inlined terminator by replacing it with a new operation
/// as necessary.
void InlinerInterface::handleTerminator(Operation *op,
                                        ArrayRef<Value> valuesToRepl) const {
  auto *handler = getInterfaceFor(op);
  assert(handler && "expected valid dialect handler");
  handler->handleTerminator(op, valuesToRepl);
}

void InlinerInterface::processInlinedCallBlocks(
    Operation *call, iterator_range<Region::iterator> inlinedBlocks) const {
  auto *handler = getInterfaceFor(call);
  assert(handler && "expected valid dialect handler");
  handler->processInlinedCallBlocks(call, inlinedBlocks);
}

/// Utility to check that all of the operations within 'src' can be inlined.
static bool isLegalToInline(InlinerInterface &interface, Region *src,
                            Region *insertRegion, bool shouldCloneInlinedRegion,
                            BlockAndValueMapping &valueMapping) {
  for (auto &block : *src) {
    for (auto &op : block) {
      // Check this operation.
      if (!interface.isLegalToInline(&op, insertRegion,
                                     shouldCloneInlinedRegion, valueMapping)) {
        LLVM_DEBUG({
          llvm::dbgs() << "* Illegal to inline because of op: ";
          op.dump();
        });
        return false;
      }
      // Check any nested regions.
      if (interface.shouldAnalyzeRecursively(&op) &&
          llvm::any_of(op.getRegions(), [&](Region &region) {
            return !isLegalToInline(interface, &region, insertRegion,
                                    shouldCloneInlinedRegion, valueMapping);
          }))
        return false;
    }
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Inline Methods
//===----------------------------------------------------------------------===//

static LogicalResult
inlineRegionImpl(InlinerInterface &interface, Region *src, Block *inlineBlock,
                 Block::iterator inlinePoint, BlockAndValueMapping &mapper,
                 ValueRange resultsToReplace, TypeRange regionResultTypes,
                 Optional<Location> inlineLoc, bool shouldCloneInlinedRegion,
                 Operation *call = nullptr) {
  assert(resultsToReplace.size() == regionResultTypes.size());
  // We expect the region to have at least one block.
  if (src->empty())
    return failure();

  // Check that all of the region arguments have been mapped.
  auto *srcEntryBlock = &src->front();
  if (llvm::any_of(srcEntryBlock->getArguments(),
                   [&](BlockArgument arg) { return !mapper.contains(arg); }))
    return failure();

  // Check that the operations within the source region are valid to inline.
  Region *insertRegion = inlineBlock->getParent();
  if (!interface.isLegalToInline(insertRegion, src, shouldCloneInlinedRegion,
                                 mapper) ||
      !isLegalToInline(interface, src, insertRegion, shouldCloneInlinedRegion,
                       mapper))
    return failure();

  // Check to see if the region is being cloned, or moved inline. In either
  // case, move the new blocks after the 'insertBlock' to improve IR
  // readability.
  Block *postInsertBlock = inlineBlock->splitBlock(inlinePoint);
  if (shouldCloneInlinedRegion)
    src->cloneInto(insertRegion, postInsertBlock->getIterator(), mapper);
  else
    insertRegion->getBlocks().splice(postInsertBlock->getIterator(),
                                     src->getBlocks(), src->begin(),
                                     src->end());

  // Get the range of newly inserted blocks.
  auto newBlocks = llvm::make_range(std::next(inlineBlock->getIterator()),
                                    postInsertBlock->getIterator());
  Block *firstNewBlock = &*newBlocks.begin();

  // Remap the locations of the inlined operations if a valid source location
  // was provided.
  if (inlineLoc && !inlineLoc->isa<UnknownLoc>())
    remapInlinedLocations(newBlocks, *inlineLoc);

  // If the blocks were moved in-place, make sure to remap any necessary
  // operands.
  if (!shouldCloneInlinedRegion)
    remapInlinedOperands(newBlocks, mapper);

  // Process the newly inlined blocks.
  if (call)
    interface.processInlinedCallBlocks(call, newBlocks);
  interface.processInlinedBlocks(newBlocks);

  // Handle the case where only a single block was inlined.
  if (std::next(newBlocks.begin()) == newBlocks.end()) {
    // Have the interface handle the terminator of this block.
    auto *firstBlockTerminator = firstNewBlock->getTerminator();
    interface.handleTerminator(firstBlockTerminator,
                               llvm::to_vector<6>(resultsToReplace));
    firstBlockTerminator->erase();

    // Merge the post insert block into the cloned entry block.
    firstNewBlock->getOperations().splice(firstNewBlock->end(),
                                          postInsertBlock->getOperations());
    postInsertBlock->erase();
  } else {
    // Otherwise, there were multiple blocks inlined. Add arguments to the post
    // insertion block to represent the results to replace.
    for (const auto &resultToRepl : llvm::enumerate(resultsToReplace)) {
      resultToRepl.value().replaceAllUsesWith(postInsertBlock->addArgument(
          regionResultTypes[resultToRepl.index()]));
    }

    /// Handle the terminators for each of the new blocks.
    for (auto &newBlock : newBlocks)
      interface.handleTerminator(newBlock.getTerminator(), postInsertBlock);
  }

  // Splice the instructions of the inlined entry block into the insert block.
  inlineBlock->getOperations().splice(inlineBlock->end(),
                                      firstNewBlock->getOperations());
  firstNewBlock->erase();
  return success();
}

static LogicalResult
inlineRegionImpl(InlinerInterface &interface, Region *src, Block *inlineBlock,
                 Block::iterator inlinePoint, ValueRange inlinedOperands,
                 ValueRange resultsToReplace, Optional<Location> inlineLoc,
                 bool shouldCloneInlinedRegion, Operation *call = nullptr) {
  // We expect the region to have at least one block.
  if (src->empty())
    return failure();

  auto *entryBlock = &src->front();
  if (inlinedOperands.size() != entryBlock->getNumArguments())
    return failure();

  // Map the provided call operands to the arguments of the region.
  BlockAndValueMapping mapper;
  for (unsigned i = 0, e = inlinedOperands.size(); i != e; ++i) {
    // Verify that the types of the provided values match the function argument
    // types.
    BlockArgument regionArg = entryBlock->getArgument(i);
    if (inlinedOperands[i].getType() != regionArg.getType())
      return failure();
    mapper.map(regionArg, inlinedOperands[i]);
  }

  // Call into the main region inliner function.
  return inlineRegionImpl(interface, src, inlineBlock, inlinePoint, mapper,
                          resultsToReplace, resultsToReplace.getTypes(),
                          inlineLoc, shouldCloneInlinedRegion, call);
}

LogicalResult mlir::inlineRegion(InlinerInterface &interface, Region *src,
                                 Operation *inlinePoint,
                                 BlockAndValueMapping &mapper,
                                 ValueRange resultsToReplace,
                                 TypeRange regionResultTypes,
                                 Optional<Location> inlineLoc,
                                 bool shouldCloneInlinedRegion) {
  return inlineRegion(interface, src, inlinePoint->getBlock(),
                      ++inlinePoint->getIterator(), mapper, resultsToReplace,
                      regionResultTypes, inlineLoc, shouldCloneInlinedRegion);
}
LogicalResult
mlir::inlineRegion(InlinerInterface &interface, Region *src, Block *inlineBlock,
                   Block::iterator inlinePoint, BlockAndValueMapping &mapper,
                   ValueRange resultsToReplace, TypeRange regionResultTypes,
                   Optional<Location> inlineLoc,
                   bool shouldCloneInlinedRegion) {
  return inlineRegionImpl(interface, src, inlineBlock, inlinePoint, mapper,
                          resultsToReplace, regionResultTypes, inlineLoc,
                          shouldCloneInlinedRegion);
}

LogicalResult mlir::inlineRegion(InlinerInterface &interface, Region *src,
                                 Operation *inlinePoint,
                                 ValueRange inlinedOperands,
                                 ValueRange resultsToReplace,
                                 Optional<Location> inlineLoc,
                                 bool shouldCloneInlinedRegion) {
  return inlineRegion(interface, src, inlinePoint->getBlock(),
                      ++inlinePoint->getIterator(), inlinedOperands,
                      resultsToReplace, inlineLoc, shouldCloneInlinedRegion);
}
LogicalResult
mlir::inlineRegion(InlinerInterface &interface, Region *src, Block *inlineBlock,
                   Block::iterator inlinePoint, ValueRange inlinedOperands,
                   ValueRange resultsToReplace, Optional<Location> inlineLoc,
                   bool shouldCloneInlinedRegion) {
  return inlineRegionImpl(interface, src, inlineBlock, inlinePoint,
                          inlinedOperands, resultsToReplace, inlineLoc,
                          shouldCloneInlinedRegion);
}

/// Utility function used to generate a cast operation from the given interface,
/// or return nullptr if a cast could not be generated.
static Value materializeConversion(const DialectInlinerInterface *interface,
                                   SmallVectorImpl<Operation *> &castOps,
                                   OpBuilder &castBuilder, Value arg, Type type,
                                   Location conversionLoc) {
  if (!interface)
    return nullptr;

  // Check to see if the interface for the call can materialize a conversion.
  Operation *castOp = interface->materializeCallConversion(castBuilder, arg,
                                                           type, conversionLoc);
  if (!castOp)
    return nullptr;
  castOps.push_back(castOp);

  // Ensure that the generated cast is correct.
  assert(castOp->getNumOperands() == 1 && castOp->getOperand(0) == arg &&
         castOp->getNumResults() == 1 && *castOp->result_type_begin() == type);
  return castOp->getResult(0);
}

/// This function inlines a given region, 'src', of a callable operation,
/// 'callable', into the location defined by the given call operation. This
/// function returns failure if inlining is not possible, success otherwise. On
/// failure, no changes are made to the module. 'shouldCloneInlinedRegion'
/// corresponds to whether the source region should be cloned into the 'call' or
/// spliced directly.
LogicalResult mlir::inlineCall(InlinerInterface &interface,
                               CallOpInterface call,
                               CallableOpInterface callable, Region *src,
                               bool shouldCloneInlinedRegion) {
  // We expect the region to have at least one block.
  if (src->empty())
    return failure();
  auto *entryBlock = &src->front();
  ArrayRef<Type> callableResultTypes = callable.getCallableResults();

  // Make sure that the number of arguments and results matchup between the call
  // and the region.
  SmallVector<Value, 8> callOperands(call.getArgOperands());
  SmallVector<Value, 8> callResults(call->getResults());
  if (callOperands.size() != entryBlock->getNumArguments() ||
      callResults.size() != callableResultTypes.size())
    return failure();

  // A set of cast operations generated to matchup the signature of the region
  // with the signature of the call.
  SmallVector<Operation *, 4> castOps;
  castOps.reserve(callOperands.size() + callResults.size());

  // Functor used to cleanup generated state on failure.
  auto cleanupState = [&] {
    for (auto *op : castOps) {
      op->getResult(0).replaceAllUsesWith(op->getOperand(0));
      op->erase();
    }
    return failure();
  };

  // Builder used for any conversion operations that need to be materialized.
  OpBuilder castBuilder(call);
  Location castLoc = call.getLoc();
  const auto *callInterface = interface.getInterfaceFor(call->getDialect());

  // Map the provided call operands to the arguments of the region.
  BlockAndValueMapping mapper;
  for (unsigned i = 0, e = callOperands.size(); i != e; ++i) {
    BlockArgument regionArg = entryBlock->getArgument(i);
    Value operand = callOperands[i];

    // If the call operand doesn't match the expected region argument, try to
    // generate a cast.
    Type regionArgType = regionArg.getType();
    if (operand.getType() != regionArgType) {
      if (!(operand = materializeConversion(callInterface, castOps, castBuilder,
                                            operand, regionArgType, castLoc)))
        return cleanupState();
    }
    mapper.map(regionArg, operand);
  }

  // Ensure that the resultant values of the call match the callable.
  castBuilder.setInsertionPointAfter(call);
  for (unsigned i = 0, e = callResults.size(); i != e; ++i) {
    Value callResult = callResults[i];
    if (callResult.getType() == callableResultTypes[i])
      continue;

    // Generate a conversion that will produce the original type, so that the IR
    // is still valid after the original call gets replaced.
    Value castResult =
        materializeConversion(callInterface, castOps, castBuilder, callResult,
                              callResult.getType(), castLoc);
    if (!castResult)
      return cleanupState();
    callResult.replaceAllUsesWith(castResult);
    castResult.getDefiningOp()->replaceUsesOfWith(castResult, callResult);
  }

  // Check that it is legal to inline the callable into the call.
  if (!interface.isLegalToInline(call, callable, shouldCloneInlinedRegion))
    return cleanupState();

  // Attempt to inline the call.
  if (failed(inlineRegionImpl(interface, src, call->getBlock(),
                              ++call->getIterator(), mapper, callResults,
                              callableResultTypes, call.getLoc(),
                              shouldCloneInlinedRegion, call)))
    return cleanupState();
  return success();
}
