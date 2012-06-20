//===- llvm/Analysis/LoopInfoImpl.h - Natural Loop Calculator ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is the generic implementation of LoopInfo used for both Loops and
// MachineLoops.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_LOOP_INFO_IMPL_H
#define LLVM_ANALYSIS_LOOP_INFO_IMPL_H

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/ADT/PostOrderIterator.h"

namespace llvm {

//===----------------------------------------------------------------------===//
// APIs for simple analysis of the loop. See header notes.

/// getExitingBlocks - Return all blocks inside the loop that have successors
/// outside of the loop.  These are the blocks _inside of the current loop_
/// which branch out.  The returned list is always unique.
///
template<class BlockT, class LoopT>
void LoopBase<BlockT, LoopT>::
getExitingBlocks(SmallVectorImpl<BlockT *> &ExitingBlocks) const {
  // Sort the blocks vector so that we can use binary search to do quick
  // lookups.
  SmallVector<BlockT*, 128> LoopBBs(block_begin(), block_end());
  std::sort(LoopBBs.begin(), LoopBBs.end());

  typedef GraphTraits<BlockT*> BlockTraits;
  for (block_iterator BI = block_begin(), BE = block_end(); BI != BE; ++BI)
    for (typename BlockTraits::ChildIteratorType I =
           BlockTraits::child_begin(*BI), E = BlockTraits::child_end(*BI);
         I != E; ++I)
      if (!std::binary_search(LoopBBs.begin(), LoopBBs.end(), *I)) {
        // Not in current loop? It must be an exit block.
        ExitingBlocks.push_back(*BI);
        break;
      }
}

/// getExitingBlock - If getExitingBlocks would return exactly one block,
/// return that block. Otherwise return null.
template<class BlockT, class LoopT>
BlockT *LoopBase<BlockT, LoopT>::getExitingBlock() const {
  SmallVector<BlockT*, 8> ExitingBlocks;
  getExitingBlocks(ExitingBlocks);
  if (ExitingBlocks.size() == 1)
    return ExitingBlocks[0];
  return 0;
}

/// getExitBlocks - Return all of the successor blocks of this loop.  These
/// are the blocks _outside of the current loop_ which are branched to.
///
template<class BlockT, class LoopT>
void LoopBase<BlockT, LoopT>::
getExitBlocks(SmallVectorImpl<BlockT*> &ExitBlocks) const {
  // Sort the blocks vector so that we can use binary search to do quick
  // lookups.
  SmallVector<BlockT*, 128> LoopBBs(block_begin(), block_end());
  std::sort(LoopBBs.begin(), LoopBBs.end());

  typedef GraphTraits<BlockT*> BlockTraits;
  for (block_iterator BI = block_begin(), BE = block_end(); BI != BE; ++BI)
    for (typename BlockTraits::ChildIteratorType I =
           BlockTraits::child_begin(*BI), E = BlockTraits::child_end(*BI);
         I != E; ++I)
      if (!std::binary_search(LoopBBs.begin(), LoopBBs.end(), *I))
        // Not in current loop? It must be an exit block.
        ExitBlocks.push_back(*I);
}

/// getExitBlock - If getExitBlocks would return exactly one block,
/// return that block. Otherwise return null.
template<class BlockT, class LoopT>
BlockT *LoopBase<BlockT, LoopT>::getExitBlock() const {
  SmallVector<BlockT*, 8> ExitBlocks;
  getExitBlocks(ExitBlocks);
  if (ExitBlocks.size() == 1)
    return ExitBlocks[0];
  return 0;
}

/// getExitEdges - Return all pairs of (_inside_block_,_outside_block_).
template<class BlockT, class LoopT>
void LoopBase<BlockT, LoopT>::
getExitEdges(SmallVectorImpl<Edge> &ExitEdges) const {
  // Sort the blocks vector so that we can use binary search to do quick
  // lookups.
  SmallVector<BlockT*, 128> LoopBBs(block_begin(), block_end());
  array_pod_sort(LoopBBs.begin(), LoopBBs.end());

  typedef GraphTraits<BlockT*> BlockTraits;
  for (block_iterator BI = block_begin(), BE = block_end(); BI != BE; ++BI)
    for (typename BlockTraits::ChildIteratorType I =
           BlockTraits::child_begin(*BI), E = BlockTraits::child_end(*BI);
         I != E; ++I)
      if (!std::binary_search(LoopBBs.begin(), LoopBBs.end(), *I))
        // Not in current loop? It must be an exit block.
        ExitEdges.push_back(Edge(*BI, *I));
}

/// getLoopPreheader - If there is a preheader for this loop, return it.  A
/// loop has a preheader if there is only one edge to the header of the loop
/// from outside of the loop.  If this is the case, the block branching to the
/// header of the loop is the preheader node.
///
/// This method returns null if there is no preheader for the loop.
///
template<class BlockT, class LoopT>
BlockT *LoopBase<BlockT, LoopT>::getLoopPreheader() const {
  // Keep track of nodes outside the loop branching to the header...
  BlockT *Out = getLoopPredecessor();
  if (!Out) return 0;

  // Make sure there is only one exit out of the preheader.
  typedef GraphTraits<BlockT*> BlockTraits;
  typename BlockTraits::ChildIteratorType SI = BlockTraits::child_begin(Out);
  ++SI;
  if (SI != BlockTraits::child_end(Out))
    return 0;  // Multiple exits from the block, must not be a preheader.

  // The predecessor has exactly one successor, so it is a preheader.
  return Out;
}

/// getLoopPredecessor - If the given loop's header has exactly one unique
/// predecessor outside the loop, return it. Otherwise return null.
/// This is less strict that the loop "preheader" concept, which requires
/// the predecessor to have exactly one successor.
///
template<class BlockT, class LoopT>
BlockT *LoopBase<BlockT, LoopT>::getLoopPredecessor() const {
  // Keep track of nodes outside the loop branching to the header...
  BlockT *Out = 0;

  // Loop over the predecessors of the header node...
  BlockT *Header = getHeader();
  typedef GraphTraits<BlockT*> BlockTraits;
  typedef GraphTraits<Inverse<BlockT*> > InvBlockTraits;
  for (typename InvBlockTraits::ChildIteratorType PI =
         InvBlockTraits::child_begin(Header),
         PE = InvBlockTraits::child_end(Header); PI != PE; ++PI) {
    typename InvBlockTraits::NodeType *N = *PI;
    if (!contains(N)) {     // If the block is not in the loop...
      if (Out && Out != N)
        return 0;             // Multiple predecessors outside the loop
      Out = N;
    }
  }

  // Make sure there is only one exit out of the preheader.
  assert(Out && "Header of loop has no predecessors from outside loop?");
  return Out;
}

/// getLoopLatch - If there is a single latch block for this loop, return it.
/// A latch block is a block that contains a branch back to the header.
template<class BlockT, class LoopT>
BlockT *LoopBase<BlockT, LoopT>::getLoopLatch() const {
  BlockT *Header = getHeader();
  typedef GraphTraits<Inverse<BlockT*> > InvBlockTraits;
  typename InvBlockTraits::ChildIteratorType PI =
    InvBlockTraits::child_begin(Header);
  typename InvBlockTraits::ChildIteratorType PE =
    InvBlockTraits::child_end(Header);
  BlockT *Latch = 0;
  for (; PI != PE; ++PI) {
    typename InvBlockTraits::NodeType *N = *PI;
    if (contains(N)) {
      if (Latch) return 0;
      Latch = N;
    }
  }

  return Latch;
}

//===----------------------------------------------------------------------===//
// APIs for updating loop information after changing the CFG
//

/// addBasicBlockToLoop - This method is used by other analyses to update loop
/// information.  NewBB is set to be a new member of the current loop.
/// Because of this, it is added as a member of all parent loops, and is added
/// to the specified LoopInfo object as being in the current basic block.  It
/// is not valid to replace the loop header with this method.
///
template<class BlockT, class LoopT>
void LoopBase<BlockT, LoopT>::
addBasicBlockToLoop(BlockT *NewBB, LoopInfoBase<BlockT, LoopT> &LIB) {
  assert((Blocks.empty() || LIB[getHeader()] == this) &&
         "Incorrect LI specified for this loop!");
  assert(NewBB && "Cannot add a null basic block to the loop!");
  assert(LIB[NewBB] == 0 && "BasicBlock already in the loop!");

  LoopT *L = static_cast<LoopT *>(this);

  // Add the loop mapping to the LoopInfo object...
  LIB.BBMap[NewBB] = L;

  // Add the basic block to this loop and all parent loops...
  while (L) {
    L->Blocks.push_back(NewBB);
    L = L->getParentLoop();
  }
}

/// replaceChildLoopWith - This is used when splitting loops up.  It replaces
/// the OldChild entry in our children list with NewChild, and updates the
/// parent pointer of OldChild to be null and the NewChild to be this loop.
/// This updates the loop depth of the new child.
template<class BlockT, class LoopT>
void LoopBase<BlockT, LoopT>::
replaceChildLoopWith(LoopT *OldChild, LoopT *NewChild) {
  assert(OldChild->ParentLoop == this && "This loop is already broken!");
  assert(NewChild->ParentLoop == 0 && "NewChild already has a parent!");
  typename std::vector<LoopT *>::iterator I =
    std::find(SubLoops.begin(), SubLoops.end(), OldChild);
  assert(I != SubLoops.end() && "OldChild not in loop!");
  *I = NewChild;
  OldChild->ParentLoop = 0;
  NewChild->ParentLoop = static_cast<LoopT *>(this);
}

/// verifyLoop - Verify loop structure
template<class BlockT, class LoopT>
void LoopBase<BlockT, LoopT>::verifyLoop() const {
#ifndef NDEBUG
  assert(!Blocks.empty() && "Loop header is missing");

  // Setup for using a depth-first iterator to visit every block in the loop.
  SmallVector<BlockT*, 8> ExitBBs;
  getExitBlocks(ExitBBs);
  llvm::SmallPtrSet<BlockT*, 8> VisitSet;
  VisitSet.insert(ExitBBs.begin(), ExitBBs.end());
  df_ext_iterator<BlockT*, llvm::SmallPtrSet<BlockT*, 8> >
    BI = df_ext_begin(getHeader(), VisitSet),
    BE = df_ext_end(getHeader(), VisitSet);

  // Keep track of the number of BBs visited.
  unsigned NumVisited = 0;

  // Sort the blocks vector so that we can use binary search to do quick
  // lookups.
  SmallVector<BlockT*, 128> LoopBBs(block_begin(), block_end());
  std::sort(LoopBBs.begin(), LoopBBs.end());

  // Check the individual blocks.
  for ( ; BI != BE; ++BI) {
    BlockT *BB = *BI;
    bool HasInsideLoopSuccs = false;
    bool HasInsideLoopPreds = false;
    SmallVector<BlockT *, 2> OutsideLoopPreds;

    typedef GraphTraits<BlockT*> BlockTraits;
    for (typename BlockTraits::ChildIteratorType SI =
           BlockTraits::child_begin(BB), SE = BlockTraits::child_end(BB);
         SI != SE; ++SI)
      if (std::binary_search(LoopBBs.begin(), LoopBBs.end(), *SI)) {
        HasInsideLoopSuccs = true;
        break;
      }
    typedef GraphTraits<Inverse<BlockT*> > InvBlockTraits;
    for (typename InvBlockTraits::ChildIteratorType PI =
           InvBlockTraits::child_begin(BB), PE = InvBlockTraits::child_end(BB);
         PI != PE; ++PI) {
      BlockT *N = *PI;
      if (std::binary_search(LoopBBs.begin(), LoopBBs.end(), N))
        HasInsideLoopPreds = true;
      else
        OutsideLoopPreds.push_back(N);
    }

    if (BB == getHeader()) {
        assert(!OutsideLoopPreds.empty() && "Loop is unreachable!");
    } else if (!OutsideLoopPreds.empty()) {
      // A non-header loop shouldn't be reachable from outside the loop,
      // though it is permitted if the predecessor is not itself actually
      // reachable.
      BlockT *EntryBB = BB->getParent()->begin();
        for (df_iterator<BlockT *> NI = df_begin(EntryBB),
               NE = df_end(EntryBB); NI != NE; ++NI)
          for (unsigned i = 0, e = OutsideLoopPreds.size(); i != e; ++i)
            assert(*NI != OutsideLoopPreds[i] &&
                   "Loop has multiple entry points!");
    }
    assert(HasInsideLoopPreds && "Loop block has no in-loop predecessors!");
    assert(HasInsideLoopSuccs && "Loop block has no in-loop successors!");
    assert(BB != getHeader()->getParent()->begin() &&
           "Loop contains function entry block!");

    NumVisited++;
  }

  assert(NumVisited == getNumBlocks() && "Unreachable block in loop");

  // Check the subloops.
  for (iterator I = begin(), E = end(); I != E; ++I)
    // Each block in each subloop should be contained within this loop.
    for (block_iterator BI = (*I)->block_begin(), BE = (*I)->block_end();
         BI != BE; ++BI) {
        assert(std::binary_search(LoopBBs.begin(), LoopBBs.end(), *BI) &&
               "Loop does not contain all the blocks of a subloop!");
    }

  // Check the parent loop pointer.
  if (ParentLoop) {
    assert(std::find(ParentLoop->begin(), ParentLoop->end(), this) !=
           ParentLoop->end() &&
           "Loop is not a subloop of its parent!");
  }
#endif
}

/// verifyLoop - Verify loop structure of this loop and all nested loops.
template<class BlockT, class LoopT>
void LoopBase<BlockT, LoopT>::verifyLoopNest(
  DenseSet<const LoopT*> *Loops) const {
  Loops->insert(static_cast<const LoopT *>(this));
  // Verify this loop.
  verifyLoop();
  // Verify the subloops.
  for (iterator I = begin(), E = end(); I != E; ++I)
    (*I)->verifyLoopNest(Loops);
}

template<class BlockT, class LoopT>
void LoopBase<BlockT, LoopT>::print(raw_ostream &OS, unsigned Depth) const {
  OS.indent(Depth*2) << "Loop at depth " << getLoopDepth()
       << " containing: ";

  for (unsigned i = 0; i < getBlocks().size(); ++i) {
    if (i) OS << ",";
    BlockT *BB = getBlocks()[i];
    WriteAsOperand(OS, BB, false);
    if (BB == getHeader())    OS << "<header>";
    if (BB == getLoopLatch()) OS << "<latch>";
    if (isLoopExiting(BB))    OS << "<exiting>";
  }
  OS << "\n";

  for (iterator I = begin(), E = end(); I != E; ++I)
    (*I)->print(OS, Depth+2);
}

//===----------------------------------------------------------------------===//
/// LoopInfo - This class builds and contains all of the top level loop
/// structures in the specified function.
///

template<class BlockT, class LoopT>
void LoopInfoBase<BlockT, LoopT>::Calculate(DominatorTreeBase<BlockT> &DT) {
  BlockT *RootNode = DT.getRootNode()->getBlock();

  for (df_iterator<BlockT*> NI = df_begin(RootNode),
         NE = df_end(RootNode); NI != NE; ++NI)
    if (LoopT *L = ConsiderForLoop(*NI, DT))
      TopLevelLoops.push_back(L);
}

template<class BlockT, class LoopT>
LoopT *LoopInfoBase<BlockT, LoopT>::
ConsiderForLoop(BlockT *BB, DominatorTreeBase<BlockT> &DT) {
  if (BBMap.count(BB)) return 0; // Haven't processed this node?

  std::vector<BlockT *> TodoStack;

  // Scan the predecessors of BB, checking to see if BB dominates any of
  // them.  This identifies backedges which target this node...
  typedef GraphTraits<Inverse<BlockT*> > InvBlockTraits;
  for (typename InvBlockTraits::ChildIteratorType I =
         InvBlockTraits::child_begin(BB), E = InvBlockTraits::child_end(BB);
       I != E; ++I) {
    typename InvBlockTraits::NodeType *N = *I;
    // If BB dominates its predecessor...
    if (DT.dominates(BB, N) && DT.isReachableFromEntry(N))
      TodoStack.push_back(N);
  }

  if (TodoStack.empty()) return 0;  // No backedges to this block...

  // Create a new loop to represent this basic block...
  LoopT *L = new LoopT(BB);
  BBMap[BB] = L;

  while (!TodoStack.empty()) {  // Process all the nodes in the loop
    BlockT *X = TodoStack.back();
    TodoStack.pop_back();

    if (!L->contains(X) &&         // As of yet unprocessed??
        DT.isReachableFromEntry(X)) {
      // Check to see if this block already belongs to a loop.  If this occurs
      // then we have a case where a loop that is supposed to be a child of
      // the current loop was processed before the current loop.  When this
      // occurs, this child loop gets added to a part of the current loop,
      // making it a sibling to the current loop.  We have to reparent this
      // loop.
      if (LoopT *SubLoop =
          const_cast<LoopT *>(getLoopFor(X)))
        if (SubLoop->getHeader() == X && isNotAlreadyContainedIn(SubLoop, L)){
          // Remove the subloop from its current parent...
          assert(SubLoop->ParentLoop && SubLoop->ParentLoop != L);
          LoopT *SLP = SubLoop->ParentLoop;  // SubLoopParent
          typename std::vector<LoopT *>::iterator I =
            std::find(SLP->SubLoops.begin(), SLP->SubLoops.end(), SubLoop);
          assert(I != SLP->SubLoops.end() &&"SubLoop not a child of parent?");
          SLP->SubLoops.erase(I);   // Remove from parent...

          // Add the subloop to THIS loop...
          SubLoop->ParentLoop = L;
          L->SubLoops.push_back(SubLoop);
        }

      // Normal case, add the block to our loop...
      L->Blocks.push_back(X);

      typedef GraphTraits<Inverse<BlockT*> > InvBlockTraits;

      // Add all of the predecessors of X to the end of the work stack...
      TodoStack.insert(TodoStack.end(), InvBlockTraits::child_begin(X),
                       InvBlockTraits::child_end(X));
    }
  }

  // If there are any loops nested within this loop, create them now!
  for (typename std::vector<BlockT*>::iterator I = L->Blocks.begin(),
         E = L->Blocks.end(); I != E; ++I)
    if (LoopT *NewLoop = ConsiderForLoop(*I, DT)) {
      L->SubLoops.push_back(NewLoop);
      NewLoop->ParentLoop = L;
    }

  // Add the basic blocks that comprise this loop to the BBMap so that this
  // loop can be found for them.
  //
  for (typename std::vector<BlockT*>::iterator I = L->Blocks.begin(),
         E = L->Blocks.end(); I != E; ++I)
    BBMap.insert(std::make_pair(*I, L));

  // Now that we have a list of all of the child loops of this loop, check to
  // see if any of them should actually be nested inside of each other.  We
  // can accidentally pull loops our of their parents, so we must make sure to
  // organize the loop nests correctly now.
  {
    std::map<BlockT *, LoopT *> ContainingLoops;
    for (unsigned i = 0; i != L->SubLoops.size(); ++i) {
      LoopT *Child = L->SubLoops[i];
      assert(Child->getParentLoop() == L && "Not proper child loop?");

      if (LoopT *ContainingLoop = ContainingLoops[Child->getHeader()]) {
        // If there is already a loop which contains this loop, move this loop
        // into the containing loop.
        MoveSiblingLoopInto(Child, ContainingLoop);
        --i;  // The loop got removed from the SubLoops list.
      } else {
        // This is currently considered to be a top-level loop.  Check to see
        // if any of the contained blocks are loop headers for subloops we
        // have already processed.
        for (unsigned b = 0, e = Child->Blocks.size(); b != e; ++b) {
          LoopT *&BlockLoop = ContainingLoops[Child->Blocks[b]];
          if (BlockLoop == 0) {   // Child block not processed yet...
            BlockLoop = Child;
          } else if (BlockLoop != Child) {
            LoopT *SubLoop = BlockLoop;
            // Reparent all of the blocks which used to belong to BlockLoops
            for (unsigned j = 0, f = SubLoop->Blocks.size(); j != f; ++j)
              ContainingLoops[SubLoop->Blocks[j]] = Child;

            // There is already a loop which contains this block, that means
            // that we should reparent the loop which the block is currently
            // considered to belong to to be a child of this loop.
            MoveSiblingLoopInto(SubLoop, Child);
            --i;  // We just shrunk the SubLoops list.
          }
        }
      }
    }
  }

  return L;
}

/// MoveSiblingLoopInto - This method moves the NewChild loop to live inside
/// of the NewParent Loop, instead of being a sibling of it.
template<class BlockT, class LoopT>
void LoopInfoBase<BlockT, LoopT>::
MoveSiblingLoopInto(LoopT *NewChild, LoopT *NewParent) {
  LoopT *OldParent = NewChild->getParentLoop();
  assert(OldParent && OldParent == NewParent->getParentLoop() &&
         NewChild != NewParent && "Not sibling loops!");

  // Remove NewChild from being a child of OldParent
  typename std::vector<LoopT *>::iterator I =
    std::find(OldParent->SubLoops.begin(), OldParent->SubLoops.end(),
              NewChild);
  assert(I != OldParent->SubLoops.end() && "Parent fields incorrect??");
  OldParent->SubLoops.erase(I);   // Remove from parent's subloops list
  NewChild->ParentLoop = 0;

  InsertLoopInto(NewChild, NewParent);
}

/// InsertLoopInto - This inserts loop L into the specified parent loop.  If
/// the parent loop contains a loop which should contain L, the loop gets
/// inserted into L instead.
template<class BlockT, class LoopT>
void LoopInfoBase<BlockT, LoopT>::InsertLoopInto(LoopT *L, LoopT *Parent) {
  BlockT *LHeader = L->getHeader();
  assert(Parent->contains(LHeader) &&
         "This loop should not be inserted here!");

  // Check to see if it belongs in a child loop...
  for (unsigned i = 0, e = static_cast<unsigned>(Parent->SubLoops.size());
       i != e; ++i)
    if (Parent->SubLoops[i]->contains(LHeader)) {
      InsertLoopInto(L, Parent->SubLoops[i]);
      return;
    }

  // If not, insert it here!
  Parent->SubLoops.push_back(L);
  L->ParentLoop = Parent;
}

//===----------------------------------------------------------------------===//
/// Stable LoopInfo Analysis - Build a loop tree using stable iterators so the
/// result does / not depend on use list (block predecessor) order.
///

/// Discover a subloop with the specified backedges such that: All blocks within
/// this loop are mapped to this loop or a subloop. And all subloops within this
/// loop have their parent loop set to this loop or a subloop.
template<class BlockT, class LoopT>
static void discoverAndMapSubloop(LoopT *L, ArrayRef<BlockT*> Backedges,
                                  LoopInfoBase<BlockT, LoopT> *LI,
                                  DominatorTreeBase<BlockT> &DomTree) {
  typedef GraphTraits<Inverse<BlockT*> > InvBlockTraits;

  unsigned NumBlocks = 0;
  unsigned NumSubloops = 0;

  // Perform a backward CFG traversal using a worklist.
  std::vector<BlockT *> ReverseCFGWorklist(Backedges.begin(), Backedges.end());
  while (!ReverseCFGWorklist.empty()) {
    BlockT *PredBB = ReverseCFGWorklist.back();
    ReverseCFGWorklist.pop_back();

    LoopT *Subloop = LI->getLoopFor(PredBB);
    if (!Subloop) {
      if (!DomTree.isReachableFromEntry(PredBB))
        continue;

      // This is an undiscovered block. Map it to the current loop.
      LI->changeLoopFor(PredBB, L);
      ++NumBlocks;
      if (PredBB == L->getHeader())
          continue;
      // Push all block predecessors on the worklist.
      ReverseCFGWorklist.insert(ReverseCFGWorklist.end(),
                                InvBlockTraits::child_begin(PredBB),
                                InvBlockTraits::child_end(PredBB));
    }
    else {
      // This is a discovered block. Find its outermost discovered loop.
      while (LoopT *Parent = Subloop->getParentLoop())
        Subloop = Parent;

      // If it is already discovered to be a subloop of this loop, continue.
      if (Subloop == L)
        continue;

      // Discover a subloop of this loop.
      Subloop->setParentLoop(L);
      ++NumSubloops;
      NumBlocks += Subloop->getBlocks().capacity();
      PredBB = Subloop->getHeader();
      // Continue traversal along predecessors that are not loop-back edges from
      // within this subloop tree itself. Note that a predecessor may directly
      // reach another subloop that is not yet discovered to be a subloop of
      // this loop, which we must traverse.
      for (typename InvBlockTraits::ChildIteratorType PI =
             InvBlockTraits::child_begin(PredBB),
             PE = InvBlockTraits::child_end(PredBB); PI != PE; ++PI) {
        if (LI->getLoopFor(*PI) != Subloop)
          ReverseCFGWorklist.push_back(*PI);
      }
    }
  }
  L->getSubLoopsVector().reserve(NumSubloops);
  L->getBlocksVector().reserve(NumBlocks);
}

namespace {
/// Populate all loop data in a stable order during a single forward DFS.
template<class BlockT, class LoopT>
class PopulateLoopsDFS {
  typedef GraphTraits<BlockT*> BlockTraits;
  typedef typename BlockTraits::ChildIteratorType SuccIterTy;

  LoopInfoBase<BlockT, LoopT> *LI;
  DenseSet<const BlockT *> VisitedBlocks;
  std::vector<std::pair<BlockT*, SuccIterTy> > DFSStack;

public:
  PopulateLoopsDFS(LoopInfoBase<BlockT, LoopT> *li):
    LI(li) {}

  void traverse(BlockT *EntryBlock);

protected:
  void reverseInsertIntoLoop(BlockT *Block);

  BlockT *dfsSource() { return DFSStack.back().first; }
  SuccIterTy &dfsSucc() { return DFSStack.back().second; }
  SuccIterTy dfsSuccEnd() { return BlockTraits::child_end(dfsSource()); }

  void pushBlock(BlockT *Block) {
    DFSStack.push_back(std::make_pair(Block, BlockTraits::child_begin(Block)));
  }
};
} // anonymous

/// Top-level driver for the forward DFS within the loop.
template<class BlockT, class LoopT>
void PopulateLoopsDFS<BlockT, LoopT>::traverse(BlockT *EntryBlock) {
  pushBlock(EntryBlock);
  VisitedBlocks.insert(EntryBlock);
  while (!DFSStack.empty()) {
    // Traverse the leftmost path as far as possible.
    while (dfsSucc() != dfsSuccEnd()) {
      BlockT *BB = *dfsSucc();
      ++dfsSucc();
      if (!VisitedBlocks.insert(BB).second)
        continue;

      // Push the next DFS successor onto the stack.
      pushBlock(BB);
    }
    // Visit the top of the stack in postorder and backtrack.
    reverseInsertIntoLoop(dfsSource());
    DFSStack.pop_back();
  }
}

/// Add a single Block to its ancestor loops in PostOrder. If the block is a
/// subloop header, add the subloop to its parent in PostOrder, then reverse the
/// Block and Subloop vectors of the now complete subloop to achieve RPO.
template<class BlockT, class LoopT>
void PopulateLoopsDFS<BlockT, LoopT>::reverseInsertIntoLoop(BlockT *Block) {
  for (LoopT *Subloop = LI->getLoopFor(Block);
       Subloop; Subloop = Subloop->getParentLoop()) {

    if (Block != Subloop->getHeader()) {
      Subloop->getBlocksVector().push_back(Block);
      continue;
    }
    if (Subloop->getParentLoop())
      Subloop->getParentLoop()->getSubLoopsVector().push_back(Subloop);
    else
      LI->addTopLevelLoop(Subloop);

    // For convenience, Blocks and Subloops are inserted in postorder. Reverse
    // the lists, except for the loop header, which is always at the beginning.
    std::reverse(Subloop->getBlocksVector().begin()+1,
                 Subloop->getBlocksVector().end());
    std::reverse(Subloop->getSubLoopsVector().begin(),
                 Subloop->getSubLoopsVector().end());
  }
}

/// Analyze LoopInfo discovers loops during a postorder DominatorTree traversal
/// interleaved with backward CFG traversals within each subloop
/// (discoverAndMapSubloop). The backward traversal skips inner subloops, so
/// this part of the algorithm is linear in the number of CFG edges. Subloop and
/// Block vectors are then populated during a single forward CFG traversal
/// (PopulateLoopDFS).
///
/// During the two CFG traversals each block is seen three times:
/// 1) Discovered and mapped by a reverse CFG traversal.
/// 2) Visited during a forward DFS CFG traversal.
/// 3) Reverse-inserted in the loop in postorder following forward DFS.
///
/// The Block vectors are inclusive, so step 3 requires loop-depth number of
/// insertions per block.
template<class BlockT, class LoopT>
void LoopInfoBase<BlockT, LoopT>::
Analyze(DominatorTreeBase<BlockT> &DomTree) {

  // Postorder traversal of the dominator tree.
  DomTreeNodeBase<BlockT>* DomRoot = DomTree.getRootNode();
  for (po_iterator<DomTreeNodeBase<BlockT>*> DomIter = po_begin(DomRoot),
         DomEnd = po_end(DomRoot); DomIter != DomEnd; ++DomIter) {

    BlockT *Header = DomIter->getBlock();
    SmallVector<BlockT *, 4> Backedges;

    // Check each predecessor of the potential loop header.
    typedef GraphTraits<Inverse<BlockT*> > InvBlockTraits;
    for (typename InvBlockTraits::ChildIteratorType PI =
           InvBlockTraits::child_begin(Header),
           PE = InvBlockTraits::child_end(Header); PI != PE; ++PI) {

      BlockT *Backedge = *PI;

      // If Header dominates predBB, this is a new loop. Collect the backedges.
      if (DomTree.dominates(Header, Backedge)
          && DomTree.isReachableFromEntry(Backedge)) {
        Backedges.push_back(Backedge);
      }
    }
    // Perform a backward CFG traversal to discover and map blocks in this loop.
    if (!Backedges.empty()) {
      LoopT *L = new LoopT(Header);
      discoverAndMapSubloop(L, ArrayRef<BlockT*>(Backedges), this, DomTree);
    }
  }
  // Perform a single forward CFG traversal to populate block and subloop
  // vectors for all loops.
  PopulateLoopsDFS<BlockT, LoopT> DFS(this);
  DFS.traverse(DomRoot->getBlock());
}

// Debugging
template<class BlockT, class LoopT>
void LoopInfoBase<BlockT, LoopT>::print(raw_ostream &OS) const {
  for (unsigned i = 0; i < TopLevelLoops.size(); ++i)
    TopLevelLoops[i]->print(OS);
#if 0
  for (DenseMap<BasicBlock*, LoopT*>::const_iterator I = BBMap.begin(),
         E = BBMap.end(); I != E; ++I)
    OS << "BB '" << I->first->getName() << "' level = "
       << I->second->getLoopDepth() << "\n";
#endif
}

} // End llvm namespace

#endif
