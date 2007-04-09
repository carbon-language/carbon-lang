//===- LoopRotation.cpp - Loop Rotation Pass ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Devang Patel and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements Loop Rotation Pass.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "loop-rotate"

#include "llvm/Transforms/Scalar.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/SmallVector.h"

using namespace llvm;

#define MAX_HEADER_SIZE 16

STATISTIC(NumRotated, "Number of loops rotated");
namespace {

  class VISIBILITY_HIDDEN RenameData {
  public:
    RenameData(Instruction *O, Instruction *P, Instruction *H) 
      : Original(O), PreHeader(P), Header(H) { }
  public:
    Instruction *Original; // Original instruction
    Instruction *PreHeader; // New pre-header replacement
    Instruction *Header; // New header replacement
  };
  
  class VISIBILITY_HIDDEN LoopRotate : public LoopPass {

  public:
    
    // Rotate Loop L as many times as possible. Return true if
    // loop is rotated at least once.
    bool runOnLoop(Loop *L, LPPassManager &LPM);

    // LCSSA form makes instruction renaming easier.
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequiredID(LCSSAID);
      AU.addPreservedID(LCSSAID);
    }

    // Helper functions

    /// Do actual work
    bool rotateLoop(Loop *L, LPPassManager &LPM);
    
    /// Initialize local data
    void initialize();

    /// Make sure all Exit block PHINodes have required incoming values.
    /// If incoming value is constant or defined outside the loop then
    /// PHINode may not have an entry for new pre-header. 
    void  updateExitBlock();

    /// Return true if this instruction is used outside original header.
    bool usedOutsideOriginalHeader(Instruction *In);

    /// Find Replacement information for instruction. Return NULL if it is
    /// not available.
    RenameData *findReplacementData(Instruction *I);

  private:

    Loop *L;
    BasicBlock *OrigHeader;
    BasicBlock *OrigPreHeader;
    BasicBlock *OrigLatch;
    BasicBlock *NewHeader;
    BasicBlock *NewPreHeader;
    BasicBlock *Exit;

    SmallVector<RenameData, MAX_HEADER_SIZE> LoopHeaderInfo;
  };
  
  RegisterPass<LoopRotate> X ("loop-rotate", "Rotate Loops");
}

LoopPass *llvm::createLoopRotatePass() { return new LoopRotate(); }

/// Rotate Loop L as many times as possible. Return true if
/// loop is rotated at least once.
bool LoopRotate::runOnLoop(Loop *Lp, LPPassManager &LPM) {
  
  bool RotatedOneLoop = false;
  initialize();

  // One loop can be rotated multiple times.
  while (rotateLoop(Lp,LPM)) {
    RotatedOneLoop = true;
    initialize();
  }

  return RotatedOneLoop;
}

/// Rotate loop LP. Return true if it loop is rotated.
bool LoopRotate::rotateLoop(Loop *Lp, LPPassManager &LPM) {

  L = Lp;

  OrigHeader =  L->getHeader();
  OrigPreHeader = L->getLoopPreheader();
  OrigLatch = L->getLoopLatch();

  // If loop has only one block then there is not much to rotate.
  if (L->getBlocks().size() == 1)
    return false;

  if (!OrigHeader || !OrigLatch || !OrigPreHeader)
    return false;

  // If loop header is not one of the loop exit block then
  // either this loop is already rotated or it is not 
  // suitable for loop rotation transformations.
  if (!L->isLoopExit(OrigHeader))
    return false;

  BranchInst *BI = dyn_cast<BranchInst>(OrigHeader->getTerminator());
  if (!BI)
    return false;
  assert (BI->isConditional() && "Branch Instruction is not condiitional");

  // Updating PHInodes in loops with multiple exits adds complexity. 
  // Keep it simple, and restrict loop rotation to loops with one exit only.
  // In future, lift this restriction and support for multiple exits if
  // required.
  std::vector<BasicBlock *> ExitBlocks;
  L->getExitBlocks(ExitBlocks);
  if (ExitBlocks.size() > 1)
    return false;

  // Find new Loop header. NewHeader is a Header's one and only successor
  // that is inside loop.  Header's other successor is out side the
  // loop. Otherwise loop is not suitable for rotation.
  Exit = BI->getSuccessor(0);
  NewHeader = BI->getSuccessor(1);
  if (L->contains(Exit))
    std::swap(Exit, NewHeader);
  assert (NewHeader && "Unable to determine new loop header");
  assert(L->contains(NewHeader) && !L->contains(Exit) && 
         "Unable to determine loop header and exit blocks");

  // Check size of original header and reject
  // loop if it is very big.
  if (OrigHeader->getInstList().size() > MAX_HEADER_SIZE)
    return false;

  // Now, this loop is suitable for rotation.

  // Copy PHI nodes and other instructions from original header
  // into new pre-header. Unlike original header, new pre-header is
  // not a member of loop. New pre-header has only one predecessor,
  // that is original loop pre-header.
  //
  // New loop header is one and only successor of original header that 
  // is inside the loop. All other original header successors are outside 
  // the loop. Copy PHI Nodes from original header into new loop header. 
  // Add second incoming value, from new loop pre-header into these phi 
  // nodes. If a value defined in original header is used outside original 
  // header then new loop header will need new phi nodes with two incoming 
  // values, one definition from original header and second definition is 
  // from new loop pre-header (which is a clone of original header definition).

  NewPreHeader = new BasicBlock("bb.nph", OrigHeader->getParent(), OrigHeader);
  BasicBlock::iterator I = OrigHeader->begin(), E = OrigHeader->end();
  for (; I != E; ++I) {
    Instruction *In = I;

    PHINode *PN = dyn_cast<PHINode>(I);
    if (!PN)
      break;

    // Create new PHI node with one value incoming from OrigPreHeader.
    // NewPreHeader has only one predecessor, OrigPreHeader.
    PHINode *NPH = new PHINode(In->getType(), In->getName());
    NPH->addIncoming(PN->getIncomingValueForBlock(OrigPreHeader), 
                     OrigPreHeader);
    NewPreHeader->getInstList().push_back(NPH);
    
    // Create new PHI node with two incoming values for NewHeader.
    // One incoming value is from OrigLatch (through OrigHeader) and 
    // second incoming value is from NewPreHeader.
    PHINode *NH = new PHINode(In->getType(), In->getName());
    NH->addIncoming(PN->getIncomingValueForBlock(OrigLatch), OrigHeader);
    NH->addIncoming(NPH, NewPreHeader);
    NewHeader->getInstList().push_front(NH);
    
    // "In" can be replaced by NPH or NH at various places.
    LoopHeaderInfo.push_back(RenameData(In, NPH, NH));
  }

  // Now, handle non-phi instructions.
  for (; I != E; ++I) {
    Instruction *In = I;

    assert (!isa<PHINode>(In) && "PHINode is not expected here");
    // This is not a PHI instruction. Insert its clone into NewPreHeader.
    // If this instruction is using a value from same basic block then
    // update it to use value from cloned instruction.
    Instruction *C = In->clone();
    C->setName(In->getName());
    NewPreHeader->getInstList().push_back(C);
    
    // If this instruction is used outside this basic block then
    // create new PHINode for this instruction.
    Instruction *NewHeaderReplacement = NULL;
    if (usedOutsideOriginalHeader(In)) {
      PHINode *PN = new PHINode(In->getType(), In->getName());
      PN->addIncoming(In, OrigHeader);
      PN->addIncoming(C, NewPreHeader);
      NewHeader->getInstList().push_front(PN);
      NewHeaderReplacement = PN;
    } 
    
    // "In" can be replaced by NPH or NH at various places.
    LoopHeaderInfo.push_back(RenameData(In, C, NewHeaderReplacement));
  }

  // Update new pre-header.
  // Rename values that are defined in original header to reflects values
  // defined in new pre-header.
  for (SmallVector<RenameData, MAX_HEADER_SIZE>::iterator 
         I = LoopHeaderInfo.begin(), E = LoopHeaderInfo.end(); I != E; ++I) {
    
    const RenameData &ILoopHeaderInfo = *I;
    Instruction *In = ILoopHeaderInfo.Original;
    Instruction *C = ILoopHeaderInfo.PreHeader;

    // If this instruction is not from new pre-header then is not new 
    // pre-header then this instruction is not handled here.
    if (C->getParent() != NewPreHeader)
      continue;

    // PHINodes uses value from pre-header predecessors.
    if (isa<PHINode>(In))
      continue;

    for (unsigned opi = 0, e = In->getNumOperands(); opi != e; ++opi) {
      if (Instruction *OpPhi = dyn_cast<PHINode>(In->getOperand(opi))) {
        if (RenameData *D = findReplacementData(OpPhi))
          C->setOperand(opi, D->PreHeader);
      }
      else if (Instruction *OpInsn = 
               dyn_cast<Instruction>(In->getOperand(opi))) {
        if (RenameData *D = findReplacementData(OpInsn))
          C->setOperand(opi, D->PreHeader);
      }
    }
  }

  // Rename uses of original header instructions to reflect their new
  // definitions (either from new pre-header node or from newly created
  // new header PHINodes.
  //
  // Original header instructions are used in
  // 1) Original header:
  //
  //    If instruction is used in non-phi instructions then it is using
  //    defintion from original heder iteself. Do not replace this use
  //    with definition from new header or new pre-header.
  //
  //    If instruction is used in phi node then it is an incoming 
  //    value. Rename its use to reflect new definition from new-preheader
  //    or new header.
  //
  // 2) Inside loop but not in original header
  //
  //    Replace this use to reflect definition from new header.
  for (SmallVector<RenameData, MAX_HEADER_SIZE>::iterator 
         I = LoopHeaderInfo.begin(), E = LoopHeaderInfo.end(); I != E; ++I) {

    const RenameData &ILoopHeaderInfo = *I;
    if (!ILoopHeaderInfo.Header)
      continue;

    Instruction *OldPhi = ILoopHeaderInfo.Original;
    Instruction *NewPhi = ILoopHeaderInfo.Header;

    // Before replacing uses, collect them first, so that iterator is
    // not invalidated.
    SmallVector<Instruction *, 16> AllUses;
    for (Value::use_iterator UI = OldPhi->use_begin(), UE = OldPhi->use_end();
         UI != UE; ++UI) {
      Instruction *U = cast<Instruction>(UI);
      AllUses.push_back(U);
    }

    for (SmallVector<Instruction *, 16>::iterator UI = AllUses.begin(), 
           UE = AllUses.end(); UI != UE; ++UI) {
      Instruction *U = *UI;
      BasicBlock *Parent = U->getParent();

      // Used inside original header
      if (Parent == OrigHeader) {
        // Do not rename uses inside original header non-phi instructions.
        PHINode *PU = dyn_cast<PHINode>(U);
        if (!PU)
          continue;

        // Do not rename uses inside original header phi nodes, if the
        // incoming value is for new header.
        if (PU->getBasicBlockIndex(NewHeader) != -1
            && PU->getIncomingValueForBlock(NewHeader) == U)
          continue;

       U->replaceUsesOfWith(OldPhi, NewPhi);
       continue;
      }

      // Used inside loop, but not in original header.
      if (L->contains(U->getParent())) {
        if (U != NewPhi)
          U->replaceUsesOfWith(OldPhi, NewPhi);
        continue;
      }

      // Used inside Exit Block. Since we are in LCSSA form, U must be PHINode.
      assert (U->getParent() == Exit 
              && "Need to propagate new PHI into Exit blocks");
      assert (isa<PHINode>(U) && "Use in Exit Block that is not PHINode");        

      PHINode *UPhi = cast<PHINode>(U);

      // UPhi already has one incoming argument from original header. 
      // Add second incoming argument from new Pre header.
      
      UPhi->addIncoming(ILoopHeaderInfo.PreHeader, NewPreHeader);
    }
  }
  
  /// Make sure all Exit block PHINodes have required incoming values.
  updateExitBlock();

  // Update CFG

  // Removing incoming branch from loop preheader to original header.
  // Now original header is inside the loop.
  OrigHeader->removePredecessor(OrigPreHeader);

  // Establish NewPreHeader as loop preheader. Add unconditional branch
  // from original loop pre-header to new loop pre-header. Add NewPreHEader
  // in loop nest.
  BranchInst *PH_BI = cast<BranchInst>(OrigPreHeader->getTerminator());
  PH_BI->setSuccessor(0, NewPreHeader);
  LoopInfo &LI = LPM.getAnalysis<LoopInfo>();
  if (Loop *PL = LI.getLoopFor(OrigPreHeader))
    PL->addBasicBlockToLoop(NewPreHeader, LI);

  // Make NewHeader as the new header for the loop.
  L->moveToHeader(NewHeader);

  NumRotated++;
  return true;
}

/// Make sure all Exit block PHINodes have required incoming values.
/// If incoming value is constant or defined outside the loop then
/// PHINode may not have an entry for new pre-header. 
void LoopRotate::updateExitBlock() {

  for (BasicBlock::iterator I = Exit->begin(), E = Exit->end();
       I != E; ++I) {

    PHINode *PN = dyn_cast<PHINode>(I);
    if (!PN)
      break;

    // There is already one incoming value from new pre-header block.
    if (PN->getBasicBlockIndex(NewPreHeader) != -1)
      return;

    RenameData *ILoopHeaderInfo;
    Value *V = PN->getIncomingValueForBlock(OrigHeader);
    if (isa<Instruction>(V) && 
        (ILoopHeaderInfo = findReplacementData(cast<Instruction>(V)))) {
      assert (ILoopHeaderInfo->PreHeader && "Missing New Preheader Instruction");
      PN->addIncoming(ILoopHeaderInfo->PreHeader, NewPreHeader);
    } else {
      PN->addIncoming(V, NewPreHeader);
    }
  }
}

/// Initialize local data
void LoopRotate::initialize() {
  L = NULL;
  OrigHeader = NULL;
  OrigPreHeader = NULL;
  NewHeader = NULL;
  NewPreHeader = NULL;
  Exit = NULL;

  LoopHeaderInfo.clear();
}

/// Return true if this instruction is used by any instructions in the loop that
/// aren't in original header.
bool LoopRotate::usedOutsideOriginalHeader(Instruction *In) {

  for (Value::use_iterator UI = In->use_begin(), UE = In->use_end();
       UI != UE; ++UI) {
    Instruction *U = cast<Instruction>(UI);
    if (U->getParent() != OrigHeader) {
      if (L->contains(U->getParent()))
        return true;
    }
  }

  return false;
}

/// Find Replacement information for instruction. Return NULL if it is
/// not available.
RenameData *LoopRotate::findReplacementData(Instruction *In) {

  // Since LoopHeaderInfo is small, linear walk is OK.
  for (SmallVector<RenameData, MAX_HEADER_SIZE>::iterator 
         I = LoopHeaderInfo.begin(), E = LoopHeaderInfo.end(); I != E; ++I) 
    if (I->Original == In)
      return &(*I);

  return NULL;
}
