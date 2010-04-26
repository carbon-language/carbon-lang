//===- MachineSSAUpdater.cpp - Unstructured SSA Update Tool ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MachineSSAUpdater class. It's based on SSAUpdater
// class in lib/Transforms/Utils.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineSSAUpdater.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/AlignOf.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

/// BBInfo - Per-basic block information used internally by MachineSSAUpdater.
class MachineSSAUpdater::BBInfo {
public:
  MachineBasicBlock *BB; // Back-pointer to the corresponding block.
  unsigned AvailableVal; // Value to use in this block.
  BBInfo *DefBB;         // Block that defines the available value.
  int BlkNum;            // Postorder number.
  BBInfo *IDom;          // Immediate dominator.
  unsigned NumPreds;     // Number of predecessor blocks.
  BBInfo **Preds;        // Array[NumPreds] of predecessor blocks.
  MachineInstr *PHITag;  // Marker for existing PHIs that match.

  BBInfo(MachineBasicBlock *ThisBB, unsigned V)
    : BB(ThisBB), AvailableVal(V), DefBB(V ? this : 0), BlkNum(0), IDom(0),
      NumPreds(0), Preds(0), PHITag(0) { }
};

typedef DenseMap<MachineBasicBlock*, MachineSSAUpdater::BBInfo*> BBMapTy;

typedef DenseMap<MachineBasicBlock*, unsigned> AvailableValsTy;
static AvailableValsTy &getAvailableVals(void *AV) {
  return *static_cast<AvailableValsTy*>(AV);
}

static BBMapTy *getBBMap(void *BM) {
  return static_cast<BBMapTy*>(BM);
}

MachineSSAUpdater::MachineSSAUpdater(MachineFunction &MF,
                                     SmallVectorImpl<MachineInstr*> *NewPHI)
  : AV(0), BM(0), InsertedPHIs(NewPHI) {
  TII = MF.getTarget().getInstrInfo();
  MRI = &MF.getRegInfo();
}

MachineSSAUpdater::~MachineSSAUpdater() {
  delete &getAvailableVals(AV);
}

/// Initialize - Reset this object to get ready for a new set of SSA
/// updates.  ProtoValue is the value used to name PHI nodes.
void MachineSSAUpdater::Initialize(unsigned V) {
  if (AV == 0)
    AV = new AvailableValsTy();
  else
    getAvailableVals(AV).clear();

  VR = V;
  VRC = MRI->getRegClass(VR);
}

/// HasValueForBlock - Return true if the MachineSSAUpdater already has a value for
/// the specified block.
bool MachineSSAUpdater::HasValueForBlock(MachineBasicBlock *BB) const {
  return getAvailableVals(AV).count(BB);
}

/// AddAvailableValue - Indicate that a rewritten value is available in the
/// specified block with the specified value.
void MachineSSAUpdater::AddAvailableValue(MachineBasicBlock *BB, unsigned V) {
  getAvailableVals(AV)[BB] = V;
}

/// GetValueAtEndOfBlock - Construct SSA form, materializing a value that is
/// live at the end of the specified block.
unsigned MachineSSAUpdater::GetValueAtEndOfBlock(MachineBasicBlock *BB) {
  return GetValueAtEndOfBlockInternal(BB);
}

static
unsigned LookForIdenticalPHI(MachineBasicBlock *BB,
          SmallVector<std::pair<MachineBasicBlock*, unsigned>, 8> &PredValues) {
  if (BB->empty())
    return 0;

  MachineBasicBlock::iterator I = BB->front();
  if (!I->isPHI())
    return 0;

  AvailableValsTy AVals;
  for (unsigned i = 0, e = PredValues.size(); i != e; ++i)
    AVals[PredValues[i].first] = PredValues[i].second;
  while (I != BB->end() && I->isPHI()) {
    bool Same = true;
    for (unsigned i = 1, e = I->getNumOperands(); i != e; i += 2) {
      unsigned SrcReg = I->getOperand(i).getReg();
      MachineBasicBlock *SrcBB = I->getOperand(i+1).getMBB();
      if (AVals[SrcBB] != SrcReg) {
        Same = false;
        break;
      }
    }
    if (Same)
      return I->getOperand(0).getReg();
    ++I;
  }
  return 0;
}

/// InsertNewDef - Insert an empty PHI or IMPLICIT_DEF instruction which define
/// a value of the given register class at the start of the specified basic
/// block. It returns the virtual register defined by the instruction.
static
MachineInstr *InsertNewDef(unsigned Opcode,
                           MachineBasicBlock *BB, MachineBasicBlock::iterator I,
                           const TargetRegisterClass *RC,
                           MachineRegisterInfo *MRI, const TargetInstrInfo *TII) {
  unsigned NewVR = MRI->createVirtualRegister(RC);
  return BuildMI(*BB, I, DebugLoc(), TII->get(Opcode), NewVR);
}

/// GetValueInMiddleOfBlock - Construct SSA form, materializing a value that
/// is live in the middle of the specified block.
///
/// GetValueInMiddleOfBlock is the same as GetValueAtEndOfBlock except in one
/// important case: if there is a definition of the rewritten value after the
/// 'use' in BB.  Consider code like this:
///
///      X1 = ...
///   SomeBB:
///      use(X)
///      X2 = ...
///      br Cond, SomeBB, OutBB
///
/// In this case, there are two values (X1 and X2) added to the AvailableVals
/// set by the client of the rewriter, and those values are both live out of
/// their respective blocks.  However, the use of X happens in the *middle* of
/// a block.  Because of this, we need to insert a new PHI node in SomeBB to
/// merge the appropriate values, and this value isn't live out of the block.
///
unsigned MachineSSAUpdater::GetValueInMiddleOfBlock(MachineBasicBlock *BB) {
  // If there is no definition of the renamed variable in this block, just use
  // GetValueAtEndOfBlock to do our work.
  if (!HasValueForBlock(BB))
    return GetValueAtEndOfBlockInternal(BB);

  // If there are no predecessors, just return undef.
  if (BB->pred_empty()) {
    // Insert an implicit_def to represent an undef value.
    MachineInstr *NewDef = InsertNewDef(TargetOpcode::IMPLICIT_DEF,
                                        BB, BB->getFirstTerminator(),
                                        VRC, MRI, TII);
    return NewDef->getOperand(0).getReg();
  }

  // Otherwise, we have the hard case.  Get the live-in values for each
  // predecessor.
  SmallVector<std::pair<MachineBasicBlock*, unsigned>, 8> PredValues;
  unsigned SingularValue = 0;

  bool isFirstPred = true;
  for (MachineBasicBlock::pred_iterator PI = BB->pred_begin(),
         E = BB->pred_end(); PI != E; ++PI) {
    MachineBasicBlock *PredBB = *PI;
    unsigned PredVal = GetValueAtEndOfBlockInternal(PredBB);
    PredValues.push_back(std::make_pair(PredBB, PredVal));

    // Compute SingularValue.
    if (isFirstPred) {
      SingularValue = PredVal;
      isFirstPred = false;
    } else if (PredVal != SingularValue)
      SingularValue = 0;
  }

  // Otherwise, if all the merged values are the same, just use it.
  if (SingularValue != 0)
    return SingularValue;

  // If an identical PHI is already in BB, just reuse it.
  unsigned DupPHI = LookForIdenticalPHI(BB, PredValues);
  if (DupPHI)
    return DupPHI;

  // Otherwise, we do need a PHI: insert one now.
  MachineBasicBlock::iterator Loc = BB->empty() ? BB->end() : BB->front();
  MachineInstr *InsertedPHI = InsertNewDef(TargetOpcode::PHI, BB,
                                           Loc, VRC, MRI, TII);

  // Fill in all the predecessors of the PHI.
  MachineInstrBuilder MIB(InsertedPHI);
  for (unsigned i = 0, e = PredValues.size(); i != e; ++i)
    MIB.addReg(PredValues[i].second).addMBB(PredValues[i].first);

  // See if the PHI node can be merged to a single value.  This can happen in
  // loop cases when we get a PHI of itself and one other value.
  if (unsigned ConstVal = InsertedPHI->isConstantValuePHI()) {
    InsertedPHI->eraseFromParent();
    return ConstVal;
  }

  // If the client wants to know about all new instructions, tell it.
  if (InsertedPHIs) InsertedPHIs->push_back(InsertedPHI);

  DEBUG(dbgs() << "  Inserted PHI: " << *InsertedPHI << "\n");
  return InsertedPHI->getOperand(0).getReg();
}

static
MachineBasicBlock *findCorrespondingPred(const MachineInstr *MI,
                                         MachineOperand *U) {
  for (unsigned i = 1, e = MI->getNumOperands(); i != e; i += 2) {
    if (&MI->getOperand(i) == U)
      return MI->getOperand(i+1).getMBB();
  }

  llvm_unreachable("MachineOperand::getParent() failure?");
  return 0;
}

/// RewriteUse - Rewrite a use of the symbolic value.  This handles PHI nodes,
/// which use their value in the corresponding predecessor.
void MachineSSAUpdater::RewriteUse(MachineOperand &U) {
  MachineInstr *UseMI = U.getParent();
  unsigned NewVR = 0;
  if (UseMI->isPHI()) {
    MachineBasicBlock *SourceBB = findCorrespondingPred(UseMI, &U);
    NewVR = GetValueAtEndOfBlockInternal(SourceBB);
  } else {
    NewVR = GetValueInMiddleOfBlock(UseMI->getParent());
  }

  U.setReg(NewVR);
}

void MachineSSAUpdater::ReplaceRegWith(unsigned OldReg, unsigned NewReg) {
  MRI->replaceRegWith(OldReg, NewReg);

  AvailableValsTy &AvailableVals = getAvailableVals(AV);
  for (DenseMap<MachineBasicBlock*, unsigned>::iterator
         I = AvailableVals.begin(), E = AvailableVals.end(); I != E; ++I)
    if (I->second == OldReg)
      I->second = NewReg;
}

/// GetValueAtEndOfBlockInternal - Check to see if AvailableVals has an entry
/// for the specified BB and if so, return it.  If not, construct SSA form by
/// first calculating the required placement of PHIs and then inserting new
/// PHIs where needed.
unsigned MachineSSAUpdater::GetValueAtEndOfBlockInternal(MachineBasicBlock *BB){
  AvailableValsTy &AvailableVals = getAvailableVals(AV);
  if (unsigned V = AvailableVals[BB])
    return V;

  // Pool allocation used internally by GetValueAtEndOfBlock.
  BumpPtrAllocator Allocator;
  BBMapTy BBMapObj;
  BM = &BBMapObj;

  SmallVector<BBInfo*, 100> BlockList;
  BuildBlockList(BB, &BlockList, &Allocator);

  // Special case: bail out if BB is unreachable.
  if (BlockList.size() == 0) {
    BM = 0;
    // Insert an implicit_def to represent an undef value.
    MachineInstr *NewDef = InsertNewDef(TargetOpcode::IMPLICIT_DEF,
                                        BB, BB->getFirstTerminator(),
                                        VRC, MRI, TII);
    unsigned V = NewDef->getOperand(0).getReg();
    AvailableVals[BB] = V;
    return V;
  }

  FindDominators(&BlockList);
  FindPHIPlacement(&BlockList);
  FindAvailableVals(&BlockList);

  BM = 0;
  return BBMapObj[BB]->DefBB->AvailableVal;
}

/// FindPredecessorBlocks - Put the predecessors of Info->BB into the Preds
/// vector, set Info->NumPreds, and allocate space in Info->Preds.
static void FindPredecessorBlocks(MachineSSAUpdater::BBInfo *Info,
                                  SmallVectorImpl<MachineBasicBlock*> *Preds,
                                  BumpPtrAllocator *Allocator) {
  MachineBasicBlock *BB = Info->BB;
  for (MachineBasicBlock::pred_iterator PI = BB->pred_begin(),
         E = BB->pred_end(); PI != E; ++PI)
    Preds->push_back(*PI);

  Info->NumPreds = Preds->size();
  Info->Preds = static_cast<MachineSSAUpdater::BBInfo**>
    (Allocator->Allocate(Info->NumPreds * sizeof(MachineSSAUpdater::BBInfo*),
                         AlignOf<MachineSSAUpdater::BBInfo*>::Alignment));
}

/// BuildBlockList - Starting from the specified basic block, traverse back
/// through its predecessors until reaching blocks with known values.  Create
/// BBInfo structures for the blocks and append them to the block list.
void MachineSSAUpdater::BuildBlockList(MachineBasicBlock *BB,
                                       BlockListTy *BlockList,
                                       BumpPtrAllocator *Allocator) {
  AvailableValsTy &AvailableVals = getAvailableVals(AV);
  BBMapTy *BBMap = getBBMap(BM);
  SmallVector<BBInfo*, 10> RootList;
  SmallVector<BBInfo*, 64> WorkList;

  BBInfo *Info = new (*Allocator) BBInfo(BB, 0);
  (*BBMap)[BB] = Info;
  WorkList.push_back(Info);

  // Search backward from BB, creating BBInfos along the way and stopping when
  // reaching blocks that define the value.  Record those defining blocks on
  // the RootList.
  SmallVector<MachineBasicBlock*, 10> Preds;
  while (!WorkList.empty()) {
    Info = WorkList.pop_back_val();
    Preds.clear();
    FindPredecessorBlocks(Info, &Preds, Allocator);

    // Treat an unreachable predecessor as a definition with 'undef'.
    if (Info->NumPreds == 0) {
      // Insert an implicit_def to represent an undef value.
      MachineInstr *NewDef = InsertNewDef(TargetOpcode::IMPLICIT_DEF,
                                          Info->BB,
                                          Info->BB->getFirstTerminator(),
                                          VRC, MRI, TII);
      Info->AvailableVal = NewDef->getOperand(0).getReg();
      Info->DefBB = Info;
      RootList.push_back(Info);
      continue;
    }

    for (unsigned p = 0; p != Info->NumPreds; ++p) {
      MachineBasicBlock *Pred = Preds[p];
      // Check if BBMap already has a BBInfo for the predecessor block.
      BBMapTy::value_type &BBMapBucket = BBMap->FindAndConstruct(Pred);
      if (BBMapBucket.second) {
        Info->Preds[p] = BBMapBucket.second;
        continue;
      }

      // Create a new BBInfo for the predecessor.
      unsigned PredVal = AvailableVals.lookup(Pred);
      BBInfo *PredInfo = new (*Allocator) BBInfo(Pred, PredVal);
      BBMapBucket.second = PredInfo;
      Info->Preds[p] = PredInfo;

      if (PredInfo->AvailableVal) {
        RootList.push_back(PredInfo);
        continue;
      }
      WorkList.push_back(PredInfo);
    }
  }

  // Now that we know what blocks are backwards-reachable from the starting
  // block, do a forward depth-first traversal to assign postorder numbers
  // to those blocks.
  BBInfo *PseudoEntry = new (*Allocator) BBInfo(0, 0);
  unsigned BlkNum = 1;

  // Initialize the worklist with the roots from the backward traversal.
  while (!RootList.empty()) {
    Info = RootList.pop_back_val();
    Info->IDom = PseudoEntry;
    Info->BlkNum = -1;
    WorkList.push_back(Info);
  }

  while (!WorkList.empty()) {
    Info = WorkList.back();

    if (Info->BlkNum == -2) {
      // All the successors have been handled; assign the postorder number.
      Info->BlkNum = BlkNum++;
      // If not a root, put it on the BlockList.
      if (!Info->AvailableVal)
        BlockList->push_back(Info);
      WorkList.pop_back();
      continue;
    }

    // Leave this entry on the worklist, but set its BlkNum to mark that its
    // successors have been put on the worklist.  When it returns to the top
    // the list, after handling its successors, it will be assigned a number.
    Info->BlkNum = -2;

    // Add unvisited successors to the work list.
    for (MachineBasicBlock::succ_iterator SI = Info->BB->succ_begin(),
           E = Info->BB->succ_end(); SI != E; ++SI) {
      BBInfo *SuccInfo = (*BBMap)[*SI];
      if (!SuccInfo || SuccInfo->BlkNum)
        continue;
      SuccInfo->BlkNum = -1;
      WorkList.push_back(SuccInfo);
    }
  }
  PseudoEntry->BlkNum = BlkNum;
}

/// IntersectDominators - This is the dataflow lattice "meet" operation for
/// finding dominators.  Given two basic blocks, it walks up the dominator
/// tree until it finds a common dominator of both.  It uses the postorder
/// number of the blocks to determine how to do that.
static MachineSSAUpdater::BBInfo *
IntersectDominators(MachineSSAUpdater::BBInfo *Blk1,
                    MachineSSAUpdater::BBInfo *Blk2) {
  while (Blk1 != Blk2) {
    while (Blk1->BlkNum < Blk2->BlkNum) {
      Blk1 = Blk1->IDom;
      if (!Blk1)
        return Blk2;
    }
    while (Blk2->BlkNum < Blk1->BlkNum) {
      Blk2 = Blk2->IDom;
      if (!Blk2)
        return Blk1;
    }
  }
  return Blk1;
}

/// FindDominators - Calculate the dominator tree for the subset of the CFG
/// corresponding to the basic blocks on the BlockList.  This uses the
/// algorithm from: "A Simple, Fast Dominance Algorithm" by Cooper, Harvey and
/// Kennedy, published in Software--Practice and Experience, 2001, 4:1-10.
/// Because the CFG subset does not include any edges leading into blocks that
/// define the value, the results are not the usual dominator tree.  The CFG
/// subset has a single pseudo-entry node with edges to a set of root nodes
/// for blocks that define the value.  The dominators for this subset CFG are
/// not the standard dominators but they are adequate for placing PHIs within
/// the subset CFG.
void MachineSSAUpdater::FindDominators(BlockListTy *BlockList) {
  bool Changed;
  do {
    Changed = false;
    // Iterate over the list in reverse order, i.e., forward on CFG edges.
    for (BlockListTy::reverse_iterator I = BlockList->rbegin(),
           E = BlockList->rend(); I != E; ++I) {
      BBInfo *Info = *I;

      // Start with the first predecessor.
      assert(Info->NumPreds > 0 && "unreachable block");
      BBInfo *NewIDom = Info->Preds[0];

      // Iterate through the block's other predecessors.
      for (unsigned p = 1; p != Info->NumPreds; ++p) {
        BBInfo *Pred = Info->Preds[p];
        NewIDom = IntersectDominators(NewIDom, Pred);
      }

      // Check if the IDom value has changed.
      if (NewIDom != Info->IDom) {
        Info->IDom = NewIDom;
        Changed = true;
      }
    }
  } while (Changed);
}

/// IsDefInDomFrontier - Search up the dominator tree from Pred to IDom for
/// any blocks containing definitions of the value.  If one is found, then the
/// successor of Pred is in the dominance frontier for the definition, and
/// this function returns true.
static bool IsDefInDomFrontier(const MachineSSAUpdater::BBInfo *Pred,
                               const MachineSSAUpdater::BBInfo *IDom) {
  for (; Pred != IDom; Pred = Pred->IDom) {
    if (Pred->DefBB == Pred)
      return true;
  }
  return false;
}

/// FindPHIPlacement - PHIs are needed in the iterated dominance frontiers of
/// the known definitions.  Iteratively add PHIs in the dom frontiers until
/// nothing changes.  Along the way, keep track of the nearest dominating
/// definitions for non-PHI blocks.
void MachineSSAUpdater::FindPHIPlacement(BlockListTy *BlockList) {
  bool Changed;
  do {
    Changed = false;
    // Iterate over the list in reverse order, i.e., forward on CFG edges.
    for (BlockListTy::reverse_iterator I = BlockList->rbegin(),
           E = BlockList->rend(); I != E; ++I) {
      BBInfo *Info = *I;

      // If this block already needs a PHI, there is nothing to do here.
      if (Info->DefBB == Info)
        continue;

      // Default to use the same def as the immediate dominator.
      BBInfo *NewDefBB = Info->IDom->DefBB;
      for (unsigned p = 0; p != Info->NumPreds; ++p) {
        if (IsDefInDomFrontier(Info->Preds[p], Info->IDom)) {
          // Need a PHI here.
          NewDefBB = Info;
          break;
        }
      }

      // Check if anything changed.
      if (NewDefBB != Info->DefBB) {
        Info->DefBB = NewDefBB;
        Changed = true;
      }
    }
  } while (Changed);
}

/// FindAvailableVal - If this block requires a PHI, first check if an existing
/// PHI matches the PHI placement and reaching definitions computed earlier,
/// and if not, create a new PHI.  Visit all the block's predecessors to
/// calculate the available value for each one and fill in the incoming values
/// for a new PHI.
void MachineSSAUpdater::FindAvailableVals(BlockListTy *BlockList) {
  AvailableValsTy &AvailableVals = getAvailableVals(AV);

  // Go through the worklist in forward order (i.e., backward through the CFG)
  // and check if existing PHIs can be used.  If not, create empty PHIs where
  // they are needed.
  for (BlockListTy::iterator I = BlockList->begin(), E = BlockList->end();
       I != E; ++I) {
    BBInfo *Info = *I;
    // Check if there needs to be a PHI in BB.
    if (Info->DefBB != Info)
      continue;

    // Look for an existing PHI.
    FindExistingPHI(Info->BB, BlockList);
    if (Info->AvailableVal)
      continue;

    MachineBasicBlock::iterator Loc =
      Info->BB->empty() ? Info->BB->end() : Info->BB->front();
    MachineInstr *InsertedPHI = InsertNewDef(TargetOpcode::PHI, Info->BB, Loc,
                                             VRC, MRI, TII);
    unsigned PHI = InsertedPHI->getOperand(0).getReg();
    Info->AvailableVal = PHI;
    AvailableVals[Info->BB] = PHI;
  }

  // Now go back through the worklist in reverse order to fill in the arguments
  // for any new PHIs added in the forward traversal.
  for (BlockListTy::reverse_iterator I = BlockList->rbegin(),
         E = BlockList->rend(); I != E; ++I) {
    BBInfo *Info = *I;

    if (Info->DefBB != Info) {
      // Record the available value at join nodes to speed up subsequent
      // uses of this SSAUpdater for the same value.
      if (Info->NumPreds > 1)
        AvailableVals[Info->BB] = Info->DefBB->AvailableVal;
      continue;
    }

    // Check if this block contains a newly added PHI.
    unsigned PHI = Info->AvailableVal;
    MachineInstr *InsertedPHI = MRI->getVRegDef(PHI);
    if (!InsertedPHI->isPHI() || InsertedPHI->getNumOperands() > 1)
      continue;

    // Iterate through the block's predecessors.
    MachineInstrBuilder MIB(InsertedPHI);
    for (unsigned p = 0; p != Info->NumPreds; ++p) {
      BBInfo *PredInfo = Info->Preds[p];
      MachineBasicBlock *Pred = PredInfo->BB;
      // Skip to the nearest preceding definition.
      if (PredInfo->DefBB != PredInfo)
        PredInfo = PredInfo->DefBB;
      MIB.addReg(PredInfo->AvailableVal).addMBB(Pred);
    }

    DEBUG(dbgs() << "  Inserted PHI: " << *InsertedPHI << "\n");

    // If the client wants to know about all new instructions, tell it.
    if (InsertedPHIs) InsertedPHIs->push_back(InsertedPHI);
  }
}

/// FindExistingPHI - Look through the PHI nodes in a block to see if any of
/// them match what is needed.
void MachineSSAUpdater::FindExistingPHI(MachineBasicBlock *BB,
                                        BlockListTy *BlockList) {
  for (MachineBasicBlock::iterator BBI = BB->begin(), BBE = BB->end();
       BBI != BBE && BBI->isPHI(); ++BBI) {
    if (CheckIfPHIMatches(BBI)) {
      RecordMatchingPHI(BBI);
      break;
    }
    // Match failed: clear all the PHITag values.
    for (BlockListTy::iterator I = BlockList->begin(), E = BlockList->end();
         I != E; ++I)
      (*I)->PHITag = 0;
  }
}

/// CheckIfPHIMatches - Check if a PHI node matches the placement and values
/// in the BBMap.
bool MachineSSAUpdater::CheckIfPHIMatches(MachineInstr *PHI) {
  BBMapTy *BBMap = getBBMap(BM);
  SmallVector<MachineInstr*, 20> WorkList;
  WorkList.push_back(PHI);

  // Mark that the block containing this PHI has been visited.
  (*BBMap)[PHI->getParent()]->PHITag = PHI;

  while (!WorkList.empty()) {
    PHI = WorkList.pop_back_val();

    // Iterate through the PHI's incoming values.
    for (unsigned i = 1, e = PHI->getNumOperands(); i != e; i += 2) {
      unsigned IncomingVal = PHI->getOperand(i).getReg();
      BBInfo *PredInfo = (*BBMap)[PHI->getOperand(i+1).getMBB()];
      // Skip to the nearest preceding definition.
      if (PredInfo->DefBB != PredInfo)
        PredInfo = PredInfo->DefBB;

      // Check if it matches the expected value.
      if (PredInfo->AvailableVal) {
        if (IncomingVal == PredInfo->AvailableVal)
          continue;
        return false;
      }

      // Check if the value is a PHI in the correct block.
      MachineInstr *IncomingPHIVal = MRI->getVRegDef(IncomingVal);
      if (!IncomingPHIVal->isPHI() ||
          IncomingPHIVal->getParent() != PredInfo->BB)
        return false;

      // If this block has already been visited, check if this PHI matches.
      if (PredInfo->PHITag) {
        if (IncomingPHIVal == PredInfo->PHITag)
          continue;
        return false;
      }
      PredInfo->PHITag = IncomingPHIVal;

      WorkList.push_back(IncomingPHIVal);
    }
  }
  return true;
}

/// RecordMatchingPHI - For a PHI node that matches, record it and its input
/// PHIs in both the BBMap and the AvailableVals mapping.
void MachineSSAUpdater::RecordMatchingPHI(MachineInstr *PHI) {
  BBMapTy *BBMap = getBBMap(BM);
  AvailableValsTy &AvailableVals = getAvailableVals(AV);
  SmallVector<MachineInstr*, 20> WorkList;
  WorkList.push_back(PHI);

  // Record this PHI.
  MachineBasicBlock *BB = PHI->getParent();
  AvailableVals[BB] = PHI->getOperand(0).getReg();
  (*BBMap)[BB]->AvailableVal = PHI->getOperand(0).getReg();

  while (!WorkList.empty()) {
    PHI = WorkList.pop_back_val();

    // Iterate through the PHI's incoming values.
    for (unsigned i = 1, e = PHI->getNumOperands(); i != e; i += 2) {
      unsigned IncomingVal = PHI->getOperand(i).getReg();
      MachineInstr *IncomingPHIVal = MRI->getVRegDef(IncomingVal);
      if (!IncomingPHIVal->isPHI()) continue;
      BB = IncomingPHIVal->getParent();
      BBInfo *Info = (*BBMap)[BB];
      if (!Info || Info->AvailableVal)
        continue;

      // Record the PHI and add it to the worklist.
      AvailableVals[BB] = IncomingVal;
      Info->AvailableVal = IncomingVal;
      WorkList.push_back(IncomingPHIVal);
    }
  }
}
