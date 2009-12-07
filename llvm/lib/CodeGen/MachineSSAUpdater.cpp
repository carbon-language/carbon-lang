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
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

typedef DenseMap<MachineBasicBlock*, unsigned> AvailableValsTy;
typedef std::vector<std::pair<MachineBasicBlock*, unsigned> >
                IncomingPredInfoTy;

static AvailableValsTy &getAvailableVals(void *AV) {
  return *static_cast<AvailableValsTy*>(AV);
}

static IncomingPredInfoTy &getIncomingPredInfo(void *IPI) {
  return *static_cast<IncomingPredInfoTy*>(IPI);
}


MachineSSAUpdater::MachineSSAUpdater(MachineFunction &MF,
                                     SmallVectorImpl<MachineInstr*> *NewPHI)
  : AV(0), IPI(0), InsertedPHIs(NewPHI) {
  TII = MF.getTarget().getInstrInfo();
  MRI = &MF.getRegInfo();
}

MachineSSAUpdater::~MachineSSAUpdater() {
  delete &getAvailableVals(AV);
  delete &getIncomingPredInfo(IPI);
}

/// Initialize - Reset this object to get ready for a new set of SSA
/// updates.  ProtoValue is the value used to name PHI nodes.
void MachineSSAUpdater::Initialize(unsigned V) {
  if (AV == 0)
    AV = new AvailableValsTy();
  else
    getAvailableVals(AV).clear();

  if (IPI == 0)
    IPI = new IncomingPredInfoTy();
  else
    getIncomingPredInfo(IPI).clear();

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
  if (I->getOpcode() != TargetInstrInfo::PHI)
    return 0;

  AvailableValsTy AVals;
  for (unsigned i = 0, e = PredValues.size(); i != e; ++i)
    AVals[PredValues[i].first] = PredValues[i].second;
  while (I != BB->end() && I->getOpcode() == TargetInstrInfo::PHI) {
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
  return BuildMI(*BB, I, DebugLoc::getUnknownLoc(), TII->get(Opcode), NewVR);
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
  if (!getAvailableVals(AV).count(BB))
    return GetValueAtEndOfBlockInternal(BB);

  // If there are no predecessors, just return undef.
  if (BB->pred_empty()) {
    // Insert an implicit_def to represent an undef value.
    MachineInstr *NewDef = InsertNewDef(TargetInstrInfo::IMPLICIT_DEF,
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
  MachineInstr *InsertedPHI = InsertNewDef(TargetInstrInfo::PHI, BB,
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

  DEBUG(errs() << "  Inserted PHI: " << *InsertedPHI << "\n");
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
  if (UseMI->getOpcode() == TargetInstrInfo::PHI) {
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
/// walking predecessors inserting PHI nodes as needed until we get to a block
/// where the value is available.
///
unsigned MachineSSAUpdater::GetValueAtEndOfBlockInternal(MachineBasicBlock *BB){
  AvailableValsTy &AvailableVals = getAvailableVals(AV);

  // Query AvailableVals by doing an insertion of null.
  std::pair<AvailableValsTy::iterator, bool> InsertRes =
    AvailableVals.insert(std::make_pair(BB, 0));

  // Handle the case when the insertion fails because we have already seen BB.
  if (!InsertRes.second) {
    // If the insertion failed, there are two cases.  The first case is that the
    // value is already available for the specified block.  If we get this, just
    // return the value.
    if (InsertRes.first->second != 0)
      return InsertRes.first->second;

    // Otherwise, if the value we find is null, then this is the value is not
    // known but it is being computed elsewhere in our recursion.  This means
    // that we have a cycle.  Handle this by inserting a PHI node and returning
    // it.  When we get back to the first instance of the recursion we will fill
    // in the PHI node.
    MachineBasicBlock::iterator Loc = BB->empty() ? BB->end() : BB->front();
    MachineInstr *NewPHI = InsertNewDef(TargetInstrInfo::PHI, BB, Loc,
                                        VRC, MRI,TII);
    unsigned NewVR = NewPHI->getOperand(0).getReg();
    InsertRes.first->second = NewVR;
    return NewVR;
  }

  // If there are no predecessors, then we must have found an unreachable block
  // just return 'undef'.  Since there are no predecessors, InsertRes must not
  // be invalidated.
  if (BB->pred_empty()) {
    // Insert an implicit_def to represent an undef value.
    MachineInstr *NewDef = InsertNewDef(TargetInstrInfo::IMPLICIT_DEF,
                                        BB, BB->getFirstTerminator(),
                                        VRC, MRI, TII);
    return InsertRes.first->second = NewDef->getOperand(0).getReg();
  }

  // Okay, the value isn't in the map and we just inserted a null in the entry
  // to indicate that we're processing the block.  Since we have no idea what
  // value is in this block, we have to recurse through our predecessors.
  //
  // While we're walking our predecessors, we keep track of them in a vector,
  // then insert a PHI node in the end if we actually need one.  We could use a
  // smallvector here, but that would take a lot of stack space for every level
  // of the recursion, just use IncomingPredInfo as an explicit stack.
  IncomingPredInfoTy &IncomingPredInfo = getIncomingPredInfo(IPI);
  unsigned FirstPredInfoEntry = IncomingPredInfo.size();

  // As we're walking the predecessors, keep track of whether they are all
  // producing the same value.  If so, this value will capture it, if not, it
  // will get reset to null.  We distinguish the no-predecessor case explicitly
  // below.
  unsigned SingularValue = 0;
  bool isFirstPred = true;
  for (MachineBasicBlock::pred_iterator PI = BB->pred_begin(),
         E = BB->pred_end(); PI != E; ++PI) {
    MachineBasicBlock *PredBB = *PI;
    unsigned PredVal = GetValueAtEndOfBlockInternal(PredBB);
    IncomingPredInfo.push_back(std::make_pair(PredBB, PredVal));

    // Compute SingularValue.
    if (isFirstPred) {
      SingularValue = PredVal;
      isFirstPred = false;
    } else if (PredVal != SingularValue)
      SingularValue = 0;
  }

  /// Look up BB's entry in AvailableVals.  'InsertRes' may be invalidated.  If
  /// this block is involved in a loop, a no-entry PHI node will have been
  /// inserted as InsertedVal.  Otherwise, we'll still have the null we inserted
  /// above.
  unsigned &InsertedVal = AvailableVals[BB];

  // If all the predecessor values are the same then we don't need to insert a
  // PHI.  This is the simple and common case.
  if (SingularValue) {
    // If a PHI node got inserted, replace it with the singlar value and delete
    // it.
    if (InsertedVal) {
      MachineInstr *OldVal = MRI->getVRegDef(InsertedVal);
      // Be careful about dead loops.  These RAUW's also update InsertedVal.
      assert(InsertedVal != SingularValue && "Dead loop?");
      ReplaceRegWith(InsertedVal, SingularValue);
      OldVal->eraseFromParent();
    }

    InsertedVal = SingularValue;

    // Drop the entries we added in IncomingPredInfo to restore the stack.
    IncomingPredInfo.erase(IncomingPredInfo.begin()+FirstPredInfoEntry,
                           IncomingPredInfo.end());
    return InsertedVal;
  }


  // Otherwise, we do need a PHI: insert one now if we don't already have one.
  MachineInstr *InsertedPHI;
  if (InsertedVal == 0) {
    MachineBasicBlock::iterator Loc = BB->empty() ? BB->end() : BB->front();
    InsertedPHI = InsertNewDef(TargetInstrInfo::PHI, BB, Loc,
                               VRC, MRI, TII);
    InsertedVal = InsertedPHI->getOperand(0).getReg();
  } else {
    InsertedPHI = MRI->getVRegDef(InsertedVal);
  }

  // Fill in all the predecessors of the PHI.
  MachineInstrBuilder MIB(InsertedPHI);
  for (IncomingPredInfoTy::iterator I =
         IncomingPredInfo.begin()+FirstPredInfoEntry,
         E = IncomingPredInfo.end(); I != E; ++I)
    MIB.addReg(I->second).addMBB(I->first);

  // Drop the entries we added in IncomingPredInfo to restore the stack.
  IncomingPredInfo.erase(IncomingPredInfo.begin()+FirstPredInfoEntry,
                         IncomingPredInfo.end());

  // See if the PHI node can be merged to a single value.  This can happen in
  // loop cases when we get a PHI of itself and one other value.
  if (unsigned ConstVal = InsertedPHI->isConstantValuePHI()) {
    MRI->replaceRegWith(InsertedVal, ConstVal);
    InsertedPHI->eraseFromParent();
    InsertedVal = ConstVal;
  } else {
    DEBUG(errs() << "  Inserted PHI: " << *InsertedPHI << "\n");

    // If the client wants to know about all new instructions, tell it.
    if (InsertedPHIs) InsertedPHIs->push_back(InsertedPHI);
  }

  return InsertedVal;
}
