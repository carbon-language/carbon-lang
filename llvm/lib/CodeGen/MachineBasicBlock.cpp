//===-- llvm/CodeGen/MachineBasicBlock.cpp ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Collect the sequence of machine instructions for a basic block.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/BasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Target/MRegisterInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/LeakDetector.h"
#include <algorithm>
using namespace llvm;

MachineBasicBlock::~MachineBasicBlock() {
  LeakDetector::removeGarbageObject(this);
}

std::ostream& llvm::operator<<(std::ostream &OS, const MachineBasicBlock &MBB) {
  MBB.print(OS);
  return OS;
}

// MBBs start out as #-1. When a MBB is added to a MachineFunction, it
// gets the next available unique MBB number. If it is removed from a
// MachineFunction, it goes back to being #-1.
void ilist_traits<MachineBasicBlock>::addNodeToList(MachineBasicBlock* N) {
  assert(N->Parent == 0 && "machine instruction already in a basic block");
  N->Parent = Parent;
  N->Number = Parent->addToMBBNumbering(N);
  LeakDetector::removeGarbageObject(N);
}

void ilist_traits<MachineBasicBlock>::removeNodeFromList(MachineBasicBlock* N) {
  assert(N->Parent != 0 && "machine instruction not in a basic block");
  N->Parent->removeFromMBBNumbering(N->Number);
  N->Number = -1;
  N->Parent = 0;
  LeakDetector::addGarbageObject(N);
}


MachineInstr* ilist_traits<MachineInstr>::createSentinel() {
  MachineInstr* dummy = new MachineInstr();
  LeakDetector::removeGarbageObject(dummy);
  return dummy;
}

void ilist_traits<MachineInstr>::addNodeToList(MachineInstr* N) {
  assert(N->parent == 0 && "machine instruction already in a basic block");
  N->parent = parent;
  LeakDetector::removeGarbageObject(N);
}

void ilist_traits<MachineInstr>::removeNodeFromList(MachineInstr* N) {
  assert(N->parent != 0 && "machine instruction not in a basic block");
  N->parent = 0;
  LeakDetector::addGarbageObject(N);
}

void ilist_traits<MachineInstr>::transferNodesFromList(
  iplist<MachineInstr, ilist_traits<MachineInstr> >& fromList,
  ilist_iterator<MachineInstr> first,
  ilist_iterator<MachineInstr> last) {
  if (parent != fromList.parent)
    for (; first != last; ++first)
      first->parent = parent;
}

MachineBasicBlock::iterator MachineBasicBlock::getFirstTerminator() {
  const TargetInstrInfo& TII = *getParent()->getTarget().getInstrInfo();
  iterator I = end();
  while (I != begin() && TII.isTerminatorInstr((--I)->getOpcode()))
    ; /*noop */
  if (I != end() && !TII.isTerminatorInstr(I->getOpcode())) ++I;
  return I;
}

void MachineBasicBlock::dump() const {
  print(*cerr.stream());
}

static inline void OutputReg(std::ostream &os, unsigned RegNo,
                             const MRegisterInfo *MRI = 0) {
  if (!RegNo || MRegisterInfo::isPhysicalRegister(RegNo)) {
    if (MRI)
      os << " %" << MRI->get(RegNo).Name;
    else
      os << " %mreg(" << RegNo << ")";
  } else
    os << " %reg" << RegNo;
}

void MachineBasicBlock::print(std::ostream &OS) const {
  const MachineFunction *MF = getParent();
  if(!MF) {
    OS << "Can't print out MachineBasicBlock because parent MachineFunction"
       << " is null\n";
    return;
  }

  const BasicBlock *LBB = getBasicBlock();
  OS << "\n";
  if (LBB) OS << LBB->getName() << ": ";
  OS << (const void*)this
     << ", LLVM BB @" << (const void*) LBB << ", ID#" << getNumber();
  if (isLandingPad()) OS << ", EH LANDING PAD";
  OS << ":\n";

  const MRegisterInfo *MRI = MF->getTarget().getRegisterInfo();  
  if (!livein_empty()) {
    OS << "Live Ins:";
    for (const_livein_iterator I = livein_begin(),E = livein_end(); I != E; ++I)
      OutputReg(OS, *I, MRI);
    OS << "\n";
  }
  // Print the preds of this block according to the CFG.
  if (!pred_empty()) {
    OS << "    Predecessors according to CFG:";
    for (const_pred_iterator PI = pred_begin(), E = pred_end(); PI != E; ++PI)
      OS << " " << *PI << " (#" << (*PI)->getNumber() << ")";
    OS << "\n";
  }
  
  for (const_iterator I = begin(); I != end(); ++I) {
    OS << "\t";
    I->print(OS, &getParent()->getTarget());
  }

  // Print the successors of this block according to the CFG.
  if (!succ_empty()) {
    OS << "    Successors according to CFG:";
    for (const_succ_iterator SI = succ_begin(), E = succ_end(); SI != E; ++SI)
      OS << " " << *SI << " (#" << (*SI)->getNumber() << ")";
    OS << "\n";
  }
}

void MachineBasicBlock::removeLiveIn(unsigned Reg) {
  livein_iterator I = std::find(livein_begin(), livein_end(), Reg);
  assert(I != livein_end() && "Not a live in!");
  LiveIns.erase(I);
}

void MachineBasicBlock::moveBefore(MachineBasicBlock *NewAfter) {
  MachineFunction::BasicBlockListType &BBList =getParent()->getBasicBlockList();
  getParent()->getBasicBlockList().splice(NewAfter, BBList, this);
}

void MachineBasicBlock::moveAfter(MachineBasicBlock *NewBefore) {
  MachineFunction::BasicBlockListType &BBList =getParent()->getBasicBlockList();
  MachineFunction::iterator BBI = NewBefore;
  getParent()->getBasicBlockList().splice(++BBI, BBList, this);
}


void MachineBasicBlock::addSuccessor(MachineBasicBlock *succ) {
  Successors.push_back(succ);
  succ->addPredecessor(this);
}

void MachineBasicBlock::removeSuccessor(MachineBasicBlock *succ) {
  succ->removePredecessor(this);
  succ_iterator I = std::find(Successors.begin(), Successors.end(), succ);
  assert(I != Successors.end() && "Not a current successor!");
  Successors.erase(I);
}

MachineBasicBlock::succ_iterator MachineBasicBlock::removeSuccessor(succ_iterator I) {
  assert(I != Successors.end() && "Not a current successor!");
  (*I)->removePredecessor(this);
  return(Successors.erase(I));
}

void MachineBasicBlock::addPredecessor(MachineBasicBlock *pred) {
  Predecessors.push_back(pred);
}

void MachineBasicBlock::removePredecessor(MachineBasicBlock *pred) {
  std::vector<MachineBasicBlock *>::iterator I =
    std::find(Predecessors.begin(), Predecessors.end(), pred);
  assert(I != Predecessors.end() && "Pred is not a predecessor of this block!");
  Predecessors.erase(I);
}

bool MachineBasicBlock::isSuccessor(MachineBasicBlock *MBB) const {
  std::vector<MachineBasicBlock *>::const_iterator I =
    std::find(Successors.begin(), Successors.end(), MBB);
  return I != Successors.end();
}

/// ReplaceUsesOfBlockWith - Given a machine basic block that branched to
/// 'Old', change the code and CFG so that it branches to 'New' instead.
void MachineBasicBlock::ReplaceUsesOfBlockWith(MachineBasicBlock *Old,
                                               MachineBasicBlock *New) {
  assert(Old != New && "Cannot replace self with self!");

  MachineBasicBlock::iterator I = end();
  while (I != begin()) {
    --I;
    if (!(I->getInstrDescriptor()->Flags & M_TERMINATOR_FLAG)) break;

    // Scan the operands of this machine instruction, replacing any uses of Old
    // with New.
    for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i)
      if (I->getOperand(i).isMachineBasicBlock() &&
          I->getOperand(i).getMachineBasicBlock() == Old)
        I->getOperand(i).setMachineBasicBlock(New);
  }

  // Update the successor information.  If New was already a successor, just
  // remove the link to Old instead of creating another one.  PR 1444.
  removeSuccessor(Old);
  if (!isSuccessor(New))
    addSuccessor(New);
}

/// CorrectExtraCFGEdges - Various pieces of code can cause excess edges in the
/// CFG to be inserted.  If we have proven that MBB can only branch to DestA and
/// DestB, remove any other MBB successors from the CFG.  DestA and DestB can
/// be null.
/// Besides DestA and DestB, retain other edges leading to LandingPads (currently
/// there can be only one; we don't check or require that here).
/// Note it is possible that DestA and/or DestB are LandingPads.
bool MachineBasicBlock::CorrectExtraCFGEdges(MachineBasicBlock *DestA,
                                             MachineBasicBlock *DestB,
                                             bool isCond) {
  bool MadeChange = false;
  bool AddedFallThrough = false;

  MachineBasicBlock *FallThru = getNext();
  
  // If this block ends with a conditional branch that falls through to its
  // successor, set DestB as the successor.
  if (isCond) {
    if (DestB == 0 && FallThru != getParent()->end()) {
      DestB = FallThru;
      AddedFallThrough = true;
    }
  } else {
    // If this is an unconditional branch with no explicit dest, it must just be
    // a fallthrough into DestB.
    if (DestA == 0 && FallThru != getParent()->end()) {
      DestA = FallThru;
      AddedFallThrough = true;
    }
  }
  
  MachineBasicBlock::succ_iterator SI = succ_begin();
  MachineBasicBlock *OrigDestA = DestA, *OrigDestB = DestB;
  while (SI != succ_end()) {
    if (*SI == DestA && DestA == DestB) {
      DestA = DestB = 0;
      ++SI;
    } else if (*SI == DestA) {
      DestA = 0;
      ++SI;
    } else if (*SI == DestB) {
      DestB = 0;
      ++SI;
    } else if ((*SI)->isLandingPad() && 
               *SI!=OrigDestA && *SI!=OrigDestB) {
      ++SI;
    } else {
      // Otherwise, this is a superfluous edge, remove it.
      SI = removeSuccessor(SI);
      MadeChange = true;
    }
  }
  if (!AddedFallThrough) {
    assert(DestA == 0 && DestB == 0 &&
           "MachineCFG is missing edges!");
  } else if (isCond) {
    assert(DestA == 0 && "MachineCFG is missing edges!");
  }
  return MadeChange;
}
