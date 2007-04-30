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
  while (I != begin() && TII.isTerminatorInstr((--I)->getOpcode()));
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
  if (livein_begin() != livein_end()) {
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

void MachineBasicBlock::removeSuccessor(succ_iterator I) {
  assert(I != Successors.end() && "Not a current successor!");
  (*I)->removePredecessor(this);
  Successors.erase(I);
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
