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
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "Support/LeakDetector.h"
using namespace llvm;

const MachineFunction *MachineBasicBlock::getParent() const {
  // Get the parent by getting the Function parent of the basic block, and
  // getting the MachineFunction from it.
  return &MachineFunction::get(getBasicBlock()->getParent());
}


MachineInstr* ilist_traits<MachineInstr>::createNode()
{
    MachineInstr* dummy = new MachineInstr(0, 0);
    LeakDetector::removeGarbageObject(dummy);
    return dummy;
}

void ilist_traits<MachineInstr>::addNodeToList(MachineInstr* N)
{
    assert(N->parent == 0 && "machine instruction already in a basic block");
    N->parent = parent;
    LeakDetector::removeGarbageObject(N);
}

void ilist_traits<MachineInstr>::removeNodeFromList(MachineInstr* N)
{
    assert(N->parent != 0 && "machine instruction not in a basic block");
    N->parent = 0;
    LeakDetector::addGarbageObject(N);
}

void ilist_traits<MachineInstr>::transferNodesFromList(
    iplist<MachineInstr, ilist_traits<MachineInstr> >& toList,
    ilist_iterator<MachineInstr> first,
    ilist_iterator<MachineInstr> last)
{
    if (parent != toList.parent)
        for (; first != last; ++first)
            first->parent = toList.parent;
}

MachineBasicBlock::iterator MachineBasicBlock::getFirstTerminator()
{
  const TargetInstrInfo& TII = MachineFunction::get(
    getBasicBlock()->getParent()).getTarget().getInstrInfo();
  iterator I = end();
  while (I != begin() && TII.isTerminatorInstr((--I)->getOpcode()));
  if (I != end() && !TII.isTerminatorInstr(I->getOpcode())) ++I;
  return I;
}

void MachineBasicBlock::dump() const
{
    print(std::cerr);
}

void MachineBasicBlock::print(std::ostream &OS) const
{
    const BasicBlock *LBB = getBasicBlock();
    OS << "\n" << LBB->getName() << " (" << (const void*)LBB << "):\n";
    for (const_iterator I = begin(); I != end(); ++I) {
        OS << "\t";
        I->print(OS, MachineFunction::get(LBB->getParent()).getTarget());
    }
}
