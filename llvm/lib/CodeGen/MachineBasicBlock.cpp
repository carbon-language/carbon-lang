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

MachineBasicBlock::~MachineBasicBlock() {
  LeakDetector::removeGarbageObject(this);
}
  


// MBBs start out as #-1. When a MBB is added to a MachineFunction, it 
// gets the next available unique MBB number. If it is removed from a
// MachineFunction, it goes back to being #-1.
void ilist_traits<MachineBasicBlock>::addNodeToList (MachineBasicBlock* N)
{
  assert(N->Parent == 0 && "machine instruction already in a basic block");
  N->Parent = Parent;
  N->Number = Parent->getNextMBBNumber();
  LeakDetector::removeGarbageObject(N);
  

}

void ilist_traits<MachineBasicBlock>::removeNodeFromList (MachineBasicBlock* N)
{
  assert(N->Parent != 0 && "machine instruction not in a basic block");
  N->Parent = 0;
  N->Number = -1;
  LeakDetector::addGarbageObject(N);
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
  const TargetInstrInfo& TII = *getParent()->getTarget().getInstrInfo();
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
  if(!getParent()) {
    OS << "Can't print out MachineBasicBlock because parent MachineFunction is null\n";
    return;
  }
    const BasicBlock *LBB = getBasicBlock();
    if(LBB)
      OS << "\n" << LBB->getName() << " (" << (const void*)LBB << "):\n";
    for (const_iterator I = begin(); I != end(); ++I) {
        OS << "\t";
        I->print(OS, getParent()->getTarget());
    }
}
