//===-- PeepholeOptimizer.cpp - X86 Peephole Optimizer --------------------===//
//
// This file contains a peephole optimizer for the X86.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

namespace {
  struct PH : public MachineFunctionPass {
    virtual bool runOnMachineFunction(MachineFunction &MF);

    bool PeepholeOptimize(MachineBasicBlock &MBB,
			  MachineBasicBlock::iterator &I);

    virtual const char *getPassName() const { return "X86 Peephole Optimizer"; }
  };
}

Pass *createX86PeepholeOptimizerPass() { return new PH(); }

bool PH::runOnMachineFunction(MachineFunction &MF) {
  bool Changed = false;

  for (MachineFunction::iterator BI = MF.begin(), E = MF.end(); BI != E; ++BI)
    for (MachineBasicBlock::iterator I = BI->begin(), E = BI->end(); I != E; )
      if (PeepholeOptimize(*BI, I))
	Changed = true;
      else
	++I;

  return Changed;
}


bool PH::PeepholeOptimize(MachineBasicBlock &MBB,
			  MachineBasicBlock::iterator &I) {
  MachineInstr *MI = *I;
  MachineInstr *Next = (I+1 != MBB.end()) ? *(I+1) : 0;
  unsigned Size = 0;
  switch (MI->getOpcode()) {
  case X86::MOVrr8:
  case X86::MOVrr16:
  case X86::MOVrr32:   // Destroy X = X copies...
    if (MI->getOperand(0).getReg() == MI->getOperand(1).getReg()) {
      I = MBB.erase(I);
      delete MI;
      return true;
    }
    return false;

#if 0
  case X86::MOVir32: Size++;
  case X86::MOVir16: Size++;
  case X86::MOVir8:
    // FIXME: We can only do this transformation if we know that flags are not
    // used here, because XOR clobbers the flags!
    if (MI->getOperand(1).isImmediate()) {         // avoid mov EAX, <value>
      int Val = MI->getOperand(1).getImmedValue();
      if (Val == 0) {                              // mov EAX, 0 -> xor EAX, EAX
	static const unsigned Opcode[] ={X86::XORrr8,X86::XORrr16,X86::XORrr32};
	unsigned Reg = MI->getOperand(0).getReg();
	*I = BuildMI(Opcode[Size], 2, Reg).addReg(Reg).addReg(Reg);
	delete MI;
	return true;
      } else if (Val == -1) {                     // mov EAX, -1 -> or EAX, -1
	// TODO: 'or Reg, -1' has a smaller encoding than 'mov Reg, -1'
      }
    }
    return false;
#endif
  case X86::BSWAPr32:        // Change bswap EAX, bswap EAX into nothing
    if (Next->getOpcode() == X86::BSWAPr32 &&
	MI->getOperand(0).getReg() == Next->getOperand(0).getReg()) {
      I = MBB.erase(MBB.erase(I));
      delete MI;
      delete Next;
      return true;
    }
    return false;
  default:
    return false;
  }
}
