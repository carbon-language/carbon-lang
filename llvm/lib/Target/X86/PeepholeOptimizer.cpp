//===-- PeepholeOptimizer.cpp - X86 Peephole Optimizer --------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file contains a peephole optimizer for the X86.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
using namespace llvm;

namespace {
  struct PH : public MachineFunctionPass {
    virtual bool runOnMachineFunction(MachineFunction &MF);

    bool PeepholeOptimize(MachineBasicBlock &MBB,
			  MachineBasicBlock::iterator &I);

    virtual const char *getPassName() const { return "X86 Peephole Optimizer"; }
  };
}

FunctionPass *llvm::createX86PeepholeOptimizerPass() { return new PH(); }

bool PH::runOnMachineFunction(MachineFunction &MF) {
  bool Changed = false;

  for (MachineFunction::iterator BI = MF.begin(), E = MF.end(); BI != E; ++BI)
    for (MachineBasicBlock::iterator I = BI->begin(); I != BI->end(); )
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

    // A large number of X86 instructions have forms which take an 8-bit
    // immediate despite the fact that the operands are 16 or 32 bits.  Because
    // this can save three bytes of code size (and icache space), we want to
    // shrink them if possible.
  case X86::ADDri16:  case X86::ADDri32:
  case X86::SUBri16:  case X86::SUBri32:
  case X86::IMULri16: case X86::IMULri32:
  case X86::ANDri16:  case X86::ANDri32:
  case X86::ORri16:   case X86::ORri32:
  case X86::XORri16:  case X86::XORri32:
    assert(MI->getNumOperands() == 3 && "These should all have 3 operands!");
    if (MI->getOperand(2).isImmediate()) {
      int Val = MI->getOperand(2).getImmedValue();
      // If the value is the same when signed extended from 8 bits...
      if (Val == (signed int)(signed char)Val) {
        unsigned Opcode;
        switch (MI->getOpcode()) {
        default: assert(0 && "Unknown opcode value!");
        case X86::ADDri16:  Opcode = X86::ADDri16b; break;
        case X86::ADDri32:  Opcode = X86::ADDri32b; break;
        case X86::SUBri16:  Opcode = X86::SUBri16b; break;
        case X86::SUBri32:  Opcode = X86::SUBri32b; break;
        case X86::IMULri16: Opcode = X86::IMULri16b; break;
        case X86::IMULri32: Opcode = X86::IMULri32b; break;
        case X86::ANDri16:  Opcode = X86::ANDri16b; break;
        case X86::ANDri32:  Opcode = X86::ANDri32b; break;
        case X86::ORri16:   Opcode = X86::ORri16b; break;
        case X86::ORri32:   Opcode = X86::ORri32b; break;
        case X86::XORri16:  Opcode = X86::XORri16b; break;
        case X86::XORri32:  Opcode = X86::XORri32b; break;
        }
        unsigned R0 = MI->getOperand(0).getReg();
        unsigned R1 = MI->getOperand(1).getReg();
        *I = BuildMI(Opcode, 2, R0).addReg(R1).addZImm((char)Val);
        delete MI;
        return true;
      }
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

