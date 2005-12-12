//===-- X86PeepholeOpt.cpp - X86 Peephole Optimizer -----------------------===//
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
#include "llvm/Target/MRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
using namespace llvm;

namespace {
  Statistic<> NumPHOpts("x86-peephole",
                        "Number of peephole optimization performed");
  Statistic<> NumPHMoves("x86-peephole", "Number of peephole moves folded");
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
      if (PeepholeOptimize(*BI, I)) {
        Changed = true;
        ++NumPHOpts;
      } else
        ++I;

  return Changed;
}


bool PH::PeepholeOptimize(MachineBasicBlock &MBB,
                          MachineBasicBlock::iterator &I) {
  assert(I != MBB.end());
  MachineBasicBlock::iterator NextI = next(I);

  MachineInstr *MI = I;
  MachineInstr *Next = (NextI != MBB.end()) ? &*NextI : (MachineInstr*)0;
  unsigned Size = 0;
  switch (MI->getOpcode()) {
  case X86::MOV8rr:
  case X86::MOV16rr:
  case X86::MOV32rr:   // Destroy X = X copies...
    if (MI->getOperand(0).getReg() == MI->getOperand(1).getReg()) {
      I = MBB.erase(I);
      return true;
    }
    return false;

    // A large number of X86 instructions have forms which take an 8-bit
    // immediate despite the fact that the operands are 16 or 32 bits.  Because
    // this can save three bytes of code size (and icache space), we want to
    // shrink them if possible.
  case X86::IMUL16rri: case X86::IMUL32rri:
    assert(MI->getNumOperands() == 3 && "These should all have 3 operands!");
    if (MI->getOperand(2).isImmediate()) {
      int Val = MI->getOperand(2).getImmedValue();
      // If the value is the same when signed extended from 8 bits...
      if (Val == (signed int)(signed char)Val) {
        unsigned Opcode;
        switch (MI->getOpcode()) {
        default: assert(0 && "Unknown opcode value!");
        case X86::IMUL16rri: Opcode = X86::IMUL16rri8; break;
        case X86::IMUL32rri: Opcode = X86::IMUL32rri8; break;
        }
        unsigned R0 = MI->getOperand(0).getReg();
        unsigned R1 = MI->getOperand(1).getReg();
        I = MBB.insert(MBB.erase(I),
                       BuildMI(Opcode, 2, R0).addReg(R1).addZImm((char)Val));
        return true;
      }
    }
    return false;

  case X86::ADD16ri:  case X86::ADD32ri:  case X86::ADC32ri:
  case X86::SUB16ri:  case X86::SUB32ri:
  case X86::SBB16ri:  case X86::SBB32ri:
  case X86::AND16ri:  case X86::AND32ri:
  case X86::OR16ri:   case X86::OR32ri:
  case X86::XOR16ri:  case X86::XOR32ri:
    assert(MI->getNumOperands() == 2 && "These should all have 2 operands!");
    if (MI->getOperand(1).isImmediate()) {
      int Val = MI->getOperand(1).getImmedValue();
      // If the value is the same when signed extended from 8 bits...
      if (Val == (signed int)(signed char)Val) {
        unsigned Opcode;
        switch (MI->getOpcode()) {
        default: assert(0 && "Unknown opcode value!");
        case X86::ADD16ri:  Opcode = X86::ADD16ri8; break;
        case X86::ADD32ri:  Opcode = X86::ADD32ri8; break;
        case X86::ADC32ri:  Opcode = X86::ADC32ri8; break;
        case X86::SUB16ri:  Opcode = X86::SUB16ri8; break;
        case X86::SUB32ri:  Opcode = X86::SUB32ri8; break;
        case X86::SBB16ri:  Opcode = X86::SBB16ri8; break;
        case X86::SBB32ri:  Opcode = X86::SBB32ri8; break;
        case X86::AND16ri:  Opcode = X86::AND16ri8; break;
        case X86::AND32ri:  Opcode = X86::AND32ri8; break;
        case X86::OR16ri:   Opcode = X86::OR16ri8; break;
        case X86::OR32ri:   Opcode = X86::OR32ri8; break;
        case X86::XOR16ri:  Opcode = X86::XOR16ri8; break;
        case X86::XOR32ri:  Opcode = X86::XOR32ri8; break;
        }
        unsigned R0 = MI->getOperand(0).getReg();
        I = MBB.insert(MBB.erase(I),
                    BuildMI(Opcode, 1, R0, MachineOperand::UseAndDef)
                      .addZImm((char)Val));
        return true;
      }
    }
    return false;

  case X86::ADD16mi:  case X86::ADD32mi:  case X86::ADC32mi:
  case X86::SUB16mi:  case X86::SUB32mi:
  case X86::SBB16mi:  case X86::SBB32mi:
  case X86::AND16mi:  case X86::AND32mi:
  case X86::OR16mi:  case X86::OR32mi:
  case X86::XOR16mi:  case X86::XOR32mi:
    assert(MI->getNumOperands() == 5 && "These should all have 5 operands!");
    if (MI->getOperand(4).isImmediate()) {
      int Val = MI->getOperand(4).getImmedValue();
      // If the value is the same when signed extended from 8 bits...
      if (Val == (signed int)(signed char)Val) {
        unsigned Opcode;
        switch (MI->getOpcode()) {
        default: assert(0 && "Unknown opcode value!");
        case X86::ADD16mi:  Opcode = X86::ADD16mi8; break;
        case X86::ADD32mi:  Opcode = X86::ADD32mi8; break;
        case X86::ADC32mi:  Opcode = X86::ADC32mi8; break;
        case X86::SUB16mi:  Opcode = X86::SUB16mi8; break;
        case X86::SUB32mi:  Opcode = X86::SUB32mi8; break;
        case X86::SBB16mi:  Opcode = X86::SBB16mi8; break;
        case X86::SBB32mi:  Opcode = X86::SBB32mi8; break;
        case X86::AND16mi:  Opcode = X86::AND16mi8; break;
        case X86::AND32mi:  Opcode = X86::AND32mi8; break;
        case X86::OR16mi:   Opcode = X86::OR16mi8; break;
        case X86::OR32mi:   Opcode = X86::OR32mi8; break;
        case X86::XOR16mi:  Opcode = X86::XOR16mi8; break;
        case X86::XOR32mi:  Opcode = X86::XOR32mi8; break;
        }
        unsigned R0 = MI->getOperand(0).getReg();
        unsigned Scale = MI->getOperand(1).getImmedValue();
        unsigned R1 = MI->getOperand(2).getReg();
        if (MI->getOperand(3).isImmediate()) {
          unsigned Offset = MI->getOperand(3).getImmedValue();
          I = MBB.insert(MBB.erase(I),
                         BuildMI(Opcode, 5).addReg(R0).addZImm(Scale).
                         addReg(R1).addSImm(Offset).addZImm((char)Val));
        } else if (MI->getOperand(3).isGlobalAddress()) {
          GlobalValue *GA = MI->getOperand(3).getGlobal();
          int Offset = MI->getOperand(3).getOffset();
          I = MBB.insert(MBB.erase(I),
                         BuildMI(Opcode, 5).addReg(R0).addZImm(Scale).
                         addReg(R1).addGlobalAddress(GA, false, Offset).
                         addZImm((char)Val));
        }
        return true;
      }
    }
    return false;
  default:
    return false;
  }
}
