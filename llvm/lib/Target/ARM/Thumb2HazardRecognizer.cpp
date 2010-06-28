//===-- Thumb2HazardRecognizer.cpp - Thumb2 postra hazard recognizer ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ARM.h"
#include "Thumb2HazardRecognizer.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/ScheduleDAG.h"
using namespace llvm;

ScheduleHazardRecognizer::HazardType
Thumb2HazardRecognizer::getHazardType(SUnit *SU) {
  if (ITBlockSize) {
    MachineInstr *MI = SU->getInstr();
    if (!MI->isDebugValue() && MI != ITBlockMIs[ITBlockSize-1])
      return Hazard;
  }

  return PostRAHazardRecognizer::getHazardType(SU);
}

void Thumb2HazardRecognizer::Reset() {
  ITBlockSize = 0;
  PostRAHazardRecognizer::Reset();
}

void Thumb2HazardRecognizer::EmitInstruction(SUnit *SU) {
  MachineInstr *MI = SU->getInstr();
  unsigned Opcode = MI->getOpcode();
  if (ITBlockSize) {
    --ITBlockSize;
  } else if (Opcode == ARM::t2IT) {
    unsigned Mask = MI->getOperand(1).getImm();
    unsigned NumTZ = CountTrailingZeros_32(Mask);
    assert(NumTZ <= 3 && "Invalid IT mask!");
    ITBlockSize = 4 - NumTZ;
    MachineBasicBlock::iterator I = MI;
    for (unsigned i = 0; i < ITBlockSize; ++i) {
      // Advance to the next instruction, skipping any dbg_value instructions.
      do {
        ++I;
      } while (I->isDebugValue());
      ITBlockMIs[ITBlockSize-1-i] = &*I;
    }
  }

  PostRAHazardRecognizer::EmitInstruction(SU);
}
