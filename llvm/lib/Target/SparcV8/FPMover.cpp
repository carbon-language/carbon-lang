//===-- FPMover.cpp - SparcV8 double-precision floating point move fixer --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Expand FpMOVD/FpABSD/FpNEGD instructions into their single-precision pieces.
//
//===----------------------------------------------------------------------===//

#include "SparcV8.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"
using namespace llvm;

namespace {
  Statistic<> NumFpDs("fpmover", "Number of instructions translated");
  Statistic<> NoopFpDs("fpmover", "Number of noop instructions removed");

  struct FPMover : public MachineFunctionPass {
    /// Target machine description which we query for reg. names, data
    /// layout, etc.
    ///
    TargetMachine &TM;

    FPMover(TargetMachine &tm) : TM(tm) { }

    virtual const char *getPassName() const {
      return "SparcV8 Double-FP Move Fixer";
    }

    bool runOnMachineBasicBlock(MachineBasicBlock &MBB);
    bool runOnMachineFunction(MachineFunction &F) {
      bool Changed = false;
      for (MachineFunction::iterator FI = F.begin(), FE = F.end();
           FI != FE; ++FI)
        Changed |= runOnMachineBasicBlock(*FI);
      return Changed;
    }

  };
} // end of anonymous namespace

/// createSparcV8FPMoverPass - Returns a pass that turns FpMOVD
/// instructions into FMOVS instructions
///
FunctionPass *llvm::createSparcV8FPMoverPass(TargetMachine &tm) {
  return new FPMover(tm);
}

/// getDoubleRegPair - Given a DFP register, return the even and odd FP
/// registers that correspond to it.
static void getDoubleRegPair(unsigned DoubleReg, unsigned &EvenReg,
                             unsigned &OddReg) {
  static const unsigned EvenHalvesOfPairs[] = {
    V8::F0, V8::F2, V8::F4, V8::F6, V8::F8, V8::F10, V8::F12, V8::F14,
    V8::F16, V8::F18, V8::F20, V8::F22, V8::F24, V8::F26, V8::F28, V8::F30
  };
  static const unsigned OddHalvesOfPairs[] = {
    V8::F1, V8::F3, V8::F5, V8::F7, V8::F9, V8::F11, V8::F13, V8::F15,
    V8::F17, V8::F19, V8::F21, V8::F23, V8::F25, V8::F27, V8::F29, V8::F31
  };
  static const unsigned DoubleRegsInOrder[] = {
    V8::D0, V8::D1, V8::D2, V8::D3, V8::D4, V8::D5, V8::D6, V8::D7, V8::D8,
    V8::D9, V8::D10, V8::D11, V8::D12, V8::D13, V8::D14, V8::D15
  };
  for (unsigned i = 0; i < sizeof(DoubleRegsInOrder)/sizeof(unsigned); ++i)
    if (DoubleRegsInOrder[i] == DoubleReg) {
      EvenReg = EvenHalvesOfPairs[i];
      OddReg = OddHalvesOfPairs[i];
      return;
    }
  assert(0 && "Can't find reg");
}

/// runOnMachineBasicBlock - Fixup FpMOVD instructions in this MBB.
///
bool FPMover::runOnMachineBasicBlock(MachineBasicBlock &MBB) {
  bool Changed = false;
  for (MachineBasicBlock::iterator I = MBB.begin(); I != MBB.end(); ) {
    MachineInstr *MI = I++;
    if (MI->getOpcode() == V8::FpMOVD) {
      unsigned DestDReg = MI->getOperand(0).getReg();
      unsigned SrcDReg  = MI->getOperand(1).getReg();
      if (DestDReg != SrcDReg || MI->getOpcode() != V8::FpMOVD) {
        unsigned EvenSrcReg = 0, OddSrcReg = 0, EvenDestReg = 0, OddDestReg = 0;
        getDoubleRegPair(DestDReg, EvenDestReg, OddDestReg);
        getDoubleRegPair(SrcDReg, EvenSrcReg, OddSrcReg);

        I->setOpcode(V8::FMOVS);
        I->SetMachineOperandReg(0, EvenDestReg);
        I->SetMachineOperandReg(1, EvenSrcReg);
        DEBUG(std::cerr << "FPMover: the modified instr is: " << *I);
        // Insert copy for the other half of the double:
        MI = BuildMI(MBB, I, V8::FMOVS, 1, OddDestReg).addReg(OddSrcReg);
        DEBUG(std::cerr << "FPMover: the inserted instr is: " << *MI);
        ++NumFpDs;
      } else {
        MBB.erase(MI);
        ++NoopFpDs;
      }
      Changed = true;
    }
  }
  return Changed;
}
