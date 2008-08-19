///===-- FastISel.cpp - Implementation of the FastISel class --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of the FastISel class.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/FastISel.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
using namespace llvm;

BasicBlock::iterator
FastISel::SelectInstructions(BasicBlock::iterator Begin, BasicBlock::iterator End,
                             DenseMap<const Value*, unsigned> &ValueMap) {
  BasicBlock::iterator I = Begin;

  for (; I != End; ++I) {
    switch (I->getOpcode()) {
    case Instruction::Add: {
      unsigned Op0 = ValueMap[I->getOperand(0)];
      unsigned Op1 = ValueMap[I->getOperand(1)];
      MVT VT = MVT::getMVT(I->getType(), /*HandleUnknown=*/true);
      if (VT == MVT::Other || !VT.isSimple()) {
        // Unhandled type. Halt "fast" selection and bail.
        return I;
      }
      unsigned ResultReg = FastEmit_rr(VT.getSimpleVT(), ISD::ADD, Op0, Op1);
      if (ResultReg == 0) {
        // Target-specific code wasn't able to find a machine opcode for
        // the given ISD opcode and type. Halt "fast" selection and bail.
        return I;
      }
      ValueMap[I] = ResultReg;
      break;
    }
    default:
      // Unhandled instruction. Halt "fast" selection and bail.
      return I;
    }
  }

  return I;
}

FastISel::~FastISel() {}

unsigned FastISel::FastEmit_(MVT::SimpleValueType, ISD::NodeType) {
  return 0;
}

unsigned FastISel::FastEmit_r(MVT::SimpleValueType, ISD::NodeType,
                              unsigned /*Op0*/) {
  return 0;
}

unsigned FastISel::FastEmit_rr(MVT::SimpleValueType, ISD::NodeType,
                               unsigned /*Op0*/, unsigned /*Op0*/) {
  return 0;
}

unsigned FastISel::FastEmitInst_(unsigned MachineInstOpcode,
                                    const TargetRegisterClass* RC) {
  MachineRegisterInfo &MRI = MF->getRegInfo();
  const TargetInstrDesc &II = TII->get(MachineInstOpcode);
  unsigned ResultReg = MRI.createVirtualRegister(RC);

  MachineInstr *MI = BuildMI(*MF, II, ResultReg);

  MBB->push_back(MI);
  return ResultReg;
}

unsigned FastISel::FastEmitInst_r(unsigned MachineInstOpcode,
                                  const TargetRegisterClass *RC,
                                  unsigned Op0) {
  MachineRegisterInfo &MRI = MF->getRegInfo();
  const TargetInstrDesc &II = TII->get(MachineInstOpcode);
  unsigned ResultReg = MRI.createVirtualRegister(RC);

  MachineInstr *MI = BuildMI(*MF, II, ResultReg);
  MI->addOperand(MachineOperand::CreateReg(Op0, false));

  MBB->push_back(MI);
  return ResultReg;
}

unsigned FastISel::FastEmitInst_rr(unsigned MachineInstOpcode,
                                   const TargetRegisterClass *RC,
                                   unsigned Op0, unsigned Op1) {
  MachineRegisterInfo &MRI = MF->getRegInfo();
  const TargetInstrDesc &II = TII->get(MachineInstOpcode);
  unsigned ResultReg = MRI.createVirtualRegister(RC);

  MachineInstr *MI = BuildMI(*MF, II, ResultReg);
  MI->addOperand(MachineOperand::CreateReg(Op0, false));
  MI->addOperand(MachineOperand::CreateReg(Op1, false));

  MBB->push_back(MI);
  return ResultReg;
}
