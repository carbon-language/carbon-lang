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

#include "llvm/Instructions.h"
#include "llvm/CodeGen/FastISel.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
using namespace llvm;

/// SelectBinaryOp - Select and emit code for a binary operator instruction,
/// which has an opcode which directly corresponds to the given ISD opcode.
///
bool FastISel::SelectBinaryOp(Instruction *I, ISD::NodeType ISDOpcode,
                              DenseMap<const Value*, unsigned> &ValueMap) {
  unsigned Op0 = ValueMap[I->getOperand(0)];
  unsigned Op1 = ValueMap[I->getOperand(1)];
  if (Op0 == 0 || Op1 == 0)
    // Unhandled operand. Halt "fast" selection and bail.
    return false;

  MVT VT = MVT::getMVT(I->getType(), /*HandleUnknown=*/true);
  if (VT == MVT::Other || !VT.isSimple())
    // Unhandled type. Halt "fast" selection and bail.
    return false;

  unsigned ResultReg = FastEmit_rr(VT.getSimpleVT(), ISDOpcode, Op0, Op1);
  if (ResultReg == 0)
    // Target-specific code wasn't able to find a machine opcode for
    // the given ISD opcode and type. Halt "fast" selection and bail.
    return false;

  // We successfully emitted code for the given LLVM Instruction.
  ValueMap[I] = ResultReg;
  return true;
}

bool FastISel::SelectGetElementPtr(Instruction *I,
                                   DenseMap<const Value*, unsigned> &ValueMap) {
  // TODO: implement me
  return false;
}

BasicBlock::iterator
FastISel::SelectInstructions(BasicBlock::iterator Begin,
                             BasicBlock::iterator End,
                             DenseMap<const Value*, unsigned> &ValueMap) {
  BasicBlock::iterator I = Begin;

  for (; I != End; ++I) {
    switch (I->getOpcode()) {
    case Instruction::Add: {
      ISD::NodeType Opc = I->getType()->isFPOrFPVector() ? ISD::FADD : ISD::ADD;
      if (!SelectBinaryOp(I, Opc, ValueMap))  return I; break;
    }
    case Instruction::Sub: {
      ISD::NodeType Opc = I->getType()->isFPOrFPVector() ? ISD::FSUB : ISD::SUB;
      if (!SelectBinaryOp(I, Opc, ValueMap))  return I; break;
    }
    case Instruction::Mul: {
      ISD::NodeType Opc = I->getType()->isFPOrFPVector() ? ISD::FMUL : ISD::MUL;
      if (!SelectBinaryOp(I, Opc, ValueMap))  return I; break;
    }
    case Instruction::SDiv:
      if (!SelectBinaryOp(I, ISD::SDIV, ValueMap)) return I; break;
    case Instruction::UDiv:
      if (!SelectBinaryOp(I, ISD::UDIV, ValueMap)) return I; break;
    case Instruction::FDiv:
      if (!SelectBinaryOp(I, ISD::FDIV, ValueMap)) return I; break;
    case Instruction::SRem:
      if (!SelectBinaryOp(I, ISD::SREM, ValueMap)) return I; break;
    case Instruction::URem:
      if (!SelectBinaryOp(I, ISD::UREM, ValueMap)) return I; break;
    case Instruction::FRem:
      if (!SelectBinaryOp(I, ISD::FREM, ValueMap)) return I; break;
    case Instruction::Shl:
      if (!SelectBinaryOp(I, ISD::SHL, ValueMap)) return I; break;
    case Instruction::LShr:
      if (!SelectBinaryOp(I, ISD::SRL, ValueMap)) return I; break;
    case Instruction::AShr:
      if (!SelectBinaryOp(I, ISD::SRA, ValueMap)) return I; break;
    case Instruction::And:
      if (!SelectBinaryOp(I, ISD::AND, ValueMap)) return I; break;
    case Instruction::Or:
      if (!SelectBinaryOp(I, ISD::OR, ValueMap)) return I; break;
    case Instruction::Xor:
      if (!SelectBinaryOp(I, ISD::XOR, ValueMap)) return I; break;

    case Instruction::GetElementPtr:
      if (!SelectGetElementPtr(I, ValueMap)) return I;
      break;

    case Instruction::Br: {
      BranchInst *BI = cast<BranchInst>(I);

      // For now, check for and handle just the most trivial case: an
      // unconditional fall-through branch.
      if (BI->isUnconditional()) {
         MachineFunction::iterator NextMBB =
           next(MachineFunction::iterator(MBB));
         if (NextMBB != MF->end() &&
             NextMBB->getBasicBlock() == BI->getSuccessor(0)) {
          MBB->addSuccessor(NextMBB);
          break;
        }
      }

      // Something more complicated. Halt "fast" selection and bail.
      return I;
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
  unsigned ResultReg = MRI.createVirtualRegister(RC);
  const TargetInstrDesc &II = TII->get(MachineInstOpcode);

  MachineInstr *MI = BuildMI(*MF, II, ResultReg);

  MBB->push_back(MI);
  return ResultReg;
}

unsigned FastISel::FastEmitInst_r(unsigned MachineInstOpcode,
                                  const TargetRegisterClass *RC,
                                  unsigned Op0) {
  MachineRegisterInfo &MRI = MF->getRegInfo();
  unsigned ResultReg = MRI.createVirtualRegister(RC);
  const TargetInstrDesc &II = TII->get(MachineInstOpcode);

  MachineInstr *MI = BuildMI(*MF, II, ResultReg);
  MI->addOperand(MachineOperand::CreateReg(Op0, false));

  MBB->push_back(MI);
  return ResultReg;
}

unsigned FastISel::FastEmitInst_rr(unsigned MachineInstOpcode,
                                   const TargetRegisterClass *RC,
                                   unsigned Op0, unsigned Op1) {
  MachineRegisterInfo &MRI = MF->getRegInfo();
  unsigned ResultReg = MRI.createVirtualRegister(RC);
  const TargetInstrDesc &II = TII->get(MachineInstOpcode);

  MachineInstr *MI = BuildMI(*MF, II, ResultReg);
  MI->addOperand(MachineOperand::CreateReg(Op0, false));
  MI->addOperand(MachineOperand::CreateReg(Op1, false));

  MBB->push_back(MI);
  return ResultReg;
}
