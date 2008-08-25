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
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetMachine.h"
using namespace llvm;

/// SelectBinaryOp - Select and emit code for a binary operator instruction,
/// which has an opcode which directly corresponds to the given ISD opcode.
///
bool FastISel::SelectBinaryOp(Instruction *I, ISD::NodeType ISDOpcode,
                              DenseMap<const Value*, unsigned> &ValueMap) {
  MVT VT = MVT::getMVT(I->getType(), /*HandleUnknown=*/true);
  if (VT == MVT::Other || !VT.isSimple())
    // Unhandled type. Halt "fast" selection and bail.
    return false;

  unsigned Op0 = ValueMap[I->getOperand(0)];
  if (Op0 == 0)
    // Unhandled operand. Halt "fast" selection and bail.
    return false;

  // Check if the second operand is a constant and handle it appropriately.
  if (ConstantInt *CI = dyn_cast<ConstantInt>(I->getOperand(1))) {
    unsigned ResultReg = FastEmit_ri_(VT.getSimpleVT(), ISDOpcode, Op0,
                                      CI->getZExtValue(), VT.getSimpleVT());
    if (ResultReg == 0)
      // Target-specific code wasn't able to find a machine opcode for
      // the given ISD opcode and type. Halt "fast" selection and bail.
      return false;

    // We successfully emitted code for the given LLVM Instruction.
    ValueMap[I] = ResultReg;
    return true;
  }

  unsigned Op1 = ValueMap[I->getOperand(1)];
  if (Op1 == 0)
    // Unhandled operand. Halt "fast" selection and bail.
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
  unsigned N = ValueMap[I->getOperand(0)];
  if (N == 0)
    // Unhandled operand. Halt "fast" selection and bail.
    return false;

  const Type *Ty = I->getOperand(0)->getType();
  MVT::SimpleValueType VT = TLI.getPointerTy().getSimpleVT();
  for (GetElementPtrInst::op_iterator OI = I->op_begin()+1, E = I->op_end();
       OI != E; ++OI) {
    Value *Idx = *OI;
    if (const StructType *StTy = dyn_cast<StructType>(Ty)) {
      unsigned Field = cast<ConstantInt>(Idx)->getZExtValue();
      if (Field) {
        // N = N + Offset
        uint64_t Offs = TD.getStructLayout(StTy)->getElementOffset(Field);
        // FIXME: This can be optimized by combining the add with a
        // subsequent one.
        N = FastEmit_ri_(VT, ISD::ADD, N, Offs, VT);
        if (N == 0)
          // Unhandled operand. Halt "fast" selection and bail.
          return false;
      }
      Ty = StTy->getElementType(Field);
    } else {
      Ty = cast<SequentialType>(Ty)->getElementType();

      // If this is a constant subscript, handle it quickly.
      if (ConstantInt *CI = dyn_cast<ConstantInt>(Idx)) {
        if (CI->getZExtValue() == 0) continue;
        uint64_t Offs = 
          TD.getABITypeSize(Ty)*cast<ConstantInt>(CI)->getSExtValue();
        N = FastEmit_ri_(VT, ISD::ADD, N, Offs, VT);
        if (N == 0)
          // Unhandled operand. Halt "fast" selection and bail.
          return false;
        continue;
      }
      
      // N = N + Idx * ElementSize;
      uint64_t ElementSize = TD.getABITypeSize(Ty);
      unsigned IdxN = ValueMap[Idx];
      if (IdxN == 0)
        // Unhandled operand. Halt "fast" selection and bail.
        return false;

      // If the index is smaller or larger than intptr_t, truncate or extend
      // it.
      MVT IdxVT = MVT::getMVT(Idx->getType(), /*HandleUnknown=*/false);
      if (IdxVT.bitsLT(VT))
        IdxN = FastEmit_r(VT, ISD::SIGN_EXTEND, IdxN);
      else if (IdxVT.bitsGT(VT))
        IdxN = FastEmit_r(VT, ISD::TRUNCATE, IdxN);
      if (IdxN == 0)
        // Unhandled operand. Halt "fast" selection and bail.
        return false;

      if (ElementSize != 1)
        IdxN = FastEmit_ri_(VT, ISD::MUL, IdxN, ElementSize, VT);
      if (IdxN == 0)
        // Unhandled operand. Halt "fast" selection and bail.
        return false;
      N = FastEmit_rr(VT, ISD::ADD, N, IdxN);
      if (N == 0)
        // Unhandled operand. Halt "fast" selection and bail.
        return false;
    }
  }

  // We successfully emitted code for the given LLVM Instruction.
  ValueMap[I] = N;
  return true;
}

BasicBlock::iterator
FastISel::SelectInstructions(BasicBlock::iterator Begin,
                             BasicBlock::iterator End,
                             DenseMap<const Value*, unsigned> &ValueMap,
                             DenseMap<const BasicBlock*,
                                      MachineBasicBlock *> &MBBMap,
                             MachineBasicBlock *mbb) {
  MBB = mbb;
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

      if (BI->isUnconditional()) {
        MachineFunction::iterator NextMBB =
           next(MachineFunction::iterator(MBB));
        BasicBlock *LLVMSucc = BI->getSuccessor(0);
        MachineBasicBlock *MSucc = MBBMap[LLVMSucc];

        if (NextMBB != MF.end() && MSucc == NextMBB) {
          // The unconditional fall-through case, which needs no instructions.
        } else {
          // The unconditional branch case.
          TII.InsertBranch(*MBB, MSucc, NULL, SmallVector<MachineOperand, 0>());
        }
        MBB->addSuccessor(MSucc);
        break;
      }

      // Conditional branches are not handed yet.
      // Halt "fast" selection and bail.
      return I;
    }

    case Instruction::PHI:
      // PHI nodes are already emitted.
      break;
      
    case Instruction::BitCast:
      // BitCast consists of either an immediate to register move
      // or a register to register move.
      if (ConstantInt* CI = dyn_cast<ConstantInt>(I->getOperand(0))) {
        if (I->getType()->isInteger()) {
          MVT VT = MVT::getMVT(I->getType(), /*HandleUnknown=*/false);
          ValueMap[I] = FastEmit_i(VT.getSimpleVT(), ISD::Constant,
                                   CI->getZExtValue());
          break;
        } else
          // TODO: Support vector and fp constants.
          return I;
      } else if (!isa<Constant>(I->getOperand(0))) {
        // Bitcasts of non-constant values become reg-reg copies.
        MVT SrcVT = MVT::getMVT(I->getOperand(0)->getType());
        MVT DstVT = MVT::getMVT(I->getOperand(0)->getType());
        
        if (SrcVT == MVT::Other || !SrcVT.isSimple() ||
            DstVT == MVT::Other || !DstVT.isSimple() ||
            !TLI.isTypeLegal(SrcVT) || !TLI.isTypeLegal(DstVT))
          // Unhandled type. Halt "fast" selection and bail.
          return I;
        if (!TLI.isConvertLegal(SrcVT, DstVT))
          // Illegal conversion.  Halt "fast" selection and bail.
          return I:
        
        // Otherwise, insert a register-to-register copy.
        TargetRegisterClass* SrcClass = TLI.getRegClassFor(SrcVT);
        TargetRegisterClass* DstClass = TLI.getRegClassFor(DstVT);
        unsigned Op0 = ValueMap[I->getOperand(0)];
        unsigned ResultReg = createResultReg(DstClass);
        
        if (Op0 == 0)
          // Unhandled operand. Halt "fast" selection and bail.
          return false;
        
        TII.copyRegToReg(*MBB, MBB->end(), ResultReg, Op0, DstClass, SrcClass);
        ValueMap[I] = ResultReg;
        
        break;
      } else
        // Casting a non-integral constant?
        return I;

    default:
      // Unhandled instruction. Halt "fast" selection and bail.
      return I;
    }
  }

  return I;
}

FastISel::FastISel(MachineFunction &mf)
  : MF(mf),
    MRI(mf.getRegInfo()),
    TM(mf.getTarget()),
    TD(*TM.getTargetData()),
    TII(*TM.getInstrInfo()),
    TLI(*TM.getTargetLowering()) {
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

unsigned FastISel::FastEmit_i(MVT::SimpleValueType, ISD::NodeType,
                              uint64_t /*Imm*/) {
  return 0;
}

unsigned FastISel::FastEmit_ri(MVT::SimpleValueType, ISD::NodeType,
                               unsigned /*Op0*/, uint64_t /*Imm*/) {
  return 0;
}

unsigned FastISel::FastEmit_rri(MVT::SimpleValueType, ISD::NodeType,
                                unsigned /*Op0*/, unsigned /*Op1*/,
                                uint64_t /*Imm*/) {
  return 0;
}

/// FastEmit_ri_ - This method is a wrapper of FastEmit_ri. It first tries
/// to emit an instruction with an immediate operand using FastEmit_ri.
/// If that fails, it materializes the immediate into a register and try
/// FastEmit_rr instead.
unsigned FastISel::FastEmit_ri_(MVT::SimpleValueType VT, ISD::NodeType Opcode,
                                unsigned Op0, uint64_t Imm,
                                MVT::SimpleValueType ImmType) {
  unsigned ResultReg = 0;
  // First check if immediate type is legal. If not, we can't use the ri form.
  if (TLI.getOperationAction(ISD::Constant, ImmType) == TargetLowering::Legal)
    ResultReg = FastEmit_ri(VT, Opcode, Op0, Imm);
  if (ResultReg != 0)
    return ResultReg;
  unsigned MaterialReg = FastEmit_i(ImmType, ISD::Constant, Imm);
  if (MaterialReg == 0)
    return 0;
  return FastEmit_rr(VT, Opcode, Op0, MaterialReg);
}

unsigned FastISel::createResultReg(const TargetRegisterClass* RC) {
  return MRI.createVirtualRegister(RC);
}

unsigned FastISel::FastEmitInst_(unsigned MachineInstOpcode,
                                 const TargetRegisterClass* RC) {
  unsigned ResultReg = createResultReg(RC);
  const TargetInstrDesc &II = TII.get(MachineInstOpcode);

  BuildMI(MBB, II, ResultReg);
  return ResultReg;
}

unsigned FastISel::FastEmitInst_r(unsigned MachineInstOpcode,
                                  const TargetRegisterClass *RC,
                                  unsigned Op0) {
  unsigned ResultReg = createResultReg(RC);
  const TargetInstrDesc &II = TII.get(MachineInstOpcode);

  BuildMI(MBB, II, ResultReg).addReg(Op0);
  return ResultReg;
}

unsigned FastISel::FastEmitInst_rr(unsigned MachineInstOpcode,
                                   const TargetRegisterClass *RC,
                                   unsigned Op0, unsigned Op1) {
  unsigned ResultReg = createResultReg(RC);
  const TargetInstrDesc &II = TII.get(MachineInstOpcode);

  BuildMI(MBB, II, ResultReg).addReg(Op0).addReg(Op1);
  return ResultReg;
}

unsigned FastISel::FastEmitInst_ri(unsigned MachineInstOpcode,
                                   const TargetRegisterClass *RC,
                                   unsigned Op0, uint64_t Imm) {
  unsigned ResultReg = createResultReg(RC);
  const TargetInstrDesc &II = TII.get(MachineInstOpcode);

  BuildMI(MBB, II, ResultReg).addReg(Op0).addImm(Imm);
  return ResultReg;
}

unsigned FastISel::FastEmitInst_rri(unsigned MachineInstOpcode,
                                    const TargetRegisterClass *RC,
                                    unsigned Op0, unsigned Op1, uint64_t Imm) {
  unsigned ResultReg = createResultReg(RC);
  const TargetInstrDesc &II = TII.get(MachineInstOpcode);

  BuildMI(MBB, II, ResultReg).addReg(Op0).addReg(Op1).addImm(Imm);
  return ResultReg;
}

unsigned FastISel::FastEmitInst_i(unsigned MachineInstOpcode,
                                  const TargetRegisterClass *RC,
                                  uint64_t Imm) {
  unsigned ResultReg = createResultReg(RC);
  const TargetInstrDesc &II = TII.get(MachineInstOpcode);
  
  BuildMI(MBB, II, ResultReg).addImm(Imm);
  return ResultReg;
}