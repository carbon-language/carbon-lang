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

unsigned FastISel::getRegForValue(Value *V, DenseMap<const Value*, unsigned> &ValueMap) {
  unsigned &Reg = ValueMap[V];
  if (Reg != 0)
    return Reg;

  MVT::SimpleValueType VT = TLI.getValueType(V->getType()).getSimpleVT();
  if (ConstantInt *CI = dyn_cast<ConstantInt>(V)) {
    if (CI->getValue().getActiveBits() > 64)
      return 0;
    Reg = FastEmit_i(VT, VT, ISD::Constant, CI->getZExtValue());
  } else if (ConstantFP *CF = dyn_cast<ConstantFP>(V)) {
    Reg = FastEmit_f(VT, VT, ISD::ConstantFP, CF);

    if (!Reg) {
      const APFloat &Flt = CF->getValueAPF();
      MVT IntVT = TLI.getPointerTy();

      uint64_t x[2];
      uint32_t IntBitWidth = IntVT.getSizeInBits();
      if (Flt.convertToInteger(x, IntBitWidth, /*isSigned=*/true,
                               APFloat::rmTowardZero) != APFloat::opOK)
        return 0;
      APInt IntVal(IntBitWidth, 2, x);

      unsigned IntegerReg = FastEmit_i(IntVT.getSimpleVT(), IntVT.getSimpleVT(),
                                       ISD::Constant, IntVal.getZExtValue());
      if (IntegerReg == 0)
        return 0;
      Reg = FastEmit_r(IntVT.getSimpleVT(), VT, ISD::SINT_TO_FP, IntegerReg);
      if (Reg == 0)
        return 0;
    }
  }

  return Reg;
}

/// SelectBinaryOp - Select and emit code for a binary operator instruction,
/// which has an opcode which directly corresponds to the given ISD opcode.
///
bool FastISel::SelectBinaryOp(Instruction *I, ISD::NodeType ISDOpcode,
                              DenseMap<const Value*, unsigned> &ValueMap) {
  MVT VT = MVT::getMVT(I->getType(), /*HandleUnknown=*/true);
  if (VT == MVT::Other || !VT.isSimple())
    // Unhandled type. Halt "fast" selection and bail.
    return false;
  // We only handle legal types. For example, on x86-32 the instruction
  // selector contains all of the 64-bit instructions from x86-64,
  // under the assumption that i64 won't be used if the target doesn't
  // support it.
  if (!TLI.isTypeLegal(VT))
    return false;

  unsigned Op0 = getRegForValue(I->getOperand(0), ValueMap);
  if (Op0 == 0)
    // Unhandled operand. Halt "fast" selection and bail.
    return false;

  // Check if the second operand is a constant and handle it appropriately.
  if (ConstantInt *CI = dyn_cast<ConstantInt>(I->getOperand(1))) {
    unsigned ResultReg = FastEmit_ri(VT.getSimpleVT(), VT.getSimpleVT(),
                                     ISDOpcode, Op0, CI->getZExtValue());
    if (ResultReg != 0) {
      // We successfully emitted code for the given LLVM Instruction.
      ValueMap[I] = ResultReg;
      return true;
    }
  }

  // Check if the second operand is a constant float.
  if (ConstantFP *CF = dyn_cast<ConstantFP>(I->getOperand(1))) {
    unsigned ResultReg = FastEmit_rf(VT.getSimpleVT(), VT.getSimpleVT(),
                                     ISDOpcode, Op0, CF);
    if (ResultReg != 0) {
      // We successfully emitted code for the given LLVM Instruction.
      ValueMap[I] = ResultReg;
      return true;
    }
  }

  unsigned Op1 = getRegForValue(I->getOperand(1), ValueMap);
  if (Op1 == 0)
    // Unhandled operand. Halt "fast" selection and bail.
    return false;

  // Now we have both operands in registers. Emit the instruction.
  unsigned ResultReg = FastEmit_rr(VT.getSimpleVT(), VT.getSimpleVT(),
                                   ISDOpcode, Op0, Op1);
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
  unsigned N = getRegForValue(I->getOperand(0), ValueMap);
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
      unsigned IdxN = getRegForValue(Idx, ValueMap);
      if (IdxN == 0)
        // Unhandled operand. Halt "fast" selection and bail.
        return false;

      // If the index is smaller or larger than intptr_t, truncate or extend
      // it.
      MVT IdxVT = MVT::getMVT(Idx->getType(), /*HandleUnknown=*/false);
      if (IdxVT.bitsLT(VT))
        IdxN = FastEmit_r(IdxVT.getSimpleVT(), VT, ISD::SIGN_EXTEND, IdxN);
      else if (IdxVT.bitsGT(VT))
        IdxN = FastEmit_r(IdxVT.getSimpleVT(), VT, ISD::TRUNCATE, IdxN);
      if (IdxN == 0)
        // Unhandled operand. Halt "fast" selection and bail.
        return false;

      if (ElementSize != 1) {
        IdxN = FastEmit_ri_(VT, ISD::MUL, IdxN, ElementSize, VT);
        if (IdxN == 0)
          // Unhandled operand. Halt "fast" selection and bail.
          return false;
      }
      N = FastEmit_rr(VT, VT, ISD::ADD, N, IdxN);
      if (N == 0)
        // Unhandled operand. Halt "fast" selection and bail.
        return false;
    }
  }

  // We successfully emitted code for the given LLVM Instruction.
  ValueMap[I] = N;
  return true;
}

bool FastISel::SelectCast(Instruction *I, ISD::NodeType Opcode,
                          DenseMap<const Value*, unsigned> &ValueMap) {
  MVT SrcVT = TLI.getValueType(I->getOperand(0)->getType());
  MVT DstVT = TLI.getValueType(I->getType());
    
  if (SrcVT == MVT::Other || !SrcVT.isSimple() ||
      DstVT == MVT::Other || !DstVT.isSimple() ||
      !TLI.isTypeLegal(SrcVT) || !TLI.isTypeLegal(DstVT))
    // Unhandled type. Halt "fast" selection and bail.
    return false;
    
  unsigned InputReg = getRegForValue(I->getOperand(0), ValueMap);
  if (!InputReg)
    // Unhandled operand.  Halt "fast" selection and bail.
    return false;
    
  unsigned ResultReg = FastEmit_r(SrcVT.getSimpleVT(),
                                  DstVT.getSimpleVT(),
                                  Opcode,
                                  InputReg);
  if (!ResultReg)
    return false;
    
  ValueMap[I] = ResultReg;
  return true;
}

bool FastISel::SelectBitCast(Instruction *I,
                             DenseMap<const Value*, unsigned> &ValueMap) {
  // If the bitcast doesn't change the type, just use the operand value.
  if (I->getType() == I->getOperand(0)->getType()) {
    ValueMap[I] = getRegForValue(I->getOperand(0), ValueMap);
    return true;
  }

  // Bitcasts of other values become reg-reg copies or BIT_CONVERT operators.
  MVT SrcVT = TLI.getValueType(I->getOperand(0)->getType());
  MVT DstVT = TLI.getValueType(I->getType());
  
  if (SrcVT == MVT::Other || !SrcVT.isSimple() ||
      DstVT == MVT::Other || !DstVT.isSimple() ||
      !TLI.isTypeLegal(SrcVT) || !TLI.isTypeLegal(DstVT))
    // Unhandled type. Halt "fast" selection and bail.
    return false;
  
  unsigned Op0 = getRegForValue(I->getOperand(0), ValueMap);
  if (Op0 == 0)
    // Unhandled operand. Halt "fast" selection and bail.
    return false;
  
  // First, try to perform the bitcast by inserting a reg-reg copy.
  unsigned ResultReg = 0;
  if (SrcVT.getSimpleVT() == DstVT.getSimpleVT()) {
    TargetRegisterClass* SrcClass = TLI.getRegClassFor(SrcVT);
    TargetRegisterClass* DstClass = TLI.getRegClassFor(DstVT);
    ResultReg = createResultReg(DstClass);
    
    bool InsertedCopy = TII.copyRegToReg(*MBB, MBB->end(), ResultReg,
                                         Op0, DstClass, SrcClass);
    if (!InsertedCopy)
      ResultReg = 0;
  }
  
  // If the reg-reg copy failed, select a BIT_CONVERT opcode.
  if (!ResultReg)
    ResultReg = FastEmit_r(SrcVT.getSimpleVT(), DstVT.getSimpleVT(),
                           ISD::BIT_CONVERT, Op0);
  
  if (!ResultReg)
    return false;
  
  ValueMap[I] = ResultReg;
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
      if (!SelectBitCast(I, ValueMap)) return I; break;

    case Instruction::FPToSI:
      if (!SelectCast(I, ISD::FP_TO_SINT, ValueMap)) return I; 
      break;
    case Instruction::ZExt:
      if (!SelectCast(I, ISD::ZERO_EXTEND, ValueMap)) return I;
      break;
    case Instruction::SExt:
      if (!SelectCast(I, ISD::SIGN_EXTEND, ValueMap)) return I;
      break;
    case Instruction::SIToFP:
      if (!SelectCast(I, ISD::SINT_TO_FP, ValueMap)) return I;
      break;

    case Instruction::IntToPtr: // Deliberate fall-through.
    case Instruction::PtrToInt: {
      MVT SrcVT = TLI.getValueType(I->getOperand(0)->getType());
      MVT DstVT = TLI.getValueType(I->getType());
      if (SrcVT.getSimpleVT() == DstVT.getSimpleVT()) {
        if (ValueMap[I->getOperand(0)]) {
          ValueMap[I] = ValueMap[I->getOperand(0)];
          break;
        } else
          // Unhandled operand
          return I;
      } else if (DstVT.bitsGT(SrcVT)) {
        if (!SelectCast(I, ISD::ZERO_EXTEND, ValueMap)) return I;
        break;
      } else {
        // TODO: Handle SrcVT > DstVT, where truncation is needed.
        return I;
      }
    }
    
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

unsigned FastISel::FastEmit_(MVT::SimpleValueType, MVT::SimpleValueType, ISD::NodeType) {
  return 0;
}

unsigned FastISel::FastEmit_r(MVT::SimpleValueType, MVT::SimpleValueType,
                              ISD::NodeType, unsigned /*Op0*/) {
  return 0;
}

unsigned FastISel::FastEmit_rr(MVT::SimpleValueType, MVT::SimpleValueType, 
                               ISD::NodeType, unsigned /*Op0*/,
                               unsigned /*Op0*/) {
  return 0;
}

unsigned FastISel::FastEmit_i(MVT::SimpleValueType, MVT::SimpleValueType,
                              ISD::NodeType, uint64_t /*Imm*/) {
  return 0;
}

unsigned FastISel::FastEmit_f(MVT::SimpleValueType, MVT::SimpleValueType,
                              ISD::NodeType, ConstantFP * /*FPImm*/) {
  return 0;
}

unsigned FastISel::FastEmit_ri(MVT::SimpleValueType, MVT::SimpleValueType,
                               ISD::NodeType, unsigned /*Op0*/,
                               uint64_t /*Imm*/) {
  return 0;
}

unsigned FastISel::FastEmit_rf(MVT::SimpleValueType, MVT::SimpleValueType,
                               ISD::NodeType, unsigned /*Op0*/,
                               ConstantFP * /*FPImm*/) {
  return 0;
}

unsigned FastISel::FastEmit_rri(MVT::SimpleValueType, MVT::SimpleValueType,
                                ISD::NodeType,
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
  // First check if immediate type is legal. If not, we can't use the ri form.
  unsigned ResultReg = FastEmit_ri(VT, VT, Opcode, Op0, Imm);
  if (ResultReg != 0)
    return ResultReg;
  unsigned MaterialReg = FastEmit_i(ImmType, ImmType, ISD::Constant, Imm);
  if (MaterialReg == 0)
    return 0;
  return FastEmit_rr(VT, VT, Opcode, Op0, MaterialReg);
}

/// FastEmit_rf_ - This method is a wrapper of FastEmit_ri. It first tries
/// to emit an instruction with a floating-point immediate operand using
/// FastEmit_rf. If that fails, it materializes the immediate into a register
/// and try FastEmit_rr instead.
unsigned FastISel::FastEmit_rf_(MVT::SimpleValueType VT, ISD::NodeType Opcode,
                                unsigned Op0, ConstantFP *FPImm,
                                MVT::SimpleValueType ImmType) {
  // First check if immediate type is legal. If not, we can't use the rf form.
  unsigned ResultReg = FastEmit_rf(VT, VT, Opcode, Op0, FPImm);
  if (ResultReg != 0)
    return ResultReg;

  // Materialize the constant in a register.
  unsigned MaterialReg = FastEmit_f(ImmType, ImmType, ISD::ConstantFP, FPImm);
  if (MaterialReg == 0) {
    // If the target doesn't have a way to directly enter a floating-point
    // value into a register, use an alternate approach.
    // TODO: The current approach only supports floating-point constants
    // that can be constructed by conversion from integer values. This should
    // be replaced by code that creates a load from a constant-pool entry,
    // which will require some target-specific work.
    const APFloat &Flt = FPImm->getValueAPF();
    MVT IntVT = TLI.getPointerTy();

    uint64_t x[2];
    uint32_t IntBitWidth = IntVT.getSizeInBits();
    if (Flt.convertToInteger(x, IntBitWidth, /*isSigned=*/true,
                             APFloat::rmTowardZero) != APFloat::opOK)
      return 0;
    APInt IntVal(IntBitWidth, 2, x);

    unsigned IntegerReg = FastEmit_i(IntVT.getSimpleVT(), IntVT.getSimpleVT(),
                                     ISD::Constant, IntVal.getZExtValue());
    if (IntegerReg == 0)
      return 0;
    MaterialReg = FastEmit_r(IntVT.getSimpleVT(), VT,
                             ISD::SINT_TO_FP, IntegerReg);
    if (MaterialReg == 0)
      return 0;
  }
  return FastEmit_rr(VT, VT, Opcode, Op0, MaterialReg);
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

unsigned FastISel::FastEmitInst_rf(unsigned MachineInstOpcode,
                                   const TargetRegisterClass *RC,
                                   unsigned Op0, ConstantFP *FPImm) {
  unsigned ResultReg = createResultReg(RC);
  const TargetInstrDesc &II = TII.get(MachineInstOpcode);

  BuildMI(MBB, II, ResultReg).addReg(Op0).addFPImm(FPImm);
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
