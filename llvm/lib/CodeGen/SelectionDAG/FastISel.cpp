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

unsigned FastISel::getRegForValue(Value *V) {
  // Look up the value to see if we already have a register for it. We
  // cache values defined by Instructions across blocks, and other values
  // only locally. This is because Instructions already have the SSA
  // def-dominatess-use requirement enforced.
  if (ValueMap.count(V))
    return ValueMap[V];
  unsigned Reg = LocalValueMap[V];
  if (Reg != 0)
    return Reg;

  MVT::SimpleValueType VT = TLI.getValueType(V->getType()).getSimpleVT();
  if (ConstantInt *CI = dyn_cast<ConstantInt>(V)) {
    if (CI->getValue().getActiveBits() > 64)
      return 0;
    // Don't cache constant materializations.  To do so would require
    // tracking what uses they dominate.
    Reg = FastEmit_i(VT, VT, ISD::Constant, CI->getZExtValue());
  } else if (isa<ConstantPointerNull>(V)) {
    Reg = FastEmit_i(VT, VT, ISD::Constant, 0);
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
  } else if (isa<UndefValue>(V)) {
    Reg = createResultReg(TLI.getRegClassFor(VT));
    BuildMI(MBB, TII.get(TargetInstrInfo::IMPLICIT_DEF), Reg);
  } else {
    return 0;
  }
  
  LocalValueMap[V] = Reg;
  return Reg;
}

/// UpdateValueMap - Update the value map to include the new mapping for this
/// instruction, or insert an extra copy to get the result in a previous
/// determined register.
/// NOTE: This is only necessary because we might select a block that uses
/// a value before we select the block that defines the value.  It might be
/// possible to fix this by selecting blocks in reverse postorder.
void FastISel::UpdateValueMap(Value* I, unsigned Reg) {
  if (!ValueMap.count(I))
    ValueMap[I] = Reg;
  else
     TII.copyRegToReg(*MBB, MBB->end(), ValueMap[I],
                      Reg, MRI.getRegClass(Reg), MRI.getRegClass(Reg));
}

/// SelectBinaryOp - Select and emit code for a binary operator instruction,
/// which has an opcode which directly corresponds to the given ISD opcode.
///
bool FastISel::SelectBinaryOp(Instruction *I, ISD::NodeType ISDOpcode) {
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

  unsigned Op0 = getRegForValue(I->getOperand(0));
  if (Op0 == 0)
    // Unhandled operand. Halt "fast" selection and bail.
    return false;

  // Check if the second operand is a constant and handle it appropriately.
  if (ConstantInt *CI = dyn_cast<ConstantInt>(I->getOperand(1))) {
    unsigned ResultReg = FastEmit_ri(VT.getSimpleVT(), VT.getSimpleVT(),
                                     ISDOpcode, Op0, CI->getZExtValue());
    if (ResultReg != 0) {
      // We successfully emitted code for the given LLVM Instruction.
      UpdateValueMap(I, ResultReg);
      return true;
    }
  }

  // Check if the second operand is a constant float.
  if (ConstantFP *CF = dyn_cast<ConstantFP>(I->getOperand(1))) {
    unsigned ResultReg = FastEmit_rf(VT.getSimpleVT(), VT.getSimpleVT(),
                                     ISDOpcode, Op0, CF);
    if (ResultReg != 0) {
      // We successfully emitted code for the given LLVM Instruction.
      UpdateValueMap(I, ResultReg);
      return true;
    }
  }

  unsigned Op1 = getRegForValue(I->getOperand(1));
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
  UpdateValueMap(I, ResultReg);
  return true;
}

bool FastISel::SelectGetElementPtr(Instruction *I) {
  unsigned N = getRegForValue(I->getOperand(0));
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
      unsigned IdxN = getRegForValue(Idx);
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
  UpdateValueMap(I, N);
  return true;
}

bool FastISel::SelectCast(Instruction *I, ISD::NodeType Opcode) {
  MVT SrcVT = TLI.getValueType(I->getOperand(0)->getType());
  MVT DstVT = TLI.getValueType(I->getType());
    
  if (SrcVT == MVT::Other || !SrcVT.isSimple() ||
      DstVT == MVT::Other || !DstVT.isSimple() ||
      !TLI.isTypeLegal(SrcVT) || !TLI.isTypeLegal(DstVT))
    // Unhandled type. Halt "fast" selection and bail.
    return false;
    
  unsigned InputReg = getRegForValue(I->getOperand(0));
  if (!InputReg)
    // Unhandled operand.  Halt "fast" selection and bail.
    return false;
    
  unsigned ResultReg = FastEmit_r(SrcVT.getSimpleVT(),
                                  DstVT.getSimpleVT(),
                                  Opcode,
                                  InputReg);
  if (!ResultReg)
    return false;
    
  UpdateValueMap(I, ResultReg);
  return true;
}

bool FastISel::SelectBitCast(Instruction *I) {
  // If the bitcast doesn't change the type, just use the operand value.
  if (I->getType() == I->getOperand(0)->getType()) {
    unsigned Reg = getRegForValue(I->getOperand(0));
    if (Reg == 0)
      return false;
    UpdateValueMap(I, Reg);
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
  
  unsigned Op0 = getRegForValue(I->getOperand(0));
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
  
  UpdateValueMap(I, ResultReg);
  return true;
}

bool
FastISel::SelectInstruction(Instruction *I) {
  switch (I->getOpcode()) {
  case Instruction::Add: {
    ISD::NodeType Opc = I->getType()->isFPOrFPVector() ? ISD::FADD : ISD::ADD;
    return SelectBinaryOp(I, Opc);
  }
  case Instruction::Sub: {
    ISD::NodeType Opc = I->getType()->isFPOrFPVector() ? ISD::FSUB : ISD::SUB;
    return SelectBinaryOp(I, Opc);
  }
  case Instruction::Mul: {
    ISD::NodeType Opc = I->getType()->isFPOrFPVector() ? ISD::FMUL : ISD::MUL;
    return SelectBinaryOp(I, Opc);
  }
  case Instruction::SDiv:
    return SelectBinaryOp(I, ISD::SDIV);
  case Instruction::UDiv:
    return SelectBinaryOp(I, ISD::UDIV);
  case Instruction::FDiv:
    return SelectBinaryOp(I, ISD::FDIV);
  case Instruction::SRem:
    return SelectBinaryOp(I, ISD::SREM);
  case Instruction::URem:
    return SelectBinaryOp(I, ISD::UREM);
  case Instruction::FRem:
    return SelectBinaryOp(I, ISD::FREM);
  case Instruction::Shl:
    return SelectBinaryOp(I, ISD::SHL);
  case Instruction::LShr:
    return SelectBinaryOp(I, ISD::SRL);
  case Instruction::AShr:
    return SelectBinaryOp(I, ISD::SRA);
  case Instruction::And:
    return SelectBinaryOp(I, ISD::AND);
  case Instruction::Or:
    return SelectBinaryOp(I, ISD::OR);
  case Instruction::Xor:
    return SelectBinaryOp(I, ISD::XOR);

  case Instruction::GetElementPtr:
    return SelectGetElementPtr(I);

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
      return true;
    }

    // Conditional branches are not handed yet.
    // Halt "fast" selection and bail.
    return false;
  }

  case Instruction::PHI:
    // PHI nodes are already emitted.
    return true;
    
  case Instruction::BitCast:
    return SelectBitCast(I);

  case Instruction::FPToSI:
    return SelectCast(I, ISD::FP_TO_SINT);
  case Instruction::ZExt:
    return SelectCast(I, ISD::ZERO_EXTEND);
  case Instruction::SExt:
    return SelectCast(I, ISD::SIGN_EXTEND);
  case Instruction::Trunc:
    return SelectCast(I, ISD::TRUNCATE);
  case Instruction::SIToFP:
    return SelectCast(I, ISD::SINT_TO_FP);

  case Instruction::IntToPtr: // Deliberate fall-through.
  case Instruction::PtrToInt: {
    MVT SrcVT = TLI.getValueType(I->getOperand(0)->getType());
    MVT DstVT = TLI.getValueType(I->getType());
    if (DstVT.bitsGT(SrcVT))
      return SelectCast(I, ISD::ZERO_EXTEND);
    if (DstVT.bitsLT(SrcVT))
      return SelectCast(I, ISD::TRUNCATE);
    unsigned Reg = getRegForValue(I->getOperand(0));
    if (Reg == 0) return false;
    UpdateValueMap(I, Reg);
    return true;
  }
  
  default:
    // Unhandled instruction. Halt "fast" selection and bail.
    return false;
  }
}

FastISel::FastISel(MachineFunction &mf,
                   DenseMap<const Value *, unsigned> &vm,
                   DenseMap<const BasicBlock *, MachineBasicBlock *> &bm)
  : MBB(0),
    ValueMap(vm),
    MBBMap(bm),
    MF(mf),
    MRI(MF.getRegInfo()),
    TM(MF.getTarget()),
    TD(*TM.getTargetData()),
    TII(*TM.getInstrInfo()),
    TLI(*TM.getTargetLowering()) {
}

FastISel::~FastISel() {}

unsigned FastISel::FastEmit_(MVT::SimpleValueType, MVT::SimpleValueType,
                             ISD::NodeType) {
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

unsigned FastISel::FastEmitInst_extractsubreg(unsigned Op0, uint32_t Idx) {
  const TargetRegisterClass* RC = MRI.getRegClass(Op0);
  const TargetRegisterClass* SRC = *(RC->subregclasses_begin()+Idx-1);
  
  unsigned ResultReg = createResultReg(SRC);
  const TargetInstrDesc &II = TII.get(TargetInstrInfo::EXTRACT_SUBREG);
  
  BuildMI(MBB, II, ResultReg).addReg(Op0).addImm(Idx);
  return ResultReg;
}
