//===-- FastISel.cpp - Implementation of the FastISel class ---------------===//
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
// "Fast" instruction selection is designed to emit very poor code quickly.
// Also, it is not designed to be able to do much lowering, so most illegal
// types (e.g. i64 on 32-bit targets) and operations are not supported.  It is
// also not intended to be able to do much optimization, except in a few cases
// where doing optimizations reduces overall compile time.  For example, folding
// constants into immediate fields is often done, because it's cheap and it
// reduces the number of instructions later phases have to examine.
//
// "Fast" instruction selection is able to fail gracefully and transfer
// control to the SelectionDAG selector for operations that it doesn't
// support.  In many cases, this allows us to avoid duplicating a lot of
// the complicated lowering logic that SelectionDAG currently has.
//
// The intended use for "fast" instruction selection is "-O0" mode
// compilation, where the quality of the generated code is irrelevant when
// weighed against the speed at which the code can be generated.  Also,
// at -O0, the LLVM optimizers are not running, and this makes the
// compile time of codegen a much higher portion of the overall compile
// time.  Despite its limitations, "fast" instruction selection is able to
// handle enough code on its own to provide noticeable overall speedups
// in -O0 compiles.
//
// Basic operations are supported in a target-independent way, by reading
// the same instruction descriptions that the SelectionDAG selector reads,
// and identifying simple arithmetic operations that can be directly selected
// from simple operators.  More complicated operations currently require
// target-specific code.
//
//===----------------------------------------------------------------------===//

#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/CodeGen/FastISel.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Analysis/DebugInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/ErrorHandling.h"
#include "FunctionLoweringInfo.h"
using namespace llvm;

bool FastISel::hasTrivialKill(const Value *V) const {
  // Don't consider constants or arguments to have trivial kills.
  const Instruction *I = dyn_cast<Instruction>(V);
  if (!I)
    return false;

  // No-op casts are trivially coalesced by fast-isel.
  if (const CastInst *Cast = dyn_cast<CastInst>(I))
    if (Cast->isNoopCast(TD.getIntPtrType(Cast->getContext())) &&
        !hasTrivialKill(Cast->getOperand(0)))
      return false;

  // Only instructions with a single use in the same basic block are considered
  // to have trivial kills.
  return I->hasOneUse() &&
         !(I->getOpcode() == Instruction::BitCast ||
           I->getOpcode() == Instruction::PtrToInt ||
           I->getOpcode() == Instruction::IntToPtr) &&
         cast<Instruction>(I->use_begin())->getParent() == I->getParent();
}

unsigned FastISel::getRegForValue(const Value *V) {
  EVT RealVT = TLI.getValueType(V->getType(), /*AllowUnknown=*/true);
  // Don't handle non-simple values in FastISel.
  if (!RealVT.isSimple())
    return 0;

  // Ignore illegal types. We must do this before looking up the value
  // in ValueMap because Arguments are given virtual registers regardless
  // of whether FastISel can handle them.
  MVT VT = RealVT.getSimpleVT();
  if (!TLI.isTypeLegal(VT)) {
    // Promote MVT::i1 to a legal type though, because it's common and easy.
    if (VT == MVT::i1)
      VT = TLI.getTypeToTransformTo(V->getContext(), VT).getSimpleVT();
    else
      return 0;
  }

  // Look up the value to see if we already have a register for it. We
  // cache values defined by Instructions across blocks, and other values
  // only locally. This is because Instructions already have the SSA
  // def-dominates-use requirement enforced.
  if (ValueMap.count(V))
    return ValueMap[V];
  unsigned Reg = LocalValueMap[V];
  if (Reg != 0)
    return Reg;

  // In bottom-up mode, just create the virtual register which will be used
  // to hold the value. It will be materialized later.
  if (IsBottomUp) {
    Reg = createResultReg(TLI.getRegClassFor(VT));
    if (isa<Instruction>(V))
      ValueMap[V] = Reg;
    else
      LocalValueMap[V] = Reg;
    return Reg;
  }

  return materializeRegForValue(V, VT);
}

/// materializeRegForValue - Helper for getRegForVale. This function is
/// called when the value isn't already available in a register and must
/// be materialized with new instructions.
unsigned FastISel::materializeRegForValue(const Value *V, MVT VT) {
  unsigned Reg = 0;

  if (const ConstantInt *CI = dyn_cast<ConstantInt>(V)) {
    if (CI->getValue().getActiveBits() <= 64)
      Reg = FastEmit_i(VT, VT, ISD::Constant, CI->getZExtValue());
  } else if (isa<AllocaInst>(V)) {
    Reg = TargetMaterializeAlloca(cast<AllocaInst>(V));
  } else if (isa<ConstantPointerNull>(V)) {
    // Translate this as an integer zero so that it can be
    // local-CSE'd with actual integer zeros.
    Reg =
      getRegForValue(Constant::getNullValue(TD.getIntPtrType(V->getContext())));
  } else if (const ConstantFP *CF = dyn_cast<ConstantFP>(V)) {
    // Try to emit the constant directly.
    Reg = FastEmit_f(VT, VT, ISD::ConstantFP, CF);

    if (!Reg) {
      // Try to emit the constant by using an integer constant with a cast.
      const APFloat &Flt = CF->getValueAPF();
      EVT IntVT = TLI.getPointerTy();

      uint64_t x[2];
      uint32_t IntBitWidth = IntVT.getSizeInBits();
      bool isExact;
      (void) Flt.convertToInteger(x, IntBitWidth, /*isSigned=*/true,
                                APFloat::rmTowardZero, &isExact);
      if (isExact) {
        APInt IntVal(IntBitWidth, 2, x);

        unsigned IntegerReg =
          getRegForValue(ConstantInt::get(V->getContext(), IntVal));
        if (IntegerReg != 0)
          Reg = FastEmit_r(IntVT.getSimpleVT(), VT, ISD::SINT_TO_FP,
                           IntegerReg, /*Kill=*/false);
      }
    }
  } else if (const Operator *Op = dyn_cast<Operator>(V)) {
    if (!SelectOperator(Op, Op->getOpcode())) return 0;
    Reg = LocalValueMap[Op];
  } else if (isa<UndefValue>(V)) {
    Reg = createResultReg(TLI.getRegClassFor(VT));
    BuildMI(MBB, DL, TII.get(TargetOpcode::IMPLICIT_DEF), Reg);
  }
  
  // If target-independent code couldn't handle the value, give target-specific
  // code a try.
  if (!Reg && isa<Constant>(V))
    Reg = TargetMaterializeConstant(cast<Constant>(V));
  
  // Don't cache constant materializations in the general ValueMap.
  // To do so would require tracking what uses they dominate.
  if (Reg != 0)
    LocalValueMap[V] = Reg;
  return Reg;
}

unsigned FastISel::lookUpRegForValue(const Value *V) {
  // Look up the value to see if we already have a register for it. We
  // cache values defined by Instructions across blocks, and other values
  // only locally. This is because Instructions already have the SSA
  // def-dominates-use requirement enforced.
  if (ValueMap.count(V))
    return ValueMap[V];
  return LocalValueMap[V];
}

/// UpdateValueMap - Update the value map to include the new mapping for this
/// instruction, or insert an extra copy to get the result in a previous
/// determined register.
/// NOTE: This is only necessary because we might select a block that uses
/// a value before we select the block that defines the value.  It might be
/// possible to fix this by selecting blocks in reverse postorder.
unsigned FastISel::UpdateValueMap(const Value *I, unsigned Reg) {
  if (!isa<Instruction>(I)) {
    LocalValueMap[I] = Reg;
    return Reg;
  }
  
  unsigned &AssignedReg = ValueMap[I];
  if (AssignedReg == 0)
    AssignedReg = Reg;
  else if (Reg != AssignedReg) {
    const TargetRegisterClass *RegClass = MRI.getRegClass(Reg);
    TII.copyRegToReg(*MBB, MBB->end(), AssignedReg,
                     Reg, RegClass, RegClass, DL);
  }
  return AssignedReg;
}

std::pair<unsigned, bool> FastISel::getRegForGEPIndex(const Value *Idx) {
  unsigned IdxN = getRegForValue(Idx);
  if (IdxN == 0)
    // Unhandled operand. Halt "fast" selection and bail.
    return std::pair<unsigned, bool>(0, false);

  bool IdxNIsKill = hasTrivialKill(Idx);

  // If the index is smaller or larger than intptr_t, truncate or extend it.
  MVT PtrVT = TLI.getPointerTy();
  EVT IdxVT = EVT::getEVT(Idx->getType(), /*HandleUnknown=*/false);
  if (IdxVT.bitsLT(PtrVT)) {
    IdxN = FastEmit_r(IdxVT.getSimpleVT(), PtrVT, ISD::SIGN_EXTEND,
                      IdxN, IdxNIsKill);
    IdxNIsKill = true;
  }
  else if (IdxVT.bitsGT(PtrVT)) {
    IdxN = FastEmit_r(IdxVT.getSimpleVT(), PtrVT, ISD::TRUNCATE,
                      IdxN, IdxNIsKill);
    IdxNIsKill = true;
  }
  return std::pair<unsigned, bool>(IdxN, IdxNIsKill);
}

/// SelectBinaryOp - Select and emit code for a binary operator instruction,
/// which has an opcode which directly corresponds to the given ISD opcode.
///
bool FastISel::SelectBinaryOp(const User *I, unsigned ISDOpcode) {
  EVT VT = EVT::getEVT(I->getType(), /*HandleUnknown=*/true);
  if (VT == MVT::Other || !VT.isSimple())
    // Unhandled type. Halt "fast" selection and bail.
    return false;

  // We only handle legal types. For example, on x86-32 the instruction
  // selector contains all of the 64-bit instructions from x86-64,
  // under the assumption that i64 won't be used if the target doesn't
  // support it.
  if (!TLI.isTypeLegal(VT)) {
    // MVT::i1 is special. Allow AND, OR, or XOR because they
    // don't require additional zeroing, which makes them easy.
    if (VT == MVT::i1 &&
        (ISDOpcode == ISD::AND || ISDOpcode == ISD::OR ||
         ISDOpcode == ISD::XOR))
      VT = TLI.getTypeToTransformTo(I->getContext(), VT);
    else
      return false;
  }

  unsigned Op0 = getRegForValue(I->getOperand(0));
  if (Op0 == 0)
    // Unhandled operand. Halt "fast" selection and bail.
    return false;

  bool Op0IsKill = hasTrivialKill(I->getOperand(0));

  // Check if the second operand is a constant and handle it appropriately.
  if (ConstantInt *CI = dyn_cast<ConstantInt>(I->getOperand(1))) {
    unsigned ResultReg = FastEmit_ri(VT.getSimpleVT(), VT.getSimpleVT(),
                                     ISDOpcode, Op0, Op0IsKill,
                                     CI->getZExtValue());
    if (ResultReg != 0) {
      // We successfully emitted code for the given LLVM Instruction.
      UpdateValueMap(I, ResultReg);
      return true;
    }
  }

  // Check if the second operand is a constant float.
  if (ConstantFP *CF = dyn_cast<ConstantFP>(I->getOperand(1))) {
    unsigned ResultReg = FastEmit_rf(VT.getSimpleVT(), VT.getSimpleVT(),
                                     ISDOpcode, Op0, Op0IsKill, CF);
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

  bool Op1IsKill = hasTrivialKill(I->getOperand(1));

  // Now we have both operands in registers. Emit the instruction.
  unsigned ResultReg = FastEmit_rr(VT.getSimpleVT(), VT.getSimpleVT(),
                                   ISDOpcode,
                                   Op0, Op0IsKill,
                                   Op1, Op1IsKill);
  if (ResultReg == 0)
    // Target-specific code wasn't able to find a machine opcode for
    // the given ISD opcode and type. Halt "fast" selection and bail.
    return false;

  // We successfully emitted code for the given LLVM Instruction.
  UpdateValueMap(I, ResultReg);
  return true;
}

bool FastISel::SelectGetElementPtr(const User *I) {
  unsigned N = getRegForValue(I->getOperand(0));
  if (N == 0)
    // Unhandled operand. Halt "fast" selection and bail.
    return false;

  bool NIsKill = hasTrivialKill(I->getOperand(0));

  const Type *Ty = I->getOperand(0)->getType();
  MVT VT = TLI.getPointerTy();
  for (GetElementPtrInst::const_op_iterator OI = I->op_begin()+1,
       E = I->op_end(); OI != E; ++OI) {
    const Value *Idx = *OI;
    if (const StructType *StTy = dyn_cast<StructType>(Ty)) {
      unsigned Field = cast<ConstantInt>(Idx)->getZExtValue();
      if (Field) {
        // N = N + Offset
        uint64_t Offs = TD.getStructLayout(StTy)->getElementOffset(Field);
        // FIXME: This can be optimized by combining the add with a
        // subsequent one.
        N = FastEmit_ri_(VT, ISD::ADD, N, NIsKill, Offs, VT);
        if (N == 0)
          // Unhandled operand. Halt "fast" selection and bail.
          return false;
        NIsKill = true;
      }
      Ty = StTy->getElementType(Field);
    } else {
      Ty = cast<SequentialType>(Ty)->getElementType();

      // If this is a constant subscript, handle it quickly.
      if (const ConstantInt *CI = dyn_cast<ConstantInt>(Idx)) {
        if (CI->getZExtValue() == 0) continue;
        uint64_t Offs = 
          TD.getTypeAllocSize(Ty)*cast<ConstantInt>(CI)->getSExtValue();
        N = FastEmit_ri_(VT, ISD::ADD, N, NIsKill, Offs, VT);
        if (N == 0)
          // Unhandled operand. Halt "fast" selection and bail.
          return false;
        NIsKill = true;
        continue;
      }
      
      // N = N + Idx * ElementSize;
      uint64_t ElementSize = TD.getTypeAllocSize(Ty);
      std::pair<unsigned, bool> Pair = getRegForGEPIndex(Idx);
      unsigned IdxN = Pair.first;
      bool IdxNIsKill = Pair.second;
      if (IdxN == 0)
        // Unhandled operand. Halt "fast" selection and bail.
        return false;

      if (ElementSize != 1) {
        IdxN = FastEmit_ri_(VT, ISD::MUL, IdxN, IdxNIsKill, ElementSize, VT);
        if (IdxN == 0)
          // Unhandled operand. Halt "fast" selection and bail.
          return false;
        IdxNIsKill = true;
      }
      N = FastEmit_rr(VT, VT, ISD::ADD, N, NIsKill, IdxN, IdxNIsKill);
      if (N == 0)
        // Unhandled operand. Halt "fast" selection and bail.
        return false;
    }
  }

  // We successfully emitted code for the given LLVM Instruction.
  UpdateValueMap(I, N);
  return true;
}

bool FastISel::SelectCall(const User *I) {
  const Function *F = cast<CallInst>(I)->getCalledFunction();
  if (!F) return false;

  // Handle selected intrinsic function calls.
  unsigned IID = F->getIntrinsicID();
  switch (IID) {
  default: break;
  case Intrinsic::dbg_declare: {
    const DbgDeclareInst *DI = cast<DbgDeclareInst>(I);
    if (!DIVariable(DI->getVariable()).Verify() ||
        !MF.getMMI().hasDebugInfo())
      return true;

    const Value *Address = DI->getAddress();
    if (!Address)
      return true;
    if (isa<UndefValue>(Address))
      return true;
    const AllocaInst *AI = dyn_cast<AllocaInst>(Address);
    // Don't handle byval struct arguments or VLAs, for example.
    // Note that if we have a byval struct argument, fast ISel is turned off;
    // those are handled in SelectionDAGBuilder.
    if (AI) {
      DenseMap<const AllocaInst*, int>::iterator SI =
        StaticAllocaMap.find(AI);
      if (SI == StaticAllocaMap.end()) break; // VLAs.
      int FI = SI->second;
      if (!DI->getDebugLoc().isUnknown())
        MF.getMMI().setVariableDbgInfo(DI->getVariable(), FI, DI->getDebugLoc());
    } else
      // Building the map above is target independent.  Generating DBG_VALUE
      // inline is target dependent; do this now.
      (void)TargetSelectInstruction(cast<Instruction>(I));
    return true;
  }
  case Intrinsic::dbg_value: {
    // This form of DBG_VALUE is target-independent.
    const DbgValueInst *DI = cast<DbgValueInst>(I);
    const TargetInstrDesc &II = TII.get(TargetOpcode::DBG_VALUE);
    const Value *V = DI->getValue();
    if (!V) {
      // Currently the optimizer can produce this; insert an undef to
      // help debugging.  Probably the optimizer should not do this.
      BuildMI(MBB, DL, II).addReg(0U).addImm(DI->getOffset()).
                                     addMetadata(DI->getVariable());
    } else if (const ConstantInt *CI = dyn_cast<ConstantInt>(V)) {
      BuildMI(MBB, DL, II).addImm(CI->getZExtValue()).addImm(DI->getOffset()).
                                     addMetadata(DI->getVariable());
    } else if (const ConstantFP *CF = dyn_cast<ConstantFP>(V)) {
      BuildMI(MBB, DL, II).addFPImm(CF).addImm(DI->getOffset()).
                                     addMetadata(DI->getVariable());
    } else if (unsigned Reg = lookUpRegForValue(V)) {
      BuildMI(MBB, DL, II).addReg(Reg, RegState::Debug).addImm(DI->getOffset()).
                                     addMetadata(DI->getVariable());
    } else {
      // We can't yet handle anything else here because it would require
      // generating code, thus altering codegen because of debug info.
      // Insert an undef so we can see what we dropped.
      BuildMI(MBB, DL, II).addReg(0U).addImm(DI->getOffset()).
                                     addMetadata(DI->getVariable());
    }     
    return true;
  }
  case Intrinsic::eh_exception: {
    EVT VT = TLI.getValueType(I->getType());
    switch (TLI.getOperationAction(ISD::EXCEPTIONADDR, VT)) {
    default: break;
    case TargetLowering::Expand: {
      assert(MBB->isLandingPad() && "Call to eh.exception not in landing pad!");
      unsigned Reg = TLI.getExceptionAddressRegister();
      const TargetRegisterClass *RC = TLI.getRegClassFor(VT);
      unsigned ResultReg = createResultReg(RC);
      bool InsertedCopy = TII.copyRegToReg(*MBB, MBB->end(), ResultReg,
                                           Reg, RC, RC, DL);
      assert(InsertedCopy && "Can't copy address registers!");
      InsertedCopy = InsertedCopy;
      UpdateValueMap(I, ResultReg);
      return true;
    }
    }
    break;
  }
  case Intrinsic::eh_selector: {
    EVT VT = TLI.getValueType(I->getType());
    switch (TLI.getOperationAction(ISD::EHSELECTION, VT)) {
    default: break;
    case TargetLowering::Expand: {
      if (MBB->isLandingPad())
        AddCatchInfo(*cast<CallInst>(I), &MF.getMMI(), MBB);
      else {
#ifndef NDEBUG
        CatchInfoLost.insert(cast<CallInst>(I));
#endif
        // FIXME: Mark exception selector register as live in.  Hack for PR1508.
        unsigned Reg = TLI.getExceptionSelectorRegister();
        if (Reg) MBB->addLiveIn(Reg);
      }

      unsigned Reg = TLI.getExceptionSelectorRegister();
      EVT SrcVT = TLI.getPointerTy();
      const TargetRegisterClass *RC = TLI.getRegClassFor(SrcVT);
      unsigned ResultReg = createResultReg(RC);
      bool InsertedCopy = TII.copyRegToReg(*MBB, MBB->end(), ResultReg, Reg,
                                           RC, RC, DL);
      assert(InsertedCopy && "Can't copy address registers!");
      InsertedCopy = InsertedCopy;

      bool ResultRegIsKill = hasTrivialKill(I);

      // Cast the register to the type of the selector.
      if (SrcVT.bitsGT(MVT::i32))
        ResultReg = FastEmit_r(SrcVT.getSimpleVT(), MVT::i32, ISD::TRUNCATE,
                               ResultReg, ResultRegIsKill);
      else if (SrcVT.bitsLT(MVT::i32))
        ResultReg = FastEmit_r(SrcVT.getSimpleVT(), MVT::i32,
                               ISD::SIGN_EXTEND, ResultReg, ResultRegIsKill);
      if (ResultReg == 0)
        // Unhandled operand. Halt "fast" selection and bail.
        return false;

      UpdateValueMap(I, ResultReg);

      return true;
    }
    }
    break;
  }
  }

  // An arbitrary call. Bail.
  return false;
}

bool FastISel::SelectCast(const User *I, unsigned Opcode) {
  EVT SrcVT = TLI.getValueType(I->getOperand(0)->getType());
  EVT DstVT = TLI.getValueType(I->getType());
    
  if (SrcVT == MVT::Other || !SrcVT.isSimple() ||
      DstVT == MVT::Other || !DstVT.isSimple())
    // Unhandled type. Halt "fast" selection and bail.
    return false;
    
  // Check if the destination type is legal. Or as a special case,
  // it may be i1 if we're doing a truncate because that's
  // easy and somewhat common.
  if (!TLI.isTypeLegal(DstVT))
    if (DstVT != MVT::i1 || Opcode != ISD::TRUNCATE)
      // Unhandled type. Halt "fast" selection and bail.
      return false;

  // Check if the source operand is legal. Or as a special case,
  // it may be i1 if we're doing zero-extension because that's
  // easy and somewhat common.
  if (!TLI.isTypeLegal(SrcVT))
    if (SrcVT != MVT::i1 || Opcode != ISD::ZERO_EXTEND)
      // Unhandled type. Halt "fast" selection and bail.
      return false;

  unsigned InputReg = getRegForValue(I->getOperand(0));
  if (!InputReg)
    // Unhandled operand.  Halt "fast" selection and bail.
    return false;

  bool InputRegIsKill = hasTrivialKill(I->getOperand(0));

  // If the operand is i1, arrange for the high bits in the register to be zero.
  if (SrcVT == MVT::i1) {
   SrcVT = TLI.getTypeToTransformTo(I->getContext(), SrcVT);
   InputReg = FastEmitZExtFromI1(SrcVT.getSimpleVT(), InputReg, InputRegIsKill);
   if (!InputReg)
     return false;
   InputRegIsKill = true;
  }
  // If the result is i1, truncate to the target's type for i1 first.
  if (DstVT == MVT::i1)
    DstVT = TLI.getTypeToTransformTo(I->getContext(), DstVT);

  unsigned ResultReg = FastEmit_r(SrcVT.getSimpleVT(),
                                  DstVT.getSimpleVT(),
                                  Opcode,
                                  InputReg, InputRegIsKill);
  if (!ResultReg)
    return false;
    
  UpdateValueMap(I, ResultReg);
  return true;
}

bool FastISel::SelectBitCast(const User *I) {
  // If the bitcast doesn't change the type, just use the operand value.
  if (I->getType() == I->getOperand(0)->getType()) {
    unsigned Reg = getRegForValue(I->getOperand(0));
    if (Reg == 0)
      return false;
    UpdateValueMap(I, Reg);
    return true;
  }

  // Bitcasts of other values become reg-reg copies or BIT_CONVERT operators.
  EVT SrcVT = TLI.getValueType(I->getOperand(0)->getType());
  EVT DstVT = TLI.getValueType(I->getType());
  
  if (SrcVT == MVT::Other || !SrcVT.isSimple() ||
      DstVT == MVT::Other || !DstVT.isSimple() ||
      !TLI.isTypeLegal(SrcVT) || !TLI.isTypeLegal(DstVT))
    // Unhandled type. Halt "fast" selection and bail.
    return false;
  
  unsigned Op0 = getRegForValue(I->getOperand(0));
  if (Op0 == 0)
    // Unhandled operand. Halt "fast" selection and bail.
    return false;

  bool Op0IsKill = hasTrivialKill(I->getOperand(0));
  
  // First, try to perform the bitcast by inserting a reg-reg copy.
  unsigned ResultReg = 0;
  if (SrcVT.getSimpleVT() == DstVT.getSimpleVT()) {
    TargetRegisterClass* SrcClass = TLI.getRegClassFor(SrcVT);
    TargetRegisterClass* DstClass = TLI.getRegClassFor(DstVT);
    ResultReg = createResultReg(DstClass);
    
    bool InsertedCopy = TII.copyRegToReg(*MBB, MBB->end(), ResultReg,
                                         Op0, DstClass, SrcClass, DL);
    if (!InsertedCopy)
      ResultReg = 0;
  }
  
  // If the reg-reg copy failed, select a BIT_CONVERT opcode.
  if (!ResultReg)
    ResultReg = FastEmit_r(SrcVT.getSimpleVT(), DstVT.getSimpleVT(),
                           ISD::BIT_CONVERT, Op0, Op0IsKill);
  
  if (!ResultReg)
    return false;
  
  UpdateValueMap(I, ResultReg);
  return true;
}

bool
FastISel::SelectInstruction(const Instruction *I) {
  // Just before the terminator instruction, insert instructions to
  // feed PHI nodes in successor blocks.
  if (isa<TerminatorInst>(I))
    if (!HandlePHINodesInSuccessorBlocks(I->getParent()))
      return false;

  DL = I->getDebugLoc();

  // First, try doing target-independent selection.
  if (SelectOperator(I, I->getOpcode())) {
    DL = DebugLoc();
    return true;
  }

  // Next, try calling the target to attempt to handle the instruction.
  if (TargetSelectInstruction(I)) {
    DL = DebugLoc();
    return true;
  }

  DL = DebugLoc();
  return false;
}

/// FastEmitBranch - Emit an unconditional branch to the given block,
/// unless it is the immediate (fall-through) successor, and update
/// the CFG.
void
FastISel::FastEmitBranch(MachineBasicBlock *MSucc) {
  if (MBB->isLayoutSuccessor(MSucc)) {
    // The unconditional fall-through case, which needs no instructions.
  } else {
    // The unconditional branch case.
    TII.InsertBranch(*MBB, MSucc, NULL, SmallVector<MachineOperand, 0>());
  }
  MBB->addSuccessor(MSucc);
}

/// SelectFNeg - Emit an FNeg operation.
///
bool
FastISel::SelectFNeg(const User *I) {
  unsigned OpReg = getRegForValue(BinaryOperator::getFNegArgument(I));
  if (OpReg == 0) return false;

  bool OpRegIsKill = hasTrivialKill(I);

  // If the target has ISD::FNEG, use it.
  EVT VT = TLI.getValueType(I->getType());
  unsigned ResultReg = FastEmit_r(VT.getSimpleVT(), VT.getSimpleVT(),
                                  ISD::FNEG, OpReg, OpRegIsKill);
  if (ResultReg != 0) {
    UpdateValueMap(I, ResultReg);
    return true;
  }

  // Bitcast the value to integer, twiddle the sign bit with xor,
  // and then bitcast it back to floating-point.
  if (VT.getSizeInBits() > 64) return false;
  EVT IntVT = EVT::getIntegerVT(I->getContext(), VT.getSizeInBits());
  if (!TLI.isTypeLegal(IntVT))
    return false;

  unsigned IntReg = FastEmit_r(VT.getSimpleVT(), IntVT.getSimpleVT(),
                               ISD::BIT_CONVERT, OpReg, OpRegIsKill);
  if (IntReg == 0)
    return false;

  unsigned IntResultReg = FastEmit_ri_(IntVT.getSimpleVT(), ISD::XOR,
                                       IntReg, /*Kill=*/true,
                                       UINT64_C(1) << (VT.getSizeInBits()-1),
                                       IntVT.getSimpleVT());
  if (IntResultReg == 0)
    return false;

  ResultReg = FastEmit_r(IntVT.getSimpleVT(), VT.getSimpleVT(),
                         ISD::BIT_CONVERT, IntResultReg, /*Kill=*/true);
  if (ResultReg == 0)
    return false;

  UpdateValueMap(I, ResultReg);
  return true;
}

bool
FastISel::SelectOperator(const User *I, unsigned Opcode) {
  switch (Opcode) {
  case Instruction::Add:
    return SelectBinaryOp(I, ISD::ADD);
  case Instruction::FAdd:
    return SelectBinaryOp(I, ISD::FADD);
  case Instruction::Sub:
    return SelectBinaryOp(I, ISD::SUB);
  case Instruction::FSub:
    // FNeg is currently represented in LLVM IR as a special case of FSub.
    if (BinaryOperator::isFNeg(I))
      return SelectFNeg(I);
    return SelectBinaryOp(I, ISD::FSUB);
  case Instruction::Mul:
    return SelectBinaryOp(I, ISD::MUL);
  case Instruction::FMul:
    return SelectBinaryOp(I, ISD::FMUL);
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
    const BranchInst *BI = cast<BranchInst>(I);

    if (BI->isUnconditional()) {
      const BasicBlock *LLVMSucc = BI->getSuccessor(0);
      MachineBasicBlock *MSucc = MBBMap[LLVMSucc];
      FastEmitBranch(MSucc);
      return true;
    }

    // Conditional branches are not handed yet.
    // Halt "fast" selection and bail.
    return false;
  }

  case Instruction::Unreachable:
    // Nothing to emit.
    return true;

  case Instruction::Alloca:
    // FunctionLowering has the static-sized case covered.
    if (StaticAllocaMap.count(cast<AllocaInst>(I)))
      return true;

    // Dynamic-sized alloca is not handled yet.
    return false;
    
  case Instruction::Call:
    return SelectCall(I);
  
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
    EVT SrcVT = TLI.getValueType(I->getOperand(0)->getType());
    EVT DstVT = TLI.getValueType(I->getType());
    if (DstVT.bitsGT(SrcVT))
      return SelectCast(I, ISD::ZERO_EXTEND);
    if (DstVT.bitsLT(SrcVT))
      return SelectCast(I, ISD::TRUNCATE);
    unsigned Reg = getRegForValue(I->getOperand(0));
    if (Reg == 0) return false;
    UpdateValueMap(I, Reg);
    return true;
  }

  case Instruction::PHI:
    llvm_unreachable("FastISel shouldn't visit PHI nodes!");

  default:
    // Unhandled instruction. Halt "fast" selection and bail.
    return false;
  }
}

FastISel::FastISel(MachineFunction &mf,
                   DenseMap<const Value *, unsigned> &vm,
                   DenseMap<const BasicBlock *, MachineBasicBlock *> &bm,
                   DenseMap<const AllocaInst *, int> &am,
                   std::vector<std::pair<MachineInstr*, unsigned> > &pn
#ifndef NDEBUG
                   , SmallSet<const Instruction *, 8> &cil
#endif
                   )
  : MBB(0),
    ValueMap(vm),
    MBBMap(bm),
    StaticAllocaMap(am),
    PHINodesToUpdate(pn),
#ifndef NDEBUG
    CatchInfoLost(cil),
#endif
    MF(mf),
    MRI(MF.getRegInfo()),
    MFI(*MF.getFrameInfo()),
    MCP(*MF.getConstantPool()),
    TM(MF.getTarget()),
    TD(*TM.getTargetData()),
    TII(*TM.getInstrInfo()),
    TLI(*TM.getTargetLowering()),
    IsBottomUp(false) {
}

FastISel::~FastISel() {}

unsigned FastISel::FastEmit_(MVT, MVT,
                             unsigned) {
  return 0;
}

unsigned FastISel::FastEmit_r(MVT, MVT,
                              unsigned,
                              unsigned /*Op0*/, bool /*Op0IsKill*/) {
  return 0;
}

unsigned FastISel::FastEmit_rr(MVT, MVT, 
                               unsigned,
                               unsigned /*Op0*/, bool /*Op0IsKill*/,
                               unsigned /*Op1*/, bool /*Op1IsKill*/) {
  return 0;
}

unsigned FastISel::FastEmit_i(MVT, MVT, unsigned, uint64_t /*Imm*/) {
  return 0;
}

unsigned FastISel::FastEmit_f(MVT, MVT,
                              unsigned, const ConstantFP * /*FPImm*/) {
  return 0;
}

unsigned FastISel::FastEmit_ri(MVT, MVT,
                               unsigned,
                               unsigned /*Op0*/, bool /*Op0IsKill*/,
                               uint64_t /*Imm*/) {
  return 0;
}

unsigned FastISel::FastEmit_rf(MVT, MVT,
                               unsigned,
                               unsigned /*Op0*/, bool /*Op0IsKill*/,
                               const ConstantFP * /*FPImm*/) {
  return 0;
}

unsigned FastISel::FastEmit_rri(MVT, MVT,
                                unsigned,
                                unsigned /*Op0*/, bool /*Op0IsKill*/,
                                unsigned /*Op1*/, bool /*Op1IsKill*/,
                                uint64_t /*Imm*/) {
  return 0;
}

/// FastEmit_ri_ - This method is a wrapper of FastEmit_ri. It first tries
/// to emit an instruction with an immediate operand using FastEmit_ri.
/// If that fails, it materializes the immediate into a register and try
/// FastEmit_rr instead.
unsigned FastISel::FastEmit_ri_(MVT VT, unsigned Opcode,
                                unsigned Op0, bool Op0IsKill,
                                uint64_t Imm, MVT ImmType) {
  // First check if immediate type is legal. If not, we can't use the ri form.
  unsigned ResultReg = FastEmit_ri(VT, VT, Opcode, Op0, Op0IsKill, Imm);
  if (ResultReg != 0)
    return ResultReg;
  unsigned MaterialReg = FastEmit_i(ImmType, ImmType, ISD::Constant, Imm);
  if (MaterialReg == 0)
    return 0;
  return FastEmit_rr(VT, VT, Opcode,
                     Op0, Op0IsKill,
                     MaterialReg, /*Kill=*/true);
}

/// FastEmit_rf_ - This method is a wrapper of FastEmit_ri. It first tries
/// to emit an instruction with a floating-point immediate operand using
/// FastEmit_rf. If that fails, it materializes the immediate into a register
/// and try FastEmit_rr instead.
unsigned FastISel::FastEmit_rf_(MVT VT, unsigned Opcode,
                                unsigned Op0, bool Op0IsKill,
                                const ConstantFP *FPImm, MVT ImmType) {
  // First check if immediate type is legal. If not, we can't use the rf form.
  unsigned ResultReg = FastEmit_rf(VT, VT, Opcode, Op0, Op0IsKill, FPImm);
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
    EVT IntVT = TLI.getPointerTy();

    uint64_t x[2];
    uint32_t IntBitWidth = IntVT.getSizeInBits();
    bool isExact;
    (void) Flt.convertToInteger(x, IntBitWidth, /*isSigned=*/true,
                             APFloat::rmTowardZero, &isExact);
    if (!isExact)
      return 0;
    APInt IntVal(IntBitWidth, 2, x);

    unsigned IntegerReg = FastEmit_i(IntVT.getSimpleVT(), IntVT.getSimpleVT(),
                                     ISD::Constant, IntVal.getZExtValue());
    if (IntegerReg == 0)
      return 0;
    MaterialReg = FastEmit_r(IntVT.getSimpleVT(), VT,
                             ISD::SINT_TO_FP, IntegerReg, /*Kill=*/true);
    if (MaterialReg == 0)
      return 0;
  }
  return FastEmit_rr(VT, VT, Opcode,
                     Op0, Op0IsKill,
                     MaterialReg, /*Kill=*/true);
}

unsigned FastISel::createResultReg(const TargetRegisterClass* RC) {
  return MRI.createVirtualRegister(RC);
}

unsigned FastISel::FastEmitInst_(unsigned MachineInstOpcode,
                                 const TargetRegisterClass* RC) {
  unsigned ResultReg = createResultReg(RC);
  const TargetInstrDesc &II = TII.get(MachineInstOpcode);

  BuildMI(MBB, DL, II, ResultReg);
  return ResultReg;
}

unsigned FastISel::FastEmitInst_r(unsigned MachineInstOpcode,
                                  const TargetRegisterClass *RC,
                                  unsigned Op0, bool Op0IsKill) {
  unsigned ResultReg = createResultReg(RC);
  const TargetInstrDesc &II = TII.get(MachineInstOpcode);

  if (II.getNumDefs() >= 1)
    BuildMI(MBB, DL, II, ResultReg).addReg(Op0, Op0IsKill * RegState::Kill);
  else {
    BuildMI(MBB, DL, II).addReg(Op0, Op0IsKill * RegState::Kill);
    bool InsertedCopy = TII.copyRegToReg(*MBB, MBB->end(), ResultReg,
                                         II.ImplicitDefs[0], RC, RC, DL);
    if (!InsertedCopy)
      ResultReg = 0;
  }

  return ResultReg;
}

unsigned FastISel::FastEmitInst_rr(unsigned MachineInstOpcode,
                                   const TargetRegisterClass *RC,
                                   unsigned Op0, bool Op0IsKill,
                                   unsigned Op1, bool Op1IsKill) {
  unsigned ResultReg = createResultReg(RC);
  const TargetInstrDesc &II = TII.get(MachineInstOpcode);

  if (II.getNumDefs() >= 1)
    BuildMI(MBB, DL, II, ResultReg)
      .addReg(Op0, Op0IsKill * RegState::Kill)
      .addReg(Op1, Op1IsKill * RegState::Kill);
  else {
    BuildMI(MBB, DL, II)
      .addReg(Op0, Op0IsKill * RegState::Kill)
      .addReg(Op1, Op1IsKill * RegState::Kill);
    bool InsertedCopy = TII.copyRegToReg(*MBB, MBB->end(), ResultReg,
                                         II.ImplicitDefs[0], RC, RC, DL);
    if (!InsertedCopy)
      ResultReg = 0;
  }
  return ResultReg;
}

unsigned FastISel::FastEmitInst_ri(unsigned MachineInstOpcode,
                                   const TargetRegisterClass *RC,
                                   unsigned Op0, bool Op0IsKill,
                                   uint64_t Imm) {
  unsigned ResultReg = createResultReg(RC);
  const TargetInstrDesc &II = TII.get(MachineInstOpcode);

  if (II.getNumDefs() >= 1)
    BuildMI(MBB, DL, II, ResultReg)
      .addReg(Op0, Op0IsKill * RegState::Kill)
      .addImm(Imm);
  else {
    BuildMI(MBB, DL, II)
      .addReg(Op0, Op0IsKill * RegState::Kill)
      .addImm(Imm);
    bool InsertedCopy = TII.copyRegToReg(*MBB, MBB->end(), ResultReg,
                                         II.ImplicitDefs[0], RC, RC, DL);
    if (!InsertedCopy)
      ResultReg = 0;
  }
  return ResultReg;
}

unsigned FastISel::FastEmitInst_rf(unsigned MachineInstOpcode,
                                   const TargetRegisterClass *RC,
                                   unsigned Op0, bool Op0IsKill,
                                   const ConstantFP *FPImm) {
  unsigned ResultReg = createResultReg(RC);
  const TargetInstrDesc &II = TII.get(MachineInstOpcode);

  if (II.getNumDefs() >= 1)
    BuildMI(MBB, DL, II, ResultReg)
      .addReg(Op0, Op0IsKill * RegState::Kill)
      .addFPImm(FPImm);
  else {
    BuildMI(MBB, DL, II)
      .addReg(Op0, Op0IsKill * RegState::Kill)
      .addFPImm(FPImm);
    bool InsertedCopy = TII.copyRegToReg(*MBB, MBB->end(), ResultReg,
                                         II.ImplicitDefs[0], RC, RC, DL);
    if (!InsertedCopy)
      ResultReg = 0;
  }
  return ResultReg;
}

unsigned FastISel::FastEmitInst_rri(unsigned MachineInstOpcode,
                                    const TargetRegisterClass *RC,
                                    unsigned Op0, bool Op0IsKill,
                                    unsigned Op1, bool Op1IsKill,
                                    uint64_t Imm) {
  unsigned ResultReg = createResultReg(RC);
  const TargetInstrDesc &II = TII.get(MachineInstOpcode);

  if (II.getNumDefs() >= 1)
    BuildMI(MBB, DL, II, ResultReg)
      .addReg(Op0, Op0IsKill * RegState::Kill)
      .addReg(Op1, Op1IsKill * RegState::Kill)
      .addImm(Imm);
  else {
    BuildMI(MBB, DL, II)
      .addReg(Op0, Op0IsKill * RegState::Kill)
      .addReg(Op1, Op1IsKill * RegState::Kill)
      .addImm(Imm);
    bool InsertedCopy = TII.copyRegToReg(*MBB, MBB->end(), ResultReg,
                                         II.ImplicitDefs[0], RC, RC, DL);
    if (!InsertedCopy)
      ResultReg = 0;
  }
  return ResultReg;
}

unsigned FastISel::FastEmitInst_i(unsigned MachineInstOpcode,
                                  const TargetRegisterClass *RC,
                                  uint64_t Imm) {
  unsigned ResultReg = createResultReg(RC);
  const TargetInstrDesc &II = TII.get(MachineInstOpcode);
  
  if (II.getNumDefs() >= 1)
    BuildMI(MBB, DL, II, ResultReg).addImm(Imm);
  else {
    BuildMI(MBB, DL, II).addImm(Imm);
    bool InsertedCopy = TII.copyRegToReg(*MBB, MBB->end(), ResultReg,
                                         II.ImplicitDefs[0], RC, RC, DL);
    if (!InsertedCopy)
      ResultReg = 0;
  }
  return ResultReg;
}

unsigned FastISel::FastEmitInst_extractsubreg(MVT RetVT,
                                              unsigned Op0, bool Op0IsKill,
                                              uint32_t Idx) {
  const TargetRegisterClass* RC = MRI.getRegClass(Op0);
  
  unsigned ResultReg = createResultReg(TLI.getRegClassFor(RetVT));
  const TargetInstrDesc &II = TII.get(TargetOpcode::EXTRACT_SUBREG);
  
  if (II.getNumDefs() >= 1)
    BuildMI(MBB, DL, II, ResultReg)
      .addReg(Op0, Op0IsKill * RegState::Kill)
      .addImm(Idx);
  else {
    BuildMI(MBB, DL, II)
      .addReg(Op0, Op0IsKill * RegState::Kill)
      .addImm(Idx);
    bool InsertedCopy = TII.copyRegToReg(*MBB, MBB->end(), ResultReg,
                                         II.ImplicitDefs[0], RC, RC, DL);
    if (!InsertedCopy)
      ResultReg = 0;
  }
  return ResultReg;
}

/// FastEmitZExtFromI1 - Emit MachineInstrs to compute the value of Op
/// with all but the least significant bit set to zero.
unsigned FastISel::FastEmitZExtFromI1(MVT VT, unsigned Op0, bool Op0IsKill) {
  return FastEmit_ri(VT, VT, ISD::AND, Op0, Op0IsKill, 1);
}

/// HandlePHINodesInSuccessorBlocks - Handle PHI nodes in successor blocks.
/// Emit code to ensure constants are copied into registers when needed.
/// Remember the virtual registers that need to be added to the Machine PHI
/// nodes as input.  We cannot just directly add them, because expansion
/// might result in multiple MBB's for one BB.  As such, the start of the
/// BB might correspond to a different MBB than the end.
bool FastISel::HandlePHINodesInSuccessorBlocks(const BasicBlock *LLVMBB) {
  const TerminatorInst *TI = LLVMBB->getTerminator();

  SmallPtrSet<MachineBasicBlock *, 4> SuccsHandled;
  unsigned OrigNumPHINodesToUpdate = PHINodesToUpdate.size();

  // Check successor nodes' PHI nodes that expect a constant to be available
  // from this block.
  for (unsigned succ = 0, e = TI->getNumSuccessors(); succ != e; ++succ) {
    const BasicBlock *SuccBB = TI->getSuccessor(succ);
    if (!isa<PHINode>(SuccBB->begin())) continue;
    MachineBasicBlock *SuccMBB = MBBMap[SuccBB];

    // If this terminator has multiple identical successors (common for
    // switches), only handle each succ once.
    if (!SuccsHandled.insert(SuccMBB)) continue;

    MachineBasicBlock::iterator MBBI = SuccMBB->begin();

    // At this point we know that there is a 1-1 correspondence between LLVM PHI
    // nodes and Machine PHI nodes, but the incoming operands have not been
    // emitted yet.
    for (BasicBlock::const_iterator I = SuccBB->begin();
         const PHINode *PN = dyn_cast<PHINode>(I); ++I) {

      // Ignore dead phi's.
      if (PN->use_empty()) continue;

      // Only handle legal types. Two interesting things to note here. First,
      // by bailing out early, we may leave behind some dead instructions,
      // since SelectionDAG's HandlePHINodesInSuccessorBlocks will insert its
      // own moves. Second, this check is necessary becuase FastISel doesn't
      // use CreateRegForValue to create registers, so it always creates
      // exactly one register for each non-void instruction.
      EVT VT = TLI.getValueType(PN->getType(), /*AllowUnknown=*/true);
      if (VT == MVT::Other || !TLI.isTypeLegal(VT)) {
        // Promote MVT::i1.
        if (VT == MVT::i1)
          VT = TLI.getTypeToTransformTo(LLVMBB->getContext(), VT);
        else {
          PHINodesToUpdate.resize(OrigNumPHINodesToUpdate);
          return false;
        }
      }

      const Value *PHIOp = PN->getIncomingValueForBlock(LLVMBB);

      // Set the DebugLoc for the copy. Prefer the location of the operand
      // if there is one; use the location of the PHI otherwise.
      DL = PN->getDebugLoc();
      if (const Instruction *Inst = dyn_cast<Instruction>(PHIOp))
        DL = Inst->getDebugLoc();

      unsigned Reg = getRegForValue(PHIOp);
      if (Reg == 0) {
        PHINodesToUpdate.resize(OrigNumPHINodesToUpdate);
        return false;
      }
      PHINodesToUpdate.push_back(std::make_pair(MBBI++, Reg));
      DL = DebugLoc();
    }
  }

  return true;
}
