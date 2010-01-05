//===- InstCombineCalls.cpp -----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the visitCall and visitInvoke functions.
//
//===----------------------------------------------------------------------===//

#include "InstCombine.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Analysis/MemoryBuiltins.h"
using namespace llvm;

/// getPromotedType - Return the specified type promoted as it would be to pass
/// though a va_arg area.
static const Type *getPromotedType(const Type *Ty) {
  if (const IntegerType* ITy = dyn_cast<IntegerType>(Ty)) {
    if (ITy->getBitWidth() < 32)
      return Type::getInt32Ty(Ty->getContext());
  }
  return Ty;
}

/// EnforceKnownAlignment - If the specified pointer points to an object that
/// we control, modify the object's alignment to PrefAlign. This isn't
/// often possible though. If alignment is important, a more reliable approach
/// is to simply align all global variables and allocation instructions to
/// their preferred alignment from the beginning.
///
static unsigned EnforceKnownAlignment(Value *V,
                                      unsigned Align, unsigned PrefAlign) {

  User *U = dyn_cast<User>(V);
  if (!U) return Align;

  switch (Operator::getOpcode(U)) {
  default: break;
  case Instruction::BitCast:
    return EnforceKnownAlignment(U->getOperand(0), Align, PrefAlign);
  case Instruction::GetElementPtr: {
    // If all indexes are zero, it is just the alignment of the base pointer.
    bool AllZeroOperands = true;
    for (User::op_iterator i = U->op_begin() + 1, e = U->op_end(); i != e; ++i)
      if (!isa<Constant>(*i) ||
          !cast<Constant>(*i)->isNullValue()) {
        AllZeroOperands = false;
        break;
      }

    if (AllZeroOperands) {
      // Treat this like a bitcast.
      return EnforceKnownAlignment(U->getOperand(0), Align, PrefAlign);
    }
    break;
  }
  }

  if (GlobalValue *GV = dyn_cast<GlobalValue>(V)) {
    // If there is a large requested alignment and we can, bump up the alignment
    // of the global.
    if (!GV->isDeclaration()) {
      if (GV->getAlignment() >= PrefAlign)
        Align = GV->getAlignment();
      else {
        GV->setAlignment(PrefAlign);
        Align = PrefAlign;
      }
    }
  } else if (AllocaInst *AI = dyn_cast<AllocaInst>(V)) {
    // If there is a requested alignment and if this is an alloca, round up.
    if (AI->getAlignment() >= PrefAlign)
      Align = AI->getAlignment();
    else {
      AI->setAlignment(PrefAlign);
      Align = PrefAlign;
    }
  }

  return Align;
}

/// GetOrEnforceKnownAlignment - If the specified pointer has an alignment that
/// we can determine, return it, otherwise return 0.  If PrefAlign is specified,
/// and it is more than the alignment of the ultimate object, see if we can
/// increase the alignment of the ultimate object, making this check succeed.
unsigned InstCombiner::GetOrEnforceKnownAlignment(Value *V,
                                                  unsigned PrefAlign) {
  unsigned BitWidth = TD ? TD->getTypeSizeInBits(V->getType()) :
                      sizeof(PrefAlign) * CHAR_BIT;
  APInt Mask = APInt::getAllOnesValue(BitWidth);
  APInt KnownZero(BitWidth, 0), KnownOne(BitWidth, 0);
  ComputeMaskedBits(V, Mask, KnownZero, KnownOne);
  unsigned TrailZ = KnownZero.countTrailingOnes();
  unsigned Align = 1u << std::min(BitWidth - 1, TrailZ);

  if (PrefAlign > Align)
    Align = EnforceKnownAlignment(V, Align, PrefAlign);
  
    // We don't need to make any adjustment.
  return Align;
}

Instruction *InstCombiner::SimplifyMemTransfer(MemIntrinsic *MI) {
  unsigned DstAlign = GetOrEnforceKnownAlignment(MI->getOperand(1));
  unsigned SrcAlign = GetOrEnforceKnownAlignment(MI->getOperand(2));
  unsigned MinAlign = std::min(DstAlign, SrcAlign);
  unsigned CopyAlign = MI->getAlignment();

  if (CopyAlign < MinAlign) {
    MI->setAlignment(ConstantInt::get(MI->getAlignmentType(), 
                                             MinAlign, false));
    return MI;
  }
  
  // If MemCpyInst length is 1/2/4/8 bytes then replace memcpy with
  // load/store.
  ConstantInt *MemOpLength = dyn_cast<ConstantInt>(MI->getOperand(3));
  if (MemOpLength == 0) return 0;
  
  // Source and destination pointer types are always "i8*" for intrinsic.  See
  // if the size is something we can handle with a single primitive load/store.
  // A single load+store correctly handles overlapping memory in the memmove
  // case.
  unsigned Size = MemOpLength->getZExtValue();
  if (Size == 0) return MI;  // Delete this mem transfer.
  
  if (Size > 8 || (Size&(Size-1)))
    return 0;  // If not 1/2/4/8 bytes, exit.
  
  // Use an integer load+store unless we can find something better.
  Type *NewPtrTy =
            PointerType::getUnqual(IntegerType::get(MI->getContext(), Size<<3));
  
  // Memcpy forces the use of i8* for the source and destination.  That means
  // that if you're using memcpy to move one double around, you'll get a cast
  // from double* to i8*.  We'd much rather use a double load+store rather than
  // an i64 load+store, here because this improves the odds that the source or
  // dest address will be promotable.  See if we can find a better type than the
  // integer datatype.
  Value *StrippedDest = MI->getOperand(1)->stripPointerCasts();
  if (StrippedDest != MI->getOperand(1)) {
    const Type *SrcETy = cast<PointerType>(StrippedDest->getType())
                                    ->getElementType();
    if (TD && SrcETy->isSized() && TD->getTypeStoreSize(SrcETy) == Size) {
      // The SrcETy might be something like {{{double}}} or [1 x double].  Rip
      // down through these levels if so.
      while (!SrcETy->isSingleValueType()) {
        if (const StructType *STy = dyn_cast<StructType>(SrcETy)) {
          if (STy->getNumElements() == 1)
            SrcETy = STy->getElementType(0);
          else
            break;
        } else if (const ArrayType *ATy = dyn_cast<ArrayType>(SrcETy)) {
          if (ATy->getNumElements() == 1)
            SrcETy = ATy->getElementType();
          else
            break;
        } else
          break;
      }
      
      if (SrcETy->isSingleValueType())
        NewPtrTy = PointerType::getUnqual(SrcETy);
    }
  }
  
  
  // If the memcpy/memmove provides better alignment info than we can
  // infer, use it.
  SrcAlign = std::max(SrcAlign, CopyAlign);
  DstAlign = std::max(DstAlign, CopyAlign);
  
  Value *Src = Builder->CreateBitCast(MI->getOperand(2), NewPtrTy);
  Value *Dest = Builder->CreateBitCast(MI->getOperand(1), NewPtrTy);
  Instruction *L = new LoadInst(Src, "tmp", false, SrcAlign);
  InsertNewInstBefore(L, *MI);
  InsertNewInstBefore(new StoreInst(L, Dest, false, DstAlign), *MI);

  // Set the size of the copy to 0, it will be deleted on the next iteration.
  MI->setOperand(3, Constant::getNullValue(MemOpLength->getType()));
  return MI;
}

Instruction *InstCombiner::SimplifyMemSet(MemSetInst *MI) {
  unsigned Alignment = GetOrEnforceKnownAlignment(MI->getDest());
  if (MI->getAlignment() < Alignment) {
    MI->setAlignment(ConstantInt::get(MI->getAlignmentType(),
                                             Alignment, false));
    return MI;
  }
  
  // Extract the length and alignment and fill if they are constant.
  ConstantInt *LenC = dyn_cast<ConstantInt>(MI->getLength());
  ConstantInt *FillC = dyn_cast<ConstantInt>(MI->getValue());
  if (!LenC || !FillC || FillC->getType() != Type::getInt8Ty(MI->getContext()))
    return 0;
  uint64_t Len = LenC->getZExtValue();
  Alignment = MI->getAlignment();
  
  // If the length is zero, this is a no-op
  if (Len == 0) return MI; // memset(d,c,0,a) -> noop
  
  // memset(s,c,n) -> store s, c (for n=1,2,4,8)
  if (Len <= 8 && isPowerOf2_32((uint32_t)Len)) {
    const Type *ITy = IntegerType::get(MI->getContext(), Len*8);  // n=1 -> i8.
    
    Value *Dest = MI->getDest();
    Dest = Builder->CreateBitCast(Dest, PointerType::getUnqual(ITy));

    // Alignment 0 is identity for alignment 1 for memset, but not store.
    if (Alignment == 0) Alignment = 1;
    
    // Extract the fill value and store.
    uint64_t Fill = FillC->getZExtValue()*0x0101010101010101ULL;
    InsertNewInstBefore(new StoreInst(ConstantInt::get(ITy, Fill),
                                      Dest, false, Alignment), *MI);
    
    // Set the size of the copy to 0, it will be deleted on the next iteration.
    MI->setLength(Constant::getNullValue(LenC->getType()));
    return MI;
  }

  return 0;
}


/// visitCallInst - CallInst simplification.  This mostly only handles folding 
/// of intrinsic instructions.  For normal calls, it allows visitCallSite to do
/// the heavy lifting.
///
Instruction *InstCombiner::visitCallInst(CallInst &CI) {
  if (isFreeCall(&CI))
    return visitFree(CI);

  // If the caller function is nounwind, mark the call as nounwind, even if the
  // callee isn't.
  if (CI.getParent()->getParent()->doesNotThrow() &&
      !CI.doesNotThrow()) {
    CI.setDoesNotThrow();
    return &CI;
  }
  
  IntrinsicInst *II = dyn_cast<IntrinsicInst>(&CI);
  if (!II) return visitCallSite(&CI);
  
  // Intrinsics cannot occur in an invoke, so handle them here instead of in
  // visitCallSite.
  if (MemIntrinsic *MI = dyn_cast<MemIntrinsic>(II)) {
    bool Changed = false;

    // memmove/cpy/set of zero bytes is a noop.
    if (Constant *NumBytes = dyn_cast<Constant>(MI->getLength())) {
      if (NumBytes->isNullValue()) return EraseInstFromFunction(CI);

      if (ConstantInt *CI = dyn_cast<ConstantInt>(NumBytes))
        if (CI->getZExtValue() == 1) {
          // Replace the instruction with just byte operations.  We would
          // transform other cases to loads/stores, but we don't know if
          // alignment is sufficient.
        }
    }

    // If we have a memmove and the source operation is a constant global,
    // then the source and dest pointers can't alias, so we can change this
    // into a call to memcpy.
    if (MemMoveInst *MMI = dyn_cast<MemMoveInst>(MI)) {
      if (GlobalVariable *GVSrc = dyn_cast<GlobalVariable>(MMI->getSource()))
        if (GVSrc->isConstant()) {
          Module *M = CI.getParent()->getParent()->getParent();
          Intrinsic::ID MemCpyID = Intrinsic::memcpy;
          const Type *Tys[1];
          Tys[0] = CI.getOperand(3)->getType();
          CI.setOperand(0, 
                        Intrinsic::getDeclaration(M, MemCpyID, Tys, 1));
          Changed = true;
        }
    }

    if (MemTransferInst *MTI = dyn_cast<MemTransferInst>(MI)) {
      // memmove(x,x,size) -> noop.
      if (MTI->getSource() == MTI->getDest())
        return EraseInstFromFunction(CI);
    }

    // If we can determine a pointer alignment that is bigger than currently
    // set, update the alignment.
    if (isa<MemTransferInst>(MI)) {
      if (Instruction *I = SimplifyMemTransfer(MI))
        return I;
    } else if (MemSetInst *MSI = dyn_cast<MemSetInst>(MI)) {
      if (Instruction *I = SimplifyMemSet(MSI))
        return I;
    }
          
    if (Changed) return II;
  }
  
  switch (II->getIntrinsicID()) {
  default: break;
  case Intrinsic::bswap:
    // bswap(bswap(x)) -> x
    if (IntrinsicInst *Operand = dyn_cast<IntrinsicInst>(II->getOperand(1)))
      if (Operand->getIntrinsicID() == Intrinsic::bswap)
        return ReplaceInstUsesWith(CI, Operand->getOperand(1));
      
    // bswap(trunc(bswap(x))) -> trunc(lshr(x, c))
    if (TruncInst *TI = dyn_cast<TruncInst>(II->getOperand(1))) {
      if (IntrinsicInst *Operand = dyn_cast<IntrinsicInst>(TI->getOperand(0)))
        if (Operand->getIntrinsicID() == Intrinsic::bswap) {
          unsigned C = Operand->getType()->getPrimitiveSizeInBits() -
                       TI->getType()->getPrimitiveSizeInBits();
          Value *CV = ConstantInt::get(Operand->getType(), C);
          Value *V = Builder->CreateLShr(Operand->getOperand(1), CV);
          return new TruncInst(V, TI->getType());
        }
    }
      
    break;
  case Intrinsic::powi:
    if (ConstantInt *Power = dyn_cast<ConstantInt>(II->getOperand(2))) {
      // powi(x, 0) -> 1.0
      if (Power->isZero())
        return ReplaceInstUsesWith(CI, ConstantFP::get(CI.getType(), 1.0));
      // powi(x, 1) -> x
      if (Power->isOne())
        return ReplaceInstUsesWith(CI, II->getOperand(1));
      // powi(x, -1) -> 1/x
      if (Power->isAllOnesValue())
        return BinaryOperator::CreateFDiv(ConstantFP::get(CI.getType(), 1.0),
                                          II->getOperand(1));
    }
    break;
  case Intrinsic::cttz: {
    // If all bits below the first known one are known zero,
    // this value is constant.
    const IntegerType *IT = cast<IntegerType>(II->getOperand(1)->getType());
    uint32_t BitWidth = IT->getBitWidth();
    APInt KnownZero(BitWidth, 0);
    APInt KnownOne(BitWidth, 0);
    ComputeMaskedBits(II->getOperand(1), APInt::getAllOnesValue(BitWidth),
                      KnownZero, KnownOne);
    unsigned TrailingZeros = KnownOne.countTrailingZeros();
    APInt Mask(APInt::getLowBitsSet(BitWidth, TrailingZeros));
    if ((Mask & KnownZero) == Mask)
      return ReplaceInstUsesWith(CI, ConstantInt::get(IT,
                                 APInt(BitWidth, TrailingZeros)));
    
    }
    break;
  case Intrinsic::ctlz: {
    // If all bits above the first known one are known zero,
    // this value is constant.
    const IntegerType *IT = cast<IntegerType>(II->getOperand(1)->getType());
    uint32_t BitWidth = IT->getBitWidth();
    APInt KnownZero(BitWidth, 0);
    APInt KnownOne(BitWidth, 0);
    ComputeMaskedBits(II->getOperand(1), APInt::getAllOnesValue(BitWidth),
                      KnownZero, KnownOne);
    unsigned LeadingZeros = KnownOne.countLeadingZeros();
    APInt Mask(APInt::getHighBitsSet(BitWidth, LeadingZeros));
    if ((Mask & KnownZero) == Mask)
      return ReplaceInstUsesWith(CI, ConstantInt::get(IT,
                                 APInt(BitWidth, LeadingZeros)));
    
    }
    break;
  case Intrinsic::uadd_with_overflow: {
    Value *LHS = II->getOperand(1), *RHS = II->getOperand(2);
    const IntegerType *IT = cast<IntegerType>(II->getOperand(1)->getType());
    uint32_t BitWidth = IT->getBitWidth();
    APInt Mask = APInt::getSignBit(BitWidth);
    APInt LHSKnownZero(BitWidth, 0);
    APInt LHSKnownOne(BitWidth, 0);
    ComputeMaskedBits(LHS, Mask, LHSKnownZero, LHSKnownOne);
    bool LHSKnownNegative = LHSKnownOne[BitWidth - 1];
    bool LHSKnownPositive = LHSKnownZero[BitWidth - 1];

    if (LHSKnownNegative || LHSKnownPositive) {
      APInt RHSKnownZero(BitWidth, 0);
      APInt RHSKnownOne(BitWidth, 0);
      ComputeMaskedBits(RHS, Mask, RHSKnownZero, RHSKnownOne);
      bool RHSKnownNegative = RHSKnownOne[BitWidth - 1];
      bool RHSKnownPositive = RHSKnownZero[BitWidth - 1];
      if (LHSKnownNegative && RHSKnownNegative) {
        // The sign bit is set in both cases: this MUST overflow.
        // Create a simple add instruction, and insert it into the struct.
        Instruction *Add = BinaryOperator::CreateAdd(LHS, RHS, "", &CI);
        Worklist.Add(Add);
        Constant *V[] = {
          UndefValue::get(LHS->getType()),ConstantInt::getTrue(II->getContext())
        };
        Constant *Struct = ConstantStruct::get(II->getContext(), V, 2, false);
        return InsertValueInst::Create(Struct, Add, 0);
      }
      
      if (LHSKnownPositive && RHSKnownPositive) {
        // The sign bit is clear in both cases: this CANNOT overflow.
        // Create a simple add instruction, and insert it into the struct.
        Instruction *Add = BinaryOperator::CreateNUWAdd(LHS, RHS, "", &CI);
        Worklist.Add(Add);
        Constant *V[] = {
          UndefValue::get(LHS->getType()),
          ConstantInt::getFalse(II->getContext())
        };
        Constant *Struct = ConstantStruct::get(II->getContext(), V, 2, false);
        return InsertValueInst::Create(Struct, Add, 0);
      }
    }
  }
  // FALL THROUGH uadd into sadd
  case Intrinsic::sadd_with_overflow:
    // Canonicalize constants into the RHS.
    if (isa<Constant>(II->getOperand(1)) &&
        !isa<Constant>(II->getOperand(2))) {
      Value *LHS = II->getOperand(1);
      II->setOperand(1, II->getOperand(2));
      II->setOperand(2, LHS);
      return II;
    }

    // X + undef -> undef
    if (isa<UndefValue>(II->getOperand(2)))
      return ReplaceInstUsesWith(CI, UndefValue::get(II->getType()));
      
    if (ConstantInt *RHS = dyn_cast<ConstantInt>(II->getOperand(2))) {
      // X + 0 -> {X, false}
      if (RHS->isZero()) {
        Constant *V[] = {
          UndefValue::get(II->getOperand(0)->getType()),
          ConstantInt::getFalse(II->getContext())
        };
        Constant *Struct = ConstantStruct::get(II->getContext(), V, 2, false);
        return InsertValueInst::Create(Struct, II->getOperand(1), 0);
      }
    }
    break;
  case Intrinsic::usub_with_overflow:
  case Intrinsic::ssub_with_overflow:
    // undef - X -> undef
    // X - undef -> undef
    if (isa<UndefValue>(II->getOperand(1)) ||
        isa<UndefValue>(II->getOperand(2)))
      return ReplaceInstUsesWith(CI, UndefValue::get(II->getType()));
      
    if (ConstantInt *RHS = dyn_cast<ConstantInt>(II->getOperand(2))) {
      // X - 0 -> {X, false}
      if (RHS->isZero()) {
        Constant *V[] = {
          UndefValue::get(II->getOperand(1)->getType()),
          ConstantInt::getFalse(II->getContext())
        };
        Constant *Struct = ConstantStruct::get(II->getContext(), V, 2, false);
        return InsertValueInst::Create(Struct, II->getOperand(1), 0);
      }
    }
    break;
  case Intrinsic::umul_with_overflow:
  case Intrinsic::smul_with_overflow:
    // Canonicalize constants into the RHS.
    if (isa<Constant>(II->getOperand(1)) &&
        !isa<Constant>(II->getOperand(2))) {
      Value *LHS = II->getOperand(1);
      II->setOperand(1, II->getOperand(2));
      II->setOperand(2, LHS);
      return II;
    }

    // X * undef -> undef
    if (isa<UndefValue>(II->getOperand(2)))
      return ReplaceInstUsesWith(CI, UndefValue::get(II->getType()));
      
    if (ConstantInt *RHSI = dyn_cast<ConstantInt>(II->getOperand(2))) {
      // X*0 -> {0, false}
      if (RHSI->isZero())
        return ReplaceInstUsesWith(CI, Constant::getNullValue(II->getType()));
      
      // X * 1 -> {X, false}
      if (RHSI->equalsInt(1)) {
        Constant *V[] = {
          UndefValue::get(II->getOperand(1)->getType()),
          ConstantInt::getFalse(II->getContext())
        };
        Constant *Struct = ConstantStruct::get(II->getContext(), V, 2, false);
        return InsertValueInst::Create(Struct, II->getOperand(1), 0);
      }
    }
    break;
  case Intrinsic::ppc_altivec_lvx:
  case Intrinsic::ppc_altivec_lvxl:
  case Intrinsic::x86_sse_loadu_ps:
  case Intrinsic::x86_sse2_loadu_pd:
  case Intrinsic::x86_sse2_loadu_dq:
    // Turn PPC lvx     -> load if the pointer is known aligned.
    // Turn X86 loadups -> load if the pointer is known aligned.
    if (GetOrEnforceKnownAlignment(II->getOperand(1), 16) >= 16) {
      Value *Ptr = Builder->CreateBitCast(II->getOperand(1),
                                         PointerType::getUnqual(II->getType()));
      return new LoadInst(Ptr);
    }
    break;
  case Intrinsic::ppc_altivec_stvx:
  case Intrinsic::ppc_altivec_stvxl:
    // Turn stvx -> store if the pointer is known aligned.
    if (GetOrEnforceKnownAlignment(II->getOperand(2), 16) >= 16) {
      const Type *OpPtrTy = 
        PointerType::getUnqual(II->getOperand(1)->getType());
      Value *Ptr = Builder->CreateBitCast(II->getOperand(2), OpPtrTy);
      return new StoreInst(II->getOperand(1), Ptr);
    }
    break;
  case Intrinsic::x86_sse_storeu_ps:
  case Intrinsic::x86_sse2_storeu_pd:
  case Intrinsic::x86_sse2_storeu_dq:
    // Turn X86 storeu -> store if the pointer is known aligned.
    if (GetOrEnforceKnownAlignment(II->getOperand(1), 16) >= 16) {
      const Type *OpPtrTy = 
        PointerType::getUnqual(II->getOperand(2)->getType());
      Value *Ptr = Builder->CreateBitCast(II->getOperand(1), OpPtrTy);
      return new StoreInst(II->getOperand(2), Ptr);
    }
    break;
    
  case Intrinsic::x86_sse_cvttss2si: {
    // These intrinsics only demands the 0th element of its input vector.  If
    // we can simplify the input based on that, do so now.
    unsigned VWidth =
      cast<VectorType>(II->getOperand(1)->getType())->getNumElements();
    APInt DemandedElts(VWidth, 1);
    APInt UndefElts(VWidth, 0);
    if (Value *V = SimplifyDemandedVectorElts(II->getOperand(1), DemandedElts,
                                              UndefElts)) {
      II->setOperand(1, V);
      return II;
    }
    break;
  }
    
  case Intrinsic::ppc_altivec_vperm:
    // Turn vperm(V1,V2,mask) -> shuffle(V1,V2,mask) if mask is a constant.
    if (ConstantVector *Mask = dyn_cast<ConstantVector>(II->getOperand(3))) {
      assert(Mask->getNumOperands() == 16 && "Bad type for intrinsic!");
      
      // Check that all of the elements are integer constants or undefs.
      bool AllEltsOk = true;
      for (unsigned i = 0; i != 16; ++i) {
        if (!isa<ConstantInt>(Mask->getOperand(i)) && 
            !isa<UndefValue>(Mask->getOperand(i))) {
          AllEltsOk = false;
          break;
        }
      }
      
      if (AllEltsOk) {
        // Cast the input vectors to byte vectors.
        Value *Op0 = Builder->CreateBitCast(II->getOperand(1), Mask->getType());
        Value *Op1 = Builder->CreateBitCast(II->getOperand(2), Mask->getType());
        Value *Result = UndefValue::get(Op0->getType());
        
        // Only extract each element once.
        Value *ExtractedElts[32];
        memset(ExtractedElts, 0, sizeof(ExtractedElts));
        
        for (unsigned i = 0; i != 16; ++i) {
          if (isa<UndefValue>(Mask->getOperand(i)))
            continue;
          unsigned Idx=cast<ConstantInt>(Mask->getOperand(i))->getZExtValue();
          Idx &= 31;  // Match the hardware behavior.
          
          if (ExtractedElts[Idx] == 0) {
            ExtractedElts[Idx] = 
              Builder->CreateExtractElement(Idx < 16 ? Op0 : Op1, 
                  ConstantInt::get(Type::getInt32Ty(II->getContext()),
                                   Idx&15, false), "tmp");
          }
        
          // Insert this value into the result vector.
          Result = Builder->CreateInsertElement(Result, ExtractedElts[Idx],
                         ConstantInt::get(Type::getInt32Ty(II->getContext()),
                                          i, false), "tmp");
        }
        return CastInst::Create(Instruction::BitCast, Result, CI.getType());
      }
    }
    break;

  case Intrinsic::stackrestore: {
    // If the save is right next to the restore, remove the restore.  This can
    // happen when variable allocas are DCE'd.
    if (IntrinsicInst *SS = dyn_cast<IntrinsicInst>(II->getOperand(1))) {
      if (SS->getIntrinsicID() == Intrinsic::stacksave) {
        BasicBlock::iterator BI = SS;
        if (&*++BI == II)
          return EraseInstFromFunction(CI);
      }
    }
    
    // Scan down this block to see if there is another stack restore in the
    // same block without an intervening call/alloca.
    BasicBlock::iterator BI = II;
    TerminatorInst *TI = II->getParent()->getTerminator();
    bool CannotRemove = false;
    for (++BI; &*BI != TI; ++BI) {
      if (isa<AllocaInst>(BI) || isMalloc(BI)) {
        CannotRemove = true;
        break;
      }
      if (CallInst *BCI = dyn_cast<CallInst>(BI)) {
        if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(BCI)) {
          // If there is a stackrestore below this one, remove this one.
          if (II->getIntrinsicID() == Intrinsic::stackrestore)
            return EraseInstFromFunction(CI);
          // Otherwise, ignore the intrinsic.
        } else {
          // If we found a non-intrinsic call, we can't remove the stack
          // restore.
          CannotRemove = true;
          break;
        }
      }
    }
    
    // If the stack restore is in a return/unwind block and if there are no
    // allocas or calls between the restore and the return, nuke the restore.
    if (!CannotRemove && (isa<ReturnInst>(TI) || isa<UnwindInst>(TI)))
      return EraseInstFromFunction(CI);
    break;
  }
  }

  return visitCallSite(II);
}

// InvokeInst simplification
//
Instruction *InstCombiner::visitInvokeInst(InvokeInst &II) {
  return visitCallSite(&II);
}

/// isSafeToEliminateVarargsCast - If this cast does not affect the value 
/// passed through the varargs area, we can eliminate the use of the cast.
static bool isSafeToEliminateVarargsCast(const CallSite CS,
                                         const CastInst * const CI,
                                         const TargetData * const TD,
                                         const int ix) {
  if (!CI->isLosslessCast())
    return false;

  // The size of ByVal arguments is derived from the type, so we
  // can't change to a type with a different size.  If the size were
  // passed explicitly we could avoid this check.
  if (!CS.paramHasAttr(ix, Attribute::ByVal))
    return true;

  const Type* SrcTy = 
            cast<PointerType>(CI->getOperand(0)->getType())->getElementType();
  const Type* DstTy = cast<PointerType>(CI->getType())->getElementType();
  if (!SrcTy->isSized() || !DstTy->isSized())
    return false;
  if (!TD || TD->getTypeAllocSize(SrcTy) != TD->getTypeAllocSize(DstTy))
    return false;
  return true;
}

// visitCallSite - Improvements for call and invoke instructions.
//
Instruction *InstCombiner::visitCallSite(CallSite CS) {
  bool Changed = false;

  // If the callee is a constexpr cast of a function, attempt to move the cast
  // to the arguments of the call/invoke.
  if (transformConstExprCastCall(CS)) return 0;

  Value *Callee = CS.getCalledValue();

  if (Function *CalleeF = dyn_cast<Function>(Callee))
    if (CalleeF->getCallingConv() != CS.getCallingConv()) {
      Instruction *OldCall = CS.getInstruction();
      // If the call and callee calling conventions don't match, this call must
      // be unreachable, as the call is undefined.
      new StoreInst(ConstantInt::getTrue(Callee->getContext()),
                UndefValue::get(Type::getInt1PtrTy(Callee->getContext())), 
                                  OldCall);
      // If OldCall dues not return void then replaceAllUsesWith undef.
      // This allows ValueHandlers and custom metadata to adjust itself.
      if (!OldCall->getType()->isVoidTy())
        OldCall->replaceAllUsesWith(UndefValue::get(OldCall->getType()));
      if (isa<CallInst>(OldCall))   // Not worth removing an invoke here.
        return EraseInstFromFunction(*OldCall);
      return 0;
    }

  if (isa<ConstantPointerNull>(Callee) || isa<UndefValue>(Callee)) {
    // This instruction is not reachable, just remove it.  We insert a store to
    // undef so that we know that this code is not reachable, despite the fact
    // that we can't modify the CFG here.
    new StoreInst(ConstantInt::getTrue(Callee->getContext()),
               UndefValue::get(Type::getInt1PtrTy(Callee->getContext())),
                  CS.getInstruction());

    // If CS dues not return void then replaceAllUsesWith undef.
    // This allows ValueHandlers and custom metadata to adjust itself.
    if (!CS.getInstruction()->getType()->isVoidTy())
      CS.getInstruction()->
        replaceAllUsesWith(UndefValue::get(CS.getInstruction()->getType()));

    if (InvokeInst *II = dyn_cast<InvokeInst>(CS.getInstruction())) {
      // Don't break the CFG, insert a dummy cond branch.
      BranchInst::Create(II->getNormalDest(), II->getUnwindDest(),
                         ConstantInt::getTrue(Callee->getContext()), II);
    }
    return EraseInstFromFunction(*CS.getInstruction());
  }

  if (BitCastInst *BC = dyn_cast<BitCastInst>(Callee))
    if (IntrinsicInst *In = dyn_cast<IntrinsicInst>(BC->getOperand(0)))
      if (In->getIntrinsicID() == Intrinsic::init_trampoline)
        return transformCallThroughTrampoline(CS);

  const PointerType *PTy = cast<PointerType>(Callee->getType());
  const FunctionType *FTy = cast<FunctionType>(PTy->getElementType());
  if (FTy->isVarArg()) {
    int ix = FTy->getNumParams() + (isa<InvokeInst>(Callee) ? 3 : 1);
    // See if we can optimize any arguments passed through the varargs area of
    // the call.
    for (CallSite::arg_iterator I = CS.arg_begin()+FTy->getNumParams(),
           E = CS.arg_end(); I != E; ++I, ++ix) {
      CastInst *CI = dyn_cast<CastInst>(*I);
      if (CI && isSafeToEliminateVarargsCast(CS, CI, TD, ix)) {
        *I = CI->getOperand(0);
        Changed = true;
      }
    }
  }

  if (isa<InlineAsm>(Callee) && !CS.doesNotThrow()) {
    // Inline asm calls cannot throw - mark them 'nounwind'.
    CS.setDoesNotThrow();
    Changed = true;
  }

  return Changed ? CS.getInstruction() : 0;
}

// transformConstExprCastCall - If the callee is a constexpr cast of a function,
// attempt to move the cast to the arguments of the call/invoke.
//
bool InstCombiner::transformConstExprCastCall(CallSite CS) {
  if (!isa<ConstantExpr>(CS.getCalledValue())) return false;
  ConstantExpr *CE = cast<ConstantExpr>(CS.getCalledValue());
  if (CE->getOpcode() != Instruction::BitCast || 
      !isa<Function>(CE->getOperand(0)))
    return false;
  Function *Callee = cast<Function>(CE->getOperand(0));
  Instruction *Caller = CS.getInstruction();
  const AttrListPtr &CallerPAL = CS.getAttributes();

  // Okay, this is a cast from a function to a different type.  Unless doing so
  // would cause a type conversion of one of our arguments, change this call to
  // be a direct call with arguments casted to the appropriate types.
  //
  const FunctionType *FT = Callee->getFunctionType();
  const Type *OldRetTy = Caller->getType();
  const Type *NewRetTy = FT->getReturnType();

  if (isa<StructType>(NewRetTy))
    return false; // TODO: Handle multiple return values.

  // Check to see if we are changing the return type...
  if (OldRetTy != NewRetTy) {
    if (Callee->isDeclaration() &&
        // Conversion is ok if changing from one pointer type to another or from
        // a pointer to an integer of the same size.
        !((isa<PointerType>(OldRetTy) || !TD ||
           OldRetTy == TD->getIntPtrType(Caller->getContext())) &&
          (isa<PointerType>(NewRetTy) || !TD ||
           NewRetTy == TD->getIntPtrType(Caller->getContext()))))
      return false;   // Cannot transform this return value.

    if (!Caller->use_empty() &&
        // void -> non-void is handled specially
        !NewRetTy->isVoidTy() && !CastInst::isCastable(NewRetTy, OldRetTy))
      return false;   // Cannot transform this return value.

    if (!CallerPAL.isEmpty() && !Caller->use_empty()) {
      Attributes RAttrs = CallerPAL.getRetAttributes();
      if (RAttrs & Attribute::typeIncompatible(NewRetTy))
        return false;   // Attribute not compatible with transformed value.
    }

    // If the callsite is an invoke instruction, and the return value is used by
    // a PHI node in a successor, we cannot change the return type of the call
    // because there is no place to put the cast instruction (without breaking
    // the critical edge).  Bail out in this case.
    if (!Caller->use_empty())
      if (InvokeInst *II = dyn_cast<InvokeInst>(Caller))
        for (Value::use_iterator UI = II->use_begin(), E = II->use_end();
             UI != E; ++UI)
          if (PHINode *PN = dyn_cast<PHINode>(*UI))
            if (PN->getParent() == II->getNormalDest() ||
                PN->getParent() == II->getUnwindDest())
              return false;
  }

  unsigned NumActualArgs = unsigned(CS.arg_end()-CS.arg_begin());
  unsigned NumCommonArgs = std::min(FT->getNumParams(), NumActualArgs);

  CallSite::arg_iterator AI = CS.arg_begin();
  for (unsigned i = 0, e = NumCommonArgs; i != e; ++i, ++AI) {
    const Type *ParamTy = FT->getParamType(i);
    const Type *ActTy = (*AI)->getType();

    if (!CastInst::isCastable(ActTy, ParamTy))
      return false;   // Cannot transform this parameter value.

    if (CallerPAL.getParamAttributes(i + 1) 
        & Attribute::typeIncompatible(ParamTy))
      return false;   // Attribute not compatible with transformed value.

    // Converting from one pointer type to another or between a pointer and an
    // integer of the same size is safe even if we do not have a body.
    bool isConvertible = ActTy == ParamTy ||
      (TD && ((isa<PointerType>(ParamTy) ||
      ParamTy == TD->getIntPtrType(Caller->getContext())) &&
              (isa<PointerType>(ActTy) ||
              ActTy == TD->getIntPtrType(Caller->getContext()))));
    if (Callee->isDeclaration() && !isConvertible) return false;
  }

  if (FT->getNumParams() < NumActualArgs && !FT->isVarArg() &&
      Callee->isDeclaration())
    return false;   // Do not delete arguments unless we have a function body.

  if (FT->getNumParams() < NumActualArgs && FT->isVarArg() &&
      !CallerPAL.isEmpty())
    // In this case we have more arguments than the new function type, but we
    // won't be dropping them.  Check that these extra arguments have attributes
    // that are compatible with being a vararg call argument.
    for (unsigned i = CallerPAL.getNumSlots(); i; --i) {
      if (CallerPAL.getSlot(i - 1).Index <= FT->getNumParams())
        break;
      Attributes PAttrs = CallerPAL.getSlot(i - 1).Attrs;
      if (PAttrs & Attribute::VarArgsIncompatible)
        return false;
    }

  // Okay, we decided that this is a safe thing to do: go ahead and start
  // inserting cast instructions as necessary...
  std::vector<Value*> Args;
  Args.reserve(NumActualArgs);
  SmallVector<AttributeWithIndex, 8> attrVec;
  attrVec.reserve(NumCommonArgs);

  // Get any return attributes.
  Attributes RAttrs = CallerPAL.getRetAttributes();

  // If the return value is not being used, the type may not be compatible
  // with the existing attributes.  Wipe out any problematic attributes.
  RAttrs &= ~Attribute::typeIncompatible(NewRetTy);

  // Add the new return attributes.
  if (RAttrs)
    attrVec.push_back(AttributeWithIndex::get(0, RAttrs));

  AI = CS.arg_begin();
  for (unsigned i = 0; i != NumCommonArgs; ++i, ++AI) {
    const Type *ParamTy = FT->getParamType(i);
    if ((*AI)->getType() == ParamTy) {
      Args.push_back(*AI);
    } else {
      Instruction::CastOps opcode = CastInst::getCastOpcode(*AI,
          false, ParamTy, false);
      Args.push_back(Builder->CreateCast(opcode, *AI, ParamTy, "tmp"));
    }

    // Add any parameter attributes.
    if (Attributes PAttrs = CallerPAL.getParamAttributes(i + 1))
      attrVec.push_back(AttributeWithIndex::get(i + 1, PAttrs));
  }

  // If the function takes more arguments than the call was taking, add them
  // now.
  for (unsigned i = NumCommonArgs; i != FT->getNumParams(); ++i)
    Args.push_back(Constant::getNullValue(FT->getParamType(i)));

  // If we are removing arguments to the function, emit an obnoxious warning.
  if (FT->getNumParams() < NumActualArgs) {
    if (!FT->isVarArg()) {
      errs() << "WARNING: While resolving call to function '"
             << Callee->getName() << "' arguments were dropped!\n";
    } else {
      // Add all of the arguments in their promoted form to the arg list.
      for (unsigned i = FT->getNumParams(); i != NumActualArgs; ++i, ++AI) {
        const Type *PTy = getPromotedType((*AI)->getType());
        if (PTy != (*AI)->getType()) {
          // Must promote to pass through va_arg area!
          Instruction::CastOps opcode =
            CastInst::getCastOpcode(*AI, false, PTy, false);
          Args.push_back(Builder->CreateCast(opcode, *AI, PTy, "tmp"));
        } else {
          Args.push_back(*AI);
        }

        // Add any parameter attributes.
        if (Attributes PAttrs = CallerPAL.getParamAttributes(i + 1))
          attrVec.push_back(AttributeWithIndex::get(i + 1, PAttrs));
      }
    }
  }

  if (Attributes FnAttrs =  CallerPAL.getFnAttributes())
    attrVec.push_back(AttributeWithIndex::get(~0, FnAttrs));

  if (NewRetTy->isVoidTy())
    Caller->setName("");   // Void type should not have a name.

  const AttrListPtr &NewCallerPAL = AttrListPtr::get(attrVec.begin(),
                                                     attrVec.end());

  Instruction *NC;
  if (InvokeInst *II = dyn_cast<InvokeInst>(Caller)) {
    NC = InvokeInst::Create(Callee, II->getNormalDest(), II->getUnwindDest(),
                            Args.begin(), Args.end(),
                            Caller->getName(), Caller);
    cast<InvokeInst>(NC)->setCallingConv(II->getCallingConv());
    cast<InvokeInst>(NC)->setAttributes(NewCallerPAL);
  } else {
    NC = CallInst::Create(Callee, Args.begin(), Args.end(),
                          Caller->getName(), Caller);
    CallInst *CI = cast<CallInst>(Caller);
    if (CI->isTailCall())
      cast<CallInst>(NC)->setTailCall();
    cast<CallInst>(NC)->setCallingConv(CI->getCallingConv());
    cast<CallInst>(NC)->setAttributes(NewCallerPAL);
  }

  // Insert a cast of the return type as necessary.
  Value *NV = NC;
  if (OldRetTy != NV->getType() && !Caller->use_empty()) {
    if (!NV->getType()->isVoidTy()) {
      Instruction::CastOps opcode = CastInst::getCastOpcode(NC, false, 
                                                            OldRetTy, false);
      NV = NC = CastInst::Create(opcode, NC, OldRetTy, "tmp");

      // If this is an invoke instruction, we should insert it after the first
      // non-phi, instruction in the normal successor block.
      if (InvokeInst *II = dyn_cast<InvokeInst>(Caller)) {
        BasicBlock::iterator I = II->getNormalDest()->getFirstNonPHI();
        InsertNewInstBefore(NC, *I);
      } else {
        // Otherwise, it's a call, just insert cast right after the call instr
        InsertNewInstBefore(NC, *Caller);
      }
      Worklist.AddUsersToWorkList(*Caller);
    } else {
      NV = UndefValue::get(Caller->getType());
    }
  }


  if (!Caller->use_empty())
    Caller->replaceAllUsesWith(NV);
  
  EraseInstFromFunction(*Caller);
  return true;
}

// transformCallThroughTrampoline - Turn a call to a function created by the
// init_trampoline intrinsic into a direct call to the underlying function.
//
Instruction *InstCombiner::transformCallThroughTrampoline(CallSite CS) {
  Value *Callee = CS.getCalledValue();
  const PointerType *PTy = cast<PointerType>(Callee->getType());
  const FunctionType *FTy = cast<FunctionType>(PTy->getElementType());
  const AttrListPtr &Attrs = CS.getAttributes();

  // If the call already has the 'nest' attribute somewhere then give up -
  // otherwise 'nest' would occur twice after splicing in the chain.
  if (Attrs.hasAttrSomewhere(Attribute::Nest))
    return 0;

  IntrinsicInst *Tramp =
    cast<IntrinsicInst>(cast<BitCastInst>(Callee)->getOperand(0));

  Function *NestF = cast<Function>(Tramp->getOperand(2)->stripPointerCasts());
  const PointerType *NestFPTy = cast<PointerType>(NestF->getType());
  const FunctionType *NestFTy = cast<FunctionType>(NestFPTy->getElementType());

  const AttrListPtr &NestAttrs = NestF->getAttributes();
  if (!NestAttrs.isEmpty()) {
    unsigned NestIdx = 1;
    const Type *NestTy = 0;
    Attributes NestAttr = Attribute::None;

    // Look for a parameter marked with the 'nest' attribute.
    for (FunctionType::param_iterator I = NestFTy->param_begin(),
         E = NestFTy->param_end(); I != E; ++NestIdx, ++I)
      if (NestAttrs.paramHasAttr(NestIdx, Attribute::Nest)) {
        // Record the parameter type and any other attributes.
        NestTy = *I;
        NestAttr = NestAttrs.getParamAttributes(NestIdx);
        break;
      }

    if (NestTy) {
      Instruction *Caller = CS.getInstruction();
      std::vector<Value*> NewArgs;
      NewArgs.reserve(unsigned(CS.arg_end()-CS.arg_begin())+1);

      SmallVector<AttributeWithIndex, 8> NewAttrs;
      NewAttrs.reserve(Attrs.getNumSlots() + 1);

      // Insert the nest argument into the call argument list, which may
      // mean appending it.  Likewise for attributes.

      // Add any result attributes.
      if (Attributes Attr = Attrs.getRetAttributes())
        NewAttrs.push_back(AttributeWithIndex::get(0, Attr));

      {
        unsigned Idx = 1;
        CallSite::arg_iterator I = CS.arg_begin(), E = CS.arg_end();
        do {
          if (Idx == NestIdx) {
            // Add the chain argument and attributes.
            Value *NestVal = Tramp->getOperand(3);
            if (NestVal->getType() != NestTy)
              NestVal = new BitCastInst(NestVal, NestTy, "nest", Caller);
            NewArgs.push_back(NestVal);
            NewAttrs.push_back(AttributeWithIndex::get(NestIdx, NestAttr));
          }

          if (I == E)
            break;

          // Add the original argument and attributes.
          NewArgs.push_back(*I);
          if (Attributes Attr = Attrs.getParamAttributes(Idx))
            NewAttrs.push_back
              (AttributeWithIndex::get(Idx + (Idx >= NestIdx), Attr));

          ++Idx, ++I;
        } while (1);
      }

      // Add any function attributes.
      if (Attributes Attr = Attrs.getFnAttributes())
        NewAttrs.push_back(AttributeWithIndex::get(~0, Attr));

      // The trampoline may have been bitcast to a bogus type (FTy).
      // Handle this by synthesizing a new function type, equal to FTy
      // with the chain parameter inserted.

      std::vector<const Type*> NewTypes;
      NewTypes.reserve(FTy->getNumParams()+1);

      // Insert the chain's type into the list of parameter types, which may
      // mean appending it.
      {
        unsigned Idx = 1;
        FunctionType::param_iterator I = FTy->param_begin(),
          E = FTy->param_end();

        do {
          if (Idx == NestIdx)
            // Add the chain's type.
            NewTypes.push_back(NestTy);

          if (I == E)
            break;

          // Add the original type.
          NewTypes.push_back(*I);

          ++Idx, ++I;
        } while (1);
      }

      // Replace the trampoline call with a direct call.  Let the generic
      // code sort out any function type mismatches.
      FunctionType *NewFTy = FunctionType::get(FTy->getReturnType(), NewTypes, 
                                                FTy->isVarArg());
      Constant *NewCallee =
        NestF->getType() == PointerType::getUnqual(NewFTy) ?
        NestF : ConstantExpr::getBitCast(NestF, 
                                         PointerType::getUnqual(NewFTy));
      const AttrListPtr &NewPAL = AttrListPtr::get(NewAttrs.begin(),
                                                   NewAttrs.end());

      Instruction *NewCaller;
      if (InvokeInst *II = dyn_cast<InvokeInst>(Caller)) {
        NewCaller = InvokeInst::Create(NewCallee,
                                       II->getNormalDest(), II->getUnwindDest(),
                                       NewArgs.begin(), NewArgs.end(),
                                       Caller->getName(), Caller);
        cast<InvokeInst>(NewCaller)->setCallingConv(II->getCallingConv());
        cast<InvokeInst>(NewCaller)->setAttributes(NewPAL);
      } else {
        NewCaller = CallInst::Create(NewCallee, NewArgs.begin(), NewArgs.end(),
                                     Caller->getName(), Caller);
        if (cast<CallInst>(Caller)->isTailCall())
          cast<CallInst>(NewCaller)->setTailCall();
        cast<CallInst>(NewCaller)->
          setCallingConv(cast<CallInst>(Caller)->getCallingConv());
        cast<CallInst>(NewCaller)->setAttributes(NewPAL);
      }
      if (!Caller->getType()->isVoidTy())
        Caller->replaceAllUsesWith(NewCaller);
      Caller->eraseFromParent();
      Worklist.Remove(Caller);
      return 0;
    }
  }

  // Replace the trampoline call with a direct call.  Since there is no 'nest'
  // parameter, there is no need to adjust the argument list.  Let the generic
  // code sort out any function type mismatches.
  Constant *NewCallee =
    NestF->getType() == PTy ? NestF : 
                              ConstantExpr::getBitCast(NestF, PTy);
  CS.setCalledFunction(NewCallee);
  return CS.getInstruction();
}

