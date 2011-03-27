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
#include "llvm/Transforms/Utils/BuildLibCalls.h"
#include "llvm/Transforms/Utils/Local.h"
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


Instruction *InstCombiner::SimplifyMemTransfer(MemIntrinsic *MI) {
  unsigned DstAlign = getKnownAlignment(MI->getArgOperand(0), TD);
  unsigned SrcAlign = getKnownAlignment(MI->getArgOperand(1), TD);
  unsigned MinAlign = std::min(DstAlign, SrcAlign);
  unsigned CopyAlign = MI->getAlignment();

  if (CopyAlign < MinAlign) {
    MI->setAlignment(ConstantInt::get(MI->getAlignmentType(), 
                                             MinAlign, false));
    return MI;
  }
  
  // If MemCpyInst length is 1/2/4/8 bytes then replace memcpy with
  // load/store.
  ConstantInt *MemOpLength = dyn_cast<ConstantInt>(MI->getArgOperand(2));
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
  unsigned SrcAddrSp =
    cast<PointerType>(MI->getArgOperand(1)->getType())->getAddressSpace();
  unsigned DstAddrSp =
    cast<PointerType>(MI->getArgOperand(0)->getType())->getAddressSpace();

  const IntegerType* IntType = IntegerType::get(MI->getContext(), Size<<3);
  Type *NewSrcPtrTy = PointerType::get(IntType, SrcAddrSp);
  Type *NewDstPtrTy = PointerType::get(IntType, DstAddrSp);
  
  // Memcpy forces the use of i8* for the source and destination.  That means
  // that if you're using memcpy to move one double around, you'll get a cast
  // from double* to i8*.  We'd much rather use a double load+store rather than
  // an i64 load+store, here because this improves the odds that the source or
  // dest address will be promotable.  See if we can find a better type than the
  // integer datatype.
  Value *StrippedDest = MI->getArgOperand(0)->stripPointerCasts();
  if (StrippedDest != MI->getArgOperand(0)) {
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
      
      if (SrcETy->isSingleValueType()) {
        NewSrcPtrTy = PointerType::get(SrcETy, SrcAddrSp);
        NewDstPtrTy = PointerType::get(SrcETy, DstAddrSp);
      }
    }
  }
  
  
  // If the memcpy/memmove provides better alignment info than we can
  // infer, use it.
  SrcAlign = std::max(SrcAlign, CopyAlign);
  DstAlign = std::max(DstAlign, CopyAlign);
  
  Value *Src = Builder->CreateBitCast(MI->getArgOperand(1), NewSrcPtrTy);
  Value *Dest = Builder->CreateBitCast(MI->getArgOperand(0), NewDstPtrTy);
  Instruction *L = new LoadInst(Src, "tmp", MI->isVolatile(), SrcAlign);
  InsertNewInstBefore(L, *MI);
  InsertNewInstBefore(new StoreInst(L, Dest, MI->isVolatile(), DstAlign),
                      *MI);

  // Set the size of the copy to 0, it will be deleted on the next iteration.
  MI->setArgOperand(2, Constant::getNullValue(MemOpLength->getType()));
  return MI;
}

Instruction *InstCombiner::SimplifyMemSet(MemSetInst *MI) {
  unsigned Alignment = getKnownAlignment(MI->getDest(), TD);
  if (MI->getAlignment() < Alignment) {
    MI->setAlignment(ConstantInt::get(MI->getAlignmentType(),
                                             Alignment, false));
    return MI;
  }
  
  // Extract the length and alignment and fill if they are constant.
  ConstantInt *LenC = dyn_cast<ConstantInt>(MI->getLength());
  ConstantInt *FillC = dyn_cast<ConstantInt>(MI->getValue());
  if (!LenC || !FillC || !FillC->getType()->isIntegerTy(8))
    return 0;
  uint64_t Len = LenC->getZExtValue();
  Alignment = MI->getAlignment();
  
  // If the length is zero, this is a no-op
  if (Len == 0) return MI; // memset(d,c,0,a) -> noop
  
  // memset(s,c,n) -> store s, c (for n=1,2,4,8)
  if (Len <= 8 && isPowerOf2_32((uint32_t)Len)) {
    const Type *ITy = IntegerType::get(MI->getContext(), Len*8);  // n=1 -> i8.
    
    Value *Dest = MI->getDest();
    unsigned DstAddrSp = cast<PointerType>(Dest->getType())->getAddressSpace();
    Type *NewDstPtrTy = PointerType::get(ITy, DstAddrSp);
    Dest = Builder->CreateBitCast(Dest, NewDstPtrTy);

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
  if (isMalloc(&CI))
    return visitMalloc(CI);

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
      if (NumBytes->isNullValue())
        return EraseInstFromFunction(CI);

      if (ConstantInt *CI = dyn_cast<ConstantInt>(NumBytes))
        if (CI->getZExtValue() == 1) {
          // Replace the instruction with just byte operations.  We would
          // transform other cases to loads/stores, but we don't know if
          // alignment is sufficient.
        }
    }
    
    // No other transformations apply to volatile transfers.
    if (MI->isVolatile())
      return 0;

    // If we have a memmove and the source operation is a constant global,
    // then the source and dest pointers can't alias, so we can change this
    // into a call to memcpy.
    if (MemMoveInst *MMI = dyn_cast<MemMoveInst>(MI)) {
      if (GlobalVariable *GVSrc = dyn_cast<GlobalVariable>(MMI->getSource()))
        if (GVSrc->isConstant()) {
          Module *M = CI.getParent()->getParent()->getParent();
          Intrinsic::ID MemCpyID = Intrinsic::memcpy;
          const Type *Tys[3] = { CI.getArgOperand(0)->getType(),
                                 CI.getArgOperand(1)->getType(),
                                 CI.getArgOperand(2)->getType() };
          CI.setCalledFunction(Intrinsic::getDeclaration(M, MemCpyID, Tys, 3));
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
  case Intrinsic::objectsize: {
    // We need target data for just about everything so depend on it.
    if (!TD) break;
    
    const Type *ReturnTy = CI.getType();
    uint64_t DontKnow = II->getArgOperand(1) == Builder->getTrue() ? 0 : -1ULL;

    // Get to the real allocated thing and offset as fast as possible.
    Value *Op1 = II->getArgOperand(0)->stripPointerCasts();

    uint64_t Offset = 0;
    uint64_t Size = -1ULL;

    // Try to look through constant GEPs.
    if (GEPOperator *GEP = dyn_cast<GEPOperator>(Op1)) {
      if (!GEP->hasAllConstantIndices()) break;

      // Get the current byte offset into the thing. Use the original
      // operand in case we're looking through a bitcast.
      SmallVector<Value*, 8> Ops(GEP->idx_begin(), GEP->idx_end());
      Offset = TD->getIndexedOffset(GEP->getPointerOperandType(),
                                    Ops.data(), Ops.size());

      Op1 = GEP->getPointerOperand()->stripPointerCasts();

      // Make sure we're not a constant offset from an external
      // global.
      if (GlobalVariable *GV = dyn_cast<GlobalVariable>(Op1))
        if (!GV->hasDefinitiveInitializer()) break;
    }

    // If we've stripped down to a single global variable that we
    // can know the size of then just return that.
    if (GlobalVariable *GV = dyn_cast<GlobalVariable>(Op1)) {
      if (GV->hasDefinitiveInitializer()) {
        Constant *C = GV->getInitializer();
        Size = TD->getTypeAllocSize(C->getType());
      } else {
        // Can't determine size of the GV.
        Constant *RetVal = ConstantInt::get(ReturnTy, DontKnow);
        return ReplaceInstUsesWith(CI, RetVal);
      }
    } else if (AllocaInst *AI = dyn_cast<AllocaInst>(Op1)) {
      // Get alloca size.
      if (AI->getAllocatedType()->isSized()) {
        Size = TD->getTypeAllocSize(AI->getAllocatedType());
        if (AI->isArrayAllocation()) {
          const ConstantInt *C = dyn_cast<ConstantInt>(AI->getArraySize());
          if (!C) break;
          Size *= C->getZExtValue();
        }
      }
    } else if (CallInst *MI = extractMallocCall(Op1)) {
      // Get allocation size.
      const Type* MallocType = getMallocAllocatedType(MI);
      if (MallocType && MallocType->isSized())
        if (Value *NElems = getMallocArraySize(MI, TD, true))
          if (ConstantInt *NElements = dyn_cast<ConstantInt>(NElems))
            Size = NElements->getZExtValue() * TD->getTypeAllocSize(MallocType);
    }

    // Do not return "I don't know" here. Later optimization passes could
    // make it possible to evaluate objectsize to a constant.
    if (Size == -1ULL)
      break;

    if (Size < Offset) {
      // Out of bound reference? Negative index normalized to large
      // index? Just return "I don't know".
      return ReplaceInstUsesWith(CI, ConstantInt::get(ReturnTy, DontKnow));
    }
    return ReplaceInstUsesWith(CI, ConstantInt::get(ReturnTy, Size-Offset));
  }
  case Intrinsic::bswap:
    // bswap(bswap(x)) -> x
    if (IntrinsicInst *Operand = dyn_cast<IntrinsicInst>(II->getArgOperand(0)))
      if (Operand->getIntrinsicID() == Intrinsic::bswap)
        return ReplaceInstUsesWith(CI, Operand->getArgOperand(0));
      
    // bswap(trunc(bswap(x))) -> trunc(lshr(x, c))
    if (TruncInst *TI = dyn_cast<TruncInst>(II->getArgOperand(0))) {
      if (IntrinsicInst *Operand = dyn_cast<IntrinsicInst>(TI->getOperand(0)))
        if (Operand->getIntrinsicID() == Intrinsic::bswap) {
          unsigned C = Operand->getType()->getPrimitiveSizeInBits() -
                       TI->getType()->getPrimitiveSizeInBits();
          Value *CV = ConstantInt::get(Operand->getType(), C);
          Value *V = Builder->CreateLShr(Operand->getArgOperand(0), CV);
          return new TruncInst(V, TI->getType());
        }
    }
      
    break;
  case Intrinsic::powi:
    if (ConstantInt *Power = dyn_cast<ConstantInt>(II->getArgOperand(1))) {
      // powi(x, 0) -> 1.0
      if (Power->isZero())
        return ReplaceInstUsesWith(CI, ConstantFP::get(CI.getType(), 1.0));
      // powi(x, 1) -> x
      if (Power->isOne())
        return ReplaceInstUsesWith(CI, II->getArgOperand(0));
      // powi(x, -1) -> 1/x
      if (Power->isAllOnesValue())
        return BinaryOperator::CreateFDiv(ConstantFP::get(CI.getType(), 1.0),
                                          II->getArgOperand(0));
    }
    break;
  case Intrinsic::cttz: {
    // If all bits below the first known one are known zero,
    // this value is constant.
    const IntegerType *IT = cast<IntegerType>(II->getArgOperand(0)->getType());
    uint32_t BitWidth = IT->getBitWidth();
    APInt KnownZero(BitWidth, 0);
    APInt KnownOne(BitWidth, 0);
    ComputeMaskedBits(II->getArgOperand(0), APInt::getAllOnesValue(BitWidth),
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
    const IntegerType *IT = cast<IntegerType>(II->getArgOperand(0)->getType());
    uint32_t BitWidth = IT->getBitWidth();
    APInt KnownZero(BitWidth, 0);
    APInt KnownOne(BitWidth, 0);
    ComputeMaskedBits(II->getArgOperand(0), APInt::getAllOnesValue(BitWidth),
                      KnownZero, KnownOne);
    unsigned LeadingZeros = KnownOne.countLeadingZeros();
    APInt Mask(APInt::getHighBitsSet(BitWidth, LeadingZeros));
    if ((Mask & KnownZero) == Mask)
      return ReplaceInstUsesWith(CI, ConstantInt::get(IT,
                                 APInt(BitWidth, LeadingZeros)));
    
    }
    break;
  case Intrinsic::uadd_with_overflow: {
    Value *LHS = II->getArgOperand(0), *RHS = II->getArgOperand(1);
    const IntegerType *IT = cast<IntegerType>(II->getArgOperand(0)->getType());
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
    if (isa<Constant>(II->getArgOperand(0)) &&
        !isa<Constant>(II->getArgOperand(1))) {
      Value *LHS = II->getArgOperand(0);
      II->setArgOperand(0, II->getArgOperand(1));
      II->setArgOperand(1, LHS);
      return II;
    }

    // X + undef -> undef
    if (isa<UndefValue>(II->getArgOperand(1)))
      return ReplaceInstUsesWith(CI, UndefValue::get(II->getType()));
      
    if (ConstantInt *RHS = dyn_cast<ConstantInt>(II->getArgOperand(1))) {
      // X + 0 -> {X, false}
      if (RHS->isZero()) {
        Constant *V[] = {
          UndefValue::get(II->getArgOperand(0)->getType()),
          ConstantInt::getFalse(II->getContext())
        };
        Constant *Struct = ConstantStruct::get(II->getContext(), V, 2, false);
        return InsertValueInst::Create(Struct, II->getArgOperand(0), 0);
      }
    }
    break;
  case Intrinsic::usub_with_overflow:
  case Intrinsic::ssub_with_overflow:
    // undef - X -> undef
    // X - undef -> undef
    if (isa<UndefValue>(II->getArgOperand(0)) ||
        isa<UndefValue>(II->getArgOperand(1)))
      return ReplaceInstUsesWith(CI, UndefValue::get(II->getType()));
      
    if (ConstantInt *RHS = dyn_cast<ConstantInt>(II->getArgOperand(1))) {
      // X - 0 -> {X, false}
      if (RHS->isZero()) {
        Constant *V[] = {
          UndefValue::get(II->getArgOperand(0)->getType()),
          ConstantInt::getFalse(II->getContext())
        };
        Constant *Struct = ConstantStruct::get(II->getContext(), V, 2, false);
        return InsertValueInst::Create(Struct, II->getArgOperand(0), 0);
      }
    }
    break;
  case Intrinsic::umul_with_overflow: {
    Value *LHS = II->getArgOperand(0), *RHS = II->getArgOperand(1);
    unsigned BitWidth = cast<IntegerType>(LHS->getType())->getBitWidth();
    APInt Mask = APInt::getAllOnesValue(BitWidth);

    APInt LHSKnownZero(BitWidth, 0);
    APInt LHSKnownOne(BitWidth, 0);
    ComputeMaskedBits(LHS, Mask, LHSKnownZero, LHSKnownOne);
    APInt RHSKnownZero(BitWidth, 0);
    APInt RHSKnownOne(BitWidth, 0);
    ComputeMaskedBits(RHS, Mask, RHSKnownZero, RHSKnownOne);

    // Get the largest possible values for each operand.
    APInt LHSMax = ~LHSKnownZero;
    APInt RHSMax = ~RHSKnownZero;

    // If multiplying the maximum values does not overflow then we can turn
    // this into a plain NUW mul.
    bool Overflow;
    LHSMax.umul_ov(RHSMax, Overflow);
    if (!Overflow) {
      Value *Mul = Builder->CreateNUWMul(LHS, RHS, "umul_with_overflow");
      Constant *V[] = {
        UndefValue::get(LHS->getType()),
        Builder->getFalse()
      };
      Constant *Struct = ConstantStruct::get(II->getContext(), V, 2, false);
      return InsertValueInst::Create(Struct, Mul, 0);
    }
  } // FALL THROUGH
  case Intrinsic::smul_with_overflow:
    // Canonicalize constants into the RHS.
    if (isa<Constant>(II->getArgOperand(0)) &&
        !isa<Constant>(II->getArgOperand(1))) {
      Value *LHS = II->getArgOperand(0);
      II->setArgOperand(0, II->getArgOperand(1));
      II->setArgOperand(1, LHS);
      return II;
    }

    // X * undef -> undef
    if (isa<UndefValue>(II->getArgOperand(1)))
      return ReplaceInstUsesWith(CI, UndefValue::get(II->getType()));
      
    if (ConstantInt *RHSI = dyn_cast<ConstantInt>(II->getArgOperand(1))) {
      // X*0 -> {0, false}
      if (RHSI->isZero())
        return ReplaceInstUsesWith(CI, Constant::getNullValue(II->getType()));
      
      // X * 1 -> {X, false}
      if (RHSI->equalsInt(1)) {
        Constant *V[] = {
          UndefValue::get(II->getArgOperand(0)->getType()),
          ConstantInt::getFalse(II->getContext())
        };
        Constant *Struct = ConstantStruct::get(II->getContext(), V, 2, false);
        return InsertValueInst::Create(Struct, II->getArgOperand(0), 0);
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
    if (getOrEnforceKnownAlignment(II->getArgOperand(0), 16, TD) >= 16) {
      Value *Ptr = Builder->CreateBitCast(II->getArgOperand(0),
                                         PointerType::getUnqual(II->getType()));
      return new LoadInst(Ptr);
    }
    break;
  case Intrinsic::ppc_altivec_stvx:
  case Intrinsic::ppc_altivec_stvxl:
    // Turn stvx -> store if the pointer is known aligned.
    if (getOrEnforceKnownAlignment(II->getArgOperand(1), 16, TD) >= 16) {
      const Type *OpPtrTy = 
        PointerType::getUnqual(II->getArgOperand(0)->getType());
      Value *Ptr = Builder->CreateBitCast(II->getArgOperand(1), OpPtrTy);
      return new StoreInst(II->getArgOperand(0), Ptr);
    }
    break;
  case Intrinsic::x86_sse_storeu_ps:
  case Intrinsic::x86_sse2_storeu_pd:
  case Intrinsic::x86_sse2_storeu_dq:
    // Turn X86 storeu -> store if the pointer is known aligned.
    if (getOrEnforceKnownAlignment(II->getArgOperand(0), 16, TD) >= 16) {
      const Type *OpPtrTy = 
        PointerType::getUnqual(II->getArgOperand(1)->getType());
      Value *Ptr = Builder->CreateBitCast(II->getArgOperand(0), OpPtrTy);
      return new StoreInst(II->getArgOperand(1), Ptr);
    }
    break;

  case Intrinsic::x86_sse_cvtss2si:
  case Intrinsic::x86_sse_cvtss2si64:
  case Intrinsic::x86_sse_cvttss2si:
  case Intrinsic::x86_sse_cvttss2si64:
  case Intrinsic::x86_sse2_cvtsd2si:
  case Intrinsic::x86_sse2_cvtsd2si64:
  case Intrinsic::x86_sse2_cvttsd2si:
  case Intrinsic::x86_sse2_cvttsd2si64: {
    // These intrinsics only demand the 0th element of their input vectors. If
    // we can simplify the input based on that, do so now.
    unsigned VWidth =
      cast<VectorType>(II->getArgOperand(0)->getType())->getNumElements();
    APInt DemandedElts(VWidth, 1);
    APInt UndefElts(VWidth, 0);
    if (Value *V = SimplifyDemandedVectorElts(II->getArgOperand(0),
                                              DemandedElts, UndefElts)) {
      II->setArgOperand(0, V);
      return II;
    }
    break;
  }

  case Intrinsic::ppc_altivec_vperm:
    // Turn vperm(V1,V2,mask) -> shuffle(V1,V2,mask) if mask is a constant.
    if (ConstantVector *Mask = dyn_cast<ConstantVector>(II->getArgOperand(2))) {
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
        Value *Op0 = Builder->CreateBitCast(II->getArgOperand(0),
                                            Mask->getType());
        Value *Op1 = Builder->CreateBitCast(II->getArgOperand(1),
                                            Mask->getType());
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

  case Intrinsic::arm_neon_vld1:
  case Intrinsic::arm_neon_vld2:
  case Intrinsic::arm_neon_vld3:
  case Intrinsic::arm_neon_vld4:
  case Intrinsic::arm_neon_vld2lane:
  case Intrinsic::arm_neon_vld3lane:
  case Intrinsic::arm_neon_vld4lane:
  case Intrinsic::arm_neon_vst1:
  case Intrinsic::arm_neon_vst2:
  case Intrinsic::arm_neon_vst3:
  case Intrinsic::arm_neon_vst4:
  case Intrinsic::arm_neon_vst2lane:
  case Intrinsic::arm_neon_vst3lane:
  case Intrinsic::arm_neon_vst4lane: {
    unsigned MemAlign = getKnownAlignment(II->getArgOperand(0), TD);
    unsigned AlignArg = II->getNumArgOperands() - 1;
    ConstantInt *IntrAlign = dyn_cast<ConstantInt>(II->getArgOperand(AlignArg));
    if (IntrAlign && IntrAlign->getZExtValue() < MemAlign) {
      II->setArgOperand(AlignArg,
                        ConstantInt::get(Type::getInt32Ty(II->getContext()),
                                         MemAlign, false));
      return II;
    }
    break;
  }

  case Intrinsic::stackrestore: {
    // If the save is right next to the restore, remove the restore.  This can
    // happen when variable allocas are DCE'd.
    if (IntrinsicInst *SS = dyn_cast<IntrinsicInst>(II->getArgOperand(0))) {
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

namespace {
class InstCombineFortifiedLibCalls : public SimplifyFortifiedLibCalls {
  InstCombiner *IC;
protected:
  void replaceCall(Value *With) {
    NewInstruction = IC->ReplaceInstUsesWith(*CI, With);
  }
  bool isFoldable(unsigned SizeCIOp, unsigned SizeArgOp, bool isString) const {
    if (CI->getArgOperand(SizeCIOp) == CI->getArgOperand(SizeArgOp))
      return true;
    if (ConstantInt *SizeCI =
                           dyn_cast<ConstantInt>(CI->getArgOperand(SizeCIOp))) {
      if (SizeCI->isAllOnesValue())
        return true;
      if (isString) {
        uint64_t Len = GetStringLength(CI->getArgOperand(SizeArgOp));
        // If the length is 0 we don't know how long it is and so we can't
        // remove the check.
        if (Len == 0) return false;
        return SizeCI->getZExtValue() >= Len;
      }
      if (ConstantInt *Arg = dyn_cast<ConstantInt>(
                                                  CI->getArgOperand(SizeArgOp)))
        return SizeCI->getZExtValue() >= Arg->getZExtValue();
    }
    return false;
  }
public:
  InstCombineFortifiedLibCalls(InstCombiner *IC) : IC(IC), NewInstruction(0) { }
  Instruction *NewInstruction;
};
} // end anonymous namespace

// Try to fold some different type of calls here.
// Currently we're only working with the checking functions, memcpy_chk, 
// mempcpy_chk, memmove_chk, memset_chk, strcpy_chk, stpcpy_chk, strncpy_chk,
// strcat_chk and strncat_chk.
Instruction *InstCombiner::tryOptimizeCall(CallInst *CI, const TargetData *TD) {
  if (CI->getCalledFunction() == 0) return 0;

  InstCombineFortifiedLibCalls Simplifier(this);
  Simplifier.fold(CI, TD);
  return Simplifier.NewInstruction;
}

// visitCallSite - Improvements for call and invoke instructions.
//
Instruction *InstCombiner::visitCallSite(CallSite CS) {
  bool Changed = false;

  // If the callee is a pointer to a function, attempt to move any casts to the
  // arguments of the call/invoke.
  Value *Callee = CS.getCalledValue();
  if (!isa<Function>(Callee) && transformConstExprCastCall(CS))
    return 0;

  if (Function *CalleeF = dyn_cast<Function>(Callee))
    // If the call and callee calling conventions don't match, this call must
    // be unreachable, as the call is undefined.
    if (CalleeF->getCallingConv() != CS.getCallingConv() &&
        // Only do this for calls to a function with a body.  A prototype may
        // not actually end up matching the implementation's calling conv for a
        // variety of reasons (e.g. it may be written in assembly).
        !CalleeF->isDeclaration()) {
      Instruction *OldCall = CS.getInstruction();
      new StoreInst(ConstantInt::getTrue(Callee->getContext()),
                UndefValue::get(Type::getInt1PtrTy(Callee->getContext())), 
                                  OldCall);
      // If OldCall dues not return void then replaceAllUsesWith undef.
      // This allows ValueHandlers and custom metadata to adjust itself.
      if (!OldCall->getType()->isVoidTy())
        OldCall->replaceAllUsesWith(UndefValue::get(OldCall->getType()));
      if (isa<CallInst>(OldCall))
        return EraseInstFromFunction(*OldCall);
      
      // We cannot remove an invoke, because it would change the CFG, just
      // change the callee to a null pointer.
      cast<InvokeInst>(OldCall)->setCalledFunction(
                                    Constant::getNullValue(CalleeF->getType()));
      return 0;
    }

  if (isa<ConstantPointerNull>(Callee) || isa<UndefValue>(Callee)) {
    // This instruction is not reachable, just remove it.  We insert a store to
    // undef so that we know that this code is not reachable, despite the fact
    // that we can't modify the CFG here.
    new StoreInst(ConstantInt::getTrue(Callee->getContext()),
               UndefValue::get(Type::getInt1PtrTy(Callee->getContext())),
                  CS.getInstruction());

    // If CS does not return void then replaceAllUsesWith undef.
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

  // Try to optimize the call if possible, we require TargetData for most of
  // this.  None of these calls are seen as possibly dead so go ahead and
  // delete the instruction now.
  if (CallInst *CI = dyn_cast<CallInst>(CS.getInstruction())) {
    Instruction *I = tryOptimizeCall(CI, TD);
    // If we changed something return the result, etc. Otherwise let
    // the fallthrough check.
    if (I) return EraseInstFromFunction(*I);
  }

  return Changed ? CS.getInstruction() : 0;
}

// transformConstExprCastCall - If the callee is a constexpr cast of a function,
// attempt to move the cast to the arguments of the call/invoke.
//
bool InstCombiner::transformConstExprCastCall(CallSite CS) {
  Function *Callee =
    dyn_cast<Function>(CS.getCalledValue()->stripPointerCasts());
  if (Callee == 0)
    return false;
  Instruction *Caller = CS.getInstruction();
  const AttrListPtr &CallerPAL = CS.getAttributes();

  // Okay, this is a cast from a function to a different type.  Unless doing so
  // would cause a type conversion of one of our arguments, change this call to
  // be a direct call with arguments casted to the appropriate types.
  //
  const FunctionType *FT = Callee->getFunctionType();
  const Type *OldRetTy = Caller->getType();
  const Type *NewRetTy = FT->getReturnType();

  if (NewRetTy->isStructTy())
    return false; // TODO: Handle multiple return values.

  // Check to see if we are changing the return type...
  if (OldRetTy != NewRetTy) {
    if (Callee->isDeclaration() &&
        // Conversion is ok if changing from one pointer type to another or from
        // a pointer to an integer of the same size.
        !((OldRetTy->isPointerTy() || !TD ||
           OldRetTy == TD->getIntPtrType(Caller->getContext())) &&
          (NewRetTy->isPointerTy() || !TD ||
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

    unsigned Attrs = CallerPAL.getParamAttributes(i + 1);
    if (Attrs & Attribute::typeIncompatible(ParamTy))
      return false;   // Attribute not compatible with transformed value.
    
    // If the parameter is passed as a byval argument, then we have to have a
    // sized type and the sized type has to have the same size as the old type.
    if (ParamTy != ActTy && (Attrs & Attribute::ByVal)) {
      const PointerType *ParamPTy = dyn_cast<PointerType>(ParamTy);
      if (ParamPTy == 0 || !ParamPTy->getElementType()->isSized() || TD == 0)
        return false;
      
      const Type *CurElTy = cast<PointerType>(ActTy)->getElementType();
      if (TD->getTypeAllocSize(CurElTy) !=
          TD->getTypeAllocSize(ParamPTy->getElementType()))
        return false;
    }

    // Converting from one pointer type to another or between a pointer and an
    // integer of the same size is safe even if we do not have a body.
    bool isConvertible = ActTy == ParamTy ||
      (TD && ((ParamTy->isPointerTy() ||
      ParamTy == TD->getIntPtrType(Caller->getContext())) &&
              (ActTy->isPointerTy() ||
              ActTy == TD->getIntPtrType(Caller->getContext()))));
    if (Callee->isDeclaration() && !isConvertible) return false;
  }

  if (Callee->isDeclaration()) {
    // Do not delete arguments unless we have a function body.
    if (FT->getNumParams() < NumActualArgs && !FT->isVarArg())
      return false;

    // If the callee is just a declaration, don't change the varargsness of the
    // call.  We don't want to introduce a varargs call where one doesn't
    // already exist.
    const PointerType *APTy = cast<PointerType>(CS.getCalledValue()->getType());
    if (FT->isVarArg()!=cast<FunctionType>(APTy->getElementType())->isVarArg())
      return false;
  }
      
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
  // inserting cast instructions as necessary.
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
      Instruction::CastOps opcode =
        CastInst::getCastOpcode(NC, false, OldRetTy, false);
      NV = NC = CastInst::Create(opcode, NC, OldRetTy, "tmp");

      // If this is an invoke instruction, we should insert it after the first
      // non-phi, instruction in the normal successor block.
      if (InvokeInst *II = dyn_cast<InvokeInst>(Caller)) {
        BasicBlock::iterator I = II->getNormalDest()->getFirstNonPHI();
        InsertNewInstBefore(NC, *I);
      } else {
        // Otherwise, it's a call, just insert cast right after the call.
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

  Function *NestF =cast<Function>(Tramp->getArgOperand(1)->stripPointerCasts());
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
            Value *NestVal = Tramp->getArgOperand(2);
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

