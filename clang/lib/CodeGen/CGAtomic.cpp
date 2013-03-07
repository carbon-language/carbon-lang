//===--- CGAtomic.cpp - Emit LLVM IR for atomic operations ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the code for emitting atomic operations.
//
//===----------------------------------------------------------------------===//

#include "CodeGenFunction.h"
#include "CGCall.h"
#include "CodeGenModule.h"
#include "clang/AST/ASTContext.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Intrinsics.h"

using namespace clang;
using namespace CodeGen;

static void
EmitAtomicOp(CodeGenFunction &CGF, AtomicExpr *E, llvm::Value *Dest,
             llvm::Value *Ptr, llvm::Value *Val1, llvm::Value *Val2,
             uint64_t Size, unsigned Align, llvm::AtomicOrdering Order) {
  llvm::AtomicRMWInst::BinOp Op = llvm::AtomicRMWInst::Add;
  llvm::Instruction::BinaryOps PostOp = (llvm::Instruction::BinaryOps)0;

  switch (E->getOp()) {
  case AtomicExpr::AO__c11_atomic_init:
    llvm_unreachable("Already handled!");

  case AtomicExpr::AO__c11_atomic_compare_exchange_strong:
  case AtomicExpr::AO__c11_atomic_compare_exchange_weak:
  case AtomicExpr::AO__atomic_compare_exchange:
  case AtomicExpr::AO__atomic_compare_exchange_n: {
    // Note that cmpxchg only supports specifying one ordering and
    // doesn't support weak cmpxchg, at least at the moment.
    llvm::LoadInst *LoadVal1 = CGF.Builder.CreateLoad(Val1);
    LoadVal1->setAlignment(Align);
    llvm::LoadInst *LoadVal2 = CGF.Builder.CreateLoad(Val2);
    LoadVal2->setAlignment(Align);
    llvm::AtomicCmpXchgInst *CXI =
        CGF.Builder.CreateAtomicCmpXchg(Ptr, LoadVal1, LoadVal2, Order);
    CXI->setVolatile(E->isVolatile());
    llvm::StoreInst *StoreVal1 = CGF.Builder.CreateStore(CXI, Val1);
    StoreVal1->setAlignment(Align);
    llvm::Value *Cmp = CGF.Builder.CreateICmpEQ(CXI, LoadVal1);
    CGF.EmitStoreOfScalar(Cmp, CGF.MakeAddrLValue(Dest, E->getType()));
    return;
  }

  case AtomicExpr::AO__c11_atomic_load:
  case AtomicExpr::AO__atomic_load_n:
  case AtomicExpr::AO__atomic_load: {
    llvm::LoadInst *Load = CGF.Builder.CreateLoad(Ptr);
    Load->setAtomic(Order);
    Load->setAlignment(Size);
    Load->setVolatile(E->isVolatile());
    llvm::StoreInst *StoreDest = CGF.Builder.CreateStore(Load, Dest);
    StoreDest->setAlignment(Align);
    return;
  }

  case AtomicExpr::AO__c11_atomic_store:
  case AtomicExpr::AO__atomic_store:
  case AtomicExpr::AO__atomic_store_n: {
    assert(!Dest && "Store does not return a value");
    llvm::LoadInst *LoadVal1 = CGF.Builder.CreateLoad(Val1);
    LoadVal1->setAlignment(Align);
    llvm::StoreInst *Store = CGF.Builder.CreateStore(LoadVal1, Ptr);
    Store->setAtomic(Order);
    Store->setAlignment(Size);
    Store->setVolatile(E->isVolatile());
    return;
  }

  case AtomicExpr::AO__c11_atomic_exchange:
  case AtomicExpr::AO__atomic_exchange_n:
  case AtomicExpr::AO__atomic_exchange:
    Op = llvm::AtomicRMWInst::Xchg;
    break;

  case AtomicExpr::AO__atomic_add_fetch:
    PostOp = llvm::Instruction::Add;
    // Fall through.
  case AtomicExpr::AO__c11_atomic_fetch_add:
  case AtomicExpr::AO__atomic_fetch_add:
    Op = llvm::AtomicRMWInst::Add;
    break;

  case AtomicExpr::AO__atomic_sub_fetch:
    PostOp = llvm::Instruction::Sub;
    // Fall through.
  case AtomicExpr::AO__c11_atomic_fetch_sub:
  case AtomicExpr::AO__atomic_fetch_sub:
    Op = llvm::AtomicRMWInst::Sub;
    break;

  case AtomicExpr::AO__atomic_and_fetch:
    PostOp = llvm::Instruction::And;
    // Fall through.
  case AtomicExpr::AO__c11_atomic_fetch_and:
  case AtomicExpr::AO__atomic_fetch_and:
    Op = llvm::AtomicRMWInst::And;
    break;

  case AtomicExpr::AO__atomic_or_fetch:
    PostOp = llvm::Instruction::Or;
    // Fall through.
  case AtomicExpr::AO__c11_atomic_fetch_or:
  case AtomicExpr::AO__atomic_fetch_or:
    Op = llvm::AtomicRMWInst::Or;
    break;

  case AtomicExpr::AO__atomic_xor_fetch:
    PostOp = llvm::Instruction::Xor;
    // Fall through.
  case AtomicExpr::AO__c11_atomic_fetch_xor:
  case AtomicExpr::AO__atomic_fetch_xor:
    Op = llvm::AtomicRMWInst::Xor;
    break;

  case AtomicExpr::AO__atomic_nand_fetch:
    PostOp = llvm::Instruction::And;
    // Fall through.
  case AtomicExpr::AO__atomic_fetch_nand:
    Op = llvm::AtomicRMWInst::Nand;
    break;
  }

  llvm::LoadInst *LoadVal1 = CGF.Builder.CreateLoad(Val1);
  LoadVal1->setAlignment(Align);
  llvm::AtomicRMWInst *RMWI =
      CGF.Builder.CreateAtomicRMW(Op, Ptr, LoadVal1, Order);
  RMWI->setVolatile(E->isVolatile());

  // For __atomic_*_fetch operations, perform the operation again to
  // determine the value which was written.
  llvm::Value *Result = RMWI;
  if (PostOp)
    Result = CGF.Builder.CreateBinOp(PostOp, RMWI, LoadVal1);
  if (E->getOp() == AtomicExpr::AO__atomic_nand_fetch)
    Result = CGF.Builder.CreateNot(Result);
  llvm::StoreInst *StoreDest = CGF.Builder.CreateStore(Result, Dest);
  StoreDest->setAlignment(Align);
}

// This function emits any expression (scalar, complex, or aggregate)
// into a temporary alloca.
static llvm::Value *
EmitValToTemp(CodeGenFunction &CGF, Expr *E) {
  llvm::Value *DeclPtr = CGF.CreateMemTemp(E->getType(), ".atomictmp");
  CGF.EmitAnyExprToMem(E, DeclPtr, E->getType().getQualifiers(),
                       /*Init*/ true);
  return DeclPtr;
}

RValue CodeGenFunction::EmitAtomicExpr(AtomicExpr *E, llvm::Value *Dest) {
  QualType AtomicTy = E->getPtr()->getType()->getPointeeType();
  QualType MemTy = AtomicTy;
  if (const AtomicType *AT = AtomicTy->getAs<AtomicType>())
    MemTy = AT->getValueType();
  CharUnits sizeChars = getContext().getTypeSizeInChars(AtomicTy);
  uint64_t Size = sizeChars.getQuantity();
  CharUnits alignChars = getContext().getTypeAlignInChars(AtomicTy);
  unsigned Align = alignChars.getQuantity();
  unsigned MaxInlineWidthInBits =
    getContext().getTargetInfo().getMaxAtomicInlineWidth();
  bool UseLibcall = (Size != Align ||
                     getContext().toBits(sizeChars) > MaxInlineWidthInBits);

  llvm::Value *Ptr, *Order, *OrderFail = 0, *Val1 = 0, *Val2 = 0;
  Ptr = EmitScalarExpr(E->getPtr());

  if (E->getOp() == AtomicExpr::AO__c11_atomic_init) {
    assert(!Dest && "Init does not return a value");
    LValue LV = MakeAddrLValue(Ptr, AtomicTy, alignChars);
    switch (getEvaluationKind(E->getVal1()->getType())) {
    case TEK_Scalar:
      EmitScalarInit(EmitScalarExpr(E->getVal1()), LV);
      return RValue::get(0);
    case TEK_Complex:
      EmitComplexExprIntoLValue(E->getVal1(), LV, /*isInit*/ true);
      return RValue::get(0);
    case TEK_Aggregate: {
      AggValueSlot Slot = AggValueSlot::forLValue(LV,
                                        AggValueSlot::IsNotDestructed,
                                        AggValueSlot::DoesNotNeedGCBarriers,
                                        AggValueSlot::IsNotAliased);
      EmitAggExpr(E->getVal1(), Slot);
      return RValue::get(0);
    }
    }
    llvm_unreachable("bad evaluation kind");
  }

  Order = EmitScalarExpr(E->getOrder());

  switch (E->getOp()) {
  case AtomicExpr::AO__c11_atomic_init:
    llvm_unreachable("Already handled!");

  case AtomicExpr::AO__c11_atomic_load:
  case AtomicExpr::AO__atomic_load_n:
    break;

  case AtomicExpr::AO__atomic_load:
    Dest = EmitScalarExpr(E->getVal1());
    break;

  case AtomicExpr::AO__atomic_store:
    Val1 = EmitScalarExpr(E->getVal1());
    break;

  case AtomicExpr::AO__atomic_exchange:
    Val1 = EmitScalarExpr(E->getVal1());
    Dest = EmitScalarExpr(E->getVal2());
    break;

  case AtomicExpr::AO__c11_atomic_compare_exchange_strong:
  case AtomicExpr::AO__c11_atomic_compare_exchange_weak:
  case AtomicExpr::AO__atomic_compare_exchange_n:
  case AtomicExpr::AO__atomic_compare_exchange:
    Val1 = EmitScalarExpr(E->getVal1());
    if (E->getOp() == AtomicExpr::AO__atomic_compare_exchange)
      Val2 = EmitScalarExpr(E->getVal2());
    else
      Val2 = EmitValToTemp(*this, E->getVal2());
    OrderFail = EmitScalarExpr(E->getOrderFail());
    // Evaluate and discard the 'weak' argument.
    if (E->getNumSubExprs() == 6)
      EmitScalarExpr(E->getWeak());
    break;

  case AtomicExpr::AO__c11_atomic_fetch_add:
  case AtomicExpr::AO__c11_atomic_fetch_sub:
    if (MemTy->isPointerType()) {
      // For pointer arithmetic, we're required to do a bit of math:
      // adding 1 to an int* is not the same as adding 1 to a uintptr_t.
      // ... but only for the C11 builtins. The GNU builtins expect the
      // user to multiply by sizeof(T).
      QualType Val1Ty = E->getVal1()->getType();
      llvm::Value *Val1Scalar = EmitScalarExpr(E->getVal1());
      CharUnits PointeeIncAmt =
          getContext().getTypeSizeInChars(MemTy->getPointeeType());
      Val1Scalar = Builder.CreateMul(Val1Scalar, CGM.getSize(PointeeIncAmt));
      Val1 = CreateMemTemp(Val1Ty, ".atomictmp");
      EmitStoreOfScalar(Val1Scalar, MakeAddrLValue(Val1, Val1Ty));
      break;
    }
    // Fall through.
  case AtomicExpr::AO__atomic_fetch_add:
  case AtomicExpr::AO__atomic_fetch_sub:
  case AtomicExpr::AO__atomic_add_fetch:
  case AtomicExpr::AO__atomic_sub_fetch:
  case AtomicExpr::AO__c11_atomic_store:
  case AtomicExpr::AO__c11_atomic_exchange:
  case AtomicExpr::AO__atomic_store_n:
  case AtomicExpr::AO__atomic_exchange_n:
  case AtomicExpr::AO__c11_atomic_fetch_and:
  case AtomicExpr::AO__c11_atomic_fetch_or:
  case AtomicExpr::AO__c11_atomic_fetch_xor:
  case AtomicExpr::AO__atomic_fetch_and:
  case AtomicExpr::AO__atomic_fetch_or:
  case AtomicExpr::AO__atomic_fetch_xor:
  case AtomicExpr::AO__atomic_fetch_nand:
  case AtomicExpr::AO__atomic_and_fetch:
  case AtomicExpr::AO__atomic_or_fetch:
  case AtomicExpr::AO__atomic_xor_fetch:
  case AtomicExpr::AO__atomic_nand_fetch:
    Val1 = EmitValToTemp(*this, E->getVal1());
    break;
  }

  if (!E->getType()->isVoidType() && !Dest)
    Dest = CreateMemTemp(E->getType(), ".atomicdst");

  // Use a library call.  See: http://gcc.gnu.org/wiki/Atomic/GCCMM/LIbrary .
  if (UseLibcall) {

    SmallVector<QualType, 5> Params;
    CallArgList Args;
    // Size is always the first parameter
    Args.add(RValue::get(llvm::ConstantInt::get(SizeTy, Size)),
             getContext().getSizeType());
    // Atomic address is always the second parameter
    Args.add(RValue::get(EmitCastToVoidPtr(Ptr)),
             getContext().VoidPtrTy);

    const char* LibCallName;
    QualType RetTy = getContext().VoidTy;
    switch (E->getOp()) {
    // There is only one libcall for compare an exchange, because there is no
    // optimisation benefit possible from a libcall version of a weak compare
    // and exchange.
    // bool __atomic_compare_exchange(size_t size, void *obj, void *expected,
    //                                void *desired, int success, int failure)
    case AtomicExpr::AO__c11_atomic_compare_exchange_weak:
    case AtomicExpr::AO__c11_atomic_compare_exchange_strong:
    case AtomicExpr::AO__atomic_compare_exchange:
    case AtomicExpr::AO__atomic_compare_exchange_n:
      LibCallName = "__atomic_compare_exchange";
      RetTy = getContext().BoolTy;
      Args.add(RValue::get(EmitCastToVoidPtr(Val1)),
               getContext().VoidPtrTy);
      Args.add(RValue::get(EmitCastToVoidPtr(Val2)),
               getContext().VoidPtrTy);
      Args.add(RValue::get(Order),
               getContext().IntTy);
      Order = OrderFail;
      break;
    // void __atomic_exchange(size_t size, void *mem, void *val, void *return,
    //                        int order)
    case AtomicExpr::AO__c11_atomic_exchange:
    case AtomicExpr::AO__atomic_exchange_n:
    case AtomicExpr::AO__atomic_exchange:
      LibCallName = "__atomic_exchange";
      Args.add(RValue::get(EmitCastToVoidPtr(Val1)),
               getContext().VoidPtrTy);
      Args.add(RValue::get(EmitCastToVoidPtr(Dest)),
               getContext().VoidPtrTy);
      break;
    // void __atomic_store(size_t size, void *mem, void *val, int order)
    case AtomicExpr::AO__c11_atomic_store:
    case AtomicExpr::AO__atomic_store:
    case AtomicExpr::AO__atomic_store_n:
      LibCallName = "__atomic_store";
      Args.add(RValue::get(EmitCastToVoidPtr(Val1)),
               getContext().VoidPtrTy);
      break;
    // void __atomic_load(size_t size, void *mem, void *return, int order)
    case AtomicExpr::AO__c11_atomic_load:
    case AtomicExpr::AO__atomic_load:
    case AtomicExpr::AO__atomic_load_n:
      LibCallName = "__atomic_load";
      Args.add(RValue::get(EmitCastToVoidPtr(Dest)),
               getContext().VoidPtrTy);
      break;
#if 0
    // These are only defined for 1-16 byte integers.  It is not clear what
    // their semantics would be on anything else...
    case AtomicExpr::Add:   LibCallName = "__atomic_fetch_add_generic"; break;
    case AtomicExpr::Sub:   LibCallName = "__atomic_fetch_sub_generic"; break;
    case AtomicExpr::And:   LibCallName = "__atomic_fetch_and_generic"; break;
    case AtomicExpr::Or:    LibCallName = "__atomic_fetch_or_generic"; break;
    case AtomicExpr::Xor:   LibCallName = "__atomic_fetch_xor_generic"; break;
#endif
    default: return EmitUnsupportedRValue(E, "atomic library call");
    }
    // order is always the last parameter
    Args.add(RValue::get(Order),
             getContext().IntTy);

    const CGFunctionInfo &FuncInfo =
        CGM.getTypes().arrangeFreeFunctionCall(RetTy, Args,
            FunctionType::ExtInfo(), RequiredArgs::All);
    llvm::FunctionType *FTy = CGM.getTypes().GetFunctionType(FuncInfo);
    llvm::Constant *Func = CGM.CreateRuntimeFunction(FTy, LibCallName);
    RValue Res = EmitCall(FuncInfo, Func, ReturnValueSlot(), Args);
    if (E->isCmpXChg())
      return Res;
    if (E->getType()->isVoidType())
      return RValue::get(0);
    return convertTempToRValue(Dest, E->getType());
  }

  bool IsStore = E->getOp() == AtomicExpr::AO__c11_atomic_store ||
                 E->getOp() == AtomicExpr::AO__atomic_store ||
                 E->getOp() == AtomicExpr::AO__atomic_store_n;
  bool IsLoad = E->getOp() == AtomicExpr::AO__c11_atomic_load ||
                E->getOp() == AtomicExpr::AO__atomic_load ||
                E->getOp() == AtomicExpr::AO__atomic_load_n;

  llvm::Type *IPtrTy =
      llvm::IntegerType::get(getLLVMContext(), Size * 8)->getPointerTo();
  llvm::Value *OrigDest = Dest;
  Ptr = Builder.CreateBitCast(Ptr, IPtrTy);
  if (Val1) Val1 = Builder.CreateBitCast(Val1, IPtrTy);
  if (Val2) Val2 = Builder.CreateBitCast(Val2, IPtrTy);
  if (Dest && !E->isCmpXChg()) Dest = Builder.CreateBitCast(Dest, IPtrTy);

  if (isa<llvm::ConstantInt>(Order)) {
    int ord = cast<llvm::ConstantInt>(Order)->getZExtValue();
    switch (ord) {
    case 0:  // memory_order_relaxed
      EmitAtomicOp(*this, E, Dest, Ptr, Val1, Val2, Size, Align,
                   llvm::Monotonic);
      break;
    case 1:  // memory_order_consume
    case 2:  // memory_order_acquire
      if (IsStore)
        break; // Avoid crashing on code with undefined behavior
      EmitAtomicOp(*this, E, Dest, Ptr, Val1, Val2, Size, Align,
                   llvm::Acquire);
      break;
    case 3:  // memory_order_release
      if (IsLoad)
        break; // Avoid crashing on code with undefined behavior
      EmitAtomicOp(*this, E, Dest, Ptr, Val1, Val2, Size, Align,
                   llvm::Release);
      break;
    case 4:  // memory_order_acq_rel
      if (IsLoad || IsStore)
        break; // Avoid crashing on code with undefined behavior
      EmitAtomicOp(*this, E, Dest, Ptr, Val1, Val2, Size, Align,
                   llvm::AcquireRelease);
      break;
    case 5:  // memory_order_seq_cst
      EmitAtomicOp(*this, E, Dest, Ptr, Val1, Val2, Size, Align,
                   llvm::SequentiallyConsistent);
      break;
    default: // invalid order
      // We should not ever get here normally, but it's hard to
      // enforce that in general.
      break;
    }
    if (E->getType()->isVoidType())
      return RValue::get(0);
    return convertTempToRValue(OrigDest, E->getType());
  }

  // Long case, when Order isn't obviously constant.

  // Create all the relevant BB's
  llvm::BasicBlock *MonotonicBB = 0, *AcquireBB = 0, *ReleaseBB = 0,
                   *AcqRelBB = 0, *SeqCstBB = 0;
  MonotonicBB = createBasicBlock("monotonic", CurFn);
  if (!IsStore)
    AcquireBB = createBasicBlock("acquire", CurFn);
  if (!IsLoad)
    ReleaseBB = createBasicBlock("release", CurFn);
  if (!IsLoad && !IsStore)
    AcqRelBB = createBasicBlock("acqrel", CurFn);
  SeqCstBB = createBasicBlock("seqcst", CurFn);
  llvm::BasicBlock *ContBB = createBasicBlock("atomic.continue", CurFn);

  // Create the switch for the split
  // MonotonicBB is arbitrarily chosen as the default case; in practice, this
  // doesn't matter unless someone is crazy enough to use something that
  // doesn't fold to a constant for the ordering.
  Order = Builder.CreateIntCast(Order, Builder.getInt32Ty(), false);
  llvm::SwitchInst *SI = Builder.CreateSwitch(Order, MonotonicBB);

  // Emit all the different atomics
  Builder.SetInsertPoint(MonotonicBB);
  EmitAtomicOp(*this, E, Dest, Ptr, Val1, Val2, Size, Align,
               llvm::Monotonic);
  Builder.CreateBr(ContBB);
  if (!IsStore) {
    Builder.SetInsertPoint(AcquireBB);
    EmitAtomicOp(*this, E, Dest, Ptr, Val1, Val2, Size, Align,
                 llvm::Acquire);
    Builder.CreateBr(ContBB);
    SI->addCase(Builder.getInt32(1), AcquireBB);
    SI->addCase(Builder.getInt32(2), AcquireBB);
  }
  if (!IsLoad) {
    Builder.SetInsertPoint(ReleaseBB);
    EmitAtomicOp(*this, E, Dest, Ptr, Val1, Val2, Size, Align,
                 llvm::Release);
    Builder.CreateBr(ContBB);
    SI->addCase(Builder.getInt32(3), ReleaseBB);
  }
  if (!IsLoad && !IsStore) {
    Builder.SetInsertPoint(AcqRelBB);
    EmitAtomicOp(*this, E, Dest, Ptr, Val1, Val2, Size, Align,
                 llvm::AcquireRelease);
    Builder.CreateBr(ContBB);
    SI->addCase(Builder.getInt32(4), AcqRelBB);
  }
  Builder.SetInsertPoint(SeqCstBB);
  EmitAtomicOp(*this, E, Dest, Ptr, Val1, Val2, Size, Align,
               llvm::SequentiallyConsistent);
  Builder.CreateBr(ContBB);
  SI->addCase(Builder.getInt32(5), SeqCstBB);

  // Cleanup and return
  Builder.SetInsertPoint(ContBB);
  if (E->getType()->isVoidType())
    return RValue::get(0);
  return convertTempToRValue(OrigDest, E->getType());
}
