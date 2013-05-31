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
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Operator.h"

using namespace clang;
using namespace CodeGen;

// The ABI values for various atomic memory orderings.
enum AtomicOrderingKind {
  AO_ABI_memory_order_relaxed = 0,
  AO_ABI_memory_order_consume = 1,
  AO_ABI_memory_order_acquire = 2,
  AO_ABI_memory_order_release = 3,
  AO_ABI_memory_order_acq_rel = 4,
  AO_ABI_memory_order_seq_cst = 5
};

namespace {
  class AtomicInfo {
    CodeGenFunction &CGF;
    QualType AtomicTy;
    QualType ValueTy;
    uint64_t AtomicSizeInBits;
    uint64_t ValueSizeInBits;
    CharUnits AtomicAlign;
    CharUnits ValueAlign;
    CharUnits LValueAlign;
    TypeEvaluationKind EvaluationKind;
    bool UseLibcall;
  public:
    AtomicInfo(CodeGenFunction &CGF, LValue &lvalue) : CGF(CGF) {
      assert(lvalue.isSimple());

      AtomicTy = lvalue.getType();
      ValueTy = AtomicTy->castAs<AtomicType>()->getValueType();
      EvaluationKind = CGF.getEvaluationKind(ValueTy);

      ASTContext &C = CGF.getContext();

      uint64_t valueAlignInBits;
      llvm::tie(ValueSizeInBits, valueAlignInBits) = C.getTypeInfo(ValueTy);

      uint64_t atomicAlignInBits;
      llvm::tie(AtomicSizeInBits, atomicAlignInBits) = C.getTypeInfo(AtomicTy);

      assert(ValueSizeInBits <= AtomicSizeInBits);
      assert(valueAlignInBits <= atomicAlignInBits);

      AtomicAlign = C.toCharUnitsFromBits(atomicAlignInBits);
      ValueAlign = C.toCharUnitsFromBits(valueAlignInBits);
      if (lvalue.getAlignment().isZero())
        lvalue.setAlignment(AtomicAlign);

      UseLibcall =
        (AtomicSizeInBits > uint64_t(C.toBits(lvalue.getAlignment())) ||
         AtomicSizeInBits > C.getTargetInfo().getMaxAtomicInlineWidth());
    }

    QualType getAtomicType() const { return AtomicTy; }
    QualType getValueType() const { return ValueTy; }
    CharUnits getAtomicAlignment() const { return AtomicAlign; }
    CharUnits getValueAlignment() const { return ValueAlign; }
    uint64_t getAtomicSizeInBits() const { return AtomicSizeInBits; }
    uint64_t getValueSizeInBits() const { return AtomicSizeInBits; }
    TypeEvaluationKind getEvaluationKind() const { return EvaluationKind; }
    bool shouldUseLibcall() const { return UseLibcall; }

    /// Is the atomic size larger than the underlying value type?
    ///
    /// Note that the absence of padding does not mean that atomic
    /// objects are completely interchangeable with non-atomic
    /// objects: we might have promoted the alignment of a type
    /// without making it bigger.
    bool hasPadding() const {
      return (ValueSizeInBits != AtomicSizeInBits);
    }

    void emitMemSetZeroIfNecessary(LValue dest) const;

    llvm::Value *getAtomicSizeValue() const {
      CharUnits size = CGF.getContext().toCharUnitsFromBits(AtomicSizeInBits);
      return CGF.CGM.getSize(size);
    }

    /// Cast the given pointer to an integer pointer suitable for
    /// atomic operations.
    llvm::Value *emitCastToAtomicIntPointer(llvm::Value *addr) const;

    /// Turn an atomic-layout object into an r-value.
    RValue convertTempToRValue(llvm::Value *addr,
                               AggValueSlot resultSlot) const;

    /// Copy an atomic r-value into atomic-layout memory.
    void emitCopyIntoMemory(RValue rvalue, LValue lvalue) const;

    /// Project an l-value down to the value field.
    LValue projectValue(LValue lvalue) const {
      llvm::Value *addr = lvalue.getAddress();
      if (hasPadding())
        addr = CGF.Builder.CreateStructGEP(addr, 0);

      return LValue::MakeAddr(addr, getValueType(), lvalue.getAlignment(),
                              CGF.getContext(), lvalue.getTBAAInfo());
    }

    /// Materialize an atomic r-value in atomic-layout memory.
    llvm::Value *materializeRValue(RValue rvalue) const;

  private:
    bool requiresMemSetZero(llvm::Type *type) const;
  };
}

static RValue emitAtomicLibcall(CodeGenFunction &CGF,
                                StringRef fnName,
                                QualType resultType,
                                CallArgList &args) {
  const CGFunctionInfo &fnInfo =
    CGF.CGM.getTypes().arrangeFreeFunctionCall(resultType, args,
            FunctionType::ExtInfo(), RequiredArgs::All);
  llvm::FunctionType *fnTy = CGF.CGM.getTypes().GetFunctionType(fnInfo);
  llvm::Constant *fn = CGF.CGM.CreateRuntimeFunction(fnTy, fnName);
  return CGF.EmitCall(fnInfo, fn, ReturnValueSlot(), args);
}

/// Does a store of the given IR type modify the full expected width?
static bool isFullSizeType(CodeGenModule &CGM, llvm::Type *type,
                           uint64_t expectedSize) {
  return (CGM.getDataLayout().getTypeStoreSize(type) * 8 == expectedSize);
}

/// Does the atomic type require memsetting to zero before initialization?
///
/// The IR type is provided as a way of making certain queries faster.
bool AtomicInfo::requiresMemSetZero(llvm::Type *type) const {
  // If the atomic type has size padding, we definitely need a memset.
  if (hasPadding()) return true;

  // Otherwise, do some simple heuristics to try to avoid it:
  switch (getEvaluationKind()) {
  // For scalars and complexes, check whether the store size of the
  // type uses the full size.
  case TEK_Scalar:
    return !isFullSizeType(CGF.CGM, type, AtomicSizeInBits);
  case TEK_Complex:
    return !isFullSizeType(CGF.CGM, type->getStructElementType(0),
                           AtomicSizeInBits / 2);

  // Just be pessimistic about aggregates.
  case TEK_Aggregate:
    return true;
  }
  llvm_unreachable("bad evaluation kind");
}

void AtomicInfo::emitMemSetZeroIfNecessary(LValue dest) const {
  llvm::Value *addr = dest.getAddress();
  if (!requiresMemSetZero(addr->getType()->getPointerElementType()))
    return;

  CGF.Builder.CreateMemSet(addr, llvm::ConstantInt::get(CGF.Int8Ty, 0),
                           AtomicSizeInBits / 8,
                           dest.getAlignment().getQuantity());
}

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

static void
AddDirectArgument(CodeGenFunction &CGF, CallArgList &Args,
                  bool UseOptimizedLibcall, llvm::Value *Val, QualType ValTy) {
  if (UseOptimizedLibcall) {
    // Load value and pass it to the function directly.
    unsigned Align = CGF.getContext().getTypeAlignInChars(ValTy).getQuantity();
    Val = CGF.EmitLoadOfScalar(Val, false, Align, ValTy);
    Args.add(RValue::get(Val), ValTy);
  } else {
    // Non-optimized functions always take a reference.
    Args.add(RValue::get(CGF.EmitCastToVoidPtr(Val)),
                         CGF.getContext().VoidPtrTy);
  }
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
    getTarget().getMaxAtomicInlineWidth();
  bool UseLibcall = (Size != Align ||
                     getContext().toBits(sizeChars) > MaxInlineWidthInBits);

  llvm::Value *Ptr, *Order, *OrderFail = 0, *Val1 = 0, *Val2 = 0;
  Ptr = EmitScalarExpr(E->getPtr());

  if (E->getOp() == AtomicExpr::AO__c11_atomic_init) {
    assert(!Dest && "Init does not return a value");
    LValue lvalue = LValue::MakeAddr(Ptr, AtomicTy, alignChars, getContext());
    EmitAtomicInit(E->getVal1(), lvalue);
    return RValue::get(0);
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
    bool UseOptimizedLibcall = false;
    switch (E->getOp()) {
    case AtomicExpr::AO__c11_atomic_fetch_add:
    case AtomicExpr::AO__atomic_fetch_add:
    case AtomicExpr::AO__c11_atomic_fetch_and:
    case AtomicExpr::AO__atomic_fetch_and:
    case AtomicExpr::AO__c11_atomic_fetch_or:
    case AtomicExpr::AO__atomic_fetch_or:
    case AtomicExpr::AO__c11_atomic_fetch_sub:
    case AtomicExpr::AO__atomic_fetch_sub:
    case AtomicExpr::AO__c11_atomic_fetch_xor:
    case AtomicExpr::AO__atomic_fetch_xor:
      // For these, only library calls for certain sizes exist.
      UseOptimizedLibcall = true;
      break;
    default:
      // Only use optimized library calls for sizes for which they exist.
      if (Size == 1 || Size == 2 || Size == 4 || Size == 8)
        UseOptimizedLibcall = true;
      break;
    }

    CallArgList Args;
    if (!UseOptimizedLibcall) {
      // For non-optimized library calls, the size is the first parameter
      Args.add(RValue::get(llvm::ConstantInt::get(SizeTy, Size)),
               getContext().getSizeType());
    }
    // Atomic address is the first or second parameter
    Args.add(RValue::get(EmitCastToVoidPtr(Ptr)),
             getContext().VoidPtrTy);

    std::string LibCallName;
    QualType RetTy;
    bool HaveRetTy = false;
    switch (E->getOp()) {
    // There is only one libcall for compare an exchange, because there is no
    // optimisation benefit possible from a libcall version of a weak compare
    // and exchange.
    // bool __atomic_compare_exchange(size_t size, void *mem, void *expected,
    //                                void *desired, int success, int failure)
    // bool __atomic_compare_exchange_N(T *mem, T *expected, T desired,
    //                                  int success, int failure)
    case AtomicExpr::AO__c11_atomic_compare_exchange_weak:
    case AtomicExpr::AO__c11_atomic_compare_exchange_strong:
    case AtomicExpr::AO__atomic_compare_exchange:
    case AtomicExpr::AO__atomic_compare_exchange_n:
      LibCallName = "__atomic_compare_exchange";
      RetTy = getContext().BoolTy;
      HaveRetTy = true;
      Args.add(RValue::get(EmitCastToVoidPtr(Val1)),
               getContext().VoidPtrTy);
      AddDirectArgument(*this, Args, UseOptimizedLibcall, Val2, MemTy);
      Args.add(RValue::get(Order),
               getContext().IntTy);
      Order = OrderFail;
      break;
    // void __atomic_exchange(size_t size, void *mem, void *val, void *return,
    //                        int order)
    // T __atomic_exchange_N(T *mem, T val, int order)
    case AtomicExpr::AO__c11_atomic_exchange:
    case AtomicExpr::AO__atomic_exchange_n:
    case AtomicExpr::AO__atomic_exchange:
      LibCallName = "__atomic_exchange";
      AddDirectArgument(*this, Args, UseOptimizedLibcall, Val1, MemTy);
      break;
    // void __atomic_store(size_t size, void *mem, void *val, int order)
    // void __atomic_store_N(T *mem, T val, int order)
    case AtomicExpr::AO__c11_atomic_store:
    case AtomicExpr::AO__atomic_store:
    case AtomicExpr::AO__atomic_store_n:
      LibCallName = "__atomic_store";
      RetTy = getContext().VoidTy;
      HaveRetTy = true;
      AddDirectArgument(*this, Args, UseOptimizedLibcall, Val1, MemTy);
      break;
    // void __atomic_load(size_t size, void *mem, void *return, int order)
    // T __atomic_load_N(T *mem, int order)
    case AtomicExpr::AO__c11_atomic_load:
    case AtomicExpr::AO__atomic_load:
    case AtomicExpr::AO__atomic_load_n:
      LibCallName = "__atomic_load";
      break;
    // T __atomic_fetch_add_N(T *mem, T val, int order)
    case AtomicExpr::AO__c11_atomic_fetch_add:
    case AtomicExpr::AO__atomic_fetch_add:
      LibCallName = "__atomic_fetch_add";
      AddDirectArgument(*this, Args, UseOptimizedLibcall, Val1, MemTy);
      break;
    // T __atomic_fetch_and_N(T *mem, T val, int order)
    case AtomicExpr::AO__c11_atomic_fetch_and:
    case AtomicExpr::AO__atomic_fetch_and:
      LibCallName = "__atomic_fetch_and";
      AddDirectArgument(*this, Args, UseOptimizedLibcall, Val1, MemTy);
      break;
    // T __atomic_fetch_or_N(T *mem, T val, int order)
    case AtomicExpr::AO__c11_atomic_fetch_or:
    case AtomicExpr::AO__atomic_fetch_or:
      LibCallName = "__atomic_fetch_or";
      AddDirectArgument(*this, Args, UseOptimizedLibcall, Val1, MemTy);
      break;
    // T __atomic_fetch_sub_N(T *mem, T val, int order)
    case AtomicExpr::AO__c11_atomic_fetch_sub:
    case AtomicExpr::AO__atomic_fetch_sub:
      LibCallName = "__atomic_fetch_sub";
      AddDirectArgument(*this, Args, UseOptimizedLibcall, Val1, MemTy);
      break;
    // T __atomic_fetch_xor_N(T *mem, T val, int order)
    case AtomicExpr::AO__c11_atomic_fetch_xor:
    case AtomicExpr::AO__atomic_fetch_xor:
      LibCallName = "__atomic_fetch_xor";
      AddDirectArgument(*this, Args, UseOptimizedLibcall, Val1, MemTy);
      break;
    default: return EmitUnsupportedRValue(E, "atomic library call");
    }

    // Optimized functions have the size in their name.
    if (UseOptimizedLibcall)
      LibCallName += "_" + llvm::utostr(Size);
    // By default, assume we return a value of the atomic type.
    if (!HaveRetTy) {
      if (UseOptimizedLibcall) {
        // Value is returned directly.
        RetTy = MemTy;
      } else {
        // Value is returned through parameter before the order.
        RetTy = getContext().VoidTy;
        Args.add(RValue::get(EmitCastToVoidPtr(Dest)),
                 getContext().VoidPtrTy);
      }
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
    if (!RetTy->isVoidType())
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
    case AO_ABI_memory_order_relaxed:
      EmitAtomicOp(*this, E, Dest, Ptr, Val1, Val2, Size, Align,
                   llvm::Monotonic);
      break;
    case AO_ABI_memory_order_consume:
    case AO_ABI_memory_order_acquire:
      if (IsStore)
        break; // Avoid crashing on code with undefined behavior
      EmitAtomicOp(*this, E, Dest, Ptr, Val1, Val2, Size, Align,
                   llvm::Acquire);
      break;
    case AO_ABI_memory_order_release:
      if (IsLoad)
        break; // Avoid crashing on code with undefined behavior
      EmitAtomicOp(*this, E, Dest, Ptr, Val1, Val2, Size, Align,
                   llvm::Release);
      break;
    case AO_ABI_memory_order_acq_rel:
      if (IsLoad || IsStore)
        break; // Avoid crashing on code with undefined behavior
      EmitAtomicOp(*this, E, Dest, Ptr, Val1, Val2, Size, Align,
                   llvm::AcquireRelease);
      break;
    case AO_ABI_memory_order_seq_cst:
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

llvm::Value *AtomicInfo::emitCastToAtomicIntPointer(llvm::Value *addr) const {
  unsigned addrspace =
    cast<llvm::PointerType>(addr->getType())->getAddressSpace();
  llvm::IntegerType *ty =
    llvm::IntegerType::get(CGF.getLLVMContext(), AtomicSizeInBits);
  return CGF.Builder.CreateBitCast(addr, ty->getPointerTo(addrspace));
}

RValue AtomicInfo::convertTempToRValue(llvm::Value *addr,
                                       AggValueSlot resultSlot) const {
  if (EvaluationKind == TEK_Aggregate) {
    // Nothing to do if the result is ignored.
    if (resultSlot.isIgnored()) return resultSlot.asRValue();

    assert(resultSlot.getAddr() == addr || hasPadding());

    // In these cases, we should have emitted directly into the result slot.
    if (!hasPadding() || resultSlot.isValueOfAtomic())
      return resultSlot.asRValue();

    // Otherwise, fall into the common path.
  }

  // Drill into the padding structure if we have one.
  if (hasPadding())
    addr = CGF.Builder.CreateStructGEP(addr, 0);

  // If we're emitting to an aggregate, copy into the result slot.
  if (EvaluationKind == TEK_Aggregate) {
    CGF.EmitAggregateCopy(resultSlot.getAddr(), addr, getValueType(),
                          resultSlot.isVolatile());
    return resultSlot.asRValue();
  }

  // Otherwise, just convert the temporary to an r-value using the
  // normal conversion routine.
  return CGF.convertTempToRValue(addr, getValueType());
}

/// Emit a load from an l-value of atomic type.  Note that the r-value
/// we produce is an r-value of the atomic *value* type.
RValue CodeGenFunction::EmitAtomicLoad(LValue src, AggValueSlot resultSlot) {
  AtomicInfo atomics(*this, src);

  // Check whether we should use a library call.
  if (atomics.shouldUseLibcall()) {
    llvm::Value *tempAddr;
    if (resultSlot.isValueOfAtomic()) {
      assert(atomics.getEvaluationKind() == TEK_Aggregate);
      tempAddr = resultSlot.getPaddedAtomicAddr();
    } else if (!resultSlot.isIgnored() && !atomics.hasPadding()) {
      assert(atomics.getEvaluationKind() == TEK_Aggregate);
      tempAddr = resultSlot.getAddr();
    } else {
      tempAddr = CreateMemTemp(atomics.getAtomicType(), "atomic-load-temp");
    }

    // void __atomic_load(size_t size, void *mem, void *return, int order);
    CallArgList args;
    args.add(RValue::get(atomics.getAtomicSizeValue()),
             getContext().getSizeType());
    args.add(RValue::get(EmitCastToVoidPtr(src.getAddress())),
             getContext().VoidPtrTy);
    args.add(RValue::get(EmitCastToVoidPtr(tempAddr)),
             getContext().VoidPtrTy);
    args.add(RValue::get(llvm::ConstantInt::get(IntTy,
                                                AO_ABI_memory_order_seq_cst)),
             getContext().IntTy);
    emitAtomicLibcall(*this, "__atomic_load", getContext().VoidTy, args);

    // Produce the r-value.
    return atomics.convertTempToRValue(tempAddr, resultSlot);
  }

  // Okay, we're doing this natively.
  llvm::Value *addr = atomics.emitCastToAtomicIntPointer(src.getAddress());
  llvm::LoadInst *load = Builder.CreateLoad(addr, "atomic-load");
  load->setAtomic(llvm::SequentiallyConsistent);

  // Other decoration.
  load->setAlignment(src.getAlignment().getQuantity());
  if (src.isVolatileQualified())
    load->setVolatile(true);
  if (src.getTBAAInfo())
    CGM.DecorateInstruction(load, src.getTBAAInfo());

  // Okay, turn that back into the original value type.
  QualType valueType = atomics.getValueType();
  llvm::Value *result = load;

  // If we're ignoring an aggregate return, don't do anything.
  if (atomics.getEvaluationKind() == TEK_Aggregate && resultSlot.isIgnored())
    return RValue::getAggregate(0, false);

  // The easiest way to do this this is to go through memory, but we
  // try not to in some easy cases.
  if (atomics.getEvaluationKind() == TEK_Scalar && !atomics.hasPadding()) {
    llvm::Type *resultTy = CGM.getTypes().ConvertTypeForMem(valueType);
    if (isa<llvm::IntegerType>(resultTy)) {
      assert(result->getType() == resultTy);
      result = EmitFromMemory(result, valueType);
    } else if (isa<llvm::PointerType>(resultTy)) {
      result = Builder.CreateIntToPtr(result, resultTy);
    } else {
      result = Builder.CreateBitCast(result, resultTy);
    }
    return RValue::get(result);
  }

  // Create a temporary.  This needs to be big enough to hold the
  // atomic integer.
  llvm::Value *temp;
  bool tempIsVolatile = false;
  CharUnits tempAlignment;
  if (atomics.getEvaluationKind() == TEK_Aggregate &&
      (!atomics.hasPadding() || resultSlot.isValueOfAtomic())) {
    assert(!resultSlot.isIgnored());
    if (resultSlot.isValueOfAtomic()) {
      temp = resultSlot.getPaddedAtomicAddr();
      tempAlignment = atomics.getAtomicAlignment();
    } else {
      temp = resultSlot.getAddr();
      tempAlignment = atomics.getValueAlignment();
    }
    tempIsVolatile = resultSlot.isVolatile();
  } else {
    temp = CreateMemTemp(atomics.getAtomicType(), "atomic-load-temp");
    tempAlignment = atomics.getAtomicAlignment();
  }

  // Slam the integer into the temporary.
  llvm::Value *castTemp = atomics.emitCastToAtomicIntPointer(temp);
  Builder.CreateAlignedStore(result, castTemp, tempAlignment.getQuantity())
    ->setVolatile(tempIsVolatile);

  return atomics.convertTempToRValue(temp, resultSlot);
}



/// Copy an r-value into memory as part of storing to an atomic type.
/// This needs to create a bit-pattern suitable for atomic operations.
void AtomicInfo::emitCopyIntoMemory(RValue rvalue, LValue dest) const {
  // If we have an r-value, the rvalue should be of the atomic type,
  // which means that the caller is responsible for having zeroed
  // any padding.  Just do an aggregate copy of that type.
  if (rvalue.isAggregate()) {
    CGF.EmitAggregateCopy(dest.getAddress(),
                          rvalue.getAggregateAddr(),
                          getAtomicType(),
                          (rvalue.isVolatileQualified()
                           || dest.isVolatileQualified()),
                          dest.getAlignment());
    return;
  }

  // Okay, otherwise we're copying stuff.

  // Zero out the buffer if necessary.
  emitMemSetZeroIfNecessary(dest);

  // Drill past the padding if present.
  dest = projectValue(dest);

  // Okay, store the rvalue in.
  if (rvalue.isScalar()) {
    CGF.EmitStoreOfScalar(rvalue.getScalarVal(), dest, /*init*/ true);
  } else {
    CGF.EmitStoreOfComplex(rvalue.getComplexVal(), dest, /*init*/ true);
  }
}


/// Materialize an r-value into memory for the purposes of storing it
/// to an atomic type.
llvm::Value *AtomicInfo::materializeRValue(RValue rvalue) const {
  // Aggregate r-values are already in memory, and EmitAtomicStore
  // requires them to be values of the atomic type.
  if (rvalue.isAggregate())
    return rvalue.getAggregateAddr();

  // Otherwise, make a temporary and materialize into it.
  llvm::Value *temp = CGF.CreateMemTemp(getAtomicType(), "atomic-store-temp");
  LValue tempLV = CGF.MakeAddrLValue(temp, getAtomicType(), getAtomicAlignment());
  emitCopyIntoMemory(rvalue, tempLV);
  return temp;
}

/// Emit a store to an l-value of atomic type.
///
/// Note that the r-value is expected to be an r-value *of the atomic
/// type*; this means that for aggregate r-values, it should include
/// storage for any padding that was necessary.
void CodeGenFunction::EmitAtomicStore(RValue rvalue, LValue dest,
                                      bool isInit) {
  // If this is an aggregate r-value, it should agree in type except
  // maybe for address-space qualification.
  assert(!rvalue.isAggregate() ||
         rvalue.getAggregateAddr()->getType()->getPointerElementType()
           == dest.getAddress()->getType()->getPointerElementType());

  AtomicInfo atomics(*this, dest);

  // If this is an initialization, just put the value there normally.
  if (isInit) {
    atomics.emitCopyIntoMemory(rvalue, dest);
    return;
  }

  // Check whether we should use a library call.
  if (atomics.shouldUseLibcall()) {
    // Produce a source address.
    llvm::Value *srcAddr = atomics.materializeRValue(rvalue);

    // void __atomic_store(size_t size, void *mem, void *val, int order)
    CallArgList args;
    args.add(RValue::get(atomics.getAtomicSizeValue()),
             getContext().getSizeType());
    args.add(RValue::get(EmitCastToVoidPtr(dest.getAddress())),
             getContext().VoidPtrTy);
    args.add(RValue::get(EmitCastToVoidPtr(srcAddr)),
             getContext().VoidPtrTy);
    args.add(RValue::get(llvm::ConstantInt::get(IntTy,
                                                AO_ABI_memory_order_seq_cst)),
             getContext().IntTy);
    emitAtomicLibcall(*this, "__atomic_store", getContext().VoidTy, args);
    return;
  }

  // Okay, we're doing this natively.
  llvm::Value *intValue;

  // If we've got a scalar value of the right size, try to avoid going
  // through memory.
  if (rvalue.isScalar() && !atomics.hasPadding()) {
    llvm::Value *value = rvalue.getScalarVal();
    if (isa<llvm::IntegerType>(value->getType())) {
      intValue = value;
    } else {
      llvm::IntegerType *inputIntTy =
        llvm::IntegerType::get(getLLVMContext(), atomics.getValueSizeInBits());
      if (isa<llvm::PointerType>(value->getType())) {
        intValue = Builder.CreatePtrToInt(value, inputIntTy);
      } else {
        intValue = Builder.CreateBitCast(value, inputIntTy);
      }
    }

  // Otherwise, we need to go through memory.
  } else {
    // Put the r-value in memory.
    llvm::Value *addr = atomics.materializeRValue(rvalue);

    // Cast the temporary to the atomic int type and pull a value out.
    addr = atomics.emitCastToAtomicIntPointer(addr);
    intValue = Builder.CreateAlignedLoad(addr,
                                 atomics.getAtomicAlignment().getQuantity());
  }

  // Do the atomic store.
  llvm::Value *addr = atomics.emitCastToAtomicIntPointer(dest.getAddress());
  llvm::StoreInst *store = Builder.CreateStore(intValue, addr);

  // Initializations don't need to be atomic.
  if (!isInit) store->setAtomic(llvm::SequentiallyConsistent);

  // Other decoration.
  store->setAlignment(dest.getAlignment().getQuantity());
  if (dest.isVolatileQualified())
    store->setVolatile(true);
  if (dest.getTBAAInfo())
    CGM.DecorateInstruction(store, dest.getTBAAInfo());
}

void CodeGenFunction::EmitAtomicInit(Expr *init, LValue dest) {
  AtomicInfo atomics(*this, dest);

  switch (atomics.getEvaluationKind()) {
  case TEK_Scalar: {
    llvm::Value *value = EmitScalarExpr(init);
    atomics.emitCopyIntoMemory(RValue::get(value), dest);
    return;
  }

  case TEK_Complex: {
    ComplexPairTy value = EmitComplexExpr(init);
    atomics.emitCopyIntoMemory(RValue::getComplex(value), dest);
    return;
  }

  case TEK_Aggregate: {
    // Memset the buffer first if there's any possibility of
    // uninitialized internal bits.
    atomics.emitMemSetZeroIfNecessary(dest);

    // HACK: whether the initializer actually has an atomic type
    // doesn't really seem reliable right now.
    if (!init->getType()->isAtomicType()) {
      dest = atomics.projectValue(dest);
    }

    // Evaluate the expression directly into the destination.
    AggValueSlot slot = AggValueSlot::forLValue(dest,
                                        AggValueSlot::IsNotDestructed,
                                        AggValueSlot::DoesNotNeedGCBarriers,
                                        AggValueSlot::IsNotAliased);
    EmitAggExpr(init, slot);
    return;
  }
  }
  llvm_unreachable("bad evaluation kind");
}
