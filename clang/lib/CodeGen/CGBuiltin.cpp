//===---- CGBuiltin.cpp - Emit LLVM Code for builtins ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Builtin calls as LLVM code.
//
//===----------------------------------------------------------------------===//

#include "CodeGenFunction.h"
#include "CGObjCRuntime.h"
#include "CodeGenModule.h"
#include "TargetInfo.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/TargetBuiltins.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CodeGen/CGFunctionInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Intrinsics.h"

using namespace clang;
using namespace CodeGen;
using namespace llvm;

/// getBuiltinLibFunction - Given a builtin id for a function like
/// "__builtin_fabsf", return a Function* for "fabsf".
llvm::Value *CodeGenModule::getBuiltinLibFunction(const FunctionDecl *FD,
                                                  unsigned BuiltinID) {
  assert(Context.BuiltinInfo.isLibFunction(BuiltinID));

  // Get the name, skip over the __builtin_ prefix (if necessary).
  StringRef Name;
  GlobalDecl D(FD);

  // If the builtin has been declared explicitly with an assembler label,
  // use the mangled name. This differs from the plain label on platforms
  // that prefix labels.
  if (FD->hasAttr<AsmLabelAttr>())
    Name = getMangledName(D);
  else
    Name = Context.BuiltinInfo.GetName(BuiltinID) + 10;

  llvm::FunctionType *Ty =
    cast<llvm::FunctionType>(getTypes().ConvertType(FD->getType()));

  return GetOrCreateLLVMFunction(Name, Ty, D, /*ForVTable=*/false);
}

/// Emit the conversions required to turn the given value into an
/// integer of the given size.
static Value *EmitToInt(CodeGenFunction &CGF, llvm::Value *V,
                        QualType T, llvm::IntegerType *IntType) {
  V = CGF.EmitToMemory(V, T);

  if (V->getType()->isPointerTy())
    return CGF.Builder.CreatePtrToInt(V, IntType);

  assert(V->getType() == IntType);
  return V;
}

static Value *EmitFromInt(CodeGenFunction &CGF, llvm::Value *V,
                          QualType T, llvm::Type *ResultType) {
  V = CGF.EmitFromMemory(V, T);

  if (ResultType->isPointerTy())
    return CGF.Builder.CreateIntToPtr(V, ResultType);

  assert(V->getType() == ResultType);
  return V;
}

/// Utility to insert an atomic instruction based on Instrinsic::ID
/// and the expression node.
static RValue EmitBinaryAtomic(CodeGenFunction &CGF,
                               llvm::AtomicRMWInst::BinOp Kind,
                               const CallExpr *E) {
  QualType T = E->getType();
  assert(E->getArg(0)->getType()->isPointerType());
  assert(CGF.getContext().hasSameUnqualifiedType(T,
                                  E->getArg(0)->getType()->getPointeeType()));
  assert(CGF.getContext().hasSameUnqualifiedType(T, E->getArg(1)->getType()));

  llvm::Value *DestPtr = CGF.EmitScalarExpr(E->getArg(0));
  unsigned AddrSpace = DestPtr->getType()->getPointerAddressSpace();

  llvm::IntegerType *IntType =
    llvm::IntegerType::get(CGF.getLLVMContext(),
                           CGF.getContext().getTypeSize(T));
  llvm::Type *IntPtrType = IntType->getPointerTo(AddrSpace);

  llvm::Value *Args[2];
  Args[0] = CGF.Builder.CreateBitCast(DestPtr, IntPtrType);
  Args[1] = CGF.EmitScalarExpr(E->getArg(1));
  llvm::Type *ValueType = Args[1]->getType();
  Args[1] = EmitToInt(CGF, Args[1], T, IntType);

  llvm::Value *Result =
      CGF.Builder.CreateAtomicRMW(Kind, Args[0], Args[1],
                                  llvm::SequentiallyConsistent);
  Result = EmitFromInt(CGF, Result, T, ValueType);
  return RValue::get(Result);
}

/// Utility to insert an atomic instruction based Instrinsic::ID and
/// the expression node, where the return value is the result of the
/// operation.
static RValue EmitBinaryAtomicPost(CodeGenFunction &CGF,
                                   llvm::AtomicRMWInst::BinOp Kind,
                                   const CallExpr *E,
                                   Instruction::BinaryOps Op) {
  QualType T = E->getType();
  assert(E->getArg(0)->getType()->isPointerType());
  assert(CGF.getContext().hasSameUnqualifiedType(T,
                                  E->getArg(0)->getType()->getPointeeType()));
  assert(CGF.getContext().hasSameUnqualifiedType(T, E->getArg(1)->getType()));

  llvm::Value *DestPtr = CGF.EmitScalarExpr(E->getArg(0));
  unsigned AddrSpace = DestPtr->getType()->getPointerAddressSpace();

  llvm::IntegerType *IntType =
    llvm::IntegerType::get(CGF.getLLVMContext(),
                           CGF.getContext().getTypeSize(T));
  llvm::Type *IntPtrType = IntType->getPointerTo(AddrSpace);

  llvm::Value *Args[2];
  Args[1] = CGF.EmitScalarExpr(E->getArg(1));
  llvm::Type *ValueType = Args[1]->getType();
  Args[1] = EmitToInt(CGF, Args[1], T, IntType);
  Args[0] = CGF.Builder.CreateBitCast(DestPtr, IntPtrType);

  llvm::Value *Result =
      CGF.Builder.CreateAtomicRMW(Kind, Args[0], Args[1],
                                  llvm::SequentiallyConsistent);
  Result = CGF.Builder.CreateBinOp(Op, Result, Args[1]);
  Result = EmitFromInt(CGF, Result, T, ValueType);
  return RValue::get(Result);
}

/// EmitFAbs - Emit a call to fabs/fabsf/fabsl, depending on the type of ValTy,
/// which must be a scalar floating point type.
static Value *EmitFAbs(CodeGenFunction &CGF, Value *V, QualType ValTy) {
  const BuiltinType *ValTyP = ValTy->getAs<BuiltinType>();
  assert(ValTyP && "isn't scalar fp type!");

  StringRef FnName;
  switch (ValTyP->getKind()) {
  default: llvm_unreachable("Isn't a scalar fp type!");
  case BuiltinType::Float:      FnName = "fabsf"; break;
  case BuiltinType::Double:     FnName = "fabs"; break;
  case BuiltinType::LongDouble: FnName = "fabsl"; break;
  }

  // The prototype is something that takes and returns whatever V's type is.
  llvm::FunctionType *FT = llvm::FunctionType::get(V->getType(), V->getType(),
                                                   false);
  llvm::Value *Fn = CGF.CGM.CreateRuntimeFunction(FT, FnName);

  return CGF.EmitNounwindRuntimeCall(Fn, V, "abs");
}

static RValue emitLibraryCall(CodeGenFunction &CGF, const FunctionDecl *Fn,
                              const CallExpr *E, llvm::Value *calleeValue) {
  return CGF.EmitCall(E->getCallee()->getType(), calleeValue, E->getLocStart(),
                      ReturnValueSlot(), E->arg_begin(), E->arg_end(), Fn);
}

/// \brief Emit a call to llvm.{sadd,uadd,ssub,usub,smul,umul}.with.overflow.*
/// depending on IntrinsicID.
///
/// \arg CGF The current codegen function.
/// \arg IntrinsicID The ID for the Intrinsic we wish to generate.
/// \arg X The first argument to the llvm.*.with.overflow.*.
/// \arg Y The second argument to the llvm.*.with.overflow.*.
/// \arg Carry The carry returned by the llvm.*.with.overflow.*.
/// \returns The result (i.e. sum/product) returned by the intrinsic.
static llvm::Value *EmitOverflowIntrinsic(CodeGenFunction &CGF,
                                          const llvm::Intrinsic::ID IntrinsicID,
                                          llvm::Value *X, llvm::Value *Y,
                                          llvm::Value *&Carry) {
  // Make sure we have integers of the same width.
  assert(X->getType() == Y->getType() &&
         "Arguments must be the same type. (Did you forget to make sure both "
         "arguments have the same integer width?)");

  llvm::Value *Callee = CGF.CGM.getIntrinsic(IntrinsicID, X->getType());
  llvm::Value *Tmp = CGF.Builder.CreateCall2(Callee, X, Y);
  Carry = CGF.Builder.CreateExtractValue(Tmp, 1);
  return CGF.Builder.CreateExtractValue(Tmp, 0);
}

RValue CodeGenFunction::EmitBuiltinExpr(const FunctionDecl *FD,
                                        unsigned BuiltinID, const CallExpr *E) {
  // See if we can constant fold this builtin.  If so, don't emit it at all.
  Expr::EvalResult Result;
  if (E->EvaluateAsRValue(Result, CGM.getContext()) &&
      !Result.hasSideEffects()) {
    if (Result.Val.isInt())
      return RValue::get(llvm::ConstantInt::get(getLLVMContext(),
                                                Result.Val.getInt()));
    if (Result.Val.isFloat())
      return RValue::get(llvm::ConstantFP::get(getLLVMContext(),
                                               Result.Val.getFloat()));
  }

  switch (BuiltinID) {
  default: break;  // Handle intrinsics and libm functions below.
  case Builtin::BI__builtin___CFStringMakeConstantString:
  case Builtin::BI__builtin___NSStringMakeConstantString:
    return RValue::get(CGM.EmitConstantExpr(E, E->getType(), 0));
  case Builtin::BI__builtin_stdarg_start:
  case Builtin::BI__builtin_va_start:
  case Builtin::BI__builtin_va_end: {
    Value *ArgValue = EmitVAListRef(E->getArg(0));
    llvm::Type *DestType = Int8PtrTy;
    if (ArgValue->getType() != DestType)
      ArgValue = Builder.CreateBitCast(ArgValue, DestType,
                                       ArgValue->getName().data());

    Intrinsic::ID inst = (BuiltinID == Builtin::BI__builtin_va_end) ?
      Intrinsic::vaend : Intrinsic::vastart;
    return RValue::get(Builder.CreateCall(CGM.getIntrinsic(inst), ArgValue));
  }
  case Builtin::BI__builtin_va_copy: {
    Value *DstPtr = EmitVAListRef(E->getArg(0));
    Value *SrcPtr = EmitVAListRef(E->getArg(1));

    llvm::Type *Type = Int8PtrTy;

    DstPtr = Builder.CreateBitCast(DstPtr, Type);
    SrcPtr = Builder.CreateBitCast(SrcPtr, Type);
    return RValue::get(Builder.CreateCall2(CGM.getIntrinsic(Intrinsic::vacopy),
                                           DstPtr, SrcPtr));
  }
  case Builtin::BI__builtin_abs:
  case Builtin::BI__builtin_labs:
  case Builtin::BI__builtin_llabs: {
    Value *ArgValue = EmitScalarExpr(E->getArg(0));

    Value *NegOp = Builder.CreateNeg(ArgValue, "neg");
    Value *CmpResult =
    Builder.CreateICmpSGE(ArgValue,
                          llvm::Constant::getNullValue(ArgValue->getType()),
                                                            "abscond");
    Value *Result =
      Builder.CreateSelect(CmpResult, ArgValue, NegOp, "abs");

    return RValue::get(Result);
  }

  case Builtin::BI__builtin_conj:
  case Builtin::BI__builtin_conjf:
  case Builtin::BI__builtin_conjl: {
    ComplexPairTy ComplexVal = EmitComplexExpr(E->getArg(0));
    Value *Real = ComplexVal.first;
    Value *Imag = ComplexVal.second;
    Value *Zero =
      Imag->getType()->isFPOrFPVectorTy()
        ? llvm::ConstantFP::getZeroValueForNegation(Imag->getType())
        : llvm::Constant::getNullValue(Imag->getType());

    Imag = Builder.CreateFSub(Zero, Imag, "sub");
    return RValue::getComplex(std::make_pair(Real, Imag));
  }
  case Builtin::BI__builtin_creal:
  case Builtin::BI__builtin_crealf:
  case Builtin::BI__builtin_creall:
  case Builtin::BIcreal:
  case Builtin::BIcrealf:
  case Builtin::BIcreall: {
    ComplexPairTy ComplexVal = EmitComplexExpr(E->getArg(0));
    return RValue::get(ComplexVal.first);
  }

  case Builtin::BI__builtin_cimag:
  case Builtin::BI__builtin_cimagf:
  case Builtin::BI__builtin_cimagl:
  case Builtin::BIcimag:
  case Builtin::BIcimagf:
  case Builtin::BIcimagl: {
    ComplexPairTy ComplexVal = EmitComplexExpr(E->getArg(0));
    return RValue::get(ComplexVal.second);
  }

  case Builtin::BI__builtin_ctzs:
  case Builtin::BI__builtin_ctz:
  case Builtin::BI__builtin_ctzl:
  case Builtin::BI__builtin_ctzll: {
    Value *ArgValue = EmitScalarExpr(E->getArg(0));

    llvm::Type *ArgType = ArgValue->getType();
    Value *F = CGM.getIntrinsic(Intrinsic::cttz, ArgType);

    llvm::Type *ResultType = ConvertType(E->getType());
    Value *ZeroUndef = Builder.getInt1(getTarget().isCLZForZeroUndef());
    Value *Result = Builder.CreateCall2(F, ArgValue, ZeroUndef);
    if (Result->getType() != ResultType)
      Result = Builder.CreateIntCast(Result, ResultType, /*isSigned*/true,
                                     "cast");
    return RValue::get(Result);
  }
  case Builtin::BI__builtin_clzs:
  case Builtin::BI__builtin_clz:
  case Builtin::BI__builtin_clzl:
  case Builtin::BI__builtin_clzll: {
    Value *ArgValue = EmitScalarExpr(E->getArg(0));

    llvm::Type *ArgType = ArgValue->getType();
    Value *F = CGM.getIntrinsic(Intrinsic::ctlz, ArgType);

    llvm::Type *ResultType = ConvertType(E->getType());
    Value *ZeroUndef = Builder.getInt1(getTarget().isCLZForZeroUndef());
    Value *Result = Builder.CreateCall2(F, ArgValue, ZeroUndef);
    if (Result->getType() != ResultType)
      Result = Builder.CreateIntCast(Result, ResultType, /*isSigned*/true,
                                     "cast");
    return RValue::get(Result);
  }
  case Builtin::BI__builtin_ffs:
  case Builtin::BI__builtin_ffsl:
  case Builtin::BI__builtin_ffsll: {
    // ffs(x) -> x ? cttz(x) + 1 : 0
    Value *ArgValue = EmitScalarExpr(E->getArg(0));

    llvm::Type *ArgType = ArgValue->getType();
    Value *F = CGM.getIntrinsic(Intrinsic::cttz, ArgType);

    llvm::Type *ResultType = ConvertType(E->getType());
    Value *Tmp = Builder.CreateAdd(Builder.CreateCall2(F, ArgValue,
                                                       Builder.getTrue()),
                                   llvm::ConstantInt::get(ArgType, 1));
    Value *Zero = llvm::Constant::getNullValue(ArgType);
    Value *IsZero = Builder.CreateICmpEQ(ArgValue, Zero, "iszero");
    Value *Result = Builder.CreateSelect(IsZero, Zero, Tmp, "ffs");
    if (Result->getType() != ResultType)
      Result = Builder.CreateIntCast(Result, ResultType, /*isSigned*/true,
                                     "cast");
    return RValue::get(Result);
  }
  case Builtin::BI__builtin_parity:
  case Builtin::BI__builtin_parityl:
  case Builtin::BI__builtin_parityll: {
    // parity(x) -> ctpop(x) & 1
    Value *ArgValue = EmitScalarExpr(E->getArg(0));

    llvm::Type *ArgType = ArgValue->getType();
    Value *F = CGM.getIntrinsic(Intrinsic::ctpop, ArgType);

    llvm::Type *ResultType = ConvertType(E->getType());
    Value *Tmp = Builder.CreateCall(F, ArgValue);
    Value *Result = Builder.CreateAnd(Tmp, llvm::ConstantInt::get(ArgType, 1));
    if (Result->getType() != ResultType)
      Result = Builder.CreateIntCast(Result, ResultType, /*isSigned*/true,
                                     "cast");
    return RValue::get(Result);
  }
  case Builtin::BI__builtin_popcount:
  case Builtin::BI__builtin_popcountl:
  case Builtin::BI__builtin_popcountll: {
    Value *ArgValue = EmitScalarExpr(E->getArg(0));

    llvm::Type *ArgType = ArgValue->getType();
    Value *F = CGM.getIntrinsic(Intrinsic::ctpop, ArgType);

    llvm::Type *ResultType = ConvertType(E->getType());
    Value *Result = Builder.CreateCall(F, ArgValue);
    if (Result->getType() != ResultType)
      Result = Builder.CreateIntCast(Result, ResultType, /*isSigned*/true,
                                     "cast");
    return RValue::get(Result);
  }
  case Builtin::BI__builtin_expect: {
    Value *ArgValue = EmitScalarExpr(E->getArg(0));
    llvm::Type *ArgType = ArgValue->getType();

    Value *FnExpect = CGM.getIntrinsic(Intrinsic::expect, ArgType);
    Value *ExpectedValue = EmitScalarExpr(E->getArg(1));

    Value *Result = Builder.CreateCall2(FnExpect, ArgValue, ExpectedValue,
                                        "expval");
    return RValue::get(Result);
  }
  case Builtin::BI__builtin_bswap16:
  case Builtin::BI__builtin_bswap32:
  case Builtin::BI__builtin_bswap64: {
    Value *ArgValue = EmitScalarExpr(E->getArg(0));
    llvm::Type *ArgType = ArgValue->getType();
    Value *F = CGM.getIntrinsic(Intrinsic::bswap, ArgType);
    return RValue::get(Builder.CreateCall(F, ArgValue));
  }
  case Builtin::BI__builtin_object_size: {
    // We rely on constant folding to deal with expressions with side effects.
    assert(!E->getArg(0)->HasSideEffects(getContext()) &&
           "should have been constant folded");

    // We pass this builtin onto the optimizer so that it can
    // figure out the object size in more complex cases.
    llvm::Type *ResType = ConvertType(E->getType());

    // LLVM only supports 0 and 2, make sure that we pass along that
    // as a boolean.
    Value *Ty = EmitScalarExpr(E->getArg(1));
    ConstantInt *CI = dyn_cast<ConstantInt>(Ty);
    assert(CI);
    uint64_t val = CI->getZExtValue();
    CI = ConstantInt::get(Builder.getInt1Ty(), (val & 0x2) >> 1);
    // FIXME: Get right address space.
    llvm::Type *Tys[] = { ResType, Builder.getInt8PtrTy(0) };
    Value *F = CGM.getIntrinsic(Intrinsic::objectsize, Tys);
    return RValue::get(Builder.CreateCall2(F, EmitScalarExpr(E->getArg(0)),CI));
  }
  case Builtin::BI__builtin_prefetch: {
    Value *Locality, *RW, *Address = EmitScalarExpr(E->getArg(0));
    // FIXME: Technically these constants should of type 'int', yes?
    RW = (E->getNumArgs() > 1) ? EmitScalarExpr(E->getArg(1)) :
      llvm::ConstantInt::get(Int32Ty, 0);
    Locality = (E->getNumArgs() > 2) ? EmitScalarExpr(E->getArg(2)) :
      llvm::ConstantInt::get(Int32Ty, 3);
    Value *Data = llvm::ConstantInt::get(Int32Ty, 1);
    Value *F = CGM.getIntrinsic(Intrinsic::prefetch);
    return RValue::get(Builder.CreateCall4(F, Address, RW, Locality, Data));
  }
  case Builtin::BI__builtin_readcyclecounter: {
    Value *F = CGM.getIntrinsic(Intrinsic::readcyclecounter);
    return RValue::get(Builder.CreateCall(F));
  }
  case Builtin::BI__builtin_trap: {
    Value *F = CGM.getIntrinsic(Intrinsic::trap);
    return RValue::get(Builder.CreateCall(F));
  }
  case Builtin::BI__debugbreak: {
    Value *F = CGM.getIntrinsic(Intrinsic::debugtrap);
    return RValue::get(Builder.CreateCall(F));
  }
  case Builtin::BI__builtin_unreachable: {
    if (SanOpts->Unreachable)
      EmitCheck(Builder.getFalse(), "builtin_unreachable",
                EmitCheckSourceLocation(E->getExprLoc()),
                ArrayRef<llvm::Value *>(), CRK_Unrecoverable);
    else
      Builder.CreateUnreachable();

    // We do need to preserve an insertion point.
    EmitBlock(createBasicBlock("unreachable.cont"));

    return RValue::get(0);
  }

  case Builtin::BI__builtin_powi:
  case Builtin::BI__builtin_powif:
  case Builtin::BI__builtin_powil: {
    Value *Base = EmitScalarExpr(E->getArg(0));
    Value *Exponent = EmitScalarExpr(E->getArg(1));
    llvm::Type *ArgType = Base->getType();
    Value *F = CGM.getIntrinsic(Intrinsic::powi, ArgType);
    return RValue::get(Builder.CreateCall2(F, Base, Exponent));
  }

  case Builtin::BI__builtin_isgreater:
  case Builtin::BI__builtin_isgreaterequal:
  case Builtin::BI__builtin_isless:
  case Builtin::BI__builtin_islessequal:
  case Builtin::BI__builtin_islessgreater:
  case Builtin::BI__builtin_isunordered: {
    // Ordered comparisons: we know the arguments to these are matching scalar
    // floating point values.
    Value *LHS = EmitScalarExpr(E->getArg(0));
    Value *RHS = EmitScalarExpr(E->getArg(1));

    switch (BuiltinID) {
    default: llvm_unreachable("Unknown ordered comparison");
    case Builtin::BI__builtin_isgreater:
      LHS = Builder.CreateFCmpOGT(LHS, RHS, "cmp");
      break;
    case Builtin::BI__builtin_isgreaterequal:
      LHS = Builder.CreateFCmpOGE(LHS, RHS, "cmp");
      break;
    case Builtin::BI__builtin_isless:
      LHS = Builder.CreateFCmpOLT(LHS, RHS, "cmp");
      break;
    case Builtin::BI__builtin_islessequal:
      LHS = Builder.CreateFCmpOLE(LHS, RHS, "cmp");
      break;
    case Builtin::BI__builtin_islessgreater:
      LHS = Builder.CreateFCmpONE(LHS, RHS, "cmp");
      break;
    case Builtin::BI__builtin_isunordered:
      LHS = Builder.CreateFCmpUNO(LHS, RHS, "cmp");
      break;
    }
    // ZExt bool to int type.
    return RValue::get(Builder.CreateZExt(LHS, ConvertType(E->getType())));
  }
  case Builtin::BI__builtin_isnan: {
    Value *V = EmitScalarExpr(E->getArg(0));
    V = Builder.CreateFCmpUNO(V, V, "cmp");
    return RValue::get(Builder.CreateZExt(V, ConvertType(E->getType())));
  }

  case Builtin::BI__builtin_isinf: {
    // isinf(x) --> fabs(x) == infinity
    Value *V = EmitScalarExpr(E->getArg(0));
    V = EmitFAbs(*this, V, E->getArg(0)->getType());

    V = Builder.CreateFCmpOEQ(V, ConstantFP::getInfinity(V->getType()),"isinf");
    return RValue::get(Builder.CreateZExt(V, ConvertType(E->getType())));
  }

  // TODO: BI__builtin_isinf_sign
  //   isinf_sign(x) -> isinf(x) ? (signbit(x) ? -1 : 1) : 0

  case Builtin::BI__builtin_isnormal: {
    // isnormal(x) --> x == x && fabsf(x) < infinity && fabsf(x) >= float_min
    Value *V = EmitScalarExpr(E->getArg(0));
    Value *Eq = Builder.CreateFCmpOEQ(V, V, "iseq");

    Value *Abs = EmitFAbs(*this, V, E->getArg(0)->getType());
    Value *IsLessThanInf =
      Builder.CreateFCmpULT(Abs, ConstantFP::getInfinity(V->getType()),"isinf");
    APFloat Smallest = APFloat::getSmallestNormalized(
                   getContext().getFloatTypeSemantics(E->getArg(0)->getType()));
    Value *IsNormal =
      Builder.CreateFCmpUGE(Abs, ConstantFP::get(V->getContext(), Smallest),
                            "isnormal");
    V = Builder.CreateAnd(Eq, IsLessThanInf, "and");
    V = Builder.CreateAnd(V, IsNormal, "and");
    return RValue::get(Builder.CreateZExt(V, ConvertType(E->getType())));
  }

  case Builtin::BI__builtin_isfinite: {
    // isfinite(x) --> x == x && fabs(x) != infinity;
    Value *V = EmitScalarExpr(E->getArg(0));
    Value *Eq = Builder.CreateFCmpOEQ(V, V, "iseq");

    Value *Abs = EmitFAbs(*this, V, E->getArg(0)->getType());
    Value *IsNotInf =
      Builder.CreateFCmpUNE(Abs, ConstantFP::getInfinity(V->getType()),"isinf");

    V = Builder.CreateAnd(Eq, IsNotInf, "and");
    return RValue::get(Builder.CreateZExt(V, ConvertType(E->getType())));
  }

  case Builtin::BI__builtin_fpclassify: {
    Value *V = EmitScalarExpr(E->getArg(5));
    llvm::Type *Ty = ConvertType(E->getArg(5)->getType());

    // Create Result
    BasicBlock *Begin = Builder.GetInsertBlock();
    BasicBlock *End = createBasicBlock("fpclassify_end", this->CurFn);
    Builder.SetInsertPoint(End);
    PHINode *Result =
      Builder.CreatePHI(ConvertType(E->getArg(0)->getType()), 4,
                        "fpclassify_result");

    // if (V==0) return FP_ZERO
    Builder.SetInsertPoint(Begin);
    Value *IsZero = Builder.CreateFCmpOEQ(V, Constant::getNullValue(Ty),
                                          "iszero");
    Value *ZeroLiteral = EmitScalarExpr(E->getArg(4));
    BasicBlock *NotZero = createBasicBlock("fpclassify_not_zero", this->CurFn);
    Builder.CreateCondBr(IsZero, End, NotZero);
    Result->addIncoming(ZeroLiteral, Begin);

    // if (V != V) return FP_NAN
    Builder.SetInsertPoint(NotZero);
    Value *IsNan = Builder.CreateFCmpUNO(V, V, "cmp");
    Value *NanLiteral = EmitScalarExpr(E->getArg(0));
    BasicBlock *NotNan = createBasicBlock("fpclassify_not_nan", this->CurFn);
    Builder.CreateCondBr(IsNan, End, NotNan);
    Result->addIncoming(NanLiteral, NotZero);

    // if (fabs(V) == infinity) return FP_INFINITY
    Builder.SetInsertPoint(NotNan);
    Value *VAbs = EmitFAbs(*this, V, E->getArg(5)->getType());
    Value *IsInf =
      Builder.CreateFCmpOEQ(VAbs, ConstantFP::getInfinity(V->getType()),
                            "isinf");
    Value *InfLiteral = EmitScalarExpr(E->getArg(1));
    BasicBlock *NotInf = createBasicBlock("fpclassify_not_inf", this->CurFn);
    Builder.CreateCondBr(IsInf, End, NotInf);
    Result->addIncoming(InfLiteral, NotNan);

    // if (fabs(V) >= MIN_NORMAL) return FP_NORMAL else FP_SUBNORMAL
    Builder.SetInsertPoint(NotInf);
    APFloat Smallest = APFloat::getSmallestNormalized(
        getContext().getFloatTypeSemantics(E->getArg(5)->getType()));
    Value *IsNormal =
      Builder.CreateFCmpUGE(VAbs, ConstantFP::get(V->getContext(), Smallest),
                            "isnormal");
    Value *NormalResult =
      Builder.CreateSelect(IsNormal, EmitScalarExpr(E->getArg(2)),
                           EmitScalarExpr(E->getArg(3)));
    Builder.CreateBr(End);
    Result->addIncoming(NormalResult, NotInf);

    // return Result
    Builder.SetInsertPoint(End);
    return RValue::get(Result);
  }

  case Builtin::BIalloca:
  case Builtin::BI_alloca:
  case Builtin::BI__builtin_alloca: {
    Value *Size = EmitScalarExpr(E->getArg(0));
    return RValue::get(Builder.CreateAlloca(Builder.getInt8Ty(), Size));
  }
  case Builtin::BIbzero:
  case Builtin::BI__builtin_bzero: {
    std::pair<llvm::Value*, unsigned> Dest =
        EmitPointerWithAlignment(E->getArg(0));
    Value *SizeVal = EmitScalarExpr(E->getArg(1));
    Builder.CreateMemSet(Dest.first, Builder.getInt8(0), SizeVal,
                         Dest.second, false);
    return RValue::get(Dest.first);
  }
  case Builtin::BImemcpy:
  case Builtin::BI__builtin_memcpy: {
    std::pair<llvm::Value*, unsigned> Dest =
        EmitPointerWithAlignment(E->getArg(0));
    std::pair<llvm::Value*, unsigned> Src =
        EmitPointerWithAlignment(E->getArg(1));
    Value *SizeVal = EmitScalarExpr(E->getArg(2));
    unsigned Align = std::min(Dest.second, Src.second);
    Builder.CreateMemCpy(Dest.first, Src.first, SizeVal, Align, false);
    return RValue::get(Dest.first);
  }

  case Builtin::BI__builtin___memcpy_chk: {
    // fold __builtin_memcpy_chk(x, y, cst1, cst2) to memcpy iff cst1<=cst2.
    llvm::APSInt Size, DstSize;
    if (!E->getArg(2)->EvaluateAsInt(Size, CGM.getContext()) ||
        !E->getArg(3)->EvaluateAsInt(DstSize, CGM.getContext()))
      break;
    if (Size.ugt(DstSize))
      break;
    std::pair<llvm::Value*, unsigned> Dest =
        EmitPointerWithAlignment(E->getArg(0));
    std::pair<llvm::Value*, unsigned> Src =
        EmitPointerWithAlignment(E->getArg(1));
    Value *SizeVal = llvm::ConstantInt::get(Builder.getContext(), Size);
    unsigned Align = std::min(Dest.second, Src.second);
    Builder.CreateMemCpy(Dest.first, Src.first, SizeVal, Align, false);
    return RValue::get(Dest.first);
  }

  case Builtin::BI__builtin_objc_memmove_collectable: {
    Value *Address = EmitScalarExpr(E->getArg(0));
    Value *SrcAddr = EmitScalarExpr(E->getArg(1));
    Value *SizeVal = EmitScalarExpr(E->getArg(2));
    CGM.getObjCRuntime().EmitGCMemmoveCollectable(*this,
                                                  Address, SrcAddr, SizeVal);
    return RValue::get(Address);
  }

  case Builtin::BI__builtin___memmove_chk: {
    // fold __builtin_memmove_chk(x, y, cst1, cst2) to memmove iff cst1<=cst2.
    llvm::APSInt Size, DstSize;
    if (!E->getArg(2)->EvaluateAsInt(Size, CGM.getContext()) ||
        !E->getArg(3)->EvaluateAsInt(DstSize, CGM.getContext()))
      break;
    if (Size.ugt(DstSize))
      break;
    std::pair<llvm::Value*, unsigned> Dest =
        EmitPointerWithAlignment(E->getArg(0));
    std::pair<llvm::Value*, unsigned> Src =
        EmitPointerWithAlignment(E->getArg(1));
    Value *SizeVal = llvm::ConstantInt::get(Builder.getContext(), Size);
    unsigned Align = std::min(Dest.second, Src.second);
    Builder.CreateMemMove(Dest.first, Src.first, SizeVal, Align, false);
    return RValue::get(Dest.first);
  }

  case Builtin::BImemmove:
  case Builtin::BI__builtin_memmove: {
    std::pair<llvm::Value*, unsigned> Dest =
        EmitPointerWithAlignment(E->getArg(0));
    std::pair<llvm::Value*, unsigned> Src =
        EmitPointerWithAlignment(E->getArg(1));
    Value *SizeVal = EmitScalarExpr(E->getArg(2));
    unsigned Align = std::min(Dest.second, Src.second);
    Builder.CreateMemMove(Dest.first, Src.first, SizeVal, Align, false);
    return RValue::get(Dest.first);
  }
  case Builtin::BImemset:
  case Builtin::BI__builtin_memset: {
    std::pair<llvm::Value*, unsigned> Dest =
        EmitPointerWithAlignment(E->getArg(0));
    Value *ByteVal = Builder.CreateTrunc(EmitScalarExpr(E->getArg(1)),
                                         Builder.getInt8Ty());
    Value *SizeVal = EmitScalarExpr(E->getArg(2));
    Builder.CreateMemSet(Dest.first, ByteVal, SizeVal, Dest.second, false);
    return RValue::get(Dest.first);
  }
  case Builtin::BI__builtin___memset_chk: {
    // fold __builtin_memset_chk(x, y, cst1, cst2) to memset iff cst1<=cst2.
    llvm::APSInt Size, DstSize;
    if (!E->getArg(2)->EvaluateAsInt(Size, CGM.getContext()) ||
        !E->getArg(3)->EvaluateAsInt(DstSize, CGM.getContext()))
      break;
    if (Size.ugt(DstSize))
      break;
    std::pair<llvm::Value*, unsigned> Dest =
        EmitPointerWithAlignment(E->getArg(0));
    Value *ByteVal = Builder.CreateTrunc(EmitScalarExpr(E->getArg(1)),
                                         Builder.getInt8Ty());
    Value *SizeVal = llvm::ConstantInt::get(Builder.getContext(), Size);
    Builder.CreateMemSet(Dest.first, ByteVal, SizeVal, Dest.second, false);
    return RValue::get(Dest.first);
  }
  case Builtin::BI__builtin_dwarf_cfa: {
    // The offset in bytes from the first argument to the CFA.
    //
    // Why on earth is this in the frontend?  Is there any reason at
    // all that the backend can't reasonably determine this while
    // lowering llvm.eh.dwarf.cfa()?
    //
    // TODO: If there's a satisfactory reason, add a target hook for
    // this instead of hard-coding 0, which is correct for most targets.
    int32_t Offset = 0;

    Value *F = CGM.getIntrinsic(Intrinsic::eh_dwarf_cfa);
    return RValue::get(Builder.CreateCall(F,
                                      llvm::ConstantInt::get(Int32Ty, Offset)));
  }
  case Builtin::BI__builtin_return_address: {
    Value *Depth = EmitScalarExpr(E->getArg(0));
    Depth = Builder.CreateIntCast(Depth, Int32Ty, false);
    Value *F = CGM.getIntrinsic(Intrinsic::returnaddress);
    return RValue::get(Builder.CreateCall(F, Depth));
  }
  case Builtin::BI__builtin_frame_address: {
    Value *Depth = EmitScalarExpr(E->getArg(0));
    Depth = Builder.CreateIntCast(Depth, Int32Ty, false);
    Value *F = CGM.getIntrinsic(Intrinsic::frameaddress);
    return RValue::get(Builder.CreateCall(F, Depth));
  }
  case Builtin::BI__builtin_extract_return_addr: {
    Value *Address = EmitScalarExpr(E->getArg(0));
    Value *Result = getTargetHooks().decodeReturnAddress(*this, Address);
    return RValue::get(Result);
  }
  case Builtin::BI__builtin_frob_return_addr: {
    Value *Address = EmitScalarExpr(E->getArg(0));
    Value *Result = getTargetHooks().encodeReturnAddress(*this, Address);
    return RValue::get(Result);
  }
  case Builtin::BI__builtin_dwarf_sp_column: {
    llvm::IntegerType *Ty
      = cast<llvm::IntegerType>(ConvertType(E->getType()));
    int Column = getTargetHooks().getDwarfEHStackPointer(CGM);
    if (Column == -1) {
      CGM.ErrorUnsupported(E, "__builtin_dwarf_sp_column");
      return RValue::get(llvm::UndefValue::get(Ty));
    }
    return RValue::get(llvm::ConstantInt::get(Ty, Column, true));
  }
  case Builtin::BI__builtin_init_dwarf_reg_size_table: {
    Value *Address = EmitScalarExpr(E->getArg(0));
    if (getTargetHooks().initDwarfEHRegSizeTable(*this, Address))
      CGM.ErrorUnsupported(E, "__builtin_init_dwarf_reg_size_table");
    return RValue::get(llvm::UndefValue::get(ConvertType(E->getType())));
  }
  case Builtin::BI__builtin_eh_return: {
    Value *Int = EmitScalarExpr(E->getArg(0));
    Value *Ptr = EmitScalarExpr(E->getArg(1));

    llvm::IntegerType *IntTy = cast<llvm::IntegerType>(Int->getType());
    assert((IntTy->getBitWidth() == 32 || IntTy->getBitWidth() == 64) &&
           "LLVM's __builtin_eh_return only supports 32- and 64-bit variants");
    Value *F = CGM.getIntrinsic(IntTy->getBitWidth() == 32
                                  ? Intrinsic::eh_return_i32
                                  : Intrinsic::eh_return_i64);
    Builder.CreateCall2(F, Int, Ptr);
    Builder.CreateUnreachable();

    // We do need to preserve an insertion point.
    EmitBlock(createBasicBlock("builtin_eh_return.cont"));

    return RValue::get(0);
  }
  case Builtin::BI__builtin_unwind_init: {
    Value *F = CGM.getIntrinsic(Intrinsic::eh_unwind_init);
    return RValue::get(Builder.CreateCall(F));
  }
  case Builtin::BI__builtin_extend_pointer: {
    // Extends a pointer to the size of an _Unwind_Word, which is
    // uint64_t on all platforms.  Generally this gets poked into a
    // register and eventually used as an address, so if the
    // addressing registers are wider than pointers and the platform
    // doesn't implicitly ignore high-order bits when doing
    // addressing, we need to make sure we zext / sext based on
    // the platform's expectations.
    //
    // See: http://gcc.gnu.org/ml/gcc-bugs/2002-02/msg00237.html

    // Cast the pointer to intptr_t.
    Value *Ptr = EmitScalarExpr(E->getArg(0));
    Value *Result = Builder.CreatePtrToInt(Ptr, IntPtrTy, "extend.cast");

    // If that's 64 bits, we're done.
    if (IntPtrTy->getBitWidth() == 64)
      return RValue::get(Result);

    // Otherwise, ask the codegen data what to do.
    if (getTargetHooks().extendPointerWithSExt())
      return RValue::get(Builder.CreateSExt(Result, Int64Ty, "extend.sext"));
    else
      return RValue::get(Builder.CreateZExt(Result, Int64Ty, "extend.zext"));
  }
  case Builtin::BI__builtin_setjmp: {
    // Buffer is a void**.
    Value *Buf = EmitScalarExpr(E->getArg(0));

    // Store the frame pointer to the setjmp buffer.
    Value *FrameAddr =
      Builder.CreateCall(CGM.getIntrinsic(Intrinsic::frameaddress),
                         ConstantInt::get(Int32Ty, 0));
    Builder.CreateStore(FrameAddr, Buf);

    // Store the stack pointer to the setjmp buffer.
    Value *StackAddr =
      Builder.CreateCall(CGM.getIntrinsic(Intrinsic::stacksave));
    Value *StackSaveSlot =
      Builder.CreateGEP(Buf, ConstantInt::get(Int32Ty, 2));
    Builder.CreateStore(StackAddr, StackSaveSlot);

    // Call LLVM's EH setjmp, which is lightweight.
    Value *F = CGM.getIntrinsic(Intrinsic::eh_sjlj_setjmp);
    Buf = Builder.CreateBitCast(Buf, Int8PtrTy);
    return RValue::get(Builder.CreateCall(F, Buf));
  }
  case Builtin::BI__builtin_longjmp: {
    Value *Buf = EmitScalarExpr(E->getArg(0));
    Buf = Builder.CreateBitCast(Buf, Int8PtrTy);

    // Call LLVM's EH longjmp, which is lightweight.
    Builder.CreateCall(CGM.getIntrinsic(Intrinsic::eh_sjlj_longjmp), Buf);

    // longjmp doesn't return; mark this as unreachable.
    Builder.CreateUnreachable();

    // We do need to preserve an insertion point.
    EmitBlock(createBasicBlock("longjmp.cont"));

    return RValue::get(0);
  }
  case Builtin::BI__sync_fetch_and_add:
  case Builtin::BI__sync_fetch_and_sub:
  case Builtin::BI__sync_fetch_and_or:
  case Builtin::BI__sync_fetch_and_and:
  case Builtin::BI__sync_fetch_and_xor:
  case Builtin::BI__sync_add_and_fetch:
  case Builtin::BI__sync_sub_and_fetch:
  case Builtin::BI__sync_and_and_fetch:
  case Builtin::BI__sync_or_and_fetch:
  case Builtin::BI__sync_xor_and_fetch:
  case Builtin::BI__sync_val_compare_and_swap:
  case Builtin::BI__sync_bool_compare_and_swap:
  case Builtin::BI__sync_lock_test_and_set:
  case Builtin::BI__sync_lock_release:
  case Builtin::BI__sync_swap:
    llvm_unreachable("Shouldn't make it through sema");
  case Builtin::BI__sync_fetch_and_add_1:
  case Builtin::BI__sync_fetch_and_add_2:
  case Builtin::BI__sync_fetch_and_add_4:
  case Builtin::BI__sync_fetch_and_add_8:
  case Builtin::BI__sync_fetch_and_add_16:
    return EmitBinaryAtomic(*this, llvm::AtomicRMWInst::Add, E);
  case Builtin::BI__sync_fetch_and_sub_1:
  case Builtin::BI__sync_fetch_and_sub_2:
  case Builtin::BI__sync_fetch_and_sub_4:
  case Builtin::BI__sync_fetch_and_sub_8:
  case Builtin::BI__sync_fetch_and_sub_16:
    return EmitBinaryAtomic(*this, llvm::AtomicRMWInst::Sub, E);
  case Builtin::BI__sync_fetch_and_or_1:
  case Builtin::BI__sync_fetch_and_or_2:
  case Builtin::BI__sync_fetch_and_or_4:
  case Builtin::BI__sync_fetch_and_or_8:
  case Builtin::BI__sync_fetch_and_or_16:
    return EmitBinaryAtomic(*this, llvm::AtomicRMWInst::Or, E);
  case Builtin::BI__sync_fetch_and_and_1:
  case Builtin::BI__sync_fetch_and_and_2:
  case Builtin::BI__sync_fetch_and_and_4:
  case Builtin::BI__sync_fetch_and_and_8:
  case Builtin::BI__sync_fetch_and_and_16:
    return EmitBinaryAtomic(*this, llvm::AtomicRMWInst::And, E);
  case Builtin::BI__sync_fetch_and_xor_1:
  case Builtin::BI__sync_fetch_and_xor_2:
  case Builtin::BI__sync_fetch_and_xor_4:
  case Builtin::BI__sync_fetch_and_xor_8:
  case Builtin::BI__sync_fetch_and_xor_16:
    return EmitBinaryAtomic(*this, llvm::AtomicRMWInst::Xor, E);

  // Clang extensions: not overloaded yet.
  case Builtin::BI__sync_fetch_and_min:
    return EmitBinaryAtomic(*this, llvm::AtomicRMWInst::Min, E);
  case Builtin::BI__sync_fetch_and_max:
    return EmitBinaryAtomic(*this, llvm::AtomicRMWInst::Max, E);
  case Builtin::BI__sync_fetch_and_umin:
    return EmitBinaryAtomic(*this, llvm::AtomicRMWInst::UMin, E);
  case Builtin::BI__sync_fetch_and_umax:
    return EmitBinaryAtomic(*this, llvm::AtomicRMWInst::UMax, E);

  case Builtin::BI__sync_add_and_fetch_1:
  case Builtin::BI__sync_add_and_fetch_2:
  case Builtin::BI__sync_add_and_fetch_4:
  case Builtin::BI__sync_add_and_fetch_8:
  case Builtin::BI__sync_add_and_fetch_16:
    return EmitBinaryAtomicPost(*this, llvm::AtomicRMWInst::Add, E,
                                llvm::Instruction::Add);
  case Builtin::BI__sync_sub_and_fetch_1:
  case Builtin::BI__sync_sub_and_fetch_2:
  case Builtin::BI__sync_sub_and_fetch_4:
  case Builtin::BI__sync_sub_and_fetch_8:
  case Builtin::BI__sync_sub_and_fetch_16:
    return EmitBinaryAtomicPost(*this, llvm::AtomicRMWInst::Sub, E,
                                llvm::Instruction::Sub);
  case Builtin::BI__sync_and_and_fetch_1:
  case Builtin::BI__sync_and_and_fetch_2:
  case Builtin::BI__sync_and_and_fetch_4:
  case Builtin::BI__sync_and_and_fetch_8:
  case Builtin::BI__sync_and_and_fetch_16:
    return EmitBinaryAtomicPost(*this, llvm::AtomicRMWInst::And, E,
                                llvm::Instruction::And);
  case Builtin::BI__sync_or_and_fetch_1:
  case Builtin::BI__sync_or_and_fetch_2:
  case Builtin::BI__sync_or_and_fetch_4:
  case Builtin::BI__sync_or_and_fetch_8:
  case Builtin::BI__sync_or_and_fetch_16:
    return EmitBinaryAtomicPost(*this, llvm::AtomicRMWInst::Or, E,
                                llvm::Instruction::Or);
  case Builtin::BI__sync_xor_and_fetch_1:
  case Builtin::BI__sync_xor_and_fetch_2:
  case Builtin::BI__sync_xor_and_fetch_4:
  case Builtin::BI__sync_xor_and_fetch_8:
  case Builtin::BI__sync_xor_and_fetch_16:
    return EmitBinaryAtomicPost(*this, llvm::AtomicRMWInst::Xor, E,
                                llvm::Instruction::Xor);

  case Builtin::BI__sync_val_compare_and_swap_1:
  case Builtin::BI__sync_val_compare_and_swap_2:
  case Builtin::BI__sync_val_compare_and_swap_4:
  case Builtin::BI__sync_val_compare_and_swap_8:
  case Builtin::BI__sync_val_compare_and_swap_16: {
    QualType T = E->getType();
    llvm::Value *DestPtr = EmitScalarExpr(E->getArg(0));
    unsigned AddrSpace = DestPtr->getType()->getPointerAddressSpace();

    llvm::IntegerType *IntType =
      llvm::IntegerType::get(getLLVMContext(),
                             getContext().getTypeSize(T));
    llvm::Type *IntPtrType = IntType->getPointerTo(AddrSpace);

    Value *Args[3];
    Args[0] = Builder.CreateBitCast(DestPtr, IntPtrType);
    Args[1] = EmitScalarExpr(E->getArg(1));
    llvm::Type *ValueType = Args[1]->getType();
    Args[1] = EmitToInt(*this, Args[1], T, IntType);
    Args[2] = EmitToInt(*this, EmitScalarExpr(E->getArg(2)), T, IntType);

    Value *Result = Builder.CreateAtomicCmpXchg(Args[0], Args[1], Args[2],
                                                llvm::SequentiallyConsistent);
    Result = EmitFromInt(*this, Result, T, ValueType);
    return RValue::get(Result);
  }

  case Builtin::BI__sync_bool_compare_and_swap_1:
  case Builtin::BI__sync_bool_compare_and_swap_2:
  case Builtin::BI__sync_bool_compare_and_swap_4:
  case Builtin::BI__sync_bool_compare_and_swap_8:
  case Builtin::BI__sync_bool_compare_and_swap_16: {
    QualType T = E->getArg(1)->getType();
    llvm::Value *DestPtr = EmitScalarExpr(E->getArg(0));
    unsigned AddrSpace = DestPtr->getType()->getPointerAddressSpace();

    llvm::IntegerType *IntType =
      llvm::IntegerType::get(getLLVMContext(),
                             getContext().getTypeSize(T));
    llvm::Type *IntPtrType = IntType->getPointerTo(AddrSpace);

    Value *Args[3];
    Args[0] = Builder.CreateBitCast(DestPtr, IntPtrType);
    Args[1] = EmitToInt(*this, EmitScalarExpr(E->getArg(1)), T, IntType);
    Args[2] = EmitToInt(*this, EmitScalarExpr(E->getArg(2)), T, IntType);

    Value *OldVal = Args[1];
    Value *PrevVal = Builder.CreateAtomicCmpXchg(Args[0], Args[1], Args[2],
                                                 llvm::SequentiallyConsistent);
    Value *Result = Builder.CreateICmpEQ(PrevVal, OldVal);
    // zext bool to int.
    Result = Builder.CreateZExt(Result, ConvertType(E->getType()));
    return RValue::get(Result);
  }

  case Builtin::BI__sync_swap_1:
  case Builtin::BI__sync_swap_2:
  case Builtin::BI__sync_swap_4:
  case Builtin::BI__sync_swap_8:
  case Builtin::BI__sync_swap_16:
    return EmitBinaryAtomic(*this, llvm::AtomicRMWInst::Xchg, E);

  case Builtin::BI__sync_lock_test_and_set_1:
  case Builtin::BI__sync_lock_test_and_set_2:
  case Builtin::BI__sync_lock_test_and_set_4:
  case Builtin::BI__sync_lock_test_and_set_8:
  case Builtin::BI__sync_lock_test_and_set_16:
    return EmitBinaryAtomic(*this, llvm::AtomicRMWInst::Xchg, E);

  case Builtin::BI__sync_lock_release_1:
  case Builtin::BI__sync_lock_release_2:
  case Builtin::BI__sync_lock_release_4:
  case Builtin::BI__sync_lock_release_8:
  case Builtin::BI__sync_lock_release_16: {
    Value *Ptr = EmitScalarExpr(E->getArg(0));
    QualType ElTy = E->getArg(0)->getType()->getPointeeType();
    CharUnits StoreSize = getContext().getTypeSizeInChars(ElTy);
    llvm::Type *ITy = llvm::IntegerType::get(getLLVMContext(),
                                             StoreSize.getQuantity() * 8);
    Ptr = Builder.CreateBitCast(Ptr, ITy->getPointerTo());
    llvm::StoreInst *Store =
      Builder.CreateStore(llvm::Constant::getNullValue(ITy), Ptr);
    Store->setAlignment(StoreSize.getQuantity());
    Store->setAtomic(llvm::Release);
    return RValue::get(0);
  }

  case Builtin::BI__sync_synchronize: {
    // We assume this is supposed to correspond to a C++0x-style
    // sequentially-consistent fence (i.e. this is only usable for
    // synchonization, not device I/O or anything like that). This intrinsic
    // is really badly designed in the sense that in theory, there isn't
    // any way to safely use it... but in practice, it mostly works
    // to use it with non-atomic loads and stores to get acquire/release
    // semantics.
    Builder.CreateFence(llvm::SequentiallyConsistent);
    return RValue::get(0);
  }

  case Builtin::BI__c11_atomic_is_lock_free:
  case Builtin::BI__atomic_is_lock_free: {
    // Call "bool __atomic_is_lock_free(size_t size, void *ptr)". For the
    // __c11 builtin, ptr is 0 (indicating a properly-aligned object), since
    // _Atomic(T) is always properly-aligned.
    const char *LibCallName = "__atomic_is_lock_free";
    CallArgList Args;
    Args.add(RValue::get(EmitScalarExpr(E->getArg(0))),
             getContext().getSizeType());
    if (BuiltinID == Builtin::BI__atomic_is_lock_free)
      Args.add(RValue::get(EmitScalarExpr(E->getArg(1))),
               getContext().VoidPtrTy);
    else
      Args.add(RValue::get(llvm::Constant::getNullValue(VoidPtrTy)),
               getContext().VoidPtrTy);
    const CGFunctionInfo &FuncInfo =
        CGM.getTypes().arrangeFreeFunctionCall(E->getType(), Args,
                                               FunctionType::ExtInfo(),
                                               RequiredArgs::All);
    llvm::FunctionType *FTy = CGM.getTypes().GetFunctionType(FuncInfo);
    llvm::Constant *Func = CGM.CreateRuntimeFunction(FTy, LibCallName);
    return EmitCall(FuncInfo, Func, ReturnValueSlot(), Args);
  }

  case Builtin::BI__atomic_test_and_set: {
    // Look at the argument type to determine whether this is a volatile
    // operation. The parameter type is always volatile.
    QualType PtrTy = E->getArg(0)->IgnoreImpCasts()->getType();
    bool Volatile =
        PtrTy->castAs<PointerType>()->getPointeeType().isVolatileQualified();

    Value *Ptr = EmitScalarExpr(E->getArg(0));
    unsigned AddrSpace = Ptr->getType()->getPointerAddressSpace();
    Ptr = Builder.CreateBitCast(Ptr, Int8Ty->getPointerTo(AddrSpace));
    Value *NewVal = Builder.getInt8(1);
    Value *Order = EmitScalarExpr(E->getArg(1));
    if (isa<llvm::ConstantInt>(Order)) {
      int ord = cast<llvm::ConstantInt>(Order)->getZExtValue();
      AtomicRMWInst *Result = 0;
      switch (ord) {
      case 0:  // memory_order_relaxed
      default: // invalid order
        Result = Builder.CreateAtomicRMW(llvm::AtomicRMWInst::Xchg,
                                         Ptr, NewVal,
                                         llvm::Monotonic);
        break;
      case 1:  // memory_order_consume
      case 2:  // memory_order_acquire
        Result = Builder.CreateAtomicRMW(llvm::AtomicRMWInst::Xchg,
                                         Ptr, NewVal,
                                         llvm::Acquire);
        break;
      case 3:  // memory_order_release
        Result = Builder.CreateAtomicRMW(llvm::AtomicRMWInst::Xchg,
                                         Ptr, NewVal,
                                         llvm::Release);
        break;
      case 4:  // memory_order_acq_rel
        Result = Builder.CreateAtomicRMW(llvm::AtomicRMWInst::Xchg,
                                         Ptr, NewVal,
                                         llvm::AcquireRelease);
        break;
      case 5:  // memory_order_seq_cst
        Result = Builder.CreateAtomicRMW(llvm::AtomicRMWInst::Xchg,
                                         Ptr, NewVal,
                                         llvm::SequentiallyConsistent);
        break;
      }
      Result->setVolatile(Volatile);
      return RValue::get(Builder.CreateIsNotNull(Result, "tobool"));
    }

    llvm::BasicBlock *ContBB = createBasicBlock("atomic.continue", CurFn);

    llvm::BasicBlock *BBs[5] = {
      createBasicBlock("monotonic", CurFn),
      createBasicBlock("acquire", CurFn),
      createBasicBlock("release", CurFn),
      createBasicBlock("acqrel", CurFn),
      createBasicBlock("seqcst", CurFn)
    };
    llvm::AtomicOrdering Orders[5] = {
      llvm::Monotonic, llvm::Acquire, llvm::Release,
      llvm::AcquireRelease, llvm::SequentiallyConsistent
    };

    Order = Builder.CreateIntCast(Order, Builder.getInt32Ty(), false);
    llvm::SwitchInst *SI = Builder.CreateSwitch(Order, BBs[0]);

    Builder.SetInsertPoint(ContBB);
    PHINode *Result = Builder.CreatePHI(Int8Ty, 5, "was_set");

    for (unsigned i = 0; i < 5; ++i) {
      Builder.SetInsertPoint(BBs[i]);
      AtomicRMWInst *RMW = Builder.CreateAtomicRMW(llvm::AtomicRMWInst::Xchg,
                                                   Ptr, NewVal, Orders[i]);
      RMW->setVolatile(Volatile);
      Result->addIncoming(RMW, BBs[i]);
      Builder.CreateBr(ContBB);
    }

    SI->addCase(Builder.getInt32(0), BBs[0]);
    SI->addCase(Builder.getInt32(1), BBs[1]);
    SI->addCase(Builder.getInt32(2), BBs[1]);
    SI->addCase(Builder.getInt32(3), BBs[2]);
    SI->addCase(Builder.getInt32(4), BBs[3]);
    SI->addCase(Builder.getInt32(5), BBs[4]);

    Builder.SetInsertPoint(ContBB);
    return RValue::get(Builder.CreateIsNotNull(Result, "tobool"));
  }

  case Builtin::BI__atomic_clear: {
    QualType PtrTy = E->getArg(0)->IgnoreImpCasts()->getType();
    bool Volatile =
        PtrTy->castAs<PointerType>()->getPointeeType().isVolatileQualified();

    Value *Ptr = EmitScalarExpr(E->getArg(0));
    unsigned AddrSpace = Ptr->getType()->getPointerAddressSpace();
    Ptr = Builder.CreateBitCast(Ptr, Int8Ty->getPointerTo(AddrSpace));
    Value *NewVal = Builder.getInt8(0);
    Value *Order = EmitScalarExpr(E->getArg(1));
    if (isa<llvm::ConstantInt>(Order)) {
      int ord = cast<llvm::ConstantInt>(Order)->getZExtValue();
      StoreInst *Store = Builder.CreateStore(NewVal, Ptr, Volatile);
      Store->setAlignment(1);
      switch (ord) {
      case 0:  // memory_order_relaxed
      default: // invalid order
        Store->setOrdering(llvm::Monotonic);
        break;
      case 3:  // memory_order_release
        Store->setOrdering(llvm::Release);
        break;
      case 5:  // memory_order_seq_cst
        Store->setOrdering(llvm::SequentiallyConsistent);
        break;
      }
      return RValue::get(0);
    }

    llvm::BasicBlock *ContBB = createBasicBlock("atomic.continue", CurFn);

    llvm::BasicBlock *BBs[3] = {
      createBasicBlock("monotonic", CurFn),
      createBasicBlock("release", CurFn),
      createBasicBlock("seqcst", CurFn)
    };
    llvm::AtomicOrdering Orders[3] = {
      llvm::Monotonic, llvm::Release, llvm::SequentiallyConsistent
    };

    Order = Builder.CreateIntCast(Order, Builder.getInt32Ty(), false);
    llvm::SwitchInst *SI = Builder.CreateSwitch(Order, BBs[0]);

    for (unsigned i = 0; i < 3; ++i) {
      Builder.SetInsertPoint(BBs[i]);
      StoreInst *Store = Builder.CreateStore(NewVal, Ptr, Volatile);
      Store->setAlignment(1);
      Store->setOrdering(Orders[i]);
      Builder.CreateBr(ContBB);
    }

    SI->addCase(Builder.getInt32(0), BBs[0]);
    SI->addCase(Builder.getInt32(3), BBs[1]);
    SI->addCase(Builder.getInt32(5), BBs[2]);

    Builder.SetInsertPoint(ContBB);
    return RValue::get(0);
  }

  case Builtin::BI__atomic_thread_fence:
  case Builtin::BI__atomic_signal_fence:
  case Builtin::BI__c11_atomic_thread_fence:
  case Builtin::BI__c11_atomic_signal_fence: {
    llvm::SynchronizationScope Scope;
    if (BuiltinID == Builtin::BI__atomic_signal_fence ||
        BuiltinID == Builtin::BI__c11_atomic_signal_fence)
      Scope = llvm::SingleThread;
    else
      Scope = llvm::CrossThread;
    Value *Order = EmitScalarExpr(E->getArg(0));
    if (isa<llvm::ConstantInt>(Order)) {
      int ord = cast<llvm::ConstantInt>(Order)->getZExtValue();
      switch (ord) {
      case 0:  // memory_order_relaxed
      default: // invalid order
        break;
      case 1:  // memory_order_consume
      case 2:  // memory_order_acquire
        Builder.CreateFence(llvm::Acquire, Scope);
        break;
      case 3:  // memory_order_release
        Builder.CreateFence(llvm::Release, Scope);
        break;
      case 4:  // memory_order_acq_rel
        Builder.CreateFence(llvm::AcquireRelease, Scope);
        break;
      case 5:  // memory_order_seq_cst
        Builder.CreateFence(llvm::SequentiallyConsistent, Scope);
        break;
      }
      return RValue::get(0);
    }

    llvm::BasicBlock *AcquireBB, *ReleaseBB, *AcqRelBB, *SeqCstBB;
    AcquireBB = createBasicBlock("acquire", CurFn);
    ReleaseBB = createBasicBlock("release", CurFn);
    AcqRelBB = createBasicBlock("acqrel", CurFn);
    SeqCstBB = createBasicBlock("seqcst", CurFn);
    llvm::BasicBlock *ContBB = createBasicBlock("atomic.continue", CurFn);

    Order = Builder.CreateIntCast(Order, Builder.getInt32Ty(), false);
    llvm::SwitchInst *SI = Builder.CreateSwitch(Order, ContBB);

    Builder.SetInsertPoint(AcquireBB);
    Builder.CreateFence(llvm::Acquire, Scope);
    Builder.CreateBr(ContBB);
    SI->addCase(Builder.getInt32(1), AcquireBB);
    SI->addCase(Builder.getInt32(2), AcquireBB);

    Builder.SetInsertPoint(ReleaseBB);
    Builder.CreateFence(llvm::Release, Scope);
    Builder.CreateBr(ContBB);
    SI->addCase(Builder.getInt32(3), ReleaseBB);

    Builder.SetInsertPoint(AcqRelBB);
    Builder.CreateFence(llvm::AcquireRelease, Scope);
    Builder.CreateBr(ContBB);
    SI->addCase(Builder.getInt32(4), AcqRelBB);

    Builder.SetInsertPoint(SeqCstBB);
    Builder.CreateFence(llvm::SequentiallyConsistent, Scope);
    Builder.CreateBr(ContBB);
    SI->addCase(Builder.getInt32(5), SeqCstBB);

    Builder.SetInsertPoint(ContBB);
    return RValue::get(0);
  }

    // Library functions with special handling.
  case Builtin::BIsqrt:
  case Builtin::BIsqrtf:
  case Builtin::BIsqrtl: {
    // Transform a call to sqrt* into a @llvm.sqrt.* intrinsic call, but only
    // in finite- or unsafe-math mode (the intrinsic has different semantics
    // for handling negative numbers compared to the library function, so
    // -fmath-errno=0 is not enough).
    if (!FD->hasAttr<ConstAttr>())
      break;
    if (!(CGM.getCodeGenOpts().UnsafeFPMath ||
          CGM.getCodeGenOpts().NoNaNsFPMath))
      break;
    Value *Arg0 = EmitScalarExpr(E->getArg(0));
    llvm::Type *ArgType = Arg0->getType();
    Value *F = CGM.getIntrinsic(Intrinsic::sqrt, ArgType);
    return RValue::get(Builder.CreateCall(F, Arg0));
  }

  case Builtin::BIpow:
  case Builtin::BIpowf:
  case Builtin::BIpowl: {
    // Transform a call to pow* into a @llvm.pow.* intrinsic call.
    if (!FD->hasAttr<ConstAttr>())
      break;
    Value *Base = EmitScalarExpr(E->getArg(0));
    Value *Exponent = EmitScalarExpr(E->getArg(1));
    llvm::Type *ArgType = Base->getType();
    Value *F = CGM.getIntrinsic(Intrinsic::pow, ArgType);
    return RValue::get(Builder.CreateCall2(F, Base, Exponent));
    break;
  }

  case Builtin::BIfma:
  case Builtin::BIfmaf:
  case Builtin::BIfmal:
  case Builtin::BI__builtin_fma:
  case Builtin::BI__builtin_fmaf:
  case Builtin::BI__builtin_fmal: {
    // Rewrite fma to intrinsic.
    Value *FirstArg = EmitScalarExpr(E->getArg(0));
    llvm::Type *ArgType = FirstArg->getType();
    Value *F = CGM.getIntrinsic(Intrinsic::fma, ArgType);
    return RValue::get(Builder.CreateCall3(F, FirstArg,
                                              EmitScalarExpr(E->getArg(1)),
                                              EmitScalarExpr(E->getArg(2))));
  }

  case Builtin::BI__builtin_signbit:
  case Builtin::BI__builtin_signbitf:
  case Builtin::BI__builtin_signbitl: {
    LLVMContext &C = CGM.getLLVMContext();

    Value *Arg = EmitScalarExpr(E->getArg(0));
    llvm::Type *ArgTy = Arg->getType();
    if (ArgTy->isPPC_FP128Ty())
      break; // FIXME: I'm not sure what the right implementation is here.
    int ArgWidth = ArgTy->getPrimitiveSizeInBits();
    llvm::Type *ArgIntTy = llvm::IntegerType::get(C, ArgWidth);
    Value *BCArg = Builder.CreateBitCast(Arg, ArgIntTy);
    Value *ZeroCmp = llvm::Constant::getNullValue(ArgIntTy);
    Value *Result = Builder.CreateICmpSLT(BCArg, ZeroCmp);
    return RValue::get(Builder.CreateZExt(Result, ConvertType(E->getType())));
  }
  case Builtin::BI__builtin_annotation: {
    llvm::Value *AnnVal = EmitScalarExpr(E->getArg(0));
    llvm::Value *F = CGM.getIntrinsic(llvm::Intrinsic::annotation,
                                      AnnVal->getType());

    // Get the annotation string, go through casts. Sema requires this to be a
    // non-wide string literal, potentially casted, so the cast<> is safe.
    const Expr *AnnotationStrExpr = E->getArg(1)->IgnoreParenCasts();
    StringRef Str = cast<StringLiteral>(AnnotationStrExpr)->getString();
    return RValue::get(EmitAnnotationCall(F, AnnVal, Str, E->getExprLoc()));
  }
  case Builtin::BI__builtin_addcb:
  case Builtin::BI__builtin_addcs:
  case Builtin::BI__builtin_addc:
  case Builtin::BI__builtin_addcl:
  case Builtin::BI__builtin_addcll:
  case Builtin::BI__builtin_subcb:
  case Builtin::BI__builtin_subcs:
  case Builtin::BI__builtin_subc:
  case Builtin::BI__builtin_subcl:
  case Builtin::BI__builtin_subcll: {

    // We translate all of these builtins from expressions of the form:
    //   int x = ..., y = ..., carryin = ..., carryout, result;
    //   result = __builtin_addc(x, y, carryin, &carryout);
    //
    // to LLVM IR of the form:
    //
    //   %tmp1 = call {i32, i1} @llvm.uadd.with.overflow.i32(i32 %x, i32 %y)
    //   %tmpsum1 = extractvalue {i32, i1} %tmp1, 0
    //   %carry1 = extractvalue {i32, i1} %tmp1, 1
    //   %tmp2 = call {i32, i1} @llvm.uadd.with.overflow.i32(i32 %tmpsum1,
    //                                                       i32 %carryin)
    //   %result = extractvalue {i32, i1} %tmp2, 0
    //   %carry2 = extractvalue {i32, i1} %tmp2, 1
    //   %tmp3 = or i1 %carry1, %carry2
    //   %tmp4 = zext i1 %tmp3 to i32
    //   store i32 %tmp4, i32* %carryout

    // Scalarize our inputs.
    llvm::Value *X = EmitScalarExpr(E->getArg(0));
    llvm::Value *Y = EmitScalarExpr(E->getArg(1));
    llvm::Value *Carryin = EmitScalarExpr(E->getArg(2));
    std::pair<llvm::Value*, unsigned> CarryOutPtr =
      EmitPointerWithAlignment(E->getArg(3));

    // Decide if we are lowering to a uadd.with.overflow or usub.with.overflow.
    llvm::Intrinsic::ID IntrinsicId;
    switch (BuiltinID) {
    default: llvm_unreachable("Unknown multiprecision builtin id.");
    case Builtin::BI__builtin_addcb:
    case Builtin::BI__builtin_addcs:
    case Builtin::BI__builtin_addc:
    case Builtin::BI__builtin_addcl:
    case Builtin::BI__builtin_addcll:
      IntrinsicId = llvm::Intrinsic::uadd_with_overflow;
      break;
    case Builtin::BI__builtin_subcb:
    case Builtin::BI__builtin_subcs:
    case Builtin::BI__builtin_subc:
    case Builtin::BI__builtin_subcl:
    case Builtin::BI__builtin_subcll:
      IntrinsicId = llvm::Intrinsic::usub_with_overflow;
      break;
    }

    // Construct our resulting LLVM IR expression.
    llvm::Value *Carry1;
    llvm::Value *Sum1 = EmitOverflowIntrinsic(*this, IntrinsicId,
                                              X, Y, Carry1);
    llvm::Value *Carry2;
    llvm::Value *Sum2 = EmitOverflowIntrinsic(*this, IntrinsicId,
                                              Sum1, Carryin, Carry2);
    llvm::Value *CarryOut = Builder.CreateZExt(Builder.CreateOr(Carry1, Carry2),
                                               X->getType());
    llvm::StoreInst *CarryOutStore = Builder.CreateStore(CarryOut,
                                                         CarryOutPtr.first);
    CarryOutStore->setAlignment(CarryOutPtr.second);
    return RValue::get(Sum2);
  }
  case Builtin::BI__builtin_uadd_overflow:
  case Builtin::BI__builtin_uaddl_overflow:
  case Builtin::BI__builtin_uaddll_overflow:
  case Builtin::BI__builtin_usub_overflow:
  case Builtin::BI__builtin_usubl_overflow:
  case Builtin::BI__builtin_usubll_overflow:
  case Builtin::BI__builtin_umul_overflow:
  case Builtin::BI__builtin_umull_overflow:
  case Builtin::BI__builtin_umulll_overflow:
  case Builtin::BI__builtin_sadd_overflow:
  case Builtin::BI__builtin_saddl_overflow:
  case Builtin::BI__builtin_saddll_overflow:
  case Builtin::BI__builtin_ssub_overflow:
  case Builtin::BI__builtin_ssubl_overflow:
  case Builtin::BI__builtin_ssubll_overflow:
  case Builtin::BI__builtin_smul_overflow:
  case Builtin::BI__builtin_smull_overflow:
  case Builtin::BI__builtin_smulll_overflow: {

    // We translate all of these builtins directly to the relevant llvm IR node.

    // Scalarize our inputs.
    llvm::Value *X = EmitScalarExpr(E->getArg(0));
    llvm::Value *Y = EmitScalarExpr(E->getArg(1));
    std::pair<llvm::Value *, unsigned> SumOutPtr =
      EmitPointerWithAlignment(E->getArg(2));

    // Decide which of the overflow intrinsics we are lowering to:
    llvm::Intrinsic::ID IntrinsicId;
    switch (BuiltinID) {
    default: llvm_unreachable("Unknown security overflow builtin id.");
    case Builtin::BI__builtin_uadd_overflow:
    case Builtin::BI__builtin_uaddl_overflow:
    case Builtin::BI__builtin_uaddll_overflow:
      IntrinsicId = llvm::Intrinsic::uadd_with_overflow;
      break;
    case Builtin::BI__builtin_usub_overflow:
    case Builtin::BI__builtin_usubl_overflow:
    case Builtin::BI__builtin_usubll_overflow:
      IntrinsicId = llvm::Intrinsic::usub_with_overflow;
      break;
    case Builtin::BI__builtin_umul_overflow:
    case Builtin::BI__builtin_umull_overflow:
    case Builtin::BI__builtin_umulll_overflow:
      IntrinsicId = llvm::Intrinsic::umul_with_overflow;
      break;
    case Builtin::BI__builtin_sadd_overflow:
    case Builtin::BI__builtin_saddl_overflow:
    case Builtin::BI__builtin_saddll_overflow:
      IntrinsicId = llvm::Intrinsic::sadd_with_overflow;
      break;
    case Builtin::BI__builtin_ssub_overflow:
    case Builtin::BI__builtin_ssubl_overflow:
    case Builtin::BI__builtin_ssubll_overflow:
      IntrinsicId = llvm::Intrinsic::ssub_with_overflow;
      break;
    case Builtin::BI__builtin_smul_overflow:
    case Builtin::BI__builtin_smull_overflow:
    case Builtin::BI__builtin_smulll_overflow:
      IntrinsicId = llvm::Intrinsic::smul_with_overflow;
      break;
    }

    
    llvm::Value *Carry;
    llvm::Value *Sum = EmitOverflowIntrinsic(*this, IntrinsicId, X, Y, Carry);
    llvm::StoreInst *SumOutStore = Builder.CreateStore(Sum, SumOutPtr.first);
    SumOutStore->setAlignment(SumOutPtr.second);

    return RValue::get(Carry);
  }
  case Builtin::BI__builtin_addressof:
    return RValue::get(EmitLValue(E->getArg(0)).getAddress());
  case Builtin::BI__noop:
    return RValue::get(0);
  case Builtin::BI_InterlockedCompareExchange: {
    AtomicCmpXchgInst *CXI = Builder.CreateAtomicCmpXchg(
        EmitScalarExpr(E->getArg(0)),
        EmitScalarExpr(E->getArg(2)),
        EmitScalarExpr(E->getArg(1)),
        SequentiallyConsistent);
      CXI->setVolatile(true);
      return RValue::get(CXI);
  }
  case Builtin::BI_InterlockedIncrement: {
    AtomicRMWInst *RMWI = Builder.CreateAtomicRMW(
      AtomicRMWInst::Add,
      EmitScalarExpr(E->getArg(0)),
      ConstantInt::get(Int32Ty, 1),
      llvm::SequentiallyConsistent);
    RMWI->setVolatile(true);
    return RValue::get(Builder.CreateAdd(RMWI, ConstantInt::get(Int32Ty, 1)));
  }
  case Builtin::BI_InterlockedDecrement: {
    AtomicRMWInst *RMWI = Builder.CreateAtomicRMW(
      AtomicRMWInst::Sub,
      EmitScalarExpr(E->getArg(0)),
      ConstantInt::get(Int32Ty, 1),
      llvm::SequentiallyConsistent);
    RMWI->setVolatile(true);
    return RValue::get(Builder.CreateSub(RMWI, ConstantInt::get(Int32Ty, 1)));
  }
  case Builtin::BI_InterlockedExchangeAdd: {
    AtomicRMWInst *RMWI = Builder.CreateAtomicRMW(
      AtomicRMWInst::Add,
      EmitScalarExpr(E->getArg(0)),
      EmitScalarExpr(E->getArg(1)),
      llvm::SequentiallyConsistent);
    RMWI->setVolatile(true);
    return RValue::get(RMWI);
  }
  }

  // If this is an alias for a lib function (e.g. __builtin_sin), emit
  // the call using the normal call path, but using the unmangled
  // version of the function name.
  if (getContext().BuiltinInfo.isLibFunction(BuiltinID))
    return emitLibraryCall(*this, FD, E,
                           CGM.getBuiltinLibFunction(FD, BuiltinID));

  // If this is a predefined lib function (e.g. malloc), emit the call
  // using exactly the normal call path.
  if (getContext().BuiltinInfo.isPredefinedLibFunction(BuiltinID))
    return emitLibraryCall(*this, FD, E, EmitScalarExpr(E->getCallee()));

  // See if we have a target specific intrinsic.
  const char *Name = getContext().BuiltinInfo.GetName(BuiltinID);
  Intrinsic::ID IntrinsicID = Intrinsic::not_intrinsic;
  if (const char *Prefix =
      llvm::Triple::getArchTypePrefix(getTarget().getTriple().getArch()))
    IntrinsicID = Intrinsic::getIntrinsicForGCCBuiltin(Prefix, Name);

  if (IntrinsicID != Intrinsic::not_intrinsic) {
    SmallVector<Value*, 16> Args;

    // Find out if any arguments are required to be integer constant
    // expressions.
    unsigned ICEArguments = 0;
    ASTContext::GetBuiltinTypeError Error;
    getContext().GetBuiltinType(BuiltinID, Error, &ICEArguments);
    assert(Error == ASTContext::GE_None && "Should not codegen an error");

    Function *F = CGM.getIntrinsic(IntrinsicID);
    llvm::FunctionType *FTy = F->getFunctionType();

    for (unsigned i = 0, e = E->getNumArgs(); i != e; ++i) {
      Value *ArgValue;
      // If this is a normal argument, just emit it as a scalar.
      if ((ICEArguments & (1 << i)) == 0) {
        ArgValue = EmitScalarExpr(E->getArg(i));
      } else {
        // If this is required to be a constant, constant fold it so that we
        // know that the generated intrinsic gets a ConstantInt.
        llvm::APSInt Result;
        bool IsConst = E->getArg(i)->isIntegerConstantExpr(Result,getContext());
        assert(IsConst && "Constant arg isn't actually constant?");
        (void)IsConst;
        ArgValue = llvm::ConstantInt::get(getLLVMContext(), Result);
      }

      // If the intrinsic arg type is different from the builtin arg type
      // we need to do a bit cast.
      llvm::Type *PTy = FTy->getParamType(i);
      if (PTy != ArgValue->getType()) {
        assert(PTy->canLosslesslyBitCastTo(FTy->getParamType(i)) &&
               "Must be able to losslessly bit cast to param");
        ArgValue = Builder.CreateBitCast(ArgValue, PTy);
      }

      Args.push_back(ArgValue);
    }

    Value *V = Builder.CreateCall(F, Args);
    QualType BuiltinRetType = E->getType();

    llvm::Type *RetTy = VoidTy;
    if (!BuiltinRetType->isVoidType())
      RetTy = ConvertType(BuiltinRetType);

    if (RetTy != V->getType()) {
      assert(V->getType()->canLosslesslyBitCastTo(RetTy) &&
             "Must be able to losslessly bit cast result type");
      V = Builder.CreateBitCast(V, RetTy);
    }

    return RValue::get(V);
  }

  // See if we have a target specific builtin that needs to be lowered.
  if (Value *V = EmitTargetBuiltinExpr(BuiltinID, E))
    return RValue::get(V);

  ErrorUnsupported(E, "builtin function");

  // Unknown builtin, for now just dump it out and return undef.
  return GetUndefRValue(E->getType());
}

Value *CodeGenFunction::EmitTargetBuiltinExpr(unsigned BuiltinID,
                                              const CallExpr *E) {
  switch (getTarget().getTriple().getArch()) {
  case llvm::Triple::aarch64:
  case llvm::Triple::aarch64_be:
    return EmitAArch64BuiltinExpr(BuiltinID, E);
  case llvm::Triple::arm:
  case llvm::Triple::thumb:
    return EmitARMBuiltinExpr(BuiltinID, E);
  case llvm::Triple::x86:
  case llvm::Triple::x86_64:
    return EmitX86BuiltinExpr(BuiltinID, E);
  case llvm::Triple::ppc:
  case llvm::Triple::ppc64:
  case llvm::Triple::ppc64le:
    return EmitPPCBuiltinExpr(BuiltinID, E);
  default:
    return 0;
  }
}

static llvm::VectorType *GetNeonType(CodeGenFunction *CGF,
                                     NeonTypeFlags TypeFlags,
                                     bool V1Ty=false) {
  int IsQuad = TypeFlags.isQuad();
  switch (TypeFlags.getEltType()) {
  case NeonTypeFlags::Int8:
  case NeonTypeFlags::Poly8:
    return llvm::VectorType::get(CGF->Int8Ty, V1Ty ? 1 : (8 << IsQuad));
  case NeonTypeFlags::Int16:
  case NeonTypeFlags::Poly16:
  case NeonTypeFlags::Float16:
    return llvm::VectorType::get(CGF->Int16Ty, V1Ty ? 1 : (4 << IsQuad));
  case NeonTypeFlags::Int32:
    return llvm::VectorType::get(CGF->Int32Ty, V1Ty ? 1 : (2 << IsQuad));
  case NeonTypeFlags::Int64:
  case NeonTypeFlags::Poly64:
    return llvm::VectorType::get(CGF->Int64Ty, V1Ty ? 1 : (1 << IsQuad));
  case NeonTypeFlags::Poly128:
    // FIXME: i128 and f128 doesn't get fully support in Clang and llvm.
    // There is a lot of i128 and f128 API missing.
    // so we use v16i8 to represent poly128 and get pattern matched.
    return llvm::VectorType::get(CGF->Int8Ty, 16);
  case NeonTypeFlags::Float32:
    return llvm::VectorType::get(CGF->FloatTy, V1Ty ? 1 : (2 << IsQuad));
  case NeonTypeFlags::Float64:
    return llvm::VectorType::get(CGF->DoubleTy, V1Ty ? 1 : (1 << IsQuad));
  }
  llvm_unreachable("Unknown vector element type!");
}

Value *CodeGenFunction::EmitNeonSplat(Value *V, Constant *C) {
  unsigned nElts = cast<llvm::VectorType>(V->getType())->getNumElements();
  Value* SV = llvm::ConstantVector::getSplat(nElts, C);
  return Builder.CreateShuffleVector(V, V, SV, "lane");
}

Value *CodeGenFunction::EmitNeonCall(Function *F, SmallVectorImpl<Value*> &Ops,
                                     const char *name,
                                     unsigned shift, bool rightshift) {
  unsigned j = 0;
  for (Function::const_arg_iterator ai = F->arg_begin(), ae = F->arg_end();
       ai != ae; ++ai, ++j)
    if (shift > 0 && shift == j)
      Ops[j] = EmitNeonShiftVector(Ops[j], ai->getType(), rightshift);
    else
      Ops[j] = Builder.CreateBitCast(Ops[j], ai->getType(), name);

  return Builder.CreateCall(F, Ops, name);
}

Value *CodeGenFunction::EmitNeonShiftVector(Value *V, llvm::Type *Ty,
                                            bool neg) {
  int SV = cast<ConstantInt>(V)->getSExtValue();

  llvm::VectorType *VTy = cast<llvm::VectorType>(Ty);
  llvm::Constant *C = ConstantInt::get(VTy->getElementType(), neg ? -SV : SV);
  return llvm::ConstantVector::getSplat(VTy->getNumElements(), C);
}

// \brief Right-shift a vector by a constant.
Value *CodeGenFunction::EmitNeonRShiftImm(Value *Vec, Value *Shift,
                                          llvm::Type *Ty, bool usgn,
                                          const char *name) {
  llvm::VectorType *VTy = cast<llvm::VectorType>(Ty);

  int ShiftAmt = cast<ConstantInt>(Shift)->getSExtValue();
  int EltSize = VTy->getScalarSizeInBits();

  Vec = Builder.CreateBitCast(Vec, Ty);

  // lshr/ashr are undefined when the shift amount is equal to the vector
  // element size.
  if (ShiftAmt == EltSize) {
    if (usgn) {
      // Right-shifting an unsigned value by its size yields 0.
      llvm::Constant *Zero = ConstantInt::get(VTy->getElementType(), 0);
      return llvm::ConstantVector::getSplat(VTy->getNumElements(), Zero);
    } else {
      // Right-shifting a signed value by its size is equivalent
      // to a shift of size-1.
      --ShiftAmt;
      Shift = ConstantInt::get(VTy->getElementType(), ShiftAmt);
    }
  }

  Shift = EmitNeonShiftVector(Shift, Ty, false);
  if (usgn)
    return Builder.CreateLShr(Vec, Shift, name);
  else
    return Builder.CreateAShr(Vec, Shift, name);
}

/// GetPointeeAlignment - Given an expression with a pointer type, find the
/// alignment of the type referenced by the pointer.  Skip over implicit
/// casts.
std::pair<llvm::Value*, unsigned>
CodeGenFunction::EmitPointerWithAlignment(const Expr *Addr) {
  assert(Addr->getType()->isPointerType());
  Addr = Addr->IgnoreParens();
  if (const ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(Addr)) {
    if ((ICE->getCastKind() == CK_BitCast || ICE->getCastKind() == CK_NoOp) &&
        ICE->getSubExpr()->getType()->isPointerType()) {
      std::pair<llvm::Value*, unsigned> Ptr =
          EmitPointerWithAlignment(ICE->getSubExpr());
      Ptr.first = Builder.CreateBitCast(Ptr.first,
                                        ConvertType(Addr->getType()));
      return Ptr;
    } else if (ICE->getCastKind() == CK_ArrayToPointerDecay) {
      LValue LV = EmitLValue(ICE->getSubExpr());
      unsigned Align = LV.getAlignment().getQuantity();
      if (!Align) {
        // FIXME: Once LValues are fixed to always set alignment,
        // zap this code.
        QualType PtTy = ICE->getSubExpr()->getType();
        if (!PtTy->isIncompleteType())
          Align = getContext().getTypeAlignInChars(PtTy).getQuantity();
        else
          Align = 1;
      }
      return std::make_pair(LV.getAddress(), Align);
    }
  }
  if (const UnaryOperator *UO = dyn_cast<UnaryOperator>(Addr)) {
    if (UO->getOpcode() == UO_AddrOf) {
      LValue LV = EmitLValue(UO->getSubExpr());
      unsigned Align = LV.getAlignment().getQuantity();
      if (!Align) {
        // FIXME: Once LValues are fixed to always set alignment,
        // zap this code.
        QualType PtTy = UO->getSubExpr()->getType();
        if (!PtTy->isIncompleteType())
          Align = getContext().getTypeAlignInChars(PtTy).getQuantity();
        else
          Align = 1;
      }
      return std::make_pair(LV.getAddress(), Align);
    }
  }

  unsigned Align = 1;
  QualType PtTy = Addr->getType()->getPointeeType();
  if (!PtTy->isIncompleteType())
    Align = getContext().getTypeAlignInChars(PtTy).getQuantity();

  return std::make_pair(EmitScalarExpr(Addr), Align);
}

enum {
  AddRetType = (1 << 0),
  Add1ArgType = (1 << 1),
  Add2ArgTypes = (1 << 2),

  VectorizeRetType = (1 << 3),
  VectorizeArgTypes = (1 << 4),

  InventFloatType = (1 << 5),
  UnsignedAlts = (1 << 6),

  Vectorize1ArgType = Add1ArgType | VectorizeArgTypes,
  VectorRet = AddRetType | VectorizeRetType,
  VectorRetGetArgs01 =
      AddRetType | Add2ArgTypes | VectorizeRetType | VectorizeArgTypes,
  FpCmpzModifiers =
      AddRetType | VectorizeRetType | Add1ArgType | InventFloatType
};

 struct NeonIntrinsicInfo {
  unsigned BuiltinID;
  unsigned LLVMIntrinsic;
  unsigned AltLLVMIntrinsic;
  const char *NameHint;
  unsigned TypeModifier;

  bool operator<(unsigned RHSBuiltinID) const {
    return BuiltinID < RHSBuiltinID;
  }
};

#define NEONMAP0(NameBase) \
  { NEON::BI__builtin_neon_ ## NameBase, 0, 0, #NameBase, 0 }

#define NEONMAP1(NameBase, LLVMIntrinsic, TypeModifier) \
  { NEON:: BI__builtin_neon_ ## NameBase, \
      Intrinsic::LLVMIntrinsic, 0, #NameBase, TypeModifier }

#define NEONMAP2(NameBase, LLVMIntrinsic, AltLLVMIntrinsic, TypeModifier) \
  { NEON:: BI__builtin_neon_ ## NameBase, \
      Intrinsic::LLVMIntrinsic, Intrinsic::AltLLVMIntrinsic, \
      #NameBase, TypeModifier }

static const NeonIntrinsicInfo AArch64SISDIntrinsicInfo[] = {
  NEONMAP1(vabdd_f64, aarch64_neon_vabd, AddRetType),
  NEONMAP1(vabds_f32, aarch64_neon_vabd, AddRetType),
  NEONMAP1(vabsd_s64, aarch64_neon_vabs, 0),
  NEONMAP1(vaddd_s64, aarch64_neon_vaddds, 0),
  NEONMAP1(vaddd_u64, aarch64_neon_vadddu, 0),
  NEONMAP1(vaddlv_s16, aarch64_neon_saddlv, VectorRet | Add1ArgType),
  NEONMAP1(vaddlv_s32, aarch64_neon_saddlv, VectorRet | Add1ArgType),
  NEONMAP1(vaddlv_s8, aarch64_neon_saddlv, VectorRet | Add1ArgType),
  NEONMAP1(vaddlv_u16, aarch64_neon_uaddlv, VectorRet | Add1ArgType),
  NEONMAP1(vaddlv_u32, aarch64_neon_uaddlv, VectorRet | Add1ArgType),
  NEONMAP1(vaddlv_u8, aarch64_neon_uaddlv, VectorRet | Add1ArgType),
  NEONMAP1(vaddlvq_s16, aarch64_neon_saddlv, VectorRet | Add1ArgType),
  NEONMAP1(vaddlvq_s32, aarch64_neon_saddlv, VectorRet | Add1ArgType),
  NEONMAP1(vaddlvq_s8, aarch64_neon_saddlv, VectorRet | Add1ArgType),
  NEONMAP1(vaddlvq_u16, aarch64_neon_uaddlv, VectorRet | Add1ArgType),
  NEONMAP1(vaddlvq_u32, aarch64_neon_uaddlv, VectorRet | Add1ArgType),
  NEONMAP1(vaddlvq_u8, aarch64_neon_uaddlv, VectorRet | Add1ArgType),
  NEONMAP1(vaddv_f32, aarch64_neon_vpfadd, AddRetType | Add1ArgType),
  NEONMAP1(vaddv_s16, aarch64_neon_vaddv, VectorRet | Add1ArgType),
  NEONMAP1(vaddv_s32, aarch64_neon_vaddv, VectorRet | Add1ArgType),
  NEONMAP1(vaddv_s8, aarch64_neon_vaddv, VectorRet | Add1ArgType),
  NEONMAP1(vaddv_u16, aarch64_neon_vaddv, VectorRet | Add1ArgType),
  NEONMAP1(vaddv_u32, aarch64_neon_vaddv, VectorRet | Add1ArgType),
  NEONMAP1(vaddv_u8, aarch64_neon_vaddv, VectorRet | Add1ArgType),
  NEONMAP1(vaddvq_f32, aarch64_neon_vpfadd, AddRetType | Add1ArgType),
  NEONMAP1(vaddvq_f64, aarch64_neon_vpfadd, AddRetType | Add1ArgType),
  NEONMAP1(vaddvq_s16, aarch64_neon_vaddv, VectorRet | Add1ArgType),
  NEONMAP1(vaddvq_s32, aarch64_neon_vaddv, VectorRet | Add1ArgType),
  NEONMAP1(vaddvq_s64, aarch64_neon_vaddv, VectorRet | Add1ArgType),
  NEONMAP1(vaddvq_s8, aarch64_neon_vaddv, VectorRet | Add1ArgType),
  NEONMAP1(vaddvq_u16, aarch64_neon_vaddv, VectorRet | Add1ArgType),
  NEONMAP1(vaddvq_u32, aarch64_neon_vaddv, VectorRet | Add1ArgType),
  NEONMAP1(vaddvq_u64, aarch64_neon_vaddv, VectorRet | Add1ArgType),
  NEONMAP1(vaddvq_u8, aarch64_neon_vaddv, VectorRet | Add1ArgType),
  NEONMAP1(vcaged_f64, aarch64_neon_fcage, VectorRet | Add2ArgTypes),
  NEONMAP1(vcages_f32, aarch64_neon_fcage, VectorRet | Add2ArgTypes),
  NEONMAP1(vcagtd_f64, aarch64_neon_fcagt, VectorRet | Add2ArgTypes),
  NEONMAP1(vcagts_f32, aarch64_neon_fcagt, VectorRet | Add2ArgTypes),
  NEONMAP1(vcaled_f64, aarch64_neon_fcage, VectorRet | Add2ArgTypes),
  NEONMAP1(vcales_f32, aarch64_neon_fcage, VectorRet | Add2ArgTypes),
  NEONMAP1(vcaltd_f64, aarch64_neon_fcagt, VectorRet | Add2ArgTypes),
  NEONMAP1(vcalts_f32, aarch64_neon_fcagt, VectorRet | Add2ArgTypes),
  NEONMAP1(vceqd_f64, aarch64_neon_fceq, VectorRet | Add2ArgTypes),
  NEONMAP1(vceqd_s64, aarch64_neon_vceq, VectorRetGetArgs01),
  NEONMAP1(vceqd_u64, aarch64_neon_vceq, VectorRetGetArgs01),
  NEONMAP1(vceqs_f32, aarch64_neon_fceq, VectorRet | Add2ArgTypes),
  NEONMAP1(vceqzd_f64, aarch64_neon_fceq, FpCmpzModifiers),
  NEONMAP1(vceqzd_s64, aarch64_neon_vceq, VectorRetGetArgs01),
  NEONMAP1(vceqzd_u64, aarch64_neon_vceq, VectorRetGetArgs01),
  NEONMAP1(vceqzs_f32, aarch64_neon_fceq, FpCmpzModifiers),
  NEONMAP1(vcged_f64, aarch64_neon_fcge, VectorRet | Add2ArgTypes),
  NEONMAP1(vcged_s64, aarch64_neon_vcge, VectorRetGetArgs01),
  NEONMAP1(vcged_u64, aarch64_neon_vchs, VectorRetGetArgs01),
  NEONMAP1(vcges_f32, aarch64_neon_fcge, VectorRet | Add2ArgTypes),
  NEONMAP1(vcgezd_f64, aarch64_neon_fcge, FpCmpzModifiers),
  NEONMAP1(vcgezd_s64, aarch64_neon_vcge, VectorRetGetArgs01),
  NEONMAP1(vcgezs_f32, aarch64_neon_fcge, FpCmpzModifiers),
  NEONMAP1(vcgtd_f64, aarch64_neon_fcgt, VectorRet | Add2ArgTypes),
  NEONMAP1(vcgtd_s64, aarch64_neon_vcgt, VectorRetGetArgs01),
  NEONMAP1(vcgtd_u64, aarch64_neon_vchi, VectorRetGetArgs01),
  NEONMAP1(vcgts_f32, aarch64_neon_fcgt, VectorRet | Add2ArgTypes),
  NEONMAP1(vcgtzd_f64, aarch64_neon_fcgt, FpCmpzModifiers),
  NEONMAP1(vcgtzd_s64, aarch64_neon_vcgt, VectorRetGetArgs01),
  NEONMAP1(vcgtzs_f32, aarch64_neon_fcgt, FpCmpzModifiers),
  NEONMAP1(vcled_f64, aarch64_neon_fcge, VectorRet | Add2ArgTypes),
  NEONMAP1(vcled_s64, aarch64_neon_vcge, VectorRetGetArgs01),
  NEONMAP1(vcled_u64, aarch64_neon_vchs, VectorRetGetArgs01),
  NEONMAP1(vcles_f32, aarch64_neon_fcge, VectorRet | Add2ArgTypes),
  NEONMAP1(vclezd_f64, aarch64_neon_fclez, FpCmpzModifiers),
  NEONMAP1(vclezd_s64, aarch64_neon_vclez, VectorRetGetArgs01),
  NEONMAP1(vclezs_f32, aarch64_neon_fclez, FpCmpzModifiers),
  NEONMAP1(vcltd_f64, aarch64_neon_fcgt, VectorRet | Add2ArgTypes),
  NEONMAP1(vcltd_s64, aarch64_neon_vcgt, VectorRetGetArgs01),
  NEONMAP1(vcltd_u64, aarch64_neon_vchi, VectorRetGetArgs01),
  NEONMAP1(vclts_f32, aarch64_neon_fcgt, VectorRet | Add2ArgTypes),
  NEONMAP1(vcltzd_f64, aarch64_neon_fcltz, FpCmpzModifiers),
  NEONMAP1(vcltzd_s64, aarch64_neon_vcltz, VectorRetGetArgs01),
  NEONMAP1(vcltzs_f32, aarch64_neon_fcltz, FpCmpzModifiers),
  NEONMAP1(vcvtad_s64_f64, aarch64_neon_fcvtas, VectorRet | Add1ArgType),
  NEONMAP1(vcvtad_u64_f64, aarch64_neon_fcvtau, VectorRet | Add1ArgType),
  NEONMAP1(vcvtas_s32_f32, aarch64_neon_fcvtas, VectorRet | Add1ArgType),
  NEONMAP1(vcvtas_u32_f32, aarch64_neon_fcvtau, VectorRet | Add1ArgType),
  NEONMAP1(vcvtd_f64_s64, aarch64_neon_vcvtint2fps, AddRetType | Vectorize1ArgType),
  NEONMAP1(vcvtd_f64_u64, aarch64_neon_vcvtint2fpu, AddRetType | Vectorize1ArgType),
  NEONMAP1(vcvtd_n_f64_s64, aarch64_neon_vcvtfxs2fp_n, AddRetType | Vectorize1ArgType),
  NEONMAP1(vcvtd_n_f64_u64, aarch64_neon_vcvtfxu2fp_n, AddRetType | Vectorize1ArgType),
  NEONMAP1(vcvtd_n_s64_f64, aarch64_neon_vcvtfp2fxs_n, VectorRet | Add1ArgType),
  NEONMAP1(vcvtd_n_u64_f64, aarch64_neon_vcvtfp2fxu_n, VectorRet | Add1ArgType),
  NEONMAP1(vcvtd_s64_f64, aarch64_neon_fcvtzs, VectorRet | Add1ArgType),
  NEONMAP1(vcvtd_u64_f64, aarch64_neon_fcvtzu, VectorRet | Add1ArgType),
  NEONMAP1(vcvtmd_s64_f64, aarch64_neon_fcvtms, VectorRet | Add1ArgType),
  NEONMAP1(vcvtmd_u64_f64, aarch64_neon_fcvtmu, VectorRet | Add1ArgType),
  NEONMAP1(vcvtms_s32_f32, aarch64_neon_fcvtms, VectorRet | Add1ArgType),
  NEONMAP1(vcvtms_u32_f32, aarch64_neon_fcvtmu, VectorRet | Add1ArgType),
  NEONMAP1(vcvtnd_s64_f64, aarch64_neon_fcvtns, VectorRet | Add1ArgType),
  NEONMAP1(vcvtnd_u64_f64, aarch64_neon_fcvtnu, VectorRet | Add1ArgType),
  NEONMAP1(vcvtns_s32_f32, aarch64_neon_fcvtns, VectorRet | Add1ArgType),
  NEONMAP1(vcvtns_u32_f32, aarch64_neon_fcvtnu, VectorRet | Add1ArgType),
  NEONMAP1(vcvtpd_s64_f64, aarch64_neon_fcvtps, VectorRet | Add1ArgType),
  NEONMAP1(vcvtpd_u64_f64, aarch64_neon_fcvtpu, VectorRet | Add1ArgType),
  NEONMAP1(vcvtps_s32_f32, aarch64_neon_fcvtps, VectorRet | Add1ArgType),
  NEONMAP1(vcvtps_u32_f32, aarch64_neon_fcvtpu, VectorRet | Add1ArgType),
  NEONMAP1(vcvts_f32_s32, aarch64_neon_vcvtint2fps, AddRetType | Vectorize1ArgType),
  NEONMAP1(vcvts_f32_u32, aarch64_neon_vcvtint2fpu, AddRetType | Vectorize1ArgType),
  NEONMAP1(vcvts_n_f32_s32, aarch64_neon_vcvtfxs2fp_n, AddRetType | Vectorize1ArgType),
  NEONMAP1(vcvts_n_f32_u32, aarch64_neon_vcvtfxu2fp_n, AddRetType | Vectorize1ArgType),
  NEONMAP1(vcvts_n_s32_f32, aarch64_neon_vcvtfp2fxs_n, VectorRet | Add1ArgType),
  NEONMAP1(vcvts_n_u32_f32, aarch64_neon_vcvtfp2fxu_n, VectorRet | Add1ArgType),
  NEONMAP1(vcvts_s32_f32, aarch64_neon_fcvtzs, VectorRet | Add1ArgType),
  NEONMAP1(vcvts_u32_f32, aarch64_neon_fcvtzu, VectorRet | Add1ArgType),
  NEONMAP1(vcvtxd_f32_f64, aarch64_neon_fcvtxn, 0),
  NEONMAP0(vdupb_lane_i8),
  NEONMAP0(vdupb_laneq_i8),
  NEONMAP0(vdupd_lane_f64),
  NEONMAP0(vdupd_lane_i64),
  NEONMAP0(vdupd_laneq_f64),
  NEONMAP0(vdupd_laneq_i64),
  NEONMAP0(vduph_lane_i16),
  NEONMAP0(vduph_laneq_i16),
  NEONMAP0(vdups_lane_f32),
  NEONMAP0(vdups_lane_i32),
  NEONMAP0(vdups_laneq_f32),
  NEONMAP0(vdups_laneq_i32),
  NEONMAP0(vfmad_lane_f64),
  NEONMAP0(vfmad_laneq_f64),
  NEONMAP0(vfmas_lane_f32),
  NEONMAP0(vfmas_laneq_f32),
  NEONMAP0(vget_lane_f32),
  NEONMAP0(vget_lane_f64),
  NEONMAP0(vget_lane_i16),
  NEONMAP0(vget_lane_i32),
  NEONMAP0(vget_lane_i64),
  NEONMAP0(vget_lane_i8),
  NEONMAP0(vgetq_lane_f32),
  NEONMAP0(vgetq_lane_f64),
  NEONMAP0(vgetq_lane_i16),
  NEONMAP0(vgetq_lane_i32),
  NEONMAP0(vgetq_lane_i64),
  NEONMAP0(vgetq_lane_i8),
  NEONMAP1(vmaxnmv_f32, aarch64_neon_vpfmaxnm, AddRetType | Add1ArgType),
  NEONMAP1(vmaxnmvq_f32, aarch64_neon_vmaxnmv, 0),
  NEONMAP1(vmaxnmvq_f64, aarch64_neon_vpfmaxnm, AddRetType | Add1ArgType),
  NEONMAP1(vmaxv_f32, aarch64_neon_vpmax, AddRetType | Add1ArgType),
  NEONMAP1(vmaxv_s16, aarch64_neon_smaxv, VectorRet | Add1ArgType),
  NEONMAP1(vmaxv_s32, aarch64_neon_smaxv, VectorRet | Add1ArgType),
  NEONMAP1(vmaxv_s8, aarch64_neon_smaxv, VectorRet | Add1ArgType),
  NEONMAP1(vmaxv_u16, aarch64_neon_umaxv, VectorRet | Add1ArgType),
  NEONMAP1(vmaxv_u32, aarch64_neon_umaxv, VectorRet | Add1ArgType),
  NEONMAP1(vmaxv_u8, aarch64_neon_umaxv, VectorRet | Add1ArgType),
  NEONMAP1(vmaxvq_f32, aarch64_neon_vmaxv, 0),
  NEONMAP1(vmaxvq_f64, aarch64_neon_vpmax, AddRetType | Add1ArgType),
  NEONMAP1(vmaxvq_s16, aarch64_neon_smaxv, VectorRet | Add1ArgType),
  NEONMAP1(vmaxvq_s32, aarch64_neon_smaxv, VectorRet | Add1ArgType),
  NEONMAP1(vmaxvq_s8, aarch64_neon_smaxv, VectorRet | Add1ArgType),
  NEONMAP1(vmaxvq_u16, aarch64_neon_umaxv, VectorRet | Add1ArgType),
  NEONMAP1(vmaxvq_u32, aarch64_neon_umaxv, VectorRet | Add1ArgType),
  NEONMAP1(vmaxvq_u8, aarch64_neon_umaxv, VectorRet | Add1ArgType),
  NEONMAP1(vminnmv_f32, aarch64_neon_vpfminnm, AddRetType | Add1ArgType),
  NEONMAP1(vminnmvq_f32, aarch64_neon_vminnmv, 0),
  NEONMAP1(vminnmvq_f64, aarch64_neon_vpfminnm, AddRetType | Add1ArgType),
  NEONMAP1(vminv_f32, aarch64_neon_vpmin, AddRetType | Add1ArgType),
  NEONMAP1(vminv_s16, aarch64_neon_sminv, VectorRet | Add1ArgType),
  NEONMAP1(vminv_s32, aarch64_neon_sminv, VectorRet | Add1ArgType),
  NEONMAP1(vminv_s8, aarch64_neon_sminv, VectorRet | Add1ArgType),
  NEONMAP1(vminv_u16, aarch64_neon_uminv, VectorRet | Add1ArgType),
  NEONMAP1(vminv_u32, aarch64_neon_uminv, VectorRet | Add1ArgType),
  NEONMAP1(vminv_u8, aarch64_neon_uminv, VectorRet | Add1ArgType),
  NEONMAP1(vminvq_f32, aarch64_neon_vminv, 0),
  NEONMAP1(vminvq_f64, aarch64_neon_vpmin, AddRetType | Add1ArgType),
  NEONMAP1(vminvq_s16, aarch64_neon_sminv, VectorRet | Add1ArgType),
  NEONMAP1(vminvq_s32, aarch64_neon_sminv, VectorRet | Add1ArgType),
  NEONMAP1(vminvq_s8, aarch64_neon_sminv, VectorRet | Add1ArgType),
  NEONMAP1(vminvq_u16, aarch64_neon_uminv, VectorRet | Add1ArgType),
  NEONMAP1(vminvq_u32, aarch64_neon_uminv, VectorRet | Add1ArgType),
  NEONMAP1(vminvq_u8, aarch64_neon_uminv, VectorRet | Add1ArgType),
  NEONMAP0(vmul_n_f64),
  NEONMAP1(vmull_p64, aarch64_neon_vmull_p64, 0),
  NEONMAP0(vmulxd_f64),
  NEONMAP0(vmulxs_f32),
  NEONMAP1(vnegd_s64, aarch64_neon_vneg, 0),
  NEONMAP1(vpaddd_f64, aarch64_neon_vpfadd, AddRetType | Add1ArgType),
  NEONMAP1(vpaddd_s64, aarch64_neon_vpadd, 0),
  NEONMAP1(vpaddd_u64, aarch64_neon_vpadd, 0),
  NEONMAP1(vpadds_f32, aarch64_neon_vpfadd, AddRetType | Add1ArgType),
  NEONMAP1(vpmaxnmqd_f64, aarch64_neon_vpfmaxnm, AddRetType | Add1ArgType),
  NEONMAP1(vpmaxnms_f32, aarch64_neon_vpfmaxnm, AddRetType | Add1ArgType),
  NEONMAP1(vpmaxqd_f64, aarch64_neon_vpmax, AddRetType | Add1ArgType),
  NEONMAP1(vpmaxs_f32, aarch64_neon_vpmax, AddRetType | Add1ArgType),
  NEONMAP1(vpminnmqd_f64, aarch64_neon_vpfminnm, AddRetType | Add1ArgType),
  NEONMAP1(vpminnms_f32, aarch64_neon_vpfminnm, AddRetType | Add1ArgType),
  NEONMAP1(vpminqd_f64, aarch64_neon_vpmin, AddRetType | Add1ArgType),
  NEONMAP1(vpmins_f32, aarch64_neon_vpmin, AddRetType | Add1ArgType),
  NEONMAP1(vqabsb_s8, arm_neon_vqabs, VectorRet),
  NEONMAP1(vqabsd_s64, arm_neon_vqabs, VectorRet),
  NEONMAP1(vqabsh_s16, arm_neon_vqabs, VectorRet),
  NEONMAP1(vqabss_s32, arm_neon_vqabs, VectorRet),
  NEONMAP1(vqaddb_s8, arm_neon_vqadds, VectorRet),
  NEONMAP1(vqaddb_u8, arm_neon_vqaddu, VectorRet),
  NEONMAP1(vqaddd_s64, arm_neon_vqadds, VectorRet),
  NEONMAP1(vqaddd_u64, arm_neon_vqaddu, VectorRet),
  NEONMAP1(vqaddh_s16, arm_neon_vqadds, VectorRet),
  NEONMAP1(vqaddh_u16, arm_neon_vqaddu, VectorRet),
  NEONMAP1(vqadds_s32, arm_neon_vqadds, VectorRet),
  NEONMAP1(vqadds_u32, arm_neon_vqaddu, VectorRet),
  NEONMAP0(vqdmlalh_lane_s16),
  NEONMAP0(vqdmlalh_laneq_s16),
  NEONMAP1(vqdmlalh_s16, aarch64_neon_vqdmlal, VectorRet),
  NEONMAP0(vqdmlals_lane_s32),
  NEONMAP0(vqdmlals_laneq_s32),
  NEONMAP1(vqdmlals_s32, aarch64_neon_vqdmlal, VectorRet),
  NEONMAP0(vqdmlslh_lane_s16),
  NEONMAP0(vqdmlslh_laneq_s16),
  NEONMAP1(vqdmlslh_s16, aarch64_neon_vqdmlsl, VectorRet),
  NEONMAP0(vqdmlsls_lane_s32),
  NEONMAP0(vqdmlsls_laneq_s32),
  NEONMAP1(vqdmlsls_s32, aarch64_neon_vqdmlsl, VectorRet),
  NEONMAP1(vqdmulhh_s16, arm_neon_vqdmulh, VectorRet),
  NEONMAP1(vqdmulhs_s32, arm_neon_vqdmulh, VectorRet),
  NEONMAP1(vqdmullh_s16, arm_neon_vqdmull, VectorRet),
  NEONMAP1(vqdmulls_s32, arm_neon_vqdmull, VectorRet),
  NEONMAP1(vqmovnd_s64, arm_neon_vqmovns, VectorRet),
  NEONMAP1(vqmovnd_u64, arm_neon_vqmovnu, VectorRet),
  NEONMAP1(vqmovnh_s16, arm_neon_vqmovns, VectorRet),
  NEONMAP1(vqmovnh_u16, arm_neon_vqmovnu, VectorRet),
  NEONMAP1(vqmovns_s32, arm_neon_vqmovns, VectorRet),
  NEONMAP1(vqmovns_u32, arm_neon_vqmovnu, VectorRet),
  NEONMAP1(vqmovund_s64, arm_neon_vqmovnsu, VectorRet),
  NEONMAP1(vqmovunh_s16, arm_neon_vqmovnsu, VectorRet),
  NEONMAP1(vqmovuns_s32, arm_neon_vqmovnsu, VectorRet),
  NEONMAP1(vqnegb_s8, arm_neon_vqneg, VectorRet),
  NEONMAP1(vqnegd_s64, arm_neon_vqneg, VectorRet),
  NEONMAP1(vqnegh_s16, arm_neon_vqneg, VectorRet),
  NEONMAP1(vqnegs_s32, arm_neon_vqneg, VectorRet),
  NEONMAP1(vqrdmulhh_s16, arm_neon_vqrdmulh, VectorRet),
  NEONMAP1(vqrdmulhs_s32, arm_neon_vqrdmulh, VectorRet),
  NEONMAP1(vqrshlb_s8, aarch64_neon_vqrshls, VectorRet),
  NEONMAP1(vqrshlb_u8, aarch64_neon_vqrshlu, VectorRet),
  NEONMAP1(vqrshld_s64, aarch64_neon_vqrshls, VectorRet),
  NEONMAP1(vqrshld_u64, aarch64_neon_vqrshlu, VectorRet),
  NEONMAP1(vqrshlh_s16, aarch64_neon_vqrshls, VectorRet),
  NEONMAP1(vqrshlh_u16, aarch64_neon_vqrshlu, VectorRet),
  NEONMAP1(vqrshls_s32, aarch64_neon_vqrshls, VectorRet),
  NEONMAP1(vqrshls_u32, aarch64_neon_vqrshlu, VectorRet),
  NEONMAP1(vqrshrnd_n_s64, aarch64_neon_vsqrshrn, VectorRet),
  NEONMAP1(vqrshrnd_n_u64, aarch64_neon_vuqrshrn, VectorRet),
  NEONMAP1(vqrshrnh_n_s16, aarch64_neon_vsqrshrn, VectorRet),
  NEONMAP1(vqrshrnh_n_u16, aarch64_neon_vuqrshrn, VectorRet),
  NEONMAP1(vqrshrns_n_s32, aarch64_neon_vsqrshrn, VectorRet),
  NEONMAP1(vqrshrns_n_u32, aarch64_neon_vuqrshrn, VectorRet),
  NEONMAP1(vqrshrund_n_s64, aarch64_neon_vsqrshrun, VectorRet),
  NEONMAP1(vqrshrunh_n_s16, aarch64_neon_vsqrshrun, VectorRet),
  NEONMAP1(vqrshruns_n_s32, aarch64_neon_vsqrshrun, VectorRet),
  NEONMAP1(vqshlb_n_s8, aarch64_neon_vqshls_n, VectorRet),
  NEONMAP1(vqshlb_n_u8, aarch64_neon_vqshlu_n, VectorRet),
  NEONMAP1(vqshlb_s8, aarch64_neon_vqshls, VectorRet),
  NEONMAP1(vqshlb_u8, aarch64_neon_vqshlu, VectorRet),
  NEONMAP1(vqshld_n_s64, aarch64_neon_vqshls_n, VectorRet),
  NEONMAP1(vqshld_n_u64, aarch64_neon_vqshlu_n, VectorRet),
  NEONMAP1(vqshld_s64, aarch64_neon_vqshls, VectorRet),
  NEONMAP1(vqshld_u64, aarch64_neon_vqshlu, VectorRet),
  NEONMAP1(vqshlh_n_s16, aarch64_neon_vqshls_n, VectorRet),
  NEONMAP1(vqshlh_n_u16, aarch64_neon_vqshlu_n, VectorRet),
  NEONMAP1(vqshlh_s16, aarch64_neon_vqshls, VectorRet),
  NEONMAP1(vqshlh_u16, aarch64_neon_vqshlu, VectorRet),
  NEONMAP1(vqshls_n_s32, aarch64_neon_vqshls_n, VectorRet),
  NEONMAP1(vqshls_n_u32, aarch64_neon_vqshlu_n, VectorRet),
  NEONMAP1(vqshls_s32, aarch64_neon_vqshls, VectorRet),
  NEONMAP1(vqshls_u32, aarch64_neon_vqshlu, VectorRet),
  NEONMAP1(vqshlub_n_s8, aarch64_neon_vsqshlu, VectorRet),
  NEONMAP1(vqshlud_n_s64, aarch64_neon_vsqshlu, VectorRet),
  NEONMAP1(vqshluh_n_s16, aarch64_neon_vsqshlu, VectorRet),
  NEONMAP1(vqshlus_n_s32, aarch64_neon_vsqshlu, VectorRet),
  NEONMAP1(vqshrnd_n_s64, aarch64_neon_vsqshrn, VectorRet),
  NEONMAP1(vqshrnd_n_u64, aarch64_neon_vuqshrn, VectorRet),
  NEONMAP1(vqshrnh_n_s16, aarch64_neon_vsqshrn, VectorRet),
  NEONMAP1(vqshrnh_n_u16, aarch64_neon_vuqshrn, VectorRet),
  NEONMAP1(vqshrns_n_s32, aarch64_neon_vsqshrn, VectorRet),
  NEONMAP1(vqshrns_n_u32, aarch64_neon_vuqshrn, VectorRet),
  NEONMAP1(vqshrund_n_s64, aarch64_neon_vsqshrun, VectorRet),
  NEONMAP1(vqshrunh_n_s16, aarch64_neon_vsqshrun, VectorRet),
  NEONMAP1(vqshruns_n_s32, aarch64_neon_vsqshrun, VectorRet),
  NEONMAP1(vqsubb_s8, arm_neon_vqsubs, VectorRet),
  NEONMAP1(vqsubb_u8, arm_neon_vqsubu, VectorRet),
  NEONMAP1(vqsubd_s64, arm_neon_vqsubs, VectorRet),
  NEONMAP1(vqsubd_u64, arm_neon_vqsubu, VectorRet),
  NEONMAP1(vqsubh_s16, arm_neon_vqsubs, VectorRet),
  NEONMAP1(vqsubh_u16, arm_neon_vqsubu, VectorRet),
  NEONMAP1(vqsubs_s32, arm_neon_vqsubs, VectorRet),
  NEONMAP1(vqsubs_u32, arm_neon_vqsubu, VectorRet),
  NEONMAP1(vrecped_f64, aarch64_neon_vrecpe, AddRetType),
  NEONMAP1(vrecpes_f32, aarch64_neon_vrecpe, AddRetType),
  NEONMAP1(vrecpsd_f64, aarch64_neon_vrecps, AddRetType),
  NEONMAP1(vrecpss_f32, aarch64_neon_vrecps, AddRetType),
  NEONMAP1(vrecpxd_f64, aarch64_neon_vrecpx, AddRetType),
  NEONMAP1(vrecpxs_f32, aarch64_neon_vrecpx, AddRetType),
  NEONMAP1(vrshld_s64, aarch64_neon_vrshlds, 0),
  NEONMAP1(vrshld_u64, aarch64_neon_vrshldu, 0),
  NEONMAP1(vrshrd_n_s64, aarch64_neon_vsrshr, VectorRet),
  NEONMAP1(vrshrd_n_u64, aarch64_neon_vurshr, VectorRet),
  NEONMAP1(vrsqrted_f64, aarch64_neon_vrsqrte, AddRetType),
  NEONMAP1(vrsqrtes_f32, aarch64_neon_vrsqrte, AddRetType),
  NEONMAP1(vrsqrtsd_f64, aarch64_neon_vrsqrts, AddRetType),
  NEONMAP1(vrsqrtss_f32, aarch64_neon_vrsqrts, AddRetType),
  NEONMAP1(vrsrad_n_s64, aarch64_neon_vrsrads_n, 0),
  NEONMAP1(vrsrad_n_u64, aarch64_neon_vrsradu_n, 0),
  NEONMAP0(vset_lane_f32),
  NEONMAP0(vset_lane_f64),
  NEONMAP0(vset_lane_i16),
  NEONMAP0(vset_lane_i32),
  NEONMAP0(vset_lane_i64),
  NEONMAP0(vset_lane_i8),
  NEONMAP0(vsetq_lane_f32),
  NEONMAP0(vsetq_lane_f64),
  NEONMAP0(vsetq_lane_i16),
  NEONMAP0(vsetq_lane_i32),
  NEONMAP0(vsetq_lane_i64),
  NEONMAP0(vsetq_lane_i8),
  NEONMAP1(vsha1cq_u32, arm_neon_sha1c, 0),
  NEONMAP1(vsha1h_u32, arm_neon_sha1h, 0),
  NEONMAP1(vsha1mq_u32, arm_neon_sha1m, 0),
  NEONMAP1(vsha1pq_u32, arm_neon_sha1p, 0),
  NEONMAP1(vshld_n_s64, aarch64_neon_vshld_n, 0),
  NEONMAP1(vshld_n_u64, aarch64_neon_vshld_n, 0),
  NEONMAP1(vshld_s64, aarch64_neon_vshlds, 0),
  NEONMAP1(vshld_u64, aarch64_neon_vshldu, 0),
  NEONMAP1(vshrd_n_s64, aarch64_neon_vshrds_n, 0),
  NEONMAP1(vshrd_n_u64, aarch64_neon_vshrdu_n, 0),
  NEONMAP1(vslid_n_s64, aarch64_neon_vsli, VectorRet),
  NEONMAP1(vslid_n_u64, aarch64_neon_vsli, VectorRet),
  NEONMAP1(vsqaddb_u8, aarch64_neon_vsqadd, VectorRet),
  NEONMAP1(vsqaddd_u64, aarch64_neon_vsqadd, VectorRet),
  NEONMAP1(vsqaddh_u16, aarch64_neon_vsqadd, VectorRet),
  NEONMAP1(vsqadds_u32, aarch64_neon_vsqadd, VectorRet),
  NEONMAP1(vsrad_n_s64, aarch64_neon_vsrads_n, 0),
  NEONMAP1(vsrad_n_u64, aarch64_neon_vsradu_n, 0),
  NEONMAP1(vsrid_n_s64, aarch64_neon_vsri, VectorRet),
  NEONMAP1(vsrid_n_u64, aarch64_neon_vsri, VectorRet),
  NEONMAP1(vsubd_s64, aarch64_neon_vsubds, 0),
  NEONMAP1(vsubd_u64, aarch64_neon_vsubdu, 0),
  NEONMAP1(vtstd_s64, aarch64_neon_vtstd, VectorRetGetArgs01),
  NEONMAP1(vtstd_u64, aarch64_neon_vtstd, VectorRetGetArgs01),
  NEONMAP1(vuqaddb_s8, aarch64_neon_vuqadd, VectorRet),
  NEONMAP1(vuqaddd_s64, aarch64_neon_vuqadd, VectorRet),
  NEONMAP1(vuqaddh_s16, aarch64_neon_vuqadd, VectorRet),
  NEONMAP1(vuqadds_s32, aarch64_neon_vuqadd, VectorRet)
};

static NeonIntrinsicInfo ARMSIMDIntrinsicMap [] = {
  NEONMAP2(vabd_v, arm_neon_vabdu, arm_neon_vabds, Add1ArgType | UnsignedAlts),
  NEONMAP2(vabdq_v, arm_neon_vabdu, arm_neon_vabds, Add1ArgType | UnsignedAlts),
  NEONMAP1(vabs_v, arm_neon_vabs, 0),
  NEONMAP1(vabsq_v, arm_neon_vabs, 0),
  NEONMAP0(vaddhn_v),
  NEONMAP1(vaesdq_v, arm_neon_aesd, 0),
  NEONMAP1(vaeseq_v, arm_neon_aese, 0),
  NEONMAP1(vaesimcq_v, arm_neon_aesimc, 0),
  NEONMAP1(vaesmcq_v, arm_neon_aesmc, 0),
  NEONMAP1(vbsl_v, arm_neon_vbsl, AddRetType),
  NEONMAP1(vbslq_v, arm_neon_vbsl, AddRetType),
  NEONMAP1(vcage_v, arm_neon_vacge, 0),
  NEONMAP1(vcageq_v, arm_neon_vacge, 0),
  NEONMAP1(vcagt_v, arm_neon_vacgt, 0),
  NEONMAP1(vcagtq_v, arm_neon_vacgt, 0),
  NEONMAP1(vcale_v, arm_neon_vacge, 0),
  NEONMAP1(vcaleq_v, arm_neon_vacge, 0),
  NEONMAP1(vcalt_v, arm_neon_vacgt, 0),
  NEONMAP1(vcaltq_v, arm_neon_vacgt, 0),
  NEONMAP1(vcls_v, arm_neon_vcls, Add1ArgType),
  NEONMAP1(vclsq_v, arm_neon_vcls, Add1ArgType),
  NEONMAP1(vclz_v, ctlz, Add1ArgType),
  NEONMAP1(vclzq_v, ctlz, Add1ArgType),
  NEONMAP1(vcnt_v, ctpop, Add1ArgType),
  NEONMAP1(vcntq_v, ctpop, Add1ArgType),
  NEONMAP1(vcvt_f16_v, arm_neon_vcvtfp2hf, 0),
  NEONMAP1(vcvt_f32_f16, arm_neon_vcvthf2fp, 0),
  NEONMAP0(vcvt_f32_v),
  NEONMAP2(vcvt_n_f32_v, arm_neon_vcvtfxu2fp, arm_neon_vcvtfxs2fp, 0),
  NEONMAP1(vcvt_n_s32_v, arm_neon_vcvtfp2fxs, 0),
  NEONMAP1(vcvt_n_s64_v, arm_neon_vcvtfp2fxs, 0),
  NEONMAP1(vcvt_n_u32_v, arm_neon_vcvtfp2fxu, 0),
  NEONMAP1(vcvt_n_u64_v, arm_neon_vcvtfp2fxu, 0),
  NEONMAP0(vcvt_s32_v),
  NEONMAP0(vcvt_s64_v),
  NEONMAP0(vcvt_u32_v),
  NEONMAP0(vcvt_u64_v),
  NEONMAP1(vcvta_s32_v, arm_neon_vcvtas, 0),
  NEONMAP1(vcvta_s64_v, arm_neon_vcvtas, 0),
  NEONMAP1(vcvta_u32_v, arm_neon_vcvtau, 0),
  NEONMAP1(vcvta_u64_v, arm_neon_vcvtau, 0),
  NEONMAP1(vcvtaq_s32_v, arm_neon_vcvtas, 0),
  NEONMAP1(vcvtaq_s64_v, arm_neon_vcvtas, 0),
  NEONMAP1(vcvtaq_u32_v, arm_neon_vcvtau, 0),
  NEONMAP1(vcvtaq_u64_v, arm_neon_vcvtau, 0),
  NEONMAP1(vcvtm_s32_v, arm_neon_vcvtms, 0),
  NEONMAP1(vcvtm_s64_v, arm_neon_vcvtms, 0),
  NEONMAP1(vcvtm_u32_v, arm_neon_vcvtmu, 0),
  NEONMAP1(vcvtm_u64_v, arm_neon_vcvtmu, 0),
  NEONMAP1(vcvtmq_s32_v, arm_neon_vcvtms, 0),
  NEONMAP1(vcvtmq_s64_v, arm_neon_vcvtms, 0),
  NEONMAP1(vcvtmq_u32_v, arm_neon_vcvtmu, 0),
  NEONMAP1(vcvtmq_u64_v, arm_neon_vcvtmu, 0),
  NEONMAP1(vcvtn_s32_v, arm_neon_vcvtns, 0),
  NEONMAP1(vcvtn_s64_v, arm_neon_vcvtns, 0),
  NEONMAP1(vcvtn_u32_v, arm_neon_vcvtnu, 0),
  NEONMAP1(vcvtn_u64_v, arm_neon_vcvtnu, 0),
  NEONMAP1(vcvtnq_s32_v, arm_neon_vcvtns, 0),
  NEONMAP1(vcvtnq_s64_v, arm_neon_vcvtns, 0),
  NEONMAP1(vcvtnq_u32_v, arm_neon_vcvtnu, 0),
  NEONMAP1(vcvtnq_u64_v, arm_neon_vcvtnu, 0),
  NEONMAP1(vcvtp_s32_v, arm_neon_vcvtps, 0),
  NEONMAP1(vcvtp_s64_v, arm_neon_vcvtps, 0),
  NEONMAP1(vcvtp_u32_v, arm_neon_vcvtpu, 0),
  NEONMAP1(vcvtp_u64_v, arm_neon_vcvtpu, 0),
  NEONMAP1(vcvtpq_s32_v, arm_neon_vcvtps, 0),
  NEONMAP1(vcvtpq_s64_v, arm_neon_vcvtps, 0),
  NEONMAP1(vcvtpq_u32_v, arm_neon_vcvtpu, 0),
  NEONMAP1(vcvtpq_u64_v, arm_neon_vcvtpu, 0),
  NEONMAP0(vcvtq_f32_v),
  NEONMAP2(vcvtq_n_f32_v, arm_neon_vcvtfxu2fp, arm_neon_vcvtfxs2fp, 0),
  NEONMAP1(vcvtq_n_s32_v, arm_neon_vcvtfp2fxs, 0),
  NEONMAP1(vcvtq_n_s64_v, arm_neon_vcvtfp2fxs, 0),
  NEONMAP1(vcvtq_n_u32_v, arm_neon_vcvtfp2fxu, 0),
  NEONMAP1(vcvtq_n_u64_v, arm_neon_vcvtfp2fxu, 0),
  NEONMAP0(vcvtq_s32_v),
  NEONMAP0(vcvtq_s64_v),
  NEONMAP0(vcvtq_u32_v),
  NEONMAP0(vcvtq_u64_v),
  NEONMAP0(vext_v),
  NEONMAP0(vextq_v),
  NEONMAP0(vfma_v),
  NEONMAP0(vfmaq_v),
  NEONMAP2(vhadd_v, arm_neon_vhaddu, arm_neon_vhadds, Add1ArgType | UnsignedAlts),
  NEONMAP2(vhaddq_v, arm_neon_vhaddu, arm_neon_vhadds, Add1ArgType | UnsignedAlts),
  NEONMAP2(vhsub_v, arm_neon_vhsubu, arm_neon_vhsubs, Add1ArgType | UnsignedAlts),
  NEONMAP2(vhsubq_v, arm_neon_vhsubu, arm_neon_vhsubs, Add1ArgType | UnsignedAlts),
  NEONMAP0(vld1_dup_v),
  NEONMAP1(vld1_v, arm_neon_vld1, 0),
  NEONMAP0(vld1q_dup_v),
  NEONMAP1(vld1q_v, arm_neon_vld1, 0),
  NEONMAP1(vld2_lane_v, arm_neon_vld2lane, 0),
  NEONMAP1(vld2_v, arm_neon_vld2, 0),
  NEONMAP1(vld2q_lane_v, arm_neon_vld2lane, 0),
  NEONMAP1(vld2q_v, arm_neon_vld2, 0),
  NEONMAP1(vld3_lane_v, arm_neon_vld3lane, 0),
  NEONMAP1(vld3_v, arm_neon_vld3, 0),
  NEONMAP1(vld3q_lane_v, arm_neon_vld3lane, 0),
  NEONMAP1(vld3q_v, arm_neon_vld3, 0),
  NEONMAP1(vld4_lane_v, arm_neon_vld4lane, 0),
  NEONMAP1(vld4_v, arm_neon_vld4, 0),
  NEONMAP1(vld4q_lane_v, arm_neon_vld4lane, 0),
  NEONMAP1(vld4q_v, arm_neon_vld4, 0),
  NEONMAP2(vmax_v, arm_neon_vmaxu, arm_neon_vmaxs, Add1ArgType | UnsignedAlts),
  NEONMAP2(vmaxq_v, arm_neon_vmaxu, arm_neon_vmaxs, Add1ArgType | UnsignedAlts),
  NEONMAP2(vmin_v, arm_neon_vminu, arm_neon_vmins, Add1ArgType | UnsignedAlts),
  NEONMAP2(vminq_v, arm_neon_vminu, arm_neon_vmins, Add1ArgType | UnsignedAlts),
  NEONMAP0(vmovl_v),
  NEONMAP0(vmovn_v),
  NEONMAP1(vmul_v, arm_neon_vmulp, Add1ArgType),
  NEONMAP0(vmull_v),
  NEONMAP1(vmulq_v, arm_neon_vmulp, Add1ArgType),
  NEONMAP2(vpadal_v, arm_neon_vpadalu, arm_neon_vpadals, UnsignedAlts),
  NEONMAP2(vpadalq_v, arm_neon_vpadalu, arm_neon_vpadals, UnsignedAlts),
  NEONMAP1(vpadd_v, arm_neon_vpadd, Add1ArgType),
  NEONMAP2(vpaddl_v, arm_neon_vpaddlu, arm_neon_vpaddls, UnsignedAlts),
  NEONMAP2(vpaddlq_v, arm_neon_vpaddlu, arm_neon_vpaddls, UnsignedAlts),
  NEONMAP1(vpaddq_v, arm_neon_vpadd, Add1ArgType),
  NEONMAP2(vpmax_v, arm_neon_vpmaxu, arm_neon_vpmaxs, Add1ArgType | UnsignedAlts),
  NEONMAP2(vpmin_v, arm_neon_vpminu, arm_neon_vpmins, Add1ArgType | UnsignedAlts),
  NEONMAP1(vqabs_v, arm_neon_vqabs, Add1ArgType),
  NEONMAP1(vqabsq_v, arm_neon_vqabs, Add1ArgType),
  NEONMAP2(vqadd_v, arm_neon_vqaddu, arm_neon_vqadds, Add1ArgType | UnsignedAlts),
  NEONMAP2(vqaddq_v, arm_neon_vqaddu, arm_neon_vqadds, Add1ArgType | UnsignedAlts),
  NEONMAP2(vqdmlal_v, arm_neon_vqdmull, arm_neon_vqadds, 0),
  NEONMAP2(vqdmlsl_v, arm_neon_vqdmull, arm_neon_vqsubs, 0),
  NEONMAP1(vqdmulh_v, arm_neon_vqdmulh, Add1ArgType),
  NEONMAP1(vqdmulhq_v, arm_neon_vqdmulh, Add1ArgType),
  NEONMAP1(vqdmull_v, arm_neon_vqdmull, Add1ArgType),
  NEONMAP2(vqmovn_v, arm_neon_vqmovnu, arm_neon_vqmovns, Add1ArgType | UnsignedAlts),
  NEONMAP1(vqmovun_v, arm_neon_vqmovnsu, Add1ArgType),
  NEONMAP1(vqneg_v, arm_neon_vqneg, Add1ArgType),
  NEONMAP1(vqnegq_v, arm_neon_vqneg, Add1ArgType),
  NEONMAP1(vqrdmulh_v, arm_neon_vqrdmulh, Add1ArgType),
  NEONMAP1(vqrdmulhq_v, arm_neon_vqrdmulh, Add1ArgType),
  NEONMAP2(vqrshl_v, arm_neon_vqrshiftu, arm_neon_vqrshifts, Add1ArgType | UnsignedAlts),
  NEONMAP2(vqrshlq_v, arm_neon_vqrshiftu, arm_neon_vqrshifts, Add1ArgType | UnsignedAlts),
  NEONMAP2(vqshl_n_v, arm_neon_vqshiftu, arm_neon_vqshifts, UnsignedAlts),
  NEONMAP2(vqshl_v, arm_neon_vqshiftu, arm_neon_vqshifts, Add1ArgType | UnsignedAlts),
  NEONMAP2(vqshlq_n_v, arm_neon_vqshiftu, arm_neon_vqshifts, UnsignedAlts),
  NEONMAP2(vqshlq_v, arm_neon_vqshiftu, arm_neon_vqshifts, Add1ArgType | UnsignedAlts),
  NEONMAP2(vqsub_v, arm_neon_vqsubu, arm_neon_vqsubs, Add1ArgType | UnsignedAlts),
  NEONMAP2(vqsubq_v, arm_neon_vqsubu, arm_neon_vqsubs, Add1ArgType | UnsignedAlts),
  NEONMAP1(vraddhn_v, arm_neon_vraddhn, Add1ArgType),
  NEONMAP2(vrecpe_v, arm_neon_vrecpe, arm_neon_vrecpe, 0),
  NEONMAP2(vrecpeq_v, arm_neon_vrecpe, arm_neon_vrecpe, 0),
  NEONMAP1(vrecps_v, arm_neon_vrecps, Add1ArgType),
  NEONMAP1(vrecpsq_v, arm_neon_vrecps, Add1ArgType),
  NEONMAP2(vrhadd_v, arm_neon_vrhaddu, arm_neon_vrhadds, Add1ArgType | UnsignedAlts),
  NEONMAP2(vrhaddq_v, arm_neon_vrhaddu, arm_neon_vrhadds, Add1ArgType | UnsignedAlts),
  NEONMAP2(vrshl_v, arm_neon_vrshiftu, arm_neon_vrshifts, Add1ArgType | UnsignedAlts),
  NEONMAP2(vrshlq_v, arm_neon_vrshiftu, arm_neon_vrshifts, Add1ArgType | UnsignedAlts),
  NEONMAP2(vrsqrte_v, arm_neon_vrsqrte, arm_neon_vrsqrte, 0),
  NEONMAP2(vrsqrteq_v, arm_neon_vrsqrte, arm_neon_vrsqrte, 0),
  NEONMAP1(vrsqrts_v, arm_neon_vrsqrts, Add1ArgType),
  NEONMAP1(vrsqrtsq_v, arm_neon_vrsqrts, Add1ArgType),
  NEONMAP1(vrsubhn_v, arm_neon_vrsubhn, Add1ArgType),
  NEONMAP1(vsha1su0q_v, arm_neon_sha1su0, 0),
  NEONMAP1(vsha1su1q_v, arm_neon_sha1su1, 0),
  NEONMAP1(vsha256h2q_v, arm_neon_sha256h2, 0),
  NEONMAP1(vsha256hq_v, arm_neon_sha256h, 0),
  NEONMAP1(vsha256su0q_v, arm_neon_sha256su0, 0),
  NEONMAP1(vsha256su1q_v, arm_neon_sha256su1, 0),
  NEONMAP0(vshl_n_v),
  NEONMAP2(vshl_v, arm_neon_vshiftu, arm_neon_vshifts, Add1ArgType | UnsignedAlts),
  NEONMAP0(vshll_n_v),
  NEONMAP0(vshlq_n_v),
  NEONMAP2(vshlq_v, arm_neon_vshiftu, arm_neon_vshifts, Add1ArgType | UnsignedAlts),
  NEONMAP0(vshr_n_v),
  NEONMAP0(vshrn_n_v),
  NEONMAP0(vshrq_n_v),
  NEONMAP1(vst1_v, arm_neon_vst1, 0),
  NEONMAP1(vst1q_v, arm_neon_vst1, 0),
  NEONMAP1(vst2_lane_v, arm_neon_vst2lane, 0),
  NEONMAP1(vst2_v, arm_neon_vst2, 0),
  NEONMAP1(vst2q_lane_v, arm_neon_vst2lane, 0),
  NEONMAP1(vst2q_v, arm_neon_vst2, 0),
  NEONMAP1(vst3_lane_v, arm_neon_vst3lane, 0),
  NEONMAP1(vst3_v, arm_neon_vst3, 0),
  NEONMAP1(vst3q_lane_v, arm_neon_vst3lane, 0),
  NEONMAP1(vst3q_v, arm_neon_vst3, 0),
  NEONMAP1(vst4_lane_v, arm_neon_vst4lane, 0),
  NEONMAP1(vst4_v, arm_neon_vst4, 0),
  NEONMAP1(vst4q_lane_v, arm_neon_vst4lane, 0),
  NEONMAP1(vst4q_v, arm_neon_vst4, 0),
  NEONMAP0(vsubhn_v),
  NEONMAP0(vtrn_v),
  NEONMAP0(vtrnq_v),
  NEONMAP0(vtst_v),
  NEONMAP0(vtstq_v),
  NEONMAP0(vuzp_v),
  NEONMAP0(vuzpq_v),
  NEONMAP0(vzip_v),
  NEONMAP0(vzipq_v)
};

#undef NEONMAP0
#undef NEONMAP1
#undef NEONMAP2

static bool NEONSIMDIntrinsicsProvenSorted = false;

static bool AArch64SISDIntrinsicInfoProvenSorted = false;

static const NeonIntrinsicInfo *
findNeonIntrinsicInMap(llvm::ArrayRef<NeonIntrinsicInfo> IntrinsicMap,
                       unsigned BuiltinID, bool &MapProvenSorted) {

#ifndef NDEBUG
  if (!MapProvenSorted) {
    // FIXME: use std::is_sorted once C++11 is allowed
    for (unsigned i = 0; i < IntrinsicMap.size() - 1; ++i)
      assert(IntrinsicMap[i].BuiltinID <= IntrinsicMap[i + 1].BuiltinID);
    MapProvenSorted = true;
  }
#endif

  const NeonIntrinsicInfo *Builtin =
      std::lower_bound(IntrinsicMap.begin(), IntrinsicMap.end(), BuiltinID);

  if (Builtin != IntrinsicMap.end() && Builtin->BuiltinID == BuiltinID)
    return Builtin;

  return 0;
}

Function *CodeGenFunction::LookupNeonLLVMIntrinsic(unsigned IntrinsicID,
                                                   unsigned Modifier,
                                                   llvm::Type *ArgType,
                                                   const CallExpr *E) {
  // Return type.
  SmallVector<llvm::Type *, 3> Tys;
  if (Modifier & AddRetType) {
    llvm::Type *Ty = ConvertType(E->getCallReturnType());
    if (Modifier & VectorizeRetType)
      Ty = llvm::VectorType::get(Ty, 1);

    Tys.push_back(Ty);
  }

  // Arguments.
  if (Modifier & VectorizeArgTypes)
    ArgType = llvm::VectorType::get(ArgType, 1);

  if (Modifier & (Add1ArgType | Add2ArgTypes))
    Tys.push_back(ArgType);

  if (Modifier & Add2ArgTypes)
    Tys.push_back(ArgType);

  if (Modifier & InventFloatType)
    Tys.push_back(FloatTy);

  return CGM.getIntrinsic(IntrinsicID, Tys);
}


static Value *EmitAArch64ScalarBuiltinExpr(CodeGenFunction &CGF,
                                           const NeonIntrinsicInfo &SISDInfo,
                                           const CallExpr *E) {
  unsigned BuiltinID = SISDInfo.BuiltinID;
  unsigned int Int = SISDInfo.LLVMIntrinsic;
  unsigned IntTypes = SISDInfo.TypeModifier;
  const char *s = SISDInfo.NameHint;

  SmallVector<Value *, 4> Ops;
  for (unsigned i = 0, e = E->getNumArgs(); i != e; i++) {
    Ops.push_back(CGF.EmitScalarExpr(E->getArg(i)));
  }

  // AArch64 scalar builtins are not overloaded, they do not have an extra
  // argument that specifies the vector type, need to handle each case.
  switch (BuiltinID) {
  default: break;
  case NEON::BI__builtin_neon_vdups_lane_f32:
  case NEON::BI__builtin_neon_vdupd_lane_f64:
  case NEON::BI__builtin_neon_vdups_laneq_f32:
  case NEON::BI__builtin_neon_vdupd_laneq_f64: {
    return CGF.Builder.CreateExtractElement(Ops[0], Ops[1], "vdup_lane");
  }
  case NEON::BI__builtin_neon_vdupb_lane_i8:
  case NEON::BI__builtin_neon_vduph_lane_i16:
  case NEON::BI__builtin_neon_vdups_lane_i32:
  case NEON::BI__builtin_neon_vdupd_lane_i64:
  case NEON::BI__builtin_neon_vdupb_laneq_i8:
  case NEON::BI__builtin_neon_vduph_laneq_i16:
  case NEON::BI__builtin_neon_vdups_laneq_i32:
  case NEON::BI__builtin_neon_vdupd_laneq_i64: {
    // The backend treats Neon scalar types as v1ix types
    // So we want to dup lane from any vector to v1ix vector
    // with shufflevector
    s = "vdup_lane";
    Value* SV = llvm::ConstantVector::getSplat(1, cast<ConstantInt>(Ops[1]));
    Value *Result = CGF.Builder.CreateShuffleVector(Ops[0], Ops[0], SV, s);
    llvm::Type *Ty = CGF.ConvertType(E->getCallReturnType());
    // AArch64 intrinsic one-element vector type cast to
    // scalar type expected by the builtin
    return CGF.Builder.CreateBitCast(Result, Ty, s);
  }
  case NEON::BI__builtin_neon_vqdmlalh_lane_s16 :
  case NEON::BI__builtin_neon_vqdmlalh_laneq_s16 :
  case NEON::BI__builtin_neon_vqdmlals_lane_s32 :
  case NEON::BI__builtin_neon_vqdmlals_laneq_s32 :
  case NEON::BI__builtin_neon_vqdmlslh_lane_s16 :
  case NEON::BI__builtin_neon_vqdmlslh_laneq_s16 :
  case NEON::BI__builtin_neon_vqdmlsls_lane_s32 :
  case NEON::BI__builtin_neon_vqdmlsls_laneq_s32 : {
    Int = Intrinsic::arm_neon_vqadds;
    if (BuiltinID == NEON::BI__builtin_neon_vqdmlslh_lane_s16 ||
        BuiltinID == NEON::BI__builtin_neon_vqdmlslh_laneq_s16 ||
        BuiltinID == NEON::BI__builtin_neon_vqdmlsls_lane_s32 ||
        BuiltinID == NEON::BI__builtin_neon_vqdmlsls_laneq_s32) {
      Int = Intrinsic::arm_neon_vqsubs;
    }
    // create vqdmull call with b * c[i]
    llvm::Type *Ty = CGF.ConvertType(E->getArg(1)->getType());
    llvm::VectorType *OpVTy = llvm::VectorType::get(Ty, 1);
    Ty = CGF.ConvertType(E->getArg(0)->getType());
    llvm::VectorType *ResVTy = llvm::VectorType::get(Ty, 1);
    Value *F = CGF.CGM.getIntrinsic(Intrinsic::arm_neon_vqdmull, ResVTy);
    Value *V = UndefValue::get(OpVTy);
    llvm::Constant *CI = ConstantInt::get(CGF.Int32Ty, 0);
    SmallVector<Value *, 2> MulOps;
    MulOps.push_back(Ops[1]);
    MulOps.push_back(Ops[2]);
    MulOps[0] = CGF.Builder.CreateInsertElement(V, MulOps[0], CI);
    MulOps[1] = CGF.Builder.CreateExtractElement(MulOps[1], Ops[3], "extract");
    MulOps[1] = CGF.Builder.CreateInsertElement(V, MulOps[1], CI);
    Value *MulRes = CGF.Builder.CreateCall2(F, MulOps[0], MulOps[1]);
    // create vqadds call with a +/- vqdmull result
    F = CGF.CGM.getIntrinsic(Int, ResVTy);
    SmallVector<Value *, 2> AddOps;
    AddOps.push_back(Ops[0]);
    AddOps.push_back(MulRes);
    V = UndefValue::get(ResVTy);
    AddOps[0] = CGF.Builder.CreateInsertElement(V, AddOps[0], CI);
    Value *AddRes = CGF.Builder.CreateCall2(F, AddOps[0], AddOps[1]);
    return CGF.Builder.CreateBitCast(AddRes, Ty);
  }
  case NEON::BI__builtin_neon_vfmas_lane_f32:
  case NEON::BI__builtin_neon_vfmas_laneq_f32:
  case NEON::BI__builtin_neon_vfmad_lane_f64:
  case NEON::BI__builtin_neon_vfmad_laneq_f64: {
    llvm::Type *Ty = CGF.ConvertType(E->getCallReturnType());
    Value *F = CGF.CGM.getIntrinsic(Intrinsic::fma, Ty);
    Ops[2] = CGF.Builder.CreateExtractElement(Ops[2], Ops[3], "extract");
    return CGF.Builder.CreateCall3(F, Ops[1], Ops[2], Ops[0]);
  }
  // Scalar Floating-point Multiply Extended
  case NEON::BI__builtin_neon_vmulxs_f32:
  case NEON::BI__builtin_neon_vmulxd_f64: {
    Int = Intrinsic::aarch64_neon_vmulx;
    llvm::Type *Ty = CGF.ConvertType(E->getCallReturnType());
    return CGF.EmitNeonCall(CGF.CGM.getIntrinsic(Int, Ty), Ops, "vmulx");
  }
  case NEON::BI__builtin_neon_vmul_n_f64: {
    // v1f64 vmul_n_f64  should be mapped to Neon scalar mul lane
    llvm::Type *VTy = GetNeonType(&CGF,
      NeonTypeFlags(NeonTypeFlags::Float64, false, false));
    Ops[0] = CGF.Builder.CreateBitCast(Ops[0], VTy);
    llvm::Value *Idx = llvm::ConstantInt::get(CGF.Int32Ty, 0);
    Ops[0] = CGF.Builder.CreateExtractElement(Ops[0], Idx, "extract");
    Value *Result = CGF.Builder.CreateFMul(Ops[0], Ops[1]);
    return CGF.Builder.CreateBitCast(Result, VTy);
  }
  case NEON::BI__builtin_neon_vget_lane_i8:
  case NEON::BI__builtin_neon_vget_lane_i16:
  case NEON::BI__builtin_neon_vget_lane_i32:
  case NEON::BI__builtin_neon_vget_lane_i64:
  case NEON::BI__builtin_neon_vget_lane_f32:
  case NEON::BI__builtin_neon_vget_lane_f64:
  case NEON::BI__builtin_neon_vgetq_lane_i8:
  case NEON::BI__builtin_neon_vgetq_lane_i16:
  case NEON::BI__builtin_neon_vgetq_lane_i32:
  case NEON::BI__builtin_neon_vgetq_lane_i64:
  case NEON::BI__builtin_neon_vgetq_lane_f32:
  case NEON::BI__builtin_neon_vgetq_lane_f64:
    return CGF.EmitARMBuiltinExpr(NEON::BI__builtin_neon_vget_lane_i8, E);
  case NEON::BI__builtin_neon_vset_lane_i8:
  case NEON::BI__builtin_neon_vset_lane_i16:
  case NEON::BI__builtin_neon_vset_lane_i32:
  case NEON::BI__builtin_neon_vset_lane_i64:
  case NEON::BI__builtin_neon_vset_lane_f32:
  case NEON::BI__builtin_neon_vset_lane_f64:
  case NEON::BI__builtin_neon_vsetq_lane_i8:
  case NEON::BI__builtin_neon_vsetq_lane_i16:
  case NEON::BI__builtin_neon_vsetq_lane_i32:
  case NEON::BI__builtin_neon_vsetq_lane_i64:
  case NEON::BI__builtin_neon_vsetq_lane_f32:
  case NEON::BI__builtin_neon_vsetq_lane_f64:
    return CGF.EmitARMBuiltinExpr(NEON::BI__builtin_neon_vset_lane_i8, E);

  case NEON::BI__builtin_neon_vcled_s64:
  case NEON::BI__builtin_neon_vcled_u64:
  case NEON::BI__builtin_neon_vcles_f32:
  case NEON::BI__builtin_neon_vcled_f64:
  case NEON::BI__builtin_neon_vcltd_s64:
  case NEON::BI__builtin_neon_vcltd_u64:
  case NEON::BI__builtin_neon_vclts_f32:
  case NEON::BI__builtin_neon_vcltd_f64:
  case NEON::BI__builtin_neon_vcales_f32:
  case NEON::BI__builtin_neon_vcaled_f64:
  case NEON::BI__builtin_neon_vcalts_f32:
  case NEON::BI__builtin_neon_vcaltd_f64:
    // Only one direction of comparisons actually exist, cmle is actually a cmge
    // with swapped operands. The table gives us the right intrinsic but we
    // still need to do the swap.
    std::swap(Ops[0], Ops[1]);
    break;
  case NEON::BI__builtin_neon_vceqzd_s64:
  case NEON::BI__builtin_neon_vceqzd_u64:
  case NEON::BI__builtin_neon_vcgezd_s64:
  case NEON::BI__builtin_neon_vcgtzd_s64:
  case NEON::BI__builtin_neon_vclezd_s64:
  case NEON::BI__builtin_neon_vcltzd_s64:
    // Add implicit zero operand.
    Ops.push_back(llvm::Constant::getNullValue(Ops[0]->getType()));
    break;
  case NEON::BI__builtin_neon_vceqzs_f32:
  case NEON::BI__builtin_neon_vceqzd_f64:
  case NEON::BI__builtin_neon_vcgezs_f32:
  case NEON::BI__builtin_neon_vcgezd_f64:
  case NEON::BI__builtin_neon_vcgtzs_f32:
  case NEON::BI__builtin_neon_vcgtzd_f64:
  case NEON::BI__builtin_neon_vclezs_f32:
  case NEON::BI__builtin_neon_vclezd_f64:
  case NEON::BI__builtin_neon_vcltzs_f32:
  case NEON::BI__builtin_neon_vcltzd_f64:
    // Add implicit zero operand.
    Ops.push_back(llvm::Constant::getNullValue(CGF.FloatTy));
    break;
  }


  assert(Int && "Generic code assumes a valid intrinsic");

  // Determine the type(s) of this overloaded AArch64 intrinsic.
  const Expr *Arg = E->getArg(0);
  llvm::Type *ArgTy = CGF.ConvertType(Arg->getType());
  Function *F = CGF.LookupNeonLLVMIntrinsic(Int, IntTypes, ArgTy, E);

  Value *Result = CGF.EmitNeonCall(F, Ops, s);
  llvm::Type *ResultType = CGF.ConvertType(E->getType());
  // AArch64 intrinsic one-element vector type cast to
  // scalar type expected by the builtin
  return CGF.Builder.CreateBitCast(Result, ResultType, s);
}

Value *CodeGenFunction::EmitCommonNeonBuiltinExpr(
    unsigned BuiltinID, unsigned LLVMIntrinsic, unsigned AltLLVMIntrinsic,
    const char *NameHint, unsigned Modifier, const CallExpr *E,
    SmallVectorImpl<llvm::Value *> &Ops, llvm::Value *Align) {
  // Get the last argument, which specifies the vector type.
  llvm::APSInt NeonTypeConst;
  const Expr *Arg = E->getArg(E->getNumArgs() - 1);
  if (!Arg->isIntegerConstantExpr(NeonTypeConst, getContext()))
    return 0;

  // Determine the type of this overloaded NEON intrinsic.
  NeonTypeFlags Type(NeonTypeConst.getZExtValue());
  bool Usgn = Type.isUnsigned();
  bool Quad = Type.isQuad();

  llvm::VectorType *VTy = GetNeonType(this, Type);
  llvm::Type *Ty = VTy;
  if (!Ty)
    return 0;

  unsigned Int = LLVMIntrinsic;
  if ((Modifier & UnsignedAlts) && !Usgn)
    Int = AltLLVMIntrinsic;

  switch (BuiltinID) {
  default: break;
  case NEON::BI__builtin_neon_vabs_v:
  case NEON::BI__builtin_neon_vabsq_v:
    if (VTy->getElementType()->isFloatingPointTy())
      return EmitNeonCall(CGM.getIntrinsic(Intrinsic::fabs, Ty), Ops, "vabs");
    return EmitNeonCall(CGM.getIntrinsic(LLVMIntrinsic, Ty), Ops, "vabs");
  case NEON::BI__builtin_neon_vaddhn_v: {
    llvm::VectorType *SrcTy =
        llvm::VectorType::getExtendedElementVectorType(VTy);

    // %sum = add <4 x i32> %lhs, %rhs
    Ops[0] = Builder.CreateBitCast(Ops[0], SrcTy);
    Ops[1] = Builder.CreateBitCast(Ops[1], SrcTy);
    Ops[0] = Builder.CreateAdd(Ops[0], Ops[1], "vaddhn");

    // %high = lshr <4 x i32> %sum, <i32 16, i32 16, i32 16, i32 16>
    Constant *ShiftAmt = ConstantInt::get(SrcTy->getElementType(),
                                       SrcTy->getScalarSizeInBits() / 2);
    ShiftAmt = ConstantVector::getSplat(VTy->getNumElements(), ShiftAmt);
    Ops[0] = Builder.CreateLShr(Ops[0], ShiftAmt, "vaddhn");

    // %res = trunc <4 x i32> %high to <4 x i16>
    return Builder.CreateTrunc(Ops[0], VTy, "vaddhn");
  }
  case NEON::BI__builtin_neon_vcale_v:
  case NEON::BI__builtin_neon_vcaleq_v:
  case NEON::BI__builtin_neon_vcalt_v:
  case NEON::BI__builtin_neon_vcaltq_v:
    std::swap(Ops[0], Ops[1]);
  case NEON::BI__builtin_neon_vcage_v:
  case NEON::BI__builtin_neon_vcageq_v:
  case NEON::BI__builtin_neon_vcagt_v:
  case NEON::BI__builtin_neon_vcagtq_v: {
    llvm::Type *VecFlt = llvm::VectorType::get(
        VTy->getScalarSizeInBits() == 32 ? FloatTy : DoubleTy,
        VTy->getNumElements());
    llvm::Type *Tys[] = { VTy, VecFlt };
    Function *F = CGM.getIntrinsic(LLVMIntrinsic, Tys);
    return EmitNeonCall(F, Ops, NameHint);
  }
  case NEON::BI__builtin_neon_vclz_v:
  case NEON::BI__builtin_neon_vclzq_v:
    // We generate target-independent intrinsic, which needs a second argument
    // for whether or not clz of zero is undefined; on ARM it isn't.
    Ops.push_back(Builder.getInt1(getTarget().isCLZForZeroUndef()));
    break;
  case NEON::BI__builtin_neon_vcvt_f32_v:
  case NEON::BI__builtin_neon_vcvtq_f32_v:
    Ops[0] = Builder.CreateBitCast(Ops[0], Ty);
    Ty = GetNeonType(this, NeonTypeFlags(NeonTypeFlags::Float32, false, Quad));
    return Usgn ? Builder.CreateUIToFP(Ops[0], Ty, "vcvt")
                : Builder.CreateSIToFP(Ops[0], Ty, "vcvt");
  case NEON::BI__builtin_neon_vcvt_n_f32_v:
  case NEON::BI__builtin_neon_vcvtq_n_f32_v: {
    bool Double =
      (cast<llvm::IntegerType>(VTy->getElementType())->getBitWidth() == 64);
    llvm::Type *FloatTy =
        GetNeonType(this, NeonTypeFlags(Double ? NeonTypeFlags::Float64
                                               : NeonTypeFlags::Float32,
                                        false, Quad));
    llvm::Type *Tys[2] = { FloatTy, Ty };
    Int = Usgn ? LLVMIntrinsic : AltLLVMIntrinsic;
    Function *F = CGM.getIntrinsic(Int, Tys);
    return EmitNeonCall(F, Ops, "vcvt_n");
  }
  case NEON::BI__builtin_neon_vcvt_n_s32_v:
  case NEON::BI__builtin_neon_vcvt_n_u32_v:
  case NEON::BI__builtin_neon_vcvt_n_s64_v:
  case NEON::BI__builtin_neon_vcvt_n_u64_v:
  case NEON::BI__builtin_neon_vcvtq_n_s32_v:
  case NEON::BI__builtin_neon_vcvtq_n_u32_v:
  case NEON::BI__builtin_neon_vcvtq_n_s64_v:
  case NEON::BI__builtin_neon_vcvtq_n_u64_v: {
    bool Double =
      (cast<llvm::IntegerType>(VTy->getElementType())->getBitWidth() == 64);
    llvm::Type *FloatTy =
        GetNeonType(this, NeonTypeFlags(Double ? NeonTypeFlags::Float64
                                               : NeonTypeFlags::Float32,
                                        false, Quad));
    llvm::Type *Tys[2] = { Ty, FloatTy };
    Function *F = CGM.getIntrinsic(LLVMIntrinsic, Tys);
    return EmitNeonCall(F, Ops, "vcvt_n");
  }
  case NEON::BI__builtin_neon_vcvt_s32_v:
  case NEON::BI__builtin_neon_vcvt_u32_v:
  case NEON::BI__builtin_neon_vcvt_s64_v:
  case NEON::BI__builtin_neon_vcvt_u64_v:
  case NEON::BI__builtin_neon_vcvtq_s32_v:
  case NEON::BI__builtin_neon_vcvtq_u32_v:
  case NEON::BI__builtin_neon_vcvtq_s64_v:
  case NEON::BI__builtin_neon_vcvtq_u64_v: {
    bool Double =
      (cast<llvm::IntegerType>(VTy->getElementType())->getBitWidth() == 64);
    llvm::Type *FloatTy =
        GetNeonType(this, NeonTypeFlags(Double ? NeonTypeFlags::Float64
                                               : NeonTypeFlags::Float32,
                                        false, Quad));
    Ops[0] = Builder.CreateBitCast(Ops[0], FloatTy);
    return Usgn ? Builder.CreateFPToUI(Ops[0], Ty, "vcvt")
                : Builder.CreateFPToSI(Ops[0], Ty, "vcvt");
  }
  case NEON::BI__builtin_neon_vcvta_s32_v:
  case NEON::BI__builtin_neon_vcvta_s64_v:
  case NEON::BI__builtin_neon_vcvta_u32_v:
  case NEON::BI__builtin_neon_vcvta_u64_v:
  case NEON::BI__builtin_neon_vcvtaq_s32_v:
  case NEON::BI__builtin_neon_vcvtaq_s64_v:
  case NEON::BI__builtin_neon_vcvtaq_u32_v:
  case NEON::BI__builtin_neon_vcvtaq_u64_v:
  case NEON::BI__builtin_neon_vcvtn_s32_v:
  case NEON::BI__builtin_neon_vcvtn_s64_v:
  case NEON::BI__builtin_neon_vcvtn_u32_v:
  case NEON::BI__builtin_neon_vcvtn_u64_v:
  case NEON::BI__builtin_neon_vcvtnq_s32_v:
  case NEON::BI__builtin_neon_vcvtnq_s64_v:
  case NEON::BI__builtin_neon_vcvtnq_u32_v:
  case NEON::BI__builtin_neon_vcvtnq_u64_v:
  case NEON::BI__builtin_neon_vcvtp_s32_v:
  case NEON::BI__builtin_neon_vcvtp_s64_v:
  case NEON::BI__builtin_neon_vcvtp_u32_v:
  case NEON::BI__builtin_neon_vcvtp_u64_v:
  case NEON::BI__builtin_neon_vcvtpq_s32_v:
  case NEON::BI__builtin_neon_vcvtpq_s64_v:
  case NEON::BI__builtin_neon_vcvtpq_u32_v:
  case NEON::BI__builtin_neon_vcvtpq_u64_v:
  case NEON::BI__builtin_neon_vcvtm_s32_v:
  case NEON::BI__builtin_neon_vcvtm_s64_v:
  case NEON::BI__builtin_neon_vcvtm_u32_v:
  case NEON::BI__builtin_neon_vcvtm_u64_v:
  case NEON::BI__builtin_neon_vcvtmq_s32_v:
  case NEON::BI__builtin_neon_vcvtmq_s64_v:
  case NEON::BI__builtin_neon_vcvtmq_u32_v:
  case NEON::BI__builtin_neon_vcvtmq_u64_v: {
    bool Double =
      (cast<llvm::IntegerType>(VTy->getElementType())->getBitWidth() == 64);
    llvm::Type *InTy =
      GetNeonType(this,
                  NeonTypeFlags(Double ? NeonTypeFlags::Float64
                                : NeonTypeFlags::Float32, false, Quad));
    llvm::Type *Tys[2] = { Ty, InTy };
    return EmitNeonCall(CGM.getIntrinsic(LLVMIntrinsic, Tys), Ops, NameHint);
  }
  case NEON::BI__builtin_neon_vext_v:
  case NEON::BI__builtin_neon_vextq_v: {
    int CV = cast<ConstantInt>(Ops[2])->getSExtValue();
    SmallVector<Constant*, 16> Indices;
    for (unsigned i = 0, e = VTy->getNumElements(); i != e; ++i)
      Indices.push_back(ConstantInt::get(Int32Ty, i+CV));

    Ops[0] = Builder.CreateBitCast(Ops[0], Ty);
    Ops[1] = Builder.CreateBitCast(Ops[1], Ty);
    Value *SV = llvm::ConstantVector::get(Indices);
    return Builder.CreateShuffleVector(Ops[0], Ops[1], SV, "vext");
  }
  case NEON::BI__builtin_neon_vfma_v:
  case NEON::BI__builtin_neon_vfmaq_v: {
    Value *F = CGM.getIntrinsic(Intrinsic::fma, Ty);
    Ops[0] = Builder.CreateBitCast(Ops[0], Ty);
    Ops[1] = Builder.CreateBitCast(Ops[1], Ty);
    Ops[2] = Builder.CreateBitCast(Ops[2], Ty);

    // NEON intrinsic puts accumulator first, unlike the LLVM fma.
    return Builder.CreateCall3(F, Ops[1], Ops[2], Ops[0]);
  }
  case NEON::BI__builtin_neon_vld1_v:
  case NEON::BI__builtin_neon_vld1q_v:
    Ops.push_back(Align);
    return EmitNeonCall(CGM.getIntrinsic(LLVMIntrinsic, Ty), Ops, "vld1");
  case NEON::BI__builtin_neon_vld2_v:
  case NEON::BI__builtin_neon_vld2q_v:
  case NEON::BI__builtin_neon_vld3_v:
  case NEON::BI__builtin_neon_vld3q_v:
  case NEON::BI__builtin_neon_vld4_v:
  case NEON::BI__builtin_neon_vld4q_v: {
    Function *F = CGM.getIntrinsic(LLVMIntrinsic, Ty);
    Ops[1] = Builder.CreateCall2(F, Ops[1], Align, NameHint);
    Ty = llvm::PointerType::getUnqual(Ops[1]->getType());
    Ops[0] = Builder.CreateBitCast(Ops[0], Ty);
    return Builder.CreateStore(Ops[1], Ops[0]);
  }
  case NEON::BI__builtin_neon_vld1_dup_v:
  case NEON::BI__builtin_neon_vld1q_dup_v: {
    Value *V = UndefValue::get(Ty);
    Ty = llvm::PointerType::getUnqual(VTy->getElementType());
    Ops[0] = Builder.CreateBitCast(Ops[0], Ty);
    LoadInst *Ld = Builder.CreateLoad(Ops[0]);
    Ld->setAlignment(cast<ConstantInt>(Align)->getZExtValue());
    llvm::Constant *CI = ConstantInt::get(Int32Ty, 0);
    Ops[0] = Builder.CreateInsertElement(V, Ld, CI);
    return EmitNeonSplat(Ops[0], CI);
  }
  case NEON::BI__builtin_neon_vld2_lane_v:
  case NEON::BI__builtin_neon_vld2q_lane_v:
  case NEON::BI__builtin_neon_vld3_lane_v:
  case NEON::BI__builtin_neon_vld3q_lane_v:
  case NEON::BI__builtin_neon_vld4_lane_v:
  case NEON::BI__builtin_neon_vld4q_lane_v: {
    Function *F = CGM.getIntrinsic(LLVMIntrinsic, Ty);
    for (unsigned I = 2; I < Ops.size() - 1; ++I)
      Ops[I] = Builder.CreateBitCast(Ops[I], Ty);
    Ops.push_back(Align);
    Ops[1] = Builder.CreateCall(F, makeArrayRef(Ops).slice(1), NameHint);
    Ty = llvm::PointerType::getUnqual(Ops[1]->getType());
    Ops[0] = Builder.CreateBitCast(Ops[0], Ty);
    return Builder.CreateStore(Ops[1], Ops[0]);
  }
  case NEON::BI__builtin_neon_vmovl_v: {
    llvm::Type *DTy =llvm::VectorType::getTruncatedElementVectorType(VTy);
    Ops[0] = Builder.CreateBitCast(Ops[0], DTy);
    if (Usgn)
      return Builder.CreateZExt(Ops[0], Ty, "vmovl");
    return Builder.CreateSExt(Ops[0], Ty, "vmovl");
  }
  case NEON::BI__builtin_neon_vmovn_v: {
    llvm::Type *QTy = llvm::VectorType::getExtendedElementVectorType(VTy);
    Ops[0] = Builder.CreateBitCast(Ops[0], QTy);
    return Builder.CreateTrunc(Ops[0], Ty, "vmovn");
  }
  case NEON::BI__builtin_neon_vmull_v:
    // FIXME: the integer vmull operations could be emitted in terms of pure
    // LLVM IR (2 exts followed by a mul). Unfortunately LLVM has a habit of
    // hoisting the exts outside loops. Until global ISel comes along that can
    // see through such movement this leads to bad CodeGen. So we need an
    // intrinsic for now.
    Int = Usgn ? Intrinsic::arm_neon_vmullu : Intrinsic::arm_neon_vmulls;
    Int = Type.isPoly() ? (unsigned)Intrinsic::arm_neon_vmullp : Int;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vmull");
  case NEON::BI__builtin_neon_vpadal_v:
  case NEON::BI__builtin_neon_vpadalq_v: {
    // The source operand type has twice as many elements of half the size.
    unsigned EltBits = VTy->getElementType()->getPrimitiveSizeInBits();
    llvm::Type *EltTy =
      llvm::IntegerType::get(getLLVMContext(), EltBits / 2);
    llvm::Type *NarrowTy =
      llvm::VectorType::get(EltTy, VTy->getNumElements() * 2);
    llvm::Type *Tys[2] = { Ty, NarrowTy };
    return EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, NameHint);
  }
  case NEON::BI__builtin_neon_vpaddl_v:
  case NEON::BI__builtin_neon_vpaddlq_v: {
    // The source operand type has twice as many elements of half the size.
    unsigned EltBits = VTy->getElementType()->getPrimitiveSizeInBits();
    llvm::Type *EltTy = llvm::IntegerType::get(getLLVMContext(), EltBits / 2);
    llvm::Type *NarrowTy =
      llvm::VectorType::get(EltTy, VTy->getNumElements() * 2);
    llvm::Type *Tys[2] = { Ty, NarrowTy };
    return EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "vpaddl");
  }
  case NEON::BI__builtin_neon_vqdmlal_v:
  case NEON::BI__builtin_neon_vqdmlsl_v: {
    SmallVector<Value *, 2> MulOps(Ops.begin() + 1, Ops.end());
    Value *Mul = EmitNeonCall(CGM.getIntrinsic(LLVMIntrinsic, Ty),
                              MulOps, "vqdmlal");

    SmallVector<Value *, 2> AccumOps;
    AccumOps.push_back(Ops[0]);
    AccumOps.push_back(Mul);
    return EmitNeonCall(CGM.getIntrinsic(AltLLVMIntrinsic, Ty),
                        AccumOps, NameHint);
  }
  case NEON::BI__builtin_neon_vqshl_n_v:
  case NEON::BI__builtin_neon_vqshlq_n_v:
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vqshl_n",
                        1, false);
  case NEON::BI__builtin_neon_vrecpe_v:
  case NEON::BI__builtin_neon_vrecpeq_v:
  case NEON::BI__builtin_neon_vrsqrte_v:
  case NEON::BI__builtin_neon_vrsqrteq_v:
    Int = Ty->isFPOrFPVectorTy() ? LLVMIntrinsic : AltLLVMIntrinsic;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, NameHint);

  case NEON::BI__builtin_neon_vshl_n_v:
  case NEON::BI__builtin_neon_vshlq_n_v:
    Ops[1] = EmitNeonShiftVector(Ops[1], Ty, false);
    return Builder.CreateShl(Builder.CreateBitCast(Ops[0],Ty), Ops[1],
                             "vshl_n");
  case NEON::BI__builtin_neon_vshll_n_v: {
    llvm::Type *SrcTy = llvm::VectorType::getTruncatedElementVectorType(VTy);
    Ops[0] = Builder.CreateBitCast(Ops[0], SrcTy);
    if (Usgn)
      Ops[0] = Builder.CreateZExt(Ops[0], VTy);
    else
      Ops[0] = Builder.CreateSExt(Ops[0], VTy);
    Ops[1] = EmitNeonShiftVector(Ops[1], VTy, false);
    return Builder.CreateShl(Ops[0], Ops[1], "vshll_n");
  }
  case NEON::BI__builtin_neon_vshrn_n_v: {
    llvm::Type *SrcTy = llvm::VectorType::getExtendedElementVectorType(VTy);
    Ops[0] = Builder.CreateBitCast(Ops[0], SrcTy);
    Ops[1] = EmitNeonShiftVector(Ops[1], SrcTy, false);
    if (Usgn)
      Ops[0] = Builder.CreateLShr(Ops[0], Ops[1]);
    else
      Ops[0] = Builder.CreateAShr(Ops[0], Ops[1]);
    return Builder.CreateTrunc(Ops[0], Ty, "vshrn_n");
  }
  case NEON::BI__builtin_neon_vshr_n_v:
  case NEON::BI__builtin_neon_vshrq_n_v:
    return EmitNeonRShiftImm(Ops[0], Ops[1], Ty, Usgn, "vshr_n");
  case NEON::BI__builtin_neon_vst1_v:
  case NEON::BI__builtin_neon_vst1q_v:
  case NEON::BI__builtin_neon_vst2_v:
  case NEON::BI__builtin_neon_vst2q_v:
  case NEON::BI__builtin_neon_vst3_v:
  case NEON::BI__builtin_neon_vst3q_v:
  case NEON::BI__builtin_neon_vst4_v:
  case NEON::BI__builtin_neon_vst4q_v:
  case NEON::BI__builtin_neon_vst2_lane_v:
  case NEON::BI__builtin_neon_vst2q_lane_v:
  case NEON::BI__builtin_neon_vst3_lane_v:
  case NEON::BI__builtin_neon_vst3q_lane_v:
  case NEON::BI__builtin_neon_vst4_lane_v:
  case NEON::BI__builtin_neon_vst4q_lane_v:
    Ops.push_back(Align);
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "");
  case NEON::BI__builtin_neon_vsubhn_v: {
    llvm::VectorType *SrcTy =
        llvm::VectorType::getExtendedElementVectorType(VTy);

    // %sum = add <4 x i32> %lhs, %rhs
    Ops[0] = Builder.CreateBitCast(Ops[0], SrcTy);
    Ops[1] = Builder.CreateBitCast(Ops[1], SrcTy);
    Ops[0] = Builder.CreateSub(Ops[0], Ops[1], "vsubhn");

    // %high = lshr <4 x i32> %sum, <i32 16, i32 16, i32 16, i32 16>
    Constant *ShiftAmt = ConstantInt::get(SrcTy->getElementType(),
                                       SrcTy->getScalarSizeInBits() / 2);
    ShiftAmt = ConstantVector::getSplat(VTy->getNumElements(), ShiftAmt);
    Ops[0] = Builder.CreateLShr(Ops[0], ShiftAmt, "vsubhn");

    // %res = trunc <4 x i32> %high to <4 x i16>
    return Builder.CreateTrunc(Ops[0], VTy, "vsubhn");
  }
  case NEON::BI__builtin_neon_vtrn_v:
  case NEON::BI__builtin_neon_vtrnq_v: {
    Ops[0] = Builder.CreateBitCast(Ops[0], llvm::PointerType::getUnqual(Ty));
    Ops[1] = Builder.CreateBitCast(Ops[1], Ty);
    Ops[2] = Builder.CreateBitCast(Ops[2], Ty);
    Value *SV = 0;

    for (unsigned vi = 0; vi != 2; ++vi) {
      SmallVector<Constant*, 16> Indices;
      for (unsigned i = 0, e = VTy->getNumElements(); i != e; i += 2) {
        Indices.push_back(Builder.getInt32(i+vi));
        Indices.push_back(Builder.getInt32(i+e+vi));
      }
      Value *Addr = Builder.CreateConstInBoundsGEP1_32(Ops[0], vi);
      SV = llvm::ConstantVector::get(Indices);
      SV = Builder.CreateShuffleVector(Ops[1], Ops[2], SV, "vtrn");
      SV = Builder.CreateStore(SV, Addr);
    }
    return SV;
  }
  case NEON::BI__builtin_neon_vtst_v:
  case NEON::BI__builtin_neon_vtstq_v: {
    Ops[0] = Builder.CreateBitCast(Ops[0], Ty);
    Ops[1] = Builder.CreateBitCast(Ops[1], Ty);
    Ops[0] = Builder.CreateAnd(Ops[0], Ops[1]);
    Ops[0] = Builder.CreateICmp(ICmpInst::ICMP_NE, Ops[0],
                                ConstantAggregateZero::get(Ty));
    return Builder.CreateSExt(Ops[0], Ty, "vtst");
  }
  case NEON::BI__builtin_neon_vuzp_v:
  case NEON::BI__builtin_neon_vuzpq_v: {
    Ops[0] = Builder.CreateBitCast(Ops[0], llvm::PointerType::getUnqual(Ty));
    Ops[1] = Builder.CreateBitCast(Ops[1], Ty);
    Ops[2] = Builder.CreateBitCast(Ops[2], Ty);
    Value *SV = 0;

    for (unsigned vi = 0; vi != 2; ++vi) {
      SmallVector<Constant*, 16> Indices;
      for (unsigned i = 0, e = VTy->getNumElements(); i != e; ++i)
        Indices.push_back(ConstantInt::get(Int32Ty, 2*i+vi));

      Value *Addr = Builder.CreateConstInBoundsGEP1_32(Ops[0], vi);
      SV = llvm::ConstantVector::get(Indices);
      SV = Builder.CreateShuffleVector(Ops[1], Ops[2], SV, "vuzp");
      SV = Builder.CreateStore(SV, Addr);
    }
    return SV;
  }
  case NEON::BI__builtin_neon_vzip_v:
  case NEON::BI__builtin_neon_vzipq_v: {
    Ops[0] = Builder.CreateBitCast(Ops[0], llvm::PointerType::getUnqual(Ty));
    Ops[1] = Builder.CreateBitCast(Ops[1], Ty);
    Ops[2] = Builder.CreateBitCast(Ops[2], Ty);
    Value *SV = 0;

    for (unsigned vi = 0; vi != 2; ++vi) {
      SmallVector<Constant*, 16> Indices;
      for (unsigned i = 0, e = VTy->getNumElements(); i != e; i += 2) {
        Indices.push_back(ConstantInt::get(Int32Ty, (i + vi*e) >> 1));
        Indices.push_back(ConstantInt::get(Int32Ty, ((i + vi*e) >> 1)+e));
      }
      Value *Addr = Builder.CreateConstInBoundsGEP1_32(Ops[0], vi);
      SV = llvm::ConstantVector::get(Indices);
      SV = Builder.CreateShuffleVector(Ops[1], Ops[2], SV, "vzip");
      SV = Builder.CreateStore(SV, Addr);
    }
    return SV;
  }
  }

  assert(Int && "Expected valid intrinsic number");

  // Determine the type(s) of this overloaded AArch64 intrinsic.
  Function *F = LookupNeonLLVMIntrinsic(Int, Modifier, Ty, E);

  Value *Result = EmitNeonCall(F, Ops, NameHint);
  llvm::Type *ResultType = ConvertType(E->getType());
  // AArch64 intrinsic one-element vector type cast to
  // scalar type expected by the builtin
  return Builder.CreateBitCast(Result, ResultType, NameHint);
}

Value *CodeGenFunction::EmitAArch64CompareBuiltinExpr(
    Value *Op, llvm::Type *Ty, const CmpInst::Predicate Fp,
    const CmpInst::Predicate Ip, const Twine &Name) {
  llvm::Type *OTy = ((llvm::User *)Op)->getOperand(0)->getType();
  if (OTy->isPointerTy())
    OTy = Ty;
  Op = Builder.CreateBitCast(Op, OTy);
  if (((llvm::VectorType *)OTy)->getElementType()->isFloatingPointTy()) {
    Op = Builder.CreateFCmp(Fp, Op, ConstantAggregateZero::get(OTy));
  } else {
    Op = Builder.CreateICmp(Ip, Op, ConstantAggregateZero::get(OTy));
  }
  return Builder.CreateSExt(Op, Ty, Name);
}

static Value *packTBLDVectorList(CodeGenFunction &CGF, ArrayRef<Value *> Ops,
                                 Value *ExtOp, Value *IndexOp,
                                 llvm::Type *ResTy, unsigned IntID,
                                 const char *Name) {
  SmallVector<Value *, 2> TblOps;
  if (ExtOp)
    TblOps.push_back(ExtOp);

  // Build a vector containing sequential number like (0, 1, 2, ..., 15)  
  SmallVector<Constant*, 16> Indices;
  llvm::VectorType *TblTy = cast<llvm::VectorType>(Ops[0]->getType());
  for (unsigned i = 0, e = TblTy->getNumElements(); i != e; ++i) {
    Indices.push_back(ConstantInt::get(CGF.Int32Ty, 2*i));
    Indices.push_back(ConstantInt::get(CGF.Int32Ty, 2*i+1));
  }
  Value *SV = llvm::ConstantVector::get(Indices);

  int PairPos = 0, End = Ops.size() - 1;
  while (PairPos < End) {
    TblOps.push_back(CGF.Builder.CreateShuffleVector(Ops[PairPos],
                                                     Ops[PairPos+1], SV, Name));
    PairPos += 2;
  }

  // If there's an odd number of 64-bit lookup table, fill the high 64-bit
  // of the 128-bit lookup table with zero.
  if (PairPos == End) {
    Value *ZeroTbl = ConstantAggregateZero::get(TblTy);
    TblOps.push_back(CGF.Builder.CreateShuffleVector(Ops[PairPos],
                                                     ZeroTbl, SV, Name));
  }

  TblTy = llvm::VectorType::get(TblTy->getElementType(),
                                2*TblTy->getNumElements());
  llvm::Type *Tys[2] = { ResTy, TblTy };

  Function *TblF;
  TblOps.push_back(IndexOp);
  TblF = CGF.CGM.getIntrinsic(IntID, Tys);
  
  return CGF.EmitNeonCall(TblF, TblOps, Name);
}

static Value *EmitAArch64TblBuiltinExpr(CodeGenFunction &CGF,
                                        unsigned BuiltinID,
                                        const CallExpr *E) {
  unsigned int Int = 0;
  const char *s = NULL;

  unsigned TblPos;
  switch (BuiltinID) {
  default:
    return 0;
  case NEON::BI__builtin_neon_vtbl1_v:
  case NEON::BI__builtin_neon_vqtbl1_v:
  case NEON::BI__builtin_neon_vqtbl1q_v:
  case NEON::BI__builtin_neon_vtbl2_v:
  case NEON::BI__builtin_neon_vqtbl2_v:
  case NEON::BI__builtin_neon_vqtbl2q_v:
  case NEON::BI__builtin_neon_vtbl3_v:
  case NEON::BI__builtin_neon_vqtbl3_v:
  case NEON::BI__builtin_neon_vqtbl3q_v:
  case NEON::BI__builtin_neon_vtbl4_v:
  case NEON::BI__builtin_neon_vqtbl4_v:
  case NEON::BI__builtin_neon_vqtbl4q_v:
    TblPos = 0;
    break;
  case NEON::BI__builtin_neon_vtbx1_v:
  case NEON::BI__builtin_neon_vqtbx1_v:
  case NEON::BI__builtin_neon_vqtbx1q_v:
  case NEON::BI__builtin_neon_vtbx2_v:
  case NEON::BI__builtin_neon_vqtbx2_v:
  case NEON::BI__builtin_neon_vqtbx2q_v:
  case NEON::BI__builtin_neon_vtbx3_v:
  case NEON::BI__builtin_neon_vqtbx3_v:
  case NEON::BI__builtin_neon_vqtbx3q_v:
  case NEON::BI__builtin_neon_vtbx4_v:
  case NEON::BI__builtin_neon_vqtbx4_v:
  case NEON::BI__builtin_neon_vqtbx4q_v:
    TblPos = 1;
    break;
  }

  assert(E->getNumArgs() >= 3);

  // Get the last argument, which specifies the vector type.
  llvm::APSInt Result;
  const Expr *Arg = E->getArg(E->getNumArgs() - 1);
  if (!Arg->isIntegerConstantExpr(Result, CGF.getContext()))
    return 0;

  // Determine the type of this overloaded NEON intrinsic.
  NeonTypeFlags Type(Result.getZExtValue());
  llvm::VectorType *VTy = GetNeonType(&CGF, Type);
  llvm::Type *Ty = VTy;
  if (!Ty)
    return 0;

  SmallVector<Value *, 4> Ops;
  for (unsigned i = 0, e = E->getNumArgs() - 1; i != e; i++) {
    Ops.push_back(CGF.EmitScalarExpr(E->getArg(i)));
  }

  Arg = E->getArg(TblPos);
  llvm::Type *TblTy = CGF.ConvertType(Arg->getType());
  llvm::VectorType *VTblTy = cast<llvm::VectorType>(TblTy);
  llvm::Type *Tys[2] = { Ty, VTblTy };
  unsigned nElts = VTy->getNumElements();  

  // AArch64 scalar builtins are not overloaded, they do not have an extra
  // argument that specifies the vector type, need to handle each case.
  SmallVector<Value *, 2> TblOps;
  switch (BuiltinID) {
  case NEON::BI__builtin_neon_vtbl1_v: {
    TblOps.push_back(Ops[0]);
    return packTBLDVectorList(CGF, TblOps, 0, Ops[1], Ty,
                              Intrinsic::aarch64_neon_vtbl1, "vtbl1");
  }
  case NEON::BI__builtin_neon_vtbl2_v: {
    TblOps.push_back(Ops[0]);
    TblOps.push_back(Ops[1]);
    return packTBLDVectorList(CGF, TblOps, 0, Ops[2], Ty,
                              Intrinsic::aarch64_neon_vtbl1, "vtbl1");
  }
  case NEON::BI__builtin_neon_vtbl3_v: {
    TblOps.push_back(Ops[0]);
    TblOps.push_back(Ops[1]);
    TblOps.push_back(Ops[2]);
    return packTBLDVectorList(CGF, TblOps, 0, Ops[3], Ty,
                              Intrinsic::aarch64_neon_vtbl2, "vtbl2");
  }
  case NEON::BI__builtin_neon_vtbl4_v: {
    TblOps.push_back(Ops[0]);
    TblOps.push_back(Ops[1]);
    TblOps.push_back(Ops[2]);
    TblOps.push_back(Ops[3]);
    return packTBLDVectorList(CGF, TblOps, 0, Ops[4], Ty,
                              Intrinsic::aarch64_neon_vtbl2, "vtbl2");
  }
  case NEON::BI__builtin_neon_vtbx1_v: {
    TblOps.push_back(Ops[1]);
    Value *TblRes = packTBLDVectorList(CGF, TblOps, 0, Ops[2], Ty,
                                    Intrinsic::aarch64_neon_vtbl1, "vtbl1");

    llvm::Constant *Eight = ConstantInt::get(VTy->getElementType(), 8);
    Value* EightV = llvm::ConstantVector::getSplat(nElts, Eight);
    Value *CmpRes = CGF.Builder.CreateICmp(ICmpInst::ICMP_UGE, Ops[2], EightV);
    CmpRes = CGF.Builder.CreateSExt(CmpRes, Ty);

    SmallVector<Value *, 4> BslOps;
    BslOps.push_back(CmpRes);
    BslOps.push_back(Ops[0]);
    BslOps.push_back(TblRes);
    Function *BslF = CGF.CGM.getIntrinsic(Intrinsic::arm_neon_vbsl, Ty);
    return CGF.EmitNeonCall(BslF, BslOps, "vbsl");
  }
  case NEON::BI__builtin_neon_vtbx2_v: {
    TblOps.push_back(Ops[1]);
    TblOps.push_back(Ops[2]);
    return packTBLDVectorList(CGF, TblOps, Ops[0], Ops[3], Ty,
                              Intrinsic::aarch64_neon_vtbx1, "vtbx1");
  }
  case NEON::BI__builtin_neon_vtbx3_v: {
    TblOps.push_back(Ops[1]);
    TblOps.push_back(Ops[2]);
    TblOps.push_back(Ops[3]);
    Value *TblRes = packTBLDVectorList(CGF, TblOps, 0, Ops[4], Ty,
                                       Intrinsic::aarch64_neon_vtbl2, "vtbl2");

    llvm::Constant *TwentyFour = ConstantInt::get(VTy->getElementType(), 24);
    Value* TwentyFourV = llvm::ConstantVector::getSplat(nElts, TwentyFour);
    Value *CmpRes = CGF.Builder.CreateICmp(ICmpInst::ICMP_UGE, Ops[4],
                                           TwentyFourV);
    CmpRes = CGF.Builder.CreateSExt(CmpRes, Ty);
  
    SmallVector<Value *, 4> BslOps;
    BslOps.push_back(CmpRes);
    BslOps.push_back(Ops[0]);
    BslOps.push_back(TblRes);
    Function *BslF = CGF.CGM.getIntrinsic(Intrinsic::arm_neon_vbsl, Ty);
    return CGF.EmitNeonCall(BslF, BslOps, "vbsl");
  }
  case NEON::BI__builtin_neon_vtbx4_v: {
    TblOps.push_back(Ops[1]);
    TblOps.push_back(Ops[2]);
    TblOps.push_back(Ops[3]);
    TblOps.push_back(Ops[4]);
    return packTBLDVectorList(CGF, TblOps, Ops[0], Ops[5], Ty,
                              Intrinsic::aarch64_neon_vtbx2, "vtbx2");
  }
  case NEON::BI__builtin_neon_vqtbl1_v:
  case NEON::BI__builtin_neon_vqtbl1q_v:
    Int = Intrinsic::aarch64_neon_vtbl1; s = "vtbl1"; break;
  case NEON::BI__builtin_neon_vqtbl2_v:
  case NEON::BI__builtin_neon_vqtbl2q_v: {
    Int = Intrinsic::aarch64_neon_vtbl2; s = "vtbl2"; break;
  case NEON::BI__builtin_neon_vqtbl3_v:
  case NEON::BI__builtin_neon_vqtbl3q_v:
    Int = Intrinsic::aarch64_neon_vtbl3; s = "vtbl3"; break;
  case NEON::BI__builtin_neon_vqtbl4_v:
  case NEON::BI__builtin_neon_vqtbl4q_v:
    Int = Intrinsic::aarch64_neon_vtbl4; s = "vtbl4"; break;
  case NEON::BI__builtin_neon_vqtbx1_v:
  case NEON::BI__builtin_neon_vqtbx1q_v:
    Int = Intrinsic::aarch64_neon_vtbx1; s = "vtbx1"; break;
  case NEON::BI__builtin_neon_vqtbx2_v:
  case NEON::BI__builtin_neon_vqtbx2q_v:
    Int = Intrinsic::aarch64_neon_vtbx2; s = "vtbx2"; break;
  case NEON::BI__builtin_neon_vqtbx3_v:
  case NEON::BI__builtin_neon_vqtbx3q_v:
    Int = Intrinsic::aarch64_neon_vtbx3; s = "vtbx3"; break;
  case NEON::BI__builtin_neon_vqtbx4_v:
  case NEON::BI__builtin_neon_vqtbx4q_v:
    Int = Intrinsic::aarch64_neon_vtbx4; s = "vtbx4"; break;
  }
  }

  if (!Int)
    return 0;

  Function *F = CGF.CGM.getIntrinsic(Int, Tys);
  return CGF.EmitNeonCall(F, Ops, s);
}

Value *CodeGenFunction::EmitAArch64BuiltinExpr(unsigned BuiltinID,
                                               const CallExpr *E) {

  // Process AArch64 scalar builtins
  llvm::ArrayRef<NeonIntrinsicInfo> SISDInfo(AArch64SISDIntrinsicInfo);
  const NeonIntrinsicInfo *Builtin = findNeonIntrinsicInMap(
      SISDInfo, BuiltinID, AArch64SISDIntrinsicInfoProvenSorted);

  if (Builtin) {
    Value *Result = EmitAArch64ScalarBuiltinExpr(*this, *Builtin, E);
    assert(Result && "SISD intrinsic should have been handled");
    return Result;
  }

  // Process AArch64 table lookup builtins
  if (Value *Result = EmitAArch64TblBuiltinExpr(*this, BuiltinID, E))
    return Result;

  if (BuiltinID == AArch64::BI__clear_cache) {
    assert(E->getNumArgs() == 2 &&
           "Variadic __clear_cache slipped through on AArch64");

    const FunctionDecl *FD = E->getDirectCallee();
    SmallVector<Value *, 2> Ops;
    for (unsigned i = 0; i < E->getNumArgs(); i++)
      Ops.push_back(EmitScalarExpr(E->getArg(i)));
    llvm::Type *Ty = CGM.getTypes().ConvertType(FD->getType());
    llvm::FunctionType *FTy = cast<llvm::FunctionType>(Ty);
    StringRef Name = FD->getName();
    return EmitNounwindRuntimeCall(CGM.CreateRuntimeFunction(FTy, Name), Ops);
  }

  SmallVector<Value *, 4> Ops;
  llvm::Value *Align = 0; // Alignment for load/store

  if (BuiltinID == NEON::BI__builtin_neon_vldrq_p128) {
   Value *Op = EmitScalarExpr(E->getArg(0));
   unsigned addressSpace =
     cast<llvm::PointerType>(Op->getType())->getAddressSpace();
   llvm::Type *Ty = llvm::Type::getFP128PtrTy(getLLVMContext(), addressSpace);
   Op = Builder.CreateBitCast(Op, Ty);
   Op = Builder.CreateLoad(Op);
   Ty = llvm::Type::getIntNTy(getLLVMContext(), 128);
   return Builder.CreateBitCast(Op, Ty);
  }
  if (BuiltinID == NEON::BI__builtin_neon_vstrq_p128) {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    unsigned addressSpace =
      cast<llvm::PointerType>(Op0->getType())->getAddressSpace();
    llvm::Type *PTy = llvm::Type::getFP128PtrTy(getLLVMContext(), addressSpace);
    Op0 = Builder.CreateBitCast(Op0, PTy);
    Value *Op1 = EmitScalarExpr(E->getArg(1));
    llvm::Type *Ty = llvm::Type::getFP128Ty(getLLVMContext());
    Op1 = Builder.CreateBitCast(Op1, Ty);
    return Builder.CreateStore(Op1, Op0);
  }
  for (unsigned i = 0, e = E->getNumArgs() - 1; i != e; i++) {
    if (i == 0) {
      switch (BuiltinID) {
      case NEON::BI__builtin_neon_vld1_v:
      case NEON::BI__builtin_neon_vld1q_v:
      case NEON::BI__builtin_neon_vst1_v:
      case NEON::BI__builtin_neon_vst1q_v:
      case NEON::BI__builtin_neon_vst2_v:
      case NEON::BI__builtin_neon_vst2q_v:
      case NEON::BI__builtin_neon_vst3_v:
      case NEON::BI__builtin_neon_vst3q_v:
      case NEON::BI__builtin_neon_vst4_v:
      case NEON::BI__builtin_neon_vst4q_v:
      case NEON::BI__builtin_neon_vst1_x2_v:
      case NEON::BI__builtin_neon_vst1q_x2_v:
      case NEON::BI__builtin_neon_vst1_x3_v:
      case NEON::BI__builtin_neon_vst1q_x3_v:
      case NEON::BI__builtin_neon_vst1_x4_v:
      case NEON::BI__builtin_neon_vst1q_x4_v:
      // Handle ld1/st1 lane in this function a little different from ARM.
      case NEON::BI__builtin_neon_vld1_lane_v:
      case NEON::BI__builtin_neon_vld1q_lane_v:
      case NEON::BI__builtin_neon_vst1_lane_v:
      case NEON::BI__builtin_neon_vst1q_lane_v:
      case NEON::BI__builtin_neon_vst2_lane_v:
      case NEON::BI__builtin_neon_vst2q_lane_v:
      case NEON::BI__builtin_neon_vst3_lane_v:
      case NEON::BI__builtin_neon_vst3q_lane_v:
      case NEON::BI__builtin_neon_vst4_lane_v:
      case NEON::BI__builtin_neon_vst4q_lane_v:
      case NEON::BI__builtin_neon_vld1_dup_v:
      case NEON::BI__builtin_neon_vld1q_dup_v:
        // Get the alignment for the argument in addition to the value;
        // we'll use it later.
        std::pair<llvm::Value *, unsigned> Src =
            EmitPointerWithAlignment(E->getArg(0));
        Ops.push_back(Src.first);
        Align = Builder.getInt32(Src.second);
        continue;
      }
    }
    if (i == 1) {
      switch (BuiltinID) {
      case NEON::BI__builtin_neon_vld2_v:
      case NEON::BI__builtin_neon_vld2q_v:
      case NEON::BI__builtin_neon_vld3_v:
      case NEON::BI__builtin_neon_vld3q_v:
      case NEON::BI__builtin_neon_vld4_v:
      case NEON::BI__builtin_neon_vld4q_v:
      case NEON::BI__builtin_neon_vld1_x2_v:
      case NEON::BI__builtin_neon_vld1q_x2_v:
      case NEON::BI__builtin_neon_vld1_x3_v:
      case NEON::BI__builtin_neon_vld1q_x3_v:
      case NEON::BI__builtin_neon_vld1_x4_v:
      case NEON::BI__builtin_neon_vld1q_x4_v:
      // Handle ld1/st1 dup lane in this function a little different from ARM.
      case NEON::BI__builtin_neon_vld2_dup_v:
      case NEON::BI__builtin_neon_vld2q_dup_v:
      case NEON::BI__builtin_neon_vld3_dup_v:
      case NEON::BI__builtin_neon_vld3q_dup_v:
      case NEON::BI__builtin_neon_vld4_dup_v:
      case NEON::BI__builtin_neon_vld4q_dup_v:
      case NEON::BI__builtin_neon_vld2_lane_v:
      case NEON::BI__builtin_neon_vld2q_lane_v:
      case NEON::BI__builtin_neon_vld3_lane_v:
      case NEON::BI__builtin_neon_vld3q_lane_v:
      case NEON::BI__builtin_neon_vld4_lane_v:
      case NEON::BI__builtin_neon_vld4q_lane_v:
        // Get the alignment for the argument in addition to the value;
        // we'll use it later.
        std::pair<llvm::Value *, unsigned> Src =
            EmitPointerWithAlignment(E->getArg(1));
        Ops.push_back(Src.first);
        Align = Builder.getInt32(Src.second);
        continue;
      }
    }
    Ops.push_back(EmitScalarExpr(E->getArg(i)));
  }

  // Get the last argument, which specifies the vector type.
  llvm::APSInt Result;
  const Expr *Arg = E->getArg(E->getNumArgs() - 1);
  if (!Arg->isIntegerConstantExpr(Result, getContext()))
    return 0;

  // Determine the type of this overloaded NEON intrinsic.
  NeonTypeFlags Type(Result.getZExtValue());
  bool usgn = Type.isUnsigned();
  bool quad = Type.isQuad();

  llvm::VectorType *VTy = GetNeonType(this, Type);
  llvm::Type *Ty = VTy;
  if (!Ty)
    return 0;


  // Many NEON builtins have identical semantics and uses in ARM and
  // AArch64. Emit these in a single function.
  llvm::ArrayRef<NeonIntrinsicInfo> IntrinsicMap(ARMSIMDIntrinsicMap);
  Builtin = findNeonIntrinsicInMap(IntrinsicMap, BuiltinID,
                                   NEONSIMDIntrinsicsProvenSorted);
  if (Builtin)
    return EmitCommonNeonBuiltinExpr(
        Builtin->BuiltinID, Builtin->LLVMIntrinsic, Builtin->AltLLVMIntrinsic,
        Builtin->NameHint, Builtin->TypeModifier, E, Ops, Align);

  unsigned Int;
  switch (BuiltinID) {
  default:
    return 0;

  // AArch64 builtins mapping to legacy ARM v7 builtins.
  // FIXME: the mapped builtins listed correspond to what has been tested
  // in aarch64-neon-intrinsics.c so far.

  // Shift by immediate
  case NEON::BI__builtin_neon_vrshr_n_v:
  case NEON::BI__builtin_neon_vrshrq_n_v:
    Int = usgn ? Intrinsic::aarch64_neon_vurshr
               : Intrinsic::aarch64_neon_vsrshr;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vrshr_n");
  case NEON::BI__builtin_neon_vsra_n_v:
    if (VTy->getElementType()->isIntegerTy(64)) {
      Int = usgn ? Intrinsic::aarch64_neon_vsradu_n
                 : Intrinsic::aarch64_neon_vsrads_n;
      return EmitNeonCall(CGM.getIntrinsic(Int), Ops, "vsra_n");
    }
    return EmitARMBuiltinExpr(NEON::BI__builtin_neon_vsra_n_v, E);
  case NEON::BI__builtin_neon_vsraq_n_v:
    return EmitARMBuiltinExpr(NEON::BI__builtin_neon_vsraq_n_v, E);
  case NEON::BI__builtin_neon_vrsra_n_v:
    if (VTy->getElementType()->isIntegerTy(64)) {
      Int = usgn ? Intrinsic::aarch64_neon_vrsradu_n
                 : Intrinsic::aarch64_neon_vrsrads_n;
      return EmitNeonCall(CGM.getIntrinsic(Int), Ops, "vrsra_n");
    }
    // fall through
  case NEON::BI__builtin_neon_vrsraq_n_v: {
    Ops[0] = Builder.CreateBitCast(Ops[0], Ty);
    Ops[1] = Builder.CreateBitCast(Ops[1], Ty);
    Int = usgn ? Intrinsic::aarch64_neon_vurshr
               : Intrinsic::aarch64_neon_vsrshr;
    Ops[1] = Builder.CreateCall2(CGM.getIntrinsic(Int, Ty), Ops[1], Ops[2]);
    return Builder.CreateAdd(Ops[0], Ops[1], "vrsra_n");
  }
  case NEON::BI__builtin_neon_vqshlu_n_v:
  case NEON::BI__builtin_neon_vqshluq_n_v:
    Int = Intrinsic::aarch64_neon_vsqshlu;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vqshlu_n");
  case NEON::BI__builtin_neon_vsri_n_v:
  case NEON::BI__builtin_neon_vsriq_n_v:
    Int = Intrinsic::aarch64_neon_vsri;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vsri_n");
  case NEON::BI__builtin_neon_vsli_n_v:
  case NEON::BI__builtin_neon_vsliq_n_v:
    Int = Intrinsic::aarch64_neon_vsli;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vsli_n");
  case NEON::BI__builtin_neon_vqshrun_n_v:
    Int = Intrinsic::aarch64_neon_vsqshrun;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vqshrun_n");
  case NEON::BI__builtin_neon_vrshrn_n_v:
    Int = Intrinsic::aarch64_neon_vrshrn;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vrshrn_n");
  case NEON::BI__builtin_neon_vqrshrun_n_v:
    Int = Intrinsic::aarch64_neon_vsqrshrun;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vqrshrun_n");
  case NEON::BI__builtin_neon_vqshrn_n_v:
    Int = usgn ? Intrinsic::aarch64_neon_vuqshrn
               : Intrinsic::aarch64_neon_vsqshrn;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vqshrn_n");
  case NEON::BI__builtin_neon_vqrshrn_n_v:
    Int = usgn ? Intrinsic::aarch64_neon_vuqrshrn
               : Intrinsic::aarch64_neon_vsqrshrn;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vqrshrn_n");

  // Convert
  case NEON::BI__builtin_neon_vcvt_n_f64_v:
  case NEON::BI__builtin_neon_vcvtq_n_f64_v: {
    llvm::Type *FloatTy =
        GetNeonType(this, NeonTypeFlags(NeonTypeFlags::Float64, false, quad));
    llvm::Type *Tys[2] = { FloatTy, Ty };
    Int = usgn ? Intrinsic::arm_neon_vcvtfxu2fp
               : Intrinsic::arm_neon_vcvtfxs2fp;
    Function *F = CGM.getIntrinsic(Int, Tys);
    return EmitNeonCall(F, Ops, "vcvt_n");
  }

  // Load/Store
  case NEON::BI__builtin_neon_vld1_x2_v:
  case NEON::BI__builtin_neon_vld1q_x2_v:
  case NEON::BI__builtin_neon_vld1_x3_v:
  case NEON::BI__builtin_neon_vld1q_x3_v:
  case NEON::BI__builtin_neon_vld1_x4_v:
  case NEON::BI__builtin_neon_vld1q_x4_v: {
    unsigned Int;
    switch (BuiltinID) {
    case NEON::BI__builtin_neon_vld1_x2_v:
    case NEON::BI__builtin_neon_vld1q_x2_v:
      Int = Intrinsic::aarch64_neon_vld1x2;
      break;
    case NEON::BI__builtin_neon_vld1_x3_v:
    case NEON::BI__builtin_neon_vld1q_x3_v:
      Int = Intrinsic::aarch64_neon_vld1x3;
      break;
    case NEON::BI__builtin_neon_vld1_x4_v:
    case NEON::BI__builtin_neon_vld1q_x4_v:
      Int = Intrinsic::aarch64_neon_vld1x4;
      break;
    }
    Function *F = CGM.getIntrinsic(Int, Ty);
    Ops[1] = Builder.CreateCall2(F, Ops[1], Align, "vld1xN");
    Ty = llvm::PointerType::getUnqual(Ops[1]->getType());
    Ops[0] = Builder.CreateBitCast(Ops[0], Ty);
    return Builder.CreateStore(Ops[1], Ops[0]);
  }
  case NEON::BI__builtin_neon_vst1_x2_v:
  case NEON::BI__builtin_neon_vst1q_x2_v:
  case NEON::BI__builtin_neon_vst1_x3_v:
  case NEON::BI__builtin_neon_vst1q_x3_v:
  case NEON::BI__builtin_neon_vst1_x4_v:
  case NEON::BI__builtin_neon_vst1q_x4_v: {
    Ops.push_back(Align);
    unsigned Int;
    switch (BuiltinID) {
    case NEON::BI__builtin_neon_vst1_x2_v:
    case NEON::BI__builtin_neon_vst1q_x2_v:
      Int = Intrinsic::aarch64_neon_vst1x2;
      break;
    case NEON::BI__builtin_neon_vst1_x3_v:
    case NEON::BI__builtin_neon_vst1q_x3_v:
      Int = Intrinsic::aarch64_neon_vst1x3;
      break;
    case NEON::BI__builtin_neon_vst1_x4_v:
    case NEON::BI__builtin_neon_vst1q_x4_v:
      Int = Intrinsic::aarch64_neon_vst1x4;
      break;
    }
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "");
  }
  case NEON::BI__builtin_neon_vld1_lane_v:
  case NEON::BI__builtin_neon_vld1q_lane_v: {
    Ops[1] = Builder.CreateBitCast(Ops[1], Ty);
    Ty = llvm::PointerType::getUnqual(VTy->getElementType());
    Ops[0] = Builder.CreateBitCast(Ops[0], Ty);
    LoadInst *Ld = Builder.CreateLoad(Ops[0]);
    Ld->setAlignment(cast<ConstantInt>(Align)->getZExtValue());
    return Builder.CreateInsertElement(Ops[1], Ld, Ops[2], "vld1_lane");
  }
  case NEON::BI__builtin_neon_vst1_lane_v:
  case NEON::BI__builtin_neon_vst1q_lane_v: {
    Ops[1] = Builder.CreateBitCast(Ops[1], Ty);
    Ops[1] = Builder.CreateExtractElement(Ops[1], Ops[2]);
    Ty = llvm::PointerType::getUnqual(Ops[1]->getType());
    StoreInst *St =
        Builder.CreateStore(Ops[1], Builder.CreateBitCast(Ops[0], Ty));
    St->setAlignment(cast<ConstantInt>(Align)->getZExtValue());
    return St;
  }
  case NEON::BI__builtin_neon_vld2_dup_v:
  case NEON::BI__builtin_neon_vld2q_dup_v:
  case NEON::BI__builtin_neon_vld3_dup_v:
  case NEON::BI__builtin_neon_vld3q_dup_v:
  case NEON::BI__builtin_neon_vld4_dup_v:
  case NEON::BI__builtin_neon_vld4q_dup_v: {
    // Handle 64-bit x 1 elements as a special-case.  There is no "dup" needed.
    if (VTy->getElementType()->getPrimitiveSizeInBits() == 64 &&
        VTy->getNumElements() == 1) {
      switch (BuiltinID) {
      case NEON::BI__builtin_neon_vld2_dup_v:
        Int = Intrinsic::arm_neon_vld2;
        break;
      case NEON::BI__builtin_neon_vld3_dup_v:
        Int = Intrinsic::arm_neon_vld3;
        break;
      case NEON::BI__builtin_neon_vld4_dup_v:
        Int = Intrinsic::arm_neon_vld4;
        break;
      default:
        llvm_unreachable("unknown vld_dup intrinsic?");
      }
      Function *F = CGM.getIntrinsic(Int, Ty);
      Ops[1] = Builder.CreateCall2(F, Ops[1], Align, "vld_dup");
      Ty = llvm::PointerType::getUnqual(Ops[1]->getType());
      Ops[0] = Builder.CreateBitCast(Ops[0], Ty);
      return Builder.CreateStore(Ops[1], Ops[0]);
    }
    switch (BuiltinID) {
    case NEON::BI__builtin_neon_vld2_dup_v:
    case NEON::BI__builtin_neon_vld2q_dup_v:
      Int = Intrinsic::arm_neon_vld2lane;
      break;
    case NEON::BI__builtin_neon_vld3_dup_v:
    case NEON::BI__builtin_neon_vld3q_dup_v:
      Int = Intrinsic::arm_neon_vld3lane;
      break;
    case NEON::BI__builtin_neon_vld4_dup_v:
    case NEON::BI__builtin_neon_vld4q_dup_v:
      Int = Intrinsic::arm_neon_vld4lane;
      break;
    }
    Function *F = CGM.getIntrinsic(Int, Ty);
    llvm::StructType *STy = cast<llvm::StructType>(F->getReturnType());

    SmallVector<Value *, 6> Args;
    Args.push_back(Ops[1]);
    Args.append(STy->getNumElements(), UndefValue::get(Ty));

    llvm::Constant *CI = ConstantInt::get(Int32Ty, 0);
    Args.push_back(CI);
    Args.push_back(Align);

    Ops[1] = Builder.CreateCall(F, Args, "vld_dup");
    // splat lane 0 to all elts in each vector of the result.
    for (unsigned i = 0, e = STy->getNumElements(); i != e; ++i) {
      Value *Val = Builder.CreateExtractValue(Ops[1], i);
      Value *Elt = Builder.CreateBitCast(Val, Ty);
      Elt = EmitNeonSplat(Elt, CI);
      Elt = Builder.CreateBitCast(Elt, Val->getType());
      Ops[1] = Builder.CreateInsertValue(Ops[1], Elt, i);
    }
    Ty = llvm::PointerType::getUnqual(Ops[1]->getType());
    Ops[0] = Builder.CreateBitCast(Ops[0], Ty);
    return Builder.CreateStore(Ops[1], Ops[0]);
  }

  case NEON::BI__builtin_neon_vmul_lane_v:
  case NEON::BI__builtin_neon_vmul_laneq_v: {
    // v1f64 vmul_lane should be mapped to Neon scalar mul lane
    bool Quad = false;
    if (BuiltinID == NEON::BI__builtin_neon_vmul_laneq_v)
      Quad = true;
    Ops[0] = Builder.CreateBitCast(Ops[0], DoubleTy);
    llvm::Type *VTy = GetNeonType(this,
      NeonTypeFlags(NeonTypeFlags::Float64, false, Quad));
    Ops[1] = Builder.CreateBitCast(Ops[1], VTy);
    Ops[1] = Builder.CreateExtractElement(Ops[1], Ops[2], "extract");
    Value *Result = Builder.CreateFMul(Ops[0], Ops[1]);
    return Builder.CreateBitCast(Result, Ty);
  }

  // AArch64-only builtins
  case NEON::BI__builtin_neon_vfmaq_laneq_v: {
    Value *F = CGM.getIntrinsic(Intrinsic::fma, Ty);
    Ops[0] = Builder.CreateBitCast(Ops[0], Ty);
    Ops[1] = Builder.CreateBitCast(Ops[1], Ty);

    Ops[2] = Builder.CreateBitCast(Ops[2], Ty);
    Ops[2] = EmitNeonSplat(Ops[2], cast<ConstantInt>(Ops[3]));
    return Builder.CreateCall3(F, Ops[2], Ops[1], Ops[0]);
  }
  case NEON::BI__builtin_neon_vfmaq_lane_v: {
    Value *F = CGM.getIntrinsic(Intrinsic::fma, Ty);
    Ops[0] = Builder.CreateBitCast(Ops[0], Ty);
    Ops[1] = Builder.CreateBitCast(Ops[1], Ty);

    llvm::VectorType *VTy = cast<llvm::VectorType>(Ty);
    llvm::Type *STy = llvm::VectorType::get(VTy->getElementType(),
                                            VTy->getNumElements() / 2);
    Ops[2] = Builder.CreateBitCast(Ops[2], STy);
    Value* SV = llvm::ConstantVector::getSplat(VTy->getNumElements(),
                                               cast<ConstantInt>(Ops[3]));
    Ops[2] = Builder.CreateShuffleVector(Ops[2], Ops[2], SV, "lane");

    return Builder.CreateCall3(F, Ops[2], Ops[1], Ops[0]);
  }
  case NEON::BI__builtin_neon_vfma_lane_v: {
    llvm::VectorType *VTy = cast<llvm::VectorType>(Ty);
    // v1f64 fma should be mapped to Neon scalar f64 fma
    if (VTy && VTy->getElementType() == DoubleTy) {
      Ops[0] = Builder.CreateBitCast(Ops[0], DoubleTy);
      Ops[1] = Builder.CreateBitCast(Ops[1], DoubleTy);
      llvm::Type *VTy = GetNeonType(this,
        NeonTypeFlags(NeonTypeFlags::Float64, false, false));
      Ops[2] = Builder.CreateBitCast(Ops[2], VTy);
      Ops[2] = Builder.CreateExtractElement(Ops[2], Ops[3], "extract");
      Value *F = CGM.getIntrinsic(Intrinsic::fma, DoubleTy);
      Value *Result = Builder.CreateCall3(F, Ops[1], Ops[2], Ops[0]);
      return Builder.CreateBitCast(Result, Ty);
    }
    Value *F = CGM.getIntrinsic(Intrinsic::fma, Ty);
    Ops[0] = Builder.CreateBitCast(Ops[0], Ty);
    Ops[1] = Builder.CreateBitCast(Ops[1], Ty);

    Ops[2] = Builder.CreateBitCast(Ops[2], Ty);
    Ops[2] = EmitNeonSplat(Ops[2], cast<ConstantInt>(Ops[3]));
    return Builder.CreateCall3(F, Ops[2], Ops[1], Ops[0]);
  }
  case NEON::BI__builtin_neon_vfma_laneq_v: {
    llvm::VectorType *VTy = cast<llvm::VectorType>(Ty);
    // v1f64 fma should be mapped to Neon scalar f64 fma
    if (VTy && VTy->getElementType() == DoubleTy) {
      Ops[0] = Builder.CreateBitCast(Ops[0], DoubleTy);
      Ops[1] = Builder.CreateBitCast(Ops[1], DoubleTy);
      llvm::Type *VTy = GetNeonType(this,
        NeonTypeFlags(NeonTypeFlags::Float64, false, true));
      Ops[2] = Builder.CreateBitCast(Ops[2], VTy);
      Ops[2] = Builder.CreateExtractElement(Ops[2], Ops[3], "extract");
      Value *F = CGM.getIntrinsic(Intrinsic::fma, DoubleTy);
      Value *Result = Builder.CreateCall3(F, Ops[1], Ops[2], Ops[0]);
      return Builder.CreateBitCast(Result, Ty);
    }
    Value *F = CGM.getIntrinsic(Intrinsic::fma, Ty);
    Ops[0] = Builder.CreateBitCast(Ops[0], Ty);
    Ops[1] = Builder.CreateBitCast(Ops[1], Ty);

    llvm::Type *STy = llvm::VectorType::get(VTy->getElementType(),
                                            VTy->getNumElements() * 2);
    Ops[2] = Builder.CreateBitCast(Ops[2], STy);
    Value* SV = llvm::ConstantVector::getSplat(VTy->getNumElements(),
                                               cast<ConstantInt>(Ops[3]));
    Ops[2] = Builder.CreateShuffleVector(Ops[2], Ops[2], SV, "lane");

    return Builder.CreateCall3(F, Ops[2], Ops[1], Ops[0]);
  }
  case NEON::BI__builtin_neon_vfms_v:
  case NEON::BI__builtin_neon_vfmsq_v: {
    Value *F = CGM.getIntrinsic(Intrinsic::fma, Ty);
    Ops[0] = Builder.CreateBitCast(Ops[0], Ty);
    Ops[1] = Builder.CreateBitCast(Ops[1], Ty);
    Ops[1] = Builder.CreateFNeg(Ops[1]);
    Ops[2] = Builder.CreateBitCast(Ops[2], Ty);

    // LLVM's fma intrinsic puts the accumulator in the last position, but the
    // AArch64 intrinsic has it first.
    return Builder.CreateCall3(F, Ops[1], Ops[2], Ops[0]);
  }
  case NEON::BI__builtin_neon_vmaxnm_v:
  case NEON::BI__builtin_neon_vmaxnmq_v: {
    Int = Intrinsic::aarch64_neon_vmaxnm;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vmaxnm");
  }
  case NEON::BI__builtin_neon_vminnm_v:
  case NEON::BI__builtin_neon_vminnmq_v: {
    Int = Intrinsic::aarch64_neon_vminnm;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vminnm");
  }
  case NEON::BI__builtin_neon_vpmaxnm_v:
  case NEON::BI__builtin_neon_vpmaxnmq_v: {
    Int = Intrinsic::aarch64_neon_vpmaxnm;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vpmaxnm");
  }
  case NEON::BI__builtin_neon_vpminnm_v:
  case NEON::BI__builtin_neon_vpminnmq_v: {
    Int = Intrinsic::aarch64_neon_vpminnm;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vpminnm");
  }
  case NEON::BI__builtin_neon_vpmaxq_v: {
    Int = usgn ? Intrinsic::arm_neon_vpmaxu : Intrinsic::arm_neon_vpmaxs;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vpmax");
  }
  case NEON::BI__builtin_neon_vpminq_v: {
    Int = usgn ? Intrinsic::arm_neon_vpminu : Intrinsic::arm_neon_vpmins;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vpmin");
  }
  case NEON::BI__builtin_neon_vmulx_v:
  case NEON::BI__builtin_neon_vmulxq_v: {
    Int = Intrinsic::aarch64_neon_vmulx;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vmulx");
  }
  case NEON::BI__builtin_neon_vsqadd_v:
  case NEON::BI__builtin_neon_vsqaddq_v: {
    Int = Intrinsic::aarch64_neon_usqadd;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vsqadd");
  }
  case NEON::BI__builtin_neon_vuqadd_v:
  case NEON::BI__builtin_neon_vuqaddq_v: {
    Int = Intrinsic::aarch64_neon_suqadd;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vuqadd");
  }
  case NEON::BI__builtin_neon_vrbit_v:
  case NEON::BI__builtin_neon_vrbitq_v:
    Int = Intrinsic::aarch64_neon_rbit;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vrbit");
  case NEON::BI__builtin_neon_vcvt_f32_f64: {
    NeonTypeFlags SrcFlag = NeonTypeFlags(NeonTypeFlags::Float64, false, true);
    Ops[0] = Builder.CreateBitCast(Ops[0], GetNeonType(this, SrcFlag));
    return Builder.CreateFPTrunc(Ops[0], Ty, "vcvt");
  }
  case NEON::BI__builtin_neon_vcvtx_f32_v: {
    llvm::Type *EltTy = FloatTy;
    llvm::Type *ResTy = llvm::VectorType::get(EltTy, 2);
    llvm::Type *Tys[2] = { ResTy, Ty };
    Int = Intrinsic::aarch64_neon_vcvtxn;
    return EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "vcvtx_f32_f64");
  }
  case NEON::BI__builtin_neon_vcvt_f64_f32: {
    llvm::Type *OpTy =
        GetNeonType(this, NeonTypeFlags(NeonTypeFlags::Float32, false, false));
    Ops[0] = Builder.CreateBitCast(Ops[0], OpTy);
    return Builder.CreateFPExt(Ops[0], Ty, "vcvt");
  }
  case NEON::BI__builtin_neon_vcvt_f64_v:
  case NEON::BI__builtin_neon_vcvtq_f64_v: {
    Ops[0] = Builder.CreateBitCast(Ops[0], Ty);
    Ty = GetNeonType(this, NeonTypeFlags(NeonTypeFlags::Float64, false, quad));
    return usgn ? Builder.CreateUIToFP(Ops[0], Ty, "vcvt")
                : Builder.CreateSIToFP(Ops[0], Ty, "vcvt");
  }
  case NEON::BI__builtin_neon_vrndn_v:
  case NEON::BI__builtin_neon_vrndnq_v: {
    Int = Intrinsic::aarch64_neon_frintn;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vrndn");
  }
  case NEON::BI__builtin_neon_vrnda_v:
  case NEON::BI__builtin_neon_vrndaq_v: {
    Int = Intrinsic::round;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vrnda");
  }
  case NEON::BI__builtin_neon_vrndp_v:
  case NEON::BI__builtin_neon_vrndpq_v: {
    Int = Intrinsic::ceil;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vrndp");
  }
  case NEON::BI__builtin_neon_vrndm_v:
  case NEON::BI__builtin_neon_vrndmq_v: {
    Int = Intrinsic::floor;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vrndm");
  }
  case NEON::BI__builtin_neon_vrndx_v:
  case NEON::BI__builtin_neon_vrndxq_v: {
    Int = Intrinsic::rint;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vrndx");
  }
  case NEON::BI__builtin_neon_vrnd_v:
  case NEON::BI__builtin_neon_vrndq_v: {
    Int = Intrinsic::trunc;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vrnd");
  }
  case NEON::BI__builtin_neon_vrndi_v:
  case NEON::BI__builtin_neon_vrndiq_v: {
    Int = Intrinsic::nearbyint;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vrndi");
  }
  case NEON::BI__builtin_neon_vsqrt_v:
  case NEON::BI__builtin_neon_vsqrtq_v: {
    Int = Intrinsic::sqrt;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vsqrt");
  }
  case NEON::BI__builtin_neon_vceqz_v:
  case NEON::BI__builtin_neon_vceqzq_v:
    return EmitAArch64CompareBuiltinExpr(Ops[0], Ty, ICmpInst::FCMP_OEQ,
                                         ICmpInst::ICMP_EQ, "vceqz");
  case NEON::BI__builtin_neon_vcgez_v:
  case NEON::BI__builtin_neon_vcgezq_v:
    return EmitAArch64CompareBuiltinExpr(Ops[0], Ty, ICmpInst::FCMP_OGE,
                                         ICmpInst::ICMP_SGE, "vcgez");
  case NEON::BI__builtin_neon_vclez_v:
  case NEON::BI__builtin_neon_vclezq_v:
    return EmitAArch64CompareBuiltinExpr(Ops[0], Ty, ICmpInst::FCMP_OLE,
                                         ICmpInst::ICMP_SLE, "vclez");
  case NEON::BI__builtin_neon_vcgtz_v:
  case NEON::BI__builtin_neon_vcgtzq_v:
    return EmitAArch64CompareBuiltinExpr(Ops[0], Ty, ICmpInst::FCMP_OGT,
                                         ICmpInst::ICMP_SGT, "vcgtz");
  case NEON::BI__builtin_neon_vcltz_v:
  case NEON::BI__builtin_neon_vcltzq_v:
    return EmitAArch64CompareBuiltinExpr(Ops[0], Ty, ICmpInst::FCMP_OLT,
                                         ICmpInst::ICMP_SLT, "vcltz");
  }
}

Value *CodeGenFunction::EmitARMBuiltinExpr(unsigned BuiltinID,
                                           const CallExpr *E) {
  if (BuiltinID == ARM::BI__clear_cache) {
    assert(E->getNumArgs() == 2 && "__clear_cache takes 2 arguments");
    const FunctionDecl *FD = E->getDirectCallee();
    SmallVector<Value*, 2> Ops;
    for (unsigned i = 0; i < 2; i++)
      Ops.push_back(EmitScalarExpr(E->getArg(i)));
    llvm::Type *Ty = CGM.getTypes().ConvertType(FD->getType());
    llvm::FunctionType *FTy = cast<llvm::FunctionType>(Ty);
    StringRef Name = FD->getName();
    return EmitNounwindRuntimeCall(CGM.CreateRuntimeFunction(FTy, Name), Ops);
  }

  if (BuiltinID == ARM::BI__builtin_arm_ldrexd ||
      (BuiltinID == ARM::BI__builtin_arm_ldrex &&
       getContext().getTypeSize(E->getType()) == 64)) {
    Function *F = CGM.getIntrinsic(Intrinsic::arm_ldrexd);

    Value *LdPtr = EmitScalarExpr(E->getArg(0));
    Value *Val = Builder.CreateCall(F, Builder.CreateBitCast(LdPtr, Int8PtrTy),
                                    "ldrexd");

    Value *Val0 = Builder.CreateExtractValue(Val, 1);
    Value *Val1 = Builder.CreateExtractValue(Val, 0);
    Val0 = Builder.CreateZExt(Val0, Int64Ty);
    Val1 = Builder.CreateZExt(Val1, Int64Ty);

    Value *ShiftCst = llvm::ConstantInt::get(Int64Ty, 32);
    Val = Builder.CreateShl(Val0, ShiftCst, "shl", true /* nuw */);
    Val = Builder.CreateOr(Val, Val1);
    return Builder.CreateBitCast(Val, ConvertType(E->getType()));
  }

  if (BuiltinID == ARM::BI__builtin_arm_ldrex) {
    Value *LoadAddr = EmitScalarExpr(E->getArg(0));

    QualType Ty = E->getType();
    llvm::Type *RealResTy = ConvertType(Ty);
    llvm::Type *IntResTy = llvm::IntegerType::get(getLLVMContext(),
                                                  getContext().getTypeSize(Ty));
    LoadAddr = Builder.CreateBitCast(LoadAddr, IntResTy->getPointerTo());

    Function *F = CGM.getIntrinsic(Intrinsic::arm_ldrex, LoadAddr->getType());
    Value *Val = Builder.CreateCall(F, LoadAddr, "ldrex");

    if (RealResTy->isPointerTy())
      return Builder.CreateIntToPtr(Val, RealResTy);
    else {
      Val = Builder.CreateTruncOrBitCast(Val, IntResTy);
      return Builder.CreateBitCast(Val, RealResTy);
    }
  }

  if (BuiltinID == ARM::BI__builtin_arm_strexd ||
      (BuiltinID == ARM::BI__builtin_arm_strex &&
       getContext().getTypeSize(E->getArg(0)->getType()) == 64)) {
    Function *F = CGM.getIntrinsic(Intrinsic::arm_strexd);
    llvm::Type *STy = llvm::StructType::get(Int32Ty, Int32Ty, NULL);

    Value *Tmp = CreateMemTemp(E->getArg(0)->getType());
    Value *Val = EmitScalarExpr(E->getArg(0));
    Builder.CreateStore(Val, Tmp);

    Value *LdPtr = Builder.CreateBitCast(Tmp,llvm::PointerType::getUnqual(STy));
    Val = Builder.CreateLoad(LdPtr);

    Value *Arg0 = Builder.CreateExtractValue(Val, 0);
    Value *Arg1 = Builder.CreateExtractValue(Val, 1);
    Value *StPtr = Builder.CreateBitCast(EmitScalarExpr(E->getArg(1)), Int8PtrTy);
    return Builder.CreateCall3(F, Arg0, Arg1, StPtr, "strexd");
  }

  if (BuiltinID == ARM::BI__builtin_arm_strex) {
    Value *StoreVal = EmitScalarExpr(E->getArg(0));
    Value *StoreAddr = EmitScalarExpr(E->getArg(1));

    QualType Ty = E->getArg(0)->getType();
    llvm::Type *StoreTy = llvm::IntegerType::get(getLLVMContext(),
                                                 getContext().getTypeSize(Ty));
    StoreAddr = Builder.CreateBitCast(StoreAddr, StoreTy->getPointerTo());

    if (StoreVal->getType()->isPointerTy())
      StoreVal = Builder.CreatePtrToInt(StoreVal, Int32Ty);
    else {
      StoreVal = Builder.CreateBitCast(StoreVal, StoreTy);
      StoreVal = Builder.CreateZExtOrBitCast(StoreVal, Int32Ty);
    }

    Function *F = CGM.getIntrinsic(Intrinsic::arm_strex, StoreAddr->getType());
    return Builder.CreateCall2(F, StoreVal, StoreAddr, "strex");
  }

  if (BuiltinID == ARM::BI__builtin_arm_clrex) {
    Function *F = CGM.getIntrinsic(Intrinsic::arm_clrex);
    return Builder.CreateCall(F);
  }

  if (BuiltinID == ARM::BI__builtin_arm_sevl) {
    Function *F = CGM.getIntrinsic(Intrinsic::arm_sevl);
    return Builder.CreateCall(F);
  }

  // CRC32
  Intrinsic::ID CRCIntrinsicID = Intrinsic::not_intrinsic;
  switch (BuiltinID) {
  case ARM::BI__builtin_arm_crc32b:
    CRCIntrinsicID = Intrinsic::arm_crc32b; break;
  case ARM::BI__builtin_arm_crc32cb:
    CRCIntrinsicID = Intrinsic::arm_crc32cb; break;
  case ARM::BI__builtin_arm_crc32h:
    CRCIntrinsicID = Intrinsic::arm_crc32h; break;
  case ARM::BI__builtin_arm_crc32ch:
    CRCIntrinsicID = Intrinsic::arm_crc32ch; break;
  case ARM::BI__builtin_arm_crc32w:
  case ARM::BI__builtin_arm_crc32d:
    CRCIntrinsicID = Intrinsic::arm_crc32w; break;
  case ARM::BI__builtin_arm_crc32cw:
  case ARM::BI__builtin_arm_crc32cd:
    CRCIntrinsicID = Intrinsic::arm_crc32cw; break;
  }

  if (CRCIntrinsicID != Intrinsic::not_intrinsic) {
    Value *Arg0 = EmitScalarExpr(E->getArg(0));
    Value *Arg1 = EmitScalarExpr(E->getArg(1));

    // crc32{c,}d intrinsics are implemnted as two calls to crc32{c,}w
    // intrinsics, hence we need different codegen for these cases.
    if (BuiltinID == ARM::BI__builtin_arm_crc32d ||
        BuiltinID == ARM::BI__builtin_arm_crc32cd) {
      Value *C1 = llvm::ConstantInt::get(Int64Ty, 32);
      Value *Arg1a = Builder.CreateTruncOrBitCast(Arg1, Int32Ty);
      Value *Arg1b = Builder.CreateLShr(Arg1, C1);
      Arg1b = Builder.CreateTruncOrBitCast(Arg1b, Int32Ty);

      Function *F = CGM.getIntrinsic(CRCIntrinsicID);
      Value *Res = Builder.CreateCall2(F, Arg0, Arg1a);
      return Builder.CreateCall2(F, Res, Arg1b);
    } else {
      Arg1 = Builder.CreateZExtOrBitCast(Arg1, Int32Ty);

      Function *F = CGM.getIntrinsic(CRCIntrinsicID);
      return Builder.CreateCall2(F, Arg0, Arg1);
    }
  }

  SmallVector<Value*, 4> Ops;
  llvm::Value *Align = 0;
  for (unsigned i = 0, e = E->getNumArgs() - 1; i != e; i++) {
    if (i == 0) {
      switch (BuiltinID) {
      case NEON::BI__builtin_neon_vld1_v:
      case NEON::BI__builtin_neon_vld1q_v:
      case NEON::BI__builtin_neon_vld1q_lane_v:
      case NEON::BI__builtin_neon_vld1_lane_v:
      case NEON::BI__builtin_neon_vld1_dup_v:
      case NEON::BI__builtin_neon_vld1q_dup_v:
      case NEON::BI__builtin_neon_vst1_v:
      case NEON::BI__builtin_neon_vst1q_v:
      case NEON::BI__builtin_neon_vst1q_lane_v:
      case NEON::BI__builtin_neon_vst1_lane_v:
      case NEON::BI__builtin_neon_vst2_v:
      case NEON::BI__builtin_neon_vst2q_v:
      case NEON::BI__builtin_neon_vst2_lane_v:
      case NEON::BI__builtin_neon_vst2q_lane_v:
      case NEON::BI__builtin_neon_vst3_v:
      case NEON::BI__builtin_neon_vst3q_v:
      case NEON::BI__builtin_neon_vst3_lane_v:
      case NEON::BI__builtin_neon_vst3q_lane_v:
      case NEON::BI__builtin_neon_vst4_v:
      case NEON::BI__builtin_neon_vst4q_v:
      case NEON::BI__builtin_neon_vst4_lane_v:
      case NEON::BI__builtin_neon_vst4q_lane_v:
        // Get the alignment for the argument in addition to the value;
        // we'll use it later.
        std::pair<llvm::Value*, unsigned> Src =
            EmitPointerWithAlignment(E->getArg(0));
        Ops.push_back(Src.first);
        Align = Builder.getInt32(Src.second);
        continue;
      }
    }
    if (i == 1) {
      switch (BuiltinID) {
      case NEON::BI__builtin_neon_vld2_v:
      case NEON::BI__builtin_neon_vld2q_v:
      case NEON::BI__builtin_neon_vld3_v:
      case NEON::BI__builtin_neon_vld3q_v:
      case NEON::BI__builtin_neon_vld4_v:
      case NEON::BI__builtin_neon_vld4q_v:
      case NEON::BI__builtin_neon_vld2_lane_v:
      case NEON::BI__builtin_neon_vld2q_lane_v:
      case NEON::BI__builtin_neon_vld3_lane_v:
      case NEON::BI__builtin_neon_vld3q_lane_v:
      case NEON::BI__builtin_neon_vld4_lane_v:
      case NEON::BI__builtin_neon_vld4q_lane_v:
      case NEON::BI__builtin_neon_vld2_dup_v:
      case NEON::BI__builtin_neon_vld3_dup_v:
      case NEON::BI__builtin_neon_vld4_dup_v:
        // Get the alignment for the argument in addition to the value;
        // we'll use it later.
        std::pair<llvm::Value*, unsigned> Src =
            EmitPointerWithAlignment(E->getArg(1));
        Ops.push_back(Src.first);
        Align = Builder.getInt32(Src.second);
        continue;
      }
    }
    Ops.push_back(EmitScalarExpr(E->getArg(i)));
  }

  switch (BuiltinID) {
  default: break;
  // vget_lane and vset_lane are not overloaded and do not have an extra
  // argument that specifies the vector type.
  case NEON::BI__builtin_neon_vget_lane_i8:
  case NEON::BI__builtin_neon_vget_lane_i16:
  case NEON::BI__builtin_neon_vget_lane_i32:
  case NEON::BI__builtin_neon_vget_lane_i64:
  case NEON::BI__builtin_neon_vget_lane_f32:
  case NEON::BI__builtin_neon_vgetq_lane_i8:
  case NEON::BI__builtin_neon_vgetq_lane_i16:
  case NEON::BI__builtin_neon_vgetq_lane_i32:
  case NEON::BI__builtin_neon_vgetq_lane_i64:
  case NEON::BI__builtin_neon_vgetq_lane_f32:
    return Builder.CreateExtractElement(Ops[0], EmitScalarExpr(E->getArg(1)),
                                        "vget_lane");
  case NEON::BI__builtin_neon_vset_lane_i8:
  case NEON::BI__builtin_neon_vset_lane_i16:
  case NEON::BI__builtin_neon_vset_lane_i32:
  case NEON::BI__builtin_neon_vset_lane_i64:
  case NEON::BI__builtin_neon_vset_lane_f32:
  case NEON::BI__builtin_neon_vsetq_lane_i8:
  case NEON::BI__builtin_neon_vsetq_lane_i16:
  case NEON::BI__builtin_neon_vsetq_lane_i32:
  case NEON::BI__builtin_neon_vsetq_lane_i64:
  case NEON::BI__builtin_neon_vsetq_lane_f32:
    Ops.push_back(EmitScalarExpr(E->getArg(2)));
    return Builder.CreateInsertElement(Ops[1], Ops[0], Ops[2], "vset_lane");

  // Non-polymorphic crypto instructions also not overloaded
  case NEON::BI__builtin_neon_vsha1h_u32:
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_sha1h), Ops,
                        "vsha1h");
  case NEON::BI__builtin_neon_vsha1cq_u32:
    Ops.push_back(EmitScalarExpr(E->getArg(2)));
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_sha1c), Ops,
                        "vsha1h");
  case NEON::BI__builtin_neon_vsha1pq_u32:
    Ops.push_back(EmitScalarExpr(E->getArg(2)));
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_sha1p), Ops,
                        "vsha1h");
  case NEON::BI__builtin_neon_vsha1mq_u32:
    Ops.push_back(EmitScalarExpr(E->getArg(2)));
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_sha1m), Ops,
                        "vsha1h");
  }

  // Get the last argument, which specifies the vector type.
  llvm::APSInt Result;
  const Expr *Arg = E->getArg(E->getNumArgs()-1);
  if (!Arg->isIntegerConstantExpr(Result, getContext()))
    return 0;

  if (BuiltinID == ARM::BI__builtin_arm_vcvtr_f ||
      BuiltinID == ARM::BI__builtin_arm_vcvtr_d) {
    // Determine the overloaded type of this builtin.
    llvm::Type *Ty;
    if (BuiltinID == ARM::BI__builtin_arm_vcvtr_f)
      Ty = FloatTy;
    else
      Ty = DoubleTy;

    // Determine whether this is an unsigned conversion or not.
    bool usgn = Result.getZExtValue() == 1;
    unsigned Int = usgn ? Intrinsic::arm_vcvtru : Intrinsic::arm_vcvtr;

    // Call the appropriate intrinsic.
    Function *F = CGM.getIntrinsic(Int, Ty);
    return Builder.CreateCall(F, Ops, "vcvtr");
  }

  // Determine the type of this overloaded NEON intrinsic.
  NeonTypeFlags Type(Result.getZExtValue());
  bool usgn = Type.isUnsigned();
  bool rightShift = false;

  llvm::VectorType *VTy = GetNeonType(this, Type);
  llvm::Type *Ty = VTy;
  if (!Ty)
    return 0;

  // Many NEON builtins have identical semantics and uses in ARM and
  // AArch64. Emit these in a single function.
  llvm::ArrayRef<NeonIntrinsicInfo> IntrinsicMap(ARMSIMDIntrinsicMap);
  const NeonIntrinsicInfo *Builtin = findNeonIntrinsicInMap(
      IntrinsicMap, BuiltinID, NEONSIMDIntrinsicsProvenSorted);
  if (Builtin)
    return EmitCommonNeonBuiltinExpr(
        Builtin->BuiltinID, Builtin->LLVMIntrinsic, Builtin->AltLLVMIntrinsic,
        Builtin->NameHint, Builtin->TypeModifier, E, Ops, Align);

  unsigned Int;
  switch (BuiltinID) {
  default: return 0;
  case NEON::BI__builtin_neon_vld1q_lane_v:
    // Handle 64-bit integer elements as a special case.  Use shuffles of
    // one-element vectors to avoid poor code for i64 in the backend.
    if (VTy->getElementType()->isIntegerTy(64)) {
      // Extract the other lane.
      Ops[1] = Builder.CreateBitCast(Ops[1], Ty);
      int Lane = cast<ConstantInt>(Ops[2])->getZExtValue();
      Value *SV = llvm::ConstantVector::get(ConstantInt::get(Int32Ty, 1-Lane));
      Ops[1] = Builder.CreateShuffleVector(Ops[1], Ops[1], SV);
      // Load the value as a one-element vector.
      Ty = llvm::VectorType::get(VTy->getElementType(), 1);
      Function *F = CGM.getIntrinsic(Intrinsic::arm_neon_vld1, Ty);
      Value *Ld = Builder.CreateCall2(F, Ops[0], Align);
      // Combine them.
      SmallVector<Constant*, 2> Indices;
      Indices.push_back(ConstantInt::get(Int32Ty, 1-Lane));
      Indices.push_back(ConstantInt::get(Int32Ty, Lane));
      SV = llvm::ConstantVector::get(Indices);
      return Builder.CreateShuffleVector(Ops[1], Ld, SV, "vld1q_lane");
    }
    // fall through
  case NEON::BI__builtin_neon_vld1_lane_v: {
    Ops[1] = Builder.CreateBitCast(Ops[1], Ty);
    Ty = llvm::PointerType::getUnqual(VTy->getElementType());
    Ops[0] = Builder.CreateBitCast(Ops[0], Ty);
    LoadInst *Ld = Builder.CreateLoad(Ops[0]);
    Ld->setAlignment(cast<ConstantInt>(Align)->getZExtValue());
    return Builder.CreateInsertElement(Ops[1], Ld, Ops[2], "vld1_lane");
  }
  case NEON::BI__builtin_neon_vld2_dup_v:
  case NEON::BI__builtin_neon_vld3_dup_v:
  case NEON::BI__builtin_neon_vld4_dup_v: {
    // Handle 64-bit elements as a special-case.  There is no "dup" needed.
    if (VTy->getElementType()->getPrimitiveSizeInBits() == 64) {
      switch (BuiltinID) {
      case NEON::BI__builtin_neon_vld2_dup_v:
        Int = Intrinsic::arm_neon_vld2;
        break;
      case NEON::BI__builtin_neon_vld3_dup_v:
        Int = Intrinsic::arm_neon_vld3;
        break;
      case NEON::BI__builtin_neon_vld4_dup_v:
        Int = Intrinsic::arm_neon_vld4;
        break;
      default: llvm_unreachable("unknown vld_dup intrinsic?");
      }
      Function *F = CGM.getIntrinsic(Int, Ty);
      Ops[1] = Builder.CreateCall2(F, Ops[1], Align, "vld_dup");
      Ty = llvm::PointerType::getUnqual(Ops[1]->getType());
      Ops[0] = Builder.CreateBitCast(Ops[0], Ty);
      return Builder.CreateStore(Ops[1], Ops[0]);
    }
    switch (BuiltinID) {
    case NEON::BI__builtin_neon_vld2_dup_v:
      Int = Intrinsic::arm_neon_vld2lane;
      break;
    case NEON::BI__builtin_neon_vld3_dup_v:
      Int = Intrinsic::arm_neon_vld3lane;
      break;
    case NEON::BI__builtin_neon_vld4_dup_v:
      Int = Intrinsic::arm_neon_vld4lane;
      break;
    default: llvm_unreachable("unknown vld_dup intrinsic?");
    }
    Function *F = CGM.getIntrinsic(Int, Ty);
    llvm::StructType *STy = cast<llvm::StructType>(F->getReturnType());

    SmallVector<Value*, 6> Args;
    Args.push_back(Ops[1]);
    Args.append(STy->getNumElements(), UndefValue::get(Ty));

    llvm::Constant *CI = ConstantInt::get(Int32Ty, 0);
    Args.push_back(CI);
    Args.push_back(Align);

    Ops[1] = Builder.CreateCall(F, Args, "vld_dup");
    // splat lane 0 to all elts in each vector of the result.
    for (unsigned i = 0, e = STy->getNumElements(); i != e; ++i) {
      Value *Val = Builder.CreateExtractValue(Ops[1], i);
      Value *Elt = Builder.CreateBitCast(Val, Ty);
      Elt = EmitNeonSplat(Elt, CI);
      Elt = Builder.CreateBitCast(Elt, Val->getType());
      Ops[1] = Builder.CreateInsertValue(Ops[1], Elt, i);
    }
    Ty = llvm::PointerType::getUnqual(Ops[1]->getType());
    Ops[0] = Builder.CreateBitCast(Ops[0], Ty);
    return Builder.CreateStore(Ops[1], Ops[0]);
  }
  case NEON::BI__builtin_neon_vqrshrn_n_v:
    Int =
      usgn ? Intrinsic::arm_neon_vqrshiftnu : Intrinsic::arm_neon_vqrshiftns;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vqrshrn_n",
                        1, true);
  case NEON::BI__builtin_neon_vqrshrun_n_v:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vqrshiftnsu, Ty),
                        Ops, "vqrshrun_n", 1, true);
  case NEON::BI__builtin_neon_vqshlu_n_v:
  case NEON::BI__builtin_neon_vqshluq_n_v:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vqshiftsu, Ty),
                        Ops, "vqshlu", 1, false);
  case NEON::BI__builtin_neon_vqshrn_n_v:
    Int = usgn ? Intrinsic::arm_neon_vqshiftnu : Intrinsic::arm_neon_vqshiftns;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vqshrn_n",
                        1, true);
  case NEON::BI__builtin_neon_vqshrun_n_v:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vqshiftnsu, Ty),
                        Ops, "vqshrun_n", 1, true);
  case NEON::BI__builtin_neon_vrecpe_v:
  case NEON::BI__builtin_neon_vrecpeq_v:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vrecpe, Ty),
                        Ops, "vrecpe");
  case NEON::BI__builtin_neon_vrshrn_n_v:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vrshiftn, Ty),
                        Ops, "vrshrn_n", 1, true);
  case NEON::BI__builtin_neon_vrshr_n_v:
  case NEON::BI__builtin_neon_vrshrq_n_v:
    Int = usgn ? Intrinsic::arm_neon_vrshiftu : Intrinsic::arm_neon_vrshifts;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vrshr_n", 1, true);
  case NEON::BI__builtin_neon_vrsra_n_v:
  case NEON::BI__builtin_neon_vrsraq_n_v:
    Ops[0] = Builder.CreateBitCast(Ops[0], Ty);
    Ops[1] = Builder.CreateBitCast(Ops[1], Ty);
    Ops[2] = EmitNeonShiftVector(Ops[2], Ty, true);
    Int = usgn ? Intrinsic::arm_neon_vrshiftu : Intrinsic::arm_neon_vrshifts;
    Ops[1] = Builder.CreateCall2(CGM.getIntrinsic(Int, Ty), Ops[1], Ops[2]);
    return Builder.CreateAdd(Ops[0], Ops[1], "vrsra_n");
  case NEON::BI__builtin_neon_vsri_n_v:
  case NEON::BI__builtin_neon_vsriq_n_v:
    rightShift = true;
  case NEON::BI__builtin_neon_vsli_n_v:
  case NEON::BI__builtin_neon_vsliq_n_v:
    Ops[2] = EmitNeonShiftVector(Ops[2], Ty, rightShift);
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vshiftins, Ty),
                        Ops, "vsli_n");
  case NEON::BI__builtin_neon_vsra_n_v:
  case NEON::BI__builtin_neon_vsraq_n_v:
    Ops[0] = Builder.CreateBitCast(Ops[0], Ty);
    Ops[1] = EmitNeonRShiftImm(Ops[1], Ops[2], Ty, usgn, "vsra_n");
    return Builder.CreateAdd(Ops[0], Ops[1]);
  case NEON::BI__builtin_neon_vst1q_lane_v:
    // Handle 64-bit integer elements as a special case.  Use a shuffle to get
    // a one-element vector and avoid poor code for i64 in the backend.
    if (VTy->getElementType()->isIntegerTy(64)) {
      Ops[1] = Builder.CreateBitCast(Ops[1], Ty);
      Value *SV = llvm::ConstantVector::get(cast<llvm::Constant>(Ops[2]));
      Ops[1] = Builder.CreateShuffleVector(Ops[1], Ops[1], SV);
      Ops[2] = Align;
      return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_neon_vst1,
                                                 Ops[1]->getType()), Ops);
    }
    // fall through
  case NEON::BI__builtin_neon_vst1_lane_v: {
    Ops[1] = Builder.CreateBitCast(Ops[1], Ty);
    Ops[1] = Builder.CreateExtractElement(Ops[1], Ops[2]);
    Ty = llvm::PointerType::getUnqual(Ops[1]->getType());
    StoreInst *St = Builder.CreateStore(Ops[1],
                                        Builder.CreateBitCast(Ops[0], Ty));
    St->setAlignment(cast<ConstantInt>(Align)->getZExtValue());
    return St;
  }
  case NEON::BI__builtin_neon_vtbl1_v:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vtbl1),
                        Ops, "vtbl1");
  case NEON::BI__builtin_neon_vtbl2_v:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vtbl2),
                        Ops, "vtbl2");
  case NEON::BI__builtin_neon_vtbl3_v:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vtbl3),
                        Ops, "vtbl3");
  case NEON::BI__builtin_neon_vtbl4_v:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vtbl4),
                        Ops, "vtbl4");
  case NEON::BI__builtin_neon_vtbx1_v:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vtbx1),
                        Ops, "vtbx1");
  case NEON::BI__builtin_neon_vtbx2_v:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vtbx2),
                        Ops, "vtbx2");
  case NEON::BI__builtin_neon_vtbx3_v:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vtbx3),
                        Ops, "vtbx3");
  case NEON::BI__builtin_neon_vtbx4_v:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vtbx4),
                        Ops, "vtbx4");
  }
}

llvm::Value *CodeGenFunction::
BuildVector(ArrayRef<llvm::Value*> Ops) {
  assert((Ops.size() & (Ops.size() - 1)) == 0 &&
         "Not a power-of-two sized vector!");
  bool AllConstants = true;
  for (unsigned i = 0, e = Ops.size(); i != e && AllConstants; ++i)
    AllConstants &= isa<Constant>(Ops[i]);

  // If this is a constant vector, create a ConstantVector.
  if (AllConstants) {
    SmallVector<llvm::Constant*, 16> CstOps;
    for (unsigned i = 0, e = Ops.size(); i != e; ++i)
      CstOps.push_back(cast<Constant>(Ops[i]));
    return llvm::ConstantVector::get(CstOps);
  }

  // Otherwise, insertelement the values to build the vector.
  Value *Result =
    llvm::UndefValue::get(llvm::VectorType::get(Ops[0]->getType(), Ops.size()));

  for (unsigned i = 0, e = Ops.size(); i != e; ++i)
    Result = Builder.CreateInsertElement(Result, Ops[i], Builder.getInt32(i));

  return Result;
}

Value *CodeGenFunction::EmitX86BuiltinExpr(unsigned BuiltinID,
                                           const CallExpr *E) {
  SmallVector<Value*, 4> Ops;

  // Find out if any arguments are required to be integer constant expressions.
  unsigned ICEArguments = 0;
  ASTContext::GetBuiltinTypeError Error;
  getContext().GetBuiltinType(BuiltinID, Error, &ICEArguments);
  assert(Error == ASTContext::GE_None && "Should not codegen an error");

  for (unsigned i = 0, e = E->getNumArgs(); i != e; i++) {
    // If this is a normal argument, just emit it as a scalar.
    if ((ICEArguments & (1 << i)) == 0) {
      Ops.push_back(EmitScalarExpr(E->getArg(i)));
      continue;
    }

    // If this is required to be a constant, constant fold it so that we know
    // that the generated intrinsic gets a ConstantInt.
    llvm::APSInt Result;
    bool IsConst = E->getArg(i)->isIntegerConstantExpr(Result, getContext());
    assert(IsConst && "Constant arg isn't actually constant?"); (void)IsConst;
    Ops.push_back(llvm::ConstantInt::get(getLLVMContext(), Result));
  }

  switch (BuiltinID) {
  default: return 0;
  case X86::BI_mm_prefetch: {
    Value *Address = EmitScalarExpr(E->getArg(0));
    Value *RW = ConstantInt::get(Int32Ty, 0);
    Value *Locality = EmitScalarExpr(E->getArg(1));
    Value *Data = ConstantInt::get(Int32Ty, 1);
    Value *F = CGM.getIntrinsic(Intrinsic::prefetch);
    return Builder.CreateCall4(F, Address, RW, Locality, Data);
  }
  case X86::BI__builtin_ia32_vec_init_v8qi:
  case X86::BI__builtin_ia32_vec_init_v4hi:
  case X86::BI__builtin_ia32_vec_init_v2si:
    return Builder.CreateBitCast(BuildVector(Ops),
                                 llvm::Type::getX86_MMXTy(getLLVMContext()));
  case X86::BI__builtin_ia32_vec_ext_v2si:
    return Builder.CreateExtractElement(Ops[0],
                                  llvm::ConstantInt::get(Ops[1]->getType(), 0));
  case X86::BI__builtin_ia32_ldmxcsr: {
    Value *Tmp = CreateMemTemp(E->getArg(0)->getType());
    Builder.CreateStore(Ops[0], Tmp);
    return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::x86_sse_ldmxcsr),
                              Builder.CreateBitCast(Tmp, Int8PtrTy));
  }
  case X86::BI__builtin_ia32_stmxcsr: {
    Value *Tmp = CreateMemTemp(E->getType());
    Builder.CreateCall(CGM.getIntrinsic(Intrinsic::x86_sse_stmxcsr),
                       Builder.CreateBitCast(Tmp, Int8PtrTy));
    return Builder.CreateLoad(Tmp, "stmxcsr");
  }
  case X86::BI__builtin_ia32_storehps:
  case X86::BI__builtin_ia32_storelps: {
    llvm::Type *PtrTy = llvm::PointerType::getUnqual(Int64Ty);
    llvm::Type *VecTy = llvm::VectorType::get(Int64Ty, 2);

    // cast val v2i64
    Ops[1] = Builder.CreateBitCast(Ops[1], VecTy, "cast");

    // extract (0, 1)
    unsigned Index = BuiltinID == X86::BI__builtin_ia32_storelps ? 0 : 1;
    llvm::Value *Idx = llvm::ConstantInt::get(Int32Ty, Index);
    Ops[1] = Builder.CreateExtractElement(Ops[1], Idx, "extract");

    // cast pointer to i64 & store
    Ops[0] = Builder.CreateBitCast(Ops[0], PtrTy);
    return Builder.CreateStore(Ops[1], Ops[0]);
  }
  case X86::BI__builtin_ia32_palignr: {
    unsigned shiftVal = cast<llvm::ConstantInt>(Ops[2])->getZExtValue();

    // If palignr is shifting the pair of input vectors less than 9 bytes,
    // emit a shuffle instruction.
    if (shiftVal <= 8) {
      SmallVector<llvm::Constant*, 8> Indices;
      for (unsigned i = 0; i != 8; ++i)
        Indices.push_back(llvm::ConstantInt::get(Int32Ty, shiftVal + i));

      Value* SV = llvm::ConstantVector::get(Indices);
      return Builder.CreateShuffleVector(Ops[1], Ops[0], SV, "palignr");
    }

    // If palignr is shifting the pair of input vectors more than 8 but less
    // than 16 bytes, emit a logical right shift of the destination.
    if (shiftVal < 16) {
      // MMX has these as 1 x i64 vectors for some odd optimization reasons.
      llvm::Type *VecTy = llvm::VectorType::get(Int64Ty, 1);

      Ops[0] = Builder.CreateBitCast(Ops[0], VecTy, "cast");
      Ops[1] = llvm::ConstantInt::get(VecTy, (shiftVal-8) * 8);

      // create i32 constant
      llvm::Function *F = CGM.getIntrinsic(Intrinsic::x86_mmx_psrl_q);
      return Builder.CreateCall(F, makeArrayRef(&Ops[0], 2), "palignr");
    }

    // If palignr is shifting the pair of vectors more than 16 bytes, emit zero.
    return llvm::Constant::getNullValue(ConvertType(E->getType()));
  }
  case X86::BI__builtin_ia32_palignr128: {
    unsigned shiftVal = cast<llvm::ConstantInt>(Ops[2])->getZExtValue();

    // If palignr is shifting the pair of input vectors less than 17 bytes,
    // emit a shuffle instruction.
    if (shiftVal <= 16) {
      SmallVector<llvm::Constant*, 16> Indices;
      for (unsigned i = 0; i != 16; ++i)
        Indices.push_back(llvm::ConstantInt::get(Int32Ty, shiftVal + i));

      Value* SV = llvm::ConstantVector::get(Indices);
      return Builder.CreateShuffleVector(Ops[1], Ops[0], SV, "palignr");
    }

    // If palignr is shifting the pair of input vectors more than 16 but less
    // than 32 bytes, emit a logical right shift of the destination.
    if (shiftVal < 32) {
      llvm::Type *VecTy = llvm::VectorType::get(Int64Ty, 2);

      Ops[0] = Builder.CreateBitCast(Ops[0], VecTy, "cast");
      Ops[1] = llvm::ConstantInt::get(Int32Ty, (shiftVal-16) * 8);

      // create i32 constant
      llvm::Function *F = CGM.getIntrinsic(Intrinsic::x86_sse2_psrl_dq);
      return Builder.CreateCall(F, makeArrayRef(&Ops[0], 2), "palignr");
    }

    // If palignr is shifting the pair of vectors more than 32 bytes, emit zero.
    return llvm::Constant::getNullValue(ConvertType(E->getType()));
  }
  case X86::BI__builtin_ia32_palignr256: {
    unsigned shiftVal = cast<llvm::ConstantInt>(Ops[2])->getZExtValue();

    // If palignr is shifting the pair of input vectors less than 17 bytes,
    // emit a shuffle instruction.
    if (shiftVal <= 16) {
      SmallVector<llvm::Constant*, 32> Indices;
      // 256-bit palignr operates on 128-bit lanes so we need to handle that
      for (unsigned l = 0; l != 2; ++l) {
        unsigned LaneStart = l * 16;
        unsigned LaneEnd = (l+1) * 16;
        for (unsigned i = 0; i != 16; ++i) {
          unsigned Idx = shiftVal + i + LaneStart;
          if (Idx >= LaneEnd) Idx += 16; // end of lane, switch operand
          Indices.push_back(llvm::ConstantInt::get(Int32Ty, Idx));
        }
      }

      Value* SV = llvm::ConstantVector::get(Indices);
      return Builder.CreateShuffleVector(Ops[1], Ops[0], SV, "palignr");
    }

    // If palignr is shifting the pair of input vectors more than 16 but less
    // than 32 bytes, emit a logical right shift of the destination.
    if (shiftVal < 32) {
      llvm::Type *VecTy = llvm::VectorType::get(Int64Ty, 4);

      Ops[0] = Builder.CreateBitCast(Ops[0], VecTy, "cast");
      Ops[1] = llvm::ConstantInt::get(Int32Ty, (shiftVal-16) * 8);

      // create i32 constant
      llvm::Function *F = CGM.getIntrinsic(Intrinsic::x86_avx2_psrl_dq);
      return Builder.CreateCall(F, makeArrayRef(&Ops[0], 2), "palignr");
    }

    // If palignr is shifting the pair of vectors more than 32 bytes, emit zero.
    return llvm::Constant::getNullValue(ConvertType(E->getType()));
  }
  case X86::BI__builtin_ia32_movntps:
  case X86::BI__builtin_ia32_movntps256:
  case X86::BI__builtin_ia32_movntpd:
  case X86::BI__builtin_ia32_movntpd256:
  case X86::BI__builtin_ia32_movntdq:
  case X86::BI__builtin_ia32_movntdq256:
  case X86::BI__builtin_ia32_movnti:
  case X86::BI__builtin_ia32_movnti64: {
    llvm::MDNode *Node = llvm::MDNode::get(getLLVMContext(),
                                           Builder.getInt32(1));

    // Convert the type of the pointer to a pointer to the stored type.
    Value *BC = Builder.CreateBitCast(Ops[0],
                                llvm::PointerType::getUnqual(Ops[1]->getType()),
                                      "cast");
    StoreInst *SI = Builder.CreateStore(Ops[1], BC);
    SI->setMetadata(CGM.getModule().getMDKindID("nontemporal"), Node);

    // If the operand is an integer, we can't assume alignment. Otherwise,
    // assume natural alignment.
    QualType ArgTy = E->getArg(1)->getType();
    unsigned Align;
    if (ArgTy->isIntegerType())
      Align = 1;
    else
      Align = getContext().getTypeSizeInChars(ArgTy).getQuantity();
    SI->setAlignment(Align);
    return SI;
  }
  // 3DNow!
  case X86::BI__builtin_ia32_pswapdsf:
  case X86::BI__builtin_ia32_pswapdsi: {
    const char *name = 0;
    Intrinsic::ID ID = Intrinsic::not_intrinsic;
    switch(BuiltinID) {
    default: llvm_unreachable("Unsupported intrinsic!");
    case X86::BI__builtin_ia32_pswapdsf:
    case X86::BI__builtin_ia32_pswapdsi:
      name = "pswapd";
      ID = Intrinsic::x86_3dnowa_pswapd;
      break;
    }
    llvm::Type *MMXTy = llvm::Type::getX86_MMXTy(getLLVMContext());
    Ops[0] = Builder.CreateBitCast(Ops[0], MMXTy, "cast");
    llvm::Function *F = CGM.getIntrinsic(ID);
    return Builder.CreateCall(F, Ops, name);
  }
  case X86::BI__builtin_ia32_rdrand16_step:
  case X86::BI__builtin_ia32_rdrand32_step:
  case X86::BI__builtin_ia32_rdrand64_step:
  case X86::BI__builtin_ia32_rdseed16_step:
  case X86::BI__builtin_ia32_rdseed32_step:
  case X86::BI__builtin_ia32_rdseed64_step: {
    Intrinsic::ID ID;
    switch (BuiltinID) {
    default: llvm_unreachable("Unsupported intrinsic!");
    case X86::BI__builtin_ia32_rdrand16_step:
      ID = Intrinsic::x86_rdrand_16;
      break;
    case X86::BI__builtin_ia32_rdrand32_step:
      ID = Intrinsic::x86_rdrand_32;
      break;
    case X86::BI__builtin_ia32_rdrand64_step:
      ID = Intrinsic::x86_rdrand_64;
      break;
    case X86::BI__builtin_ia32_rdseed16_step:
      ID = Intrinsic::x86_rdseed_16;
      break;
    case X86::BI__builtin_ia32_rdseed32_step:
      ID = Intrinsic::x86_rdseed_32;
      break;
    case X86::BI__builtin_ia32_rdseed64_step:
      ID = Intrinsic::x86_rdseed_64;
      break;
    }

    Value *Call = Builder.CreateCall(CGM.getIntrinsic(ID));
    Builder.CreateStore(Builder.CreateExtractValue(Call, 0), Ops[0]);
    return Builder.CreateExtractValue(Call, 1);
  }
  // AVX2 broadcast
  case X86::BI__builtin_ia32_vbroadcastsi256: {
    Value *VecTmp = CreateMemTemp(E->getArg(0)->getType());
    Builder.CreateStore(Ops[0], VecTmp);
    Value *F = CGM.getIntrinsic(Intrinsic::x86_avx2_vbroadcasti128);
    return Builder.CreateCall(F, Builder.CreateBitCast(VecTmp, Int8PtrTy));
  }
  }
}


Value *CodeGenFunction::EmitPPCBuiltinExpr(unsigned BuiltinID,
                                           const CallExpr *E) {
  SmallVector<Value*, 4> Ops;

  for (unsigned i = 0, e = E->getNumArgs(); i != e; i++)
    Ops.push_back(EmitScalarExpr(E->getArg(i)));

  Intrinsic::ID ID = Intrinsic::not_intrinsic;

  switch (BuiltinID) {
  default: return 0;

  // vec_ld, vec_lvsl, vec_lvsr
  case PPC::BI__builtin_altivec_lvx:
  case PPC::BI__builtin_altivec_lvxl:
  case PPC::BI__builtin_altivec_lvebx:
  case PPC::BI__builtin_altivec_lvehx:
  case PPC::BI__builtin_altivec_lvewx:
  case PPC::BI__builtin_altivec_lvsl:
  case PPC::BI__builtin_altivec_lvsr:
  {
    Ops[1] = Builder.CreateBitCast(Ops[1], Int8PtrTy);

    Ops[0] = Builder.CreateGEP(Ops[1], Ops[0]);
    Ops.pop_back();

    switch (BuiltinID) {
    default: llvm_unreachable("Unsupported ld/lvsl/lvsr intrinsic!");
    case PPC::BI__builtin_altivec_lvx:
      ID = Intrinsic::ppc_altivec_lvx;
      break;
    case PPC::BI__builtin_altivec_lvxl:
      ID = Intrinsic::ppc_altivec_lvxl;
      break;
    case PPC::BI__builtin_altivec_lvebx:
      ID = Intrinsic::ppc_altivec_lvebx;
      break;
    case PPC::BI__builtin_altivec_lvehx:
      ID = Intrinsic::ppc_altivec_lvehx;
      break;
    case PPC::BI__builtin_altivec_lvewx:
      ID = Intrinsic::ppc_altivec_lvewx;
      break;
    case PPC::BI__builtin_altivec_lvsl:
      ID = Intrinsic::ppc_altivec_lvsl;
      break;
    case PPC::BI__builtin_altivec_lvsr:
      ID = Intrinsic::ppc_altivec_lvsr;
      break;
    }
    llvm::Function *F = CGM.getIntrinsic(ID);
    return Builder.CreateCall(F, Ops, "");
  }

  // vec_st
  case PPC::BI__builtin_altivec_stvx:
  case PPC::BI__builtin_altivec_stvxl:
  case PPC::BI__builtin_altivec_stvebx:
  case PPC::BI__builtin_altivec_stvehx:
  case PPC::BI__builtin_altivec_stvewx:
  {
    Ops[2] = Builder.CreateBitCast(Ops[2], Int8PtrTy);
    Ops[1] = Builder.CreateGEP(Ops[2], Ops[1]);
    Ops.pop_back();

    switch (BuiltinID) {
    default: llvm_unreachable("Unsupported st intrinsic!");
    case PPC::BI__builtin_altivec_stvx:
      ID = Intrinsic::ppc_altivec_stvx;
      break;
    case PPC::BI__builtin_altivec_stvxl:
      ID = Intrinsic::ppc_altivec_stvxl;
      break;
    case PPC::BI__builtin_altivec_stvebx:
      ID = Intrinsic::ppc_altivec_stvebx;
      break;
    case PPC::BI__builtin_altivec_stvehx:
      ID = Intrinsic::ppc_altivec_stvehx;
      break;
    case PPC::BI__builtin_altivec_stvewx:
      ID = Intrinsic::ppc_altivec_stvewx;
      break;
    }
    llvm::Function *F = CGM.getIntrinsic(ID);
    return Builder.CreateCall(F, Ops, "");
  }
  }
}
