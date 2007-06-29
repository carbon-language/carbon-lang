//===--- CGExpr.cpp - Emit LLVM Code from Expressions ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Expr nodes as LLVM code.
//
//===----------------------------------------------------------------------===//

#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "clang/AST/AST.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
using namespace clang;
using namespace CodeGen;

//===--------------------------------------------------------------------===//
//                        Miscellaneous Helper Methods
//===--------------------------------------------------------------------===//

/// CreateTempAlloca - This creates a alloca and inserts it into the entry
/// block.
llvm::AllocaInst *CodeGenFunction::CreateTempAlloca(const llvm::Type *Ty,
                                                    const char *Name) {
  return new llvm::AllocaInst(Ty, 0, Name, AllocaInsertPt);
}

/// EvaluateExprAsBool - Perform the usual unary conversions on the specified
/// expression and compare the result against zero, returning an Int1Ty value.
llvm::Value *CodeGenFunction::EvaluateExprAsBool(const Expr *E) {
  QualType Ty;
  RValue Val = EmitExprWithUsualUnaryConversions(E, Ty);
  return ConvertScalarValueToBool(Val, Ty);
}

/// EmitLoadOfComplex - Given an RValue reference for a complex, emit code to
/// load the real and imaginary pieces, returning them as Real/Imag.
void CodeGenFunction::EmitLoadOfComplex(RValue V,
                                        llvm::Value *&Real, llvm::Value *&Imag){
  llvm::Value *Ptr = V.getAggregateAddr();
  
  llvm::Constant *Zero = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0);
  llvm::Constant *One  = llvm::ConstantInt::get(llvm::Type::Int32Ty, 1);
  llvm::Value *RealPtr = Builder.CreateGEP(Ptr, Zero, Zero, "realp");
  llvm::Value *ImagPtr = Builder.CreateGEP(Ptr, Zero, One, "imagp");
  
  // FIXME: Handle volatility.
  Real = Builder.CreateLoad(RealPtr, "real");
  Imag = Builder.CreateLoad(ImagPtr, "imag");
}

/// EmitStoreOfComplex - Store the specified real/imag parts into the
/// specified value pointer.
void CodeGenFunction::EmitStoreOfComplex(llvm::Value *Real, llvm::Value *Imag,
                                         llvm::Value *ResPtr) {
  llvm::Constant *Zero = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0);
  llvm::Constant *One  = llvm::ConstantInt::get(llvm::Type::Int32Ty, 1);
  llvm::Value *RealPtr = Builder.CreateGEP(ResPtr, Zero, Zero, "real");
  llvm::Value *ImagPtr = Builder.CreateGEP(ResPtr, Zero, One, "imag");
  
  // FIXME: Handle volatility.
  Builder.CreateStore(Real, RealPtr);
  Builder.CreateStore(Imag, ImagPtr);
}

//===--------------------------------------------------------------------===//
//                               Conversions
//===--------------------------------------------------------------------===//

/// EmitConversion - Convert the value specied by Val, whose type is ValTy, to
/// the type specified by DstTy, following the rules of C99 6.3.
RValue CodeGenFunction::EmitConversion(RValue Val, QualType ValTy,
                                       QualType DstTy) {
  ValTy = ValTy.getCanonicalType();
  DstTy = DstTy.getCanonicalType();
  if (ValTy == DstTy) return Val;

  // Handle conversions to bool first, they are special: comparisons against 0.
  if (const BuiltinType *DestBT = dyn_cast<BuiltinType>(DstTy))
    if (DestBT->getKind() == BuiltinType::Bool)
      return RValue::get(ConvertScalarValueToBool(Val, ValTy));
  
  // Handle pointer conversions next: pointers can only be converted to/from
  // other pointers and integers.
  if (isa<PointerType>(DstTy)) {
    const llvm::Type *DestTy = ConvertType(DstTy);
    
    // The source value may be an integer, or a pointer.
    assert(Val.isScalar() && "Can only convert from integer or pointer");
    if (isa<llvm::PointerType>(Val.getVal()->getType()))
      return RValue::get(Builder.CreateBitCast(Val.getVal(), DestTy, "conv"));
    assert(ValTy->isIntegerType() && "Not ptr->ptr or int->ptr conversion?");
    return RValue::get(Builder.CreatePtrToInt(Val.getVal(), DestTy, "conv"));
  }
  
  if (isa<PointerType>(ValTy)) {
    // Must be an ptr to int cast.
    const llvm::Type *DestTy = ConvertType(DstTy);
    assert(isa<llvm::IntegerType>(DestTy) && "not ptr->int?");
    return RValue::get(Builder.CreateIntToPtr(Val.getVal(), DestTy, "conv"));
  }
  
  // Finally, we have the arithmetic types: real int/float and complex
  // int/float.  Handle real->real conversions first, they are the most
  // common.
  if (Val.isScalar() && DstTy->isRealType()) {
    // We know that these are representable as scalars in LLVM, convert to LLVM
    // types since they are easier to reason about.
    llvm::Value *SrcVal = Val.getVal();
    const llvm::Type *DestTy = ConvertType(DstTy);
    if (SrcVal->getType() == DestTy) return Val;
    
    llvm::Value *Result;
    if (isa<llvm::IntegerType>(SrcVal->getType())) {
      bool InputSigned = ValTy->isSignedIntegerType();
      if (isa<llvm::IntegerType>(DestTy))
        Result = Builder.CreateIntCast(SrcVal, DestTy, InputSigned, "conv");
      else if (InputSigned)
        Result = Builder.CreateSIToFP(SrcVal, DestTy, "conv");
      else
        Result = Builder.CreateUIToFP(SrcVal, DestTy, "conv");
    } else {
      assert(SrcVal->getType()->isFloatingPoint() && "Unknown real conversion");
      if (isa<llvm::IntegerType>(DestTy)) {
        if (DstTy->isSignedIntegerType())
          Result = Builder.CreateFPToSI(SrcVal, DestTy, "conv");
        else
          Result = Builder.CreateFPToUI(SrcVal, DestTy, "conv");
      } else {
        assert(DestTy->isFloatingPoint() && "Unknown real conversion");
        if (DestTy->getTypeID() < SrcVal->getType()->getTypeID())
          Result = Builder.CreateFPTrunc(SrcVal, DestTy, "conv");
        else
          Result = Builder.CreateFPExt(SrcVal, DestTy, "conv");
      }
    }
    return RValue::get(Result);
  }
  
  assert(0 && "FIXME: We don't support complex conversions yet!");
}


/// ConvertScalarValueToBool - Convert the specified expression value to a
/// boolean (i1) truth value.  This is equivalent to "Val == 0".
llvm::Value *CodeGenFunction::ConvertScalarValueToBool(RValue Val, QualType Ty){
  Ty = Ty.getCanonicalType();
  llvm::Value *Result;
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(Ty)) {
    switch (BT->getKind()) {
    default: assert(0 && "Unknown scalar value");
    case BuiltinType::Bool:
      Result = Val.getVal();
      // Bool is already evaluated right.
      assert(Result->getType() == llvm::Type::Int1Ty &&
             "Unexpected bool value type!");
      return Result;
    case BuiltinType::Char_S:
    case BuiltinType::Char_U:
    case BuiltinType::SChar:
    case BuiltinType::UChar:
    case BuiltinType::Short:
    case BuiltinType::UShort:
    case BuiltinType::Int:
    case BuiltinType::UInt:
    case BuiltinType::Long:
    case BuiltinType::ULong:
    case BuiltinType::LongLong:
    case BuiltinType::ULongLong:
      // Code below handles simple integers.
      break;
    case BuiltinType::Float:
    case BuiltinType::Double:
    case BuiltinType::LongDouble: {
      // Compare against 0.0 for fp scalars.
      Result = Val.getVal();
      llvm::Value *Zero = llvm::Constant::getNullValue(Result->getType());
      // FIXME: llvm-gcc produces a une comparison: validate this is right.
      Result = Builder.CreateFCmpUNE(Result, Zero, "tobool");
      return Result;
    }
    }
  } else if (isa<PointerType>(Ty) || 
             cast<TagType>(Ty)->getDecl()->getKind() == Decl::Enum) {
    // Code below handles this fine.
  } else {
    assert(isa<ComplexType>(Ty) && "Unknwon type!");
    assert(0 && "FIXME: comparisons against complex not implemented yet");
  }
  
  // Usual case for integers, pointers, and enums: compare against zero.
  Result = Val.getVal();
  
  // Because of the type rules of C, we often end up computing a logical value,
  // then zero extending it to int, then wanting it as a logical value again.
  // Optimize this common case.
  if (llvm::ZExtInst *ZI = dyn_cast<llvm::ZExtInst>(Result)) {
    if (ZI->getOperand(0)->getType() == llvm::Type::Int1Ty) {
      Result = ZI->getOperand(0);
      ZI->eraseFromParent();
      return Result;
    }
  }
  
  llvm::Value *Zero = llvm::Constant::getNullValue(Result->getType());
  return Builder.CreateICmpNE(Result, Zero, "tobool");
}

//===----------------------------------------------------------------------===//
//                         LValue Expression Emission
//===----------------------------------------------------------------------===//

/// EmitLValue - Emit code to compute a designator that specifies the location
/// of the expression.
///
/// This can return one of two things: a simple address or a bitfield
/// reference.  In either case, the LLVM Value* in the LValue structure is
/// guaranteed to be an LLVM pointer type.
///
/// If this returns a bitfield reference, nothing about the pointee type of
/// the LLVM value is known: For example, it may not be a pointer to an
/// integer.
///
/// If this returns a normal address, and if the lvalue's C type is fixed
/// size, this method guarantees that the returned pointer type will point to
/// an LLVM type of the same size of the lvalue's type.  If the lvalue has a
/// variable length type, this is not possible.
///
LValue CodeGenFunction::EmitLValue(const Expr *E) {
  switch (E->getStmtClass()) {
  default:
    fprintf(stderr, "Unimplemented lvalue expr!\n");
    E->dump();
    return LValue::getAddr(llvm::UndefValue::get(
                              llvm::PointerType::get(llvm::Type::Int32Ty)));

  case Expr::DeclRefExprClass: return EmitDeclRefLValue(cast<DeclRefExpr>(E));
  case Expr::ParenExprClass:return EmitLValue(cast<ParenExpr>(E)->getSubExpr());
  case Expr::StringLiteralClass:
    return EmitStringLiteralLValue(cast<StringLiteral>(E));
    
  case Expr::UnaryOperatorClass: 
    return EmitUnaryOpLValue(cast<UnaryOperator>(E));
  case Expr::ArraySubscriptExprClass:
    return EmitArraySubscriptExpr(cast<ArraySubscriptExpr>(E));
  }
}

/// EmitLoadOfLValue - Given an expression that represents a value lvalue,
/// this method emits the address of the lvalue, then loads the result as an
/// rvalue, returning the rvalue.
RValue CodeGenFunction::EmitLoadOfLValue(LValue LV, QualType ExprType) {
  ExprType = ExprType.getCanonicalType();
  
  // FIXME: this is silly and obviously wrong for non-scalars.
  assert(!LV.isBitfield());
  llvm::Value *Ptr = LV.getAddress();
  const llvm::Type *EltTy =
    cast<llvm::PointerType>(Ptr->getType())->getElementType();
  
  // Simple scalar l-value.
  if (EltTy->isFirstClassType())
    return RValue::get(Builder.CreateLoad(Ptr, "tmp"));
  
  // Otherwise, we have an aggregate lvalue.
  return RValue::getAggregate(Ptr);
}

RValue CodeGenFunction::EmitLoadOfLValue(const Expr *E) {
  return EmitLoadOfLValue(EmitLValue(E), E->getType());
}


/// EmitStoreThroughLValue - Store the specified rvalue into the specified
/// lvalue, where both are guaranteed to the have the same type, and that type
/// is 'Ty'.
void CodeGenFunction::EmitStoreThroughLValue(RValue Src, LValue Dst, 
                                             QualType Ty) {
  assert(!Dst.isBitfield() && "FIXME: Don't support store to bitfield yet");
  
  llvm::Value *DstAddr = Dst.getAddress();
  if (Src.isScalar()) {
    // FIXME: Handle volatility etc.
    const llvm::Type *SrcTy = Src.getVal()->getType();
    const llvm::Type *AddrTy = 
      cast<llvm::PointerType>(DstAddr->getType())->getElementType();
    
    if (AddrTy != SrcTy)
      DstAddr = Builder.CreateBitCast(DstAddr, llvm::PointerType::get(SrcTy),
                                      "storetmp");
    Builder.CreateStore(Src.getVal(), DstAddr);
    return;
  }
  
  // Don't use memcpy for complex numbers.
  if (Ty->isComplexType()) {
    llvm::Value *Real, *Imag;
    EmitLoadOfComplex(Src, Real, Imag);
    EmitStoreOfComplex(Real, Imag, Dst.getAddress());
    return;
  }
  
  // Aggregate assignment turns into llvm.memcpy.
  const llvm::Type *SBP = llvm::PointerType::get(llvm::Type::Int8Ty);
  llvm::Value *SrcAddr = Src.getAggregateAddr();
  
  if (DstAddr->getType() != SBP)
    DstAddr = Builder.CreateBitCast(DstAddr, SBP, "tmp");
  if (SrcAddr->getType() != SBP)
    SrcAddr = Builder.CreateBitCast(SrcAddr, SBP, "tmp");

  unsigned Align = 1;   // FIXME: Compute type alignments.
  unsigned Size = 1234; // FIXME: Compute type sizes.
  
  // FIXME: Handle variable sized types.
  const llvm::Type *IntPtr = llvm::IntegerType::get(LLVMPointerWidth);
  llvm::Value *SizeVal = llvm::ConstantInt::get(IntPtr, Size);
  
  llvm::Value *MemCpyOps[4] = {
    DstAddr, SrcAddr, SizeVal,llvm::ConstantInt::get(llvm::Type::Int32Ty, Align)
  };
  
  Builder.CreateCall(CGM.getMemCpyFn(), MemCpyOps, 4);
}


LValue CodeGenFunction::EmitDeclRefLValue(const DeclRefExpr *E) {
  const Decl *D = E->getDecl();
  if (isa<BlockVarDecl>(D) || isa<ParmVarDecl>(D)) {
    llvm::Value *V = LocalDeclMap[D];
    assert(V && "BlockVarDecl not entered in LocalDeclMap?");
    return LValue::getAddr(V);
  } else if (isa<FunctionDecl>(D) || isa<FileVarDecl>(D)) {
    return LValue::getAddr(CGM.GetAddrOfGlobalDecl(D));
  }
  assert(0 && "Unimp declref");
}

LValue CodeGenFunction::EmitUnaryOpLValue(const UnaryOperator *E) {
  // __extension__ doesn't affect lvalue-ness.
  if (E->getOpcode() == UnaryOperator::Extension)
    return EmitLValue(E->getSubExpr());
  
  assert(E->getOpcode() == UnaryOperator::Deref &&
         "'*' is the only unary operator that produces an lvalue");
  return LValue::getAddr(EmitExpr(E->getSubExpr()).getVal());
}

LValue CodeGenFunction::EmitStringLiteralLValue(const StringLiteral *E) {
  assert(!E->isWide() && "FIXME: Wide strings not supported yet!");
  const char *StrData = E->getStrData();
  unsigned Len = E->getByteLength();
  
  // FIXME: Can cache/reuse these within the module.
  llvm::Constant *C=llvm::ConstantArray::get(std::string(StrData, StrData+Len));
  
  // Create a global variable for this.
  C = new llvm::GlobalVariable(C->getType(), true, 
                               llvm::GlobalValue::InternalLinkage,
                               C, ".str", CurFn->getParent());
  llvm::Constant *Zero = llvm::Constant::getNullValue(llvm::Type::Int32Ty);
  llvm::Constant *Zeros[] = { Zero, Zero };
  C = llvm::ConstantExpr::getGetElementPtr(C, Zeros, 2);
  return LValue::getAddr(C);
}

LValue CodeGenFunction::EmitArraySubscriptExpr(const ArraySubscriptExpr *E) {
  // The base and index must be pointers or integers, neither of which are
  // aggregates.  Emit them.
  QualType BaseTy;
  llvm::Value *Base =
    EmitExprWithUsualUnaryConversions(E->getBase(), BaseTy).getVal();
  QualType IdxTy;
  llvm::Value *Idx = 
    EmitExprWithUsualUnaryConversions(E->getIdx(), IdxTy).getVal();
  
  // Usually the base is the pointer type, but sometimes it is the index.
  // Canonicalize to have the pointer as the base.
  if (isa<llvm::PointerType>(Idx->getType())) {
    std::swap(Base, Idx);
    std::swap(BaseTy, IdxTy);
  }
  
  // The pointer is now the base.  Extend or truncate the index type to 32 or
  // 64-bits.
  bool IdxSigned = IdxTy->isSignedIntegerType();
  unsigned IdxBitwidth = cast<llvm::IntegerType>(Idx->getType())->getBitWidth();
  if (IdxBitwidth != LLVMPointerWidth)
    Idx = Builder.CreateIntCast(Idx, llvm::IntegerType::get(LLVMPointerWidth),
                                IdxSigned, "idxprom");

  // We know that the pointer points to a type of the correct size, unless the
  // size is a VLA.
  if (!E->getType()->isConstantSizeType())
    assert(0 && "VLA idx not implemented");
  return LValue::getAddr(Builder.CreateGEP(Base, Idx, "arrayidx"));
}

//===--------------------------------------------------------------------===//
//                             Expression Emission
//===--------------------------------------------------------------------===//

RValue CodeGenFunction::EmitExpr(const Expr *E) {
  assert(E && "Null expression?");
  
  switch (E->getStmtClass()) {
  default:
    fprintf(stderr, "Unimplemented expr!\n");
    E->dump();
    return RValue::get(llvm::UndefValue::get(llvm::Type::Int32Ty));
    
  // l-values.
  case Expr::DeclRefExprClass:
    // DeclRef's of EnumConstantDecl's are simple rvalues.
    if (const EnumConstantDecl *EC = 
          dyn_cast<EnumConstantDecl>(cast<DeclRefExpr>(E)->getDecl()))
      return RValue::get(llvm::ConstantInt::get(EC->getInitVal()));
    
    // FALLTHROUGH
  case Expr::ArraySubscriptExprClass:
    return EmitLoadOfLValue(E);
  case Expr::StringLiteralClass:
    return RValue::get(EmitLValue(E).getAddress());
    
  // Leaf expressions.
  case Expr::IntegerLiteralClass:
    return EmitIntegerLiteral(cast<IntegerLiteral>(E)); 
    
  // Operators.  
  case Expr::ParenExprClass:
    return EmitExpr(cast<ParenExpr>(E)->getSubExpr());
  case Expr::UnaryOperatorClass:
    return EmitUnaryOperator(cast<UnaryOperator>(E));
  case Expr::CastExprClass: 
    return EmitCastExpr(cast<CastExpr>(E));
  case Expr::CallExprClass:
    return EmitCallExpr(cast<CallExpr>(E));
  case Expr::BinaryOperatorClass:
    return EmitBinaryOperator(cast<BinaryOperator>(E));
  }
  
}

RValue CodeGenFunction::EmitIntegerLiteral(const IntegerLiteral *E) {
  return RValue::get(llvm::ConstantInt::get(E->getValue()));
}

RValue CodeGenFunction::EmitCastExpr(const CastExpr *E) {
  QualType SrcTy;
  RValue Src = EmitExprWithUsualUnaryConversions(E->getSubExpr(), SrcTy);
  
  // If the destination is void, just evaluate the source.
  if (E->getType()->isVoidType())
    return RValue::getAggregate(0);
  
  return EmitConversion(Src, SrcTy, E->getType());
}

RValue CodeGenFunction::EmitCallExpr(const CallExpr *E) {
  QualType Ty;
  llvm::Value *Callee =
    EmitExprWithUsualUnaryConversions(E->getCallee(), Ty).getVal();
  
  llvm::SmallVector<llvm::Value*, 16> Args;
  
  // FIXME: Handle struct return.
  for (unsigned i = 0, e = E->getNumArgs(); i != e; ++i) {
    RValue ArgVal = EmitExprWithUsualUnaryConversions(E->getArg(i), Ty);
    
    if (ArgVal.isScalar())
      Args.push_back(ArgVal.getVal());
    else  // Pass by-address.  FIXME: Set attribute bit on call.
      Args.push_back(ArgVal.getAggregateAddr());
  }
  
  llvm::Value *V = Builder.CreateCall(Callee, &Args[0], Args.size());
  if (V->getType() != llvm::Type::VoidTy)
    V->setName("call");
  
  // FIXME: Struct return;
  return RValue::get(V);
}


//===----------------------------------------------------------------------===//
//                           Unary Operator Emission
//===----------------------------------------------------------------------===//

RValue CodeGenFunction::EmitExprWithUsualUnaryConversions(const Expr *E, 
                                                          QualType &ResTy) {
  ResTy = E->getType().getCanonicalType();
  
  if (isa<FunctionType>(ResTy)) { // C99 6.3.2.1p4
    // Functions are promoted to their address.
    ResTy = getContext().getPointerType(ResTy);
    return RValue::get(EmitLValue(E).getAddress());
  } else if (const ArrayType *ary = dyn_cast<ArrayType>(ResTy)) {
    // C99 6.3.2.1p3
    ResTy = getContext().getPointerType(ary->getElementType());
    
    // FIXME: For now we assume that all source arrays map to LLVM arrays.  This
    // will not true when we add support for VLAs.
    llvm::Value *V = EmitLValue(E).getAddress();  // Bitfields can't be arrays.
    
    assert(isa<llvm::PointerType>(V->getType()) &&
           isa<llvm::ArrayType>(cast<llvm::PointerType>(V->getType())
                                ->getElementType()) &&
           "Doesn't support VLAs yet!");
    llvm::Constant *Idx0 = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0);
    return RValue::get(Builder.CreateGEP(V, Idx0, Idx0, "arraydecay"));
  } else if (ResTy->isPromotableIntegerType()) { // C99 6.3.1.1p2
    // FIXME: this probably isn't right, pending clarification from Steve.
    llvm::Value *Val = EmitExpr(E).getVal();
    
    // If the input is a signed integer, sign extend to the destination.
    if (ResTy->isSignedIntegerType()) {
      Val = Builder.CreateSExt(Val, LLVMIntTy, "promote");
    } else {
      // This handles unsigned types, including bool.
      Val = Builder.CreateZExt(Val, LLVMIntTy, "promote");
    }
    ResTy = getContext().IntTy;
    
    return RValue::get(Val);
  }
  
  // Otherwise, this is a float, double, int, struct, etc.
  return EmitExpr(E);
}


RValue CodeGenFunction::EmitUnaryOperator(const UnaryOperator *E) {
  switch (E->getOpcode()) {
  default:
    printf("Unimplemented unary expr!\n");
    E->dump();
    return RValue::get(llvm::UndefValue::get(llvm::Type::Int32Ty));
  // FIXME: pre/post inc/dec
  case UnaryOperator::AddrOf: return EmitUnaryAddrOf(E);
  case UnaryOperator::Deref : return EmitLoadOfLValue(E);
  case UnaryOperator::Plus  : return EmitUnaryPlus(E);
  case UnaryOperator::Minus : return EmitUnaryMinus(E);
  case UnaryOperator::Not   : return EmitUnaryNot(E);
  case UnaryOperator::LNot  : return EmitUnaryLNot(E);
  // FIXME: SIZEOF/ALIGNOF(expr).
  // FIXME: real/imag
  case UnaryOperator::Extension: return EmitExpr(E->getSubExpr());
  }
}

/// C99 6.5.3.2
RValue CodeGenFunction::EmitUnaryAddrOf(const UnaryOperator *E) {
  // The address of the operand is just its lvalue.  It cannot be a bitfield.
  return RValue::get(EmitLValue(E->getSubExpr()).getAddress());
}

RValue CodeGenFunction::EmitUnaryPlus(const UnaryOperator *E) {
  // Unary plus just performs promotions on its arithmetic operand.
  QualType Ty;
  return EmitExprWithUsualUnaryConversions(E->getSubExpr(), Ty);
}

RValue CodeGenFunction::EmitUnaryMinus(const UnaryOperator *E) {
  // Unary minus performs promotions, then negates its arithmetic operand.
  QualType Ty;
  RValue V = EmitExprWithUsualUnaryConversions(E->getSubExpr(), Ty);
  
  if (V.isScalar())
    return RValue::get(Builder.CreateNeg(V.getVal(), "neg"));
  
  assert(0 && "FIXME: This doesn't handle complex operands yet");
}

RValue CodeGenFunction::EmitUnaryNot(const UnaryOperator *E) {
  // Unary not performs promotions, then complements its integer operand.
  QualType Ty;
  RValue V = EmitExprWithUsualUnaryConversions(E->getSubExpr(), Ty);
  
  if (V.isScalar())
    return RValue::get(Builder.CreateNot(V.getVal(), "neg"));
                      
  assert(0 && "FIXME: This doesn't handle integer complex operands yet (GNU)");
}


/// C99 6.5.3.3
RValue CodeGenFunction::EmitUnaryLNot(const UnaryOperator *E) {
  // Compare operand to zero.
  llvm::Value *BoolVal = EvaluateExprAsBool(E->getSubExpr());
  
  // Invert value.
  // TODO: Could dynamically modify easy computations here.  For example, if
  // the operand is an icmp ne, turn into icmp eq.
  BoolVal = Builder.CreateNot(BoolVal, "lnot");
  
  // ZExt result to int.
  return RValue::get(Builder.CreateZExt(BoolVal, LLVMIntTy, "lnot.ext"));
}


//===--------------------------------------------------------------------===//
//                         Binary Operator Emission
//===--------------------------------------------------------------------===//

// FIXME describe.
QualType CodeGenFunction::
EmitUsualArithmeticConversions(const BinaryOperator *E, RValue &LHS, 
                               RValue &RHS) {
  QualType LHSType, RHSType;
  LHS = EmitExprWithUsualUnaryConversions(E->getLHS(), LHSType);
  RHS = EmitExprWithUsualUnaryConversions(E->getRHS(), RHSType);

  // If both operands have the same source type, we're done already.
  if (LHSType == RHSType) return LHSType;

  // If either side is a non-arithmetic type (e.g. a pointer), we are done.
  // The caller can deal with this (e.g. pointer + int).
  if (!LHSType->isArithmeticType() || !RHSType->isArithmeticType())
    return LHSType;

  // At this point, we have two different arithmetic types. 
  
  // Handle complex types first (C99 6.3.1.8p1).
  if (LHSType->isComplexType() || RHSType->isComplexType()) {
    assert(0 && "FIXME: complex types unimp");
#if 0
    // if we have an integer operand, the result is the complex type.
    if (rhs->isIntegerType())
      return lhs;
    if (lhs->isIntegerType())
      return rhs;
    return Context.maxComplexType(lhs, rhs);
#endif
  }
  
  // If neither operand is complex, they must be scalars.
  llvm::Value *LHSV = LHS.getVal();
  llvm::Value *RHSV = RHS.getVal();
  
  // If the LLVM types are already equal, then they only differed in sign, or it
  // was something like char/signed char or double/long double.
  if (LHSV->getType() == RHSV->getType())
    return LHSType;
  
  // Now handle "real" floating types (i.e. float, double, long double).
  if (LHSType->isRealFloatingType() || RHSType->isRealFloatingType()) {
    // if we have an integer operand, the result is the real floating type, and
    // the integer converts to FP.
    if (RHSType->isIntegerType()) {
      // Promote the RHS to an FP type of the LHS, with the sign following the
      // RHS.
      if (RHSType->isSignedIntegerType())
        RHS = RValue::get(Builder.CreateSIToFP(RHSV,LHSV->getType(),"promote"));
      else
        RHS = RValue::get(Builder.CreateUIToFP(RHSV,LHSV->getType(),"promote"));
      return LHSType;
    }
    
    if (LHSType->isIntegerType()) {
      // Promote the LHS to an FP type of the RHS, with the sign following the
      // LHS.
      if (LHSType->isSignedIntegerType())
        LHS = RValue::get(Builder.CreateSIToFP(LHSV,RHSV->getType(),"promote"));
      else
        LHS = RValue::get(Builder.CreateUIToFP(LHSV,RHSV->getType(),"promote"));
      return RHSType;
    }
    
    // Otherwise, they are two FP types.  Promote the smaller operand to the
    // bigger result.
    QualType BiggerType = ASTContext::maxFloatingType(LHSType, RHSType);
    
    if (BiggerType == LHSType)
      RHS = RValue::get(Builder.CreateFPExt(RHSV, LHSV->getType(), "promote"));
    else
      LHS = RValue::get(Builder.CreateFPExt(LHSV, RHSV->getType(), "promote"));
    return BiggerType;
  }
  
  // Finally, we have two integer types that are different according to C.  Do
  // a sign or zero extension if needed.
  
  // Otherwise, one type is smaller than the other.  
  QualType ResTy = ASTContext::maxIntegerType(LHSType, RHSType);
  
  if (LHSType == ResTy) {
    if (RHSType->isSignedIntegerType())
      RHS = RValue::get(Builder.CreateSExt(RHSV, LHSV->getType(), "promote"));
    else
      RHS = RValue::get(Builder.CreateZExt(RHSV, LHSV->getType(), "promote"));
  } else {
    assert(RHSType == ResTy && "Unknown conversion");
    if (LHSType->isSignedIntegerType())
      LHS = RValue::get(Builder.CreateSExt(LHSV, RHSV->getType(), "promote"));
    else
      LHS = RValue::get(Builder.CreateZExt(LHSV, RHSV->getType(), "promote"));
  }  
  return ResTy;
}

/// EmitCompoundAssignmentOperands - Compound assignment operations (like +=)
/// are strange in that the result of the operation is not the same type as the
/// intermediate computation.  This function emits the LHS and RHS operands of
/// the compound assignment, promoting them to their common computation type.
///
/// Since the LHS is an lvalue, and the result is stored back through it, we
/// return the lvalue as well as the LHS/RHS rvalues.  On return, the LHS and
/// RHS values are both in the computation type for the operator.
void CodeGenFunction::
EmitCompoundAssignmentOperands(const CompoundAssignOperator *E,
                               LValue &LHSLV, RValue &LHS, RValue &RHS) {
  LHSLV = EmitLValue(E->getLHS());
  
  // Load the LHS and RHS operands.
  QualType LHSTy = E->getLHS()->getType();
  LHS = EmitLoadOfLValue(LHSLV, LHSTy);
  QualType RHSTy;
  RHS = EmitExprWithUsualUnaryConversions(E->getRHS(), RHSTy);
  
  // Convert the LHS and RHS to the common evaluation type.
  LHS = EmitConversion(LHS, LHSTy, E->getComputationType());
  RHS = EmitConversion(RHS, RHSTy, E->getComputationType());
}

/// EmitCompoundAssignmentResult - Given a result value in the computation type,
/// truncate it down to the actual result type, store it through the LHS lvalue,
/// and return it.
RValue CodeGenFunction::
EmitCompoundAssignmentResult(const CompoundAssignOperator *E,
                             LValue LHSLV, RValue ResV) {
  
  // Truncate back to the destination type.
  if (E->getComputationType() != E->getType())
    ResV = EmitConversion(ResV, E->getComputationType(), E->getType());
  
  // Store the result value into the LHS.
  EmitStoreThroughLValue(ResV, LHSLV, E->getType());
  
  // Return the result.
  return ResV;
}


RValue CodeGenFunction::EmitBinaryOperator(const BinaryOperator *E) {
  RValue LHS, RHS;
  switch (E->getOpcode()) {
  default:
    fprintf(stderr, "Unimplemented expr!\n");
    E->dump();
    return RValue::get(llvm::UndefValue::get(llvm::Type::Int32Ty));
  case BinaryOperator::Mul: return EmitBinaryMul(E);
  case BinaryOperator::Div: return EmitBinaryDiv(E);
  case BinaryOperator::Rem: return EmitBinaryRem(E);
  case BinaryOperator::Add:
    // FIXME: This doesn't handle ptr+int etc yet.
    EmitUsualArithmeticConversions(E, LHS, RHS);
    return EmitAdd(LHS, RHS, E->getType());
  case BinaryOperator::Sub:
    // FIXME: This doesn't handle ptr-int etc yet.
    EmitUsualArithmeticConversions(E, LHS, RHS);
    return EmitSub(LHS, RHS, E->getType());
  case BinaryOperator::Shl: return EmitBinaryShl(E);
  case BinaryOperator::Shr: return EmitBinaryShr(E);
  case BinaryOperator::And: return EmitBinaryAnd(E);
  case BinaryOperator::Xor: return EmitBinaryXor(E);
  case BinaryOperator::Or : return EmitBinaryOr(E);
  case BinaryOperator::LAnd: return EmitBinaryLAnd(E);
  case BinaryOperator::LOr: return EmitBinaryLOr(E);
  case BinaryOperator::LT:
    return EmitBinaryCompare(E, llvm::ICmpInst::ICMP_ULT,
                             llvm::ICmpInst::ICMP_SLT,
                             llvm::FCmpInst::FCMP_OLT);
  case BinaryOperator::GT:
    return EmitBinaryCompare(E, llvm::ICmpInst::ICMP_UGT,
                             llvm::ICmpInst::ICMP_SGT,
                             llvm::FCmpInst::FCMP_OGT);
  case BinaryOperator::LE:
    return EmitBinaryCompare(E, llvm::ICmpInst::ICMP_ULE,
                             llvm::ICmpInst::ICMP_SLE,
                             llvm::FCmpInst::FCMP_OLE);
  case BinaryOperator::GE:
    return EmitBinaryCompare(E, llvm::ICmpInst::ICMP_UGE,
                             llvm::ICmpInst::ICMP_SGE,
                             llvm::FCmpInst::FCMP_OGE);
  case BinaryOperator::EQ:
    return EmitBinaryCompare(E, llvm::ICmpInst::ICMP_EQ,
                             llvm::ICmpInst::ICMP_EQ,
                             llvm::FCmpInst::FCMP_OEQ);
  case BinaryOperator::NE:
    return EmitBinaryCompare(E, llvm::ICmpInst::ICMP_NE,
                             llvm::ICmpInst::ICMP_NE, 
                             llvm::FCmpInst::FCMP_UNE);
  case BinaryOperator::Assign:
    return EmitBinaryAssign(E);
    
  case BinaryOperator::AddAssign: {
    const CompoundAssignOperator *CAO = cast<CompoundAssignOperator>(E);
    LValue LHSLV;
    EmitCompoundAssignmentOperands(CAO, LHSLV, LHS, RHS);
    LHS = EmitAdd(LHS, RHS, CAO->getComputationType());
    return EmitCompoundAssignmentResult(CAO, LHSLV, LHS);
  }
  case BinaryOperator::SubAssign: {
    const CompoundAssignOperator *CAO = cast<CompoundAssignOperator>(E);
    LValue LHSLV;
    EmitCompoundAssignmentOperands(CAO, LHSLV, LHS, RHS);
    LHS = EmitSub(LHS, RHS, CAO->getComputationType());
    return EmitCompoundAssignmentResult(CAO, LHSLV, LHS);
  }
    
    // FIXME: Assignment.
  case BinaryOperator::Comma: return EmitBinaryComma(E);
  }
}

RValue CodeGenFunction::EmitBinaryMul(const BinaryOperator *E) {
  RValue LHS, RHS;
  EmitUsualArithmeticConversions(E, LHS, RHS);
  
  if (LHS.isScalar())
    return RValue::get(Builder.CreateMul(LHS.getVal(), RHS.getVal(), "mul"));
  
  assert(0 && "FIXME: This doesn't handle complex operands yet");
}

RValue CodeGenFunction::EmitBinaryDiv(const BinaryOperator *E) {
  RValue LHS, RHS;
  EmitUsualArithmeticConversions(E, LHS, RHS);
  
  if (LHS.isScalar()) {
    llvm::Value *RV;
    if (LHS.getVal()->getType()->isFloatingPoint())
      RV = Builder.CreateFDiv(LHS.getVal(), RHS.getVal(), "div");
    else if (E->getType()->isUnsignedIntegerType())
      RV = Builder.CreateUDiv(LHS.getVal(), RHS.getVal(), "div");
    else
      RV = Builder.CreateSDiv(LHS.getVal(), RHS.getVal(), "div");
    return RValue::get(RV);
  }
  assert(0 && "FIXME: This doesn't handle complex operands yet");
}

RValue CodeGenFunction::EmitBinaryRem(const BinaryOperator *E) {
  RValue LHS, RHS;
  EmitUsualArithmeticConversions(E, LHS, RHS);
  
  if (LHS.isScalar()) {
    llvm::Value *RV;
    // Rem in C can't be a floating point type: C99 6.5.5p2.
    if (E->getType()->isUnsignedIntegerType())
      RV = Builder.CreateURem(LHS.getVal(), RHS.getVal(), "rem");
    else
      RV = Builder.CreateSRem(LHS.getVal(), RHS.getVal(), "rem");
    return RValue::get(RV);
  }
  
  assert(0 && "FIXME: This doesn't handle complex operands yet");
}

RValue CodeGenFunction::EmitAdd(RValue LHS, RValue RHS, QualType ResTy) {
  if (LHS.isScalar())
    return RValue::get(Builder.CreateAdd(LHS.getVal(), RHS.getVal(), "add"));
  
  // Otherwise, this must be a complex number.
  llvm::Value *LHSR, *LHSI, *RHSR, *RHSI;
  
  EmitLoadOfComplex(LHS, LHSR, LHSI);
  EmitLoadOfComplex(RHS, RHSR, RHSI);
  
  llvm::Value *ResR = Builder.CreateAdd(LHSR, RHSR, "add.r");
  llvm::Value *ResI = Builder.CreateAdd(LHSI, RHSI, "add.i");
  
  llvm::Value *Res = CreateTempAlloca(ConvertType(ResTy));
  EmitStoreOfComplex(ResR, ResI, Res);
  return RValue::getAggregate(Res);
}

RValue CodeGenFunction::EmitSub(RValue LHS, RValue RHS, QualType ResTy) {
  if (LHS.isScalar())
    return RValue::get(Builder.CreateSub(LHS.getVal(), RHS.getVal(), "sub"));
  
  assert(0 && "FIXME: This doesn't handle complex operands yet");
}


RValue CodeGenFunction::EmitBinaryShl(const BinaryOperator *E) {
  // For shifts, integer promotions are performed, but the usual arithmetic 
  // conversions are not.  The LHS and RHS need not have the same type.
  
  QualType ResTy;
  llvm::Value *LHS = 
    EmitExprWithUsualUnaryConversions(E->getLHS(), ResTy).getVal();
  llvm::Value *RHS = 
    EmitExprWithUsualUnaryConversions(E->getRHS(), ResTy).getVal();

  // LLVM requires the LHS and RHS to be the same type, promote or truncate the
  // RHS to the same size as the LHS.
  if (LHS->getType() != RHS->getType())
    RHS = Builder.CreateIntCast(RHS, LHS->getType(), false, "sh_prom");
  
  return RValue::get(Builder.CreateShl(LHS, RHS, "shl"));
}

RValue CodeGenFunction::EmitBinaryShr(const BinaryOperator *E) {
  // For shifts, integer promotions are performed, but the usual arithmetic 
  // conversions are not.  The LHS and RHS need not have the same type.
  
  QualType ResTy;
  llvm::Value *LHS = 
    EmitExprWithUsualUnaryConversions(E->getLHS(), ResTy).getVal();
  llvm::Value *RHS = 
    EmitExprWithUsualUnaryConversions(E->getRHS(), ResTy).getVal();
  
  // LLVM requires the LHS and RHS to be the same type, promote or truncate the
  // RHS to the same size as the LHS.
  if (LHS->getType() != RHS->getType())
    RHS = Builder.CreateIntCast(RHS, LHS->getType(), false, "sh_prom");
  
  if (E->getType()->isUnsignedIntegerType())
    return RValue::get(Builder.CreateLShr(LHS, RHS, "shr"));
  else
    return RValue::get(Builder.CreateAShr(LHS, RHS, "shr"));
}

RValue CodeGenFunction::EmitBinaryCompare(const BinaryOperator *E,
                                          unsigned UICmpOpc, unsigned SICmpOpc,
                                          unsigned FCmpOpc) {
  RValue LHS, RHS;
  EmitUsualArithmeticConversions(E, LHS, RHS);

  llvm::Value *Result;
  if (LHS.isScalar()) {
    if (LHS.getVal()->getType()->isFloatingPoint()) {
      Result = Builder.CreateFCmp((llvm::FCmpInst::Predicate)FCmpOpc,
                                  LHS.getVal(), RHS.getVal(), "cmp");
    } else if (E->getLHS()->getType()->isUnsignedIntegerType()) {
      // FIXME: This check isn't right for "unsigned short < int" where ushort
      // promotes to int and does a signed compare.
      Result = Builder.CreateICmp((llvm::ICmpInst::Predicate)UICmpOpc,
                                  LHS.getVal(), RHS.getVal(), "cmp");
    } else {
      // Signed integers and pointers.
      Result = Builder.CreateICmp((llvm::ICmpInst::Predicate)SICmpOpc,
                                  LHS.getVal(), RHS.getVal(), "cmp");
    }
  } else {
    // Struct/union/complex
    assert(0 && "Aggregate comparisons not implemented yet!");
  }
  
  // ZExt result to int.
  return RValue::get(Builder.CreateZExt(Result, LLVMIntTy, "cmp.ext"));
}

RValue CodeGenFunction::EmitBinaryAnd(const BinaryOperator *E) {
  RValue LHS, RHS;
  EmitUsualArithmeticConversions(E, LHS, RHS);
  
  if (LHS.isScalar())
    return RValue::get(Builder.CreateAnd(LHS.getVal(), RHS.getVal(), "and"));
  
  assert(0 && "FIXME: This doesn't handle complex integer operands yet (GNU)");
}

RValue CodeGenFunction::EmitBinaryXor(const BinaryOperator *E) {
  RValue LHS, RHS;
  EmitUsualArithmeticConversions(E, LHS, RHS);
  
  if (LHS.isScalar())
    return RValue::get(Builder.CreateXor(LHS.getVal(), RHS.getVal(), "xor"));
  
  assert(0 && "FIXME: This doesn't handle complex integer operands yet (GNU)");
}

RValue CodeGenFunction::EmitBinaryOr(const BinaryOperator *E) {
  RValue LHS, RHS;
  EmitUsualArithmeticConversions(E, LHS, RHS);
  
  if (LHS.isScalar())
    return RValue::get(Builder.CreateOr(LHS.getVal(), RHS.getVal(), "or"));
  
  assert(0 && "FIXME: This doesn't handle complex integer operands yet (GNU)");
}

RValue CodeGenFunction::EmitBinaryLAnd(const BinaryOperator *E) {
  llvm::Value *LHSCond = EvaluateExprAsBool(E->getLHS());
  
  llvm::BasicBlock *ContBlock = new llvm::BasicBlock("land_cont");
  llvm::BasicBlock *RHSBlock = new llvm::BasicBlock("land_rhs");

  llvm::BasicBlock *OrigBlock = Builder.GetInsertBlock();
  Builder.CreateCondBr(LHSCond, RHSBlock, ContBlock);
  
  EmitBlock(RHSBlock);
  llvm::Value *RHSCond = EvaluateExprAsBool(E->getRHS());
  
  // Reaquire the RHS block, as there may be subblocks inserted.
  RHSBlock = Builder.GetInsertBlock();
  EmitBlock(ContBlock);
  
  // Create a PHI node.  If we just evaluted the LHS condition, the result is
  // false.  If we evaluated both, the result is the RHS condition.
  llvm::PHINode *PN = Builder.CreatePHI(llvm::Type::Int1Ty, "land");
  PN->reserveOperandSpace(2);
  PN->addIncoming(llvm::ConstantInt::getFalse(), OrigBlock);
  PN->addIncoming(RHSCond, RHSBlock);
  
  // ZExt result to int.
  return RValue::get(Builder.CreateZExt(PN, LLVMIntTy, "land.ext"));
}

RValue CodeGenFunction::EmitBinaryLOr(const BinaryOperator *E) {
  llvm::Value *LHSCond = EvaluateExprAsBool(E->getLHS());
  
  llvm::BasicBlock *ContBlock = new llvm::BasicBlock("lor_cont");
  llvm::BasicBlock *RHSBlock = new llvm::BasicBlock("lor_rhs");
  
  llvm::BasicBlock *OrigBlock = Builder.GetInsertBlock();
  Builder.CreateCondBr(LHSCond, ContBlock, RHSBlock);
  
  EmitBlock(RHSBlock);
  llvm::Value *RHSCond = EvaluateExprAsBool(E->getRHS());
  
  // Reaquire the RHS block, as there may be subblocks inserted.
  RHSBlock = Builder.GetInsertBlock();
  EmitBlock(ContBlock);
  
  // Create a PHI node.  If we just evaluted the LHS condition, the result is
  // true.  If we evaluated both, the result is the RHS condition.
  llvm::PHINode *PN = Builder.CreatePHI(llvm::Type::Int1Ty, "lor");
  PN->reserveOperandSpace(2);
  PN->addIncoming(llvm::ConstantInt::getTrue(), OrigBlock);
  PN->addIncoming(RHSCond, RHSBlock);
  
  // ZExt result to int.
  return RValue::get(Builder.CreateZExt(PN, LLVMIntTy, "lor.ext"));
}

RValue CodeGenFunction::EmitBinaryAssign(const BinaryOperator *E) {
  LValue LHS = EmitLValue(E->getLHS());
  
  QualType RHSTy;
  RValue RHS = EmitExprWithUsualUnaryConversions(E->getRHS(), RHSTy);
  
  // Convert the RHS to the type of the LHS.
  RHS = EmitConversion(RHS, RHSTy, E->getType());
  
  // Store the value into the LHS.
  EmitStoreThroughLValue(RHS, LHS, E->getType());
  
  // Return the converted RHS.
  return RHS;
}


RValue CodeGenFunction::EmitBinaryComma(const BinaryOperator *E) {
  EmitExpr(E->getLHS());
  return EmitExpr(E->getRHS());
}
