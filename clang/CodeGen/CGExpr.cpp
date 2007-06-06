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
#include "clang/AST/AST.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
using namespace llvm;
using namespace clang;
using namespace CodeGen;

//===--------------------------------------------------------------------===//
//                        Miscellaneous Helper Methods
//===--------------------------------------------------------------------===//


/// EvaluateExprAsBool - Perform the usual unary conversions on the specified
/// expression and compare the result against zero, returning an Int1Ty value.
Value *CodeGenFunction::EvaluateExprAsBool(const Expr *E) {
  QualType Ty;
  RValue Val = EmitExprWithUsualUnaryConversions(E, Ty);
  return ConvertScalarValueToBool(Val, Ty);
}

//===--------------------------------------------------------------------===//
//                               Conversions
//===--------------------------------------------------------------------===//

/// EmitConversion - Convert the value specied by Val, whose type is ValTy, to
/// the type specified by DstTy, following the rules of C99 6.3.
RValue CodeGenFunction::EmitConversion(RValue Val, QualType ValTy,
                                       QualType DstTy, SourceLocation Loc) {
  ValTy = ValTy.getCanonicalType();
  DstTy = DstTy.getCanonicalType();
  if (ValTy == DstTy) return Val;
  
  if (isa<PointerType>(DstTy)) {
    const llvm::Type *DestTy = ConvertType(DstTy, Loc);
    
    // The source value may be an integer, or a pointer.
    assert(Val.isScalar() && "Can only convert from integer or pointer");
    if (isa<llvm::PointerType>(Val.getVal()->getType()))
      return RValue::get(Builder.CreateBitCast(Val.getVal(), DestTy, "conv"));
    assert(ValTy->isIntegerType() && "Not ptr->ptr or int->ptr conversion?");
    return RValue::get(Builder.CreatePtrToInt(Val.getVal(), DestTy, "conv"));
  } else if (isa<PointerType>(ValTy)) {
    // Must be an ptr to int cast.
    const llvm::Type *DestTy = ConvertType(DstTy, Loc);
    assert(isa<llvm::IntegerType>(DestTy) && "not ptr->int?");
    return RValue::get(Builder.CreateIntToPtr(Val.getVal(), DestTy, "conv"));
  } else if (const BuiltinType *DestBT = dyn_cast<BuiltinType>(DstTy)) {
    if (DestBT->getKind() == BuiltinType::Bool)
      return RValue::get(ConvertScalarValueToBool(Val, ValTy));
  }
  assert(0 && "FIXME: Unsupported conversion!");
}


/// ConvertScalarValueToBool - Convert the specified expression value to a
/// boolean (i1) truth value.  This is equivalent to "Val == 0".
Value *CodeGenFunction::ConvertScalarValueToBool(RValue Val, QualType Ty) {
  Ty = Ty.getCanonicalType();
  Value *Result;
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
      llvm::Value *Zero = Constant::getNullValue(Result->getType());
      // FIXME: llvm-gcc produces a une comparison: validate this is right.
      Result = Builder.CreateFCmpUNE(Result, Zero, "tobool");
      return Result;
    }
      
    case BuiltinType::FloatComplex:
    case BuiltinType::DoubleComplex:
    case BuiltinType::LongDoubleComplex:
      assert(0 && "comparisons against complex not implemented yet");
    }
  } else {
    assert((isa<PointerType>(Ty) || 
           cast<TagType>(Ty)->getDecl()->getKind() == Decl::Enum) &&
           "Unknown scalar type");
    // Code below handles this fine.
  }
  
  // Usual case for integers, pointers, and enums: compare against zero.
  Result = Val.getVal();
  
  // Because of the type rules of C, we often end up computing a logical value,
  // then zero extending it to int, then wanting it as a logical value again.
  // Optimize this common case.
  if (llvm::ZExtInst *ZI = dyn_cast<ZExtInst>(Result)) {
    if (ZI->getOperand(0)->getType() == llvm::Type::Int1Ty) {
      Result = ZI->getOperand(0);
      ZI->eraseFromParent();
      return Result;
    }
  }
  
  llvm::Value *Zero = Constant::getNullValue(Result->getType());
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
    return LValue::getAddr(UndefValue::get(
                              llvm::PointerType::get(llvm::Type::Int32Ty)));

  case Expr::DeclRefExprClass: return EmitDeclRefLValue(cast<DeclRefExpr>(E));
  case Expr::ParenExprClass:return EmitLValue(cast<ParenExpr>(E)->getSubExpr());
    
    
  case Expr::UnaryOperatorClass: 
    return EmitUnaryOpLValue(cast<UnaryOperator>(E));
  }
}

/// EmitLoadOfLValue - Given an expression that represents a value lvalue,
/// this method emits the address of the lvalue, then loads the result as an
/// rvalue, returning the rvalue.
RValue CodeGenFunction::EmitLoadOfLValue(const Expr *E) {
  LValue LV = EmitLValue(E);
  
  QualType ExprTy = E->getType().getCanonicalType();
  
  // FIXME: this is silly and obviously wrong for non-scalars.
  assert(!LV.isBitfield());
  return RValue::get(Builder.CreateLoad(LV.getAddress(), "tmp"));
}

/// EmitStoreThroughLValue - Store the specified rvalue into the specified
/// lvalue, where both are guaranteed to the have the same type, and that type
/// is 'Ty'.
void CodeGenFunction::EmitStoreThroughLValue(RValue Src, LValue Dst, 
                                             QualType Ty) {
  // FIXME: This is obviously bogus.
  assert(!Dst.isBitfield() && "FIXME: Don't support store to bitfield yet");
  assert(Src.isScalar() && "FIXME: Don't support store of aggregate yet");
  
  // TODO: Handle volatility etc.
  Value *Addr = Dst.getAddress();
  const llvm::Type *SrcTy = Src.getVal()->getType();
  const llvm::Type *AddrTy = 
    cast<llvm::PointerType>(Addr->getType())->getElementType();
  
  if (AddrTy != SrcTy)
    Addr = Builder.CreateBitCast(Addr, llvm::PointerType::get(SrcTy),
                                 "storetmp");
  Builder.CreateStore(Src.getVal(), Addr);
}


LValue CodeGenFunction::EmitDeclRefLValue(const DeclRefExpr *E) {
  const Decl *D = E->getDecl();
  if (isa<BlockVarDecl>(D)) {
    Value *V = LocalDeclMap[D];
    assert(V && "BlockVarDecl not entered in LocalDeclMap?");
    return LValue::getAddr(V);
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

//===--------------------------------------------------------------------===//
//                             Expression Emission
//===--------------------------------------------------------------------===//

RValue CodeGenFunction::EmitExpr(const Expr *E) {
  assert(E && "Null expression?");
  
  switch (E->getStmtClass()) {
  default:
    printf("Unimplemented expr!\n");
    E->dump();
    return RValue::get(UndefValue::get(llvm::Type::Int32Ty));
    
  // l-values.
  case Expr::DeclRefExprClass:
    // FIXME: EnumConstantDecl's are not lvalues.  This is wrong for them.
    return EmitLoadOfLValue(E);
    
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
  case Expr::BinaryOperatorClass:
    return EmitBinaryOperator(cast<BinaryOperator>(E));
  }
  
}

RValue CodeGenFunction::EmitIntegerLiteral(const IntegerLiteral *E) {
  return RValue::get(ConstantInt::get(E->getValue()));
}

RValue CodeGenFunction::EmitCastExpr(const CastExpr *E) {
  QualType SrcTy;
  RValue Src = EmitExprWithUsualUnaryConversions(E->getSubExpr(), SrcTy);
  
  // If the destination is void, just evaluate the source.
  if (E->getType()->isVoidType())
    return RValue::getAggregate(0);
  
  return EmitConversion(Src, SrcTy, E->getType(), E->getLParenLoc());
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
    return RValue::get(UndefValue::get(llvm::Type::Int32Ty));
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
  return EmitExprWithUsualUnaryConversions(E, Ty);
}

RValue CodeGenFunction::EmitUnaryMinus(const UnaryOperator *E) {
  // Unary minus performs promotions, then negates its arithmetic operand.
  QualType Ty;
  RValue V = EmitExprWithUsualUnaryConversions(E, Ty);
  
  if (V.isScalar())
    return RValue::get(Builder.CreateNeg(V.getVal(), "neg"));
  
  assert(0 && "FIXME: This doesn't handle complex operands yet");
}

RValue CodeGenFunction::EmitUnaryNot(const UnaryOperator *E) {
  // Unary not performs promotions, then complements its integer operand.
  QualType Ty;
  RValue V = EmitExprWithUsualUnaryConversions(E, Ty);
  
  if (V.isScalar())
    return RValue::get(Builder.CreateNot(V.getVal(), "neg"));
                      
  assert(0 && "FIXME: This doesn't handle integer complex operands yet (GNU)");
}


/// C99 6.5.3.3
RValue CodeGenFunction::EmitUnaryLNot(const UnaryOperator *E) {
  // Compare operand to zero.
  Value *BoolVal = EvaluateExprAsBool(E->getSubExpr());
  
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


RValue CodeGenFunction::EmitBinaryOperator(const BinaryOperator *E) {
  switch (E->getOpcode()) {
  default:
    fprintf(stderr, "Unimplemented expr!\n");
    E->dump();
    return RValue::get(UndefValue::get(llvm::Type::Int32Ty));
  case BinaryOperator::Mul: return EmitBinaryMul(E);
  case BinaryOperator::Div: return EmitBinaryDiv(E);
  case BinaryOperator::Rem: return EmitBinaryRem(E);
  case BinaryOperator::Add: return EmitBinaryAdd(E);
  case BinaryOperator::Sub: return EmitBinarySub(E);
  case BinaryOperator::Shl: return EmitBinaryShl(E);
  case BinaryOperator::Shr: return EmitBinaryShr(E);
    
    // FIXME: relational
    
  case BinaryOperator::And: return EmitBinaryAnd(E);
  case BinaryOperator::Xor: return EmitBinaryXor(E);
  case BinaryOperator::Or : return EmitBinaryOr(E);
  case BinaryOperator::LAnd: return EmitBinaryLAnd(E);
  case BinaryOperator::LOr: return EmitBinaryLOr(E);

  case BinaryOperator::Assign: return EmitBinaryAssign(E);
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
    Value *RV;
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
    Value *RV;
    // Rem in C can't be a floating point type: C99 6.5.5p2.
    if (E->getType()->isUnsignedIntegerType())
      RV = Builder.CreateURem(LHS.getVal(), RHS.getVal(), "rem");
    else
      RV = Builder.CreateSRem(LHS.getVal(), RHS.getVal(), "rem");
    return RValue::get(RV);
  }
  
  assert(0 && "FIXME: This doesn't handle complex operands yet");
}

RValue CodeGenFunction::EmitBinaryAdd(const BinaryOperator *E) {
  RValue LHS, RHS;
  EmitUsualArithmeticConversions(E, LHS, RHS);

  // FIXME: This doesn't handle ptr+int etc yet.
  
  if (LHS.isScalar())
    return RValue::get(Builder.CreateAdd(LHS.getVal(), RHS.getVal(), "add"));
  
  assert(0 && "FIXME: This doesn't handle complex operands yet");

}

RValue CodeGenFunction::EmitBinarySub(const BinaryOperator *E) {
  RValue LHS, RHS;
  EmitUsualArithmeticConversions(E, LHS, RHS);
  
  // FIXME: This doesn't handle ptr-int or ptr-ptr, etc yet.
  
  if (LHS.isScalar())
    return RValue::get(Builder.CreateSub(LHS.getVal(), RHS.getVal(), "sub"));
  
  assert(0 && "FIXME: This doesn't handle complex operands yet");
  
}

RValue CodeGenFunction::EmitBinaryShl(const BinaryOperator *E) {
  // For shifts, integer promotions are performed, but the usual arithmetic 
  // conversions are not.  The LHS and RHS need not have the same type.
  
  QualType ResTy;
  Value *LHS = EmitExprWithUsualUnaryConversions(E->getLHS(), ResTy).getVal();
  Value *RHS = EmitExprWithUsualUnaryConversions(E->getRHS(), ResTy).getVal();

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
  Value *LHS = EmitExprWithUsualUnaryConversions(E->getLHS(), ResTy).getVal();
  Value *RHS = EmitExprWithUsualUnaryConversions(E->getRHS(), ResTy).getVal();
  
  // LLVM requires the LHS and RHS to be the same type, promote or truncate the
  // RHS to the same size as the LHS.
  if (LHS->getType() != RHS->getType())
    RHS = Builder.CreateIntCast(RHS, LHS->getType(), false, "sh_prom");
  
  if (E->getType()->isUnsignedIntegerType())
    return RValue::get(Builder.CreateLShr(LHS, RHS, "shr"));
  else
    return RValue::get(Builder.CreateAShr(LHS, RHS, "shr"));
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
  Value *LHSCond = EvaluateExprAsBool(E->getLHS());
  
  BasicBlock *ContBlock = new BasicBlock("land_cont");
  BasicBlock *RHSBlock = new BasicBlock("land_rhs");

  BasicBlock *OrigBlock = Builder.GetInsertBlock();
  Builder.CreateCondBr(LHSCond, RHSBlock, ContBlock);
  
  EmitBlock(RHSBlock);
  Value *RHSCond = EvaluateExprAsBool(E->getRHS());
  
  // Reaquire the RHS block, as there may be subblocks inserted.
  RHSBlock = Builder.GetInsertBlock();
  EmitBlock(ContBlock);
  
  // Create a PHI node.  If we just evaluted the LHS condition, the result is
  // false.  If we evaluated both, the result is the RHS condition.
  PHINode *PN = Builder.CreatePHI(llvm::Type::Int1Ty, "land");
  PN->reserveOperandSpace(2);
  PN->addIncoming(ConstantInt::getFalse(), OrigBlock);
  PN->addIncoming(RHSCond, RHSBlock);
  
  // ZExt result to int.
  return RValue::get(Builder.CreateZExt(PN, LLVMIntTy, "land.ext"));
}

RValue CodeGenFunction::EmitBinaryLOr(const BinaryOperator *E) {
  Value *LHSCond = EvaluateExprAsBool(E->getLHS());
  
  BasicBlock *ContBlock = new BasicBlock("lor_cont");
  BasicBlock *RHSBlock = new BasicBlock("lor_rhs");
  
  BasicBlock *OrigBlock = Builder.GetInsertBlock();
  Builder.CreateCondBr(LHSCond, ContBlock, RHSBlock);
  
  EmitBlock(RHSBlock);
  Value *RHSCond = EvaluateExprAsBool(E->getRHS());
  
  // Reaquire the RHS block, as there may be subblocks inserted.
  RHSBlock = Builder.GetInsertBlock();
  EmitBlock(ContBlock);
  
  // Create a PHI node.  If we just evaluted the LHS condition, the result is
  // true.  If we evaluated both, the result is the RHS condition.
  PHINode *PN = Builder.CreatePHI(llvm::Type::Int1Ty, "lor");
  PN->reserveOperandSpace(2);
  PN->addIncoming(ConstantInt::getTrue(), OrigBlock);
  PN->addIncoming(RHSCond, RHSBlock);
  
  // ZExt result to int.
  return RValue::get(Builder.CreateZExt(PN, LLVMIntTy, "lor.ext"));
}

RValue CodeGenFunction::EmitBinaryAssign(const BinaryOperator *E) {
  LValue LHS = EmitLValue(E->getLHS());
  
  QualType RHSTy;
  RValue RHS = EmitExprWithUsualUnaryConversions(E->getRHS(), RHSTy);
  
  // Convert the RHS to the type of the LHS.
  // FIXME: I'm not thrilled about having to call getLocStart() here... :(
  RHS = EmitConversion(RHS, RHSTy, E->getType(), E->getLocStart());
  
  // Store the value into the LHS.
  EmitStoreThroughLValue(RHS, LHS, E->getType());
  
  // Return the converted RHS.
  return RHS;
}


RValue CodeGenFunction::EmitBinaryComma(const BinaryOperator *E) {
  EmitExpr(E->getLHS());
  return EmitExpr(E->getRHS());
}
