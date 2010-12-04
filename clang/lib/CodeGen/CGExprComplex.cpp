//===--- CGExprComplex.cpp - Emit LLVM Code for Complex Exprs -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Expr nodes with complex types as LLVM code.
//
//===----------------------------------------------------------------------===//

#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/StmtVisitor.h"
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/ADT/SmallString.h"
using namespace clang;
using namespace CodeGen;

//===----------------------------------------------------------------------===//
//                        Complex Expression Emitter
//===----------------------------------------------------------------------===//

typedef CodeGenFunction::ComplexPairTy ComplexPairTy;

namespace  {
class ComplexExprEmitter
  : public StmtVisitor<ComplexExprEmitter, ComplexPairTy> {
  CodeGenFunction &CGF;
  CGBuilderTy &Builder;
  // True is we should ignore the value of a
  bool IgnoreReal;
  bool IgnoreImag;
public:
  ComplexExprEmitter(CodeGenFunction &cgf, bool ir=false, bool ii=false)
    : CGF(cgf), Builder(CGF.Builder), IgnoreReal(ir), IgnoreImag(ii) {
  }


  //===--------------------------------------------------------------------===//
  //                               Utilities
  //===--------------------------------------------------------------------===//

  bool TestAndClearIgnoreReal() {
    bool I = IgnoreReal;
    IgnoreReal = false;
    return I;
  }
  bool TestAndClearIgnoreImag() {
    bool I = IgnoreImag;
    IgnoreImag = false;
    return I;
  }

  /// EmitLoadOfLValue - Given an expression with complex type that represents a
  /// value l-value, this method emits the address of the l-value, then loads
  /// and returns the result.
  ComplexPairTy EmitLoadOfLValue(const Expr *E) {
    LValue LV = CGF.EmitLValue(E);
    if (LV.isSimple())
      return EmitLoadOfComplex(LV.getAddress(), LV.isVolatileQualified());

    assert(LV.isPropertyRef() && "Unknown LValue type!");
    return CGF.EmitLoadOfPropertyRefLValue(LV).getComplexVal();
  }

  /// EmitLoadOfComplex - Given a pointer to a complex value, emit code to load
  /// the real and imaginary pieces.
  ComplexPairTy EmitLoadOfComplex(llvm::Value *SrcPtr, bool isVolatile);

  /// EmitStoreOfComplex - Store the specified real/imag parts into the
  /// specified value pointer.
  void EmitStoreOfComplex(ComplexPairTy Val, llvm::Value *ResPtr, bool isVol);

  /// EmitComplexToComplexCast - Emit a cast from complex value Val to DestType.
  ComplexPairTy EmitComplexToComplexCast(ComplexPairTy Val, QualType SrcType,
                                         QualType DestType);

  //===--------------------------------------------------------------------===//
  //                            Visitor Methods
  //===--------------------------------------------------------------------===//

  ComplexPairTy Visit(Expr *E) {
    llvm::DenseMap<const Expr *, ComplexPairTy>::iterator I = 
      CGF.ConditionalSaveComplexExprs.find(E);
    if (I != CGF.ConditionalSaveComplexExprs.end())
      return I->second;
      
      return StmtVisitor<ComplexExprEmitter, ComplexPairTy>::Visit(E);
  }
    
  ComplexPairTy VisitStmt(Stmt *S) {
    S->dump(CGF.getContext().getSourceManager());
    assert(0 && "Stmt can't have complex result type!");
    return ComplexPairTy();
  }
  ComplexPairTy VisitExpr(Expr *S);
  ComplexPairTy VisitParenExpr(ParenExpr *PE) { return Visit(PE->getSubExpr());}
  ComplexPairTy VisitImaginaryLiteral(const ImaginaryLiteral *IL);

  // l-values.
  ComplexPairTy VisitDeclRefExpr(const Expr *E) { return EmitLoadOfLValue(E); }
  ComplexPairTy VisitObjCIvarRefExpr(ObjCIvarRefExpr *E) {
    return EmitLoadOfLValue(E);
  }
  ComplexPairTy VisitObjCPropertyRefExpr(ObjCPropertyRefExpr *E) {
    assert(E->getObjectKind() == OK_Ordinary);
    return EmitLoadOfLValue(E);
  }
  ComplexPairTy VisitObjCMessageExpr(ObjCMessageExpr *E) {
    return CGF.EmitObjCMessageExpr(E).getComplexVal();
  }
  ComplexPairTy VisitArraySubscriptExpr(Expr *E) { return EmitLoadOfLValue(E); }
  ComplexPairTy VisitMemberExpr(const Expr *E) { return EmitLoadOfLValue(E); }

  // FIXME: CompoundLiteralExpr

  ComplexPairTy EmitCast(CastExpr::CastKind CK, Expr *Op, QualType DestTy);
  ComplexPairTy VisitImplicitCastExpr(ImplicitCastExpr *E) {
    // Unlike for scalars, we don't have to worry about function->ptr demotion
    // here.
    return EmitCast(E->getCastKind(), E->getSubExpr(), E->getType());
  }
  ComplexPairTy VisitCastExpr(CastExpr *E) {
    return EmitCast(E->getCastKind(), E->getSubExpr(), E->getType());
  }
  ComplexPairTy VisitCallExpr(const CallExpr *E);
  ComplexPairTy VisitStmtExpr(const StmtExpr *E);

  // Operators.
  ComplexPairTy VisitPrePostIncDec(const UnaryOperator *E,
                                   bool isInc, bool isPre) {
    LValue LV = CGF.EmitLValue(E->getSubExpr());
    return CGF.EmitComplexPrePostIncDec(E, LV, isInc, isPre);
  }
  ComplexPairTy VisitUnaryPostDec(const UnaryOperator *E) {
    return VisitPrePostIncDec(E, false, false);
  }
  ComplexPairTy VisitUnaryPostInc(const UnaryOperator *E) {
    return VisitPrePostIncDec(E, true, false);
  }
  ComplexPairTy VisitUnaryPreDec(const UnaryOperator *E) {
    return VisitPrePostIncDec(E, false, true);
  }
  ComplexPairTy VisitUnaryPreInc(const UnaryOperator *E) {
    return VisitPrePostIncDec(E, true, true);
  }
  ComplexPairTy VisitUnaryDeref(const Expr *E) { return EmitLoadOfLValue(E); }
  ComplexPairTy VisitUnaryPlus     (const UnaryOperator *E) {
    TestAndClearIgnoreReal();
    TestAndClearIgnoreImag();
    return Visit(E->getSubExpr());
  }
  ComplexPairTy VisitUnaryMinus    (const UnaryOperator *E);
  ComplexPairTy VisitUnaryNot      (const UnaryOperator *E);
  // LNot,Real,Imag never return complex.
  ComplexPairTy VisitUnaryExtension(const UnaryOperator *E) {
    return Visit(E->getSubExpr());
  }
  ComplexPairTy VisitCXXDefaultArgExpr(CXXDefaultArgExpr *DAE) {
    return Visit(DAE->getExpr());
  }
  ComplexPairTy VisitCXXExprWithTemporaries(CXXExprWithTemporaries *E) {
    return CGF.EmitCXXExprWithTemporaries(E).getComplexVal();
  }
  ComplexPairTy VisitCXXScalarValueInitExpr(CXXScalarValueInitExpr *E) {
    assert(E->getType()->isAnyComplexType() && "Expected complex type!");
    QualType Elem = E->getType()->getAs<ComplexType>()->getElementType();
    llvm::Constant *Null = llvm::Constant::getNullValue(CGF.ConvertType(Elem));
    return ComplexPairTy(Null, Null);
  }
  ComplexPairTy VisitImplicitValueInitExpr(ImplicitValueInitExpr *E) {
    assert(E->getType()->isAnyComplexType() && "Expected complex type!");
    QualType Elem = E->getType()->getAs<ComplexType>()->getElementType();
    llvm::Constant *Null =
                       llvm::Constant::getNullValue(CGF.ConvertType(Elem));
    return ComplexPairTy(Null, Null);
  }

  struct BinOpInfo {
    ComplexPairTy LHS;
    ComplexPairTy RHS;
    QualType Ty;  // Computation Type.
  };

  BinOpInfo EmitBinOps(const BinaryOperator *E);
  LValue EmitCompoundAssignLValue(const CompoundAssignOperator *E,
                                  ComplexPairTy (ComplexExprEmitter::*Func)
                                  (const BinOpInfo &),
                                  ComplexPairTy &Val);
  ComplexPairTy EmitCompoundAssign(const CompoundAssignOperator *E,
                                   ComplexPairTy (ComplexExprEmitter::*Func)
                                   (const BinOpInfo &));

  ComplexPairTy EmitBinAdd(const BinOpInfo &Op);
  ComplexPairTy EmitBinSub(const BinOpInfo &Op);
  ComplexPairTy EmitBinMul(const BinOpInfo &Op);
  ComplexPairTy EmitBinDiv(const BinOpInfo &Op);

  ComplexPairTy VisitBinAdd(const BinaryOperator *E) {
    return EmitBinAdd(EmitBinOps(E));
  }
  ComplexPairTy VisitBinSub(const BinaryOperator *E) {
    return EmitBinSub(EmitBinOps(E));
  }
  ComplexPairTy VisitBinMul(const BinaryOperator *E) {
    return EmitBinMul(EmitBinOps(E));
  }
  ComplexPairTy VisitBinDiv(const BinaryOperator *E) {
    return EmitBinDiv(EmitBinOps(E));
  }

  // Compound assignments.
  ComplexPairTy VisitBinAddAssign(const CompoundAssignOperator *E) {
    return EmitCompoundAssign(E, &ComplexExprEmitter::EmitBinAdd);
  }
  ComplexPairTy VisitBinSubAssign(const CompoundAssignOperator *E) {
    return EmitCompoundAssign(E, &ComplexExprEmitter::EmitBinSub);
  }
  ComplexPairTy VisitBinMulAssign(const CompoundAssignOperator *E) {
    return EmitCompoundAssign(E, &ComplexExprEmitter::EmitBinMul);
  }
  ComplexPairTy VisitBinDivAssign(const CompoundAssignOperator *E) {
    return EmitCompoundAssign(E, &ComplexExprEmitter::EmitBinDiv);
  }

  // GCC rejects rem/and/or/xor for integer complex.
  // Logical and/or always return int, never complex.

  // No comparisons produce a complex result.

  LValue EmitBinAssignLValue(const BinaryOperator *E,
                             ComplexPairTy &Val);
  ComplexPairTy VisitBinAssign     (const BinaryOperator *E);
  ComplexPairTy VisitBinComma      (const BinaryOperator *E);


  ComplexPairTy VisitConditionalOperator(const ConditionalOperator *CO);
  ComplexPairTy VisitChooseExpr(ChooseExpr *CE);

  ComplexPairTy VisitInitListExpr(InitListExpr *E);

  ComplexPairTy VisitVAArgExpr(VAArgExpr *E);
};
}  // end anonymous namespace.

//===----------------------------------------------------------------------===//
//                                Utilities
//===----------------------------------------------------------------------===//

/// EmitLoadOfComplex - Given an RValue reference for a complex, emit code to
/// load the real and imaginary pieces, returning them as Real/Imag.
ComplexPairTy ComplexExprEmitter::EmitLoadOfComplex(llvm::Value *SrcPtr,
                                                    bool isVolatile) {
  llvm::Value *Real=0, *Imag=0;

  if (!IgnoreReal || isVolatile) {
    llvm::Value *RealP = Builder.CreateStructGEP(SrcPtr, 0,
                                                 SrcPtr->getName() + ".realp");
    Real = Builder.CreateLoad(RealP, isVolatile, SrcPtr->getName() + ".real");
  }

  if (!IgnoreImag || isVolatile) {
    llvm::Value *ImagP = Builder.CreateStructGEP(SrcPtr, 1,
                                                 SrcPtr->getName() + ".imagp");
    Imag = Builder.CreateLoad(ImagP, isVolatile, SrcPtr->getName() + ".imag");
  }
  return ComplexPairTy(Real, Imag);
}

/// EmitStoreOfComplex - Store the specified real/imag parts into the
/// specified value pointer.
void ComplexExprEmitter::EmitStoreOfComplex(ComplexPairTy Val, llvm::Value *Ptr,
                                            bool isVolatile) {
  llvm::Value *RealPtr = Builder.CreateStructGEP(Ptr, 0, "real");
  llvm::Value *ImagPtr = Builder.CreateStructGEP(Ptr, 1, "imag");

  Builder.CreateStore(Val.first, RealPtr, isVolatile);
  Builder.CreateStore(Val.second, ImagPtr, isVolatile);
}



//===----------------------------------------------------------------------===//
//                            Visitor Methods
//===----------------------------------------------------------------------===//

ComplexPairTy ComplexExprEmitter::VisitExpr(Expr *E) {
  CGF.ErrorUnsupported(E, "complex expression");
  const llvm::Type *EltTy =
    CGF.ConvertType(E->getType()->getAs<ComplexType>()->getElementType());
  llvm::Value *U = llvm::UndefValue::get(EltTy);
  return ComplexPairTy(U, U);
}

ComplexPairTy ComplexExprEmitter::
VisitImaginaryLiteral(const ImaginaryLiteral *IL) {
  llvm::Value *Imag = CGF.EmitScalarExpr(IL->getSubExpr());
  return
        ComplexPairTy(llvm::Constant::getNullValue(Imag->getType()), Imag);
}


ComplexPairTy ComplexExprEmitter::VisitCallExpr(const CallExpr *E) {
  if (E->getCallReturnType()->isReferenceType())
    return EmitLoadOfLValue(E);

  return CGF.EmitCallExpr(E).getComplexVal();
}

ComplexPairTy ComplexExprEmitter::VisitStmtExpr(const StmtExpr *E) {
  return CGF.EmitCompoundStmt(*E->getSubStmt(), true).getComplexVal();
}

/// EmitComplexToComplexCast - Emit a cast from complex value Val to DestType.
ComplexPairTy ComplexExprEmitter::EmitComplexToComplexCast(ComplexPairTy Val,
                                                           QualType SrcType,
                                                           QualType DestType) {
  // Get the src/dest element type.
  SrcType = SrcType->getAs<ComplexType>()->getElementType();
  DestType = DestType->getAs<ComplexType>()->getElementType();

  // C99 6.3.1.6: When a value of complex type is converted to another
  // complex type, both the real and imaginary parts follow the conversion
  // rules for the corresponding real types.
  Val.first = CGF.EmitScalarConversion(Val.first, SrcType, DestType);
  Val.second = CGF.EmitScalarConversion(Val.second, SrcType, DestType);
  return Val;
}

ComplexPairTy ComplexExprEmitter::EmitCast(CastExpr::CastKind CK, Expr *Op, 
                                           QualType DestTy) {
  switch (CK) {
  case CK_GetObjCProperty: {
    LValue LV = CGF.EmitLValue(Op);
    assert(LV.isPropertyRef() && "Unknown LValue type!");
    return CGF.EmitLoadOfPropertyRefLValue(LV).getComplexVal();
  }

  case CK_NoOp:
  case CK_LValueToRValue:
    return Visit(Op);

  // TODO: do all of these
  default:
    break;
  }

  // Two cases here: cast from (complex to complex) and (scalar to complex).
  if (Op->getType()->isAnyComplexType())
    return EmitComplexToComplexCast(Visit(Op), Op->getType(), DestTy);

  // FIXME: We should be looking at all of the cast kinds here, not
  // cherry-picking the ones we have test cases for.
  if (CK == CK_LValueBitCast) {
    llvm::Value *V = CGF.EmitLValue(Op).getAddress();
    V = Builder.CreateBitCast(V, 
                      CGF.ConvertType(CGF.getContext().getPointerType(DestTy)));
    // FIXME: Are the qualifiers correct here?
    return EmitLoadOfComplex(V, DestTy.isVolatileQualified());
  }
  
  // C99 6.3.1.7: When a value of real type is converted to a complex type, the
  // real part of the complex result value is determined by the rules of
  // conversion to the corresponding real type and the imaginary part of the
  // complex result value is a positive zero or an unsigned zero.
  llvm::Value *Elt = CGF.EmitScalarExpr(Op);

  // Convert the input element to the element type of the complex.
  DestTy = DestTy->getAs<ComplexType>()->getElementType();
  Elt = CGF.EmitScalarConversion(Elt, Op->getType(), DestTy);

  // Return (realval, 0).
  return ComplexPairTy(Elt, llvm::Constant::getNullValue(Elt->getType()));
}

ComplexPairTy ComplexExprEmitter::VisitUnaryMinus(const UnaryOperator *E) {
  TestAndClearIgnoreReal();
  TestAndClearIgnoreImag();
  ComplexPairTy Op = Visit(E->getSubExpr());

  llvm::Value *ResR, *ResI;
  if (Op.first->getType()->isFloatingPointTy()) {
    ResR = Builder.CreateFNeg(Op.first,  "neg.r");
    ResI = Builder.CreateFNeg(Op.second, "neg.i");
  } else {
    ResR = Builder.CreateNeg(Op.first,  "neg.r");
    ResI = Builder.CreateNeg(Op.second, "neg.i");
  }
  return ComplexPairTy(ResR, ResI);
}

ComplexPairTy ComplexExprEmitter::VisitUnaryNot(const UnaryOperator *E) {
  TestAndClearIgnoreReal();
  TestAndClearIgnoreImag();
  // ~(a+ib) = a + i*-b
  ComplexPairTy Op = Visit(E->getSubExpr());
  llvm::Value *ResI;
  if (Op.second->getType()->isFloatingPointTy())
    ResI = Builder.CreateFNeg(Op.second, "conj.i");
  else
    ResI = Builder.CreateNeg(Op.second, "conj.i");

  return ComplexPairTy(Op.first, ResI);
}

ComplexPairTy ComplexExprEmitter::EmitBinAdd(const BinOpInfo &Op) {
  llvm::Value *ResR, *ResI;

  if (Op.LHS.first->getType()->isFloatingPointTy()) {
    ResR = Builder.CreateFAdd(Op.LHS.first,  Op.RHS.first,  "add.r");
    ResI = Builder.CreateFAdd(Op.LHS.second, Op.RHS.second, "add.i");
  } else {
    ResR = Builder.CreateAdd(Op.LHS.first,  Op.RHS.first,  "add.r");
    ResI = Builder.CreateAdd(Op.LHS.second, Op.RHS.second, "add.i");
  }
  return ComplexPairTy(ResR, ResI);
}

ComplexPairTy ComplexExprEmitter::EmitBinSub(const BinOpInfo &Op) {
  llvm::Value *ResR, *ResI;
  if (Op.LHS.first->getType()->isFloatingPointTy()) {
    ResR = Builder.CreateFSub(Op.LHS.first,  Op.RHS.first,  "sub.r");
    ResI = Builder.CreateFSub(Op.LHS.second, Op.RHS.second, "sub.i");
  } else {
    ResR = Builder.CreateSub(Op.LHS.first,  Op.RHS.first,  "sub.r");
    ResI = Builder.CreateSub(Op.LHS.second, Op.RHS.second, "sub.i");
  }
  return ComplexPairTy(ResR, ResI);
}


ComplexPairTy ComplexExprEmitter::EmitBinMul(const BinOpInfo &Op) {
  using llvm::Value;
  Value *ResR, *ResI;

  if (Op.LHS.first->getType()->isFloatingPointTy()) {
    Value *ResRl = Builder.CreateFMul(Op.LHS.first, Op.RHS.first, "mul.rl");
    Value *ResRr = Builder.CreateFMul(Op.LHS.second, Op.RHS.second,"mul.rr");
    ResR  = Builder.CreateFSub(ResRl, ResRr, "mul.r");

    Value *ResIl = Builder.CreateFMul(Op.LHS.second, Op.RHS.first, "mul.il");
    Value *ResIr = Builder.CreateFMul(Op.LHS.first, Op.RHS.second, "mul.ir");
    ResI  = Builder.CreateFAdd(ResIl, ResIr, "mul.i");
  } else {
    Value *ResRl = Builder.CreateMul(Op.LHS.first, Op.RHS.first, "mul.rl");
    Value *ResRr = Builder.CreateMul(Op.LHS.second, Op.RHS.second,"mul.rr");
    ResR  = Builder.CreateSub(ResRl, ResRr, "mul.r");

    Value *ResIl = Builder.CreateMul(Op.LHS.second, Op.RHS.first, "mul.il");
    Value *ResIr = Builder.CreateMul(Op.LHS.first, Op.RHS.second, "mul.ir");
    ResI  = Builder.CreateAdd(ResIl, ResIr, "mul.i");
  }
  return ComplexPairTy(ResR, ResI);
}

ComplexPairTy ComplexExprEmitter::EmitBinDiv(const BinOpInfo &Op) {
  llvm::Value *LHSr = Op.LHS.first, *LHSi = Op.LHS.second;
  llvm::Value *RHSr = Op.RHS.first, *RHSi = Op.RHS.second;


  llvm::Value *DSTr, *DSTi;
  if (Op.LHS.first->getType()->isFloatingPointTy()) {
    // (a+ib) / (c+id) = ((ac+bd)/(cc+dd)) + i((bc-ad)/(cc+dd))
    llvm::Value *Tmp1 = Builder.CreateFMul(LHSr, RHSr, "tmp"); // a*c
    llvm::Value *Tmp2 = Builder.CreateFMul(LHSi, RHSi, "tmp"); // b*d
    llvm::Value *Tmp3 = Builder.CreateFAdd(Tmp1, Tmp2, "tmp"); // ac+bd

    llvm::Value *Tmp4 = Builder.CreateFMul(RHSr, RHSr, "tmp"); // c*c
    llvm::Value *Tmp5 = Builder.CreateFMul(RHSi, RHSi, "tmp"); // d*d
    llvm::Value *Tmp6 = Builder.CreateFAdd(Tmp4, Tmp5, "tmp"); // cc+dd

    llvm::Value *Tmp7 = Builder.CreateFMul(LHSi, RHSr, "tmp"); // b*c
    llvm::Value *Tmp8 = Builder.CreateFMul(LHSr, RHSi, "tmp"); // a*d
    llvm::Value *Tmp9 = Builder.CreateFSub(Tmp7, Tmp8, "tmp"); // bc-ad

    DSTr = Builder.CreateFDiv(Tmp3, Tmp6, "tmp");
    DSTi = Builder.CreateFDiv(Tmp9, Tmp6, "tmp");
  } else {
    // (a+ib) / (c+id) = ((ac+bd)/(cc+dd)) + i((bc-ad)/(cc+dd))
    llvm::Value *Tmp1 = Builder.CreateMul(LHSr, RHSr, "tmp"); // a*c
    llvm::Value *Tmp2 = Builder.CreateMul(LHSi, RHSi, "tmp"); // b*d
    llvm::Value *Tmp3 = Builder.CreateAdd(Tmp1, Tmp2, "tmp"); // ac+bd

    llvm::Value *Tmp4 = Builder.CreateMul(RHSr, RHSr, "tmp"); // c*c
    llvm::Value *Tmp5 = Builder.CreateMul(RHSi, RHSi, "tmp"); // d*d
    llvm::Value *Tmp6 = Builder.CreateAdd(Tmp4, Tmp5, "tmp"); // cc+dd

    llvm::Value *Tmp7 = Builder.CreateMul(LHSi, RHSr, "tmp"); // b*c
    llvm::Value *Tmp8 = Builder.CreateMul(LHSr, RHSi, "tmp"); // a*d
    llvm::Value *Tmp9 = Builder.CreateSub(Tmp7, Tmp8, "tmp"); // bc-ad

    if (Op.Ty->getAs<ComplexType>()->getElementType()->isUnsignedIntegerType()) {
      DSTr = Builder.CreateUDiv(Tmp3, Tmp6, "tmp");
      DSTi = Builder.CreateUDiv(Tmp9, Tmp6, "tmp");
    } else {
      DSTr = Builder.CreateSDiv(Tmp3, Tmp6, "tmp");
      DSTi = Builder.CreateSDiv(Tmp9, Tmp6, "tmp");
    }
  }

  return ComplexPairTy(DSTr, DSTi);
}

ComplexExprEmitter::BinOpInfo
ComplexExprEmitter::EmitBinOps(const BinaryOperator *E) {
  TestAndClearIgnoreReal();
  TestAndClearIgnoreImag();
  BinOpInfo Ops;
  Ops.LHS = Visit(E->getLHS());
  Ops.RHS = Visit(E->getRHS());
  Ops.Ty = E->getType();
  return Ops;
}


LValue ComplexExprEmitter::
EmitCompoundAssignLValue(const CompoundAssignOperator *E,
          ComplexPairTy (ComplexExprEmitter::*Func)(const BinOpInfo&),
                         ComplexPairTy &Val) {
  TestAndClearIgnoreReal();
  TestAndClearIgnoreImag();
  QualType LHSTy = E->getLHS()->getType(), RHSTy = E->getRHS()->getType();

  BinOpInfo OpInfo;

  // Load the RHS and LHS operands.
  // __block variables need to have the rhs evaluated first, plus this should
  // improve codegen a little.
  OpInfo.Ty = E->getComputationResultType();

  // The RHS should have been converted to the computation type.
  assert(OpInfo.Ty->isAnyComplexType());
  assert(CGF.getContext().hasSameUnqualifiedType(OpInfo.Ty,
                                                 E->getRHS()->getType()));
  OpInfo.RHS = Visit(E->getRHS());
  
  LValue LHS = CGF.EmitLValue(E->getLHS());
  // We know the LHS is a complex lvalue.
  ComplexPairTy LHSComplexPair;
  if (LHS.isPropertyRef())
    LHSComplexPair = CGF.EmitLoadOfPropertyRefLValue(LHS).getComplexVal();
  else
    LHSComplexPair = EmitLoadOfComplex(LHS.getAddress(),
                                       LHS.isVolatileQualified());
  
  OpInfo.LHS = EmitComplexToComplexCast(LHSComplexPair, LHSTy, OpInfo.Ty);

  // Expand the binary operator.
  ComplexPairTy Result = (this->*Func)(OpInfo);

  // Truncate the result back to the LHS type.
  Result = EmitComplexToComplexCast(Result, OpInfo.Ty, LHSTy);
  Val = Result;

  // Store the result value into the LHS lvalue.
  if (LHS.isPropertyRef())
    CGF.EmitStoreThroughPropertyRefLValue(RValue::getComplex(Result), LHS);
  else
    EmitStoreOfComplex(Result, LHS.getAddress(), LHS.isVolatileQualified());

  return LHS;
}

// Compound assignments.
ComplexPairTy ComplexExprEmitter::
EmitCompoundAssign(const CompoundAssignOperator *E,
                   ComplexPairTy (ComplexExprEmitter::*Func)(const BinOpInfo&)){
  ComplexPairTy Val;
  LValue LV = EmitCompoundAssignLValue(E, Func, Val);

  // The result of an assignment in C is the assigned r-value.
  if (!CGF.getContext().getLangOptions().CPlusPlus)
    return Val;

  // Objective-C property assignment never reloads the value following a store.
  if (LV.isPropertyRef())
    return Val;

  // If the lvalue is non-volatile, return the computed value of the assignment.
  if (!LV.isVolatileQualified())
    return Val;

  return EmitLoadOfComplex(LV.getAddress(), LV.isVolatileQualified());
}

LValue ComplexExprEmitter::EmitBinAssignLValue(const BinaryOperator *E,
                                               ComplexPairTy &Val) {
  assert(CGF.getContext().hasSameUnqualifiedType(E->getLHS()->getType(), 
                                                 E->getRHS()->getType()) &&
         "Invalid assignment");
  TestAndClearIgnoreReal();
  TestAndClearIgnoreImag();

  // Emit the RHS.
  Val = Visit(E->getRHS());

  // Compute the address to store into.
  LValue LHS = CGF.EmitLValue(E->getLHS());

  // Store the result value into the LHS lvalue.
  if (LHS.isPropertyRef())
    CGF.EmitStoreThroughPropertyRefLValue(RValue::getComplex(Val), LHS);
  else
    EmitStoreOfComplex(Val, LHS.getAddress(), LHS.isVolatileQualified());

  return LHS;
}

ComplexPairTy ComplexExprEmitter::VisitBinAssign(const BinaryOperator *E) {
  ComplexPairTy Val;
  LValue LV = EmitBinAssignLValue(E, Val);

  // The result of an assignment in C is the assigned r-value.
  if (!CGF.getContext().getLangOptions().CPlusPlus)
    return Val;

  // Objective-C property assignment never reloads the value following a store.
  if (LV.isPropertyRef())
    return Val;

  // If the lvalue is non-volatile, return the computed value of the assignment.
  if (!LV.isVolatileQualified())
    return Val;

  return EmitLoadOfComplex(LV.getAddress(), LV.isVolatileQualified());
}

ComplexPairTy ComplexExprEmitter::VisitBinComma(const BinaryOperator *E) {
  CGF.EmitStmt(E->getLHS());
  CGF.EnsureInsertPoint();
  return Visit(E->getRHS());
}

ComplexPairTy ComplexExprEmitter::
VisitConditionalOperator(const ConditionalOperator *E) {
  TestAndClearIgnoreReal();
  TestAndClearIgnoreImag();
  llvm::BasicBlock *LHSBlock = CGF.createBasicBlock("cond.true");
  llvm::BasicBlock *RHSBlock = CGF.createBasicBlock("cond.false");
  llvm::BasicBlock *ContBlock = CGF.createBasicBlock("cond.end");

  if (E->getLHS())
    CGF.EmitBranchOnBoolExpr(E->getCond(), LHSBlock, RHSBlock);
  else {
    Expr *save = E->getSAVE();
    assert(save && "VisitConditionalOperator - save is null");
    // Intentianlly not doing direct assignment to ConditionalSaveExprs[save] !!
    ComplexPairTy SaveVal = Visit(save);
    CGF.ConditionalSaveComplexExprs[save] = SaveVal;
    CGF.EmitBranchOnBoolExpr(E->getCond(), LHSBlock, RHSBlock);
  }

  CGF.EmitBlock(LHSBlock);
  ComplexPairTy LHS = Visit(E->getTrueExpr());
  LHSBlock = Builder.GetInsertBlock();
  CGF.EmitBranch(ContBlock);

  CGF.EmitBlock(RHSBlock);

  ComplexPairTy RHS = Visit(E->getRHS());
  RHSBlock = Builder.GetInsertBlock();
  CGF.EmitBranch(ContBlock);

  CGF.EmitBlock(ContBlock);

  // Create a PHI node for the real part.
  llvm::PHINode *RealPN = Builder.CreatePHI(LHS.first->getType(), "cond.r");
  RealPN->reserveOperandSpace(2);
  RealPN->addIncoming(LHS.first, LHSBlock);
  RealPN->addIncoming(RHS.first, RHSBlock);

  // Create a PHI node for the imaginary part.
  llvm::PHINode *ImagPN = Builder.CreatePHI(LHS.first->getType(), "cond.i");
  ImagPN->reserveOperandSpace(2);
  ImagPN->addIncoming(LHS.second, LHSBlock);
  ImagPN->addIncoming(RHS.second, RHSBlock);

  return ComplexPairTy(RealPN, ImagPN);
}

ComplexPairTy ComplexExprEmitter::VisitChooseExpr(ChooseExpr *E) {
  return Visit(E->getChosenSubExpr(CGF.getContext()));
}

ComplexPairTy ComplexExprEmitter::VisitInitListExpr(InitListExpr *E) {
    bool Ignore = TestAndClearIgnoreReal();
    (void)Ignore;
    assert (Ignore == false && "init list ignored");
    Ignore = TestAndClearIgnoreImag();
    (void)Ignore;
    assert (Ignore == false && "init list ignored");
  if (E->getNumInits())
    return Visit(E->getInit(0));

  // Empty init list intializes to null
  QualType Ty = E->getType()->getAs<ComplexType>()->getElementType();
  const llvm::Type* LTy = CGF.ConvertType(Ty);
  llvm::Value* zeroConstant = llvm::Constant::getNullValue(LTy);
  return ComplexPairTy(zeroConstant, zeroConstant);
}

ComplexPairTy ComplexExprEmitter::VisitVAArgExpr(VAArgExpr *E) {
  llvm::Value *ArgValue = CGF.EmitVAListRef(E->getSubExpr());
  llvm::Value *ArgPtr = CGF.EmitVAArg(ArgValue, E->getType());

  if (!ArgPtr) {
    CGF.ErrorUnsupported(E, "complex va_arg expression");
    const llvm::Type *EltTy =
      CGF.ConvertType(E->getType()->getAs<ComplexType>()->getElementType());
    llvm::Value *U = llvm::UndefValue::get(EltTy);
    return ComplexPairTy(U, U);
  }

  // FIXME Volatility.
  return EmitLoadOfComplex(ArgPtr, false);
}

//===----------------------------------------------------------------------===//
//                         Entry Point into this File
//===----------------------------------------------------------------------===//

/// EmitComplexExpr - Emit the computation of the specified expression of
/// complex type, ignoring the result.
ComplexPairTy CodeGenFunction::EmitComplexExpr(const Expr *E, bool IgnoreReal,
                                               bool IgnoreImag) {
  assert(E && E->getType()->isAnyComplexType() &&
         "Invalid complex expression to emit");

  return ComplexExprEmitter(*this, IgnoreReal, IgnoreImag)
    .Visit(const_cast<Expr*>(E));
}

/// EmitComplexExprIntoAddr - Emit the computation of the specified expression
/// of complex type, storing into the specified Value*.
void CodeGenFunction::EmitComplexExprIntoAddr(const Expr *E,
                                              llvm::Value *DestAddr,
                                              bool DestIsVolatile) {
  assert(E && E->getType()->isAnyComplexType() &&
         "Invalid complex expression to emit");
  ComplexExprEmitter Emitter(*this);
  ComplexPairTy Val = Emitter.Visit(const_cast<Expr*>(E));
  Emitter.EmitStoreOfComplex(Val, DestAddr, DestIsVolatile);
}

/// StoreComplexToAddr - Store a complex number into the specified address.
void CodeGenFunction::StoreComplexToAddr(ComplexPairTy V,
                                         llvm::Value *DestAddr,
                                         bool DestIsVolatile) {
  ComplexExprEmitter(*this).EmitStoreOfComplex(V, DestAddr, DestIsVolatile);
}

/// LoadComplexFromAddr - Load a complex number from the specified address.
ComplexPairTy CodeGenFunction::LoadComplexFromAddr(llvm::Value *SrcAddr,
                                                   bool SrcIsVolatile) {
  return ComplexExprEmitter(*this).EmitLoadOfComplex(SrcAddr, SrcIsVolatile);
}

LValue CodeGenFunction::EmitComplexAssignmentLValue(const BinaryOperator *E) {
  ComplexPairTy Val; // ignored

  ComplexPairTy(ComplexExprEmitter::*Op)(const ComplexExprEmitter::BinOpInfo &);

  switch (E->getOpcode()) {
  case BO_Assign:
    return ComplexExprEmitter(*this).EmitBinAssignLValue(E, Val);

  case BO_MulAssign: Op = &ComplexExprEmitter::EmitBinMul; break;
  case BO_DivAssign: Op = &ComplexExprEmitter::EmitBinDiv; break;
  case BO_SubAssign: Op = &ComplexExprEmitter::EmitBinSub; break;
  case BO_AddAssign: Op = &ComplexExprEmitter::EmitBinAdd; break;

  default:
    llvm_unreachable("unexpected complex compound assignment");
    Op = 0;
  }

  return ComplexExprEmitter(*this).EmitCompoundAssignLValue(
                                   cast<CompoundAssignOperator>(E), Op, Val);
}
