//===--- CGExprComplex.cpp - Emit LLVM Code for Complex Exprs -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Expr nodes with complex types as LLVM code.
//
//===----------------------------------------------------------------------===//

#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "clang/AST/AST.h"
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/Support/Compiler.h"
using namespace clang;
using namespace CodeGen;

//===----------------------------------------------------------------------===//
//                        Complex Expression Emitter
//===----------------------------------------------------------------------===//

typedef CodeGenFunction::ComplexPairTy ComplexPairTy;

namespace  {
class VISIBILITY_HIDDEN ComplexExprEmitter
  : public StmtVisitor<ComplexExprEmitter, ComplexPairTy> {
  CodeGenFunction &CGF;
  llvm::LLVMBuilder &Builder;
public:
  ComplexExprEmitter(CodeGenFunction &cgf) : CGF(cgf), Builder(CGF.Builder) {
  }

  
  //===--------------------------------------------------------------------===//
  //                               Utilities
  //===--------------------------------------------------------------------===//

  /// EmitLoadOfLValue - Given an expression with complex type that represents a
  /// value l-value, this method emits the address of the l-value, then loads
  /// and returns the result.
  ComplexPairTy EmitLoadOfLValue(const Expr *E) {
    LValue LV = CGF.EmitLValue(E);
    // FIXME: Volatile
    return EmitLoadOfComplex(LV.getAddress(), false);
  }
  
  /// EmitLoadOfComplex - Given a pointer to a complex value, emit code to load
  /// the real and imaginary pieces.
  ComplexPairTy EmitLoadOfComplex(llvm::Value *SrcPtr, bool isVolatile);
  
  /// EmitStoreOfComplex - Store the specified real/imag parts into the
  /// specified value pointer.
  void EmitStoreOfComplex(ComplexPairTy Val, llvm::Value *ResPtr, bool isVol);
  
  //===--------------------------------------------------------------------===//
  //                            Visitor Methods
  //===--------------------------------------------------------------------===//

  ComplexPairTy VisitStmt(Stmt *S) {
    S->dump();
    assert(0 && "Stmt can't have complex result type!");
    return ComplexPairTy();
  }
  ComplexPairTy VisitExpr(Expr *S);
  ComplexPairTy VisitParenExpr(ParenExpr *PE) { return Visit(PE->getSubExpr());}
  ComplexPairTy VisitImaginaryLiteral(const ImaginaryLiteral *IL);
  
  // l-values.
  ComplexPairTy VisitDeclRefExpr(const Expr *E) { return EmitLoadOfLValue(E); }
  ComplexPairTy VisitArraySubscriptExpr(Expr *E) { return EmitLoadOfLValue(E); }
  ComplexPairTy VisitMemberExpr(const Expr *E) { return EmitLoadOfLValue(E); }

  // FIXME: CompoundLiteralExpr
  
  ComplexPairTy EmitCast(Expr *Op, QualType DestTy);
  ComplexPairTy VisitImplicitCastExpr(ImplicitCastExpr *E) {
    // Unlike for scalars, we don't have to worry about function->ptr demotion
    // here.
    return EmitCast(E->getSubExpr(), E->getType());
  }
  ComplexPairTy VisitCastExpr(CastExpr *E) {
    return EmitCast(E->getSubExpr(), E->getType());
  }
  ComplexPairTy VisitCallExpr(const CallExpr *E);
  
  // Operators.
  ComplexPairTy VisitPrePostIncDec(const UnaryOperator *E,
                                   bool isInc, bool isPre);
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
    return Visit(E->getSubExpr());
  }
  ComplexPairTy VisitUnaryMinus    (const UnaryOperator *E);
  ComplexPairTy VisitUnaryNot      (const UnaryOperator *E);
  // LNot,SizeOf,AlignOf,Real,Imag never return complex.
  ComplexPairTy VisitUnaryExtension(const UnaryOperator *E) {
    return Visit(E->getSubExpr());
  }
  
  ComplexPairTy VisitBinMul        (const BinaryOperator *E);
  ComplexPairTy VisitBinAdd        (const BinaryOperator *E);
  ComplexPairTy VisitBinSub        (const BinaryOperator *E);
  ComplexPairTy VisitBinDiv        (const BinaryOperator *E);
  // GCC rejects rem/and/or/xor for integer complex.
  // Logical and/or always return int, never complex.

  // No comparisons produce a complex result.
  ComplexPairTy VisitBinAssign     (const BinaryOperator *E);
  // FIXME: Compound assignment operators.
  ComplexPairTy VisitBinComma      (const BinaryOperator *E);

  
  ComplexPairTy VisitConditionalOperator(const ConditionalOperator *CO);
  ComplexPairTy VisitChooseExpr(ChooseExpr *CE);
};
}  // end anonymous namespace.

//===----------------------------------------------------------------------===//
//                                Utilities
//===----------------------------------------------------------------------===//

/// EmitLoadOfComplex - Given an RValue reference for a complex, emit code to
/// load the real and imaginary pieces, returning them as Real/Imag.
ComplexPairTy ComplexExprEmitter::EmitLoadOfComplex(llvm::Value *SrcPtr,
                                                    bool isVolatile) {
  llvm::Constant *Zero = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0);
  llvm::Constant *One  = llvm::ConstantInt::get(llvm::Type::Int32Ty, 1);
  // FIXME: It would be nice to make this "Ptr->getName()+realp"
  llvm::Value *RealPtr = Builder.CreateGEP(SrcPtr, Zero, Zero, "realp");
  llvm::Value *ImagPtr = Builder.CreateGEP(SrcPtr, Zero, One, "imagp");
  
  // FIXME: It would be nice to make this "Ptr->getName()+real"
  llvm::Value *Real = Builder.CreateLoad(RealPtr, isVolatile, "real");
  llvm::Value *Imag = Builder.CreateLoad(ImagPtr, isVolatile, "imag");
  return ComplexPairTy(Real, Imag);
}

/// EmitStoreOfComplex - Store the specified real/imag parts into the
/// specified value pointer.
void ComplexExprEmitter::EmitStoreOfComplex(ComplexPairTy Val, llvm::Value *Ptr,
                                            bool isVolatile) {
  llvm::Constant *Zero = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0);
  llvm::Constant *One  = llvm::ConstantInt::get(llvm::Type::Int32Ty, 1);
  llvm::Value *RealPtr = Builder.CreateGEP(Ptr, Zero, Zero, "real");
  llvm::Value *ImagPtr = Builder.CreateGEP(Ptr, Zero, One, "imag");
  
  Builder.CreateStore(Val.first, RealPtr, isVolatile);
  Builder.CreateStore(Val.second, ImagPtr, isVolatile);
}



//===----------------------------------------------------------------------===//
//                            Visitor Methods
//===----------------------------------------------------------------------===//

ComplexPairTy ComplexExprEmitter::VisitExpr(Expr *E) {
  fprintf(stderr, "Unimplemented complex expr!\n");
  E->dump();
  const llvm::Type *EltTy = 
    CGF.ConvertType(E->getType()->getAsComplexType()->getElementType());
  llvm::Value *U = llvm::UndefValue::get(EltTy);
  return ComplexPairTy(U, U);
}

ComplexPairTy ComplexExprEmitter::
VisitImaginaryLiteral(const ImaginaryLiteral *IL) {
  llvm::Value *Imag = CGF.EmitScalarExpr(IL->getSubExpr());
  return ComplexPairTy(llvm::Constant::getNullValue(Imag->getType()), Imag);
}


ComplexPairTy ComplexExprEmitter::VisitCallExpr(const CallExpr *E) {
  llvm::Value *AggPtr = CGF.EmitCallExpr(E).getAggregateAddr();
  return EmitLoadOfComplex(AggPtr, false);
}

ComplexPairTy ComplexExprEmitter::EmitCast(Expr *Op, QualType DestTy) {
  // Get the destination element type.
  DestTy = cast<ComplexType>(DestTy.getCanonicalType())->getElementType();

  // Two cases here: cast from (complex to complex) and (scalar to complex).
  if (const ComplexType *CT = Op->getType()->getAsComplexType()) {
    // C99 6.3.1.6: When a value of complextype is converted to another
    // complex type, both the real and imaginary parts followthe conversion
    // rules for the corresponding real types.
    ComplexPairTy Res = Visit(Op);
    QualType SrcEltTy = CT->getElementType();
    Res.first = CGF.EmitScalarConversion(Res.first, SrcEltTy, DestTy);
    Res.second = CGF.EmitScalarConversion(Res.second, SrcEltTy, DestTy);
    return Res;
  }
  // C99 6.3.1.7: When a value of real type is converted to a complex type, the
  // real part of the complex  result value is determined by the rules of
  // conversion to the corresponding real type and the imaginary part of the
  // complex result value is a positive zero or an unsigned zero.
  llvm::Value *Elt = CGF.EmitScalarExpr(Op);

  // Convert the input element to the element type of the complex.
  Elt = CGF.EmitScalarConversion(Elt, Op->getType(), DestTy);
  
  // Return (realval, 0).
  return ComplexPairTy(Elt, llvm::Constant::getNullValue(Elt->getType()));
}

ComplexPairTy ComplexExprEmitter::VisitPrePostIncDec(const UnaryOperator *E,
                                                     bool isInc, bool isPre) {
  LValue LV = CGF.EmitLValue(E->getSubExpr());
  // FIXME: Handle volatile!
  ComplexPairTy InVal = EmitLoadOfComplex(LV.getAddress(), false);
  
  int AmountVal = isInc ? 1 : -1;
  
  llvm::Value *NextVal;
  if (isa<llvm::IntegerType>(InVal.first->getType()))
    NextVal = llvm::ConstantInt::get(InVal.first->getType(), AmountVal);
  else
    NextVal = llvm::ConstantFP::get(InVal.first->getType(), AmountVal);
  
  // Add the inc/dec to the real part.
  NextVal = Builder.CreateAdd(InVal.first, NextVal, isInc ? "inc" : "dec");
  
  ComplexPairTy IncVal(NextVal, InVal.second);
  
  // Store the updated result through the lvalue.
  EmitStoreOfComplex(IncVal, LV.getAddress(), false);  /* FIXME: Volatile */
  
  // If this is a postinc, return the value read from memory, otherwise use the
  // updated value.
  return isPre ? IncVal : InVal;
}

ComplexPairTy ComplexExprEmitter::VisitUnaryMinus(const UnaryOperator *E) {
  ComplexPairTy Op = Visit(E->getSubExpr());
  llvm::Value *ResR = Builder.CreateNeg(Op.first,  "neg.r");
  llvm::Value *ResI = Builder.CreateNeg(Op.second, "neg.i");
  return ComplexPairTy(ResR, ResI);
}

ComplexPairTy ComplexExprEmitter::VisitUnaryNot(const UnaryOperator *E) {
  // ~(a+ib) = a + i*-b
  ComplexPairTy Op = Visit(E->getSubExpr());
  llvm::Value *ResI = Builder.CreateNeg(Op.second, "conj.i");
  return ComplexPairTy(Op.first, ResI);
}

ComplexPairTy ComplexExprEmitter::VisitBinAdd(const BinaryOperator *E) {
  ComplexPairTy LHS = Visit(E->getLHS());
  ComplexPairTy RHS = Visit(E->getRHS());
  
  llvm::Value *ResR = Builder.CreateAdd(LHS.first,  RHS.first,  "add.r");
  llvm::Value *ResI = Builder.CreateAdd(LHS.second, RHS.second, "add.i");

  return ComplexPairTy(ResR, ResI);
}

ComplexPairTy ComplexExprEmitter::VisitBinSub(const BinaryOperator *E) {
  ComplexPairTy LHS = Visit(E->getLHS());
  ComplexPairTy RHS = Visit(E->getRHS());
  
  llvm::Value *ResR = Builder.CreateSub(LHS.first,  RHS.first,  "sub.r");
  llvm::Value *ResI = Builder.CreateSub(LHS.second, RHS.second, "sub.i");
  
  return ComplexPairTy(ResR, ResI);
}


ComplexPairTy ComplexExprEmitter::VisitBinMul(const BinaryOperator *E) {
  ComplexPairTy LHS = Visit(E->getLHS());
  ComplexPairTy RHS = Visit(E->getRHS());
  
  llvm::Value *ResRl = Builder.CreateMul(LHS.first, RHS.first, "mul.rl");
  llvm::Value *ResRr = Builder.CreateMul(LHS.second, RHS.second, "mul.rr");
  llvm::Value *ResR  = Builder.CreateSub(ResRl, ResRr, "mul.r");
  
  llvm::Value *ResIl = Builder.CreateMul(LHS.second, RHS.first, "mul.il");
  llvm::Value *ResIr = Builder.CreateMul(LHS.first, RHS.second, "mul.ir");
  llvm::Value *ResI  = Builder.CreateAdd(ResIl, ResIr, "mul.i");

  return ComplexPairTy(ResR, ResI);
}

ComplexPairTy ComplexExprEmitter::VisitBinDiv(const BinaryOperator *E) {
  ComplexPairTy LHS = Visit(E->getLHS());
  ComplexPairTy RHS = Visit(E->getRHS());
  llvm::Value *LHSr = LHS.first, *LHSi = LHS.second;
  llvm::Value *RHSr = RHS.first, *RHSi = RHS.second;
  
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

  llvm::Value *DSTr, *DSTi;
  if (Tmp3->getType()->isFloatingPoint()) {
    DSTr = Builder.CreateFDiv(Tmp3, Tmp6, "tmp");
    DSTi = Builder.CreateFDiv(Tmp9, Tmp6, "tmp");
  } else {
    QualType ExprTy = E->getType().getCanonicalType();
    if (cast<ComplexType>(ExprTy)->getElementType()->isUnsignedIntegerType()) {
      DSTr = Builder.CreateUDiv(Tmp3, Tmp6, "tmp");
      DSTi = Builder.CreateUDiv(Tmp9, Tmp6, "tmp");
    } else {
      DSTr = Builder.CreateSDiv(Tmp3, Tmp6, "tmp");
      DSTi = Builder.CreateSDiv(Tmp9, Tmp6, "tmp");
    }
  }
    
  return ComplexPairTy(DSTr, DSTi);
}


ComplexPairTy ComplexExprEmitter::VisitBinAssign(const BinaryOperator *E) {
  assert(E->getLHS()->getType().getCanonicalType() ==
         E->getRHS()->getType().getCanonicalType() && "Invalid assignment");
  // Emit the RHS.
  ComplexPairTy Val = Visit(E->getRHS());

  // Compute the address to store into.
  LValue LHS = CGF.EmitLValue(E->getLHS());
  
  // Store into it.
  // FIXME: Volatility!
  EmitStoreOfComplex(Val, LHS.getAddress(), false);
  return Val;
}

ComplexPairTy ComplexExprEmitter::VisitBinComma(const BinaryOperator *E) {
  CGF.EmitStmt(E->getLHS());
  return Visit(E->getRHS());
}

ComplexPairTy ComplexExprEmitter::
VisitConditionalOperator(const ConditionalOperator *E) {
  llvm::BasicBlock *LHSBlock = new llvm::BasicBlock("cond.?");
  llvm::BasicBlock *RHSBlock = new llvm::BasicBlock("cond.:");
  llvm::BasicBlock *ContBlock = new llvm::BasicBlock("cond.cont");
  
  llvm::Value *Cond = CGF.EvaluateExprAsBool(E->getCond());
  Builder.CreateCondBr(Cond, LHSBlock, RHSBlock);
  
  CGF.EmitBlock(LHSBlock);
  
  // Handle the GNU extension for missing LHS.
  assert(E->getLHS() && "Must have LHS for complex value");

  ComplexPairTy LHS = Visit(E->getLHS());
  Builder.CreateBr(ContBlock);
  LHSBlock = Builder.GetInsertBlock();
  
  CGF.EmitBlock(RHSBlock);
  
  ComplexPairTy RHS = Visit(E->getRHS());
  Builder.CreateBr(ContBlock);
  RHSBlock = Builder.GetInsertBlock();
  
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
  llvm::APSInt CondVal(32);
  bool IsConst = E->getCond()->isIntegerConstantExpr(CondVal, CGF.getContext());
  assert(IsConst && "Condition of choose expr must be i-c-e"); IsConst=IsConst;
  
  // Emit the LHS or RHS as appropriate.
  return Visit(CondVal != 0 ? E->getLHS() : E->getRHS());
}

//===----------------------------------------------------------------------===//
//                         Entry Point into this File
//===----------------------------------------------------------------------===//

/// EmitComplexExpr - Emit the computation of the specified expression of
/// complex type, ignoring the result.
ComplexPairTy CodeGenFunction::EmitComplexExpr(const Expr *E) {
  assert(E && E->getType()->isComplexType() &&
         "Invalid complex expression to emit");
  
  return ComplexExprEmitter(*this).Visit(const_cast<Expr*>(E));
}

/// EmitComplexExprIntoAddr - Emit the computation of the specified expression
/// of complex type, storing into the specified Value*.
void CodeGenFunction::EmitComplexExprIntoAddr(const Expr *E,
                                              llvm::Value *DestAddr,
                                              bool DestIsVolatile) {
  assert(E && E->getType()->isComplexType() &&
         "Invalid complex expression to emit");
  ComplexExprEmitter Emitter(*this);
  ComplexPairTy Val = Emitter.Visit(const_cast<Expr*>(E));
  Emitter.EmitStoreOfComplex(Val, DestAddr, DestIsVolatile);
}
