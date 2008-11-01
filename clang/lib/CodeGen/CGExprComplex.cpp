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
  CGBuilderTy &Builder;
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
  
  /// EmitComplexToComplexCast - Emit a cast from complex value Val to DestType.
  ComplexPairTy EmitComplexToComplexCast(ComplexPairTy Val, QualType SrcType,
                                         QualType DestType);
  
  //===--------------------------------------------------------------------===//
  //                            Visitor Methods
  //===--------------------------------------------------------------------===//

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
  ComplexPairTy VisitStmtExpr(const StmtExpr *E);
  ComplexPairTy VisitOverloadExpr(const OverloadExpr *OE);

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
  ComplexPairTy VisitCXXDefaultArgExpr(CXXDefaultArgExpr *DAE) {
    return Visit(DAE->getExpr());
  }
  ComplexPairTy VisitCXXZeroInitValueExpr(CXXZeroInitValueExpr *E) {
    assert(E->getType()->isAnyComplexType() && "Expected complex type!");
    QualType Elem = E->getType()->getAsComplexType()->getElementType();
    llvm::Constant *Null = llvm::Constant::getNullValue(CGF.ConvertType(Elem));
    return ComplexPairTy(Null, Null);
  }
  
  struct BinOpInfo {
    ComplexPairTy LHS;
    ComplexPairTy RHS;
    QualType Ty;  // Computation Type.
  };    
  
  BinOpInfo EmitBinOps(const BinaryOperator *E);
  ComplexPairTy EmitCompoundAssign(const CompoundAssignOperator *E,
                                   ComplexPairTy (ComplexExprEmitter::*Func)
                                   (const BinOpInfo &));

  ComplexPairTy EmitBinAdd(const BinOpInfo &Op);
  ComplexPairTy EmitBinSub(const BinOpInfo &Op);
  ComplexPairTy EmitBinMul(const BinOpInfo &Op);
  ComplexPairTy EmitBinDiv(const BinOpInfo &Op);
  
  ComplexPairTy VisitBinMul(const BinaryOperator *E) {
    return EmitBinMul(EmitBinOps(E));
  }
  ComplexPairTy VisitBinAdd(const BinaryOperator *E) {
    return EmitBinAdd(EmitBinOps(E));
  }
  ComplexPairTy VisitBinSub(const BinaryOperator *E) {
    return EmitBinSub(EmitBinOps(E));
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
  ComplexPairTy VisitBinAssign     (const BinaryOperator *E);
  ComplexPairTy VisitBinComma      (const BinaryOperator *E);

  
  ComplexPairTy VisitConditionalOperator(const ConditionalOperator *CO);
  ComplexPairTy VisitChooseExpr(ChooseExpr *CE);

  ComplexPairTy VisitInitListExpr(InitListExpr *E);
};
}  // end anonymous namespace.

//===----------------------------------------------------------------------===//
//                                Utilities
//===----------------------------------------------------------------------===//

/// EmitLoadOfComplex - Given an RValue reference for a complex, emit code to
/// load the real and imaginary pieces, returning them as Real/Imag.
ComplexPairTy ComplexExprEmitter::EmitLoadOfComplex(llvm::Value *SrcPtr,
                                                    bool isVolatile) {
  llvm::SmallString<64> Name(SrcPtr->getNameStart(),
                             SrcPtr->getNameStart()+SrcPtr->getNameLen());
  
  Name += ".realp";
  llvm::Value *RealPtr = Builder.CreateStructGEP(SrcPtr, 0, Name.c_str());

  Name.pop_back();  // .realp -> .real
  llvm::Value *Real = Builder.CreateLoad(RealPtr, isVolatile, Name.c_str());
  
  Name.resize(Name.size()-4); // .real -> .imagp
  Name += "imagp";
  
  llvm::Value *ImagPtr = Builder.CreateStructGEP(SrcPtr, 1, Name.c_str());

  Name.pop_back();  // .imagp -> .imag
  llvm::Value *Imag = Builder.CreateLoad(ImagPtr, isVolatile, Name.c_str());
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
  return CGF.EmitCallExpr(E).getComplexVal();
}

ComplexPairTy ComplexExprEmitter::VisitOverloadExpr(const OverloadExpr *E) {
  return CGF.EmitCallExpr(E->getFn(), E->arg_begin(), 
                          E->arg_end(CGF.getContext())).getComplexVal();
}

ComplexPairTy ComplexExprEmitter::VisitStmtExpr(const StmtExpr *E) {
  return CGF.EmitCompoundStmt(*E->getSubStmt(), true).getComplexVal();
}

/// EmitComplexToComplexCast - Emit a cast from complex value Val to DestType.
ComplexPairTy ComplexExprEmitter::EmitComplexToComplexCast(ComplexPairTy Val,
                                                           QualType SrcType,
                                                           QualType DestType) {
  // Get the src/dest element type.
  SrcType = SrcType->getAsComplexType()->getElementType();
  DestType = DestType->getAsComplexType()->getElementType();

  // C99 6.3.1.6: When a value of complextype is converted to another
  // complex type, both the real and imaginary parts followthe conversion
  // rules for the corresponding real types.
  Val.first = CGF.EmitScalarConversion(Val.first, SrcType, DestType);
  Val.second = CGF.EmitScalarConversion(Val.second, SrcType, DestType);
  return Val;
}

ComplexPairTy ComplexExprEmitter::EmitCast(Expr *Op, QualType DestTy) {
  // Two cases here: cast from (complex to complex) and (scalar to complex).
  if (Op->getType()->isAnyComplexType())
    return EmitComplexToComplexCast(Visit(Op), Op->getType(), DestTy);
  
  // C99 6.3.1.7: When a value of real type is converted to a complex type, the
  // real part of the complex  result value is determined by the rules of
  // conversion to the corresponding real type and the imaginary part of the
  // complex result value is a positive zero or an unsigned zero.
  llvm::Value *Elt = CGF.EmitScalarExpr(Op);

  // Convert the input element to the element type of the complex.
  DestTy = DestTy->getAsComplexType()->getElementType();
  Elt = CGF.EmitScalarConversion(Elt, Op->getType(), DestTy);
  
  // Return (realval, 0).
  return ComplexPairTy(Elt, llvm::Constant::getNullValue(Elt->getType()));
}

ComplexPairTy ComplexExprEmitter::VisitPrePostIncDec(const UnaryOperator *E,
                                                     bool isInc, bool isPre) {
  LValue LV = CGF.EmitLValue(E->getSubExpr());
  // FIXME: Handle volatile!
  ComplexPairTy InVal = EmitLoadOfComplex(LV.getAddress(), false);
  
  uint64_t AmountVal = isInc ? 1 : -1;
  
  llvm::Value *NextVal;
  if (isa<llvm::IntegerType>(InVal.first->getType()))
    NextVal = llvm::ConstantInt::get(InVal.first->getType(), AmountVal);
  else if (InVal.first->getType() == llvm::Type::FloatTy)
    // FIXME: Handle long double.
    NextVal = 
      llvm::ConstantFP::get(llvm::APFloat(static_cast<float>(AmountVal)));
  else {
    // FIXME: Handle long double.
    assert(InVal.first->getType() == llvm::Type::DoubleTy);
    NextVal = 
      llvm::ConstantFP::get(llvm::APFloat(static_cast<double>(AmountVal)));
  }
  
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

ComplexPairTy ComplexExprEmitter::EmitBinAdd(const BinOpInfo &Op) {
  llvm::Value *ResR = Builder.CreateAdd(Op.LHS.first,  Op.RHS.first,  "add.r");
  llvm::Value *ResI = Builder.CreateAdd(Op.LHS.second, Op.RHS.second, "add.i");
  return ComplexPairTy(ResR, ResI);
}

ComplexPairTy ComplexExprEmitter::EmitBinSub(const BinOpInfo &Op) {
  llvm::Value *ResR = Builder.CreateSub(Op.LHS.first,  Op.RHS.first,  "sub.r");
  llvm::Value *ResI = Builder.CreateSub(Op.LHS.second, Op.RHS.second, "sub.i");
  return ComplexPairTy(ResR, ResI);
}


ComplexPairTy ComplexExprEmitter::EmitBinMul(const BinOpInfo &Op) {
  llvm::Value *ResRl = Builder.CreateMul(Op.LHS.first, Op.RHS.first, "mul.rl");
  llvm::Value *ResRr = Builder.CreateMul(Op.LHS.second, Op.RHS.second,"mul.rr");
  llvm::Value *ResR  = Builder.CreateSub(ResRl, ResRr, "mul.r");
  
  llvm::Value *ResIl = Builder.CreateMul(Op.LHS.second, Op.RHS.first, "mul.il");
  llvm::Value *ResIr = Builder.CreateMul(Op.LHS.first, Op.RHS.second, "mul.ir");
  llvm::Value *ResI  = Builder.CreateAdd(ResIl, ResIr, "mul.i");
  return ComplexPairTy(ResR, ResI);
}

ComplexPairTy ComplexExprEmitter::EmitBinDiv(const BinOpInfo &Op) {
  llvm::Value *LHSr = Op.LHS.first, *LHSi = Op.LHS.second;
  llvm::Value *RHSr = Op.RHS.first, *RHSi = Op.RHS.second;
  
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
    if (Op.Ty->getAsComplexType()->getElementType()->isUnsignedIntegerType()) {
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
  BinOpInfo Ops;
  Ops.LHS = Visit(E->getLHS());
  Ops.RHS = Visit(E->getRHS());
  Ops.Ty = E->getType();
  return Ops;
}


// Compound assignments.
ComplexPairTy ComplexExprEmitter::
EmitCompoundAssign(const CompoundAssignOperator *E,
                   ComplexPairTy (ComplexExprEmitter::*Func)(const BinOpInfo&)){
  QualType LHSTy = E->getLHS()->getType(), RHSTy = E->getRHS()->getType();
  
  // Load the LHS and RHS operands.
  LValue LHSLV = CGF.EmitLValue(E->getLHS());

  BinOpInfo OpInfo;
  OpInfo.Ty = E->getComputationType();

  // We know the LHS is a complex lvalue.
  OpInfo.LHS = EmitLoadOfComplex(LHSLV.getAddress(), false);// FIXME: Volatile.
  OpInfo.LHS = EmitComplexToComplexCast(OpInfo.LHS, LHSTy, OpInfo.Ty);
    
  // It is possible for the RHS to be complex or scalar.
  OpInfo.RHS = EmitCast(E->getRHS(), OpInfo.Ty);
  
  // Expand the binary operator.
  ComplexPairTy Result = (this->*Func)(OpInfo);
  
  // Truncate the result back to the LHS type.
  Result = EmitComplexToComplexCast(Result, OpInfo.Ty, LHSTy);
  
  // Store the result value into the LHS lvalue.
  EmitStoreOfComplex(Result, LHSLV.getAddress(), false); // FIXME: VOLATILE
  return Result;
}

ComplexPairTy ComplexExprEmitter::VisitBinAssign(const BinaryOperator *E) {
  assert(CGF.getContext().getCanonicalType(E->getLHS()->getType()) ==
         CGF.getContext().getCanonicalType(E->getRHS()->getType()) &&
         "Invalid assignment");
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
  llvm::BasicBlock *LHSBlock = llvm::BasicBlock::Create("cond.?");
  llvm::BasicBlock *RHSBlock = llvm::BasicBlock::Create("cond.:");
  llvm::BasicBlock *ContBlock = llvm::BasicBlock::Create("cond.cont");
  
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
  // Emit the LHS or RHS as appropriate.
  return Visit(E->isConditionTrue(CGF.getContext()) ? E->getLHS() :E->getRHS());
}

ComplexPairTy ComplexExprEmitter::VisitInitListExpr(InitListExpr *E) {
  if (E->getNumInits())
    return Visit(E->getInit(0));

  // Empty init list intializes to null
  QualType Ty = E->getType()->getAsComplexType()->getElementType();
  const llvm::Type* LTy = CGF.ConvertType(Ty);
  llvm::Value* zeroConstant = llvm::Constant::getNullValue(LTy);
  return ComplexPairTy(zeroConstant, zeroConstant);
}

//===----------------------------------------------------------------------===//
//                         Entry Point into this File
//===----------------------------------------------------------------------===//

/// EmitComplexExpr - Emit the computation of the specified expression of
/// complex type, ignoring the result.
ComplexPairTy CodeGenFunction::EmitComplexExpr(const Expr *E) {
  assert(E && E->getType()->isAnyComplexType() &&
         "Invalid complex expression to emit");
  
  return ComplexExprEmitter(*this).Visit(const_cast<Expr*>(E));
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
