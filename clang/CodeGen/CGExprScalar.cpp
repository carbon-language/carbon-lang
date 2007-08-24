//===--- CGExprScalar.cpp - Emit LLVM Code for Scalar Exprs ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Expr nodes with scalar LLVM types as LLVM code.
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
using llvm::Value;

//===----------------------------------------------------------------------===//
//                         Scalar Expression Emitter
//===----------------------------------------------------------------------===//

struct BinOpInfo {
  Value *LHS;
  Value *RHS;
  QualType Ty;  // Computation Type.
  const BinaryOperator *E;
};

namespace {
class VISIBILITY_HIDDEN ScalarExprEmitter
  : public StmtVisitor<ScalarExprEmitter, Value*> {
  CodeGenFunction &CGF;
  llvm::LLVMBuilder &Builder;
public:

  ScalarExprEmitter(CodeGenFunction &cgf) : CGF(cgf), Builder(CGF.Builder) {
  }

  
  //===--------------------------------------------------------------------===//
  //                               Utilities
  //===--------------------------------------------------------------------===//

  const llvm::Type *ConvertType(QualType T) { return CGF.ConvertType(T); }
  LValue EmitLValue(const Expr *E) { return CGF.EmitLValue(E); }

  Value *EmitLoadOfLValue(LValue LV, QualType T) {
    return CGF.EmitLoadOfLValue(LV, T).getVal();
  }
    
  /// EmitLoadOfLValue - Given an expression with complex type that represents a
  /// value l-value, this method emits the address of the l-value, then loads
  /// and returns the result.
  Value *EmitLoadOfLValue(const Expr *E) {
    // FIXME: Volatile
    return EmitLoadOfLValue(EmitLValue(E), E->getType());
  }
    
  //===--------------------------------------------------------------------===//
  //                            Visitor Methods
  //===--------------------------------------------------------------------===//

  Value *VisitStmt(Stmt *S) {
    S->dump();
    assert(0 && "Stmt can't have complex result type!");
    return 0;
  }
  Value *VisitExpr(Expr *S);
  Value *VisitParenExpr(ParenExpr *PE) { return Visit(PE->getSubExpr()); }

  // Leaves.
  Value *VisitIntegerLiteral(const IntegerLiteral *E) {
    return llvm::ConstantInt::get(E->getValue());
  }
  Value *VisitFloatingLiteral(const FloatingLiteral *E) {
    return llvm::ConstantFP::get(ConvertType(E->getType()), E->getValue());
  }
  Value *VisitCharacterLiteral(const CharacterLiteral *E) {
    return llvm::ConstantInt::get(ConvertType(E->getType()), E->getValue());
  }
  Value *VisitTypesCompatibleExpr(const TypesCompatibleExpr *E) {
    return llvm::ConstantInt::get(ConvertType(E->getType()),
                                  E->typesAreCompatible());
  }
  Value *VisitSizeOfAlignOfTypeExpr(const SizeOfAlignOfTypeExpr *E) {
    return EmitSizeAlignOf(E->getArgumentType(), E->getType(), E->isSizeOf());
  }
    
  // l-values.
  Value *VisitDeclRefExpr(DeclRefExpr *E) {
    if (const EnumConstantDecl *EC = dyn_cast<EnumConstantDecl>(E->getDecl()))
      return llvm::ConstantInt::get(EC->getInitVal());
    return EmitLoadOfLValue(E);
  }
  Value *VisitArraySubscriptExpr(ArraySubscriptExpr *E);
  Value *VisitMemberExpr(Expr *E)           { return EmitLoadOfLValue(E); }
  Value *VisitOCUVectorElementExpr(Expr *E) { return EmitLoadOfLValue(E); }
  Value *VisitStringLiteral(Expr *E)  { return EmitLValue(E).getAddress(); }
  Value *VisitPreDefinedExpr(Expr *E) { return EmitLValue(E).getAddress(); }
  
  // FIXME: CompoundLiteralExpr
  Value *VisitImplicitCastExpr(const ImplicitCastExpr *E);
  Value *VisitCastExpr(const CastExpr *E) { 
    return EmitCastExpr(E->getSubExpr(), E->getType());
  }
  Value *EmitCastExpr(const Expr *E, QualType T);

  Value *VisitCallExpr(const CallExpr *E) {
    return CGF.EmitCallExpr(E).getVal();
  }
  
  // Unary Operators.
  Value *VisitPrePostIncDec(const UnaryOperator *E, bool isInc, bool isPre);
  Value *VisitUnaryPostDec(const UnaryOperator *E) {
    return VisitPrePostIncDec(E, false, false);
  }
  Value *VisitUnaryPostInc(const UnaryOperator *E) {
    return VisitPrePostIncDec(E, true, false);
  }
  Value *VisitUnaryPreDec(const UnaryOperator *E) {
    return VisitPrePostIncDec(E, false, true);
  }
  Value *VisitUnaryPreInc(const UnaryOperator *E) {
    return VisitPrePostIncDec(E, true, true);
  }
  Value *VisitUnaryAddrOf(const UnaryOperator *E) {
    return EmitLValue(E->getSubExpr()).getAddress();
  }
  Value *VisitUnaryDeref(const Expr *E) { return EmitLoadOfLValue(E); }
  Value *VisitUnaryPlus(const UnaryOperator *E) {
    return Visit(E->getSubExpr());
  }
  Value *VisitUnaryMinus    (const UnaryOperator *E);
  Value *VisitUnaryNot      (const UnaryOperator *E);
  Value *VisitUnaryLNot     (const UnaryOperator *E);
  Value *VisitUnarySizeOf   (const UnaryOperator *E) {
    return EmitSizeAlignOf(E->getSubExpr()->getType(), E->getType(), true);
  }
  Value *VisitUnaryAlignOf  (const UnaryOperator *E) {
    return EmitSizeAlignOf(E->getSubExpr()->getType(), E->getType(), false);
  }
  Value *EmitSizeAlignOf(QualType TypeToSize, QualType RetType,
                               bool isSizeOf);
  // FIXME: Real,Imag.
  Value *VisitUnaryExtension(const UnaryOperator *E) {
    return Visit(E->getSubExpr());
  }
  
  // Binary Operators.
  Value *EmitMul(const BinOpInfo &Ops) {
    return Builder.CreateMul(Ops.LHS, Ops.RHS, "mul");
  }
  Value *EmitDiv(const BinOpInfo &Ops);
  Value *EmitRem(const BinOpInfo &Ops);
  Value *EmitAdd(const BinOpInfo &Ops);
  Value *EmitSub(const BinOpInfo &Ops);
  Value *EmitShl(const BinOpInfo &Ops);
  Value *EmitShr(const BinOpInfo &Ops);
  Value *EmitAnd(const BinOpInfo &Ops) {
    return Builder.CreateAnd(Ops.LHS, Ops.RHS, "and");
  }
  Value *EmitXor(const BinOpInfo &Ops) {
    return Builder.CreateXor(Ops.LHS, Ops.RHS, "xor");
  }
  Value *EmitOr (const BinOpInfo &Ops) {
    return Builder.CreateOr(Ops.LHS, Ops.RHS, "or");
  }

  BinOpInfo EmitBinOps(const BinaryOperator *E);
  Value *EmitCompoundAssign(const BinaryOperator *E,
                            Value *(ScalarExprEmitter::*F)(const BinOpInfo &));

  // Binary operators and binary compound assignment operators.
#define HANDLEBINOP(OP) \
  Value *VisitBin ## OP(const BinaryOperator *E) {                 \
    return Emit ## OP(EmitBinOps(E));                              \
  }                                                                \
  Value *VisitBin ## OP ## Assign(const BinaryOperator *E) {       \
    return EmitCompoundAssign(E, &ScalarExprEmitter::Emit ## OP);  \
  }
  HANDLEBINOP(Mul);
  HANDLEBINOP(Div);
  HANDLEBINOP(Rem);
  HANDLEBINOP(Add);
  //         (Sub) - Sub is handled specially below for ptr-ptr subtract.
  HANDLEBINOP(Shl);
  HANDLEBINOP(Shr);
  HANDLEBINOP(And);
  HANDLEBINOP(Xor);
  HANDLEBINOP(Or);
#undef HANDLEBINOP
  Value *VisitBinSub(const BinaryOperator *E);
  Value *VisitBinSubAssign(const BinaryOperator *E) {
    return EmitCompoundAssign(E, &ScalarExprEmitter::EmitSub);
  }
  
  // Comparisons.
  Value *EmitCompare(const BinaryOperator *E, unsigned UICmpOpc,
                     unsigned SICmpOpc, unsigned FCmpOpc);
#define VISITCOMP(CODE, UI, SI, FP) \
    Value *VisitBin##CODE(const BinaryOperator *E) { \
      return EmitCompare(E, llvm::ICmpInst::UI, llvm::ICmpInst::SI, \
                         llvm::FCmpInst::FP); }
  VISITCOMP(LT, ICMP_ULT, ICMP_SLT, FCMP_OLT);
  VISITCOMP(GT, ICMP_UGT, ICMP_SGT, FCMP_OGT);
  VISITCOMP(LE, ICMP_ULE, ICMP_SLE, FCMP_OLE);
  VISITCOMP(GE, ICMP_UGE, ICMP_SGE, FCMP_OGE);
  VISITCOMP(EQ, ICMP_EQ , ICMP_EQ , FCMP_OEQ);
  VISITCOMP(NE, ICMP_NE , ICMP_NE , FCMP_UNE);
#undef VISITCOMP
  
  Value *VisitBinAssign     (const BinaryOperator *E);

  Value *VisitBinLAnd       (const BinaryOperator *E);
  Value *VisitBinLOr        (const BinaryOperator *E);

  // FIXME: Compound assignment operators.
  Value *VisitBinComma      (const BinaryOperator *E);

  // Other Operators.
  Value *VisitConditionalOperator(const ConditionalOperator *CO);
  Value *VisitChooseExpr(ChooseExpr *CE);
  Value *VisitObjCStringLiteral(const ObjCStringLiteral *E) {
    return CGF.EmitObjCStringLiteral(E);
  }
};
}  // end anonymous namespace.

//===----------------------------------------------------------------------===//
//                                Utilities
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//                            Visitor Methods
//===----------------------------------------------------------------------===//

Value *ScalarExprEmitter::VisitExpr(Expr *E) {
  fprintf(stderr, "Unimplemented scalar expr!\n");
  E->dump();
  if (E->getType()->isVoidType())
    return 0;
  return llvm::UndefValue::get(CGF.ConvertType(E->getType()));
}

Value *ScalarExprEmitter::VisitArraySubscriptExpr(ArraySubscriptExpr *E) {
  // Emit subscript expressions in rvalue context's.  For most cases, this just
  // loads the lvalue formed by the subscript expr.  However, we have to be
  // careful, because the base of a vector subscript is occasionally an rvalue,
  // so we can't get it as an lvalue.
  if (!E->getBase()->getType()->isVectorType())
    return EmitLoadOfLValue(E);
  
  // Handle the vector case.  The base must be a vector, the index must be an
  // integer value.
  Value *Base = Visit(E->getBase());
  Value *Idx  = Visit(E->getIdx());
  
  // FIXME: Convert Idx to i32 type.
  return Builder.CreateExtractElement(Base, Idx, "vecext");
}

/// VisitImplicitCastExpr - Implicit casts are the same as normal casts, but
/// also handle things like function to pointer-to-function decay, and array to
/// pointer decay.
Value *ScalarExprEmitter::VisitImplicitCastExpr(const ImplicitCastExpr *E) {
  const Expr *Op = E->getSubExpr();
  
  // If this is due to array->pointer conversion, emit the array expression as
  // an l-value.
  if (Op->getType()->isArrayType()) {
    // FIXME: For now we assume that all source arrays map to LLVM arrays.  This
    // will not true when we add support for VLAs.
    llvm::Value *V = EmitLValue(Op).getAddress();  // Bitfields can't be arrays.
    
    assert(isa<llvm::PointerType>(V->getType()) &&
           isa<llvm::ArrayType>(cast<llvm::PointerType>(V->getType())
                                ->getElementType()) &&
           "Doesn't support VLAs yet!");
    llvm::Constant *Idx0 = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0);
    return Builder.CreateGEP(V, Idx0, Idx0, "arraydecay");
  }
  
  return EmitCastExpr(Op, E->getType());
}


// VisitCastExpr - Emit code for an explicit or implicit cast.  Implicit casts
// have to handle a more broad range of conversions than explicit casts, as they
// handle things like function to ptr-to-function decay etc.
Value *ScalarExprEmitter::EmitCastExpr(const Expr *E, QualType DestTy) {
  RValue Src = CGF.EmitAnyExpr(E);
  
  // If the destination is void, just evaluate the source.
  if (DestTy->isVoidType())
    return 0;
  
  // FIXME: Refactor EmitConversion to not return an RValue.  Sink it into this
  // method.
  return CGF.EmitConversion(Src, E->getType(), DestTy).getVal();
}

//===----------------------------------------------------------------------===//
//                             Unary Operators
//===----------------------------------------------------------------------===//

Value *ScalarExprEmitter::VisitPrePostIncDec(const UnaryOperator *E,
                                             bool isInc, bool isPre) {
  LValue LV = EmitLValue(E->getSubExpr());
  // FIXME: Handle volatile!
  Value *InVal = CGF.EmitLoadOfLValue(LV/* false*/,
                                           E->getSubExpr()->getType()).getVal();
  
  int AmountVal = isInc ? 1 : -1;
  
  Value *NextVal;
  if (isa<llvm::IntegerType>(InVal->getType()))
    NextVal = llvm::ConstantInt::get(InVal->getType(), AmountVal);
  else
    NextVal = llvm::ConstantFP::get(InVal->getType(), AmountVal);
  
  // Add the inc/dec to the real part.
  NextVal = Builder.CreateAdd(InVal, NextVal, isInc ? "inc" : "dec");
  
  // Store the updated result through the lvalue.
  CGF.EmitStoreThroughLValue(RValue::get(NextVal), LV, 
                             E->getSubExpr()->getType());

  // If this is a postinc, return the value read from memory, otherwise use the
  // updated value.
  return isPre ? NextVal : InVal;
}


Value *ScalarExprEmitter::VisitUnaryMinus(const UnaryOperator *E) {
  Value *Op = Visit(E->getSubExpr());
  return Builder.CreateNeg(Op, "neg");
}

Value *ScalarExprEmitter::VisitUnaryNot(const UnaryOperator *E) {
  Value *Op = Visit(E->getSubExpr());
  return Builder.CreateNot(Op, "neg");
}

Value *ScalarExprEmitter::VisitUnaryLNot(const UnaryOperator *E) {
  // Compare operand to zero.
  Value *BoolVal = CGF.EvaluateExprAsBool(E->getSubExpr());
  
  // Invert value.
  // TODO: Could dynamically modify easy computations here.  For example, if
  // the operand is an icmp ne, turn into icmp eq.
  BoolVal = Builder.CreateNot(BoolVal, "lnot");
  
  // ZExt result to int.
  return Builder.CreateZExt(BoolVal, CGF.LLVMIntTy, "lnot.ext");
}

/// EmitSizeAlignOf - Return the size or alignment of the 'TypeToSize' type as
/// an integer (RetType).
Value *ScalarExprEmitter::EmitSizeAlignOf(QualType TypeToSize, 
                                                QualType RetType,bool isSizeOf){
  /// FIXME: This doesn't handle VLAs yet!
  std::pair<uint64_t, unsigned> Info =
    CGF.getContext().getTypeInfo(TypeToSize, SourceLocation());
  
  uint64_t Val = isSizeOf ? Info.first : Info.second;
  Val /= 8;  // Return size in bytes, not bits.
  
  assert(RetType->isIntegerType() && "Result type must be an integer!");
  
  unsigned ResultWidth = CGF.getContext().getTypeSize(RetType,SourceLocation());
  return llvm::ConstantInt::get(llvm::APInt(ResultWidth, Val));
}

//===----------------------------------------------------------------------===//
//                           Binary Operators
//===----------------------------------------------------------------------===//

BinOpInfo ScalarExprEmitter::EmitBinOps(const BinaryOperator *E) {
  BinOpInfo Result;
  Result.LHS = Visit(E->getLHS());
  Result.RHS = Visit(E->getRHS());
  Result.Ty  = E->getType();
  Result.E = E;
  return Result;
}

Value *ScalarExprEmitter::EmitCompoundAssign(const BinaryOperator *E,
                      Value *(ScalarExprEmitter::*Func)(const BinOpInfo &)) {
  QualType LHSTy = E->getLHS()->getType(), RHSTy = E->getRHS()->getType();

  BinOpInfo OpInfo;

  // Load the LHS and RHS operands.
  LValue LHSLV = EmitLValue(E->getLHS());
  OpInfo.LHS = EmitLoadOfLValue(LHSLV, LHSTy);
  
  // FIXME: It is possible for the RHS to be complex.
  OpInfo.RHS = Visit(E->getRHS());
  
  // Convert the LHS/RHS values to the computation type.
  const CompoundAssignOperator *CAO = cast<CompoundAssignOperator>(E);
  QualType ComputeType = CAO->getComputationType();
  
  // FIXME: it's possible for the computation type to be complex if the RHS
  // is complex.  Handle this!
  OpInfo.LHS = CGF.EmitConversion(RValue::get(OpInfo.LHS), LHSTy,
                                  ComputeType).getVal();
  
  // Do not merge types for -= where the LHS is a pointer.
  if (E->getOpcode() != BinaryOperator::SubAssign &&
      E->getLHS()->getType()->isPointerType()) {
    OpInfo.RHS = CGF.EmitConversion(RValue::get(OpInfo.RHS), RHSTy,
                                    ComputeType).getVal();
  }
  OpInfo.Ty = ComputeType;
  OpInfo.E = E;
  
  // Expand the binary operator.
  Value *Result = (this->*Func)(OpInfo);
  
  // Truncate the result back to the LHS type.
  Result = CGF.EmitConversion(RValue::get(Result), ComputeType, LHSTy).getVal();
  
  // Store the result value into the LHS lvalue.
  CGF.EmitStoreThroughLValue(RValue::get(Result), LHSLV, E->getType());

  return Result;
}


Value *ScalarExprEmitter::EmitDiv(const BinOpInfo &Ops) {
  if (Ops.LHS->getType()->isFloatingPoint())
    return Builder.CreateFDiv(Ops.LHS, Ops.RHS, "div");
  else if (Ops.Ty->isUnsignedIntegerType())
    return Builder.CreateUDiv(Ops.LHS, Ops.RHS, "div");
  else
    return Builder.CreateSDiv(Ops.LHS, Ops.RHS, "div");
}

Value *ScalarExprEmitter::EmitRem(const BinOpInfo &Ops) {
  // Rem in C can't be a floating point type: C99 6.5.5p2.
  if (Ops.Ty->isUnsignedIntegerType())
    return Builder.CreateURem(Ops.LHS, Ops.RHS, "rem");
  else
    return Builder.CreateSRem(Ops.LHS, Ops.RHS, "rem");
}


Value *ScalarExprEmitter::EmitAdd(const BinOpInfo &Ops) {
  if (!Ops.Ty->isPointerType())
    return Builder.CreateAdd(Ops.LHS, Ops.RHS, "add");
  
  // FIXME: What about a pointer to a VLA?
  if (isa<llvm::PointerType>(Ops.LHS->getType())) // pointer + int
    return Builder.CreateGEP(Ops.LHS, Ops.RHS, "add.ptr");
  // int + pointer
  return Builder.CreateGEP(Ops.RHS, Ops.LHS, "add.ptr");
}

Value *ScalarExprEmitter::EmitSub(const BinOpInfo &Ops) {
  if (!isa<llvm::PointerType>(Ops.LHS->getType()))
    return Builder.CreateSub(Ops.LHS, Ops.RHS, "sub");
  
  // pointer - int
  assert(!isa<llvm::PointerType>(Ops.RHS->getType()) &&
         "ptr-ptr shouldn't get here");
  // FIXME: The pointer could point to a VLA.
  Value *NegatedRHS = Builder.CreateNeg(Ops.RHS, "sub.ptr.neg");
  return Builder.CreateGEP(Ops.LHS, NegatedRHS, "sub.ptr");
}

Value *ScalarExprEmitter::VisitBinSub(const BinaryOperator *E) {
  // "X - Y" is different from "X -= Y" in one case: when Y is a pointer.  In
  // the compound assignment case it is invalid, so just handle it here.
  if (!E->getRHS()->getType()->isPointerType())
    return EmitSub(EmitBinOps(E));
  
  // pointer - pointer
  Value *LHS = Visit(E->getLHS());
  Value *RHS = Visit(E->getRHS());
  
  const PointerType *LHSPtrType = E->getLHS()->getType()->getAsPointerType();
  assert(LHSPtrType == E->getRHS()->getType()->getAsPointerType() &&
         "Can't subtract different pointer types");
  
  QualType LHSElementType = LHSPtrType->getPointeeType();
  uint64_t ElementSize = CGF.getContext().getTypeSize(LHSElementType,
                                                      SourceLocation()) / 8;
  
  const llvm::Type *ResultType = ConvertType(E->getType());
  LHS = Builder.CreatePtrToInt(LHS, ResultType, "sub.ptr.lhs.cast");
  RHS = Builder.CreatePtrToInt(RHS, ResultType, "sub.ptr.rhs.cast");
  Value *BytesBetween = Builder.CreateSub(LHS, RHS, "sub.ptr.sub");
  
  // HACK: LLVM doesn't have an divide instruction that 'knows' there is no
  // remainder.  As such, we handle common power-of-two cases here to generate
  // better code.
  if (llvm::isPowerOf2_64(ElementSize)) {
    Value *ShAmt =
    llvm::ConstantInt::get(ResultType, llvm::Log2_64(ElementSize));
    return Builder.CreateAShr(BytesBetween, ShAmt, "sub.ptr.shr");
  }
  
  // Otherwise, do a full sdiv.
  Value *BytesPerElt = llvm::ConstantInt::get(ResultType, ElementSize);
  return Builder.CreateSDiv(BytesBetween, BytesPerElt, "sub.ptr.div");
}


Value *ScalarExprEmitter::EmitShl(const BinOpInfo &Ops) {
  // LLVM requires the LHS and RHS to be the same type: promote or truncate the
  // RHS to the same size as the LHS.
  Value *RHS = Ops.RHS;
  if (Ops.LHS->getType() != RHS->getType())
    RHS = Builder.CreateIntCast(RHS, Ops.LHS->getType(), false, "sh_prom");
  
  return Builder.CreateShl(Ops.LHS, RHS, "shl");
}

Value *ScalarExprEmitter::EmitShr(const BinOpInfo &Ops) {
  // LLVM requires the LHS and RHS to be the same type: promote or truncate the
  // RHS to the same size as the LHS.
  Value *RHS = Ops.RHS;
  if (Ops.LHS->getType() != RHS->getType())
    RHS = Builder.CreateIntCast(RHS, Ops.LHS->getType(), false, "sh_prom");
  
  if (Ops.Ty->isUnsignedIntegerType())
    return Builder.CreateLShr(Ops.LHS, RHS, "shr");
  return Builder.CreateAShr(Ops.LHS, RHS, "shr");
}

Value *ScalarExprEmitter::EmitCompare(const BinaryOperator *E,unsigned UICmpOpc,
                                      unsigned SICmpOpc, unsigned FCmpOpc) {
  llvm::Value *Result;
  QualType LHSTy = E->getLHS()->getType();
  if (!LHSTy->isComplexType()) {
    Value *LHS = Visit(E->getLHS());
    Value *RHS = Visit(E->getRHS());
    
    if (LHS->getType()->isFloatingPoint()) {
      Result = Builder.CreateFCmp((llvm::FCmpInst::Predicate)FCmpOpc,
                                  LHS, RHS, "cmp");
    } else if (LHSTy->isUnsignedIntegerType()) {
      Result = Builder.CreateICmp((llvm::ICmpInst::Predicate)UICmpOpc,
                                  LHS, RHS, "cmp");
    } else {
      // Signed integers and pointers.
      Result = Builder.CreateICmp((llvm::ICmpInst::Predicate)SICmpOpc,
                                  LHS, RHS, "cmp");
    }
  } else {
    // Complex Comparison: can only be an equality comparison.
    CodeGenFunction::ComplexPairTy LHS = CGF.EmitComplexExpr(E->getLHS());
    CodeGenFunction::ComplexPairTy RHS = CGF.EmitComplexExpr(E->getRHS());
    
    QualType CETy = 
      cast<ComplexType>(LHSTy.getCanonicalType())->getElementType();
    
    llvm::Value *ResultR, *ResultI;
    if (CETy->isRealFloatingType()) {
      ResultR = Builder.CreateFCmp((llvm::FCmpInst::Predicate)FCmpOpc,
                                   LHS.first, RHS.first, "cmp.r");
      ResultI = Builder.CreateFCmp((llvm::FCmpInst::Predicate)FCmpOpc,
                                   LHS.second, RHS.second, "cmp.i");
    } else {
      // Complex comparisons can only be equality comparisons.  As such, signed
      // and unsigned opcodes are the same.
      ResultR = Builder.CreateICmp((llvm::ICmpInst::Predicate)UICmpOpc,
                                   LHS.first, RHS.first, "cmp.r");
      ResultI = Builder.CreateICmp((llvm::ICmpInst::Predicate)UICmpOpc,
                                   LHS.second, RHS.second, "cmp.i");
    }
    
    if (E->getOpcode() == BinaryOperator::EQ) {
      Result = Builder.CreateAnd(ResultR, ResultI, "and.ri");
    } else {
      assert(E->getOpcode() == BinaryOperator::NE &&
             "Complex comparison other than == or != ?");
      Result = Builder.CreateOr(ResultR, ResultI, "or.ri");
    }
  }
  
  // ZExt result to int.
  return Builder.CreateZExt(Result, CGF.LLVMIntTy, "cmp.ext");
}

Value *ScalarExprEmitter::VisitBinAssign(const BinaryOperator *E) {
  LValue LHS = EmitLValue(E->getLHS());
  Value *RHS = Visit(E->getRHS());
  
  // Store the value into the LHS.
  // FIXME: Volatility!
  CGF.EmitStoreThroughLValue(RValue::get(RHS), LHS, E->getType());
  
  // Return the RHS.
  return RHS;
}

Value *ScalarExprEmitter::VisitBinLAnd(const BinaryOperator *E) {
  Value *LHSCond = CGF.EvaluateExprAsBool(E->getLHS());
  
  llvm::BasicBlock *ContBlock = new llvm::BasicBlock("land_cont");
  llvm::BasicBlock *RHSBlock = new llvm::BasicBlock("land_rhs");
  
  llvm::BasicBlock *OrigBlock = Builder.GetInsertBlock();
  Builder.CreateCondBr(LHSCond, RHSBlock, ContBlock);
  
  CGF.EmitBlock(RHSBlock);
  Value *RHSCond = CGF.EvaluateExprAsBool(E->getRHS());
  
  // Reaquire the RHS block, as there may be subblocks inserted.
  RHSBlock = Builder.GetInsertBlock();
  CGF.EmitBlock(ContBlock);
  
  // Create a PHI node.  If we just evaluted the LHS condition, the result is
  // false.  If we evaluated both, the result is the RHS condition.
  llvm::PHINode *PN = Builder.CreatePHI(llvm::Type::Int1Ty, "land");
  PN->reserveOperandSpace(2);
  PN->addIncoming(llvm::ConstantInt::getFalse(), OrigBlock);
  PN->addIncoming(RHSCond, RHSBlock);
  
  // ZExt result to int.
  return Builder.CreateZExt(PN, CGF.LLVMIntTy, "land.ext");
}

Value *ScalarExprEmitter::VisitBinLOr(const BinaryOperator *E) {
  Value *LHSCond = CGF.EvaluateExprAsBool(E->getLHS());
  
  llvm::BasicBlock *ContBlock = new llvm::BasicBlock("lor_cont");
  llvm::BasicBlock *RHSBlock = new llvm::BasicBlock("lor_rhs");
  
  llvm::BasicBlock *OrigBlock = Builder.GetInsertBlock();
  Builder.CreateCondBr(LHSCond, ContBlock, RHSBlock);
  
  CGF.EmitBlock(RHSBlock);
  Value *RHSCond = CGF.EvaluateExprAsBool(E->getRHS());
  
  // Reaquire the RHS block, as there may be subblocks inserted.
  RHSBlock = Builder.GetInsertBlock();
  CGF.EmitBlock(ContBlock);
  
  // Create a PHI node.  If we just evaluted the LHS condition, the result is
  // true.  If we evaluated both, the result is the RHS condition.
  llvm::PHINode *PN = Builder.CreatePHI(llvm::Type::Int1Ty, "lor");
  PN->reserveOperandSpace(2);
  PN->addIncoming(llvm::ConstantInt::getTrue(), OrigBlock);
  PN->addIncoming(RHSCond, RHSBlock);
  
  // ZExt result to int.
  return Builder.CreateZExt(PN, CGF.LLVMIntTy, "lor.ext");
}

Value *ScalarExprEmitter::VisitBinComma(const BinaryOperator *E) {
  CGF.EmitStmt(E->getLHS());
  return Visit(E->getRHS());
}

//===----------------------------------------------------------------------===//
//                             Other Operators
//===----------------------------------------------------------------------===//

Value *ScalarExprEmitter::
VisitConditionalOperator(const ConditionalOperator *E) {
  llvm::BasicBlock *LHSBlock = new llvm::BasicBlock("cond.?");
  llvm::BasicBlock *RHSBlock = new llvm::BasicBlock("cond.:");
  llvm::BasicBlock *ContBlock = new llvm::BasicBlock("cond.cont");
  
  Value *Cond = CGF.EvaluateExprAsBool(E->getCond());
  Builder.CreateCondBr(Cond, LHSBlock, RHSBlock);
  
  CGF.EmitBlock(LHSBlock);
  
  // Handle the GNU extension for missing LHS.
  Value *LHS = E->getLHS() ? Visit(E->getLHS()) : Cond;
  Builder.CreateBr(ContBlock);
  LHSBlock = Builder.GetInsertBlock();
  
  CGF.EmitBlock(RHSBlock);
  
  Value *RHS = Visit(E->getRHS());
  Builder.CreateBr(ContBlock);
  RHSBlock = Builder.GetInsertBlock();
  
  CGF.EmitBlock(ContBlock);
  
  // Create a PHI node for the real part.
  llvm::PHINode *PN = Builder.CreatePHI(LHS->getType(), "cond");
  PN->reserveOperandSpace(2);
  PN->addIncoming(LHS, LHSBlock);
  PN->addIncoming(RHS, RHSBlock);
  return PN;
}

Value *ScalarExprEmitter::VisitChooseExpr(ChooseExpr *E) {
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
Value *CodeGenFunction::EmitScalarExpr(const Expr *E) {
  assert(E && !hasAggregateLLVMType(E->getType()) &&
         "Invalid scalar expression to emit");
  
  return ScalarExprEmitter(*this).Visit(const_cast<Expr*>(E));
}
