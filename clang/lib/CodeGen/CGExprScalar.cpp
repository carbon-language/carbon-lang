//===--- CGExprScalar.cpp - Emit LLVM Code for Scalar Exprs ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Expr nodes with scalar LLVM types as LLVM code.
//
//===----------------------------------------------------------------------===//

#include "CodeGenFunction.h"
#include "CGObjCRuntime.h"
#include "CodeGenModule.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Intrinsics.h"
#include "llvm/Module.h"
#include "llvm/Support/CFG.h"
#include "llvm/Target/TargetData.h"
#include <cstdarg>

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
class ScalarExprEmitter
  : public StmtVisitor<ScalarExprEmitter, Value*> {
  CodeGenFunction &CGF;
  CGBuilderTy &Builder;
  bool IgnoreResultAssign;
  llvm::LLVMContext &VMContext;
public:

  ScalarExprEmitter(CodeGenFunction &cgf, bool ira=false)
    : CGF(cgf), Builder(CGF.Builder), IgnoreResultAssign(ira),
      VMContext(cgf.getLLVMContext()) {
  }

  //===--------------------------------------------------------------------===//
  //                               Utilities
  //===--------------------------------------------------------------------===//

  bool TestAndClearIgnoreResultAssign() {
    bool I = IgnoreResultAssign;
    IgnoreResultAssign = false;
    return I;
  }

  const llvm::Type *ConvertType(QualType T) { return CGF.ConvertType(T); }
  LValue EmitLValue(const Expr *E) { return CGF.EmitLValue(E); }

  Value *EmitLoadOfLValue(LValue LV, QualType T) {
    return CGF.EmitLoadOfLValue(LV, T).getScalarVal();
  }

  /// EmitLoadOfLValue - Given an expression with complex type that represents a
  /// value l-value, this method emits the address of the l-value, then loads
  /// and returns the result.
  Value *EmitLoadOfLValue(const Expr *E) {
    return EmitLoadOfLValue(EmitLValue(E), E->getType());
  }

  /// EmitConversionToBool - Convert the specified expression value to a
  /// boolean (i1) truth value.  This is equivalent to "Val != 0".
  Value *EmitConversionToBool(Value *Src, QualType DstTy);

  /// EmitScalarConversion - Emit a conversion from the specified type to the
  /// specified destination type, both of which are LLVM scalar types.
  Value *EmitScalarConversion(Value *Src, QualType SrcTy, QualType DstTy);

  /// EmitComplexToScalarConversion - Emit a conversion from the specified
  /// complex type to the specified destination type, where the destination type
  /// is an LLVM scalar type.
  Value *EmitComplexToScalarConversion(CodeGenFunction::ComplexPairTy Src,
                                       QualType SrcTy, QualType DstTy);

  //===--------------------------------------------------------------------===//
  //                            Visitor Methods
  //===--------------------------------------------------------------------===//

  Value *VisitStmt(Stmt *S) {
    S->dump(CGF.getContext().getSourceManager());
    assert(0 && "Stmt can't have complex result type!");
    return 0;
  }
  Value *VisitExpr(Expr *S);
  
  Value *VisitParenExpr(ParenExpr *PE) { return Visit(PE->getSubExpr()); }

  // Leaves.
  Value *VisitIntegerLiteral(const IntegerLiteral *E) {
    return llvm::ConstantInt::get(VMContext, E->getValue());
  }
  Value *VisitFloatingLiteral(const FloatingLiteral *E) {
    return llvm::ConstantFP::get(VMContext, E->getValue());
  }
  Value *VisitCharacterLiteral(const CharacterLiteral *E) {
    return llvm::ConstantInt::get(ConvertType(E->getType()), E->getValue());
  }
  Value *VisitCXXBoolLiteralExpr(const CXXBoolLiteralExpr *E) {
    return llvm::ConstantInt::get(ConvertType(E->getType()), E->getValue());
  }
  Value *VisitCXXZeroInitValueExpr(const CXXZeroInitValueExpr *E) {
    return llvm::Constant::getNullValue(ConvertType(E->getType()));
  }
  Value *VisitGNUNullExpr(const GNUNullExpr *E) {
    return llvm::Constant::getNullValue(ConvertType(E->getType()));
  }
  Value *VisitTypesCompatibleExpr(const TypesCompatibleExpr *E) {
    return llvm::ConstantInt::get(ConvertType(E->getType()),
                                  CGF.getContext().typesAreCompatible(
                                    E->getArgType1(), E->getArgType2()));
  }
  Value *VisitSizeOfAlignOfExpr(const SizeOfAlignOfExpr *E);
  Value *VisitAddrLabelExpr(const AddrLabelExpr *E) {
    llvm::Value *V = CGF.GetAddrOfLabel(E->getLabel());
    return Builder.CreateBitCast(V, ConvertType(E->getType()));
  }

  // l-values.
  Value *VisitDeclRefExpr(DeclRefExpr *E) {
    Expr::EvalResult Result;
    if (E->Evaluate(Result, CGF.getContext()) && Result.Val.isInt()) {
      assert(!Result.HasSideEffects && "Constant declref with side-effect?!");
      return llvm::ConstantInt::get(VMContext, Result.Val.getInt());
    }
    return EmitLoadOfLValue(E);
  }
  Value *VisitObjCSelectorExpr(ObjCSelectorExpr *E) {
    return CGF.EmitObjCSelectorExpr(E);
  }
  Value *VisitObjCProtocolExpr(ObjCProtocolExpr *E) {
    return CGF.EmitObjCProtocolExpr(E);
  }
  Value *VisitObjCIvarRefExpr(ObjCIvarRefExpr *E) {
    return EmitLoadOfLValue(E);
  }
  Value *VisitObjCPropertyRefExpr(ObjCPropertyRefExpr *E) {
    return EmitLoadOfLValue(E);
  }
  Value *VisitObjCImplicitSetterGetterRefExpr(
                        ObjCImplicitSetterGetterRefExpr *E) {
    return EmitLoadOfLValue(E);
  }
  Value *VisitObjCMessageExpr(ObjCMessageExpr *E) {
    return CGF.EmitObjCMessageExpr(E).getScalarVal();
  }

  Value *VisitObjCIsaExpr(ObjCIsaExpr *E) {
    LValue LV = CGF.EmitObjCIsaExpr(E);
    Value *V = CGF.EmitLoadOfLValue(LV, E->getType()).getScalarVal();
    return V;
  }

  Value *VisitArraySubscriptExpr(ArraySubscriptExpr *E);
  Value *VisitShuffleVectorExpr(ShuffleVectorExpr *E);
  Value *VisitMemberExpr(MemberExpr *E);
  Value *VisitExtVectorElementExpr(Expr *E) { return EmitLoadOfLValue(E); }
  Value *VisitCompoundLiteralExpr(CompoundLiteralExpr *E) {
    return EmitLoadOfLValue(E);
  }
  Value *VisitStringLiteral(Expr *E)  { return EmitLValue(E).getAddress(); }
  Value *VisitObjCEncodeExpr(const ObjCEncodeExpr *E) {
     return EmitLValue(E).getAddress();
  }

  Value *VisitPredefinedExpr(Expr *E) { return EmitLValue(E).getAddress(); }

  Value *VisitInitListExpr(InitListExpr *E);

  Value *VisitImplicitValueInitExpr(const ImplicitValueInitExpr *E) {
    return llvm::Constant::getNullValue(ConvertType(E->getType()));
  }
  Value *VisitCastExpr(CastExpr *E) {
    // Make sure to evaluate VLA bounds now so that we have them for later.
    if (E->getType()->isVariablyModifiedType())
      CGF.EmitVLASize(E->getType());

    return EmitCastExpr(E);
  }
  Value *EmitCastExpr(CastExpr *E);

  Value *VisitCallExpr(const CallExpr *E) {
    if (E->getCallReturnType()->isReferenceType())
      return EmitLoadOfLValue(E);

    return CGF.EmitCallExpr(E).getScalarVal();
  }

  Value *VisitStmtExpr(const StmtExpr *E);

  Value *VisitBlockDeclRefExpr(const BlockDeclRefExpr *E);

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
    // This differs from gcc, though, most likely due to a bug in gcc.
    TestAndClearIgnoreResultAssign();
    return Visit(E->getSubExpr());
  }
  Value *VisitUnaryMinus    (const UnaryOperator *E);
  Value *VisitUnaryNot      (const UnaryOperator *E);
  Value *VisitUnaryLNot     (const UnaryOperator *E);
  Value *VisitUnaryReal     (const UnaryOperator *E);
  Value *VisitUnaryImag     (const UnaryOperator *E);
  Value *VisitUnaryExtension(const UnaryOperator *E) {
    return Visit(E->getSubExpr());
  }
  Value *VisitUnaryOffsetOf(const UnaryOperator *E);

  // C++
  Value *VisitCXXDefaultArgExpr(CXXDefaultArgExpr *DAE) {
    return Visit(DAE->getExpr());
  }
  Value *VisitCXXThisExpr(CXXThisExpr *TE) {
    return CGF.LoadCXXThis();
  }

  Value *VisitCXXExprWithTemporaries(CXXExprWithTemporaries *E) {
    return CGF.EmitCXXExprWithTemporaries(E).getScalarVal();
  }
  Value *VisitCXXNewExpr(const CXXNewExpr *E) {
    return CGF.EmitCXXNewExpr(E);
  }
  Value *VisitCXXDeleteExpr(const CXXDeleteExpr *E) {
    CGF.EmitCXXDeleteExpr(E);
    return 0;
  }
  Value *VisitUnaryTypeTraitExpr(const UnaryTypeTraitExpr *E) {
    return llvm::ConstantInt::get(Builder.getInt1Ty(),
                                  E->EvaluateTrait(CGF.getContext()));
  }

  Value *VisitCXXPseudoDestructorExpr(const CXXPseudoDestructorExpr *E) {
    // C++ [expr.pseudo]p1:
    //   The result shall only be used as the operand for the function call
    //   operator (), and the result of such a call has type void. The only
    //   effect is the evaluation of the postfix-expression before the dot or
    //   arrow.
    CGF.EmitScalarExpr(E->getBase());
    return 0;
  }

  Value *VisitCXXNullPtrLiteralExpr(const CXXNullPtrLiteralExpr *E) {
    return llvm::Constant::getNullValue(ConvertType(E->getType()));
  }

  Value *VisitCXXThrowExpr(const CXXThrowExpr *E) {
    CGF.EmitCXXThrowExpr(E);
    return 0;
  }

  // Binary Operators.
  Value *EmitMul(const BinOpInfo &Ops) {
    if (CGF.getContext().getLangOptions().OverflowChecking
        && Ops.Ty->isSignedIntegerType())
      return EmitOverflowCheckedBinOp(Ops);
    if (Ops.LHS->getType()->isFPOrFPVector())
      return Builder.CreateFMul(Ops.LHS, Ops.RHS, "mul");
    return Builder.CreateMul(Ops.LHS, Ops.RHS, "mul");
  }
  /// Create a binary op that checks for overflow.
  /// Currently only supports +, - and *.
  Value *EmitOverflowCheckedBinOp(const BinOpInfo &Ops);
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
  Value *EmitCompoundAssign(const CompoundAssignOperator *E,
                            Value *(ScalarExprEmitter::*F)(const BinOpInfo &));

  // Binary operators and binary compound assignment operators.
#define HANDLEBINOP(OP) \
  Value *VisitBin ## OP(const BinaryOperator *E) {                         \
    return Emit ## OP(EmitBinOps(E));                                      \
  }                                                                        \
  Value *VisitBin ## OP ## Assign(const CompoundAssignOperator *E) {       \
    return EmitCompoundAssign(E, &ScalarExprEmitter::Emit ## OP);          \
  }
  HANDLEBINOP(Mul);
  HANDLEBINOP(Div);
  HANDLEBINOP(Rem);
  HANDLEBINOP(Add);
  HANDLEBINOP(Sub);
  HANDLEBINOP(Shl);
  HANDLEBINOP(Shr);
  HANDLEBINOP(And);
  HANDLEBINOP(Xor);
  HANDLEBINOP(Or);
#undef HANDLEBINOP

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
  Value *VisitBinComma      (const BinaryOperator *E);

  Value *VisitBinPtrMemD(const Expr *E) { return EmitLoadOfLValue(E); }
  Value *VisitBinPtrMemI(const Expr *E) { return EmitLoadOfLValue(E); }

  // Other Operators.
  Value *VisitBlockExpr(const BlockExpr *BE);
  Value *VisitConditionalOperator(const ConditionalOperator *CO);
  Value *VisitChooseExpr(ChooseExpr *CE);
  Value *VisitVAArgExpr(VAArgExpr *VE);
  Value *VisitObjCStringLiteral(const ObjCStringLiteral *E) {
    return CGF.EmitObjCStringLiteral(E);
  }
};
}  // end anonymous namespace.

//===----------------------------------------------------------------------===//
//                                Utilities
//===----------------------------------------------------------------------===//

/// EmitConversionToBool - Convert the specified expression value to a
/// boolean (i1) truth value.  This is equivalent to "Val != 0".
Value *ScalarExprEmitter::EmitConversionToBool(Value *Src, QualType SrcType) {
  assert(SrcType.isCanonical() && "EmitScalarConversion strips typedefs");

  if (SrcType->isRealFloatingType()) {
    // Compare against 0.0 for fp scalars.
    llvm::Value *Zero = llvm::Constant::getNullValue(Src->getType());
    return Builder.CreateFCmpUNE(Src, Zero, "tobool");
  }

  if (SrcType->isMemberPointerType()) {
    // FIXME: This is ABI specific.

    // Compare against -1.
    llvm::Value *NegativeOne = llvm::Constant::getAllOnesValue(Src->getType());
    return Builder.CreateICmpNE(Src, NegativeOne, "tobool");
  }

  assert((SrcType->isIntegerType() || isa<llvm::PointerType>(Src->getType())) &&
         "Unknown scalar type to convert");

  // Because of the type rules of C, we often end up computing a logical value,
  // then zero extending it to int, then wanting it as a logical value again.
  // Optimize this common case.
  if (llvm::ZExtInst *ZI = dyn_cast<llvm::ZExtInst>(Src)) {
    if (ZI->getOperand(0)->getType() ==
        llvm::Type::getInt1Ty(CGF.getLLVMContext())) {
      Value *Result = ZI->getOperand(0);
      // If there aren't any more uses, zap the instruction to save space.
      // Note that there can be more uses, for example if this
      // is the result of an assignment.
      if (ZI->use_empty())
        ZI->eraseFromParent();
      return Result;
    }
  }

  // Compare against an integer or pointer null.
  llvm::Value *Zero = llvm::Constant::getNullValue(Src->getType());
  return Builder.CreateICmpNE(Src, Zero, "tobool");
}

/// EmitScalarConversion - Emit a conversion from the specified type to the
/// specified destination type, both of which are LLVM scalar types.
Value *ScalarExprEmitter::EmitScalarConversion(Value *Src, QualType SrcType,
                                               QualType DstType) {
  SrcType = CGF.getContext().getCanonicalType(SrcType);
  DstType = CGF.getContext().getCanonicalType(DstType);
  if (SrcType == DstType) return Src;

  if (DstType->isVoidType()) return 0;

  llvm::LLVMContext &VMContext = CGF.getLLVMContext();

  // Handle conversions to bool first, they are special: comparisons against 0.
  if (DstType->isBooleanType())
    return EmitConversionToBool(Src, SrcType);

  const llvm::Type *DstTy = ConvertType(DstType);

  // Ignore conversions like int -> uint.
  if (Src->getType() == DstTy)
    return Src;

  // Handle pointer conversions next: pointers can only be converted to/from
  // other pointers and integers. Check for pointer types in terms of LLVM, as
  // some native types (like Obj-C id) may map to a pointer type.
  if (isa<llvm::PointerType>(DstTy)) {
    // The source value may be an integer, or a pointer.
    if (isa<llvm::PointerType>(Src->getType()))
      return Builder.CreateBitCast(Src, DstTy, "conv");

    assert(SrcType->isIntegerType() && "Not ptr->ptr or int->ptr conversion?");
    // First, convert to the correct width so that we control the kind of
    // extension.
    const llvm::Type *MiddleTy =
          llvm::IntegerType::get(VMContext, CGF.LLVMPointerWidth);
    bool InputSigned = SrcType->isSignedIntegerType();
    llvm::Value* IntResult =
        Builder.CreateIntCast(Src, MiddleTy, InputSigned, "conv");
    // Then, cast to pointer.
    return Builder.CreateIntToPtr(IntResult, DstTy, "conv");
  }

  if (isa<llvm::PointerType>(Src->getType())) {
    // Must be an ptr to int cast.
    assert(isa<llvm::IntegerType>(DstTy) && "not ptr->int?");
    return Builder.CreatePtrToInt(Src, DstTy, "conv");
  }

  // A scalar can be splatted to an extended vector of the same element type
  if (DstType->isExtVectorType() && !SrcType->isVectorType()) {
    // Cast the scalar to element type
    QualType EltTy = DstType->getAs<ExtVectorType>()->getElementType();
    llvm::Value *Elt = EmitScalarConversion(Src, SrcType, EltTy);

    // Insert the element in element zero of an undef vector
    llvm::Value *UnV = llvm::UndefValue::get(DstTy);
    llvm::Value *Idx =
        llvm::ConstantInt::get(llvm::Type::getInt32Ty(VMContext), 0);
    UnV = Builder.CreateInsertElement(UnV, Elt, Idx, "tmp");

    // Splat the element across to all elements
    llvm::SmallVector<llvm::Constant*, 16> Args;
    unsigned NumElements = cast<llvm::VectorType>(DstTy)->getNumElements();
    for (unsigned i = 0; i < NumElements; i++)
      Args.push_back(llvm::ConstantInt::get(
                                        llvm::Type::getInt32Ty(VMContext), 0));

    llvm::Constant *Mask = llvm::ConstantVector::get(&Args[0], NumElements);
    llvm::Value *Yay = Builder.CreateShuffleVector(UnV, UnV, Mask, "splat");
    return Yay;
  }

  // Allow bitcast from vector to integer/fp of the same size.
  if (isa<llvm::VectorType>(Src->getType()) ||
      isa<llvm::VectorType>(DstTy))
    return Builder.CreateBitCast(Src, DstTy, "conv");

  // Finally, we have the arithmetic types: real int/float.
  if (isa<llvm::IntegerType>(Src->getType())) {
    bool InputSigned = SrcType->isSignedIntegerType();
    if (isa<llvm::IntegerType>(DstTy))
      return Builder.CreateIntCast(Src, DstTy, InputSigned, "conv");
    else if (InputSigned)
      return Builder.CreateSIToFP(Src, DstTy, "conv");
    else
      return Builder.CreateUIToFP(Src, DstTy, "conv");
  }

  assert(Src->getType()->isFloatingPoint() && "Unknown real conversion");
  if (isa<llvm::IntegerType>(DstTy)) {
    if (DstType->isSignedIntegerType())
      return Builder.CreateFPToSI(Src, DstTy, "conv");
    else
      return Builder.CreateFPToUI(Src, DstTy, "conv");
  }

  assert(DstTy->isFloatingPoint() && "Unknown real conversion");
  if (DstTy->getTypeID() < Src->getType()->getTypeID())
    return Builder.CreateFPTrunc(Src, DstTy, "conv");
  else
    return Builder.CreateFPExt(Src, DstTy, "conv");
}

/// EmitComplexToScalarConversion - Emit a conversion from the specified complex
/// type to the specified destination type, where the destination type is an
/// LLVM scalar type.
Value *ScalarExprEmitter::
EmitComplexToScalarConversion(CodeGenFunction::ComplexPairTy Src,
                              QualType SrcTy, QualType DstTy) {
  // Get the source element type.
  SrcTy = SrcTy->getAs<ComplexType>()->getElementType();

  // Handle conversions to bool first, they are special: comparisons against 0.
  if (DstTy->isBooleanType()) {
    //  Complex != 0  -> (Real != 0) | (Imag != 0)
    Src.first  = EmitScalarConversion(Src.first, SrcTy, DstTy);
    Src.second = EmitScalarConversion(Src.second, SrcTy, DstTy);
    return Builder.CreateOr(Src.first, Src.second, "tobool");
  }

  // C99 6.3.1.7p2: "When a value of complex type is converted to a real type,
  // the imaginary part of the complex value is discarded and the value of the
  // real part is converted according to the conversion rules for the
  // corresponding real type.
  return EmitScalarConversion(Src.first, SrcTy, DstTy);
}


//===----------------------------------------------------------------------===//
//                            Visitor Methods
//===----------------------------------------------------------------------===//

Value *ScalarExprEmitter::VisitExpr(Expr *E) {
  CGF.ErrorUnsupported(E, "scalar expression");
  if (E->getType()->isVoidType())
    return 0;
  return llvm::UndefValue::get(CGF.ConvertType(E->getType()));
}

Value *ScalarExprEmitter::VisitShuffleVectorExpr(ShuffleVectorExpr *E) {
  llvm::SmallVector<llvm::Constant*, 32> indices;
  for (unsigned i = 2; i < E->getNumSubExprs(); i++) {
    indices.push_back(cast<llvm::Constant>(CGF.EmitScalarExpr(E->getExpr(i))));
  }
  Value* V1 = CGF.EmitScalarExpr(E->getExpr(0));
  Value* V2 = CGF.EmitScalarExpr(E->getExpr(1));
  Value* SV = llvm::ConstantVector::get(indices.begin(), indices.size());
  return Builder.CreateShuffleVector(V1, V2, SV, "shuffle");
}
Value *ScalarExprEmitter::VisitMemberExpr(MemberExpr *E) {
  Expr::EvalResult Result;
  if (E->Evaluate(Result, CGF.getContext()) && Result.Val.isInt()) {
    if (E->isArrow())
      CGF.EmitScalarExpr(E->getBase());
    else
      EmitLValue(E->getBase());
    return llvm::ConstantInt::get(VMContext, Result.Val.getInt());
  }
  return EmitLoadOfLValue(E);
}

Value *ScalarExprEmitter::VisitArraySubscriptExpr(ArraySubscriptExpr *E) {
  TestAndClearIgnoreResultAssign();

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
  bool IdxSigned = E->getIdx()->getType()->isSignedIntegerType();
  Idx = Builder.CreateIntCast(Idx,
                              llvm::Type::getInt32Ty(CGF.getLLVMContext()),
                              IdxSigned,
                              "vecidxcast");
  return Builder.CreateExtractElement(Base, Idx, "vecext");
}

static llvm::Constant *getMaskElt(llvm::ShuffleVectorInst *SVI, unsigned Idx,
                                  unsigned Off, const llvm::Type *I32Ty) {
  int MV = SVI->getMaskValue(Idx);
  if (MV == -1) 
    return llvm::UndefValue::get(I32Ty);
  return llvm::ConstantInt::get(I32Ty, Off+MV);
}

Value *ScalarExprEmitter::VisitInitListExpr(InitListExpr *E) {
  bool Ignore = TestAndClearIgnoreResultAssign();
  (void)Ignore;
  assert (Ignore == false && "init list ignored");
  unsigned NumInitElements = E->getNumInits();
  
  if (E->hadArrayRangeDesignator())
    CGF.ErrorUnsupported(E, "GNU array range designator extension");
  
  const llvm::VectorType *VType =
    dyn_cast<llvm::VectorType>(ConvertType(E->getType()));
  
  // We have a scalar in braces. Just use the first element.
  if (!VType)
    return Visit(E->getInit(0));
  
  unsigned ResElts = VType->getNumElements();
  const llvm::Type *I32Ty = llvm::Type::getInt32Ty(CGF.getLLVMContext());
  
  // Loop over initializers collecting the Value for each, and remembering 
  // whether the source was swizzle (ExtVectorElementExpr).  This will allow
  // us to fold the shuffle for the swizzle into the shuffle for the vector
  // initializer, since LLVM optimizers generally do not want to touch
  // shuffles.
  unsigned CurIdx = 0;
  bool VIsUndefShuffle = false;
  llvm::Value *V = llvm::UndefValue::get(VType);
  for (unsigned i = 0; i != NumInitElements; ++i) {
    Expr *IE = E->getInit(i);
    Value *Init = Visit(IE);
    llvm::SmallVector<llvm::Constant*, 16> Args;
    
    const llvm::VectorType *VVT = dyn_cast<llvm::VectorType>(Init->getType());
    
    // Handle scalar elements.  If the scalar initializer is actually one
    // element of a different vector of the same width, use shuffle instead of 
    // extract+insert.
    if (!VVT) {
      if (isa<ExtVectorElementExpr>(IE)) {
        llvm::ExtractElementInst *EI = cast<llvm::ExtractElementInst>(Init);

        if (EI->getVectorOperandType()->getNumElements() == ResElts) {
          llvm::ConstantInt *C = cast<llvm::ConstantInt>(EI->getIndexOperand());
          Value *LHS = 0, *RHS = 0;
          if (CurIdx == 0) {
            // insert into undef -> shuffle (src, undef)
            Args.push_back(C);
            for (unsigned j = 1; j != ResElts; ++j)
              Args.push_back(llvm::UndefValue::get(I32Ty));

            LHS = EI->getVectorOperand();
            RHS = V;
            VIsUndefShuffle = true;
          } else if (VIsUndefShuffle) {
            // insert into undefshuffle && size match -> shuffle (v, src)
            llvm::ShuffleVectorInst *SVV = cast<llvm::ShuffleVectorInst>(V);
            for (unsigned j = 0; j != CurIdx; ++j)
              Args.push_back(getMaskElt(SVV, j, 0, I32Ty));
            Args.push_back(llvm::ConstantInt::get(I32Ty, 
                                                  ResElts + C->getZExtValue()));
            for (unsigned j = CurIdx + 1; j != ResElts; ++j)
              Args.push_back(llvm::UndefValue::get(I32Ty));
            
            LHS = cast<llvm::ShuffleVectorInst>(V)->getOperand(0);
            RHS = EI->getVectorOperand();
            VIsUndefShuffle = false;
          }
          if (!Args.empty()) {
            llvm::Constant *Mask = llvm::ConstantVector::get(&Args[0], ResElts);
            V = Builder.CreateShuffleVector(LHS, RHS, Mask);
            ++CurIdx;
            continue;
          }
        }
      }
      Value *Idx = llvm::ConstantInt::get(I32Ty, CurIdx);
      V = Builder.CreateInsertElement(V, Init, Idx, "vecinit");
      VIsUndefShuffle = false;
      ++CurIdx;
      continue;
    }
    
    unsigned InitElts = VVT->getNumElements();

    // If the initializer is an ExtVecEltExpr (a swizzle), and the swizzle's 
    // input is the same width as the vector being constructed, generate an
    // optimized shuffle of the swizzle input into the result.
    unsigned Offset = (CurIdx == 0) ? 0 : ResElts;
    if (isa<ExtVectorElementExpr>(IE)) {
      llvm::ShuffleVectorInst *SVI = cast<llvm::ShuffleVectorInst>(Init);
      Value *SVOp = SVI->getOperand(0);
      const llvm::VectorType *OpTy = cast<llvm::VectorType>(SVOp->getType());
      
      if (OpTy->getNumElements() == ResElts) {
        for (unsigned j = 0; j != CurIdx; ++j) {
          // If the current vector initializer is a shuffle with undef, merge
          // this shuffle directly into it.
          if (VIsUndefShuffle) {
            Args.push_back(getMaskElt(cast<llvm::ShuffleVectorInst>(V), j, 0,
                                      I32Ty));
          } else {
            Args.push_back(llvm::ConstantInt::get(I32Ty, j));
          }
        }
        for (unsigned j = 0, je = InitElts; j != je; ++j)
          Args.push_back(getMaskElt(SVI, j, Offset, I32Ty));
        for (unsigned j = CurIdx + InitElts; j != ResElts; ++j)
          Args.push_back(llvm::UndefValue::get(I32Ty));

        if (VIsUndefShuffle)
          V = cast<llvm::ShuffleVectorInst>(V)->getOperand(0);

        Init = SVOp;
      }
    }

    // Extend init to result vector length, and then shuffle its contribution
    // to the vector initializer into V.
    if (Args.empty()) {
      for (unsigned j = 0; j != InitElts; ++j)
        Args.push_back(llvm::ConstantInt::get(I32Ty, j));
      for (unsigned j = InitElts; j != ResElts; ++j)
        Args.push_back(llvm::UndefValue::get(I32Ty));
      llvm::Constant *Mask = llvm::ConstantVector::get(&Args[0], ResElts);
      Init = Builder.CreateShuffleVector(Init, llvm::UndefValue::get(VVT),
                                         Mask, "vext");

      Args.clear();
      for (unsigned j = 0; j != CurIdx; ++j)
        Args.push_back(llvm::ConstantInt::get(I32Ty, j));
      for (unsigned j = 0; j != InitElts; ++j)
        Args.push_back(llvm::ConstantInt::get(I32Ty, j+Offset));
      for (unsigned j = CurIdx + InitElts; j != ResElts; ++j)
        Args.push_back(llvm::UndefValue::get(I32Ty));
    }

    // If V is undef, make sure it ends up on the RHS of the shuffle to aid
    // merging subsequent shuffles into this one.
    if (CurIdx == 0)
      std::swap(V, Init);
    llvm::Constant *Mask = llvm::ConstantVector::get(&Args[0], ResElts);
    V = Builder.CreateShuffleVector(V, Init, Mask, "vecinit");
    VIsUndefShuffle = isa<llvm::UndefValue>(Init);
    CurIdx += InitElts;
  }
  
  // FIXME: evaluate codegen vs. shuffling against constant null vector.
  // Emit remaining default initializers.
  const llvm::Type *EltTy = VType->getElementType();
  
  // Emit remaining default initializers
  for (/* Do not initialize i*/; CurIdx < ResElts; ++CurIdx) {
    Value *Idx = llvm::ConstantInt::get(I32Ty, CurIdx);
    llvm::Value *Init = llvm::Constant::getNullValue(EltTy);
    V = Builder.CreateInsertElement(V, Init, Idx, "vecinit");
  }
  return V;
}

static bool ShouldNullCheckClassCastValue(const CastExpr *CE) {
  const Expr *E = CE->getSubExpr();
  
  if (isa<CXXThisExpr>(E)) {
    // We always assume that 'this' is never null.
    return false;
  }
  
  if (const ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(CE)) {
    // And that lvalue casts are never null.
    if (ICE->isLvalueCast())
      return false;
  }

  return true;
}

// VisitCastExpr - Emit code for an explicit or implicit cast.  Implicit casts
// have to handle a more broad range of conversions than explicit casts, as they
// handle things like function to ptr-to-function decay etc.
Value *ScalarExprEmitter::EmitCastExpr(CastExpr *CE) {
  Expr *E = CE->getSubExpr();
  QualType DestTy = CE->getType();
  CastExpr::CastKind Kind = CE->getCastKind();
  
  if (!DestTy->isVoidType())
    TestAndClearIgnoreResultAssign();

  // Since almost all cast kinds apply to scalars, this switch doesn't have
  // a default case, so the compiler will warn on a missing case.  The cases
  // are in the same order as in the CastKind enum.
  switch (Kind) {
  case CastExpr::CK_Unknown:
    // FIXME: All casts should have a known kind!
    //assert(0 && "Unknown cast kind!");
    break;

  case CastExpr::CK_AnyPointerToObjCPointerCast:
  case CastExpr::CK_AnyPointerToBlockPointerCast:
  case CastExpr::CK_BitCast: {
    Value *Src = Visit(const_cast<Expr*>(E));
    return Builder.CreateBitCast(Src, ConvertType(DestTy));
  }
  case CastExpr::CK_NoOp:
    return Visit(const_cast<Expr*>(E));

  case CastExpr::CK_BaseToDerived: {
    const CXXRecordDecl *BaseClassDecl = 
      E->getType()->getCXXRecordDeclForPointerType();
    const CXXRecordDecl *DerivedClassDecl = 
      DestTy->getCXXRecordDeclForPointerType();
    
    Value *Src = Visit(const_cast<Expr*>(E));
    
    bool NullCheckValue = ShouldNullCheckClassCastValue(CE);
    return CGF.GetAddressOfDerivedClass(Src, BaseClassDecl, DerivedClassDecl, 
                                        NullCheckValue);
  }
  case CastExpr::CK_DerivedToBase: {
    const RecordType *DerivedClassTy = 
      E->getType()->getAs<PointerType>()->getPointeeType()->getAs<RecordType>();
    CXXRecordDecl *DerivedClassDecl = 
      cast<CXXRecordDecl>(DerivedClassTy->getDecl());

    const RecordType *BaseClassTy = 
      DestTy->getAs<PointerType>()->getPointeeType()->getAs<RecordType>();
    CXXRecordDecl *BaseClassDecl = cast<CXXRecordDecl>(BaseClassTy->getDecl());
    
    Value *Src = Visit(const_cast<Expr*>(E));

    bool NullCheckValue = ShouldNullCheckClassCastValue(CE);
    return CGF.GetAddressOfBaseClass(Src, DerivedClassDecl, BaseClassDecl,
                                     NullCheckValue);
  }
  case CastExpr::CK_Dynamic: {
    Value *V = Visit(const_cast<Expr*>(E));
    const CXXDynamicCastExpr *DCE = cast<CXXDynamicCastExpr>(CE);
    return CGF.EmitDynamicCast(V, DCE);
  }
  case CastExpr::CK_ToUnion:
    assert(0 && "Should be unreachable!");
    break;

  case CastExpr::CK_ArrayToPointerDecay: {
    assert(E->getType()->isArrayType() &&
           "Array to pointer decay must have array source type!");

    Value *V = EmitLValue(E).getAddress();  // Bitfields can't be arrays.

    // Note that VLA pointers are always decayed, so we don't need to do
    // anything here.
    if (!E->getType()->isVariableArrayType()) {
      assert(isa<llvm::PointerType>(V->getType()) && "Expected pointer");
      assert(isa<llvm::ArrayType>(cast<llvm::PointerType>(V->getType())
                                 ->getElementType()) &&
             "Expected pointer to array");
      V = Builder.CreateStructGEP(V, 0, "arraydecay");
    }

    return V;
  }
  case CastExpr::CK_FunctionToPointerDecay:
    return EmitLValue(E).getAddress();

  case CastExpr::CK_NullToMemberPointer:
    return CGF.CGM.EmitNullConstant(DestTy);

  case CastExpr::CK_BaseToDerivedMemberPointer:
  case CastExpr::CK_DerivedToBaseMemberPointer: {
    Value *Src = Visit(E);

    // See if we need to adjust the pointer.
    const CXXRecordDecl *BaseDecl = 
      cast<CXXRecordDecl>(E->getType()->getAs<MemberPointerType>()->
                          getClass()->getAs<RecordType>()->getDecl());
    const CXXRecordDecl *DerivedDecl = 
      cast<CXXRecordDecl>(CE->getType()->getAs<MemberPointerType>()->
                          getClass()->getAs<RecordType>()->getDecl());
    if (CE->getCastKind() == CastExpr::CK_DerivedToBaseMemberPointer)
      std::swap(DerivedDecl, BaseDecl);

    llvm::Constant *Adj = CGF.CGM.GetCXXBaseClassOffset(DerivedDecl, BaseDecl);
    if (Adj) {
      if (CE->getCastKind() == CastExpr::CK_DerivedToBaseMemberPointer)
        Src = Builder.CreateSub(Src, Adj, "adj");
      else
        Src = Builder.CreateAdd(Src, Adj, "adj");
    }
    return Src;
  }

  case CastExpr::CK_UserDefinedConversion:
  case CastExpr::CK_ConstructorConversion:
    assert(0 && "Should be unreachable!");
    break;

  case CastExpr::CK_IntegralToPointer: {
    Value *Src = Visit(const_cast<Expr*>(E));
    
    // First, convert to the correct width so that we control the kind of
    // extension.
    const llvm::Type *MiddleTy =
      llvm::IntegerType::get(VMContext, CGF.LLVMPointerWidth);
    bool InputSigned = E->getType()->isSignedIntegerType();
    llvm::Value* IntResult =
      Builder.CreateIntCast(Src, MiddleTy, InputSigned, "conv");
    
    return Builder.CreateIntToPtr(IntResult, ConvertType(DestTy));
  }
  case CastExpr::CK_PointerToIntegral: {
    Value *Src = Visit(const_cast<Expr*>(E));
    return Builder.CreatePtrToInt(Src, ConvertType(DestTy));
  }
  case CastExpr::CK_ToVoid: {
    CGF.EmitAnyExpr(E, 0, false, true);
    return 0;
  }
  case CastExpr::CK_VectorSplat: {
    const llvm::Type *DstTy = ConvertType(DestTy);
    Value *Elt = Visit(const_cast<Expr*>(E));

    // Insert the element in element zero of an undef vector
    llvm::Value *UnV = llvm::UndefValue::get(DstTy);
    llvm::Value *Idx =
        llvm::ConstantInt::get(llvm::Type::getInt32Ty(VMContext), 0);
    UnV = Builder.CreateInsertElement(UnV, Elt, Idx, "tmp");

    // Splat the element across to all elements
    llvm::SmallVector<llvm::Constant*, 16> Args;
    unsigned NumElements = cast<llvm::VectorType>(DstTy)->getNumElements();
    for (unsigned i = 0; i < NumElements; i++)
      Args.push_back(llvm::ConstantInt::get(
                                        llvm::Type::getInt32Ty(VMContext), 0));

    llvm::Constant *Mask = llvm::ConstantVector::get(&Args[0], NumElements);
    llvm::Value *Yay = Builder.CreateShuffleVector(UnV, UnV, Mask, "splat");
    return Yay;
  }
  case CastExpr::CK_IntegralCast:
  case CastExpr::CK_IntegralToFloating:
  case CastExpr::CK_FloatingToIntegral:
  case CastExpr::CK_FloatingCast:
    return EmitScalarConversion(Visit(E), E->getType(), DestTy);

  case CastExpr::CK_MemberPointerToBoolean:
    return CGF.EvaluateExprAsBool(E);
  }

  // Handle cases where the source is an non-complex type.

  if (!CGF.hasAggregateLLVMType(E->getType())) {
    Value *Src = Visit(const_cast<Expr*>(E));

    // Use EmitScalarConversion to perform the conversion.
    return EmitScalarConversion(Src, E->getType(), DestTy);
  }

  if (E->getType()->isAnyComplexType()) {
    // Handle cases where the source is a complex type.
    bool IgnoreImag = true;
    bool IgnoreImagAssign = true;
    bool IgnoreReal = IgnoreResultAssign;
    bool IgnoreRealAssign = IgnoreResultAssign;
    if (DestTy->isBooleanType())
      IgnoreImagAssign = IgnoreImag = false;
    else if (DestTy->isVoidType()) {
      IgnoreReal = IgnoreImag = false;
      IgnoreRealAssign = IgnoreImagAssign = true;
    }
    CodeGenFunction::ComplexPairTy V
      = CGF.EmitComplexExpr(E, IgnoreReal, IgnoreImag, IgnoreRealAssign,
                            IgnoreImagAssign);
    return EmitComplexToScalarConversion(V, E->getType(), DestTy);
  }

  // Okay, this is a cast from an aggregate.  It must be a cast to void.  Just
  // evaluate the result and return.
  CGF.EmitAggExpr(E, 0, false, true);
  return 0;
}

Value *ScalarExprEmitter::VisitStmtExpr(const StmtExpr *E) {
  return CGF.EmitCompoundStmt(*E->getSubStmt(),
                              !E->getType()->isVoidType()).getScalarVal();
}

Value *ScalarExprEmitter::VisitBlockDeclRefExpr(const BlockDeclRefExpr *E) {
  llvm::Value *V = CGF.GetAddrOfBlockDecl(E);
  if (E->getType().isObjCGCWeak())
    return CGF.CGM.getObjCRuntime().EmitObjCWeakRead(CGF, V);
  return Builder.CreateLoad(V, "tmp");
}

//===----------------------------------------------------------------------===//
//                             Unary Operators
//===----------------------------------------------------------------------===//

Value *ScalarExprEmitter::VisitPrePostIncDec(const UnaryOperator *E,
                                             bool isInc, bool isPre) {
  LValue LV = EmitLValue(E->getSubExpr());
  QualType ValTy = E->getSubExpr()->getType();
  Value *InVal = CGF.EmitLoadOfLValue(LV, ValTy).getScalarVal();

  llvm::LLVMContext &VMContext = CGF.getLLVMContext();

  int AmountVal = isInc ? 1 : -1;

  if (ValTy->isPointerType() &&
      ValTy->getAs<PointerType>()->isVariableArrayType()) {
    // The amount of the addition/subtraction needs to account for the VLA size
    CGF.ErrorUnsupported(E, "VLA pointer inc/dec");
  }

  Value *NextVal;
  if (const llvm::PointerType *PT =
         dyn_cast<llvm::PointerType>(InVal->getType())) {
    llvm::Constant *Inc =
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(VMContext), AmountVal);
    if (!isa<llvm::FunctionType>(PT->getElementType())) {
      QualType PTEE = ValTy->getPointeeType();
      if (const ObjCInterfaceType *OIT =
          dyn_cast<ObjCInterfaceType>(PTEE)) {
        // Handle interface types, which are not represented with a concrete type.
        int size = CGF.getContext().getTypeSize(OIT) / 8;
        if (!isInc)
          size = -size;
        Inc = llvm::ConstantInt::get(Inc->getType(), size);
        const llvm::Type *i8Ty = llvm::Type::getInt8PtrTy(VMContext);
        InVal = Builder.CreateBitCast(InVal, i8Ty);
        NextVal = Builder.CreateGEP(InVal, Inc, "add.ptr");
        llvm::Value *lhs = LV.getAddress();
        lhs = Builder.CreateBitCast(lhs, llvm::PointerType::getUnqual(i8Ty));
        LV = LValue::MakeAddr(lhs, CGF.MakeQualifiers(ValTy));
      } else
        NextVal = Builder.CreateInBoundsGEP(InVal, Inc, "ptrincdec");
    } else {
      const llvm::Type *i8Ty = llvm::Type::getInt8PtrTy(VMContext);
      NextVal = Builder.CreateBitCast(InVal, i8Ty, "tmp");
      NextVal = Builder.CreateGEP(NextVal, Inc, "ptrincdec");
      NextVal = Builder.CreateBitCast(NextVal, InVal->getType());
    }
  } else if (InVal->getType() == llvm::Type::getInt1Ty(VMContext) && isInc) {
    // Bool++ is an interesting case, due to promotion rules, we get:
    // Bool++ -> Bool = Bool+1 -> Bool = (int)Bool+1 ->
    // Bool = ((int)Bool+1) != 0
    // An interesting aspect of this is that increment is always true.
    // Decrement does not have this property.
    NextVal = llvm::ConstantInt::getTrue(VMContext);
  } else if (isa<llvm::IntegerType>(InVal->getType())) {
    NextVal = llvm::ConstantInt::get(InVal->getType(), AmountVal);

    // Signed integer overflow is undefined behavior.
    if (ValTy->isSignedIntegerType())
      NextVal = Builder.CreateNSWAdd(InVal, NextVal, isInc ? "inc" : "dec");
    else
      NextVal = Builder.CreateAdd(InVal, NextVal, isInc ? "inc" : "dec");
  } else {
    // Add the inc/dec to the real part.
    if (InVal->getType()->isFloatTy())
      NextVal =
        llvm::ConstantFP::get(VMContext,
                              llvm::APFloat(static_cast<float>(AmountVal)));
    else if (InVal->getType()->isDoubleTy())
      NextVal =
        llvm::ConstantFP::get(VMContext,
                              llvm::APFloat(static_cast<double>(AmountVal)));
    else {
      llvm::APFloat F(static_cast<float>(AmountVal));
      bool ignored;
      F.convert(CGF.Target.getLongDoubleFormat(), llvm::APFloat::rmTowardZero,
                &ignored);
      NextVal = llvm::ConstantFP::get(VMContext, F);
    }
    NextVal = Builder.CreateFAdd(InVal, NextVal, isInc ? "inc" : "dec");
  }

  // Store the updated result through the lvalue.
  if (LV.isBitfield())
    CGF.EmitStoreThroughBitfieldLValue(RValue::get(NextVal), LV, ValTy,
                                       &NextVal);
  else
    CGF.EmitStoreThroughLValue(RValue::get(NextVal), LV, ValTy);

  // If this is a postinc, return the value read from memory, otherwise use the
  // updated value.
  return isPre ? NextVal : InVal;
}


Value *ScalarExprEmitter::VisitUnaryMinus(const UnaryOperator *E) {
  TestAndClearIgnoreResultAssign();
  Value *Op = Visit(E->getSubExpr());
  if (Op->getType()->isFPOrFPVector())
    return Builder.CreateFNeg(Op, "neg");
  return Builder.CreateNeg(Op, "neg");
}

Value *ScalarExprEmitter::VisitUnaryNot(const UnaryOperator *E) {
  TestAndClearIgnoreResultAssign();
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

  // ZExt result to the expr type.
  return Builder.CreateZExt(BoolVal, ConvertType(E->getType()), "lnot.ext");
}

/// VisitSizeOfAlignOfExpr - Return the size or alignment of the type of
/// argument of the sizeof expression as an integer.
Value *
ScalarExprEmitter::VisitSizeOfAlignOfExpr(const SizeOfAlignOfExpr *E) {
  QualType TypeToSize = E->getTypeOfArgument();
  if (E->isSizeOf()) {
    if (const VariableArrayType *VAT =
          CGF.getContext().getAsVariableArrayType(TypeToSize)) {
      if (E->isArgumentType()) {
        // sizeof(type) - make sure to emit the VLA size.
        CGF.EmitVLASize(TypeToSize);
      } else {
        // C99 6.5.3.4p2: If the argument is an expression of type
        // VLA, it is evaluated.
        CGF.EmitAnyExpr(E->getArgumentExpr());
      }

      return CGF.GetVLASize(VAT);
    }
  }

  // If this isn't sizeof(vla), the result must be constant; use the constant
  // folding logic so we don't have to duplicate it here.
  Expr::EvalResult Result;
  E->Evaluate(Result, CGF.getContext());
  return llvm::ConstantInt::get(VMContext, Result.Val.getInt());
}

Value *ScalarExprEmitter::VisitUnaryReal(const UnaryOperator *E) {
  Expr *Op = E->getSubExpr();
  if (Op->getType()->isAnyComplexType())
    return CGF.EmitComplexExpr(Op, false, true, false, true).first;
  return Visit(Op);
}
Value *ScalarExprEmitter::VisitUnaryImag(const UnaryOperator *E) {
  Expr *Op = E->getSubExpr();
  if (Op->getType()->isAnyComplexType())
    return CGF.EmitComplexExpr(Op, true, false, true, false).second;

  // __imag on a scalar returns zero.  Emit the subexpr to ensure side
  // effects are evaluated, but not the actual value.
  if (E->isLvalue(CGF.getContext()) == Expr::LV_Valid)
    CGF.EmitLValue(Op);
  else
    CGF.EmitScalarExpr(Op, true);
  return llvm::Constant::getNullValue(ConvertType(E->getType()));
}

Value *ScalarExprEmitter::VisitUnaryOffsetOf(const UnaryOperator *E) {
  Value* ResultAsPtr = EmitLValue(E->getSubExpr()).getAddress();
  const llvm::Type* ResultType = ConvertType(E->getType());
  return Builder.CreatePtrToInt(ResultAsPtr, ResultType, "offsetof");
}

//===----------------------------------------------------------------------===//
//                           Binary Operators
//===----------------------------------------------------------------------===//

BinOpInfo ScalarExprEmitter::EmitBinOps(const BinaryOperator *E) {
  TestAndClearIgnoreResultAssign();
  BinOpInfo Result;
  Result.LHS = Visit(E->getLHS());
  Result.RHS = Visit(E->getRHS());
  Result.Ty  = E->getType();
  Result.E = E;
  return Result;
}

Value *ScalarExprEmitter::EmitCompoundAssign(const CompoundAssignOperator *E,
                      Value *(ScalarExprEmitter::*Func)(const BinOpInfo &)) {
  bool Ignore = TestAndClearIgnoreResultAssign();
  QualType LHSTy = E->getLHS()->getType(), RHSTy = E->getRHS()->getType();

  BinOpInfo OpInfo;

  if (E->getComputationResultType()->isAnyComplexType()) {
    // This needs to go through the complex expression emitter, but it's a tad
    // complicated to do that... I'm leaving it out for now.  (Note that we do
    // actually need the imaginary part of the RHS for multiplication and
    // division.)
    CGF.ErrorUnsupported(E, "complex compound assignment");
    return llvm::UndefValue::get(CGF.ConvertType(E->getType()));
  }

  // Emit the RHS first.  __block variables need to have the rhs evaluated
  // first, plus this should improve codegen a little.
  OpInfo.RHS = Visit(E->getRHS());
  OpInfo.Ty = E->getComputationResultType();
  OpInfo.E = E;
  // Load/convert the LHS.
  LValue LHSLV = EmitLValue(E->getLHS());
  OpInfo.LHS = EmitLoadOfLValue(LHSLV, LHSTy);
  OpInfo.LHS = EmitScalarConversion(OpInfo.LHS, LHSTy,
                                    E->getComputationLHSType());

  // Expand the binary operator.
  Value *Result = (this->*Func)(OpInfo);

  // Convert the result back to the LHS type.
  Result = EmitScalarConversion(Result, E->getComputationResultType(), LHSTy);

  // Store the result value into the LHS lvalue. Bit-fields are handled
  // specially because the result is altered by the store, i.e., [C99 6.5.16p1]
  // 'An assignment expression has the value of the left operand after the
  // assignment...'.
  if (LHSLV.isBitfield()) {
    if (!LHSLV.isVolatileQualified()) {
      CGF.EmitStoreThroughBitfieldLValue(RValue::get(Result), LHSLV, LHSTy,
                                         &Result);
      return Result;
    } else
      CGF.EmitStoreThroughBitfieldLValue(RValue::get(Result), LHSLV, LHSTy);
  } else
    CGF.EmitStoreThroughLValue(RValue::get(Result), LHSLV, LHSTy);
  if (Ignore)
    return 0;
  return EmitLoadOfLValue(LHSLV, E->getType());
}


Value *ScalarExprEmitter::EmitDiv(const BinOpInfo &Ops) {
  if (Ops.LHS->getType()->isFPOrFPVector())
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

Value *ScalarExprEmitter::EmitOverflowCheckedBinOp(const BinOpInfo &Ops) {
  unsigned IID;
  unsigned OpID = 0;

  switch (Ops.E->getOpcode()) {
  case BinaryOperator::Add:
  case BinaryOperator::AddAssign:
    OpID = 1;
    IID = llvm::Intrinsic::sadd_with_overflow;
    break;
  case BinaryOperator::Sub:
  case BinaryOperator::SubAssign:
    OpID = 2;
    IID = llvm::Intrinsic::ssub_with_overflow;
    break;
  case BinaryOperator::Mul:
  case BinaryOperator::MulAssign:
    OpID = 3;
    IID = llvm::Intrinsic::smul_with_overflow;
    break;
  default:
    assert(false && "Unsupported operation for overflow detection");
    IID = 0;
  }
  OpID <<= 1;
  OpID |= 1;

  const llvm::Type *opTy = CGF.CGM.getTypes().ConvertType(Ops.Ty);

  llvm::Function *intrinsic = CGF.CGM.getIntrinsic(IID, &opTy, 1);

  Value *resultAndOverflow = Builder.CreateCall2(intrinsic, Ops.LHS, Ops.RHS);
  Value *result = Builder.CreateExtractValue(resultAndOverflow, 0);
  Value *overflow = Builder.CreateExtractValue(resultAndOverflow, 1);

  // Branch in case of overflow.
  llvm::BasicBlock *initialBB = Builder.GetInsertBlock();
  llvm::BasicBlock *overflowBB =
    CGF.createBasicBlock("overflow", CGF.CurFn);
  llvm::BasicBlock *continueBB =
    CGF.createBasicBlock("overflow.continue", CGF.CurFn);

  Builder.CreateCondBr(overflow, overflowBB, continueBB);

  // Handle overflow

  Builder.SetInsertPoint(overflowBB);

  // Handler is:
  // long long *__overflow_handler)(long long a, long long b, char op,
  // char width)
  std::vector<const llvm::Type*> handerArgTypes;
  handerArgTypes.push_back(llvm::Type::getInt64Ty(VMContext));
  handerArgTypes.push_back(llvm::Type::getInt64Ty(VMContext));
  handerArgTypes.push_back(llvm::Type::getInt8Ty(VMContext));
  handerArgTypes.push_back(llvm::Type::getInt8Ty(VMContext));
  llvm::FunctionType *handlerTy = llvm::FunctionType::get(
      llvm::Type::getInt64Ty(VMContext), handerArgTypes, false);
  llvm::Value *handlerFunction =
    CGF.CGM.getModule().getOrInsertGlobal("__overflow_handler",
        llvm::PointerType::getUnqual(handlerTy));
  handlerFunction = Builder.CreateLoad(handlerFunction);

  llvm::Value *handlerResult = Builder.CreateCall4(handlerFunction,
      Builder.CreateSExt(Ops.LHS, llvm::Type::getInt64Ty(VMContext)),
      Builder.CreateSExt(Ops.RHS, llvm::Type::getInt64Ty(VMContext)),
      llvm::ConstantInt::get(llvm::Type::getInt8Ty(VMContext), OpID),
      llvm::ConstantInt::get(llvm::Type::getInt8Ty(VMContext),
        cast<llvm::IntegerType>(opTy)->getBitWidth()));

  handlerResult = Builder.CreateTrunc(handlerResult, opTy);

  Builder.CreateBr(continueBB);

  // Set up the continuation
  Builder.SetInsertPoint(continueBB);
  // Get the correct result
  llvm::PHINode *phi = Builder.CreatePHI(opTy);
  phi->reserveOperandSpace(2);
  phi->addIncoming(result, initialBB);
  phi->addIncoming(handlerResult, overflowBB);

  return phi;
}

Value *ScalarExprEmitter::EmitAdd(const BinOpInfo &Ops) {
  if (!Ops.Ty->isAnyPointerType()) {
    if (CGF.getContext().getLangOptions().OverflowChecking &&
        Ops.Ty->isSignedIntegerType())
      return EmitOverflowCheckedBinOp(Ops);

    if (Ops.LHS->getType()->isFPOrFPVector())
      return Builder.CreateFAdd(Ops.LHS, Ops.RHS, "add");

    // Signed integer overflow is undefined behavior.
    if (Ops.Ty->isSignedIntegerType())
      return Builder.CreateNSWAdd(Ops.LHS, Ops.RHS, "add");

    return Builder.CreateAdd(Ops.LHS, Ops.RHS, "add");
  }

  if (Ops.Ty->isPointerType() &&
      Ops.Ty->getAs<PointerType>()->isVariableArrayType()) {
    // The amount of the addition needs to account for the VLA size
    CGF.ErrorUnsupported(Ops.E, "VLA pointer addition");
  }
  Value *Ptr, *Idx;
  Expr *IdxExp;
  const PointerType *PT = Ops.E->getLHS()->getType()->getAs<PointerType>();
  const ObjCObjectPointerType *OPT =
    Ops.E->getLHS()->getType()->getAs<ObjCObjectPointerType>();
  if (PT || OPT) {
    Ptr = Ops.LHS;
    Idx = Ops.RHS;
    IdxExp = Ops.E->getRHS();
  } else {  // int + pointer
    PT = Ops.E->getRHS()->getType()->getAs<PointerType>();
    OPT = Ops.E->getRHS()->getType()->getAs<ObjCObjectPointerType>();
    assert((PT || OPT) && "Invalid add expr");
    Ptr = Ops.RHS;
    Idx = Ops.LHS;
    IdxExp = Ops.E->getLHS();
  }

  unsigned Width = cast<llvm::IntegerType>(Idx->getType())->getBitWidth();
  if (Width < CGF.LLVMPointerWidth) {
    // Zero or sign extend the pointer value based on whether the index is
    // signed or not.
    const llvm::Type *IdxType =
        llvm::IntegerType::get(VMContext, CGF.LLVMPointerWidth);
    if (IdxExp->getType()->isSignedIntegerType())
      Idx = Builder.CreateSExt(Idx, IdxType, "idx.ext");
    else
      Idx = Builder.CreateZExt(Idx, IdxType, "idx.ext");
  }
  const QualType ElementType = PT ? PT->getPointeeType() : OPT->getPointeeType();
  // Handle interface types, which are not represented with a concrete type.
  if (const ObjCInterfaceType *OIT = dyn_cast<ObjCInterfaceType>(ElementType)) {
    llvm::Value *InterfaceSize =
      llvm::ConstantInt::get(Idx->getType(),
                             CGF.getContext().getTypeSize(OIT) / 8);
    Idx = Builder.CreateMul(Idx, InterfaceSize);
    const llvm::Type *i8Ty = llvm::Type::getInt8PtrTy(VMContext);
    Value *Casted = Builder.CreateBitCast(Ptr, i8Ty);
    Value *Res = Builder.CreateGEP(Casted, Idx, "add.ptr");
    return Builder.CreateBitCast(Res, Ptr->getType());
  }

  // Explicitly handle GNU void* and function pointer arithmetic extensions. The
  // GNU void* casts amount to no-ops since our void* type is i8*, but this is
  // future proof.
  if (ElementType->isVoidType() || ElementType->isFunctionType()) {
    const llvm::Type *i8Ty = llvm::Type::getInt8PtrTy(VMContext);
    Value *Casted = Builder.CreateBitCast(Ptr, i8Ty);
    Value *Res = Builder.CreateGEP(Casted, Idx, "add.ptr");
    return Builder.CreateBitCast(Res, Ptr->getType());
  }

  return Builder.CreateInBoundsGEP(Ptr, Idx, "add.ptr");
}

Value *ScalarExprEmitter::EmitSub(const BinOpInfo &Ops) {
  if (!isa<llvm::PointerType>(Ops.LHS->getType())) {
    if (CGF.getContext().getLangOptions().OverflowChecking
        && Ops.Ty->isSignedIntegerType())
      return EmitOverflowCheckedBinOp(Ops);

    if (Ops.LHS->getType()->isFPOrFPVector())
      return Builder.CreateFSub(Ops.LHS, Ops.RHS, "sub");
    return Builder.CreateSub(Ops.LHS, Ops.RHS, "sub");
  }

  if (Ops.E->getLHS()->getType()->isPointerType() &&
      Ops.E->getLHS()->getType()->getAs<PointerType>()->isVariableArrayType()) {
    // The amount of the addition needs to account for the VLA size for
    // ptr-int
    // The amount of the division needs to account for the VLA size for
    // ptr-ptr.
    CGF.ErrorUnsupported(Ops.E, "VLA pointer subtraction");
  }

  const QualType LHSType = Ops.E->getLHS()->getType();
  const QualType LHSElementType = LHSType->getPointeeType();
  if (!isa<llvm::PointerType>(Ops.RHS->getType())) {
    // pointer - int
    Value *Idx = Ops.RHS;
    unsigned Width = cast<llvm::IntegerType>(Idx->getType())->getBitWidth();
    if (Width < CGF.LLVMPointerWidth) {
      // Zero or sign extend the pointer value based on whether the index is
      // signed or not.
      const llvm::Type *IdxType =
          llvm::IntegerType::get(VMContext, CGF.LLVMPointerWidth);
      if (Ops.E->getRHS()->getType()->isSignedIntegerType())
        Idx = Builder.CreateSExt(Idx, IdxType, "idx.ext");
      else
        Idx = Builder.CreateZExt(Idx, IdxType, "idx.ext");
    }
    Idx = Builder.CreateNeg(Idx, "sub.ptr.neg");

    // Handle interface types, which are not represented with a concrete type.
    if (const ObjCInterfaceType *OIT =
        dyn_cast<ObjCInterfaceType>(LHSElementType)) {
      llvm::Value *InterfaceSize =
        llvm::ConstantInt::get(Idx->getType(),
                               CGF.getContext().getTypeSize(OIT) / 8);
      Idx = Builder.CreateMul(Idx, InterfaceSize);
      const llvm::Type *i8Ty = llvm::Type::getInt8PtrTy(VMContext);
      Value *LHSCasted = Builder.CreateBitCast(Ops.LHS, i8Ty);
      Value *Res = Builder.CreateGEP(LHSCasted, Idx, "add.ptr");
      return Builder.CreateBitCast(Res, Ops.LHS->getType());
    }

    // Explicitly handle GNU void* and function pointer arithmetic
    // extensions. The GNU void* casts amount to no-ops since our void* type is
    // i8*, but this is future proof.
    if (LHSElementType->isVoidType() || LHSElementType->isFunctionType()) {
      const llvm::Type *i8Ty = llvm::Type::getInt8PtrTy(VMContext);
      Value *LHSCasted = Builder.CreateBitCast(Ops.LHS, i8Ty);
      Value *Res = Builder.CreateGEP(LHSCasted, Idx, "sub.ptr");
      return Builder.CreateBitCast(Res, Ops.LHS->getType());
    }

    return Builder.CreateInBoundsGEP(Ops.LHS, Idx, "sub.ptr");
  } else {
    // pointer - pointer
    Value *LHS = Ops.LHS;
    Value *RHS = Ops.RHS;

    uint64_t ElementSize;

    // Handle GCC extension for pointer arithmetic on void* and function pointer
    // types.
    if (LHSElementType->isVoidType() || LHSElementType->isFunctionType()) {
      ElementSize = 1;
    } else {
      ElementSize = CGF.getContext().getTypeSize(LHSElementType) / 8;
    }

    const llvm::Type *ResultType = ConvertType(Ops.Ty);
    LHS = Builder.CreatePtrToInt(LHS, ResultType, "sub.ptr.lhs.cast");
    RHS = Builder.CreatePtrToInt(RHS, ResultType, "sub.ptr.rhs.cast");
    Value *BytesBetween = Builder.CreateSub(LHS, RHS, "sub.ptr.sub");

    // Optimize out the shift for element size of 1.
    if (ElementSize == 1)
      return BytesBetween;

    // Otherwise, do a full sdiv. This uses the "exact" form of sdiv, since
    // pointer difference in C is only defined in the case where both operands
    // are pointing to elements of an array.
    Value *BytesPerElt = llvm::ConstantInt::get(ResultType, ElementSize);
    return Builder.CreateExactSDiv(BytesBetween, BytesPerElt, "sub.ptr.div");
  }
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
  TestAndClearIgnoreResultAssign();
  Value *Result;
  QualType LHSTy = E->getLHS()->getType();
  if (LHSTy->isMemberFunctionPointerType()) {
    Value *LHSPtr = CGF.EmitAnyExprToTemp(E->getLHS()).getAggregateAddr();
    Value *RHSPtr = CGF.EmitAnyExprToTemp(E->getRHS()).getAggregateAddr();
    llvm::Value *LHSFunc = Builder.CreateStructGEP(LHSPtr, 0);
    LHSFunc = Builder.CreateLoad(LHSFunc);
    llvm::Value *RHSFunc = Builder.CreateStructGEP(RHSPtr, 0);
    RHSFunc = Builder.CreateLoad(RHSFunc);
    Value *ResultF = Builder.CreateICmp((llvm::ICmpInst::Predicate)UICmpOpc,
                                        LHSFunc, RHSFunc, "cmp.func");
    Value *NullPtr = llvm::Constant::getNullValue(LHSFunc->getType());
    Value *ResultNull = Builder.CreateICmp((llvm::ICmpInst::Predicate)UICmpOpc,
                                           LHSFunc, NullPtr, "cmp.null");
    llvm::Value *LHSAdj = Builder.CreateStructGEP(LHSPtr, 1);
    LHSAdj = Builder.CreateLoad(LHSAdj);
    llvm::Value *RHSAdj = Builder.CreateStructGEP(RHSPtr, 1);
    RHSAdj = Builder.CreateLoad(RHSAdj);
    Value *ResultA = Builder.CreateICmp((llvm::ICmpInst::Predicate)UICmpOpc,
                                        LHSAdj, RHSAdj, "cmp.adj");
    if (E->getOpcode() == BinaryOperator::EQ) {
      Result = Builder.CreateOr(ResultNull, ResultA, "or.na");
      Result = Builder.CreateAnd(Result, ResultF, "and.f");
    } else {
      assert(E->getOpcode() == BinaryOperator::NE &&
             "Member pointer comparison other than == or != ?");
      Result = Builder.CreateAnd(ResultNull, ResultA, "and.na");
      Result = Builder.CreateOr(Result, ResultF, "or.f");
    }
  } else if (!LHSTy->isAnyComplexType()) {
    Value *LHS = Visit(E->getLHS());
    Value *RHS = Visit(E->getRHS());

    if (LHS->getType()->isFPOrFPVector()) {
      Result = Builder.CreateFCmp((llvm::CmpInst::Predicate)FCmpOpc,
                                  LHS, RHS, "cmp");
    } else if (LHSTy->isSignedIntegerType()) {
      Result = Builder.CreateICmp((llvm::ICmpInst::Predicate)SICmpOpc,
                                  LHS, RHS, "cmp");
    } else {
      // Unsigned integers and pointers.
      Result = Builder.CreateICmp((llvm::ICmpInst::Predicate)UICmpOpc,
                                  LHS, RHS, "cmp");
    }

    // If this is a vector comparison, sign extend the result to the appropriate
    // vector integer type and return it (don't convert to bool).
    if (LHSTy->isVectorType())
      return Builder.CreateSExt(Result, ConvertType(E->getType()), "sext");

  } else {
    // Complex Comparison: can only be an equality comparison.
    CodeGenFunction::ComplexPairTy LHS = CGF.EmitComplexExpr(E->getLHS());
    CodeGenFunction::ComplexPairTy RHS = CGF.EmitComplexExpr(E->getRHS());

    QualType CETy = LHSTy->getAs<ComplexType>()->getElementType();

    Value *ResultR, *ResultI;
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

  return EmitScalarConversion(Result, CGF.getContext().BoolTy, E->getType());
}

Value *ScalarExprEmitter::VisitBinAssign(const BinaryOperator *E) {
  bool Ignore = TestAndClearIgnoreResultAssign();

  // __block variables need to have the rhs evaluated first, plus this should
  // improve codegen just a little.
  Value *RHS = Visit(E->getRHS());
  LValue LHS = EmitLValue(E->getLHS());

  // Store the value into the LHS.  Bit-fields are handled specially
  // because the result is altered by the store, i.e., [C99 6.5.16p1]
  // 'An assignment expression has the value of the left operand after
  // the assignment...'.
  if (LHS.isBitfield()) {
    if (!LHS.isVolatileQualified()) {
      CGF.EmitStoreThroughBitfieldLValue(RValue::get(RHS), LHS, E->getType(),
                                         &RHS);
      return RHS;
    } else
      CGF.EmitStoreThroughBitfieldLValue(RValue::get(RHS), LHS, E->getType());
  } else
    CGF.EmitStoreThroughLValue(RValue::get(RHS), LHS, E->getType());
  if (Ignore)
    return 0;
  return EmitLoadOfLValue(LHS, E->getType());
}

Value *ScalarExprEmitter::VisitBinLAnd(const BinaryOperator *E) {
  const llvm::Type *ResTy = ConvertType(E->getType());
  
  // If we have 0 && RHS, see if we can elide RHS, if so, just return 0.
  // If we have 1 && X, just emit X without inserting the control flow.
  if (int Cond = CGF.ConstantFoldsToSimpleInteger(E->getLHS())) {
    if (Cond == 1) { // If we have 1 && X, just emit X.
      Value *RHSCond = CGF.EvaluateExprAsBool(E->getRHS());
      // ZExt result to int or bool.
      return Builder.CreateZExtOrBitCast(RHSCond, ResTy, "land.ext");
    }

    // 0 && RHS: If it is safe, just elide the RHS, and return 0/false.
    if (!CGF.ContainsLabel(E->getRHS()))
      return llvm::Constant::getNullValue(ResTy);
  }

  llvm::BasicBlock *ContBlock = CGF.createBasicBlock("land.end");
  llvm::BasicBlock *RHSBlock  = CGF.createBasicBlock("land.rhs");

  // Branch on the LHS first.  If it is false, go to the failure (cont) block.
  CGF.EmitBranchOnBoolExpr(E->getLHS(), RHSBlock, ContBlock);

  // Any edges into the ContBlock are now from an (indeterminate number of)
  // edges from this first condition.  All of these values will be false.  Start
  // setting up the PHI node in the Cont Block for this.
  llvm::PHINode *PN = llvm::PHINode::Create(llvm::Type::getInt1Ty(VMContext),
                                            "", ContBlock);
  PN->reserveOperandSpace(2);  // Normal case, two inputs.
  for (llvm::pred_iterator PI = pred_begin(ContBlock), PE = pred_end(ContBlock);
       PI != PE; ++PI)
    PN->addIncoming(llvm::ConstantInt::getFalse(VMContext), *PI);

  CGF.StartConditionalBranch();
  CGF.EmitBlock(RHSBlock);
  Value *RHSCond = CGF.EvaluateExprAsBool(E->getRHS());
  CGF.FinishConditionalBranch();

  // Reaquire the RHS block, as there may be subblocks inserted.
  RHSBlock = Builder.GetInsertBlock();

  // Emit an unconditional branch from this block to ContBlock.  Insert an entry
  // into the phi node for the edge with the value of RHSCond.
  CGF.EmitBlock(ContBlock);
  PN->addIncoming(RHSCond, RHSBlock);

  // ZExt result to int.
  return Builder.CreateZExtOrBitCast(PN, ResTy, "land.ext");
}

Value *ScalarExprEmitter::VisitBinLOr(const BinaryOperator *E) {
  const llvm::Type *ResTy = ConvertType(E->getType());
  
  // If we have 1 || RHS, see if we can elide RHS, if so, just return 1.
  // If we have 0 || X, just emit X without inserting the control flow.
  if (int Cond = CGF.ConstantFoldsToSimpleInteger(E->getLHS())) {
    if (Cond == -1) { // If we have 0 || X, just emit X.
      Value *RHSCond = CGF.EvaluateExprAsBool(E->getRHS());
      // ZExt result to int or bool.
      return Builder.CreateZExtOrBitCast(RHSCond, ResTy, "lor.ext");
    }

    // 1 || RHS: If it is safe, just elide the RHS, and return 1/true.
    if (!CGF.ContainsLabel(E->getRHS()))
      return llvm::ConstantInt::get(ResTy, 1);
  }

  llvm::BasicBlock *ContBlock = CGF.createBasicBlock("lor.end");
  llvm::BasicBlock *RHSBlock = CGF.createBasicBlock("lor.rhs");

  // Branch on the LHS first.  If it is true, go to the success (cont) block.
  CGF.EmitBranchOnBoolExpr(E->getLHS(), ContBlock, RHSBlock);

  // Any edges into the ContBlock are now from an (indeterminate number of)
  // edges from this first condition.  All of these values will be true.  Start
  // setting up the PHI node in the Cont Block for this.
  llvm::PHINode *PN = llvm::PHINode::Create(llvm::Type::getInt1Ty(VMContext),
                                            "", ContBlock);
  PN->reserveOperandSpace(2);  // Normal case, two inputs.
  for (llvm::pred_iterator PI = pred_begin(ContBlock), PE = pred_end(ContBlock);
       PI != PE; ++PI)
    PN->addIncoming(llvm::ConstantInt::getTrue(VMContext), *PI);

  CGF.StartConditionalBranch();

  // Emit the RHS condition as a bool value.
  CGF.EmitBlock(RHSBlock);
  Value *RHSCond = CGF.EvaluateExprAsBool(E->getRHS());

  CGF.FinishConditionalBranch();

  // Reaquire the RHS block, as there may be subblocks inserted.
  RHSBlock = Builder.GetInsertBlock();

  // Emit an unconditional branch from this block to ContBlock.  Insert an entry
  // into the phi node for the edge with the value of RHSCond.
  CGF.EmitBlock(ContBlock);
  PN->addIncoming(RHSCond, RHSBlock);

  // ZExt result to int.
  return Builder.CreateZExtOrBitCast(PN, ResTy, "lor.ext");
}

Value *ScalarExprEmitter::VisitBinComma(const BinaryOperator *E) {
  CGF.EmitStmt(E->getLHS());
  CGF.EnsureInsertPoint();
  return Visit(E->getRHS());
}

//===----------------------------------------------------------------------===//
//                             Other Operators
//===----------------------------------------------------------------------===//

/// isCheapEnoughToEvaluateUnconditionally - Return true if the specified
/// expression is cheap enough and side-effect-free enough to evaluate
/// unconditionally instead of conditionally.  This is used to convert control
/// flow into selects in some cases.
static bool isCheapEnoughToEvaluateUnconditionally(const Expr *E,
                                                   CodeGenFunction &CGF) {
  if (const ParenExpr *PE = dyn_cast<ParenExpr>(E))
    return isCheapEnoughToEvaluateUnconditionally(PE->getSubExpr(), CGF);

  // TODO: Allow anything we can constant fold to an integer or fp constant.
  if (isa<IntegerLiteral>(E) || isa<CharacterLiteral>(E) ||
      isa<FloatingLiteral>(E))
    return true;

  // Non-volatile automatic variables too, to get "cond ? X : Y" where
  // X and Y are local variables.
  if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E))
    if (const VarDecl *VD = dyn_cast<VarDecl>(DRE->getDecl()))
      if (VD->hasLocalStorage() && !(CGF.getContext()
                                     .getCanonicalType(VD->getType())
                                     .isVolatileQualified()))
        return true;

  return false;
}


Value *ScalarExprEmitter::
VisitConditionalOperator(const ConditionalOperator *E) {
  TestAndClearIgnoreResultAssign();
  // If the condition constant folds and can be elided, try to avoid emitting
  // the condition and the dead arm.
  if (int Cond = CGF.ConstantFoldsToSimpleInteger(E->getCond())){
    Expr *Live = E->getLHS(), *Dead = E->getRHS();
    if (Cond == -1)
      std::swap(Live, Dead);

    // If the dead side doesn't have labels we need, and if the Live side isn't
    // the gnu missing ?: extension (which we could handle, but don't bother
    // to), just emit the Live part.
    if ((!Dead || !CGF.ContainsLabel(Dead)) &&  // No labels in dead part
        Live)                                   // Live part isn't missing.
      return Visit(Live);
  }


  // If this is a really simple expression (like x ? 4 : 5), emit this as a
  // select instead of as control flow.  We can only do this if it is cheap and
  // safe to evaluate the LHS and RHS unconditionally.
  if (E->getLHS() && isCheapEnoughToEvaluateUnconditionally(E->getLHS(),
                                                            CGF) &&
      isCheapEnoughToEvaluateUnconditionally(E->getRHS(), CGF)) {
    llvm::Value *CondV = CGF.EvaluateExprAsBool(E->getCond());
    llvm::Value *LHS = Visit(E->getLHS());
    llvm::Value *RHS = Visit(E->getRHS());
    return Builder.CreateSelect(CondV, LHS, RHS, "cond");
  }


  llvm::BasicBlock *LHSBlock = CGF.createBasicBlock("cond.true");
  llvm::BasicBlock *RHSBlock = CGF.createBasicBlock("cond.false");
  llvm::BasicBlock *ContBlock = CGF.createBasicBlock("cond.end");
  Value *CondVal = 0;

  // If we don't have the GNU missing condition extension, emit a branch on bool
  // the normal way.
  if (E->getLHS()) {
    // Otherwise, just use EmitBranchOnBoolExpr to get small and simple code for
    // the branch on bool.
    CGF.EmitBranchOnBoolExpr(E->getCond(), LHSBlock, RHSBlock);
  } else {
    // Otherwise, for the ?: extension, evaluate the conditional and then
    // convert it to bool the hard way.  We do this explicitly because we need
    // the unconverted value for the missing middle value of the ?:.
    CondVal = CGF.EmitScalarExpr(E->getCond());

    // In some cases, EmitScalarConversion will delete the "CondVal" expression
    // if there are no extra uses (an optimization).  Inhibit this by making an
    // extra dead use, because we're going to add a use of CondVal later.  We
    // don't use the builder for this, because we don't want it to get optimized
    // away.  This leaves dead code, but the ?: extension isn't common.
    new llvm::BitCastInst(CondVal, CondVal->getType(), "dummy?:holder",
                          Builder.GetInsertBlock());

    Value *CondBoolVal =
      CGF.EmitScalarConversion(CondVal, E->getCond()->getType(),
                               CGF.getContext().BoolTy);
    Builder.CreateCondBr(CondBoolVal, LHSBlock, RHSBlock);
  }

  CGF.StartConditionalBranch();
  CGF.EmitBlock(LHSBlock);

  // Handle the GNU extension for missing LHS.
  Value *LHS;
  if (E->getLHS())
    LHS = Visit(E->getLHS());
  else    // Perform promotions, to handle cases like "short ?: int"
    LHS = EmitScalarConversion(CondVal, E->getCond()->getType(), E->getType());

  CGF.FinishConditionalBranch();
  LHSBlock = Builder.GetInsertBlock();
  CGF.EmitBranch(ContBlock);

  CGF.StartConditionalBranch();
  CGF.EmitBlock(RHSBlock);

  Value *RHS = Visit(E->getRHS());
  CGF.FinishConditionalBranch();
  RHSBlock = Builder.GetInsertBlock();
  CGF.EmitBranch(ContBlock);

  CGF.EmitBlock(ContBlock);

  // If the LHS or RHS is a throw expression, it will be legitimately null.
  if (!LHS)
    return RHS;
  if (!RHS)
    return LHS;

  // Create a PHI node for the real part.
  llvm::PHINode *PN = Builder.CreatePHI(LHS->getType(), "cond");
  PN->reserveOperandSpace(2);
  PN->addIncoming(LHS, LHSBlock);
  PN->addIncoming(RHS, RHSBlock);
  return PN;
}

Value *ScalarExprEmitter::VisitChooseExpr(ChooseExpr *E) {
  return Visit(E->getChosenSubExpr(CGF.getContext()));
}

Value *ScalarExprEmitter::VisitVAArgExpr(VAArgExpr *VE) {
  llvm::Value *ArgValue = CGF.EmitVAListRef(VE->getSubExpr());
  llvm::Value *ArgPtr = CGF.EmitVAArg(ArgValue, VE->getType());

  // If EmitVAArg fails, we fall back to the LLVM instruction.
  if (!ArgPtr)
    return Builder.CreateVAArg(ArgValue, ConvertType(VE->getType()));

  // FIXME Volatility.
  return Builder.CreateLoad(ArgPtr);
}

Value *ScalarExprEmitter::VisitBlockExpr(const BlockExpr *BE) {
  return CGF.BuildBlockLiteralTmp(BE);
}

//===----------------------------------------------------------------------===//
//                         Entry Point into this File
//===----------------------------------------------------------------------===//

/// EmitScalarExpr - Emit the computation of the specified expression of scalar
/// type, ignoring the result.
Value *CodeGenFunction::EmitScalarExpr(const Expr *E, bool IgnoreResultAssign) {
  assert(E && !hasAggregateLLVMType(E->getType()) &&
         "Invalid scalar expression to emit");

  return ScalarExprEmitter(*this, IgnoreResultAssign)
    .Visit(const_cast<Expr*>(E));
}

/// EmitScalarConversion - Emit a conversion from the specified type to the
/// specified destination type, both of which are LLVM scalar types.
Value *CodeGenFunction::EmitScalarConversion(Value *Src, QualType SrcTy,
                                             QualType DstTy) {
  assert(!hasAggregateLLVMType(SrcTy) && !hasAggregateLLVMType(DstTy) &&
         "Invalid scalar expression to emit");
  return ScalarExprEmitter(*this).EmitScalarConversion(Src, SrcTy, DstTy);
}

/// EmitComplexToScalarConversion - Emit a conversion from the specified complex
/// type to the specified destination type, where the destination type is an
/// LLVM scalar type.
Value *CodeGenFunction::EmitComplexToScalarConversion(ComplexPairTy Src,
                                                      QualType SrcTy,
                                                      QualType DstTy) {
  assert(SrcTy->isAnyComplexType() && !hasAggregateLLVMType(DstTy) &&
         "Invalid complex -> scalar conversion");
  return ScalarExprEmitter(*this).EmitComplexToScalarConversion(Src, SrcTy,
                                                                DstTy);
}

Value *CodeGenFunction::EmitShuffleVector(Value* V1, Value *V2, ...) {
  assert(V1->getType() == V2->getType() &&
         "Vector operands must be of the same type");
  unsigned NumElements =
    cast<llvm::VectorType>(V1->getType())->getNumElements();

  va_list va;
  va_start(va, V2);

  llvm::SmallVector<llvm::Constant*, 16> Args;
  for (unsigned i = 0; i < NumElements; i++) {
    int n = va_arg(va, int);
    assert(n >= 0 && n < (int)NumElements * 2 &&
           "Vector shuffle index out of bounds!");
    Args.push_back(llvm::ConstantInt::get(
                                         llvm::Type::getInt32Ty(VMContext), n));
  }

  const char *Name = va_arg(va, const char *);
  va_end(va);

  llvm::Constant *Mask = llvm::ConstantVector::get(&Args[0], NumElements);

  return Builder.CreateShuffleVector(V1, V2, Mask, Name);
}

llvm::Value *CodeGenFunction::EmitVector(llvm::Value * const *Vals,
                                         unsigned NumVals, bool isSplat) {
  llvm::Value *Vec
    = llvm::UndefValue::get(llvm::VectorType::get(Vals[0]->getType(), NumVals));

  for (unsigned i = 0, e = NumVals; i != e; ++i) {
    llvm::Value *Val = isSplat ? Vals[0] : Vals[i];
    llvm::Value *Idx = llvm::ConstantInt::get(
                                          llvm::Type::getInt32Ty(VMContext), i);
    Vec = Builder.CreateInsertElement(Vec, Val, Idx, "tmp");
  }

  return Vec;
}

LValue CodeGenFunction::EmitObjCIsaExpr(const ObjCIsaExpr *E) {
  llvm::Value *V;
  // object->isa or (*object).isa
  // Generate code as for: *(Class*)object
  Expr *BaseExpr = E->getBase();
  if (E->isArrow())
    V = ScalarExprEmitter(*this).EmitLoadOfLValue(BaseExpr);
  else
    V  = EmitLValue(BaseExpr).getAddress();
  
  // build Class* type
  const llvm::Type *ClassPtrTy = ConvertType(E->getType());
  ClassPtrTy = ClassPtrTy->getPointerTo();
  V = Builder.CreateBitCast(V, ClassPtrTy);
  LValue LV = LValue::MakeAddr(V, MakeQualifiers(E->getType()));
  return LV;
}

