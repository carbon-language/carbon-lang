//===--- CGExprConstant.cpp - Emit LLVM Code from Constant Expressions ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Constant Expr nodes as LLVM code.
//
//===----------------------------------------------------------------------===//

#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "clang/AST/AST.h"
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Support/Compiler.h"
using namespace clang;
using namespace CodeGen;

namespace  {
class VISIBILITY_HIDDEN ConstExprEmitter : 
  public StmtVisitor<ConstExprEmitter, llvm::Constant*> {
  CodeGenModule &CGM;
  CodeGenFunction *CGF;
public:
  ConstExprEmitter(CodeGenModule &cgm, CodeGenFunction *cgf)
    : CGM(cgm), CGF(cgf) {
  }
    
  //===--------------------------------------------------------------------===//
  //                            Visitor Methods
  //===--------------------------------------------------------------------===//
    
  llvm::Constant *VisitStmt(Stmt *S) {
    CGM.WarnUnsupported(S, "constant expression");
    QualType T = cast<Expr>(S)->getType();
    return llvm::UndefValue::get(CGM.getTypes().ConvertType(T));
  }
  
  llvm::Constant *VisitParenExpr(ParenExpr *PE) { 
    return Visit(PE->getSubExpr()); 
  }
  
  // Leaves
  llvm::Constant *VisitIntegerLiteral(const IntegerLiteral *E) {
    return llvm::ConstantInt::get(E->getValue());
  }
  llvm::Constant *VisitFloatingLiteral(const FloatingLiteral *E) {
    return llvm::ConstantFP::get(E->getValue());
  }
  llvm::Constant *VisitCharacterLiteral(const CharacterLiteral *E) {
    return llvm::ConstantInt::get(ConvertType(E->getType()), E->getValue());
  }
  llvm::Constant *VisitCXXBoolLiteralExpr(const CXXBoolLiteralExpr *E) {
    return llvm::ConstantInt::get(ConvertType(E->getType()), E->getValue());
  }
  
  llvm::Constant *VisitCompoundLiteralExpr(CompoundLiteralExpr *E) {
    return Visit(E->getInitializer());
  }
  
  llvm::Constant *VisitCastExpr(const CastExpr* E) {
    llvm::Constant *C = Visit(E->getSubExpr());
    
    return EmitConversion(C, E->getSubExpr()->getType(), E->getType());    
  }

  llvm::Constant *VisitCXXDefaultArgExpr(CXXDefaultArgExpr *DAE) {
    return Visit(DAE->getExpr());
  }

  llvm::Constant *EmitArrayInitialization(InitListExpr *ILE,
                                          const llvm::ArrayType *AType) {
    std::vector<llvm::Constant*> Elts;
    unsigned NumInitElements = ILE->getNumInits();
    // FIXME: Check for wide strings
    if (NumInitElements > 0 && isa<StringLiteral>(ILE->getInit(0)) &&
        ILE->getType()->getAsArrayType()->getElementType()->isCharType())
      return Visit(ILE->getInit(0));
    const llvm::Type *ElemTy = AType->getElementType();
    unsigned NumElements = AType->getNumElements();

    // Initialising an array requires us to automatically 
    // initialise any elements that have not been initialised explicitly
    unsigned NumInitableElts = std::min(NumInitElements, NumElements);

    // Copy initializer elements.
    unsigned i = 0;
    for (; i < NumInitableElts; ++i) {
      llvm::Constant *C = Visit(ILE->getInit(i));
      assert (C && "Failed to create initializer expression");
      Elts.push_back(C);
    }
    
    // Initialize remaining array elements.
    for (; i < NumElements; ++i)
      Elts.push_back(llvm::Constant::getNullValue(ElemTy));

    return llvm::ConstantArray::get(AType, Elts);    
  }

  llvm::Constant *EmitStructInitialization(InitListExpr *ILE,
                                           const llvm::StructType *SType) {

    RecordDecl *RD = ILE->getType()->getAsRecordType()->getDecl();
    std::vector<llvm::Constant*> Elts;

    // Initialize the whole structure to zero.
    for (unsigned i = 0; i < SType->getNumElements(); ++i) {
      const llvm::Type *FieldTy = SType->getElementType(i);
      Elts.push_back(llvm::Constant::getNullValue(FieldTy));
    }

    // Copy initializer elements. Skip padding fields.
    unsigned EltNo = 0;  // Element no in ILE
    int FieldNo = 0; // Field no in RecordDecl
    while (EltNo < ILE->getNumInits() && FieldNo < RD->getNumMembers()) {
      FieldDecl* curField = RD->getMember(FieldNo);
      FieldNo++;
      if (!curField->getIdentifier())
        continue;

      llvm::Constant *C = Visit(ILE->getInit(EltNo));
      assert (C && "Failed to create initializer expression");

      if (curField->isBitField()) {
        CGM.WarnUnsupported(ILE->getInit(EltNo), "bitfield initialization");
      } else {
        Elts[CGM.getTypes().getLLVMFieldNo(curField)] = C;
      }
      EltNo++;
    }

    return llvm::ConstantStruct::get(SType, Elts);
  }

  llvm::Constant *EmitVectorInitialization(InitListExpr *ILE,
                                           const llvm::VectorType *VType) {

    std::vector<llvm::Constant*> Elts;    
    unsigned NumInitElements = ILE->getNumInits();      
    unsigned NumElements = VType->getNumElements();

    assert (NumInitElements == NumElements 
            && "Unsufficient vector init elelments");
    // Copy initializer elements.
    unsigned i = 0;
    for (; i < NumElements; ++i) {
      llvm::Constant *C = Visit(ILE->getInit(i));
      assert (C && "Failed to create initializer expression");
      Elts.push_back(C);
    }

    return llvm::ConstantVector::get(VType, Elts);    
  }
                                          
  llvm::Constant *VisitInitListExpr(InitListExpr *ILE) {
    const llvm::CompositeType *CType = 
      dyn_cast<llvm::CompositeType>(ConvertType(ILE->getType()));

    if (!CType) {
        // We have a scalar in braces. Just use the first element.
        return Visit(ILE->getInit(0));
    }
      
    if (const llvm::ArrayType *AType = dyn_cast<llvm::ArrayType>(CType))
      return EmitArrayInitialization(ILE, AType);

    if (const llvm::StructType *SType = dyn_cast<llvm::StructType>(CType))
      return EmitStructInitialization(ILE, SType);

    if (const llvm::VectorType *VType = dyn_cast<llvm::VectorType>(CType))
      return EmitVectorInitialization(ILE, VType);
    
    // Make sure we have an array at this point
    assert(0 && "Unable to handle InitListExpr");
    // Get rid of control reaches end of void function warning.
    // Not reached.
    return 0;
  }

  llvm::Constant *VisitImplicitCastExpr(ImplicitCastExpr *ICExpr) {
    Expr* SExpr = ICExpr->getSubExpr();
    QualType SType = SExpr->getType();
    llvm::Constant *C; // the intermediate expression
    QualType T;        // the type of the intermediate expression
    if (SType->isArrayType()) {
      // Arrays decay to a pointer to the first element
      // VLAs would require special handling, but they can't occur here
      C = EmitLValue(SExpr);
      llvm::Constant *Idx0 = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0);
      llvm::Constant *Ops[] = {Idx0, Idx0};
      C = llvm::ConstantExpr::getGetElementPtr(C, Ops, 2);

      QualType ElemType = SType->getAsArrayType()->getElementType();
      T = CGM.getContext().getPointerType(ElemType);
    } else if (SType->isFunctionType()) {
      // Function types decay to a pointer to the function
      C = EmitLValue(SExpr);
      T = CGM.getContext().getPointerType(SType);
    } else {
      C = Visit(SExpr);
      T = SType;
    }

    // Perform the conversion; note that an implicit cast can both promote
    // and convert an array/function
    return EmitConversion(C, T, ICExpr->getType());
  }

  llvm::Constant *VisitStringLiteral(StringLiteral *E) {
    const char *StrData = E->getStrData();
    unsigned Len = E->getByteLength();
    assert(!E->getType()->isPointerType() && "Strings are always arrays");
    
    // Otherwise this must be a string initializing an array in a static
    // initializer.  Don't emit it as the address of the string, emit the string
    // data itself as an inline array.
    const ConstantArrayType *CAT = E->getType()->getAsConstantArrayType();
    assert(CAT && "String isn't pointer or array!");
    
    std::string Str(StrData, StrData + Len);
    // Null terminate the string before potentially truncating it.
    // FIXME: What about wchar_t strings?
    Str.push_back(0);
    
    uint64_t RealLen = CAT->getSize().getZExtValue();
    // String or grow the initializer to the required size.
    if (RealLen != Str.size())
      Str.resize(RealLen);
    
    return llvm::ConstantArray::get(Str, false);
  }

  llvm::Constant *VisitDeclRefExpr(DeclRefExpr *E) {
    const ValueDecl *Decl = E->getDecl();
    if (const EnumConstantDecl *EC = dyn_cast<EnumConstantDecl>(Decl))
      return llvm::ConstantInt::get(EC->getInitVal());
    assert(0 && "Unsupported decl ref type!");
    return 0;
  }

  llvm::Constant *VisitSizeOfAlignOfTypeExpr(const SizeOfAlignOfTypeExpr *E) {
    return EmitSizeAlignOf(E->getArgumentType(), E->getType(), E->isSizeOf());
  }

  // Unary operators
  llvm::Constant *VisitUnaryPlus(const UnaryOperator *E) {
    return Visit(E->getSubExpr());
  }
  llvm::Constant *VisitUnaryMinus(const UnaryOperator *E) {
    return llvm::ConstantExpr::getNeg(Visit(E->getSubExpr()));
  }
  llvm::Constant *VisitUnaryNot(const UnaryOperator *E) {
    return llvm::ConstantExpr::getNot(Visit(E->getSubExpr()));
  }  
  llvm::Constant *VisitUnaryLNot(const UnaryOperator *E) {
    llvm::Constant *SubExpr = Visit(E->getSubExpr());
    
    if (E->getSubExpr()->getType()->isRealFloatingType()) {
      // Compare against 0.0 for fp scalars.
      llvm::Constant *Zero = llvm::Constant::getNullValue(SubExpr->getType());
      SubExpr = llvm::ConstantExpr::getFCmp(llvm::FCmpInst::FCMP_UEQ, SubExpr,
                                            Zero);
    } else {
      assert((E->getSubExpr()->getType()->isIntegerType() ||
              E->getSubExpr()->getType()->isPointerType()) &&
             "Unknown scalar type to convert");
      // Compare against an integer or pointer null.
      llvm::Constant *Zero = llvm::Constant::getNullValue(SubExpr->getType());
      SubExpr = llvm::ConstantExpr::getICmp(llvm::ICmpInst::ICMP_EQ, SubExpr,
                                            Zero);
    }

    return llvm::ConstantExpr::getZExt(SubExpr, ConvertType(E->getType()));
  }
  llvm::Constant *VisitUnarySizeOf(const UnaryOperator *E) {
    return EmitSizeAlignOf(E->getSubExpr()->getType(), E->getType(), true);
  }
  llvm::Constant *VisitUnaryAlignOf(const UnaryOperator *E) {
    return EmitSizeAlignOf(E->getSubExpr()->getType(), E->getType(), false);
  }
  llvm::Constant *VisitUnaryAddrOf(const UnaryOperator *E) {
    return EmitLValue(E->getSubExpr());
  }
  llvm::Constant *VisitUnaryOffsetOf(const UnaryOperator *E) {
    int64_t Val = E->evaluateOffsetOf(CGM.getContext());
    
    assert(E->getType()->isIntegerType() && "Result type must be an integer!");
    
    uint32_t ResultWidth =
      static_cast<uint32_t>(CGM.getContext().getTypeSize(E->getType()));
    return llvm::ConstantInt::get(llvm::APInt(ResultWidth, Val));    
  }
  
  // Binary operators
  llvm::Constant *VisitBinOr(const BinaryOperator *E) {
    llvm::Constant *LHS = Visit(E->getLHS());
    llvm::Constant *RHS = Visit(E->getRHS());
    
    return llvm::ConstantExpr::getOr(LHS, RHS);
  }
  llvm::Constant *VisitBinSub(const BinaryOperator *E) {
    llvm::Constant *LHS = Visit(E->getLHS());
    llvm::Constant *RHS = Visit(E->getRHS());
    
    if (!isa<llvm::PointerType>(RHS->getType())) {
      // pointer - int
      if (isa<llvm::PointerType>(LHS->getType())) {
        llvm::Constant *Idx = llvm::ConstantExpr::getNeg(RHS);
      
        return llvm::ConstantExpr::getGetElementPtr(LHS, &Idx, 1);
      }
      
      // int - int
      return llvm::ConstantExpr::getSub(LHS, RHS);
    }
    
    assert(0 && "Unhandled bin sub case!");
    return 0;
  }
    
  llvm::Constant *VisitBinShl(const BinaryOperator *E) {
    llvm::Constant *LHS = Visit(E->getLHS());
    llvm::Constant *RHS = Visit(E->getRHS());

    // LLVM requires the LHS and RHS to be the same type: promote or truncate the
    // RHS to the same size as the LHS.
    if (LHS->getType() != RHS->getType())
      RHS = llvm::ConstantExpr::getIntegerCast(RHS, LHS->getType(), false);
    
    return llvm::ConstantExpr::getShl(LHS, RHS);
  }
    
  llvm::Constant *VisitBinMul(const BinaryOperator *E) {
    llvm::Constant *LHS = Visit(E->getLHS());
    llvm::Constant *RHS = Visit(E->getRHS());

    return llvm::ConstantExpr::getMul(LHS, RHS);
  }

  llvm::Constant *VisitBinDiv(const BinaryOperator *E) {
    llvm::Constant *LHS = Visit(E->getLHS());
    llvm::Constant *RHS = Visit(E->getRHS());
    
    if (LHS->getType()->isFPOrFPVector())
      return llvm::ConstantExpr::getFDiv(LHS, RHS);
    else if (E->getType()->isUnsignedIntegerType())
      return llvm::ConstantExpr::getUDiv(LHS, RHS);
    else
      return llvm::ConstantExpr::getSDiv(LHS, RHS);
  }

  llvm::Constant *VisitBinAdd(const BinaryOperator *E) {
    llvm::Constant *LHS = Visit(E->getLHS());
    llvm::Constant *RHS = Visit(E->getRHS());

    if (!E->getType()->isPointerType())
      return llvm::ConstantExpr::getAdd(LHS, RHS);
    
    llvm::Constant *Ptr, *Idx;
    if (isa<llvm::PointerType>(LHS->getType())) { // pointer + int
      Ptr = LHS;
      Idx = RHS;
    } else { // int + pointer
      Ptr = RHS;
      Idx = LHS;
    }
    
    return llvm::ConstantExpr::getGetElementPtr(Ptr, &Idx, 1);
  }
    
  llvm::Constant *VisitBinAnd(const BinaryOperator *E) {
    llvm::Constant *LHS = Visit(E->getLHS());
    llvm::Constant *RHS = Visit(E->getRHS());

    return llvm::ConstantExpr::getAnd(LHS, RHS);
  }
    
  // Utility methods
  const llvm::Type *ConvertType(QualType T) {
    return CGM.getTypes().ConvertType(T);
  }
  
  llvm::Constant *EmitConversionToBool(llvm::Constant *Src, QualType SrcType) {
    assert(SrcType->isCanonical() && "EmitConversion strips typedefs");
    
    if (SrcType->isRealFloatingType()) {
      // Compare against 0.0 for fp scalars.
      llvm::Constant *Zero = llvm::Constant::getNullValue(Src->getType());
      return llvm::ConstantExpr::getFCmp(llvm::FCmpInst::FCMP_UNE, Src, Zero); 
    }
    
    assert((SrcType->isIntegerType() || SrcType->isPointerType()) &&
           "Unknown scalar type to convert");
    
    // Compare against an integer or pointer null.
    llvm::Constant *Zero = llvm::Constant::getNullValue(Src->getType());
    return llvm::ConstantExpr::getICmp(llvm::ICmpInst::ICMP_NE, Src, Zero);
  }    
  
  llvm::Constant *EmitConversion(llvm::Constant *Src, QualType SrcType, 
                                 QualType DstType) {
    SrcType = SrcType.getCanonicalType();
    DstType = DstType.getCanonicalType();
    if (SrcType == DstType) return Src;
    
    // Handle conversions to bool first, they are special: comparisons against 0.
    if (DstType->isBooleanType())
      return EmitConversionToBool(Src, SrcType);
    
    const llvm::Type *DstTy = ConvertType(DstType);
    
    // Ignore conversions like int -> uint.
    if (Src->getType() == DstTy)
      return Src;

    // Handle pointer conversions next: pointers can only be converted to/from
    // other pointers and integers.
    if (isa<PointerType>(DstType)) {
      // The source value may be an integer, or a pointer.
      if (isa<llvm::PointerType>(Src->getType()))
        return llvm::ConstantExpr::getBitCast(Src, DstTy);
      assert(SrcType->isIntegerType() &&"Not ptr->ptr or int->ptr conversion?");
      return llvm::ConstantExpr::getIntToPtr(Src, DstTy);
    }
    
    if (isa<PointerType>(SrcType)) {
      // Must be an ptr to int cast.
      assert(isa<llvm::IntegerType>(DstTy) && "not ptr->int?");
      return llvm::ConstantExpr::getPtrToInt(Src, DstTy);
    }
    
    // A scalar source can be splatted to a vector of the same element type
    if (isa<llvm::VectorType>(DstTy) && !isa<VectorType>(SrcType)) {
      const llvm::VectorType *VT = cast<llvm::VectorType>(DstTy);
      assert((VT->getElementType() == Src->getType()) &&
             "Vector element type must match scalar type to splat.");
      unsigned NumElements = DstType->getAsVectorType()->getNumElements();
      llvm::SmallVector<llvm::Constant*, 16> Elements;
      for (unsigned i = 0; i < NumElements; i++)
        Elements.push_back(Src);
        
      return llvm::ConstantVector::get(&Elements[0], NumElements);
    }
    
    if (isa<llvm::VectorType>(Src->getType()) ||
        isa<llvm::VectorType>(DstTy)) {
      return llvm::ConstantExpr::getBitCast(Src, DstTy);
    }
    
    // Finally, we have the arithmetic types: real int/float.
    if (isa<llvm::IntegerType>(Src->getType())) {
      bool InputSigned = SrcType->isSignedIntegerType();
      if (isa<llvm::IntegerType>(DstTy))
        return llvm::ConstantExpr::getIntegerCast(Src, DstTy, InputSigned);
      else if (InputSigned)
        return llvm::ConstantExpr::getSIToFP(Src, DstTy);
      else
        return llvm::ConstantExpr::getUIToFP(Src, DstTy);
    }
    
    assert(Src->getType()->isFloatingPoint() && "Unknown real conversion");
    if (isa<llvm::IntegerType>(DstTy)) {
      if (DstType->isSignedIntegerType())
        return llvm::ConstantExpr::getFPToSI(Src, DstTy);
      else
        return llvm::ConstantExpr::getFPToUI(Src, DstTy);
    }
    
    assert(DstTy->isFloatingPoint() && "Unknown real conversion");
    if (DstTy->getTypeID() < Src->getType()->getTypeID())
      return llvm::ConstantExpr::getFPTrunc(Src, DstTy);
    else
      return llvm::ConstantExpr::getFPExtend(Src, DstTy);
  }
  
  llvm::Constant *EmitSizeAlignOf(QualType TypeToSize, 
                                  QualType RetType, bool isSizeOf) {
    std::pair<uint64_t, unsigned> Info =
      CGM.getContext().getTypeInfo(TypeToSize);
    
    uint64_t Val = isSizeOf ? Info.first : Info.second;
    Val /= 8;  // Return size in bytes, not bits.
    
    assert(RetType->isIntegerType() && "Result type must be an integer!");
    
    uint32_t ResultWidth = 
      static_cast<uint32_t>(CGM.getContext().getTypeSize(RetType));
    return llvm::ConstantInt::get(llvm::APInt(ResultWidth, Val));
  }

  llvm::Constant *EmitLValue(Expr *E) {
    switch (E->getStmtClass()) {
    default: break;
    case Expr::ParenExprClass:
      // Elide parenthesis
      return EmitLValue(cast<ParenExpr>(E)->getSubExpr());
    case Expr::CompoundLiteralExprClass: {
      // Note that due to the nature of compound literals, this is guaranteed
      // to be the only use of the variable, so we just generate it here.
      CompoundLiteralExpr *CLE = cast<CompoundLiteralExpr>(E);
      llvm::Constant* C = Visit(CLE->getInitializer());
      C = new llvm::GlobalVariable(C->getType(),E->getType().isConstQualified(), 
                                   llvm::GlobalValue::InternalLinkage,
                                   C, ".compoundliteral", &CGM.getModule());
      return C;
    }
    case Expr::DeclRefExprClass: {
      ValueDecl *Decl = cast<DeclRefExpr>(E)->getDecl();
      if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(Decl))
        return CGM.GetAddrOfFunctionDecl(FD, false);
      if (const VarDecl* VD = dyn_cast<VarDecl>(Decl)) {
        if (VD->isFileVarDecl())
          return CGM.GetAddrOfGlobalVar(VD, false);
        else if (VD->isBlockVarDecl()) {
          assert(CGF && "Can't access static local vars without CGF");
          return CGF->GetAddrOfStaticLocalVar(VD);
        }
      }
      break;
    }
    case Expr::MemberExprClass: {
      MemberExpr* ME = cast<MemberExpr>(E);
      llvm::Constant *Base;
      if (ME->isArrow())
        Base = Visit(ME->getBase());
      else
        Base = EmitLValue(ME->getBase());

      unsigned FieldNumber = CGM.getTypes().getLLVMFieldNo(ME->getMemberDecl());
      llvm::Constant *Zero = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0);
      llvm::Constant *Idx = llvm::ConstantInt::get(llvm::Type::Int32Ty,
                                                   FieldNumber);
      llvm::Value *Ops[] = {Zero, Idx};
      return llvm::ConstantExpr::getGetElementPtr(Base, Ops, 2);
    }
    case Expr::ArraySubscriptExprClass: {
      ArraySubscriptExpr* ASExpr = cast<ArraySubscriptExpr>(E);
      llvm::Constant *Base = Visit(ASExpr->getBase());
      llvm::Constant *Index = Visit(ASExpr->getIdx());
      assert(!ASExpr->getBase()->getType()->isVectorType() &&
             "Taking the address of a vector component is illegal!");
      return llvm::ConstantExpr::getGetElementPtr(Base, &Index, 1);
    }
    case Expr::StringLiteralClass: {
      StringLiteral *String = cast<StringLiteral>(E);
      assert(!String->isWide() && "Cannot codegen wide strings yet");
      const char *StrData = String->getStrData();
      unsigned Len = String->getByteLength();

      return CGM.GetAddrOfConstantString(std::string(StrData, StrData + Len));
    }
    case Expr::UnaryOperatorClass: {
      UnaryOperator *Exp = cast<UnaryOperator>(E);
      switch (Exp->getOpcode()) {
      default: break;
      case UnaryOperator::Extension:
        // Extension is just a wrapper for expressions
        return EmitLValue(Exp->getSubExpr());
      case UnaryOperator::Real:
      case UnaryOperator::Imag: {
        // The address of __real or __imag is just a GEP off the address
        // of the internal expression
        llvm::Constant* C = EmitLValue(Exp->getSubExpr());
        llvm::Constant *Zero = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0);
        llvm::Constant *Idx  = llvm::ConstantInt::get(llvm::Type::Int32Ty,
                                       Exp->getOpcode() == UnaryOperator::Imag);
        llvm::Value *Ops[] = {Zero, Idx};
        return llvm::ConstantExpr::getGetElementPtr(C, Ops, 2);
      }
      case UnaryOperator::Deref:
        // The address of a deref is just the value of the expression
        return Visit(Exp->getSubExpr());
      }
      break;
    }
    }
    CGM.WarnUnsupported(E, "constant l-value expression");
    llvm::Type *Ty = llvm::PointerType::getUnqual(ConvertType(E->getType()));
    return llvm::UndefValue::get(Ty);
  }

};
  
}  // end anonymous namespace.


llvm::Constant *CodeGenModule::EmitConstantExpr(const Expr *E,
                                                CodeGenFunction *CGF)
{
  QualType type = E->getType().getCanonicalType();
  
  if (type->isIntegerType()) {
    llvm::APSInt Value(static_cast<uint32_t>(Context.getTypeSize(type)));
    if (E->isIntegerConstantExpr(Value, Context)) {
      return llvm::ConstantInt::get(Value);
    } 
  }
  
  return ConstExprEmitter(*this, CGF).Visit(const_cast<Expr*>(E));
}
