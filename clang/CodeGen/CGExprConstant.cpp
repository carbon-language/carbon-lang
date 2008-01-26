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
class VISIBILITY_HIDDEN ConstExprEmitter : public StmtVisitor<ConstExprEmitter, llvm::Constant*> {
  CodeGenModule &CGM;
public:
  ConstExprEmitter(CodeGenModule &cgm)
    : CGM(cgm) {
  }
    
  //===--------------------------------------------------------------------===//
  //                            Visitor Methods
  //===--------------------------------------------------------------------===//
    
  llvm::Constant *VisitStmt(Stmt *S) {
    CGM.WarnUnsupported(S, "constant expression");
    return 0;
  }
  
  llvm::Constant *VisitParenExpr(ParenExpr *PE) { 
    return Visit(PE->getSubExpr()); 
  }
  
  // Leaves
  llvm::Constant *VisitIntegerLiteral(const IntegerLiteral *E) {
    return llvm::ConstantInt::get(E->getValue());
  }
  llvm::Constant *VisitFloatingLiteral(const FloatingLiteral *E) {
    return llvm::ConstantFP::get(ConvertType(E->getType()), E->getValue());
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
  
  llvm::Constant *VisitInitListExpr(InitListExpr *ILE) {
    if (ILE->getType()->isVoidType()) {
      // FIXME: Remove this when sema of initializers is finished (and the code
      // below).
      CGM.WarnUnsupported(ILE, "initializer");
      return 0;
    }
    
    assert((ILE->getType()->isArrayType() || ILE->getType()->isStructureType() ||
            ILE->getType()->isVectorType()) &&
           "Bad type for init list!");
    CodeGenTypes& Types = CGM.getTypes();
    
    unsigned NumInitElements = ILE->getNumInits();
    unsigned NumInitableElts = NumInitElements;
    
    const llvm::CompositeType *CType = 
    cast<llvm::CompositeType>(Types.ConvertType(ILE->getType()));
    assert(CType);
    std::vector<llvm::Constant*> Elts;    
    
    // Initialising an array requires us to automatically initialise any 
    // elements that have not been initialised explicitly
    const llvm::ArrayType *AType = 0; 
    const llvm::Type *AElemTy = 0;
    unsigned NumArrayElements = 0;
    
    // If this is an array, we may have to truncate the initializer
    if ((AType = dyn_cast<llvm::ArrayType>(CType))) {
      NumArrayElements = AType->getNumElements();
      AElemTy = AType->getElementType();
      NumInitableElts = std::min(NumInitableElts, NumArrayElements);
    }
    
    // Copy initializer elements.
    unsigned i = 0;
    for (i = 0; i < NumInitableElts; ++i) {
      llvm::Constant *C = Visit(ILE->getInit(i));
      // FIXME: Remove this when sema of initializers is finished (and the code
      // above).
      if (C == 0 && ILE->getInit(i)->getType()->isVoidType()) {
        if (ILE->getType()->isVoidType()) return 0;
        return llvm::UndefValue::get(CType);
      }
      assert (C && "Failed to create initializer expression");
      Elts.push_back(C);
    }
    
    if (ILE->getType()->isStructureType())
      return llvm::ConstantStruct::get(cast<llvm::StructType>(CType), Elts);
    
    if (ILE->getType()->isVectorType())
      return llvm::ConstantVector::get(cast<llvm::VectorType>(CType), Elts);
    
    // Make sure we have an array at this point
    assert(AType);
    
    // Initialize remaining array elements.
    for (; i < NumArrayElements; ++i)
      Elts.push_back(llvm::Constant::getNullValue(AElemTy));
    
    return llvm::ConstantArray::get(AType, Elts);    
  }
  
  llvm::Constant *VisitImplicitCastExpr(ImplicitCastExpr *ICExpr) {
    // If this is due to array->pointer conversion, emit the array expression as
    // an l-value.
    if (ICExpr->getSubExpr()->getType()->isArrayType()) {
      // Note that VLAs can't exist for global variables.
      // The only thing that can have array type like this is a
      // DeclRefExpr(FileVarDecl)?
      const DeclRefExpr *DRE = cast<DeclRefExpr>(ICExpr->getSubExpr());
      const VarDecl *VD = cast<VarDecl>(DRE->getDecl());
      llvm::Constant *C = CGM.GetAddrOfGlobalVar(VD, false);
      assert(isa<llvm::PointerType>(C->getType()) &&
             isa<llvm::ArrayType>(cast<llvm::PointerType>(C->getType())
                                  ->getElementType()));
      llvm::Constant *Idx0 = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0);
      
      llvm::Constant *Ops[] = {Idx0, Idx0};
      C = llvm::ConstantExpr::getGetElementPtr(C, Ops, 2);
      
      // The resultant pointer type can be implicitly cast to other pointer
      // types as well, for example void*.
      const llvm::Type *DestPTy = ConvertType(ICExpr->getType());
      assert(isa<llvm::PointerType>(DestPTy) &&
             "Only expect implicit cast to pointer");
      return llvm::ConstantExpr::getBitCast(C, DestPTy);
    }
    
    llvm::Constant *C = Visit(ICExpr->getSubExpr());
    
    return EmitConversion(C, ICExpr->getSubExpr()->getType(),ICExpr->getType());
  }
  
  llvm::Constant *VisitStringLiteral(StringLiteral *E) {
    const char *StrData = E->getStrData();
    unsigned Len = E->getByteLength();
    
    // If the string has a pointer type, emit it as a global and use the pointer
    // to the global as its value.
    if (E->getType()->isPointerType()) 
      return CGM.GetAddrOfConstantString(std::string(StrData, StrData + Len));
    
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
    if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(Decl))
      return CGM.GetAddrOfFunctionDecl(FD, false);
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
      assert(SrcType->isIntegerType() && "Not ptr->ptr or int->ptr conversion?");
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
    CGM.getContext().getTypeInfo(TypeToSize, SourceLocation());
    
    uint64_t Val = isSizeOf ? Info.first : Info.second;
    Val /= 8;  // Return size in bytes, not bits.
    
    assert(RetType->isIntegerType() && "Result type must be an integer!");
    
    uint32_t ResultWidth = static_cast<uint32_t>(
      CGM.getContext().getTypeSize(RetType, SourceLocation()));
    return llvm::ConstantInt::get(llvm::APInt(ResultWidth, Val));
  }
  
  };
}  // end anonymous namespace.


llvm::Constant *CodeGenModule::EmitConstantExpr(const Expr *E)
{
  QualType type = E->getType().getCanonicalType();
  
  if (type->isIntegerType()) {
    llvm::APSInt
    Value(static_cast<uint32_t>(Context.getTypeSize(type, SourceLocation())));
    if (E->isIntegerConstantExpr(Value, Context)) {
      return llvm::ConstantInt::get(Value);
    } 
  }
  
  return ConstExprEmitter(*this).Visit(const_cast<Expr*>(E));
}
