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
#include "CGObjCRuntime.h"
#include "clang/AST/APValue.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/StmtVisitor.h"
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Target/TargetData.h"
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
    CGM.ErrorUnsupported(S, "constant expression");
    QualType T = cast<Expr>(S)->getType();
    return llvm::UndefValue::get(CGM.getTypes().ConvertType(T));
  }
  
  llvm::Constant *VisitParenExpr(ParenExpr *PE) { 
    return Visit(PE->getSubExpr()); 
  }
    
  llvm::Constant *VisitCompoundLiteralExpr(CompoundLiteralExpr *E) {
    return Visit(E->getInitializer());
  }
  
  llvm::Constant *VisitCastExpr(CastExpr* E) {
    // GCC cast to union extension
    if (E->getType()->isUnionType()) {
      const llvm::Type *Ty = ConvertType(E->getType());
      return EmitUnion(CGM.EmitConstantExpr(E->getSubExpr(), CGF), Ty);
    }

    llvm::Constant *C = Visit(E->getSubExpr());
    return EmitConversion(C, E->getSubExpr()->getType(), E->getType());    
  }

  llvm::Constant *VisitCXXDefaultArgExpr(CXXDefaultArgExpr *DAE) {
    return Visit(DAE->getExpr());
  }

  llvm::Constant *EmitArrayInitialization(InitListExpr *ILE) {
    std::vector<llvm::Constant*> Elts;
    const llvm::ArrayType *AType =
        cast<llvm::ArrayType>(ConvertType(ILE->getType()));
    unsigned NumInitElements = ILE->getNumInits();
    // FIXME: Check for wide strings
    if (NumInitElements > 0 && isa<StringLiteral>(ILE->getInit(0)) &&
        ILE->getType()->getArrayElementTypeNoTypeQual()->isCharType())
      return Visit(ILE->getInit(0));
    const llvm::Type *ElemTy = AType->getElementType();
    unsigned NumElements = AType->getNumElements();

    // Initialising an array requires us to automatically 
    // initialise any elements that have not been initialised explicitly
    unsigned NumInitableElts = std::min(NumInitElements, NumElements);

    // Copy initializer elements.
    unsigned i = 0;
    bool RewriteType = false;
    for (; i < NumInitableElts; ++i) {
      llvm::Constant *C = CGM.EmitConstantExpr(ILE->getInit(i), CGF);
      if (!C)
        return 0;
      RewriteType |= (C->getType() != ElemTy);
      Elts.push_back(C);
    }

    // Initialize remaining array elements.
    for (; i < NumElements; ++i)
      Elts.push_back(llvm::Constant::getNullValue(ElemTy));

    if (RewriteType) {
      // FIXME: Try to avoid packing the array
      std::vector<const llvm::Type*> Types;
      for (unsigned i = 0; i < Elts.size(); ++i)
        Types.push_back(Elts[i]->getType());
      const llvm::StructType *SType = llvm::StructType::get(Types, true);
      return llvm::ConstantStruct::get(SType, Elts);
    }

    return llvm::ConstantArray::get(AType, Elts);    
  }

  void InsertBitfieldIntoStruct(std::vector<llvm::Constant*>& Elts,
                                FieldDecl* Field, Expr* E) {
    // Calculate the value to insert
    llvm::Constant *C = CGM.EmitConstantExpr(E, CGF);
    if (!C)
      return;

    llvm::ConstantInt *CI = dyn_cast<llvm::ConstantInt>(C);
    if (!CI) {
      CGM.ErrorUnsupported(E, "bitfield initialization");
      return;
    }
    llvm::APInt V = CI->getValue();

    // Calculate information about the relevant field
    const llvm::Type* Ty = CI->getType();
    const llvm::TargetData &TD = CGM.getTypes().getTargetData();
    unsigned size = TD.getTypePaddedSizeInBits(Ty);
    unsigned fieldOffset = CGM.getTypes().getLLVMFieldNo(Field) * size;
    CodeGenTypes::BitFieldInfo bitFieldInfo =
        CGM.getTypes().getBitFieldInfo(Field);
    fieldOffset += bitFieldInfo.Begin;

    // Find where to start the insertion
    // FIXME: This is O(n^2) in the number of bit-fields!
    // FIXME: This won't work if the struct isn't completely packed!
    unsigned offset = 0, i = 0;
    while (offset < (fieldOffset & -8))
      offset += TD.getTypePaddedSizeInBits(Elts[i++]->getType());

    // Advance over 0 sized elements (must terminate in bounds since
    // the bitfield must have a size).
    while (TD.getTypePaddedSizeInBits(Elts[i]->getType()) == 0)
      ++i;

    // Promote the size of V if necessary
    // FIXME: This should never occur, but currently it can because
    // initializer constants are cast to bool, and because clang is
    // not enforcing bitfield width limits.
    if (bitFieldInfo.Size > V.getBitWidth())
      V.zext(bitFieldInfo.Size);

    // Insert the bits into the struct
    // FIXME: This algorthm is only correct on X86!
    // FIXME: THis algorthm assumes bit-fields only have byte-size elements!
    unsigned bitsToInsert = bitFieldInfo.Size;
    unsigned curBits = std::min(8 - (fieldOffset & 7), bitsToInsert);
    unsigned byte = V.getLoBits(curBits).getZExtValue() << (fieldOffset & 7);
    do {
      llvm::Constant* byteC = llvm::ConstantInt::get(llvm::Type::Int8Ty, byte);
      Elts[i] = llvm::ConstantExpr::getOr(Elts[i], byteC);
      ++i;
      V = V.lshr(curBits);
      bitsToInsert -= curBits;

      if (!bitsToInsert)
        break;

      curBits = bitsToInsert > 8 ? 8 : bitsToInsert;
      byte = V.getLoBits(curBits).getZExtValue();
    } while (true);
  }

  llvm::Constant *EmitStructInitialization(InitListExpr *ILE) {
    const llvm::StructType *SType =
        cast<llvm::StructType>(ConvertType(ILE->getType()));
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
    bool RewriteType = false;
    for (RecordDecl::field_iterator Field = RD->field_begin(),
                                 FieldEnd = RD->field_end();
         EltNo < ILE->getNumInits() && Field != FieldEnd; ++Field) {
      FieldNo++;
      if (!Field->getIdentifier())
        continue;

      if (Field->isBitField()) {
        InsertBitfieldIntoStruct(Elts, *Field, ILE->getInit(EltNo));
      } else {
        unsigned FieldNo = CGM.getTypes().getLLVMFieldNo(*Field);
        llvm::Constant *C = CGM.EmitConstantExpr(ILE->getInit(EltNo), CGF);
        if (!C) return 0;
        RewriteType |= (C->getType() != Elts[FieldNo]->getType());
        Elts[FieldNo] = C;
      }
      EltNo++;
    }

    if (RewriteType) {
      // FIXME: Make this work for non-packed structs
      assert(SType->isPacked() && "Cannot recreate unpacked structs");
      std::vector<const llvm::Type*> Types;
      for (unsigned i = 0; i < Elts.size(); ++i)
        Types.push_back(Elts[i]->getType());
      SType = llvm::StructType::get(Types, true);
    }

    return llvm::ConstantStruct::get(SType, Elts);
  }

  llvm::Constant *EmitUnion(llvm::Constant *C, const llvm::Type *Ty) {
    if (!C)
      return 0;

    // Build a struct with the union sub-element as the first member,
    // and padded to the appropriate size
    std::vector<llvm::Constant*> Elts;
    std::vector<const llvm::Type*> Types;
    Elts.push_back(C);
    Types.push_back(C->getType());
    unsigned CurSize = CGM.getTargetData().getTypePaddedSize(C->getType());
    unsigned TotalSize = CGM.getTargetData().getTypePaddedSize(Ty);
    while (CurSize < TotalSize) {
      Elts.push_back(llvm::Constant::getNullValue(llvm::Type::Int8Ty));
      Types.push_back(llvm::Type::Int8Ty);
      CurSize++;
    }

    // This always generates a packed struct
    // FIXME: Try to generate an unpacked struct when we can
    llvm::StructType* STy = llvm::StructType::get(Types, true);
    return llvm::ConstantStruct::get(STy, Elts);
  }

  llvm::Constant *EmitUnionInitialization(InitListExpr *ILE) {
    const llvm::Type *Ty = ConvertType(ILE->getType());

    // If this is an empty initializer list, we value-initialize the
    // union.
    if (ILE->getNumInits() == 0)
      return llvm::Constant::getNullValue(Ty);

    FieldDecl* curField = ILE->getInitializedFieldInUnion();
    if (!curField) {
      // There's no field to initialize, so value-initialize the union.
#ifndef NDEBUG
      // Make sure that it's really an empty and not a failure of
      // semantic analysis.
      RecordDecl *RD = ILE->getType()->getAsRecordType()->getDecl();
      for (RecordDecl::field_iterator Field = RD->field_begin(),
                                   FieldEnd = RD->field_end();
           Field != FieldEnd; ++Field)
        assert(Field->isUnnamedBitfield() && "Only unnamed bitfields allowed");
#endif
      return llvm::Constant::getNullValue(Ty);
    }

    if (curField->isBitField()) {
      // Create a dummy struct for bit-field insertion
      unsigned NumElts = CGM.getTargetData().getTypePaddedSize(Ty) / 8;
      llvm::Constant* NV = llvm::Constant::getNullValue(llvm::Type::Int8Ty);
      std::vector<llvm::Constant*> Elts(NumElts, NV);

      InsertBitfieldIntoStruct(Elts, curField, ILE->getInit(0));
      const llvm::ArrayType *RetTy =
          llvm::ArrayType::get(NV->getType(), NumElts);
      return llvm::ConstantArray::get(RetTy, Elts);
    }

    return EmitUnion(CGM.EmitConstantExpr(ILE->getInit(0), CGF), Ty);
  }

  llvm::Constant *EmitVectorInitialization(InitListExpr *ILE) {
    const llvm::VectorType *VType =
        cast<llvm::VectorType>(ConvertType(ILE->getType()));
    const llvm::Type *ElemTy = VType->getElementType();
    std::vector<llvm::Constant*> Elts;
    unsigned NumElements = VType->getNumElements();
    unsigned NumInitElements = ILE->getNumInits();

    unsigned NumInitableElts = std::min(NumInitElements, NumElements);

    // Copy initializer elements.
    unsigned i = 0;
    for (; i < NumInitableElts; ++i) {
      llvm::Constant *C = CGM.EmitConstantExpr(ILE->getInit(i), CGF);
      if (!C)
        return 0;
      Elts.push_back(C);
    }

    for (; i < NumElements; ++i)
      Elts.push_back(llvm::Constant::getNullValue(ElemTy));

    return llvm::ConstantVector::get(VType, Elts);    
  }
  
  llvm::Constant *VisitImplicitValueInitExpr(ImplicitValueInitExpr* E) {
    const llvm::Type* RetTy = CGM.getTypes().ConvertType(E->getType());
    return llvm::Constant::getNullValue(RetTy);
  }
    
  llvm::Constant *VisitInitListExpr(InitListExpr *ILE) {
    if (ILE->getType()->isScalarType()) {
      // We have a scalar in braces. Just use the first element.
      if (ILE->getNumInits() > 0)
        return CGM.EmitConstantExpr(ILE->getInit(0), CGF);

      const llvm::Type* RetTy = CGM.getTypes().ConvertType(ILE->getType());
      return llvm::Constant::getNullValue(RetTy);
    }
    
    if (ILE->getType()->isArrayType())
      return EmitArrayInitialization(ILE);

    if (ILE->getType()->isStructureType())
      return EmitStructInitialization(ILE);

    if (ILE->getType()->isUnionType())
      return EmitUnionInitialization(ILE);

    if (ILE->getType()->isVectorType())
      return EmitVectorInitialization(ILE);

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
      T = CGM.getContext().getArrayDecayedType(SType);
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
    assert(!E->getType()->isPointerType() && "Strings are always arrays");
    
    // Otherwise this must be a string initializing an array in a static
    // initializer.  Don't emit it as the address of the string, emit the string
    // data itself as an inline array.
    return llvm::ConstantArray::get(CGM.GetStringForStringLiteral(E), false);
  }

  llvm::Constant *VisitUnaryExtension(const UnaryOperator *E) {
    return Visit(E->getSubExpr());
  }
    
  llvm::Constant *VisitBlockExpr(const BlockExpr *E) {
    const char *Name = "";
    if (const NamedDecl *ND = dyn_cast<NamedDecl>(CGF->CurFuncDecl))
      Name = ND->getNameAsString().c_str();
    return CGM.GetAddrOfGlobalBlock(E, Name);
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
    if (!Src)
      return 0;

    SrcType = CGM.getContext().getCanonicalType(SrcType);
    DstType = CGM.getContext().getCanonicalType(DstType);
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
    if (isa<llvm::PointerType>(DstTy)) {
      // The source value may be an integer, or a pointer.
      if (isa<llvm::PointerType>(Src->getType()))
        return llvm::ConstantExpr::getBitCast(Src, DstTy);
      assert(SrcType->isIntegerType() &&"Not ptr->ptr or int->ptr conversion?");
      return llvm::ConstantExpr::getIntToPtr(Src, DstTy);
    }
    
    if (isa<llvm::PointerType>(Src->getType())) {
      // Must be an ptr to int cast.
      assert(isa<llvm::IntegerType>(DstTy) && "not ptr->int?");
      return llvm::ConstantExpr::getPtrToInt(Src, DstTy);
    }
    
    // A scalar source can be splatted to a vector of the same element type
    if (isa<llvm::VectorType>(DstTy) && !isa<VectorType>(SrcType)) {
      assert((cast<llvm::VectorType>(DstTy)->getElementType()
              == Src->getType()) &&
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

public:
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
      // FIXME: "Leaked" on failure.
      if (C)
        C = new llvm::GlobalVariable(C->getType(),
                                     E->getType().isConstQualified(), 
                                     llvm::GlobalValue::InternalLinkage,
                                     C, ".compoundliteral", &CGM.getModule());
      return C;
    }
    case Expr::DeclRefExprClass: 
    case Expr::QualifiedDeclRefExprClass: {
      NamedDecl *Decl = cast<DeclRefExpr>(E)->getDecl();
      if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(Decl))
        return CGM.GetAddrOfFunction(FD);
      if (const VarDecl* VD = dyn_cast<VarDecl>(Decl)) {
        if (VD->isFileVarDecl())
          return CGM.GetAddrOfGlobalVar(VD);
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
      if (!Base)
        return 0;

      FieldDecl *Field = dyn_cast<FieldDecl>(ME->getMemberDecl());
      // FIXME: Handle other kinds of member expressions.
      assert(Field && "No code generation for non-field member expressions");
      unsigned FieldNumber = CGM.getTypes().getLLVMFieldNo(Field);
      llvm::Constant *Zero = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0);
      llvm::Constant *Idx = llvm::ConstantInt::get(llvm::Type::Int32Ty,
                                                   FieldNumber);
      llvm::Value *Ops[] = {Zero, Idx};
      return llvm::ConstantExpr::getGetElementPtr(Base, Ops, 2);
    }
    case Expr::ArraySubscriptExprClass: {
      ArraySubscriptExpr* ASExpr = cast<ArraySubscriptExpr>(E);
      assert(!ASExpr->getBase()->getType()->isVectorType() &&
             "Taking the address of a vector component is illegal!");

      llvm::Constant *Base = Visit(ASExpr->getBase());
      llvm::Constant *Index = Visit(ASExpr->getIdx());
      if (!Base || !Index)
        return 0;
      return llvm::ConstantExpr::getGetElementPtr(Base, &Index, 1);
    }
    case Expr::StringLiteralClass:
      return CGM.GetAddrOfConstantStringFromLiteral(cast<StringLiteral>(E));
    case Expr::ObjCStringLiteralClass: {
      ObjCStringLiteral* SL = cast<ObjCStringLiteral>(E);
      std::string S(SL->getString()->getStrData(), 
                    SL->getString()->getByteLength());
      llvm::Constant *C = CGM.getObjCRuntime().GenerateConstantString(S);
      return llvm::ConstantExpr::getBitCast(C, ConvertType(E->getType()));
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
        
    case Expr::PredefinedExprClass: {
      // __func__/__FUNCTION__ -> "".  __PRETTY_FUNCTION__ -> "top level".
      std::string Str;
      if (cast<PredefinedExpr>(E)->getIdentType() == 
          PredefinedExpr::PrettyFunction)
        Str = "top level";
      
      return CGM.GetAddrOfConstantCString(Str, ".tmp");
    }
    case Expr::AddrLabelExprClass: {
      assert(CGF && "Invalid address of label expression outside function.");
      unsigned id = CGF->GetIDForAddrOfLabel(cast<AddrLabelExpr>(E)->getLabel());
      llvm::Constant *C = llvm::ConstantInt::get(llvm::Type::Int32Ty, id);
      return llvm::ConstantExpr::getIntToPtr(C, ConvertType(E->getType()));
    }
    case Expr::CallExprClass: {
      CallExpr* CE = cast<CallExpr>(E);
      if (CE->isBuiltinCall(CGM.getContext()) != 
            Builtin::BI__builtin___CFStringMakeConstantString)
        break;
      const Expr *Arg = CE->getArg(0)->IgnoreParenCasts();
      const StringLiteral *Literal = cast<StringLiteral>(Arg);
      std::string S(Literal->getStrData(), Literal->getByteLength());
      return CGM.GetAddrOfConstantCFString(S);
    }
    }

    return 0;
  }
};
  
}  // end anonymous namespace.

llvm::Constant *CodeGenModule::EmitConstantExpr(const Expr *E,
                                                CodeGenFunction *CGF) {
  Expr::EvalResult Result;
  
  if (E->Evaluate(Result, Context)) {
    assert(!Result.HasSideEffects && 
           "Constant expr should not have any side effects!");
    switch (Result.Val.getKind()) {
    case APValue::Uninitialized:
      assert(0 && "Constant expressions should be initialized.");
      return 0;
    case APValue::LValue: {
      llvm::Constant *Offset = 
        llvm::ConstantInt::get(llvm::Type::Int64Ty, 
                               Result.Val.getLValueOffset());
      
      if (const Expr *LVBase = Result.Val.getLValueBase()) {
        llvm::Constant *C = 
          ConstExprEmitter(*this, CGF).EmitLValue(const_cast<Expr*>(LVBase));

        const llvm::Type *Type = 
          llvm::PointerType::getUnqual(llvm::Type::Int8Ty);
        const llvm::Type *DestType = getTypes().ConvertTypeForMem(E->getType());
        
        // FIXME: It's a little ugly that we need to cast to a pointer,
        // apply the GEP and then cast back.
        C = llvm::ConstantExpr::getBitCast(C, Type);
        C = llvm::ConstantExpr::getGetElementPtr(C, &Offset, 1);
        
        return llvm::ConstantExpr::getBitCast(C, DestType);
      }
      
      return llvm::ConstantExpr::getIntToPtr(Offset, 
                                          getTypes().ConvertType(E->getType()));
    }
    case APValue::Int: {
      llvm::Constant *C = llvm::ConstantInt::get(Result.Val.getInt());
      
      if (C->getType() == llvm::Type::Int1Ty) {
        const llvm::Type *BoolTy = getTypes().ConvertTypeForMem(E->getType());
        C = llvm::ConstantExpr::getZExt(C, BoolTy);
      }
      return C;
    }
    case APValue::ComplexInt: {
      llvm::Constant *Complex[2];
      
      Complex[0] = llvm::ConstantInt::get(Result.Val.getComplexIntReal());
      Complex[1] = llvm::ConstantInt::get(Result.Val.getComplexIntImag());
      
      return llvm::ConstantStruct::get(Complex, 2);
    }
    case APValue::Float:
      return llvm::ConstantFP::get(Result.Val.getFloat());
    case APValue::ComplexFloat: {
      llvm::Constant *Complex[2];
      
      Complex[0] = llvm::ConstantFP::get(Result.Val.getComplexFloatReal());
      Complex[1] = llvm::ConstantFP::get(Result.Val.getComplexFloatImag());
      
      return llvm::ConstantStruct::get(Complex, 2);
    }
    case APValue::Vector: {
      llvm::SmallVector<llvm::Constant *, 4> Inits;
      unsigned NumElts = Result.Val.getVectorLength();
      
      for (unsigned i = 0; i != NumElts; ++i) {
        APValue &Elt = Result.Val.getVectorElt(i);
        if (Elt.isInt())
          Inits.push_back(llvm::ConstantInt::get(Elt.getInt()));
        else
          Inits.push_back(llvm::ConstantFP::get(Elt.getFloat()));
      }
      return llvm::ConstantVector::get(&Inits[0], Inits.size());
    }
    }
  }

  llvm::Constant* C = ConstExprEmitter(*this, CGF).Visit(const_cast<Expr*>(E));
  if (C && C->getType() == llvm::Type::Int1Ty) {
    const llvm::Type *BoolTy = getTypes().ConvertTypeForMem(E->getType());
    C = llvm::ConstantExpr::getZExt(C, BoolTy);
  }
  return C;
}
