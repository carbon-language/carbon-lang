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
    return 0;
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
      Expr *SubExpr = E->getSubExpr();
      return EmitUnion(CGM.EmitConstantExpr(SubExpr, SubExpr->getType(), CGF), 
                       Ty);
    }
    if (CGM.getContext().getCanonicalType(E->getSubExpr()->getType()) ==
        CGM.getContext().getCanonicalType(E->getType())) {
      return Visit(E->getSubExpr());
    }
    return 0;
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
    // FIXME: Check for NumInitElements exactly equal to 1??
    if (NumInitElements > 0 && 
        (isa<StringLiteral>(ILE->getInit(0)) ||
         isa<ObjCEncodeExpr>(ILE->getInit(0))) &&
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
      Expr *Init = ILE->getInit(i);
      llvm::Constant *C = CGM.EmitConstantExpr(Init, Init->getType(), CGF);
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
    llvm::Constant *C = CGM.EmitConstantExpr(E, Field->getType(), CGF);
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
    for (RecordDecl::field_iterator Field = RD->field_begin(CGM.getContext()),
                                 FieldEnd = RD->field_end(CGM.getContext());
         EltNo < ILE->getNumInits() && Field != FieldEnd; ++Field) {
      FieldNo++;
      if (!Field->getIdentifier())
        continue;

      if (Field->isBitField()) {
        InsertBitfieldIntoStruct(Elts, *Field, ILE->getInit(EltNo));
      } else {
        unsigned FieldNo = CGM.getTypes().getLLVMFieldNo(*Field);
        llvm::Constant *C = CGM.EmitConstantExpr(ILE->getInit(EltNo), 
                                                 Field->getType(), CGF);
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
      for (RecordDecl::field_iterator Field = RD->field_begin(CGM.getContext()),
                                   FieldEnd = RD->field_end(CGM.getContext());
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

    Expr *Init = ILE->getInit(0);
    return EmitUnion(CGM.EmitConstantExpr(Init, Init->getType(), CGF), Ty);
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
      Expr *Init = ILE->getInit(i);
      llvm::Constant *C = CGM.EmitConstantExpr(Init, Init->getType(), CGF);
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
      if (ILE->getNumInits() > 0) {
        Expr *Init = ILE->getInit(0);
        return CGM.EmitConstantExpr(Init, Init->getType(), CGF);
      }
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

  llvm::Constant *VisitStringLiteral(StringLiteral *E) {
    assert(!E->getType()->isPointerType() && "Strings are always arrays");
    
    // This must be a string initializing an array in a static initializer.
    // Don't emit it as the address of the string, emit the string data itself
    // as an inline array.
    return llvm::ConstantArray::get(CGM.GetStringForStringLiteral(E), false);
  }

  llvm::Constant *VisitObjCEncodeExpr(ObjCEncodeExpr *E) {
    // This must be an @encode initializing an array in a static initializer.
    // Don't emit it as the address of the string, emit the string data itself
    // as an inline array.
    std::string Str;
    CGM.getContext().getObjCEncodingForType(E->getEncodedType(), Str);
    const ConstantArrayType *CAT = cast<ConstantArrayType>(E->getType());
    
    // Resize the string to the right size, adding zeros at the end, or
    // truncating as needed.
    Str.resize(CAT->getSize().getZExtValue(), '\0');
    return llvm::ConstantArray::get(Str, false);
  }
    
  llvm::Constant *VisitUnaryExtension(const UnaryOperator *E) {
    return Visit(E->getSubExpr());
  }

  // Utility methods
  const llvm::Type *ConvertType(QualType T) {
    return CGM.getTypes().ConvertType(T);
  }

public:
  llvm::Constant *EmitLValue(Expr *E) {
    switch (E->getStmtClass()) {
    default: break;
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
        // We can never refer to a variable with local storage.
        if (!VD->hasLocalStorage()) {          
          if (VD->isFileVarDecl() || VD->hasExternalStorage())
            return CGM.GetAddrOfGlobalVar(VD);
          else if (VD->isBlockVarDecl()) {
            assert(CGF && "Can't access static local vars without CGF");
            return CGF->GetAddrOfStaticLocalVar(VD);
          }
        }
      }
      break;
    }
    case Expr::StringLiteralClass:
      return CGM.GetAddrOfConstantStringFromLiteral(cast<StringLiteral>(E));
    case Expr::ObjCEncodeExprClass:
      return CGM.GetAddrOfConstantStringFromObjCEncode(cast<ObjCEncodeExpr>(E));
    case Expr::ObjCStringLiteralClass: {
      ObjCStringLiteral* SL = cast<ObjCStringLiteral>(E);
      llvm::Constant *C = CGM.getObjCRuntime().GenerateConstantString(SL);
      return llvm::ConstantExpr::getBitCast(C, ConvertType(E->getType()));
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
      // FIXME: need to deal with UCN conversion issues.
      return CGM.GetAddrOfConstantCFString(Literal);
    }
    case Expr::BlockExprClass: {
      std::string FunctionName;
      if (CGF)
        FunctionName = CGF->CurFn->getName();
      else
        FunctionName = "global";

      return CGM.GetAddrOfGlobalBlock(cast<BlockExpr>(E), FunctionName.c_str());
    }
    }

    return 0;
  }
};
  
}  // end anonymous namespace.

llvm::Constant *CodeGenModule::EmitConstantExpr(const Expr *E,
                                                QualType DestType,
                                                CodeGenFunction *CGF) {
  Expr::EvalResult Result;
  
  bool Success = false;
  
  if (DestType->isReferenceType()) {
    // If the destination type is a reference type, we need to evaluate it
    // as an lvalue.
    if (E->EvaluateAsLValue(Result, Context)) {
      if (const Expr *LVBase = Result.Val.getLValueBase()) {
        if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(LVBase)) {
          const ValueDecl *VD = cast<ValueDecl>(DRE->getDecl());

          // We can only initialize a reference with an lvalue if the lvalue
          // is not a reference itself.
          Success = !VD->getType()->isReferenceType();
        }
      }
    }
  } else 
    Success = E->Evaluate(Result, Context);
  
  if (Success) {
    assert(!Result.HasSideEffects && 
           "Constant expr should not have any side effects!");
    switch (Result.Val.getKind()) {
    case APValue::Uninitialized:
      assert(0 && "Constant expressions should be initialized.");
      return 0;
    case APValue::LValue: {
      const llvm::Type *DestTy = getTypes().ConvertTypeForMem(DestType);
      llvm::Constant *Offset = 
        llvm::ConstantInt::get(llvm::Type::Int64Ty, 
                               Result.Val.getLValueOffset());
      
      llvm::Constant *C;
      if (const Expr *LVBase = Result.Val.getLValueBase()) {
        C = ConstExprEmitter(*this, CGF).EmitLValue(const_cast<Expr*>(LVBase));

        // Apply offset if necessary.
        if (!Offset->isNullValue()) {
          const llvm::Type *Type = 
            llvm::PointerType::getUnqual(llvm::Type::Int8Ty);
          llvm::Constant *Casted = llvm::ConstantExpr::getBitCast(C, Type);
          Casted = llvm::ConstantExpr::getGetElementPtr(Casted, &Offset, 1);
          C = llvm::ConstantExpr::getBitCast(Casted, C->getType());
        }

        // Convert to the appropriate type; this could be an lvalue for
        // an integer.
        if (isa<llvm::PointerType>(DestTy))
          return llvm::ConstantExpr::getBitCast(C, DestTy);

        return llvm::ConstantExpr::getPtrToInt(C, DestTy);
      } else {
        C = Offset;

        // Convert to the appropriate type; this could be an lvalue for
        // an integer.
        if (isa<llvm::PointerType>(DestTy))
          return llvm::ConstantExpr::getIntToPtr(C, DestTy);

        // If the types don't match this should only be a truncate.
        if (C->getType() != DestTy)
          return llvm::ConstantExpr::getTrunc(C, DestTy);

        return C;
      }
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
