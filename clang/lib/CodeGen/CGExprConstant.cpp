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
  llvm::Constant *VisitObjCStringLiteral(const ObjCStringLiteral *E) {
    std::string S(E->getString()->getStrData(), 
                  E->getString()->getByteLength());
    llvm::Constant *C = CGM.getObjCRuntime().GenerateConstantString(S);
    return llvm::ConstantExpr::getBitCast(C, ConvertType(E->getType()));
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
      llvm::Constant *C = Visit(ILE->getInit(i));
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
    llvm::Constant *C = Visit(E);
    llvm::ConstantInt *CI = dyn_cast<llvm::ConstantInt>(C);
    if (!CI) {
      CGM.WarnUnsupported(E, "bitfield initialization");
      return;
    }
    llvm::APInt V = CI->getValue();

    // Calculate information about the relevant field
    const llvm::Type* Ty = CI->getType();
    const llvm::TargetData &TD = CGM.getTypes().getTargetData();
    unsigned size = TD.getTypeStoreSizeInBits(Ty);
    unsigned fieldOffset = CGM.getTypes().getLLVMFieldNo(Field) * size;
    CodeGenTypes::BitFieldInfo bitFieldInfo =
        CGM.getTypes().getBitFieldInfo(Field);
    fieldOffset += bitFieldInfo.Begin;

    // Find where to start the insertion
    // FIXME: This is O(n^2) in the number of bit-fields!
    // FIXME: This won't work if the struct isn't completely packed!
    unsigned offset = 0, i = 0;
    while (offset < (fieldOffset & -8))
      offset += TD.getTypeStoreSizeInBits(Elts[i++]->getType());

    // Advance over 0 sized elements (must terminate in bounds since
    // the bitfield must have a size).
    while (TD.getTypeStoreSizeInBits(Elts[i]->getType()) == 0)
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
    while (EltNo < ILE->getNumInits() && FieldNo < RD->getNumMembers()) {
      FieldDecl* curField = RD->getMember(FieldNo);
      FieldNo++;
      if (!curField->getIdentifier())
        continue;

      if (curField->isBitField()) {
        InsertBitfieldIntoStruct(Elts, curField, ILE->getInit(EltNo));
      } else {
        unsigned FieldNo = CGM.getTypes().getLLVMFieldNo(curField);
        llvm::Constant* C = Visit(ILE->getInit(EltNo));
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

  llvm::Constant *EmitUnionInitialization(InitListExpr *ILE) {
    RecordDecl *RD = ILE->getType()->getAsRecordType()->getDecl();
    const llvm::Type *Ty = ConvertType(ILE->getType());

    // Find the field decl we're initializing, if any
    int FieldNo = 0; // Field no in RecordDecl
    FieldDecl* curField = 0;
    while (FieldNo < RD->getNumMembers()) {
      curField = RD->getMember(FieldNo);
      FieldNo++;
      if (curField->getIdentifier())
        break;
    }

    if (!curField || !curField->getIdentifier() || ILE->getNumInits() == 0)
      return llvm::Constant::getNullValue(Ty);

    if (curField->isBitField()) {
      // Create a dummy struct for bit-field insertion
      unsigned NumElts = CGM.getTargetData().getABITypeSize(Ty) / 8;
      llvm::Constant* NV = llvm::Constant::getNullValue(llvm::Type::Int8Ty);
      std::vector<llvm::Constant*> Elts(NumElts, NV);

      InsertBitfieldIntoStruct(Elts, curField, ILE->getInit(0));
      const llvm::ArrayType *RetTy =
          llvm::ArrayType::get(NV->getType(), NumElts);
      return llvm::ConstantArray::get(RetTy, Elts);
    }

    llvm::Constant *C = Visit(ILE->getInit(0));

    // Build a struct with the union sub-element as the first member,
    // and padded to the appropriate size
    std::vector<llvm::Constant*> Elts;
    std::vector<const llvm::Type*> Types;
    Elts.push_back(C);
    Types.push_back(C->getType());
    unsigned CurSize = CGM.getTargetData().getTypeStoreSize(C->getType());
    unsigned TotalSize = CGM.getTargetData().getTypeStoreSize(Ty);
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
      llvm::Constant *C = Visit(ILE->getInit(i));
      Elts.push_back(C);
    }

    for (; i < NumElements; ++i)
      Elts.push_back(llvm::Constant::getNullValue(ElemTy));

    return llvm::ConstantVector::get(VType, Elts);    
  }
                                          
  llvm::Constant *VisitInitListExpr(InitListExpr *ILE) {
    if (ILE->getType()->isScalarType()) {
      // We have a scalar in braces. Just use the first element.
      if (ILE->getNumInits() > 0)
        return Visit(ILE->getInit(0));

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

  llvm::Constant *VisitUnaryExtension(const UnaryOperator *E) {
    return Visit(E->getSubExpr());
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

    assert(isa<llvm::PointerType>(LHS->getType()));

    const llvm::Type *ResultType = ConvertType(E->getType());
    const QualType Type = E->getLHS()->getType();
    const QualType ElementType = Type->getAsPointerType()->getPointeeType();

    LHS = llvm::ConstantExpr::getPtrToInt(LHS, ResultType);
    RHS = llvm::ConstantExpr::getPtrToInt(RHS, ResultType);

    llvm::Constant *sub = llvm::ConstantExpr::getSub(LHS, RHS);
    llvm::Constant *size = EmitSizeAlignOf(ElementType, E->getType(), true);
    return llvm::ConstantExpr::getSDiv(sub, size);
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

  llvm::Constant *EmitCmp(const BinaryOperator *E,
                          llvm::CmpInst::Predicate SignedPred,
                          llvm::CmpInst::Predicate UnsignedPred,
                          llvm::CmpInst::Predicate FloatPred) {
    llvm::Constant *LHS = Visit(E->getLHS());
    llvm::Constant *RHS = Visit(E->getRHS());
    llvm::Constant *Result;
    if (LHS->getType()->isInteger() ||
        isa<llvm::PointerType>(LHS->getType())) {
      if (E->getLHS()->getType()->isSignedIntegerType())
        Result = llvm::ConstantExpr::getICmp(SignedPred, LHS, RHS);
      else
        Result = llvm::ConstantExpr::getICmp(UnsignedPred, LHS, RHS);
    } else if (LHS->getType()->isFloatingPoint()) {
      Result = llvm::ConstantExpr::getFCmp(FloatPred, LHS, RHS);
    } else {
      CGM.WarnUnsupported(E, "constant expression");
      Result = llvm::ConstantInt::getFalse();
    }

    const llvm::Type* ResultType = ConvertType(E->getType());
    return llvm::ConstantExpr::getZExtOrBitCast(Result, ResultType);
  }

  llvm::Constant *VisitBinNE(const BinaryOperator *E) {
    return EmitCmp(E, llvm::CmpInst::ICMP_NE, llvm::CmpInst::ICMP_NE,
                   llvm::CmpInst::FCMP_ONE);
  }

  llvm::Constant *VisitBinEQ(const BinaryOperator *E) {
    return EmitCmp(E, llvm::CmpInst::ICMP_EQ, llvm::CmpInst::ICMP_EQ,
                   llvm::CmpInst::FCMP_OEQ);
  }

  llvm::Constant *VisitBinLT(const BinaryOperator *E) {
    return EmitCmp(E, llvm::CmpInst::ICMP_SLT, llvm::CmpInst::ICMP_ULT,
                   llvm::CmpInst::FCMP_OLT);
  }

  llvm::Constant *VisitBinLE(const BinaryOperator *E) {
    return EmitCmp(E, llvm::CmpInst::ICMP_SLE, llvm::CmpInst::ICMP_ULE,
                   llvm::CmpInst::FCMP_OLE);
  }

  llvm::Constant *VisitBinGT(const BinaryOperator *E) {
    return EmitCmp(E, llvm::CmpInst::ICMP_SGT, llvm::CmpInst::ICMP_UGT,
                   llvm::CmpInst::FCMP_OGT);
  }

  llvm::Constant *VisitBinGE(const BinaryOperator *E) {
    return EmitCmp(E, llvm::CmpInst::ICMP_SGE, llvm::CmpInst::ICMP_SGE,
                   llvm::CmpInst::FCMP_OGE);
  }

  llvm::Constant *VisitConditionalOperator(const ConditionalOperator *E) {
    llvm::Constant *Cond = Visit(E->getCond());
    llvm::Constant *CondVal = EmitConversionToBool(Cond, E->getType());
    llvm::ConstantInt *CondValInt = dyn_cast<llvm::ConstantInt>(CondVal);
    if (!CondValInt) {
      CGM.WarnUnsupported(E, "constant expression");
      return llvm::Constant::getNullValue(ConvertType(E->getType()));
    }
    if (CondValInt->isOne()) {
      if (E->getLHS())
        return Visit(E->getLHS());
      return Cond;
    }

    return Visit(E->getRHS());
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
    case Expr::StringLiteralClass:
      return CGM.GetAddrOfConstantStringFromLiteral(cast<StringLiteral>(E));
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
                                                CodeGenFunction *CGF) {
  QualType type = Context.getCanonicalType(E->getType());

  if (type->isIntegerType()) {
    llvm::APSInt Value(static_cast<uint32_t>(Context.getTypeSize(type)));
    if (E->isIntegerConstantExpr(Value, Context)) {
      return llvm::ConstantInt::get(Value);
    } 
  }

  llvm::Constant* C = ConstExprEmitter(*this, CGF).Visit(const_cast<Expr*>(E));
  if (C->getType() == llvm::Type::Int1Ty) {
    const llvm::Type *BoolTy = getTypes().ConvertTypeForMem(E->getType());
    C = llvm::ConstantExpr::getZExt(C, BoolTy);
  }
  return C;
}
