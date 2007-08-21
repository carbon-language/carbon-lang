//===--- CGExpr.cpp - Emit LLVM Code from Expressions ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Expr nodes as LLVM code.
//
//===----------------------------------------------------------------------===//

#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "clang/AST/AST.h"
#include "clang/Lex/IdentifierTable.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Support/MathExtras.h"
using namespace clang;
using namespace CodeGen;

//===--------------------------------------------------------------------===//
//                        Miscellaneous Helper Methods
//===--------------------------------------------------------------------===//

/// CreateTempAlloca - This creates a alloca and inserts it into the entry
/// block.
llvm::AllocaInst *CodeGenFunction::CreateTempAlloca(const llvm::Type *Ty,
                                                    const char *Name) {
  return new llvm::AllocaInst(Ty, 0, Name, AllocaInsertPt);
}

/// EvaluateExprAsBool - Perform the usual unary conversions on the specified
/// expression and compare the result against zero, returning an Int1Ty value.
llvm::Value *CodeGenFunction::EvaluateExprAsBool(const Expr *E) {
  return ConvertScalarValueToBool(EmitExpr(E), E->getType());
}

/// EmitLoadOfComplex - Given an RValue reference for a complex, emit code to
/// load the real and imaginary pieces, returning them as Real/Imag.
void CodeGenFunction::EmitLoadOfComplex(llvm::Value *SrcPtr,
                                        llvm::Value *&Real, llvm::Value *&Imag){
  llvm::Constant *Zero = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0);
  llvm::Constant *One  = llvm::ConstantInt::get(llvm::Type::Int32Ty, 1);
  // FIXME: It would be nice to make this "Ptr->getName()+realp"
  llvm::Value *RealPtr = Builder.CreateGEP(SrcPtr, Zero, Zero, "realp");
  llvm::Value *ImagPtr = Builder.CreateGEP(SrcPtr, Zero, One, "imagp");
  
  // FIXME: Handle volatility.
  // FIXME: It would be nice to make this "Ptr->getName()+real"
  Real = Builder.CreateLoad(RealPtr, "real");
  Imag = Builder.CreateLoad(ImagPtr, "imag");
}

/// EmitStoreOfComplex - Store the specified real/imag parts into the
/// specified value pointer.
void CodeGenFunction::EmitStoreOfComplex(llvm::Value *Real, llvm::Value *Imag,
                                         llvm::Value *ResPtr) {
  llvm::Constant *Zero = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0);
  llvm::Constant *One  = llvm::ConstantInt::get(llvm::Type::Int32Ty, 1);
  llvm::Value *RealPtr = Builder.CreateGEP(ResPtr, Zero, Zero, "real");
  llvm::Value *ImagPtr = Builder.CreateGEP(ResPtr, Zero, One, "imag");
  
  // FIXME: Handle volatility.
  Builder.CreateStore(Real, RealPtr);
  Builder.CreateStore(Imag, ImagPtr);
}

//===--------------------------------------------------------------------===//
//                               Conversions
//===--------------------------------------------------------------------===//

/// EmitConversion - Convert the value specied by Val, whose type is ValTy, to
/// the type specified by DstTy, following the rules of C99 6.3.
RValue CodeGenFunction::EmitConversion(RValue Val, QualType ValTy,
                                       QualType DstTy) {
  ValTy = ValTy.getCanonicalType();
  DstTy = DstTy.getCanonicalType();
  if (ValTy == DstTy) return Val;

  // Handle conversions to bool first, they are special: comparisons against 0.
  if (const BuiltinType *DestBT = dyn_cast<BuiltinType>(DstTy))
    if (DestBT->getKind() == BuiltinType::Bool)
      return RValue::get(ConvertScalarValueToBool(Val, ValTy));
  
  // Handle pointer conversions next: pointers can only be converted to/from
  // other pointers and integers.
  if (isa<PointerType>(DstTy)) {
    const llvm::Type *DestTy = ConvertType(DstTy);
    
    if (Val.getVal()->getType() == DestTy)
      return Val;
    
    // The source value may be an integer, or a pointer.
    assert(Val.isScalar() && "Can only convert from integer or pointer");
    if (isa<llvm::PointerType>(Val.getVal()->getType()))
      return RValue::get(Builder.CreateBitCast(Val.getVal(), DestTy, "conv"));
    assert(ValTy->isIntegerType() && "Not ptr->ptr or int->ptr conversion?");
    return RValue::get(Builder.CreateIntToPtr(Val.getVal(), DestTy, "conv"));
  }
  
  if (isa<PointerType>(ValTy)) {
    // Must be an ptr to int cast.
    const llvm::Type *DestTy = ConvertType(DstTy);
    assert(isa<llvm::IntegerType>(DestTy) && "not ptr->int?");
    return RValue::get(Builder.CreateIntToPtr(Val.getVal(), DestTy, "conv"));
  }
  
  // Finally, we have the arithmetic types: real int/float and complex
  // int/float.  Handle real->real conversions first, they are the most
  // common.
  if (Val.isScalar() && DstTy->isRealType()) {
    // We know that these are representable as scalars in LLVM, convert to LLVM
    // types since they are easier to reason about.
    llvm::Value *SrcVal = Val.getVal();
    const llvm::Type *DestTy = ConvertType(DstTy);
    if (SrcVal->getType() == DestTy) return Val;
    
    llvm::Value *Result;
    if (isa<llvm::IntegerType>(SrcVal->getType())) {
      bool InputSigned = ValTy->isSignedIntegerType();
      if (isa<llvm::IntegerType>(DestTy))
        Result = Builder.CreateIntCast(SrcVal, DestTy, InputSigned, "conv");
      else if (InputSigned)
        Result = Builder.CreateSIToFP(SrcVal, DestTy, "conv");
      else
        Result = Builder.CreateUIToFP(SrcVal, DestTy, "conv");
    } else {
      assert(SrcVal->getType()->isFloatingPoint() && "Unknown real conversion");
      if (isa<llvm::IntegerType>(DestTy)) {
        if (DstTy->isSignedIntegerType())
          Result = Builder.CreateFPToSI(SrcVal, DestTy, "conv");
        else
          Result = Builder.CreateFPToUI(SrcVal, DestTy, "conv");
      } else {
        assert(DestTy->isFloatingPoint() && "Unknown real conversion");
        if (DestTy->getTypeID() < SrcVal->getType()->getTypeID())
          Result = Builder.CreateFPTrunc(SrcVal, DestTy, "conv");
        else
          Result = Builder.CreateFPExt(SrcVal, DestTy, "conv");
      }
    }
    return RValue::get(Result);
  }
  
  assert(0 && "FIXME: We don't support complex conversions yet!");
}


/// ConvertScalarValueToBool - Convert the specified expression value to a
/// boolean (i1) truth value.  This is equivalent to "Val == 0".
llvm::Value *CodeGenFunction::ConvertScalarValueToBool(RValue Val, QualType Ty){
  Ty = Ty.getCanonicalType();
  llvm::Value *Result;
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(Ty)) {
    switch (BT->getKind()) {
    default: assert(0 && "Unknown scalar value");
    case BuiltinType::Bool:
      Result = Val.getVal();
      // Bool is already evaluated right.
      assert(Result->getType() == llvm::Type::Int1Ty &&
             "Unexpected bool value type!");
      return Result;
    case BuiltinType::Char_S:
    case BuiltinType::Char_U:
    case BuiltinType::SChar:
    case BuiltinType::UChar:
    case BuiltinType::Short:
    case BuiltinType::UShort:
    case BuiltinType::Int:
    case BuiltinType::UInt:
    case BuiltinType::Long:
    case BuiltinType::ULong:
    case BuiltinType::LongLong:
    case BuiltinType::ULongLong:
      // Code below handles simple integers.
      break;
    case BuiltinType::Float:
    case BuiltinType::Double:
    case BuiltinType::LongDouble: {
      // Compare against 0.0 for fp scalars.
      Result = Val.getVal();
      llvm::Value *Zero = llvm::Constant::getNullValue(Result->getType());
      // FIXME: llvm-gcc produces a une comparison: validate this is right.
      Result = Builder.CreateFCmpUNE(Result, Zero, "tobool");
      return Result;
    }
    }
  } else if (isa<PointerType>(Ty) || 
             cast<TagType>(Ty)->getDecl()->getKind() == Decl::Enum) {
    // Code below handles this fine.
  } else {
    assert(isa<ComplexType>(Ty) && "Unknwon type!");
    assert(0 && "FIXME: comparisons against complex not implemented yet");
  }
  
  // Usual case for integers, pointers, and enums: compare against zero.
  Result = Val.getVal();
  
  // Because of the type rules of C, we often end up computing a logical value,
  // then zero extending it to int, then wanting it as a logical value again.
  // Optimize this common case.
  if (llvm::ZExtInst *ZI = dyn_cast<llvm::ZExtInst>(Result)) {
    if (ZI->getOperand(0)->getType() == llvm::Type::Int1Ty) {
      Result = ZI->getOperand(0);
      ZI->eraseFromParent();
      return Result;
    }
  }
  
  llvm::Value *Zero = llvm::Constant::getNullValue(Result->getType());
  return Builder.CreateICmpNE(Result, Zero, "tobool");
}

//===----------------------------------------------------------------------===//
//                         LValue Expression Emission
//===----------------------------------------------------------------------===//

/// EmitLValue - Emit code to compute a designator that specifies the location
/// of the expression.
///
/// This can return one of two things: a simple address or a bitfield
/// reference.  In either case, the LLVM Value* in the LValue structure is
/// guaranteed to be an LLVM pointer type.
///
/// If this returns a bitfield reference, nothing about the pointee type of
/// the LLVM value is known: For example, it may not be a pointer to an
/// integer.
///
/// If this returns a normal address, and if the lvalue's C type is fixed
/// size, this method guarantees that the returned pointer type will point to
/// an LLVM type of the same size of the lvalue's type.  If the lvalue has a
/// variable length type, this is not possible.
///
LValue CodeGenFunction::EmitLValue(const Expr *E) {
  switch (E->getStmtClass()) {
  default:
    fprintf(stderr, "Unimplemented lvalue expr!\n");
    E->dump();
    return LValue::MakeAddr(llvm::UndefValue::get(
                              llvm::PointerType::get(llvm::Type::Int32Ty)));

  case Expr::DeclRefExprClass: return EmitDeclRefLValue(cast<DeclRefExpr>(E));
  case Expr::ParenExprClass:return EmitLValue(cast<ParenExpr>(E)->getSubExpr());
  case Expr::PreDefinedExprClass:
    return EmitPreDefinedLValue(cast<PreDefinedExpr>(E));
  case Expr::StringLiteralClass:
    return EmitStringLiteralLValue(cast<StringLiteral>(E));
    
  case Expr::UnaryOperatorClass: 
    return EmitUnaryOpLValue(cast<UnaryOperator>(E));
  case Expr::ArraySubscriptExprClass:
    return EmitArraySubscriptExpr(cast<ArraySubscriptExpr>(E));
  case Expr::OCUVectorElementExprClass:
    return EmitOCUVectorElementExpr(cast<OCUVectorElementExpr>(E));
  }
}

/// EmitLoadOfLValue - Given an expression that represents a value lvalue,
/// this method emits the address of the lvalue, then loads the result as an
/// rvalue, returning the rvalue.
RValue CodeGenFunction::EmitLoadOfLValue(LValue LV, QualType ExprType) {
  if (LV.isSimple()) {
    llvm::Value *Ptr = LV.getAddress();
    const llvm::Type *EltTy =
      cast<llvm::PointerType>(Ptr->getType())->getElementType();
    
    // Simple scalar l-value.
    if (EltTy->isFirstClassType())
      return RValue::get(Builder.CreateLoad(Ptr, "tmp"));
    
    assert(ExprType->isFunctionType() && "Unknown scalar value");
    return RValue::get(Ptr);
  }
  
  if (LV.isVectorElt()) {
    llvm::Value *Vec = Builder.CreateLoad(LV.getVectorAddr(), "tmp");
    return RValue::get(Builder.CreateExtractElement(Vec, LV.getVectorIdx(),
                                                    "vecext"));
  }

  // If this is a reference to a subset of the elements of a vector, either
  // shuffle the input or extract/insert them as appropriate.
  if (LV.isOCUVectorElt())
    return EmitLoadOfOCUElementLValue(LV, ExprType);
  
  assert(0 && "Bitfield ref not impl!");
}

// If this is a reference to a subset of the elements of a vector, either
// shuffle the input or extract/insert them as appropriate.
RValue CodeGenFunction::EmitLoadOfOCUElementLValue(LValue LV,
                                                   QualType ExprType) {
  llvm::Value *Vec = Builder.CreateLoad(LV.getOCUVectorAddr(), "tmp");
  
  unsigned EncFields = LV.getOCUVectorElts();
  
  // If the result of the expression is a non-vector type, we must be
  // extracting a single element.  Just codegen as an extractelement.
  const VectorType *ExprVT = ExprType->getAsVectorType();
  if (!ExprVT) {
    unsigned InIdx = OCUVectorElementExpr::getAccessedFieldNo(0, EncFields);
    llvm::Value *Elt = llvm::ConstantInt::get(llvm::Type::Int32Ty, InIdx);
    return RValue::get(Builder.CreateExtractElement(Vec, Elt, "tmp"));
  }
  
  // If the source and destination have the same number of elements, use a
  // vector shuffle instead of insert/extracts.
  unsigned NumResultElts = ExprVT->getNumElements();
  unsigned NumSourceElts =
    cast<llvm::VectorType>(Vec->getType())->getNumElements();
  
  if (NumResultElts == NumSourceElts) {
    llvm::SmallVector<llvm::Constant*, 4> Mask;
    for (unsigned i = 0; i != NumResultElts; ++i) {
      unsigned InIdx = OCUVectorElementExpr::getAccessedFieldNo(i, EncFields);
      Mask.push_back(llvm::ConstantInt::get(llvm::Type::Int32Ty, InIdx));
    }
    
    llvm::Value *MaskV = llvm::ConstantVector::get(&Mask[0], Mask.size());
    Vec = Builder.CreateShuffleVector(Vec,
                                      llvm::UndefValue::get(Vec->getType()),
                                      MaskV, "tmp");
    return RValue::get(Vec);
  }
  
  // Start out with an undef of the result type.
  llvm::Value *Result = llvm::UndefValue::get(ConvertType(ExprType));
  
  // Extract/Insert each element of the result.
  for (unsigned i = 0; i != NumResultElts; ++i) {
    unsigned InIdx = OCUVectorElementExpr::getAccessedFieldNo(i, EncFields);
    llvm::Value *Elt = llvm::ConstantInt::get(llvm::Type::Int32Ty, InIdx);
    Elt = Builder.CreateExtractElement(Vec, Elt, "tmp");
    
    llvm::Value *OutIdx = llvm::ConstantInt::get(llvm::Type::Int32Ty, i);
    Result = Builder.CreateInsertElement(Result, Elt, OutIdx, "tmp");
  }
  
  return RValue::get(Result);
}


RValue CodeGenFunction::EmitLoadOfLValue(const Expr *E) {
  return EmitLoadOfLValue(EmitLValue(E), E->getType());
}


/// EmitStoreThroughLValue - Store the specified rvalue into the specified
/// lvalue, where both are guaranteed to the have the same type, and that type
/// is 'Ty'.
void CodeGenFunction::EmitStoreThroughLValue(RValue Src, LValue Dst, 
                                             QualType Ty) {
  if (!Dst.isSimple()) {
    if (Dst.isVectorElt()) {
      // Read/modify/write the vector, inserting the new element.
      // FIXME: Volatility.
      llvm::Value *Vec = Builder.CreateLoad(Dst.getVectorAddr(), "tmp");
      Vec = Builder.CreateInsertElement(Vec, Src.getVal(),
                                        Dst.getVectorIdx(), "vecins");
      Builder.CreateStore(Vec, Dst.getVectorAddr());
      return;
    }
  
    // If this is an update of elements of a vector, insert them as appropriate.
    if (Dst.isOCUVectorElt())
      return EmitStoreThroughOCUComponentLValue(Src, Dst, Ty);
  
    assert(0 && "FIXME: Don't support store to bitfield yet");
  }
  
  llvm::Value *DstAddr = Dst.getAddress();
  assert(Src.isScalar() && "Can't emit an agg store with this method");
  // FIXME: Handle volatility etc.
  const llvm::Type *SrcTy = Src.getVal()->getType();
  const llvm::Type *AddrTy = 
    cast<llvm::PointerType>(DstAddr->getType())->getElementType();
  
  if (AddrTy != SrcTy)
    DstAddr = Builder.CreateBitCast(DstAddr, llvm::PointerType::get(SrcTy),
                                    "storetmp");
  Builder.CreateStore(Src.getVal(), DstAddr);
}

void CodeGenFunction::EmitStoreThroughOCUComponentLValue(RValue Src, LValue Dst, 
                                                         QualType Ty) {
  // This access turns into a read/modify/write of the vector.  Load the input
  // value now.
  llvm::Value *Vec = Builder.CreateLoad(Dst.getOCUVectorAddr(), "tmp");
  // FIXME: Volatility.
  unsigned EncFields = Dst.getOCUVectorElts();
  
  llvm::Value *SrcVal = Src.getVal();
  
  if (const VectorType *VTy = Ty->getAsVectorType()) {
    unsigned NumSrcElts = VTy->getNumElements();

    // Extract/Insert each element.
    for (unsigned i = 0; i != NumSrcElts; ++i) {
      llvm::Value *Elt = llvm::ConstantInt::get(llvm::Type::Int32Ty, i);
      Elt = Builder.CreateExtractElement(SrcVal, Elt, "tmp");
      
      unsigned Idx = OCUVectorElementExpr::getAccessedFieldNo(i, EncFields);
      llvm::Value *OutIdx = llvm::ConstantInt::get(llvm::Type::Int32Ty, Idx);
      Vec = Builder.CreateInsertElement(Vec, Elt, OutIdx, "tmp");
    }
  } else {
    // If the Src is a scalar (not a vector) it must be updating one element.
    unsigned InIdx = OCUVectorElementExpr::getAccessedFieldNo(0, EncFields);
    llvm::Value *Elt = llvm::ConstantInt::get(llvm::Type::Int32Ty, InIdx);
    Vec = Builder.CreateInsertElement(Vec, SrcVal, Elt, "tmp");
  }
  
  Builder.CreateStore(Vec, Dst.getOCUVectorAddr());
}


LValue CodeGenFunction::EmitDeclRefLValue(const DeclRefExpr *E) {
  const Decl *D = E->getDecl();
  if (isa<BlockVarDecl>(D) || isa<ParmVarDecl>(D)) {
    llvm::Value *V = LocalDeclMap[D];
    assert(V && "BlockVarDecl not entered in LocalDeclMap?");
    return LValue::MakeAddr(V);
  } else if (isa<FunctionDecl>(D) || isa<FileVarDecl>(D)) {
    return LValue::MakeAddr(CGM.GetAddrOfGlobalDecl(D));
  }
  assert(0 && "Unimp declref");
}

LValue CodeGenFunction::EmitUnaryOpLValue(const UnaryOperator *E) {
  // __extension__ doesn't affect lvalue-ness.
  if (E->getOpcode() == UnaryOperator::Extension)
    return EmitLValue(E->getSubExpr());
  
  assert(E->getOpcode() == UnaryOperator::Deref &&
         "'*' is the only unary operator that produces an lvalue");
  return LValue::MakeAddr(EmitExpr(E->getSubExpr()).getVal());
}

LValue CodeGenFunction::EmitStringLiteralLValue(const StringLiteral *E) {
  assert(!E->isWide() && "FIXME: Wide strings not supported yet!");
  const char *StrData = E->getStrData();
  unsigned Len = E->getByteLength();
  
  // FIXME: Can cache/reuse these within the module.
  llvm::Constant *C=llvm::ConstantArray::get(std::string(StrData, StrData+Len));
  
  // Create a global variable for this.
  C = new llvm::GlobalVariable(C->getType(), true, 
                               llvm::GlobalValue::InternalLinkage,
                               C, ".str", CurFn->getParent());
  llvm::Constant *Zero = llvm::Constant::getNullValue(llvm::Type::Int32Ty);
  llvm::Constant *Zeros[] = { Zero, Zero };
  C = llvm::ConstantExpr::getGetElementPtr(C, Zeros, 2);
  return LValue::MakeAddr(C);
}

LValue CodeGenFunction::EmitPreDefinedLValue(const PreDefinedExpr *E) {
  std::string FunctionName(CurFuncDecl->getName());
  std::string GlobalVarName;
  
  switch (E->getIdentType()) {
    default:
      assert(0 && "unknown pre-defined ident type");
    case PreDefinedExpr::Func:
      GlobalVarName = "__func__.";
      break;
    case PreDefinedExpr::Function:
      GlobalVarName = "__FUNCTION__.";
      break;
    case PreDefinedExpr::PrettyFunction:
      // FIXME:: Demangle C++ method names
      GlobalVarName = "__PRETTY_FUNCTION__.";
      break;
  }
  
  GlobalVarName += CurFuncDecl->getName();
  
  // FIXME: Can cache/reuse these within the module.
  llvm::Constant *C=llvm::ConstantArray::get(FunctionName);
  
  // Create a global variable for this.
  C = new llvm::GlobalVariable(C->getType(), true, 
                               llvm::GlobalValue::InternalLinkage,
                               C, GlobalVarName, CurFn->getParent());
  llvm::Constant *Zero = llvm::Constant::getNullValue(llvm::Type::Int32Ty);
  llvm::Constant *Zeros[] = { Zero, Zero };
  C = llvm::ConstantExpr::getGetElementPtr(C, Zeros, 2);
  return LValue::MakeAddr(C);
}

LValue CodeGenFunction::EmitArraySubscriptExpr(const ArraySubscriptExpr *E) {
  // The index must always be an integer, which is not an aggregate.  Emit it.
  llvm::Value *Idx = EmitExpr(E->getIdx()).getVal();
  
  // If the base is a vector type, then we are forming a vector element lvalue
  // with this subscript.
  if (E->getLHS()->getType()->isVectorType()) {
    // Emit the vector as an lvalue to get its address.
    LValue LHS = EmitLValue(E->getLHS());
    assert(LHS.isSimple() && "Can only subscript lvalue vectors here!");
    // FIXME: This should properly sign/zero/extend or truncate Idx to i32.
    return LValue::MakeVectorElt(LHS.getAddress(), Idx);
  }
  
  // The base must be a pointer, which is not an aggregate.  Emit it.
  llvm::Value *Base = EmitExpr(E->getBase()).getVal();
  
  // Extend or truncate the index type to 32 or 64-bits.
  QualType IdxTy  = E->getIdx()->getType();
  bool IdxSigned = IdxTy->isSignedIntegerType();
  unsigned IdxBitwidth = cast<llvm::IntegerType>(Idx->getType())->getBitWidth();
  if (IdxBitwidth != LLVMPointerWidth)
    Idx = Builder.CreateIntCast(Idx, llvm::IntegerType::get(LLVMPointerWidth),
                                IdxSigned, "idxprom");

  // We know that the pointer points to a type of the correct size, unless the
  // size is a VLA.
  if (!E->getType()->isConstantSizeType(getContext()))
    assert(0 && "VLA idx not implemented");
  return LValue::MakeAddr(Builder.CreateGEP(Base, Idx, "arrayidx"));
}

LValue CodeGenFunction::
EmitOCUVectorElementExpr(const OCUVectorElementExpr *E) {
  // Emit the base vector as an l-value.
  LValue Base = EmitLValue(E->getBase());
  assert(Base.isSimple() && "Can only subscript lvalue vectors here!");

  return LValue::MakeOCUVectorElt(Base.getAddress(), 
                                  E->getEncodedElementAccess());
}

//===--------------------------------------------------------------------===//
//                             Expression Emission
//===--------------------------------------------------------------------===//

RValue CodeGenFunction::EmitExpr(const Expr *E) {
  assert(E && !hasAggregateLLVMType(E->getType()) &&
         "Invalid scalar expression to emit");
  
  switch (E->getStmtClass()) {
  default:
    fprintf(stderr, "Unimplemented expr!\n");
    E->dump();
    return RValue::get(llvm::UndefValue::get(llvm::Type::Int32Ty));
    
  // l-values.
  case Expr::DeclRefExprClass:
    // DeclRef's of EnumConstantDecl's are simple rvalues.
    if (const EnumConstantDecl *EC = 
          dyn_cast<EnumConstantDecl>(cast<DeclRefExpr>(E)->getDecl()))
      return RValue::get(llvm::ConstantInt::get(EC->getInitVal()));
    return EmitLoadOfLValue(E);
  case Expr::ArraySubscriptExprClass:
    return EmitArraySubscriptExprRV(cast<ArraySubscriptExpr>(E));
  case Expr::OCUVectorElementExprClass:
    return EmitLoadOfLValue(E);
  case Expr::PreDefinedExprClass:
  case Expr::StringLiteralClass:
    return RValue::get(EmitLValue(E).getAddress());
    
  // Leaf expressions.
  case Expr::IntegerLiteralClass:
    return EmitIntegerLiteral(cast<IntegerLiteral>(E)); 
  case Expr::FloatingLiteralClass:
    return EmitFloatingLiteral(cast<FloatingLiteral>(E));
  case Expr::CharacterLiteralClass:
    return EmitCharacterLiteral(cast<CharacterLiteral>(E));
  case Expr::TypesCompatibleExprClass:
    return EmitTypesCompatibleExpr(cast<TypesCompatibleExpr>(E));
    
  // Operators.  
  case Expr::ParenExprClass:
    return EmitExpr(cast<ParenExpr>(E)->getSubExpr());
  case Expr::UnaryOperatorClass:
    return EmitUnaryOperator(cast<UnaryOperator>(E));
  case Expr::SizeOfAlignOfTypeExprClass:
    return EmitSizeAlignOf(cast<SizeOfAlignOfTypeExpr>(E)->getArgumentType(),
                           E->getType(),
                           cast<SizeOfAlignOfTypeExpr>(E)->isSizeOf());
  case Expr::ImplicitCastExprClass:
    return EmitImplicitCastExpr(cast<ImplicitCastExpr>(E));
  case Expr::CastExprClass: 
    return EmitCastExpr(cast<CastExpr>(E)->getSubExpr(), E->getType());
  case Expr::CallExprClass:
    return EmitCallExpr(cast<CallExpr>(E));
  case Expr::BinaryOperatorClass:
    return EmitBinaryOperator(cast<BinaryOperator>(E));
  
  case Expr::ConditionalOperatorClass:
    return EmitConditionalOperator(cast<ConditionalOperator>(E));
  case Expr::ChooseExprClass:
    return EmitChooseExpr(cast<ChooseExpr>(E));
  }
}

RValue CodeGenFunction::EmitIntegerLiteral(const IntegerLiteral *E) {
  return RValue::get(llvm::ConstantInt::get(E->getValue()));
}
RValue CodeGenFunction::EmitFloatingLiteral(const FloatingLiteral *E) {
  return RValue::get(llvm::ConstantFP::get(ConvertType(E->getType()),
                                           E->getValue()));
}
RValue CodeGenFunction::EmitCharacterLiteral(const CharacterLiteral *E) {
  return RValue::get(llvm::ConstantInt::get(ConvertType(E->getType()),
                                            E->getValue()));
}

RValue CodeGenFunction::EmitTypesCompatibleExpr(const TypesCompatibleExpr *E) {
  return RValue::get(llvm::ConstantInt::get(ConvertType(E->getType()),
                                            E->typesAreCompatible()));
}

/// EmitChooseExpr - Implement __builtin_choose_expr.
RValue CodeGenFunction::EmitChooseExpr(const ChooseExpr *E) {
  llvm::APSInt CondVal(32);
  bool IsConst = E->getCond()->isIntegerConstantExpr(CondVal, getContext());
  assert(IsConst && "Condition of choose expr must be i-c-e"); IsConst=IsConst;
  
  // Emit the LHS or RHS as appropriate.
  return EmitExpr(CondVal != 0 ? E->getLHS() : E->getRHS());
}


RValue CodeGenFunction::EmitArraySubscriptExprRV(const ArraySubscriptExpr *E) {
  // Emit subscript expressions in rvalue context's.  For most cases, this just
  // loads the lvalue formed by the subscript expr.  However, we have to be
  // careful, because the base of a vector subscript is occasionally an rvalue,
  // so we can't get it as an lvalue.
  if (!E->getBase()->getType()->isVectorType())
    return EmitLoadOfLValue(E);

  // Handle the vector case.  The base must be a vector, the index must be an
  // integer value.
  llvm::Value *Base = EmitExpr(E->getBase()).getVal();
  llvm::Value *Idx  = EmitExpr(E->getIdx()).getVal();
  
  // FIXME: Convert Idx to i32 type.
  
  return RValue::get(Builder.CreateExtractElement(Base, Idx, "vecext"));
}

// EmitCastExpr - Emit code for an explicit or implicit cast.  Implicit casts
// have to handle a more broad range of conversions than explicit casts, as they
// handle things like function to ptr-to-function decay etc.
RValue CodeGenFunction::EmitCastExpr(const Expr *Op, QualType DestTy) {
  RValue Src = EmitExpr(Op);
  
  // If the destination is void, just evaluate the source.
  if (DestTy->isVoidType())
    return RValue::getAggregate(0);
  
  return EmitConversion(Src, Op->getType(), DestTy);
}

/// EmitImplicitCastExpr - Implicit casts are the same as normal casts, but also
/// handle things like function to pointer-to-function decay, and array to
/// pointer decay.
RValue CodeGenFunction::EmitImplicitCastExpr(const ImplicitCastExpr *E) {
  const Expr *Op = E->getSubExpr();
  QualType OpTy = Op->getType().getCanonicalType();
  
  // If this is due to array->pointer conversion, emit the array expression as
  // an l-value.
  if (isa<ArrayType>(OpTy)) {
    // FIXME: For now we assume that all source arrays map to LLVM arrays.  This
    // will not true when we add support for VLAs.
    llvm::Value *V = EmitLValue(Op).getAddress();  // Bitfields can't be arrays.
    
    assert(isa<llvm::PointerType>(V->getType()) &&
           isa<llvm::ArrayType>(cast<llvm::PointerType>(V->getType())
                                ->getElementType()) &&
           "Doesn't support VLAs yet!");
    llvm::Constant *Idx0 = llvm::ConstantInt::get(llvm::Type::Int32Ty, 0);
    return RValue::get(Builder.CreateGEP(V, Idx0, Idx0, "arraydecay"));
  }
  
  return EmitCastExpr(Op, E->getType());
}

RValue CodeGenFunction::EmitCallExpr(const CallExpr *E) {
  if (const ImplicitCastExpr *IcExpr = 
      dyn_cast<const ImplicitCastExpr>(E->getCallee()))
    if (const DeclRefExpr *DRExpr = 
        dyn_cast<const DeclRefExpr>(IcExpr->getSubExpr()))
      if (const FunctionDecl *FDecl = 
          dyn_cast<const FunctionDecl>(DRExpr->getDecl()))
        if (unsigned builtinID = FDecl->getIdentifier()->getBuiltinID())
          return EmitBuiltinExpr(builtinID, E);
        
  llvm::Value *Callee = EmitExpr(E->getCallee()).getVal();
  
  // The callee type will always be a pointer to function type, get the function
  // type.
  QualType CalleeTy = E->getCallee()->getType();
  CalleeTy = cast<PointerType>(CalleeTy.getCanonicalType())->getPointeeType();
  
  // Get information about the argument types.
  FunctionTypeProto::arg_type_iterator ArgTyIt = 0, ArgTyEnd = 0;
  
  // Calling unprototyped functions provides no argument info.
  if (const FunctionTypeProto *FTP = dyn_cast<FunctionTypeProto>(CalleeTy)) {
    ArgTyIt  = FTP->arg_type_begin();
    ArgTyEnd = FTP->arg_type_end();
  }
  
  llvm::SmallVector<llvm::Value*, 16> Args;
  
  // Handle struct-return functions by passing a pointer to the location that
  // we would like to return into.
  if (hasAggregateLLVMType(E->getType())) {
    // Create a temporary alloca to hold the result of the call. :(
    Args.push_back(CreateTempAlloca(ConvertType(E->getType())));
    // FIXME: set the stret attribute on the argument.
  }
  
  for (unsigned i = 0, e = E->getNumArgs(); i != e; ++i) {
    QualType ArgTy = E->getArg(i)->getType();
    RValue ArgVal = EmitExpr(E->getArg(i));
    
    // If this argument has prototype information, convert it.
    if (ArgTyIt != ArgTyEnd) {
      ArgVal = EmitConversion(ArgVal, ArgTy, *ArgTyIt++);
    } else {
      // Otherwise, if passing through "..." or to a function with no prototype,
      // perform the "default argument promotions" (C99 6.5.2.2p6), which
      // includes the usual unary conversions, but also promotes float to
      // double.
      if (const BuiltinType *BT = 
          dyn_cast<BuiltinType>(ArgTy.getCanonicalType())) {
        if (BT->getKind() == BuiltinType::Float)
          ArgVal = RValue::get(Builder.CreateFPExt(ArgVal.getVal(),
                                                   llvm::Type::DoubleTy,"tmp"));
      }
    }
    
    
    if (ArgVal.isScalar())
      Args.push_back(ArgVal.getVal());
    else  // Pass by-address.  FIXME: Set attribute bit on call.
      Args.push_back(ArgVal.getAggregateAddr());
  }
  
  llvm::Value *V = Builder.CreateCall(Callee, &Args[0], &Args[0]+Args.size());
  if (V->getType() != llvm::Type::VoidTy)
    V->setName("call");
  else if (hasAggregateLLVMType(E->getType()))
    // Struct return.
    return RValue::getAggregate(Args[0]);
      
  return RValue::get(V);
}


//===----------------------------------------------------------------------===//
//                           Unary Operator Emission
//===----------------------------------------------------------------------===//

RValue CodeGenFunction::EmitUnaryOperator(const UnaryOperator *E) {
  switch (E->getOpcode()) {
  default:
    printf("Unimplemented unary expr!\n");
    E->dump();
    return RValue::get(llvm::UndefValue::get(llvm::Type::Int32Ty));
  case UnaryOperator::PostInc:
  case UnaryOperator::PostDec:
  case UnaryOperator::PreInc :
  case UnaryOperator::PreDec : return EmitUnaryIncDec(E);
  case UnaryOperator::AddrOf : return EmitUnaryAddrOf(E);
  case UnaryOperator::Deref  : return EmitLoadOfLValue(E);
  case UnaryOperator::Plus   : return EmitUnaryPlus(E);
  case UnaryOperator::Minus  : return EmitUnaryMinus(E);
  case UnaryOperator::Not    : return EmitUnaryNot(E);
  case UnaryOperator::LNot   : return EmitUnaryLNot(E);
  case UnaryOperator::SizeOf :
    return EmitSizeAlignOf(E->getSubExpr()->getType(), E->getType(), true);
  case UnaryOperator::AlignOf :
    return EmitSizeAlignOf(E->getSubExpr()->getType(), E->getType(), false);
  // FIXME: real/imag
  case UnaryOperator::Extension: return EmitExpr(E->getSubExpr());
  }
}

RValue CodeGenFunction::EmitUnaryIncDec(const UnaryOperator *E) {
  LValue LV = EmitLValue(E->getSubExpr());
  RValue InVal = EmitLoadOfLValue(LV, E->getSubExpr()->getType());
  
  // We know the operand is real or pointer type, so it must be an LLVM scalar.
  assert(InVal.isScalar() && "Unknown thing to increment");
  llvm::Value *InV = InVal.getVal();

  int AmountVal = 1;
  if (E->getOpcode() == UnaryOperator::PreDec ||
      E->getOpcode() == UnaryOperator::PostDec)
    AmountVal = -1;
  
  llvm::Value *NextVal;
  if (isa<llvm::IntegerType>(InV->getType())) {
    NextVal = llvm::ConstantInt::get(InV->getType(), AmountVal);
    NextVal = Builder.CreateAdd(InV, NextVal, AmountVal == 1 ? "inc" : "dec");
  } else if (InV->getType()->isFloatingPoint()) {
    NextVal = llvm::ConstantFP::get(InV->getType(), AmountVal);
    NextVal = Builder.CreateAdd(InV, NextVal, AmountVal == 1 ? "inc" : "dec");
  } else {
    // FIXME: This is not right for pointers to VLA types.
    assert(isa<llvm::PointerType>(InV->getType()));
    NextVal = llvm::ConstantInt::get(llvm::Type::Int32Ty, AmountVal);
    NextVal = Builder.CreateGEP(InV, NextVal, AmountVal == 1 ? "inc" : "dec");
  }

  RValue NextValToStore = RValue::get(NextVal);

  // Store the updated result through the lvalue.
  EmitStoreThroughLValue(NextValToStore, LV, E->getSubExpr()->getType());
                         
  // If this is a postinc, return the value read from memory, otherwise use the
  // updated value.
  if (E->getOpcode() == UnaryOperator::PreDec ||
      E->getOpcode() == UnaryOperator::PreInc)
    return NextValToStore;
  else
    return InVal;
}

/// C99 6.5.3.2
RValue CodeGenFunction::EmitUnaryAddrOf(const UnaryOperator *E) {
  // The address of the operand is just its lvalue.  It cannot be a bitfield.
  return RValue::get(EmitLValue(E->getSubExpr()).getAddress());
}

RValue CodeGenFunction::EmitUnaryPlus(const UnaryOperator *E) {
  assert(E->getType().getCanonicalType() == 
         E->getSubExpr()->getType().getCanonicalType() && "Bad unary plus!");
  // Unary plus just returns its value.
  return EmitExpr(E->getSubExpr());
}

RValue CodeGenFunction::EmitUnaryMinus(const UnaryOperator *E) {
  assert(E->getType().getCanonicalType() == 
         E->getSubExpr()->getType().getCanonicalType() && "Bad unary minus!");

  // Unary minus performs promotions, then negates its arithmetic operand.
  RValue V = EmitExpr(E->getSubExpr());
  
  if (V.isScalar())
    return RValue::get(Builder.CreateNeg(V.getVal(), "neg"));
  
  assert(0 && "FIXME: This doesn't handle complex operands yet");
}

RValue CodeGenFunction::EmitUnaryNot(const UnaryOperator *E) {
  // Unary not performs promotions, then complements its integer operand.
  RValue V = EmitExpr(E->getSubExpr());
  
  if (V.isScalar())
    return RValue::get(Builder.CreateNot(V.getVal(), "neg"));
                      
  assert(0 && "FIXME: This doesn't handle integer complex operands yet (GNU)");
}


/// C99 6.5.3.3
RValue CodeGenFunction::EmitUnaryLNot(const UnaryOperator *E) {
  // Compare operand to zero.
  llvm::Value *BoolVal = EvaluateExprAsBool(E->getSubExpr());
  
  // Invert value.
  // TODO: Could dynamically modify easy computations here.  For example, if
  // the operand is an icmp ne, turn into icmp eq.
  BoolVal = Builder.CreateNot(BoolVal, "lnot");
  
  // ZExt result to int.
  return RValue::get(Builder.CreateZExt(BoolVal, LLVMIntTy, "lnot.ext"));
}

/// EmitSizeAlignOf - Return the size or alignment of the 'TypeToSize' type as
/// an integer (RetType).
RValue CodeGenFunction::EmitSizeAlignOf(QualType TypeToSize,
                                        QualType RetType, bool isSizeOf) {
  /// FIXME: This doesn't handle VLAs yet!
  std::pair<uint64_t, unsigned> Info =
    getContext().getTypeInfo(TypeToSize, SourceLocation());
  
  uint64_t Val = isSizeOf ? Info.first : Info.second;
  Val /= 8;  // Return size in bytes, not bits.
  
  assert(RetType->isIntegerType() && "Result type must be an integer!");

  unsigned ResultWidth = getContext().getTypeSize(RetType, SourceLocation());
  return RValue::get(llvm::ConstantInt::get(llvm::APInt(ResultWidth, Val)));
}


//===--------------------------------------------------------------------===//
//                         Binary Operator Emission
//===--------------------------------------------------------------------===//


/// EmitCompoundAssignmentOperands - Compound assignment operations (like +=)
/// are strange in that the result of the operation is not the same type as the
/// intermediate computation.  This function emits the LHS and RHS operands of
/// the compound assignment, promoting them to their common computation type.
///
/// Since the LHS is an lvalue, and the result is stored back through it, we
/// return the lvalue as well as the LHS/RHS rvalues.  On return, the LHS and
/// RHS values are both in the computation type for the operator.
void CodeGenFunction::
EmitCompoundAssignmentOperands(const CompoundAssignOperator *E,
                               LValue &LHSLV, RValue &LHS, RValue &RHS) {
  LHSLV = EmitLValue(E->getLHS());
  
  // Load the LHS and RHS operands.
  QualType LHSTy = E->getLHS()->getType();
  LHS = EmitLoadOfLValue(LHSLV, LHSTy);
  RHS = EmitExpr(E->getRHS());
  QualType RHSTy = E->getRHS()->getType();
  
  // Convert the LHS and RHS to the common evaluation type.
  LHS = EmitConversion(LHS, LHSTy, E->getComputationType());
  RHS = EmitConversion(RHS, RHSTy, E->getComputationType());
}

/// EmitCompoundAssignmentResult - Given a result value in the computation type,
/// truncate it down to the actual result type, store it through the LHS lvalue,
/// and return it.
RValue CodeGenFunction::
EmitCompoundAssignmentResult(const CompoundAssignOperator *E,
                             LValue LHSLV, RValue ResV) {
  
  // Truncate back to the destination type.
  if (E->getComputationType() != E->getType())
    ResV = EmitConversion(ResV, E->getComputationType(), E->getType());
  
  // Store the result value into the LHS.
  EmitStoreThroughLValue(ResV, LHSLV, E->getType());
  
  // Return the result.
  return ResV;
}


RValue CodeGenFunction::EmitBinaryOperator(const BinaryOperator *E) {
  RValue LHS, RHS;
  switch (E->getOpcode()) {
  default:
    fprintf(stderr, "Unimplemented binary expr!\n");
    E->dump();
    return RValue::get(llvm::UndefValue::get(llvm::Type::Int32Ty));
  case BinaryOperator::Mul:
    LHS = EmitExpr(E->getLHS());
    RHS = EmitExpr(E->getRHS());
    return EmitMul(LHS, RHS, E->getType());
  case BinaryOperator::Div:
    LHS = EmitExpr(E->getLHS());
    RHS = EmitExpr(E->getRHS());
    return EmitDiv(LHS, RHS, E->getType());
  case BinaryOperator::Rem:
    LHS = EmitExpr(E->getLHS());
    RHS = EmitExpr(E->getRHS());
    return EmitRem(LHS, RHS, E->getType());
  case BinaryOperator::Add:
    LHS = EmitExpr(E->getLHS());
    RHS = EmitExpr(E->getRHS());
    if (!E->getType()->isPointerType())
      return EmitAdd(LHS, RHS, E->getType());
      
    return EmitPointerAdd(LHS, E->getLHS()->getType(),
                          RHS, E->getRHS()->getType(), E->getType());
  case BinaryOperator::Sub:
    LHS = EmitExpr(E->getLHS());
    RHS = EmitExpr(E->getRHS());

    if (!E->getLHS()->getType()->isPointerType())
      return EmitSub(LHS, RHS, E->getType());
      
    return EmitPointerSub(LHS, E->getLHS()->getType(),
                          RHS, E->getRHS()->getType(), E->getType());
  case BinaryOperator::Shl:
    LHS = EmitExpr(E->getLHS());
    RHS = EmitExpr(E->getRHS());
    return EmitShl(LHS, RHS, E->getType());
  case BinaryOperator::Shr:
    LHS = EmitExpr(E->getLHS());
    RHS = EmitExpr(E->getRHS());
    return EmitShr(LHS, RHS, E->getType());
  case BinaryOperator::And:
    LHS = EmitExpr(E->getLHS());
    RHS = EmitExpr(E->getRHS());
    return EmitAnd(LHS, RHS, E->getType());
  case BinaryOperator::Xor:
    LHS = EmitExpr(E->getLHS());
    RHS = EmitExpr(E->getRHS());
    return EmitXor(LHS, RHS, E->getType());
  case BinaryOperator::Or :
    LHS = EmitExpr(E->getLHS());
    RHS = EmitExpr(E->getRHS());
    return EmitOr(LHS, RHS, E->getType());
  case BinaryOperator::LAnd: return EmitBinaryLAnd(E);
  case BinaryOperator::LOr: return EmitBinaryLOr(E);
  case BinaryOperator::LT:
    return EmitBinaryCompare(E, llvm::ICmpInst::ICMP_ULT,
                             llvm::ICmpInst::ICMP_SLT,
                             llvm::FCmpInst::FCMP_OLT);
  case BinaryOperator::GT:
    return EmitBinaryCompare(E, llvm::ICmpInst::ICMP_UGT,
                             llvm::ICmpInst::ICMP_SGT,
                             llvm::FCmpInst::FCMP_OGT);
  case BinaryOperator::LE:
    return EmitBinaryCompare(E, llvm::ICmpInst::ICMP_ULE,
                             llvm::ICmpInst::ICMP_SLE,
                             llvm::FCmpInst::FCMP_OLE);
  case BinaryOperator::GE:
    return EmitBinaryCompare(E, llvm::ICmpInst::ICMP_UGE,
                             llvm::ICmpInst::ICMP_SGE,
                             llvm::FCmpInst::FCMP_OGE);
  case BinaryOperator::EQ:
    return EmitBinaryCompare(E, llvm::ICmpInst::ICMP_EQ,
                             llvm::ICmpInst::ICMP_EQ,
                             llvm::FCmpInst::FCMP_OEQ);
  case BinaryOperator::NE:
    return EmitBinaryCompare(E, llvm::ICmpInst::ICMP_NE,
                             llvm::ICmpInst::ICMP_NE, 
                             llvm::FCmpInst::FCMP_UNE);
  case BinaryOperator::Assign:
    return EmitBinaryAssign(E);
    
  case BinaryOperator::MulAssign: {
    const CompoundAssignOperator *CAO = cast<CompoundAssignOperator>(E);
    LValue LHSLV;
    EmitCompoundAssignmentOperands(CAO, LHSLV, LHS, RHS);
    LHS = EmitMul(LHS, RHS, CAO->getComputationType());
    return EmitCompoundAssignmentResult(CAO, LHSLV, LHS);
  }
  case BinaryOperator::DivAssign: {
    const CompoundAssignOperator *CAO = cast<CompoundAssignOperator>(E);
    LValue LHSLV;
    EmitCompoundAssignmentOperands(CAO, LHSLV, LHS, RHS);
    LHS = EmitDiv(LHS, RHS, CAO->getComputationType());
    return EmitCompoundAssignmentResult(CAO, LHSLV, LHS);
  }
  case BinaryOperator::RemAssign: {
    const CompoundAssignOperator *CAO = cast<CompoundAssignOperator>(E);
    LValue LHSLV;
    EmitCompoundAssignmentOperands(CAO, LHSLV, LHS, RHS);
    LHS = EmitRem(LHS, RHS, CAO->getComputationType());
    return EmitCompoundAssignmentResult(CAO, LHSLV, LHS);
  }
  case BinaryOperator::AddAssign: {
    const CompoundAssignOperator *CAO = cast<CompoundAssignOperator>(E);
    LValue LHSLV;
    EmitCompoundAssignmentOperands(CAO, LHSLV, LHS, RHS);
    LHS = EmitAdd(LHS, RHS, CAO->getComputationType());
    return EmitCompoundAssignmentResult(CAO, LHSLV, LHS);
  }
  case BinaryOperator::SubAssign: {
    const CompoundAssignOperator *CAO = cast<CompoundAssignOperator>(E);
    LValue LHSLV;
    EmitCompoundAssignmentOperands(CAO, LHSLV, LHS, RHS);
    LHS = EmitSub(LHS, RHS, CAO->getComputationType());
    return EmitCompoundAssignmentResult(CAO, LHSLV, LHS);
  }
  case BinaryOperator::ShlAssign: {
    const CompoundAssignOperator *CAO = cast<CompoundAssignOperator>(E);
    LValue LHSLV;
    EmitCompoundAssignmentOperands(CAO, LHSLV, LHS, RHS);
    LHS = EmitShl(LHS, RHS, CAO->getComputationType());
    return EmitCompoundAssignmentResult(CAO, LHSLV, LHS);
  }
  case BinaryOperator::ShrAssign: {
    const CompoundAssignOperator *CAO = cast<CompoundAssignOperator>(E);
    LValue LHSLV;
    EmitCompoundAssignmentOperands(CAO, LHSLV, LHS, RHS);
    LHS = EmitShr(LHS, RHS, CAO->getComputationType());
    return EmitCompoundAssignmentResult(CAO, LHSLV, LHS);
  }
  case BinaryOperator::AndAssign: {
    const CompoundAssignOperator *CAO = cast<CompoundAssignOperator>(E);
    LValue LHSLV;
    EmitCompoundAssignmentOperands(CAO, LHSLV, LHS, RHS);
    LHS = EmitAnd(LHS, RHS, CAO->getComputationType());
    return EmitCompoundAssignmentResult(CAO, LHSLV, LHS);
  }
  case BinaryOperator::OrAssign: {
    const CompoundAssignOperator *CAO = cast<CompoundAssignOperator>(E);
    LValue LHSLV;
    EmitCompoundAssignmentOperands(CAO, LHSLV, LHS, RHS);
    LHS = EmitOr(LHS, RHS, CAO->getComputationType());
    return EmitCompoundAssignmentResult(CAO, LHSLV, LHS);
  }
  case BinaryOperator::XorAssign: {
    const CompoundAssignOperator *CAO = cast<CompoundAssignOperator>(E);
    LValue LHSLV;
    EmitCompoundAssignmentOperands(CAO, LHSLV, LHS, RHS);
    LHS = EmitXor(LHS, RHS, CAO->getComputationType());
    return EmitCompoundAssignmentResult(CAO, LHSLV, LHS);
  }
  case BinaryOperator::Comma: return EmitBinaryComma(E);
  }
}

RValue CodeGenFunction::EmitMul(RValue LHS, RValue RHS, QualType ResTy) {
  return RValue::get(Builder.CreateMul(LHS.getVal(), RHS.getVal(), "mul"));
  
#if 0
  // Otherwise, this must be a complex number.
  llvm::Value *LHSR, *LHSI, *RHSR, *RHSI;

  EmitLoadOfComplex(LHS, LHSR, LHSI);
  EmitLoadOfComplex(RHS, RHSR, RHSI);
  
  llvm::Value *ResRl = Builder.CreateMul(LHSR, RHSR, "mul.rl");
  llvm::Value *ResRr = Builder.CreateMul(LHSI, RHSI, "mul.rr");
  llvm::Value *ResR = Builder.CreateSub(ResRl, ResRr, "mul.r");

  llvm::Value *ResIl = Builder.CreateMul(LHSI, RHSR, "mul.il");
  llvm::Value *ResIr = Builder.CreateMul(LHSR, RHSI, "mul.ir");
  llvm::Value *ResI = Builder.CreateAdd(ResIl, ResIr, "mul.i");
  
  llvm::Value *Res = CreateTempAlloca(ConvertType(ResTy));
  EmitStoreOfComplex(ResR, ResI, Res);
  return RValue::getAggregate(Res);
#endif
}

RValue CodeGenFunction::EmitDiv(RValue LHS, RValue RHS, QualType ResTy) {
  if (LHS.isScalar()) {
    llvm::Value *RV;
    if (LHS.getVal()->getType()->isFloatingPoint())
      RV = Builder.CreateFDiv(LHS.getVal(), RHS.getVal(), "div");
    else if (ResTy->isUnsignedIntegerType())
      RV = Builder.CreateUDiv(LHS.getVal(), RHS.getVal(), "div");
    else
      RV = Builder.CreateSDiv(LHS.getVal(), RHS.getVal(), "div");
    return RValue::get(RV);
  }
  assert(0 && "FIXME: This doesn't handle complex operands yet");
}

RValue CodeGenFunction::EmitRem(RValue LHS, RValue RHS, QualType ResTy) {
  if (LHS.isScalar()) {
    llvm::Value *RV;
    // Rem in C can't be a floating point type: C99 6.5.5p2.
    if (ResTy->isUnsignedIntegerType())
      RV = Builder.CreateURem(LHS.getVal(), RHS.getVal(), "rem");
    else
      RV = Builder.CreateSRem(LHS.getVal(), RHS.getVal(), "rem");
    return RValue::get(RV);
  }
  
  assert(0 && "FIXME: This doesn't handle complex operands yet");
}

RValue CodeGenFunction::EmitAdd(RValue LHS, RValue RHS, QualType ResTy) {
  return RValue::get(Builder.CreateAdd(LHS.getVal(), RHS.getVal(), "add"));
}

RValue CodeGenFunction::EmitPointerAdd(RValue LHS, QualType LHSTy,
                                       RValue RHS, QualType RHSTy,
                                       QualType ResTy) {
  llvm::Value *LHSValue = LHS.getVal();
  llvm::Value *RHSValue = RHS.getVal();
  if (LHSTy->isPointerType()) {
    // pointer + int
    return RValue::get(Builder.CreateGEP(LHSValue, RHSValue, "add.ptr"));
  } else {
    // int + pointer
    return RValue::get(Builder.CreateGEP(RHSValue, LHSValue, "add.ptr"));
  }
}

RValue CodeGenFunction::EmitSub(RValue LHS, RValue RHS, QualType ResTy) {
  if (LHS.isScalar())
    return RValue::get(Builder.CreateSub(LHS.getVal(), RHS.getVal(), "sub"));
  
  assert(0 && "FIXME: This doesn't handle complex operands yet");
}

RValue CodeGenFunction::EmitPointerSub(RValue LHS, QualType LHSTy,
                                       RValue RHS, QualType RHSTy,
                                       QualType ResTy) {
  llvm::Value *LHSValue = LHS.getVal();
  llvm::Value *RHSValue = RHS.getVal();
  if (const PointerType *RHSPtrType =
        dyn_cast<PointerType>(RHSTy.getTypePtr())) {
    // pointer - pointer
    const PointerType *LHSPtrType = cast<PointerType>(LHSTy.getTypePtr());
    QualType LHSElementType = LHSPtrType->getPointeeType();
    assert(LHSElementType == RHSPtrType->getPointeeType() &&
      "can't subtract pointers with differing element types");
    uint64_t ElementSize = getContext().getTypeSize(LHSElementType,
                                                    SourceLocation()) / 8;
    const llvm::Type *ResultType = ConvertType(ResTy);
    llvm::Value *CastLHS = Builder.CreatePtrToInt(LHSValue, ResultType,
                                                  "sub.ptr.lhs.cast");
    llvm::Value *CastRHS = Builder.CreatePtrToInt(RHSValue, ResultType,
                                                  "sub.ptr.rhs.cast");
    llvm::Value *BytesBetween = Builder.CreateSub(CastLHS, CastRHS,
                                                  "sub.ptr.sub");
    
    // HACK: LLVM doesn't have an divide instruction that 'knows' there is no
    // remainder.  As such, we handle common power-of-two cases here to generate
    // better code.
    if (llvm::isPowerOf2_64(ElementSize)) {
      llvm::Value *ShAmt =
        llvm::ConstantInt::get(ResultType, llvm::Log2_64(ElementSize));
      return RValue::get(Builder.CreateAShr(BytesBetween, ShAmt,"sub.ptr.shr"));
    } else {
      // Otherwise, do a full sdiv.
      llvm::Value *BytesPerElement =
        llvm::ConstantInt::get(ResultType, ElementSize);
      return RValue::get(Builder.CreateSDiv(BytesBetween, BytesPerElement,
                                            "sub.ptr.div"));
    }
  } else {
    // pointer - int
    llvm::Value *NegatedRHS = Builder.CreateNeg(RHSValue, "sub.ptr.neg");
    return RValue::get(Builder.CreateGEP(LHSValue, NegatedRHS, "sub.ptr"));
  }
}

RValue CodeGenFunction::EmitShl(RValue LHSV, RValue RHSV, QualType ResTy) {
  llvm::Value *LHS = LHSV.getVal(), *RHS = RHSV.getVal();
  
  // LLVM requires the LHS and RHS to be the same type, promote or truncate the
  // RHS to the same size as the LHS.
  if (LHS->getType() != RHS->getType())
    RHS = Builder.CreateIntCast(RHS, LHS->getType(), false, "sh_prom");
  
  return RValue::get(Builder.CreateShl(LHS, RHS, "shl"));
}

RValue CodeGenFunction::EmitShr(RValue LHSV, RValue RHSV, QualType ResTy) {
  llvm::Value *LHS = LHSV.getVal(), *RHS = RHSV.getVal();
  
  // LLVM requires the LHS and RHS to be the same type, promote or truncate the
  // RHS to the same size as the LHS.
  if (LHS->getType() != RHS->getType())
    RHS = Builder.CreateIntCast(RHS, LHS->getType(), false, "sh_prom");
  
  if (ResTy->isUnsignedIntegerType())
    return RValue::get(Builder.CreateLShr(LHS, RHS, "shr"));
  else
    return RValue::get(Builder.CreateAShr(LHS, RHS, "shr"));
}

RValue CodeGenFunction::EmitBinaryCompare(const BinaryOperator *E,
                                          unsigned UICmpOpc, unsigned SICmpOpc,
                                          unsigned FCmpOpc) {
  RValue LHS = EmitExpr(E->getLHS());
  RValue RHS = EmitExpr(E->getRHS());

  llvm::Value *Result;
  if (LHS.isScalar()) {
    if (LHS.getVal()->getType()->isFloatingPoint()) {
      Result = Builder.CreateFCmp((llvm::FCmpInst::Predicate)FCmpOpc,
                                  LHS.getVal(), RHS.getVal(), "cmp");
    } else if (E->getLHS()->getType()->isUnsignedIntegerType()) {
      // FIXME: This check isn't right for "unsigned short < int" where ushort
      // promotes to int and does a signed compare.
      Result = Builder.CreateICmp((llvm::ICmpInst::Predicate)UICmpOpc,
                                  LHS.getVal(), RHS.getVal(), "cmp");
    } else {
      // Signed integers and pointers.
      Result = Builder.CreateICmp((llvm::ICmpInst::Predicate)SICmpOpc,
                                  LHS.getVal(), RHS.getVal(), "cmp");
    }
  } else {
#if 0
    // Struct/union/complex
    llvm::Value *LHSR, *LHSI, *RHSR, *RHSI, *ResultR, *ResultI;
    EmitLoadOfComplex(LHS, LHSR, LHSI);
    EmitLoadOfComplex(RHS, RHSR, RHSI);

    // FIXME: need to consider _Complex over integers too!

    ResultR = Builder.CreateFCmp((llvm::FCmpInst::Predicate)FCmpOpc,
				 LHSR, RHSR, "cmp.r");
    ResultI = Builder.CreateFCmp((llvm::FCmpInst::Predicate)FCmpOpc,
				 LHSI, RHSI, "cmp.i");
    if (BinaryOperator::EQ == E->getOpcode()) {
      Result = Builder.CreateAnd(ResultR, ResultI, "and.ri");
    } else if (BinaryOperator::NE == E->getOpcode()) {
      Result = Builder.CreateOr(ResultR, ResultI, "or.ri");
    } else {
      assert(0 && "Complex comparison other than == or != ?");
    }
#endif
  }

  // ZExt result to int.
  return RValue::get(Builder.CreateZExt(Result, LLVMIntTy, "cmp.ext"));
}

RValue CodeGenFunction::EmitAnd(RValue LHS, RValue RHS, QualType ResTy) {
  if (LHS.isScalar())
    return RValue::get(Builder.CreateAnd(LHS.getVal(), RHS.getVal(), "and"));
  
  assert(0 && "FIXME: This doesn't handle complex integer operands yet (GNU)");
}

RValue CodeGenFunction::EmitXor(RValue LHS, RValue RHS, QualType ResTy) {
  if (LHS.isScalar())
    return RValue::get(Builder.CreateXor(LHS.getVal(), RHS.getVal(), "xor"));
  
  assert(0 && "FIXME: This doesn't handle complex integer operands yet (GNU)");
}

RValue CodeGenFunction::EmitOr(RValue LHS, RValue RHS, QualType ResTy) {
  if (LHS.isScalar())
    return RValue::get(Builder.CreateOr(LHS.getVal(), RHS.getVal(), "or"));
  
  assert(0 && "FIXME: This doesn't handle complex integer operands yet (GNU)");
}

RValue CodeGenFunction::EmitBinaryLAnd(const BinaryOperator *E) {
  llvm::Value *LHSCond = EvaluateExprAsBool(E->getLHS());
  
  llvm::BasicBlock *ContBlock = new llvm::BasicBlock("land_cont");
  llvm::BasicBlock *RHSBlock = new llvm::BasicBlock("land_rhs");

  llvm::BasicBlock *OrigBlock = Builder.GetInsertBlock();
  Builder.CreateCondBr(LHSCond, RHSBlock, ContBlock);
  
  EmitBlock(RHSBlock);
  llvm::Value *RHSCond = EvaluateExprAsBool(E->getRHS());
  
  // Reaquire the RHS block, as there may be subblocks inserted.
  RHSBlock = Builder.GetInsertBlock();
  EmitBlock(ContBlock);
  
  // Create a PHI node.  If we just evaluted the LHS condition, the result is
  // false.  If we evaluated both, the result is the RHS condition.
  llvm::PHINode *PN = Builder.CreatePHI(llvm::Type::Int1Ty, "land");
  PN->reserveOperandSpace(2);
  PN->addIncoming(llvm::ConstantInt::getFalse(), OrigBlock);
  PN->addIncoming(RHSCond, RHSBlock);
  
  // ZExt result to int.
  return RValue::get(Builder.CreateZExt(PN, LLVMIntTy, "land.ext"));
}

RValue CodeGenFunction::EmitBinaryLOr(const BinaryOperator *E) {
  llvm::Value *LHSCond = EvaluateExprAsBool(E->getLHS());
  
  llvm::BasicBlock *ContBlock = new llvm::BasicBlock("lor_cont");
  llvm::BasicBlock *RHSBlock = new llvm::BasicBlock("lor_rhs");
  
  llvm::BasicBlock *OrigBlock = Builder.GetInsertBlock();
  Builder.CreateCondBr(LHSCond, ContBlock, RHSBlock);
  
  EmitBlock(RHSBlock);
  llvm::Value *RHSCond = EvaluateExprAsBool(E->getRHS());
  
  // Reaquire the RHS block, as there may be subblocks inserted.
  RHSBlock = Builder.GetInsertBlock();
  EmitBlock(ContBlock);
  
  // Create a PHI node.  If we just evaluted the LHS condition, the result is
  // true.  If we evaluated both, the result is the RHS condition.
  llvm::PHINode *PN = Builder.CreatePHI(llvm::Type::Int1Ty, "lor");
  PN->reserveOperandSpace(2);
  PN->addIncoming(llvm::ConstantInt::getTrue(), OrigBlock);
  PN->addIncoming(RHSCond, RHSBlock);
  
  // ZExt result to int.
  return RValue::get(Builder.CreateZExt(PN, LLVMIntTy, "lor.ext"));
}

RValue CodeGenFunction::EmitBinaryAssign(const BinaryOperator *E) {
  assert(E->getLHS()->getType().getCanonicalType() ==
         E->getRHS()->getType().getCanonicalType() && "Invalid assignment");
  LValue LHS = EmitLValue(E->getLHS());
  RValue RHS = EmitExpr(E->getRHS());
  
  // Store the value into the LHS.
  EmitStoreThroughLValue(RHS, LHS, E->getType());

  // Return the RHS.
  return RHS;
}


RValue CodeGenFunction::EmitBinaryComma(const BinaryOperator *E) {
  EmitExpr(E->getLHS());
  return EmitExpr(E->getRHS());
}

RValue CodeGenFunction::EmitConditionalOperator(const ConditionalOperator *E) {
  llvm::BasicBlock *LHSBlock = new llvm::BasicBlock("cond.?");
  llvm::BasicBlock *RHSBlock = new llvm::BasicBlock("cond.:");
  llvm::BasicBlock *ContBlock = new llvm::BasicBlock("cond.cont");
  
  llvm::Value *Cond = EvaluateExprAsBool(E->getCond());
  Builder.CreateCondBr(Cond, LHSBlock, RHSBlock);
  
  EmitBlock(LHSBlock);
  // Handle the GNU extension for missing LHS.
  llvm::Value *LHSValue = E->getLHS() ? EmitExpr(E->getLHS()).getVal() : Cond;
  Builder.CreateBr(ContBlock);
  LHSBlock = Builder.GetInsertBlock();
  
  EmitBlock(RHSBlock);

  llvm::Value *RHSValue = EmitExpr(E->getRHS()).getVal();
  Builder.CreateBr(ContBlock);
  RHSBlock = Builder.GetInsertBlock();
  
  const llvm::Type *LHSType = LHSValue->getType();
  assert(LHSType == RHSValue->getType() && "?: LHS & RHS must have same type");
  
  EmitBlock(ContBlock);
  llvm::PHINode *PN = Builder.CreatePHI(LHSType, "cond");
  PN->reserveOperandSpace(2);
  PN->addIncoming(LHSValue, LHSBlock);
  PN->addIncoming(RHSValue, RHSBlock);
  
  return RValue::get(PN);
}
