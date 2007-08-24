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
  return ConvertScalarValueToBool(EmitAnyExpr(E), E->getType());
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
  } else if (isa<ComplexType>(Ty)) {
    assert(0 && "implement complex -> bool");
    
  } else {
    assert((isa<PointerType>(Ty) ||
            (isa<TagType>(Ty) &&
             cast<TagType>(Ty)->getDecl()->getKind() == Decl::Enum)) &&
           "Unknown Type");
    // Code below handles this case fine.
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
  return LValue::MakeAddr(EmitScalarExpr(E->getSubExpr()));
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
  llvm::Value *Idx = EmitScalarExpr(E->getIdx());
  
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
  llvm::Value *Base = EmitScalarExpr(E->getBase());
  
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

/// EmitAnyExpr - Emit an expression of any type: scalar, complex, aggregate,
/// returning an rvalue corresponding to it.  If NeedResult is false, the
/// result of the expression doesn't need to be generated into memory.
RValue CodeGenFunction::EmitAnyExpr(const Expr *E, bool NeedResult) {
  if (!hasAggregateLLVMType(E->getType()))
    return RValue::get(EmitScalarExpr(E));
  
  llvm::Value *DestMem = 0;
  if (NeedResult)
    DestMem = CreateTempAlloca(ConvertType(E->getType()));
  
  if (!E->getType()->isComplexType()) {
    EmitAggExpr(E, DestMem, false);
  } else if (NeedResult)
    EmitComplexExprIntoAddr(E, DestMem);
  else
    EmitComplexExpr(E);
  
  return RValue::getAggregate(DestMem);
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
        
  llvm::Value *Callee = EmitScalarExpr(E->getCallee());
  
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
    RValue ArgVal = EmitAnyExpr(E->getArg(i));
    
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
