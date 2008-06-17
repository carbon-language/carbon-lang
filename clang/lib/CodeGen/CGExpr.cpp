//===--- CGExpr.cpp - Emit LLVM Code from Expressions ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Expr nodes as LLVM code.
//
//===----------------------------------------------------------------------===//

#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "clang/AST/AST.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Target/TargetData.h"
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
  QualType BoolTy = getContext().BoolTy;
  if (!E->getType()->isAnyComplexType())
    return EmitScalarConversion(EmitScalarExpr(E), E->getType(), BoolTy);

  return EmitComplexToScalarConversion(EmitComplexExpr(E), E->getType(),BoolTy);
}

/// EmitAnyExpr - Emit code to compute the specified expression which can have
/// any type.  The result is returned as an RValue struct.  If this is an
/// aggregate expression, the aggloc/agglocvolatile arguments indicate where
/// the result should be returned.
RValue CodeGenFunction::EmitAnyExpr(const Expr *E, llvm::Value *AggLoc, 
                                    bool isAggLocVolatile) {
  if (!hasAggregateLLVMType(E->getType()))
    return RValue::get(EmitScalarExpr(E));
  else if (E->getType()->isAnyComplexType())
    return RValue::getComplex(EmitComplexExpr(E));
  
  EmitAggExpr(E, AggLoc, isAggLocVolatile);
  return RValue::getAggregate(AggLoc);
}

/// getAccessedFieldNo - Given an encoded value and a result number, return
/// the input field number being accessed.
unsigned CodeGenFunction::getAccessedFieldNo(unsigned Idx, 
                                             const llvm::Constant *Elts) {
  if (isa<llvm::ConstantAggregateZero>(Elts))
    return 0;
  
  return cast<llvm::ConstantInt>(Elts->getOperand(Idx))->getZExtValue();
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
  default: {
    printf("Statement class: %d\n", E->getStmtClass());
    WarnUnsupported(E, "l-value expression");
    llvm::Type *Ty = llvm::PointerType::getUnqual(ConvertType(E->getType()));
    return LValue::MakeAddr(llvm::UndefValue::get(Ty),
                            E->getType().getCVRQualifiers());
  }

  case Expr::CallExprClass: return EmitCallExprLValue(cast<CallExpr>(E));
  case Expr::DeclRefExprClass: return EmitDeclRefLValue(cast<DeclRefExpr>(E));
  case Expr::ParenExprClass:return EmitLValue(cast<ParenExpr>(E)->getSubExpr());
  case Expr::PreDefinedExprClass:
    return EmitPreDefinedLValue(cast<PreDefinedExpr>(E));
  case Expr::StringLiteralClass:
    return EmitStringLiteralLValue(cast<StringLiteral>(E));

  case Expr::ObjCIvarRefExprClass: 
    return EmitObjCIvarRefLValue(cast<ObjCIvarRefExpr>(E));
    
  case Expr::UnaryOperatorClass: 
    return EmitUnaryOpLValue(cast<UnaryOperator>(E));
  case Expr::ArraySubscriptExprClass:
    return EmitArraySubscriptExpr(cast<ArraySubscriptExpr>(E));
  case Expr::ExtVectorElementExprClass:
    return EmitExtVectorElementExpr(cast<ExtVectorElementExpr>(E));
  case Expr::MemberExprClass: return EmitMemberExpr(cast<MemberExpr>(E));
  case Expr::CompoundLiteralExprClass:
    return EmitCompoundLiteralLValue(cast<CompoundLiteralExpr>(E));
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
    if (EltTy->isSingleValueType()) {
      llvm::Value *V = Builder.CreateLoad(Ptr, LV.isVolatileQualified(),"tmp");
      
      // Bool can have different representation in memory than in registers.
      if (ExprType->isBooleanType()) {
        if (V->getType() != llvm::Type::Int1Ty)
          V = Builder.CreateTrunc(V, llvm::Type::Int1Ty, "tobool");
      }
      
      return RValue::get(V);
    }
    
    assert(ExprType->isFunctionType() && "Unknown scalar value");
    return RValue::get(Ptr);
  }
  
  if (LV.isVectorElt()) {
    llvm::Value *Vec = Builder.CreateLoad(LV.getVectorAddr(),
                                          LV.isVolatileQualified(), "tmp");
    return RValue::get(Builder.CreateExtractElement(Vec, LV.getVectorIdx(),
                                                    "vecext"));
  }

  // If this is a reference to a subset of the elements of a vector, either
  // shuffle the input or extract/insert them as appropriate.
  if (LV.isExtVectorElt())
    return EmitLoadOfExtVectorElementLValue(LV, ExprType);

  if (LV.isBitfield())
    return EmitLoadOfBitfieldLValue(LV, ExprType);

  assert(0 && "Unknown LValue type!");
  //an invalid RValue, but the assert will
  //ensure that this point is never reached
  return RValue();
}

RValue CodeGenFunction::EmitLoadOfBitfieldLValue(LValue LV,
                                                 QualType ExprType) {
  llvm::Value *Ptr = LV.getBitfieldAddr();
  const llvm::Type *EltTy =
    cast<llvm::PointerType>(Ptr->getType())->getElementType();
  unsigned EltTySize = EltTy->getPrimitiveSizeInBits();
  unsigned short BitfieldSize = LV.getBitfieldSize();
  unsigned short EndBit = LV.getBitfieldStartBit() + BitfieldSize;

  llvm::Value *V = Builder.CreateLoad(Ptr, LV.isVolatileQualified(), "tmp");

  llvm::Value *ShAmt = llvm::ConstantInt::get(EltTy, EltTySize - EndBit);
  V = Builder.CreateShl(V, ShAmt, "tmp");

  ShAmt = llvm::ConstantInt::get(EltTy, EltTySize - BitfieldSize);
  V = LV.isBitfieldSigned() ?
    Builder.CreateAShr(V, ShAmt, "tmp") :
    Builder.CreateLShr(V, ShAmt, "tmp");

  // The bitfield type and the normal type differ when the storage sizes
  // differ (currently just _Bool).
  V = Builder.CreateIntCast(V, ConvertType(ExprType), false, "tmp");

  return RValue::get(V);
}

// If this is a reference to a subset of the elements of a vector, either
// shuffle the input or extract/insert them as appropriate.
RValue CodeGenFunction::EmitLoadOfExtVectorElementLValue(LValue LV,
                                                         QualType ExprType) {
  llvm::Value *Vec = Builder.CreateLoad(LV.getExtVectorAddr(),
                                        LV.isVolatileQualified(), "tmp");
  
  const llvm::Constant *Elts = LV.getExtVectorElts();
  
  // If the result of the expression is a non-vector type, we must be
  // extracting a single element.  Just codegen as an extractelement.
  const VectorType *ExprVT = ExprType->getAsVectorType();
  if (!ExprVT) {
    unsigned InIdx = getAccessedFieldNo(0, Elts);
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
      unsigned InIdx = getAccessedFieldNo(i, Elts);
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
    unsigned InIdx = getAccessedFieldNo(i, Elts);
    llvm::Value *Elt = llvm::ConstantInt::get(llvm::Type::Int32Ty, InIdx);
    Elt = Builder.CreateExtractElement(Vec, Elt, "tmp");
    
    llvm::Value *OutIdx = llvm::ConstantInt::get(llvm::Type::Int32Ty, i);
    Result = Builder.CreateInsertElement(Result, Elt, OutIdx, "tmp");
  }
  
  return RValue::get(Result);
}



/// EmitStoreThroughLValue - Store the specified rvalue into the specified
/// lvalue, where both are guaranteed to the have the same type, and that type
/// is 'Ty'.
void CodeGenFunction::EmitStoreThroughLValue(RValue Src, LValue Dst, 
                                             QualType Ty) {
  if (!Dst.isSimple()) {
    if (Dst.isVectorElt()) {
      // Read/modify/write the vector, inserting the new element.
      llvm::Value *Vec = Builder.CreateLoad(Dst.getVectorAddr(),
                                            Dst.isVolatileQualified(), "tmp");
      Vec = Builder.CreateInsertElement(Vec, Src.getScalarVal(),
                                        Dst.getVectorIdx(), "vecins");
      Builder.CreateStore(Vec, Dst.getVectorAddr(),Dst.isVolatileQualified());
      return;
    }
  
    // If this is an update of extended vector elements, insert them as
    // appropriate.
    if (Dst.isExtVectorElt())
      return EmitStoreThroughExtVectorComponentLValue(Src, Dst, Ty);

    if (Dst.isBitfield())
      return EmitStoreThroughBitfieldLValue(Src, Dst, Ty);

    assert(0 && "Unknown LValue type");
  }
  
  llvm::Value *DstAddr = Dst.getAddress();
  assert(Src.isScalar() && "Can't emit an agg store with this method");
  // FIXME: Handle volatility etc.
  const llvm::Type *SrcTy = Src.getScalarVal()->getType();
  const llvm::PointerType *DstPtr = cast<llvm::PointerType>(DstAddr->getType());
  const llvm::Type *AddrTy = DstPtr->getElementType();
  unsigned AS = DstPtr->getAddressSpace();
  
  if (AddrTy != SrcTy)
    DstAddr = Builder.CreateBitCast(DstAddr, 
                                    llvm::PointerType::get(SrcTy, AS),
                                    "storetmp");
  Builder.CreateStore(Src.getScalarVal(), DstAddr, Dst.isVolatileQualified());
}

void CodeGenFunction::EmitStoreThroughBitfieldLValue(RValue Src, LValue Dst,
                                                     QualType Ty) {
  unsigned short StartBit = Dst.getBitfieldStartBit();
  unsigned short BitfieldSize = Dst.getBitfieldSize();
  llvm::Value *Ptr = Dst.getBitfieldAddr();

  llvm::Value *NewVal = Src.getScalarVal();
  llvm::Value *OldVal = Builder.CreateLoad(Ptr, Dst.isVolatileQualified(),
                                           "tmp");

  // The bitfield type and the normal type differ when the storage sizes
  // differ (currently just _Bool).
  const llvm::Type *EltTy = OldVal->getType();
  unsigned EltTySize = CGM.getTargetData().getABITypeSizeInBits(EltTy);

  NewVal = Builder.CreateIntCast(NewVal, EltTy, false, "tmp");

  // Move the bits into the appropriate location
  llvm::Value *ShAmt = llvm::ConstantInt::get(EltTy, StartBit);
  NewVal = Builder.CreateShl(NewVal, ShAmt, "tmp");

  llvm::Constant *Mask = llvm::ConstantInt::get(
           llvm::APInt::getBitsSet(EltTySize, StartBit,
                                   StartBit + BitfieldSize));

  // Mask out any bits that shouldn't be set in the result.
  NewVal = Builder.CreateAnd(NewVal, Mask, "tmp");

  // Next, mask out the bits this bit-field should include from the old value.
  Mask = llvm::ConstantExpr::getNot(Mask);
  OldVal = Builder.CreateAnd(OldVal, Mask, "tmp");

  // Finally, merge the two together and store it.
  NewVal = Builder.CreateOr(OldVal, NewVal, "tmp");

  Builder.CreateStore(NewVal, Ptr, Dst.isVolatileQualified());
}

void CodeGenFunction::EmitStoreThroughExtVectorComponentLValue(RValue Src,
                                                               LValue Dst,
                                                               QualType Ty) {
  // This access turns into a read/modify/write of the vector.  Load the input
  // value now.
  llvm::Value *Vec = Builder.CreateLoad(Dst.getExtVectorAddr(),
                                        Dst.isVolatileQualified(), "tmp");
  const llvm::Constant *Elts = Dst.getExtVectorElts();
  
  llvm::Value *SrcVal = Src.getScalarVal();
  
  if (const VectorType *VTy = Ty->getAsVectorType()) {
    unsigned NumSrcElts = VTy->getNumElements();

    // Extract/Insert each element.
    for (unsigned i = 0; i != NumSrcElts; ++i) {
      llvm::Value *Elt = llvm::ConstantInt::get(llvm::Type::Int32Ty, i);
      Elt = Builder.CreateExtractElement(SrcVal, Elt, "tmp");
      
      unsigned Idx = getAccessedFieldNo(i, Elts);
      llvm::Value *OutIdx = llvm::ConstantInt::get(llvm::Type::Int32Ty, Idx);
      Vec = Builder.CreateInsertElement(Vec, Elt, OutIdx, "tmp");
    }
  } else {
    // If the Src is a scalar (not a vector) it must be updating one element.
    unsigned InIdx = getAccessedFieldNo(0, Elts);
    llvm::Value *Elt = llvm::ConstantInt::get(llvm::Type::Int32Ty, InIdx);
    Vec = Builder.CreateInsertElement(Vec, SrcVal, Elt, "tmp");
  }
  
  Builder.CreateStore(Vec, Dst.getExtVectorAddr(), Dst.isVolatileQualified());
}


LValue CodeGenFunction::EmitDeclRefLValue(const DeclRefExpr *E) {
  const VarDecl *VD = dyn_cast<VarDecl>(E->getDecl());
  
  if (VD && (VD->isBlockVarDecl() || isa<ParmVarDecl>(VD) ||
        isa<ImplicitParamDecl>(VD))) {
    if (VD->getStorageClass() == VarDecl::Extern)
      return LValue::MakeAddr(CGM.GetAddrOfGlobalVar(VD, false),
                              E->getType().getCVRQualifiers());
    else {
      llvm::Value *V = LocalDeclMap[VD];
      assert(V && "BlockVarDecl not entered in LocalDeclMap?");
      return LValue::MakeAddr(V, E->getType().getCVRQualifiers());
    }
  } else if (VD && VD->isFileVarDecl()) {
    return LValue::MakeAddr(CGM.GetAddrOfGlobalVar(VD, false),
                            E->getType().getCVRQualifiers());
  } else if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(E->getDecl())) {
    return LValue::MakeAddr(CGM.GetAddrOfFunctionDecl(FD, false),
                            E->getType().getCVRQualifiers());
  }
  else if (const ImplicitParamDecl *IPD =
      dyn_cast<ImplicitParamDecl>(E->getDecl())) {
    llvm::Value *V = LocalDeclMap[IPD];
    assert(V && "BlockVarDecl not entered in LocalDeclMap?");
    return LValue::MakeAddr(V, E->getType().getCVRQualifiers());
  }
  assert(0 && "Unimp declref");
  //an invalid LValue, but the assert will
  //ensure that this point is never reached.
  return LValue();
}

LValue CodeGenFunction::EmitUnaryOpLValue(const UnaryOperator *E) {
  // __extension__ doesn't affect lvalue-ness.
  if (E->getOpcode() == UnaryOperator::Extension)
    return EmitLValue(E->getSubExpr());
  
  switch (E->getOpcode()) {
  default: assert(0 && "Unknown unary operator lvalue!");
  case UnaryOperator::Deref:
    return LValue::MakeAddr(EmitScalarExpr(E->getSubExpr()),
      E->getSubExpr()->getType().getCanonicalType()->getAsPointerType()
      ->getPointeeType().getCVRQualifiers());
  case UnaryOperator::Real:
  case UnaryOperator::Imag:
    LValue LV = EmitLValue(E->getSubExpr());
    unsigned Idx = E->getOpcode() == UnaryOperator::Imag;
    return LValue::MakeAddr(Builder.CreateStructGEP(LV.getAddress(),
      Idx, "idx"),E->getSubExpr()->getType().getCVRQualifiers());
  }
}

LValue CodeGenFunction::EmitStringLiteralLValue(const StringLiteral *E) {
  assert(!E->isWide() && "FIXME: Wide strings not supported yet!");
  // Get the string data
  const char *StrData = E->getStrData();
  unsigned Len = E->getByteLength();
  std::string StringLiteral(StrData, StrData+Len);

  // Resize the string to the right size
  const ConstantArrayType *CAT = E->getType()->getAsConstantArrayType();
  uint64_t RealLen = CAT->getSize().getZExtValue();
  StringLiteral.resize(RealLen, '\0');

  return LValue::MakeAddr(CGM.GetAddrOfConstantString(StringLiteral),0);
}

LValue CodeGenFunction::EmitPreDefinedLValue(const PreDefinedExpr *E) {
  std::string FunctionName;
  if(const FunctionDecl *FD = dyn_cast<FunctionDecl>(CurFuncDecl)) {
    FunctionName = FD->getName();
  }
  else {
    assert(0 && "Attempting to load predefined constant for invalid decl type");
  }
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
  
  GlobalVarName += FunctionName;
  
  // FIXME: Can cache/reuse these within the module.
  llvm::Constant *C=llvm::ConstantArray::get(FunctionName);
  
  // Create a global variable for this.
  C = new llvm::GlobalVariable(C->getType(), true, 
                               llvm::GlobalValue::InternalLinkage,
                               C, GlobalVarName, CurFn->getParent());
  return LValue::MakeAddr(C,0);
}

LValue CodeGenFunction::EmitArraySubscriptExpr(const ArraySubscriptExpr *E) {
  // The index must always be an integer, which is not an aggregate.  Emit it.
  llvm::Value *Idx = EmitScalarExpr(E->getIdx());
  
  // If the base is a vector type, then we are forming a vector element lvalue
  // with this subscript.
  if (E->getBase()->getType()->isVectorType()) {
    // Emit the vector as an lvalue to get its address.
    LValue LHS = EmitLValue(E->getBase());
    assert(LHS.isSimple() && "Can only subscript lvalue vectors here!");
    // FIXME: This should properly sign/zero/extend or truncate Idx to i32.
    return LValue::MakeVectorElt(LHS.getAddress(), Idx,
      E->getBase()->getType().getCVRQualifiers());
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
  if (!E->getType()->isConstantSizeType())
    assert(0 && "VLA idx not implemented");
  return LValue::MakeAddr(Builder.CreateGEP(Base, Idx, "arrayidx"),
    E->getBase()->getType().getCanonicalType()->getAsPointerType()
    ->getPointeeType().getCVRQualifiers());
}

static 
llvm::Constant *GenerateConstantVector(llvm::SmallVector<unsigned, 4> &Elts) {
  llvm::SmallVector<llvm::Constant *, 4> CElts;
  
  for (unsigned i = 0, e = Elts.size(); i != e; ++i)
    CElts.push_back(llvm::ConstantInt::get(llvm::Type::Int32Ty, Elts[i]));

  return llvm::ConstantVector::get(&CElts[0], CElts.size());
}

LValue CodeGenFunction::
EmitExtVectorElementExpr(const ExtVectorElementExpr *E) {
  // Emit the base vector as an l-value.
  LValue Base = EmitLValue(E->getBase());

  // Encode the element access list into a vector of unsigned indices.
  llvm::SmallVector<unsigned, 4> Indices;
  E->getEncodedElementAccess(Indices);

  if (Base.isSimple()) {
    llvm::Constant *CV = GenerateConstantVector(Indices);
    return LValue::MakeExtVectorElt(Base.getAddress(), CV,
                                   E->getBase()->getType().getCVRQualifiers());
  }
  assert(Base.isExtVectorElt() && "Can only subscript lvalue vec elts here!");

  llvm::Constant *BaseElts = Base.getExtVectorElts();
  llvm::SmallVector<llvm::Constant *, 4> CElts;

  for (unsigned i = 0, e = Indices.size(); i != e; ++i) {
    if (isa<llvm::ConstantAggregateZero>(BaseElts))
      CElts.push_back(llvm::ConstantInt::get(llvm::Type::Int32Ty, 0));
    else
      CElts.push_back(BaseElts->getOperand(Indices[i]));
  }
  llvm::Constant *CV = llvm::ConstantVector::get(&CElts[0], CElts.size());
  return LValue::MakeExtVectorElt(Base.getExtVectorAddr(), CV,
                                  E->getBase()->getType().getCVRQualifiers());
}

LValue CodeGenFunction::EmitMemberExpr(const MemberExpr *E) {
  bool isUnion = false;
  Expr *BaseExpr = E->getBase();
  llvm::Value *BaseValue = NULL;
  unsigned CVRQualifiers=0;

  // If this is s.x, emit s as an lvalue.  If it is s->x, emit s as a scalar.
  if (E->isArrow()) {
    BaseValue = EmitScalarExpr(BaseExpr);
    const PointerType *PTy = 
      cast<PointerType>(BaseExpr->getType().getCanonicalType());
    if (PTy->getPointeeType()->isUnionType())
      isUnion = true;
    CVRQualifiers = PTy->getPointeeType().getCVRQualifiers();
  }
  else {
    LValue BaseLV = EmitLValue(BaseExpr);
    // FIXME: this isn't right for bitfields.
    BaseValue = BaseLV.getAddress();
    if (BaseExpr->getType()->isUnionType())
      isUnion = true;
    CVRQualifiers = BaseExpr->getType().getCVRQualifiers();
  }

  FieldDecl *Field = E->getMemberDecl();
  return EmitLValueForField(BaseValue, Field, isUnion, CVRQualifiers);
}

LValue CodeGenFunction::EmitLValueForField(llvm::Value* BaseValue,
                                           FieldDecl* Field,
                                           bool isUnion,
                                           unsigned CVRQualifiers)
{
  llvm::Value *V;
  unsigned idx = CGM.getTypes().getLLVMFieldNo(Field);

  if (Field->isBitField()) {
    // FIXME: CodeGenTypes should expose a method to get the appropriate
    // type for FieldTy (the appropriate type is ABI-dependent).
    const llvm::Type *FieldTy = CGM.getTypes().ConvertTypeForMem(Field->getType());
    const llvm::PointerType *BaseTy =
      cast<llvm::PointerType>(BaseValue->getType());
    unsigned AS = BaseTy->getAddressSpace();
    BaseValue = Builder.CreateBitCast(BaseValue,
                                      llvm::PointerType::get(FieldTy, AS),
                                      "tmp");
    V = Builder.CreateGEP(BaseValue,
                          llvm::ConstantInt::get(llvm::Type::Int32Ty, idx),
                          "tmp");

    CodeGenTypes::BitFieldInfo bitFieldInfo =
      CGM.getTypes().getBitFieldInfo(Field);
    return LValue::MakeBitfield(V, bitFieldInfo.Begin, bitFieldInfo.Size,
                                Field->getType()->isSignedIntegerType(),
                            Field->getType().getCVRQualifiers()|CVRQualifiers);
  }

  V = Builder.CreateStructGEP(BaseValue, idx, "tmp");

  // Match union field type.
  if (isUnion) {
    const llvm::Type *FieldTy = 
      CGM.getTypes().ConvertTypeForMem(Field->getType());
    const llvm::PointerType * BaseTy = 
      cast<llvm::PointerType>(BaseValue->getType());
    unsigned AS = BaseTy->getAddressSpace();
    V = Builder.CreateBitCast(V, 
                              llvm::PointerType::get(FieldTy, AS), 
                              "tmp");
  }

  return LValue::MakeAddr(V, 
                          Field->getType().getCVRQualifiers()|CVRQualifiers);
}

LValue CodeGenFunction::EmitCompoundLiteralLValue(const CompoundLiteralExpr* E)
{
  const llvm::Type *LTy = ConvertType(E->getType());
  llvm::Value *DeclPtr = CreateTempAlloca(LTy, ".compoundliteral");

  const Expr* InitExpr = E->getInitializer();
  LValue Result = LValue::MakeAddr(DeclPtr, E->getType().getCVRQualifiers());

  if (E->getType()->isComplexType()) {
    EmitComplexExprIntoAddr(InitExpr, DeclPtr, false);
  } else if (hasAggregateLLVMType(E->getType())) {
    EmitAnyExpr(InitExpr, DeclPtr, false);
  } else {
    EmitStoreThroughLValue(EmitAnyExpr(InitExpr), Result, E->getType());
  }

  return Result;
}

//===--------------------------------------------------------------------===//
//                             Expression Emission
//===--------------------------------------------------------------------===//


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
  return EmitCallExpr(Callee, E->getCallee()->getType(),
                      E->arg_begin(), E->arg_end());
}

RValue CodeGenFunction::EmitCallExpr(Expr *FnExpr,
                                     CallExpr::const_arg_iterator ArgBeg,
                                     CallExpr::const_arg_iterator ArgEnd) {

  llvm::Value *Callee = EmitScalarExpr(FnExpr);
  return EmitCallExpr(Callee, FnExpr->getType(), ArgBeg, ArgEnd);
}

LValue CodeGenFunction::EmitCallExprLValue(const CallExpr *E) {
  // Can only get l-value for call expression returning aggregate type
  RValue RV = EmitCallExpr(E);
  // FIXME: can this be volatile?
  return LValue::MakeAddr(RV.getAggregateAddr(),
                          E->getType().getCVRQualifiers());
}

LValue CodeGenFunction::EmitObjCIvarRefLValue(const ObjCIvarRefExpr *E) {
  // Objective-C objects are traditionally C structures with their layout
  // defined at compile-time.  In some implementations, their layout is not
  // defined until run time in order to allow instance variables to be added to
  // a class without recompiling all of the subclasses.  If this is the case
  // then the CGObjCRuntime subclass must return true to LateBoundIvars and
  // implement the lookup itself.
  if (CGM.getObjCRuntime()->LateBoundIVars()) {
    assert(0 && "FIXME: Implement support for late-bound instance variables");
    return LValue(); // Not reached.
  }
  
  // Get a structure type for the object
  QualType ExprTy = E->getBase()->getType();
  const llvm::Type *ObjectType = ConvertType(ExprTy);
  // TODO:  Add a special case for isa (index 0)
  // Work out which index the ivar is
  const ObjCIvarDecl *Decl = E->getDecl();
  unsigned Index = CGM.getTypes().getLLVMFieldNo(Decl);
    
  // Get object pointer and coerce object pointer to correct type.
  llvm::Value *Object = EmitLValue(E->getBase()).getAddress();
  // FIXME: Volatility
  Object = Builder.CreateLoad(Object, E->getDecl()->getName());
  if (Object->getType() != ObjectType)
    Object = Builder.CreateBitCast(Object, ObjectType);

  
  // Return a pointer to the right element.
  // FIXME: volatile
  return LValue::MakeAddr(Builder.CreateStructGEP(Object, Index,
                                                  Decl->getName()),0);
}

RValue CodeGenFunction::EmitCallExpr(llvm::Value *Callee, QualType FnType, 
                                     CallExpr::const_arg_iterator ArgBeg,
                                     CallExpr::const_arg_iterator ArgEnd) {
  
  // The callee type will always be a pointer to function type, get the function
  // type.
  FnType = cast<PointerType>(FnType.getCanonicalType())->getPointeeType();
  QualType ResultType = cast<FunctionType>(FnType)->getResultType();

  llvm::SmallVector<llvm::Value*, 16> Args;
  
  // Handle struct-return functions by passing a pointer to the location that
  // we would like to return into.
  if (hasAggregateLLVMType(ResultType)) {
    // Create a temporary alloca to hold the result of the call. :(
    Args.push_back(CreateTempAlloca(ConvertType(ResultType)));
    // FIXME: set the stret attribute on the argument.
  }
  
  for (CallExpr::const_arg_iterator I = ArgBeg; I != ArgEnd; ++I) {
    QualType ArgTy = I->getType();

    if (!hasAggregateLLVMType(ArgTy)) {
      // Scalar argument is passed by-value.
      Args.push_back(EmitScalarExpr(*I));
    } else if (ArgTy->isAnyComplexType()) {
      // Make a temporary alloca to pass the argument.
      llvm::Value *DestMem = CreateTempAlloca(ConvertType(ArgTy));
      EmitComplexExprIntoAddr(*I, DestMem, false);
      Args.push_back(DestMem);
    } else {
      llvm::Value *DestMem = CreateTempAlloca(ConvertType(ArgTy));
      EmitAggExpr(*I, DestMem, false);
      Args.push_back(DestMem);
    }
  }
  
  llvm::CallInst *CI = Builder.CreateCall(Callee,&Args[0],&Args[0]+Args.size());

  // Note that there is parallel code in SetFunctionAttributes in CodeGenModule
  llvm::SmallVector<llvm::ParamAttrsWithIndex, 8> ParamAttrList;
  if (hasAggregateLLVMType(ResultType))
    ParamAttrList.push_back(
        llvm::ParamAttrsWithIndex::get(1, llvm::ParamAttr::StructRet));
  unsigned increment = hasAggregateLLVMType(ResultType) ? 2 : 1;
  
  unsigned i = 0;
  for (CallExpr::const_arg_iterator I = ArgBeg; I != ArgEnd; ++I, ++i) {
    QualType ParamType = I->getType();
    unsigned ParamAttrs = 0;
    if (ParamType->isRecordType())
      ParamAttrs |= llvm::ParamAttr::ByVal;
    if (ParamType->isSignedIntegerType() && ParamType->isPromotableIntegerType())
      ParamAttrs |= llvm::ParamAttr::SExt;
    if (ParamType->isUnsignedIntegerType() && ParamType->isPromotableIntegerType())
      ParamAttrs |= llvm::ParamAttr::ZExt;
    if (ParamAttrs)
      ParamAttrList.push_back(llvm::ParamAttrsWithIndex::get(i + increment,
                                                             ParamAttrs));
  }
  CI->setParamAttrs(llvm::PAListPtr::get(ParamAttrList.begin(),
                                         ParamAttrList.size()));

  if (const llvm::Function *F = dyn_cast<llvm::Function>(Callee))
    CI->setCallingConv(F->getCallingConv());
  if (CI->getType() != llvm::Type::VoidTy)
    CI->setName("call");
  else if (ResultType->isAnyComplexType())
    return RValue::getComplex(LoadComplexFromAddr(Args[0], false));
  else if (hasAggregateLLVMType(ResultType))
    // Struct return.
    return RValue::getAggregate(Args[0]);
  else {
    // void return.
    assert(ResultType->isVoidType() && "Should only have a void expr here");
    CI = 0;
  }
      
  return RValue::get(CI);
}
