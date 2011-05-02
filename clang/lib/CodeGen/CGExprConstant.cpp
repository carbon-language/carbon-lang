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
#include "CGCXXABI.h"
#include "CGObjCRuntime.h"
#include "CGRecordLayout.h"
#include "clang/AST/APValue.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/Builtins.h"
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Target/TargetData.h"
using namespace clang;
using namespace CodeGen;

//===----------------------------------------------------------------------===//
//                            ConstStructBuilder
//===----------------------------------------------------------------------===//

namespace {
class ConstStructBuilder {
  CodeGenModule &CGM;
  CodeGenFunction *CGF;

  bool Packed;
  CharUnits NextFieldOffsetInChars;
  CharUnits LLVMStructAlignment;
  std::vector<llvm::Constant *> Elements;
public:
  static llvm::Constant *BuildStruct(CodeGenModule &CGM, CodeGenFunction *CGF,
                                     InitListExpr *ILE);
  
private:  
  ConstStructBuilder(CodeGenModule &CGM, CodeGenFunction *CGF)
    : CGM(CGM), CGF(CGF), Packed(false), 
    NextFieldOffsetInChars(CharUnits::Zero()),
    LLVMStructAlignment(CharUnits::One()) { }

  bool AppendField(const FieldDecl *Field, uint64_t FieldOffset,
                   llvm::Constant *InitExpr);

  void AppendBitField(const FieldDecl *Field, uint64_t FieldOffset,
                      llvm::ConstantInt *InitExpr);

  void AppendPadding(CharUnits PadSize);

  void AppendTailPadding(CharUnits RecordSize);

  void ConvertStructToPacked();
                              
  bool Build(InitListExpr *ILE);

  CharUnits getAlignment(const llvm::Constant *C) const {
    if (Packed)  return CharUnits::One();
    return CharUnits::fromQuantity(
        CGM.getTargetData().getABITypeAlignment(C->getType()));
  }

  CharUnits getSizeInChars(const llvm::Constant *C) const {
    return CharUnits::fromQuantity(
        CGM.getTargetData().getTypeAllocSize(C->getType()));
  }
};

bool ConstStructBuilder::
AppendField(const FieldDecl *Field, uint64_t FieldOffset,
            llvm::Constant *InitCst) {

  const ASTContext &Context = CGM.getContext();

  CharUnits FieldOffsetInChars = Context.toCharUnitsFromBits(FieldOffset);

  assert(NextFieldOffsetInChars <= FieldOffsetInChars
         && "Field offset mismatch!");

  CharUnits FieldAlignment = getAlignment(InitCst);

  // Round up the field offset to the alignment of the field type.
  CharUnits AlignedNextFieldOffsetInChars =
    NextFieldOffsetInChars.RoundUpToAlignment(FieldAlignment);

  if (AlignedNextFieldOffsetInChars > FieldOffsetInChars) {
    assert(!Packed && "Alignment is wrong even with a packed struct!");

    // Convert the struct to a packed struct.
    ConvertStructToPacked();
    
    AlignedNextFieldOffsetInChars = NextFieldOffsetInChars;
  }

  if (AlignedNextFieldOffsetInChars < FieldOffsetInChars) {
    // We need to append padding.
    AppendPadding(
        FieldOffsetInChars - NextFieldOffsetInChars);

    assert(NextFieldOffsetInChars == FieldOffsetInChars &&
           "Did not add enough padding!");

    AlignedNextFieldOffsetInChars = NextFieldOffsetInChars;
  }

  // Add the field.
  Elements.push_back(InitCst);
  NextFieldOffsetInChars = AlignedNextFieldOffsetInChars +
                           getSizeInChars(InitCst);
  
  if (Packed)
    assert(LLVMStructAlignment == CharUnits::One() && 
           "Packed struct not byte-aligned!");
  else
    LLVMStructAlignment = std::max(LLVMStructAlignment, FieldAlignment);

  return true;
}

void ConstStructBuilder::AppendBitField(const FieldDecl *Field,
                                        uint64_t FieldOffset,
                                        llvm::ConstantInt *CI) {
  const ASTContext &Context = CGM.getContext();
  const uint64_t CharWidth = Context.getCharWidth();
  uint64_t NextFieldOffsetInBits = Context.toBits(NextFieldOffsetInChars);
  if (FieldOffset > NextFieldOffsetInBits) {
    // We need to add padding.
    CharUnits PadSize = Context.toCharUnitsFromBits(
      llvm::RoundUpToAlignment(FieldOffset - NextFieldOffsetInBits, 
                               Context.Target.getCharAlign()));

    AppendPadding(PadSize);
  }

  uint64_t FieldSize =
    Field->getBitWidth()->EvaluateAsInt(Context).getZExtValue();

  llvm::APInt FieldValue = CI->getValue();

  // Promote the size of FieldValue if necessary
  // FIXME: This should never occur, but currently it can because initializer
  // constants are cast to bool, and because clang is not enforcing bitfield
  // width limits.
  if (FieldSize > FieldValue.getBitWidth())
    FieldValue = FieldValue.zext(FieldSize);

  // Truncate the size of FieldValue to the bit field size.
  if (FieldSize < FieldValue.getBitWidth())
    FieldValue = FieldValue.trunc(FieldSize);

  NextFieldOffsetInBits = Context.toBits(NextFieldOffsetInChars);
  if (FieldOffset < NextFieldOffsetInBits) {
    // Either part of the field or the entire field can go into the previous
    // byte.
    assert(!Elements.empty() && "Elements can't be empty!");

    unsigned BitsInPreviousByte = NextFieldOffsetInBits - FieldOffset;

    bool FitsCompletelyInPreviousByte =
      BitsInPreviousByte >= FieldValue.getBitWidth();

    llvm::APInt Tmp = FieldValue;

    if (!FitsCompletelyInPreviousByte) {
      unsigned NewFieldWidth = FieldSize - BitsInPreviousByte;

      if (CGM.getTargetData().isBigEndian()) {
        Tmp = Tmp.lshr(NewFieldWidth);
        Tmp = Tmp.trunc(BitsInPreviousByte);

        // We want the remaining high bits.
        FieldValue = FieldValue.trunc(NewFieldWidth);
      } else {
        Tmp = Tmp.trunc(BitsInPreviousByte);

        // We want the remaining low bits.
        FieldValue = FieldValue.lshr(BitsInPreviousByte);
        FieldValue = FieldValue.trunc(NewFieldWidth);
      }
    }

    Tmp = Tmp.zext(CharWidth);
    if (CGM.getTargetData().isBigEndian()) {
      if (FitsCompletelyInPreviousByte)
        Tmp = Tmp.shl(BitsInPreviousByte - FieldValue.getBitWidth());
    } else {
      Tmp = Tmp.shl(CharWidth - BitsInPreviousByte);
    }

    // 'or' in the bits that go into the previous byte.
    llvm::Value *LastElt = Elements.back();
    if (llvm::ConstantInt *Val = dyn_cast<llvm::ConstantInt>(LastElt))
      Tmp |= Val->getValue();
    else {
      assert(isa<llvm::UndefValue>(LastElt));
      // If there is an undef field that we're adding to, it can either be a
      // scalar undef (in which case, we just replace it with our field) or it
      // is an array.  If it is an array, we have to pull one byte off the
      // array so that the other undef bytes stay around.
      if (!isa<llvm::IntegerType>(LastElt->getType())) {
        // The undef padding will be a multibyte array, create a new smaller
        // padding and then an hole for our i8 to get plopped into.
        assert(isa<llvm::ArrayType>(LastElt->getType()) &&
               "Expected array padding of undefs");
        const llvm::ArrayType *AT = cast<llvm::ArrayType>(LastElt->getType());
        assert(AT->getElementType()->isIntegerTy(CharWidth) &&
               AT->getNumElements() != 0 &&
               "Expected non-empty array padding of undefs");
        
        // Remove the padding array.
        NextFieldOffsetInChars -= CharUnits::fromQuantity(AT->getNumElements());
        Elements.pop_back();
        
        // Add the padding back in two chunks.
        AppendPadding(CharUnits::fromQuantity(AT->getNumElements()-1));
        AppendPadding(CharUnits::One());
        assert(isa<llvm::UndefValue>(Elements.back()) &&
               Elements.back()->getType()->isIntegerTy(CharWidth) &&
               "Padding addition didn't work right");
      }
    }

    Elements.back() = llvm::ConstantInt::get(CGM.getLLVMContext(), Tmp);

    if (FitsCompletelyInPreviousByte)
      return;
  }

  while (FieldValue.getBitWidth() > CharWidth) {
    llvm::APInt Tmp;

    if (CGM.getTargetData().isBigEndian()) {
      // We want the high bits.
      Tmp = 
        FieldValue.lshr(FieldValue.getBitWidth() - CharWidth).trunc(CharWidth);
    } else {
      // We want the low bits.
      Tmp = FieldValue.trunc(CharWidth);

      FieldValue = FieldValue.lshr(CharWidth);
    }

    Elements.push_back(llvm::ConstantInt::get(CGM.getLLVMContext(), Tmp));
    ++NextFieldOffsetInChars;

    FieldValue = FieldValue.trunc(FieldValue.getBitWidth() - CharWidth);
  }

  assert(FieldValue.getBitWidth() > 0 &&
         "Should have at least one bit left!");
  assert(FieldValue.getBitWidth() <= CharWidth &&
         "Should not have more than a byte left!");

  if (FieldValue.getBitWidth() < CharWidth) {
    if (CGM.getTargetData().isBigEndian()) {
      unsigned BitWidth = FieldValue.getBitWidth();

      FieldValue = FieldValue.zext(CharWidth) << (CharWidth - BitWidth);
    } else
      FieldValue = FieldValue.zext(CharWidth);
  }

  // Append the last element.
  Elements.push_back(llvm::ConstantInt::get(CGM.getLLVMContext(),
                                            FieldValue));
  ++NextFieldOffsetInChars;
}

void ConstStructBuilder::AppendPadding(CharUnits PadSize) {
  if (PadSize.isZero())
    return;

  const llvm::Type *Ty = llvm::Type::getInt8Ty(CGM.getLLVMContext());
  if (PadSize > CharUnits::One())
    Ty = llvm::ArrayType::get(Ty, PadSize.getQuantity());

  llvm::Constant *C = llvm::UndefValue::get(Ty);
  Elements.push_back(C);
  assert(getAlignment(C) == CharUnits::One() && 
         "Padding must have 1 byte alignment!");

  NextFieldOffsetInChars += getSizeInChars(C);
}

void ConstStructBuilder::AppendTailPadding(CharUnits RecordSize) {
  assert(NextFieldOffsetInChars <= RecordSize && 
         "Size mismatch!");

  AppendPadding(RecordSize - NextFieldOffsetInChars);
}

void ConstStructBuilder::ConvertStructToPacked() {
  std::vector<llvm::Constant *> PackedElements;
  CharUnits ElementOffsetInChars = CharUnits::Zero();

  for (unsigned i = 0, e = Elements.size(); i != e; ++i) {
    llvm::Constant *C = Elements[i];

    CharUnits ElementAlign = CharUnits::fromQuantity(
      CGM.getTargetData().getABITypeAlignment(C->getType()));
    CharUnits AlignedElementOffsetInChars =
      ElementOffsetInChars.RoundUpToAlignment(ElementAlign);

    if (AlignedElementOffsetInChars > ElementOffsetInChars) {
      // We need some padding.
      CharUnits NumChars =
        AlignedElementOffsetInChars - ElementOffsetInChars;

      const llvm::Type *Ty = llvm::Type::getInt8Ty(CGM.getLLVMContext());
      if (NumChars > CharUnits::One())
        Ty = llvm::ArrayType::get(Ty, NumChars.getQuantity());

      llvm::Constant *Padding = llvm::UndefValue::get(Ty);
      PackedElements.push_back(Padding);
      ElementOffsetInChars += getSizeInChars(Padding);
    }

    PackedElements.push_back(C);
    ElementOffsetInChars += getSizeInChars(C);
  }

  assert(ElementOffsetInChars == NextFieldOffsetInChars &&
         "Packing the struct changed its size!");

  Elements = PackedElements;
  LLVMStructAlignment = CharUnits::One();
  Packed = true;
}
                            
bool ConstStructBuilder::Build(InitListExpr *ILE) {
  RecordDecl *RD = ILE->getType()->getAs<RecordType>()->getDecl();
  const ASTRecordLayout &Layout = CGM.getContext().getASTRecordLayout(RD);

  unsigned FieldNo = 0;
  unsigned ElementNo = 0;
  const FieldDecl *LastFD = 0;
  bool IsMsStruct = RD->hasAttr<MsStructAttr>();
  
  for (RecordDecl::field_iterator Field = RD->field_begin(),
       FieldEnd = RD->field_end(); Field != FieldEnd; ++Field, ++FieldNo) {
    if (IsMsStruct) {
      // Zero-length bitfields following non-bitfield members are
      // ignored:
      if (CGM.getContext().ZeroBitfieldFollowsNonBitfield((*Field), LastFD) ||
          CGM.getContext().ZeroBitfieldFollowsBitfield((*Field), LastFD)) {
        --FieldNo;
        continue;
      }
      LastFD = (*Field);
    }
    
    // If this is a union, skip all the fields that aren't being initialized.
    if (RD->isUnion() && ILE->getInitializedFieldInUnion() != *Field)
      continue;

    // Don't emit anonymous bitfields, they just affect layout.
    if (Field->isBitField() && !Field->getIdentifier()) {
      LastFD = (*Field);
      continue;
    }

    // Get the initializer.  A struct can include fields without initializers,
    // we just use explicit null values for them.
    llvm::Constant *EltInit;
    if (ElementNo < ILE->getNumInits())
      EltInit = CGM.EmitConstantExpr(ILE->getInit(ElementNo++),
                                     Field->getType(), CGF);
    else
      EltInit = CGM.EmitNullConstant(Field->getType());

    if (!EltInit)
      return false;
    
    if (!Field->isBitField()) {
      // Handle non-bitfield members.
      if (!AppendField(*Field, Layout.getFieldOffset(FieldNo), EltInit))
        return false;
    } else {
      // Otherwise we have a bitfield.
      AppendBitField(*Field, Layout.getFieldOffset(FieldNo),
                     cast<llvm::ConstantInt>(EltInit));
    }
  }

  CharUnits LayoutSizeInChars = Layout.getSize();

  if (NextFieldOffsetInChars > LayoutSizeInChars) {
    // If the struct is bigger than the size of the record type,
    // we must have a flexible array member at the end.
    assert(RD->hasFlexibleArrayMember() &&
           "Must have flexible array member if struct is bigger than type!");
    
    // No tail padding is necessary.
    return true;
  }

  CharUnits LLVMSizeInChars = 
    NextFieldOffsetInChars.RoundUpToAlignment(LLVMStructAlignment);

  // Check if we need to convert the struct to a packed struct.
  if (NextFieldOffsetInChars <= LayoutSizeInChars && 
      LLVMSizeInChars > LayoutSizeInChars) {
    assert(!Packed && "Size mismatch!");
    
    ConvertStructToPacked();
    assert(NextFieldOffsetInChars <= LayoutSizeInChars &&
           "Converting to packed did not help!");
  }

  // Append tail padding if necessary.
  AppendTailPadding(LayoutSizeInChars);

  assert(LayoutSizeInChars == NextFieldOffsetInChars &&
         "Tail padding mismatch!");

  return true;
}
  
llvm::Constant *ConstStructBuilder::
  BuildStruct(CodeGenModule &CGM, CodeGenFunction *CGF, InitListExpr *ILE) {
  ConstStructBuilder Builder(CGM, CGF);
  
  if (!Builder.Build(ILE))
    return 0;
  
  llvm::Constant *Result =
  llvm::ConstantStruct::get(CGM.getLLVMContext(),
                            Builder.Elements, Builder.Packed);
  
  assert(Builder.NextFieldOffsetInChars.RoundUpToAlignment(
           Builder.getAlignment(Result)) ==
         Builder.getSizeInChars(Result) && "Size mismatch!");
  
  return Result;
}

  
//===----------------------------------------------------------------------===//
//                             ConstExprEmitter
//===----------------------------------------------------------------------===//
  
class ConstExprEmitter :
  public StmtVisitor<ConstExprEmitter, llvm::Constant*> {
  CodeGenModule &CGM;
  CodeGenFunction *CGF;
  llvm::LLVMContext &VMContext;
public:
  ConstExprEmitter(CodeGenModule &cgm, CodeGenFunction *cgf)
    : CGM(cgm), CGF(cgf), VMContext(cgm.getLLVMContext()) {
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

  llvm::Constant *VisitGenericSelectionExpr(GenericSelectionExpr *GE) {
    return Visit(GE->getResultExpr());
  }

  llvm::Constant *VisitCompoundLiteralExpr(CompoundLiteralExpr *E) {
    return Visit(E->getInitializer());
  }

  llvm::Constant *VisitUnaryAddrOf(UnaryOperator *E) {
    if (E->getType()->isMemberPointerType())
      return CGM.getMemberPointerConstant(E);

    return 0;
  }
    
  llvm::Constant *VisitBinSub(BinaryOperator *E) {
    // This must be a pointer/pointer subtraction.  This only happens for
    // address of label.
    if (!isa<AddrLabelExpr>(E->getLHS()->IgnoreParenNoopCasts(CGM.getContext())) ||
       !isa<AddrLabelExpr>(E->getRHS()->IgnoreParenNoopCasts(CGM.getContext())))
      return 0;
    
    llvm::Constant *LHS = CGM.EmitConstantExpr(E->getLHS(),
                                               E->getLHS()->getType(), CGF);
    llvm::Constant *RHS = CGM.EmitConstantExpr(E->getRHS(),
                                               E->getRHS()->getType(), CGF);

    const llvm::Type *ResultType = ConvertType(E->getType());
    LHS = llvm::ConstantExpr::getPtrToInt(LHS, ResultType);
    RHS = llvm::ConstantExpr::getPtrToInt(RHS, ResultType);
        
    // No need to divide by element size, since addr of label is always void*,
    // which has size 1 in GNUish.
    return llvm::ConstantExpr::getSub(LHS, RHS);
  }
    
  llvm::Constant *VisitCastExpr(CastExpr* E) {
    Expr *subExpr = E->getSubExpr();
    llvm::Constant *C = CGM.EmitConstantExpr(subExpr, subExpr->getType(), CGF);
    if (!C) return 0;

    const llvm::Type *destType = ConvertType(E->getType());

    switch (E->getCastKind()) {
    case CK_ToUnion: {
      // GCC cast to union extension
      assert(E->getType()->isUnionType() &&
             "Destination type is not union type!");

      // Build a struct with the union sub-element as the first member,
      // and padded to the appropriate size
      std::vector<llvm::Constant*> Elts;
      std::vector<const llvm::Type*> Types;
      Elts.push_back(C);
      Types.push_back(C->getType());
      unsigned CurSize = CGM.getTargetData().getTypeAllocSize(C->getType());
      unsigned TotalSize = CGM.getTargetData().getTypeAllocSize(destType);

      assert(CurSize <= TotalSize && "Union size mismatch!");
      if (unsigned NumPadBytes = TotalSize - CurSize) {
        const llvm::Type *Ty = llvm::Type::getInt8Ty(VMContext);
        if (NumPadBytes > 1)
          Ty = llvm::ArrayType::get(Ty, NumPadBytes);

        Elts.push_back(llvm::UndefValue::get(Ty));
        Types.push_back(Ty);
      }

      llvm::StructType* STy =
        llvm::StructType::get(C->getType()->getContext(), Types, false);
      return llvm::ConstantStruct::get(STy, Elts);
    }
    case CK_NullToMemberPointer: {
      const MemberPointerType *MPT = E->getType()->getAs<MemberPointerType>();
      return CGM.getCXXABI().EmitNullMemberPointer(MPT);
    }

    case CK_DerivedToBaseMemberPointer:
    case CK_BaseToDerivedMemberPointer:
      return CGM.getCXXABI().EmitMemberPointerConversion(C, E);

    case CK_LValueToRValue:
    case CK_NoOp:
      return C;

    case CK_AnyPointerToObjCPointerCast:
    case CK_AnyPointerToBlockPointerCast:
    case CK_LValueBitCast:
    case CK_BitCast:
      if (C->getType() == destType) return C;
      return llvm::ConstantExpr::getBitCast(C, destType);

    case CK_Dependent: llvm_unreachable("saw dependent cast!");

    // These will never be supported.
    case CK_ObjCObjectLValueCast:
    case CK_GetObjCProperty:
    case CK_ToVoid:
    case CK_Dynamic:
      return 0;

    // These might need to be supported for constexpr.
    case CK_UserDefinedConversion:
    case CK_ConstructorConversion:
      return 0;

    // These should eventually be supported.
    case CK_ArrayToPointerDecay:
    case CK_FunctionToPointerDecay:
    case CK_BaseToDerived:
    case CK_DerivedToBase:
    case CK_UncheckedDerivedToBase:
    case CK_MemberPointerToBoolean:
    case CK_VectorSplat:
    case CK_FloatingRealToComplex:
    case CK_FloatingComplexToReal:
    case CK_FloatingComplexToBoolean:
    case CK_FloatingComplexCast:
    case CK_FloatingComplexToIntegralComplex:
    case CK_IntegralRealToComplex:
    case CK_IntegralComplexToReal:
    case CK_IntegralComplexToBoolean:
    case CK_IntegralComplexCast:
    case CK_IntegralComplexToFloatingComplex:
      return 0;

    case CK_PointerToIntegral:
      if (!E->getType()->isBooleanType())
        return llvm::ConstantExpr::getPtrToInt(C, destType);
      // fallthrough

    case CK_PointerToBoolean:
      return llvm::ConstantExpr::getICmp(llvm::CmpInst::ICMP_EQ, C,
        llvm::ConstantPointerNull::get(cast<llvm::PointerType>(C->getType())));

    case CK_NullToPointer:
      return llvm::ConstantPointerNull::get(cast<llvm::PointerType>(destType));

    case CK_IntegralCast: {
      bool isSigned = subExpr->getType()->isSignedIntegerType();
      return llvm::ConstantExpr::getIntegerCast(C, destType, isSigned);
    }

    case CK_IntegralToPointer: {
      bool isSigned = subExpr->getType()->isSignedIntegerType();
      C = llvm::ConstantExpr::getIntegerCast(C, CGM.IntPtrTy, isSigned);
      return llvm::ConstantExpr::getIntToPtr(C, destType);
    }

    case CK_IntegralToBoolean:
      return llvm::ConstantExpr::getICmp(llvm::CmpInst::ICMP_EQ, C,
                             llvm::Constant::getNullValue(C->getType()));

    case CK_IntegralToFloating:
      if (subExpr->getType()->isSignedIntegerType())
        return llvm::ConstantExpr::getSIToFP(C, destType);
      else
        return llvm::ConstantExpr::getUIToFP(C, destType);

    case CK_FloatingToIntegral:
      if (E->getType()->isSignedIntegerType())
        return llvm::ConstantExpr::getFPToSI(C, destType);
      else
        return llvm::ConstantExpr::getFPToUI(C, destType);

    case CK_FloatingToBoolean:
      return llvm::ConstantExpr::getFCmp(llvm::CmpInst::FCMP_UNE, C,
                             llvm::Constant::getNullValue(C->getType()));

    case CK_FloatingCast:
      return llvm::ConstantExpr::getFPCast(C, destType);
    }
    llvm_unreachable("Invalid CastKind");
  }

  llvm::Constant *VisitCXXDefaultArgExpr(CXXDefaultArgExpr *DAE) {
    return Visit(DAE->getExpr());
  }

  llvm::Constant *EmitArrayInitialization(InitListExpr *ILE) {
    unsigned NumInitElements = ILE->getNumInits();
    if (NumInitElements == 1 && ILE->getType() == ILE->getInit(0)->getType() &&
        (isa<StringLiteral>(ILE->getInit(0)) ||
         isa<ObjCEncodeExpr>(ILE->getInit(0))))
      return Visit(ILE->getInit(0));

    std::vector<llvm::Constant*> Elts;
    const llvm::ArrayType *AType =
        cast<llvm::ArrayType>(ConvertType(ILE->getType()));
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
    // FIXME: This doesn't handle member pointers correctly!
    llvm::Constant *fillC;
    if (Expr *filler = ILE->getArrayFiller())
      fillC = CGM.EmitConstantExpr(filler, filler->getType(), CGF);
    else
      fillC = llvm::Constant::getNullValue(ElemTy);
    if (!fillC)
      return 0;
    RewriteType |= (fillC->getType() != ElemTy);
    for (; i < NumElements; ++i)
      Elts.push_back(fillC);

    if (RewriteType) {
      // FIXME: Try to avoid packing the array
      std::vector<const llvm::Type*> Types;
      for (unsigned i = 0; i < Elts.size(); ++i)
        Types.push_back(Elts[i]->getType());
      const llvm::StructType *SType = llvm::StructType::get(AType->getContext(),
                                                            Types, true);
      return llvm::ConstantStruct::get(SType, Elts);
    }

    return llvm::ConstantArray::get(AType, Elts);
  }

  llvm::Constant *EmitStructInitialization(InitListExpr *ILE) {
    return ConstStructBuilder::BuildStruct(CGM, CGF, ILE);
  }

  llvm::Constant *EmitUnionInitialization(InitListExpr *ILE) {
    return ConstStructBuilder::BuildStruct(CGM, CGF, ILE);
  }

  llvm::Constant *VisitImplicitValueInitExpr(ImplicitValueInitExpr* E) {
    return CGM.EmitNullConstant(E->getType());
  }

  llvm::Constant *VisitInitListExpr(InitListExpr *ILE) {
    if (ILE->getType()->isScalarType()) {
      // We have a scalar in braces. Just use the first element.
      if (ILE->getNumInits() > 0) {
        Expr *Init = ILE->getInit(0);
        return CGM.EmitConstantExpr(Init, Init->getType(), CGF);
      }
      return CGM.EmitNullConstant(ILE->getType());
    }

    if (ILE->getType()->isArrayType())
      return EmitArrayInitialization(ILE);

    if (ILE->getType()->isRecordType())
      return EmitStructInitialization(ILE);

    if (ILE->getType()->isUnionType())
      return EmitUnionInitialization(ILE);

    // If ILE was a constant vector, we would have handled it already.
    if (ILE->getType()->isVectorType())
      return 0;

    assert(0 && "Unable to handle InitListExpr");
    // Get rid of control reaches end of void function warning.
    // Not reached.
    return 0;
  }

  llvm::Constant *VisitCXXConstructExpr(CXXConstructExpr *E) {
    if (!E->getConstructor()->isTrivial())
      return 0;

    QualType Ty = E->getType();

    // FIXME: We should not have to call getBaseElementType here.
    const RecordType *RT = 
      CGM.getContext().getBaseElementType(Ty)->getAs<RecordType>();
    const CXXRecordDecl *RD = cast<CXXRecordDecl>(RT->getDecl());
    
    // If the class doesn't have a trivial destructor, we can't emit it as a
    // constant expr.
    if (!RD->hasTrivialDestructor())
      return 0;
    
    // Only copy and default constructors can be trivial.


    if (E->getNumArgs()) {
      assert(E->getNumArgs() == 1 && "trivial ctor with > 1 argument");
      assert(E->getConstructor()->isCopyConstructor() &&
             "trivial ctor has argument but isn't a copy ctor");

      Expr *Arg = E->getArg(0);
      assert(CGM.getContext().hasSameUnqualifiedType(Ty, Arg->getType()) &&
             "argument to copy ctor is of wrong type");

      return Visit(Arg);
    }

    return CGM.EmitNullConstant(Ty);
  }

  llvm::Constant *VisitStringLiteral(StringLiteral *E) {
    assert(!E->getType()->isPointerType() && "Strings are always arrays");

    // This must be a string initializing an array in a static initializer.
    // Don't emit it as the address of the string, emit the string data itself
    // as an inline array.
    return llvm::ConstantArray::get(VMContext,
                                    CGM.GetStringForStringLiteral(E), false);
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
    return llvm::ConstantArray::get(VMContext, Str, false);
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
        C = new llvm::GlobalVariable(CGM.getModule(), C->getType(),
                                     E->getType().isConstant(CGM.getContext()),
                                     llvm::GlobalValue::InternalLinkage,
                                     C, ".compoundliteral", 0, false,
                          CGM.getContext().getTargetAddressSpace(E->getType()));
      return C;
    }
    case Expr::DeclRefExprClass: {
      ValueDecl *Decl = cast<DeclRefExpr>(E)->getDecl();
      if (Decl->hasAttr<WeakRefAttr>())
        return CGM.GetWeakRefReference(Decl);
      if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(Decl))
        return CGM.GetAddrOfFunction(FD);
      if (const VarDecl* VD = dyn_cast<VarDecl>(Decl)) {
        // We can never refer to a variable with local storage.
        if (!VD->hasLocalStorage()) {
          if (VD->isFileVarDecl() || VD->hasExternalStorage())
            return CGM.GetAddrOfGlobalVar(VD);
          else if (VD->isLocalVarDecl()) {
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
      llvm::Constant *C =
          CGM.getObjCRuntime().GenerateConstantString(SL->getString());
      return llvm::ConstantExpr::getBitCast(C, ConvertType(E->getType()));
    }
    case Expr::PredefinedExprClass: {
      unsigned Type = cast<PredefinedExpr>(E)->getIdentType();
      if (CGF) {
        LValue Res = CGF->EmitPredefinedLValue(cast<PredefinedExpr>(E));
        return cast<llvm::Constant>(Res.getAddress());
      } else if (Type == PredefinedExpr::PrettyFunction) {
        return CGM.GetAddrOfConstantCString("top level", ".tmp");
      }

      return CGM.GetAddrOfConstantCString("", ".tmp");
    }
    case Expr::AddrLabelExprClass: {
      assert(CGF && "Invalid address of label expression outside function.");
      llvm::Constant *Ptr =
        CGF->GetAddrOfLabel(cast<AddrLabelExpr>(E)->getLabel());
      return llvm::ConstantExpr::getBitCast(Ptr, ConvertType(E->getType()));
    }
    case Expr::CallExprClass: {
      CallExpr* CE = cast<CallExpr>(E);
      unsigned builtin = CE->isBuiltinCall(CGM.getContext());
      if (builtin !=
            Builtin::BI__builtin___CFStringMakeConstantString &&
          builtin !=
            Builtin::BI__builtin___NSStringMakeConstantString)
        break;
      const Expr *Arg = CE->getArg(0)->IgnoreParenCasts();
      const StringLiteral *Literal = cast<StringLiteral>(Arg);
      if (builtin ==
            Builtin::BI__builtin___NSStringMakeConstantString) {
        return CGM.getObjCRuntime().GenerateConstantString(Literal);
      }
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

  if (DestType->isReferenceType())
    Success = E->EvaluateAsLValue(Result, Context);
  else
    Success = E->Evaluate(Result, Context);

  if (Success && !Result.HasSideEffects) {
    switch (Result.Val.getKind()) {
    case APValue::Uninitialized:
      assert(0 && "Constant expressions should be initialized.");
      return 0;
    case APValue::LValue: {
      const llvm::Type *DestTy = getTypes().ConvertTypeForMem(DestType);
      llvm::Constant *Offset =
        llvm::ConstantInt::get(llvm::Type::getInt64Ty(VMContext),
                               Result.Val.getLValueOffset().getQuantity());

      llvm::Constant *C;
      if (const Expr *LVBase = Result.Val.getLValueBase()) {
        C = ConstExprEmitter(*this, CGF).EmitLValue(const_cast<Expr*>(LVBase));

        // Apply offset if necessary.
        if (!Offset->isNullValue()) {
          const llvm::Type *Type = llvm::Type::getInt8PtrTy(VMContext);
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
      llvm::Constant *C = llvm::ConstantInt::get(VMContext,
                                                 Result.Val.getInt());

      if (C->getType()->isIntegerTy(1)) {
        const llvm::Type *BoolTy = getTypes().ConvertTypeForMem(E->getType());
        C = llvm::ConstantExpr::getZExt(C, BoolTy);
      }
      return C;
    }
    case APValue::ComplexInt: {
      llvm::Constant *Complex[2];

      Complex[0] = llvm::ConstantInt::get(VMContext,
                                          Result.Val.getComplexIntReal());
      Complex[1] = llvm::ConstantInt::get(VMContext,
                                          Result.Val.getComplexIntImag());

      // FIXME: the target may want to specify that this is packed.
      return llvm::ConstantStruct::get(VMContext, Complex, 2, false);
    }
    case APValue::Float:
      return llvm::ConstantFP::get(VMContext, Result.Val.getFloat());
    case APValue::ComplexFloat: {
      llvm::Constant *Complex[2];

      Complex[0] = llvm::ConstantFP::get(VMContext,
                                         Result.Val.getComplexFloatReal());
      Complex[1] = llvm::ConstantFP::get(VMContext,
                                         Result.Val.getComplexFloatImag());

      // FIXME: the target may want to specify that this is packed.
      return llvm::ConstantStruct::get(VMContext, Complex, 2, false);
    }
    case APValue::Vector: {
      llvm::SmallVector<llvm::Constant *, 4> Inits;
      unsigned NumElts = Result.Val.getVectorLength();

      if (Context.getLangOptions().AltiVec &&
          isa<CastExpr>(E) &&
          cast<CastExpr>(E)->getCastKind() == CK_VectorSplat) {
        // AltiVec vector initialization with a single literal
        APValue &Elt = Result.Val.getVectorElt(0);

        llvm::Constant* InitValue = Elt.isInt()
          ? cast<llvm::Constant>
              (llvm::ConstantInt::get(VMContext, Elt.getInt()))
          : cast<llvm::Constant>
              (llvm::ConstantFP::get(VMContext, Elt.getFloat()));

        for (unsigned i = 0; i != NumElts; ++i)
          Inits.push_back(InitValue);

      } else {
        for (unsigned i = 0; i != NumElts; ++i) {
          APValue &Elt = Result.Val.getVectorElt(i);
          if (Elt.isInt())
            Inits.push_back(llvm::ConstantInt::get(VMContext, Elt.getInt()));
          else
            Inits.push_back(llvm::ConstantFP::get(VMContext, Elt.getFloat()));
        }
      }
      return llvm::ConstantVector::get(Inits);
    }
    }
  }

  llvm::Constant* C = ConstExprEmitter(*this, CGF).Visit(const_cast<Expr*>(E));
  if (C && C->getType()->isIntegerTy(1)) {
    const llvm::Type *BoolTy = getTypes().ConvertTypeForMem(E->getType());
    C = llvm::ConstantExpr::getZExt(C, BoolTy);
  }
  return C;
}

static uint64_t getFieldOffset(ASTContext &C, const FieldDecl *field) {
  const ASTRecordLayout &layout = C.getASTRecordLayout(field->getParent());
  return layout.getFieldOffset(field->getFieldIndex());
}
    
llvm::Constant *
CodeGenModule::getMemberPointerConstant(const UnaryOperator *uo) {
  // Member pointer constants always have a very particular form.
  const MemberPointerType *type = cast<MemberPointerType>(uo->getType());
  const ValueDecl *decl = cast<DeclRefExpr>(uo->getSubExpr())->getDecl();

  // A member function pointer.
  if (const CXXMethodDecl *method = dyn_cast<CXXMethodDecl>(decl))
    return getCXXABI().EmitMemberPointer(method);

  // Otherwise, a member data pointer.
  uint64_t fieldOffset;
  if (const FieldDecl *field = dyn_cast<FieldDecl>(decl))
    fieldOffset = getFieldOffset(getContext(), field);
  else {
    const IndirectFieldDecl *ifield = cast<IndirectFieldDecl>(decl);

    fieldOffset = 0;
    for (IndirectFieldDecl::chain_iterator ci = ifield->chain_begin(),
           ce = ifield->chain_end(); ci != ce; ++ci)
      fieldOffset += getFieldOffset(getContext(), cast<FieldDecl>(*ci));
  }

  CharUnits chars = getContext().toCharUnitsFromBits((int64_t) fieldOffset);
  return getCXXABI().EmitMemberDataPointer(type, chars);
}

static void
FillInNullDataMemberPointers(CodeGenModule &CGM, QualType T,
                             std::vector<llvm::Constant *> &Elements,
                             uint64_t StartOffset) {
  assert(StartOffset % CGM.getContext().getCharWidth() == 0 && 
         "StartOffset not byte aligned!");

  if (CGM.getTypes().isZeroInitializable(T))
    return;

  if (const ConstantArrayType *CAT = 
        CGM.getContext().getAsConstantArrayType(T)) {
    QualType ElementTy = CAT->getElementType();
    uint64_t ElementSize = CGM.getContext().getTypeSize(ElementTy);
    
    for (uint64_t I = 0, E = CAT->getSize().getZExtValue(); I != E; ++I) {
      FillInNullDataMemberPointers(CGM, ElementTy, Elements,
                                   StartOffset + I * ElementSize);
    }
  } else if (const RecordType *RT = T->getAs<RecordType>()) {
    const CXXRecordDecl *RD = cast<CXXRecordDecl>(RT->getDecl());
    const ASTRecordLayout &Layout = CGM.getContext().getASTRecordLayout(RD);

    // Go through all bases and fill in any null pointer to data members.
    for (CXXRecordDecl::base_class_const_iterator I = RD->bases_begin(),
         E = RD->bases_end(); I != E; ++I) {
      if (I->isVirtual()) {
        // Ignore virtual bases.
        continue;
      }
      
      const CXXRecordDecl *BaseDecl = 
      cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());
      
      // Ignore empty bases.
      if (BaseDecl->isEmpty())
        continue;
      
      // Ignore bases that don't have any pointer to data members.
      if (CGM.getTypes().isZeroInitializable(BaseDecl))
        continue;

      uint64_t BaseOffset = Layout.getBaseClassOffsetInBits(BaseDecl);
      FillInNullDataMemberPointers(CGM, I->getType(),
                                   Elements, StartOffset + BaseOffset);
    }
    
    // Visit all fields.
    unsigned FieldNo = 0;
    for (RecordDecl::field_iterator I = RD->field_begin(),
         E = RD->field_end(); I != E; ++I, ++FieldNo) {
      QualType FieldType = I->getType();
      
      if (CGM.getTypes().isZeroInitializable(FieldType))
        continue;

      uint64_t FieldOffset = StartOffset + Layout.getFieldOffset(FieldNo);
      FillInNullDataMemberPointers(CGM, FieldType, Elements, FieldOffset);
    }
  } else {
    assert(T->isMemberPointerType() && "Should only see member pointers here!");
    assert(!T->getAs<MemberPointerType>()->getPointeeType()->isFunctionType() &&
           "Should only see pointers to data members here!");
  
    CharUnits StartIndex = CGM.getContext().toCharUnitsFromBits(StartOffset);
    CharUnits EndIndex = StartIndex + CGM.getContext().getTypeSizeInChars(T);

    // FIXME: hardcodes Itanium member pointer representation!
    llvm::Constant *NegativeOne =
      llvm::ConstantInt::get(llvm::Type::getInt8Ty(CGM.getLLVMContext()),
                             -1ULL, /*isSigned*/true);

    // Fill in the null data member pointer.
    for (CharUnits I = StartIndex; I != EndIndex; ++I)
      Elements[I.getQuantity()] = NegativeOne;
  }
}

static llvm::Constant *EmitNullConstantForBase(CodeGenModule &CGM,
                                               const llvm::Type *baseType,
                                               const CXXRecordDecl *base);

static llvm::Constant *EmitNullConstant(CodeGenModule &CGM,
                                        const CXXRecordDecl *record,
                                        bool asCompleteObject) {
  const CGRecordLayout &layout = CGM.getTypes().getCGRecordLayout(record);
  const llvm::StructType *structure =
    (asCompleteObject ? layout.getLLVMType()
                      : layout.getBaseSubobjectLLVMType());

  unsigned numElements = structure->getNumElements();
  std::vector<llvm::Constant *> elements(numElements);

  // Fill in all the bases.
  for (CXXRecordDecl::base_class_const_iterator
         I = record->bases_begin(), E = record->bases_end(); I != E; ++I) {
    if (I->isVirtual()) {
      // Ignore virtual bases; if we're laying out for a complete
      // object, we'll lay these out later.
      continue;
    }

    const CXXRecordDecl *base = 
      cast<CXXRecordDecl>(I->getType()->castAs<RecordType>()->getDecl());

    // Ignore empty bases.
    if (base->isEmpty())
      continue;
    
    unsigned fieldIndex = layout.getNonVirtualBaseLLVMFieldNo(base);
    const llvm::Type *baseType = structure->getElementType(fieldIndex);
    elements[fieldIndex] = EmitNullConstantForBase(CGM, baseType, base);
  }

  // Fill in all the fields.
  for (RecordDecl::field_iterator I = record->field_begin(),
         E = record->field_end(); I != E; ++I) {
    const FieldDecl *field = *I;
    
    // Ignore bit fields.
    if (field->isBitField())
      continue;
    
    unsigned fieldIndex = layout.getLLVMFieldNo(field);
    elements[fieldIndex] = CGM.EmitNullConstant(field->getType());
  }

  // Fill in the virtual bases, if we're working with the complete object.
  if (asCompleteObject) {
    for (CXXRecordDecl::base_class_const_iterator
           I = record->vbases_begin(), E = record->vbases_end(); I != E; ++I) {
      const CXXRecordDecl *base = 
        cast<CXXRecordDecl>(I->getType()->castAs<RecordType>()->getDecl());

      // Ignore empty bases.
      if (base->isEmpty())
        continue;

      unsigned fieldIndex = layout.getVirtualBaseIndex(base);

      // We might have already laid this field out.
      if (elements[fieldIndex]) continue;

      const llvm::Type *baseType = structure->getElementType(fieldIndex);
      elements[fieldIndex] = EmitNullConstantForBase(CGM, baseType, base);
    }
  }

  // Now go through all other fields and zero them out.
  for (unsigned i = 0; i != numElements; ++i) {
    if (!elements[i])
      elements[i] = llvm::Constant::getNullValue(structure->getElementType(i));
  }
  
  return llvm::ConstantStruct::get(structure, elements);
}

/// Emit the null constant for a base subobject.
static llvm::Constant *EmitNullConstantForBase(CodeGenModule &CGM,
                                               const llvm::Type *baseType,
                                               const CXXRecordDecl *base) {
  const CGRecordLayout &baseLayout = CGM.getTypes().getCGRecordLayout(base);

  // Just zero out bases that don't have any pointer to data members.
  if (baseLayout.isZeroInitializableAsBase())
    return llvm::Constant::getNullValue(baseType);

  // If the base type is a struct, we can just use its null constant.
  if (isa<llvm::StructType>(baseType)) {
    return EmitNullConstant(CGM, base, /*complete*/ false);
  }

  // Otherwise, some bases are represented as arrays of i8 if the size
  // of the base is smaller than its corresponding LLVM type.  Figure
  // out how many elements this base array has.
  const llvm::ArrayType *baseArrayType = cast<llvm::ArrayType>(baseType);
  unsigned numBaseElements = baseArrayType->getNumElements();

  // Fill in null data member pointers.
  std::vector<llvm::Constant *> baseElements(numBaseElements);
  FillInNullDataMemberPointers(CGM, CGM.getContext().getTypeDeclType(base),
                               baseElements, 0);

  // Now go through all other elements and zero them out.
  if (numBaseElements) {
    const llvm::Type *i8 = llvm::Type::getInt8Ty(CGM.getLLVMContext());
    llvm::Constant *i8_zero = llvm::Constant::getNullValue(i8);
    for (unsigned i = 0; i != numBaseElements; ++i) {
      if (!baseElements[i])
        baseElements[i] = i8_zero;
    }
  }
      
  return llvm::ConstantArray::get(baseArrayType, baseElements);
}

llvm::Constant *CodeGenModule::EmitNullConstant(QualType T) {
  if (getTypes().isZeroInitializable(T))
    return llvm::Constant::getNullValue(getTypes().ConvertTypeForMem(T));
    
  if (const ConstantArrayType *CAT = Context.getAsConstantArrayType(T)) {

    QualType ElementTy = CAT->getElementType();

    llvm::Constant *Element = EmitNullConstant(ElementTy);
    unsigned NumElements = CAT->getSize().getZExtValue();
    std::vector<llvm::Constant *> Array(NumElements);
    for (unsigned i = 0; i != NumElements; ++i)
      Array[i] = Element;

    const llvm::ArrayType *ATy =
      cast<llvm::ArrayType>(getTypes().ConvertTypeForMem(T));
    return llvm::ConstantArray::get(ATy, Array);
  }

  if (const RecordType *RT = T->getAs<RecordType>()) {
    const CXXRecordDecl *RD = cast<CXXRecordDecl>(RT->getDecl());
    return ::EmitNullConstant(*this, RD, /*complete object*/ true);
  }

  assert(T->isMemberPointerType() && "Should only see member pointers here!");
  assert(!T->getAs<MemberPointerType>()->getPointeeType()->isFunctionType() &&
         "Should only see pointers to data members here!");
  
  // Itanium C++ ABI 2.3:
  //   A NULL pointer is represented as -1.
  return getCXXABI().EmitNullMemberPointer(T->castAs<MemberPointerType>());
}
