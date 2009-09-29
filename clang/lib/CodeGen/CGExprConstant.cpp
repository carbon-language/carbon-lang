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
#include "clang/AST/RecordLayout.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/Builtins.h"
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Target/TargetData.h"
using namespace clang;
using namespace CodeGen;

namespace  {

class VISIBILITY_HIDDEN ConstStructBuilder {
  CodeGenModule &CGM;
  CodeGenFunction *CGF;

  bool Packed;

  unsigned NextFieldOffsetInBytes;

  std::vector<llvm::Constant *> Elements;

  ConstStructBuilder(CodeGenModule &CGM, CodeGenFunction *CGF)
    : CGM(CGM), CGF(CGF), Packed(false), NextFieldOffsetInBytes(0) { }

  bool AppendField(const FieldDecl *Field, uint64_t FieldOffset,
                   const Expr *InitExpr) {
    uint64_t FieldOffsetInBytes = FieldOffset / 8;

    assert(NextFieldOffsetInBytes <= FieldOffsetInBytes
           && "Field offset mismatch!");

    // Emit the field.
    llvm::Constant *C = CGM.EmitConstantExpr(InitExpr, Field->getType(), CGF);
    if (!C)
      return false;

    unsigned FieldAlignment = getAlignment(C);

    // Round up the field offset to the alignment of the field type.
    uint64_t AlignedNextFieldOffsetInBytes =
      llvm::RoundUpToAlignment(NextFieldOffsetInBytes, FieldAlignment);

    if (AlignedNextFieldOffsetInBytes > FieldOffsetInBytes) {
      std::vector<llvm::Constant *> PackedElements;

      assert(!Packed && "Alignment is wrong even with a packed struct!");

      // Convert the struct to a packed struct.
      uint64_t ElementOffsetInBytes = 0;

      for (unsigned i = 0, e = Elements.size(); i != e; ++i) {
        llvm::Constant *C = Elements[i];

        unsigned ElementAlign =
          CGM.getTargetData().getABITypeAlignment(C->getType());
        uint64_t AlignedElementOffsetInBytes =
          llvm::RoundUpToAlignment(ElementOffsetInBytes, ElementAlign);

        if (AlignedElementOffsetInBytes > ElementOffsetInBytes) {
          // We need some padding.
          uint64_t NumBytes =
            AlignedElementOffsetInBytes - ElementOffsetInBytes;

          const llvm::Type *Ty = llvm::Type::getInt8Ty(CGF->getLLVMContext());
          if (NumBytes > 1)
            Ty = llvm::ArrayType::get(Ty, NumBytes);

          llvm::Constant *Padding = llvm::Constant::getNullValue(Ty);
          PackedElements.push_back(Padding);
          ElementOffsetInBytes += getSizeInBytes(Padding);
        }

        PackedElements.push_back(C);
        ElementOffsetInBytes += getSizeInBytes(C);
      }

      assert(ElementOffsetInBytes == NextFieldOffsetInBytes &&
             "Packing the struct changed its size!");

      Elements = PackedElements;
      Packed = true;
      AlignedNextFieldOffsetInBytes = NextFieldOffsetInBytes;
    }

    if (AlignedNextFieldOffsetInBytes < FieldOffsetInBytes) {
      // We need to append padding.
      AppendPadding(FieldOffsetInBytes - NextFieldOffsetInBytes);

      assert(NextFieldOffsetInBytes == FieldOffsetInBytes &&
             "Did not add enough padding!");

      AlignedNextFieldOffsetInBytes = NextFieldOffsetInBytes;
    }

    // Add the field.
    Elements.push_back(C);
    NextFieldOffsetInBytes = AlignedNextFieldOffsetInBytes + getSizeInBytes(C);

    return true;
  }

  bool AppendBitField(const FieldDecl *Field, uint64_t FieldOffset,
                      const Expr *InitExpr) {
    llvm::ConstantInt *CI =
      cast_or_null<llvm::ConstantInt>(CGM.EmitConstantExpr(InitExpr,
                                                           Field->getType(),
                                                           CGF));
    // FIXME: Can this ever happen?
    if (!CI)
      return false;

    if (FieldOffset > NextFieldOffsetInBytes * 8) {
      // We need to add padding.
      uint64_t NumBytes =
        llvm::RoundUpToAlignment(FieldOffset -
                                 NextFieldOffsetInBytes * 8, 8) / 8;

      AppendPadding(NumBytes);
    }

    uint64_t FieldSize =
      Field->getBitWidth()->EvaluateAsInt(CGM.getContext()).getZExtValue();

    llvm::APInt FieldValue = CI->getValue();

    // Promote the size of FieldValue if necessary
    // FIXME: This should never occur, but currently it can because initializer
    // constants are cast to bool, and because clang is not enforcing bitfield
    // width limits.
    if (FieldSize > FieldValue.getBitWidth())
      FieldValue.zext(FieldSize);

    // Truncate the size of FieldValue to the bit field size.
    if (FieldSize < FieldValue.getBitWidth())
      FieldValue.trunc(FieldSize);

    if (FieldOffset < NextFieldOffsetInBytes * 8) {
      // Either part of the field or the entire field can go into the previous
      // byte.
      assert(!Elements.empty() && "Elements can't be empty!");

      unsigned BitsInPreviousByte =
        NextFieldOffsetInBytes * 8 - FieldOffset;

      bool FitsCompletelyInPreviousByte =
        BitsInPreviousByte >= FieldValue.getBitWidth();

      llvm::APInt Tmp = FieldValue;

      if (!FitsCompletelyInPreviousByte) {
        unsigned NewFieldWidth = FieldSize - BitsInPreviousByte;

        if (CGM.getTargetData().isBigEndian()) {
          Tmp = Tmp.lshr(NewFieldWidth);
          Tmp.trunc(BitsInPreviousByte);

          // We want the remaining high bits.
          FieldValue.trunc(NewFieldWidth);
        } else {
          Tmp.trunc(BitsInPreviousByte);

          // We want the remaining low bits.
          FieldValue = FieldValue.lshr(BitsInPreviousByte);
          FieldValue.trunc(NewFieldWidth);
        }
      }

      Tmp.zext(8);
      if (CGM.getTargetData().isBigEndian()) {
        if (FitsCompletelyInPreviousByte)
          Tmp = Tmp.shl(BitsInPreviousByte - FieldValue.getBitWidth());
      } else {
        Tmp = Tmp.shl(8 - BitsInPreviousByte);
      }

      // Or in the bits that go into the previous byte.
      Tmp |= cast<llvm::ConstantInt>(Elements.back())->getValue();
      Elements.back() = llvm::ConstantInt::get(CGM.getLLVMContext(), Tmp);

      if (FitsCompletelyInPreviousByte)
        return true;
    }

    while (FieldValue.getBitWidth() > 8) {
      llvm::APInt Tmp;

      if (CGM.getTargetData().isBigEndian()) {
        // We want the high bits.
        Tmp = FieldValue;
        Tmp = Tmp.lshr(Tmp.getBitWidth() - 8);
        Tmp.trunc(8);
      } else {
        // We want the low bits.
        Tmp = FieldValue;
        Tmp.trunc(8);

        FieldValue = FieldValue.lshr(8);
      }

      Elements.push_back(llvm::ConstantInt::get(CGM.getLLVMContext(), Tmp));
      NextFieldOffsetInBytes++;

      FieldValue.trunc(FieldValue.getBitWidth() - 8);
    }

    assert(FieldValue.getBitWidth() > 0 &&
           "Should have at least one bit left!");
    assert(FieldValue.getBitWidth() <= 8 &&
           "Should not have more than a byte left!");

    if (FieldValue.getBitWidth() < 8) {
      if (CGM.getTargetData().isBigEndian()) {
        unsigned BitWidth = FieldValue.getBitWidth();

        FieldValue.zext(8);
        FieldValue = FieldValue << (8 - BitWidth);
      } else
        FieldValue.zext(8);
    }

    // Append the last element.
    Elements.push_back(llvm::ConstantInt::get(CGM.getLLVMContext(),
                                              FieldValue));
    NextFieldOffsetInBytes++;
    return true;
  }

  void AppendPadding(uint64_t NumBytes) {
    if (!NumBytes)
      return;

    const llvm::Type *Ty = llvm::Type::getInt8Ty(CGM.getLLVMContext());
    if (NumBytes > 1)
      Ty = llvm::ArrayType::get(Ty, NumBytes);

    llvm::Constant *C = llvm::Constant::getNullValue(Ty);
    Elements.push_back(C);
    assert(getAlignment(C) == 1 && "Padding must have 1 byte alignment!");

    NextFieldOffsetInBytes += getSizeInBytes(C);
  }

  void AppendTailPadding(uint64_t RecordSize) {
    assert(RecordSize % 8 == 0 && "Invalid record size!");

    uint64_t RecordSizeInBytes = RecordSize / 8;
    assert(NextFieldOffsetInBytes <= RecordSizeInBytes && "Size mismatch!");

    unsigned NumPadBytes = RecordSizeInBytes - NextFieldOffsetInBytes;
    AppendPadding(NumPadBytes);
  }

  bool Build(InitListExpr *ILE) {
    RecordDecl *RD = ILE->getType()->getAs<RecordType>()->getDecl();
    const ASTRecordLayout &Layout = CGM.getContext().getASTRecordLayout(RD);

    unsigned FieldNo = 0;
    unsigned ElementNo = 0;
    for (RecordDecl::field_iterator Field = RD->field_begin(),
         FieldEnd = RD->field_end();
         ElementNo < ILE->getNumInits() && Field != FieldEnd;
         ++Field, ++FieldNo) {
      if (RD->isUnion() && ILE->getInitializedFieldInUnion() != *Field)
        continue;

      if (Field->isBitField()) {
        if (!Field->getIdentifier())
          continue;

        if (!AppendBitField(*Field, Layout.getFieldOffset(FieldNo),
                            ILE->getInit(ElementNo)))
          return false;
      } else {
        if (!AppendField(*Field, Layout.getFieldOffset(FieldNo),
                         ILE->getInit(ElementNo)))
          return false;
      }

      ElementNo++;
    }

    uint64_t LayoutSizeInBytes = Layout.getSize() / 8;

    if (NextFieldOffsetInBytes > LayoutSizeInBytes) {
      // If the struct is bigger than the size of the record type,
      // we must have a flexible array member at the end.
      assert(RD->hasFlexibleArrayMember() &&
             "Must have flexible array member if struct is bigger than type!");

      // No tail padding is necessary.
      return true;
    }

    // Append tail padding if necessary.
    AppendTailPadding(Layout.getSize());

    assert(Layout.getSize() / 8 == NextFieldOffsetInBytes &&
           "Tail padding mismatch!");

    return true;
  }

  unsigned getAlignment(const llvm::Constant *C) const {
    if (Packed)
      return 1;

    return CGM.getTargetData().getABITypeAlignment(C->getType());
  }

  uint64_t getSizeInBytes(const llvm::Constant *C) const {
    return CGM.getTargetData().getTypeAllocSize(C->getType());
  }

public:
  static llvm::Constant *BuildStruct(CodeGenModule &CGM, CodeGenFunction *CGF,
                                     InitListExpr *ILE) {
    ConstStructBuilder Builder(CGM, CGF);

    if (!Builder.Build(ILE))
      return 0;

    llvm::Constant *Result =
      llvm::ConstantStruct::get(CGM.getLLVMContext(),
                                Builder.Elements, Builder.Packed);

    assert(llvm::RoundUpToAlignment(Builder.NextFieldOffsetInBytes,
                                    Builder.getAlignment(Result)) ==
           Builder.getSizeInBytes(Result) && "Size mismatch!");

    return Result;
  }
};

class VISIBILITY_HIDDEN ConstExprEmitter :
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

  llvm::Constant *VisitCompoundLiteralExpr(CompoundLiteralExpr *E) {
    return Visit(E->getInitializer());
  }

  llvm::Constant *VisitCastExpr(CastExpr* E) {
    switch (E->getCastKind()) {
    case CastExpr::CK_ToUnion: {
      // GCC cast to union extension
      assert(E->getType()->isUnionType() &&
             "Destination type is not union type!");
      const llvm::Type *Ty = ConvertType(E->getType());
      Expr *SubExpr = E->getSubExpr();

      llvm::Constant *C =
        CGM.EmitConstantExpr(SubExpr, SubExpr->getType(), CGF);
      if (!C)
        return 0;

      // Build a struct with the union sub-element as the first member,
      // and padded to the appropriate size
      std::vector<llvm::Constant*> Elts;
      std::vector<const llvm::Type*> Types;
      Elts.push_back(C);
      Types.push_back(C->getType());
      unsigned CurSize = CGM.getTargetData().getTypeAllocSize(C->getType());
      unsigned TotalSize = CGM.getTargetData().getTypeAllocSize(Ty);

      assert(CurSize <= TotalSize && "Union size mismatch!");
      if (unsigned NumPadBytes = TotalSize - CurSize) {
        const llvm::Type *Ty = llvm::Type::getInt8Ty(VMContext);
        if (NumPadBytes > 1)
          Ty = llvm::ArrayType::get(Ty, NumPadBytes);

        Elts.push_back(llvm::Constant::getNullValue(Ty));
        Types.push_back(Ty);
      }

      llvm::StructType* STy =
        llvm::StructType::get(C->getType()->getContext(), Types, false);
      return llvm::ConstantStruct::get(STy, Elts);
    }
    case CastExpr::CK_NullToMemberPointer:
      return CGM.EmitNullConstant(E->getType());
    default: {
      // FIXME: This should be handled by the CK_NoOp cast kind.
      // Explicit and implicit no-op casts
      QualType Ty = E->getType(), SubTy = E->getSubExpr()->getType();
      if (CGM.getContext().hasSameUnqualifiedType(Ty, SubTy))
          return Visit(E->getSubExpr());
      return 0;
    }
    }
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
    // FIXME: This doesn't handle member pointers correctly!
    for (; i < NumElements; ++i)
      Elts.push_back(llvm::Constant::getNullValue(ElemTy));

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
                                     E->getType().getAddressSpace());
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
      unsigned id =
          CGF->GetIDForAddrOfLabel(cast<AddrLabelExpr>(E)->getLabel());
      llvm::Constant *C =
            llvm::ConstantInt::get(llvm::Type::getInt32Ty(VMContext), id);
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

  if (DestType->isReferenceType())
    Success = E->EvaluateAsLValue(Result, Context);
  else
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
        llvm::ConstantInt::get(llvm::Type::getInt64Ty(VMContext),
                               Result.Val.getLValueOffset());

      llvm::Constant *C;
      if (const Expr *LVBase = Result.Val.getLValueBase()) {
        C = ConstExprEmitter(*this, CGF).EmitLValue(const_cast<Expr*>(LVBase));

        // Apply offset if necessary.
        if (!Offset->isNullValue()) {
          const llvm::Type *Type =
            llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(VMContext));
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

      if (C->getType() == llvm::Type::getInt1Ty(VMContext)) {
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

      for (unsigned i = 0; i != NumElts; ++i) {
        APValue &Elt = Result.Val.getVectorElt(i);
        if (Elt.isInt())
          Inits.push_back(llvm::ConstantInt::get(VMContext, Elt.getInt()));
        else
          Inits.push_back(llvm::ConstantFP::get(VMContext, Elt.getFloat()));
      }
      return llvm::ConstantVector::get(&Inits[0], Inits.size());
    }
    }
  }

  llvm::Constant* C = ConstExprEmitter(*this, CGF).Visit(const_cast<Expr*>(E));
  if (C && C->getType() == llvm::Type::getInt1Ty(VMContext)) {
    const llvm::Type *BoolTy = getTypes().ConvertTypeForMem(E->getType());
    C = llvm::ConstantExpr::getZExt(C, BoolTy);
  }
  return C;
}

static inline bool isDataMemberPointerType(QualType T) {
  if (const MemberPointerType *MPT = T->getAs<MemberPointerType>())
    return !MPT->getPointeeType()->isFunctionType();
  
  return false;
}

llvm::Constant *CodeGenModule::EmitNullConstant(QualType T) {
  // No need to check for member pointers when not compiling C++.
  if (!getContext().getLangOptions().CPlusPlus)
    return llvm::Constant::getNullValue(getTypes().ConvertTypeForMem(T));

  if (const ConstantArrayType *CAT = Context.getAsConstantArrayType(T)) {

    QualType ElementTy = CAT->getElementType();

    // FIXME: Handle arrays of structs that contain member pointers.
    if (isDataMemberPointerType(Context.getBaseElementType(ElementTy))) {
      llvm::Constant *Element = EmitNullConstant(ElementTy);
      uint64_t NumElements = CAT->getSize().getZExtValue();
      std::vector<llvm::Constant *> Array(NumElements);
      for (uint64_t i = 0; i != NumElements; ++i)
        Array[i] = Element;

      const llvm::ArrayType *ATy =
        cast<llvm::ArrayType>(getTypes().ConvertTypeForMem(T));
      return llvm::ConstantArray::get(ATy, Array);
    }
  }

  if (const RecordType *RT = T->getAs<RecordType>()) {
    const RecordDecl *RD = RT->getDecl();
    // FIXME: It would be better if there was a way to explicitly compute the
    // record layout instead of converting to a type.
    Types.ConvertTagDeclType(RD);

    const CGRecordLayout &Layout = Types.getCGRecordLayout(RD);
    if (Layout.containsMemberPointer()) {
      assert(0 && "FIXME: No support for structs with member pointers yet!");
    }
  }

  // FIXME: Handle structs that contain member pointers.
  if (isDataMemberPointerType(T))
    return llvm::Constant::getAllOnesValue(getTypes().ConvertTypeForMem(T));

  return llvm::Constant::getNullValue(getTypes().ConvertTypeForMem(T));
}
