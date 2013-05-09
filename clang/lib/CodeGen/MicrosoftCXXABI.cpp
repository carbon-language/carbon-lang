//===--- MicrosoftCXXABI.cpp - Emit LLVM Code from ASTs for a Module ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This provides C++ code generation targeting the Microsoft Visual C++ ABI.
// The class in this file generates structures that follow the Microsoft
// Visual C++ ABI, which is actually not very well documented at all outside
// of Microsoft.
//
//===----------------------------------------------------------------------===//

#include "CGCXXABI.h"
#include "CodeGenModule.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"

using namespace clang;
using namespace CodeGen;

namespace {

class MicrosoftCXXABI : public CGCXXABI {
public:
  MicrosoftCXXABI(CodeGenModule &CGM) : CGCXXABI(CGM) {}

  bool isReturnTypeIndirect(const CXXRecordDecl *RD) const {
    // Structures that are not C++03 PODs are always indirect.
    return !RD->isPOD();
  }

  RecordArgABI getRecordArgABI(const CXXRecordDecl *RD) const {
    if (RD->hasNonTrivialCopyConstructor())
      return RAA_DirectInMemory;
    return RAA_Default;
  }

  StringRef GetPureVirtualCallName() { return "_purecall"; }
  // No known support for deleted functions in MSVC yet, so this choice is
  // arbitrary.
  StringRef GetDeletedVirtualCallName() { return "_purecall"; }

  llvm::Value *adjustToCompleteObject(CodeGenFunction &CGF,
                                      llvm::Value *ptr,
                                      QualType type);

  void BuildConstructorSignature(const CXXConstructorDecl *Ctor,
                                 CXXCtorType Type,
                                 CanQualType &ResTy,
                                 SmallVectorImpl<CanQualType> &ArgTys);

  llvm::BasicBlock *EmitCtorCompleteObjectHandler(CodeGenFunction &CGF);

  void BuildDestructorSignature(const CXXDestructorDecl *Ctor,
                                CXXDtorType Type,
                                CanQualType &ResTy,
                                SmallVectorImpl<CanQualType> &ArgTys);

  void BuildInstanceFunctionParams(CodeGenFunction &CGF,
                                   QualType &ResTy,
                                   FunctionArgList &Params);

  void EmitInstanceFunctionProlog(CodeGenFunction &CGF);

  llvm::Value *EmitConstructorCall(CodeGenFunction &CGF,
                           const CXXConstructorDecl *D,
                           CXXCtorType Type, bool ForVirtualBase,
                           bool Delegating,
                           llvm::Value *This,
                           CallExpr::const_arg_iterator ArgBeg,
                           CallExpr::const_arg_iterator ArgEnd);

  RValue EmitVirtualDestructorCall(CodeGenFunction &CGF,
                                   const CXXDestructorDecl *Dtor,
                                   CXXDtorType DtorType,
                                   SourceLocation CallLoc,
                                   ReturnValueSlot ReturnValue,
                                   llvm::Value *This);

  void EmitGuardedInit(CodeGenFunction &CGF, const VarDecl &D,
                       llvm::GlobalVariable *DeclPtr,
                       bool PerformInit);

  // ==== Notes on array cookies =========
  //
  // MSVC seems to only use cookies when the class has a destructor; a
  // two-argument usual array deallocation function isn't sufficient.
  //
  // For example, this code prints "100" and "1":
  //   struct A {
  //     char x;
  //     void *operator new[](size_t sz) {
  //       printf("%u\n", sz);
  //       return malloc(sz);
  //     }
  //     void operator delete[](void *p, size_t sz) {
  //       printf("%u\n", sz);
  //       free(p);
  //     }
  //   };
  //   int main() {
  //     A *p = new A[100];
  //     delete[] p;
  //   }
  // Whereas it prints "104" and "104" if you give A a destructor.

  bool requiresArrayCookie(const CXXDeleteExpr *expr, QualType elementType);
  bool requiresArrayCookie(const CXXNewExpr *expr);
  CharUnits getArrayCookieSizeImpl(QualType type);
  llvm::Value *InitializeArrayCookie(CodeGenFunction &CGF,
                                     llvm::Value *NewPtr,
                                     llvm::Value *NumElements,
                                     const CXXNewExpr *expr,
                                     QualType ElementType);
  llvm::Value *readArrayCookieImpl(CodeGenFunction &CGF,
                                   llvm::Value *allocPtr,
                                   CharUnits cookieSize);
  static bool needThisReturn(GlobalDecl GD);

private:
  llvm::Constant *getZeroInt() {
    return llvm::ConstantInt::get(CGM.IntTy, 0);
  }

  llvm::Constant *getAllOnesInt() {
    return  llvm::Constant::getAllOnesValue(CGM.IntTy);
  }

  llvm::Constant *getConstantOrZeroInt(llvm::Constant *C) {
    return C ? C : getZeroInt();
  }

  llvm::Value *getValueOrZeroInt(llvm::Value *C) {
    return C ? C : getZeroInt();
  }

  void
  GetNullMemberPointerFields(const MemberPointerType *MPT,
                             llvm::SmallVectorImpl<llvm::Constant *> &fields);

  llvm::Value *AdjustVirtualBase(CodeGenFunction &CGF, const CXXRecordDecl *RD,
                                 llvm::Value *Base,
                                 llvm::Value *VirtualBaseAdjustmentOffset,
                                 llvm::Value *VBPtrOffset /* optional */);

  /// \brief Emits a full member pointer with the fields common to data and
  /// function member pointers.
  llvm::Constant *EmitFullMemberPointer(llvm::Constant *FirstField,
                                        bool IsMemberFunction,
                                        const CXXRecordDecl *RD,
                                        CharUnits NonVirtualBaseAdjustment);

  llvm::Constant *BuildMemberPointer(const CXXRecordDecl *RD,
                                     const CXXMethodDecl *MD,
                                     CharUnits NonVirtualBaseAdjustment);

  bool MemberPointerConstantIsNull(const MemberPointerType *MPT,
                                   llvm::Constant *MP);

public:
  virtual llvm::Type *ConvertMemberPointerType(const MemberPointerType *MPT);

  virtual bool isZeroInitializable(const MemberPointerType *MPT);

  virtual llvm::Constant *EmitNullMemberPointer(const MemberPointerType *MPT);

  virtual llvm::Constant *EmitMemberDataPointer(const MemberPointerType *MPT,
                                                CharUnits offset);
  virtual llvm::Constant *EmitMemberPointer(const CXXMethodDecl *MD);
  virtual llvm::Constant *EmitMemberPointer(const APValue &MP, QualType MPT);

  virtual llvm::Value *EmitMemberPointerComparison(CodeGenFunction &CGF,
                                                   llvm::Value *L,
                                                   llvm::Value *R,
                                                   const MemberPointerType *MPT,
                                                   bool Inequality);

  virtual llvm::Value *EmitMemberPointerIsNotNull(CodeGenFunction &CGF,
                                                  llvm::Value *MemPtr,
                                                  const MemberPointerType *MPT);

  virtual llvm::Value *EmitMemberDataPointerAddress(CodeGenFunction &CGF,
                                                    llvm::Value *Base,
                                                    llvm::Value *MemPtr,
                                                  const MemberPointerType *MPT);

  virtual llvm::Value *EmitMemberPointerConversion(CodeGenFunction &CGF,
                                                   const CastExpr *E,
                                                   llvm::Value *Src);

  virtual llvm::Constant *EmitMemberPointerConversion(const CastExpr *E,
                                                      llvm::Constant *Src);

  virtual llvm::Value *
  EmitLoadOfMemberFunctionPointer(CodeGenFunction &CGF,
                                  llvm::Value *&This,
                                  llvm::Value *MemPtr,
                                  const MemberPointerType *MPT);

};

}

llvm::Value *MicrosoftCXXABI::adjustToCompleteObject(CodeGenFunction &CGF,
                                                     llvm::Value *ptr,
                                                     QualType type) {
  // FIXME: implement
  return ptr;
}

bool MicrosoftCXXABI::needThisReturn(GlobalDecl GD) {
  const CXXMethodDecl* MD = cast<CXXMethodDecl>(GD.getDecl());
  return isa<CXXConstructorDecl>(MD);
}

void MicrosoftCXXABI::BuildConstructorSignature(const CXXConstructorDecl *Ctor,
                                 CXXCtorType Type,
                                 CanQualType &ResTy,
                                 SmallVectorImpl<CanQualType> &ArgTys) {
  // 'this' is already in place

  // Ctor returns this ptr
  ResTy = ArgTys[0];

  const CXXRecordDecl *Class = Ctor->getParent();
  if (Class->getNumVBases()) {
    // Constructors of classes with virtual bases take an implicit parameter.
    ArgTys.push_back(CGM.getContext().IntTy);
  }
}

llvm::BasicBlock *MicrosoftCXXABI::EmitCtorCompleteObjectHandler(
                                                         CodeGenFunction &CGF) {
  llvm::Value *IsMostDerivedClass = getStructorImplicitParamValue(CGF);
  assert(IsMostDerivedClass &&
         "ctor for a class with virtual bases must have an implicit parameter");
  llvm::Value *IsCompleteObject
    = CGF.Builder.CreateIsNotNull(IsMostDerivedClass, "is_complete_object");

  llvm::BasicBlock *CallVbaseCtorsBB = CGF.createBasicBlock("ctor.init_vbases");
  llvm::BasicBlock *SkipVbaseCtorsBB = CGF.createBasicBlock("ctor.skip_vbases");
  CGF.Builder.CreateCondBr(IsCompleteObject,
                           CallVbaseCtorsBB, SkipVbaseCtorsBB);

  CGF.EmitBlock(CallVbaseCtorsBB);
  // FIXME: emit vbtables somewhere around here.

  // CGF will put the base ctor calls in this basic block for us later.

  return SkipVbaseCtorsBB;
}

void MicrosoftCXXABI::BuildDestructorSignature(const CXXDestructorDecl *Dtor,
                                               CXXDtorType Type,
                                               CanQualType &ResTy,
                                        SmallVectorImpl<CanQualType> &ArgTys) {
  // 'this' is already in place
  // TODO: 'for base' flag

  if (Type == Dtor_Deleting) {
    // The scalar deleting destructor takes an implicit bool parameter.
    ArgTys.push_back(CGM.getContext().BoolTy);
  }
}

static bool IsDeletingDtor(GlobalDecl GD) {
  const CXXMethodDecl* MD = cast<CXXMethodDecl>(GD.getDecl());
  if (isa<CXXDestructorDecl>(MD)) {
    return GD.getDtorType() == Dtor_Deleting;
  }
  return false;
}

void MicrosoftCXXABI::BuildInstanceFunctionParams(CodeGenFunction &CGF,
                                                  QualType &ResTy,
                                                  FunctionArgList &Params) {
  BuildThisParam(CGF, Params);
  if (needThisReturn(CGF.CurGD)) {
    ResTy = Params[0]->getType();
  }

  ASTContext &Context = getContext();
  const CXXMethodDecl *MD = cast<CXXMethodDecl>(CGF.CurGD.getDecl());
  if (isa<CXXConstructorDecl>(MD) && MD->getParent()->getNumVBases()) {
    ImplicitParamDecl *IsMostDerived
      = ImplicitParamDecl::Create(Context, 0,
                                  CGF.CurGD.getDecl()->getLocation(),
                                  &Context.Idents.get("is_most_derived"),
                                  Context.IntTy);
    Params.push_back(IsMostDerived);
    getStructorImplicitParamDecl(CGF) = IsMostDerived;
  } else if (IsDeletingDtor(CGF.CurGD)) {
    ImplicitParamDecl *ShouldDelete
      = ImplicitParamDecl::Create(Context, 0,
                                  CGF.CurGD.getDecl()->getLocation(),
                                  &Context.Idents.get("should_call_delete"),
                                  Context.BoolTy);
    Params.push_back(ShouldDelete);
    getStructorImplicitParamDecl(CGF) = ShouldDelete;
  }
}

void MicrosoftCXXABI::EmitInstanceFunctionProlog(CodeGenFunction &CGF) {
  EmitThisParam(CGF);
  if (needThisReturn(CGF.CurGD)) {
    CGF.Builder.CreateStore(getThisValue(CGF), CGF.ReturnValue);
  }

  const CXXMethodDecl *MD = cast<CXXMethodDecl>(CGF.CurGD.getDecl());
  if (isa<CXXConstructorDecl>(MD) && MD->getParent()->getNumVBases()) {
    assert(getStructorImplicitParamDecl(CGF) &&
           "no implicit parameter for a constructor with virtual bases?");
    getStructorImplicitParamValue(CGF)
      = CGF.Builder.CreateLoad(
          CGF.GetAddrOfLocalVar(getStructorImplicitParamDecl(CGF)),
          "is_most_derived");
  }

  if (IsDeletingDtor(CGF.CurGD)) {
    assert(getStructorImplicitParamDecl(CGF) &&
           "no implicit parameter for a deleting destructor?");
    getStructorImplicitParamValue(CGF)
      = CGF.Builder.CreateLoad(
          CGF.GetAddrOfLocalVar(getStructorImplicitParamDecl(CGF)),
          "should_call_delete");
  }
}

llvm::Value *MicrosoftCXXABI::EmitConstructorCall(CodeGenFunction &CGF,
                                          const CXXConstructorDecl *D,
                                          CXXCtorType Type, bool ForVirtualBase,
                                          bool Delegating,
                                          llvm::Value *This,
                                          CallExpr::const_arg_iterator ArgBeg,
                                          CallExpr::const_arg_iterator ArgEnd) {
  assert(Type == Ctor_Complete || Type == Ctor_Base);
  llvm::Value *Callee = CGM.GetAddrOfCXXConstructor(D, Ctor_Complete);

  llvm::Value *ImplicitParam = 0;
  QualType ImplicitParamTy;
  if (D->getParent()->getNumVBases()) {
    ImplicitParam = llvm::ConstantInt::get(CGM.Int32Ty, Type == Ctor_Complete);
    ImplicitParamTy = getContext().IntTy;
  }

  // FIXME: Provide a source location here.
  CGF.EmitCXXMemberCall(D, SourceLocation(), Callee, ReturnValueSlot(), This,
                        ImplicitParam, ImplicitParamTy,
                        ArgBeg, ArgEnd);
  return Callee;
}

RValue MicrosoftCXXABI::EmitVirtualDestructorCall(CodeGenFunction &CGF,
                                                  const CXXDestructorDecl *Dtor,
                                                  CXXDtorType DtorType,
                                                  SourceLocation CallLoc,
                                                  ReturnValueSlot ReturnValue,
                                                  llvm::Value *This) {
  assert(DtorType == Dtor_Deleting || DtorType == Dtor_Complete);

  // We have only one destructor in the vftable but can get both behaviors
  // by passing an implicit bool parameter.
  const CGFunctionInfo *FInfo
      = &CGM.getTypes().arrangeCXXDestructor(Dtor, Dtor_Deleting);
  llvm::Type *Ty = CGF.CGM.getTypes().GetFunctionType(*FInfo);
  llvm::Value *Callee = CGF.BuildVirtualCall(Dtor, Dtor_Deleting, This, Ty);

  ASTContext &Context = CGF.getContext();
  llvm::Value *ImplicitParam
    = llvm::ConstantInt::get(llvm::IntegerType::getInt1Ty(CGF.getLLVMContext()),
                             DtorType == Dtor_Deleting);

  return CGF.EmitCXXMemberCall(Dtor, CallLoc, Callee, ReturnValue, This,
                               ImplicitParam, Context.BoolTy, 0, 0);
}

bool MicrosoftCXXABI::requiresArrayCookie(const CXXDeleteExpr *expr,
                                   QualType elementType) {
  // Microsoft seems to completely ignore the possibility of a
  // two-argument usual deallocation function.
  return elementType.isDestructedType();
}

bool MicrosoftCXXABI::requiresArrayCookie(const CXXNewExpr *expr) {
  // Microsoft seems to completely ignore the possibility of a
  // two-argument usual deallocation function.
  return expr->getAllocatedType().isDestructedType();
}

CharUnits MicrosoftCXXABI::getArrayCookieSizeImpl(QualType type) {
  // The array cookie is always a size_t; we then pad that out to the
  // alignment of the element type.
  ASTContext &Ctx = getContext();
  return std::max(Ctx.getTypeSizeInChars(Ctx.getSizeType()),
                  Ctx.getTypeAlignInChars(type));
}

llvm::Value *MicrosoftCXXABI::readArrayCookieImpl(CodeGenFunction &CGF,
                                                  llvm::Value *allocPtr,
                                                  CharUnits cookieSize) {
  unsigned AS = allocPtr->getType()->getPointerAddressSpace();
  llvm::Value *numElementsPtr =
    CGF.Builder.CreateBitCast(allocPtr, CGF.SizeTy->getPointerTo(AS));
  return CGF.Builder.CreateLoad(numElementsPtr);
}

llvm::Value* MicrosoftCXXABI::InitializeArrayCookie(CodeGenFunction &CGF,
                                                    llvm::Value *newPtr,
                                                    llvm::Value *numElements,
                                                    const CXXNewExpr *expr,
                                                    QualType elementType) {
  assert(requiresArrayCookie(expr));

  // The size of the cookie.
  CharUnits cookieSize = getArrayCookieSizeImpl(elementType);

  // Compute an offset to the cookie.
  llvm::Value *cookiePtr = newPtr;

  // Write the number of elements into the appropriate slot.
  unsigned AS = newPtr->getType()->getPointerAddressSpace();
  llvm::Value *numElementsPtr
    = CGF.Builder.CreateBitCast(cookiePtr, CGF.SizeTy->getPointerTo(AS));
  CGF.Builder.CreateStore(numElements, numElementsPtr);

  // Finally, compute a pointer to the actual data buffer by skipping
  // over the cookie completely.
  return CGF.Builder.CreateConstInBoundsGEP1_64(newPtr,
                                                cookieSize.getQuantity());
}

void MicrosoftCXXABI::EmitGuardedInit(CodeGenFunction &CGF, const VarDecl &D,
                                      llvm::GlobalVariable *DeclPtr,
                                      bool PerformInit) {
  // FIXME: this code was only tested for global initialization.
  // Not sure whether we want thread-safe static local variables as VS
  // doesn't make them thread-safe.

  if (D.getTLSKind())
    CGM.ErrorUnsupported(&D, "dynamic TLS initialization");

  // Emit the initializer and add a global destructor if appropriate.
  CGF.EmitCXXGlobalVarDeclInit(D, DeclPtr, PerformInit);
}

// Member pointer helpers.
static bool hasVBPtrOffsetField(MSInheritanceModel Inheritance) {
  return Inheritance == MSIM_Unspecified;
}

static bool hasOnlyOneField(bool IsMemberFunction,
                            MSInheritanceModel Inheritance) {
  return Inheritance <= MSIM_SinglePolymorphic ||
      (!IsMemberFunction && Inheritance <= MSIM_MultiplePolymorphic);
}

// Only member pointers to functions need a this adjustment, since it can be
// combined with the field offset for data pointers.
static bool hasNonVirtualBaseAdjustmentField(bool IsMemberFunction,
                                             MSInheritanceModel Inheritance) {
  return (IsMemberFunction && Inheritance >= MSIM_Multiple);
}

static bool hasVirtualBaseAdjustmentField(MSInheritanceModel Inheritance) {
  return Inheritance >= MSIM_Virtual;
}

// Use zero for the field offset of a null data member pointer if we can
// guarantee that zero is not a valid field offset, or if the member pointer has
// multiple fields.  Polymorphic classes have a vfptr at offset zero, so we can
// use zero for null.  If there are multiple fields, we can use zero even if it
// is a valid field offset because null-ness testing will check the other
// fields.
static bool nullFieldOffsetIsZero(MSInheritanceModel Inheritance) {
  return Inheritance != MSIM_Multiple && Inheritance != MSIM_Single;
}

bool MicrosoftCXXABI::isZeroInitializable(const MemberPointerType *MPT) {
  // Null-ness for function memptrs only depends on the first field, which is
  // the function pointer.  The rest don't matter, so we can zero initialize.
  if (MPT->isMemberFunctionPointer())
    return true;

  // The virtual base adjustment field is always -1 for null, so if we have one
  // we can't zero initialize.  The field offset is sometimes also -1 if 0 is a
  // valid field offset.
  const CXXRecordDecl *RD = MPT->getClass()->getAsCXXRecordDecl();
  MSInheritanceModel Inheritance = RD->getMSInheritanceModel();
  return (!hasVirtualBaseAdjustmentField(Inheritance) &&
          nullFieldOffsetIsZero(Inheritance));
}

llvm::Type *
MicrosoftCXXABI::ConvertMemberPointerType(const MemberPointerType *MPT) {
  const CXXRecordDecl *RD = MPT->getClass()->getAsCXXRecordDecl();
  MSInheritanceModel Inheritance = RD->getMSInheritanceModel();
  llvm::SmallVector<llvm::Type *, 4> fields;
  if (MPT->isMemberFunctionPointer())
    fields.push_back(CGM.VoidPtrTy);  // FunctionPointerOrVirtualThunk
  else
    fields.push_back(CGM.IntTy);  // FieldOffset

  if (hasNonVirtualBaseAdjustmentField(MPT->isMemberFunctionPointer(),
                                       Inheritance))
    fields.push_back(CGM.IntTy);
  if (hasVBPtrOffsetField(Inheritance))
    fields.push_back(CGM.IntTy);
  if (hasVirtualBaseAdjustmentField(Inheritance))
    fields.push_back(CGM.IntTy);  // VirtualBaseAdjustmentOffset

  if (fields.size() == 1)
    return fields[0];
  return llvm::StructType::get(CGM.getLLVMContext(), fields);
}

void MicrosoftCXXABI::
GetNullMemberPointerFields(const MemberPointerType *MPT,
                           llvm::SmallVectorImpl<llvm::Constant *> &fields) {
  assert(fields.empty());
  const CXXRecordDecl *RD = MPT->getClass()->getAsCXXRecordDecl();
  MSInheritanceModel Inheritance = RD->getMSInheritanceModel();
  if (MPT->isMemberFunctionPointer()) {
    // FunctionPointerOrVirtualThunk
    fields.push_back(llvm::Constant::getNullValue(CGM.VoidPtrTy));
  } else {
    if (nullFieldOffsetIsZero(Inheritance))
      fields.push_back(getZeroInt());  // FieldOffset
    else
      fields.push_back(getAllOnesInt());  // FieldOffset
  }

  if (hasNonVirtualBaseAdjustmentField(MPT->isMemberFunctionPointer(),
                                       Inheritance))
    fields.push_back(getZeroInt());
  if (hasVBPtrOffsetField(Inheritance))
    fields.push_back(getZeroInt());
  if (hasVirtualBaseAdjustmentField(Inheritance))
    fields.push_back(getAllOnesInt());
}

llvm::Constant *
MicrosoftCXXABI::EmitNullMemberPointer(const MemberPointerType *MPT) {
  llvm::SmallVector<llvm::Constant *, 4> fields;
  GetNullMemberPointerFields(MPT, fields);
  if (fields.size() == 1)
    return fields[0];
  llvm::Constant *Res = llvm::ConstantStruct::getAnon(fields);
  assert(Res->getType() == ConvertMemberPointerType(MPT));
  return Res;
}

llvm::Constant *
MicrosoftCXXABI::EmitFullMemberPointer(llvm::Constant *FirstField,
                                       bool IsMemberFunction,
                                       const CXXRecordDecl *RD,
                                       CharUnits NonVirtualBaseAdjustment)
{
  MSInheritanceModel Inheritance = RD->getMSInheritanceModel();

  // Single inheritance class member pointer are represented as scalars instead
  // of aggregates.
  if (hasOnlyOneField(IsMemberFunction, Inheritance))
    return FirstField;

  llvm::SmallVector<llvm::Constant *, 4> fields;
  fields.push_back(FirstField);

  if (hasNonVirtualBaseAdjustmentField(IsMemberFunction, Inheritance))
    fields.push_back(llvm::ConstantInt::get(
      CGM.IntTy, NonVirtualBaseAdjustment.getQuantity()));

  if (hasVBPtrOffsetField(Inheritance)) {
    // FIXME: We actually need to search non-virtual bases for vbptrs.
    int64_t VBPtrOffset =
      getContext().getASTRecordLayout(RD).getVBPtrOffset().getQuantity();
    if (VBPtrOffset == -1)
      VBPtrOffset = 0;
    fields.push_back(llvm::ConstantInt::get(CGM.IntTy, VBPtrOffset));
  }

  // The rest of the fields are adjusted by conversions to a more derived class.
  if (hasVirtualBaseAdjustmentField(Inheritance))
    fields.push_back(getZeroInt());

  return llvm::ConstantStruct::getAnon(fields);
}

llvm::Constant *
MicrosoftCXXABI::EmitMemberDataPointer(const MemberPointerType *MPT,
                                       CharUnits offset) {
  const CXXRecordDecl *RD = MPT->getClass()->getAsCXXRecordDecl();
  llvm::Constant *FirstField =
    llvm::ConstantInt::get(CGM.IntTy, offset.getQuantity());
  return EmitFullMemberPointer(FirstField, /*IsMemberFunction=*/false, RD,
                               CharUnits::Zero());
}

llvm::Constant *MicrosoftCXXABI::EmitMemberPointer(const CXXMethodDecl *MD) {
  return BuildMemberPointer(MD->getParent(), MD, CharUnits::Zero());
}

llvm::Constant *MicrosoftCXXABI::EmitMemberPointer(const APValue &MP,
                                                   QualType MPType) {
  const MemberPointerType *MPT = MPType->castAs<MemberPointerType>();
  const ValueDecl *MPD = MP.getMemberPointerDecl();
  if (!MPD)
    return EmitNullMemberPointer(MPT);

  CharUnits ThisAdjustment = getMemberPointerPathAdjustment(MP);

  // FIXME PR15713: Support virtual inheritance paths.

  if (const CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(MPD))
    return BuildMemberPointer(MPT->getClass()->getAsCXXRecordDecl(),
                              MD, ThisAdjustment);

  CharUnits FieldOffset =
    getContext().toCharUnitsFromBits(getContext().getFieldOffset(MPD));
  return EmitMemberDataPointer(MPT, ThisAdjustment + FieldOffset);
}

llvm::Constant *
MicrosoftCXXABI::BuildMemberPointer(const CXXRecordDecl *RD,
                                    const CXXMethodDecl *MD,
                                    CharUnits NonVirtualBaseAdjustment) {
  assert(MD->isInstance() && "Member function must not be static!");
  MD = MD->getCanonicalDecl();
  CodeGenTypes &Types = CGM.getTypes();

  llvm::Constant *FirstField;
  if (MD->isVirtual()) {
    // FIXME: We have to instantiate a thunk that loads the vftable and jumps to
    // the right offset.
    FirstField = llvm::Constant::getNullValue(CGM.VoidPtrTy);
  } else {
    const FunctionProtoType *FPT = MD->getType()->castAs<FunctionProtoType>();
    llvm::Type *Ty;
    // Check whether the function has a computable LLVM signature.
    if (Types.isFuncTypeConvertible(FPT)) {
      // The function has a computable LLVM signature; use the correct type.
      Ty = Types.GetFunctionType(Types.arrangeCXXMethodDeclaration(MD));
    } else {
      // Use an arbitrary non-function type to tell GetAddrOfFunction that the
      // function type is incomplete.
      Ty = CGM.PtrDiffTy;
    }
    FirstField = CGM.GetAddrOfFunction(MD, Ty);
    FirstField = llvm::ConstantExpr::getBitCast(FirstField, CGM.VoidPtrTy);
  }

  // The rest of the fields are common with data member pointers.
  return EmitFullMemberPointer(FirstField, /*IsMemberFunction=*/true, RD,
                               NonVirtualBaseAdjustment);
}

/// Member pointers are the same if they're either bitwise identical *or* both
/// null.  Null-ness for function members is determined by the first field,
/// while for data member pointers we must compare all fields.
llvm::Value *
MicrosoftCXXABI::EmitMemberPointerComparison(CodeGenFunction &CGF,
                                             llvm::Value *L,
                                             llvm::Value *R,
                                             const MemberPointerType *MPT,
                                             bool Inequality) {
  CGBuilderTy &Builder = CGF.Builder;

  // Handle != comparisons by switching the sense of all boolean operations.
  llvm::ICmpInst::Predicate Eq;
  llvm::Instruction::BinaryOps And, Or;
  if (Inequality) {
    Eq = llvm::ICmpInst::ICMP_NE;
    And = llvm::Instruction::Or;
    Or = llvm::Instruction::And;
  } else {
    Eq = llvm::ICmpInst::ICMP_EQ;
    And = llvm::Instruction::And;
    Or = llvm::Instruction::Or;
  }

  // If this is a single field member pointer (single inheritance), this is a
  // single icmp.
  const CXXRecordDecl *RD = MPT->getClass()->getAsCXXRecordDecl();
  MSInheritanceModel Inheritance = RD->getMSInheritanceModel();
  if (hasOnlyOneField(MPT->isMemberFunctionPointer(), Inheritance))
    return Builder.CreateICmp(Eq, L, R);

  // Compare the first field.
  llvm::Value *L0 = Builder.CreateExtractValue(L, 0, "lhs.0");
  llvm::Value *R0 = Builder.CreateExtractValue(R, 0, "rhs.0");
  llvm::Value *Cmp0 = Builder.CreateICmp(Eq, L0, R0, "memptr.cmp.first");

  // Compare everything other than the first field.
  llvm::Value *Res = 0;
  llvm::StructType *LType = cast<llvm::StructType>(L->getType());
  for (unsigned I = 1, E = LType->getNumElements(); I != E; ++I) {
    llvm::Value *LF = Builder.CreateExtractValue(L, I);
    llvm::Value *RF = Builder.CreateExtractValue(R, I);
    llvm::Value *Cmp = Builder.CreateICmp(Eq, LF, RF, "memptr.cmp.rest");
    if (Res)
      Res = Builder.CreateBinOp(And, Res, Cmp);
    else
      Res = Cmp;
  }

  // Check if the first field is 0 if this is a function pointer.
  if (MPT->isMemberFunctionPointer()) {
    // (l1 == r1 && ...) || l0 == 0
    llvm::Value *Zero = llvm::Constant::getNullValue(L0->getType());
    llvm::Value *IsZero = Builder.CreateICmp(Eq, L0, Zero, "memptr.cmp.iszero");
    Res = Builder.CreateBinOp(Or, Res, IsZero);
  }

  // Combine the comparison of the first field, which must always be true for
  // this comparison to succeeed.
  return Builder.CreateBinOp(And, Res, Cmp0, "memptr.cmp");
}

llvm::Value *
MicrosoftCXXABI::EmitMemberPointerIsNotNull(CodeGenFunction &CGF,
                                            llvm::Value *MemPtr,
                                            const MemberPointerType *MPT) {
  CGBuilderTy &Builder = CGF.Builder;
  llvm::SmallVector<llvm::Constant *, 4> fields;
  // We only need one field for member functions.
  if (MPT->isMemberFunctionPointer())
    fields.push_back(llvm::Constant::getNullValue(CGM.VoidPtrTy));
  else
    GetNullMemberPointerFields(MPT, fields);
  assert(!fields.empty());
  llvm::Value *FirstField = MemPtr;
  if (MemPtr->getType()->isStructTy())
    FirstField = Builder.CreateExtractValue(MemPtr, 0);
  llvm::Value *Res = Builder.CreateICmpNE(FirstField, fields[0], "memptr.cmp0");

  // For function member pointers, we only need to test the function pointer
  // field.  The other fields if any can be garbage.
  if (MPT->isMemberFunctionPointer())
    return Res;

  // Otherwise, emit a series of compares and combine the results.
  for (int I = 1, E = fields.size(); I < E; ++I) {
    llvm::Value *Field = Builder.CreateExtractValue(MemPtr, I);
    llvm::Value *Next = Builder.CreateICmpNE(Field, fields[I], "memptr.cmp");
    Res = Builder.CreateAnd(Res, Next, "memptr.tobool");
  }
  return Res;
}

bool MicrosoftCXXABI::MemberPointerConstantIsNull(const MemberPointerType *MPT,
                                                  llvm::Constant *Val) {
  // Function pointers are null if the pointer in the first field is null.
  if (MPT->isMemberFunctionPointer()) {
    llvm::Constant *FirstField = Val->getType()->isStructTy() ?
      Val->getAggregateElement(0U) : Val;
    return FirstField->isNullValue();
  }

  // If it's not a function pointer and it's zero initializable, we can easily
  // check zero.
  if (isZeroInitializable(MPT) && Val->isNullValue())
    return true;

  // Otherwise, break down all the fields for comparison.  Hopefully these
  // little Constants are reused, while a big null struct might not be.
  llvm::SmallVector<llvm::Constant *, 4> Fields;
  GetNullMemberPointerFields(MPT, Fields);
  if (Fields.size() == 1) {
    assert(Val->getType()->isIntegerTy());
    return Val == Fields[0];
  }

  unsigned I, E;
  for (I = 0, E = Fields.size(); I != E; ++I) {
    if (Val->getAggregateElement(I) != Fields[I])
      break;
  }
  return I == E;
}

// Returns an adjusted base cast to i8*, since we do more address arithmetic on
// it.
llvm::Value *
MicrosoftCXXABI::AdjustVirtualBase(CodeGenFunction &CGF,
                                   const CXXRecordDecl *RD, llvm::Value *Base,
                                   llvm::Value *VirtualBaseAdjustmentOffset,
                                   llvm::Value *VBPtrOffset) {
  CGBuilderTy &Builder = CGF.Builder;
  Base = Builder.CreateBitCast(Base, CGM.Int8PtrTy);
  llvm::BasicBlock *OriginalBB = 0;
  llvm::BasicBlock *SkipAdjustBB = 0;
  llvm::BasicBlock *VBaseAdjustBB = 0;

  // In the unspecified inheritance model, there might not be a vbtable at all,
  // in which case we need to skip the virtual base lookup.  If there is a
  // vbtable, the first entry is a no-op entry that gives back the original
  // base, so look for a virtual base adjustment offset of zero.
  if (VBPtrOffset) {
    OriginalBB = Builder.GetInsertBlock();
    VBaseAdjustBB = CGF.createBasicBlock("memptr.vadjust");
    SkipAdjustBB = CGF.createBasicBlock("memptr.skip_vadjust");
    llvm::Value *IsVirtual =
      Builder.CreateICmpNE(VirtualBaseAdjustmentOffset, getZeroInt(),
                           "memptr.is_vbase");
    Builder.CreateCondBr(IsVirtual, VBaseAdjustBB, SkipAdjustBB);
    CGF.EmitBlock(VBaseAdjustBB);
  }

  // If we weren't given a dynamic vbptr offset, RD should be complete and we'll
  // know the vbptr offset.
  if (!VBPtrOffset) {
    CharUnits offs = getContext().getASTRecordLayout(RD).getVBPtrOffset();
    VBPtrOffset = llvm::ConstantInt::get(CGM.IntTy, offs.getQuantity());
  }
  // Load the vbtable pointer from the vbtable offset in the instance.
  llvm::Value *VBPtr =
    Builder.CreateInBoundsGEP(Base, VBPtrOffset, "memptr.vbptr");
  llvm::Value *VBTable =
    Builder.CreateBitCast(VBPtr, CGM.Int8PtrTy->getPointerTo(0));
  VBTable = Builder.CreateLoad(VBTable, "memptr.vbtable");
  // Load an i32 offset from the vb-table.
  llvm::Value *VBaseOffs =
    Builder.CreateInBoundsGEP(VBTable, VirtualBaseAdjustmentOffset);
  VBaseOffs = Builder.CreateBitCast(VBaseOffs, CGM.Int32Ty->getPointerTo(0));
  VBaseOffs = Builder.CreateLoad(VBaseOffs, "memptr.vbase_offs");
  // Add it to VBPtr.  GEP will sign extend the i32 value for us.
  llvm::Value *AdjustedBase = Builder.CreateInBoundsGEP(VBPtr, VBaseOffs);

  // Merge control flow with the case where we didn't have to adjust.
  if (VBaseAdjustBB) {
    Builder.CreateBr(SkipAdjustBB);
    CGF.EmitBlock(SkipAdjustBB);
    llvm::PHINode *Phi = Builder.CreatePHI(CGM.Int8PtrTy, 2, "memptr.base");
    Phi->addIncoming(Base, OriginalBB);
    Phi->addIncoming(AdjustedBase, VBaseAdjustBB);
    return Phi;
  }
  return AdjustedBase;
}

llvm::Value *
MicrosoftCXXABI::EmitMemberDataPointerAddress(CodeGenFunction &CGF,
                                              llvm::Value *Base,
                                              llvm::Value *MemPtr,
                                              const MemberPointerType *MPT) {
  assert(MPT->isMemberDataPointer());
  unsigned AS = Base->getType()->getPointerAddressSpace();
  llvm::Type *PType =
      CGF.ConvertTypeForMem(MPT->getPointeeType())->getPointerTo(AS);
  CGBuilderTy &Builder = CGF.Builder;
  const CXXRecordDecl *RD = MPT->getClass()->getAsCXXRecordDecl();
  MSInheritanceModel Inheritance = RD->getMSInheritanceModel();

  // Extract the fields we need, regardless of model.  We'll apply them if we
  // have them.
  llvm::Value *FieldOffset = MemPtr;
  llvm::Value *VirtualBaseAdjustmentOffset = 0;
  llvm::Value *VBPtrOffset = 0;
  if (MemPtr->getType()->isStructTy()) {
    // We need to extract values.
    unsigned I = 0;
    FieldOffset = Builder.CreateExtractValue(MemPtr, I++);
    if (hasVBPtrOffsetField(Inheritance))
      VBPtrOffset = Builder.CreateExtractValue(MemPtr, I++);
    if (hasVirtualBaseAdjustmentField(Inheritance))
      VirtualBaseAdjustmentOffset = Builder.CreateExtractValue(MemPtr, I++);
  }

  if (VirtualBaseAdjustmentOffset) {
    Base = AdjustVirtualBase(CGF, RD, Base, VirtualBaseAdjustmentOffset,
                             VBPtrOffset);
  }
  llvm::Value *Addr =
    Builder.CreateInBoundsGEP(Base, FieldOffset, "memptr.offset");

  // Cast the address to the appropriate pointer type, adopting the address
  // space of the base pointer.
  return Builder.CreateBitCast(Addr, PType);
}

static MSInheritanceModel
getInheritanceFromMemptr(const MemberPointerType *MPT) {
  return MPT->getClass()->getAsCXXRecordDecl()->getMSInheritanceModel();
}

llvm::Value *
MicrosoftCXXABI::EmitMemberPointerConversion(CodeGenFunction &CGF,
                                             const CastExpr *E,
                                             llvm::Value *Src) {
  assert(E->getCastKind() == CK_DerivedToBaseMemberPointer ||
         E->getCastKind() == CK_BaseToDerivedMemberPointer ||
         E->getCastKind() == CK_ReinterpretMemberPointer);

  // Use constant emission if we can.
  if (isa<llvm::Constant>(Src))
    return EmitMemberPointerConversion(E, cast<llvm::Constant>(Src));

  // We may be adding or dropping fields from the member pointer, so we need
  // both types and the inheritance models of both records.
  const MemberPointerType *SrcTy =
    E->getSubExpr()->getType()->castAs<MemberPointerType>();
  const MemberPointerType *DstTy = E->getType()->castAs<MemberPointerType>();
  MSInheritanceModel SrcInheritance = getInheritanceFromMemptr(SrcTy);
  MSInheritanceModel DstInheritance = getInheritanceFromMemptr(DstTy);
  bool IsFunc = SrcTy->isMemberFunctionPointer();

  // If the classes use the same null representation, reinterpret_cast is a nop.
  bool IsReinterpret = E->getCastKind() == CK_ReinterpretMemberPointer;
  if (IsReinterpret && (IsFunc ||
                        nullFieldOffsetIsZero(SrcInheritance) ==
                        nullFieldOffsetIsZero(DstInheritance)))
    return Src;

  CGBuilderTy &Builder = CGF.Builder;

  // Branch past the conversion if Src is null.
  llvm::Value *IsNotNull = EmitMemberPointerIsNotNull(CGF, Src, SrcTy);
  llvm::Constant *DstNull = EmitNullMemberPointer(DstTy);

  // C++ 5.2.10p9: The null member pointer value is converted to the null member
  //   pointer value of the destination type.
  if (IsReinterpret) {
    // For reinterpret casts, sema ensures that src and dst are both functions
    // or data and have the same size, which means the LLVM types should match.
    assert(Src->getType() == DstNull->getType());
    return Builder.CreateSelect(IsNotNull, Src, DstNull);
  }

  llvm::BasicBlock *OriginalBB = Builder.GetInsertBlock();
  llvm::BasicBlock *ConvertBB = CGF.createBasicBlock("memptr.convert");
  llvm::BasicBlock *ContinueBB = CGF.createBasicBlock("memptr.converted");
  Builder.CreateCondBr(IsNotNull, ConvertBB, ContinueBB);
  CGF.EmitBlock(ConvertBB);

  // Decompose src.
  llvm::Value *FirstField = Src;
  llvm::Value *NonVirtualBaseAdjustment = 0;
  llvm::Value *VirtualBaseAdjustmentOffset = 0;
  llvm::Value *VBPtrOffset = 0;
  if (!hasOnlyOneField(IsFunc, SrcInheritance)) {
    // We need to extract values.
    unsigned I = 0;
    FirstField = Builder.CreateExtractValue(Src, I++);
    if (hasNonVirtualBaseAdjustmentField(IsFunc, SrcInheritance))
      NonVirtualBaseAdjustment = Builder.CreateExtractValue(Src, I++);
    if (hasVBPtrOffsetField(SrcInheritance))
      VBPtrOffset = Builder.CreateExtractValue(Src, I++);
    if (hasVirtualBaseAdjustmentField(SrcInheritance))
      VirtualBaseAdjustmentOffset = Builder.CreateExtractValue(Src, I++);
  }

  // For data pointers, we adjust the field offset directly.  For functions, we
  // have a separate field.
  llvm::Constant *Adj = getMemberPointerAdjustment(E);
  if (Adj) {
    Adj = llvm::ConstantExpr::getTruncOrBitCast(Adj, CGM.IntTy);
    llvm::Value *&NVAdjustField = IsFunc ? NonVirtualBaseAdjustment : FirstField;
    bool isDerivedToBase = (E->getCastKind() == CK_DerivedToBaseMemberPointer);
    if (!NVAdjustField)  // If this field didn't exist in src, it's zero.
      NVAdjustField = getZeroInt();
    if (isDerivedToBase)
      NVAdjustField = Builder.CreateNSWSub(NVAdjustField, Adj, "adj");
    else
      NVAdjustField = Builder.CreateNSWAdd(NVAdjustField, Adj, "adj");
  }

  // FIXME PR15713: Support conversions through virtually derived classes.

  // Recompose dst from the null struct and the adjusted fields from src.
  llvm::Value *Dst;
  if (hasOnlyOneField(IsFunc, DstInheritance)) {
    Dst = FirstField;
  } else {
    Dst = llvm::UndefValue::get(DstNull->getType());
    unsigned Idx = 0;
    Dst = Builder.CreateInsertValue(Dst, FirstField, Idx++);
    if (hasNonVirtualBaseAdjustmentField(IsFunc, DstInheritance))
      Dst = Builder.CreateInsertValue(
        Dst, getValueOrZeroInt(NonVirtualBaseAdjustment), Idx++);
    if (hasVBPtrOffsetField(DstInheritance))
      Dst = Builder.CreateInsertValue(
        Dst, getValueOrZeroInt(VBPtrOffset), Idx++);
    if (hasVirtualBaseAdjustmentField(DstInheritance))
      Dst = Builder.CreateInsertValue(
        Dst, getValueOrZeroInt(VirtualBaseAdjustmentOffset), Idx++);
  }
  Builder.CreateBr(ContinueBB);

  // In the continuation, choose between DstNull and Dst.
  CGF.EmitBlock(ContinueBB);
  llvm::PHINode *Phi = Builder.CreatePHI(DstNull->getType(), 2, "memptr.converted");
  Phi->addIncoming(DstNull, OriginalBB);
  Phi->addIncoming(Dst, ConvertBB);
  return Phi;
}

llvm::Constant *
MicrosoftCXXABI::EmitMemberPointerConversion(const CastExpr *E,
                                             llvm::Constant *Src) {
  const MemberPointerType *SrcTy =
    E->getSubExpr()->getType()->castAs<MemberPointerType>();
  const MemberPointerType *DstTy = E->getType()->castAs<MemberPointerType>();

  // If src is null, emit a new null for dst.  We can't return src because dst
  // might have a new representation.
  if (MemberPointerConstantIsNull(SrcTy, Src))
    return EmitNullMemberPointer(DstTy);

  // We don't need to do anything for reinterpret_casts of non-null member
  // pointers.  We should only get here when the two type representations have
  // the same size.
  if (E->getCastKind() == CK_ReinterpretMemberPointer)
    return Src;

  MSInheritanceModel SrcInheritance = getInheritanceFromMemptr(SrcTy);
  MSInheritanceModel DstInheritance = getInheritanceFromMemptr(DstTy);

  // Decompose src.
  llvm::Constant *FirstField = Src;
  llvm::Constant *NonVirtualBaseAdjustment = 0;
  llvm::Constant *VirtualBaseAdjustmentOffset = 0;
  llvm::Constant *VBPtrOffset = 0;
  bool IsFunc = SrcTy->isMemberFunctionPointer();
  if (!hasOnlyOneField(IsFunc, SrcInheritance)) {
    // We need to extract values.
    unsigned I = 0;
    FirstField = Src->getAggregateElement(I++);
    if (hasNonVirtualBaseAdjustmentField(IsFunc, SrcInheritance))
      NonVirtualBaseAdjustment = Src->getAggregateElement(I++);
    if (hasVBPtrOffsetField(SrcInheritance))
      VBPtrOffset = Src->getAggregateElement(I++);
    if (hasVirtualBaseAdjustmentField(SrcInheritance))
      VirtualBaseAdjustmentOffset = Src->getAggregateElement(I++);
  }

  // For data pointers, we adjust the field offset directly.  For functions, we
  // have a separate field.
  llvm::Constant *Adj = getMemberPointerAdjustment(E);
  if (Adj) {
    Adj = llvm::ConstantExpr::getTruncOrBitCast(Adj, CGM.IntTy);
    llvm::Constant *&NVAdjustField =
      IsFunc ? NonVirtualBaseAdjustment : FirstField;
    bool IsDerivedToBase = (E->getCastKind() == CK_DerivedToBaseMemberPointer);
    if (!NVAdjustField)  // If this field didn't exist in src, it's zero.
      NVAdjustField = getZeroInt();
    if (IsDerivedToBase)
      NVAdjustField = llvm::ConstantExpr::getNSWSub(NVAdjustField, Adj);
    else
      NVAdjustField = llvm::ConstantExpr::getNSWAdd(NVAdjustField, Adj);
  }

  // FIXME PR15713: Support conversions through virtually derived classes.

  // Recompose dst from the null struct and the adjusted fields from src.
  if (hasOnlyOneField(IsFunc, DstInheritance))
    return FirstField;

  llvm::SmallVector<llvm::Constant *, 4> Fields;
  Fields.push_back(FirstField);
  if (hasNonVirtualBaseAdjustmentField(IsFunc, DstInheritance))
    Fields.push_back(getConstantOrZeroInt(NonVirtualBaseAdjustment));
  if (hasVBPtrOffsetField(DstInheritance))
    Fields.push_back(getConstantOrZeroInt(VBPtrOffset));
  if (hasVirtualBaseAdjustmentField(DstInheritance))
    Fields.push_back(getConstantOrZeroInt(VirtualBaseAdjustmentOffset));
  return llvm::ConstantStruct::getAnon(Fields);
}

llvm::Value *
MicrosoftCXXABI::EmitLoadOfMemberFunctionPointer(CodeGenFunction &CGF,
                                                 llvm::Value *&This,
                                                 llvm::Value *MemPtr,
                                                 const MemberPointerType *MPT) {
  assert(MPT->isMemberFunctionPointer());
  const FunctionProtoType *FPT =
    MPT->getPointeeType()->castAs<FunctionProtoType>();
  const CXXRecordDecl *RD = MPT->getClass()->getAsCXXRecordDecl();
  llvm::FunctionType *FTy =
    CGM.getTypes().GetFunctionType(
      CGM.getTypes().arrangeCXXMethodType(RD, FPT));
  CGBuilderTy &Builder = CGF.Builder;

  MSInheritanceModel Inheritance = RD->getMSInheritanceModel();

  // Extract the fields we need, regardless of model.  We'll apply them if we
  // have them.
  llvm::Value *FunctionPointer = MemPtr;
  llvm::Value *NonVirtualBaseAdjustment = NULL;
  llvm::Value *VirtualBaseAdjustmentOffset = NULL;
  llvm::Value *VBPtrOffset = NULL;
  if (MemPtr->getType()->isStructTy()) {
    // We need to extract values.
    unsigned I = 0;
    FunctionPointer = Builder.CreateExtractValue(MemPtr, I++);
    if (hasNonVirtualBaseAdjustmentField(MPT, Inheritance))
      NonVirtualBaseAdjustment = Builder.CreateExtractValue(MemPtr, I++);
    if (hasVBPtrOffsetField(Inheritance))
      VBPtrOffset = Builder.CreateExtractValue(MemPtr, I++);
    if (hasVirtualBaseAdjustmentField(Inheritance))
      VirtualBaseAdjustmentOffset = Builder.CreateExtractValue(MemPtr, I++);
  }

  if (VirtualBaseAdjustmentOffset) {
    This = AdjustVirtualBase(CGF, RD, This, VirtualBaseAdjustmentOffset,
                             VBPtrOffset);
  }

  if (NonVirtualBaseAdjustment) {
    // Apply the adjustment and cast back to the original struct type.
    llvm::Value *Ptr = Builder.CreateBitCast(This, Builder.getInt8PtrTy());
    Ptr = Builder.CreateInBoundsGEP(Ptr, NonVirtualBaseAdjustment);
    This = Builder.CreateBitCast(Ptr, This->getType(), "this.adjusted");
  }

  return Builder.CreateBitCast(FunctionPointer, FTy->getPointerTo());
}

CGCXXABI *clang::CodeGen::CreateMicrosoftCXXABI(CodeGenModule &CGM) {
  return new MicrosoftCXXABI(CGM);
}

