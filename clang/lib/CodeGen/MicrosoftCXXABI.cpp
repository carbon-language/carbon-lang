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

  void
  GetNullMemberPointerFields(const MemberPointerType *MPT,
                             llvm::SmallVectorImpl<llvm::Constant *> &fields);

  llvm::Value *AdjustVirtualBase(CodeGenFunction &CGF, const CXXRecordDecl *RD,
                                 llvm::Value *Base,
                                 llvm::Value *VirtualBaseAdjustmentOffset,
                                 llvm::Value *VBPtrOffset /* optional */);

public:
  virtual llvm::Type *ConvertMemberPointerType(const MemberPointerType *MPT);

  virtual bool isZeroInitializable(const MemberPointerType *MPT);

  virtual llvm::Constant *EmitNullMemberPointer(const MemberPointerType *MPT);

  virtual llvm::Constant *EmitMemberDataPointer(const MemberPointerType *MPT,
                                                CharUnits offset);

  virtual llvm::Value *EmitMemberPointerIsNotNull(CodeGenFunction &CGF,
                                                  llvm::Value *MemPtr,
                                                  const MemberPointerType *MPT);

  virtual llvm::Value *EmitMemberDataPointerAddress(CodeGenFunction &CGF,
                                                    llvm::Value *Base,
                                                    llvm::Value *MemPtr,
                                                  const MemberPointerType *MPT);

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

// Only member pointers to functions need a this adjustment, since it can be
// combined with the field offset for data pointers.
static bool hasNonVirtualBaseAdjustmentField(const MemberPointerType *MPT,
                                             MSInheritanceModel Inheritance) {
  return (MPT->isMemberFunctionPointer() &&
          Inheritance >= MSIM_Multiple);
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

  if (hasVBPtrOffsetField(Inheritance))
    fields.push_back(CGM.IntTy);
  if (hasNonVirtualBaseAdjustmentField(MPT, Inheritance))
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

  if (hasVBPtrOffsetField(Inheritance))
    fields.push_back(getZeroInt());
  if (hasNonVirtualBaseAdjustmentField(MPT, Inheritance))
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
MicrosoftCXXABI::EmitMemberDataPointer(const MemberPointerType *MPT,
                                       CharUnits offset) {
  const CXXRecordDecl *RD = MPT->getClass()->getAsCXXRecordDecl();
  MSInheritanceModel Inheritance = RD->getMSInheritanceModel();
  llvm::SmallVector<llvm::Constant *, 4> fields;
  fields.push_back(llvm::ConstantInt::get(CGM.IntTy, offset.getQuantity()));
  if (hasVBPtrOffsetField(Inheritance)) {
    int64_t VBPtrOffset =
      getContext().getASTRecordLayout(RD).getVBPtrOffset().getQuantity();
    fields.push_back(llvm::ConstantInt::get(CGM.IntTy, VBPtrOffset));
  }
  assert(!hasNonVirtualBaseAdjustmentField(MPT, Inheritance));
  // The virtual base field starts out zero.  It is adjusted by conversions to
  // member pointer types of a more derived class.  See http://llvm.org/PR15713
  if (hasVirtualBaseAdjustmentField(Inheritance))
    fields.push_back(getZeroInt());
  if (fields.size() == 1)
    return fields[0];
  return llvm::ConstantStruct::getAnon(fields);
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
    if (hasVBPtrOffsetField(Inheritance))
      VBPtrOffset = Builder.CreateExtractValue(MemPtr, I++);
    if (hasNonVirtualBaseAdjustmentField(MPT, Inheritance))
      NonVirtualBaseAdjustment = Builder.CreateExtractValue(MemPtr, I++);
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

