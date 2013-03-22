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
  llvm::Constant *getSimpleNullMemberPointer(const MemberPointerType *MPT);

  llvm::Constant *getZeroPtrDiff() {
    return llvm::ConstantInt::get(CGM.PtrDiffTy, 0);
  }

  llvm::Constant *getAllOnesPtrDiff() {
    return  llvm::Constant::getAllOnesValue(CGM.PtrDiffTy);
  }

public:
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

  // Emit the initializer and add a global destructor if appropriate.
  CGF.EmitCXXGlobalVarDeclInit(D, DeclPtr, PerformInit);
}

// Returns true for member pointer types that we know how to represent with a
// simple ptrdiff_t.  Currently we only know how to emit, test, and load member
// data pointers for complete single inheritance classes.
static bool isSimpleMemberPointer(const MemberPointerType *MPT) {
  const CXXRecordDecl *RD = MPT->getClass()->getAsCXXRecordDecl();
  return (MPT->isMemberDataPointer() &&
          !MPT->getClass()->isIncompleteType() &&
          RD->getNumVBases() == 0);
}

llvm::Constant *
MicrosoftCXXABI::getSimpleNullMemberPointer(const MemberPointerType *MPT) {
  if (isSimpleMemberPointer(MPT)) {
    const CXXRecordDecl *RD = MPT->getClass()->getAsCXXRecordDecl();
    // A null member data pointer is represented as -1 if the class is not
    // polymorphic, and 0 otherwise.
    if (RD->isPolymorphic())
      return getZeroPtrDiff();
    return getAllOnesPtrDiff();
  }
  return GetBogusMemberPointer(QualType(MPT, 0));
}

llvm::Constant *
MicrosoftCXXABI::EmitNullMemberPointer(const MemberPointerType *MPT) {
  if (isSimpleMemberPointer(MPT))
    return getSimpleNullMemberPointer(MPT);
  // FIXME: Implement function member pointers.
  return GetBogusMemberPointer(QualType(MPT, 0));
}

llvm::Constant *
MicrosoftCXXABI::EmitMemberDataPointer(const MemberPointerType *MPT,
                                       CharUnits offset) {
  // Member data pointers are plain offsets when no virtual bases are involved.
  if (isSimpleMemberPointer(MPT))
    return llvm::ConstantInt::get(CGM.PtrDiffTy, offset.getQuantity());
  // FIXME: Implement member pointers other inheritance models.
  return GetBogusMemberPointer(QualType(MPT, 0));
}

llvm::Value *
MicrosoftCXXABI::EmitMemberPointerIsNotNull(CodeGenFunction &CGF,
                                            llvm::Value *MemPtr,
                                            const MemberPointerType *MPT) {
  CGBuilderTy &Builder = CGF.Builder;

  // For member data pointers, this is just a check against -1 or 0.
  if (isSimpleMemberPointer(MPT)) {
    llvm::Constant *Val = getSimpleNullMemberPointer(MPT);
    return Builder.CreateICmpNE(MemPtr, Val, "memptr.tobool");
  }

  // FIXME: Implement member pointers other inheritance models.
  ErrorUnsupportedABI(CGF, "function member pointer tests");
  return GetBogusMemberPointer(QualType(MPT, 0));
}

llvm::Value *
MicrosoftCXXABI::EmitMemberDataPointerAddress(CodeGenFunction &CGF,
                                              llvm::Value *Base,
                                              llvm::Value *MemPtr,
                                              const MemberPointerType *MPT) {
  unsigned AS = Base->getType()->getPointerAddressSpace();
  llvm::Type *PType =
      CGF.ConvertTypeForMem(MPT->getPointeeType())->getPointerTo(AS);
  CGBuilderTy &Builder = CGF.Builder;

  if (MPT->isMemberFunctionPointer()) {
    ErrorUnsupportedABI(CGF, "function member pointer address");
    return llvm::Constant::getNullValue(PType);
  }

  llvm::Value *Addr;
  if (isSimpleMemberPointer(MPT)) {
    // Add the offset with GEP and i8*.
    assert(MemPtr->getType() == CGM.PtrDiffTy);
    Base = Builder.CreateBitCast(Base, Builder.getInt8Ty()->getPointerTo(AS));
    Addr = Builder.CreateInBoundsGEP(Base, MemPtr, "memptr.offset");
  } else {
    ErrorUnsupportedABI(CGF, "non-scalar member pointers");
    return llvm::Constant::getNullValue(PType);
  }

  // Cast the address to the appropriate pointer type, adopting the address
  // space of the base pointer.
  return Builder.CreateBitCast(Addr, PType);
}

CGCXXABI *clang::CodeGen::CreateMicrosoftCXXABI(CodeGenModule &CGM) {
  return new MicrosoftCXXABI(CGM);
}

