//===----- CGCXXABI.cpp - Interface to C++ ABIs -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This provides an abstract class for C++ code generation. Concrete subclasses
// of this implement code generation for specific C++ ABIs.
//
//===----------------------------------------------------------------------===//

#include "CGCXXABI.h"

using namespace clang;
using namespace CodeGen;

void CGCXXABI::_anchor() {}

static void ErrorUnsupportedABI(CodeGenFunction &CGF,
                                llvm::StringRef S) {
  Diagnostic &Diags = CGF.CGM.getDiags();
  unsigned DiagID = Diags.getCustomDiagID(Diagnostic::Error,
                                          "cannot yet compile %1 in this ABI");
  Diags.Report(CGF.getContext().getFullLoc(CGF.CurCodeDecl->getLocation()),
               DiagID)
    << S;
}

static llvm::Constant *GetBogusMemberPointer(CodeGenModule &CGM,
                                             QualType T) {
  return llvm::Constant::getNullValue(CGM.getTypes().ConvertType(T));
}

const llvm::Type *
CGCXXABI::ConvertMemberPointerType(const MemberPointerType *MPT) {
  return CGM.getTypes().ConvertType(CGM.getContext().getPointerDiffType());
}

llvm::Value *CGCXXABI::EmitLoadOfMemberFunctionPointer(CodeGenFunction &CGF,
                                                       llvm::Value *&This,
                                                       llvm::Value *MemPtr,
                                                 const MemberPointerType *MPT) {
  ErrorUnsupportedABI(CGF, "calls through member pointers");

  const FunctionProtoType *FPT = 
    MPT->getPointeeType()->getAs<FunctionProtoType>();
  const CXXRecordDecl *RD = 
    cast<CXXRecordDecl>(MPT->getClass()->getAs<RecordType>()->getDecl());
  const llvm::FunctionType *FTy = 
    CGM.getTypes().GetFunctionType(CGM.getTypes().getFunctionInfo(RD, FPT),
                                   FPT->isVariadic());
  return llvm::Constant::getNullValue(FTy->getPointerTo());
}

llvm::Value *CGCXXABI::EmitMemberDataPointerAddress(CodeGenFunction &CGF,
                                                    llvm::Value *Base,
                                                    llvm::Value *MemPtr,
                                              const MemberPointerType *MPT) {
  ErrorUnsupportedABI(CGF, "loads of member pointers");
  const llvm::Type *Ty = CGF.ConvertType(MPT->getPointeeType())->getPointerTo();
  return llvm::Constant::getNullValue(Ty);
}

llvm::Value *CGCXXABI::EmitMemberPointerConversion(CodeGenFunction &CGF,
                                                   const CastExpr *E,
                                                   llvm::Value *Src) {
  ErrorUnsupportedABI(CGF, "member function pointer conversions");
  return GetBogusMemberPointer(CGM, E->getType());
}

llvm::Value *
CGCXXABI::EmitMemberPointerComparison(CodeGenFunction &CGF,
                                      llvm::Value *L,
                                      llvm::Value *R,
                                      const MemberPointerType *MPT,
                                      bool Inequality) {
  ErrorUnsupportedABI(CGF, "member function pointer comparison");
  return CGF.Builder.getFalse();
}

llvm::Value *
CGCXXABI::EmitMemberPointerIsNotNull(CodeGenFunction &CGF,
                                     llvm::Value *MemPtr,
                                     const MemberPointerType *MPT) {
  ErrorUnsupportedABI(CGF, "member function pointer null testing");
  return CGF.Builder.getFalse();
}

llvm::Constant *
CGCXXABI::EmitMemberPointerConversion(llvm::Constant *C, const CastExpr *E) {
  return GetBogusMemberPointer(CGM, E->getType());
}

llvm::Constant *
CGCXXABI::EmitNullMemberPointer(const MemberPointerType *MPT) {
  return GetBogusMemberPointer(CGM, QualType(MPT, 0));
}

llvm::Constant *CGCXXABI::EmitMemberPointer(const CXXMethodDecl *MD) {
  return GetBogusMemberPointer(CGM,
                         CGM.getContext().getMemberPointerType(MD->getType(),
                                         MD->getParent()->getTypeForDecl()));
}

llvm::Constant *CGCXXABI::EmitMemberPointer(const FieldDecl *FD) {
  return GetBogusMemberPointer(CGM,
                         CGM.getContext().getMemberPointerType(FD->getType(),
                                         FD->getParent()->getTypeForDecl()));
}

bool CGCXXABI::isZeroInitializable(const MemberPointerType *MPT) {
  // Fake answer.
  return true;
}

void CGCXXABI::BuildThisParam(CodeGenFunction &CGF, FunctionArgList &Params) {
  const CXXMethodDecl *MD = cast<CXXMethodDecl>(CGF.CurGD.getDecl());

  // FIXME: I'm not entirely sure I like using a fake decl just for code
  // generation. Maybe we can come up with a better way?
  ImplicitParamDecl *ThisDecl
    = ImplicitParamDecl::Create(CGM.getContext(), 0, MD->getLocation(),
                                &CGM.getContext().Idents.get("this"),
                                MD->getThisType(CGM.getContext()));
  Params.push_back(std::make_pair(ThisDecl, ThisDecl->getType()));
  getThisDecl(CGF) = ThisDecl;
}

void CGCXXABI::EmitThisParam(CodeGenFunction &CGF) {
  /// Initialize the 'this' slot.
  assert(getThisDecl(CGF) && "no 'this' variable for function");
  getThisValue(CGF)
    = CGF.Builder.CreateLoad(CGF.GetAddrOfLocalVar(getThisDecl(CGF)),
                             "this");
}

void CGCXXABI::EmitReturnFromThunk(CodeGenFunction &CGF,
                                   RValue RV, QualType ResultType) {
  CGF.EmitReturnOfRValue(RV, ResultType);
}

CharUnits CGCXXABI::GetArrayCookieSize(QualType ElementType) {
  return CharUnits::Zero();
}

llvm::Value *CGCXXABI::InitializeArrayCookie(CodeGenFunction &CGF,
                                             llvm::Value *NewPtr,
                                             llvm::Value *NumElements,
                                             QualType ElementType) {
  // Should never be called.
  ErrorUnsupportedABI(CGF, "array cookie initialization");
  return 0;
}

void CGCXXABI::ReadArrayCookie(CodeGenFunction &CGF, llvm::Value *Ptr,
                               QualType ElementType, llvm::Value *&NumElements,
                               llvm::Value *&AllocPtr, CharUnits &CookieSize) {
  ErrorUnsupportedABI(CGF, "array cookie reading");

  // This should be enough to avoid assertions.
  NumElements = 0;
  AllocPtr = llvm::Constant::getNullValue(CGF.Builder.getInt8PtrTy());
  CookieSize = CharUnits::Zero();
}

void CGCXXABI::EmitGuardedInit(CodeGenFunction &CGF,
                               const VarDecl &D,
                               llvm::GlobalVariable *GV) {
  ErrorUnsupportedABI(CGF, "static local variable initialization");
}
