//===----- CGCXXABI.cpp - Interface to C++ ABIs ---------------------------===//
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

CGCXXABI::~CGCXXABI() { }

void CGCXXABI::ErrorUnsupportedABI(CodeGenFunction &CGF, StringRef S) {
  DiagnosticsEngine &Diags = CGF.CGM.getDiags();
  unsigned DiagID = Diags.getCustomDiagID(DiagnosticsEngine::Error,
                                          "cannot yet compile %0 in this ABI");
  Diags.Report(CGF.getContext().getFullLoc(CGF.CurCodeDecl->getLocation()),
               DiagID)
    << S;
}

llvm::Constant *CGCXXABI::GetBogusMemberPointer(QualType T) {
  return llvm::Constant::getNullValue(CGM.getTypes().ConvertType(T));
}

llvm::Type *
CGCXXABI::ConvertMemberPointerType(const MemberPointerType *MPT) {
  return CGM.getTypes().ConvertType(CGM.getContext().getPointerDiffType());
}

llvm::Value *CGCXXABI::EmitLoadOfMemberFunctionPointer(
    CodeGenFunction &CGF, const Expr *E, llvm::Value *&This,
    llvm::Value *MemPtr, const MemberPointerType *MPT) {
  ErrorUnsupportedABI(CGF, "calls through member pointers");

  const FunctionProtoType *FPT = 
    MPT->getPointeeType()->getAs<FunctionProtoType>();
  const CXXRecordDecl *RD = 
    cast<CXXRecordDecl>(MPT->getClass()->getAs<RecordType>()->getDecl());
  llvm::FunctionType *FTy = CGM.getTypes().GetFunctionType(
                              CGM.getTypes().arrangeCXXMethodType(RD, FPT));
  return llvm::Constant::getNullValue(FTy->getPointerTo());
}

llvm::Value *
CGCXXABI::EmitMemberDataPointerAddress(CodeGenFunction &CGF, const Expr *E,
                                       llvm::Value *Base, llvm::Value *MemPtr,
                                       const MemberPointerType *MPT) {
  ErrorUnsupportedABI(CGF, "loads of member pointers");
  llvm::Type *Ty = CGF.ConvertType(MPT->getPointeeType())->getPointerTo();
  return llvm::Constant::getNullValue(Ty);
}

llvm::Value *CGCXXABI::EmitMemberPointerConversion(CodeGenFunction &CGF,
                                                   const CastExpr *E,
                                                   llvm::Value *Src) {
  ErrorUnsupportedABI(CGF, "member function pointer conversions");
  return GetBogusMemberPointer(E->getType());
}

llvm::Constant *CGCXXABI::EmitMemberPointerConversion(const CastExpr *E,
                                                      llvm::Constant *Src) {
  return GetBogusMemberPointer(E->getType());
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
CGCXXABI::EmitNullMemberPointer(const MemberPointerType *MPT) {
  return GetBogusMemberPointer(QualType(MPT, 0));
}

llvm::Constant *CGCXXABI::EmitMemberPointer(const CXXMethodDecl *MD) {
  return GetBogusMemberPointer(
                         CGM.getContext().getMemberPointerType(MD->getType(),
                                         MD->getParent()->getTypeForDecl()));
}

llvm::Constant *CGCXXABI::EmitMemberDataPointer(const MemberPointerType *MPT,
                                                CharUnits offset) {
  return GetBogusMemberPointer(QualType(MPT, 0));
}

llvm::Constant *CGCXXABI::EmitMemberPointer(const APValue &MP, QualType MPT) {
  return GetBogusMemberPointer(MPT);
}

bool CGCXXABI::isZeroInitializable(const MemberPointerType *MPT) {
  // Fake answer.
  return true;
}

void CGCXXABI::buildThisParam(CodeGenFunction &CGF, FunctionArgList &params) {
  const CXXMethodDecl *MD = cast<CXXMethodDecl>(CGF.CurGD.getDecl());

  // FIXME: I'm not entirely sure I like using a fake decl just for code
  // generation. Maybe we can come up with a better way?
  ImplicitParamDecl *ThisDecl
    = ImplicitParamDecl::Create(CGM.getContext(), 0, MD->getLocation(),
                                &CGM.getContext().Idents.get("this"),
                                MD->getThisType(CGM.getContext()));
  params.push_back(ThisDecl);
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

CharUnits CGCXXABI::GetArrayCookieSize(const CXXNewExpr *expr) {
  if (!requiresArrayCookie(expr))
    return CharUnits::Zero();
  return getArrayCookieSizeImpl(expr->getAllocatedType());
}

CharUnits CGCXXABI::getArrayCookieSizeImpl(QualType elementType) {
  // BOGUS
  return CharUnits::Zero();
}

llvm::Value *CGCXXABI::InitializeArrayCookie(CodeGenFunction &CGF,
                                             llvm::Value *NewPtr,
                                             llvm::Value *NumElements,
                                             const CXXNewExpr *expr,
                                             QualType ElementType) {
  // Should never be called.
  ErrorUnsupportedABI(CGF, "array cookie initialization");
  return 0;
}

bool CGCXXABI::requiresArrayCookie(const CXXDeleteExpr *expr,
                                   QualType elementType) {
  // If the class's usual deallocation function takes two arguments,
  // it needs a cookie.
  if (expr->doesUsualArrayDeleteWantSize())
    return true;

  return elementType.isDestructedType();
}

bool CGCXXABI::requiresArrayCookie(const CXXNewExpr *expr) {
  // If the class's usual deallocation function takes two arguments,
  // it needs a cookie.
  if (expr->doesUsualArrayDeleteWantSize())
    return true;

  return expr->getAllocatedType().isDestructedType();
}

void CGCXXABI::ReadArrayCookie(CodeGenFunction &CGF, llvm::Value *ptr,
                               const CXXDeleteExpr *expr, QualType eltTy,
                               llvm::Value *&numElements,
                               llvm::Value *&allocPtr, CharUnits &cookieSize) {
  // Derive a char* in the same address space as the pointer.
  unsigned AS = ptr->getType()->getPointerAddressSpace();
  llvm::Type *charPtrTy = CGF.Int8Ty->getPointerTo(AS);
  ptr = CGF.Builder.CreateBitCast(ptr, charPtrTy);

  // If we don't need an array cookie, bail out early.
  if (!requiresArrayCookie(expr, eltTy)) {
    allocPtr = ptr;
    numElements = 0;
    cookieSize = CharUnits::Zero();
    return;
  }

  cookieSize = getArrayCookieSizeImpl(eltTy);
  allocPtr = CGF.Builder.CreateConstInBoundsGEP1_64(ptr,
                                                    -cookieSize.getQuantity());
  numElements = readArrayCookieImpl(CGF, allocPtr, cookieSize);
}

llvm::Value *CGCXXABI::readArrayCookieImpl(CodeGenFunction &CGF,
                                           llvm::Value *ptr,
                                           CharUnits cookieSize) {
  ErrorUnsupportedABI(CGF, "reading a new[] cookie");
  return llvm::ConstantInt::get(CGF.SizeTy, 0);
}

void CGCXXABI::registerGlobalDtor(CodeGenFunction &CGF,
                                  const VarDecl &D,
                                  llvm::Constant *dtor,
                                  llvm::Constant *addr) {
  if (D.getTLSKind())
    CGM.ErrorUnsupported(&D, "non-trivial TLS destruction");

  // The default behavior is to use atexit.
  CGF.registerGlobalDtorWithAtExit(D, dtor, addr);
}

/// Returns the adjustment, in bytes, required for the given
/// member-pointer operation.  Returns null if no adjustment is
/// required.
llvm::Constant *CGCXXABI::getMemberPointerAdjustment(const CastExpr *E) {
  assert(E->getCastKind() == CK_DerivedToBaseMemberPointer ||
         E->getCastKind() == CK_BaseToDerivedMemberPointer);

  QualType derivedType;
  if (E->getCastKind() == CK_DerivedToBaseMemberPointer)
    derivedType = E->getSubExpr()->getType();
  else
    derivedType = E->getType();

  const CXXRecordDecl *derivedClass =
    derivedType->castAs<MemberPointerType>()->getClass()->getAsCXXRecordDecl();

  return CGM.GetNonVirtualBaseClassOffset(derivedClass,
                                          E->path_begin(),
                                          E->path_end());
}

CharUnits CGCXXABI::getMemberPointerPathAdjustment(const APValue &MP) {
  // TODO: Store base specifiers in APValue member pointer paths so we can
  // easily reuse CGM.GetNonVirtualBaseClassOffset().
  const ValueDecl *MPD = MP.getMemberPointerDecl();
  CharUnits ThisAdjustment = CharUnits::Zero();
  ArrayRef<const CXXRecordDecl*> Path = MP.getMemberPointerPath();
  bool DerivedMember = MP.isMemberPointerToDerivedMember();
  const CXXRecordDecl *RD = cast<CXXRecordDecl>(MPD->getDeclContext());
  for (unsigned I = 0, N = Path.size(); I != N; ++I) {
    const CXXRecordDecl *Base = RD;
    const CXXRecordDecl *Derived = Path[I];
    if (DerivedMember)
      std::swap(Base, Derived);
    ThisAdjustment +=
      getContext().getASTRecordLayout(Derived).getBaseClassOffset(Base);
    RD = Path[I];
  }
  if (DerivedMember)
    ThisAdjustment = -ThisAdjustment;
  return ThisAdjustment;
}

llvm::BasicBlock *
CGCXXABI::EmitCtorCompleteObjectHandler(CodeGenFunction &CGF,
                                        const CXXRecordDecl *RD) {
  if (CGM.getTarget().getCXXABI().hasConstructorVariants())
    llvm_unreachable("shouldn't be called in this ABI");

  ErrorUnsupportedABI(CGF, "complete object detection in ctor");
  return 0;
}

void CGCXXABI::EmitThreadLocalInitFuncs(
    llvm::ArrayRef<std::pair<const VarDecl *, llvm::GlobalVariable *> > Decls,
    llvm::Function *InitFunc) {
}

LValue CGCXXABI::EmitThreadLocalVarDeclLValue(CodeGenFunction &CGF,
                                              const VarDecl *VD,
                                              QualType LValType) {
  ErrorUnsupportedABI(CGF, "odr-use of thread_local global");
  return LValue();
}

bool CGCXXABI::NeedsVTTParameter(GlobalDecl GD) {
  return false;
}

/// What sort of uniqueness rules should we use for the RTTI for the
/// given type?
CGCXXABI::RTTIUniquenessKind
CGCXXABI::classifyRTTIUniqueness(QualType CanTy,
                                 llvm::GlobalValue::LinkageTypes Linkage) {
  if (shouldRTTIBeUnique())
    return RUK_Unique;

  // It's only necessary for linkonce_odr or weak_odr linkage.
  if (Linkage != llvm::GlobalValue::LinkOnceODRLinkage &&
      Linkage != llvm::GlobalValue::WeakODRLinkage)
    return RUK_Unique;

  // It's only necessary with default visibility.
  if (CanTy->getVisibility() != DefaultVisibility)
    return RUK_Unique;

  // If we're not required to publish this symbol, hide it.
  if (Linkage == llvm::GlobalValue::LinkOnceODRLinkage)
    return RUK_NonUniqueHidden;

  // If we're required to publish this symbol, as we might be under an
  // explicit instantiation, leave it with default visibility but
  // enable string-comparisons.
  assert(Linkage == llvm::GlobalValue::WeakODRLinkage);
  return RUK_NonUniqueVisible;
}
