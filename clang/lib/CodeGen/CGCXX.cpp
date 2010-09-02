//===--- CGDecl.cpp - Emit LLVM Code for declarations ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with C++ code generation.
//
//===----------------------------------------------------------------------===//

// We might split this into multiple files if it gets too unwieldy

#include "CGCXXABI.h"
#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "Mangle.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/StmtCXX.h"
#include "clang/Frontend/CodeGenOptions.h"
#include "llvm/ADT/StringExtras.h"
using namespace clang;
using namespace CodeGen;

/// Determines whether the given function has a trivial body that does
/// not require any specific codegen.
static bool HasTrivialBody(const FunctionDecl *FD) {
  Stmt *S = FD->getBody();
  if (!S)
    return true;
  if (isa<CompoundStmt>(S) && cast<CompoundStmt>(S)->body_empty())
    return true;
  return false;
}

/// Try to emit a base destructor as an alias to its primary
/// base-class destructor.
bool CodeGenModule::TryEmitBaseDestructorAsAlias(const CXXDestructorDecl *D) {
  if (!getCodeGenOpts().CXXCtorDtorAliases)
    return true;

  // If the destructor doesn't have a trivial body, we have to emit it
  // separately.
  if (!HasTrivialBody(D))
    return true;

  const CXXRecordDecl *Class = D->getParent();

  // If we need to manipulate a VTT parameter, give up.
  if (Class->getNumVBases()) {
    // Extra Credit:  passing extra parameters is perfectly safe
    // in many calling conventions, so only bail out if the ctor's
    // calling convention is nonstandard.
    return true;
  }

  // If any fields have a non-trivial destructor, we have to emit it
  // separately.
  for (CXXRecordDecl::field_iterator I = Class->field_begin(),
         E = Class->field_end(); I != E; ++I)
    if (const RecordType *RT = (*I)->getType()->getAs<RecordType>())
      if (!cast<CXXRecordDecl>(RT->getDecl())->hasTrivialDestructor())
        return true;

  // Try to find a unique base class with a non-trivial destructor.
  const CXXRecordDecl *UniqueBase = 0;
  for (CXXRecordDecl::base_class_const_iterator I = Class->bases_begin(),
         E = Class->bases_end(); I != E; ++I) {

    // We're in the base destructor, so skip virtual bases.
    if (I->isVirtual()) continue;

    // Skip base classes with trivial destructors.
    const CXXRecordDecl *Base
      = cast<CXXRecordDecl>(I->getType()->getAs<RecordType>()->getDecl());
    if (Base->hasTrivialDestructor()) continue;

    // If we've already found a base class with a non-trivial
    // destructor, give up.
    if (UniqueBase) return true;
    UniqueBase = Base;
  }

  // If we didn't find any bases with a non-trivial destructor, then
  // the base destructor is actually effectively trivial, which can
  // happen if it was needlessly user-defined or if there are virtual
  // bases with non-trivial destructors.
  if (!UniqueBase)
    return true;

  /// If we don't have a definition for the destructor yet, don't
  /// emit.  We can't emit aliases to declarations; that's just not
  /// how aliases work.
  const CXXDestructorDecl *BaseD = UniqueBase->getDestructor();
  if (!BaseD->isImplicit() && !BaseD->hasBody())
    return true;

  // If the base is at a non-zero offset, give up.
  const ASTRecordLayout &ClassLayout = Context.getASTRecordLayout(Class);
  if (ClassLayout.getBaseClassOffset(UniqueBase) != 0)
    return true;

  return TryEmitDefinitionAsAlias(GlobalDecl(D, Dtor_Base),
                                  GlobalDecl(BaseD, Dtor_Base));
}

/// Try to emit a definition as a global alias for another definition.
bool CodeGenModule::TryEmitDefinitionAsAlias(GlobalDecl AliasDecl,
                                             GlobalDecl TargetDecl) {
  if (!getCodeGenOpts().CXXCtorDtorAliases)
    return true;

  // The alias will use the linkage of the referrent.  If we can't
  // support aliases with that linkage, fail.
  llvm::GlobalValue::LinkageTypes Linkage
    = getFunctionLinkage(cast<FunctionDecl>(AliasDecl.getDecl()));

  switch (Linkage) {
  // We can definitely emit aliases to definitions with external linkage.
  case llvm::GlobalValue::ExternalLinkage:
  case llvm::GlobalValue::ExternalWeakLinkage:
    break;

  // Same with local linkage.
  case llvm::GlobalValue::InternalLinkage:
  case llvm::GlobalValue::PrivateLinkage:
  case llvm::GlobalValue::LinkerPrivateLinkage:
    break;

  // We should try to support linkonce linkages.
  case llvm::GlobalValue::LinkOnceAnyLinkage:
  case llvm::GlobalValue::LinkOnceODRLinkage:
    return true;

  // Other linkages will probably never be supported.
  default:
    return true;
  }

  llvm::GlobalValue::LinkageTypes TargetLinkage
    = getFunctionLinkage(cast<FunctionDecl>(TargetDecl.getDecl()));

  if (llvm::GlobalValue::isWeakForLinker(TargetLinkage))
    return true;

  // Derive the type for the alias.
  const llvm::PointerType *AliasType
    = getTypes().GetFunctionType(AliasDecl)->getPointerTo();

  // Find the referrent.  Some aliases might require a bitcast, in
  // which case the caller is responsible for ensuring the soundness
  // of these semantics.
  llvm::GlobalValue *Ref = cast<llvm::GlobalValue>(GetAddrOfGlobal(TargetDecl));
  llvm::Constant *Aliasee = Ref;
  if (Ref->getType() != AliasType)
    Aliasee = llvm::ConstantExpr::getBitCast(Ref, AliasType);

  // Create the alias with no name.
  llvm::GlobalAlias *Alias = 
    new llvm::GlobalAlias(AliasType, Linkage, "", Aliasee, &getModule());

  // Switch any previous uses to the alias.
  llvm::StringRef MangledName = getMangledName(AliasDecl);
  llvm::GlobalValue *Entry = GetGlobalValue(MangledName);
  if (Entry) {
    assert(Entry->isDeclaration() && "definition already exists for alias");
    assert(Entry->getType() == AliasType &&
           "declaration exists with different type");
    Alias->takeName(Entry);
    Entry->replaceAllUsesWith(Alias);
    Entry->eraseFromParent();
  } else {
    Alias->setName(MangledName);
  }

  // Finally, set up the alias with its proper name and attributes.
  SetCommonAttributes(AliasDecl.getDecl(), Alias);

  return false;
}

void CodeGenModule::EmitCXXConstructors(const CXXConstructorDecl *D) {
  // The constructor used for constructing this as a complete class;
  // constucts the virtual bases, then calls the base constructor.
  EmitGlobal(GlobalDecl(D, Ctor_Complete));

  // The constructor used for constructing this as a base class;
  // ignores virtual bases.
  EmitGlobal(GlobalDecl(D, Ctor_Base));
}

void CodeGenModule::EmitCXXConstructor(const CXXConstructorDecl *D,
                                       CXXCtorType Type) {
  // The complete constructor is equivalent to the base constructor
  // for classes with no virtual bases.  Try to emit it as an alias.
  if (Type == Ctor_Complete &&
      !D->getParent()->getNumVBases() &&
      !TryEmitDefinitionAsAlias(GlobalDecl(D, Ctor_Complete),
                                GlobalDecl(D, Ctor_Base)))
    return;

  llvm::Function *Fn = cast<llvm::Function>(GetAddrOfCXXConstructor(D, Type));
  setFunctionLinkage(D, Fn);

  CodeGenFunction(*this).GenerateCode(GlobalDecl(D, Type), Fn);

  SetFunctionDefinitionAttributes(D, Fn);
  SetLLVMFunctionAttributesForDefinition(D, Fn);
}

llvm::GlobalValue *
CodeGenModule::GetAddrOfCXXConstructor(const CXXConstructorDecl *D,
                                       CXXCtorType Type) {
  GlobalDecl GD(D, Type);
  
  llvm::StringRef Name = getMangledName(GD);
  if (llvm::GlobalValue *V = GetGlobalValue(Name))
    return V;

  const FunctionProtoType *FPT = D->getType()->getAs<FunctionProtoType>();
  const llvm::FunctionType *FTy =
    getTypes().GetFunctionType(getTypes().getFunctionInfo(D, Type), 
                               FPT->isVariadic());
  return cast<llvm::Function>(GetOrCreateLLVMFunction(Name, FTy, GD));
}

void CodeGenModule::EmitCXXDestructors(const CXXDestructorDecl *D) {
  // The destructor in a virtual table is always a 'deleting'
  // destructor, which calls the complete destructor and then uses the
  // appropriate operator delete.
  if (D->isVirtual())
    EmitGlobal(GlobalDecl(D, Dtor_Deleting));

  // The destructor used for destructing this as a most-derived class;
  // call the base destructor and then destructs any virtual bases.
  EmitGlobal(GlobalDecl(D, Dtor_Complete));

  // The destructor used for destructing this as a base class; ignores
  // virtual bases.
  EmitGlobal(GlobalDecl(D, Dtor_Base));
}

void CodeGenModule::EmitCXXDestructor(const CXXDestructorDecl *D,
                                      CXXDtorType Type) {
  // The complete destructor is equivalent to the base destructor for
  // classes with no virtual bases, so try to emit it as an alias.
  if (Type == Dtor_Complete &&
      !D->getParent()->getNumVBases() &&
      !TryEmitDefinitionAsAlias(GlobalDecl(D, Dtor_Complete),
                                GlobalDecl(D, Dtor_Base)))
    return;

  // The base destructor is equivalent to the base destructor of its
  // base class if there is exactly one non-virtual base class with a
  // non-trivial destructor, there are no fields with a non-trivial
  // destructor, and the body of the destructor is trivial.
  if (Type == Dtor_Base && !TryEmitBaseDestructorAsAlias(D))
    return;

  llvm::Function *Fn = cast<llvm::Function>(GetAddrOfCXXDestructor(D, Type));
  setFunctionLinkage(D, Fn);

  CodeGenFunction(*this).GenerateCode(GlobalDecl(D, Type), Fn);

  SetFunctionDefinitionAttributes(D, Fn);
  SetLLVMFunctionAttributesForDefinition(D, Fn);
}

llvm::GlobalValue *
CodeGenModule::GetAddrOfCXXDestructor(const CXXDestructorDecl *D,
                                      CXXDtorType Type) {
  GlobalDecl GD(D, Type);

  llvm::StringRef Name = getMangledName(GD);
  if (llvm::GlobalValue *V = GetGlobalValue(Name))
    return V;

  const llvm::FunctionType *FTy =
    getTypes().GetFunctionType(getTypes().getFunctionInfo(D, Type), false);

  return cast<llvm::Function>(GetOrCreateLLVMFunction(Name, FTy, GD));
}

static llvm::Value *BuildVirtualCall(CodeGenFunction &CGF, uint64_t VTableIndex, 
                                     llvm::Value *This, const llvm::Type *Ty) {
  Ty = Ty->getPointerTo()->getPointerTo()->getPointerTo();
  
  llvm::Value *VTable = CGF.Builder.CreateBitCast(This, Ty);
  VTable = CGF.Builder.CreateLoad(VTable);
  
  llvm::Value *VFuncPtr = 
    CGF.Builder.CreateConstInBoundsGEP1_64(VTable, VTableIndex, "vfn");
  return CGF.Builder.CreateLoad(VFuncPtr);
}

llvm::Value *
CodeGenFunction::BuildVirtualCall(const CXXMethodDecl *MD, llvm::Value *This,
                                  const llvm::Type *Ty) {
  MD = MD->getCanonicalDecl();
  uint64_t VTableIndex = CGM.getVTables().getMethodVTableIndex(MD);
  
  return ::BuildVirtualCall(*this, VTableIndex, This, Ty);
}

llvm::Value *
CodeGenFunction::BuildVirtualCall(const CXXDestructorDecl *DD, CXXDtorType Type, 
                                  llvm::Value *&This, const llvm::Type *Ty) {
  DD = cast<CXXDestructorDecl>(DD->getCanonicalDecl());
  uint64_t VTableIndex = 
    CGM.getVTables().getMethodVTableIndex(GlobalDecl(DD, Type));

  return ::BuildVirtualCall(*this, VTableIndex, This, Ty);
}

/// Implementation for CGCXXABI.  Possibly this should be moved into
/// the incomplete ABI implementations?

CGCXXABI::~CGCXXABI() {}

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
