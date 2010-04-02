//===--- CodeGenModule.cpp - Emit LLVM Code from ASTs for a Module --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This coordinates the per-module state used while generating code.
//
//===----------------------------------------------------------------------===//

#include "CodeGenModule.h"
#include "CGDebugInfo.h"
#include "CodeGenFunction.h"
#include "CGCall.h"
#include "CGObjCRuntime.h"
#include "Mangle.h"
#include "TargetInfo.h"
#include "clang/CodeGen/CodeGenOptions.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/CharUnits.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/RecordLayout.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/ConvertUTF.h"
#include "llvm/CallingConv.h"
#include "llvm/Module.h"
#include "llvm/Intrinsics.h"
#include "llvm/LLVMContext.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Support/ErrorHandling.h"
using namespace clang;
using namespace CodeGen;


CodeGenModule::CodeGenModule(ASTContext &C, const CodeGenOptions &CGO,
                             llvm::Module &M, const llvm::TargetData &TD,
                             Diagnostic &diags)
  : BlockModule(C, M, TD, Types, *this), Context(C),
    Features(C.getLangOptions()), CodeGenOpts(CGO), TheModule(M),
    TheTargetData(TD), TheTargetCodeGenInfo(0), Diags(diags),
    Types(C, M, TD, getTargetCodeGenInfo().getABIInfo()),
    MangleCtx(C), VTables(*this), Runtime(0), CFConstantStringClassRef(0),
    VMContext(M.getContext()) {

  if (!Features.ObjC1)
    Runtime = 0;
  else if (!Features.NeXTRuntime)
    Runtime = CreateGNUObjCRuntime(*this);
  else if (Features.ObjCNonFragileABI)
    Runtime = CreateMacNonFragileABIObjCRuntime(*this);
  else
    Runtime = CreateMacObjCRuntime(*this);

  // If debug info generation is enabled, create the CGDebugInfo object.
  DebugInfo = CodeGenOpts.DebugInfo ? new CGDebugInfo(*this) : 0;
}

CodeGenModule::~CodeGenModule() {
  delete Runtime;
  delete DebugInfo;
}

void CodeGenModule::createObjCRuntime() {
  if (!Features.NeXTRuntime)
    Runtime = CreateGNUObjCRuntime(*this);
  else if (Features.ObjCNonFragileABI)
    Runtime = CreateMacNonFragileABIObjCRuntime(*this);
  else
    Runtime = CreateMacObjCRuntime(*this);
}

void CodeGenModule::Release() {
  EmitFundamentalRTTIDescriptors();
  EmitDeferred();
  EmitCXXGlobalInitFunc();
  EmitCXXGlobalDtorFunc();
  if (Runtime)
    if (llvm::Function *ObjCInitFunction = Runtime->ModuleInitFunction())
      AddGlobalCtor(ObjCInitFunction);
  EmitCtorList(GlobalCtors, "llvm.global_ctors");
  EmitCtorList(GlobalDtors, "llvm.global_dtors");
  EmitAnnotations();
  EmitLLVMUsed();
}

bool CodeGenModule::isTargetDarwin() const {
  return getContext().Target.getTriple().getOS() == llvm::Triple::Darwin;
}

/// ErrorUnsupported - Print out an error that codegen doesn't support the
/// specified stmt yet.
void CodeGenModule::ErrorUnsupported(const Stmt *S, const char *Type,
                                     bool OmitOnError) {
  if (OmitOnError && getDiags().hasErrorOccurred())
    return;
  unsigned DiagID = getDiags().getCustomDiagID(Diagnostic::Error,
                                               "cannot compile this %0 yet");
  std::string Msg = Type;
  getDiags().Report(Context.getFullLoc(S->getLocStart()), DiagID)
    << Msg << S->getSourceRange();
}

/// ErrorUnsupported - Print out an error that codegen doesn't support the
/// specified decl yet.
void CodeGenModule::ErrorUnsupported(const Decl *D, const char *Type,
                                     bool OmitOnError) {
  if (OmitOnError && getDiags().hasErrorOccurred())
    return;
  unsigned DiagID = getDiags().getCustomDiagID(Diagnostic::Error,
                                               "cannot compile this %0 yet");
  std::string Msg = Type;
  getDiags().Report(Context.getFullLoc(D->getLocation()), DiagID) << Msg;
}

LangOptions::VisibilityMode
CodeGenModule::getDeclVisibilityMode(const Decl *D) const {
  if (const VarDecl *VD = dyn_cast<VarDecl>(D))
    if (VD->getStorageClass() == VarDecl::PrivateExtern)
      return LangOptions::Hidden;

  if (const VisibilityAttr *attr = D->getAttr<VisibilityAttr>()) {
    switch (attr->getVisibility()) {
    default: assert(0 && "Unknown visibility!");
    case VisibilityAttr::DefaultVisibility:
      return LangOptions::Default;
    case VisibilityAttr::HiddenVisibility:
      return LangOptions::Hidden;
    case VisibilityAttr::ProtectedVisibility:
      return LangOptions::Protected;
    }
  }

  // This decl should have the same visibility as its parent.
  if (const DeclContext *DC = D->getDeclContext()) 
    return getDeclVisibilityMode(cast<Decl>(DC));

  return getLangOptions().getVisibilityMode();
}

void CodeGenModule::setGlobalVisibility(llvm::GlobalValue *GV,
                                        const Decl *D) const {
  // Internal definitions always have default visibility.
  if (GV->hasLocalLinkage()) {
    GV->setVisibility(llvm::GlobalValue::DefaultVisibility);
    return;
  }

  switch (getDeclVisibilityMode(D)) {
  default: assert(0 && "Unknown visibility!");
  case LangOptions::Default:
    return GV->setVisibility(llvm::GlobalValue::DefaultVisibility);
  case LangOptions::Hidden:
    return GV->setVisibility(llvm::GlobalValue::HiddenVisibility);
  case LangOptions::Protected:
    return GV->setVisibility(llvm::GlobalValue::ProtectedVisibility);
  }
}

void CodeGenModule::getMangledName(MangleBuffer &Buffer, GlobalDecl GD) {
  const NamedDecl *ND = cast<NamedDecl>(GD.getDecl());

  if (const CXXConstructorDecl *D = dyn_cast<CXXConstructorDecl>(ND))
    return getMangledCXXCtorName(Buffer, D, GD.getCtorType());
  if (const CXXDestructorDecl *D = dyn_cast<CXXDestructorDecl>(ND))
    return getMangledCXXDtorName(Buffer, D, GD.getDtorType());

  return getMangledName(Buffer, ND);
}

/// \brief Retrieves the mangled name for the given declaration.
///
/// If the given declaration requires a mangled name, returns an
/// const char* containing the mangled name.  Otherwise, returns
/// the unmangled name.
///
void CodeGenModule::getMangledName(MangleBuffer &Buffer,
                                   const NamedDecl *ND) {
  if (!getMangleContext().shouldMangleDeclName(ND)) {
    assert(ND->getIdentifier() && "Attempt to mangle unnamed decl.");
    Buffer.setString(ND->getNameAsCString());
    return;
  }

  getMangleContext().mangleName(ND, Buffer.getBuffer());
}

llvm::GlobalValue *CodeGenModule::GetGlobalValue(llvm::StringRef Name) {
  return getModule().getNamedValue(Name);
}

/// AddGlobalCtor - Add a function to the list that will be called before
/// main() runs.
void CodeGenModule::AddGlobalCtor(llvm::Function * Ctor, int Priority) {
  // FIXME: Type coercion of void()* types.
  GlobalCtors.push_back(std::make_pair(Ctor, Priority));
}

/// AddGlobalDtor - Add a function to the list that will be called
/// when the module is unloaded.
void CodeGenModule::AddGlobalDtor(llvm::Function * Dtor, int Priority) {
  // FIXME: Type coercion of void()* types.
  GlobalDtors.push_back(std::make_pair(Dtor, Priority));
}

void CodeGenModule::EmitCtorList(const CtorList &Fns, const char *GlobalName) {
  // Ctor function type is void()*.
  llvm::FunctionType* CtorFTy =
    llvm::FunctionType::get(llvm::Type::getVoidTy(VMContext),
                            std::vector<const llvm::Type*>(),
                            false);
  llvm::Type *CtorPFTy = llvm::PointerType::getUnqual(CtorFTy);

  // Get the type of a ctor entry, { i32, void ()* }.
  llvm::StructType* CtorStructTy =
    llvm::StructType::get(VMContext, llvm::Type::getInt32Ty(VMContext),
                          llvm::PointerType::getUnqual(CtorFTy), NULL);

  // Construct the constructor and destructor arrays.
  std::vector<llvm::Constant*> Ctors;
  for (CtorList::const_iterator I = Fns.begin(), E = Fns.end(); I != E; ++I) {
    std::vector<llvm::Constant*> S;
    S.push_back(llvm::ConstantInt::get(llvm::Type::getInt32Ty(VMContext),
                I->second, false));
    S.push_back(llvm::ConstantExpr::getBitCast(I->first, CtorPFTy));
    Ctors.push_back(llvm::ConstantStruct::get(CtorStructTy, S));
  }

  if (!Ctors.empty()) {
    llvm::ArrayType *AT = llvm::ArrayType::get(CtorStructTy, Ctors.size());
    new llvm::GlobalVariable(TheModule, AT, false,
                             llvm::GlobalValue::AppendingLinkage,
                             llvm::ConstantArray::get(AT, Ctors),
                             GlobalName);
  }
}

void CodeGenModule::EmitAnnotations() {
  if (Annotations.empty())
    return;

  // Create a new global variable for the ConstantStruct in the Module.
  llvm::Constant *Array =
  llvm::ConstantArray::get(llvm::ArrayType::get(Annotations[0]->getType(),
                                                Annotations.size()),
                           Annotations);
  llvm::GlobalValue *gv =
  new llvm::GlobalVariable(TheModule, Array->getType(), false,
                           llvm::GlobalValue::AppendingLinkage, Array,
                           "llvm.global.annotations");
  gv->setSection("llvm.metadata");
}

static CodeGenModule::GVALinkage
GetLinkageForFunction(ASTContext &Context, const FunctionDecl *FD,
                      const LangOptions &Features) {
  CodeGenModule::GVALinkage External = CodeGenModule::GVA_StrongExternal;

  Linkage L = FD->getLinkage();
  if (L == ExternalLinkage && Context.getLangOptions().CPlusPlus &&
      FD->getType()->getLinkage() == UniqueExternalLinkage)
    L = UniqueExternalLinkage;
  
  switch (L) {
  case NoLinkage:
  case InternalLinkage:
  case UniqueExternalLinkage:
    return CodeGenModule::GVA_Internal;
    
  case ExternalLinkage:
    switch (FD->getTemplateSpecializationKind()) {
    case TSK_Undeclared:
    case TSK_ExplicitSpecialization:
      External = CodeGenModule::GVA_StrongExternal;
      break;

    case TSK_ExplicitInstantiationDefinition:
      return CodeGenModule::GVA_ExplicitTemplateInstantiation;

    case TSK_ExplicitInstantiationDeclaration:
    case TSK_ImplicitInstantiation:
      External = CodeGenModule::GVA_TemplateInstantiation;
      break;
    }
  }

  if (!FD->isInlined())
    return External;
    
  if (!Features.CPlusPlus || FD->hasAttr<GNUInlineAttr>()) {
    // GNU or C99 inline semantics. Determine whether this symbol should be
    // externally visible.
    if (FD->isInlineDefinitionExternallyVisible())
      return External;

    // C99 inline semantics, where the symbol is not externally visible.
    return CodeGenModule::GVA_C99Inline;
  }

  // C++0x [temp.explicit]p9:
  //   [ Note: The intent is that an inline function that is the subject of 
  //   an explicit instantiation declaration will still be implicitly 
  //   instantiated when used so that the body can be considered for 
  //   inlining, but that no out-of-line copy of the inline function would be
  //   generated in the translation unit. -- end note ]
  if (FD->getTemplateSpecializationKind() 
                                       == TSK_ExplicitInstantiationDeclaration)
    return CodeGenModule::GVA_C99Inline;
  
  return CodeGenModule::GVA_CXXInline;
}

llvm::GlobalValue::LinkageTypes
CodeGenModule::getFunctionLinkage(const FunctionDecl *D) {
  GVALinkage Linkage = GetLinkageForFunction(getContext(), D, Features);

  if (Linkage == GVA_Internal) {
    return llvm::Function::InternalLinkage;
  } else if (D->hasAttr<DLLExportAttr>()) {
    return llvm::Function::DLLExportLinkage;
  } else if (D->hasAttr<WeakAttr>()) {
    return llvm::Function::WeakAnyLinkage;
  } else if (Linkage == GVA_C99Inline) {
    // In C99 mode, 'inline' functions are guaranteed to have a strong
    // definition somewhere else, so we can use available_externally linkage.
    return llvm::Function::AvailableExternallyLinkage;
  } else if (Linkage == GVA_CXXInline || Linkage == GVA_TemplateInstantiation) {
    // In C++, the compiler has to emit a definition in every translation unit
    // that references the function.  We should use linkonce_odr because
    // a) if all references in this translation unit are optimized away, we
    // don't need to codegen it.  b) if the function persists, it needs to be
    // merged with other definitions. c) C++ has the ODR, so we know the
    // definition is dependable.
    return llvm::Function::LinkOnceODRLinkage;
  } else if (Linkage == GVA_ExplicitTemplateInstantiation) {
    // An explicit instantiation of a template has weak linkage, since
    // explicit instantiations can occur in multiple translation units
    // and must all be equivalent. However, we are not allowed to
    // throw away these explicit instantiations.
    return llvm::Function::WeakODRLinkage;
  } else {
    assert(Linkage == GVA_StrongExternal);
    // Otherwise, we have strong external linkage.
    return llvm::Function::ExternalLinkage;
  }
}


/// SetFunctionDefinitionAttributes - Set attributes for a global.
///
/// FIXME: This is currently only done for aliases and functions, but not for
/// variables (these details are set in EmitGlobalVarDefinition for variables).
void CodeGenModule::SetFunctionDefinitionAttributes(const FunctionDecl *D,
                                                    llvm::GlobalValue *GV) {
  GV->setLinkage(getFunctionLinkage(D));
  SetCommonAttributes(D, GV);
}

void CodeGenModule::SetLLVMFunctionAttributes(const Decl *D,
                                              const CGFunctionInfo &Info,
                                              llvm::Function *F) {
  unsigned CallingConv;
  AttributeListType AttributeList;
  ConstructAttributeList(Info, D, AttributeList, CallingConv);
  F->setAttributes(llvm::AttrListPtr::get(AttributeList.begin(),
                                          AttributeList.size()));
  F->setCallingConv(static_cast<llvm::CallingConv::ID>(CallingConv));
}

void CodeGenModule::SetLLVMFunctionAttributesForDefinition(const Decl *D,
                                                           llvm::Function *F) {
  if (!Features.Exceptions && !Features.ObjCNonFragileABI)
    F->addFnAttr(llvm::Attribute::NoUnwind);

  if (D->hasAttr<AlwaysInlineAttr>())
    F->addFnAttr(llvm::Attribute::AlwaysInline);

  if (D->hasAttr<NoInlineAttr>())
    F->addFnAttr(llvm::Attribute::NoInline);

  if (Features.getStackProtectorMode() == LangOptions::SSPOn)
    F->addFnAttr(llvm::Attribute::StackProtect);
  else if (Features.getStackProtectorMode() == LangOptions::SSPReq)
    F->addFnAttr(llvm::Attribute::StackProtectReq);
  
  if (const AlignedAttr *AA = D->getAttr<AlignedAttr>()) {
    unsigned width = Context.Target.getCharWidth();
    F->setAlignment(AA->getAlignment() / width);
    while ((AA = AA->getNext<AlignedAttr>()))
      F->setAlignment(std::max(F->getAlignment(), AA->getAlignment() / width));
  }
  // C++ ABI requires 2-byte alignment for member functions.
  if (F->getAlignment() < 2 && isa<CXXMethodDecl>(D))
    F->setAlignment(2);
}

void CodeGenModule::SetCommonAttributes(const Decl *D,
                                        llvm::GlobalValue *GV) {
  setGlobalVisibility(GV, D);

  if (D->hasAttr<UsedAttr>())
    AddUsedGlobal(GV);

  if (const SectionAttr *SA = D->getAttr<SectionAttr>())
    GV->setSection(SA->getName());

  getTargetCodeGenInfo().SetTargetAttributes(D, GV, *this);
}

void CodeGenModule::SetInternalFunctionAttributes(const Decl *D,
                                                  llvm::Function *F,
                                                  const CGFunctionInfo &FI) {
  SetLLVMFunctionAttributes(D, FI, F);
  SetLLVMFunctionAttributesForDefinition(D, F);

  F->setLinkage(llvm::Function::InternalLinkage);

  SetCommonAttributes(D, F);
}

void CodeGenModule::SetFunctionAttributes(GlobalDecl GD,
                                          llvm::Function *F,
                                          bool IsIncompleteFunction) {
  const FunctionDecl *FD = cast<FunctionDecl>(GD.getDecl());

  if (!IsIncompleteFunction)
    SetLLVMFunctionAttributes(FD, getTypes().getFunctionInfo(GD), F);

  // Only a few attributes are set on declarations; these may later be
  // overridden by a definition.

  if (FD->hasAttr<DLLImportAttr>()) {
    F->setLinkage(llvm::Function::DLLImportLinkage);
  } else if (FD->hasAttr<WeakAttr>() ||
             FD->hasAttr<WeakImportAttr>()) {
    // "extern_weak" is overloaded in LLVM; we probably should have
    // separate linkage types for this.
    F->setLinkage(llvm::Function::ExternalWeakLinkage);
  } else {
    F->setLinkage(llvm::Function::ExternalLinkage);
  }

  if (const SectionAttr *SA = FD->getAttr<SectionAttr>())
    F->setSection(SA->getName());
}

void CodeGenModule::AddUsedGlobal(llvm::GlobalValue *GV) {
  assert(!GV->isDeclaration() &&
         "Only globals with definition can force usage.");
  LLVMUsed.push_back(GV);
}

void CodeGenModule::EmitLLVMUsed() {
  // Don't create llvm.used if there is no need.
  if (LLVMUsed.empty())
    return;

  const llvm::Type *i8PTy = llvm::Type::getInt8PtrTy(VMContext);

  // Convert LLVMUsed to what ConstantArray needs.
  std::vector<llvm::Constant*> UsedArray;
  UsedArray.resize(LLVMUsed.size());
  for (unsigned i = 0, e = LLVMUsed.size(); i != e; ++i) {
    UsedArray[i] =
     llvm::ConstantExpr::getBitCast(cast<llvm::Constant>(&*LLVMUsed[i]),
                                      i8PTy);
  }

  if (UsedArray.empty())
    return;
  llvm::ArrayType *ATy = llvm::ArrayType::get(i8PTy, UsedArray.size());

  llvm::GlobalVariable *GV =
    new llvm::GlobalVariable(getModule(), ATy, false,
                             llvm::GlobalValue::AppendingLinkage,
                             llvm::ConstantArray::get(ATy, UsedArray),
                             "llvm.used");

  GV->setSection("llvm.metadata");
}

void CodeGenModule::EmitDeferred() {
  // Emit code for any potentially referenced deferred decls.  Since a
  // previously unused static decl may become used during the generation of code
  // for a static function, iterate until no  changes are made.

  while (!DeferredDeclsToEmit.empty() || !DeferredVtables.empty()) {
    if (!DeferredVtables.empty()) {
      const CXXRecordDecl *RD = DeferredVtables.back();
      DeferredVtables.pop_back();
      getVTables().GenerateClassData(getVtableLinkage(RD), RD);
      continue;
    }

    GlobalDecl D = DeferredDeclsToEmit.back();
    DeferredDeclsToEmit.pop_back();

    // Look it up to see if it was defined with a stronger definition (e.g. an
    // extern inline function with a strong function redefinition).  If so,
    // just ignore the deferred decl.
    MangleBuffer Name;
    getMangledName(Name, D);
    llvm::GlobalValue *CGRef = GetGlobalValue(Name);
    assert(CGRef && "Deferred decl wasn't referenced?");

    if (!CGRef->isDeclaration())
      continue;

    // Otherwise, emit the definition and move on to the next one.
    EmitGlobalDefinition(D);
  }
}

/// EmitAnnotateAttr - Generate the llvm::ConstantStruct which contains the
/// annotation information for a given GlobalValue.  The annotation struct is
/// {i8 *, i8 *, i8 *, i32}.  The first field is a constant expression, the
/// GlobalValue being annotated.  The second field is the constant string
/// created from the AnnotateAttr's annotation.  The third field is a constant
/// string containing the name of the translation unit.  The fourth field is
/// the line number in the file of the annotated value declaration.
///
/// FIXME: this does not unique the annotation string constants, as llvm-gcc
///        appears to.
///
llvm::Constant *CodeGenModule::EmitAnnotateAttr(llvm::GlobalValue *GV,
                                                const AnnotateAttr *AA,
                                                unsigned LineNo) {
  llvm::Module *M = &getModule();

  // get [N x i8] constants for the annotation string, and the filename string
  // which are the 2nd and 3rd elements of the global annotation structure.
  const llvm::Type *SBP = llvm::Type::getInt8PtrTy(VMContext);
  llvm::Constant *anno = llvm::ConstantArray::get(VMContext,
                                                  AA->getAnnotation(), true);
  llvm::Constant *unit = llvm::ConstantArray::get(VMContext,
                                                  M->getModuleIdentifier(),
                                                  true);

  // Get the two global values corresponding to the ConstantArrays we just
  // created to hold the bytes of the strings.
  llvm::GlobalValue *annoGV =
    new llvm::GlobalVariable(*M, anno->getType(), false,
                             llvm::GlobalValue::PrivateLinkage, anno,
                             GV->getName());
  // translation unit name string, emitted into the llvm.metadata section.
  llvm::GlobalValue *unitGV =
    new llvm::GlobalVariable(*M, unit->getType(), false,
                             llvm::GlobalValue::PrivateLinkage, unit,
                             ".str");

  // Create the ConstantStruct for the global annotation.
  llvm::Constant *Fields[4] = {
    llvm::ConstantExpr::getBitCast(GV, SBP),
    llvm::ConstantExpr::getBitCast(annoGV, SBP),
    llvm::ConstantExpr::getBitCast(unitGV, SBP),
    llvm::ConstantInt::get(llvm::Type::getInt32Ty(VMContext), LineNo)
  };
  return llvm::ConstantStruct::get(VMContext, Fields, 4, false);
}

bool CodeGenModule::MayDeferGeneration(const ValueDecl *Global) {
  // Never defer when EmitAllDecls is specified or the decl has
  // attribute used.
  if (Features.EmitAllDecls || Global->hasAttr<UsedAttr>())
    return false;

  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(Global)) {
    // Constructors and destructors should never be deferred.
    if (FD->hasAttr<ConstructorAttr>() ||
        FD->hasAttr<DestructorAttr>())
      return false;

    // The key function for a class must never be deferred.
    if (const CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(Global)) {
      const CXXRecordDecl *RD = MD->getParent();
      if (MD->isOutOfLine() && RD->isDynamicClass()) {
        const CXXMethodDecl *KeyFunction = getContext().getKeyFunction(RD);
        if (KeyFunction && 
            KeyFunction->getCanonicalDecl() == MD->getCanonicalDecl())
          return false;
      }
    }

    GVALinkage Linkage = GetLinkageForFunction(getContext(), FD, Features);

    // static, static inline, always_inline, and extern inline functions can
    // always be deferred.  Normal inline functions can be deferred in C99/C++.
    // Implicit template instantiations can also be deferred in C++.
    if (Linkage == GVA_Internal || Linkage == GVA_C99Inline ||
        Linkage == GVA_CXXInline || Linkage == GVA_TemplateInstantiation)
      return true;
    return false;
  }

  const VarDecl *VD = cast<VarDecl>(Global);
  assert(VD->isFileVarDecl() && "Invalid decl");

  // We never want to defer structs that have non-trivial constructors or 
  // destructors.
  
  // FIXME: Handle references.
  if (const RecordType *RT = VD->getType()->getAs<RecordType>()) {
    if (const CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(RT->getDecl())) {
      if (!RD->hasTrivialConstructor() || !RD->hasTrivialDestructor())
        return false;
    }
  }
      
  // Static data may be deferred, but out-of-line static data members
  // cannot be.
  Linkage L = VD->getLinkage();
  if (L == ExternalLinkage && getContext().getLangOptions().CPlusPlus &&
      VD->getType()->getLinkage() == UniqueExternalLinkage)
    L = UniqueExternalLinkage;

  switch (L) {
  case NoLinkage:
  case InternalLinkage:
  case UniqueExternalLinkage:
    // Initializer has side effects?
    if (VD->getInit() && VD->getInit()->HasSideEffects(Context))
      return false;
    return !(VD->isStaticDataMember() && VD->isOutOfLine());

  case ExternalLinkage:
    break;
  }

  return false;
}

llvm::Constant *CodeGenModule::GetWeakRefReference(const ValueDecl *VD) {
  const AliasAttr *AA = VD->getAttr<AliasAttr>();
  assert(AA && "No alias?");

  const llvm::Type *DeclTy = getTypes().ConvertTypeForMem(VD->getType());

  // See if there is already something with the target's name in the module.
  llvm::GlobalValue *Entry = GetGlobalValue(AA->getAliasee());

  llvm::Constant *Aliasee;
  if (isa<llvm::FunctionType>(DeclTy))
    Aliasee = GetOrCreateLLVMFunction(AA->getAliasee(), DeclTy, GlobalDecl());
  else
    Aliasee = GetOrCreateLLVMGlobal(AA->getAliasee(),
                                    llvm::PointerType::getUnqual(DeclTy), 0);
  if (!Entry) {
    llvm::GlobalValue* F = cast<llvm::GlobalValue>(Aliasee);
    F->setLinkage(llvm::Function::ExternalWeakLinkage);    
    WeakRefReferences.insert(F);
  }

  return Aliasee;
}

void CodeGenModule::EmitGlobal(GlobalDecl GD) {
  const ValueDecl *Global = cast<ValueDecl>(GD.getDecl());

  // Weak references don't produce any output by themselves.
  if (Global->hasAttr<WeakRefAttr>())
    return;

  // If this is an alias definition (which otherwise looks like a declaration)
  // emit it now.
  if (Global->hasAttr<AliasAttr>())
    return EmitAliasDefinition(GD);

  // Ignore declarations, they will be emitted on their first use.
  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(Global)) {
    // Forward declarations are emitted lazily on first use.
    if (!FD->isThisDeclarationADefinition())
      return;
  } else {
    const VarDecl *VD = cast<VarDecl>(Global);
    assert(VD->isFileVarDecl() && "Cannot emit local var decl as global.");

    if (VD->isThisDeclarationADefinition() != VarDecl::Definition)
      return;
  }

  // Defer code generation when possible if this is a static definition, inline
  // function etc.  These we only want to emit if they are used.
  if (MayDeferGeneration(Global)) {
    // If the value has already been used, add it directly to the
    // DeferredDeclsToEmit list.
    MangleBuffer MangledName;
    getMangledName(MangledName, GD);
    if (GetGlobalValue(MangledName))
      DeferredDeclsToEmit.push_back(GD);
    else {
      // Otherwise, remember that we saw a deferred decl with this name.  The
      // first use of the mangled name will cause it to move into
      // DeferredDeclsToEmit.
      DeferredDecls[MangledName] = GD;
    }
    return;
  }

  // Otherwise emit the definition.
  EmitGlobalDefinition(GD);
}

void CodeGenModule::EmitGlobalDefinition(GlobalDecl GD) {
  const ValueDecl *D = cast<ValueDecl>(GD.getDecl());

  PrettyStackTraceDecl CrashInfo((ValueDecl *)D, D->getLocation(), 
                                 Context.getSourceManager(),
                                 "Generating code for declaration");
  
  if (isa<CXXMethodDecl>(D))
    getVTables().EmitVTableRelatedData(GD);

  if (const CXXConstructorDecl *CD = dyn_cast<CXXConstructorDecl>(D))
    EmitCXXConstructor(CD, GD.getCtorType());
  else if (const CXXDestructorDecl *DD = dyn_cast<CXXDestructorDecl>(D))
    EmitCXXDestructor(DD, GD.getDtorType());
  else if (isa<FunctionDecl>(D))
    EmitGlobalFunctionDefinition(GD);
  else if (const VarDecl *VD = dyn_cast<VarDecl>(D))
    EmitGlobalVarDefinition(VD);
  else {
    assert(0 && "Invalid argument to EmitGlobalDefinition()");
  }
}

/// GetOrCreateLLVMFunction - If the specified mangled name is not in the
/// module, create and return an llvm Function with the specified type. If there
/// is something in the module with the specified name, return it potentially
/// bitcasted to the right type.
///
/// If D is non-null, it specifies a decl that correspond to this.  This is used
/// to set the attributes on the function when it is first created.
llvm::Constant *
CodeGenModule::GetOrCreateLLVMFunction(llvm::StringRef MangledName,
                                       const llvm::Type *Ty,
                                       GlobalDecl D) {
  // Lookup the entry, lazily creating it if necessary.
  llvm::GlobalValue *Entry = GetGlobalValue(MangledName);
  if (Entry) {
    if (WeakRefReferences.count(Entry)) {
      const FunctionDecl *FD = cast_or_null<FunctionDecl>(D.getDecl());
      if (FD && !FD->hasAttr<WeakAttr>())
        Entry->setLinkage(llvm::Function::ExternalLinkage);

      WeakRefReferences.erase(Entry);
    }

    if (Entry->getType()->getElementType() == Ty)
      return Entry;

    // Make sure the result is of the correct type.
    const llvm::Type *PTy = llvm::PointerType::getUnqual(Ty);
    return llvm::ConstantExpr::getBitCast(Entry, PTy);
  }

  // This function doesn't have a complete type (for example, the return
  // type is an incomplete struct). Use a fake type instead, and make
  // sure not to try to set attributes.
  bool IsIncompleteFunction = false;
  if (!isa<llvm::FunctionType>(Ty)) {
    Ty = llvm::FunctionType::get(llvm::Type::getVoidTy(VMContext),
                                 std::vector<const llvm::Type*>(), false);
    IsIncompleteFunction = true;
  }
  llvm::Function *F = llvm::Function::Create(cast<llvm::FunctionType>(Ty),
                                             llvm::Function::ExternalLinkage,
                                             MangledName, &getModule());
  assert(F->getName() == MangledName && "name was uniqued!");
  if (D.getDecl())
    SetFunctionAttributes(D, F, IsIncompleteFunction);

  // This is the first use or definition of a mangled name.  If there is a
  // deferred decl with this name, remember that we need to emit it at the end
  // of the file.
  llvm::StringMap<GlobalDecl>::iterator DDI = DeferredDecls.find(MangledName);
  if (DDI != DeferredDecls.end()) {
    // Move the potentially referenced deferred decl to the DeferredDeclsToEmit
    // list, and remove it from DeferredDecls (since we don't need it anymore).
    DeferredDeclsToEmit.push_back(DDI->second);
    DeferredDecls.erase(DDI);
  } else if (const FunctionDecl *FD = cast_or_null<FunctionDecl>(D.getDecl())) {
    // If this the first reference to a C++ inline function in a class, queue up
    // the deferred function body for emission.  These are not seen as
    // top-level declarations.
    if (FD->isThisDeclarationADefinition() && MayDeferGeneration(FD))
      DeferredDeclsToEmit.push_back(D);
    // A called constructor which has no definition or declaration need be
    // synthesized.
    else if (const CXXConstructorDecl *CD = dyn_cast<CXXConstructorDecl>(FD)) {
      if (CD->isImplicit()) {
        assert(CD->isUsed() && "Sema doesn't consider constructor as used.");
        DeferredDeclsToEmit.push_back(D);
      }
    } else if (const CXXDestructorDecl *DD = dyn_cast<CXXDestructorDecl>(FD)) {
      if (DD->isImplicit()) {
        assert(DD->isUsed() && "Sema doesn't consider destructor as used.");
        DeferredDeclsToEmit.push_back(D);
      }
    } else if (const CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(FD)) {
      if (MD->isCopyAssignment() && MD->isImplicit()) {
        assert(MD->isUsed() && "Sema doesn't consider CopyAssignment as used.");
        DeferredDeclsToEmit.push_back(D);
      }
    }
  }

  return F;
}

/// GetAddrOfFunction - Return the address of the given function.  If Ty is
/// non-null, then this function will use the specified type if it has to
/// create it (this occurs when we see a definition of the function).
llvm::Constant *CodeGenModule::GetAddrOfFunction(GlobalDecl GD,
                                                 const llvm::Type *Ty) {
  // If there was no specific requested type, just convert it now.
  if (!Ty)
    Ty = getTypes().ConvertType(cast<ValueDecl>(GD.getDecl())->getType());
  MangleBuffer MangledName;
  getMangledName(MangledName, GD);
  return GetOrCreateLLVMFunction(MangledName, Ty, GD);
}

/// CreateRuntimeFunction - Create a new runtime function with the specified
/// type and name.
llvm::Constant *
CodeGenModule::CreateRuntimeFunction(const llvm::FunctionType *FTy,
                                     llvm::StringRef Name) {
  return GetOrCreateLLVMFunction(Name, FTy, GlobalDecl());
}

static bool DeclIsConstantGlobal(ASTContext &Context, const VarDecl *D) {
  if (!D->getType().isConstant(Context) && !D->getType()->isReferenceType())
    return false;
  if (Context.getLangOptions().CPlusPlus &&
      Context.getBaseElementType(D->getType())->getAs<RecordType>()) {
    // FIXME: We should do something fancier here!
    return false;
  }
  return true;
}

/// GetOrCreateLLVMGlobal - If the specified mangled name is not in the module,
/// create and return an llvm GlobalVariable with the specified type.  If there
/// is something in the module with the specified name, return it potentially
/// bitcasted to the right type.
///
/// If D is non-null, it specifies a decl that correspond to this.  This is used
/// to set the attributes on the global when it is first created.
llvm::Constant *
CodeGenModule::GetOrCreateLLVMGlobal(llvm::StringRef MangledName,
                                     const llvm::PointerType *Ty,
                                     const VarDecl *D) {
  // Lookup the entry, lazily creating it if necessary.
  llvm::GlobalValue *Entry = GetGlobalValue(MangledName);
  if (Entry) {
    if (WeakRefReferences.count(Entry)) {
      if (D && !D->hasAttr<WeakAttr>())
        Entry->setLinkage(llvm::Function::ExternalLinkage);

      WeakRefReferences.erase(Entry);
    }

    if (Entry->getType() == Ty)
      return Entry;

    // Make sure the result is of the correct type.
    return llvm::ConstantExpr::getBitCast(Entry, Ty);
  }

  // This is the first use or definition of a mangled name.  If there is a
  // deferred decl with this name, remember that we need to emit it at the end
  // of the file.
  llvm::StringMap<GlobalDecl>::iterator DDI = DeferredDecls.find(MangledName);
  if (DDI != DeferredDecls.end()) {
    // Move the potentially referenced deferred decl to the DeferredDeclsToEmit
    // list, and remove it from DeferredDecls (since we don't need it anymore).
    DeferredDeclsToEmit.push_back(DDI->second);
    DeferredDecls.erase(DDI);
  }

  llvm::GlobalVariable *GV =
    new llvm::GlobalVariable(getModule(), Ty->getElementType(), false,
                             llvm::GlobalValue::ExternalLinkage,
                             0, MangledName, 0,
                             false, Ty->getAddressSpace());

  // Handle things which are present even on external declarations.
  if (D) {
    // FIXME: This code is overly simple and should be merged with other global
    // handling.
    GV->setConstant(DeclIsConstantGlobal(Context, D));

    // FIXME: Merge with other attribute handling code.
    if (D->getStorageClass() == VarDecl::PrivateExtern)
      GV->setVisibility(llvm::GlobalValue::HiddenVisibility);

    if (D->hasAttr<WeakAttr>() ||
        D->hasAttr<WeakImportAttr>())
      GV->setLinkage(llvm::GlobalValue::ExternalWeakLinkage);

    GV->setThreadLocal(D->isThreadSpecified());
  }

  return GV;
}


/// GetAddrOfGlobalVar - Return the llvm::Constant for the address of the
/// given global variable.  If Ty is non-null and if the global doesn't exist,
/// then it will be greated with the specified type instead of whatever the
/// normal requested type would be.
llvm::Constant *CodeGenModule::GetAddrOfGlobalVar(const VarDecl *D,
                                                  const llvm::Type *Ty) {
  assert(D->hasGlobalStorage() && "Not a global variable");
  QualType ASTTy = D->getType();
  if (Ty == 0)
    Ty = getTypes().ConvertTypeForMem(ASTTy);

  const llvm::PointerType *PTy =
    llvm::PointerType::get(Ty, ASTTy.getAddressSpace());

  MangleBuffer MangledName;
  getMangledName(MangledName, D);
  return GetOrCreateLLVMGlobal(MangledName, PTy, D);
}

/// CreateRuntimeVariable - Create a new runtime global variable with the
/// specified type and name.
llvm::Constant *
CodeGenModule::CreateRuntimeVariable(const llvm::Type *Ty,
                                     llvm::StringRef Name) {
  return GetOrCreateLLVMGlobal(Name, llvm::PointerType::getUnqual(Ty), 0);
}

void CodeGenModule::EmitTentativeDefinition(const VarDecl *D) {
  assert(!D->getInit() && "Cannot emit definite definitions here!");

  if (MayDeferGeneration(D)) {
    // If we have not seen a reference to this variable yet, place it
    // into the deferred declarations table to be emitted if needed
    // later.
    MangleBuffer MangledName;
    getMangledName(MangledName, D);
    if (!GetGlobalValue(MangledName)) {
      DeferredDecls[MangledName] = D;
      return;
    }
  }

  // The tentative definition is the only definition.
  EmitGlobalVarDefinition(D);
}

llvm::GlobalVariable::LinkageTypes 
CodeGenModule::getVtableLinkage(const CXXRecordDecl *RD) {
  if (RD->isInAnonymousNamespace() || !RD->hasLinkage())
    return llvm::GlobalVariable::InternalLinkage;

  if (const CXXMethodDecl *KeyFunction
                                    = RD->getASTContext().getKeyFunction(RD)) {
    // If this class has a key function, use that to determine the linkage of
    // the vtable.
    const FunctionDecl *Def = 0;
    if (KeyFunction->getBody(Def))
      KeyFunction = cast<CXXMethodDecl>(Def);
    
    switch (KeyFunction->getTemplateSpecializationKind()) {
      case TSK_Undeclared:
      case TSK_ExplicitSpecialization:
        if (KeyFunction->isInlined())
          return llvm::GlobalVariable::WeakODRLinkage;
        
        return llvm::GlobalVariable::ExternalLinkage;
        
      case TSK_ImplicitInstantiation:
      case TSK_ExplicitInstantiationDefinition:
        return llvm::GlobalVariable::WeakODRLinkage;
        
      case TSK_ExplicitInstantiationDeclaration:
        // FIXME: Use available_externally linkage. However, this currently
        // breaks LLVM's build due to undefined symbols.
        //      return llvm::GlobalVariable::AvailableExternallyLinkage;
        return llvm::GlobalVariable::WeakODRLinkage;
    }
  }
  
  switch (RD->getTemplateSpecializationKind()) {
  case TSK_Undeclared:
  case TSK_ExplicitSpecialization:
  case TSK_ImplicitInstantiation:
  case TSK_ExplicitInstantiationDefinition:
    return llvm::GlobalVariable::WeakODRLinkage;
    
  case TSK_ExplicitInstantiationDeclaration:
    // FIXME: Use available_externally linkage. However, this currently
    // breaks LLVM's build due to undefined symbols.
    //   return llvm::GlobalVariable::AvailableExternallyLinkage;
    return llvm::GlobalVariable::WeakODRLinkage;
  }
  
  // Silence GCC warning.
  return llvm::GlobalVariable::WeakODRLinkage;
}

static CodeGenModule::GVALinkage
GetLinkageForVariable(ASTContext &Context, const VarDecl *VD) {
  // If this is a static data member, compute the kind of template
  // specialization. Otherwise, this variable is not part of a
  // template.
  TemplateSpecializationKind TSK = TSK_Undeclared;
  if (VD->isStaticDataMember())
    TSK = VD->getTemplateSpecializationKind();

  Linkage L = VD->getLinkage();
  if (L == ExternalLinkage && Context.getLangOptions().CPlusPlus &&
      VD->getType()->getLinkage() == UniqueExternalLinkage)
    L = UniqueExternalLinkage;

  switch (L) {
  case NoLinkage:
  case InternalLinkage:
  case UniqueExternalLinkage:
    return CodeGenModule::GVA_Internal;

  case ExternalLinkage:
    switch (TSK) {
    case TSK_Undeclared:
    case TSK_ExplicitSpecialization:
      return CodeGenModule::GVA_StrongExternal;

    case TSK_ExplicitInstantiationDeclaration:
      llvm_unreachable("Variable should not be instantiated");
      // Fall through to treat this like any other instantiation.
        
    case TSK_ExplicitInstantiationDefinition:
      return CodeGenModule::GVA_ExplicitTemplateInstantiation;

    case TSK_ImplicitInstantiation:
      return CodeGenModule::GVA_TemplateInstantiation;      
    }
  }

  return CodeGenModule::GVA_StrongExternal;
}

CharUnits CodeGenModule::GetTargetTypeStoreSize(const llvm::Type *Ty) const {
    return CharUnits::fromQuantity(
      TheTargetData.getTypeStoreSizeInBits(Ty) / Context.getCharWidth());
}

void CodeGenModule::EmitGlobalVarDefinition(const VarDecl *D) {
  llvm::Constant *Init = 0;
  QualType ASTTy = D->getType();
  bool NonConstInit = false;

  const Expr *InitExpr = D->getAnyInitializer();
  
  if (!InitExpr) {
    // This is a tentative definition; tentative definitions are
    // implicitly initialized with { 0 }.
    //
    // Note that tentative definitions are only emitted at the end of
    // a translation unit, so they should never have incomplete
    // type. In addition, EmitTentativeDefinition makes sure that we
    // never attempt to emit a tentative definition if a real one
    // exists. A use may still exists, however, so we still may need
    // to do a RAUW.
    assert(!ASTTy->isIncompleteType() && "Unexpected incomplete type");
    Init = EmitNullConstant(D->getType());
  } else {
    Init = EmitConstantExpr(InitExpr, D->getType());

    if (!Init) {
      QualType T = InitExpr->getType();
      if (getLangOptions().CPlusPlus) {
        EmitCXXGlobalVarDeclInitFunc(D);
        Init = EmitNullConstant(T);
        NonConstInit = true;
      } else {
        ErrorUnsupported(D, "static initializer");
        Init = llvm::UndefValue::get(getTypes().ConvertType(T));
      }
    }
  }

  const llvm::Type* InitType = Init->getType();
  llvm::Constant *Entry = GetAddrOfGlobalVar(D, InitType);

  // Strip off a bitcast if we got one back.
  if (llvm::ConstantExpr *CE = dyn_cast<llvm::ConstantExpr>(Entry)) {
    assert(CE->getOpcode() == llvm::Instruction::BitCast ||
           // all zero index gep.
           CE->getOpcode() == llvm::Instruction::GetElementPtr);
    Entry = CE->getOperand(0);
  }

  // Entry is now either a Function or GlobalVariable.
  llvm::GlobalVariable *GV = dyn_cast<llvm::GlobalVariable>(Entry);

  // We have a definition after a declaration with the wrong type.
  // We must make a new GlobalVariable* and update everything that used OldGV
  // (a declaration or tentative definition) with the new GlobalVariable*
  // (which will be a definition).
  //
  // This happens if there is a prototype for a global (e.g.
  // "extern int x[];") and then a definition of a different type (e.g.
  // "int x[10];"). This also happens when an initializer has a different type
  // from the type of the global (this happens with unions).
  if (GV == 0 ||
      GV->getType()->getElementType() != InitType ||
      GV->getType()->getAddressSpace() != ASTTy.getAddressSpace()) {

    // Move the old entry aside so that we'll create a new one.
    Entry->setName(llvm::StringRef());

    // Make a new global with the correct type, this is now guaranteed to work.
    GV = cast<llvm::GlobalVariable>(GetAddrOfGlobalVar(D, InitType));

    // Replace all uses of the old global with the new global
    llvm::Constant *NewPtrForOldDecl =
        llvm::ConstantExpr::getBitCast(GV, Entry->getType());
    Entry->replaceAllUsesWith(NewPtrForOldDecl);

    // Erase the old global, since it is no longer used.
    cast<llvm::GlobalValue>(Entry)->eraseFromParent();
  }

  if (const AnnotateAttr *AA = D->getAttr<AnnotateAttr>()) {
    SourceManager &SM = Context.getSourceManager();
    AddAnnotation(EmitAnnotateAttr(GV, AA,
                              SM.getInstantiationLineNumber(D->getLocation())));
  }

  GV->setInitializer(Init);

  // If it is safe to mark the global 'constant', do so now.
  GV->setConstant(false);
  if (!NonConstInit && DeclIsConstantGlobal(Context, D))
    GV->setConstant(true);

  GV->setAlignment(getContext().getDeclAlign(D).getQuantity());

  // Set the llvm linkage type as appropriate.
  GVALinkage Linkage = GetLinkageForVariable(getContext(), D);
  if (Linkage == GVA_Internal)
    GV->setLinkage(llvm::Function::InternalLinkage);
  else if (D->hasAttr<DLLImportAttr>())
    GV->setLinkage(llvm::Function::DLLImportLinkage);
  else if (D->hasAttr<DLLExportAttr>())
    GV->setLinkage(llvm::Function::DLLExportLinkage);
  else if (D->hasAttr<WeakAttr>()) {
    if (GV->isConstant())
      GV->setLinkage(llvm::GlobalVariable::WeakODRLinkage);
    else
      GV->setLinkage(llvm::GlobalVariable::WeakAnyLinkage);
  } else if (Linkage == GVA_TemplateInstantiation ||
             Linkage == GVA_ExplicitTemplateInstantiation)
    // FIXME: It seems like we can provide more specific linkage here
    // (LinkOnceODR, WeakODR).
    GV->setLinkage(llvm::GlobalVariable::WeakAnyLinkage);   
  else if (!getLangOptions().CPlusPlus && !CodeGenOpts.NoCommon &&
           !D->hasExternalStorage() && !D->getInit() &&
           !D->getAttr<SectionAttr>()) {
    GV->setLinkage(llvm::GlobalVariable::CommonLinkage);
    // common vars aren't constant even if declared const.
    GV->setConstant(false);
  } else
    GV->setLinkage(llvm::GlobalVariable::ExternalLinkage);

  SetCommonAttributes(D, GV);

  // Emit global variable debug information.
  if (CGDebugInfo *DI = getDebugInfo()) {
    DI->setLocation(D->getLocation());
    DI->EmitGlobalVariable(GV, D);
  }
}

/// ReplaceUsesOfNonProtoTypeWithRealFunction - This function is called when we
/// implement a function with no prototype, e.g. "int foo() {}".  If there are
/// existing call uses of the old function in the module, this adjusts them to
/// call the new function directly.
///
/// This is not just a cleanup: the always_inline pass requires direct calls to
/// functions to be able to inline them.  If there is a bitcast in the way, it
/// won't inline them.  Instcombine normally deletes these calls, but it isn't
/// run at -O0.
static void ReplaceUsesOfNonProtoTypeWithRealFunction(llvm::GlobalValue *Old,
                                                      llvm::Function *NewFn) {
  // If we're redefining a global as a function, don't transform it.
  llvm::Function *OldFn = dyn_cast<llvm::Function>(Old);
  if (OldFn == 0) return;

  const llvm::Type *NewRetTy = NewFn->getReturnType();
  llvm::SmallVector<llvm::Value*, 4> ArgList;

  for (llvm::Value::use_iterator UI = OldFn->use_begin(), E = OldFn->use_end();
       UI != E; ) {
    // TODO: Do invokes ever occur in C code?  If so, we should handle them too.
    unsigned OpNo = UI.getOperandNo();
    llvm::CallInst *CI = dyn_cast<llvm::CallInst>(*UI++);
    if (!CI || OpNo != 0) continue;

    // If the return types don't match exactly, and if the call isn't dead, then
    // we can't transform this call.
    if (CI->getType() != NewRetTy && !CI->use_empty())
      continue;

    // If the function was passed too few arguments, don't transform.  If extra
    // arguments were passed, we silently drop them.  If any of the types
    // mismatch, we don't transform.
    unsigned ArgNo = 0;
    bool DontTransform = false;
    for (llvm::Function::arg_iterator AI = NewFn->arg_begin(),
         E = NewFn->arg_end(); AI != E; ++AI, ++ArgNo) {
      if (CI->getNumOperands()-1 == ArgNo ||
          CI->getOperand(ArgNo+1)->getType() != AI->getType()) {
        DontTransform = true;
        break;
      }
    }
    if (DontTransform)
      continue;

    // Okay, we can transform this.  Create the new call instruction and copy
    // over the required information.
    ArgList.append(CI->op_begin()+1, CI->op_begin()+1+ArgNo);
    llvm::CallInst *NewCall = llvm::CallInst::Create(NewFn, ArgList.begin(),
                                                     ArgList.end(), "", CI);
    ArgList.clear();
    if (!NewCall->getType()->isVoidTy())
      NewCall->takeName(CI);
    NewCall->setAttributes(CI->getAttributes());
    NewCall->setCallingConv(CI->getCallingConv());

    // Finally, remove the old call, replacing any uses with the new one.
    if (!CI->use_empty())
      CI->replaceAllUsesWith(NewCall);

    // Copy debug location attached to CI.
    if (!CI->getDebugLoc().isUnknown())
      NewCall->setDebugLoc(CI->getDebugLoc());
    CI->eraseFromParent();
  }
}


void CodeGenModule::EmitGlobalFunctionDefinition(GlobalDecl GD) {
  const FunctionDecl *D = cast<FunctionDecl>(GD.getDecl());
  const llvm::FunctionType *Ty = getTypes().GetFunctionType(GD);
  getMangleContext().mangleInitDiscriminator();
  // Get or create the prototype for the function.
  llvm::Constant *Entry = GetAddrOfFunction(GD, Ty);

  // Strip off a bitcast if we got one back.
  if (llvm::ConstantExpr *CE = dyn_cast<llvm::ConstantExpr>(Entry)) {
    assert(CE->getOpcode() == llvm::Instruction::BitCast);
    Entry = CE->getOperand(0);
  }


  if (cast<llvm::GlobalValue>(Entry)->getType()->getElementType() != Ty) {
    llvm::GlobalValue *OldFn = cast<llvm::GlobalValue>(Entry);

    // If the types mismatch then we have to rewrite the definition.
    assert(OldFn->isDeclaration() &&
           "Shouldn't replace non-declaration");

    // F is the Function* for the one with the wrong type, we must make a new
    // Function* and update everything that used F (a declaration) with the new
    // Function* (which will be a definition).
    //
    // This happens if there is a prototype for a function
    // (e.g. "int f()") and then a definition of a different type
    // (e.g. "int f(int x)").  Move the old function aside so that it
    // doesn't interfere with GetAddrOfFunction.
    OldFn->setName(llvm::StringRef());
    llvm::Function *NewFn = cast<llvm::Function>(GetAddrOfFunction(GD, Ty));

    // If this is an implementation of a function without a prototype, try to
    // replace any existing uses of the function (which may be calls) with uses
    // of the new function
    if (D->getType()->isFunctionNoProtoType()) {
      ReplaceUsesOfNonProtoTypeWithRealFunction(OldFn, NewFn);
      OldFn->removeDeadConstantUsers();
    }

    // Replace uses of F with the Function we will endow with a body.
    if (!Entry->use_empty()) {
      llvm::Constant *NewPtrForOldDecl =
        llvm::ConstantExpr::getBitCast(NewFn, Entry->getType());
      Entry->replaceAllUsesWith(NewPtrForOldDecl);
    }

    // Ok, delete the old function now, which is dead.
    OldFn->eraseFromParent();

    Entry = NewFn;
  }

  llvm::Function *Fn = cast<llvm::Function>(Entry);

  CodeGenFunction(*this).GenerateCode(D, Fn);

  SetFunctionDefinitionAttributes(D, Fn);
  SetLLVMFunctionAttributesForDefinition(D, Fn);

  if (const ConstructorAttr *CA = D->getAttr<ConstructorAttr>())
    AddGlobalCtor(Fn, CA->getPriority());
  if (const DestructorAttr *DA = D->getAttr<DestructorAttr>())
    AddGlobalDtor(Fn, DA->getPriority());
}

void CodeGenModule::EmitAliasDefinition(GlobalDecl GD) {
  const ValueDecl *D = cast<ValueDecl>(GD.getDecl());
  const AliasAttr *AA = D->getAttr<AliasAttr>();
  assert(AA && "Not an alias?");

  MangleBuffer MangledName;
  getMangledName(MangledName, GD);

  // If there is a definition in the module, then it wins over the alias.
  // This is dubious, but allow it to be safe.  Just ignore the alias.
  llvm::GlobalValue *Entry = GetGlobalValue(MangledName);
  if (Entry && !Entry->isDeclaration())
    return;

  const llvm::Type *DeclTy = getTypes().ConvertTypeForMem(D->getType());

  // Create a reference to the named value.  This ensures that it is emitted
  // if a deferred decl.
  llvm::Constant *Aliasee;
  if (isa<llvm::FunctionType>(DeclTy))
    Aliasee = GetOrCreateLLVMFunction(AA->getAliasee(), DeclTy, GlobalDecl());
  else
    Aliasee = GetOrCreateLLVMGlobal(AA->getAliasee(),
                                    llvm::PointerType::getUnqual(DeclTy), 0);

  // Create the new alias itself, but don't set a name yet.
  llvm::GlobalValue *GA =
    new llvm::GlobalAlias(Aliasee->getType(),
                          llvm::Function::ExternalLinkage,
                          "", Aliasee, &getModule());

  if (Entry) {
    assert(Entry->isDeclaration());

    // If there is a declaration in the module, then we had an extern followed
    // by the alias, as in:
    //   extern int test6();
    //   ...
    //   int test6() __attribute__((alias("test7")));
    //
    // Remove it and replace uses of it with the alias.
    GA->takeName(Entry);

    Entry->replaceAllUsesWith(llvm::ConstantExpr::getBitCast(GA,
                                                          Entry->getType()));
    Entry->eraseFromParent();
  } else {
    GA->setName(MangledName.getString());
  }

  // Set attributes which are particular to an alias; this is a
  // specialization of the attributes which may be set on a global
  // variable/function.
  if (D->hasAttr<DLLExportAttr>()) {
    if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
      // The dllexport attribute is ignored for undefined symbols.
      if (FD->getBody())
        GA->setLinkage(llvm::Function::DLLExportLinkage);
    } else {
      GA->setLinkage(llvm::Function::DLLExportLinkage);
    }
  } else if (D->hasAttr<WeakAttr>() ||
             D->hasAttr<WeakRefAttr>() ||
             D->hasAttr<WeakImportAttr>()) {
    GA->setLinkage(llvm::Function::WeakAnyLinkage);
  }

  SetCommonAttributes(D, GA);
}

/// getBuiltinLibFunction - Given a builtin id for a function like
/// "__builtin_fabsf", return a Function* for "fabsf".
llvm::Value *CodeGenModule::getBuiltinLibFunction(const FunctionDecl *FD,
                                                  unsigned BuiltinID) {
  assert((Context.BuiltinInfo.isLibFunction(BuiltinID) ||
          Context.BuiltinInfo.isPredefinedLibFunction(BuiltinID)) &&
         "isn't a lib fn");

  // Get the name, skip over the __builtin_ prefix (if necessary).
  const char *Name = Context.BuiltinInfo.GetName(BuiltinID);
  if (Context.BuiltinInfo.isLibFunction(BuiltinID))
    Name += 10;

  const llvm::FunctionType *Ty =
    cast<llvm::FunctionType>(getTypes().ConvertType(FD->getType()));

  return GetOrCreateLLVMFunction(Name, Ty, GlobalDecl(FD));
}

llvm::Function *CodeGenModule::getIntrinsic(unsigned IID,const llvm::Type **Tys,
                                            unsigned NumTys) {
  return llvm::Intrinsic::getDeclaration(&getModule(),
                                         (llvm::Intrinsic::ID)IID, Tys, NumTys);
}


llvm::Function *CodeGenModule::getMemCpyFn(const llvm::Type *DestType,
                                           const llvm::Type *SrcType,
                                           const llvm::Type *SizeType) {
  const llvm::Type *ArgTypes[3] = {DestType, SrcType, SizeType };
  return getIntrinsic(llvm::Intrinsic::memcpy, ArgTypes, 3);
}

llvm::Function *CodeGenModule::getMemMoveFn(const llvm::Type *DestType,
                                            const llvm::Type *SrcType,
                                            const llvm::Type *SizeType) {
  const llvm::Type *ArgTypes[3] = {DestType, SrcType, SizeType };
  return getIntrinsic(llvm::Intrinsic::memmove, ArgTypes, 3);
}

llvm::Function *CodeGenModule::getMemSetFn(const llvm::Type *DestType,
                                           const llvm::Type *SizeType) {
  const llvm::Type *ArgTypes[2] = { DestType, SizeType };
  return getIntrinsic(llvm::Intrinsic::memset, ArgTypes, 2);
}

static llvm::StringMapEntry<llvm::Constant*> &
GetConstantCFStringEntry(llvm::StringMap<llvm::Constant*> &Map,
                         const StringLiteral *Literal,
                         bool TargetIsLSB,
                         bool &IsUTF16,
                         unsigned &StringLength) {
  unsigned NumBytes = Literal->getByteLength();

  // Check for simple case.
  if (!Literal->containsNonAsciiOrNull()) {
    StringLength = NumBytes;
    return Map.GetOrCreateValue(llvm::StringRef(Literal->getStrData(),
                                                StringLength));
  }

  // Otherwise, convert the UTF8 literals into a byte string.
  llvm::SmallVector<UTF16, 128> ToBuf(NumBytes);
  const UTF8 *FromPtr = (UTF8 *)Literal->getStrData();
  UTF16 *ToPtr = &ToBuf[0];

  ConversionResult Result = ConvertUTF8toUTF16(&FromPtr, FromPtr + NumBytes,
                                               &ToPtr, ToPtr + NumBytes,
                                               strictConversion);

  // Check for conversion failure.
  if (Result != conversionOK) {
    // FIXME: Have Sema::CheckObjCString() validate the UTF-8 string and remove
    // this duplicate code.
    assert(Result == sourceIllegal && "UTF-8 to UTF-16 conversion failed");
    StringLength = NumBytes;
    return Map.GetOrCreateValue(llvm::StringRef(Literal->getStrData(),
                                                StringLength));
  }

  // ConvertUTF8toUTF16 returns the length in ToPtr.
  StringLength = ToPtr - &ToBuf[0];

  // Render the UTF-16 string into a byte array and convert to the target byte
  // order.
  //
  // FIXME: This isn't something we should need to do here.
  llvm::SmallString<128> AsBytes;
  AsBytes.reserve(StringLength * 2);
  for (unsigned i = 0; i != StringLength; ++i) {
    unsigned short Val = ToBuf[i];
    if (TargetIsLSB) {
      AsBytes.push_back(Val & 0xFF);
      AsBytes.push_back(Val >> 8);
    } else {
      AsBytes.push_back(Val >> 8);
      AsBytes.push_back(Val & 0xFF);
    }
  }
  // Append one extra null character, the second is automatically added by our
  // caller.
  AsBytes.push_back(0);

  IsUTF16 = true;
  return Map.GetOrCreateValue(llvm::StringRef(AsBytes.data(), AsBytes.size()));
}

llvm::Constant *
CodeGenModule::GetAddrOfConstantCFString(const StringLiteral *Literal) {
  unsigned StringLength = 0;
  bool isUTF16 = false;
  llvm::StringMapEntry<llvm::Constant*> &Entry =
    GetConstantCFStringEntry(CFConstantStringMap, Literal,
                             getTargetData().isLittleEndian(),
                             isUTF16, StringLength);

  if (llvm::Constant *C = Entry.getValue())
    return C;

  llvm::Constant *Zero =
      llvm::Constant::getNullValue(llvm::Type::getInt32Ty(VMContext));
  llvm::Constant *Zeros[] = { Zero, Zero };

  // If we don't already have it, get __CFConstantStringClassReference.
  if (!CFConstantStringClassRef) {
    const llvm::Type *Ty = getTypes().ConvertType(getContext().IntTy);
    Ty = llvm::ArrayType::get(Ty, 0);
    llvm::Constant *GV = CreateRuntimeVariable(Ty,
                                           "__CFConstantStringClassReference");
    // Decay array -> ptr
    CFConstantStringClassRef =
      llvm::ConstantExpr::getGetElementPtr(GV, Zeros, 2);
  }

  QualType CFTy = getContext().getCFConstantStringType();

  const llvm::StructType *STy =
    cast<llvm::StructType>(getTypes().ConvertType(CFTy));

  std::vector<llvm::Constant*> Fields(4);

  // Class pointer.
  Fields[0] = CFConstantStringClassRef;

  // Flags.
  const llvm::Type *Ty = getTypes().ConvertType(getContext().UnsignedIntTy);
  Fields[1] = isUTF16 ? llvm::ConstantInt::get(Ty, 0x07d0) :
    llvm::ConstantInt::get(Ty, 0x07C8);

  // String pointer.
  llvm::Constant *C = llvm::ConstantArray::get(VMContext, Entry.getKey().str());

  llvm::GlobalValue::LinkageTypes Linkage;
  bool isConstant;
  if (isUTF16) {
    // FIXME: why do utf strings get "_" labels instead of "L" labels?
    Linkage = llvm::GlobalValue::InternalLinkage;
    // Note: -fwritable-strings doesn't make unicode CFStrings writable, but
    // does make plain ascii ones writable.
    isConstant = true;
  } else {
    Linkage = llvm::GlobalValue::PrivateLinkage;
    isConstant = !Features.WritableStrings;
  }
  
  llvm::GlobalVariable *GV =
    new llvm::GlobalVariable(getModule(), C->getType(), isConstant, Linkage, C,
                             ".str");
  if (isUTF16) {
    CharUnits Align = getContext().getTypeAlignInChars(getContext().ShortTy);
    GV->setAlignment(Align.getQuantity());
  }
  Fields[2] = llvm::ConstantExpr::getGetElementPtr(GV, Zeros, 2);

  // String length.
  Ty = getTypes().ConvertType(getContext().LongTy);
  Fields[3] = llvm::ConstantInt::get(Ty, StringLength);

  // The struct.
  C = llvm::ConstantStruct::get(STy, Fields);
  GV = new llvm::GlobalVariable(getModule(), C->getType(), true,
                                llvm::GlobalVariable::PrivateLinkage, C,
                                "_unnamed_cfstring_");
  if (const char *Sect = getContext().Target.getCFStringSection())
    GV->setSection(Sect);
  Entry.setValue(GV);

  return GV;
}

/// GetStringForStringLiteral - Return the appropriate bytes for a
/// string literal, properly padded to match the literal type.
std::string CodeGenModule::GetStringForStringLiteral(const StringLiteral *E) {
  const char *StrData = E->getStrData();
  unsigned Len = E->getByteLength();

  const ConstantArrayType *CAT =
    getContext().getAsConstantArrayType(E->getType());
  assert(CAT && "String isn't pointer or array!");

  // Resize the string to the right size.
  std::string Str(StrData, StrData+Len);
  uint64_t RealLen = CAT->getSize().getZExtValue();

  if (E->isWide())
    RealLen *= getContext().Target.getWCharWidth()/8;

  Str.resize(RealLen, '\0');

  return Str;
}

/// GetAddrOfConstantStringFromLiteral - Return a pointer to a
/// constant array for the given string literal.
llvm::Constant *
CodeGenModule::GetAddrOfConstantStringFromLiteral(const StringLiteral *S) {
  // FIXME: This can be more efficient.
  // FIXME: We shouldn't need to bitcast the constant in the wide string case.
  llvm::Constant *C = GetAddrOfConstantString(GetStringForStringLiteral(S));
  if (S->isWide()) {
    llvm::Type *DestTy =
        llvm::PointerType::getUnqual(getTypes().ConvertType(S->getType()));
    C = llvm::ConstantExpr::getBitCast(C, DestTy);
  }
  return C;
}

/// GetAddrOfConstantStringFromObjCEncode - Return a pointer to a constant
/// array for the given ObjCEncodeExpr node.
llvm::Constant *
CodeGenModule::GetAddrOfConstantStringFromObjCEncode(const ObjCEncodeExpr *E) {
  std::string Str;
  getContext().getObjCEncodingForType(E->getEncodedType(), Str);

  return GetAddrOfConstantCString(Str);
}


/// GenerateWritableString -- Creates storage for a string literal.
static llvm::Constant *GenerateStringLiteral(const std::string &str,
                                             bool constant,
                                             CodeGenModule &CGM,
                                             const char *GlobalName) {
  // Create Constant for this string literal. Don't add a '\0'.
  llvm::Constant *C =
      llvm::ConstantArray::get(CGM.getLLVMContext(), str, false);

  // Create a global variable for this string
  return new llvm::GlobalVariable(CGM.getModule(), C->getType(), constant,
                                  llvm::GlobalValue::PrivateLinkage,
                                  C, GlobalName);
}

/// GetAddrOfConstantString - Returns a pointer to a character array
/// containing the literal. This contents are exactly that of the
/// given string, i.e. it will not be null terminated automatically;
/// see GetAddrOfConstantCString. Note that whether the result is
/// actually a pointer to an LLVM constant depends on
/// Feature.WriteableStrings.
///
/// The result has pointer to array type.
llvm::Constant *CodeGenModule::GetAddrOfConstantString(const std::string &str,
                                                       const char *GlobalName) {
  bool IsConstant = !Features.WritableStrings;

  // Get the default prefix if a name wasn't specified.
  if (!GlobalName)
    GlobalName = ".str";

  // Don't share any string literals if strings aren't constant.
  if (!IsConstant)
    return GenerateStringLiteral(str, false, *this, GlobalName);

  llvm::StringMapEntry<llvm::Constant *> &Entry =
    ConstantStringMap.GetOrCreateValue(&str[0], &str[str.length()]);

  if (Entry.getValue())
    return Entry.getValue();

  // Create a global variable for this.
  llvm::Constant *C = GenerateStringLiteral(str, true, *this, GlobalName);
  Entry.setValue(C);
  return C;
}

/// GetAddrOfConstantCString - Returns a pointer to a character
/// array containing the literal and a terminating '\-'
/// character. The result has pointer to array type.
llvm::Constant *CodeGenModule::GetAddrOfConstantCString(const std::string &str,
                                                        const char *GlobalName){
  return GetAddrOfConstantString(str + '\0', GlobalName);
}

/// EmitObjCPropertyImplementations - Emit information for synthesized
/// properties for an implementation.
void CodeGenModule::EmitObjCPropertyImplementations(const
                                                    ObjCImplementationDecl *D) {
  for (ObjCImplementationDecl::propimpl_iterator
         i = D->propimpl_begin(), e = D->propimpl_end(); i != e; ++i) {
    ObjCPropertyImplDecl *PID = *i;

    // Dynamic is just for type-checking.
    if (PID->getPropertyImplementation() == ObjCPropertyImplDecl::Synthesize) {
      ObjCPropertyDecl *PD = PID->getPropertyDecl();

      // Determine which methods need to be implemented, some may have
      // been overridden. Note that ::isSynthesized is not the method
      // we want, that just indicates if the decl came from a
      // property. What we want to know is if the method is defined in
      // this implementation.
      if (!D->getInstanceMethod(PD->getGetterName()))
        CodeGenFunction(*this).GenerateObjCGetter(
                                 const_cast<ObjCImplementationDecl *>(D), PID);
      if (!PD->isReadOnly() &&
          !D->getInstanceMethod(PD->getSetterName()))
        CodeGenFunction(*this).GenerateObjCSetter(
                                 const_cast<ObjCImplementationDecl *>(D), PID);
    }
  }
}

/// EmitNamespace - Emit all declarations in a namespace.
void CodeGenModule::EmitNamespace(const NamespaceDecl *ND) {
  for (RecordDecl::decl_iterator I = ND->decls_begin(), E = ND->decls_end();
       I != E; ++I)
    EmitTopLevelDecl(*I);
}

// EmitLinkageSpec - Emit all declarations in a linkage spec.
void CodeGenModule::EmitLinkageSpec(const LinkageSpecDecl *LSD) {
  if (LSD->getLanguage() != LinkageSpecDecl::lang_c &&
      LSD->getLanguage() != LinkageSpecDecl::lang_cxx) {
    ErrorUnsupported(LSD, "linkage spec");
    return;
  }

  for (RecordDecl::decl_iterator I = LSD->decls_begin(), E = LSD->decls_end();
       I != E; ++I)
    EmitTopLevelDecl(*I);
}

/// EmitTopLevelDecl - Emit code for a single top level declaration.
void CodeGenModule::EmitTopLevelDecl(Decl *D) {
  // If an error has occurred, stop code generation, but continue
  // parsing and semantic analysis (to ensure all warnings and errors
  // are emitted).
  if (Diags.hasErrorOccurred())
    return;

  // Ignore dependent declarations.
  if (D->getDeclContext() && D->getDeclContext()->isDependentContext())
    return;

  switch (D->getKind()) {
  case Decl::CXXConversion:
  case Decl::CXXMethod:
  case Decl::Function:
    // Skip function templates
    if (cast<FunctionDecl>(D)->getDescribedFunctionTemplate())
      return;

    EmitGlobal(cast<FunctionDecl>(D));
    break;
      
  case Decl::Var:
    EmitGlobal(cast<VarDecl>(D));
    break;

  // C++ Decls
  case Decl::Namespace:
    EmitNamespace(cast<NamespaceDecl>(D));
    break;
    // No code generation needed.
  case Decl::UsingShadow:
  case Decl::Using:
  case Decl::UsingDirective:
  case Decl::ClassTemplate:
  case Decl::FunctionTemplate:
  case Decl::NamespaceAlias:
    break;
  case Decl::CXXConstructor:
    // Skip function templates
    if (cast<FunctionDecl>(D)->getDescribedFunctionTemplate())
      return;
      
    EmitCXXConstructors(cast<CXXConstructorDecl>(D));
    break;
  case Decl::CXXDestructor:
    EmitCXXDestructors(cast<CXXDestructorDecl>(D));
    break;

  case Decl::StaticAssert:
    // Nothing to do.
    break;

  // Objective-C Decls

  // Forward declarations, no (immediate) code generation.
  case Decl::ObjCClass:
  case Decl::ObjCForwardProtocol:
  case Decl::ObjCCategory:
  case Decl::ObjCInterface:
    break;

  case Decl::ObjCProtocol:
    Runtime->GenerateProtocol(cast<ObjCProtocolDecl>(D));
    break;

  case Decl::ObjCCategoryImpl:
    // Categories have properties but don't support synthesize so we
    // can ignore them here.
    Runtime->GenerateCategory(cast<ObjCCategoryImplDecl>(D));
    break;

  case Decl::ObjCImplementation: {
    ObjCImplementationDecl *OMD = cast<ObjCImplementationDecl>(D);
    EmitObjCPropertyImplementations(OMD);
    Runtime->GenerateClass(OMD);
    break;
  }
  case Decl::ObjCMethod: {
    ObjCMethodDecl *OMD = cast<ObjCMethodDecl>(D);
    // If this is not a prototype, emit the body.
    if (OMD->getBody())
      CodeGenFunction(*this).GenerateObjCMethod(OMD);
    break;
  }
  case Decl::ObjCCompatibleAlias:
    // compatibility-alias is a directive and has no code gen.
    break;

  case Decl::LinkageSpec:
    EmitLinkageSpec(cast<LinkageSpecDecl>(D));
    break;

  case Decl::FileScopeAsm: {
    FileScopeAsmDecl *AD = cast<FileScopeAsmDecl>(D);
    llvm::StringRef AsmString = AD->getAsmString()->getString();

    const std::string &S = getModule().getModuleInlineAsm();
    if (S.empty())
      getModule().setModuleInlineAsm(AsmString);
    else
      getModule().setModuleInlineAsm(S + '\n' + AsmString.str());
    break;
  }

  default:
    // Make sure we handled everything we should, every other kind is a
    // non-top-level decl.  FIXME: Would be nice to have an isTopLevelDeclKind
    // function. Need to recode Decl::Kind to do that easily.
    assert(isa<TypeDecl>(D) && "Unsupported decl kind");
  }
}
