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
#include "clang/Frontend/CompileOptions.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclCXX.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/ConvertUTF.h"
#include "llvm/CallingConv.h"
#include "llvm/Module.h"
#include "llvm/Intrinsics.h"
#include "llvm/Target/TargetData.h"
using namespace clang;
using namespace CodeGen;


CodeGenModule::CodeGenModule(ASTContext &C, const CompileOptions &compileOpts,
                             llvm::Module &M, const llvm::TargetData &TD,
                             Diagnostic &diags)
  : BlockModule(C, M, TD, Types, *this), Context(C),
    Features(C.getLangOptions()), CompileOpts(compileOpts), TheModule(M),
    TheTargetData(TD), Diags(diags), Types(C, M, TD), Runtime(0),
    MemCpyFn(0), MemMoveFn(0), MemSetFn(0), CFConstantStringClassRef(0) {

  if (!Features.ObjC1)
    Runtime = 0;
  else if (!Features.NeXTRuntime)
    Runtime = CreateGNUObjCRuntime(*this);
  else if (Features.ObjCNonFragileABI)
    Runtime = CreateMacNonFragileABIObjCRuntime(*this);
  else
    Runtime = CreateMacObjCRuntime(*this);

  // If debug info generation is enabled, create the CGDebugInfo object.
  DebugInfo = CompileOpts.DebugInfo ? new CGDebugInfo(this) : 0;
}

CodeGenModule::~CodeGenModule() {
  delete Runtime;
  delete DebugInfo;
}

void CodeGenModule::Release() {
  EmitDeferred();
  if (Runtime)
    if (llvm::Function *ObjCInitFunction = Runtime->ModuleInitFunction())
      AddGlobalCtor(ObjCInitFunction);
  EmitCtorList(GlobalCtors, "llvm.global_ctors");
  EmitCtorList(GlobalDtors, "llvm.global_dtors");
  EmitAnnotations();
  EmitLLVMUsed();
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

const char *CodeGenModule::getMangledName(const GlobalDecl &GD) {
  const NamedDecl *ND = GD.getDecl();
  
  if (const CXXConstructorDecl *D = dyn_cast<CXXConstructorDecl>(ND))
    return getMangledCXXCtorName(D, GD.getCtorType());
  if (const CXXDestructorDecl *D = dyn_cast<CXXDestructorDecl>(ND))
    return getMangledCXXDtorName(D, GD.getDtorType());
  
  return getMangledName(ND);
}

/// \brief Retrieves the mangled name for the given declaration.
///
/// If the given declaration requires a mangled name, returns an
/// const char* containing the mangled name.  Otherwise, returns
/// the unmangled name.
///
const char *CodeGenModule::getMangledName(const NamedDecl *ND) {
  // In C, functions with no attributes never need to be mangled. Fastpath them.
  if (!getLangOptions().CPlusPlus && !ND->hasAttrs()) {
    assert(ND->getIdentifier() && "Attempt to mangle unnamed decl.");
    return ND->getNameAsCString();
  }
    
  llvm::SmallString<256> Name;
  llvm::raw_svector_ostream Out(Name);
  if (!mangleName(ND, Context, Out)) {
    assert(ND->getIdentifier() && "Attempt to mangle unnamed decl.");
    return ND->getNameAsCString();
  }

  Name += '\0';
  return UniqueMangledName(Name.begin(), Name.end());
}

const char *CodeGenModule::UniqueMangledName(const char *NameStart,
                                             const char *NameEnd) {
  assert(*(NameEnd - 1) == '\0' && "Mangled name must be null terminated!");
  
  return MangledNames.GetOrCreateValue(NameStart, NameEnd).getKeyData();
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
    llvm::FunctionType::get(llvm::Type::VoidTy, 
                            std::vector<const llvm::Type*>(),
                            false);
  llvm::Type *CtorPFTy = llvm::PointerType::getUnqual(CtorFTy);

  // Get the type of a ctor entry, { i32, void ()* }.
  llvm::StructType* CtorStructTy = 
    llvm::StructType::get(llvm::Type::Int32Ty, 
                          llvm::PointerType::getUnqual(CtorFTy), NULL);

  // Construct the constructor and destructor arrays.
  std::vector<llvm::Constant*> Ctors;
  for (CtorList::const_iterator I = Fns.begin(), E = Fns.end(); I != E; ++I) {
    std::vector<llvm::Constant*> S;
    S.push_back(llvm::ConstantInt::get(llvm::Type::Int32Ty, I->second, false));
    S.push_back(llvm::ConstantExpr::getBitCast(I->first, CtorPFTy));
    Ctors.push_back(llvm::ConstantStruct::get(CtorStructTy, S));
  }

  if (!Ctors.empty()) {
    llvm::ArrayType *AT = llvm::ArrayType::get(CtorStructTy, Ctors.size());
    new llvm::GlobalVariable(AT, false,
                             llvm::GlobalValue::AppendingLinkage,
                             llvm::ConstantArray::get(AT, Ctors),
                             GlobalName, 
                             &TheModule);
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
  new llvm::GlobalVariable(Array->getType(), false,  
                           llvm::GlobalValue::AppendingLinkage, Array, 
                           "llvm.global.annotations", &TheModule);
  gv->setSection("llvm.metadata");
}

static CodeGenModule::GVALinkage
GetLinkageForFunction(const FunctionDecl *FD, const LangOptions &Features) {
  // "static" functions get internal linkage.
  if (FD->getStorageClass() == FunctionDecl::Static)
    return CodeGenModule::GVA_Internal;

  if (!FD->isInline())
    return CodeGenModule::GVA_StrongExternal;
  
  // If the inline function explicitly has the GNU inline attribute on it, or if
  // this is C89 mode, we use to GNU semantics.
  if (!Features.C99 && !Features.CPlusPlus) {
    // extern inline in GNU mode is like C99 inline.
    if (FD->getStorageClass() == FunctionDecl::Extern)
      return CodeGenModule::GVA_C99Inline;
    // Normal inline is a strong symbol.
    return CodeGenModule::GVA_StrongExternal;
  } else if (FD->hasActiveGNUInlineAttribute()) {
    // GCC in C99 mode seems to use a different decision-making
    // process for extern inline, which factors in previous
    // declarations.
    if (FD->isExternGNUInline())
      return CodeGenModule::GVA_C99Inline;
    // Normal inline is a strong symbol.
    return CodeGenModule::GVA_StrongExternal;
  }

  // The definition of inline changes based on the language.  Note that we
  // have already handled "static inline" above, with the GVA_Internal case.
  if (Features.CPlusPlus)  // inline and extern inline.
    return CodeGenModule::GVA_CXXInline;
  
  assert(Features.C99 && "Must be in C99 mode if not in C89 or C++ mode");
  if (FD->isC99InlineDefinition())
    return CodeGenModule::GVA_C99Inline;

  return CodeGenModule::GVA_StrongExternal;
}

/// SetFunctionDefinitionAttributes - Set attributes for a global.
///
/// FIXME: This is currently only done for aliases and functions, but
/// not for variables (these details are set in
/// EmitGlobalVarDefinition for variables).
void CodeGenModule::SetFunctionDefinitionAttributes(const FunctionDecl *D,
                                                    llvm::GlobalValue *GV) {
  GVALinkage Linkage = GetLinkageForFunction(D, Features);

  if (Linkage == GVA_Internal) {
    GV->setLinkage(llvm::Function::InternalLinkage);
  } else if (D->hasAttr<DLLExportAttr>()) {
    GV->setLinkage(llvm::Function::DLLExportLinkage);
  } else if (D->hasAttr<WeakAttr>() || D->hasAttr<WeakImportAttr>()) {
    GV->setLinkage(llvm::Function::WeakAnyLinkage);
  } else if (Linkage == GVA_C99Inline) {
    // In C99 mode, 'inline' functions are guaranteed to have a strong
    // definition somewhere else, so we can use available_externally linkage.
    GV->setLinkage(llvm::Function::AvailableExternallyLinkage);
  } else if (Linkage == GVA_CXXInline) {
    // In C++, the compiler has to emit a definition in every translation unit
    // that references the function.  We should use linkonce_odr because
    // a) if all references in this translation unit are optimized away, we
    // don't need to codegen it.  b) if the function persists, it needs to be
    // merged with other definitions. c) C++ has the ODR, so we know the
    // definition is dependable.
    GV->setLinkage(llvm::Function::LinkOnceODRLinkage);
  } else {
    assert(Linkage == GVA_StrongExternal);
    // Otherwise, we have strong external linkage.
    GV->setLinkage(llvm::Function::ExternalLinkage);
  }

  SetCommonAttributes(D, GV);
}

void CodeGenModule::SetLLVMFunctionAttributes(const Decl *D,
                                              const CGFunctionInfo &Info, 
                                              llvm::Function *F) {
  AttributeListType AttributeList;
  ConstructAttributeList(Info, D, AttributeList);

  F->setAttributes(llvm::AttrListPtr::get(AttributeList.begin(),
                                        AttributeList.size()));

  // Set the appropriate calling convention for the Function.
  if (D->hasAttr<FastCallAttr>())
    F->setCallingConv(llvm::CallingConv::X86_FastCall);

  if (D->hasAttr<StdCallAttr>())
    F->setCallingConv(llvm::CallingConv::X86_StdCall);
}

void CodeGenModule::SetLLVMFunctionAttributesForDefinition(const Decl *D,
                                                           llvm::Function *F) {
  if (!Features.Exceptions && !Features.ObjCNonFragileABI)
    F->addFnAttr(llvm::Attribute::NoUnwind);  

  if (D->hasAttr<AlwaysInlineAttr>())
    F->addFnAttr(llvm::Attribute::AlwaysInline);
  
  if (D->hasAttr<NoinlineAttr>())
    F->addFnAttr(llvm::Attribute::NoInline);
}

void CodeGenModule::SetCommonAttributes(const Decl *D, 
                                        llvm::GlobalValue *GV) {
  setGlobalVisibility(GV, D);

  if (D->hasAttr<UsedAttr>())
    AddUsedGlobal(GV);

  if (const SectionAttr *SA = D->getAttr<SectionAttr>())
    GV->setSection(SA->getName());
}

void CodeGenModule::SetInternalFunctionAttributes(const Decl *D,
                                                  llvm::Function *F,
                                                  const CGFunctionInfo &FI) {
  SetLLVMFunctionAttributes(D, FI, F);
  SetLLVMFunctionAttributesForDefinition(D, F);

  F->setLinkage(llvm::Function::InternalLinkage);

  SetCommonAttributes(D, F);
}

void CodeGenModule::SetFunctionAttributes(const FunctionDecl *FD,
                                          llvm::Function *F) {
  SetLLVMFunctionAttributes(FD, getTypes().getFunctionInfo(FD), F);
  
  // Only a few attributes are set on declarations; these may later be
  // overridden by a definition.
  
  if (FD->hasAttr<DLLImportAttr>()) {
    F->setLinkage(llvm::Function::DLLImportLinkage);
  } else if (FD->hasAttr<WeakAttr>() || FD->hasAttr<WeakImportAttr>()) {
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

  llvm::Type *i8PTy = llvm::PointerType::getUnqual(llvm::Type::Int8Ty);
  llvm::ArrayType *ATy = llvm::ArrayType::get(i8PTy, LLVMUsed.size());
  
  // Convert LLVMUsed to what ConstantArray needs.
  std::vector<llvm::Constant*> UsedArray;
  UsedArray.resize(LLVMUsed.size());
  for (unsigned i = 0, e = LLVMUsed.size(); i != e; ++i) {
    UsedArray[i] = 
     llvm::ConstantExpr::getBitCast(cast<llvm::Constant>(&*LLVMUsed[i]), i8PTy);
  }
  
  llvm::GlobalVariable *GV = 
    new llvm::GlobalVariable(ATy, false, 
                             llvm::GlobalValue::AppendingLinkage,
                             llvm::ConstantArray::get(ATy, UsedArray),
                             "llvm.used", &getModule());

  GV->setSection("llvm.metadata");
}

void CodeGenModule::EmitDeferred() {
  // Emit code for any potentially referenced deferred decls.  Since a
  // previously unused static decl may become used during the generation of code
  // for a static function, iterate until no  changes are made.
  while (!DeferredDeclsToEmit.empty()) {
    GlobalDecl D = DeferredDeclsToEmit.back();
    DeferredDeclsToEmit.pop_back();

    // The mangled name for the decl must have been emitted in GlobalDeclMap.
    // Look it up to see if it was defined with a stronger definition (e.g. an
    // extern inline function with a strong function redefinition).  If so,
    // just ignore the deferred decl.
    llvm::GlobalValue *CGRef = GlobalDeclMap[getMangledName(D)];
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
  const llvm::Type *SBP = llvm::PointerType::getUnqual(llvm::Type::Int8Ty);
  llvm::Constant *anno = llvm::ConstantArray::get(AA->getAnnotation(), true);
  llvm::Constant *unit = llvm::ConstantArray::get(M->getModuleIdentifier(),
                                                  true);

  // Get the two global values corresponding to the ConstantArrays we just
  // created to hold the bytes of the strings.
  const char *StringPrefix = getContext().Target.getStringSymbolPrefix(true);
  llvm::GlobalValue *annoGV = 
  new llvm::GlobalVariable(anno->getType(), false,
                           llvm::GlobalValue::InternalLinkage, anno,
                           GV->getName() + StringPrefix, M);
  // translation unit name string, emitted into the llvm.metadata section.
  llvm::GlobalValue *unitGV =
  new llvm::GlobalVariable(unit->getType(), false,
                           llvm::GlobalValue::InternalLinkage, unit, 
                           StringPrefix, M);

  // Create the ConstantStruct for the global annotation.
  llvm::Constant *Fields[4] = {
    llvm::ConstantExpr::getBitCast(GV, SBP),
    llvm::ConstantExpr::getBitCast(annoGV, SBP),
    llvm::ConstantExpr::getBitCast(unitGV, SBP),
    llvm::ConstantInt::get(llvm::Type::Int32Ty, LineNo)
  };
  return llvm::ConstantStruct::get(Fields, 4, false);
}

bool CodeGenModule::MayDeferGeneration(const ValueDecl *Global) {
  // Never defer when EmitAllDecls is specified or the decl has
  // attribute used.
  if (Features.EmitAllDecls || Global->hasAttr<UsedAttr>())
    return false;

  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(Global)) {
    // Constructors and destructors should never be deferred.
    if (FD->hasAttr<ConstructorAttr>() || FD->hasAttr<DestructorAttr>())
      return false;

    GVALinkage Linkage = GetLinkageForFunction(FD, Features);
    
    // static, static inline, always_inline, and extern inline functions can
    // always be deferred.  Normal inline functions can be deferred in C99/C++.
    if (Linkage == GVA_Internal || Linkage == GVA_C99Inline ||
        Linkage == GVA_CXXInline)
      return true;
    return false;
  }
  
  const VarDecl *VD = cast<VarDecl>(Global);
  assert(VD->isFileVarDecl() && "Invalid decl");

  return VD->getStorageClass() == VarDecl::Static;
}

void CodeGenModule::EmitGlobal(const GlobalDecl &GD) {
  const ValueDecl *Global = GD.getDecl();
  
  // If this is an alias definition (which otherwise looks like a declaration)
  // emit it now.
  if (Global->hasAttr<AliasAttr>())
    return EmitAliasDefinition(Global);

  // Ignore declarations, they will be emitted on their first use.
  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(Global)) {
    // Forward declarations are emitted lazily on first use.
    if (!FD->isThisDeclarationADefinition())
      return;
  } else {
    const VarDecl *VD = cast<VarDecl>(Global);
    assert(VD->isFileVarDecl() && "Cannot emit local var decl as global.");

    // In C++, if this is marked "extern", defer code generation.
    if (getLangOptions().CPlusPlus && 
        VD->getStorageClass() == VarDecl::Extern && !VD->getInit())
      return;

    // In C, if this isn't a definition, defer code generation.
    if (!getLangOptions().CPlusPlus && !VD->getInit())
      return;
  }

  // Defer code generation when possible if this is a static definition, inline
  // function etc.  These we only want to emit if they are used.
  if (MayDeferGeneration(Global)) {
    // If the value has already been used, add it directly to the
    // DeferredDeclsToEmit list.
    const char *MangledName = getMangledName(GD);
    if (GlobalDeclMap.count(MangledName))
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

void CodeGenModule::EmitGlobalDefinition(const GlobalDecl &GD) {
  const ValueDecl *D = GD.getDecl();
  
  if (const CXXConstructorDecl *CD = dyn_cast<CXXConstructorDecl>(D))
    EmitCXXConstructor(CD, GD.getCtorType());
  else if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D))
    EmitGlobalFunctionDefinition(FD);
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
llvm::Constant *CodeGenModule::GetOrCreateLLVMFunction(const char *MangledName,
                                                       const llvm::Type *Ty,
                                                       const FunctionDecl *D) {
  // Lookup the entry, lazily creating it if necessary.
  llvm::GlobalValue *&Entry = GlobalDeclMap[MangledName];
  if (Entry) {
    if (Entry->getType()->getElementType() == Ty)
      return Entry;
    
    // Make sure the result is of the correct type.
    const llvm::Type *PTy = llvm::PointerType::getUnqual(Ty);
    return llvm::ConstantExpr::getBitCast(Entry, PTy);
  }
  
  // This is the first use or definition of a mangled name.  If there is a
  // deferred decl with this name, remember that we need to emit it at the end
  // of the file.
  llvm::DenseMap<const char*, GlobalDecl>::iterator DDI = 
  DeferredDecls.find(MangledName);
  if (DDI != DeferredDecls.end()) {
    // Move the potentially referenced deferred decl to the DeferredDeclsToEmit
    // list, and remove it from DeferredDecls (since we don't need it anymore).
    DeferredDeclsToEmit.push_back(DDI->second);
    DeferredDecls.erase(DDI);
  }
  
  // This function doesn't have a complete type (for example, the return
  // type is an incomplete struct). Use a fake type instead, and make
  // sure not to try to set attributes.
  bool ShouldSetAttributes = true;
  if (!isa<llvm::FunctionType>(Ty)) {
    Ty = llvm::FunctionType::get(llvm::Type::VoidTy,
                                 std::vector<const llvm::Type*>(), false);
    ShouldSetAttributes = false;
  }
  llvm::Function *F = llvm::Function::Create(cast<llvm::FunctionType>(Ty), 
                                             llvm::Function::ExternalLinkage,
                                             "", &getModule());
  F->setName(MangledName);
  if (D && ShouldSetAttributes)
    SetFunctionAttributes(D, F);
  Entry = F;
  return F;
}

/// GetAddrOfFunction - Return the address of the given function.  If Ty is
/// non-null, then this function will use the specified type if it has to
/// create it (this occurs when we see a definition of the function).
llvm::Constant *CodeGenModule::GetAddrOfFunction(const FunctionDecl *D,
                                                 const llvm::Type *Ty) {
  // If there was no specific requested type, just convert it now.
  if (!Ty)
    Ty = getTypes().ConvertType(D->getType());
  return GetOrCreateLLVMFunction(getMangledName(D), Ty, D);
}

/// CreateRuntimeFunction - Create a new runtime function with the specified
/// type and name.
llvm::Constant *
CodeGenModule::CreateRuntimeFunction(const llvm::FunctionType *FTy,
                                     const char *Name) {
  // Convert Name to be a uniqued string from the IdentifierInfo table.
  Name = getContext().Idents.get(Name).getName();
  return GetOrCreateLLVMFunction(Name, FTy, 0);
}

/// GetOrCreateLLVMGlobal - If the specified mangled name is not in the module,
/// create and return an llvm GlobalVariable with the specified type.  If there
/// is something in the module with the specified name, return it potentially
/// bitcasted to the right type.
///
/// If D is non-null, it specifies a decl that correspond to this.  This is used
/// to set the attributes on the global when it is first created.
llvm::Constant *CodeGenModule::GetOrCreateLLVMGlobal(const char *MangledName,
                                                     const llvm::PointerType*Ty,
                                                     const VarDecl *D) {
  // Lookup the entry, lazily creating it if necessary.
  llvm::GlobalValue *&Entry = GlobalDeclMap[MangledName];
  if (Entry) {
    if (Entry->getType() == Ty)
      return Entry;
        
    // Make sure the result is of the correct type.
    return llvm::ConstantExpr::getBitCast(Entry, Ty);
  }
  
  // This is the first use or definition of a mangled name.  If there is a
  // deferred decl with this name, remember that we need to emit it at the end
  // of the file.
  llvm::DenseMap<const char*, GlobalDecl>::iterator DDI = 
    DeferredDecls.find(MangledName);
  if (DDI != DeferredDecls.end()) {
    // Move the potentially referenced deferred decl to the DeferredDeclsToEmit
    // list, and remove it from DeferredDecls (since we don't need it anymore).
    DeferredDeclsToEmit.push_back(DDI->second);
    DeferredDecls.erase(DDI);
  }
  
  llvm::GlobalVariable *GV = 
    new llvm::GlobalVariable(Ty->getElementType(), false, 
                             llvm::GlobalValue::ExternalLinkage,
                             0, "", &getModule(), 
                             false, Ty->getAddressSpace());
  GV->setName(MangledName);

  // Handle things which are present even on external declarations.
  if (D) {
    // FIXME: This code is overly simple and should be merged with
    // other global handling.
    GV->setConstant(D->getType().isConstant(Context));

    // FIXME: Merge with other attribute handling code.
    if (D->getStorageClass() == VarDecl::PrivateExtern)
      GV->setVisibility(llvm::GlobalValue::HiddenVisibility);

    if (D->hasAttr<WeakAttr>() || D->hasAttr<WeakImportAttr>())
      GV->setLinkage(llvm::GlobalValue::ExternalWeakLinkage);

    GV->setThreadLocal(D->isThreadSpecified());
  }
  
  return Entry = GV;
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
  return GetOrCreateLLVMGlobal(getMangledName(D), PTy, D);
}

/// CreateRuntimeVariable - Create a new runtime global variable with the
/// specified type and name.
llvm::Constant *
CodeGenModule::CreateRuntimeVariable(const llvm::Type *Ty,
                                     const char *Name) {
  // Convert Name to be a uniqued string from the IdentifierInfo table.
  Name = getContext().Idents.get(Name).getName();
  return GetOrCreateLLVMGlobal(Name, llvm::PointerType::getUnqual(Ty), 0);
}

void CodeGenModule::EmitTentativeDefinition(const VarDecl *D) {
  assert(!D->getInit() && "Cannot emit definite definitions here!");

  if (MayDeferGeneration(D)) {
    // If we have not seen a reference to this variable yet, place it
    // into the deferred declarations table to be emitted if needed
    // later.
    const char *MangledName = getMangledName(D);
    if (GlobalDeclMap.count(MangledName) == 0) {
      DeferredDecls[MangledName] = GlobalDecl(D);
      return;
    }
  }

  // The tentative definition is the only definition.
  EmitGlobalVarDefinition(D);
}

void CodeGenModule::EmitGlobalVarDefinition(const VarDecl *D) {
  llvm::Constant *Init = 0;
  QualType ASTTy = D->getType();
  
  if (D->getInit() == 0) {
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
    Init = llvm::Constant::getNullValue(getTypes().ConvertTypeForMem(ASTTy));
  } else {
    Init = EmitConstantExpr(D->getInit(), D->getType());
    if (!Init) {
      ErrorUnsupported(D, "static initializer");
      QualType T = D->getInit()->getType();
      Init = llvm::UndefValue::get(getTypes().ConvertType(T));
    }
  }

  const llvm::Type* InitType = Init->getType();
  llvm::Constant *Entry = GetAddrOfGlobalVar(D, InitType);
  
  // Strip off a bitcast if we got one back.
  if (llvm::ConstantExpr *CE = dyn_cast<llvm::ConstantExpr>(Entry)) {
    assert(CE->getOpcode() == llvm::Instruction::BitCast);
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
    
    // Remove the old entry from GlobalDeclMap so that we'll create a new one.
    GlobalDeclMap.erase(getMangledName(D));

    // Make a new global with the correct type, this is now guaranteed to work.
    GV = cast<llvm::GlobalVariable>(GetAddrOfGlobalVar(D, InitType));
    GV->takeName(cast<llvm::GlobalValue>(Entry));

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
  GV->setConstant(D->getType().isConstant(Context));
  GV->setAlignment(getContext().getDeclAlignInBytes(D));

  // Set the llvm linkage type as appropriate.
  if (D->getStorageClass() == VarDecl::Static)
    GV->setLinkage(llvm::Function::InternalLinkage);
  else if (D->hasAttr<DLLImportAttr>())
    GV->setLinkage(llvm::Function::DLLImportLinkage);
  else if (D->hasAttr<DLLExportAttr>())
    GV->setLinkage(llvm::Function::DLLExportLinkage);
  else if (D->hasAttr<WeakAttr>() || D->hasAttr<WeakImportAttr>())
    GV->setLinkage(llvm::GlobalVariable::WeakAnyLinkage);
  else if (!CompileOpts.NoCommon &&
           (!D->hasExternalStorage() && !D->getInit()))
    GV->setLinkage(llvm::GlobalVariable::CommonLinkage);
  else
    GV->setLinkage(llvm::GlobalVariable::ExternalLinkage);

  SetCommonAttributes(D, GV);

  // Emit global variable debug information.
  if (CGDebugInfo *DI = getDebugInfo()) {
    DI->setLocation(D->getLocation());
    DI->EmitGlobalVariable(GV, D);
  }
}


void CodeGenModule::EmitGlobalFunctionDefinition(const FunctionDecl *D) {
  const llvm::FunctionType *Ty;

  if (const CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(D)) {
    bool isVariadic = D->getType()->getAsFunctionProtoType()->isVariadic();
    
    Ty = getTypes().GetFunctionType(getTypes().getFunctionInfo(MD), isVariadic);
  } else {
    Ty = cast<llvm::FunctionType>(getTypes().ConvertType(D->getType()));
    
    // As a special case, make sure that definitions of K&R function
    // "type foo()" aren't declared as varargs (which forces the backend
    // to do unnecessary work).
    if (D->getType()->isFunctionNoProtoType()) {
      assert(Ty->isVarArg() && "Didn't lower type as expected");
      // Due to stret, the lowered function could have arguments. 
      // Just create the same type as was lowered by ConvertType 
      // but strip off the varargs bit.
      std::vector<const llvm::Type*> Args(Ty->param_begin(), Ty->param_end());
      Ty = llvm::FunctionType::get(Ty->getReturnType(), Args, false);
    }
  }

  // Get or create the prototype for teh function.
  llvm::Constant *Entry = GetAddrOfFunction(D, Ty);
  
  // Strip off a bitcast if we got one back.
  if (llvm::ConstantExpr *CE = dyn_cast<llvm::ConstantExpr>(Entry)) {
    assert(CE->getOpcode() == llvm::Instruction::BitCast);
    Entry = CE->getOperand(0);
  }
  
  
  if (cast<llvm::GlobalValue>(Entry)->getType()->getElementType() != Ty) {
    // If the types mismatch then we have to rewrite the definition.
    assert(cast<llvm::GlobalValue>(Entry)->isDeclaration() &&
           "Shouldn't replace non-declaration");

    // F is the Function* for the one with the wrong type, we must make a new
    // Function* and update everything that used F (a declaration) with the new
    // Function* (which will be a definition).
    //
    // This happens if there is a prototype for a function
    // (e.g. "int f()") and then a definition of a different type
    // (e.g. "int f(int x)").  Start by making a new function of the
    // correct type, RAUW, then steal the name.
    GlobalDeclMap.erase(getMangledName(D));
    llvm::Function *NewFn = cast<llvm::Function>(GetAddrOfFunction(D, Ty));
    NewFn->takeName(cast<llvm::GlobalValue>(Entry));
    
    // Replace uses of F with the Function we will endow with a body.
    llvm::Constant *NewPtrForOldDecl = 
      llvm::ConstantExpr::getBitCast(NewFn, Entry->getType());
    Entry->replaceAllUsesWith(NewPtrForOldDecl);
    
    // Ok, delete the old function now, which is dead.
    cast<llvm::GlobalValue>(Entry)->eraseFromParent();
    
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

void CodeGenModule::EmitAliasDefinition(const ValueDecl *D) {
  const AliasAttr *AA = D->getAttr<AliasAttr>();
  assert(AA && "Not an alias?");

  const llvm::Type *DeclTy = getTypes().ConvertTypeForMem(D->getType());
  
  // Unique the name through the identifier table.
  const char *AliaseeName = AA->getAliasee().c_str();
  AliaseeName = getContext().Idents.get(AliaseeName).getName();

  // Create a reference to the named value.  This ensures that it is emitted
  // if a deferred decl.
  llvm::Constant *Aliasee;
  if (isa<llvm::FunctionType>(DeclTy))
    Aliasee = GetOrCreateLLVMFunction(AliaseeName, DeclTy, 0);
  else
    Aliasee = GetOrCreateLLVMGlobal(AliaseeName,
                                    llvm::PointerType::getUnqual(DeclTy), 0);

  // Create the new alias itself, but don't set a name yet.
  llvm::GlobalValue *GA = 
    new llvm::GlobalAlias(Aliasee->getType(),
                          llvm::Function::ExternalLinkage,
                          "", Aliasee, &getModule());
  
  // See if there is already something with the alias' name in the module.
  const char *MangledName = getMangledName(D);
  llvm::GlobalValue *&Entry = GlobalDeclMap[MangledName];
  
  if (Entry && !Entry->isDeclaration()) {
    // If there is a definition in the module, then it wins over the alias.
    // This is dubious, but allow it to be safe.  Just ignore the alias.
    GA->eraseFromParent();
    return;
  }
  
  if (Entry) {
    // If there is a declaration in the module, then we had an extern followed
    // by the alias, as in:
    //   extern int test6();
    //   ...
    //   int test6() __attribute__((alias("test7")));
    //
    // Remove it and replace uses of it with the alias.
    
    Entry->replaceAllUsesWith(llvm::ConstantExpr::getBitCast(GA,
                                                          Entry->getType()));
    Entry->eraseFromParent();
  }
  
  // Now we know that there is no conflict, set the name.
  Entry = GA;
  GA->setName(MangledName);

  // Set attributes which are particular to an alias; this is a
  // specialization of the attributes which may be set on a global
  // variable/function.
  if (D->hasAttr<DLLExportAttr>()) {
    if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
      // The dllexport attribute is ignored for undefined symbols.
      if (FD->getBody(getContext()))
        GA->setLinkage(llvm::Function::DLLExportLinkage);
    } else {
      GA->setLinkage(llvm::Function::DLLExportLinkage);
    }
  } else if (D->hasAttr<WeakAttr>() || D->hasAttr<WeakImportAttr>()) {
    GA->setLinkage(llvm::Function::WeakAnyLinkage);
  }

  SetCommonAttributes(D, GA);
}

/// getBuiltinLibFunction - Given a builtin id for a function like
/// "__builtin_fabsf", return a Function* for "fabsf".
llvm::Value *CodeGenModule::getBuiltinLibFunction(unsigned BuiltinID) {
  assert((Context.BuiltinInfo.isLibFunction(BuiltinID) ||
          Context.BuiltinInfo.isPredefinedLibFunction(BuiltinID)) && 
         "isn't a lib fn");
  
  // Get the name, skip over the __builtin_ prefix (if necessary).
  const char *Name = Context.BuiltinInfo.GetName(BuiltinID);
  if (Context.BuiltinInfo.isLibFunction(BuiltinID))
    Name += 10;
  
  // Get the type for the builtin.
  Builtin::Context::GetBuiltinTypeError Error;
  QualType Type = Context.BuiltinInfo.GetBuiltinType(BuiltinID, Context, Error);
  assert(Error == Builtin::Context::GE_None && "Can't get builtin type");

  const llvm::FunctionType *Ty = 
    cast<llvm::FunctionType>(getTypes().ConvertType(Type));

  // Unique the name through the identifier table.
  Name = getContext().Idents.get(Name).getName();
  // FIXME: param attributes for sext/zext etc.
  return GetOrCreateLLVMFunction(Name, Ty, 0);
}

llvm::Function *CodeGenModule::getIntrinsic(unsigned IID,const llvm::Type **Tys,
                                            unsigned NumTys) {
  return llvm::Intrinsic::getDeclaration(&getModule(),
                                         (llvm::Intrinsic::ID)IID, Tys, NumTys);
}

llvm::Function *CodeGenModule::getMemCpyFn() {
  if (MemCpyFn) return MemCpyFn;
  const llvm::Type *IntPtr = TheTargetData.getIntPtrType();
  return MemCpyFn = getIntrinsic(llvm::Intrinsic::memcpy, &IntPtr, 1);
}

llvm::Function *CodeGenModule::getMemMoveFn() {
  if (MemMoveFn) return MemMoveFn;
  const llvm::Type *IntPtr = TheTargetData.getIntPtrType();
  return MemMoveFn = getIntrinsic(llvm::Intrinsic::memmove, &IntPtr, 1);
}

llvm::Function *CodeGenModule::getMemSetFn() {
  if (MemSetFn) return MemSetFn;
  const llvm::Type *IntPtr = TheTargetData.getIntPtrType();
  return MemSetFn = getIntrinsic(llvm::Intrinsic::memset, &IntPtr, 1);
}

static void appendFieldAndPadding(CodeGenModule &CGM,
                                  std::vector<llvm::Constant*>& Fields,
                                  FieldDecl *FieldD, FieldDecl *NextFieldD,
                                  llvm::Constant* Field,
                                  RecordDecl* RD, const llvm::StructType *STy) {
  // Append the field.
  Fields.push_back(Field);
  
  int StructFieldNo = CGM.getTypes().getLLVMFieldNo(FieldD);
  
  int NextStructFieldNo;
  if (!NextFieldD) {
    NextStructFieldNo = STy->getNumElements();
  } else {
    NextStructFieldNo = CGM.getTypes().getLLVMFieldNo(NextFieldD);
  }
  
  // Append padding
  for (int i = StructFieldNo + 1; i < NextStructFieldNo; i++) {
    llvm::Constant *C = 
      llvm::Constant::getNullValue(STy->getElementType(StructFieldNo + 1));
    
    Fields.push_back(C);
  }
}

llvm::Constant *CodeGenModule::
GetAddrOfConstantCFString(const StringLiteral *Literal) {
  std::string str;
  unsigned StringLength = 0;
  
  bool isUTF16 = false;
  if (Literal->containsNonAsciiOrNull()) {
    // Convert from UTF-8 to UTF-16.
    llvm::SmallVector<UTF16, 128> ToBuf(Literal->getByteLength());
    const UTF8 *FromPtr = (UTF8 *)Literal->getStrData();
    UTF16 *ToPtr = &ToBuf[0];
        
    ConversionResult Result;
    Result = ConvertUTF8toUTF16(&FromPtr, FromPtr+Literal->getByteLength(),
                                &ToPtr, ToPtr+Literal->getByteLength(),
                                strictConversion);
    if (Result == conversionOK) {
      // FIXME: Storing UTF-16 in a C string is a hack to test Unicode strings
      // without doing more surgery to this routine. Since we aren't explicitly
      // checking for endianness here, it's also a bug (when generating code for
      // a target that doesn't match the host endianness). Modeling this as an
      // i16 array is likely the cleanest solution.
      StringLength = ToPtr-&ToBuf[0];
      str.assign((char *)&ToBuf[0], StringLength*2);// Twice as many UTF8 chars.
      isUTF16 = true;
    } else if (Result == sourceIllegal) {
      // FIXME: Have Sema::CheckObjCString() validate the UTF-8 string.
      str.assign(Literal->getStrData(), Literal->getByteLength());
      StringLength = str.length();
    } else
      assert(Result == conversionOK && "UTF-8 to UTF-16 conversion failed");
    
  } else {
    str.assign(Literal->getStrData(), Literal->getByteLength());
    StringLength = str.length();
  }
  llvm::StringMapEntry<llvm::Constant *> &Entry = 
    CFConstantStringMap.GetOrCreateValue(&str[0], &str[str.length()]);
  
  if (llvm::Constant *C = Entry.getValue())
    return C;
  
  llvm::Constant *Zero = llvm::Constant::getNullValue(llvm::Type::Int32Ty);
  llvm::Constant *Zeros[] = { Zero, Zero };
  
  if (!CFConstantStringClassRef) {
    const llvm::Type *Ty = getTypes().ConvertType(getContext().IntTy);
    Ty = llvm::ArrayType::get(Ty, 0);

    // FIXME: This is fairly broken if
    // __CFConstantStringClassReference is already defined, in that it
    // will get renamed and the user will most likely see an opaque
    // error message. This is a general issue with relying on
    // particular names.
    llvm::GlobalVariable *GV = 
      new llvm::GlobalVariable(Ty, false,
                               llvm::GlobalVariable::ExternalLinkage, 0, 
                               "__CFConstantStringClassReference", 
                               &getModule());
    
    // Decay array -> ptr
    CFConstantStringClassRef =
      llvm::ConstantExpr::getGetElementPtr(GV, Zeros, 2);
  }
  
  QualType CFTy = getContext().getCFConstantStringType();
  RecordDecl *CFRD = CFTy->getAsRecordType()->getDecl();

  const llvm::StructType *STy = 
    cast<llvm::StructType>(getTypes().ConvertType(CFTy));

  std::vector<llvm::Constant*> Fields;
  RecordDecl::field_iterator Field = CFRD->field_begin(getContext());

  // Class pointer.
  FieldDecl *CurField = *Field++;
  FieldDecl *NextField = *Field++;
  appendFieldAndPadding(*this, Fields, CurField, NextField,
                        CFConstantStringClassRef, CFRD, STy);
  
  // Flags.
  CurField = NextField;
  NextField = *Field++;
  const llvm::Type *Ty = getTypes().ConvertType(getContext().UnsignedIntTy);
  appendFieldAndPadding(*this, Fields, CurField, NextField,
                        isUTF16 ? llvm::ConstantInt::get(Ty, 0x07d0)
                                : llvm::ConstantInt::get(Ty, 0x07C8), 
                        CFRD, STy);
    
  // String pointer.
  CurField = NextField;
  NextField = *Field++;
  llvm::Constant *C = llvm::ConstantArray::get(str);

  const char *Sect, *Prefix;
  bool isConstant;
  if (isUTF16) {
    Prefix = getContext().Target.getUnicodeStringSymbolPrefix();
    Sect = getContext().Target.getUnicodeStringSection();
    // FIXME: Why does GCC not set constant here?
    isConstant = false;
  } else {
    Prefix = getContext().Target.getStringSymbolPrefix(true);
    Sect = getContext().Target.getCFStringDataSection();
    // FIXME: -fwritable-strings should probably affect this, but we
    // are following gcc here.
    isConstant = true;
  }
  llvm::GlobalVariable *GV = 
    new llvm::GlobalVariable(C->getType(), isConstant, 
                             llvm::GlobalValue::InternalLinkage,
                             C, Prefix, &getModule());
  if (Sect)
    GV->setSection(Sect);
  if (isUTF16) {
    unsigned Align = getContext().getTypeAlign(getContext().ShortTy)/8;
    GV->setAlignment(Align); 
  }
  appendFieldAndPadding(*this, Fields, CurField, NextField,
                        llvm::ConstantExpr::getGetElementPtr(GV, Zeros, 2),
                        CFRD, STy);
  
  // String length.
  CurField = NextField;
  NextField = 0;
  Ty = getTypes().ConvertType(getContext().LongTy);
  appendFieldAndPadding(*this, Fields, CurField, NextField,
                        llvm::ConstantInt::get(Ty, StringLength), CFRD, STy);
  
  // The struct.
  C = llvm::ConstantStruct::get(STy, Fields);
  GV = new llvm::GlobalVariable(C->getType(), true, 
                                llvm::GlobalVariable::InternalLinkage, C, 
                                getContext().Target.getCFStringSymbolPrefix(), 
                                &getModule());
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
  return GetAddrOfConstantString(GetStringForStringLiteral(S));
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
  llvm::Constant *C = llvm::ConstantArray::get(str, false);
  
  // Create a global variable for this string
  return new llvm::GlobalVariable(C->getType(), constant, 
                                  llvm::GlobalValue::InternalLinkage,
                                  C, GlobalName, &CGM.getModule());
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
    GlobalName = getContext().Target.getStringSymbolPrefix(IsConstant);

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
         i = D->propimpl_begin(getContext()),
         e = D->propimpl_end(getContext()); i != e; ++i) {
    ObjCPropertyImplDecl *PID = *i;
    
    // Dynamic is just for type-checking.
    if (PID->getPropertyImplementation() == ObjCPropertyImplDecl::Synthesize) {
      ObjCPropertyDecl *PD = PID->getPropertyDecl();

      // Determine which methods need to be implemented, some may have
      // been overridden. Note that ::isSynthesized is not the method
      // we want, that just indicates if the decl came from a
      // property. What we want to know is if the method is defined in
      // this implementation.
      if (!D->getInstanceMethod(getContext(), PD->getGetterName()))
        CodeGenFunction(*this).GenerateObjCGetter(
                                 const_cast<ObjCImplementationDecl *>(D), PID);
      if (!PD->isReadOnly() &&
          !D->getInstanceMethod(getContext(), PD->getSetterName()))
        CodeGenFunction(*this).GenerateObjCSetter(
                                 const_cast<ObjCImplementationDecl *>(D), PID);
    }
  }
}

/// EmitNamespace - Emit all declarations in a namespace.
void CodeGenModule::EmitNamespace(const NamespaceDecl *ND) {
  for (RecordDecl::decl_iterator I = ND->decls_begin(getContext()),
         E = ND->decls_end(getContext());
       I != E; ++I)
    EmitTopLevelDecl(*I);
}

// EmitLinkageSpec - Emit all declarations in a linkage spec.
void CodeGenModule::EmitLinkageSpec(const LinkageSpecDecl *LSD) {
  if (LSD->getLanguage() != LinkageSpecDecl::lang_c) {
    ErrorUnsupported(LSD, "linkage spec");
    return;
  }

  for (RecordDecl::decl_iterator I = LSD->decls_begin(getContext()),
         E = LSD->decls_end(getContext());
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

  switch (D->getKind()) {
  case Decl::CXXMethod:
  case Decl::Function:
  case Decl::Var:
    EmitGlobal(GlobalDecl(cast<ValueDecl>(D)));
    break;

  // C++ Decls
  case Decl::Namespace:
    EmitNamespace(cast<NamespaceDecl>(D));
    break;
  case Decl::CXXConstructor:
    EmitCXXConstructors(cast<CXXConstructorDecl>(D));
    break;
  case Decl::CXXDestructor:
    EmitCXXDestructors(cast<CXXDestructorDecl>(D));
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
    if (OMD->getBody(getContext()))
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
    std::string AsmString(AD->getAsmString()->getStrData(),
                          AD->getAsmString()->getByteLength());
    
    const std::string &S = getModule().getModuleInlineAsm();
    if (S.empty())
      getModule().setModuleInlineAsm(AsmString);
    else
      getModule().setModuleInlineAsm(S + '\n' + AsmString);
    break;
  }
   
  default: 
    // Make sure we handled everything we should, every other kind is
    // a non-top-level decl.  FIXME: Would be nice to have an
    // isTopLevelDeclKind function. Need to recode Decl::Kind to do
    // that easily.
    assert(isa<TypeDecl>(D) && "Unsupported decl kind");
  }
}
