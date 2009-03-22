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

#include "CGDebugInfo.h"
#include "CodeGenModule.h"
#include "CodeGenFunction.h"
#include "CGCall.h"
#include "CGObjCRuntime.h"
#include "Mangle.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclCXX.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/CallingConv.h"
#include "llvm/Module.h"
#include "llvm/Intrinsics.h"
#include "llvm/Target/TargetData.h"
using namespace clang;
using namespace CodeGen;


CodeGenModule::CodeGenModule(ASTContext &C, const LangOptions &LO,
                             llvm::Module &M, const llvm::TargetData &TD,
                             Diagnostic &diags, bool GenerateDebugInfo)
  : BlockModule(C, M, TD, Types, *this), Context(C), Features(LO), TheModule(M),
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
  DebugInfo = GenerateDebugInfo ? new CGDebugInfo(this) : 0;
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

/// setGlobalVisibility - Set the visibility for the given LLVM
/// GlobalValue according to the given clang AST visibility value.
static void setGlobalVisibility(llvm::GlobalValue *GV,
                                VisibilityAttr::VisibilityTypes Vis) {
  switch (Vis) {
  default: assert(0 && "Unknown visibility!");
  case VisibilityAttr::DefaultVisibility:
    GV->setVisibility(llvm::GlobalValue::DefaultVisibility);
    break;
  case VisibilityAttr::HiddenVisibility:
    GV->setVisibility(llvm::GlobalValue::HiddenVisibility);
    break;
  case VisibilityAttr::ProtectedVisibility:
    GV->setVisibility(llvm::GlobalValue::ProtectedVisibility);
    break;
  }
}

/// \brief Retrieves the mangled name for the given declaration.
///
/// If the given declaration requires a mangled name, returns an
/// const char* containing the mangled name.  Otherwise, returns
/// the unmangled name.
///
/// FIXME: Returning an IdentifierInfo* here is a total hack. We
/// really need some kind of string abstraction that either stores a
/// mangled name or stores an IdentifierInfo*. This will require
/// changes to the GlobalDeclMap, too. (I disagree, I think what we
/// actually need is for Sema to provide some notion of which Decls
/// refer to the same semantic decl. We shouldn't need to mangle the
/// names and see what comes out the same to figure this out. - DWD)
///
/// FIXME: Performance here is going to be terribly until we start
/// caching mangled names. However, we should fix the problem above
/// first.
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
  return MangledNames.GetOrCreateValue(Name.begin(), Name.end()).getKeyData();
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

void CodeGenModule::SetGlobalValueAttributes(const Decl *D, 
                                             bool IsInternal,
                                             bool IsInline,
                                             llvm::GlobalValue *GV,
                                             bool ForDefinition) {
  // FIXME: Set up linkage and many other things.  Note, this is a simple 
  // approximation of what we really want.
  if (!ForDefinition) {
    // Only a few attributes are set on declarations.
    if (D->getAttr<DLLImportAttr>()) {
      // The dllimport attribute is overridden by a subsequent declaration as
      // dllexport.
      if (!D->getAttr<DLLExportAttr>()) {
        // dllimport attribute can be applied only to function decls, not to
        // definitions.
        if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
          if (!FD->getBody())
            GV->setLinkage(llvm::Function::DLLImportLinkage);
        } else
          GV->setLinkage(llvm::Function::DLLImportLinkage);
      }
    } else if (D->getAttr<WeakAttr>() || 
               D->getAttr<WeakImportAttr>()) {
      // "extern_weak" is overloaded in LLVM; we probably should have
      // separate linkage types for this. 
      GV->setLinkage(llvm::Function::ExternalWeakLinkage);
   }
  } else {
    if (IsInternal) {
      GV->setLinkage(llvm::Function::InternalLinkage);
    } else {
      if (D->getAttr<DLLExportAttr>()) {
        if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
          // The dllexport attribute is ignored for undefined symbols.
          if (FD->getBody())
            GV->setLinkage(llvm::Function::DLLExportLinkage);
        } else
          GV->setLinkage(llvm::Function::DLLExportLinkage);
      } else if (D->getAttr<WeakAttr>() || D->getAttr<WeakImportAttr>() || 
                 IsInline)
        GV->setLinkage(llvm::Function::WeakAnyLinkage);
    }
  }

  // FIXME: Figure out the relative priority of the attribute,
  // -fvisibility, and private_extern.
  if (const VisibilityAttr *attr = D->getAttr<VisibilityAttr>())
    setGlobalVisibility(GV, attr->getVisibility());
  // FIXME: else handle -fvisibility

  if (const SectionAttr *SA = D->getAttr<SectionAttr>())
    GV->setSection(SA->getName());

  // Only add to llvm.used when we see a definition, otherwise we
  // might add multiple times or risk the value being replaced by a
  // subsequent RAUW.
  if (ForDefinition) {
    if (D->getAttr<UsedAttr>())
      AddUsedGlobal(GV);
  }
}

void CodeGenModule::SetFunctionAttributes(const Decl *D,
                                          const CGFunctionInfo &Info, 
                                          llvm::Function *F) {
  AttributeListType AttributeList;
  ConstructAttributeList(Info, D, AttributeList);

  F->setAttributes(llvm::AttrListPtr::get(AttributeList.begin(),
                                        AttributeList.size()));

  // Set the appropriate calling convention for the Function.
  if (D->getAttr<FastCallAttr>())
    F->setCallingConv(llvm::CallingConv::X86_FastCall);

  if (D->getAttr<StdCallAttr>())
    F->setCallingConv(llvm::CallingConv::X86_StdCall);
}

/// SetFunctionAttributesForDefinition - Set function attributes
/// specific to a function definition.
void CodeGenModule::SetFunctionAttributesForDefinition(const Decl *D,
                                                       llvm::Function *F) {
  if (isa<ObjCMethodDecl>(D)) {
    SetGlobalValueAttributes(D, true, false, F, true);
  } else {
    const FunctionDecl *FD = cast<FunctionDecl>(D);
    SetGlobalValueAttributes(FD, FD->getStorageClass() == FunctionDecl::Static,
                             FD->isInline(), F, true);
  }
                             
  if (!Features.Exceptions && !Features.ObjCNonFragileABI)
    F->addFnAttr(llvm::Attribute::NoUnwind);  

  if (D->getAttr<AlwaysInlineAttr>())
    F->addFnAttr(llvm::Attribute::AlwaysInline);
  
  if (D->getAttr<NoinlineAttr>())
    F->addFnAttr(llvm::Attribute::NoInline);
}

void CodeGenModule::SetMethodAttributes(const ObjCMethodDecl *MD,
                                        llvm::Function *F) {
  SetFunctionAttributes(MD, getTypes().getFunctionInfo(MD), F);
  
  SetFunctionAttributesForDefinition(MD, F);
}

void CodeGenModule::SetFunctionAttributes(const FunctionDecl *FD,
                                          llvm::Function *F) {
  SetFunctionAttributes(FD, getTypes().getFunctionInfo(FD), F);
  
  SetGlobalValueAttributes(FD, FD->getStorageClass() == FunctionDecl::Static,
                           FD->isInline(), F, false);
}

void CodeGenModule::AddUsedGlobal(llvm::GlobalValue *GV) {
  assert(!GV->isDeclaration() && 
         "Only globals with definition can force usage.");
  llvm::Type *i8PTy = llvm::PointerType::getUnqual(llvm::Type::Int8Ty);
  LLVMUsed.push_back(llvm::ConstantExpr::getBitCast(GV, i8PTy));
}

void CodeGenModule::EmitLLVMUsed() {
  // Don't create llvm.used if there is no need.
  if (LLVMUsed.empty())
    return;

  llvm::ArrayType *ATy = llvm::ArrayType::get(LLVMUsed[0]->getType(), 
                                              LLVMUsed.size());
  llvm::GlobalVariable *GV = 
    new llvm::GlobalVariable(ATy, false, 
                             llvm::GlobalValue::AppendingLinkage,
                             llvm::ConstantArray::get(ATy, LLVMUsed),
                             "llvm.used", &getModule());

  GV->setSection("llvm.metadata");
}

void CodeGenModule::EmitDeferred() {
  // Emit code for any potentially referenced deferred decls.  Since a
  // previously unused static decl may become used during the generation of code
  // for a static function, iterate until no  changes are made.
  while (!DeferredDeclsToEmit.empty()) {
    const ValueDecl *D = DeferredDeclsToEmit.back();
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
  llvm::GlobalValue *annoGV = 
  new llvm::GlobalVariable(anno->getType(), false,
                           llvm::GlobalValue::InternalLinkage, anno,
                           GV->getName() + ".str", M);
  // translation unit name string, emitted into the llvm.metadata section.
  llvm::GlobalValue *unitGV =
  new llvm::GlobalVariable(unit->getType(), false,
                           llvm::GlobalValue::InternalLinkage, unit, ".str", M);

  // Create the ConstantStruct that is the global annotion.
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
  if (Features.EmitAllDecls || Global->getAttr<UsedAttr>())
    return false;

  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(Global)) {
    // Constructors and destructors should never be deferred.
    if (FD->getAttr<ConstructorAttr>() || FD->getAttr<DestructorAttr>())
      return false;

    // FIXME: What about inline, and/or extern inline?
    if (FD->getStorageClass() != FunctionDecl::Static)
      return false;
  } else {
    const VarDecl *VD = cast<VarDecl>(Global);
    assert(VD->isFileVarDecl() && "Invalid decl");

    if (VD->getStorageClass() != VarDecl::Static)
      return false;
  }

  return true;
}

void CodeGenModule::EmitGlobal(const ValueDecl *Global) {
  // If this is an alias definition (which otherwise looks like a declaration)
  // emit it now.
  if (Global->getAttr<AliasAttr>())
    return EmitAliasDefinition(Global);

  // Ignore declarations, they will be emitted on their first use.
  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(Global)) {
    // Forward declarations are emitted lazily on first use.
    if (!FD->isThisDeclarationADefinition())
      return;
  } else {
    const VarDecl *VD = cast<VarDecl>(Global);
    assert(VD->isFileVarDecl() && "Cannot emit local var decl as global.");

    // Forward declarations are emitted lazily on first use.
    if (!VD->getInit() && VD->hasExternalStorage())
      return;
  }

  // Defer code generation when possible if this is a static definition, inline
  // function etc.  These we only want to emit if they are used.
  if (MayDeferGeneration(Global)) {
    // If the value has already been used, add it directly to the
    // DeferredDeclsToEmit list.
    const char *MangledName = getMangledName(Global);
    if (GlobalDeclMap.count(MangledName))
      DeferredDeclsToEmit.push_back(Global);
    else {
      // Otherwise, remember that we saw a deferred decl with this name.  The
      // first use of the mangled name will cause it to move into
      // DeferredDeclsToEmit.
      DeferredDecls[MangledName] = Global;
    }
    return;
  }

  // Otherwise emit the definition.
  EmitGlobalDefinition(Global);
}

void CodeGenModule::EmitGlobalDefinition(const ValueDecl *D) {
  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    EmitGlobalFunctionDefinition(FD);
  } else if (const VarDecl *VD = dyn_cast<VarDecl>(D)) {
    EmitGlobalVarDefinition(VD);
  } else {
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
  llvm::DenseMap<const char*, const ValueDecl*>::iterator DDI = 
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
  llvm::DenseMap<const char*, const ValueDecl*>::iterator DDI = 
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
                             0, Ty->getAddressSpace());
  GV->setName(MangledName);

  // Handle things which are present even on external declarations.
  if (D) {
    // FIXME: This code is overly simple and should be merged with
    // other global handling.
    GV->setConstant(D->getType().isConstant(Context));

    // FIXME: Merge with other attribute handling code.
    if (D->getStorageClass() == VarDecl::PrivateExtern)
      setGlobalVisibility(GV, VisibilityAttr::HiddenVisibility);

    if (D->getAttr<WeakAttr>() || D->getAttr<WeakImportAttr>())
      GV->setLinkage(llvm::GlobalValue::ExternalWeakLinkage);
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

void CodeGenModule::EmitGlobalVarDefinition(const VarDecl *D) {
  llvm::Constant *Init = 0;
  QualType ASTTy = D->getType();

  if (D->getInit() == 0) {
    // This is a tentative definition; tentative definitions are
    // implicitly initialized with { 0 }
    const llvm::Type *InitTy = getTypes().ConvertTypeForMem(ASTTy);
    if (ASTTy->isIncompleteArrayType()) {
      // An incomplete array is normally [ TYPE x 0 ], but we need
      // to fix it to [ TYPE x 1 ].
      const llvm::ArrayType* ATy = cast<llvm::ArrayType>(InitTy);
      InitTy = llvm::ArrayType::get(ATy->getElementType(), 1);
    }
    Init = llvm::Constant::getNullValue(InitTy);
  } else {
    Init = EmitConstantExpr(D->getInit());
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
  
  // If we already have this global and it has an initializer, then
  // we are in the rare situation where we emitted the defining
  // declaration of the global and are now being asked to emit a
  // definition which would be common. This occurs, for example, in
  // the following situation because statics can be emitted out of
  // order:
  //
  //  static int x;
  //  static int *y = &x;
  //  static int x = 10;
  //  int **z = &y;
  //
  // Bail here so we don't blow away the definition. Note that if we
  // can't distinguish here if we emitted a definition with a null
  // initializer, but this case is safe.
  if (GV && GV->hasInitializer() && !GV->getInitializer()->isNullValue()) {
    assert(!D->getInit() && "Emitting multiple definitions of a decl!");
    return;
  }
  
  // We have a definition after a declaration with the wrong type.
  // We must make a new GlobalVariable* and update everything that used OldGV
  // (a declaration or tentative definition) with the new GlobalVariable*
  // (which will be a definition).
  //
  // This happens if there is a prototype for a global (e.g.
  // "extern int x[];") and then a definition of a different type (e.g.
  // "int x[10];"). This also happens when an initializer has a different type
  // from the type of the global (this happens with unions).
  //
  // FIXME: This also ends up happening if there's a definition followed by
  // a tentative definition!  (Although Sema rejects that construct
  // at the moment.)
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
    // FIXME: What if it was attribute used?  Dangling pointer from LLVMUsed.
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

  if (const VisibilityAttr *attr = D->getAttr<VisibilityAttr>())
    setGlobalVisibility(GV, attr->getVisibility());
  // FIXME: else handle -fvisibility

  // Set the llvm linkage type as appropriate.
  if (D->getStorageClass() == VarDecl::Static)
    GV->setLinkage(llvm::Function::InternalLinkage);
  else if (D->getAttr<DLLImportAttr>())
    GV->setLinkage(llvm::Function::DLLImportLinkage);
  else if (D->getAttr<DLLExportAttr>())
    GV->setLinkage(llvm::Function::DLLExportLinkage);
  else if (D->getAttr<WeakAttr>() || D->getAttr<WeakImportAttr>())
    GV->setLinkage(llvm::GlobalVariable::WeakAnyLinkage);
  else {
    // FIXME: This isn't right.  This should handle common linkage and other
    // stuff.
    switch (D->getStorageClass()) {
    case VarDecl::Static: assert(0 && "This case handled above");
    case VarDecl::Auto:
    case VarDecl::Register:
      assert(0 && "Can't have auto or register globals");
    case VarDecl::None:
      if (!D->getInit())
        GV->setLinkage(llvm::GlobalVariable::CommonLinkage);
      else
        GV->setLinkage(llvm::GlobalVariable::ExternalLinkage);
      break;
    case VarDecl::Extern:
      // FIXME: common
      break;
      
    case VarDecl::PrivateExtern:
      GV->setVisibility(llvm::GlobalValue::HiddenVisibility);
      // FIXME: common
      break;
    }
  }

  if (const SectionAttr *SA = D->getAttr<SectionAttr>())
    GV->setSection(SA->getName());

  if (D->getAttr<UsedAttr>())
    AddUsedGlobal(GV);

  // Emit global variable debug information.
  if (CGDebugInfo *DI = getDebugInfo()) {
    DI->setLocation(D->getLocation());
    DI->EmitGlobalVariable(GV, D);
  }
}


void CodeGenModule::EmitGlobalFunctionDefinition(const FunctionDecl *D) {
  const llvm::FunctionType *Ty = 
    cast<llvm::FunctionType>(getTypes().ConvertType(D->getType()));

  // As a special case, make sure that definitions of K&R function
  // "type foo()" aren't declared as varargs (which forces the backend
  // to do unnecessary work).
  if (D->getType()->isFunctionNoProtoType()) {
    assert(Ty->isVarArg() && "Didn't lower type as expected");
    // Due to stret, the lowered function could have arguments.  Just create the
    // same type as was lowered by ConvertType but strip off the varargs bit.
    std::vector<const llvm::Type*> Args(Ty->param_begin(), Ty->param_end());
    Ty = llvm::FunctionType::get(Ty->getReturnType(), Args, false);
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
    // FIXME: If it was attribute(used) the pointer will dangle from the
    // LLVMUsed array!
    cast<llvm::GlobalValue>(Entry)->eraseFromParent();
    
    Entry = NewFn;
  }
  
  llvm::Function *Fn = cast<llvm::Function>(Entry);

  CodeGenFunction(*this).GenerateCode(D, Fn);

  SetFunctionAttributesForDefinition(D, Fn);
  
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
    // FIXME: What if it was attribute used?  Dangling pointer from LLVMUsed.
    Entry->eraseFromParent();
  }
  
  // Now we know that there is no conflict, set the name.
  Entry = GA;
  GA->setName(MangledName);

  // Alias should never be internal or inline.
  SetGlobalValueAttributes(D, false, false, GA, true);
}

void CodeGenModule::UpdateCompletedType(const TagDecl *TD) {
  // Make sure that this type is translated.
  Types.UpdateCompletedType(TD);
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

// We still need to work out the details of handling UTF-16. 
// See: <rdr://2996215>
llvm::Constant *CodeGenModule::
GetAddrOfConstantCFString(const std::string &str) {
  llvm::StringMapEntry<llvm::Constant *> &Entry = 
    CFConstantStringMap.GetOrCreateValue(&str[0], &str[str.length()]);
  
  if (Entry.getValue())
    return Entry.getValue();
  
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
  RecordDecl::field_iterator Field = CFRD->field_begin();

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
                        llvm::ConstantInt::get(Ty, 0x07C8), CFRD, STy);
    
  // String pointer.
  CurField = NextField;
  NextField = *Field++;
  llvm::Constant *C = llvm::ConstantArray::get(str);
  C = new llvm::GlobalVariable(C->getType(), true, 
                               llvm::GlobalValue::InternalLinkage,
                               C, ".str", &getModule());
  appendFieldAndPadding(*this, Fields, CurField, NextField,
                        llvm::ConstantExpr::getGetElementPtr(C, Zeros, 2),
                        CFRD, STy);
  
  // String length.
  CurField = NextField;
  NextField = 0;
  Ty = getTypes().ConvertType(getContext().LongTy);
  appendFieldAndPadding(*this, Fields, CurField, NextField,
                        llvm::ConstantInt::get(Ty, str.length()), CFRD, STy);
  
  // The struct.
  C = llvm::ConstantStruct::get(STy, Fields);
  llvm::GlobalVariable *GV = 
    new llvm::GlobalVariable(C->getType(), true, 
                             llvm::GlobalVariable::InternalLinkage, 
                             C, "", &getModule());
  
  GV->setSection("__DATA,__cfstring");
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
                                  C, GlobalName ? GlobalName : ".str", 
                                  &CGM.getModule());
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
  // Don't share any string literals if writable-strings is turned on.
  if (Features.WritableStrings)
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
  for (ObjCImplementationDecl::propimpl_iterator i = D->propimpl_begin(),
         e = D->propimpl_end(); i != e; ++i) {
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

/// EmitTopLevelDecl - Emit code for a single top level declaration.
void CodeGenModule::EmitTopLevelDecl(Decl *D) {
  // If an error has occurred, stop code generation, but continue
  // parsing and semantic analysis (to ensure all warnings and errors
  // are emitted).
  if (Diags.hasErrorOccurred())
    return;

  switch (D->getKind()) {
  case Decl::Function:
  case Decl::Var:
    EmitGlobal(cast<ValueDecl>(D));
    break;

  case Decl::Namespace:
    ErrorUnsupported(D, "namespace");
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

  case Decl::LinkageSpec: {
    LinkageSpecDecl *LSD = cast<LinkageSpecDecl>(D);
    if (LSD->getLanguage() == LinkageSpecDecl::lang_cxx)
      ErrorUnsupported(LSD, "linkage spec");
    // FIXME: implement C++ linkage, C linkage works mostly by C
    // language reuse already.
    break;
  }

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
