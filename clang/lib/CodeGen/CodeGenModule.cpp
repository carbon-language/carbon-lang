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

  if (Features.ObjC1) {
    if (Features.NeXTRuntime) {
      Runtime = Features.ObjCNonFragileABI ? CreateMacNonFragileABIObjCRuntime(*this) 
                                       : CreateMacObjCRuntime(*this);
    } else {
      Runtime = CreateGNUObjCRuntime(*this);
    }
  }

  // If debug info generation is enabled, create the CGDebugInfo object.
  DebugInfo = GenerateDebugInfo ? new CGDebugInfo(this) : 0;
}

CodeGenModule::~CodeGenModule() {
  delete Runtime;
  delete DebugInfo;
}

void CodeGenModule::Release() {
  EmitDeferred();
  EmitAliases();
  if (Runtime)
    if (llvm::Function *ObjCInitFunction = Runtime->ModuleInitFunction())
      AddGlobalCtor(ObjCInitFunction);
  EmitCtorList(GlobalCtors, "llvm.global_ctors");
  EmitCtorList(GlobalDtors, "llvm.global_dtors");
  EmitAnnotations();
  EmitLLVMUsed();
  BindRuntimeGlobals();
}

void CodeGenModule::BindRuntimeGlobals() {
  // Deal with protecting runtime function names.
  for (unsigned i = 0, e = RuntimeGlobals.size(); i < e; ++i) {
    llvm::GlobalValue *GV = RuntimeGlobals[i].first;
    const std::string &Name = RuntimeGlobals[i].second;
    
    // Discard unused runtime declarations.
    if (GV->isDeclaration() && GV->use_empty()) {
      GV->eraseFromParent();
      continue;
    }
      
    // See if there is a conflict against a function.
    llvm::GlobalValue *Conflict = TheModule.getNamedValue(Name);
    if (Conflict) {
      // Decide which version to take. If the conflict is a definition
      // we are forced to take that, otherwise assume the runtime
      // knows best.

      // FIXME: This will fail phenomenally when the conflict is the
      // wrong type of value. Just bail on it for now. This should
      // really reuse something inside the LLVM Linker code.
      assert(GV->getValueID() == Conflict->getValueID() &&
             "Unable to resolve conflict between globals of different types.");
      if (!Conflict->isDeclaration()) {
        llvm::Value *Casted = 
          llvm::ConstantExpr::getBitCast(Conflict, GV->getType());
        GV->replaceAllUsesWith(Casted);
        GV->eraseFromParent();
      } else {
        GV->takeName(Conflict);
        llvm::Value *Casted = 
          llvm::ConstantExpr::getBitCast(GV, Conflict->getType());
        Conflict->replaceAllUsesWith(Casted);
        Conflict->eraseFromParent();
      }
    } else
      GV->setName(Name);
  }
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
    return ND->getIdentifier()->getName();
  }
    
  llvm::SmallString<256> Name;
  llvm::raw_svector_ostream Out(Name);
  if (!mangleName(ND, Context, Out)) {
    assert(ND->getIdentifier() && "Attempt to mangle unnamed decl.");
    return ND->getIdentifier()->getName();
  }

  Name += '\0';
  return MangledNames.GetOrCreateValue(Name.begin(), Name.end())
           .getKeyData();
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

  // Prefaced with special LLVM marker to indicate that the name
  // should not be munged.
  if (const AsmLabelAttr *ALA = D->getAttr<AsmLabelAttr>())
    GV->setName("\01" + ALA->getLabel());

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


void CodeGenModule::EmitAliases() {
  for (unsigned i = 0, e = Aliases.size(); i != e; ++i) {
    const ValueDecl *D = Aliases[i];
    const AliasAttr *AA = D->getAttr<AliasAttr>();

    // This is something of a hack, if the FunctionDecl got overridden
    // then its attributes will be moved to the new declaration. In
    // this case the current decl has no alias attribute, but we will
    // eventually see it.
    if (!AA)
      continue;

    const std::string& aliaseeName = AA->getAliasee();
    llvm::GlobalValue *aliasee = getModule().getNamedValue(aliaseeName);
    if (!aliasee) {
      // FIXME: This isn't unsupported, this is just an error, which
      // sema should catch, but...
      ErrorUnsupported(D, "alias referencing a missing function");
      continue;
    }

    llvm::GlobalValue *GA = 
      new llvm::GlobalAlias(aliasee->getType(),
                            llvm::Function::ExternalLinkage,
                            getMangledName(D), aliasee, 
                            &getModule());
    
    llvm::GlobalValue *&Entry = GlobalDeclMap[getMangledName(D)];
    if (Entry) {
      // If we created a dummy function for this then replace it.
      GA->takeName(Entry);
            
      llvm::Value *Casted = 
        llvm::ConstantExpr::getBitCast(GA, Entry->getType());
      Entry->replaceAllUsesWith(Casted);
      Entry->eraseFromParent();

      Entry = GA;
    }

    // Alias should never be internal or inline.
    SetGlobalValueAttributes(D, false, false, GA, true);
  }
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
  // Emit code for any deferred decl which was used.  Since a
  // previously unused static decl may become used during the
  // generation of code for a static function, iterate until no
  // changes are made.
  bool Changed;
  do {
    Changed = false;
    
    for (std::list<const ValueDecl*>::iterator i = DeferredDecls.begin(),
         e = DeferredDecls.end(); i != e; ) {
      const ValueDecl *D = *i;
      
      // Check if we have used a decl with the same name
      // FIXME: The AST should have some sort of aggregate decls or
      // global symbol map.
      // FIXME: This is missing some important cases. For example, we
      // need to check for uses in an alias.
      if (!GlobalDeclMap.count(getMangledName(D))) {
        ++i;
        continue;
      }
      
      // Emit the definition.
      EmitGlobalDefinition(D);

      // Erase the used decl from the list.
      i = DeferredDecls.erase(i);

      // Remember that we made a change.
      Changed = true;
    }
  } while (Changed);
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

    if (FD->getStorageClass() != FunctionDecl::Static)
      return false;
  } else {
    const VarDecl *VD = cast<VarDecl>(Global);
    assert(VD->isFileVarDecl() && "Invalid decl.");

    if (VD->getStorageClass() != VarDecl::Static)
      return false;
  }

  return true;
}

void CodeGenModule::EmitGlobal(const ValueDecl *Global) {
  // Aliases are deferred until code for everything else has been
  // emitted.
  if (Global->getAttr<AliasAttr>()) {
    Aliases.push_back(Global);
    return;
  }

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

  // Defer code generation when possible.
  if (MayDeferGeneration(Global)) {
    DeferredDecls.push_back(Global);
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

 llvm::Constant *CodeGenModule::GetAddrOfGlobalVar(const VarDecl *D) {
  assert(D->hasGlobalStorage() && "Not a global variable");

  QualType ASTTy = D->getType();
  const llvm::Type *Ty = getTypes().ConvertTypeForMem(ASTTy);
  const llvm::Type *PTy = llvm::PointerType::get(Ty, ASTTy.getAddressSpace());

  // Lookup the entry, lazily creating it if necessary.
  llvm::GlobalValue *&Entry = GlobalDeclMap[getMangledName(D)];
  if (!Entry) {
    llvm::GlobalVariable *GV = 
      new llvm::GlobalVariable(Ty, false, 
                               llvm::GlobalValue::ExternalLinkage,
                               0, getMangledName(D), &getModule(), 
                               0, ASTTy.getAddressSpace());
    Entry = GV;

    // Handle things which are present even on external declarations.

    // FIXME: This code is overly simple and should be merged with
    // other global handling.

    GV->setConstant(D->getType().isConstant(Context));

    // FIXME: Merge with other attribute handling code.

    if (D->getStorageClass() == VarDecl::PrivateExtern)
      setGlobalVisibility(GV, VisibilityAttr::HiddenVisibility);

    if (D->getAttr<WeakAttr>() || D->getAttr<WeakImportAttr>())
      GV->setLinkage(llvm::GlobalValue::ExternalWeakLinkage);

    if (const AsmLabelAttr *ALA = D->getAttr<AsmLabelAttr>()) {
      // Prefaced with special LLVM marker to indicate that the name
      // should not be munged.
      GV->setName("\01" + ALA->getLabel());
    }  
  }
  
  // Make sure the result is of the correct type.
  return llvm::ConstantExpr::getBitCast(Entry, PTy);
}

void CodeGenModule::EmitGlobalVarDefinition(const VarDecl *D) {
  llvm::Constant *Init = 0;
  QualType ASTTy = D->getType();
  const llvm::Type *VarTy = getTypes().ConvertTypeForMem(ASTTy);

  if (D->getInit() == 0) {
    // This is a tentative definition; tentative definitions are
    // implicitly initialized with { 0 }
    const llvm::Type* InitTy;
    if (ASTTy->isIncompleteArrayType()) {
      // An incomplete array is normally [ TYPE x 0 ], but we need
      // to fix it to [ TYPE x 1 ].
      const llvm::ArrayType* ATy = cast<llvm::ArrayType>(VarTy);
      InitTy = llvm::ArrayType::get(ATy->getElementType(), 1);
    } else {
      InitTy = VarTy;
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

  llvm::GlobalValue *&Entry = GlobalDeclMap[getMangledName(D)];
  llvm::GlobalVariable *GV = cast_or_null<llvm::GlobalVariable>(Entry);
  
  if (!GV) {
    GV = new llvm::GlobalVariable(InitType, false, 
                                  llvm::GlobalValue::ExternalLinkage,
                                  0, getMangledName(D), 
                                  &getModule(), 0, ASTTy.getAddressSpace());

  } else if (GV->hasInitializer() && !GV->getInitializer()->isNullValue()) {
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
    assert(!D->getInit() && "Emitting multiple definitions of a decl!");
    return;

  } else if (GV->getType() != 
             llvm::PointerType::get(InitType, ASTTy.getAddressSpace())) {
    // We have a definition after a prototype with the wrong type.
    // We must make a new GlobalVariable* and update everything that used OldGV
    // (a declaration or tentative definition) with the new GlobalVariable*
    // (which will be a definition).
    //
    // This happens if there is a prototype for a global (e.g. "extern int x[];")
    // and then a definition of a different type (e.g. "int x[10];"). This also
    // happens when an initializer has a different type from the type of the
    // global (this happens with unions).
    //
    // FIXME: This also ends up happening if there's a definition followed by
    // a tentative definition!  (Although Sema rejects that construct
    // at the moment.)

    // Save the old global
    llvm::GlobalVariable *OldGV = GV;

    // Make a new global with the correct type
    GV = new llvm::GlobalVariable(InitType, false, 
                                  llvm::GlobalValue::ExternalLinkage,
                                  0, getMangledName(D), 
                                  &getModule(), 0, ASTTy.getAddressSpace());
    // Steal the name of the old global
    GV->takeName(OldGV);

    // Replace all uses of the old global with the new global
    llvm::Constant *NewPtrForOldDecl = 
        llvm::ConstantExpr::getBitCast(GV, OldGV->getType());
    OldGV->replaceAllUsesWith(NewPtrForOldDecl);

    // Erase the old global, since it is no longer used.
    OldGV->eraseFromParent();
  }

  Entry = GV;

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

  if (const AsmLabelAttr *ALA = D->getAttr<AsmLabelAttr>()) {
    // Prefaced with special LLVM marker to indicate that the name
    // should not be munged.
    GV->setName("\01" + ALA->getLabel());
  }
  
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
  CGDebugInfo *DI = getDebugInfo();
  if(DI) {
    DI->setLocation(D->getLocation());
    DI->EmitGlobalVariable(GV, D);
  }
}

llvm::GlobalValue *
CodeGenModule::EmitForwardFunctionDefinition(const FunctionDecl *D,
                                             const llvm::Type *Ty) {
  bool DoSetAttributes = true;
  if (!Ty) {
    Ty = getTypes().ConvertType(D->getType());
    if (!isa<llvm::FunctionType>(Ty)) {
      // This function doesn't have a complete type (for example, the return
      // type is an incomplete struct). Use a fake type instead, and make
      // sure not to try to set attributes.
      Ty = llvm::FunctionType::get(llvm::Type::VoidTy,
                                   std::vector<const llvm::Type*>(), false);
      DoSetAttributes = false;
    }
  }
  llvm::Function *F = llvm::Function::Create(cast<llvm::FunctionType>(Ty), 
                                             llvm::Function::ExternalLinkage,
                                             getMangledName(D),
                                             &getModule());
  if (DoSetAttributes)
    SetFunctionAttributes(D, F);
  return F;
}

llvm::Constant *CodeGenModule::GetAddrOfFunction(const FunctionDecl *D) {
  QualType ASTTy = D->getType();
  const llvm::Type *Ty = getTypes().ConvertTypeForMem(ASTTy);
  const llvm::Type *PTy = llvm::PointerType::get(Ty, ASTTy.getAddressSpace());
  
  // Lookup the entry, lazily creating it if necessary.
  llvm::GlobalValue *&Entry = GlobalDeclMap[getMangledName(D)];
  if (!Entry)
    Entry = EmitForwardFunctionDefinition(D, 0);

  return llvm::ConstantExpr::getBitCast(Entry, PTy);
}

void CodeGenModule::EmitGlobalFunctionDefinition(const FunctionDecl *D) {
  const llvm::FunctionType *Ty = 
    cast<llvm::FunctionType>(getTypes().ConvertType(D->getType()));

  // As a special case, make sure that definitions of K&R function
  // "type foo()" aren't declared as varargs (which forces the backend
  // to do unnecessary work).
  if (Ty->isVarArg() && Ty->getNumParams() == 0 && Ty->isVarArg())
    Ty = llvm::FunctionType::get(Ty->getReturnType(),
                                 std::vector<const llvm::Type*>(),
                                 false);

  llvm::GlobalValue *&Entry = GlobalDeclMap[getMangledName(D)];
  if (!Entry) {
    Entry = EmitForwardFunctionDefinition(D, Ty);
  } else {
    // If the types mismatch then we have to rewrite the definition.
    if (Entry->getType() != llvm::PointerType::getUnqual(Ty)) {
      // Otherwise, we have a definition after a prototype with the
      // wrong type.  F is the Function* for the one with the wrong
      // type, we must make a new Function* and update everything that
      // used F (a declaration) with the new Function* (which will be
      // a definition).
      //
      // This happens if there is a prototype for a function
      // (e.g. "int f()") and then a definition of a different type
      // (e.g. "int f(int x)").  Start by making a new function of the
      // correct type, RAUW, then steal the name.
      llvm::GlobalValue *NewFn = EmitForwardFunctionDefinition(D, Ty);
      NewFn->takeName(Entry);
      
      // Replace uses of F with the Function we will endow with a body.
      llvm::Constant *NewPtrForOldDecl = 
        llvm::ConstantExpr::getBitCast(NewFn, Entry->getType());
      Entry->replaceAllUsesWith(NewPtrForOldDecl);
            
      // Ok, delete the old function now, which is dead.
      assert(Entry->isDeclaration() && "Shouldn't replace non-declaration");
      Entry->eraseFromParent();
      
      Entry = NewFn;
    }
  }

  llvm::Function *Fn = cast<llvm::Function>(Entry);    
  CodeGenFunction(*this).GenerateCode(D, Fn);

  SetFunctionAttributesForDefinition(D, Fn);
  
  if (const ConstructorAttr *CA = D->getAttr<ConstructorAttr>()) {
    AddGlobalCtor(Fn, CA->getPriority());
  } else if (const DestructorAttr *DA = D->getAttr<DestructorAttr>()) {
    AddGlobalDtor(Fn, DA->getPriority());
  }
}

llvm::Function *
CodeGenModule::CreateRuntimeFunction(const llvm::FunctionType *FTy,
                                     const std::string &Name) {
  llvm::Function *Fn = llvm::Function::Create(FTy, 
                                              llvm::Function::ExternalLinkage,
                                              "", &TheModule);
  RuntimeGlobals.push_back(std::make_pair(Fn, Name));
  return Fn;
}

llvm::GlobalVariable *
CodeGenModule::CreateRuntimeVariable(const llvm::Type *Ty,
                                     const std::string &Name) {
  llvm::GlobalVariable *GV = 
    new llvm::GlobalVariable(Ty, /*Constant=*/false,
                             llvm::GlobalValue::ExternalLinkage,
                             0, "", &TheModule);
  RuntimeGlobals.push_back(std::make_pair(GV, Name));
  return GV;
}

void CodeGenModule::UpdateCompletedType(const TagDecl *TD) {
  // Make sure that this type is translated.
  Types.UpdateCompletedType(TD);
}


/// getBuiltinLibFunction
llvm::Value *CodeGenModule::getBuiltinLibFunction(unsigned BuiltinID) {
  if (BuiltinID > BuiltinFunctions.size())
    BuiltinFunctions.resize(BuiltinID);
  
  // Cache looked up functions.  Since builtin id #0 is invalid we don't reserve
  // a slot for it.
  assert(BuiltinID && "Invalid Builtin ID");
  llvm::Value *&FunctionSlot = BuiltinFunctions[BuiltinID-1];
  if (FunctionSlot)
    return FunctionSlot;
  
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

  // FIXME: This has a serious problem with code like this:
  //  void abs() {}
  //    ... __builtin_abs(x);
  // The two versions of abs will collide.  The fix is for the builtin to win,
  // and for the existing one to be turned into a constantexpr cast of the
  // builtin.  In the case where the existing one is a static function, it
  // should just be renamed.
  if (llvm::Function *Existing = getModule().getFunction(Name)) {
    if (Existing->getFunctionType() == Ty && Existing->hasExternalLinkage())
      return FunctionSlot = Existing;
    assert(Existing == 0 && "FIXME: Name collision");
  }

  llvm::GlobalValue *&ExistingFn =
    GlobalDeclMap[getContext().Idents.get(Name).getName()];
  if (ExistingFn)
    return FunctionSlot = llvm::ConstantExpr::getBitCast(ExistingFn, Ty);

  // FIXME: param attributes for sext/zext etc.
  return FunctionSlot = ExistingFn =
    llvm::Function::Create(Ty, llvm::Function::ExternalLinkage, Name,
                           &getModule());
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
                                  RecordDecl* RD, const llvm::StructType *STy)
{
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
    break;
      
  case Decl::ObjCProtocol:
  case Decl::ObjCCategory:
  case Decl::ObjCInterface: {
    ObjCContainerDecl *OCD = cast<ObjCContainerDecl>(D);
    for (ObjCContainerDecl::tuvar_iterator i = OCD->tuvar_begin(),
         e = OCD->tuvar_end(); i != e; ++i) {
        VarDecl *VD = *i;
        EmitGlobal(VD);
    }
    if (D->getKind() == Decl::ObjCProtocol) 
      Runtime->GenerateProtocol(cast<ObjCProtocolDecl>(D));
    break;
  }

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
