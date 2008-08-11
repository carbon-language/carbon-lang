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
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclObjC.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/CallingConv.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/Intrinsics.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Analysis/Verifier.h"
#include <algorithm>
using namespace clang;
using namespace CodeGen;


CodeGenModule::CodeGenModule(ASTContext &C, const LangOptions &LO,
                             llvm::Module &M, const llvm::TargetData &TD,
                             Diagnostic &diags, bool GenerateDebugInfo,
                             bool UseMacObjCRuntime)
  : Context(C), Features(LO), TheModule(M), TheTargetData(TD), Diags(diags),
    Types(C, M, TD), MemCpyFn(0), MemMoveFn(0), MemSetFn(0),
    CFConstantStringClassRef(0) {
  //TODO: Make this selectable at runtime
  if (UseMacObjCRuntime) {
    Runtime = CreateMacObjCRuntime(*this);
  } else {
    Runtime = CreateGNUObjCRuntime(*this);
  }

  // If debug info generation is enabled, create the CGDebugInfo object.
  DebugInfo = GenerateDebugInfo ? new CGDebugInfo(this) : 0;      
}

CodeGenModule::~CodeGenModule() {
  delete Runtime;
  delete DebugInfo;
}

void CodeGenModule::Release() {
  EmitStatics();
  llvm::Function *ObjCInitFunction = Runtime->ModuleInitFunction();
  if (ObjCInitFunction)
    AddGlobalCtor(ObjCInitFunction);
  EmitCtorList(GlobalCtors, "llvm.global_ctors");
  EmitCtorList(GlobalDtors, "llvm.global_dtors");
  EmitAnnotations();
  // Run the verifier to check that the generated code is consistent.
  assert(!verifyModule(TheModule));
}

/// WarnUnsupported - Print out a warning that codegen doesn't support the
/// specified stmt yet.
void CodeGenModule::WarnUnsupported(const Stmt *S, const char *Type) {
  unsigned DiagID = getDiags().getCustomDiagID(Diagnostic::Warning, 
                                               "cannot codegen this %0 yet");
  SourceRange Range = S->getSourceRange();
  std::string Msg = Type;
  getDiags().Report(Context.getFullLoc(S->getLocStart()), DiagID,
                    &Msg, 1, &Range, 1);
}

/// WarnUnsupported - Print out a warning that codegen doesn't support the
/// specified decl yet.
void CodeGenModule::WarnUnsupported(const Decl *D, const char *Type) {
  unsigned DiagID = getDiags().getCustomDiagID(Diagnostic::Warning, 
                                               "cannot codegen this %0 yet");
  std::string Msg = Type;
  getDiags().Report(Context.getFullLoc(D->getLocation()), DiagID,
                    &Msg, 1);
}

/// setVisibility - Set the visibility for the given LLVM GlobalValue
/// according to the given clang AST visibility value.
void CodeGenModule::setVisibility(llvm::GlobalValue *GV,
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

/// AddGlobalCtor - Add a function to the list that will be called before
/// main() runs.
void CodeGenModule::AddGlobalCtor(llvm::Function * Ctor, int Priority) {
  // TODO: Type coercion of void()* types.
  GlobalCtors.push_back(std::make_pair(Ctor, Priority));
}

/// AddGlobalDtor - Add a function to the list that will be called
/// when the module is unloaded.
void CodeGenModule::AddGlobalDtor(llvm::Function * Dtor, int Priority) {
  // TODO: Type coercion of void()* types.
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

bool hasAggregateLLVMType(QualType T) {
  return !T->isRealType() && !T->isPointerLikeType() &&
         !T->isVoidType() && !T->isVectorType() && !T->isFunctionType();
}

void CodeGenModule::SetGlobalValueAttributes(const FunctionDecl *FD,
                                             llvm::GlobalValue *GV) {
  // TODO: Set up linkage and many other things.  Note, this is a simple 
  // approximation of what we really want.
  if (FD->getStorageClass() == FunctionDecl::Static)
    GV->setLinkage(llvm::Function::InternalLinkage);
  else if (FD->getAttr<DLLImportAttr>())
    GV->setLinkage(llvm::Function::DLLImportLinkage);
  else if (FD->getAttr<DLLExportAttr>())
    GV->setLinkage(llvm::Function::DLLExportLinkage);
  else if (FD->getAttr<WeakAttr>() || FD->isInline())
    GV->setLinkage(llvm::Function::WeakLinkage);

  if (const VisibilityAttr *attr = FD->getAttr<VisibilityAttr>())
    CodeGenModule::setVisibility(GV, attr->getVisibility());
  // FIXME: else handle -fvisibility

  if (const AsmLabelAttr *ALA = FD->getAttr<AsmLabelAttr>()) {
    // Prefaced with special LLVM marker to indicate that the name
    // should not be munged.
    GV->setName("\01" + ALA->getLabel());
  }
}

void CodeGenModule::SetFunctionAttributes(const FunctionDecl *FD,
                                          llvm::Function *F,
                                          const llvm::FunctionType *FTy) {
  unsigned FuncAttrs = 0;
  if (FD->getAttr<NoThrowAttr>())
    FuncAttrs |= llvm::ParamAttr::NoUnwind;
  if (FD->getAttr<NoReturnAttr>())
    FuncAttrs |= llvm::ParamAttr::NoReturn;

  llvm::SmallVector<llvm::ParamAttrsWithIndex, 8> ParamAttrList;
  if (FuncAttrs)
    ParamAttrList.push_back(llvm::ParamAttrsWithIndex::get(0, FuncAttrs));
  // Note that there is parallel code in CodeGenFunction::EmitCallExpr
  bool AggregateReturn = hasAggregateLLVMType(FD->getResultType());
  if (AggregateReturn)
    ParamAttrList.push_back(
        llvm::ParamAttrsWithIndex::get(1, llvm::ParamAttr::StructRet));
  unsigned increment = AggregateReturn ? 2 : 1;
  const FunctionTypeProto* FTP = dyn_cast<FunctionTypeProto>(FD->getType());
  if (FTP) {
    for (unsigned i = 0; i < FTP->getNumArgs(); i++) {
      QualType ParamType = FTP->getArgType(i);
      unsigned ParamAttrs = 0;
      if (ParamType->isRecordType())
        ParamAttrs |= llvm::ParamAttr::ByVal;
      if (ParamType->isSignedIntegerType() &&
          ParamType->isPromotableIntegerType())
        ParamAttrs |= llvm::ParamAttr::SExt;
      if (ParamType->isUnsignedIntegerType() &&
          ParamType->isPromotableIntegerType())
        ParamAttrs |= llvm::ParamAttr::ZExt;
      if (ParamAttrs)
        ParamAttrList.push_back(llvm::ParamAttrsWithIndex::get(i + increment,
                                                               ParamAttrs));
    }
  }

  F->setParamAttrs(llvm::PAListPtr::get(ParamAttrList.begin(),
                                        ParamAttrList.size()));

  // Set the appropriate calling convention for the Function.
  if (FD->getAttr<FastCallAttr>())
    F->setCallingConv(llvm::CallingConv::Fast);

  SetGlobalValueAttributes(FD, F);
}

void CodeGenModule::EmitObjCMethod(const ObjCMethodDecl *OMD) {
  // If this is not a prototype, emit the body.
  if (OMD->getBody())
    CodeGenFunction(*this).GenerateObjCMethod(OMD);
}
void CodeGenModule::EmitObjCProtocolImplementation(const ObjCProtocolDecl *PD){
  llvm::SmallVector<std::string, 16> Protocols;
  for (ObjCProtocolDecl::protocol_iterator PI = PD->protocol_begin(),
       E = PD->protocol_end(); PI != E; ++PI)
    Protocols.push_back((*PI)->getName());
  llvm::SmallVector<llvm::Constant*, 16> InstanceMethodNames;
  llvm::SmallVector<llvm::Constant*, 16> InstanceMethodTypes;
  for (ObjCProtocolDecl::instmeth_iterator iter = PD->instmeth_begin(),
       E = PD->instmeth_end(); iter != E; iter++) {
    std::string TypeStr;
    Context.getObjCEncodingForMethodDecl(*iter, TypeStr);
    InstanceMethodNames.push_back(
        GetAddrOfConstantString((*iter)->getSelector().getName()));
    InstanceMethodTypes.push_back(GetAddrOfConstantString(TypeStr));
  }
  // Collect information about class methods:
  llvm::SmallVector<llvm::Constant*, 16> ClassMethodNames;
  llvm::SmallVector<llvm::Constant*, 16> ClassMethodTypes;
  for (ObjCProtocolDecl::classmeth_iterator iter = PD->classmeth_begin(),
      endIter = PD->classmeth_end() ; iter != endIter ; iter++) {
    std::string TypeStr;
    Context.getObjCEncodingForMethodDecl((*iter),TypeStr);
    ClassMethodNames.push_back(
        GetAddrOfConstantString((*iter)->getSelector().getName()));
    ClassMethodTypes.push_back(GetAddrOfConstantString(TypeStr));
  }
  Runtime->GenerateProtocol(PD->getName(), Protocols, InstanceMethodNames,
      InstanceMethodTypes, ClassMethodNames, ClassMethodTypes);
}

void CodeGenModule::EmitObjCCategoryImpl(const ObjCCategoryImplDecl *OCD) {

  // Collect information about instance methods
  llvm::SmallVector<Selector, 16> InstanceMethodSels;
  llvm::SmallVector<llvm::Constant*, 16> InstanceMethodTypes;
  for (ObjCCategoryDecl::instmeth_iterator iter = OCD->instmeth_begin(),
      endIter = OCD->instmeth_end() ; iter != endIter ; iter++) {
    InstanceMethodSels.push_back((*iter)->getSelector());
    std::string TypeStr;
    Context.getObjCEncodingForMethodDecl(*iter,TypeStr);
    InstanceMethodTypes.push_back(GetAddrOfConstantString(TypeStr));
  }

  // Collect information about class methods
  llvm::SmallVector<Selector, 16> ClassMethodSels;
  llvm::SmallVector<llvm::Constant*, 16> ClassMethodTypes;
  for (ObjCCategoryDecl::classmeth_iterator iter = OCD->classmeth_begin(),
      endIter = OCD->classmeth_end() ; iter != endIter ; iter++) {
    ClassMethodSels.push_back((*iter)->getSelector());
    std::string TypeStr;
    Context.getObjCEncodingForMethodDecl(*iter,TypeStr);
    ClassMethodTypes.push_back(GetAddrOfConstantString(TypeStr));
  }

  // Collect the names of referenced protocols
  llvm::SmallVector<std::string, 16> Protocols;
  const ObjCInterfaceDecl *ClassDecl = OCD->getClassInterface();
  const ObjCList<ObjCProtocolDecl> &Protos =ClassDecl->getReferencedProtocols();
  for (ObjCList<ObjCProtocolDecl>::iterator I = Protos.begin(),
       E = Protos.end(); I != E; ++I)
    Protocols.push_back((*I)->getName());

  // Generate the category
  Runtime->GenerateCategory(OCD->getClassInterface()->getName(),
      OCD->getName(), InstanceMethodSels, InstanceMethodTypes,
      ClassMethodSels, ClassMethodTypes, Protocols);
}

void CodeGenModule::EmitObjCClassImplementation(
    const ObjCImplementationDecl *OID) {
  // Get the superclass name.
  const ObjCInterfaceDecl * SCDecl = OID->getClassInterface()->getSuperClass();
  const char * SCName = NULL;
  if (SCDecl) {
    SCName = SCDecl->getName();
  }

  // Get the class name
  ObjCInterfaceDecl * ClassDecl = (ObjCInterfaceDecl*)OID->getClassInterface();
  const char * ClassName = ClassDecl->getName();

  // Get the size of instances.  For runtimes that support late-bound instances
  // this should probably be something different (size just of instance
  // varaibles in this class, not superclasses?).
  int instanceSize = 0;
  const llvm::Type *ObjTy;
  if (!Runtime->LateBoundIVars()) {
    ObjTy = getTypes().ConvertType(Context.getObjCInterfaceType(ClassDecl));
    instanceSize = TheTargetData.getABITypeSize(ObjTy);
  }

  // Collect information about instance variables.
  llvm::SmallVector<llvm::Constant*, 16> IvarNames;
  llvm::SmallVector<llvm::Constant*, 16> IvarTypes;
  llvm::SmallVector<llvm::Constant*, 16> IvarOffsets;
  const llvm::StructLayout *Layout =
    TheTargetData.getStructLayout(cast<llvm::StructType>(ObjTy));
  ObjTy = llvm::PointerType::getUnqual(ObjTy);
  for (ObjCInterfaceDecl::ivar_iterator iter = ClassDecl->ivar_begin(),
      endIter = ClassDecl->ivar_end() ; iter != endIter ; iter++) {
      // Store the name
      IvarNames.push_back(GetAddrOfConstantString((*iter)->getName()));
      // Get the type encoding for this ivar
      std::string TypeStr;
      llvm::SmallVector<const RecordType *, 8> EncodingRecordTypes;
      Context.getObjCEncodingForType((*iter)->getType(), TypeStr,
                                     EncodingRecordTypes);
      IvarTypes.push_back(GetAddrOfConstantString(TypeStr));
      // Get the offset
      int offset =
        (int)Layout->getElementOffset(getTypes().getLLVMFieldNo(*iter));
      IvarOffsets.push_back(
          llvm::ConstantInt::get(llvm::Type::Int32Ty, offset));
  }

  // Collect information about instance methods
  llvm::SmallVector<Selector, 16> InstanceMethodSels;
  llvm::SmallVector<llvm::Constant*, 16> InstanceMethodTypes;
  for (ObjCImplementationDecl::instmeth_iterator iter = OID->instmeth_begin(),
      endIter = OID->instmeth_end() ; iter != endIter ; iter++) {
    InstanceMethodSels.push_back((*iter)->getSelector());
    std::string TypeStr;
    Context.getObjCEncodingForMethodDecl((*iter),TypeStr);
    InstanceMethodTypes.push_back(GetAddrOfConstantString(TypeStr));
  }

  // Collect information about class methods
  llvm::SmallVector<Selector, 16> ClassMethodSels;
  llvm::SmallVector<llvm::Constant*, 16> ClassMethodTypes;
  for (ObjCImplementationDecl::classmeth_iterator iter = OID->classmeth_begin(),
      endIter = OID->classmeth_end() ; iter != endIter ; iter++) {
    ClassMethodSels.push_back((*iter)->getSelector());
    std::string TypeStr;
    Context.getObjCEncodingForMethodDecl((*iter),TypeStr);
    ClassMethodTypes.push_back(GetAddrOfConstantString(TypeStr));
  }
  // Collect the names of referenced protocols
  llvm::SmallVector<std::string, 16> Protocols;
  const ObjCList<ObjCProtocolDecl> &Protos =ClassDecl->getReferencedProtocols();
  for (ObjCList<ObjCProtocolDecl>::iterator I = Protos.begin(),
       E = Protos.end(); I != E; ++I)
    Protocols.push_back((*I)->getName());

  // Generate the category
  Runtime->GenerateClass(ClassName, SCName, instanceSize, IvarNames, IvarTypes,
                         IvarOffsets, InstanceMethodSels, InstanceMethodTypes,
                         ClassMethodSels, ClassMethodTypes, Protocols);
}

void CodeGenModule::EmitStatics() {
  // Emit code for each used static decl encountered.  Since a previously unused
  // static decl may become used during the generation of code for a static
  // function, iterate until no changes are made.
  bool Changed;
  do {
    Changed = false;
    for (unsigned i = 0, e = StaticDecls.size(); i != e; ++i) {
      const ValueDecl *D = StaticDecls[i];

      // Check if we have used a decl with the same name
      // FIXME: The AST should have some sort of aggregate decls or
      // global symbol map.
      if (!GlobalDeclMap.count(D->getName()))
        continue;

      // Emit the definition.
      EmitGlobalDefinition(D);

      // Erase the used decl from the list.
      StaticDecls[i] = StaticDecls.back();
      StaticDecls.pop_back();
      --i;
      --e;
      
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

void CodeGenModule::EmitGlobal(const ValueDecl *Global) {
  bool isDef, isStatic;

  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(Global)) {
    isDef = (FD->isThisDeclarationADefinition() ||
             FD->getAttr<AliasAttr>());
    isStatic = FD->getStorageClass() == FunctionDecl::Static;
  } else if (const VarDecl *VD = cast<VarDecl>(Global)) {
    assert(VD->isFileVarDecl() && "Cannot emit local var decl as global.");

    isDef = !(VD->getStorageClass() == VarDecl::Extern && VD->getInit() == 0);
    isStatic = VD->getStorageClass() == VarDecl::Static;
  } else {
    assert(0 && "Invalid argument to EmitGlobal");
    return;
  }

  // Forward declarations are emitted lazily on first use.
  if (!isDef)
    return;

  // If the global is a static, defer code generation until later so
  // we can easily omit unused statics.
  if (isStatic) {
    StaticDecls.push_back(Global);
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
  llvm::GlobalValue *&Entry = GlobalDeclMap[D->getName()];
  if (!Entry)
    Entry = new llvm::GlobalVariable(Ty, false, 
                                     llvm::GlobalValue::ExternalLinkage,
                                     0, D->getName(), &getModule(), 0,
                                     ASTTy.getAddressSpace());
  
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
  }
  const llvm::Type* InitType = Init->getType();

  llvm::GlobalValue *&Entry = GlobalDeclMap[D->getName()];
  llvm::GlobalVariable *GV = cast_or_null<llvm::GlobalVariable>(Entry);
  
  if (!GV) {
    GV = new llvm::GlobalVariable(InitType, false, 
                                  llvm::GlobalValue::ExternalLinkage,
                                  0, D->getName(), &getModule(), 0,
                                  ASTTy.getAddressSpace());
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
                                  0, D->getName(), &getModule(), 0,
                                  ASTTy.getAddressSpace());
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
                                   SM.getLogicalLineNumber(D->getLocation())));
  }

  GV->setInitializer(Init);

  // FIXME: This is silly; getTypeAlign should just work for incomplete arrays
  unsigned Align;
  if (const IncompleteArrayType* IAT =
        Context.getAsIncompleteArrayType(D->getType()))
    Align = Context.getTypeAlign(IAT->getElementType());
  else
    Align = Context.getTypeAlign(D->getType());
  if (const AlignedAttr* AA = D->getAttr<AlignedAttr>()) {
    Align = std::max(Align, AA->getAlignment());
  }
  GV->setAlignment(Align / 8);

  if (const VisibilityAttr *attr = D->getAttr<VisibilityAttr>())
    setVisibility(GV, attr->getVisibility());
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
  else if (D->getAttr<WeakAttr>())
    GV->setLinkage(llvm::GlobalVariable::WeakLinkage);
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
      break;
    case VarDecl::Extern:
    case VarDecl::PrivateExtern:
      // todo: common
      break;
    }
  }

  // Emit global variable debug information.
  CGDebugInfo *DI = getDebugInfo();
  if(DI) {
    if(D->getLocation().isValid())
      DI->setLocation(D->getLocation());
    DI->EmitGlobalVariable(GV, D);
  }
}

llvm::GlobalValue *
CodeGenModule::EmitForwardFunctionDefinition(const FunctionDecl *D) {
  // FIXME: param attributes for sext/zext etc.
  if (const AliasAttr *AA = D->getAttr<AliasAttr>()) {
    assert(!D->getBody() && "Unexpected alias attr on function with body.");
    
    const std::string& aliaseeName = AA->getAliasee();
    llvm::Function *aliasee = getModule().getFunction(aliaseeName);
    llvm::GlobalValue *alias = new llvm::GlobalAlias(aliasee->getType(),
                                              llvm::Function::ExternalLinkage,
                                                     D->getName(),
                                                     aliasee,
                                                     &getModule());
    SetGlobalValueAttributes(D, alias);
    return alias;
  } else {
    const llvm::Type *Ty = getTypes().ConvertType(D->getType());
    const llvm::FunctionType *FTy = cast<llvm::FunctionType>(Ty);
    llvm::Function *F = llvm::Function::Create(FTy, 
                                               llvm::Function::ExternalLinkage,
                                               D->getName(), &getModule());
    
    SetFunctionAttributes(D, F, FTy);
    return F;
  }
}

llvm::Constant *CodeGenModule::GetAddrOfFunction(const FunctionDecl *D) {
  QualType ASTTy = D->getType();
  const llvm::Type *Ty = getTypes().ConvertTypeForMem(ASTTy);
  const llvm::Type *PTy = llvm::PointerType::get(Ty, ASTTy.getAddressSpace());
  
  // Lookup the entry, lazily creating it if necessary.
  llvm::GlobalValue *&Entry = GlobalDeclMap[D->getName()];
  if (!Entry)
    Entry = EmitForwardFunctionDefinition(D);

  return llvm::ConstantExpr::getBitCast(Entry, PTy);
}

void CodeGenModule::EmitGlobalFunctionDefinition(const FunctionDecl *D) {
  llvm::GlobalValue *&Entry = GlobalDeclMap[D->getName()];
  if (!Entry) {
    Entry = EmitForwardFunctionDefinition(D);
  } else {
    // If the types mismatch then we have to rewrite the definition.
    const llvm::Type *Ty = getTypes().ConvertType(D->getType());
    if (Entry->getType() != llvm::PointerType::getUnqual(Ty)) {
      // Otherwise, we have a definition after a prototype with the wrong type.
      // F is the Function* for the one with the wrong type, we must make a new
      // Function* and update everything that used F (a declaration) with the new
      // Function* (which will be a definition).
      //
      // This happens if there is a prototype for a function (e.g. "int f()") and
      // then a definition of a different type (e.g. "int f(int x)").  Start by
      // making a new function of the correct type, RAUW, then steal the name.
      llvm::GlobalValue *NewFn = EmitForwardFunctionDefinition(D);
      NewFn->takeName(Entry);
      
      // Replace uses of F with the Function we will endow with a body.
      llvm::Constant *NewPtrForOldDecl = 
        llvm::ConstantExpr::getBitCast(NewFn, Entry->getType());
      Entry->replaceAllUsesWith(NewPtrForOldDecl);
            
      // Ok, delete the old function now, which is dead.
      // FIXME: Add GlobalValue->eraseFromParent().
      assert(Entry->isDeclaration() && "Shouldn't replace non-declaration");
      if (llvm::Function *F = dyn_cast<llvm::Function>(Entry)) {
        F->eraseFromParent();
      } else if (llvm::GlobalAlias *GA = dyn_cast<llvm::GlobalAlias>(Entry)) {
        GA->eraseFromParent();
      } else {
        assert(0 && "Invalid global variable type.");
      }
      
      Entry = NewFn;
    }
  }

  if (D->getAttr<AliasAttr>()) {
    ;
  } else {
    llvm::Function *Fn = cast<llvm::Function>(Entry);    
    CodeGenFunction(*this).GenerateCode(D, Fn);

    if (const ConstructorAttr *CA = D->getAttr<ConstructorAttr>()) {
      AddGlobalCtor(Fn, CA->getPriority());
    } else if (const DestructorAttr *DA = D->getAttr<DestructorAttr>()) {
      AddGlobalDtor(Fn, DA->getPriority());
    }
  }
}

void CodeGenModule::UpdateCompletedType(const TagDecl *TD) {
  // Make sure that this type is translated.
  Types.UpdateCompletedType(TD);
}


/// getBuiltinLibFunction
llvm::Function *CodeGenModule::getBuiltinLibFunction(unsigned BuiltinID) {
  if (BuiltinID > BuiltinFunctions.size())
    BuiltinFunctions.resize(BuiltinID);
  
  // Cache looked up functions.  Since builtin id #0 is invalid we don't reserve
  // a slot for it.
  assert(BuiltinID && "Invalid Builtin ID");
  llvm::Function *&FunctionSlot = BuiltinFunctions[BuiltinID-1];
  if (FunctionSlot)
    return FunctionSlot;
  
  assert(Context.BuiltinInfo.isLibFunction(BuiltinID) && "isn't a lib fn");
  
  // Get the name, skip over the __builtin_ prefix.
  const char *Name = Context.BuiltinInfo.GetName(BuiltinID)+10;
  
  // Get the type for the builtin.
  QualType Type = Context.BuiltinInfo.GetBuiltinType(BuiltinID, Context);
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

  // FIXME: param attributes for sext/zext etc.
  return FunctionSlot = 
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
  llvm::Intrinsic::ID IID;
  switch (Context.Target.getPointerWidth(0)) {
  default: assert(0 && "Unknown ptr width");
  case 32: IID = llvm::Intrinsic::memcpy_i32; break;
  case 64: IID = llvm::Intrinsic::memcpy_i64; break;
  }
  return MemCpyFn = getIntrinsic(IID);
}

llvm::Function *CodeGenModule::getMemMoveFn() {
  if (MemMoveFn) return MemMoveFn;
  llvm::Intrinsic::ID IID;
  switch (Context.Target.getPointerWidth(0)) {
  default: assert(0 && "Unknown ptr width");
  case 32: IID = llvm::Intrinsic::memmove_i32; break;
  case 64: IID = llvm::Intrinsic::memmove_i64; break;
  }
  return MemMoveFn = getIntrinsic(IID);
}

llvm::Function *CodeGenModule::getMemSetFn() {
  if (MemSetFn) return MemSetFn;
  llvm::Intrinsic::ID IID;
  switch (Context.Target.getPointerWidth(0)) {
  default: assert(0 && "Unknown ptr width");
  case 32: IID = llvm::Intrinsic::memset_i32; break;
  case 64: IID = llvm::Intrinsic::memset_i64; break;
  }
  return MemSetFn = getIntrinsic(IID);
}

// FIXME: This needs moving into an Apple Objective-C runtime class
llvm::Constant *CodeGenModule::
GetAddrOfConstantCFString(const std::string &str) {
  llvm::StringMapEntry<llvm::Constant *> &Entry = 
    CFConstantStringMap.GetOrCreateValue(&str[0], &str[str.length()]);
  
  if (Entry.getValue())
    return Entry.getValue();
  
  std::vector<llvm::Constant*> Fields;
  
  if (!CFConstantStringClassRef) {
    const llvm::Type *Ty = getTypes().ConvertType(getContext().IntTy);
    Ty = llvm::ArrayType::get(Ty, 0);
  
    CFConstantStringClassRef = 
      new llvm::GlobalVariable(Ty, false,
                               llvm::GlobalVariable::ExternalLinkage, 0, 
                               "__CFConstantStringClassReference", 
                               &getModule());
  }
  
  // Class pointer.
  llvm::Constant *Zero = llvm::Constant::getNullValue(llvm::Type::Int32Ty);
  llvm::Constant *Zeros[] = { Zero, Zero };
  llvm::Constant *C = 
    llvm::ConstantExpr::getGetElementPtr(CFConstantStringClassRef, Zeros, 2);
  Fields.push_back(C);
  
  // Flags.
  const llvm::Type *Ty = getTypes().ConvertType(getContext().IntTy);
  Fields.push_back(llvm::ConstantInt::get(Ty, 1992));
    
  // String pointer.
  C = llvm::ConstantArray::get(str);
  C = new llvm::GlobalVariable(C->getType(), true, 
                               llvm::GlobalValue::InternalLinkage,
                               C, ".str", &getModule());
  
  C = llvm::ConstantExpr::getGetElementPtr(C, Zeros, 2);
  Fields.push_back(C);
  
  // String length.
  Ty = getTypes().ConvertType(getContext().LongTy);
  Fields.push_back(llvm::ConstantInt::get(Ty, str.length()));
  
  // The struct.
  Ty = getTypes().ConvertType(getContext().getCFConstantStringType());
  C = llvm::ConstantStruct::get(cast<llvm::StructType>(Ty), Fields);
  llvm::GlobalVariable *GV = 
    new llvm::GlobalVariable(C->getType(), true, 
                             llvm::GlobalVariable::InternalLinkage, 
                             C, "", &getModule());
  GV->setSection("__DATA,__cfstring");
  Entry.setValue(GV);
  return GV;
}

/// getStringForStringLiteral - Return the appropriate bytes for a
/// string literal, properly padded to match the literal type.
std::string CodeGenModule::getStringForStringLiteral(const StringLiteral *E) {
  assert(!E->isWide() && "FIXME: Wide strings not supported yet!");
  const char *StrData = E->getStrData();
  unsigned Len = E->getByteLength();

  const ConstantArrayType *CAT =
    getContext().getAsConstantArrayType(E->getType());
  assert(CAT && "String isn't pointer or array!");
  
  // Resize the string to the right size
  // FIXME: What about wchar_t strings?
  std::string Str(StrData, StrData+Len);
  uint64_t RealLen = CAT->getSize().getZExtValue();
  Str.resize(RealLen, '\0');
  
  return Str;
}

/// GenerateWritableString -- Creates storage for a string literal.
static llvm::Constant *GenerateStringLiteral(const std::string &str, 
                                             bool constant,
                                             CodeGenModule &CGM) {
  // Create Constant for this string literal
  llvm::Constant *C = llvm::ConstantArray::get(str);
  
  // Create a global variable for this string
  C = new llvm::GlobalVariable(C->getType(), constant, 
                               llvm::GlobalValue::InternalLinkage,
                               C, ".str", &CGM.getModule());
  return C;
}

/// CodeGenModule::GetAddrOfConstantString -- returns a pointer to the character
/// array containing the literal.  The result is pointer to array type.
llvm::Constant *CodeGenModule::GetAddrOfConstantString(const std::string &str) {
  // Don't share any string literals if writable-strings is turned on.
  if (Features.WritableStrings)
    return GenerateStringLiteral(str, false, *this);
  
  llvm::StringMapEntry<llvm::Constant *> &Entry = 
  ConstantStringMap.GetOrCreateValue(&str[0], &str[str.length()]);

  if (Entry.getValue())
      return Entry.getValue();

  // Create a global variable for this.
  llvm::Constant *C = GenerateStringLiteral(str, true, *this);
  Entry.setValue(C);
  return C;
}
