//===- CIndexHigh.cpp - Higher level API functions ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "IndexingContext.h"
#include "CXTranslationUnit.h"
#include "CIndexDiagnostic.h"

#include "clang/Frontend/ASTUnit.h"
#include "clang/AST/DeclObjC.h"

using namespace clang;
using namespace cxindex;
using namespace cxcursor;

const char *IndexingContext::StrAdapter::toCStr(StringRef Str) {
  if (Str.empty())
    return "";
  if (Str.data()[Str.size()] == '\0')
    return Str.data();
  Scratch += Str;
  Scratch.push_back('\0');
  return Scratch.data() + (Scratch.size() - Str.size() - 1);
}

void IndexingContext::setASTContext(ASTContext &ctx) {
  Ctx = &ctx;
  static_cast<ASTUnit*>(CXTU->TUData)->setASTContext(&ctx);
}

void IndexingContext::enteredMainFile(const FileEntry *File) {
  if (File && CB.enteredMainFile) {
    CXIdxClientFile idxFile = CB.enteredMainFile(ClientData, (CXFile)File, 0);
    FileMap[File] = idxFile;
  }
}

void IndexingContext::ppIncludedFile(SourceLocation hashLoc,
                                     StringRef filename,
                                     const FileEntry *File,
                                     bool isImport, bool isAngled) {
  if (!CB.ppIncludedFile)
    return;

  StrAdapter SA(*this);
  CXIdxIncludedFileInfo Info = { getIndexLoc(hashLoc),
                                 SA.toCStr(filename),
                                 (CXFile)File,
                                 isImport, isAngled };
  CXIdxClientFile idxFile = CB.ppIncludedFile(ClientData, &Info);
  FileMap[File] = idxFile;
}

void IndexingContext::ppMacroDefined(SourceLocation Loc, StringRef Name,
                                     SourceLocation DefBegin, unsigned Length,
                                     const void *OpaqueMacro) {
  if (!CB.ppMacroDefined)
    return;

  StrAdapter SA(*this);
  CXIdxMacroInfo MacroInfo =  { getIndexLoc(Loc), SA.toCStr(Name) }; 
  CXIdxMacroDefinedInfo Info = { &MacroInfo,
                                 getIndexLoc(DefBegin), Length };
  CXIdxClientMacro idxMacro = CB.ppMacroDefined(ClientData, &Info);
  MacroMap[OpaqueMacro] = idxMacro;
}

void IndexingContext::ppMacroUndefined(SourceLocation Loc, StringRef Name,
                                       const void *OpaqueMacro) {
  if (!CB.ppMacroUndefined)
    return;

  StrAdapter SA(*this);
  CXIdxMacroUndefinedInfo Info = { getIndexLoc(Loc),
                                   SA.toCStr(Name), 0 };
  CB.ppMacroUndefined(ClientData, &Info);
}

void IndexingContext::ppMacroExpanded(SourceLocation Loc, StringRef Name,
                                      const void *OpaqueMacro) {
  if (!CB.ppMacroExpanded)
    return;

  StrAdapter SA(*this);
  CXIdxMacroExpandedInfo Info = { getIndexLoc(Loc),
                                   SA.toCStr(Name), 0 };
  CB.ppMacroExpanded(ClientData, &Info);
}

void IndexingContext::invokeStartedTranslationUnit() {
  CXIdxClientContainer idxCont = 0;
  if (CB.startedTranslationUnit)
    idxCont = CB.startedTranslationUnit(ClientData, 0);
  addContainerInMap(Ctx->getTranslationUnitDecl(), idxCont);
}

void IndexingContext::invokeFinishedTranslationUnit() {
  endContainer(Ctx->getTranslationUnitDecl());
}

void IndexingContext::handleDiagnostic(const StoredDiagnostic &StoredDiag) {
  if (!CB.diagnostic)
    return;

  CXStoredDiagnostic CXDiag(StoredDiag, Ctx->getLangOptions());
  CB.diagnostic(ClientData, &CXDiag, 0);
}

void IndexingContext::handleDecl(const NamedDecl *D,
                                 SourceLocation Loc, CXCursor Cursor,
                                 bool isRedeclaration, bool isDefinition,
                                 DeclInfo &DInfo) {
  if (!CB.indexDeclaration)
    return;

  StrAdapter SA(*this);
  getEntityInfo(D, DInfo.CXEntInfo, SA);
  DInfo.entityInfo = &DInfo.CXEntInfo;
  DInfo.cursor = Cursor;
  DInfo.loc = getIndexLoc(Loc);
  DInfo.container = getIndexContainer(D);
  DInfo.isRedeclaration = isRedeclaration;
  DInfo.isDefinition = isDefinition;

  CXIdxClientEntity
    clientEnt = CB.indexDeclaration(ClientData, &DInfo);

  if (!isRedeclaration)
    addEntityInMap(D, clientEnt);
}

void IndexingContext::handleObjCContainer(const ObjCContainerDecl *D,
                                          SourceLocation Loc, CXCursor Cursor,
                                          bool isForwardRef,
                                          bool isRedeclaration,
                                          bool isImplementation,
                                          ObjCContainerDeclInfo &ContDInfo) {
  ContDInfo.CXObjCContDeclInfo.declInfo = &ContDInfo;
  if (isForwardRef)
    ContDInfo.CXObjCContDeclInfo.kind = CXIdxObjCContainer_ForwardRef;
  else if (isImplementation)
    ContDInfo.CXObjCContDeclInfo.kind = CXIdxObjCContainer_Implementation;
  else
    ContDInfo.CXObjCContDeclInfo.kind = CXIdxObjCContainer_Interface;

  handleDecl(D, Loc, Cursor,
             isRedeclaration, /*isDefinition=*/!isForwardRef, ContDInfo);
}

void IndexingContext::handleFunction(const FunctionDecl *D) {
  DeclInfo DInfo;
  handleDecl(D, D->getLocation(), getCursor(D),
             !D->isFirstDeclaration(), D->isThisDeclarationADefinition(),
             DInfo);
}

void IndexingContext::handleVar(const VarDecl *D) {
  DeclInfo DInfo;
  handleDecl(D, D->getLocation(), getCursor(D),
             !D->isFirstDeclaration(), D->isThisDeclarationADefinition(),
             DInfo);
}

void IndexingContext::handleField(const FieldDecl *D) {
  DeclInfo DInfo;
  handleDecl(D, D->getLocation(), getCursor(D),
             /*isRedeclaration=*/false, /*isDefinition=*/false, DInfo);
}

void IndexingContext::handleEnumerator(const EnumConstantDecl *D) {
  DeclInfo DInfo;
  handleDecl(D, D->getLocation(), getCursor(D),
             /*isRedeclaration=*/false, /*isDefinition=*/true, DInfo);
}

void IndexingContext::handleTagDecl(const TagDecl *D) {
  TagDeclInfo TagDInfo;
  TagDInfo.CXTagDeclInfo.declInfo = &TagDInfo;
  TagDInfo.CXTagDeclInfo.isAnonymous = D->getIdentifier() == 0;
  handleDecl(D, D->getLocation(), getCursor(D),
             !D->isFirstDeclaration(), D->isThisDeclarationADefinition(),
             TagDInfo);
}

void IndexingContext::handleTypedef(const TypedefDecl *D) {
  DeclInfo DInfo;
  handleDecl(D, D->getLocation(), getCursor(D),
             !D->isFirstDeclaration(), /*isDefinition=*/true, DInfo);
}

void IndexingContext::handleObjCClass(const ObjCClassDecl *D) {
  ObjCContainerDeclInfo ContDInfo;
  const ObjCClassDecl::ObjCClassRef *Ref = D->getForwardDecl();
  ObjCInterfaceDecl *IFaceD = Ref->getInterface();
  SourceLocation Loc = Ref->getLocation();
  bool isRedeclaration = IFaceD->getLocation() != Loc;
  handleObjCContainer(IFaceD, Loc, MakeCursorObjCClassRef(IFaceD, Loc, CXTU),
                      /*isForwardRef=*/true, isRedeclaration,
                      /*isImplementation=*/false, ContDInfo);
}

void IndexingContext::handleObjCInterface(const ObjCInterfaceDecl *D) {
  ObjCContainerDeclInfo ContDInfo;
  handleObjCContainer(D, D->getLocation(), getCursor(D),
                      /*isForwardRef=*/false,
                      /*isRedeclaration=*/D->isInitiallyForwardDecl(),
                      /*isImplementation=*/false, ContDInfo);
}

void IndexingContext::handleObjCImplementation(
                                              const ObjCImplementationDecl *D) {
  ObjCContainerDeclInfo ContDInfo;
  const ObjCInterfaceDecl *Class = D->getClassInterface();
  handleObjCContainer(Class, D->getLocation(), getCursor(D),
                      /*isForwardRef=*/false,
                      /*isRedeclaration=*/!Class->isImplicitInterfaceDecl(),
                      /*isImplementation=*/true, ContDInfo);
}

void IndexingContext::handleObjCForwardProtocol(const ObjCProtocolDecl *D,
                                                SourceLocation Loc,
                                                bool isRedeclaration) {
  ObjCContainerDeclInfo ContDInfo;
  handleObjCContainer(D, Loc, MakeCursorObjCProtocolRef(D, Loc, CXTU),
                      /*isForwardRef=*/true,
                      isRedeclaration,
                      /*isImplementation=*/false, ContDInfo);
}

void IndexingContext::handleObjCProtocol(const ObjCProtocolDecl *D) {
  ObjCContainerDeclInfo ContDInfo;
  handleObjCContainer(D, D->getLocation(), getCursor(D),
                      /*isForwardRef=*/false,
                      /*isRedeclaration=*/D->isInitiallyForwardDecl(),
                      /*isImplementation=*/false, ContDInfo);
}

void IndexingContext::defineObjCInterface(const ObjCInterfaceDecl *D) {
  if (!CB.defineObjCClass)
    return;

  StrAdapter SA(*this);
  CXIdxObjCBaseClassInfo BaseClass;
  CXIdxEntityInfo BaseEntity;
  if (D->getSuperClass()) {
    getEntityInfo(D->getSuperClass(), BaseEntity, SA);
    BaseClass.objcClass = &BaseEntity;
    BaseClass.loc = getIndexLoc(D->getSuperClassLoc());
  }

  SmallVector<CXIdxObjCProtocolRefInfo, 4> ProtInfos;
  SmallVector<CXIdxEntityInfo, 4> ProtEntities;
  ObjCInterfaceDecl::protocol_loc_iterator LI = D->protocol_loc_begin();
  for (ObjCInterfaceDecl::protocol_iterator
         I = D->protocol_begin(), E = D->protocol_end(); I != E; ++I, ++LI) {
    SourceLocation Loc = *LI;
    ObjCProtocolDecl *PD = *I;
    ProtEntities.push_back(CXIdxEntityInfo());
    getEntityInfo(PD, ProtEntities.back(), SA);
    CXIdxObjCProtocolRefInfo ProtInfo = { 0, getIndexLoc(Loc) };
    ProtInfos.push_back(ProtInfo);
  }
  
  for (unsigned i = 0, e = ProtInfos.size(); i != e; ++i)
    ProtInfos[i].protocol = &ProtEntities[i];

  SmallVector<CXIdxObjCProtocolRefInfo *, 4> Prots;
  for (unsigned i = 0, e = Prots.size(); i != e; ++i)
    Prots.push_back(&ProtInfos[i]);
  
  CXIdxEntityInfo ClassEntity;
  getEntityInfo(D, ClassEntity, SA);
  CXIdxObjCClassDefineInfo Info = { getCursor(D),
                                    &ClassEntity, 
                                    getIndexContainerForDC(D),
                                    D->getSuperClass() ? &BaseClass : 0,
                                    Prots.data(),
                                    static_cast<unsigned>(Prots.size()) };
  CB.defineObjCClass(ClientData, &Info);
}

void IndexingContext::handleObjCCategory(const ObjCCategoryDecl *D) {
  ObjCCategoryDeclInfo CatDInfo;
  CXIdxEntityInfo ClassEntity;
  StrAdapter SA(*this);
  getEntityInfo(D->getClassInterface(), ClassEntity, SA);

  CatDInfo.CXObjCCatDeclInfo.containerInfo = &CatDInfo.CXObjCContDeclInfo;
  CatDInfo.CXObjCCatDeclInfo.objcClass = &ClassEntity;
  handleObjCContainer(D, D->getLocation(), getCursor(D),
                      /*isForwardRef=*/false,
                      /*isRedeclaration=*/false,
                      /*isImplementation=*/false, CatDInfo);
}

void IndexingContext::handleObjCCategoryImpl(const ObjCCategoryImplDecl *D) {
  const ObjCCategoryDecl *CatD = D->getCategoryDecl();
  ObjCCategoryDeclInfo CatDInfo;
  CXIdxEntityInfo ClassEntity;
  StrAdapter SA(*this);
  getEntityInfo(CatD->getClassInterface(), ClassEntity, SA);

  CatDInfo.CXObjCCatDeclInfo.containerInfo = &CatDInfo.CXObjCContDeclInfo;
  CatDInfo.CXObjCCatDeclInfo.objcClass = &ClassEntity;
  handleObjCContainer(CatD, D->getLocation(), getCursor(D),
                      /*isForwardRef=*/false,
                      /*isRedeclaration=*/true,
                      /*isImplementation=*/true, CatDInfo);
}

void IndexingContext::handleObjCMethod(const ObjCMethodDecl *D) {
  DeclInfo DInfo;
  handleDecl(D, D->getLocation(), getCursor(D),
             !D->isCanonicalDecl(), D->isThisDeclarationADefinition(),
             DInfo);
}

void IndexingContext::handleObjCProperty(const ObjCPropertyDecl *D) {
  DeclInfo DInfo;
  handleDecl(D, D->getLocation(), getCursor(D),
             /*isRedeclaration=*/false, /*isDefinition=*/false,
             DInfo);
}

void IndexingContext::handleReference(const NamedDecl *D, SourceLocation Loc,
                                      const NamedDecl *Parent,
                                      const DeclContext *DC,
                                      const Expr *E,
                                      CXIdxEntityRefKind Kind) {
  if (Loc.isInvalid())
    return;
  if (!CB.indexEntityReference)
    return;
  if (isNotFromSourceFile(D->getLocation()))
    return;

  StrAdapter SA(*this);
  CXCursor Cursor = E ? MakeCXCursor(const_cast<Expr*>(E),
                                     const_cast<Decl*>(cast<Decl>(DC)), CXTU)
                      : getRefCursor(D, Loc);

  CXIdxEntityInfo RefEntity, ParentEntity;
  getEntityInfo(D, RefEntity, SA);
  getEntityInfo(Parent, ParentEntity, SA);
  CXIdxEntityRefInfo Info = { Cursor,
                              getIndexLoc(Loc),
                              &RefEntity,
                              &ParentEntity,
                              getIndexContainerForDC(DC),
                              Kind };
  CB.indexEntityReference(ClientData, &Info);
}

void IndexingContext::startContainer(const NamedDecl *D, bool isStmtBody,
                                     const DeclContext *DC) {
  if (!CB.startedContainer)
    return;

  if (!DC)
    DC = cast<DeclContext>(D);
  
  StrAdapter SA(*this);
  CXIdxEntityInfo Entity;
  getEntityInfo(D, Entity, SA);
  CXIdxContainerInfo Info;
  Info.entity = &Entity;
  Info.cursor = getCursor(D);
  Info.loc = getIndexLoc(D->getLocation());
  Info.isObjCImpl = isa<ObjCImplDecl>(D);

  CXIdxClientContainer clientCont = CB.startedContainer(ClientData, &Info);
  addContainerInMap(DC, clientCont);
}

void IndexingContext::endContainer(const DeclContext *DC) {
  if (CB.endedContainer) {
    CXIdxEndContainerInfo Info = { getIndexContainerForDC(DC),
                                   getIndexLoc(cast<Decl>(DC)->getLocEnd()) };
    CB.endedContainer(ClientData, &Info);
  }
}

bool IndexingContext::isNotFromSourceFile(SourceLocation Loc) const {
  if (Loc.isInvalid())
    return true;
  SourceManager &SM = Ctx->getSourceManager();
  SourceLocation FileLoc = SM.getFileLoc(Loc);
  FileID FID = SM.getFileID(FileLoc);
  return SM.getFileEntryForID(FID) == 0;
}

void IndexingContext::addContainerInMap(const DeclContext *DC,
                                        CXIdxClientContainer container) {
  assert(getScopedContext(DC) == DC);
  ContainerMapTy::iterator I = ContainerMap.find(DC);
  if (I == ContainerMap.end()) {
    if (container)
      ContainerMap[DC] = container;
    return;
  }
  // Allow changing the container of a previously seen DeclContext so we
  // can handle invalid user code, like a function re-definition.
  if (container)
    I->second = container;
  else
    ContainerMap.erase(I);
}

void IndexingContext::addEntityInMap(const NamedDecl *D,
                                     CXIdxClientEntity entity) {
  assert(getEntityDecl(D) == D &&
         "Tried to add a non-entity (canonical) decl");
  assert(EntityMap.find(D) == EntityMap.end());
  if (entity || D->isFromASTFile())
    EntityMap[D] = entity;
}

CXIdxClientEntity IndexingContext::getClientEntity(const NamedDecl *D) {
  if (!D)
    return 0;
  D = getEntityDecl(D);
  EntityMapTy::const_iterator I = EntityMap.find(D);
  if (I != EntityMap.end())
    return I->second;

  if (!D->isFromASTFile()) {
    //assert(0 && "Entity not in map");
    return 0;
  }

  StrAdapter SA(*this);
  
  CXIdxClientEntity idxEntity = 0;
  if (CB.importedEntity) {
    CXIdxEntityInfo EntityInfo;
    getEntityInfo(D, EntityInfo, SA);
    CXIdxImportedEntityInfo Info = { &EntityInfo,
                                     getCursor(D),
                                     getIndexLoc(D->getLocation()),
                                     /*CXIdxASTFile*/0 };
    idxEntity = CB.importedEntity(ClientData, &Info);
  }
  addEntityInMap(D, idxEntity);
  return idxEntity;
}

const NamedDecl *IndexingContext::getEntityDecl(const NamedDecl *D) const {
  assert(D);
  D = cast<NamedDecl>(D->getCanonicalDecl());

  if (const ObjCCategoryDecl *Cat = dyn_cast<ObjCCategoryDecl>(D)) {
    if (Cat->IsClassExtension())
      return getEntityDecl(Cat->getClassInterface());

  } else if (const ObjCImplementationDecl *
               ImplD = dyn_cast<ObjCImplementationDecl>(D)) {
    return getEntityDecl(ImplD->getClassInterface());

  } else if (const ObjCCategoryImplDecl *
               CatImplD = dyn_cast<ObjCCategoryImplDecl>(D)) {
    return getEntityDecl(CatImplD->getCategoryDecl());
  }

  return D;
}

const DeclContext *
IndexingContext::getScopedContext(const DeclContext *DC) const {
  // Local contexts are ignored for indexing.
  const DeclContext *FuncCtx = cast<Decl>(DC)->getParentFunctionOrMethod();
  if (FuncCtx)
    return FuncCtx;

  // We consider enums always scoped for indexing.
  if (isa<TagDecl>(DC))
    return DC;

  if (const NamespaceDecl *NS = dyn_cast<NamespaceDecl>(DC)) {
    if (NS->isAnonymousNamespace())
      return getScopedContext(NS->getParent());
    return NS;
  }

  return DC->getRedeclContext();
}

CXIdxClientContainer
IndexingContext::getIndexContainerForDC(const DeclContext *DC) const {
  DC = getScopedContext(DC);
  ContainerMapTy::const_iterator I = ContainerMap.find(DC);
//  assert(I != ContainerMap.end() &&
//         "Failed to include a scoped context in the container map");
  return I->second;
}

CXIdxClientFile IndexingContext::getIndexFile(const FileEntry *File) {
  if (!File)
    return 0;

  FileMapTy::iterator FI = FileMap.find(File);
  if (FI != FileMap.end())
    return FI->second;

  return 0;
}

CXIdxLoc IndexingContext::getIndexLoc(SourceLocation Loc) const {
  CXIdxLoc idxLoc =  { {0, 0}, 0 };
  if (Loc.isInvalid())
    return idxLoc;

  idxLoc.ptr_data[0] = (void*)this;
  idxLoc.int_data = Loc.getRawEncoding();
  return idxLoc;
}

void IndexingContext::translateLoc(SourceLocation Loc,
                                   CXIdxClientFile *indexFile, CXFile *file,
                                   unsigned *line, unsigned *column,
                                   unsigned *offset) {
  if (Loc.isInvalid())
    return;

  SourceManager &SM = Ctx->getSourceManager();
  Loc = SM.getFileLoc(Loc);

  std::pair<FileID, unsigned> LocInfo = SM.getDecomposedLoc(Loc);
  FileID FID = LocInfo.first;
  unsigned FileOffset = LocInfo.second;

  if (FID.isInvalid())
    return;
  
  const FileEntry *FE = SM.getFileEntryForID(FID);
  if (indexFile)
    *indexFile = getIndexFile(FE);
  if (file)
    *file = (void *)FE;
  if (line)
    *line = SM.getLineNumber(FID, FileOffset);
  if (column)
    *column = SM.getColumnNumber(FID, FileOffset);
  if (offset)
    *offset = FileOffset;
}

void IndexingContext::getEntityInfo(const NamedDecl *D,
                                     CXIdxEntityInfo &EntityInfo,
                                     StrAdapter &SA) {
  D = getEntityDecl(D);
  EntityInfo.kind = CXIdxEntity_Unexposed;
  EntityInfo.clientEntity = getClientEntity(D);

  if (const TagDecl *TD = dyn_cast<TagDecl>(D)) {
    switch (TD->getTagKind()) {
    case TTK_Struct:
      EntityInfo.kind = CXIdxEntity_Struct; break;
    case TTK_Union:
      EntityInfo.kind = CXIdxEntity_Union; break;
    case TTK_Class:
      EntityInfo.kind = CXIdxEntity_CXXClass; break;
    case TTK_Enum:
      EntityInfo.kind = CXIdxEntity_Enum; break;
    }

  } else {
    switch (D->getKind()) {
    case Decl::Typedef:
      EntityInfo.kind = CXIdxEntity_Typedef; break;
    case Decl::Function:
      EntityInfo.kind = CXIdxEntity_Function; break;
    case Decl::Var:
      EntityInfo.kind = CXIdxEntity_Variable; break;
    case Decl::Field:
      EntityInfo.kind = CXIdxEntity_Field; break;
    case Decl::EnumConstant:
      EntityInfo.kind = CXIdxEntity_EnumConstant; break;
    case Decl::ObjCInterface:
      EntityInfo.kind = CXIdxEntity_ObjCClass; break;
    case Decl::ObjCProtocol:
      EntityInfo.kind = CXIdxEntity_ObjCProtocol; break;
    case Decl::ObjCCategory:
      EntityInfo.kind = CXIdxEntity_ObjCCategory; break;
    case Decl::ObjCMethod:
      EntityInfo.kind = CXIdxEntity_ObjCMethod; break;
    case Decl::ObjCProperty:
      EntityInfo.kind = CXIdxEntity_ObjCProperty; break;
    case Decl::ObjCIvar:
      EntityInfo.kind = CXIdxEntity_ObjCIvar; break;
    default:
      break;
    }
  }

  if (IdentifierInfo *II = D->getIdentifier()) {
    EntityInfo.name = SA.toCStr(II->getName());

  } else if (isa<RecordDecl>(D) || isa<NamespaceDecl>(D)) {
    EntityInfo.name = 0; // anonymous record/namespace.

  } else {
    unsigned Begin = SA.getCurSize();
    {
      llvm::raw_svector_ostream OS(SA.getBuffer());
      D->printName(OS);
    }
    EntityInfo.name = SA.getCStr(Begin);
  }

  {
    unsigned Begin = SA.getCurSize();
    bool Ignore = getDeclCursorUSR(D, SA.getBuffer());
    if (Ignore) {
      EntityInfo.USR = "";
    } else {
      EntityInfo.USR = SA.getCStr(Begin);
    }
  }
}

CXCursor IndexingContext::getRefCursor(const NamedDecl *D, SourceLocation Loc) {
  if (const TypeDecl *TD = dyn_cast<TypeDecl>(D))
    return MakeCursorTypeRef(TD, Loc, CXTU);
  if (const ObjCInterfaceDecl *ID = dyn_cast<ObjCInterfaceDecl>(D))
    return MakeCursorObjCClassRef(ID, Loc, CXTU);
  if (const ObjCProtocolDecl *PD = dyn_cast<ObjCProtocolDecl>(D))
    return MakeCursorObjCProtocolRef(PD, Loc, CXTU);
  
  //assert(0 && "not yet");
  return clang_getNullCursor();
}
