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

void IndexingContext::ppIncludedFile(SourceLocation hashLoc,
                                     StringRef filename,
                                     const FileEntry *File,
                                     bool isImport, bool isAngled) {
  if (!CB.ppIncludedFile)
    return;

  StrAdapter SA(this);
  CXIdxIncludedFileInfo Info = { getIndexLoc(hashLoc),
                                 SA.toCStr(filename),
                                 getIndexFile(File),
                                 isImport, isAngled };
  CB.ppIncludedFile(ClientData, &Info);
}

void IndexingContext::ppMacroDefined(SourceLocation Loc, StringRef Name,
                                     SourceLocation DefBegin, unsigned Length,
                                     const void *OpaqueMacro) {
  if (!CB.ppMacroDefined)
    return;

  StrAdapter SA(this);
  CXIdxMacroInfo MacroInfo =  { getIndexLoc(Loc), SA.toCStr(Name) }; 
  CXIdxMacroDefinedInfo Info = { &MacroInfo,
                                 getIndexLoc(DefBegin), Length };
  CXIdxMacro idxMacro = CB.ppMacroDefined(ClientData, &Info);
  MacroMap[OpaqueMacro] = idxMacro;
}

void IndexingContext::ppMacroUndefined(SourceLocation Loc, StringRef Name,
                                       const void *OpaqueMacro) {
  if (!CB.ppMacroUndefined)
    return;

  StrAdapter SA(this);
  CXIdxMacroUndefinedInfo Info = { getIndexLoc(Loc),
                                   SA.toCStr(Name), 0 };
  CB.ppMacroUndefined(ClientData, &Info);
}

void IndexingContext::ppMacroExpanded(SourceLocation Loc, StringRef Name,
                                      const void *OpaqueMacro) {
  if (!CB.ppMacroExpanded)
    return;

  StrAdapter SA(this);
  CXIdxMacroExpandedInfo Info = { getIndexLoc(Loc),
                                   SA.toCStr(Name), 0 };
  CB.ppMacroExpanded(ClientData, &Info);
}

void IndexingContext::invokeStartedTranslationUnit() {
  CXIdxContainer idxCont = 0;
  if (CB.startedTranslationUnit)
    idxCont = CB.startedTranslationUnit(ClientData, 0);
  addContainerInMap(Ctx->getTranslationUnitDecl(), idxCont);
}

void IndexingContext::invokeFinishedTranslationUnit() {
  invokeEndedContainer(Ctx->getTranslationUnitDecl());
}

void IndexingContext::handleDiagnostic(const StoredDiagnostic &StoredDiag) {
  if (!CB.diagnostic)
    return;

  CXStoredDiagnostic CXDiag(StoredDiag, Ctx->getLangOptions());
  CB.diagnostic(ClientData, &CXDiag, 0);
}

void IndexingContext::handleFunction(const FunctionDecl *D) {
  StrAdapter SA(this);

  if (D->isFirstDeclaration()) {
    CXIdxEntity idxEntity = 0;
    if (CB.indexFunction) {
      CXIdxEntityInfo EntityInfo;
      CXIdxIndexedDeclInfo DeclInfo;
      CXIdxIndexedEntityInfo IdxEntityInfo;
      getIndexedEntityInfo(D, IdxEntityInfo, EntityInfo, DeclInfo, SA);
      CXIdxFunctionInfo Info = { &IdxEntityInfo,
                                 D->isThisDeclarationADefinition() };

      idxEntity = CB.indexFunction(ClientData, &Info);
    }

    addEntityInMap(D, idxEntity);

  } else {
    if (CB.indexFunctionRedeclaration) {
      CXIdxIndexedDeclInfo DeclInfo;
      CXIdxIndexedRedeclInfo RedeclInfo;
      getIndexedRedeclInfo(D, RedeclInfo, DeclInfo);
      CXIdxFunctionRedeclInfo Info = { &RedeclInfo,
                                       D->isThisDeclarationADefinition() };

      CB.indexFunctionRedeclaration(ClientData, &Info);
    }
  }
}

void IndexingContext::handleVar(const VarDecl *D) {
  StrAdapter SA(this);

  if (D->isFirstDeclaration()) {
    CXIdxEntity idxEntity = 0;
    if (CB.indexVariable) {
      CXIdxEntityInfo EntityInfo;
      CXIdxIndexedDeclInfo DeclInfo;
      CXIdxIndexedEntityInfo IdxEntityInfo;
      getIndexedEntityInfo(D, IdxEntityInfo, EntityInfo, DeclInfo, SA);
      CXIdxVariableInfo Info = { &IdxEntityInfo,
                                 D->isThisDeclarationADefinition() };

      idxEntity = CB.indexVariable(ClientData, &Info);
    }

    addEntityInMap(D, idxEntity);

  } else {
    if (CB.indexVariableRedeclaration) {
      CXIdxIndexedDeclInfo DeclInfo;
      CXIdxIndexedRedeclInfo RedeclInfo;
      getIndexedRedeclInfo(D, RedeclInfo, DeclInfo);
      CXIdxVariableRedeclInfo Info = { &RedeclInfo,
                                       D->isThisDeclarationADefinition() };

      CB.indexVariableRedeclaration(ClientData, &Info);
    }
  }
}

void IndexingContext::handleField(const FieldDecl *D) {
  StrAdapter SA(this);

  CXIdxEntity idxEntity = 0;
  if (CB.indexTypedef) {
    CXIdxEntityInfo EntityInfo;
    CXIdxIndexedDeclInfo DeclInfo;
    CXIdxIndexedEntityInfo IdxEntityInfo;
    getIndexedEntityInfo(D, IdxEntityInfo, EntityInfo, DeclInfo, SA);
    CXIdxFieldInfo Info = { &IdxEntityInfo };

    idxEntity = CB.indexField(ClientData, &Info);
  }

  addEntityInMap(D, idxEntity);
}

void IndexingContext::handleEnumerator(const EnumConstantDecl *D) {
  StrAdapter SA(this);

  CXIdxEntity idxEntity = 0;
  if (CB.indexTypedef) {
    CXIdxEntityInfo EntityInfo;
    CXIdxIndexedDeclInfo DeclInfo;
    CXIdxIndexedEntityInfo IdxEntityInfo;
    getIndexedEntityInfo(D, IdxEntityInfo, EntityInfo, DeclInfo, SA);
    CXIdxEnumeratorInfo Info = { &IdxEntityInfo };

    idxEntity = CB.indexEnumerator(ClientData, &Info);
  }

  addEntityInMap(D, idxEntity);
}

void IndexingContext::handleTagDecl(const TagDecl *D) {
  StrAdapter SA(this);

  if (D->isFirstDeclaration()) {
    CXIdxEntity idxEntity = 0;
    if (CB.indexTagType) {
      CXIdxEntityInfo EntityInfo;
      CXIdxIndexedDeclInfo DeclInfo;
      CXIdxIndexedEntityInfo IdxEntityInfo;
      getIndexedEntityInfo(D, IdxEntityInfo, EntityInfo, DeclInfo, SA);
      CXIdxTagTypeInfo Info = { &IdxEntityInfo,
                                 D->isThisDeclarationADefinition(),
                                 D->getIdentifier() == 0};

      idxEntity = CB.indexTagType(ClientData, &Info);
    }

    addEntityInMap(D, idxEntity);

  } else {
    if (CB.indexTagTypeRedeclaration) {
      CXIdxIndexedDeclInfo DeclInfo;
      CXIdxIndexedRedeclInfo RedeclInfo;
      getIndexedRedeclInfo(D, RedeclInfo, DeclInfo);
      CXIdxTagTypeRedeclInfo Info = { &RedeclInfo,
                                      D->isThisDeclarationADefinition() };

      CB.indexTagTypeRedeclaration(ClientData, &Info);
    }
  }
}

void IndexingContext::handleTypedef(const TypedefDecl *D) {
  StrAdapter SA(this);

  CXIdxEntity idxEntity = 0;
  if (CB.indexTypedef) {
    CXIdxEntityInfo EntityInfo;
    CXIdxIndexedDeclInfo DeclInfo;
    CXIdxIndexedEntityInfo IdxEntityInfo;
    getIndexedEntityInfo(D, IdxEntityInfo, EntityInfo, DeclInfo, SA);
    CXIdxTypedefInfo Info = { &IdxEntityInfo };

    idxEntity = CB.indexTypedef(ClientData, &Info);
  }

  addEntityInMap(D, idxEntity);
}

void IndexingContext::handleObjCInterface(const ObjCInterfaceDecl *D) {
  StrAdapter SA(this);

  CXIdxEntity idxEntity = 0;
  if (CB.indexObjCClass) {
    CXIdxEntityInfo EntityInfo;
    CXIdxIndexedDeclInfo DeclInfo;
    CXIdxIndexedEntityInfo IdxEntityInfo;
    getIndexedEntityInfo(D, IdxEntityInfo, EntityInfo, DeclInfo, SA);
    CXIdxObjCClassInfo Info = { &IdxEntityInfo,
                                D->isForwardDecl() };

    idxEntity = CB.indexObjCClass(ClientData, &Info);
  }

  addEntityInMap(D, idxEntity);
}

void IndexingContext::defineObjCInterface(const ObjCInterfaceDecl *D) {
  if (!CB.defineObjCClass)
    return;

  CXIdxObjCBaseClassInfo BaseClass = { getIndexEntity(D->getSuperClass()),
                                       getIndexLoc(D->getSuperClassLoc()) };
  if (D->getSuperClass()) {
    BaseClass.objcClass = getIndexEntity(D->getSuperClass());
    BaseClass.loc = getIndexLoc(D->getSuperClassLoc());
  }
  
  SmallVector<CXIdxObjCProtocolRefInfo, 4> ProtInfos;
  ObjCInterfaceDecl::protocol_loc_iterator LI = D->protocol_loc_begin();
  for (ObjCInterfaceDecl::protocol_iterator
         I = D->protocol_begin(), E = D->protocol_end(); I != E; ++I, ++LI) {
    SourceLocation Loc = *LI;
    ObjCProtocolDecl *PD = *I;
    CXIdxObjCProtocolRefInfo ProtInfo = { getIndexEntity(PD),
                                          getIndexLoc(Loc) };
    ProtInfos.push_back(ProtInfo);
  }

  SmallVector<CXIdxObjCProtocolRefInfo *, 4> Prots;
  for (unsigned i = 0, e = Prots.size(); i != e; ++i)
    Prots.push_back(&ProtInfos[i]);
  
  CXIdxObjCClassDefineInfo Info = { getCursor(D),
                                    getIndexEntity(D), 
                                    getIndexContainerForDC(D),
                                    D->getSuperClass() ? &BaseClass : 0,
                                    Prots.data(),
                                    static_cast<unsigned>(Prots.size()) };
  CB.defineObjCClass(ClientData, &Info);
}

void IndexingContext::handleObjCProtocol(const ObjCProtocolDecl *D) {
  StrAdapter SA(this);

  CXIdxEntity idxEntity = 0;
  if (CB.indexObjCProtocol) {
    CXIdxEntityInfo EntityInfo;
    CXIdxIndexedDeclInfo DeclInfo;
    CXIdxIndexedEntityInfo IdxEntityInfo;
    getIndexedEntityInfo(D, IdxEntityInfo, EntityInfo, DeclInfo, SA);
    CXIdxObjCProtocolInfo Info = { &IdxEntityInfo,
                                D->isForwardDecl() };

    idxEntity = CB.indexObjCProtocol(ClientData, &Info);
  }

  addEntityInMap(D, idxEntity);
}

void IndexingContext::handleObjCCategory(const ObjCCategoryDecl *D) {
  StrAdapter SA(this);

  CXIdxEntity idxEntity = 0;
  if (CB.indexObjCCategory) {
    CXIdxEntityInfo EntityInfo;
    CXIdxIndexedDeclInfo DeclInfo;
    CXIdxIndexedEntityInfo IdxEntityInfo;
    getIndexedEntityInfo(D, IdxEntityInfo, EntityInfo, DeclInfo, SA);
    CXIdxObjCCategoryInfo Info = { &IdxEntityInfo,
                                   getIndexEntity(D->getClassInterface()) };

    idxEntity = CB.indexObjCCategory(ClientData, &Info);
  }

  addEntityInMap(D, idxEntity);
}

void IndexingContext::handleObjCMethod(const ObjCMethodDecl *D) {
  StrAdapter SA(this);

  if (D->isCanonicalDecl()) {
    CXIdxEntity idxEntity = 0;
    if (CB.indexObjCMethod) {
      CXIdxEntityInfo EntityInfo;
      CXIdxIndexedDeclInfo DeclInfo;
      CXIdxIndexedEntityInfo IdxEntityInfo;
      getIndexedEntityInfo(D, IdxEntityInfo, EntityInfo, DeclInfo, SA);
      CXIdxObjCMethodInfo Info = { &IdxEntityInfo,
                                   D->isThisDeclarationADefinition() };

      idxEntity = CB.indexObjCMethod(ClientData, &Info);
    }

    addEntityInMap(D, idxEntity);

  } else {
    if (CB.indexObjCMethodRedeclaration) {
      CXIdxIndexedRedeclInfo RedeclInfo;
      CXIdxIndexedDeclInfo DeclInfo;
      getIndexedRedeclInfo(D, RedeclInfo, DeclInfo);
      CXIdxObjCMethodRedeclInfo Info = { &RedeclInfo,
                                         D->isThisDeclarationADefinition() };

      CB.indexObjCMethodRedeclaration(ClientData, &Info);
    }
  }
}

void IndexingContext::handleObjCProperty(const ObjCPropertyDecl *D) {
  StrAdapter SA(this);

  CXIdxEntity idxEntity = 0;
  if (CB.indexObjCProperty) {
    CXIdxEntityInfo EntityInfo;
    CXIdxIndexedDeclInfo DeclInfo;
    CXIdxIndexedEntityInfo IdxEntityInfo;
    getIndexedEntityInfo(D, IdxEntityInfo, EntityInfo, DeclInfo, SA);
    CXIdxObjCPropertyInfo Info = { &IdxEntityInfo };

    idxEntity = CB.indexObjCProperty(ClientData, &Info);
  }

  addEntityInMap(D, idxEntity);
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

  CXIdxEntityRefInfo Info = { E ? MakeCXCursor((Stmt*)E,
                                               (Decl*)cast<Decl>(DC), CXTU)
                                : getRefCursor(D, Loc),
                              getIndexLoc(Loc),
                              getIndexEntity(D),
                              getIndexEntity(Parent),
                              getIndexContainerForDC(DC),
                              Kind };
  CB.indexEntityReference(ClientData, &Info);
}

void IndexingContext::invokeStartedStatementBody(const NamedDecl *D,
                                                 const DeclContext *DC) {
  const Stmt *Body = cast<Decl>(DC)->getBody();
  assert(Body);

  CXIdxContainer idxCont = 0;
  if (CB.startedStatementBody) {
    CXIdxContainerInfo ContainerInfo;
    getContainerInfo(D, ContainerInfo);
    CXIdxStmtBodyInfo Info = { &ContainerInfo,
                               getIndexLoc(Body->getLocStart()) };

    idxCont = CB.startedStatementBody(ClientData, &Info);
  }
  addContainerInMap(DC, idxCont);
}

void IndexingContext::invokeStartedTagTypeDefinition(const TagDecl *D) {
  CXIdxContainer idxCont = 0;
  if (CB.startedTagTypeDefinition) {
    CXIdxContainerInfo ContainerInfo;
    getContainerInfo(D, ContainerInfo);
    CXIdxTagTypeDefinitionInfo Info = { &ContainerInfo };

    idxCont = CB.startedTagTypeDefinition(ClientData, &Info);
  }
  addContainerInMap(D, idxCont);
}

void IndexingContext::invokeStartedObjCContainer(const ObjCContainerDecl *D) {
  CXIdxContainer idxCont = 0;
  if (CB.startedObjCContainer) {
    CXIdxContainerInfo ContainerInfo;
    getContainerInfo(D, ContainerInfo);
    CXIdxObjCContainerInfo Info = { &ContainerInfo };

    idxCont = CB.startedObjCContainer(ClientData, &Info);
  }
  addContainerInMap(D, idxCont);
}

void IndexingContext::invokeEndedContainer(const DeclContext *DC) {
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
                                        CXIdxContainer container) {
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

void IndexingContext::addEntityInMap(const NamedDecl *D, CXIdxEntity entity) {
  assert(getEntityDecl(D) == D &&
         "Tried to add a non-entity (canonical) decl");
  assert(EntityMap.find(D) == EntityMap.end());
  if (entity || D->isFromASTFile())
    EntityMap[D] = entity;
}

CXIdxEntity IndexingContext::getIndexEntity(const NamedDecl *D) {
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

  StrAdapter SA(this);
  
  CXIdxEntity idxEntity = 0;
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

CXIdxContainer
IndexingContext::getIndexContainerForDC(const DeclContext *DC) const {
  DC = getScopedContext(DC);
  ContainerMapTy::const_iterator I = ContainerMap.find(DC);
//  assert(I != ContainerMap.end() &&
//         "Failed to include a scoped context in the container map");
  return I->second;
}

CXIdxFile IndexingContext::getIndexFile(const FileEntry *File) {
  if (!File)
    return 0;
  if (!CB.recordFile)
    return 0;

  FileMapTy::iterator FI = FileMap.find(File);
  if (FI != FileMap.end())
    return FI->second;

  CXIdxFile idxFile = CB.recordFile(ClientData, (CXFile)File, 0);
  FileMap[File] = idxFile;
  return idxFile;
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
                                   CXIdxFile *indexFile, CXFile *file,
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

void IndexingContext::getIndexedEntityInfo(const NamedDecl *D,
                          CXIdxIndexedEntityInfo &IdxEntityInfo,
                          CXIdxEntityInfo &EntityInfo,
                          CXIdxIndexedDeclInfo &IdxDeclInfo,
                          StrAdapter &SA) {
  getEntityInfo(D, EntityInfo, SA);
  getIndexedDeclInfo(D, IdxDeclInfo);
  IdxEntityInfo.entityInfo = &EntityInfo;
  IdxEntityInfo.declInfo = &IdxDeclInfo;
}

void IndexingContext::getIndexedDeclInfo(const NamedDecl *D,
                                         CXIdxIndexedDeclInfo &IdxDeclInfo) {
  IdxDeclInfo.cursor = getCursor(D);
  IdxDeclInfo.loc = getIndexLoc(D->getLocation());
  IdxDeclInfo.container = getIndexContainer(D);
}

void IndexingContext::getIndexedRedeclInfo(const NamedDecl *D,
                          CXIdxIndexedRedeclInfo &RedeclInfo,
                          CXIdxIndexedDeclInfo &IdxDeclInfo) {
  getIndexedDeclInfo(D, IdxDeclInfo);
  RedeclInfo.declInfo = &IdxDeclInfo;
  RedeclInfo.entity = getIndexEntity(D);
}

void IndexingContext::getContainerInfo(const NamedDecl *D,
                          CXIdxContainerInfo &ContainerInfo) {
  ContainerInfo.cursor = getCursor(D);
  ContainerInfo.loc = getIndexLoc(D->getLocation());
  ContainerInfo.entity = getIndexEntity(D);
}

void IndexingContext::getEntityInfo(const NamedDecl *D,
                          CXIdxEntityInfo &EntityInfo,
                          StrAdapter &SA) {
  if (IdentifierInfo *II = D->getIdentifier()) {
    EntityInfo.name = SA.toCStr(II->getName());

  } else if (isa<RecordDecl>(D) || isa<NamespaceDecl>(D)) {
    EntityInfo.name = 0;

  } else {
    unsigned Begin = SA.getCurSize();
    {
      llvm::raw_svector_ostream OS(SA.getBuffer());
      D->printName(OS);
    }
    EntityInfo.name = SA.getCStr(Begin);
  }

  unsigned Begin = SA.getCurSize();
  bool Ignore = getDeclCursorUSR(D, SA.getBuffer());
  if (Ignore) {
    EntityInfo.USR = "";
  } else {
    EntityInfo.USR = SA.getCStr(Begin);
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
