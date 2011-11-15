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

IndexingContext::ObjCProtocolListInfo::ObjCProtocolListInfo(
                                    const ObjCProtocolList &ProtList,
                                    IndexingContext &IdxCtx,
                                    StrAdapter &SA) {
  ObjCInterfaceDecl::protocol_loc_iterator LI = ProtList.loc_begin();
  for (ObjCInterfaceDecl::protocol_iterator
         I = ProtList.begin(), E = ProtList.end(); I != E; ++I, ++LI) {
    SourceLocation Loc = *LI;
    ObjCProtocolDecl *PD = *I;
    ProtEntities.push_back(CXIdxEntityInfo());
    IdxCtx.getEntityInfo(PD, ProtEntities.back(), SA);
    CXIdxObjCProtocolRefInfo ProtInfo = { 0,
                                MakeCursorObjCProtocolRef(PD, Loc, IdxCtx.CXTU),
                                IdxCtx.getIndexLoc(Loc) };
    ProtInfos.push_back(ProtInfo);
  }

  for (unsigned i = 0, e = ProtInfos.size(); i != e; ++i)
    ProtInfos[i].protocol = &ProtEntities[i];

  for (unsigned i = 0, e = ProtInfos.size(); i != e; ++i)
    Prots.push_back(&ProtInfos[i]);
}

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

void IndexingContext::startedTranslationUnit() {
  CXIdxClientContainer idxCont = 0;
  if (CB.startedTranslationUnit)
    idxCont = CB.startedTranslationUnit(ClientData, 0);
  addContainerInMap(Ctx->getTranslationUnitDecl(), idxCont);
}

void IndexingContext::handleDiagnostic(const StoredDiagnostic &StoredDiag) {
  if (!CB.diagnostic)
    return;

  CXStoredDiagnostic CXDiag(StoredDiag, Ctx->getLangOptions());
  CB.diagnostic(ClientData, &CXDiag, 0);
}

void IndexingContext::handleDiagnostic(CXDiagnostic CXDiag) {
  if (!CB.diagnostic)
    return;

  CB.diagnostic(ClientData, CXDiag, 0);
}

void IndexingContext::handleDecl(const NamedDecl *D,
                                 SourceLocation Loc, CXCursor Cursor,
                                 DeclInfo &DInfo) {
  if (!CB.indexDeclaration)
    return;

  StrAdapter SA(*this);
  getEntityInfo(D, DInfo.CXEntInfo, SA);
  DInfo.entityInfo = &DInfo.CXEntInfo;
  DInfo.cursor = Cursor;
  DInfo.loc = getIndexLoc(Loc);
  DInfo.container = getIndexContainer(D);
  DInfo.isImplicit = D->isImplicit();

  CXIdxClientContainer clientCont = 0;
  CXIdxDeclOut DeclOut = { DInfo.isContainer ? &clientCont : 0 };
  CB.indexDeclaration(ClientData, &DInfo, &DeclOut);

  if (DInfo.isContainer)
    addContainerInMap(cast<DeclContext>(D), clientCont);
}

void IndexingContext::handleObjCContainer(const ObjCContainerDecl *D,
                                          SourceLocation Loc, CXCursor Cursor,
                                          ObjCContainerDeclInfo &ContDInfo) {
  ContDInfo.ObjCContDeclInfo.declInfo = &ContDInfo;
  handleDecl(D, Loc, Cursor, ContDInfo);
}

void IndexingContext::handleFunction(const FunctionDecl *D) {
  DeclInfo DInfo(!D->isFirstDeclaration(), D->isThisDeclarationADefinition(),
                 D->isThisDeclarationADefinition());
  handleDecl(D, D->getLocation(), getCursor(D), DInfo);
}

void IndexingContext::handleVar(const VarDecl *D) {
  DeclInfo DInfo(!D->isFirstDeclaration(), D->isThisDeclarationADefinition(),
                 /*isContainer=*/false);
  handleDecl(D, D->getLocation(), getCursor(D), DInfo);
}

void IndexingContext::handleField(const FieldDecl *D) {
  DeclInfo DInfo(/*isRedeclaration=*/false, /*isDefinition=*/true,
                 /*isContainer=*/false);
  handleDecl(D, D->getLocation(), getCursor(D), DInfo);
}

void IndexingContext::handleEnumerator(const EnumConstantDecl *D) {
  DeclInfo DInfo(/*isRedeclaration=*/false, /*isDefinition=*/true,
                 /*isContainer=*/false);
  handleDecl(D, D->getLocation(), getCursor(D), DInfo);
}

void IndexingContext::handleTagDecl(const TagDecl *D) {
  DeclInfo DInfo(!D->isFirstDeclaration(), D->isThisDeclarationADefinition(),
                 D->isThisDeclarationADefinition());
  handleDecl(D, D->getLocation(), getCursor(D), DInfo);
}

void IndexingContext::handleTypedef(const TypedefDecl *D) {
  DeclInfo DInfo(!D->isFirstDeclaration(), /*isDefinition=*/true,
                 /*isContainer=*/false);
  handleDecl(D, D->getLocation(), getCursor(D), DInfo);
}

void IndexingContext::handleObjCClass(const ObjCClassDecl *D) {
  const ObjCClassDecl::ObjCClassRef *Ref = D->getForwardDecl();
  ObjCInterfaceDecl *IFaceD = Ref->getInterface();
  SourceLocation Loc = Ref->getLocation();
  bool isRedeclaration = IFaceD->getLocation() != Loc;
 
  ObjCContainerDeclInfo ContDInfo(/*isForwardRef=*/true, isRedeclaration,
                                  /*isImplementation=*/false);
  handleObjCContainer(IFaceD, Loc, MakeCursorObjCClassRef(IFaceD, Loc, CXTU),
                      ContDInfo);
}

void IndexingContext::handleObjCInterface(const ObjCInterfaceDecl *D) {
  StrAdapter SA(*this);

  CXIdxBaseClassInfo BaseClass;
  CXIdxEntityInfo BaseEntity;
  BaseClass.cursor = clang_getNullCursor();
  if (ObjCInterfaceDecl *SuperD = D->getSuperClass()) {
    getEntityInfo(SuperD, BaseEntity, SA);
    SourceLocation SuperLoc = D->getSuperClassLoc();
    BaseClass.base = &BaseEntity;
    BaseClass.cursor = MakeCursorObjCSuperClassRef(SuperD, SuperLoc, CXTU);
    BaseClass.loc = getIndexLoc(SuperLoc);
  }
  
  ObjCProtocolListInfo ProtInfo(D->getReferencedProtocols(), *this, SA);
  
  ObjCInterfaceDeclInfo InterInfo(D);
  InterInfo.ObjCProtoListInfo = ProtInfo.getListInfo();
  InterInfo.ObjCInterDeclInfo.containerInfo = &InterInfo.ObjCContDeclInfo;
  InterInfo.ObjCInterDeclInfo.superInfo = D->getSuperClass() ? &BaseClass : 0;
  InterInfo.ObjCInterDeclInfo.protocols = &InterInfo.ObjCProtoListInfo;

  handleObjCContainer(D, D->getLocation(), getCursor(D), InterInfo);
}

void IndexingContext::handleObjCImplementation(
                                              const ObjCImplementationDecl *D) {
  const ObjCInterfaceDecl *Class = D->getClassInterface();
  ObjCContainerDeclInfo ContDInfo(/*isForwardRef=*/false,
                      /*isRedeclaration=*/!Class->isImplicitInterfaceDecl(),
                      /*isImplementation=*/true);
  handleObjCContainer(D, D->getLocation(), getCursor(D), ContDInfo);
}

void IndexingContext::handleObjCForwardProtocol(const ObjCProtocolDecl *D,
                                                SourceLocation Loc,
                                                bool isRedeclaration) {
  ObjCContainerDeclInfo ContDInfo(/*isForwardRef=*/true,
                                  isRedeclaration,
                                  /*isImplementation=*/false);
  handleObjCContainer(D, Loc, MakeCursorObjCProtocolRef(D, Loc, CXTU),
                      ContDInfo);
}

void IndexingContext::handleObjCProtocol(const ObjCProtocolDecl *D) {
  StrAdapter SA(*this);
  ObjCProtocolListInfo ProtListInfo(D->getReferencedProtocols(), *this, SA);
  
  ObjCProtocolDeclInfo ProtInfo(D);
  ProtInfo.ObjCProtoRefListInfo = ProtListInfo.getListInfo();

  handleObjCContainer(D, D->getLocation(), getCursor(D), ProtInfo);
}

void IndexingContext::handleObjCCategory(const ObjCCategoryDecl *D) {
  ObjCCategoryDeclInfo CatDInfo(/*isImplementation=*/false);
  CXIdxEntityInfo ClassEntity;
  StrAdapter SA(*this);
  const ObjCInterfaceDecl *IFaceD = D->getClassInterface();
  SourceLocation ClassLoc = D->getLocation();
  SourceLocation CategoryLoc = D->getCategoryNameLoc();
  getEntityInfo(D->getClassInterface(), ClassEntity, SA);

  CatDInfo.ObjCCatDeclInfo.containerInfo = &CatDInfo.ObjCContDeclInfo;
  CatDInfo.ObjCCatDeclInfo.objcClass = &ClassEntity;
  CatDInfo.ObjCCatDeclInfo.classCursor =
      MakeCursorObjCClassRef(IFaceD, ClassLoc, CXTU);
  CatDInfo.ObjCCatDeclInfo.classLoc = getIndexLoc(ClassLoc);
  handleObjCContainer(D, CategoryLoc, getCursor(D), CatDInfo);
}

void IndexingContext::handleObjCCategoryImpl(const ObjCCategoryImplDecl *D) {
  const ObjCCategoryDecl *CatD = D->getCategoryDecl();
  ObjCCategoryDeclInfo CatDInfo(/*isImplementation=*/true);
  CXIdxEntityInfo ClassEntity;
  StrAdapter SA(*this);
  getEntityInfo(CatD->getClassInterface(), ClassEntity, SA);

  CatDInfo.ObjCCatDeclInfo.containerInfo = &CatDInfo.ObjCContDeclInfo;
  CatDInfo.ObjCCatDeclInfo.objcClass = &ClassEntity;
  handleObjCContainer(D, D->getLocation(), getCursor(D), CatDInfo);
}

void IndexingContext::handleObjCMethod(const ObjCMethodDecl *D) {
  DeclInfo DInfo(!D->isCanonicalDecl(), D->isThisDeclarationADefinition(),
                 D->isThisDeclarationADefinition());
  handleDecl(D, D->getLocation(), getCursor(D), DInfo);
}

void IndexingContext::handleObjCProperty(const ObjCPropertyDecl *D) {
  DeclInfo DInfo(/*isRedeclaration=*/false, /*isDefinition=*/false,
                 /*isContainer=*/false);
  handleDecl(D, D->getLocation(), getCursor(D), DInfo);
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
      if (cast<ObjCMethodDecl>(D)->isInstanceMethod())
        EntityInfo.kind = CXIdxEntity_ObjCInstanceMethod;
      else
        EntityInfo.kind = CXIdxEntity_ObjCClassMethod;
      break;
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
