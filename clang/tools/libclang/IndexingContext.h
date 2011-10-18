//===- IndexingContext.h - Higher level API functions ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Index_Internal.h"
#include "CXCursor.h"

#include "clang/AST/Decl.h"
#include "clang/AST/DeclGroup.h"
#include "llvm/ADT/DenseMap.h"

namespace clang {
  class FileEntry;
  class ObjCPropertyDecl;

namespace cxindex {
  class IndexingContext;

class IndexingContext {
  ASTContext *Ctx;
  CXClientData ClientData;
  IndexerCallbacks &CB;
  unsigned IndexOptions;
  CXTranslationUnit CXTU;
  
  typedef llvm::DenseMap<const FileEntry *, CXIdxFile> FileMapTy;
  typedef llvm::DenseMap<const NamedDecl *, CXIdxEntity> EntityMapTy;
  typedef llvm::DenseMap<const void *, CXIdxMacro> MacroMapTy;
  typedef llvm::DenseMap<const DeclContext *, CXIdxContainer> ContainerMapTy;
  FileMapTy FileMap;
  EntityMapTy EntityMap;
  MacroMapTy MacroMap;
  ContainerMapTy ContainerMap;

  SmallVector<DeclGroupRef, 8> TUDeclsInObjCContainer;
  
  llvm::SmallString<256> StrScratch;

  class StrAdapter {
    llvm::SmallString<256> &Scratch;

  public:
    StrAdapter(IndexingContext *indexCtx)
      : Scratch(indexCtx->StrScratch) {}
    ~StrAdapter() { Scratch.clear(); }

    const char *toCStr(StringRef Str);

    unsigned getCurSize() const { return Scratch.size(); }

    const char *getCStr(unsigned CharIndex) {
      Scratch.push_back('\0');
      return Scratch.data() + CharIndex;
    }

    SmallVectorImpl<char> &getBuffer() { return Scratch; }
  };

public:
  IndexingContext(CXClientData clientData, IndexerCallbacks &indexCallbacks,
                  unsigned indexOptions, CXTranslationUnit cxTU)
    : Ctx(0), ClientData(clientData), CB(indexCallbacks),
      IndexOptions(indexOptions), CXTU(cxTU) { }

  ASTContext &getASTContext() const { return *Ctx; }

  void setASTContext(ASTContext &ctx);

  void ppIncludedFile(SourceLocation hashLoc,
                      StringRef filename, const FileEntry *File,
                      bool isImport, bool isAngled);

  void ppMacroDefined(SourceLocation Loc, StringRef Name,
                      SourceLocation DefBegin, unsigned Length,
                      const void *OpaqueMacro);

  void ppMacroUndefined(SourceLocation Loc, StringRef Name,
                        const void *OpaqueMacro);

  void ppMacroExpanded(SourceLocation Loc, StringRef Name,
                       const void *OpaqueMacro);

  void invokeStartedTranslationUnit();

  void invokeFinishedTranslationUnit();

  void indexDecl(const Decl *D);

  void indexTagDecl(const TagDecl *D);

  void indexTypeSourceInfo(TypeSourceInfo *TInfo, const NamedDecl *Parent,
                           const DeclContext *DC = 0);

  void indexTypeLoc(TypeLoc TL, const NamedDecl *Parent,
                           const DeclContext *DC);

  void indexDeclContext(const DeclContext *DC);
  
  void indexBody(const Stmt *S, const DeclContext *DC);

  void handleDiagnostic(const StoredDiagnostic &StoredDiag);

  void handleFunction(const FunctionDecl *FD);

  void handleVar(const VarDecl *D);

  void handleField(const FieldDecl *D);

  void handleEnumerator(const EnumConstantDecl *D);

  void handleTagDecl(const TagDecl *D);
  
  void handleTypedef(const TypedefDecl *D);

  void handleObjCInterface(const ObjCInterfaceDecl *D);
  
  void defineObjCInterface(const ObjCInterfaceDecl *D);

  void handleObjCProtocol(const ObjCProtocolDecl *D);

  void handleObjCCategory(const ObjCCategoryDecl *D);

  void handleObjCMethod(const ObjCMethodDecl *D);

  void handleObjCProperty(const ObjCPropertyDecl *D);

  void handleReference(const NamedDecl *D, SourceLocation Loc,
                       const NamedDecl *Parent,
                       const DeclContext *DC,
                       const Expr *E = 0,
                       CXIdxEntityRefKind Kind = CXIdxEntityRef_Direct);
  
  void invokeStartedTagTypeDefinition(const TagDecl *D);

  void invokeStartedStatementBody(const NamedDecl *D, const DeclContext *DC);
  
  void invokeStartedObjCContainer(const ObjCContainerDecl *D);

  void invokeEndedContainer(const DeclContext *DC);

  bool isNotFromSourceFile(SourceLocation Loc) const;

  void indexTUDeclsInObjCContainer();
  void indexDeclGroupRef(DeclGroupRef DG);

  void addTUDeclInObjCContainer(DeclGroupRef DG) {
    TUDeclsInObjCContainer.push_back(DG);
  }

  void translateLoc(SourceLocation Loc, CXIdxFile *indexFile, CXFile *file,
                    unsigned *line, unsigned *column, unsigned *offset);

private:
  void addEntityInMap(const NamedDecl *D, CXIdxEntity entity);

  void addContainerInMap(const DeclContext *DC, CXIdxContainer container);

  CXIdxEntity getIndexEntity(const NamedDecl *D);

  const NamedDecl *getEntityDecl(const NamedDecl *D) const;

  CXIdxContainer getIndexContainer(const NamedDecl *D) const {
    return getIndexContainerForDC(D->getDeclContext());
  }

  const DeclContext *getScopedContext(const DeclContext *DC) const;
  CXIdxContainer getIndexContainerForDC(const DeclContext *DC) const;

  CXIdxFile getIndexFile(const FileEntry *File);
  
  CXIdxLoc getIndexLoc(SourceLocation Loc) const;

  void getIndexedEntityInfo(const NamedDecl *D,
                            CXIdxIndexedEntityInfo &IdxEntityInfo,
                            CXIdxEntityInfo &EntityInfo,
                            CXIdxIndexedDeclInfo &IdxDeclInfo,
                            StrAdapter &SA);

  void getIndexedDeclInfo(const NamedDecl *D,
                          CXIdxIndexedDeclInfo &IdxDeclInfo);

  void getIndexedRedeclInfo(const NamedDecl *D,
                            CXIdxIndexedRedeclInfo &RedeclInfo,
                            CXIdxIndexedDeclInfo &IdxDeclInfo);

  void getContainerInfo(const NamedDecl *D,
                        CXIdxContainerInfo &ContainerInfo);

  void getEntityInfo(const NamedDecl *D,
                     CXIdxEntityInfo &EntityInfo,
                     StrAdapter &SA);

  CXCursor getCursor(const NamedDecl *D) {
    return cxcursor::MakeCXCursor(const_cast<NamedDecl*>(D), CXTU);
  }

  CXCursor getRefCursor(const NamedDecl *D, SourceLocation Loc);
};

}} // end clang::cxindex
