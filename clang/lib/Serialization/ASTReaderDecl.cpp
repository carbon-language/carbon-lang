//===--- ASTReaderDecl.cpp - Decl Deserialization ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the ASTReader::ReadDeclRecord method, which is the
// entrypoint for loading a decl.
//
//===----------------------------------------------------------------------===//

#include "clang/Serialization/ASTReader.h"
#include "ASTCommon.h"
#include "ASTReaderInternals.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclGroup.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/Expr.h"
#include "clang/Sema/IdentifierResolver.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "llvm/Support/SaveAndRestore.h"
using namespace clang;
using namespace clang::serialization;

//===----------------------------------------------------------------------===//
// Declaration deserialization
//===----------------------------------------------------------------------===//

namespace clang {
  class ASTDeclReader : public DeclVisitor<ASTDeclReader, void> {
    ASTReader &Reader;
    ModuleFile &F;
    const DeclID ThisDeclID;
    const unsigned RawLocation;
    typedef ASTReader::RecordData RecordData;
    const RecordData &Record;
    unsigned &Idx;
    TypeID TypeIDForTypeDecl;
    
    bool HasPendingBody;

    uint64_t GetCurrentCursorOffset();
    
    SourceLocation ReadSourceLocation(const RecordData &R, unsigned &I) {
      return Reader.ReadSourceLocation(F, R, I);
    }
    
    SourceRange ReadSourceRange(const RecordData &R, unsigned &I) {
      return Reader.ReadSourceRange(F, R, I);
    }
    
    TypeSourceInfo *GetTypeSourceInfo(const RecordData &R, unsigned &I) {
      return Reader.GetTypeSourceInfo(F, R, I);
    }
    
    serialization::DeclID ReadDeclID(const RecordData &R, unsigned &I) {
      return Reader.ReadDeclID(F, R, I);
    }
    
    Decl *ReadDecl(const RecordData &R, unsigned &I) {
      return Reader.ReadDecl(F, R, I);
    }

    template<typename T>
    T *ReadDeclAs(const RecordData &R, unsigned &I) {
      return Reader.ReadDeclAs<T>(F, R, I);
    }

    void ReadQualifierInfo(QualifierInfo &Info,
                           const RecordData &R, unsigned &I) {
      Reader.ReadQualifierInfo(F, Info, R, I);
    }
    
    void ReadDeclarationNameLoc(DeclarationNameLoc &DNLoc, DeclarationName Name,
                                const RecordData &R, unsigned &I) {
      Reader.ReadDeclarationNameLoc(F, DNLoc, Name, R, I);
    }
    
    void ReadDeclarationNameInfo(DeclarationNameInfo &NameInfo,
                                const RecordData &R, unsigned &I) {
      Reader.ReadDeclarationNameInfo(F, NameInfo, R, I);
    }

    serialization::SubmoduleID readSubmoduleID(const RecordData &R, 
                                               unsigned &I) {
      if (I >= R.size())
        return 0;
      
      return Reader.getGlobalSubmoduleID(F, R[I++]);
    }
    
    Module *readModule(const RecordData &R, unsigned &I) {
      return Reader.getSubmodule(readSubmoduleID(R, I));
    }

    void ReadCXXRecordDefinition(CXXRecordDecl *D);
    void ReadCXXDefinitionData(struct CXXRecordDecl::DefinitionData &Data,
                               const RecordData &R, unsigned &I);
    void MergeDefinitionData(CXXRecordDecl *D,
                             struct CXXRecordDecl::DefinitionData &NewDD);

    /// \brief RAII class used to capture the first ID within a redeclaration
    /// chain and to introduce it into the list of pending redeclaration chains
    /// on destruction.
    ///
    /// The caller can choose not to introduce this ID into the redeclaration
    /// chain by calling \c suppress().
    class RedeclarableResult {
      ASTReader &Reader;
      GlobalDeclID FirstID;
      mutable bool Owning;
      Decl::Kind DeclKind;
      
      void operator=(RedeclarableResult &) LLVM_DELETED_FUNCTION;
      
    public:
      RedeclarableResult(ASTReader &Reader, GlobalDeclID FirstID,
                         Decl::Kind DeclKind)
        : Reader(Reader), FirstID(FirstID), Owning(true), DeclKind(DeclKind) { }

      RedeclarableResult(const RedeclarableResult &Other)
        : Reader(Other.Reader), FirstID(Other.FirstID), Owning(Other.Owning) ,
          DeclKind(Other.DeclKind)
      { 
        Other.Owning = false;
      }

      ~RedeclarableResult() {
        if (FirstID && Owning && isRedeclarableDeclKind(DeclKind) &&
            Reader.PendingDeclChainsKnown.insert(FirstID))
          Reader.PendingDeclChains.push_back(FirstID);
      }
      
      /// \brief Retrieve the first ID.
      GlobalDeclID getFirstID() const { return FirstID; }
      
      /// \brief Do not introduce this declaration ID into the set of pending
      /// declaration chains.
      void suppress() {
        Owning = false;
      }
    };

    /// \brief Class used to capture the result of searching for an existing
    /// declaration of a specific kind and name, along with the ability
    /// to update the place where this result was found (the declaration
    /// chain hanging off an identifier or the DeclContext we searched in)
    /// if requested.
    class FindExistingResult {
      ASTReader &Reader;
      NamedDecl *New;
      NamedDecl *Existing;
      mutable bool AddResult;
      
      void operator=(FindExistingResult&) LLVM_DELETED_FUNCTION;
      
    public:
      FindExistingResult(ASTReader &Reader)
        : Reader(Reader), New(nullptr), Existing(nullptr), AddResult(false) {}

      FindExistingResult(ASTReader &Reader, NamedDecl *New, NamedDecl *Existing)
        : Reader(Reader), New(New), Existing(Existing), AddResult(true) { }
      
      FindExistingResult(const FindExistingResult &Other)
        : Reader(Other.Reader), New(Other.New), Existing(Other.Existing), 
          AddResult(Other.AddResult)
      {
        Other.AddResult = false;
      }
      
      ~FindExistingResult();
      
      /// \brief Suppress the addition of this result into the known set of
      /// names.
      void suppress() { AddResult = false; }
      
      operator NamedDecl*() const { return Existing; }
      
      template<typename T>
      operator T*() const { return dyn_cast_or_null<T>(Existing); }
    };
    
    FindExistingResult findExisting(NamedDecl *D);
    
  public:
    ASTDeclReader(ASTReader &Reader, ModuleFile &F,
                  DeclID thisDeclID,
                  unsigned RawLocation,
                  const RecordData &Record, unsigned &Idx)
      : Reader(Reader), F(F), ThisDeclID(thisDeclID),
        RawLocation(RawLocation), Record(Record), Idx(Idx),
        TypeIDForTypeDecl(0), HasPendingBody(false) { }

    template <typename DeclT>
    static void attachPreviousDeclImpl(Redeclarable<DeclT> *D, Decl *Previous);
    static void attachPreviousDeclImpl(...);
    static void attachPreviousDecl(Decl *D, Decl *previous);

    template <typename DeclT>
    static void attachLatestDeclImpl(Redeclarable<DeclT> *D, Decl *Latest);
    static void attachLatestDeclImpl(...);
    static void attachLatestDecl(Decl *D, Decl *latest);

    template <typename DeclT>
    static void markIncompleteDeclChainImpl(Redeclarable<DeclT> *D);
    static void markIncompleteDeclChainImpl(...);

    /// \brief Determine whether this declaration has a pending body.
    bool hasPendingBody() const { return HasPendingBody; }

    void Visit(Decl *D);

    void UpdateDecl(Decl *D, ModuleFile &ModuleFile,
                    const RecordData &Record);

    static void setNextObjCCategory(ObjCCategoryDecl *Cat,
                                    ObjCCategoryDecl *Next) {
      Cat->NextClassCategory = Next;
    }

    void VisitDecl(Decl *D);
    void VisitTranslationUnitDecl(TranslationUnitDecl *TU);
    void VisitNamedDecl(NamedDecl *ND);
    void VisitLabelDecl(LabelDecl *LD);
    void VisitNamespaceDecl(NamespaceDecl *D);
    void VisitUsingDirectiveDecl(UsingDirectiveDecl *D);
    void VisitNamespaceAliasDecl(NamespaceAliasDecl *D);
    void VisitTypeDecl(TypeDecl *TD);
    void VisitTypedefNameDecl(TypedefNameDecl *TD);
    void VisitTypedefDecl(TypedefDecl *TD);
    void VisitTypeAliasDecl(TypeAliasDecl *TD);
    void VisitUnresolvedUsingTypenameDecl(UnresolvedUsingTypenameDecl *D);
    RedeclarableResult VisitTagDecl(TagDecl *TD);
    void VisitEnumDecl(EnumDecl *ED);
    RedeclarableResult VisitRecordDeclImpl(RecordDecl *RD);
    void VisitRecordDecl(RecordDecl *RD) { VisitRecordDeclImpl(RD); }
    RedeclarableResult VisitCXXRecordDeclImpl(CXXRecordDecl *D);
    void VisitCXXRecordDecl(CXXRecordDecl *D) { VisitCXXRecordDeclImpl(D); }
    RedeclarableResult VisitClassTemplateSpecializationDeclImpl(
                                            ClassTemplateSpecializationDecl *D);
    void VisitClassTemplateSpecializationDecl(
        ClassTemplateSpecializationDecl *D) {
      VisitClassTemplateSpecializationDeclImpl(D);
    }
    void VisitClassTemplatePartialSpecializationDecl(
                                     ClassTemplatePartialSpecializationDecl *D);
    void VisitClassScopeFunctionSpecializationDecl(
                                       ClassScopeFunctionSpecializationDecl *D);
    RedeclarableResult
    VisitVarTemplateSpecializationDeclImpl(VarTemplateSpecializationDecl *D);
    void VisitVarTemplateSpecializationDecl(VarTemplateSpecializationDecl *D) {
      VisitVarTemplateSpecializationDeclImpl(D);
    }
    void VisitVarTemplatePartialSpecializationDecl(
        VarTemplatePartialSpecializationDecl *D);
    void VisitTemplateTypeParmDecl(TemplateTypeParmDecl *D);
    void VisitValueDecl(ValueDecl *VD);
    void VisitEnumConstantDecl(EnumConstantDecl *ECD);
    void VisitUnresolvedUsingValueDecl(UnresolvedUsingValueDecl *D);
    void VisitDeclaratorDecl(DeclaratorDecl *DD);
    void VisitFunctionDecl(FunctionDecl *FD);
    void VisitCXXMethodDecl(CXXMethodDecl *D);
    void VisitCXXConstructorDecl(CXXConstructorDecl *D);
    void VisitCXXDestructorDecl(CXXDestructorDecl *D);
    void VisitCXXConversionDecl(CXXConversionDecl *D);
    void VisitFieldDecl(FieldDecl *FD);
    void VisitMSPropertyDecl(MSPropertyDecl *FD);
    void VisitIndirectFieldDecl(IndirectFieldDecl *FD);
    RedeclarableResult VisitVarDeclImpl(VarDecl *D);
    void VisitVarDecl(VarDecl *VD) { VisitVarDeclImpl(VD); }
    void VisitImplicitParamDecl(ImplicitParamDecl *PD);
    void VisitParmVarDecl(ParmVarDecl *PD);
    void VisitNonTypeTemplateParmDecl(NonTypeTemplateParmDecl *D);
    DeclID VisitTemplateDecl(TemplateDecl *D);
    RedeclarableResult VisitRedeclarableTemplateDecl(RedeclarableTemplateDecl *D);
    void VisitClassTemplateDecl(ClassTemplateDecl *D);
    void VisitVarTemplateDecl(VarTemplateDecl *D);
    void VisitFunctionTemplateDecl(FunctionTemplateDecl *D);
    void VisitTemplateTemplateParmDecl(TemplateTemplateParmDecl *D);
    void VisitTypeAliasTemplateDecl(TypeAliasTemplateDecl *D);
    void VisitUsingDecl(UsingDecl *D);
    void VisitUsingShadowDecl(UsingShadowDecl *D);
    void VisitLinkageSpecDecl(LinkageSpecDecl *D);
    void VisitFileScopeAsmDecl(FileScopeAsmDecl *AD);
    void VisitImportDecl(ImportDecl *D);
    void VisitAccessSpecDecl(AccessSpecDecl *D);
    void VisitFriendDecl(FriendDecl *D);
    void VisitFriendTemplateDecl(FriendTemplateDecl *D);
    void VisitStaticAssertDecl(StaticAssertDecl *D);
    void VisitBlockDecl(BlockDecl *BD);
    void VisitCapturedDecl(CapturedDecl *CD);
    void VisitEmptyDecl(EmptyDecl *D);

    std::pair<uint64_t, uint64_t> VisitDeclContext(DeclContext *DC);

    template<typename T>
    RedeclarableResult VisitRedeclarable(Redeclarable<T> *D);

    template<typename T>
    void mergeRedeclarable(Redeclarable<T> *D, RedeclarableResult &Redecl,
                           DeclID TemplatePatternID = 0);

    template<typename T>
    void mergeRedeclarable(Redeclarable<T> *D, T *Existing,
                           RedeclarableResult &Redecl,
                           DeclID TemplatePatternID = 0);

    template<typename T>
    void mergeMergeable(Mergeable<T> *D);

    void mergeTemplatePattern(RedeclarableTemplateDecl *D,
                              RedeclarableTemplateDecl *Existing,
                              DeclID DsID);

    // FIXME: Reorder according to DeclNodes.td?
    void VisitObjCMethodDecl(ObjCMethodDecl *D);
    void VisitObjCContainerDecl(ObjCContainerDecl *D);
    void VisitObjCInterfaceDecl(ObjCInterfaceDecl *D);
    void VisitObjCIvarDecl(ObjCIvarDecl *D);
    void VisitObjCProtocolDecl(ObjCProtocolDecl *D);
    void VisitObjCAtDefsFieldDecl(ObjCAtDefsFieldDecl *D);
    void VisitObjCCategoryDecl(ObjCCategoryDecl *D);
    void VisitObjCImplDecl(ObjCImplDecl *D);
    void VisitObjCCategoryImplDecl(ObjCCategoryImplDecl *D);
    void VisitObjCImplementationDecl(ObjCImplementationDecl *D);
    void VisitObjCCompatibleAliasDecl(ObjCCompatibleAliasDecl *D);
    void VisitObjCPropertyDecl(ObjCPropertyDecl *D);
    void VisitObjCPropertyImplDecl(ObjCPropertyImplDecl *D);
    void VisitOMPThreadPrivateDecl(OMPThreadPrivateDecl *D);
  };
}

uint64_t ASTDeclReader::GetCurrentCursorOffset() {
  return F.DeclsCursor.GetCurrentBitNo() + F.GlobalBitOffset;
}

void ASTDeclReader::Visit(Decl *D) {
  DeclVisitor<ASTDeclReader, void>::Visit(D);

  if (DeclaratorDecl *DD = dyn_cast<DeclaratorDecl>(D)) {
    if (DD->DeclInfo) {
      DeclaratorDecl::ExtInfo *Info =
          DD->DeclInfo.get<DeclaratorDecl::ExtInfo *>();
      Info->TInfo =
          GetTypeSourceInfo(Record, Idx);
    }
    else {
      DD->DeclInfo = GetTypeSourceInfo(Record, Idx);
    }
  }

  if (TypeDecl *TD = dyn_cast<TypeDecl>(D)) {
    // We have a fully initialized TypeDecl. Read its type now.
    TD->setTypeForDecl(Reader.GetType(TypeIDForTypeDecl).getTypePtrOrNull());
  } else if (ObjCInterfaceDecl *ID = dyn_cast<ObjCInterfaceDecl>(D)) {
    // if we have a fully initialized TypeDecl, we can safely read its type now.
    ID->TypeForDecl = Reader.GetType(TypeIDForTypeDecl).getTypePtrOrNull();
  } else if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    // FunctionDecl's body was written last after all other Stmts/Exprs.
    // We only read it if FD doesn't already have a body (e.g., from another
    // module).
    // FIXME: Also consider = default and = delete.
    // FIXME: Can we diagnose ODR violations somehow?
    if (Record[Idx++]) {
      Reader.PendingBodies[FD] = GetCurrentCursorOffset();
      HasPendingBody = true;
    }
  }
}

void ASTDeclReader::VisitDecl(Decl *D) {
  if (D->isTemplateParameter() || D->isTemplateParameterPack() ||
      isa<ParmVarDecl>(D)) {
    // We don't want to deserialize the DeclContext of a template
    // parameter or of a parameter of a function template immediately.   These
    // entities might be used in the formulation of its DeclContext (for
    // example, a function parameter can be used in decltype() in trailing
    // return type of the function).  Use the translation unit DeclContext as a
    // placeholder.
    GlobalDeclID SemaDCIDForTemplateParmDecl = ReadDeclID(Record, Idx);
    GlobalDeclID LexicalDCIDForTemplateParmDecl = ReadDeclID(Record, Idx);
    Reader.addPendingDeclContextInfo(D,
                                     SemaDCIDForTemplateParmDecl,
                                     LexicalDCIDForTemplateParmDecl);
    D->setDeclContext(Reader.getContext().getTranslationUnitDecl()); 
  } else {
    DeclContext *SemaDC = ReadDeclAs<DeclContext>(Record, Idx);
    DeclContext *LexicalDC = ReadDeclAs<DeclContext>(Record, Idx);
    DeclContext *MergedSemaDC = Reader.MergedDeclContexts.lookup(SemaDC);
    // Avoid calling setLexicalDeclContext() directly because it uses
    // Decl::getASTContext() internally which is unsafe during derialization.
    D->setDeclContextsImpl(MergedSemaDC ? MergedSemaDC : SemaDC, LexicalDC,
                           Reader.getContext());
  }
  D->setLocation(Reader.ReadSourceLocation(F, RawLocation));
  D->setInvalidDecl(Record[Idx++]);
  if (Record[Idx++]) { // hasAttrs
    AttrVec Attrs;
    Reader.ReadAttributes(F, Attrs, Record, Idx);
    // Avoid calling setAttrs() directly because it uses Decl::getASTContext()
    // internally which is unsafe during derialization.
    D->setAttrsImpl(Attrs, Reader.getContext());
  }
  D->setImplicit(Record[Idx++]);
  D->Used = Record[Idx++];
  D->setReferenced(Record[Idx++]);
  D->setTopLevelDeclInObjCContainer(Record[Idx++]);
  D->setAccess((AccessSpecifier)Record[Idx++]);
  D->FromASTFile = true;
  D->setModulePrivate(Record[Idx++]);
  D->Hidden = D->isModulePrivate();
  
  // Determine whether this declaration is part of a (sub)module. If so, it
  // may not yet be visible.
  if (unsigned SubmoduleID = readSubmoduleID(Record, Idx)) {
    // Store the owning submodule ID in the declaration.
    D->setOwningModuleID(SubmoduleID);
    
    // Module-private declarations are never visible, so there is no work to do.
    if (!D->isModulePrivate()) {
      if (Module *Owner = Reader.getSubmodule(SubmoduleID)) {
        if (Owner->NameVisibility != Module::AllVisible) {
          // The owning module is not visible. Mark this declaration as hidden.
          D->Hidden = true;
          
          // Note that this declaration was hidden because its owning module is 
          // not yet visible.
          Reader.HiddenNamesMap[Owner].HiddenDecls.push_back(D);
        }
      }
    }
  }
}

void ASTDeclReader::VisitTranslationUnitDecl(TranslationUnitDecl *TU) {
  llvm_unreachable("Translation units are not serialized");
}

void ASTDeclReader::VisitNamedDecl(NamedDecl *ND) {
  VisitDecl(ND);
  ND->setDeclName(Reader.ReadDeclarationName(F, Record, Idx));
}

void ASTDeclReader::VisitTypeDecl(TypeDecl *TD) {
  VisitNamedDecl(TD);
  TD->setLocStart(ReadSourceLocation(Record, Idx));
  // Delay type reading until after we have fully initialized the decl.
  TypeIDForTypeDecl = Reader.getGlobalTypeID(F, Record[Idx++]);
}

void ASTDeclReader::VisitTypedefNameDecl(TypedefNameDecl *TD) {
  RedeclarableResult Redecl = VisitRedeclarable(TD);
  VisitTypeDecl(TD);
  TypeSourceInfo *TInfo = GetTypeSourceInfo(Record, Idx);
  if (Record[Idx++]) { // isModed
    QualType modedT = Reader.readType(F, Record, Idx);
    TD->setModedTypeSourceInfo(TInfo, modedT);
  } else
    TD->setTypeSourceInfo(TInfo);
  mergeRedeclarable(TD, Redecl);
}

void ASTDeclReader::VisitTypedefDecl(TypedefDecl *TD) {
  VisitTypedefNameDecl(TD);
}

void ASTDeclReader::VisitTypeAliasDecl(TypeAliasDecl *TD) {
  VisitTypedefNameDecl(TD);
}

ASTDeclReader::RedeclarableResult ASTDeclReader::VisitTagDecl(TagDecl *TD) {
  RedeclarableResult Redecl = VisitRedeclarable(TD);
  VisitTypeDecl(TD);
  
  TD->IdentifierNamespace = Record[Idx++];
  TD->setTagKind((TagDecl::TagKind)Record[Idx++]);
  TD->setCompleteDefinition(Record[Idx++]);
  TD->setEmbeddedInDeclarator(Record[Idx++]);
  TD->setFreeStanding(Record[Idx++]);
  TD->setCompleteDefinitionRequired(Record[Idx++]);
  TD->setRBraceLoc(ReadSourceLocation(Record, Idx));
  
  if (Record[Idx++]) { // hasExtInfo
    TagDecl::ExtInfo *Info = new (Reader.getContext()) TagDecl::ExtInfo();
    ReadQualifierInfo(*Info, Record, Idx);
    TD->NamedDeclOrQualifier = Info;
  } else
    TD->NamedDeclOrQualifier = ReadDeclAs<NamedDecl>(Record, Idx);

  if (!isa<CXXRecordDecl>(TD))
    mergeRedeclarable(TD, Redecl);
  return Redecl;
}

void ASTDeclReader::VisitEnumDecl(EnumDecl *ED) {
  VisitTagDecl(ED);
  if (TypeSourceInfo *TI = Reader.GetTypeSourceInfo(F, Record, Idx))
    ED->setIntegerTypeSourceInfo(TI);
  else
    ED->setIntegerType(Reader.readType(F, Record, Idx));
  ED->setPromotionType(Reader.readType(F, Record, Idx));
  ED->setNumPositiveBits(Record[Idx++]);
  ED->setNumNegativeBits(Record[Idx++]);
  ED->IsScoped = Record[Idx++];
  ED->IsScopedUsingClassTag = Record[Idx++];
  ED->IsFixed = Record[Idx++];

  // If this is a definition subject to the ODR, and we already have a
  // definition, merge this one into it.
  if (ED->IsCompleteDefinition &&
      Reader.getContext().getLangOpts().Modules &&
      Reader.getContext().getLangOpts().CPlusPlus) {
    if (EnumDecl *&OldDef = Reader.EnumDefinitions[ED->getCanonicalDecl()]) {
      Reader.MergedDeclContexts.insert(std::make_pair(ED, OldDef));
      ED->IsCompleteDefinition = false;
    } else {
      OldDef = ED;
    }
  }

  if (EnumDecl *InstED = ReadDeclAs<EnumDecl>(Record, Idx)) {
    TemplateSpecializationKind TSK = (TemplateSpecializationKind)Record[Idx++];
    SourceLocation POI = ReadSourceLocation(Record, Idx);
    ED->setInstantiationOfMemberEnum(Reader.getContext(), InstED, TSK);
    ED->getMemberSpecializationInfo()->setPointOfInstantiation(POI);
  }
}

ASTDeclReader::RedeclarableResult
ASTDeclReader::VisitRecordDeclImpl(RecordDecl *RD) {
  RedeclarableResult Redecl = VisitTagDecl(RD);
  RD->setHasFlexibleArrayMember(Record[Idx++]);
  RD->setAnonymousStructOrUnion(Record[Idx++]);
  RD->setHasObjectMember(Record[Idx++]);
  RD->setHasVolatileMember(Record[Idx++]);
  return Redecl;
}

void ASTDeclReader::VisitValueDecl(ValueDecl *VD) {
  VisitNamedDecl(VD);
  VD->setType(Reader.readType(F, Record, Idx));
}

void ASTDeclReader::VisitEnumConstantDecl(EnumConstantDecl *ECD) {
  VisitValueDecl(ECD);
  if (Record[Idx++])
    ECD->setInitExpr(Reader.ReadExpr(F));
  ECD->setInitVal(Reader.ReadAPSInt(Record, Idx));
  mergeMergeable(ECD);
}

void ASTDeclReader::VisitDeclaratorDecl(DeclaratorDecl *DD) {
  VisitValueDecl(DD);
  DD->setInnerLocStart(ReadSourceLocation(Record, Idx));
  if (Record[Idx++]) { // hasExtInfo
    DeclaratorDecl::ExtInfo *Info
        = new (Reader.getContext()) DeclaratorDecl::ExtInfo();
    ReadQualifierInfo(*Info, Record, Idx);
    DD->DeclInfo = Info;
  }
}

void ASTDeclReader::VisitFunctionDecl(FunctionDecl *FD) {
  RedeclarableResult Redecl = VisitRedeclarable(FD);
  VisitDeclaratorDecl(FD);

  ReadDeclarationNameLoc(FD->DNLoc, FD->getDeclName(), Record, Idx);
  FD->IdentifierNamespace = Record[Idx++];
  
  // FunctionDecl's body is handled last at ASTDeclReader::Visit,
  // after everything else is read.

  FD->SClass = (StorageClass)Record[Idx++];
  FD->IsInline = Record[Idx++];
  FD->IsInlineSpecified = Record[Idx++];
  FD->IsVirtualAsWritten = Record[Idx++];
  FD->IsPure = Record[Idx++];
  FD->HasInheritedPrototype = Record[Idx++];
  FD->HasWrittenPrototype = Record[Idx++];
  FD->IsDeleted = Record[Idx++];
  FD->IsTrivial = Record[Idx++];
  FD->IsDefaulted = Record[Idx++];
  FD->IsExplicitlyDefaulted = Record[Idx++];
  FD->HasImplicitReturnZero = Record[Idx++];
  FD->IsConstexpr = Record[Idx++];
  FD->HasSkippedBody = Record[Idx++];
  FD->IsLateTemplateParsed = Record[Idx++];
  FD->setCachedLinkage(Linkage(Record[Idx++]));
  FD->EndRangeLoc = ReadSourceLocation(Record, Idx);

  switch ((FunctionDecl::TemplatedKind)Record[Idx++]) {
  case FunctionDecl::TK_NonTemplate:
    mergeRedeclarable(FD, Redecl);
    break;
  case FunctionDecl::TK_FunctionTemplate:
    // Merged when we merge the template.
    FD->setDescribedFunctionTemplate(ReadDeclAs<FunctionTemplateDecl>(Record, 
                                                                      Idx));
    break;
  case FunctionDecl::TK_MemberSpecialization: {
    FunctionDecl *InstFD = ReadDeclAs<FunctionDecl>(Record, Idx);
    TemplateSpecializationKind TSK = (TemplateSpecializationKind)Record[Idx++];
    SourceLocation POI = ReadSourceLocation(Record, Idx);
    FD->setInstantiationOfMemberFunction(Reader.getContext(), InstFD, TSK);
    FD->getMemberSpecializationInfo()->setPointOfInstantiation(POI);
    mergeRedeclarable(FD, Redecl);
    break;
  }
  case FunctionDecl::TK_FunctionTemplateSpecialization: {
    FunctionTemplateDecl *Template = ReadDeclAs<FunctionTemplateDecl>(Record, 
                                                                      Idx);
    TemplateSpecializationKind TSK = (TemplateSpecializationKind)Record[Idx++];
    
    // Template arguments.
    SmallVector<TemplateArgument, 8> TemplArgs;
    Reader.ReadTemplateArgumentList(TemplArgs, F, Record, Idx);
    
    // Template args as written.
    SmallVector<TemplateArgumentLoc, 8> TemplArgLocs;
    SourceLocation LAngleLoc, RAngleLoc;
    bool HasTemplateArgumentsAsWritten = Record[Idx++];
    if (HasTemplateArgumentsAsWritten) {
      unsigned NumTemplateArgLocs = Record[Idx++];
      TemplArgLocs.reserve(NumTemplateArgLocs);
      for (unsigned i=0; i != NumTemplateArgLocs; ++i)
        TemplArgLocs.push_back(
            Reader.ReadTemplateArgumentLoc(F, Record, Idx));
  
      LAngleLoc = ReadSourceLocation(Record, Idx);
      RAngleLoc = ReadSourceLocation(Record, Idx);
    }
    
    SourceLocation POI = ReadSourceLocation(Record, Idx);

    ASTContext &C = Reader.getContext();
    TemplateArgumentList *TemplArgList
      = TemplateArgumentList::CreateCopy(C, TemplArgs.data(), TemplArgs.size());
    TemplateArgumentListInfo TemplArgsInfo(LAngleLoc, RAngleLoc);
    for (unsigned i=0, e = TemplArgLocs.size(); i != e; ++i)
      TemplArgsInfo.addArgument(TemplArgLocs[i]);
    FunctionTemplateSpecializationInfo *FTInfo
        = FunctionTemplateSpecializationInfo::Create(C, FD, Template, TSK,
                                                     TemplArgList,
                             HasTemplateArgumentsAsWritten ? &TemplArgsInfo
                                                           : nullptr,
                                                     POI);
    FD->TemplateOrSpecialization = FTInfo;

    if (FD->isCanonicalDecl()) { // if canonical add to template's set.
      // The template that contains the specializations set. It's not safe to
      // use getCanonicalDecl on Template since it may still be initializing.
      FunctionTemplateDecl *CanonTemplate
        = ReadDeclAs<FunctionTemplateDecl>(Record, Idx);
      // Get the InsertPos by FindNodeOrInsertPos() instead of calling
      // InsertNode(FTInfo) directly to avoid the getASTContext() call in
      // FunctionTemplateSpecializationInfo's Profile().
      // We avoid getASTContext because a decl in the parent hierarchy may
      // be initializing.
      llvm::FoldingSetNodeID ID;
      FunctionTemplateSpecializationInfo::Profile(ID, TemplArgs.data(),
                                                  TemplArgs.size(), C);
      void *InsertPos = nullptr;
      FunctionTemplateDecl::Common *CommonPtr = CanonTemplate->getCommonPtr();
      CommonPtr->Specializations.FindNodeOrInsertPos(ID, InsertPos);
      if (InsertPos)
        CommonPtr->Specializations.InsertNode(FTInfo, InsertPos);
      else {
        assert(Reader.getContext().getLangOpts().Modules &&
               "already deserialized this template specialization");
        // FIXME: This specialization is a redeclaration of one from another
        // module. Merge it.
      }
    }
    break;
  }
  case FunctionDecl::TK_DependentFunctionTemplateSpecialization: {
    // Templates.
    UnresolvedSet<8> TemplDecls;
    unsigned NumTemplates = Record[Idx++];
    while (NumTemplates--)
      TemplDecls.addDecl(ReadDeclAs<NamedDecl>(Record, Idx));
    
    // Templates args.
    TemplateArgumentListInfo TemplArgs;
    unsigned NumArgs = Record[Idx++];
    while (NumArgs--)
      TemplArgs.addArgument(Reader.ReadTemplateArgumentLoc(F, Record, Idx));
    TemplArgs.setLAngleLoc(ReadSourceLocation(Record, Idx));
    TemplArgs.setRAngleLoc(ReadSourceLocation(Record, Idx));
    
    FD->setDependentTemplateSpecialization(Reader.getContext(),
                                           TemplDecls, TemplArgs);

    // FIXME: Merging.
    break;
  }
  }

  // Read in the parameters.
  unsigned NumParams = Record[Idx++];
  SmallVector<ParmVarDecl *, 16> Params;
  Params.reserve(NumParams);
  for (unsigned I = 0; I != NumParams; ++I)
    Params.push_back(ReadDeclAs<ParmVarDecl>(Record, Idx));
  FD->setParams(Reader.getContext(), Params);
}

void ASTDeclReader::VisitObjCMethodDecl(ObjCMethodDecl *MD) {
  VisitNamedDecl(MD);
  if (Record[Idx++]) {
    // Load the body on-demand. Most clients won't care, because method
    // definitions rarely show up in headers.
    Reader.PendingBodies[MD] = GetCurrentCursorOffset();
    HasPendingBody = true;
    MD->setSelfDecl(ReadDeclAs<ImplicitParamDecl>(Record, Idx));
    MD->setCmdDecl(ReadDeclAs<ImplicitParamDecl>(Record, Idx));
  }
  MD->setInstanceMethod(Record[Idx++]);
  MD->setVariadic(Record[Idx++]);
  MD->setPropertyAccessor(Record[Idx++]);
  MD->setDefined(Record[Idx++]);
  MD->IsOverriding = Record[Idx++];
  MD->HasSkippedBody = Record[Idx++];

  MD->IsRedeclaration = Record[Idx++];
  MD->HasRedeclaration = Record[Idx++];
  if (MD->HasRedeclaration)
    Reader.getContext().setObjCMethodRedeclaration(MD,
                                       ReadDeclAs<ObjCMethodDecl>(Record, Idx));

  MD->setDeclImplementation((ObjCMethodDecl::ImplementationControl)Record[Idx++]);
  MD->setObjCDeclQualifier((Decl::ObjCDeclQualifier)Record[Idx++]);
  MD->SetRelatedResultType(Record[Idx++]);
  MD->setReturnType(Reader.readType(F, Record, Idx));
  MD->setReturnTypeSourceInfo(GetTypeSourceInfo(Record, Idx));
  MD->DeclEndLoc = ReadSourceLocation(Record, Idx);
  unsigned NumParams = Record[Idx++];
  SmallVector<ParmVarDecl *, 16> Params;
  Params.reserve(NumParams);
  for (unsigned I = 0; I != NumParams; ++I)
    Params.push_back(ReadDeclAs<ParmVarDecl>(Record, Idx));

  MD->SelLocsKind = Record[Idx++];
  unsigned NumStoredSelLocs = Record[Idx++];
  SmallVector<SourceLocation, 16> SelLocs;
  SelLocs.reserve(NumStoredSelLocs);
  for (unsigned i = 0; i != NumStoredSelLocs; ++i)
    SelLocs.push_back(ReadSourceLocation(Record, Idx));

  MD->setParamsAndSelLocs(Reader.getContext(), Params, SelLocs);
}

void ASTDeclReader::VisitObjCContainerDecl(ObjCContainerDecl *CD) {
  VisitNamedDecl(CD);
  CD->setAtStartLoc(ReadSourceLocation(Record, Idx));
  CD->setAtEndRange(ReadSourceRange(Record, Idx));
}

void ASTDeclReader::VisitObjCInterfaceDecl(ObjCInterfaceDecl *ID) {
  RedeclarableResult Redecl = VisitRedeclarable(ID);
  VisitObjCContainerDecl(ID);
  TypeIDForTypeDecl = Reader.getGlobalTypeID(F, Record[Idx++]);
  mergeRedeclarable(ID, Redecl);
  
  if (Record[Idx++]) {
    // Read the definition.
    ID->allocateDefinitionData();
    
    // Set the definition data of the canonical declaration, so other
    // redeclarations will see it.
    ID->getCanonicalDecl()->Data = ID->Data;
    
    ObjCInterfaceDecl::DefinitionData &Data = ID->data();
    
    // Read the superclass.
    Data.SuperClass = ReadDeclAs<ObjCInterfaceDecl>(Record, Idx);
    Data.SuperClassLoc = ReadSourceLocation(Record, Idx);

    Data.EndLoc = ReadSourceLocation(Record, Idx);
    Data.HasDesignatedInitializers = Record[Idx++];
    
    // Read the directly referenced protocols and their SourceLocations.
    unsigned NumProtocols = Record[Idx++];
    SmallVector<ObjCProtocolDecl *, 16> Protocols;
    Protocols.reserve(NumProtocols);
    for (unsigned I = 0; I != NumProtocols; ++I)
      Protocols.push_back(ReadDeclAs<ObjCProtocolDecl>(Record, Idx));
    SmallVector<SourceLocation, 16> ProtoLocs;
    ProtoLocs.reserve(NumProtocols);
    for (unsigned I = 0; I != NumProtocols; ++I)
      ProtoLocs.push_back(ReadSourceLocation(Record, Idx));
    ID->setProtocolList(Protocols.data(), NumProtocols, ProtoLocs.data(),
                        Reader.getContext());
  
    // Read the transitive closure of protocols referenced by this class.
    NumProtocols = Record[Idx++];
    Protocols.clear();
    Protocols.reserve(NumProtocols);
    for (unsigned I = 0; I != NumProtocols; ++I)
      Protocols.push_back(ReadDeclAs<ObjCProtocolDecl>(Record, Idx));
    ID->data().AllReferencedProtocols.set(Protocols.data(), NumProtocols,
                                          Reader.getContext());
  
    // We will rebuild this list lazily.
    ID->setIvarList(nullptr);

    // Note that we have deserialized a definition.
    Reader.PendingDefinitions.insert(ID);
    
    // Note that we've loaded this Objective-C class.
    Reader.ObjCClassesLoaded.push_back(ID);
  } else {
    ID->Data = ID->getCanonicalDecl()->Data;
  }
}

void ASTDeclReader::VisitObjCIvarDecl(ObjCIvarDecl *IVD) {
  VisitFieldDecl(IVD);
  IVD->setAccessControl((ObjCIvarDecl::AccessControl)Record[Idx++]);
  // This field will be built lazily.
  IVD->setNextIvar(nullptr);
  bool synth = Record[Idx++];
  IVD->setSynthesize(synth);
}

void ASTDeclReader::VisitObjCProtocolDecl(ObjCProtocolDecl *PD) {
  RedeclarableResult Redecl = VisitRedeclarable(PD);
  VisitObjCContainerDecl(PD);
  mergeRedeclarable(PD, Redecl);
  
  if (Record[Idx++]) {
    // Read the definition.
    PD->allocateDefinitionData();
    
    // Set the definition data of the canonical declaration, so other
    // redeclarations will see it.
    PD->getCanonicalDecl()->Data = PD->Data;

    unsigned NumProtoRefs = Record[Idx++];
    SmallVector<ObjCProtocolDecl *, 16> ProtoRefs;
    ProtoRefs.reserve(NumProtoRefs);
    for (unsigned I = 0; I != NumProtoRefs; ++I)
      ProtoRefs.push_back(ReadDeclAs<ObjCProtocolDecl>(Record, Idx));
    SmallVector<SourceLocation, 16> ProtoLocs;
    ProtoLocs.reserve(NumProtoRefs);
    for (unsigned I = 0; I != NumProtoRefs; ++I)
      ProtoLocs.push_back(ReadSourceLocation(Record, Idx));
    PD->setProtocolList(ProtoRefs.data(), NumProtoRefs, ProtoLocs.data(),
                        Reader.getContext());
    
    // Note that we have deserialized a definition.
    Reader.PendingDefinitions.insert(PD);
  } else {
    PD->Data = PD->getCanonicalDecl()->Data;
  }
}

void ASTDeclReader::VisitObjCAtDefsFieldDecl(ObjCAtDefsFieldDecl *FD) {
  VisitFieldDecl(FD);
}

void ASTDeclReader::VisitObjCCategoryDecl(ObjCCategoryDecl *CD) {
  VisitObjCContainerDecl(CD);
  CD->setCategoryNameLoc(ReadSourceLocation(Record, Idx));
  CD->setIvarLBraceLoc(ReadSourceLocation(Record, Idx));
  CD->setIvarRBraceLoc(ReadSourceLocation(Record, Idx));
  
  // Note that this category has been deserialized. We do this before
  // deserializing the interface declaration, so that it will consider this
  /// category.
  Reader.CategoriesDeserialized.insert(CD);

  CD->ClassInterface = ReadDeclAs<ObjCInterfaceDecl>(Record, Idx);
  unsigned NumProtoRefs = Record[Idx++];
  SmallVector<ObjCProtocolDecl *, 16> ProtoRefs;
  ProtoRefs.reserve(NumProtoRefs);
  for (unsigned I = 0; I != NumProtoRefs; ++I)
    ProtoRefs.push_back(ReadDeclAs<ObjCProtocolDecl>(Record, Idx));
  SmallVector<SourceLocation, 16> ProtoLocs;
  ProtoLocs.reserve(NumProtoRefs);
  for (unsigned I = 0; I != NumProtoRefs; ++I)
    ProtoLocs.push_back(ReadSourceLocation(Record, Idx));
  CD->setProtocolList(ProtoRefs.data(), NumProtoRefs, ProtoLocs.data(),
                      Reader.getContext());
}

void ASTDeclReader::VisitObjCCompatibleAliasDecl(ObjCCompatibleAliasDecl *CAD) {
  VisitNamedDecl(CAD);
  CAD->setClassInterface(ReadDeclAs<ObjCInterfaceDecl>(Record, Idx));
}

void ASTDeclReader::VisitObjCPropertyDecl(ObjCPropertyDecl *D) {
  VisitNamedDecl(D);
  D->setAtLoc(ReadSourceLocation(Record, Idx));
  D->setLParenLoc(ReadSourceLocation(Record, Idx));
  D->setType(GetTypeSourceInfo(Record, Idx));
  // FIXME: stable encoding
  D->setPropertyAttributes(
                      (ObjCPropertyDecl::PropertyAttributeKind)Record[Idx++]);
  D->setPropertyAttributesAsWritten(
                      (ObjCPropertyDecl::PropertyAttributeKind)Record[Idx++]);
  // FIXME: stable encoding
  D->setPropertyImplementation(
                            (ObjCPropertyDecl::PropertyControl)Record[Idx++]);
  D->setGetterName(Reader.ReadDeclarationName(F,Record, Idx).getObjCSelector());
  D->setSetterName(Reader.ReadDeclarationName(F,Record, Idx).getObjCSelector());
  D->setGetterMethodDecl(ReadDeclAs<ObjCMethodDecl>(Record, Idx));
  D->setSetterMethodDecl(ReadDeclAs<ObjCMethodDecl>(Record, Idx));
  D->setPropertyIvarDecl(ReadDeclAs<ObjCIvarDecl>(Record, Idx));
}

void ASTDeclReader::VisitObjCImplDecl(ObjCImplDecl *D) {
  VisitObjCContainerDecl(D);
  D->setClassInterface(ReadDeclAs<ObjCInterfaceDecl>(Record, Idx));
}

void ASTDeclReader::VisitObjCCategoryImplDecl(ObjCCategoryImplDecl *D) {
  VisitObjCImplDecl(D);
  D->setIdentifier(Reader.GetIdentifierInfo(F, Record, Idx));
  D->CategoryNameLoc = ReadSourceLocation(Record, Idx);
}

void ASTDeclReader::VisitObjCImplementationDecl(ObjCImplementationDecl *D) {
  VisitObjCImplDecl(D);
  D->setSuperClass(ReadDeclAs<ObjCInterfaceDecl>(Record, Idx));
  D->SuperLoc = ReadSourceLocation(Record, Idx);
  D->setIvarLBraceLoc(ReadSourceLocation(Record, Idx));
  D->setIvarRBraceLoc(ReadSourceLocation(Record, Idx));
  D->setHasNonZeroConstructors(Record[Idx++]);
  D->setHasDestructors(Record[Idx++]);
  std::tie(D->IvarInitializers, D->NumIvarInitializers) =
      Reader.ReadCXXCtorInitializers(F, Record, Idx);
}


void ASTDeclReader::VisitObjCPropertyImplDecl(ObjCPropertyImplDecl *D) {
  VisitDecl(D);
  D->setAtLoc(ReadSourceLocation(Record, Idx));
  D->setPropertyDecl(ReadDeclAs<ObjCPropertyDecl>(Record, Idx));
  D->PropertyIvarDecl = ReadDeclAs<ObjCIvarDecl>(Record, Idx);
  D->IvarLoc = ReadSourceLocation(Record, Idx);
  D->setGetterCXXConstructor(Reader.ReadExpr(F));
  D->setSetterCXXAssignment(Reader.ReadExpr(F));
}

void ASTDeclReader::VisitFieldDecl(FieldDecl *FD) {
  VisitDeclaratorDecl(FD);
  FD->Mutable = Record[Idx++];
  if (int BitWidthOrInitializer = Record[Idx++]) {
    FD->InitializerOrBitWidth.setInt(BitWidthOrInitializer - 1);
    FD->InitializerOrBitWidth.setPointer(Reader.ReadExpr(F));
  }
  if (!FD->getDeclName()) {
    if (FieldDecl *Tmpl = ReadDeclAs<FieldDecl>(Record, Idx))
      Reader.getContext().setInstantiatedFromUnnamedFieldDecl(FD, Tmpl);
  }
  mergeMergeable(FD);
}

void ASTDeclReader::VisitMSPropertyDecl(MSPropertyDecl *PD) {
  VisitDeclaratorDecl(PD);
  PD->GetterId = Reader.GetIdentifierInfo(F, Record, Idx);
  PD->SetterId = Reader.GetIdentifierInfo(F, Record, Idx);
}

void ASTDeclReader::VisitIndirectFieldDecl(IndirectFieldDecl *FD) {
  VisitValueDecl(FD);

  FD->ChainingSize = Record[Idx++];
  assert(FD->ChainingSize >= 2 && "Anonymous chaining must be >= 2");
  FD->Chaining = new (Reader.getContext())NamedDecl*[FD->ChainingSize];

  for (unsigned I = 0; I != FD->ChainingSize; ++I)
    FD->Chaining[I] = ReadDeclAs<NamedDecl>(Record, Idx);
}

ASTDeclReader::RedeclarableResult ASTDeclReader::VisitVarDeclImpl(VarDecl *VD) {
  RedeclarableResult Redecl = VisitRedeclarable(VD);
  VisitDeclaratorDecl(VD);

  VD->VarDeclBits.SClass = (StorageClass)Record[Idx++];
  VD->VarDeclBits.TSCSpec = Record[Idx++];
  VD->VarDeclBits.InitStyle = Record[Idx++];
  VD->VarDeclBits.ExceptionVar = Record[Idx++];
  VD->VarDeclBits.NRVOVariable = Record[Idx++];
  VD->VarDeclBits.CXXForRangeDecl = Record[Idx++];
  VD->VarDeclBits.ARCPseudoStrong = Record[Idx++];
  VD->VarDeclBits.IsConstexpr = Record[Idx++];
  VD->VarDeclBits.IsInitCapture = Record[Idx++];
  VD->VarDeclBits.PreviousDeclInSameBlockScope = Record[Idx++];
  Linkage VarLinkage = Linkage(Record[Idx++]);
  VD->setCachedLinkage(VarLinkage);

  // Reconstruct the one piece of the IdentifierNamespace that we need.
  if (VD->getStorageClass() == SC_Extern && VarLinkage != NoLinkage &&
      VD->getLexicalDeclContext()->isFunctionOrMethod())
    VD->setLocalExternDecl();

  if (uint64_t Val = Record[Idx++]) {
    VD->setInit(Reader.ReadExpr(F));
    if (Val > 1) {
      EvaluatedStmt *Eval = VD->ensureEvaluatedStmt();
      Eval->CheckedICE = true;
      Eval->IsICE = Val == 3;
    }
  }

  enum VarKind {
    VarNotTemplate = 0, VarTemplate, StaticDataMemberSpecialization
  };
  switch ((VarKind)Record[Idx++]) {
  case VarNotTemplate:
    // Only true variables (not parameters or implicit parameters) can be merged
    if (VD->getKind() != Decl::ParmVar && VD->getKind() != Decl::ImplicitParam)
      mergeRedeclarable(VD, Redecl);
    break;
  case VarTemplate:
    // Merged when we merge the template.
    VD->setDescribedVarTemplate(ReadDeclAs<VarTemplateDecl>(Record, Idx));
    break;
  case StaticDataMemberSpecialization: { // HasMemberSpecializationInfo.
    VarDecl *Tmpl = ReadDeclAs<VarDecl>(Record, Idx);
    TemplateSpecializationKind TSK = (TemplateSpecializationKind)Record[Idx++];
    SourceLocation POI = ReadSourceLocation(Record, Idx);
    Reader.getContext().setInstantiatedFromStaticDataMember(VD, Tmpl, TSK,POI);
    mergeRedeclarable(VD, Redecl);
    break;
  }
  }

  return Redecl;
}

void ASTDeclReader::VisitImplicitParamDecl(ImplicitParamDecl *PD) {
  VisitVarDecl(PD);
}

void ASTDeclReader::VisitParmVarDecl(ParmVarDecl *PD) {
  VisitVarDecl(PD);
  unsigned isObjCMethodParam = Record[Idx++];
  unsigned scopeDepth = Record[Idx++];
  unsigned scopeIndex = Record[Idx++];
  unsigned declQualifier = Record[Idx++];
  if (isObjCMethodParam) {
    assert(scopeDepth == 0);
    PD->setObjCMethodScopeInfo(scopeIndex);
    PD->ParmVarDeclBits.ScopeDepthOrObjCQuals = declQualifier;
  } else {
    PD->setScopeInfo(scopeDepth, scopeIndex);
  }
  PD->ParmVarDeclBits.IsKNRPromoted = Record[Idx++];
  PD->ParmVarDeclBits.HasInheritedDefaultArg = Record[Idx++];
  if (Record[Idx++]) // hasUninstantiatedDefaultArg.
    PD->setUninstantiatedDefaultArg(Reader.ReadExpr(F));

  // FIXME: If this is a redeclaration of a function from another module, handle
  // inheritance of default arguments.
}

void ASTDeclReader::VisitFileScopeAsmDecl(FileScopeAsmDecl *AD) {
  VisitDecl(AD);
  AD->setAsmString(cast<StringLiteral>(Reader.ReadExpr(F)));
  AD->setRParenLoc(ReadSourceLocation(Record, Idx));
}

void ASTDeclReader::VisitBlockDecl(BlockDecl *BD) {
  VisitDecl(BD);
  BD->setBody(cast_or_null<CompoundStmt>(Reader.ReadStmt(F)));
  BD->setSignatureAsWritten(GetTypeSourceInfo(Record, Idx));
  unsigned NumParams = Record[Idx++];
  SmallVector<ParmVarDecl *, 16> Params;
  Params.reserve(NumParams);
  for (unsigned I = 0; I != NumParams; ++I)
    Params.push_back(ReadDeclAs<ParmVarDecl>(Record, Idx));
  BD->setParams(Params);

  BD->setIsVariadic(Record[Idx++]);
  BD->setBlockMissingReturnType(Record[Idx++]);
  BD->setIsConversionFromLambda(Record[Idx++]);

  bool capturesCXXThis = Record[Idx++];
  unsigned numCaptures = Record[Idx++];
  SmallVector<BlockDecl::Capture, 16> captures;
  captures.reserve(numCaptures);
  for (unsigned i = 0; i != numCaptures; ++i) {
    VarDecl *decl = ReadDeclAs<VarDecl>(Record, Idx);
    unsigned flags = Record[Idx++];
    bool byRef = (flags & 1);
    bool nested = (flags & 2);
    Expr *copyExpr = ((flags & 4) ? Reader.ReadExpr(F) : nullptr);

    captures.push_back(BlockDecl::Capture(decl, byRef, nested, copyExpr));
  }
  BD->setCaptures(Reader.getContext(), captures.begin(),
                  captures.end(), capturesCXXThis);
}

void ASTDeclReader::VisitCapturedDecl(CapturedDecl *CD) {
  VisitDecl(CD);
  unsigned ContextParamPos = Record[Idx++];
  CD->setNothrow(Record[Idx++] != 0);
  // Body is set by VisitCapturedStmt.
  for (unsigned I = 0; I < CD->NumParams; ++I) {
    if (I != ContextParamPos)
      CD->setParam(I, ReadDeclAs<ImplicitParamDecl>(Record, Idx));
    else
      CD->setContextParam(I, ReadDeclAs<ImplicitParamDecl>(Record, Idx));
  }
}

void ASTDeclReader::VisitLinkageSpecDecl(LinkageSpecDecl *D) {
  VisitDecl(D);
  D->setLanguage((LinkageSpecDecl::LanguageIDs)Record[Idx++]);
  D->setExternLoc(ReadSourceLocation(Record, Idx));
  D->setRBraceLoc(ReadSourceLocation(Record, Idx));
}

void ASTDeclReader::VisitLabelDecl(LabelDecl *D) {
  VisitNamedDecl(D);
  D->setLocStart(ReadSourceLocation(Record, Idx));
}


void ASTDeclReader::VisitNamespaceDecl(NamespaceDecl *D) {
  RedeclarableResult Redecl = VisitRedeclarable(D);
  VisitNamedDecl(D);
  D->setInline(Record[Idx++]);
  D->LocStart = ReadSourceLocation(Record, Idx);
  D->RBraceLoc = ReadSourceLocation(Record, Idx);
  // FIXME: At the point of this call, D->getCanonicalDecl() returns 0.
  mergeRedeclarable(D, Redecl);

  if (Redecl.getFirstID() == ThisDeclID) {
    // Each module has its own anonymous namespace, which is disjoint from
    // any other module's anonymous namespaces, so don't attach the anonymous
    // namespace at all.
    NamespaceDecl *Anon = ReadDeclAs<NamespaceDecl>(Record, Idx);
    if (F.Kind != MK_Module)
      D->setAnonymousNamespace(Anon);
  } else {
    // Link this namespace back to the first declaration, which has already
    // been deserialized.
    D->AnonOrFirstNamespaceAndInline.setPointer(D->getFirstDecl());
  }
}

void ASTDeclReader::VisitNamespaceAliasDecl(NamespaceAliasDecl *D) {
  VisitNamedDecl(D);
  D->NamespaceLoc = ReadSourceLocation(Record, Idx);
  D->IdentLoc = ReadSourceLocation(Record, Idx);
  D->QualifierLoc = Reader.ReadNestedNameSpecifierLoc(F, Record, Idx);
  D->Namespace = ReadDeclAs<NamedDecl>(Record, Idx);
}

void ASTDeclReader::VisitUsingDecl(UsingDecl *D) {
  VisitNamedDecl(D);
  D->setUsingLoc(ReadSourceLocation(Record, Idx));
  D->QualifierLoc = Reader.ReadNestedNameSpecifierLoc(F, Record, Idx);
  ReadDeclarationNameLoc(D->DNLoc, D->getDeclName(), Record, Idx);
  D->FirstUsingShadow.setPointer(ReadDeclAs<UsingShadowDecl>(Record, Idx));
  D->setTypename(Record[Idx++]);
  if (NamedDecl *Pattern = ReadDeclAs<NamedDecl>(Record, Idx))
    Reader.getContext().setInstantiatedFromUsingDecl(D, Pattern);
}

void ASTDeclReader::VisitUsingShadowDecl(UsingShadowDecl *D) {
  RedeclarableResult Redecl = VisitRedeclarable(D);
  VisitNamedDecl(D);
  D->setTargetDecl(ReadDeclAs<NamedDecl>(Record, Idx));
  D->UsingOrNextShadow = ReadDeclAs<NamedDecl>(Record, Idx);
  UsingShadowDecl *Pattern = ReadDeclAs<UsingShadowDecl>(Record, Idx);
  if (Pattern)
    Reader.getContext().setInstantiatedFromUsingShadowDecl(D, Pattern);
  mergeRedeclarable(D, Redecl);
}

void ASTDeclReader::VisitUsingDirectiveDecl(UsingDirectiveDecl *D) {
  VisitNamedDecl(D);
  D->UsingLoc = ReadSourceLocation(Record, Idx);
  D->NamespaceLoc = ReadSourceLocation(Record, Idx);
  D->QualifierLoc = Reader.ReadNestedNameSpecifierLoc(F, Record, Idx);
  D->NominatedNamespace = ReadDeclAs<NamedDecl>(Record, Idx);
  D->CommonAncestor = ReadDeclAs<DeclContext>(Record, Idx);
}

void ASTDeclReader::VisitUnresolvedUsingValueDecl(UnresolvedUsingValueDecl *D) {
  VisitValueDecl(D);
  D->setUsingLoc(ReadSourceLocation(Record, Idx));
  D->QualifierLoc = Reader.ReadNestedNameSpecifierLoc(F, Record, Idx);
  ReadDeclarationNameLoc(D->DNLoc, D->getDeclName(), Record, Idx);
}

void ASTDeclReader::VisitUnresolvedUsingTypenameDecl(
                                               UnresolvedUsingTypenameDecl *D) {
  VisitTypeDecl(D);
  D->TypenameLocation = ReadSourceLocation(Record, Idx);
  D->QualifierLoc = Reader.ReadNestedNameSpecifierLoc(F, Record, Idx);
}

void ASTDeclReader::ReadCXXDefinitionData(
                                   struct CXXRecordDecl::DefinitionData &Data,
                                   const RecordData &Record, unsigned &Idx) {
  // Note: the caller has deserialized the IsLambda bit already.
  Data.UserDeclaredConstructor = Record[Idx++];
  Data.UserDeclaredSpecialMembers = Record[Idx++];
  Data.Aggregate = Record[Idx++];
  Data.PlainOldData = Record[Idx++];
  Data.Empty = Record[Idx++];
  Data.Polymorphic = Record[Idx++];
  Data.Abstract = Record[Idx++];
  Data.IsStandardLayout = Record[Idx++];
  Data.HasNoNonEmptyBases = Record[Idx++];
  Data.HasPrivateFields = Record[Idx++];
  Data.HasProtectedFields = Record[Idx++];
  Data.HasPublicFields = Record[Idx++];
  Data.HasMutableFields = Record[Idx++];
  Data.HasVariantMembers = Record[Idx++];
  Data.HasOnlyCMembers = Record[Idx++];
  Data.HasInClassInitializer = Record[Idx++];
  Data.HasUninitializedReferenceMember = Record[Idx++];
  Data.NeedOverloadResolutionForMoveConstructor = Record[Idx++];
  Data.NeedOverloadResolutionForMoveAssignment = Record[Idx++];
  Data.NeedOverloadResolutionForDestructor = Record[Idx++];
  Data.DefaultedMoveConstructorIsDeleted = Record[Idx++];
  Data.DefaultedMoveAssignmentIsDeleted = Record[Idx++];
  Data.DefaultedDestructorIsDeleted = Record[Idx++];
  Data.HasTrivialSpecialMembers = Record[Idx++];
  Data.DeclaredNonTrivialSpecialMembers = Record[Idx++];
  Data.HasIrrelevantDestructor = Record[Idx++];
  Data.HasConstexprNonCopyMoveConstructor = Record[Idx++];
  Data.DefaultedDefaultConstructorIsConstexpr = Record[Idx++];
  Data.HasConstexprDefaultConstructor = Record[Idx++];
  Data.HasNonLiteralTypeFieldsOrBases = Record[Idx++];
  Data.ComputedVisibleConversions = Record[Idx++];
  Data.UserProvidedDefaultConstructor = Record[Idx++];
  Data.DeclaredSpecialMembers = Record[Idx++];
  Data.ImplicitCopyConstructorHasConstParam = Record[Idx++];
  Data.ImplicitCopyAssignmentHasConstParam = Record[Idx++];
  Data.HasDeclaredCopyConstructorWithConstParam = Record[Idx++];
  Data.HasDeclaredCopyAssignmentWithConstParam = Record[Idx++];

  Data.NumBases = Record[Idx++];
  if (Data.NumBases)
    Data.Bases = Reader.readCXXBaseSpecifiers(F, Record, Idx);
  Data.NumVBases = Record[Idx++];
  if (Data.NumVBases)
    Data.VBases = Reader.readCXXBaseSpecifiers(F, Record, Idx);
  
  Reader.ReadUnresolvedSet(F, Data.Conversions, Record, Idx);
  Reader.ReadUnresolvedSet(F, Data.VisibleConversions, Record, Idx);
  assert(Data.Definition && "Data.Definition should be already set!");
  Data.FirstFriend = ReadDeclID(Record, Idx);

  if (Data.IsLambda) {
    typedef LambdaCapture Capture;
    CXXRecordDecl::LambdaDefinitionData &Lambda
      = static_cast<CXXRecordDecl::LambdaDefinitionData &>(Data);
    Lambda.Dependent = Record[Idx++];
    Lambda.IsGenericLambda = Record[Idx++];
    Lambda.CaptureDefault = Record[Idx++];
    Lambda.NumCaptures = Record[Idx++];
    Lambda.NumExplicitCaptures = Record[Idx++];
    Lambda.ManglingNumber = Record[Idx++];
    Lambda.ContextDecl = ReadDecl(Record, Idx);
    Lambda.Captures 
      = (Capture*)Reader.Context.Allocate(sizeof(Capture)*Lambda.NumCaptures);
    Capture *ToCapture = Lambda.Captures;
    Lambda.MethodTyInfo = GetTypeSourceInfo(Record, Idx);
    for (unsigned I = 0, N = Lambda.NumCaptures; I != N; ++I) {
      SourceLocation Loc = ReadSourceLocation(Record, Idx);
      bool IsImplicit = Record[Idx++];
      LambdaCaptureKind Kind = static_cast<LambdaCaptureKind>(Record[Idx++]);
      switch (Kind) {
      case LCK_This:
        *ToCapture++ = Capture(Loc, IsImplicit, Kind, nullptr,SourceLocation());
        break;
      case LCK_ByCopy:
      case LCK_ByRef:
        VarDecl *Var = ReadDeclAs<VarDecl>(Record, Idx);
        SourceLocation EllipsisLoc = ReadSourceLocation(Record, Idx);
        *ToCapture++ = Capture(Loc, IsImplicit, Kind, Var, EllipsisLoc);
        break;
      }
    }
  }
}

void ASTDeclReader::MergeDefinitionData(
    CXXRecordDecl *D, struct CXXRecordDecl::DefinitionData &MergeDD) {
  assert(D->DefinitionData.getNotUpdated() &&
         "merging class definition into non-definition");
  auto &DD = *D->DefinitionData.getNotUpdated();

  // If the new definition has new special members, let the name lookup
  // code know that it needs to look in the new definition too.
  if ((MergeDD.DeclaredSpecialMembers & ~DD.DeclaredSpecialMembers) &&
      DD.Definition != MergeDD.Definition) {
    Reader.MergedLookups[DD.Definition].push_back(MergeDD.Definition);
    DD.Definition->setHasExternalVisibleStorage();
  }

  // FIXME: Move this out into a .def file?
  // FIXME: Issue a diagnostic on a mismatched MATCH_FIELD, rather than
  // asserting; this can happen in the case of an ODR violation.
  bool DetectedOdrViolation = false;
#define OR_FIELD(Field) DD.Field |= MergeDD.Field;
#define MATCH_FIELD(Field) \
    DetectedOdrViolation |= DD.Field != MergeDD.Field; \
    OR_FIELD(Field)
  MATCH_FIELD(UserDeclaredConstructor)
  MATCH_FIELD(UserDeclaredSpecialMembers)
  MATCH_FIELD(Aggregate)
  MATCH_FIELD(PlainOldData)
  MATCH_FIELD(Empty)
  MATCH_FIELD(Polymorphic)
  MATCH_FIELD(Abstract)
  MATCH_FIELD(IsStandardLayout)
  MATCH_FIELD(HasNoNonEmptyBases)
  MATCH_FIELD(HasPrivateFields)
  MATCH_FIELD(HasProtectedFields)
  MATCH_FIELD(HasPublicFields)
  MATCH_FIELD(HasMutableFields)
  MATCH_FIELD(HasVariantMembers)
  MATCH_FIELD(HasOnlyCMembers)
  MATCH_FIELD(HasInClassInitializer)
  MATCH_FIELD(HasUninitializedReferenceMember)
  MATCH_FIELD(NeedOverloadResolutionForMoveConstructor)
  MATCH_FIELD(NeedOverloadResolutionForMoveAssignment)
  MATCH_FIELD(NeedOverloadResolutionForDestructor)
  MATCH_FIELD(DefaultedMoveConstructorIsDeleted)
  MATCH_FIELD(DefaultedMoveAssignmentIsDeleted)
  MATCH_FIELD(DefaultedDestructorIsDeleted)
  OR_FIELD(HasTrivialSpecialMembers)
  OR_FIELD(DeclaredNonTrivialSpecialMembers)
  MATCH_FIELD(HasIrrelevantDestructor)
  OR_FIELD(HasConstexprNonCopyMoveConstructor)
  MATCH_FIELD(DefaultedDefaultConstructorIsConstexpr)
  OR_FIELD(HasConstexprDefaultConstructor)
  MATCH_FIELD(HasNonLiteralTypeFieldsOrBases)
  // ComputedVisibleConversions is handled below.
  MATCH_FIELD(UserProvidedDefaultConstructor)
  OR_FIELD(DeclaredSpecialMembers)
  MATCH_FIELD(ImplicitCopyConstructorHasConstParam)
  MATCH_FIELD(ImplicitCopyAssignmentHasConstParam)
  OR_FIELD(HasDeclaredCopyConstructorWithConstParam)
  OR_FIELD(HasDeclaredCopyAssignmentWithConstParam)
  MATCH_FIELD(IsLambda)
#undef OR_FIELD
#undef MATCH_FIELD

  if (DD.NumBases != MergeDD.NumBases || DD.NumVBases != MergeDD.NumVBases)
    DetectedOdrViolation = true;
  // FIXME: Issue a diagnostic if the base classes don't match when we come
  // to lazily load them.

  // FIXME: Issue a diagnostic if the list of conversion functions doesn't
  // match when we come to lazily load them.
  if (MergeDD.ComputedVisibleConversions && !DD.ComputedVisibleConversions) {
    DD.VisibleConversions = std::move(MergeDD.VisibleConversions);
    DD.ComputedVisibleConversions = true;
  }

  // FIXME: Issue a diagnostic if FirstFriend doesn't match when we come to
  // lazily load it.

  if (DD.IsLambda) {
    // FIXME: ODR-checking for merging lambdas (this happens, for instance,
    // when they occur within the body of a function template specialization).
  }

  if (DetectedOdrViolation)
    Reader.PendingOdrMergeFailures[DD.Definition].push_back(MergeDD.Definition);
}

void ASTDeclReader::ReadCXXRecordDefinition(CXXRecordDecl *D) {
  struct CXXRecordDecl::DefinitionData *DD;
  ASTContext &C = Reader.getContext();

  // Determine whether this is a lambda closure type, so that we can
  // allocate the appropriate DefinitionData structure.
  bool IsLambda = Record[Idx++];
  if (IsLambda)
    DD = new (C) CXXRecordDecl::LambdaDefinitionData(D, nullptr, false, false,
                                                     LCD_None);
  else
    DD = new (C) struct CXXRecordDecl::DefinitionData(D);

  ReadCXXDefinitionData(*DD, Record, Idx);

  // If we're reading an update record, we might already have a definition for
  // this record. If so, just merge into it.
  if (D->DefinitionData.getNotUpdated()) {
    MergeDefinitionData(D, *DD);
    return;
  }

  // Propagate the DefinitionData pointer to the canonical declaration, so
  // that all other deserialized declarations will see it.
  CXXRecordDecl *Canon = D->getCanonicalDecl();
  if (Canon == D) {
    D->DefinitionData = DD;
    D->IsCompleteDefinition = true;
  } else if (auto *CanonDD = Canon->DefinitionData.getNotUpdated()) {
    // We have already deserialized a definition of this record. This
    // definition is no longer really a definition. Note that the pre-existing
    // definition is the *real* definition.
    Reader.MergedDeclContexts.insert(
        std::make_pair(D, CanonDD->Definition));
    D->DefinitionData = Canon->DefinitionData;
    D->IsCompleteDefinition = false;
    MergeDefinitionData(D, *DD);
  } else {
    Canon->DefinitionData = DD;
    D->DefinitionData = Canon->DefinitionData;
    D->IsCompleteDefinition = true;

    // Note that we have deserialized a definition. Any declarations
    // deserialized before this one will be be given the DefinitionData
    // pointer at the end.
    Reader.PendingDefinitions.insert(D);
  }
}

ASTDeclReader::RedeclarableResult
ASTDeclReader::VisitCXXRecordDeclImpl(CXXRecordDecl *D) {
  RedeclarableResult Redecl = VisitRecordDeclImpl(D);

  ASTContext &C = Reader.getContext();

  enum CXXRecKind {
    CXXRecNotTemplate = 0, CXXRecTemplate, CXXRecMemberSpecialization
  };
  switch ((CXXRecKind)Record[Idx++]) {
  case CXXRecNotTemplate:
    mergeRedeclarable(D, Redecl);
    break;
  case CXXRecTemplate: {
    // Merged when we merge the template.
    ClassTemplateDecl *Template = ReadDeclAs<ClassTemplateDecl>(Record, Idx);
    D->TemplateOrInstantiation = Template;
    if (!Template->getTemplatedDecl()) {
      // We've not actually loaded the ClassTemplateDecl yet, because we're
      // currently being loaded as its pattern. Rely on it to set up our
      // TypeForDecl (see VisitClassTemplateDecl).
      //
      // Beware: we do not yet know our canonical declaration, and may still
      // get merged once the surrounding class template has got off the ground.
      TypeIDForTypeDecl = 0;
    }
    break;
  }
  case CXXRecMemberSpecialization: {
    CXXRecordDecl *RD = ReadDeclAs<CXXRecordDecl>(Record, Idx);
    TemplateSpecializationKind TSK = (TemplateSpecializationKind)Record[Idx++];
    SourceLocation POI = ReadSourceLocation(Record, Idx);
    MemberSpecializationInfo *MSI = new (C) MemberSpecializationInfo(RD, TSK);
    MSI->setPointOfInstantiation(POI);
    D->TemplateOrInstantiation = MSI;
    mergeRedeclarable(D, Redecl);
    break;
  }
  }

  bool WasDefinition = Record[Idx++];
  if (WasDefinition)
    ReadCXXRecordDefinition(D);
  else
    // Propagate DefinitionData pointer from the canonical declaration.
    D->DefinitionData = D->getCanonicalDecl()->DefinitionData;

  // Lazily load the key function to avoid deserializing every method so we can
  // compute it.
  if (WasDefinition) {
    DeclID KeyFn = ReadDeclID(Record, Idx);
    if (KeyFn && D->IsCompleteDefinition)
      C.KeyFunctions[D] = KeyFn;
  }

  return Redecl;
}

void ASTDeclReader::VisitCXXMethodDecl(CXXMethodDecl *D) {
  VisitFunctionDecl(D);
  unsigned NumOverridenMethods = Record[Idx++];
  while (NumOverridenMethods--) {
    // Avoid invariant checking of CXXMethodDecl::addOverriddenMethod,
    // MD may be initializing.
    if (CXXMethodDecl *MD = ReadDeclAs<CXXMethodDecl>(Record, Idx))
      Reader.getContext().addOverriddenMethod(D, MD);
  }
}

void ASTDeclReader::VisitCXXConstructorDecl(CXXConstructorDecl *D) {
  VisitCXXMethodDecl(D);

  if (auto *CD = ReadDeclAs<CXXConstructorDecl>(Record, Idx))
    D->setInheritedConstructor(CD);
  D->IsExplicitSpecified = Record[Idx++];
  // FIXME: We should defer loading this until we need the constructor's body.
  std::tie(D->CtorInitializers, D->NumCtorInitializers) =
      Reader.ReadCXXCtorInitializers(F, Record, Idx);
}

void ASTDeclReader::VisitCXXDestructorDecl(CXXDestructorDecl *D) {
  VisitCXXMethodDecl(D);

  D->OperatorDelete = ReadDeclAs<FunctionDecl>(Record, Idx);
}

void ASTDeclReader::VisitCXXConversionDecl(CXXConversionDecl *D) {
  VisitCXXMethodDecl(D);
  D->IsExplicitSpecified = Record[Idx++];
}

void ASTDeclReader::VisitImportDecl(ImportDecl *D) {
  VisitDecl(D);
  D->ImportedAndComplete.setPointer(readModule(Record, Idx));
  D->ImportedAndComplete.setInt(Record[Idx++]);
  SourceLocation *StoredLocs = reinterpret_cast<SourceLocation *>(D + 1);
  for (unsigned I = 0, N = Record.back(); I != N; ++I)
    StoredLocs[I] = ReadSourceLocation(Record, Idx);
  ++Idx; // The number of stored source locations.
}

void ASTDeclReader::VisitAccessSpecDecl(AccessSpecDecl *D) {
  VisitDecl(D);
  D->setColonLoc(ReadSourceLocation(Record, Idx));
}

void ASTDeclReader::VisitFriendDecl(FriendDecl *D) {
  VisitDecl(D);
  if (Record[Idx++]) // hasFriendDecl
    D->Friend = ReadDeclAs<NamedDecl>(Record, Idx);
  else
    D->Friend = GetTypeSourceInfo(Record, Idx);
  for (unsigned i = 0; i != D->NumTPLists; ++i)
    D->getTPLists()[i] = Reader.ReadTemplateParameterList(F, Record, Idx);
  D->NextFriend = ReadDeclID(Record, Idx);
  D->UnsupportedFriend = (Record[Idx++] != 0);
  D->FriendLoc = ReadSourceLocation(Record, Idx);
}

void ASTDeclReader::VisitFriendTemplateDecl(FriendTemplateDecl *D) {
  VisitDecl(D);
  unsigned NumParams = Record[Idx++];
  D->NumParams = NumParams;
  D->Params = new TemplateParameterList*[NumParams];
  for (unsigned i = 0; i != NumParams; ++i)
    D->Params[i] = Reader.ReadTemplateParameterList(F, Record, Idx);
  if (Record[Idx++]) // HasFriendDecl
    D->Friend = ReadDeclAs<NamedDecl>(Record, Idx);
  else
    D->Friend = GetTypeSourceInfo(Record, Idx);
  D->FriendLoc = ReadSourceLocation(Record, Idx);
}

DeclID ASTDeclReader::VisitTemplateDecl(TemplateDecl *D) {
  VisitNamedDecl(D);

  DeclID PatternID = ReadDeclID(Record, Idx);
  NamedDecl *TemplatedDecl = cast_or_null<NamedDecl>(Reader.GetDecl(PatternID));
  TemplateParameterList* TemplateParams
      = Reader.ReadTemplateParameterList(F, Record, Idx); 
  D->init(TemplatedDecl, TemplateParams);

  // FIXME: If this is a redeclaration of a template from another module, handle
  // inheritance of default template arguments.

  return PatternID;
}

ASTDeclReader::RedeclarableResult 
ASTDeclReader::VisitRedeclarableTemplateDecl(RedeclarableTemplateDecl *D) {
  RedeclarableResult Redecl = VisitRedeclarable(D);

  // Make sure we've allocated the Common pointer first. We do this before
  // VisitTemplateDecl so that getCommonPtr() can be used during initialization.
  RedeclarableTemplateDecl *CanonD = D->getCanonicalDecl();
  if (!CanonD->Common) {
    CanonD->Common = CanonD->newCommon(Reader.getContext());
    Reader.PendingDefinitions.insert(CanonD);
  }
  D->Common = CanonD->Common;

  // If this is the first declaration of the template, fill in the information
  // for the 'common' pointer.
  if (ThisDeclID == Redecl.getFirstID()) {
    if (RedeclarableTemplateDecl *RTD
          = ReadDeclAs<RedeclarableTemplateDecl>(Record, Idx)) {
      assert(RTD->getKind() == D->getKind() &&
             "InstantiatedFromMemberTemplate kind mismatch");
      D->setInstantiatedFromMemberTemplate(RTD);
      if (Record[Idx++])
        D->setMemberSpecialization();
    }
  }

  DeclID PatternID = VisitTemplateDecl(D);
  D->IdentifierNamespace = Record[Idx++];

  mergeRedeclarable(D, Redecl, PatternID);

  // If we merged the template with a prior declaration chain, merge the common
  // pointer.
  // FIXME: Actually merge here, don't just overwrite.
  D->Common = D->getCanonicalDecl()->Common;

  return Redecl;
}

void ASTDeclReader::VisitClassTemplateDecl(ClassTemplateDecl *D) {
  RedeclarableResult Redecl = VisitRedeclarableTemplateDecl(D);

  if (ThisDeclID == Redecl.getFirstID()) {
    // This ClassTemplateDecl owns a CommonPtr; read it to keep track of all of
    // the specializations.
    SmallVector<serialization::DeclID, 2> SpecIDs;
    SpecIDs.push_back(0);
    
    // Specializations.
    unsigned Size = Record[Idx++];
    SpecIDs[0] += Size;
    for (unsigned I = 0; I != Size; ++I)
      SpecIDs.push_back(ReadDeclID(Record, Idx));

    // Partial specializations.
    Size = Record[Idx++];
    SpecIDs[0] += Size;
    for (unsigned I = 0; I != Size; ++I)
      SpecIDs.push_back(ReadDeclID(Record, Idx));

    ClassTemplateDecl::Common *CommonPtr = D->getCommonPtr();
    if (SpecIDs[0]) {
      typedef serialization::DeclID DeclID;
      
      // FIXME: Append specializations!
      CommonPtr->LazySpecializations
        = new (Reader.getContext()) DeclID [SpecIDs.size()];
      memcpy(CommonPtr->LazySpecializations, SpecIDs.data(), 
             SpecIDs.size() * sizeof(DeclID));
    }
  }

  if (D->getTemplatedDecl()->TemplateOrInstantiation) {
    // We were loaded before our templated declaration was. We've not set up
    // its corresponding type yet (see VisitCXXRecordDeclImpl), so reconstruct
    // it now.
    Reader.Context.getInjectedClassNameType(
        D->getTemplatedDecl(), D->getInjectedClassNameSpecialization());
  }
}

/// TODO: Unify with ClassTemplateDecl version?
///       May require unifying ClassTemplateDecl and
///        VarTemplateDecl beyond TemplateDecl...
void ASTDeclReader::VisitVarTemplateDecl(VarTemplateDecl *D) {
  RedeclarableResult Redecl = VisitRedeclarableTemplateDecl(D);

  if (ThisDeclID == Redecl.getFirstID()) {
    // This VarTemplateDecl owns a CommonPtr; read it to keep track of all of
    // the specializations.
    SmallVector<serialization::DeclID, 2> SpecIDs;
    SpecIDs.push_back(0);

    // Specializations.
    unsigned Size = Record[Idx++];
    SpecIDs[0] += Size;
    for (unsigned I = 0; I != Size; ++I)
      SpecIDs.push_back(ReadDeclID(Record, Idx));

    // Partial specializations.
    Size = Record[Idx++];
    SpecIDs[0] += Size;
    for (unsigned I = 0; I != Size; ++I)
      SpecIDs.push_back(ReadDeclID(Record, Idx));

    VarTemplateDecl::Common *CommonPtr = D->getCommonPtr();
    if (SpecIDs[0]) {
      typedef serialization::DeclID DeclID;

      // FIXME: Append specializations!
      CommonPtr->LazySpecializations =
          new (Reader.getContext()) DeclID[SpecIDs.size()];
      memcpy(CommonPtr->LazySpecializations, SpecIDs.data(),
             SpecIDs.size() * sizeof(DeclID));
    }
  }
}

ASTDeclReader::RedeclarableResult
ASTDeclReader::VisitClassTemplateSpecializationDeclImpl(
    ClassTemplateSpecializationDecl *D) {
  RedeclarableResult Redecl = VisitCXXRecordDeclImpl(D);
  
  ASTContext &C = Reader.getContext();
  if (Decl *InstD = ReadDecl(Record, Idx)) {
    if (ClassTemplateDecl *CTD = dyn_cast<ClassTemplateDecl>(InstD)) {
      D->SpecializedTemplate = CTD;
    } else {
      SmallVector<TemplateArgument, 8> TemplArgs;
      Reader.ReadTemplateArgumentList(TemplArgs, F, Record, Idx);
      TemplateArgumentList *ArgList
        = TemplateArgumentList::CreateCopy(C, TemplArgs.data(), 
                                           TemplArgs.size());
      ClassTemplateSpecializationDecl::SpecializedPartialSpecialization *PS
          = new (C) ClassTemplateSpecializationDecl::
                                             SpecializedPartialSpecialization();
      PS->PartialSpecialization
          = cast<ClassTemplatePartialSpecializationDecl>(InstD);
      PS->TemplateArgs = ArgList;
      D->SpecializedTemplate = PS;
    }
  }

  SmallVector<TemplateArgument, 8> TemplArgs;
  Reader.ReadTemplateArgumentList(TemplArgs, F, Record, Idx);
  D->TemplateArgs = TemplateArgumentList::CreateCopy(C, TemplArgs.data(), 
                                                     TemplArgs.size());
  D->PointOfInstantiation = ReadSourceLocation(Record, Idx);
  D->SpecializationKind = (TemplateSpecializationKind)Record[Idx++];

  bool writtenAsCanonicalDecl = Record[Idx++];
  if (writtenAsCanonicalDecl) {
    ClassTemplateDecl *CanonPattern = ReadDeclAs<ClassTemplateDecl>(Record,Idx);
    if (D->isCanonicalDecl()) { // It's kept in the folding set.
      // Set this as, or find, the canonical declaration for this specialization
      ClassTemplateSpecializationDecl *CanonSpec;
      if (ClassTemplatePartialSpecializationDecl *Partial =
              dyn_cast<ClassTemplatePartialSpecializationDecl>(D)) {
        CanonSpec = CanonPattern->getCommonPtr()->PartialSpecializations
            .GetOrInsertNode(Partial);
      } else {
        CanonSpec =
            CanonPattern->getCommonPtr()->Specializations.GetOrInsertNode(D);
      }
      // If there was already a canonical specialization, merge into it.
      if (CanonSpec != D) {
        mergeRedeclarable<TagDecl>(D, CanonSpec, Redecl);

        // This declaration might be a definition. Merge with any existing
        // definition.
        if (auto *DDD = D->DefinitionData.getNotUpdated()) {
          if (auto *CanonDD = CanonSpec->DefinitionData.getNotUpdated()) {
            MergeDefinitionData(CanonSpec, *DDD);
            Reader.PendingDefinitions.erase(D);
            Reader.MergedDeclContexts.insert(
                std::make_pair(D, CanonDD->Definition));
            D->IsCompleteDefinition = false;
            D->DefinitionData = CanonSpec->DefinitionData;
          } else {
            CanonSpec->DefinitionData = D->DefinitionData;
          }
        }
      }
    }
  }

  // Explicit info.
  if (TypeSourceInfo *TyInfo = GetTypeSourceInfo(Record, Idx)) {
    ClassTemplateSpecializationDecl::ExplicitSpecializationInfo *ExplicitInfo
        = new (C) ClassTemplateSpecializationDecl::ExplicitSpecializationInfo;
    ExplicitInfo->TypeAsWritten = TyInfo;
    ExplicitInfo->ExternLoc = ReadSourceLocation(Record, Idx);
    ExplicitInfo->TemplateKeywordLoc = ReadSourceLocation(Record, Idx);
    D->ExplicitInfo = ExplicitInfo;
  }

  return Redecl;
}

void ASTDeclReader::VisitClassTemplatePartialSpecializationDecl(
                                    ClassTemplatePartialSpecializationDecl *D) {
  RedeclarableResult Redecl = VisitClassTemplateSpecializationDeclImpl(D);

  D->TemplateParams = Reader.ReadTemplateParameterList(F, Record, Idx);
  D->ArgsAsWritten = Reader.ReadASTTemplateArgumentListInfo(F, Record, Idx);

  // These are read/set from/to the first declaration.
  if (ThisDeclID == Redecl.getFirstID()) {
    D->InstantiatedFromMember.setPointer(
      ReadDeclAs<ClassTemplatePartialSpecializationDecl>(Record, Idx));
    D->InstantiatedFromMember.setInt(Record[Idx++]);
  }
}

void ASTDeclReader::VisitClassScopeFunctionSpecializationDecl(
                                    ClassScopeFunctionSpecializationDecl *D) {
  VisitDecl(D);
  D->Specialization = ReadDeclAs<CXXMethodDecl>(Record, Idx);
}

void ASTDeclReader::VisitFunctionTemplateDecl(FunctionTemplateDecl *D) {
  RedeclarableResult Redecl = VisitRedeclarableTemplateDecl(D);

  if (ThisDeclID == Redecl.getFirstID()) {
    // This FunctionTemplateDecl owns a CommonPtr; read it.

    // Read the function specialization declaration IDs. The specializations
    // themselves will be loaded if they're needed.
    if (unsigned NumSpecs = Record[Idx++]) {
      // FIXME: Append specializations!
      FunctionTemplateDecl::Common *CommonPtr = D->getCommonPtr();
      CommonPtr->LazySpecializations = new (Reader.getContext())
          serialization::DeclID[NumSpecs + 1];
      CommonPtr->LazySpecializations[0] = NumSpecs;
      for (unsigned I = 0; I != NumSpecs; ++I)
        CommonPtr->LazySpecializations[I + 1] = ReadDeclID(Record, Idx);
    }
  }
}

/// TODO: Unify with ClassTemplateSpecializationDecl version?
///       May require unifying ClassTemplate(Partial)SpecializationDecl and
///        VarTemplate(Partial)SpecializationDecl with a new data
///        structure Template(Partial)SpecializationDecl, and
///        using Template(Partial)SpecializationDecl as input type.
ASTDeclReader::RedeclarableResult
ASTDeclReader::VisitVarTemplateSpecializationDeclImpl(
    VarTemplateSpecializationDecl *D) {
  RedeclarableResult Redecl = VisitVarDeclImpl(D);

  ASTContext &C = Reader.getContext();
  if (Decl *InstD = ReadDecl(Record, Idx)) {
    if (VarTemplateDecl *VTD = dyn_cast<VarTemplateDecl>(InstD)) {
      D->SpecializedTemplate = VTD;
    } else {
      SmallVector<TemplateArgument, 8> TemplArgs;
      Reader.ReadTemplateArgumentList(TemplArgs, F, Record, Idx);
      TemplateArgumentList *ArgList = TemplateArgumentList::CreateCopy(
          C, TemplArgs.data(), TemplArgs.size());
      VarTemplateSpecializationDecl::SpecializedPartialSpecialization *PS =
          new (C)
          VarTemplateSpecializationDecl::SpecializedPartialSpecialization();
      PS->PartialSpecialization =
          cast<VarTemplatePartialSpecializationDecl>(InstD);
      PS->TemplateArgs = ArgList;
      D->SpecializedTemplate = PS;
    }
  }

  // Explicit info.
  if (TypeSourceInfo *TyInfo = GetTypeSourceInfo(Record, Idx)) {
    VarTemplateSpecializationDecl::ExplicitSpecializationInfo *ExplicitInfo =
        new (C) VarTemplateSpecializationDecl::ExplicitSpecializationInfo;
    ExplicitInfo->TypeAsWritten = TyInfo;
    ExplicitInfo->ExternLoc = ReadSourceLocation(Record, Idx);
    ExplicitInfo->TemplateKeywordLoc = ReadSourceLocation(Record, Idx);
    D->ExplicitInfo = ExplicitInfo;
  }

  SmallVector<TemplateArgument, 8> TemplArgs;
  Reader.ReadTemplateArgumentList(TemplArgs, F, Record, Idx);
  D->TemplateArgs =
      TemplateArgumentList::CreateCopy(C, TemplArgs.data(), TemplArgs.size());
  D->PointOfInstantiation = ReadSourceLocation(Record, Idx);
  D->SpecializationKind = (TemplateSpecializationKind)Record[Idx++];

  bool writtenAsCanonicalDecl = Record[Idx++];
  if (writtenAsCanonicalDecl) {
    VarTemplateDecl *CanonPattern = ReadDeclAs<VarTemplateDecl>(Record, Idx);
    if (D->isCanonicalDecl()) { // It's kept in the folding set.
      if (VarTemplatePartialSpecializationDecl *Partial =
              dyn_cast<VarTemplatePartialSpecializationDecl>(D)) {
        CanonPattern->getCommonPtr()->PartialSpecializations
            .GetOrInsertNode(Partial);
      } else {
        CanonPattern->getCommonPtr()->Specializations.GetOrInsertNode(D);
      }
    }
  }

  return Redecl;
}

/// TODO: Unify with ClassTemplatePartialSpecializationDecl version?
///       May require unifying ClassTemplate(Partial)SpecializationDecl and
///        VarTemplate(Partial)SpecializationDecl with a new data
///        structure Template(Partial)SpecializationDecl, and
///        using Template(Partial)SpecializationDecl as input type.
void ASTDeclReader::VisitVarTemplatePartialSpecializationDecl(
    VarTemplatePartialSpecializationDecl *D) {
  RedeclarableResult Redecl = VisitVarTemplateSpecializationDeclImpl(D);

  D->TemplateParams = Reader.ReadTemplateParameterList(F, Record, Idx);
  D->ArgsAsWritten = Reader.ReadASTTemplateArgumentListInfo(F, Record, Idx);

  // These are read/set from/to the first declaration.
  if (ThisDeclID == Redecl.getFirstID()) {
    D->InstantiatedFromMember.setPointer(
        ReadDeclAs<VarTemplatePartialSpecializationDecl>(Record, Idx));
    D->InstantiatedFromMember.setInt(Record[Idx++]);
  }
}

void ASTDeclReader::VisitTemplateTypeParmDecl(TemplateTypeParmDecl *D) {
  VisitTypeDecl(D);

  D->setDeclaredWithTypename(Record[Idx++]);

  bool Inherited = Record[Idx++];
  TypeSourceInfo *DefArg = GetTypeSourceInfo(Record, Idx);
  D->setDefaultArgument(DefArg, Inherited);
}

void ASTDeclReader::VisitNonTypeTemplateParmDecl(NonTypeTemplateParmDecl *D) {
  VisitDeclaratorDecl(D);
  // TemplateParmPosition.
  D->setDepth(Record[Idx++]);
  D->setPosition(Record[Idx++]);
  if (D->isExpandedParameterPack()) {
    void **Data = reinterpret_cast<void **>(D + 1);
    for (unsigned I = 0, N = D->getNumExpansionTypes(); I != N; ++I) {
      Data[2*I] = Reader.readType(F, Record, Idx).getAsOpaquePtr();
      Data[2*I + 1] = GetTypeSourceInfo(Record, Idx);
    }
  } else {
    // Rest of NonTypeTemplateParmDecl.
    D->ParameterPack = Record[Idx++];
    if (Record[Idx++]) {
      Expr *DefArg = Reader.ReadExpr(F);
      bool Inherited = Record[Idx++];
      D->setDefaultArgument(DefArg, Inherited);
   }
  }
}

void ASTDeclReader::VisitTemplateTemplateParmDecl(TemplateTemplateParmDecl *D) {
  VisitTemplateDecl(D);
  // TemplateParmPosition.
  D->setDepth(Record[Idx++]);
  D->setPosition(Record[Idx++]);
  if (D->isExpandedParameterPack()) {
    void **Data = reinterpret_cast<void **>(D + 1);
    for (unsigned I = 0, N = D->getNumExpansionTemplateParameters();
         I != N; ++I)
      Data[I] = Reader.ReadTemplateParameterList(F, Record, Idx);
  } else {
    // Rest of TemplateTemplateParmDecl.
    TemplateArgumentLoc Arg = Reader.ReadTemplateArgumentLoc(F, Record, Idx);
    bool IsInherited = Record[Idx++];
    D->setDefaultArgument(Arg, IsInherited);
    D->ParameterPack = Record[Idx++];
  }
}

void ASTDeclReader::VisitTypeAliasTemplateDecl(TypeAliasTemplateDecl *D) {
  VisitRedeclarableTemplateDecl(D);
}

void ASTDeclReader::VisitStaticAssertDecl(StaticAssertDecl *D) {
  VisitDecl(D);
  D->AssertExprAndFailed.setPointer(Reader.ReadExpr(F));
  D->AssertExprAndFailed.setInt(Record[Idx++]);
  D->Message = cast<StringLiteral>(Reader.ReadExpr(F));
  D->RParenLoc = ReadSourceLocation(Record, Idx);
}

void ASTDeclReader::VisitEmptyDecl(EmptyDecl *D) {
  VisitDecl(D);
}

std::pair<uint64_t, uint64_t>
ASTDeclReader::VisitDeclContext(DeclContext *DC) {
  uint64_t LexicalOffset = Record[Idx++];
  uint64_t VisibleOffset = Record[Idx++];
  return std::make_pair(LexicalOffset, VisibleOffset);
}

template <typename T>
ASTDeclReader::RedeclarableResult 
ASTDeclReader::VisitRedeclarable(Redeclarable<T> *D) {
  DeclID FirstDeclID = ReadDeclID(Record, Idx);
  
  // 0 indicates that this declaration was the only declaration of its entity,
  // and is used for space optimization.
  if (FirstDeclID == 0)
    FirstDeclID = ThisDeclID;
  
  T *FirstDecl = cast_or_null<T>(Reader.GetDecl(FirstDeclID));
  if (FirstDecl != D) {
    // We delay loading of the redeclaration chain to avoid deeply nested calls.
    // We temporarily set the first (canonical) declaration as the previous one
    // which is the one that matters and mark the real previous DeclID to be
    // loaded & attached later on.
    D->RedeclLink = Redeclarable<T>::PreviousDeclLink(FirstDecl);
  }    
  
  // Note that this declaration has been deserialized.
  Reader.RedeclsDeserialized.insert(static_cast<T *>(D));
                             
  // The result structure takes care to note that we need to load the 
  // other declaration chains for this ID.
  return RedeclarableResult(Reader, FirstDeclID,
                            static_cast<T *>(D)->getKind());
}

/// \brief Attempts to merge the given declaration (D) with another declaration
/// of the same entity.
template<typename T>
void ASTDeclReader::mergeRedeclarable(Redeclarable<T> *D,
                                      RedeclarableResult &Redecl,
                                      DeclID TemplatePatternID) {
  // If modules are not available, there is no reason to perform this merge.
  if (!Reader.getContext().getLangOpts().Modules)
    return;

  if (FindExistingResult ExistingRes = findExisting(static_cast<T*>(D)))
    if (T *Existing = ExistingRes)
      mergeRedeclarable(D, Existing, Redecl, TemplatePatternID);
}

/// \brief "Cast" to type T, asserting if we don't have an implicit conversion.
/// We use this to put code in a template that will only be valid for certain
/// instantiations.
template<typename T> static T assert_cast(T t) { return t; }
template<typename T> static T assert_cast(...) {
  llvm_unreachable("bad assert_cast");
}

/// \brief Merge together the pattern declarations from two template
/// declarations.
void ASTDeclReader::mergeTemplatePattern(RedeclarableTemplateDecl *D,
                                         RedeclarableTemplateDecl *Existing,
                                         DeclID DsID) {
  auto *DPattern = D->getTemplatedDecl();
  auto *ExistingPattern = Existing->getTemplatedDecl();
  RedeclarableResult Result(Reader, DsID, DPattern->getKind());
  if (auto *DClass = dyn_cast<CXXRecordDecl>(DPattern))
    // FIXME: Merge definitions here, if both declarations had definitions.
    return mergeRedeclarable(DClass, cast<TagDecl>(ExistingPattern),
                             Result);
  if (auto *DFunction = dyn_cast<FunctionDecl>(DPattern))
    return mergeRedeclarable(DFunction, cast<FunctionDecl>(ExistingPattern),
                             Result);
  if (auto *DVar = dyn_cast<VarDecl>(DPattern))
    return mergeRedeclarable(DVar, cast<VarDecl>(ExistingPattern), Result);
  llvm_unreachable("merged an unknown kind of redeclarable template");
}

/// \brief Attempts to merge the given declaration (D) with another declaration
/// of the same entity.
template<typename T>
void ASTDeclReader::mergeRedeclarable(Redeclarable<T> *DBase, T *Existing,
                                      RedeclarableResult &Redecl,
                                      DeclID TemplatePatternID) {
  T *D = static_cast<T*>(DBase);
  T *ExistingCanon = Existing->getCanonicalDecl();
  T *DCanon = D->getCanonicalDecl();
  if (ExistingCanon != DCanon) {
    // Have our redeclaration link point back at the canonical declaration
    // of the existing declaration, so that this declaration has the 
    // appropriate canonical declaration.
    D->RedeclLink = Redeclarable<T>::PreviousDeclLink(ExistingCanon);

    // When we merge a namespace, update its pointer to the first namespace.
    if (auto *Namespace = dyn_cast<NamespaceDecl>(D))
      Namespace->AnonOrFirstNamespaceAndInline.setPointer(
          assert_cast<NamespaceDecl*>(ExistingCanon));

    // When we merge a template, merge its pattern.
    if (auto *DTemplate = dyn_cast<RedeclarableTemplateDecl>(D))
      mergeTemplatePattern(
          DTemplate, assert_cast<RedeclarableTemplateDecl*>(ExistingCanon),
          TemplatePatternID);

    // Don't introduce DCanon into the set of pending declaration chains.
    Redecl.suppress();

    // Introduce ExistingCanon into the set of pending declaration chains,
    // if in fact it came from a module file.
    if (ExistingCanon->isFromASTFile()) {
      GlobalDeclID ExistingCanonID = ExistingCanon->getGlobalID();
      assert(ExistingCanonID && "Unrecorded canonical declaration ID?");
      if (Reader.PendingDeclChainsKnown.insert(ExistingCanonID))
        Reader.PendingDeclChains.push_back(ExistingCanonID);
    }

    // If this declaration was the canonical declaration, make a note of
    // that. We accept the linear algorithm here because the number of
    // unique canonical declarations of an entity should always be tiny.
    if (DCanon == D) {
      SmallVectorImpl<DeclID> &Merged = Reader.MergedDecls[ExistingCanon];
      if (std::find(Merged.begin(), Merged.end(), Redecl.getFirstID())
            == Merged.end())
        Merged.push_back(Redecl.getFirstID());

      // If ExistingCanon did not come from a module file, introduce the
      // first declaration that *does* come from a module file to the
      // set of pending declaration chains, so that we merge this
      // declaration.
      if (!ExistingCanon->isFromASTFile() &&
          Reader.PendingDeclChainsKnown.insert(Redecl.getFirstID()))
        Reader.PendingDeclChains.push_back(Merged[0]);
    }
  }
}

/// \brief Attempts to merge the given declaration (D) with another declaration
/// of the same entity, for the case where the entity is not actually
/// redeclarable. This happens, for instance, when merging the fields of
/// identical class definitions from two different modules.
template<typename T>
void ASTDeclReader::mergeMergeable(Mergeable<T> *D) {
  // If modules are not available, there is no reason to perform this merge.
  if (!Reader.getContext().getLangOpts().Modules)
    return;

  // ODR-based merging is only performed in C++. In C, identically-named things
  // in different translation units are not redeclarations (but may still have
  // compatible types).
  if (!Reader.getContext().getLangOpts().CPlusPlus)
    return;

  if (FindExistingResult ExistingRes = findExisting(static_cast<T*>(D)))
    if (T *Existing = ExistingRes)
      Reader.Context.setPrimaryMergedDecl(static_cast<T*>(D),
                                          Existing->getCanonicalDecl());
}

void ASTDeclReader::VisitOMPThreadPrivateDecl(OMPThreadPrivateDecl *D) {
  VisitDecl(D);
  unsigned NumVars = D->varlist_size();
  SmallVector<Expr *, 16> Vars;
  Vars.reserve(NumVars);
  for (unsigned i = 0; i != NumVars; ++i) {
    Vars.push_back(Reader.ReadExpr(F));
  }
  D->setVars(Vars);
}

//===----------------------------------------------------------------------===//
// Attribute Reading
//===----------------------------------------------------------------------===//

/// \brief Reads attributes from the current stream position.
void ASTReader::ReadAttributes(ModuleFile &F, AttrVec &Attrs,
                               const RecordData &Record, unsigned &Idx) {
  for (unsigned i = 0, e = Record[Idx++]; i != e; ++i) {
    Attr *New = nullptr;
    attr::Kind Kind = (attr::Kind)Record[Idx++];
    SourceRange Range = ReadSourceRange(F, Record, Idx);

#include "clang/Serialization/AttrPCHRead.inc"

    assert(New && "Unable to decode attribute?");
    Attrs.push_back(New);
  }
}

//===----------------------------------------------------------------------===//
// ASTReader Implementation
//===----------------------------------------------------------------------===//

/// \brief Note that we have loaded the declaration with the given
/// Index.
///
/// This routine notes that this declaration has already been loaded,
/// so that future GetDecl calls will return this declaration rather
/// than trying to load a new declaration.
inline void ASTReader::LoadedDecl(unsigned Index, Decl *D) {
  assert(!DeclsLoaded[Index] && "Decl loaded twice?");
  DeclsLoaded[Index] = D;
}


/// \brief Determine whether the consumer will be interested in seeing
/// this declaration (via HandleTopLevelDecl).
///
/// This routine should return true for anything that might affect
/// code generation, e.g., inline function definitions, Objective-C
/// declarations with metadata, etc.
static bool isConsumerInterestedIn(Decl *D, bool HasBody) {
  // An ObjCMethodDecl is never considered as "interesting" because its
  // implementation container always is.

  if (isa<FileScopeAsmDecl>(D) || 
      isa<ObjCProtocolDecl>(D) || 
      isa<ObjCImplDecl>(D) ||
      isa<ImportDecl>(D))
    return true;
  if (VarDecl *Var = dyn_cast<VarDecl>(D))
    return Var->isFileVarDecl() &&
           Var->isThisDeclarationADefinition() == VarDecl::Definition;
  if (FunctionDecl *Func = dyn_cast<FunctionDecl>(D))
    return Func->doesThisDeclarationHaveABody() || HasBody;
  
  return false;
}

/// \brief Get the correct cursor and offset for loading a declaration.
ASTReader::RecordLocation
ASTReader::DeclCursorForID(DeclID ID, unsigned &RawLocation) {
  // See if there's an override.
  DeclReplacementMap::iterator It = ReplacedDecls.find(ID);
  if (It != ReplacedDecls.end()) {
    RawLocation = It->second.RawLoc;
    return RecordLocation(It->second.Mod, It->second.Offset);
  }

  GlobalDeclMapType::iterator I = GlobalDeclMap.find(ID);
  assert(I != GlobalDeclMap.end() && "Corrupted global declaration map");
  ModuleFile *M = I->second;
  const DeclOffset &
    DOffs =  M->DeclOffsets[ID - M->BaseDeclID - NUM_PREDEF_DECL_IDS];
  RawLocation = DOffs.Loc;
  return RecordLocation(M, DOffs.BitOffset);
}

ASTReader::RecordLocation ASTReader::getLocalBitOffset(uint64_t GlobalOffset) {
  ContinuousRangeMap<uint64_t, ModuleFile*, 4>::iterator I
    = GlobalBitOffsetsMap.find(GlobalOffset);

  assert(I != GlobalBitOffsetsMap.end() && "Corrupted global bit offsets map");
  return RecordLocation(I->second, GlobalOffset - I->second->GlobalBitOffset);
}

uint64_t ASTReader::getGlobalBitOffset(ModuleFile &M, uint32_t LocalOffset) {
  return LocalOffset + M.GlobalBitOffset;
}

static bool isSameTemplateParameterList(const TemplateParameterList *X,
                                        const TemplateParameterList *Y);

/// \brief Determine whether two template parameters are similar enough
/// that they may be used in declarations of the same template.
static bool isSameTemplateParameter(const NamedDecl *X,
                                    const NamedDecl *Y) {
  if (X->getKind() != Y->getKind())
    return false;

  if (const TemplateTypeParmDecl *TX = dyn_cast<TemplateTypeParmDecl>(X)) {
    const TemplateTypeParmDecl *TY = cast<TemplateTypeParmDecl>(Y);
    return TX->isParameterPack() == TY->isParameterPack();
  }

  if (const NonTypeTemplateParmDecl *TX = dyn_cast<NonTypeTemplateParmDecl>(X)) {
    const NonTypeTemplateParmDecl *TY = cast<NonTypeTemplateParmDecl>(Y);
    return TX->isParameterPack() == TY->isParameterPack() &&
           TX->getASTContext().hasSameType(TX->getType(), TY->getType());
  }

  const TemplateTemplateParmDecl *TX = cast<TemplateTemplateParmDecl>(X);
  const TemplateTemplateParmDecl *TY = cast<TemplateTemplateParmDecl>(Y);
  return TX->isParameterPack() == TY->isParameterPack() &&
         isSameTemplateParameterList(TX->getTemplateParameters(),
                                     TY->getTemplateParameters());
}

/// \brief Determine whether two template parameter lists are similar enough
/// that they may be used in declarations of the same template.
static bool isSameTemplateParameterList(const TemplateParameterList *X,
                                        const TemplateParameterList *Y) {
  if (X->size() != Y->size())
    return false;

  for (unsigned I = 0, N = X->size(); I != N; ++I)
    if (!isSameTemplateParameter(X->getParam(I), Y->getParam(I)))
      return false;

  return true;
}

/// \brief Determine whether the two declarations refer to the same entity.
static bool isSameEntity(NamedDecl *X, NamedDecl *Y) {
  assert(X->getDeclName() == Y->getDeclName() && "Declaration name mismatch!");
  
  if (X == Y)
    return true;
  
  // Must be in the same context.
  if (!X->getDeclContext()->getRedeclContext()->Equals(
         Y->getDeclContext()->getRedeclContext()))
    return false;

  // Two typedefs refer to the same entity if they have the same underlying
  // type.
  if (TypedefNameDecl *TypedefX = dyn_cast<TypedefNameDecl>(X))
    if (TypedefNameDecl *TypedefY = dyn_cast<TypedefNameDecl>(Y))
      return X->getASTContext().hasSameType(TypedefX->getUnderlyingType(),
                                            TypedefY->getUnderlyingType());
  
  // Must have the same kind.
  if (X->getKind() != Y->getKind())
    return false;
    
  // Objective-C classes and protocols with the same name always match.
  if (isa<ObjCInterfaceDecl>(X) || isa<ObjCProtocolDecl>(X))
    return true;

  if (isa<ClassTemplateSpecializationDecl>(X)) {
    // No need to handle these here: we merge them when adding them to the
    // template.
    return false;
  }

  // Compatible tags match.
  if (TagDecl *TagX = dyn_cast<TagDecl>(X)) {
    TagDecl *TagY = cast<TagDecl>(Y);
    return (TagX->getTagKind() == TagY->getTagKind()) ||
      ((TagX->getTagKind() == TTK_Struct || TagX->getTagKind() == TTK_Class ||
        TagX->getTagKind() == TTK_Interface) &&
       (TagY->getTagKind() == TTK_Struct || TagY->getTagKind() == TTK_Class ||
        TagY->getTagKind() == TTK_Interface));
  }

  // Functions with the same type and linkage match.
  // FIXME: This needs to cope with function template specializations,
  // merging of prototyped/non-prototyped functions, etc.
  if (FunctionDecl *FuncX = dyn_cast<FunctionDecl>(X)) {
    FunctionDecl *FuncY = cast<FunctionDecl>(Y);
    return (FuncX->getLinkageInternal() == FuncY->getLinkageInternal()) &&
      FuncX->getASTContext().hasSameType(FuncX->getType(), FuncY->getType());
  }

  // Variables with the same type and linkage match.
  if (VarDecl *VarX = dyn_cast<VarDecl>(X)) {
    VarDecl *VarY = cast<VarDecl>(Y);
    return (VarX->getLinkageInternal() == VarY->getLinkageInternal()) &&
      VarX->getASTContext().hasSameType(VarX->getType(), VarY->getType());
  }

  // Namespaces with the same name and inlinedness match.
  if (NamespaceDecl *NamespaceX = dyn_cast<NamespaceDecl>(X)) {
    NamespaceDecl *NamespaceY = cast<NamespaceDecl>(Y);
    return NamespaceX->isInline() == NamespaceY->isInline();
  }

  // Identical template names and kinds match if their template parameter lists
  // and patterns match.
  if (TemplateDecl *TemplateX = dyn_cast<TemplateDecl>(X)) {
    TemplateDecl *TemplateY = cast<TemplateDecl>(Y);
    return isSameEntity(TemplateX->getTemplatedDecl(),
                        TemplateY->getTemplatedDecl()) &&
           isSameTemplateParameterList(TemplateX->getTemplateParameters(),
                                       TemplateY->getTemplateParameters());
  }

  // Fields with the same name and the same type match.
  if (FieldDecl *FDX = dyn_cast<FieldDecl>(X)) {
    FieldDecl *FDY = cast<FieldDecl>(Y);
    // FIXME: Diagnose if the types don't match.
    // FIXME: Also check the bitwidth is odr-equivalent, if any.
    return X->getASTContext().hasSameType(FDX->getType(), FDY->getType());
  }

  // Enumerators with the same name match.
  if (isa<EnumConstantDecl>(X))
    // FIXME: Also check the value is odr-equivalent.
    return true;

  // Using shadow declarations with the same target match.
  if (UsingShadowDecl *USX = dyn_cast<UsingShadowDecl>(X)) {
    UsingShadowDecl *USY = cast<UsingShadowDecl>(Y);
    return USX->getTargetDecl() == USY->getTargetDecl();
  }

  // FIXME: Many other cases to implement.
  return false;
}

/// Find the context in which we should search for previous declarations when
/// looking for declarations to merge.
static DeclContext *getPrimaryContextForMerging(DeclContext *DC) {
  if (NamespaceDecl *ND = dyn_cast<NamespaceDecl>(DC))
    return ND->getOriginalNamespace();

  if (CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(DC))
    return RD->getDefinition();

  if (EnumDecl *ED = dyn_cast<EnumDecl>(DC))
    return ED->getASTContext().getLangOpts().CPlusPlus? ED->getDefinition()
                                                      : nullptr;

  return nullptr;
}

ASTDeclReader::FindExistingResult::~FindExistingResult() {
  if (!AddResult || Existing)
    return;

  DeclContext *DC = New->getDeclContext()->getRedeclContext();
  if (DC->isTranslationUnit() && Reader.SemaObj) {
    Reader.SemaObj->IdResolver.tryAddTopLevelDecl(New, New->getDeclName());
  } else if (DeclContext *MergeDC = getPrimaryContextForMerging(DC)) {
    // Add the declaration to its redeclaration context so later merging
    // lookups will find it.
    MergeDC->makeDeclVisibleInContextImpl(New, /*Internal*/true);
  }
}

ASTDeclReader::FindExistingResult ASTDeclReader::findExisting(NamedDecl *D) {
  DeclarationName Name = D->getDeclName();
  if (!Name) {
    // Don't bother trying to find unnamed declarations.
    FindExistingResult Result(Reader, D, /*Existing=*/nullptr);
    Result.suppress();
    return Result;
  }

  // FIXME: Bail out for non-canonical declarations. We will have performed any
  // necessary merging already.

  DeclContext *DC = D->getDeclContext()->getRedeclContext();
  if (DC->isTranslationUnit() && Reader.SemaObj) {
    IdentifierResolver &IdResolver = Reader.SemaObj->IdResolver;

    // Temporarily consider the identifier to be up-to-date. We don't want to
    // cause additional lookups here.
    class UpToDateIdentifierRAII {
      IdentifierInfo *II;
      bool WasOutToDate;

    public:
      explicit UpToDateIdentifierRAII(IdentifierInfo *II)
        : II(II), WasOutToDate(false)
      {
        if (II) {
          WasOutToDate = II->isOutOfDate();
          if (WasOutToDate)
            II->setOutOfDate(false);
        }
      }

      ~UpToDateIdentifierRAII() {
        if (WasOutToDate)
          II->setOutOfDate(true);
      }
    } UpToDate(Name.getAsIdentifierInfo());

    for (IdentifierResolver::iterator I = IdResolver.begin(Name), 
                                   IEnd = IdResolver.end();
         I != IEnd; ++I) {
      if (isSameEntity(*I, D))
        return FindExistingResult(Reader, D, *I);
    }
  } else if (DeclContext *MergeDC = getPrimaryContextForMerging(DC)) {
    DeclContext::lookup_result R = MergeDC->noload_lookup(Name);
    for (DeclContext::lookup_iterator I = R.begin(), E = R.end(); I != E; ++I) {
      if (isSameEntity(*I, D))
        return FindExistingResult(Reader, D, *I);
    }
  } else {
    // Not in a mergeable context.
    return FindExistingResult(Reader);
  }

  // If this declaration is from a merged context, make a note that we need to
  // check that the canonical definition of that context contains the decl.
  //
  // FIXME: We should do something similar if we merge two definitions of the
  // same template specialization into the same CXXRecordDecl.
  if (Reader.MergedDeclContexts.count(D->getLexicalDeclContext()))
    Reader.PendingOdrMergeChecks.push_back(D);

  return FindExistingResult(Reader, D, /*Existing=*/nullptr);
}

template<typename DeclT>
void ASTDeclReader::attachPreviousDeclImpl(Redeclarable<DeclT> *D,
                                           Decl *Previous) {
  D->RedeclLink.setPrevious(cast<DeclT>(Previous));
}
void ASTDeclReader::attachPreviousDeclImpl(...) {
  llvm_unreachable("attachPreviousDecl on non-redeclarable declaration");
}

void ASTDeclReader::attachPreviousDecl(Decl *D, Decl *Previous) {
  assert(D && Previous);

  switch (D->getKind()) {
#define ABSTRACT_DECL(TYPE)
#define DECL(TYPE, BASE)                                   \
  case Decl::TYPE:                                         \
    attachPreviousDeclImpl(cast<TYPE##Decl>(D), Previous); \
    break;
#include "clang/AST/DeclNodes.inc"
  }

  // If the declaration was visible in one module, a redeclaration of it in
  // another module remains visible even if it wouldn't be visible by itself.
  //
  // FIXME: In this case, the declaration should only be visible if a module
  //        that makes it visible has been imported.
  D->IdentifierNamespace |=
      Previous->IdentifierNamespace &
      (Decl::IDNS_Ordinary | Decl::IDNS_Tag | Decl::IDNS_Type);
}

template<typename DeclT>
void ASTDeclReader::attachLatestDeclImpl(Redeclarable<DeclT> *D, Decl *Latest) {
  D->RedeclLink.setLatest(cast<DeclT>(Latest));
}
void ASTDeclReader::attachLatestDeclImpl(...) {
  llvm_unreachable("attachLatestDecl on non-redeclarable declaration");
}

void ASTDeclReader::attachLatestDecl(Decl *D, Decl *Latest) {
  assert(D && Latest);

  switch (D->getKind()) {
#define ABSTRACT_DECL(TYPE)
#define DECL(TYPE, BASE)                                  \
  case Decl::TYPE:                                        \
    attachLatestDeclImpl(cast<TYPE##Decl>(D), Latest); \
    break;
#include "clang/AST/DeclNodes.inc"
  }
}

template<typename DeclT>
void ASTDeclReader::markIncompleteDeclChainImpl(Redeclarable<DeclT> *D) {
  D->RedeclLink.markIncomplete();
}
void ASTDeclReader::markIncompleteDeclChainImpl(...) {
  llvm_unreachable("markIncompleteDeclChain on non-redeclarable declaration");
}

void ASTReader::markIncompleteDeclChain(Decl *D) {
  switch (D->getKind()) {
#define ABSTRACT_DECL(TYPE)
#define DECL(TYPE, BASE)                                             \
  case Decl::TYPE:                                                   \
    ASTDeclReader::markIncompleteDeclChainImpl(cast<TYPE##Decl>(D)); \
    break;
#include "clang/AST/DeclNodes.inc"
  }
}

ASTReader::MergedDeclsMap::iterator
ASTReader::combineStoredMergedDecls(Decl *Canon, GlobalDeclID CanonID) {
  // If we don't have any stored merged declarations, just look in the
  // merged declarations set.
  StoredMergedDeclsMap::iterator StoredPos = StoredMergedDecls.find(CanonID);
  if (StoredPos == StoredMergedDecls.end())
    return MergedDecls.find(Canon);

  // Append the stored merged declarations to the merged declarations set.
  MergedDeclsMap::iterator Pos = MergedDecls.find(Canon);
  if (Pos == MergedDecls.end())
    Pos = MergedDecls.insert(std::make_pair(Canon, 
                                            SmallVector<DeclID, 2>())).first;
  Pos->second.append(StoredPos->second.begin(), StoredPos->second.end());
  StoredMergedDecls.erase(StoredPos);
  
  // Sort and uniquify the set of merged declarations.
  llvm::array_pod_sort(Pos->second.begin(), Pos->second.end());
  Pos->second.erase(std::unique(Pos->second.begin(), Pos->second.end()),
                    Pos->second.end());
  return Pos;
}

/// \brief Read the declaration at the given offset from the AST file.
Decl *ASTReader::ReadDeclRecord(DeclID ID) {
  unsigned Index = ID - NUM_PREDEF_DECL_IDS;
  unsigned RawLocation = 0;
  RecordLocation Loc = DeclCursorForID(ID, RawLocation);
  llvm::BitstreamCursor &DeclsCursor = Loc.F->DeclsCursor;
  // Keep track of where we are in the stream, then jump back there
  // after reading this declaration.
  SavedStreamPosition SavedPosition(DeclsCursor);

  ReadingKindTracker ReadingKind(Read_Decl, *this);

  // Note that we are loading a declaration record.
  Deserializing ADecl(this);

  DeclsCursor.JumpToBit(Loc.Offset);
  RecordData Record;
  unsigned Code = DeclsCursor.ReadCode();
  unsigned Idx = 0;
  ASTDeclReader Reader(*this, *Loc.F, ID, RawLocation, Record,Idx);

  Decl *D = nullptr;
  switch ((DeclCode)DeclsCursor.readRecord(Code, Record)) {
  case DECL_CONTEXT_LEXICAL:
  case DECL_CONTEXT_VISIBLE:
    llvm_unreachable("Record cannot be de-serialized with ReadDeclRecord");
  case DECL_TYPEDEF:
    D = TypedefDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_TYPEALIAS:
    D = TypeAliasDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_ENUM:
    D = EnumDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_RECORD:
    D = RecordDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_ENUM_CONSTANT:
    D = EnumConstantDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_FUNCTION:
    D = FunctionDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_LINKAGE_SPEC:
    D = LinkageSpecDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_LABEL:
    D = LabelDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_NAMESPACE:
    D = NamespaceDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_NAMESPACE_ALIAS:
    D = NamespaceAliasDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_USING:
    D = UsingDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_USING_SHADOW:
    D = UsingShadowDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_USING_DIRECTIVE:
    D = UsingDirectiveDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_UNRESOLVED_USING_VALUE:
    D = UnresolvedUsingValueDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_UNRESOLVED_USING_TYPENAME:
    D = UnresolvedUsingTypenameDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_CXX_RECORD:
    D = CXXRecordDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_CXX_METHOD:
    D = CXXMethodDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_CXX_CONSTRUCTOR:
    D = CXXConstructorDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_CXX_DESTRUCTOR:
    D = CXXDestructorDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_CXX_CONVERSION:
    D = CXXConversionDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_ACCESS_SPEC:
    D = AccessSpecDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_FRIEND:
    D = FriendDecl::CreateDeserialized(Context, ID, Record[Idx++]);
    break;
  case DECL_FRIEND_TEMPLATE:
    D = FriendTemplateDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_CLASS_TEMPLATE:
    D = ClassTemplateDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_CLASS_TEMPLATE_SPECIALIZATION:
    D = ClassTemplateSpecializationDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_CLASS_TEMPLATE_PARTIAL_SPECIALIZATION:
    D = ClassTemplatePartialSpecializationDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_VAR_TEMPLATE:
    D = VarTemplateDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_VAR_TEMPLATE_SPECIALIZATION:
    D = VarTemplateSpecializationDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_VAR_TEMPLATE_PARTIAL_SPECIALIZATION:
    D = VarTemplatePartialSpecializationDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_CLASS_SCOPE_FUNCTION_SPECIALIZATION:
    D = ClassScopeFunctionSpecializationDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_FUNCTION_TEMPLATE:
    D = FunctionTemplateDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_TEMPLATE_TYPE_PARM:
    D = TemplateTypeParmDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_NON_TYPE_TEMPLATE_PARM:
    D = NonTypeTemplateParmDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_EXPANDED_NON_TYPE_TEMPLATE_PARM_PACK:
    D = NonTypeTemplateParmDecl::CreateDeserialized(Context, ID, Record[Idx++]);
    break;
  case DECL_TEMPLATE_TEMPLATE_PARM:
    D = TemplateTemplateParmDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_EXPANDED_TEMPLATE_TEMPLATE_PARM_PACK:
    D = TemplateTemplateParmDecl::CreateDeserialized(Context, ID,
                                                     Record[Idx++]);
    break;
  case DECL_TYPE_ALIAS_TEMPLATE:
    D = TypeAliasTemplateDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_STATIC_ASSERT:
    D = StaticAssertDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_OBJC_METHOD:
    D = ObjCMethodDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_OBJC_INTERFACE:
    D = ObjCInterfaceDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_OBJC_IVAR:
    D = ObjCIvarDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_OBJC_PROTOCOL:
    D = ObjCProtocolDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_OBJC_AT_DEFS_FIELD:
    D = ObjCAtDefsFieldDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_OBJC_CATEGORY:
    D = ObjCCategoryDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_OBJC_CATEGORY_IMPL:
    D = ObjCCategoryImplDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_OBJC_IMPLEMENTATION:
    D = ObjCImplementationDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_OBJC_COMPATIBLE_ALIAS:
    D = ObjCCompatibleAliasDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_OBJC_PROPERTY:
    D = ObjCPropertyDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_OBJC_PROPERTY_IMPL:
    D = ObjCPropertyImplDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_FIELD:
    D = FieldDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_INDIRECTFIELD:
    D = IndirectFieldDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_VAR:
    D = VarDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_IMPLICIT_PARAM:
    D = ImplicitParamDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_PARM_VAR:
    D = ParmVarDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_FILE_SCOPE_ASM:
    D = FileScopeAsmDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_BLOCK:
    D = BlockDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_MS_PROPERTY:
    D = MSPropertyDecl::CreateDeserialized(Context, ID);
    break;
  case DECL_CAPTURED:
    D = CapturedDecl::CreateDeserialized(Context, ID, Record[Idx++]);
    break;
  case DECL_CXX_BASE_SPECIFIERS:
    Error("attempt to read a C++ base-specifier record as a declaration");
    return nullptr;
  case DECL_IMPORT:
    // Note: last entry of the ImportDecl record is the number of stored source 
    // locations.
    D = ImportDecl::CreateDeserialized(Context, ID, Record.back());
    break;
  case DECL_OMP_THREADPRIVATE:
    D = OMPThreadPrivateDecl::CreateDeserialized(Context, ID, Record[Idx++]);
    break;
  case DECL_EMPTY:
    D = EmptyDecl::CreateDeserialized(Context, ID);
    break;
  }

  assert(D && "Unknown declaration reading AST file");
  LoadedDecl(Index, D);
  // Set the DeclContext before doing any deserialization, to make sure internal
  // calls to Decl::getASTContext() by Decl's methods will find the
  // TranslationUnitDecl without crashing.
  D->setDeclContext(Context.getTranslationUnitDecl());
  Reader.Visit(D);

  // If this declaration is also a declaration context, get the
  // offsets for its tables of lexical and visible declarations.
  if (DeclContext *DC = dyn_cast<DeclContext>(D)) {
    // FIXME: This should really be
    //     DeclContext *LookupDC = DC->getPrimaryContext();
    // but that can walk the redeclaration chain, which might not work yet.
    DeclContext *LookupDC = DC;
    if (isa<NamespaceDecl>(DC))
      LookupDC = DC->getPrimaryContext();
    std::pair<uint64_t, uint64_t> Offsets = Reader.VisitDeclContext(DC);
    if (Offsets.first || Offsets.second) {
      if (Offsets.first != 0)
        DC->setHasExternalLexicalStorage(true);
      if (Offsets.second != 0)
        LookupDC->setHasExternalVisibleStorage(true);
      if (ReadDeclContextStorage(*Loc.F, DeclsCursor, Offsets, 
                                 Loc.F->DeclContextInfos[DC]))
        return nullptr;
    }

    // Now add the pending visible updates for this decl context, if it has any.
    DeclContextVisibleUpdatesPending::iterator I =
        PendingVisibleUpdates.find(ID);
    if (I != PendingVisibleUpdates.end()) {
      // There are updates. This means the context has external visible
      // storage, even if the original stored version didn't.
      LookupDC->setHasExternalVisibleStorage(true);
      for (const auto &Update : I->second) {
        DeclContextInfo &Info = Update.second->DeclContextInfos[DC];
        delete Info.NameLookupTableData;
        Info.NameLookupTableData = Update.first;
      }
      PendingVisibleUpdates.erase(I);
    }
  }
  assert(Idx == Record.size());

  // Load any relevant update records.
  PendingUpdateRecords.push_back(std::make_pair(ID, D));

  // Load the categories after recursive loading is finished.
  if (ObjCInterfaceDecl *Class = dyn_cast<ObjCInterfaceDecl>(D))
    if (Class->isThisDeclarationADefinition())
      loadObjCCategories(ID, Class);
  
  // If we have deserialized a declaration that has a definition the
  // AST consumer might need to know about, queue it.
  // We don't pass it to the consumer immediately because we may be in recursive
  // loading, and some declarations may still be initializing.
  if (isConsumerInterestedIn(D, Reader.hasPendingBody()))
    InterestingDecls.push_back(D);

  return D;
}

void ASTReader::loadDeclUpdateRecords(serialization::DeclID ID, Decl *D) {
  // The declaration may have been modified by files later in the chain.
  // If this is the case, read the record containing the updates from each file
  // and pass it to ASTDeclReader to make the modifications.
  DeclUpdateOffsetsMap::iterator UpdI = DeclUpdateOffsets.find(ID);
  if (UpdI != DeclUpdateOffsets.end()) {
    FileOffsetsTy &UpdateOffsets = UpdI->second;
    bool WasInteresting = isConsumerInterestedIn(D, false);
    for (FileOffsetsTy::iterator
         I = UpdateOffsets.begin(), E = UpdateOffsets.end(); I != E; ++I) {
      ModuleFile *F = I->first;
      uint64_t Offset = I->second;
      llvm::BitstreamCursor &Cursor = F->DeclsCursor;
      SavedStreamPosition SavedPosition(Cursor);
      Cursor.JumpToBit(Offset);
      RecordData Record;
      unsigned Code = Cursor.ReadCode();
      unsigned RecCode = Cursor.readRecord(Code, Record);
      (void)RecCode;
      assert(RecCode == DECL_UPDATES && "Expected DECL_UPDATES record!");

      unsigned Idx = 0;
      ASTDeclReader Reader(*this, *F, ID, 0, Record, Idx);
      Reader.UpdateDecl(D, *F, Record);

      // We might have made this declaration interesting. If so, remember that
      // we need to hand it off to the consumer.
      if (!WasInteresting &&
          isConsumerInterestedIn(D, Reader.hasPendingBody())) {
        InterestingDecls.push_back(D);
        WasInteresting = true;
      }
    }
  }
}

namespace {
  /// \brief Module visitor class that finds all of the redeclarations of a 
  /// 
  class RedeclChainVisitor {
    ASTReader &Reader;
    SmallVectorImpl<DeclID> &SearchDecls;
    llvm::SmallPtrSet<Decl *, 16> &Deserialized;
    GlobalDeclID CanonID;
    SmallVector<Decl *, 4> Chain;
    
  public:
    RedeclChainVisitor(ASTReader &Reader, SmallVectorImpl<DeclID> &SearchDecls,
                       llvm::SmallPtrSet<Decl *, 16> &Deserialized,
                       GlobalDeclID CanonID)
      : Reader(Reader), SearchDecls(SearchDecls), Deserialized(Deserialized),
        CanonID(CanonID) { 
      for (unsigned I = 0, N = SearchDecls.size(); I != N; ++I)
        addToChain(Reader.GetDecl(SearchDecls[I]));
    }
    
    static bool visit(ModuleFile &M, bool Preorder, void *UserData) {
      if (Preorder)
        return false;
      
      return static_cast<RedeclChainVisitor *>(UserData)->visit(M);
    }
    
    void addToChain(Decl *D) {
      if (!D)
        return;
      
      if (Deserialized.erase(D))
        Chain.push_back(D);
    }
    
    void searchForID(ModuleFile &M, GlobalDeclID GlobalID) {
      // Map global ID of the first declaration down to the local ID
      // used in this module file.
      DeclID ID = Reader.mapGlobalIDToModuleFileGlobalID(M, GlobalID);
      if (!ID)
        return;
      
      // Perform a binary search to find the local redeclarations for this
      // declaration (if any).
      const LocalRedeclarationsInfo Compare = { ID, 0 };
      const LocalRedeclarationsInfo *Result
        = std::lower_bound(M.RedeclarationsMap,
                           M.RedeclarationsMap + M.LocalNumRedeclarationsInMap, 
                           Compare);
      if (Result == M.RedeclarationsMap + M.LocalNumRedeclarationsInMap ||
          Result->FirstID != ID) {
        // If we have a previously-canonical singleton declaration that was 
        // merged into another redeclaration chain, create a trivial chain
        // for this single declaration so that it will get wired into the 
        // complete redeclaration chain.
        if (GlobalID != CanonID && 
            GlobalID - NUM_PREDEF_DECL_IDS >= M.BaseDeclID && 
            GlobalID - NUM_PREDEF_DECL_IDS < M.BaseDeclID + M.LocalNumDecls) {
          addToChain(Reader.GetDecl(GlobalID));
        }
        
        return;
      }
      
      // Dig out all of the redeclarations.
      unsigned Offset = Result->Offset;
      unsigned N = M.RedeclarationChains[Offset];
      M.RedeclarationChains[Offset++] = 0; // Don't try to deserialize again
      for (unsigned I = 0; I != N; ++I)
        addToChain(Reader.GetLocalDecl(M, M.RedeclarationChains[Offset++]));
    }
    
    bool visit(ModuleFile &M) {
      // Visit each of the declarations.
      for (unsigned I = 0, N = SearchDecls.size(); I != N; ++I)
        searchForID(M, SearchDecls[I]);
      return false;
    }
    
    ArrayRef<Decl *> getChain() const {
      return Chain;
    }
  };
}

void ASTReader::loadPendingDeclChain(serialization::GlobalDeclID ID) {
  Decl *D = GetDecl(ID);  
  Decl *CanonDecl = D->getCanonicalDecl();
  
  // Determine the set of declaration IDs we'll be searching for.
  SmallVector<DeclID, 1> SearchDecls;
  GlobalDeclID CanonID = 0;
  if (D == CanonDecl) {
    SearchDecls.push_back(ID); // Always first.
    CanonID = ID;
  }
  MergedDeclsMap::iterator MergedPos = combineStoredMergedDecls(CanonDecl, ID);
  if (MergedPos != MergedDecls.end())
    SearchDecls.append(MergedPos->second.begin(), MergedPos->second.end());
  
  // Build up the list of redeclarations.
  RedeclChainVisitor Visitor(*this, SearchDecls, RedeclsDeserialized, CanonID);
  ModuleMgr.visitDepthFirst(&RedeclChainVisitor::visit, &Visitor);
  
  // Retrieve the chains.
  ArrayRef<Decl *> Chain = Visitor.getChain();
  if (Chain.empty())
    return;
    
  // Hook up the chains.
  Decl *MostRecent = CanonDecl->getMostRecentDecl();
  for (unsigned I = 0, N = Chain.size(); I != N; ++I) {
    if (Chain[I] == CanonDecl)
      continue;

    ASTDeclReader::attachPreviousDecl(Chain[I], MostRecent);
    MostRecent = Chain[I];
  }
  
  ASTDeclReader::attachLatestDecl(CanonDecl, MostRecent);  
}

namespace {
  /// \brief Given an ObjC interface, goes through the modules and links to the
  /// interface all the categories for it.
  class ObjCCategoriesVisitor {
    ASTReader &Reader;
    serialization::GlobalDeclID InterfaceID;
    ObjCInterfaceDecl *Interface;
    llvm::SmallPtrSet<ObjCCategoryDecl *, 16> &Deserialized;
    unsigned PreviousGeneration;
    ObjCCategoryDecl *Tail;
    llvm::DenseMap<DeclarationName, ObjCCategoryDecl *> NameCategoryMap;
    
    void add(ObjCCategoryDecl *Cat) {
      // Only process each category once.
      if (!Deserialized.erase(Cat))
        return;
      
      // Check for duplicate categories.
      if (Cat->getDeclName()) {
        ObjCCategoryDecl *&Existing = NameCategoryMap[Cat->getDeclName()];
        if (Existing && 
            Reader.getOwningModuleFile(Existing) 
                                          != Reader.getOwningModuleFile(Cat)) {
          // FIXME: We should not warn for duplicates in diamond:
          //
          //   MT     //
          //  /  \    //
          // ML  MR   //
          //  \  /    //
          //   MB     //
          //
          // If there are duplicates in ML/MR, there will be warning when 
          // creating MB *and* when importing MB. We should not warn when 
          // importing.
          Reader.Diag(Cat->getLocation(), diag::warn_dup_category_def)
            << Interface->getDeclName() << Cat->getDeclName();
          Reader.Diag(Existing->getLocation(), diag::note_previous_definition);
        } else if (!Existing) {
          // Record this category.
          Existing = Cat;
        }
      }
      
      // Add this category to the end of the chain.
      if (Tail)
        ASTDeclReader::setNextObjCCategory(Tail, Cat);
      else
        Interface->setCategoryListRaw(Cat);
      Tail = Cat;
    }
    
  public:
    ObjCCategoriesVisitor(ASTReader &Reader,
                          serialization::GlobalDeclID InterfaceID,
                          ObjCInterfaceDecl *Interface,
                        llvm::SmallPtrSet<ObjCCategoryDecl *, 16> &Deserialized,
                          unsigned PreviousGeneration)
      : Reader(Reader), InterfaceID(InterfaceID), Interface(Interface),
        Deserialized(Deserialized), PreviousGeneration(PreviousGeneration),
        Tail(nullptr)
    {
      // Populate the name -> category map with the set of known categories.
      for (auto *Cat : Interface->known_categories()) {
        if (Cat->getDeclName())
          NameCategoryMap[Cat->getDeclName()] = Cat;
        
        // Keep track of the tail of the category list.
        Tail = Cat;
      }
    }

    static bool visit(ModuleFile &M, void *UserData) {
      return static_cast<ObjCCategoriesVisitor *>(UserData)->visit(M);
    }

    bool visit(ModuleFile &M) {
      // If we've loaded all of the category information we care about from
      // this module file, we're done.
      if (M.Generation <= PreviousGeneration)
        return true;
      
      // Map global ID of the definition down to the local ID used in this 
      // module file. If there is no such mapping, we'll find nothing here
      // (or in any module it imports).
      DeclID LocalID = Reader.mapGlobalIDToModuleFileGlobalID(M, InterfaceID);
      if (!LocalID)
        return true;

      // Perform a binary search to find the local redeclarations for this
      // declaration (if any).
      const ObjCCategoriesInfo Compare = { LocalID, 0 };
      const ObjCCategoriesInfo *Result
        = std::lower_bound(M.ObjCCategoriesMap,
                           M.ObjCCategoriesMap + M.LocalNumObjCCategoriesInMap, 
                           Compare);
      if (Result == M.ObjCCategoriesMap + M.LocalNumObjCCategoriesInMap ||
          Result->DefinitionID != LocalID) {
        // We didn't find anything. If the class definition is in this module
        // file, then the module files it depends on cannot have any categories,
        // so suppress further lookup.
        return Reader.isDeclIDFromModule(InterfaceID, M);
      }
      
      // We found something. Dig out all of the categories.
      unsigned Offset = Result->Offset;
      unsigned N = M.ObjCCategories[Offset];
      M.ObjCCategories[Offset++] = 0; // Don't try to deserialize again
      for (unsigned I = 0; I != N; ++I)
        add(cast_or_null<ObjCCategoryDecl>(
              Reader.GetLocalDecl(M, M.ObjCCategories[Offset++])));
      return true;
    }
  };
}

void ASTReader::loadObjCCategories(serialization::GlobalDeclID ID,
                                   ObjCInterfaceDecl *D,
                                   unsigned PreviousGeneration) {
  ObjCCategoriesVisitor Visitor(*this, ID, D, CategoriesDeserialized,
                                PreviousGeneration);
  ModuleMgr.visit(ObjCCategoriesVisitor::visit, &Visitor);
}

void ASTDeclReader::UpdateDecl(Decl *D, ModuleFile &ModuleFile,
                               const RecordData &Record) {
  while (Idx < Record.size()) {
    switch ((DeclUpdateKind)Record[Idx++]) {
    case UPD_CXX_ADDED_IMPLICIT_MEMBER: {
      Decl *MD = Reader.ReadDecl(ModuleFile, Record, Idx);
      assert(MD && "couldn't read decl from update record");
      cast<CXXRecordDecl>(D)->addedMember(MD);
      break;
    }

    case UPD_CXX_ADDED_TEMPLATE_SPECIALIZATION:
      // It will be added to the template's specializations set when loaded.
      (void)Reader.ReadDecl(ModuleFile, Record, Idx);
      break;

    case UPD_CXX_ADDED_ANONYMOUS_NAMESPACE: {
      NamespaceDecl *Anon
        = Reader.ReadDeclAs<NamespaceDecl>(ModuleFile, Record, Idx);
      
      // Each module has its own anonymous namespace, which is disjoint from
      // any other module's anonymous namespaces, so don't attach the anonymous
      // namespace at all.
      if (ModuleFile.Kind != MK_Module) {
        if (TranslationUnitDecl *TU = dyn_cast<TranslationUnitDecl>(D))
          TU->setAnonymousNamespace(Anon);
        else
          cast<NamespaceDecl>(D)->setAnonymousNamespace(Anon);
      }
      break;
    }

    case UPD_CXX_INSTANTIATED_STATIC_DATA_MEMBER:
      cast<VarDecl>(D)->getMemberSpecializationInfo()->setPointOfInstantiation(
          Reader.ReadSourceLocation(ModuleFile, Record, Idx));
      break;

    case UPD_CXX_INSTANTIATED_FUNCTION_DEFINITION: {
      FunctionDecl *FD = cast<FunctionDecl>(D);
      if (Reader.PendingBodies[FD]) {
        // FIXME: Maybe check for ODR violations.
        // It's safe to stop now because this update record is always last.
        return;
      }

      if (Record[Idx++])
        FD->setImplicitlyInline();
      FD->setInnerLocStart(Reader.ReadSourceLocation(ModuleFile, Record, Idx));
      if (auto *CD = dyn_cast<CXXConstructorDecl>(FD))
        std::tie(CD->CtorInitializers, CD->NumCtorInitializers) =
            Reader.ReadCXXCtorInitializers(ModuleFile, Record, Idx);
      // Store the offset of the body so we can lazily load it later.
      Reader.PendingBodies[FD] = GetCurrentCursorOffset();
      HasPendingBody = true;
      assert(Idx == Record.size() && "lazy body must be last");
      break;
    }

    case UPD_CXX_INSTANTIATED_CLASS_DEFINITION: {
      auto *RD = cast<CXXRecordDecl>(D);
      bool HadDefinition = RD->getDefinition();
      ReadCXXRecordDefinition(RD);
      // Visible update is handled separately.
      uint64_t LexicalOffset = Record[Idx++];
      if (!HadDefinition && LexicalOffset) {
        RD->setHasExternalLexicalStorage(true);
        Reader.ReadDeclContextStorage(ModuleFile, ModuleFile.DeclsCursor,
                                          std::make_pair(LexicalOffset, 0),
                                          ModuleFile.DeclContextInfos[RD]);
      }

      auto TSK = (TemplateSpecializationKind)Record[Idx++];
      SourceLocation POI = Reader.ReadSourceLocation(ModuleFile, Record, Idx);
      if (MemberSpecializationInfo *MSInfo =
              RD->getMemberSpecializationInfo()) {
        MSInfo->setTemplateSpecializationKind(TSK);
        MSInfo->setPointOfInstantiation(POI);
      } else {
        ClassTemplateSpecializationDecl *Spec =
            cast<ClassTemplateSpecializationDecl>(RD);
        Spec->setTemplateSpecializationKind(TSK);
        Spec->setPointOfInstantiation(POI);

        if (Record[Idx++]) {
          auto PartialSpec =
              ReadDeclAs<ClassTemplatePartialSpecializationDecl>(Record, Idx);
          SmallVector<TemplateArgument, 8> TemplArgs;
          Reader.ReadTemplateArgumentList(TemplArgs, F, Record, Idx);
          auto *TemplArgList = TemplateArgumentList::CreateCopy(
              Reader.getContext(), TemplArgs.data(), TemplArgs.size());
          Spec->setInstantiationOf(PartialSpec, TemplArgList);
        }
      }

      RD->setTagKind((TagTypeKind)Record[Idx++]);
      RD->setLocation(Reader.ReadSourceLocation(ModuleFile, Record, Idx));
      RD->setLocStart(Reader.ReadSourceLocation(ModuleFile, Record, Idx));
      RD->setRBraceLoc(Reader.ReadSourceLocation(ModuleFile, Record, Idx));

      if (Record[Idx++]) {
        AttrVec Attrs;
        Reader.ReadAttributes(F, Attrs, Record, Idx);
        D->setAttrsImpl(Attrs, Reader.getContext());
      }
      break;
    }

    case UPD_CXX_RESOLVED_EXCEPTION_SPEC: {
      auto *FD = cast<FunctionDecl>(D);
      auto *FPT = FD->getType()->castAs<FunctionProtoType>();
      auto EPI = FPT->getExtProtoInfo();
      SmallVector<QualType, 8> ExceptionStorage;
      Reader.readExceptionSpec(ModuleFile, ExceptionStorage, EPI, Record, Idx);
      FD->setType(Reader.Context.getFunctionType(FPT->getReturnType(),
                                                 FPT->getParamTypes(), EPI));
      break;
    }

    case UPD_CXX_DEDUCED_RETURN_TYPE: {
      FunctionDecl *FD = cast<FunctionDecl>(D);
      Reader.Context.adjustDeducedFunctionResultType(
          FD, Reader.readType(ModuleFile, Record, Idx));
      break;
    }

    case UPD_DECL_MARKED_USED: {
      // FIXME: This doesn't send the right notifications if there are
      // ASTMutationListeners other than an ASTWriter.
      D->Used = true;
      break;
    }

    case UPD_MANGLING_NUMBER:
      Reader.Context.setManglingNumber(cast<NamedDecl>(D), Record[Idx++]);
      break;

    case UPD_STATIC_LOCAL_NUMBER:
      Reader.Context.setStaticLocalNumber(cast<VarDecl>(D), Record[Idx++]);
      break;
    }
  }
}
