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

#include "ASTCommon.h"
#include "clang/Serialization/ASTReader.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/DeclGroup.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
using namespace clang;
using namespace clang::serialization;

//===----------------------------------------------------------------------===//
// Declaration deserialization
//===----------------------------------------------------------------------===//

namespace clang {
  class ASTDeclReader : public DeclVisitor<ASTDeclReader, void> {
    ASTReader &Reader;
    ASTReader::PerFileData &F;
    llvm::BitstreamCursor &Cursor;
    const DeclID ThisDeclID;
    typedef ASTReader::RecordData RecordData;
    const RecordData &Record;
    unsigned &Idx;
    TypeID TypeIDForTypeDecl;
    
    DeclID DeclContextIDForTemplateParmDecl;
    DeclID LexicalDeclContextIDForTemplateParmDecl;

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

    void ReadCXXDefinitionData(struct CXXRecordDecl::DefinitionData &Data,
                               const RecordData &R, unsigned &I);

    void InitializeCXXDefinitionData(CXXRecordDecl *D,
                                     CXXRecordDecl *DefinitionDecl,
                                     const RecordData &Record, unsigned &Idx);
  public:
    ASTDeclReader(ASTReader &Reader, ASTReader::PerFileData &F,
                  llvm::BitstreamCursor &Cursor, DeclID thisDeclID,
                  const RecordData &Record, unsigned &Idx)
      : Reader(Reader), F(F), Cursor(Cursor), ThisDeclID(thisDeclID),
        Record(Record), Idx(Idx), TypeIDForTypeDecl(0) { }

    static void attachPreviousDecl(Decl *D, Decl *previous);

    void Visit(Decl *D);

    void UpdateDecl(Decl *D, ASTReader::PerFileData &Module,
                    const RecordData &Record);

    void VisitDecl(Decl *D);
    void VisitTranslationUnitDecl(TranslationUnitDecl *TU);
    void VisitNamedDecl(NamedDecl *ND);
    void VisitLabelDecl(LabelDecl *LD);
    void VisitNamespaceDecl(NamespaceDecl *D);
    void VisitUsingDirectiveDecl(UsingDirectiveDecl *D);
    void VisitNamespaceAliasDecl(NamespaceAliasDecl *D);
    void VisitTypeDecl(TypeDecl *TD);
    void VisitTypedefDecl(TypedefDecl *TD);
    void VisitTypeAliasDecl(TypeAliasDecl *TD);
    void VisitUnresolvedUsingTypenameDecl(UnresolvedUsingTypenameDecl *D);
    void VisitTagDecl(TagDecl *TD);
    void VisitEnumDecl(EnumDecl *ED);
    void VisitRecordDecl(RecordDecl *RD);
    void VisitCXXRecordDecl(CXXRecordDecl *D);
    void VisitClassTemplateSpecializationDecl(
                                            ClassTemplateSpecializationDecl *D);
    void VisitClassTemplatePartialSpecializationDecl(
                                     ClassTemplatePartialSpecializationDecl *D);
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
    void VisitIndirectFieldDecl(IndirectFieldDecl *FD);
    void VisitVarDecl(VarDecl *VD);
    void VisitImplicitParamDecl(ImplicitParamDecl *PD);
    void VisitParmVarDecl(ParmVarDecl *PD);
    void VisitNonTypeTemplateParmDecl(NonTypeTemplateParmDecl *D);
    void VisitTemplateDecl(TemplateDecl *D);
    void VisitRedeclarableTemplateDecl(RedeclarableTemplateDecl *D);
    void VisitClassTemplateDecl(ClassTemplateDecl *D);
    void VisitFunctionTemplateDecl(FunctionTemplateDecl *D);
    void VisitTemplateTemplateParmDecl(TemplateTemplateParmDecl *D);
    void VisitUsingDecl(UsingDecl *D);
    void VisitUsingShadowDecl(UsingShadowDecl *D);
    void VisitLinkageSpecDecl(LinkageSpecDecl *D);
    void VisitFileScopeAsmDecl(FileScopeAsmDecl *AD);
    void VisitAccessSpecDecl(AccessSpecDecl *D);
    void VisitFriendDecl(FriendDecl *D);
    void VisitFriendTemplateDecl(FriendTemplateDecl *D);
    void VisitStaticAssertDecl(StaticAssertDecl *D);
    void VisitBlockDecl(BlockDecl *BD);

    std::pair<uint64_t, uint64_t> VisitDeclContext(DeclContext *DC);
    template <typename T> void VisitRedeclarable(Redeclarable<T> *D);

    // FIXME: Reorder according to DeclNodes.td?
    void VisitObjCMethodDecl(ObjCMethodDecl *D);
    void VisitObjCContainerDecl(ObjCContainerDecl *D);
    void VisitObjCInterfaceDecl(ObjCInterfaceDecl *D);
    void VisitObjCIvarDecl(ObjCIvarDecl *D);
    void VisitObjCProtocolDecl(ObjCProtocolDecl *D);
    void VisitObjCAtDefsFieldDecl(ObjCAtDefsFieldDecl *D);
    void VisitObjCClassDecl(ObjCClassDecl *D);
    void VisitObjCForwardProtocolDecl(ObjCForwardProtocolDecl *D);
    void VisitObjCCategoryDecl(ObjCCategoryDecl *D);
    void VisitObjCImplDecl(ObjCImplDecl *D);
    void VisitObjCCategoryImplDecl(ObjCCategoryImplDecl *D);
    void VisitObjCImplementationDecl(ObjCImplementationDecl *D);
    void VisitObjCCompatibleAliasDecl(ObjCCompatibleAliasDecl *D);
    void VisitObjCPropertyDecl(ObjCPropertyDecl *D);
    void VisitObjCPropertyImplDecl(ObjCPropertyImplDecl *D);
  };
}

uint64_t ASTDeclReader::GetCurrentCursorOffset() {
  uint64_t Off = 0;
  for (unsigned I = 0, N = Reader.Chain.size(); I != N; ++I) {
    ASTReader::PerFileData &F = *Reader.Chain[N - I - 1];
    if (&Cursor == &F.DeclsCursor) {
      Off += F.DeclsCursor.GetCurrentBitNo();
      break;
    }
    Off += F.SizeInBits;
  }
  return Off;
}

void ASTDeclReader::Visit(Decl *D) {
  DeclVisitor<ASTDeclReader, void>::Visit(D);

  if (TypeDecl *TD = dyn_cast<TypeDecl>(D)) {
    // if we have a fully initialized TypeDecl, we can safely read its type now.
    TD->setTypeForDecl(Reader.GetType(TypeIDForTypeDecl).getTypePtrOrNull());
  } else if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    // FunctionDecl's body was written last after all other Stmts/Exprs.
    if (Record[Idx++])
      FD->setLazyBody(GetCurrentCursorOffset());
  } else if (D->isTemplateParameter()) {
    // If we have a fully initialized template parameter, we can now
    // set its DeclContext.
    D->setDeclContext(
          cast_or_null<DeclContext>(
                            Reader.GetDecl(DeclContextIDForTemplateParmDecl)));
    D->setLexicalDeclContext(
          cast_or_null<DeclContext>(
                      Reader.GetDecl(LexicalDeclContextIDForTemplateParmDecl)));
  }
}

void ASTDeclReader::VisitDecl(Decl *D) {
  if (D->isTemplateParameter()) {
    // We don't want to deserialize the DeclContext of a template
    // parameter immediately, because the template parameter might be
    // used in the formulation of its DeclContext. Use the translation
    // unit DeclContext as a placeholder.
    DeclContextIDForTemplateParmDecl = Record[Idx++];
    LexicalDeclContextIDForTemplateParmDecl = Record[Idx++];
    D->setDeclContext(Reader.getContext()->getTranslationUnitDecl()); 
  } else {
    D->setDeclContext(cast_or_null<DeclContext>(Reader.GetDecl(Record[Idx++])));
    D->setLexicalDeclContext(
                     cast_or_null<DeclContext>(Reader.GetDecl(Record[Idx++])));
  }
  D->setLocation(ReadSourceLocation(Record, Idx));
  D->setInvalidDecl(Record[Idx++]);
  if (Record[Idx++]) { // hasAttrs
    AttrVec Attrs;
    Reader.ReadAttributes(F, Attrs, Record, Idx);
    D->setAttrs(Attrs);
  }
  D->setImplicit(Record[Idx++]);
  D->setUsed(Record[Idx++]);
  D->setReferenced(Record[Idx++]);
  D->setAccess((AccessSpecifier)Record[Idx++]);
  D->setPCHLevel(Record[Idx++] + (F.Type <= ASTReader::PCH));
}

void ASTDeclReader::VisitTranslationUnitDecl(TranslationUnitDecl *TU) {
  VisitDecl(TU);
  TU->setAnonymousNamespace(
                    cast_or_null<NamespaceDecl>(Reader.GetDecl(Record[Idx++])));
}

void ASTDeclReader::VisitNamedDecl(NamedDecl *ND) {
  VisitDecl(ND);
  ND->setDeclName(Reader.ReadDeclarationName(Record, Idx));
}

void ASTDeclReader::VisitTypeDecl(TypeDecl *TD) {
  VisitNamedDecl(TD);
  TD->setLocStart(ReadSourceLocation(Record, Idx));
  // Delay type reading until after we have fully initialized the decl.
  TypeIDForTypeDecl = Record[Idx++];
}

void ASTDeclReader::VisitTypedefDecl(TypedefDecl *TD) {
  VisitTypeDecl(TD);
  TD->setTypeSourceInfo(GetTypeSourceInfo(Record, Idx));
}

void ASTDeclReader::VisitTypeAliasDecl(TypeAliasDecl *TD) {
  VisitTypeDecl(TD);
  TD->setTypeSourceInfo(GetTypeSourceInfo(Record, Idx));
}

void ASTDeclReader::VisitTagDecl(TagDecl *TD) {
  VisitTypeDecl(TD);
  VisitRedeclarable(TD);
  TD->IdentifierNamespace = Record[Idx++];
  TD->setTagKind((TagDecl::TagKind)Record[Idx++]);
  TD->setDefinition(Record[Idx++]);
  TD->setEmbeddedInDeclarator(Record[Idx++]);
  TD->setRBraceLoc(ReadSourceLocation(Record, Idx));
  if (Record[Idx++]) { // hasExtInfo
    TagDecl::ExtInfo *Info = new (*Reader.getContext()) TagDecl::ExtInfo();
    ReadQualifierInfo(*Info, Record, Idx);
    TD->TypedefNameDeclOrQualifier = Info;
  } else
    TD->setTypedefNameForAnonDecl(
                  cast_or_null<TypedefNameDecl>(Reader.GetDecl(Record[Idx++])));
}

void ASTDeclReader::VisitEnumDecl(EnumDecl *ED) {
  VisitTagDecl(ED);
  if (TypeSourceInfo *TI = Reader.GetTypeSourceInfo(F, Record, Idx))
    ED->setIntegerTypeSourceInfo(TI);
  else
    ED->setIntegerType(Reader.GetType(Record[Idx++]));
  ED->setPromotionType(Reader.GetType(Record[Idx++]));
  ED->setNumPositiveBits(Record[Idx++]);
  ED->setNumNegativeBits(Record[Idx++]);
  ED->IsScoped = Record[Idx++];
  ED->IsScopedUsingClassTag = Record[Idx++];
  ED->IsFixed = Record[Idx++];
  ED->setInstantiationOfMemberEnum(
                         cast_or_null<EnumDecl>(Reader.GetDecl(Record[Idx++])));
}

void ASTDeclReader::VisitRecordDecl(RecordDecl *RD) {
  VisitTagDecl(RD);
  RD->setHasFlexibleArrayMember(Record[Idx++]);
  RD->setAnonymousStructOrUnion(Record[Idx++]);
  RD->setHasObjectMember(Record[Idx++]);
}

void ASTDeclReader::VisitValueDecl(ValueDecl *VD) {
  VisitNamedDecl(VD);
  VD->setType(Reader.GetType(Record[Idx++]));
}

void ASTDeclReader::VisitEnumConstantDecl(EnumConstantDecl *ECD) {
  VisitValueDecl(ECD);
  if (Record[Idx++])
    ECD->setInitExpr(Reader.ReadExpr(F));
  ECD->setInitVal(Reader.ReadAPSInt(Record, Idx));
}

void ASTDeclReader::VisitDeclaratorDecl(DeclaratorDecl *DD) {
  VisitValueDecl(DD);
  DD->setInnerLocStart(ReadSourceLocation(Record, Idx));
  if (Record[Idx++]) { // hasExtInfo
    DeclaratorDecl::ExtInfo *Info
        = new (*Reader.getContext()) DeclaratorDecl::ExtInfo();
    ReadQualifierInfo(*Info, Record, Idx);
    Info->TInfo = GetTypeSourceInfo(Record, Idx);
    DD->DeclInfo = Info;
  } else
    DD->DeclInfo = GetTypeSourceInfo(Record, Idx);
}

void ASTDeclReader::VisitFunctionDecl(FunctionDecl *FD) {
  VisitDeclaratorDecl(FD);
  VisitRedeclarable(FD);

  ReadDeclarationNameLoc(FD->DNLoc, FD->getDeclName(), Record, Idx);
  FD->IdentifierNamespace = Record[Idx++];
  switch ((FunctionDecl::TemplatedKind)Record[Idx++]) {
  default: assert(false && "Unhandled TemplatedKind!");
    break;
  case FunctionDecl::TK_NonTemplate:
    break;
  case FunctionDecl::TK_FunctionTemplate:
    FD->setDescribedFunctionTemplate(
                     cast<FunctionTemplateDecl>(Reader.GetDecl(Record[Idx++])));
    break;
  case FunctionDecl::TK_MemberSpecialization: {
    FunctionDecl *InstFD = cast<FunctionDecl>(Reader.GetDecl(Record[Idx++]));
    TemplateSpecializationKind TSK = (TemplateSpecializationKind)Record[Idx++];
    SourceLocation POI = ReadSourceLocation(Record, Idx);
    FD->setInstantiationOfMemberFunction(*Reader.getContext(), InstFD, TSK);
    FD->getMemberSpecializationInfo()->setPointOfInstantiation(POI);
    break;
  }
  case FunctionDecl::TK_FunctionTemplateSpecialization: {
    FunctionTemplateDecl *Template
      = cast<FunctionTemplateDecl>(Reader.GetDecl(Record[Idx++]));
    TemplateSpecializationKind TSK = (TemplateSpecializationKind)Record[Idx++];
    
    // Template arguments.
    llvm::SmallVector<TemplateArgument, 8> TemplArgs;
    Reader.ReadTemplateArgumentList(TemplArgs, F, Record, Idx);
    
    // Template args as written.
    llvm::SmallVector<TemplateArgumentLoc, 8> TemplArgLocs;
    SourceLocation LAngleLoc, RAngleLoc;
    if (Record[Idx++]) {  // TemplateArgumentsAsWritten != 0
      unsigned NumTemplateArgLocs = Record[Idx++];
      TemplArgLocs.reserve(NumTemplateArgLocs);
      for (unsigned i=0; i != NumTemplateArgLocs; ++i)
        TemplArgLocs.push_back(
            Reader.ReadTemplateArgumentLoc(F, Record, Idx));
  
      LAngleLoc = ReadSourceLocation(Record, Idx);
      RAngleLoc = ReadSourceLocation(Record, Idx);
    }
    
    SourceLocation POI = ReadSourceLocation(Record, Idx);

    ASTContext &C = *Reader.getContext();
    TemplateArgumentList *TemplArgList
      = TemplateArgumentList::CreateCopy(C, TemplArgs.data(), TemplArgs.size());
    TemplateArgumentListInfo *TemplArgsInfo
      = new (C) TemplateArgumentListInfo(LAngleLoc, RAngleLoc);
    for (unsigned i=0, e = TemplArgLocs.size(); i != e; ++i)
      TemplArgsInfo->addArgument(TemplArgLocs[i]);
    FunctionTemplateSpecializationInfo *FTInfo
        = FunctionTemplateSpecializationInfo::Create(C, FD, Template, TSK,
                                                     TemplArgList,
                                                     TemplArgsInfo, POI);
    FD->TemplateOrSpecialization = FTInfo;

    if (FD->isCanonicalDecl()) { // if canonical add to template's set.
      // The template that contains the specializations set. It's not safe to
      // use getCanonicalDecl on Template since it may still be initializing.
      FunctionTemplateDecl *CanonTemplate
        = cast<FunctionTemplateDecl>(Reader.GetDecl(Record[Idx++]));
      // Get the InsertPos by FindNodeOrInsertPos() instead of calling
      // InsertNode(FTInfo) directly to avoid the getASTContext() call in
      // FunctionTemplateSpecializationInfo's Profile().
      // We avoid getASTContext because a decl in the parent hierarchy may
      // be initializing.
      llvm::FoldingSetNodeID ID;
      FunctionTemplateSpecializationInfo::Profile(ID, TemplArgs.data(),
                                                  TemplArgs.size(), C);
      void *InsertPos = 0;
      CanonTemplate->getSpecializations().FindNodeOrInsertPos(ID, InsertPos);
      assert(InsertPos && "Another specialization already inserted!");
      CanonTemplate->getSpecializations().InsertNode(FTInfo, InsertPos);
    }
    break;
  }
  case FunctionDecl::TK_DependentFunctionTemplateSpecialization: {
    // Templates.
    UnresolvedSet<8> TemplDecls;
    unsigned NumTemplates = Record[Idx++];
    while (NumTemplates--)
      TemplDecls.addDecl(cast<NamedDecl>(Reader.GetDecl(Record[Idx++])));
    
    // Templates args.
    TemplateArgumentListInfo TemplArgs;
    unsigned NumArgs = Record[Idx++];
    while (NumArgs--)
      TemplArgs.addArgument(Reader.ReadTemplateArgumentLoc(F, Record, Idx));
    TemplArgs.setLAngleLoc(ReadSourceLocation(Record, Idx));
    TemplArgs.setRAngleLoc(ReadSourceLocation(Record, Idx));
    
    FD->setDependentTemplateSpecialization(*Reader.getContext(),
                                           TemplDecls, TemplArgs);
    break;
  }
  }

  // FunctionDecl's body is handled last at ASTDeclReader::Visit,
  // after everything else is read.

  FD->SClass = (StorageClass)Record[Idx++];
  FD->SClassAsWritten = (StorageClass)Record[Idx++];
  FD->IsInline = Record[Idx++];
  FD->IsInlineSpecified = Record[Idx++];
  FD->IsVirtualAsWritten = Record[Idx++];
  FD->IsPure = Record[Idx++];
  FD->HasInheritedPrototype = Record[Idx++];
  FD->HasWrittenPrototype = Record[Idx++];
  FD->IsDeleted = Record[Idx++];
  FD->IsTrivial = Record[Idx++];
  FD->HasImplicitReturnZero = Record[Idx++];
  FD->EndRangeLoc = ReadSourceLocation(Record, Idx);

  // Read in the parameters.
  unsigned NumParams = Record[Idx++];
  llvm::SmallVector<ParmVarDecl *, 16> Params;
  Params.reserve(NumParams);
  for (unsigned I = 0; I != NumParams; ++I)
    Params.push_back(cast<ParmVarDecl>(Reader.GetDecl(Record[Idx++])));
  FD->setParams(*Reader.getContext(), Params.data(), NumParams);
}

void ASTDeclReader::VisitObjCMethodDecl(ObjCMethodDecl *MD) {
  VisitNamedDecl(MD);
  if (Record[Idx++]) {
    // In practice, this won't be executed (since method definitions
    // don't occur in header files).
    MD->setBody(Reader.ReadStmt(F));
    MD->setSelfDecl(cast<ImplicitParamDecl>(Reader.GetDecl(Record[Idx++])));
    MD->setCmdDecl(cast<ImplicitParamDecl>(Reader.GetDecl(Record[Idx++])));
  }
  MD->setInstanceMethod(Record[Idx++]);
  MD->setVariadic(Record[Idx++]);
  MD->setSynthesized(Record[Idx++]);
  MD->setDefined(Record[Idx++]);
  MD->setDeclImplementation((ObjCMethodDecl::ImplementationControl)Record[Idx++]);
  MD->setObjCDeclQualifier((Decl::ObjCDeclQualifier)Record[Idx++]);
  MD->setNumSelectorArgs(unsigned(Record[Idx++]));
  MD->setResultType(Reader.GetType(Record[Idx++]));
  MD->setResultTypeSourceInfo(GetTypeSourceInfo(Record, Idx));
  MD->setEndLoc(ReadSourceLocation(Record, Idx));
  unsigned NumParams = Record[Idx++];
  llvm::SmallVector<ParmVarDecl *, 16> Params;
  Params.reserve(NumParams);
  for (unsigned I = 0; I != NumParams; ++I)
    Params.push_back(cast<ParmVarDecl>(Reader.GetDecl(Record[Idx++])));
  MD->setMethodParams(*Reader.getContext(), Params.data(), NumParams,
                      NumParams);
}

void ASTDeclReader::VisitObjCContainerDecl(ObjCContainerDecl *CD) {
  VisitNamedDecl(CD);
  SourceLocation A = ReadSourceLocation(Record, Idx);
  SourceLocation B = ReadSourceLocation(Record, Idx);
  CD->setAtEndRange(SourceRange(A, B));
}

void ASTDeclReader::VisitObjCInterfaceDecl(ObjCInterfaceDecl *ID) {
  VisitObjCContainerDecl(ID);
  ID->setTypeForDecl(Reader.GetType(Record[Idx++]).getTypePtrOrNull());
  ID->setSuperClass(cast_or_null<ObjCInterfaceDecl>
                       (Reader.GetDecl(Record[Idx++])));
  
  // Read the directly referenced protocols and their SourceLocations.
  unsigned NumProtocols = Record[Idx++];
  llvm::SmallVector<ObjCProtocolDecl *, 16> Protocols;
  Protocols.reserve(NumProtocols);
  for (unsigned I = 0; I != NumProtocols; ++I)
    Protocols.push_back(cast<ObjCProtocolDecl>(Reader.GetDecl(Record[Idx++])));
  llvm::SmallVector<SourceLocation, 16> ProtoLocs;
  ProtoLocs.reserve(NumProtocols);
  for (unsigned I = 0; I != NumProtocols; ++I)
    ProtoLocs.push_back(ReadSourceLocation(Record, Idx));
  ID->setProtocolList(Protocols.data(), NumProtocols, ProtoLocs.data(),
                      *Reader.getContext());
  
  // Read the transitive closure of protocols referenced by this class.
  NumProtocols = Record[Idx++];
  Protocols.clear();
  Protocols.reserve(NumProtocols);
  for (unsigned I = 0; I != NumProtocols; ++I)
    Protocols.push_back(cast<ObjCProtocolDecl>(Reader.GetDecl(Record[Idx++])));
  ID->AllReferencedProtocols.set(Protocols.data(), NumProtocols,
                                 *Reader.getContext());
  
  // Read the ivars.
  unsigned NumIvars = Record[Idx++];
  llvm::SmallVector<ObjCIvarDecl *, 16> IVars;
  IVars.reserve(NumIvars);
  for (unsigned I = 0; I != NumIvars; ++I)
    IVars.push_back(cast<ObjCIvarDecl>(Reader.GetDecl(Record[Idx++])));
  ID->setCategoryList(
               cast_or_null<ObjCCategoryDecl>(Reader.GetDecl(Record[Idx++])));
  // We will rebuild this list lazily.
  ID->setIvarList(0);
  ID->setForwardDecl(Record[Idx++]);
  ID->setImplicitInterfaceDecl(Record[Idx++]);
  ID->setClassLoc(ReadSourceLocation(Record, Idx));
  ID->setSuperClassLoc(ReadSourceLocation(Record, Idx));
  ID->setLocEnd(ReadSourceLocation(Record, Idx));
}

void ASTDeclReader::VisitObjCIvarDecl(ObjCIvarDecl *IVD) {
  VisitFieldDecl(IVD);
  IVD->setAccessControl((ObjCIvarDecl::AccessControl)Record[Idx++]);
  // This field will be built lazily.
  IVD->setNextIvar(0);
  bool synth = Record[Idx++];
  IVD->setSynthesize(synth);
}

void ASTDeclReader::VisitObjCProtocolDecl(ObjCProtocolDecl *PD) {
  VisitObjCContainerDecl(PD);
  PD->setForwardDecl(Record[Idx++]);
  PD->setLocEnd(ReadSourceLocation(Record, Idx));
  unsigned NumProtoRefs = Record[Idx++];
  llvm::SmallVector<ObjCProtocolDecl *, 16> ProtoRefs;
  ProtoRefs.reserve(NumProtoRefs);
  for (unsigned I = 0; I != NumProtoRefs; ++I)
    ProtoRefs.push_back(cast<ObjCProtocolDecl>(Reader.GetDecl(Record[Idx++])));
  llvm::SmallVector<SourceLocation, 16> ProtoLocs;
  ProtoLocs.reserve(NumProtoRefs);
  for (unsigned I = 0; I != NumProtoRefs; ++I)
    ProtoLocs.push_back(ReadSourceLocation(Record, Idx));
  PD->setProtocolList(ProtoRefs.data(), NumProtoRefs, ProtoLocs.data(),
                      *Reader.getContext());
}

void ASTDeclReader::VisitObjCAtDefsFieldDecl(ObjCAtDefsFieldDecl *FD) {
  VisitFieldDecl(FD);
}

void ASTDeclReader::VisitObjCClassDecl(ObjCClassDecl *CD) {
  VisitDecl(CD);
  unsigned NumClassRefs = Record[Idx++];
  llvm::SmallVector<ObjCInterfaceDecl *, 16> ClassRefs;
  ClassRefs.reserve(NumClassRefs);
  for (unsigned I = 0; I != NumClassRefs; ++I)
    ClassRefs.push_back(cast<ObjCInterfaceDecl>(Reader.GetDecl(Record[Idx++])));
  llvm::SmallVector<SourceLocation, 16> SLocs;
  SLocs.reserve(NumClassRefs);
  for (unsigned I = 0; I != NumClassRefs; ++I)
    SLocs.push_back(ReadSourceLocation(Record, Idx));
  CD->setClassList(*Reader.getContext(), ClassRefs.data(), SLocs.data(),
                   NumClassRefs);
}

void ASTDeclReader::VisitObjCForwardProtocolDecl(ObjCForwardProtocolDecl *FPD) {
  VisitDecl(FPD);
  unsigned NumProtoRefs = Record[Idx++];
  llvm::SmallVector<ObjCProtocolDecl *, 16> ProtoRefs;
  ProtoRefs.reserve(NumProtoRefs);
  for (unsigned I = 0; I != NumProtoRefs; ++I)
    ProtoRefs.push_back(cast<ObjCProtocolDecl>(Reader.GetDecl(Record[Idx++])));
  llvm::SmallVector<SourceLocation, 16> ProtoLocs;
  ProtoLocs.reserve(NumProtoRefs);
  for (unsigned I = 0; I != NumProtoRefs; ++I)
    ProtoLocs.push_back(ReadSourceLocation(Record, Idx));
  FPD->setProtocolList(ProtoRefs.data(), NumProtoRefs, ProtoLocs.data(),
                       *Reader.getContext());
}

void ASTDeclReader::VisitObjCCategoryDecl(ObjCCategoryDecl *CD) {
  VisitObjCContainerDecl(CD);
  CD->setClassInterface(cast<ObjCInterfaceDecl>(Reader.GetDecl(Record[Idx++])));
  unsigned NumProtoRefs = Record[Idx++];
  llvm::SmallVector<ObjCProtocolDecl *, 16> ProtoRefs;
  ProtoRefs.reserve(NumProtoRefs);
  for (unsigned I = 0; I != NumProtoRefs; ++I)
    ProtoRefs.push_back(cast<ObjCProtocolDecl>(Reader.GetDecl(Record[Idx++])));
  llvm::SmallVector<SourceLocation, 16> ProtoLocs;
  ProtoLocs.reserve(NumProtoRefs);
  for (unsigned I = 0; I != NumProtoRefs; ++I)
    ProtoLocs.push_back(ReadSourceLocation(Record, Idx));
  CD->setProtocolList(ProtoRefs.data(), NumProtoRefs, ProtoLocs.data(),
                      *Reader.getContext());
  CD->setNextClassCategory(cast_or_null<ObjCCategoryDecl>(Reader.GetDecl(Record[Idx++])));
  CD->setHasSynthBitfield(Record[Idx++]);
  CD->setAtLoc(ReadSourceLocation(Record, Idx));
  CD->setCategoryNameLoc(ReadSourceLocation(Record, Idx));
}

void ASTDeclReader::VisitObjCCompatibleAliasDecl(ObjCCompatibleAliasDecl *CAD) {
  VisitNamedDecl(CAD);
  CAD->setClassInterface(cast<ObjCInterfaceDecl>(Reader.GetDecl(Record[Idx++])));
}

void ASTDeclReader::VisitObjCPropertyDecl(ObjCPropertyDecl *D) {
  VisitNamedDecl(D);
  D->setAtLoc(ReadSourceLocation(Record, Idx));
  D->setType(GetTypeSourceInfo(Record, Idx));
  // FIXME: stable encoding
  D->setPropertyAttributes(
                      (ObjCPropertyDecl::PropertyAttributeKind)Record[Idx++]);
  D->setPropertyAttributesAsWritten(
                      (ObjCPropertyDecl::PropertyAttributeKind)Record[Idx++]);
  // FIXME: stable encoding
  D->setPropertyImplementation(
                            (ObjCPropertyDecl::PropertyControl)Record[Idx++]);
  D->setGetterName(Reader.ReadDeclarationName(Record, Idx).getObjCSelector());
  D->setSetterName(Reader.ReadDeclarationName(Record, Idx).getObjCSelector());
  D->setGetterMethodDecl(
                 cast_or_null<ObjCMethodDecl>(Reader.GetDecl(Record[Idx++])));
  D->setSetterMethodDecl(
                 cast_or_null<ObjCMethodDecl>(Reader.GetDecl(Record[Idx++])));
  D->setPropertyIvarDecl(
                   cast_or_null<ObjCIvarDecl>(Reader.GetDecl(Record[Idx++])));
}

void ASTDeclReader::VisitObjCImplDecl(ObjCImplDecl *D) {
  VisitObjCContainerDecl(D);
  D->setClassInterface(
              cast_or_null<ObjCInterfaceDecl>(Reader.GetDecl(Record[Idx++])));
}

void ASTDeclReader::VisitObjCCategoryImplDecl(ObjCCategoryImplDecl *D) {
  VisitObjCImplDecl(D);
  D->setIdentifier(Reader.GetIdentifierInfo(Record, Idx));
}

void ASTDeclReader::VisitObjCImplementationDecl(ObjCImplementationDecl *D) {
  VisitObjCImplDecl(D);
  D->setSuperClass(
              cast_or_null<ObjCInterfaceDecl>(Reader.GetDecl(Record[Idx++])));
  llvm::tie(D->IvarInitializers, D->NumIvarInitializers)
      = Reader.ReadCXXCtorInitializers(F, Record, Idx);
  D->setHasSynthBitfield(Record[Idx++]);
}


void ASTDeclReader::VisitObjCPropertyImplDecl(ObjCPropertyImplDecl *D) {
  VisitDecl(D);
  D->setAtLoc(ReadSourceLocation(Record, Idx));
  D->setPropertyDecl(
               cast_or_null<ObjCPropertyDecl>(Reader.GetDecl(Record[Idx++])));
  D->PropertyIvarDecl = 
                   cast_or_null<ObjCIvarDecl>(Reader.GetDecl(Record[Idx++]));
  D->IvarLoc = ReadSourceLocation(Record, Idx);
  D->setGetterCXXConstructor(Reader.ReadExpr(F));
  D->setSetterCXXAssignment(Reader.ReadExpr(F));
}

void ASTDeclReader::VisitFieldDecl(FieldDecl *FD) {
  VisitDeclaratorDecl(FD);
  FD->setMutable(Record[Idx++]);
  if (Record[Idx++])
    FD->setBitWidth(Reader.ReadExpr(F));
  if (!FD->getDeclName()) {
    FieldDecl *Tmpl = cast_or_null<FieldDecl>(Reader.GetDecl(Record[Idx++]));
    if (Tmpl)
      Reader.getContext()->setInstantiatedFromUnnamedFieldDecl(FD, Tmpl);
  }
}

void ASTDeclReader::VisitIndirectFieldDecl(IndirectFieldDecl *FD) {
  VisitValueDecl(FD);

  FD->ChainingSize = Record[Idx++];
  assert(FD->ChainingSize >= 2 && "Anonymous chaining must be >= 2");
  FD->Chaining = new (*Reader.getContext())NamedDecl*[FD->ChainingSize];

  for (unsigned I = 0; I != FD->ChainingSize; ++I)
    FD->Chaining[I] = cast<NamedDecl>(Reader.GetDecl(Record[Idx++]));
}

void ASTDeclReader::VisitVarDecl(VarDecl *VD) {
  VisitDeclaratorDecl(VD);
  VisitRedeclarable(VD);
  VD->SClass = (StorageClass)Record[Idx++];
  VD->setStorageClassAsWritten((StorageClass)Record[Idx++]);
  VD->setThreadSpecified(Record[Idx++]);
  VD->setCXXDirectInitializer(Record[Idx++]);
  VD->setExceptionVariable(Record[Idx++]);
  VD->setNRVOVariable(Record[Idx++]);
  VD->setCXXForRangeDecl(Record[Idx++]);
  if (Record[Idx++])
    VD->setInit(Reader.ReadExpr(F));

  if (Record[Idx++]) { // HasMemberSpecializationInfo.
    VarDecl *Tmpl = cast<VarDecl>(Reader.GetDecl(Record[Idx++]));
    TemplateSpecializationKind TSK = (TemplateSpecializationKind)Record[Idx++];
    SourceLocation POI = ReadSourceLocation(Record, Idx);
    Reader.getContext()->setInstantiatedFromStaticDataMember(VD, Tmpl, TSK,POI);
  }
}

void ASTDeclReader::VisitImplicitParamDecl(ImplicitParamDecl *PD) {
  VisitVarDecl(PD);
}

void ASTDeclReader::VisitParmVarDecl(ParmVarDecl *PD) {
  VisitVarDecl(PD);
  PD->setObjCDeclQualifier((Decl::ObjCDeclQualifier)Record[Idx++]);
  PD->setKNRPromoted(Record[Idx++]);
  PD->setHasInheritedDefaultArg(Record[Idx++]);
  if (Record[Idx++]) // hasUninstantiatedDefaultArg.
    PD->setUninstantiatedDefaultArg(Reader.ReadExpr(F));
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
  llvm::SmallVector<ParmVarDecl *, 16> Params;
  Params.reserve(NumParams);
  for (unsigned I = 0; I != NumParams; ++I)
    Params.push_back(cast<ParmVarDecl>(Reader.GetDecl(Record[Idx++])));
  BD->setParams(Params.data(), NumParams);

  bool capturesCXXThis = Record[Idx++];
  unsigned numCaptures = Record[Idx++];
  llvm::SmallVector<BlockDecl::Capture, 16> captures;
  captures.reserve(numCaptures);
  for (unsigned i = 0; i != numCaptures; ++i) {
    VarDecl *decl = cast<VarDecl>(Reader.GetDecl(Record[Idx++]));
    unsigned flags = Record[Idx++];
    bool byRef = (flags & 1);
    bool nested = (flags & 2);
    Expr *copyExpr = ((flags & 4) ? Reader.ReadExpr(F) : 0);

    captures.push_back(BlockDecl::Capture(decl, byRef, nested, copyExpr));
  }
  BD->setCaptures(*Reader.getContext(), captures.begin(),
                  captures.end(), capturesCXXThis);
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
  VisitNamedDecl(D);
  D->IsInline = Record[Idx++];
  D->LocStart = ReadSourceLocation(Record, Idx);
  D->RBraceLoc = ReadSourceLocation(Record, Idx);
  D->NextNamespace = Record[Idx++];

  bool IsOriginal = Record[Idx++];
  D->OrigOrAnonNamespace.setInt(IsOriginal);
  D->OrigOrAnonNamespace.setPointer(
                    cast_or_null<NamespaceDecl>(Reader.GetDecl(Record[Idx++])));
}

void ASTDeclReader::VisitNamespaceAliasDecl(NamespaceAliasDecl *D) {
  VisitNamedDecl(D);
  D->NamespaceLoc = ReadSourceLocation(Record, Idx);
  D->IdentLoc = ReadSourceLocation(Record, Idx);
  D->QualifierLoc = Reader.ReadNestedNameSpecifierLoc(F, Record, Idx);
  D->Namespace = cast<NamedDecl>(Reader.GetDecl(Record[Idx++]));
}

void ASTDeclReader::VisitUsingDecl(UsingDecl *D) {
  VisitNamedDecl(D);
  D->setUsingLocation(ReadSourceLocation(Record, Idx));
  D->QualifierLoc = Reader.ReadNestedNameSpecifierLoc(F, Record, Idx);
  ReadDeclarationNameLoc(D->DNLoc, D->getDeclName(), Record, Idx);
  D->FirstUsingShadow = cast_or_null<UsingShadowDecl>(Reader.GetDecl(Record[Idx++]));
  D->setTypeName(Record[Idx++]);
  NamedDecl *Pattern = cast_or_null<NamedDecl>(Reader.GetDecl(Record[Idx++]));
  if (Pattern)
    Reader.getContext()->setInstantiatedFromUsingDecl(D, Pattern);
}

void ASTDeclReader::VisitUsingShadowDecl(UsingShadowDecl *D) {
  VisitNamedDecl(D);
  D->setTargetDecl(cast<NamedDecl>(Reader.GetDecl(Record[Idx++])));
  D->UsingOrNextShadow = cast_or_null<NamedDecl>(Reader.GetDecl(Record[Idx++]));
  UsingShadowDecl *Pattern
      = cast_or_null<UsingShadowDecl>(Reader.GetDecl(Record[Idx++]));
  if (Pattern)
    Reader.getContext()->setInstantiatedFromUsingShadowDecl(D, Pattern);
}

void ASTDeclReader::VisitUsingDirectiveDecl(UsingDirectiveDecl *D) {
  VisitNamedDecl(D);
  D->UsingLoc = ReadSourceLocation(Record, Idx);
  D->NamespaceLoc = ReadSourceLocation(Record, Idx);
  D->QualifierLoc = Reader.ReadNestedNameSpecifierLoc(F, Record, Idx);
  D->NominatedNamespace = cast<NamedDecl>(Reader.GetDecl(Record[Idx++]));
  D->CommonAncestor = cast_or_null<DeclContext>(Reader.GetDecl(Record[Idx++]));
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
  Data.UserDeclaredConstructor = Record[Idx++];
  Data.UserDeclaredCopyConstructor = Record[Idx++];
  Data.UserDeclaredCopyAssignment = Record[Idx++];
  Data.UserDeclaredDestructor = Record[Idx++];
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
  Data.HasTrivialConstructor = Record[Idx++];
  Data.HasConstExprNonCopyMoveConstructor = Record[Idx++];
  Data.HasTrivialCopyConstructor = Record[Idx++];
  Data.HasTrivialMoveConstructor = Record[Idx++];
  Data.HasTrivialCopyAssignment = Record[Idx++];
  Data.HasTrivialMoveAssignment = Record[Idx++];
  Data.HasTrivialDestructor = Record[Idx++];
  Data.HasNonLiteralTypeFieldsOrBases = Record[Idx++];
  Data.ComputedVisibleConversions = Record[Idx++];
  Data.DeclaredDefaultConstructor = Record[Idx++];
  Data.DeclaredCopyConstructor = Record[Idx++];
  Data.DeclaredCopyAssignment = Record[Idx++];
  Data.DeclaredDestructor = Record[Idx++];

  Data.NumBases = Record[Idx++];
  if (Data.NumBases)
    Data.Bases = Reader.GetCXXBaseSpecifiersOffset(Record[Idx++]);
  Data.NumVBases = Record[Idx++];
  if (Data.NumVBases)
    Data.VBases = Reader.GetCXXBaseSpecifiersOffset(Record[Idx++]);
  
  Reader.ReadUnresolvedSet(Data.Conversions, Record, Idx);
  Reader.ReadUnresolvedSet(Data.VisibleConversions, Record, Idx);
  assert(Data.Definition && "Data.Definition should be already set!");
  Data.FirstFriend
      = cast_or_null<FriendDecl>(Reader.GetDecl(Record[Idx++]));
}

void ASTDeclReader::InitializeCXXDefinitionData(CXXRecordDecl *D,
                                                CXXRecordDecl *DefinitionDecl,
                                                const RecordData &Record,
                                                unsigned &Idx) {
  ASTContext &C = *Reader.getContext();

  if (D == DefinitionDecl) {
    D->DefinitionData = new (C) struct CXXRecordDecl::DefinitionData(D);
    ReadCXXDefinitionData(*D->DefinitionData, Record, Idx);
    // We read the definition info. Check if there are pending forward
    // references that need to point to this DefinitionData pointer.
    ASTReader::PendingForwardRefsMap::iterator
        FindI = Reader.PendingForwardRefs.find(D);
    if (FindI != Reader.PendingForwardRefs.end()) {
      ASTReader::ForwardRefs &Refs = FindI->second;
      for (ASTReader::ForwardRefs::iterator
             I = Refs.begin(), E = Refs.end(); I != E; ++I)
        (*I)->DefinitionData = D->DefinitionData;
#ifndef NDEBUG
      // We later check whether PendingForwardRefs is empty to make sure all
      // pending references were linked.
      Reader.PendingForwardRefs.erase(D);
#endif
    }
  } else if (DefinitionDecl) {
    if (DefinitionDecl->DefinitionData) {
      D->DefinitionData = DefinitionDecl->DefinitionData;
    } else {
      // The definition is still initializing.
      Reader.PendingForwardRefs[DefinitionDecl].push_back(D);
    }
  }
}

void ASTDeclReader::VisitCXXRecordDecl(CXXRecordDecl *D) {
  VisitRecordDecl(D);

  CXXRecordDecl *DefinitionDecl
      = cast_or_null<CXXRecordDecl>(Reader.GetDecl(Record[Idx++]));
  InitializeCXXDefinitionData(D, DefinitionDecl, Record, Idx);

  ASTContext &C = *Reader.getContext();

  enum CXXRecKind {
    CXXRecNotTemplate = 0, CXXRecTemplate, CXXRecMemberSpecialization
  };
  switch ((CXXRecKind)Record[Idx++]) {
  default:
    assert(false && "Out of sync with ASTDeclWriter::VisitCXXRecordDecl?");
  case CXXRecNotTemplate:
    break;
  case CXXRecTemplate:
    D->TemplateOrInstantiation
        = cast<ClassTemplateDecl>(Reader.GetDecl(Record[Idx++]));
    break;
  case CXXRecMemberSpecialization: {
    CXXRecordDecl *RD = cast<CXXRecordDecl>(Reader.GetDecl(Record[Idx++]));
    TemplateSpecializationKind TSK = (TemplateSpecializationKind)Record[Idx++];
    SourceLocation POI = ReadSourceLocation(Record, Idx);
    MemberSpecializationInfo *MSI = new (C) MemberSpecializationInfo(RD, TSK);
    MSI->setPointOfInstantiation(POI);
    D->TemplateOrInstantiation = MSI;
    break;
  }
  }

  // Load the key function to avoid deserializing every method so we can
  // compute it.
  if (D->IsDefinition) {
    CXXMethodDecl *Key
        = cast_or_null<CXXMethodDecl>(Reader.GetDecl(Record[Idx++]));
    if (Key)
      C.KeyFunctions[D] = Key;
  }
}

void ASTDeclReader::VisitCXXMethodDecl(CXXMethodDecl *D) {
  VisitFunctionDecl(D);
  unsigned NumOverridenMethods = Record[Idx++];
  while (NumOverridenMethods--) {
    CXXMethodDecl *MD = cast<CXXMethodDecl>(Reader.GetDecl(Record[Idx++]));
    // Avoid invariant checking of CXXMethodDecl::addOverriddenMethod,
    // MD may be initializing.
    Reader.getContext()->addOverriddenMethod(D, MD);
  }
}

void ASTDeclReader::VisitCXXConstructorDecl(CXXConstructorDecl *D) {
  VisitCXXMethodDecl(D);
  
  D->IsExplicitSpecified = Record[Idx++];
  D->ImplicitlyDefined = Record[Idx++];
  llvm::tie(D->CtorInitializers, D->NumCtorInitializers)
      = Reader.ReadCXXCtorInitializers(F, Record, Idx);
}

void ASTDeclReader::VisitCXXDestructorDecl(CXXDestructorDecl *D) {
  VisitCXXMethodDecl(D);

  D->ImplicitlyDefined = Record[Idx++];
  D->OperatorDelete = cast_or_null<FunctionDecl>(Reader.GetDecl(Record[Idx++]));
}

void ASTDeclReader::VisitCXXConversionDecl(CXXConversionDecl *D) {
  VisitCXXMethodDecl(D);
  D->IsExplicitSpecified = Record[Idx++];
}

void ASTDeclReader::VisitAccessSpecDecl(AccessSpecDecl *D) {
  VisitDecl(D);
  D->setColonLoc(ReadSourceLocation(Record, Idx));
}

void ASTDeclReader::VisitFriendDecl(FriendDecl *D) {
  VisitDecl(D);
  if (Record[Idx++])
    D->Friend = GetTypeSourceInfo(Record, Idx);
  else
    D->Friend = cast<NamedDecl>(Reader.GetDecl(Record[Idx++]));
  D->NextFriend = Record[Idx++];
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
    D->Friend = cast<NamedDecl>(Reader.GetDecl(Record[Idx++]));
  else
    D->Friend = GetTypeSourceInfo(Record, Idx);
  D->FriendLoc = ReadSourceLocation(Record, Idx);
}

void ASTDeclReader::VisitTemplateDecl(TemplateDecl *D) {
  VisitNamedDecl(D);

  NamedDecl *TemplatedDecl
    = cast_or_null<NamedDecl>(Reader.GetDecl(Record[Idx++]));
  TemplateParameterList* TemplateParams
      = Reader.ReadTemplateParameterList(F, Record, Idx); 
  D->init(TemplatedDecl, TemplateParams);
}

void ASTDeclReader::VisitRedeclarableTemplateDecl(RedeclarableTemplateDecl *D) {
  // Initialize CommonOrPrev before VisitTemplateDecl so that getCommonPtr()
  // can be used while this is still initializing.

  assert(D->CommonOrPrev.isNull() && "getCommonPtr was called earlier on this");
  DeclID PreviousDeclID = Record[Idx++];
  DeclID FirstDeclID =  PreviousDeclID ? Record[Idx++] : 0;
  // We delay loading of the redeclaration chain to avoid deeply nested calls.
  // We temporarily set the first (canonical) declaration as the previous one
  // which is the one that matters and mark the real previous DeclID to be
  // loaded & attached later on.
  RedeclarableTemplateDecl *FirstDecl =
      cast_or_null<RedeclarableTemplateDecl>(Reader.GetDecl(FirstDeclID));
  assert((FirstDecl == 0 || FirstDecl->getKind() == D->getKind()) &&
         "FirstDecl kind mismatch");
  if (FirstDecl) {
    D->CommonOrPrev = FirstDecl;
    // Mark the real previous DeclID to be loaded & attached later on.
    if (PreviousDeclID != FirstDeclID)
      Reader.PendingPreviousDecls.push_back(std::make_pair(D, PreviousDeclID));
  } else {
    D->CommonOrPrev = D->newCommon(*Reader.getContext());
    if (RedeclarableTemplateDecl *RTD
          = cast_or_null<RedeclarableTemplateDecl>(Reader.GetDecl(Record[Idx++]))) {
      assert(RTD->getKind() == D->getKind() &&
             "InstantiatedFromMemberTemplate kind mismatch");
      D->setInstantiatedFromMemberTemplateImpl(RTD);
      if (Record[Idx++])
        D->setMemberSpecialization();
    }

    RedeclarableTemplateDecl *LatestDecl = 
        cast_or_null<RedeclarableTemplateDecl>(Reader.GetDecl(Record[Idx++]));
  
    // This decl is a first one and the latest declaration that it points to is
    // in the same AST file. However, if this actually needs to point to a
    // redeclaration in another AST file, we need to update it by checking
    // the FirstLatestDeclIDs map which tracks this kind of decls.
    assert(Reader.GetDecl(ThisDeclID) == D && "Invalid ThisDeclID ?");
    ASTReader::FirstLatestDeclIDMap::iterator I
        = Reader.FirstLatestDeclIDs.find(ThisDeclID);
    if (I != Reader.FirstLatestDeclIDs.end()) {
      Decl *NewLatest = Reader.GetDecl(I->second);
      assert((LatestDecl->getLocation().isInvalid() ||
              NewLatest->getLocation().isInvalid()  ||
              !Reader.SourceMgr.isBeforeInTranslationUnit(
                                                  NewLatest->getLocation(),
                                                  LatestDecl->getLocation())) &&
             "The new latest is supposed to come after the previous latest");
      LatestDecl = cast<RedeclarableTemplateDecl>(NewLatest);
    }

    assert(LatestDecl->getKind() == D->getKind() && "Latest kind mismatch");
    D->getCommonPtr()->Latest = LatestDecl;
  }

  VisitTemplateDecl(D);
  D->IdentifierNamespace = Record[Idx++];
}

void ASTDeclReader::VisitClassTemplateDecl(ClassTemplateDecl *D) {
  VisitRedeclarableTemplateDecl(D);

  if (D->getPreviousDeclaration() == 0) {
    // This ClassTemplateDecl owns a CommonPtr; read it to keep track of all of
    // the specializations.
    llvm::SmallVector<serialization::DeclID, 2> SpecIDs;
    SpecIDs.push_back(0);
    
    // Specializations.
    unsigned Size = Record[Idx++];
    SpecIDs[0] += Size;
    SpecIDs.append(Record.begin() + Idx, Record.begin() + Idx + Size);
    Idx += Size;

    // Partial specializations.
    Size = Record[Idx++];
    SpecIDs[0] += Size;
    SpecIDs.append(Record.begin() + Idx, Record.begin() + Idx + Size);
    Idx += Size;

    if (SpecIDs[0]) {
      typedef serialization::DeclID DeclID;
      
      ClassTemplateDecl::Common *CommonPtr = D->getCommonPtr();
      CommonPtr->LazySpecializations
        = new (*Reader.getContext()) DeclID [SpecIDs.size()];
      memcpy(CommonPtr->LazySpecializations, SpecIDs.data(), 
             SpecIDs.size() * sizeof(DeclID));
    }
    
    // InjectedClassNameType is computed.
  }
}

void ASTDeclReader::VisitClassTemplateSpecializationDecl(
                                           ClassTemplateSpecializationDecl *D) {
  VisitCXXRecordDecl(D);
  
  ASTContext &C = *Reader.getContext();
  if (Decl *InstD = Reader.GetDecl(Record[Idx++])) {
    if (ClassTemplateDecl *CTD = dyn_cast<ClassTemplateDecl>(InstD)) {
      D->SpecializedTemplate = CTD;
    } else {
      llvm::SmallVector<TemplateArgument, 8> TemplArgs;
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

  // Explicit info.
  if (TypeSourceInfo *TyInfo = GetTypeSourceInfo(Record, Idx)) {
    ClassTemplateSpecializationDecl::ExplicitSpecializationInfo *ExplicitInfo
        = new (C) ClassTemplateSpecializationDecl::ExplicitSpecializationInfo;
    ExplicitInfo->TypeAsWritten = TyInfo;
    ExplicitInfo->ExternLoc = ReadSourceLocation(Record, Idx);
    ExplicitInfo->TemplateKeywordLoc = ReadSourceLocation(Record, Idx);
    D->ExplicitInfo = ExplicitInfo;
  }

  llvm::SmallVector<TemplateArgument, 8> TemplArgs;
  Reader.ReadTemplateArgumentList(TemplArgs, F, Record, Idx);
  D->TemplateArgs = TemplateArgumentList::CreateCopy(C, TemplArgs.data(), 
                                                     TemplArgs.size());
  D->PointOfInstantiation = ReadSourceLocation(Record, Idx);
  D->SpecializationKind = (TemplateSpecializationKind)Record[Idx++];
  
  if (D->isCanonicalDecl()) { // It's kept in the folding set.
    ClassTemplateDecl *CanonPattern
                       = cast<ClassTemplateDecl>(Reader.GetDecl(Record[Idx++]));
    if (ClassTemplatePartialSpecializationDecl *Partial
                       = dyn_cast<ClassTemplatePartialSpecializationDecl>(D)) {
      CanonPattern->getCommonPtr()->PartialSpecializations.InsertNode(Partial);
    } else {
      CanonPattern->getCommonPtr()->Specializations.InsertNode(D);
    }
  }
}

void ASTDeclReader::VisitClassTemplatePartialSpecializationDecl(
                                    ClassTemplatePartialSpecializationDecl *D) {
  VisitClassTemplateSpecializationDecl(D);

  ASTContext &C = *Reader.getContext();
  D->TemplateParams = Reader.ReadTemplateParameterList(F, Record, Idx);

  unsigned NumArgs = Record[Idx++];
  if (NumArgs) {
    D->NumArgsAsWritten = NumArgs;
    D->ArgsAsWritten = new (C) TemplateArgumentLoc[NumArgs];
    for (unsigned i=0; i != NumArgs; ++i)
      D->ArgsAsWritten[i] = Reader.ReadTemplateArgumentLoc(F, Record, Idx);
  }

  D->SequenceNumber = Record[Idx++];

  // These are read/set from/to the first declaration.
  if (D->getPreviousDeclaration() == 0) {
    D->InstantiatedFromMember.setPointer(
        cast_or_null<ClassTemplatePartialSpecializationDecl>(
                                                Reader.GetDecl(Record[Idx++])));
    D->InstantiatedFromMember.setInt(Record[Idx++]);
  }
}

void ASTDeclReader::VisitFunctionTemplateDecl(FunctionTemplateDecl *D) {
  VisitRedeclarableTemplateDecl(D);

  if (D->getPreviousDeclaration() == 0) {
    // This FunctionTemplateDecl owns a CommonPtr; read it.

    // Read the function specialization declarations.
    // FunctionTemplateDecl's FunctionTemplateSpecializationInfos are filled
    // when reading the specialized FunctionDecl.
    unsigned NumSpecs = Record[Idx++];
    while (NumSpecs--)
      Reader.GetDecl(Record[Idx++]);
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
      Data[2*I] = Reader.GetType(Record[Idx++]).getAsOpaquePtr();
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
  // Rest of TemplateTemplateParmDecl.
  TemplateArgumentLoc Arg = Reader.ReadTemplateArgumentLoc(F, Record, Idx);
  bool IsInherited = Record[Idx++];
  D->setDefaultArgument(Arg, IsInherited);
  D->ParameterPack = Record[Idx++];
}

void ASTDeclReader::VisitStaticAssertDecl(StaticAssertDecl *D) {
  VisitDecl(D);
  D->AssertExpr = Reader.ReadExpr(F);
  D->Message = cast<StringLiteral>(Reader.ReadExpr(F));
  D->RParenLoc = ReadSourceLocation(Record, Idx);
}

std::pair<uint64_t, uint64_t>
ASTDeclReader::VisitDeclContext(DeclContext *DC) {
  uint64_t LexicalOffset = Record[Idx++];
  uint64_t VisibleOffset = Record[Idx++];
  return std::make_pair(LexicalOffset, VisibleOffset);
}

template <typename T>
void ASTDeclReader::VisitRedeclarable(Redeclarable<T> *D) {
  enum RedeclKind { NoRedeclaration = 0, PointsToPrevious, PointsToLatest };
  RedeclKind Kind = (RedeclKind)Record[Idx++];
  switch (Kind) {
  default:
    assert(0 && "Out of sync with ASTDeclWriter::VisitRedeclarable or messed up"
                " reading");
  case NoRedeclaration:
    break;
  case PointsToPrevious: {
    DeclID PreviousDeclID = Record[Idx++];
    DeclID FirstDeclID = Record[Idx++];
    // We delay loading of the redeclaration chain to avoid deeply nested calls.
    // We temporarily set the first (canonical) declaration as the previous one
    // which is the one that matters and mark the real previous DeclID to be
    // loaded & attached later on.
    D->RedeclLink = typename Redeclarable<T>::PreviousDeclLink(
                                cast_or_null<T>(Reader.GetDecl(FirstDeclID)));
    if (PreviousDeclID != FirstDeclID)
      Reader.PendingPreviousDecls.push_back(std::make_pair(static_cast<T*>(D),
                                                           PreviousDeclID));
    break;
  }
  case PointsToLatest:
    D->RedeclLink = typename Redeclarable<T>::LatestDeclLink(
                                cast_or_null<T>(Reader.GetDecl(Record[Idx++])));
    break;
  }

  assert(!(Kind == PointsToPrevious &&
           Reader.FirstLatestDeclIDs.find(ThisDeclID) !=
               Reader.FirstLatestDeclIDs.end()) &&
         "This decl is not first, it should not be in the map");
  if (Kind == PointsToPrevious)
    return;

  // This decl is a first one and the latest declaration that it points to is in
  // the same AST file. However, if this actually needs to point to a
  // redeclaration in another AST file, we need to update it by checking the
  // FirstLatestDeclIDs map which tracks this kind of decls.
  assert(Reader.GetDecl(ThisDeclID) == static_cast<T*>(D) &&
         "Invalid ThisDeclID ?");
  ASTReader::FirstLatestDeclIDMap::iterator I
      = Reader.FirstLatestDeclIDs.find(ThisDeclID);
  if (I != Reader.FirstLatestDeclIDs.end()) {
    Decl *NewLatest = Reader.GetDecl(I->second);
    D->RedeclLink
        = typename Redeclarable<T>::LatestDeclLink(cast_or_null<T>(NewLatest));
  }
}

//===----------------------------------------------------------------------===//
// Attribute Reading
//===----------------------------------------------------------------------===//

/// \brief Reads attributes from the current stream position.
void ASTReader::ReadAttributes(PerFileData &F, AttrVec &Attrs,
                               const RecordData &Record, unsigned &Idx) {
  for (unsigned i = 0, e = Record[Idx++]; i != e; ++i) {
    Attr *New = 0;
    attr::Kind Kind = (attr::Kind)Record[Idx++];
    SourceLocation Loc = ReadSourceLocation(F, Record, Idx);

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
static bool isConsumerInterestedIn(Decl *D) {
  if (isa<FileScopeAsmDecl>(D))
    return true;
  if (VarDecl *Var = dyn_cast<VarDecl>(D))
    return Var->isFileVarDecl() &&
           Var->isThisDeclarationADefinition() == VarDecl::Definition;
  if (FunctionDecl *Func = dyn_cast<FunctionDecl>(D))
    return Func->isThisDeclarationADefinition();
  return isa<ObjCProtocolDecl>(D) || isa<ObjCImplementationDecl>(D);
}

/// \brief Get the correct cursor and offset for loading a type.
ASTReader::RecordLocation
ASTReader::DeclCursorForIndex(unsigned Index, DeclID ID) {
  // See if there's an override.
  DeclReplacementMap::iterator It = ReplacedDecls.find(ID);
  if (It != ReplacedDecls.end())
    return RecordLocation(It->second.first, It->second.second);

  PerFileData *F = 0;
  for (unsigned I = 0, N = Chain.size(); I != N; ++I) {
    F = Chain[N - I - 1];
    if (Index < F->LocalNumDecls)
      break;
    Index -= F->LocalNumDecls;
  }
  assert(F && F->LocalNumDecls > Index && "Broken chain");
  return RecordLocation(F, F->DeclOffsets[Index]);
}

void ASTDeclReader::attachPreviousDecl(Decl *D, Decl *previous) {
  assert(D && previous);
  if (TagDecl *TD = dyn_cast<TagDecl>(D)) {
    TD->RedeclLink.setPointer(cast<TagDecl>(previous));
  } else if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    FD->RedeclLink.setPointer(cast<FunctionDecl>(previous));
  } else if (VarDecl *VD = dyn_cast<VarDecl>(D)) {
    VD->RedeclLink.setPointer(cast<VarDecl>(previous));
  } else {
    RedeclarableTemplateDecl *TD = cast<RedeclarableTemplateDecl>(D);
    TD->CommonOrPrev = cast<RedeclarableTemplateDecl>(previous);
  }
}

void ASTReader::loadAndAttachPreviousDecl(Decl *D, serialization::DeclID ID) {
  Decl *previous = GetDecl(ID);
  ASTDeclReader::attachPreviousDecl(D, previous);
}

/// \brief Read the declaration at the given offset from the AST file.
Decl *ASTReader::ReadDeclRecord(unsigned Index, DeclID ID) {
  RecordLocation Loc = DeclCursorForIndex(Index, ID);
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
  ASTDeclReader Reader(*this, *Loc.F, DeclsCursor, ID, Record, Idx);

  Decl *D = 0;
  switch ((DeclCode)DeclsCursor.ReadRecord(Code, Record)) {
  case DECL_CONTEXT_LEXICAL:
  case DECL_CONTEXT_VISIBLE:
    assert(false && "Record cannot be de-serialized with ReadDeclRecord");
    break;
  case DECL_TRANSLATION_UNIT:
    assert(Index == 0 && "Translation unit must be at index 0");
    D = Context->getTranslationUnitDecl();
    break;
  case DECL_TYPEDEF:
    D = TypedefDecl::Create(*Context, 0, SourceLocation(), SourceLocation(),
                            0, 0);
    break;
  case DECL_TYPEALIAS:
    D = TypeAliasDecl::Create(*Context, 0, SourceLocation(), SourceLocation(),
                              0, 0);
    break;
  case DECL_ENUM:
    D = EnumDecl::Create(*Context, Decl::EmptyShell());
    break;
  case DECL_RECORD:
    D = RecordDecl::Create(*Context, Decl::EmptyShell());
    break;
  case DECL_ENUM_CONSTANT:
    D = EnumConstantDecl::Create(*Context, 0, SourceLocation(), 0, QualType(),
                                 0, llvm::APSInt());
    break;
  case DECL_FUNCTION:
    D = FunctionDecl::Create(*Context, 0, SourceLocation(), SourceLocation(),
                             DeclarationName(), QualType(), 0);
    break;
  case DECL_LINKAGE_SPEC:
    D = LinkageSpecDecl::Create(*Context, 0, SourceLocation(), SourceLocation(),
                                (LinkageSpecDecl::LanguageIDs)0,
                                SourceLocation());
    break;
  case DECL_LABEL:
    D = LabelDecl::Create(*Context, 0, SourceLocation(), 0);
    break;
  case DECL_NAMESPACE:
    D = NamespaceDecl::Create(*Context, 0, SourceLocation(),
                              SourceLocation(), 0);
    break;
  case DECL_NAMESPACE_ALIAS:
    D = NamespaceAliasDecl::Create(*Context, 0, SourceLocation(),
                                   SourceLocation(), 0, 
                                   NestedNameSpecifierLoc(),
                                   SourceLocation(), 0);
    break;
  case DECL_USING:
    D = UsingDecl::Create(*Context, 0, SourceLocation(),
                          NestedNameSpecifierLoc(), DeclarationNameInfo(), 
                          false);
    break;
  case DECL_USING_SHADOW:
    D = UsingShadowDecl::Create(*Context, 0, SourceLocation(), 0, 0);
    break;
  case DECL_USING_DIRECTIVE:
    D = UsingDirectiveDecl::Create(*Context, 0, SourceLocation(),
                                   SourceLocation(), NestedNameSpecifierLoc(),
                                   SourceLocation(), 0, 0);
    break;
  case DECL_UNRESOLVED_USING_VALUE:
    D = UnresolvedUsingValueDecl::Create(*Context, 0, SourceLocation(),
                                         NestedNameSpecifierLoc(), 
                                         DeclarationNameInfo());
    break;
  case DECL_UNRESOLVED_USING_TYPENAME:
    D = UnresolvedUsingTypenameDecl::Create(*Context, 0, SourceLocation(),
                                            SourceLocation(), 
                                            NestedNameSpecifierLoc(),
                                            SourceLocation(),
                                            DeclarationName());
    break;
  case DECL_CXX_RECORD:
    D = CXXRecordDecl::Create(*Context, Decl::EmptyShell());
    break;
  case DECL_CXX_METHOD:
    D = CXXMethodDecl::Create(*Context, 0, SourceLocation(),
                              DeclarationNameInfo(), QualType(), 0,
                              false, SC_None, false, SourceLocation());
    break;
  case DECL_CXX_CONSTRUCTOR:
    D = CXXConstructorDecl::Create(*Context, Decl::EmptyShell());
    break;
  case DECL_CXX_DESTRUCTOR:
    D = CXXDestructorDecl::Create(*Context, Decl::EmptyShell());
    break;
  case DECL_CXX_CONVERSION:
    D = CXXConversionDecl::Create(*Context, Decl::EmptyShell());
    break;
  case DECL_ACCESS_SPEC:
    D = AccessSpecDecl::Create(*Context, Decl::EmptyShell());
    break;
  case DECL_FRIEND:
    D = FriendDecl::Create(*Context, Decl::EmptyShell());
    break;
  case DECL_FRIEND_TEMPLATE:
    D = FriendTemplateDecl::Create(*Context, Decl::EmptyShell());
    break;
  case DECL_CLASS_TEMPLATE:
    D = ClassTemplateDecl::Create(*Context, Decl::EmptyShell());
    break;
  case DECL_CLASS_TEMPLATE_SPECIALIZATION:
    D = ClassTemplateSpecializationDecl::Create(*Context, Decl::EmptyShell());
    break;
  case DECL_CLASS_TEMPLATE_PARTIAL_SPECIALIZATION:
    D = ClassTemplatePartialSpecializationDecl::Create(*Context,
                                                       Decl::EmptyShell());
    break;
  case DECL_FUNCTION_TEMPLATE:
      D = FunctionTemplateDecl::Create(*Context, Decl::EmptyShell());
    break;
  case DECL_TEMPLATE_TYPE_PARM:
    D = TemplateTypeParmDecl::Create(*Context, Decl::EmptyShell());
    break;
  case DECL_NON_TYPE_TEMPLATE_PARM:
    D = NonTypeTemplateParmDecl::Create(*Context, 0, SourceLocation(),
                                        SourceLocation(), 0, 0, 0, QualType(),
                                        false, 0);
    break;
  case DECL_EXPANDED_NON_TYPE_TEMPLATE_PARM_PACK:
    D = NonTypeTemplateParmDecl::Create(*Context, 0, SourceLocation(),
                                        SourceLocation(), 0, 0, 0, QualType(),
                                        0, 0, Record[Idx++], 0);
    break;
  case DECL_TEMPLATE_TEMPLATE_PARM:
    D = TemplateTemplateParmDecl::Create(*Context, 0, SourceLocation(), 0, 0,
                                         false, 0, 0);
    break;
  case DECL_STATIC_ASSERT:
    D = StaticAssertDecl::Create(*Context, 0, SourceLocation(), 0, 0,
                                 SourceLocation());
    break;

  case DECL_OBJC_METHOD:
    D = ObjCMethodDecl::Create(*Context, SourceLocation(), SourceLocation(),
                               Selector(), QualType(), 0, 0);
    break;
  case DECL_OBJC_INTERFACE:
    D = ObjCInterfaceDecl::Create(*Context, 0, SourceLocation(), 0);
    break;
  case DECL_OBJC_IVAR:
    D = ObjCIvarDecl::Create(*Context, 0, SourceLocation(), SourceLocation(),
                             0, QualType(), 0, ObjCIvarDecl::None);
    break;
  case DECL_OBJC_PROTOCOL:
    D = ObjCProtocolDecl::Create(*Context, 0, SourceLocation(), 0);
    break;
  case DECL_OBJC_AT_DEFS_FIELD:
    D = ObjCAtDefsFieldDecl::Create(*Context, 0, SourceLocation(),
                                    SourceLocation(), 0, QualType(), 0);
    break;
  case DECL_OBJC_CLASS:
    D = ObjCClassDecl::Create(*Context, 0, SourceLocation());
    break;
  case DECL_OBJC_FORWARD_PROTOCOL:
    D = ObjCForwardProtocolDecl::Create(*Context, 0, SourceLocation());
    break;
  case DECL_OBJC_CATEGORY:
    D = ObjCCategoryDecl::Create(*Context, 0, SourceLocation(), 
                                 SourceLocation(), SourceLocation(), 0);
    break;
  case DECL_OBJC_CATEGORY_IMPL:
    D = ObjCCategoryImplDecl::Create(*Context, 0, SourceLocation(), 0, 0);
    break;
  case DECL_OBJC_IMPLEMENTATION:
    D = ObjCImplementationDecl::Create(*Context, 0, SourceLocation(), 0, 0);
    break;
  case DECL_OBJC_COMPATIBLE_ALIAS:
    D = ObjCCompatibleAliasDecl::Create(*Context, 0, SourceLocation(), 0, 0);
    break;
  case DECL_OBJC_PROPERTY:
    D = ObjCPropertyDecl::Create(*Context, 0, SourceLocation(), 0, SourceLocation(),
                                 0);
    break;
  case DECL_OBJC_PROPERTY_IMPL:
    D = ObjCPropertyImplDecl::Create(*Context, 0, SourceLocation(),
                                     SourceLocation(), 0,
                                     ObjCPropertyImplDecl::Dynamic, 0,
                                     SourceLocation());
    break;
  case DECL_FIELD:
    D = FieldDecl::Create(*Context, 0, SourceLocation(), SourceLocation(), 0,
                          QualType(), 0, 0, false);
    break;
  case DECL_INDIRECTFIELD:
    D = IndirectFieldDecl::Create(*Context, 0, SourceLocation(), 0, QualType(),
                                  0, 0);
    break;
  case DECL_VAR:
    D = VarDecl::Create(*Context, 0, SourceLocation(), SourceLocation(), 0,
                        QualType(), 0, SC_None, SC_None);
    break;

  case DECL_IMPLICIT_PARAM:
    D = ImplicitParamDecl::Create(*Context, 0, SourceLocation(), 0, QualType());
    break;

  case DECL_PARM_VAR:
    D = ParmVarDecl::Create(*Context, 0, SourceLocation(), SourceLocation(), 0,
                            QualType(), 0, SC_None, SC_None, 0);
    break;
  case DECL_FILE_SCOPE_ASM:
    D = FileScopeAsmDecl::Create(*Context, 0, 0, SourceLocation(),
                                 SourceLocation());
    break;
  case DECL_BLOCK:
    D = BlockDecl::Create(*Context, 0, SourceLocation());
    break;
  case DECL_CXX_BASE_SPECIFIERS:
    Error("attempt to read a C++ base-specifier record as a declaration");
    return 0;
  }

  assert(D && "Unknown declaration reading AST file");
  LoadedDecl(Index, D);
  Reader.Visit(D);

  // If this declaration is also a declaration context, get the
  // offsets for its tables of lexical and visible declarations.
  if (DeclContext *DC = dyn_cast<DeclContext>(D)) {
    std::pair<uint64_t, uint64_t> Offsets = Reader.VisitDeclContext(DC);
    if (Offsets.first || Offsets.second) {
      DC->setHasExternalLexicalStorage(Offsets.first != 0);
      DC->setHasExternalVisibleStorage(Offsets.second != 0);
      DeclContextInfo Info;
      if (ReadDeclContextStorage(DeclsCursor, Offsets, Info))
        return 0;
      DeclContextInfos &Infos = DeclContextOffsets[DC];
      // Reading the TU will happen after reading its lexical update blocks,
      // so we need to make sure we insert in front. For all other contexts,
      // the vector is empty here anyway, so there's no loss in efficiency.
      Infos.insert(Infos.begin(), Info);
    }

    // Now add the pending visible updates for this decl context, if it has any.
    DeclContextVisibleUpdatesPending::iterator I =
        PendingVisibleUpdates.find(ID);
    if (I != PendingVisibleUpdates.end()) {
      // There are updates. This means the context has external visible
      // storage, even if the original stored version didn't.
      DC->setHasExternalVisibleStorage(true);
      DeclContextVisibleUpdates &U = I->second;
      DeclContextInfos &Infos = DeclContextOffsets[DC];
      DeclContextInfo Info;
      Info.LexicalDecls = 0;
      Info.NumLexicalDecls = 0;
      for (DeclContextVisibleUpdates::iterator UI = U.begin(), UE = U.end();
           UI != UE; ++UI) {
        Info.NameLookupTableData = *UI;
        Infos.push_back(Info);
      }
      PendingVisibleUpdates.erase(I);
    }
  }
  assert(Idx == Record.size());

  // The declaration may have been modified by files later in the chain.
  // If this is the case, read the record containing the updates from each file
  // and pass it to ASTDeclReader to make the modifications.
  DeclUpdateOffsetsMap::iterator UpdI = DeclUpdateOffsets.find(ID);
  if (UpdI != DeclUpdateOffsets.end()) {
    FileOffsetsTy &UpdateOffsets = UpdI->second;
    for (FileOffsetsTy::iterator
           I = UpdateOffsets.begin(), E = UpdateOffsets.end(); I != E; ++I) {
      PerFileData *F = I->first;
      uint64_t Offset = I->second;
      llvm::BitstreamCursor &Cursor = F->DeclsCursor;
      SavedStreamPosition SavedPosition(Cursor);
      Cursor.JumpToBit(Offset);
      RecordData Record;
      unsigned Code = Cursor.ReadCode();
      unsigned RecCode = Cursor.ReadRecord(Code, Record);
      (void)RecCode;
      assert(RecCode == DECL_UPDATES && "Expected DECL_UPDATES record!");
      Reader.UpdateDecl(D, *F, Record);
    }
  }

  // If we have deserialized a declaration that has a definition the
  // AST consumer might need to know about, queue it.
  // We don't pass it to the consumer immediately because we may be in recursive
  // loading, and some declarations may still be initializing.
  if (isConsumerInterestedIn(D))
    InterestingDecls.push_back(D);

  return D;
}

void ASTDeclReader::UpdateDecl(Decl *D, ASTReader::PerFileData &Module,
                               const RecordData &Record) {
  unsigned Idx = 0;
  while (Idx < Record.size()) {
    switch ((DeclUpdateKind)Record[Idx++]) {
    case UPD_CXX_SET_DEFINITIONDATA: {
      CXXRecordDecl *RD = cast<CXXRecordDecl>(D);
      CXXRecordDecl *
          DefinitionDecl = cast<CXXRecordDecl>(Reader.GetDecl(Record[Idx++]));
      assert(!RD->DefinitionData && "DefinitionData is already set!");
      InitializeCXXDefinitionData(RD, DefinitionDecl, Record, Idx);
      break;
    }

    case UPD_CXX_ADDED_IMPLICIT_MEMBER:
      cast<CXXRecordDecl>(D)->addedMember(Reader.GetDecl(Record[Idx++]));
      break;

    case UPD_CXX_ADDED_TEMPLATE_SPECIALIZATION:
      // It will be added to the template's specializations set when loaded.
      Reader.GetDecl(Record[Idx++]);
      break;

    case UPD_CXX_ADDED_ANONYMOUS_NAMESPACE: {
      NamespaceDecl *Anon = cast<NamespaceDecl>(Reader.GetDecl(Record[Idx++]));
      // Guard against these being loaded out of original order. Don't use
      // getNextNamespace(), since it tries to access the context and can't in
      // the middle of deserialization.
      if (!Anon->NextNamespace) {
        if (TranslationUnitDecl *TU = dyn_cast<TranslationUnitDecl>(D))
          TU->setAnonymousNamespace(Anon);
        else
          cast<NamespaceDecl>(D)->OrigOrAnonNamespace.setPointer(Anon);
      }
      break;
    }

    case UPD_CXX_INSTANTIATED_STATIC_DATA_MEMBER:
      cast<VarDecl>(D)->getMemberSpecializationInfo()->setPointOfInstantiation(
          Reader.ReadSourceLocation(Module, Record, Idx));
      break;
    }
  }
}
