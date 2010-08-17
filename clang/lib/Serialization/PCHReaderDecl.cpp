//===--- PCHReaderDecl.cpp - Decl Deserialization ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the PCHReader::ReadDeclRecord method, which is the
// entrypoint for loading a decl.
//
//===----------------------------------------------------------------------===//

#include "clang/Serialization/PCHReader.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/DeclGroup.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
using namespace clang;


//===----------------------------------------------------------------------===//
// Declaration deserialization
//===----------------------------------------------------------------------===//

namespace clang {
  class PCHDeclReader : public DeclVisitor<PCHDeclReader, void> {
    PCHReader &Reader;
    llvm::BitstreamCursor &Cursor;
    const pch::DeclID ThisDeclID;
    const PCHReader::RecordData &Record;
    unsigned &Idx;
    pch::TypeID TypeIDForTypeDecl;

    uint64_t GetCurrentCursorOffset();

  public:
    PCHDeclReader(PCHReader &Reader, llvm::BitstreamCursor &Cursor,
                  pch::DeclID thisDeclID, const PCHReader::RecordData &Record,
                  unsigned &Idx)
      : Reader(Reader), Cursor(Cursor), ThisDeclID(thisDeclID), Record(Record),
        Idx(Idx), TypeIDForTypeDecl(0) { }

    void Visit(Decl *D);

    void VisitDecl(Decl *D);
    void VisitTranslationUnitDecl(TranslationUnitDecl *TU);
    void VisitNamedDecl(NamedDecl *ND);
    void VisitNamespaceDecl(NamespaceDecl *D);
    void VisitUsingDirectiveDecl(UsingDirectiveDecl *D);
    void VisitNamespaceAliasDecl(NamespaceAliasDecl *D);
    void VisitTypeDecl(TypeDecl *TD);
    void VisitTypedefDecl(TypedefDecl *TD);
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

uint64_t PCHDeclReader::GetCurrentCursorOffset() {
  uint64_t Off = 0;
  for (unsigned I = 0, N = Reader.Chain.size(); I != N; ++I) {
    PCHReader::PerFileData &F = *Reader.Chain[N - I - 1];
    if (&Cursor == &F.DeclsCursor) {
      Off += F.DeclsCursor.GetCurrentBitNo();
      break;
    }
    Off += F.SizeInBits;
  }
  return Off;
}

void PCHDeclReader::Visit(Decl *D) {
  DeclVisitor<PCHDeclReader, void>::Visit(D);

  if (TypeDecl *TD = dyn_cast<TypeDecl>(D)) {
    // if we have a fully initialized TypeDecl, we can safely read its type now.
    TD->setTypeForDecl(Reader.GetType(TypeIDForTypeDecl).getTypePtr());
  } else if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    // FunctionDecl's body was written last after all other Stmts/Exprs.
    if (Record[Idx++])
      FD->setLazyBody(GetCurrentCursorOffset());
  }
}

void PCHDeclReader::VisitDecl(Decl *D) {
  D->setDeclContext(cast_or_null<DeclContext>(Reader.GetDecl(Record[Idx++])));
  D->setLexicalDeclContext(
                     cast_or_null<DeclContext>(Reader.GetDecl(Record[Idx++])));
  D->setLocation(SourceLocation::getFromRawEncoding(Record[Idx++]));
  D->setInvalidDecl(Record[Idx++]);
  if (Record[Idx++])
    D->initAttrs(Reader.ReadAttributes(Cursor));
  D->setImplicit(Record[Idx++]);
  D->setUsed(Record[Idx++]);
  D->setAccess((AccessSpecifier)Record[Idx++]);
  D->setPCHLevel(Record[Idx++] + 1);
}

void PCHDeclReader::VisitTranslationUnitDecl(TranslationUnitDecl *TU) {
  VisitDecl(TU);
  TU->setAnonymousNamespace(
                    cast_or_null<NamespaceDecl>(Reader.GetDecl(Record[Idx++])));
}

void PCHDeclReader::VisitNamedDecl(NamedDecl *ND) {
  VisitDecl(ND);
  ND->setDeclName(Reader.ReadDeclarationName(Record, Idx));
}

void PCHDeclReader::VisitTypeDecl(TypeDecl *TD) {
  VisitNamedDecl(TD);
  // Delay type reading until after we have fully initialized the decl.
  TypeIDForTypeDecl = Record[Idx++];
}

void PCHDeclReader::VisitTypedefDecl(TypedefDecl *TD) {
  VisitTypeDecl(TD);
  TD->setTypeSourceInfo(Reader.GetTypeSourceInfo(Cursor, Record, Idx));
}

void PCHDeclReader::VisitTagDecl(TagDecl *TD) {
  VisitTypeDecl(TD);
  TD->IdentifierNamespace = Record[Idx++];
  VisitRedeclarable(TD);
  TD->setTagKind((TagDecl::TagKind)Record[Idx++]);
  TD->setDefinition(Record[Idx++]);
  TD->setEmbeddedInDeclarator(Record[Idx++]);
  TD->setRBraceLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  TD->setTagKeywordLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  // FIXME: maybe read optional qualifier and its range.
  TD->setTypedefForAnonDecl(
                    cast_or_null<TypedefDecl>(Reader.GetDecl(Record[Idx++])));
}

void PCHDeclReader::VisitEnumDecl(EnumDecl *ED) {
  VisitTagDecl(ED);
  ED->setIntegerType(Reader.GetType(Record[Idx++]));
  ED->setPromotionType(Reader.GetType(Record[Idx++]));
  ED->setNumPositiveBits(Record[Idx++]);
  ED->setNumNegativeBits(Record[Idx++]);
  ED->setInstantiationOfMemberEnum(
                         cast_or_null<EnumDecl>(Reader.GetDecl(Record[Idx++])));
}

void PCHDeclReader::VisitRecordDecl(RecordDecl *RD) {
  VisitTagDecl(RD);
  RD->setHasFlexibleArrayMember(Record[Idx++]);
  RD->setAnonymousStructOrUnion(Record[Idx++]);
  RD->setHasObjectMember(Record[Idx++]);
}

void PCHDeclReader::VisitValueDecl(ValueDecl *VD) {
  VisitNamedDecl(VD);
  VD->setType(Reader.GetType(Record[Idx++]));
}

void PCHDeclReader::VisitEnumConstantDecl(EnumConstantDecl *ECD) {
  VisitValueDecl(ECD);
  if (Record[Idx++])
    ECD->setInitExpr(Reader.ReadExpr(Cursor));
  ECD->setInitVal(Reader.ReadAPSInt(Record, Idx));
}

void PCHDeclReader::VisitDeclaratorDecl(DeclaratorDecl *DD) {
  VisitValueDecl(DD);
  TypeSourceInfo *TInfo = Reader.GetTypeSourceInfo(Cursor, Record, Idx);
  if (TInfo)
    DD->setTypeSourceInfo(TInfo);
  // FIXME: read optional qualifier and its range.
}

void PCHDeclReader::VisitFunctionDecl(FunctionDecl *FD) {
  VisitDeclaratorDecl(FD);
  // FIXME: read DeclarationNameLoc.

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
    SourceLocation POI = Reader.ReadSourceLocation(Record, Idx);
    FD->setInstantiationOfMemberFunction(InstFD, TSK);
    FD->getMemberSpecializationInfo()->setPointOfInstantiation(POI);
    break;
  }
  case FunctionDecl::TK_FunctionTemplateSpecialization: {
    FunctionTemplateDecl *Template
      = cast<FunctionTemplateDecl>(Reader.GetDecl(Record[Idx++]));
    TemplateSpecializationKind TSK = (TemplateSpecializationKind)Record[Idx++];
    
    // Template arguments.
    llvm::SmallVector<TemplateArgument, 8> TemplArgs;
    Reader.ReadTemplateArgumentList(TemplArgs, Cursor, Record, Idx);
    
    // Template args as written.
    llvm::SmallVector<TemplateArgumentLoc, 8> TemplArgLocs;
    SourceLocation LAngleLoc, RAngleLoc;
    if (Record[Idx++]) {  // TemplateArgumentsAsWritten != 0
      unsigned NumTemplateArgLocs = Record[Idx++];
      TemplArgLocs.reserve(NumTemplateArgLocs);
      for (unsigned i=0; i != NumTemplateArgLocs; ++i)
        TemplArgLocs.push_back(
            Reader.ReadTemplateArgumentLoc(Cursor, Record, Idx));
  
      LAngleLoc = Reader.ReadSourceLocation(Record, Idx);
      RAngleLoc = Reader.ReadSourceLocation(Record, Idx);
    }
    
    SourceLocation POI = Reader.ReadSourceLocation(Record, Idx);

    if (FD->isCanonicalDecl()) // if canonical add to template's set.
      FD->setFunctionTemplateSpecialization(Template, TemplArgs.size(),
                                            TemplArgs.data(), TSK,
                                            TemplArgLocs.size(),
                                            TemplArgLocs.data(),
                                            LAngleLoc, RAngleLoc, POI);
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
      TemplArgs.addArgument(Reader.ReadTemplateArgumentLoc(Cursor,Record, Idx));
    TemplArgs.setLAngleLoc(Reader.ReadSourceLocation(Record, Idx));
    TemplArgs.setRAngleLoc(Reader.ReadSourceLocation(Record, Idx));
    
    FD->setDependentTemplateSpecialization(*Reader.getContext(),
                                           TemplDecls, TemplArgs);
    break;
  }
  }

  // FunctionDecl's body is handled last at PCHReaderDecl::Visit,
  // after everything else is read.

  VisitRedeclarable(FD);
  FD->setStorageClass((FunctionDecl::StorageClass)Record[Idx++]);
  FD->setStorageClassAsWritten((FunctionDecl::StorageClass)Record[Idx++]);
  FD->setInlineSpecified(Record[Idx++]);
  FD->setVirtualAsWritten(Record[Idx++]);
  FD->setPure(Record[Idx++]);
  FD->setHasInheritedPrototype(Record[Idx++]);
  FD->setHasWrittenPrototype(Record[Idx++]);
  FD->setDeleted(Record[Idx++]);
  FD->setTrivial(Record[Idx++]);
  FD->setCopyAssignment(Record[Idx++]);
  FD->setHasImplicitReturnZero(Record[Idx++]);
  FD->setLocEnd(SourceLocation::getFromRawEncoding(Record[Idx++]));

  // Read in the parameters.
  unsigned NumParams = Record[Idx++];
  llvm::SmallVector<ParmVarDecl *, 16> Params;
  Params.reserve(NumParams);
  for (unsigned I = 0; I != NumParams; ++I)
    Params.push_back(cast<ParmVarDecl>(Reader.GetDecl(Record[Idx++])));
  FD->setParams(Params.data(), NumParams);
}

void PCHDeclReader::VisitObjCMethodDecl(ObjCMethodDecl *MD) {
  VisitNamedDecl(MD);
  if (Record[Idx++]) {
    // In practice, this won't be executed (since method definitions
    // don't occur in header files).
    MD->setBody(Reader.ReadStmt(Cursor));
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
  MD->setResultTypeSourceInfo(Reader.GetTypeSourceInfo(Cursor, Record, Idx));
  MD->setEndLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  unsigned NumParams = Record[Idx++];
  llvm::SmallVector<ParmVarDecl *, 16> Params;
  Params.reserve(NumParams);
  for (unsigned I = 0; I != NumParams; ++I)
    Params.push_back(cast<ParmVarDecl>(Reader.GetDecl(Record[Idx++])));
  MD->setMethodParams(*Reader.getContext(), Params.data(), NumParams,
                      NumParams);
}

void PCHDeclReader::VisitObjCContainerDecl(ObjCContainerDecl *CD) {
  VisitNamedDecl(CD);
  SourceLocation A = SourceLocation::getFromRawEncoding(Record[Idx++]);
  SourceLocation B = SourceLocation::getFromRawEncoding(Record[Idx++]);
  CD->setAtEndRange(SourceRange(A, B));
}

void PCHDeclReader::VisitObjCInterfaceDecl(ObjCInterfaceDecl *ID) {
  VisitObjCContainerDecl(ID);
  ID->setTypeForDecl(Reader.GetType(Record[Idx++]).getTypePtr());
  ID->setSuperClass(cast_or_null<ObjCInterfaceDecl>
                       (Reader.GetDecl(Record[Idx++])));
  unsigned NumProtocols = Record[Idx++];
  llvm::SmallVector<ObjCProtocolDecl *, 16> Protocols;
  Protocols.reserve(NumProtocols);
  for (unsigned I = 0; I != NumProtocols; ++I)
    Protocols.push_back(cast<ObjCProtocolDecl>(Reader.GetDecl(Record[Idx++])));
  llvm::SmallVector<SourceLocation, 16> ProtoLocs;
  ProtoLocs.reserve(NumProtocols);
  for (unsigned I = 0; I != NumProtocols; ++I)
    ProtoLocs.push_back(SourceLocation::getFromRawEncoding(Record[Idx++]));
  ID->setProtocolList(Protocols.data(), NumProtocols, ProtoLocs.data(),
                      *Reader.getContext());
  unsigned NumIvars = Record[Idx++];
  llvm::SmallVector<ObjCIvarDecl *, 16> IVars;
  IVars.reserve(NumIvars);
  for (unsigned I = 0; I != NumIvars; ++I)
    IVars.push_back(cast<ObjCIvarDecl>(Reader.GetDecl(Record[Idx++])));
  ID->setCategoryList(
               cast_or_null<ObjCCategoryDecl>(Reader.GetDecl(Record[Idx++])));
  ID->setForwardDecl(Record[Idx++]);
  ID->setImplicitInterfaceDecl(Record[Idx++]);
  ID->setClassLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  ID->setSuperClassLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  ID->setLocEnd(SourceLocation::getFromRawEncoding(Record[Idx++]));
}

void PCHDeclReader::VisitObjCIvarDecl(ObjCIvarDecl *IVD) {
  VisitFieldDecl(IVD);
  IVD->setAccessControl((ObjCIvarDecl::AccessControl)Record[Idx++]);
  bool synth = Record[Idx++];
  IVD->setSynthesize(synth);
}

void PCHDeclReader::VisitObjCProtocolDecl(ObjCProtocolDecl *PD) {
  VisitObjCContainerDecl(PD);
  PD->setForwardDecl(Record[Idx++]);
  PD->setLocEnd(SourceLocation::getFromRawEncoding(Record[Idx++]));
  unsigned NumProtoRefs = Record[Idx++];
  llvm::SmallVector<ObjCProtocolDecl *, 16> ProtoRefs;
  ProtoRefs.reserve(NumProtoRefs);
  for (unsigned I = 0; I != NumProtoRefs; ++I)
    ProtoRefs.push_back(cast<ObjCProtocolDecl>(Reader.GetDecl(Record[Idx++])));
  llvm::SmallVector<SourceLocation, 16> ProtoLocs;
  ProtoLocs.reserve(NumProtoRefs);
  for (unsigned I = 0; I != NumProtoRefs; ++I)
    ProtoLocs.push_back(SourceLocation::getFromRawEncoding(Record[Idx++]));
  PD->setProtocolList(ProtoRefs.data(), NumProtoRefs, ProtoLocs.data(),
                      *Reader.getContext());
}

void PCHDeclReader::VisitObjCAtDefsFieldDecl(ObjCAtDefsFieldDecl *FD) {
  VisitFieldDecl(FD);
}

void PCHDeclReader::VisitObjCClassDecl(ObjCClassDecl *CD) {
  VisitDecl(CD);
  unsigned NumClassRefs = Record[Idx++];
  llvm::SmallVector<ObjCInterfaceDecl *, 16> ClassRefs;
  ClassRefs.reserve(NumClassRefs);
  for (unsigned I = 0; I != NumClassRefs; ++I)
    ClassRefs.push_back(cast<ObjCInterfaceDecl>(Reader.GetDecl(Record[Idx++])));
  llvm::SmallVector<SourceLocation, 16> SLocs;
  SLocs.reserve(NumClassRefs);
  for (unsigned I = 0; I != NumClassRefs; ++I)
    SLocs.push_back(SourceLocation::getFromRawEncoding(Record[Idx++]));
  CD->setClassList(*Reader.getContext(), ClassRefs.data(), SLocs.data(),
                   NumClassRefs);
}

void PCHDeclReader::VisitObjCForwardProtocolDecl(ObjCForwardProtocolDecl *FPD) {
  VisitDecl(FPD);
  unsigned NumProtoRefs = Record[Idx++];
  llvm::SmallVector<ObjCProtocolDecl *, 16> ProtoRefs;
  ProtoRefs.reserve(NumProtoRefs);
  for (unsigned I = 0; I != NumProtoRefs; ++I)
    ProtoRefs.push_back(cast<ObjCProtocolDecl>(Reader.GetDecl(Record[Idx++])));
  llvm::SmallVector<SourceLocation, 16> ProtoLocs;
  ProtoLocs.reserve(NumProtoRefs);
  for (unsigned I = 0; I != NumProtoRefs; ++I)
    ProtoLocs.push_back(SourceLocation::getFromRawEncoding(Record[Idx++]));
  FPD->setProtocolList(ProtoRefs.data(), NumProtoRefs, ProtoLocs.data(),
                       *Reader.getContext());
}

void PCHDeclReader::VisitObjCCategoryDecl(ObjCCategoryDecl *CD) {
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
    ProtoLocs.push_back(SourceLocation::getFromRawEncoding(Record[Idx++]));
  CD->setProtocolList(ProtoRefs.data(), NumProtoRefs, ProtoLocs.data(),
                      *Reader.getContext());
  CD->setNextClassCategory(cast_or_null<ObjCCategoryDecl>(Reader.GetDecl(Record[Idx++])));
  CD->setAtLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  CD->setCategoryNameLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
}

void PCHDeclReader::VisitObjCCompatibleAliasDecl(ObjCCompatibleAliasDecl *CAD) {
  VisitNamedDecl(CAD);
  CAD->setClassInterface(cast<ObjCInterfaceDecl>(Reader.GetDecl(Record[Idx++])));
}

void PCHDeclReader::VisitObjCPropertyDecl(ObjCPropertyDecl *D) {
  VisitNamedDecl(D);
  D->setAtLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  D->setType(Reader.GetTypeSourceInfo(Cursor, Record, Idx));
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

void PCHDeclReader::VisitObjCImplDecl(ObjCImplDecl *D) {
  VisitObjCContainerDecl(D);
  D->setClassInterface(
              cast_or_null<ObjCInterfaceDecl>(Reader.GetDecl(Record[Idx++])));
}

void PCHDeclReader::VisitObjCCategoryImplDecl(ObjCCategoryImplDecl *D) {
  VisitObjCImplDecl(D);
  D->setIdentifier(Reader.GetIdentifierInfo(Record, Idx));
}

void PCHDeclReader::VisitObjCImplementationDecl(ObjCImplementationDecl *D) {
  VisitObjCImplDecl(D);
  D->setSuperClass(
              cast_or_null<ObjCInterfaceDecl>(Reader.GetDecl(Record[Idx++])));
  llvm::tie(D->IvarInitializers, D->NumIvarInitializers)
      = Reader.ReadCXXBaseOrMemberInitializers(Cursor, Record, Idx);
}


void PCHDeclReader::VisitObjCPropertyImplDecl(ObjCPropertyImplDecl *D) {
  VisitDecl(D);
  D->setAtLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  D->setPropertyDecl(
               cast_or_null<ObjCPropertyDecl>(Reader.GetDecl(Record[Idx++])));
  D->setPropertyIvarDecl(
                   cast_or_null<ObjCIvarDecl>(Reader.GetDecl(Record[Idx++])));
  D->setGetterCXXConstructor(Reader.ReadExpr(Cursor));
  D->setSetterCXXAssignment(Reader.ReadExpr(Cursor));
}

void PCHDeclReader::VisitFieldDecl(FieldDecl *FD) {
  VisitDeclaratorDecl(FD);
  FD->setMutable(Record[Idx++]);
  if (Record[Idx++])
    FD->setBitWidth(Reader.ReadExpr(Cursor));
  if (!FD->getDeclName()) {
    FieldDecl *Tmpl = cast_or_null<FieldDecl>(Reader.GetDecl(Record[Idx++]));
    if (Tmpl)
      Reader.getContext()->setInstantiatedFromUnnamedFieldDecl(FD, Tmpl);
  }
}

void PCHDeclReader::VisitVarDecl(VarDecl *VD) {
  VisitDeclaratorDecl(VD);
  VD->setStorageClass((VarDecl::StorageClass)Record[Idx++]);
  VD->setStorageClassAsWritten((VarDecl::StorageClass)Record[Idx++]);
  VD->setThreadSpecified(Record[Idx++]);
  VD->setCXXDirectInitializer(Record[Idx++]);
  VD->setExceptionVariable(Record[Idx++]);
  VD->setNRVOVariable(Record[Idx++]);
  VisitRedeclarable(VD);
  if (Record[Idx++])
    VD->setInit(Reader.ReadExpr(Cursor));

  if (Record[Idx++]) { // HasMemberSpecializationInfo.
    VarDecl *Tmpl = cast<VarDecl>(Reader.GetDecl(Record[Idx++]));
    TemplateSpecializationKind TSK = (TemplateSpecializationKind)Record[Idx++];
    SourceLocation POI = Reader.ReadSourceLocation(Record, Idx);
    Reader.getContext()->setInstantiatedFromStaticDataMember(VD, Tmpl, TSK,POI);
  }
}

void PCHDeclReader::VisitImplicitParamDecl(ImplicitParamDecl *PD) {
  VisitVarDecl(PD);
}

void PCHDeclReader::VisitParmVarDecl(ParmVarDecl *PD) {
  VisitVarDecl(PD);
  PD->setObjCDeclQualifier((Decl::ObjCDeclQualifier)Record[Idx++]);
  PD->setHasInheritedDefaultArg(Record[Idx++]);
  if (Record[Idx++]) // hasUninstantiatedDefaultArg.
    PD->setUninstantiatedDefaultArg(Reader.ReadExpr(Cursor));
}

void PCHDeclReader::VisitFileScopeAsmDecl(FileScopeAsmDecl *AD) {
  VisitDecl(AD);
  AD->setAsmString(cast<StringLiteral>(Reader.ReadExpr(Cursor)));
}

void PCHDeclReader::VisitBlockDecl(BlockDecl *BD) {
  VisitDecl(BD);
  BD->setBody(cast_or_null<CompoundStmt>(Reader.ReadStmt(Cursor)));
  BD->setSignatureAsWritten(Reader.GetTypeSourceInfo(Cursor, Record, Idx));
  unsigned NumParams = Record[Idx++];
  llvm::SmallVector<ParmVarDecl *, 16> Params;
  Params.reserve(NumParams);
  for (unsigned I = 0; I != NumParams; ++I)
    Params.push_back(cast<ParmVarDecl>(Reader.GetDecl(Record[Idx++])));
  BD->setParams(Params.data(), NumParams);
}

void PCHDeclReader::VisitLinkageSpecDecl(LinkageSpecDecl *D) {
  VisitDecl(D);
  D->setLanguage((LinkageSpecDecl::LanguageIDs)Record[Idx++]);
  D->setHasBraces(Record[Idx++]);
}

void PCHDeclReader::VisitNamespaceDecl(NamespaceDecl *D) {
  VisitNamedDecl(D);
  D->setLBracLoc(Reader.ReadSourceLocation(Record, Idx));
  D->setRBracLoc(Reader.ReadSourceLocation(Record, Idx));
  D->setNextNamespace(
                    cast_or_null<NamespaceDecl>(Reader.GetDecl(Record[Idx++])));

  bool IsOriginal = Record[Idx++];
  D->OrigOrAnonNamespace.setInt(IsOriginal);
  D->OrigOrAnonNamespace.setPointer(
                    cast_or_null<NamespaceDecl>(Reader.GetDecl(Record[Idx++])));
}

void PCHDeclReader::VisitNamespaceAliasDecl(NamespaceAliasDecl *D) {
  VisitNamedDecl(D);

  D->setAliasLoc(Reader.ReadSourceLocation(Record, Idx));
  D->setQualifierRange(Reader.ReadSourceRange(Record, Idx));
  D->setQualifier(Reader.ReadNestedNameSpecifier(Record, Idx));
  D->setTargetNameLoc(Reader.ReadSourceLocation(Record, Idx));
  D->setAliasedNamespace(cast<NamedDecl>(Reader.GetDecl(Record[Idx++])));
}

void PCHDeclReader::VisitUsingDecl(UsingDecl *D) {
  VisitNamedDecl(D);
  D->setUsingLocation(Reader.ReadSourceLocation(Record, Idx));
  D->setNestedNameRange(Reader.ReadSourceRange(Record, Idx));
  D->setTargetNestedNameDecl(Reader.ReadNestedNameSpecifier(Record, Idx));
  // FIXME: read the DNLoc component.

  // FIXME: It would probably be more efficient to read these into a vector
  // and then re-cosntruct the shadow decl set over that vector since it
  // would avoid existence checks.
  unsigned NumShadows = Record[Idx++];
  for(unsigned I = 0; I != NumShadows; ++I) {
    // Avoid invariant checking of UsingDecl::addShadowDecl, the decl may still
    // be initializing.
    D->Shadows.insert(cast<UsingShadowDecl>(Reader.GetDecl(Record[Idx++])));
  }
  D->setTypeName(Record[Idx++]);
  NamedDecl *Pattern = cast_or_null<NamedDecl>(Reader.GetDecl(Record[Idx++]));
  if (Pattern)
    Reader.getContext()->setInstantiatedFromUsingDecl(D, Pattern);
}

void PCHDeclReader::VisitUsingShadowDecl(UsingShadowDecl *D) {
  VisitNamedDecl(D);
  D->setTargetDecl(cast<NamedDecl>(Reader.GetDecl(Record[Idx++])));
  D->setUsingDecl(cast<UsingDecl>(Reader.GetDecl(Record[Idx++])));
  UsingShadowDecl *Pattern
      = cast_or_null<UsingShadowDecl>(Reader.GetDecl(Record[Idx++]));
  if (Pattern)
    Reader.getContext()->setInstantiatedFromUsingShadowDecl(D, Pattern);
}

void PCHDeclReader::VisitUsingDirectiveDecl(UsingDirectiveDecl *D) {
  VisitNamedDecl(D);
  D->setNamespaceKeyLocation(Reader.ReadSourceLocation(Record, Idx));
  D->setQualifierRange(Reader.ReadSourceRange(Record, Idx));
  D->setQualifier(Reader.ReadNestedNameSpecifier(Record, Idx));
  D->setIdentLocation(Reader.ReadSourceLocation(Record, Idx));
  D->setNominatedNamespace(cast<NamedDecl>(Reader.GetDecl(Record[Idx++])));
  D->setCommonAncestor(cast_or_null<DeclContext>(
                                                Reader.GetDecl(Record[Idx++])));
}

void PCHDeclReader::VisitUnresolvedUsingValueDecl(UnresolvedUsingValueDecl *D) {
  VisitValueDecl(D);
  D->setTargetNestedNameRange(Reader.ReadSourceRange(Record, Idx));
  D->setUsingLoc(Reader.ReadSourceLocation(Record, Idx));
  D->setTargetNestedNameSpecifier(Reader.ReadNestedNameSpecifier(Record, Idx));
  // FIXME: read the DNLoc component.
}

void PCHDeclReader::VisitUnresolvedUsingTypenameDecl(
                                               UnresolvedUsingTypenameDecl *D) {
  VisitTypeDecl(D);
  D->setTargetNestedNameRange(Reader.ReadSourceRange(Record, Idx));
  D->setUsingLoc(Reader.ReadSourceLocation(Record, Idx));
  D->setTypenameLoc(Reader.ReadSourceLocation(Record, Idx));
  D->setTargetNestedNameSpecifier(Reader.ReadNestedNameSpecifier(Record, Idx));
}

void PCHDeclReader::VisitCXXRecordDecl(CXXRecordDecl *D) {
  ASTContext &C = *Reader.getContext();

  // We need to allocate the DefinitionData struct ahead of VisitRecordDecl
  // so that the other CXXRecordDecls can get a pointer even when the owner
  // is still initializing.
  bool OwnsDefinitionData = false;
  enum DataOwnership { Data_NoDefData, Data_Owner, Data_NotOwner };
  switch ((DataOwnership)Record[Idx++]) {
  default:
    assert(0 && "Out of sync with PCHDeclWriter or messed up reading");
  case Data_NoDefData:
    break;
  case Data_Owner:
    OwnsDefinitionData = true;
    D->DefinitionData = new (C) struct CXXRecordDecl::DefinitionData(D);
    break;
  case Data_NotOwner:
    D->DefinitionData
        = cast<CXXRecordDecl>(Reader.GetDecl(Record[Idx++]))->DefinitionData;
    break;
  }

  VisitRecordDecl(D);

  if (OwnsDefinitionData) {
    assert(D->DefinitionData);
    struct CXXRecordDecl::DefinitionData &Data = *D->DefinitionData;

    Data.UserDeclaredConstructor = Record[Idx++];
    Data.UserDeclaredCopyConstructor = Record[Idx++];
    Data.UserDeclaredCopyAssignment = Record[Idx++];
    Data.UserDeclaredDestructor = Record[Idx++];
    Data.Aggregate = Record[Idx++];
    Data.PlainOldData = Record[Idx++];
    Data.Empty = Record[Idx++];
    Data.Polymorphic = Record[Idx++];
    Data.Abstract = Record[Idx++];
    Data.HasTrivialConstructor = Record[Idx++];
    Data.HasTrivialCopyConstructor = Record[Idx++];
    Data.HasTrivialCopyAssignment = Record[Idx++];
    Data.HasTrivialDestructor = Record[Idx++];
    Data.ComputedVisibleConversions = Record[Idx++];
    Data.DeclaredDefaultConstructor = Record[Idx++];
    Data.DeclaredCopyConstructor = Record[Idx++];
    Data.DeclaredCopyAssignment = Record[Idx++];
    Data.DeclaredDestructor = Record[Idx++];

    // setBases() is unsuitable since it may try to iterate the bases of an
    // uninitialized base.
    Data.NumBases = Record[Idx++];
    Data.Bases = new(C) CXXBaseSpecifier [Data.NumBases];
    for (unsigned i = 0; i != Data.NumBases; ++i)
      Data.Bases[i] = Reader.ReadCXXBaseSpecifier(Cursor, Record, Idx);

    // FIXME: Make VBases lazily computed when needed to avoid storing them.
    Data.NumVBases = Record[Idx++];
    Data.VBases = new(C) CXXBaseSpecifier [Data.NumVBases];
    for (unsigned i = 0; i != Data.NumVBases; ++i)
      Data.VBases[i] = Reader.ReadCXXBaseSpecifier(Cursor, Record, Idx);

    Reader.ReadUnresolvedSet(Data.Conversions, Record, Idx);
    Reader.ReadUnresolvedSet(Data.VisibleConversions, Record, Idx);
    assert(Data.Definition && "Data.Definition should be already set!");
    Data.FirstFriend
        = cast_or_null<FriendDecl>(Reader.GetDecl(Record[Idx++]));
  }

  enum CXXRecKind {
    CXXRecNotTemplate = 0, CXXRecTemplate, CXXRecMemberSpecialization
  };
  switch ((CXXRecKind)Record[Idx++]) {
  default:
    assert(false && "Out of sync with PCHDeclWriter::VisitCXXRecordDecl?");
  case CXXRecNotTemplate:
    break;
  case CXXRecTemplate:
    D->setDescribedClassTemplate(
                        cast<ClassTemplateDecl>(Reader.GetDecl(Record[Idx++])));
    break;
  case CXXRecMemberSpecialization: {
    CXXRecordDecl *RD = cast<CXXRecordDecl>(Reader.GetDecl(Record[Idx++]));
    TemplateSpecializationKind TSK = (TemplateSpecializationKind)Record[Idx++];
    SourceLocation POI = Reader.ReadSourceLocation(Record, Idx);
    D->setInstantiationOfMemberClass(RD, TSK);
    D->getMemberSpecializationInfo()->setPointOfInstantiation(POI);
    break;
  }
  }
}

void PCHDeclReader::VisitCXXMethodDecl(CXXMethodDecl *D) {
  VisitFunctionDecl(D);
  unsigned NumOverridenMethods = Record[Idx++];
  while (NumOverridenMethods--) {
    CXXMethodDecl *MD = cast<CXXMethodDecl>(Reader.GetDecl(Record[Idx++]));
    // Avoid invariant checking of CXXMethodDecl::addOverriddenMethod,
    // MD may be initializing.
    Reader.getContext()->addOverriddenMethod(D, MD);
  }
}

void PCHDeclReader::VisitCXXConstructorDecl(CXXConstructorDecl *D) {
  VisitCXXMethodDecl(D);
  
  D->IsExplicitSpecified = Record[Idx++];
  D->ImplicitlyDefined = Record[Idx++];
  llvm::tie(D->BaseOrMemberInitializers, D->NumBaseOrMemberInitializers)
      = Reader.ReadCXXBaseOrMemberInitializers(Cursor, Record, Idx);
}

void PCHDeclReader::VisitCXXDestructorDecl(CXXDestructorDecl *D) {
  VisitCXXMethodDecl(D);

  D->ImplicitlyDefined = Record[Idx++];
  D->OperatorDelete = cast_or_null<FunctionDecl>(Reader.GetDecl(Record[Idx++]));
}

void PCHDeclReader::VisitCXXConversionDecl(CXXConversionDecl *D) {
  VisitCXXMethodDecl(D);
  D->IsExplicitSpecified = Record[Idx++];
}

void PCHDeclReader::VisitAccessSpecDecl(AccessSpecDecl *D) {
  VisitDecl(D);
  D->setColonLoc(Reader.ReadSourceLocation(Record, Idx));
}

void PCHDeclReader::VisitFriendDecl(FriendDecl *D) {
  VisitDecl(D);
  if (Record[Idx++])
    D->Friend = Reader.GetTypeSourceInfo(Cursor, Record, Idx);
  else
    D->Friend = cast<NamedDecl>(Reader.GetDecl(Record[Idx++]));
  D->NextFriend = cast_or_null<FriendDecl>(Reader.GetDecl(Record[Idx++]));
  D->FriendLoc = Reader.ReadSourceLocation(Record, Idx);
}

void PCHDeclReader::VisitFriendTemplateDecl(FriendTemplateDecl *D) {
  VisitDecl(D);
  unsigned NumParams = Record[Idx++];
  D->NumParams = NumParams;
  D->Params = new TemplateParameterList*[NumParams];
  for (unsigned i = 0; i != NumParams; ++i)
    D->Params[i] = Reader.ReadTemplateParameterList(Record, Idx);
  if (Record[Idx++]) // HasFriendDecl
    D->Friend = cast<NamedDecl>(Reader.GetDecl(Record[Idx++]));
  else
    D->Friend = Reader.GetTypeSourceInfo(Cursor, Record, Idx);
  D->FriendLoc = Reader.ReadSourceLocation(Record, Idx);
}

void PCHDeclReader::VisitTemplateDecl(TemplateDecl *D) {
  VisitNamedDecl(D);

  NamedDecl *TemplatedDecl
    = cast_or_null<NamedDecl>(Reader.GetDecl(Record[Idx++]));
  TemplateParameterList* TemplateParams
      = Reader.ReadTemplateParameterList(Record, Idx); 
  D->init(TemplatedDecl, TemplateParams);
}

void PCHDeclReader::VisitRedeclarableTemplateDecl(RedeclarableTemplateDecl *D) {
  VisitTemplateDecl(D);

  D->IdentifierNamespace = Record[Idx++];
  RedeclarableTemplateDecl *PrevDecl =
      cast_or_null<RedeclarableTemplateDecl>(Reader.GetDecl(Record[Idx++]));
  assert((PrevDecl == 0 || PrevDecl->getKind() == D->getKind()) &&
         "PrevDecl kind mismatch");
  if (PrevDecl)
    D->CommonOrPrev = PrevDecl;
  if (PrevDecl == 0) {
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
    // in the same PCH. However, if this actually needs to point to a
    // redeclaration in another chained PCH, we need to update it by checking
    // the FirstLatestDeclIDs map which tracks this kind of decls.
    assert(Reader.GetDecl(ThisDeclID) == D && "Invalid ThisDeclID ?");
    PCHReader::FirstLatestDeclIDMap::iterator I
        = Reader.FirstLatestDeclIDs.find(ThisDeclID);
    if (I != Reader.FirstLatestDeclIDs.end()) {
      Decl *NewLatest = Reader.GetDecl(I->second);
      assert((LatestDecl->getLocation().isInvalid() ||
              NewLatest->getLocation().isInvalid()  ||
              Reader.SourceMgr.isBeforeInTranslationUnit(
                                                   LatestDecl->getLocation(),
                                                   NewLatest->getLocation())) &&
             "The new latest is supposed to come after the previous latest");
      LatestDecl = cast<RedeclarableTemplateDecl>(NewLatest);
    }

    assert(LatestDecl->getKind() == D->getKind() && "Latest kind mismatch");
    D->getCommonPtr()->Latest = LatestDecl;
  }
}

void PCHDeclReader::VisitClassTemplateDecl(ClassTemplateDecl *D) {
  VisitRedeclarableTemplateDecl(D);

  if (D->getPreviousDeclaration() == 0) {
    // This ClassTemplateDecl owns a CommonPtr; read it.

    // FoldingSets are filled in VisitClassTemplateSpecializationDecl.
    unsigned size = Record[Idx++];
    while (size--)
      cast<ClassTemplateSpecializationDecl>(Reader.GetDecl(Record[Idx++]));

    size = Record[Idx++];
    while (size--)
      cast<ClassTemplatePartialSpecializationDecl>(
                                                 Reader.GetDecl(Record[Idx++]));

    // InjectedClassNameType is computed.
  }
}

void PCHDeclReader::VisitClassTemplateSpecializationDecl(
                                           ClassTemplateSpecializationDecl *D) {
  VisitCXXRecordDecl(D);

  if (Decl *InstD = Reader.GetDecl(Record[Idx++])) {
    if (ClassTemplateDecl *CTD = dyn_cast<ClassTemplateDecl>(InstD)) {
      D->setInstantiationOf(CTD);
    } else {
      llvm::SmallVector<TemplateArgument, 8> TemplArgs;
      Reader.ReadTemplateArgumentList(TemplArgs, Cursor, Record, Idx);
      D->setInstantiationOf(cast<ClassTemplatePartialSpecializationDecl>(InstD),
                            TemplArgs.data(), TemplArgs.size());
    }
  }

  // Explicit info.
  if (TypeSourceInfo *TyInfo = Reader.GetTypeSourceInfo(Cursor, Record, Idx)) {
    D->setTypeAsWritten(TyInfo);
    D->setExternLoc(Reader.ReadSourceLocation(Record, Idx));
    D->setTemplateKeywordLoc(Reader.ReadSourceLocation(Record, Idx));
  }

  llvm::SmallVector<TemplateArgument, 8> TemplArgs;
  Reader.ReadTemplateArgumentList(TemplArgs, Cursor, Record, Idx);
  D->initTemplateArgs(TemplArgs.data(), TemplArgs.size());
  SourceLocation POI = Reader.ReadSourceLocation(Record, Idx);
  if (POI.isValid())
    D->setPointOfInstantiation(POI);
  D->setSpecializationKind((TemplateSpecializationKind)Record[Idx++]);

  if (D->isCanonicalDecl()) { // It's kept in the folding set.
    ClassTemplateDecl *CanonPattern
                       = cast<ClassTemplateDecl>(Reader.GetDecl(Record[Idx++]));
    if (ClassTemplatePartialSpecializationDecl *Partial
            = dyn_cast<ClassTemplatePartialSpecializationDecl>(D)) {
      CanonPattern->getPartialSpecializations().InsertNode(Partial);
    } else {
      CanonPattern->getSpecializations().InsertNode(D);
    }
  }
}

void PCHDeclReader::VisitClassTemplatePartialSpecializationDecl(
                                    ClassTemplatePartialSpecializationDecl *D) {
  VisitClassTemplateSpecializationDecl(D);

  D->initTemplateParameters(Reader.ReadTemplateParameterList(Record, Idx));
  
  TemplateArgumentListInfo ArgInfos;
  unsigned NumArgs = Record[Idx++];
  while (NumArgs--)
    ArgInfos.addArgument(Reader.ReadTemplateArgumentLoc(Cursor, Record, Idx));
  D->initTemplateArgsAsWritten(ArgInfos);
  
  D->setSequenceNumber(Record[Idx++]);

  // These are read/set from/to the first declaration.
  if (D->getPreviousDeclaration() == 0) {
    D->setInstantiatedFromMember(
        cast_or_null<ClassTemplatePartialSpecializationDecl>(
                                                Reader.GetDecl(Record[Idx++])));
    if (Record[Idx++])
      D->setMemberSpecialization();
  }
}

void PCHDeclReader::VisitFunctionTemplateDecl(FunctionTemplateDecl *D) {
  VisitRedeclarableTemplateDecl(D);

  if (D->getPreviousDeclaration() == 0) {
    // This FunctionTemplateDecl owns a CommonPtr; read it.

    // Read the function specialization declarations.
    // FunctionTemplateDecl's FunctionTemplateSpecializationInfos are filled
    // through the specialized FunctionDecl's setFunctionTemplateSpecialization.
    unsigned NumSpecs = Record[Idx++];
    while (NumSpecs--)
      Reader.GetDecl(Record[Idx++]);
  }
}

void PCHDeclReader::VisitTemplateTypeParmDecl(TemplateTypeParmDecl *D) {
  VisitTypeDecl(D);

  D->setDeclaredWithTypename(Record[Idx++]);
  D->setParameterPack(Record[Idx++]);

  bool Inherited = Record[Idx++];
  TypeSourceInfo *DefArg = Reader.GetTypeSourceInfo(Cursor, Record, Idx);
  D->setDefaultArgument(DefArg, Inherited);
}

void PCHDeclReader::VisitNonTypeTemplateParmDecl(NonTypeTemplateParmDecl *D) {
  VisitVarDecl(D);
  // TemplateParmPosition.
  D->setDepth(Record[Idx++]);
  D->setPosition(Record[Idx++]);
  // Rest of NonTypeTemplateParmDecl.
  if (Record[Idx++]) {
    Expr *DefArg = Reader.ReadExpr(Cursor);
    bool Inherited = Record[Idx++];
    D->setDefaultArgument(DefArg, Inherited);
 }
}

void PCHDeclReader::VisitTemplateTemplateParmDecl(TemplateTemplateParmDecl *D) {
  VisitTemplateDecl(D);
  // TemplateParmPosition.
  D->setDepth(Record[Idx++]);
  D->setPosition(Record[Idx++]);
  // Rest of TemplateTemplateParmDecl.
  TemplateArgumentLoc Arg = Reader.ReadTemplateArgumentLoc(Cursor, Record, Idx);
  bool IsInherited = Record[Idx++];
  D->setDefaultArgument(Arg, IsInherited);
}

void PCHDeclReader::VisitStaticAssertDecl(StaticAssertDecl *D) {
  VisitDecl(D);
  D->AssertExpr = Reader.ReadExpr(Cursor);
  D->Message = cast<StringLiteral>(Reader.ReadExpr(Cursor));
}

std::pair<uint64_t, uint64_t>
PCHDeclReader::VisitDeclContext(DeclContext *DC) {
  uint64_t LexicalOffset = Record[Idx++];
  uint64_t VisibleOffset = Record[Idx++];
  return std::make_pair(LexicalOffset, VisibleOffset);
}

template <typename T>
void PCHDeclReader::VisitRedeclarable(Redeclarable<T> *D) {
  enum RedeclKind { NoRedeclaration = 0, PointsToPrevious, PointsToLatest };
  RedeclKind Kind = (RedeclKind)Record[Idx++];
  switch (Kind) {
  default:
    assert(0 && "Out of sync with PCHDeclWriter::VisitRedeclarable or messed up"
                " reading");
  case NoRedeclaration:
    break;
  case PointsToPrevious:
    D->RedeclLink = typename Redeclarable<T>::PreviousDeclLink(
                                cast_or_null<T>(Reader.GetDecl(Record[Idx++])));
    break;
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
  // the same PCH. However, if this actually needs to point to a redeclaration
  // in another chained PCH, we need to update it by checking the
  // FirstLatestDeclIDs map which tracks this kind of decls.
  assert(Reader.GetDecl(ThisDeclID) == static_cast<T*>(D) &&
         "Invalid ThisDeclID ?");
  PCHReader::FirstLatestDeclIDMap::iterator I
      = Reader.FirstLatestDeclIDs.find(ThisDeclID);
  if (I != Reader.FirstLatestDeclIDs.end()) {
    Decl *NewLatest = Reader.GetDecl(I->second);
    assert((D->getMostRecentDeclaration()->getLocation().isInvalid() ||
            NewLatest->getLocation().isInvalid() ||
            Reader.SourceMgr.isBeforeInTranslationUnit(
                                   D->getMostRecentDeclaration()->getLocation(),
                                   NewLatest->getLocation())) &&
           "The new latest is supposed to come after the previous latest");
    D->RedeclLink
        = typename Redeclarable<T>::LatestDeclLink(cast_or_null<T>(NewLatest));
  }
}

//===----------------------------------------------------------------------===//
// Attribute Reading
//===----------------------------------------------------------------------===//

/// \brief Reads attributes from the current stream position.
Attr *PCHReader::ReadAttributes(llvm::BitstreamCursor &DeclsCursor) {
  unsigned Code = DeclsCursor.ReadCode();
  assert(Code == llvm::bitc::UNABBREV_RECORD &&
         "Expected unabbreviated record"); (void)Code;

  RecordData Record;
  unsigned Idx = 0;
  unsigned RecCode = DeclsCursor.ReadRecord(Code, Record);
  assert(RecCode == pch::DECL_ATTR && "Expected attribute record");
  (void)RecCode;

#define SIMPLE_ATTR(Name)                       \
 case attr::Name:                               \
   New = ::new (*Context) Name##Attr();         \
   break

#define STRING_ATTR(Name)                                       \
 case attr::Name:                                               \
   New = ::new (*Context) Name##Attr(*Context, ReadString(Record, Idx));  \
   break

#define UNSIGNED_ATTR(Name)                             \
 case attr::Name:                                       \
   New = ::new (*Context) Name##Attr(Record[Idx++]);    \
   break

  Attr *Attrs = 0;
  while (Idx < Record.size()) {
    Attr *New = 0;
    attr::Kind Kind = (attr::Kind)Record[Idx++];
    bool IsInherited = Record[Idx++];

    switch (Kind) {
    default:
      assert(0 && "Unknown attribute!");
      break;
    STRING_ATTR(Alias);
    SIMPLE_ATTR(AlignMac68k);
    UNSIGNED_ATTR(Aligned);
    SIMPLE_ATTR(AlwaysInline);
    SIMPLE_ATTR(AnalyzerNoReturn);
    STRING_ATTR(Annotate);
    STRING_ATTR(AsmLabel);
    SIMPLE_ATTR(BaseCheck);

    case attr::Blocks:
      New = ::new (*Context) BlocksAttr(
                                  (BlocksAttr::BlocksAttrTypes)Record[Idx++]);
      break;

    SIMPLE_ATTR(CDecl);

    case attr::Cleanup:
      New = ::new (*Context) CleanupAttr(
                                  cast<FunctionDecl>(GetDecl(Record[Idx++])));
      break;

    SIMPLE_ATTR(Const);
    UNSIGNED_ATTR(Constructor);
    SIMPLE_ATTR(DLLExport);
    SIMPLE_ATTR(DLLImport);
    SIMPLE_ATTR(Deprecated);
    UNSIGNED_ATTR(Destructor);
    SIMPLE_ATTR(FastCall);
    SIMPLE_ATTR(Final);

    case attr::Format: {
      std::string Type = ReadString(Record, Idx);
      unsigned FormatIdx = Record[Idx++];
      unsigned FirstArg = Record[Idx++];
      New = ::new (*Context) FormatAttr(*Context, Type, FormatIdx, FirstArg);
      break;
    }

    case attr::FormatArg: {
      unsigned FormatIdx = Record[Idx++];
      New = ::new (*Context) FormatArgAttr(FormatIdx);
      break;
    }

    case attr::Sentinel: {
      int sentinel = Record[Idx++];
      int nullPos = Record[Idx++];
      New = ::new (*Context) SentinelAttr(sentinel, nullPos);
      break;
    }

    SIMPLE_ATTR(GNUInline);
    SIMPLE_ATTR(Hiding);

    case attr::IBAction:
      New = ::new (*Context) IBActionAttr();
      break;

    case attr::IBOutlet:
      New = ::new (*Context) IBOutletAttr();
      break;

    case attr::IBOutletCollection: {
      QualType QT = GetType(Record[Idx++]);
      New = ::new (*Context) IBOutletCollectionAttr(QT);
      break;
    }

    SIMPLE_ATTR(Malloc);
    SIMPLE_ATTR(NoDebug);
    SIMPLE_ATTR(NoInline);
    SIMPLE_ATTR(NoReturn);
    SIMPLE_ATTR(NoThrow);

    case attr::NonNull: {
      unsigned Size = Record[Idx++];
      llvm::SmallVector<unsigned, 16> ArgNums;
      ArgNums.insert(ArgNums.end(), &Record[Idx], &Record[Idx] + Size);
      Idx += Size;
      New = ::new (*Context) NonNullAttr(*Context, ArgNums.data(), Size);
      break;
    }

    case attr::ReqdWorkGroupSize: {
      unsigned X = Record[Idx++];
      unsigned Y = Record[Idx++];
      unsigned Z = Record[Idx++];
      New = ::new (*Context) ReqdWorkGroupSizeAttr(X, Y, Z);
      break;
    }

    SIMPLE_ATTR(ObjCException);
    SIMPLE_ATTR(ObjCNSObject);
    SIMPLE_ATTR(CFReturnsNotRetained);
    SIMPLE_ATTR(CFReturnsRetained);
    SIMPLE_ATTR(NSReturnsNotRetained);
    SIMPLE_ATTR(NSReturnsRetained);
    SIMPLE_ATTR(Overloadable);
    SIMPLE_ATTR(Override);
    SIMPLE_ATTR(Packed);
    UNSIGNED_ATTR(MaxFieldAlignment);
    SIMPLE_ATTR(Pure);
    UNSIGNED_ATTR(Regparm);
    STRING_ATTR(Section);
    SIMPLE_ATTR(StdCall);
    SIMPLE_ATTR(ThisCall);
    SIMPLE_ATTR(TransparentUnion);
    SIMPLE_ATTR(Unavailable);
    SIMPLE_ATTR(Unused);
    SIMPLE_ATTR(Used);

    case attr::Visibility:
      New = ::new (*Context) VisibilityAttr(
                              (VisibilityAttr::VisibilityTypes)Record[Idx++],
                              (bool)Record[Idx++]);
      break;

    SIMPLE_ATTR(WarnUnusedResult);
    SIMPLE_ATTR(Weak);
    SIMPLE_ATTR(WeakRef);
    SIMPLE_ATTR(WeakImport);
    }

    assert(New && "Unable to decode attribute?");
    New->setInherited(IsInherited);
    New->setNext(Attrs);
    Attrs = New;
  }
#undef UNSIGNED_ATTR
#undef STRING_ATTR
#undef SIMPLE_ATTR

  // The list of attributes was built backwards. Reverse the list
  // before returning it.
  Attr *PrevAttr = 0, *NextAttr = 0;
  while (Attrs) {
    NextAttr = Attrs->getNext();
    Attrs->setNext(PrevAttr);
    PrevAttr = Attrs;
    Attrs = NextAttr;
  }

  return PrevAttr;
}

//===----------------------------------------------------------------------===//
// PCHReader Implementation
//===----------------------------------------------------------------------===//

/// \brief Note that we have loaded the declaration with the given
/// Index.
///
/// This routine notes that this declaration has already been loaded,
/// so that future GetDecl calls will return this declaration rather
/// than trying to load a new declaration.
inline void PCHReader::LoadedDecl(unsigned Index, Decl *D) {
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
PCHReader::RecordLocation
PCHReader::DeclCursorForIndex(unsigned Index, pch::DeclID ID) {
  // See if there's an override.
  DeclReplacementMap::iterator It = ReplacedDecls.find(ID);
  if (It != ReplacedDecls.end())
    return RecordLocation(&It->second.first->DeclsCursor, It->second.second);

  PerFileData *F = 0;
  for (unsigned I = 0, N = Chain.size(); I != N; ++I) {
    F = Chain[N - I - 1];
    if (Index < F->LocalNumDecls)
      break;
    Index -= F->LocalNumDecls;
  }
  assert(F && F->LocalNumDecls > Index && "Broken chain");
  return RecordLocation(&F->DeclsCursor, F->DeclOffsets[Index]);
}

/// \brief Read the declaration at the given offset from the PCH file.
Decl *PCHReader::ReadDeclRecord(unsigned Index, pch::DeclID ID) {
  RecordLocation Loc = DeclCursorForIndex(Index, ID);
  llvm::BitstreamCursor &DeclsCursor = *Loc.first;
  // Keep track of where we are in the stream, then jump back there
  // after reading this declaration.
  SavedStreamPosition SavedPosition(DeclsCursor);

  ReadingKindTracker ReadingKind(Read_Decl, *this);

  // Note that we are loading a declaration record.
  Deserializing ADecl(this);

  DeclsCursor.JumpToBit(Loc.second);
  RecordData Record;
  unsigned Code = DeclsCursor.ReadCode();
  unsigned Idx = 0;
  PCHDeclReader Reader(*this, DeclsCursor, ID, Record, Idx);

  Decl *D = 0;
  switch ((pch::DeclCode)DeclsCursor.ReadRecord(Code, Record)) {
  case pch::DECL_ATTR:
  case pch::DECL_CONTEXT_LEXICAL:
  case pch::DECL_CONTEXT_VISIBLE:
    assert(false && "Record cannot be de-serialized with ReadDeclRecord");
    break;
  case pch::DECL_TRANSLATION_UNIT:
    assert(Index == 0 && "Translation unit must be at index 0");
    D = Context->getTranslationUnitDecl();
    break;
  case pch::DECL_TYPEDEF:
    D = TypedefDecl::Create(*Context, 0, SourceLocation(), 0, 0);
    break;
  case pch::DECL_ENUM:
    D = EnumDecl::Create(*Context, Decl::EmptyShell());
    break;
  case pch::DECL_RECORD:
    D = RecordDecl::Create(*Context, Decl::EmptyShell());
    break;
  case pch::DECL_ENUM_CONSTANT:
    D = EnumConstantDecl::Create(*Context, 0, SourceLocation(), 0, QualType(),
                                 0, llvm::APSInt());
    break;
  case pch::DECL_FUNCTION:
    D = FunctionDecl::Create(*Context, 0, SourceLocation(), DeclarationName(),
                             QualType(), 0);
    break;
  case pch::DECL_LINKAGE_SPEC:
    D = LinkageSpecDecl::Create(*Context, 0, SourceLocation(),
                                (LinkageSpecDecl::LanguageIDs)0,
                                false);
    break;
  case pch::DECL_NAMESPACE:
    D = NamespaceDecl::Create(*Context, 0, SourceLocation(), 0);
    break;
  case pch::DECL_NAMESPACE_ALIAS:
    D = NamespaceAliasDecl::Create(*Context, 0, SourceLocation(),
                                   SourceLocation(), 0, SourceRange(), 0,
                                   SourceLocation(), 0);
    break;
  case pch::DECL_USING:
    D = UsingDecl::Create(*Context, 0, SourceRange(), SourceLocation(),
                          0, DeclarationNameInfo(), false);
    break;
  case pch::DECL_USING_SHADOW:
    D = UsingShadowDecl::Create(*Context, 0, SourceLocation(), 0, 0);
    break;
  case pch::DECL_USING_DIRECTIVE:
    D = UsingDirectiveDecl::Create(*Context, 0, SourceLocation(),
                                   SourceLocation(), SourceRange(), 0,
                                   SourceLocation(), 0, 0);
    break;
  case pch::DECL_UNRESOLVED_USING_VALUE:
    D = UnresolvedUsingValueDecl::Create(*Context, 0, SourceLocation(),
                                         SourceRange(), 0,
                                         DeclarationNameInfo());
    break;
  case pch::DECL_UNRESOLVED_USING_TYPENAME:
    D = UnresolvedUsingTypenameDecl::Create(*Context, 0, SourceLocation(),
                                            SourceLocation(), SourceRange(),
                                            0, SourceLocation(),
                                            DeclarationName());
    break;
  case pch::DECL_CXX_RECORD:
    D = CXXRecordDecl::Create(*Context, Decl::EmptyShell());
    break;
  case pch::DECL_CXX_METHOD:
    D = CXXMethodDecl::Create(*Context, 0, DeclarationNameInfo(),
                              QualType(), 0);
    break;
  case pch::DECL_CXX_CONSTRUCTOR:
    D = CXXConstructorDecl::Create(*Context, Decl::EmptyShell());
    break;
  case pch::DECL_CXX_DESTRUCTOR:
    D = CXXDestructorDecl::Create(*Context, Decl::EmptyShell());
    break;
  case pch::DECL_CXX_CONVERSION:
    D = CXXConversionDecl::Create(*Context, Decl::EmptyShell());
    break;
  case pch::DECL_ACCESS_SPEC:
    D = AccessSpecDecl::Create(*Context, AS_none, 0, SourceLocation(),
                               SourceLocation());
    break;
  case pch::DECL_FRIEND:
    D = FriendDecl::Create(*Context, Decl::EmptyShell());
    break;
  case pch::DECL_FRIEND_TEMPLATE:
    D = FriendTemplateDecl::Create(*Context, Decl::EmptyShell());
    break;
  case pch::DECL_CLASS_TEMPLATE:
    D = ClassTemplateDecl::Create(*Context, 0, SourceLocation(),
                                  DeclarationName(), 0, 0, 0);
    break;
  case pch::DECL_CLASS_TEMPLATE_SPECIALIZATION:
    D = ClassTemplateSpecializationDecl::Create(*Context, Decl::EmptyShell());
    break;
  case pch::DECL_CLASS_TEMPLATE_PARTIAL_SPECIALIZATION:
    D = ClassTemplatePartialSpecializationDecl::Create(*Context,
                                                            Decl::EmptyShell());
    break;
  case pch::DECL_FUNCTION_TEMPLATE:
    D = FunctionTemplateDecl::Create(*Context, 0, SourceLocation(),
                                     DeclarationName(), 0, 0);
    break;
  case pch::DECL_TEMPLATE_TYPE_PARM:
    D = TemplateTypeParmDecl::Create(*Context, Decl::EmptyShell());
    break;
  case pch::DECL_NON_TYPE_TEMPLATE_PARM:
    D = NonTypeTemplateParmDecl::Create(*Context, 0, SourceLocation(), 0,0,0,
                                        QualType(),0);
    break;
  case pch::DECL_TEMPLATE_TEMPLATE_PARM:
    D = TemplateTemplateParmDecl::Create(*Context, 0, SourceLocation(),0,0,0,0);
    break;
  case pch::DECL_STATIC_ASSERT:
    D = StaticAssertDecl::Create(*Context, 0, SourceLocation(), 0, 0);
    break;

  case pch::DECL_OBJC_METHOD:
    D = ObjCMethodDecl::Create(*Context, SourceLocation(), SourceLocation(),
                               Selector(), QualType(), 0, 0);
    break;
  case pch::DECL_OBJC_INTERFACE:
    D = ObjCInterfaceDecl::Create(*Context, 0, SourceLocation(), 0);
    break;
  case pch::DECL_OBJC_IVAR:
    D = ObjCIvarDecl::Create(*Context, 0, SourceLocation(), 0, QualType(), 0,
                             ObjCIvarDecl::None);
    break;
  case pch::DECL_OBJC_PROTOCOL:
    D = ObjCProtocolDecl::Create(*Context, 0, SourceLocation(), 0);
    break;
  case pch::DECL_OBJC_AT_DEFS_FIELD:
    D = ObjCAtDefsFieldDecl::Create(*Context, 0, SourceLocation(), 0,
                                    QualType(), 0);
    break;
  case pch::DECL_OBJC_CLASS:
    D = ObjCClassDecl::Create(*Context, 0, SourceLocation());
    break;
  case pch::DECL_OBJC_FORWARD_PROTOCOL:
    D = ObjCForwardProtocolDecl::Create(*Context, 0, SourceLocation());
    break;
  case pch::DECL_OBJC_CATEGORY:
    D = ObjCCategoryDecl::Create(*Context, 0, SourceLocation(), 
                                 SourceLocation(), SourceLocation(), 0);
    break;
  case pch::DECL_OBJC_CATEGORY_IMPL:
    D = ObjCCategoryImplDecl::Create(*Context, 0, SourceLocation(), 0, 0);
    break;
  case pch::DECL_OBJC_IMPLEMENTATION:
    D = ObjCImplementationDecl::Create(*Context, 0, SourceLocation(), 0, 0);
    break;
  case pch::DECL_OBJC_COMPATIBLE_ALIAS:
    D = ObjCCompatibleAliasDecl::Create(*Context, 0, SourceLocation(), 0, 0);
    break;
  case pch::DECL_OBJC_PROPERTY:
    D = ObjCPropertyDecl::Create(*Context, 0, SourceLocation(), 0, SourceLocation(),
                                 0);
    break;
  case pch::DECL_OBJC_PROPERTY_IMPL:
    D = ObjCPropertyImplDecl::Create(*Context, 0, SourceLocation(),
                                     SourceLocation(), 0,
                                     ObjCPropertyImplDecl::Dynamic, 0);
    break;
  case pch::DECL_FIELD:
    D = FieldDecl::Create(*Context, 0, SourceLocation(), 0, QualType(), 0, 0,
                          false);
    break;
  case pch::DECL_VAR:
    D = VarDecl::Create(*Context, 0, SourceLocation(), 0, QualType(), 0,
                        VarDecl::None, VarDecl::None);
    break;

  case pch::DECL_IMPLICIT_PARAM:
    D = ImplicitParamDecl::Create(*Context, 0, SourceLocation(), 0, QualType());
    break;

  case pch::DECL_PARM_VAR:
    D = ParmVarDecl::Create(*Context, 0, SourceLocation(), 0, QualType(), 0,
                            VarDecl::None, VarDecl::None, 0);
    break;
  case pch::DECL_FILE_SCOPE_ASM:
    D = FileScopeAsmDecl::Create(*Context, 0, SourceLocation(), 0);
    break;
  case pch::DECL_BLOCK:
    D = BlockDecl::Create(*Context, 0, SourceLocation());
    break;
  }

  assert(D && "Unknown declaration reading PCH file");
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
      // Reading the TU will happen after reading its update blocks, so we need
      // to make sure we insert in front. For all other contexts, the vector
      // is empty here anyway, so there's no loss in efficiency.
      Infos.insert(Infos.begin(), Info);
    }
  }
  assert(Idx == Record.size());

  // If we have deserialized a declaration that has a definition the
  // AST consumer might need to know about, queue it.
  // We don't pass it to the consumer immediately because we may be in recursive
  // loading, and some declarations may still be initializing.
  if (isConsumerInterestedIn(D))
    InterestingDecls.push_back(D);

  return D;
}

bool PCHReader::ReadDeclContextStorage(llvm::BitstreamCursor &Cursor,
                                   const std::pair<uint64_t, uint64_t> &Offsets,
                                       DeclContextInfo &Info) {
  SavedStreamPosition SavedPosition(Cursor);
  // First the lexical decls.
  if (Offsets.first != 0) {
    Cursor.JumpToBit(Offsets.first);

    RecordData Record;
    const char *Blob;
    unsigned BlobLen;
    unsigned Code = Cursor.ReadCode();
    unsigned RecCode = Cursor.ReadRecord(Code, Record, &Blob, &BlobLen);
    if (RecCode != pch::DECL_CONTEXT_LEXICAL) {
      Error("Expected lexical block");
      return true;
    }

    Info.LexicalDecls = reinterpret_cast<const pch::DeclID*>(Blob);
    Info.NumLexicalDecls = BlobLen / sizeof(pch::DeclID);
  } else {
    Info.LexicalDecls = 0;
    Info.NumLexicalDecls = 0;
  }

  // Now the visible decls.
  Info.Stream = &Cursor;
  Info.OffsetToVisibleDecls = Offsets.second;

  return false;
}
