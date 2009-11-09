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

#include "clang/Frontend/PCHReader.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/DeclGroup.h"
#include "clang/AST/Expr.h"
using namespace clang;


//===----------------------------------------------------------------------===//
// Declaration deserialization
//===----------------------------------------------------------------------===//

namespace {
  class PCHDeclReader : public DeclVisitor<PCHDeclReader, void> {
    PCHReader &Reader;
    const PCHReader::RecordData &Record;
    unsigned &Idx;

  public:
    PCHDeclReader(PCHReader &Reader, const PCHReader::RecordData &Record,
                  unsigned &Idx)
      : Reader(Reader), Record(Record), Idx(Idx) { }

    void VisitDecl(Decl *D);
    void VisitTranslationUnitDecl(TranslationUnitDecl *TU);
    void VisitNamedDecl(NamedDecl *ND);
    void VisitTypeDecl(TypeDecl *TD);
    void VisitTypedefDecl(TypedefDecl *TD);
    void VisitTagDecl(TagDecl *TD);
    void VisitEnumDecl(EnumDecl *ED);
    void VisitRecordDecl(RecordDecl *RD);
    void VisitValueDecl(ValueDecl *VD);
    void VisitEnumConstantDecl(EnumConstantDecl *ECD);
    void VisitDeclaratorDecl(DeclaratorDecl *DD);
    void VisitFunctionDecl(FunctionDecl *FD);
    void VisitFieldDecl(FieldDecl *FD);
    void VisitVarDecl(VarDecl *VD);
    void VisitImplicitParamDecl(ImplicitParamDecl *PD);
    void VisitParmVarDecl(ParmVarDecl *PD);
    void VisitFileScopeAsmDecl(FileScopeAsmDecl *AD);
    void VisitBlockDecl(BlockDecl *BD);
    std::pair<uint64_t, uint64_t> VisitDeclContext(DeclContext *DC);
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

void PCHDeclReader::VisitDecl(Decl *D) {
  D->setDeclContext(cast_or_null<DeclContext>(Reader.GetDecl(Record[Idx++])));
  D->setLexicalDeclContext(
                     cast_or_null<DeclContext>(Reader.GetDecl(Record[Idx++])));
  D->setLocation(SourceLocation::getFromRawEncoding(Record[Idx++]));
  D->setInvalidDecl(Record[Idx++]);
  if (Record[Idx++])
    D->addAttr(Reader.ReadAttributes());
  D->setImplicit(Record[Idx++]);
  D->setUsed(Record[Idx++]);
  D->setAccess((AccessSpecifier)Record[Idx++]);
  D->setPCHLevel(Record[Idx++] + 1);
}

void PCHDeclReader::VisitTranslationUnitDecl(TranslationUnitDecl *TU) {
  VisitDecl(TU);
}

void PCHDeclReader::VisitNamedDecl(NamedDecl *ND) {
  VisitDecl(ND);
  ND->setDeclName(Reader.ReadDeclarationName(Record, Idx));
}

void PCHDeclReader::VisitTypeDecl(TypeDecl *TD) {
  VisitNamedDecl(TD);
  TD->setTypeForDecl(Reader.GetType(Record[Idx++]).getTypePtr());
}

void PCHDeclReader::VisitTypedefDecl(TypedefDecl *TD) {
  // Note that we cannot use VisitTypeDecl here, because we need to
  // set the underlying type of the typedef *before* we try to read
  // the type associated with the TypedefDecl.
  VisitNamedDecl(TD);
  uint64_t TypeData = Record[Idx++];
  TD->setTypeDeclaratorInfo(Reader.GetDeclaratorInfo(Record, Idx));
  TD->setTypeForDecl(Reader.GetType(TypeData).getTypePtr());
}

void PCHDeclReader::VisitTagDecl(TagDecl *TD) {
  VisitTypeDecl(TD);
  TD->setPreviousDeclaration(
                        cast_or_null<TagDecl>(Reader.GetDecl(Record[Idx++])));
  TD->setTagKind((TagDecl::TagKind)Record[Idx++]);
  TD->setDefinition(Record[Idx++]);
  TD->setTypedefForAnonDecl(
                    cast_or_null<TypedefDecl>(Reader.GetDecl(Record[Idx++])));
  TD->setRBraceLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  TD->setTagKeywordLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
}

void PCHDeclReader::VisitEnumDecl(EnumDecl *ED) {
  VisitTagDecl(ED);
  ED->setIntegerType(Reader.GetType(Record[Idx++]));
  // FIXME: C++ InstantiatedFrom
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
    ECD->setInitExpr(Reader.ReadDeclExpr());
  ECD->setInitVal(Reader.ReadAPSInt(Record, Idx));
}

void PCHDeclReader::VisitDeclaratorDecl(DeclaratorDecl *DD) {
  VisitValueDecl(DD);
  DeclaratorInfo *DInfo = Reader.GetDeclaratorInfo(Record, Idx);
  if (DInfo)
    DD->setDeclaratorInfo(DInfo);
}

void PCHDeclReader::VisitFunctionDecl(FunctionDecl *FD) {
  VisitDeclaratorDecl(FD);
  if (Record[Idx++])
    FD->setLazyBody(Reader.getDeclsCursor().GetCurrentBitNo());
  FD->setPreviousDeclaration(
                   cast_or_null<FunctionDecl>(Reader.GetDecl(Record[Idx++])));
  FD->setStorageClass((FunctionDecl::StorageClass)Record[Idx++]);
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
  // FIXME: C++ TemplateOrInstantiation
  unsigned NumParams = Record[Idx++];
  llvm::SmallVector<ParmVarDecl *, 16> Params;
  Params.reserve(NumParams);
  for (unsigned I = 0; I != NumParams; ++I)
    Params.push_back(cast<ParmVarDecl>(Reader.GetDecl(Record[Idx++])));
  FD->setParams(*Reader.getContext(), Params.data(), NumParams);
}

void PCHDeclReader::VisitObjCMethodDecl(ObjCMethodDecl *MD) {
  VisitNamedDecl(MD);
  if (Record[Idx++]) {
    // In practice, this won't be executed (since method definitions
    // don't occur in header files).
    MD->setBody(Reader.ReadDeclStmt());
    MD->setSelfDecl(cast<ImplicitParamDecl>(Reader.GetDecl(Record[Idx++])));
    MD->setCmdDecl(cast<ImplicitParamDecl>(Reader.GetDecl(Record[Idx++])));
  }
  MD->setInstanceMethod(Record[Idx++]);
  MD->setVariadic(Record[Idx++]);
  MD->setSynthesized(Record[Idx++]);
  MD->setDeclImplementation((ObjCMethodDecl::ImplementationControl)Record[Idx++]);
  MD->setObjCDeclQualifier((Decl::ObjCDeclQualifier)Record[Idx++]);
  MD->setResultType(Reader.GetType(Record[Idx++]));
  MD->setEndLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  unsigned NumParams = Record[Idx++];
  llvm::SmallVector<ParmVarDecl *, 16> Params;
  Params.reserve(NumParams);
  for (unsigned I = 0; I != NumParams; ++I)
    Params.push_back(cast<ParmVarDecl>(Reader.GetDecl(Record[Idx++])));
  MD->setMethodParams(*Reader.getContext(), Params.data(), NumParams);
}

void PCHDeclReader::VisitObjCContainerDecl(ObjCContainerDecl *CD) {
  VisitNamedDecl(CD);
  CD->setAtEndLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
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
  ID->setProtocolList(Protocols.data(), NumProtocols, *Reader.getContext());
  unsigned NumIvars = Record[Idx++];
  llvm::SmallVector<ObjCIvarDecl *, 16> IVars;
  IVars.reserve(NumIvars);
  for (unsigned I = 0; I != NumIvars; ++I)
    IVars.push_back(cast<ObjCIvarDecl>(Reader.GetDecl(Record[Idx++])));
  ID->setIVarList(IVars.data(), NumIvars, *Reader.getContext());
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
  PD->setProtocolList(ProtoRefs.data(), NumProtoRefs, *Reader.getContext());
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
  CD->setClassList(*Reader.getContext(), ClassRefs.data(), NumClassRefs);
}

void PCHDeclReader::VisitObjCForwardProtocolDecl(ObjCForwardProtocolDecl *FPD) {
  VisitDecl(FPD);
  unsigned NumProtoRefs = Record[Idx++];
  llvm::SmallVector<ObjCProtocolDecl *, 16> ProtoRefs;
  ProtoRefs.reserve(NumProtoRefs);
  for (unsigned I = 0; I != NumProtoRefs; ++I)
    ProtoRefs.push_back(cast<ObjCProtocolDecl>(Reader.GetDecl(Record[Idx++])));
  FPD->setProtocolList(ProtoRefs.data(), NumProtoRefs, *Reader.getContext());
}

void PCHDeclReader::VisitObjCCategoryDecl(ObjCCategoryDecl *CD) {
  VisitObjCContainerDecl(CD);
  CD->setClassInterface(cast<ObjCInterfaceDecl>(Reader.GetDecl(Record[Idx++])));
  unsigned NumProtoRefs = Record[Idx++];
  llvm::SmallVector<ObjCProtocolDecl *, 16> ProtoRefs;
  ProtoRefs.reserve(NumProtoRefs);
  for (unsigned I = 0; I != NumProtoRefs; ++I)
    ProtoRefs.push_back(cast<ObjCProtocolDecl>(Reader.GetDecl(Record[Idx++])));
  CD->setProtocolList(ProtoRefs.data(), NumProtoRefs, *Reader.getContext());
  CD->setNextClassCategory(cast_or_null<ObjCCategoryDecl>(Reader.GetDecl(Record[Idx++])));
  CD->setLocEnd(SourceLocation::getFromRawEncoding(Record[Idx++]));
}

void PCHDeclReader::VisitObjCCompatibleAliasDecl(ObjCCompatibleAliasDecl *CAD) {
  VisitNamedDecl(CAD);
  CAD->setClassInterface(cast<ObjCInterfaceDecl>(Reader.GetDecl(Record[Idx++])));
}

void PCHDeclReader::VisitObjCPropertyDecl(ObjCPropertyDecl *D) {
  VisitNamedDecl(D);
  D->setType(Reader.GetType(Record[Idx++]));
  // FIXME: stable encoding
  D->setPropertyAttributes(
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
}


void PCHDeclReader::VisitObjCPropertyImplDecl(ObjCPropertyImplDecl *D) {
  VisitDecl(D);
  D->setAtLoc(SourceLocation::getFromRawEncoding(Record[Idx++]));
  D->setPropertyDecl(
               cast_or_null<ObjCPropertyDecl>(Reader.GetDecl(Record[Idx++])));
  D->setPropertyIvarDecl(
                   cast_or_null<ObjCIvarDecl>(Reader.GetDecl(Record[Idx++])));
}

void PCHDeclReader::VisitFieldDecl(FieldDecl *FD) {
  VisitDeclaratorDecl(FD);
  FD->setMutable(Record[Idx++]);
  if (Record[Idx++])
    FD->setBitWidth(Reader.ReadDeclExpr());
}

void PCHDeclReader::VisitVarDecl(VarDecl *VD) {
  VisitDeclaratorDecl(VD);
  VD->setStorageClass((VarDecl::StorageClass)Record[Idx++]);
  VD->setThreadSpecified(Record[Idx++]);
  VD->setCXXDirectInitializer(Record[Idx++]);
  VD->setDeclaredInCondition(Record[Idx++]);
  VD->setPreviousDeclaration(
                         cast_or_null<VarDecl>(Reader.GetDecl(Record[Idx++])));
  if (Record[Idx++])
    VD->setInit(*Reader.getContext(), Reader.ReadDeclExpr());
}

void PCHDeclReader::VisitImplicitParamDecl(ImplicitParamDecl *PD) {
  VisitVarDecl(PD);
}

void PCHDeclReader::VisitParmVarDecl(ParmVarDecl *PD) {
  VisitVarDecl(PD);
  PD->setObjCDeclQualifier((Decl::ObjCDeclQualifier)Record[Idx++]);
}

void PCHDeclReader::VisitFileScopeAsmDecl(FileScopeAsmDecl *AD) {
  VisitDecl(AD);
  AD->setAsmString(cast<StringLiteral>(Reader.ReadDeclExpr()));
}

void PCHDeclReader::VisitBlockDecl(BlockDecl *BD) {
  VisitDecl(BD);
  BD->setBody(cast_or_null<CompoundStmt>(Reader.ReadDeclStmt()));
  unsigned NumParams = Record[Idx++];
  llvm::SmallVector<ParmVarDecl *, 16> Params;
  Params.reserve(NumParams);
  for (unsigned I = 0; I != NumParams; ++I)
    Params.push_back(cast<ParmVarDecl>(Reader.GetDecl(Record[Idx++])));
  BD->setParams(*Reader.getContext(), Params.data(), NumParams);
}

std::pair<uint64_t, uint64_t>
PCHDeclReader::VisitDeclContext(DeclContext *DC) {
  uint64_t LexicalOffset = Record[Idx++];
  uint64_t VisibleOffset = Record[Idx++];
  return std::make_pair(LexicalOffset, VisibleOffset);
}

//===----------------------------------------------------------------------===//
// Attribute Reading
//===----------------------------------------------------------------------===//

/// \brief Reads attributes from the current stream position.
Attr *PCHReader::ReadAttributes() {
  unsigned Code = DeclsCursor.ReadCode();
  assert(Code == llvm::bitc::UNABBREV_RECORD &&
         "Expected unabbreviated record"); (void)Code;

  RecordData Record;
  unsigned Idx = 0;
  unsigned RecCode = DeclsCursor.ReadRecord(Code, Record);
  assert(RecCode == pch::DECL_ATTR && "Expected attribute record");
  (void)RecCode;

#define SIMPLE_ATTR(Name)                       \
 case Attr::Name:                               \
   New = ::new (*Context) Name##Attr();         \
   break

#define STRING_ATTR(Name)                                       \
 case Attr::Name:                                               \
   New = ::new (*Context) Name##Attr(ReadString(Record, Idx));  \
   break

#define UNSIGNED_ATTR(Name)                             \
 case Attr::Name:                                       \
   New = ::new (*Context) Name##Attr(Record[Idx++]);    \
   break

  Attr *Attrs = 0;
  while (Idx < Record.size()) {
    Attr *New = 0;
    Attr::Kind Kind = (Attr::Kind)Record[Idx++];
    bool IsInherited = Record[Idx++];

    switch (Kind) {
    STRING_ATTR(Alias);
    UNSIGNED_ATTR(Aligned);
    SIMPLE_ATTR(AlwaysInline);
    SIMPLE_ATTR(AnalyzerNoReturn);
    STRING_ATTR(Annotate);
    STRING_ATTR(AsmLabel);

    case Attr::Blocks:
      New = ::new (*Context) BlocksAttr(
                                  (BlocksAttr::BlocksAttrTypes)Record[Idx++]);
      break;

    SIMPLE_ATTR(CDecl);

    case Attr::Cleanup:
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

    case Attr::Format: {
      std::string Type = ReadString(Record, Idx);
      unsigned FormatIdx = Record[Idx++];
      unsigned FirstArg = Record[Idx++];
      New = ::new (*Context) FormatAttr(Type, FormatIdx, FirstArg);
      break;
    }

    case Attr::FormatArg: {
      unsigned FormatIdx = Record[Idx++];
      New = ::new (*Context) FormatArgAttr(FormatIdx);
      break;
    }

    case Attr::Sentinel: {
      int sentinel = Record[Idx++];
      int nullPos = Record[Idx++];
      New = ::new (*Context) SentinelAttr(sentinel, nullPos);
      break;
    }

    SIMPLE_ATTR(GNUInline);

    case Attr::IBOutletKind:
      New = ::new (*Context) IBOutletAttr();
      break;

    SIMPLE_ATTR(Malloc);
    SIMPLE_ATTR(NoDebug);
    SIMPLE_ATTR(NoInline);
    SIMPLE_ATTR(NoReturn);
    SIMPLE_ATTR(NoThrow);

    case Attr::NonNull: {
      unsigned Size = Record[Idx++];
      llvm::SmallVector<unsigned, 16> ArgNums;
      ArgNums.insert(ArgNums.end(), &Record[Idx], &Record[Idx] + Size);
      Idx += Size;
      New = ::new (*Context) NonNullAttr(ArgNums.data(), Size);
      break;
    }

    case Attr::ReqdWorkGroupSize: {
      unsigned X = Record[Idx++];
      unsigned Y = Record[Idx++];
      unsigned Z = Record[Idx++];
      New = ::new (*Context) ReqdWorkGroupSizeAttr(X, Y, Z);
      break;
    }

    SIMPLE_ATTR(ObjCException);
    SIMPLE_ATTR(ObjCNSObject);
    SIMPLE_ATTR(CFReturnsRetained);
    SIMPLE_ATTR(NSReturnsRetained);
    SIMPLE_ATTR(Overloadable);
    SIMPLE_ATTR(Packed);
    UNSIGNED_ATTR(PragmaPack);
    SIMPLE_ATTR(Pure);
    UNSIGNED_ATTR(Regparm);
    STRING_ATTR(Section);
    SIMPLE_ATTR(StdCall);
    SIMPLE_ATTR(TransparentUnion);
    SIMPLE_ATTR(Unavailable);
    SIMPLE_ATTR(Unused);
    SIMPLE_ATTR(Used);

    case Attr::Visibility:
      New = ::new (*Context) VisibilityAttr(
                              (VisibilityAttr::VisibilityTypes)Record[Idx++]);
      break;

    SIMPLE_ATTR(WarnUnusedResult);
    SIMPLE_ATTR(Weak);
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
    return Var->isFileVarDecl() && Var->getInit();
  if (FunctionDecl *Func = dyn_cast<FunctionDecl>(D))
    return Func->isThisDeclarationADefinition();
  return isa<ObjCProtocolDecl>(D);
}

/// \brief Read the declaration at the given offset from the PCH file.
Decl *PCHReader::ReadDeclRecord(uint64_t Offset, unsigned Index) {
  // Keep track of where we are in the stream, then jump back there
  // after reading this declaration.
  SavedStreamPosition SavedPosition(DeclsCursor);

  // Note that we are loading a declaration record.
  LoadingTypeOrDecl Loading(*this);

  DeclsCursor.JumpToBit(Offset);
  RecordData Record;
  unsigned Code = DeclsCursor.ReadCode();
  unsigned Idx = 0;
  PCHDeclReader Reader(*this, Record, Idx);

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
    D = EnumDecl::Create(*Context, 0, SourceLocation(), 0, SourceLocation(), 0);
    break;
  case pch::DECL_RECORD:
    D = RecordDecl::Create(*Context, TagDecl::TK_struct, 0, SourceLocation(),
                           0, SourceLocation(), 0);
    break;
  case pch::DECL_ENUM_CONSTANT:
    D = EnumConstantDecl::Create(*Context, 0, SourceLocation(), 0, QualType(),
                                 0, llvm::APSInt());
    break;
  case pch::DECL_FUNCTION:
    D = FunctionDecl::Create(*Context, 0, SourceLocation(), DeclarationName(),
                             QualType(), 0);
    break;
  case pch::DECL_OBJC_METHOD:
    D = ObjCMethodDecl::Create(*Context, SourceLocation(), SourceLocation(),
                               Selector(), QualType(), 0);
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
    D = ObjCCategoryDecl::Create(*Context, 0, SourceLocation(), 0);
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
    D = ObjCPropertyDecl::Create(*Context, 0, SourceLocation(), 0, QualType());
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
                        VarDecl::None);
    break;

  case pch::DECL_IMPLICIT_PARAM:
    D = ImplicitParamDecl::Create(*Context, 0, SourceLocation(), 0, QualType());
    break;

  case pch::DECL_PARM_VAR:
    D = ParmVarDecl::Create(*Context, 0, SourceLocation(), 0, QualType(), 0,
                            VarDecl::None, 0);
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
      DeclContextOffsets[DC] = Offsets;
    }
  }
  assert(Idx == Record.size());

  // If we have deserialized a declaration that has a definition the
  // AST consumer might need to know about, notify the consumer
  // about that definition now or queue it for later.
  if (isConsumerInterestedIn(D)) {
    if (Consumer) {
      DeclGroupRef DG(D);
      Consumer->HandleTopLevelDecl(DG);
    } else {
      InterestingDecls.push_back(D);
    }
  }

  return D;
}
