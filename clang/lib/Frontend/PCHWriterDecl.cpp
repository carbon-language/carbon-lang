//===--- PCHWriterDecl.cpp - Declaration Serialization --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements serialization for Declarations.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/PCHWriter.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/Expr.h"
#include "clang/AST/TypeLocVisitor.h"
#include "llvm/Bitcode/BitstreamWriter.h"
#include <cstdio>

using namespace clang;

//===----------------------------------------------------------------------===//
// Declaration serialization
//===----------------------------------------------------------------------===//

namespace {
  class PCHDeclWriter : public DeclVisitor<PCHDeclWriter, void> {

    PCHWriter &Writer;
    ASTContext &Context;
    PCHWriter::RecordData &Record;

  public:
    pch::DeclCode Code;
    unsigned AbbrevToUse;

    PCHDeclWriter(PCHWriter &Writer, ASTContext &Context,
                  PCHWriter::RecordData &Record)
      : Writer(Writer), Context(Context), Record(Record) {
    }

    void VisitDecl(Decl *D);
    void VisitTranslationUnitDecl(TranslationUnitDecl *D);
    void VisitNamedDecl(NamedDecl *D);
    void VisitTypeDecl(TypeDecl *D);
    void VisitTypedefDecl(TypedefDecl *D);
    void VisitTagDecl(TagDecl *D);
    void VisitEnumDecl(EnumDecl *D);
    void VisitRecordDecl(RecordDecl *D);
    void VisitValueDecl(ValueDecl *D);
    void VisitEnumConstantDecl(EnumConstantDecl *D);
    void VisitDeclaratorDecl(DeclaratorDecl *D);
    void VisitFunctionDecl(FunctionDecl *D);
    void VisitFieldDecl(FieldDecl *D);
    void VisitVarDecl(VarDecl *D);
    void VisitImplicitParamDecl(ImplicitParamDecl *D);
    void VisitParmVarDecl(ParmVarDecl *D);
    void VisitOriginalParmVarDecl(OriginalParmVarDecl *D);
    void VisitFileScopeAsmDecl(FileScopeAsmDecl *D);
    void VisitBlockDecl(BlockDecl *D);
    void VisitDeclContext(DeclContext *DC, uint64_t LexicalOffset,
                          uint64_t VisibleOffset);
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

void PCHDeclWriter::VisitDecl(Decl *D) {
  Writer.AddDeclRef(cast_or_null<Decl>(D->getDeclContext()), Record);
  Writer.AddDeclRef(cast_or_null<Decl>(D->getLexicalDeclContext()), Record);
  Writer.AddSourceLocation(D->getLocation(), Record);
  Record.push_back(D->isInvalidDecl());
  Record.push_back(D->hasAttrs());
  Record.push_back(D->isImplicit());
  Record.push_back(D->isUsed());
  Record.push_back(D->getAccess());
  Record.push_back(D->getPCHLevel());
}

void PCHDeclWriter::VisitTranslationUnitDecl(TranslationUnitDecl *D) {
  VisitDecl(D);
  Code = pch::DECL_TRANSLATION_UNIT;
}

void PCHDeclWriter::VisitNamedDecl(NamedDecl *D) {
  VisitDecl(D);
  Writer.AddDeclarationName(D->getDeclName(), Record);
}

void PCHDeclWriter::VisitTypeDecl(TypeDecl *D) {
  VisitNamedDecl(D);
  Writer.AddTypeRef(QualType(D->getTypeForDecl(), 0), Record);
}

void PCHDeclWriter::VisitTypedefDecl(TypedefDecl *D) {
  VisitTypeDecl(D);
  Writer.AddTypeRef(D->getUnderlyingType(), Record);
  Code = pch::DECL_TYPEDEF;
}

void PCHDeclWriter::VisitTagDecl(TagDecl *D) {
  VisitTypeDecl(D);
  Writer.AddDeclRef(D->getPreviousDeclaration(), Record);
  Record.push_back((unsigned)D->getTagKind()); // FIXME: stable encoding
  Record.push_back(D->isDefinition());
  Writer.AddDeclRef(D->getTypedefForAnonDecl(), Record);
  Writer.AddSourceLocation(D->getRBraceLoc(), Record);
  Writer.AddSourceLocation(D->getTagKeywordLoc(), Record);
}

void PCHDeclWriter::VisitEnumDecl(EnumDecl *D) {
  VisitTagDecl(D);
  Writer.AddTypeRef(D->getIntegerType(), Record);
  // FIXME: C++ InstantiatedFrom
  Code = pch::DECL_ENUM;
}

void PCHDeclWriter::VisitRecordDecl(RecordDecl *D) {
  VisitTagDecl(D);
  Record.push_back(D->hasFlexibleArrayMember());
  Record.push_back(D->isAnonymousStructOrUnion());
  Record.push_back(D->hasObjectMember());
  Code = pch::DECL_RECORD;
}

void PCHDeclWriter::VisitValueDecl(ValueDecl *D) {
  VisitNamedDecl(D);
  Writer.AddTypeRef(D->getType(), Record);
}

void PCHDeclWriter::VisitEnumConstantDecl(EnumConstantDecl *D) {
  VisitValueDecl(D);
  Record.push_back(D->getInitExpr()? 1 : 0);
  if (D->getInitExpr())
    Writer.AddStmt(D->getInitExpr());
  Writer.AddAPSInt(D->getInitVal(), Record);
  Code = pch::DECL_ENUM_CONSTANT;
}
namespace {

class TypeLocWriter : public TypeLocVisitor<TypeLocWriter> {
  PCHWriter &Writer;
  PCHWriter::RecordData &Record;

public:
  TypeLocWriter(PCHWriter &Writer, PCHWriter::RecordData &Record)
    : Writer(Writer), Record(Record) { }

#define ABSTRACT_TYPELOC(CLASS)
#define TYPELOC(CLASS, PARENT) \
    void Visit##CLASS(CLASS TyLoc);
#include "clang/AST/TypeLocNodes.def"

  void VisitTypeLoc(TypeLoc TyLoc) {
    assert(0 && "A type loc wrapper was not handled!");
  }
};

}

void TypeLocWriter::VisitQualifiedLoc(QualifiedLoc TyLoc) {
  // nothing to do here
}
void TypeLocWriter::VisitDefaultTypeSpecLoc(DefaultTypeSpecLoc TyLoc) {
  Writer.AddSourceLocation(TyLoc.getStartLoc(), Record);
}
void TypeLocWriter::VisitTypedefLoc(TypedefLoc TyLoc) {
  Writer.AddSourceLocation(TyLoc.getNameLoc(), Record);
}
void TypeLocWriter::VisitObjCInterfaceLoc(ObjCInterfaceLoc TyLoc) {
  Writer.AddSourceLocation(TyLoc.getNameLoc(), Record);
}
void TypeLocWriter::VisitObjCProtocolListLoc(ObjCProtocolListLoc TyLoc) {
  Writer.AddSourceLocation(TyLoc.getLAngleLoc(), Record);
  Writer.AddSourceLocation(TyLoc.getRAngleLoc(), Record);
  for (unsigned i = 0, e = TyLoc.getNumProtocols(); i != e; ++i)
    Writer.AddSourceLocation(TyLoc.getProtocolLoc(i), Record);
}
void TypeLocWriter::VisitPointerLoc(PointerLoc TyLoc) {
  Writer.AddSourceLocation(TyLoc.getStarLoc(), Record);
}
void TypeLocWriter::VisitBlockPointerLoc(BlockPointerLoc TyLoc) {
  Writer.AddSourceLocation(TyLoc.getCaretLoc(), Record);
}
void TypeLocWriter::VisitMemberPointerLoc(MemberPointerLoc TyLoc) {
  Writer.AddSourceLocation(TyLoc.getStarLoc(), Record);
}
void TypeLocWriter::VisitReferenceLoc(ReferenceLoc TyLoc) {
  Writer.AddSourceLocation(TyLoc.getAmpLoc(), Record);
}
void TypeLocWriter::VisitFunctionLoc(FunctionLoc TyLoc) {
  Writer.AddSourceLocation(TyLoc.getLParenLoc(), Record);
  Writer.AddSourceLocation(TyLoc.getRParenLoc(), Record);
  for (unsigned i = 0, e = TyLoc.getNumArgs(); i != e; ++i)
    Writer.AddDeclRef(TyLoc.getArg(i), Record);
}
void TypeLocWriter::VisitArrayLoc(ArrayLoc TyLoc) {
  Writer.AddSourceLocation(TyLoc.getLBracketLoc(), Record);
  Writer.AddSourceLocation(TyLoc.getRBracketLoc(), Record);
  Record.push_back(TyLoc.getSizeExpr() ? 1 : 0);
  if (TyLoc.getSizeExpr())
    Writer.AddStmt(TyLoc.getSizeExpr());
}

void PCHDeclWriter::VisitDeclaratorDecl(DeclaratorDecl *D) {
  VisitValueDecl(D);
  DeclaratorInfo *DInfo = D->getDeclaratorInfo();
  if (DInfo == 0) {
    Writer.AddTypeRef(QualType(), Record);
    return;
  }

  Writer.AddTypeRef(DInfo->getTypeLoc().getSourceType(), Record);
  TypeLocWriter TLW(Writer, Record);
  for (TypeLoc TL = DInfo->getTypeLoc(); !TL.isNull(); TL = TL.getNextTypeLoc())
    TLW.Visit(TL);
}

void PCHDeclWriter::VisitFunctionDecl(FunctionDecl *D) {
  VisitDeclaratorDecl(D);
  Record.push_back(D->isThisDeclarationADefinition());
  if (D->isThisDeclarationADefinition())
    Writer.AddStmt(D->getBody());
  Writer.AddDeclRef(D->getPreviousDeclaration(), Record);
  Record.push_back(D->getStorageClass()); // FIXME: stable encoding
  Record.push_back(D->isInline());
  Record.push_back(D->isVirtualAsWritten());
  Record.push_back(D->isPure());
  Record.push_back(D->hasInheritedPrototype());
  Record.push_back(D->hasWrittenPrototype());
  Record.push_back(D->isDeleted());
  Record.push_back(D->isTrivial());
  Record.push_back(D->isCopyAssignment());
  Record.push_back(D->hasImplicitReturnZero());
  Writer.AddSourceLocation(D->getLocEnd(), Record);
  // FIXME: C++ TemplateOrInstantiation
  Record.push_back(D->param_size());
  for (FunctionDecl::param_iterator P = D->param_begin(), PEnd = D->param_end();
       P != PEnd; ++P)
    Writer.AddDeclRef(*P, Record);
  Code = pch::DECL_FUNCTION;
}

void PCHDeclWriter::VisitObjCMethodDecl(ObjCMethodDecl *D) {
  VisitNamedDecl(D);
  // FIXME: convert to LazyStmtPtr?
  // Unlike C/C++, method bodies will never be in header files.
  Record.push_back(D->getBody() != 0);
  if (D->getBody() != 0) {
    Writer.AddStmt(D->getBody());
    Writer.AddDeclRef(D->getSelfDecl(), Record);
    Writer.AddDeclRef(D->getCmdDecl(), Record);
  }
  Record.push_back(D->isInstanceMethod());
  Record.push_back(D->isVariadic());
  Record.push_back(D->isSynthesized());
  // FIXME: stable encoding for @required/@optional
  Record.push_back(D->getImplementationControl());
  // FIXME: stable encoding for in/out/inout/bycopy/byref/oneway
  Record.push_back(D->getObjCDeclQualifier());
  Writer.AddTypeRef(D->getResultType(), Record);
  Writer.AddSourceLocation(D->getLocEnd(), Record);
  Record.push_back(D->param_size());
  for (ObjCMethodDecl::param_iterator P = D->param_begin(),
                                   PEnd = D->param_end(); P != PEnd; ++P)
    Writer.AddDeclRef(*P, Record);
  Code = pch::DECL_OBJC_METHOD;
}

void PCHDeclWriter::VisitObjCContainerDecl(ObjCContainerDecl *D) {
  VisitNamedDecl(D);
  Writer.AddSourceLocation(D->getAtEndLoc(), Record);
  // Abstract class (no need to define a stable pch::DECL code).
}

void PCHDeclWriter::VisitObjCInterfaceDecl(ObjCInterfaceDecl *D) {
  VisitObjCContainerDecl(D);
  Writer.AddTypeRef(QualType(D->getTypeForDecl(), 0), Record);
  Writer.AddDeclRef(D->getSuperClass(), Record);
  Record.push_back(D->protocol_size());
  for (ObjCInterfaceDecl::protocol_iterator P = D->protocol_begin(),
         PEnd = D->protocol_end();
       P != PEnd; ++P)
    Writer.AddDeclRef(*P, Record);
  Record.push_back(D->ivar_size());
  for (ObjCInterfaceDecl::ivar_iterator I = D->ivar_begin(),
                                     IEnd = D->ivar_end(); I != IEnd; ++I)
    Writer.AddDeclRef(*I, Record);
  Writer.AddDeclRef(D->getCategoryList(), Record);
  Record.push_back(D->isForwardDecl());
  Record.push_back(D->isImplicitInterfaceDecl());
  Writer.AddSourceLocation(D->getClassLoc(), Record);
  Writer.AddSourceLocation(D->getSuperClassLoc(), Record);
  Writer.AddSourceLocation(D->getLocEnd(), Record);
  Code = pch::DECL_OBJC_INTERFACE;
}

void PCHDeclWriter::VisitObjCIvarDecl(ObjCIvarDecl *D) {
  VisitFieldDecl(D);
  // FIXME: stable encoding for @public/@private/@protected/@package
  Record.push_back(D->getAccessControl());
  Code = pch::DECL_OBJC_IVAR;
}

void PCHDeclWriter::VisitObjCProtocolDecl(ObjCProtocolDecl *D) {
  VisitObjCContainerDecl(D);
  Record.push_back(D->isForwardDecl());
  Writer.AddSourceLocation(D->getLocEnd(), Record);
  Record.push_back(D->protocol_size());
  for (ObjCProtocolDecl::protocol_iterator
       I = D->protocol_begin(), IEnd = D->protocol_end(); I != IEnd; ++I)
    Writer.AddDeclRef(*I, Record);
  Code = pch::DECL_OBJC_PROTOCOL;
}

void PCHDeclWriter::VisitObjCAtDefsFieldDecl(ObjCAtDefsFieldDecl *D) {
  VisitFieldDecl(D);
  Code = pch::DECL_OBJC_AT_DEFS_FIELD;
}

void PCHDeclWriter::VisitObjCClassDecl(ObjCClassDecl *D) {
  VisitDecl(D);
  Record.push_back(D->size());
  for (ObjCClassDecl::iterator I = D->begin(), IEnd = D->end(); I != IEnd; ++I)
    Writer.AddDeclRef(*I, Record);
  Code = pch::DECL_OBJC_CLASS;
}

void PCHDeclWriter::VisitObjCForwardProtocolDecl(ObjCForwardProtocolDecl *D) {
  VisitDecl(D);
  Record.push_back(D->protocol_size());
  for (ObjCProtocolDecl::protocol_iterator
       I = D->protocol_begin(), IEnd = D->protocol_end(); I != IEnd; ++I)
    Writer.AddDeclRef(*I, Record);
  Code = pch::DECL_OBJC_FORWARD_PROTOCOL;
}

void PCHDeclWriter::VisitObjCCategoryDecl(ObjCCategoryDecl *D) {
  VisitObjCContainerDecl(D);
  Writer.AddDeclRef(D->getClassInterface(), Record);
  Record.push_back(D->protocol_size());
  for (ObjCProtocolDecl::protocol_iterator
       I = D->protocol_begin(), IEnd = D->protocol_end(); I != IEnd; ++I)
    Writer.AddDeclRef(*I, Record);
  Writer.AddDeclRef(D->getNextClassCategory(), Record);
  Writer.AddSourceLocation(D->getLocEnd(), Record);
  Code = pch::DECL_OBJC_CATEGORY;
}

void PCHDeclWriter::VisitObjCCompatibleAliasDecl(ObjCCompatibleAliasDecl *D) {
  VisitNamedDecl(D);
  Writer.AddDeclRef(D->getClassInterface(), Record);
  Code = pch::DECL_OBJC_COMPATIBLE_ALIAS;
}

void PCHDeclWriter::VisitObjCPropertyDecl(ObjCPropertyDecl *D) {
  VisitNamedDecl(D);
  Writer.AddTypeRef(D->getType(), Record);
  // FIXME: stable encoding
  Record.push_back((unsigned)D->getPropertyAttributes());
  // FIXME: stable encoding
  Record.push_back((unsigned)D->getPropertyImplementation());
  Writer.AddDeclarationName(D->getGetterName(), Record);
  Writer.AddDeclarationName(D->getSetterName(), Record);
  Writer.AddDeclRef(D->getGetterMethodDecl(), Record);
  Writer.AddDeclRef(D->getSetterMethodDecl(), Record);
  Writer.AddDeclRef(D->getPropertyIvarDecl(), Record);
  Code = pch::DECL_OBJC_PROPERTY;
}

void PCHDeclWriter::VisitObjCImplDecl(ObjCImplDecl *D) {
  VisitObjCContainerDecl(D);
  Writer.AddDeclRef(D->getClassInterface(), Record);
  // Abstract class (no need to define a stable pch::DECL code).
}

void PCHDeclWriter::VisitObjCCategoryImplDecl(ObjCCategoryImplDecl *D) {
  VisitObjCImplDecl(D);
  Writer.AddIdentifierRef(D->getIdentifier(), Record);
  Code = pch::DECL_OBJC_CATEGORY_IMPL;
}

void PCHDeclWriter::VisitObjCImplementationDecl(ObjCImplementationDecl *D) {
  VisitObjCImplDecl(D);
  Writer.AddDeclRef(D->getSuperClass(), Record);
  Code = pch::DECL_OBJC_IMPLEMENTATION;
}

void PCHDeclWriter::VisitObjCPropertyImplDecl(ObjCPropertyImplDecl *D) {
  VisitDecl(D);
  Writer.AddSourceLocation(D->getLocStart(), Record);
  Writer.AddDeclRef(D->getPropertyDecl(), Record);
  Writer.AddDeclRef(D->getPropertyIvarDecl(), Record);
  Code = pch::DECL_OBJC_PROPERTY_IMPL;
}

void PCHDeclWriter::VisitFieldDecl(FieldDecl *D) {
  VisitDeclaratorDecl(D);
  Record.push_back(D->isMutable());
  Record.push_back(D->getBitWidth()? 1 : 0);
  if (D->getBitWidth())
    Writer.AddStmt(D->getBitWidth());
  Code = pch::DECL_FIELD;
}

void PCHDeclWriter::VisitVarDecl(VarDecl *D) {
  VisitDeclaratorDecl(D);
  Record.push_back(D->getStorageClass()); // FIXME: stable encoding
  Record.push_back(D->isThreadSpecified());
  Record.push_back(D->hasCXXDirectInitializer());
  Record.push_back(D->isDeclaredInCondition());
  Writer.AddDeclRef(D->getPreviousDeclaration(), Record);
  Record.push_back(D->getInit()? 1 : 0);
  if (D->getInit())
    Writer.AddStmt(D->getInit());
  Code = pch::DECL_VAR;
}

void PCHDeclWriter::VisitImplicitParamDecl(ImplicitParamDecl *D) {
  VisitVarDecl(D);
  Code = pch::DECL_IMPLICIT_PARAM;
}

void PCHDeclWriter::VisitParmVarDecl(ParmVarDecl *D) {
  VisitVarDecl(D);
  Record.push_back(D->getObjCDeclQualifier()); // FIXME: stable encoding
  Code = pch::DECL_PARM_VAR;


  // If the assumptions about the DECL_PARM_VAR abbrev are true, use it.  Here
  // we dynamically check for the properties that we optimize for, but don't
  // know are true of all PARM_VAR_DECLs.
  if (!D->getDeclaratorInfo() &&
      !D->hasAttrs() &&
      !D->isImplicit() &&
      !D->isUsed() &&
      D->getAccess() == AS_none &&
      D->getPCHLevel() == 0 &&
      D->getStorageClass() == 0 &&
      !D->hasCXXDirectInitializer() && // Can params have this ever?
      D->getObjCDeclQualifier() == 0)
    AbbrevToUse = Writer.getParmVarDeclAbbrev();

  // Check things we know are true of *every* PARM_VAR_DECL, which is more than
  // just us assuming it.
  assert(!D->isInvalidDecl() && "Shouldn't emit invalid decls");
  assert(!D->isThreadSpecified() && "PARM_VAR_DECL can't be __thread");
  assert(D->getAccess() == AS_none && "PARM_VAR_DECL can't be public/private");
  assert(!D->isDeclaredInCondition() && "PARM_VAR_DECL can't be in condition");
  assert(D->getPreviousDeclaration() == 0 && "PARM_VAR_DECL can't be redecl");
  assert(D->getInit() == 0 && "PARM_VAR_DECL never has init");
}

void PCHDeclWriter::VisitOriginalParmVarDecl(OriginalParmVarDecl *D) {
  VisitParmVarDecl(D);
  Writer.AddTypeRef(D->getOriginalType(), Record);
  Code = pch::DECL_ORIGINAL_PARM_VAR;
  AbbrevToUse = 0;
}

void PCHDeclWriter::VisitFileScopeAsmDecl(FileScopeAsmDecl *D) {
  VisitDecl(D);
  Writer.AddStmt(D->getAsmString());
  Code = pch::DECL_FILE_SCOPE_ASM;
}

void PCHDeclWriter::VisitBlockDecl(BlockDecl *D) {
  VisitDecl(D);
  Writer.AddStmt(D->getBody());
  Record.push_back(D->param_size());
  for (FunctionDecl::param_iterator P = D->param_begin(), PEnd = D->param_end();
       P != PEnd; ++P)
    Writer.AddDeclRef(*P, Record);
  Code = pch::DECL_BLOCK;
}

/// \brief Emit the DeclContext part of a declaration context decl.
///
/// \param LexicalOffset the offset at which the DECL_CONTEXT_LEXICAL
/// block for this declaration context is stored. May be 0 to indicate
/// that there are no declarations stored within this context.
///
/// \param VisibleOffset the offset at which the DECL_CONTEXT_VISIBLE
/// block for this declaration context is stored. May be 0 to indicate
/// that there are no declarations visible from this context. Note
/// that this value will not be emitted for non-primary declaration
/// contexts.
void PCHDeclWriter::VisitDeclContext(DeclContext *DC, uint64_t LexicalOffset,
                                     uint64_t VisibleOffset) {
  Record.push_back(LexicalOffset);
  Record.push_back(VisibleOffset);
}


//===----------------------------------------------------------------------===//
// PCHWriter Implementation
//===----------------------------------------------------------------------===//

void PCHWriter::WriteDeclsBlockAbbrevs() {
  using namespace llvm;
  // Abbreviation for DECL_PARM_VAR.
  BitCodeAbbrev *Abv = new BitCodeAbbrev();
  Abv->Add(BitCodeAbbrevOp(pch::DECL_PARM_VAR));

  // Decl
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // DeclContext
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // LexicalDeclContext
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // Location
  Abv->Add(BitCodeAbbrevOp(0));                       // isInvalidDecl (!?)
  Abv->Add(BitCodeAbbrevOp(0));                       // HasAttrs
  Abv->Add(BitCodeAbbrevOp(0));                       // isImplicit
  Abv->Add(BitCodeAbbrevOp(0));                       // isUsed
  Abv->Add(BitCodeAbbrevOp(AS_none));                 // C++ AccessSpecifier
  Abv->Add(BitCodeAbbrevOp(0));                       // PCH level

  // NamedDecl
  Abv->Add(BitCodeAbbrevOp(0));                       // NameKind = Identifier
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // Name
  // ValueDecl
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // Type
  // DeclaratorDecl
  Abv->Add(BitCodeAbbrevOp(pch::PREDEF_TYPE_NULL_ID)); // InfoType
  // VarDecl
  Abv->Add(BitCodeAbbrevOp(0));                       // StorageClass
  Abv->Add(BitCodeAbbrevOp(0));                       // isThreadSpecified
  Abv->Add(BitCodeAbbrevOp(0));                       // hasCXXDirectInitializer
  Abv->Add(BitCodeAbbrevOp(0));                       // isDeclaredInCondition
  Abv->Add(BitCodeAbbrevOp(0));                       // PrevDecl
  Abv->Add(BitCodeAbbrevOp(0));                       // HasInit
  // ParmVarDecl
  Abv->Add(BitCodeAbbrevOp(0));                       // ObjCDeclQualifier

  ParmVarDeclAbbrev = Stream.EmitAbbrev(Abv);
}

/// isRequiredDecl - Check if this is a "required" Decl, which must be seen by
/// consumers of the AST.
///
/// Such decls will always be deserialized from the PCH file, so we would like
/// this to be as restrictive as possible. Currently the predicate is driven by
/// code generation requirements, if other clients have a different notion of
/// what is "required" then we may have to consider an alternate scheme where
/// clients can iterate over the top-level decls and get information on them,
/// without necessary deserializing them. We could explicitly require such
/// clients to use a separate API call to "realize" the decl. This should be
/// relatively painless since they would presumably only do it for top-level
/// decls.
//
// FIXME: This predicate is essentially IRgen's predicate to determine whether a
// declaration can be deferred. Merge them somehow.
static bool isRequiredDecl(const Decl *D, ASTContext &Context) {
  // File scoped assembly must be seen.
  if (isa<FileScopeAsmDecl>(D))
    return true;

  // Otherwise if this isn't a function or a file scoped variable it doesn't
  // need to be seen.
  if (const VarDecl *VD = dyn_cast<VarDecl>(D)) {
    if (!VD->isFileVarDecl())
      return false;
  } else if (!isa<FunctionDecl>(D))
    return false;

  // Aliases and used decls must be seen.
  if (D->hasAttr<AliasAttr>() || D->hasAttr<UsedAttr>())
    return true;

  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    // Forward declarations don't need to be seen.
    if (!FD->isThisDeclarationADefinition())
      return false;

    // Constructors and destructors must be seen.
    if (FD->hasAttr<ConstructorAttr>() || FD->hasAttr<DestructorAttr>())
      return true;

    // Otherwise, this is required unless it is static.
    //
    // FIXME: Inlines.
    return FD->getStorageClass() != FunctionDecl::Static;
  } else {
    const VarDecl *VD = cast<VarDecl>(D);

    // In C++, this doesn't need to be seen if it is marked "extern".
    if (Context.getLangOptions().CPlusPlus && !VD->getInit() &&
        (VD->getStorageClass() == VarDecl::Extern ||
         VD->isExternC()))
      return false;

    // In C, this doesn't need to be seen unless it is a definition.
    if (!Context.getLangOptions().CPlusPlus && !VD->getInit())
      return false;

    // Otherwise, this is required unless it is static.
    return VD->getStorageClass() != VarDecl::Static;
  }
}

/// \brief Write a block containing all of the declarations.
void PCHWriter::WriteDeclsBlock(ASTContext &Context) {
  // Enter the declarations block.
  Stream.EnterSubblock(pch::DECLS_BLOCK_ID, 3);

  // Output the abbreviations that we will use in this block.
  WriteDeclsBlockAbbrevs();

  // Emit all of the declarations.
  RecordData Record;
  PCHDeclWriter W(*this, Context, Record);
  while (!DeclsToEmit.empty()) {
    // Pull the next declaration off the queue
    Decl *D = DeclsToEmit.front();
    DeclsToEmit.pop();

    // If this declaration is also a DeclContext, write blocks for the
    // declarations that lexically stored inside its context and those
    // declarations that are visible from its context. These blocks
    // are written before the declaration itself so that we can put
    // their offsets into the record for the declaration.
    uint64_t LexicalOffset = 0;
    uint64_t VisibleOffset = 0;
    DeclContext *DC = dyn_cast<DeclContext>(D);
    if (DC) {
      LexicalOffset = WriteDeclContextLexicalBlock(Context, DC);
      VisibleOffset = WriteDeclContextVisibleBlock(Context, DC);
    }

    // Determine the ID for this declaration
    pch::DeclID &ID = DeclIDs[D];
    if (ID == 0)
      ID = DeclIDs.size();

    unsigned Index = ID - 1;

    // Record the offset for this declaration
    if (DeclOffsets.size() == Index)
      DeclOffsets.push_back(Stream.GetCurrentBitNo());
    else if (DeclOffsets.size() < Index) {
      DeclOffsets.resize(Index+1);
      DeclOffsets[Index] = Stream.GetCurrentBitNo();
    }

    // Build and emit a record for this declaration
    Record.clear();
    W.Code = (pch::DeclCode)0;
    W.AbbrevToUse = 0;
    W.Visit(D);
    if (DC) W.VisitDeclContext(DC, LexicalOffset, VisibleOffset);

    if (!W.Code) {
      fprintf(stderr, "Cannot serialize declaration of kind %s\n",
              D->getDeclKindName());
      assert(false && "Unhandled declaration kind while generating PCH");
      exit(-1);
    }
    Stream.EmitRecord(W.Code, Record, W.AbbrevToUse);

    // If the declaration had any attributes, write them now.
    if (D->hasAttrs())
      WriteAttributeRecord(D->getAttrs());

    // Flush any expressions that were written as part of this declaration.
    FlushStmts();

    // Note "external" declarations so that we can add them to a record in the
    // PCH file later.
    //
    // FIXME: This should be renamed, the predicate is much more complicated.
    if (isRequiredDecl(D, Context))
      ExternalDefinitions.push_back(ID);
  }

  // Exit the declarations block
  Stream.ExitBlock();
}
