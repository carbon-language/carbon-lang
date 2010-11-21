//===--- ASTWriterDecl.cpp - Declaration Serialization --------------------===//
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

#include "clang/Serialization/ASTWriter.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/DeclContextInternals.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Bitcode/BitstreamWriter.h"
#include "llvm/Support/ErrorHandling.h"
using namespace clang;

//===----------------------------------------------------------------------===//
// Declaration serialization
//===----------------------------------------------------------------------===//

namespace clang {
  class ASTDeclWriter : public DeclVisitor<ASTDeclWriter, void> {

    ASTWriter &Writer;
    ASTContext &Context;
    typedef ASTWriter::RecordData RecordData;
    RecordData &Record;

  public:
    serialization::DeclCode Code;
    unsigned AbbrevToUse;

    ASTDeclWriter(ASTWriter &Writer, ASTContext &Context, RecordData &Record)
      : Writer(Writer), Context(Context), Record(Record) {
    }

    void Visit(Decl *D);

    void VisitDecl(Decl *D);
    void VisitTranslationUnitDecl(TranslationUnitDecl *D);
    void VisitNamedDecl(NamedDecl *D);
    void VisitNamespaceDecl(NamespaceDecl *D);
    void VisitUsingDirectiveDecl(UsingDirectiveDecl *D);
    void VisitNamespaceAliasDecl(NamespaceAliasDecl *D);
    void VisitTypeDecl(TypeDecl *D);
    void VisitTypedefDecl(TypedefDecl *D);
    void VisitUnresolvedUsingTypenameDecl(UnresolvedUsingTypenameDecl *D);
    void VisitTagDecl(TagDecl *D);
    void VisitEnumDecl(EnumDecl *D);
    void VisitRecordDecl(RecordDecl *D);
    void VisitCXXRecordDecl(CXXRecordDecl *D);
    void VisitClassTemplateSpecializationDecl(
                                            ClassTemplateSpecializationDecl *D);
    void VisitClassTemplatePartialSpecializationDecl(
                                     ClassTemplatePartialSpecializationDecl *D);
    void VisitTemplateTypeParmDecl(TemplateTypeParmDecl *D);
    void VisitValueDecl(ValueDecl *D);
    void VisitEnumConstantDecl(EnumConstantDecl *D);
    void VisitUnresolvedUsingValueDecl(UnresolvedUsingValueDecl *D);
    void VisitDeclaratorDecl(DeclaratorDecl *D);
    void VisitFunctionDecl(FunctionDecl *D);
    void VisitCXXMethodDecl(CXXMethodDecl *D);
    void VisitCXXConstructorDecl(CXXConstructorDecl *D);
    void VisitCXXDestructorDecl(CXXDestructorDecl *D);
    void VisitCXXConversionDecl(CXXConversionDecl *D);
    void VisitFieldDecl(FieldDecl *D);
    void VisitIndirectFieldDecl(IndirectFieldDecl *D);
    void VisitVarDecl(VarDecl *D);
    void VisitImplicitParamDecl(ImplicitParamDecl *D);
    void VisitParmVarDecl(ParmVarDecl *D);
    void VisitNonTypeTemplateParmDecl(NonTypeTemplateParmDecl *D);
    void VisitTemplateDecl(TemplateDecl *D);
    void VisitRedeclarableTemplateDecl(RedeclarableTemplateDecl *D);
    void VisitClassTemplateDecl(ClassTemplateDecl *D);
    void VisitFunctionTemplateDecl(FunctionTemplateDecl *D);
    void VisitTemplateTemplateParmDecl(TemplateTemplateParmDecl *D);
    void VisitUsingDecl(UsingDecl *D);
    void VisitUsingShadowDecl(UsingShadowDecl *D);
    void VisitLinkageSpecDecl(LinkageSpecDecl *D);
    void VisitFileScopeAsmDecl(FileScopeAsmDecl *D);
    void VisitAccessSpecDecl(AccessSpecDecl *D);
    void VisitFriendDecl(FriendDecl *D);
    void VisitFriendTemplateDecl(FriendTemplateDecl *D);
    void VisitStaticAssertDecl(StaticAssertDecl *D);
    void VisitBlockDecl(BlockDecl *D);

    void VisitDeclContext(DeclContext *DC, uint64_t LexicalOffset,
                          uint64_t VisibleOffset);
    template <typename T> void VisitRedeclarable(Redeclarable<T> *D);


    // FIXME: Put in the same order is DeclNodes.td?
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

void ASTDeclWriter::Visit(Decl *D) {
  DeclVisitor<ASTDeclWriter>::Visit(D);

  // Handle FunctionDecl's body here and write it after all other Stmts/Exprs
  // have been written. We want it last because we will not read it back when
  // retrieving it from the AST, we'll just lazily set the offset. 
  if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    Record.push_back(FD->isThisDeclarationADefinition());
    if (FD->isThisDeclarationADefinition())
      Writer.AddStmt(FD->getBody());
  }
}

void ASTDeclWriter::VisitDecl(Decl *D) {
  Writer.AddDeclRef(cast_or_null<Decl>(D->getDeclContext()), Record);
  Writer.AddDeclRef(cast_or_null<Decl>(D->getLexicalDeclContext()), Record);
  Writer.AddSourceLocation(D->getLocation(), Record);
  Record.push_back(D->isInvalidDecl());
  Record.push_back(D->hasAttrs());
  if (D->hasAttrs())
    Writer.WriteAttributes(D->getAttrs(), Record);
  Record.push_back(D->isImplicit());
  Record.push_back(D->isUsed(false));
  Record.push_back(D->getAccess());
  Record.push_back(D->getPCHLevel());
}

void ASTDeclWriter::VisitTranslationUnitDecl(TranslationUnitDecl *D) {
  VisitDecl(D);
  Writer.AddDeclRef(D->getAnonymousNamespace(), Record);
  Code = serialization::DECL_TRANSLATION_UNIT;
}

void ASTDeclWriter::VisitNamedDecl(NamedDecl *D) {
  VisitDecl(D);
  Writer.AddDeclarationName(D->getDeclName(), Record);
}

void ASTDeclWriter::VisitTypeDecl(TypeDecl *D) {
  VisitNamedDecl(D);
  Writer.AddTypeRef(QualType(D->getTypeForDecl(), 0), Record);
}

void ASTDeclWriter::VisitTypedefDecl(TypedefDecl *D) {
  VisitTypeDecl(D);
  Writer.AddTypeSourceInfo(D->getTypeSourceInfo(), Record);
  Code = serialization::DECL_TYPEDEF;
}

void ASTDeclWriter::VisitTagDecl(TagDecl *D) {
  VisitTypeDecl(D);
  VisitRedeclarable(D);
  Record.push_back(D->getIdentifierNamespace());
  Record.push_back((unsigned)D->getTagKind()); // FIXME: stable encoding
  Record.push_back(D->isDefinition());
  Record.push_back(D->isEmbeddedInDeclarator());
  Writer.AddSourceLocation(D->getRBraceLoc(), Record);
  Writer.AddSourceLocation(D->getTagKeywordLoc(), Record);
  Record.push_back(D->hasExtInfo());
  if (D->hasExtInfo())
    Writer.AddQualifierInfo(*D->getExtInfo(), Record);
  else
    Writer.AddDeclRef(D->getTypedefForAnonDecl(), Record);
}

void ASTDeclWriter::VisitEnumDecl(EnumDecl *D) {
  VisitTagDecl(D);
  Writer.AddTypeSourceInfo(D->getIntegerTypeSourceInfo(), Record);
  if (!D->getIntegerTypeSourceInfo())
    Writer.AddTypeRef(D->getIntegerType(), Record);
  Writer.AddTypeRef(D->getPromotionType(), Record);
  Record.push_back(D->getNumPositiveBits());
  Record.push_back(D->getNumNegativeBits());
  Record.push_back(D->isScoped());
  Record.push_back(D->isFixed());
  Writer.AddDeclRef(D->getInstantiatedFromMemberEnum(), Record);
  Code = serialization::DECL_ENUM;
}

void ASTDeclWriter::VisitRecordDecl(RecordDecl *D) {
  VisitTagDecl(D);
  Record.push_back(D->hasFlexibleArrayMember());
  Record.push_back(D->isAnonymousStructOrUnion());
  Record.push_back(D->hasObjectMember());
  Code = serialization::DECL_RECORD;
}

void ASTDeclWriter::VisitValueDecl(ValueDecl *D) {
  VisitNamedDecl(D);
  Writer.AddTypeRef(D->getType(), Record);
}

void ASTDeclWriter::VisitEnumConstantDecl(EnumConstantDecl *D) {
  VisitValueDecl(D);
  Record.push_back(D->getInitExpr()? 1 : 0);
  if (D->getInitExpr())
    Writer.AddStmt(D->getInitExpr());
  Writer.AddAPSInt(D->getInitVal(), Record);
  Code = serialization::DECL_ENUM_CONSTANT;
}

void ASTDeclWriter::VisitDeclaratorDecl(DeclaratorDecl *D) {
  VisitValueDecl(D);
  Record.push_back(D->hasExtInfo());
  if (D->hasExtInfo())
    Writer.AddQualifierInfo(*D->getExtInfo(), Record);
  Writer.AddTypeSourceInfo(D->getTypeSourceInfo(), Record);
}

void ASTDeclWriter::VisitFunctionDecl(FunctionDecl *D) {
  VisitDeclaratorDecl(D);
  VisitRedeclarable(D);

  Writer.AddDeclarationNameLoc(D->DNLoc, D->getDeclName(), Record);
  Record.push_back(D->getIdentifierNamespace());
  Record.push_back(D->getTemplatedKind());
  switch (D->getTemplatedKind()) {
  default: assert(false && "Unhandled TemplatedKind!");
    break;
  case FunctionDecl::TK_NonTemplate:
    break;
  case FunctionDecl::TK_FunctionTemplate:
    Writer.AddDeclRef(D->getDescribedFunctionTemplate(), Record);
    break;
  case FunctionDecl::TK_MemberSpecialization: {
    MemberSpecializationInfo *MemberInfo = D->getMemberSpecializationInfo();
    Writer.AddDeclRef(MemberInfo->getInstantiatedFrom(), Record);
    Record.push_back(MemberInfo->getTemplateSpecializationKind());
    Writer.AddSourceLocation(MemberInfo->getPointOfInstantiation(), Record);
    break;
  }
  case FunctionDecl::TK_FunctionTemplateSpecialization: {
    FunctionTemplateSpecializationInfo *
      FTSInfo = D->getTemplateSpecializationInfo();
    Writer.AddDeclRef(FTSInfo->getTemplate(), Record);
    Record.push_back(FTSInfo->getTemplateSpecializationKind());
    
    // Template arguments.
    Writer.AddTemplateArgumentList(FTSInfo->TemplateArguments, Record);
    
    // Template args as written.
    Record.push_back(FTSInfo->TemplateArgumentsAsWritten != 0);
    if (FTSInfo->TemplateArgumentsAsWritten) {
      Record.push_back(FTSInfo->TemplateArgumentsAsWritten->size());
      for (int i=0, e = FTSInfo->TemplateArgumentsAsWritten->size(); i!=e; ++i)
        Writer.AddTemplateArgumentLoc((*FTSInfo->TemplateArgumentsAsWritten)[i],
                                      Record);
      Writer.AddSourceLocation(FTSInfo->TemplateArgumentsAsWritten->getLAngleLoc(),
                               Record);
      Writer.AddSourceLocation(FTSInfo->TemplateArgumentsAsWritten->getRAngleLoc(),
                               Record);
    }
    
    Writer.AddSourceLocation(FTSInfo->getPointOfInstantiation(), Record);

    if (D->isCanonicalDecl()) {
      // Write the template that contains the specializations set. We will
      // add a FunctionTemplateSpecializationInfo to it when reading.
      Writer.AddDeclRef(FTSInfo->getTemplate()->getCanonicalDecl(), Record);
    }
    break;
  }
  case FunctionDecl::TK_DependentFunctionTemplateSpecialization: {
    DependentFunctionTemplateSpecializationInfo *
      DFTSInfo = D->getDependentSpecializationInfo();
    
    // Templates.
    Record.push_back(DFTSInfo->getNumTemplates());
    for (int i=0, e = DFTSInfo->getNumTemplates(); i != e; ++i)
      Writer.AddDeclRef(DFTSInfo->getTemplate(i), Record);
    
    // Templates args.
    Record.push_back(DFTSInfo->getNumTemplateArgs());
    for (int i=0, e = DFTSInfo->getNumTemplateArgs(); i != e; ++i)
      Writer.AddTemplateArgumentLoc(DFTSInfo->getTemplateArg(i), Record);
    Writer.AddSourceLocation(DFTSInfo->getLAngleLoc(), Record);
    Writer.AddSourceLocation(DFTSInfo->getRAngleLoc(), Record);
    break;
  }
  }

  // FunctionDecl's body is handled last at ASTWriterDecl::Visit,
  // after everything else is written.

  Record.push_back(D->getStorageClass()); // FIXME: stable encoding
  Record.push_back(D->getStorageClassAsWritten());
  Record.push_back(D->isInlineSpecified());
  Record.push_back(D->isVirtualAsWritten());
  Record.push_back(D->isPure());
  Record.push_back(D->hasInheritedPrototype());
  Record.push_back(D->hasWrittenPrototype());
  Record.push_back(D->isDeleted());
  Record.push_back(D->isTrivial());
  Record.push_back(D->hasImplicitReturnZero());
  Writer.AddSourceLocation(D->getLocEnd(), Record);

  Record.push_back(D->param_size());
  for (FunctionDecl::param_iterator P = D->param_begin(), PEnd = D->param_end();
       P != PEnd; ++P)
    Writer.AddDeclRef(*P, Record);
  Code = serialization::DECL_FUNCTION;
}

void ASTDeclWriter::VisitObjCMethodDecl(ObjCMethodDecl *D) {
  VisitNamedDecl(D);
  // FIXME: convert to LazyStmtPtr?
  // Unlike C/C++, method bodies will never be in header files.
  bool HasBodyStuff = D->getBody() != 0     ||
                      D->getSelfDecl() != 0 || D->getCmdDecl() != 0;
  Record.push_back(HasBodyStuff);
  if (HasBodyStuff) {
    Writer.AddStmt(D->getBody());
    Writer.AddDeclRef(D->getSelfDecl(), Record);
    Writer.AddDeclRef(D->getCmdDecl(), Record);
  }
  Record.push_back(D->isInstanceMethod());
  Record.push_back(D->isVariadic());
  Record.push_back(D->isSynthesized());
  Record.push_back(D->isDefined());
  // FIXME: stable encoding for @required/@optional
  Record.push_back(D->getImplementationControl());
  // FIXME: stable encoding for in/out/inout/bycopy/byref/oneway
  Record.push_back(D->getObjCDeclQualifier());
  Record.push_back(D->getNumSelectorArgs());
  Writer.AddTypeRef(D->getResultType(), Record);
  Writer.AddTypeSourceInfo(D->getResultTypeSourceInfo(), Record);
  Writer.AddSourceLocation(D->getLocEnd(), Record);
  Record.push_back(D->param_size());
  for (ObjCMethodDecl::param_iterator P = D->param_begin(),
                                   PEnd = D->param_end(); P != PEnd; ++P)
    Writer.AddDeclRef(*P, Record);
  Code = serialization::DECL_OBJC_METHOD;
}

void ASTDeclWriter::VisitObjCContainerDecl(ObjCContainerDecl *D) {
  VisitNamedDecl(D);
  Writer.AddSourceRange(D->getAtEndRange(), Record);
  // Abstract class (no need to define a stable serialization::DECL code).
}

void ASTDeclWriter::VisitObjCInterfaceDecl(ObjCInterfaceDecl *D) {
  VisitObjCContainerDecl(D);
  Writer.AddTypeRef(QualType(D->getTypeForDecl(), 0), Record);
  Writer.AddDeclRef(D->getSuperClass(), Record);

  // Write out the protocols that are directly referenced by the @interface.
  Record.push_back(D->ReferencedProtocols.size());
  for (ObjCInterfaceDecl::protocol_iterator P = D->protocol_begin(),
         PEnd = D->protocol_end();
       P != PEnd; ++P)
    Writer.AddDeclRef(*P, Record);
  for (ObjCInterfaceDecl::protocol_loc_iterator PL = D->protocol_loc_begin(),
         PLEnd = D->protocol_loc_end();
       PL != PLEnd; ++PL)
    Writer.AddSourceLocation(*PL, Record);

  // Write out the protocols that are transitively referenced.
  Record.push_back(D->AllReferencedProtocols.size());
  for (ObjCList<ObjCProtocolDecl>::iterator
        P = D->AllReferencedProtocols.begin(),
        PEnd = D->AllReferencedProtocols.end();
       P != PEnd; ++P)
    Writer.AddDeclRef(*P, Record);
  
  // Write out the ivars.
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
  Code = serialization::DECL_OBJC_INTERFACE;
}

void ASTDeclWriter::VisitObjCIvarDecl(ObjCIvarDecl *D) {
  VisitFieldDecl(D);
  // FIXME: stable encoding for @public/@private/@protected/@package
  Record.push_back(D->getAccessControl());
  Record.push_back(D->getSynthesize());
  Code = serialization::DECL_OBJC_IVAR;
}

void ASTDeclWriter::VisitObjCProtocolDecl(ObjCProtocolDecl *D) {
  VisitObjCContainerDecl(D);
  Record.push_back(D->isForwardDecl());
  Writer.AddSourceLocation(D->getLocEnd(), Record);
  Record.push_back(D->protocol_size());
  for (ObjCProtocolDecl::protocol_iterator
       I = D->protocol_begin(), IEnd = D->protocol_end(); I != IEnd; ++I)
    Writer.AddDeclRef(*I, Record);
  for (ObjCProtocolDecl::protocol_loc_iterator PL = D->protocol_loc_begin(),
         PLEnd = D->protocol_loc_end();
       PL != PLEnd; ++PL)
    Writer.AddSourceLocation(*PL, Record);
  Code = serialization::DECL_OBJC_PROTOCOL;
}

void ASTDeclWriter::VisitObjCAtDefsFieldDecl(ObjCAtDefsFieldDecl *D) {
  VisitFieldDecl(D);
  Code = serialization::DECL_OBJC_AT_DEFS_FIELD;
}

void ASTDeclWriter::VisitObjCClassDecl(ObjCClassDecl *D) {
  VisitDecl(D);
  Record.push_back(D->size());
  for (ObjCClassDecl::iterator I = D->begin(), IEnd = D->end(); I != IEnd; ++I)
    Writer.AddDeclRef(I->getInterface(), Record);
  for (ObjCClassDecl::iterator I = D->begin(), IEnd = D->end(); I != IEnd; ++I)
    Writer.AddSourceLocation(I->getLocation(), Record);
  Code = serialization::DECL_OBJC_CLASS;
}

void ASTDeclWriter::VisitObjCForwardProtocolDecl(ObjCForwardProtocolDecl *D) {
  VisitDecl(D);
  Record.push_back(D->protocol_size());
  for (ObjCForwardProtocolDecl::protocol_iterator
       I = D->protocol_begin(), IEnd = D->protocol_end(); I != IEnd; ++I)
    Writer.AddDeclRef(*I, Record);
  for (ObjCForwardProtocolDecl::protocol_loc_iterator 
         PL = D->protocol_loc_begin(), PLEnd = D->protocol_loc_end();
       PL != PLEnd; ++PL)
    Writer.AddSourceLocation(*PL, Record);
  Code = serialization::DECL_OBJC_FORWARD_PROTOCOL;
}

void ASTDeclWriter::VisitObjCCategoryDecl(ObjCCategoryDecl *D) {
  VisitObjCContainerDecl(D);
  Writer.AddDeclRef(D->getClassInterface(), Record);
  Record.push_back(D->protocol_size());
  for (ObjCCategoryDecl::protocol_iterator
       I = D->protocol_begin(), IEnd = D->protocol_end(); I != IEnd; ++I)
    Writer.AddDeclRef(*I, Record);
  for (ObjCCategoryDecl::protocol_loc_iterator 
         PL = D->protocol_loc_begin(), PLEnd = D->protocol_loc_end();
       PL != PLEnd; ++PL)
    Writer.AddSourceLocation(*PL, Record);
  Writer.AddDeclRef(D->getNextClassCategory(), Record);
  Record.push_back(D->hasSynthBitfield());
  Writer.AddSourceLocation(D->getAtLoc(), Record);
  Writer.AddSourceLocation(D->getCategoryNameLoc(), Record);
  Code = serialization::DECL_OBJC_CATEGORY;
}

void ASTDeclWriter::VisitObjCCompatibleAliasDecl(ObjCCompatibleAliasDecl *D) {
  VisitNamedDecl(D);
  Writer.AddDeclRef(D->getClassInterface(), Record);
  Code = serialization::DECL_OBJC_COMPATIBLE_ALIAS;
}

void ASTDeclWriter::VisitObjCPropertyDecl(ObjCPropertyDecl *D) {
  VisitNamedDecl(D);
  Writer.AddSourceLocation(D->getAtLoc(), Record);
  Writer.AddTypeSourceInfo(D->getTypeSourceInfo(), Record);
  // FIXME: stable encoding
  Record.push_back((unsigned)D->getPropertyAttributes());
  Record.push_back((unsigned)D->getPropertyAttributesAsWritten());
  // FIXME: stable encoding
  Record.push_back((unsigned)D->getPropertyImplementation());
  Writer.AddDeclarationName(D->getGetterName(), Record);
  Writer.AddDeclarationName(D->getSetterName(), Record);
  Writer.AddDeclRef(D->getGetterMethodDecl(), Record);
  Writer.AddDeclRef(D->getSetterMethodDecl(), Record);
  Writer.AddDeclRef(D->getPropertyIvarDecl(), Record);
  Code = serialization::DECL_OBJC_PROPERTY;
}

void ASTDeclWriter::VisitObjCImplDecl(ObjCImplDecl *D) {
  VisitObjCContainerDecl(D);
  Writer.AddDeclRef(D->getClassInterface(), Record);
  // Abstract class (no need to define a stable serialization::DECL code).
}

void ASTDeclWriter::VisitObjCCategoryImplDecl(ObjCCategoryImplDecl *D) {
  VisitObjCImplDecl(D);
  Writer.AddIdentifierRef(D->getIdentifier(), Record);
  Code = serialization::DECL_OBJC_CATEGORY_IMPL;
}

void ASTDeclWriter::VisitObjCImplementationDecl(ObjCImplementationDecl *D) {
  VisitObjCImplDecl(D);
  Writer.AddDeclRef(D->getSuperClass(), Record);
  Writer.AddCXXBaseOrMemberInitializers(D->IvarInitializers,
                                        D->NumIvarInitializers, Record);
  Record.push_back(D->hasSynthBitfield());
  Code = serialization::DECL_OBJC_IMPLEMENTATION;
}

void ASTDeclWriter::VisitObjCPropertyImplDecl(ObjCPropertyImplDecl *D) {
  VisitDecl(D);
  Writer.AddSourceLocation(D->getLocStart(), Record);
  Writer.AddDeclRef(D->getPropertyDecl(), Record);
  Writer.AddDeclRef(D->getPropertyIvarDecl(), Record);
  Writer.AddSourceLocation(D->getPropertyIvarDeclLoc(), Record);
  Writer.AddStmt(D->getGetterCXXConstructor());
  Writer.AddStmt(D->getSetterCXXAssignment());
  Code = serialization::DECL_OBJC_PROPERTY_IMPL;
}

void ASTDeclWriter::VisitFieldDecl(FieldDecl *D) {
  VisitDeclaratorDecl(D);
  Record.push_back(D->isMutable());
  Record.push_back(D->getBitWidth()? 1 : 0);
  if (D->getBitWidth())
    Writer.AddStmt(D->getBitWidth());
  if (!D->getDeclName())
    Writer.AddDeclRef(Context.getInstantiatedFromUnnamedFieldDecl(D), Record);
  Code = serialization::DECL_FIELD;
}

void ASTDeclWriter::VisitIndirectFieldDecl(IndirectFieldDecl *D) {
  VisitValueDecl(D);
  Record.push_back(D->getChainingSize());

  for (IndirectFieldDecl::chain_iterator
       P = D->chain_begin(),
       PEnd = D->chain_end(); P != PEnd; ++P)
    Writer.AddDeclRef(*P, Record);
  Code = serialization::DECL_INDIRECTFIELD;
}

void ASTDeclWriter::VisitVarDecl(VarDecl *D) {
  VisitDeclaratorDecl(D);
  VisitRedeclarable(D);
  Record.push_back(D->getStorageClass()); // FIXME: stable encoding
  Record.push_back(D->getStorageClassAsWritten());
  Record.push_back(D->isThreadSpecified());
  Record.push_back(D->hasCXXDirectInitializer());
  Record.push_back(D->isExceptionVariable());
  Record.push_back(D->isNRVOVariable());
  Record.push_back(D->getInit() ? 1 : 0);
  if (D->getInit())
    Writer.AddStmt(D->getInit());

  MemberSpecializationInfo *SpecInfo
    = D->isStaticDataMember() ? D->getMemberSpecializationInfo() : 0;
  Record.push_back(SpecInfo != 0);
  if (SpecInfo) {
    Writer.AddDeclRef(SpecInfo->getInstantiatedFrom(), Record);
    Record.push_back(SpecInfo->getTemplateSpecializationKind());
    Writer.AddSourceLocation(SpecInfo->getPointOfInstantiation(), Record);
  }

  Code = serialization::DECL_VAR;
}

void ASTDeclWriter::VisitImplicitParamDecl(ImplicitParamDecl *D) {
  VisitVarDecl(D);
  Code = serialization::DECL_IMPLICIT_PARAM;
}

void ASTDeclWriter::VisitParmVarDecl(ParmVarDecl *D) {
  VisitVarDecl(D);
  Record.push_back(D->getObjCDeclQualifier()); // FIXME: stable encoding
  Record.push_back(D->hasInheritedDefaultArg());
  Record.push_back(D->hasUninstantiatedDefaultArg());
  if (D->hasUninstantiatedDefaultArg())
    Writer.AddStmt(D->getUninstantiatedDefaultArg());
  Code = serialization::DECL_PARM_VAR;

  // If the assumptions about the DECL_PARM_VAR abbrev are true, use it.  Here
  // we dynamically check for the properties that we optimize for, but don't
  // know are true of all PARM_VAR_DECLs.
  if (!D->getTypeSourceInfo() &&
      !D->hasAttrs() &&
      !D->isImplicit() &&
      !D->isUsed(false) &&
      D->getAccess() == AS_none &&
      D->getPCHLevel() == 0 &&
      D->getStorageClass() == 0 &&
      !D->hasCXXDirectInitializer() && // Can params have this ever?
      D->getObjCDeclQualifier() == 0 &&
      !D->hasInheritedDefaultArg() &&
      D->getInit() == 0 &&
      !D->hasUninstantiatedDefaultArg())  // No default expr.
    AbbrevToUse = Writer.getParmVarDeclAbbrev();

  // Check things we know are true of *every* PARM_VAR_DECL, which is more than
  // just us assuming it.
  assert(!D->isInvalidDecl() && "Shouldn't emit invalid decls");
  assert(!D->isThreadSpecified() && "PARM_VAR_DECL can't be __thread");
  assert(D->getAccess() == AS_none && "PARM_VAR_DECL can't be public/private");
  assert(!D->isExceptionVariable() && "PARM_VAR_DECL can't be exception var");
  assert(D->getPreviousDeclaration() == 0 && "PARM_VAR_DECL can't be redecl");
  assert(!D->isStaticDataMember() &&
         "PARM_VAR_DECL can't be static data member");
}

void ASTDeclWriter::VisitFileScopeAsmDecl(FileScopeAsmDecl *D) {
  VisitDecl(D);
  Writer.AddStmt(D->getAsmString());
  Code = serialization::DECL_FILE_SCOPE_ASM;
}

void ASTDeclWriter::VisitBlockDecl(BlockDecl *D) {
  VisitDecl(D);
  Writer.AddStmt(D->getBody());
  Writer.AddTypeSourceInfo(D->getSignatureAsWritten(), Record);
  Record.push_back(D->param_size());
  for (FunctionDecl::param_iterator P = D->param_begin(), PEnd = D->param_end();
       P != PEnd; ++P)
    Writer.AddDeclRef(*P, Record);
  Code = serialization::DECL_BLOCK;
}

void ASTDeclWriter::VisitLinkageSpecDecl(LinkageSpecDecl *D) {
  VisitDecl(D);
  // FIXME: It might be nice to serialize the brace locations for this
  // declaration, which don't seem to be readily available in the AST.
  Record.push_back(D->getLanguage());
  Record.push_back(D->hasBraces());
  Code = serialization::DECL_LINKAGE_SPEC;
}

void ASTDeclWriter::VisitNamespaceDecl(NamespaceDecl *D) {
  VisitNamedDecl(D);
  Record.push_back(D->isInline());
  Writer.AddSourceLocation(D->getLBracLoc(), Record);
  Writer.AddSourceLocation(D->getRBracLoc(), Record);
  Writer.AddDeclRef(D->getNextNamespace(), Record);

  // Only write one reference--original or anonymous
  Record.push_back(D->isOriginalNamespace());
  if (D->isOriginalNamespace())
    Writer.AddDeclRef(D->getAnonymousNamespace(), Record);
  else
    Writer.AddDeclRef(D->getOriginalNamespace(), Record);
  Code = serialization::DECL_NAMESPACE;

  if (Writer.hasChain() && !D->isOriginalNamespace() &&
      D->getOriginalNamespace()->getPCHLevel() > 0) {
    NamespaceDecl *NS = D->getOriginalNamespace();
    Writer.AddUpdatedDeclContext(NS);

    // Make sure all visible decls are written. They will be recorded later.
    NS->lookup(DeclarationName());
    StoredDeclsMap *Map = static_cast<StoredDeclsMap*>(NS->getLookupPtr());
    if (Map) {
      for (StoredDeclsMap::iterator D = Map->begin(), DEnd = Map->end();
           D != DEnd; ++D) {
        DeclarationName Name = D->first;
        DeclContext::lookup_result Result = D->second.getLookupResult();
        while (Result.first != Result.second) {
          Writer.GetDeclRef(*Result.first);
          ++Result.first;
        }
      }
    }
  }
}

void ASTDeclWriter::VisitNamespaceAliasDecl(NamespaceAliasDecl *D) {
  VisitNamedDecl(D);
  Writer.AddSourceLocation(D->getNamespaceLoc(), Record);
  Writer.AddSourceRange(D->getQualifierRange(), Record);
  Writer.AddNestedNameSpecifier(D->getQualifier(), Record);
  Writer.AddSourceLocation(D->getTargetNameLoc(), Record);
  Writer.AddDeclRef(D->getNamespace(), Record);
  Code = serialization::DECL_NAMESPACE_ALIAS;
}

void ASTDeclWriter::VisitUsingDecl(UsingDecl *D) {
  VisitNamedDecl(D);
  Writer.AddSourceRange(D->getNestedNameRange(), Record);
  Writer.AddSourceLocation(D->getUsingLocation(), Record);
  Writer.AddNestedNameSpecifier(D->getTargetNestedNameDecl(), Record);
  Writer.AddDeclarationNameLoc(D->DNLoc, D->getDeclName(), Record);
  Writer.AddDeclRef(D->FirstUsingShadow, Record);
  Record.push_back(D->isTypeName());
  Writer.AddDeclRef(Context.getInstantiatedFromUsingDecl(D), Record);
  Code = serialization::DECL_USING;
}

void ASTDeclWriter::VisitUsingShadowDecl(UsingShadowDecl *D) {
  VisitNamedDecl(D);
  Writer.AddDeclRef(D->getTargetDecl(), Record);
  Writer.AddDeclRef(D->UsingOrNextShadow, Record);
  Writer.AddDeclRef(Context.getInstantiatedFromUsingShadowDecl(D), Record);
  Code = serialization::DECL_USING_SHADOW;
}

void ASTDeclWriter::VisitUsingDirectiveDecl(UsingDirectiveDecl *D) {
  VisitNamedDecl(D);
  Writer.AddSourceLocation(D->getUsingLoc(), Record);
  Writer.AddSourceLocation(D->getNamespaceKeyLocation(), Record);
  Writer.AddSourceRange(D->getQualifierRange(), Record);
  Writer.AddNestedNameSpecifier(D->getQualifier(), Record);
  Writer.AddDeclRef(D->getNominatedNamespace(), Record);
  Writer.AddDeclRef(dyn_cast<Decl>(D->getCommonAncestor()), Record);
  Code = serialization::DECL_USING_DIRECTIVE;
}

void ASTDeclWriter::VisitUnresolvedUsingValueDecl(UnresolvedUsingValueDecl *D) {
  VisitValueDecl(D);
  Writer.AddSourceRange(D->getTargetNestedNameRange(), Record);
  Writer.AddSourceLocation(D->getUsingLoc(), Record);
  Writer.AddNestedNameSpecifier(D->getTargetNestedNameSpecifier(), Record);
  Writer.AddDeclarationNameLoc(D->DNLoc, D->getDeclName(), Record);
  Code = serialization::DECL_UNRESOLVED_USING_VALUE;
}

void ASTDeclWriter::VisitUnresolvedUsingTypenameDecl(
                                               UnresolvedUsingTypenameDecl *D) {
  VisitTypeDecl(D);
  Writer.AddSourceRange(D->getTargetNestedNameRange(), Record);
  Writer.AddSourceLocation(D->getUsingLoc(), Record);
  Writer.AddSourceLocation(D->getTypenameLoc(), Record);
  Writer.AddNestedNameSpecifier(D->getTargetNestedNameSpecifier(), Record);
  Code = serialization::DECL_UNRESOLVED_USING_TYPENAME;
}

void ASTDeclWriter::VisitCXXRecordDecl(CXXRecordDecl *D) {
  VisitRecordDecl(D);

  CXXRecordDecl *DefinitionDecl = 0;
  if (D->DefinitionData)
    DefinitionDecl = D->DefinitionData->Definition;
  Writer.AddDeclRef(DefinitionDecl, Record);
  if (D == DefinitionDecl)
    Writer.AddCXXDefinitionData(D, Record);

  enum {
    CXXRecNotTemplate = 0, CXXRecTemplate, CXXRecMemberSpecialization
  };
  if (ClassTemplateDecl *TemplD = D->getDescribedClassTemplate()) {
    Record.push_back(CXXRecTemplate);
    Writer.AddDeclRef(TemplD, Record);
  } else if (MemberSpecializationInfo *MSInfo
               = D->getMemberSpecializationInfo()) {
    Record.push_back(CXXRecMemberSpecialization);
    Writer.AddDeclRef(MSInfo->getInstantiatedFrom(), Record);
    Record.push_back(MSInfo->getTemplateSpecializationKind());
    Writer.AddSourceLocation(MSInfo->getPointOfInstantiation(), Record);
  } else {
    Record.push_back(CXXRecNotTemplate);
  }

  // Store the key function to avoid deserializing every method so we can
  // compute it.
  if (D->IsDefinition)
    Writer.AddDeclRef(Context.getKeyFunction(D), Record);

  Code = serialization::DECL_CXX_RECORD;
}

void ASTDeclWriter::VisitCXXMethodDecl(CXXMethodDecl *D) {
  VisitFunctionDecl(D);
  Record.push_back(D->size_overridden_methods());
  for (CXXMethodDecl::method_iterator
         I = D->begin_overridden_methods(), E = D->end_overridden_methods();
         I != E; ++I)
    Writer.AddDeclRef(*I, Record);
  Code = serialization::DECL_CXX_METHOD;
}

void ASTDeclWriter::VisitCXXConstructorDecl(CXXConstructorDecl *D) {
  VisitCXXMethodDecl(D);

  Record.push_back(D->IsExplicitSpecified);
  Record.push_back(D->ImplicitlyDefined);
  Writer.AddCXXBaseOrMemberInitializers(D->BaseOrMemberInitializers,
                                        D->NumBaseOrMemberInitializers, Record);

  Code = serialization::DECL_CXX_CONSTRUCTOR;
}

void ASTDeclWriter::VisitCXXDestructorDecl(CXXDestructorDecl *D) {
  VisitCXXMethodDecl(D);

  Record.push_back(D->ImplicitlyDefined);
  Writer.AddDeclRef(D->OperatorDelete, Record);

  Code = serialization::DECL_CXX_DESTRUCTOR;
}

void ASTDeclWriter::VisitCXXConversionDecl(CXXConversionDecl *D) {
  VisitCXXMethodDecl(D);
  Record.push_back(D->IsExplicitSpecified);
  Code = serialization::DECL_CXX_CONVERSION;
}

void ASTDeclWriter::VisitAccessSpecDecl(AccessSpecDecl *D) {
  VisitDecl(D);
  Writer.AddSourceLocation(D->getColonLoc(), Record);
  Code = serialization::DECL_ACCESS_SPEC;
}

void ASTDeclWriter::VisitFriendDecl(FriendDecl *D) {
  VisitDecl(D);
  Record.push_back(D->Friend.is<TypeSourceInfo*>());
  if (D->Friend.is<TypeSourceInfo*>())
    Writer.AddTypeSourceInfo(D->Friend.get<TypeSourceInfo*>(), Record);
  else
    Writer.AddDeclRef(D->Friend.get<NamedDecl*>(), Record);
  Writer.AddDeclRef(D->getNextFriend(), Record);
  Record.push_back(D->UnsupportedFriend);
  Writer.AddSourceLocation(D->FriendLoc, Record);
  Code = serialization::DECL_FRIEND;
}

void ASTDeclWriter::VisitFriendTemplateDecl(FriendTemplateDecl *D) {
  VisitDecl(D);
  Record.push_back(D->getNumTemplateParameters());
  for (unsigned i = 0, e = D->getNumTemplateParameters(); i != e; ++i)
    Writer.AddTemplateParameterList(D->getTemplateParameterList(i), Record);
  Record.push_back(D->getFriendDecl() != 0);
  if (D->getFriendDecl())
    Writer.AddDeclRef(D->getFriendDecl(), Record);
  else
    Writer.AddTypeSourceInfo(D->getFriendType(), Record);
  Writer.AddSourceLocation(D->getFriendLoc(), Record);
  Code = serialization::DECL_FRIEND_TEMPLATE;
}

void ASTDeclWriter::VisitTemplateDecl(TemplateDecl *D) {
  VisitNamedDecl(D);

  Writer.AddDeclRef(D->getTemplatedDecl(), Record);
  Writer.AddTemplateParameterList(D->getTemplateParameters(), Record);
}

void ASTDeclWriter::VisitRedeclarableTemplateDecl(RedeclarableTemplateDecl *D) {
  // Emit data to initialize CommonOrPrev before VisitTemplateDecl so that
  // getCommonPtr() can be used while this is still initializing.

  Writer.AddDeclRef(D->getPreviousDeclaration(), Record);
  if (D->getPreviousDeclaration() == 0) {
    // This TemplateDecl owns the CommonPtr; write it.
    assert(D->isCanonicalDecl());

    Writer.AddDeclRef(D->getInstantiatedFromMemberTemplate(), Record);
    if (D->getInstantiatedFromMemberTemplate())
      Record.push_back(D->isMemberSpecialization());

    Writer.AddDeclRef(D->getCommonPtr()->Latest, Record);
  } else {
    RedeclarableTemplateDecl *First = D->getFirstDeclaration();
    assert(First != D);
    // If this is a most recent redeclaration that is pointed to by a first decl
    // in a chained PCH, keep track of the association with the map so we can
    // update the first decl during AST reading.
    if (First->getMostRecentDeclaration() == D &&
        First->getPCHLevel() > D->getPCHLevel()) {
      assert(Writer.FirstLatestDecls.find(First)==Writer.FirstLatestDecls.end()
             && "The latest is already set");
      Writer.FirstLatestDecls[First] = D;
    }
  }

  VisitTemplateDecl(D);
  Record.push_back(D->getIdentifierNamespace());
}

void ASTDeclWriter::VisitClassTemplateDecl(ClassTemplateDecl *D) {
  VisitRedeclarableTemplateDecl(D);

  if (D->getPreviousDeclaration() == 0) {
    typedef llvm::FoldingSet<ClassTemplateSpecializationDecl> CTSDSetTy;
    CTSDSetTy &CTSDSet = D->getSpecializations();
    Record.push_back(CTSDSet.size());
    for (CTSDSetTy::iterator I=CTSDSet.begin(), E = CTSDSet.end(); I!=E; ++I) {
      assert(I->isCanonicalDecl() && "Expected only canonical decls in set");
      Writer.AddDeclRef(&*I, Record);
    }

    typedef llvm::FoldingSet<ClassTemplatePartialSpecializationDecl> CTPSDSetTy;
    CTPSDSetTy &CTPSDSet = D->getPartialSpecializations();
    Record.push_back(CTPSDSet.size());
    for (CTPSDSetTy::iterator I=CTPSDSet.begin(), E=CTPSDSet.end(); I!=E; ++I) {
      assert(I->isCanonicalDecl() && "Expected only canonical decls in set");
      Writer.AddDeclRef(&*I, Record); 
    }

    // InjectedClassNameType is computed, no need to write it.
  }
  Code = serialization::DECL_CLASS_TEMPLATE;
}

void ASTDeclWriter::VisitClassTemplateSpecializationDecl(
                                           ClassTemplateSpecializationDecl *D) {
  VisitCXXRecordDecl(D);

  llvm::PointerUnion<ClassTemplateDecl *,
                     ClassTemplatePartialSpecializationDecl *> InstFrom
    = D->getSpecializedTemplateOrPartial();
  Decl *InstFromD;
  if (InstFrom.is<ClassTemplateDecl *>()) {
    InstFromD = InstFrom.get<ClassTemplateDecl *>();
    Writer.AddDeclRef(InstFromD, Record);
  } else {
    InstFromD = InstFrom.get<ClassTemplatePartialSpecializationDecl *>();
    Writer.AddDeclRef(InstFromD, Record);
    Writer.AddTemplateArgumentList(&D->getTemplateInstantiationArgs(), Record);
    InstFromD = cast<ClassTemplatePartialSpecializationDecl>(InstFromD)->
                    getSpecializedTemplate();
  }

  // Explicit info.
  Writer.AddTypeSourceInfo(D->getTypeAsWritten(), Record);
  if (D->getTypeAsWritten()) {
    Writer.AddSourceLocation(D->getExternLoc(), Record);
    Writer.AddSourceLocation(D->getTemplateKeywordLoc(), Record);
  }

  Writer.AddTemplateArgumentList(&D->getTemplateArgs(), Record);
  Writer.AddSourceLocation(D->getPointOfInstantiation(), Record);
  Record.push_back(D->getSpecializationKind());

  if (D->isCanonicalDecl()) {
    // When reading, we'll add it to the folding set of the following template. 
    Writer.AddDeclRef(D->getSpecializedTemplate()->getCanonicalDecl(), Record);
  }

  Code = serialization::DECL_CLASS_TEMPLATE_SPECIALIZATION;
}

void ASTDeclWriter::VisitClassTemplatePartialSpecializationDecl(
                                    ClassTemplatePartialSpecializationDecl *D) {
  VisitClassTemplateSpecializationDecl(D);

  Writer.AddTemplateParameterList(D->getTemplateParameters(), Record);

  Record.push_back(D->getNumTemplateArgsAsWritten());
  for (int i = 0, e = D->getNumTemplateArgsAsWritten(); i != e; ++i)
    Writer.AddTemplateArgumentLoc(D->getTemplateArgsAsWritten()[i], Record);

  Record.push_back(D->getSequenceNumber());

  // These are read/set from/to the first declaration.
  if (D->getPreviousDeclaration() == 0) {
    Writer.AddDeclRef(D->getInstantiatedFromMember(), Record);
    Record.push_back(D->isMemberSpecialization());
  }

  Code = serialization::DECL_CLASS_TEMPLATE_PARTIAL_SPECIALIZATION;
}

void ASTDeclWriter::VisitFunctionTemplateDecl(FunctionTemplateDecl *D) {
  VisitRedeclarableTemplateDecl(D);

  if (D->getPreviousDeclaration() == 0) {
    // This FunctionTemplateDecl owns the CommonPtr; write it.

    // Write the function specialization declarations.
    Record.push_back(D->getSpecializations().size());
    for (llvm::FoldingSet<FunctionTemplateSpecializationInfo>::iterator
           I = D->getSpecializations().begin(),
           E = D->getSpecializations().end()   ; I != E; ++I) {
      assert(I->Function->isCanonicalDecl() &&
             "Expected only canonical decls in set");
      Writer.AddDeclRef(I->Function, Record);
    }
  }
  Code = serialization::DECL_FUNCTION_TEMPLATE;
}

void ASTDeclWriter::VisitTemplateTypeParmDecl(TemplateTypeParmDecl *D) {
  VisitTypeDecl(D);

  Record.push_back(D->wasDeclaredWithTypename());
  Record.push_back(D->isParameterPack());
  Record.push_back(D->defaultArgumentWasInherited());
  Writer.AddTypeSourceInfo(D->getDefaultArgumentInfo(), Record);

  Code = serialization::DECL_TEMPLATE_TYPE_PARM;
}

void ASTDeclWriter::VisitNonTypeTemplateParmDecl(NonTypeTemplateParmDecl *D) {
  VisitVarDecl(D);
  // TemplateParmPosition.
  Record.push_back(D->getDepth());
  Record.push_back(D->getPosition());
  // Rest of NonTypeTemplateParmDecl.
  Record.push_back(D->getDefaultArgument() != 0);
  if (D->getDefaultArgument()) {
    Writer.AddStmt(D->getDefaultArgument());
    Record.push_back(D->defaultArgumentWasInherited());
  }
  Code = serialization::DECL_NON_TYPE_TEMPLATE_PARM;
}

void ASTDeclWriter::VisitTemplateTemplateParmDecl(TemplateTemplateParmDecl *D) {
  VisitTemplateDecl(D);
  // TemplateParmPosition.
  Record.push_back(D->getDepth());
  Record.push_back(D->getPosition());
  // Rest of TemplateTemplateParmDecl.
  Writer.AddTemplateArgumentLoc(D->getDefaultArgument(), Record);
  Record.push_back(D->defaultArgumentWasInherited());
  Code = serialization::DECL_TEMPLATE_TEMPLATE_PARM;
}

void ASTDeclWriter::VisitStaticAssertDecl(StaticAssertDecl *D) {
  VisitDecl(D);
  Writer.AddStmt(D->getAssertExpr());
  Writer.AddStmt(D->getMessage());
  Code = serialization::DECL_STATIC_ASSERT;
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
void ASTDeclWriter::VisitDeclContext(DeclContext *DC, uint64_t LexicalOffset,
                                     uint64_t VisibleOffset) {
  Record.push_back(LexicalOffset);
  Record.push_back(VisibleOffset);
}

template <typename T>
void ASTDeclWriter::VisitRedeclarable(Redeclarable<T> *D) {
  enum { NoRedeclaration = 0, PointsToPrevious, PointsToLatest };
  if (D->RedeclLink.getNext() == D) {
    Record.push_back(NoRedeclaration);
  } else {
    Record.push_back(D->RedeclLink.NextIsPrevious() ? PointsToPrevious
                                                    : PointsToLatest);
    Writer.AddDeclRef(D->RedeclLink.getPointer(), Record);
  }

  T *First = D->getFirstDeclaration();
  T *ThisDecl = static_cast<T*>(D);
  // If this is a most recent redeclaration that is pointed to by a first decl
  // in a chained PCH, keep track of the association with the map so we can
  // update the first decl during AST reading.
  if (ThisDecl != First && First->getMostRecentDeclaration() == ThisDecl &&
      First->getPCHLevel() > ThisDecl->getPCHLevel()) {
    assert(Writer.FirstLatestDecls.find(First) == Writer.FirstLatestDecls.end()
           && "The latest is already set");
    Writer.FirstLatestDecls[First] = ThisDecl;
  }
}

//===----------------------------------------------------------------------===//
// ASTWriter Implementation
//===----------------------------------------------------------------------===//

void ASTWriter::WriteDeclsBlockAbbrevs() {
  using namespace llvm;
  // Abbreviation for DECL_PARM_VAR.
  BitCodeAbbrev *Abv = new BitCodeAbbrev();
  Abv->Add(BitCodeAbbrevOp(serialization::DECL_PARM_VAR));

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
  Abv->Add(BitCodeAbbrevOp(0));                       // hasExtInfo
  Abv->Add(BitCodeAbbrevOp(serialization::PREDEF_TYPE_NULL_ID)); // InfoType
  // VarDecl
  Abv->Add(BitCodeAbbrevOp(0));                       // No redeclaration
  Abv->Add(BitCodeAbbrevOp(0));                       // StorageClass
  Abv->Add(BitCodeAbbrevOp(0));                       // StorageClassAsWritten
  Abv->Add(BitCodeAbbrevOp(0));                       // isThreadSpecified
  Abv->Add(BitCodeAbbrevOp(0));                       // hasCXXDirectInitializer
  Abv->Add(BitCodeAbbrevOp(0));                       // isExceptionVariable
  Abv->Add(BitCodeAbbrevOp(0));                       // isNRVOVariable
  Abv->Add(BitCodeAbbrevOp(0));                       // HasInit
  Abv->Add(BitCodeAbbrevOp(0));                   // HasMemberSpecializationInfo
  // ParmVarDecl
  Abv->Add(BitCodeAbbrevOp(0));                       // ObjCDeclQualifier
  Abv->Add(BitCodeAbbrevOp(0));                       // HasInheritedDefaultArg
  Abv->Add(BitCodeAbbrevOp(0));                   // HasUninstantiatedDefaultArg

  ParmVarDeclAbbrev = Stream.EmitAbbrev(Abv);

  Abv = new BitCodeAbbrev();
  Abv->Add(BitCodeAbbrevOp(serialization::DECL_CONTEXT_LEXICAL));
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob));
  DeclContextLexicalAbbrev = Stream.EmitAbbrev(Abv);

  Abv = new BitCodeAbbrev();
  Abv->Add(BitCodeAbbrevOp(serialization::DECL_CONTEXT_VISIBLE));
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 32));
  Abv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob));
  DeclContextVisibleLookupAbbrev = Stream.EmitAbbrev(Abv);
}

/// isRequiredDecl - Check if this is a "required" Decl, which must be seen by
/// consumers of the AST.
///
/// Such decls will always be deserialized from the AST file, so we would like
/// this to be as restrictive as possible. Currently the predicate is driven by
/// code generation requirements, if other clients have a different notion of
/// what is "required" then we may have to consider an alternate scheme where
/// clients can iterate over the top-level decls and get information on them,
/// without necessary deserializing them. We could explicitly require such
/// clients to use a separate API call to "realize" the decl. This should be
/// relatively painless since they would presumably only do it for top-level
/// decls.
static bool isRequiredDecl(const Decl *D, ASTContext &Context) {
  // File scoped assembly or obj-c implementation must be seen.
  if (isa<FileScopeAsmDecl>(D) || isa<ObjCImplementationDecl>(D))
    return true;

  return Context.DeclMustBeEmitted(D);
}

void ASTWriter::WriteDecl(ASTContext &Context, Decl *D) {
  // Switch case IDs are per Decl.
  ClearSwitchCaseIDs();

  RecordData Record;
  ASTDeclWriter W(*this, Context, Record);

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
  serialization::DeclID &IDR = DeclIDs[D];
  if (IDR == 0)
    IDR = NextDeclID++;
  serialization::DeclID ID = IDR;

  if (ID < FirstDeclID) {
    // We're replacing a decl in a previous file.
    ReplacedDecls.push_back(std::make_pair(ID, Stream.GetCurrentBitNo()));
  } else {
    unsigned Index = ID - FirstDeclID;

    // Record the offset for this declaration
    if (DeclOffsets.size() == Index)
      DeclOffsets.push_back(Stream.GetCurrentBitNo());
    else if (DeclOffsets.size() < Index) {
      DeclOffsets.resize(Index+1);
      DeclOffsets[Index] = Stream.GetCurrentBitNo();
    }
  }

  // Build and emit a record for this declaration
  Record.clear();
  W.Code = (serialization::DeclCode)0;
  W.AbbrevToUse = 0;
  W.Visit(D);
  if (DC) W.VisitDeclContext(DC, LexicalOffset, VisibleOffset);

  if (!W.Code)
    llvm::report_fatal_error(llvm::StringRef("unexpected declaration kind '") +
                            D->getDeclKindName() + "'");
  Stream.EmitRecord(W.Code, Record, W.AbbrevToUse);

  // Flush any expressions that were written as part of this declaration.
  FlushStmts();
  
  // Flush C++ base specifiers, if there are any.
  FlushCXXBaseSpecifiers();
  
  // Note "external" declarations so that we can add them to a record in the
  // AST file later.
  //
  // FIXME: This should be renamed, the predicate is much more complicated.
  if (isRequiredDecl(D, Context))
    ExternalDefinitions.push_back(ID);
}
