//===--- ASTDumper.cpp - Dumping implementation for ASTs ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the AST dump methods, which dump out the
// AST in a form that exposes type details and other fields.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/Module.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/Support/raw_ostream.h"
using namespace clang;

//===----------------------------------------------------------------------===//
// ASTDumper Visitor
//===----------------------------------------------------------------------===//

namespace  {
  class ASTDumper
      : public DeclVisitor<ASTDumper>, public StmtVisitor<ASTDumper> {
    SourceManager *SM;
    raw_ostream &OS;
    unsigned IndentLevel;
    bool IsFirstLine;

    /// Keep track of the last location we print out so that we can
    /// print out deltas from then on out.
    const char *LastLocFilename;
    unsigned LastLocLine;

    class IndentScope {
      ASTDumper &Dumper;
    public:
      IndentScope(ASTDumper &Dumper) : Dumper(Dumper) {
        Dumper.indent();
      }
      ~IndentScope() {
        Dumper.unindent();
      }
    };

  public:
    ASTDumper(SourceManager *SM, raw_ostream &OS)
      : SM(SM), OS(OS), IndentLevel(0), IsFirstLine(true),
        LastLocFilename(""), LastLocLine(~0U) { }

    ~ASTDumper() {
      OS << "\n";
    }

    void dumpDecl(Decl *D);
    void dumpStmt(Stmt *S);

    // Utilities
    void indent();
    void unindent();
    void dumpPointer(const void *Ptr);
    void dumpSourceRange(SourceRange R);
    void dumpLocation(SourceLocation Loc);
    void dumpBareType(QualType T);
    void dumpType(QualType T);
    void dumpBareDeclRef(const Decl *Node);
    void dumpDeclRef(const Decl *Node, const char *Label = 0);
    void dumpName(const NamedDecl *D);
    void dumpDeclContext(const DeclContext *DC);
    void dumpAttr(const Attr *A);

    // C++ Utilities
    void dumpAccessSpecifier(AccessSpecifier AS);
    void dumpCXXCtorInitializer(const CXXCtorInitializer *Init);
    void dumpTemplateParameters(const TemplateParameterList *TPL);
    void dumpTemplateArgumentListInfo(const TemplateArgumentListInfo &TALI);
    void dumpTemplateArgumentLoc(const TemplateArgumentLoc &A);
    void dumpTemplateArgumentList(const TemplateArgumentList &TAL);
    void dumpTemplateArgument(const TemplateArgument &A,
                              SourceRange R = SourceRange());

    // Decls
    void VisitLabelDecl(LabelDecl *D);
    void VisitTypedefDecl(TypedefDecl *D);
    void VisitEnumDecl(EnumDecl *D);
    void VisitRecordDecl(RecordDecl *D);
    void VisitEnumConstantDecl(EnumConstantDecl *D);
    void VisitIndirectFieldDecl(IndirectFieldDecl *D);
    void VisitFunctionDecl(FunctionDecl *D);
    void VisitFieldDecl(FieldDecl *D);
    void VisitVarDecl(VarDecl *D);
    void VisitFileScopeAsmDecl(FileScopeAsmDecl *D);
    void VisitImportDecl(ImportDecl *D);

    // C++ Decls
    void VisitNamespaceDecl(NamespaceDecl *D);
    void VisitUsingDirectiveDecl(UsingDirectiveDecl *D);
    void VisitNamespaceAliasDecl(NamespaceAliasDecl *D);
    void VisitTypeAliasDecl(TypeAliasDecl *D);
    void VisitTypeAliasTemplateDecl(TypeAliasTemplateDecl *D);
    void VisitCXXRecordDecl(CXXRecordDecl *D);
    void VisitStaticAssertDecl(StaticAssertDecl *D);
    void VisitFunctionTemplateDecl(FunctionTemplateDecl *D);
    void VisitClassTemplateDecl(ClassTemplateDecl *D);
    void VisitClassTemplateSpecializationDecl(
        ClassTemplateSpecializationDecl *D);
    void VisitClassTemplatePartialSpecializationDecl(
        ClassTemplatePartialSpecializationDecl *D);
    void VisitClassScopeFunctionSpecializationDecl(
        ClassScopeFunctionSpecializationDecl *D);
    void VisitTemplateTypeParmDecl(TemplateTypeParmDecl *D);
    void VisitNonTypeTemplateParmDecl(NonTypeTemplateParmDecl *D);
    void VisitTemplateTemplateParmDecl(TemplateTemplateParmDecl *D);
    void VisitUsingDecl(UsingDecl *D);
    void VisitUnresolvedUsingTypenameDecl(UnresolvedUsingTypenameDecl *D);
    void VisitUnresolvedUsingValueDecl(UnresolvedUsingValueDecl *D);
    void VisitUsingShadowDecl(UsingShadowDecl *D);
    void VisitLinkageSpecDecl(LinkageSpecDecl *D);
    void VisitAccessSpecDecl(AccessSpecDecl *D);
    void VisitFriendDecl(FriendDecl *D);

    // ObjC Decls
    void VisitObjCIvarDecl(ObjCIvarDecl *D);
    void VisitObjCMethodDecl(ObjCMethodDecl *D);
    void VisitObjCCategoryDecl(ObjCCategoryDecl *D);
    void VisitObjCCategoryImplDecl(ObjCCategoryImplDecl *D);
    void VisitObjCProtocolDecl(ObjCProtocolDecl *D);
    void VisitObjCInterfaceDecl(ObjCInterfaceDecl *D);
    void VisitObjCImplementationDecl(ObjCImplementationDecl *D);
    void VisitObjCCompatibleAliasDecl(ObjCCompatibleAliasDecl *D);
    void VisitObjCPropertyDecl(ObjCPropertyDecl *D);
    void VisitObjCPropertyImplDecl(ObjCPropertyImplDecl *D);
    void VisitBlockDecl(BlockDecl *D);

    // Stmts.
    void VisitStmt(Stmt *Node);
    void VisitDeclStmt(DeclStmt *Node);
    void VisitAttributedStmt(AttributedStmt *Node);
    void VisitLabelStmt(LabelStmt *Node);
    void VisitGotoStmt(GotoStmt *Node);

    // Exprs
    void VisitExpr(Expr *Node);
    void VisitCastExpr(CastExpr *Node);
    void VisitDeclRefExpr(DeclRefExpr *Node);
    void VisitPredefinedExpr(PredefinedExpr *Node);
    void VisitCharacterLiteral(CharacterLiteral *Node);
    void VisitIntegerLiteral(IntegerLiteral *Node);
    void VisitFloatingLiteral(FloatingLiteral *Node);
    void VisitStringLiteral(StringLiteral *Str);
    void VisitUnaryOperator(UnaryOperator *Node);
    void VisitUnaryExprOrTypeTraitExpr(UnaryExprOrTypeTraitExpr *Node);
    void VisitMemberExpr(MemberExpr *Node);
    void VisitExtVectorElementExpr(ExtVectorElementExpr *Node);
    void VisitBinaryOperator(BinaryOperator *Node);
    void VisitCompoundAssignOperator(CompoundAssignOperator *Node);
    void VisitAddrLabelExpr(AddrLabelExpr *Node);
    void VisitBlockExpr(BlockExpr *Node);
    void VisitOpaqueValueExpr(OpaqueValueExpr *Node);

    // C++
    void VisitCXXNamedCastExpr(CXXNamedCastExpr *Node);
    void VisitCXXBoolLiteralExpr(CXXBoolLiteralExpr *Node);
    void VisitCXXThisExpr(CXXThisExpr *Node);
    void VisitCXXFunctionalCastExpr(CXXFunctionalCastExpr *Node);
    void VisitCXXConstructExpr(CXXConstructExpr *Node);
    void VisitCXXBindTemporaryExpr(CXXBindTemporaryExpr *Node);
    void VisitExprWithCleanups(ExprWithCleanups *Node);
    void VisitUnresolvedLookupExpr(UnresolvedLookupExpr *Node);
    void dumpCXXTemporary(CXXTemporary *Temporary);

    // ObjC
    void VisitObjCAtCatchStmt(ObjCAtCatchStmt *Node);
    void VisitObjCEncodeExpr(ObjCEncodeExpr *Node);
    void VisitObjCMessageExpr(ObjCMessageExpr *Node);
    void VisitObjCBoxedExpr(ObjCBoxedExpr *Node);
    void VisitObjCSelectorExpr(ObjCSelectorExpr *Node);
    void VisitObjCProtocolExpr(ObjCProtocolExpr *Node);
    void VisitObjCPropertyRefExpr(ObjCPropertyRefExpr *Node);
    void VisitObjCSubscriptRefExpr(ObjCSubscriptRefExpr *Node);
    void VisitObjCIvarRefExpr(ObjCIvarRefExpr *Node);
    void VisitObjCBoolLiteralExpr(ObjCBoolLiteralExpr *Node);
  };
}

//===----------------------------------------------------------------------===//
//  Utilities
//===----------------------------------------------------------------------===//

void ASTDumper::indent() {
  if (IsFirstLine)
    IsFirstLine = false;
  else
    OS << "\n";
  OS.indent(IndentLevel * 2);
  OS << "(";
  IndentLevel++;
}

void ASTDumper::unindent() {
  OS << ")";
  IndentLevel--;
}

void ASTDumper::dumpPointer(const void *Ptr) {
  OS << ' ' << Ptr;
}

void ASTDumper::dumpLocation(SourceLocation Loc) {
  SourceLocation SpellingLoc = SM->getSpellingLoc(Loc);

  // The general format we print out is filename:line:col, but we drop pieces
  // that haven't changed since the last loc printed.
  PresumedLoc PLoc = SM->getPresumedLoc(SpellingLoc);

  if (PLoc.isInvalid()) {
    OS << "<invalid sloc>";
    return;
  }

  if (strcmp(PLoc.getFilename(), LastLocFilename) != 0) {
    OS << PLoc.getFilename() << ':' << PLoc.getLine()
       << ':' << PLoc.getColumn();
    LastLocFilename = PLoc.getFilename();
    LastLocLine = PLoc.getLine();
  } else if (PLoc.getLine() != LastLocLine) {
    OS << "line" << ':' << PLoc.getLine()
       << ':' << PLoc.getColumn();
    LastLocLine = PLoc.getLine();
  } else {
    OS << "col" << ':' << PLoc.getColumn();
  }
}

void ASTDumper::dumpSourceRange(SourceRange R) {
  // Can't translate locations if a SourceManager isn't available.
  if (!SM)
    return;

  OS << " <";
  dumpLocation(R.getBegin());
  if (R.getBegin() != R.getEnd()) {
    OS << ", ";
    dumpLocation(R.getEnd());
  }
  OS << ">";

  // <t2.c:123:421[blah], t2.c:412:321>

}

void ASTDumper::dumpBareType(QualType T) {
  SplitQualType T_split = T.split();
  OS << "'" << QualType::getAsString(T_split) << "'";

  if (!T.isNull()) {
    // If the type is sugared, also dump a (shallow) desugared type.
    SplitQualType D_split = T.getSplitDesugaredType();
    if (T_split != D_split)
      OS << ":'" << QualType::getAsString(D_split) << "'";
  }
}

void ASTDumper::dumpType(QualType T) {
  OS << ' ';
  dumpBareType(T);
}

void ASTDumper::dumpBareDeclRef(const Decl *D) {
  OS << D->getDeclKindName();
  dumpPointer(D);

  if (const NamedDecl *ND = dyn_cast<NamedDecl>(D)) {
    OS << " '";
    ND->getDeclName().printName(OS);
    OS << "'";
  }

  if (const ValueDecl *VD = dyn_cast<ValueDecl>(D))
    dumpType(VD->getType());
}

void ASTDumper::dumpDeclRef(const Decl *D, const char *Label) {
  if (!D)
    return;

  IndentScope Indent(*this);
  if (Label)
    OS << Label << ' ';
  dumpBareDeclRef(D);
}

void ASTDumper::dumpName(const NamedDecl *ND) {
  if (ND->getDeclName())
    OS << ' ' << ND->getNameAsString();
}

void ASTDumper::dumpDeclContext(const DeclContext *DC) {
  if (!DC)
    return;
  for (DeclContext::decl_iterator I = DC->decls_begin(), E = DC->decls_end();
       I != E; ++I)
    dumpDecl(*I);
}

void ASTDumper::dumpAttr(const Attr *A) {
  IndentScope Indent(*this);
  switch (A->getKind()) {
#define ATTR(X) case attr::X: OS << #X; break;
#include "clang/Basic/AttrList.inc"
  default: llvm_unreachable("unexpected attribute kind");
  }
  OS << "Attr";
  dumpPointer(A);
  dumpSourceRange(A->getRange());
#include "clang/AST/AttrDump.inc"
}

//===----------------------------------------------------------------------===//
//  C++ Utilities
//===----------------------------------------------------------------------===//

void ASTDumper::dumpAccessSpecifier(AccessSpecifier AS) {
  switch (AS) {
  case AS_none:
    break;
  case AS_public:
    OS << "public";
    break;
  case AS_protected:
    OS << "protected";
    break;
  case AS_private:
    OS << "private";
    break;
  }
}

void ASTDumper::dumpCXXCtorInitializer(const CXXCtorInitializer *Init) {
  IndentScope Indent(*this);
  OS << "CXXCtorInitializer";
  if (Init->isAnyMemberInitializer()) {
    OS << ' ';
    dumpBareDeclRef(Init->getAnyMember());
  } else {
    dumpType(QualType(Init->getBaseClass(), 0));
  }
  dumpStmt(Init->getInit());
}

void ASTDumper::dumpTemplateParameters(const TemplateParameterList *TPL) {
  if (!TPL)
    return;

  for (TemplateParameterList::const_iterator I = TPL->begin(), E = TPL->end();
       I != E; ++I)
    dumpDecl(*I);
}

void ASTDumper::dumpTemplateArgumentListInfo(
    const TemplateArgumentListInfo &TALI) {
  for (unsigned i = 0, e = TALI.size(); i < e; ++i)
    dumpTemplateArgumentLoc(TALI[i]);
}

void ASTDumper::dumpTemplateArgumentLoc(const TemplateArgumentLoc &A) {
  dumpTemplateArgument(A.getArgument(), A.getSourceRange());
}

void ASTDumper::dumpTemplateArgumentList(const TemplateArgumentList &TAL) {
  for (unsigned i = 0, e = TAL.size(); i < e; ++i)
    dumpTemplateArgument(TAL[i]);
}

void ASTDumper::dumpTemplateArgument(const TemplateArgument &A, SourceRange R) {
  IndentScope Indent(*this);
  OS << "TemplateArgument";
  if (R.isValid())
    dumpSourceRange(R);

  switch (A.getKind()) {
  case TemplateArgument::Null:
    OS << " null";
    break;
  case TemplateArgument::Type:
    OS << " type";
    dumpType(A.getAsType());
    break;
  case TemplateArgument::Declaration:
    OS << " decl";
    dumpDeclRef(A.getAsDecl());
    break;
  case TemplateArgument::NullPtr:
    OS << " nullptr";
    break;
  case TemplateArgument::Integral:
    OS << " integral " << A.getAsIntegral();
    break;
  case TemplateArgument::Template:
    OS << " template ";
    A.getAsTemplate().dump(OS);
    break;
  case TemplateArgument::TemplateExpansion:
    OS << " template expansion";
    A.getAsTemplateOrTemplatePattern().dump(OS);
    break;
  case TemplateArgument::Expression:
    OS << " expr";
    dumpStmt(A.getAsExpr());
    break;
  case TemplateArgument::Pack:
    OS << " pack";
    for (TemplateArgument::pack_iterator I = A.pack_begin(), E = A.pack_end();
         I != E; ++I)
      dumpTemplateArgument(*I);
    break;
  }
}

//===----------------------------------------------------------------------===//
//  Decl dumping methods.
//===----------------------------------------------------------------------===//

void ASTDumper::dumpDecl(Decl *D) {
  IndentScope Indent(*this);

  if (!D) {
    OS << "<<<NULL>>>";
    return;
  }

  OS << D->getDeclKindName() << "Decl";
  dumpPointer(D);
  dumpSourceRange(D->getSourceRange());
  DeclVisitor<ASTDumper>::Visit(D);
  if (D->hasAttrs()) {
    for (AttrVec::const_iterator I = D->getAttrs().begin(),
         E = D->getAttrs().end(); I != E; ++I)
      dumpAttr(*I);
  }
  // Decls within functions are visited by the body
  if (!isa<FunctionDecl>(*D) && !isa<ObjCMethodDecl>(*D))
    dumpDeclContext(dyn_cast<DeclContext>(D));
}

void ASTDumper::VisitLabelDecl(LabelDecl *D) {
  dumpName(D);
}

void ASTDumper::VisitTypedefDecl(TypedefDecl *D) {
  dumpName(D);
  dumpType(D->getUnderlyingType());
  if (D->isModulePrivate())
    OS << " __module_private__";
}

void ASTDumper::VisitEnumDecl(EnumDecl *D) {
  if (D->isScoped()) {
    if (D->isScopedUsingClassTag())
      OS << " class";
    else
      OS << " struct";
  }
  dumpName(D);
  if (D->isModulePrivate())
    OS << " __module_private__";
  if (D->isFixed())
    dumpType(D->getIntegerType());
}

void ASTDumper::VisitRecordDecl(RecordDecl *D) {
  OS << ' ' << D->getKindName();
  dumpName(D);
  if (D->isModulePrivate())
    OS << " __module_private__";
}

void ASTDumper::VisitEnumConstantDecl(EnumConstantDecl *D) {
  dumpName(D);
  dumpType(D->getType());
  if (Expr *Init = D->getInitExpr())
    dumpStmt(Init);
}

void ASTDumper::VisitIndirectFieldDecl(IndirectFieldDecl *D) {
  dumpName(D);
  dumpType(D->getType());
  for (IndirectFieldDecl::chain_iterator I = D->chain_begin(),
       E = D->chain_end(); I != E; ++I)
    dumpDeclRef(*I);
}

void ASTDumper::VisitFunctionDecl(FunctionDecl *D) {
  dumpName(D);
  dumpType(D->getType());

  StorageClass SC = D->getStorageClassAsWritten();
  if (SC != SC_None)
    OS << ' ' << VarDecl::getStorageClassSpecifierString(SC);
  if (D->isInlineSpecified())
    OS << " inline";
  if (D->isVirtualAsWritten())
    OS << " virtual";
  if (D->isModulePrivate())
    OS << " __module_private__";

  if (D->isPure())
    OS << " pure";
  else if (D->isDeletedAsWritten())
    OS << " delete";

  if (const FunctionTemplateSpecializationInfo *FTSI =
      D->getTemplateSpecializationInfo())
    dumpTemplateArgumentList(*FTSI->TemplateArguments);

  for (llvm::ArrayRef<NamedDecl*>::iterator
       I = D->getDeclsInPrototypeScope().begin(),
       E = D->getDeclsInPrototypeScope().end(); I != E; ++I)
    dumpDecl(*I);

  for (FunctionDecl::param_iterator I = D->param_begin(), E = D->param_end();
       I != E; ++I)
    dumpDecl(*I);

  if (CXXConstructorDecl *C = dyn_cast<CXXConstructorDecl>(D))
    for (CXXConstructorDecl::init_const_iterator I = C->init_begin(),
         E = C->init_end(); I != E; ++I)
      dumpCXXCtorInitializer(*I);

  if (D->doesThisDeclarationHaveABody())
    dumpStmt(D->getBody());
}

void ASTDumper::VisitFieldDecl(FieldDecl *D) {
  dumpName(D);
  dumpType(D->getType());
  if (D->isMutable())
    OS << " mutable";
  if (D->isModulePrivate())
    OS << " __module_private__";
  if (D->isBitField())
    dumpStmt(D->getBitWidth());
  if (Expr *Init = D->getInClassInitializer())
    dumpStmt(Init);
}

void ASTDumper::VisitVarDecl(VarDecl *D) {
  dumpName(D);
  dumpType(D->getType());
  StorageClass SC = D->getStorageClassAsWritten();
  if (SC != SC_None)
    OS << ' ' << VarDecl::getStorageClassSpecifierString(SC);
  if (D->isThreadSpecified())
    OS << " __thread";
  if (D->isModulePrivate())
    OS << " __module_private__";
  if (D->isNRVOVariable())
    OS << " nrvo";
  if (D->hasInit())
    dumpStmt(D->getInit());
}

void ASTDumper::VisitFileScopeAsmDecl(FileScopeAsmDecl *D) {
  dumpStmt(D->getAsmString());
}

void ASTDumper::VisitImportDecl(ImportDecl *D) {
  OS << ' ' << D->getImportedModule()->getFullModuleName();
}

//===----------------------------------------------------------------------===//
// C++ Declarations
//===----------------------------------------------------------------------===//

void ASTDumper::VisitNamespaceDecl(NamespaceDecl *D) {
  dumpName(D);
  if (D->isInline())
    OS << " inline";
  if (!D->isOriginalNamespace())
    dumpDeclRef(D->getOriginalNamespace(), "original");
}

void ASTDumper::VisitUsingDirectiveDecl(UsingDirectiveDecl *D) {
  OS << ' ';
  dumpBareDeclRef(D->getNominatedNamespace());
}

void ASTDumper::VisitNamespaceAliasDecl(NamespaceAliasDecl *D) {
  dumpName(D);
  dumpDeclRef(D->getAliasedNamespace());
}

void ASTDumper::VisitTypeAliasDecl(TypeAliasDecl *D) {
  dumpName(D);
  dumpType(D->getUnderlyingType());
}

void ASTDumper::VisitTypeAliasTemplateDecl(TypeAliasTemplateDecl *D) {
  dumpName(D);
  dumpTemplateParameters(D->getTemplateParameters());
  dumpDecl(D->getTemplatedDecl());
}

void ASTDumper::VisitCXXRecordDecl(CXXRecordDecl *D) {
  VisitRecordDecl(D);
  if (!D->isCompleteDefinition())
    return;

  for (CXXRecordDecl::base_class_iterator I = D->bases_begin(),
       E = D->bases_end(); I != E; ++I) {
    IndentScope Indent(*this);
    if (I->isVirtual())
      OS << "virtual ";
    dumpAccessSpecifier(I->getAccessSpecifier());
    dumpType(I->getType());
    if (I->isPackExpansion())
      OS << "...";
  }
}

void ASTDumper::VisitStaticAssertDecl(StaticAssertDecl *D) {
  dumpStmt(D->getAssertExpr());
  dumpStmt(D->getMessage());
}

void ASTDumper::VisitFunctionTemplateDecl(FunctionTemplateDecl *D) {
  dumpName(D);
  dumpTemplateParameters(D->getTemplateParameters());
  dumpDecl(D->getTemplatedDecl());
  for (FunctionTemplateDecl::spec_iterator I = D->spec_begin(),
       E = D->spec_end(); I != E; ++I) {
    switch (I->getTemplateSpecializationKind()) {
    case TSK_Undeclared:
    case TSK_ImplicitInstantiation:
    case TSK_ExplicitInstantiationDeclaration:
    case TSK_ExplicitInstantiationDefinition:
      dumpDecl(*I);
      break;
    case TSK_ExplicitSpecialization:
      dumpDeclRef(*I);
      break;
    }
  }
}

void ASTDumper::VisitClassTemplateDecl(ClassTemplateDecl *D) {
  dumpName(D);
  dumpTemplateParameters(D->getTemplateParameters());
  dumpDecl(D->getTemplatedDecl());
  for (ClassTemplateDecl::spec_iterator I = D->spec_begin(), E = D->spec_end();
       I != E; ++I) {
    switch (I->getTemplateSpecializationKind()) {
    case TSK_Undeclared:
    case TSK_ImplicitInstantiation:
      dumpDecl(*I);
      break;
    case TSK_ExplicitSpecialization:
    case TSK_ExplicitInstantiationDeclaration:
    case TSK_ExplicitInstantiationDefinition:
      dumpDeclRef(*I);
      break;
    }
  }
}

void ASTDumper::VisitClassTemplateSpecializationDecl(
    ClassTemplateSpecializationDecl *D) {
  VisitCXXRecordDecl(D);
  dumpTemplateArgumentList(D->getTemplateArgs());
}

void ASTDumper::VisitClassTemplatePartialSpecializationDecl(
    ClassTemplatePartialSpecializationDecl *D) {
  VisitClassTemplateSpecializationDecl(D);
  dumpTemplateParameters(D->getTemplateParameters());
}

void ASTDumper::VisitClassScopeFunctionSpecializationDecl(
    ClassScopeFunctionSpecializationDecl *D) {
  dumpDeclRef(D->getSpecialization());
  if (D->hasExplicitTemplateArgs())
    dumpTemplateArgumentListInfo(D->templateArgs());
}

void ASTDumper::VisitTemplateTypeParmDecl(TemplateTypeParmDecl *D) {
  if (D->wasDeclaredWithTypename())
    OS << " typename";
  else
    OS << " class";
  if (D->isParameterPack())
    OS << " ...";
  dumpName(D);
  if (D->hasDefaultArgument())
    dumpType(D->getDefaultArgument());
}

void ASTDumper::VisitNonTypeTemplateParmDecl(NonTypeTemplateParmDecl *D) {
  dumpType(D->getType());
  if (D->isParameterPack())
    OS << " ...";
  dumpName(D);
  if (D->hasDefaultArgument())
    dumpStmt(D->getDefaultArgument());
}

void ASTDumper::VisitTemplateTemplateParmDecl(TemplateTemplateParmDecl *D) {
  if (D->isParameterPack())
    OS << " ...";
  dumpName(D);
  dumpTemplateParameters(D->getTemplateParameters());
  if (D->hasDefaultArgument())
    dumpTemplateArgumentLoc(D->getDefaultArgument());
}

void ASTDumper::VisitUsingDecl(UsingDecl *D) {
  OS << ' ';
  D->getQualifier()->print(OS, D->getASTContext().getPrintingPolicy());
  OS << D->getNameAsString();
}

void
ASTDumper::VisitUnresolvedUsingTypenameDecl(UnresolvedUsingTypenameDecl *D) {
  OS << ' ';
  D->getQualifier()->print(OS, D->getASTContext().getPrintingPolicy());
  OS << D->getNameAsString();
}

void ASTDumper::VisitUnresolvedUsingValueDecl(UnresolvedUsingValueDecl *D) {
  OS << ' ';
  D->getQualifier()->print(OS, D->getASTContext().getPrintingPolicy());
  OS << D->getNameAsString();
  dumpType(D->getType());
}

void ASTDumper::VisitUsingShadowDecl(UsingShadowDecl *D) {
  OS << ' ';
  dumpBareDeclRef(D->getTargetDecl());
}

void ASTDumper::VisitLinkageSpecDecl(LinkageSpecDecl *D) {
  switch (D->getLanguage()) {
  case LinkageSpecDecl::lang_c: OS << " C"; break;
  case LinkageSpecDecl::lang_cxx: OS << " C++"; break;
  }
}

void ASTDumper::VisitAccessSpecDecl(AccessSpecDecl *D) {
  OS << ' ';
  dumpAccessSpecifier(D->getAccess());
}

void ASTDumper::VisitFriendDecl(FriendDecl *D) {
  if (TypeSourceInfo *T = D->getFriendType())
    dumpType(T->getType());
  else
    dumpDecl(D->getFriendDecl());
}

//===----------------------------------------------------------------------===//
// Obj-C Declarations
//===----------------------------------------------------------------------===//

void ASTDumper::VisitObjCIvarDecl(ObjCIvarDecl *D) {
  dumpName(D);
  dumpType(D->getType());
  if (D->getSynthesize())
    OS << " synthesize";

  switch (D->getAccessControl()) {
  case ObjCIvarDecl::None:
    OS << " none";
    break;
  case ObjCIvarDecl::Private:
    OS << " private";
    break;
  case ObjCIvarDecl::Protected:
    OS << " protected";
    break;
  case ObjCIvarDecl::Public:
    OS << " public";
    break;
  case ObjCIvarDecl::Package:
    OS << " package";
    break;
  }
}

void ASTDumper::VisitObjCMethodDecl(ObjCMethodDecl *D) {
  if (D->isInstanceMethod())
    OS << " -";
  else
    OS << " +";
  dumpName(D);
  dumpType(D->getResultType());

  if (D->isThisDeclarationADefinition())
    dumpDeclContext(D);
  else {
    for (ObjCMethodDecl::param_iterator I = D->param_begin(),
         E = D->param_end(); I != E; ++I) {
      dumpDecl(*I);
    }
  }

  if (D->isVariadic()) {
    IndentScope Indent(*this);
    OS << "...";
  }

  if (D->hasBody())
    dumpStmt(D->getBody());
}

void ASTDumper::VisitObjCCategoryDecl(ObjCCategoryDecl *D) {
  dumpName(D);
  dumpDeclRef(D->getClassInterface());
  dumpDeclRef(D->getImplementation());
  for (ObjCCategoryDecl::protocol_iterator I = D->protocol_begin(),
       E = D->protocol_end(); I != E; ++I)
    dumpDeclRef(*I);
}

void ASTDumper::VisitObjCCategoryImplDecl(ObjCCategoryImplDecl *D) {
  dumpName(D);
  dumpDeclRef(D->getClassInterface());
  dumpDeclRef(D->getCategoryDecl());
}

void ASTDumper::VisitObjCProtocolDecl(ObjCProtocolDecl *D) {
  dumpName(D);
  for (ObjCProtocolDecl::protocol_iterator I = D->protocol_begin(),
       E = D->protocol_end(); I != E; ++I)
    dumpDeclRef(*I);
}

void ASTDumper::VisitObjCInterfaceDecl(ObjCInterfaceDecl *D) {
  dumpName(D);
  dumpDeclRef(D->getSuperClass(), "super");
  dumpDeclRef(D->getImplementation());
  for (ObjCInterfaceDecl::protocol_iterator I = D->protocol_begin(),
       E = D->protocol_end(); I != E; ++I)
    dumpDeclRef(*I);
}

void ASTDumper::VisitObjCImplementationDecl(ObjCImplementationDecl *D) {
  dumpName(D);
  dumpDeclRef(D->getSuperClass(), "super");
  dumpDeclRef(D->getClassInterface());
  for (ObjCImplementationDecl::init_iterator I = D->init_begin(),
       E = D->init_end(); I != E; ++I)
    dumpCXXCtorInitializer(*I);
}

void ASTDumper::VisitObjCCompatibleAliasDecl(ObjCCompatibleAliasDecl *D) {
  dumpName(D);
  dumpDeclRef(D->getClassInterface());
}

void ASTDumper::VisitObjCPropertyDecl(ObjCPropertyDecl *D) {
  dumpName(D);
  dumpType(D->getType());

  if (D->getPropertyImplementation() == ObjCPropertyDecl::Required)
    OS << " required";
  else if (D->getPropertyImplementation() == ObjCPropertyDecl::Optional)
    OS << " optional";

  ObjCPropertyDecl::PropertyAttributeKind Attrs = D->getPropertyAttributes();
  if (Attrs != ObjCPropertyDecl::OBJC_PR_noattr) {
    if (Attrs & ObjCPropertyDecl::OBJC_PR_readonly)
      OS << " readonly";
    if (Attrs & ObjCPropertyDecl::OBJC_PR_assign)
      OS << " assign";
    if (Attrs & ObjCPropertyDecl::OBJC_PR_readwrite)
      OS << " readwrite";
    if (Attrs & ObjCPropertyDecl::OBJC_PR_retain)
      OS << " retain";
    if (Attrs & ObjCPropertyDecl::OBJC_PR_copy)
      OS << " copy";
    if (Attrs & ObjCPropertyDecl::OBJC_PR_nonatomic)
      OS << " nonatomic";
    if (Attrs & ObjCPropertyDecl::OBJC_PR_atomic)
      OS << " atomic";
    if (Attrs & ObjCPropertyDecl::OBJC_PR_weak)
      OS << " weak";
    if (Attrs & ObjCPropertyDecl::OBJC_PR_strong)
      OS << " strong";
    if (Attrs & ObjCPropertyDecl::OBJC_PR_unsafe_unretained)
      OS << " unsafe_unretained";
    if (Attrs & ObjCPropertyDecl::OBJC_PR_getter)
      dumpDeclRef(D->getGetterMethodDecl(), "getter");
    if (Attrs & ObjCPropertyDecl::OBJC_PR_setter)
      dumpDeclRef(D->getSetterMethodDecl(), "setter");
  }
}

void ASTDumper::VisitObjCPropertyImplDecl(ObjCPropertyImplDecl *D) {
  dumpName(D->getPropertyDecl());
  if (D->getPropertyImplementation() == ObjCPropertyImplDecl::Synthesize)
    OS << " synthesize";
  else
    OS << " dynamic";
  dumpDeclRef(D->getPropertyDecl());
  dumpDeclRef(D->getPropertyIvarDecl());
}

void ASTDumper::VisitBlockDecl(BlockDecl *D) {
  for (BlockDecl::param_iterator I = D->param_begin(), E = D->param_end();
       I != E; ++I)
    dumpDecl(*I);

  if (D->isVariadic()) {
    IndentScope Indent(*this);
    OS << "...";
  }

  if (D->capturesCXXThis()) {
    IndentScope Indent(*this);
    OS << "capture this";
  }
  for (BlockDecl::capture_iterator I = D->capture_begin(),
       E = D->capture_end(); I != E; ++I) {
    IndentScope Indent(*this);
    OS << "capture";
    if (I->isByRef())
      OS << " byref";
    if (I->isNested())
      OS << " nested";
    if (I->getVariable()) {
      OS << ' ';
      dumpBareDeclRef(I->getVariable());
    }
    if (I->hasCopyExpr())
      dumpStmt(I->getCopyExpr());
  }

  dumpStmt(D->getBody());
}

//===----------------------------------------------------------------------===//
//  Stmt dumping methods.
//===----------------------------------------------------------------------===//

void ASTDumper::dumpStmt(Stmt *S) {
  IndentScope Indent(*this);

  if (!S) {
    OS << "<<<NULL>>>";
    return;
  }

  if (DeclStmt *DS = dyn_cast<DeclStmt>(S)) {
    VisitDeclStmt(DS);
    return;
  }

  StmtVisitor<ASTDumper>::Visit(S);
  for (Stmt::child_range CI = S->children(); CI; ++CI)
    dumpStmt(*CI);
}

void ASTDumper::VisitStmt(Stmt *Node) {
  OS << Node->getStmtClassName();
  dumpPointer(Node);
  dumpSourceRange(Node->getSourceRange());
}

void ASTDumper::VisitDeclStmt(DeclStmt *Node) {
  VisitStmt(Node);
  for (DeclStmt::decl_iterator I = Node->decl_begin(), E = Node->decl_end();
       I != E; ++I)
    dumpDecl(*I);
}

void ASTDumper::VisitAttributedStmt(AttributedStmt *Node) {
  VisitStmt(Node);
  for (ArrayRef<const Attr*>::iterator I = Node->getAttrs().begin(),
       E = Node->getAttrs().end(); I != E; ++I)
    dumpAttr(*I);
}

void ASTDumper::VisitLabelStmt(LabelStmt *Node) {
  VisitStmt(Node);
  OS << " '" << Node->getName() << "'";
}

void ASTDumper::VisitGotoStmt(GotoStmt *Node) {
  VisitStmt(Node);
  OS << " '" << Node->getLabel()->getName() << "'";
  dumpPointer(Node->getLabel());
}

//===----------------------------------------------------------------------===//
//  Expr dumping methods.
//===----------------------------------------------------------------------===//

void ASTDumper::VisitExpr(Expr *Node) {
  VisitStmt(Node);
  dumpType(Node->getType());

  switch (Node->getValueKind()) {
  case VK_RValue:
    break;
  case VK_LValue:
    OS << " lvalue";
    break;
  case VK_XValue:
    OS << " xvalue";
    break;
  }

  switch (Node->getObjectKind()) {
  case OK_Ordinary:
    break;
  case OK_BitField:
    OS << " bitfield";
    break;
  case OK_ObjCProperty:
    OS << " objcproperty";
    break;
  case OK_ObjCSubscript:
    OS << " objcsubscript";
    break;
  case OK_VectorComponent:
    OS << " vectorcomponent";
    break;
  }
}

static void dumpBasePath(raw_ostream &OS, CastExpr *Node) {
  if (Node->path_empty())
    return;

  OS << " (";
  bool First = true;
  for (CastExpr::path_iterator
         I = Node->path_begin(), E = Node->path_end(); I != E; ++I) {
    const CXXBaseSpecifier *Base = *I;
    if (!First)
      OS << " -> ";

    const CXXRecordDecl *RD =
    cast<CXXRecordDecl>(Base->getType()->getAs<RecordType>()->getDecl());

    if (Base->isVirtual())
      OS << "virtual ";
    OS << RD->getName();
    First = false;
  }

  OS << ')';
}

void ASTDumper::VisitCastExpr(CastExpr *Node) {
  VisitExpr(Node);
  OS << " <" << Node->getCastKindName();
  dumpBasePath(OS, Node);
  OS << ">";
}

void ASTDumper::VisitDeclRefExpr(DeclRefExpr *Node) {
  VisitExpr(Node);

  OS << " ";
  dumpBareDeclRef(Node->getDecl());
  if (Node->getDecl() != Node->getFoundDecl()) {
    OS << " (";
    dumpBareDeclRef(Node->getFoundDecl());
    OS << ")";
  }
}

void ASTDumper::VisitUnresolvedLookupExpr(UnresolvedLookupExpr *Node) {
  VisitExpr(Node);
  OS << " (";
  if (!Node->requiresADL())
    OS << "no ";
  OS << "ADL) = '" << Node->getName() << '\'';

  UnresolvedLookupExpr::decls_iterator
    I = Node->decls_begin(), E = Node->decls_end();
  if (I == E)
    OS << " empty";
  for (; I != E; ++I)
    dumpPointer(*I);
}

void ASTDumper::VisitObjCIvarRefExpr(ObjCIvarRefExpr *Node) {
  VisitExpr(Node);

  OS << " " << Node->getDecl()->getDeclKindName()
     << "Decl='" << *Node->getDecl() << "'";
  dumpPointer(Node->getDecl());
  if (Node->isFreeIvar())
    OS << " isFreeIvar";
}

void ASTDumper::VisitPredefinedExpr(PredefinedExpr *Node) {
  VisitExpr(Node);
  switch (Node->getIdentType()) {
  default: llvm_unreachable("unknown case");
  case PredefinedExpr::Func:           OS <<  " __func__"; break;
  case PredefinedExpr::Function:       OS <<  " __FUNCTION__"; break;
  case PredefinedExpr::LFunction:      OS <<  " L__FUNCTION__"; break;
  case PredefinedExpr::PrettyFunction: OS <<  " __PRETTY_FUNCTION__";break;
  }
}

void ASTDumper::VisitCharacterLiteral(CharacterLiteral *Node) {
  VisitExpr(Node);
  OS << " " << Node->getValue();
}

void ASTDumper::VisitIntegerLiteral(IntegerLiteral *Node) {
  VisitExpr(Node);

  bool isSigned = Node->getType()->isSignedIntegerType();
  OS << " " << Node->getValue().toString(10, isSigned);
}

void ASTDumper::VisitFloatingLiteral(FloatingLiteral *Node) {
  VisitExpr(Node);
  OS << " " << Node->getValueAsApproximateDouble();
}

void ASTDumper::VisitStringLiteral(StringLiteral *Str) {
  VisitExpr(Str);
  OS << " ";
  Str->outputString(OS);
}

void ASTDumper::VisitUnaryOperator(UnaryOperator *Node) {
  VisitExpr(Node);
  OS << " " << (Node->isPostfix() ? "postfix" : "prefix")
     << " '" << UnaryOperator::getOpcodeStr(Node->getOpcode()) << "'";
}

void ASTDumper::VisitUnaryExprOrTypeTraitExpr(UnaryExprOrTypeTraitExpr *Node) {
  VisitExpr(Node);
  switch(Node->getKind()) {
  case UETT_SizeOf:
    OS << " sizeof";
    break;
  case UETT_AlignOf:
    OS << " alignof";
    break;
  case UETT_VecStep:
    OS << " vec_step";
    break;
  }
  if (Node->isArgumentType())
    dumpType(Node->getArgumentType());
}

void ASTDumper::VisitMemberExpr(MemberExpr *Node) {
  VisitExpr(Node);
  OS << " " << (Node->isArrow() ? "->" : ".") << *Node->getMemberDecl();
  dumpPointer(Node->getMemberDecl());
}

void ASTDumper::VisitExtVectorElementExpr(ExtVectorElementExpr *Node) {
  VisitExpr(Node);
  OS << " " << Node->getAccessor().getNameStart();
}

void ASTDumper::VisitBinaryOperator(BinaryOperator *Node) {
  VisitExpr(Node);
  OS << " '" << BinaryOperator::getOpcodeStr(Node->getOpcode()) << "'";
}

void ASTDumper::VisitCompoundAssignOperator(CompoundAssignOperator *Node) {
  VisitExpr(Node);
  OS << " '" << BinaryOperator::getOpcodeStr(Node->getOpcode())
     << "' ComputeLHSTy=";
  dumpBareType(Node->getComputationLHSType());
  OS << " ComputeResultTy=";
  dumpBareType(Node->getComputationResultType());
}

void ASTDumper::VisitBlockExpr(BlockExpr *Node) {
  VisitExpr(Node);
  dumpDecl(Node->getBlockDecl());
}

void ASTDumper::VisitOpaqueValueExpr(OpaqueValueExpr *Node) {
  VisitExpr(Node);

  if (Expr *Source = Node->getSourceExpr())
    dumpStmt(Source);
}

// GNU extensions.

void ASTDumper::VisitAddrLabelExpr(AddrLabelExpr *Node) {
  VisitExpr(Node);
  OS << " " << Node->getLabel()->getName();
  dumpPointer(Node->getLabel());
}

//===----------------------------------------------------------------------===//
// C++ Expressions
//===----------------------------------------------------------------------===//

void ASTDumper::VisitCXXNamedCastExpr(CXXNamedCastExpr *Node) {
  VisitExpr(Node);
  OS << " " << Node->getCastName()
     << "<" << Node->getTypeAsWritten().getAsString() << ">"
     << " <" << Node->getCastKindName();
  dumpBasePath(OS, Node);
  OS << ">";
}

void ASTDumper::VisitCXXBoolLiteralExpr(CXXBoolLiteralExpr *Node) {
  VisitExpr(Node);
  OS << " " << (Node->getValue() ? "true" : "false");
}

void ASTDumper::VisitCXXThisExpr(CXXThisExpr *Node) {
  VisitExpr(Node);
  OS << " this";
}

void ASTDumper::VisitCXXFunctionalCastExpr(CXXFunctionalCastExpr *Node) {
  VisitExpr(Node);
  OS << " functional cast to " << Node->getTypeAsWritten().getAsString()
     << " <" << Node->getCastKindName() << ">";
}

void ASTDumper::VisitCXXConstructExpr(CXXConstructExpr *Node) {
  VisitExpr(Node);
  CXXConstructorDecl *Ctor = Node->getConstructor();
  dumpType(Ctor->getType());
  if (Node->isElidable())
    OS << " elidable";
  if (Node->requiresZeroInitialization())
    OS << " zeroing";
}

void ASTDumper::VisitCXXBindTemporaryExpr(CXXBindTemporaryExpr *Node) {
  VisitExpr(Node);
  OS << " ";
  dumpCXXTemporary(Node->getTemporary());
}

void ASTDumper::VisitExprWithCleanups(ExprWithCleanups *Node) {
  VisitExpr(Node);
  for (unsigned i = 0, e = Node->getNumObjects(); i != e; ++i)
    dumpDeclRef(Node->getObject(i), "cleanup");
}

void ASTDumper::dumpCXXTemporary(CXXTemporary *Temporary) {
  OS << "(CXXTemporary";
  dumpPointer(Temporary);
  OS << ")";
}

//===----------------------------------------------------------------------===//
// Obj-C Expressions
//===----------------------------------------------------------------------===//

void ASTDumper::VisitObjCMessageExpr(ObjCMessageExpr *Node) {
  VisitExpr(Node);
  OS << " selector=" << Node->getSelector().getAsString();
  switch (Node->getReceiverKind()) {
  case ObjCMessageExpr::Instance:
    break;

  case ObjCMessageExpr::Class:
    OS << " class=";
    dumpBareType(Node->getClassReceiver());
    break;

  case ObjCMessageExpr::SuperInstance:
    OS << " super (instance)";
    break;

  case ObjCMessageExpr::SuperClass:
    OS << " super (class)";
    break;
  }
}

void ASTDumper::VisitObjCBoxedExpr(ObjCBoxedExpr *Node) {
  VisitExpr(Node);
  OS << " selector=" << Node->getBoxingMethod()->getSelector().getAsString();
}

void ASTDumper::VisitObjCAtCatchStmt(ObjCAtCatchStmt *Node) {
  VisitStmt(Node);
  if (VarDecl *CatchParam = Node->getCatchParamDecl())
    dumpDecl(CatchParam);
  else
    OS << " catch all";
}

void ASTDumper::VisitObjCEncodeExpr(ObjCEncodeExpr *Node) {
  VisitExpr(Node);
  dumpType(Node->getEncodedType());
}

void ASTDumper::VisitObjCSelectorExpr(ObjCSelectorExpr *Node) {
  VisitExpr(Node);

  OS << " " << Node->getSelector().getAsString();
}

void ASTDumper::VisitObjCProtocolExpr(ObjCProtocolExpr *Node) {
  VisitExpr(Node);

  OS << ' ' << *Node->getProtocol();
}

void ASTDumper::VisitObjCPropertyRefExpr(ObjCPropertyRefExpr *Node) {
  VisitExpr(Node);
  if (Node->isImplicitProperty()) {
    OS << " Kind=MethodRef Getter=\"";
    if (Node->getImplicitPropertyGetter())
      OS << Node->getImplicitPropertyGetter()->getSelector().getAsString();
    else
      OS << "(null)";

    OS << "\" Setter=\"";
    if (ObjCMethodDecl *Setter = Node->getImplicitPropertySetter())
      OS << Setter->getSelector().getAsString();
    else
      OS << "(null)";
    OS << "\"";
  } else {
    OS << " Kind=PropertyRef Property=\"" << *Node->getExplicitProperty() <<'"';
  }

  if (Node->isSuperReceiver())
    OS << " super";

  OS << " Messaging=";
  if (Node->isMessagingGetter() && Node->isMessagingSetter())
    OS << "Getter&Setter";
  else if (Node->isMessagingGetter())
    OS << "Getter";
  else if (Node->isMessagingSetter())
    OS << "Setter";
}

void ASTDumper::VisitObjCSubscriptRefExpr(ObjCSubscriptRefExpr *Node) {
  VisitExpr(Node);
  if (Node->isArraySubscriptRefExpr())
    OS << " Kind=ArraySubscript GetterForArray=\"";
  else
    OS << " Kind=DictionarySubscript GetterForDictionary=\"";
  if (Node->getAtIndexMethodDecl())
    OS << Node->getAtIndexMethodDecl()->getSelector().getAsString();
  else
    OS << "(null)";

  if (Node->isArraySubscriptRefExpr())
    OS << "\" SetterForArray=\"";
  else
    OS << "\" SetterForDictionary=\"";
  if (Node->setAtIndexMethodDecl())
    OS << Node->setAtIndexMethodDecl()->getSelector().getAsString();
  else
    OS << "(null)";
}

void ASTDumper::VisitObjCBoolLiteralExpr(ObjCBoolLiteralExpr *Node) {
  VisitExpr(Node);
  OS << " " << (Node->getValue() ? "__objc_yes" : "__objc_no");
}

//===----------------------------------------------------------------------===//
// Decl method implementations
//===----------------------------------------------------------------------===//

void Decl::dump() const {
  dump(llvm::errs());
}

void Decl::dump(raw_ostream &OS) const {
  ASTDumper P(&getASTContext().getSourceManager(), OS);
  P.dumpDecl(const_cast<Decl*>(this));
}

//===----------------------------------------------------------------------===//
// Stmt method implementations
//===----------------------------------------------------------------------===//

void Stmt::dump(SourceManager &SM) const {
  dump(llvm::errs(), SM);
}

void Stmt::dump(raw_ostream &OS, SourceManager &SM) const {
  ASTDumper P(&SM, OS);
  P.dumpStmt(const_cast<Stmt*>(this));
}

void Stmt::dump() const {
  ASTDumper P(0, llvm::errs());
  P.dumpStmt(const_cast<Stmt*>(this));
}
