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

#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/Support/raw_ostream.h"
using namespace clang;

//===----------------------------------------------------------------------===//
// ASTDumper Visitor
//===----------------------------------------------------------------------===//

namespace  {
  class ASTDumper : public StmtVisitor<ASTDumper> {
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
    void dumpSourceRange(const Stmt *Node);
    void dumpLocation(SourceLocation Loc);
    void dumpType(QualType T);
    void dumpDeclRef(Decl *node);

    // Stmts.
    void VisitStmt(Stmt *Node);
    void VisitDeclStmt(DeclStmt *Node);
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

void ASTDumper::dumpSourceRange(const Stmt *Node) {
  // Can't translate locations if a SourceManager isn't available.
  if (!SM)
    return;

  // TODO: If the parent expression is available, we can print a delta vs its
  // location.
  SourceRange R = Node->getSourceRange();

  OS << " <";
  dumpLocation(R.getBegin());
  if (R.getBegin() != R.getEnd()) {
    OS << ", ";
    dumpLocation(R.getEnd());
  }
  OS << ">";

  // <t2.c:123:421[blah], t2.c:412:321>

}

void ASTDumper::dumpType(QualType T) {
  SplitQualType T_split = T.split();
  OS << "'" << QualType::getAsString(T_split) << "'";

  if (!T.isNull()) {
    // If the type is sugared, also dump a (shallow) desugared type.
    SplitQualType D_split = T.getSplitDesugaredType();
    if (T_split != D_split)
      OS << ":'" << QualType::getAsString(D_split) << "'";
  }
}

void ASTDumper::dumpDeclRef(Decl *D) {
  OS << D->getDeclKindName() << ' ' << (void*) D;

  if (NamedDecl *ND = dyn_cast<NamedDecl>(D)) {
    OS << " '";
    ND->getDeclName().printName(OS);
    OS << "'";
  }

  if (ValueDecl *VD = dyn_cast<ValueDecl>(D)) {
    OS << ' ';
    dumpType(VD->getType());
  }
}

//===----------------------------------------------------------------------===//
//  Decl dumping methods.
//===----------------------------------------------------------------------===//

void ASTDumper::dumpDecl(Decl *D) {
  // FIXME: Need to complete/beautify this... this code simply shows the
  // nodes are where they need to be.
  if (TypedefDecl *localType = dyn_cast<TypedefDecl>(D)) {
    OS << "\"typedef " << localType->getUnderlyingType().getAsString()
       << ' ' << *localType << '"';
  } else if (TypeAliasDecl *localType = dyn_cast<TypeAliasDecl>(D)) {
    OS << "\"using " << *localType << " = "
       << localType->getUnderlyingType().getAsString() << '"';
  } else if (ValueDecl *VD = dyn_cast<ValueDecl>(D)) {
    OS << "\"";
    // Emit storage class for vardecls.
    if (VarDecl *V = dyn_cast<VarDecl>(VD)) {
      if (V->getStorageClass() != SC_None)
        OS << VarDecl::getStorageClassSpecifierString(V->getStorageClass())
           << " ";
    }

    std::string Name = VD->getNameAsString();
    VD->getType().getAsStringInternal(Name,
                          PrintingPolicy(VD->getASTContext().getLangOpts()));
    OS << Name;

    // If this is a vardecl with an initializer, emit it.
    if (VarDecl *V = dyn_cast<VarDecl>(VD)) {
      if (V->getInit()) {
        OS << " =";
        dumpStmt(V->getInit());
      }
    }
    OS << '"';
  } else if (TagDecl *TD = dyn_cast<TagDecl>(D)) {
    // print a free standing tag decl (e.g. "struct x;").
    const char *tagname;
    if (const IdentifierInfo *II = TD->getIdentifier())
      tagname = II->getNameStart();
    else
      tagname = "<anonymous>";
    OS << '"' << TD->getKindName() << ' ' << tagname << ";\"";
    // FIXME: print tag bodies.
  } else if (UsingDirectiveDecl *UD = dyn_cast<UsingDirectiveDecl>(D)) {
    // print using-directive decl (e.g. "using namespace x;")
    const char *ns;
    if (const IdentifierInfo *II = UD->getNominatedNamespace()->getIdentifier())
      ns = II->getNameStart();
    else
      ns = "<anonymous>";
    OS << '"' << UD->getDeclKindName() << ns << ";\"";
  } else if (UsingDecl *UD = dyn_cast<UsingDecl>(D)) {
    // print using decl (e.g. "using std::string;")
    const char *tn = UD->isTypeName() ? "typename " : "";
    OS << '"' << UD->getDeclKindName() << tn;
    UD->getQualifier()->print(OS,
                        PrintingPolicy(UD->getASTContext().getLangOpts()));
    OS << ";\"";
  } else if (LabelDecl *LD = dyn_cast<LabelDecl>(D)) {
    OS << "label " << *LD;
  } else if (StaticAssertDecl *SAD = dyn_cast<StaticAssertDecl>(D)) {
    OS << "\"static_assert(";
    dumpStmt(SAD->getAssertExpr());
    OS << ",";
    dumpStmt(SAD->getMessage());
    OS << ");\"";
  } else {
    llvm_unreachable("Unexpected decl");
  }
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

  Visit(S);
  for (Stmt::child_range CI = S->children(); CI; ++CI)
    dumpStmt(*CI);
}

void ASTDumper::VisitStmt(Stmt *Node) {
  OS << Node->getStmtClassName() << " " << (const void *)Node;
  dumpSourceRange(Node);
}

void ASTDumper::VisitDeclStmt(DeclStmt *Node) {
  VisitStmt(Node);
  for (DeclStmt::decl_iterator DI = Node->decl_begin(), DE = Node->decl_end();
       DI != DE; ++DI) {
    IndentScope Indent(*this);
    Decl *D = *DI;
    OS << (void*) D << " ";
    dumpDecl(D);
  }
}

void ASTDumper::VisitLabelStmt(LabelStmt *Node) {
  VisitStmt(Node);
  OS << " '" << Node->getName() << "'";
}

void ASTDumper::VisitGotoStmt(GotoStmt *Node) {
  VisitStmt(Node);
  OS << " '" << Node->getLabel()->getName()
     << "':" << (void*)Node->getLabel();
}

//===----------------------------------------------------------------------===//
//  Expr dumping methods.
//===----------------------------------------------------------------------===//

void ASTDumper::VisitExpr(Expr *Node) {
  VisitStmt(Node);
  OS << ' ';
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
  dumpDeclRef(Node->getDecl());
  if (Node->getDecl() != Node->getFoundDecl()) {
    OS << " (";
    dumpDeclRef(Node->getFoundDecl());
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
    OS << " " << (void*) *I;
}

void ASTDumper::VisitObjCIvarRefExpr(ObjCIvarRefExpr *Node) {
  VisitExpr(Node);

  OS << " " << Node->getDecl()->getDeclKindName()
     << "Decl='" << *Node->getDecl()
     << "' " << (void*)Node->getDecl();
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
    OS << " sizeof ";
    break;
  case UETT_AlignOf:
    OS << " alignof ";
    break;
  case UETT_VecStep:
    OS << " vec_step ";
    break;
  }
  if (Node->isArgumentType())
    dumpType(Node->getArgumentType());
}

void ASTDumper::VisitMemberExpr(MemberExpr *Node) {
  VisitExpr(Node);
  OS << " " << (Node->isArrow() ? "->" : ".")
     << *Node->getMemberDecl() << ' '
     << (void*)Node->getMemberDecl();
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
  dumpType(Node->getComputationLHSType());
  OS << " ComputeResultTy=";
  dumpType(Node->getComputationResultType());
}

void ASTDumper::VisitBlockExpr(BlockExpr *Node) {
  VisitExpr(Node);

  BlockDecl *block = Node->getBlockDecl();
  OS << " decl=" << block;

  if (block->capturesCXXThis()) {
    IndentScope Indent(*this);
    OS << "capture this";
  }
  for (BlockDecl::capture_iterator
         i = block->capture_begin(), e = block->capture_end(); i != e; ++i) {
    IndentScope Indent(*this);
    OS << "capture ";
    if (i->isByRef())
      OS << "byref ";
    if (i->isNested())
      OS << "nested ";
    if (i->getVariable())
      dumpDeclRef(i->getVariable());
    if (i->hasCopyExpr())
      dumpStmt(i->getCopyExpr());
  }

  dumpStmt(block->getBody());
}

void ASTDumper::VisitOpaqueValueExpr(OpaqueValueExpr *Node) {
  VisitExpr(Node);

  if (Expr *Source = Node->getSourceExpr())
    dumpStmt(Source);
}

// GNU extensions.

void ASTDumper::VisitAddrLabelExpr(AddrLabelExpr *Node) {
  VisitExpr(Node);
  OS << " " << Node->getLabel()->getName()
     << " " << (void*)Node->getLabel();
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
  for (unsigned i = 0, e = Node->getNumObjects(); i != e; ++i) {
    IndentScope Indent(*this);
    OS << "cleanup ";
    dumpDeclRef(Node->getObject(i));
  }
}

void ASTDumper::dumpCXXTemporary(CXXTemporary *Temporary) {
  OS << "(CXXTemporary " << (void *)Temporary << ")";
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
    dumpType(Node->getClassReceiver());
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
  if (VarDecl *CatchParam = Node->getCatchParamDecl()) {
    OS << " catch parm = ";
    dumpDecl(CatchParam);
  } else {
    OS << " catch all";
  }
}

void ASTDumper::VisitObjCEncodeExpr(ObjCEncodeExpr *Node) {
  VisitExpr(Node);
  OS << " ";
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
