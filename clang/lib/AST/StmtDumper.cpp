//===--- StmtDumper.cpp - Dumping implementation for Stmt ASTs ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Stmt::dump/Stmt::print methods, which dump out the
// AST in a form that exposes type details and other fields.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/StmtVisitor.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/Support/raw_ostream.h"
using namespace clang;

//===----------------------------------------------------------------------===//
// StmtDumper Visitor
//===----------------------------------------------------------------------===//

namespace  {
  class StmtDumper : public StmtVisitor<StmtDumper> {
    SourceManager *SM;
    llvm::raw_ostream &OS;
    unsigned IndentLevel;

    /// MaxDepth - When doing a normal dump (not dumpAll) we only want to dump
    /// the first few levels of an AST.  This keeps track of how many ast levels
    /// are left.
    unsigned MaxDepth;

    /// LastLocFilename/LastLocLine - Keep track of the last location we print
    /// out so that we can print out deltas from then on out.
    const char *LastLocFilename;
    unsigned LastLocLine;

  public:
    StmtDumper(SourceManager *sm, llvm::raw_ostream &os, unsigned maxDepth)
      : SM(sm), OS(os), IndentLevel(0-1), MaxDepth(maxDepth) {
      LastLocFilename = "";
      LastLocLine = ~0U;
    }

    void DumpSubTree(Stmt *S) {
      // Prune the recursion if not using dump all.
      if (MaxDepth == 0) return;

      ++IndentLevel;
      if (S) {
        if (DeclStmt* DS = dyn_cast<DeclStmt>(S))
          VisitDeclStmt(DS);
        else {
          Visit(S);

          // Print out children.
          Stmt::child_iterator CI = S->child_begin(), CE = S->child_end();
          if (CI != CE) {
            while (CI != CE) {
              OS << '\n';
              DumpSubTree(*CI++);
            }
          }
          OS << ')';
        }
      } else {
        Indent();
        OS << "<<<NULL>>>";
      }
      --IndentLevel;
    }

    void DumpDeclarator(Decl *D);

    void Indent() const {
      for (int i = 0, e = IndentLevel; i < e; ++i)
        OS << "  ";
    }

    void DumpType(QualType T) {
      OS << "'" << T.getAsString() << "'";

      if (!T.isNull()) {
        // If the type is sugared, also dump a (shallow) desugared type.
        QualType Simplified = T.getDesugaredType();
        if (Simplified != T)
          OS << ":'" << Simplified.getAsString() << "'";
      }
    }
    void DumpStmt(const Stmt *Node) {
      Indent();
      OS << "(" << Node->getStmtClassName()
         << " " << (void*)Node;
      DumpSourceRange(Node);
    }
    void DumpExpr(const Expr *Node) {
      DumpStmt(Node);
      OS << ' ';
      DumpType(Node->getType());
    }
    void DumpSourceRange(const Stmt *Node);
    void DumpLocation(SourceLocation Loc);

    // Stmts.
    void VisitStmt(Stmt *Node);
    void VisitDeclStmt(DeclStmt *Node);
    void VisitLabelStmt(LabelStmt *Node);
    void VisitGotoStmt(GotoStmt *Node);

    // Exprs
    void VisitExpr(Expr *Node);
    void VisitCastExpr(CastExpr *Node);
    void VisitImplicitCastExpr(ImplicitCastExpr *Node);
    void VisitDeclRefExpr(DeclRefExpr *Node);
    void VisitPredefinedExpr(PredefinedExpr *Node);
    void VisitCharacterLiteral(CharacterLiteral *Node);
    void VisitIntegerLiteral(IntegerLiteral *Node);
    void VisitFloatingLiteral(FloatingLiteral *Node);
    void VisitStringLiteral(StringLiteral *Str);
    void VisitUnaryOperator(UnaryOperator *Node);
    void VisitSizeOfAlignOfExpr(SizeOfAlignOfExpr *Node);
    void VisitMemberExpr(MemberExpr *Node);
    void VisitExtVectorElementExpr(ExtVectorElementExpr *Node);
    void VisitBinaryOperator(BinaryOperator *Node);
    void VisitCompoundAssignOperator(CompoundAssignOperator *Node);
    void VisitAddrLabelExpr(AddrLabelExpr *Node);
    void VisitTypesCompatibleExpr(TypesCompatibleExpr *Node);

    // C++
    void VisitCXXNamedCastExpr(CXXNamedCastExpr *Node);
    void VisitCXXBoolLiteralExpr(CXXBoolLiteralExpr *Node);
    void VisitCXXThisExpr(CXXThisExpr *Node);
    void VisitCXXFunctionalCastExpr(CXXFunctionalCastExpr *Node);
    void VisitCXXConstructExpr(CXXConstructExpr *Node);
    void VisitCXXBindTemporaryExpr(CXXBindTemporaryExpr *Node);
    void VisitCXXExprWithTemporaries(CXXExprWithTemporaries *Node);
    void VisitUnresolvedLookupExpr(UnresolvedLookupExpr *Node);
    void DumpCXXTemporary(CXXTemporary *Temporary);

    // ObjC
    void VisitObjCAtCatchStmt(ObjCAtCatchStmt *Node);
    void VisitObjCEncodeExpr(ObjCEncodeExpr *Node);
    void VisitObjCMessageExpr(ObjCMessageExpr* Node);
    void VisitObjCSelectorExpr(ObjCSelectorExpr *Node);
    void VisitObjCProtocolExpr(ObjCProtocolExpr *Node);
    void VisitObjCPropertyRefExpr(ObjCPropertyRefExpr *Node);
    void VisitObjCImplicitSetterGetterRefExpr(
                                          ObjCImplicitSetterGetterRefExpr *Node);
    void VisitObjCIvarRefExpr(ObjCIvarRefExpr *Node);
    void VisitObjCSuperExpr(ObjCSuperExpr *Node);
  };
}

//===----------------------------------------------------------------------===//
//  Utilities
//===----------------------------------------------------------------------===//

void StmtDumper::DumpLocation(SourceLocation Loc) {
  SourceLocation SpellingLoc = SM->getSpellingLoc(Loc);

  if (SpellingLoc.isInvalid()) {
    OS << "<invalid sloc>";
    return;
  }

  // The general format we print out is filename:line:col, but we drop pieces
  // that haven't changed since the last loc printed.
  PresumedLoc PLoc = SM->getPresumedLoc(SpellingLoc);

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

void StmtDumper::DumpSourceRange(const Stmt *Node) {
  // Can't translate locations if a SourceManager isn't available.
  if (SM == 0) return;

  // TODO: If the parent expression is available, we can print a delta vs its
  // location.
  SourceRange R = Node->getSourceRange();

  OS << " <";
  DumpLocation(R.getBegin());
  if (R.getBegin() != R.getEnd()) {
    OS << ", ";
    DumpLocation(R.getEnd());
  }
  OS << ">";

  // <t2.c:123:421[blah], t2.c:412:321>

}


//===----------------------------------------------------------------------===//
//  Stmt printing methods.
//===----------------------------------------------------------------------===//

void StmtDumper::VisitStmt(Stmt *Node) {
  DumpStmt(Node);
}

void StmtDumper::DumpDeclarator(Decl *D) {
  // FIXME: Need to complete/beautify this... this code simply shows the
  // nodes are where they need to be.
  if (TypedefDecl *localType = dyn_cast<TypedefDecl>(D)) {
    OS << "\"typedef " << localType->getUnderlyingType().getAsString()
       << ' ' << localType << '"';
  } else if (ValueDecl *VD = dyn_cast<ValueDecl>(D)) {
    OS << "\"";
    // Emit storage class for vardecls.
    if (VarDecl *V = dyn_cast<VarDecl>(VD)) {
      if (V->getStorageClass() != VarDecl::None)
        OS << VarDecl::getStorageClassSpecifierString(V->getStorageClass())
           << " ";
    }

    std::string Name = VD->getNameAsString();
    VD->getType().getAsStringInternal(Name,
                          PrintingPolicy(VD->getASTContext().getLangOptions()));
    OS << Name;

    // If this is a vardecl with an initializer, emit it.
    if (VarDecl *V = dyn_cast<VarDecl>(VD)) {
      if (V->getInit()) {
        OS << " =\n";
        DumpSubTree(V->getInit());
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
    UD->getTargetNestedNameDecl()->print(OS,
        PrintingPolicy(UD->getASTContext().getLangOptions()));
    OS << ";\"";
  } else {
    assert(0 && "Unexpected decl");
  }
}

void StmtDumper::VisitDeclStmt(DeclStmt *Node) {
  DumpStmt(Node);
  OS << "\n";
  for (DeclStmt::decl_iterator DI = Node->decl_begin(), DE = Node->decl_end();
       DI != DE; ++DI) {
    Decl* D = *DI;
    ++IndentLevel;
    Indent();
    OS << (void*) D << " ";
    DumpDeclarator(D);
    if (DI+1 != DE)
      OS << "\n";
    --IndentLevel;
  }
}

void StmtDumper::VisitLabelStmt(LabelStmt *Node) {
  DumpStmt(Node);
  OS << " '" << Node->getName() << "'";
}

void StmtDumper::VisitGotoStmt(GotoStmt *Node) {
  DumpStmt(Node);
  OS << " '" << Node->getLabel()->getName()
     << "':" << (void*)Node->getLabel();
}

//===----------------------------------------------------------------------===//
//  Expr printing methods.
//===----------------------------------------------------------------------===//

void StmtDumper::VisitExpr(Expr *Node) {
  DumpExpr(Node);
}

static void DumpBasePath(llvm::raw_ostream &OS, CastExpr *Node) {
  if (Node->getBasePath().empty())
    return;

  OS << " (";
  bool First = true;
  for (CXXBaseSpecifierArray::iterator I = Node->getBasePath().begin(),
       E = Node->getBasePath().end(); I != E; ++I) {
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

void StmtDumper::VisitCastExpr(CastExpr *Node) {
  DumpExpr(Node);
  OS << " <" << Node->getCastKindName();
  DumpBasePath(OS, Node);
  OS << ">";
}

void StmtDumper::VisitImplicitCastExpr(ImplicitCastExpr *Node) {
  VisitCastExpr(Node);
  if (Node->isLvalueCast())
    OS << " lvalue";
}

void StmtDumper::VisitDeclRefExpr(DeclRefExpr *Node) {
  DumpExpr(Node);

  OS << " ";
  switch (Node->getDecl()->getKind()) {
  default: OS << "Decl"; break;
  case Decl::Function: OS << "FunctionDecl"; break;
  case Decl::Var: OS << "Var"; break;
  case Decl::ParmVar: OS << "ParmVar"; break;
  case Decl::EnumConstant: OS << "EnumConstant"; break;
  case Decl::Typedef: OS << "Typedef"; break;
  case Decl::Record: OS << "Record"; break;
  case Decl::Enum: OS << "Enum"; break;
  case Decl::CXXRecord: OS << "CXXRecord"; break;
  case Decl::ObjCInterface: OS << "ObjCInterface"; break;
  case Decl::ObjCClass: OS << "ObjCClass"; break;
  }

  OS << "='" << Node->getDecl() << "' " << (void*)Node->getDecl();
}

void StmtDumper::VisitUnresolvedLookupExpr(UnresolvedLookupExpr *Node) {
  DumpExpr(Node);
  OS << " (";
  if (!Node->requiresADL()) OS << "no ";
  OS << "ADL) = '" << Node->getName() << '\'';

  UnresolvedLookupExpr::decls_iterator
    I = Node->decls_begin(), E = Node->decls_end();
  if (I == E) OS << " empty";
  for (; I != E; ++I)
    OS << " " << (void*) *I;
}

void StmtDumper::VisitObjCIvarRefExpr(ObjCIvarRefExpr *Node) {
  DumpExpr(Node);

  OS << " " << Node->getDecl()->getDeclKindName()
     << "Decl='" << Node->getDecl()
     << "' " << (void*)Node->getDecl();
  if (Node->isFreeIvar())
    OS << " isFreeIvar";
}

void StmtDumper::VisitPredefinedExpr(PredefinedExpr *Node) {
  DumpExpr(Node);
  switch (Node->getIdentType()) {
  default: assert(0 && "unknown case");
  case PredefinedExpr::Func:           OS <<  " __func__"; break;
  case PredefinedExpr::Function:       OS <<  " __FUNCTION__"; break;
  case PredefinedExpr::PrettyFunction: OS <<  " __PRETTY_FUNCTION__";break;
  }
}

void StmtDumper::VisitCharacterLiteral(CharacterLiteral *Node) {
  DumpExpr(Node);
  OS << Node->getValue();
}

void StmtDumper::VisitIntegerLiteral(IntegerLiteral *Node) {
  DumpExpr(Node);

  bool isSigned = Node->getType()->isSignedIntegerType();
  OS << " " << Node->getValue().toString(10, isSigned);
}
void StmtDumper::VisitFloatingLiteral(FloatingLiteral *Node) {
  DumpExpr(Node);
  OS << " " << Node->getValueAsApproximateDouble();
}

void StmtDumper::VisitStringLiteral(StringLiteral *Str) {
  DumpExpr(Str);
  // FIXME: this doesn't print wstrings right.
  OS << " ";
  if (Str->isWide())
    OS << "L";
  OS << '"';
  OS.write_escaped(llvm::StringRef(Str->getStrData(),
                                   Str->getByteLength()));
  OS << '"';
}

void StmtDumper::VisitUnaryOperator(UnaryOperator *Node) {
  DumpExpr(Node);
  OS << " " << (Node->isPostfix() ? "postfix" : "prefix")
     << " '" << UnaryOperator::getOpcodeStr(Node->getOpcode()) << "'";
}
void StmtDumper::VisitSizeOfAlignOfExpr(SizeOfAlignOfExpr *Node) {
  DumpExpr(Node);
  OS << " " << (Node->isSizeOf() ? "sizeof" : "alignof") << " ";
  if (Node->isArgumentType())
    DumpType(Node->getArgumentType());
}

void StmtDumper::VisitMemberExpr(MemberExpr *Node) {
  DumpExpr(Node);
  OS << " " << (Node->isArrow() ? "->" : ".")
     << Node->getMemberDecl() << ' '
     << (void*)Node->getMemberDecl();
}
void StmtDumper::VisitExtVectorElementExpr(ExtVectorElementExpr *Node) {
  DumpExpr(Node);
  OS << " " << Node->getAccessor().getNameStart();
}
void StmtDumper::VisitBinaryOperator(BinaryOperator *Node) {
  DumpExpr(Node);
  OS << " '" << BinaryOperator::getOpcodeStr(Node->getOpcode()) << "'";
}
void StmtDumper::VisitCompoundAssignOperator(CompoundAssignOperator *Node) {
  DumpExpr(Node);
  OS << " '" << BinaryOperator::getOpcodeStr(Node->getOpcode())
     << "' ComputeLHSTy=";
  DumpType(Node->getComputationLHSType());
  OS << " ComputeResultTy=";
  DumpType(Node->getComputationResultType());
}

// GNU extensions.

void StmtDumper::VisitAddrLabelExpr(AddrLabelExpr *Node) {
  DumpExpr(Node);
  OS << " " << Node->getLabel()->getName()
     << " " << (void*)Node->getLabel();
}

void StmtDumper::VisitTypesCompatibleExpr(TypesCompatibleExpr *Node) {
  DumpExpr(Node);
  OS << " ";
  DumpType(Node->getArgType1());
  OS << " ";
  DumpType(Node->getArgType2());
}

//===----------------------------------------------------------------------===//
// C++ Expressions
//===----------------------------------------------------------------------===//

void StmtDumper::VisitCXXNamedCastExpr(CXXNamedCastExpr *Node) {
  DumpExpr(Node);
  OS << " " << Node->getCastName() 
     << "<" << Node->getTypeAsWritten().getAsString() << ">"
     << " <" << Node->getCastKindName();
  DumpBasePath(OS, Node);
  OS << ">";
}

void StmtDumper::VisitCXXBoolLiteralExpr(CXXBoolLiteralExpr *Node) {
  DumpExpr(Node);
  OS << " " << (Node->getValue() ? "true" : "false");
}

void StmtDumper::VisitCXXThisExpr(CXXThisExpr *Node) {
  DumpExpr(Node);
  OS << " this";
}

void StmtDumper::VisitCXXFunctionalCastExpr(CXXFunctionalCastExpr *Node) {
  DumpExpr(Node);
  OS << " functional cast to " << Node->getTypeAsWritten().getAsString();
}

void StmtDumper::VisitCXXConstructExpr(CXXConstructExpr *Node) {
  DumpExpr(Node);
  CXXConstructorDecl *Ctor = Node->getConstructor();
  DumpType(Ctor->getType());
  if (Node->isElidable())
    OS << " elidable";
}

void StmtDumper::VisitCXXBindTemporaryExpr(CXXBindTemporaryExpr *Node) {
  DumpExpr(Node);
  OS << " ";
  DumpCXXTemporary(Node->getTemporary());
}

void StmtDumper::VisitCXXExprWithTemporaries(CXXExprWithTemporaries *Node) {
  DumpExpr(Node);
  ++IndentLevel;
  for (unsigned i = 0, e = Node->getNumTemporaries(); i != e; ++i) {
    OS << "\n";
    Indent();
    DumpCXXTemporary(Node->getTemporary(i));
  }
  --IndentLevel;
}

void StmtDumper::DumpCXXTemporary(CXXTemporary *Temporary) {
  OS << "(CXXTemporary " << (void *)Temporary << ")";
}

//===----------------------------------------------------------------------===//
// Obj-C Expressions
//===----------------------------------------------------------------------===//

void StmtDumper::VisitObjCMessageExpr(ObjCMessageExpr* Node) {
  DumpExpr(Node);
  OS << " selector=" << Node->getSelector().getAsString();
  switch (Node->getReceiverKind()) {
  case ObjCMessageExpr::Instance:
    break;

  case ObjCMessageExpr::Class:
    OS << " class=";
    DumpType(Node->getClassReceiver());
    break;

  case ObjCMessageExpr::SuperInstance:
    OS << " super (instance)";
    break;

  case ObjCMessageExpr::SuperClass:
    OS << " super (class)";
    break;
  }
}

void StmtDumper::VisitObjCAtCatchStmt(ObjCAtCatchStmt *Node) {
  DumpStmt(Node);
  if (VarDecl *CatchParam = Node->getCatchParamDecl()) {
    OS << " catch parm = ";
    DumpDeclarator(CatchParam);
  } else {
    OS << " catch all";
  }
}

void StmtDumper::VisitObjCEncodeExpr(ObjCEncodeExpr *Node) {
  DumpExpr(Node);
  OS << " ";
  DumpType(Node->getEncodedType());
}

void StmtDumper::VisitObjCSelectorExpr(ObjCSelectorExpr *Node) {
  DumpExpr(Node);

  OS << " " << Node->getSelector().getAsString();
}

void StmtDumper::VisitObjCProtocolExpr(ObjCProtocolExpr *Node) {
  DumpExpr(Node);

  OS << ' ' << Node->getProtocol();
}

void StmtDumper::VisitObjCPropertyRefExpr(ObjCPropertyRefExpr *Node) {
  DumpExpr(Node);

  OS << " Kind=PropertyRef Property=\"" << Node->getProperty() << '"';
}

void StmtDumper::VisitObjCImplicitSetterGetterRefExpr(
                                        ObjCImplicitSetterGetterRefExpr *Node) {
  DumpExpr(Node);

  ObjCMethodDecl *Getter = Node->getGetterMethod();
  ObjCMethodDecl *Setter = Node->getSetterMethod();
  OS << " Kind=MethodRef Getter=\""
     << Getter->getSelector().getAsString()
     << "\" Setter=\"";
  if (Setter)
    OS << Setter->getSelector().getAsString();
  else
    OS << "(null)";
  OS << "\"";
}

void StmtDumper::VisitObjCSuperExpr(ObjCSuperExpr *Node) {
  DumpExpr(Node);
  OS << " super";
}

//===----------------------------------------------------------------------===//
// Stmt method implementations
//===----------------------------------------------------------------------===//

/// dump - This does a local dump of the specified AST fragment.  It dumps the
/// specified node and a few nodes underneath it, but not the whole subtree.
/// This is useful in a debugger.
void Stmt::dump(SourceManager &SM) const {
  StmtDumper P(&SM, llvm::errs(), 4);
  P.DumpSubTree(const_cast<Stmt*>(this));
  llvm::errs() << "\n";
}

/// dump - This does a local dump of the specified AST fragment.  It dumps the
/// specified node and a few nodes underneath it, but not the whole subtree.
/// This is useful in a debugger.
void Stmt::dump() const {
  StmtDumper P(0, llvm::errs(), 4);
  P.DumpSubTree(const_cast<Stmt*>(this));
  llvm::errs() << "\n";
}

/// dumpAll - This does a dump of the specified AST fragment and all subtrees.
void Stmt::dumpAll(SourceManager &SM) const {
  StmtDumper P(&SM, llvm::errs(), ~0U);
  P.DumpSubTree(const_cast<Stmt*>(this));
  llvm::errs() << "\n";
}

/// dumpAll - This does a dump of the specified AST fragment and all subtrees.
void Stmt::dumpAll() const {
  StmtDumper P(0, llvm::errs(), ~0U);
  P.DumpSubTree(const_cast<Stmt*>(this));
  llvm::errs() << "\n";
}
