//===--- StmtDumper.cpp - Dumping implementation for Stmt ASTs ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Stmt::dump/Stmt::print methods, which dump out the
// AST in a form that exposes type details and other fields.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/StmtVisitor.h"
#include "clang/AST/Decl.h"
#include "clang/AST/ExprCXX.h"
#include "clang/Lex/IdentifierTable.h"
#include "llvm/Support/Compiler.h"
#include <cstdio>
using namespace clang;

//===----------------------------------------------------------------------===//
// StmtDumper Visitor
//===----------------------------------------------------------------------===//

namespace  {
  class VISIBILITY_HIDDEN StmtDumper : public StmtVisitor {
    FILE *F;
    unsigned IndentLevel;
    
    /// MaxDepth - When doing a normal dump (not dumpAll) we only want to dump
    /// the first few levels of an AST.  This keeps track of how many ast levels
    /// are left.
    unsigned MaxDepth;
  public:
    StmtDumper(FILE *f, unsigned maxDepth)
      : F(f), IndentLevel(0), MaxDepth(maxDepth) {}
    
    void DumpSubTree(Stmt *S, int SubIndent = 1) {
      // Prune the recursion if not using dump all.
      if (MaxDepth == 0) return;
      
      IndentLevel += SubIndent;
      if (S) {
        S->visit(*this);
      } else {
        Indent();
        fprintf(F, "<<<NULL>>>\n");
      }
      IndentLevel -= SubIndent;
    }
    
    void PrintRawDecl(Decl *D);
    
    void Indent() const {
      for (int i = 0, e = IndentLevel; i < e; ++i)
        fprintf(F, "  ");
    }
    
    void DumpStmt(const Stmt *Node) const {
      Indent();
      fprintf(F, "(%s %p", Node->getStmtClassName(), (void*)Node);
    }
    
    void DumpExpr(Expr *Node) const {
      DumpStmt(Node);
      // TODO: DUMP TYPE
    }
    
    virtual void VisitStmt(Stmt *Node);
#define STMT(N, CLASS, PARENT) \
    virtual void Visit##CLASS(CLASS *Node);
#include "clang/AST/StmtNodes.def"
  };
}

//===----------------------------------------------------------------------===//
//  Stmt printing methods.
//===----------------------------------------------------------------------===//

void StmtDumper::VisitStmt(Stmt *Node) {
  Indent();
  fprintf(F, "<<unknown stmt type>>\n");
}

void StmtDumper::PrintRawDecl(Decl *D) {
#if 0
  // FIXME: Need to complete/beautify this... this code simply shows the
  // nodes are where they need to be.
  if (TypedefDecl *localType = dyn_cast<TypedefDecl>(D)) {
    OS << "typedef " << localType->getUnderlyingType().getAsString();
    OS << " " << localType->getName();
  } else if (ValueDecl *VD = dyn_cast<ValueDecl>(D)) {
    // Emit storage class for vardecls.
    if (VarDecl *V = dyn_cast<VarDecl>(VD)) {
      switch (V->getStorageClass()) {
        default: assert(0 && "Unknown storage class!");
        case VarDecl::None:     break;
        case VarDecl::Extern:   OS << "extern "; break;
        case VarDecl::Static:   OS << "static "; break; 
        case VarDecl::Auto:     OS << "auto "; break;
        case VarDecl::Register: OS << "register "; break;
      }
    }
    
    std::string Name = VD->getName();
    VD->getType().getAsStringInternal(Name);
    OS << Name;
    
    // If this is a vardecl with an initializer, emit it.
    if (VarDecl *V = dyn_cast<VarDecl>(VD)) {
      if (V->getInit()) {
        OS << " = ";
        DumpExpr(V->getInit());
      }
    }
  } else {
    // FIXME: "struct x;"
    assert(0 && "Unexpected decl");
  }
#endif
}


void StmtDumper::VisitNullStmt(NullStmt *Node) {
  DumpStmt(Node);
  fprintf(F, ")");
}

void StmtDumper::VisitDeclStmt(DeclStmt *Node) {
  DumpStmt(Node);
  // FIXME: implement this better :)
  fprintf(F, ")");
#if 0
  for (Decl *D = Node->getDecl(); D; D = D->getNextDeclarator()) {
    Indent();
    PrintRawDecl(D);
    OS << ";\n";
  }
#endif
}

void StmtDumper::VisitCompoundStmt(CompoundStmt *Node) {
  DumpStmt(Node);
  if (!Node->body_empty()) fprintf(F, "\n");
  
  for (CompoundStmt::body_iterator I = Node->body_begin(), E = Node->body_end();
       I != E; ) {
    DumpSubTree(*I);
    ++I;
    if (I != E)
      fprintf(F, "\n");
  }
  fprintf(F, ")");
}

void StmtDumper::VisitCaseStmt(CaseStmt *Node) {
#if 0
  Indent(-1) << "case ";
  DumpExpr(Node->getLHS());
  if (Node->getRHS()) {
    OS << " ... ";
    DumpExpr(Node->getRHS());
  }
  OS << ":\n";
  
  DumpSubTree(Node->getSubStmt(), 0);
#endif
}

void StmtDumper::VisitDefaultStmt(DefaultStmt *Node) {
  DumpStmt(Node);
  fprintf(F, "\n");
  DumpSubTree(Node->getSubStmt());
  fprintf(F, ")");
}

void StmtDumper::VisitLabelStmt(LabelStmt *Node) {
  DumpStmt(Node);
  fprintf(F, " '%s'\n", Node->getName());
  DumpSubTree(Node->getSubStmt());
  fprintf(F, "\n");
}

void StmtDumper::VisitIfStmt(IfStmt *Node) {
  DumpStmt(Node);
  fprintf(F, "\n");
  DumpSubTree(Node->getCond());
  fprintf(F, "\n");
  DumpSubTree(Node->getThen());
  fprintf(F, "\n");
  DumpSubTree(Node->getElse());
  fprintf(F, ")");
}

void StmtDumper::VisitSwitchStmt(SwitchStmt *Node) {
  DumpStmt(Node);
  fprintf(F, "\n");
  DumpSubTree(Node->getCond());
  fprintf(F, "\n");
  DumpSubTree(Node->getBody());
  fprintf(F, ")");
}

void StmtDumper::VisitSwitchCase(SwitchCase*) {
  assert(0 && "SwitchCase is an abstract class");
}

void StmtDumper::VisitWhileStmt(WhileStmt *Node) {
  DumpStmt(Node);
  fprintf(F, "\n");
  DumpSubTree(Node->getCond());
  fprintf(F, "\n");
  DumpSubTree(Node->getBody());
  fprintf(F, ")");
}

void StmtDumper::VisitDoStmt(DoStmt *Node) {
  DumpStmt(Node);
  fprintf(F, "\n");
  DumpSubTree(Node->getBody());
  fprintf(F, "\n");
  DumpSubTree(Node->getCond());
  fprintf(F, ")");
}

void StmtDumper::VisitForStmt(ForStmt *Node) {
  DumpStmt(Node);
  fprintf(F, "\n");
  DumpSubTree(Node->getInit());
  fprintf(F, "\n");
  DumpSubTree(Node->getCond());
  fprintf(F, "\n");
  DumpSubTree(Node->getInc());
  fprintf(F, "\n");
  DumpSubTree(Node->getBody());
  fprintf(F, ")");
}

void StmtDumper::VisitGotoStmt(GotoStmt *Node) {
  DumpStmt(Node);
  fprintf(F, " '%s':%p)", Node->getLabel()->getName(), (void*)Node->getLabel());
}

void StmtDumper::VisitIndirectGotoStmt(IndirectGotoStmt *Node) {
  DumpStmt(Node);
  fprintf(F, "\n");
  DumpSubTree(Node->getTarget());
  fprintf(F, ")");
}

void StmtDumper::VisitContinueStmt(ContinueStmt *Node) {
  DumpStmt(Node);
  fprintf(F, ")");
}

void StmtDumper::VisitBreakStmt(BreakStmt *Node) {
  DumpStmt(Node);
  fprintf(F, ")");
}


void StmtDumper::VisitReturnStmt(ReturnStmt *Node) {
  DumpStmt(Node);
  if (Expr *RV = Node->getRetValue()) {
    fprintf(F, "\n");
    DumpSubTree(RV);
  }
  fprintf(F, ")");
}

//===----------------------------------------------------------------------===//
//  Expr printing methods.
//===----------------------------------------------------------------------===//

void StmtDumper::VisitExpr(Expr *Node) {
  DumpExpr(Node);
  fprintf(F, ": UNKNOWN EXPR to StmtDumper)");
}

void StmtDumper::VisitDeclRefExpr(DeclRefExpr *Node) {
  DumpExpr(Node);
  fprintf(F, " Decl='%s' %p)", Node->getDecl()->getName(),
          (void*)Node->getDecl());
}

void StmtDumper::VisitPreDefinedExpr(PreDefinedExpr *Node) {
  DumpExpr(Node);
  switch (Node->getIdentType()) {
  default:
    assert(0 && "unknown case");
  case PreDefinedExpr::Func:
    fprintf(F, " __func__)");
    break;
  case PreDefinedExpr::Function:
    fprintf(F, " __FUNCTION__)");
    break;
  case PreDefinedExpr::PrettyFunction:
    fprintf(F, " __PRETTY_FUNCTION__)");
    break;
  }
}

void StmtDumper::VisitCharacterLiteral(CharacterLiteral *Node) {
#if 0
  // FIXME should print an L for wchar_t constants
  unsigned value = Node->getValue();
  switch (value) {
  case '\\':
    OS << "'\\\\'";
    break;
  case '\'':
    OS << "'\\''";
    break;
  case '\a':
    // TODO: K&R: the meaning of '\\a' is different in traditional C
    OS << "'\\a'";
    break;
  case '\b':
    OS << "'\\b'";
    break;
  // Nonstandard escape sequence.
  /*case '\e':
    OS << "'\\e'";
    break;*/
  case '\f':
    OS << "'\\f'";
    break;
  case '\n':
    OS << "'\\n'";
    break;
  case '\r':
    OS << "'\\r'";
    break;
  case '\t':
    OS << "'\\t'";
    break;
  case '\v':
    OS << "'\\v'";
    break;
  default:
    if (isprint(value) && value < 256) {
      OS << "'" << (char)value << "'";
    } else if (value < 256) {
      OS << "'\\x" << std::hex << value << std::dec << "'";
    } else {
      // FIXME what to really do here?
      OS << value;
    }
  }
#endif
}

void StmtDumper::VisitIntegerLiteral(IntegerLiteral *Node) {
  DumpExpr(Node);

  bool isSigned = Node->getType()->isSignedIntegerType();
  fprintf(F, " %s)", Node->getValue().toString(10, isSigned).c_str());
}
void StmtDumper::VisitFloatingLiteral(FloatingLiteral *Node) {
  DumpExpr(Node);
  fprintf(F, " %f)", Node->getValue());
}
void StmtDumper::VisitStringLiteral(StringLiteral *Str) {
#if 0
  if (Str->isWide()) OS << 'L';
  OS << '"';
  
  // FIXME: this doesn't print wstrings right.
  for (unsigned i = 0, e = Str->getByteLength(); i != e; ++i) {
    switch (Str->getStrData()[i]) {
    default: OS << Str->getStrData()[i]; break;
    // Handle some common ones to make dumps prettier.
    case '\\': OS << "\\\\"; break;
    case '"': OS << "\\\""; break;
    case '\n': OS << "\\n"; break;
    case '\t': OS << "\\t"; break;
    case '\a': OS << "\\a"; break;
    case '\b': OS << "\\b"; break;
    }
  }
  OS << '"';
#endif
}
void StmtDumper::VisitParenExpr(ParenExpr *Node) {
  DumpExpr(Node);
  fprintf(F, "\n");
  DumpSubTree(Node->getSubExpr());
  fprintf(F, ")");
}
void StmtDumper::VisitUnaryOperator(UnaryOperator *Node) {
#if 0
  if (!Node->isPostfix())
    OS << UnaryOperator::getOpcodeStr(Node->getOpcode());
  DumpExpr(Node->getSubExpr());
  
  if (Node->isPostfix())
    OS << UnaryOperator::getOpcodeStr(Node->getOpcode());

#endif
}
void StmtDumper::VisitSizeOfAlignOfTypeExpr(SizeOfAlignOfTypeExpr *Node) {
#if 0
  OS << (Node->isSizeOf() ? "sizeof(" : "__alignof(");
  OS << Node->getArgumentType().getAsString() << ")";
#endif
}
void StmtDumper::VisitArraySubscriptExpr(ArraySubscriptExpr *Node) {
  DumpExpr(Node);
  fprintf(F, "\n");
  DumpSubTree(Node->getBase());
  fprintf(F, "\n");
  DumpSubTree(Node->getIdx());
  fprintf(F, ")");
}

void StmtDumper::VisitCallExpr(CallExpr *Node) {
  DumpExpr(Node);
  fprintf(F, "\n");
  DumpSubTree(Node->getCallee());
  
  for (unsigned i = 0, e = Node->getNumArgs(); i != e; ++i) {
    fprintf(F, "\n");
    DumpSubTree(Node->getArg(i));
  }
  fprintf(F, ")");
}
void StmtDumper::VisitMemberExpr(MemberExpr *Node) {
#if 0
  DumpExpr(Node->getBase());
  OS << (Node->isArrow() ? "->" : ".");
  
  FieldDecl *Field = Node->getMemberDecl();
  assert(Field && "MemberExpr should alway reference a field!");
  OS << Field->getName();
#endif
}
void StmtDumper::VisitOCUVectorElementExpr(OCUVectorElementExpr *Node) {
#if 0
  DumpExpr(Node->getBase());
  OS << ".";
  OS << Node->getAccessor().getName();
#endif
}
void StmtDumper::VisitCastExpr(CastExpr *Node) {
#if 0
  OS << "(" << Node->getType().getAsString() << ")";
  DumpExpr(Node->getSubExpr());
#endif
}
void StmtDumper::VisitCompoundLiteralExpr(CompoundLiteralExpr *Node) {
#if 0
  OS << "(" << Node->getType().getAsString() << ")";
  DumpExpr(Node->getInitializer());
#endif
}
void StmtDumper::VisitImplicitCastExpr(ImplicitCastExpr *Node) {
  DumpExpr(Node);
  fprintf(F, "\n");
  DumpSubTree(Node->getSubExpr());
  fprintf(F, ")");
}
void StmtDumper::VisitBinaryOperator(BinaryOperator *Node) {
  DumpExpr(Node);
  fprintf(F, " '%s'\n", BinaryOperator::getOpcodeStr(Node->getOpcode()));
  DumpSubTree(Node->getLHS());
  fprintf(F, "\n");
  DumpSubTree(Node->getRHS());
  fprintf(F, ")");
}
void StmtDumper::VisitConditionalOperator(ConditionalOperator *Node) {
  DumpExpr(Node);
  fprintf(F, "\n");
  DumpSubTree(Node->getCond());
  fprintf(F, "\n");
  DumpSubTree(Node->getLHS());
  fprintf(F, "\n");
  DumpSubTree(Node->getRHS());
  fprintf(F, ")");
}

// GNU extensions.

void StmtDumper::VisitAddrLabelExpr(AddrLabelExpr *Node) {
#if 0
  OS << "&&" << Node->getLabel()->getName();
#endif
}

void StmtDumper::VisitStmtExpr(StmtExpr *E) {
#if 0
  OS << "(";
  DumpSubTree(E->getSubStmt());
  OS << ")";
#endif
}

void StmtDumper::VisitTypesCompatibleExpr(TypesCompatibleExpr *Node) {
#if 0
  OS << "__builtin_types_compatible_p(";
  OS << Node->getArgType1().getAsString() << ",";
  OS << Node->getArgType2().getAsString() << ")";
#endif
}

void StmtDumper::VisitChooseExpr(ChooseExpr *Node) {
#if 0
  OS << "__builtin_choose_expr(";
  DumpExpr(Node->getCond());
  OS << ", ";
  DumpExpr(Node->getLHS());
  OS << ", ";
  DumpExpr(Node->getRHS());
  OS << ")";
#endif
}

// C++

void StmtDumper::VisitCXXCastExpr(CXXCastExpr *Node) {
#if 0
  switch (Node->getOpcode()) {
    default:
      assert(0 && "Not a C++ cast expression");
      abort();
    case CXXCastExpr::ConstCast:       OS << "const_cast<";       break;
    case CXXCastExpr::DynamicCast:     OS << "dynamic_cast<";     break;
    case CXXCastExpr::ReinterpretCast: OS << "reinterpret_cast<"; break;
    case CXXCastExpr::StaticCast:      OS << "static_cast<";      break;
  }
  
  OS << Node->getDestType().getAsString() << ">(";
  DumpExpr(Node->getSubExpr());
  OS << ")";
#endif
}

void StmtDumper::VisitCXXBoolLiteralExpr(CXXBoolLiteralExpr *Node) {
#if 0
  OS << (Node->getValue() ? "true" : "false");
#endif
}


//===----------------------------------------------------------------------===//
// Stmt method implementations
//===----------------------------------------------------------------------===//

/// dump - This does a local dump of the specified AST fragment.  It dumps the
/// specified node and a few nodes underneath it, but not the whole subtree.
/// This is useful in a debugger.
void Stmt::dump() const {
  StmtDumper P(stderr, 4);
  const_cast<Stmt*>(this)->visit(P);
}

/// dumpAll - This does a dump of the specified AST fragment and all subtrees.
void Stmt::dumpAll() const {
  StmtDumper P(stderr, ~0U);
  const_cast<Stmt*>(this)->visit(P);
}
