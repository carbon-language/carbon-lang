//===--- StmtPrinter.cpp - Printing implementation for Stmt ASTs ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Stmt::dump/Stmt::print methods.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/StmtVisitor.h"
#include "clang/AST/Expr.h"
#include "llvm/Support/Compiler.h"
#include <iostream>
using namespace llvm;
using namespace clang;

namespace {
  struct VISIBILITY_HIDDEN IsExprStmtVisitor : public StmtVisitor {
    bool &Result;
    IsExprStmtVisitor(bool &R) : Result(R) { Result = false; }
    
    virtual void VisitExpr(Expr *Node) {
      Result = true;
    }
  };
}

static bool isExpr(Stmt *S) {
  bool Val = false;
  IsExprStmtVisitor V(Val);
  S->visit(V);
  return Val;
}

//===----------------------------------------------------------------------===//
// StmtPrinter Visitor
//===----------------------------------------------------------------------===//

namespace  {
  class VISIBILITY_HIDDEN StmtPrinter : public StmtVisitor {
    std::ostream &OS;
    unsigned IndentLevel;
  public:
    StmtPrinter(std::ostream &os) : OS(os), IndentLevel(0) {}
    
    void PrintStmt(Stmt *S) {
      ++IndentLevel;
      if (S && isExpr(S)) {
        // If this is an expr used in a stmt context, indent and newline it.
        Indent();
        S->visit(*this);
        OS << "\n";
      } else if (S) {
        S->visit(*this);
      } else {
        Indent() << "<null stmt>\n";
      }
      --IndentLevel;
    }

    void PrintExpr(Expr *E) {
      if (E)
        E->visit(*this);
      else
        OS << "<null expr>";
    }
    
    std::ostream &Indent() const {
      for (unsigned i = 0, e = IndentLevel; i != e; ++i)
        OS << "  ";
      return OS;
    }
    
    virtual void VisitStmt(Stmt *Node);
    virtual void VisitCompoundStmt(CompoundStmt *Node);
    virtual void VisitIfStmt(IfStmt *Node);
    virtual void VisitReturnStmt(ReturnStmt *Node);

    virtual void VisitExpr(Expr *Node);
    virtual void VisitDeclRefExpr(DeclRefExpr *Node);
    virtual void VisitIntegerConstant(IntegerConstant *Node);
    virtual void VisitFloatingConstant(FloatingConstant *Node);
    virtual void VisitStringExpr(StringExpr *Node);
    virtual void VisitParenExpr(ParenExpr *Node);
    virtual void VisitUnaryOperator(UnaryOperator *Node);
    virtual void VisitSizeOfAlignOfTypeExpr(SizeOfAlignOfTypeExpr *Node);
    virtual void VisitArraySubscriptExpr(ArraySubscriptExpr *Node);
    virtual void VisitCallExpr(CallExpr *Node);
    virtual void VisitMemberExpr(MemberExpr *Node);
    virtual void VisitCastExpr(CastExpr *Node);
    virtual void VisitBinaryOperator(BinaryOperator *Node);
    virtual void VisitConditionalOperator(ConditionalOperator *Node);
};
}

void StmtPrinter::VisitStmt(Stmt *Node) {
  Indent() << "<<unknown stmt type>>\n";
}

void StmtPrinter::VisitCompoundStmt(CompoundStmt *Node) {
  Indent() << "{\n";
  
  for (CompoundStmt::body_iterator I = Node->body_begin(), E = Node->body_end();
       I != E; ++I)
    PrintStmt(*I);
  
  Indent() << "}\n";
}

void StmtPrinter::VisitIfStmt(IfStmt *If) {
  Indent() << "if ";
  PrintExpr(If->getCond());

  OS << " then\n";
  PrintStmt(If->getThen());
  if (If->getElse()) {
    Indent() << "else\n";
    PrintStmt(If->getElse());
  }
  Indent() << "endif\n";
}

void StmtPrinter::VisitReturnStmt(ReturnStmt *Node) {
  Indent() << "return";
  if (Node->getRetValue()) {
    OS << " ";
    PrintExpr(Node->getRetValue());
  }
  OS << "\n";
}


void StmtPrinter::VisitExpr(Expr *Node) {
  OS << "<<unknown expr type>>";
}

void StmtPrinter::VisitDeclRefExpr(DeclRefExpr *Node) {
  // FIXME: print name.
  OS << "x";
}

void StmtPrinter::VisitIntegerConstant(IntegerConstant *Node) {
  // FIXME: print value.
  OS << "1";
}
void StmtPrinter::VisitFloatingConstant(FloatingConstant *Node) {
  // FIXME: print value.
  OS << "1.0";
}
void StmtPrinter::VisitStringExpr(StringExpr *Str) {
  if (Str->isWide()) OS << 'L';
  OS << '"' << Str->getStrData() << '"';
}
void StmtPrinter::VisitParenExpr(ParenExpr *Node) {
  OS << "(";
  PrintExpr(Node->getSubExpr());
  OS << ")'";
}
void StmtPrinter::VisitUnaryOperator(UnaryOperator *Node) {
  OS << UnaryOperator::getOpcodeStr(Node->getOpcode());
  PrintExpr(Node->getSubExpr());
}
void StmtPrinter::VisitSizeOfAlignOfTypeExpr(SizeOfAlignOfTypeExpr *Node) {
  OS << (Node->isSizeOf() ? "sizeof(" : "alignof(");
  // FIXME: print type.
  OS << "ty)";
}
void StmtPrinter::VisitArraySubscriptExpr(ArraySubscriptExpr *Node) {
  PrintExpr(Node->getBase());
  OS << "[";
  PrintExpr(Node->getIdx());
  OS << "]";
}

void StmtPrinter::VisitCallExpr(CallExpr *Call) {
  PrintExpr(Call->getCallee());
  OS << "(";
  for (unsigned i = 0, e = Call->getNumArgs(); i != e; ++i) {
    if (i) OS << ", ";
    PrintExpr(Call->getArg(i));
  }
  OS << ")";
}
void StmtPrinter::VisitMemberExpr(MemberExpr *Node) {
  PrintExpr(Node->getBase());
  OS << (Node->isArrow() ? "->" : ".");
  
  if (Node->getMemberDecl())
    assert(0 && "TODO: should print member decl!");
  OS << "member";
}
void StmtPrinter::VisitCastExpr(CastExpr *Node) {
  OS << "(";
  // TODO PRINT TYPE
  OS << "<type>";
  OS << ")";
  PrintExpr(Node->getSubExpr());
}
void StmtPrinter::VisitBinaryOperator(BinaryOperator *Node) {
  PrintExpr(Node->getLHS());
  OS << " " << BinaryOperator::getOpcodeStr(Node->getOpcode()) << " ";
  PrintExpr(Node->getRHS());
}
void StmtPrinter::VisitConditionalOperator(ConditionalOperator *Node) {
  PrintExpr(Node->getCond());
  OS << " ? ";
  PrintExpr(Node->getLHS());
  std::cerr << " : ";
  PrintExpr(Node->getRHS());
}

//===----------------------------------------------------------------------===//
// Stmt method implementations
//===----------------------------------------------------------------------===//

void Stmt::dump() const {
  print(std::cerr);
}

void Stmt::print(std::ostream &OS) const {
  if (this == 0) {
    OS << "<NULL>";
    return;
  }

  StmtPrinter P(OS);
  const_cast<Stmt*>(this)->visit(P);
}
