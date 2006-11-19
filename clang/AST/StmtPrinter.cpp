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
#include "clang/Lex/IdentifierTable.h"
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
        Indent() << ";\n";
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
#define STMT(CLASS, PARENT) \
    virtual void Visit##CLASS(CLASS *Node);
#include "clang/AST/StmtNodes.def"
  };
}

//===----------------------------------------------------------------------===//
//  Stmt printing methods.
//===----------------------------------------------------------------------===//

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

void StmtPrinter::VisitCaseStmt(CaseStmt *Node) {
  Indent() << "case ";
  PrintExpr(Node->getLHS());
  if (Node->getRHS()) {
    OS << " ... ";
    PrintExpr(Node->getRHS());
  }
  OS << ":\n";
  
  // FIXME: This recursively indents consequtive cases.
  PrintStmt(Node->getSubStmt());
}

void StmtPrinter::VisitDefaultStmt(DefaultStmt *Node) {
  Indent() << "default:\n";
  // FIXME: This recursively indents consequtive cases.
  PrintStmt(Node->getSubStmt());
}

void StmtPrinter::VisitLabelStmt(LabelStmt *Node) {
  Indent() << Node->getLabel()->getName() << ":\n";
  // FIXME: This recursively indents consequtive cases.
  PrintStmt(Node->getSubStmt());
}

void StmtPrinter::VisitIfStmt(IfStmt *If) {
  Indent() << "if (";
  PrintExpr(If->getCond());

  OS << ")\n";
  PrintStmt(If->getThen());
  if (If->getElse()) {
    Indent() << "else\n";
    PrintStmt(If->getElse());
  }
}

void StmtPrinter::VisitSwitchStmt(SwitchStmt *Node) {
  Indent() << "switch (";
  PrintExpr(Node->getCond());
  OS << ")\n";
  PrintStmt(Node->getBody());
}

void StmtPrinter::VisitWhileStmt(WhileStmt *Node) {
  Indent() << "while (";
  PrintExpr(Node->getCond());
  OS << ")\n";
  PrintStmt(Node->getBody());
}

void StmtPrinter::VisitDoStmt(DoStmt *Node) {
  Indent() << "do\n";
  PrintStmt(Node->getBody());
  Indent() << "while (";
  PrintExpr(Node->getCond());
  OS << ")\n";
}

void StmtPrinter::VisitForStmt(ForStmt *Node) {
  Indent() << "for (";
  if (Node->getFirst())
    PrintExpr((Expr*)Node->getFirst());
  OS << "; ";
  if (Node->getSecond())
    PrintExpr(Node->getSecond());
  OS << "; ";
  if (Node->getThird())
    PrintExpr(Node->getThird());
  OS << ")\n";
  PrintStmt(Node->getBody());
}

void StmtPrinter::VisitGotoStmt(GotoStmt *Node) {
  Indent() << "goto " << Node->getLabel()->getName() << "\n";
}

void StmtPrinter::VisitIndirectGotoStmt(IndirectGotoStmt *Node) {
  Indent() << "goto *";
  PrintExpr(Node->getTarget());
  OS << "\n";
}

void StmtPrinter::VisitContinueStmt(ContinueStmt *Node) {
  Indent() << "continue\n";
}

void StmtPrinter::VisitBreakStmt(BreakStmt *Node) {
  Indent() << "break\n";
}


void StmtPrinter::VisitReturnStmt(ReturnStmt *Node) {
  Indent() << "return";
  if (Node->getRetValue()) {
    OS << " ";
    PrintExpr(Node->getRetValue());
  }
  OS << "\n";
}

//===----------------------------------------------------------------------===//
//  Expr printing methods.
//===----------------------------------------------------------------------===//

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
}
void StmtPrinter::VisitParenExpr(ParenExpr *Node) {
  OS << "(";
  PrintExpr(Node->getSubExpr());
  OS << ")'";
}
void StmtPrinter::VisitUnaryOperator(UnaryOperator *Node) {
  if (!Node->isPostfix())
    OS << UnaryOperator::getOpcodeStr(Node->getOpcode());
  PrintExpr(Node->getSubExpr());
  
  if (Node->isPostfix())
    OS << UnaryOperator::getOpcodeStr(Node->getOpcode());

}
void StmtPrinter::VisitSizeOfAlignOfTypeExpr(SizeOfAlignOfTypeExpr *Node) {
  OS << (Node->isSizeOf() ? "sizeof(" : "__alignof(");
  
  std::string TypeStr;
  Node->getArgumentType()->getAsString(TypeStr);
  OS << TypeStr << ")";
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
