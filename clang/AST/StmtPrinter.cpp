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

//===----------------------------------------------------------------------===//
// StmtPrinter Visitor
//===----------------------------------------------------------------------===//

namespace  {
  class VISIBILITY_HIDDEN StmtPrinter : public StmtVisitor {
    std::ostream &OS;
    unsigned IndentLevel;
  public:
    StmtPrinter(std::ostream &os) : OS(os), IndentLevel(0) {}
    
    void visit(Stmt *S) {
      if (S)
        S->visit(*this);
      else
        VisitNull();
    }
    
    std::ostream &Indent() const {
      for (unsigned i = 0, e = IndentLevel; i != e; ++i)
        OS << "  ";
      return OS;
    }
    
    void VisitNull();
    void VisitCompoundStmt(CompoundStmt *Node);
    void VisitIfStmt(IfStmt *Node);
  };
}

void StmtPrinter::VisitNull() {
  Indent() << "<nullptr>\n";
}

// FIXME: split out ExprPrinter from StmtPrinter.


void StmtPrinter::VisitCompoundStmt(CompoundStmt *Node) {
  Indent() << "{\n";
  ++IndentLevel;
  
  for (CompoundStmt::body_iterator I = Node->body_begin(), E = Node->body_end();
       I != E; ++I) {
    visit(*I);
  }
  
  --IndentLevel;
  Indent() << "}\n";
}

void StmtPrinter::VisitIfStmt(IfStmt *If) {
  Indent() << "if ";
  visit(If->getCond());

  OS << " then\n";
  ++IndentLevel;
  visit(If->getThen());
  --IndentLevel;
  Indent() << "else\n";
  ++IndentLevel;
  visit(If->getElse());
  --IndentLevel;
  Indent() << "endif\n";
}


#if 0

void ReturnStmt::dump_impl() const {
  std::cerr << "return ";
  if (RetExpr)
    RetExpr->dump();
}

void DeclRefExpr::dump_impl() const {
  std::cerr << "x";
}

void IntegerConstant::dump_impl() const {
  std::cerr << "1";
}

void FloatingConstant::dump_impl() const {
  std::cerr << "1.0";
}

void StringExpr::dump_impl() const {
  if (isWide) std::cerr << 'L';
  std::cerr << '"' << StrData << '"';
}



void ParenExpr::dump_impl() const {
  std::cerr << "'('";
  Val->dump();
  std::cerr << "')'";
}

void UnaryOperator::dump_impl() const {
  std::cerr << getOpcodeStr(Opc);
  Input->dump();
}

void SizeOfAlignOfTypeExpr::dump_impl() const {
  std::cerr << (isSizeof ? "sizeof(" : "alignof(");
  // FIXME: print type.
  std::cerr << "ty)";
}

void ArraySubscriptExpr::dump_impl() const {
  Base->dump();
  std::cerr << "[";
  Idx->dump();
  std::cerr << "]";
}

void CallExpr::dump_impl() const {
  Fn->dump();
  std::cerr << "(";
  for (unsigned i = 0, e = getNumArgs(); i != e; ++i) {
    if (i) std::cerr << ", ";
    getArg(i)->dump();
  }
  std::cerr << ")";
}


void MemberExpr::dump_impl() const {
  Base->dump();
  std::cerr << (isArrow ? "->" : ".");
  
  if (MemberDecl)
    /*TODO: Print MemberDecl*/;
  std::cerr << "member";
}


void CastExpr::dump_impl() const {
  std::cerr << "'('";
  // TODO PRINT TYPE
  std::cerr << "<type>";
  std::cerr << "')'";
  Op->dump();
}


void BinaryOperator::dump_impl() const {
  LHS->dump();
  std::cerr << " " << getOpcodeStr(Opc) << " ";
  RHS->dump();
}

void ConditionalOperator::dump_impl() const {
  Cond->dump();
  std::cerr << " ? ";
  LHS->dump();
  std::cerr << " : ";
  RHS->dump();
}
#endif

//===----------------------------------------------------------------------===//
// Stmt method implementations
//===----------------------------------------------------------------------===//

void Stmt::dump() const {
  print(std::cerr);
}

void Stmt::print(std::ostream &OS) const {
  StmtPrinter P(OS);
  const_cast<Stmt*>(this)->visit(P);
}
