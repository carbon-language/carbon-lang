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
#include "clang/AST/Decl.h"
#include "clang/AST/ExprCXX.h"
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
    
    void PrintStmt(Stmt *S, int SubIndent = 1) {
      IndentLevel += SubIndent;
      if (S && isExpr(S)) {
        // If this is an expr used in a stmt context, indent and newline it.
        Indent();
        S->visit(*this);
        OS << "\n";
      } else if (S) {
        S->visit(*this);
      } else {
        Indent() << "<<<NULL STATEMENT>>>\n";
      }
      IndentLevel -= SubIndent;
    }
    
    void PrintRawCompoundStmt(CompoundStmt *S);

    void PrintExpr(Expr *E) {
      if (E)
        E->visit(*this);
      else
        OS << "<null expr>";
    }
    
    std::ostream &Indent(int Delta = 0) const {
      for (unsigned i = 0, e = IndentLevel+Delta; i != e; ++i)
        OS << "  ";
      return OS;
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

void StmtPrinter::VisitStmt(Stmt *Node) {
  Indent() << "<<unknown stmt type>>\n";
}

/// PrintRawCompoundStmt - Print a compound stmt without indenting the {, and
/// with no newline after the }.
void StmtPrinter::PrintRawCompoundStmt(CompoundStmt *Node) {
  OS << "{\n";
  for (CompoundStmt::body_iterator I = Node->body_begin(), E = Node->body_end();
       I != E; ++I)
    PrintStmt(*I);
  
  Indent() << "}";
}

void StmtPrinter::VisitNullStmt(NullStmt *Node) {
  Indent() << ";\n";
}

void StmtPrinter::VisitDeclStmt(DeclStmt *Node) {
  // FIXME: Need to complete/beautify this...this code simply shows the
  // nodes are where they need to be.
  if (BlockVarDecl *localVar = dyn_cast<BlockVarDecl>(Node->getDecl())) {
    Indent() << localVar->getType().getAsString();
    OS << " " << localVar->getName() << ";\n";
  } else if (TypedefDecl *localType = dyn_cast<TypedefDecl>(Node->getDecl())) {
    Indent() << "typedef " << localType->getUnderlyingType().getAsString();
    OS << " " << localType->getName() << ";\n";
  }
}

void StmtPrinter::VisitCompoundStmt(CompoundStmt *Node) {
  Indent();
  PrintRawCompoundStmt(Node);
}

void StmtPrinter::VisitCaseStmt(CaseStmt *Node) {
  Indent(-1) << "case ";
  PrintExpr(Node->getLHS());
  if (Node->getRHS()) {
    OS << " ... ";
    PrintExpr(Node->getRHS());
  }
  OS << ":\n";
  
  PrintStmt(Node->getSubStmt(), 0);
}

void StmtPrinter::VisitDefaultStmt(DefaultStmt *Node) {
  Indent(-1) << "default:\n";
  PrintStmt(Node->getSubStmt(), 0);
}

void StmtPrinter::VisitLabelStmt(LabelStmt *Node) {
  Indent(-1) << Node->getName() << ":\n";
  PrintStmt(Node->getSubStmt(), 0);
}

void StmtPrinter::VisitIfStmt(IfStmt *If) {
  Indent() << "if (";
  PrintExpr(If->getCond());
  OS << ')';
  
  if (CompoundStmt *CS = dyn_cast<CompoundStmt>(If->getThen())) {
    OS << ' ';
    PrintRawCompoundStmt(CS);
    OS << (If->getElse() ? ' ' : '\n');
  } else {
    OS << '\n';
    PrintStmt(If->getThen());
    if (If->getElse()) Indent();
  }

  if (Stmt *Else = If->getElse()) {
    OS << "else";
    
    if (CompoundStmt *CS = dyn_cast<CompoundStmt>(Else)) {
      OS << ' ';
      PrintRawCompoundStmt(CS);
      OS << '\n';
    } else {
      OS << '\n';
      PrintStmt(If->getElse());
    }
  }
}

void StmtPrinter::VisitSwitchStmt(SwitchStmt *Node) {
  Indent() << "switch (";
  PrintExpr(Node->getCond());
  OS << ")";
  
  // Pretty print compoundstmt bodies (very common).
  if (CompoundStmt *CS = dyn_cast<CompoundStmt>(Node->getBody())) {
    OS << " ";
    PrintRawCompoundStmt(CS);
    OS << "\n";
  } else {
    OS << "\n";
    PrintStmt(Node->getBody());
  }
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
  Indent() << "goto " << Node->getLabel()->getName() << ";\n";
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
  OS << Node->getDecl()->getName();
}

void StmtPrinter::VisitCharacterLiteral(CharacterLiteral *Node) {
  // FIXME: print value.
  OS << "x";
}

void StmtPrinter::VisitIntegerLiteral(IntegerLiteral *Node) {
  bool isSigned = Node->getType()->isSignedIntegerType();
  OS << Node->getValue().toString(10, isSigned);
  
  // Emit suffixes.  Integer literals are always a builtin integer type.
  switch (cast<BuiltinType>(Node->getType().getCanonicalType())->getKind()) {
  default: assert(0 && "Unexpected type for integer literal!");
  case BuiltinType::Int:       break; // no suffix.
  case BuiltinType::UInt:      OS << 'U'; break;
  case BuiltinType::Long:      OS << 'L'; break;
  case BuiltinType::ULong:     OS << "UL"; break;
  case BuiltinType::LongLong:  OS << "LL"; break;
  case BuiltinType::ULongLong: OS << "ULL"; break;
  }
}
void StmtPrinter::VisitFloatingLiteral(FloatingLiteral *Node) {
  // FIXME: print value.
  OS << "~1.0~";
}
void StmtPrinter::VisitStringLiteral(StringLiteral *Str) {
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
  OS << Node->getArgumentType().getAsString() << ")";
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
  
  FieldDecl *Field = Node->getMemberDecl();
  assert(Field && "MemberExpr should alway reference a field!");
  OS << Field->getName();
}
void StmtPrinter::VisitCastExpr(CastExpr *Node) {
  OS << "(" << Node->getDestType().getAsString() << ")";
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
  OS << " : ";
  PrintExpr(Node->getRHS());
}

// GNU extensions.

void StmtPrinter::VisitAddrLabel(AddrLabel *Node) {
  OS << "&&" << Node->getLabel()->getName();
  
}

// C++

void StmtPrinter::VisitCXXCastExpr(CXXCastExpr *Node) {
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
  PrintExpr(Node->getSubExpr());
  OS << ")";
}

void StmtPrinter::VisitCXXBoolLiteralExpr(CXXBoolLiteralExpr *Node) {
  OS << (Node->getValue() ? "true" : "false");
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
