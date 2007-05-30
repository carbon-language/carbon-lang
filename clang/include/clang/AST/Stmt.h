//===--- Stmt.h - Classes for representing statements -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the Stmt interface and subclasses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_STMT_H
#define LLVM_CLANG_AST_STMT_H

#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/SmallVector.h"
#include <iosfwd>

namespace llvm {
namespace clang {
  class Expr;
  class Decl;
  class IdentifierInfo;
  class StmtVisitor;
  
/// Stmt - This represents one statement.
///
class Stmt {
public:
  enum StmtClass {
#define STMT(N, CLASS, PARENT) CLASS##Class = N,
#define FIRST_STMT(N) firstStmtConstant = N,
#define LAST_STMT(N) lastStmtConstant = N,
#define FIRST_EXPR(N) firstExprConstant = N,
#define LAST_EXPR(N) lastExprConstant = N
#include "clang/AST/StmtNodes.def"
};
private:
  const StmtClass sClass;
public:
  Stmt(StmtClass SC) : sClass(SC) { 
    if (Stmt::CollectingStats()) Stmt::addStmtClass(SC);
  }
  virtual ~Stmt() {}

  StmtClass getStmtClass() const { return sClass; }
  const char *getStmtClassName() const;

  // global temp stats (until we have a per-module visitor)
  static void addStmtClass(const StmtClass s);
  static bool CollectingStats(bool enable=false);
  static void PrintStats();
  
  void dump() const;
  void print(std::ostream &OS) const;
  
  // Implement visitor support.
  virtual void visit(StmtVisitor &Visitor);

  // Implement isa<T> support.
  static bool classof(const Stmt *) { return true; }
};

/// DeclStmt - Adaptor class for mixing declarations with statements and
/// expressions. For example, CompoundStmt mixes statements, expressions
/// and declarations (variables, types). Another example is ForStmt, where 
/// the first statement can be an expression or a declaration.
///
class DeclStmt : public Stmt {
  Decl *BlockVarOrTypedefDecl;
public:
  DeclStmt(Decl *D) : Stmt(DeclStmtClass), BlockVarOrTypedefDecl(D) {}
  
  Decl *getDecl() const { return BlockVarOrTypedefDecl; }
  
  virtual void visit(StmtVisitor &Visitor);
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == DeclStmtClass; 
  }
  static bool classof(const DeclStmt *) { return true; }
};

/// NullStmt - This is the null statement ";": C99 6.8.3p3.
///
class NullStmt : public Stmt {
  SourceLocation SemiLoc;
public:
  NullStmt(SourceLocation L) : Stmt(NullStmtClass), SemiLoc(L) {}

  SourceLocation getSemiLoc() const { return SemiLoc; }
  
  virtual void visit(StmtVisitor &Visitor);
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == NullStmtClass; 
  }
  static bool classof(const NullStmt *) { return true; }
};

/// CompoundStmt - This represents a group of statements like { stmt stmt }.
///
class CompoundStmt : public Stmt {
  SmallVector<Stmt*, 16> Body;
public:
  CompoundStmt(Stmt **StmtStart, unsigned NumStmts)
    : Stmt(CompoundStmtClass), Body(StmtStart, StmtStart+NumStmts) {}
  
  typedef SmallVector<Stmt*, 16>::iterator body_iterator;
  body_iterator body_begin() { return Body.begin(); }
  body_iterator body_end() { return Body.end(); }

  typedef SmallVector<Stmt*, 16>::const_iterator const_body_iterator;
  const_body_iterator body_begin() const { return Body.begin(); }
  const_body_iterator body_end() const { return Body.end(); }
  
  void push_back(Stmt *S) { Body.push_back(S); }
    
  virtual void visit(StmtVisitor &Visitor);
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == CompoundStmtClass; 
  }
  static bool classof(const CompoundStmt *) { return true; }
};

class CaseStmt : public Stmt {
  Expr *LHSVal;
  Expr *RHSVal;  // Non-null for GNU "case 1 ... 4" extension
  Stmt *SubStmt;
public:
  CaseStmt(Expr *lhs, Expr *rhs, Stmt *substmt) 
    : Stmt(CaseStmtClass), LHSVal(lhs), RHSVal(rhs), SubStmt(substmt) {}
  
  Expr *getLHS() { return LHSVal; }
  Expr *getRHS() { return RHSVal; }
  Stmt *getSubStmt() { return SubStmt; }

  virtual void visit(StmtVisitor &Visitor);
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == CaseStmtClass; 
  }
  static bool classof(const CaseStmt *) { return true; }
};

class DefaultStmt : public Stmt {
  Stmt *SubStmt;
public:
  DefaultStmt(Stmt *substmt) : Stmt(DefaultStmtClass), SubStmt(substmt) {}
  
  Stmt *getSubStmt() { return SubStmt; }

  virtual void visit(StmtVisitor &Visitor);
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == DefaultStmtClass; 
  }
  static bool classof(const DefaultStmt *) { return true; }
};

class LabelStmt : public Stmt {
  SourceLocation IdentLoc;
  IdentifierInfo *Label;
  Stmt *SubStmt;
public:
  LabelStmt(SourceLocation IL, IdentifierInfo *label, Stmt *substmt)
    : Stmt(LabelStmtClass), IdentLoc(IL), Label(label), SubStmt(substmt) {}
  
  SourceLocation getIdentLoc() const { return IdentLoc; }
  IdentifierInfo *getID() const { return Label; }
  const char *getName() const;
  Stmt *getSubStmt() { return SubStmt; }
  const Stmt *getSubStmt() const { return SubStmt; }

  void setIdentLoc(SourceLocation L) { IdentLoc = L; }
  void setSubStmt(Stmt *SS) { SubStmt = SS; }
  
  virtual void visit(StmtVisitor &Visitor);
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == LabelStmtClass; 
  }
  static bool classof(const LabelStmt *) { return true; }
};


/// IfStmt - This represents an if/then/else.
///
class IfStmt : public Stmt {
  Expr *Cond;
  Stmt *Then, *Else;
public:
  IfStmt(Expr *cond, Stmt *then, Stmt *elsev = 0)
    : Stmt(IfStmtClass), Cond(cond), Then(then), Else(elsev) {}
  
  const Expr *getCond() const { return Cond; }
  const Stmt *getThen() const { return Then; }
  const Stmt *getElse() const { return Else; }

  Expr *getCond() { return Cond; }
  Stmt *getThen() { return Then; }
  Stmt *getElse() { return Else; }
  
  virtual void visit(StmtVisitor &Visitor);
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == IfStmtClass; 
  }
  static bool classof(const IfStmt *) { return true; }
};

/// SwitchStmt - This represents a 'switch' stmt.
///
class SwitchStmt : public Stmt {
  Expr *Cond;
  Stmt *Body;
public:
  SwitchStmt(Expr *cond, Stmt *body)
    : Stmt(SwitchStmtClass), Cond(cond), Body(body) {}
  
  Expr *getCond() { return Cond; }
  Stmt *getBody() { return Body; }
  
  virtual void visit(StmtVisitor &Visitor);
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == SwitchStmtClass; 
  }
  static bool classof(const SwitchStmt *) { return true; }
};


/// WhileStmt - This represents a 'while' stmt.
///
class WhileStmt : public Stmt {
  Expr *Cond;
  Stmt *Body;
public:
  WhileStmt(Expr *cond, Stmt *body)
    : Stmt(WhileStmtClass), Cond(cond), Body(body) {}
  
  Expr *getCond() { return Cond; }
  Stmt *getBody() { return Body; }
  
  virtual void visit(StmtVisitor &Visitor);
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == WhileStmtClass; 
  }
  static bool classof(const WhileStmt *) { return true; }
};

/// DoStmt - This represents a 'do/while' stmt.
///
class DoStmt : public Stmt {
  Stmt *Body;
  Expr *Cond;
public:
  DoStmt(Stmt *body, Expr *cond)
    : Stmt(DoStmtClass), Body(body), Cond(cond) {}
  
  Stmt *getBody() { return Body; }
  Expr *getCond() { return Cond; }
  
  virtual void visit(StmtVisitor &Visitor);
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == DoStmtClass; 
  }
  static bool classof(const DoStmt *) { return true; }
};


/// ForStmt - This represents a 'for' stmt.
///
class ForStmt : public Stmt {
  Stmt *First;  // Expression or decl.
  Expr *Second, *Third;
  Stmt *Body;
public:
  ForStmt(Stmt *first, Expr *second, Expr *third, Stmt *body)
    : Stmt(ForStmtClass),
      First(first), Second(second), Third(third), Body(body) {}
  
  Stmt *getFirst() { return First; }
  Expr *getSecond() { return Second; }
  Expr *getThird() { return Third; }
  Stmt *getBody() { return Body; }
 
  virtual void visit(StmtVisitor &Visitor);
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == ForStmtClass; 
  }
  static bool classof(const ForStmt *) { return true; }
};

/// GotoStmt - This represents a direct goto.
///
class GotoStmt : public Stmt {
  LabelStmt *Label;
public:
  GotoStmt(LabelStmt *label) : Stmt(GotoStmtClass), Label(label) {}
  
  LabelStmt *getLabel() const { return Label; }
  
  virtual void visit(StmtVisitor &Visitor);
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == GotoStmtClass; 
  }
  static bool classof(const GotoStmt *) { return true; }
};

/// IndirectGotoStmt - This represents an indirect goto.
///
class IndirectGotoStmt : public Stmt {
  Expr *Target;
public:
  IndirectGotoStmt(Expr *target) : Stmt(IndirectGotoStmtClass), 
                                   Target(target) {}
  
  Expr *getTarget() { return Target; }
  
  virtual void visit(StmtVisitor &Visitor);
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == IndirectGotoStmtClass; 
  }
  static bool classof(const IndirectGotoStmt *) { return true; }
};


/// ContinueStmt - This represents a continue.
///
class ContinueStmt : public Stmt {
public:
  ContinueStmt() : Stmt(ContinueStmtClass) {}
  virtual void visit(StmtVisitor &Visitor);
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == ContinueStmtClass; 
  }
  static bool classof(const ContinueStmt *) { return true; }
};

/// BreakStmt - This represents a break.
///
class BreakStmt : public Stmt {
public:
  BreakStmt() : Stmt(BreakStmtClass) {}
  virtual void visit(StmtVisitor &Visitor);
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == BreakStmtClass; 
  }
  static bool classof(const BreakStmt *) { return true; }
};


/// ReturnStmt - This represents a return, optionally of an expression.
///
class ReturnStmt : public Stmt {
  Expr *RetExpr;
public:
  ReturnStmt(Expr *E = 0) : Stmt(ReturnStmtClass), RetExpr(E) {}
  
  Expr *getRetValue() { return RetExpr; }
  
  virtual void visit(StmtVisitor &Visitor);
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == ReturnStmtClass; 
  }
  static bool classof(const ReturnStmt *) { return true; }
};


  
}  // end namespace clang
}  // end namespace llvm

#endif
