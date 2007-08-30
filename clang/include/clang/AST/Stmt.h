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
#include "llvm/ADT/iterator"
#include <iosfwd>

namespace clang {
  class Expr;
  class Decl;
  class IdentifierInfo;
  class SourceManager;
  class SwitchStmt;
  
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

  /// dump - This does a local dump of the specified AST fragment.  It dumps the
  /// specified node and a few nodes underneath it, but not the whole subtree.
  /// This is useful in a debugger.
  void dump() const;
  void dump(SourceManager &SM) const;

  /// dumpAll - This does a dump of the specified AST fragment and all subtrees.
  void dumpAll() const;
  void dumpAll(SourceManager &SM) const;

  /// dumpPretty/printPretty - These two methods do a "pretty print" of the AST
  /// back to its original source language syntax.
  void dumpPretty() const;
  void printPretty(std::ostream &OS) const;
  
  // Implement isa<T> support.
  static bool classof(const Stmt *) { return true; }  
  
  /// Child Iterators: All subclasses must implement child_begin and child_end
  ///  to permit easy iteration over the substatements/subexpessions of an
  ///  AST node.  This permits easy iteration over all nodes in the AST.
  typedef Stmt**                                               child_iterator;
  typedef Stmt* const *                                  const_child_iterator;
  
  typedef std::reverse_iterator<child_iterator>                
  reverse_child_iterator;
  typedef std::reverse_iterator<const_child_iterator> 
  const_reverse_child_iterator;
  
  virtual child_iterator child_begin() = 0;
  virtual child_iterator child_end()   = 0;
  
  const_child_iterator child_begin() const {
    return (child_iterator) const_cast<Stmt*>(this)->child_begin();
  }
  
  const_child_iterator child_end() const {
    return (child_iterator) const_cast<Stmt*>(this)->child_end();  
  }
  
  reverse_child_iterator child_rbegin() {
    return reverse_child_iterator(child_end());
  }
  
  reverse_child_iterator child_rend() {
    return reverse_child_iterator(child_begin());
  }
  
  const_reverse_child_iterator child_rbegin() const {
    return const_reverse_child_iterator(child_end());
  }
  
  const_reverse_child_iterator child_rend() const {
    return const_reverse_child_iterator(child_begin());
  }
};

/// DeclStmt - Adaptor class for mixing declarations with statements and
/// expressions. For example, CompoundStmt mixes statements, expressions
/// and declarations (variables, types). Another example is ForStmt, where 
/// the first statement can be an expression or a declaration.
///
class DeclStmt : public Stmt {
  Decl *TheDecl;
public:
  DeclStmt(Decl *D) : Stmt(DeclStmtClass), TheDecl(D) {}
  
  const Decl *getDecl() const { return TheDecl; }
  Decl *getDecl() { return TheDecl; }
  
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == DeclStmtClass; 
  }
  static bool classof(const DeclStmt *) { return true; }
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};

/// NullStmt - This is the null statement ";": C99 6.8.3p3.
///
class NullStmt : public Stmt {
  SourceLocation SemiLoc;
public:
  NullStmt(SourceLocation L) : Stmt(NullStmtClass), SemiLoc(L) {}

  SourceLocation getSemiLoc() const { return SemiLoc; }
  
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == NullStmtClass; 
  }
  static bool classof(const NullStmt *) { return true; }
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};

/// CompoundStmt - This represents a group of statements like { stmt stmt }.
///
class CompoundStmt : public Stmt {
  llvm::SmallVector<Stmt*, 16> Body;
public:
  CompoundStmt(Stmt **StmtStart, unsigned NumStmts)
    : Stmt(CompoundStmtClass), Body(StmtStart, StmtStart+NumStmts) {}
  
  bool body_empty() const { return Body.empty(); }
  
  typedef llvm::SmallVector<Stmt*, 16>::iterator body_iterator;
  body_iterator body_begin() { return Body.begin(); }
  body_iterator body_end() { return Body.end(); }
  Stmt *body_back() { return Body.back(); }

  typedef llvm::SmallVector<Stmt*, 16>::const_iterator const_body_iterator;
  const_body_iterator body_begin() const { return Body.begin(); }
  const_body_iterator body_end() const { return Body.end(); }
  const Stmt *body_back() const { return Body.back(); }

  typedef llvm::SmallVector<Stmt*, 16>::reverse_iterator reverse_body_iterator;
  reverse_body_iterator body_rbegin() { return Body.rbegin(); }
  reverse_body_iterator body_rend() { return Body.rend(); }

  typedef llvm::SmallVector<Stmt*, 16>::const_reverse_iterator 
    const_reverse_body_iterator;
  const_reverse_body_iterator body_rbegin() const { return Body.rbegin(); }
  const_reverse_body_iterator body_rend() const { return Body.rend(); }
    
  void push_back(Stmt *S) { Body.push_back(S); }
    
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == CompoundStmtClass; 
  }
  static bool classof(const CompoundStmt *) { return true; }
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};

// SwitchCase is the base class for CaseStmt and DefaultStmt,
class SwitchCase : public Stmt {
  // A pointer to the following CaseStmt or DefaultStmt class,
  // used by SwitchStmt.
  SwitchCase *NextSwitchCase;
  Stmt *SubStmt;
protected:
  SwitchCase(StmtClass SC, Stmt* substmt) : Stmt(SC), NextSwitchCase(0),
                                            SubStmt(substmt) {}
  
public:
  const SwitchCase *getNextSwitchCase() const { return NextSwitchCase; }

  SwitchCase *getNextSwitchCase() { return NextSwitchCase; }

  void setNextSwitchCase(SwitchCase *SC) { NextSwitchCase = SC; }
  
  Stmt *getSubStmt() { return SubStmt; }
  
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == CaseStmtClass || 
    T->getStmtClass() == DefaultStmtClass;
  }
  static bool classof(const SwitchCase *) { return true; }
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};

class CaseStmt : public SwitchCase {
  Expr *LHSVal;
  Expr *RHSVal;  // Non-null for GNU "case 1 ... 4" extension
public:
  CaseStmt(Expr *lhs, Expr *rhs, Stmt *substmt) 
    : SwitchCase(CaseStmtClass,substmt), LHSVal(lhs), RHSVal(rhs) {}
  
  Expr *getLHS() { return LHSVal; }
  Expr *getRHS() { return RHSVal; }

  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == CaseStmtClass; 
  }
  static bool classof(const CaseStmt *) { return true; }
};

class DefaultStmt : public SwitchCase {
  SourceLocation DefaultLoc;
public:
  DefaultStmt(SourceLocation DL, Stmt *substmt) : 
    SwitchCase(DefaultStmtClass,substmt), DefaultLoc(DL) {}
  
  SourceLocation getDefaultLoc() const { return DefaultLoc; }

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
  
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == LabelStmtClass; 
  }
  static bool classof(const LabelStmt *) { return true; }
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};


/// IfStmt - This represents an if/then/else.
///
class IfStmt : public Stmt {
  enum { COND, THEN, ELSE, END_EXPR };
  Stmt* SubExprs[END_EXPR];
public:
  IfStmt(Expr *cond, Stmt *then, Stmt *elsev = 0) : Stmt(IfStmtClass)  {
    SubExprs[COND] = reinterpret_cast<Stmt*>(cond);
    SubExprs[THEN] = then;
    SubExprs[ELSE] = elsev;
  }
  
  const Expr *getCond() const { return reinterpret_cast<Expr*>(SubExprs[COND]);}
  const Stmt *getThen() const { return SubExprs[THEN]; }
  const Stmt *getElse() const { return SubExprs[ELSE]; }

  Expr *getCond() { return reinterpret_cast<Expr*>(SubExprs[COND]); }
  Stmt *getThen() { return SubExprs[THEN]; }
  Stmt *getElse() { return SubExprs[ELSE]; }
  
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == IfStmtClass; 
  }
  static bool classof(const IfStmt *) { return true; }
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};

/// SwitchStmt - This represents a 'switch' stmt.
///
class SwitchStmt : public Stmt {
  enum { COND, BODY, END_EXPR };
  Stmt* SubExprs[END_EXPR];  
  // This points to a linked list of case and default statements.
  SwitchCase *FirstCase;
public:
  SwitchStmt(Expr *cond) : Stmt(SwitchStmtClass), FirstCase(0) {
      SubExprs[COND] = reinterpret_cast<Stmt*>(cond);
      SubExprs[BODY] = NULL;
    }
  
  const Expr *getCond() const { return reinterpret_cast<Expr*>(SubExprs[COND]);}
  const Stmt *getBody() const { return SubExprs[BODY]; }
  const SwitchCase *getSwitchCaseList() const { return FirstCase; }

  Expr *getCond() { return reinterpret_cast<Expr*>(SubExprs[COND]);}
  Stmt *getBody() { return SubExprs[BODY]; }
  SwitchCase *getSwitchCaseList() { return FirstCase; }

  void setBody(Stmt *S) { SubExprs[BODY] = S; }  
  
  void addSwitchCase(SwitchCase *SC) {
    if (FirstCase)
      SC->setNextSwitchCase(FirstCase);

    FirstCase = SC;
  }
  
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == SwitchStmtClass; 
  }
  static bool classof(const SwitchStmt *) { return true; }
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};


/// WhileStmt - This represents a 'while' stmt.
///
class WhileStmt : public Stmt {
  enum { COND, BODY, END_EXPR };
  Stmt* SubExprs[END_EXPR];
public:
  WhileStmt(Expr *cond, Stmt *body) : Stmt(WhileStmtClass) {
    SubExprs[COND] = reinterpret_cast<Stmt*>(cond);
    SubExprs[BODY] = body;
  }
  
  Expr *getCond() { return reinterpret_cast<Expr*>(SubExprs[COND]); }
  const Expr *getCond() const { return reinterpret_cast<Expr*>(SubExprs[COND]);}
  Stmt *getBody() { return SubExprs[BODY]; }
  const Stmt *getBody() const { return SubExprs[BODY]; }
  
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == WhileStmtClass; 
  }
  static bool classof(const WhileStmt *) { return true; }
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};

/// DoStmt - This represents a 'do/while' stmt.
///
class DoStmt : public Stmt {
  enum { COND, BODY, END_EXPR };
  Stmt* SubExprs[END_EXPR];
public:
  DoStmt(Stmt *body, Expr *cond) : Stmt(DoStmtClass) {
    SubExprs[COND] = reinterpret_cast<Stmt*>(cond);
    SubExprs[BODY] = body;
  }  
  
  Expr *getCond() { return reinterpret_cast<Expr*>(SubExprs[COND]); }
  const Expr *getCond() const { return reinterpret_cast<Expr*>(SubExprs[COND]);}
  Stmt *getBody() { return SubExprs[BODY]; }
  const Stmt *getBody() const { return SubExprs[BODY]; }  
  
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == DoStmtClass; 
  }
  static bool classof(const DoStmt *) { return true; }

  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};


/// ForStmt - This represents a 'for (init;cond;inc)' stmt.  Note that any of
/// the init/cond/inc parts of the ForStmt will be null if they were not
/// specified in the source.
///
class ForStmt : public Stmt {
  enum { INIT, COND, INC, BODY, END_EXPR };
  Stmt* SubExprs[END_EXPR]; // SubExprs[INIT] is an expression or declstmt.
public:
  ForStmt(Stmt *Init, Expr *Cond, Expr *Inc, Stmt *Body) : Stmt(ForStmtClass) {
    SubExprs[INIT] = Init;
    SubExprs[COND] = reinterpret_cast<Stmt*>(Cond);
    SubExprs[INC] = reinterpret_cast<Stmt*>(Inc);
    SubExprs[BODY] = Body;
  }
  
  Stmt *getInit() { return SubExprs[INIT]; }
  Expr *getCond() { return reinterpret_cast<Expr*>(SubExprs[COND]); }
  Expr *getInc()  { return reinterpret_cast<Expr*>(SubExprs[INC]); }
  Stmt *getBody() { return SubExprs[BODY]; }

  const Stmt *getInit() const { return SubExprs[INIT]; }
  const Expr *getCond() const { return reinterpret_cast<Expr*>(SubExprs[COND]);}
  const Expr *getInc()  const { return reinterpret_cast<Expr*>(SubExprs[INC]); }
  const Stmt *getBody() const { return SubExprs[BODY]; }
  
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == ForStmtClass; 
  }
  static bool classof(const ForStmt *) { return true; }
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};

/// GotoStmt - This represents a direct goto.
///
class GotoStmt : public Stmt {
  LabelStmt *Label;
public:
  GotoStmt(LabelStmt *label) : Stmt(GotoStmtClass), Label(label) {}
  
  LabelStmt *getLabel() const { return Label; }
  
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == GotoStmtClass; 
  }
  static bool classof(const GotoStmt *) { return true; }
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};

/// IndirectGotoStmt - This represents an indirect goto.
///
class IndirectGotoStmt : public Stmt {
  Expr *Target;
public:
  IndirectGotoStmt(Expr *target) : Stmt(IndirectGotoStmtClass), Target(target){}
  
  Expr *getTarget() { return Target; }
  const Expr *getTarget() const { return Target; }
  
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == IndirectGotoStmtClass; 
  }
  static bool classof(const IndirectGotoStmt *) { return true; }
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};


/// ContinueStmt - This represents a continue.
///
class ContinueStmt : public Stmt {
public:
  ContinueStmt() : Stmt(ContinueStmtClass) {}
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == ContinueStmtClass; 
  }
  static bool classof(const ContinueStmt *) { return true; }
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};

/// BreakStmt - This represents a break.
///
class BreakStmt : public Stmt {
public:
  BreakStmt() : Stmt(BreakStmtClass) {}
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == BreakStmtClass; 
  }
  static bool classof(const BreakStmt *) { return true; }
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};


/// ReturnStmt - This represents a return, optionally of an expression.
///
class ReturnStmt : public Stmt {
  Expr *RetExpr;
public:
  ReturnStmt(Expr *E = 0) : Stmt(ReturnStmtClass), RetExpr(E) {}
  
  const Expr *getRetValue() const { return RetExpr; }
  Expr *getRetValue() { return RetExpr; }
  
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == ReturnStmtClass; 
  }
  static bool classof(const ReturnStmt *) { return true; }
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};

}  // end namespace clang

#endif
