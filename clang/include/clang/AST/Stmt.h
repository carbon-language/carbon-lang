//===--- Stmt.h - Classes for representing statements -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the Stmt interface and subclasses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_STMT_H
#define LLVM_CLANG_AST_STMT_H

#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/AST/StmtIterator.h"
#include "clang/AST/DeclGroup.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Bitcode/SerializationFwd.h"
#include <string>

using llvm::dyn_cast_or_null;

namespace clang {
  class ASTContext;
  class Expr;
  class Decl;
  class ScopedDecl;
  class IdentifierInfo;
  class SourceManager;
  class StringLiteral;
  class SwitchStmt;
  class PrinterHelper;
    
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
  
protected:
  /// DestroyChildren - Invoked by destructors of subclasses of Stmt to
  ///  recursively release child AST nodes.
  void DestroyChildren(ASTContext& Ctx);
  
public:
  Stmt(StmtClass SC) : sClass(SC) { 
    if (Stmt::CollectingStats()) Stmt::addStmtClass(SC);
  }
  virtual ~Stmt() {}
  
  virtual void Destroy(ASTContext& Ctx);

  StmtClass getStmtClass() const { return sClass; }
  const char *getStmtClassName() const;
  
  /// SourceLocation tokens are not useful in isolation - they are low level
  /// value objects created/interpreted by SourceManager. We assume AST
  /// clients will have a pointer to the respective SourceManager.
  virtual SourceRange getSourceRange() const = 0;
  SourceLocation getLocStart() const { return getSourceRange().getBegin(); }
  SourceLocation getLocEnd() const { return getSourceRange().getEnd(); }

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
  void printPretty(llvm::raw_ostream &OS, PrinterHelper* = NULL) const;
  
  /// viewAST - Visualize an AST rooted at this Stmt* using GraphViz.  Only
  ///   works on systems with GraphViz (Mac OS X) or dot+gv installed.
  void viewAST() const;
  
  // Implement isa<T> support.
  static bool classof(const Stmt *) { return true; }  
  
  /// hasImplicitControlFlow - Some statements (e.g. short circuited operations)
  ///  contain implicit control-flow in the order their subexpressions
  ///  are evaluated.  This predicate returns true if this statement has
  ///  such implicit control-flow.  Such statements are also specially handled
  ///  within CFGs.
  bool hasImplicitControlFlow() const;

  /// Child Iterators: All subclasses must implement child_begin and child_end
  ///  to permit easy iteration over the substatements/subexpessions of an
  ///  AST node.  This permits easy iteration over all nodes in the AST.
  typedef StmtIterator       child_iterator;
  typedef ConstStmtIterator  const_child_iterator;
  
  virtual child_iterator child_begin() = 0;
  virtual child_iterator child_end()   = 0;
  
  const_child_iterator child_begin() const {
    return const_child_iterator(const_cast<Stmt*>(this)->child_begin());
  }
  
  const_child_iterator child_end() const {
    return const_child_iterator(const_cast<Stmt*>(this)->child_end());
  }
  
  void Emit(llvm::Serializer& S) const;
  static Stmt* Create(llvm::Deserializer& D, ASTContext& C);

  virtual void EmitImpl(llvm::Serializer& S) const {
    // This method will eventually be a pure-virtual function.
    assert (false && "Not implemented.");
  }
};

/// DeclStmt - Adaptor class for mixing declarations with statements and
/// expressions. For example, CompoundStmt mixes statements, expressions
/// and declarations (variables, types). Another example is ForStmt, where 
/// the first statement can be an expression or a declaration.
///
class DeclStmt : public Stmt {
protected:
  DeclGroupOwningRef DG;
  SourceLocation StartLoc, EndLoc;
public:
  DeclStmt(DeclGroupOwningRef& dg, SourceLocation startLoc, 
           SourceLocation endLoc) : Stmt(DeclStmtClass), DG(dg),
                                    StartLoc(startLoc), EndLoc(endLoc) {}
  
  virtual void Destroy(ASTContext& Ctx);

  // hasSolitaryDecl - This method returns true if this DeclStmt refers
  // to a single Decl.
  bool hasSolitaryDecl() const {
    return DG.hasSolitaryDecl();
  }
 
  const ScopedDecl* getSolitaryDecl() const {
    assert (hasSolitaryDecl() &&
            "Caller assumes this DeclStmt points to one Decl*");
    return llvm::cast<ScopedDecl>(*DG.begin());
  }
  
  ScopedDecl* getSolitaryDecl() {
    assert (hasSolitaryDecl() &&
            "Caller assumes this DeclStmt points to one Decl*");
    return llvm::cast<ScopedDecl>(*DG.begin());
  }  

  SourceLocation getStartLoc() const { return StartLoc; }
  SourceLocation getEndLoc() const { return EndLoc; }
  
  SourceRange getSourceRange() const {
    return SourceRange(StartLoc, EndLoc);
  }
  
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == DeclStmtClass; 
  }
  static bool classof(const DeclStmt *) { return true; }
  
  // Iterators over subexpressions.
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
  
  class decl_iterator {
    DeclGroupRef::iterator I;
  public:
    decl_iterator(DeclGroupRef::iterator i) : I(i) {}
    decl_iterator& operator++() { ++I; return *this; }
    bool operator==(const decl_iterator& R) const {
      return R.I == I;
    }
    bool operator!=(const decl_iterator& R) const {
      return R.I != I;
    }
    ScopedDecl* operator*() const {
      return llvm::cast<ScopedDecl>(*I);
    }
  };
    
  class const_decl_iterator {
    DeclGroupRef::const_iterator I;
  public:
    const_decl_iterator(DeclGroupRef::const_iterator i) : I(i) {}
    const_decl_iterator& operator++() { ++I; return *this; }
    bool operator==(const const_decl_iterator& R) const {
      return R.I == I;
    }
    bool operator!=(const const_decl_iterator& R) const {
      return R.I != I;
    }
    ScopedDecl* operator*() const {
      return llvm::cast<ScopedDecl>(*I);
    }
  };
  
  decl_iterator decl_begin() { return DG.begin(); }
  decl_iterator decl_end() { return DG.end(); }
  const_decl_iterator decl_begin() const { return DG.begin(); }
  const_decl_iterator decl_end() const { return DG.end(); }
  
  // Serialization.  
  virtual void EmitImpl(llvm::Serializer& S) const;
  static DeclStmt* CreateImpl(llvm::Deserializer& D, ASTContext& C);
};

/// NullStmt - This is the null statement ";": C99 6.8.3p3.
///
class NullStmt : public Stmt {
  SourceLocation SemiLoc;
public:
  NullStmt(SourceLocation L) : Stmt(NullStmtClass), SemiLoc(L) {}

  SourceLocation getSemiLoc() const { return SemiLoc; }

  virtual SourceRange getSourceRange() const { return SourceRange(SemiLoc); }
  
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == NullStmtClass; 
  }
  static bool classof(const NullStmt *) { return true; }
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
  
  virtual void EmitImpl(llvm::Serializer& S) const;
  static NullStmt* CreateImpl(llvm::Deserializer& D, ASTContext& C);
};

/// CompoundStmt - This represents a group of statements like { stmt stmt }.
///
class CompoundStmt : public Stmt {
  llvm::SmallVector<Stmt*, 16> Body;
  SourceLocation LBracLoc, RBracLoc;
public:
  CompoundStmt(Stmt **StmtStart, unsigned NumStmts, 
               SourceLocation LB, SourceLocation RB)
    : Stmt(CompoundStmtClass), Body(StmtStart, StmtStart+NumStmts),
      LBracLoc(LB), RBracLoc(RB) {}
    
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
  
  virtual SourceRange getSourceRange() const { 
    return SourceRange(LBracLoc, RBracLoc); 
  }
  
  SourceLocation getLBracLoc() const { return LBracLoc; }
  SourceLocation getRBracLoc() const { return RBracLoc; }
  
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == CompoundStmtClass; 
  }
  static bool classof(const CompoundStmt *) { return true; }
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
  
  virtual void EmitImpl(llvm::Serializer& S) const;
  static CompoundStmt* CreateImpl(llvm::Deserializer& D, ASTContext& C);
};

// SwitchCase is the base class for CaseStmt and DefaultStmt,
class SwitchCase : public Stmt {
protected:
  // A pointer to the following CaseStmt or DefaultStmt class,
  // used by SwitchStmt.
  SwitchCase *NextSwitchCase;

  SwitchCase(StmtClass SC) : Stmt(SC), NextSwitchCase(0) {}
  
public:
  const SwitchCase *getNextSwitchCase() const { return NextSwitchCase; }

  SwitchCase *getNextSwitchCase() { return NextSwitchCase; }

  void setNextSwitchCase(SwitchCase *SC) { NextSwitchCase = SC; }

  virtual Stmt* v_getSubStmt() = 0;  
  Stmt *getSubStmt() { return v_getSubStmt(); }

  virtual SourceRange getSourceRange() const { return SourceRange(); }
  
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == CaseStmtClass || 
    T->getStmtClass() == DefaultStmtClass;
  }
  static bool classof(const SwitchCase *) { return true; }
};

class CaseStmt : public SwitchCase {
  enum { SUBSTMT, LHS, RHS, END_EXPR };
  Stmt* SubExprs[END_EXPR];  // The expression for the RHS is Non-null for 
                             // GNU "case 1 ... 4" extension
  SourceLocation CaseLoc;
public:
  CaseStmt(Expr *lhs, Expr *rhs, Stmt *substmt, SourceLocation caseLoc) 
    : SwitchCase(CaseStmtClass) {
    SubExprs[SUBSTMT] = substmt;
    SubExprs[LHS] = reinterpret_cast<Stmt*>(lhs);
    SubExprs[RHS] = reinterpret_cast<Stmt*>(rhs);
    CaseLoc = caseLoc;
  }
  
  SourceLocation getCaseLoc() const { return CaseLoc; }
  
  Expr *getLHS() { return reinterpret_cast<Expr*>(SubExprs[LHS]); }
  Expr *getRHS() { return reinterpret_cast<Expr*>(SubExprs[RHS]); }
  Stmt *getSubStmt() { return SubExprs[SUBSTMT]; }
  virtual Stmt* v_getSubStmt() { return getSubStmt(); }
  const Expr *getLHS() const { 
    return reinterpret_cast<const Expr*>(SubExprs[LHS]); 
  }
  const Expr *getRHS() const { 
    return reinterpret_cast<const Expr*>(SubExprs[RHS]); 
  }
  const Stmt *getSubStmt() const { return SubExprs[SUBSTMT]; }

  void setSubStmt(Stmt *S) { SubExprs[SUBSTMT] = S; }
  void setLHS(Expr *Val) { SubExprs[LHS] = reinterpret_cast<Stmt*>(Val); }
  void setRHS(Expr *Val) { SubExprs[RHS] = reinterpret_cast<Stmt*>(Val); }
  
  
  virtual SourceRange getSourceRange() const { 
    return SourceRange(CaseLoc, SubExprs[SUBSTMT]->getLocEnd()); 
  }
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == CaseStmtClass; 
  }
  static bool classof(const CaseStmt *) { return true; }
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
  
  virtual void EmitImpl(llvm::Serializer& S) const;
  static CaseStmt* CreateImpl(llvm::Deserializer& D, ASTContext& C);
};

class DefaultStmt : public SwitchCase {
  Stmt* SubStmt;
  SourceLocation DefaultLoc;
public:
  DefaultStmt(SourceLocation DL, Stmt *substmt) : 
    SwitchCase(DefaultStmtClass), SubStmt(substmt), DefaultLoc(DL) {}
    
  Stmt *getSubStmt() { return SubStmt; }
  virtual Stmt* v_getSubStmt() { return getSubStmt(); }
  const Stmt *getSubStmt() const { return SubStmt; }
    
  SourceLocation getDefaultLoc() const { return DefaultLoc; }

  virtual SourceRange getSourceRange() const { 
    return SourceRange(DefaultLoc, SubStmt->getLocEnd()); 
  }
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == DefaultStmtClass; 
  }
  static bool classof(const DefaultStmt *) { return true; }
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
  
  virtual void EmitImpl(llvm::Serializer& S) const;
  static DefaultStmt* CreateImpl(llvm::Deserializer& D, ASTContext& C);
};

class LabelStmt : public Stmt {
  IdentifierInfo *Label;
  Stmt *SubStmt;
  SourceLocation IdentLoc;
public:
  LabelStmt(SourceLocation IL, IdentifierInfo *label, Stmt *substmt) 
    : Stmt(LabelStmtClass), Label(label), 
      SubStmt(substmt), IdentLoc(IL) {}
  
  SourceLocation getIdentLoc() const { return IdentLoc; }
  IdentifierInfo *getID() const { return Label; }
  const char *getName() const;
  Stmt *getSubStmt() { return SubStmt; }
  const Stmt *getSubStmt() const { return SubStmt; }

  void setIdentLoc(SourceLocation L) { IdentLoc = L; }
  void setSubStmt(Stmt *SS) { SubStmt = SS; }

  virtual SourceRange getSourceRange() const { 
    return SourceRange(IdentLoc, SubStmt->getLocEnd()); 
  }  
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == LabelStmtClass; 
  }
  static bool classof(const LabelStmt *) { return true; }
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
  
  virtual void EmitImpl(llvm::Serializer& S) const;
  static LabelStmt* CreateImpl(llvm::Deserializer& D, ASTContext& C);
};


/// IfStmt - This represents an if/then/else.
///
class IfStmt : public Stmt {
  enum { COND, THEN, ELSE, END_EXPR };
  Stmt* SubExprs[END_EXPR];
  SourceLocation IfLoc;
public:
  IfStmt(SourceLocation IL, Expr *cond, Stmt *then, Stmt *elsev = 0) 
    : Stmt(IfStmtClass)  {
    SubExprs[COND] = reinterpret_cast<Stmt*>(cond);
    SubExprs[THEN] = then;
    SubExprs[ELSE] = elsev;
    IfLoc = IL;
  }
  
  const Expr *getCond() const { return reinterpret_cast<Expr*>(SubExprs[COND]);}
  const Stmt *getThen() const { return SubExprs[THEN]; }
  const Stmt *getElse() const { return SubExprs[ELSE]; }

  Expr *getCond() { return reinterpret_cast<Expr*>(SubExprs[COND]); }
  Stmt *getThen() { return SubExprs[THEN]; }
  Stmt *getElse() { return SubExprs[ELSE]; }

  virtual SourceRange getSourceRange() const { 
    if (SubExprs[ELSE])
      return SourceRange(IfLoc, SubExprs[ELSE]->getLocEnd());
    else
      return SourceRange(IfLoc, SubExprs[THEN]->getLocEnd());
  }
  
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == IfStmtClass; 
  }
  static bool classof(const IfStmt *) { return true; }
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
  
  virtual void EmitImpl(llvm::Serializer& S) const;
  static IfStmt* CreateImpl(llvm::Deserializer& D, ASTContext& C);
};

/// SwitchStmt - This represents a 'switch' stmt.
///
class SwitchStmt : public Stmt {
  enum { COND, BODY, END_EXPR };
  Stmt* SubExprs[END_EXPR];  
  // This points to a linked list of case and default statements.
  SwitchCase *FirstCase;
  SourceLocation SwitchLoc;
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

  void setBody(Stmt *S, SourceLocation SL) { 
    SubExprs[BODY] = S; 
    SwitchLoc = SL;
  }  
  void addSwitchCase(SwitchCase *SC) {
    if (FirstCase)
      SC->setNextSwitchCase(FirstCase);

    FirstCase = SC;
  }
  virtual SourceRange getSourceRange() const { 
    return SourceRange(SwitchLoc, SubExprs[BODY]->getLocEnd()); 
  }
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == SwitchStmtClass; 
  }
  static bool classof(const SwitchStmt *) { return true; }
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
  
  virtual void EmitImpl(llvm::Serializer& S) const;
  static SwitchStmt* CreateImpl(llvm::Deserializer& D, ASTContext& C);
};


/// WhileStmt - This represents a 'while' stmt.
///
class WhileStmt : public Stmt {
  enum { COND, BODY, END_EXPR };
  Stmt* SubExprs[END_EXPR];
  SourceLocation WhileLoc;
public:
  WhileStmt(Expr *cond, Stmt *body, SourceLocation WL) : Stmt(WhileStmtClass) {
    SubExprs[COND] = reinterpret_cast<Stmt*>(cond);
    SubExprs[BODY] = body;
    WhileLoc = WL;
  }
  
  Expr *getCond() { return reinterpret_cast<Expr*>(SubExprs[COND]); }
  const Expr *getCond() const { return reinterpret_cast<Expr*>(SubExprs[COND]);}
  Stmt *getBody() { return SubExprs[BODY]; }
  const Stmt *getBody() const { return SubExprs[BODY]; }

  virtual SourceRange getSourceRange() const { 
    return SourceRange(WhileLoc, SubExprs[BODY]->getLocEnd()); 
  }
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == WhileStmtClass; 
  }
  static bool classof(const WhileStmt *) { return true; }
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
  
  virtual void EmitImpl(llvm::Serializer& S) const;
  static WhileStmt* CreateImpl(llvm::Deserializer& D, ASTContext& C);
};

/// DoStmt - This represents a 'do/while' stmt.
///
class DoStmt : public Stmt {
  enum { COND, BODY, END_EXPR };
  Stmt* SubExprs[END_EXPR];
  SourceLocation DoLoc;
public:
  DoStmt(Stmt *body, Expr *cond, SourceLocation DL) 
    : Stmt(DoStmtClass), DoLoc(DL) {
    SubExprs[COND] = reinterpret_cast<Stmt*>(cond);
    SubExprs[BODY] = body;
    DoLoc = DL;
  }  
  
  Expr *getCond() { return reinterpret_cast<Expr*>(SubExprs[COND]); }
  const Expr *getCond() const { return reinterpret_cast<Expr*>(SubExprs[COND]);}
  Stmt *getBody() { return SubExprs[BODY]; }
  const Stmt *getBody() const { return SubExprs[BODY]; }  

  virtual SourceRange getSourceRange() const { 
    return SourceRange(DoLoc, SubExprs[BODY]->getLocEnd()); 
  }
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == DoStmtClass; 
  }
  static bool classof(const DoStmt *) { return true; }

  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
  
  virtual void EmitImpl(llvm::Serializer& S) const;
  static DoStmt* CreateImpl(llvm::Deserializer& D, ASTContext& C);
};


/// ForStmt - This represents a 'for (init;cond;inc)' stmt.  Note that any of
/// the init/cond/inc parts of the ForStmt will be null if they were not
/// specified in the source.
///
class ForStmt : public Stmt {
  enum { INIT, COND, INC, BODY, END_EXPR };
  Stmt* SubExprs[END_EXPR]; // SubExprs[INIT] is an expression or declstmt.
  SourceLocation ForLoc;
public:
  ForStmt(Stmt *Init, Expr *Cond, Expr *Inc, Stmt *Body, SourceLocation FL) 
    : Stmt(ForStmtClass) {
    SubExprs[INIT] = Init;
    SubExprs[COND] = reinterpret_cast<Stmt*>(Cond);
    SubExprs[INC] = reinterpret_cast<Stmt*>(Inc);
    SubExprs[BODY] = Body;
    ForLoc = FL;
  }
  
  Stmt *getInit() { return SubExprs[INIT]; }
  Expr *getCond() { return reinterpret_cast<Expr*>(SubExprs[COND]); }
  Expr *getInc()  { return reinterpret_cast<Expr*>(SubExprs[INC]); }
  Stmt *getBody() { return SubExprs[BODY]; }

  const Stmt *getInit() const { return SubExprs[INIT]; }
  const Expr *getCond() const { return reinterpret_cast<Expr*>(SubExprs[COND]);}
  const Expr *getInc()  const { return reinterpret_cast<Expr*>(SubExprs[INC]); }
  const Stmt *getBody() const { return SubExprs[BODY]; }

  virtual SourceRange getSourceRange() const { 
    return SourceRange(ForLoc, SubExprs[BODY]->getLocEnd()); 
  }
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == ForStmtClass; 
  }
  static bool classof(const ForStmt *) { return true; }
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
  
  virtual void EmitImpl(llvm::Serializer& S) const;
  static ForStmt* CreateImpl(llvm::Deserializer& D, ASTContext& C);
};
  
/// GotoStmt - This represents a direct goto.
///
class GotoStmt : public Stmt {
  LabelStmt *Label;
  SourceLocation GotoLoc;
  SourceLocation LabelLoc;
public:
  GotoStmt(LabelStmt *label, SourceLocation GL, SourceLocation LL) 
    : Stmt(GotoStmtClass), Label(label), GotoLoc(GL), LabelLoc(LL) {}
  
  LabelStmt *getLabel() const { return Label; }

  virtual SourceRange getSourceRange() const { 
    return SourceRange(GotoLoc, LabelLoc); 
  }
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == GotoStmtClass; 
  }
  static bool classof(const GotoStmt *) { return true; }
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
  
  virtual void EmitImpl(llvm::Serializer& S) const;
  static GotoStmt* CreateImpl(llvm::Deserializer& D, ASTContext& C);
};

/// IndirectGotoStmt - This represents an indirect goto.
///
class IndirectGotoStmt : public Stmt {
  Stmt *Target;
  // FIXME: Add location information (e.g. SourceLocation objects).
  //        When doing so, update the serialization routines.
public:
  IndirectGotoStmt(Expr *target) : Stmt(IndirectGotoStmtClass),
                                   Target((Stmt*)target){}
  
  Expr *getTarget();
  const Expr *getTarget() const;

  virtual SourceRange getSourceRange() const { return SourceRange(); }
  
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == IndirectGotoStmtClass; 
  }
  static bool classof(const IndirectGotoStmt *) { return true; }
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
  
  virtual void EmitImpl(llvm::Serializer& S) const;
  static IndirectGotoStmt* CreateImpl(llvm::Deserializer& D, ASTContext& C);
};


/// ContinueStmt - This represents a continue.
///
class ContinueStmt : public Stmt {
  SourceLocation ContinueLoc;
public:
  ContinueStmt(SourceLocation CL) : Stmt(ContinueStmtClass), ContinueLoc(CL) {}
  
  virtual SourceRange getSourceRange() const { 
    return SourceRange(ContinueLoc); 
  }
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == ContinueStmtClass; 
  }
  static bool classof(const ContinueStmt *) { return true; }
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
  
  virtual void EmitImpl(llvm::Serializer& S) const;
  static ContinueStmt* CreateImpl(llvm::Deserializer& D, ASTContext& C);
};

/// BreakStmt - This represents a break.
///
class BreakStmt : public Stmt {
  SourceLocation BreakLoc;
public:
  BreakStmt(SourceLocation BL) : Stmt(BreakStmtClass), BreakLoc(BL) {}
  
  virtual SourceRange getSourceRange() const { return SourceRange(BreakLoc); }

  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == BreakStmtClass; 
  }
  static bool classof(const BreakStmt *) { return true; }
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
  
  virtual void EmitImpl(llvm::Serializer& S) const;
  static BreakStmt* CreateImpl(llvm::Deserializer& D, ASTContext& C);
};


/// ReturnStmt - This represents a return, optionally of an expression:
///   return;
///   return 4;
///
/// Note that GCC allows return with no argument in a function declared to
/// return a value, and it allows returning a value in functions declared to
/// return void.  We explicitly model this in the AST, which means you can't
/// depend on the return type of the function and the presence of an argument.
///
class ReturnStmt : public Stmt {
  Stmt *RetExpr;
  SourceLocation RetLoc;
public:
  ReturnStmt(SourceLocation RL, Expr *E = 0) : Stmt(ReturnStmtClass), 
    RetExpr((Stmt*) E), RetLoc(RL) {}
  
  const Expr *getRetValue() const;
  Expr *getRetValue();

  virtual SourceRange getSourceRange() const;
  
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == ReturnStmtClass; 
  }
  static bool classof(const ReturnStmt *) { return true; }
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
  
  virtual void EmitImpl(llvm::Serializer& S) const;
  static ReturnStmt* CreateImpl(llvm::Deserializer& D, ASTContext& C);
};

/// AsmStmt - This represents a GNU inline-assembly statement extension.
///
class AsmStmt : public Stmt {
  SourceLocation AsmLoc, RParenLoc;
  StringLiteral *AsmStr;

  bool IsSimple;
  bool IsVolatile;
  
  unsigned NumOutputs;
  unsigned NumInputs;
  
  llvm::SmallVector<std::string, 4> Names;
  llvm::SmallVector<StringLiteral*, 4> Constraints;
  llvm::SmallVector<Expr*, 4> Exprs;

  llvm::SmallVector<StringLiteral*, 4> Clobbers;
public:
  AsmStmt(SourceLocation asmloc, bool issimple, bool isvolatile, 
          unsigned numoutputs, unsigned numinputs, 
          std::string *names, StringLiteral **constraints,
          Expr **exprs, StringLiteral *asmstr, unsigned numclobbers,
          StringLiteral **clobbers, SourceLocation rparenloc);

  bool isVolatile() const { return IsVolatile; }
  bool isSimple() const { return IsSimple; }

  unsigned getNumOutputs() const { return NumOutputs; }
  const std::string &getOutputName(unsigned i) const
    { return Names[i]; }
  const StringLiteral *getOutputConstraint(unsigned i) const
  { return Constraints[i]; }
  StringLiteral *getOutputConstraint(unsigned i)
    { return Constraints[i]; }
  const Expr *getOutputExpr(unsigned i) const { return Exprs[i]; }
  Expr *getOutputExpr(unsigned i) { return Exprs[i]; }
  
  unsigned getNumInputs() const { return NumInputs; }  
  const std::string &getInputName(unsigned i) const
    { return Names[i + NumOutputs]; }
  StringLiteral *getInputConstraint(unsigned i) 
    { return Constraints[i + NumOutputs]; }
  const StringLiteral *getInputConstraint(unsigned i) const
    { return Constraints[i + NumOutputs]; }
  Expr *getInputExpr(unsigned i) { return Exprs[i + NumOutputs]; }
  const Expr *getInputExpr(unsigned i) const { return Exprs[i + NumOutputs]; }

  const StringLiteral *getAsmString() const { return AsmStr; }
  StringLiteral *getAsmString() { return AsmStr; }

  unsigned getNumClobbers() const { return Clobbers.size(); }
  StringLiteral *getClobber(unsigned i) { return Clobbers[i]; }
  const StringLiteral *getClobber(unsigned i) const { return Clobbers[i]; }
  
  virtual SourceRange getSourceRange() const {
    return SourceRange(AsmLoc, RParenLoc);
  }
  
  static bool classof(const Stmt *T) {return T->getStmtClass() == AsmStmtClass;}
  static bool classof(const AsmStmt *) { return true; }
  
  // Input expr iterators.
  
  typedef Expr* const * inputs_iterator;
  typedef const Expr* const* const_inputs_iterator;
  
  inputs_iterator begin_inputs() { return &Exprs[0] + NumOutputs; }
  inputs_iterator end_inputs() { return begin_inputs() + NumInputs; }
  
  const_inputs_iterator begin_inputs() const { return &Exprs[0] + NumOutputs; }
  const_inputs_iterator end_inputs() const { return begin_inputs() + NumInputs;}
  
  // Output expr iterators.
  
  typedef Expr* const * outputs_iterator;
  typedef const Expr* const* const_outputs_iterator;
  
  outputs_iterator begin_outputs() { return &Exprs[0]; }
  outputs_iterator end_outputs() { return begin_outputs() + NumOutputs; }
  
  const_outputs_iterator begin_outputs() const { return &Exprs[0]; }
  const_outputs_iterator end_outputs() const {
    return begin_outputs() + NumOutputs;
  }
  
  // Child iterators  
  
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
  
  virtual void EmitImpl(llvm::Serializer& S) const;
  static AsmStmt* CreateImpl(llvm::Deserializer& D, ASTContext& C);
};

/// ObjCForCollectionStmt - This represents Objective-c's collection statement;
/// represented as 'for (element 'in' collection-expression)' stmt.
///
class ObjCForCollectionStmt : public Stmt {
  enum { ELEM, COLLECTION, BODY, END_EXPR };
  Stmt* SubExprs[END_EXPR]; // SubExprs[ELEM] is an expression or declstmt.
  SourceLocation ForLoc;
  SourceLocation RParenLoc;
public:
  ObjCForCollectionStmt(Stmt *Elem, Expr *Collect, Stmt *Body, 
                        SourceLocation FCL, SourceLocation RPL);
  
  Stmt *getElement() { return SubExprs[ELEM]; }
  Expr *getCollection() { 
    return reinterpret_cast<Expr*>(SubExprs[COLLECTION]); 
  }
  Stmt *getBody() { return SubExprs[BODY]; }
  
  const Stmt *getElement() const { return SubExprs[ELEM]; }
  const Expr *getCollection() const { 
    return reinterpret_cast<Expr*>(SubExprs[COLLECTION]);
  }
  const Stmt *getBody() const { return SubExprs[BODY]; }
  
  SourceLocation getRParenLoc() const { return RParenLoc; }
  
  virtual SourceRange getSourceRange() const { 
    return SourceRange(ForLoc, SubExprs[BODY]->getLocEnd()); 
  }
  static bool classof(const Stmt *T) { 
    return T->getStmtClass() == ObjCForCollectionStmtClass; 
  }
  static bool classof(const ObjCForCollectionStmt *) { return true; }
  
  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
  
  virtual void EmitImpl(llvm::Serializer& S) const;
  static ObjCForCollectionStmt* CreateImpl(llvm::Deserializer& D, ASTContext& C);
};  
  
/// ObjCAtCatchStmt - This represents objective-c's @catch statement.
class ObjCAtCatchStmt : public Stmt {
private:
  enum { SELECTOR, BODY, NEXT_CATCH, END_EXPR };
  Stmt *SubExprs[END_EXPR];
  SourceLocation AtCatchLoc, RParenLoc;

  // Used by deserialization.
  ObjCAtCatchStmt(SourceLocation atCatchLoc, SourceLocation rparenloc)
  : Stmt(ObjCAtCatchStmtClass), AtCatchLoc(atCatchLoc), RParenLoc(rparenloc) {}

public:
  ObjCAtCatchStmt(SourceLocation atCatchLoc, SourceLocation rparenloc,
                  Stmt *catchVarStmtDecl, Stmt *atCatchStmt, Stmt *atCatchList);
  
  const Stmt *getCatchBody() const { return SubExprs[BODY]; }
  Stmt *getCatchBody() { return SubExprs[BODY]; }

  const ObjCAtCatchStmt *getNextCatchStmt() const {
    return static_cast<const ObjCAtCatchStmt*>(SubExprs[NEXT_CATCH]);
  }
  ObjCAtCatchStmt *getNextCatchStmt() { 
    return static_cast<ObjCAtCatchStmt*>(SubExprs[NEXT_CATCH]);
  }

  const Stmt *getCatchParamStmt() const { return SubExprs[SELECTOR]; }
  Stmt *getCatchParamStmt() { return SubExprs[SELECTOR]; }
  
  SourceLocation getRParenLoc() const { return RParenLoc; }
  
  virtual SourceRange getSourceRange() const { 
    return SourceRange(AtCatchLoc, SubExprs[BODY]->getLocEnd()); 
  }

  bool hasEllipsis() const { return getCatchParamStmt() == 0; }
  
  static bool classof(const Stmt *T) {
    return T->getStmtClass() == ObjCAtCatchStmtClass;
  }
  static bool classof(const ObjCAtCatchStmt *) { return true; }
  
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
  
  virtual void EmitImpl(llvm::Serializer& S) const;
  static ObjCAtCatchStmt* CreateImpl(llvm::Deserializer& D, ASTContext& C);
};
  
/// ObjCAtFinallyStmt - This represent objective-c's @finally Statement 
class ObjCAtFinallyStmt : public Stmt {
  Stmt *AtFinallyStmt;
  SourceLocation AtFinallyLoc;    
public:
  ObjCAtFinallyStmt(SourceLocation atFinallyLoc, Stmt *atFinallyStmt)
  : Stmt(ObjCAtFinallyStmtClass), 
    AtFinallyStmt(atFinallyStmt), AtFinallyLoc(atFinallyLoc) {}
  
  const Stmt *getFinallyBody () const { return AtFinallyStmt; }
  Stmt *getFinallyBody () { return AtFinallyStmt; }

  virtual SourceRange getSourceRange() const { 
    return SourceRange(AtFinallyLoc, AtFinallyStmt->getLocEnd()); 
  }
  
  static bool classof(const Stmt *T) {
    return T->getStmtClass() == ObjCAtFinallyStmtClass;
  }
  static bool classof(const ObjCAtFinallyStmt *) { return true; }
  
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
  
  virtual void EmitImpl(llvm::Serializer& S) const;
  static ObjCAtFinallyStmt* CreateImpl(llvm::Deserializer& D, ASTContext& C);
};
  
/// ObjCAtTryStmt - This represent objective-c's over-all 
/// @try ... @catch ... @finally statement.
class ObjCAtTryStmt : public Stmt {
private:
  enum { TRY, CATCH, FINALLY, END_EXPR };
  Stmt* SubStmts[END_EXPR]; 
  
  SourceLocation AtTryLoc;      
public:
  ObjCAtTryStmt(SourceLocation atTryLoc, Stmt *atTryStmt, 
                Stmt *atCatchStmt, 
                Stmt *atFinallyStmt)
  : Stmt(ObjCAtTryStmtClass) {
      SubStmts[TRY] = atTryStmt;
      SubStmts[CATCH] = atCatchStmt;
      SubStmts[FINALLY] = atFinallyStmt;
      AtTryLoc = atTryLoc;
    }
    
  const Stmt *getTryBody() const { return SubStmts[TRY]; }
  Stmt *getTryBody() { return SubStmts[TRY]; }
  const ObjCAtCatchStmt *getCatchStmts() const { 
    return dyn_cast_or_null<ObjCAtCatchStmt>(SubStmts[CATCH]); 
  }
  ObjCAtCatchStmt *getCatchStmts() { 
    return dyn_cast_or_null<ObjCAtCatchStmt>(SubStmts[CATCH]); 
  }
  const ObjCAtFinallyStmt *getFinallyStmt() const { 
    return dyn_cast_or_null<ObjCAtFinallyStmt>(SubStmts[FINALLY]); 
  }
  ObjCAtFinallyStmt *getFinallyStmt() { 
    return dyn_cast_or_null<ObjCAtFinallyStmt>(SubStmts[FINALLY]); 
  }
  virtual SourceRange getSourceRange() const { 
    return SourceRange(AtTryLoc, SubStmts[TRY]->getLocEnd()); 
  }
    
  static bool classof(const Stmt *T) {
    return T->getStmtClass() == ObjCAtTryStmtClass;
  }
  static bool classof(const ObjCAtTryStmt *) { return true; }
    
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
  
  virtual void EmitImpl(llvm::Serializer& S) const;
  static ObjCAtTryStmt* CreateImpl(llvm::Deserializer& D, ASTContext& C);
};

/// ObjCAtSynchronizedStmt - This is for objective-c's @synchronized statement.
/// Example: @synchronized (sem) {
///             do-something;
///          }
///
class ObjCAtSynchronizedStmt : public Stmt {
private:
  enum { SYNC_EXPR, SYNC_BODY, END_EXPR };
  Stmt* SubStmts[END_EXPR];
  SourceLocation AtSynchronizedLoc;
  
public:
  ObjCAtSynchronizedStmt(SourceLocation atSynchronizedLoc, Stmt *synchExpr,
                         Stmt *synchBody)
  : Stmt(ObjCAtSynchronizedStmtClass) {
      SubStmts[SYNC_EXPR] = synchExpr;
      SubStmts[SYNC_BODY] = synchBody;
      AtSynchronizedLoc = atSynchronizedLoc;
    }
  
  const CompoundStmt *getSynchBody() const {
    return reinterpret_cast<CompoundStmt*>(SubStmts[SYNC_BODY]);
  }
  CompoundStmt *getSynchBody() { 
    return reinterpret_cast<CompoundStmt*>(SubStmts[SYNC_BODY]); 
  }
  
  const Expr *getSynchExpr() const { 
    return reinterpret_cast<Expr*>(SubStmts[SYNC_EXPR]); 
  }
  Expr *getSynchExpr() { 
    return reinterpret_cast<Expr*>(SubStmts[SYNC_EXPR]); 
  }
  
  virtual SourceRange getSourceRange() const { 
    return SourceRange(AtSynchronizedLoc, getSynchBody()->getLocEnd()); 
  }
  
  static bool classof(const Stmt *T) {
    return T->getStmtClass() == ObjCAtSynchronizedStmtClass;
  }
  static bool classof(const ObjCAtSynchronizedStmt *) { return true; }
  
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
  
  virtual void EmitImpl(llvm::Serializer& S) const;
  static ObjCAtSynchronizedStmt* CreateImpl(llvm::Deserializer& D,
                                            ASTContext& C);
};
  
/// ObjCAtThrowStmt - This represents objective-c's @throw statement.
class ObjCAtThrowStmt : public Stmt {
  Stmt *Throw;
  SourceLocation AtThrowLoc;
public:
  ObjCAtThrowStmt(SourceLocation atThrowLoc, Stmt *throwExpr)
  : Stmt(ObjCAtThrowStmtClass), Throw(throwExpr) {
    AtThrowLoc = atThrowLoc;
  }
  
  const Expr *getThrowExpr() const { return reinterpret_cast<Expr*>(Throw); }
  Expr *getThrowExpr() { return reinterpret_cast<Expr*>(Throw); }
  
  virtual SourceRange getSourceRange() const {
    if (Throw)
      return SourceRange(AtThrowLoc, Throw->getLocEnd()); 
    else 
      return SourceRange(AtThrowLoc);
  }
  
  static bool classof(const Stmt *T) {
    return T->getStmtClass() == ObjCAtThrowStmtClass;
  }
  static bool classof(const ObjCAtThrowStmt *) { return true; }
  
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
  
  virtual void EmitImpl(llvm::Serializer& S) const;
  static ObjCAtThrowStmt* CreateImpl(llvm::Deserializer& D, ASTContext& C);
};

}  // end namespace clang

#endif
