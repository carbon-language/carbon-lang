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
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/StmtIterator.h"
#include "clang/AST/DeclGroup.h"
#include "clang/AST/FullExpr.h"
#include "llvm/ADT/SmallVector.h"
#include "clang/AST/ASTContext.h"
#include <string>
using llvm::dyn_cast_or_null;

namespace llvm {
  class FoldingSetNodeID;
}

namespace clang {
  class ASTContext;
  class Expr;
  class Decl;
  class ParmVarDecl;
  class QualType;
  class IdentifierInfo;
  class SourceManager;
  class StringLiteral;
  class SwitchStmt;

  //===----------------------------------------------------------------------===//
  // ExprIterator - Iterators for iterating over Stmt* arrays that contain
  //  only Expr*.  This is needed because AST nodes use Stmt* arrays to store
  //  references to children (to be compatible with StmtIterator).
  //===----------------------------------------------------------------------===//

  class Stmt;
  class Expr;

  class ExprIterator {
    Stmt** I;
  public:
    ExprIterator(Stmt** i) : I(i) {}
    ExprIterator() : I(0) {}
    ExprIterator& operator++() { ++I; return *this; }
    ExprIterator operator-(size_t i) { return I-i; }
    ExprIterator operator+(size_t i) { return I+i; }
    Expr* operator[](size_t idx);
    // FIXME: Verify that this will correctly return a signed distance.
    signed operator-(const ExprIterator& R) const { return I - R.I; }
    Expr* operator*() const;
    Expr* operator->() const;
    bool operator==(const ExprIterator& R) const { return I == R.I; }
    bool operator!=(const ExprIterator& R) const { return I != R.I; }
    bool operator>(const ExprIterator& R) const { return I > R.I; }
    bool operator>=(const ExprIterator& R) const { return I >= R.I; }
  };

  class ConstExprIterator {
    Stmt* const * I;
  public:
    ConstExprIterator(Stmt* const* i) : I(i) {}
    ConstExprIterator() : I(0) {}
    ConstExprIterator& operator++() { ++I; return *this; }
    ConstExprIterator operator+(size_t i) { return I+i; }
    ConstExprIterator operator-(size_t i) { return I-i; }
    const Expr * operator[](size_t idx) const;
    signed operator-(const ConstExprIterator& R) const { return I - R.I; }
    const Expr * operator*() const;
    const Expr * operator->() const;
    bool operator==(const ConstExprIterator& R) const { return I == R.I; }
    bool operator!=(const ConstExprIterator& R) const { return I != R.I; }
    bool operator>(const ConstExprIterator& R) const { return I > R.I; }
    bool operator>=(const ConstExprIterator& R) const { return I >= R.I; }
  };

//===----------------------------------------------------------------------===//
// AST classes for statements.
//===----------------------------------------------------------------------===//

/// Stmt - This represents one statement.
///
class Stmt {
public:
  enum StmtClass {
    NoStmtClass = 0,
#define STMT(CLASS, PARENT) CLASS##Class,
#define FIRST_STMT(CLASS) firstStmtConstant = CLASS##Class,
#define LAST_STMT(CLASS) lastStmtConstant = CLASS##Class,
#define FIRST_EXPR(CLASS) firstExprConstant = CLASS##Class,
#define LAST_EXPR(CLASS) lastExprConstant = CLASS##Class
#include "clang/AST/StmtNodes.def"
};
private:
  /// \brief The statement class.
  const unsigned sClass : 8;

  /// \brief The reference count for this statement.
  unsigned RefCount : 24;

  // Make vanilla 'new' and 'delete' illegal for Stmts.
protected:
  void* operator new(size_t bytes) throw() {
    assert(0 && "Stmts cannot be allocated with regular 'new'.");
    return 0;
  }
  void operator delete(void* data) throw() {
    assert(0 && "Stmts cannot be released with regular 'delete'.");
  }

public:
  // Only allow allocation of Stmts using the allocator in ASTContext
  // or by doing a placement new.
  void* operator new(size_t bytes, ASTContext& C,
                     unsigned alignment = 16) throw() {
    return ::operator new(bytes, C, alignment);
  }

  void* operator new(size_t bytes, ASTContext* C,
                     unsigned alignment = 16) throw() {
    return ::operator new(bytes, *C, alignment);
  }

  void* operator new(size_t bytes, void* mem) throw() {
    return mem;
  }

  void operator delete(void*, ASTContext&, unsigned) throw() { }
  void operator delete(void*, ASTContext*, unsigned) throw() { }
  void operator delete(void*, std::size_t) throw() { }
  void operator delete(void*, void*) throw() { }

public:
  /// \brief A placeholder type used to construct an empty shell of a
  /// type, that will be filled in later (e.g., by some
  /// de-serialization).
  struct EmptyShell { };

protected:
  /// DestroyChildren - Invoked by destructors of subclasses of Stmt to
  ///  recursively release child AST nodes.
  void DestroyChildren(ASTContext& Ctx);

  /// \brief Construct an empty statement.
  explicit Stmt(StmtClass SC, EmptyShell) : sClass(SC), RefCount(1) {
    if (Stmt::CollectingStats()) Stmt::addStmtClass(SC);
  }

  /// \brief Virtual method that performs the actual destruction of
  /// this statement.
  ///
  /// Subclasses should override this method (not Destroy()) to
  /// provide class-specific destruction.
  virtual void DoDestroy(ASTContext &Ctx);

public:
  Stmt(StmtClass SC) : sClass(SC), RefCount(1) {
    if (Stmt::CollectingStats()) Stmt::addStmtClass(SC);
  }
  virtual ~Stmt() {}

#ifndef NDEBUG
  /// \brief True if this statement's refcount is in a valid state.
  /// Should be used only in assertions.
  bool isRetained() const {
    return (RefCount >= 1);
  }
#endif

  /// \brief Destroy the current statement and its children.
  void Destroy(ASTContext &Ctx) {
    assert(RefCount >= 1);
    if (--RefCount == 0)
      DoDestroy(Ctx);
  }

  /// \brief Increases the reference count for this statement.
  ///
  /// Invoke the Retain() operation when this statement or expression
  /// is being shared by another owner.
  Stmt *Retain() {
    assert(RefCount >= 1);
    ++RefCount;
    return this;
  }

  StmtClass getStmtClass() const { 
    assert(RefCount >= 1 && "Referencing already-destroyed statement!");
    return (StmtClass)sClass; 
  }
  const char *getStmtClassName() const;

  /// SourceLocation tokens are not useful in isolation - they are low level
  /// value objects created/interpreted by SourceManager. We assume AST
  /// clients will have a pointer to the respective SourceManager.
  virtual SourceRange getSourceRange() const = 0;
  SourceLocation getLocStart() const { return getSourceRange().getBegin(); }
  SourceLocation getLocEnd() const { return getSourceRange().getEnd(); }

  // global temp stats (until we have a per-module visitor)
  static void addStmtClass(const StmtClass s);
  static bool CollectingStats(bool Enable = false);
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
  void dumpPretty(ASTContext& Context) const;
  void printPretty(llvm::raw_ostream &OS, PrinterHelper *Helper,
                   const PrintingPolicy &Policy,
                   unsigned Indentation = 0) const {
    printPretty(OS, *(ASTContext*)0, Helper, Policy, Indentation);
  }
  void printPretty(llvm::raw_ostream &OS, ASTContext &Context,
                   PrinterHelper *Helper,
                   const PrintingPolicy &Policy,
                   unsigned Indentation = 0) const;

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

  /// \brief Produce a unique representation of the given statement.
  ///
  /// \brief ID once the profiling operation is complete, will contain
  /// the unique representation of the given statement.
  ///
  /// \brief Context the AST context in which the statement resides
  ///
  /// \brief Canonical whether the profile should be based on the canonical
  /// representation of this statement (e.g., where non-type template
  /// parameters are identified by index/level rather than their
  /// declaration pointers) or the exact representation of the statement as
  /// written in the source.
  void Profile(llvm::FoldingSetNodeID &ID, ASTContext &Context,
               bool Canonical);
};

/// DeclStmt - Adaptor class for mixing declarations with statements and
/// expressions. For example, CompoundStmt mixes statements, expressions
/// and declarations (variables, types). Another example is ForStmt, where
/// the first statement can be an expression or a declaration.
///
class DeclStmt : public Stmt {
  DeclGroupRef DG;
  SourceLocation StartLoc, EndLoc;

protected:
  virtual void DoDestroy(ASTContext &Ctx);

public:
  DeclStmt(DeclGroupRef dg, SourceLocation startLoc,
           SourceLocation endLoc) : Stmt(DeclStmtClass), DG(dg),
                                    StartLoc(startLoc), EndLoc(endLoc) {}

  /// \brief Build an empty declaration statement.
  explicit DeclStmt(EmptyShell Empty) : Stmt(DeclStmtClass, Empty) { }

  /// isSingleDecl - This method returns true if this DeclStmt refers
  /// to a single Decl.
  bool isSingleDecl() const {
    return DG.isSingleDecl();
  }

  const Decl *getSingleDecl() const { return DG.getSingleDecl(); }
  Decl *getSingleDecl() { return DG.getSingleDecl(); }

  const DeclGroupRef getDeclGroup() const { return DG; }
  DeclGroupRef getDeclGroup() { return DG; }
  void setDeclGroup(DeclGroupRef DGR) { DG = DGR; }

  SourceLocation getStartLoc() const { return StartLoc; }
  void setStartLoc(SourceLocation L) { StartLoc = L; }
  SourceLocation getEndLoc() const { return EndLoc; }
  void setEndLoc(SourceLocation L) { EndLoc = L; }

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

  typedef DeclGroupRef::iterator decl_iterator;
  typedef DeclGroupRef::const_iterator const_decl_iterator;

  decl_iterator decl_begin() { return DG.begin(); }
  decl_iterator decl_end() { return DG.end(); }
  const_decl_iterator decl_begin() const { return DG.begin(); }
  const_decl_iterator decl_end() const { return DG.end(); }
};

/// NullStmt - This is the null statement ";": C99 6.8.3p3.
///
class NullStmt : public Stmt {
  SourceLocation SemiLoc;
public:
  NullStmt(SourceLocation L) : Stmt(NullStmtClass), SemiLoc(L) {}

  /// \brief Build an empty null statement.
  explicit NullStmt(EmptyShell Empty) : Stmt(NullStmtClass, Empty) { }

  SourceLocation getSemiLoc() const { return SemiLoc; }
  void setSemiLoc(SourceLocation L) { SemiLoc = L; }

  virtual SourceRange getSourceRange() const { return SourceRange(SemiLoc); }

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
  Stmt** Body;
  unsigned NumStmts;
  SourceLocation LBracLoc, RBracLoc;
public:
  CompoundStmt(ASTContext& C, Stmt **StmtStart, unsigned numStmts,
                             SourceLocation LB, SourceLocation RB)
  : Stmt(CompoundStmtClass), NumStmts(numStmts), LBracLoc(LB), RBracLoc(RB) {
    if (NumStmts == 0) {
      Body = 0;
      return;
    }

    Body = new (C) Stmt*[NumStmts];
    memcpy(Body, StmtStart, numStmts * sizeof(*Body));
  }

  // \brief Build an empty compound statement.
  explicit CompoundStmt(EmptyShell Empty)
    : Stmt(CompoundStmtClass, Empty), Body(0), NumStmts(0) { }

  void setStmts(ASTContext &C, Stmt **Stmts, unsigned NumStmts);

  bool body_empty() const { return NumStmts == 0; }
  unsigned size() const { return NumStmts; }

  typedef Stmt** body_iterator;
  body_iterator body_begin() { return Body; }
  body_iterator body_end() { return Body + NumStmts; }
  Stmt *body_back() { return NumStmts ? Body[NumStmts-1] : 0; }

  typedef Stmt* const * const_body_iterator;
  const_body_iterator body_begin() const { return Body; }
  const_body_iterator body_end() const { return Body + NumStmts; }
  const Stmt *body_back() const { return NumStmts ? Body[NumStmts-1] : 0; }

  typedef std::reverse_iterator<body_iterator> reverse_body_iterator;
  reverse_body_iterator body_rbegin() {
    return reverse_body_iterator(body_end());
  }
  reverse_body_iterator body_rend() {
    return reverse_body_iterator(body_begin());
  }

  typedef std::reverse_iterator<const_body_iterator>
          const_reverse_body_iterator;

  const_reverse_body_iterator body_rbegin() const {
    return const_reverse_body_iterator(body_end());
  }

  const_reverse_body_iterator body_rend() const {
    return const_reverse_body_iterator(body_begin());
  }

  virtual SourceRange getSourceRange() const {
    return SourceRange(LBracLoc, RBracLoc);
  }

  SourceLocation getLBracLoc() const { return LBracLoc; }
  void setLBracLoc(SourceLocation L) { LBracLoc = L; }
  SourceLocation getRBracLoc() const { return RBracLoc; }
  void setRBracLoc(SourceLocation L) { RBracLoc = L; }

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
protected:
  // A pointer to the following CaseStmt or DefaultStmt class,
  // used by SwitchStmt.
  SwitchCase *NextSwitchCase;

  SwitchCase(StmtClass SC) : Stmt(SC), NextSwitchCase(0) {}

public:
  const SwitchCase *getNextSwitchCase() const { return NextSwitchCase; }

  SwitchCase *getNextSwitchCase() { return NextSwitchCase; }

  void setNextSwitchCase(SwitchCase *SC) { NextSwitchCase = SC; }

  Stmt *getSubStmt() { return v_getSubStmt(); }

  virtual SourceRange getSourceRange() const { return SourceRange(); }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == CaseStmtClass ||
    T->getStmtClass() == DefaultStmtClass;
  }
  static bool classof(const SwitchCase *) { return true; }
protected:
  virtual Stmt* v_getSubStmt() = 0;
};

class CaseStmt : public SwitchCase {
  enum { SUBSTMT, LHS, RHS, END_EXPR };
  Stmt* SubExprs[END_EXPR];  // The expression for the RHS is Non-null for
                             // GNU "case 1 ... 4" extension
  SourceLocation CaseLoc;
  SourceLocation EllipsisLoc;
  SourceLocation ColonLoc;

  virtual Stmt* v_getSubStmt() { return getSubStmt(); }
public:
  CaseStmt(Expr *lhs, Expr *rhs, SourceLocation caseLoc,
           SourceLocation ellipsisLoc, SourceLocation colonLoc)
    : SwitchCase(CaseStmtClass) {
    SubExprs[SUBSTMT] = 0;
    SubExprs[LHS] = reinterpret_cast<Stmt*>(lhs);
    SubExprs[RHS] = reinterpret_cast<Stmt*>(rhs);
    CaseLoc = caseLoc;
    EllipsisLoc = ellipsisLoc;
    ColonLoc = colonLoc;
  }

  /// \brief Build an empty switch case statement.
  explicit CaseStmt(EmptyShell Empty) : SwitchCase(CaseStmtClass) { }

  SourceLocation getCaseLoc() const { return CaseLoc; }
  void setCaseLoc(SourceLocation L) { CaseLoc = L; }
  SourceLocation getEllipsisLoc() const { return EllipsisLoc; }
  void setEllipsisLoc(SourceLocation L) { EllipsisLoc = L; }
  SourceLocation getColonLoc() const { return ColonLoc; }
  void setColonLoc(SourceLocation L) { ColonLoc = L; }

  Expr *getLHS() { return reinterpret_cast<Expr*>(SubExprs[LHS]); }
  Expr *getRHS() { return reinterpret_cast<Expr*>(SubExprs[RHS]); }
  Stmt *getSubStmt() { return SubExprs[SUBSTMT]; }

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
    // Handle deeply nested case statements with iteration instead of recursion.
    const CaseStmt *CS = this;
    while (const CaseStmt *CS2 = dyn_cast<CaseStmt>(CS->getSubStmt()))
      CS = CS2;

    return SourceRange(CaseLoc, CS->getSubStmt()->getLocEnd());
  }
  static bool classof(const Stmt *T) {
    return T->getStmtClass() == CaseStmtClass;
  }
  static bool classof(const CaseStmt *) { return true; }

  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};

class DefaultStmt : public SwitchCase {
  Stmt* SubStmt;
  SourceLocation DefaultLoc;
  SourceLocation ColonLoc;
  virtual Stmt* v_getSubStmt() { return getSubStmt(); }
public:
  DefaultStmt(SourceLocation DL, SourceLocation CL, Stmt *substmt) :
    SwitchCase(DefaultStmtClass), SubStmt(substmt), DefaultLoc(DL),
    ColonLoc(CL) {}

  /// \brief Build an empty default statement.
  explicit DefaultStmt(EmptyShell) : SwitchCase(DefaultStmtClass) { }

  Stmt *getSubStmt() { return SubStmt; }
  const Stmt *getSubStmt() const { return SubStmt; }
  void setSubStmt(Stmt *S) { SubStmt = S; }

  SourceLocation getDefaultLoc() const { return DefaultLoc; }
  void setDefaultLoc(SourceLocation L) { DefaultLoc = L; }
  SourceLocation getColonLoc() const { return ColonLoc; }
  void setColonLoc(SourceLocation L) { ColonLoc = L; }

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
};

class LabelStmt : public Stmt {
  IdentifierInfo *Label;
  Stmt *SubStmt;
  SourceLocation IdentLoc;
public:
  LabelStmt(SourceLocation IL, IdentifierInfo *label, Stmt *substmt)
    : Stmt(LabelStmtClass), Label(label),
      SubStmt(substmt), IdentLoc(IL) {}

  // \brief Build an empty label statement.
  explicit LabelStmt(EmptyShell Empty) : Stmt(LabelStmtClass, Empty) { }

  SourceLocation getIdentLoc() const { return IdentLoc; }
  IdentifierInfo *getID() const { return Label; }
  void setID(IdentifierInfo *II) { Label = II; }
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
};


/// IfStmt - This represents an if/then/else.
///
class IfStmt : public Stmt {
  enum { COND, THEN, ELSE, END_EXPR };
  Stmt* SubExprs[END_EXPR];

  /// \brief If non-NULL, the declaration in the "if" statement.
  VarDecl *Var;
  
  SourceLocation IfLoc;
  SourceLocation ElseLoc;
  
public:
  IfStmt(SourceLocation IL, VarDecl *var, Expr *cond, Stmt *then,
         SourceLocation EL = SourceLocation(), Stmt *elsev = 0)
    : Stmt(IfStmtClass), Var(var), IfLoc(IL), ElseLoc(EL)  {
    SubExprs[COND] = reinterpret_cast<Stmt*>(cond);
    SubExprs[THEN] = then;
    SubExprs[ELSE] = elsev;
  }

  /// \brief Build an empty if/then/else statement
  explicit IfStmt(EmptyShell Empty) : Stmt(IfStmtClass, Empty) { }

  /// \brief Retrieve the variable declared in this "if" statement, if any.
  ///
  /// In the following example, "x" is the condition variable.
  /// \code
  /// if (int x = foo()) {
  ///   printf("x is %d", x);
  /// }
  /// \endcode
  VarDecl *getConditionVariable() const { return Var; }
  void setConditionVariable(VarDecl *V) { Var = V; }
  
  const Expr *getCond() const { return reinterpret_cast<Expr*>(SubExprs[COND]);}
  void setCond(Expr *E) { SubExprs[COND] = reinterpret_cast<Stmt *>(E); }
  const Stmt *getThen() const { return SubExprs[THEN]; }
  void setThen(Stmt *S) { SubExprs[THEN] = S; }
  const Stmt *getElse() const { return SubExprs[ELSE]; }
  void setElse(Stmt *S) { SubExprs[ELSE] = S; }

  Expr *getCond() { return reinterpret_cast<Expr*>(SubExprs[COND]); }
  Stmt *getThen() { return SubExprs[THEN]; }
  Stmt *getElse() { return SubExprs[ELSE]; }

  SourceLocation getIfLoc() const { return IfLoc; }
  void setIfLoc(SourceLocation L) { IfLoc = L; }
  SourceLocation getElseLoc() const { return ElseLoc; }
  void setElseLoc(SourceLocation L) { ElseLoc = L; }

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

  // Iterators over subexpressions.  The iterators will include iterating
  // over the initialization expression referenced by the condition variable.
  virtual child_iterator child_begin();
  virtual child_iterator child_end();

protected:
  virtual void DoDestroy(ASTContext &Ctx);
};

/// SwitchStmt - This represents a 'switch' stmt.
///
class SwitchStmt : public Stmt {
  enum { COND, BODY, END_EXPR };
  Stmt* SubExprs[END_EXPR];
  VarDecl *Var;
  // This points to a linked list of case and default statements.
  SwitchCase *FirstCase;
  SourceLocation SwitchLoc;

protected:
  virtual void DoDestroy(ASTContext &Ctx);

public:
  SwitchStmt(VarDecl *Var, Expr *cond) 
    : Stmt(SwitchStmtClass), Var(Var), FirstCase(0) 
  {
    SubExprs[COND] = reinterpret_cast<Stmt*>(cond);
    SubExprs[BODY] = NULL;
  }

  /// \brief Build a empty switch statement.
  explicit SwitchStmt(EmptyShell Empty) : Stmt(SwitchStmtClass, Empty) { }

  /// \brief Retrieve the variable declared in this "switch" statement, if any.
  ///
  /// In the following example, "x" is the condition variable.
  /// \code
  /// switch (int x = foo()) {
  ///   case 0: break;
  ///   // ...
  /// }
  /// \endcode
  VarDecl *getConditionVariable() const { return Var; }
  void setConditionVariable(VarDecl *V) { Var = V; }

  const Expr *getCond() const { return reinterpret_cast<Expr*>(SubExprs[COND]);}
  const Stmt *getBody() const { return SubExprs[BODY]; }
  const SwitchCase *getSwitchCaseList() const { return FirstCase; }

  Expr *getCond() { return reinterpret_cast<Expr*>(SubExprs[COND]);}
  void setCond(Expr *E) { SubExprs[COND] = reinterpret_cast<Stmt *>(E); }
  Stmt *getBody() { return SubExprs[BODY]; }
  void setBody(Stmt *S) { SubExprs[BODY] = S; }
  SwitchCase *getSwitchCaseList() { return FirstCase; }

  /// \brief Set the case list for this switch statement.
  ///
  /// The caller is responsible for incrementing the retain counts on
  /// all of the SwitchCase statements in this list.
  void setSwitchCaseList(SwitchCase *SC) { FirstCase = SC; }

  SourceLocation getSwitchLoc() const { return SwitchLoc; }
  void setSwitchLoc(SourceLocation L) { SwitchLoc = L; }

  void setBody(Stmt *S, SourceLocation SL) {
    SubExprs[BODY] = S;
    SwitchLoc = SL;
  }
  void addSwitchCase(SwitchCase *SC) {
    assert(!SC->getNextSwitchCase() && "case/default already added to a switch");
    SC->Retain();
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
};


/// WhileStmt - This represents a 'while' stmt.
///
class WhileStmt : public Stmt {
  enum { COND, BODY, END_EXPR };
  VarDecl *Var;
  Stmt* SubExprs[END_EXPR];
  SourceLocation WhileLoc;
public:
  WhileStmt(VarDecl *Var, Expr *cond, Stmt *body, SourceLocation WL)
    : Stmt(WhileStmtClass), Var(Var) 
  {
    SubExprs[COND] = reinterpret_cast<Stmt*>(cond);
    SubExprs[BODY] = body;
    WhileLoc = WL;
  }

  /// \brief Build an empty while statement.
  explicit WhileStmt(EmptyShell Empty) : Stmt(WhileStmtClass, Empty) { }

  /// \brief Retrieve the variable declared in this "while" statement, if any.
  ///
  /// In the following example, "x" is the condition variable.
  /// \code
  /// while (int x = random()) {
  ///   // ...
  /// }
  /// \endcode
  VarDecl *getConditionVariable() const { return Var; }
  void setConditionVariable(VarDecl *V) { Var = V; }

  Expr *getCond() { return reinterpret_cast<Expr*>(SubExprs[COND]); }
  const Expr *getCond() const { return reinterpret_cast<Expr*>(SubExprs[COND]);}
  void setCond(Expr *E) { SubExprs[COND] = reinterpret_cast<Stmt*>(E); }
  Stmt *getBody() { return SubExprs[BODY]; }
  const Stmt *getBody() const { return SubExprs[BODY]; }
  void setBody(Stmt *S) { SubExprs[BODY] = S; }

  SourceLocation getWhileLoc() const { return WhileLoc; }
  void setWhileLoc(SourceLocation L) { WhileLoc = L; }

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
  
protected:
  virtual void DoDestroy(ASTContext &Ctx);
};

/// DoStmt - This represents a 'do/while' stmt.
///
class DoStmt : public Stmt {
  enum { COND, BODY, END_EXPR };
  Stmt* SubExprs[END_EXPR];
  SourceLocation DoLoc;
  SourceLocation WhileLoc;
  SourceLocation RParenLoc;  // Location of final ')' in do stmt condition.

public:
  DoStmt(Stmt *body, Expr *cond, SourceLocation DL, SourceLocation WL,
         SourceLocation RP)
    : Stmt(DoStmtClass), DoLoc(DL), WhileLoc(WL), RParenLoc(RP) {
    SubExprs[COND] = reinterpret_cast<Stmt*>(cond);
    SubExprs[BODY] = body;
  }

  /// \brief Build an empty do-while statement.
  explicit DoStmt(EmptyShell Empty) : Stmt(DoStmtClass, Empty) { }

  Expr *getCond() { return reinterpret_cast<Expr*>(SubExprs[COND]); }
  const Expr *getCond() const { return reinterpret_cast<Expr*>(SubExprs[COND]);}
  void setCond(Expr *E) { SubExprs[COND] = reinterpret_cast<Stmt*>(E); }
  Stmt *getBody() { return SubExprs[BODY]; }
  const Stmt *getBody() const { return SubExprs[BODY]; }
  void setBody(Stmt *S) { SubExprs[BODY] = S; }

  SourceLocation getDoLoc() const { return DoLoc; }
  void setDoLoc(SourceLocation L) { DoLoc = L; }
  SourceLocation getWhileLoc() const { return WhileLoc; }
  void setWhileLoc(SourceLocation L) { WhileLoc = L; }

  SourceLocation getRParenLoc() const { return RParenLoc; }
  void setRParenLoc(SourceLocation L) { RParenLoc = L; }

  virtual SourceRange getSourceRange() const {
    return SourceRange(DoLoc, RParenLoc);
  }
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
  VarDecl *CondVar;
  SourceLocation ForLoc;
  SourceLocation LParenLoc, RParenLoc;

public:
  ForStmt(Stmt *Init, Expr *Cond, VarDecl *condVar, Expr *Inc, Stmt *Body, 
          SourceLocation FL, SourceLocation LP, SourceLocation RP)
    : Stmt(ForStmtClass), CondVar(condVar), ForLoc(FL), LParenLoc(LP), 
      RParenLoc(RP) 
  {
    SubExprs[INIT] = Init;
    SubExprs[COND] = reinterpret_cast<Stmt*>(Cond);
    SubExprs[INC] = reinterpret_cast<Stmt*>(Inc);
    SubExprs[BODY] = Body;
  }

  /// \brief Build an empty for statement.
  explicit ForStmt(EmptyShell Empty) : Stmt(ForStmtClass, Empty) { }

  Stmt *getInit() { return SubExprs[INIT]; }
  
  /// \brief Retrieve the variable declared in this "for" statement, if any.
  ///
  /// In the following example, "y" is the condition variable.
  /// \code
  /// for (int x = random(); int y = mangle(x); ++x) {
  ///   // ...
  /// }
  /// \endcode
  VarDecl *getConditionVariable() const { return CondVar; }
  void setConditionVariable(VarDecl *V) { CondVar = V; }
  
  Expr *getCond() { return reinterpret_cast<Expr*>(SubExprs[COND]); }
  Expr *getInc()  { return reinterpret_cast<Expr*>(SubExprs[INC]); }
  Stmt *getBody() { return SubExprs[BODY]; }

  const Stmt *getInit() const { return SubExprs[INIT]; }
  const Expr *getCond() const { return reinterpret_cast<Expr*>(SubExprs[COND]);}
  const Expr *getInc()  const { return reinterpret_cast<Expr*>(SubExprs[INC]); }
  const Stmt *getBody() const { return SubExprs[BODY]; }

  void setInit(Stmt *S) { SubExprs[INIT] = S; }
  void setCond(Expr *E) { SubExprs[COND] = reinterpret_cast<Stmt*>(E); }
  void setInc(Expr *E) { SubExprs[INC] = reinterpret_cast<Stmt*>(E); }
  void setBody(Stmt *S) { SubExprs[BODY] = S; }

  SourceLocation getForLoc() const { return ForLoc; }
  void setForLoc(SourceLocation L) { ForLoc = L; }
  SourceLocation getLParenLoc() const { return LParenLoc; }
  void setLParenLoc(SourceLocation L) { LParenLoc = L; }
  SourceLocation getRParenLoc() const { return RParenLoc; }
  void setRParenLoc(SourceLocation L) { RParenLoc = L; }

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
  
protected:
  virtual void DoDestroy(ASTContext &Ctx);
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

  /// \brief Build an empty goto statement.
  explicit GotoStmt(EmptyShell Empty) : Stmt(GotoStmtClass, Empty) { }

  LabelStmt *getLabel() const { return Label; }
  void setLabel(LabelStmt *S) { Label = S; }

  SourceLocation getGotoLoc() const { return GotoLoc; }
  void setGotoLoc(SourceLocation L) { GotoLoc = L; }
  SourceLocation getLabelLoc() const { return LabelLoc; }
  void setLabelLoc(SourceLocation L) { LabelLoc = L; }

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
};

/// IndirectGotoStmt - This represents an indirect goto.
///
class IndirectGotoStmt : public Stmt {
  SourceLocation GotoLoc;
  SourceLocation StarLoc;
  Stmt *Target;
public:
  IndirectGotoStmt(SourceLocation gotoLoc, SourceLocation starLoc,
                   Expr *target)
    : Stmt(IndirectGotoStmtClass), GotoLoc(gotoLoc), StarLoc(starLoc),
      Target((Stmt*)target) {}

  /// \brief Build an empty indirect goto statement.
  explicit IndirectGotoStmt(EmptyShell Empty)
    : Stmt(IndirectGotoStmtClass, Empty) { }

  void setGotoLoc(SourceLocation L) { GotoLoc = L; }
  SourceLocation getGotoLoc() const { return GotoLoc; }
  void setStarLoc(SourceLocation L) { StarLoc = L; }
  SourceLocation getStarLoc() const { return StarLoc; }

  Expr *getTarget();
  const Expr *getTarget() const;
  void setTarget(Expr *E) { Target = reinterpret_cast<Stmt*>(E); }

  virtual SourceRange getSourceRange() const {
    return SourceRange(GotoLoc, Target->getLocEnd());
  }

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
  SourceLocation ContinueLoc;
public:
  ContinueStmt(SourceLocation CL) : Stmt(ContinueStmtClass), ContinueLoc(CL) {}

  /// \brief Build an empty continue statement.
  explicit ContinueStmt(EmptyShell Empty) : Stmt(ContinueStmtClass, Empty) { }

  SourceLocation getContinueLoc() const { return ContinueLoc; }
  void setContinueLoc(SourceLocation L) { ContinueLoc = L; }

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
};

/// BreakStmt - This represents a break.
///
class BreakStmt : public Stmt {
  SourceLocation BreakLoc;
public:
  BreakStmt(SourceLocation BL) : Stmt(BreakStmtClass), BreakLoc(BL) {}

  /// \brief Build an empty break statement.
  explicit BreakStmt(EmptyShell Empty) : Stmt(BreakStmtClass, Empty) { }

  SourceLocation getBreakLoc() const { return BreakLoc; }
  void setBreakLoc(SourceLocation L) { BreakLoc = L; }

  virtual SourceRange getSourceRange() const { return SourceRange(BreakLoc); }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == BreakStmtClass;
  }
  static bool classof(const BreakStmt *) { return true; }

  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
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

  /// \brief Build an empty return expression.
  explicit ReturnStmt(EmptyShell Empty) : Stmt(ReturnStmtClass, Empty) { }

  const Expr *getRetValue() const;
  Expr *getRetValue();
  void setRetValue(Expr *E) { RetExpr = reinterpret_cast<Stmt*>(E); }

  SourceLocation getReturnLoc() const { return RetLoc; }
  void setReturnLoc(SourceLocation L) { RetLoc = L; }

  virtual SourceRange getSourceRange() const;

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == ReturnStmtClass;
  }
  static bool classof(const ReturnStmt *) { return true; }

  // Iterators
  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};

/// AsmStmt - This represents a GNU inline-assembly statement extension.
///
class AsmStmt : public Stmt {
  SourceLocation AsmLoc, RParenLoc;
  StringLiteral *AsmStr;

  bool IsSimple;
  bool IsVolatile;
  bool MSAsm;

  unsigned NumOutputs;
  unsigned NumInputs;

  llvm::SmallVector<std::string, 4> Names;
  llvm::SmallVector<StringLiteral*, 4> Constraints;
  llvm::SmallVector<Stmt*, 4> Exprs;

  llvm::SmallVector<StringLiteral*, 4> Clobbers;
public:
  AsmStmt(SourceLocation asmloc, bool issimple, bool isvolatile, bool msasm,
          unsigned numoutputs, unsigned numinputs,
          std::string *names, StringLiteral **constraints,
          Expr **exprs, StringLiteral *asmstr, unsigned numclobbers,
          StringLiteral **clobbers, SourceLocation rparenloc);

  /// \brief Build an empty inline-assembly statement.
  explicit AsmStmt(EmptyShell Empty) : Stmt(AsmStmtClass, Empty) { }

  SourceLocation getAsmLoc() const { return AsmLoc; }
  void setAsmLoc(SourceLocation L) { AsmLoc = L; }
  SourceLocation getRParenLoc() const { return RParenLoc; }
  void setRParenLoc(SourceLocation L) { RParenLoc = L; }

  bool isVolatile() const { return IsVolatile; }
  void setVolatile(bool V) { IsVolatile = V; }
  bool isSimple() const { return IsSimple; }
  void setSimple(bool V) { IsSimple = false; }
  bool isMSAsm() const { return MSAsm; }
  void setMSAsm(bool V) { MSAsm = V; }

  //===--- Asm String Analysis ---===//

  const StringLiteral *getAsmString() const { return AsmStr; }
  StringLiteral *getAsmString() { return AsmStr; }
  void setAsmString(StringLiteral *E) { AsmStr = E; }

  /// AsmStringPiece - this is part of a decomposed asm string specification
  /// (for use with the AnalyzeAsmString function below).  An asm string is
  /// considered to be a concatenation of these parts.
  class AsmStringPiece {
  public:
    enum Kind {
      String,  // String in .ll asm string form, "$" -> "$$" and "%%" -> "%".
      Operand  // Operand reference, with optional modifier %c4.
    };
  private:
    Kind MyKind;
    std::string Str;
    unsigned OperandNo;
  public:
    AsmStringPiece(const std::string &S) : MyKind(String), Str(S) {}
    AsmStringPiece(unsigned OpNo, char Modifier)
      : MyKind(Operand), Str(), OperandNo(OpNo) {
      Str += Modifier;
    }

    bool isString() const { return MyKind == String; }
    bool isOperand() const { return MyKind == Operand; }

    const std::string &getString() const {
      assert(isString());
      return Str;
    }

    unsigned getOperandNo() const {
      assert(isOperand());
      return OperandNo;
    }

    /// getModifier - Get the modifier for this operand, if present.  This
    /// returns '\0' if there was no modifier.
    char getModifier() const {
      assert(isOperand());
      return Str[0];
    }
  };

  /// AnalyzeAsmString - Analyze the asm string of the current asm, decomposing
  /// it into pieces.  If the asm string is erroneous, emit errors and return
  /// true, otherwise return false.  This handles canonicalization and
  /// translation of strings from GCC syntax to LLVM IR syntax, and handles
  //// flattening of named references like %[foo] to Operand AsmStringPiece's.
  unsigned AnalyzeAsmString(llvm::SmallVectorImpl<AsmStringPiece> &Pieces,
                            ASTContext &C, unsigned &DiagOffs) const;


  //===--- Output operands ---===//

  unsigned getNumOutputs() const { return NumOutputs; }

  const std::string &getOutputName(unsigned i) const {
    return Names[i];
  }

  /// getOutputConstraint - Return the constraint string for the specified
  /// output operand.  All output constraints are known to be non-empty (either
  /// '=' or '+').
  std::string getOutputConstraint(unsigned i) const;

  const StringLiteral *getOutputConstraintLiteral(unsigned i) const {
    return Constraints[i];
  }
  StringLiteral *getOutputConstraintLiteral(unsigned i) {
    return Constraints[i];
  }


  Expr *getOutputExpr(unsigned i);

  const Expr *getOutputExpr(unsigned i) const {
    return const_cast<AsmStmt*>(this)->getOutputExpr(i);
  }

  /// isOutputPlusConstraint - Return true if the specified output constraint
  /// is a "+" constraint (which is both an input and an output) or false if it
  /// is an "=" constraint (just an output).
  bool isOutputPlusConstraint(unsigned i) const {
    return getOutputConstraint(i)[0] == '+';
  }

  /// getNumPlusOperands - Return the number of output operands that have a "+"
  /// constraint.
  unsigned getNumPlusOperands() const;

  //===--- Input operands ---===//

  unsigned getNumInputs() const { return NumInputs; }

  const std::string &getInputName(unsigned i) const {
    return Names[i + NumOutputs];
  }

  /// getInputConstraint - Return the specified input constraint.  Unlike output
  /// constraints, these can be empty.
  std::string getInputConstraint(unsigned i) const;

  const StringLiteral *getInputConstraintLiteral(unsigned i) const {
    return Constraints[i + NumOutputs];
  }
  StringLiteral *getInputConstraintLiteral(unsigned i) {
    return Constraints[i + NumOutputs];
  }


  Expr *getInputExpr(unsigned i);

  const Expr *getInputExpr(unsigned i) const {
    return const_cast<AsmStmt*>(this)->getInputExpr(i);
  }

  void setOutputsAndInputs(unsigned NumOutputs,
                           unsigned NumInputs,
                           const std::string *Names,
                           StringLiteral **Constraints,
                           Stmt **Exprs);

  //===--- Other ---===//

  /// getNamedOperand - Given a symbolic operand reference like %[foo],
  /// translate this into a numeric value needed to reference the same operand.
  /// This returns -1 if the operand name is invalid.
  int getNamedOperand(const std::string &SymbolicName) const;



  unsigned getNumClobbers() const { return Clobbers.size(); }
  StringLiteral *getClobber(unsigned i) { return Clobbers[i]; }
  const StringLiteral *getClobber(unsigned i) const { return Clobbers[i]; }
  void setClobbers(StringLiteral **Clobbers, unsigned NumClobbers);

  virtual SourceRange getSourceRange() const {
    return SourceRange(AsmLoc, RParenLoc);
  }

  static bool classof(const Stmt *T) {return T->getStmtClass() == AsmStmtClass;}
  static bool classof(const AsmStmt *) { return true; }

  // Input expr iterators.

  typedef ExprIterator inputs_iterator;
  typedef ConstExprIterator const_inputs_iterator;

  inputs_iterator begin_inputs() {
    return Exprs.data() + NumOutputs;
  }

  inputs_iterator end_inputs() {
    return Exprs.data() + NumOutputs + NumInputs;
  }

  const_inputs_iterator begin_inputs() const {
    return Exprs.data() + NumOutputs;
  }

  const_inputs_iterator end_inputs() const {
    return Exprs.data() + NumOutputs + NumInputs;
  }

  // Output expr iterators.

  typedef ExprIterator outputs_iterator;
  typedef ConstExprIterator const_outputs_iterator;

  outputs_iterator begin_outputs() {
    return Exprs.data();
  }
  outputs_iterator end_outputs() {
    return Exprs.data() + NumOutputs;
  }

  const_outputs_iterator begin_outputs() const {
    return Exprs.data();
  }
  const_outputs_iterator end_outputs() const {
    return Exprs.data() + NumOutputs;
  }

  // Input name iterator.

  const std::string *begin_output_names() const {
    return &Names[0];
  }

  const std::string *end_output_names() const {
    return &Names[0] + NumOutputs;
  }

  // Child iterators

  virtual child_iterator child_begin();
  virtual child_iterator child_end();
};

}  // end namespace clang

#endif
