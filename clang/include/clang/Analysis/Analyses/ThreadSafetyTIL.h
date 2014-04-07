//===- ThreadSafetyTIL.h ---------------------------------------*- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a simple intermediate language that is used by the
// thread safety analysis (See ThreadSafety.cpp).  The thread safety analysis
// works by comparing mutex expressions, e.g.
//
// class A { Mutex mu; int dat GUARDED_BY(this->mu); }
// class B { A a; }
//
// void foo(B* b) {
//   (*b).a.mu.lock();     // locks (*b).a.mu
//   b->a.dat = 0;         // substitute &b->a for 'this';
//                         // requires lock on (&b->a)->mu
//   (b->a.mu).unlock();   // unlocks (b->a.mu)
// }
//
// As illustrated by the above example, clang Exprs are not well-suited to
// represent mutex expressions directly, since there is no easy way to compare
// Exprs for equivalence.  The thread safety analysis thus lowers clang Exprs
// into a simple intermediate language (IL).  The IL supports:
//
// (1) comparisons for semantic equality of expressions
// (2) SSA renaming of variables
// (3) wildcards and pattern matching over expressions
// (4) hash-based expression lookup
//
// The IL is currently very experimental, is intended only for use within
// the thread safety analysis, and is subject to change without notice.
// After the API stabilizes and matures, it may be appropriate to make this
// more generally available to other analyses.
//
// UNDER CONSTRUCTION.  USE AT YOUR OWN RISK.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_THREAD_SAFETY_TIL_H
#define LLVM_CLANG_THREAD_SAFETY_TIL_H

#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/Type.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/AlignOf.h"
#include "llvm/Support/Allocator.h"

#include <cassert>
#include <cstddef>

namespace clang {
namespace threadSafety {
namespace til {


// Simple wrapper class to abstract away from the details of memory management.
// SExprs are allocated in pools, and deallocated all at once.
class MemRegionRef {
private:
  union AlignmentType {
    double d;
    void *p;
    long double dd;
    long long ii;
  };

public:
  MemRegionRef() : Allocator(nullptr) {}
  MemRegionRef(llvm::BumpPtrAllocator *A) : Allocator(A) {}

  void *allocate(size_t Sz) {
    return Allocator->Allocate(Sz, llvm::AlignOf<AlignmentType>::Alignment);
  }

  template <typename T> T *allocateT() { return Allocator->Allocate<T>(); }

  template <typename T> T *allocateT(size_t NumElems) {
    return Allocator->Allocate<T>(NumElems);
  }

private:
  llvm::BumpPtrAllocator *Allocator;
};


} // end namespace til
} // end namespace threadSafety
} // end namespace clang


inline void *operator new(size_t Sz,
                          clang::threadSafety::til::MemRegionRef &R) {
  return R.allocate(Sz);
}


namespace clang {
namespace threadSafety {
namespace til {

using llvm::StringRef;

// A simple fixed size array class that does not manage its own memory,
// suitable for use with bump pointer allocation.
template <class T> class SimpleArray {
public:
  SimpleArray() : Data(nullptr), Size(0), Capacity(0) {}
  SimpleArray(T *Dat, size_t Cp, size_t Sz = 0)
      : Data(Dat), Size(0), Capacity(Cp) {}
  SimpleArray(MemRegionRef A, size_t Cp)
      : Data(A.allocateT<T>(Cp)), Size(0), Capacity(Cp) {}
  SimpleArray(SimpleArray<T> &A, bool Steal)
      : Data(A.Data), Size(A.Size), Capacity(A.Capacity) {
    A.Data = nullptr;
    A.Size = 0;
    A.Capacity = 0;
  }

  T *resize(size_t Ncp, MemRegionRef A) {
    T *Odata = Data;
    Data = A.allocateT<T>(Ncp);
    memcpy(Data, Odata, sizeof(T) * Size);
    return Odata;
  }

  typedef T *iterator;
  typedef const T *const_iterator;

  size_t size() const { return Size; }
  size_t capacity() const { return Capacity; }

  T &operator[](unsigned I) { return Data[I]; }
  const T &operator[](unsigned I) const { return Data[I]; }

  iterator begin() { return Data; }
  iterator end() { return Data + Size; }

  const_iterator cbegin() const { return Data; }
  const_iterator cend() const { return Data + Size; }

  void push_back(const T &Elem) {
    assert(Size < Capacity);
    Data[Size++] = Elem;
  }

  template <class Iter> unsigned append(Iter I, Iter E) {
    size_t Osz = Size;
    size_t J = Osz;
    for (; J < Capacity && I != E; ++J, ++I)
      Data[J] = *I;
    Size = J;
    return J - Osz;
  }

private:
  T *Data;
  size_t Size;
  size_t Capacity;
};


enum TIL_Opcode {
#define TIL_OPCODE_DEF(X) COP_##X,
#include "clang/Analysis/Analyses/ThreadSafetyOps.def"
#undef TIL_OPCODE_DEF
  COP_MAX
};


typedef clang::BinaryOperatorKind TIL_BinaryOpcode;
typedef clang::UnaryOperatorKind TIL_UnaryOpcode;
typedef clang::CastKind TIL_CastOpcode;


enum TIL_TraversalKind {
  TRV_Normal,
  TRV_Lazy, // subexpression may need to be traversed lazily
  TRV_Tail  // subexpression occurs in a tail position
};


// Base class for AST nodes in the typed intermediate language.
class SExpr {
public:
  TIL_Opcode opcode() const { return static_cast<TIL_Opcode>(Opcode); }

  // Subclasses of SExpr must define the following:
  //
  // This(const This& E, ...) {
  //   copy constructor: construct copy of E, with some additional arguments.
  // }
  //
  // template <class V> typename V::R_SExpr traverse(V &Visitor) {
  //   traverse all subexpressions, following the traversal/rewriter interface
  // }
  //
  // template <class C> typename C::CType compare(CType* E, C& Cmp) {
  //   compare all subexpressions, following the comparator interface
  // }

protected:
  SExpr(TIL_Opcode Op) : Opcode(Op), Reserved(0), Flags(0) {}
  SExpr(const SExpr &E) : Opcode(E.Opcode), Reserved(0), Flags(E.Flags) {}

  const unsigned char Opcode;
  unsigned char Reserved;
  unsigned short Flags;

private:
  SExpr();
};


// Class for owning references to SExprs.
// Includes attach/detach logic for counting variable references and lazy
// rewriting strategies.
class SExprRef {
public:
  SExprRef() : Ptr(nullptr) { }
  SExprRef(std::nullptr_t P) : Ptr(nullptr) { }
  SExprRef(SExprRef &&R) : Ptr(R.Ptr) { }

  // Defined after Variable and Future, below.
  inline SExprRef(SExpr *P);
  inline ~SExprRef();

  SExpr       *get()       { return Ptr; }
  const SExpr *get() const { return Ptr; }

  SExpr       *operator->()       { return get(); }
  const SExpr *operator->() const { return get(); }

  SExpr       &operator*()        { return *Ptr; }
  const SExpr &operator*() const  { return *Ptr; }

  bool operator==(const SExprRef& R) const { return Ptr == R.Ptr; }
  bool operator==(const SExpr* P)    const { return Ptr == P; }
  bool operator==(std::nullptr_t P)  const { return Ptr == nullptr; }

  inline void reset(SExpr *E);

private:
  inline void attach();
  inline void detach();

  SExprRef(const SExprRef& R) : Ptr(R.Ptr) { }

  SExpr *Ptr;
};


// Contains various helper functions for SExprs.
class ThreadSafetyTIL {
public:
  static const int MaxOpcode = COP_MAX;

  static inline bool isTrivial(SExpr *E) {
    unsigned Op = E->opcode();
    return Op == COP_Variable || Op == COP_Literal || Op == COP_LiteralPtr;
  }

  static inline bool isLargeValue(SExpr *E) {
    unsigned Op = E->opcode();
    return (Op >= COP_Function && Op <= COP_Code);
  }
};


class Function;
class SFunction;
class BasicBlock;


// A named variable, e.g. "x".
//
// There are two distinct places in which a Variable can appear in the AST.
// A variable declaration introduces a new variable, and can occur in 3 places:
//   Let-expressions:           (Let (x = t) u)
//   Functions:                 (Function (x : t) u)
//   Self-applicable functions  (SFunction (x) t)
//
// If a variable occurs in any other location, it is a reference to an existing
// variable declaration -- e.g. 'x' in (x * y + z). To save space, we don't
// allocate a separate AST node for variable references; a reference is just a
// pointer to the original declaration.
class Variable : public SExpr {
public:
  static bool classof(const SExpr *E) { return E->opcode() == COP_Variable; }

  // Let-variable, function parameter, or self-variable
  enum VariableKind {
    VK_Let,
    VK_Fun,
    VK_SFun
  };

  // These are defined after SExprRef contructor, below
  inline Variable(VariableKind K, SExpr *D = nullptr,
                  const clang::ValueDecl *Cvd = nullptr);
  inline Variable(const clang::ValueDecl *Cvd, SExpr *D = nullptr);
  inline Variable(const Variable &Vd, SExpr *D);

  VariableKind kind() const { return static_cast<VariableKind>(Flags); }

  StringRef name() const { return Cvdecl ? Cvdecl->getName() : "_x"; }
  const clang::ValueDecl *clangDecl() const { return Cvdecl; }

  // Returns the definition (for let vars) or type (for parameter & self vars)
  SExpr *definition() { return Definition.get(); }

  void attachVar() const { ++NumUses; }
  void detachVar() const { --NumUses; }

  unsigned getID() { return Id; }
  unsigned getBlockID() { return BlockID; }

  void setID(unsigned Bid, unsigned I) {
    BlockID = static_cast<unsigned short>(Bid);
    Id = static_cast<unsigned short>(I);
  }

  template <class V> typename V::R_SExpr traverse(V &Visitor) {
    // This routine is only called for variable references.
    return Visitor.reduceVariableRef(this);
  }

  template <class C> typename C::CType compare(Variable* E, C& Cmp) {
    return Cmp.compareVariableRefs(this, E);
  }

private:
  friend class Function;
  friend class SFunction;
  friend class BasicBlock;

  // Function, SFunction, and BasicBlock will reset the kind.
  void setKind(VariableKind K) { Flags = K; }

  SExprRef Definition;             // The TIL type or definition
  const clang::ValueDecl *Cvdecl;  // The clang declaration for this variable.

  unsigned short BlockID;
  unsigned short Id;
  mutable int NumUses;
};


// Placeholder for an expression that has not yet been created.
// Used to implement lazy copy and rewriting strategies.
class Future : public SExpr {
public:
  static bool classof(const SExpr *E) { return E->opcode() == COP_Future; }

  enum FutureStatus {
    FS_pending,
    FS_evaluating,
    FS_done
  };

  Future() :
    SExpr(COP_Future), Status(FS_pending), Result(nullptr), Location(nullptr)
  {}
  virtual ~Future() {}

  // Registers the location in the AST where this future is stored.
  // Forcing the future will automatically update the AST.
  static inline void registerLocation(SExprRef *Member) {
    if (Future *F = dyn_cast_or_null<Future>(Member->get()))
      F->Location = Member;
  }

  // A lazy rewriting strategy should subclass Future and override this method.
  virtual SExpr *create() { return nullptr; }

  // Return the result of this future if it exists, otherwise return null.
  SExpr *maybeGetResult() {
    return Result;
  }

  // Return the result of this future; forcing it if necessary.
  SExpr *result() {
    switch (Status) {
    case FS_pending:
      force();
      return Result;
    case FS_evaluating:
      return nullptr; // infinite loop; illegal recursion.
    case FS_done:
      return Result;
    }
  }

  template <class V> typename V::R_SExpr traverse(V &Visitor) {
    assert(Result && "Cannot traverse Future that has not been forced.");
    return Visitor.traverse(Result);
  }

  template <class C> typename C::CType compare(Future* E, C& Cmp) {
    if (!Result || !E->Result)
      return Cmp.comparePointers(this, E);
    return Cmp.compare(Result, E->Result);
  }

private:
  // Force the future.
  inline void force();

  FutureStatus Status;
  SExpr *Result;
  SExprRef *Location;
};



void SExprRef::attach() {
  if (!Ptr)
    return;

  TIL_Opcode Op = Ptr->opcode();
  if (Op == COP_Variable) {
    cast<Variable>(Ptr)->attachVar();
  }
  else if (Op == COP_Future) {
    cast<Future>(Ptr)->registerLocation(this);
  }
}

void SExprRef::detach() {
  if (Ptr && Ptr->opcode() == COP_Variable) {
    cast<Variable>(Ptr)->detachVar();
  }
}

SExprRef::SExprRef(SExpr *P) : Ptr(P) {
  if (P)
    attach();
}

SExprRef::~SExprRef() {
  detach();
}

void SExprRef::reset(SExpr *P) {
  if (Ptr)
    detach();
  Ptr = P;
  if (P)
    attach();
}


Variable::Variable(VariableKind K, SExpr *D, const clang::ValueDecl *Cvd)
    : SExpr(COP_Variable), Definition(D), Cvdecl(Cvd),
      BlockID(0), Id(0),  NumUses(0) {
  Flags = K;
}

Variable::Variable(const clang::ValueDecl *Cvd, SExpr *D)
    : SExpr(COP_Variable), Definition(D), Cvdecl(Cvd),
      BlockID(0), Id(0),  NumUses(0) {
  Flags = VK_Let;
}

Variable::Variable(const Variable &Vd, SExpr *D) // rewrite constructor
    : SExpr(Vd), Definition(D), Cvdecl(Vd.Cvdecl),
      BlockID(0), Id(0), NumUses(0) {
  Flags = Vd.kind();
}


void Future::force() {
  Status = FS_evaluating;
  SExpr *R = create();
  Result = R;
  if (Location) {
    Location->reset(R);
  }
  Status = FS_done;
}



// Placeholder for C++ expressions that cannot be represented in the TIL.
class Undefined : public SExpr {
public:
  static bool classof(const SExpr *E) { return E->opcode() == COP_Undefined; }

  Undefined(const clang::Stmt *S = nullptr) : SExpr(COP_Undefined), Cstmt(S) {}
  Undefined(const Undefined &U) : SExpr(U), Cstmt(U.Cstmt) {}

  template <class V> typename V::R_SExpr traverse(V &Visitor) {
    return Visitor.reduceUndefined(*this);
  }

  template <class C> typename C::CType compare(Undefined* E, C& Cmp) {
    return Cmp.comparePointers(Cstmt, E->Cstmt);
  }

private:
  const clang::Stmt *Cstmt;
};


// Placeholder for a wildcard that matches any other expression.
class Wildcard : public SExpr {
public:
  static bool classof(const SExpr *E) { return E->opcode() == COP_Wildcard; }

  Wildcard() : SExpr(COP_Wildcard) {}
  Wildcard(const Wildcard &W) : SExpr(W) {}

  template <class V> typename V::R_SExpr traverse(V &Visitor) {
    return Visitor.reduceWildcard(*this);
  }

  template <class C> typename C::CType compare(Wildcard* E, C& Cmp) {
    return Cmp.trueResult();
  }
};


// Base class for literal values.
class Literal : public SExpr {
public:
  static bool classof(const SExpr *E) { return E->opcode() == COP_Literal; }

  Literal(const clang::Expr *C) : SExpr(COP_Literal), Cexpr(C) {}
  Literal(const Literal &L) : SExpr(L), Cexpr(L.Cexpr) {}

  // The clang expression for this literal.
  const clang::Expr *clangExpr() { return Cexpr; }

  template <class V> typename V::R_SExpr traverse(V &Visitor) {
    return Visitor.reduceLiteral(*this);
  }

  template <class C> typename C::CType compare(Literal* E, C& Cmp) {
    // TODO -- use value, not pointer equality
    return Cmp.comparePointers(Cexpr, E->Cexpr);
  }

private:
  const clang::Expr *Cexpr;
};


// Literal pointer to an object allocated in memory.
// At compile time, pointer literals are represented by symbolic names.
class LiteralPtr : public SExpr {
public:
  static bool classof(const SExpr *E) { return E->opcode() == COP_LiteralPtr; }

  LiteralPtr(const clang::ValueDecl *D) : SExpr(COP_LiteralPtr), Cvdecl(D) {}
  LiteralPtr(const LiteralPtr &R) : SExpr(R), Cvdecl(R.Cvdecl) {}

  // The clang declaration for the value that this pointer points to.
  const clang::ValueDecl *clangDecl() { return Cvdecl; }

  template <class V> typename V::R_SExpr traverse(V &Visitor) {
    return Visitor.reduceLiteralPtr(*this);
  }

  template <class C> typename C::CType compare(LiteralPtr* E, C& Cmp) {
    return Cmp.comparePointers(Cvdecl, E->Cvdecl);
  }

private:
  const clang::ValueDecl *Cvdecl;
};





// A function -- a.k.a. lambda abstraction.
// Functions with multiple arguments are created by currying,
// e.g. (function (x: Int) (function (y: Int) (add x y)))
class Function : public SExpr {
public:
  static bool classof(const SExpr *E) { return E->opcode() == COP_Function; }

  Function(Variable *Vd, SExpr *Bd)
      : SExpr(COP_Function), VarDecl(Vd), Body(Bd) {
    Vd->setKind(Variable::VK_Fun);
  }
  Function(const Function &F, Variable *Vd, SExpr *Bd) // rewrite constructor
      : SExpr(F), VarDecl(Vd), Body(Bd) {
    Vd->setKind(Variable::VK_Fun);
  }

  Variable *variableDecl()  { return VarDecl; }
  const Variable *variableDecl() const { return VarDecl; }

  SExpr *body() { return Body.get(); }
  const SExpr *body() const { return Body.get(); }

  template <class V> typename V::R_SExpr traverse(V &Visitor) {
    // This is a variable declaration, so traverse the definition.
    typename V::R_SExpr E0 = Visitor.traverse(VarDecl->Definition, TRV_Lazy);
    // Tell the rewriter to enter the scope of the function.
    Variable *Nvd = Visitor.enterScope(*VarDecl, E0);
    typename V::R_SExpr E1 = Visitor.traverse(Body);
    Visitor.exitScope(*VarDecl);
    return Visitor.reduceFunction(*this, Nvd, E1);
  }

  template <class C> typename C::CType compare(Function* E, C& Cmp) {
    typename C::CType Ct =
      Cmp.compare(VarDecl->definition(), E->VarDecl->definition());
    if (Cmp.notTrue(Ct))
      return Ct;
    Cmp.enterScope(variableDecl(), E->variableDecl());
    Ct = Cmp.compare(body(), E->body());
    Cmp.leaveScope();
    return Ct;
  }

private:
  Variable *VarDecl;
  SExprRef Body;
};


// A self-applicable function.
// A self-applicable function can be applied to itself.  It's useful for
// implementing objects and late binding
class SFunction : public SExpr {
public:
  static bool classof(const SExpr *E) { return E->opcode() == COP_SFunction; }

  SFunction(Variable *Vd, SExpr *B)
      : SExpr(COP_SFunction), VarDecl(Vd), Body(B) {
    assert(Vd->Definition == nullptr);
    Vd->setKind(Variable::VK_SFun);
    Vd->Definition.reset(this);
  }
  SFunction(const SFunction &F, Variable *Vd, SExpr *B) // rewrite constructor
      : SExpr(F),
        VarDecl(Vd),
        Body(B) {
    assert(Vd->Definition == nullptr);
    Vd->setKind(Variable::VK_SFun);
    Vd->Definition.reset(this);
  }

  Variable *variableDecl() { return VarDecl; }
  const Variable *variableDecl() const { return VarDecl; }

  SExpr *body() { return Body.get(); }
  const SExpr *body() const { return Body.get(); }

  template <class V> typename V::R_SExpr traverse(V &Visitor) {
    // A self-variable points to the SFunction itself.
    // A rewrite must introduce the variable with a null definition, and update
    // it after 'this' has been rewritten.
    Variable *Nvd = Visitor.enterScope(*VarDecl, nullptr /* def */);
    typename V::R_SExpr E1 = Visitor.traverse(Body);
    Visitor.exitScope(*VarDecl);
    // A rewrite operation will call SFun constructor to set Vvd->Definition.
    return Visitor.reduceSFunction(*this, Nvd, E1);
  }

  template <class C> typename C::CType compare(SFunction* E, C& Cmp) {
    Cmp.enterScope(variableDecl(), E->variableDecl());
    typename C::CType Ct = Cmp.compare(body(), E->body());
    Cmp.leaveScope();
    return Ct;
  }

private:
  Variable *VarDecl;
  SExprRef Body;
};


// A block of code -- e.g. the body of a function.
class Code : public SExpr {
public:
  static bool classof(const SExpr *E) { return E->opcode() == COP_Code; }

  Code(SExpr *T, SExpr *B) : SExpr(COP_Code), ReturnType(T), Body(B) {}
  Code(const Code &C, SExpr *T, SExpr *B) // rewrite constructor
      : SExpr(C), ReturnType(T), Body(B) {}

  SExpr *returnType() { return ReturnType.get(); }
  const SExpr *returnType() const { return ReturnType.get(); }

  SExpr *body() { return Body.get(); }
  const SExpr *body() const { return Body.get(); }

  template <class V> typename V::R_SExpr traverse(V &Visitor) {
    typename V::R_SExpr Nt = Visitor.traverse(ReturnType, TRV_Lazy);
    typename V::R_SExpr Nb = Visitor.traverse(Body, TRV_Lazy);
    return Visitor.reduceCode(*this, Nt, Nb);
  }

  template <class C> typename C::CType compare(Code* E, C& Cmp) {
    typename C::CType Ct = Cmp.compare(returnType(), E->returnType());
    if (Cmp.notTrue(Ct))
      return Ct;
    return Cmp.compare(body(), E->body());
  }

private:
  SExprRef ReturnType;
  SExprRef Body;
};


// Apply an argument to a function
class Apply : public SExpr {
public:
  static bool classof(const SExpr *E) { return E->opcode() == COP_Apply; }

  Apply(SExpr *F, SExpr *A) : SExpr(COP_Apply), Fun(F), Arg(A) {}
  Apply(const Apply &A, SExpr *F, SExpr *Ar)  // rewrite constructor
      : SExpr(A), Fun(F), Arg(Ar)
  {}

  SExpr *fun() { return Fun.get(); }
  const SExpr *fun() const { return Fun.get(); }

  SExpr *arg() { return Arg.get(); }
  const SExpr *arg() const { return Arg.get(); }

  template <class V> typename V::R_SExpr traverse(V &Visitor) {
    typename V::R_SExpr Nf = Visitor.traverse(Fun);
    typename V::R_SExpr Na = Visitor.traverse(Arg);
    return Visitor.reduceApply(*this, Nf, Na);
  }

  template <class C> typename C::CType compare(Apply* E, C& Cmp) {
    typename C::CType Ct = Cmp.compare(fun(), E->fun());
    if (Cmp.notTrue(Ct))
      return Ct;
    return Cmp.compare(arg(), E->arg());
  }

private:
  SExprRef Fun;
  SExprRef Arg;
};


// Apply a self-argument to a self-applicable function
class SApply : public SExpr {
public:
  static bool classof(const SExpr *E) { return E->opcode() == COP_SApply; }

  SApply(SExpr *Sf, SExpr *A = nullptr)
      : SExpr(COP_SApply), Sfun(Sf), Arg(A)
  {}
  SApply(SApply &A, SExpr *Sf, SExpr *Ar = nullptr)  // rewrite constructor
      : SExpr(A),  Sfun(Sf), Arg(Ar)
  {}

  SExpr *sfun() { return Sfun.get(); }
  const SExpr *sfun() const { return Sfun.get(); }

  SExpr *arg() { return Arg.get() ? Arg.get() : Sfun.get(); }
  const SExpr *arg() const { return Arg.get() ? Arg.get() : Sfun.get(); }

  bool isDelegation() const { return Arg == nullptr; }

  template <class V> typename V::R_SExpr traverse(V &Visitor) {
    typename V::R_SExpr Nf = Visitor.traverse(Sfun);
    typename V::R_SExpr Na = Arg.get() ? Visitor.traverse(Arg) : nullptr;
    return Visitor.reduceSApply(*this, Nf, Na);
  }

  template <class C> typename C::CType compare(SApply* E, C& Cmp) {
    typename C::CType Ct = Cmp.compare(sfun(), E->sfun());
    if (Cmp.notTrue(Ct) || (!arg() && !E->arg()))
      return Ct;
    return Cmp.compare(arg(), E->arg());
  }

private:
  SExprRef Sfun;
  SExprRef Arg;
};


// Project a named slot from a C++ struct or class.
class Project : public SExpr {
public:
  static bool classof(const SExpr *E) { return E->opcode() == COP_Project; }

  Project(SExpr *R, clang::ValueDecl *Cvd)
      : SExpr(COP_Project), Rec(R), Cvdecl(Cvd) {}
  Project(const Project &P, SExpr *R) : SExpr(P), Rec(R), Cvdecl(P.Cvdecl) {}

  SExpr *record() { return Rec.get(); }
  const SExpr *record() const { return Rec.get(); }

  const clang::ValueDecl *clangValueDecl() const { return Cvdecl; }

  StringRef slotName() const { return Cvdecl->getName(); }

  template <class V> typename V::R_SExpr traverse(V &Visitor) {
    typename V::R_SExpr Nr = Visitor.traverse(Rec);
    return Visitor.reduceProject(*this, Nr);
  }

  template <class C> typename C::CType compare(Project* E, C& Cmp) {
    typename C::CType Ct = Cmp.compare(record(), E->record());
    if (Cmp.notTrue(Ct))
      return Ct;
    return Cmp.comparePointers(Cvdecl, E->Cvdecl);
  }

private:
  SExprRef Rec;
  clang::ValueDecl *Cvdecl;
};


// Call a function (after all arguments have been applied).
class Call : public SExpr {
public:
  static bool classof(const SExpr *E) { return E->opcode() == COP_Call; }

  Call(SExpr *T, const clang::CallExpr *Ce = nullptr)
      : SExpr(COP_Call), Target(T), Cexpr(Ce) {}
  Call(const Call &C, SExpr *T) : SExpr(C), Target(T), Cexpr(C.Cexpr) {}

  SExpr *target() { return Target.get(); }
  const SExpr *target() const { return Target.get(); }

  const clang::CallExpr *clangCallExpr() { return Cexpr; }

  template <class V> typename V::R_SExpr traverse(V &Visitor) {
    typename V::R_SExpr Nt = Visitor.traverse(Target);
    return Visitor.reduceCall(*this, Nt);
  }

  template <class C> typename C::CType compare(Call* E, C& Cmp) {
    return Cmp.compare(target(), E->target());
  }

private:
  SExprRef Target;
  const clang::CallExpr *Cexpr;
};


// Allocate memory for a new value on the heap or stack.
class Alloc : public SExpr {
public:
  static bool classof(const SExpr *E) { return E->opcode() == COP_Call; }

  enum AllocKind {
    AK_Stack,
    AK_Heap
  };

  Alloc(SExpr* D, AllocKind K) : SExpr(COP_Alloc), Dtype(D) {
    Flags = K;
  }
  Alloc(const Alloc &A, SExpr* Dt) : SExpr(A), Dtype(Dt) {
    Flags = A.kind();
  }

  AllocKind kind() const { return static_cast<AllocKind>(Flags); }

  SExpr* dataType() { return Dtype.get(); }
  const SExpr* dataType() const { return Dtype.get(); }

  template <class V> typename V::R_SExpr traverse(V &Visitor) {
    typename V::R_SExpr Nd = Visitor.traverse(Dtype);
    return Visitor.reduceAlloc(*this, Nd);
  }

  template <class C> typename C::CType compare(Alloc* E, C& Cmp) {
    typename C::CType Ct = Cmp.compareIntegers(kind(), E->kind());
    if (Cmp.notTrue(Ct))
      return Ct;
    return Cmp.compare(dataType(), E->dataType());
  }

private:
  SExprRef Dtype;
};


// Load a value from memory.
class Load : public SExpr {
public:
  static bool classof(const SExpr *E) { return E->opcode() == COP_Load; }

  Load(SExpr *P) : SExpr(COP_Load), Ptr(P) {}
  Load(const Load &L, SExpr *P) : SExpr(L), Ptr(P) {}

  SExpr *pointer() { return Ptr.get(); }
  const SExpr *pointer() const { return Ptr.get(); }

  template <class V> typename V::R_SExpr traverse(V &Visitor) {
    typename V::R_SExpr Np = Visitor.traverse(Ptr);
    return Visitor.reduceLoad(*this, Np);
  }

  template <class C> typename C::CType compare(Load* E, C& Cmp) {
    return Cmp.compare(pointer(), E->pointer());
  }

private:
  SExprRef Ptr;
};


// Store a value to memory.
class Store : public SExpr {
public:
  static bool classof(const SExpr *E) { return E->opcode() == COP_Store; }

  Store(SExpr *P, SExpr *V) : SExpr(COP_Store), Dest(P), Source(V) {}
  Store(const Store &S, SExpr *P, SExpr *V) : SExpr(S), Dest(P), Source(V) {}

  SExpr *destination() { return Dest.get(); }  // Address to store to
  const SExpr *destination() const { return Dest.get(); }

  SExpr *source() { return Source.get(); }     // Value to store
  const SExpr *source() const { return Source.get(); }

  template <class V> typename V::R_SExpr traverse(V &Visitor) {
    typename V::R_SExpr Np = Visitor.traverse(Dest);
    typename V::R_SExpr Nv = Visitor.traverse(Source);
    return Visitor.reduceStore(*this, Np, Nv);
  }

  template <class C> typename C::CType compare(Store* E, C& Cmp) {
    typename C::CType Ct = Cmp.compare(destination(), E->destination());
    if (Cmp.notTrue(Ct))
      return Ct;
    return Cmp.compare(source(), E->source());
  }

  SExprRef Dest;
  SExprRef Source;
};


// Simple unary operation -- e.g. !, ~, etc.
class UnaryOp : public SExpr {
public:
  static bool classof(const SExpr *E) { return E->opcode() == COP_UnaryOp; }

  UnaryOp(TIL_UnaryOpcode Op, SExpr *E) : SExpr(COP_UnaryOp), Expr0(E) {
    Flags = Op;
  }
  UnaryOp(const UnaryOp &U, SExpr *E) : SExpr(U) { Flags = U.Flags; }

  TIL_UnaryOpcode unaryOpcode() { return static_cast<TIL_UnaryOpcode>(Flags); }

  SExpr *expr() { return Expr0.get(); }
  const SExpr *expr() const { return Expr0.get(); }

  template <class V> typename V::R_SExpr traverse(V &Visitor) {
    typename V::R_SExpr Ne = Visitor.traverse(Expr0);
    return Visitor.reduceUnaryOp(*this, Ne);
  }

  template <class C> typename C::CType compare(UnaryOp* E, C& Cmp) {
    typename C::CType Ct =
      Cmp.compareIntegers(unaryOpcode(), E->unaryOpcode());
    if (Cmp.notTrue(Ct))
      return Ct;
    return Cmp.compare(expr(), E->expr());
  }

private:
  SExprRef Expr0;
};


// Simple binary operation -- e.g. +, -, etc.
class BinaryOp : public SExpr {
public:
  static bool classof(const SExpr *E) { return E->opcode() == COP_BinaryOp; }

  BinaryOp(TIL_BinaryOpcode Op, SExpr *E0, SExpr *E1)
      : SExpr(COP_BinaryOp), Expr0(E0), Expr1(E1) {
    Flags = Op;
  }
  BinaryOp(const BinaryOp &B, SExpr *E0, SExpr *E1)
      : SExpr(B), Expr0(E0), Expr1(E1) {
    Flags = B.Flags;
  }

  TIL_BinaryOpcode binaryOpcode() {
    return static_cast<TIL_BinaryOpcode>(Flags);
  }

  SExpr *expr0() { return Expr0.get(); }
  const SExpr *expr0() const { return Expr0.get(); }

  SExpr *expr1() { return Expr1.get(); }
  const SExpr *expr1() const { return Expr1.get(); }

  template <class V> typename V::R_SExpr traverse(V &Visitor) {
    typename V::R_SExpr Ne0 = Visitor.traverse(Expr0);
    typename V::R_SExpr Ne1 = Visitor.traverse(Expr1);
    return Visitor.reduceBinaryOp(*this, Ne0, Ne1);
  }

  template <class C> typename C::CType compare(BinaryOp* E, C& Cmp) {
    typename C::CType Ct =
      Cmp.compareIntegers(binaryOpcode(), E->binaryOpcode());
    if (Cmp.notTrue(Ct))
      return Ct;
    Ct = Cmp.compare(expr0(), E->expr0());
    if (Cmp.notTrue(Ct))
      return Ct;
    return Cmp.compare(expr1(), E->expr1());
  }

private:
  SExprRef Expr0;
  SExprRef Expr1;
};


// Cast expression
class Cast : public SExpr {
public:
  static bool classof(const SExpr *E) { return E->opcode() == COP_Cast; }

  Cast(TIL_CastOpcode Op, SExpr *E) : SExpr(COP_Cast), Expr0(E) { Flags = Op; }
  Cast(const Cast &C, SExpr *E) : SExpr(C), Expr0(E) { Flags = C.Flags; }

  TIL_BinaryOpcode castOpcode() {
    return static_cast<TIL_BinaryOpcode>(Flags);
  }

  SExpr *expr() { return Expr0.get(); }
  const SExpr *expr() const { return Expr0.get(); }

  template <class V> typename V::R_SExpr traverse(V &Visitor) {
    typename V::R_SExpr Ne = Visitor.traverse(Expr0);
    return Visitor.reduceCast(*this, Ne);
  }

  template <class C> typename C::CType compare(Cast* E, C& Cmp) {
    typename C::CType Ct =
      Cmp.compareIntegers(castOpcode(), E->castOpcode());
    if (Cmp.notTrue(Ct))
      return Ct;
    return Cmp.compare(expr(), E->expr());
  }

private:
  SExprRef Expr0;
};




class BasicBlock;


// An SCFG is a control-flow graph.  It consists of a set of basic blocks, each
// of which terminates in a branch to another basic block.  There is one
// entry point, and one exit point.
class SCFG : public SExpr {
public:
  typedef SimpleArray<BasicBlock*> BlockArray;

  static bool classof(const SExpr *E) { return E->opcode() == COP_SCFG; }

  SCFG(MemRegionRef A, unsigned Nblocks)
      : SExpr(COP_SCFG), Blocks(A, Nblocks), Entry(nullptr), Exit(nullptr) {}
  SCFG(const SCFG &Cfg, BlockArray &Ba) // steals memory from ba
      : SExpr(COP_SCFG),
        Blocks(Ba, true),
        Entry(nullptr),
        Exit(nullptr) {
    // TODO: set entry and exit!
  }

  typedef BlockArray::iterator iterator;
  typedef BlockArray::const_iterator const_iterator;

  iterator begin() { return Blocks.begin(); }
  iterator end() { return Blocks.end(); }

  const_iterator cbegin() const { return Blocks.cbegin(); }
  const_iterator cend() const { return Blocks.cend(); }

  BasicBlock *entry() const { return Entry; }
  BasicBlock *exit() const { return Exit; }

  void add(BasicBlock *BB) { Blocks.push_back(BB); }
  void setEntry(BasicBlock *BB) { Entry = BB; }
  void setExit(BasicBlock *BB) { Exit = BB; }

  template <class V> typename V::R_SExpr traverse(V &Visitor);

  template <class C> typename C::CType compare(SCFG* E, C& Cmp) {
    // TODO -- implement CFG comparisons
    return Cmp.comparePointers(this, E);
  }

private:
  BlockArray Blocks;
  BasicBlock *Entry;
  BasicBlock *Exit;
};


// A basic block is part of an SCFG, and can be treated as a function in
// continuation passing style.  It consists of a sequence of phi nodes, which
// are "arguments" to the function, followed by a sequence of instructions.
// Both arguments and instructions define new variables.  It ends with a
// branch or goto to another basic block in the same SCFG.
class BasicBlock {
public:
  typedef SimpleArray<Variable*> VarArray;

  BasicBlock(MemRegionRef A, unsigned Nargs, unsigned Nins,
             SExpr *Term = nullptr)
      : BlockID(0), Parent(nullptr), Args(A, Nargs), Instrs(A, Nins),
        Terminator(Term) {}
  BasicBlock(const BasicBlock &B, VarArray &As, VarArray &Is, SExpr *T)
      : BlockID(0), Parent(nullptr), Args(As, true), Instrs(Is, true),
        Terminator(T)
  {}

  unsigned blockID() const { return BlockID; }
  BasicBlock *parent() const { return Parent; }

  const VarArray &arguments() const { return Args; }
  VarArray &arguments() { return Args; }

  const VarArray &instructions() const { return Instrs; }
  VarArray &instructions() { return Instrs; }

  const SExpr *terminator() const { return Terminator.get(); }
  SExpr *terminator() { return Terminator.get(); }

  void setParent(BasicBlock *P) { Parent = P; }
  void setBlockID(unsigned i) { BlockID = i; }
  void setTerminator(SExpr *E) { Terminator.reset(E); }
  void addArgument(Variable *V) { Args.push_back(V); }
  void addInstr(Variable *V) { Args.push_back(V); }

  template <class V> BasicBlock *traverse(V &Visitor) {
    typename V::template Container<Variable*> Nas(Visitor, Args.size());
    typename V::template Container<Variable*> Nis(Visitor, Instrs.size());

    for (unsigned I = 0; I < Args.size(); ++I) {
      typename V::R_SExpr Ne = Visitor.traverse(Args[I]->Definition);
      Variable *Nvd = Visitor.enterScope(*Args[I], Ne);
      Nas.push_back(Nvd);
    }
    for (unsigned J = 0; J < Instrs.size(); ++J) {
      typename V::R_SExpr Ne = Visitor.traverse(Instrs[J]->Definition);
      Variable *Nvd = Visitor.enterScope(*Instrs[J], Ne);
      Nis.push_back(Nvd);
    }
    typename V::R_SExpr Nt = Visitor.traverse(Terminator);

    for (unsigned J = 0, JN = Instrs.size(); J < JN; ++J)
      Visitor.exitScope(*Instrs[JN-J]);
    for (unsigned I = 0, IN = Instrs.size(); I < IN; ++I)
      Visitor.exitScope(*Args[IN-I]);

    return Visitor.reduceBasicBlock(*this, Nas, Nis, Nt);
  }

  template <class C> typename C::CType compare(BasicBlock* E, C& Cmp) {
    // TODO -- implement CFG comparisons
    return Cmp.comparePointers(this, E);
  }

private:
  friend class SCFG;

  unsigned BlockID;
  BasicBlock *Parent;   // The parent block is the enclosing lexical scope.
                        // The parent dominates this block.
  VarArray Args;        // Phi nodes
  VarArray Instrs;
  SExprRef Terminator;
};


template <class V>
typename V::R_SExpr SCFG::traverse(V &Visitor) {
  Visitor.enterCFG(*this);
  typename V::template Container<BasicBlock *> Bbs(Visitor, Blocks.size());
  for (unsigned I = 0; I < Blocks.size(); ++I) {
    BasicBlock *Nbb = Blocks[I]->traverse(Visitor);
    Bbs.push_back(Nbb);
  }
  Visitor.exitCFG(*this);
  return Visitor.reduceSCFG(*this, Bbs);
}



class Phi : public SExpr {
public:
  // TODO: change to SExprRef
  typedef SimpleArray<SExpr*> ValArray;

  static bool classof(const SExpr *E) { return E->opcode() == COP_Phi; }

  Phi(MemRegionRef A, unsigned Nvals) : SExpr(COP_Phi), Values(A, Nvals) {}
  Phi(const Phi &P, ValArray &Vs)  // steals memory of vs
      : SExpr(COP_Phi), Values(Vs, true) {}

  const ValArray &values() const { return Values; }
  ValArray &values() { return Values; }

  template <class V> typename V::R_SExpr traverse(V &Visitor) {
    typename V::template Container<typename V::R_SExpr> Nvs(Visitor,
                                                            Values.size());
    for (ValArray::iterator I = Values.begin(), E = Values.end();
         I != E; ++I) {
      typename V::R_SExpr Nv = Visitor.traverse(*I);
      Nvs.push_back(Nv);
    }
    return Visitor.reducePhi(*this, Nvs);
  }

  template <class C> typename C::CType compare(Phi* E, C& Cmp) {
    // TODO -- implement CFG comparisons
    return Cmp.comparePointers(this, E);
  }

private:
  ValArray Values;
};


class Goto : public SExpr {
public:
  static bool classof(const SExpr *E) { return E->opcode() == COP_Goto; }

  Goto(BasicBlock *B, unsigned Index)
      : SExpr(COP_Goto), TargetBlock(B) {}
  Goto(const Goto &G, BasicBlock *B, unsigned Index)
      : SExpr(COP_Goto), TargetBlock(B) {}

  BasicBlock *targetBlock() const { return TargetBlock; }
  unsigned index() const { return Index; }

  template <class V> typename V::R_SExpr traverse(V &Visitor) {
    // TODO -- rewrite indices properly
    BasicBlock *Ntb = Visitor.reduceBasicBlockRef(TargetBlock);
    return Visitor.reduceGoto(*this, Ntb, Index);
  }

  template <class C> typename C::CType compare(Goto* E, C& Cmp) {
    // TODO -- implement CFG comparisons
    return Cmp.comparePointers(this, E);
  }

private:
  BasicBlock *TargetBlock;
  unsigned Index;   // Index into Phi nodes of target block.
};


class Branch : public SExpr {
public:
  static bool classof(const SExpr *E) { return E->opcode() == COP_Branch; }

  Branch(SExpr *C, BasicBlock *T, BasicBlock *E)
      : SExpr(COP_Branch), Condition(C), ThenBlock(T), ElseBlock(E) {}
  Branch(const Branch &Br, SExpr *C, BasicBlock *T, BasicBlock *E)
      : SExpr(COP_Branch), Condition(C), ThenBlock(T), ElseBlock(E) {}

  SExpr *condition() { return Condition; }
  BasicBlock *thenBlock() { return ThenBlock; }
  BasicBlock *elseBlock() { return ElseBlock; }

  template <class V> typename V::R_SExpr traverse(V &Visitor) {
    typename V::R_SExpr Nc = Visitor.traverse(Condition);
    BasicBlock *Ntb = Visitor.reduceBasicBlockRef(ThenBlock);
    BasicBlock *Nte = Visitor.reduceBasicBlockRef(ElseBlock);
    return Visitor.reduceBranch(*this, Nc, Ntb, Nte);
  }

  template <class C> typename C::CType compare(Branch* E, C& Cmp) {
    // TODO -- implement CFG comparisons
    return Cmp.comparePointers(this, E);
  }

private:
  SExpr *Condition;
  BasicBlock *ThenBlock;
  BasicBlock *ElseBlock;
};



// Defines an interface used to traverse SExprs.  Traversals have been made as
// generic as possible, and are intended to handle any kind of pass over the
// AST, e.g. visiters, copying, non-destructive rewriting, destructive
// (in-place) rewriting, hashing, typing, etc.
//
// Traversals implement the functional notion of a "fold" operation on SExprs.
// Each SExpr class provides a traverse method, which does the following:
//   * e->traverse(v):
//       // compute a result r_i for each subexpression e_i
//       for (i = 1..n)  r_i = v.traverse(e_i);
//       // combine results into a result for e,  where X is the class of e
//       return v.reduceX(*e, r_1, .. r_n).
//
// A visitor can control the traversal by overriding the following methods:
//   * v.traverse(e):
//       return v.traverseByCase(e), which returns v.traverseX(e)
//   * v.traverseX(e):   (X is the class of e)
//       return e->traverse(v).
//   * v.reduceX(*e, r_1, .. r_n):
//       compute a result for a node of type X
//
// The reduceX methods control the kind of traversal (visitor, copy, etc.).
// These are separated into a separate class R for the purpose of code reuse.
// The full reducer interface also has methods to handle scopes
template <class Self, class R> class TILTraversal : public R {
public:
  Self *self() { return reinterpret_cast<Self *>(this); }

  // Traverse an expression -- returning a result of type R_SExpr.
  // Override this method to do something for every expression, regardless
  // of which kind it is.  TIL_TraversalKind indicates the context in which
  // the expression occurs, and can be:
  //   TRV_Normal
  //   TRV_Lazy   -- e may need to be traversed lazily, using a Future.
  //   TRV_Tail   -- e occurs in a tail position
  typename R::R_SExpr traverse(SExprRef &E, TIL_TraversalKind K = TRV_Normal) {
    return traverse(E.get(), K);
  }

  typename R::R_SExpr traverse(SExpr *E, TIL_TraversalKind K = TRV_Normal) {
    return traverseByCase(E);
  }

  // Helper method to call traverseX(e) on the appropriate type.
  typename R::R_SExpr traverseByCase(SExpr *E) {
    switch (E->opcode()) {
#define TIL_OPCODE_DEF(X)                                                   \
    case COP_##X:                                                           \
      return self()->traverse##X(cast<X>(E));
#include "clang/Analysis/Analyses/ThreadSafetyOps.def"
#undef TIL_OPCODE_DEF
    case COP_MAX:
      return self()->reduceNull();
    }
  }

// Traverse e, by static dispatch on the type "X" of e.
// Override these methods to do something for a particular kind of term.
#define TIL_OPCODE_DEF(X)                                                   \
  typename R::R_SExpr traverse##X(X *e) { return e->traverse(*self()); }
#include "clang/Analysis/Analyses/ThreadSafetyOps.def"
#undef TIL_OPCODE_DEF
};


// Implements a Reducer that makes a deep copy of an SExpr.
// The default behavior of reduce##X(...) is to create a copy of the original.
// Subclasses can override reduce##X to implement non-destructive rewriting
// passes.
class TILCopyReducer {
public:
  TILCopyReducer() {}

  void setArena(MemRegionRef A) { Arena = A; }

  // R_SExpr is the result type for a traversal.
  // A copy or non-destructive rewrite returns a newly allocated term.
  typedef SExpr *R_SExpr;

  // Container is a minimal interface used to store results when traversing
  // SExprs of variable arity, such as Phi, Goto, and SCFG.
  template <class T> class Container {
  public:
    // Allocate a new container with a capacity for n elements.
    Container(TILCopyReducer &R, unsigned N) : Elems(R.Arena, N) {}

    // Push a new element onto the container.
    void push_back(T E) { Elems.push_back(E); }

  private:
    friend class TILCopyReducer;
    SimpleArray<T> Elems;
  };

public:
  R_SExpr reduceNull() {
    return nullptr;
  }
  // R_SExpr reduceFuture(...)  is never used.

  R_SExpr reduceUndefined(Undefined &Orig) {
    return new (Arena) Undefined(Orig);
  }
  R_SExpr reduceWildcard(Wildcard &Orig) {
    return new (Arena) Wildcard(Orig);
  }

  R_SExpr reduceLiteral(Literal &Orig) {
    return new (Arena) Literal(Orig);
  }
  R_SExpr reduceLiteralPtr(LiteralPtr &Orig) {
    return new (Arena) LiteralPtr(Orig);
  }

  R_SExpr reduceFunction(Function &Orig, Variable *Nvd, R_SExpr E0) {
    return new (Arena) Function(Orig, Nvd, E0);
  }
  R_SExpr reduceSFunction(SFunction &Orig, Variable *Nvd, R_SExpr E0) {
    return new (Arena) SFunction(Orig, Nvd, E0);
  }
  R_SExpr reduceCode(Code &Orig, R_SExpr E0, R_SExpr E1) {
    return new (Arena) Code(Orig, E0, E1);
  }

  R_SExpr reduceApply(Apply &Orig, R_SExpr E0, R_SExpr E1) {
    return new (Arena) Apply(Orig, E0, E1);
  }
  R_SExpr reduceSApply(SApply &Orig, R_SExpr E0, R_SExpr E1) {
    return new (Arena) SApply(Orig, E0, E1);
  }
  R_SExpr reduceProject(Project &Orig, R_SExpr E0) {
    return new (Arena) Project(Orig, E0);
  }
  R_SExpr reduceCall(Call &Orig, R_SExpr E0) {
    return new (Arena) Call(Orig, E0);
  }

  R_SExpr reduceAlloc(Alloc &Orig, R_SExpr E0) {
    return new (Arena) Alloc(Orig, E0);
  }
  R_SExpr reduceLoad(Load &Orig, R_SExpr E0) {
    return new (Arena) Load(Orig, E0);
  }
  R_SExpr reduceStore(Store &Orig, R_SExpr E0, R_SExpr E1) {
    return new (Arena) Store(Orig, E0, E1);
  }
  R_SExpr reduceUnaryOp(UnaryOp &Orig, R_SExpr E0) {
    return new (Arena) UnaryOp(Orig, E0);
  }
  R_SExpr reduceBinaryOp(BinaryOp &Orig, R_SExpr E0, R_SExpr E1) {
    return new (Arena) BinaryOp(Orig, E0, E1);
  }
  R_SExpr reduceCast(Cast &Orig, R_SExpr E0) {
    return new (Arena) Cast(Orig, E0);
  }

  R_SExpr reduceSCFG(SCFG &Orig, Container<BasicBlock *> Bbs) {
    return new (Arena) SCFG(Orig, Bbs.Elems);
  }
  R_SExpr reducePhi(Phi &Orig, Container<R_SExpr> As) {
    return new (Arena) Phi(Orig, As.Elems);
  }
  R_SExpr reduceGoto(Goto &Orig, BasicBlock *B, unsigned Index) {
    return new (Arena) Goto(Orig, B, Index);
  }
  R_SExpr reduceBranch(Branch &O, R_SExpr C, BasicBlock *B0, BasicBlock *B1) {
    return new (Arena) Branch(O, C, B0, B1);
  }

  BasicBlock *reduceBasicBlock(BasicBlock &Orig, Container<Variable *> &As,
                               Container<Variable *> &Is, R_SExpr T) {
    return new (Arena) BasicBlock(Orig, As.Elems, Is.Elems, T);
  }

  // Create a new variable from orig, and push it onto the lexical scope.
  Variable *enterScope(Variable &Orig, R_SExpr E0) {
    return new (Arena) Variable(Orig, E0);
  }
  // Exit the lexical scope of orig.
  void exitScope(const Variable &Orig) {}

  void enterCFG(SCFG &Cfg) {}
  void exitCFG(SCFG &Cfg) {}

  // Map Variable references to their rewritten definitions.
  Variable *reduceVariableRef(Variable *Ovd) { return Ovd; }

  // Map BasicBlock references to their rewritten defs.
  BasicBlock *reduceBasicBlockRef(BasicBlock *Obb) { return Obb; }

private:
  MemRegionRef Arena;
};


class SExprCopier : public TILTraversal<SExprCopier, TILCopyReducer> {
public:
  SExprCopier(MemRegionRef A) { setArena(A); }

  // Create a copy of e in region a.
  static SExpr *copy(SExpr *E, MemRegionRef A) {
    SExprCopier Copier(A);
    return Copier.traverse(E);
  }
};


// Implements a Reducer that visits each subexpression, and returns either
// true or false.
class TILVisitReducer {
public:
  TILVisitReducer() {}

  // A visitor returns a bool, representing success or failure.
  typedef bool R_SExpr;

  // A visitor "container" is a single bool, which accumulates success.
  template <class T> class Container {
  public:
    Container(TILVisitReducer &R, unsigned N) : Success(true) {}
    void push_back(bool E) { Success = Success && E; }

  private:
    friend class TILVisitReducer;
    bool Success;
  };

public:
  R_SExpr reduceNull() { return true; }
  R_SExpr reduceUndefined(Undefined &Orig) { return true; }
  R_SExpr reduceWildcard(Wildcard &Orig) { return true; }

  R_SExpr reduceLiteral(Literal &Orig) { return true; }
  R_SExpr reduceLiteralPtr(Literal &Orig) { return true; }

  R_SExpr reduceFunction(Function &Orig, Variable *Nvd, R_SExpr E0) {
    return Nvd && E0;
  }
  R_SExpr reduceSFunction(SFunction &Orig, Variable *Nvd, R_SExpr E0) {
    return Nvd && E0;
  }
  R_SExpr reduceCode(Code &Orig, R_SExpr E0, R_SExpr E1) {
    return E0 && E1;
  }
  R_SExpr reduceApply(Apply &Orig, R_SExpr E0, R_SExpr E1) {
    return E0 && E1;
  }
  R_SExpr reduceSApply(SApply &Orig, R_SExpr E0, R_SExpr E1) {
    return E0 && E1;
  }
  R_SExpr reduceProject(Project &Orig, R_SExpr E0) { return E0; }
  R_SExpr reduceCall(Call &Orig, R_SExpr E0) { return E0; }
  R_SExpr reduceAlloc(Alloc &Orig, R_SExpr E0) { return E0; }
  R_SExpr reduceLoad(Load &Orig, R_SExpr E0) { return E0; }
  R_SExpr reduceStore(Store &Orig, R_SExpr E0, R_SExpr E1) { return E0 && E1; }
  R_SExpr reduceUnaryOp(UnaryOp &Orig, R_SExpr E0) { return E0; }
  R_SExpr reduceBinaryOp(BinaryOp &Orig, R_SExpr E0, R_SExpr E1) {
    return E0 && E1;
  }
  R_SExpr reduceCast(Cast &Orig, R_SExpr E0) { return E0; }

  R_SExpr reduceSCFG(SCFG &Orig, Container<BasicBlock *> Bbs) {
    return Bbs.Success;
  }
   R_SExpr reducePhi(Phi &Orig, Container<R_SExpr> As) {
    return As.Success;
  }
  R_SExpr reduceGoto(Goto &Orig, BasicBlock *B, Container<R_SExpr> As) {
    return As.Success;
  }
  R_SExpr reduceBranch(Branch &O, R_SExpr C, BasicBlock *B0, BasicBlock *B1) {
    return C;
  }

  BasicBlock *reduceBasicBlock(BasicBlock &Orig, Container<Variable *> &As,
                               Container<Variable *> &Is, R_SExpr T) {
    return (As.Success && Is.Success && T) ? &Orig : nullptr;
  }

  Variable *enterScope(Variable &Orig, R_SExpr E0) {
    return E0 ? &Orig : nullptr;
  }
  void exitScope(const Variable &Orig) {}

  void enterCFG(SCFG &Cfg) {}
  void exitCFG(SCFG &Cfg) {}

  Variable *reduceVariableRef(Variable *Ovd) { return Ovd; }

  BasicBlock *reduceBasicBlockRef(BasicBlock *Obb) { return Obb; }
};


// A visitor will visit each node, and halt if any reducer returns false.
template <class Self>
class SExprVisitor : public TILTraversal<Self, TILVisitReducer> {
public:
  SExprVisitor() : Success(true) {}

  bool traverse(SExpr *E, TIL_TraversalKind K = TRV_Normal) {
    Success = Success && this->traverseByCase(E);
    return Success;
  }

  static bool visit(SExpr *E) {
    SExprVisitor Visitor;
    return Visitor.traverse(E);
  }

private:
  bool Success;
};


// Basic class for comparison operations over expressions.
template <typename Self>
class TILComparator {
public:
  Self *self() { return reinterpret_cast<Self *>(this); }

  bool compareByCase(SExpr *E1, SExpr* E2) {
    switch (E1->opcode()) {
#define TIL_OPCODE_DEF(X)                                                     \
    case COP_##X:                                                             \
      return cast<X>(E1)->compare(cast<X>(E2), *self());
#include "clang/Analysis/Analyses/ThreadSafetyOps.def"
#undef TIL_OPCODE_DEF
    case COP_MAX:
      return false;
    }
  }
};


class TILEqualsComparator : public TILComparator<TILEqualsComparator> {
public:
  // Result type for the comparison, e.g. bool for simple equality,
  // or int for lexigraphic comparison (-1, 0, 1).  Must have one value which
  // denotes "true".
  typedef bool CType;

  CType trueResult() { return true; }
  bool notTrue(CType ct) { return !ct; }

  bool compareIntegers(unsigned i, unsigned j) { return i == j; }
  bool comparePointers(const void* P, const void* Q) { return P == Q; }

  bool compare(SExpr *E1, SExpr* E2) {
    if (E1->opcode() != E2->opcode())
      return false;
    return compareByCase(E1, E2);
  }

  // TODO -- handle alpha-renaming of variables
  void enterScope(Variable* V1, Variable* V2) { }
  void leaveScope() { }

  bool compareVariableRefs(Variable* V1, Variable* V2) {
    return V1 == V2;
  }

  static bool compareExprs(SExpr *E1, SExpr* E2) {
    TILEqualsComparator Eq;
    return Eq.compare(E1, E2);
  }
};


// Pretty printer for TIL expressions
template <typename Self, typename StreamType>
class TILPrettyPrinter {
public:
  static void print(SExpr *E, StreamType &SS) {
    Self printer;
    printer.printSExpr(E, SS, Prec_MAX);
  }

protected:
  Self *self() { return reinterpret_cast<Self *>(this); }

  void newline(StreamType &SS) {
    SS << "\n";
  }

  // TODO: further distinguish between binary operations.
  static const unsigned Prec_Atom = 0;
  static const unsigned Prec_Postfix = 1;
  static const unsigned Prec_Unary = 2;
  static const unsigned Prec_Binary = 3;
  static const unsigned Prec_Other = 4;
  static const unsigned Prec_Decl = 5;
  static const unsigned Prec_MAX = 6;

  // Return the precedence of a given node, for use in pretty printing.
  unsigned precedence(SExpr *E) {
    switch (E->opcode()) {
      case COP_Future:     return Prec_Atom;
      case COP_Undefined:  return Prec_Atom;
      case COP_Wildcard:   return Prec_Atom;

      case COP_Literal:    return Prec_Atom;
      case COP_LiteralPtr: return Prec_Atom;
      case COP_Variable:   return Prec_Atom;
      case COP_Function:   return Prec_Decl;
      case COP_SFunction:  return Prec_Decl;
      case COP_Code:       return Prec_Decl;

      case COP_Apply:      return Prec_Postfix;
      case COP_SApply:     return Prec_Postfix;
      case COP_Project:    return Prec_Postfix;

      case COP_Call:       return Prec_Postfix;
      case COP_Alloc:      return Prec_Other;
      case COP_Load:       return Prec_Postfix;
      case COP_Store:      return Prec_Other;

      case COP_UnaryOp:    return Prec_Unary;
      case COP_BinaryOp:   return Prec_Binary;
      case COP_Cast:       return Prec_Unary;

      case COP_SCFG:       return Prec_Decl;
      case COP_Phi:        return Prec_Atom;
      case COP_Goto:       return Prec_Atom;
      case COP_Branch:     return Prec_Atom;
      case COP_MAX:        return Prec_MAX;
    }
    return Prec_MAX;
  }

  void printSExpr(SExpr *E, StreamType &SS, unsigned P) {
    if (!E) {
      self()->printNull(SS);
      return;
    }
    if (self()->precedence(E) > P) {
      // Wrap expr in () if necessary.
      SS << "(";
      self()->printSExpr(E, SS, Prec_MAX);
      SS << ")";
      return;
    }

    switch (E->opcode()) {
#define TIL_OPCODE_DEF(X)                                                  \
    case COP_##X:                                                          \
      self()->print##X(cast<X>(E), SS);                                    \
      return;
#include "clang/Analysis/Analyses/ThreadSafetyOps.def"
#undef TIL_OPCODE_DEF
    case COP_MAX:
      return;
    }
  }

  void printNull(StreamType &SS) {
    SS << "#null";
  }

  void printFuture(Future *E, StreamType &SS) {
    self()->printSExpr(E->maybeGetResult(), SS, Prec_Atom);
  }

  void printUndefined(Undefined *E, StreamType &SS) {
    SS << "#undefined";
  }

  void printWildcard(Wildcard *E, StreamType &SS) {
    SS << "_";
  }

  void printLiteral(Literal *E, StreamType &SS) {
    // TODO: actually pretty print the literal.
    SS << "#lit";
  }

  void printLiteralPtr(LiteralPtr *E, StreamType &SS) {
    SS << E->clangDecl()->getName();
  }

  void printVariable(Variable *E, StreamType &SS) {
    SS << E->name() << E->getBlockID() << "_" << E->getID();
  }

  void printFunction(Function *E, StreamType &SS, unsigned sugared = 0) {
    switch (sugared) {
      default:
        SS << "\\(";   // Lambda
      case 1:
        SS << "(";     // Slot declarations
        break;
      case 2:
        SS << ", ";    // Curried functions
        break;
    }
    self()->printVariable(E->variableDecl(), SS);
    SS << ": ";
    self()->printSExpr(E->variableDecl()->definition(), SS, Prec_MAX);

    SExpr *B = E->body();
    if (B && B->opcode() == COP_Function)
      self()->printFunction(cast<Function>(B), SS, 2);
    else
      self()->printSExpr(B, SS, Prec_Decl);
  }

  void printSFunction(SFunction *E, StreamType &SS) {
    SS << "@";
    self()->printVariable(E->variableDecl(), SS);
    SS << " ";
    self()->printSExpr(E->body(), SS, Prec_Decl);
  }

  void printCode(Code *E, StreamType &SS) {
    SS << ": ";
    self()->printSExpr(E->returnType(), SS, Prec_Decl-1);
    SS << " = ";
    self()->printSExpr(E->body(), SS, Prec_Decl);
  }

  void printApply(Apply *E, StreamType &SS, bool sugared = false) {
    SExpr *F = E->fun();
    if (F->opcode() == COP_Apply) {
      printApply(cast<Apply>(F), SS, true);
      SS << ", ";
    } else {
      self()->printSExpr(F, SS, Prec_Postfix);
      SS << "(";
    }
    self()->printSExpr(E->arg(), SS, Prec_MAX);
    if (!sugared)
      SS << ")$";
  }

  void printSApply(SApply *E, StreamType &SS) {
    self()->printSExpr(E->sfun(), SS, Prec_Postfix);
    SS << "@(";
    self()->printSExpr(E->arg(), SS, Prec_MAX);
    SS << ")";
  }

  void printProject(Project *E, StreamType &SS) {
    self()->printSExpr(E->record(), SS, Prec_Postfix);
    SS << ".";
    SS << E->slotName();
  }

  void printCall(Call *E, StreamType &SS) {
    SExpr *T = E->target();
    if (T->opcode() == COP_Apply) {
      self()->printApply(cast<Apply>(T), SS, true);
      SS << ")";
    }
    else {
      self()->printSExpr(T, SS, Prec_Postfix);
      SS << "()";
    }
  }

  void printAlloc(Alloc *E, StreamType &SS) {
    SS << "#alloc ";
    self()->printSExpr(E->dataType(), SS, Prec_Other-1);
  }

  void printLoad(Load *E, StreamType &SS) {
    self()->printSExpr(E->pointer(), SS, Prec_Postfix);
    SS << "^";
  }

  void printStore(Store *E, StreamType &SS) {
    self()->printSExpr(E->destination(), SS, Prec_Other-1);
    SS << " = ";
    self()->printSExpr(E->source(), SS, Prec_Other-1);
  }

  void printUnaryOp(UnaryOp *E, StreamType &SS) {
    self()->printSExpr(E->expr(), SS, Prec_Unary);
  }

  void printBinaryOp(BinaryOp *E, StreamType &SS) {
    self()->printSExpr(E->expr0(), SS, Prec_Binary-1);
    SS << " " << clang::BinaryOperator::getOpcodeStr(E->binaryOpcode()) << " ";
    self()->printSExpr(E->expr1(), SS, Prec_Binary-1);
  }

  void printCast(Cast *E, StreamType &SS) {
    SS << "~";
    self()->printSExpr(E->expr(), SS, Prec_Unary);
  }

  void printSCFG(SCFG *E, StreamType &SS) {
    SS << "#CFG {\n";
    for (auto BBI : *E) {
      SS << "BB_" << BBI->blockID() << ":";
      newline(SS);
      for (auto I : BBI->arguments()) {
        SS << "let ";
        self()->printVariable(I, SS);
        SS << " = ";
        self()->printSExpr(I->definition(), SS, Prec_MAX);
        SS << ";";
        newline(SS);
      }
      for (auto I : BBI->instructions()) {
        SS << "let ";
        self()->printVariable(I, SS);
        SS << " = ";
        self()->printSExpr(I->definition(), SS, Prec_MAX);
        SS << ";";
        newline(SS);
      }
      SExpr *T = BBI->terminator();
      if (T) {
        self()->printSExpr(T, SS, Prec_MAX);
        SS << ";";
        newline(SS);
      }
      newline(SS);
    }
    SS << "}";
    newline(SS);
  }

  void printPhi(Phi *E, StreamType &SS) {
    SS << "#phi(";
    unsigned i = 0;
    for (auto V : E->values()) {
      ++i;
      if (i > 0)
        SS << ", ";
      self()->printSExpr(V, SS, Prec_MAX);
    }
    SS << ")";
  }

  void printGoto(Goto *E, StreamType &SS) {
    SS << "#goto BB_";
    SS << E->targetBlock()->blockID();
    SS << ":";
    SS << E->index();
  }

  void printBranch(Branch *E, StreamType &SS) {
    SS << "#branch (";
    self()->printSExpr(E->condition(), SS, Prec_MAX);
    SS << ") BB_";
    SS << E->thenBlock()->blockID();
    SS << " BB_";
    SS << E->elseBlock()->blockID();
  }
};

} // end namespace til



} // end namespace threadSafety
} // end namespace clang

#endif // THREAD_SAFETY_TIL_H
