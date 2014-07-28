//===- ThreadSafetyTIL.h ---------------------------------------*- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT in the llvm repository for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a simple Typed Intermediate Language, or TIL, that is used
// by the thread safety analysis (See ThreadSafety.cpp).  The TIL is intended
// to be largely independent of clang, in the hope that the analysis can be
// reused for other non-C++ languages.  All dependencies on clang/llvm should
// go in ThreadSafetyUtil.h.
//
// Thread safety analysis works by comparing mutex expressions, e.g.
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
// The TIL is currently very experimental, is intended only for use within
// the thread safety analysis, and is subject to change without notice.
// After the API stabilizes and matures, it may be appropriate to make this
// more generally available to other analyses.
//
// UNDER CONSTRUCTION.  USE AT YOUR OWN RISK.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_THREAD_SAFETY_TIL_H
#define LLVM_CLANG_THREAD_SAFETY_TIL_H

// All clang include dependencies for this file must be put in
// ThreadSafetyUtil.h.
#include "ThreadSafetyUtil.h"

#include <stdint.h>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <utility>


namespace clang {
namespace threadSafety {
namespace til {


enum TIL_Opcode {
#define TIL_OPCODE_DEF(X) COP_##X,
#include "ThreadSafetyOps.def"
#undef TIL_OPCODE_DEF
};

enum TIL_UnaryOpcode : unsigned char {
  UOP_Minus,        //  -
  UOP_BitNot,       //  ~
  UOP_LogicNot      //  !
};

enum TIL_BinaryOpcode : unsigned char {
  BOP_Mul,          //  *
  BOP_Div,          //  /
  BOP_Rem,          //  %
  BOP_Add,          //  +
  BOP_Sub,          //  -
  BOP_Shl,          //  <<
  BOP_Shr,          //  >>
  BOP_BitAnd,       //  &
  BOP_BitXor,       //  ^
  BOP_BitOr,        //  |
  BOP_Eq,           //  ==
  BOP_Neq,          //  !=
  BOP_Lt,           //  <
  BOP_Leq,          //  <=
  BOP_LogicAnd,     //  &&
  BOP_LogicOr       //  ||
};

enum TIL_CastOpcode : unsigned char {
  CAST_none = 0,
  CAST_extendNum,   // extend precision of numeric type
  CAST_truncNum,    // truncate precision of numeric type
  CAST_toFloat,     // convert to floating point type
  CAST_toInt,       // convert to integer type
  CAST_objToPtr     // convert smart pointer to pointer  (C++ only)
};

const TIL_Opcode       COP_Min  = COP_Future;
const TIL_Opcode       COP_Max  = COP_Branch;
const TIL_UnaryOpcode  UOP_Min  = UOP_Minus;
const TIL_UnaryOpcode  UOP_Max  = UOP_LogicNot;
const TIL_BinaryOpcode BOP_Min  = BOP_Mul;
const TIL_BinaryOpcode BOP_Max  = BOP_LogicOr;
const TIL_CastOpcode   CAST_Min = CAST_none;
const TIL_CastOpcode   CAST_Max = CAST_toInt;

StringRef getUnaryOpcodeString(TIL_UnaryOpcode Op);
StringRef getBinaryOpcodeString(TIL_BinaryOpcode Op);


// ValueTypes are data types that can actually be held in registers.
// All variables and expressions must have a vBNF_Nonealue type.
// Pointer types are further subdivided into the various heap-allocated
// types, such as functions, records, etc.
// Structured types that are passed by value (e.g. complex numbers)
// require special handling; they use BT_ValueRef, and size ST_0.
struct ValueType {
  enum BaseType : unsigned char {
    BT_Void = 0,
    BT_Bool,
    BT_Int,
    BT_Float,
    BT_String,    // String literals
    BT_Pointer,
    BT_ValueRef
  };

  enum SizeType : unsigned char {
    ST_0 = 0,
    ST_1,
    ST_8,
    ST_16,
    ST_32,
    ST_64,
    ST_128
  };

  inline static SizeType getSizeType(unsigned nbytes);

  template <class T>
  inline static ValueType getValueType();

  ValueType(BaseType B, SizeType Sz, bool S, unsigned char VS)
      : Base(B), Size(Sz), Signed(S), VectSize(VS)
  { }

  BaseType      Base;
  SizeType      Size;
  bool          Signed;
  unsigned char VectSize;  // 0 for scalar, otherwise num elements in vector
};


inline ValueType::SizeType ValueType::getSizeType(unsigned nbytes) {
  switch (nbytes) {
    case 1: return ST_8;
    case 2: return ST_16;
    case 4: return ST_32;
    case 8: return ST_64;
    case 16: return ST_128;
    default: return ST_0;
  }
}


template<>
inline ValueType ValueType::getValueType<void>() {
  return ValueType(BT_Void, ST_0, false, 0);
}

template<>
inline ValueType ValueType::getValueType<bool>() {
  return ValueType(BT_Bool, ST_1, false, 0);
}

template<>
inline ValueType ValueType::getValueType<int8_t>() {
  return ValueType(BT_Int, ST_8, true, 0);
}

template<>
inline ValueType ValueType::getValueType<uint8_t>() {
  return ValueType(BT_Int, ST_8, false, 0);
}

template<>
inline ValueType ValueType::getValueType<int16_t>() {
  return ValueType(BT_Int, ST_16, true, 0);
}

template<>
inline ValueType ValueType::getValueType<uint16_t>() {
  return ValueType(BT_Int, ST_16, false, 0);
}

template<>
inline ValueType ValueType::getValueType<int32_t>() {
  return ValueType(BT_Int, ST_32, true, 0);
}

template<>
inline ValueType ValueType::getValueType<uint32_t>() {
  return ValueType(BT_Int, ST_32, false, 0);
}

template<>
inline ValueType ValueType::getValueType<int64_t>() {
  return ValueType(BT_Int, ST_64, true, 0);
}

template<>
inline ValueType ValueType::getValueType<uint64_t>() {
  return ValueType(BT_Int, ST_64, false, 0);
}

template<>
inline ValueType ValueType::getValueType<float>() {
  return ValueType(BT_Float, ST_32, true, 0);
}

template<>
inline ValueType ValueType::getValueType<double>() {
  return ValueType(BT_Float, ST_64, true, 0);
}

template<>
inline ValueType ValueType::getValueType<long double>() {
  return ValueType(BT_Float, ST_128, true, 0);
}

template<>
inline ValueType ValueType::getValueType<StringRef>() {
  return ValueType(BT_String, getSizeType(sizeof(StringRef)), false, 0);
}

template<>
inline ValueType ValueType::getValueType<void*>() {
  return ValueType(BT_Pointer, getSizeType(sizeof(void*)), false, 0);
}



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
  // template <class V>
  // typename V::R_SExpr traverse(V &Vs, typename V::R_Ctx Ctx) {
  //   traverse all subexpressions, following the traversal/rewriter interface.
  // }
  //
  // template <class C> typename C::CType compare(CType* E, C& Cmp) {
  //   compare all subexpressions, following the comparator interface
  // }

  void *operator new(size_t S, MemRegionRef &R) {
    return ::operator new(S, R);
  }

  // SExpr objects cannot be deleted.
  // This declaration is public to workaround a gcc bug that breaks building
  // with REQUIRES_EH=1.
  void operator delete(void *) LLVM_DELETED_FUNCTION;

protected:
  SExpr(TIL_Opcode Op) : Opcode(Op), Reserved(0), Flags(0) {}
  SExpr(const SExpr &E) : Opcode(E.Opcode), Reserved(0), Flags(E.Flags) {}

  const unsigned char Opcode;
  unsigned char Reserved;
  unsigned short Flags;

private:
  SExpr() LLVM_DELETED_FUNCTION;

  // SExpr objects must be created in an arena.
  void *operator new(size_t) LLVM_DELETED_FUNCTION;
};


// Class for owning references to SExprs.
// Includes attach/detach logic for counting variable references and lazy
// rewriting strategies.
class SExprRef {
public:
  SExprRef() : Ptr(nullptr) { }
  SExprRef(std::nullptr_t P) : Ptr(nullptr) { }
  SExprRef(SExprRef &&R) : Ptr(R.Ptr) { R.Ptr = nullptr; }

  // Defined after Variable and Future, below.
  inline SExprRef(SExpr *P);
  inline ~SExprRef();

  SExpr       *get()       { return Ptr; }
  const SExpr *get() const { return Ptr; }

  SExpr       *operator->()       { return get(); }
  const SExpr *operator->() const { return get(); }

  SExpr       &operator*()        { return *Ptr; }
  const SExpr &operator*() const  { return *Ptr; }

  bool operator==(const SExprRef &R) const { return Ptr == R.Ptr; }
  bool operator!=(const SExprRef &R) const { return !operator==(R); }
  bool operator==(const SExpr *P)    const { return Ptr == P; }
  bool operator!=(const SExpr *P)    const { return !operator==(P); }
  bool operator==(std::nullptr_t)    const { return Ptr == nullptr; }
  bool operator!=(std::nullptr_t)    const { return Ptr != nullptr; }

  inline void reset(SExpr *E);

private:
  inline void attach();
  inline void detach();

  SExpr *Ptr;
};


// Contains various helper functions for SExprs.
namespace ThreadSafetyTIL {
  inline bool isTrivial(const SExpr *E) {
    unsigned Op = E->opcode();
    return Op == COP_Variable || Op == COP_Literal || Op == COP_LiteralPtr;
  }
}

// Nodes which declare variables
class Function;
class SFunction;
class BasicBlock;
class Let;


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
    VK_LetBB,
    VK_Fun,
    VK_SFun
  };

  // These are defined after SExprRef contructor, below
  inline Variable(SExpr *D, const clang::ValueDecl *Cvd = nullptr);
  inline Variable(StringRef s, SExpr *D = nullptr);
  inline Variable(const Variable &Vd, SExpr *D);

  VariableKind kind() const { return static_cast<VariableKind>(Flags); }

  const StringRef name() const { return Name; }
  const clang::ValueDecl *clangDecl() const { return Cvdecl; }

  // Returns the definition (for let vars) or type (for parameter & self vars)
  SExpr *definition() { return Definition.get(); }
  const SExpr *definition() const { return Definition.get(); }

  void attachVar() const { ++NumUses; }
  void detachVar() const { assert(NumUses > 0); --NumUses; }

  unsigned getID() const { return Id; }
  unsigned getBlockID() const { return BlockID; }

  void setName(StringRef S) { Name = S; }
  void setID(unsigned Bid, unsigned I) {
    BlockID = static_cast<unsigned short>(Bid);
    Id = static_cast<unsigned short>(I);
  }
  void setClangDecl(const clang::ValueDecl *VD) { Cvdecl = VD; }
  void setDefinition(SExpr *E);
  void setKind(VariableKind K) { Flags = K; }

  template <class V>
  typename V::R_SExpr traverse(V &Vs, typename V::R_Ctx Ctx) {
    // This routine is only called for variable references.
    return Vs.reduceVariableRef(this);
  }

  template <class C>
  typename C::CType compare(const Variable* E, C& Cmp) const {
    return Cmp.compareVariableRefs(this, E);
  }

private:
  friend class Function;
  friend class SFunction;
  friend class BasicBlock;
  friend class Let;

  StringRef Name;                  // The name of the variable.
  SExprRef  Definition;            // The TIL type or definition
  const clang::ValueDecl *Cvdecl;  // The clang declaration for this variable.

  unsigned short BlockID;
  unsigned short Id;
  mutable unsigned NumUses;
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
private:
  virtual ~Future() LLVM_DELETED_FUNCTION;
public:

  // Registers the location in the AST where this future is stored.
  // Forcing the future will automatically update the AST.
  static inline void registerLocation(SExprRef *Member) {
    if (Future *F = dyn_cast_or_null<Future>(Member->get()))
      F->Location = Member;
  }

  // A lazy rewriting strategy should subclass Future and override this method.
  virtual SExpr *create() { return nullptr; }

  // Return the result of this future if it exists, otherwise return null.
  SExpr *maybeGetResult() const {
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

  template <class V>
  typename V::R_SExpr traverse(V &Vs, typename V::R_Ctx Ctx) {
    assert(Result && "Cannot traverse Future that has not been forced.");
    return Vs.traverse(Result, Ctx);
  }

  template <class C>
  typename C::CType compare(const Future* E, C& Cmp) const {
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


inline void SExprRef::attach() {
  if (!Ptr)
    return;

  TIL_Opcode Op = Ptr->opcode();
  if (Op == COP_Variable) {
    cast<Variable>(Ptr)->attachVar();
  } else if (Op == COP_Future) {
    cast<Future>(Ptr)->registerLocation(this);
  }
}

inline void SExprRef::detach() {
  if (Ptr && Ptr->opcode() == COP_Variable) {
    cast<Variable>(Ptr)->detachVar();
  }
}

inline SExprRef::SExprRef(SExpr *P) : Ptr(P) {
  attach();
}

inline SExprRef::~SExprRef() {
  detach();
}

inline void SExprRef::reset(SExpr *P) {
  detach();
  Ptr = P;
  attach();
}


inline Variable::Variable(StringRef s, SExpr *D)
    : SExpr(COP_Variable), Name(s), Definition(D), Cvdecl(nullptr),
      BlockID(0), Id(0), NumUses(0) {
  Flags = VK_Let;
}

inline Variable::Variable(SExpr *D, const clang::ValueDecl *Cvd)
    : SExpr(COP_Variable), Name(Cvd ? Cvd->getName() : "_x"),
      Definition(D), Cvdecl(Cvd), BlockID(0), Id(0), NumUses(0) {
  Flags = VK_Let;
}

inline Variable::Variable(const Variable &Vd, SExpr *D) // rewrite constructor
    : SExpr(Vd), Name(Vd.Name), Definition(D), Cvdecl(Vd.Cvdecl),
      BlockID(0), Id(0), NumUses(0) {
  Flags = Vd.kind();
}

inline void Variable::setDefinition(SExpr *E) {
  Definition.reset(E);
}

void Future::force() {
  Status = FS_evaluating;
  SExpr *R = create();
  Result = R;
  if (Location)
    Location->reset(R);
  Status = FS_done;
}


// Placeholder for C++ expressions that cannot be represented in the TIL.
class Undefined : public SExpr {
public:
  static bool classof(const SExpr *E) { return E->opcode() == COP_Undefined; }

  Undefined(const clang::Stmt *S = nullptr) : SExpr(COP_Undefined), Cstmt(S) {}
  Undefined(const Undefined &U) : SExpr(U), Cstmt(U.Cstmt) {}

  template <class V>
  typename V::R_SExpr traverse(V &Vs, typename V::R_Ctx Ctx) {
    return Vs.reduceUndefined(*this);
  }

  template <class C>
  typename C::CType compare(const Undefined* E, C& Cmp) const {
    return Cmp.trueResult();
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

  template <class V> typename V::R_SExpr traverse(V &Vs, typename V::R_Ctx Ctx) {
    return Vs.reduceWildcard(*this);
  }

  template <class C>
  typename C::CType compare(const Wildcard* E, C& Cmp) const {
    return Cmp.trueResult();
  }
};


template <class T> class LiteralT;

// Base class for literal values.
class Literal : public SExpr {
public:
  static bool classof(const SExpr *E) { return E->opcode() == COP_Literal; }

  Literal(const clang::Expr *C)
     : SExpr(COP_Literal), ValType(ValueType::getValueType<void>()), Cexpr(C)
  { }
  Literal(ValueType VT) : SExpr(COP_Literal), ValType(VT), Cexpr(nullptr) {}
  Literal(const Literal &L) : SExpr(L), ValType(L.ValType), Cexpr(L.Cexpr) {}

  // The clang expression for this literal.
  const clang::Expr *clangExpr() const { return Cexpr; }

  ValueType valueType() const { return ValType; }

  template<class T> const LiteralT<T>& as() const {
    return *static_cast<const LiteralT<T>*>(this);
  }
  template<class T> LiteralT<T>& as() {
    return *static_cast<LiteralT<T>*>(this);
  }

  template <class V> typename V::R_SExpr traverse(V &Vs, typename V::R_Ctx Ctx);

  template <class C>
  typename C::CType compare(const Literal* E, C& Cmp) const {
    // TODO: defer actual comparison to LiteralT
    return Cmp.trueResult();
  }

private:
  const ValueType ValType;
  const clang::Expr *Cexpr;
};


// Derived class for literal values, which stores the actual value.
template<class T>
class LiteralT : public Literal {
public:
  LiteralT(T Dat) : Literal(ValueType::getValueType<T>()), Val(Dat) { }
  LiteralT(const LiteralT<T> &L) : Literal(L), Val(L.Val) { }

  T  value() const { return Val;}
  T& value() { return Val; }

private:
  T Val;
};



template <class V>
typename V::R_SExpr Literal::traverse(V &Vs, typename V::R_Ctx Ctx) {
  if (Cexpr)
    return Vs.reduceLiteral(*this);

  switch (ValType.Base) {
  case ValueType::BT_Void:
    break;
  case ValueType::BT_Bool:
    return Vs.reduceLiteralT(as<bool>());
  case ValueType::BT_Int: {
    switch (ValType.Size) {
    case ValueType::ST_8:
      if (ValType.Signed)
        return Vs.reduceLiteralT(as<int8_t>());
      else
        return Vs.reduceLiteralT(as<uint8_t>());
    case ValueType::ST_16:
      if (ValType.Signed)
        return Vs.reduceLiteralT(as<int16_t>());
      else
        return Vs.reduceLiteralT(as<uint16_t>());
    case ValueType::ST_32:
      if (ValType.Signed)
        return Vs.reduceLiteralT(as<int32_t>());
      else
        return Vs.reduceLiteralT(as<uint32_t>());
    case ValueType::ST_64:
      if (ValType.Signed)
        return Vs.reduceLiteralT(as<int64_t>());
      else
        return Vs.reduceLiteralT(as<uint64_t>());
    default:
      break;
    }
  }
  case ValueType::BT_Float: {
    switch (ValType.Size) {
    case ValueType::ST_32:
      return Vs.reduceLiteralT(as<float>());
    case ValueType::ST_64:
      return Vs.reduceLiteralT(as<double>());
    default:
      break;
    }
  }
  case ValueType::BT_String:
    return Vs.reduceLiteralT(as<StringRef>());
  case ValueType::BT_Pointer:
    return Vs.reduceLiteralT(as<void*>());
  case ValueType::BT_ValueRef:
    break;
  }
  return Vs.reduceLiteral(*this);
}


// Literal pointer to an object allocated in memory.
// At compile time, pointer literals are represented by symbolic names.
class LiteralPtr : public SExpr {
public:
  static bool classof(const SExpr *E) { return E->opcode() == COP_LiteralPtr; }

  LiteralPtr(const clang::ValueDecl *D) : SExpr(COP_LiteralPtr), Cvdecl(D) {}
  LiteralPtr(const LiteralPtr &R) : SExpr(R), Cvdecl(R.Cvdecl) {}

  // The clang declaration for the value that this pointer points to.
  const clang::ValueDecl *clangDecl() const { return Cvdecl; }

  template <class V>
  typename V::R_SExpr traverse(V &Vs, typename V::R_Ctx Ctx) {
    return Vs.reduceLiteralPtr(*this);
  }

  template <class C>
  typename C::CType compare(const LiteralPtr* E, C& Cmp) const {
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

  template <class V>
  typename V::R_SExpr traverse(V &Vs, typename V::R_Ctx Ctx) {
    // This is a variable declaration, so traverse the definition.
    auto E0 = Vs.traverse(VarDecl->Definition, Vs.typeCtx(Ctx));
    // Tell the rewriter to enter the scope of the function.
    Variable *Nvd = Vs.enterScope(*VarDecl, E0);
    auto E1 = Vs.traverse(Body, Vs.declCtx(Ctx));
    Vs.exitScope(*VarDecl);
    return Vs.reduceFunction(*this, Nvd, E1);
  }

  template <class C>
  typename C::CType compare(const Function* E, C& Cmp) const {
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
      : SExpr(F), VarDecl(Vd), Body(B) {
    assert(Vd->Definition == nullptr);
    Vd->setKind(Variable::VK_SFun);
    Vd->Definition.reset(this);
  }

  Variable *variableDecl() { return VarDecl; }
  const Variable *variableDecl() const { return VarDecl; }

  SExpr *body() { return Body.get(); }
  const SExpr *body() const { return Body.get(); }

  template <class V>
  typename V::R_SExpr traverse(V &Vs, typename V::R_Ctx Ctx) {
    // A self-variable points to the SFunction itself.
    // A rewrite must introduce the variable with a null definition, and update
    // it after 'this' has been rewritten.
    Variable *Nvd = Vs.enterScope(*VarDecl, nullptr);
    auto E1 = Vs.traverse(Body, Vs.declCtx(Ctx));
    Vs.exitScope(*VarDecl);
    // A rewrite operation will call SFun constructor to set Vvd->Definition.
    return Vs.reduceSFunction(*this, Nvd, E1);
  }

  template <class C>
  typename C::CType compare(const SFunction* E, C& Cmp) const {
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

  template <class V>
  typename V::R_SExpr traverse(V &Vs, typename V::R_Ctx Ctx) {
    auto Nt = Vs.traverse(ReturnType, Vs.typeCtx(Ctx));
    auto Nb = Vs.traverse(Body,       Vs.lazyCtx(Ctx));
    return Vs.reduceCode(*this, Nt, Nb);
  }

  template <class C>
  typename C::CType compare(const Code* E, C& Cmp) const {
    typename C::CType Ct = Cmp.compare(returnType(), E->returnType());
    if (Cmp.notTrue(Ct))
      return Ct;
    return Cmp.compare(body(), E->body());
  }

private:
  SExprRef ReturnType;
  SExprRef Body;
};


// A typed, writable location in memory
class Field : public SExpr {
public:
  static bool classof(const SExpr *E) { return E->opcode() == COP_Field; }

  Field(SExpr *R, SExpr *B) : SExpr(COP_Field), Range(R), Body(B) {}
  Field(const Field &C, SExpr *R, SExpr *B) // rewrite constructor
      : SExpr(C), Range(R), Body(B) {}

  SExpr *range() { return Range.get(); }
  const SExpr *range() const { return Range.get(); }

  SExpr *body() { return Body.get(); }
  const SExpr *body() const { return Body.get(); }

  template <class V>
  typename V::R_SExpr traverse(V &Vs, typename V::R_Ctx Ctx) {
    auto Nr = Vs.traverse(Range, Vs.typeCtx(Ctx));
    auto Nb = Vs.traverse(Body,  Vs.lazyCtx(Ctx));
    return Vs.reduceField(*this, Nr, Nb);
  }

  template <class C>
  typename C::CType compare(const Field* E, C& Cmp) const {
    typename C::CType Ct = Cmp.compare(range(), E->range());
    if (Cmp.notTrue(Ct))
      return Ct;
    return Cmp.compare(body(), E->body());
  }

private:
  SExprRef Range;
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

  template <class V>
  typename V::R_SExpr traverse(V &Vs, typename V::R_Ctx Ctx) {
    auto Nf = Vs.traverse(Fun, Vs.subExprCtx(Ctx));
    auto Na = Vs.traverse(Arg, Vs.subExprCtx(Ctx));
    return Vs.reduceApply(*this, Nf, Na);
  }

  template <class C>
  typename C::CType compare(const Apply* E, C& Cmp) const {
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

  SApply(SExpr *Sf, SExpr *A = nullptr) : SExpr(COP_SApply), Sfun(Sf), Arg(A) {}
  SApply(SApply &A, SExpr *Sf, SExpr *Ar = nullptr) // rewrite constructor
      : SExpr(A), Sfun(Sf), Arg(Ar) {}

  SExpr *sfun() { return Sfun.get(); }
  const SExpr *sfun() const { return Sfun.get(); }

  SExpr *arg() { return Arg.get() ? Arg.get() : Sfun.get(); }
  const SExpr *arg() const { return Arg.get() ? Arg.get() : Sfun.get(); }

  bool isDelegation() const { return Arg != nullptr; }

  template <class V>
  typename V::R_SExpr traverse(V &Vs, typename V::R_Ctx Ctx) {
    auto Nf = Vs.traverse(Sfun, Vs.subExprCtx(Ctx));
    typename V::R_SExpr Na = Arg.get() ? Vs.traverse(Arg, Vs.subExprCtx(Ctx))
                                       : nullptr;
    return Vs.reduceSApply(*this, Nf, Na);
  }

  template <class C>
  typename C::CType compare(const SApply* E, C& Cmp) const {
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

  Project(SExpr *R, StringRef SName)
      : SExpr(COP_Project), Rec(R), SlotName(SName), Cvdecl(nullptr)
  { }
  Project(SExpr *R, const clang::ValueDecl *Cvd)
      : SExpr(COP_Project), Rec(R), SlotName(Cvd->getName()), Cvdecl(Cvd)
  { }
  Project(const Project &P, SExpr *R)
      : SExpr(P), Rec(R), SlotName(P.SlotName), Cvdecl(P.Cvdecl)
  { }

  SExpr *record() { return Rec.get(); }
  const SExpr *record() const { return Rec.get(); }

  const clang::ValueDecl *clangDecl() const { return Cvdecl; }

  bool isArrow() const { return (Flags & 0x01) != 0; }
  void setArrow(bool b) {
    if (b) Flags |= 0x01;
    else Flags &= 0xFFFE;
  }

  StringRef slotName() const {
    if (Cvdecl)
      return Cvdecl->getName();
    else
      return SlotName;
  }

  template <class V>
  typename V::R_SExpr traverse(V &Vs, typename V::R_Ctx Ctx) {
    auto Nr = Vs.traverse(Rec, Vs.subExprCtx(Ctx));
    return Vs.reduceProject(*this, Nr);
  }

  template <class C>
  typename C::CType compare(const Project* E, C& Cmp) const {
    typename C::CType Ct = Cmp.compare(record(), E->record());
    if (Cmp.notTrue(Ct))
      return Ct;
    return Cmp.comparePointers(Cvdecl, E->Cvdecl);
  }

private:
  SExprRef Rec;
  StringRef SlotName;
  const clang::ValueDecl *Cvdecl;
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

  const clang::CallExpr *clangCallExpr() const { return Cexpr; }

  template <class V>
  typename V::R_SExpr traverse(V &Vs, typename V::R_Ctx Ctx) {
    auto Nt = Vs.traverse(Target, Vs.subExprCtx(Ctx));
    return Vs.reduceCall(*this, Nt);
  }

  template <class C>
  typename C::CType compare(const Call* E, C& Cmp) const {
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

  Alloc(SExpr *D, AllocKind K) : SExpr(COP_Alloc), Dtype(D) { Flags = K; }
  Alloc(const Alloc &A, SExpr *Dt) : SExpr(A), Dtype(Dt) { Flags = A.kind(); }

  AllocKind kind() const { return static_cast<AllocKind>(Flags); }

  SExpr *dataType() { return Dtype.get(); }
  const SExpr *dataType() const { return Dtype.get(); }

  template <class V>
  typename V::R_SExpr traverse(V &Vs, typename V::R_Ctx Ctx) {
    auto Nd = Vs.traverse(Dtype, Vs.declCtx(Ctx));
    return Vs.reduceAlloc(*this, Nd);
  }

  template <class C>
  typename C::CType compare(const Alloc* E, C& Cmp) const {
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

  template <class V>
  typename V::R_SExpr traverse(V &Vs, typename V::R_Ctx Ctx) {
    auto Np = Vs.traverse(Ptr, Vs.subExprCtx(Ctx));
    return Vs.reduceLoad(*this, Np);
  }

  template <class C>
  typename C::CType compare(const Load* E, C& Cmp) const {
    return Cmp.compare(pointer(), E->pointer());
  }

private:
  SExprRef Ptr;
};


// Store a value to memory.
// Source is a pointer, destination is the value to store.
class Store : public SExpr {
public:
  static bool classof(const SExpr *E) { return E->opcode() == COP_Store; }

  Store(SExpr *P, SExpr *V) : SExpr(COP_Store), Dest(P), Source(V) {}
  Store(const Store &S, SExpr *P, SExpr *V) : SExpr(S), Dest(P), Source(V) {}

  SExpr *destination() { return Dest.get(); }  // Address to store to
  const SExpr *destination() const { return Dest.get(); }

  SExpr *source() { return Source.get(); }     // Value to store
  const SExpr *source() const { return Source.get(); }

  template <class V>
  typename V::R_SExpr traverse(V &Vs, typename V::R_Ctx Ctx) {
    auto Np = Vs.traverse(Dest,   Vs.subExprCtx(Ctx));
    auto Nv = Vs.traverse(Source, Vs.subExprCtx(Ctx));
    return Vs.reduceStore(*this, Np, Nv);
  }

  template <class C>
  typename C::CType compare(const Store* E, C& Cmp) const {
    typename C::CType Ct = Cmp.compare(destination(), E->destination());
    if (Cmp.notTrue(Ct))
      return Ct;
    return Cmp.compare(source(), E->source());
  }

private:
  SExprRef Dest;
  SExprRef Source;
};


// If p is a reference to an array, then first(p) is a reference to the first
// element.  The usual array notation p[i]  becomes first(p + i).
class ArrayIndex : public SExpr {
public:
  static bool classof(const SExpr *E) { return E->opcode() == COP_ArrayIndex; }

  ArrayIndex(SExpr *A, SExpr *N) : SExpr(COP_ArrayIndex), Array(A), Index(N) {}
  ArrayIndex(const ArrayIndex &E, SExpr *A, SExpr *N)
    : SExpr(E), Array(A), Index(N) {}

  SExpr *array() { return Array.get(); }
  const SExpr *array() const { return Array.get(); }

  SExpr *index() { return Index.get(); }
  const SExpr *index() const { return Index.get(); }

  template <class V>
  typename V::R_SExpr traverse(V &Vs, typename V::R_Ctx Ctx) {
    auto Na = Vs.traverse(Array, Vs.subExprCtx(Ctx));
    auto Ni = Vs.traverse(Index, Vs.subExprCtx(Ctx));
    return Vs.reduceArrayIndex(*this, Na, Ni);
  }

  template <class C>
  typename C::CType compare(const ArrayIndex* E, C& Cmp) const {
    typename C::CType Ct = Cmp.compare(array(), E->array());
    if (Cmp.notTrue(Ct))
      return Ct;
    return Cmp.compare(index(), E->index());
  }

private:
  SExprRef Array;
  SExprRef Index;
};


// Pointer arithmetic, restricted to arrays only.
// If p is a reference to an array, then p + n, where n is an integer, is
// a reference to a subarray.
class ArrayAdd : public SExpr {
public:
  static bool classof(const SExpr *E) { return E->opcode() == COP_ArrayAdd; }

  ArrayAdd(SExpr *A, SExpr *N) : SExpr(COP_ArrayAdd), Array(A), Index(N) {}
  ArrayAdd(const ArrayAdd &E, SExpr *A, SExpr *N)
    : SExpr(E), Array(A), Index(N) {}

  SExpr *array() { return Array.get(); }
  const SExpr *array() const { return Array.get(); }

  SExpr *index() { return Index.get(); }
  const SExpr *index() const { return Index.get(); }

  template <class V>
  typename V::R_SExpr traverse(V &Vs, typename V::R_Ctx Ctx) {
    auto Na = Vs.traverse(Array, Vs.subExprCtx(Ctx));
    auto Ni = Vs.traverse(Index, Vs.subExprCtx(Ctx));
    return Vs.reduceArrayAdd(*this, Na, Ni);
  }

  template <class C>
  typename C::CType compare(const ArrayAdd* E, C& Cmp) const {
    typename C::CType Ct = Cmp.compare(array(), E->array());
    if (Cmp.notTrue(Ct))
      return Ct;
    return Cmp.compare(index(), E->index());
  }

private:
  SExprRef Array;
  SExprRef Index;
};


// Simple unary operation -- e.g. !, ~, etc.
class UnaryOp : public SExpr {
public:
  static bool classof(const SExpr *E) { return E->opcode() == COP_UnaryOp; }

  UnaryOp(TIL_UnaryOpcode Op, SExpr *E) : SExpr(COP_UnaryOp), Expr0(E) {
    Flags = Op;
  }
  UnaryOp(const UnaryOp &U, SExpr *E) : SExpr(U), Expr0(E) { Flags = U.Flags; }

  TIL_UnaryOpcode unaryOpcode() const {
    return static_cast<TIL_UnaryOpcode>(Flags);
  }

  SExpr *expr() { return Expr0.get(); }
  const SExpr *expr() const { return Expr0.get(); }

  template <class V>
  typename V::R_SExpr traverse(V &Vs, typename V::R_Ctx Ctx) {
    auto Ne = Vs.traverse(Expr0, Vs.subExprCtx(Ctx));
    return Vs.reduceUnaryOp(*this, Ne);
  }

  template <class C>
  typename C::CType compare(const UnaryOp* E, C& Cmp) const {
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

  TIL_BinaryOpcode binaryOpcode() const {
    return static_cast<TIL_BinaryOpcode>(Flags);
  }

  SExpr *expr0() { return Expr0.get(); }
  const SExpr *expr0() const { return Expr0.get(); }

  SExpr *expr1() { return Expr1.get(); }
  const SExpr *expr1() const { return Expr1.get(); }

  template <class V>
  typename V::R_SExpr traverse(V &Vs, typename V::R_Ctx Ctx) {
    auto Ne0 = Vs.traverse(Expr0, Vs.subExprCtx(Ctx));
    auto Ne1 = Vs.traverse(Expr1, Vs.subExprCtx(Ctx));
    return Vs.reduceBinaryOp(*this, Ne0, Ne1);
  }

  template <class C>
  typename C::CType compare(const BinaryOp* E, C& Cmp) const {
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

  TIL_CastOpcode castOpcode() const {
    return static_cast<TIL_CastOpcode>(Flags);
  }

  SExpr *expr() { return Expr0.get(); }
  const SExpr *expr() const { return Expr0.get(); }

  template <class V>
  typename V::R_SExpr traverse(V &Vs, typename V::R_Ctx Ctx) {
    auto Ne = Vs.traverse(Expr0, Vs.subExprCtx(Ctx));
    return Vs.reduceCast(*this, Ne);
  }

  template <class C>
  typename C::CType compare(const Cast* E, C& Cmp) const {
    typename C::CType Ct =
      Cmp.compareIntegers(castOpcode(), E->castOpcode());
    if (Cmp.notTrue(Ct))
      return Ct;
    return Cmp.compare(expr(), E->expr());
  }

private:
  SExprRef Expr0;
};


class SCFG;


class Phi : public SExpr {
public:
  // TODO: change to SExprRef
  typedef SimpleArray<SExpr *> ValArray;

  // In minimal SSA form, all Phi nodes are MultiVal.
  // During conversion to SSA, incomplete Phi nodes may be introduced, which
  // are later determined to be SingleVal, and are thus redundant.
  enum Status {
    PH_MultiVal = 0, // Phi node has multiple distinct values.  (Normal)
    PH_SingleVal,    // Phi node has one distinct value, and can be eliminated
    PH_Incomplete    // Phi node is incomplete
  };

  static bool classof(const SExpr *E) { return E->opcode() == COP_Phi; }

  Phi() : SExpr(COP_Phi) {}
  Phi(MemRegionRef A, unsigned Nvals) : SExpr(COP_Phi), Values(A, Nvals) {}
  Phi(const Phi &P, ValArray &&Vs)    : SExpr(P), Values(std::move(Vs)) {}

  const ValArray &values() const { return Values; }
  ValArray &values() { return Values; }

  Status status() const { return static_cast<Status>(Flags); }
  void setStatus(Status s) { Flags = s; }

  template <class V>
  typename V::R_SExpr traverse(V &Vs, typename V::R_Ctx Ctx) {
    typename V::template Container<typename V::R_SExpr>
      Nvs(Vs, Values.size());

    for (auto *Val : Values) {
      Nvs.push_back( Vs.traverse(Val, Vs.subExprCtx(Ctx)) );
    }
    return Vs.reducePhi(*this, Nvs);
  }

  template <class C>
  typename C::CType compare(const Phi *E, C &Cmp) const {
    // TODO: implement CFG comparisons
    return Cmp.comparePointers(this, E);
  }

private:
  ValArray Values;
};


// A basic block is part of an SCFG, and can be treated as a function in
// continuation passing style.  It consists of a sequence of phi nodes, which
// are "arguments" to the function, followed by a sequence of instructions.
// Both arguments and instructions define new variables.  It ends with a
// branch or goto to another basic block in the same SCFG.
class BasicBlock : public SExpr {
public:
  typedef SimpleArray<Variable*>   VarArray;
  typedef SimpleArray<BasicBlock*> BlockArray;

  static bool classof(const SExpr *E) { return E->opcode() == COP_BasicBlock; }

  explicit BasicBlock(MemRegionRef A, BasicBlock* P = nullptr)
      : SExpr(COP_BasicBlock), Arena(A), CFGPtr(nullptr), BlockID(0),
        Parent(P), Terminator(nullptr)
  { }
  BasicBlock(BasicBlock &B, VarArray &&As, VarArray &&Is, SExpr *T)
      : SExpr(COP_BasicBlock), Arena(B.Arena), CFGPtr(nullptr), BlockID(0),
        Parent(nullptr), Args(std::move(As)), Instrs(std::move(Is)),
        Terminator(T)
  { }

  unsigned blockID() const { return BlockID; }
  unsigned numPredecessors() const { return Predecessors.size(); }

  const SCFG* cfg() const { return CFGPtr; }
  SCFG* cfg() { return CFGPtr; }

  const BasicBlock *parent() const { return Parent; }
  BasicBlock *parent() { return Parent; }

  const VarArray &arguments() const { return Args; }
  VarArray &arguments() { return Args; }

  const VarArray &instructions() const { return Instrs; }
  VarArray &instructions() { return Instrs; }

  const BlockArray &predecessors() const { return Predecessors; }
  BlockArray &predecessors() { return Predecessors; }

  const SExpr *terminator() const { return Terminator.get(); }
  SExpr *terminator() { return Terminator.get(); }

  void setBlockID(unsigned i)   { BlockID = i; }
  void setParent(BasicBlock *P) { Parent = P;  }
  void setTerminator(SExpr *E)  { Terminator.reset(E); }

  // Add a new argument.  V must define a phi-node.
  void addArgument(Variable *V) {
    V->setKind(Variable::VK_LetBB);
    Args.reserveCheck(1, Arena);
    Args.push_back(V);
  }
  // Add a new instruction.
  void addInstruction(Variable *V) {
    V->setKind(Variable::VK_LetBB);
    Instrs.reserveCheck(1, Arena);
    Instrs.push_back(V);
  }
  // Add a new predecessor, and return the phi-node index for it.
  // Will add an argument to all phi-nodes, initialized to nullptr.
  unsigned addPredecessor(BasicBlock *Pred);

  // Reserve space for Nargs arguments.
  void reserveArguments(unsigned Nargs)   { Args.reserve(Nargs, Arena); }

  // Reserve space for Nins instructions.
  void reserveInstructions(unsigned Nins) { Instrs.reserve(Nins, Arena); }

  // Reserve space for NumPreds predecessors, including space in phi nodes.
  void reservePredecessors(unsigned NumPreds);

  // Return the index of BB, or Predecessors.size if BB is not a predecessor.
  unsigned findPredecessorIndex(const BasicBlock *BB) const {
    auto I = std::find(Predecessors.cbegin(), Predecessors.cend(), BB);
    return std::distance(Predecessors.cbegin(), I);
  }

  // Set id numbers for variables.
  void renumberVars();

  template <class V>
  typename V::R_BasicBlock traverse(V &Vs, typename V::R_Ctx Ctx) {
    typename V::template Container<Variable*> Nas(Vs, Args.size());
    typename V::template Container<Variable*> Nis(Vs, Instrs.size());

    // Entering the basic block should do any scope initialization.
    Vs.enterBasicBlock(*this);

    for (auto *A : Args) {
      auto Ne = Vs.traverse(A->Definition, Vs.subExprCtx(Ctx));
      Variable *Nvd = Vs.enterScope(*A, Ne);
      Nas.push_back(Nvd);
    }
    for (auto *I : Instrs) {
      auto Ne = Vs.traverse(I->Definition, Vs.subExprCtx(Ctx));
      Variable *Nvd = Vs.enterScope(*I, Ne);
      Nis.push_back(Nvd);
    }
    auto Nt = Vs.traverse(Terminator, Ctx);

    // Exiting the basic block should handle any scope cleanup.
    Vs.exitBasicBlock(*this);

    return Vs.reduceBasicBlock(*this, Nas, Nis, Nt);
  }

  template <class C>
  typename C::CType compare(const BasicBlock *E, C &Cmp) const {
    // TODO: implement CFG comparisons
    return Cmp.comparePointers(this, E);
  }

private:
  friend class SCFG;

  MemRegionRef Arena;

  SCFG       *CFGPtr;       // The CFG that contains this block.
  unsigned   BlockID;       // unique id for this BB in the containing CFG
  BasicBlock *Parent;       // The parent block is the enclosing lexical scope.
                            // The parent dominates this block.
  BlockArray Predecessors;  // Predecessor blocks in the CFG.
  VarArray   Args;          // Phi nodes.  One argument per predecessor.
  VarArray   Instrs;        // Instructions.
  SExprRef   Terminator;    // Branch or Goto
};


// An SCFG is a control-flow graph.  It consists of a set of basic blocks, each
// of which terminates in a branch to another basic block.  There is one
// entry point, and one exit point.
class SCFG : public SExpr {
public:
  typedef SimpleArray<BasicBlock *> BlockArray;
  typedef BlockArray::iterator iterator;
  typedef BlockArray::const_iterator const_iterator;

  static bool classof(const SExpr *E) { return E->opcode() == COP_SCFG; }

  SCFG(MemRegionRef A, unsigned Nblocks)
    : SExpr(COP_SCFG), Arena(A), Blocks(A, Nblocks),
      Entry(nullptr), Exit(nullptr) {
    Entry = new (A) BasicBlock(A, nullptr);
    Exit  = new (A) BasicBlock(A, Entry);
    auto *V = new (A) Variable(new (A) Phi());
    Exit->addArgument(V);
    add(Entry);
    add(Exit);
  }
  SCFG(const SCFG &Cfg, BlockArray &&Ba) // steals memory from Ba
      : SExpr(COP_SCFG), Arena(Cfg.Arena), Blocks(std::move(Ba)),
        Entry(nullptr), Exit(nullptr) {
    // TODO: set entry and exit!
  }

  iterator begin() { return Blocks.begin(); }
  iterator end() { return Blocks.end(); }

  const_iterator begin() const { return cbegin(); }
  const_iterator end() const { return cend(); }

  const_iterator cbegin() const { return Blocks.cbegin(); }
  const_iterator cend() const { return Blocks.cend(); }

  const BasicBlock *entry() const { return Entry; }
  BasicBlock *entry() { return Entry; }
  const BasicBlock *exit() const { return Exit; }
  BasicBlock *exit() { return Exit; }

  inline void add(BasicBlock *BB) {
    assert(BB->CFGPtr == nullptr || BB->CFGPtr == this);
    BB->setBlockID(Blocks.size());
    BB->CFGPtr = this;
    Blocks.reserveCheck(1, Arena);
    Blocks.push_back(BB);
  }

  void setEntry(BasicBlock *BB) { Entry = BB; }
  void setExit(BasicBlock *BB)  { Exit = BB;  }

  // Set varable ids in all blocks.
  void renumberVars();

  template <class V>
  typename V::R_SExpr traverse(V &Vs, typename V::R_Ctx Ctx) {
    Vs.enterCFG(*this);
    typename V::template Container<BasicBlock *> Bbs(Vs, Blocks.size());
    for (auto *B : Blocks) {
      Bbs.push_back( B->traverse(Vs, Vs.subExprCtx(Ctx)) );
    }
    Vs.exitCFG(*this);
    return Vs.reduceSCFG(*this, Bbs);
  }

  template <class C>
  typename C::CType compare(const SCFG *E, C &Cmp) const {
    // TODO -- implement CFG comparisons
    return Cmp.comparePointers(this, E);
  }

private:
  MemRegionRef Arena;
  BlockArray   Blocks;
  BasicBlock   *Entry;
  BasicBlock   *Exit;
};


class Goto : public SExpr {
public:
  static bool classof(const SExpr *E) { return E->opcode() == COP_Goto; }

  Goto(BasicBlock *B, unsigned I)
      : SExpr(COP_Goto), TargetBlock(B), Index(I) {}
  Goto(const Goto &G, BasicBlock *B, unsigned I)
      : SExpr(COP_Goto), TargetBlock(B), Index(I) {}

  const BasicBlock *targetBlock() const { return TargetBlock; }
  BasicBlock *targetBlock() { return TargetBlock; }

  unsigned index() const { return Index; }

  template <class V>
  typename V::R_SExpr traverse(V &Vs, typename V::R_Ctx Ctx) {
    BasicBlock *Ntb = Vs.reduceBasicBlockRef(TargetBlock);
    return Vs.reduceGoto(*this, Ntb);
  }

  template <class C>
  typename C::CType compare(const Goto *E, C &Cmp) const {
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

  Branch(SExpr *C, BasicBlock *T, BasicBlock *E, unsigned TI, unsigned EI)
      : SExpr(COP_Branch), Condition(C), ThenBlock(T), ElseBlock(E),
        ThenIndex(TI), ElseIndex(EI)
  {}
  Branch(const Branch &Br, SExpr *C, BasicBlock *T, BasicBlock *E,
         unsigned TI, unsigned EI)
      : SExpr(COP_Branch), Condition(C), ThenBlock(T), ElseBlock(E),
        ThenIndex(TI), ElseIndex(EI)
  {}

  const SExpr *condition() const { return Condition; }
  SExpr *condition() { return Condition; }

  const BasicBlock *thenBlock() const { return ThenBlock; }
  BasicBlock *thenBlock() { return ThenBlock; }

  const BasicBlock *elseBlock() const { return ElseBlock; }
  BasicBlock *elseBlock() { return ElseBlock; }

  unsigned thenIndex() const { return ThenIndex; }
  unsigned elseIndex() const { return ElseIndex; }

  template <class V>
  typename V::R_SExpr traverse(V &Vs, typename V::R_Ctx Ctx) {
    auto Nc = Vs.traverse(Condition, Vs.subExprCtx(Ctx));
    BasicBlock *Ntb = Vs.reduceBasicBlockRef(ThenBlock);
    BasicBlock *Nte = Vs.reduceBasicBlockRef(ElseBlock);
    return Vs.reduceBranch(*this, Nc, Ntb, Nte);
  }

  template <class C>
  typename C::CType compare(const Branch *E, C &Cmp) const {
    // TODO -- implement CFG comparisons
    return Cmp.comparePointers(this, E);
  }

private:
  SExpr *Condition;
  BasicBlock *ThenBlock;
  BasicBlock *ElseBlock;
  unsigned ThenIndex;
  unsigned ElseIndex;
};


// An identifier, e.g. 'foo' or 'x'.
// This is a pseduo-term; it will be lowered to a variable or projection.
class Identifier : public SExpr {
public:
  static bool classof(const SExpr *E) { return E->opcode() == COP_Identifier; }

  Identifier(StringRef Id): SExpr(COP_Identifier), Name(Id) { }
  Identifier(const Identifier& I) : SExpr(I), Name(I.Name)  { }

  StringRef name() const { return Name; }

  template <class V>
  typename V::R_SExpr traverse(V &Vs, typename V::R_Ctx Ctx) {
    return Vs.reduceIdentifier(*this);
  }

  template <class C>
  typename C::CType compare(const Identifier* E, C& Cmp) const {
    return Cmp.compareStrings(name(), E->name());
  }

private:
  StringRef Name;
};


// An if-then-else expression.
// This is a pseduo-term; it will be lowered to a branch in a CFG.
class IfThenElse : public SExpr {
public:
  static bool classof(const SExpr *E) { return E->opcode() == COP_IfThenElse; }

  IfThenElse(SExpr *C, SExpr *T, SExpr *E)
    : SExpr(COP_IfThenElse), Condition(C), ThenExpr(T), ElseExpr(E)
  { }
  IfThenElse(const IfThenElse &I, SExpr *C, SExpr *T, SExpr *E)
    : SExpr(I), Condition(C), ThenExpr(T), ElseExpr(E)
  { }

  SExpr *condition() { return Condition.get(); }   // Address to store to
  const SExpr *condition() const { return Condition.get(); }

  SExpr *thenExpr() { return ThenExpr.get(); }     // Value to store
  const SExpr *thenExpr() const { return ThenExpr.get(); }

  SExpr *elseExpr() { return ElseExpr.get(); }     // Value to store
  const SExpr *elseExpr() const { return ElseExpr.get(); }

  template <class V>
  typename V::R_SExpr traverse(V &Vs, typename V::R_Ctx Ctx) {
    auto Nc = Vs.traverse(Condition, Vs.subExprCtx(Ctx));
    auto Nt = Vs.traverse(ThenExpr,  Vs.subExprCtx(Ctx));
    auto Ne = Vs.traverse(ElseExpr,  Vs.subExprCtx(Ctx));
    return Vs.reduceIfThenElse(*this, Nc, Nt, Ne);
  }

  template <class C>
  typename C::CType compare(const IfThenElse* E, C& Cmp) const {
    typename C::CType Ct = Cmp.compare(condition(), E->condition());
    if (Cmp.notTrue(Ct))
      return Ct;
    Ct = Cmp.compare(thenExpr(), E->thenExpr());
    if (Cmp.notTrue(Ct))
      return Ct;
    return Cmp.compare(elseExpr(), E->elseExpr());
  }

private:
  SExprRef Condition;
  SExprRef ThenExpr;
  SExprRef ElseExpr;
};


// A let-expression,  e.g.  let x=t; u.
// This is a pseduo-term; it will be lowered to instructions in a CFG.
class Let : public SExpr {
public:
  static bool classof(const SExpr *E) { return E->opcode() == COP_Let; }

  Let(Variable *Vd, SExpr *Bd) : SExpr(COP_Let), VarDecl(Vd), Body(Bd) {
    Vd->setKind(Variable::VK_Let);
  }
  Let(const Let &L, Variable *Vd, SExpr *Bd) : SExpr(L), VarDecl(Vd), Body(Bd) {
    Vd->setKind(Variable::VK_Let);
  }

  Variable *variableDecl()  { return VarDecl; }
  const Variable *variableDecl() const { return VarDecl; }

  SExpr *body() { return Body.get(); }
  const SExpr *body() const { return Body.get(); }

  template <class V>
  typename V::R_SExpr traverse(V &Vs, typename V::R_Ctx Ctx) {
    // This is a variable declaration, so traverse the definition.
    auto E0 = Vs.traverse(VarDecl->Definition, Vs.subExprCtx(Ctx));
    // Tell the rewriter to enter the scope of the let variable.
    Variable *Nvd = Vs.enterScope(*VarDecl, E0);
    auto E1 = Vs.traverse(Body, Ctx);
    Vs.exitScope(*VarDecl);
    return Vs.reduceLet(*this, Nvd, E1);
  }

  template <class C>
  typename C::CType compare(const Let* E, C& Cmp) const {
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



const SExpr *getCanonicalVal(const SExpr *E);
SExpr* simplifyToCanonicalVal(SExpr *E);
void simplifyIncompleteArg(Variable *V, til::Phi *Ph);


} // end namespace til
} // end namespace threadSafety
} // end namespace clang

#endif // LLVM_CLANG_THREAD_SAFETY_TIL_H
