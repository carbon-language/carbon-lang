// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef FORTRAN_FIR_STATEMENTS_H_
#define FORTRAN_FIR_STATEMENTS_H_

#include "basicblock.h"
#include <initializer_list>

namespace Fortran::FIR {

class ReturnStmt;
class BranchStmt;
class SwitchStmt;
class SwitchCaseStmt;
class SwitchTypeStmt;
class SwitchRankStmt;
class IndirectBranchStmt;
class UnreachableStmt;
class ApplyExprStmt;
class LocateExprStmt;
class AllocateInsn;
class DeallocateInsn;
class AllocateLocalInsn;
class LoadInsn;
class StoreInsn;
class DisassociateInsn;
class CallStmt;
class RuntimeStmt;
class IORuntimeStmt;
class ScopeEnterStmt;
class ScopeExitStmt;
class PHIStmt;

class Statement;

CLASS_TRAIT(StatementTrait)
CLASS_TRAIT(TerminatorTrait)
CLASS_TRAIT(ActionTrait)

class Stmt_impl {
public:
  using StatementTrait = std::true_type;
};

// Some uses of a Statement should be constrained.  These contraints are imposed
// at compile time.
template<typename A> class QualifiedStmt {
public:
  template<typename T, typename U,
      std::enable_if_t<std::is_base_of_v<T, U>, int>>
  friend QualifiedStmt<T> QualifiedStmtCreate(Statement *s);

  QualifiedStmt() = delete;

  // create a stub, where stmt == nullptr
  QualifiedStmt(std::nullptr_t) : stmt{nullptr} {}
  operator Statement *() const { return stmt; }
  operator bool() const { return stmt; }
  operator A *() const;

  Statement *stmt;

private:
  QualifiedStmt(Statement *stmt) : stmt{stmt} {}
};

template<typename A, typename B,
    std::enable_if_t<std::is_base_of_v<A, B>, int> = 0>
QualifiedStmt<A> QualifiedStmtCreate(Statement *s) {
  return QualifiedStmt<A>{s};
}

// Every basic block must end in a terminator
class TerminatorStmt_impl : virtual public Stmt_impl {
public:
  virtual std::list<BasicBlock *> succ_blocks() const = 0;
  virtual ~TerminatorStmt_impl();
  using TerminatorTrait = std::true_type;
};

// Transfer control out of the current procedure
class ReturnStmt : public TerminatorStmt_impl {
public:
  static ReturnStmt Create(QualifiedStmt<ApplyExprStmt> stmt) {
    return ReturnStmt{stmt};
  }
  static ReturnStmt Create() { return ReturnStmt{}; }
  std::list<BasicBlock *> succ_blocks() const override { return {}; }
  bool has_value() const { return value_; }
  Statement *value() const { return value_; }

private:
  QualifiedStmt<ApplyExprStmt> value_;
  explicit ReturnStmt(QualifiedStmt<ApplyExprStmt> exp);
  explicit ReturnStmt();
};

// Encodes two-way conditional branch and one-way absolute branch
class BranchStmt : public TerminatorStmt_impl {
public:
  static BranchStmt Create(
      Value condition, BasicBlock *trueBlock, BasicBlock *falseBlock) {
    return BranchStmt{condition, trueBlock, falseBlock};
  }
  static BranchStmt Create(BasicBlock *succ) {
    return BranchStmt{std::nullopt, succ, nullptr};
  }
  bool hasCondition() const { return condition_.has_value(); }
  Value getCond() const { return condition_.value(); }
  std::list<BasicBlock *> succ_blocks() const override {
    if (hasCondition()) {
      return {succs_[TrueIndex], succs_[FalseIndex]};
    }
    return {succs_[TrueIndex]};
  }
  BasicBlock *getTrueSucc() const { return succs_[TrueIndex]; }
  BasicBlock *getFalseSucc() const { return succs_[FalseIndex]; }

private:
  explicit BranchStmt(const std::optional<Value> &condition,
      BasicBlock *trueBlock, BasicBlock *falseBlock);
  static constexpr int TrueIndex{0};
  static constexpr int FalseIndex{1};
  std::optional<Value> condition_;
  BasicBlock *succs_[2];
};

// Switch on an expression into a set of constant values
class SwitchStmt : public TerminatorStmt_impl {
public:
  using ValueType = Value;
  using ValueSuccPairType = std::pair<ValueType, BasicBlock *>;
  using ValueSuccPairListType = std::vector<ValueSuccPairType>;
  static SwitchStmt Create(
      const Value &switchEval, const ValueSuccPairListType &args) {
    return SwitchStmt{switchEval, args};
  }
  BasicBlock *defaultSucc() const;
  std::list<BasicBlock *> succ_blocks() const override;
  Value getCond() const { return condition_; }

private:
  explicit SwitchStmt(
      const Value &condition, const ValueSuccPairListType &args);

  Value condition_;
  ValueSuccPairListType valueSuccPairs_;
};

// Switch on an expression into a set of value (open or closed) ranges
class SwitchCaseStmt : public TerminatorStmt_impl {
public:
  struct Default {};
  struct Exactly {  // selector == v
    ApplyExprStmt *v;
  };
  struct InclusiveAbove {  // v <= selector
    ApplyExprStmt *v;
  };
  struct InclusiveBelow {  // selector <= v
    ApplyExprStmt *v;
  };
  struct InclusiveRange {  // lower <= selector <= upper
    ApplyExprStmt *lower;
    ApplyExprStmt *upper;
  };
  using RangeAlternative =
      std::variant<Exactly, InclusiveAbove, InclusiveBelow, InclusiveRange>;
  using ValueType = std::variant<Default, std::vector<RangeAlternative>>;
  using ValueSuccPairType = std::pair<ValueType, BasicBlock *>;
  using ValueSuccPairListType = std::vector<ValueSuccPairType>;

  static SwitchCaseStmt Create(
      Value switchEval, const ValueSuccPairListType &args) {
    return SwitchCaseStmt{switchEval, args};
  }
  BasicBlock *defaultSucc() const;
  std::list<BasicBlock *> succ_blocks() const override;
  Value getCond() const { return condition_; }

private:
  explicit SwitchCaseStmt(Value condition, const ValueSuccPairListType &args);

  Value condition_;
  ValueSuccPairListType valueSuccPairs_;
};

// Switch on the TYPE of the selector into a set of TYPES, etc.
class SwitchTypeStmt : public TerminatorStmt_impl {
public:
  struct Default {};
  struct TypeSpec {
    Type v;
  };
  struct DerivedTypeSpec {
    Type v;
  };
  using ValueType = std::variant<Default, TypeSpec, DerivedTypeSpec>;
  using ValueSuccPairType = std::pair<ValueType, BasicBlock *>;
  using ValueSuccPairListType = std::vector<ValueSuccPairType>;
  static SwitchTypeStmt Create(
      Value switchEval, const ValueSuccPairListType &args) {
    return SwitchTypeStmt{switchEval, args};
  }
  BasicBlock *defaultSucc() const;
  std::list<BasicBlock *> succ_blocks() const override;
  Value getCond() const { return condition_; }

private:
  explicit SwitchTypeStmt(Value condition, const ValueSuccPairListType &args);
  Value condition_;
  ValueSuccPairListType valueSuccPairs_;
};

// Switch on the RANK of the selector into a set of constant integers, etc.
class SwitchRankStmt : public TerminatorStmt_impl {
public:
  struct Default {};  // RANK DEFAULT
  struct AssumedSize {};  // RANK(*)
  struct Exactly {  // RANK(n)
    Expression *v;
  };
  using ValueType = std::variant<Exactly, AssumedSize, Default>;
  using ValueSuccPairType = std::pair<ValueType, BasicBlock *>;
  using ValueSuccPairListType = std::vector<ValueSuccPairType>;
  static SwitchRankStmt Create(
      Value switchEval, const ValueSuccPairListType &args) {
    return SwitchRankStmt{switchEval, args};
  }
  BasicBlock *defaultSucc() const;
  std::list<BasicBlock *> succ_blocks() const override;
  Value getCond() const { return condition_; }

private:
  explicit SwitchRankStmt(Value condition, const ValueSuccPairListType &args);

  Value condition_;
  ValueSuccPairListType valueSuccPairs_;
};

class IndirectBranchStmt : public TerminatorStmt_impl {
public:
  using TargetListType = std::vector<BasicBlock *>;
  static IndirectBranchStmt Create(
      Variable *variable, const TargetListType &potentialTargets) {
    return IndirectBranchStmt{variable, potentialTargets};
  }

  Variable *variable() const { return variable_; }
  std::list<BasicBlock *> succ_blocks() const override {
    return {potentialTargets_.begin(), potentialTargets_.end()};
  }

private:
  explicit IndirectBranchStmt(
      Variable *variable, const TargetListType &potentialTargets)
    : variable_{variable}, potentialTargets_{potentialTargets} {}
  Variable *variable_;
  TargetListType potentialTargets_;
};

// This statement is not reachable
class UnreachableStmt : public TerminatorStmt_impl {
public:
  static UnreachableStmt Create() { return UnreachableStmt{}; }
  std::list<BasicBlock *> succ_blocks() const override { return {}; }

private:
  explicit UnreachableStmt() = default;
};

class ActionStmt_impl : virtual public Stmt_impl {
public:
  using ActionTrait = std::true_type;

protected:
  ActionStmt_impl() : type{std::nullopt} {}

  // TODO: DynamicType is a placeholder for now
  std::optional<evaluate::DynamicType> type;
};

// Compute the value of an expression
class ApplyExprStmt : public ActionStmt_impl {
public:
  static ApplyExprStmt Create(const Expression *e) { return ApplyExprStmt{*e}; }
  static ApplyExprStmt Create(Expression &&e) {
    return ApplyExprStmt{std::move(e)};
  }

  Expression expression() const { return expression_; }

private:
  explicit ApplyExprStmt(const Expression &e) : expression_{e} {}
  explicit ApplyExprStmt(Expression &&e) : expression_{std::move(e)} {}

  Expression expression_;
};

// Base class of all addressable statements
class Addressable_impl : public ActionStmt_impl {
public:
  Expression address() const { return addrExpr_.value(); }

protected:
  Addressable_impl() : addrExpr_{std::nullopt} {}
  explicit Addressable_impl(const Expression &ae) : addrExpr_{ae} {}
  std::optional<Expression> addrExpr_;
};

// Compute the location of an expression
class LocateExprStmt : public Addressable_impl {
public:
  static LocateExprStmt Create(const Expression *e) {
    return LocateExprStmt(*e);
  }
  static LocateExprStmt Create(Expression &&e) { return LocateExprStmt(e); }

  const Expression &expression() const { return addrExpr_.value(); }

private:
  explicit LocateExprStmt(const Expression &e) : Addressable_impl{e} {}
};

// has memory effect
class MemoryStmt_impl : public ActionStmt_impl {
protected:
  MemoryStmt_impl() {}
};

// Allocate storage (per ALLOCATE)
class AllocateInsn : public Addressable_impl, public MemoryStmt_impl {
public:
  static AllocateInsn Create(Type type, int alignment = 0) {
    return AllocateInsn{type, alignment};
  }

  Type type() const { return type_; }
  int alignment() const { return alignment_; }

private:
  explicit AllocateInsn(Type type, int alignment)
    : type_{type}, alignment_{alignment} {}

  Type type_;
  int alignment_;
};

// Deallocate storage (per DEALLOCATE)
class DeallocateInsn : public MemoryStmt_impl {
public:
  static DeallocateInsn Create(QualifiedStmt<AllocateInsn> alloc) {
    return DeallocateInsn{alloc};
  }

  Statement *alloc() const { return alloc_; }

private:
  explicit DeallocateInsn(QualifiedStmt<AllocateInsn> alloc) : alloc_{alloc} {}
  QualifiedStmt<AllocateInsn> alloc_;
};

// Allocate space for a temporary by its Type. The lifetime of the temporary
// will not exceed that of the containing Procedure.
class AllocateLocalInsn : public Addressable_impl, public MemoryStmt_impl {
public:
  static AllocateLocalInsn Create(
      Type type, const Expression &expr, int alignment = 0) {
    return AllocateLocalInsn{type, alignment, expr};
  }

  Type type() const { return type_; }
  int alignment() const { return alignment_; }
  Expression variable() const { return addrExpr_.value(); }

private:
  explicit AllocateLocalInsn(Type type, int alignment, const Expression &expr)
    : Addressable_impl{expr}, type_{type}, alignment_{alignment} {}

  Type type_;
  int alignment_;
};

// Load value(s) from a location
class LoadInsn : public MemoryStmt_impl {
public:
  static LoadInsn Create(const Value &addr) { return LoadInsn{addr}; }
  static LoadInsn Create(Value &&addr) { return LoadInsn{addr}; }
  static LoadInsn Create(Statement *addr) { return LoadInsn{addr}; }

  Value address() const { return address_; }

private:
  explicit LoadInsn(const Value &addr);
  explicit LoadInsn(Value &&addr);
  explicit LoadInsn(Statement *addr);
  Value address_;
};

// Store value(s) from an applied expression to a location
class StoreInsn : public MemoryStmt_impl {
public:
  static StoreInsn Create(QualifiedStmt<Addressable_impl> addr, Value value) {
    return StoreInsn{addr, value};
  }

  Addressable_impl *address() const { return address_; }
  Value value() const { return value_; }

private:
  explicit StoreInsn(QualifiedStmt<Addressable_impl> addr, Value val);

  QualifiedStmt<Addressable_impl> address_;
  Value value_;
};

// NULLIFY - make pointer object disassociated
class DisassociateInsn : public ActionStmt_impl {
public:
  static DisassociateInsn Create(Statement *s) { return DisassociateInsn{s}; }

  Statement *disassociate() { return disassociate_; }

private:
  DisassociateInsn(Statement *s) : disassociate_{s} {}
  Statement *disassociate_;
};

// base class for all call-like IR statements
class CallStmt_impl : public ActionStmt_impl {
public:
  Value Callee() const { return callee_; }
  unsigned NumArgs() const { return arguments_.size(); }

protected:
  CallStmt_impl(
      const FunctionType *functionType, Value callee, CallArguments &&arguments)
    : functionType_{functionType}, callee_{callee}, arguments_{arguments} {}

  const FunctionType *functionType_;
  Value callee_;
  CallArguments arguments_;
};

// CALL statements and function references
// A CallStmt has pass-by-value semantics. Pass-by-reference must be done
// explicitly by passing addresses of objects or temporaries.
class CallStmt : public CallStmt_impl {
public:
  static CallStmt Create(
      const FunctionType *type, Value callee, CallArguments &&arguments) {
    return CallStmt{type, callee, std::move(arguments)};
  }

private:
  explicit CallStmt(
      const FunctionType *functionType, Value callee, CallArguments &&arguments)
    : CallStmt_impl{functionType, callee, std::move(arguments)} {}
};

// Miscellaneous statements that turn into runtime calls
class RuntimeStmt : public CallStmt_impl {
public:
  static RuntimeStmt Create(
      RuntimeCallType call, RuntimeCallArguments &&argument) {
    return RuntimeStmt{call, std::move(argument)};
  }

  RuntimeCallType call() const { return call_; }

private:
  explicit RuntimeStmt(RuntimeCallType call, RuntimeCallArguments &&arguments)
    : CallStmt_impl{nullptr, Procedure::CreateIntrinsicProcedure(call),
          std::move(arguments)},
      call_{call} {}

  RuntimeCallType call_;
};

// The 13 Fortran I/O statements. Will be lowered to whatever becomes of the
// I/O runtime.
class IORuntimeStmt : public CallStmt_impl {
public:
  static IORuntimeStmt Create(
      InputOutputCallType call, IOCallArguments &&arguments) {
    return IORuntimeStmt{call, std::move(arguments)};
  }

  InputOutputCallType call() const { return call_; }

private:
  explicit IORuntimeStmt(InputOutputCallType call, IOCallArguments &&arguments)
    : CallStmt_impl{nullptr, Procedure::CreateIntrinsicProcedure(call),
          std::move(arguments)},
      call_{call} {}

  InputOutputCallType call_;
};

class ScopeStmt_impl : public ActionStmt_impl {
public:
  Scope *GetScope() const { return scope; }

protected:
  ScopeStmt_impl(Scope *scope) : scope{nullptr} {}
  Scope *scope;
};

// From the CFG document
class ScopeEnterStmt : public ScopeStmt_impl {
public:
  static ScopeEnterStmt Create(Scope *scope) { return ScopeEnterStmt{scope}; }

private:
  ScopeEnterStmt(Scope *scope) : ScopeStmt_impl{scope} {}
};

// From the CFG document
class ScopeExitStmt : public ScopeStmt_impl {
public:
  static ScopeExitStmt Create(Scope *scope) { return ScopeExitStmt{scope}; }

private:
  ScopeExitStmt(Scope *scope) : ScopeStmt_impl{scope} {}
};

// From the CFG document to support SSA
class PHIStmt : public ActionStmt_impl {
public:
  static PHIStmt Create(unsigned numReservedValues) {
    return PHIStmt{numReservedValues};
  }

private:
  PHIStmt(unsigned size) : inputs_(size) {}

  std::vector<PHIPair> inputs_;
};

// Sum type over all statement classes
class Statement : public SumTypeMixin<ReturnStmt,  //
                      BranchStmt,  //
                      SwitchStmt,  //
                      SwitchCaseStmt,  //
                      SwitchTypeStmt,  //
                      SwitchRankStmt,  //
                      IndirectBranchStmt,  //
                      UnreachableStmt,  //
                      ApplyExprStmt,  //
                      LocateExprStmt,  //
                      AllocateInsn,  //
                      DeallocateInsn,  //
                      AllocateLocalInsn,  //
                      LoadInsn,  //
                      StoreInsn,  //
                      DisassociateInsn,  //
                      CallStmt,  //
                      RuntimeStmt,  //
                      IORuntimeStmt,  //
                      ScopeEnterStmt,  //
                      ScopeExitStmt,  //
                      PHIStmt  //
                      >,
                  public Value_impl,
                  public ChildMixin<Statement, BasicBlock>,
                  public llvm::ilist_node<Statement> {
public:
  template<typename A> static Statement *Create(BasicBlock *p, A &&t) {
    return new Statement(p, std::forward<A>(t));
  }
  std::string dump() const;  // LLVM expected name

  void Dump(std::ostream &os) const { os << dump(); }

private:
  template<typename A>
  explicit Statement(BasicBlock *p, A &&t)
    : SumTypeMixin{std::forward<A>(t)}, ChildMixin{p} {
    parent->insertBefore(this);
  }
};

template<typename A> inline QualifiedStmt<A>::operator A *() const {
  return reinterpret_cast<A *>(&stmt->u);
}

inline std::list<BasicBlock *> succ_list(BasicBlock &block) {
  if (auto *terminator{block.terminator()}) {
    return reinterpret_cast<const TerminatorStmt_impl *>(&terminator->u)
        ->succ_blocks();
  }
  // CHECK(false && "block does not have terminator");
  return {};
}

inline ApplyExprStmt *GetApplyExpr(Statement *stmt) {
  return std::get_if<ApplyExprStmt>(&stmt->u);
}

inline AllocateLocalInsn *GetLocal(Statement *stmt) {
  return std::get_if<AllocateLocalInsn>(&stmt->u);
}

Addressable_impl *GetAddressable(Statement *stmt);

template<typename A> std::string ToString(const A *a) {
  std::stringstream ss;
  ss << std::hex << reinterpret_cast<std::intptr_t>(a);
  return ss.str();
}
}

#endif  // FORTRAN_FIR_STATEMENTS_H_
