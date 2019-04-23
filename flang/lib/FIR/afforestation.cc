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

#include "afforestation.h"
#include "builder.h"
#include "flattened.h"
#include "mixin.h"
#include "../evaluate/fold.h"
#include "../evaluate/tools.h"
#include "../parser/parse-tree-visitor.h"
#include "../semantics/expression.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/FileSystem.h"

namespace Fortran::FIR {
namespace {
Expression *ExprRef(const parser::Expr &a) {
  CHECK(a.typedExpr);
  CHECK(a.typedExpr->v);
  return &*a.typedExpr->v;
}
Expression *ExprRef(const common::Indirection<parser::Expr> &a) {
  return ExprRef(a.value());
}

template<typename STMTTYPE, typename CT>
const std::optional<parser::Name> &GetSwitchAssociateName(
    const CT *selectConstruct) {
  return std::get<1>(
      std::get<parser::Statement<STMTTYPE>>(selectConstruct->t).statement.t);
}

template<typename CONSTRUCT>
void DumpSwitchWithSelector(
    const CONSTRUCT *construct, char const *const name) {
  /// auto selector{getSelector(construct)};
  DebugChannel() << name << "(";  // << selector.dump()
}
}  // end namespace

template<typename T> struct SwitchArgs {
  Value exp;
  std::vector<T> values;
  std::vector<flat::LabelRef> labels;
};
using SwitchArguments = SwitchArgs<SwitchStmt::ValueType>;
using SwitchCaseArguments = SwitchArgs<SwitchCaseStmt::ValueType>;
using SwitchRankArguments = SwitchArgs<SwitchRankStmt::ValueType>;
using SwitchTypeArguments = SwitchArgs<SwitchTypeStmt::ValueType>;

template<typename T> bool IsDefault(const typename T::ValueType &valueType) {
  return std::holds_alternative<typename T::Default>(valueType);
}

// move the default case to be first
template<typename T>
void cleanupSwitchPairs(std::vector<typename T::ValueType> &values,
    std::vector<flat::LabelRef> &labels) {
  CHECK(values.size() == labels.size());
  for (std::size_t i{1}, len{values.size()}; i < len; ++i) {
    if (IsDefault<T>(values[i])) {
      auto v{values[0]};
      values[0] = values[i];
      values[i] = v;
      auto w{labels[0]};
      labels[0] = labels[i];
      labels[i] = w;
      break;
    }
  }
}

static std::vector<SwitchCaseStmt::ValueType> populateSwitchValues(
    FIRBuilder *builder, const std::list<parser::CaseConstruct::Case> &list) {
  std::vector<SwitchCaseStmt::ValueType> result;
  for (auto &v : list) {
    auto &caseSelector{std::get<parser::CaseSelector>(
        std::get<parser::Statement<parser::CaseStmt>>(v.t).statement.t)};
    if (std::holds_alternative<parser::Default>(caseSelector.u)) {
      result.emplace_back(SwitchCaseStmt::Default{});
    } else {
      std::vector<SwitchCaseStmt::RangeAlternative> valueList;
      for (auto &r :
          std::get<std::list<parser::CaseValueRange>>(caseSelector.u)) {
        std::visit(
            common::visitors{
                [&](const parser::CaseValue &caseValue) {
                  const auto &e{caseValue.thing.thing.value()};
                  auto *app{builder->MakeAsExpr(ExprRef(e))};
                  valueList.emplace_back(SwitchCaseStmt::Exactly{app});
                },
                [&](const parser::CaseValueRange::Range &range) {
                  if (range.lower.has_value()) {
                    if (range.upper.has_value()) {
                      auto *appl{builder->MakeAsExpr(
                          ExprRef(range.lower->thing.thing))};
                      auto *apph{builder->MakeAsExpr(
                          ExprRef(range.upper->thing.thing))};
                      valueList.emplace_back(
                          SwitchCaseStmt::InclusiveRange{appl, apph});
                    } else {
                      auto *app{builder->MakeAsExpr(
                          ExprRef(range.lower->thing.thing))};
                      valueList.emplace_back(
                          SwitchCaseStmt::InclusiveAbove{app});
                    }
                  } else {
                    auto *app{
                        builder->MakeAsExpr(ExprRef(range.upper->thing.thing))};
                    valueList.emplace_back(SwitchCaseStmt::InclusiveBelow{app});
                  }
                },
            },
            r.u);
      }
      result.emplace_back(valueList);
    }
  }
  return result;
}

static std::vector<SwitchRankStmt::ValueType> populateSwitchValues(
    const std::list<parser::SelectRankConstruct::RankCase> &list) {
  std::vector<SwitchRankStmt::ValueType> result;
  for (auto &v : list) {
    auto &rank{std::get<parser::SelectRankCaseStmt::Rank>(
        std::get<parser::Statement<parser::SelectRankCaseStmt>>(v.t)
            .statement.t)};
    std::visit(
        common::visitors{
            [&](const parser::ScalarIntConstantExpr &exp) {
              const auto &e{exp.thing.thing.thing.value()};
              result.emplace_back(SwitchRankStmt::Exactly{ExprRef(e)});
            },
            [&](const parser::Star &) {
              result.emplace_back(SwitchRankStmt::AssumedSize{});
            },
            [&](const parser::Default &) {
              result.emplace_back(SwitchRankStmt::Default{});
            },
        },
        rank.u);
  }
  return result;
}

static std::vector<SwitchTypeStmt::ValueType> populateSwitchValues(
    const std::list<parser::SelectTypeConstruct::TypeCase> &list) {
  std::vector<SwitchTypeStmt::ValueType> result;
  for (auto &v : list) {
    auto &guard{std::get<parser::TypeGuardStmt::Guard>(
        std::get<parser::Statement<parser::TypeGuardStmt>>(v.t).statement.t)};
    std::visit(
        common::visitors{
            [&](const parser::TypeSpec &spec) {
              result.emplace_back(SwitchTypeStmt::TypeSpec{spec.declTypeSpec});
            },
            [&](const parser::DerivedTypeSpec &spec) {
              result.emplace_back(
                  SwitchTypeStmt::DerivedTypeSpec{nullptr /* FIXME */});
            },
            [&](const parser::Default &) {
              result.emplace_back(SwitchTypeStmt::Default{});
            },
        },
        guard.u);
  }
  return result;
}

template<typename T>
const T *FindReadWriteSpecifier(
    const std::list<parser::IoControlSpec> &specifiers) {
  for (const auto &specifier : specifiers) {
    if (auto *result{std::get_if<T>(&specifier.u)}) {
      return result;
    }
  }
  return nullptr;
}

const parser::IoUnit *FindReadWriteIoUnit(
    const std::optional<parser::IoUnit> &ioUnit,
    const std::list<parser::IoControlSpec> &specifiers) {
  if (ioUnit.has_value()) {
    return &ioUnit.value();
  }
  if (const auto *result{FindReadWriteSpecifier<parser::IoUnit>(specifiers)}) {
    return result;
  }
  SEMANTICS_FAILED("no UNIT spec");
  return {};
}

const parser::Format *FindReadWriteFormat(
    const std::optional<parser::Format> &format,
    const std::list<parser::IoControlSpec> &specifiers) {
  if (format.has_value()) {
    return &format.value();
  }
  return FindReadWriteSpecifier<parser::Format>(specifiers);
}

static Expression AlwaysTrueExpression() {
  using A = evaluate::Type<evaluate::TypeCategory::Logical, 1>;
  return {evaluate::AsGenericExpr(evaluate::Constant<A>{true})};
}

// create an integer constant as an expression
static Expression CreateConstant(int64_t value) {
  using A = evaluate::SubscriptInteger;
  return {evaluate::AsGenericExpr(evaluate::Constant<A>{value})};
}

static void CreateSwitchHelper(FIRBuilder *builder, Value condition,
    const SwitchStmt::ValueSuccPairListType &rest) {
  builder->CreateSwitch(condition, rest);
}
static void CreateSwitchCaseHelper(FIRBuilder *builder, Value condition,
    const SwitchCaseStmt::ValueSuccPairListType &rest) {
  builder->CreateSwitchCase(condition, rest);
}
static void CreateSwitchRankHelper(FIRBuilder *builder, Value condition,
    const SwitchRankStmt::ValueSuccPairListType &rest) {
  builder->CreateSwitchRank(condition, rest);
}
static void CreateSwitchTypeHelper(FIRBuilder *builder, Value condition,
    const SwitchTypeStmt::ValueSuccPairListType &rest) {
  builder->CreateSwitchType(condition, rest);
}

static Expression getApplyExpr(Statement *s) {
  return GetApplyExpr(s)->expression();
}
static Expression getLocalVariable(Statement *s) {
  return GetLocal(s)->variable();
}

// create a new temporary name (as heap garbage)
static parser::CharBlock NewTemporaryName() {
  constexpr int SizeMagicValue{32};
  static int counter;
  char cache[SizeMagicValue];
  int bytesWritten{snprintf(cache, SizeMagicValue, ".t%d", counter++)};
  CHECK(bytesWritten < SizeMagicValue);
  auto len{strlen(cache)};
  char *name{new char[len]};  // XXX: add these to a pool?
  memcpy(name, cache, len);
  return {name, name + len};
}

static TypeRep GetDefaultIntegerType(semantics::SemanticsContext &c) {
  evaluate::ExpressionAnalyzer analyzer{c};
  return c.MakeNumericType(common::TypeCategory::Integer,
      analyzer.GetDefaultKind(common::TypeCategory::Integer));
}

/*static*/ TypeRep GetDefaultLogicalType(semantics::SemanticsContext &c) {
  evaluate::ExpressionAnalyzer analyzer{c};
  return c.MakeLogicalType(
      analyzer.GetDefaultKind(common::TypeCategory::Logical));
}

class FortranIRLowering {
public:
  using LabelMapType = std::map<flat::LabelRef, BasicBlock *>;
  using Closure = std::function<void(const LabelMapType &)>;

  FortranIRLowering(semantics::SemanticsContext &sc, bool debugLinearIR)
    : fir_{new Program("program_name")}, semanticsContext_{sc},
      debugLinearFIR_{debugLinearIR} {}
  ~FortranIRLowering() { CHECK(!builder_); }

  template<typename A> constexpr bool Pre(const A &) { return true; }
  template<typename A> constexpr void Post(const A &) {}

  void Post(const parser::MainProgram &mainp) {
    std::string mainName{"_MAIN"s};
    if (auto &ps{
            std::get<std::optional<parser::Statement<parser::ProgramStmt>>>(
                mainp.t)}) {
      mainName = ps->statement.v.ToString();
    }
    ProcessRoutine(mainp, mainName);
  }
  void Post(const parser::FunctionSubprogram &subp) {
    ProcessRoutine(subp,
        std::get<parser::Name>(
            std::get<parser::Statement<parser::FunctionStmt>>(subp.t)
                .statement.t)
            .ToString());
  }
  void Post(const parser::SubroutineSubprogram &subp) {
    ProcessRoutine(subp,
        std::get<parser::Name>(
            std::get<parser::Statement<parser::SubroutineStmt>>(subp.t)
                .statement.t)
            .ToString());
  }

  Program *program() { return fir_; }

  // convert a parse tree data reference to an Expression
  template<typename A> Expression ToExpression(const A &a) {
    return {std::move(semantics::AnalyzeExpr(semanticsContext_, a).value())};
  }

  TypeRep GetDefaultIntegerType() {
    return FIR::GetDefaultIntegerType(semanticsContext_);
  }
  // build a simple arithmetic Expression
  template<template<typename> class OPR>
  Expression ConsExpr(Expression e1, Expression e2) {
    evaluate::ExpressionAnalyzer context{semanticsContext_};
    ConformabilityCheck(context.GetContextualMessages(), e1, e2);
    return evaluate::NumericOperation<OPR>(context.GetContextualMessages(),
        std::move(e1), std::move(e2),
        context.GetDefaultKind(common::TypeCategory::Real))
        .value();
  }
  Expression ConsExpr(
      common::RelationalOperator op, Expression e1, Expression e2) {
    evaluate::ExpressionAnalyzer context{semanticsContext_};
    return evaluate::AsGenericExpr(evaluate::Relate(
        context.GetContextualMessages(), op, std::move(e1), std::move(e2))
                                       .value());
  }
  parser::Name MakeTemp(Type tempType) {
    auto name{NewTemporaryName()};
    auto details{semantics::ObjectEntityDetails{true}};
    details.set_type(std::move(*tempType));
    auto *sym{&semanticsContext_.globalScope().MakeSymbol(
        name, {}, std::move(details))};
    return {name, sym};
  }
  QualifiedStmt<Addressable_impl> CreateTemp(TypeRep &&spec) {
    TypeRep declSpec{std::move(spec)};
    auto temp{MakeTemp(&declSpec)};
    auto expr{ToExpression(temp)};
    auto *localType{temp.symbol->get<semantics::ObjectEntityDetails>().type()};
    return builder_->CreateLocal(localType, expr);
  }

  template<typename T>
  void ProcessRoutine(const T &here, const std::string &name) {
    CHECK(!fir_->containsProcedure(name));
    auto *subp{fir_->getOrInsertProcedure(name, nullptr, {})};
    builder_ = new FIRBuilder(*CreateBlock(subp->getLastRegion()));
    AnalysisData ad;
    CreateFlatIR(here, linearOperations_, ad);
    if (debugLinearFIR_) {
      DebugChannel() << "define @" << name << "(...) {\n";
      dump(linearOperations_);
      DebugChannel() << "}\n";
    }
    ConstructFIR(ad);
    DrawRemainingArcs();
    Cleanup();
  }

  template<typename A>
  Statement *BindArrayWithBoundSpecifier(
      const parser::DataRef &dataRef, const std::list<A> &bl) {
    // TODO
    return nullptr;
  }

  Statement *CreatePointerValue(const parser::PointerAssignmentStmt &stmt) {
    auto &dataRef{std::get<parser::DataRef>(stmt.t)};
    auto &bounds{std::get<parser::PointerAssignmentStmt::Bounds>(stmt.t)};
    auto *remap{std::visit(
        common::visitors{
            [&](const std::list<parser::BoundsRemapping> &bl) -> Statement * {
              if (bl.empty()) {
                return nullptr;
              }
              return BindArrayWithBoundSpecifier(dataRef, bl);
            },
            [&](const std::list<parser::BoundsSpec> &bl) -> Statement * {
              if (bl.empty()) {
                return nullptr;
              }
              return BindArrayWithBoundSpecifier(dataRef, bl);
            },
        },
        bounds.u)};
    if (remap) {
      return remap;
    }
    return builder_->CreateAddr(ToExpression(dataRef));
  }
  Type CreateAllocationValue(const parser::Allocation *allocation,
      const parser::AllocateStmt *statement) {
    auto &obj{std::get<parser::AllocateObject>(allocation->t)};
    (void)obj;
    // TODO: build an expression for the allocation
    return nullptr;
  }
  QualifiedStmt<AllocateInsn> CreateDeallocationValue(
      const parser::AllocateObject *allocateObject,
      const parser::DeallocateStmt *statement) {
    // TODO: build an expression for the deallocation
    return QualifiedStmt<AllocateInsn>{nullptr};
  }

  // IO argument translations ...
  IOCallArguments CreateBackspaceArguments(
      const std::list<parser::PositionOrFlushSpec> &);
  IOCallArguments CreateCloseArguments(
      const std::list<parser::CloseStmt::CloseSpec> &);
  IOCallArguments CreateEndfileArguments(
      const std::list<parser::PositionOrFlushSpec> &);
  IOCallArguments CreateFlushArguments(
      const std::list<parser::PositionOrFlushSpec> &);
  IOCallArguments CreateRewindArguments(
      const std::list<parser::PositionOrFlushSpec> &);
  IOCallArguments CreateInquireArguments(
      const std::list<parser::InquireSpec> &);
  IOCallArguments CreateInquireArguments(const parser::InquireStmt::Iolength &);
  IOCallArguments CreateOpenArguments(const std::list<parser::ConnectSpec> &);
  IOCallArguments CreateWaitArguments(const std::list<parser::WaitSpec> &);
  IOCallArguments CreatePrintArguments(const parser::Format &format,
      const std::list<parser::OutputItem> &outputs);
  IOCallArguments CreateReadArguments(
      const std::optional<parser::IoUnit> &ioUnit,
      const std::optional<parser::Format> &format,
      const std::list<parser::IoControlSpec> &controls,
      const std::list<parser::InputItem> &inputs);
  IOCallArguments CreateWriteArguments(
      const std::optional<parser::IoUnit> &ioUnit,
      const std::optional<parser::Format> &format,
      const std::list<parser::IoControlSpec> &controls,
      const std::list<parser::OutputItem> &outputs);

  // Image related runtime argument translations ...
  RuntimeCallArguments CreateEventPostArguments(const parser::EventPostStmt &);
  RuntimeCallArguments CreateEventWaitArguments(const parser::EventWaitStmt &);
  RuntimeCallArguments CreateFailImageArguments(const parser::FailImageStmt &);
  RuntimeCallArguments CreateFormTeamArguments(const parser::FormTeamStmt &);
  RuntimeCallArguments CreateLockArguments(const parser::LockStmt &);
  RuntimeCallArguments CreatePauseArguments(const parser::PauseStmt &);
  RuntimeCallArguments CreateStopArguments(const parser::StopStmt &);
  RuntimeCallArguments CreateSyncAllArguments(const parser::SyncAllStmt &);
  RuntimeCallArguments CreateSyncImagesArguments(
      const parser::SyncImagesStmt &);
  RuntimeCallArguments CreateSyncMemoryArguments(
      const parser::SyncMemoryStmt &);
  RuntimeCallArguments CreateSyncTeamArguments(const parser::SyncTeamStmt &);
  RuntimeCallArguments CreateUnlockArguments(const parser::UnlockStmt &);

  // CALL translations ...
  const Value CreateCalleeValue(const parser::ProcedureDesignator &designator) {
    return NOTHING;
  }
  CallArguments CreateCallArguments(
      const std::list<parser::ActualArgSpec> &arguments) {
    return CallArguments{};
  }

  template<typename STMTTYPE, typename CT>
  Statement *GetSwitchSelector(const CT *selectConstruct) {
    return std::visit(
        common::visitors{
            [&](const parser::Expr &e) {
              return builder_->CreateExpr(ExprRef(e));
            },
            [&](const parser::Variable &v) {
              return builder_->CreateExpr(ToExpression(v));
            },
        },
        std::get<parser::Selector>(
            std::get<parser::Statement<STMTTYPE>>(selectConstruct->t)
                .statement.t)
            .u);
  }
  Statement *GetSwitchRankSelector(
      const parser::SelectRankConstruct *selectRankConstruct) {
    return GetSwitchSelector<parser::SelectRankStmt>(selectRankConstruct);
  }
  Statement *GetSwitchTypeSelector(
      const parser::SelectTypeConstruct *selectTypeConstruct) {
    return GetSwitchSelector<parser::SelectTypeStmt>(selectTypeConstruct);
  }
  Statement *GetSwitchCaseSelector(const parser::CaseConstruct *construct) {
    using A = parser::Statement<parser::SelectCaseStmt>;
    const auto &x{std::get<parser::Scalar<parser::Expr>>(
        std::get<A>(construct->t).statement.t)};
    return builder_->CreateExpr(ExprRef(x.thing));
  }

  SwitchArguments ComposeIOSwitchArgs(const flat::SwitchIOOp &IOp) {
    return {};  // FIXME
  }
  SwitchArguments ComposeSwitchArgs(const flat::SwitchOp &op) {
    return std::visit(
        common::visitors{
            [&](const parser::ComputedGotoStmt *c) {
              const auto &e{std::get<parser::ScalarIntExpr>(c->t)};
              auto *exp{builder_->CreateExpr(ExprRef(e.thing.thing))};
              return SwitchArguments{exp, {}, op.refs};
            },
            [&](const parser::ArithmeticIfStmt *c) {
              const auto &e{std::get<parser::Expr>(c->t)};
              auto *exp{builder_->CreateExpr(ExprRef(e))};
              return SwitchArguments{exp, {}, op.refs};
            },
            [&](const parser::CallStmt *c) {
              auto exp{NOTHING};  // fixme - result of call
              return SwitchArguments{exp, {}, op.refs};
            },
            [](const auto *) {
              WRONG_PATH();
              return SwitchArguments{};
            },
        },
        op.u);
  }

  SwitchCaseArguments ComposeSwitchCaseArguments(
      const parser::CaseConstruct *caseConstruct,
      const std::vector<flat::LabelRef> &refs) {
    using A = std::list<parser::CaseConstruct::Case>;
    auto &cases{std::get<A>(caseConstruct->t)};
    SwitchCaseArguments result{GetSwitchCaseSelector(caseConstruct),
        populateSwitchValues(builder_, cases), std::move(refs)};
    cleanupSwitchPairs<SwitchCaseStmt>(result.values, result.labels);
    return result;
  }
  SwitchRankArguments ComposeSwitchRankArguments(
      const parser::SelectRankConstruct *crct,
      const std::vector<flat::LabelRef> &refs) {
    auto &ranks{
        std::get<std::list<parser::SelectRankConstruct::RankCase>>(crct->t)};
    SwitchRankArguments result{GetSwitchRankSelector(crct),
        populateSwitchValues(ranks), std::move(refs)};
    if (auto &name{GetSwitchAssociateName<parser::SelectRankStmt>(crct)}) {
      (void)name;  // get rid of warning
      // TODO: handle associate-name -> Add an assignment stmt?
    }
    cleanupSwitchPairs<SwitchRankStmt>(result.values, result.labels);
    return result;
  }
  SwitchTypeArguments ComposeSwitchTypeArguments(
      const parser::SelectTypeConstruct *selectTypeConstruct,
      const std::vector<flat::LabelRef> &refs) {
    auto &types{std::get<std::list<parser::SelectTypeConstruct::TypeCase>>(
        selectTypeConstruct->t)};
    SwitchTypeArguments result{GetSwitchTypeSelector(selectTypeConstruct),
        populateSwitchValues(types), std::move(refs)};
    if (auto &name{GetSwitchAssociateName<parser::SelectTypeStmt>(
            selectTypeConstruct)}) {
      (void)name;  // get rid of warning
      // TODO: handle associate-name -> Add an assignment stmt?
    }
    cleanupSwitchPairs<SwitchTypeStmt>(result.values, result.labels);
    return result;
  }

  void handleIntrinsicAssignmentStmt(const parser::AssignmentStmt &stmt) {
    // TODO: check if allocation or reallocation should happen, etc.
    auto *value{builder_->CreateExpr(ExprRef(std::get<parser::Expr>(stmt.t)))};
    auto addr{
        builder_->CreateAddr(ToExpression(std::get<parser::Variable>(stmt.t)))};
    builder_->CreateStore(addr, value);
  }
  void handleDefinedAssignmentStmt(const parser::AssignmentStmt &stmt) {
    CHECK(false && "TODO defined assignment");
  }
  void handleAssignmentStmt(const parser::AssignmentStmt &stmt) {
    // TODO: is this an intrinsic assignment or a defined assignment?
    if (true) {
      handleIntrinsicAssignmentStmt(stmt);
    } else {
      handleDefinedAssignmentStmt(stmt);
    }
  }

  struct AllocOpts {
    std::optional<Expression> mold;
    std::optional<Expression> source;
    std::optional<Expression> stat;
    std::optional<Expression> errmsg;
  };
  void handleAllocateStmt(const parser::AllocateStmt &stmt) {
    // extract options from list -> opts
    AllocOpts opts;
    for (auto &allocOpt : std::get<std::list<parser::AllocOpt>>(stmt.t)) {
      std::visit(
          common::visitors{
              [&](const parser::AllocOpt::Mold &m) {
                opts.mold = *ExprRef(m.v);
              },
              [&](const parser::AllocOpt::Source &s) {
                opts.source = *ExprRef(s.v);
              },
              [&](const parser::StatOrErrmsg &var) {
                std::visit(
                    common::visitors{
                        [&](const parser::StatVariable &sv) {
                          opts.stat = ToExpression(sv.v.thing.thing);
                        },
                        [&](const parser::MsgVariable &mv) {
                          opts.errmsg = ToExpression(mv.v.thing.thing);
                        },
                    },
                    var.u);
              },
          },
          allocOpt.u);
    }
    // process the list of allocations
    for (auto &allocation : std::get<std::list<parser::Allocation>>(stmt.t)) {
      // TODO: add more arguments to builder as needed
      builder_->CreateAlloc(CreateAllocationValue(&allocation, &stmt));
    }
  }

  void handleActionStatement(
      AnalysisData &ad, const parser::Statement<parser::ActionStmt> &stmt);

  void handleLinearAction(const flat::ActionOp &action, AnalysisData &ad) {
    handleActionStatement(ad, *action.v);
  }

  // DO loop handlers
  struct DoBoundsInfo {
    QualifiedStmt<Addressable_impl> doVariable;
    QualifiedStmt<Addressable_impl> counter;
    Statement *stepExpr;
    Statement *condition;
  };
  void PushDoContext(const parser::NonLabelDoStmt *doStmt,
      QualifiedStmt<Addressable_impl> doVar = nullptr,
      QualifiedStmt<Addressable_impl> counter = nullptr,
      Statement *stepExp = nullptr) {
    doMap_.emplace(doStmt, DoBoundsInfo{doVar, counter, stepExp});
  }
  void PopDoContext(const parser::NonLabelDoStmt *doStmt) {
    doMap_.erase(doStmt);
  }
  template<typename T> DoBoundsInfo *GetBoundsInfo(const T &linearOp) {
    auto *s{&std::get<parser::Statement<parser::NonLabelDoStmt>>(linearOp.v->t)
                 .statement};
    auto iter{doMap_.find(s)};
    if (iter != doMap_.end()) {
      return &iter->second;
    }
    CHECK(false && "DO context not present");
    return nullptr;
  }

  void handleLinearDoIncrement(const flat::DoIncrementOp &inc) {
    auto *info{GetBoundsInfo(inc)};
    if (info->doVariable) {
      if (info->stepExpr) {
        // evaluate: do_var = do_var + e3; counter--
        auto *incremented{builder_->CreateExpr(
            ConsExpr<evaluate::Add>(GetAddressable(info->doVariable)->address(),
                GetApplyExpr(info->stepExpr)->expression()))};
        builder_->CreateStore(info->doVariable, incremented);
        auto *decremented{builder_->CreateExpr(ConsExpr<evaluate::Subtract>(
            GetAddressable(info->counter)->address(), CreateConstant(1)))};
        builder_->CreateStore(info->counter, decremented);
      }
    }
  }

  // is (counter > 0)?
  void handleLinearDoCompare(const flat::DoCompareOp &cmp) {
    auto *info{GetBoundsInfo(cmp)};
    if (info->doVariable) {
      if (info->stepExpr) {
        Expression compare{ConsExpr(common::RelationalOperator::GT,
            getLocalVariable(info->counter), CreateConstant(0))};
        auto *cond{builder_->CreateExpr(&compare)};
        info->condition = cond;
      }
    }
  }

  // InitiateConstruct - many constructs require some initial setup
  void InitiateConstruct(const parser::AssociateStmt *stmt);
  void InitiateConstruct(const parser::SelectCaseStmt *stmt);
  void InitiateConstruct(const parser::ChangeTeamStmt *stmt);
  void InitiateConstruct(const parser::IfThenStmt *stmt);
  void InitiateConstruct(const parser::WhereConstructStmt *stmt);
  void InitiateConstruct(const parser::ForallConstructStmt *stmt);
  void InitiateConstruct(const parser::NonLabelDoStmt *stmt);

  // finish DO construct construction
  void FinishConstruct(const parser::NonLabelDoStmt *stmt) {
    auto &ctrl{std::get<std::optional<parser::LoopControl>>(stmt->t)};
    if (ctrl.has_value()) {
      if (std::holds_alternative<parser::LoopControl::Bounds>(ctrl->u)) {
        PopDoContext(stmt);
      }
    }
  }

  Statement *BuildLoopLatchExpression(const parser::NonLabelDoStmt *stmt) {
    auto &loopCtrl{std::get<std::optional<parser::LoopControl>>(stmt->t)};
    if (loopCtrl.has_value()) {
      return std::visit(
          common::visitors{
              [&](const parser::LoopControl::Bounds &) {
                return doMap_.find(stmt)->second.condition;
              },
              [&](const parser::ScalarLogicalExpr &sle) {
                auto &exp{sle.thing.thing.value()};
                SEMANTICS_CHECK(ExprRef(exp), "DO WHILE condition missing");
                return builder_->CreateExpr(ExprRef(exp));
              },
              [&](const parser::LoopControl::Concurrent &concurrent) {
                // FIXME - how do we want to lower DO CONCURRENT?
                return builder_->CreateExpr(AlwaysTrueExpression());
              },
          },
          loopCtrl->u);
    }
    return builder_->CreateExpr(AlwaysTrueExpression());
  }

  template<typename SWITCHTYPE, typename F>
  void AddOrQueueSwitch(Value condition,
      const std::vector<typename SWITCHTYPE::ValueType> &values,
      const std::vector<flat::LabelRef> &labels, F function);
  void AddOrQueueBranch(flat::LabelRef dest);
  void AddOrQueueCGoto(Statement *condition, flat::LabelRef trueBlock,
      flat::LabelRef falseBlock);
  void AddOrQueueIGoto(AnalysisData &ad, const semantics::Symbol *symbol,
      const std::vector<flat::LabelRef> &labels);

  void ConstructFIR(AnalysisData &ad);

  void EnterRegion(const parser::CharBlock &pos) {
    auto *region{builder_->GetCurrentRegion()};
    auto *scope{semanticsContext_.globalScope().FindScope(pos)};
    auto *newRegion{Region::Create(region->getParent(), scope, region)};
    auto *block{CreateBlock(newRegion)};
    CheckInsertionPoint();
    builder_->CreateBranch(block);
    builder_->SetInsertionPoint(block);
  }

  void ExitRegion() {
    builder_->SetCurrentRegion(builder_->GetCurrentRegion()->GetEnclosing());
  }

  void CheckInsertionPoint() {
    if (!builder_->GetInsertionPoint()) {
      builder_->SetInsertionPoint(CreateBlock(builder_->GetCurrentRegion()));
    }
  }

  Variable *ConvertToVariable(const semantics::Symbol *symbol) {
    // FIXME: how to convert semantics::Symbol to evaluate::Variable?
    return new Variable(symbol);
  }

  void DrawRemainingArcs() {
    for (auto &arc : controlFlowEdgesToAdd_) {
      arc(blockMap_);
    }
  }

  BasicBlock *CreateBlock(Region *region) { return BasicBlock::Create(region); }

  void Cleanup() {
    delete builder_;
    builder_ = nullptr;
    linearOperations_.clear();
    controlFlowEdgesToAdd_.clear();
    blockMap_.clear();
  }

  FIRBuilder *builder_{nullptr};
  Program *fir_;
  std::list<flat::Op> linearOperations_;
  std::list<Closure> controlFlowEdgesToAdd_;
  std::map<const parser::NonLabelDoStmt *, DoBoundsInfo> doMap_;
  LabelMapType blockMap_;
  semantics::SemanticsContext &semanticsContext_;
  bool debugLinearFIR_;
};

void FortranIRLowering::handleActionStatement(
    AnalysisData &ad, const parser::Statement<parser::ActionStmt> &stmt) {
  std::visit(
      common::visitors{
          [&](const common::Indirection<parser::AllocateStmt> &s) {
            handleAllocateStmt(s.value());
          },
          [&](const common::Indirection<parser::AssignmentStmt> &s) {
            handleAssignmentStmt(s.value());
          },
          [&](const common::Indirection<parser::BackspaceStmt> &s) {
            builder_->CreateIOCall(InputOutputCallBackspace,
                CreateBackspaceArguments(s.value().v));
          },
          [&](const common::Indirection<parser::CallStmt> &s) {
            builder_->CreateCall(nullptr,
                CreateCalleeValue(
                    std::get<parser::ProcedureDesignator>(s.value().v.t)),
                CreateCallArguments(
                    std::get<std::list<parser::ActualArgSpec>>(s.value().v.t)));
          },
          [&](const common::Indirection<parser::CloseStmt> &s) {
            builder_->CreateIOCall(
                InputOutputCallClose, CreateCloseArguments(s.value().v));
          },
          [](const parser::ContinueStmt &) { WRONG_PATH(); },
          [](const common::Indirection<parser::CycleStmt> &) { WRONG_PATH(); },
          [&](const common::Indirection<parser::DeallocateStmt> &s) {
            for (auto &alloc :
                std::get<std::list<parser::AllocateObject>>(s.value().t)) {
              builder_->CreateDealloc(
                  CreateDeallocationValue(&alloc, &s.value()));
            }
          },
          [&](const common::Indirection<parser::EndfileStmt> &s) {
            builder_->CreateIOCall(
                InputOutputCallEndfile, CreateEndfileArguments(s.value().v));
          },
          [&](const common::Indirection<parser::EventPostStmt> &s) {
            builder_->CreateRuntimeCall(
                RuntimeCallEventPost, CreateEventPostArguments(s.value()));
          },
          [&](const common::Indirection<parser::EventWaitStmt> &s) {
            builder_->CreateRuntimeCall(
                RuntimeCallEventWait, CreateEventWaitArguments(s.value()));
          },
          [](const common::Indirection<parser::ExitStmt> &) { WRONG_PATH(); },
          [&](const parser::FailImageStmt &s) {
            builder_->CreateRuntimeCall(
                RuntimeCallFailImage, CreateFailImageArguments(s));
          },
          [&](const common::Indirection<parser::FlushStmt> &s) {
            builder_->CreateIOCall(
                InputOutputCallFlush, CreateFlushArguments(s.value().v));
          },
          [&](const common::Indirection<parser::FormTeamStmt> &s) {
            builder_->CreateRuntimeCall(
                RuntimeCallFormTeam, CreateFormTeamArguments(s.value()));
          },
          [](const common::Indirection<parser::GotoStmt> &) { WRONG_PATH(); },
          [](const common::Indirection<parser::IfStmt> &) { WRONG_PATH(); },
          [&](const common::Indirection<parser::InquireStmt> &s) {
            std::visit(
                common::visitors{
                    [&](const std::list<parser::InquireSpec> &specifiers) {
                      builder_->CreateIOCall(InputOutputCallInquire,
                          CreateInquireArguments(specifiers));
                    },
                    [&](const parser::InquireStmt::Iolength &iolength) {
                      builder_->CreateIOCall(InputOutputCallInquire,
                          CreateInquireArguments(iolength));
                    },
                },
                s.value().u);
          },
          [&](const common::Indirection<parser::LockStmt> &s) {
            builder_->CreateRuntimeCall(
                RuntimeCallLock, CreateLockArguments(s.value()));
          },
          [&](const common::Indirection<parser::NullifyStmt> &s) {
            for (auto &obj : s.value().v) {
              std::visit(
                  common::visitors{
                      [&](const parser::Name &n) {
                        auto s{builder_->CreateAddr(ToExpression(n))};
                        builder_->CreateNullify(s);
                      },
                      [&](const parser::StructureComponent &sc) {
                        auto s{builder_->CreateAddr(ToExpression(sc))};
                        builder_->CreateNullify(s);
                      },
                  },
                  obj.u);
            }
          },
          [&](const common::Indirection<parser::OpenStmt> &s) {
            builder_->CreateIOCall(
                InputOutputCallOpen, CreateOpenArguments(s.value().v));
          },
          [&](const common::Indirection<parser::PointerAssignmentStmt> &s) {
            auto *value{CreatePointerValue(s.value())};
            auto addr{builder_->CreateAddr(
                ExprRef(std::get<parser::Expr>(s.value().t)))};
            builder_->CreateStore(addr, value);
          },
          [&](const common::Indirection<parser::PrintStmt> &s) {
            builder_->CreateIOCall(InputOutputCallPrint,
                CreatePrintArguments(std::get<parser::Format>(s.value().t),
                    std::get<std::list<parser::OutputItem>>(s.value().t)));
          },
          [&](const common::Indirection<parser::ReadStmt> &s) {
            builder_->CreateIOCall(InputOutputCallRead,
                CreateReadArguments(s.value().iounit, s.value().format,
                    s.value().controls, s.value().items));
          },
          [](const common::Indirection<parser::ReturnStmt> &) { WRONG_PATH(); },
          [&](const common::Indirection<parser::RewindStmt> &s) {
            builder_->CreateIOCall(
                InputOutputCallRewind, CreateRewindArguments(s.value().v));
          },
          [&](const common::Indirection<parser::StopStmt> &s) {
            builder_->CreateRuntimeCall(
                RuntimeCallStop, CreateStopArguments(s.value()));
          },
          [&](const common::Indirection<parser::SyncAllStmt> &s) {
            builder_->CreateRuntimeCall(
                RuntimeCallSyncAll, CreateSyncAllArguments(s.value()));
          },
          [&](const common::Indirection<parser::SyncImagesStmt> &s) {
            builder_->CreateRuntimeCall(
                RuntimeCallSyncImages, CreateSyncImagesArguments(s.value()));
          },
          [&](const common::Indirection<parser::SyncMemoryStmt> &s) {
            builder_->CreateRuntimeCall(
                RuntimeCallSyncMemory, CreateSyncMemoryArguments(s.value()));
          },
          [&](const common::Indirection<parser::SyncTeamStmt> &s) {
            builder_->CreateRuntimeCall(
                RuntimeCallSyncTeam, CreateSyncTeamArguments(s.value()));
          },
          [&](const common::Indirection<parser::UnlockStmt> &s) {
            builder_->CreateRuntimeCall(
                RuntimeCallUnlock, CreateUnlockArguments(s.value()));
          },
          [&](const common::Indirection<parser::WaitStmt> &s) {
            builder_->CreateIOCall(
                InputOutputCallWait, CreateWaitArguments(s.value().v));
          },
          [](const common::Indirection<parser::WhereStmt> &) { /*fixme*/ },
          [&](const common::Indirection<parser::WriteStmt> &s) {
            builder_->CreateIOCall(InputOutputCallWrite,
                CreateWriteArguments(s.value().iounit, s.value().format,
                    s.value().controls, s.value().items));
          },
          [](const common::Indirection<parser::ComputedGotoStmt> &) {
            WRONG_PATH();
          },
          [](const common::Indirection<parser::ForallStmt> &) { /*fixme*/ },
          [](const common::Indirection<parser::ArithmeticIfStmt> &) {
            WRONG_PATH();
          },
          [&](const common::Indirection<parser::AssignStmt> &s) {
            auto addr{builder_->CreateAddr(
                ToExpression(std::get<parser::Name>(s.value().t)))};
            auto *block{blockMap_
                            .find(flat::FetchLabel(
                                ad, std::get<parser::Label>(s.value().t))
                                      .get())
                            ->second};
            builder_->CreateStore(addr, block);
          },
          [](const common::Indirection<parser::AssignedGotoStmt> &) {
            WRONG_PATH();
          },
          [&](const common::Indirection<parser::PauseStmt> &s) {
            builder_->CreateRuntimeCall(
                RuntimeCallPause, CreatePauseArguments(s.value()));
          },
      },
      stmt.statement.u);
}

template<typename SWITCHTYPE, typename F>
void FortranIRLowering::AddOrQueueSwitch(Value condition,
    const std::vector<typename SWITCHTYPE::ValueType> &values,
    const std::vector<flat::LabelRef> &labels, F function) {
  auto defer{false};
  typename SWITCHTYPE::ValueSuccPairListType cases;
  CHECK(values.size() == labels.size());
  auto valiter{values.begin()};
  for (auto lab : labels) {
    auto labIter{blockMap_.find(lab)};
    if (labIter == blockMap_.end()) {
      defer = true;
      break;
    } else {
      cases.emplace_back(*valiter++, labIter->second);
    }
  }
  if (defer) {
    using namespace std::placeholders;
    controlFlowEdgesToAdd_.emplace_back(std::bind(
        [](FIRBuilder *builder, BasicBlock *block, Value expr,
            const std::vector<typename SWITCHTYPE::ValueType> &values,
            const std::vector<flat::LabelRef> &labels, F function,
            const LabelMapType &map) {
          builder->SetInsertionPoint(block);
          typename SWITCHTYPE::ValueSuccPairListType cases;
          auto valiter{values.begin()};
          for (auto &lab : labels) {
            cases.emplace_back(*valiter++, map.find(lab)->second);
          }
          function(builder, expr, cases);
        },
        builder_, builder_->GetInsertionPoint(), condition, values, labels,
        function, _1));
  } else {
    function(builder_, condition, cases);
  }
}

void FortranIRLowering::AddOrQueueBranch(flat::LabelRef dest) {
  auto iter{blockMap_.find(dest)};
  if (iter != blockMap_.end()) {
    builder_->CreateBranch(iter->second);
  } else {
    using namespace std::placeholders;
    controlFlowEdgesToAdd_.emplace_back(std::bind(
        [](FIRBuilder *builder, BasicBlock *block, flat::LabelRef dest,
            const LabelMapType &map) {
          builder->SetInsertionPoint(block);
          CHECK(map.find(dest) != map.end() && "no destination");
          builder->CreateBranch(map.find(dest)->second);
        },
        builder_, builder_->GetInsertionPoint(), dest, _1));
  }
}

void FortranIRLowering::AddOrQueueCGoto(
    Statement *condition, flat::LabelRef trueBlock, flat::LabelRef falseBlock) {
  auto trueIter{blockMap_.find(trueBlock)};
  auto falseIter{blockMap_.find(falseBlock)};
  if (trueIter != blockMap_.end() && falseIter != blockMap_.end()) {
    builder_->CreateConditionalBranch(
        condition, trueIter->second, falseIter->second);
  } else {
    using namespace std::placeholders;
    controlFlowEdgesToAdd_.emplace_back(std::bind(
        [](FIRBuilder *builder, BasicBlock *block, Statement *expr,
            flat::LabelRef trueDest, flat::LabelRef falseDest,
            const LabelMapType &map) {
          builder->SetInsertionPoint(block);
          CHECK(map.find(trueDest) != map.end());
          CHECK(map.find(falseDest) != map.end());
          builder->CreateConditionalBranch(
              expr, map.find(trueDest)->second, map.find(falseDest)->second);
        },
        builder_, builder_->GetInsertionPoint(), condition, trueBlock,
        falseBlock, _1));
  }
}

void FortranIRLowering::AddOrQueueIGoto(AnalysisData &ad,
    const semantics::Symbol *symbol,
    const std::vector<flat::LabelRef> &labels) {
  auto useLabels{labels.empty() ? flat::GetAssign(ad, symbol) : labels};
  auto defer{false};
  IndirectBranchStmt::TargetListType blocks;
  for (auto lab : useLabels) {
    auto iter{blockMap_.find(lab)};
    if (iter == blockMap_.end()) {
      defer = true;
      break;
    } else {
      blocks.push_back(iter->second);
    }
  }
  if (defer) {
    using namespace std::placeholders;
    controlFlowEdgesToAdd_.emplace_back(std::bind(
        [](FIRBuilder *builder, BasicBlock *block, Variable *variable,
            const std::vector<flat::LabelRef> &fixme, const LabelMapType &map) {
          builder->SetInsertionPoint(block);
          builder->CreateIndirectBr(variable, {});  // FIXME
        },
        builder_, builder_->GetInsertionPoint(), nullptr /*symbol*/, useLabels,
        _1));
  } else {
    builder_->CreateIndirectBr(ConvertToVariable(symbol), blocks);
  }
}

void FortranIRLowering::InitiateConstruct(const parser::AssociateStmt *stmt) {
  for (auto &assoc : std::get<std::list<parser::Association>>(stmt->t)) {
    auto &selector{std::get<parser::Selector>(assoc.t)};
    auto *expr{builder_->CreateExpr(std::visit(
        common::visitors{
            [&](const parser::Variable &v) { return ToExpression(v); },
            [](const parser::Expr &e) { return *ExprRef(e); },
        },
        selector.u))};
    auto name{
        builder_->CreateAddr(ToExpression(std::get<parser::Name>(assoc.t)))};
    builder_->CreateStore(name, expr);
  }
}

void FortranIRLowering::InitiateConstruct(const parser::SelectCaseStmt *stmt) {
  builder_->CreateExpr(
      ExprRef(std::get<parser::Scalar<parser::Expr>>(stmt->t).thing));
}

void FortranIRLowering::InitiateConstruct(const parser::ChangeTeamStmt *stmt) {
  // FIXME
}

void FortranIRLowering::InitiateConstruct(const parser::IfThenStmt *stmt) {
  const auto &e{std::get<parser::ScalarLogicalExpr>(stmt->t).thing};
  builder_->CreateExpr(ExprRef(e.thing));
}

void FortranIRLowering::InitiateConstruct(
    const parser::WhereConstructStmt *stmt) {
  const auto &e{std::get<parser::LogicalExpr>(stmt->t)};
  builder_->CreateExpr(ExprRef(e.thing));
}

void FortranIRLowering::InitiateConstruct(
    const parser::ForallConstructStmt *stmt) {
  // FIXME
}

void FortranIRLowering::InitiateConstruct(const parser::NonLabelDoStmt *stmt) {
  auto &ctrl{std::get<std::optional<parser::LoopControl>>(stmt->t)};
  if (ctrl.has_value()) {
    std::visit(
        common::visitors{
          [&](const parser::LoopControl::Bounds &bounds) {
              auto name{
                  builder_->CreateAddr(ToExpression(bounds.name.thing))};
              // evaluate e1, e2 [, e3] ...
              auto *e1{builder_->CreateExpr(ExprRef(bounds.lower.thing))};
              auto *e2{builder_->CreateExpr(ExprRef(bounds.upper.thing))};
              Statement *e3;
              if (bounds.step.has_value()) {
                e3 = builder_->CreateExpr(ExprRef(bounds.step->thing));
              } else {
                e3 = builder_->CreateExpr(CreateConstant(1));
              }
              // name <- e1
              builder_->CreateStore(name, e1);
              auto tripCounter{CreateTemp(GetDefaultIntegerType())};
              // See 11.1.7.4.1, para. 1, item (3)
              // totalTrips ::= iteration count = a
              //   where a = (e2 - e1 + e3) / e3 if a > 0 and 0 otherwise
              Expression tripExpr{ConsExpr<evaluate::Divide>(
                  ConsExpr<evaluate::Add>(
                      ConsExpr<evaluate::Subtract>(
                          getApplyExpr(e2), getApplyExpr(e1)),
                      getApplyExpr(e3)),
                  getApplyExpr(e3))};
              auto *totalTrips{builder_->CreateExpr(&tripExpr)};
              builder_->CreateStore(tripCounter, totalTrips);
              PushDoContext(stmt, name, tripCounter, e3);
            },
            [&](const parser::ScalarLogicalExpr &expr) {
              // See 11.1.7.4.1, para. 2
              // See BuildLoopLatchExpression()
              PushDoContext(stmt);
            },
            [&](const parser::LoopControl::Concurrent &cc) {
              // See 11.1.7.4.2
              // FIXME
            },
        },
        ctrl->u);
  } else {
    // loop forever (See 11.1.7.4.1, para. 2)
    PushDoContext(stmt);
  }
}

// Image related runtime argument translations ...
RuntimeCallArguments FortranIRLowering::CreateEventPostArguments(
    const parser::EventPostStmt &stmt) {
  return RuntimeCallArguments{};
}
RuntimeCallArguments FortranIRLowering::CreateEventWaitArguments(
    const parser::EventWaitStmt &stmt) {
  return RuntimeCallArguments{};
}
RuntimeCallArguments FortranIRLowering::CreateFailImageArguments(
    const parser::FailImageStmt &stmt) {
  return RuntimeCallArguments{};
}
RuntimeCallArguments FortranIRLowering::CreateFormTeamArguments(
    const parser::FormTeamStmt &stmt) {
  return RuntimeCallArguments{};
}
RuntimeCallArguments FortranIRLowering::CreateLockArguments(
    const parser::LockStmt &stmt) {
  return RuntimeCallArguments{};
}
RuntimeCallArguments FortranIRLowering::CreatePauseArguments(
    const parser::PauseStmt &stmt) {
  return RuntimeCallArguments{};
}
RuntimeCallArguments FortranIRLowering::CreateStopArguments(
    const parser::StopStmt &stmt) {
  return RuntimeCallArguments{};
}
RuntimeCallArguments FortranIRLowering::CreateSyncAllArguments(
    const parser::SyncAllStmt &stmt) {
  return RuntimeCallArguments{};
}
RuntimeCallArguments FortranIRLowering::CreateSyncImagesArguments(
    const parser::SyncImagesStmt &stmt) {
  return RuntimeCallArguments{};
}
RuntimeCallArguments FortranIRLowering::CreateSyncMemoryArguments(
    const parser::SyncMemoryStmt &stmt) {
  return RuntimeCallArguments{};
}
RuntimeCallArguments FortranIRLowering::CreateSyncTeamArguments(
    const parser::SyncTeamStmt &stmt) {
  return RuntimeCallArguments{};
}
RuntimeCallArguments FortranIRLowering::CreateUnlockArguments(
    const parser::UnlockStmt &stmt) {
  return RuntimeCallArguments{};
}

// I/O related runtime argument translations ...
IOCallArguments FortranIRLowering::CreateBackspaceArguments(
    const std::list<parser::PositionOrFlushSpec> &specifiers) {
  return IOCallArguments{};
}
IOCallArguments FortranIRLowering::CreateCloseArguments(
    const std::list<parser::CloseStmt::CloseSpec> &specifiers) {
  return IOCallArguments{};
}
IOCallArguments FortranIRLowering::CreateEndfileArguments(
    const std::list<parser::PositionOrFlushSpec> &specifiers) {
  return IOCallArguments{};
}
IOCallArguments FortranIRLowering::CreateFlushArguments(
    const std::list<parser::PositionOrFlushSpec> &specifiers) {
  return IOCallArguments{};
}
IOCallArguments FortranIRLowering::CreateRewindArguments(
    const std::list<parser::PositionOrFlushSpec> &specifiers) {
  return IOCallArguments{};
}
IOCallArguments FortranIRLowering::CreateInquireArguments(
    const std::list<parser::InquireSpec> &specifiers) {
  return IOCallArguments{};
}
IOCallArguments FortranIRLowering::CreateInquireArguments(
    const parser::InquireStmt::Iolength &iolength) {
  return IOCallArguments{};
}
IOCallArguments FortranIRLowering::CreateOpenArguments(
    const std::list<parser::ConnectSpec> &specifiers) {
  return IOCallArguments{};
}
IOCallArguments FortranIRLowering::CreateWaitArguments(
    const std::list<parser::WaitSpec> &specifiers) {
  return IOCallArguments{};
}
IOCallArguments FortranIRLowering::CreatePrintArguments(
    const parser::Format &format,
    const std::list<parser::OutputItem> &outputs) {
  return IOCallArguments{};
}
IOCallArguments FortranIRLowering::CreateReadArguments(
    const std::optional<parser::IoUnit> &ioUnit,
    const std::optional<parser::Format> &format,
    const std::list<parser::IoControlSpec> &controls,
    const std::list<parser::InputItem> &inputs) {
  return IOCallArguments{};
}
IOCallArguments FortranIRLowering::CreateWriteArguments(
    const std::optional<parser::IoUnit> &ioUnit,
    const std::optional<parser::Format> &format,
    const std::list<parser::IoControlSpec> &controls,
    const std::list<parser::OutputItem> &outputs) {
  return IOCallArguments{};
}

void FortranIRLowering::ConstructFIR(AnalysisData &ad) {
  for (auto iter{linearOperations_.begin()}, iend{linearOperations_.end()};
       iter != iend; ++iter) {
    const auto &op{*iter};
    std::visit(
        common::visitors{
            [&](const flat::LabelOp &op) {
              auto *newBlock{CreateBlock(builder_->GetCurrentRegion())};
              blockMap_.insert({op.get(), newBlock});
              if (builder_->GetInsertionPoint()) {
                builder_->CreateBranch(newBlock);
              }
              builder_->SetInsertionPoint(newBlock);
            },
            [&](const flat::GotoOp &op) {
              CheckInsertionPoint();
              AddOrQueueBranch(op.target);
              builder_->ClearInsertionPoint();
            },
            [&](const flat::IndirectGotoOp &op) {
              CheckInsertionPoint();
              AddOrQueueIGoto(ad, op.symbol, op.labelRefs);
              builder_->ClearInsertionPoint();
            },
            [&](const flat::ReturnOp &op) {
              CheckInsertionPoint();
              std::visit(
                  common::visitors{
                      [&](const parser::FailImageStmt *s) {
                        builder_->CreateRuntimeCall(
                            RuntimeCallFailImage, CreateFailImageArguments(*s));
                        builder_->CreateUnreachable();
                      },
                      [&](const parser::ReturnStmt *s) {
                        // alt-return
                        if (s->v) {
                          auto *exp{ExprRef(s->v->thing.thing)};
                          auto app{builder_->QualifiedCreateExpr(exp)};
                          builder_->CreateReturn(app);
                        } else {
                          auto zero{
                              builder_->QualifiedCreateExpr(CreateConstant(0))};
                          builder_->CreateReturn(zero);
                        }
                      },
                      [&](const parser::StopStmt *s) {
                        builder_->CreateRuntimeCall(
                            RuntimeCallStop, CreateStopArguments(*s));
                        builder_->CreateUnreachable();
                      },
                  },
                  op.u);
              builder_->ClearInsertionPoint();
            },
            [&](const flat::ConditionalGotoOp &cop) {
              CheckInsertionPoint();
              std::visit(
                  common::visitors{
                      [&](const parser::Statement<parser::IfThenStmt> *s) {
                        const auto &exp{
                            std::get<parser::ScalarLogicalExpr>(s->statement.t)
                                .thing.thing.value()};
                        SEMANTICS_CHECK(ExprRef(exp),
                            "IF THEN condition expression missing");
                        auto *cond{builder_->CreateExpr(ExprRef(exp))};
                        AddOrQueueCGoto(cond, cop.trueLabel, cop.falseLabel);
                      },
                      [&](const parser::Statement<parser::ElseIfStmt> *s) {
                        const auto &exp{
                            std::get<parser::ScalarLogicalExpr>(s->statement.t)
                                .thing.thing.value()};
                        SEMANTICS_CHECK(ExprRef(exp),
                            "ELSE IF condition expression missing");
                        auto *cond{builder_->CreateExpr(ExprRef(exp))};
                        AddOrQueueCGoto(cond, cop.trueLabel, cop.falseLabel);
                      },
                      [&](const parser::IfStmt *s) {
                        const auto &exp{
                            std::get<parser::ScalarLogicalExpr>(s->t)
                                .thing.thing.value()};
                        SEMANTICS_CHECK(
                            ExprRef(exp), "IF condition expression missing");
                        auto *cond{builder_->CreateExpr(ExprRef(exp))};
                        AddOrQueueCGoto(cond, cop.trueLabel, cop.falseLabel);
                      },
                      [&](const parser::Statement<parser::NonLabelDoStmt> *s) {
                        AddOrQueueCGoto(BuildLoopLatchExpression(&s->statement),
                            cop.trueLabel, cop.falseLabel);
                      }},
                  cop.u);
              builder_->ClearInsertionPoint();
            },
            [&](const flat::SwitchIOOp &IOp) {
              CheckInsertionPoint();
              auto args{ComposeIOSwitchArgs(IOp)};
              AddOrQueueSwitch<SwitchStmt>(
                  args.exp, args.values, args.labels, CreateSwitchHelper);
              builder_->ClearInsertionPoint();
            },
            [&](const flat::SwitchOp &sop) {
              CheckInsertionPoint();
              std::visit(
                  common::visitors{
                      [&](auto) {
                        auto args{ComposeSwitchArgs(sop)};
                        AddOrQueueSwitch<SwitchStmt>(args.exp, args.values,
                            args.labels, CreateSwitchHelper);
                      },
                      [&](const parser::CaseConstruct *crct) {
                        auto args{ComposeSwitchCaseArguments(crct, sop.refs)};
                        AddOrQueueSwitch<SwitchCaseStmt>(args.exp, args.values,
                            args.labels, CreateSwitchCaseHelper);
                      },
                      [&](const parser::SelectRankConstruct *crct) {
                        auto args{ComposeSwitchRankArguments(crct, sop.refs)};
                        AddOrQueueSwitch<SwitchRankStmt>(args.exp, args.values,
                            args.labels, CreateSwitchRankHelper);
                      },
                      [&](const parser::SelectTypeConstruct *crct) {
                        auto args{ComposeSwitchTypeArguments(crct, sop.refs)};
                        AddOrQueueSwitch<SwitchTypeStmt>(args.exp, args.values,
                            args.labels, CreateSwitchTypeHelper);
                      },
                  },
                  sop.u);
              builder_->ClearInsertionPoint();
            },
            [&](const flat::ActionOp &action) {
              CheckInsertionPoint();
              handleLinearAction(action, ad);
            },
            [&](const flat::DoIncrementOp &inc) {
              CheckInsertionPoint();
              handleLinearDoIncrement(inc);
            },
            [&](const flat::DoCompareOp &cmp) {
              CheckInsertionPoint();
              handleLinearDoCompare(cmp);
            },
            [&](const flat::BeginOp &con) {
              std::visit(
                  common::visitors{
                      [&](const parser::AssociateConstruct *crct) {
                        using A = parser::Statement<parser::AssociateStmt>;
                        const auto &statement{std::get<A>(crct->t)};
                        const auto &position{statement.source};
                        EnterRegion(position);
                        InitiateConstruct(&statement.statement);
                      },
                      [&](const parser::BlockConstruct *crct) {
                        using A = parser::Statement<parser::BlockStmt>;
                        EnterRegion(std::get<A>(crct->t).source);
                      },
                      [&](const parser::CaseConstruct *crct) {
                        using A = parser::Statement<parser::SelectCaseStmt>;
                        InitiateConstruct(&std::get<A>(crct->t).statement);
                      },
                      [&](const parser::ChangeTeamConstruct *crct) {
                        using A = parser::Statement<parser::ChangeTeamStmt>;
                        const auto &statement{std::get<A>(crct->t)};
                        EnterRegion(statement.source);
                        InitiateConstruct(&statement.statement);
                      },
                      [&](const parser::DoConstruct *crct) {
                        using A = parser::Statement<parser::NonLabelDoStmt>;
                        const auto &statement{std::get<A>(crct->t)};
                        EnterRegion(statement.source);
                        InitiateConstruct(&statement.statement);
                      },
                      [&](const parser::IfConstruct *crct) {
                        using A = parser::Statement<parser::IfThenStmt>;
                        InitiateConstruct(&std::get<A>(crct->t).statement);
                      },
                      [&](const parser::SelectRankConstruct *crct) {
                        using A = parser::Statement<parser::SelectRankStmt>;
                        const auto &statement{std::get<A>(crct->t)};
                        EnterRegion(statement.source);
                      },
                      [&](const parser::SelectTypeConstruct *crct) {
                        using A = parser::Statement<parser::SelectTypeStmt>;
                        const auto &statement{std::get<A>(crct->t)};
                        EnterRegion(statement.source);
                      },
                      [&](const parser::WhereConstruct *crct) {
                        using A = parser::Statement<parser::WhereConstructStmt>;
                        InitiateConstruct(&std::get<A>(crct->t).statement);
                      },
                      [&](const parser::ForallConstruct *crct) {
                        using A =
                            parser::Statement<parser::ForallConstructStmt>;
                        InitiateConstruct(&std::get<A>(crct->t).statement);
                      },
                      [](const parser::CriticalConstruct *) { /*fixme*/ },
                      [](const parser::CompilerDirective *) { /*fixme*/ },
                      [](const parser::OpenMPConstruct *) { /*fixme*/ },
                      [](const parser::OpenMPEndLoopDirective *) { /*fixme*/ },
                  },
                  con.u);
              auto next{iter};
              const auto &nextOp{*(++next)};
              if (auto *op{std::get_if<flat::LabelOp>(&nextOp.u)}) {
                blockMap_.insert({op->get(), builder_->GetInsertionPoint()});
                ++iter;
              }
            },
            [&](const flat::EndOp &con) {
              std::visit(
                  common::visitors{
                      [](const auto &) {},
                      [&](const parser::BlockConstruct *) { ExitRegion(); },
                      [&](const parser::DoConstruct *crct) {
                        const auto &statement{
                            std::get<parser::Statement<parser::NonLabelDoStmt>>(
                                crct->t)};
                        FinishConstruct(&statement.statement);
                        ExitRegion();
                      },
                      [&](const parser::AssociateConstruct *) { ExitRegion(); },
                      [&](const parser::ChangeTeamConstruct *) {
                        ExitRegion();
                      },
                      [&](const parser::SelectTypeConstruct *) {
                        ExitRegion();
                      },
                  },
                  con.u);
            },
        },
        op.u);
  }
}

Program *CreateFortranIR(const parser::Program &program,
    semantics::SemanticsContext &semanticsContext, bool debugLinearIR) {
  FortranIRLowering converter{semanticsContext, debugLinearIR};
  Walk(program, converter);
  return converter.program();
}

// debug channel
llvm::raw_ostream *debugChannel;

llvm::raw_ostream &DebugChannel() {
  return debugChannel ? *debugChannel : llvm::errs();
}

static void SetDebugChannel(llvm::raw_ostream *output) {
  debugChannel = output;
}

void SetDebugChannel(const std::string &filename) {
  std::error_code ec;
  SetDebugChannel(
      new llvm::raw_fd_ostream(filename, ec, llvm::sys::fs::F_None));
  CHECK(!ec);
}
}
