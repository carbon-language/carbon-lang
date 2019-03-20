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
Expression *ExprRef(const parser::Expr &a) { return a.typedExpr.get()->v; }
Expression *ExprRef(const common::Indirection<parser::Expr> &a) {
  return a.value().typedExpr.get()->v;
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
  flat::LabelRef defLab;
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

template<typename T>
void cleanupSwitchPairs(flat::LabelRef &defLab,
    std::vector<typename T::ValueType> &values,
    std::vector<flat::LabelRef> &labels) {
  CHECK(values.size() == labels.size());
  for (std::size_t i{0}, len{values.size()}; i < len; ++i) {
    if (IsDefault<T>(values[i])) {
      defLab = labels[i];
      for (std::size_t j{i}; j < len - 1; ++j) {
        values[j] = values[j + 1];
        labels[j] = labels[j + 1];
      }
      values.pop_back();
      labels.pop_back();
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
            [&](const parser::TypeSpec &typeSpec) {
              result.emplace_back(
                  SwitchTypeStmt::TypeSpec{typeSpec.declTypeSpec});
            },
            [&](const parser::DerivedTypeSpec &derivedTypeSpec) {
              result.emplace_back(
                  SwitchTypeStmt::DerivedTypeSpec{nullptr /*FIXME*/});
            },
            [&](const parser::Default &) {
              result.emplace_back(SwitchTypeStmt::Default{});
            },
        },
        guard.u);
  }
  return result;
}

static void buildMultiwayDefaultNext(SwitchArguments &result) {
  result.defLab = result.labels.back();
  result.labels.pop_back();
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
  using T = evaluate::Type<evaluate::TypeCategory::Logical, 1>;
  return {evaluate::AsGenericExpr(evaluate::Constant<T>{true})};
}

// create an integer constant as an expression
static Expression CreateConstant(int64_t value) {
  using T = evaluate::SubscriptInteger;
  return {evaluate::AsGenericExpr(evaluate::Constant<T>{value})};
}

static void CreateSwitchHelper(FIRBuilder *builder, Value condition,
    BasicBlock *defaultCase, const SwitchStmt::ValueSuccPairListType &rest) {
  builder->CreateSwitch(condition, defaultCase, rest);
}
static void CreateSwitchCaseHelper(FIRBuilder *builder, Value condition,
    BasicBlock *defaultCase,
    const SwitchCaseStmt::ValueSuccPairListType &rest) {
  builder->CreateSwitchCase(condition, defaultCase, rest);
}
static void CreateSwitchRankHelper(FIRBuilder *builder, Value condition,
    BasicBlock *defaultCase,
    const SwitchRankStmt::ValueSuccPairListType &rest) {
  builder->CreateSwitchRank(condition, defaultCase, rest);
}
static void CreateSwitchTypeHelper(FIRBuilder *builder, Value condition,
    BasicBlock *defaultCase,
    const SwitchTypeStmt::ValueSuccPairListType &rest) {
  builder->CreateSwitchType(condition, defaultCase, rest);
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

  template<typename T>
  void ProcessRoutine(const T &here, const std::string &name) {
    CHECK(!fir_->containsProcedure(name));
    auto *subp{fir_->getOrInsertProcedure(name, nullptr, {})};
    builder_ = new FIRBuilder(*CreateBlock(subp->getLastRegion()));
    AnalysisData ad;
#if 0
    ControlFlowAnalyzer linearize{linearOperations_, ad};
    Walk(here, linearize);
#else
    CreateFlatIR(here, linearOperations_, ad);
#endif
    if (debugLinearFIR_) {
      dump(linearOperations_);
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
    return builder_->CreateAddr(DataRefToExpression(dataRef));
  }
  Type CreateAllocationValue(const parser::Allocation *allocation,
      const parser::AllocateStmt *statement) {
    auto &obj{std::get<parser::AllocateObject>(allocation->t)};
    (void)obj;
    // TODO: build an expression for the allocation
    return nullptr;
  }
  AllocateInsn *CreateDeallocationValue(
      const parser::AllocateObject *allocateObject,
      const parser::DeallocateStmt *statement) {
    // TODO: build an expression for the deallocation
    return nullptr;
  }

  // IO argument translations ...
  IOCallArguments CreateBackspaceArguments(
      const std::list<parser::PositionOrFlushSpec> &specifiers) {
    return IOCallArguments{};
  }
  IOCallArguments CreateCloseArguments(
      const std::list<parser::CloseStmt::CloseSpec> &specifiers) {
    return IOCallArguments{};
  }
  IOCallArguments CreateEndfileArguments(
      const std::list<parser::PositionOrFlushSpec> &specifiers) {
    return IOCallArguments{};
  }
  IOCallArguments CreateFlushArguments(
      const std::list<parser::PositionOrFlushSpec> &specifiers) {
    return IOCallArguments{};
  }
  IOCallArguments CreateRewindArguments(
      const std::list<parser::PositionOrFlushSpec> &specifiers) {
    return IOCallArguments{};
  }
  IOCallArguments CreateInquireArguments(
      const std::list<parser::InquireSpec> &specifiers) {
    return IOCallArguments{};
  }
  IOCallArguments CreateInquireArguments(
      const parser::InquireStmt::Iolength &iolength) {
    return IOCallArguments{};
  }
  IOCallArguments CreateOpenArguments(
      const std::list<parser::ConnectSpec> &specifiers) {
    return IOCallArguments{};
  }
  IOCallArguments CreateWaitArguments(
      const std::list<parser::WaitSpec> &specifiers) {
    return IOCallArguments{};
  }
  IOCallArguments CreatePrintArguments(const parser::Format &format,
      const std::list<parser::OutputItem> &outputs) {
    return IOCallArguments{};
  }
  IOCallArguments CreateReadArguments(
      const std::optional<parser::IoUnit> &ioUnit,
      const std::optional<parser::Format> &format,
      const std::list<parser::IoControlSpec> &controls,
      const std::list<parser::InputItem> &inputs) {
    return IOCallArguments{};
  }
  IOCallArguments CreateWriteArguments(
      const std::optional<parser::IoUnit> &ioUnit,
      const std::optional<parser::Format> &format,
      const std::list<parser::IoControlSpec> &controls,
      const std::list<parser::OutputItem> &outputs) {
    return IOCallArguments{};
  }

  // Runtime argument translations ...
  RuntimeCallArguments CreateEventPostArguments(
      const parser::EventPostStmt &eventPostStatement) {
    return RuntimeCallArguments{};
  }
  RuntimeCallArguments CreateEventWaitArguments(
      const parser::EventWaitStmt &eventWaitStatement) {
    return RuntimeCallArguments{};
  }
  RuntimeCallArguments CreateFailImageArguments(
      const parser::FailImageStmt &failImageStatement) {
    return RuntimeCallArguments{};
  }
  RuntimeCallArguments CreateFormTeamArguments(
      const parser::FormTeamStmt &formTeamStatement) {
    return RuntimeCallArguments{};
  }
  RuntimeCallArguments CreateLockArguments(
      const parser::LockStmt &lockStatement) {
    return RuntimeCallArguments{};
  }
  RuntimeCallArguments CreatePauseArguments(
      const parser::PauseStmt &pauseStatement) {
    return RuntimeCallArguments{};
  }
  RuntimeCallArguments CreateStopArguments(
      const parser::StopStmt &stopStatement) {
    return RuntimeCallArguments{};
  }
  RuntimeCallArguments CreateSyncAllArguments(
      const parser::SyncAllStmt &syncAllStatement) {
    return RuntimeCallArguments{};
  }
  RuntimeCallArguments CreateSyncImagesArguments(
      const parser::SyncImagesStmt &syncImagesStatement) {
    return RuntimeCallArguments{};
  }
  RuntimeCallArguments CreateSyncMemoryArguments(
      const parser::SyncMemoryStmt &syncMemoryStatement) {
    return RuntimeCallArguments{};
  }
  RuntimeCallArguments CreateSyncTeamArguments(
      const parser::SyncTeamStmt &syncTeamStatement) {
    return RuntimeCallArguments{};
  }
  RuntimeCallArguments CreateUnlockArguments(
      const parser::UnlockStmt &unlockStatement) {
    return RuntimeCallArguments{};
  }

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
              return builder_->CreateExpr(VariableToExpression(v));
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
    const auto &x{std::get<parser::Scalar<parser::Expr>>(
        std::get<parser::Statement<parser::SelectCaseStmt>>(construct->t)
            .statement.t)};
    return builder_->CreateExpr(ExprRef(x.thing));
  }
  SwitchArguments ComposeSwitchArgs(const flat::SwitchOp &op) {
    SwitchArguments result{NOTHING, flat::unspecifiedLabel, {}, op.refs};
    std::visit(
        common::visitors{
            [&](const parser::ComputedGotoStmt *c) {
              const auto &e{std::get<parser::ScalarIntExpr>(c->t)};
              result.exp = builder_->CreateExpr(ExprRef(e.thing.thing));
              buildMultiwayDefaultNext(result);
            },
            [&](const parser::ArithmeticIfStmt *c) {
              result.exp =
                  builder_->CreateExpr(ExprRef(std::get<parser::Expr>(c->t)));
            },
            [&](const parser::CallStmt *c) {
              result.exp = NOTHING;  // fixme - result of call
              buildMultiwayDefaultNext(result);
            },
            [](const auto *) { WRONG_PATH(); },
        },
        op.u);
    return result;
  }
  SwitchCaseArguments ComposeSwitchCaseArguments(
      const parser::CaseConstruct *caseConstruct,
      const std::vector<flat::LabelRef> &refs) {
    auto &cases{
        std::get<std::list<parser::CaseConstruct::Case>>(caseConstruct->t)};
    SwitchCaseArguments result{GetSwitchCaseSelector(caseConstruct),
        flat::unspecifiedLabel, populateSwitchValues(builder_, cases),
        std::move(refs)};
    cleanupSwitchPairs<SwitchCaseStmt>(
        result.defLab, result.values, result.labels);
    return result;
  }
  SwitchRankArguments ComposeSwitchRankArguments(
      const parser::SelectRankConstruct *selectRankConstruct,
      const std::vector<flat::LabelRef> &refs) {
    auto &ranks{std::get<std::list<parser::SelectRankConstruct::RankCase>>(
        selectRankConstruct->t)};
    SwitchRankArguments result{GetSwitchRankSelector(selectRankConstruct),
        flat::unspecifiedLabel, populateSwitchValues(ranks), std::move(refs)};
    if (auto &name{GetSwitchAssociateName<parser::SelectRankStmt>(
            selectRankConstruct)}) {
      (void)name;  // get rid of warning
      // TODO: handle associate-name -> Add an assignment stmt?
    }
    cleanupSwitchPairs<SwitchRankStmt>(
        result.defLab, result.values, result.labels);
    return result;
  }
  SwitchTypeArguments ComposeSwitchTypeArguments(
      const parser::SelectTypeConstruct *selectTypeConstruct,
      const std::vector<flat::LabelRef> &refs) {
    auto &types{std::get<std::list<parser::SelectTypeConstruct::TypeCase>>(
        selectTypeConstruct->t)};
    SwitchTypeArguments result{GetSwitchTypeSelector(selectTypeConstruct),
        flat::unspecifiedLabel, populateSwitchValues(types), std::move(refs)};
    if (auto &name{GetSwitchAssociateName<parser::SelectTypeStmt>(
            selectTypeConstruct)}) {
      (void)name;  // get rid of warning
      // TODO: handle associate-name -> Add an assignment stmt?
    }
    cleanupSwitchPairs<SwitchTypeStmt>(
        result.defLab, result.values, result.labels);
    return result;
  }

  Expression VariableToExpression(const parser::Variable &var) {
    evaluate::ExpressionAnalyzer analyzer{semanticsContext_};
    return {std::move(analyzer.Analyze(var).value())};
  }
  Expression DataRefToExpression(const parser::DataRef &dr) {
    evaluate::ExpressionAnalyzer analyzer{semanticsContext_};
    return {std::move(analyzer.Analyze(dr).value())};
  }
  Expression NameToExpression(const parser::Name &name) {
    evaluate::ExpressionAnalyzer analyzer{semanticsContext_};
    return {std::move(analyzer.Analyze(name).value())};
  }
  Expression StructureComponentToExpression(
      const parser::StructureComponent &sc) {
    evaluate::ExpressionAnalyzer analyzer{semanticsContext_};
    return {std::move(analyzer.Analyze(sc).value())};
  }

  void handleIntrinsicAssignmentStmt(const parser::AssignmentStmt &stmt) {
    // TODO: check if allocation or reallocation should happen, etc.
    auto *value{builder_->CreateExpr(ExprRef(std::get<parser::Expr>(stmt.t)))};
    auto *addr{builder_->CreateAddr(
        VariableToExpression(std::get<parser::Variable>(stmt.t)))};
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
                          opts.stat = VariableToExpression(sv.v.thing.thing);
                        },
                        [&](const parser::MsgVariable &mv) {
                          opts.errmsg = VariableToExpression(mv.v.thing.thing);
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
                      std::get<std::list<parser::ActualArgSpec>>(
                          s.value().v.t)));
            },
            [&](const common::Indirection<parser::CloseStmt> &s) {
              builder_->CreateIOCall(
                  InputOutputCallClose, CreateCloseArguments(s.value().v));
            },
            [](const parser::ContinueStmt &) { WRONG_PATH(); },
            [](const common::Indirection<parser::CycleStmt> &) {
              WRONG_PATH();
            },
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
                          auto *s{builder_->CreateAddr(NameToExpression(n))};
                          builder_->CreateNullify(s);
                        },
                        [&](const parser::StructureComponent &sc) {
                          auto *s{builder_->CreateAddr(
                              StructureComponentToExpression(sc))};
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
              auto *addr{builder_->CreateAddr(
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
            [](const common::Indirection<parser::ReturnStmt> &) {
              WRONG_PATH();
            },
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
              auto *addr{builder_->CreateAddr(
                  NameToExpression(std::get<parser::Name>(s.value().t)))};
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
  void handleLinearAction(const flat::ActionOp &action, AnalysisData &ad) {
    handleActionStatement(ad, *action.v);
  }

  // DO loop handlers
  struct DoBoundsInfo {
    Statement *doVariable;
    Statement *lowerBound;
    Statement *upperBound;
    Statement *stepExpr;
    Statement *condition;
  };
  void PushDoContext(const parser::NonLabelDoStmt *doStmt, Statement *doVar,
      Statement *lowBound, Statement *upBound, Statement *stepExp) {
    doMap_.emplace(doStmt, DoBoundsInfo{doVar, lowBound, upBound, stepExp});
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

  // do_var = do_var + e3
  void handleLinearDoIncrement(const flat::DoIncrementOp &inc) {
    auto *info{GetBoundsInfo(inc)};
    auto *var{builder_->CreateLoad(info->doVariable)};
    builder_->CreateIncrement(var, info->stepExpr);
  }

  // (e3 > 0 && do_var <= e2) || (e3 < 0 && do_var >= e2)
  void handleLinearDoCompare(const flat::DoCompareOp &cmp) {
    auto *info{GetBoundsInfo(cmp)};
    auto *var{builder_->CreateLoad(info->doVariable)};
    auto *cond{
        builder_->CreateDoCondition(info->stepExpr, var, info->upperBound)};
    info->condition = cond;
  }

  // InitiateConstruct - many constructs require some initial setup
  void InitiateConstruct(const parser::AssociateStmt *stmt) {
    for (auto &assoc : std::get<std::list<parser::Association>>(stmt->t)) {
      auto &selector{std::get<parser::Selector>(assoc.t)};
      auto *expr{builder_->CreateExpr(std::visit(
          common::visitors{
              [&](const parser::Variable &v) {
                return VariableToExpression(v);
              },
              [](const parser::Expr &e) { return *ExprRef(e); },
          },
          selector.u))};
      auto *name{builder_->CreateAddr(
          NameToExpression(std::get<parser::Name>(assoc.t)))};
      builder_->CreateStore(name, expr);
    }
  }
  void InitiateConstruct(const parser::SelectCaseStmt *stmt) {
    builder_->CreateExpr(
        ExprRef(std::get<parser::Scalar<parser::Expr>>(stmt->t).thing));
  }
  void InitiateConstruct(const parser::ChangeTeamStmt *changeTeamStmt) {
    // FIXME
  }
  void InitiateConstruct(const parser::IfThenStmt *ifThenStmt) {
    const auto &e{std::get<parser::ScalarLogicalExpr>(ifThenStmt->t).thing};
    builder_->CreateExpr(ExprRef(e.thing));
  }
  void InitiateConstruct(const parser::WhereConstructStmt *whereConstructStmt) {
    const auto &e{std::get<parser::LogicalExpr>(whereConstructStmt->t)};
    builder_->CreateExpr(ExprRef(e.thing));
  }
  void InitiateConstruct(
      const parser::ForallConstructStmt *forallConstructStmt) {
    // FIXME
  }

  void InitiateConstruct(const parser::NonLabelDoStmt *stmt) {
    auto &ctrl{std::get<std::optional<parser::LoopControl>>(stmt->t)};
    if (ctrl.has_value()) {
      std::visit(
          common::visitors{
              [&](const parser::LoopBounds<parser::ScalarIntExpr> &bounds) {
                auto *var = builder_->CreateAddr(
                    NameToExpression(bounds.name.thing.thing));
                // evaluate e1, e2 [, e3] ...
                auto *e1{
                    builder_->CreateExpr(ExprRef(bounds.lower.thing.thing))};
                auto *e2{
                    builder_->CreateExpr(ExprRef(bounds.upper.thing.thing))};
                Statement *e3;
                if (bounds.step.has_value()) {
                  e3 = builder_->CreateExpr(ExprRef(bounds.step->thing.thing));
                } else {
                  e3 = builder_->CreateExpr(CreateConstant(1));
                }
                builder_->CreateStore(var, e1);
                PushDoContext(stmt, var, e1, e2, e3);
              },
              [&](const parser::ScalarLogicalExpr &whileExpr) {},
              [&](const parser::LoopControl::Concurrent &cc) {},
          },
          ctrl->u);
    } else {
      // loop forever
    }
  }

  // finish DO construct construction
  void FinishConstruct(const parser::NonLabelDoStmt *stmt) {
    auto &ctrl{std::get<std::optional<parser::LoopControl>>(stmt->t)};
    if (ctrl.has_value()) {
      std::visit(
          common::visitors{
              [&](const parser::LoopBounds<parser::ScalarIntExpr> &) {
                PopDoContext(stmt);
              },
              [&](auto &) {
                // do nothing
              },
          },
          ctrl->u);
    }
  }

  Statement *BuildLoopLatchExpression(const parser::NonLabelDoStmt *stmt) {
    auto &loopCtrl{std::get<std::optional<parser::LoopControl>>(stmt->t)};
    if (loopCtrl.has_value()) {
      return std::visit(
          common::visitors{
              [&](const parser::LoopBounds<parser::ScalarIntExpr> &) {
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

  void ConstructFIR(AnalysisData &ad) {
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
                          builder_->CreateRuntimeCall(RuntimeCallFailImage,
                              CreateFailImageArguments(*s));
                          builder_->CreateUnreachable();
                        },
                        [&](const parser::ReturnStmt *s) {
                          // alt-return
                          if (s->v) {
                            auto *app{builder_->CreateExpr(
                                ExprRef(s->v->thing.thing))};
                            builder_->CreateReturn(app);
                          } else {
                            auto *zero{builder_->CreateExpr(CreateConstant(0))};
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
                          const auto &exp{std::get<parser::ScalarLogicalExpr>(
                              s->statement.t)
                                              .thing.thing.value()};
                          SEMANTICS_CHECK(ExprRef(exp),
                              "IF THEN condition expression missing");
                          auto *cond{builder_->CreateExpr(ExprRef(exp))};
                          AddOrQueueCGoto(cond, cop.trueLabel, cop.falseLabel);
                        },
                        [&](const parser::Statement<parser::ElseIfStmt> *s) {
                          const auto &exp{std::get<parser::ScalarLogicalExpr>(
                              s->statement.t)
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
                        [&](const parser::Statement<parser::NonLabelDoStmt>
                                *s) {
                          AddOrQueueCGoto(
                              BuildLoopLatchExpression(&s->statement),
                              cop.trueLabel, cop.falseLabel);
                        }},
                    cop.u);
                builder_->ClearInsertionPoint();
              },
              [&](const flat::SwitchIOOp &IOp) {
                CheckInsertionPoint();
                AddOrQueueSwitch<SwitchStmt>(
                    NOTHING, IOp.next, {}, {}, CreateSwitchHelper);
                builder_->ClearInsertionPoint();
              },
              [&](const flat::SwitchOp &sop) {
                CheckInsertionPoint();
                std::visit(
                    common::visitors{
                        [&](auto) {
                          auto args{ComposeSwitchArgs(sop)};
                          AddOrQueueSwitch<SwitchStmt>(args.exp, args.defLab,
                              args.values, args.labels, CreateSwitchHelper);
                        },
                        [&](const parser::CaseConstruct *caseConstruct) {
                          auto args{ComposeSwitchCaseArguments(
                              caseConstruct, sop.refs)};
                          AddOrQueueSwitch<SwitchCaseStmt>(args.exp,
                              args.defLab, args.values, args.labels,
                              CreateSwitchCaseHelper);
                        },
                        [&](const parser::SelectRankConstruct
                                *selectRankConstruct) {
                          auto args{ComposeSwitchRankArguments(
                              selectRankConstruct, sop.refs)};
                          AddOrQueueSwitch<SwitchRankStmt>(args.exp,
                              args.defLab, args.values, args.labels,
                              CreateSwitchRankHelper);
                        },
                        [&](const parser::SelectTypeConstruct
                                *selectTypeConstruct) {
                          auto args{ComposeSwitchTypeArguments(
                              selectTypeConstruct, sop.refs)};
                          AddOrQueueSwitch<SwitchTypeStmt>(args.exp,
                              args.defLab, args.values, args.labels,
                              CreateSwitchTypeHelper);
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
                          const auto &statement{std::get<
                              parser::Statement<parser::AssociateStmt>>(
                              crct->t)};
                          const auto &position{statement.source};
                          EnterRegion(position);
                          InitiateConstruct(&statement.statement);
                        },
                        [&](const parser::BlockConstruct *crct) {
                          EnterRegion(
                              std::get<parser::Statement<parser::BlockStmt>>(
                                  crct->t)
                                  .source);
                        },
                        [&](const parser::CaseConstruct *crct) {
                          InitiateConstruct(
                              &std::get<
                                  parser::Statement<parser::SelectCaseStmt>>(
                                  crct->t)
                                   .statement);
                        },
                        [&](const parser::ChangeTeamConstruct *crct) {
                          const auto &statement{std::get<
                              parser::Statement<parser::ChangeTeamStmt>>(
                              crct->t)};
                          EnterRegion(statement.source);
                          InitiateConstruct(&statement.statement);
                        },
                        [&](const parser::DoConstruct *crct) {
                          const auto &statement{std::get<
                              parser::Statement<parser::NonLabelDoStmt>>(
                              crct->t)};
                          EnterRegion(statement.source);
                          InitiateConstruct(&statement.statement);
                        },
                        [&](const parser::IfConstruct *crct) {
                          InitiateConstruct(
                              &std::get<parser::Statement<parser::IfThenStmt>>(
                                  crct->t)
                                   .statement);
                        },
                        [&](const parser::SelectRankConstruct *crct) {
                          const auto &statement{std::get<
                              parser::Statement<parser::SelectRankStmt>>(
                              crct->t)};
                          EnterRegion(statement.source);
                        },
                        [&](const parser::SelectTypeConstruct *crct) {
                          const auto &statement{std::get<
                              parser::Statement<parser::SelectTypeStmt>>(
                              crct->t)};
                          EnterRegion(statement.source);
                        },
                        [&](const parser::WhereConstruct *crct) {
                          InitiateConstruct(
                              &std::get<parser::Statement<
                                   parser::WhereConstructStmt>>(crct->t)
                                   .statement);
                        },
                        [&](const parser::ForallConstruct *crct) {
                          InitiateConstruct(
                              &std::get<parser::Statement<
                                   parser::ForallConstructStmt>>(crct->t)
                                   .statement);
                        },
                        [](const parser::CriticalConstruct *) { /*fixme*/ },
                        [](const parser::CompilerDirective *) { /*fixme*/ },
                        [](const parser::OpenMPConstruct *) { /*fixme*/ },
                        [](const parser::OpenMPEndLoopDirective
                                *) { /*fixme*/ },
                    },
                    con.u);
                auto next{iter};
                const auto &nextOp{*(++next)};
                std::visit(
                    common::visitors{
                        [](const auto &) {},
                        [&](const flat::LabelOp &op) {
                          blockMap_.insert(
                              {op.get(), builder_->GetInsertionPoint()});
                          ++iter;
                        },
                    },
                    nextOp.u);
              },
              [&](const flat::EndOp &con) {
                std::visit(
                    common::visitors{
                        [](const auto &) {},
                        [&](const parser::BlockConstruct *) { ExitRegion(); },
                        [&](const parser::DoConstruct *crct) {
                          const auto &statement{std::get<
                              parser::Statement<parser::NonLabelDoStmt>>(
                              crct->t)};
                          FinishConstruct(&statement.statement);
                          ExitRegion();
                        },
                        [&](const parser::AssociateConstruct *) {
                          ExitRegion();
                        },
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

  void AddOrQueueBranch(flat::LabelRef dest) {
    auto iter{blockMap_.find(dest)};
    if (iter != blockMap_.end()) {
      builder_->CreateBranch(iter->second);
    } else {
      using namespace std::placeholders;
      controlFlowEdgesToAdd_.emplace_back(std::bind(
          [](FIRBuilder *builder, BasicBlock *block, flat::LabelRef dest,
              const LabelMapType &map) {
            builder->SetInsertionPoint(block);
            CHECK(map.find(dest) != map.end());
            builder->CreateBranch(map.find(dest)->second);
          },
          builder_, builder_->GetInsertionPoint(), dest, _1));
    }
  }

  void AddOrQueueCGoto(Statement *condition, flat::LabelRef trueBlock,
      flat::LabelRef falseBlock) {
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

  template<typename SWITCHTYPE, typename F>
  void AddOrQueueSwitch(Value condition, flat::LabelRef defaultLabel,
      const std::vector<typename SWITCHTYPE::ValueType> &values,
      const std::vector<flat::LabelRef> &labels, F function) {
    auto defer{false};
    auto defaultIter{blockMap_.find(defaultLabel)};
    typename SWITCHTYPE::ValueSuccPairListType cases;
    if (defaultIter == blockMap_.end()) {
      defer = true;
    } else {
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
    }
    if (defer) {
      using namespace std::placeholders;
      controlFlowEdgesToAdd_.emplace_back(std::bind(
          [](FIRBuilder *builder, BasicBlock *block, Value expr,
              flat::LabelRef defaultDest,
              const std::vector<typename SWITCHTYPE::ValueType> &values,
              const std::vector<flat::LabelRef> &labels, F function,
              const LabelMapType &map) {
            builder->SetInsertionPoint(block);
            typename SWITCHTYPE::ValueSuccPairListType cases;
            auto valiter{values.begin()};
            for (auto &lab : labels) {
              cases.emplace_back(*valiter++, map.find(lab)->second);
            }
            function(builder, expr, map.find(defaultDest)->second, cases);
          },
          builder_, builder_->GetInsertionPoint(), condition, defaultLabel,
          values, labels, function, _1));
    } else {
      function(builder_, condition, defaultIter->second, cases);
    }
  }

  Variable *ConvertToVariable(const semantics::Symbol *symbol) {
    // FIXME: how to convert semantics::Symbol to evaluate::Variable?
    return new Variable(symbol);
  }

  void AddOrQueueIGoto(AnalysisData &ad, const semantics::Symbol *symbol,
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
              const std::vector<flat::LabelRef> &fixme,
              const LabelMapType &map) {
            builder->SetInsertionPoint(block);
            builder->CreateIndirectBr(variable, {});  // FIXME
          },
          builder_, builder_->GetInsertionPoint(), nullptr /*symbol*/,
          useLabels, _1));
    } else {
      builder_->CreateIndirectBr(ConvertToVariable(symbol), blocks);
    }
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
