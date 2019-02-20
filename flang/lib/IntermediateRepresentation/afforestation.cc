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
#include "mixin.h"
#include "../evaluate/fold.h"
#include "../evaluate/tools.h"
#include "../parser/parse-tree-visitor.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

namespace Fortran::IntermediateRepresentation {

static llvm::raw_ostream *debugChannel;
static llvm::raw_ostream &DebugChannel() {
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

namespace {
struct LinearOp;

using LinearLabelRef = unsigned;
constexpr LinearLabelRef unspecifiedLabel{~0u};

struct LinearLabelBuilder {
  LinearLabelBuilder() : referenced(32), counter{0u} {}
  LinearLabelRef getNext() {
    LinearLabelRef next{counter++};
    auto cap{referenced.capacity()};
    if (cap < counter) referenced.reserve(2 * cap);
    referenced[next] = false;
    return next;
  }
  void setReferenced(LinearLabelRef label) { referenced[label] = true; }
  bool isReferenced(LinearLabelRef label) const { return referenced[label]; }
  std::vector<bool> referenced;
  unsigned counter;
};

struct LinearLabel {
  explicit LinearLabel(LinearLabelBuilder &builder)
    : builder_{builder}, label_{builder.getNext()} {}
  LinearLabel(const LinearLabel &that)
    : builder_{that.builder_}, label_{that.label_} {}
  LinearLabel &operator=(const LinearLabel &that) {
    CHECK(&builder_ == &that.builder_);
    label_ = that.label_;
    return *this;
  }
  void setReferenced() const { builder_.setReferenced(label_); }
  bool isReferenced() const { return builder_.isReferenced(label_); }
  LinearLabelRef get() const { return label_; }
  operator LinearLabelRef() const { return get(); }

private:
  LinearLabelBuilder &builder_;
  LinearLabelRef label_;
};

struct LinearGoto {
  struct LinearArtificial {};
  LinearGoto(LinearLabelRef dest) : u{LinearArtificial{}}, target{dest} {}
  template<typename T>
  LinearGoto(const T &stmt, LinearLabelRef dest) : u{&stmt}, target{dest} {}
  std::variant<const parser::CycleStmt *, const parser::ExitStmt *,
      const parser::GotoStmt *, LinearArtificial>
      u;
  LinearLabelRef target;
};

struct LinearReturn
  : public SumTypeCopyMixin<std::variant<const parser::FailImageStmt *,
        const parser::ReturnStmt *, const parser::StopStmt *>> {
  SUM_TYPE_COPY_MIXIN(LinearReturn)
  template<typename T> LinearReturn(const T &stmt) : SumTypeCopyMixin{&stmt} {}
};

struct LinearConditionalGoto {
  template<typename T>
  LinearConditionalGoto(const T &cond, LinearLabelRef tb, LinearLabelRef fb)
    : u{&cond}, trueLabel{tb}, falseLabel{fb} {}
  std::variant<const parser::Statement<parser::IfThenStmt> *,
      const parser::Statement<parser::ElseIfStmt> *, const parser::IfStmt *,
      const parser::Statement<parser::NonLabelDoStmt> *>
      u;
  LinearLabelRef trueLabel;
  LinearLabelRef falseLabel;
};

struct LinearIndirectGoto {
  LinearIndirectGoto(
      const semantics::Symbol *symbol, std::vector<LinearLabelRef> &&labelRefs)
    : symbol{symbol}, labelRefs{labelRefs} {}
  const semantics::Symbol *symbol;
  std::vector<LinearLabelRef> labelRefs;
};

struct LinearSwitchingIO {
  template<typename T>
  LinearSwitchingIO(const T &io, LinearLabelRef next,
      std::optional<LinearLabelRef> errLab,
      std::optional<LinearLabelRef> eorLab = std::nullopt,
      std::optional<LinearLabelRef> endLab = std::nullopt)
    : u{&io}, next{next}, errLabel{errLab}, eorLabel{eorLab}, endLabel{endLab} {
  }
  std::variant<const parser::ReadStmt *, const parser::WriteStmt *,
      const parser::WaitStmt *, const parser::OpenStmt *,
      const parser::CloseStmt *, const parser::BackspaceStmt *,
      const parser::EndfileStmt *, const parser::RewindStmt *,
      const parser::FlushStmt *, const parser::InquireStmt *>
      u;
  LinearLabelRef next;
  std::optional<LinearLabelRef> errLabel;
  std::optional<LinearLabelRef> eorLabel;
  std::optional<LinearLabelRef> endLabel;
};

struct LinearSwitch {
  template<typename T>
  LinearSwitch(const T &sw, const std::vector<LinearLabelRef> &refs)
    : u{&sw}, refs{refs} {}
  std::variant<const parser::CallStmt *, const parser::ComputedGotoStmt *,
      const parser::ArithmeticIfStmt *, const parser::CaseConstruct *,
      const parser::SelectRankConstruct *, const parser::SelectTypeConstruct *>
      u;
  const std::vector<LinearLabelRef> refs;
};

struct LinearAction {
  LinearAction(const parser::Statement<parser::ActionStmt> &stmt) : v{&stmt} {}
  parser::CharBlock getSource() const { return v->source; }

  const parser::Statement<parser::ActionStmt> *v;
};

using ConstructVariant = std::variant<const parser::AssociateConstruct *,
    const parser::BlockConstruct *, const parser::CaseConstruct *,
    const parser::ChangeTeamConstruct *, const parser::CriticalConstruct *,
    const parser::DoConstruct *, const parser::IfConstruct *,
    const parser::SelectRankConstruct *, const parser::SelectTypeConstruct *,
    const parser::WhereConstruct *, const parser::ForallConstruct *,
    const parser::CompilerDirective *, const parser::OpenMPConstruct *,
    const parser::OpenMPEndLoopDirective *>;

struct LinearBeginConstruct : public SumTypeCopyMixin<ConstructVariant> {
  SUM_TYPE_COPY_MIXIN(LinearBeginConstruct)
  template<typename T>
  LinearBeginConstruct(const T &c) : SumTypeCopyMixin{&c} {}
};
struct LinearEndConstruct : public SumTypeCopyMixin<ConstructVariant> {
  SUM_TYPE_COPY_MIXIN(LinearEndConstruct)
  template<typename T> LinearEndConstruct(const T &c) : SumTypeCopyMixin{&c} {}
};

template<typename CONSTRUCT>
const char *GetConstructName(const CONSTRUCT &construct) {
  return std::visit(
      common::visitors{
          [](const parser::AssociateConstruct *) { return "ASSOCIATE"; },
          [](const parser::BlockConstruct *) { return "BLOCK"; },
          [](const parser::CaseConstruct *) { return "SELECT CASE"; },
          [](const parser::ChangeTeamConstruct *) { return "CHANGE TEAM"; },
          [](const parser::CriticalConstruct *) { return "CRITICAL"; },
          [](const parser::DoConstruct *) { return "DO"; },
          [](const parser::IfConstruct *) { return "IF"; },
          [](const parser::SelectRankConstruct *) { return "SELECT RANK"; },
          [](const parser::SelectTypeConstruct *) { return "SELECT TYPE"; },
          [](const parser::WhereConstruct *) { return "WHERE"; },
          [](const parser::ForallConstruct *) { return "FORALL"; },
          [](const parser::CompilerDirective *) { return "<directive>"; },
          [](const parser::OpenMPConstruct *) { return "<open-mp>"; },
          [](const parser::OpenMPEndLoopDirective *) {
            return "<open-mp-end-loop>";
          }},
      construct.u);
}

struct AnalysisData {
  std::map<parser::Label, LinearLabel> labelMap;
  std::vector<std::tuple<const parser::Name *, LinearLabelRef, LinearLabelRef>>
      nameStack;
  LinearLabelBuilder labelBuilder;
  std::map<const semantics::Symbol *, std::set<parser::Label>> assignMap;
};

void AddAssign(AnalysisData &ad, const semantics::Symbol *symbol,
    const parser::Label &label) {
  ad.assignMap[symbol].insert(label);
}
std::vector<LinearLabelRef> GetAssign(
    AnalysisData &ad, const semantics::Symbol *symbol) {
  std::vector<LinearLabelRef> result;
  for (auto lab : ad.assignMap[symbol]) {
    result.emplace_back(lab);
  }
  return result;
}
LinearLabel BuildNewLabel(AnalysisData &ad) {
  return LinearLabel{ad.labelBuilder};
}
LinearLabel FetchLabel(AnalysisData &ad, const parser::Label &label) {
  auto iter{ad.labelMap.find(label)};
  if (iter == ad.labelMap.end()) {
    LinearLabel ll{ad.labelBuilder};
    ll.setReferenced();
    ad.labelMap.insert({label, ll});
    return ll;
  }
  return iter->second;
}
std::tuple<const parser::Name *, LinearLabelRef, LinearLabelRef> FindStack(
    const std::vector<std::tuple<const parser::Name *, LinearLabelRef,
        LinearLabelRef>> &stack,
    const parser::Name *key) {
  for (auto iter{stack.rbegin()}, iend{stack.rend()}; iter != iend; ++iter) {
    if (std::get<0>(*iter) == key) return *iter;
  }
  SEMANTICS_FAILED("construct name not on stack");
  return {};
}
template<typename T> parser::Label GetErr(const T &stmt) {
  if constexpr (std::is_same_v<T, parser::ReadStmt> ||
      std::is_same_v<T, parser::WriteStmt>) {
    for (const auto &control : stmt.controls) {
      if (std::holds_alternative<parser::ErrLabel>(control.u)) {
        return std::get<parser::ErrLabel>(control.u).v;
      }
    }
  }
  if constexpr (std::is_same_v<T, parser::WaitStmt> ||
      std::is_same_v<T, parser::OpenStmt> ||
      std::is_same_v<T, parser::CloseStmt> ||
      std::is_same_v<T, parser::BackspaceStmt> ||
      std::is_same_v<T, parser::EndfileStmt> ||
      std::is_same_v<T, parser::RewindStmt> ||
      std::is_same_v<T, parser::FlushStmt>) {
    for (const auto &spec : stmt.v) {
      if (std::holds_alternative<parser::ErrLabel>(spec.u)) {
        return std::get<parser::ErrLabel>(spec.u).v;
      }
    }
  }
  if constexpr (std::is_same_v<T, parser::InquireStmt>) {
    for (const auto &spec : std::get<std::list<parser::InquireSpec>>(stmt.u)) {
      if (std::holds_alternative<parser::ErrLabel>(spec.u)) {
        return std::get<parser::ErrLabel>(spec.u).v;
      }
    }
  }
  return 0;
}
template<typename T> parser::Label GetEor(const T &stmt) {
  if constexpr (std::is_same_v<T, parser::ReadStmt> ||
      std::is_same_v<T, parser::WriteStmt>) {
    for (const auto &control : stmt.controls) {
      if (std::holds_alternative<parser::EorLabel>(control.u)) {
        return std::get<parser::EorLabel>(control.u).v;
      }
    }
  }
  if constexpr (std::is_same_v<T, parser::WaitStmt>) {
    for (const auto &waitSpec : stmt.v) {
      if (std::holds_alternative<parser::EorLabel>(waitSpec.u)) {
        return std::get<parser::EorLabel>(waitSpec.u).v;
      }
    }
  }
  return 0;
}
template<typename T> parser::Label GetEnd(const T &stmt) {
  if constexpr (std::is_same_v<T, parser::ReadStmt> ||
      std::is_same_v<T, parser::WriteStmt>) {
    for (const auto &control : stmt.controls) {
      if (std::holds_alternative<parser::EndLabel>(control.u)) {
        return std::get<parser::EndLabel>(control.u).v;
      }
    }
  }
  if constexpr (std::is_same_v<T, parser::WaitStmt>) {
    for (const auto &waitSpec : stmt.v) {
      if (std::holds_alternative<parser::EndLabel>(waitSpec.u)) {
        return std::get<parser::EndLabel>(waitSpec.u).v;
      }
    }
  }
  return 0;
}
template<typename T>
void errLabelSpec(const T &s, std::list<LinearOp> &ops,
    const parser::Statement<parser::ActionStmt> &ec, AnalysisData &ad) {
  if (auto errLab{GetErr(s)}) {
    std::optional<LinearLabelRef> errRef{FetchLabel(ad, errLab).get()};
    LinearLabel next{BuildNewLabel(ad)};
    ops.emplace_back(LinearSwitchingIO{s, next, errRef});
    ops.emplace_back(next);
  } else {
    ops.emplace_back(LinearAction{ec});
  }
}
template<typename T>
void threeLabelSpec(const T &s, std::list<LinearOp> &ops,
    const parser::Statement<parser::ActionStmt> &ec, AnalysisData &ad) {
  auto errLab{GetErr(s)};
  auto eorLab{GetEor(s)};
  auto endLab{GetEnd(s)};
  if (errLab || eorLab || endLab) {
    std::optional<LinearLabelRef> errRef{std::nullopt};
    if (errLab) errRef = FetchLabel(ad, errLab).get();
    std::optional<LinearLabelRef> eorRef{std::nullopt};
    if (eorLab) eorRef = FetchLabel(ad, eorLab).get();
    std::optional<LinearLabelRef> endRef{std::nullopt};
    if (endLab) endRef = FetchLabel(ad, endLab).get();
    LinearLabel next{BuildNewLabel(ad)};
    ops.emplace_back(LinearSwitchingIO{s, next, errRef, eorRef, endRef});
    ops.emplace_back(next);
  } else {
    ops.emplace_back(LinearAction{ec});
  }
}
template<typename T>
std::vector<LinearLabelRef> toLabelRef(AnalysisData &ad, const T &labels) {
  std::vector<LinearLabelRef> result;
  for (auto label : labels) {
    result.emplace_back(FetchLabel(ad, label).get());
  }
  CHECK(result.size() == labels.size());
  return result;
}

bool hasAltReturns(const parser::CallStmt &callStmt) {
  const auto &args{std::get<std::list<parser::ActualArgSpec>>(callStmt.v.t)};
  for (const auto &arg : args) {
    const auto &actual{std::get<parser::ActualArg>(arg.t)};
    if (std::holds_alternative<parser::AltReturnSpec>(actual.u)) {
      return true;
    }
  }
  return false;
}
std::list<parser::Label> getAltReturnLabels(const parser::Call &call) {
  std::list<parser::Label> result;
  const auto &args{std::get<std::list<parser::ActualArgSpec>>(call.t)};
  for (const auto &arg : args) {
    const auto &actual{std::get<parser::ActualArg>(arg.t)};
    if (const auto *p{std::get_if<parser::AltReturnSpec>(&actual.u)}) {
      result.push_back(p->v);
    }
  }
  return result;
}
LinearLabelRef NearestEnclosingDoConstruct(AnalysisData &ad) {
  for (auto iterator{ad.nameStack.rbegin()}, endIterator{ad.nameStack.rend()};
       iterator != endIterator; ++iterator) {
    auto labelReference{std::get<2>(*iterator)};
    if (labelReference != unspecifiedLabel) {
      return labelReference;
    }
  }
  SEMANTICS_FAILED("CYCLE|EXIT not in loop");
  return unspecifiedLabel;
}

struct LinearOp
  : public SumTypeMixin<std::variant<LinearLabel, LinearGoto, LinearReturn,
        LinearConditionalGoto, LinearSwitchingIO, LinearSwitch, LinearAction,
        LinearBeginConstruct, LinearEndConstruct, LinearIndirectGoto>> {
  template<typename T> LinearOp(const T &thing) : SumTypeMixin{thing} {}
  void dump() const;
  static void Build(std::list<LinearOp> &ops,
      const parser::Statement<parser::ActionStmt> &ec, AnalysisData &ad) {
    std::visit(
        common::visitors{
            [&](const auto &s) { ops.emplace_back(LinearAction{ec}); },
            [&](const common::Indirection<parser::CallStmt> &s) {
              if (hasAltReturns(*s)) {
                auto next{BuildNewLabel(ad)};
                auto labels{toLabelRef(ad, getAltReturnLabels(s->v))};
                labels.push_back(next);
                ops.emplace_back(LinearSwitch{*s, std::move(labels)});
                ops.emplace_back(next);
              } else {
                ops.emplace_back(LinearAction{ec});
              }
            },
            [&](const common::Indirection<parser::AssignStmt> &s) {
              AddAssign(ad, std::get<parser::Name>(s->t).symbol,
                  std::get<parser::Label>(s->t));
              ops.emplace_back(LinearAction{ec});
            },
            [&](const common::Indirection<parser::CycleStmt> &s) {
              ops.emplace_back(LinearGoto{*s,
                  s->v ? std::get<2>(FindStack(ad.nameStack, &*s->v))
                       : NearestEnclosingDoConstruct(ad)});
            },
            [&](const common::Indirection<parser::ExitStmt> &s) {
              ops.emplace_back(LinearGoto{*s,
                  s->v ? std::get<1>(FindStack(ad.nameStack, &*s->v))
                       : NearestEnclosingDoConstruct(ad)});
            },
            [&](const common::Indirection<parser::GotoStmt> &s) {
              ops.emplace_back(LinearGoto{*s, FetchLabel(ad, s->v).get()});
            },
            [&](const parser::FailImageStmt &s) {
              ops.emplace_back(LinearReturn{s});
            },
            [&](const common::Indirection<parser::ReturnStmt> &s) {
              ops.emplace_back(LinearReturn{*s});
            },
            [&](const common::Indirection<parser::StopStmt> &s) {
              ops.emplace_back(LinearAction{ec});
              ops.emplace_back(LinearReturn{*s});
            },
            [&](const common::Indirection<const parser::ReadStmt> &s) {
              threeLabelSpec(*s, ops, ec, ad);
            },
            [&](const common::Indirection<const parser::WriteStmt> &s) {
              threeLabelSpec(*s, ops, ec, ad);
            },
            [&](const common::Indirection<const parser::WaitStmt> &s) {
              threeLabelSpec(*s, ops, ec, ad);
            },
            [&](const common::Indirection<const parser::OpenStmt> &s) {
              errLabelSpec(*s, ops, ec, ad);
            },
            [&](const common::Indirection<const parser::CloseStmt> &s) {
              errLabelSpec(*s, ops, ec, ad);
            },
            [&](const common::Indirection<const parser::BackspaceStmt> &s) {
              errLabelSpec(*s, ops, ec, ad);
            },
            [&](const common::Indirection<const parser::EndfileStmt> &s) {
              errLabelSpec(*s, ops, ec, ad);
            },
            [&](const common::Indirection<const parser::RewindStmt> &s) {
              errLabelSpec(*s, ops, ec, ad);
            },
            [&](const common::Indirection<const parser::FlushStmt> &s) {
              errLabelSpec(*s, ops, ec, ad);
            },
            [&](const common::Indirection<const parser::InquireStmt> &s) {
              errLabelSpec(*s, ops, ec, ad);
            },
            [&](const common::Indirection<parser::ComputedGotoStmt> &s) {
              auto next{BuildNewLabel(ad)};
              auto labels{
                  toLabelRef(ad, std::get<std::list<parser::Label>>(s->t))};
              labels.push_back(next);
              ops.emplace_back(LinearSwitch{*s, std::move(labels)});
              ops.emplace_back(next);
            },
            [&](const common::Indirection<parser::ArithmeticIfStmt> &s) {
              ops.emplace_back(LinearSwitch{*s,
                  toLabelRef(ad,
                      std::list{std::get<1>(s->t), std::get<2>(s->t),
                          std::get<3>(s->t)})});
            },
            [&](const common::Indirection<parser::AssignedGotoStmt> &s) {
              ops.emplace_back(LinearIndirectGoto{
                  std::get<parser::Name>(s->t).symbol,
                  toLabelRef(ad, std::get<std::list<parser::Label>>(s->t))});
            },
            [&](const common::Indirection<parser::IfStmt> &s) {
              auto then{BuildNewLabel(ad)};
              auto endif{BuildNewLabel(ad)};
              ops.emplace_back(LinearConditionalGoto{*s, then, endif});
              ops.emplace_back(then);
              ops.emplace_back(LinearAction{ec});
              ops.emplace_back(endif);
            },
        },
        ec.statement.u);
  }
};

template<typename STMTTYPE, typename CT>
Evaluation GetSwitchSelector(const CT *selectConstruct) {
  const auto &selector{std::get<parser::Selector>(
      std::get<parser::Statement<STMTTYPE>>(selectConstruct->t).statement.t)};
  return std::visit(common::visitors{
                        [](const parser::Expr &expression) {
                          return Evaluation{expression.typedExpr.get()};
                        },
                        [](const parser::Variable &variable) {
                          return Evaluation{&variable};
                        },
                    },
      selector.u);
}
Evaluation GetSwitchRankSelector(
    const parser::SelectRankConstruct *selectRankConstruct) {
  return GetSwitchSelector<parser::SelectRankStmt>(selectRankConstruct);
}
Evaluation GetSwitchTypeSelector(
    const parser::SelectTypeConstruct *selectTypeConstruct) {
  return GetSwitchSelector<parser::SelectTypeStmt>(selectTypeConstruct);
}
Evaluation GetSwitchCaseSelector(const parser::CaseConstruct *caseConstruct) {
  return Evaluation{std::get<parser::Scalar<parser::Expr>>(
      std::get<parser::Statement<parser::SelectCaseStmt>>(caseConstruct->t)
          .statement.t)
                        .thing.typedExpr.get()};
}
template<typename STMTTYPE, typename CT>
const std::optional<parser::Name> &GetSwitchAssociateName(
    const CT *selectConstruct) {
  return std::get<1>(
      std::get<parser::Statement<STMTTYPE>>(selectConstruct->t).statement.t);
}
template<typename CONSTRUCT, typename GSF>
void DumpSwitchWithSelector(
    const CONSTRUCT *construct, char const *const name, GSF getSelector) {
  auto selector{getSelector(construct)};
  DebugChannel() << name << "(" << selector.dump();
}

void LinearOp::dump() const {
  std::visit(
      common::visitors{
          [](const LinearLabel &t) {
            DebugChannel() << "label: " << t.get() << '\n';
          },
          [](const LinearGoto &t) {
            DebugChannel() << "goto " << t.target << '\n';
          },
          [](const LinearReturn &) { DebugChannel() << "return\n"; },
          [](const LinearConditionalGoto &t) {
            DebugChannel() << "cbranch (?) " << t.trueLabel << ' '
                           << t.falseLabel << '\n';
          },
          [](const LinearSwitchingIO &t) {
            DebugChannel() << "io-op";
            if (t.errLabel) DebugChannel() << " ERR=" << t.errLabel.value();
            if (t.eorLabel) DebugChannel() << " EOR=" << t.eorLabel.value();
            if (t.endLabel) DebugChannel() << " END=" << t.endLabel.value();
            DebugChannel() << '\n';
          },
          [](const LinearSwitch &lswitch) {
            DebugChannel() << "switch-";
            std::visit(
                common::visitors{
                    [](const parser::CaseConstruct *caseConstruct) {
                      DumpSwitchWithSelector(
                          caseConstruct, "case", GetSwitchCaseSelector);
                    },
                    [](const parser::SelectRankConstruct *selectRankConstruct) {
                      DumpSwitchWithSelector(
                          selectRankConstruct, "rank", GetSwitchRankSelector);
                    },
                    [](const parser::SelectTypeConstruct *selectTypeConstruct) {
                      DumpSwitchWithSelector(
                          selectTypeConstruct, "type", GetSwitchTypeSelector);
                    },
                    [](const parser::ComputedGotoStmt *computedGotoStmt) {
                      DebugChannel() << "igoto(?";
                    },
                    [](const parser::ArithmeticIfStmt *arithmeticIfStmt) {
                      DebugChannel() << "<=>(?";
                    },
                    [](const parser::CallStmt *callStmt) {
                      DebugChannel() << "alt-return(?";
                    },
                },
                lswitch.u);
            DebugChannel() << ") [...]\n";
          },
          [](const LinearAction &t) {
            DebugChannel() << "action: " << t.getSource().ToString() << '\n';
          },
          [](const LinearBeginConstruct &construct) {
            DebugChannel() << "construct-" << GetConstructName(construct)
                           << " {\n";
          },
          [](const LinearEndConstruct &construct) {
            DebugChannel() << "} construct-" << GetConstructName(construct)
                           << "\n";
          },
          [](const LinearIndirectGoto &) { DebugChannel() << "igoto\n"; },
      },
      u);
}
}  // end namespace

struct ControlFlowAnalyzer {
  explicit ControlFlowAnalyzer(std::list<LinearOp> &ops, AnalysisData &ad)
    : linearOps{ops}, ad{ad} {}

  LinearLabel buildNewLabel() { return BuildNewLabel(ad); }
  LinearOp findLabel(const parser::Label &lab) {
    auto iter{ad.labelMap.find(lab)};
    if (iter == ad.labelMap.end()) {
      LinearLabel ll{ad.labelBuilder};
      ad.labelMap.insert({lab, ll});
      return {ll};
    }
    return {iter->second};
  }
  template<typename A> constexpr bool Pre(const A &) { return true; }
  template<typename A> constexpr void Post(const A &) {}
  template<typename A> bool Pre(const parser::Statement<A> &stmt) {
    if (stmt.label) {
      linearOps.emplace_back(findLabel(*stmt.label));
    }
    if constexpr (std::is_same_v<A, parser::ActionStmt>) {
      LinearOp::Build(linearOps, stmt, ad);
    }
    return true;
  }

  // named constructs
  template<typename T> bool linearConstruct(const T &construct) {
    std::list<LinearOp> ops;
    LinearLabel label{buildNewLabel()};
    const parser::Name *name{getName(construct)};
    ad.nameStack.emplace_back(name, GetLabelRef(label), unspecifiedLabel);
    ops.emplace_back(LinearBeginConstruct{construct});
    ControlFlowAnalyzer cfa{ops, ad};
    Walk(std::get<parser::Block>(construct.t), cfa);
    ops.emplace_back(label);
    ops.emplace_back(LinearEndConstruct{construct});
    linearOps.splice(linearOps.end(), ops);
    ad.nameStack.pop_back();
    return false;
  }
  bool Pre(const parser::AssociateConstruct &c) { return linearConstruct(c); }
  bool Pre(const parser::ChangeTeamConstruct &c) { return linearConstruct(c); }
  bool Pre(const parser::CriticalConstruct &c) { return linearConstruct(c); }
  bool Pre(const parser::BlockConstruct &construct) {
    std::list<LinearOp> ops;
    LinearLabel label{buildNewLabel()};
    const auto &optName{
        std::get<parser::Statement<parser::BlockStmt>>(construct.t)
            .statement.v};
    const parser::Name *name{optName ? &*optName : nullptr};
    ad.nameStack.emplace_back(name, GetLabelRef(label), unspecifiedLabel);
    ops.emplace_back(LinearBeginConstruct{construct});
    ControlFlowAnalyzer cfa{ops, ad};
    Walk(std::get<parser::Block>(construct.t), cfa);
    ops.emplace_back(LinearEndConstruct{construct});
    ops.emplace_back(label);
    linearOps.splice(linearOps.end(), ops);
    ad.nameStack.pop_back();
    return false;
  }
  bool Pre(const parser::DoConstruct &construct) {
    std::list<LinearOp> ops;
    LinearLabel backedgeLab{buildNewLabel()};
    LinearLabel entryLab{buildNewLabel()};
    LinearLabel exitLab{buildNewLabel()};
    const parser::Name *name{getName(construct)};
    LinearLabelRef exitOpRef{GetLabelRef(exitLab)};
    ad.nameStack.emplace_back(name, exitOpRef, GetLabelRef(backedgeLab));
    ops.emplace_back(LinearBeginConstruct{construct});
    ops.emplace_back(backedgeLab);
    ops.emplace_back(LinearConditionalGoto{
        std::get<parser::Statement<parser::NonLabelDoStmt>>(construct.t),
        GetLabelRef(entryLab), exitOpRef});
    ops.push_back(entryLab);
    ControlFlowAnalyzer cfa{ops, ad};
    Walk(std::get<parser::Block>(construct.t), cfa);
    ops.emplace_back(LinearGoto{GetLabelRef(backedgeLab)});
    ops.emplace_back(LinearEndConstruct{construct});
    ops.emplace_back(exitLab);
    linearOps.splice(linearOps.end(), ops);
    ad.nameStack.pop_back();
    return false;
  }
  bool Pre(const parser::IfConstruct &construct) {
    std::list<LinearOp> ops;
    LinearLabel thenLab{buildNewLabel()};
    LinearLabel elseLab{buildNewLabel()};
    LinearLabel exitLab{buildNewLabel()};
    const parser::Name *name{getName(construct)};
    ad.nameStack.emplace_back(name, GetLabelRef(exitLab), unspecifiedLabel);
    ops.emplace_back(LinearBeginConstruct{construct});
    ops.emplace_back(LinearConditionalGoto{
        std::get<parser::Statement<parser::IfThenStmt>>(construct.t),
        GetLabelRef(thenLab), GetLabelRef(elseLab)});
    ops.emplace_back(thenLab);
    ControlFlowAnalyzer cfa{ops, ad};
    Walk(std::get<parser::Block>(construct.t), cfa);
    LinearLabelRef exitOpRef{GetLabelRef(exitLab)};
    ops.emplace_back(LinearGoto{exitOpRef});
    for (const auto &elseIfBlock :
        std::get<std::list<parser::IfConstruct::ElseIfBlock>>(construct.t)) {
      ops.emplace_back(elseLab);
      LinearLabel newThenLab{buildNewLabel()};
      LinearLabel newElseLab{buildNewLabel()};
      ops.emplace_back(LinearConditionalGoto{
          std::get<parser::Statement<parser::ElseIfStmt>>(elseIfBlock.t),
          GetLabelRef(newThenLab), GetLabelRef(newElseLab)});
      ops.emplace_back(newThenLab);
      Walk(std::get<parser::Block>(elseIfBlock.t), cfa);
      ops.emplace_back(LinearGoto{exitOpRef});
      elseLab = newElseLab;
    }
    ops.emplace_back(elseLab);
    if (const auto &optElseBlock{
            std::get<std::optional<parser::IfConstruct::ElseBlock>>(
                construct.t)}) {
      Walk(std::get<parser::Block>(optElseBlock->t), cfa);
    }
    ops.emplace_back(LinearGoto{exitOpRef});
    ops.emplace_back(exitLab);
    ops.emplace_back(LinearEndConstruct{construct});
    linearOps.splice(linearOps.end(), ops);
    ad.nameStack.pop_back();
    return false;
  }
  template<typename A,
      typename B = std::conditional_t<std::is_same_v<A, parser::CaseConstruct>,
          parser::CaseConstruct::Case,
          std::conditional_t<std::is_same_v<A, parser::SelectRankConstruct>,
              parser::SelectRankConstruct::RankCase,
              std::conditional_t<std::is_same_v<A, parser::SelectTypeConstruct>,
                  parser::SelectTypeConstruct::TypeCase, void>>>>
  bool Multiway(const A &construct) {
    std::list<LinearOp> ops;
    LinearLabel exitLab{buildNewLabel()};
    const parser::Name *name{getName(construct)};
    ad.nameStack.emplace_back(name, GetLabelRef(exitLab), unspecifiedLabel);
    ops.emplace_back(LinearBeginConstruct{construct});
    const auto N{std::get<std::list<B>>(construct.t).size()};
    LinearLabelRef exitOpRef{GetLabelRef(exitLab)};
    if (N > 0) {
      typename std::list<B>::size_type i;
      std::vector<LinearLabel> toLabels;
      for (i = 0; i != N; ++i) {
        toLabels.emplace_back(buildNewLabel());
      }
      std::vector<LinearLabelRef> targets;
      for (i = 0; i != N; ++i) {
        targets.emplace_back(GetLabelRef(toLabels[i]));
      }
      ops.emplace_back(LinearSwitch{construct, targets});
      ControlFlowAnalyzer cfa{ops, ad};
      i = 0;
      for (const auto &caseBlock : std::get<std::list<B>>(construct.t)) {
        ops.emplace_back(toLabels[i++]);
        Walk(std::get<parser::Block>(caseBlock.t), cfa);
        ops.emplace_back(LinearGoto{exitOpRef});
      }
    }
    ops.emplace_back(exitLab);
    ops.emplace_back(LinearEndConstruct{construct});
    linearOps.splice(linearOps.end(), ops);
    ad.nameStack.pop_back();
    return false;
  }
  bool Pre(const parser::CaseConstruct &c) { return Multiway(c); }
  bool Pre(const parser::SelectRankConstruct &c) { return Multiway(c); }
  bool Pre(const parser::SelectTypeConstruct &c) { return Multiway(c); }
  bool Pre(const parser::WhereConstruct &c) {
    std::list<LinearOp> ops;
    LinearLabel label{buildNewLabel()};
    const parser::Name *name{getName(c)};
    ad.nameStack.emplace_back(name, GetLabelRef(label), unspecifiedLabel);
    ops.emplace_back(LinearBeginConstruct{c});
    ControlFlowAnalyzer cfa{ops, ad};
    Walk(std::get<std::list<parser::WhereBodyConstruct>>(c.t), cfa);
    Walk(
        std::get<std::list<parser::WhereConstruct::MaskedElsewhere>>(c.t), cfa);
    Walk(std::get<std::optional<parser::WhereConstruct::Elsewhere>>(c.t), cfa);
    ops.emplace_back(label);
    ops.emplace_back(LinearEndConstruct{c});
    linearOps.splice(linearOps.end(), ops);
    ad.nameStack.pop_back();
    return false;
  }
  bool Pre(const parser::ForallConstruct &construct) {
    std::list<LinearOp> ops;
    LinearLabel label{buildNewLabel()};
    const parser::Name *name{getName(construct)};
    ad.nameStack.emplace_back(name, GetLabelRef(label), unspecifiedLabel);
    ops.emplace_back(LinearBeginConstruct{construct});
    ControlFlowAnalyzer cfa{ops, ad};
    Walk(std::get<std::list<parser::ForallBodyConstruct>>(construct.t), cfa);
    ops.emplace_back(label);
    ops.emplace_back(LinearEndConstruct{construct});
    linearOps.splice(linearOps.end(), ops);
    ad.nameStack.pop_back();
    return false;
  }
  template<typename A> const parser::Name *getName(const A &a) {
    const auto &optName{std::get<0>(std::get<0>(a.t).statement.t)};
    return optName ? &*optName : nullptr;
  }
  LinearLabelRef GetLabelRef(const LinearLabel &label) {
    label.setReferenced();
    return label;
  }
  LinearLabelRef GetLabelRef(const parser::Label &label) {
    return FetchLabel(ad, label);
  }

  std::list<LinearOp> &linearOps;
  AnalysisData &ad;
};

struct SwitchArguments {
  Evaluation exp;
  LinearLabelRef defLab;
  std::vector<SwitchStmt::ValueType> values;
  std::vector<LinearLabelRef> labels;
};
struct SwitchCaseArguments {
  Evaluation exp;
  LinearLabelRef defLab;
  std::vector<SwitchCaseStmt::ValueType> ranges;
  std::vector<LinearLabelRef> labels;
};
struct SwitchRankArguments {
  Evaluation exp;
  LinearLabelRef defLab;
  std::vector<SwitchRankStmt::ValueType> ranks;
  std::vector<LinearLabelRef> labels;
};
struct SwitchTypeArguments {
  Evaluation exp;
  LinearLabelRef defLab;
  std::vector<SwitchTypeStmt::ValueType> types;
  std::vector<LinearLabelRef> labels;
};

template<typename T>
static bool IsDefault(const typename T::ValueType &valueType) {
  return std::holds_alternative<typename T::Default>(valueType);
}
template<typename T>
void cleanupSwitchPairs(LinearLabelRef &defLab,
    std::vector<typename T::ValueType> &values,
    std::vector<LinearLabelRef> &labels) {
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
    const std::list<parser::CaseConstruct::Case> &list) {
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
                  valueList.emplace_back(SwitchCaseStmt::Exactly{
                      caseValue.thing.thing->typedExpr.get()});
                },
                [&](const parser::CaseValueRange::Range &range) {
                  if (range.lower.has_value()) {
                    if (range.upper.has_value()) {
                      valueList.emplace_back(SwitchCaseStmt::InclusiveRange{
                          range.lower->thing.thing->typedExpr.get(),
                          range.upper->thing.thing->typedExpr.get()});
                    } else {
                      valueList.emplace_back(SwitchCaseStmt::InclusiveAbove{
                          range.lower->thing.thing->typedExpr.get()});
                    }
                  } else {
                    valueList.emplace_back(SwitchCaseStmt::InclusiveBelow{
                        range.upper->thing.thing->typedExpr.get()});
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
    std::visit(common::visitors{
                   [&](const parser::ScalarIntConstantExpr &expression) {
                     result.emplace_back(SwitchRankStmt::Exactly{
                         expression.thing.thing.thing->typedExpr.get()});
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
    std::visit(common::visitors{
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
static SwitchCaseArguments ComposeSwitchCaseArguments(
    const parser::CaseConstruct *caseConstruct,
    const std::vector<LinearLabelRef> &refs) {
  auto &cases{
      std::get<std::list<parser::CaseConstruct::Case>>(caseConstruct->t)};
  SwitchCaseArguments result{GetSwitchCaseSelector(caseConstruct),
      unspecifiedLabel, populateSwitchValues(cases), std::move(refs)};
  cleanupSwitchPairs<SwitchCaseStmt>(
      result.defLab, result.ranges, result.labels);
  return result;
}
static SwitchRankArguments ComposeSwitchRankArguments(
    const parser::SelectRankConstruct *selectRankConstruct,
    const std::vector<LinearLabelRef> &refs) {
  auto &ranks{std::get<std::list<parser::SelectRankConstruct::RankCase>>(
      selectRankConstruct->t)};
  SwitchRankArguments result{GetSwitchRankSelector(selectRankConstruct),
      unspecifiedLabel, populateSwitchValues(ranks), std::move(refs)};
  if (auto &name{GetSwitchAssociateName<parser::SelectRankStmt>(
          selectRankConstruct)}) {
    (void)name;  // get rid of warning
    // TODO: handle associate-name -> Add an assignment stmt?
  }
  cleanupSwitchPairs<SwitchRankStmt>(
      result.defLab, result.ranks, result.labels);
  return result;
}
static SwitchTypeArguments ComposeSwitchTypeArguments(
    const parser::SelectTypeConstruct *selectTypeConstruct,
    const std::vector<LinearLabelRef> &refs) {
  auto &types{std::get<std::list<parser::SelectTypeConstruct::TypeCase>>(
      selectTypeConstruct->t)};
  SwitchTypeArguments result{GetSwitchTypeSelector(selectTypeConstruct),
      unspecifiedLabel, populateSwitchValues(types), std::move(refs)};
  if (auto &name{GetSwitchAssociateName<parser::SelectTypeStmt>(
          selectTypeConstruct)}) {
    (void)name;  // get rid of warning
    // TODO: handle associate-name -> Add an assignment stmt?
  }
  cleanupSwitchPairs<SwitchTypeStmt>(
      result.defLab, result.types, result.labels);
  return result;
}

static void buildMultiwayDefaultNext(SwitchArguments &result) {
  result.defLab = result.labels.back();
  result.labels.pop_back();
}
static SwitchArguments ComposeSwitchArgs(const LinearSwitch &op) {
  SwitchArguments result{nullptr, unspecifiedLabel, {}, op.refs};
  std::visit(common::visitors{
                 [](const auto *) { WRONG_PATH(); },
                 [&](const parser::ComputedGotoStmt *c) {
                   result.exp = std::get<parser::ScalarIntExpr>(c->t)
                                    .thing.thing->typedExpr.get();
                   buildMultiwayDefaultNext(result);
                 },
                 [&](const parser::ArithmeticIfStmt *c) {
                   result.exp = std::get<parser::Expr>(c->t).typedExpr.get();
                 },
                 [&](const parser::CallStmt *c) {
                   result.exp = nullptr;  // fixme - result of call
                   buildMultiwayDefaultNext(result);
                 },
             },
      op.u);
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

static Expression *AlwaysTrueExpression() {
  auto result{common::SearchTypes(
      evaluate::TypeKindVisitor<evaluate::TypeCategory::Logical,
          evaluate::Constant, bool>{1, true})};
  CHECK(result.has_value());
  // TODO: this really ought to be hashconsed, but alas trees
  return new evaluate::GenericExprWrapper(std::move(*result));
}
static Expression *LessThanOrEqualToComparison(
    const semantics::Symbol *symbol, Expression *second) {
  return second;  // FIXME
}
static Expression *GreaterThanOrEqualToComparison(
    const semantics::Symbol *symbol, Expression *second) {
  return second;  // FIXME
}
static Expression *BuildLoopLatchExpression(
    const std::optional<parser::LoopControl> &loopControl) {
  if (loopControl.has_value()) {
    return std::visit(
        common::visitors{
            [](const parser::LoopBounds<parser::ScalarIntExpr> &loopBounds) {
              auto *loopVariable{loopBounds.name.thing.thing.symbol};
              auto *second{loopBounds.upper.thing.thing->typedExpr.get()};
              if (loopBounds.step.has_value()) {
                auto *step{loopBounds.step->thing.thing->typedExpr.get()};
                SEMANTICS_CHECK(step, "DO step expression missing");
                if (evaluate::IsConstantExpr(step->v)) {
                  auto stepValue{evaluate::ToInt64(step->v)};
                  SEMANTICS_CHECK(stepValue != 0, "step cannot be zero");
                  if (stepValue > 0) {
                    return LessThanOrEqualToComparison(loopVariable, second);
                  }
                  return GreaterThanOrEqualToComparison(loopVariable, second);
                }
                // expr ::= (step > 0 && v <= snd) || (step < 0 && v >= snd)
                return AlwaysTrueExpression();  // FIXME
              }
              return LessThanOrEqualToComparison(loopVariable, second);
            },
            [](const parser::ScalarLogicalExpr &scalarLogicalExpr) {
              auto &expression{scalarLogicalExpr.thing.thing};
              SEMANTICS_CHECK(
                  expression->typedExpr.get(), "DO WHILE condition missing");
              return expression->typedExpr.get();
            },
            [](const parser::LoopControl::Concurrent &concurrent) {
              return AlwaysTrueExpression();  // FIXME
            },
        },
        loopControl->u);
  }
  return AlwaysTrueExpression();
}

static void CreateSwitchHelper(IntermediateRepresentationBuilder *builder,
    const Evaluation &condition, BasicBlock *defaultCase,
    const SwitchStmt::ValueSuccPairListType &rest) {
  builder->CreateSwitch(condition, defaultCase, rest);
}
static void CreateSwitchCaseHelper(IntermediateRepresentationBuilder *builder,
    const Evaluation &condition, BasicBlock *defaultCase,
    const SwitchCaseStmt::ValueSuccPairListType &rest) {
  builder->CreateSwitchCase(condition, defaultCase, rest);
}
static void CreateSwitchRankHelper(IntermediateRepresentationBuilder *builder,
    const Evaluation &condition, BasicBlock *defaultCase,
    const SwitchRankStmt::ValueSuccPairListType &rest) {
  builder->CreateSwitchRank(condition, defaultCase, rest);
}
static void CreateSwitchTypeHelper(IntermediateRepresentationBuilder *builder,
    const Evaluation &condition, BasicBlock *defaultCase,
    const SwitchTypeStmt::ValueSuccPairListType &rest) {
  builder->CreateSwitchType(condition, defaultCase, rest);
}

struct FortranIRLowering {
  using LabelMapType = std::map<LinearLabelRef, BasicBlock *>;
  using Closure = std::function<void(const LabelMapType &)>;

  FortranIRLowering(semantics::SemanticsContext &sc, bool debugLinearIR)
    : fir_{new Program("program_name")}, semanticsContext_{sc},
      debugLinearIntermediateRepresentation_{debugLinearIR} {}
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
    builder_ = new IntermediateRepresentationBuilder(
        *CreateBlock(subp->getLastRegion()));
    AnalysisData ad;
    ControlFlowAnalyzer linearize{linearOperations_, ad};
    Walk(here, linearize);
    if (debugLinearIntermediateRepresentation_) {
      dumpLinearRepresentation();
    }
    ConstructIntermediateRepresentation(ad);
    DrawRemainingArcs();
    Cleanup();
  }
  void dumpLinearRepresentation() const {
    for (const auto &op : linearOperations_) {
      op.dump();
    }
    DebugChannel() << "--- END ---\n";
  }
  const Expression *CreatePointerValue(
      const parser::PointerAssignmentStmt *stmt) {
    // TODO: build a RHS expression to assign to a POINTER
    return static_cast<const Expression *>(nullptr);
  }
  const Expression *CreateAllocationValue(const parser::Allocation *allocation,
      const parser::AllocateStmt *statement) {
    auto &obj{std::get<parser::AllocateObject>(allocation->t)};
    (void)obj;
    // TODO: build an expression for the allocation
    return static_cast<const Expression *>(nullptr);
  }
  const Expression *CreateDeallocationValue(
      const parser::AllocateObject *allocateObject,
      const parser::DeallocateStmt *statement) {
    // TODO: build an expression for the allocation
    return static_cast<const Expression *>(nullptr);
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
  const Value *CreateCalleeValue(
      const parser::ProcedureDesignator &designator) {
    return static_cast<const Value *>(nullptr);
  }
  CallArguments CreateCallArguments(
      const std::list<parser::ActualArgSpec> &arguments) {
    return CallArguments{};
  }

  void handleActionStatement(
      AnalysisData &ad, const parser::Statement<parser::ActionStmt> &stmt) {
    std::visit(
        common::visitors{
            [&](const common::Indirection<parser::AllocateStmt> &statement) {
              for (auto &allocation :
                  std::get<std::list<parser::Allocation>>(statement->t)) {
                // TODO: add more arguments to builder as needed
                builder_->CreateAlloc(
                    CreateAllocationValue(&allocation, &*statement));
              }
            },
            [&](const common::Indirection<parser::AssignmentStmt> &statement) {
              builder_->CreateAssign(&std::get<parser::Variable>(statement->t),
                  std::get<parser::Expr>(statement->t).typedExpr.get());
            },
            [&](const common::Indirection<parser::BackspaceStmt> &statement) {
              builder_->CreateIOCall(InputOutputCallBackspace,
                  CreateBackspaceArguments(statement->v));
            },
            [&](const common::Indirection<parser::CallStmt> &statement) {
              builder_->CreateCall(static_cast<const FunctionType *>(nullptr),
                  CreateCalleeValue(
                      std::get<parser::ProcedureDesignator>(statement->v.t)),
                  CreateCallArguments(
                      std::get<std::list<parser::ActualArgSpec>>(
                          statement->v.t)));
            },
            [&](const common::Indirection<parser::CloseStmt> &statement) {
              builder_->CreateIOCall(
                  InputOutputCallClose, CreateCloseArguments(statement->v));
            },
            [](const parser::ContinueStmt &) { WRONG_PATH(); },
            [](const common::Indirection<parser::CycleStmt> &) {
              WRONG_PATH();
            },
            [&](const common::Indirection<parser::DeallocateStmt> &statement) {
              for (auto &allocateObject :
                  std::get<std::list<parser::AllocateObject>>(statement->t)) {
                builder_->CreateDealloc(
                    CreateDeallocationValue(&allocateObject, &*statement));
              }
            },
            [&](const common::Indirection<parser::EndfileStmt> &statement) {
              builder_->CreateIOCall(
                  InputOutputCallEndfile, CreateEndfileArguments(statement->v));
            },
            [&](const common::Indirection<parser::EventPostStmt> &statement) {
              builder_->CreateRuntimeCall(
                  RuntimeCallEventPost, CreateEventPostArguments(*statement));
            },
            [&](const common::Indirection<parser::EventWaitStmt> &statement) {
              builder_->CreateRuntimeCall(
                  RuntimeCallEventWait, CreateEventWaitArguments(*statement));
            },
            [](const common::Indirection<parser::ExitStmt> &) { WRONG_PATH(); },
            [&](const parser::FailImageStmt &statement) {
              builder_->CreateRuntimeCall(
                  RuntimeCallFailImage, CreateFailImageArguments(statement));
            },
            [&](const common::Indirection<parser::FlushStmt> &statement) {
              builder_->CreateIOCall(
                  InputOutputCallFlush, CreateFlushArguments(statement->v));
            },
            [&](const common::Indirection<parser::FormTeamStmt> &statement) {
              builder_->CreateRuntimeCall(
                  RuntimeCallFormTeam, CreateFormTeamArguments(*statement));
            },
            [](const common::Indirection<parser::GotoStmt> &) { WRONG_PATH(); },
            [](const common::Indirection<parser::IfStmt> &) { WRONG_PATH(); },
            [&](const common::Indirection<parser::InquireStmt> &statement) {
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
                  statement->u);
            },
            [&](const common::Indirection<parser::LockStmt> &statement) {
              builder_->CreateRuntimeCall(
                  RuntimeCallLock, CreateLockArguments(*statement));
            },
            [&](const common::Indirection<parser::NullifyStmt> &statement) {
              builder_->CreateNullify(&*statement);
            },
            [&](const common::Indirection<parser::OpenStmt> &statement) {
              builder_->CreateIOCall(
                  InputOutputCallOpen, CreateOpenArguments(statement->v));
            },
            [&](const common::Indirection<parser::PointerAssignmentStmt>
                    &statement) {
              builder_->CreatePointerAssign(
                  std::get<parser::Expr>(statement->t).typedExpr.get(),
                  CreatePointerValue(&*statement));
            },
            [&](const common::Indirection<parser::PrintStmt> &statement) {
              builder_->CreateIOCall(InputOutputCallPrint,
                  CreatePrintArguments(std::get<parser::Format>(statement->t),
                      std::get<std::list<parser::OutputItem>>(statement->t)));
            },
            [&](const common::Indirection<parser::ReadStmt> &statement) {
              builder_->CreateIOCall(InputOutputCallRead,
                  CreateReadArguments(statement->iounit, statement->format,
                      statement->controls, statement->items));
            },
            [](const common::Indirection<parser::ReturnStmt> &) {
              WRONG_PATH();
            },
            [&](const common::Indirection<parser::RewindStmt> &statement) {
              builder_->CreateIOCall(
                  InputOutputCallRewind, CreateRewindArguments(statement->v));
            },
            [&](const common::Indirection<parser::StopStmt> &statement) {
              builder_->CreateRuntimeCall(
                  RuntimeCallStop, CreateStopArguments(*statement));
            },
            [&](const common::Indirection<parser::SyncAllStmt> &statement) {
              builder_->CreateRuntimeCall(
                  RuntimeCallSyncAll, CreateSyncAllArguments(*statement));
            },
            [&](const common::Indirection<parser::SyncImagesStmt> &statement) {
              builder_->CreateRuntimeCall(
                  RuntimeCallSyncImages, CreateSyncImagesArguments(*statement));
            },
            [&](const common::Indirection<parser::SyncMemoryStmt> &statement) {
              builder_->CreateRuntimeCall(
                  RuntimeCallSyncMemory, CreateSyncMemoryArguments(*statement));
            },
            [&](const common::Indirection<parser::SyncTeamStmt> &statement) {
              builder_->CreateRuntimeCall(
                  RuntimeCallSyncTeam, CreateSyncTeamArguments(*statement));
            },
            [&](const common::Indirection<parser::UnlockStmt> &statement) {
              builder_->CreateRuntimeCall(
                  RuntimeCallUnlock, CreateUnlockArguments(*statement));
            },
            [&](const common::Indirection<parser::WaitStmt> &statement) {
              builder_->CreateIOCall(
                  InputOutputCallWait, CreateWaitArguments(statement->v));
            },
            [](const common::Indirection<parser::WhereStmt> &) { /*fixme*/ },
            [&](const common::Indirection<parser::WriteStmt> &statement) {
              builder_->CreateIOCall(InputOutputCallWrite,
                  CreateWriteArguments(statement->iounit, statement->format,
                      statement->controls, statement->items));
            },
            [](const common::Indirection<parser::ComputedGotoStmt> &) {
              WRONG_PATH();
            },
            [](const common::Indirection<parser::ForallStmt> &) { /*fixme*/ },
            [](const common::Indirection<parser::ArithmeticIfStmt> &) {
              WRONG_PATH();
            },
            [&](const common::Indirection<parser::AssignStmt> &statement) {
              builder_->CreateAssign(
                  std::get<parser::Name>(statement->t).symbol,
                  blockMap_
                      .find(
                          FetchLabel(ad, std::get<parser::Label>(statement->t))
                              .get())
                      ->second);
            },
            [](const common::Indirection<parser::AssignedGotoStmt> &) {
              WRONG_PATH();
            },
            [&](const common::Indirection<parser::PauseStmt> &statement) {
              builder_->CreateRuntimeCall(
                  RuntimeCallPause, CreatePauseArguments(*statement));
            },
        },
        stmt.statement.u);
  }
  void handleLinearAction(const LinearAction &linearAction, AnalysisData &ad) {
    handleActionStatement(ad, *linearAction.v);
  }

  // InitiateConstruct - many constructs require some initial setup
  void InitiateConstruct(const parser::AssociateStmt *associateStmt) {
    for (auto &association :
        std::get<std::list<parser::Association>>(associateStmt->t)) {
      auto &name{std::get<parser::Name>(association.t)};
      // Evaluation eval{...};
      (void)name;
    }
    builder_->CreateExpr(associateStmt);
  }
  void InitiateConstruct(const parser::SelectCaseStmt *selectCaseStmt) {
    builder_->CreateExpr(
        std::get<parser::Scalar<parser::Expr>>(selectCaseStmt->t)
            .thing.typedExpr.get());
  }
  void InitiateConstruct(const parser::ChangeTeamStmt *changeTeamStmt) {
    builder_->CreateExpr(changeTeamStmt);
  }
  void InitiateConstruct(const parser::NonLabelDoStmt *nonlabelDoStmt) {
    // evaluate e1, e2 [, e3] ...
    builder_->CreateExpr(nonlabelDoStmt);
  }
  void InitiateConstruct(const parser::IfThenStmt *ifThenStmt) {
    builder_->CreateExpr(std::get<parser::ScalarLogicalExpr>(ifThenStmt->t)
                             .thing.thing->typedExpr.get());
  }
  void InitiateConstruct(const parser::WhereConstructStmt *whereConstructStmt) {
    builder_->CreateExpr(std::get<parser::LogicalExpr>(whereConstructStmt->t)
                             .thing->typedExpr.get());
  }
  void InitiateConstruct(
      const parser::ForallConstructStmt *forallConstructStmt) {
    builder_->CreateExpr(forallConstructStmt);
  }

  void ConstructIntermediateRepresentation(AnalysisData &ad) {
    for (auto iter{linearOperations_.begin()}, iend{linearOperations_.end()};
         iter != iend; ++iter) {
      const auto &op{*iter};
      std::visit(
          common::visitors{
              [&](const LinearLabel &linearLabel) {
                auto *newBlock{CreateBlock(builder_->GetCurrentRegion())};
                blockMap_.insert({linearLabel.get(), newBlock});
                if (builder_->GetInsertionPoint()) {
                  builder_->CreateBranch(newBlock);
                }
                builder_->SetInsertionPoint(newBlock);
              },
              [&](const LinearGoto &linearGoto) {
                CheckInsertionPoint();
                AddOrQueueBranch(linearGoto.target);
                builder_->ClearInsertionPoint();
              },
              [&](const LinearIndirectGoto &linearIGoto) {
                CheckInsertionPoint();
                AddOrQueueIGoto(ad, linearIGoto.symbol, linearIGoto.labelRefs);
                builder_->ClearInsertionPoint();
              },
              [&](const LinearReturn &linearReturn) {
                CheckInsertionPoint();
                std::visit(common::visitors{
                               [&](const parser::FailImageStmt *s) {
                                 builder_->CreateRuntimeCall(
                                     RuntimeCallFailImage,
                                     CreateFailImageArguments(*s));
                                 builder_->CreateUnreachable();
                               },
                               [&](const parser::ReturnStmt *s) {
                                 if (s->v) {
                                   builder_->CreateReturn(
                                       s->v->thing.thing->typedExpr.get());
                                 } else {
                                   builder_->CreateRetVoid();
                                 }
                               },
                               [&](const parser::StopStmt *s) {
                                 builder_->CreateRuntimeCall(
                                     RuntimeCallStop, CreateStopArguments(*s));
                                 builder_->CreateUnreachable();
                               },
                           },
                    linearReturn.u);
                builder_->ClearInsertionPoint();
              },
              [&](const LinearConditionalGoto &linearConditionalGoto) {
                CheckInsertionPoint();
                std::visit(
                    common::visitors{
                        [&](const parser::Statement<parser::IfThenStmt>
                                *statement) {
                          const auto &expression{
                              std::get<parser::ScalarLogicalExpr>(
                                  statement->statement.t)
                                  .thing.thing};
                          SEMANTICS_CHECK(expression->typedExpr.get(),
                              "IF THEN condition expression missing");
                          AddOrQueueCGoto(expression->typedExpr.get(),
                              linearConditionalGoto.trueLabel,
                              linearConditionalGoto.falseLabel);
                        },
                        [&](const parser::Statement<parser::ElseIfStmt>
                                *statement) {
                          const auto &expression{
                              std::get<parser::ScalarLogicalExpr>(
                                  statement->statement.t)
                                  .thing.thing};
                          SEMANTICS_CHECK(expression->typedExpr.get(),
                              "ELSE IF condition expression missing");
                          AddOrQueueCGoto(expression->typedExpr.get(),
                              linearConditionalGoto.trueLabel,
                              linearConditionalGoto.falseLabel);
                        },
                        [&](const parser::IfStmt *statement) {
                          const auto &expression{
                              std::get<parser::ScalarLogicalExpr>(statement->t)
                                  .thing.thing};
                          SEMANTICS_CHECK(expression->typedExpr.get(),
                              "IF condition expression missing");
                          AddOrQueueCGoto(expression->typedExpr.get(),
                              linearConditionalGoto.trueLabel,
                              linearConditionalGoto.falseLabel);
                        },
                        [&](const parser::Statement<parser::NonLabelDoStmt>
                                *statement) {
                          AddOrQueueCGoto(
                              BuildLoopLatchExpression(
                                  std::get<std::optional<parser::LoopControl>>(
                                      statement->statement.t)),
                              linearConditionalGoto.trueLabel,
                              linearConditionalGoto.falseLabel);
                        }},
                    linearConditionalGoto.u);
                builder_->ClearInsertionPoint();
              },
              [&](const LinearSwitchingIO &linearIO) {
                CheckInsertionPoint();
                AddOrQueueSwitch<SwitchStmt>(
                    nullptr, linearIO.next, {}, {}, CreateSwitchHelper);
                builder_->ClearInsertionPoint();
              },
              [&](const LinearSwitch &linearSwitch) {
                CheckInsertionPoint();
                std::visit(common::visitors{
                               [&](auto) {
                                 auto args{ComposeSwitchArgs(linearSwitch)};
                                 AddOrQueueSwitch<SwitchStmt>(args.exp,
                                     args.defLab, args.values, args.labels,
                                     CreateSwitchHelper);
                               },
                               [&](const parser::CaseConstruct *caseConstruct) {
                                 auto args{ComposeSwitchCaseArguments(
                                     caseConstruct, linearSwitch.refs)};
                                 AddOrQueueSwitch<SwitchCaseStmt>(args.exp,
                                     args.defLab, args.ranges, args.labels,
                                     CreateSwitchCaseHelper);
                               },
                               [&](const parser::SelectRankConstruct
                                       *selectRankConstruct) {
                                 auto args{ComposeSwitchRankArguments(
                                     selectRankConstruct, linearSwitch.refs)};
                                 AddOrQueueSwitch<SwitchRankStmt>(args.exp,
                                     args.defLab, args.ranks, args.labels,
                                     CreateSwitchRankHelper);
                               },
                               [&](const parser::SelectTypeConstruct
                                       *selectTypeConstruct) {
                                 auto args{ComposeSwitchTypeArguments(
                                     selectTypeConstruct, linearSwitch.refs)};
                                 AddOrQueueSwitch<SwitchTypeStmt>(args.exp,
                                     args.defLab, args.types, args.labels,
                                     CreateSwitchTypeHelper);
                               },
                           },
                    linearSwitch.u);
                builder_->ClearInsertionPoint();
              },
              [&](const LinearAction &linearAction) {
                CheckInsertionPoint();
                handleLinearAction(linearAction, ad);
              },
              [&](const LinearBeginConstruct &linearConstruct) {
                std::visit(
                    common::visitors{
                        [&](const parser::AssociateConstruct *construct) {
                          const auto &statement{std::get<
                              parser::Statement<parser::AssociateStmt>>(
                              construct->t)};
                          const auto &position{statement.source};
                          EnterRegion(position);
                          InitiateConstruct(&statement.statement);
                        },
                        [&](const parser::BlockConstruct *construct) {
                          EnterRegion(
                              std::get<parser::Statement<parser::BlockStmt>>(
                                  construct->t)
                                  .source);
                        },
                        [&](const parser::CaseConstruct *construct) {
                          InitiateConstruct(
                              &std::get<
                                  parser::Statement<parser::SelectCaseStmt>>(
                                  construct->t)
                                   .statement);
                        },
                        [&](const parser::ChangeTeamConstruct *construct) {
                          const auto &statement{std::get<
                              parser::Statement<parser::ChangeTeamStmt>>(
                              construct->t)};
                          EnterRegion(statement.source);
                          InitiateConstruct(&statement.statement);
                        },
                        [&](const parser::DoConstruct *construct) {
                          const auto &statement{std::get<
                              parser::Statement<parser::NonLabelDoStmt>>(
                              construct->t)};
                          EnterRegion(statement.source);
                          InitiateConstruct(&statement.statement);
                        },
                        [&](const parser::IfConstruct *construct) {
                          InitiateConstruct(
                              &std::get<parser::Statement<parser::IfThenStmt>>(
                                  construct->t)
                                   .statement);
                        },
                        [&](const parser::SelectRankConstruct *construct) {
                          const auto &statement{std::get<
                              parser::Statement<parser::SelectRankStmt>>(
                              construct->t)};
                          EnterRegion(statement.source);
                        },
                        [&](const parser::SelectTypeConstruct *construct) {
                          const auto &statement{std::get<
                              parser::Statement<parser::SelectTypeStmt>>(
                              construct->t)};
                          EnterRegion(statement.source);
                        },
                        [&](const parser::WhereConstruct *construct) {
                          InitiateConstruct(
                              &std::get<parser::Statement<
                                   parser::WhereConstructStmt>>(construct->t)
                                   .statement);
                        },
                        [&](const parser::ForallConstruct *construct) {
                          InitiateConstruct(
                              &std::get<parser::Statement<
                                   parser::ForallConstructStmt>>(construct->t)
                                   .statement);
                        },
                        [](const parser::CriticalConstruct *) { /*fixme*/ },
                        [](const parser::CompilerDirective *) { /*fixme*/ },
                        [](const parser::OpenMPConstruct *) { /*fixme*/ },
                        [](const parser::OpenMPEndLoopDirective
                                *) { /*fixme*/ },
                    },
                    linearConstruct.u);
                auto next{iter};
                const auto &nextOp{*(++next)};
                std::visit(common::visitors{
                               [](const auto &) {},
                               [&](const LinearLabel &linearLabel) {
                                 blockMap_.insert({linearLabel.get(),
                                     builder_->GetInsertionPoint()});
                                 ++iter;
                               },
                           },
                    nextOp.u);
              },
              [&](const LinearEndConstruct &linearConstruct) {
                std::visit(
                    common::visitors{
                        [](const auto &) {},
                        [&](const parser::BlockConstruct *) { ExitRegion(); },
                        [&](const parser::DoConstruct *) { ExitRegion(); },
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
                    linearConstruct.u);
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
  void AddOrQueueBranch(LinearLabelRef dest) {
    auto iter{blockMap_.find(dest)};
    if (iter != blockMap_.end()) {
      builder_->CreateBranch(iter->second);
    } else {
      using namespace std::placeholders;
      controlFlowEdgesToAdd_.emplace_back(std::bind(
          [](IntermediateRepresentationBuilder *builder, BasicBlock *block,
              LinearLabelRef dest, const LabelMapType &map) {
            builder->SetInsertionPoint(block);
            CHECK(map.find(dest) != map.end());
            builder->CreateBranch(map.find(dest)->second);
          },
          builder_, builder_->GetInsertionPoint(), dest, _1));
    }
  }
  void AddOrQueueCGoto(Expression *condition, LinearLabelRef trueBlock,
      LinearLabelRef falseBlock) {
    auto trueIter{blockMap_.find(trueBlock)};
    auto falseIter{blockMap_.find(falseBlock)};
    if (trueIter != blockMap_.end() && falseIter != blockMap_.end()) {
      builder_->CreateConditionalBranch(
          condition, trueIter->second, falseIter->second);
    } else {
      using namespace std::placeholders;
      controlFlowEdgesToAdd_.emplace_back(std::bind(
          [](IntermediateRepresentationBuilder *builder, BasicBlock *block,
              Expression *expr, LinearLabelRef trueDest,
              LinearLabelRef falseDest, const LabelMapType &map) {
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
  void AddOrQueueSwitch(const Evaluation &condition,
      LinearLabelRef defaultLabel,
      const std::vector<typename SWITCHTYPE::ValueType> &values,
      const std::vector<LinearLabelRef> &labels, F function) {
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
          [](IntermediateRepresentationBuilder *builder, BasicBlock *block,
              const Evaluation &expr, LinearLabelRef defaultDest,
              const std::vector<typename SWITCHTYPE::ValueType> &values,
              const std::vector<LinearLabelRef> &labels, F function,
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
      const std::vector<LinearLabelRef> &labels) {
    auto useLabels{labels.empty() ? GetAssign(ad, symbol) : labels};
    auto defer{false};
    IndirectBrStmt::TargetListType blocks;
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
          [](IntermediateRepresentationBuilder *builder, BasicBlock *block,
              Variable *variable, const std::vector<LinearLabelRef> &fixme,
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

  IntermediateRepresentationBuilder *builder_{nullptr};
  Program *fir_;
  std::list<LinearOp> linearOperations_;
  std::list<Closure> controlFlowEdgesToAdd_;
  LabelMapType blockMap_;
  semantics::SemanticsContext &semanticsContext_;
  bool debugLinearIntermediateRepresentation_;
};

Program *CreateFortranIR(const parser::Program &program,
    semantics::SemanticsContext &semanticsContext, bool debugLinearIR) {
  FortranIRLowering converter{semanticsContext, debugLinearIR};
  Walk(program, converter);
  return converter.program();
}
}
