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

#include "flattened.h"
#include "../parser/parse-tree-visitor.h"

namespace Fortran::FIR {
namespace flat {

LabelBuilder::LabelBuilder() : referenced(32), counter{0u} {}

LabelRef LabelBuilder::getNext() {
  LabelRef next{counter++};
  auto cap{referenced.capacity()};
  if (cap < counter) {
    referenced.reserve(2 * cap);
  }
  referenced[next] = false;
  return next;
}

void LabelBuilder::setReferenced(LabelRef label) { referenced[label] = true; }

bool LabelBuilder::isReferenced(LabelRef label) const {
  return referenced[label];
}

LabelOp::LabelOp(LabelBuilder &builder)
  : builder_{builder}, label_{builder.getNext()} {}

LabelOp::LabelOp(const LabelOp &that)
  : builder_{that.builder_}, label_{that.label_} {}

LabelOp &LabelOp::operator=(const LabelOp &that) {
  CHECK(&builder_ == &that.builder_);
  label_ = that.label_;
  return *this;
}

void LabelOp::setReferenced() const { builder_.setReferenced(label_); }

bool LabelOp::isReferenced() const { return builder_.isReferenced(label_); }

static void AddAssign(AnalysisData &ad, const semantics::Symbol *symbol,
    const parser::Label &label) {
  ad.assignMap[symbol].insert(label);
}

std::vector<LabelRef> GetAssign(
    AnalysisData &ad, const semantics::Symbol *symbol) {
  std::vector<LabelRef> result;
  for (auto lab : ad.assignMap[symbol]) {
    result.emplace_back(lab);
  }
  return result;
}

static std::tuple<const parser::Name *, LabelRef, LabelRef> FindStack(
    const std::vector<std::tuple<const parser::Name *, LabelRef, LabelRef>>
        &stack,
    const parser::Name *key) {
  for (auto iter{stack.rbegin()}, iend{stack.rend()}; iter != iend; ++iter) {
    if (std::get<0>(*iter) == key) {
      return *iter;
    }
  }
  SEMANTICS_FAILED("construct name not on stack");
  return {};
}

LabelOp FetchLabel(AnalysisData &ad, const parser::Label &label) {
  auto iter{ad.labelMap.find(label)};
  if (iter == ad.labelMap.end()) {
    LabelOp ll{ad.labelBuilder};
    ll.setReferenced();
    ad.labelMap.insert({label, ll});
    return ll;
  }
  return iter->second;
}

static LabelOp BuildNewLabel(AnalysisData &ad) {
  return LabelOp{ad.labelBuilder};
}

template<typename A> parser::Label GetErr(const A &stmt) {
  if constexpr (std::is_same_v<A, parser::ReadStmt> ||
      std::is_same_v<A, parser::WriteStmt>) {
    for (const auto &control : stmt.controls) {
      if (std::holds_alternative<parser::ErrLabel>(control.u)) {
        return std::get<parser::ErrLabel>(control.u).v;
      }
    }
  }
  if constexpr (std::is_same_v<A, parser::WaitStmt> ||
      std::is_same_v<A, parser::OpenStmt> ||
      std::is_same_v<A, parser::CloseStmt> ||
      std::is_same_v<A, parser::BackspaceStmt> ||
      std::is_same_v<A, parser::EndfileStmt> ||
      std::is_same_v<A, parser::RewindStmt> ||
      std::is_same_v<A, parser::FlushStmt>) {
    for (const auto &spec : stmt.v) {
      if (std::holds_alternative<parser::ErrLabel>(spec.u)) {
        return std::get<parser::ErrLabel>(spec.u).v;
      }
    }
  }
  if constexpr (std::is_same_v<A, parser::InquireStmt>) {
    for (const auto &spec : std::get<std::list<parser::InquireSpec>>(stmt.u)) {
      if (std::holds_alternative<parser::ErrLabel>(spec.u)) {
        return std::get<parser::ErrLabel>(spec.u).v;
      }
    }
  }
  return 0;
}

template<typename A> parser::Label GetEor(const A &stmt) {
  if constexpr (std::is_same_v<A, parser::ReadStmt> ||
      std::is_same_v<A, parser::WriteStmt>) {
    for (const auto &control : stmt.controls) {
      if (std::holds_alternative<parser::EorLabel>(control.u)) {
        return std::get<parser::EorLabel>(control.u).v;
      }
    }
  }
  if constexpr (std::is_same_v<A, parser::WaitStmt>) {
    for (const auto &waitSpec : stmt.v) {
      if (std::holds_alternative<parser::EorLabel>(waitSpec.u)) {
        return std::get<parser::EorLabel>(waitSpec.u).v;
      }
    }
  }
  return 0;
}

template<typename A> parser::Label GetEnd(const A &stmt) {
  if constexpr (std::is_same_v<A, parser::ReadStmt> ||
      std::is_same_v<A, parser::WriteStmt>) {
    for (const auto &control : stmt.controls) {
      if (std::holds_alternative<parser::EndLabel>(control.u)) {
        return std::get<parser::EndLabel>(control.u).v;
      }
    }
  }
  if constexpr (std::is_same_v<A, parser::WaitStmt>) {
    for (const auto &waitSpec : stmt.v) {
      if (std::holds_alternative<parser::EndLabel>(waitSpec.u)) {
        return std::get<parser::EndLabel>(waitSpec.u).v;
      }
    }
  }
  return 0;
}

template<typename A>
void errLabelSpec(const A &s, std::list<Op> &ops,
    const parser::Statement<parser::ActionStmt> &ec, AnalysisData &ad) {
  if (auto errLab{GetErr(s)}) {
    std::optional<LabelRef> errRef{FetchLabel(ad, errLab).get()};
    LabelOp next{BuildNewLabel(ad)};
    ops.emplace_back(SwitchIOOp{s, next, ec.source, errRef});
    ops.emplace_back(next);
  } else {
    ops.emplace_back(ActionOp{ec});
  }
}

template<typename A>
void threeLabelSpec(const A &s, std::list<Op> &ops,
    const parser::Statement<parser::ActionStmt> &ec, AnalysisData &ad) {
  auto errLab{GetErr(s)};
  auto eorLab{GetEor(s)};
  auto endLab{GetEnd(s)};
  if (errLab || eorLab || endLab) {
    std::optional<LabelRef> errRef;
    if (errLab) {
      errRef = FetchLabel(ad, errLab).get();
    }
    std::optional<LabelRef> eorRef;
    if (eorLab) {
      eorRef = FetchLabel(ad, eorLab).get();
    }
    std::optional<LabelRef> endRef;
    if (endLab) {
      endRef = FetchLabel(ad, endLab).get();
    }
    auto next{BuildNewLabel(ad)};
    ops.emplace_back(SwitchIOOp{s, next, ec.source, errRef, eorRef, endRef});
    ops.emplace_back(next);
  } else {
    ops.emplace_back(ActionOp{ec});
  }
}

template<typename A>
std::vector<LabelRef> toLabelRef(AnalysisData &ad, const A &labels) {
  std::vector<LabelRef> result;
  for (auto label : labels) {
    result.emplace_back(FetchLabel(ad, label).get());
  }
  CHECK(result.size() == labels.size());
  return result;
}

template<typename A>
std::vector<LabelRef> toLabelRef(
    const LabelOp &next, AnalysisData &ad, const A &labels) {
  std::vector<LabelRef> result;
  result.emplace_back(next);
  auto refs{toLabelRef(ad, labels)};
  result.insert(result.end(), refs.begin(), refs.end());
  CHECK(result.size() == labels.size() + 1);
  return result;
}

static bool hasAltReturns(const parser::CallStmt &callStmt) {
  const auto &args{std::get<std::list<parser::ActualArgSpec>>(callStmt.v.t)};
  for (const auto &arg : args) {
    const auto &actual{std::get<parser::ActualArg>(arg.t)};
    if (std::holds_alternative<parser::AltReturnSpec>(actual.u)) {
      return true;
    }
  }
  return false;
}

static std::list<parser::Label> getAltReturnLabels(const parser::Call &call) {
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

static LabelRef NearestEnclosingDoConstruct(AnalysisData &ad) {
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

template<typename A> std::string GetSource(const A *s) {
  return s->source.ToString();
}

template<typename A, typename B> std::string GetSource(const B *s) {
  return GetSource(&std::get<parser::Statement<A>>(s->t));
}

void LabelOp::dump() const { DebugChannel() << "label_" << get() << ":\n"; }

void GotoOp::dump() const {
  DebugChannel() << "\tgoto %label_" << target << " ["
                 << std::visit(
                        common::visitors{
                            [](ArtificialJump) { return ""s; },
                            [&](auto *) { return GetSource(this); },
                        },
                        u)
                 << "]\n";
}

void ReturnOp::dump() const {
  DebugChannel() << "\treturn [" << GetSource(this) << "]\n";
}

void ConditionalGotoOp::dump() const {
  DebugChannel() << "\tcbranch .T.: %label_" << trueLabel << " .F.: %label_"
                 << falseLabel << " ["
                 << std::visit(
                        common::visitors{
                            [](const parser::IfStmt *) { return "if-stmt"s; },
                            [&](auto *s) { return GetSource(s); },
                        },
                        u)
                 << "]\n";
}

void SwitchIOOp::dump() const {
  DebugChannel() << "\tio-call";
  if (errLabel.has_value()) {
    DebugChannel() << " ERR: %label_" << errLabel.value();
  }
  if (eorLabel.has_value()) {
    DebugChannel() << " EOR: %label_" << eorLabel.value();
  }
  if (endLabel.has_value()) {
    DebugChannel() << " END: %label_" << endLabel.value();
  }
  DebugChannel() << " [" << GetSource(this) << "]\n";
}

void SwitchOp::dump() const {
  DebugChannel() << "\tswitch [" << GetSource(this) << "]\n";
}

void ActionOp::dump() const { DebugChannel() << '\t' << GetSource(v) << '\n'; }

template<typename A> std::string dumpConstruct(const A &a) {
  return std::visit(
      common::visitors{
          [](const parser::AssociateConstruct *c) {
            return GetSource<parser::AssociateStmt>(c);
          },
          [](const parser::BlockConstruct *c) {
            return GetSource<parser::BlockStmt>(c);
          },
          [](const parser::CaseConstruct *c) {
            return GetSource<parser::SelectCaseStmt>(c);
          },
          [](const parser::ChangeTeamConstruct *c) {
            return GetSource<parser::ChangeTeamStmt>(c);
          },
          [](const parser::CriticalConstruct *c) {
            return GetSource<parser::CriticalStmt>(c);
          },
          [](const parser::DoConstruct *c) {
            return GetSource<parser::NonLabelDoStmt>(c);
          },
          [](const parser::IfConstruct *c) {
            return GetSource<parser::IfThenStmt>(c);
          },
          [](const parser::SelectRankConstruct *c) {
            return GetSource<parser::SelectRankStmt>(c);
          },
          [](const parser::SelectTypeConstruct *c) {
            return GetSource<parser::SelectTypeStmt>(c);
          },
          [](const parser::WhereConstruct *c) {
            return GetSource<parser::WhereConstructStmt>(c);
          },
          [](const parser::ForallConstruct *c) {
            return GetSource(
                &std::get<parser::Statement<parser::ForallConstructStmt>>(
                    c->t));
          },
          [](const parser::CompilerDirective *c) { return GetSource(c); },
          [](const parser::OpenMPConstruct *) { return "openmp"s; },
          [](const parser::OpenMPEndLoopDirective *) {
            return "openmp end loop"s;
          },
      },
      a);
}

void BeginOp::dump() const {
  DebugChannel() << "\t[" << dumpConstruct(u) << "] {\n";
}

void EndOp::dump() const {
  DebugChannel() << "\t} [" << dumpConstruct(u) << "]\n";
}

void IndirectGotoOp::dump() const {
  DebugChannel() << "\tindirect-goto " << symbol->name().ToString() << ':';
  for (auto lab : labelRefs) {
    DebugChannel() << ' ' << lab;
  }
  DebugChannel() << '\n';
}

void DoIncrementOp::dump() const {
  using A = parser::Statement<parser::NonLabelDoStmt>;
  DebugChannel() << "\tincrement [" << GetSource(&std::get<A>(v->t)) << "]\n";
}

void DoCompareOp::dump() const {
  using A = parser::Statement<parser::NonLabelDoStmt>;
  DebugChannel() << "\tcompare [" << GetSource(&std::get<A>(v->t)) << "]\n";
}

void Op::Build(std::list<Op> &ops,
    const parser::Statement<parser::ActionStmt> &ec, AnalysisData &ad) {
  std::visit(
      common::visitors{
          [&](const auto &s) { ops.emplace_back(ActionOp{ec}); },
          [&](const common::Indirection<parser::CallStmt> &s) {
            if (hasAltReturns(s.value())) {
              auto next{BuildNewLabel(ad)};
              auto alts{getAltReturnLabels(s.value().v)};
              auto labels{toLabelRef(next, ad, alts)};
              ops.emplace_back(
                  SwitchOp{s.value(), std::move(labels), ec.source});
              ops.emplace_back(next);
            } else {
              ops.emplace_back(ActionOp{ec});
            }
          },
          [&](const common::Indirection<parser::AssignStmt> &s) {
            AddAssign(ad, std::get<parser::Name>(s.value().t).symbol,
                std::get<parser::Label>(s.value().t));
            ops.emplace_back(ActionOp{ec});
          },
          [&](const common::Indirection<parser::CycleStmt> &s) {
            ops.emplace_back(GotoOp{s.value(),
                s.value().v
                    ? std::get<2>(FindStack(ad.nameStack, &s.value().v.value()))
                    : NearestEnclosingDoConstruct(ad),
                ec.source});
          },
          [&](const common::Indirection<parser::ExitStmt> &s) {
            ops.emplace_back(GotoOp{s.value(),
                s.value().v
                    ? std::get<1>(FindStack(ad.nameStack, &s.value().v.value()))
                    : NearestEnclosingDoConstruct(ad),
                ec.source});
          },
          [&](const common::Indirection<parser::GotoStmt> &s) {
            ops.emplace_back(GotoOp{
                s.value(), FetchLabel(ad, s.value().v).get(), ec.source});
          },
          [&](const parser::FailImageStmt &s) {
            ops.emplace_back(ReturnOp{s, ec.source});
          },
          [&](const common::Indirection<parser::ReturnStmt> &s) {
            ops.emplace_back(ReturnOp{s.value(), ec.source});
          },
          [&](const common::Indirection<parser::StopStmt> &s) {
            ops.emplace_back(ActionOp{ec});
            ops.emplace_back(ReturnOp{s.value(), ec.source});
          },
          [&](const common::Indirection<const parser::ReadStmt> &s) {
            threeLabelSpec(s.value(), ops, ec, ad);
          },
          [&](const common::Indirection<const parser::WriteStmt> &s) {
            threeLabelSpec(s.value(), ops, ec, ad);
          },
          [&](const common::Indirection<const parser::WaitStmt> &s) {
            threeLabelSpec(s.value(), ops, ec, ad);
          },
          [&](const common::Indirection<const parser::OpenStmt> &s) {
            errLabelSpec(s.value(), ops, ec, ad);
          },
          [&](const common::Indirection<const parser::CloseStmt> &s) {
            errLabelSpec(s.value(), ops, ec, ad);
          },
          [&](const common::Indirection<const parser::BackspaceStmt> &s) {
            errLabelSpec(s.value(), ops, ec, ad);
          },
          [&](const common::Indirection<const parser::EndfileStmt> &s) {
            errLabelSpec(s.value(), ops, ec, ad);
          },
          [&](const common::Indirection<const parser::RewindStmt> &s) {
            errLabelSpec(s.value(), ops, ec, ad);
          },
          [&](const common::Indirection<const parser::FlushStmt> &s) {
            errLabelSpec(s.value(), ops, ec, ad);
          },
          [&](const common::Indirection<const parser::InquireStmt> &s) {
            errLabelSpec(s.value(), ops, ec, ad);
          },
          [&](const common::Indirection<parser::ComputedGotoStmt> &s) {
            auto next{BuildNewLabel(ad)};
            auto labels{toLabelRef(
                next, ad, std::get<std::list<parser::Label>>(s.value().t))};
            ops.emplace_back(SwitchOp{s.value(), std::move(labels), ec.source});
            ops.emplace_back(next);
          },
          [&](const common::Indirection<parser::ArithmeticIfStmt> &s) {
            ops.emplace_back(SwitchOp{s.value(),
                toLabelRef(ad,
                    std::list{std::get<1>(s.value().t),
                        std::get<2>(s.value().t), std::get<3>(s.value().t)}),
                ec.source});
          },
          [&](const common::Indirection<parser::AssignedGotoStmt> &s) {
            ops.emplace_back(
                IndirectGotoOp{std::get<parser::Name>(s.value().t).symbol,
                    toLabelRef(
                        ad, std::get<std::list<parser::Label>>(s.value().t))});
          },
          [&](const common::Indirection<parser::IfStmt> &s) {
            auto then{BuildNewLabel(ad)};
            auto endif{BuildNewLabel(ad)};
            ops.emplace_back(ConditionalGotoOp{s.value(), then, endif});
            ops.emplace_back(then);
            ops.emplace_back(ActionOp{ec});
            ops.emplace_back(endif);
          },
      },
      ec.statement.u);
}

template<typename> struct ElementMap;
template<> struct ElementMap<parser::CaseConstruct> {
  using type = parser::CaseConstruct::Case;
};
template<> struct ElementMap<parser::SelectRankConstruct> {
  using type = parser::SelectRankConstruct::RankCase;
};
template<> struct ElementMap<parser::SelectTypeConstruct> {
  using type = parser::SelectTypeConstruct::TypeCase;
};

struct ControlFlowAnalyzer {
  explicit ControlFlowAnalyzer(std::list<Op> &ops, AnalysisData &ad)
    : linearOps{ops}, ad{ad} {}

  LabelOp buildNewLabel() { return BuildNewLabel(ad); }

  Op findLabel(const parser::Label &lab) {
    auto iter{ad.labelMap.find(lab)};
    if (iter == ad.labelMap.end()) {
      LabelOp ll{ad.labelBuilder};
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
      Op::Build(linearOps, stmt, ad);
    }
    return true;
  }
  template<typename A>
  void appendIfLabeled(const parser::Statement<A> &stmt, std::list<Op> &ops) {
    if (stmt.label) {
      ops.emplace_back(findLabel(*stmt.label));
    }
  }

  // named constructs
  template<typename A> bool linearConstruct(const A &construct) {
    std::list<Op> ops;
    LabelOp label{buildNewLabel()};
    const parser::Name *name{getName(construct)};
    ad.nameStack.emplace_back(name, GetLabelRef(label), unspecifiedLabel);
    appendIfLabeled(std::get<0>(construct.t), ops);
    ops.emplace_back(BeginOp{construct});
    ControlFlowAnalyzer cfa{ops, ad};
    Walk(std::get<parser::Block>(construct.t), cfa);
    ops.emplace_back(label);
    appendIfLabeled(std::get<2>(construct.t), ops);
    ops.emplace_back(EndOp{construct});
    linearOps.splice(linearOps.end(), ops);
    ad.nameStack.pop_back();
    return false;
  }

  bool Pre(const parser::AssociateConstruct &c) { return linearConstruct(c); }
  bool Pre(const parser::ChangeTeamConstruct &c) { return linearConstruct(c); }
  bool Pre(const parser::CriticalConstruct &c) { return linearConstruct(c); }

  bool Pre(const parser::BlockConstruct &construct) {
    std::list<Op> ops;
    LabelOp label{buildNewLabel()};
    const auto &optName{
        std::get<parser::Statement<parser::BlockStmt>>(construct.t)
            .statement.v};
    const parser::Name *name{optName ? &*optName : nullptr};
    ad.nameStack.emplace_back(name, GetLabelRef(label), unspecifiedLabel);
    appendIfLabeled(
        std::get<parser::Statement<parser::BlockStmt>>(construct.t), ops);
    ops.emplace_back(BeginOp{construct});
    ControlFlowAnalyzer cfa{ops, ad};
    Walk(std::get<parser::Block>(construct.t), cfa);
    appendIfLabeled(
        std::get<parser::Statement<parser::EndBlockStmt>>(construct.t), ops);
    ops.emplace_back(EndOp{construct});
    ops.emplace_back(label);
    linearOps.splice(linearOps.end(), ops);
    ad.nameStack.pop_back();
    return false;
  }

  bool Pre(const parser::DoConstruct &construct) {
    std::list<Op> ops;
    LabelOp backedgeLab{buildNewLabel()};
    LabelOp incrementLab{buildNewLabel()};
    LabelOp entryLab{buildNewLabel()};
    LabelOp exitLab{buildNewLabel()};
    const parser::Name *name{getName(construct)};
    LabelRef exitOpRef{GetLabelRef(exitLab)};
    ad.nameStack.emplace_back(name, exitOpRef, GetLabelRef(incrementLab));
    appendIfLabeled(
        std::get<parser::Statement<parser::NonLabelDoStmt>>(construct.t), ops);
    ops.emplace_back(BeginOp{construct});
    ops.emplace_back(GotoOp{GetLabelRef(backedgeLab)});
    ops.emplace_back(incrementLab);
    ops.emplace_back(DoIncrementOp{construct});
    ops.emplace_back(backedgeLab);
    ops.emplace_back(DoCompareOp{construct});
    ops.emplace_back(ConditionalGotoOp{
        std::get<parser::Statement<parser::NonLabelDoStmt>>(construct.t),
        GetLabelRef(entryLab), exitOpRef});
    ops.push_back(entryLab);
    ControlFlowAnalyzer cfa{ops, ad};
    Walk(std::get<parser::Block>(construct.t), cfa);
    appendIfLabeled(
        std::get<parser::Statement<parser::EndDoStmt>>(construct.t), ops);
    ops.emplace_back(GotoOp{GetLabelRef(incrementLab)});
    ops.emplace_back(EndOp{construct});
    ops.emplace_back(exitLab);
    linearOps.splice(linearOps.end(), ops);
    ad.nameStack.pop_back();
    return false;
  }

  bool Pre(const parser::IfConstruct &construct) {
    std::list<Op> ops;
    LabelOp thenLab{buildNewLabel()};
    LabelOp elseLab{buildNewLabel()};
    LabelOp exitLab{buildNewLabel()};
    const parser::Name *name{getName(construct)};
    ad.nameStack.emplace_back(name, GetLabelRef(exitLab), unspecifiedLabel);
    appendIfLabeled(
        std::get<parser::Statement<parser::IfThenStmt>>(construct.t), ops);
    ops.emplace_back(BeginOp{construct});
    ops.emplace_back(ConditionalGotoOp{
        std::get<parser::Statement<parser::IfThenStmt>>(construct.t),
        GetLabelRef(thenLab), GetLabelRef(elseLab)});
    ops.emplace_back(thenLab);
    ControlFlowAnalyzer cfa{ops, ad};
    Walk(std::get<parser::Block>(construct.t), cfa);
    LabelRef exitOpRef{GetLabelRef(exitLab)};
    ops.emplace_back(GotoOp{exitOpRef});
    for (const auto &elseIfBlock :
        std::get<std::list<parser::IfConstruct::ElseIfBlock>>(construct.t)) {
      appendIfLabeled(
          std::get<parser::Statement<parser::ElseIfStmt>>(elseIfBlock.t), ops);
      ops.emplace_back(elseLab);
      LabelOp newThenLab{buildNewLabel()};
      LabelOp newElseLab{buildNewLabel()};
      ops.emplace_back(ConditionalGotoOp{
          std::get<parser::Statement<parser::ElseIfStmt>>(elseIfBlock.t),
          GetLabelRef(newThenLab), GetLabelRef(newElseLab)});
      ops.emplace_back(newThenLab);
      Walk(std::get<parser::Block>(elseIfBlock.t), cfa);
      ops.emplace_back(GotoOp{exitOpRef});
      elseLab = newElseLab;
    }
    ops.emplace_back(elseLab);
    if (const auto &optElseBlock{
            std::get<std::optional<parser::IfConstruct::ElseBlock>>(
                construct.t)}) {
      appendIfLabeled(
          std::get<parser::Statement<parser::ElseStmt>>(optElseBlock->t), ops);
      Walk(std::get<parser::Block>(optElseBlock->t), cfa);
    }
    ops.emplace_back(GotoOp{exitOpRef});
    ops.emplace_back(exitLab);
    appendIfLabeled(
        std::get<parser::Statement<parser::EndIfStmt>>(construct.t), ops);
    ops.emplace_back(EndOp{construct});
    linearOps.splice(linearOps.end(), ops);
    ad.nameStack.pop_back();
    return false;
  }

  template<typename A> bool Multiway(const A &construct) {
    using B = typename ElementMap<A>::type;
    std::list<Op> ops;
    LabelOp exitLab{buildNewLabel()};
    const parser::Name *name{getName(construct)};
    ad.nameStack.emplace_back(name, GetLabelRef(exitLab), unspecifiedLabel);
    appendIfLabeled(std::get<0>(construct.t), ops);
    ops.emplace_back(BeginOp{construct});
    const auto N{std::get<std::list<B>>(construct.t).size()};
    LabelRef exitOpRef{GetLabelRef(exitLab)};
    if (N > 0) {
      typename std::list<B>::size_type i;
      std::vector<LabelOp> toLabels;
      for (i = 0; i != N; ++i) {
        toLabels.emplace_back(buildNewLabel());
      }
      std::vector<LabelRef> targets;
      for (i = 0; i != N; ++i) {
        targets.emplace_back(GetLabelRef(toLabels[i]));
      }
      ops.emplace_back(
          SwitchOp{construct, targets, std::get<0>(construct.t).source});
      ControlFlowAnalyzer cfa{ops, ad};
      i = 0;
      for (const auto &caseBlock : std::get<std::list<B>>(construct.t)) {
        ops.emplace_back(toLabels[i++]);
        appendIfLabeled(std::get<0>(caseBlock.t), ops);
        Walk(std::get<parser::Block>(caseBlock.t), cfa);
        ops.emplace_back(GotoOp{exitOpRef});
      }
    }
    ops.emplace_back(exitLab);
    appendIfLabeled(std::get<2>(construct.t), ops);
    ops.emplace_back(EndOp{construct});
    linearOps.splice(linearOps.end(), ops);
    ad.nameStack.pop_back();
    return false;
  }

  bool Pre(const parser::CaseConstruct &c) { return Multiway(c); }
  bool Pre(const parser::SelectRankConstruct &c) { return Multiway(c); }
  bool Pre(const parser::SelectTypeConstruct &c) { return Multiway(c); }

  bool Pre(const parser::WhereConstruct &c) {
    std::list<Op> ops;
    LabelOp label{buildNewLabel()};
    const parser::Name *name{getName(c)};
    ad.nameStack.emplace_back(name, GetLabelRef(label), unspecifiedLabel);
    appendIfLabeled(
        std::get<parser::Statement<parser::WhereConstructStmt>>(c.t), ops);
    ops.emplace_back(BeginOp{c});
    ControlFlowAnalyzer cfa{ops, ad};
    Walk(std::get<std::list<parser::WhereBodyConstruct>>(c.t), cfa);
    Walk(
        std::get<std::list<parser::WhereConstruct::MaskedElsewhere>>(c.t), cfa);
    Walk(std::get<std::optional<parser::WhereConstruct::Elsewhere>>(c.t), cfa);
    ops.emplace_back(label);
    appendIfLabeled(
        std::get<parser::Statement<parser::EndWhereStmt>>(c.t), ops);
    ops.emplace_back(EndOp{c});
    linearOps.splice(linearOps.end(), ops);
    ad.nameStack.pop_back();
    return false;
  }

  bool Pre(const parser::ForallConstruct &construct) {
    std::list<Op> ops;
    LabelOp label{buildNewLabel()};
    const parser::Name *name{getName(construct)};
    ad.nameStack.emplace_back(name, GetLabelRef(label), unspecifiedLabel);
    appendIfLabeled(
        std::get<parser::Statement<parser::ForallConstructStmt>>(construct.t),
        ops);
    ops.emplace_back(BeginOp{construct});
    ControlFlowAnalyzer cfa{ops, ad};
    Walk(std::get<std::list<parser::ForallBodyConstruct>>(construct.t), cfa);
    ops.emplace_back(label);
    appendIfLabeled(
        std::get<parser::Statement<parser::EndForallStmt>>(construct.t), ops);
    ops.emplace_back(EndOp{construct});
    linearOps.splice(linearOps.end(), ops);
    ad.nameStack.pop_back();
    return false;
  }

  template<typename A> const parser::Name *getName(const A &a) {
    const auto &optName{std::get<0>(std::get<0>(a.t).statement.t)};
    return optName ? &*optName : nullptr;
  }

  LabelRef GetLabelRef(const LabelOp &label) {
    label.setReferenced();
    return label;
  }

  LabelRef GetLabelRef(const parser::Label &label) {
    return FetchLabel(ad, label);
  }

  std::list<Op> &linearOps;
  AnalysisData &ad;
};
}

void dump(const std::list<flat::Op> &ops) {
  for (auto &op : ops) {
    op.dump();
  }
}

template<typename A>
void CreateFlatIR(const A &ptree, std::list<flat::Op> &ops, AnalysisData &ad) {
  flat::ControlFlowAnalyzer linearize{ops, ad};
  Walk(ptree, linearize);
}

#define INSTANTIATE_EXPLICITLY(T) \
  template void CreateFlatIR<parser::T>( \
      const parser::T &, std::list<flat::Op> &, AnalysisData &)
INSTANTIATE_EXPLICITLY(MainProgram);
INSTANTIATE_EXPLICITLY(FunctionSubprogram);
INSTANTIATE_EXPLICITLY(SubroutineSubprogram);
}
