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

#ifndef FORTRAN_FIR_FLATTENED_H_
#define FORTRAN_FIR_FLATTENED_H_

#include "common.h"
#include "mixin.h"
#include "../parser/parse-tree.h"
#include <list>
#include <map>
#include <set>
#include <vector>

namespace Fortran::FIR {

struct AnalysisData;

namespace flat {

// This is a flattened, linearized representation of the parse tree. It captures
// the executable specification of the input program. The flattened IR can be
// used to construct the Fortran IR.

struct Op;
struct LabelOp;
struct GotoOp;
struct ReturnOp;
struct ConditionalGotoOp;
struct SwitchIOOp;
struct SwitchOp;
struct ActionOp;
struct BeginOp;
struct EndOp;
struct IndirectGotoOp;
struct DoIncrementOp;
struct DoCompareOp;

using LabelRef = unsigned;
constexpr LabelRef unspecifiedLabel{~0u};

using Location = parser::CharBlock;
struct LabelBuilder;

// target for a control-flow edge
struct LabelOp {
  explicit LabelOp(LabelBuilder &builder);
  LabelOp(const LabelOp &that);
  LabelOp &operator=(const LabelOp &that);
  void setReferenced() const;
  bool isReferenced() const;
  LabelRef get() const { return label_; }
  operator LabelRef() const { return get(); }
  void dump() const;

private:
  LabelBuilder &builder_;
  LabelRef label_;
};

struct ArtificialJump {};
constexpr ArtificialJump ARTIFICIAL{};

// a source of an absolute control flow edge
struct GotoOp
  : public SumTypeCopyMixin<const parser::CycleStmt *, const parser::ExitStmt *,
        const parser::GotoStmt *, ArtificialJump> {
  template<typename A>
  GotoOp(const A &stmt, LabelRef dest, const Location &source)
    : SumTypeCopyMixin{&stmt}, target{dest}, source{source} {}
  explicit GotoOp(LabelRef dest) : SumTypeCopyMixin{ARTIFICIAL}, target{dest} {}
  void dump() const;

  LabelRef target;
  Location source;
};

// control exits the procedure
struct ReturnOp : public SumTypeCopyMixin<const parser::FailImageStmt *,
                      const parser::ReturnStmt *, const parser::StopStmt *> {
  template<typename A>
  ReturnOp(const A &stmt, const Location &source)
    : SumTypeCopyMixin{&stmt}, source{source} {}
  void dump() const;

  Location source;
};

// two-way branch based on a condition
struct ConditionalGotoOp
  : public SumTypeCopyMixin<const parser::Statement<parser::IfThenStmt> *,
        const parser::Statement<parser::ElseIfStmt> *, const parser::IfStmt *,
        const parser::Statement<parser::NonLabelDoStmt> *> {
  template<typename A>
  ConditionalGotoOp(const A &cond, LabelRef tb, LabelRef fb)
    : SumTypeCopyMixin{&cond}, trueLabel{tb}, falseLabel{fb} {}
  void dump() const;

  LabelRef trueLabel;
  LabelRef falseLabel;
};

// multi-way branch based on a target-value of a variable
struct IndirectGotoOp {
  IndirectGotoOp(
      const semantics::Symbol *symbol, std::vector<LabelRef> &&labelRefs)
    : symbol{symbol}, labelRefs{labelRefs} {}
  void dump() const;

  const semantics::Symbol *symbol;
  std::vector<LabelRef> labelRefs;
};

// intrinsic IO operations can return with an implied multi-way branch
struct SwitchIOOp
  : public SumTypeCopyMixin<const parser::ReadStmt *, const parser::WriteStmt *,
        const parser::WaitStmt *, const parser::OpenStmt *,
        const parser::CloseStmt *, const parser::BackspaceStmt *,
        const parser::EndfileStmt *, const parser::RewindStmt *,
        const parser::FlushStmt *, const parser::InquireStmt *> {
  template<typename A>
  SwitchIOOp(const A &io, LabelRef next, const Location &source,
      std::optional<LabelRef> errLab,
      std::optional<LabelRef> eorLab = std::nullopt,
      std::optional<LabelRef> endLab = std::nullopt)
    : SumTypeCopyMixin{&io}, next{next}, source{source}, errLabel{errLab},
      eorLabel{eorLab}, endLabel{endLab} {}
  void dump() const;

  LabelRef next;
  Location source;
  std::optional<LabelRef> errLabel;
  std::optional<LabelRef> eorLabel;
  std::optional<LabelRef> endLabel;
};

// multi-way branch based on conditions
struct SwitchOp
  : public SumTypeCopyMixin<const parser::CallStmt *,
        const parser::ComputedGotoStmt *, const parser::ArithmeticIfStmt *,
        const parser::CaseConstruct *, const parser::SelectRankConstruct *,
        const parser::SelectTypeConstruct *> {
  template<typename A>
  SwitchOp(
      const A &sw, const std::vector<LabelRef> &refs, const Location &source)
    : SumTypeCopyMixin{&sw}, refs{refs}, source{source} {}
  void dump() const;

  const std::vector<LabelRef> refs;
  Location source;
};

// a compute step
struct ActionOp {
  ActionOp(const parser::Statement<parser::ActionStmt> &stmt) : v{&stmt} {}
  void dump() const;

  const parser::Statement<parser::ActionStmt> *v;
};

#define CONSTRUCT_TYPES \
  const parser::AssociateConstruct *, const parser::BlockConstruct *, \
      const parser::CaseConstruct *, const parser::ChangeTeamConstruct *, \
      const parser::CriticalConstruct *, const parser::DoConstruct *, \
      const parser::IfConstruct *, const parser::SelectRankConstruct *, \
      const parser::SelectTypeConstruct *, const parser::WhereConstruct *, \
      const parser::ForallConstruct *, const parser::CompilerDirective *, \
      const parser::OpenMPConstruct *, const parser::OpenMPEndLoopDirective *

// entry into a Fortran construct
struct BeginOp : public SumTypeCopyMixin<CONSTRUCT_TYPES> {
  SUM_TYPE_COPY_MIXIN(BeginOp)
  void dump() const;

  template<typename A> BeginOp(const A &c) : SumTypeCopyMixin{&c} {}
};

// exit from a Fortran construct
struct EndOp : public SumTypeCopyMixin<CONSTRUCT_TYPES> {
  SUM_TYPE_COPY_MIXIN(EndOp)
  void dump() const;

  template<typename A> EndOp(const A &c) : SumTypeCopyMixin{&c} {}
};

struct DoIncrementOp {
  DoIncrementOp(const parser::DoConstruct &stmt) : v{&stmt} {}
  void dump() const;

  const parser::DoConstruct *v;
};

struct DoCompareOp {
  DoCompareOp(const parser::DoConstruct &stmt) : v{&stmt} {}
  void dump() const;

  const parser::DoConstruct *v;
};

// the flat FIR is a list of Ops, where an Op is any of ...
struct Op : public SumTypeMixin<LabelOp, GotoOp, ReturnOp, ConditionalGotoOp,
                SwitchIOOp, SwitchOp, ActionOp, BeginOp, EndOp, IndirectGotoOp,
                DoIncrementOp, DoCompareOp> {
  template<typename A> Op(const A &thing) : SumTypeMixin{thing} {}

  void dump() const {
    std::visit([](const auto &op) { op.dump(); }, u);
  }

  static void Build(std::list<Op> &ops,
      const parser::Statement<parser::ActionStmt> &ec, AnalysisData &ad);
};

// helper to build unique labels
struct LabelBuilder {
  LabelBuilder();
  LabelRef getNext();
  void setReferenced(LabelRef label);
  bool isReferenced(LabelRef label) const;
  std::vector<bool> referenced;
  unsigned counter;
};

LabelOp FetchLabel(AnalysisData &ad, const parser::Label &label);

std::vector<LabelRef> GetAssign(
    AnalysisData &ad, const semantics::Symbol *symbol);
}

struct AnalysisData {
  std::map<parser::Label, flat::LabelOp> labelMap;
  std::vector<std::tuple<const parser::Name *, flat::LabelRef, flat::LabelRef>>
      nameStack;
  flat::LabelBuilder labelBuilder;
  std::map<const semantics::Symbol *, std::set<parser::Label>> assignMap;
};

// entry-point into building the flat IR
template<typename A>
void CreateFlatIR(const A &ptree, std::list<flat::Op> &ops, AnalysisData &ad);

#define EXPLICIT_INSTANTIATION(T) \
  extern template void CreateFlatIR<parser::T>( \
      const parser::T &, std::list<flat::Op> &, AnalysisData &)
EXPLICIT_INSTANTIATION(MainProgram);
EXPLICIT_INSTANTIATION(FunctionSubprogram);
EXPLICIT_INSTANTIATION(SubroutineSubprogram);

// dump flat IR
void dump(const std::list<flat::Op> &ops);
}

#endif  // FORTRAN_FIR_FLATTENED_H_
