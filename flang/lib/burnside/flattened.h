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

#ifndef FORTRAN_BURNSIDE_FLATTENED_H_
#define FORTRAN_BURNSIDE_FLATTENED_H_

#include "mixin.h"
#include "../parser/parse-tree.h"
#include "llvm/ADT/BitVector.h"
#include <cstdint>
#include <list>
#include <map>
#include <set>
#include <vector>

namespace Fortran::burnside {

struct AnalysisData;

namespace flat {

/// This is a flattened, linearized representation of the parse
/// tree. It captures the executable specification of the input
/// program. The flattened IR can be used to construct FIR.
///
/// [Coding style](https://llvm.org/docs/CodingStandards.html)

using LabelRef = unsigned;
constexpr LabelRef UnspecifiedLabel{UINT_MAX};

using Location = parser::CharBlock;
struct LabelBuilder;

// target for a control-flow edge
struct LabelOp {
  explicit LabelOp(LabelBuilder &builder);
  LabelOp(const LabelOp &that);
  LabelOp &operator=(const LabelOp &that);
  void setReferenced() const;
  bool isReferenced() const;
  LabelRef get() const { return label; }
  operator LabelRef() const { return get(); }

private:
  LabelBuilder &builder;
  LabelRef label;
};

struct ArtificialJump {};

// a source of an absolute control flow edge
struct GotoOp
  : public SumTypeCopyMixin<const parser::CycleStmt *, const parser::ExitStmt *,
        const parser::GotoStmt *, ArtificialJump> {
  template<typename A>
  explicit GotoOp(const A &stmt, LabelRef dest, const Location &source)
    : SumTypeCopyMixin{&stmt}, target{dest}, source{source} {}
  explicit GotoOp(LabelRef dest)
    : SumTypeCopyMixin{ArtificialJump{}}, target{dest} {}

  LabelRef target;
  Location source;
};

// control exits the procedure
struct ReturnOp : public SumTypeCopyMixin<const parser::FailImageStmt *,
                      const parser::ReturnStmt *, const parser::StopStmt *> {
  template<typename A>
  explicit ReturnOp(const A &stmt, const Location &source)
    : SumTypeCopyMixin{&stmt}, source{source} {}

  Location source;
};

// two-way branch based on a condition
struct ConditionalGotoOp
  : public SumTypeCopyMixin<const parser::Statement<parser::IfThenStmt> *,
        const parser::Statement<parser::ElseIfStmt> *, const parser::IfStmt *,
        const parser::Statement<parser::NonLabelDoStmt> *> {
  template<typename A>
  explicit ConditionalGotoOp(const A &cond, LabelRef tb, LabelRef fb)
    : SumTypeCopyMixin{&cond}, trueLabel{tb}, falseLabel{fb} {}

  LabelRef trueLabel;
  LabelRef falseLabel;
};

// multi-way branch based on a target-value of a variable
struct IndirectGotoOp {
  explicit IndirectGotoOp(
      const semantics::Symbol *symbol, std::vector<LabelRef> &&labelRefs)
      : labelRefs{labelRefs}, symbol{symbol} {}

  std::vector<LabelRef> labelRefs;
  const semantics::Symbol *symbol;
};

// intrinsic IO operations can return with an implied multi-way branch
struct SwitchIOOp
  : public SumTypeCopyMixin<const parser::ReadStmt *, const parser::WriteStmt *,
        const parser::WaitStmt *, const parser::OpenStmt *,
        const parser::CloseStmt *, const parser::BackspaceStmt *,
        const parser::EndfileStmt *, const parser::RewindStmt *,
        const parser::FlushStmt *, const parser::InquireStmt *> {
  template<typename A>
  explicit SwitchIOOp(const A &io, LabelRef next, const Location &source,
      std::optional<LabelRef> errLab,
      std::optional<LabelRef> eorLab = std::nullopt,
      std::optional<LabelRef> endLab = std::nullopt)
    : SumTypeCopyMixin{&io}, next{next}, source{source}, errLabel{errLab},
      eorLabel{eorLab}, endLabel{endLab} {}
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
  explicit SwitchOp(
      const A &sw, const std::vector<LabelRef> &refs, const Location &source)
    : SumTypeCopyMixin{&sw}, refs{refs}, source{source} {}

  const std::vector<LabelRef> refs;
  Location source;
};

// a compute step
struct ActionOp {
  explicit ActionOp(const parser::Statement<parser::ActionStmt> &stmt)
    : v{&stmt} {}

  const parser::Statement<parser::ActionStmt> *v;
};

#define CONSTRUCT_TYPES \
  const parser::AssociateConstruct *, const parser::BlockConstruct *, \
      const parser::CaseConstruct *, const parser::ChangeTeamConstruct *, \
      const parser::CriticalConstruct *, const parser::DoConstruct *, \
      const parser::IfConstruct *, const parser::SelectRankConstruct *, \
      const parser::SelectTypeConstruct *, const parser::WhereConstruct *, \
      const parser::ForallConstruct *, const parser::CompilerDirective *, \
      const parser::OpenMPConstruct *, const parser::OmpEndLoopDirective *

// entry into a Fortran construct
struct BeginOp : public SumTypeCopyMixin<CONSTRUCT_TYPES> {
  SUM_TYPE_COPY_MIXIN(BeginOp)

  template<typename A> explicit BeginOp(const A &c) : SumTypeCopyMixin{&c} {}
};

// exit from a Fortran construct
struct EndOp : public SumTypeCopyMixin<CONSTRUCT_TYPES> {
  SUM_TYPE_COPY_MIXIN(EndOp)

  template<typename A> explicit EndOp(const A &c) : SumTypeCopyMixin{&c} {}
};

struct DoIncrementOp {
  explicit DoIncrementOp(const parser::DoConstruct &stmt) : v{&stmt} {}

  const parser::DoConstruct *v;
};

struct DoCompareOp {
  DoCompareOp(const parser::DoConstruct &stmt) : v{&stmt} {}

  const parser::DoConstruct *v;
};

// the flat structure is a list of Ops, where an Op is any of ...
struct Op : public SumTypeMixin<LabelOp, GotoOp, ReturnOp, ConditionalGotoOp,
                SwitchIOOp, SwitchOp, ActionOp, BeginOp, EndOp, IndirectGotoOp,
                DoIncrementOp, DoCompareOp> {
  template<typename A> Op(const A &thing) : SumTypeMixin{thing} {}

  static void Build(std::list<Op> &ops,
      const parser::Statement<parser::ActionStmt> &ec, AnalysisData &ad);
};

// helper to build unique labels
struct LabelBuilder {
  LabelBuilder();
  LabelRef getNext();
  void setReferenced(LabelRef label);
  bool isReferenced(LabelRef label) const;
  llvm::BitVector referenced;
  unsigned counter;
};

LabelOp FetchLabel(AnalysisData &ad, const parser::Label &label);

std::vector<LabelRef> GetAssign(
    AnalysisData &ad, const semantics::Symbol *symbol);

}  // namespace flat

// Collection of data maintained internally by the flattening algorithm
struct AnalysisData {
  std::map<parser::Label, flat::LabelOp> labelMap;
  std::vector<std::tuple<const parser::Name *, flat::LabelRef, flat::LabelRef>>
      constructContextStack;
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

}  // namespace burnside

#endif  // FORTRAN_BURNSIDE_FLATTENED_H_
