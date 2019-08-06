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

// OpenMP structure validity check list
//    1. invalid clauses on directive
//    2. invalid repeated clauses on directive
//    3. TODO: invalid nesting of regions

#ifndef FORTRAN_SEMANTICS_CHECK_OMP_STRUCTURE_H_
#define FORTRAN_SEMANTICS_CHECK_OMP_STRUCTURE_H_

#include "semantics.h"
#include "../common/enum-set.h"
#include "../parser/parse-tree.h"

namespace Fortran::semantics {

ENUM_CLASS(OmpDirective, PARALLEL, DO, SECTIONS, SECTION, SINGLE, END_SINGLE,
    WORKSHARE, SIMD, DECLARE_SIMD, DO_SIMD, TASK, TASKLOOP, TASKLOOP_SIMD,
    TASKYIELD, TARGET_DATA, TARGET_ENTER_DATA, TARGET_EXIT_DATA, TARGET,
    TARGET_UPDATE, DECLARE_TARGET, TEAMS, DISTRIBUTE, DISTRIBUTE_SIMD,
    DISTRIBUTE_PARALLEL_DO, DISTRIBUTE_PARALLEL_DO_SIMD, PARALLEL_DO,
    PARALLEL_SECTIONS, PARALLEL_WORKSHARE, PARALLEL_DO_SIMD, TARGET_PARALLEL,
    TARGET_PARALLEL_DO, TARGET_PARALLEL_DO_SIMD, TARGET_SIMD, TARGET_TEAMS,
    TEAMS_DISTRIBUTE, TEAMS_DISTRIBUTE_SIMD, TARGET_TEAMS_DISTRIBUTE,
    TARGET_TEAMS_DISTRIBUTE_SIMD, TEAMS_DISTRIBUTE_PARALLEL_DO,
    TARGET_TEAMS_DISTRIBUTE_PARALLEL_DO, TEAMS_DISTRIBUTE_PARALLEL_DO_SIMD,
    TARGET_TEAMS_DISTRIBUTE_PARALLEL_DO_SIMD, MASTER, CRITICAL, BARRIER,
    TASKWAIT, TASKGROUP, ATOMIC, FLUSH, ORDERED, CANCEL, CANCELLATION_POINT,
    THREADPRIVATE, DECLARE_REDUCTION)

using OmpDirectiveSet = common::EnumSet<OmpDirective, OmpDirective_enumSize>;

ENUM_CLASS(OmpClause, DEFAULTMAP, INBRANCH, MERGEABLE, NOGROUP, NOTINBRANCH,
    NOWAIT, UNTIED, THREADS, SIMD, COLLAPSE, COPYIN, COPYPRIVATE, DEVICE,
    DIST_SCHEDULE, FINAL, FIRSTPRIVATE, FROM, GRAINSIZE, LASTPRIVATE, NUM_TASKS,
    NUM_TEAMS, NUM_THREADS, ORDERED, PRIORITY, PRIVATE, SAFELEN, SHARED,
    SIMDLEN, THREAD_LIMIT, TO, LINK, UNIFORM, USE_DEVICE_PTR, IS_DEVICE_PTR,
    ALIGNED, DEFAULT, DEPEND, IF, LINEAR, MAP, PROC_BIND, REDUCTION, SCHEDULE)

using OmpClauseSet = common::EnumSet<OmpClause, OmpClause_enumSize>;

class OmpStructureChecker : public virtual BaseChecker {
public:
  OmpStructureChecker(SemanticsContext &context) : context_{context} {}

  void Enter(const parser::OpenMPConstruct &);
  void Enter(const parser::OpenMPLoopConstruct &);
  void Leave(const parser::OpenMPLoopConstruct &);
  void Enter(const parser::OmpLoopDirective &);

  void Enter(const parser::OpenMPBlockConstruct &);
  void Leave(const parser::OpenMPBlockConstruct &);
  void Enter(const parser::OmpBlockDirective &);

  void Enter(const parser::OpenMPSectionsConstruct &);
  void Leave(const parser::OpenMPSectionsConstruct &);
  void Enter(const parser::OmpSection &);

  void Enter(const parser::OpenMPSingleConstruct &);
  void Leave(const parser::OpenMPSingleConstruct &);
  void Enter(const parser::OmpEndSingle &);
  void Leave(const parser::OmpEndSingle &);

  void Enter(const parser::OpenMPWorkshareConstruct &);
  void Leave(const parser::OpenMPWorkshareConstruct &);

  void Enter(const parser::OpenMPDeclareSimdConstruct &);
  void Leave(const parser::OpenMPDeclareSimdConstruct &);

  void Enter(const parser::OpenMPSimpleStandaloneConstruct &);
  void Leave(const parser::OpenMPSimpleStandaloneConstruct &);
  void Enter(const parser::OmpSimpleStandaloneDirective &);
  void Enter(const parser::OpenMPFlushConstruct &);
  void Leave(const parser::OpenMPFlushConstruct &);
  void Enter(const parser::OpenMPCancelConstruct &);
  void Leave(const parser::OpenMPCancelConstruct &);
  void Enter(const parser::OpenMPCancellationPointConstruct &);
  void Leave(const parser::OpenMPCancellationPointConstruct &);

  void Leave(const parser::OmpClauseList &);
  void Enter(const parser::OmpClause &);
  void Enter(const parser::OmpNowait &);
  void Enter(const parser::OmpClause::Defaultmap &);
  void Enter(const parser::OmpClause::Inbranch &);
  void Enter(const parser::OmpClause::Mergeable &);
  void Enter(const parser::OmpClause::Nogroup &);
  void Enter(const parser::OmpClause::Notinbranch &);
  void Enter(const parser::OmpClause::Untied &);
  void Enter(const parser::OmpClause::Collapse &);
  void Enter(const parser::OmpClause::Copyin &);
  void Enter(const parser::OmpClause::Copyprivate &);
  void Enter(const parser::OmpClause::Device &);
  void Enter(const parser::OmpClause::DistSchedule &);
  void Enter(const parser::OmpClause::Final &);
  void Enter(const parser::OmpClause::Firstprivate &);
  void Enter(const parser::OmpClause::From &);
  void Enter(const parser::OmpClause::Grainsize &);
  void Enter(const parser::OmpClause::Lastprivate &);
  void Enter(const parser::OmpClause::NumTasks &);
  void Enter(const parser::OmpClause::NumTeams &);
  void Enter(const parser::OmpClause::NumThreads &);
  void Enter(const parser::OmpClause::Ordered &);
  void Enter(const parser::OmpClause::Priority &);
  void Enter(const parser::OmpClause::Private &);
  void Enter(const parser::OmpClause::Safelen &);
  void Enter(const parser::OmpClause::Shared &);
  void Enter(const parser::OmpClause::Simdlen &);
  void Enter(const parser::OmpClause::ThreadLimit &);
  void Enter(const parser::OmpClause::To &);
  void Enter(const parser::OmpClause::Link &);
  void Enter(const parser::OmpClause::Uniform &);
  void Enter(const parser::OmpClause::UseDevicePtr &);
  void Enter(const parser::OmpClause::IsDevicePtr &);

  void Enter(const parser::OmpAlignedClause &);
  void Enter(const parser::OmpDefaultClause &);
  void Enter(const parser::OmpDependClause &);
  void Enter(const parser::OmpIfClause &);
  void Enter(const parser::OmpLinearClause &);
  void Enter(const parser::OmpMapClause &);
  void Enter(const parser::OmpProcBindClause &);
  void Enter(const parser::OmpReductionClause &);
  void Enter(const parser::OmpScheduleClause &);

private:
  struct OmpContext {
    parser::CharBlock directiveSource{nullptr};
    parser::CharBlock clauseSource{nullptr};
    OmpDirective directive;
    OmpClauseSet allowedClauses;
    OmpClauseSet allowedOnceClauses;

    const parser::OmpClause *clause{nullptr};
    std::multimap<OmpClause, const parser::OmpClause *> clauseInfo;
  };
  // back() is the top of the stack
  OmpContext &GetContext() {
    CHECK(!ompContext_.empty());
    return ompContext_.back();
  }
  void SetContextDirectiveSource(const parser::CharBlock &directive) {
    GetContext().directiveSource = directive;
  }
  void SetContextClause(const parser::OmpClause &clause) {
    GetContext().clauseSource = clause.source;
    GetContext().clause = &clause;
  }
  void SetContextDirectiveEnum(const OmpDirective &dir) {
    GetContext().directive = dir;
  }
  void SetContextAllowed(const OmpClauseSet &allowed) {
    GetContext().allowedClauses = allowed;
  }
  void SetContextAllowedOnce(const OmpClauseSet &allowedOnce) {
    GetContext().allowedOnceClauses = allowedOnce;
  }
  void SetContextClauseInfo(const OmpClause &type) {
    GetContext().clauseInfo.emplace(type, GetContext().clause);
  }
  const parser::OmpClause *FindClause(const OmpClause &type) {
    auto it{GetContext().clauseInfo.find(type)};
    if (it != GetContext().clauseInfo.end()) {
      return it->second;
    }
    return nullptr;
  }
  void PushContext(const parser::CharBlock &source) {
    ompContext_.push_back(OmpContext{source});
  }
  void PushContext(const parser::CharBlock &source, const OmpDirective &dir) {
    PushContext(source);
    SetContextDirectiveEnum(dir);
  }

  bool CurrentDirectiveIsNested() { return ompContext_.size() > 0; };
  bool HasInvalidWorksharingNesting(
      const parser::CharBlock &, const OmpDirectiveSet &);
  void CheckAllowed(const OmpClause &);
  std::string ContextDirectiveAsFortran();

  // specific clause related
  bool ScheduleModifierHasType(const parser::OmpScheduleClause &,
      const parser::OmpScheduleModifierType::ModType &);

  SemanticsContext &context_;
  std::vector<OmpContext> ompContext_;  // used as a stack
};
}
#endif  // FORTRAN_SEMANTICS_CHECK_OMP_STRUCTURE_H_
