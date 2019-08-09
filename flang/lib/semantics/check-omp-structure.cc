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

#include "check-omp-structure.h"
#include "tools.h"
#include "../parser/parse-tree.h"

namespace Fortran::semantics {

static constexpr OmpDirectiveSet doSet{OmpDirective::DO,
    OmpDirective::PARALLEL_DO, OmpDirective::DO_SIMD,
    OmpDirective::PARALLEL_DO_SIMD};
static constexpr OmpDirectiveSet simdSet{
    OmpDirective::SIMD, OmpDirective::DO_SIMD, OmpDirective::PARALLEL_DO_SIMD};
static constexpr OmpDirectiveSet doSimdSet{
    OmpDirective::DO_SIMD, OmpDirective::PARALLEL_DO_SIMD};

std::string OmpStructureChecker::ContextDirectiveAsFortran() {
  auto dir{EnumToString(GetContext().directive)};
  std::replace(dir.begin(), dir.end(), '_', ' ');
  return dir;
}

bool OmpStructureChecker::HasInvalidWorksharingNesting(
    const parser::CharBlock &source, const OmpDirectiveSet &set) {
  // set contains all the invalid closely nested directives
  // for the given directive (`source` here)
  if (CurrentDirectiveIsNested() && set.test(GetContext().directive)) {
    context_.Say(source,
        "A worksharing region may not be closely nested inside a "
        "worksharing, explicit task, taskloop, critical, ordered, atomic, or "
        "master region."_err_en_US);
    return true;
  }
  return false;
}

void OmpStructureChecker::CheckAllowed(const OmpClause &type) {
  if (!GetContext().allowedClauses.test(type) &&
      !GetContext().allowedOnceClauses.test(type)) {
    context_.Say(GetContext().clauseSource,
        "%s clause is not allowed on the %s directive"_err_en_US,
        EnumToString(type),
        parser::ToUpperCaseLetters(GetContext().directiveSource.ToString()));
    return;
  }
  if (GetContext().allowedOnceClauses.test(type) && FindClause(type)) {
    context_.Say(GetContext().clauseSource,
        "At most one %s clause can appear on the %s directive"_err_en_US,
        EnumToString(type),
        parser::ToUpperCaseLetters(GetContext().directiveSource.ToString()));
    return;
  }
  SetContextClauseInfo(type);
}

void OmpStructureChecker::Enter(const parser::OpenMPConstruct &x) {
  // 2.8.1 TODO: Simd Construct with Ordered Construct Nesting check
}

void OmpStructureChecker::Enter(const parser::OpenMPLoopConstruct &x) {
  const auto &dir{std::get<parser::OmpLoopDirective>(x.t)};
  switch (dir.v) {
  // 2.7.1 do-clause -> private-clause |
  //                    firstprivate-clause |
  //                    lastprivate-clause |
  //                    linear-clause |
  //                    reduction-clause |
  //                    schedule-clause |
  //                    collapse-clause |
  //                    ordered-clause
  case parser::OmpLoopDirective::Directive::Do: {
    // nesting check
    HasInvalidWorksharingNesting(dir.source,
        {OmpDirective::DO, OmpDirective::SECTIONS, OmpDirective::SINGLE,
            OmpDirective::WORKSHARE, OmpDirective::TASK, OmpDirective::TASKLOOP,
            OmpDirective::CRITICAL, OmpDirective::ORDERED, OmpDirective::ATOMIC,
            OmpDirective::MASTER});

    PushContext(dir.source, OmpDirective::DO);
    OmpClauseSet allowed{OmpClause::PRIVATE, OmpClause::FIRSTPRIVATE,
        OmpClause::LASTPRIVATE, OmpClause::LINEAR, OmpClause::REDUCTION};
    SetContextAllowed(allowed);
    OmpClauseSet allowedOnce{
        OmpClause::SCHEDULE, OmpClause::COLLAPSE, OmpClause::ORDERED};
    SetContextAllowedOnce(allowedOnce);
  } break;

  // 2.11.1 parallel-do-clause -> parallel-clause |
  //                              do-clause
  case parser::OmpLoopDirective::Directive::ParallelDo: {
    PushContext(dir.source, OmpDirective::PARALLEL_DO);
    OmpClauseSet allowed{OmpClause::DEFAULT, OmpClause::PRIVATE,
        OmpClause::FIRSTPRIVATE, OmpClause::SHARED, OmpClause::COPYIN,
        OmpClause::REDUCTION, OmpClause::LASTPRIVATE, OmpClause::LINEAR};
    SetContextAllowed(allowed);
    OmpClauseSet allowedOnce{OmpClause::IF, OmpClause::NUM_THREADS,
        OmpClause::PROC_BIND, OmpClause::SCHEDULE, OmpClause::COLLAPSE,
        OmpClause::ORDERED};
    SetContextAllowedOnce(allowedOnce);
  } break;

  // 2.8.1 simd-clause -> safelen-clause |
  //                      simdlen-clause |
  //                      linear-clause |
  //                      aligned-clause |
  //                      private-clause |
  //                      lastprivate-clause |
  //                      reduction-clause |
  //                      collapse-clause
  case parser::OmpLoopDirective::Directive::Simd: {
    PushContext(dir.source, OmpDirective::SIMD);
    OmpClauseSet allowed{OmpClause::LINEAR, OmpClause::ALIGNED,
        OmpClause::PRIVATE, OmpClause::LASTPRIVATE, OmpClause::REDUCTION};
    SetContextAllowed(allowed);
    OmpClauseSet allowedOnce{
        OmpClause::COLLAPSE, OmpClause::SAFELEN, OmpClause::SIMDLEN};
    SetContextAllowedOnce(allowedOnce);
  } break;

  // 2.8.3 do-simd-clause -> do-clause |
  //                         simd-clause
  case parser::OmpLoopDirective::Directive::DoSimd: {
    PushContext(dir.source, OmpDirective::DO_SIMD);
    OmpClauseSet allowed{OmpClause::PRIVATE, OmpClause::FIRSTPRIVATE,
        OmpClause::LASTPRIVATE, OmpClause::LINEAR, OmpClause::REDUCTION,
        OmpClause::ALIGNED};
    SetContextAllowed(allowed);
    OmpClauseSet allowedOnce{OmpClause::SCHEDULE, OmpClause::COLLAPSE,
        OmpClause::ORDERED, OmpClause::SAFELEN, OmpClause::SIMDLEN};
    SetContextAllowedOnce(allowedOnce);
  } break;

  // 2.11.4 parallel-do-simd-clause -> parallel-clause |
  //                                   do-simd-clause
  case parser::OmpLoopDirective::Directive::ParallelDoSimd: {
    PushContext(dir.source, OmpDirective::PARALLEL_DO_SIMD);
    OmpClauseSet allowed{OmpClause::DEFAULT, OmpClause::PRIVATE,
        OmpClause::FIRSTPRIVATE, OmpClause::SHARED, OmpClause::COPYIN,
        OmpClause::REDUCTION, OmpClause::LASTPRIVATE, OmpClause::LINEAR,
        OmpClause::ALIGNED};
    SetContextAllowed(allowed);
    OmpClauseSet allowedOnce{OmpClause::IF, OmpClause::NUM_THREADS,
        OmpClause::PROC_BIND, OmpClause::SCHEDULE, OmpClause::COLLAPSE,
        OmpClause::ORDERED, OmpClause::SAFELEN, OmpClause::SIMDLEN};
    SetContextAllowedOnce(allowedOnce);
  } break;

  default:
    // TODO others
    break;
  }
}

void OmpStructureChecker::Leave(const parser::OpenMPLoopConstruct &) {
  ompContext_.pop_back();
}

void OmpStructureChecker::Enter(const parser::OpenMPBlockConstruct &x) {
  const auto &beginBlockDir{std::get<parser::OmpBeginBlockDirective>(x.t)};
  const auto &beginDir{std::get<parser::OmpBlockDirective>(beginBlockDir.t)};
  switch (beginDir.v) {
  // 2.5 parallel-clause -> if-clause |
  //                        num-threads-clause |
  //                        default-clause |
  //                        private-clause |
  //                        firstprivate-clause |
  //                        shared-clause |
  //                        copyin-clause |
  //                        reduction-clause |
  //                        proc-bind-clause
  case parser::OmpBlockDirective::Directive::Parallel: {
    // reserve for nesting check
    PushContext(beginDir.source, OmpDirective::PARALLEL);
    OmpClauseSet allowed{OmpClause::DEFAULT, OmpClause::PRIVATE,
        OmpClause::FIRSTPRIVATE, OmpClause::SHARED, OmpClause::COPYIN,
        OmpClause::REDUCTION};
    SetContextAllowed(allowed);
    OmpClauseSet allowedOnce{
        OmpClause::IF, OmpClause::NUM_THREADS, OmpClause::PROC_BIND};
    SetContextAllowedOnce(allowedOnce);
  } break;
  // 2.7.3 single-clause -> private-clause |
  //                        firstprivate-clause
  case parser::OmpBlockDirective::Directive::Single:
    PushContext(beginDir.source, OmpDirective::SINGLE);
    SetContextAllowed({OmpClause::PRIVATE, OmpClause::FIRSTPRIVATE});
    break;
  // 2.7.4 workshare (no clauses are allowed)
  case parser::OmpBlockDirective::Directive::Workshare:
    PushContext(beginDir.source, OmpDirective::WORKSHARE);
    break;
  default:
    // TODO others
    break;
  }
}

void OmpStructureChecker::Leave(const parser::OpenMPBlockConstruct &) {
  ompContext_.pop_back();
}

void OmpStructureChecker::Enter(const parser::OpenMPSectionsConstruct &x) {
  const auto &dir{std::get<parser::Verbatim>(x.t)};
  PushContext(dir.source, OmpDirective::SECTIONS);
  OmpClauseSet allowed{OmpClause::PRIVATE, OmpClause::FIRSTPRIVATE,
      OmpClause::LASTPRIVATE, OmpClause::REDUCTION};
  SetContextAllowed(allowed);
}

void OmpStructureChecker::Leave(const parser::OpenMPSectionsConstruct &) {
  ompContext_.pop_back();
}

void OmpStructureChecker::Enter(const parser::OmpSection &x) {
  const auto &dir{x.v};  // Verbatim
  if (!CurrentDirectiveIsNested() ||
      GetContext().directive != OmpDirective::SECTIONS) {
    // if not currently nested, SECTION is orphaned
    context_.Say(dir.source,
        "SECTION directive must appear within "
        "the SECTIONS construct"_err_en_US);
  }
}

void OmpStructureChecker::Enter(const parser::OpenMPDeclareSimdConstruct &x) {
  const auto &dir{std::get<parser::Verbatim>(x.t)};
  PushContext(dir.source, OmpDirective::DECLARE_SIMD);
}

void OmpStructureChecker::Leave(const parser::OpenMPDeclareSimdConstruct &) {
  ompContext_.pop_back();
}

void OmpStructureChecker::Enter(
    const parser::OpenMPSimpleStandaloneConstruct &x) {
  const auto &dir{std::get<parser::OmpSimpleStandaloneDirective>(x.t)};
  switch (dir.v) {
  case parser::OmpSimpleStandaloneDirective::Directive::Barrier: {
    // 2.13.3 barrier
    PushContext(dir.source, OmpDirective::BARRIER);
  } break;
  case parser::OmpSimpleStandaloneDirective::Directive::Taskwait: {
    // 2.13.4 taskwait
    PushContext(dir.source, OmpDirective::TASKWAIT);
  } break;
  case parser::OmpSimpleStandaloneDirective::Directive::Taskyield: {
    // 2.9.4 taskyield
    PushContext(dir.source, OmpDirective::TASKYIELD);
  } break;
  case parser::OmpSimpleStandaloneDirective::Directive::TargetEnterData: {
    // 2.10.2 target-enter-data
    PushContext(dir.source, OmpDirective::TARGET_ENTER_DATA);
  } break;
  case parser::OmpSimpleStandaloneDirective::Directive::TargetExitData: {
    // 2.10.3 target-exit-data
    PushContext(dir.source, OmpDirective::TARGET_EXIT_DATA);
  } break;
  case parser::OmpSimpleStandaloneDirective::Directive::TargetUpdate: {
    // 2.10.5 target-update
    PushContext(dir.source, OmpDirective::TARGET_UPDATE);
  } break;
  case parser::OmpSimpleStandaloneDirective::Directive::Ordered: {
    // 2.13.8 ordered-construct-clause -> depend-clause
    PushContext(dir.source, OmpDirective::ORDERED);
    OmpClauseSet allowed{OmpClause::DEPEND};
    SetContextAllowed(allowed);
  } break;
  default:
    // TODO others
    break;
  }
}

void OmpStructureChecker::Leave(
    const parser::OpenMPSimpleStandaloneConstruct &) {
  ompContext_.pop_back();
}

void OmpStructureChecker::Enter(const parser::OpenMPFlushConstruct &x) {
  const auto &dir{std::get<parser::Verbatim>(x.t)};
  PushContext(dir.source, OmpDirective::FLUSH);
}

void OmpStructureChecker::Leave(const parser::OpenMPFlushConstruct &) {
  ompContext_.pop_back();
}

void OmpStructureChecker::Enter(const parser::OpenMPCancelConstruct &x) {
  const auto &dir{std::get<parser::Verbatim>(x.t)};
  PushContext(dir.source, OmpDirective::CANCEL);
}

void OmpStructureChecker::Leave(const parser::OpenMPCancelConstruct &) {
  ompContext_.pop_back();
}

void OmpStructureChecker::Enter(
    const parser::OpenMPCancellationPointConstruct &x) {
  const auto &dir{std::get<parser::Verbatim>(x.t)};
  PushContext(dir.source, OmpDirective::CANCELLATION_POINT);
}

void OmpStructureChecker::Leave(
    const parser::OpenMPCancellationPointConstruct &) {
  ompContext_.pop_back();
}

void OmpStructureChecker::Enter(const parser::OmpEndBlockDirective &x) {
  const auto &dir{std::get<parser::OmpBlockDirective>(x.t)};
  ResetPartialContext(dir.source);
  switch (dir.v) {
  // 2.7.3 end-single-clause -> copyprivate-clause |
  //                            nowait-clause
  case parser::OmpBlockDirective::Directive::Single: {
    SetContextDirectiveEnum(OmpDirective::END_SINGLE);
    OmpClauseSet allowed{OmpClause::COPYPRIVATE};
    SetContextAllowed(allowed);
    OmpClauseSet allowedOnce{OmpClause::NOWAIT};
    SetContextAllowedOnce(allowedOnce);
  } break;
  // 2.7.4 end-workshare -> END WORKSHARE [nowait-clause]
  case parser::OmpBlockDirective::Directive::Workshare:
    SetContextDirectiveEnum(OmpDirective::END_WORKSHARE);
    SetContextAllowed(OmpClauseSet{OmpClause::NOWAIT});
    break;
  default:
    // no clauses are allowed
    break;
  }
}

void OmpStructureChecker::Leave(const parser::OmpClauseList &) {
  // 2.7 Loop Construct Restriction
  if (doSet.test(GetContext().directive)) {
    if (auto *clause{FindClause(OmpClause::SCHEDULE)}) {
      // only one schedule clause is allowed
      const auto &schedClause{std::get<parser::OmpScheduleClause>(clause->u)};
      if (ScheduleModifierHasType(schedClause,
              parser::OmpScheduleModifierType::ModType::Nonmonotonic)) {
        if (FindClause(OmpClause::ORDERED)) {
          context_.Say(clause->source,
              "The NONMONOTONIC modifier cannot be specified "
              "if an ORDERED clause is specified"_err_en_US);
        }
        if (ScheduleModifierHasType(schedClause,
                parser::OmpScheduleModifierType::ModType::Monotonic)) {
          context_.Say(clause->source,
              "The MONOTONIC and NONMONOTONIC modifiers "
              "cannot be both specified"_err_en_US);
        }
      }
    }

    if (auto *clause{FindClause(OmpClause::ORDERED)}) {
      // only one ordered clause is allowed
      const auto &orderedClause{
          std::get<parser::OmpClause::Ordered>(clause->u)};

      if (orderedClause.v.has_value()) {
        if (FindClause(OmpClause::LINEAR)) {
          context_.Say(clause->source,
              "A loop directive may not have both a LINEAR clause and "
              "an ORDERED clause with a parameter"_err_en_US);
        }

        if (auto *clause2{FindClause(OmpClause::COLLAPSE)}) {
          const auto &collapseClause{
              std::get<parser::OmpClause::Collapse>(clause2->u)};
          // ordered and collapse both have parameters
          if (const auto orderedValue{GetIntValue(orderedClause.v)}) {
            if (const auto collapseValue{GetIntValue(collapseClause.v)}) {
              if (*orderedValue > 0 && *orderedValue < *collapseValue) {
                context_.Say(clause->source,
                    "The parameter of the ORDERED clause must be "
                    "greater than or equal to "
                    "the parameter of the COLLAPSE clause"_err_en_US);
              }
            }
          }
        }
      }

      // TODO: ordered region binding check (requires nesting implementation)
    }
  }  // doSet

  // 2.8.1 Simd Construct Restriction
  if (simdSet.test(GetContext().directive)) {
    if (auto *clause{FindClause(OmpClause::SIMDLEN)}) {
      if (auto *clause2{FindClause(OmpClause::SAFELEN)}) {
        const auto &simdlenClause{
            std::get<parser::OmpClause::Simdlen>(clause->u)};
        const auto &safelenClause{
            std::get<parser::OmpClause::Safelen>(clause2->u)};
        // simdlen and safelen both have parameters
        if (const auto simdlenValue{GetIntValue(simdlenClause.v)}) {
          if (const auto safelenValue{GetIntValue(safelenClause.v)}) {
            if (*safelenValue > 0 && *simdlenValue > *safelenValue) {
              context_.Say(clause->source,
                  "The parameter of the SIMDLEN clause must be less than or "
                  "equal to the parameter of the SAFELEN clause"_err_en_US);
            }
          }
        }
      }
    }

    // TODO: A list-item cannot appear in more than one aligned clause
  }  // SIMD

  // 2.7.3 Single Construct Restriction
  if (GetContext().directive == OmpDirective::END_SINGLE) {
    if (auto *clause{FindClause(OmpClause::COPYPRIVATE)}) {
      if (FindClause(OmpClause::NOWAIT)) {
        context_.Say(clause->source,
            "The COPYPRIVATE clause must not be used with "
            "the NOWAIT clause"_err_en_US);
      }
    }
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause &x) {
  SetContextClause(x);
}

void OmpStructureChecker::Enter(const parser::OmpNowait &) {
  switch (GetContext().directive) {
  case OmpDirective::SECTIONS:
  case OmpDirective::WORKSHARE:
  case OmpDirective::DO_SIMD:
  case OmpDirective::DO: break;  // NOWAIT is handled by parser
  default: CheckAllowed(OmpClause::NOWAIT); break;
  }
}
void OmpStructureChecker::Enter(const parser::OmpClause::Defaultmap &) {
  CheckAllowed(OmpClause::DEFAULTMAP);
}
void OmpStructureChecker::Enter(const parser::OmpClause::Inbranch &) {
  CheckAllowed(OmpClause::INBRANCH);
}
void OmpStructureChecker::Enter(const parser::OmpClause::Mergeable &) {
  CheckAllowed(OmpClause::MERGEABLE);
}
void OmpStructureChecker::Enter(const parser::OmpClause::Nogroup &) {
  CheckAllowed(OmpClause::NOGROUP);
}
void OmpStructureChecker::Enter(const parser::OmpClause::Notinbranch &) {
  CheckAllowed(OmpClause::NOTINBRANCH);
}
void OmpStructureChecker::Enter(const parser::OmpClause::Untied &) {
  CheckAllowed(OmpClause::UNTIED);
}

void OmpStructureChecker::Enter(const parser::OmpClause::Collapse &x) {
  CheckAllowed(OmpClause::COLLAPSE);
  // collapse clause must have a parameter
  if (const auto v{GetIntValue(x.v)}) {
    if (*v <= 0) {
      context_.Say(GetContext().clauseSource,
          "The parameter of the COLLAPSE clause must be "
          "a constant positive integer expression"_err_en_US);
    }
  }
}

void OmpStructureChecker::Enter(const parser::OmpClause::Copyin &) {
  CheckAllowed(OmpClause::COPYIN);
}
void OmpStructureChecker::Enter(const parser::OmpClause::Copyprivate &) {
  CheckAllowed(OmpClause::COPYPRIVATE);
}
void OmpStructureChecker::Enter(const parser::OmpClause::Device &) {
  CheckAllowed(OmpClause::DEVICE);
}
void OmpStructureChecker::Enter(const parser::OmpClause::DistSchedule &) {
  CheckAllowed(OmpClause::DIST_SCHEDULE);
}
void OmpStructureChecker::Enter(const parser::OmpClause::Final &) {
  CheckAllowed(OmpClause::FINAL);
}
void OmpStructureChecker::Enter(const parser::OmpClause::Firstprivate &) {
  CheckAllowed(OmpClause::FIRSTPRIVATE);
}
void OmpStructureChecker::Enter(const parser::OmpClause::From &) {
  CheckAllowed(OmpClause::FROM);
}
void OmpStructureChecker::Enter(const parser::OmpClause::Grainsize &) {
  CheckAllowed(OmpClause::GRAINSIZE);
}
void OmpStructureChecker::Enter(const parser::OmpClause::Lastprivate &) {
  CheckAllowed(OmpClause::LASTPRIVATE);
}
void OmpStructureChecker::Enter(const parser::OmpClause::NumTasks &) {
  CheckAllowed(OmpClause::NUM_TASKS);
}
void OmpStructureChecker::Enter(const parser::OmpClause::NumTeams &) {
  CheckAllowed(OmpClause::NUM_TEAMS);
}
void OmpStructureChecker::Enter(const parser::OmpClause::NumThreads &x) {
  CheckAllowed(OmpClause::NUM_THREADS);
  if (const auto v{GetIntValue(x.v)}) {
    if (*v <= 0) {
      context_.Say(GetContext().clauseSource,
          "The parameter of the NUM_THREADS clause must be "
          "a positive integer expression"_err_en_US);
    }
  }
  // if parameter is variable, defer to Expression Analysis
}

void OmpStructureChecker::Enter(const parser::OmpClause::Ordered &x) {
  CheckAllowed(OmpClause::ORDERED);
  // the parameter of ordered clause is optional
  if (const auto &expr{x.v}) {
    if (const auto v{GetIntValue(expr)}) {
      if (*v <= 0) {
        context_.Say(GetContext().clauseSource,
            "The parameter of the ORDERED clause must be "
            "a constant positive integer expression"_err_en_US);
      }
    }

    // 2.8.3 Loop SIMD Construct Restriction
    if (doSimdSet.test(GetContext().directive)) {
      context_.Say(GetContext().clauseSource,
          "No ORDERED clause with a parameter can be specified "
          "on the %s directive"_err_en_US,
          ContextDirectiveAsFortran());
    }
  }
}
void OmpStructureChecker::Enter(const parser::OmpClause::Priority &) {
  CheckAllowed(OmpClause::PRIORITY);
}
void OmpStructureChecker::Enter(const parser::OmpClause::Private &) {
  CheckAllowed(OmpClause::PRIVATE);
}
void OmpStructureChecker::Enter(const parser::OmpClause::Safelen &x) {
  CheckAllowed(OmpClause::SAFELEN);

  if (const auto v{GetIntValue(x.v)}) {
    if (*v <= 0) {
      context_.Say(GetContext().clauseSource,
          "The parameter of the SAFELEN clause must be "
          "a constant positive integer expression"_err_en_US);
    }
  }
}
void OmpStructureChecker::Enter(const parser::OmpClause::Shared &) {
  CheckAllowed(OmpClause::SHARED);
}
void OmpStructureChecker::Enter(const parser::OmpClause::Simdlen &x) {
  CheckAllowed(OmpClause::SIMDLEN);

  if (const auto v{GetIntValue(x.v)}) {
    if (*v <= 0) {
      context_.Say(GetContext().clauseSource,
          "The parameter of the SIMDLEN clause must be "
          "a constant positive integer expression"_err_en_US);
    }
  }
}
void OmpStructureChecker::Enter(const parser::OmpClause::ThreadLimit &) {
  CheckAllowed(OmpClause::THREAD_LIMIT);
}
void OmpStructureChecker::Enter(const parser::OmpClause::To &) {
  CheckAllowed(OmpClause::TO);
}
void OmpStructureChecker::Enter(const parser::OmpClause::Link &) {
  CheckAllowed(OmpClause::LINK);
}
void OmpStructureChecker::Enter(const parser::OmpClause::Uniform &) {
  CheckAllowed(OmpClause::UNIFORM);
}
void OmpStructureChecker::Enter(const parser::OmpClause::UseDevicePtr &) {
  CheckAllowed(OmpClause::USE_DEVICE_PTR);
}
void OmpStructureChecker::Enter(const parser::OmpClause::IsDevicePtr &) {
  CheckAllowed(OmpClause::IS_DEVICE_PTR);
}

void OmpStructureChecker::Enter(const parser::OmpAlignedClause &x) {
  CheckAllowed(OmpClause::ALIGNED);

  if (const auto &expr{
          std::get<std::optional<parser::ScalarIntConstantExpr>>(x.t)}) {
    if (const auto v{GetIntValue(*expr)}) {
      if (*v <= 0) {
        context_.Say(GetContext().clauseSource,
            "The ALIGNMENT parameter of the ALIGNED clause must be "
            "a constant positive integer expression"_err_en_US);
      }
    }
  }
  // 2.8.1 TODO: list-item attribute check
}
void OmpStructureChecker::Enter(const parser::OmpDefaultClause &) {
  CheckAllowed(OmpClause::DEFAULT);
}
void OmpStructureChecker::Enter(const parser::OmpDependClause &) {
  CheckAllowed(OmpClause::DEPEND);
}
void OmpStructureChecker::Enter(const parser::OmpIfClause &) {
  CheckAllowed(OmpClause::IF);
}
void OmpStructureChecker::Enter(const parser::OmpLinearClause &x) {
  CheckAllowed(OmpClause::LINEAR);

  // 2.7 Loop Construct Restriction
  if ((doSet | simdSet).test(GetContext().directive)) {
    if (std::holds_alternative<parser::OmpLinearClause::WithModifier>(x.u)) {
      context_.Say(GetContext().clauseSource,
          "A modifier may not be specified in a LINEAR clause "
          "on the %s directive"_err_en_US,
          ContextDirectiveAsFortran());
    }
  }
}
void OmpStructureChecker::Enter(const parser::OmpMapClause &) {
  CheckAllowed(OmpClause::MAP);
}
void OmpStructureChecker::Enter(const parser::OmpProcBindClause &) {
  CheckAllowed(OmpClause::PROC_BIND);
}
void OmpStructureChecker::Enter(const parser::OmpReductionClause &) {
  CheckAllowed(OmpClause::REDUCTION);
}

bool OmpStructureChecker::ScheduleModifierHasType(
    const parser::OmpScheduleClause &x,
    const parser::OmpScheduleModifierType::ModType &type) {
  const auto &modifier{
      std::get<std::optional<parser::OmpScheduleModifier>>(x.t)};
  if (modifier.has_value()) {
    const auto &modType1{
        std::get<parser::OmpScheduleModifier::Modifier1>(modifier->t)};
    const auto &modType2{
        std::get<std::optional<parser::OmpScheduleModifier::Modifier2>>(
            modifier->t)};
    if (modType1.v.v == type ||
        (modType2.has_value() && modType2->v.v == type)) {
      return true;
    }
  }
  return false;
}
void OmpStructureChecker::Enter(const parser::OmpScheduleClause &x) {
  CheckAllowed(OmpClause::SCHEDULE);

  // 2.7 Loop Construct Restriction
  if (doSet.test(GetContext().directive)) {
    const auto &kind{std::get<1>(x.t)};
    const auto &chunk{std::get<2>(x.t)};
    if (chunk.has_value()) {
      if (kind == parser::OmpScheduleClause::ScheduleType::Runtime ||
          kind == parser::OmpScheduleClause::ScheduleType::Auto) {
        context_.Say(GetContext().clauseSource,
            "When SCHEDULE clause has %s specified, "
            "it must not have chunk size specified"_err_en_US,
            parser::ToUpperCaseLetters(
                parser::OmpScheduleClause::EnumToString(kind)));
      }
    }

    if (ScheduleModifierHasType(
            x, parser::OmpScheduleModifierType::ModType::Nonmonotonic)) {
      if (kind != parser::OmpScheduleClause::ScheduleType::Dynamic &&
          kind != parser::OmpScheduleClause::ScheduleType::Guided) {
        context_.Say(GetContext().clauseSource,
            "The NONMONOTONIC modifier can only be specified with "
            "SCHEDULE(DYNAMIC) or SCHEDULE(GUIDED)"_err_en_US);
      }
    }
  }
}
}
