// Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
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

#ifndef FORTRAN_PARSER_OPENMP_GRAMMAR_H_
#define FORTRAN_PARSER_OPENMP_GRAMMAR_H_

// Top-level grammar specification for OpenMP.
// See OpenMP-4.5-grammar.txt for documentation.

#include "basic-parsers.h"
#include "characters.h"
#include "debug-parser.h"
#include "grammar.h"
#include "parse-tree.h"
#include "stmt-parser.h"
#include "token-parsers.h"
#include "type-parsers.h"
#include "user-state.h"
#include <cinttypes>
#include <cstdio>
#include <functional>
#include <list>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

// OpenMP Directives and Clauses
namespace Fortran::parser {

constexpr auto startOmpLine = skipStuffBeforeStatement >> "!$OMP "_sptok;
constexpr auto endOmpLine = space >> endOfLine;

template<typename A> constexpr decltype(auto) verbatim(A x) {
  return sourced(construct<Verbatim>(x));
}

// OpenMP Clauses
// DEFAULT (PRIVATE | FIRSTPRIVATE | SHARED | NONE )
TYPE_PARSER(construct<OmpDefaultClause>(
    "PRIVATE" >> pure(OmpDefaultClause::Type::Private) ||
    "FIRSTPRIVATE" >> pure(OmpDefaultClause::Type::Firstprivate) ||
    "SHARED" >> pure(OmpDefaultClause::Type::Shared) ||
    "NONE" >> pure(OmpDefaultClause::Type::None)))

// PROC_BIND(CLOSE | MASTER | SPREAD)
TYPE_PARSER(construct<OmpProcBindClause>(
    "CLOSE" >> pure(OmpProcBindClause::Type::Close) ||
    "MASTER" >> pure(OmpProcBindClause::Type::Master) ||
    "SPREAD" >> pure(OmpProcBindClause::Type::Spread)))

// MAP ([ [map-type-modifier[,]] map-type : ] list)
// map-type-modifier -> ALWAYS
// map-type -> TO | FROM | TOFROM | ALLOC | RELEASE | DELETE
TYPE_PARSER(construct<OmpMapType>(
    maybe("ALWAYS" >> construct<OmpMapType::Always>() / maybe(","_tok)),
    ("TO"_id >> pure(OmpMapType::Type::To) ||
        "FROM" >> pure(OmpMapType::Type::From) ||
        "TOFROM" >> pure(OmpMapType::Type::Tofrom) ||
        "ALLOC" >> pure(OmpMapType::Type::Alloc) ||
        "RELEASE" >> pure(OmpMapType::Type::Release) ||
        "DELETE" >> pure(OmpMapType::Type::Delete)) /
        ":"))

TYPE_PARSER(construct<OmpMapClause>(
    maybe(Parser<OmpMapType>{}), Parser<OmpObjectList>{}))

// SCHEDULE ([modifier [, modifier]:]kind[, chunk_size])
// Modifier ->  MONITONIC | NONMONOTONIC | SIMD
// kind -> STATIC | DYNAMIC | GUIDED | AUTO | RUNTIME
// chunk_size -> ScalarIntExpr
TYPE_PARSER(construct<OmpScheduleModifierType>(
    "MONOTONIC" >> pure(OmpScheduleModifierType::ModType::Monotonic) ||
    "NONMONOTONIC" >> pure(OmpScheduleModifierType::ModType::Nonmonotonic) ||
    "SIMD" >> pure(OmpScheduleModifierType::ModType::Simd)))

TYPE_PARSER(construct<OmpScheduleModifier>(Parser<OmpScheduleModifierType>{},
    maybe("," >> Parser<OmpScheduleModifierType>{}) / ":"))

TYPE_PARSER(construct<OmpScheduleClause>(maybe(Parser<OmpScheduleModifier>{}),
    "STATIC" >> pure(OmpScheduleClause::ScheduleType::Static) ||
        "DYNAMIC" >> pure(OmpScheduleClause::ScheduleType::Dynamic) ||
        "GUIDED" >> pure(OmpScheduleClause::ScheduleType::Guided) ||
        "AUTO" >> pure(OmpScheduleClause::ScheduleType::Auto) ||
        "RUNTIME" >> pure(OmpScheduleClause::ScheduleType::Runtime),
    maybe("," >> scalarIntExpr)))

// IF(directive-name-modifier: scalar-logical-expr)
TYPE_PARSER(construct<OmpIfClause>(
    maybe(("PARALLEL" >> pure(OmpIfClause::DirectiveNameModifier::Parallel) ||
              "TARGET ENTER DATA" >>
                  pure(OmpIfClause::DirectiveNameModifier::TargetEnterData) ||
              "TARGET EXIT DATA" >>
                  pure(OmpIfClause::DirectiveNameModifier::TargetExitData) ||
              "TARGET DATA" >>
                  pure(OmpIfClause::DirectiveNameModifier::TargetData) ||
              "TARGET UPDATE" >>
                  pure(OmpIfClause::DirectiveNameModifier::TargetUpdate) ||
              "TARGET" >> pure(OmpIfClause::DirectiveNameModifier::Target) ||
              "TASK"_id >> pure(OmpIfClause::DirectiveNameModifier::Taskloop) ||
              "TASKLOOP" >> pure(OmpIfClause::DirectiveNameModifier::Task)) /
        ":"),
    scalarLogicalExpr))

TYPE_PARSER(construct<OmpReductionOperator>(Parser<DefinedOperator>{}) ||
    construct<OmpReductionOperator>(Parser<ProcedureDesignator>{}))

TYPE_PARSER(construct<OmpReductionClause>(
    Parser<OmpReductionOperator>{} / ":", nonemptyList(designator)))

// DEPEND(SOURCE | SINK : vec | (IN | OUT | INOUT) : list
TYPE_PARSER(construct<OmpDependSinkVecLength>(
    Parser<DefinedOperator>{}, scalarIntConstantExpr))

TYPE_PARSER(
    construct<OmpDependSinkVec>(name, maybe(Parser<OmpDependSinkVecLength>{})))

TYPE_PARSER(
    construct<OmpDependenceType>("IN"_id >> pure(OmpDependenceType::Type::In) ||
        "INOUT" >> pure(OmpDependenceType::Type::Inout) ||
        "OUT" >> pure(OmpDependenceType::Type::Out)))

TYPE_CONTEXT_PARSER("Omp Depend clause"_en_US,
    construct<OmpDependClause>(construct<OmpDependClause::Sink>(
        "SINK :" >> nonemptyList(Parser<OmpDependSinkVec>{}))) ||
        construct<OmpDependClause>(
            construct<OmpDependClause::Source>("SOURCE"_tok)) ||
        construct<OmpDependClause>(construct<OmpDependClause::InOut>(
            Parser<OmpDependenceType>{}, ":" >> nonemptyList(designator))))

// linear-modifier
TYPE_PARSER(
    construct<OmpLinearModifier>("REF" >> pure(OmpLinearModifier::Type::Ref) ||
        "VAL" >> pure(OmpLinearModifier::Type::Val) ||
        "UVAL" >> pure(OmpLinearModifier::Type::Uval)))

// LINEAR(list: linear-step)
TYPE_CONTEXT_PARSER("Omp LINEAR clause"_en_US,
    construct<OmpLinearClause>(
        construct<OmpLinearClause>(construct<OmpLinearClause::WithModifier>(
            Parser<OmpLinearModifier>{}, parenthesized(nonemptyList(name)),
            maybe(":" >> scalarIntConstantExpr))) ||
        construct<OmpLinearClause>(construct<OmpLinearClause::WithoutModifier>(
            nonemptyList(name), maybe(":" >> scalarIntConstantExpr)))))

// ALIGNED(list: alignment)
TYPE_PARSER(construct<OmpAlignedClause>(
    nonemptyList(name), maybe(":"_tok) >> scalarIntConstantExpr))

TYPE_PARSER(construct<OmpObject>(pure(OmpObject::Kind::Object), designator) ||
    construct<OmpObject>(
        "/" >> pure(OmpObject::Kind::Common), designator / "/"))

TYPE_PARSER("ALIGNED" >>
        construct<OmpClause>(parenthesized(Parser<OmpAlignedClause>{})) ||
    "COLLAPSE" >> construct<OmpClause>(construct<OmpClause::Collapse>(
                      parenthesized(scalarIntConstantExpr))) ||
    "COPYIN" >> construct<OmpClause>(construct<OmpClause::Copyin>(
                    parenthesized(Parser<OmpObjectList>{}))) ||
    "COPYPRIVATE" >> construct<OmpClause>(construct<OmpClause::Copyprivate>(
                         (parenthesized(Parser<OmpObjectList>{})))) ||
    "DEFAULT"_id >>
        construct<OmpClause>(parenthesized(Parser<OmpDefaultClause>{})) ||
    "DEFAULTMAP" >> construct<OmpClause>(construct<OmpClause::Defaultmap>(
                        "( TOFROM : SCALAR )"_tok)) ||
    "DEPEND" >>
        construct<OmpClause>(parenthesized(Parser<OmpDependClause>{})) ||
    "DEVICE" >> construct<OmpClause>(construct<OmpClause::Device>(
                    parenthesized(scalarIntExpr))) ||
    "DIST_SCHEDULE" >> construct<OmpClause>(construct<OmpClause::DistSchedule>(
                           parenthesized("STATIC ," >> scalarIntExpr))) ||
    "FINAL" >> construct<OmpClause>(
                   construct<OmpClause::Final>(parenthesized(scalarIntExpr))) ||
    "FIRSTPRIVATE" >> construct<OmpClause>(construct<OmpClause::Firstprivate>(
                          parenthesized(Parser<OmpObjectList>{}))) ||
    "FROM" >> construct<OmpClause>(construct<OmpClause::From>(
                  parenthesized(nonemptyList(designator)))) ||
    "GRAINSIZE" >> construct<OmpClause>(construct<OmpClause::Grainsize>(
                       parenthesized(scalarIntExpr))) ||
    "IF" >> construct<OmpClause>(parenthesized(Parser<OmpIfClause>{})) ||
    "INBRANCH" >> construct<OmpClause>(construct<OmpClause::Inbranch>()) ||
    "IS_DEVICE_PTR" >> construct<OmpClause>(construct<OmpClause::IsDevicePtr>(
                           parenthesized(nonemptyList(name)))) ||
    "LASTPRIVATE" >> construct<OmpClause>(construct<OmpClause::Lastprivate>(
                         parenthesized(Parser<OmpObjectList>{}))) ||
    "LINEAR" >>
        construct<OmpClause>(parenthesized(Parser<OmpLinearClause>{})) ||
    "LINK" >> construct<OmpClause>(construct<OmpClause::Link>(
                  parenthesized(nonemptyList(designator)))) ||
    "MAP" >> construct<OmpClause>(parenthesized(Parser<OmpMapClause>{})) ||
    "MERGEABLE" >> construct<OmpClause>(construct<OmpClause::Mergeable>()) ||
    "NOGROUP" >> construct<OmpClause>(construct<OmpClause::Nogroup>()) ||
    "NOTINBRANCH" >>
        construct<OmpClause>(construct<OmpClause::Notinbranch>()) ||
    "NOWAIT" >> construct<OmpClause>(construct<OmpNowait>()) ||
    "NUM_TASKS" >> construct<OmpClause>(construct<OmpClause::NumTasks>(
                       parenthesized(scalarIntExpr))) ||
    "NUM_TEAMS" >> construct<OmpClause>(construct<OmpClause::NumTeams>(
                       parenthesized(scalarIntExpr))) ||
    "NUM_THREADS" >> construct<OmpClause>(construct<OmpClause::NumThreads>(
                         parenthesized(scalarIntExpr))) ||
    "ORDERED" >> construct<OmpClause>(construct<OmpClause::Ordered>(
                     maybe(parenthesized(scalarIntConstantExpr)))) ||
    "PRIORITY" >> construct<OmpClause>(construct<OmpClause::Priority>(
                      parenthesized(scalarIntExpr))) ||
    "PRIVATE" >> construct<OmpClause>(construct<OmpClause::Private>(
                     parenthesized(Parser<OmpObjectList>{}))) ||
    "PROC_BIND" >>
        construct<OmpClause>(parenthesized(Parser<OmpProcBindClause>{})) ||
    "REDUCTION" >>
        construct<OmpClause>(parenthesized(Parser<OmpReductionClause>{})) ||
    "SAFELEN" >> construct<OmpClause>(construct<OmpClause::Safelen>(
                     parenthesized(scalarIntConstantExpr))) ||
    "SCHEDULE" >>
        construct<OmpClause>(parenthesized(Parser<OmpScheduleClause>{})) ||
    "SHARED" >> construct<OmpClause>(construct<OmpClause::Shared>(
                    parenthesized(Parser<OmpObjectList>{}))) ||
    "SIMD"_id >> construct<OmpClause>(construct<OmpClause::Simd>()) ||
    "SIMDLEN" >> construct<OmpClause>(construct<OmpClause::Simdlen>(
                     parenthesized(scalarIntConstantExpr))) ||
    "THREADS" >> construct<OmpClause>(construct<OmpClause::Threads>()) ||
    "THREAD_LIMIT" >> construct<OmpClause>(construct<OmpClause::ThreadLimit>(
                          parenthesized(scalarIntExpr))) ||
    "TO" >> construct<OmpClause>(construct<OmpClause::To>(
                parenthesized(nonemptyList(designator)))) ||
    "USE_DEVICE_PTR" >> construct<OmpClause>(construct<OmpClause::UseDevicePtr>(
                            parenthesized(nonemptyList(name)))) ||
    "UNIFORM" >> construct<OmpClause>(construct<OmpClause::Uniform>(
                     parenthesized(nonemptyList(name)))) ||
    "UNTIED" >> construct<OmpClause>(construct<OmpClause::Untied>()))

// [Clause, [Clause], ...]
TYPE_PARSER(sourced(construct<OmpClauseList>(
    many(maybe(","_tok) >> sourced(Parser<OmpClause>{})))))

// (variable | /common-block | array-sections)
TYPE_PARSER(construct<OmpObjectList>(nonemptyList(Parser<OmpObject>{})))

// Omp directives enclosing do loop
TYPE_PARSER(sourced(construct<OmpLoopDirective>(first(
    "DISTRIBUTE PARALLEL DO SIMD" >>
        pure(OmpLoopDirective::Directive::DistributeParallelDoSimd),
    "DISTRIBUTE PARALLEL DO" >>
        pure(OmpLoopDirective::Directive::DistributeParallelDo),

    "DISTRIBUTE SIMD" >> pure(OmpLoopDirective::Directive::DistributeSimd),

    "DISTRIBUTE" >> pure(OmpLoopDirective::Directive::Distribute),

    "DO SIMD" >> pure(OmpLoopDirective::Directive::DoSimd),
    "DO" >> pure(OmpLoopDirective::Directive::Do),
    "PARALLEL DO SIMD" >> pure(OmpLoopDirective::Directive::ParallelDoSimd),

    "PARALLEL DO" >> pure(OmpLoopDirective::Directive::ParallelDo),

    "SIMD" >> pure(OmpLoopDirective::Directive::Simd),
    "TARGET PARALLEL DO SIMD" >>
        pure(OmpLoopDirective::Directive::TargetParallelDoSimd),
    "TARGET PARALLEL DO" >> pure(OmpLoopDirective::Directive::TargetParallelDo),

    "TARGET SIMD" >> pure(OmpLoopDirective::Directive::TargetSimd),
    "TARGET TEAMS DISTRIBUTE PARALLEL DO SIMD" >>
        pure(OmpLoopDirective::Directive::TargetTeamsDistributeParallelDoSimd),
    "TARGET TEAMS DISTRIBUTE PARALLEL DO" >>
        pure(OmpLoopDirective::Directive::TargetTeamsDistributeParallelDo),
    "TARGET TEAMS DISTRIBUTE SIMD" >>
        pure(OmpLoopDirective::Directive::TargetTeamsDistributeSimd),
    "TARGET TEAMS DISTRIBUTE" >>
        pure(OmpLoopDirective::Directive::TargetTeamsDistribute),

    "TASKLOOP SIMD" >> pure(OmpLoopDirective::Directive::TaskloopSimd),

    "TASKLOOP" >> pure(OmpLoopDirective::Directive::Taskloop),
    "TEAMS DISTRIBUTE PARALLEL DO SIMD" >>
        pure(OmpLoopDirective::Directive::TeamsDistributeParallelDoSimd),
    "TEAMS DISTRIBUTE PARALLEL DO" >>
        pure(OmpLoopDirective::Directive::TeamsDistributeParallelDo),
    "TEAMS DISTRIBUTE SIMD" >>
        pure(OmpLoopDirective::Directive::TeamsDistributeSimd),
    "TEAMS DISTRIBUTE" >> pure(OmpLoopDirective::Directive::TeamsDistribute)))))

TYPE_PARSER(sourced(construct<OmpCancelType>(
    first("PARALLEL" >> pure(OmpCancelType::Type::Parallel),
        "SECTIONS" >> pure(OmpCancelType::Type::Sections),
        "DO" >> pure(OmpCancelType::Type::Do),
        "TASKGROUP" >> pure(OmpCancelType::Type::Taskgroup)))))

// Cancellation Point construct
TYPE_PARSER(construct<OpenMPCancellationPointConstruct>(
    sourced("CANCELLATION POINT" >> Parser<OmpCancelType>{})))

// Cancel construct
TYPE_PARSER(construct<OpenMPCancelConstruct>(
    sourced("CANCEL" >> Parser<OmpCancelType>{}),
    maybe("IF" >> parenthesized(scalarLogicalExpr))))

// Flush construct
TYPE_PARSER(sourced(construct<OpenMPFlushConstruct>(
    "FLUSH" >> maybe(parenthesized(Parser<OmpObjectList>{})))))

// Standalone directives
TYPE_PARSER(sourced(construct<OmpStandaloneDirective>(first(
    "TARGET ENTER DATA" >>
        pure(OmpStandaloneDirective::Directive::TargetEnterData),
    "TARGET EXIT DATA" >>
        pure(OmpStandaloneDirective::Directive::TargetExitData),
    "TARGET UPDATE" >> pure(OmpStandaloneDirective::Directive::TargetUpdate)))))

// Directives enclosing structured-block
TYPE_PARSER(sourced(construct<OmpBlockDirective>(
    first("MASTER" >> pure(OmpBlockDirective::Directive::Master),
        "ORDERED" >> pure(OmpBlockDirective::Directive::Ordered),
        "PARALLEL WORKSHARE" >>
            pure(OmpBlockDirective::Directive::ParallelWorkshare),
        "PARALLEL" >> pure(OmpBlockDirective::Directive::Parallel),
        "TARGET DATA" >> pure(OmpBlockDirective::Directive::TargetData),
        "TARGET PARALLEL" >> pure(OmpBlockDirective::Directive::TargetParallel),
        "TARGET TEAMS" >> pure(OmpBlockDirective::Directive::TargetTeams),
        "TARGET" >> pure(OmpBlockDirective::Directive::Target),
        "TASK"_id >> pure(OmpBlockDirective::Directive::Task),
        "TASKGROUP" >> pure(OmpBlockDirective::Directive::Taskgroup),
        "TEAMS" >> pure(OmpBlockDirective::Directive::Teams)))))

TYPE_PARSER(construct<OmpReductionInitializerClause>(
    "INITIALIZER" >> parenthesized("OMP_PRIV =" >> expr)))

// Declare Reduction Construct
TYPE_PARSER(sourced(construct<OpenMPDeclareReductionConstruct>(
    verbatim("DECLARE REDUCTION"_tok),
    "(" >> Parser<OmpReductionOperator>{} / ":",
    nonemptyList(Parser<DeclarationTypeSpec>{}) / ":",
    Parser<OmpReductionCombiner>{} / ")",
    maybe(Parser<OmpReductionInitializerClause>{}))))

// declare-target-map-type
TYPE_PARSER(construct<OmpDeclareTargetMapType>(
    "LINK" >> pure(OmpDeclareTargetMapType::Type::Link) ||
    "TO" >> pure(OmpDeclareTargetMapType::Type::To)))

// declare-target-specifier
TYPE_PARSER(construct<OpenMPDeclareTargetSpecifier>(
                construct<OpenMPDeclareTargetSpecifier::WithClause>(
                    Parser<OmpDeclareTargetMapType>{},
                    parenthesized(Parser<OmpObjectList>{}))) ||
    lookAhead(endOfLine) >>
        construct<OpenMPDeclareTargetSpecifier>(
            construct<OpenMPDeclareTargetSpecifier::Implicit>()) ||
    construct<OpenMPDeclareTargetSpecifier>(
        parenthesized(construct<OpenMPDeclareTargetSpecifier::WithExtendedList>(
            Parser<OmpObjectList>{}))))

// Declare Target Construct
TYPE_PARSER(sourced(construct<OpenMPDeclareTargetConstruct>(
    verbatim("DECLARE TARGET"_tok), Parser<OpenMPDeclareTargetSpecifier>{})))

TYPE_PARSER(construct<OmpReductionCombiner>(Parser<AssignmentStmt>{}) ||
    construct<OmpReductionCombiner>(
        construct<OmpReductionCombiner::FunctionCombiner>(
            construct<Call>(Parser<ProcedureDesignator>{},
                parenthesized(optionalList(actualArgSpec))))))

// OMP END ATOMIC
TYPE_PARSER(construct<OmpEndAtomic>(startOmpLine >> "END ATOMIC"_tok))

// OMP [SEQ_CST] ATOMIC READ [SEQ_CST]
TYPE_PARSER(construct<OmpAtomicRead>(
    maybe("SEQ_CST" >> construct<OmpAtomicRead::SeqCst1>() / maybe(","_tok)),
    "READ" >> maybe(","_tok) >>
        maybe("SEQ_CST" >> construct<OmpAtomicRead::SeqCst2>()) / endOmpLine,
    statement(assignmentStmt), maybe(Parser<OmpEndAtomic>{} / endOmpLine)))

// OMP ATOMIC [SEQ_CST] CAPTURE [SEQ_CST]
TYPE_PARSER(construct<OmpAtomicCapture>(
    maybe("SEQ_CST" >> construct<OmpAtomicCapture::SeqCst1>() / maybe(","_tok)),
    "CAPTURE" >> maybe(","_tok) >>
        maybe("SEQ_CST" >> construct<OmpAtomicCapture::SeqCst2>()) / endOmpLine,
    statement(assignmentStmt), statement(assignmentStmt),
    Parser<OmpEndAtomic>{} / endOmpLine))

// OMP ATOMIC [SEQ_CST] UPDATE [SEQ_CST]
TYPE_PARSER(construct<OmpAtomicUpdate>(
    maybe("SEQ_CST" >> construct<OmpAtomicUpdate::SeqCst1>() / maybe(","_tok)),
    "UPDATE" >> maybe(","_tok) >>
        maybe("SEQ_CST" >> construct<OmpAtomicUpdate::SeqCst2>()) / endOmpLine,
    statement(assignmentStmt), maybe(Parser<OmpEndAtomic>{} / endOmpLine)))

// OMP ATOMIC [SEQ_CST]
TYPE_PARSER(construct<OmpAtomic>(
    maybe("SEQ_CST" >> construct<OmpAtomic::SeqCst>()) / endOmpLine,
    statement(assignmentStmt), maybe(Parser<OmpEndAtomic>{} / endOmpLine)))

// ATOMIC [SEQ_CST] WRITE [SEQ_CST]
TYPE_PARSER(construct<OmpAtomicWrite>(
    maybe("SEQ_CST" >> construct<OmpAtomicWrite::SeqCst1>() / maybe(","_tok)),
    "WRITE" >> maybe(","_tok) >>
        maybe("SEQ_CST" >> construct<OmpAtomicWrite::SeqCst2>()) / endOmpLine,
    statement(assignmentStmt), maybe(Parser<OmpEndAtomic>{} / endOmpLine)))

// Atomic Construct
TYPE_PARSER("ATOMIC" >>
    (construct<OpenMPAtomicConstruct>(Parser<OmpAtomicRead>{}) ||
        construct<OpenMPAtomicConstruct>(Parser<OmpAtomicCapture>{}) ||
        construct<OpenMPAtomicConstruct>(Parser<OmpAtomicWrite>{}) ||
        construct<OpenMPAtomicConstruct>(Parser<OmpAtomicUpdate>{}) ||
        construct<OpenMPAtomicConstruct>(Parser<OmpAtomic>{})))

// OMP CRITICAL
TYPE_PARSER(startOmpLine >> "END CRITICAL"_tok >>
    construct<OmpEndCritical>(maybe(parenthesized(name))) / endOmpLine)

TYPE_PARSER(
    sourced(construct<OpenMPCriticalConstructDirective>(
        "CRITICAL" >> maybe(parenthesized(name)),
        maybe("HINT" >> construct<OpenMPCriticalConstructDirective::Hint>(
                            parenthesized(constantExpr))))) /
    endOmpLine)
TYPE_PARSER(construct<OpenMPCriticalConstruct>(
    Parser<OpenMPCriticalConstructDirective>{}, block,
    Parser<OmpEndCritical>{}))

// Declare Simd construct
TYPE_PARSER(
    sourced(construct<OpenMPDeclareSimdConstruct>(verbatim("DECLARE SIMD"_tok),
        maybe(parenthesized(name)), Parser<OmpClauseList>{})))

// Threadprivate directive
TYPE_PARSER(sourced(construct<OpenMPThreadprivate>(
    verbatim("THREADPRIVATE"_tok), parenthesized(Parser<OmpObjectList>{}))))

// Declarative constructs
TYPE_PARSER(startOmpLine >>
    sourced(construct<OpenMPDeclarativeConstruct>(
                Parser<OpenMPDeclareReductionConstruct>{}) ||
        construct<OpenMPDeclarativeConstruct>(
            Parser<OpenMPDeclareSimdConstruct>{}) ||
        construct<OpenMPDeclarativeConstruct>(
            Parser<OpenMPDeclareTargetConstruct>{}) ||
        construct<OpenMPDeclarativeConstruct>(Parser<OpenMPThreadprivate>{})) /
        endOmpLine)

// Block Construct
TYPE_PARSER(construct<OpenMPBlockConstruct>(Parser<OmpBlockDirective>{},
    Parser<OmpClauseList>{} / endOmpLine, block,
    Parser<OmpEndBlockDirective>{}))

// Simple constructs without clauses
TYPE_PARSER(
    sourced(construct<OpenMPSimpleConstruct>(first(
        "BARRIER" >> pure(OpenMPSimpleConstruct::Directive::Barrier),
        "TASKWAIT" >> pure(OpenMPSimpleConstruct::Directive::Taskwait),
        "TASKYIELD" >> pure(OpenMPSimpleConstruct::Directive::Taskyield)))) /
    endOmpLine)

TYPE_PARSER(construct<OpenMPStandaloneConstruct>(
    Parser<OmpStandaloneDirective>{}, Parser<OmpClauseList>{} / endOmpLine))

// OMP SINGLE
TYPE_PARSER(startOmpLine >> construct<OmpEndSingle>(verbatim("END SINGLE"_tok),
                                Parser<OmpClauseList>{}) /
        endOmpLine)

TYPE_PARSER(construct<OpenMPSingleConstruct>(verbatim("SINGLE"_tok),
    Parser<OmpClauseList>{} / endOmpLine, block, Parser<OmpEndSingle>{}))

// OMP WORKSHARE
TYPE_PARSER(construct<OpenMPWorkshareConstruct>(
    verbatim("WORKSHARE"_tok) / endOmpLine, block,
    startOmpLine >> "END WORKSHARE"_tok >>
        maybe(construct<OmpNowait>("NOWAIT"_tok) / endOmpLine)))

// OMP END DO SIMD [NOWAIT]
TYPE_PARSER(construct<OmpEndDoSimd>(maybe(construct<OmpNowait>("NOWAIT"_tok))))

// OMP END DO [NOWAIT]
TYPE_PARSER(construct<OmpEndDo>(maybe(construct<OmpNowait>("NOWAIT"_tok))))

// OMP END SECTIONS [NOWAIT]
TYPE_PARSER(startOmpLine >> "END SECTIONS"_tok >>
    construct<OmpEndSections>(
        maybe("NOWAIT" >> construct<OmpNowait>()) / endOmpLine))

// OMP SECTIONS
TYPE_PARSER(construct<OpenMPSectionsConstruct>(verbatim("SECTIONS"_tok),
    Parser<OmpClauseList>{} / endOmpLine, block, Parser<OmpEndSections>{}))

// OMP END PARALLEL SECTIONS
TYPE_PARSER(construct<OmpEndParallelSections>(
    startOmpLine >> "END PARALLEL SECTIONS"_tok / endOmpLine))

// OMP PARALLEL SECTIONS
TYPE_PARSER(construct<OpenMPParallelSectionsConstruct>(
    verbatim("PARALLEL SECTIONS"_tok), Parser<OmpClauseList>{} / endOmpLine,
    block, Parser<OmpEndParallelSections>{}))

TYPE_PARSER(construct<OmpSection>(verbatim("SECTION"_tok) / endOmpLine))

TYPE_CONTEXT_PARSER("OpenMP construct"_en_US,
    startOmpLine >>
        first(construct<OpenMPConstruct>(Parser<OpenMPStandaloneConstruct>{}),
            construct<OpenMPConstruct>(Parser<OpenMPSimpleConstruct>{}),
            construct<OpenMPConstruct>(Parser<OpenMPSingleConstruct>{}),
            construct<OpenMPConstruct>(Parser<OpenMPSectionsConstruct>{}),
            construct<OpenMPConstruct>(
                Parser<OpenMPParallelSectionsConstruct>{}),
            construct<OpenMPConstruct>(Parser<OpenMPWorkshareConstruct>{}),
            construct<OpenMPConstruct>(Parser<OpenMPLoopConstruct>{}),
            construct<OpenMPConstruct>(Parser<OpenMPBlockConstruct>{}),
            construct<OpenMPConstruct>(Parser<OpenMPAtomicConstruct>{}),
            construct<OpenMPConstruct>(Parser<OpenMPCriticalConstruct>{}),
            construct<OpenMPConstruct>(Parser<OpenMPCancelConstruct>{}),
            construct<OpenMPConstruct>(
                Parser<OpenMPCancellationPointConstruct>{}),
            construct<OpenMPConstruct>(Parser<OpenMPFlushConstruct>{}),
            construct<OpenMPConstruct>(Parser<OmpSection>{})))

// END OMP Block directives
TYPE_PARSER(startOmpLine >> "END"_tok >>
    construct<OmpEndBlockDirective>(Parser<OmpBlockDirective>{}) / endOmpLine)

// END OMP Loop directives
TYPE_PARSER(startOmpLine >> "END"_tok >>
    (construct<OpenMPEndLoopDirective>("DO SIMD" >> Parser<OmpEndDoSimd>{}) ||
        construct<OpenMPEndLoopDirective>("DO" >> Parser<OmpEndDo>{}) ||
        construct<OpenMPEndLoopDirective>(Parser<OmpLoopDirective>{}) /
            endOmpLine))

TYPE_PARSER(construct<OpenMPLoopConstruct>(
    Parser<OmpLoopDirective>{}, Parser<OmpClauseList>{} / endOmpLine))
}
#endif  // FORTRAN_PARSER_OPENMP_GRAMMAR_H_
