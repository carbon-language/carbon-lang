// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

constexpr auto beginOmpDirective{skipStuffBeforeStatement >> "!$OMP "_sptok};

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
TYPE_PARSER(construct<OmpMapClause>(
    maybe(maybe("ALWAYS" >> maybe(","_tok)) >>
        ("TO" >> pure(OmpMapClause::Type::To) / ":"_tok ||
            "FROM" >> pure(OmpMapClause::Type::From) / ":"_tok ||
            "TOFROM" >> pure(OmpMapClause::Type::Tofrom) / ":"_tok ||
            "ALLOC" >> pure(OmpMapClause::Type::Alloc) / ":"_tok ||
            "RELEASE" >> pure(OmpMapClause::Type::Release) / ":"_tok ||
            "DELETE" >> pure(OmpMapClause::Type::Delete) / ":"_tok)),
    nonemptyList(name)))

// SCHEDULE ([modifier [, modifier]:]kind[, chunk_size])
// Modifier ->  MONITONIC | NONMONOTONIC | SIMD
// kind -> STATIC | DYNAMIC | GUIDED | AUTO | RUNTIME
// chunk_size -> ScalarIntExpr
TYPE_PARSER(construct<OmpScheduleModifierType>(
    "MONOTONIC" >> pure(OmpScheduleModifierType::ModType::Monotonic) ||
    "NONMONOTONIC" >> pure(OmpScheduleModifierType::ModType::Nonmonotonic) ||
    "SIMD" >> pure(OmpScheduleModifierType::ModType::Simd)))

TYPE_PARSER(construct<OmpScheduleModifier>(Parser<OmpScheduleModifierType>{},
    maybe(","_tok >> Parser<OmpScheduleModifierType>{})))

TYPE_PARSER(construct<OmpScheduleClause>(maybe(Parser<OmpScheduleModifier>{}),
    "STATIC" >> pure(OmpScheduleClause::ScheduleType::Static) ||
        "DYNAMIC" >> pure(OmpScheduleClause::ScheduleType::Dynamic) ||
        "GUIDED" >> pure(OmpScheduleClause::ScheduleType::Guided) ||
        "AUTO" >> pure(OmpScheduleClause::ScheduleType::Auto) ||
        "RUNTIME" >> pure(OmpScheduleClause::ScheduleType::Runtime),
    maybe(","_tok) >> scalarIntExpr))

// IF(directive-name-modifier: scalar-logical-expr)
TYPE_PARSER(construct<OmpIfClause>(
    maybe(
        "PARALLEL"_tok >> pure(OmpIfClause::DirectiveNameModifier::Parallel) ||
        "TARGET ENTER DATA"_tok >>
            pure(OmpIfClause::DirectiveNameModifier::TargetEnterData) ||
        "TARGET EXIT DATA"_tok >>
            pure(OmpIfClause::DirectiveNameModifier::TargetExitData) ||
        "TARGET DATA"_tok >>
            pure(OmpIfClause::DirectiveNameModifier::TargetData) ||
        "TARGET UPDATE"_tok >>
            pure(OmpIfClause::DirectiveNameModifier::TargetUpdate) ||
        "TARGET"_tok >> pure(OmpIfClause::DirectiveNameModifier::Target) ||
        "TASKLOOP"_tok >> pure(OmpIfClause::DirectiveNameModifier::Taskloop) ||
        "TASK"_tok >> pure(OmpIfClause::DirectiveNameModifier::Task)) /
        maybe(":"_tok),
    scalarLogicalExpr))

// REDUCTION(reduction-identifier: list)
constexpr auto reductionBinaryOperator{
    "+" >> pure(OmpReductionOperator::BinaryOperator::Add) ||
    "-" >> pure(OmpReductionOperator::BinaryOperator::Subtract) ||
    "*" >> pure(OmpReductionOperator::BinaryOperator::Multiply) ||
    ".AND." >> pure(OmpReductionOperator::BinaryOperator::AND) ||
    ".OR." >> pure(OmpReductionOperator::BinaryOperator::OR) ||
    ".EQV." >> pure(OmpReductionOperator::BinaryOperator::EQV) ||
    ".NEQV." >> pure(OmpReductionOperator::BinaryOperator::NEQV)};

constexpr auto reductionProcedureOperator{
    "MIN" >> pure(OmpReductionOperator::ProcedureOperator::MIN) ||
    "MAX" >> pure(OmpReductionOperator::ProcedureOperator::MAX) ||
    "IAND" >> pure(OmpReductionOperator::ProcedureOperator::IAND) ||
    "IOR" >> pure(OmpReductionOperator::ProcedureOperator::IOR) ||
    "IEOR" >> pure(OmpReductionOperator::ProcedureOperator::IEOR)};

TYPE_PARSER(construct<OmpReductionOperator>(reductionBinaryOperator) ||
    construct<OmpReductionOperator>(reductionProcedureOperator))

TYPE_PARSER(construct<OmpReductionClause>(
    Parser<OmpReductionOperator>{} / ":"_tok, nonemptyList(designator)))

// DEPEND(SOURCE | SINK : vec | (IN | OUT | INOUT) : list
TYPE_PARSER(construct<OmpDependSinkVecLength>(
    indirect(Parser<DefinedOperator>{}), scalarIntConstantExpr))

TYPE_PARSER(
    construct<OmpDependSinkVec>(name, maybe(Parser<OmpDependSinkVecLength>{})))

TYPE_PARSER(construct<OmpDependenceType>(
    "IN"_tok >> pure(OmpDependenceType::Type::In) ||
    "OUT"_tok >> pure(OmpDependenceType::Type::Out) ||
    "INOUT"_tok >> pure(OmpDependenceType::Type::Inout)))

TYPE_CONTEXT_PARSER("Omp Depend clause"_en_US,
    construct<OmpDependClause>(construct<OmpDependClause::Sink>(
        "SINK"_tok >> ":"_tok >> nonemptyList(Parser<OmpDependSinkVec>{}))) ||
        construct<OmpDependClause>(
            construct<OmpDependClause::Source>("SOURCE"_tok)) ||
        construct<OmpDependClause>(construct<OmpDependClause::InOut>(
            Parser<OmpDependenceType>{}, ":"_tok >> nonemptyList(designator))))

// LINEAR(list: linear-step)
TYPE_PARSER(construct<OmpLinearModifier>(
    "REF"_tok >> pure(OmpLinearModifier::Type::Ref) ||
    "VAL"_tok >> pure(OmpLinearModifier::Type::Val) ||
    "UVAL"_tok >> pure(OmpLinearModifier::Type::Uval)))

TYPE_CONTEXT_PARSER("Omp LINEAR clause"_en_US,
    construct<OmpLinearClause>(
        construct<OmpLinearClause>(construct<OmpLinearClause::WithModifier>(
            Parser<OmpLinearModifier>{}, parenthesized(nonemptyList(name)),
            maybe(":"_tok >> scalarIntConstantExpr))) ||
        construct<OmpLinearClause>(construct<OmpLinearClause::WithoutModifier>(
            nonemptyList(name), maybe(":"_tok >> scalarIntConstantExpr)))))

// ALIGNED(list: alignment)
TYPE_PARSER(construct<OmpAlignedClause>(
    nonemptyList(name), maybe(":"_tok) >> scalarIntConstantExpr))

TYPE_PARSER(construct<OmpNameList>(pure(OmpNameList::Kind::Object), name) ||
    construct<OmpNameList>("/" >> pure(OmpNameList::Kind::Common), name / "/"))

TYPE_PARSER(
    construct<OmpClause>(construct<OmpClause::Defaultmap>("DEFAULTMAP"_tok >>
        parenthesized("TOFROM"_tok >> ":"_tok >> "SCALAR"_tok))) ||
    construct<OmpClause>(construct<OmpClause::Inbranch>("INBRANCH"_tok)) ||
    construct<OmpClause>(construct<OmpClause::Mergeable>("MERGEABLE"_tok)) ||
    construct<OmpClause>(construct<OmpClause::Nogroup>("NOGROUP"_tok)) ||
    construct<OmpClause>(
        construct<OmpClause::Notinbranch>("NOTINBRANCH"_tok)) ||
    construct<OmpClause>(construct<OmpClause::Nowait>("NOWAIT"_tok)) ||
    construct<OmpClause>(construct<OmpClause::Untied>("UNTIED"_tok)) ||
    construct<OmpClause>(construct<OmpClause::Collapse>(
        "COLLAPSE"_tok >> parenthesized(scalarIntConstantExpr))) ||
    construct<OmpClause>(construct<OmpClause::Copyin>(
        "COPYIN"_tok >> parenthesized(nonemptyList(Parser<OmpNameList>{})))) ||
    construct<OmpClause>(construct<OmpClause::Copyprivate>("COPYPRIVATE"_tok >>
        parenthesized(nonemptyList(Parser<OmpNameList>{})))) ||
    construct<OmpClause>(construct<OmpClause::Device>(
        "DEVICE"_tok >> parenthesized(scalarIntExpr))) ||
    construct<OmpClause>(
        construct<OmpClause::DistSchedule>("DIST_SCHEDULE"_tok >>
            parenthesized("STATIC"_tok >> ","_tok >> scalarIntExpr))) ||
    construct<OmpClause>(construct<OmpClause::Final>(
        "FINAL"_tok >> parenthesized(scalarIntExpr))) ||
    construct<OmpClause>(
        construct<OmpClause::Firstprivate>("FIRSTPRIVATE"_tok >>
            parenthesized(nonemptyList(Parser<OmpNameList>{})))) ||
    construct<OmpClause>(construct<OmpClause::From>(
        "FROM"_tok >> parenthesized(nonemptyList(designator)))) ||
    construct<OmpClause>(construct<OmpClause::Grainsize>(
        "GRAINSIZE"_tok >> parenthesized(scalarIntExpr))) ||
    construct<OmpClause>(construct<OmpClause::Lastprivate>("LASTPRIVATE"_tok >>
        parenthesized(nonemptyList(Parser<OmpNameList>{})))) ||
    construct<OmpClause>(construct<OmpClause::Link>(
        "LINK"_tok >> parenthesized(nonemptyList(name)))) ||
    construct<OmpClause>(construct<OmpClause::NumTasks>(
        "NUM_TASKS"_tok >> parenthesized(scalarIntExpr))) ||
    construct<OmpClause>(construct<OmpClause::NumTeams>(
        "NUM_TEAMS"_tok >> parenthesized(scalarIntExpr))) ||
    construct<OmpClause>(construct<OmpClause::NumThreads>(
        "NUM_THREADS"_tok >> parenthesized(scalarIntExpr))) ||
    construct<OmpClause>(construct<OmpClause::Ordered>(
        "ORDERED"_tok >> maybe(parenthesized(scalarIntConstantExpr)))) ||
    construct<OmpClause>(construct<OmpClause::Priority>(
        "PRIORITY"_tok >> parenthesized(scalarIntExpr))) ||
    construct<OmpClause>(construct<OmpClause::Private>(
        "PRIVATE"_tok >> parenthesized(nonemptyList(Parser<OmpNameList>{})))) ||
    construct<OmpClause>(construct<OmpClause::Safelen>(
        "SAFELEN"_tok >> parenthesized(scalarIntConstantExpr))) ||
    construct<OmpClause>(construct<OmpClause::Shared>(
        "SHARED"_tok >> parenthesized(nonemptyList(Parser<OmpNameList>{})))) ||
    construct<OmpClause>(construct<OmpClause::Simdlen>(
        "SIMDLEN"_tok >> parenthesized(scalarIntConstantExpr))) ||
    construct<OmpClause>(construct<OmpClause::ThreadLimit>(
        "THREAD_LIMIT"_tok >> parenthesized(scalarIntExpr))) ||
    construct<OmpClause>(construct<OmpClause::To>(
        "TO"_tok >> parenthesized(nonemptyList(designator)))) ||
    construct<OmpClause>(construct<OmpClause::Uniform>(
        "UNIFORM"_tok >> parenthesized(nonemptyList(name)))) ||
    construct<OmpClause>(construct<OmpClause::UseDevicePtr>(
        "USE_DEVICE_PTR"_tok >> parenthesized(nonemptyList(name)))) ||
    construct<OmpClause>(
        "ALIGNED"_tok >> parenthesized(Parser<OmpAlignedClause>{})) ||
    construct<OmpClause>(
        "DEFAULT"_tok >> parenthesized(Parser<OmpDefaultClause>{})) ||
    construct<OmpClause>(
        "DEPEND"_tok >> parenthesized(Parser<OmpDependClause>{})) ||
    construct<OmpClause>("IF"_tok >> parenthesized(Parser<OmpIfClause>{})) ||
    construct<OmpClause>(
        "LINEAR"_tok >> parenthesized(Parser<OmpLinearClause>{})) ||
    construct<OmpClause>("MAP"_tok >> parenthesized(Parser<OmpMapClause>{})) ||
    construct<OmpClause>(
        "PROC_BIND"_tok >> parenthesized(Parser<OmpProcBindClause>{})) ||
    construct<OmpClause>(
        "REDUCTION"_tok >> parenthesized(Parser<OmpReductionClause>{})) ||
    construct<OmpClause>(
        "SCHEDULE"_tok >> parenthesized(Parser<OmpScheduleClause>{})))

TYPE_PARSER(skipStuffBeforeStatement >> "!$OMP END"_sptok >>
    (construct<OmpEndDirective>(Parser<OmpLoopDirective>{})))

// Omp directives enclosing do loop
TYPE_PARSER(
    construct<OmpLoopDirective>(
        construct<OmpLoopDirective::DistributeParallelDoSimd>(
            "DISTRIBUTE PARALLEL DO SIMD"_tok >> many(Parser<OmpClause>{}))) ||
    construct<OmpLoopDirective>(
        construct<OmpLoopDirective::DistributeParallelDo>(
            "DISTRIBUTE PARALLEL DO"_tok >> many(Parser<OmpClause>{}))) ||
    construct<OmpLoopDirective>(construct<OmpLoopDirective::DistributeSimd>(
        "DISTRIBUTE SIMD"_tok >> many(Parser<OmpClause>{}))) ||
    construct<OmpLoopDirective>(construct<OmpLoopDirective::Distribute>(
        "DISTRIBUTE"_tok >> many(Parser<OmpClause>{}))) ||
    construct<OmpLoopDirective>(construct<OmpLoopDirective::DoSimd>(
        "DO SIMD"_tok >> many(Parser<OmpClause>{}))) ||
    construct<OmpLoopDirective>(construct<OmpLoopDirective::Do>(
        "DO"_tok >> many(Parser<OmpClause>{}))) ||
    construct<OmpLoopDirective>(construct<OmpLoopDirective::ParallelDoSimd>(
        "PARALLEL DO SIMD"_tok >> many(Parser<OmpClause>{}))) ||
    construct<OmpLoopDirective>(construct<OmpLoopDirective::ParallelDo>(
        "PARALLEL DO"_tok >> many(Parser<OmpClause>{}))) ||
    construct<OmpLoopDirective>(construct<OmpLoopDirective::Simd>(
        "SIMD"_tok >> many(Parser<OmpClause>{}))) ||
    construct<OmpLoopDirective>(
        construct<OmpLoopDirective::TargetParallelDoSimd>(
            "TARGET PARALLEL DO SIMD"_tok >> many(Parser<OmpClause>{}))) ||
    construct<OmpLoopDirective>(construct<OmpLoopDirective::TargetParallelDo>(
        "TARGET PARALLEL DO"_tok >> many(Parser<OmpClause>{}))) ||
    construct<OmpLoopDirective>(construct<OmpLoopDirective::TargetSimd>(
        "TARGET SIMD"_tok >> many(Parser<OmpClause>{}))) ||
    construct<OmpLoopDirective>(
        construct<OmpLoopDirective::TargetTeamsDistributeParallelDoSimd>(
            "TARGET TEAMS DISTRIBUTE PARALLEL DO SIMD"_tok >>
            many(Parser<OmpClause>{}))) ||
    construct<OmpLoopDirective>(
        construct<OmpLoopDirective::TargetTeamsDistributeParallelDo>(
            "TARGET TEAMS DISTRIBUTE PARALLEL DO"_tok >>
            many(Parser<OmpClause>{}))) ||
    construct<OmpLoopDirective>(
        construct<OmpLoopDirective::TargetTeamsDistributeSimd>(
            "TARGET TEAMS DISTRIBUTE SIMD"_tok >> many(Parser<OmpClause>{}))) ||
    construct<OmpLoopDirective>(
        construct<OmpLoopDirective::TargetTeamsDistribute>(
            "TARGET TEAMS DISTRIBUTE"_tok >> many(Parser<OmpClause>{}))) ||
    construct<OmpLoopDirective>(construct<OmpLoopDirective::TaskloopSimd>(
        "TASKLOOP SIMD" >> many(Parser<OmpClause>{}))) ||
    construct<OmpLoopDirective>(construct<OmpLoopDirective::Taskloop>(
        "TASKLOOP" >> many(Parser<OmpClause>{}))) ||
    construct<OmpLoopDirective>(
        construct<OmpLoopDirective::TeamsDistributeParallelDoSimd>(
            "TEAMS DISTRIBUTE PARALLEL DO SIMD"_tok >>
            many(Parser<OmpClause>{}))) ||
    construct<OmpLoopDirective>(
        construct<OmpLoopDirective::TeamsDistributeParallelDo>(
            "TEAMS DISTRIBUTE PARALLEL DO"_tok >> many(Parser<OmpClause>{}))) ||
    construct<OmpLoopDirective>(
        construct<OmpLoopDirective::TeamsDistributeSimd>(
            "TEAMS DISTRIBUTE SIMD"_tok >> many(Parser<OmpClause>{}))) ||
    construct<OmpLoopDirective>(construct<OmpLoopDirective::TeamsDistribute>(
        "TEAMS DISTRIBUTE"_tok >> many(Parser<OmpClause>{}))))

TYPE_PARSER(construct<OmpStandaloneDirective>(
                construct<OmpStandaloneDirective::Barrier>(
                    "BARRIER"_tok >> many(Parser<OmpClause>{}))) ||
    construct<OmpStandaloneDirective>(
        construct<OmpStandaloneDirective::CancellationPoint>(
            "CANCELLATION POINT"_tok >> many(Parser<OmpClause>{}))) ||
    construct<OmpStandaloneDirective>(construct<OmpStandaloneDirective::Cancel>(
        "CANCEL"_tok >> many(Parser<OmpClause>{}))) ||
    construct<OmpStandaloneDirective>(construct<OmpStandaloneDirective::Flush>(
        "FLUSH"_tok >> many(Parser<OmpClause>{}))) ||
    construct<OmpStandaloneDirective>(
        construct<OmpStandaloneDirective::TargetEnterData>(
            "TARGET ENTER DATA"_tok >> many(Parser<OmpClause>{}))) ||
    construct<OmpStandaloneDirective>(
        construct<OmpStandaloneDirective::TargetExitData>(
            "TARGET EXIT DATA"_tok >> many(Parser<OmpClause>{}))) ||
    construct<OmpStandaloneDirective>(
        construct<OmpStandaloneDirective::TargetUpdate>(
            "TARGET UPDATE"_tok >> many(Parser<OmpClause>{}))) ||
    construct<OmpStandaloneDirective>(
        construct<OmpStandaloneDirective::Taskwait>(
            "TASKWAIT"_tok >> many(Parser<OmpClause>{}))) ||
    construct<OmpStandaloneDirective>(
        construct<OmpStandaloneDirective::Taskyield>(
            "TASKYIELD"_tok >> many(Parser<OmpClause>{}))))

TYPE_PARSER(
    construct<OpenMPLoopConstruct>(statement(Parser<OmpLoopDirective>{}),
        Parser<DoConstruct>{}, maybe(Parser<OmpEndDirective>{})))

TYPE_PARSER(construct<OpenMPStandaloneConstruct>(
    statement(Parser<OmpStandaloneDirective>{})))

TYPE_CONTEXT_PARSER("OpenMP construct"_en_US,
    beginOmpDirective >> (construct<OpenMPConstruct>(
                              indirect(Parser<OpenMPStandaloneConstruct>{})) ||
                             construct<OpenMPConstruct>(
                                 indirect(Parser<OpenMPLoopConstruct>{}))))

}  // namespace Fortran::parser
#endif  // FORTRAN_PARSER_OPENMP_GRAMMAR_H_
