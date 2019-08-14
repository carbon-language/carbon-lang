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

#ifndef FORTRAN_PARSER_GRAMMAR_H_
#define FORTRAN_PARSER_GRAMMAR_H_

// Top-level grammar specification for Fortran.  These parsers drive
// the tokenization parsers in cooked-tokens.h to consume characters,
// recognize the productions of Fortran, and to construct a parse tree.
// See ParserCombinators.md for documentation on the parser combinator
// library used here to implement an LL recursive descent recognizer.

#include "basic-parsers.h"
#include "characters.h"
#include "debug-parser.h"
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

namespace Fortran::parser {

// The productions that follow are derived from the draft Fortran 2018
// standard, with some necessary modifications to remove left recursion
// and some generalization in order to defer cases where parses depend
// on the definitions of symbols.  The "Rxxx" numbers that appear in
// comments refer to these numbered requirements in the Fortran standard.

// R507 declaration-construct ->
//        specification-construct | data-stmt | format-stmt |
//        entry-stmt | stmt-function-stmt
// N.B. These parsers incorporate recognition of some other statements that
// may have been misplaced in the sequence of statements that are acceptable
// as a specification part in order to improve error recovery.
// Also note that many instances of specification-part in the standard grammar
// are in contexts that impose constraints on the kinds of statements that
// are allowed, and so we have a variant production for declaration-construct
// that implements those constraints.
constexpr auto execPartLookAhead{
    first(actionStmt >> ok, ompEndLoopDirective >> ok, openmpConstruct >> ok,
        "ASSOCIATE ("_tok, "BLOCK"_tok, "SELECT"_tok, "CHANGE TEAM"_sptok,
        "CRITICAL"_tok, "DO"_tok, "IF ("_tok, "WHERE ("_tok, "FORALL ("_tok)};
constexpr auto declErrorRecovery{
    stmtErrorRecoveryStart >> !execPartLookAhead >> skipStmtErrorRecovery};
constexpr auto misplacedSpecificationStmt{Parser<UseStmt>{} >>
        fail<DeclarationConstruct>("misplaced USE statement"_err_en_US) ||
    Parser<ImportStmt>{} >>
        fail<DeclarationConstruct>(
            "IMPORT statements must follow any USE statements and precede all other declarations"_err_en_US) ||
    Parser<ImplicitStmt>{} >>
        fail<DeclarationConstruct>(
            "IMPLICIT statements must follow USE and IMPORT and precede all other declarations"_err_en_US)};

TYPE_PARSER(recovery(
    withMessage("expected declaration construct"_err_en_US,
        CONTEXT_PARSER("declaration construct"_en_US,
            first(construct<DeclarationConstruct>(specificationConstruct),
                construct<DeclarationConstruct>(statement(indirect(dataStmt))),
                construct<DeclarationConstruct>(
                    statement(indirect(formatStmt))),
                construct<DeclarationConstruct>(statement(indirect(entryStmt))),
                construct<DeclarationConstruct>(
                    statement(indirect(Parser<StmtFunctionStmt>{}))),
                misplacedSpecificationStmt))),
    construct<DeclarationConstruct>(declErrorRecovery)))

// R507 variant of declaration-construct for use in limitedSpecificationPart.
constexpr auto invalidDeclarationStmt{formatStmt >>
        fail<DeclarationConstruct>(
            "FORMAT statements are not permitted in this specification part"_err_en_US) ||
    entryStmt >>
        fail<DeclarationConstruct>(
            "ENTRY statements are not permitted in this specification part"_err_en_US)};

constexpr auto limitedDeclarationConstruct{recovery(
    withMessage("expected declaration construct"_err_en_US,
        inContext("declaration construct"_en_US,
            first(construct<DeclarationConstruct>(specificationConstruct),
                construct<DeclarationConstruct>(statement(indirect(dataStmt))),
                misplacedSpecificationStmt, invalidDeclarationStmt))),
    construct<DeclarationConstruct>(
        stmtErrorRecoveryStart >> skipStmtErrorRecovery))};

// R508 specification-construct ->
//        derived-type-def | enum-def | generic-stmt | interface-block |
//        parameter-stmt | procedure-declaration-stmt |
//        other-specification-stmt | type-declaration-stmt
TYPE_CONTEXT_PARSER("specification construct"_en_US,
    first(construct<SpecificationConstruct>(indirect(Parser<DerivedTypeDef>{})),
        construct<SpecificationConstruct>(indirect(Parser<EnumDef>{})),
        construct<SpecificationConstruct>(
            statement(indirect(Parser<GenericStmt>{}))),
        construct<SpecificationConstruct>(indirect(interfaceBlock)),
        construct<SpecificationConstruct>(statement(indirect(parameterStmt))),
        construct<SpecificationConstruct>(
            statement(indirect(oldParameterStmt))),
        construct<SpecificationConstruct>(
            statement(indirect(Parser<ProcedureDeclarationStmt>{}))),
        construct<SpecificationConstruct>(
            statement(Parser<OtherSpecificationStmt>{})),
        construct<SpecificationConstruct>(
            statement(indirect(typeDeclarationStmt))),
        construct<SpecificationConstruct>(indirect(Parser<StructureDef>{})),
        construct<SpecificationConstruct>(indirect(openmpDeclarativeConstruct)),
        construct<SpecificationConstruct>(indirect(compilerDirective))))

// R513 other-specification-stmt ->
//        access-stmt | allocatable-stmt | asynchronous-stmt | bind-stmt |
//        codimension-stmt | contiguous-stmt | dimension-stmt | external-stmt |
//        intent-stmt | intrinsic-stmt | namelist-stmt | optional-stmt |
//        pointer-stmt | protected-stmt | save-stmt | target-stmt |
//        volatile-stmt | value-stmt | common-stmt | equivalence-stmt
TYPE_PARSER(first(
    construct<OtherSpecificationStmt>(indirect(Parser<AccessStmt>{})),
    construct<OtherSpecificationStmt>(indirect(Parser<AllocatableStmt>{})),
    construct<OtherSpecificationStmt>(indirect(Parser<AsynchronousStmt>{})),
    construct<OtherSpecificationStmt>(indirect(Parser<BindStmt>{})),
    construct<OtherSpecificationStmt>(indirect(Parser<CodimensionStmt>{})),
    construct<OtherSpecificationStmt>(indirect(Parser<ContiguousStmt>{})),
    construct<OtherSpecificationStmt>(indirect(Parser<DimensionStmt>{})),
    construct<OtherSpecificationStmt>(indirect(Parser<ExternalStmt>{})),
    construct<OtherSpecificationStmt>(indirect(Parser<IntentStmt>{})),
    construct<OtherSpecificationStmt>(indirect(Parser<IntrinsicStmt>{})),
    construct<OtherSpecificationStmt>(indirect(Parser<NamelistStmt>{})),
    construct<OtherSpecificationStmt>(indirect(Parser<OptionalStmt>{})),
    construct<OtherSpecificationStmt>(indirect(Parser<PointerStmt>{})),
    construct<OtherSpecificationStmt>(indirect(Parser<ProtectedStmt>{})),
    construct<OtherSpecificationStmt>(indirect(Parser<SaveStmt>{})),
    construct<OtherSpecificationStmt>(indirect(Parser<TargetStmt>{})),
    construct<OtherSpecificationStmt>(indirect(Parser<ValueStmt>{})),
    construct<OtherSpecificationStmt>(indirect(Parser<VolatileStmt>{})),
    construct<OtherSpecificationStmt>(indirect(Parser<CommonStmt>{})),
    construct<OtherSpecificationStmt>(indirect(Parser<EquivalenceStmt>{})),
    construct<OtherSpecificationStmt>(indirect(Parser<BasedPointerStmt>{}))))

// R604 constant ->  literal-constant | named-constant
// Used only via R607 int-constant and R845 data-stmt-constant.
// The look-ahead check prevents occlusion of constant-subobject in
// data-stmt-constant.
TYPE_PARSER(construct<ConstantValue>(literalConstant) ||
    construct<ConstantValue>(namedConstant / !"%"_tok / !"("_tok))

// R608 intrinsic-operator ->
//        power-op | mult-op | add-op | concat-op | rel-op |
//        not-op | and-op | or-op | equiv-op
// R610 extended-intrinsic-op -> intrinsic-operator
// These parsers must be ordered carefully to avoid misrecognition.
constexpr auto namedIntrinsicOperator{
    ".LT." >> pure(DefinedOperator::IntrinsicOperator::LT) ||
    ".LE." >> pure(DefinedOperator::IntrinsicOperator::LE) ||
    ".EQ." >> pure(DefinedOperator::IntrinsicOperator::EQ) ||
    ".NE." >> pure(DefinedOperator::IntrinsicOperator::NE) ||
    ".GE." >> pure(DefinedOperator::IntrinsicOperator::GE) ||
    ".GT." >> pure(DefinedOperator::IntrinsicOperator::GT) ||
    ".NOT." >> pure(DefinedOperator::IntrinsicOperator::NOT) ||
    ".AND." >> pure(DefinedOperator::IntrinsicOperator::AND) ||
    ".OR." >> pure(DefinedOperator::IntrinsicOperator::OR) ||
    ".EQV." >> pure(DefinedOperator::IntrinsicOperator::EQV) ||
    ".NEQV." >> pure(DefinedOperator::IntrinsicOperator::NEQV) ||
    extension<LanguageFeature::XOROperator>(
        ".XOR." >> pure(DefinedOperator::IntrinsicOperator::XOR)) ||
    extension<LanguageFeature::LogicalAbbreviations>(
        ".N." >> pure(DefinedOperator::IntrinsicOperator::NOT) ||
        ".A." >> pure(DefinedOperator::IntrinsicOperator::AND) ||
        ".O." >> pure(DefinedOperator::IntrinsicOperator::OR) ||
        ".X." >> pure(DefinedOperator::IntrinsicOperator::XOR))};

constexpr auto intrinsicOperator{
    "**" >> pure(DefinedOperator::IntrinsicOperator::Power) ||
    "*" >> pure(DefinedOperator::IntrinsicOperator::Multiply) ||
    "//" >> pure(DefinedOperator::IntrinsicOperator::Concat) ||
    "/=" >> pure(DefinedOperator::IntrinsicOperator::NE) ||
    "/" >> pure(DefinedOperator::IntrinsicOperator::Divide) ||
    "+" >> pure(DefinedOperator::IntrinsicOperator::Add) ||
    "-" >> pure(DefinedOperator::IntrinsicOperator::Subtract) ||
    "<=" >> pure(DefinedOperator::IntrinsicOperator::LE) ||
    extension<LanguageFeature::AlternativeNE>(
        "<>" >> pure(DefinedOperator::IntrinsicOperator::NE)) ||
    "<" >> pure(DefinedOperator::IntrinsicOperator::LT) ||
    "==" >> pure(DefinedOperator::IntrinsicOperator::EQ) ||
    ">=" >> pure(DefinedOperator::IntrinsicOperator::GE) ||
    ">" >> pure(DefinedOperator::IntrinsicOperator::GT) ||
    namedIntrinsicOperator};

// R609 defined-operator ->
//        defined-unary-op | defined-binary-op | extended-intrinsic-op
TYPE_PARSER(construct<DefinedOperator>(intrinsicOperator) ||
    construct<DefinedOperator>(definedOpName))

// R401 xzy-list -> xzy [, xzy]...
template<typename PA> inline constexpr auto nonemptyList(const PA &p) {
  return nonemptySeparated(p, ","_tok);  // p-list
}

template<typename PA>
inline constexpr auto nonemptyList(MessageFixedText error, const PA &p) {
  return withMessage(error, nonemptySeparated(p, ","_tok));  // p-list
}

template<typename PA> inline constexpr auto optionalList(const PA &p) {
  return defaulted(nonemptySeparated(p, ","_tok));  // [p-list]
}

// R402 xzy-name -> name

// R403 scalar-xyz -> xyz
// Also define constant-xyz, int-xyz, default-char-xyz.
template<typename PA> inline constexpr auto scalar(const PA &p) {
  return construct<Scalar<typename PA::resultType>>(p);  // scalar-p
}

template<typename PA> inline constexpr auto constant(const PA &p) {
  return construct<Constant<typename PA::resultType>>(p);  // constant-p
}

template<typename PA> inline constexpr auto integer(const PA &p) {
  return construct<Integer<typename PA::resultType>>(p);  // int-p
}

template<typename PA> inline constexpr auto logical(const PA &p) {
  return construct<Logical<typename PA::resultType>>(p);  // logical-p
}

template<typename PA> inline constexpr auto defaultChar(const PA &p) {
  return construct<DefaultChar<typename PA::resultType>>(p);  // default-char-p
}

// R1024 logical-expr -> expr
constexpr auto logicalExpr{logical(indirect(expr))};
constexpr auto scalarLogicalExpr{scalar(logicalExpr)};

// R1025 default-char-expr -> expr
constexpr auto defaultCharExpr{defaultChar(indirect(expr))};
constexpr auto scalarDefaultCharExpr{scalar(defaultCharExpr)};

// R1026 int-expr -> expr
constexpr auto intExpr{integer(indirect(expr))};
constexpr auto scalarIntExpr{scalar(intExpr)};

// R1029 constant-expr -> expr
constexpr auto constantExpr{constant(indirect(expr))};
constexpr auto scalarExpr{scalar(indirect(expr))};

// R1030 default-char-constant-expr -> default-char-expr
constexpr auto scalarDefaultCharConstantExpr{scalar(defaultChar(constantExpr))};

// R1031 int-constant-expr -> int-expr
constexpr auto intConstantExpr{integer(constantExpr)};
constexpr auto scalarIntConstantExpr{scalar(intConstantExpr)};

// R501 program -> program-unit [program-unit]...
// This is the top-level production for the Fortran language.
// F'2018 6.3.1 defines a program unit as a sequence of one or more lines,
// implying that a line can't be part of two distinct program units.
// Consequently, a program unit END statement should be the last statement
// on its line.  We parse those END statements via unterminatedStatement()
// and then skip over the end of the line here.
TYPE_PARSER(construct<Program>(some(StartNewSubprogram{} >>
                Parser<ProgramUnit>{} / skipMany(";"_tok) / space /
                    recovery(endOfLine, SkipPast<'\n'>{}))) /
    skipStuffBeforeStatement)

// R502 program-unit ->
//        main-program | external-subprogram | module | submodule | block-data
// R503 external-subprogram -> function-subprogram | subroutine-subprogram
// N.B. "module" must precede "external-subprogram" in this sequence of
// alternatives to avoid ambiguity with the MODULE keyword prefix that
// they recognize.  I.e., "modulesubroutinefoo" should start a module
// "subroutinefoo", not a subroutine "foo" with the MODULE prefix.  The
// ambiguity is exacerbated by the extension that accepts a function
// statement without an otherwise empty list of dummy arguments.  That
// MODULE prefix is disallowed by a constraint (C1547) in this context,
// so the standard language is not ambiguous, but disabling its misrecognition
// here would require context-sensitive keyword recognition or (or via)
// variant parsers for several productions; giving the "module" production
// priority here is a cleaner solution, though regrettably subtle.  Enforcing
// C1547 is done in semantics.
TYPE_PARSER(construct<ProgramUnit>(indirect(Parser<Module>{})) ||
    construct<ProgramUnit>(indirect(functionSubprogram)) ||
    construct<ProgramUnit>(indirect(subroutineSubprogram)) ||
    construct<ProgramUnit>(indirect(Parser<Submodule>{})) ||
    construct<ProgramUnit>(indirect(Parser<BlockData>{})) ||
    construct<ProgramUnit>(indirect(Parser<MainProgram>{})))

// R504 specification-part ->
//         [use-stmt]... [import-stmt]... [implicit-part]
//         [declaration-construct]...
TYPE_CONTEXT_PARSER("specification part"_en_US,
    construct<SpecificationPart>(many(openmpDeclarativeConstruct),
        many(unambiguousStatement(indirect(Parser<UseStmt>{}))),
        many(unambiguousStatement(indirect(Parser<ImportStmt>{}))),
        implicitPart, many(declarationConstruct)))

// R504 variant for many contexts (modules, submodules, BLOCK DATA subprograms,
// and interfaces) which have constraints on their specification parts that
// preclude FORMAT, ENTRY, and statement functions, and benefit from
// specialized error recovery in the event of a spurious executable
// statement.
constexpr auto limitedSpecificationPart{inContext("specification part"_en_US,
    construct<SpecificationPart>(many(openmpDeclarativeConstruct),
        many(unambiguousStatement(indirect(Parser<UseStmt>{}))),
        many(unambiguousStatement(indirect(Parser<ImportStmt>{}))),
        implicitPart, many(limitedDeclarationConstruct)))};

// R505 implicit-part -> [implicit-part-stmt]... implicit-stmt
// TODO: Can overshoot; any trailing PARAMETER, FORMAT, & ENTRY
// statements after the last IMPLICIT should be transferred to the
// list of declaration-constructs.
TYPE_CONTEXT_PARSER("implicit part"_en_US,
    construct<ImplicitPart>(many(Parser<ImplicitPartStmt>{})))

// R506 implicit-part-stmt ->
//         implicit-stmt | parameter-stmt | format-stmt | entry-stmt
TYPE_PARSER(first(
    construct<ImplicitPartStmt>(statement(indirect(Parser<ImplicitStmt>{}))),
    construct<ImplicitPartStmt>(statement(indirect(parameterStmt))),
    construct<ImplicitPartStmt>(statement(indirect(oldParameterStmt))),
    construct<ImplicitPartStmt>(statement(indirect(formatStmt))),
    construct<ImplicitPartStmt>(statement(indirect(entryStmt)))))

// R512 internal-subprogram -> function-subprogram | subroutine-subprogram
// Internal subprograms are not program units, so their END statements
// can be followed by ';' and another statement on the same line.
TYPE_CONTEXT_PARSER("internal subprogram"_en_US,
    (construct<InternalSubprogram>(indirect(functionSubprogram)) ||
        construct<InternalSubprogram>(indirect(subroutineSubprogram))) /
        forceEndOfStmt)

// R511 internal-subprogram-part -> contains-stmt [internal-subprogram]...
TYPE_CONTEXT_PARSER("internal subprogram part"_en_US,
    construct<InternalSubprogramPart>(statement(containsStmt),
        many(StartNewSubprogram{} >> Parser<InternalSubprogram>{})))

// R515 action-stmt ->
//        allocate-stmt | assignment-stmt | backspace-stmt | call-stmt |
//        close-stmt | continue-stmt | cycle-stmt | deallocate-stmt |
//        endfile-stmt | error-stop-stmt | event-post-stmt | event-wait-stmt |
//        exit-stmt | fail-image-stmt | flush-stmt | form-team-stmt |
//        goto-stmt | if-stmt | inquire-stmt | lock-stmt | nullify-stmt |
//        open-stmt | pointer-assignment-stmt | print-stmt | read-stmt |
//        return-stmt | rewind-stmt | stop-stmt | sync-all-stmt |
//        sync-images-stmt | sync-memory-stmt | sync-team-stmt | unlock-stmt |
//        wait-stmt | where-stmt | write-stmt | computed-goto-stmt | forall-stmt
// R1159 continue-stmt -> CONTINUE
// R1163 fail-image-stmt -> FAIL IMAGE
TYPE_PARSER(first(construct<ActionStmt>(indirect(Parser<AllocateStmt>{})),
    construct<ActionStmt>(indirect(assignmentStmt)),
    construct<ActionStmt>(indirect(pointerAssignmentStmt)),
    construct<ActionStmt>(indirect(Parser<BackspaceStmt>{})),
    construct<ActionStmt>(indirect(Parser<CallStmt>{})),
    construct<ActionStmt>(indirect(Parser<CloseStmt>{})),
    construct<ActionStmt>(construct<ContinueStmt>("CONTINUE"_tok)),
    construct<ActionStmt>(indirect(Parser<CycleStmt>{})),
    construct<ActionStmt>(indirect(Parser<DeallocateStmt>{})),
    construct<ActionStmt>(indirect(Parser<EndfileStmt>{})),
    construct<ActionStmt>(indirect(Parser<EventPostStmt>{})),
    construct<ActionStmt>(indirect(Parser<EventWaitStmt>{})),
    construct<ActionStmt>(indirect(Parser<ExitStmt>{})),
    construct<ActionStmt>(construct<FailImageStmt>("FAIL IMAGE"_sptok)),
    construct<ActionStmt>(indirect(Parser<FlushStmt>{})),
    construct<ActionStmt>(indirect(Parser<FormTeamStmt>{})),
    construct<ActionStmt>(indirect(Parser<GotoStmt>{})),
    construct<ActionStmt>(indirect(Parser<IfStmt>{})),
    construct<ActionStmt>(indirect(Parser<InquireStmt>{})),
    construct<ActionStmt>(indirect(Parser<LockStmt>{})),
    construct<ActionStmt>(indirect(Parser<NullifyStmt>{})),
    construct<ActionStmt>(indirect(Parser<OpenStmt>{})),
    construct<ActionStmt>(indirect(Parser<PrintStmt>{})),
    construct<ActionStmt>(indirect(Parser<ReadStmt>{})),
    construct<ActionStmt>(indirect(Parser<ReturnStmt>{})),
    construct<ActionStmt>(indirect(Parser<RewindStmt>{})),
    construct<ActionStmt>(indirect(Parser<StopStmt>{})),  // & error-stop-stmt
    construct<ActionStmt>(indirect(Parser<SyncAllStmt>{})),
    construct<ActionStmt>(indirect(Parser<SyncImagesStmt>{})),
    construct<ActionStmt>(indirect(Parser<SyncMemoryStmt>{})),
    construct<ActionStmt>(indirect(Parser<SyncTeamStmt>{})),
    construct<ActionStmt>(indirect(Parser<UnlockStmt>{})),
    construct<ActionStmt>(indirect(Parser<WaitStmt>{})),
    construct<ActionStmt>(indirect(whereStmt)),
    construct<ActionStmt>(indirect(Parser<WriteStmt>{})),
    construct<ActionStmt>(indirect(Parser<ComputedGotoStmt>{})),
    construct<ActionStmt>(indirect(forallStmt)),
    construct<ActionStmt>(indirect(Parser<ArithmeticIfStmt>{})),
    construct<ActionStmt>(indirect(Parser<AssignStmt>{})),
    construct<ActionStmt>(indirect(Parser<AssignedGotoStmt>{})),
    construct<ActionStmt>(indirect(Parser<PauseStmt>{}))))

// Fortran allows the statement with the corresponding label at the end of
// a do-construct that begins with an old-style label-do-stmt to be a
// new-style END DO statement; e.g., DO 10 I=1,N; ...; 10 END DO.  Usually,
// END DO statements appear only at the ends of do-constructs that begin
// with a nonlabel-do-stmt, so care must be taken to recognize this case and
// essentially treat them like CONTINUE statements.

// R514 executable-construct ->
//        action-stmt | associate-construct | block-construct |
//        case-construct | change-team-construct | critical-construct |
//        do-construct | if-construct | select-rank-construct |
//        select-type-construct | where-construct | forall-construct
constexpr auto executableConstruct{
    first(construct<ExecutableConstruct>(CapturedLabelDoStmt{}),
        construct<ExecutableConstruct>(EndDoStmtForCapturedLabelDoStmt{}),
        construct<ExecutableConstruct>(indirect(Parser<DoConstruct>{})),
        // Attempt DO statements before assignment statements for better
        // error messages in cases like "DO10I=1,(error)".
        construct<ExecutableConstruct>(statement(actionStmt)),
        construct<ExecutableConstruct>(indirect(Parser<AssociateConstruct>{})),
        construct<ExecutableConstruct>(indirect(Parser<BlockConstruct>{})),
        construct<ExecutableConstruct>(indirect(Parser<CaseConstruct>{})),
        construct<ExecutableConstruct>(indirect(Parser<ChangeTeamConstruct>{})),
        construct<ExecutableConstruct>(indirect(Parser<CriticalConstruct>{})),
        construct<ExecutableConstruct>(indirect(Parser<IfConstruct>{})),
        construct<ExecutableConstruct>(indirect(Parser<SelectRankConstruct>{})),
        construct<ExecutableConstruct>(indirect(Parser<SelectTypeConstruct>{})),
        construct<ExecutableConstruct>(indirect(whereConstruct)),
        construct<ExecutableConstruct>(indirect(forallConstruct)),
        construct<ExecutableConstruct>(indirect(ompEndLoopDirective)),
        construct<ExecutableConstruct>(indirect(openmpConstruct)),
        construct<ExecutableConstruct>(indirect(compilerDirective)))};

// R510 execution-part-construct ->
//        executable-construct | format-stmt | entry-stmt | data-stmt
// Extension (PGI/Intel): also accept NAMELIST in execution part
constexpr auto obsoleteExecutionPartConstruct{recovery(ignoredStatementPrefix >>
        fail<ExecutionPartConstruct>(
            "obsolete legacy extension is not supported"_err_en_US),
    construct<ExecutionPartConstruct>(construct<ErrorRecovery>(ok /
        statement("REDIMENSION" >> name /
                parenthesized(nonemptyList(Parser<AllocateShapeSpec>{}))))))};

TYPE_PARSER(recovery(
    withMessage("expected execution part construct"_err_en_US,
        CONTEXT_PARSER("execution part construct"_en_US,
            first(construct<ExecutionPartConstruct>(executableConstruct),
                construct<ExecutionPartConstruct>(
                    statement(indirect(formatStmt))),
                construct<ExecutionPartConstruct>(
                    statement(indirect(entryStmt))),
                construct<ExecutionPartConstruct>(
                    statement(indirect(dataStmt))),
                extension<LanguageFeature::ExecutionPartNamelist>(
                    construct<ExecutionPartConstruct>(
                        statement(indirect(Parser<NamelistStmt>{})))),
                obsoleteExecutionPartConstruct))),
    construct<ExecutionPartConstruct>(executionPartErrorRecovery)))

// R509 execution-part -> executable-construct [execution-part-construct]...
TYPE_CONTEXT_PARSER("execution part"_en_US,
    construct<ExecutionPart>(many(executionPartConstruct)))

// R605 literal-constant ->
//        int-literal-constant | real-literal-constant |
//        complex-literal-constant | logical-literal-constant |
//        char-literal-constant | boz-literal-constant
TYPE_PARSER(
    first(construct<LiteralConstant>(Parser<HollerithLiteralConstant>{}),
        construct<LiteralConstant>(realLiteralConstant),
        construct<LiteralConstant>(intLiteralConstant),
        construct<LiteralConstant>(Parser<ComplexLiteralConstant>{}),
        construct<LiteralConstant>(Parser<BOZLiteralConstant>{}),
        construct<LiteralConstant>(charLiteralConstant),
        construct<LiteralConstant>(Parser<LogicalLiteralConstant>{})))

// R606 named-constant -> name
TYPE_PARSER(construct<NamedConstant>(name))

// R701 type-param-value -> scalar-int-expr | * | :
constexpr auto star{construct<Star>("*"_tok)};
TYPE_PARSER(construct<TypeParamValue>(scalarIntExpr) ||
    construct<TypeParamValue>(star) ||
    construct<TypeParamValue>(construct<TypeParamValue::Deferred>(":"_tok)))

// R702 type-spec -> intrinsic-type-spec | derived-type-spec
// N.B. This type-spec production is one of two instances in the Fortran
// grammar where intrinsic types and bare derived type names can clash;
// the other is below in R703 declaration-type-spec.  Look-ahead is required
// to disambiguate the cases where a derived type name begins with the name
// of an intrinsic type, e.g., REALITY.
TYPE_CONTEXT_PARSER("type spec"_en_US,
    construct<TypeSpec>(intrinsicTypeSpec / lookAhead("::"_tok || ")"_tok)) ||
        construct<TypeSpec>(derivedTypeSpec))

// R703 declaration-type-spec ->
//        intrinsic-type-spec | TYPE ( intrinsic-type-spec ) |
//        TYPE ( derived-type-spec ) | CLASS ( derived-type-spec ) |
//        CLASS ( * ) | TYPE ( * )
// N.B. It is critical to distribute "parenthesized()" over the alternatives
// for TYPE (...), rather than putting the alternatives within it, which
// would fail on "TYPE(real_derived)" with a misrecognition of "real" as an
// intrinsic-type-spec.
TYPE_CONTEXT_PARSER("declaration type spec"_en_US,
    construct<DeclarationTypeSpec>(intrinsicTypeSpec) ||
        "TYPE" >>
            (parenthesized(construct<DeclarationTypeSpec>(intrinsicTypeSpec)) ||
                parenthesized(construct<DeclarationTypeSpec>(
                    construct<DeclarationTypeSpec::Type>(derivedTypeSpec))) ||
                construct<DeclarationTypeSpec>(
                    "( * )" >> construct<DeclarationTypeSpec::TypeStar>())) ||
        "CLASS" >> parenthesized(construct<DeclarationTypeSpec>(
                                     construct<DeclarationTypeSpec::Class>(
                                         derivedTypeSpec)) ||
                       construct<DeclarationTypeSpec>("*" >>
                           construct<DeclarationTypeSpec::ClassStar>())) ||
        extension<LanguageFeature::DECStructures>(
            construct<DeclarationTypeSpec>(
                construct<DeclarationTypeSpec::Record>(
                    "RECORD /" >> name / "/"))))

// R704 intrinsic-type-spec ->
//        integer-type-spec | REAL [kind-selector] | DOUBLE PRECISION |
//        COMPLEX [kind-selector] | CHARACTER [char-selector] |
//        LOGICAL [kind-selector]
// Extensions: DOUBLE COMPLEX, BYTE
TYPE_CONTEXT_PARSER("intrinsic type spec"_en_US,
    first(construct<IntrinsicTypeSpec>(integerTypeSpec),
        construct<IntrinsicTypeSpec>(
            construct<IntrinsicTypeSpec::Real>("REAL" >> maybe(kindSelector))),
        construct<IntrinsicTypeSpec>("DOUBLE PRECISION" >>
            construct<IntrinsicTypeSpec::DoublePrecision>()),
        construct<IntrinsicTypeSpec>(construct<IntrinsicTypeSpec::Complex>(
            "COMPLEX" >> maybe(kindSelector))),
        construct<IntrinsicTypeSpec>(construct<IntrinsicTypeSpec::Character>(
            "CHARACTER" >> maybe(Parser<CharSelector>{}))),
        construct<IntrinsicTypeSpec>(construct<IntrinsicTypeSpec::Logical>(
            "LOGICAL" >> maybe(kindSelector))),
        construct<IntrinsicTypeSpec>("DOUBLE COMPLEX" >>
            extension<LanguageFeature::DoubleComplex>(
                construct<IntrinsicTypeSpec::DoubleComplex>())),
        extension<LanguageFeature::Byte>(
            construct<IntrinsicTypeSpec>(construct<IntegerTypeSpec>(
                "BYTE" >> construct<std::optional<KindSelector>>(pure(1)))))))

// R705 integer-type-spec -> INTEGER [kind-selector]
TYPE_PARSER(construct<IntegerTypeSpec>("INTEGER" >> maybe(kindSelector)))

// R706 kind-selector -> ( [KIND =] scalar-int-constant-expr )
// Legacy extension: kind-selector -> * digit-string
TYPE_PARSER(construct<KindSelector>(
                parenthesized(maybe("KIND ="_tok) >> scalarIntConstantExpr)) ||
    extension<LanguageFeature::StarKind>(construct<KindSelector>(
        construct<KindSelector::StarSize>("*" >> digitString64 / spaceCheck))))

// R707 signed-int-literal-constant -> [sign] int-literal-constant
TYPE_PARSER(sourced(construct<SignedIntLiteralConstant>(
    SignedIntLiteralConstantWithoutKind{}, maybe(underscore >> kindParam))))

// R708 int-literal-constant -> digit-string [_ kind-param]
// The negated look-ahead for a trailing underscore prevents misrecognition
// when the digit string is a numeric kind parameter of a character literal.
TYPE_PARSER(construct<IntLiteralConstant>(
    space >> digitString, maybe(underscore >> kindParam) / !underscore))

// R709 kind-param -> digit-string | scalar-int-constant-name
TYPE_PARSER(construct<KindParam>(digitString64) ||
    construct<KindParam>(scalar(integer(constant(name)))))

// R712 sign -> + | -
// N.B. A sign constitutes a whole token, so a space is allowed in free form
// after the sign and before a real-literal-constant or
// complex-literal-constant.  A sign is not a unary operator in these contexts.
constexpr auto sign{
    "+"_tok >> pure(Sign::Positive) || "-"_tok >> pure(Sign::Negative)};

// R713 signed-real-literal-constant -> [sign] real-literal-constant
constexpr auto signedRealLiteralConstant{
    construct<SignedRealLiteralConstant>(maybe(sign), realLiteralConstant)};

// R714 real-literal-constant ->
//        significand [exponent-letter exponent] [_ kind-param] |
//        digit-string exponent-letter exponent [_ kind-param]
// R715 significand -> digit-string . [digit-string] | . digit-string
// R716 exponent-letter -> E | D
// Extension: Q
// R717 exponent -> signed-digit-string
constexpr auto exponentPart{
    ("ed"_ch || extension<LanguageFeature::QuadPrecision>("q"_ch)) >>
    SignedDigitString{}};

TYPE_CONTEXT_PARSER("REAL literal constant"_en_US,
    space >>
        construct<RealLiteralConstant>(
            sourced((digitString >> "."_ch >>
                            !(some(letter) >>
                                "."_ch /* don't misinterpret 1.AND. */) >>
                            maybe(digitString) >> maybe(exponentPart) >> ok ||
                        "."_ch >> digitString >> maybe(exponentPart) >> ok ||
                        digitString >> exponentPart >> ok) >>
                construct<RealLiteralConstant::Real>()),
            maybe(underscore >> kindParam)))

// R718 complex-literal-constant -> ( real-part , imag-part )
TYPE_CONTEXT_PARSER("COMPLEX literal constant"_en_US,
    parenthesized(construct<ComplexLiteralConstant>(
        Parser<ComplexPart>{} / ",", Parser<ComplexPart>{})))

// PGI/Intel extension: signed complex literal constant
TYPE_PARSER(construct<SignedComplexLiteralConstant>(
    sign, Parser<ComplexLiteralConstant>{}))

// R719 real-part ->
//        signed-int-literal-constant | signed-real-literal-constant |
//        named-constant
// R720 imag-part ->
//        signed-int-literal-constant | signed-real-literal-constant |
//        named-constant
TYPE_PARSER(construct<ComplexPart>(signedRealLiteralConstant) ||
    construct<ComplexPart>(signedIntLiteralConstant) ||
    construct<ComplexPart>(namedConstant))

// R721 char-selector ->
//        length-selector |
//        ( LEN = type-param-value , KIND = scalar-int-constant-expr ) |
//        ( type-param-value , [KIND =] scalar-int-constant-expr ) |
//        ( KIND = scalar-int-constant-expr [, LEN = type-param-value] )
TYPE_PARSER(construct<CharSelector>(Parser<LengthSelector>{}) ||
    parenthesized(construct<CharSelector>(
        "LEN =" >> typeParamValue, ", KIND =" >> scalarIntConstantExpr)) ||
    parenthesized(construct<CharSelector>(
        typeParamValue / ",", maybe("KIND ="_tok) >> scalarIntConstantExpr)) ||
    parenthesized(construct<CharSelector>(
        "KIND =" >> scalarIntConstantExpr, maybe(", LEN =" >> typeParamValue))))

// R722 length-selector -> ( [LEN =] type-param-value ) | * char-length [,]
// N.B. The trailing [,] in the production is permitted by the Standard
// only in the context of a type-declaration-stmt, but even with that
// limitation, it would seem to be unnecessary and buggy to consume the comma
// here.
TYPE_PARSER(construct<LengthSelector>(
                parenthesized(maybe("LEN ="_tok) >> typeParamValue)) ||
    construct<LengthSelector>("*" >> charLength /* / maybe(","_tok) */))

// R723 char-length -> ( type-param-value ) | digit-string
TYPE_PARSER(construct<CharLength>(parenthesized(typeParamValue)) ||
    construct<CharLength>(space >> digitString64 / spaceCheck))

// R724 char-literal-constant ->
//        [kind-param _] ' [rep-char]... ' |
//        [kind-param _] " [rep-char]... "
// "rep-char" is any non-control character.  Doubled interior quotes are
// combined.  Backslash escapes can be enabled.
// N.B. charLiteralConstantWithoutKind does not skip preceding space.
// N.B. the parsing of "name" takes care to not consume the '_'.
constexpr auto charLiteralConstantWithoutKind{
    "'"_ch >> CharLiteral<'\''>{} || "\""_ch >> CharLiteral<'"'>{}};

TYPE_CONTEXT_PARSER("CHARACTER literal constant"_en_US,
    construct<CharLiteralConstant>(
        kindParam / underscore, charLiteralConstantWithoutKind) ||
        construct<CharLiteralConstant>(construct<std::optional<KindParam>>(),
            space >> charLiteralConstantWithoutKind))

// deprecated: Hollerith literals
constexpr auto rawHollerithLiteral{
    deprecated<LanguageFeature::Hollerith>(HollerithLiteral{})};

TYPE_CONTEXT_PARSER(
    "Hollerith"_en_US, construct<HollerithLiteralConstant>(rawHollerithLiteral))

// R725 logical-literal-constant ->
//        .TRUE. [_ kind-param] | .FALSE. [_ kind-param]
// Also accept .T. and .F. as extensions.
TYPE_PARSER(construct<LogicalLiteralConstant>(
                logicalTRUE, maybe(underscore >> kindParam)) ||
    construct<LogicalLiteralConstant>(
        logicalFALSE, maybe(underscore >> kindParam)))

// R726 derived-type-def ->
//        derived-type-stmt [type-param-def-stmt]...
//        [private-or-sequence]... [component-part]
//        [type-bound-procedure-part] end-type-stmt
// R735 component-part -> [component-def-stmt]...
TYPE_CONTEXT_PARSER("derived type definition"_en_US,
    construct<DerivedTypeDef>(statement(Parser<DerivedTypeStmt>{}),
        many(unambiguousStatement(Parser<TypeParamDefStmt>{})),
        many(statement(Parser<PrivateOrSequence>{})),
        many(inContext("component"_en_US,
            unambiguousStatement(Parser<ComponentDefStmt>{}))),
        maybe(Parser<TypeBoundProcedurePart>{}),
        statement(Parser<EndTypeStmt>{})))

// R727 derived-type-stmt ->
//        TYPE [[, type-attr-spec-list] ::] type-name [(
//        type-param-name-list )]
constexpr auto listOfNames{nonemptyList("expected names"_err_en_US, name)};
TYPE_CONTEXT_PARSER("TYPE statement"_en_US,
    construct<DerivedTypeStmt>(
        "TYPE" >> optionalListBeforeColons(Parser<TypeAttrSpec>{}), name,
        defaulted(parenthesized(nonemptyList(name)))))

// R728 type-attr-spec ->
//        ABSTRACT | access-spec | BIND(C) | EXTENDS ( parent-type-name )
TYPE_PARSER(construct<TypeAttrSpec>(construct<Abstract>("ABSTRACT"_tok)) ||
    construct<TypeAttrSpec>(construct<TypeAttrSpec::BindC>("BIND ( C )"_tok)) ||
    construct<TypeAttrSpec>(
        construct<TypeAttrSpec::Extends>("EXTENDS" >> parenthesized(name))) ||
    construct<TypeAttrSpec>(accessSpec))

// R729 private-or-sequence -> private-components-stmt | sequence-stmt
TYPE_PARSER(construct<PrivateOrSequence>(Parser<PrivateStmt>{}) ||
    construct<PrivateOrSequence>(Parser<SequenceStmt>{}))

// R730 end-type-stmt -> END TYPE [type-name]
TYPE_PARSER(construct<EndTypeStmt>(
    recovery("END TYPE" >> maybe(name), endStmtErrorRecovery)))

// R731 sequence-stmt -> SEQUENCE
TYPE_PARSER(construct<SequenceStmt>("SEQUENCE"_tok))

// R732 type-param-def-stmt ->
//        integer-type-spec , type-param-attr-spec :: type-param-decl-list
// R734 type-param-attr-spec -> KIND | LEN
constexpr auto kindOrLen{"KIND" >> pure(common::TypeParamAttr::Kind) ||
    "LEN" >> pure(common::TypeParamAttr::Len)};
TYPE_PARSER(construct<TypeParamDefStmt>(integerTypeSpec / ",", kindOrLen,
    "::" >> nonemptyList("expected type parameter declarations"_err_en_US,
                Parser<TypeParamDecl>{})))

// R733 type-param-decl -> type-param-name [= scalar-int-constant-expr]
TYPE_PARSER(construct<TypeParamDecl>(name, maybe("=" >> scalarIntConstantExpr)))

// R736 component-def-stmt -> data-component-def-stmt |
//        proc-component-def-stmt
// Accidental extension not enabled here: PGI accepts type-param-def-stmt in
// component-part of derived-type-def.
TYPE_PARSER(recovery(
    withMessage("expected component definition"_err_en_US,
        first(construct<ComponentDefStmt>(Parser<DataComponentDefStmt>{}),
            construct<ComponentDefStmt>(Parser<ProcComponentDefStmt>{}))),
    construct<ComponentDefStmt>(inStmtErrorRecovery)))

// R737 data-component-def-stmt ->
//        declaration-type-spec [[, component-attr-spec-list] ::]
//        component-decl-list
TYPE_PARSER(construct<DataComponentDefStmt>(declarationTypeSpec,
    optionalListBeforeColons(Parser<ComponentAttrSpec>{}),
    nonemptyList(
        "expected component declarations"_err_en_US, Parser<ComponentDecl>{})))

// R738 component-attr-spec ->
//        access-spec | ALLOCATABLE |
//        CODIMENSION lbracket coarray-spec rbracket |
//        CONTIGUOUS | DIMENSION ( component-array-spec ) | POINTER
constexpr auto allocatable{construct<Allocatable>("ALLOCATABLE"_tok)};
constexpr auto contiguous{construct<Contiguous>("CONTIGUOUS"_tok)};
constexpr auto pointer{construct<Pointer>("POINTER"_tok)};
TYPE_PARSER(construct<ComponentAttrSpec>(accessSpec) ||
    construct<ComponentAttrSpec>(allocatable) ||
    construct<ComponentAttrSpec>("CODIMENSION" >> coarraySpec) ||
    construct<ComponentAttrSpec>(contiguous) ||
    construct<ComponentAttrSpec>("DIMENSION" >> Parser<ComponentArraySpec>{}) ||
    construct<ComponentAttrSpec>(pointer) ||
    construct<ComponentAttrSpec>(recovery(
        fail<ErrorRecovery>(
            "type parameter definitions must appear before component declarations"_err_en_US),
        kindOrLen >> construct<ErrorRecovery>())))

// R739 component-decl ->
//        component-name [( component-array-spec )]
//        [lbracket coarray-spec rbracket] [* char-length]
//        [component-initialization]
TYPE_CONTEXT_PARSER("component declaration"_en_US,
    construct<ComponentDecl>(name, maybe(Parser<ComponentArraySpec>{}),
        maybe(coarraySpec), maybe("*" >> charLength), maybe(initialization)))

// R740 component-array-spec ->
//        explicit-shape-spec-list | deferred-shape-spec-list
// N.B. Parenthesized here rather than around references to this production.
TYPE_PARSER(construct<ComponentArraySpec>(parenthesized(
                nonemptyList("expected explicit shape specifications"_err_en_US,
                    explicitShapeSpec))) ||
    construct<ComponentArraySpec>(parenthesized(deferredShapeSpecList)))

// R741 proc-component-def-stmt ->
//        PROCEDURE ( [proc-interface] ) , proc-component-attr-spec-list
//          :: proc-decl-list
TYPE_CONTEXT_PARSER("PROCEDURE component definition statement"_en_US,
    construct<ProcComponentDefStmt>(
        "PROCEDURE" >> parenthesized(maybe(procInterface)),
        localRecovery("expected PROCEDURE component attributes"_err_en_US,
            "," >> nonemptyList(Parser<ProcComponentAttrSpec>{}), ok),
        localRecovery("expected PROCEDURE declarations"_err_en_US,
            "::" >> nonemptyList(procDecl), SkipTo<'\n'>{})))

// R742 proc-component-attr-spec ->
//        access-spec | NOPASS | PASS [(arg-name)] | POINTER
constexpr auto noPass{construct<NoPass>("NOPASS"_tok)};
constexpr auto pass{construct<Pass>("PASS" >> maybe(parenthesized(name)))};
TYPE_PARSER(construct<ProcComponentAttrSpec>(accessSpec) ||
    construct<ProcComponentAttrSpec>(noPass) ||
    construct<ProcComponentAttrSpec>(pass) ||
    construct<ProcComponentAttrSpec>(pointer))

// R744 initial-data-target -> designator
constexpr auto initialDataTarget{indirect(designator)};

// R743 component-initialization ->
//        = constant-expr | => null-init | => initial-data-target
// R805 initialization ->
//        = constant-expr | => null-init | => initial-data-target
// Universal extension: initialization -> / data-stmt-value-list /
TYPE_PARSER(construct<Initialization>("=>" >> nullInit) ||
    construct<Initialization>("=>" >> initialDataTarget) ||
    construct<Initialization>("=" >> constantExpr) ||
    extension<LanguageFeature::SlashInitialization>(construct<Initialization>(
        "/" >> nonemptyList("expected values"_err_en_US,
                   indirect(Parser<DataStmtValue>{})) /
            "/")))

// R745 private-components-stmt -> PRIVATE
// R747 binding-private-stmt -> PRIVATE
TYPE_PARSER(construct<PrivateStmt>("PRIVATE"_tok))

// R746 type-bound-procedure-part ->
//        contains-stmt [binding-private-stmt] [type-bound-proc-binding]...
TYPE_CONTEXT_PARSER("type bound procedure part"_en_US,
    construct<TypeBoundProcedurePart>(statement(containsStmt),
        maybe(statement(Parser<PrivateStmt>{})),
        many(statement(Parser<TypeBoundProcBinding>{}))))

// R748 type-bound-proc-binding ->
//        type-bound-procedure-stmt | type-bound-generic-stmt |
//        final-procedure-stmt
TYPE_CONTEXT_PARSER("type bound procedure binding"_en_US,
    recovery(
        first(construct<TypeBoundProcBinding>(Parser<TypeBoundProcedureStmt>{}),
            construct<TypeBoundProcBinding>(Parser<TypeBoundGenericStmt>{}),
            construct<TypeBoundProcBinding>(Parser<FinalProcedureStmt>{})),
        construct<TypeBoundProcBinding>(
            !"END"_tok >> SkipTo<'\n'>{} >> construct<ErrorRecovery>())))

// R749 type-bound-procedure-stmt ->
//        PROCEDURE [[, bind-attr-list] ::] type-bound-proc-decl-list |
//        PROCEDURE ( interface-name ) , bind-attr-list :: binding-name-list
TYPE_CONTEXT_PARSER("type bound PROCEDURE statement"_en_US,
    "PROCEDURE" >>
        (construct<TypeBoundProcedureStmt>(
             construct<TypeBoundProcedureStmt::WithInterface>(
                 parenthesized(name),
                 localRecovery("expected list of binding attributes"_err_en_US,
                     "," >> nonemptyList(Parser<BindAttr>{}), ok),
                 localRecovery("expected list of binding names"_err_en_US,
                     "::" >> listOfNames, SkipTo<'\n'>{}))) ||
            construct<TypeBoundProcedureStmt>(
                construct<TypeBoundProcedureStmt::WithoutInterface>(
                    optionalListBeforeColons(Parser<BindAttr>{}),
                    nonemptyList(
                        "expected type bound procedure declarations"_err_en_US,
                        Parser<TypeBoundProcDecl>{})))))

// R750 type-bound-proc-decl -> binding-name [=> procedure-name]
TYPE_PARSER(construct<TypeBoundProcDecl>(name, maybe("=>" >> name)))

// R751 type-bound-generic-stmt ->
//        GENERIC [, access-spec] :: generic-spec => binding-name-list
TYPE_CONTEXT_PARSER("type bound GENERIC statement"_en_US,
    construct<TypeBoundGenericStmt>("GENERIC" >> maybe("," >> accessSpec),
        "::" >> indirect(genericSpec), "=>" >> listOfNames))

// R752 bind-attr ->
//        access-spec | DEFERRED | NON_OVERRIDABLE | NOPASS | PASS [(arg-name)]
TYPE_PARSER(construct<BindAttr>(accessSpec) ||
    construct<BindAttr>(construct<BindAttr::Deferred>("DEFERRED"_tok)) ||
    construct<BindAttr>(
        construct<BindAttr::Non_Overridable>("NON_OVERRIDABLE"_tok)) ||
    construct<BindAttr>(noPass) || construct<BindAttr>(pass))

// R753 final-procedure-stmt -> FINAL [::] final-subroutine-name-list
TYPE_CONTEXT_PARSER("FINAL statement"_en_US,
    construct<FinalProcedureStmt>("FINAL" >> maybe("::"_tok) >> listOfNames))

// R754 derived-type-spec -> type-name [(type-param-spec-list)]
TYPE_PARSER(construct<DerivedTypeSpec>(name,
    defaulted(parenthesized(nonemptyList(
        "expected type parameters"_err_en_US, Parser<TypeParamSpec>{})))))

// R755 type-param-spec -> [keyword =] type-param-value
TYPE_PARSER(construct<TypeParamSpec>(maybe(keyword / "="), typeParamValue))

// R756 structure-constructor -> derived-type-spec ( [component-spec-list] )
TYPE_PARSER((construct<StructureConstructor>(derivedTypeSpec,
                 parenthesized(optionalList(Parser<ComponentSpec>{}))) ||
                // This alternative corrects misrecognition of the
                // component-spec-list as the type-param-spec-list in
                // derived-type-spec.
                construct<StructureConstructor>(
                    construct<DerivedTypeSpec>(
                        name, construct<std::list<TypeParamSpec>>()),
                    parenthesized(optionalList(Parser<ComponentSpec>{})))) /
    !"("_tok)

// R757 component-spec -> [keyword =] component-data-source
TYPE_PARSER(construct<ComponentSpec>(
    maybe(keyword / "="), Parser<ComponentDataSource>{}))

// R758 component-data-source -> expr | data-target | proc-target
TYPE_PARSER(construct<ComponentDataSource>(indirect(expr)))

// R759 enum-def ->
//        enum-def-stmt enumerator-def-stmt [enumerator-def-stmt]...
//        end-enum-stmt
TYPE_CONTEXT_PARSER("enum definition"_en_US,
    construct<EnumDef>(statement(Parser<EnumDefStmt>{}),
        some(unambiguousStatement(Parser<EnumeratorDefStmt>{})),
        statement(Parser<EndEnumStmt>{})))

// R760 enum-def-stmt -> ENUM, BIND(C)
TYPE_PARSER(construct<EnumDefStmt>("ENUM , BIND ( C )"_tok))

// R761 enumerator-def-stmt -> ENUMERATOR [::] enumerator-list
TYPE_CONTEXT_PARSER("ENUMERATOR statement"_en_US,
    construct<EnumeratorDefStmt>("ENUMERATOR" >> maybe("::"_tok) >>
        nonemptyList("expected enumerators"_err_en_US, Parser<Enumerator>{})))

// R762 enumerator -> named-constant [= scalar-int-constant-expr]
TYPE_PARSER(
    construct<Enumerator>(namedConstant, maybe("=" >> scalarIntConstantExpr)))

// R763 end-enum-stmt -> END ENUM
TYPE_PARSER(recovery("END ENUM"_tok, "END" >> SkipPast<'\n'>{}) >>
    construct<EndEnumStmt>())

// R764 boz-literal-constant -> binary-constant | octal-constant | hex-constant
// R765 binary-constant -> B ' digit [digit]... ' | B " digit [digit]... "
// R766 octal-constant -> O ' digit [digit]... ' | O " digit [digit]... "
// R767 hex-constant ->
//        Z ' hex-digit [hex-digit]... ' | Z " hex-digit [hex-digit]... "
// extension: X accepted for Z
// extension: BOZX suffix accepted
TYPE_PARSER(construct<BOZLiteralConstant>(BOZLiteral{}))

// R1124 do-variable -> scalar-int-variable-name
constexpr auto doVariable{scalar(integer(name))};

// NOTE: In loop-control we allow REAL name and bounds too.
// This means parse them without the integer constraint and check later.

inline constexpr auto loopBounds(decltype(scalarExpr) &p) {
  return construct<LoopBounds<ScalarName, ScalarExpr>>(
      scalar(name) / "=", p / ",", p, maybe("," >> p));
}
template<typename PA> inline constexpr auto loopBounds(const PA &p) {
  return construct<LoopBounds<DoVariable, typename PA::resultType>>(
      doVariable / "=", p / ",", p, maybe("," >> p));
}

// R769 array-constructor -> (/ ac-spec /) | lbracket ac-spec rbracket
TYPE_CONTEXT_PARSER("array constructor"_en_US,
    construct<ArrayConstructor>(
        "(/" >> Parser<AcSpec>{} / "/)" || bracketed(Parser<AcSpec>{})))

// R770 ac-spec -> type-spec :: | [type-spec ::] ac-value-list
TYPE_PARSER(construct<AcSpec>(maybe(typeSpec / "::"),
                nonemptyList("expected array constructor values"_err_en_US,
                    Parser<AcValue>{})) ||
    construct<AcSpec>(typeSpec / "::"))

// R773 ac-value -> expr | ac-implied-do
TYPE_PARSER(
    // PGI/Intel extension: accept triplets in array constructors
    extension<LanguageFeature::TripletInArrayConstructor>(
        construct<AcValue>(construct<AcValue::Triplet>(scalarIntExpr,
            ":" >> scalarIntExpr, maybe(":" >> scalarIntExpr)))) ||
    construct<AcValue>(indirect(expr)) ||
    construct<AcValue>(indirect(Parser<AcImpliedDo>{})))

// R774 ac-implied-do -> ( ac-value-list , ac-implied-do-control )
TYPE_PARSER(parenthesized(
    construct<AcImpliedDo>(nonemptyList(Parser<AcValue>{} / lookAhead(","_tok)),
        "," >> Parser<AcImpliedDoControl>{})))

// R775 ac-implied-do-control ->
//        [integer-type-spec ::] ac-do-variable = scalar-int-expr ,
//        scalar-int-expr [, scalar-int-expr]
// R776 ac-do-variable -> do-variable
TYPE_PARSER(construct<AcImpliedDoControl>(
    maybe(integerTypeSpec / "::"), loopBounds(scalarIntExpr)))

// R801 type-declaration-stmt ->
//        declaration-type-spec [[, attr-spec]... ::] entity-decl-list
TYPE_PARSER(
    construct<TypeDeclarationStmt>(declarationTypeSpec,
        optionalListBeforeColons(Parser<AttrSpec>{}),
        nonemptyList("expected entity declarations"_err_en_US, entityDecl)) ||
    // PGI-only extension: don't require the colons
    // N.B.: The standard requires the colons if the entity
    // declarations contain initializers.
    extension<LanguageFeature::MissingColons>(construct<TypeDeclarationStmt>(
        declarationTypeSpec, defaulted("," >> nonemptyList(Parser<AttrSpec>{})),
        withMessage("expected entity declarations"_err_en_US,
            "," >> nonemptyList(entityDecl)))))

// R802 attr-spec ->
//        access-spec | ALLOCATABLE | ASYNCHRONOUS |
//        CODIMENSION lbracket coarray-spec rbracket | CONTIGUOUS |
//        DIMENSION ( array-spec ) | EXTERNAL | INTENT ( intent-spec ) |
//        INTRINSIC | language-binding-spec | OPTIONAL | PARAMETER | POINTER |
//        PROTECTED | SAVE | TARGET | VALUE | VOLATILE
constexpr auto optional{construct<Optional>("OPTIONAL"_tok)};
constexpr auto protectedAttr{construct<Protected>("PROTECTED"_tok)};
constexpr auto save{construct<Save>("SAVE"_tok)};
TYPE_PARSER(construct<AttrSpec>(accessSpec) ||
    construct<AttrSpec>(allocatable) ||
    construct<AttrSpec>(construct<Asynchronous>("ASYNCHRONOUS"_tok)) ||
    construct<AttrSpec>("CODIMENSION" >> coarraySpec) ||
    construct<AttrSpec>(contiguous) ||
    construct<AttrSpec>("DIMENSION" >> arraySpec) ||
    construct<AttrSpec>(construct<External>("EXTERNAL"_tok)) ||
    construct<AttrSpec>("INTENT" >> parenthesized(intentSpec)) ||
    construct<AttrSpec>(construct<Intrinsic>("INTRINSIC"_tok)) ||
    construct<AttrSpec>(languageBindingSpec) || construct<AttrSpec>(optional) ||
    construct<AttrSpec>(construct<Parameter>("PARAMETER"_tok)) ||
    construct<AttrSpec>(pointer) || construct<AttrSpec>(protectedAttr) ||
    construct<AttrSpec>(save) ||
    construct<AttrSpec>(construct<Target>("TARGET"_tok)) ||
    construct<AttrSpec>(construct<Value>("VALUE"_tok)) ||
    construct<AttrSpec>(construct<Volatile>("VOLATILE"_tok)))

// R804 object-name -> name
constexpr auto objectName{name};

// R803 entity-decl ->
//        object-name [( array-spec )] [lbracket coarray-spec rbracket]
//          [* char-length] [initialization] |
//        function-name [* char-length]
TYPE_PARSER(construct<EntityDecl>(objectName, maybe(arraySpec),
    maybe(coarraySpec), maybe("*" >> charLength), maybe(initialization)))

// R806 null-init -> function-reference
// TODO: confirm in semantics that NULL still intrinsic in this scope
TYPE_PARSER(construct<NullInit>("NULL ( )"_tok) / !"("_tok)

// R807 access-spec -> PUBLIC | PRIVATE
TYPE_PARSER(construct<AccessSpec>("PUBLIC" >> pure(AccessSpec::Kind::Public)) ||
    construct<AccessSpec>("PRIVATE" >> pure(AccessSpec::Kind::Private)))

// R808 language-binding-spec ->
//        BIND ( C [, NAME = scalar-default-char-constant-expr] )
// R1528 proc-language-binding-spec -> language-binding-spec
TYPE_PARSER(construct<LanguageBindingSpec>(
    "BIND ( C" >> maybe(", NAME =" >> scalarDefaultCharConstantExpr) / ")"))

// R809 coarray-spec -> deferred-coshape-spec-list | explicit-coshape-spec
// N.B. Bracketed here rather than around references, for consistency with
// array-spec.
TYPE_PARSER(
    construct<CoarraySpec>(bracketed(Parser<DeferredCoshapeSpecList>{})) ||
    construct<CoarraySpec>(bracketed(Parser<ExplicitCoshapeSpec>{})))

// R810 deferred-coshape-spec -> :
// deferred-coshape-spec-list - just a list of colons
inline int listLength(std::list<Success> &&xs) { return xs.size(); }

TYPE_PARSER(construct<DeferredCoshapeSpecList>(
    applyFunction(listLength, nonemptyList(":"_tok))))

// R811 explicit-coshape-spec ->
//        [[lower-cobound :] upper-cobound ,]... [lower-cobound :] *
// R812 lower-cobound -> specification-expr
// R813 upper-cobound -> specification-expr
TYPE_PARSER(construct<ExplicitCoshapeSpec>(
    many(explicitShapeSpec / ","), maybe(specificationExpr / ":") / "*"))

// R815 array-spec ->
//        explicit-shape-spec-list | assumed-shape-spec-list |
//        deferred-shape-spec-list | assumed-size-spec | implied-shape-spec |
//        implied-shape-or-assumed-size-spec | assumed-rank-spec
// N.B. Parenthesized here rather than around references to avoid
// a need for forced look-ahead.
// Shape specs that could be deferred-shape-spec or assumed-shape-spec
// (e.g. '(:,:)') are parsed as the former.
TYPE_PARSER(
    construct<ArraySpec>(parenthesized(nonemptyList(explicitShapeSpec))) ||
    construct<ArraySpec>(parenthesized(deferredShapeSpecList)) ||
    construct<ArraySpec>(
        parenthesized(nonemptyList(Parser<AssumedShapeSpec>{}))) ||
    construct<ArraySpec>(parenthesized(Parser<AssumedSizeSpec>{})) ||
    construct<ArraySpec>(parenthesized(Parser<ImpliedShapeSpec>{})) ||
    construct<ArraySpec>(parenthesized(Parser<AssumedRankSpec>{})))

// R816 explicit-shape-spec -> [lower-bound :] upper-bound
// R817 lower-bound -> specification-expr
// R818 upper-bound -> specification-expr
TYPE_PARSER(construct<ExplicitShapeSpec>(
    maybe(specificationExpr / ":"), specificationExpr))

// R819 assumed-shape-spec -> [lower-bound] :
TYPE_PARSER(construct<AssumedShapeSpec>(maybe(specificationExpr) / ":"))

// R820 deferred-shape-spec -> :
// deferred-shape-spec-list - just a list of colons
TYPE_PARSER(construct<DeferredShapeSpecList>(
    applyFunction(listLength, nonemptyList(":"_tok))))

// R821 assumed-implied-spec -> [lower-bound :] *
TYPE_PARSER(construct<AssumedImpliedSpec>(maybe(specificationExpr / ":") / "*"))

// R822 assumed-size-spec -> explicit-shape-spec-list , assumed-implied-spec
TYPE_PARSER(construct<AssumedSizeSpec>(
    nonemptyList(explicitShapeSpec) / ",", assumedImpliedSpec))

// R823 implied-shape-or-assumed-size-spec -> assumed-implied-spec
// R824 implied-shape-spec -> assumed-implied-spec , assumed-implied-spec-list
// I.e., when the assumed-implied-spec-list has a single item, it constitutes an
// implied-shape-or-assumed-size-spec; otherwise, an implied-shape-spec.
TYPE_PARSER(construct<ImpliedShapeSpec>(nonemptyList(assumedImpliedSpec)))

// R825 assumed-rank-spec -> ..
TYPE_PARSER(construct<AssumedRankSpec>(".."_tok))

// R826 intent-spec -> IN | OUT | INOUT
TYPE_PARSER(construct<IntentSpec>("IN OUT" >> pure(IntentSpec::Intent::InOut) ||
    "IN" >> pure(IntentSpec::Intent::In) ||
    "OUT" >> pure(IntentSpec::Intent::Out)))

// R827 access-stmt -> access-spec [[::] access-id-list]
TYPE_PARSER(construct<AccessStmt>(accessSpec,
    defaulted(maybe("::"_tok) >>
        nonemptyList("expected names and generic specifications"_err_en_US,
            Parser<AccessId>{}))))

// R828 access-id -> access-name | generic-spec
TYPE_PARSER(construct<AccessId>(indirect(genericSpec)) ||
    construct<AccessId>(name))  // initially ambiguous with genericSpec

// R829 allocatable-stmt -> ALLOCATABLE [::] allocatable-decl-list
TYPE_PARSER(construct<AllocatableStmt>("ALLOCATABLE" >> maybe("::"_tok) >>
    nonemptyList(
        "expected object declarations"_err_en_US, Parser<ObjectDecl>{})))

// R830 allocatable-decl ->
//        object-name [( array-spec )] [lbracket coarray-spec rbracket]
// R860 target-decl ->
//        object-name [( array-spec )] [lbracket coarray-spec rbracket]
TYPE_PARSER(
    construct<ObjectDecl>(objectName, maybe(arraySpec), maybe(coarraySpec)))

// R831 asynchronous-stmt -> ASYNCHRONOUS [::] object-name-list
TYPE_PARSER(construct<AsynchronousStmt>("ASYNCHRONOUS" >> maybe("::"_tok) >>
    nonemptyList("expected object names"_err_en_US, objectName)))

// R832 bind-stmt -> language-binding-spec [::] bind-entity-list
TYPE_PARSER(construct<BindStmt>(languageBindingSpec / maybe("::"_tok),
    nonemptyList("expected bind entities"_err_en_US, Parser<BindEntity>{})))

// R833 bind-entity -> entity-name | / common-block-name /
TYPE_PARSER(construct<BindEntity>(pure(BindEntity::Kind::Object), name) ||
    construct<BindEntity>("/" >> pure(BindEntity::Kind::Common), name / "/"))

// R834 codimension-stmt -> CODIMENSION [::] codimension-decl-list
TYPE_PARSER(construct<CodimensionStmt>("CODIMENSION" >> maybe("::"_tok) >>
    nonemptyList("expected codimension declarations"_err_en_US,
        Parser<CodimensionDecl>{})))

// R835 codimension-decl -> coarray-name lbracket coarray-spec rbracket
TYPE_PARSER(construct<CodimensionDecl>(name, coarraySpec))

// R836 contiguous-stmt -> CONTIGUOUS [::] object-name-list
TYPE_PARSER(construct<ContiguousStmt>("CONTIGUOUS" >> maybe("::"_tok) >>
    nonemptyList("expected object names"_err_en_US, objectName)))

// R837 data-stmt -> DATA data-stmt-set [[,] data-stmt-set]...
TYPE_CONTEXT_PARSER("DATA statement"_en_US,
    construct<DataStmt>(
        "DATA" >> nonemptySeparated(Parser<DataStmtSet>{}, maybe(","_tok))))

// R838 data-stmt-set -> data-stmt-object-list / data-stmt-value-list /
TYPE_PARSER(construct<DataStmtSet>(
    nonemptyList(
        "expected DATA statement objects"_err_en_US, Parser<DataStmtObject>{}),
    withMessage("expected DATA statement value list"_err_en_US,
        "/"_tok >> nonemptyList("expected DATA statement values"_err_en_US,
                       Parser<DataStmtValue>{})) /
        "/"))

// R839 data-stmt-object -> variable | data-implied-do
TYPE_PARSER(construct<DataStmtObject>(indirect(variable)) ||
    construct<DataStmtObject>(dataImpliedDo))

// R840 data-implied-do ->
//        ( data-i-do-object-list , [integer-type-spec ::] data-i-do-variable
//        = scalar-int-constant-expr , scalar-int-constant-expr
//        [, scalar-int-constant-expr] )
// R842 data-i-do-variable -> do-variable
TYPE_PARSER(parenthesized(construct<DataImpliedDo>(
    nonemptyList(Parser<DataIDoObject>{} / lookAhead(","_tok)) / ",",
    maybe(integerTypeSpec / "::"), loopBounds(scalarIntConstantExpr))))

// R841 data-i-do-object ->
//        array-element | scalar-structure-component | data-implied-do
TYPE_PARSER(construct<DataIDoObject>(scalar(indirect(designator))) ||
    construct<DataIDoObject>(indirect(dataImpliedDo)))

// R843 data-stmt-value -> [data-stmt-repeat *] data-stmt-constant
TYPE_PARSER(construct<DataStmtValue>(
    maybe(Parser<DataStmtRepeat>{} / "*"), Parser<DataStmtConstant>{}))

// R847 constant-subobject -> designator
// R846 int-constant-subobject -> constant-subobject
constexpr auto constantSubobject{constant(indirect(designator))};

// R844 data-stmt-repeat -> scalar-int-constant | scalar-int-constant-subobject
// R607 int-constant -> constant
// Factored into:
//   constant -> literal-constant -> int-literal-constant   and
//   constant -> named-constant
TYPE_PARSER(construct<DataStmtRepeat>(intLiteralConstant) ||
    construct<DataStmtRepeat>(scalar(integer(constantSubobject))) ||
    construct<DataStmtRepeat>(scalar(integer(namedConstant))))

// R845 data-stmt-constant ->
//        scalar-constant | scalar-constant-subobject |
//        signed-int-literal-constant | signed-real-literal-constant |
//        null-init | initial-data-target | structure-constructor
// TODO: Some structure constructors can be misrecognized as array
// references into constant subobjects.
TYPE_PARSER(first(construct<DataStmtConstant>(scalar(Parser<ConstantValue>{})),
    construct<DataStmtConstant>(nullInit),
    construct<DataStmtConstant>(Parser<StructureConstructor>{}),
    construct<DataStmtConstant>(scalar(constantSubobject)),
    construct<DataStmtConstant>(signedRealLiteralConstant),
    construct<DataStmtConstant>(signedIntLiteralConstant),
    extension<LanguageFeature::SignedComplexLiteral>(
        construct<DataStmtConstant>(Parser<SignedComplexLiteralConstant>{})),
    construct<DataStmtConstant>(initialDataTarget)))

// R848 dimension-stmt ->
//        DIMENSION [::] array-name ( array-spec )
//        [, array-name ( array-spec )]...
TYPE_CONTEXT_PARSER("DIMENSION statement"_en_US,
    construct<DimensionStmt>("DIMENSION" >> maybe("::"_tok) >>
        nonemptyList("expected array specifications"_err_en_US,
            construct<DimensionStmt::Declaration>(name, arraySpec))))

// R849 intent-stmt -> INTENT ( intent-spec ) [::] dummy-arg-name-list
TYPE_CONTEXT_PARSER("INTENT statement"_en_US,
    construct<IntentStmt>(
        "INTENT" >> parenthesized(intentSpec) / maybe("::"_tok), listOfNames))

// R850 optional-stmt -> OPTIONAL [::] dummy-arg-name-list
TYPE_PARSER(
    construct<OptionalStmt>("OPTIONAL" >> maybe("::"_tok) >> listOfNames))

// R851 parameter-stmt -> PARAMETER ( named-constant-def-list )
// Legacy extension: omitted parentheses, no implicit typing from names
TYPE_CONTEXT_PARSER("PARAMETER statement"_en_US,
    construct<ParameterStmt>(
        "PARAMETER" >> parenthesized(nonemptyList(Parser<NamedConstantDef>{}))))
TYPE_CONTEXT_PARSER("old style PARAMETER statement"_en_US,
    extension<LanguageFeature::OldStyleParameter>(construct<OldParameterStmt>(
        "PARAMETER" >> nonemptyList(Parser<NamedConstantDef>{}))))

// R852 named-constant-def -> named-constant = constant-expr
TYPE_PARSER(construct<NamedConstantDef>(namedConstant, "=" >> constantExpr))

// R853 pointer-stmt -> POINTER [::] pointer-decl-list
TYPE_PARSER(construct<PointerStmt>("POINTER" >> maybe("::"_tok) >>
    nonemptyList(
        "expected pointer declarations"_err_en_US, Parser<PointerDecl>{})))

// R854 pointer-decl ->
//        object-name [( deferred-shape-spec-list )] | proc-entity-name
TYPE_PARSER(
    construct<PointerDecl>(name, maybe(parenthesized(deferredShapeSpecList))))

// R855 protected-stmt -> PROTECTED [::] entity-name-list
TYPE_PARSER(
    construct<ProtectedStmt>("PROTECTED" >> maybe("::"_tok) >> listOfNames))

// R856 save-stmt -> SAVE [[::] saved-entity-list]
TYPE_PARSER(construct<SaveStmt>(
    "SAVE" >> defaulted(maybe("::"_tok) >>
                  nonemptyList("expected SAVE entities"_err_en_US,
                      Parser<SavedEntity>{}))))

// R857 saved-entity -> object-name | proc-pointer-name | / common-block-name /
// R858 proc-pointer-name -> name
TYPE_PARSER(construct<SavedEntity>(pure(SavedEntity::Kind::Entity), name) ||
    construct<SavedEntity>("/" >> pure(SavedEntity::Kind::Common), name / "/"))

// R859 target-stmt -> TARGET [::] target-decl-list
TYPE_PARSER(construct<TargetStmt>("TARGET" >> maybe("::"_tok) >>
    nonemptyList("expected objects"_err_en_US, Parser<ObjectDecl>{})))

// R861 value-stmt -> VALUE [::] dummy-arg-name-list
TYPE_PARSER(construct<ValueStmt>("VALUE" >> maybe("::"_tok) >> listOfNames))

// R862 volatile-stmt -> VOLATILE [::] object-name-list
TYPE_PARSER(construct<VolatileStmt>("VOLATILE" >> maybe("::"_tok) >>
    nonemptyList("expected object names"_err_en_US, objectName)))

// R866 implicit-name-spec -> EXTERNAL | TYPE
constexpr auto implicitNameSpec{
    "EXTERNAL" >> pure(ImplicitStmt::ImplicitNoneNameSpec::External) ||
    "TYPE" >> pure(ImplicitStmt::ImplicitNoneNameSpec::Type)};

// R863 implicit-stmt ->
//        IMPLICIT implicit-spec-list |
//        IMPLICIT NONE [( [implicit-name-spec-list] )]
TYPE_CONTEXT_PARSER("IMPLICIT statement"_en_US,
    construct<ImplicitStmt>(
        "IMPLICIT" >> nonemptyList("expected IMPLICIT specifications"_err_en_US,
                          Parser<ImplicitSpec>{})) ||
        construct<ImplicitStmt>("IMPLICIT NONE"_sptok >>
            defaulted(parenthesized(optionalList(implicitNameSpec)))))

// R864 implicit-spec -> declaration-type-spec ( letter-spec-list )
// The variant form of declarationTypeSpec is meant to avoid misrecognition
// of a letter-spec as a simple parenthesized expression for kind or character
// length, e.g., PARAMETER(I=5,N=1); IMPLICIT REAL(I-N)(O-Z) vs.
// IMPLICIT REAL(I-N).  The variant form needs to attempt to reparse only
// types with optional parenthesized kind/length expressions, so derived
// type specs, DOUBLE PRECISION, and DOUBLE COMPLEX need not be considered.
constexpr auto noKindSelector{construct<std::optional<KindSelector>>()};
constexpr auto implicitSpecDeclarationTypeSpecRetry{
    construct<DeclarationTypeSpec>(first(
        construct<IntrinsicTypeSpec>(
            construct<IntegerTypeSpec>("INTEGER" >> noKindSelector)),
        construct<IntrinsicTypeSpec>(
            construct<IntrinsicTypeSpec::Real>("REAL" >> noKindSelector)),
        construct<IntrinsicTypeSpec>(
            construct<IntrinsicTypeSpec::Complex>("COMPLEX" >> noKindSelector)),
        construct<IntrinsicTypeSpec>(construct<IntrinsicTypeSpec::Character>(
            "CHARACTER" >> construct<std::optional<CharSelector>>())),
        construct<IntrinsicTypeSpec>(construct<IntrinsicTypeSpec::Logical>(
            "LOGICAL" >> noKindSelector))))};

TYPE_PARSER(construct<ImplicitSpec>(declarationTypeSpec,
                parenthesized(nonemptyList(Parser<LetterSpec>{}))) ||
    construct<ImplicitSpec>(implicitSpecDeclarationTypeSpecRetry,
        parenthesized(nonemptyList(Parser<LetterSpec>{}))))

// R865 letter-spec -> letter [- letter]
TYPE_PARSER(space >> (construct<LetterSpec>(letter, maybe("-" >> letter)) ||
                         construct<LetterSpec>(otherIdChar,
                             construct<std::optional<const char *>>())))

// R867 import-stmt ->
//        IMPORT [[::] import-name-list] |
//        IMPORT , ONLY : import-name-list | IMPORT , NONE | IMPORT , ALL
TYPE_CONTEXT_PARSER("IMPORT statement"_en_US,
    construct<ImportStmt>(
        "IMPORT , ONLY :" >> pure(common::ImportKind::Only), listOfNames) ||
        construct<ImportStmt>(
            "IMPORT , NONE" >> pure(common::ImportKind::None)) ||
        construct<ImportStmt>(
            "IMPORT , ALL" >> pure(common::ImportKind::All)) ||
        construct<ImportStmt>(
            "IMPORT" >> maybe("::"_tok) >> optionalList(name)))

// R868 namelist-stmt ->
//        NAMELIST / namelist-group-name / namelist-group-object-list
//        [[,] / namelist-group-name / namelist-group-object-list]...
// R869 namelist-group-object -> variable-name
TYPE_PARSER(construct<NamelistStmt>("NAMELIST" >>
    nonemptySeparated(
        construct<NamelistStmt::Group>("/" >> name / "/", listOfNames),
        maybe(","_tok))))

// R870 equivalence-stmt -> EQUIVALENCE equivalence-set-list
// R871 equivalence-set -> ( equivalence-object , equivalence-object-list )
TYPE_PARSER(construct<EquivalenceStmt>("EQUIVALENCE" >>
    nonemptyList(
        parenthesized(nonemptyList("expected EQUIVALENCE objects"_err_en_US,
            Parser<EquivalenceObject>{})))))

// R872 equivalence-object -> variable-name | array-element | substring
TYPE_PARSER(construct<EquivalenceObject>(indirect(designator)))

// R873 common-stmt ->
//        COMMON [/ [common-block-name] /] common-block-object-list
//        [[,] / [common-block-name] / common-block-object-list]...
TYPE_PARSER(
    construct<CommonStmt>("COMMON" >> defaulted("/" >> maybe(name) / "/"),
        nonemptyList("expected COMMON block objects"_err_en_US,
            Parser<CommonBlockObject>{}),
        many(maybe(","_tok) >>
            construct<CommonStmt::Block>("/" >> maybe(name) / "/",
                nonemptyList("expected COMMON block objects"_err_en_US,
                    Parser<CommonBlockObject>{})))))

// R874 common-block-object -> variable-name [( array-spec )]
TYPE_PARSER(construct<CommonBlockObject>(name, maybe(arraySpec)))

// R901 designator -> object-name | array-element | array-section |
//                    coindexed-named-object | complex-part-designator |
//                    structure-component | substring
// The Standard's productions for designator and its alternatives are
// ambiguous without recourse to a symbol table.  Many of the alternatives
// for designator (viz., array-element, coindexed-named-object,
// and structure-component) are all syntactically just data-ref.
// What designator boils down to is this:
//  It starts with either a name or a character literal.
//  If it starts with a character literal, it must be a substring.
//  If it starts with a name, it's a sequence of %-separated parts;
//  each part is a name, maybe a (section-subscript-list), and
//  maybe an [image-selector].
//  If it's a substring, it ends with (substring-range).
TYPE_CONTEXT_PARSER("designator"_en_US,
    sourced(construct<Designator>(substring) || construct<Designator>(dataRef)))

constexpr auto percentOrDot{"%"_tok ||
    // legacy VAX extension for RECORD field access
    extension<LanguageFeature::DECStructures>(
        "."_tok / lookAhead(OldStructureComponentName{}))};

// R902 variable -> designator | function-reference
// This production appears to be left-recursive in the grammar via
//   function-reference ->  procedure-designator -> proc-component-ref ->
//     scalar-variable
// and would be so if we were to allow functions to be called via procedure
// pointer components within derived type results of other function references
// (a reasonable extension, esp. in the case of procedure pointer components
// that are NOPASS).  However, Fortran constrains the use of a variable in a
// proc-component-ref to be a data-ref without coindices (C1027).
// Some array element references will be misrecognized as function references.
constexpr auto noMoreAddressing{!"("_tok >> !"["_tok >> !percentOrDot};
TYPE_CONTEXT_PARSER("variable"_en_US,
    construct<Variable>(indirect(functionReference / noMoreAddressing)) ||
        construct<Variable>(indirect(designator)))

// R904 logical-variable -> variable
// Appears only as part of scalar-logical-variable.
constexpr auto scalarLogicalVariable{scalar(logical(variable))};

// R906 default-char-variable -> variable
// Appears only as part of scalar-default-char-variable.
constexpr auto scalarDefaultCharVariable{scalar(defaultChar(variable))};

// R907 int-variable -> variable
// Appears only as part of scalar-int-variable.
constexpr auto scalarIntVariable{scalar(integer(variable))};

// R908 substring -> parent-string ( substring-range )
// R909 parent-string ->
//        scalar-variable-name | array-element | coindexed-named-object |
//        scalar-structure-component | scalar-char-literal-constant |
//        scalar-named-constant
TYPE_PARSER(
    construct<Substring>(dataRef, parenthesized(Parser<SubstringRange>{})))

TYPE_PARSER(construct<CharLiteralConstantSubstring>(
    charLiteralConstant, parenthesized(Parser<SubstringRange>{})))

// R910 substring-range -> [scalar-int-expr] : [scalar-int-expr]
TYPE_PARSER(construct<SubstringRange>(
    maybe(scalarIntExpr), ":" >> maybe(scalarIntExpr)))

// R911 data-ref -> part-ref [% part-ref]...
// R914 coindexed-named-object -> data-ref
// R917 array-element -> data-ref
TYPE_PARSER(
    construct<DataRef>(nonemptySeparated(Parser<PartRef>{}, percentOrDot)))

// R912 part-ref -> part-name [( section-subscript-list )] [image-selector]
TYPE_PARSER(construct<PartRef>(name,
    defaulted(
        parenthesized(nonemptyList(Parser<SectionSubscript>{})) / !"=>"_tok),
    maybe(Parser<ImageSelector>{})))

// R913 structure-component -> data-ref
TYPE_PARSER(construct<StructureComponent>(
    construct<DataRef>(some(Parser<PartRef>{} / percentOrDot)), name))

// R919 subscript -> scalar-int-expr
constexpr auto subscript{scalarIntExpr};

// R920 section-subscript -> subscript | subscript-triplet | vector-subscript
// R923 vector-subscript -> int-expr
// N.B. The distinction that needs to be made between "subscript" and
// "vector-subscript" is deferred to semantic analysis.
TYPE_PARSER(construct<SectionSubscript>(Parser<SubscriptTriplet>{}) ||
    construct<SectionSubscript>(intExpr))

// R921 subscript-triplet -> [subscript] : [subscript] [: stride]
TYPE_PARSER(construct<SubscriptTriplet>(
    maybe(subscript), ":" >> maybe(subscript), maybe(":" >> subscript)))

// R925 cosubscript -> scalar-int-expr
constexpr auto cosubscript{scalarIntExpr};

// R924 image-selector ->
//        lbracket cosubscript-list [, image-selector-spec-list] rbracket
TYPE_CONTEXT_PARSER("image selector"_en_US,
    construct<ImageSelector>("[" >> nonemptyList(cosubscript / !"="_tok),
        defaulted("," >> nonemptyList(Parser<ImageSelectorSpec>{})) / "]"))

// R1115 team-value -> scalar-expr
constexpr auto teamValue{scalar(indirect(expr))};

// R926 image-selector-spec ->
//        STAT = stat-variable | TEAM = team-value |
//        TEAM_NUMBER = scalar-int-expr
TYPE_PARSER(construct<ImageSelectorSpec>(construct<ImageSelectorSpec::Stat>(
                "STAT =" >> scalar(integer(indirect(variable))))) ||
    construct<ImageSelectorSpec>(construct<TeamValue>("TEAM =" >> teamValue)) ||
    construct<ImageSelectorSpec>(construct<ImageSelectorSpec::Team_Number>(
        "TEAM_NUMBER =" >> scalarIntExpr)))

// R927 allocate-stmt ->
//        ALLOCATE ( [type-spec ::] allocation-list [, alloc-opt-list] )
TYPE_CONTEXT_PARSER("ALLOCATE statement"_en_US,
    construct<AllocateStmt>("ALLOCATE (" >> maybe(typeSpec / "::"),
        nonemptyList(Parser<Allocation>{}),
        defaulted("," >> nonemptyList(Parser<AllocOpt>{})) / ")"))

// R928 alloc-opt ->
//        ERRMSG = errmsg-variable | MOLD = source-expr |
//        SOURCE = source-expr | STAT = stat-variable
// R931 source-expr -> expr
TYPE_PARSER(construct<AllocOpt>(
                construct<AllocOpt::Mold>("MOLD =" >> indirect(expr))) ||
    construct<AllocOpt>(
        construct<AllocOpt::Source>("SOURCE =" >> indirect(expr))) ||
    construct<AllocOpt>(statOrErrmsg))

// R929 stat-variable -> scalar-int-variable
TYPE_PARSER(construct<StatVariable>(scalar(integer(variable))))

// R930 errmsg-variable -> scalar-default-char-variable
// R1207 iomsg-variable -> scalar-default-char-variable
constexpr auto msgVariable{construct<MsgVariable>(scalarDefaultCharVariable)};

// R932 allocation ->
//        allocate-object [( allocate-shape-spec-list )]
//        [lbracket allocate-coarray-spec rbracket]
// TODO: allocate-shape-spec-list might be misrecognized as
// the final list of subscripts in allocate-object.
TYPE_PARSER(construct<Allocation>(Parser<AllocateObject>{},
    defaulted(parenthesized(nonemptyList(Parser<AllocateShapeSpec>{}))),
    maybe(bracketed(Parser<AllocateCoarraySpec>{}))))

// R933 allocate-object -> variable-name | structure-component
TYPE_PARSER(construct<AllocateObject>(structureComponent) ||
    construct<AllocateObject>(name / !"="_tok))

// R935 lower-bound-expr -> scalar-int-expr
// R936 upper-bound-expr -> scalar-int-expr
constexpr auto boundExpr{scalarIntExpr};

// R934 allocate-shape-spec -> [lower-bound-expr :] upper-bound-expr
// R938 allocate-coshape-spec -> [lower-bound-expr :] upper-bound-expr
TYPE_PARSER(construct<AllocateShapeSpec>(maybe(boundExpr / ":"), boundExpr))

// R937 allocate-coarray-spec ->
//      [allocate-coshape-spec-list ,] [lower-bound-expr :] *
TYPE_PARSER(construct<AllocateCoarraySpec>(
    defaulted(nonemptyList(Parser<AllocateShapeSpec>{}) / ","),
    maybe(boundExpr / ":") / "*"))

// R939 nullify-stmt -> NULLIFY ( pointer-object-list )
TYPE_CONTEXT_PARSER("NULLIFY statement"_en_US,
    "NULLIFY" >> parenthesized(construct<NullifyStmt>(
                     nonemptyList(Parser<PointerObject>{}))))

// R940 pointer-object ->
//        variable-name | structure-component | proc-pointer-name
TYPE_PARSER(construct<PointerObject>(structureComponent) ||
    construct<PointerObject>(name))

// R941 deallocate-stmt ->
//        DEALLOCATE ( allocate-object-list [, dealloc-opt-list] )
TYPE_CONTEXT_PARSER("DEALLOCATE statement"_en_US,
    construct<DeallocateStmt>(
        "DEALLOCATE (" >> nonemptyList(Parser<AllocateObject>{}),
        defaulted("," >> nonemptyList(statOrErrmsg)) / ")"))

// R942 dealloc-opt -> STAT = stat-variable | ERRMSG = errmsg-variable
// R1165 sync-stat -> STAT = stat-variable | ERRMSG = errmsg-variable
TYPE_PARSER(construct<StatOrErrmsg>("STAT =" >> statVariable) ||
    construct<StatOrErrmsg>("ERRMSG =" >> msgVariable))

// R1001 primary ->
//         literal-constant | designator | array-constructor |
//         structure-constructor | function-reference | type-param-inquiry |
//         type-param-name | ( expr )
// N.B. type-param-inquiry is parsed as a structure component
constexpr auto primary{instrumented("primary"_en_US,
    first(construct<Expr>(indirect(Parser<CharLiteralConstantSubstring>{})),
        construct<Expr>(literalConstant),
        construct<Expr>(construct<Expr::Parentheses>(parenthesized(expr))),
        construct<Expr>(indirect(functionReference) / !"("_tok),
        construct<Expr>(designator / !"("_tok),
        construct<Expr>(Parser<StructureConstructor>{}),
        construct<Expr>(Parser<ArrayConstructor>{}),
        // PGI/XLF extension: COMPLEX constructor (x,y)
        extension<LanguageFeature::ComplexConstructor>(
            construct<Expr>(parenthesized(
                construct<Expr::ComplexConstructor>(expr, "," >> expr)))),
        extension<LanguageFeature::PercentLOC>(construct<Expr>("%LOC" >>
            parenthesized(construct<Expr::PercentLoc>(indirect(variable)))))))};

// R1002 level-1-expr -> [defined-unary-op] primary
// TODO: Reasonable extension: permit multiple defined-unary-ops
constexpr auto level1Expr{sourced(
    first(primary,  // must come before define op to resolve .TRUE._8 ambiguity
        construct<Expr>(construct<Expr::DefinedUnary>(definedOpName, primary)),
        extension<LanguageFeature::SignedPrimary>(
            construct<Expr>(construct<Expr::UnaryPlus>("+" >> primary))),
        extension<LanguageFeature::SignedPrimary>(
            construct<Expr>(construct<Expr::Negate>("-" >> primary)))))};

// R1004 mult-operand -> level-1-expr [power-op mult-operand]
// R1007 power-op -> **
// Exponentiation (**) is Fortran's only right-associative binary operation.
struct MultOperand {
  using resultType = Expr;
  constexpr MultOperand() {}
  static inline std::optional<Expr> Parse(ParseState &);
};

static constexpr auto multOperand{sourced(MultOperand{})};

inline std::optional<Expr> MultOperand::Parse(ParseState &state) {
  std::optional<Expr> result{level1Expr.Parse(state)};
  if (result) {
    static constexpr auto op{attempt("**"_tok)};
    if (op.Parse(state)) {
      std::function<Expr(Expr &&)> power{[&result](Expr &&right) {
        return Expr{Expr::Power(std::move(result).value(), std::move(right))};
      }};
      return applyLambda(power, multOperand).Parse(state);  // right-recursive
    }
  }
  return result;
}

// R1005 add-operand -> [add-operand mult-op] mult-operand
// R1008 mult-op -> * | /
// The left recursion in the grammar is implemented iteratively.
constexpr struct AddOperand {
  using resultType = Expr;
  constexpr AddOperand() {}
  static inline std::optional<Expr> Parse(ParseState &state) {
    std::optional<Expr> result{multOperand.Parse(state)};
    if (result) {
      auto source{result->source};
      std::function<Expr(Expr &&)> multiply{[&result](Expr &&right) {
        return Expr{
            Expr::Multiply(std::move(result).value(), std::move(right))};
      }};
      std::function<Expr(Expr &&)> divide{[&result](Expr &&right) {
        return Expr{Expr::Divide(std::move(result).value(), std::move(right))};
      }};
      auto more{attempt(sourced("*" >> applyLambda(multiply, multOperand) ||
          "/" >> applyLambda(divide, multOperand)))};
      while (std::optional<Expr> next{more.Parse(state)}) {
        result = std::move(next);
        result->source.ExtendToCover(source);
      }
    }
    return result;
  }
} addOperand;

// R1006 level-2-expr -> [[level-2-expr] add-op] add-operand
// R1009 add-op -> + | -
// These are left-recursive productions, implemented iteratively.
// Note that standard Fortran admits a unary + or - to appear only here,
// by means of a missing first operand; e.g., 2*-3 is valid in C but not
// standard Fortran.  We accept unary + and - to appear before any primary
// as an extension.
constexpr struct Level2Expr {
  using resultType = Expr;
  constexpr Level2Expr() {}
  static inline std::optional<Expr> Parse(ParseState &state) {
    static constexpr auto unary{
        sourced(
            construct<Expr>(construct<Expr::UnaryPlus>("+" >> addOperand)) ||
            construct<Expr>(construct<Expr::Negate>("-" >> addOperand))) ||
        addOperand};
    std::optional<Expr> result{unary.Parse(state)};
    if (result) {
      auto source{result->source};
      std::function<Expr(Expr &&)> add{[&result](Expr &&right) {
        return Expr{Expr::Add(std::move(result).value(), std::move(right))};
      }};
      std::function<Expr(Expr &&)> subtract{[&result](Expr &&right) {
        return Expr{
            Expr::Subtract(std::move(result).value(), std::move(right))};
      }};
      auto more{attempt(sourced("+" >> applyLambda(add, addOperand) ||
          "-" >> applyLambda(subtract, addOperand)))};
      while (std::optional<Expr> next{more.Parse(state)}) {
        result = std::move(next);
        result->source.ExtendToCover(source);
      }
    }
    return result;
  }
} level2Expr;

// R1010 level-3-expr -> [level-3-expr concat-op] level-2-expr
// R1011 concat-op -> //
// Concatenation (//) is left-associative for parsing performance, although
// one would never notice if it were right-associated.
constexpr struct Level3Expr {
  using resultType = Expr;
  constexpr Level3Expr() {}
  static inline std::optional<Expr> Parse(ParseState &state) {
    std::optional<Expr> result{level2Expr.Parse(state)};
    if (result) {
      auto source{result->source};
      std::function<Expr(Expr &&)> concat{[&result](Expr &&right) {
        return Expr{Expr::Concat(std::move(result).value(), std::move(right))};
      }};
      auto more{attempt(sourced("//" >> applyLambda(concat, level2Expr)))};
      while (std::optional<Expr> next{more.Parse(state)}) {
        result = std::move(next);
        result->source.ExtendToCover(source);
      }
    }
    return result;
  }
} level3Expr;

// R1012 level-4-expr -> [level-3-expr rel-op] level-3-expr
// R1013 rel-op ->
//         .EQ. | .NE. | .LT. | .LE. | .GT. | .GE. |
//          == | /= | < | <= | > | >=  @ | <>
// N.B. relations are not recursive (i.e., LOGICAL is not ordered)
constexpr struct Level4Expr {
  using resultType = Expr;
  constexpr Level4Expr() {}
  static inline std::optional<Expr> Parse(ParseState &state) {
    std::optional<Expr> result{level3Expr.Parse(state)};
    if (result) {
      auto source{result->source};
      std::function<Expr(Expr &&)> lt{[&result](Expr &&right) {
        return Expr{Expr::LT(std::move(result).value(), std::move(right))};
      }};
      std::function<Expr(Expr &&)> le{[&result](Expr &&right) {
        return Expr{Expr::LE(std::move(result).value(), std::move(right))};
      }};
      std::function<Expr(Expr &&)> eq{[&result](Expr &&right) {
        return Expr{Expr::EQ(std::move(result).value(), std::move(right))};
      }};
      std::function<Expr(Expr &&)> ne{[&result](Expr &&right) {
        return Expr{Expr::NE(std::move(result).value(), std::move(right))};
      }};
      std::function<Expr(Expr &&)> ge{[&result](Expr &&right) {
        return Expr{Expr::GE(std::move(result).value(), std::move(right))};
      }};
      std::function<Expr(Expr &&)> gt{[&result](Expr &&right) {
        return Expr{Expr::GT(std::move(result).value(), std::move(right))};
      }};
      auto more{attempt(
          sourced((".LT."_tok || "<"_tok) >> applyLambda(lt, level3Expr) ||
              (".LE."_tok || "<="_tok) >> applyLambda(le, level3Expr) ||
              (".EQ."_tok || "=="_tok) >> applyLambda(eq, level3Expr) ||
              (".NE."_tok || "/="_tok ||
                  extension<LanguageFeature::AlternativeNE>(
                      "<>"_tok /* PGI/Cray extension; Cray also has .LG. */)) >>
                  applyLambda(ne, level3Expr) ||
              (".GE."_tok || ">="_tok) >> applyLambda(ge, level3Expr) ||
              (".GT."_tok || ">"_tok) >> applyLambda(gt, level3Expr)))};
      if (std::optional<Expr> next{more.Parse(state)}) {
        next->source.ExtendToCover(source);
        return next;
      }
    }
    return result;
  }
} level4Expr;

// R1014 and-operand -> [not-op] level-4-expr
// R1018 not-op -> .NOT.
// N.B. Fortran's .NOT. binds less tightly than its comparison operators do.
// PGI/Intel extension: accept multiple .NOT. operators
constexpr struct AndOperand {
  using resultType = Expr;
  constexpr AndOperand() {}
  static inline std::optional<Expr> Parse(ParseState &);
} andOperand;

inline std::optional<Expr> AndOperand::Parse(ParseState &state) {
  static constexpr auto notOp{attempt(".NOT."_tok >> andOperand)};
  if (std::optional<Expr> negation{notOp.Parse(state)}) {
    return Expr{Expr::NOT{std::move(*negation)}};
  } else {
    return level4Expr.Parse(state);
  }
}

// R1015 or-operand -> [or-operand and-op] and-operand
// R1019 and-op -> .AND.
// .AND. is left-associative
constexpr struct OrOperand {
  using resultType = Expr;
  constexpr OrOperand() {}
  static inline std::optional<Expr> Parse(ParseState &state) {
    static constexpr auto operand{sourced(andOperand)};
    std::optional<Expr> result{operand.Parse(state)};
    if (result) {
      auto source{result->source};
      std::function<Expr(Expr &&)> logicalAnd{[&result](Expr &&right) {
        return Expr{Expr::AND(std::move(result).value(), std::move(right))};
      }};
      auto more{
          attempt(sourced(".AND." >> applyLambda(logicalAnd, andOperand)))};
      while (std::optional<Expr> next{more.Parse(state)}) {
        result = std::move(next);
        result->source.ExtendToCover(source);
      }
    }
    return result;
  }
} orOperand;

// R1016 equiv-operand -> [equiv-operand or-op] or-operand
// R1020 or-op -> .OR.
// .OR. is left-associative
constexpr struct EquivOperand {
  using resultType = Expr;
  constexpr EquivOperand() {}
  static inline std::optional<Expr> Parse(ParseState &state) {
    std::optional<Expr> result{orOperand.Parse(state)};
    if (result) {
      auto source{result->source};
      std::function<Expr(Expr &&)> logicalOr{[&result](Expr &&right) {
        return Expr{Expr::OR(std::move(result).value(), std::move(right))};
      }};
      auto more{attempt(sourced(".OR." >> applyLambda(logicalOr, orOperand)))};
      while (std::optional<Expr> next{more.Parse(state)}) {
        result = std::move(next);
        result->source.ExtendToCover(source);
      }
    }
    return result;
  }
} equivOperand;

// R1017 level-5-expr -> [level-5-expr equiv-op] equiv-operand
// R1021 equiv-op -> .EQV. | .NEQV.
// Logical equivalence is left-associative.
// Extension: .XOR. as synonym for .NEQV.
constexpr struct Level5Expr {
  using resultType = Expr;
  constexpr Level5Expr() {}
  static inline std::optional<Expr> Parse(ParseState &state) {
    std::optional<Expr> result{equivOperand.Parse(state)};
    if (result) {
      auto source{result->source};
      std::function<Expr(Expr &&)> eqv{[&result](Expr &&right) {
        return Expr{Expr::EQV(std::move(result).value(), std::move(right))};
      }};
      std::function<Expr(Expr &&)> neqv{[&result](Expr &&right) {
        return Expr{Expr::NEQV(std::move(result).value(), std::move(right))};
      }};
      std::function<Expr(Expr &&)> logicalXor{[&result](Expr &&right) {
        return Expr{Expr::XOR(std::move(result).value(), std::move(right))};
      }};
      auto more{attempt(sourced(".EQV." >> applyLambda(eqv, equivOperand) ||
          ".NEQV." >> applyLambda(neqv, equivOperand) ||
          extension<LanguageFeature::XOROperator>(
              ".XOR." >> applyLambda(logicalXor, equivOperand))))};
      while (std::optional<Expr> next{more.Parse(state)}) {
        result = std::move(next);
        result->source.ExtendToCover(source);
      }
    }
    return result;
  }
} level5Expr;

// R1022 expr -> [expr defined-binary-op] level-5-expr
// Defined binary operators associate leftwards.
template<> inline std::optional<Expr> Parser<Expr>::Parse(ParseState &state) {
  std::optional<Expr> result{level5Expr.Parse(state)};
  if (result) {
    auto source{result->source};
    std::function<Expr(DefinedOpName &&, Expr &&)> defBinOp{
        [&result](DefinedOpName &&op, Expr &&right) {
          return Expr{Expr::DefinedBinary(
              std::move(op), std::move(result).value(), std::move(right))};
        }};
    auto more{
        attempt(sourced(applyLambda(defBinOp, definedOpName, level5Expr)))};
    while (std::optional<Expr> next{more.Parse(state)}) {
      result = std::move(next);
      result->source.ExtendToCover(source);
    }
  }
  return result;
}

// R1028 specification-expr -> scalar-int-expr
TYPE_PARSER(construct<SpecificationExpr>(scalarIntExpr))

// R1032 assignment-stmt -> variable = expr
TYPE_CONTEXT_PARSER("assignment statement"_en_US,
    construct<AssignmentStmt>(variable / "=", expr))

// R1033 pointer-assignment-stmt ->
//         data-pointer-object [( bounds-spec-list )] => data-target |
//         data-pointer-object ( bounds-remapping-list ) => data-target |
//         proc-pointer-object => proc-target
// R1034 data-pointer-object ->
//         variable-name | scalar-variable % data-pointer-component-name
//   C1022 a scalar-variable shall be a data-ref
//   C1024 a data-pointer-object shall not be a coindexed object
// R1038 proc-pointer-object -> proc-pointer-name | proc-component-ref
//
// A distinction can't be made at the time of the initial parse between
// data-pointer-object and proc-pointer-object, or between data-target
// and proc-target.
TYPE_CONTEXT_PARSER("pointer assignment statement"_en_US,
    construct<PointerAssignmentStmt>(dataRef,
        parenthesized(nonemptyList(Parser<BoundsRemapping>{})), "=>" >> expr) ||
        construct<PointerAssignmentStmt>(dataRef,
            defaulted(parenthesized(nonemptyList(Parser<BoundsSpec>{}))),
            "=>" >> expr))

// R1035 bounds-spec -> lower-bound-expr :
TYPE_PARSER(construct<BoundsSpec>(boundExpr / ":"))

// R1036 bounds-remapping -> lower-bound-expr : upper-bound-expr
TYPE_PARSER(construct<BoundsRemapping>(boundExpr / ":", boundExpr))

// R1039 proc-component-ref -> scalar-variable % procedure-component-name
//   C1027 the scalar-variable must be a data-ref without coindices.
TYPE_PARSER(construct<ProcComponentRef>(structureComponent))

// R1041 where-stmt -> WHERE ( mask-expr ) where-assignment-stmt
// R1045 where-assignment-stmt -> assignment-stmt
// R1046 mask-expr -> logical-expr
TYPE_CONTEXT_PARSER("WHERE statement"_en_US,
    construct<WhereStmt>("WHERE" >> parenthesized(logicalExpr), assignmentStmt))

// R1042 where-construct ->
//         where-construct-stmt [where-body-construct]...
//         [masked-elsewhere-stmt [where-body-construct]...]...
//         [elsewhere-stmt [where-body-construct]...] end-where-stmt
TYPE_CONTEXT_PARSER("WHERE construct"_en_US,
    construct<WhereConstruct>(statement(Parser<WhereConstructStmt>{}),
        many(whereBodyConstruct),
        many(construct<WhereConstruct::MaskedElsewhere>(
            statement(Parser<MaskedElsewhereStmt>{}),
            many(whereBodyConstruct))),
        maybe(construct<WhereConstruct::Elsewhere>(
            statement(Parser<ElsewhereStmt>{}), many(whereBodyConstruct))),
        statement(Parser<EndWhereStmt>{})))

// R1043 where-construct-stmt -> [where-construct-name :] WHERE ( mask-expr )
TYPE_CONTEXT_PARSER("WHERE construct statement"_en_US,
    construct<WhereConstructStmt>(
        maybe(name / ":"), "WHERE" >> parenthesized(logicalExpr)))

// R1044 where-body-construct ->
//         where-assignment-stmt | where-stmt | where-construct
TYPE_PARSER(construct<WhereBodyConstruct>(statement(assignmentStmt)) ||
    construct<WhereBodyConstruct>(statement(whereStmt)) ||
    construct<WhereBodyConstruct>(indirect(whereConstruct)))

// R1047 masked-elsewhere-stmt ->
//         ELSEWHERE ( mask-expr ) [where-construct-name]
TYPE_CONTEXT_PARSER("masked ELSEWHERE statement"_en_US,
    construct<MaskedElsewhereStmt>(
        "ELSE WHERE" >> parenthesized(logicalExpr), maybe(name)))

// R1048 elsewhere-stmt -> ELSEWHERE [where-construct-name]
TYPE_CONTEXT_PARSER("ELSEWHERE statement"_en_US,
    construct<ElsewhereStmt>("ELSE WHERE" >> maybe(name)))

// R1049 end-where-stmt -> ENDWHERE [where-construct-name]
TYPE_CONTEXT_PARSER("END WHERE statement"_en_US,
    construct<EndWhereStmt>(
        recovery("END WHERE" >> maybe(name), endStmtErrorRecovery)))

// R1050 forall-construct ->
//         forall-construct-stmt [forall-body-construct]... end-forall-stmt
TYPE_CONTEXT_PARSER("FORALL construct"_en_US,
    construct<ForallConstruct>(statement(Parser<ForallConstructStmt>{}),
        many(Parser<ForallBodyConstruct>{}),
        statement(Parser<EndForallStmt>{})))

// R1051 forall-construct-stmt ->
//         [forall-construct-name :] FORALL concurrent-header
TYPE_CONTEXT_PARSER("FORALL construct statement"_en_US,
    construct<ForallConstructStmt>(
        maybe(name / ":"), "FORALL" >> indirect(concurrentHeader)))

// R1052 forall-body-construct ->
//         forall-assignment-stmt | where-stmt | where-construct |
//         forall-construct | forall-stmt
TYPE_PARSER(construct<ForallBodyConstruct>(statement(forallAssignmentStmt)) ||
    construct<ForallBodyConstruct>(statement(whereStmt)) ||
    construct<ForallBodyConstruct>(whereConstruct) ||
    construct<ForallBodyConstruct>(indirect(forallConstruct)) ||
    construct<ForallBodyConstruct>(statement(forallStmt)))

// R1053 forall-assignment-stmt -> assignment-stmt | pointer-assignment-stmt
TYPE_PARSER(construct<ForallAssignmentStmt>(assignmentStmt) ||
    construct<ForallAssignmentStmt>(pointerAssignmentStmt))

// R1054 end-forall-stmt -> END FORALL [forall-construct-name]
TYPE_CONTEXT_PARSER("END FORALL statement"_en_US,
    construct<EndForallStmt>(
        recovery("END FORALL" >> maybe(name), endStmtErrorRecovery)))

// R1055 forall-stmt -> FORALL concurrent-header forall-assignment-stmt
TYPE_CONTEXT_PARSER("FORALL statement"_en_US,
    construct<ForallStmt>("FORALL" >> indirect(concurrentHeader),
        unlabeledStatement(forallAssignmentStmt)))

// R1101 block -> [execution-part-construct]...
constexpr auto block{many(executionPartConstruct)};

// R1102 associate-construct -> associate-stmt block end-associate-stmt
TYPE_CONTEXT_PARSER("ASSOCIATE construct"_en_US,
    construct<AssociateConstruct>(statement(Parser<AssociateStmt>{}), block,
        statement(Parser<EndAssociateStmt>{})))

// R1103 associate-stmt ->
//        [associate-construct-name :] ASSOCIATE ( association-list )
TYPE_CONTEXT_PARSER("ASSOCIATE statement"_en_US,
    construct<AssociateStmt>(maybe(name / ":"),
        "ASSOCIATE" >> parenthesized(nonemptyList(Parser<Association>{}))))

// R1104 association -> associate-name => selector
TYPE_PARSER(construct<Association>(name, "=>" >> selector))

// R1105 selector -> expr | variable
TYPE_PARSER(construct<Selector>(variable) / lookAhead(","_tok || ")"_tok) ||
    construct<Selector>(expr))

// R1106 end-associate-stmt -> END ASSOCIATE [associate-construct-name]
TYPE_PARSER(construct<EndAssociateStmt>(
    recovery("END ASSOCIATE" >> maybe(name), endStmtErrorRecovery)))

// R1107 block-construct ->
//         block-stmt [block-specification-part] block end-block-stmt
TYPE_CONTEXT_PARSER("BLOCK construct"_en_US,
    construct<BlockConstruct>(statement(Parser<BlockStmt>{}),
        Parser<BlockSpecificationPart>{},  // can be empty
        block, statement(Parser<EndBlockStmt>{})))

// R1108 block-stmt -> [block-construct-name :] BLOCK
TYPE_PARSER(construct<BlockStmt>(maybe(name / ":") / "BLOCK"))

// R1109 block-specification-part ->
//         [use-stmt]... [import-stmt]... [implicit-part]
//         [[declaration-construct]... specification-construct]
// C1107 prohibits COMMON, EQUIVALENCE, INTENT, NAMELIST, OPTIONAL, VALUE,
// and statement function definitions.  C1108 prohibits SAVE /common/.
// C1570 indirectly prohibits ENTRY.  These constraints are best enforced later.
// The odd grammar rule above would have the effect of forcing any
// trailing FORMAT and DATA statements after the last specification-construct
// to be recognized as part of the block-construct's block part rather than
// its block-specification-part, a distinction without any apparent difference.
TYPE_PARSER(construct<BlockSpecificationPart>(specificationPart))

// R1110 end-block-stmt -> END BLOCK [block-construct-name]
TYPE_PARSER(construct<EndBlockStmt>(
    recovery("END BLOCK" >> maybe(name), endStmtErrorRecovery)))

// R1111 change-team-construct -> change-team-stmt block end-change-team-stmt
TYPE_CONTEXT_PARSER("CHANGE TEAM construct"_en_US,
    construct<ChangeTeamConstruct>(statement(Parser<ChangeTeamStmt>{}), block,
        statement(Parser<EndChangeTeamStmt>{})))

// R1112 change-team-stmt ->
//         [team-construct-name :] CHANGE TEAM
//         ( team-value [, coarray-association-list] [, sync-stat-list] )
TYPE_CONTEXT_PARSER("CHANGE TEAM statement"_en_US,
    construct<ChangeTeamStmt>(maybe(name / ":"),
        "CHANGE TEAM"_sptok >> "("_tok >> teamValue,
        defaulted("," >> nonemptyList(Parser<CoarrayAssociation>{})),
        defaulted("," >> nonemptyList(statOrErrmsg))) /
        ")")

// R1113 coarray-association -> codimension-decl => selector
TYPE_PARSER(
    construct<CoarrayAssociation>(Parser<CodimensionDecl>{}, "=>" >> selector))

// R1114 end-change-team-stmt ->
//         END TEAM [( [sync-stat-list] )] [team-construct-name]
TYPE_CONTEXT_PARSER("END TEAM statement"_en_US,
    construct<EndChangeTeamStmt>(
        "END TEAM" >> defaulted(parenthesized(optionalList(statOrErrmsg))),
        maybe(name)))

// R1117 critical-stmt ->
//         [critical-construct-name :] CRITICAL [( [sync-stat-list] )]
TYPE_CONTEXT_PARSER("CRITICAL statement"_en_US,
    construct<CriticalStmt>(maybe(name / ":"),
        "CRITICAL" >> defaulted(parenthesized(optionalList(statOrErrmsg)))))

// R1116 critical-construct -> critical-stmt block end-critical-stmt
TYPE_CONTEXT_PARSER("CRITICAL construct"_en_US,
    construct<CriticalConstruct>(statement(Parser<CriticalStmt>{}), block,
        statement(Parser<EndCriticalStmt>{})))

// R1118 end-critical-stmt -> END CRITICAL [critical-construct-name]
TYPE_PARSER(construct<EndCriticalStmt>(
    recovery("END CRITICAL" >> maybe(name), endStmtErrorRecovery)))

// R1119 do-construct -> do-stmt block end-do
// R1120 do-stmt -> nonlabel-do-stmt | label-do-stmt
TYPE_CONTEXT_PARSER("DO construct"_en_US,
    construct<DoConstruct>(
        statement(Parser<NonLabelDoStmt>{}) / EnterNonlabelDoConstruct{}, block,
        statement(Parser<EndDoStmt>{}) / LeaveDoConstruct{}))

// R1125 concurrent-header ->
//         ( [integer-type-spec ::] concurrent-control-list
//         [, scalar-mask-expr] )
TYPE_PARSER(parenthesized(construct<ConcurrentHeader>(
    maybe(integerTypeSpec / "::"), nonemptyList(Parser<ConcurrentControl>{}),
    maybe("," >> scalarLogicalExpr))))

// R1126 concurrent-control ->
//         index-name = concurrent-limit : concurrent-limit [: concurrent-step]
// R1127 concurrent-limit -> scalar-int-expr
// R1128 concurrent-step -> scalar-int-expr
TYPE_PARSER(construct<ConcurrentControl>(name / "=", scalarIntExpr / ":",
    scalarIntExpr, maybe(":" >> scalarIntExpr)))

// R1130 locality-spec ->
//         LOCAL ( variable-name-list ) | LOCAL_INIT ( variable-name-list ) |
//         SHARED ( variable-name-list ) | DEFAULT ( NONE )
TYPE_PARSER(construct<LocalitySpec>(construct<LocalitySpec::Local>(
                "LOCAL" >> parenthesized(listOfNames))) ||
    construct<LocalitySpec>(construct<LocalitySpec::LocalInit>(
        "LOCAL_INIT"_sptok >> parenthesized(listOfNames))) ||
    construct<LocalitySpec>(construct<LocalitySpec::Shared>(
        "SHARED" >> parenthesized(listOfNames))) ||
    construct<LocalitySpec>(
        construct<LocalitySpec::DefaultNone>("DEFAULT ( NONE )"_tok)))

// R1123 loop-control ->
//         [,] do-variable = scalar-int-expr , scalar-int-expr
//           [, scalar-int-expr] |
//         [,] WHILE ( scalar-logical-expr ) |
//         [,] CONCURRENT concurrent-header concurrent-locality
// R1129 concurrent-locality -> [locality-spec]...
TYPE_CONTEXT_PARSER("loop control"_en_US,
    maybe(","_tok) >>
        (construct<LoopControl>(loopBounds(scalarExpr)) ||
            construct<LoopControl>(
                "WHILE" >> parenthesized(scalarLogicalExpr)) ||
            construct<LoopControl>(construct<LoopControl::Concurrent>(
                "CONCURRENT" >> concurrentHeader,
                many(Parser<LocalitySpec>{})))))

// R1121 label-do-stmt -> [do-construct-name :] DO label [loop-control]
TYPE_CONTEXT_PARSER("label DO statement"_en_US,
    construct<LabelDoStmt>(
        maybe(name / ":"), "DO" >> label, maybe(loopControl)))

// R1122 nonlabel-do-stmt -> [do-construct-name :] DO [loop-control]
TYPE_CONTEXT_PARSER("nonlabel DO statement"_en_US,
    construct<NonLabelDoStmt>(maybe(name / ":"), "DO" >> maybe(loopControl)))

// R1132 end-do-stmt -> END DO [do-construct-name]
TYPE_CONTEXT_PARSER("END DO statement"_en_US,
    construct<EndDoStmt>(
        recovery("END DO" >> maybe(name), endStmtErrorRecovery)))

// R1133 cycle-stmt -> CYCLE [do-construct-name]
TYPE_CONTEXT_PARSER(
    "CYCLE statement"_en_US, construct<CycleStmt>("CYCLE" >> maybe(name)))

// R1134 if-construct ->
//         if-then-stmt block [else-if-stmt block]...
//         [else-stmt block] end-if-stmt
// R1135 if-then-stmt -> [if-construct-name :] IF ( scalar-logical-expr )
// THEN R1136 else-if-stmt ->
//         ELSE IF ( scalar-logical-expr ) THEN [if-construct-name]
// R1137 else-stmt -> ELSE [if-construct-name]
// R1138 end-if-stmt -> END IF [if-construct-name]
TYPE_CONTEXT_PARSER("IF construct"_en_US,
    construct<IfConstruct>(
        statement(construct<IfThenStmt>(maybe(name / ":"),
            "IF" >> parenthesized(scalarLogicalExpr) / "THEN")),
        block,
        many(construct<IfConstruct::ElseIfBlock>(
            unambiguousStatement(construct<ElseIfStmt>(
                "ELSE IF" >> parenthesized(scalarLogicalExpr),
                "THEN" >> maybe(name))),
            block)),
        maybe(construct<IfConstruct::ElseBlock>(
            statement(construct<ElseStmt>("ELSE" >> maybe(name))), block)),
        statement(construct<EndIfStmt>(
            recovery("END IF" >> maybe(name), endStmtErrorRecovery)))))

// R1139 if-stmt -> IF ( scalar-logical-expr ) action-stmt
TYPE_CONTEXT_PARSER("IF statement"_en_US,
    construct<IfStmt>("IF" >> parenthesized(scalarLogicalExpr),
        unlabeledStatement(actionStmt)))

// R1140 case-construct ->
//         select-case-stmt [case-stmt block]... end-select-stmt
TYPE_CONTEXT_PARSER("SELECT CASE construct"_en_US,
    construct<CaseConstruct>(statement(Parser<SelectCaseStmt>{}),
        many(construct<CaseConstruct::Case>(
            unambiguousStatement(Parser<CaseStmt>{}), block)),
        statement(endSelectStmt)))

// R1141 select-case-stmt -> [case-construct-name :] SELECT CASE ( case-expr
// ) R1144 case-expr -> scalar-expr
TYPE_CONTEXT_PARSER("SELECT CASE statement"_en_US,
    construct<SelectCaseStmt>(
        maybe(name / ":"), "SELECT CASE" >> parenthesized(scalar(expr))))

// R1142 case-stmt -> CASE case-selector [case-construct-name]
TYPE_CONTEXT_PARSER("CASE statement"_en_US,
    construct<CaseStmt>("CASE" >> Parser<CaseSelector>{}, maybe(name)))

// R1143 end-select-stmt -> END SELECT [case-construct-name]
// R1151 end-select-rank-stmt -> END SELECT [select-construct-name]
// R1155 end-select-type-stmt -> END SELECT [select-construct-name]
TYPE_PARSER(construct<EndSelectStmt>(
    recovery("END SELECT" >> maybe(name), endStmtErrorRecovery)))

// R1145 case-selector -> ( case-value-range-list ) | DEFAULT
constexpr auto defaultKeyword{construct<Default>("DEFAULT"_tok)};
TYPE_PARSER(parenthesized(construct<CaseSelector>(
                nonemptyList(Parser<CaseValueRange>{}))) ||
    construct<CaseSelector>(defaultKeyword))

// R1147 case-value -> scalar-constant-expr
constexpr auto caseValue{scalar(constantExpr)};

// R1146 case-value-range ->
//         case-value | case-value : | : case-value | case-value : case-value
TYPE_PARSER(construct<CaseValueRange>(construct<CaseValueRange::Range>(
                construct<std::optional<CaseValue>>(caseValue),
                ":" >> maybe(caseValue))) ||
    construct<CaseValueRange>(
        construct<CaseValueRange::Range>(construct<std::optional<CaseValue>>(),
            ":" >> construct<std::optional<CaseValue>>(caseValue))) ||
    construct<CaseValueRange>(caseValue))

// R1148 select-rank-construct ->
//         select-rank-stmt [select-rank-case-stmt block]...
//         end-select-rank-stmt
TYPE_CONTEXT_PARSER("SELECT RANK construct"_en_US,
    construct<SelectRankConstruct>(statement(Parser<SelectRankStmt>{}),
        many(construct<SelectRankConstruct::RankCase>(
            unambiguousStatement(Parser<SelectRankCaseStmt>{}), block)),
        statement(endSelectStmt)))

// R1149 select-rank-stmt ->
//         [select-construct-name :] SELECT RANK
//         ( [associate-name =>] selector )
TYPE_CONTEXT_PARSER("SELECT RANK statement"_en_US,
    construct<SelectRankStmt>(maybe(name / ":"),
        "SELECT RANK"_sptok >> "("_tok >> maybe(name / "=>"), selector / ")"))

// R1150 select-rank-case-stmt ->
//         RANK ( scalar-int-constant-expr ) [select-construct-name] |
//         RANK ( * ) [select-construct-name] |
//         RANK DEFAULT [select-construct-name]
TYPE_CONTEXT_PARSER("RANK case statement"_en_US,
    "RANK" >> (construct<SelectRankCaseStmt>(
                  parenthesized(construct<SelectRankCaseStmt::Rank>(
                                    scalarIntConstantExpr) ||
                      construct<SelectRankCaseStmt::Rank>(star)) ||
                      construct<SelectRankCaseStmt::Rank>(defaultKeyword),
                  maybe(name))))

// R1152 select-type-construct ->
//         select-type-stmt [type-guard-stmt block]... end-select-type-stmt
TYPE_CONTEXT_PARSER("SELECT TYPE construct"_en_US,
    construct<SelectTypeConstruct>(statement(Parser<SelectTypeStmt>{}),
        many(construct<SelectTypeConstruct::TypeCase>(
            unambiguousStatement(Parser<TypeGuardStmt>{}), block)),
        statement(endSelectStmt)))

// R1153 select-type-stmt ->
//         [select-construct-name :] SELECT TYPE
//         ( [associate-name =>] selector )
TYPE_CONTEXT_PARSER("SELECT TYPE statement"_en_US,
    construct<SelectTypeStmt>(maybe(name / ":"),
        "SELECT TYPE (" >> maybe(name / "=>"), selector / ")"))

// R1154 type-guard-stmt ->
//         TYPE IS ( type-spec ) [select-construct-name] |
//         CLASS IS ( derived-type-spec ) [select-construct-name] |
//         CLASS DEFAULT [select-construct-name]
TYPE_CONTEXT_PARSER("type guard statement"_en_US,
    construct<TypeGuardStmt>("TYPE IS"_sptok >>
                parenthesized(construct<TypeGuardStmt::Guard>(typeSpec)) ||
            "CLASS IS"_sptok >> parenthesized(construct<TypeGuardStmt::Guard>(
                                    derivedTypeSpec)) ||
            construct<TypeGuardStmt::Guard>("CLASS" >> defaultKeyword),
        maybe(name)))

// R1156 exit-stmt -> EXIT [construct-name]
TYPE_CONTEXT_PARSER(
    "EXIT statement"_en_US, construct<ExitStmt>("EXIT" >> maybe(name)))

// R1157 goto-stmt -> GO TO label
TYPE_CONTEXT_PARSER(
    "GOTO statement"_en_US, construct<GotoStmt>("GO TO" >> label))

// R1158 computed-goto-stmt -> GO TO ( label-list ) [,] scalar-int-expr
TYPE_CONTEXT_PARSER("computed GOTO statement"_en_US,
    construct<ComputedGotoStmt>("GO TO" >> parenthesized(nonemptyList(label)),
        maybe(","_tok) >> scalarIntExpr))

// R1160 stop-stmt -> STOP [stop-code] [, QUIET = scalar-logical-expr]
// R1161 error-stop-stmt ->
//         ERROR STOP [stop-code] [, QUIET = scalar-logical-expr]
TYPE_CONTEXT_PARSER("STOP statement"_en_US,
    construct<StopStmt>("STOP" >> pure(StopStmt::Kind::Stop) ||
            "ERROR STOP"_sptok >> pure(StopStmt::Kind::ErrorStop),
        maybe(Parser<StopCode>{}), maybe(", QUIET =" >> scalarLogicalExpr)))

// R1162 stop-code -> scalar-default-char-expr | scalar-int-expr
// The two alternatives for stop-code can't be distinguished at
// parse time.
TYPE_PARSER(construct<StopCode>(scalar(expr)))

// R1164 sync-all-stmt -> SYNC ALL [( [sync-stat-list] )]
TYPE_CONTEXT_PARSER("SYNC ALL statement"_en_US,
    construct<SyncAllStmt>("SYNC ALL"_sptok >>
        defaulted(parenthesized(optionalList(statOrErrmsg)))))

// R1166 sync-images-stmt -> SYNC IMAGES ( image-set [, sync-stat-list] )
// R1167 image-set -> int-expr | *
TYPE_CONTEXT_PARSER("SYNC IMAGES statement"_en_US,
    "SYNC IMAGES"_sptok >> parenthesized(construct<SyncImagesStmt>(
                               construct<SyncImagesStmt::ImageSet>(intExpr) ||
                                   construct<SyncImagesStmt::ImageSet>(star),
                               defaulted("," >> nonemptyList(statOrErrmsg)))))

// R1168 sync-memory-stmt -> SYNC MEMORY [( [sync-stat-list] )]
TYPE_CONTEXT_PARSER("SYNC MEMORY statement"_en_US,
    construct<SyncMemoryStmt>("SYNC MEMORY"_sptok >>
        defaulted(parenthesized(optionalList(statOrErrmsg)))))

// R1169 sync-team-stmt -> SYNC TEAM ( team-value [, sync-stat-list] )
TYPE_CONTEXT_PARSER("SYNC TEAM statement"_en_US,
    construct<SyncTeamStmt>("SYNC TEAM"_sptok >> "("_tok >> teamValue,
        defaulted("," >> nonemptyList(statOrErrmsg)) / ")"))

// R1170 event-post-stmt -> EVENT POST ( event-variable [, sync-stat-list] )
// R1171 event-variable -> scalar-variable
TYPE_CONTEXT_PARSER("EVENT POST statement"_en_US,
    construct<EventPostStmt>("EVENT POST"_sptok >> "("_tok >> scalar(variable),
        defaulted("," >> nonemptyList(statOrErrmsg)) / ")"))

// R1172 event-wait-stmt ->
//         EVENT WAIT ( event-variable [, event-wait-spec-list] )
TYPE_CONTEXT_PARSER("EVENT WAIT statement"_en_US,
    construct<EventWaitStmt>("EVENT WAIT"_sptok >> "("_tok >> scalar(variable),
        defaulted("," >> nonemptyList(Parser<EventWaitStmt::EventWaitSpec>{})) /
            ")"))

// R1174 until-spec -> UNTIL_COUNT = scalar-int-expr
constexpr auto untilSpec{"UNTIL_COUNT =" >> scalarIntExpr};

// R1173 event-wait-spec -> until-spec | sync-stat
TYPE_PARSER(construct<EventWaitStmt::EventWaitSpec>(untilSpec) ||
    construct<EventWaitStmt::EventWaitSpec>(statOrErrmsg))

// R1177 team-variable -> scalar-variable
constexpr auto teamVariable{scalar(variable)};

// R1175 form-team-stmt ->
//         FORM TEAM ( team-number , team-variable [, form-team-spec-list] )
// R1176 team-number -> scalar-int-expr
TYPE_CONTEXT_PARSER("FORM TEAM statement"_en_US,
    construct<FormTeamStmt>("FORM TEAM"_sptok >> "("_tok >> scalarIntExpr,
        "," >> teamVariable,
        defaulted("," >> nonemptyList(Parser<FormTeamStmt::FormTeamSpec>{})) /
            ")"))

// R1178 form-team-spec -> NEW_INDEX = scalar-int-expr | sync-stat
TYPE_PARSER(
    construct<FormTeamStmt::FormTeamSpec>("NEW_INDEX =" >> scalarIntExpr) ||
    construct<FormTeamStmt::FormTeamSpec>(statOrErrmsg))

// R1182 lock-variable -> scalar-variable
constexpr auto lockVariable{scalar(variable)};

// R1179 lock-stmt -> LOCK ( lock-variable [, lock-stat-list] )
TYPE_CONTEXT_PARSER("LOCK statement"_en_US,
    construct<LockStmt>("LOCK (" >> lockVariable,
        defaulted("," >> nonemptyList(Parser<LockStmt::LockStat>{})) / ")"))

// R1180 lock-stat -> ACQUIRED_LOCK = scalar-logical-variable | sync-stat
TYPE_PARSER(
    construct<LockStmt::LockStat>("ACQUIRED_LOCK =" >> scalarLogicalVariable) ||
    construct<LockStmt::LockStat>(statOrErrmsg))

// R1181 unlock-stmt -> UNLOCK ( lock-variable [, sync-stat-list] )
TYPE_CONTEXT_PARSER("UNLOCK statement"_en_US,
    construct<UnlockStmt>("UNLOCK (" >> lockVariable,
        defaulted("," >> nonemptyList(statOrErrmsg)) / ")"))

// R1201 io-unit -> file-unit-number | * | internal-file-variable
// R1203 internal-file-variable -> char-variable
// R905 char-variable -> variable
// "char-variable" is attempted first since it's not type constrained but
// syntactically ambiguous with "file-unit-number", which is constrained.
TYPE_PARSER(construct<IoUnit>(variable / !"="_tok) ||
    construct<IoUnit>(fileUnitNumber) || construct<IoUnit>(star))

// R1202 file-unit-number -> scalar-int-expr
TYPE_PARSER(construct<FileUnitNumber>(scalarIntExpr / !"="_tok))

// R1204 open-stmt -> OPEN ( connect-spec-list )
TYPE_CONTEXT_PARSER("OPEN statement"_en_US,
    construct<OpenStmt>(
        "OPEN (" >> nonemptyList("expected connection specifications"_err_en_US,
                        Parser<ConnectSpec>{}) /
            ")"))

// R1206 file-name-expr -> scalar-default-char-expr
constexpr auto fileNameExpr{scalarDefaultCharExpr};

// R1205 connect-spec ->
//         [UNIT =] file-unit-number | ACCESS = scalar-default-char-expr |
//         ACTION = scalar-default-char-expr |
//         ASYNCHRONOUS = scalar-default-char-expr |
//         BLANK = scalar-default-char-expr |
//         DECIMAL = scalar-default-char-expr |
//         DELIM = scalar-default-char-expr |
//         ENCODING = scalar-default-char-expr | ERR = label |
//         FILE = file-name-expr | FORM = scalar-default-char-expr |
//         IOMSG = iomsg-variable | IOSTAT = scalar-int-variable |
//         NEWUNIT = scalar-int-variable | PAD = scalar-default-char-expr |
//         POSITION = scalar-default-char-expr | RECL = scalar-int-expr |
//         ROUND = scalar-default-char-expr | SIGN = scalar-default-char-expr |
//         STATUS = scalar-default-char-expr
//         @ | CONVERT = scalar-default-char-variable
//         @ | DISPOSE = scalar-default-char-variable
constexpr auto statusExpr{construct<StatusExpr>(scalarDefaultCharExpr)};
constexpr auto errLabel{construct<ErrLabel>(label)};

TYPE_PARSER(first(construct<ConnectSpec>(maybe("UNIT ="_tok) >> fileUnitNumber),
    construct<ConnectSpec>(construct<ConnectSpec::CharExpr>(
        "ACCESS =" >> pure(ConnectSpec::CharExpr::Kind::Access),
        scalarDefaultCharExpr)),
    construct<ConnectSpec>(construct<ConnectSpec::CharExpr>(
        "ACTION =" >> pure(ConnectSpec::CharExpr::Kind::Action),
        scalarDefaultCharExpr)),
    construct<ConnectSpec>(construct<ConnectSpec::CharExpr>(
        "ASYNCHRONOUS =" >> pure(ConnectSpec::CharExpr::Kind::Asynchronous),
        scalarDefaultCharExpr)),
    construct<ConnectSpec>(construct<ConnectSpec::CharExpr>(
        "BLANK =" >> pure(ConnectSpec::CharExpr::Kind::Blank),
        scalarDefaultCharExpr)),
    construct<ConnectSpec>(construct<ConnectSpec::CharExpr>(
        "DECIMAL =" >> pure(ConnectSpec::CharExpr::Kind::Decimal),
        scalarDefaultCharExpr)),
    construct<ConnectSpec>(construct<ConnectSpec::CharExpr>(
        "DELIM =" >> pure(ConnectSpec::CharExpr::Kind::Delim),
        scalarDefaultCharExpr)),
    construct<ConnectSpec>(construct<ConnectSpec::CharExpr>(
        "ENCODING =" >> pure(ConnectSpec::CharExpr::Kind::Encoding),
        scalarDefaultCharExpr)),
    construct<ConnectSpec>("ERR =" >> errLabel),
    construct<ConnectSpec>("FILE =" >> fileNameExpr),
    extension<LanguageFeature::FileName>(
        construct<ConnectSpec>("NAME =" >> fileNameExpr)),
    construct<ConnectSpec>(construct<ConnectSpec::CharExpr>(
        "FORM =" >> pure(ConnectSpec::CharExpr::Kind::Form),
        scalarDefaultCharExpr)),
    construct<ConnectSpec>("IOMSG =" >> msgVariable),
    construct<ConnectSpec>("IOSTAT =" >> statVariable),
    construct<ConnectSpec>(construct<ConnectSpec::Newunit>(
        "NEWUNIT =" >> scalar(integer(variable)))),
    construct<ConnectSpec>(construct<ConnectSpec::CharExpr>(
        "PAD =" >> pure(ConnectSpec::CharExpr::Kind::Pad),
        scalarDefaultCharExpr)),
    construct<ConnectSpec>(construct<ConnectSpec::CharExpr>(
        "POSITION =" >> pure(ConnectSpec::CharExpr::Kind::Position),
        scalarDefaultCharExpr)),
    construct<ConnectSpec>(
        construct<ConnectSpec::Recl>("RECL =" >> scalarIntExpr)),
    construct<ConnectSpec>(construct<ConnectSpec::CharExpr>(
        "ROUND =" >> pure(ConnectSpec::CharExpr::Kind::Round),
        scalarDefaultCharExpr)),
    construct<ConnectSpec>(construct<ConnectSpec::CharExpr>(
        "SIGN =" >> pure(ConnectSpec::CharExpr::Kind::Sign),
        scalarDefaultCharExpr)),
    construct<ConnectSpec>("STATUS =" >> statusExpr),
    extension<LanguageFeature::Convert>(
        construct<ConnectSpec>(construct<ConnectSpec::CharExpr>(
            "CONVERT =" >> pure(ConnectSpec::CharExpr::Kind::Convert),
            scalarDefaultCharExpr))),
    extension<LanguageFeature::Dispose>(
        construct<ConnectSpec>(construct<ConnectSpec::CharExpr>(
            "DISPOSE =" >> pure(ConnectSpec::CharExpr::Kind::Dispose),
            scalarDefaultCharExpr)))))

// R1209 close-spec ->
//         [UNIT =] file-unit-number | IOSTAT = scalar-int-variable |
//         IOMSG = iomsg-variable | ERR = label |
//         STATUS = scalar-default-char-expr
constexpr auto closeSpec{first(
    construct<CloseStmt::CloseSpec>(maybe("UNIT ="_tok) >> fileUnitNumber),
    construct<CloseStmt::CloseSpec>("IOSTAT =" >> statVariable),
    construct<CloseStmt::CloseSpec>("IOMSG =" >> msgVariable),
    construct<CloseStmt::CloseSpec>("ERR =" >> errLabel),
    construct<CloseStmt::CloseSpec>("STATUS =" >> statusExpr))};

// R1208 close-stmt -> CLOSE ( close-spec-list )
TYPE_CONTEXT_PARSER("CLOSE statement"_en_US,
    construct<CloseStmt>("CLOSE" >> parenthesized(nonemptyList(closeSpec))))

// R1210 read-stmt ->
//         READ ( io-control-spec-list ) [input-item-list] |
//         READ format [, input-item-list]
constexpr auto inputItemList{
    extension<LanguageFeature::IOListLeadingComma>(
        some("," >> inputItem)) ||  // legacy extension: leading comma
    optionalList(inputItem)};

TYPE_CONTEXT_PARSER("READ statement"_en_US,
    construct<ReadStmt>("READ (" >>
            construct<std::optional<IoUnit>>(maybe("UNIT ="_tok) >> ioUnit),
        "," >> construct<std::optional<Format>>(format),
        defaulted("," >> nonemptyList(ioControlSpec)) / ")", inputItemList) ||
        construct<ReadStmt>(
            "READ (" >> construct<std::optional<IoUnit>>(ioUnit),
            construct<std::optional<Format>>(),
            defaulted("," >> nonemptyList(ioControlSpec)) / ")",
            inputItemList) ||
        construct<ReadStmt>("READ" >> construct<std::optional<IoUnit>>(),
            construct<std::optional<Format>>(),
            parenthesized(nonemptyList(ioControlSpec)), inputItemList) ||
        construct<ReadStmt>("READ" >> construct<std::optional<IoUnit>>(),
            construct<std::optional<Format>>(format),
            construct<std::list<IoControlSpec>>(), many("," >> inputItem)))

// R1214 id-variable -> scalar-int-variable
constexpr auto idVariable{construct<IdVariable>(scalarIntVariable)};

// R1213 io-control-spec ->
//         [UNIT =] io-unit | [FMT =] format | [NML =] namelist-group-name |
//         ADVANCE = scalar-default-char-expr |
//         ASYNCHRONOUS = scalar-default-char-constant-expr |
//         BLANK = scalar-default-char-expr |
//         DECIMAL = scalar-default-char-expr |
//         DELIM = scalar-default-char-expr | END = label | EOR = label |
//         ERR = label | ID = id-variable | IOMSG = iomsg-variable |
//         IOSTAT = scalar-int-variable | PAD = scalar-default-char-expr |
//         POS = scalar-int-expr | REC = scalar-int-expr |
//         ROUND = scalar-default-char-expr | SIGN = scalar-default-char-expr |
//         SIZE = scalar-int-variable
constexpr auto endLabel{construct<EndLabel>(label)};
constexpr auto eorLabel{construct<EorLabel>(label)};
TYPE_PARSER(first(construct<IoControlSpec>("UNIT =" >> ioUnit),
    construct<IoControlSpec>("FMT =" >> format),
    construct<IoControlSpec>("NML =" >> name),
    construct<IoControlSpec>(
        "ADVANCE =" >> construct<IoControlSpec::CharExpr>(
                           pure(IoControlSpec::CharExpr::Kind::Advance),
                           scalarDefaultCharExpr)),
    construct<IoControlSpec>(construct<IoControlSpec::Asynchronous>(
        "ASYNCHRONOUS =" >> scalarDefaultCharConstantExpr)),
    construct<IoControlSpec>("BLANK =" >>
        construct<IoControlSpec::CharExpr>(
            pure(IoControlSpec::CharExpr::Kind::Blank), scalarDefaultCharExpr)),
    construct<IoControlSpec>(
        "DECIMAL =" >> construct<IoControlSpec::CharExpr>(
                           pure(IoControlSpec::CharExpr::Kind::Decimal),
                           scalarDefaultCharExpr)),
    construct<IoControlSpec>("DELIM =" >>
        construct<IoControlSpec::CharExpr>(
            pure(IoControlSpec::CharExpr::Kind::Delim), scalarDefaultCharExpr)),
    construct<IoControlSpec>("END =" >> endLabel),
    construct<IoControlSpec>("EOR =" >> eorLabel),
    construct<IoControlSpec>("ERR =" >> errLabel),
    construct<IoControlSpec>("ID =" >> idVariable),
    construct<IoControlSpec>("IOMSG = " >> msgVariable),
    construct<IoControlSpec>("IOSTAT = " >> statVariable),
    construct<IoControlSpec>("PAD =" >>
        construct<IoControlSpec::CharExpr>(
            pure(IoControlSpec::CharExpr::Kind::Pad), scalarDefaultCharExpr)),
    construct<IoControlSpec>(
        "POS =" >> construct<IoControlSpec::Pos>(scalarIntExpr)),
    construct<IoControlSpec>(
        "REC =" >> construct<IoControlSpec::Rec>(scalarIntExpr)),
    construct<IoControlSpec>("ROUND =" >>
        construct<IoControlSpec::CharExpr>(
            pure(IoControlSpec::CharExpr::Kind::Round), scalarDefaultCharExpr)),
    construct<IoControlSpec>("SIGN =" >>
        construct<IoControlSpec::CharExpr>(
            pure(IoControlSpec::CharExpr::Kind::Sign), scalarDefaultCharExpr)),
    construct<IoControlSpec>(
        "SIZE =" >> construct<IoControlSpec::Size>(scalarIntVariable))))

// R1211 write-stmt -> WRITE ( io-control-spec-list ) [output-item-list]
constexpr auto outputItemList{
    extension<LanguageFeature::IOListLeadingComma>(
        some("," >> outputItem)) ||  // legacy: allow leading comma
    optionalList(outputItem)};

TYPE_CONTEXT_PARSER("WRITE statement"_en_US,
    construct<WriteStmt>("WRITE (" >>
            construct<std::optional<IoUnit>>(maybe("UNIT ="_tok) >> ioUnit),
        "," >> construct<std::optional<Format>>(format),
        defaulted("," >> nonemptyList(ioControlSpec)) / ")", outputItemList) ||
        construct<WriteStmt>(
            "WRITE (" >> construct<std::optional<IoUnit>>(ioUnit),
            construct<std::optional<Format>>(),
            defaulted("," >> nonemptyList(ioControlSpec)) / ")",
            outputItemList) ||
        construct<WriteStmt>("WRITE" >> construct<std::optional<IoUnit>>(),
            construct<std::optional<Format>>(),
            parenthesized(nonemptyList(ioControlSpec)), outputItemList))

// R1212 print-stmt PRINT format [, output-item-list]
TYPE_CONTEXT_PARSER("PRINT statement"_en_US,
    construct<PrintStmt>(
        "PRINT" >> format, defaulted("," >> nonemptyList(outputItem))))

// R1215 format -> default-char-expr | label | *
TYPE_PARSER(construct<Format>(label / !"_."_ch) ||
    construct<Format>(defaultCharExpr / !"="_tok) || construct<Format>(star))

// R1216 input-item -> variable | io-implied-do
TYPE_PARSER(construct<InputItem>(variable) ||
    construct<InputItem>(indirect(inputImpliedDo)))

// R1217 output-item -> expr | io-implied-do
TYPE_PARSER(construct<OutputItem>(expr) ||
    construct<OutputItem>(indirect(outputImpliedDo)))

// R1220 io-implied-do-control ->
//         do-variable = scalar-int-expr , scalar-int-expr [, scalar-int-expr]
constexpr auto ioImpliedDoControl{loopBounds(scalarIntExpr)};

// R1218 io-implied-do -> ( io-implied-do-object-list , io-implied-do-control )
// R1219 io-implied-do-object -> input-item | output-item
TYPE_CONTEXT_PARSER("input implied DO"_en_US,
    parenthesized(
        construct<InputImpliedDo>(nonemptyList(inputItem / lookAhead(","_tok)),
            "," >> ioImpliedDoControl)))
TYPE_CONTEXT_PARSER("output implied DO"_en_US,
    parenthesized(construct<OutputImpliedDo>(
        nonemptyList(outputItem / lookAhead(","_tok)),
        "," >> ioImpliedDoControl)))

// R1222 wait-stmt -> WAIT ( wait-spec-list )
TYPE_CONTEXT_PARSER("WAIT statement"_en_US,
    "WAIT" >>
        parenthesized(construct<WaitStmt>(nonemptyList(Parser<WaitSpec>{}))))

// R1223 wait-spec ->
//         [UNIT =] file-unit-number | END = label | EOR = label | ERR = label |
//         ID = scalar-int-expr | IOMSG = iomsg-variable |
//         IOSTAT = scalar-int-variable
constexpr auto idExpr{construct<IdExpr>(scalarIntExpr)};

TYPE_PARSER(first(construct<WaitSpec>(maybe("UNIT ="_tok) >> fileUnitNumber),
    construct<WaitSpec>("END =" >> endLabel),
    construct<WaitSpec>("EOR =" >> eorLabel),
    construct<WaitSpec>("ERR =" >> errLabel),
    construct<WaitSpec>("ID =" >> idExpr),
    construct<WaitSpec>("IOMSG =" >> msgVariable),
    construct<WaitSpec>("IOSTAT =" >> statVariable)))

template<typename A> common::IfNoLvalue<std::list<A>, A> singletonList(A &&x) {
  std::list<A> result;
  result.push_front(std::move(x));
  return result;
}
constexpr auto bareUnitNumberAsList{
    applyFunction(singletonList<PositionOrFlushSpec>,
        construct<PositionOrFlushSpec>(fileUnitNumber))};
constexpr auto positionOrFlushSpecList{
    parenthesized(nonemptyList(positionOrFlushSpec)) || bareUnitNumberAsList};

// R1224 backspace-stmt ->
//         BACKSPACE file-unit-number | BACKSPACE ( position-spec-list )
TYPE_CONTEXT_PARSER("BACKSPACE statement"_en_US,
    construct<BackspaceStmt>("BACKSPACE" >> positionOrFlushSpecList))

// R1225 endfile-stmt ->
//         ENDFILE file-unit-number | ENDFILE ( position-spec-list )
TYPE_CONTEXT_PARSER("ENDFILE statement"_en_US,
    construct<EndfileStmt>("ENDFILE" >> positionOrFlushSpecList))

// R1226 rewind-stmt -> REWIND file-unit-number | REWIND ( position-spec-list )
TYPE_CONTEXT_PARSER("REWIND statement"_en_US,
    construct<RewindStmt>("REWIND" >> positionOrFlushSpecList))

// R1227 position-spec ->
//         [UNIT =] file-unit-number | IOMSG = iomsg-variable |
//         IOSTAT = scalar-int-variable | ERR = label
// R1229 flush-spec ->
//         [UNIT =] file-unit-number | IOSTAT = scalar-int-variable |
//         IOMSG = iomsg-variable | ERR = label
TYPE_PARSER(
    construct<PositionOrFlushSpec>(maybe("UNIT ="_tok) >> fileUnitNumber) ||
    construct<PositionOrFlushSpec>("IOMSG =" >> msgVariable) ||
    construct<PositionOrFlushSpec>("IOSTAT =" >> statVariable) ||
    construct<PositionOrFlushSpec>("ERR =" >> errLabel))

// R1228 flush-stmt -> FLUSH file-unit-number | FLUSH ( flush-spec-list )
TYPE_CONTEXT_PARSER("FLUSH statement"_en_US,
    construct<FlushStmt>("FLUSH" >> positionOrFlushSpecList))

// R1231 inquire-spec ->
//         [UNIT =] file-unit-number | FILE = file-name-expr |
//         ACCESS = scalar-default-char-variable |
//         ACTION = scalar-default-char-variable |
//         ASYNCHRONOUS = scalar-default-char-variable |
//         BLANK = scalar-default-char-variable |
//         DECIMAL = scalar-default-char-variable |
//         DELIM = scalar-default-char-variable |
//         ENCODING = scalar-default-char-variable |
//         ERR = label | EXIST = scalar-logical-variable |
//         FORM = scalar-default-char-variable |
//         FORMATTED = scalar-default-char-variable |
//         ID = scalar-int-expr | IOMSG = iomsg-variable |
//         IOSTAT = scalar-int-variable |
//         NAME = scalar-default-char-variable |
//         NAMED = scalar-logical-variable |
//         NEXTREC = scalar-int-variable | NUMBER = scalar-int-variable |
//         OPENED = scalar-logical-variable |
//         PAD = scalar-default-char-variable |
//         PENDING = scalar-logical-variable | POS = scalar-int-variable |
//         POSITION = scalar-default-char-variable |
//         READ = scalar-default-char-variable |
//         READWRITE = scalar-default-char-variable |
//         RECL = scalar-int-variable | ROUND = scalar-default-char-variable |
//         SEQUENTIAL = scalar-default-char-variable |
//         SIGN = scalar-default-char-variable |
//         SIZE = scalar-int-variable |
//         STREAM = scalar-default-char-variable |
//         STATUS = scalar-default-char-variable |
//         WRITE = scalar-default-char-variable
//         @ | CONVERT = scalar-default-char-variable
//           | DISPOSE = scalar-default-char-variable
TYPE_PARSER(first(construct<InquireSpec>(maybe("UNIT ="_tok) >> fileUnitNumber),
    construct<InquireSpec>("FILE =" >> fileNameExpr),
    construct<InquireSpec>(
        "ACCESS =" >> construct<InquireSpec::CharVar>(
                          pure(InquireSpec::CharVar::Kind::Access),
                          scalarDefaultCharVariable)),
    construct<InquireSpec>(
        "ACTION =" >> construct<InquireSpec::CharVar>(
                          pure(InquireSpec::CharVar::Kind::Action),
                          scalarDefaultCharVariable)),
    construct<InquireSpec>(
        "ASYNCHRONOUS =" >> construct<InquireSpec::CharVar>(
                                pure(InquireSpec::CharVar::Kind::Asynchronous),
                                scalarDefaultCharVariable)),
    construct<InquireSpec>("BLANK =" >>
        construct<InquireSpec::CharVar>(pure(InquireSpec::CharVar::Kind::Blank),
            scalarDefaultCharVariable)),
    construct<InquireSpec>(
        "DECIMAL =" >> construct<InquireSpec::CharVar>(
                           pure(InquireSpec::CharVar::Kind::Decimal),
                           scalarDefaultCharVariable)),
    construct<InquireSpec>("DELIM =" >>
        construct<InquireSpec::CharVar>(pure(InquireSpec::CharVar::Kind::Delim),
            scalarDefaultCharVariable)),
    construct<InquireSpec>(
        "DIRECT =" >> construct<InquireSpec::CharVar>(
                          pure(InquireSpec::CharVar::Kind::Direct),
                          scalarDefaultCharVariable)),
    construct<InquireSpec>(
        "ENCODING =" >> construct<InquireSpec::CharVar>(
                            pure(InquireSpec::CharVar::Kind::Encoding),
                            scalarDefaultCharVariable)),
    construct<InquireSpec>("ERR =" >> errLabel),
    construct<InquireSpec>("EXIST =" >>
        construct<InquireSpec::LogVar>(
            pure(InquireSpec::LogVar::Kind::Exist), scalarLogicalVariable)),
    construct<InquireSpec>("FORM =" >>
        construct<InquireSpec::CharVar>(
            pure(InquireSpec::CharVar::Kind::Form), scalarDefaultCharVariable)),
    construct<InquireSpec>(
        "FORMATTED =" >> construct<InquireSpec::CharVar>(
                             pure(InquireSpec::CharVar::Kind::Formatted),
                             scalarDefaultCharVariable)),
    construct<InquireSpec>("ID =" >> idExpr),
    construct<InquireSpec>("IOMSG =" >>
        construct<InquireSpec::CharVar>(pure(InquireSpec::CharVar::Kind::Iomsg),
            scalarDefaultCharVariable)),
    construct<InquireSpec>("IOSTAT =" >>
        construct<InquireSpec::IntVar>(pure(InquireSpec::IntVar::Kind::Iostat),
            scalar(integer(variable)))),
    construct<InquireSpec>("NAME =" >>
        construct<InquireSpec::CharVar>(
            pure(InquireSpec::CharVar::Kind::Name), scalarDefaultCharVariable)),
    construct<InquireSpec>("NAMED =" >>
        construct<InquireSpec::LogVar>(
            pure(InquireSpec::LogVar::Kind::Named), scalarLogicalVariable)),
    construct<InquireSpec>("NEXTREC =" >>
        construct<InquireSpec::IntVar>(pure(InquireSpec::IntVar::Kind::Nextrec),
            scalar(integer(variable)))),
    construct<InquireSpec>("NUMBER =" >>
        construct<InquireSpec::IntVar>(pure(InquireSpec::IntVar::Kind::Number),
            scalar(integer(variable)))),
    construct<InquireSpec>("OPENED =" >>
        construct<InquireSpec::LogVar>(
            pure(InquireSpec::LogVar::Kind::Opened), scalarLogicalVariable)),
    construct<InquireSpec>("PAD =" >>
        construct<InquireSpec::CharVar>(
            pure(InquireSpec::CharVar::Kind::Pad), scalarDefaultCharVariable)),
    construct<InquireSpec>("PENDING =" >>
        construct<InquireSpec::LogVar>(
            pure(InquireSpec::LogVar::Kind::Pending), scalarLogicalVariable)),
    construct<InquireSpec>("POS =" >>
        construct<InquireSpec::IntVar>(
            pure(InquireSpec::IntVar::Kind::Pos), scalar(integer(variable)))),
    construct<InquireSpec>(
        "POSITION =" >> construct<InquireSpec::CharVar>(
                            pure(InquireSpec::CharVar::Kind::Position),
                            scalarDefaultCharVariable)),
    construct<InquireSpec>("READ =" >>
        construct<InquireSpec::CharVar>(
            pure(InquireSpec::CharVar::Kind::Read), scalarDefaultCharVariable)),
    construct<InquireSpec>(
        "READWRITE =" >> construct<InquireSpec::CharVar>(
                             pure(InquireSpec::CharVar::Kind::Readwrite),
                             scalarDefaultCharVariable)),
    construct<InquireSpec>("RECL =" >>
        construct<InquireSpec::IntVar>(
            pure(InquireSpec::IntVar::Kind::Recl), scalar(integer(variable)))),
    construct<InquireSpec>("ROUND =" >>
        construct<InquireSpec::CharVar>(pure(InquireSpec::CharVar::Kind::Round),
            scalarDefaultCharVariable)),
    construct<InquireSpec>(
        "SEQUENTIAL =" >> construct<InquireSpec::CharVar>(
                              pure(InquireSpec::CharVar::Kind::Sequential),
                              scalarDefaultCharVariable)),
    construct<InquireSpec>("SIGN =" >>
        construct<InquireSpec::CharVar>(
            pure(InquireSpec::CharVar::Kind::Sign), scalarDefaultCharVariable)),
    construct<InquireSpec>("SIZE =" >>
        construct<InquireSpec::IntVar>(
            pure(InquireSpec::IntVar::Kind::Size), scalar(integer(variable)))),
    construct<InquireSpec>(
        "STREAM =" >> construct<InquireSpec::CharVar>(
                          pure(InquireSpec::CharVar::Kind::Stream),
                          scalarDefaultCharVariable)),
    construct<InquireSpec>(
        "STATUS =" >> construct<InquireSpec::CharVar>(
                          pure(InquireSpec::CharVar::Kind::Status),
                          scalarDefaultCharVariable)),
    construct<InquireSpec>(
        "UNFORMATTED =" >> construct<InquireSpec::CharVar>(
                               pure(InquireSpec::CharVar::Kind::Unformatted),
                               scalarDefaultCharVariable)),
    construct<InquireSpec>("WRITE =" >>
        construct<InquireSpec::CharVar>(pure(InquireSpec::CharVar::Kind::Write),
            scalarDefaultCharVariable)),
    extension<LanguageFeature::Convert>(construct<InquireSpec>(
        "CONVERT =" >> construct<InquireSpec::CharVar>(
                           pure(InquireSpec::CharVar::Kind::Convert),
                           scalarDefaultCharVariable))),
    extension<LanguageFeature::Dispose>(construct<InquireSpec>(
        "DISPOSE =" >> construct<InquireSpec::CharVar>(
                           pure(InquireSpec::CharVar::Kind::Dispose),
                           scalarDefaultCharVariable)))))

// R1230 inquire-stmt ->
//         INQUIRE ( inquire-spec-list ) |
//         INQUIRE ( IOLENGTH = scalar-int-variable ) output-item-list
TYPE_CONTEXT_PARSER("INQUIRE statement"_en_US,
    "INQUIRE" >>
        (construct<InquireStmt>(
             parenthesized(nonemptyList(Parser<InquireSpec>{}))) ||
            construct<InquireStmt>(construct<InquireStmt::Iolength>(
                parenthesized("IOLENGTH =" >> scalar(integer(variable))),
                nonemptyList(outputItem)))))

// R1301 format-stmt -> FORMAT format-specification
// 13.2.1 allows spaces to appear "at any point" within a format specification
// without effect, except of course within a character string edit descriptor.
TYPE_CONTEXT_PARSER("FORMAT statement"_en_US,
    construct<FormatStmt>("FORMAT" >> Parser<format::FormatSpecification>{}))

// R1321 char-string-edit-desc
// N.B. C1313 disallows any kind parameter on the character literal.
constexpr auto charStringEditDesc{
    space >> (charLiteralConstantWithoutKind || rawHollerithLiteral)};

// R1303 format-items -> format-item [[,] format-item]...
constexpr auto formatItems{
    nonemptySeparated(space >> Parser<format::FormatItem>{}, maybe(","_tok))};

// R1306 r -> digit-string
constexpr DigitStringIgnoreSpaces repeat;

// R1304 format-item ->
//         [r] data-edit-desc | control-edit-desc | char-string-edit-desc |
//         [r] ( format-items )
TYPE_PARSER(construct<format::FormatItem>(
                maybe(repeat), Parser<format::IntrinsicTypeDataEditDesc>{}) ||
    construct<format::FormatItem>(
        maybe(repeat), Parser<format::DerivedTypeDataEditDesc>{}) ||
    construct<format::FormatItem>(Parser<format::ControlEditDesc>{}) ||
    construct<format::FormatItem>(charStringEditDesc) ||
    construct<format::FormatItem>(maybe(repeat), parenthesized(formatItems)))

// R1302 format-specification ->
//         ( [format-items] ) | ( [format-items ,] unlimited-format-item )
// R1305 unlimited-format-item -> * ( format-items )
// minor extension: the comma is optional before the unlimited-format-item
TYPE_PARSER(parenthesized(construct<format::FormatSpecification>(
                              defaulted(formatItems / maybe(","_tok)),
                              "*" >> parenthesized(formatItems)) ||
    construct<format::FormatSpecification>(defaulted(formatItems))))
// R1308 w -> digit-string
// R1309 m -> digit-string
// R1310 d -> digit-string
// R1311 e -> digit-string
constexpr auto width{repeat};
constexpr auto mandatoryWidth{construct<std::optional<int>>(width)};
constexpr auto digits{repeat};
constexpr auto noInt{construct<std::optional<int>>()};
constexpr auto mandatoryDigits{construct<std::optional<int>>("." >> width)};

// R1307 data-edit-desc ->
//         I w [. m] | B w [. m] | O w [. m] | Z w [. m] | F w . d |
//         E w . d [E e] | EN w . d [E e] | ES w . d [E e] | EX w . d [E e] |
//         G w [. d [E e]] | L w | A [w] | D w . d |
//         DT [char-literal-constant] [( v-list )]
// (part 1 of 2)
TYPE_PARSER(construct<format::IntrinsicTypeDataEditDesc>(
                "I" >> pure(format::IntrinsicTypeDataEditDesc::Kind::I) ||
                    "B" >> pure(format::IntrinsicTypeDataEditDesc::Kind::B) ||
                    "O" >> pure(format::IntrinsicTypeDataEditDesc::Kind::O) ||
                    "Z" >> pure(format::IntrinsicTypeDataEditDesc::Kind::Z),
                mandatoryWidth, maybe("." >> digits), noInt) ||
    construct<format::IntrinsicTypeDataEditDesc>(
        "F" >> pure(format::IntrinsicTypeDataEditDesc::Kind::F) ||
            "D" >> pure(format::IntrinsicTypeDataEditDesc::Kind::D),
        mandatoryWidth, mandatoryDigits, noInt) ||
    construct<format::IntrinsicTypeDataEditDesc>(
        "E" >> ("N" >> pure(format::IntrinsicTypeDataEditDesc::Kind::EN) ||
                   "S" >> pure(format::IntrinsicTypeDataEditDesc::Kind::ES) ||
                   "X" >> pure(format::IntrinsicTypeDataEditDesc::Kind::EX) ||
                   pure(format::IntrinsicTypeDataEditDesc::Kind::E)),
        mandatoryWidth, mandatoryDigits, maybe("E" >> digits)) ||
    construct<format::IntrinsicTypeDataEditDesc>(
        "G" >> pure(format::IntrinsicTypeDataEditDesc::Kind::G), mandatoryWidth,
        mandatoryDigits, maybe("E" >> digits)) ||
    construct<format::IntrinsicTypeDataEditDesc>(
        "G" >> pure(format::IntrinsicTypeDataEditDesc::Kind::G) ||
            "L" >> pure(format::IntrinsicTypeDataEditDesc::Kind::L),
        mandatoryWidth, noInt, noInt) ||
    construct<format::IntrinsicTypeDataEditDesc>(
        "A" >> pure(format::IntrinsicTypeDataEditDesc::Kind::A), maybe(width),
        noInt, noInt) ||
    // PGI/Intel extension: omitting width (and all else that follows)
    extension<LanguageFeature::AbbreviatedEditDescriptor>(
        construct<format::IntrinsicTypeDataEditDesc>(
            "I" >> pure(format::IntrinsicTypeDataEditDesc::Kind::I) ||
                ("B"_tok / !letter /* don't occlude BN & BZ */) >>
                    pure(format::IntrinsicTypeDataEditDesc::Kind::B) ||
                "O" >> pure(format::IntrinsicTypeDataEditDesc::Kind::O) ||
                "Z" >> pure(format::IntrinsicTypeDataEditDesc::Kind::Z) ||
                "F" >> pure(format::IntrinsicTypeDataEditDesc::Kind::F) ||
                ("D"_tok / !letter /* don't occlude DT, DC, & DP */) >>
                    pure(format::IntrinsicTypeDataEditDesc::Kind::D) ||
                "E" >>
                    ("N" >> pure(format::IntrinsicTypeDataEditDesc::Kind::EN) ||
                        "S" >>
                            pure(format::IntrinsicTypeDataEditDesc::Kind::ES) ||
                        "X" >>
                            pure(format::IntrinsicTypeDataEditDesc::Kind::EX) ||
                        pure(format::IntrinsicTypeDataEditDesc::Kind::E)) ||
                "G" >> pure(format::IntrinsicTypeDataEditDesc::Kind::G) ||
                "L" >> pure(format::IntrinsicTypeDataEditDesc::Kind::L),
            noInt, noInt, noInt)))

// R1307 data-edit-desc (part 2 of 2)
// R1312 v -> [sign] digit-string
constexpr SignedDigitStringIgnoreSpaces scaleFactor;
TYPE_PARSER(construct<format::DerivedTypeDataEditDesc>(
    "D" >> "T"_tok >> defaulted(charLiteralConstantWithoutKind),
    defaulted(parenthesized(nonemptyList(scaleFactor)))))

// R1314 k -> [sign] digit-string
constexpr PositiveDigitStringIgnoreSpaces count;

// R1313 control-edit-desc ->
//         position-edit-desc | [r] / | : | sign-edit-desc | k P |
//         blank-interp-edit-desc | round-edit-desc | decimal-edit-desc |
//         @ \ | $
// R1315 position-edit-desc -> T n | TL n | TR n | n X
// R1316 n -> digit-string
// R1317 sign-edit-desc -> SS | SP | S
// R1318 blank-interp-edit-desc -> BN | BZ
// R1319 round-edit-desc -> RU | RD | RZ | RN | RC | RP
// R1320 decimal-edit-desc -> DC | DP
TYPE_PARSER(construct<format::ControlEditDesc>(
                "T" >> ("L" >> pure(format::ControlEditDesc::Kind::TL) ||
                           "R" >> pure(format::ControlEditDesc::Kind::TR) ||
                           pure(format::ControlEditDesc::Kind::T)),
                count) ||
    construct<format::ControlEditDesc>(count,
        "X" >> pure(format::ControlEditDesc::Kind::X) ||
            "/" >> pure(format::ControlEditDesc::Kind::Slash)) ||
    construct<format::ControlEditDesc>(
        "X" >> pure(format::ControlEditDesc::Kind::X) ||
        "/" >> pure(format::ControlEditDesc::Kind::Slash)) ||
    construct<format::ControlEditDesc>(
        scaleFactor, "P" >> pure(format::ControlEditDesc::Kind::P)) ||
    construct<format::ControlEditDesc>(
        ":" >> pure(format::ControlEditDesc::Kind::Colon)) ||
    "S" >> ("S" >> construct<format::ControlEditDesc>(
                       pure(format::ControlEditDesc::Kind::SS)) ||
               "P" >> construct<format::ControlEditDesc>(
                          pure(format::ControlEditDesc::Kind::SP)) ||
               construct<format::ControlEditDesc>(
                   pure(format::ControlEditDesc::Kind::S))) ||
    "B" >> ("N" >> construct<format::ControlEditDesc>(
                       pure(format::ControlEditDesc::Kind::BN)) ||
               "Z" >> construct<format::ControlEditDesc>(
                          pure(format::ControlEditDesc::Kind::BZ))) ||
    "R" >> ("U" >> construct<format::ControlEditDesc>(
                       pure(format::ControlEditDesc::Kind::RU)) ||
               "D" >> construct<format::ControlEditDesc>(
                          pure(format::ControlEditDesc::Kind::RD)) ||
               "Z" >> construct<format::ControlEditDesc>(
                          pure(format::ControlEditDesc::Kind::RZ)) ||
               "N" >> construct<format::ControlEditDesc>(
                          pure(format::ControlEditDesc::Kind::RN)) ||
               "C" >> construct<format::ControlEditDesc>(
                          pure(format::ControlEditDesc::Kind::RC)) ||
               "P" >> construct<format::ControlEditDesc>(
                          pure(format::ControlEditDesc::Kind::RP))) ||
    "D" >> ("C" >> construct<format::ControlEditDesc>(
                       pure(format::ControlEditDesc::Kind::DC)) ||
               "P" >> construct<format::ControlEditDesc>(
                          pure(format::ControlEditDesc::Kind::DP))) ||
    extension<LanguageFeature::AdditionalFormats>(
        "$" >> construct<format::ControlEditDesc>(
                   pure(format::ControlEditDesc::Kind::Dollar)) ||
        "\\" >> construct<format::ControlEditDesc>(
                    pure(format::ControlEditDesc::Kind::Backslash))))

// R1401 main-program ->
//         [program-stmt] [specification-part] [execution-part]
//         [internal-subprogram-part] end-program-stmt
TYPE_CONTEXT_PARSER("main program"_en_US,
    construct<MainProgram>(maybe(statement(Parser<ProgramStmt>{})),
        specificationPart, executionPart, maybe(internalSubprogramPart),
        unterminatedStatement(Parser<EndProgramStmt>{})))

// R1402 program-stmt -> PROGRAM program-name
// PGI allows empty parentheses after the name.
TYPE_CONTEXT_PARSER("PROGRAM statement"_en_US,
    construct<ProgramStmt>("PROGRAM" >> name /
            maybe(extension<LanguageFeature::ProgramParentheses>(
                parenthesized(ok)))))

// R1403 end-program-stmt -> END [PROGRAM [program-name]]
TYPE_CONTEXT_PARSER("END PROGRAM statement"_en_US,
    construct<EndProgramStmt>(recovery(
        "END PROGRAM" >> maybe(name) || bareEnd, progUnitEndStmtErrorRecovery)))

// R1404 module ->
//         module-stmt [specification-part] [module-subprogram-part]
//         end-module-stmt
TYPE_CONTEXT_PARSER("module"_en_US,
    construct<Module>(statement(Parser<ModuleStmt>{}), limitedSpecificationPart,
        maybe(Parser<ModuleSubprogramPart>{}),
        unterminatedStatement(Parser<EndModuleStmt>{})))

// R1405 module-stmt -> MODULE module-name
TYPE_CONTEXT_PARSER(
    "MODULE statement"_en_US, construct<ModuleStmt>("MODULE" >> name))

// R1406 end-module-stmt -> END [MODULE [module-name]]
TYPE_CONTEXT_PARSER("END MODULE statement"_en_US,
    construct<EndModuleStmt>(recovery(
        "END MODULE" >> maybe(name) || bareEnd, progUnitEndStmtErrorRecovery)))

// R1407 module-subprogram-part -> contains-stmt [module-subprogram]...
TYPE_CONTEXT_PARSER("module subprogram part"_en_US,
    construct<ModuleSubprogramPart>(statement(containsStmt),
        many(StartNewSubprogram{} >> Parser<ModuleSubprogram>{})))

// R1408 module-subprogram ->
//         function-subprogram | subroutine-subprogram |
//         separate-module-subprogram
TYPE_PARSER(construct<ModuleSubprogram>(indirect(functionSubprogram)) ||
    construct<ModuleSubprogram>(indirect(subroutineSubprogram)) ||
    construct<ModuleSubprogram>(indirect(Parser<SeparateModuleSubprogram>{})))

// R1410 module-nature -> INTRINSIC | NON_INTRINSIC
constexpr auto moduleNature{
    "INTRINSIC" >> pure(UseStmt::ModuleNature::Intrinsic) ||
    "NON_INTRINSIC" >> pure(UseStmt::ModuleNature::Non_Intrinsic)};

// R1409 use-stmt ->
//         USE [[, module-nature] ::] module-name [, rename-list] |
//         USE [[, module-nature] ::] module-name , ONLY : [only-list]
TYPE_PARSER(construct<UseStmt>("USE" >> optionalBeforeColons(moduleNature),
                name, ", ONLY :" >> optionalList(Parser<Only>{})) ||
    construct<UseStmt>("USE" >> optionalBeforeColons(moduleNature), name,
        defaulted("," >>
            nonemptyList("expected renamings"_err_en_US, Parser<Rename>{})) /
            lookAhead(endOfStmt)))

// R1411 rename ->
//         local-name => use-name |
//         OPERATOR ( local-defined-operator ) =>
//           OPERATOR ( use-defined-operator )
TYPE_PARSER(construct<Rename>("OPERATOR (" >>
                construct<Rename::Operators>(
                    definedOpName / ") => OPERATOR (", definedOpName / ")")) ||
    construct<Rename>(construct<Rename::Names>(name, "=>" >> name)))

// R1412 only -> generic-spec | only-use-name | rename
// R1413 only-use-name -> use-name
TYPE_PARSER(construct<Only>(Parser<Rename>{}) ||
    construct<Only>(indirect(genericSpec)) ||
    construct<Only>(name))  // TODO: ambiguous, accepted by genericSpec

// R1416 submodule ->
//         submodule-stmt [specification-part] [module-subprogram-part]
//         end-submodule-stmt
TYPE_CONTEXT_PARSER("submodule"_en_US,
    construct<Submodule>(statement(Parser<SubmoduleStmt>{}),
        limitedSpecificationPart, maybe(Parser<ModuleSubprogramPart>{}),
        unterminatedStatement(Parser<EndSubmoduleStmt>{})))

// R1417 submodule-stmt -> SUBMODULE ( parent-identifier ) submodule-name
TYPE_CONTEXT_PARSER("SUBMODULE statement"_en_US,
    construct<SubmoduleStmt>(
        "SUBMODULE" >> parenthesized(Parser<ParentIdentifier>{}), name))

// R1418 parent-identifier -> ancestor-module-name [: parent-submodule-name]
TYPE_PARSER(construct<ParentIdentifier>(name, maybe(":" >> name)))

// R1419 end-submodule-stmt -> END [SUBMODULE [submodule-name]]
TYPE_CONTEXT_PARSER("END SUBMODULE statement"_en_US,
    construct<EndSubmoduleStmt>(
        recovery("END SUBMODULE" >> maybe(name) || bareEnd,
            progUnitEndStmtErrorRecovery)))

// R1420 block-data -> block-data-stmt [specification-part] end-block-data-stmt
TYPE_CONTEXT_PARSER("BLOCK DATA subprogram"_en_US,
    construct<BlockData>(statement(Parser<BlockDataStmt>{}),
        limitedSpecificationPart,
        unterminatedStatement(Parser<EndBlockDataStmt>{})))

// R1421 block-data-stmt -> BLOCK DATA [block-data-name]
TYPE_CONTEXT_PARSER("BLOCK DATA statement"_en_US,
    construct<BlockDataStmt>("BLOCK DATA" >> maybe(name)))

// R1422 end-block-data-stmt -> END [BLOCK DATA [block-data-name]]
TYPE_CONTEXT_PARSER("END BLOCK DATA statement"_en_US,
    construct<EndBlockDataStmt>(
        recovery("END BLOCK DATA" >> maybe(name) || bareEnd,
            progUnitEndStmtErrorRecovery)))

// R1501 interface-block ->
//         interface-stmt [interface-specification]... end-interface-stmt
TYPE_PARSER(construct<InterfaceBlock>(statement(Parser<InterfaceStmt>{}),
    many(Parser<InterfaceSpecification>{}),
    statement(Parser<EndInterfaceStmt>{})))

// R1502 interface-specification -> interface-body | procedure-stmt
TYPE_PARSER(construct<InterfaceSpecification>(Parser<InterfaceBody>{}) ||
    construct<InterfaceSpecification>(statement(Parser<ProcedureStmt>{})))

// R1503 interface-stmt -> INTERFACE [generic-spec] | ABSTRACT INTERFACE
TYPE_PARSER(construct<InterfaceStmt>("INTERFACE" >> maybe(genericSpec)) ||
    construct<InterfaceStmt>(construct<Abstract>("ABSTRACT INTERFACE"_sptok)))

// R1504 end-interface-stmt -> END INTERFACE [generic-spec]
TYPE_PARSER(construct<EndInterfaceStmt>("END INTERFACE" >> maybe(genericSpec)))

// R1505 interface-body ->
//         function-stmt [specification-part] end-function-stmt |
//         subroutine-stmt [specification-part] end-subroutine-stmt
TYPE_CONTEXT_PARSER("interface body"_en_US,
    construct<InterfaceBody>(
        construct<InterfaceBody::Function>(statement(functionStmt),
            indirect(limitedSpecificationPart), statement(endFunctionStmt))) ||
        construct<InterfaceBody>(construct<InterfaceBody::Subroutine>(
            statement(subroutineStmt), indirect(limitedSpecificationPart),
            statement(endSubroutineStmt))))

// R1507 specific-procedure -> procedure-name
constexpr auto specificProcedures{
    nonemptyList("expected specific procedure names"_err_en_US, name)};

// R1506 procedure-stmt -> [MODULE] PROCEDURE [::] specific-procedure-list
TYPE_PARSER(construct<ProcedureStmt>("MODULE PROCEDURE"_sptok >>
                    pure(ProcedureStmt::Kind::ModuleProcedure),
                maybe("::"_tok) >> specificProcedures) ||
    construct<ProcedureStmt>(
        "PROCEDURE" >> pure(ProcedureStmt::Kind::Procedure),
        maybe("::"_tok) >> specificProcedures))

// R1508 generic-spec ->
//         generic-name | OPERATOR ( defined-operator ) |
//         ASSIGNMENT ( = ) | defined-io-generic-spec
// R1509 defined-io-generic-spec ->
//         READ ( FORMATTED ) | READ ( UNFORMATTED ) |
//         WRITE ( FORMATTED ) | WRITE ( UNFORMATTED )
TYPE_PARSER(sourced(first(construct<GenericSpec>("OPERATOR" >>
                              parenthesized(Parser<DefinedOperator>{})),
    construct<GenericSpec>(
        construct<GenericSpec::Assignment>("ASSIGNMENT ( = )"_tok)),
    construct<GenericSpec>(
        construct<GenericSpec::ReadFormatted>("READ ( FORMATTED )"_tok)),
    construct<GenericSpec>(
        construct<GenericSpec::ReadUnformatted>("READ ( UNFORMATTED )"_tok)),
    construct<GenericSpec>(
        construct<GenericSpec::WriteFormatted>("WRITE ( FORMATTED )"_tok)),
    construct<GenericSpec>(
        construct<GenericSpec::WriteUnformatted>("WRITE ( UNFORMATTED )"_tok)),
    construct<GenericSpec>(name))))

// R1510 generic-stmt ->
//         GENERIC [, access-spec] :: generic-spec => specific-procedure-list
TYPE_PARSER(construct<GenericStmt>("GENERIC" >> maybe("," >> accessSpec),
    "::" >> genericSpec, "=>" >> specificProcedures))

// R1511 external-stmt -> EXTERNAL [::] external-name-list
TYPE_PARSER(
    "EXTERNAL" >> maybe("::"_tok) >> construct<ExternalStmt>(listOfNames))

// R1512 procedure-declaration-stmt ->
//         PROCEDURE ( [proc-interface] ) [[, proc-attr-spec]... ::]
//         proc-decl-list
TYPE_PARSER("PROCEDURE" >>
    construct<ProcedureDeclarationStmt>(parenthesized(maybe(procInterface)),
        optionalListBeforeColons(Parser<ProcAttrSpec>{}),
        nonemptyList("expected procedure declarations"_err_en_US, procDecl)))

// R1513 proc-interface -> interface-name | declaration-type-spec
// R1516 interface-name -> name
// N.B. Simple names of intrinsic types (e.g., "REAL") are not
// ambiguous here - they take precedence over derived type names
// thanks to C1516.
TYPE_PARSER(
    construct<ProcInterface>(declarationTypeSpec / lookAhead(")"_tok)) ||
    construct<ProcInterface>(name))

// R1514 proc-attr-spec ->
//         access-spec | proc-language-binding-spec | INTENT ( intent-spec ) |
//         OPTIONAL | POINTER | PROTECTED | SAVE
TYPE_PARSER(construct<ProcAttrSpec>(accessSpec) ||
    construct<ProcAttrSpec>(languageBindingSpec) ||
    construct<ProcAttrSpec>("INTENT" >> parenthesized(intentSpec)) ||
    construct<ProcAttrSpec>(optional) || construct<ProcAttrSpec>(pointer) ||
    construct<ProcAttrSpec>(protectedAttr) || construct<ProcAttrSpec>(save))

// R1515 proc-decl -> procedure-entity-name [=> proc-pointer-init]
TYPE_PARSER(construct<ProcDecl>(name, maybe("=>" >> Parser<ProcPointerInit>{})))

// R1517 proc-pointer-init -> null-init | initial-proc-target
// R1518 initial-proc-target -> procedure-name
TYPE_PARSER(
    construct<ProcPointerInit>(nullInit) || construct<ProcPointerInit>(name))

// R1519 intrinsic-stmt -> INTRINSIC [::] intrinsic-procedure-name-list
TYPE_PARSER(
    "INTRINSIC" >> maybe("::"_tok) >> construct<IntrinsicStmt>(listOfNames))

// R1520 function-reference -> procedure-designator ( [actual-arg-spec-list] )
TYPE_CONTEXT_PARSER("function reference"_en_US,
    construct<FunctionReference>(
        sourced(construct<Call>(Parser<ProcedureDesignator>{},
            parenthesized(optionalList(actualArgSpec))))) /
        !"["_tok)

// R1521 call-stmt -> CALL procedure-designator [( [actual-arg-spec-list] )]
TYPE_PARSER(construct<CallStmt>(
    sourced(construct<Call>("CALL" >> Parser<ProcedureDesignator>{},
        defaulted(parenthesized(optionalList(actualArgSpec)))))))

// R1522 procedure-designator ->
//         procedure-name | proc-component-ref | data-ref % binding-name
TYPE_PARSER(construct<ProcedureDesignator>(Parser<ProcComponentRef>{}) ||
    construct<ProcedureDesignator>(name))

// R1523 actual-arg-spec -> [keyword =] actual-arg
TYPE_PARSER(construct<ActualArgSpec>(
    maybe(keyword / "=" / !"="_ch), Parser<ActualArg>{}))

// R1524 actual-arg ->
//         expr | variable | procedure-name | proc-component-ref |
//         alt-return-spec
// N.B. the "procedure-name" and "proc-component-ref" alternatives can't
// yet be distinguished from "variable", many instances of which can't be
// distinguished from "expr" anyway (to do so would misparse structure
// constructors and function calls as array elements).
// Semantics sorts it all out later.
TYPE_PARSER(construct<ActualArg>(expr) ||
    construct<ActualArg>(Parser<AltReturnSpec>{}) ||
    extension<LanguageFeature::PercentRefAndVal>(construct<ActualArg>(
        construct<ActualArg::PercentRef>("%REF" >> parenthesized(variable)))) ||
    extension<LanguageFeature::PercentRefAndVal>(construct<ActualArg>(
        construct<ActualArg::PercentVal>("%VAL" >> parenthesized(expr)))))

// R1525 alt-return-spec -> * label
TYPE_PARSER(construct<AltReturnSpec>(star >> label))

// R1527 prefix-spec ->
//         declaration-type-spec | ELEMENTAL | IMPURE | MODULE |
//         NON_RECURSIVE | PURE | RECURSIVE
TYPE_PARSER(first(construct<PrefixSpec>(declarationTypeSpec),
    construct<PrefixSpec>(construct<PrefixSpec::Elemental>("ELEMENTAL"_tok)),
    construct<PrefixSpec>(construct<PrefixSpec::Impure>("IMPURE"_tok)),
    construct<PrefixSpec>(construct<PrefixSpec::Module>("MODULE"_tok)),
    construct<PrefixSpec>(
        construct<PrefixSpec::Non_Recursive>("NON_RECURSIVE"_tok)),
    construct<PrefixSpec>(construct<PrefixSpec::Pure>("PURE"_tok)),
    construct<PrefixSpec>(construct<PrefixSpec::Recursive>("RECURSIVE"_tok))))

// R1529 function-subprogram ->
//         function-stmt [specification-part] [execution-part]
//         [internal-subprogram-part] end-function-stmt
TYPE_CONTEXT_PARSER("FUNCTION subprogram"_en_US,
    construct<FunctionSubprogram>(statement(functionStmt), specificationPart,
        executionPart, maybe(internalSubprogramPart),
        unterminatedStatement(endFunctionStmt)))

// R1530 function-stmt ->
//         [prefix] FUNCTION function-name ( [dummy-arg-name-list] ) [suffix]
// R1526 prefix -> prefix-spec [prefix-spec]...
// R1531 dummy-arg-name -> name
TYPE_CONTEXT_PARSER("FUNCTION statement"_en_US,
    construct<FunctionStmt>(many(prefixSpec), "FUNCTION" >> name,
        parenthesized(optionalList(name)), maybe(suffix)) ||
        extension<LanguageFeature::OmitFunctionDummies>(
            construct<FunctionStmt>(  // PGI & Intel accept "FUNCTION F"
                many(prefixSpec), "FUNCTION" >> name,
                construct<std::list<Name>>(),
                construct<std::optional<Suffix>>())))

// R1532 suffix ->
//         proc-language-binding-spec [RESULT ( result-name )] |
//         RESULT ( result-name ) [proc-language-binding-spec]
TYPE_PARSER(construct<Suffix>(
                languageBindingSpec, maybe("RESULT" >> parenthesized(name))) ||
    construct<Suffix>(
        "RESULT" >> parenthesized(name), maybe(languageBindingSpec)))

// R1533 end-function-stmt -> END [FUNCTION [function-name]]
TYPE_PARSER(construct<EndFunctionStmt>(recovery(
    "END FUNCTION" >> maybe(name) || bareEnd, progUnitEndStmtErrorRecovery)))

// R1534 subroutine-subprogram ->
//         subroutine-stmt [specification-part] [execution-part]
//         [internal-subprogram-part] end-subroutine-stmt
TYPE_CONTEXT_PARSER("SUBROUTINE subprogram"_en_US,
    construct<SubroutineSubprogram>(statement(subroutineStmt),
        specificationPart, executionPart, maybe(internalSubprogramPart),
        unterminatedStatement(endSubroutineStmt)))

// R1535 subroutine-stmt ->
//         [prefix] SUBROUTINE subroutine-name [( [dummy-arg-list] )
//         [proc-language-binding-spec]]
TYPE_PARSER(
    construct<SubroutineStmt>(many(prefixSpec), "SUBROUTINE" >> name,
        parenthesized(optionalList(dummyArg)), maybe(languageBindingSpec)) ||
    construct<SubroutineStmt>(many(prefixSpec), "SUBROUTINE" >> name,
        defaulted(cut >> many(dummyArg)),
        defaulted(cut >> maybe(languageBindingSpec))))

// R1536 dummy-arg -> dummy-arg-name | *
TYPE_PARSER(construct<DummyArg>(name) || construct<DummyArg>(star))

// R1537 end-subroutine-stmt -> END [SUBROUTINE [subroutine-name]]
TYPE_PARSER(construct<EndSubroutineStmt>(recovery(
    "END SUBROUTINE" >> maybe(name) || bareEnd, progUnitEndStmtErrorRecovery)))

// R1538 separate-module-subprogram ->
//         mp-subprogram-stmt [specification-part] [execution-part]
//         [internal-subprogram-part] end-mp-subprogram-stmt
TYPE_CONTEXT_PARSER("separate module subprogram"_en_US,
    construct<SeparateModuleSubprogram>(statement(Parser<MpSubprogramStmt>{}),
        specificationPart, executionPart, maybe(internalSubprogramPart),
        statement(Parser<EndMpSubprogramStmt>{})))

// R1539 mp-subprogram-stmt -> MODULE PROCEDURE procedure-name
TYPE_CONTEXT_PARSER("MODULE PROCEDURE statement"_en_US,
    construct<MpSubprogramStmt>("MODULE PROCEDURE"_sptok >> name))

// R1540 end-mp-subprogram-stmt -> END [PROCEDURE [procedure-name]]
TYPE_CONTEXT_PARSER("END PROCEDURE statement"_en_US,
    construct<EndMpSubprogramStmt>(
        recovery("END PROCEDURE" >> maybe(name) || bareEnd,
            progUnitEndStmtErrorRecovery)))

// R1541 entry-stmt -> ENTRY entry-name [( [dummy-arg-list] ) [suffix]]
TYPE_PARSER(
    "ENTRY" >> (construct<EntryStmt>(name,
                    parenthesized(optionalList(dummyArg)), maybe(suffix)) ||
                   construct<EntryStmt>(name, construct<std::list<DummyArg>>(),
                       construct<std::optional<Suffix>>())))

// R1542 return-stmt -> RETURN [scalar-int-expr]
TYPE_CONTEXT_PARSER("RETURN statement"_en_US,
    construct<ReturnStmt>("RETURN" >> maybe(scalarIntExpr)))

// R1543 contains-stmt -> CONTAINS
TYPE_PARSER(construct<ContainsStmt>("CONTAINS"_tok))

// R1544 stmt-function-stmt ->
//         function-name ( [dummy-arg-name-list] ) = scalar-expr
TYPE_CONTEXT_PARSER("statement function definition"_en_US,
    construct<StmtFunctionStmt>(
        name, parenthesized(optionalList(name)), "=" >> scalar(expr)))

// Directives, extensions, and deprecated statements
// !DIR$ IGNORE_TKR [ [(tkr...)] name ]...
// !DIR$ name...
constexpr auto beginDirective{skipStuffBeforeStatement >> "!"_ch};
constexpr auto endDirective{space >> endOfLine};
constexpr auto ignore_tkr{
    "DIR$ IGNORE_TKR" >> optionalList(construct<CompilerDirective::IgnoreTKR>(
                             defaulted(parenthesized(some("tkr"_ch))), name))};
TYPE_PARSER(
    beginDirective >> sourced(construct<CompilerDirective>(ignore_tkr) ||
                          construct<CompilerDirective>("DIR$" >> many(name))) /
        endDirective)

TYPE_PARSER(extension<LanguageFeature::CrayPointer>(construct<BasedPointerStmt>(
    "POINTER" >> nonemptyList("expected POINTER associations"_err_en_US,
                     construct<BasedPointer>("(" >> objectName / ",",
                         objectName, maybe(Parser<ArraySpec>{}) / ")")))))

TYPE_PARSER(construct<StructureStmt>("STRUCTURE /" >> name / "/", pure(true),
                optionalList(entityDecl)) ||
    construct<StructureStmt>(
        "STRUCTURE" >> name, pure(false), defaulted(cut >> many(entityDecl))))

TYPE_PARSER(construct<StructureField>(statement(StructureComponents{})) ||
    construct<StructureField>(indirect(Parser<Union>{})) ||
    construct<StructureField>(indirect(Parser<StructureDef>{})))

TYPE_CONTEXT_PARSER("STRUCTURE definition"_en_US,
    extension<LanguageFeature::DECStructures>(construct<StructureDef>(
        statement(Parser<StructureStmt>{}), many(Parser<StructureField>{}),
        statement(
            construct<StructureDef::EndStructureStmt>("END STRUCTURE"_tok)))))

TYPE_CONTEXT_PARSER("UNION definition"_en_US,
    construct<Union>(statement(construct<Union::UnionStmt>("UNION"_tok)),
        many(Parser<Map>{}),
        statement(construct<Union::EndUnionStmt>("END UNION"_tok))))

TYPE_CONTEXT_PARSER("MAP definition"_en_US,
    construct<Map>(statement(construct<Map::MapStmt>("MAP"_tok)),
        many(Parser<StructureField>{}),
        statement(construct<Map::EndMapStmt>("END MAP"_tok))))

TYPE_CONTEXT_PARSER("arithmetic IF statement"_en_US,
    deprecated<LanguageFeature::ArithmeticIF>(construct<ArithmeticIfStmt>(
        "IF" >> parenthesized(expr), label / ",", label / ",", label)))

TYPE_CONTEXT_PARSER("ASSIGN statement"_en_US,
    deprecated<LanguageFeature::Assign>(
        construct<AssignStmt>("ASSIGN" >> label, "TO" >> name)))

TYPE_CONTEXT_PARSER("assigned GOTO statement"_en_US,
    deprecated<LanguageFeature::AssignedGOTO>(construct<AssignedGotoStmt>(
        "GO TO" >> name,
        defaulted(maybe(","_tok) >>
            parenthesized(nonemptyList("expected labels"_err_en_US, label))))))

TYPE_CONTEXT_PARSER("PAUSE statement"_en_US,
    deprecated<LanguageFeature::Pause>(
        construct<PauseStmt>("PAUSE" >> maybe(Parser<StopCode>{}))))

// These requirement productions are defined by the Fortran standard but never
// used directly by the grammar:
//   R620 delimiter -> ( | ) | / | [ | ] | (/ | /)
//   R1027 numeric-expr -> expr
//   R1031 int-constant-expr -> int-expr
//   R1221 dtv-type-spec -> TYPE ( derived-type-spec ) |
//           CLASS ( derived-type-spec )
//
// These requirement productions are defined and used, but need not be
// defined independently here in this file:
//   R771 lbracket -> [
//   R772 rbracket -> ]
//
// Further note that:
//   R607 int-constant -> constant
//     is used only once via R844 scalar-int-constant
//   R904 logical-variable -> variable
//     is used only via scalar-logical-variable
//   R906 default-char-variable -> variable
//     is used only via scalar-default-char-variable
//   R907 int-variable -> variable
//     is used only via scalar-int-variable
//   R915 complex-part-designator -> designator % RE | designator % IM
//     %RE and %IM are initially recognized as structure components
//   R916 type-param-inquiry -> designator % type-param-name
//     is occulted by structure component designators
//   R918 array-section ->
//        data-ref [( substring-range )] | complex-part-designator
//     is not used because parsing is not sensitive to rank
//   R1030 default-char-constant-expr -> default-char-expr
//     is only used via scalar-default-char-constant-expr
}
#endif  // FORTRAN_PARSER_GRAMMAR_H_
