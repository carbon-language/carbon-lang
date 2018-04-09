#ifndef FORTRAN_PARSER_GRAMMAR_H_
#define FORTRAN_PARSER_GRAMMAR_H_

// Top-level grammar specification for Fortran.  These parsers drive
// the tokenization parsers in cooked-tokens.h to consume characters,
// recognize the productions of Fortran, and to construct a parse tree.
// See parser-combinators.txt for documentation on the parser combinator
// library used here to implement an LL recursive descent recognizer.

#include "basic-parsers.h"
#include "characters.h"
#include "parse-tree.h"
#include "token-parsers.h"
#include "user-state.h"
#include <cinttypes>
#include <cstdio>
#include <functional>
#include <list>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

namespace Fortran {
namespace parser {

// The productions that follow are derived from the draft Fortran 2018
// standard, with some necessary modifications to remove left recursion
// and some generalization in order to defer cases where parses depend
// on the definitions of symbols.  The "Rxxx" numbers that appear in
// comments refer to these numbered requirements in the Fortran standard.

// Many parsers in this grammar are defined as instances of this Parser<>
// template class.  This usage requires that their Parse() member functions
// be defined separately, typically with a parsing expression wrapped up
// in an TYPE_PARSER() macro call.
template<typename A> struct Parser {
  using resultType = A;
  constexpr Parser() {}
  static inline std::optional<A> Parse(ParseState *);
};

#define TYPE_PARSER(pexpr) \
  template<> \
  inline std::optional<typename decltype(pexpr)::resultType> \
  Parser<typename decltype(pexpr)::resultType>::Parse(ParseState *state) { \
    return (pexpr).Parse(state); \
  }

#define TYPE_CONTEXT_PARSER(contextText, pexpr) \
  template<> \
  inline std::optional<typename decltype(pexpr)::resultType> \
  Parser<typename decltype(pexpr)::resultType>::Parse(ParseState *state) { \
    return inContext((contextText), (pexpr)).Parse(state); \
  }

// Some specializations of Parser<> are used multiple times, or are
// of some special importance, so we instantiate them once here and
// give them names rather than referencing them as anonymous Parser<T>{}
// objects in the right-hand sides of productions.
constexpr Parser<Program> program;  //  R501 - the "top level" production
constexpr Parser<SpecificationPart> specificationPart;  //  R504
constexpr Parser<ImplicitPart> implicitPart;  //  R505
constexpr Parser<DeclarationConstruct> declarationConstruct;  //  R507
constexpr Parser<SpecificationConstruct> specificationConstruct;  //  R508
constexpr Parser<ExecutionPartConstruct> executionPartConstruct;  //  R510
constexpr Parser<InternalSubprogramPart> internalSubprogramPart;  //  R511
constexpr Parser<ActionStmt> actionStmt;  // R515
constexpr Parser<Name> name;  // R603
constexpr Parser<LiteralConstant> literalConstant;  // R605
constexpr Parser<NamedConstant> namedConstant;  // R606
constexpr Parser<TypeParamValue> typeParamValue;  // R701
constexpr Parser<TypeSpec> typeSpec;  // R702
constexpr Parser<DeclarationTypeSpec> declarationTypeSpec;  // R703
constexpr Parser<IntrinsicTypeSpec> intrinsicTypeSpec;  // R704
constexpr Parser<IntegerTypeSpec> integerTypeSpec;  // R705
constexpr Parser<KindSelector> kindSelector;  // R706
constexpr Parser<SignedIntLiteralConstant> signedIntLiteralConstant;  // R707
constexpr Parser<IntLiteralConstant> intLiteralConstant;  // R708
constexpr Parser<KindParam> kindParam;  // R709
constexpr Parser<RealLiteralConstant> realLiteralConstant;  // R714
constexpr Parser<CharLength> charLength;  // R723
constexpr Parser<CharLiteralConstant> charLiteralConstant;  // R724
constexpr Parser<Initialization> initialization;  // R743 & R805
constexpr Parser<DerivedTypeSpec> derivedTypeSpec;  // R754
constexpr Parser<TypeDeclarationStmt> typeDeclarationStmt;  // R801
constexpr Parser<NullInit> nullInit;  // R806
constexpr Parser<AccessSpec> accessSpec;  // R807
constexpr Parser<LanguageBindingSpec> languageBindingSpec;  // R808, R1528
constexpr Parser<EntityDecl> entityDecl;  // R803
constexpr Parser<CoarraySpec> coarraySpec;  // R809
constexpr Parser<ArraySpec> arraySpec;  // R815
constexpr Parser<ExplicitShapeSpec> explicitShapeSpec;  // R816
constexpr Parser<DeferredShapeSpecList> deferredShapeSpecList;  // R820
constexpr Parser<AssumedImpliedSpec> assumedImpliedSpec;  // R821
constexpr Parser<IntentSpec> intentSpec;  // R826
constexpr Parser<DataStmt> dataStmt;  // R837
constexpr Parser<DataImpliedDo> dataImpliedDo;  // R840
constexpr Parser<ParameterStmt> parameterStmt;  // R851
constexpr Parser<OldParameterStmt> oldParameterStmt;
constexpr Parser<Designator> designator;  // R901
constexpr Parser<Variable> variable;  // R902
constexpr Parser<Substring> substring;  // R908
constexpr Parser<DataReference> dataReference;  // R911, R914, R917
constexpr Parser<StructureComponent> structureComponent;  // R913
constexpr Parser<StatVariable> statVariable;  // R929
constexpr Parser<StatOrErrmsg> statOrErrmsg;  // R942 & R1165
constexpr Parser<DefinedOpName> definedOpName;  // R1003, R1023, R1414, & R1415
constexpr Parser<Expr> expr;  // R1022
constexpr Parser<SpecificationExpr> specificationExpr;  // R1028
constexpr Parser<AssignmentStmt> assignmentStmt;  // R1032
constexpr Parser<PointerAssignmentStmt> pointerAssignmentStmt;  // R1033
constexpr Parser<WhereStmt> whereStmt;  // R1041, R1045, R1046
constexpr Parser<WhereConstruct> whereConstruct;  // R1042
constexpr Parser<WhereBodyConstruct> whereBodyConstruct;  // R1044
constexpr Parser<ForallConstruct> forallConstruct;  // R1050
constexpr Parser<ForallAssignmentStmt> forallAssignmentStmt;  // R1053
constexpr Parser<ForallStmt> forallStmt;  // R1055
constexpr Parser<Selector> selector;  // R1105
constexpr Parser<EndSelectStmt> endSelectStmt;  // R1143 & R1151 & R1155
constexpr Parser<LoopControl> loopControl;  // R1123
constexpr Parser<ConcurrentHeader> concurrentHeader;  // R1125
constexpr Parser<EndDoStmt> endDoStmt;  // R1132
constexpr Parser<IoUnit> ioUnit;  // R1201, R1203
constexpr Parser<FileUnitNumber> fileUnitNumber;  // R1202
constexpr Parser<IoControlSpec> ioControlSpec;  // R1213, R1214
constexpr Parser<Format> format;  // R1215
constexpr Parser<InputItem> inputItem;  // R1216
constexpr Parser<OutputItem> outputItem;  // R1217
constexpr Parser<InputImpliedDo> inputImpliedDo;  // R1218, R1219
constexpr Parser<OutputImpliedDo> outputImpliedDo;  // R1218, R1219
constexpr Parser<PositionOrFlushSpec> positionOrFlushSpec;  // R1227 & R1229
constexpr Parser<FormatStmt> formatStmt;  // R1301
constexpr Parser<InterfaceBlock> interfaceBlock;  // R1501
constexpr Parser<GenericSpec> genericSpec;  // R1508
constexpr Parser<ProcInterface> procInterface;  // R1513
constexpr Parser<ProcDecl> procDecl;  // R1515
constexpr Parser<FunctionReference> functionReference;  // R1520
constexpr Parser<ActualArgSpec> actualArgSpec;  // R1523
constexpr Parser<PrefixSpec> prefixSpec;  // R1527
constexpr Parser<FunctionSubprogram> functionSubprogram;  // R1529
constexpr Parser<FunctionStmt> functionStmt;  // R1530
constexpr Parser<Suffix> suffix;  // R1532
constexpr Parser<EndFunctionStmt> endFunctionStmt;  // R1533
constexpr Parser<SubroutineSubprogram> subroutineSubprogram;  // R1534
constexpr Parser<SubroutineStmt> subroutineStmt;  // R1535
constexpr Parser<DummyArg> dummyArg;  // R1536
constexpr Parser<EndSubroutineStmt> endSubroutineStmt;  // R1537
constexpr Parser<EntryStmt> entryStmt;  // R1541
constexpr Parser<ContainsStmt> containsStmt;  // R1543
constexpr Parser<CompilerDirective> compilerDirective;

// For a parser p, indirect(p) returns a parser that builds an indirect
// reference to p's return type.
template<typename PA> inline constexpr auto indirect(const PA &p) {
  return construct<Indirection<typename PA::resultType>>{}(p);
}

// R711 digit-string -> digit [digit]...
// N.B. not a token -- no space is skipped
constexpr DigitString digitString;

// statement(p) parses Statement<P> for some statement type P that is the
// result type of the argument parser p, while also handling labels and
// end-of-statement markers.

// R611 label -> digit [digit]...
constexpr auto label = space >> digitString / spaceCheck;

template<typename PA>
using statementConstructor = construct<Statement<typename PA::resultType>>;

template<typename PA> inline constexpr auto unterminatedStatement(const PA &p) {
  return skipEmptyLines >>
      sourced(statementConstructor<PA>{}(maybe(label), space >> p));
}

constexpr auto endOfLine = "\n"_ch / skipEmptyLines ||
    fail<const char *>("expected end of line"_err_en_US);

constexpr auto endOfStmt = space >>
    (";"_ch / skipMany(";"_tok) / maybe(endOfLine) || endOfLine);

template<typename PA> inline constexpr auto statement(const PA &p) {
  return unterminatedStatement(p) / endOfStmt;
}

constexpr auto ignoredStatementPrefix = skipEmptyLines >> maybe(label) >>
    maybe(name / ":") >> space;

// Error recovery within statements: skip to the end of the line,
// but not over an END or CONTAINS statement.
constexpr auto errorRecovery = construct<ErrorRecovery>{};
constexpr auto skipToEndOfLine = SkipTo<'\n'>{} >> errorRecovery;
constexpr auto stmtErrorRecovery =
    !"END"_tok >> !"CONTAINS"_tok >> skipToEndOfLine;

// Error recovery across statements: skip the line, unless it looks
// like it might end the containing construct.
constexpr auto errorRecoveryStart = ignoredStatementPrefix;
constexpr auto skipBadLine = SkipPast<'\n'>{} >> errorRecovery;
constexpr auto executionPartErrorRecovery = errorRecoveryStart >> !"END"_tok >>
    !"CONTAINS"_tok >> !"ELSE"_tok >> !"CASE"_tok >> !"TYPE IS"_tok >>
    !"CLASS"_tok >> !"RANK"_tok >> skipBadLine;

// R507 declaration-construct ->
//        specification-construct | data-stmt | format-stmt |
//        entry-stmt | stmt-function-stmt
constexpr auto execPartLookAhead = actionStmt >> ok || "ASSOCIATE ("_tok ||
    "BLOCK"_tok || "SELECT"_tok || "CHANGE TEAM"_sptok || "CRITICAL"_tok ||
    "DO"_tok || "IF ("_tok || "WHERE ("_tok || "FORALL ("_tok;
constexpr auto declErrorRecovery =
    errorRecoveryStart >> !execPartLookAhead >> stmtErrorRecovery;
TYPE_CONTEXT_PARSER("declaration construct"_en_US,
    recovery(construct<DeclarationConstruct>{}(specificationConstruct) ||
            construct<DeclarationConstruct>{}(statement(indirect(dataStmt))) ||
            construct<DeclarationConstruct>{}(
                statement(indirect(formatStmt))) ||
            construct<DeclarationConstruct>{}(statement(indirect(entryStmt))) ||
            construct<DeclarationConstruct>{}(
                statement(indirect(Parser<StmtFunctionStmt>{}))),
        construct<DeclarationConstruct>{}(declErrorRecovery)))

// R508 specification-construct ->
//        derived-type-def | enum-def | generic-stmt | interface-block |
//        parameter-stmt | procedure-declaration-stmt |
//        other-specification-stmt | type-declaration-stmt
TYPE_CONTEXT_PARSER("specification construct"_en_US,
    construct<SpecificationConstruct>{}(indirect(Parser<DerivedTypeDef>{})) ||
        construct<SpecificationConstruct>{}(indirect(Parser<EnumDef>{})) ||
        construct<SpecificationConstruct>{}(
            statement(indirect(Parser<GenericStmt>{}))) ||
        construct<SpecificationConstruct>{}(indirect(interfaceBlock)) ||
        construct<SpecificationConstruct>{}(
            statement(indirect(parameterStmt))) ||
        construct<SpecificationConstruct>{}(
            statement(indirect(oldParameterStmt))) ||
        construct<SpecificationConstruct>{}(
            statement(indirect(Parser<ProcedureDeclarationStmt>{}))) ||
        construct<SpecificationConstruct>{}(
            statement(Parser<OtherSpecificationStmt>{})) ||
        construct<SpecificationConstruct>{}(
            statement(indirect(typeDeclarationStmt))) ||
        construct<SpecificationConstruct>{}(indirect(Parser<StructureDef>{})) ||
        construct<SpecificationConstruct>{}(indirect(compilerDirective)))

// R513 other-specification-stmt ->
//        access-stmt | allocatable-stmt | asynchronous-stmt | bind-stmt |
//        codimension-stmt | contiguous-stmt | dimension-stmt | external-stmt |
//        intent-stmt | intrinsic-stmt | namelist-stmt | optional-stmt |
//        pointer-stmt | protected-stmt | save-stmt | target-stmt |
//        volatile-stmt | value-stmt | common-stmt | equivalence-stmt
TYPE_PARSER(
    construct<OtherSpecificationStmt>{}(indirect(Parser<AccessStmt>{})) ||
    construct<OtherSpecificationStmt>{}(indirect(Parser<AllocatableStmt>{})) ||
    construct<OtherSpecificationStmt>{}(indirect(Parser<AsynchronousStmt>{})) ||
    construct<OtherSpecificationStmt>{}(indirect(Parser<BindStmt>{})) ||
    construct<OtherSpecificationStmt>{}(indirect(Parser<CodimensionStmt>{})) ||
    construct<OtherSpecificationStmt>{}(indirect(Parser<ContiguousStmt>{})) ||
    construct<OtherSpecificationStmt>{}(indirect(Parser<DimensionStmt>{})) ||
    construct<OtherSpecificationStmt>{}(indirect(Parser<ExternalStmt>{})) ||
    construct<OtherSpecificationStmt>{}(indirect(Parser<IntentStmt>{})) ||
    construct<OtherSpecificationStmt>{}(indirect(Parser<IntrinsicStmt>{})) ||
    construct<OtherSpecificationStmt>{}(indirect(Parser<NamelistStmt>{})) ||
    construct<OtherSpecificationStmt>{}(indirect(Parser<OptionalStmt>{})) ||
    construct<OtherSpecificationStmt>{}(indirect(Parser<PointerStmt>{})) ||
    construct<OtherSpecificationStmt>{}(indirect(Parser<ProtectedStmt>{})) ||
    construct<OtherSpecificationStmt>{}(indirect(Parser<SaveStmt>{})) ||
    construct<OtherSpecificationStmt>{}(indirect(Parser<TargetStmt>{})) ||
    construct<OtherSpecificationStmt>{}(indirect(Parser<ValueStmt>{})) ||
    construct<OtherSpecificationStmt>{}(indirect(Parser<VolatileStmt>{})) ||
    construct<OtherSpecificationStmt>{}(indirect(Parser<CommonStmt>{})) ||
    construct<OtherSpecificationStmt>{}(indirect(Parser<EquivalenceStmt>{})) ||
    construct<OtherSpecificationStmt>{}(indirect(Parser<BasedPointerStmt>{})))

// R604 constant ->  literal-constant | named-constant
// Used only via R607 int-constant and R845 data-stmt-constant.
// The look-ahead check prevents occlusion of constant-subobject in
// data-stmt-constant.
TYPE_PARSER(construct<ConstantValue>{}(literalConstant) ||
    construct<ConstantValue>{}(namedConstant / !"%"_tok / !"("_tok))

// R608 intrinsic-operator ->
//        power-op | mult-op | add-op | concat-op | rel-op |
//        not-op | and-op | or-op | equiv-op
// R610 extended-intrinsic-op -> intrinsic-operator
// These parsers must be ordered carefully to avoid misrecognition.
constexpr auto namedIntrinsicOperator = ".LT." >>
        pure(DefinedOperator::IntrinsicOperator::LT) ||
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
    extension(".XOR." >> pure(DefinedOperator::IntrinsicOperator::XOR) ||
        ".N." >> pure(DefinedOperator::IntrinsicOperator::NOT) ||
        ".A." >> pure(DefinedOperator::IntrinsicOperator::AND) ||
        ".O." >> pure(DefinedOperator::IntrinsicOperator::OR) ||
        ".X." >> pure(DefinedOperator::IntrinsicOperator::XOR));

constexpr auto intrinsicOperator = "**" >>
        pure(DefinedOperator::IntrinsicOperator::Power) ||
    "*" >> pure(DefinedOperator::IntrinsicOperator::Multiply) ||
    "//" >> pure(DefinedOperator::IntrinsicOperator::Concat) ||
    "/=" >> pure(DefinedOperator::IntrinsicOperator::NE) ||
    "/" >> pure(DefinedOperator::IntrinsicOperator::Divide) ||
    "+" >> pure(DefinedOperator::IntrinsicOperator::Add) ||
    "-" >> pure(DefinedOperator::IntrinsicOperator::Subtract) ||
    "<=" >> pure(DefinedOperator::IntrinsicOperator::LE) ||
    extension("<>" >> pure(DefinedOperator::IntrinsicOperator::NE)) ||
    "<" >> pure(DefinedOperator::IntrinsicOperator::LT) ||
    "==" >> pure(DefinedOperator::IntrinsicOperator::EQ) ||
    ">=" >> pure(DefinedOperator::IntrinsicOperator::GE) ||
    ">" >> pure(DefinedOperator::IntrinsicOperator::GT) ||
    namedIntrinsicOperator;

// R609 defined-operator ->
//        defined-unary-op | defined-binary-op | extended-intrinsic-op
TYPE_PARSER(construct<DefinedOperator>{}(intrinsicOperator) ||
    construct<DefinedOperator>{}(definedOpName))

// R401 xzy-list -> xzy [, xzy]...
template<typename PA> inline constexpr auto nonemptyList(const PA &p) {
  return nonemptySeparated(p, ","_tok);  // p-list
}

template<typename PA> inline constexpr auto optionalList(const PA &p) {
  return defaulted(nonemptySeparated(p, ","_tok));  // [p-list]
}

// R402 xzy-name -> name

// R403 scalar-xyz -> xyz
// Also define constant-xyz, int-xyz, default-char-xyz.
template<typename PA> inline constexpr auto scalar(const PA &p) {
  return construct<Scalar<typename PA::resultType>>{}(p);  // scalar-p
}

template<typename PA> inline constexpr auto constant(const PA &p) {
  return construct<Constant<typename PA::resultType>>{}(p);  // constant-p
}

template<typename PA> inline constexpr auto integer(const PA &p) {
  return construct<Integer<typename PA::resultType>>{}(p);  // int-p
}

template<typename PA> inline constexpr auto logical(const PA &p) {
  return construct<Logical<typename PA::resultType>>{}(p);  // logical-p
}

template<typename PA> inline constexpr auto defaultChar(const PA &p) {
  return construct<DefaultChar<typename PA::resultType>>{}(
      p);  // default-char-p
}

// R1024 logical-expr -> expr
constexpr auto logicalExpr = logical(indirect(expr));
constexpr auto scalarLogicalExpr = scalar(logicalExpr);

// R1025 default-char-expr -> expr
constexpr auto defaultCharExpr = defaultChar(indirect(expr));
constexpr auto scalarDefaultCharExpr = scalar(defaultCharExpr);

// R1026 int-expr -> expr
constexpr auto intExpr = integer(indirect(expr));
constexpr auto scalarIntExpr = scalar(intExpr);

// R1029 constant-expr -> expr
constexpr auto constantExpr = constant(indirect(expr));

// R1030 default-char-constant-expr -> default-char-expr
constexpr auto scalarDefaultCharConstantExpr =
    scalar(defaultChar(constantExpr));

// R1031 int-constant-expr -> int-expr
constexpr auto intConstantExpr = integer(constantExpr);
constexpr auto scalarIntConstantExpr = scalar(intConstantExpr);

// R501 program -> program-unit [program-unit]...
// This is the top-level production for the Fortran language.
struct StartNewSubprogram {
  using resultType = Success;
  static std::optional<Success> Parse(ParseState *state) {
    if (auto ustate = state->userState()) {
      ustate->NewSubprogram();
    }
    return {Success{}};
  }
} startNewSubprogram;

TYPE_PARSER(
    construct<Program>{}(
        // statements consume only trailing noise; consume leading noise here.
        skipEmptyLines >>
        some(startNewSubprogram >> Parser<ProgramUnit>{} / endOfLine)) /
    consumedAllInput)

// R502 program-unit ->
//        main-program | external-subprogram | module | submodule | block-data
// R503 external-subprogram -> function-subprogram | subroutine-subprogram
TYPE_PARSER(construct<ProgramUnit>{}(indirect(functionSubprogram)) ||
    construct<ProgramUnit>{}(indirect(subroutineSubprogram)) ||
    construct<ProgramUnit>{}(indirect(Parser<Module>{})) ||
    construct<ProgramUnit>{}(indirect(Parser<Submodule>{})) ||
    construct<ProgramUnit>{}(indirect(Parser<BlockData>{})) ||
    construct<ProgramUnit>{}(indirect(Parser<MainProgram>{})))

// R504 specification-part ->
//         [use-stmt]... [import-stmt]... [implicit-part]
//         [declaration-construct]...
TYPE_CONTEXT_PARSER("specification part"_en_US,
    construct<SpecificationPart>{}(many(statement(indirect(Parser<UseStmt>{}))),
        many(statement(indirect(Parser<ImportStmt>{}))), implicitPart,
        many(declarationConstruct)))

// R505 implicit-part -> [implicit-part-stmt]... implicit-stmt
// TODO: Can overshoot; any trailing PARAMETER, FORMAT, & ENTRY
// statements after the last IMPLICIT should be transferred to the
// list of declaration-constructs.
TYPE_CONTEXT_PARSER("implicit part"_en_US,
    construct<ImplicitPart>{}(many(Parser<ImplicitPartStmt>{})))

// R506 implicit-part-stmt ->
//         implicit-stmt | parameter-stmt | format-stmt | entry-stmt
TYPE_PARSER(construct<ImplicitPartStmt>{}(
                statement(indirect(Parser<ImplicitStmt>{}))) ||
    construct<ImplicitPartStmt>{}(statement(indirect(parameterStmt))) ||
    construct<ImplicitPartStmt>{}(statement(indirect(oldParameterStmt))) ||
    construct<ImplicitPartStmt>{}(statement(indirect(formatStmt))) ||
    construct<ImplicitPartStmt>{}(statement(indirect(entryStmt))))

// R512 internal-subprogram -> function-subprogram | subroutine-subprogram
constexpr auto internalSubprogram =
    (construct<InternalSubprogram>{}(indirect(functionSubprogram)) ||
        construct<InternalSubprogram>{}(indirect(subroutineSubprogram))) /
    endOfStmt;

// R511 internal-subprogram-part -> contains-stmt [internal-subprogram]...
TYPE_CONTEXT_PARSER("internal subprogram part"_en_US,
    construct<InternalSubprogramPart>{}(statement(containsStmt),
        many(startNewSubprogram >> internalSubprogram)))

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
TYPE_PARSER(construct<ActionStmt>{}(indirect(Parser<AllocateStmt>{})) ||
    construct<ActionStmt>{}(indirect(assignmentStmt)) ||
    construct<ActionStmt>{}(indirect(Parser<BackspaceStmt>{})) ||
    construct<ActionStmt>{}(indirect(Parser<CallStmt>{})) ||
    construct<ActionStmt>{}(indirect(Parser<CloseStmt>{})) ||
    "CONTINUE" >> construct<ActionStmt>{}(construct<ContinueStmt>{}) ||
    construct<ActionStmt>{}(indirect(Parser<CycleStmt>{})) ||
    construct<ActionStmt>{}(indirect(Parser<DeallocateStmt>{})) ||
    construct<ActionStmt>{}(indirect(Parser<EndfileStmt>{})) ||
    construct<ActionStmt>{}(indirect(Parser<EventPostStmt>{})) ||
    construct<ActionStmt>{}(indirect(Parser<EventWaitStmt>{})) ||
    construct<ActionStmt>{}(indirect(Parser<ExitStmt>{})) ||
    "FAIL IMAGE"_sptok >> construct<ActionStmt>{}(construct<FailImageStmt>{}) ||
    construct<ActionStmt>{}(indirect(Parser<FlushStmt>{})) ||
    construct<ActionStmt>{}(indirect(Parser<FormTeamStmt>{})) ||
    construct<ActionStmt>{}(indirect(Parser<GotoStmt>{})) ||
    construct<ActionStmt>{}(indirect(Parser<IfStmt>{})) ||
    construct<ActionStmt>{}(indirect(Parser<InquireStmt>{})) ||
    construct<ActionStmt>{}(indirect(Parser<LockStmt>{})) ||
    construct<ActionStmt>{}(indirect(Parser<NullifyStmt>{})) ||
    construct<ActionStmt>{}(indirect(Parser<OpenStmt>{})) ||
    construct<ActionStmt>{}(indirect(pointerAssignmentStmt)) ||
    construct<ActionStmt>{}(indirect(Parser<PrintStmt>{})) ||
    construct<ActionStmt>{}(indirect(Parser<ReadStmt>{})) ||
    construct<ActionStmt>{}(indirect(Parser<ReturnStmt>{})) ||
    construct<ActionStmt>{}(indirect(Parser<RewindStmt>{})) ||
    construct<ActionStmt>{}(
        indirect(Parser<StopStmt>{})) ||  // & error-stop-stmt
    construct<ActionStmt>{}(indirect(Parser<SyncAllStmt>{})) ||
    construct<ActionStmt>{}(indirect(Parser<SyncImagesStmt>{})) ||
    construct<ActionStmt>{}(indirect(Parser<SyncMemoryStmt>{})) ||
    construct<ActionStmt>{}(indirect(Parser<SyncTeamStmt>{})) ||
    construct<ActionStmt>{}(indirect(Parser<UnlockStmt>{})) ||
    construct<ActionStmt>{}(indirect(Parser<WaitStmt>{})) ||
    construct<ActionStmt>{}(indirect(whereStmt)) ||
    construct<ActionStmt>{}(indirect(Parser<WriteStmt>{})) ||
    construct<ActionStmt>{}(indirect(Parser<ComputedGotoStmt>{})) ||
    construct<ActionStmt>{}(indirect(forallStmt)) ||
    construct<ActionStmt>{}(indirect(Parser<ArithmeticIfStmt>{})) ||
    construct<ActionStmt>{}(indirect(Parser<AssignStmt>{})) ||
    construct<ActionStmt>{}(indirect(Parser<AssignedGotoStmt>{})) ||
    construct<ActionStmt>{}(indirect(Parser<PauseStmt>{})))

// Fortran allows the statement with the corresponding label at the end of
// a do-construct that begins with an old-style label-do-stmt to be a
// new-style END DO statement; e.g., DO 10 I=1,N; ...; 10 END DO.  Usually,
// END DO statements appear only at the ends of do-constructs that begin
// with a nonlabel-do-stmt, so care must be taken to recognize this case and
// essentially treat them like CONTINUE statements.
struct CapturedLabelDoStmt {
  static constexpr auto parser = statement(indirect(Parser<LabelDoStmt>{}));
  using resultType = Statement<Indirection<LabelDoStmt>>;
  static std::optional<resultType> Parse(ParseState *state) {
    auto result = parser.Parse(state);
    if (result) {
      if (auto ustate = state->userState()) {
        ustate->NewDoLabel(std::get<Label>(result->statement->t));
      }
    }
    return result;
  }
} capturedLabelDoStmt;

struct EndDoStmtForCapturedLabelDoStmt {
  static constexpr auto parser = statement(indirect(endDoStmt));
  using resultType = Statement<Indirection<EndDoStmt>>;
  static std::optional<resultType> Parse(ParseState *state) {
    if (auto enddo = parser.Parse(state)) {
      if (enddo->label.has_value()) {
        if (auto ustate = state->userState()) {
          if (!ustate->InNonlabelDoConstruct() &&
              ustate->IsDoLabel(enddo->label.value())) {
            return enddo;
          }
        }
      }
    }
    return {};
  }
} endDoStmtForCapturedLabelDoStmt;

// R514 executable-construct ->
//        action-stmt | associate-construct | block-construct |
//        case-construct | change-team-construct | critical-construct |
//        do-construct | if-construct | select-rank-construct |
//        select-type-construct | where-construct | forall-construct
constexpr auto executableConstruct =
    construct<ExecutableConstruct>{}(statement(actionStmt)) ||
    construct<ExecutableConstruct>{}(indirect(Parser<AssociateConstruct>{})) ||
    construct<ExecutableConstruct>{}(indirect(Parser<BlockConstruct>{})) ||
    construct<ExecutableConstruct>{}(indirect(Parser<CaseConstruct>{})) ||
    construct<ExecutableConstruct>{}(indirect(Parser<ChangeTeamConstruct>{})) ||
    construct<ExecutableConstruct>{}(indirect(Parser<CriticalConstruct>{})) ||
    construct<ExecutableConstruct>{}(capturedLabelDoStmt) ||
    construct<ExecutableConstruct>{}(endDoStmtForCapturedLabelDoStmt) ||
    construct<ExecutableConstruct>{}(indirect(Parser<DoConstruct>{})) ||
    construct<ExecutableConstruct>{}(indirect(Parser<IfConstruct>{})) ||
    construct<ExecutableConstruct>{}(indirect(Parser<SelectRankConstruct>{})) ||
    construct<ExecutableConstruct>{}(indirect(Parser<SelectTypeConstruct>{})) ||
    construct<ExecutableConstruct>{}(indirect(whereConstruct)) ||
    construct<ExecutableConstruct>{}(indirect(forallConstruct)) ||
    construct<ExecutableConstruct>{}(indirect(compilerDirective));

// R510 execution-part-construct ->
//        executable-construct | format-stmt | entry-stmt | data-stmt
// Extension (PGI/Intel): also accept NAMELIST in execution part
constexpr auto obsoleteExecutionPartConstruct = recovery(
    ignoredStatementPrefix >>
        fail<ExecutionPartConstruct>(
            "obsolete legacy extension is not supported"_err_en_US),
    construct<ExecutionPartConstruct>{}(
        statement("REDIMENSION" >> name >>
            parenthesized(nonemptyList(Parser<AllocateShapeSpec>{})) >> ok) >>
        errorRecovery));

TYPE_CONTEXT_PARSER("execution part construct"_en_US,
    recovery(construct<ExecutionPartConstruct>{}(executableConstruct) ||
            construct<ExecutionPartConstruct>{}(
                statement(indirect(formatStmt))) ||
            construct<ExecutionPartConstruct>{}(
                statement(indirect(entryStmt))) ||
            construct<ExecutionPartConstruct>{}(
                statement(indirect(dataStmt))) ||
            extension(construct<ExecutionPartConstruct>{}(
                          statement(indirect(Parser<NamelistStmt>{}))) ||
                obsoleteExecutionPartConstruct),
        construct<ExecutionPartConstruct>{}(executionPartErrorRecovery)))

// R509 execution-part -> executable-construct [execution-part-construct]...
constexpr auto executionPart =
    inContext("execution part"_en_US, many(executionPartConstruct));

// R602 underscore -> _
constexpr auto underscore = "_"_ch;

// R516 keyword -> name
// R601 alphanumeric-character -> letter | digit | underscore
// R603 name -> letter [alphanumeric-character]...
// N.B. Don't accept an underscore if it is immediately followed by a
// quotation mark, so that kindParameter_"character literal" is parsed properly.
// PGI and ifort accept '$' in identifiers, even as the initial character.
// Cray and gfortran accept '$', but not as the first character.
// Cray accepts '@' as well.
constexpr auto otherIdChar = underscore / !"'\""_ch || extension("$@"_ch);
constexpr auto nonDigitIdChar = letter || otherIdChar;
constexpr auto rawName = nonDigitIdChar >> many(nonDigitIdChar || digit);
TYPE_PARSER(space >> sourced(attempt(rawName) >> construct<Name>{}))
constexpr auto keyword = construct<Keyword>{}(name);

// R605 literal-constant ->
//        int-literal-constant | real-literal-constant |
//        complex-literal-constant | logical-literal-constant |
//        char-literal-constant | boz-literal-constant
TYPE_PARSER(construct<LiteralConstant>{}(Parser<HollerithLiteralConstant>{}) ||
    construct<LiteralConstant>{}(space >> realLiteralConstant) ||
    construct<LiteralConstant>{}(intLiteralConstant) ||
    construct<LiteralConstant>{}(Parser<ComplexLiteralConstant>{}) ||
    construct<LiteralConstant>{}(Parser<BOZLiteralConstant>{}) ||
    construct<LiteralConstant>{}(charLiteralConstant) ||
    construct<LiteralConstant>{}(Parser<LogicalLiteralConstant>{}))

// R606 named-constant -> name
TYPE_PARSER(construct<NamedConstant>{}(name))

// R701 type-param-value -> scalar-int-expr | * | :
constexpr auto star = "*" >> construct<Star>{};
TYPE_PARSER(construct<TypeParamValue>{}(scalarIntExpr) ||
    construct<TypeParamValue>{}(star) ||
    construct<TypeParamValue>{}(":" >> construct<TypeParamValue::Deferred>{}))

// R702 type-spec -> intrinsic-type-spec | derived-type-spec
// N.B. This type-spec production is one of two instances in the Fortran
// grammar where intrinsic types and bare derived type names can clash;
// the other is below in R703 declaration-type-spec.  Look-ahead is required
// to disambiguate the cases where a derived type name begins with the name
// of an intrinsic type, e.g., REALITY.
TYPE_CONTEXT_PARSER("type spec"_en_US,
    construct<TypeSpec>{}(intrinsicTypeSpec / lookAhead("::"_tok || ")"_tok)) ||
        construct<TypeSpec>{}(derivedTypeSpec))

// R703 declaration-type-spec ->
//        intrinsic-type-spec | TYPE ( intrinsic-type-spec ) |
//        TYPE ( derived-type-spec ) | CLASS ( derived-type-spec ) |
//        CLASS ( * ) | TYPE ( * )
// N.B. It is critical to distribute "parenthesized()" over the alternatives
// for TYPE (...), rather than putting the alternatives within it, which
// would fail on "TYPE(real_derived)" with a misrecognition of "real" as an
// intrinsic-type-spec.
TYPE_CONTEXT_PARSER("declaration type spec"_en_US,
    construct<DeclarationTypeSpec>{}(intrinsicTypeSpec) ||
        "TYPE" >>
            (parenthesized(
                 construct<DeclarationTypeSpec>{}(intrinsicTypeSpec)) ||
                parenthesized(construct<DeclarationTypeSpec>{}(
                    construct<DeclarationTypeSpec::Type>{}(derivedTypeSpec))) ||
                "( * )" >> construct<DeclarationTypeSpec>{}(
                               construct<DeclarationTypeSpec::TypeStar>{})) ||
        "CLASS" >>
            parenthesized(
                construct<DeclarationTypeSpec>{}(
                    construct<DeclarationTypeSpec::Class>{}(derivedTypeSpec)) ||
                "*" >> construct<DeclarationTypeSpec>{}(
                           construct<DeclarationTypeSpec::ClassStar>{})) ||
        extension("RECORD /" >>
            construct<DeclarationTypeSpec>{}(
                construct<DeclarationTypeSpec::Record>{}(name / "/"))))

// R704 intrinsic-type-spec ->
//        integer-type-spec | REAL [kind-selector] | DOUBLE PRECISION |
//        COMPLEX [kind-selector] | CHARACTER [char-selector] |
//        LOGICAL [kind-selector]
// Extensions: DOUBLE COMPLEX, NCHARACTER, BYTE
TYPE_CONTEXT_PARSER("intrinsic type spec"_en_US,
    construct<IntrinsicTypeSpec>{}(integerTypeSpec) ||
        "REAL" >>
            construct<IntrinsicTypeSpec>{}(
                construct<IntrinsicTypeSpec::Real>{}(maybe(kindSelector))) ||
        "DOUBLE PRECISION" >>
            construct<IntrinsicTypeSpec>{}(
                construct<IntrinsicTypeSpec::DoublePrecision>{}) ||
        "COMPLEX" >>
            construct<IntrinsicTypeSpec>{}(
                construct<IntrinsicTypeSpec::Complex>{}(maybe(kindSelector))) ||
        "CHARACTER" >> construct<IntrinsicTypeSpec>{}(
                           construct<IntrinsicTypeSpec::Character>{}(
                               maybe(Parser<CharSelector>{}))) ||
        "LOGICAL" >>
            construct<IntrinsicTypeSpec>{}(
                construct<IntrinsicTypeSpec::Logical>{}(maybe(kindSelector))) ||
        "DOUBLE COMPLEX" >>
            construct<IntrinsicTypeSpec>{}(
                extension(construct<IntrinsicTypeSpec::DoubleComplex>{})) ||
        "NCHARACTER" >> construct<IntrinsicTypeSpec>{}(extension(
                            construct<IntrinsicTypeSpec::NCharacter>{}(
                                maybe(Parser<LengthSelector>{})))) ||
        extension("BYTE" >>
            construct<IntrinsicTypeSpec>{}(construct<IntegerTypeSpec>{}(
                construct<std::optional<KindSelector>>{}(pure(1))))))

// R705 integer-type-spec -> INTEGER [kind-selector]
TYPE_PARSER(construct<IntegerTypeSpec>{}("INTEGER" >> maybe(kindSelector)))

// R706 kind-selector -> ( [KIND =] scalar-int-constant-expr )
// Legacy extension: kind-selector -> * digit-string
TYPE_PARSER(construct<KindSelector>{}(
                parenthesized(maybe("KIND ="_tok) >> scalarIntConstantExpr)) ||
    extension(construct<KindSelector>{}(
        construct<KindSelector::StarSize>{}("*" >> digitString / spaceCheck))))

// R710 signed-digit-string -> [sign] digit-string
// N.B. Not a complete token -- no space is skipped.
constexpr SignedDigitString signedDigitString;

// R707 signed-int-literal-constant -> [sign] int-literal-constant
TYPE_PARSER(space >> sourced(construct<SignedIntLiteralConstant>{}(
                         signedDigitString, maybe(underscore >> kindParam))))

// R708 int-literal-constant -> digit-string [_ kind-param]
// The negated look-ahead for a trailing underscore prevents misrecognition
// when the digit string is a numeric kind parameter of a character literal.
TYPE_PARSER(construct<IntLiteralConstant>{}(
    space >> digitString, maybe(underscore >> kindParam) / !underscore))

// R709 kind-param -> digit-string | scalar-int-constant-name
TYPE_PARSER(construct<KindParam>{}(digitString) ||
    construct<KindParam>{}(scalar(integer(constant(name)))))

// R712 sign -> + | -
// Not a complete token.
constexpr auto sign = "+"_ch >> pure(Sign::Positive) ||
    "-"_ch >> pure(Sign::Negative);

// R713 signed-real-literal-constant -> [sign] real-literal-constant
constexpr auto signedRealLiteralConstant = space >>
    construct<SignedRealLiteralConstant>{}(maybe(sign), realLiteralConstant);

// R714 real-literal-constant ->
//        significand [exponent-letter exponent] [_ kind-param] |
//        digit-string exponent-letter exponent [_ kind-param]
// R715 significand -> digit-string . [digit-string] | . digit-string
// R716 exponent-letter -> E | D
// Extension: Q
// R717 exponent -> signed-digit-string
// N.B. Preceding space is not skipped.
constexpr auto exponentPart =
    ("ed"_ch || extension("q"_ch)) >> signedDigitString;

TYPE_CONTEXT_PARSER("REAL literal constant"_en_US,
    construct<RealLiteralConstant>{}(
        sourced(
            (skipDigitString >> "."_ch >>
                    !(some(letter) >> "."_ch /* don't misinterpret 1.AND. */) >>
                    maybe(skipDigitString) >> maybe(exponentPart) >> ok ||
                "."_ch >> skipDigitString >> maybe(exponentPart) >> ok ||
                skipDigitString >> exponentPart >> ok) >>
            construct<RealLiteralConstant::Real>{}),
        maybe(underscore >> kindParam)))

// R718 complex-literal-constant -> ( real-part , imag-part )
TYPE_CONTEXT_PARSER("COMPLEX literal constant"_en_US,
    parenthesized(construct<ComplexLiteralConstant>{}(
        Parser<ComplexPart>{} / ",", Parser<ComplexPart>{})))

// PGI/Intel extension: signed complex literal constant
TYPE_PARSER(construct<SignedComplexLiteralConstant>{}(
    space >> sign, Parser<ComplexLiteralConstant>{}))

// R719 real-part ->
//        signed-int-literal-constant | signed-real-literal-constant |
//        named-constant
// R720 imag-part ->
//        signed-int-literal-constant | signed-real-literal-constant |
//        named-constant
TYPE_PARSER(construct<ComplexPart>{}(signedRealLiteralConstant) ||
    construct<ComplexPart>{}(signedIntLiteralConstant) ||
    construct<ComplexPart>{}(namedConstant))

// R721 char-selector ->
//        length-selector |
//        ( LEN = type-param-value , KIND = scalar-int-constant-expr ) |
//        ( type-param-value , [KIND =] scalar-int-constant-expr ) |
//        ( KIND = scalar-int-constant-expr [, LEN = type-param-value] )
TYPE_PARSER(construct<CharSelector>{}(Parser<LengthSelector>{}) ||
    parenthesized(construct<CharSelector>{}(
        "LEN =" >> typeParamValue, ", KIND =" >> scalarIntConstantExpr)) ||
    parenthesized(construct<CharSelector>{}(
        typeParamValue / ",", maybe("KIND ="_tok) >> scalarIntConstantExpr)) ||
    parenthesized(construct<CharSelector>{}(
        "KIND =" >> scalarIntConstantExpr, maybe(", LEN =" >> typeParamValue))))

// R722 length-selector -> ( [LEN =] type-param-value ) | * char-length [,]
// N.B. The trailing [,] in the production is permitted by the Standard
// only in the context of a type-declaration-stmt, but even with that
// limitation, it would seem to be unnecessary and buggy to consume the comma
// here.
TYPE_PARSER(construct<LengthSelector>{}(
                parenthesized(maybe("LEN ="_tok) >> typeParamValue)) ||
    construct<LengthSelector>{}("*" >> charLength /* / maybe(","_tok) */))

// R723 char-length -> ( type-param-value ) | digit-string
TYPE_PARSER(construct<CharLength>{}(parenthesized(typeParamValue)) ||
    construct<CharLength>{}(space >> digitString / spaceCheck))

// R724 char-literal-constant ->
//        [kind-param _] ' [rep-char]... ' |
//        [kind-param _] " [rep-char]... "
// "rep-char" is any non-control character.  Doubled interior quotes are
// combined.  Backslash escapes can be enabled.
// PGI extension: nc'...' is Kanji.
// N.B. charLiteralConstantWithoutKind does not skip preceding space.
// N.B. the parsing of "name" takes care to not consume the '_'.
constexpr auto charLiteralConstantWithoutKind =
    "'"_ch >> CharLiteral<'\''>{} || "\""_ch >> CharLiteral<'"'>{};

TYPE_CONTEXT_PARSER("CHARACTER literal constant"_en_US,
    construct<CharLiteralConstant>{}(
        kindParam / underscore, charLiteralConstantWithoutKind) ||
        construct<CharLiteralConstant>{}(construct<std::optional<KindParam>>{},
            space >> charLiteralConstantWithoutKind) ||
        construct<CharLiteralConstant>{}(
            "NC" >> construct<std::optional<KindParam>>{}(
                        construct<KindParam>{}(construct<KindParam::Kanji>{})),
            charLiteralConstantWithoutKind))

// deprecated: Hollerith literals
constexpr auto rawHollerithLiteral = deprecated(HollerithLiteral{});

TYPE_CONTEXT_PARSER("Hollerith"_en_US,
    construct<HollerithLiteralConstant>{}(rawHollerithLiteral))

// R725 logical-literal-constant ->
//        .TRUE. [_ kind-param] | .FALSE. [_ kind-param]
// Also accept .T. and .F. as extensions.
TYPE_PARSER(construct<LogicalLiteralConstant>{}(
                (".TRUE."_tok || extension(".T."_tok)) >> pure(true),
                maybe(underscore >> kindParam)) ||
    construct<LogicalLiteralConstant>{}(
        (".FALSE."_tok || extension(".F."_tok)) >> pure(false),
        maybe(underscore >> kindParam)))

// R726 derived-type-def ->
//        derived-type-stmt [type-param-def-stmt]...
//        [private-or-sequence]... [component-part]
//        [type-bound-procedure-part] end-type-stmt
// R735 component-part -> [component-def-stmt]...
TYPE_CONTEXT_PARSER("derived type definition"_en_US,
    construct<DerivedTypeDef>{}(statement(Parser<DerivedTypeStmt>{}),
        many(statement(Parser<TypeParamDefStmt>{})),
        many(statement(Parser<PrivateOrSequence>{})),
        many(statement(Parser<ComponentDefStmt>{})),
        maybe(Parser<TypeBoundProcedurePart>{}),
        statement(Parser<EndTypeStmt>{})))

// R727 derived-type-stmt ->
//        TYPE [[, type-attr-spec-list] ::] type-name [(
//        type-param-name-list )]
TYPE_CONTEXT_PARSER("TYPE statement"_en_US,
    construct<DerivedTypeStmt>{}(
        "TYPE" >> optionalListBeforeColons(Parser<TypeAttrSpec>{}), name,
        defaulted(parenthesized(nonemptyList(name)))))

// R728 type-attr-spec ->
//        ABSTRACT | access-spec | BIND(C) | EXTENDS ( parent-type-name )
TYPE_PARSER(construct<TypeAttrSpec>{}("ABSTRACT" >> construct<Abstract>{}) ||
    construct<TypeAttrSpec>{}(
        "BIND ( C )" >> construct<TypeAttrSpec::BindC>{}) ||
    construct<TypeAttrSpec>{}(
        "EXTENDS" >> construct<TypeAttrSpec::Extends>{}(parenthesized(name))) ||
    construct<TypeAttrSpec>{}(accessSpec))

// R729 private-or-sequence -> private-components-stmt | sequence-stmt
TYPE_PARSER(construct<PrivateOrSequence>{}(Parser<PrivateStmt>{}) ||
    construct<PrivateOrSequence>{}(Parser<SequenceStmt>{}))

// R730 end-type-stmt -> END TYPE [type-name]
constexpr auto noNameEnd = "END" >> defaulted(cut >> maybe(name));
constexpr auto bareEnd = noNameEnd / lookAhead(endOfStmt);
constexpr auto endStmtErrorRecovery = noNameEnd / SkipTo<'\n'>{};
TYPE_PARSER(construct<EndTypeStmt>{}(
    recovery("END TYPE" >> maybe(name), endStmtErrorRecovery)))

// R731 sequence-stmt -> SEQUENCE
TYPE_PARSER("SEQUENCE" >> construct<SequenceStmt>{})

// R732 type-param-def-stmt ->
//        integer-type-spec , type-param-attr-spec :: type-param-decl-list
// R734 type-param-attr-spec -> KIND | LEN
TYPE_PARSER(construct<TypeParamDefStmt>{}(integerTypeSpec / ",",
    "KIND" >> pure(TypeParamDefStmt::KindOrLen::Kind) ||
        "LEN" >> pure(TypeParamDefStmt::KindOrLen::Len),
    "::" >> nonemptyList(Parser<TypeParamDecl>{})))

// R733 type-param-decl -> type-param-name [= scalar-int-constant-expr]
TYPE_PARSER(
    construct<TypeParamDecl>{}(name, maybe("=" >> scalarIntConstantExpr)))

// R736 component-def-stmt -> data-component-def-stmt |
//        proc-component-def-stmt
// Accidental extension not enabled here: PGI accepts type-param-def-stmt in
// component-part of derived-type-def.
TYPE_PARSER(
    recovery(construct<ComponentDefStmt>{}(Parser<DataComponentDefStmt>{}) ||
            construct<ComponentDefStmt>{}(Parser<ProcComponentDefStmt>{}),
        construct<ComponentDefStmt>{}(stmtErrorRecovery)))

// R737 data-component-def-stmt ->
//        declaration-type-spec [[, component-attr-spec-list] ::]
//        component-decl-list
TYPE_PARSER(construct<DataComponentDefStmt>{}(declarationTypeSpec,
    optionalListBeforeColons(Parser<ComponentAttrSpec>{}),
    nonemptyList(Parser<ComponentDecl>{})))

// R738 component-attr-spec ->
//        access-spec | ALLOCATABLE |
//        CODIMENSION lbracket coarray-spec rbracket |
//        CONTIGUOUS | DIMENSION ( component-array-spec ) | POINTER
constexpr auto allocatable = "ALLOCATABLE" >> construct<Allocatable>{};
constexpr auto contiguous = "CONTIGUOUS" >> construct<Contiguous>{};
constexpr auto pointer = "POINTER" >> construct<Pointer>{};
TYPE_PARSER(construct<ComponentAttrSpec>{}(accessSpec) ||
    construct<ComponentAttrSpec>{}(allocatable) ||
    "CODIMENSION" >> construct<ComponentAttrSpec>{}(coarraySpec) ||
    construct<ComponentAttrSpec>{}(contiguous) ||
    "DIMENSION" >>
        construct<ComponentAttrSpec>{}(Parser<ComponentArraySpec>{}) ||
    construct<ComponentAttrSpec>{}(pointer))

// R739 component-decl ->
//        component-name [( component-array-spec )]
//        [lbracket coarray-spec rbracket] [* char-length]
//        [component-initialization]
TYPE_CONTEXT_PARSER("component declaration"_en_US,
    construct<ComponentDecl>{}(name, maybe(Parser<ComponentArraySpec>{}),
        maybe(coarraySpec), maybe("*" >> charLength), maybe(initialization)))

// R740 component-array-spec ->
//        explicit-shape-spec-list | deferred-shape-spec-list
// N.B. Parenthesized here rather than around references to this production.
TYPE_PARSER(construct<ComponentArraySpec>{}(
                parenthesized(nonemptyList(explicitShapeSpec))) ||
    construct<ComponentArraySpec>{}(parenthesized(deferredShapeSpecList)))

// R741 proc-component-def-stmt ->
//        PROCEDURE ( [proc-interface] ) , proc-component-attr-spec-list
//          :: proc-decl-list
TYPE_CONTEXT_PARSER("PROCEDURE component definition statement"_en_US,
    "PROCEDURE" >>
        construct<ProcComponentDefStmt>{}(parenthesized(maybe(procInterface)),
            "," >> nonemptyList(Parser<ProcComponentAttrSpec>{}) / "::",
            nonemptyList(procDecl)))

// R742 proc-component-attr-spec ->
//        access-spec | NOPASS | PASS [(arg-name)] | POINTER
constexpr auto noPass = "NOPASS" >> construct<NoPass>{};
constexpr auto pass = "PASS" >> construct<Pass>{}(maybe(parenthesized(name)));
TYPE_PARSER(construct<ProcComponentAttrSpec>{}(accessSpec) ||
    construct<ProcComponentAttrSpec>{}(noPass) ||
    construct<ProcComponentAttrSpec>{}(pass) ||
    construct<ProcComponentAttrSpec>{}(pointer))

// R744 initial-data-target -> designator
constexpr auto initialDataTarget = indirect(designator);

// R743 component-initialization ->
//        = constant-expr | => null-init | => initial-data-target
// R805 initialization ->
//        = constant-expr | => null-init | => initial-data-target
// Universal extension: initialization -> / data-stmt-value-list /
TYPE_PARSER("=>" >> construct<Initialization>{}(nullInit) ||
    "=>" >> construct<Initialization>{}(initialDataTarget) ||
    "=" >> construct<Initialization>{}(constantExpr) ||
    extension("/" >> construct<Initialization>{}(
                         nonemptyList(indirect(Parser<DataStmtValue>{}))) /
            "/"))

// R745 private-components-stmt -> PRIVATE
// R747 binding-private-stmt -> PRIVATE
TYPE_PARSER("PRIVATE" >> construct<PrivateStmt>{})

// R746 type-bound-procedure-part ->
//        contains-stmt [binding-private-stmt] [type-bound-proc-binding]...
TYPE_CONTEXT_PARSER("type bound procedure part"_en_US,
    construct<TypeBoundProcedurePart>{}(statement(containsStmt),
        maybe(statement(Parser<PrivateStmt>{})),
        many(statement(Parser<TypeBoundProcBinding>{}))))

// R748 type-bound-proc-binding ->
//        type-bound-procedure-stmt | type-bound-generic-stmt |
//        final-procedure-stmt
TYPE_PARSER(recovery(
    construct<TypeBoundProcBinding>{}(Parser<TypeBoundProcedureStmt>{}) ||
        construct<TypeBoundProcBinding>{}(Parser<TypeBoundGenericStmt>{}) ||
        construct<TypeBoundProcBinding>{}(Parser<FinalProcedureStmt>{}),
    construct<TypeBoundProcBinding>{}(stmtErrorRecovery)))

// R749 type-bound-procedure-stmt ->
//        PROCEDURE [[, bind-attr-list] ::] type-bound-proc-decl-list |
//        PROCEDURE ( interface-name ) , bind-attr-list :: binding-name-list
TYPE_CONTEXT_PARSER("type bound PROCEDURE statement"_en_US,
    "PROCEDURE" >>
        (construct<TypeBoundProcedureStmt>{}(
             construct<TypeBoundProcedureStmt::WithInterface>{}(
                 parenthesized(name) / ",",
                 nonemptyList(Parser<BindAttr>{}) / "::",
                 nonemptyList(name))) ||
            construct<TypeBoundProcedureStmt>{}(
                construct<TypeBoundProcedureStmt::WithoutInterface>{}(
                    optionalListBeforeColons(Parser<BindAttr>{}),
                    nonemptyList(Parser<TypeBoundProcDecl>{})))))

// R750 type-bound-proc-decl -> binding-name [=> procedure-name]
TYPE_PARSER(construct<TypeBoundProcDecl>{}(name, maybe("=>" >> name)))

// R751 type-bound-generic-stmt ->
//        GENERIC [, access-spec] :: generic-spec => binding-name-list
TYPE_CONTEXT_PARSER("type bound GENERIC statement"_en_US,
    "GENERIC" >> construct<TypeBoundGenericStmt>{}(maybe("," >> accessSpec),
                     "::" >> indirect(genericSpec), "=>" >> nonemptyList(name)))

// R752 bind-attr ->
//        access-spec | DEFERRED | NON_OVERRIDABLE | NOPASS | PASS [(arg-name)]
TYPE_PARSER(construct<BindAttr>{}(accessSpec) ||
    "DEFERRED" >> construct<BindAttr>{}(construct<BindAttr::Deferred>{}) ||
    "NON_OVERRIDABLE" >>
        construct<BindAttr>{}(construct<BindAttr::Non_Overridable>{}) ||
    construct<BindAttr>{}(noPass) || construct<BindAttr>{}(pass))

// R753 final-procedure-stmt -> FINAL [::] final-subroutine-name-list
TYPE_CONTEXT_PARSER("FINAL statement"_en_US,
    "FINAL" >> maybe("::"_tok) >>
        construct<FinalProcedureStmt>{}(nonemptyList(name)))

// R754 derived-type-spec -> type-name [(type-param-spec-list)]
TYPE_PARSER(construct<DerivedTypeSpec>{}(
    name, defaulted(parenthesized(nonemptyList(Parser<TypeParamSpec>{})))))

// R755 type-param-spec -> [keyword =] type-param-value
TYPE_PARSER(construct<TypeParamSpec>{}(maybe(keyword / "="), typeParamValue))

// R756 structure-constructor -> derived-type-spec ( [component-spec-list] )
TYPE_PARSER((construct<StructureConstructor>{}(derivedTypeSpec,
                 parenthesized(optionalList(Parser<ComponentSpec>{}))) ||
                // This alternative corrects misrecognition of the
                // component-spec-list as the type-param-spec-list in
                // derived-type-spec.
                construct<StructureConstructor>{}(
                    construct<DerivedTypeSpec>{}(
                        name, construct<std::list<TypeParamSpec>>{}),
                    parenthesized(optionalList(Parser<ComponentSpec>{})))) /
    !"("_tok)

// R757 component-spec -> [keyword =] component-data-source
TYPE_PARSER(construct<ComponentSpec>{}(
    maybe(keyword / "="), Parser<ComponentDataSource>{}))

// R758 component-data-source -> expr | data-target | proc-target
TYPE_PARSER(construct<ComponentDataSource>{}(indirect(expr)))

// R759 enum-def ->
//        enum-def-stmt enumerator-def-stmt [enumerator-def-stmt]...
//        end-enum-stmt
TYPE_CONTEXT_PARSER("enum definition"_en_US,
    construct<EnumDef>{}(statement(Parser<EnumDefStmt>{}),
        some(statement(Parser<EnumeratorDefStmt>{})),
        statement(Parser<EndEnumStmt>{})))

// R760 enum-def-stmt -> ENUM, BIND(C)
TYPE_PARSER("ENUM , BIND ( C )" >> construct<EnumDefStmt>{})

// R761 enumerator-def-stmt -> ENUMERATOR [::] enumerator-list
TYPE_CONTEXT_PARSER("ENUMERATOR statement"_en_US,
    construct<EnumeratorDefStmt>{}(
        "ENUMERATOR" >> maybe("::"_tok) >> nonemptyList(Parser<Enumerator>{})))

// R762 enumerator -> named-constant [= scalar-int-constant-expr]
TYPE_PARSER(
    construct<Enumerator>{}(namedConstant, maybe("=" >> scalarIntConstantExpr)))

// R763 end-enum-stmt -> END ENUM
TYPE_PARSER(recovery("END ENUM"_tok, "END" >> SkipTo<'\n'>{}) >>
    construct<EndEnumStmt>{})

// R764 boz-literal-constant -> binary-constant | octal-constant | hex-constant
// R765 binary-constant -> B ' digit [digit]... ' | B " digit [digit]... "
// R766 octal-constant -> O ' digit [digit]... ' | O " digit [digit]... "
// R767 hex-constant ->
//        Z ' hex-digit [hex-digit]... ' | Z " hex-digit [hex-digit]... "
// extension: X accepted for Z
// extension: BOZX suffix accepted
TYPE_PARSER(construct<BOZLiteralConstant>{}(BOZLiteral{}))

// R1124 do-variable -> scalar-int-variable-name
constexpr auto doVariable = scalar(integer(name));

template<typename PA> inline constexpr auto loopBounds(const PA &p) {
  return construct<LoopBounds<typename PA::resultType>>{}(
      doVariable / "=", p / ",", p, maybe("," >> p));
}

// R769 array-constructor -> (/ ac-spec /) | lbracket ac-spec rbracket
TYPE_CONTEXT_PARSER("array constructor"_en_US,
    construct<ArrayConstructor>{}(
        "(/" >> Parser<AcSpec>{} / "/)" || bracketed(Parser<AcSpec>{})))

// R770 ac-spec -> type-spec :: | [type-spec ::] ac-value-list
TYPE_PARSER(construct<AcSpec>{}(maybe(indirect(typeSpec) / "::"),
                nonemptyList(Parser<AcValue>{})) ||
    construct<AcSpec>{}(indirect(typeSpec) / "::"))

// R773 ac-value -> expr | ac-implied-do
TYPE_PARSER(
    // PGI/Intel extension: accept triplets in array constructors
    extension(construct<AcValue>{}(construct<AcValue::Triplet>{}(
        scalarIntExpr, ":" >> scalarIntExpr, maybe(":" >> scalarIntExpr)))) ||
    construct<AcValue>{}(indirect(expr)) ||
    construct<AcValue>{}(indirect(Parser<AcImpliedDo>{})))

// R774 ac-implied-do -> ( ac-value-list , ac-implied-do-control )
TYPE_PARSER(parenthesized(construct<AcImpliedDo>{}(
    nonemptyList(Parser<AcValue>{} / lookAhead(","_tok)),
    "," >> Parser<AcImpliedDoControl>{})))

// R775 ac-implied-do-control ->
//        [integer-type-spec ::] ac-do-variable = scalar-int-expr ,
//        scalar-int-expr [, scalar-int-expr]
// R776 ac-do-variable -> do-variable
TYPE_PARSER(construct<AcImpliedDoControl>{}(
    maybe(integerTypeSpec / "::"), loopBounds(scalarIntExpr)))

// R801 type-declaration-stmt ->
//        declaration-type-spec [[, attr-spec]... ::] entity-decl-list
TYPE_PARSER(construct<TypeDeclarationStmt>{}(declarationTypeSpec,
                optionalListBeforeColons(Parser<AttrSpec>{}),
                nonemptyList(entityDecl)) ||
    // PGI-only extension: don't require the colons
    // N.B.: The standard requires the colons if the entity
    // declarations contain initializers.
    extension(construct<TypeDeclarationStmt>{}(declarationTypeSpec,
        defaulted("," >> nonemptyList(Parser<AttrSpec>{})),
        "," >> nonemptyList(entityDecl))))

// R802 attr-spec ->
//        access-spec | ALLOCATABLE | ASYNCHRONOUS |
//        CODIMENSION lbracket coarray-spec rbracket | CONTIGUOUS |
//        DIMENSION ( array-spec ) | EXTERNAL | INTENT ( intent-spec ) |
//        INTRINSIC | language-binding-spec | OPTIONAL | PARAMETER | POINTER |
//        PROTECTED | SAVE | TARGET | VALUE | VOLATILE
constexpr auto optional = "OPTIONAL" >> construct<Optional>{};
constexpr auto protectedAttr = "PROTECTED" >> construct<Protected>{};
constexpr auto save = "SAVE" >> construct<Save>{};
TYPE_PARSER(construct<AttrSpec>{}(accessSpec) ||
    construct<AttrSpec>{}(allocatable) ||
    construct<AttrSpec>{}("ASYNCHRONOUS" >> construct<Asynchronous>{}) ||
    construct<AttrSpec>{}("CODIMENSION" >> coarraySpec) ||
    construct<AttrSpec>{}(contiguous) ||
    construct<AttrSpec>{}("DIMENSION" >> arraySpec) ||
    construct<AttrSpec>{}("EXTERNAL" >> construct<External>{}) ||
    construct<AttrSpec>{}("INTENT" >> parenthesized(intentSpec)) ||
    construct<AttrSpec>{}("INTRINSIC" >> construct<Intrinsic>{}) ||
    construct<AttrSpec>{}(languageBindingSpec) ||
    construct<AttrSpec>{}(optional) ||
    construct<AttrSpec>{}("PARAMETER" >> construct<Parameter>{}) ||
    construct<AttrSpec>{}(pointer) || construct<AttrSpec>{}(protectedAttr) ||
    construct<AttrSpec>{}(save) ||
    construct<AttrSpec>{}("TARGET" >> construct<Target>{}) ||
    construct<AttrSpec>{}("VALUE" >> construct<Value>{}) ||
    construct<AttrSpec>{}("VOLATILE" >> construct<Volatile>{}))

// R804 object-name -> name
constexpr auto objectName = name;

// R803 entity-decl ->
//        object-name [( array-spec )] [lbracket coarray-spec rbracket]
//          [* char-length] [initialization] |
//        function-name [* char-length]
TYPE_PARSER(construct<EntityDecl>{}(objectName, maybe(arraySpec),
    maybe(coarraySpec), maybe("*" >> charLength), maybe(initialization)))

// R806 null-init -> function-reference
// TODO: confirm in semantics that NULL still intrinsic in this scope
TYPE_PARSER("NULL ( )" >> construct<NullInit>{} / !"("_tok)

// R807 access-spec -> PUBLIC | PRIVATE
TYPE_PARSER(
    "PUBLIC" >> construct<AccessSpec>{}(pure(AccessSpec::Kind::Public)) ||
    "PRIVATE" >> construct<AccessSpec>{}(pure(AccessSpec::Kind::Private)))

// R808 language-binding-spec ->
//        BIND ( C [, NAME = scalar-default-char-constant-expr] )
// R1528 proc-language-binding-spec -> language-binding-spec
TYPE_PARSER(construct<LanguageBindingSpec>{}(
    "BIND ( C" >> maybe(", NAME =" >> scalarDefaultCharConstantExpr) / ")"))

// R809 coarray-spec -> deferred-coshape-spec-list | explicit-coshape-spec
// N.B. Bracketed here rather than around references, for consistency with
// array-spec.
TYPE_PARSER(
    construct<CoarraySpec>{}(bracketed(Parser<DeferredCoshapeSpecList>{})) ||
    construct<CoarraySpec>{}(bracketed(Parser<ExplicitCoshapeSpec>{})))

// R810 deferred-coshape-spec -> :
// deferred-coshape-spec-list - just a list of colons
int listLength(std::list<Success> &&xs) { return xs.size(); }

TYPE_PARSER(construct<DeferredCoshapeSpecList>{}(
    applyFunction(listLength, nonemptyList(":"_tok))))

// R811 explicit-coshape-spec ->
//        [[lower-cobound :] upper-cobound ,]... [lower-cobound :] *
// R812 lower-cobound -> specification-expr
// R813 upper-cobound -> specification-expr
TYPE_PARSER(construct<ExplicitCoshapeSpec>{}(
    many(explicitShapeSpec / ","), maybe(specificationExpr / ":") / "*"))

// R815 array-spec ->
//        explicit-shape-spec-list | assumed-shape-spec-list |
//        deferred-shape-spec-list | assumed-size-spec | implied-shape-spec |
//        implied-shape-or-assumed-size-spec | assumed-rank-spec
// N.B. Parenthesized here rather than around references to avoid
// a need for forced look-ahead.
TYPE_PARSER(
    construct<ArraySpec>{}(parenthesized(nonemptyList(explicitShapeSpec))) ||
    construct<ArraySpec>{}(
        parenthesized(nonemptyList(Parser<AssumedShapeSpec>{}))) ||
    construct<ArraySpec>{}(parenthesized(deferredShapeSpecList)) ||
    construct<ArraySpec>{}(parenthesized(Parser<AssumedSizeSpec>{})) ||
    construct<ArraySpec>{}(parenthesized(Parser<ImpliedShapeSpec>{})) ||
    construct<ArraySpec>{}(parenthesized(Parser<AssumedRankSpec>{})))

// R816 explicit-shape-spec -> [lower-bound :] upper-bound
// R817 lower-bound -> specification-expr
// R818 upper-bound -> specification-expr
TYPE_PARSER(construct<ExplicitShapeSpec>{}(
    maybe(specificationExpr / ":"), specificationExpr))

// R819 assumed-shape-spec -> [lower-bound] :
TYPE_PARSER(construct<AssumedShapeSpec>{}(maybe(specificationExpr) / ":"))

// R820 deferred-shape-spec -> :
// deferred-shape-spec-list - just a list of colons
TYPE_PARSER(construct<DeferredShapeSpecList>{}(
    applyFunction(listLength, nonemptyList(":"_tok))))

// R821 assumed-implied-spec -> [lower-bound :] *
TYPE_PARSER(
    construct<AssumedImpliedSpec>{}(maybe(specificationExpr / ":") / "*"))

// R822 assumed-size-spec -> explicit-shape-spec-list , assumed-implied-spec
TYPE_PARSER(construct<AssumedSizeSpec>{}(
    nonemptyList(explicitShapeSpec) / ",", assumedImpliedSpec))

// R823 implied-shape-or-assumed-size-spec -> assumed-implied-spec
// R824 implied-shape-spec -> assumed-implied-spec , assumed-implied-spec-list
// I.e., when the assumed-implied-spec-list has a single item, it constitutes an
// implied-shape-or-assumed-size-spec; otherwise, an implied-shape-spec.
TYPE_PARSER(construct<ImpliedShapeSpec>{}(nonemptyList(assumedImpliedSpec)))

// R825 assumed-rank-spec -> ..
TYPE_PARSER(".." >> construct<AssumedRankSpec>{})

// R826 intent-spec -> IN | OUT | INOUT
TYPE_PARSER(
    construct<IntentSpec>{}("IN OUT" >> pure(IntentSpec::Intent::InOut) ||
        "IN" >> pure(IntentSpec::Intent::In) ||
        "OUT" >> pure(IntentSpec::Intent::Out)))

// R827 access-stmt -> access-spec [[::] access-id-list]
TYPE_PARSER(construct<AccessStmt>{}(
    accessSpec, defaulted(maybe("::"_tok) >> nonemptyList(Parser<AccessId>{}))))

// R828 access-id -> access-name | generic-spec
TYPE_PARSER(construct<AccessId>{}(indirect(genericSpec)) ||
    construct<AccessId>{}(name))  // initially ambiguous with genericSpec

// R829 allocatable-stmt -> ALLOCATABLE [::] allocatable-decl-list
TYPE_PARSER("ALLOCATABLE" >> maybe("::"_tok) >>
    construct<AllocatableStmt>{}(nonemptyList(Parser<ObjectDecl>{})))

// R830 allocatable-decl ->
//        object-name [( array-spec )] [lbracket coarray-spec rbracket]
// R860 target-decl ->
//        object-name [( array-spec )] [lbracket coarray-spec rbracket]
TYPE_PARSER(
    construct<ObjectDecl>{}(objectName, maybe(arraySpec), maybe(coarraySpec)))

// R831 asynchronous-stmt -> ASYNCHRONOUS [::] object-name-list
TYPE_PARSER("ASYNCHRONOUS" >> maybe("::"_tok) >>
    construct<AsynchronousStmt>{}(nonemptyList(objectName)))

// R832 bind-stmt -> language-binding-spec [::] bind-entity-list
TYPE_PARSER(construct<BindStmt>{}(
    languageBindingSpec / maybe("::"_tok), nonemptyList(Parser<BindEntity>{})))

// R833 bind-entity -> entity-name | / common-block-name /
TYPE_PARSER(construct<BindEntity>{}(pure(BindEntity::Kind::Object), name) ||
    "/" >> construct<BindEntity>{}(pure(BindEntity::Kind::Common), name) / "/")

// R834 codimension-stmt -> CODIMENSION [::] codimension-decl-list
TYPE_PARSER("CODIMENSION" >> maybe("::"_tok) >>
    construct<CodimensionStmt>{}(nonemptyList(Parser<CodimensionDecl>{})))

// R835 codimension-decl -> coarray-name lbracket coarray-spec rbracket
TYPE_PARSER(construct<CodimensionDecl>{}(name, coarraySpec))

// R836 contiguous-stmt -> CONTIGUOUS [::] object-name-list
TYPE_PARSER("CONTIGUOUS" >> maybe("::"_tok) >>
    construct<ContiguousStmt>{}(nonemptyList(objectName)))

// R837 data-stmt -> DATA data-stmt-set [[,] data-stmt-set]...
TYPE_CONTEXT_PARSER("DATA statement"_en_US,
    "DATA" >> construct<DataStmt>{}(
                  nonemptySeparated(Parser<DataStmtSet>{}, maybe(","_tok))))

// R838 data-stmt-set -> data-stmt-object-list / data-stmt-value-list /
TYPE_PARSER(construct<DataStmtSet>{}(nonemptyList(Parser<DataStmtObject>{}),
    "/"_tok >> nonemptyList(Parser<DataStmtValue>{}) / "/"))

// R839 data-stmt-object -> variable | data-implied-do
TYPE_PARSER(construct<DataStmtObject>{}(indirect(variable)) ||
    construct<DataStmtObject>{}(dataImpliedDo))

// R840 data-implied-do ->
//        ( data-i-do-object-list , [integer-type-spec ::] data-i-do-variable
//        = scalar-int-constant-expr , scalar-int-constant-expr
//        [, scalar-int-constant-expr] )
// R842 data-i-do-variable -> do-variable
TYPE_PARSER(parenthesized(construct<DataImpliedDo>{}(
    nonemptyList(Parser<DataIDoObject>{} / lookAhead(","_tok)) / ",",
    maybe(integerTypeSpec / "::"), loopBounds(scalarIntConstantExpr))))

// R841 data-i-do-object ->
//        array-element | scalar-structure-component | data-implied-do
TYPE_PARSER(construct<DataIDoObject>{}(scalar(indirect(designator))) ||
    construct<DataIDoObject>{}(indirect(dataImpliedDo)))

// R843 data-stmt-value -> [data-stmt-repeat *] data-stmt-constant
TYPE_PARSER(construct<DataStmtValue>{}(
    maybe(Parser<DataStmtRepeat>{} / "*"), Parser<DataStmtConstant>{}))

// R847 constant-subobject -> designator
// R846 int-constant-subobject -> constant-subobject
constexpr auto constantSubobject = constant(indirect(designator));

// R844 data-stmt-repeat -> scalar-int-constant | scalar-int-constant-subobject
// R607 int-constant -> constant
// Factored into:
//   constant -> literal-constant -> int-literal-constant   and
//   constant -> named-constant
TYPE_PARSER(construct<DataStmtRepeat>{}(intLiteralConstant) ||
    construct<DataStmtRepeat>{}(scalar(integer(constantSubobject))) ||
    construct<DataStmtRepeat>{}(scalar(integer(namedConstant))))

// R845 data-stmt-constant ->
//        scalar-constant | scalar-constant-subobject |
//        signed-int-literal-constant | signed-real-literal-constant |
//        null-init | initial-data-target | structure-constructor
// TODO: Some structure constructors can be misrecognized as array
// references into constant subobjects.
TYPE_PARSER(construct<DataStmtConstant>{}(scalar(Parser<ConstantValue>{})) ||
    construct<DataStmtConstant>{}(nullInit) ||
    construct<DataStmtConstant>{}(Parser<StructureConstructor>{}) ||
    construct<DataStmtConstant>{}(scalar(constantSubobject)) ||
    construct<DataStmtConstant>{}(signedRealLiteralConstant) ||
    construct<DataStmtConstant>{}(signedIntLiteralConstant) ||
    extension(construct<DataStmtConstant>{}(
        Parser<SignedComplexLiteralConstant>{})) ||
    construct<DataStmtConstant>{}(initialDataTarget))

// R848 dimension-stmt ->
//        DIMENSION [::] array-name ( array-spec )
//        [, array-name ( array-spec )]...
TYPE_CONTEXT_PARSER("DIMENSION statement"_en_US,
    "DIMENSION" >> maybe("::"_tok) >>
        construct<DimensionStmt>{}(nonemptyList(
            construct<DimensionStmt::Declaration>{}(name, arraySpec))))

// R849 intent-stmt -> INTENT ( intent-spec ) [::] dummy-arg-name-list
TYPE_CONTEXT_PARSER("INTENT statement"_en_US,
    "INTENT" >>
        construct<IntentStmt>{}(
            parenthesized(intentSpec) / maybe("::"_tok), nonemptyList(name)))

// R850 optional-stmt -> OPTIONAL [::] dummy-arg-name-list
TYPE_PARSER("OPTIONAL" >> maybe("::"_tok) >>
    construct<OptionalStmt>{}(nonemptyList(name)))

// R851 parameter-stmt -> PARAMETER ( named-constant-def-list )
// Legacy extension: omitted parentheses, no implicit typing from names
TYPE_CONTEXT_PARSER("PARAMETER statement"_en_US,
    construct<ParameterStmt>{}(
        "PARAMETER" >> parenthesized(nonemptyList(Parser<NamedConstantDef>{}))))
TYPE_CONTEXT_PARSER("old style PARAMETER statement"_en_US,
    extension(construct<OldParameterStmt>{}(
        "PARAMETER" >> nonemptyList(Parser<NamedConstantDef>{}))))

// R852 named-constant-def -> named-constant = constant-expr
TYPE_PARSER(construct<NamedConstantDef>{}(namedConstant, "=" >> constantExpr))

// R853 pointer-stmt -> POINTER [::] pointer-decl-list
TYPE_PARSER("POINTER" >> maybe("::"_tok) >>
    construct<PointerStmt>{}(nonemptyList(Parser<PointerDecl>{})))

// R854 pointer-decl ->
//        object-name [( deferred-shape-spec-list )] | proc-entity-name
TYPE_PARSER(
    construct<PointerDecl>{}(name, maybe(parenthesized(deferredShapeSpecList))))

// R855 protected-stmt -> PROTECTED [::] entity-name-list
TYPE_PARSER("PROTECTED" >> maybe("::"_tok) >>
    construct<ProtectedStmt>{}(nonemptyList(name)))

// R856 save-stmt -> SAVE [[::] saved-entity-list]
TYPE_PARSER("SAVE" >> construct<SaveStmt>{}(defaulted(maybe("::"_tok) >>
                          nonemptyList(Parser<SavedEntity>{}))))

// R857 saved-entity -> object-name | proc-pointer-name | / common-block-name /
// R858 proc-pointer-name -> name
// TODO: Distinguish Kind::ProcPointer and Kind::Object
TYPE_PARSER(construct<SavedEntity>{}(pure(SavedEntity::Kind::Object), name) ||
    "/" >>
        construct<SavedEntity>{}(pure(SavedEntity::Kind::Common), name) / "/")

// R859 target-stmt -> TARGET [::] target-decl-list
TYPE_PARSER("TARGET" >> maybe("::"_tok) >>
    construct<TargetStmt>{}(nonemptyList(Parser<ObjectDecl>{})))

// R861 value-stmt -> VALUE [::] dummy-arg-name-list
TYPE_PARSER(
    "VALUE" >> maybe("::"_tok) >> construct<ValueStmt>{}(nonemptyList(name)))

// R862 volatile-stmt -> VOLATILE [::] object-name-list
TYPE_PARSER("VOLATILE" >> maybe("::"_tok) >>
    construct<VolatileStmt>{}(nonemptyList(objectName)))

// R866 implicit-name-spec -> EXTERNAL | TYPE
constexpr auto implicitNameSpec = "EXTERNAL" >>
        pure(ImplicitStmt::ImplicitNoneNameSpec::External) ||
    "TYPE" >> pure(ImplicitStmt::ImplicitNoneNameSpec::Type);

// R863 implicit-stmt ->
//        IMPLICIT implicit-spec-list |
//        IMPLICIT NONE [( [implicit-name-spec-list] )]
TYPE_CONTEXT_PARSER("IMPLICIT statement"_en_US,
    "IMPLICIT" >>
        (construct<ImplicitStmt>{}(nonemptyList(Parser<ImplicitSpec>{})) ||
            construct<ImplicitStmt>{}("NONE" >>
                defaulted(parenthesized(optionalList(implicitNameSpec))))))

// R864 implicit-spec -> declaration-type-spec ( letter-spec-list )
// The variant form of declarationTypeSpec is meant to avoid misrecognition
// of a letter-spec as a simple parenthesized expression for kind or character
// length, e.g., PARAMETER(I=5,N=1); IMPLICIT REAL(I-N)(O-Z) vs.
// IMPLICIT REAL(I-N).  The variant form needs to attempt to reparse only
// types with optional parenthesized kind/length expressions, so derived
// type specs, DOUBLE PRECISION, and DOUBLE COMPLEX need not be considered.
constexpr auto noKindSelector = construct<std::optional<KindSelector>>{};
constexpr auto implicitSpecDeclarationTypeSpecRetry =
    construct<DeclarationTypeSpec>{}(
        "INTEGER" >> construct<IntrinsicTypeSpec>{}(
                         construct<IntegerTypeSpec>{}(noKindSelector)) ||
        "REAL" >> construct<IntrinsicTypeSpec>{}(
                      construct<IntrinsicTypeSpec::Real>{}(noKindSelector)) ||
        "COMPLEX" >>
            construct<IntrinsicTypeSpec>{}(
                construct<IntrinsicTypeSpec::Complex>{}(noKindSelector)) ||
        "CHARACTER" >> construct<IntrinsicTypeSpec>{}(
                           construct<IntrinsicTypeSpec::Character>{}(
                               construct<std::optional<CharSelector>>{})) ||
        "LOGICAL" >>
            construct<IntrinsicTypeSpec>{}(
                construct<IntrinsicTypeSpec::Logical>{}(noKindSelector)));

TYPE_PARSER(construct<ImplicitSpec>{}(declarationTypeSpec,
                parenthesized(nonemptyList(Parser<LetterSpec>{}))) ||
    construct<ImplicitSpec>{}(implicitSpecDeclarationTypeSpecRetry,
        parenthesized(nonemptyList(Parser<LetterSpec>{}))))

// R865 letter-spec -> letter [- letter]
TYPE_PARSER(space >> (construct<LetterSpec>{}(letter, maybe("-" >> letter)) ||
                         construct<LetterSpec>{}(otherIdChar,
                             construct<std::optional<const char *>>{})))

// R867 import-stmt ->
//        IMPORT [[::] import-name-list] |
//        IMPORT , ONLY : import-name-list | IMPORT , NONE | IMPORT , ALL
TYPE_CONTEXT_PARSER("IMPORT statement"_en_US,
    "IMPORT" >>
        (construct<ImportStmt>{}(
             ", ONLY :" >> pure(ImportStmt::Kind::Only), nonemptyList(name)) ||
            construct<ImportStmt>{}(", NONE" >> pure(ImportStmt::Kind::None)) ||
            construct<ImportStmt>{}(", ALL" >> pure(ImportStmt::Kind::All)) ||
            construct<ImportStmt>{}(maybe("::"_tok) >> optionalList(name))))

// R868 namelist-stmt ->
//        NAMELIST / namelist-group-name / namelist-group-object-list
//        [[,] / namelist-group-name / namelist-group-object-list]...
// R869 namelist-group-object -> variable-name
TYPE_PARSER("NAMELIST" >>
    construct<NamelistStmt>{}(nonemptySeparated(
        construct<NamelistStmt::Group>{}("/" >> name / "/", nonemptyList(name)),
        maybe(","_tok))))

// R870 equivalence-stmt -> EQUIVALENCE equivalence-set-list
// R871 equivalence-set -> ( equivalence-object , equivalence-object-list )
TYPE_PARSER("EQUIVALENCE" >>
    construct<EquivalenceStmt>{}(
        nonemptyList(parenthesized(nonemptyList(Parser<EquivalenceObject>{})))))

// R872 equivalence-object -> variable-name | array-element | substring
TYPE_PARSER(construct<EquivalenceObject>{}(indirect(designator)))

// R873 common-stmt ->
//        COMMON [/ [common-block-name] /] common-block-object-list
//        [[,] / [common-block-name] / common-block-object-list]...
TYPE_PARSER(
    "COMMON" >> construct<CommonStmt>{}(maybe("/" >> maybe(name) / "/"),
                    nonemptyList(Parser<CommonBlockObject>{}),
                    many(maybe(","_tok) >>
                        construct<CommonStmt::Block>{}("/" >> maybe(name) / "/",
                            nonemptyList(Parser<CommonBlockObject>{})))))

// R874 common-block-object -> variable-name [( array-spec )]
TYPE_PARSER(construct<CommonBlockObject>{}(name, maybe(arraySpec)))

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
TYPE_PARSER(construct<Designator>{}(substring) ||
    construct<Designator>{}(dataReference))

// R902 variable -> designator | function-reference
// This production is left-recursive in the case of a function reference
// (via procedure-designator -> proc-component-ref -> scalar-variable)
// so it is implemented iteratively here.  When a variable is a
// function-reference, the called function must return a pointer in order
// to be valid as a variable, but we can't know that yet here.  So we first
// parse a designator, and if it's not a substring, we then allow an
// (actual-arg-spec-list), followed by zero or more "% name (a-a-s-list)".
//
// It is not valid Fortran to immediately invoke the result of a call to
// a function that returns a bare pointer to a function, although that would
// be a reasonable extension.  This restriction means that adjacent actual
// argument lists cannot occur (e.g.*, f()()).  One must instead return a
// pointer to a derived type instance containing a procedure pointer
// component in order to accomplish roughly the same thing.
//
// Some function references with dummy arguments present will be
// misrecognized as array element designators and need to be corrected
// in semantic analysis.
template<> std::optional<Variable> Parser<Variable>::Parse(ParseState *state) {
  std::optional<Designator> desig{designator.Parse(state)};
  if (!desig.has_value()) {
    return {};
  }
  if (!desig->EndsInBareName()) {
    return {Variable{Indirection<Designator>{std::move(desig.value())}}};
  }
  static constexpr auto argList = parenthesized(optionalList(actualArgSpec));
  static constexpr auto tryArgList = attempt(argList);
  auto args = tryArgList.Parse(state);
  if (!args.has_value()) {
    return {Variable{Indirection<Designator>{std::move(desig.value())}}};
  }

  // Create a procedure-designator from the original designator and
  // combine it with the actual arguments as a function-reference.
  ProcedureDesignator pd{desig.value().ConvertToProcedureDesignator()};
  Variable var{Indirection<FunctionReference>{
      Call{std::move(pd), std::move(args.value())}}};

  // Repeatedly accept additional function calls through components of
  // a derived type result.
  struct ResultComponentCall {
    ResultComponentCall(ResultComponentCall &&) = default;
    ResultComponentCall &operator=(ResultComponentCall &&) = default;
    ResultComponentCall(Name &&n, std::list<ActualArgSpec> &&as)
      : name{std::move(n)}, args(std::move(as)) {}
    Name name;
    std::list<ActualArgSpec> args;
  };
  static constexpr auto resultComponentCall =
      attempt("%" >> construct<ResultComponentCall>{}(name, argList));
  while (auto more = resultComponentCall.Parse(state)) {
    var = Variable{Indirection<FunctionReference>{Call{
        ProcedureDesignator{ProcComponentRef{
            Scalar<Variable>{std::move(var)}, std::move(more.value().name)}},
        std::move(more.value().args)}}};
  }

  return {std::move(var)};
}

// R904 logical-variable -> variable
// Appears only as part of scalar-logical-variable.
constexpr auto scalarLogicalVariable = scalar(logical(variable));

// R905 char-variable -> variable
constexpr auto charVariable = construct<CharVariable>{}(variable);

// R906 default-char-variable -> variable
// Appears only as part of scalar-default-char-variable.
constexpr auto scalarDefaultCharVariable = scalar(defaultChar(variable));

// R907 int-variable -> variable
// Appears only as part of scalar-int-variable.
constexpr auto scalarIntVariable = scalar(integer(variable));

// R908 substring -> parent-string ( substring-range )
// R909 parent-string ->
//        scalar-variable-name | array-element | coindexed-named-object |
//        scalar-structure-component | scalar-char-literal-constant |
//        scalar-named-constant
TYPE_PARSER(construct<Substring>{}(
    dataReference, parenthesized(Parser<SubstringRange>{})))

TYPE_PARSER(construct<CharLiteralConstantSubstring>{}(
    charLiteralConstant, parenthesized(Parser<SubstringRange>{})))

// R910 substring-range -> [scalar-int-expr] : [scalar-int-expr]
TYPE_PARSER(construct<SubstringRange>{}(
    maybe(scalarIntExpr), ":" >> maybe(scalarIntExpr)))

// R1003 defined-unary-op -> . letter [letter]... .
// R1023 defined-binary-op -> . letter [letter]... .
// R1414 local-defined-operator -> defined-unary-op | defined-binary-op
// R1415 use-defined-operator -> defined-unary-op | defined-binary-op
// N.B. The name of the operator is captured without the periods around it.
TYPE_PARSER(space >> "."_ch >>
    construct<DefinedOpName>{}(sourced(some(letter) >> construct<Name>{})) /
        "."_ch)

// R911 data-ref -> part-ref [% part-ref]...
// R914 coindexed-named-object -> data-ref
// R917 array-element -> data-ref
constexpr struct StructureComponentName {
  using resultType = Name;
  static std::optional<Name> Parse(ParseState *state) {
    if (std::optional<Name> n{name.Parse(state)}) {
      if (const auto *user = state->userState()) {
        if (user->IsOldStructureComponent(n->source)) {
          return n;
        }
      }
    }
    return {};
  }
} structureComponentName;

constexpr auto percentOrDot = "%"_tok ||
    // legacy VAX extension for RECORD field access
    extension("."_tok / lookAhead(structureComponentName));

TYPE_PARSER(construct<DataReference>{}(
    nonemptySeparated(Parser<PartRef>{}, percentOrDot)))

// R912 part-ref -> part-name [( section-subscript-list )] [image-selector]
TYPE_PARSER(construct<PartRef>{}(name,
    defaulted(parenthesized(nonemptyList(Parser<SectionSubscript>{}))),
    maybe(Parser<ImageSelector>{})))

// R913 structure-component -> data-ref
TYPE_PARSER(construct<StructureComponent>{}(
    construct<DataReference>{}(some(Parser<PartRef>{} / percentOrDot)), name))

// R915 complex-part-designator -> designator % RE | designator % IM
// %RE and %IM are initially recognized as structure components.
constexpr auto complexPartDesignator =
    construct<ComplexPartDesignator>{}(dataReference);

// R916 type-param-inquiry -> designator % type-param-name
// Type parameter inquiries are initially recognized as structure components.
TYPE_PARSER(construct<TypeParamInquiry>{}(structureComponent))

// R918 array-section ->
//        data-ref [( substring-range )] | complex-part-designator
constexpr auto arraySection = construct<ArraySection>{}(designator);

// R919 subscript -> scalar-int-expr
constexpr auto subscript = scalarIntExpr;

// R923 vector-subscript -> int-expr
constexpr auto vectorSubscript = intExpr;

// R920 section-subscript -> subscript | subscript-triplet | vector-subscript
// N.B. The distinction that needs to be made between "subscript" and
// "vector-subscript" is deferred to semantic analysis.
TYPE_PARSER(construct<SectionSubscript>{}(Parser<SubscriptTriplet>{}) ||
    construct<SectionSubscript>{}(vectorSubscript) ||
    construct<SectionSubscript>{}(subscript))

// R921 subscript-triplet -> [subscript] : [subscript] [: stride]
TYPE_PARSER(construct<SubscriptTriplet>{}(
    maybe(subscript), ":" >> maybe(subscript), maybe(":" >> subscript)))

// R925 cosubscript -> scalar-int-expr
constexpr auto cosubscript = scalarIntExpr;

// R924 image-selector ->
//        lbracket cosubscript-list [, image-selector-spec-list] rbracket
TYPE_PARSER(bracketed(construct<ImageSelector>{}(nonemptyList(cosubscript),
    defaulted("," >> nonemptyList(Parser<ImageSelectorSpec>{})))))

// R1115 team-variable -> scalar-variable
constexpr auto teamVariable = scalar(indirect(variable));

// R926 image-selector-spec ->
//        STAT = stat-variable | TEAM = team-variable |
//        TEAM_NUMBER = scalar-int-expr
TYPE_PARSER("STAT =" >>
        construct<ImageSelectorSpec>{}(construct<ImageSelectorSpec::Stat>{}(
            scalar(integer(indirect(variable))))) ||
    "TEAM =" >> construct<ImageSelectorSpec>{}(
                    construct<ImageSelectorSpec::Team>{}(teamVariable)) ||
    "TEAM_NUMBER =" >>
        construct<ImageSelectorSpec>{}(
            construct<ImageSelectorSpec::Team_Number>{}(scalarIntExpr)))

// R927 allocate-stmt ->
//        ALLOCATE ( [type-spec ::] allocation-list [, alloc-opt-list] )
TYPE_CONTEXT_PARSER("ALLOCATE statement"_en_US,
    "ALLOCATE" >>
        parenthesized(construct<AllocateStmt>{}(maybe(typeSpec / "::"),
            nonemptyList(Parser<Allocation>{}),
            defaulted("," >> nonemptyList(Parser<AllocOpt>{})))))

// R928 alloc-opt ->
//        ERRMSG = errmsg-variable | MOLD = source-expr |
//        SOURCE = source-expr | STAT = stat-variable
// R931 source-expr -> expr
TYPE_PARSER("MOLD =" >>
        construct<AllocOpt>{}(construct<AllocOpt::Mold>{}(indirect(expr))) ||
    "SOURCE =" >>
        construct<AllocOpt>{}(construct<AllocOpt::Source>{}(indirect(expr))) ||
    construct<AllocOpt>{}(statOrErrmsg))

// R929 stat-variable -> scalar-int-variable
TYPE_PARSER(construct<StatVariable>{}(scalar(integer(variable))))

// R930 errmsg-variable -> scalar-default-char-variable
// R1207 iomsg-variable -> scalar-default-char-variable
constexpr auto msgVariable =
    construct<MsgVariable>{}(scalarDefaultCharVariable);

// R932 allocation ->
//        allocate-object [( allocate-shape-spec-list )]
//        [lbracket allocate-coarray-spec rbracket]
// TODO: allocate-shape-spec-list might be misrecognized as
// the final list of subscripts in allocate-object.
TYPE_PARSER(construct<Allocation>{}(Parser<AllocateObject>{},
    defaulted(parenthesized(nonemptyList(Parser<AllocateShapeSpec>{}))),
    maybe(bracketed(Parser<AllocateCoarraySpec>{}))))

// R933 allocate-object -> variable-name | structure-component
TYPE_PARSER(construct<AllocateObject>{}(structureComponent) ||
    construct<AllocateObject>{}(name / !"="_tok))

// R935 lower-bound-expr -> scalar-int-expr
// R936 upper-bound-expr -> scalar-int-expr
constexpr auto boundExpr = scalarIntExpr;

// R934 allocate-shape-spec -> [lower-bound-expr :] upper-bound-expr
// R938 allocate-coshape-spec -> [lower-bound-expr :] upper-bound-expr
TYPE_PARSER(construct<AllocateShapeSpec>{}(maybe(boundExpr / ":"), boundExpr))

// R937 allocate-coarray-spec ->
//      [allocate-coshape-spec-list ,] [lower-bound-expr :] *
TYPE_PARSER(construct<AllocateCoarraySpec>{}(
    defaulted(nonemptyList(Parser<AllocateShapeSpec>{}) / ","),
    maybe(boundExpr / ":") / "*"))

// R939 nullify-stmt -> NULLIFY ( pointer-object-list )
TYPE_CONTEXT_PARSER("NULLIFY statement"_en_US,
    "NULLIFY" >> parenthesized(construct<NullifyStmt>{}(
                     nonemptyList(Parser<PointerObject>{}))))

// R940 pointer-object ->
//        variable-name | structure-component | proc-pointer-name
TYPE_PARSER(construct<PointerObject>{}(structureComponent) ||
    construct<PointerObject>{}(name))

// R941 deallocate-stmt ->
//        DEALLOCATE ( allocate-object-list [, dealloc-opt-list] )
TYPE_CONTEXT_PARSER("DEALLOCATE statement"_en_US,
    "DEALLOCATE" >> parenthesized(construct<DeallocateStmt>{}(
                        nonemptyList(Parser<AllocateObject>{}),
                        defaulted("," >> nonemptyList(statOrErrmsg)))))

// R942 dealloc-opt -> STAT = stat-variable | ERRMSG = errmsg-variable
// R1165 sync-stat -> STAT = stat-variable | ERRMSG = errmsg-variable
TYPE_PARSER("STAT =" >> construct<StatOrErrmsg>{}(statVariable) ||
    "ERRMSG =" >> construct<StatOrErrmsg>{}(msgVariable))

// R1001 primary ->
//         literal-constant | designator | array-constructor |
//         structure-constructor | function-reference | type-param-inquiry |
//         type-param-name | ( expr )
constexpr auto primary =
    construct<Expr>{}(indirect(Parser<CharLiteralConstantSubstring>{})) ||
    construct<Expr>{}(literalConstant) ||
    construct<Expr>{}(construct<Expr::Parentheses>{}(parenthesized(expr))) ||
    construct<Expr>{}(indirect(functionReference) / !"("_tok) ||
    construct<Expr>{}(designator / !"("_tok) ||
    construct<Expr>{}(Parser<StructureConstructor>{}) ||
    construct<Expr>{}(Parser<ArrayConstructor>{}) ||
    construct<Expr>{}(indirect(Parser<TypeParamInquiry>{})) ||  // occulted
    // PGI/XLF extension: COMPLEX constructor (x,y)
    extension(construct<Expr>{}(parenthesized(
        construct<Expr::ComplexConstructor>{}(expr, "," >> expr)))) ||
    extension(construct<Expr>{}("%LOC" >>
        parenthesized(construct<Expr::PercentLoc>{}(indirect(variable)))));

// R1002 level-1-expr -> [defined-unary-op] primary
// TODO: Reasonable extension: permit multiple defined-unary-ops
constexpr auto level1Expr = construct<Expr>{}(construct<Expr::DefinedUnary>{}(
                                definedOpName, primary)) ||
    primary ||
    extension(
        "+" >> construct<Expr>{}(construct<Expr::UnaryPlus>{}(primary))) ||
    extension("-" >> construct<Expr>{}(construct<Expr::Negate>{}(primary)));

// R1004 mult-operand -> level-1-expr [power-op mult-operand]
// R1007 power-op -> **
// Exponentiation (**) is Fortran's only right-associative binary operation.
constexpr struct MultOperand {
  using resultType = Expr;
  constexpr MultOperand() {}
  static std::optional<Expr> Parse(ParseState *);
} multOperand;

std::optional<Expr> MultOperand::Parse(ParseState *state) {
  std::optional<Expr> result{level1Expr.Parse(state)};
  if (result) {
    static constexpr auto op = attempt("**"_tok);
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
  static std::optional<Expr> Parse(ParseState *state) {
    std::optional<Expr> result{multOperand.Parse(state)};
    if (result) {
      std::function<Expr(Expr &&)> multiply{[&result](Expr &&right) {
        return Expr{
            Expr::Multiply(std::move(result).value(), std::move(right))};
      }},
          divide{[&result](Expr &&right) {
            return Expr{
                Expr::Divide(std::move(result).value(), std::move(right))};
          }};
      auto more = "*" >> applyLambda(multiply, multOperand) ||
          "/" >> applyLambda(divide, multOperand);
      while (std::optional<Expr> next{attempt(more).Parse(state)}) {
        result = std::move(next);
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
  static std::optional<Expr> Parse(ParseState *state) {
    static constexpr auto unary =
        "+" >> construct<Expr>{}(construct<Expr::UnaryPlus>{}(addOperand)) ||
        "-" >> construct<Expr>{}(construct<Expr::Negate>{}(addOperand)) ||
        addOperand;
    std::optional<Expr> result{unary.Parse(state)};
    if (result) {
      std::function<Expr(Expr &&)> add{[&result](Expr &&right) {
        return Expr{Expr::Add(std::move(result).value(), std::move(right))};
      }},
          subtract{[&result](Expr &&right) {
            return Expr{
                Expr::Subtract(std::move(result).value(), std::move(right))};
          }};
      auto more = "+" >> applyLambda(add, addOperand) ||
          "-" >> applyLambda(subtract, addOperand);
      while (std::optional<Expr> next{attempt(more).Parse(state)}) {
        result = std::move(next);
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
  static std::optional<Expr> Parse(ParseState *state) {
    std::optional<Expr> result{level2Expr.Parse(state)};
    if (result) {
      std::function<Expr(Expr &&)> concat{[&result](Expr &&right) {
        return Expr{Expr::Concat(std::move(result).value(), std::move(right))};
      }};
      auto more = "//" >> applyLambda(concat, level2Expr);
      while (std::optional<Expr> next{attempt(more).Parse(state)}) {
        result = std::move(next);
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
  static std::optional<Expr> Parse(ParseState *state) {
    std::optional<Expr> result{level3Expr.Parse(state)};
    if (result) {
      std::function<Expr(Expr &&)> lt{[&result](Expr &&right) {
        return Expr{Expr::LT(std::move(result).value(), std::move(right))};
      }},
          le{[&result](Expr &&right) {
            return Expr{Expr::LE(std::move(result).value(), std::move(right))};
          }},
          eq{[&result](Expr &&right) {
            return Expr{Expr::EQ(std::move(result).value(), std::move(right))};
          }},
          ne{[&result](Expr &&right) {
            return Expr{Expr::NE(std::move(result).value(), std::move(right))};
          }},
          ge{[&result](Expr &&right) {
            return Expr{Expr::GE(std::move(result).value(), std::move(right))};
          }},
          gt{[&result](Expr &&right) {
            return Expr{Expr::GT(std::move(result).value(), std::move(right))};
          }};
      auto more = (".LT."_tok || "<"_tok) >> applyLambda(lt, level3Expr) ||
          (".LE."_tok || "<="_tok) >> applyLambda(le, level3Expr) ||
          (".EQ."_tok || "=="_tok) >> applyLambda(eq, level3Expr) ||
          (".NE."_tok || "/="_tok ||
              extension(
                  "<>"_tok /* PGI/Cray extension; Cray also has .LG. */)) >>
              applyLambda(ne, level3Expr) ||
          (".GE."_tok || ">="_tok) >> applyLambda(ge, level3Expr) ||
          (".GT."_tok || ">"_tok) >> applyLambda(gt, level3Expr);
      if (std::optional<Expr> next{attempt(more).Parse(state)}) {
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
  static std::optional<Expr> Parse(ParseState *);
} andOperand;

std::optional<Expr> AndOperand::Parse(ParseState *state) {
  static constexpr auto op = attempt(".NOT."_tok);
  int complements{0};
  while (op.Parse(state)) {
    ++complements;
  }
  std::optional<Expr> result{level4Expr.Parse(state)};
  if (result.has_value()) {
    while (complements-- > 0) {
      result = Expr{Expr::NOT{std::move(*result)}};
    }
  }
  return result;
}

// R1015 or-operand -> [or-operand and-op] and-operand
// R1019 and-op -> .AND.
// .AND. is left-associative
constexpr struct OrOperand {
  using resultType = Expr;
  constexpr OrOperand() {}
  static std::optional<Expr> Parse(ParseState *state) {
    std::optional<Expr> result{andOperand.Parse(state)};
    if (result) {
      std::function<Expr(Expr &&)> logicalAnd{[&result](Expr &&right) {
        return Expr{Expr::AND(std::move(result).value(), std::move(right))};
      }};
      auto more = ".AND." >> applyLambda(logicalAnd, andOperand);
      while (std::optional<Expr> next{attempt(more).Parse(state)}) {
        result = std::move(next);
      }
    }
    return result;
  }
} orOperand;

// R1016 equiv-operand -> [equiv-operand or-op] or-operand
// R1020 or-op -> .OR.
// .OR. is left-associative
static constexpr struct EquivOperand {
  using resultType = Expr;
  constexpr EquivOperand() {}
  static std::optional<Expr> Parse(ParseState *state) {
    std::optional<Expr> result{orOperand.Parse(state)};
    if (result) {
      std::function<Expr(Expr &&)> logicalOr{[&result](Expr &&right) {
        return Expr{Expr::OR(std::move(result).value(), std::move(right))};
      }};
      auto more = ".OR." >> applyLambda(logicalOr, orOperand);
      while (std::optional<Expr> next{attempt(more).Parse(state)}) {
        result = std::move(next);
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
  static std::optional<Expr> Parse(ParseState *state) {
    std::optional<Expr> result{equivOperand.Parse(state)};
    if (result) {
      std::function<Expr(Expr &&)> eqv{[&result](Expr &&right) {
        return Expr{Expr::EQV(std::move(result).value(), std::move(right))};
      }},
          neqv{[&result](Expr &&right) {
            return Expr{
                Expr::NEQV(std::move(result).value(), std::move(right))};
          }},
          logicalXor{[&result](Expr &&right) {
            return Expr{Expr::XOR(std::move(result).value(), std::move(right))};
          }};
      auto more = ".EQV." >> applyLambda(eqv, equivOperand) ||
          ".NEQV." >> applyLambda(neqv, equivOperand) ||
          extension(".XOR." >> applyLambda(logicalXor, equivOperand));
      while (std::optional<Expr> next{attempt(more).Parse(state)}) {
        result = std::move(next);
      }
    }
    return result;
  }
} level5Expr;

// R1022 expr -> [expr defined-binary-op] level-5-expr
// Defined binary operators associate leftwards.
template<> std::optional<Expr> Parser<Expr>::Parse(ParseState *state) {
  std::optional<Expr> result{level5Expr.Parse(state)};
  if (result) {
    std::function<Expr(DefinedOpName &&, Expr &&)> defBinOp{
        [&result](DefinedOpName &&op, Expr &&right) {
          return Expr{Expr::DefinedBinary(
              std::move(op), std::move(result).value(), std::move(right))};
        }};
    auto more = applyLambda(defBinOp, definedOpName, level5Expr);
    while (std::optional<Expr> next{attempt(more).Parse(state)}) {
      result = std::move(next);
    }
  }
  return result;
}

// R1028 specification-expr -> scalar-int-expr
TYPE_PARSER(construct<SpecificationExpr>{}(scalarIntExpr))

// R1032 assignment-stmt -> variable = expr
TYPE_CONTEXT_PARSER("assignment statement"_en_US,
    construct<AssignmentStmt>{}(variable / "=", expr))

// R1033 pointer-assignment-stmt ->
//         data-pointer-object [( bounds-spec-list )] => data-target |
//         data-pointer-object ( bounds-remapping-list ) => data-target |
//         proc-pointer-object => proc-target
// R1034 data-pointer-object ->
//         variable-name | scalar-variable % data-pointer-component-name
// R1038 proc-pointer-object -> proc-pointer-name | proc-component-ref
//
// A distinction can't be made at the time of the initial parse between
// data-pointer-object and proc-pointer-object, or between data-target
// and proc-target.
TYPE_CONTEXT_PARSER("pointer assignment statement"_en_US,
    construct<PointerAssignmentStmt>{}(variable,
        parenthesized(nonemptyList(Parser<BoundsRemapping>{})), "=>" >> expr) ||
        construct<PointerAssignmentStmt>{}(variable,
            defaulted(parenthesized(nonemptyList(Parser<BoundsSpec>{}))),
            "=>" >> expr))

// R1035 bounds-spec -> lower-bound-expr :
TYPE_PARSER(construct<BoundsSpec>{}(boundExpr / ":"))

// R1036 bounds-remapping -> lower-bound-expr : upper-bound-expr
TYPE_PARSER(construct<BoundsRemapping>{}(boundExpr / ":", boundExpr))

// R1039 proc-component-ref -> scalar-variable % procedure-component-name
// N.B. Never parsed as such; instead, reconstructed as necessary from
// parses of variable.

// R1041 where-stmt -> WHERE ( mask-expr ) where-assignment-stmt
// R1045 where-assignment-stmt -> assignment-stmt
// R1046 mask-expr -> logical-expr
TYPE_CONTEXT_PARSER("WHERE statement"_en_US,
    "WHERE" >>
        construct<WhereStmt>{}(parenthesized(logicalExpr), assignmentStmt))

// R1042 where-construct ->
//         where-construct-stmt [where-body-construct]...
//         [masked-elsewhere-stmt [where-body-construct]...]...
//         [elsewhere-stmt [where-body-construct]...] end-where-stmt
TYPE_CONTEXT_PARSER("WHERE construct"_en_US,
    construct<WhereConstruct>{}(statement(Parser<WhereConstructStmt>{}),
        many(whereBodyConstruct),
        many(construct<WhereConstruct::MaskedElsewhere>{}(
            statement(Parser<MaskedElsewhereStmt>{}),
            many(whereBodyConstruct))),
        maybe(construct<WhereConstruct::Elsewhere>{}(
            statement(Parser<ElsewhereStmt>{}), many(whereBodyConstruct))),
        statement(Parser<EndWhereStmt>{})))

// R1043 where-construct-stmt -> [where-construct-name :] WHERE ( mask-expr )
TYPE_CONTEXT_PARSER("WHERE construct statement"_en_US,
    construct<WhereConstructStmt>{}(
        maybe(name / ":"), "WHERE" >> parenthesized(logicalExpr)))

// R1044 where-body-construct ->
//         where-assignment-stmt | where-stmt | where-construct
TYPE_PARSER(construct<WhereBodyConstruct>{}(statement(assignmentStmt)) ||
    construct<WhereBodyConstruct>{}(statement(whereStmt)) ||
    construct<WhereBodyConstruct>{}(indirect(whereConstruct)))

// R1047 masked-elsewhere-stmt ->
//         ELSEWHERE ( mask-expr ) [where-construct-name]
TYPE_CONTEXT_PARSER("masked ELSEWHERE statement"_en_US,
    "ELSE WHERE" >> construct<MaskedElsewhereStmt>{}(
                        parenthesized(logicalExpr), maybe(name)))

// R1048 elsewhere-stmt -> ELSEWHERE [where-construct-name]
TYPE_CONTEXT_PARSER("ELSEWHERE statement"_en_US,
    "ELSE WHERE" >> construct<ElsewhereStmt>{}(maybe(name)))

// R1049 end-where-stmt -> ENDWHERE [where-construct-name]
TYPE_CONTEXT_PARSER("END WHERE statement"_en_US,
    "END WHERE" >> construct<EndWhereStmt>{}(maybe(name)))

// R1050 forall-construct ->
//         forall-construct-stmt [forall-body-construct]... end-forall-stmt
TYPE_CONTEXT_PARSER("FORALL construct"_en_US,
    construct<ForallConstruct>{}(statement(Parser<ForallConstructStmt>{}),
        many(Parser<ForallBodyConstruct>{}),
        statement(Parser<EndForallStmt>{})))

// R1051 forall-construct-stmt ->
//         [forall-construct-name :] FORALL concurrent-header
TYPE_CONTEXT_PARSER("FORALL construct statement"_en_US,
    construct<ForallConstructStmt>{}(
        maybe(name / ":"), "FORALL" >> indirect(concurrentHeader)))

// R1052 forall-body-construct ->
//         forall-assignment-stmt | where-stmt | where-construct |
//         forall-construct | forall-stmt
TYPE_PARSER(construct<ForallBodyConstruct>{}(statement(forallAssignmentStmt)) ||
    construct<ForallBodyConstruct>{}(statement(whereStmt)) ||
    construct<ForallBodyConstruct>{}(whereConstruct) ||
    construct<ForallBodyConstruct>{}(indirect(forallConstruct)) ||
    construct<ForallBodyConstruct>{}(statement(forallStmt)))

// R1053 forall-assignment-stmt -> assignment-stmt | pointer-assignment-stmt
TYPE_PARSER(construct<ForallAssignmentStmt>{}(assignmentStmt) ||
    construct<ForallAssignmentStmt>{}(pointerAssignmentStmt))

// R1054 end-forall-stmt -> END FORALL [forall-construct-name]
TYPE_CONTEXT_PARSER("END FORALL statement"_en_US,
    "END FORALL" >> construct<EndForallStmt>{}(maybe(name)))

// R1055 forall-stmt -> FORALL concurrent-header forall-assignment-stmt
TYPE_CONTEXT_PARSER("FORALL statement"_en_US,
    "FORALL" >> construct<ForallStmt>{}(
                    indirect(concurrentHeader), forallAssignmentStmt))

// R1101 block -> [execution-part-construct]...
constexpr auto block = many(executionPartConstruct);

// R1102 associate-construct -> associate-stmt block end-associate-stmt
TYPE_CONTEXT_PARSER("ASSOCIATE construct"_en_US,
    construct<AssociateConstruct>{}(statement(Parser<AssociateStmt>{}), block,
        statement(Parser<EndAssociateStmt>{})))

// R1103 associate-stmt ->
//        [associate-construct-name :] ASSOCIATE ( association-list )
TYPE_CONTEXT_PARSER("ASSOCIATE statement"_en_US,
    construct<AssociateStmt>{}(maybe(name / ":"),
        "ASSOCIATE" >> parenthesized(nonemptyList(Parser<Association>{}))))

// R1104 association -> associate-name => selector
TYPE_PARSER(construct<Association>{}(name, "=>" >> selector))

// R1105 selector -> expr | variable
TYPE_PARSER(construct<Selector>{}(variable) / lookAhead(","_tok || ")"_tok) ||
    construct<Selector>{}(expr))

// R1106 end-associate-stmt -> END ASSOCIATE [associate-construct-name]
TYPE_PARSER("END ASSOCIATE" >> construct<EndAssociateStmt>{}(maybe(name)))

// R1107 block-construct ->
//         block-stmt [block-specification-part] block end-block-stmt
TYPE_CONTEXT_PARSER("BLOCK construct"_en_US,
    construct<BlockConstruct>{}(statement(Parser<BlockStmt>{}),
        Parser<BlockSpecificationPart>{},  // can be empty
        block, statement(Parser<EndBlockStmt>{})))

// R1108 block-stmt -> [block-construct-name :] BLOCK
TYPE_PARSER(construct<BlockStmt>{}(maybe(name / ":") / "BLOCK"))

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
TYPE_PARSER(construct<BlockSpecificationPart>{}(specificationPart))

// R1110 end-block-stmt -> END BLOCK [block-construct-name]
TYPE_PARSER(construct<EndBlockStmt>{}("END BLOCK" >> maybe(name)))

// R1111 change-team-construct -> change-team-stmt block end-change-team-stmt
TYPE_CONTEXT_PARSER("CHANGE TEAM construct"_en_US,
    construct<ChangeTeamConstruct>{}(statement(Parser<ChangeTeamStmt>{}), block,
        statement(Parser<EndChangeTeamStmt>{})))

// R1112 change-team-stmt ->
//         [team-construct-name :] CHANGE TEAM
//         ( team-variable [, coarray-association-list] [, sync-stat-list] )
TYPE_CONTEXT_PARSER("CHANGE TEAM statement"_en_US,
    construct<ChangeTeamStmt>{}(maybe(name / ":"),
        "CHANGE TEAM"_sptok >> "("_tok >> teamVariable,
        defaulted("," >> nonemptyList(Parser<CoarrayAssociation>{})),
        defaulted("," >> nonemptyList(statOrErrmsg))) /
        ")")

// R1113 coarray-association -> codimension-decl => selector
TYPE_PARSER(construct<CoarrayAssociation>{}(
    Parser<CodimensionDecl>{}, "=>" >> selector))

// R1114 end-change-team-stmt ->
//         END TEAM [( [sync-stat-list] )] [team-construct-name]
TYPE_CONTEXT_PARSER("END TEAM statement"_en_US,
    "END TEAM" >>
        construct<EndChangeTeamStmt>{}(
            defaulted(parenthesized(optionalList(statOrErrmsg))), maybe(name)))

// R1117 critical-stmt ->
//         [critical-construct-name :] CRITICAL [( [sync-stat-list] )]
TYPE_CONTEXT_PARSER("CRITICAL statement"_en_US,
    construct<CriticalStmt>{}(maybe(name / ":"),
        "CRITICAL" >> defaulted(parenthesized(optionalList(statOrErrmsg)))))

// R1116 critical-construct -> critical-stmt block end-critical-stmt
TYPE_CONTEXT_PARSER("CRITICAL construct"_en_US,
    construct<CriticalConstruct>{}(statement(Parser<CriticalStmt>{}), block,
        statement(Parser<EndCriticalStmt>{})))

// R1118 end-critical-stmt -> END CRITICAL [critical-construct-name]
TYPE_PARSER("END CRITICAL" >> construct<EndCriticalStmt>{}(maybe(name)))

// R1119 do-construct -> do-stmt block end-do
// R1120 do-stmt -> nonlabel-do-stmt | label-do-stmt
constexpr struct EnterNonlabelDoConstruct {
  using resultType = Success;
  static std::optional<Success> Parse(ParseState *state) {
    if (auto ustate = state->userState()) {
      ustate->EnterNonlabelDoConstruct();
    }
    return {Success{}};
  }
} enterNonlabelDoConstruct;

constexpr struct LeaveDoConstruct {
  using resultType = Success;
  static std::optional<Success> Parse(ParseState *state) {
    if (auto ustate = state->userState()) {
      ustate->LeaveDoConstruct();
    }
    return {Success{}};
  }
} leaveDoConstruct;

TYPE_CONTEXT_PARSER("DO construct"_en_US,
    construct<DoConstruct>{}(
        statement(Parser<NonLabelDoStmt>{}) / enterNonlabelDoConstruct, block,
        statement(endDoStmt) / leaveDoConstruct))

// R1125 concurrent-header ->
//         ( [integer-type-spec ::] concurrent-control-list
//         [, scalar-mask-expr] )
TYPE_PARSER(parenthesized(construct<ConcurrentHeader>{}(
    maybe(integerTypeSpec / "::"), nonemptyList(Parser<ConcurrentControl>{}),
    maybe("," >> scalarLogicalExpr))))

// R1126 concurrent-control ->
//         index-name = concurrent-limit : concurrent-limit [: concurrent-step]
// R1127 concurrent-limit -> scalar-int-expr
// R1128 concurrent-step -> scalar-int-expr
TYPE_PARSER(construct<ConcurrentControl>{}(name / "=", scalarIntExpr / ":",
    scalarIntExpr, maybe(":" >> scalarIntExpr)))

// R1130 locality-spec ->
//         LOCAL ( variable-name-list ) | LOCAL INIT ( variable-name-list ) |
//         SHARED ( variable-name-list ) | DEFAULT ( NONE )
TYPE_PARSER(
    "LOCAL" >> construct<LocalitySpec>{}(construct<LocalitySpec::Local>{}(
                   parenthesized(nonemptyList(name)))) ||
    "LOCAL INIT"_sptok >>
        construct<LocalitySpec>{}(construct<LocalitySpec::LocalInit>{}(
            parenthesized(nonemptyList(name)))) ||
    "SHARED" >> construct<LocalitySpec>{}(construct<LocalitySpec::Shared>{}(
                    parenthesized(nonemptyList(name)))) ||
    "DEFAULT ( NONE )" >>
        construct<LocalitySpec>{}(construct<LocalitySpec::DefaultNone>{}))

// R1123 loop-control ->
//         [,] do-variable = scalar-int-expr , scalar-int-expr
//           [, scalar-int-expr] |
//         [,] WHILE ( scalar-logical-expr ) |
//         [,] CONCURRENT concurrent-header concurrent-locality
// R1129 concurrent-locality -> [locality-spec]...
TYPE_CONTEXT_PARSER("loop control"_en_US,
    maybe(","_tok) >>
        (construct<LoopControl>{}(loopBounds(scalarIntExpr)) ||
            "WHILE" >>
                construct<LoopControl>{}(parenthesized(scalarLogicalExpr)) ||
            "CONCURRENT" >>
                construct<LoopControl>{}(construct<LoopControl::Concurrent>{}(
                    concurrentHeader, many(Parser<LocalitySpec>{})))))

// R1121 label-do-stmt -> [do-construct-name :] DO label [loop-control]
TYPE_CONTEXT_PARSER("label DO statement"_en_US,
    construct<LabelDoStmt>{}(
        maybe(name / ":"), "DO" >> label, maybe(loopControl)))

// R1122 nonlabel-do-stmt -> [do-construct-name :] DO [loop-control]
TYPE_CONTEXT_PARSER("nonlabel DO statement"_en_US,
    construct<NonLabelDoStmt>{}(maybe(name / ":"), "DO" >> maybe(loopControl)))

// R1132 end-do-stmt -> END DO [do-construct-name]
TYPE_CONTEXT_PARSER(
    "END DO statement"_en_US, "END DO" >> construct<EndDoStmt>{}(maybe(name)))

// R1133 cycle-stmt -> CYCLE [do-construct-name]
TYPE_CONTEXT_PARSER(
    "CYCLE statement"_en_US, "CYCLE" >> construct<CycleStmt>{}(maybe(name)))

// R1134 if-construct ->
//         if-then-stmt block [else-if-stmt block]...
//         [else-stmt block] end-if-stmt
// R1135 if-then-stmt -> [if-construct-name :] IF ( scalar-logical-expr )
// THEN R1136 else-if-stmt ->
//         ELSE IF ( scalar-logical-expr ) THEN [if-construct-name]
// R1137 else-stmt -> ELSE [if-construct-name]
// R1138 end-if-stmt -> END IF [if-construct-name]
TYPE_CONTEXT_PARSER("IF construct"_en_US,
    construct<IfConstruct>{}(
        statement(construct<IfThenStmt>{}(maybe(name / ":"),
            "IF" >> parenthesized(scalarLogicalExpr) / "THEN")),
        block,
        many(construct<IfConstruct::ElseIfBlock>{}(
            statement(construct<ElseIfStmt>{}(
                "ELSE IF" >> parenthesized(scalarLogicalExpr),
                "THEN" >> maybe(name))),
            block)),
        maybe(construct<IfConstruct::ElseBlock>{}(
            statement(construct<ElseStmt>{}("ELSE" >> maybe(name))), block)),
        statement(construct<EndIfStmt>{}("END IF" >> maybe(name)))))

// R1139 if-stmt -> IF ( scalar-logical-expr ) action-stmt
TYPE_CONTEXT_PARSER("IF statement"_en_US,
    "IF" >> construct<IfStmt>{}(parenthesized(scalarLogicalExpr), actionStmt))

// R1140 case-construct ->
//         select-case-stmt [case-stmt block]... end-select-stmt
TYPE_CONTEXT_PARSER("SELECT CASE construct"_en_US,
    construct<CaseConstruct>{}(statement(Parser<SelectCaseStmt>{}),
        many(construct<CaseConstruct::Case>{}(
            statement(Parser<CaseStmt>{}), block)),
        statement(endSelectStmt)))

// R1141 select-case-stmt -> [case-construct-name :] SELECT CASE ( case-expr
// ) R1144 case-expr -> scalar-expr
TYPE_CONTEXT_PARSER("SELECT CASE statement"_en_US,
    construct<SelectCaseStmt>{}(
        maybe(name / ":"), "SELECT CASE" >> parenthesized(scalar(expr))))

// R1142 case-stmt -> CASE case-selector [case-construct-name]
TYPE_CONTEXT_PARSER("CASE statement"_en_US,
    "CASE" >> construct<CaseStmt>{}(Parser<CaseSelector>{}, maybe(name)))

// R1143 end-select-stmt -> END SELECT [case-construct-name]
// R1151 end-select-rank-stmt -> END SELECT [select-construct-name]
// R1155 end-select-type-stmt -> END SELECT [select-construct-name]
TYPE_PARSER("END SELECT" >> construct<EndSelectStmt>{}(maybe(name)))

// R1145 case-selector -> ( case-value-range-list ) | DEFAULT
constexpr auto defaultKeyword = "DEFAULT" >> construct<Default>{};
TYPE_PARSER(parenthesized(construct<CaseSelector>{}(
                nonemptyList(Parser<CaseValueRange>{}))) ||
    construct<CaseSelector>{}(defaultKeyword))

// R1147 case-value -> scalar-constant-expr
constexpr auto caseValue = scalar(constantExpr);

// R1146 case-value-range ->
//         case-value | case-value : | : case-value | case-value : case-value
TYPE_PARSER(construct<CaseValueRange>{}(construct<CaseValueRange::Range>{}(
                construct<std::optional<CaseValue>>{}(caseValue),
                ":" >> maybe(caseValue))) ||
    construct<CaseValueRange>{}(construct<CaseValueRange::Range>{}(
        construct<std::optional<CaseValue>>{},
        ":" >> construct<std::optional<CaseValue>>{}(caseValue))) ||
    construct<CaseValueRange>{}(caseValue))

// R1148 select-rank-construct ->
//         select-rank-stmt [select-rank-case-stmt block]...
//         end-select-rank-stmt
TYPE_CONTEXT_PARSER("SELECT RANK construct"_en_US,
    construct<SelectRankConstruct>{}(statement(Parser<SelectRankStmt>{}),
        many(construct<SelectRankConstruct::RankCase>{}(
            statement(Parser<SelectRankCaseStmt>{}), block)),
        statement(endSelectStmt)))

// R1149 select-rank-stmt ->
//         [select-construct-name :] SELECT RANK
//         ( [associate-name =>] selector )
TYPE_CONTEXT_PARSER("SELECT RANK statement"_en_US,
    construct<SelectRankStmt>{}(maybe(name / ":"),
        "SELECT RANK"_sptok >> "("_tok >> maybe(name / "=>"), selector / ")"))

// R1150 select-rank-case-stmt ->
//         RANK ( scalar-int-constant-expr ) [select-construct-name] |
//         RANK ( * ) [select-construct-name] |
//         RANK DEFAULT [select-construct-name]
TYPE_CONTEXT_PARSER("RANK case statement"_en_US,
    "RANK" >> (construct<SelectRankCaseStmt>{}(
                  parenthesized(construct<SelectRankCaseStmt::Rank>{}(
                                    scalarIntConstantExpr) ||
                      construct<SelectRankCaseStmt::Rank>{}(star)) ||
                      construct<SelectRankCaseStmt::Rank>{}(defaultKeyword),
                  maybe(name))))

// R1152 select-type-construct ->
//         select-type-stmt [type-guard-stmt block]... end-select-type-stmt
TYPE_CONTEXT_PARSER("SELECT TYPE construct"_en_US,
    construct<SelectTypeConstruct>{}(statement(Parser<SelectTypeStmt>{}),
        many(construct<SelectTypeConstruct::TypeCase>{}(
            statement(Parser<TypeGuardStmt>{}), block)),
        statement(endSelectStmt)))

// R1153 select-type-stmt ->
//         [select-construct-name :] SELECT TYPE
//         ( [associate-name =>] selector )
TYPE_CONTEXT_PARSER("SELECT TYPE statement"_en_US,
    construct<SelectTypeStmt>{}(maybe(name / ":"),
        "SELECT TYPE (" >> maybe(name / "=>"), selector / ")"))

// R1154 type-guard-stmt ->
//         TYPE IS ( type-spec ) [select-construct-name] |
//         CLASS IS ( derived-type-spec ) [select-construct-name] |
//         CLASS DEFAULT [select-construct-name]
TYPE_CONTEXT_PARSER("type guard statement"_en_US,
    construct<TypeGuardStmt>{}("TYPE IS"_sptok >>
                parenthesized(construct<TypeGuardStmt::Guard>{}(typeSpec)) ||
            "CLASS IS"_sptok >> parenthesized(construct<TypeGuardStmt::Guard>{}(
                                    derivedTypeSpec)) ||
            "CLASS" >> construct<TypeGuardStmt::Guard>{}(defaultKeyword),
        maybe(name)))

// R1156 exit-stmt -> EXIT [construct-name]
TYPE_CONTEXT_PARSER(
    "EXIT statement"_en_US, "EXIT" >> construct<ExitStmt>{}(maybe(name)))

// R1157 goto-stmt -> GO TO label
TYPE_CONTEXT_PARSER(
    "GOTO statement"_en_US, "GO TO" >> construct<GotoStmt>{}(label))

// R1158 computed-goto-stmt -> GO TO ( label-list ) [,] scalar-int-expr
TYPE_CONTEXT_PARSER("computed GOTO statement"_en_US,
    "GO TO" >> construct<ComputedGotoStmt>{}(parenthesized(nonemptyList(label)),
                   maybe(","_tok) >> scalarIntExpr))

// R1160 stop-stmt -> STOP [stop-code] [, QUIET = scalar-logical-expr]
// R1161 error-stop-stmt ->
//         ERROR STOP [stop-code] [, QUIET = scalar-logical-expr]
TYPE_CONTEXT_PARSER("STOP statement"_en_US,
    construct<StopStmt>{}("STOP" >> pure(StopStmt::Kind::Stop) ||
            "ERROR STOP"_sptok >> pure(StopStmt::Kind::ErrorStop),
        maybe(Parser<StopCode>{}), maybe(", QUIET =" >> scalarLogicalExpr)))

// R1162 stop-code -> scalar-default-char-expr | scalar-int-expr
TYPE_PARSER(construct<StopCode>{}(scalarDefaultCharExpr) ||
    construct<StopCode>{}(scalarIntExpr))

// R1164 sync-all-stmt -> SYNC ALL [( [sync-stat-list] )]
TYPE_CONTEXT_PARSER("SYNC ALL statement"_en_US,
    "SYNC ALL"_sptok >> construct<SyncAllStmt>{}(defaulted(
                            parenthesized(optionalList(statOrErrmsg)))))

// R1166 sync-images-stmt -> SYNC IMAGES ( image-set [, sync-stat-list] )
// R1167 image-set -> int-expr | *
TYPE_CONTEXT_PARSER("SYNC IMAGES statement"_en_US,
    "SYNC IMAGES"_sptok >> parenthesized(construct<SyncImagesStmt>{}(
                               construct<SyncImagesStmt::ImageSet>{}(intExpr) ||
                                   construct<SyncImagesStmt::ImageSet>{}(star),
                               defaulted("," >> nonemptyList(statOrErrmsg)))))

// R1168 sync-memory-stmt -> SYNC MEMORY [( [sync-stat-list] )]
TYPE_CONTEXT_PARSER("SYNC MEMORY statement"_en_US,
    "SYNC MEMORY"_sptok >> construct<SyncMemoryStmt>{}(defaulted(
                               parenthesized(optionalList(statOrErrmsg)))))

// R1169 sync-team-stmt -> SYNC TEAM ( team-variable [, sync-stat-list] )
TYPE_CONTEXT_PARSER("SYNC TEAM statement"_en_US,
    "SYNC TEAM"_sptok >> parenthesized(construct<SyncTeamStmt>{}(teamVariable,
                             defaulted("," >> nonemptyList(statOrErrmsg)))))

// R1170 event-post-stmt -> EVENT POST ( event-variable [, sync-stat-list] )
// R1171 event-variable -> scalar-variable
TYPE_CONTEXT_PARSER("EVENT POST statement"_en_US,
    "EVENT POST"_sptok >>
        parenthesized(construct<EventPostStmt>{}(
            scalar(variable), defaulted("," >> nonemptyList(statOrErrmsg)))))

// R1172 event-wait-stmt ->
//         EVENT WAIT ( event-variable [, event-wait-spec-list] )
TYPE_CONTEXT_PARSER("EVENT WAIT statement"_en_US,
    "EVENT WAIT"_sptok >>
        parenthesized(construct<EventWaitStmt>{}(scalar(variable),
            defaulted(
                "," >> nonemptyList(Parser<EventWaitStmt::EventWaitSpec>{})))))

// R1174 until-spec -> UNTIL_COUNT = scalar-int-expr
constexpr auto untilSpec = "UNTIL_COUNT =" >> scalarIntExpr;

// R1173 event-wait-spec -> until-spec | sync-stat
TYPE_PARSER(construct<EventWaitStmt::EventWaitSpec>{}(untilSpec) ||
    construct<EventWaitStmt::EventWaitSpec>{}(statOrErrmsg))

// R1175 form-team-stmt ->
//         FORM TEAM ( team-number , team-variable [, form-team-spec-list] )
// R1176 team-number -> scalar-int-expr
TYPE_CONTEXT_PARSER("FORM TEAM statement"_en_US,
    "FORM TEAM"_sptok >>
        parenthesized(construct<FormTeamStmt>{}(scalarIntExpr,
            "," >> teamVariable,
            defaulted(
                "," >> nonemptyList(Parser<FormTeamStmt::FormTeamSpec>{})))))

// R1177 form-team-spec -> NEW_INDEX = scalar-int-expr | sync-stat
TYPE_PARSER(
    "NEW_INDEX =" >> construct<FormTeamStmt::FormTeamSpec>{}(scalarIntExpr) ||
    construct<FormTeamStmt::FormTeamSpec>{}(statOrErrmsg))

// R1181 lock-variable -> scalar-variable
constexpr auto lockVariable = scalar(variable);

// R1178 lock-stmt -> LOCK ( lock-variable [, lock-stat-list] )
TYPE_CONTEXT_PARSER("LOCK statement"_en_US,
    "LOCK" >>
        parenthesized(construct<LockStmt>{}(lockVariable,
            defaulted("," >> nonemptyList(Parser<LockStmt::LockStat>{})))))

// R1179 lock-stat -> ACQUIRED_LOCK = scalar-logical-variable | sync-stat
TYPE_PARSER("ACQUIRED_LOCK =" >>
        construct<LockStmt::LockStat>{}(scalarLogicalVariable) ||
    construct<LockStmt::LockStat>{}(statOrErrmsg))

// R1180 unlock-stmt -> UNLOCK ( lock-variable [, sync-stat-list] )
TYPE_CONTEXT_PARSER("UNLOCK statement"_en_US,
    "UNLOCK" >> parenthesized(construct<UnlockStmt>{}(lockVariable,
                    defaulted("," >> nonemptyList(statOrErrmsg)))))

// R1201 io-unit -> file-unit-number | * | internal-file-variable
// R1203 internal-file-variable -> char-variable
TYPE_PARSER(construct<IoUnit>{}(fileUnitNumber) || construct<IoUnit>{}(star) ||
    construct<IoUnit>{}(charVariable / !"="_tok))

// R1202 file-unit-number -> scalar-int-expr
TYPE_PARSER(construct<FileUnitNumber>{}(scalarIntExpr / !"="_tok))

// R1204 open-stmt -> OPEN ( connect-spec-list )
TYPE_CONTEXT_PARSER("OPEN statement"_en_US,
    "OPEN" >> parenthesized(
                  construct<OpenStmt>{}(nonemptyList(Parser<ConnectSpec>{}))))

// R1206 file-name-expr -> scalar-default-char-expr
constexpr auto fileNameExpr = scalarDefaultCharExpr;

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
constexpr auto statusExpr = construct<StatusExpr>{}(scalarDefaultCharExpr);
constexpr auto errLabel = construct<ErrLabel>{}(label);

TYPE_PARSER(maybe("UNIT ="_tok) >> construct<ConnectSpec>{}(fileUnitNumber) ||
    "ACCESS =" >> construct<ConnectSpec>{}(construct<ConnectSpec::CharExpr>{}(
                      pure(ConnectSpec::CharExpr::Kind::Access),
                      scalarDefaultCharExpr)) ||
    "ACTION =" >> construct<ConnectSpec>{}(construct<ConnectSpec::CharExpr>{}(
                      pure(ConnectSpec::CharExpr::Kind::Action),
                      scalarDefaultCharExpr)) ||
    "ASYNCHRONOUS =" >>
        construct<ConnectSpec>{}(construct<ConnectSpec::CharExpr>{}(
            pure(ConnectSpec::CharExpr::Kind::Asynchronous),
            scalarDefaultCharExpr)) ||
    "BLANK =" >>
        construct<ConnectSpec>{}(construct<ConnectSpec::CharExpr>{}(
            pure(ConnectSpec::CharExpr::Kind::Blank), scalarDefaultCharExpr)) ||
    "DECIMAL =" >> construct<ConnectSpec>{}(construct<ConnectSpec::CharExpr>{}(
                       pure(ConnectSpec::CharExpr::Kind::Decimal),
                       scalarDefaultCharExpr)) ||
    "DELIM =" >>
        construct<ConnectSpec>{}(construct<ConnectSpec::CharExpr>{}(
            pure(ConnectSpec::CharExpr::Kind::Delim), scalarDefaultCharExpr)) ||
    "ENCODING =" >> construct<ConnectSpec>{}(construct<ConnectSpec::CharExpr>{}(
                        pure(ConnectSpec::CharExpr::Kind::Encoding),
                        scalarDefaultCharExpr)) ||
    "ERR =" >> construct<ConnectSpec>{}(errLabel) ||
    "FILE =" >> construct<ConnectSpec>{}(fileNameExpr) ||
    extension("NAME =" >> construct<ConnectSpec>{}(fileNameExpr)) ||
    "FORM =" >>
        construct<ConnectSpec>{}(construct<ConnectSpec::CharExpr>{}(
            pure(ConnectSpec::CharExpr::Kind::Form), scalarDefaultCharExpr)) ||
    "IOMSG =" >> construct<ConnectSpec>{}(msgVariable) ||
    "IOSTAT =" >> construct<ConnectSpec>{}(statVariable) ||
    "NEWUNIT =" >> construct<ConnectSpec>{}(construct<ConnectSpec::Newunit>{}(
                       scalar(integer(variable)))) ||
    "PAD =" >>
        construct<ConnectSpec>{}(construct<ConnectSpec::CharExpr>{}(
            pure(ConnectSpec::CharExpr::Kind::Pad), scalarDefaultCharExpr)) ||
    "POSITION =" >> construct<ConnectSpec>{}(construct<ConnectSpec::CharExpr>{}(
                        pure(ConnectSpec::CharExpr::Kind::Position),
                        scalarDefaultCharExpr)) ||
    "RECL =" >> construct<ConnectSpec>{}(
                    construct<ConnectSpec::Recl>{}(scalarIntExpr)) ||
    "ROUND =" >>
        construct<ConnectSpec>{}(construct<ConnectSpec::CharExpr>{}(
            pure(ConnectSpec::CharExpr::Kind::Round), scalarDefaultCharExpr)) ||
    "SIGN =" >>
        construct<ConnectSpec>{}(construct<ConnectSpec::CharExpr>{}(
            pure(ConnectSpec::CharExpr::Kind::Sign), scalarDefaultCharExpr)) ||
    "STATUS =" >> construct<ConnectSpec>{}(statusExpr) ||
    extension(construct<ConnectSpec>{}(construct<ConnectSpec::CharExpr>{}(
                  "CONVERT =" >> pure(ConnectSpec::CharExpr::Kind::Convert),
                  scalarDefaultCharExpr)) ||
        construct<ConnectSpec>{}(construct<ConnectSpec::CharExpr>{}(
            "DISPOSE =" >> pure(ConnectSpec::CharExpr::Kind::Dispose),
            scalarDefaultCharExpr))))

// R1209 close-spec ->
//         [UNIT =] file-unit-number | IOSTAT = scalar-int-variable |
//         IOMSG = iomsg-variable | ERR = label |
//         STATUS = scalar-default-char-expr
constexpr auto closeSpec = maybe("UNIT ="_tok) >>
        construct<CloseStmt::CloseSpec>{}(fileUnitNumber) ||
    "IOSTAT =" >> construct<CloseStmt::CloseSpec>{}(statVariable) ||
    "IOMSG =" >> construct<CloseStmt::CloseSpec>{}(msgVariable) ||
    "ERR =" >> construct<CloseStmt::CloseSpec>{}(errLabel) ||
    "STATUS =" >> construct<CloseStmt::CloseSpec>{}(statusExpr);

// R1208 close-stmt -> CLOSE ( close-spec-list )
TYPE_CONTEXT_PARSER("CLOSE statement"_en_US,
    "CLOSE" >> construct<CloseStmt>{}(parenthesized(nonemptyList(closeSpec))))

// R1210 read-stmt ->
//         READ ( io-control-spec-list ) [input-item-list] |
//         READ format [, input-item-list]
constexpr auto inputItemList =
    extension(some("," >> inputItem)) ||  // legacy extension: leading comma
    optionalList(inputItem);

TYPE_CONTEXT_PARSER("READ statement"_en_US,
    "READ" >>
        ("(" >> construct<ReadStmt>{}(construct<std::optional<IoUnit>>{}(
                                          maybe("UNIT ="_tok) >> ioUnit),
                    "," >> construct<std::optional<Format>>{}(format),
                    defaulted("," >> nonemptyList(ioControlSpec)) / ")",
                    inputItemList) ||
            "(" >> construct<ReadStmt>{}(
                       construct<std::optional<IoUnit>>{}(ioUnit),
                       construct<std::optional<Format>>{},
                       defaulted("," >> nonemptyList(ioControlSpec)) / ")",
                       inputItemList) ||
            construct<ReadStmt>{}(construct<std::optional<IoUnit>>{},
                construct<std::optional<Format>>{},
                parenthesized(nonemptyList(ioControlSpec)), inputItemList) ||
            construct<ReadStmt>{}(construct<std::optional<IoUnit>>{},
                construct<std::optional<Format>>{}(format),
                construct<std::list<IoControlSpec>>{}, many("," >> inputItem))))

// R1214 id-variable -> scalar-int-variable
constexpr auto idVariable = construct<IdVariable>{}(scalarIntVariable);

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
constexpr auto endLabel = construct<EndLabel>{}(label);
constexpr auto eorLabel = construct<EorLabel>{}(label);
TYPE_PARSER("UNIT =" >> construct<IoControlSpec>{}(ioUnit) ||
    "FMT =" >> construct<IoControlSpec>{}(format) ||
    "NML =" >> construct<IoControlSpec>{}(name) ||
    "ADVANCE =" >>
        construct<IoControlSpec>{}(construct<IoControlSpec::CharExpr>{}(
            pure(IoControlSpec::CharExpr::Kind::Advance),
            scalarDefaultCharExpr)) ||
    "ASYNCHRONOUS =" >>
        construct<IoControlSpec>{}(construct<IoControlSpec::Asynchronous>{}(
            scalarDefaultCharConstantExpr)) ||
    "BLANK =" >>
        construct<IoControlSpec>{}(construct<IoControlSpec::CharExpr>{}(
            pure(IoControlSpec::CharExpr::Kind::Blank),
            scalarDefaultCharExpr)) ||
    "DECIMAL =" >>
        construct<IoControlSpec>{}(construct<IoControlSpec::CharExpr>{}(
            pure(IoControlSpec::CharExpr::Kind::Decimal),
            scalarDefaultCharExpr)) ||
    "DELIM =" >>
        construct<IoControlSpec>{}(construct<IoControlSpec::CharExpr>{}(
            pure(IoControlSpec::CharExpr::Kind::Delim),
            scalarDefaultCharExpr)) ||
    "END =" >> construct<IoControlSpec>{}(endLabel) ||
    "EOR =" >> construct<IoControlSpec>{}(eorLabel) ||
    "ERR =" >> construct<IoControlSpec>{}(errLabel) ||
    "ID =" >> construct<IoControlSpec>{}(idVariable) ||
    "IOMSG = " >> construct<IoControlSpec>{}(msgVariable) ||
    "IOSTAT = " >> construct<IoControlSpec>{}(statVariable) ||
    "PAD =" >>
        construct<IoControlSpec>{}(construct<IoControlSpec::CharExpr>{}(
            pure(IoControlSpec::CharExpr::Kind::Pad), scalarDefaultCharExpr)) ||
    "POS =" >> construct<IoControlSpec>{}(
                   construct<IoControlSpec::Pos>{}(scalarIntExpr)) ||
    "REC =" >> construct<IoControlSpec>{}(
                   construct<IoControlSpec::Rec>{}(scalarIntExpr)) ||
    "ROUND =" >>
        construct<IoControlSpec>{}(construct<IoControlSpec::CharExpr>{}(
            pure(IoControlSpec::CharExpr::Kind::Round),
            scalarDefaultCharExpr)) ||
    "SIGN =" >> construct<IoControlSpec>{}(construct<IoControlSpec::CharExpr>{}(
                    pure(IoControlSpec::CharExpr::Kind::Sign),
                    scalarDefaultCharExpr)) ||
    "SIZE =" >> construct<IoControlSpec>{}(
                    construct<IoControlSpec::Size>{}(scalarIntVariable)))

// R1211 write-stmt -> WRITE ( io-control-spec-list ) [output-item-list]
constexpr auto outputItemList =
    extension(some("," >> outputItem)) ||  // legacy: allow leading comma
    optionalList(outputItem);

TYPE_CONTEXT_PARSER("WRITE statement"_en_US,
    "WRITE" >>
        (construct<WriteStmt>{}("(" >> construct<std::optional<IoUnit>>{}(
                                           maybe("UNIT ="_tok) >> ioUnit),
             "," >> construct<std::optional<Format>>{}(format),
             defaulted("," >> nonemptyList(ioControlSpec)) / ")",
             outputItemList) ||
            construct<WriteStmt>{}(
                "(" >> construct<std::optional<IoUnit>>{}(ioUnit),
                construct<std::optional<Format>>{},
                defaulted("," >> nonemptyList(ioControlSpec)) / ")",
                outputItemList) ||
            construct<WriteStmt>{}(construct<std::optional<IoUnit>>{},
                construct<std::optional<Format>>{},
                parenthesized(nonemptyList(ioControlSpec)), outputItemList)))

// R1212 print-stmt PRINT format [, output-item-list]
TYPE_CONTEXT_PARSER("PRINT statement"_en_US,
    "PRINT" >> construct<PrintStmt>{}(
                   format, defaulted("," >> nonemptyList(outputItem))))

// R1215 format -> default-char-expr | label | *
TYPE_PARSER(construct<Format>{}(defaultCharExpr / !"="_tok) ||
    construct<Format>{}(label) || construct<Format>{}(star))

// R1216 input-item -> variable | io-implied-do
TYPE_PARSER(construct<InputItem>{}(variable) ||
    construct<InputItem>{}(indirect(inputImpliedDo)))

// R1217 output-item -> expr | io-implied-do
TYPE_PARSER(construct<OutputItem>{}(expr) ||
    construct<OutputItem>{}(indirect(outputImpliedDo)))

// R1220 io-implied-do-control ->
//         do-variable = scalar-int-expr , scalar-int-expr [, scalar-int-expr]
constexpr auto ioImpliedDoControl = loopBounds(scalarIntExpr);

// R1218 io-implied-do -> ( io-implied-do-object-list , io-implied-do-control )
// R1219 io-implied-do-object -> input-item | output-item
TYPE_CONTEXT_PARSER("input implied DO"_en_US,
    parenthesized(construct<InputImpliedDo>{}(
        nonemptyList(inputItem / lookAhead(","_tok)),
        "," >> ioImpliedDoControl)))
TYPE_CONTEXT_PARSER("output implied DO"_en_US,
    parenthesized(construct<OutputImpliedDo>{}(
        nonemptyList(outputItem / lookAhead(","_tok)),
        "," >> ioImpliedDoControl)))

// R1222 wait-stmt -> WAIT ( wait-spec-list )
TYPE_CONTEXT_PARSER("WAIT statement"_en_US,
    "WAIT" >>
        parenthesized(construct<WaitStmt>{}(nonemptyList(Parser<WaitSpec>{}))))

// R1223 wait-spec ->
//         [UNIT =] file-unit-number | END = label | EOR = label | ERR = label |
//         ID = scalar-int-expr | IOMSG = iomsg-variable |
//         IOSTAT = scalar-int-variable
constexpr auto idExpr = construct<IdExpr>{}(scalarIntExpr);

TYPE_PARSER(maybe("UNIT ="_tok) >> construct<WaitSpec>{}(fileUnitNumber) ||
    "END =" >> construct<WaitSpec>{}(endLabel) ||
    "EOR =" >> construct<WaitSpec>{}(eorLabel) ||
    "ERR =" >> construct<WaitSpec>{}(errLabel) ||
    "ID =" >> construct<WaitSpec>{}(idExpr) ||
    "IOMSG =" >> construct<WaitSpec>{}(msgVariable) ||
    "IOSTAT =" >> construct<WaitSpec>{}(statVariable))

template<typename A> std::list<A> singletonList(A &&x) {
  std::list<A> result;
  result.push_front(std::move(x));
  return result;
}
constexpr auto bareUnitNumberAsList =
    applyFunction(singletonList<PositionOrFlushSpec>,
        construct<PositionOrFlushSpec>{}(fileUnitNumber));
constexpr auto positionOrFlushSpecList =
    parenthesized(nonemptyList(positionOrFlushSpec)) || bareUnitNumberAsList;

// R1224 backspace-stmt ->
//         BACKSPACE file-unit-number | BACKSPACE ( position-spec-list )
TYPE_CONTEXT_PARSER("BACKSPACE statement"_en_US,
    construct<BackspaceStmt>{}("BACKSPACE" >> positionOrFlushSpecList))

// R1225 endfile-stmt ->
//         ENDFILE file-unit-number | ENDFILE ( position-spec-list )
TYPE_CONTEXT_PARSER("ENDFILE statement"_en_US,
    construct<EndfileStmt>{}("ENDFILE" >> positionOrFlushSpecList))

// R1226 rewind-stmt -> REWIND file-unit-number | REWIND ( position-spec-list )
TYPE_CONTEXT_PARSER("REWIND statement"_en_US,
    construct<RewindStmt>{}("REWIND" >> positionOrFlushSpecList))

// R1227 position-spec ->
//         [UNIT =] file-unit-number | IOMSG = iomsg-variable |
//         IOSTAT = scalar-int-variable | ERR = label
// R1229 flush-spec ->
//         [UNIT =] file-unit-number | IOSTAT = scalar-int-variable |
//         IOMSG = iomsg-variable | ERR = label
TYPE_PARSER(
    maybe("UNIT ="_tok) >> construct<PositionOrFlushSpec>{}(fileUnitNumber) ||
    "IOMSG =" >> construct<PositionOrFlushSpec>{}(msgVariable) ||
    "IOSTAT =" >> construct<PositionOrFlushSpec>{}(statVariable) ||
    "ERR =" >> construct<PositionOrFlushSpec>{}(errLabel))

// R1228 flush-stmt -> FLUSH file-unit-number | FLUSH ( flush-spec-list )
TYPE_CONTEXT_PARSER("FLUSH statement"_en_US,
    construct<FlushStmt>{}("FLUSH" >> positionOrFlushSpecList))

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
TYPE_PARSER(maybe("UNIT ="_tok) >> construct<InquireSpec>{}(fileUnitNumber) ||
    "FILE =" >> construct<InquireSpec>{}(fileNameExpr) ||
    "ACCESS =" >> construct<InquireSpec>{}(construct<InquireSpec::CharVar>{}(
                      pure(InquireSpec::CharVar::Kind::Access),
                      scalarDefaultCharVariable)) ||
    "ACTION =" >> construct<InquireSpec>{}(construct<InquireSpec::CharVar>{}(
                      pure(InquireSpec::CharVar::Kind::Action),
                      scalarDefaultCharVariable)) ||
    "ASYNCHRONOUS =" >>
        construct<InquireSpec>{}(construct<InquireSpec::CharVar>{}(
            pure(InquireSpec::CharVar::Kind::Asynchronous),
            scalarDefaultCharVariable)) ||
    "BLANK =" >> construct<InquireSpec>{}(construct<InquireSpec::CharVar>{}(
                     pure(InquireSpec::CharVar::Kind::Blank),
                     scalarDefaultCharVariable)) ||
    "DECIMAL =" >> construct<InquireSpec>{}(construct<InquireSpec::CharVar>{}(
                       pure(InquireSpec::CharVar::Kind::Decimal),
                       scalarDefaultCharVariable)) ||
    "DELIM =" >> construct<InquireSpec>{}(construct<InquireSpec::CharVar>{}(
                     pure(InquireSpec::CharVar::Kind::Delim),
                     scalarDefaultCharVariable)) ||
    "DIRECT =" >> construct<InquireSpec>{}(construct<InquireSpec::CharVar>{}(
                      pure(InquireSpec::CharVar::Kind::Direct),
                      scalarDefaultCharVariable)) ||
    "ENCODING =" >> construct<InquireSpec>{}(construct<InquireSpec::CharVar>{}(
                        pure(InquireSpec::CharVar::Kind::Encoding),
                        scalarDefaultCharVariable)) ||
    "ERR =" >> construct<InquireSpec>{}(errLabel) ||
    "EXIST =" >>
        construct<InquireSpec>{}(construct<InquireSpec::LogVar>{}(
            pure(InquireSpec::LogVar::Kind::Exist), scalarLogicalVariable)) ||
    "FORM =" >> construct<InquireSpec>{}(construct<InquireSpec::CharVar>{}(
                    pure(InquireSpec::CharVar::Kind::Form),
                    scalarDefaultCharVariable)) ||
    "FORMATTED =" >> construct<InquireSpec>{}(construct<InquireSpec::CharVar>{}(
                         pure(InquireSpec::CharVar::Kind::Formatted),
                         scalarDefaultCharVariable)) ||
    "ID =" >> construct<InquireSpec>{}(idExpr) ||
    "IOMSG =" >> construct<InquireSpec>{}(construct<InquireSpec::CharVar>{}(
                     pure(InquireSpec::CharVar::Kind::Iomsg),
                     scalarDefaultCharVariable)) ||
    "IOSTAT =" >> construct<InquireSpec>{}(construct<InquireSpec::IntVar>{}(
                      pure(InquireSpec::IntVar::Kind::Iostat),
                      scalar(integer(variable)))) ||
    "NAME =" >> construct<InquireSpec>{}(construct<InquireSpec::CharVar>{}(
                    pure(InquireSpec::CharVar::Kind::Name),
                    scalarDefaultCharVariable)) ||
    "NAMED =" >>
        construct<InquireSpec>{}(construct<InquireSpec::LogVar>{}(
            pure(InquireSpec::LogVar::Kind::Named), scalarLogicalVariable)) ||
    "NEXTREC =" >> construct<InquireSpec>{}(construct<InquireSpec::IntVar>{}(
                       pure(InquireSpec::IntVar::Kind::Nextrec),
                       scalar(integer(variable)))) ||
    "NUMBER =" >> construct<InquireSpec>{}(construct<InquireSpec::IntVar>{}(
                      pure(InquireSpec::IntVar::Kind::Number),
                      scalar(integer(variable)))) ||
    "OPENED =" >>
        construct<InquireSpec>{}(construct<InquireSpec::LogVar>{}(
            pure(InquireSpec::LogVar::Kind::Opened), scalarLogicalVariable)) ||
    "PAD =" >> construct<InquireSpec>{}(construct<InquireSpec::CharVar>{}(
                   pure(InquireSpec::CharVar::Kind::Pad),
                   scalarDefaultCharVariable)) ||
    "PENDING =" >>
        construct<InquireSpec>{}(construct<InquireSpec::LogVar>{}(
            pure(InquireSpec::LogVar::Kind::Pending), scalarLogicalVariable)) ||
    "POS =" >>
        construct<InquireSpec>{}(construct<InquireSpec::IntVar>{}(
            pure(InquireSpec::IntVar::Kind::Pos), scalar(integer(variable)))) ||
    "POSITION =" >> construct<InquireSpec>{}(construct<InquireSpec::CharVar>{}(
                        pure(InquireSpec::CharVar::Kind::Position),
                        scalarDefaultCharVariable)) ||
    "READ =" >> construct<InquireSpec>{}(construct<InquireSpec::CharVar>{}(
                    pure(InquireSpec::CharVar::Kind::Read),
                    scalarDefaultCharVariable)) ||
    "READWRITE =" >> construct<InquireSpec>{}(construct<InquireSpec::CharVar>{}(
                         pure(InquireSpec::CharVar::Kind::Readwrite),
                         scalarDefaultCharVariable)) ||
    "RECL =" >> construct<InquireSpec>{}(construct<InquireSpec::IntVar>{}(
                    pure(InquireSpec::IntVar::Kind::Recl),
                    scalar(integer(variable)))) ||
    "ROUND =" >> construct<InquireSpec>{}(construct<InquireSpec::CharVar>{}(
                     pure(InquireSpec::CharVar::Kind::Round),
                     scalarDefaultCharVariable)) ||
    "SEQUENTIAL =" >>
        construct<InquireSpec>{}(construct<InquireSpec::CharVar>{}(
            pure(InquireSpec::CharVar::Kind::Sequential),
            scalarDefaultCharVariable)) ||
    "SIGN =" >> construct<InquireSpec>{}(construct<InquireSpec::CharVar>{}(
                    pure(InquireSpec::CharVar::Kind::Sign),
                    scalarDefaultCharVariable)) ||
    "SIZE =" >> construct<InquireSpec>{}(construct<InquireSpec::IntVar>{}(
                    pure(InquireSpec::IntVar::Kind::Size),
                    scalar(integer(variable)))) ||
    "STREAM =" >> construct<InquireSpec>{}(construct<InquireSpec::CharVar>{}(
                      pure(InquireSpec::CharVar::Kind::Stream),
                      scalarDefaultCharVariable)) ||
    "STATUS =" >> construct<InquireSpec>{}(construct<InquireSpec::CharVar>{}(
                      pure(InquireSpec::CharVar::Kind::Status),
                      scalarDefaultCharVariable)) ||
    "UNFORMATTED =" >>
        construct<InquireSpec>{}(construct<InquireSpec::CharVar>{}(
            pure(InquireSpec::CharVar::Kind::Unformatted),
            scalarDefaultCharVariable)) ||
    "WRITE =" >> construct<InquireSpec>{}(construct<InquireSpec::CharVar>{}(
                     pure(InquireSpec::CharVar::Kind::Write),
                     scalarDefaultCharVariable)))

// R1230 inquire-stmt ->
//         INQUIRE ( inquire-spec-list ) |
//         INQUIRE ( IOLENGTH = scalar-int-variable ) output-item-list
TYPE_CONTEXT_PARSER("INQUIRE statement"_en_US,
    "INQUIRE" >>
        (construct<InquireStmt>{}(
             parenthesized(nonemptyList(Parser<InquireSpec>{}))) ||
            construct<InquireStmt>{}(construct<InquireStmt::Iolength>{}(
                parenthesized("IOLENGTH =" >> scalar(integer(variable))),
                nonemptyList(outputItem)))))

// R1301 format-stmt -> FORMAT format-specification
TYPE_CONTEXT_PARSER("FORMAT statement"_en_US,
    "FORMAT" >> construct<FormatStmt>{}(Parser<format::FormatSpecification>{}))

// R1321 char-string-edit-desc
// N.B. C1313 disallows any kind parameter on the character literal.
constexpr auto charStringEditDesc = space >>
    (charLiteralConstantWithoutKind || rawHollerithLiteral);

// R1303 format-items -> format-item [[,] format-item]...
constexpr auto formatItems =
    nonemptySeparated(space >> Parser<format::FormatItem>{}, maybe(","_tok));

// R1306 r -> digit-string
constexpr auto repeat = space >> digitString;

// R1304 format-item ->
//         [r] data-edit-desc | control-edit-desc | char-string-edit-desc |
//         [r] ( format-items )
TYPE_PARSER(construct<format::FormatItem>{}(
                maybe(repeat), Parser<format::IntrinsicTypeDataEditDesc>{}) ||
    construct<format::FormatItem>{}(
        maybe(repeat), Parser<format::DerivedTypeDataEditDesc>{}) ||
    construct<format::FormatItem>{}(Parser<format::ControlEditDesc>{}) ||
    construct<format::FormatItem>{}(charStringEditDesc) ||
    construct<format::FormatItem>{}(maybe(repeat), parenthesized(formatItems)))

// R1302 format-specification ->
//         ( [format-items] ) | ( [format-items ,] unlimited-format-item )
// R1305 unlimited-format-item -> * ( format-items )
TYPE_PARSER(parenthesized(
    construct<format::FormatSpecification>{}(
        defaulted(formatItems / ","), "*" >> parenthesized(formatItems)) ||
    construct<format::FormatSpecification>{}(defaulted(formatItems))))
// R1308 w -> digit-string
// R1309 m -> digit-string
// R1310 d -> digit-string
// R1311 e -> digit-string
constexpr auto width = repeat;
constexpr auto mandatoryWidth = construct<std::optional<int>>{}(width);
constexpr auto digits = repeat;
constexpr auto noInt = construct<std::optional<int>>{};
constexpr auto mandatoryDigits = "." >> construct<std::optional<int>>{}(width);

// R1307 data-edit-desc ->
//         I w [. m] | B w [. m] | O w [. m] | Z w [. m] | F w . d |
//         E w . d [E e] | EN w . d [E e] | ES w . d [E e] | EX w . d [E e] |
//         G w [. d [E e]] | L w | A [w] | D w . d |
//         DT [char-literal-constant] [( v-list )]
// (part 1 of 2)
TYPE_PARSER(
    construct<format::IntrinsicTypeDataEditDesc>{}(
        "I"_ch >> pure(format::IntrinsicTypeDataEditDesc::Kind::I) ||
            "B"_ch >> pure(format::IntrinsicTypeDataEditDesc::Kind::B) ||
            "O"_ch >> pure(format::IntrinsicTypeDataEditDesc::Kind::O) ||
            "Z"_ch >> pure(format::IntrinsicTypeDataEditDesc::Kind::Z),
        mandatoryWidth, maybe("." >> digits), noInt) ||
    construct<format::IntrinsicTypeDataEditDesc>{}(
        "F"_ch >> pure(format::IntrinsicTypeDataEditDesc::Kind::F) ||
            "D"_ch >> pure(format::IntrinsicTypeDataEditDesc::Kind::D),
        mandatoryWidth, mandatoryDigits, noInt) ||
    construct<format::IntrinsicTypeDataEditDesc>{}("E"_ch >>
            ("N"_ch >> pure(format::IntrinsicTypeDataEditDesc::Kind::EN) ||
                "S"_ch >> pure(format::IntrinsicTypeDataEditDesc::Kind::ES) ||
                "X"_ch >> pure(format::IntrinsicTypeDataEditDesc::Kind::EX) ||
                pure(format::IntrinsicTypeDataEditDesc::Kind::E)),
        mandatoryWidth, mandatoryDigits, maybe("E"_ch >> digits)) ||
    construct<format::IntrinsicTypeDataEditDesc>{}(
        "G"_ch >> pure(format::IntrinsicTypeDataEditDesc::Kind::G),
        mandatoryWidth, mandatoryDigits, maybe("E"_ch >> digits)) ||
    construct<format::IntrinsicTypeDataEditDesc>{}(
        "G"_ch >> pure(format::IntrinsicTypeDataEditDesc::Kind::G) ||
            "L"_ch >> pure(format::IntrinsicTypeDataEditDesc::Kind::L),
        mandatoryWidth, noInt, noInt) ||
    construct<format::IntrinsicTypeDataEditDesc>{}(
        "A"_ch >> pure(format::IntrinsicTypeDataEditDesc::Kind::A),
        maybe(width), noInt, noInt) ||
    // PGI/Intel extension: omitting width (and all else that follows)
    extension(construct<format::IntrinsicTypeDataEditDesc>{}(
        "I"_ch >> pure(format::IntrinsicTypeDataEditDesc::Kind::I) ||
            ("B"_ch / !letter /* don't occlude BN & BZ */) >>
                pure(format::IntrinsicTypeDataEditDesc::Kind::B) ||
            "O"_ch >> pure(format::IntrinsicTypeDataEditDesc::Kind::O) ||
            "Z"_ch >> pure(format::IntrinsicTypeDataEditDesc::Kind::Z) ||
            "F"_ch >> pure(format::IntrinsicTypeDataEditDesc::Kind::F) ||
            ("D"_ch / !letter /* don't occlude DC & DP */) >>
                pure(format::IntrinsicTypeDataEditDesc::Kind::D) ||
            "E"_ch >>
                ("N"_ch >> pure(format::IntrinsicTypeDataEditDesc::Kind::EN) ||
                    "S"_ch >>
                        pure(format::IntrinsicTypeDataEditDesc::Kind::ES) ||
                    "X"_ch >>
                        pure(format::IntrinsicTypeDataEditDesc::Kind::EX) ||
                    pure(format::IntrinsicTypeDataEditDesc::Kind::E)) ||
            "G"_ch >> pure(format::IntrinsicTypeDataEditDesc::Kind::G) ||
            "L"_ch >> pure(format::IntrinsicTypeDataEditDesc::Kind::L),
        noInt, noInt, noInt)))

// R1307 data-edit-desc (part 2 of 2)
// R1312 v -> [sign] digit-string
TYPE_PARSER("D"_ch >> "T"_ch >>
    construct<format::DerivedTypeDataEditDesc>{}(
        space >> defaulted(charLiteralConstantWithoutKind),
        defaulted(parenthesized(nonemptyList(space >> signedDigitString)))))

// R1314 k -> [sign] digit-string
constexpr auto count = space >> DigitStringAsPositive{};
constexpr auto scaleFactor = space >> signedDigitString;

// R1313 control-edit-desc ->
//         position-edit-desc | [r] / | : | sign-edit-desc | k P |
//         blank-interp-edit-desc | round-edit-desc | decimal-edit-desc
// R1315 position-edit-desc -> T n | TL n | TR n | n X
// R1316 n -> digit-string
// R1317 sign-edit-desc -> SS | SP | S
// R1318 blank-interp-edit-desc -> BN | BZ
// R1319 round-edit-desc -> RU | RD | RZ | RN | RC | RP
// R1320 decimal-edit-desc -> DC | DP
TYPE_PARSER(construct<format::ControlEditDesc>{}("T"_ch >>
                    ("L"_ch >> pure(format::ControlEditDesc::Kind::TL) ||
                        "R"_ch >> pure(format::ControlEditDesc::Kind::TR) ||
                        pure(format::ControlEditDesc::Kind::T)),
                count) ||
    construct<format::ControlEditDesc>{}(count,
        "X"_ch >> pure(format::ControlEditDesc::Kind::X) ||
            "/"_ch >> pure(format::ControlEditDesc::Kind::Slash)) ||
    construct<format::ControlEditDesc>{}(
        "X"_ch >> pure(format::ControlEditDesc::Kind::X) ||
        "/"_ch >> pure(format::ControlEditDesc::Kind::Slash)) ||
    construct<format::ControlEditDesc>{}(
        scaleFactor, "P"_ch >> pure(format::ControlEditDesc::Kind::P)) ||
    ":"_ch >> construct<format::ControlEditDesc>{}(
                  pure(format::ControlEditDesc::Kind::Colon)) ||
    "S"_ch >> ("S"_ch >> construct<format::ControlEditDesc>{}(
                             pure(format::ControlEditDesc::Kind::SS)) ||
                  "P"_ch >> construct<format::ControlEditDesc>{}(
                                pure(format::ControlEditDesc::Kind::SP)) ||
                  construct<format::ControlEditDesc>{}(
                      pure(format::ControlEditDesc::Kind::S))) ||
    "B"_ch >> ("N"_ch >> construct<format::ControlEditDesc>{}(
                             pure(format::ControlEditDesc::Kind::BN)) ||
                  "Z"_ch >> construct<format::ControlEditDesc>{}(
                                pure(format::ControlEditDesc::Kind::BZ))) ||
    "R"_ch >> ("U"_ch >> construct<format::ControlEditDesc>{}(
                             pure(format::ControlEditDesc::Kind::RU)) ||
                  "D"_ch >> construct<format::ControlEditDesc>{}(
                                pure(format::ControlEditDesc::Kind::RD)) ||
                  "Z"_ch >> construct<format::ControlEditDesc>{}(
                                pure(format::ControlEditDesc::Kind::RZ)) ||
                  "N"_ch >> construct<format::ControlEditDesc>{}(
                                pure(format::ControlEditDesc::Kind::RN)) ||
                  "C"_ch >> construct<format::ControlEditDesc>{}(
                                pure(format::ControlEditDesc::Kind::RC)) ||
                  "P"_ch >> construct<format::ControlEditDesc>{}(
                                pure(format::ControlEditDesc::Kind::RP))) ||
    "D"_ch >> ("C"_ch >> construct<format::ControlEditDesc>{}(
                             pure(format::ControlEditDesc::Kind::DC)) ||
                  "P"_ch >> construct<format::ControlEditDesc>{}(
                                pure(format::ControlEditDesc::Kind::DP))))

// R1401 main-program ->
//         [program-stmt] [specification-part] [execution-part]
//         [internal-subprogram-part] end-program-stmt
TYPE_CONTEXT_PARSER("main program"_en_US,
    construct<MainProgram>{}(maybe(statement(Parser<ProgramStmt>{})),
        specificationPart, executionPart, maybe(internalSubprogramPart),
        unterminatedStatement(Parser<EndProgramStmt>{})))

// R1402 program-stmt -> PROGRAM program-name
// PGI allows empty parentheses after the name.
TYPE_CONTEXT_PARSER("PROGRAM statement"_en_US,
    construct<ProgramStmt>{}(
        "PROGRAM" >> name / maybe(extension(parenthesized(ok)))))

// R1403 end-program-stmt -> END [PROGRAM [program-name]]
TYPE_CONTEXT_PARSER("END PROGRAM statement"_en_US,
    construct<EndProgramStmt>{}(recovery(
        "END PROGRAM" >> maybe(name) || bareEnd, endStmtErrorRecovery)))

// R1404 module ->
//         module-stmt [specification-part] [module-subprogram-part]
//         end-module-stmt
TYPE_CONTEXT_PARSER("module"_en_US,
    construct<Module>{}(statement(Parser<ModuleStmt>{}), specificationPart,
        maybe(Parser<ModuleSubprogramPart>{}),
        unterminatedStatement(Parser<EndModuleStmt>{})))

// R1405 module-stmt -> MODULE module-name
TYPE_CONTEXT_PARSER(
    "MODULE statement"_en_US, "MODULE" >> construct<ModuleStmt>{}(name))

// R1406 end-module-stmt -> END [MODULE [module-name]]
TYPE_CONTEXT_PARSER("END MODULE statement"_en_US,
    construct<EndModuleStmt>{}(
        recovery("END MODULE" >> maybe(name) || bareEnd, endStmtErrorRecovery)))

// R1407 module-subprogram-part -> contains-stmt [module-subprogram]...
TYPE_CONTEXT_PARSER("module subprogram part"_en_US,
    construct<ModuleSubprogramPart>{}(statement(containsStmt),
        many(startNewSubprogram >> Parser<ModuleSubprogram>{})))

// R1408 module-subprogram ->
//         function-subprogram | subroutine-subprogram |
//         separate-module-subprogram
TYPE_PARSER(construct<ModuleSubprogram>{}(indirect(functionSubprogram)) ||
    construct<ModuleSubprogram>{}(indirect(subroutineSubprogram)) ||
    construct<ModuleSubprogram>{}(indirect(Parser<SeparateModuleSubprogram>{})))

// R1410 module-nature -> INTRINSIC | NON_INTRINSIC
constexpr auto moduleNature = "INTRINSIC" >>
        pure(UseStmt::ModuleNature::Intrinsic) ||
    "NON_INTRINSIC" >> pure(UseStmt::ModuleNature::Non_Intrinsic);

// R1409 use-stmt ->
//         USE [[, module-nature] ::] module-name [, rename-list] |
//         USE [[, module-nature] ::] module-name , ONLY : [only-list]
TYPE_PARSER(construct<UseStmt>{}("USE" >> optionalBeforeColons(moduleNature),
                name, ", ONLY :" >> optionalList(Parser<Only>{})) ||
    construct<UseStmt>{}("USE" >> optionalBeforeColons(moduleNature), name,
        defaulted("," >> nonemptyList(Parser<Rename>{}))))

// R1411 rename ->
//         local-name => use-name |
//         OPERATOR ( local-defined-operator ) =>
//           OPERATOR ( use-defined-operator )
TYPE_PARSER(construct<Rename>{}("OPERATOR (" >>
                construct<Rename::Operators>{}(
                    definedOpName / ") => OPERATOR (", definedOpName / ")")) ||
    construct<Rename>{}(construct<Rename::Names>{}(name, "=>" >> name)))

// R1412 only -> generic-spec | only-use-name | rename
// R1413 only-use-name -> use-name
TYPE_PARSER(construct<Only>{}(Parser<Rename>{}) ||
    construct<Only>{}(indirect(genericSpec)) ||
    construct<Only>{}(name))  // TODO: ambiguous, accepted by genericSpec

// R1416 submodule ->
//         submodule-stmt [specification-part] [module-subprogram-part]
//         end-submodule-stmt
TYPE_CONTEXT_PARSER("submodule"_en_US,
    construct<Submodule>{}(statement(Parser<SubmoduleStmt>{}),
        specificationPart, maybe(Parser<ModuleSubprogramPart>{}),
        unterminatedStatement(Parser<EndSubmoduleStmt>{})))

// R1417 submodule-stmt -> SUBMODULE ( parent-identifier ) submodule-name
TYPE_CONTEXT_PARSER("SUBMODULE statement"_en_US,
    construct<SubmoduleStmt>{}(
        "SUBMODULE" >> parenthesized(Parser<ParentIdentifier>{}), name))

// R1418 parent-identifier -> ancestor-module-name [: parent-submodule-name]
TYPE_PARSER(construct<ParentIdentifier>{}(name, maybe(":" >> name)))

// R1419 end-submodule-stmt -> END [SUBMODULE [submodule-name]]
TYPE_CONTEXT_PARSER("END SUBMODULE statement"_en_US,
    construct<EndSubmoduleStmt>{}(recovery(
        "END SUBMODULE" >> maybe(name) || bareEnd, endStmtErrorRecovery)))

// R1420 block-data -> block-data-stmt [specification-part] end-block-data-stmt
TYPE_CONTEXT_PARSER("BLOCK DATA subprogram"_en_US,
    construct<BlockData>{}(statement(Parser<BlockDataStmt>{}),
        specificationPart, unterminatedStatement(Parser<EndBlockDataStmt>{})))

// R1421 block-data-stmt -> BLOCK DATA [block-data-name]
TYPE_CONTEXT_PARSER("BLOCK DATA statement"_en_US,
    "BLOCK DATA" >> construct<BlockDataStmt>{}(maybe(name)))

// R1422 end-block-data-stmt -> END [BLOCK DATA [block-data-name]]
TYPE_CONTEXT_PARSER("END BLOCK DATA statement"_en_US,
    construct<EndBlockDataStmt>{}(recovery(
        "END BLOCK DATA" >> maybe(name) || bareEnd, endStmtErrorRecovery)))

// R1501 interface-block ->
//         interface-stmt [interface-specification]... end-interface-stmt
TYPE_PARSER(construct<InterfaceBlock>{}(statement(Parser<InterfaceStmt>{}),
    many(Parser<InterfaceSpecification>{}),
    statement(Parser<EndInterfaceStmt>{})))

// R1502 interface-specification -> interface-body | procedure-stmt
TYPE_PARSER(construct<InterfaceSpecification>{}(Parser<InterfaceBody>{}) ||
    construct<InterfaceSpecification>{}(statement(Parser<ProcedureStmt>{})))

// R1503 interface-stmt -> INTERFACE [generic-spec] | ABSTRACT INTERFACE
TYPE_PARSER("INTERFACE" >> construct<InterfaceStmt>{}(maybe(genericSpec)) ||
    "ABSTRACT INTERFACE"_sptok >>
        construct<InterfaceStmt>{}(construct<Abstract>{}))

// R1504 end-interface-stmt -> END INTERFACE [generic-spec]
TYPE_PARSER(
    "END INTERFACE" >> construct<EndInterfaceStmt>{}(maybe(genericSpec)))

// R1505 interface-body ->
//         function-stmt [specification-part] end-function-stmt |
//         subroutine-stmt [specification-part] end-subroutine-stmt
TYPE_CONTEXT_PARSER("interface body"_en_US,
    construct<InterfaceBody>{}(
        construct<InterfaceBody::Function>{}(statement(functionStmt),
            indirect(specificationPart), statement(endFunctionStmt))) ||
        construct<InterfaceBody>{}(
            construct<InterfaceBody::Subroutine>{}(statement(subroutineStmt),
                indirect(specificationPart), statement(endSubroutineStmt))))

// R1507 specific-procedure -> procedure-name
constexpr auto specificProcedure = name;

// R1506 procedure-stmt -> [MODULE] PROCEDURE [::] specific-procedure-list
TYPE_PARSER(construct<ProcedureStmt>{}("MODULE PROCEDURE"_sptok >>
                    pure(ProcedureStmt::Kind::ModuleProcedure),
                maybe("::"_tok) >> nonemptyList(specificProcedure)) ||
    construct<ProcedureStmt>{}(
        "PROCEDURE" >> pure(ProcedureStmt::Kind::Procedure),
        maybe("::"_tok) >> nonemptyList(specificProcedure)))

// R1508 generic-spec ->
//         generic-name | OPERATOR ( defined-operator ) |
//         ASSIGNMENT ( = ) | defined-io-generic-spec
// R1509 defined-io-generic-spec ->
//         READ ( FORMATTED ) | READ ( UNFORMATTED ) |
//         WRITE ( FORMATTED ) | WRITE ( UNFORMATTED )
TYPE_PARSER(construct<GenericSpec>{}(
                "OPERATOR" >> parenthesized(Parser<DefinedOperator>{})) ||
    construct<GenericSpec>{}(
        "ASSIGNMENT ( = )" >> construct<GenericSpec::Assignment>{}) ||
    construct<GenericSpec>{}(
        "READ ( FORMATTED )" >> construct<GenericSpec::ReadFormatted>{}) ||
    construct<GenericSpec>{}(
        "READ ( UNFORMATTED )" >> construct<GenericSpec::ReadUnformatted>{}) ||
    construct<GenericSpec>{}(
        "WRITE ( FORMATTED )" >> construct<GenericSpec::WriteFormatted>{}) ||
    construct<GenericSpec>{}("WRITE ( UNFORMATTED )" >>
        construct<GenericSpec::WriteUnformatted>{}) ||
    construct<GenericSpec>{}(name))

// R1510 generic-stmt ->
//         GENERIC [, access-spec] :: generic-spec => specific-procedure-list
TYPE_PARSER("GENERIC" >> construct<GenericStmt>{}(maybe("," >> accessSpec),
                             "::" >> genericSpec,
                             "=>" >> nonemptyList(specificProcedure)))

// R1511 external-stmt -> EXTERNAL [::] external-name-list
TYPE_PARSER("EXTERNAL" >> maybe("::"_tok) >>
    construct<ExternalStmt>{}(nonemptyList(name)))

// R1512 procedure-declaration-stmt ->
//         PROCEDURE ( [proc-interface] ) [[, proc-attr-spec]... ::]
//         proc-decl-list
TYPE_PARSER("PROCEDURE" >>
    construct<ProcedureDeclarationStmt>{}(parenthesized(maybe(procInterface)),
        optionalListBeforeColons(Parser<ProcAttrSpec>{}),
        nonemptyList(procDecl)))

// R1513 proc-interface -> interface-name | declaration-type-spec
// R1516 interface-name -> name
TYPE_PARSER(construct<ProcInterface>{}(declarationTypeSpec) ||
    construct<ProcInterface>{}(name))

// R1514 proc-attr-spec ->
//         access-spec | proc-language-binding-spec | INTENT ( intent-spec ) |
//         OPTIONAL | POINTER | PROTECTED | SAVE
TYPE_PARSER(construct<ProcAttrSpec>{}(accessSpec) ||
    construct<ProcAttrSpec>{}(languageBindingSpec) ||
    construct<ProcAttrSpec>{}("INTENT" >> parenthesized(intentSpec)) ||
    construct<ProcAttrSpec>{}(optional) || construct<ProcAttrSpec>{}(pointer) ||
    construct<ProcAttrSpec>{}(protectedAttr) || construct<ProcAttrSpec>{}(save))

// R1515 proc-decl -> procedure-entity-name [=> proc-pointer-init]
TYPE_PARSER(
    construct<ProcDecl>{}(name, maybe("=>" >> Parser<ProcPointerInit>{})))

// R1517 proc-pointer-init -> null-init | initial-proc-target
// R1518 initial-proc-target -> procedure-name
TYPE_PARSER(construct<ProcPointerInit>{}(nullInit) ||
    construct<ProcPointerInit>{}(name))

// R1519 intrinsic-stmt -> INTRINSIC [::] intrinsic-procedure-name-list
TYPE_PARSER("INTRINSIC" >> maybe("::"_tok) >>
    construct<IntrinsicStmt>{}(nonemptyList(name)))

// R1520 function-reference -> procedure-designator ( [actual-arg-spec-list] )
// Without recourse to a symbol table, a parse of the production for
// variable as part of a procedure-designator will overshoot and consume
// any actual argument list, since a pointer-valued function-reference is
// acceptable as an alternative for a variable (since Fortran 2008).
template<>
std::optional<FunctionReference> Parser<FunctionReference>::Parse(
    ParseState *state) {
  state->PushContext("function reference"_en_US);
  std::optional<Variable> var{variable.Parse(state)};
  if (var.has_value()) {
    if (auto funcref = std::get_if<Indirection<FunctionReference>>(&var->u)) {
      // The parsed variable is a function-reference, so just return it.
      state->PopContext();
      return {std::move(**funcref)};
    }
    Designator *desig{&*std::get<Indirection<Designator>>(var->u)};
    if (std::optional<Call> call{desig->ConvertToCall(state->userState())}) {
      if (!std::get<std::list<ActualArgSpec>>(call.value().t).empty()) {
        // Parsed a designator that ended with a nonempty list of subscripts
        // that have all been converted to actual arguments.
        state->PopContext();
        return {FunctionReference{std::move(call.value())}};
      }
    }
    state->Say("expected (arguments)"_err_en_US);
  }
  state->PopContext();
  return {};
}

// R1521 call-stmt -> CALL procedure-designator [( [actual-arg-spec-list] )]
template<> std::optional<CallStmt> Parser<CallStmt>::Parse(ParseState *state) {
  static constexpr auto parser =
      inContext("CALL statement"_en_US, "CALL" >> variable);
  std::optional<Variable> var{parser.Parse(state)};
  if (var.has_value()) {
    if (auto funcref = std::get_if<Indirection<FunctionReference>>(&var->u)) {
      state->PopContext();
      return {CallStmt{std::move((*funcref)->v)}};
    }
    Designator *desig{&*std::get<Indirection<Designator>>(var->u)};
    if (std::optional<Call> call{desig->ConvertToCall(state->userState())}) {
      return {CallStmt{std::move(call.value())}};
    }
  }
  return {};
}

// R1522 procedure-designator ->
//         procedure-name | proc-component-ref | data-ref % binding-name
// N.B. Not implemented as an independent production; instead, instances
// of procedure-designator must be reconstructed from portions of parses of
// variable.

// R1523 actual-arg-spec -> [keyword =] actual-arg
TYPE_PARSER(
    construct<ActualArgSpec>{}(maybe(keyword / "="), Parser<ActualArg>{}))

// R1524 actual-arg ->
//         expr | variable | procedure-name | proc-component-ref |
//         alt-return-spec
// N.B. the "procedure-name" and "proc-component-ref" alternatives can't
// yet be distinguished from "variable".
TYPE_PARSER(construct<ActualArg>{}(variable) / lookAhead(","_tok || ")"_tok) ||
    construct<ActualArg>{}(expr) ||
    construct<ActualArg>{}(Parser<AltReturnSpec>{}) ||
    extension("%REF" >>
        construct<ActualArg>{}(
            construct<ActualArg::PercentRef>{}(parenthesized(variable)))) ||
    extension(
        "%VAL" >> construct<ActualArg>{}(
                      construct<ActualArg::PercentVal>{}(parenthesized(expr)))))

// R1525 alt-return-spec -> * label
TYPE_PARSER(star >> construct<AltReturnSpec>{}(label))

// R1527 prefix-spec ->
//         declaration-type-spec | ELEMENTAL | IMPURE | MODULE |
//         NON_RECURSIVE | PURE | RECURSIVE
TYPE_PARSER(construct<PrefixSpec>{}(declarationTypeSpec) ||
    "ELEMENTAL" >>
        construct<PrefixSpec>{}(construct<PrefixSpec::Elemental>{}) ||
    "IMPURE" >> construct<PrefixSpec>{}(construct<PrefixSpec::Impure>{}) ||
    "MODULE" >> construct<PrefixSpec>{}(construct<PrefixSpec::Module>{}) ||
    "NON_RECURSIVE" >>
        construct<PrefixSpec>{}(construct<PrefixSpec::Non_Recursive>{}) ||
    "PURE" >> construct<PrefixSpec>{}(construct<PrefixSpec::Pure>{}) ||
    "RECURSIVE" >> construct<PrefixSpec>{}(construct<PrefixSpec::Recursive>{}))

// R1529 function-subprogram ->
//         function-stmt [specification-part] [execution-part]
//         [internal-subprogram-part] end-function-stmt
TYPE_CONTEXT_PARSER("FUNCTION subprogram"_en_US,
    construct<FunctionSubprogram>{}(statement(functionStmt), specificationPart,
        executionPart, maybe(internalSubprogramPart),
        unterminatedStatement(endFunctionStmt)))

// R1530 function-stmt ->
//         [prefix] FUNCTION function-name ( [dummy-arg-name-list] ) [suffix]
// R1526 prefix -> prefix-spec [prefix-spec]...
// R1531 dummy-arg-name -> name
TYPE_CONTEXT_PARSER("FUNCTION statement"_en_US,
    construct<FunctionStmt>{}(many(prefixSpec), "FUNCTION" >> name,
        parenthesized(optionalList(name)), maybe(suffix)) ||
        extension(construct<FunctionStmt>{}(  // PGI & Intel accept "FUNCTION F"
            many(prefixSpec), "FUNCTION" >> name, construct<std::list<Name>>{},
            construct<std::optional<Suffix>>{})))

// R1532 suffix ->
//         proc-language-binding-spec [RESULT ( result-name )] |
//         RESULT ( result-name ) [proc-language-binding-spec]
TYPE_PARSER(construct<Suffix>{}(
                languageBindingSpec, maybe("RESULT" >> parenthesized(name))) ||
    construct<Suffix>{}(
        "RESULT" >> parenthesized(name), maybe(languageBindingSpec)))

// R1533 end-function-stmt -> END [FUNCTION [function-name]]
TYPE_PARSER(construct<EndFunctionStmt>{}(
    recovery("END FUNCTION" >> maybe(name) || bareEnd, endStmtErrorRecovery)))

// R1534 subroutine-subprogram ->
//         subroutine-stmt [specification-part] [execution-part]
//         [internal-subprogram-part] end-subroutine-stmt
TYPE_CONTEXT_PARSER("SUBROUTINE subprogram"_en_US,
    construct<SubroutineSubprogram>{}(statement(subroutineStmt),
        specificationPart, executionPart, maybe(internalSubprogramPart),
        unterminatedStatement(endSubroutineStmt)))

// R1535 subroutine-stmt ->
//         [prefix] SUBROUTINE subroutine-name [( [dummy-arg-list] )
//         [proc-language-binding-spec]]
TYPE_PARSER(
    construct<SubroutineStmt>{}(many(prefixSpec), "SUBROUTINE" >> name,
        parenthesized(optionalList(dummyArg)), maybe(languageBindingSpec)) ||
    construct<SubroutineStmt>{}(many(prefixSpec), "SUBROUTINE" >> name,
        defaulted(cut >> many(dummyArg)),
        defaulted(cut >> maybe(languageBindingSpec))))

// R1536 dummy-arg -> dummy-arg-name | *
TYPE_PARSER(construct<DummyArg>{}(name) || construct<DummyArg>{}(star))

// R1537 end-subroutine-stmt -> END [SUBROUTINE [subroutine-name]]
TYPE_PARSER(construct<EndSubroutineStmt>{}(
    recovery("END SUBROUTINE" >> maybe(name) || bareEnd, endStmtErrorRecovery)))

// R1538 separate-module-subprogram ->
//         mp-subprogram-stmt [specification-part] [execution-part]
//         [internal-subprogram-part] end-mp-subprogram-stmt
TYPE_CONTEXT_PARSER("separate module subprogram"_en_US,
    construct<SeparateModuleSubprogram>{}(statement(Parser<MpSubprogramStmt>{}),
        specificationPart, executionPart, maybe(internalSubprogramPart),
        statement(Parser<EndMpSubprogramStmt>{})))

// R1539 mp-subprogram-stmt -> MODULE PROCEDURE procedure-name
TYPE_CONTEXT_PARSER("MODULE PROCEDURE statement"_en_US,
    construct<MpSubprogramStmt>{}("MODULE PROCEDURE"_sptok >> name))

// R1540 end-mp-subprogram-stmt -> END [PROCEDURE [procedure-name]]
TYPE_CONTEXT_PARSER("END PROCEDURE statement"_en_US,
    construct<EndMpSubprogramStmt>{}(recovery(
        "END PROCEDURE" >> maybe(name) || bareEnd, endStmtErrorRecovery)))

// R1541 entry-stmt -> ENTRY entry-name [( [dummy-arg-list] ) [suffix]]
TYPE_PARSER("ENTRY" >>
    (construct<EntryStmt>{}(
         name, parenthesized(optionalList(dummyArg)), maybe(suffix)) ||
        construct<EntryStmt>{}(name, construct<std::list<DummyArg>>{},
            construct<std::optional<Suffix>>{})))

// R1542 return-stmt -> RETURN [scalar-int-expr]
TYPE_CONTEXT_PARSER("RETURN statement"_en_US,
    "RETURN" >> construct<ReturnStmt>{}(maybe(scalarIntExpr)))

// R1543 contains-stmt -> CONTAINS
TYPE_PARSER("CONTAINS" >> construct<ContainsStmt>{})

// R1544 stmt-function-stmt ->
//         function-name ( [dummy-arg-name-list] ) = scalar-expr
TYPE_CONTEXT_PARSER("statement function definition"_en_US,
    construct<StmtFunctionStmt>{}(
        name, parenthesized(optionalList(name)), "=" >> scalar(expr)))

// Directives, extensions, and deprecated statements
// !DIR$ IVDEP
// !DIR$ IGNORE_TKR [ [(tkr...)] name ]...
constexpr auto beginDirective = skipEmptyLines >> space >> "!"_ch;
constexpr auto endDirective = space >> endOfLine;
constexpr auto ivdep = "DIR$ IVDEP" >> construct<CompilerDirective::IVDEP>{};
constexpr auto ignore_tkr = "DIR$ IGNORE_TKR" >>
    optionalList(construct<CompilerDirective::IgnoreTKR>{}(
        defaulted(parenthesized(some("tkr"_ch))), name));
TYPE_PARSER(beginDirective >> sourced(construct<CompilerDirective>{}(ivdep) ||
                                  construct<CompilerDirective>{}(ignore_tkr)) /
        endDirective)

TYPE_PARSER(
    extension(construct<BasedPointerStmt>{}("POINTER (" >> objectName / ",",
        objectName, maybe(Parser<ArraySpec>{}) / ")")))

TYPE_PARSER(construct<StructureStmt>{}("STRUCTURE /" >> name / "/", pure(true),
                optionalList(entityDecl)) ||
    construct<StructureStmt>{}(
        "STRUCTURE" >> name, pure(false), defaulted(cut >> many(entityDecl))))

constexpr struct StructureComponents {
  using resultType = DataComponentDefStmt;
  static std::optional<DataComponentDefStmt> Parse(ParseState *state) {
    static constexpr auto stmt = Parser<DataComponentDefStmt>{};
    std::optional<DataComponentDefStmt> defs{stmt.Parse(state)};
    if (defs.has_value()) {
      if (auto ustate = state->userState()) {
        for (const auto &decl : std::get<std::list<ComponentDecl>>(defs->t)) {
          ustate->NoteOldStructureComponent(std::get<Name>(decl.t).source);
        }
      }
    }
    return defs;
  }
} structureComponents;

TYPE_PARSER(construct<StructureField>{}(statement(structureComponents)) ||
    construct<StructureField>{}(indirect(Parser<Union>{})) ||
    construct<StructureField>{}(indirect(Parser<StructureDef>{})))

TYPE_CONTEXT_PARSER("STRUCTURE definition"_en_US,
    extension(construct<StructureDef>{}(statement(Parser<StructureStmt>{}),
        many(Parser<StructureField>{}),
        statement(
            "END STRUCTURE" >> construct<StructureDef::EndStructureStmt>{}))))

TYPE_CONTEXT_PARSER("UNION definition"_en_US,
    construct<Union>{}(statement("UNION" >> construct<Union::UnionStmt>{}),
        many(Parser<Map>{}),
        statement("END UNION" >> construct<Union::EndUnionStmt>{})))

TYPE_CONTEXT_PARSER("MAP definition"_en_US,
    construct<Map>{}(statement("MAP" >> construct<Map::MapStmt>{}),
        many(Parser<StructureField>{}),
        statement("END MAP" >> construct<Map::EndMapStmt>{})))

TYPE_CONTEXT_PARSER("arithmetic IF statement"_en_US,
    deprecated("IF" >> construct<ArithmeticIfStmt>{}(parenthesized(expr),
                           label / ",", label / ",", label)))

TYPE_CONTEXT_PARSER("ASSIGN statement"_en_US,
    deprecated("ASSIGN" >> construct<AssignStmt>{}(label, "TO" >> name)))

TYPE_CONTEXT_PARSER("assigned GOTO statement"_en_US,
    deprecated("GO TO" >>
        construct<AssignedGotoStmt>{}(name,
            defaulted(maybe(","_tok) >> parenthesized(nonemptyList(label))))))

TYPE_CONTEXT_PARSER("PAUSE statement"_en_US,
    deprecated("PAUSE" >> construct<PauseStmt>{}(maybe(Parser<StopCode>{}))))

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
//   R1030 default-char-constant-expr -> default-char-expr
//     is only used via scalar-default-char-constant-expr
}  // namespace parser
}  // namespace Fortran
#endif  // FORTRAN_PARSER_GRAMMAR_H_
