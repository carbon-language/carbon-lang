#include "LoopMatchers.h"

namespace clang {
namespace loop_migrate {

using namespace clang::ast_matchers;
const char LoopName[] = "forLoop";
const char ConditionBoundName[] = "conditionBound";
const char ConditionVarName[] = "conditionVar";
const char IncrementVarName[] = "incrementVar";
const char InitVarName[] = "initVar";
const char EndCallName[] = "endCall";
const char ConditionEndVarName[] = "conditionEndVar";
const char EndVarName[] = "endVar";

// shared matchers
static const TypeMatcher AnyType = anything();

static const StatementMatcher IntegerComparisonMatcher =
    expression(ignoringParenImpCasts(declarationReference(to(
        variable(hasType(isInteger())).bind(ConditionVarName)))));

static const DeclarationMatcher InitToZeroMatcher =
    variable(hasInitializer(ignoringParenImpCasts(
        integerLiteral(equals(0))))).bind(InitVarName);

static const StatementMatcher IncrementVarMatcher =
    declarationReference(to(
        variable(hasType(isInteger())).bind(IncrementVarName)));

// FIXME: How best to document complicated matcher expressions? They're fairly
// self-documenting...but there may be some unintuitive parts.

/// \brief The matcher for loops over arrays.
///
/// In this general example, assuming 'j' and 'k' are of integral type:
///   for (int i = 0; j < 3 + 2; ++k) { ... }
/// The following string identifers are bound to the parts of the AST:
///   ConditionVarName: 'j' (as a VarDecl)
///   ConditionBoundName: '3 + 2' (as an Expr)
///   InitVarName: 'i' (as a VarDecl)
///   IncrementVarName: 'k' (as a VarDecl)
///   LoopName: The entire for loop (as a ForStmt)
///
/// Client code will need to make sure that:
///   - The three index variables identified by the matcher are the same
///     VarDecl.
///   - The index variable is only used as an array index.
///   - All arrays indexed by the loop are the same.
StatementMatcher makeArrayLoopMatcher() {
  StatementMatcher ArrayBoundMatcher =
      expression(hasType(isInteger())).bind(ConditionBoundName);

  return forStmt(
      hasLoopInit(declarationStatement(hasSingleDecl(InitToZeroMatcher))),
      hasCondition(anyOf(binaryOperator(hasOperatorName("<"),
                                        hasLHS(IntegerComparisonMatcher),
                                        hasRHS(ArrayBoundMatcher)),
                         binaryOperator(hasOperatorName(">"),
                                        hasLHS(ArrayBoundMatcher),
                                        hasRHS(IntegerComparisonMatcher)))),
      hasIncrement(unaryOperator(hasOperatorName("++"),
                                 hasUnaryOperand(IncrementVarMatcher))))
      .bind(LoopName);
}

/// \brief The matcher used for iterator-based for loops.
///
/// This matcher is more flexible than array-based loops. It will match
/// catch loops of the following textual forms (regardless of whether the
/// iterator type is actually a pointer type or a class type):
///
/// Assuming f, g, and h are of type containerType::iterator,
///   for (containerType::iterator it = container.begin(),
///        e = createIterator(); f != g; ++h) { ... }
///   for (containerType::iterator it = container.begin();
///        f != anotherContainer.end(); ++h) { ... }
/// The following string identifiers are bound to the parts of the AST:
///   InitVarName: 'it' (as a VarDecl)
///   ConditionVarName: 'f' (as a VarDecl)
///   LoopName: The entire for loop (as a ForStmt)
///   In the first example only:
///     EndVarName: 'e' (as a VarDecl)
///     ConditionEndVarName: 'g' (as a VarDecl)
///   In the second example only:
///     EndCallName: 'container.end()' (as a CXXMemberCallExpr)
///
/// Client code will need to make sure that:
///   - The iterator variables 'it', 'f', and 'h' are the same
///   - The two containers on which 'begin' and 'end' are called are the same
///   - If the end iterator variable 'g' is defined, it is the same as 'f'
StatementMatcher makeIteratorLoopMatcher() {
  StatementMatcher BeginCallMatcher =
      memberCall(argumentCountIs(0), callee(method(hasName("begin"))));

  DeclarationMatcher InitDeclMatcher =
      variable(hasInitializer(anything())).bind(InitVarName);

  DeclarationMatcher EndDeclMatcher =
      variable(hasInitializer(anything())).bind(EndVarName);

  StatementMatcher EndCallMatcher =
      memberCall(argumentCountIs(0), callee(method(hasName("end"))));

  StatementMatcher IteratorBoundMatcher =
      expression(anyOf(ignoringParenImpCasts(declarationReference(to(
          variable().bind(ConditionEndVarName)))),
                       ignoringParenImpCasts(
                           expression(EndCallMatcher).bind(EndCallName)),
                       materializeTempExpr(
                           ignoringParenImpCasts(
                               expression(EndCallMatcher).bind(EndCallName)))));

  StatementMatcher IteratorComparisonMatcher =
      expression(ignoringParenImpCasts(declarationReference(to(
          variable().bind(ConditionVarName)))));

  StatementMatcher OverloadedNEQMatcher = overloadedOperatorCall(
      hasOverloadedOperatorName("!="),
      argumentCountIs(2),
      hasArgument(0, IteratorComparisonMatcher),
      hasArgument(1, IteratorBoundMatcher));

  return forStmt(
      hasLoopInit(anyOf(
          declarationStatement(declCountIs(2),
                               containsDeclaration(0, InitDeclMatcher),
                               containsDeclaration(1, EndDeclMatcher)),
          declarationStatement(hasSingleDecl(InitDeclMatcher)))),
      hasCondition(anyOf(
          binaryOperator(hasOperatorName("!="),
                         hasLHS(IteratorComparisonMatcher),
                         hasRHS(IteratorBoundMatcher)),
          binaryOperator(hasOperatorName("!="),
                         hasLHS(IteratorBoundMatcher),
                         hasRHS(IteratorComparisonMatcher)),
          OverloadedNEQMatcher)),
      hasIncrement(anyOf(
          unaryOperator(hasOperatorName("++"),
                        hasUnaryOperand(declarationReference(to(
                            variable(hasType(pointsTo(AnyType)))
                            .bind(IncrementVarName))))),
          overloadedOperatorCall(
              hasOverloadedOperatorName("++"),
              hasArgument(0, declarationReference(to(
                  variable().bind(IncrementVarName))))))))
      .bind(LoopName);
}

/// \brief The matcher used for array-like containers (pseudoarrays).
///
/// This matcher is more flexible than array-based loops. It will match
/// loops of the following textual forms (regardless of whether the
/// iterator type is actually a pointer type or a class type):
///
/// Assuming f, g, and h are of type containerType::iterator,
///   for (int i = 0, j = container.size(); f < g; ++h) { ... }
///   for (int i = 0; f < container.size(); ++h) { ... }
/// The following string identifiers are bound to the parts of the AST:
///   InitVarName: 'i' (as a VarDecl)
///   ConditionVarName: 'f' (as a VarDecl)
///   LoopName: The entire for loop (as a ForStmt)
///   In the first example only:
///     EndVarName: 'j' (as a VarDecl)
///     ConditionEndVarName: 'g' (as a VarDecl)
///   In the second example only:
///     EndCallName: 'container.size()' (as a CXXMemberCallExpr)
///
/// Client code will need to make sure that:
///   - The index variables 'i', 'f', and 'h' are the same
///   - The containers on which 'size()' is called is the container indexed
///   - The index variable is only used in overloaded operator[] or
///     container.at()
///   - If the end iterator variable 'g' is defined, it is the same as 'j'
///   - The container's iterators would not be invalidated during the loop
StatementMatcher makePseudoArrayLoopMatcher() {
  StatementMatcher SizeCallMatcher =
      memberCall(argumentCountIs(0), callee(method(anyOf(hasName("size"),
                                                         hasName("length")))));

  StatementMatcher EndInitMatcher =
      expression(anyOf(
          ignoringParenImpCasts(expression(SizeCallMatcher).bind(EndCallName)),
          explicitCast(hasSourceExpression(ignoringParenImpCasts(
              expression(SizeCallMatcher).bind(EndCallName))))));

  DeclarationMatcher EndDeclMatcher =
       variable(hasInitializer(EndInitMatcher)).bind(EndVarName);

  StatementMatcher IndexBoundMatcher =
      expression(anyOf(
          ignoringParenImpCasts(declarationReference(to(
              variable(hasType(isInteger())).bind(ConditionEndVarName)))),
          EndInitMatcher));

  return forStmt(
      hasLoopInit(anyOf(
          declarationStatement(declCountIs(2),
                               containsDeclaration(0, InitToZeroMatcher),
                               containsDeclaration(1, EndDeclMatcher)),
          declarationStatement(hasSingleDecl(InitToZeroMatcher)))),
      hasCondition(anyOf(
          binaryOperator(hasOperatorName("<"),
                         hasLHS(IntegerComparisonMatcher),
                         hasRHS(IndexBoundMatcher)),
          binaryOperator(hasOperatorName(">"),
                         hasLHS(IndexBoundMatcher),
                         hasRHS(IntegerComparisonMatcher)))),
      hasIncrement(unaryOperator(
          hasOperatorName("++"),
          hasUnaryOperand(IncrementVarMatcher))))
      .bind(LoopName);
}

} // namespace loop_migrate
} // namespace clang
