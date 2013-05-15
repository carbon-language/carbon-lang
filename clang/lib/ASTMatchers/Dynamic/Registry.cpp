//===--- Registry.cpp - Matcher registry -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===------------------------------------------------------------===//
///
/// \file
/// \brief Registry map populated at static initialization time.
///
//===------------------------------------------------------------===//

#include "clang/ASTMatchers/Dynamic/Registry.h"

#include <utility>

#include "Marshallers.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ManagedStatic.h"

namespace clang {
namespace ast_matchers {
namespace dynamic {
namespace {

using internal::MatcherCreateCallback;

typedef llvm::StringMap<const MatcherCreateCallback *> ConstructorMap;
class RegistryMaps {
public:
  RegistryMaps();
  ~RegistryMaps();

  const ConstructorMap &constructors() const { return Constructors; }

private:
  void registerMatcher(StringRef MatcherName, MatcherCreateCallback *Callback);
  ConstructorMap Constructors;
};

void RegistryMaps::registerMatcher(StringRef MatcherName,
                                   MatcherCreateCallback *Callback) {
  Constructors[MatcherName] = Callback;
}

#define REGISTER_MATCHER(name)                                                 \
  registerMatcher(#name, internal::makeMatcherAutoMarshall(                    \
                             ::clang::ast_matchers::name, #name));

/// \brief Generate a registry map with all the known matchers.
RegistryMaps::RegistryMaps() {
  // TODO: This list is not complete. It only has non-overloaded matchers,
  // which are the simplest to add to the system. Overloaded matchers require
  // more supporting code that was omitted from the first revision for
  // simplicitly of code review.

  REGISTER_MATCHER(binaryOperator);
  REGISTER_MATCHER(bindTemporaryExpr);
  REGISTER_MATCHER(boolLiteral);
  REGISTER_MATCHER(callExpr);
  REGISTER_MATCHER(characterLiteral);
  REGISTER_MATCHER(compoundStmt);
  REGISTER_MATCHER(conditionalOperator);
  REGISTER_MATCHER(constCastExpr);
  REGISTER_MATCHER(constructExpr);
  REGISTER_MATCHER(constructorDecl);
  REGISTER_MATCHER(declRefExpr);
  REGISTER_MATCHER(declStmt);
  REGISTER_MATCHER(defaultArgExpr);
  REGISTER_MATCHER(doStmt);
  REGISTER_MATCHER(dynamicCastExpr);
  REGISTER_MATCHER(explicitCastExpr);
  REGISTER_MATCHER(expr);
  REGISTER_MATCHER(fieldDecl);
  REGISTER_MATCHER(forStmt);
  REGISTER_MATCHER(functionDecl);
  REGISTER_MATCHER(hasAnyParameter);
  REGISTER_MATCHER(hasAnySubstatement);
  REGISTER_MATCHER(hasConditionVariableStatement);
  REGISTER_MATCHER(hasDestinationType);
  REGISTER_MATCHER(hasEitherOperand);
  REGISTER_MATCHER(hasFalseExpression);
  REGISTER_MATCHER(hasImplicitDestinationType);
  REGISTER_MATCHER(hasInitializer);
  REGISTER_MATCHER(hasLHS);
  REGISTER_MATCHER(hasName);
  REGISTER_MATCHER(hasObjectExpression);
  REGISTER_MATCHER(hasRHS);
  REGISTER_MATCHER(hasSourceExpression);
  REGISTER_MATCHER(hasTrueExpression);
  REGISTER_MATCHER(hasUnaryOperand);
  REGISTER_MATCHER(ifStmt);
  REGISTER_MATCHER(implicitCastExpr);
  REGISTER_MATCHER(integerLiteral);
  REGISTER_MATCHER(isArrow);
  REGISTER_MATCHER(isConstQualified);
  REGISTER_MATCHER(isImplicit);
  REGISTER_MATCHER(member);
  REGISTER_MATCHER(memberExpr);
  REGISTER_MATCHER(methodDecl);
  REGISTER_MATCHER(namedDecl);
  REGISTER_MATCHER(newExpr);
  REGISTER_MATCHER(ofClass);
  REGISTER_MATCHER(on);
  REGISTER_MATCHER(onImplicitObjectArgument);
  REGISTER_MATCHER(operatorCallExpr);
  REGISTER_MATCHER(recordDecl);
  REGISTER_MATCHER(reinterpretCastExpr);
  REGISTER_MATCHER(staticCastExpr);
  REGISTER_MATCHER(stmt);
  REGISTER_MATCHER(stringLiteral);
  REGISTER_MATCHER(switchCase);
  REGISTER_MATCHER(to);
  REGISTER_MATCHER(unaryOperator);
  REGISTER_MATCHER(varDecl);
  REGISTER_MATCHER(whileStmt);
}

RegistryMaps::~RegistryMaps() {
  for (ConstructorMap::iterator it = Constructors.begin(),
                                end = Constructors.end();
       it != end; ++it) {
    delete it->second;
  }
}

static llvm::ManagedStatic<RegistryMaps> RegistryData;

} // anonymous namespace

// static
DynTypedMatcher *Registry::constructMatcher(StringRef MatcherName,
                                            const SourceRange &NameRange,
                                            ArrayRef<ParserValue> Args,
                                            Diagnostics *Error) {
  ConstructorMap::const_iterator it =
      RegistryData->constructors().find(MatcherName);
  if (it == RegistryData->constructors().end()) {
    Error->pushErrorFrame(NameRange, Error->ET_RegistryNotFound)
        << MatcherName;
    return NULL;
  }

  return it->second->run(NameRange, Args, Error);
}

}  // namespace dynamic
}  // namespace ast_matchers
}  // namespace clang
