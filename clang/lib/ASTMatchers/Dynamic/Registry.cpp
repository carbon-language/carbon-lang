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
  assert(Constructors.find(MatcherName) == Constructors.end());
  Constructors[MatcherName] = Callback;
}

#define REGISTER_MATCHER(name)                                                 \
  registerMatcher(#name, internal::makeMatcherAutoMarshall(                    \
                             ::clang::ast_matchers::name, #name));

/// \brief Generate a registry map with all the known matchers.
RegistryMaps::RegistryMaps() {
  // TODO: Here is the list of the missing matchers, grouped by reason.
  //
  // Need DynTypedNode fixes:
  // hasAnyTemplateArgument
  // hasTemplateArgument
  //
  // Need Variant/Parser fixes:
  // ofKind
  //
  // CXXCtorInitializer support:
  // hasAnyConstructorInitializer
  // forField
  // withInitializer
  // isWritten
  //
  // Type traversal:
  // hasElementType
  // hasValueType
  // hasDeducedType
  // innerType
  // pointee
  //
  // Function overloaded by args:
  // hasType
  // callee
  // hasPrefix
  // isDerivedFrom
  // isSameOrDerivedFrom
  // pointsTo
  // references
  // thisPointerType
  //
  // Polymorphic + argument overload:
  // unless
  // eachOf
  // anyOf
  // allOf
  // findAll
  //
  // Adaptative matcher (similar to polymorphic matcher):
  // has
  // forEach
  // forEachDescendant
  // hasDescendant
  // hasParent
  // hasAncestor
  //
  // Other:
  // loc
  // equals
  // equalsNode
  // hasDeclaration

  REGISTER_MATCHER(accessSpecDecl);
  REGISTER_MATCHER(alignOfExpr);
  REGISTER_MATCHER(anything);
  REGISTER_MATCHER(argumentCountIs);
  REGISTER_MATCHER(arraySubscriptExpr);
  REGISTER_MATCHER(arrayType);
  REGISTER_MATCHER(asString);
  REGISTER_MATCHER(asmStmt);
  REGISTER_MATCHER(atomicType);
  REGISTER_MATCHER(autoType);
  REGISTER_MATCHER(binaryOperator);
  REGISTER_MATCHER(bindTemporaryExpr);
  REGISTER_MATCHER(blockPointerType);
  REGISTER_MATCHER(boolLiteral);
  REGISTER_MATCHER(breakStmt);
  REGISTER_MATCHER(builtinType);
  REGISTER_MATCHER(cStyleCastExpr);
  REGISTER_MATCHER(callExpr);
  REGISTER_MATCHER(castExpr);
  REGISTER_MATCHER(catchStmt);
  REGISTER_MATCHER(characterLiteral);
  REGISTER_MATCHER(classTemplateDecl);
  REGISTER_MATCHER(classTemplateSpecializationDecl);
  REGISTER_MATCHER(complexType);
  REGISTER_MATCHER(compoundLiteralExpr);
  REGISTER_MATCHER(compoundStmt);
  REGISTER_MATCHER(conditionalOperator);
  REGISTER_MATCHER(constCastExpr);
  REGISTER_MATCHER(constantArrayType);
  REGISTER_MATCHER(constructExpr);
  REGISTER_MATCHER(constructorDecl);
  REGISTER_MATCHER(containsDeclaration);
  REGISTER_MATCHER(continueStmt);
  REGISTER_MATCHER(decl);
  REGISTER_MATCHER(declCountIs);
  REGISTER_MATCHER(declRefExpr);
  REGISTER_MATCHER(declStmt);
  REGISTER_MATCHER(defaultArgExpr);
  REGISTER_MATCHER(deleteExpr);
  REGISTER_MATCHER(dependentSizedArrayType);
  REGISTER_MATCHER(destructorDecl);
  REGISTER_MATCHER(doStmt);
  REGISTER_MATCHER(dynamicCastExpr);
  REGISTER_MATCHER(elaboratedType);
  REGISTER_MATCHER(enumConstantDecl);
  REGISTER_MATCHER(enumDecl);
  REGISTER_MATCHER(explicitCastExpr);
  REGISTER_MATCHER(expr);
  REGISTER_MATCHER(fieldDecl);
  REGISTER_MATCHER(forRangeStmt);
  REGISTER_MATCHER(forStmt);
  REGISTER_MATCHER(functionDecl);
  REGISTER_MATCHER(functionTemplateDecl);
  REGISTER_MATCHER(functionType);
  REGISTER_MATCHER(functionalCastExpr);
  REGISTER_MATCHER(gotoStmt);
  REGISTER_MATCHER(hasAnyArgument);
  REGISTER_MATCHER(hasAnyParameter);
  REGISTER_MATCHER(hasAnySubstatement);
  REGISTER_MATCHER(hasAnyUsingShadowDecl);
  REGISTER_MATCHER(hasArgument);
  REGISTER_MATCHER(hasArgumentOfType);
  REGISTER_MATCHER(hasBase);
  REGISTER_MATCHER(hasBody);
  REGISTER_MATCHER(hasCanonicalType);
  REGISTER_MATCHER(hasCondition);
  REGISTER_MATCHER(hasConditionVariableStatement);
  REGISTER_MATCHER(hasDeclContext);
  REGISTER_MATCHER(hasDestinationType);
  REGISTER_MATCHER(hasEitherOperand);
  REGISTER_MATCHER(hasFalseExpression);
  REGISTER_MATCHER(hasImplicitDestinationType);
  REGISTER_MATCHER(hasIncrement);
  REGISTER_MATCHER(hasIndex);
  REGISTER_MATCHER(hasInitializer);
  REGISTER_MATCHER(hasLHS);
  REGISTER_MATCHER(hasLocalQualifiers);
  REGISTER_MATCHER(hasLoopInit);
  REGISTER_MATCHER(hasMethod);
  REGISTER_MATCHER(hasName);
  REGISTER_MATCHER(hasObjectExpression);
  REGISTER_MATCHER(hasOperatorName);
  REGISTER_MATCHER(hasOverloadedOperatorName);
  REGISTER_MATCHER(hasParameter);
  REGISTER_MATCHER(hasQualifier);
  REGISTER_MATCHER(hasRHS);
  REGISTER_MATCHER(hasSingleDecl);
  REGISTER_MATCHER(hasSize);
  REGISTER_MATCHER(hasSizeExpr);
  REGISTER_MATCHER(hasSourceExpression);
  REGISTER_MATCHER(hasTargetDecl);
  REGISTER_MATCHER(hasTrueExpression);
  REGISTER_MATCHER(hasUnaryOperand);
  REGISTER_MATCHER(ifStmt);
  REGISTER_MATCHER(ignoringImpCasts);
  REGISTER_MATCHER(ignoringParenCasts);
  REGISTER_MATCHER(ignoringParenImpCasts);
  REGISTER_MATCHER(implicitCastExpr);
  REGISTER_MATCHER(incompleteArrayType);
  REGISTER_MATCHER(initListExpr);
  REGISTER_MATCHER(integerLiteral);
  REGISTER_MATCHER(isArrow);
  REGISTER_MATCHER(isConstQualified);
  REGISTER_MATCHER(isDefinition);
  REGISTER_MATCHER(isExplicitTemplateSpecialization);
  REGISTER_MATCHER(isExternC);
  REGISTER_MATCHER(isImplicit);
  REGISTER_MATCHER(isInteger);
  REGISTER_MATCHER(isOverride);
  REGISTER_MATCHER(isPrivate);
  REGISTER_MATCHER(isProtected);
  REGISTER_MATCHER(isPublic);
  REGISTER_MATCHER(isTemplateInstantiation);
  REGISTER_MATCHER(isVirtual);
  REGISTER_MATCHER(lValueReferenceType);
  REGISTER_MATCHER(labelStmt);
  REGISTER_MATCHER(lambdaExpr);
  REGISTER_MATCHER(matchesName);
  REGISTER_MATCHER(materializeTemporaryExpr);
  REGISTER_MATCHER(member);
  REGISTER_MATCHER(memberCallExpr);
  REGISTER_MATCHER(memberExpr);
  REGISTER_MATCHER(memberPointerType);
  REGISTER_MATCHER(methodDecl);
  REGISTER_MATCHER(namedDecl);
  REGISTER_MATCHER(namesType);
  REGISTER_MATCHER(namespaceDecl);
  REGISTER_MATCHER(nestedNameSpecifier);
  REGISTER_MATCHER(nestedNameSpecifierLoc);
  REGISTER_MATCHER(newExpr);
  REGISTER_MATCHER(nullPtrLiteralExpr);
  REGISTER_MATCHER(nullStmt);
  REGISTER_MATCHER(ofClass);
  REGISTER_MATCHER(on);
  REGISTER_MATCHER(onImplicitObjectArgument);
  REGISTER_MATCHER(operatorCallExpr);
  REGISTER_MATCHER(parameterCountIs);
  REGISTER_MATCHER(parenType);
  REGISTER_MATCHER(pointerType);
  REGISTER_MATCHER(qualType);
  REGISTER_MATCHER(rValueReferenceType);
  REGISTER_MATCHER(recordDecl);
  REGISTER_MATCHER(recordType);
  REGISTER_MATCHER(referenceType);
  REGISTER_MATCHER(refersToDeclaration);
  REGISTER_MATCHER(refersToType);
  REGISTER_MATCHER(reinterpretCastExpr);
  REGISTER_MATCHER(returnStmt);
  REGISTER_MATCHER(returns);
  REGISTER_MATCHER(sizeOfExpr);
  REGISTER_MATCHER(specifiesNamespace);
  REGISTER_MATCHER(specifiesType);
  REGISTER_MATCHER(specifiesTypeLoc);
  REGISTER_MATCHER(statementCountIs);
  REGISTER_MATCHER(staticCastExpr);
  REGISTER_MATCHER(stmt);
  REGISTER_MATCHER(stringLiteral);
  REGISTER_MATCHER(switchCase);
  REGISTER_MATCHER(switchStmt);
  REGISTER_MATCHER(templateSpecializationType);
  REGISTER_MATCHER(thisExpr);
  REGISTER_MATCHER(throughUsingDecl);
  REGISTER_MATCHER(throwExpr);
  REGISTER_MATCHER(to);
  REGISTER_MATCHER(tryStmt);
  REGISTER_MATCHER(type);
  REGISTER_MATCHER(typeLoc);
  REGISTER_MATCHER(typedefType);
  REGISTER_MATCHER(unaryExprOrTypeTraitExpr);
  REGISTER_MATCHER(unaryOperator);
  REGISTER_MATCHER(userDefinedLiteral);
  REGISTER_MATCHER(usingDecl);
  REGISTER_MATCHER(varDecl);
  REGISTER_MATCHER(variableArrayType);
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
MatcherList Registry::constructMatcher(StringRef MatcherName,
                                       const SourceRange &NameRange,
                                       ArrayRef<ParserValue> Args,
                                       Diagnostics *Error) {
  ConstructorMap::const_iterator it =
      RegistryData->constructors().find(MatcherName);
  if (it == RegistryData->constructors().end()) {
    Error->pushErrorFrame(NameRange, Error->ET_RegistryNotFound)
        << MatcherName;
    return MatcherList();
  }

  return it->second->run(NameRange, Args, Error);
}

// static
MatcherList Registry::constructBoundMatcher(StringRef MatcherName,
                                            const SourceRange &NameRange,
                                            StringRef BindID,
                                            ArrayRef<ParserValue> Args,
                                            Diagnostics *Error) {
  MatcherList Out = constructMatcher(MatcherName, NameRange, Args, Error);
  if (Out.empty()) return Out;

  ArrayRef<const DynTypedMatcher*> Matchers = Out.matchers();
  if (Matchers.size() == 1) {
    OwningPtr<DynTypedMatcher> Bound(Matchers[0]->tryBind(BindID));
    if (Bound) {
      return *Bound;
    }
  }
  Error->pushErrorFrame(NameRange, Error->ET_RegistryNotBindable);
  return MatcherList();
}

}  // namespace dynamic
}  // namespace ast_matchers
}  // namespace clang
