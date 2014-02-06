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
#include "Marshallers.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ManagedStatic.h"
#include <set>
#include <utility>

using namespace clang::ast_type_traits;

namespace clang {
namespace ast_matchers {
namespace dynamic {
namespace {

using internal::MatcherDescriptor;

typedef llvm::StringMap<const MatcherDescriptor *> ConstructorMap;
class RegistryMaps {
public:
  RegistryMaps();
  ~RegistryMaps();

  const ConstructorMap &constructors() const { return Constructors; }

private:
  void registerMatcher(StringRef MatcherName, MatcherDescriptor *Callback);
  ConstructorMap Constructors;
};

void RegistryMaps::registerMatcher(StringRef MatcherName,
                                   MatcherDescriptor *Callback) {
  assert(Constructors.find(MatcherName) == Constructors.end());
  Constructors[MatcherName] = Callback;
}

#define REGISTER_MATCHER(name)                                                 \
  registerMatcher(#name, internal::makeMatcherAutoMarshall(                    \
                             ::clang::ast_matchers::name, #name));

#define SPECIFIC_MATCHER_OVERLOAD(name, Id)                                    \
  static_cast< ::clang::ast_matchers::name##_Type##Id>(                        \
      ::clang::ast_matchers::name)

#define REGISTER_OVERLOADED_2(name)                                            \
  do {                                                                         \
    MatcherDescriptor *Callbacks[] = {                                         \
      internal::makeMatcherAutoMarshall(SPECIFIC_MATCHER_OVERLOAD(name, 0),    \
                                        #name),                                \
      internal::makeMatcherAutoMarshall(SPECIFIC_MATCHER_OVERLOAD(name, 1),    \
                                        #name)                                 \
    };                                                                         \
    registerMatcher(#name,                                                     \
                    new internal::OverloadedMatcherDescriptor(Callbacks));     \
  } while (0)

/// \brief Generate a registry map with all the known matchers.
RegistryMaps::RegistryMaps() {
  // TODO: Here is the list of the missing matchers, grouped by reason.
  //
  // Need Variant/Parser fixes:
  // ofKind
  //
  // Polymorphic + argument overload:
  // findAll
  //
  // Other:
  // loc
  // equals
  // equalsNode

  REGISTER_OVERLOADED_2(callee);
  REGISTER_OVERLOADED_2(hasPrefix);
  REGISTER_OVERLOADED_2(hasType);
  REGISTER_OVERLOADED_2(isDerivedFrom);
  REGISTER_OVERLOADED_2(isSameOrDerivedFrom);
  REGISTER_OVERLOADED_2(pointsTo);
  REGISTER_OVERLOADED_2(references);
  REGISTER_OVERLOADED_2(thisPointerType);

  REGISTER_MATCHER(accessSpecDecl);
  REGISTER_MATCHER(alignOfExpr);
  REGISTER_MATCHER(allOf);
  REGISTER_MATCHER(anyOf);
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
  REGISTER_MATCHER(caseStmt);
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
  REGISTER_MATCHER(ctorInitializer);
  REGISTER_MATCHER(decl);
  REGISTER_MATCHER(declCountIs);
  REGISTER_MATCHER(declRefExpr);
  REGISTER_MATCHER(declStmt);
  REGISTER_MATCHER(declaratorDecl);
  REGISTER_MATCHER(defaultArgExpr);
  REGISTER_MATCHER(defaultStmt);
  REGISTER_MATCHER(deleteExpr);
  REGISTER_MATCHER(dependentSizedArrayType);
  REGISTER_MATCHER(destructorDecl);
  REGISTER_MATCHER(doStmt);
  REGISTER_MATCHER(dynamicCastExpr);
  REGISTER_MATCHER(eachOf);
  REGISTER_MATCHER(elaboratedType);
  REGISTER_MATCHER(enumConstantDecl);
  REGISTER_MATCHER(enumDecl);
  REGISTER_MATCHER(equalsBoundNode);
  REGISTER_MATCHER(explicitCastExpr);
  REGISTER_MATCHER(expr);
  REGISTER_MATCHER(fieldDecl);
  REGISTER_MATCHER(floatLiteral);
  REGISTER_MATCHER(forEach);
  REGISTER_MATCHER(forEachConstructorInitializer);
  REGISTER_MATCHER(forEachDescendant);
  REGISTER_MATCHER(forEachSwitchCase);
  REGISTER_MATCHER(forField);
  REGISTER_MATCHER(forRangeStmt);
  REGISTER_MATCHER(forStmt);
  REGISTER_MATCHER(friendDecl);
  REGISTER_MATCHER(functionDecl);
  REGISTER_MATCHER(functionTemplateDecl);
  REGISTER_MATCHER(functionType);
  REGISTER_MATCHER(functionalCastExpr);
  REGISTER_MATCHER(gotoStmt);
  REGISTER_MATCHER(has);
  REGISTER_MATCHER(hasAncestor);
  REGISTER_MATCHER(hasAnyArgument);
  REGISTER_MATCHER(hasAnyConstructorInitializer);
  REGISTER_MATCHER(hasAnyParameter);
  REGISTER_MATCHER(hasAnySubstatement);
  REGISTER_MATCHER(hasAnyTemplateArgument);
  REGISTER_MATCHER(hasAnyUsingShadowDecl);
  REGISTER_MATCHER(hasArgument);
  REGISTER_MATCHER(hasArgumentOfType);
  REGISTER_MATCHER(hasBase);
  REGISTER_MATCHER(hasBody);
  REGISTER_MATCHER(hasCanonicalType);
  REGISTER_MATCHER(hasCaseConstant);
  REGISTER_MATCHER(hasCondition);
  REGISTER_MATCHER(hasConditionVariableStatement);
  REGISTER_MATCHER(hasDeclContext);
  REGISTER_MATCHER(hasDeclaration);
  REGISTER_MATCHER(hasDeducedType);
  REGISTER_MATCHER(hasDescendant);
  REGISTER_MATCHER(hasDestinationType);
  REGISTER_MATCHER(hasEitherOperand);
  REGISTER_MATCHER(hasElementType);
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
  REGISTER_MATCHER(hasParent);
  REGISTER_MATCHER(hasQualifier);
  REGISTER_MATCHER(hasRHS);
  REGISTER_MATCHER(hasSingleDecl);
  REGISTER_MATCHER(hasSize);
  REGISTER_MATCHER(hasSizeExpr);
  REGISTER_MATCHER(hasSourceExpression);
  REGISTER_MATCHER(hasTargetDecl);
  REGISTER_MATCHER(hasTemplateArgument);
  REGISTER_MATCHER(hasTrueExpression);
  REGISTER_MATCHER(hasTypeLoc);
  REGISTER_MATCHER(hasUnaryOperand);
  REGISTER_MATCHER(hasValueType);
  REGISTER_MATCHER(ifStmt);
  REGISTER_MATCHER(ignoringImpCasts);
  REGISTER_MATCHER(ignoringParenCasts);
  REGISTER_MATCHER(ignoringParenImpCasts);
  REGISTER_MATCHER(implicitCastExpr);
  REGISTER_MATCHER(incompleteArrayType);
  REGISTER_MATCHER(initListExpr);
  REGISTER_MATCHER(innerType);
  REGISTER_MATCHER(integerLiteral);
  REGISTER_MATCHER(isArrow);
  REGISTER_MATCHER(isConst);
  REGISTER_MATCHER(isConstQualified);
  REGISTER_MATCHER(isDefinition);
  REGISTER_MATCHER(isExplicitTemplateSpecialization);
  REGISTER_MATCHER(isExternC);
  REGISTER_MATCHER(isImplicit);
  REGISTER_MATCHER(isInteger);
  REGISTER_MATCHER(isListInitialization);
  REGISTER_MATCHER(isOverride);
  REGISTER_MATCHER(isPrivate);
  REGISTER_MATCHER(isProtected);
  REGISTER_MATCHER(isPublic);
  REGISTER_MATCHER(isTemplateInstantiation);
  REGISTER_MATCHER(isVirtual);
  REGISTER_MATCHER(isWritten);
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
  REGISTER_MATCHER(parmVarDecl);
  REGISTER_MATCHER(pointee);
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
  REGISTER_MATCHER(temporaryObjectExpr);
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
  REGISTER_MATCHER(unaryTransformType);
  REGISTER_MATCHER(unless);
  REGISTER_MATCHER(unresolvedConstructExpr);
  REGISTER_MATCHER(unresolvedUsingValueDecl);
  REGISTER_MATCHER(userDefinedLiteral);
  REGISTER_MATCHER(usingDecl);
  REGISTER_MATCHER(varDecl);
  REGISTER_MATCHER(variableArrayType);
  REGISTER_MATCHER(whileStmt);
  REGISTER_MATCHER(withInitializer);
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
llvm::Optional<MatcherCtor>
Registry::lookupMatcherCtor(StringRef MatcherName, const SourceRange &NameRange,
                            Diagnostics *Error) {
  ConstructorMap::const_iterator it =
      RegistryData->constructors().find(MatcherName);
  if (it == RegistryData->constructors().end()) {
    Error->addError(NameRange, Error->ET_RegistryNotFound) << MatcherName;
    return llvm::Optional<MatcherCtor>();
  }

  return it->second;
}

namespace {

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              const std::set<ASTNodeKind> &KS) {
  unsigned Count = 0;
  for (std::set<ASTNodeKind>::const_iterator I = KS.begin(), E = KS.end();
       I != E; ++I) {
    if (I != KS.begin())
      OS << "|";
    if (Count++ == 3) {
      OS << "...";
      break;
    }
    OS << *I;
  }
  return OS;
}

struct ReverseSpecificityThenName {
  bool operator()(const std::pair<unsigned, std::string> &A,
                  const std::pair<unsigned, std::string> &B) const {
    return A.first > B.first || (A.first == B.first && A.second < B.second);
  }
};

}

std::vector<MatcherCompletion> Registry::getCompletions(
    llvm::ArrayRef<std::pair<MatcherCtor, unsigned> > Context) {
  ASTNodeKind InitialTypes[] = {
    ASTNodeKind::getFromNodeKind<Decl>(),
    ASTNodeKind::getFromNodeKind<QualType>(),
    ASTNodeKind::getFromNodeKind<Type>(),
    ASTNodeKind::getFromNodeKind<Stmt>(),
    ASTNodeKind::getFromNodeKind<NestedNameSpecifier>(),
    ASTNodeKind::getFromNodeKind<NestedNameSpecifierLoc>(),
    ASTNodeKind::getFromNodeKind<TypeLoc>()
  };
  llvm::ArrayRef<ASTNodeKind> InitialTypesRef(InitialTypes);

  // Starting with the above seed of acceptable top-level matcher types, compute
  // the acceptable type set for the argument indicated by each context element.
  std::set<ASTNodeKind> TypeSet(InitialTypesRef.begin(), InitialTypesRef.end());
  for (llvm::ArrayRef<std::pair<MatcherCtor, unsigned> >::iterator
           CtxI = Context.begin(),
           CtxE = Context.end();
       CtxI != CtxE; ++CtxI) {
    std::vector<internal::ArgKind> NextTypeSet;
    for (std::set<ASTNodeKind>::iterator I = TypeSet.begin(), E = TypeSet.end();
         I != E; ++I) {
      if (CtxI->first->isConvertibleTo(*I) &&
          (CtxI->first->isVariadic() ||
           CtxI->second < CtxI->first->getNumArgs()))
        CtxI->first->getArgKinds(*I, CtxI->second, NextTypeSet);
    }
    TypeSet.clear();
    for (std::vector<internal::ArgKind>::iterator I = NextTypeSet.begin(),
                                                  E = NextTypeSet.end();
         I != E; ++I) {
      if (I->getArgKind() == internal::ArgKind::AK_Matcher)
        TypeSet.insert(I->getMatcherKind());
    }
  }

  typedef std::map<std::pair<unsigned, std::string>, MatcherCompletion,
                   ReverseSpecificityThenName> CompletionsTy;
  CompletionsTy Completions;

  // TypeSet now contains the list of acceptable types for the argument we are
  // completing.  Search the registry for acceptable matchers.
  for (ConstructorMap::const_iterator I = RegistryData->constructors().begin(),
                                      E = RegistryData->constructors().end();
       I != E; ++I) {
    std::set<ASTNodeKind> RetKinds;
    unsigned NumArgs = I->second->isVariadic() ? 1 : I->second->getNumArgs();
    bool IsPolymorphic = I->second->isPolymorphic();
    std::vector<std::vector<internal::ArgKind> > ArgsKinds(NumArgs);
    unsigned MaxSpecificity = 0;
    for (std::set<ASTNodeKind>::iterator TI = TypeSet.begin(),
                                         TE = TypeSet.end();
         TI != TE; ++TI) {
      unsigned Specificity;
      ASTNodeKind LeastDerivedKind;
      if (I->second->isConvertibleTo(*TI, &Specificity, &LeastDerivedKind)) {
        if (MaxSpecificity < Specificity)
          MaxSpecificity = Specificity;
        RetKinds.insert(LeastDerivedKind);
        for (unsigned Arg = 0; Arg != NumArgs; ++Arg)
          I->second->getArgKinds(*TI, Arg, ArgsKinds[Arg]);
        if (IsPolymorphic)
          break;
      }
    }

    if (!RetKinds.empty() && MaxSpecificity > 0) {
      std::string Decl;
      llvm::raw_string_ostream OS(Decl);

      if (IsPolymorphic) {
        OS << "Matcher<T> " << I->first() << "(Matcher<T>";
      } else {
        OS << "Matcher<" << RetKinds << "> " << I->first() << "(";
        for (std::vector<std::vector<internal::ArgKind> >::iterator
                 KI = ArgsKinds.begin(),
                 KE = ArgsKinds.end();
             KI != KE; ++KI) {
          if (KI != ArgsKinds.begin())
            OS << ", ";
          // This currently assumes that a matcher may not overload a
          // non-matcher, and all non-matcher overloads have identical
          // arguments.
          if ((*KI)[0].getArgKind() == internal::ArgKind::AK_Matcher) {
            std::set<ASTNodeKind> MatcherKinds;
            std::transform(
                KI->begin(), KI->end(),
                std::inserter(MatcherKinds, MatcherKinds.end()),
                std::mem_fun_ref(&internal::ArgKind::getMatcherKind));
            OS << "Matcher<" << MatcherKinds << ">";
          } else {
            OS << (*KI)[0].asString();
          }
        }
      }
      if (I->second->isVariadic())
        OS << "...";
      OS << ")";

      std::string TypedText = I->first();
      TypedText += "(";
      if (ArgsKinds.empty())
        TypedText += ")";
      else if (ArgsKinds[0][0].getArgKind() == internal::ArgKind::AK_String)
        TypedText += "\"";

      Completions[std::make_pair(MaxSpecificity, I->first())] =
          MatcherCompletion(TypedText, OS.str());
    }
  }

  std::vector<MatcherCompletion> RetVal;
  for (CompletionsTy::iterator I = Completions.begin(), E = Completions.end();
       I != E; ++I)
    RetVal.push_back(I->second);
  return RetVal;
}

// static
VariantMatcher Registry::constructMatcher(MatcherCtor Ctor,
                                          const SourceRange &NameRange,
                                          ArrayRef<ParserValue> Args,
                                          Diagnostics *Error) {
  return Ctor->create(NameRange, Args, Error);
}

// static
VariantMatcher Registry::constructBoundMatcher(MatcherCtor Ctor,
                                               const SourceRange &NameRange,
                                               StringRef BindID,
                                               ArrayRef<ParserValue> Args,
                                               Diagnostics *Error) {
  VariantMatcher Out = constructMatcher(Ctor, NameRange, Args, Error);
  if (Out.isNull()) return Out;

  llvm::Optional<DynTypedMatcher> Result = Out.getSingleMatcher();
  if (Result.hasValue()) {
    llvm::Optional<DynTypedMatcher> Bound = Result->tryBind(BindID);
    if (Bound.hasValue()) {
      return VariantMatcher::SingleMatcher(*Bound);
    }
  }
  Error->addError(NameRange, Error->ET_RegistryNotBindable);
  return VariantMatcher();
}

}  // namespace dynamic
}  // namespace ast_matchers
}  // namespace clang
