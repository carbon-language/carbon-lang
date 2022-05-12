//===--- UseAutoCheck.cpp - clang-tidy-------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseAutoCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/CharInfo.h"
#include "clang/Tooling/FixIt.h"

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::ast_matchers::internal;

namespace clang {
namespace tidy {
namespace modernize {
namespace {

const char IteratorDeclStmtId[] = "iterator_decl";
const char DeclWithNewId[] = "decl_new";
const char DeclWithCastId[] = "decl_cast";
const char DeclWithTemplateCastId[] = "decl_template";

size_t getTypeNameLength(bool RemoveStars, StringRef Text) {
  enum CharType { Space, Alpha, Punctuation };
  CharType LastChar = Space, BeforeSpace = Punctuation;
  size_t NumChars = 0;
  int TemplateTypenameCntr = 0;
  for (const unsigned char C : Text) {
    if (C == '<')
      ++TemplateTypenameCntr;
    else if (C == '>')
      --TemplateTypenameCntr;
    const CharType NextChar =
        isAlphanumeric(C)
            ? Alpha
            : (isWhitespace(C) ||
               (!RemoveStars && TemplateTypenameCntr == 0 && C == '*'))
                  ? Space
                  : Punctuation;
    if (NextChar != Space) {
      ++NumChars; // Count the non-space character.
      if (LastChar == Space && NextChar == Alpha && BeforeSpace == Alpha)
        ++NumChars; // Count a single space character between two words.
      BeforeSpace = NextChar;
    }
    LastChar = NextChar;
  }
  return NumChars;
}

/// Matches variable declarations that have explicit initializers that
/// are not initializer lists.
///
/// Given
/// \code
///   iterator I = Container.begin();
///   MyType A(42);
///   MyType B{2};
///   MyType C;
/// \endcode
///
/// varDecl(hasWrittenNonListInitializer()) maches \c I and \c A but not \c B
/// or \c C.
AST_MATCHER(VarDecl, hasWrittenNonListInitializer) {
  const Expr *Init = Node.getAnyInitializer();
  if (!Init)
    return false;

  Init = Init->IgnoreImplicit();

  // The following test is based on DeclPrinter::VisitVarDecl() to find if an
  // initializer is implicit or not.
  if (const auto *Construct = dyn_cast<CXXConstructExpr>(Init)) {
    return !Construct->isListInitialization() && Construct->getNumArgs() > 0 &&
           !Construct->getArg(0)->isDefaultArgument();
  }
  return Node.getInitStyle() != VarDecl::ListInit;
}

/// Matches QualTypes that are type sugar for QualTypes that match \c
/// SugarMatcher.
///
/// Given
/// \code
///   class C {};
///   typedef C my_type;
///   typedef my_type my_other_type;
/// \endcode
///
/// qualType(isSugarFor(recordType(hasDeclaration(namedDecl(hasName("C"))))))
/// matches \c my_type and \c my_other_type.
AST_MATCHER_P(QualType, isSugarFor, Matcher<QualType>, SugarMatcher) {
  QualType QT = Node;
  while (true) {
    if (SugarMatcher.matches(QT, Finder, Builder))
      return true;

    QualType NewQT = QT.getSingleStepDesugaredType(Finder->getASTContext());
    if (NewQT == QT)
      return false;
    QT = NewQT;
  }
}

/// Matches named declarations that have one of the standard iterator
/// names: iterator, reverse_iterator, const_iterator, const_reverse_iterator.
///
/// Given
/// \code
///   iterator I;
///   const_iterator CI;
/// \endcode
///
/// namedDecl(hasStdIteratorName()) matches \c I and \c CI.
Matcher<NamedDecl> hasStdIteratorName() {
  static const StringRef IteratorNames[] = {"iterator", "reverse_iterator",
                                            "const_iterator",
                                            "const_reverse_iterator"};
  return hasAnyName(IteratorNames);
}

/// Matches named declarations that have one of the standard container
/// names.
///
/// Given
/// \code
///   class vector {};
///   class forward_list {};
///   class my_ver{};
/// \endcode
///
/// recordDecl(hasStdContainerName()) matches \c vector and \c forward_list
/// but not \c my_vec.
Matcher<NamedDecl> hasStdContainerName() {
  static StringRef ContainerNames[] = {"array",         "deque",
                                       "forward_list",  "list",
                                       "vector",

                                       "map",           "multimap",
                                       "set",           "multiset",

                                       "unordered_map", "unordered_multimap",
                                       "unordered_set", "unordered_multiset",

                                       "queue",         "priority_queue",
                                       "stack"};

  return hasAnyName(ContainerNames);
}

/// Matches declaration reference or member expressions with explicit template
/// arguments.
AST_POLYMORPHIC_MATCHER(hasExplicitTemplateArgs,
                        AST_POLYMORPHIC_SUPPORTED_TYPES(DeclRefExpr,
                                                        MemberExpr)) {
  return Node.hasExplicitTemplateArgs();
}

/// Returns a DeclarationMatcher that matches standard iterators nested
/// inside records with a standard container name.
DeclarationMatcher standardIterator() {
  return decl(
      namedDecl(hasStdIteratorName()),
      hasDeclContext(recordDecl(hasStdContainerName(), isInStdNamespace())));
}

/// Returns a TypeMatcher that matches typedefs for standard iterators
/// inside records with a standard container name.
TypeMatcher typedefIterator() {
  return typedefType(hasDeclaration(standardIterator()));
}

/// Returns a TypeMatcher that matches records named for standard
/// iterators nested inside records named for standard containers.
TypeMatcher nestedIterator() {
  return recordType(hasDeclaration(standardIterator()));
}

/// Returns a TypeMatcher that matches types declared with using
/// declarations and which name standard iterators for standard containers.
TypeMatcher iteratorFromUsingDeclaration() {
  auto HasIteratorDecl = hasDeclaration(namedDecl(hasStdIteratorName()));
  // Types resulting from using declarations are represented by elaboratedType.
  return elaboratedType(
      // Unwrap the nested name specifier to test for one of the standard
      // containers.
      hasQualifier(specifiesType(templateSpecializationType(hasDeclaration(
          namedDecl(hasStdContainerName(), isInStdNamespace()))))),
      // the named type is what comes after the final '::' in the type. It
      // should name one of the standard iterator names.
      namesType(
          anyOf(typedefType(HasIteratorDecl), recordType(HasIteratorDecl))));
}

/// This matcher returns declaration statements that contain variable
/// declarations with written non-list initializer for standard iterators.
StatementMatcher makeIteratorDeclMatcher() {
  return declStmt(unless(has(
                      varDecl(anyOf(unless(hasWrittenNonListInitializer()),
                                    unless(hasType(isSugarFor(anyOf(
                                        typedefIterator(), nestedIterator(),
                                        iteratorFromUsingDeclaration())))))))))
      .bind(IteratorDeclStmtId);
}

StatementMatcher makeDeclWithNewMatcher() {
  return declStmt(
             unless(has(varDecl(anyOf(
                 unless(hasInitializer(ignoringParenImpCasts(cxxNewExpr()))),
                 // FIXME: TypeLoc information is not reliable where CV
                 // qualifiers are concerned so these types can't be
                 // handled for now.
                 hasType(pointerType(
                     pointee(hasCanonicalType(hasLocalQualifiers())))),

                 // FIXME: Handle function pointers. For now we ignore them
                 // because the replacement replaces the entire type
                 // specifier source range which includes the identifier.
                 hasType(pointsTo(
                     pointsTo(parenType(innerType(functionType()))))))))))
      .bind(DeclWithNewId);
}

StatementMatcher makeDeclWithCastMatcher() {
  return declStmt(
             unless(has(varDecl(unless(hasInitializer(explicitCastExpr()))))))
      .bind(DeclWithCastId);
}

StatementMatcher makeDeclWithTemplateCastMatcher() {
  auto ST =
      substTemplateTypeParmType(hasReplacementType(equalsBoundNode("arg")));

  auto ExplicitCall =
      anyOf(has(memberExpr(hasExplicitTemplateArgs())),
            has(ignoringImpCasts(declRefExpr(hasExplicitTemplateArgs()))));

  auto TemplateArg =
      hasTemplateArgument(0, refersToType(qualType().bind("arg")));

  auto TemplateCall = callExpr(
      ExplicitCall,
      callee(functionDecl(TemplateArg,
                          returns(anyOf(ST, pointsTo(ST), references(ST))))));

  return declStmt(unless(has(varDecl(
                      unless(hasInitializer(ignoringImplicit(TemplateCall)))))))
      .bind(DeclWithTemplateCastId);
}

StatementMatcher makeCombinedMatcher() {
  return declStmt(
      // At least one varDecl should be a child of the declStmt to ensure
      // it's a declaration list and avoid matching other declarations,
      // e.g. using directives.
      has(varDecl(unless(isImplicit()))),
      // Skip declarations that are already using auto.
      unless(has(varDecl(anyOf(hasType(autoType()),
                               hasType(qualType(hasDescendant(autoType()))))))),
      anyOf(makeIteratorDeclMatcher(), makeDeclWithNewMatcher(),
            makeDeclWithCastMatcher(), makeDeclWithTemplateCastMatcher()));
}

} // namespace

UseAutoCheck::UseAutoCheck(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      MinTypeNameLength(Options.get("MinTypeNameLength", 5)),
      RemoveStars(Options.get("RemoveStars", false)) {}

void UseAutoCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "MinTypeNameLength", MinTypeNameLength);
  Options.store(Opts, "RemoveStars", RemoveStars);
}

void UseAutoCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(traverse(TK_AsIs, makeCombinedMatcher()), this);
}

void UseAutoCheck::replaceIterators(const DeclStmt *D, ASTContext *Context) {
  for (const auto *Dec : D->decls()) {
    const auto *V = cast<VarDecl>(Dec);
    const Expr *ExprInit = V->getInit();

    // Skip expressions with cleanups from the initializer expression.
    if (const auto *E = dyn_cast<ExprWithCleanups>(ExprInit))
      ExprInit = E->getSubExpr();

    const auto *Construct = dyn_cast<CXXConstructExpr>(ExprInit);
    if (!Construct)
      continue;

    // Ensure that the constructor receives a single argument.
    if (Construct->getNumArgs() != 1)
      return;

    // Drill down to the as-written initializer.
    const Expr *E = (*Construct->arg_begin())->IgnoreParenImpCasts();
    if (E != E->IgnoreConversionOperatorSingleStep()) {
      // We hit a conversion operator. Early-out now as they imply an implicit
      // conversion from a different type. Could also mean an explicit
      // conversion from the same type but that's pretty rare.
      return;
    }

    if (const auto *NestedConstruct = dyn_cast<CXXConstructExpr>(E)) {
      // If we ran into an implicit conversion constructor, can't convert.
      //
      // FIXME: The following only checks if the constructor can be used
      // implicitly, not if it actually was. Cases where the converting
      // constructor was used explicitly won't get converted.
      if (NestedConstruct->getConstructor()->isConvertingConstructor(false))
        return;
    }
    if (!Context->hasSameType(V->getType(), E->getType()))
      return;
  }

  // Get the type location using the first declaration.
  const auto *V = cast<VarDecl>(*D->decl_begin());

  // WARNING: TypeLoc::getSourceRange() will include the identifier for things
  // like function pointers. Not a concern since this action only works with
  // iterators but something to keep in mind in the future.

  SourceRange Range(V->getTypeSourceInfo()->getTypeLoc().getSourceRange());
  diag(Range.getBegin(), "use auto when declaring iterators")
      << FixItHint::CreateReplacement(Range, "auto");
}

void UseAutoCheck::replaceExpr(
    const DeclStmt *D, ASTContext *Context,
    llvm::function_ref<QualType(const Expr *)> GetType, StringRef Message) {
  const auto *FirstDecl = dyn_cast<VarDecl>(*D->decl_begin());
  // Ensure that there is at least one VarDecl within the DeclStmt.
  if (!FirstDecl)
    return;

  const QualType FirstDeclType = FirstDecl->getType().getCanonicalType();

  std::vector<FixItHint> StarRemovals;
  for (const auto *Dec : D->decls()) {
    const auto *V = cast<VarDecl>(Dec);
    // Ensure that every DeclStmt child is a VarDecl.
    if (!V)
      return;

    const auto *Expr = V->getInit()->IgnoreParenImpCasts();
    // Ensure that every VarDecl has an initializer.
    if (!Expr)
      return;

    // If VarDecl and Initializer have mismatching unqualified types.
    if (!Context->hasSameUnqualifiedType(V->getType(), GetType(Expr)))
      return;

    // All subsequent variables in this declaration should have the same
    // canonical type.  For example, we don't want to use `auto` in
    // `T *p = new T, **pp = new T*;`.
    if (FirstDeclType != V->getType().getCanonicalType())
      return;

    if (RemoveStars) {
      // Remove explicitly written '*' from declarations where there's more than
      // one declaration in the declaration list.
      if (Dec == *D->decl_begin())
        continue;

      auto Q = V->getTypeSourceInfo()->getTypeLoc().getAs<PointerTypeLoc>();
      while (!Q.isNull()) {
        StarRemovals.push_back(FixItHint::CreateRemoval(Q.getStarLoc()));
        Q = Q.getNextTypeLoc().getAs<PointerTypeLoc>();
      }
    }
  }

  // FIXME: There is, however, one case we can address: when the VarDecl pointee
  // is the same as the initializer, just more CV-qualified. However, TypeLoc
  // information is not reliable where CV qualifiers are concerned so we can't
  // do anything about this case for now.
  TypeLoc Loc = FirstDecl->getTypeSourceInfo()->getTypeLoc();
  if (!RemoveStars) {
    while (Loc.getTypeLocClass() == TypeLoc::Pointer ||
           Loc.getTypeLocClass() == TypeLoc::Qualified)
      Loc = Loc.getNextTypeLoc();
  }
  while (Loc.getTypeLocClass() == TypeLoc::LValueReference ||
         Loc.getTypeLocClass() == TypeLoc::RValueReference ||
         Loc.getTypeLocClass() == TypeLoc::Qualified) {
    Loc = Loc.getNextTypeLoc();
  }
  SourceRange Range(Loc.getSourceRange());

  if (MinTypeNameLength != 0 &&
      getTypeNameLength(RemoveStars,
                        tooling::fixit::getText(Loc.getSourceRange(),
                                                FirstDecl->getASTContext())) <
          MinTypeNameLength)
    return;

  auto Diag = diag(Range.getBegin(), Message);

  // Space after 'auto' to handle cases where the '*' in the pointer type is
  // next to the identifier. This avoids changing 'int *p' into 'autop'.
  // FIXME: This doesn't work for function pointers because the variable name
  // is inside the type.
  Diag << FixItHint::CreateReplacement(Range, RemoveStars ? "auto " : "auto")
       << StarRemovals;
}

void UseAutoCheck::check(const MatchFinder::MatchResult &Result) {
  if (const auto *Decl = Result.Nodes.getNodeAs<DeclStmt>(IteratorDeclStmtId)) {
    replaceIterators(Decl, Result.Context);
  } else if (const auto *Decl =
                 Result.Nodes.getNodeAs<DeclStmt>(DeclWithNewId)) {
    replaceExpr(Decl, Result.Context,
                [](const Expr *Expr) { return Expr->getType(); },
                "use auto when initializing with new to avoid "
                "duplicating the type name");
  } else if (const auto *Decl =
                 Result.Nodes.getNodeAs<DeclStmt>(DeclWithCastId)) {
    replaceExpr(
        Decl, Result.Context,
        [](const Expr *Expr) {
          return cast<ExplicitCastExpr>(Expr)->getTypeAsWritten();
        },
        "use auto when initializing with a cast to avoid duplicating the type "
        "name");
  } else if (const auto *Decl =
                 Result.Nodes.getNodeAs<DeclStmt>(DeclWithTemplateCastId)) {
    replaceExpr(
        Decl, Result.Context,
        [](const Expr *Expr) {
          return cast<CallExpr>(Expr->IgnoreImplicit())
              ->getDirectCallee()
              ->getReturnType();
        },
        "use auto when initializing with a template cast to avoid duplicating "
        "the type name");
  } else {
    llvm_unreachable("Bad Callback. No node provided.");
  }
}

} // namespace modernize
} // namespace tidy
} // namespace clang
