//===--- InlayHints.cpp ------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "InlayHints.h"
#include "Config.h"
#include "HeuristicResolver.h"
#include "ParsedAST.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceManager.h"

namespace clang {
namespace clangd {
namespace {

// For now, inlay hints are always anchored at the left or right of their range.
enum class HintSide { Left, Right };

class InlayHintVisitor : public RecursiveASTVisitor<InlayHintVisitor> {
public:
  InlayHintVisitor(std::vector<InlayHint> &Results, ParsedAST &AST,
                   const Config &Cfg, llvm::Optional<Range> RestrictRange)
      : Results(Results), AST(AST.getASTContext()), Cfg(Cfg),
        RestrictRange(std::move(RestrictRange)),
        MainFileID(AST.getSourceManager().getMainFileID()),
        Resolver(AST.getHeuristicResolver()),
        TypeHintPolicy(this->AST.getPrintingPolicy()),
        StructuredBindingPolicy(this->AST.getPrintingPolicy()) {
    bool Invalid = false;
    llvm::StringRef Buf =
        AST.getSourceManager().getBufferData(MainFileID, &Invalid);
    MainFileBuf = Invalid ? StringRef{} : Buf;

    TypeHintPolicy.SuppressScope = true; // keep type names short
    TypeHintPolicy.AnonymousTagLocations =
        false; // do not print lambda locations

    // For structured bindings, print canonical types. This is important because
    // for bindings that use the tuple_element protocol, the non-canonical types
    // would be "tuple_element<I, A>::type".
    // For "auto", we often prefer sugared types.
    // Not setting PrintCanonicalTypes for "auto" allows
    // SuppressDefaultTemplateArgs (set by default) to have an effect.
    StructuredBindingPolicy = TypeHintPolicy;
    StructuredBindingPolicy.PrintCanonicalTypes = true;
  }

  bool VisitCXXConstructExpr(CXXConstructExpr *E) {
    // Weed out constructor calls that don't look like a function call with
    // an argument list, by checking the validity of getParenOrBraceRange().
    // Also weed out std::initializer_list constructors as there are no names
    // for the individual arguments.
    if (!E->getParenOrBraceRange().isValid() ||
        E->isStdInitListInitialization()) {
      return true;
    }

    processCall(E->getParenOrBraceRange().getBegin(), E->getConstructor(),
                {E->getArgs(), E->getNumArgs()});
    return true;
  }

  bool VisitCallExpr(CallExpr *E) {
    if (!Cfg.InlayHints.Parameters)
      return true;

    // Do not show parameter hints for operator calls written using operator
    // syntax or user-defined literals. (Among other reasons, the resulting
    // hints can look awkard, e.g. the expression can itself be a function
    // argument and then we'd get two hints side by side).
    if (isa<CXXOperatorCallExpr>(E) || isa<UserDefinedLiteral>(E))
      return true;

    auto CalleeDecls = Resolver->resolveCalleeOfCallExpr(E);
    if (CalleeDecls.size() != 1)
      return true;
    const FunctionDecl *Callee = nullptr;
    if (const auto *FD = dyn_cast<FunctionDecl>(CalleeDecls[0]))
      Callee = FD;
    else if (const auto *FTD = dyn_cast<FunctionTemplateDecl>(CalleeDecls[0]))
      Callee = FTD->getTemplatedDecl();
    if (!Callee)
      return true;

    processCall(E->getRParenLoc(), Callee, {E->getArgs(), E->getNumArgs()});
    return true;
  }

  bool VisitFunctionDecl(FunctionDecl *D) {
    if (auto *AT = D->getReturnType()->getContainedAutoType()) {
      QualType Deduced = AT->getDeducedType();
      if (!Deduced.isNull()) {
        addTypeHint(D->getFunctionTypeLoc().getRParenLoc(), D->getReturnType(),
                    /*Prefix=*/"-> ");
      }
    }

    return true;
  }

  bool VisitVarDecl(VarDecl *D) {
    // Do not show hints for the aggregate in a structured binding,
    // but show hints for the individual bindings.
    if (auto *DD = dyn_cast<DecompositionDecl>(D)) {
      for (auto *Binding : DD->bindings()) {
        addTypeHint(Binding->getLocation(), Binding->getType(), /*Prefix=*/": ",
                    StructuredBindingPolicy);
      }
      return true;
    }

    if (D->getType()->getContainedAutoType()) {
      if (!D->getType()->isDependentType()) {
        // Our current approach is to place the hint on the variable
        // and accordingly print the full type
        // (e.g. for `const auto& x = 42`, print `const int&`).
        // Alternatively, we could place the hint on the `auto`
        // (and then just print the type deduced for the `auto`).
        addTypeHint(D->getLocation(), D->getType(), /*Prefix=*/": ");
      }
    }
    return true;
  }

  // FIXME: Handle RecoveryExpr to try to hint some invalid calls.

private:
  using NameVec = SmallVector<StringRef, 8>;

  // The purpose of Anchor is to deal with macros. It should be the call's
  // opening or closing parenthesis or brace. (Always using the opening would
  // make more sense but CallExpr only exposes the closing.) We heuristically
  // assume that if this location does not come from a macro definition, then
  // the entire argument list likely appears in the main file and can be hinted.
  void processCall(SourceLocation Anchor, const FunctionDecl *Callee,
                   llvm::ArrayRef<const Expr *const> Args) {
    if (!Cfg.InlayHints.Parameters || Args.size() == 0 || !Callee)
      return;

    // If the anchor location comes from a macro defintion, there's nowhere to
    // put hints.
    if (!AST.getSourceManager().getTopMacroCallerLoc(Anchor).isFileID())
      return;

    // The parameter name of a move or copy constructor is not very interesting.
    if (auto *Ctor = dyn_cast<CXXConstructorDecl>(Callee))
      if (Ctor->isCopyOrMoveConstructor())
        return;

    // Don't show hints for variadic parameters.
    size_t FixedParamCount = getFixedParamCount(Callee);
    size_t ArgCount = std::min(FixedParamCount, Args.size());

    NameVec ParameterNames = chooseParameterNames(Callee, ArgCount);

    // Exclude setters (i.e. functions with one argument whose name begins with
    // "set"), as their parameter name is also not likely to be interesting.
    if (isSetter(Callee, ParameterNames))
      return;

    for (size_t I = 0; I < ArgCount; ++I) {
      StringRef Name = ParameterNames[I];
      if (!shouldHint(Args[I], Name))
        continue;

      addInlayHint(Args[I]->getSourceRange(), HintSide::Left,
                   InlayHintKind::ParameterHint, /*Prefix=*/"", Name,
                   /*Suffix=*/": ");
    }
  }

  static bool isSetter(const FunctionDecl *Callee, const NameVec &ParamNames) {
    if (ParamNames.size() != 1)
      return false;

    StringRef Name = getSimpleName(*Callee);
    if (!Name.startswith_insensitive("set"))
      return false;

    // In addition to checking that the function has one parameter and its
    // name starts with "set", also check that the part after "set" matches
    // the name of the parameter (ignoring case). The idea here is that if
    // the parameter name differs, it may contain extra information that
    // may be useful to show in a hint, as in:
    //   void setTimeout(int timeoutMillis);
    // This currently doesn't handle cases where params use snake_case
    // and functions don't, e.g.
    //   void setExceptionHandler(EHFunc exception_handler);
    // We could improve this by replacing `equals_insensitive` with some
    // `sloppy_equals` which ignores case and also skips underscores.
    StringRef WhatItIsSetting = Name.substr(3).ltrim("_");
    return WhatItIsSetting.equals_insensitive(ParamNames[0]);
  }

  bool shouldHint(const Expr *Arg, StringRef ParamName) {
    if (ParamName.empty())
      return false;

    // If the argument expression is a single name and it matches the
    // parameter name exactly, omit the hint.
    if (ParamName == getSpelledIdentifier(Arg))
      return false;

    // Exclude argument expressions preceded by a /*paramName*/.
    if (isPrecededByParamNameComment(Arg, ParamName))
      return false;

    return true;
  }

  // Checks if "E" is spelled in the main file and preceded by a C-style comment
  // whose contents match ParamName (allowing for whitespace and an optional "="
  // at the end.
  bool isPrecededByParamNameComment(const Expr *E, StringRef ParamName) {
    auto &SM = AST.getSourceManager();
    auto ExprStartLoc = SM.getTopMacroCallerLoc(E->getBeginLoc());
    auto Decomposed = SM.getDecomposedLoc(ExprStartLoc);
    if (Decomposed.first != MainFileID)
      return false;

    StringRef SourcePrefix = MainFileBuf.substr(0, Decomposed.second);
    // Allow whitespace between comment and expression.
    SourcePrefix = SourcePrefix.rtrim();
    // Check for comment ending.
    if (!SourcePrefix.consume_back("*/"))
      return false;
    // Allow whitespace and "=" at end of comment.
    SourcePrefix = SourcePrefix.rtrim().rtrim('=').rtrim();
    // Other than that, the comment must contain exactly ParamName.
    if (!SourcePrefix.consume_back(ParamName))
      return false;
    return SourcePrefix.rtrim().endswith("/*");
  }

  // If "E" spells a single unqualified identifier, return that name.
  // Otherwise, return an empty string.
  static StringRef getSpelledIdentifier(const Expr *E) {
    E = E->IgnoreUnlessSpelledInSource();

    if (auto *DRE = dyn_cast<DeclRefExpr>(E))
      if (!DRE->getQualifier())
        return getSimpleName(*DRE->getDecl());

    if (auto *ME = dyn_cast<MemberExpr>(E))
      if (!ME->getQualifier() && ME->isImplicitAccess())
        return getSimpleName(*ME->getMemberDecl());

    return {};
  }

  NameVec chooseParameterNames(const FunctionDecl *Callee, size_t ArgCount) {
    // The current strategy here is to use all the parameter names from the
    // canonical declaration, unless they're all empty, in which case we
    // use all the parameter names from the definition (in present in the
    // translation unit).
    // We could try a bit harder, e.g.:
    //   - try all re-declarations, not just canonical + definition
    //   - fall back arg-by-arg rather than wholesale

    NameVec ParameterNames = getParameterNamesForDecl(Callee, ArgCount);

    if (llvm::all_of(ParameterNames, std::mem_fn(&StringRef::empty))) {
      if (const FunctionDecl *Def = Callee->getDefinition()) {
        ParameterNames = getParameterNamesForDecl(Def, ArgCount);
      }
    }
    assert(ParameterNames.size() == ArgCount);

    // Standard library functions often have parameter names that start
    // with underscores, which makes the hints noisy, so strip them out.
    for (auto &Name : ParameterNames)
      stripLeadingUnderscores(Name);

    return ParameterNames;
  }

  static void stripLeadingUnderscores(StringRef &Name) {
    Name = Name.ltrim('_');
  }

  // Return the number of fixed parameters Function has, that is, not counting
  // parameters that are variadic (instantiated from a parameter pack) or
  // C-style varargs.
  static size_t getFixedParamCount(const FunctionDecl *Function) {
    if (FunctionTemplateDecl *Template = Function->getPrimaryTemplate()) {
      FunctionDecl *F = Template->getTemplatedDecl();
      size_t Result = 0;
      for (ParmVarDecl *Parm : F->parameters()) {
        if (Parm->isParameterPack()) {
          break;
        }
        ++Result;
      }
      return Result;
    }
    // C-style varargs don't need special handling, they're already
    // not included in getNumParams().
    return Function->getNumParams();
  }

  static StringRef getSimpleName(const NamedDecl &D) {
    if (IdentifierInfo *Ident = D.getDeclName().getAsIdentifierInfo()) {
      return Ident->getName();
    }

    return StringRef();
  }

  NameVec getParameterNamesForDecl(const FunctionDecl *Function,
                                   size_t ArgCount) {
    NameVec Result;
    for (size_t I = 0; I < ArgCount; ++I) {
      const ParmVarDecl *Parm = Function->getParamDecl(I);
      assert(Parm);
      Result.emplace_back(getSimpleName(*Parm));
    }
    return Result;
  }

  // We pass HintSide rather than SourceLocation because we want to ensure 
  // it is in the same file as the common file range.
  void addInlayHint(SourceRange R, HintSide Side, InlayHintKind Kind,
                    llvm::StringRef Prefix, llvm::StringRef Label,
                    llvm::StringRef Suffix) {
    // We shouldn't get as far as adding a hint if the category is disabled.
    // We'd like to disable as much of the analysis as possible above instead.
    // Assert in debug mode but add a dynamic check in production.
    assert(Cfg.InlayHints.Enabled && "Shouldn't get here if disabled!");
    switch (Kind) {
#define CHECK_KIND(Enumerator, ConfigProperty)                                 \
  case InlayHintKind::Enumerator:                                              \
    assert(Cfg.InlayHints.ConfigProperty &&                                    \
           "Shouldn't get here if kind is disabled!");                         \
    if (!Cfg.InlayHints.ConfigProperty)                                        \
      return;                                                                  \
    break
      CHECK_KIND(ParameterHint, Parameters);
      CHECK_KIND(TypeHint, DeducedTypes);
#undef CHECK_KIND
    }

    auto FileRange =
        toHalfOpenFileRange(AST.getSourceManager(), AST.getLangOpts(), R);
    if (!FileRange)
      return;
    Range LSPRange{
        sourceLocToPosition(AST.getSourceManager(), FileRange->getBegin()),
        sourceLocToPosition(AST.getSourceManager(), FileRange->getEnd())};
    Position LSPPos = Side == HintSide::Left ? LSPRange.start : LSPRange.end;
    if (RestrictRange &&
        (LSPPos < RestrictRange->start || !(LSPPos < RestrictRange->end)))
      return;
    // The hint may be in a file other than the main file (for example, a header
    // file that was included after the preamble), do not show in that case.
    if (!AST.getSourceManager().isWrittenInMainFile(FileRange->getBegin()))
      return;
    Results.push_back(
        InlayHint{LSPPos, LSPRange, Kind, (Prefix + Label + Suffix).str()});
  }

  void addTypeHint(SourceRange R, QualType T, llvm::StringRef Prefix) {
    addTypeHint(R, T, Prefix, TypeHintPolicy);
  }

  void addTypeHint(SourceRange R, QualType T, llvm::StringRef Prefix,
                   const PrintingPolicy &Policy) {
    if (!Cfg.InlayHints.DeducedTypes || T.isNull())
      return;

    std::string TypeName = T.getAsString(Policy);
    if (TypeName.length() < TypeNameLimit)
      addInlayHint(R, HintSide::Right, InlayHintKind::TypeHint, Prefix,
                   TypeName, /*Suffix=*/"");
  }

  std::vector<InlayHint> &Results;
  ASTContext &AST;
  const Config &Cfg;
  llvm::Optional<Range> RestrictRange;
  FileID MainFileID;
  StringRef MainFileBuf;
  const HeuristicResolver *Resolver;
  // We want to suppress default template arguments, but otherwise print
  // canonical types. Unfortunately, they're conflicting policies so we can't
  // have both. For regular types, suppressing template arguments is more
  // important, whereas printing canonical types is crucial for structured
  // bindings, so we use two separate policies. (See the constructor where
  // the policies are initialized for more details.)
  PrintingPolicy TypeHintPolicy;
  PrintingPolicy StructuredBindingPolicy;

  static const size_t TypeNameLimit = 32;
};

} // namespace

std::vector<InlayHint> inlayHints(ParsedAST &AST,
                                  llvm::Optional<Range> RestrictRange) {
  std::vector<InlayHint> Results;
  const auto &Cfg = Config::current();
  if (!Cfg.InlayHints.Enabled)
    return Results;
  InlayHintVisitor Visitor(Results, AST, Cfg, std::move(RestrictRange));
  Visitor.TraverseAST(AST.getASTContext());

  // De-duplicate hints. Duplicates can sometimes occur due to e.g. explicit
  // template instantiations.
  llvm::sort(Results);
  Results.erase(std::unique(Results.begin(), Results.end()), Results.end());

  return Results;
}

} // namespace clangd
} // namespace clang
