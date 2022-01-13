//===--- ArgumentCommentCheck.cpp - clang-tidy ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ArgumentCommentCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/Token.h"

#include "../utils/LexerUtils.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace bugprone {
namespace {
AST_MATCHER(Decl, isFromStdNamespaceOrSystemHeader) {
  if (const auto *D = Node.getDeclContext()->getEnclosingNamespaceContext())
    if (D->isStdNamespace())
      return true;
  return Node.getASTContext().getSourceManager().isInSystemHeader(
      Node.getLocation());
}
} // namespace

ArgumentCommentCheck::ArgumentCommentCheck(StringRef Name,
                                           ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      StrictMode(Options.getLocalOrGlobal("StrictMode", false)),
      IgnoreSingleArgument(Options.get("IgnoreSingleArgument", false)),
      CommentBoolLiterals(Options.get("CommentBoolLiterals", false)),
      CommentIntegerLiterals(Options.get("CommentIntegerLiterals", false)),
      CommentFloatLiterals(Options.get("CommentFloatLiterals", false)),
      CommentStringLiterals(Options.get("CommentStringLiterals", false)),
      CommentUserDefinedLiterals(
          Options.get("CommentUserDefinedLiterals", false)),
      CommentCharacterLiterals(Options.get("CommentCharacterLiterals", false)),
      CommentNullPtrs(Options.get("CommentNullPtrs", false)),
      IdentRE("^(/\\* *)([_A-Za-z][_A-Za-z0-9]*)( *= *\\*/)$") {}

void ArgumentCommentCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "StrictMode", StrictMode);
  Options.store(Opts, "IgnoreSingleArgument", IgnoreSingleArgument);
  Options.store(Opts, "CommentBoolLiterals", CommentBoolLiterals);
  Options.store(Opts, "CommentIntegerLiterals", CommentIntegerLiterals);
  Options.store(Opts, "CommentFloatLiterals", CommentFloatLiterals);
  Options.store(Opts, "CommentStringLiterals", CommentStringLiterals);
  Options.store(Opts, "CommentUserDefinedLiterals", CommentUserDefinedLiterals);
  Options.store(Opts, "CommentCharacterLiterals", CommentCharacterLiterals);
  Options.store(Opts, "CommentNullPtrs", CommentNullPtrs);
}

void ArgumentCommentCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      callExpr(unless(cxxOperatorCallExpr()),
               // NewCallback's arguments relate to the pointed function,
               // don't check them against NewCallback's parameter names.
               // FIXME: Make this configurable.
               unless(hasDeclaration(functionDecl(
                   hasAnyName("NewCallback", "NewPermanentCallback")))),
               // Ignore APIs from the standard library, since their names are
               // not specified by the standard, and standard library
               // implementations in practice have to use reserved names to
               // avoid conflicts with same-named macros.
               unless(hasDeclaration(isFromStdNamespaceOrSystemHeader())))
          .bind("expr"),
      this);
  Finder->addMatcher(cxxConstructExpr(unless(hasDeclaration(
                                          isFromStdNamespaceOrSystemHeader())))
                         .bind("expr"),
                     this);
}

static std::vector<std::pair<SourceLocation, StringRef>>
getCommentsInRange(ASTContext *Ctx, CharSourceRange Range) {
  std::vector<std::pair<SourceLocation, StringRef>> Comments;
  auto &SM = Ctx->getSourceManager();
  std::pair<FileID, unsigned> BeginLoc = SM.getDecomposedLoc(Range.getBegin()),
                              EndLoc = SM.getDecomposedLoc(Range.getEnd());

  if (BeginLoc.first != EndLoc.first)
    return Comments;

  bool Invalid = false;
  StringRef Buffer = SM.getBufferData(BeginLoc.first, &Invalid);
  if (Invalid)
    return Comments;

  const char *StrData = Buffer.data() + BeginLoc.second;

  Lexer TheLexer(SM.getLocForStartOfFile(BeginLoc.first), Ctx->getLangOpts(),
                 Buffer.begin(), StrData, Buffer.end());
  TheLexer.SetCommentRetentionState(true);

  while (true) {
    Token Tok;
    if (TheLexer.LexFromRawLexer(Tok))
      break;
    if (Tok.getLocation() == Range.getEnd() || Tok.is(tok::eof))
      break;

    if (Tok.is(tok::comment)) {
      std::pair<FileID, unsigned> CommentLoc =
          SM.getDecomposedLoc(Tok.getLocation());
      assert(CommentLoc.first == BeginLoc.first);
      Comments.emplace_back(
          Tok.getLocation(),
          StringRef(Buffer.begin() + CommentLoc.second, Tok.getLength()));
    } else {
      // Clear comments found before the different token, e.g. comma.
      Comments.clear();
    }
  }

  return Comments;
}

static std::vector<std::pair<SourceLocation, StringRef>>
getCommentsBeforeLoc(ASTContext *Ctx, SourceLocation Loc) {
  std::vector<std::pair<SourceLocation, StringRef>> Comments;
  while (Loc.isValid()) {
    clang::Token Tok = utils::lexer::getPreviousToken(
        Loc, Ctx->getSourceManager(), Ctx->getLangOpts(),
        /*SkipComments=*/false);
    if (Tok.isNot(tok::comment))
      break;
    Loc = Tok.getLocation();
    Comments.emplace_back(
        Loc,
        Lexer::getSourceText(CharSourceRange::getCharRange(
                                 Loc, Loc.getLocWithOffset(Tok.getLength())),
                             Ctx->getSourceManager(), Ctx->getLangOpts()));
  }
  return Comments;
}

static bool isLikelyTypo(llvm::ArrayRef<ParmVarDecl *> Params,
                         StringRef ArgName, unsigned ArgIndex) {
  std::string ArgNameLowerStr = ArgName.lower();
  StringRef ArgNameLower = ArgNameLowerStr;
  // The threshold is arbitrary.
  unsigned UpperBound = (ArgName.size() + 2) / 3 + 1;
  unsigned ThisED = ArgNameLower.edit_distance(
      Params[ArgIndex]->getIdentifier()->getName().lower(),
      /*AllowReplacements=*/true, UpperBound);
  if (ThisED >= UpperBound)
    return false;

  for (unsigned I = 0, E = Params.size(); I != E; ++I) {
    if (I == ArgIndex)
      continue;
    IdentifierInfo *II = Params[I]->getIdentifier();
    if (!II)
      continue;

    const unsigned Threshold = 2;
    // Other parameters must be an edit distance at least Threshold more away
    // from this parameter. This gives us greater confidence that this is a
    // typo of this parameter and not one with a similar name.
    unsigned OtherED = ArgNameLower.edit_distance(II->getName().lower(),
                                                  /*AllowReplacements=*/true,
                                                  ThisED + Threshold);
    if (OtherED < ThisED + Threshold)
      return false;
  }

  return true;
}

static bool sameName(StringRef InComment, StringRef InDecl, bool StrictMode) {
  if (StrictMode)
    return InComment == InDecl;
  InComment = InComment.trim('_');
  InDecl = InDecl.trim('_');
  // FIXME: compare_insensitive only works for ASCII.
  return InComment.compare_insensitive(InDecl) == 0;
}

static bool looksLikeExpectMethod(const CXXMethodDecl *Expect) {
  return Expect != nullptr && Expect->getLocation().isMacroID() &&
         Expect->getNameInfo().getName().isIdentifier() &&
         Expect->getName().startswith("gmock_");
}
static bool areMockAndExpectMethods(const CXXMethodDecl *Mock,
                                    const CXXMethodDecl *Expect) {
  assert(looksLikeExpectMethod(Expect));
  return Mock != nullptr && Mock->getNextDeclInContext() == Expect &&
         Mock->getNumParams() == Expect->getNumParams() &&
         Mock->getLocation().isMacroID() &&
         Mock->getNameInfo().getName().isIdentifier() &&
         Mock->getName() == Expect->getName().substr(strlen("gmock_"));
}

// This uses implementation details of MOCK_METHODx_ macros: for each mocked
// method M it defines M() with appropriate signature and a method used to set
// up expectations - gmock_M() - with each argument's type changed the
// corresponding matcher. This function returns M when given either M or
// gmock_M.
static const CXXMethodDecl *findMockedMethod(const CXXMethodDecl *Method) {
  if (looksLikeExpectMethod(Method)) {
    const DeclContext *Ctx = Method->getDeclContext();
    if (Ctx == nullptr || !Ctx->isRecord())
      return nullptr;
    for (const auto *D : Ctx->decls()) {
      if (D->getNextDeclInContext() == Method) {
        const auto *Previous = dyn_cast<CXXMethodDecl>(D);
        return areMockAndExpectMethods(Previous, Method) ? Previous : nullptr;
      }
    }
    return nullptr;
  }
  if (const auto *Next =
          dyn_cast_or_null<CXXMethodDecl>(Method->getNextDeclInContext())) {
    if (looksLikeExpectMethod(Next) && areMockAndExpectMethods(Method, Next))
      return Method;
  }
  return nullptr;
}

// For gmock expectation builder method (the target of the call generated by
// `EXPECT_CALL(obj, Method(...))`) tries to find the real method being mocked
// (returns nullptr, if the mock method doesn't override anything). For other
// functions returns the function itself.
static const FunctionDecl *resolveMocks(const FunctionDecl *Func) {
  if (const auto *Method = dyn_cast<CXXMethodDecl>(Func)) {
    if (const auto *MockedMethod = findMockedMethod(Method)) {
      // If mocked method overrides the real one, we can use its parameter
      // names, otherwise we're out of luck.
      if (MockedMethod->size_overridden_methods() > 0) {
        return *MockedMethod->begin_overridden_methods();
      }
      return nullptr;
    }
  }
  return Func;
}

// Given the argument type and the options determine if we should
// be adding an argument comment.
bool ArgumentCommentCheck::shouldAddComment(const Expr *Arg) const {
  Arg = Arg->IgnoreImpCasts();
  if (isa<UnaryOperator>(Arg))
    Arg = cast<UnaryOperator>(Arg)->getSubExpr();
  if (Arg->getExprLoc().isMacroID())
    return false;
  return (CommentBoolLiterals && isa<CXXBoolLiteralExpr>(Arg)) ||
         (CommentIntegerLiterals && isa<IntegerLiteral>(Arg)) ||
         (CommentFloatLiterals && isa<FloatingLiteral>(Arg)) ||
         (CommentUserDefinedLiterals && isa<UserDefinedLiteral>(Arg)) ||
         (CommentCharacterLiterals && isa<CharacterLiteral>(Arg)) ||
         (CommentStringLiterals && isa<StringLiteral>(Arg)) ||
         (CommentNullPtrs && isa<CXXNullPtrLiteralExpr>(Arg));
}

void ArgumentCommentCheck::checkCallArgs(ASTContext *Ctx,
                                         const FunctionDecl *OriginalCallee,
                                         SourceLocation ArgBeginLoc,
                                         llvm::ArrayRef<const Expr *> Args) {
  const FunctionDecl *Callee = resolveMocks(OriginalCallee);
  if (!Callee)
    return;

  Callee = Callee->getFirstDecl();
  unsigned NumArgs = std::min<unsigned>(Args.size(), Callee->getNumParams());
  if ((NumArgs == 0) || (IgnoreSingleArgument && NumArgs == 1))
    return;

  auto MakeFileCharRange = [Ctx](SourceLocation Begin, SourceLocation End) {
    return Lexer::makeFileCharRange(CharSourceRange::getCharRange(Begin, End),
                                    Ctx->getSourceManager(),
                                    Ctx->getLangOpts());
  };

  for (unsigned I = 0; I < NumArgs; ++I) {
    const ParmVarDecl *PVD = Callee->getParamDecl(I);
    IdentifierInfo *II = PVD->getIdentifier();
    if (!II)
      continue;
    if (FunctionDecl *Template = Callee->getTemplateInstantiationPattern()) {
      // Don't warn on arguments for parameters instantiated from template
      // parameter packs. If we find more arguments than the template
      // definition has, it also means that they correspond to a parameter
      // pack.
      if (Template->getNumParams() <= I ||
          Template->getParamDecl(I)->isParameterPack()) {
        continue;
      }
    }

    CharSourceRange BeforeArgument =
        MakeFileCharRange(ArgBeginLoc, Args[I]->getBeginLoc());
    ArgBeginLoc = Args[I]->getEndLoc();

    std::vector<std::pair<SourceLocation, StringRef>> Comments;
    if (BeforeArgument.isValid()) {
      Comments = getCommentsInRange(Ctx, BeforeArgument);
    } else {
      // Fall back to parsing back from the start of the argument.
      CharSourceRange ArgsRange =
          MakeFileCharRange(Args[I]->getBeginLoc(), Args[I]->getEndLoc());
      Comments = getCommentsBeforeLoc(Ctx, ArgsRange.getBegin());
    }

    for (auto Comment : Comments) {
      llvm::SmallVector<StringRef, 2> Matches;
      if (IdentRE.match(Comment.second, &Matches) &&
          !sameName(Matches[2], II->getName(), StrictMode)) {
        {
          DiagnosticBuilder Diag =
              diag(Comment.first, "argument name '%0' in comment does not "
                                  "match parameter name %1")
              << Matches[2] << II;
          if (isLikelyTypo(Callee->parameters(), Matches[2], I)) {
            Diag << FixItHint::CreateReplacement(
                Comment.first, (Matches[1] + II->getName() + Matches[3]).str());
          }
        }
        diag(PVD->getLocation(), "%0 declared here", DiagnosticIDs::Note) << II;
        if (OriginalCallee != Callee) {
          diag(OriginalCallee->getLocation(),
               "actual callee (%0) is declared here", DiagnosticIDs::Note)
              << OriginalCallee;
        }
      }
    }

    // If the argument comments are missing for literals add them.
    if (Comments.empty() && shouldAddComment(Args[I])) {
      std::string ArgComment =
          (llvm::Twine("/*") + II->getName() + "=*/").str();
      DiagnosticBuilder Diag =
          diag(Args[I]->getBeginLoc(),
               "argument comment missing for literal argument %0")
          << II
          << FixItHint::CreateInsertion(Args[I]->getBeginLoc(), ArgComment);
    }
  }
}

void ArgumentCommentCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *E = Result.Nodes.getNodeAs<Expr>("expr");
  if (const auto *Call = dyn_cast<CallExpr>(E)) {
    const FunctionDecl *Callee = Call->getDirectCallee();
    if (!Callee)
      return;

    checkCallArgs(Result.Context, Callee, Call->getCallee()->getEndLoc(),
                  llvm::makeArrayRef(Call->getArgs(), Call->getNumArgs()));
  } else {
    const auto *Construct = cast<CXXConstructExpr>(E);
    if (Construct->getNumArgs() > 0 &&
        Construct->getArg(0)->getSourceRange() == Construct->getSourceRange()) {
      // Ignore implicit construction.
      return;
    }
    checkCallArgs(
        Result.Context, Construct->getConstructor(),
        Construct->getParenOrBraceRange().getBegin(),
        llvm::makeArrayRef(Construct->getArgs(), Construct->getNumArgs()));
  }
}

} // namespace bugprone
} // namespace tidy
} // namespace clang
