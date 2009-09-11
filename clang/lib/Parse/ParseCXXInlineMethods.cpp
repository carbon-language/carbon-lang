//===--- ParseCXXInlineMethods.cpp - C++ class inline methods parsing------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements parsing for C++ class inline methods.
//
//===----------------------------------------------------------------------===//

#include "clang/Parse/ParseDiagnostic.h"
#include "clang/Parse/Parser.h"
#include "clang/Parse/DeclSpec.h"
#include "clang/Parse/Scope.h"
using namespace clang;

/// ParseCXXInlineMethodDef - We parsed and verified that the specified
/// Declarator is a well formed C++ inline method definition. Now lex its body
/// and store its tokens for parsing after the C++ class is complete.
Parser::DeclPtrTy
Parser::ParseCXXInlineMethodDef(AccessSpecifier AS, Declarator &D,
                                const ParsedTemplateInfo &TemplateInfo) {
  assert(D.getTypeObject(0).Kind == DeclaratorChunk::Function &&
         "This isn't a function declarator!");
  assert((Tok.is(tok::l_brace) || Tok.is(tok::colon) || Tok.is(tok::kw_try)) &&
         "Current token not a '{', ':' or 'try'!");

  Action::MultiTemplateParamsArg TemplateParams(Actions,
                                                TemplateInfo.TemplateParams? TemplateInfo.TemplateParams->data() : 0,
                                                TemplateInfo.TemplateParams? TemplateInfo.TemplateParams->size() : 0);
  DeclPtrTy FnD;
  if (D.getDeclSpec().isFriendSpecified())
    // FIXME: Friend templates
    FnD = Actions.ActOnFriendFunctionDecl(CurScope, D, true, move(TemplateParams));
  else // FIXME: pass template information through
    FnD = Actions.ActOnCXXMemberDeclarator(CurScope, AS, D,
                                           move(TemplateParams), 0, 0);

  HandleMemberFunctionDefaultArgs(D, FnD);

  // Consume the tokens and store them for later parsing.

  getCurrentClass().MethodDefs.push_back(LexedMethod(FnD));
  getCurrentClass().MethodDefs.back().TemplateScope
    = CurScope->isTemplateParamScope();
  CachedTokens &Toks = getCurrentClass().MethodDefs.back().Toks;

  tok::TokenKind kind = Tok.getKind();
  // We may have a constructor initializer or function-try-block here.
  if (kind == tok::colon || kind == tok::kw_try) {
    // Consume everything up to (and including) the left brace.
    if (!ConsumeAndStoreUntil(tok::l_brace, tok::unknown, Toks, tok::semi)) {
      // We didn't find the left-brace we expected after the
      // constructor initializer.
      if (Tok.is(tok::semi)) {
        // We found a semicolon; complain, consume the semicolon, and
        // don't try to parse this method later.
        Diag(Tok.getLocation(), diag::err_expected_lbrace);
        ConsumeAnyToken();
        getCurrentClass().MethodDefs.pop_back();
        return FnD;
      }
    }

  } else {
    // Begin by storing the '{' token.
    Toks.push_back(Tok);
    ConsumeBrace();
  }
  // Consume everything up to (and including) the matching right brace.
  ConsumeAndStoreUntil(tok::r_brace, tok::unknown, Toks);

  // If we're in a function-try-block, we need to store all the catch blocks.
  if (kind == tok::kw_try) {
    while (Tok.is(tok::kw_catch)) {
      ConsumeAndStoreUntil(tok::l_brace, tok::unknown, Toks);
      ConsumeAndStoreUntil(tok::r_brace, tok::unknown, Toks);
    }
  }

  return FnD;
}

/// ParseLexedMethodDeclarations - We finished parsing the member
/// specification of a top (non-nested) C++ class. Now go over the
/// stack of method declarations with some parts for which parsing was
/// delayed (such as default arguments) and parse them.
void Parser::ParseLexedMethodDeclarations(ParsingClass &Class) {
  bool HasTemplateScope = !Class.TopLevelClass && Class.TemplateScope;
  ParseScope TemplateScope(this, Scope::TemplateParamScope, HasTemplateScope);
  if (HasTemplateScope)
    Actions.ActOnReenterTemplateScope(CurScope, Class.TagOrTemplate);

  bool HasClassScope = !Class.TopLevelClass;
  ParseScope ClassScope(this, Scope::ClassScope|Scope::DeclScope,
                        HasClassScope);

  for (; !Class.MethodDecls.empty(); Class.MethodDecls.pop_front()) {
    LateParsedMethodDeclaration &LM = Class.MethodDecls.front();

    // If this is a member template, introduce the template parameter scope.
    ParseScope TemplateScope(this, Scope::TemplateParamScope, LM.TemplateScope);
    if (LM.TemplateScope)
      Actions.ActOnReenterTemplateScope(CurScope, LM.Method);

    // Start the delayed C++ method declaration
    Actions.ActOnStartDelayedCXXMethodDeclaration(CurScope, LM.Method);

    // Introduce the parameters into scope and parse their default
    // arguments.
    ParseScope PrototypeScope(this,
                              Scope::FunctionPrototypeScope|Scope::DeclScope);
    for (unsigned I = 0, N = LM.DefaultArgs.size(); I != N; ++I) {
      // Introduce the parameter into scope.
      Actions.ActOnDelayedCXXMethodParameter(CurScope, LM.DefaultArgs[I].Param);

      if (CachedTokens *Toks = LM.DefaultArgs[I].Toks) {
        // Parse the default argument from its saved token stream.
        Toks->push_back(Tok); // So that the current token doesn't get lost
        PP.EnterTokenStream(&Toks->front(), Toks->size(), true, false);

        // Consume the previously-pushed token.
        ConsumeAnyToken();

        // Consume the '='.
        assert(Tok.is(tok::equal) && "Default argument not starting with '='");
        SourceLocation EqualLoc = ConsumeToken();

        OwningExprResult DefArgResult(ParseAssignmentExpression());
        if (DefArgResult.isInvalid())
          Actions.ActOnParamDefaultArgumentError(LM.DefaultArgs[I].Param);
        else
          Actions.ActOnParamDefaultArgument(LM.DefaultArgs[I].Param, EqualLoc,
                                            move(DefArgResult));
        delete Toks;
        LM.DefaultArgs[I].Toks = 0;
      }
    }
    PrototypeScope.Exit();

    // Finish the delayed C++ method declaration.
    Actions.ActOnFinishDelayedCXXMethodDeclaration(CurScope, LM.Method);
  }

  for (unsigned I = 0, N = Class.NestedClasses.size(); I != N; ++I)
    ParseLexedMethodDeclarations(*Class.NestedClasses[I]);
}

/// ParseLexedMethodDefs - We finished parsing the member specification of a top
/// (non-nested) C++ class. Now go over the stack of lexed methods that were
/// collected during its parsing and parse them all.
void Parser::ParseLexedMethodDefs(ParsingClass &Class) {
  bool HasTemplateScope = !Class.TopLevelClass && Class.TemplateScope;
  ParseScope TemplateScope(this, Scope::TemplateParamScope, HasTemplateScope);
  if (HasTemplateScope)
    Actions.ActOnReenterTemplateScope(CurScope, Class.TagOrTemplate);

  bool HasClassScope = !Class.TopLevelClass;
  ParseScope ClassScope(this, Scope::ClassScope|Scope::DeclScope,
                        HasClassScope);

  for (; !Class.MethodDefs.empty(); Class.MethodDefs.pop_front()) {
    LexedMethod &LM = Class.MethodDefs.front();

    // If this is a member template, introduce the template parameter scope.
    ParseScope TemplateScope(this, Scope::TemplateParamScope, LM.TemplateScope);
    if (LM.TemplateScope)
      Actions.ActOnReenterTemplateScope(CurScope, LM.D);

    assert(!LM.Toks.empty() && "Empty body!");
    // Append the current token at the end of the new token stream so that it
    // doesn't get lost.
    LM.Toks.push_back(Tok);
    PP.EnterTokenStream(LM.Toks.data(), LM.Toks.size(), true, false);

    // Consume the previously pushed token.
    ConsumeAnyToken();
    assert((Tok.is(tok::l_brace) || Tok.is(tok::colon) || Tok.is(tok::kw_try))
           && "Inline method not starting with '{', ':' or 'try'");

    // Parse the method body. Function body parsing code is similar enough
    // to be re-used for method bodies as well.
    ParseScope FnScope(this, Scope::FnScope|Scope::DeclScope);
    Actions.ActOnStartOfFunctionDef(CurScope, LM.D);

    if (Tok.is(tok::kw_try)) {
      ParseFunctionTryBlock(LM.D);
      continue;
    }
    if (Tok.is(tok::colon))
      ParseConstructorInitializer(LM.D);
    else
      Actions.ActOnDefaultCtorInitializers(LM.D);

    // FIXME: What if ParseConstructorInitializer doesn't leave us with a '{'??
    ParseFunctionStatementBody(LM.D);
  }

  for (unsigned I = 0, N = Class.NestedClasses.size(); I != N; ++I)
    ParseLexedMethodDefs(*Class.NestedClasses[I]);
}

/// ConsumeAndStoreUntil - Consume and store the token at the passed token
/// container until the token 'T' is reached (which gets
/// consumed/stored too, if ConsumeFinalToken).
/// If EarlyAbortIf is specified, then we will stop early if we find that
/// token at the top level.
/// Returns true if token 'T1' or 'T2' was found.
/// NOTE: This is a specialized version of Parser::SkipUntil.
bool Parser::ConsumeAndStoreUntil(tok::TokenKind T1, tok::TokenKind T2,
                                  CachedTokens &Toks,
                                  tok::TokenKind EarlyAbortIf,
                                  bool ConsumeFinalToken) {
  // We always want this function to consume at least one token if the first
  // token isn't T and if not at EOF.
  bool isFirstTokenConsumed = true;
  while (1) {
    // If we found one of the tokens, stop and return true.
    if (Tok.is(T1) || Tok.is(T2)) {
      if (ConsumeFinalToken) {
        Toks.push_back(Tok);
        ConsumeAnyToken();
      }
      return true;
    }

    // If we found the early-abort token, return.
    if (Tok.is(EarlyAbortIf))
      return false;

    switch (Tok.getKind()) {
    case tok::eof:
      // Ran out of tokens.
      return false;

    case tok::l_paren:
      // Recursively consume properly-nested parens.
      Toks.push_back(Tok);
      ConsumeParen();
      ConsumeAndStoreUntil(tok::r_paren, tok::unknown, Toks);
      break;
    case tok::l_square:
      // Recursively consume properly-nested square brackets.
      Toks.push_back(Tok);
      ConsumeBracket();
      ConsumeAndStoreUntil(tok::r_square, tok::unknown, Toks);
      break;
    case tok::l_brace:
      // Recursively consume properly-nested braces.
      Toks.push_back(Tok);
      ConsumeBrace();
      ConsumeAndStoreUntil(tok::r_brace, tok::unknown, Toks);
      break;

    // Okay, we found a ']' or '}' or ')', which we think should be balanced.
    // Since the user wasn't looking for this token (if they were, it would
    // already be handled), this isn't balanced.  If there is a LHS token at a
    // higher level, we will assume that this matches the unbalanced token
    // and return it.  Otherwise, this is a spurious RHS token, which we skip.
    case tok::r_paren:
      if (ParenCount && !isFirstTokenConsumed)
        return false;  // Matches something.
      Toks.push_back(Tok);
      ConsumeParen();
      break;
    case tok::r_square:
      if (BracketCount && !isFirstTokenConsumed)
        return false;  // Matches something.
      Toks.push_back(Tok);
      ConsumeBracket();
      break;
    case tok::r_brace:
      if (BraceCount && !isFirstTokenConsumed)
        return false;  // Matches something.
      Toks.push_back(Tok);
      ConsumeBrace();
      break;

    case tok::string_literal:
    case tok::wide_string_literal:
      Toks.push_back(Tok);
      ConsumeStringToken();
      break;
    default:
      // consume this token.
      Toks.push_back(Tok);
      ConsumeToken();
      break;
    }
    isFirstTokenConsumed = false;
  }
}
