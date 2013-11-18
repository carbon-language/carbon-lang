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

#include "clang/Parse/Parser.h"
#include "RAIIObjectsForParser.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/Parse/ParseDiagnostic.h"
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/Scope.h"
using namespace clang;

/// Get the FunctionDecl for a function or function template decl.
static FunctionDecl *getFunctionDecl(Decl *D) {
  if (FunctionDecl *fn = dyn_cast<FunctionDecl>(D))
    return fn;
  return cast<FunctionTemplateDecl>(D)->getTemplatedDecl();
}

/// ParseCXXInlineMethodDef - We parsed and verified that the specified
/// Declarator is a well formed C++ inline method definition. Now lex its body
/// and store its tokens for parsing after the C++ class is complete.
NamedDecl *Parser::ParseCXXInlineMethodDef(AccessSpecifier AS,
                                      AttributeList *AccessAttrs,
                                      ParsingDeclarator &D,
                                      const ParsedTemplateInfo &TemplateInfo,
                                      const VirtSpecifiers& VS, 
                                      FunctionDefinitionKind DefinitionKind,
                                      ExprResult& Init) {
  assert(D.isFunctionDeclarator() && "This isn't a function declarator!");
  assert((Tok.is(tok::l_brace) || Tok.is(tok::colon) || Tok.is(tok::kw_try) ||
          Tok.is(tok::equal)) &&
         "Current token not a '{', ':', '=', or 'try'!");

  MultiTemplateParamsArg TemplateParams(
          TemplateInfo.TemplateParams ? TemplateInfo.TemplateParams->data() : 0,
          TemplateInfo.TemplateParams ? TemplateInfo.TemplateParams->size() : 0);

  NamedDecl *FnD;
  D.setFunctionDefinitionKind(DefinitionKind);
  if (D.getDeclSpec().isFriendSpecified())
    FnD = Actions.ActOnFriendFunctionDecl(getCurScope(), D,
                                          TemplateParams);
  else {
    FnD = Actions.ActOnCXXMemberDeclarator(getCurScope(), AS, D,
                                           TemplateParams, 0,
                                           VS, ICIS_NoInit);
    if (FnD) {
      Actions.ProcessDeclAttributeList(getCurScope(), FnD, AccessAttrs);
      bool TypeSpecContainsAuto = D.getDeclSpec().containsPlaceholderType();
      if (Init.isUsable())
        Actions.AddInitializerToDecl(FnD, Init.get(), false,
                                     TypeSpecContainsAuto);
      else
        Actions.ActOnUninitializedDecl(FnD, TypeSpecContainsAuto);
    }
  }

  HandleMemberFunctionDeclDelays(D, FnD);

  D.complete(FnD);

  if (Tok.is(tok::equal)) {
    ConsumeToken();

    if (!FnD) {
      SkipUntil(tok::semi);
      return 0;
    }

    bool Delete = false;
    SourceLocation KWLoc;
    if (Tok.is(tok::kw_delete)) {
      Diag(Tok, getLangOpts().CPlusPlus11 ?
           diag::warn_cxx98_compat_deleted_function :
           diag::ext_deleted_function);

      KWLoc = ConsumeToken();
      Actions.SetDeclDeleted(FnD, KWLoc);
      Delete = true;
    } else if (Tok.is(tok::kw_default)) {
      Diag(Tok, getLangOpts().CPlusPlus11 ?
           diag::warn_cxx98_compat_defaulted_function :
           diag::ext_defaulted_function);

      KWLoc = ConsumeToken();
      Actions.SetDeclDefaulted(FnD, KWLoc);
    } else {
      llvm_unreachable("function definition after = not 'delete' or 'default'");
    }

    if (Tok.is(tok::comma)) {
      Diag(KWLoc, diag::err_default_delete_in_multiple_declaration)
        << Delete;
      SkipUntil(tok::semi);
    } else {
      ExpectAndConsume(tok::semi, diag::err_expected_semi_after,
                       Delete ? "delete" : "default", tok::semi);
    }

    return FnD;
  }
  
  // In delayed template parsing mode, if we are within a class template
  // or if we are about to parse function member template then consume
  // the tokens and store them for parsing at the end of the translation unit.
  if (getLangOpts().DelayedTemplateParsing &&
      DefinitionKind == FDK_Definition &&
      !D.getDeclSpec().isConstexprSpecified() && 
      !(FnD && getFunctionDecl(FnD) && 
          getFunctionDecl(FnD)->getResultType()->getContainedAutoType()) &&
      ((Actions.CurContext->isDependentContext() ||
        (TemplateInfo.Kind != ParsedTemplateInfo::NonTemplate &&
         TemplateInfo.Kind != ParsedTemplateInfo::ExplicitSpecialization)) &&
       !Actions.IsInsideALocalClassWithinATemplateFunction())) {

    CachedTokens Toks;
    LexTemplateFunctionForLateParsing(Toks);

    if (FnD) {
      FunctionDecl *FD = getFunctionDecl(FnD);
      Actions.CheckForFunctionRedefinition(FD);
      Actions.MarkAsLateParsedTemplate(FD, FnD, Toks);
    }

    return FnD;
  }

  // Consume the tokens and store them for later parsing.

  LexedMethod* LM = new LexedMethod(this, FnD);
  getCurrentClass().LateParsedDeclarations.push_back(LM);
  LM->TemplateScope = getCurScope()->isTemplateParamScope();
  CachedTokens &Toks = LM->Toks;

  tok::TokenKind kind = Tok.getKind();
  // Consume everything up to (and including) the left brace of the
  // function body.
  if (ConsumeAndStoreFunctionPrologue(Toks)) {
    // We didn't find the left-brace we expected after the
    // constructor initializer; we already printed an error, and it's likely
    // impossible to recover, so don't try to parse this method later.
    // Skip over the rest of the decl and back to somewhere that looks
    // reasonable.
    SkipMalformedDecl();
    delete getCurrentClass().LateParsedDeclarations.back();
    getCurrentClass().LateParsedDeclarations.pop_back();
    return FnD;
  } else {
    // Consume everything up to (and including) the matching right brace.
    ConsumeAndStoreUntil(tok::r_brace, Toks, /*StopAtSemi=*/false);
  }

  // If we're in a function-try-block, we need to store all the catch blocks.
  if (kind == tok::kw_try) {
    while (Tok.is(tok::kw_catch)) {
      ConsumeAndStoreUntil(tok::l_brace, Toks, /*StopAtSemi=*/false);
      ConsumeAndStoreUntil(tok::r_brace, Toks, /*StopAtSemi=*/false);
    }
  }

  if (FnD) {
    // If this is a friend function, mark that it's late-parsed so that
    // it's still known to be a definition even before we attach the
    // parsed body.  Sema needs to treat friend function definitions
    // differently during template instantiation, and it's possible for
    // the containing class to be instantiated before all its member
    // function definitions are parsed.
    //
    // If you remove this, you can remove the code that clears the flag
    // after parsing the member.
    if (D.getDeclSpec().isFriendSpecified()) {
      FunctionDecl *FD = getFunctionDecl(FnD);
      Actions.CheckForFunctionRedefinition(FD);
      FD->setLateTemplateParsed(true);
    }
  } else {
    // If semantic analysis could not build a function declaration,
    // just throw away the late-parsed declaration.
    delete getCurrentClass().LateParsedDeclarations.back();
    getCurrentClass().LateParsedDeclarations.pop_back();
  }

  return FnD;
}

/// ParseCXXNonStaticMemberInitializer - We parsed and verified that the
/// specified Declarator is a well formed C++ non-static data member
/// declaration. Now lex its initializer and store its tokens for parsing
/// after the class is complete.
void Parser::ParseCXXNonStaticMemberInitializer(Decl *VarD) {
  assert((Tok.is(tok::l_brace) || Tok.is(tok::equal)) &&
         "Current token not a '{' or '='!");

  LateParsedMemberInitializer *MI =
    new LateParsedMemberInitializer(this, VarD);
  getCurrentClass().LateParsedDeclarations.push_back(MI);
  CachedTokens &Toks = MI->Toks;

  tok::TokenKind kind = Tok.getKind();
  if (kind == tok::equal) {
    Toks.push_back(Tok);
    ConsumeToken();
  }

  if (kind == tok::l_brace) {
    // Begin by storing the '{' token.
    Toks.push_back(Tok);
    ConsumeBrace();

    // Consume everything up to (and including) the matching right brace.
    ConsumeAndStoreUntil(tok::r_brace, Toks, /*StopAtSemi=*/true);
  } else {
    // Consume everything up to (but excluding) the comma or semicolon.
    ConsumeAndStoreInitializer(Toks, CIK_DefaultInitializer);
  }

  // Store an artificial EOF token to ensure that we don't run off the end of
  // the initializer when we come to parse it.
  Token Eof;
  Eof.startToken();
  Eof.setKind(tok::eof);
  Eof.setLocation(Tok.getLocation());
  Toks.push_back(Eof);
}

Parser::LateParsedDeclaration::~LateParsedDeclaration() {}
void Parser::LateParsedDeclaration::ParseLexedMethodDeclarations() {}
void Parser::LateParsedDeclaration::ParseLexedMemberInitializers() {}
void Parser::LateParsedDeclaration::ParseLexedMethodDefs() {}

Parser::LateParsedClass::LateParsedClass(Parser *P, ParsingClass *C)
  : Self(P), Class(C) {}

Parser::LateParsedClass::~LateParsedClass() {
  Self->DeallocateParsedClasses(Class);
}

void Parser::LateParsedClass::ParseLexedMethodDeclarations() {
  Self->ParseLexedMethodDeclarations(*Class);
}

void Parser::LateParsedClass::ParseLexedMemberInitializers() {
  Self->ParseLexedMemberInitializers(*Class);
}

void Parser::LateParsedClass::ParseLexedMethodDefs() {
  Self->ParseLexedMethodDefs(*Class);
}

void Parser::LateParsedMethodDeclaration::ParseLexedMethodDeclarations() {
  Self->ParseLexedMethodDeclaration(*this);
}

void Parser::LexedMethod::ParseLexedMethodDefs() {
  Self->ParseLexedMethodDef(*this);
}

void Parser::LateParsedMemberInitializer::ParseLexedMemberInitializers() {
  Self->ParseLexedMemberInitializer(*this);
}

/// ParseLexedMethodDeclarations - We finished parsing the member
/// specification of a top (non-nested) C++ class. Now go over the
/// stack of method declarations with some parts for which parsing was
/// delayed (such as default arguments) and parse them.
void Parser::ParseLexedMethodDeclarations(ParsingClass &Class) {
  bool HasTemplateScope = !Class.TopLevelClass && Class.TemplateScope;
  ParseScope ClassTemplateScope(this, Scope::TemplateParamScope, HasTemplateScope);
  TemplateParameterDepthRAII CurTemplateDepthTracker(TemplateParameterDepth);
  if (HasTemplateScope) {
    Actions.ActOnReenterTemplateScope(getCurScope(), Class.TagOrTemplate);
    ++CurTemplateDepthTracker;
  }

  // The current scope is still active if we're the top-level class.
  // Otherwise we'll need to push and enter a new scope.
  bool HasClassScope = !Class.TopLevelClass;
  ParseScope ClassScope(this, Scope::ClassScope|Scope::DeclScope,
                        HasClassScope);
  if (HasClassScope)
    Actions.ActOnStartDelayedMemberDeclarations(getCurScope(), Class.TagOrTemplate);

  for (size_t i = 0; i < Class.LateParsedDeclarations.size(); ++i) {
    Class.LateParsedDeclarations[i]->ParseLexedMethodDeclarations();
  }

  if (HasClassScope)
    Actions.ActOnFinishDelayedMemberDeclarations(getCurScope(), Class.TagOrTemplate);
}

void Parser::ParseLexedMethodDeclaration(LateParsedMethodDeclaration &LM) {
  // If this is a member template, introduce the template parameter scope.
  ParseScope TemplateScope(this, Scope::TemplateParamScope, LM.TemplateScope);
  TemplateParameterDepthRAII CurTemplateDepthTracker(TemplateParameterDepth);
  if (LM.TemplateScope) {
    Actions.ActOnReenterTemplateScope(getCurScope(), LM.Method);
    ++CurTemplateDepthTracker;
  }
  // Start the delayed C++ method declaration
  Actions.ActOnStartDelayedCXXMethodDeclaration(getCurScope(), LM.Method);

  // Introduce the parameters into scope and parse their default
  // arguments.
  ParseScope PrototypeScope(this, Scope::FunctionPrototypeScope |
                            Scope::FunctionDeclarationScope | Scope::DeclScope);
  for (unsigned I = 0, N = LM.DefaultArgs.size(); I != N; ++I) {
    // Introduce the parameter into scope.
    Actions.ActOnDelayedCXXMethodParameter(getCurScope(), 
                                           LM.DefaultArgs[I].Param);

    if (CachedTokens *Toks = LM.DefaultArgs[I].Toks) {
      // Save the current token position.
      SourceLocation origLoc = Tok.getLocation();

      // Parse the default argument from its saved token stream.
      Toks->push_back(Tok); // So that the current token doesn't get lost
      PP.EnterTokenStream(&Toks->front(), Toks->size(), true, false);

      // Consume the previously-pushed token.
      ConsumeAnyToken();

      // Consume the '='.
      assert(Tok.is(tok::equal) && "Default argument not starting with '='");
      SourceLocation EqualLoc = ConsumeToken();

      // The argument isn't actually potentially evaluated unless it is
      // used.
      EnterExpressionEvaluationContext Eval(Actions,
                                            Sema::PotentiallyEvaluatedIfUsed,
                                            LM.DefaultArgs[I].Param);

      ExprResult DefArgResult;
      if (getLangOpts().CPlusPlus11 && Tok.is(tok::l_brace)) {
        Diag(Tok, diag::warn_cxx98_compat_generalized_initializer_lists);
        DefArgResult = ParseBraceInitializer();
      } else
        DefArgResult = ParseAssignmentExpression();
      if (DefArgResult.isInvalid())
        Actions.ActOnParamDefaultArgumentError(LM.DefaultArgs[I].Param);
      else {
        if (Tok.is(tok::cxx_defaultarg_end))
          ConsumeToken();
        else {
          // The last two tokens are the terminator and the saved value of
          // Tok; the last token in the default argument is the one before
          // those.
          assert(Toks->size() >= 3 && "expected a token in default arg");
          Diag(Tok.getLocation(), diag::err_default_arg_unparsed)
            << SourceRange(Tok.getLocation(),
                           (*Toks)[Toks->size() - 3].getLocation());
        }
        Actions.ActOnParamDefaultArgument(LM.DefaultArgs[I].Param, EqualLoc,
                                          DefArgResult.take());
      }

      assert(!PP.getSourceManager().isBeforeInTranslationUnit(origLoc,
                                                         Tok.getLocation()) &&
             "ParseAssignmentExpression went over the default arg tokens!");
      // There could be leftover tokens (e.g. because of an error).
      // Skip through until we reach the original token position.
      while (Tok.getLocation() != origLoc && Tok.isNot(tok::eof))
        ConsumeAnyToken();

      delete Toks;
      LM.DefaultArgs[I].Toks = 0;
    }
  }

  PrototypeScope.Exit();

  // Finish the delayed C++ method declaration.
  Actions.ActOnFinishDelayedCXXMethodDeclaration(getCurScope(), LM.Method);
}

/// ParseLexedMethodDefs - We finished parsing the member specification of a top
/// (non-nested) C++ class. Now go over the stack of lexed methods that were
/// collected during its parsing and parse them all.
void Parser::ParseLexedMethodDefs(ParsingClass &Class) {
  bool HasTemplateScope = !Class.TopLevelClass && Class.TemplateScope;
  ParseScope ClassTemplateScope(this, Scope::TemplateParamScope, HasTemplateScope);
  TemplateParameterDepthRAII CurTemplateDepthTracker(TemplateParameterDepth);
  if (HasTemplateScope) {
    Actions.ActOnReenterTemplateScope(getCurScope(), Class.TagOrTemplate);
    ++CurTemplateDepthTracker;
  }
  bool HasClassScope = !Class.TopLevelClass;
  ParseScope ClassScope(this, Scope::ClassScope|Scope::DeclScope,
                        HasClassScope);

  for (size_t i = 0; i < Class.LateParsedDeclarations.size(); ++i) {
    Class.LateParsedDeclarations[i]->ParseLexedMethodDefs();
  }
}

void Parser::ParseLexedMethodDef(LexedMethod &LM) {
  // If this is a member template, introduce the template parameter scope.
  ParseScope TemplateScope(this, Scope::TemplateParamScope, LM.TemplateScope);
  TemplateParameterDepthRAII CurTemplateDepthTracker(TemplateParameterDepth);
  if (LM.TemplateScope) {
    Actions.ActOnReenterTemplateScope(getCurScope(), LM.D);
    ++CurTemplateDepthTracker;
  }
  // Save the current token position.
  SourceLocation origLoc = Tok.getLocation();

  assert(!LM.Toks.empty() && "Empty body!");
  // Append the current token at the end of the new token stream so that it
  // doesn't get lost.
  LM.Toks.push_back(Tok);
  PP.EnterTokenStream(LM.Toks.data(), LM.Toks.size(), true, false);

  // Consume the previously pushed token.
  ConsumeAnyToken(/*ConsumeCodeCompletionTok=*/true);
  assert((Tok.is(tok::l_brace) || Tok.is(tok::colon) || Tok.is(tok::kw_try))
         && "Inline method not starting with '{', ':' or 'try'");

  // Parse the method body. Function body parsing code is similar enough
  // to be re-used for method bodies as well.
  ParseScope FnScope(this, Scope::FnScope|Scope::DeclScope);
  Actions.ActOnStartOfFunctionDef(getCurScope(), LM.D);

  if (Tok.is(tok::kw_try)) {
    ParseFunctionTryBlock(LM.D, FnScope);
    assert(!PP.getSourceManager().isBeforeInTranslationUnit(origLoc,
                                                         Tok.getLocation()) &&
           "ParseFunctionTryBlock went over the cached tokens!");
    // There could be leftover tokens (e.g. because of an error).
    // Skip through until we reach the original token position.
    while (Tok.getLocation() != origLoc && Tok.isNot(tok::eof))
      ConsumeAnyToken();
    return;
  }
  if (Tok.is(tok::colon)) {
    ParseConstructorInitializer(LM.D);

    // Error recovery.
    if (!Tok.is(tok::l_brace)) {
      FnScope.Exit();
      Actions.ActOnFinishFunctionBody(LM.D, 0);
      while (Tok.getLocation() != origLoc && Tok.isNot(tok::eof))
        ConsumeAnyToken();
      return;
    }
  } else
    Actions.ActOnDefaultCtorInitializers(LM.D);

  assert((Actions.getDiagnostics().hasErrorOccurred() ||
          !isa<FunctionTemplateDecl>(LM.D) ||
          cast<FunctionTemplateDecl>(LM.D)->getTemplateParameters()->getDepth()
            < TemplateParameterDepth) &&
         "TemplateParameterDepth should be greater than the depth of "
         "current template being instantiated!");

  ParseFunctionStatementBody(LM.D, FnScope);

  // Clear the late-template-parsed bit if we set it before.
  if (LM.D) getFunctionDecl(LM.D)->setLateTemplateParsed(false);

  if (Tok.getLocation() != origLoc) {
    // Due to parsing error, we either went over the cached tokens or
    // there are still cached tokens left. If it's the latter case skip the
    // leftover tokens.
    // Since this is an uncommon situation that should be avoided, use the
    // expensive isBeforeInTranslationUnit call.
    if (PP.getSourceManager().isBeforeInTranslationUnit(Tok.getLocation(),
                                                        origLoc))
      while (Tok.getLocation() != origLoc && Tok.isNot(tok::eof))
        ConsumeAnyToken();
  }
}

/// ParseLexedMemberInitializers - We finished parsing the member specification
/// of a top (non-nested) C++ class. Now go over the stack of lexed data member
/// initializers that were collected during its parsing and parse them all.
void Parser::ParseLexedMemberInitializers(ParsingClass &Class) {
  bool HasTemplateScope = !Class.TopLevelClass && Class.TemplateScope;
  ParseScope ClassTemplateScope(this, Scope::TemplateParamScope,
                                HasTemplateScope);
  TemplateParameterDepthRAII CurTemplateDepthTracker(TemplateParameterDepth);
  if (HasTemplateScope) {
    Actions.ActOnReenterTemplateScope(getCurScope(), Class.TagOrTemplate);
    ++CurTemplateDepthTracker;
  }
  // Set or update the scope flags.
  bool AlreadyHasClassScope = Class.TopLevelClass;
  unsigned ScopeFlags = Scope::ClassScope|Scope::DeclScope;
  ParseScope ClassScope(this, ScopeFlags, !AlreadyHasClassScope);
  ParseScopeFlags ClassScopeFlags(this, ScopeFlags, AlreadyHasClassScope);

  if (!AlreadyHasClassScope)
    Actions.ActOnStartDelayedMemberDeclarations(getCurScope(),
                                                Class.TagOrTemplate);

  if (!Class.LateParsedDeclarations.empty()) {
    // C++11 [expr.prim.general]p4:
    //   Otherwise, if a member-declarator declares a non-static data member 
    //  (9.2) of a class X, the expression this is a prvalue of type "pointer
    //  to X" within the optional brace-or-equal-initializer. It shall not 
    //  appear elsewhere in the member-declarator.
    Sema::CXXThisScopeRAII ThisScope(Actions, Class.TagOrTemplate,
                                     /*TypeQuals=*/(unsigned)0);

    for (size_t i = 0; i < Class.LateParsedDeclarations.size(); ++i) {
      Class.LateParsedDeclarations[i]->ParseLexedMemberInitializers();
    }
  }
  
  if (!AlreadyHasClassScope)
    Actions.ActOnFinishDelayedMemberDeclarations(getCurScope(),
                                                 Class.TagOrTemplate);

  Actions.ActOnFinishDelayedMemberInitializers(Class.TagOrTemplate);
}

void Parser::ParseLexedMemberInitializer(LateParsedMemberInitializer &MI) {
  if (!MI.Field || MI.Field->isInvalidDecl())
    return;

  // Append the current token at the end of the new token stream so that it
  // doesn't get lost.
  MI.Toks.push_back(Tok);
  PP.EnterTokenStream(MI.Toks.data(), MI.Toks.size(), true, false);

  // Consume the previously pushed token.
  ConsumeAnyToken(/*ConsumeCodeCompletionTok=*/true);

  SourceLocation EqualLoc;

  ExprResult Init = ParseCXXMemberInitializer(MI.Field, /*IsFunction=*/false, 
                                              EqualLoc);

  Actions.ActOnCXXInClassMemberInitializer(MI.Field, EqualLoc, Init.release());

  // The next token should be our artificial terminating EOF token.
  if (Tok.isNot(tok::eof)) {
    SourceLocation EndLoc = PP.getLocForEndOfToken(PrevTokLocation);
    if (!EndLoc.isValid())
      EndLoc = Tok.getLocation();
    // No fixit; we can't recover as if there were a semicolon here.
    Diag(EndLoc, diag::err_expected_semi_decl_list);

    // Consume tokens until we hit the artificial EOF.
    while (Tok.isNot(tok::eof))
      ConsumeAnyToken();
  }
  ConsumeAnyToken();
}

/// ConsumeAndStoreUntil - Consume and store the token at the passed token
/// container until the token 'T' is reached (which gets
/// consumed/stored too, if ConsumeFinalToken).
/// If StopAtSemi is true, then we will stop early at a ';' character.
/// Returns true if token 'T1' or 'T2' was found.
/// NOTE: This is a specialized version of Parser::SkipUntil.
bool Parser::ConsumeAndStoreUntil(tok::TokenKind T1, tok::TokenKind T2,
                                  CachedTokens &Toks,
                                  bool StopAtSemi, bool ConsumeFinalToken) {
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

    switch (Tok.getKind()) {
    case tok::eof:
      // Ran out of tokens.
      return false;

    case tok::l_paren:
      // Recursively consume properly-nested parens.
      Toks.push_back(Tok);
      ConsumeParen();
      ConsumeAndStoreUntil(tok::r_paren, Toks, /*StopAtSemi=*/false);
      break;
    case tok::l_square:
      // Recursively consume properly-nested square brackets.
      Toks.push_back(Tok);
      ConsumeBracket();
      ConsumeAndStoreUntil(tok::r_square, Toks, /*StopAtSemi=*/false);
      break;
    case tok::l_brace:
      // Recursively consume properly-nested braces.
      Toks.push_back(Tok);
      ConsumeBrace();
      ConsumeAndStoreUntil(tok::r_brace, Toks, /*StopAtSemi=*/false);
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

    case tok::code_completion:
      Toks.push_back(Tok);
      ConsumeCodeCompletionToken();
      break;

    case tok::string_literal:
    case tok::wide_string_literal:
    case tok::utf8_string_literal:
    case tok::utf16_string_literal:
    case tok::utf32_string_literal:
      Toks.push_back(Tok);
      ConsumeStringToken();
      break;
    case tok::semi:
      if (StopAtSemi)
        return false;
      // FALL THROUGH.
    default:
      // consume this token.
      Toks.push_back(Tok);
      ConsumeToken();
      break;
    }
    isFirstTokenConsumed = false;
  }
}

/// \brief Consume tokens and store them in the passed token container until
/// we've passed the try keyword and constructor initializers and have consumed
/// the opening brace of the function body. The opening brace will be consumed
/// if and only if there was no error.
///
/// \return True on error.
bool Parser::ConsumeAndStoreFunctionPrologue(CachedTokens &Toks) {
  if (Tok.is(tok::kw_try)) {
    Toks.push_back(Tok);
    ConsumeToken();
  }

  if (Tok.isNot(tok::colon)) {
    // Easy case, just a function body.

    // Grab any remaining garbage to be diagnosed later. We stop when we reach a
    // brace: an opening one is the function body, while a closing one probably
    // means we've reached the end of the class.
    ConsumeAndStoreUntil(tok::l_brace, tok::r_brace, Toks,
                         /*StopAtSemi=*/true,
                         /*ConsumeFinalToken=*/false);
    if (Tok.isNot(tok::l_brace))
      return Diag(Tok.getLocation(), diag::err_expected_lbrace);

    Toks.push_back(Tok);
    ConsumeBrace();
    return false;
  }

  Toks.push_back(Tok);
  ConsumeToken();

  // We can't reliably skip over a mem-initializer-id, because it could be
  // a template-id involving not-yet-declared names. Given:
  //
  //   S ( ) : a < b < c > ( e )
  //
  // 'e' might be an initializer or part of a template argument, depending
  // on whether 'b' is a template.

  // Track whether we might be inside a template argument. We can give
  // significantly better diagnostics if we know that we're not.
  bool MightBeTemplateArgument = false;

  while (true) {
    // Skip over the mem-initializer-id, if possible.
    if (Tok.is(tok::kw_decltype)) {
      Toks.push_back(Tok);
      SourceLocation OpenLoc = ConsumeToken();
      if (Tok.isNot(tok::l_paren))
        return Diag(Tok.getLocation(), diag::err_expected_lparen_after)
                 << "decltype";
      Toks.push_back(Tok);
      ConsumeParen();
      if (!ConsumeAndStoreUntil(tok::r_paren, Toks, /*StopAtSemi=*/true)) {
        Diag(Tok.getLocation(), diag::err_expected_rparen);
        Diag(OpenLoc, diag::note_matching) << "(";
        return true;
      }
    }
    do {
      // Walk over a component of a nested-name-specifier.
      if (Tok.is(tok::coloncolon)) {
        Toks.push_back(Tok);
        ConsumeToken();

        if (Tok.is(tok::kw_template)) {
          Toks.push_back(Tok);
          ConsumeToken();
        }
      }

      if (Tok.is(tok::identifier) || Tok.is(tok::kw_template)) {
        Toks.push_back(Tok);
        ConsumeToken();
      } else if (Tok.is(tok::code_completion)) {
        Toks.push_back(Tok);
        ConsumeCodeCompletionToken();
        // Consume the rest of the initializers permissively.
        // FIXME: We should be able to perform code-completion here even if
        //        there isn't a subsequent '{' token.
        MightBeTemplateArgument = true;
        break;
      } else {
        break;
      }
    } while (Tok.is(tok::coloncolon));

    if (Tok.is(tok::less))
      MightBeTemplateArgument = true;

    if (MightBeTemplateArgument) {
      // We may be inside a template argument list. Grab up to the start of the
      // next parenthesized initializer or braced-init-list. This *might* be the
      // initializer, or it might be a subexpression in the template argument
      // list.
      // FIXME: Count angle brackets, and clear MightBeTemplateArgument
      //        if all angles are closed.
      if (!ConsumeAndStoreUntil(tok::l_paren, tok::l_brace, Toks,
                                /*StopAtSemi=*/true,
                                /*ConsumeFinalToken=*/false)) {
        // We're not just missing the initializer, we're also missing the
        // function body!
        return Diag(Tok.getLocation(), diag::err_expected_lbrace);
      }
    } else if (Tok.isNot(tok::l_paren) && Tok.isNot(tok::l_brace)) {
      // We found something weird in a mem-initializer-id.
      return Diag(Tok.getLocation(), getLangOpts().CPlusPlus11
                                         ? diag::err_expected_lparen_or_lbrace
                                         : diag::err_expected_lparen);
    }

    tok::TokenKind kind = Tok.getKind();
    Toks.push_back(Tok);
    bool IsLParen = (kind == tok::l_paren);
    SourceLocation OpenLoc = Tok.getLocation();

    if (IsLParen) {
      ConsumeParen();
    } else {
      assert(kind == tok::l_brace && "Must be left paren or brace here.");
      ConsumeBrace();
      // In C++03, this has to be the start of the function body, which
      // means the initializer is malformed; we'll diagnose it later.
      if (!getLangOpts().CPlusPlus11)
        return false;
    }

    // Grab the initializer (or the subexpression of the template argument).
    // FIXME: If we support lambdas here, we'll need to set StopAtSemi to false
    //        if we might be inside the braces of a lambda-expression.
    if (!ConsumeAndStoreUntil(IsLParen ? tok::r_paren : tok::r_brace,
                              Toks, /*StopAtSemi=*/true)) {
      Diag(Tok, IsLParen ? diag::err_expected_rparen :
                           diag::err_expected_rbrace);
      Diag(OpenLoc, diag::note_matching) << (IsLParen ? "(" : "{");
      return true;
    }

    // Grab pack ellipsis, if present.
    if (Tok.is(tok::ellipsis)) {
      Toks.push_back(Tok);
      ConsumeToken();
    }

    // If we know we just consumed a mem-initializer, we must have ',' or '{'
    // next.
    if (Tok.is(tok::comma)) {
      Toks.push_back(Tok);
      ConsumeToken();
    } else if (Tok.is(tok::l_brace)) {
      // This is the function body if the ')' or '}' is immediately followed by
      // a '{'. That cannot happen within a template argument, apart from the
      // case where a template argument contains a compound literal:
      //
      //   S ( ) : a < b < c > ( d ) { }
      //   // End of declaration, or still inside the template argument?
      //
      // ... and the case where the template argument contains a lambda:
      //
      //   S ( ) : a < 0 && b < c > ( d ) + [ ] ( ) { return 0; }
      //     ( ) > ( ) { }
      //
      // FIXME: Disambiguate these cases. Note that the latter case is probably
      //        going to be made ill-formed by core issue 1607.
      Toks.push_back(Tok);
      ConsumeBrace();
      return false;
    } else if (!MightBeTemplateArgument) {
      return Diag(Tok.getLocation(), diag::err_expected_lbrace_or_comma);
    }
  }
}

/// \brief Consume and store tokens from the '?' to the ':' in a conditional
/// expression.
bool Parser::ConsumeAndStoreConditional(CachedTokens &Toks) {
  // Consume '?'.
  assert(Tok.is(tok::question));
  Toks.push_back(Tok);
  ConsumeToken();

  while (Tok.isNot(tok::colon)) {
    if (!ConsumeAndStoreUntil(tok::question, tok::colon, Toks, /*StopAtSemi*/true,
                              /*ConsumeFinalToken*/false))
      return false;

    // If we found a nested conditional, consume it.
    if (Tok.is(tok::question) && !ConsumeAndStoreConditional(Toks))
      return false;
  }

  // Consume ':'.
  Toks.push_back(Tok);
  ConsumeToken();
  return true;
}

/// \brief A tentative parsing action that can also revert token annotations.
class Parser::UnannotatedTentativeParsingAction : public TentativeParsingAction {
public:
  explicit UnannotatedTentativeParsingAction(Parser &Self,
                                             tok::TokenKind EndKind)
      : TentativeParsingAction(Self), Self(Self), EndKind(EndKind) {
    // Stash away the old token stream, so we can restore it once the
    // tentative parse is complete.
    TentativeParsingAction Inner(Self);
    Self.ConsumeAndStoreUntil(EndKind, Toks, true, /*ConsumeFinalToken*/false);
    Inner.Revert();
  }

  void RevertAnnotations() {
    Revert();

    // Put back the original tokens.
    Self.SkipUntil(EndKind, StopAtSemi | StopBeforeMatch);
    if (Toks.size()) {
      Token *Buffer = new Token[Toks.size()];
      std::copy(Toks.begin() + 1, Toks.end(), Buffer);
      Buffer[Toks.size() - 1] = Self.Tok;
      Self.PP.EnterTokenStream(Buffer, Toks.size(), true, /*Owned*/true);

      Self.Tok = Toks.front();
    }
  }

private:
  Parser &Self;
  CachedTokens Toks;
  tok::TokenKind EndKind;
};

/// ConsumeAndStoreInitializer - Consume and store the token at the passed token
/// container until the end of the current initializer expression (either a
/// default argument or an in-class initializer for a non-static data member).
/// The final token is not consumed.
bool Parser::ConsumeAndStoreInitializer(CachedTokens &Toks,
                                        CachedInitKind CIK) {
  // We always want this function to consume at least one token if not at EOF.
  bool IsFirstTokenConsumed = true;

  // Number of possible unclosed <s we've seen so far. These might be templates,
  // and might not, but if there were none of them (or we know for sure that
  // we're within a template), we can avoid a tentative parse.
  unsigned AngleCount = 0;
  unsigned KnownTemplateCount = 0;

  while (1) {
    switch (Tok.getKind()) {
    case tok::comma:
      // If we might be in a template, perform a tentative parse to check.
      if (!AngleCount)
        // Not a template argument: this is the end of the initializer.
        return true;
      if (KnownTemplateCount)
        goto consume_token;

      // We hit a comma inside angle brackets. This is the hard case. The
      // rule we follow is:
      //  * For a default argument, if the tokens after the comma form a
      //    syntactically-valid parameter-declaration-clause, in which each
      //    parameter has an initializer, then this comma ends the default
      //    argument.
      //  * For a default initializer, if the tokens after the comma form a
      //    syntactically-valid init-declarator-list, then this comma ends
      //    the default initializer.
      {
        UnannotatedTentativeParsingAction PA(*this,
                                             CIK == CIK_DefaultInitializer
                                               ? tok::semi : tok::r_paren);
        Sema::TentativeAnalysisScope Scope(Actions);

        TPResult Result = TPResult::Error();
        ConsumeToken();
        switch (CIK) {
        case CIK_DefaultInitializer:
          Result = TryParseInitDeclaratorList();
          // If we parsed a complete, ambiguous init-declarator-list, this
          // is only syntactically-valid if it's followed by a semicolon.
          if (Result == TPResult::Ambiguous() && Tok.isNot(tok::semi))
            Result = TPResult::False();
          break;

        case CIK_DefaultArgument:
          bool InvalidAsDeclaration = false;
          Result = TryParseParameterDeclarationClause(
              &InvalidAsDeclaration, /*VersusTemplateArgument*/true);
          // If this is an expression or a declaration with a missing
          // 'typename', assume it's not a declaration.
          if (Result == TPResult::Ambiguous() && InvalidAsDeclaration)
            Result = TPResult::False();
          break;
        }

        // If what follows could be a declaration, it is a declaration.
        if (Result != TPResult::False() && Result != TPResult::Error()) {
          PA.Revert();
          return true;
        }

        // In the uncommon case that we decide the following tokens are part
        // of a template argument, revert any annotations we've performed in
        // those tokens. We're not going to look them up until we've parsed
        // the rest of the class, and that might add more declarations.
        PA.RevertAnnotations();
      }

      // Keep going. We know we're inside a template argument list now.
      ++KnownTemplateCount;
      goto consume_token;

    case tok::eof:
      // Ran out of tokens.
      return false;

    case tok::less:
      // FIXME: A '<' can only start a template-id if it's preceded by an
      // identifier, an operator-function-id, or a literal-operator-id.
      ++AngleCount;
      goto consume_token;

    case tok::question:
      // In 'a ? b : c', 'b' can contain an unparenthesized comma. If it does,
      // that is *never* the end of the initializer. Skip to the ':'.
      if (!ConsumeAndStoreConditional(Toks))
        return false;
      break;

    case tok::greatergreatergreater:
      if (!getLangOpts().CPlusPlus11)
        goto consume_token;
      if (AngleCount) --AngleCount;
      if (KnownTemplateCount) --KnownTemplateCount;
      // Fall through.
    case tok::greatergreater:
      if (!getLangOpts().CPlusPlus11)
        goto consume_token;
      if (AngleCount) --AngleCount;
      if (KnownTemplateCount) --KnownTemplateCount;
      // Fall through.
    case tok::greater:
      if (AngleCount) --AngleCount;
      if (KnownTemplateCount) --KnownTemplateCount;
      goto consume_token;

    case tok::kw_template:
      // 'template' identifier '<' is known to start a template argument list,
      // and can be used to disambiguate the parse.
      // FIXME: Support all forms of 'template' unqualified-id '<'.
      Toks.push_back(Tok);
      ConsumeToken();
      if (Tok.is(tok::identifier)) {
        Toks.push_back(Tok);
        ConsumeToken();
        if (Tok.is(tok::less)) {
          ++KnownTemplateCount;
          Toks.push_back(Tok);
          ConsumeToken();
        }
      }
      break;

    case tok::kw_operator:
      // If 'operator' precedes other punctuation, that punctuation loses
      // its special behavior.
      Toks.push_back(Tok);
      ConsumeToken();
      switch (Tok.getKind()) {
      case tok::comma:
      case tok::greatergreatergreater:
      case tok::greatergreater:
      case tok::greater:
      case tok::less:
        Toks.push_back(Tok);
        ConsumeToken();
        break;
      default:
        break;
      }
      break;

    case tok::l_paren:
      // Recursively consume properly-nested parens.
      Toks.push_back(Tok);
      ConsumeParen();
      ConsumeAndStoreUntil(tok::r_paren, Toks, /*StopAtSemi=*/false);
      break;
    case tok::l_square:
      // Recursively consume properly-nested square brackets.
      Toks.push_back(Tok);
      ConsumeBracket();
      ConsumeAndStoreUntil(tok::r_square, Toks, /*StopAtSemi=*/false);
      break;
    case tok::l_brace:
      // Recursively consume properly-nested braces.
      Toks.push_back(Tok);
      ConsumeBrace();
      ConsumeAndStoreUntil(tok::r_brace, Toks, /*StopAtSemi=*/false);
      break;

    // Okay, we found a ']' or '}' or ')', which we think should be balanced.
    // Since the user wasn't looking for this token (if they were, it would
    // already be handled), this isn't balanced.  If there is a LHS token at a
    // higher level, we will assume that this matches the unbalanced token
    // and return it.  Otherwise, this is a spurious RHS token, which we skip.
    case tok::r_paren:
      if (CIK == CIK_DefaultArgument)
        return true; // End of the default argument.
      if (ParenCount && !IsFirstTokenConsumed)
        return false;  // Matches something.
      goto consume_token;
    case tok::r_square:
      if (BracketCount && !IsFirstTokenConsumed)
        return false;  // Matches something.
      goto consume_token;
    case tok::r_brace:
      if (BraceCount && !IsFirstTokenConsumed)
        return false;  // Matches something.
      goto consume_token;

    case tok::code_completion:
      Toks.push_back(Tok);
      ConsumeCodeCompletionToken();
      break;

    case tok::string_literal:
    case tok::wide_string_literal:
    case tok::utf8_string_literal:
    case tok::utf16_string_literal:
    case tok::utf32_string_literal:
      Toks.push_back(Tok);
      ConsumeStringToken();
      break;
    case tok::semi:
      if (CIK == CIK_DefaultInitializer)
        return true; // End of the default initializer.
      // FALL THROUGH.
    default:
    consume_token:
      Toks.push_back(Tok);
      ConsumeToken();
      break;
    }
    IsFirstTokenConsumed = false;
  }
}
