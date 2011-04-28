//===--- ParseExprCXX.cpp - C++ Expression Parsing ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Expression parsing implementation for C++.
//
//===----------------------------------------------------------------------===//

#include "clang/Parse/ParseDiagnostic.h"
#include "clang/Parse/Parser.h"
#include "RAIIObjectsForParser.h"
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/ParsedTemplate.h"
#include "llvm/Support/ErrorHandling.h"

using namespace clang;

static int SelectDigraphErrorMessage(tok::TokenKind Kind) {
  switch (Kind) {
    case tok::kw_template:         return 0;
    case tok::kw_const_cast:       return 1;
    case tok::kw_dynamic_cast:     return 2;
    case tok::kw_reinterpret_cast: return 3;
    case tok::kw_static_cast:      return 4;
    default:
      assert(0 && "Unknown type for digraph error message.");
      return -1;
  }
}

// Are the two tokens adjacent in the same source file?
static bool AreTokensAdjacent(Preprocessor &PP, Token &First, Token &Second) {
  SourceManager &SM = PP.getSourceManager();
  SourceLocation FirstLoc = SM.getSpellingLoc(First.getLocation());
  SourceLocation FirstEnd = FirstLoc.getFileLocWithOffset(First.getLength());
  return FirstEnd == SM.getSpellingLoc(Second.getLocation());
}

// Suggest fixit for "<::" after a cast.
static void FixDigraph(Parser &P, Preprocessor &PP, Token &DigraphToken,
                       Token &ColonToken, tok::TokenKind Kind, bool AtDigraph) {
  // Pull '<:' and ':' off token stream.
  if (!AtDigraph)
    PP.Lex(DigraphToken);
  PP.Lex(ColonToken);

  SourceRange Range;
  Range.setBegin(DigraphToken.getLocation());
  Range.setEnd(ColonToken.getLocation());
  P.Diag(DigraphToken.getLocation(), diag::err_missing_whitespace_digraph)
      << SelectDigraphErrorMessage(Kind)
      << FixItHint::CreateReplacement(Range, "< ::");

  // Update token information to reflect their change in token type.
  ColonToken.setKind(tok::coloncolon);
  ColonToken.setLocation(ColonToken.getLocation().getFileLocWithOffset(-1));
  ColonToken.setLength(2);
  DigraphToken.setKind(tok::less);
  DigraphToken.setLength(1);

  // Push new tokens back to token stream.
  PP.EnterToken(ColonToken);
  if (!AtDigraph)
    PP.EnterToken(DigraphToken);
}

/// \brief Parse global scope or nested-name-specifier if present.
///
/// Parses a C++ global scope specifier ('::') or nested-name-specifier (which
/// may be preceded by '::'). Note that this routine will not parse ::new or
/// ::delete; it will just leave them in the token stream.
///
///       '::'[opt] nested-name-specifier
///       '::'
///
///       nested-name-specifier:
///         type-name '::'
///         namespace-name '::'
///         nested-name-specifier identifier '::'
///         nested-name-specifier 'template'[opt] simple-template-id '::'
///
///
/// \param SS the scope specifier that will be set to the parsed
/// nested-name-specifier (or empty)
///
/// \param ObjectType if this nested-name-specifier is being parsed following
/// the "." or "->" of a member access expression, this parameter provides the
/// type of the object whose members are being accessed.
///
/// \param EnteringContext whether we will be entering into the context of
/// the nested-name-specifier after parsing it.
///
/// \param MayBePseudoDestructor When non-NULL, points to a flag that
/// indicates whether this nested-name-specifier may be part of a
/// pseudo-destructor name. In this case, the flag will be set false
/// if we don't actually end up parsing a destructor name. Moreorover,
/// if we do end up determining that we are parsing a destructor name,
/// the last component of the nested-name-specifier is not parsed as
/// part of the scope specifier.

/// member access expression, e.g., the \p T:: in \p p->T::m.
///
/// \returns true if there was an error parsing a scope specifier
bool Parser::ParseOptionalCXXScopeSpecifier(CXXScopeSpec &SS,
                                            ParsedType ObjectType,
                                            bool EnteringContext,
                                            bool *MayBePseudoDestructor,
                                            bool IsTypename) {
  assert(getLang().CPlusPlus &&
         "Call sites of this function should be guarded by checking for C++");

  if (Tok.is(tok::annot_cxxscope)) {
    Actions.RestoreNestedNameSpecifierAnnotation(Tok.getAnnotationValue(),
                                                 Tok.getAnnotationRange(),
                                                 SS);
    ConsumeToken();
    return false;
  }

  bool HasScopeSpecifier = false;

  if (Tok.is(tok::coloncolon)) {
    // ::new and ::delete aren't nested-name-specifiers.
    tok::TokenKind NextKind = NextToken().getKind();
    if (NextKind == tok::kw_new || NextKind == tok::kw_delete)
      return false;

    // '::' - Global scope qualifier.
    if (Actions.ActOnCXXGlobalScopeSpecifier(getCurScope(), ConsumeToken(), SS))
      return true;
    
    HasScopeSpecifier = true;
  }

  bool CheckForDestructor = false;
  if (MayBePseudoDestructor && *MayBePseudoDestructor) {
    CheckForDestructor = true;
    *MayBePseudoDestructor = false;
  }

  while (true) {
    if (HasScopeSpecifier) {
      // C++ [basic.lookup.classref]p5:
      //   If the qualified-id has the form
      //
      //       ::class-name-or-namespace-name::...
      //
      //   the class-name-or-namespace-name is looked up in global scope as a
      //   class-name or namespace-name.
      //
      // To implement this, we clear out the object type as soon as we've
      // seen a leading '::' or part of a nested-name-specifier.
      ObjectType = ParsedType();
      
      if (Tok.is(tok::code_completion)) {
        // Code completion for a nested-name-specifier, where the code
        // code completion token follows the '::'.
        Actions.CodeCompleteQualifiedId(getCurScope(), SS, EnteringContext);
        SourceLocation ccLoc = ConsumeCodeCompletionToken();
        // Include code completion token into the range of the scope otherwise
        // when we try to annotate the scope tokens the dangling code completion
        // token will cause assertion in
        // Preprocessor::AnnotatePreviousCachedTokens.
        SS.setEndLoc(ccLoc);
      }
    }

    // nested-name-specifier:
    //   nested-name-specifier 'template'[opt] simple-template-id '::'

    // Parse the optional 'template' keyword, then make sure we have
    // 'identifier <' after it.
    if (Tok.is(tok::kw_template)) {
      // If we don't have a scope specifier or an object type, this isn't a
      // nested-name-specifier, since they aren't allowed to start with
      // 'template'.
      if (!HasScopeSpecifier && !ObjectType)
        break;

      TentativeParsingAction TPA(*this);
      SourceLocation TemplateKWLoc = ConsumeToken();
      
      UnqualifiedId TemplateName;
      if (Tok.is(tok::identifier)) {
        // Consume the identifier.
        TemplateName.setIdentifier(Tok.getIdentifierInfo(), Tok.getLocation());
        ConsumeToken();
      } else if (Tok.is(tok::kw_operator)) {
        if (ParseUnqualifiedIdOperator(SS, EnteringContext, ObjectType, 
                                       TemplateName)) {
          TPA.Commit();
          break;
        }
        
        if (TemplateName.getKind() != UnqualifiedId::IK_OperatorFunctionId &&
            TemplateName.getKind() != UnqualifiedId::IK_LiteralOperatorId) {
          Diag(TemplateName.getSourceRange().getBegin(),
               diag::err_id_after_template_in_nested_name_spec)
            << TemplateName.getSourceRange();
          TPA.Commit();
          break;
        }
      } else {
        TPA.Revert();
        break;
      }

      // If the next token is not '<', we have a qualified-id that refers
      // to a template name, such as T::template apply, but is not a 
      // template-id.
      if (Tok.isNot(tok::less)) {
        TPA.Revert();
        break;
      }        
      
      // Commit to parsing the template-id.
      TPA.Commit();
      TemplateTy Template;
      if (TemplateNameKind TNK = Actions.ActOnDependentTemplateName(getCurScope(), 
                                                                TemplateKWLoc, 
                                                                    SS, 
                                                                  TemplateName,
                                                                    ObjectType, 
                                                                EnteringContext,
                                                                    Template)) {
        if (AnnotateTemplateIdToken(Template, TNK, SS, TemplateName, 
                                    TemplateKWLoc, false))
          return true;
      } else
        return true;

      continue;
    }

    if (Tok.is(tok::annot_template_id) && NextToken().is(tok::coloncolon)) {
      // We have
      //
      //   simple-template-id '::'
      //
      // So we need to check whether the simple-template-id is of the
      // right kind (it should name a type or be dependent), and then
      // convert it into a type within the nested-name-specifier.
      TemplateIdAnnotation *TemplateId
        = static_cast<TemplateIdAnnotation *>(Tok.getAnnotationValue());
      if (CheckForDestructor && GetLookAheadToken(2).is(tok::tilde)) {
        *MayBePseudoDestructor = true;
        return false;
      }

      // Consume the template-id token.
      ConsumeToken();
      
      assert(Tok.is(tok::coloncolon) && "NextToken() not working properly!");
      SourceLocation CCLoc = ConsumeToken();

      if (!HasScopeSpecifier)
        HasScopeSpecifier = true;
      
      ASTTemplateArgsPtr TemplateArgsPtr(Actions,
                                         TemplateId->getTemplateArgs(),
                                         TemplateId->NumArgs);
      
      if (Actions.ActOnCXXNestedNameSpecifier(getCurScope(),
                                              /*FIXME:*/SourceLocation(),
                                              SS, 
                                              TemplateId->Template,
                                              TemplateId->TemplateNameLoc,
                                              TemplateId->LAngleLoc,
                                              TemplateArgsPtr,
                                              TemplateId->RAngleLoc,
                                              CCLoc,
                                              EnteringContext)) {
        SourceLocation StartLoc 
          = SS.getBeginLoc().isValid()? SS.getBeginLoc()
                                      : TemplateId->TemplateNameLoc;
        SS.SetInvalid(SourceRange(StartLoc, CCLoc));
      }
      
      TemplateId->Destroy();
      continue;
    }


    // The rest of the nested-name-specifier possibilities start with
    // tok::identifier.
    if (Tok.isNot(tok::identifier))
      break;

    IdentifierInfo &II = *Tok.getIdentifierInfo();

    // nested-name-specifier:
    //   type-name '::'
    //   namespace-name '::'
    //   nested-name-specifier identifier '::'
    Token Next = NextToken();
    
    // If we get foo:bar, this is almost certainly a typo for foo::bar.  Recover
    // and emit a fixit hint for it.
    if (Next.is(tok::colon) && !ColonIsSacred) {
      if (Actions.IsInvalidUnlessNestedName(getCurScope(), SS, II, 
                                            Tok.getLocation(), 
                                            Next.getLocation(), ObjectType,
                                            EnteringContext) &&
          // If the token after the colon isn't an identifier, it's still an
          // error, but they probably meant something else strange so don't
          // recover like this.
          PP.LookAhead(1).is(tok::identifier)) {
        Diag(Next, diag::err_unexected_colon_in_nested_name_spec)
          << FixItHint::CreateReplacement(Next.getLocation(), "::");
        
        // Recover as if the user wrote '::'.
        Next.setKind(tok::coloncolon);
      }
    }
    
    if (Next.is(tok::coloncolon)) {
      if (CheckForDestructor && GetLookAheadToken(2).is(tok::tilde) &&
          !Actions.isNonTypeNestedNameSpecifier(getCurScope(), SS, Tok.getLocation(),
                                                II, ObjectType)) {
        *MayBePseudoDestructor = true;
        return false;
      }

      // We have an identifier followed by a '::'. Lookup this name
      // as the name in a nested-name-specifier.
      SourceLocation IdLoc = ConsumeToken();
      assert((Tok.is(tok::coloncolon) || Tok.is(tok::colon)) &&
             "NextToken() not working properly!");
      SourceLocation CCLoc = ConsumeToken();

      HasScopeSpecifier = true;
      if (Actions.ActOnCXXNestedNameSpecifier(getCurScope(), II, IdLoc, CCLoc,
                                              ObjectType, EnteringContext, SS))
        SS.SetInvalid(SourceRange(IdLoc, CCLoc));
      
      continue;
    }

    // Check for '<::' which should be '< ::' instead of '[:' when following
    // a template name.
    if (Next.is(tok::l_square) && Next.getLength() == 2) {
      Token SecondToken = GetLookAheadToken(2);
      if (SecondToken.is(tok::colon) &&
          AreTokensAdjacent(PP, Next, SecondToken)) {
        TemplateTy Template;
        UnqualifiedId TemplateName;
        TemplateName.setIdentifier(&II, Tok.getLocation());
        bool MemberOfUnknownSpecialization;
        if (Actions.isTemplateName(getCurScope(), SS,
                                   /*hasTemplateKeyword=*/false,
                                   TemplateName,
                                   ObjectType,
                                   EnteringContext,
                                   Template,
                                   MemberOfUnknownSpecialization)) {
          FixDigraph(*this, PP, Next, SecondToken, tok::kw_template,
                     /*AtDigraph*/false);
        }
      }
    }

    // nested-name-specifier:
    //   type-name '<'
    if (Next.is(tok::less)) {
      TemplateTy Template;
      UnqualifiedId TemplateName;
      TemplateName.setIdentifier(&II, Tok.getLocation());
      bool MemberOfUnknownSpecialization;
      if (TemplateNameKind TNK = Actions.isTemplateName(getCurScope(), SS, 
                                              /*hasTemplateKeyword=*/false,
                                                        TemplateName,
                                                        ObjectType,
                                                        EnteringContext,
                                                        Template,
                                              MemberOfUnknownSpecialization)) {
        // We have found a template name, so annotate this this token
        // with a template-id annotation. We do not permit the
        // template-id to be translated into a type annotation,
        // because some clients (e.g., the parsing of class template
        // specializations) still want to see the original template-id
        // token.
        ConsumeToken();
        if (AnnotateTemplateIdToken(Template, TNK, SS, TemplateName, 
                                    SourceLocation(), false))
          return true;
        continue;
      } 
      
      if (MemberOfUnknownSpecialization && (ObjectType || SS.isSet()) && 
          (IsTypename || IsTemplateArgumentList(1))) {
        // We have something like t::getAs<T>, where getAs is a 
        // member of an unknown specialization. However, this will only
        // parse correctly as a template, so suggest the keyword 'template'
        // before 'getAs' and treat this as a dependent template name.
        unsigned DiagID = diag::err_missing_dependent_template_keyword;
        if (getLang().Microsoft)
          DiagID = diag::warn_missing_dependent_template_keyword;
        
        Diag(Tok.getLocation(), DiagID)
          << II.getName()
          << FixItHint::CreateInsertion(Tok.getLocation(), "template ");
        
        if (TemplateNameKind TNK 
              = Actions.ActOnDependentTemplateName(getCurScope(), 
                                                   Tok.getLocation(), SS, 
                                                   TemplateName, ObjectType,
                                                   EnteringContext, Template)) {
          // Consume the identifier.
          ConsumeToken();
          if (AnnotateTemplateIdToken(Template, TNK, SS, TemplateName, 
                                      SourceLocation(), false))
            return true;                
        }
        else
          return true;     
                
        continue;        
      }
    }

    // We don't have any tokens that form the beginning of a
    // nested-name-specifier, so we're done.
    break;
  }

  // Even if we didn't see any pieces of a nested-name-specifier, we
  // still check whether there is a tilde in this position, which
  // indicates a potential pseudo-destructor.
  if (CheckForDestructor && Tok.is(tok::tilde))
    *MayBePseudoDestructor = true;

  return false;
}

/// ParseCXXIdExpression - Handle id-expression.
///
///       id-expression:
///         unqualified-id
///         qualified-id
///
///       qualified-id:
///         '::'[opt] nested-name-specifier 'template'[opt] unqualified-id
///         '::' identifier
///         '::' operator-function-id
///         '::' template-id
///
/// NOTE: The standard specifies that, for qualified-id, the parser does not
/// expect:
///
///   '::' conversion-function-id
///   '::' '~' class-name
///
/// This may cause a slight inconsistency on diagnostics:
///
/// class C {};
/// namespace A {}
/// void f() {
///   :: A :: ~ C(); // Some Sema error about using destructor with a
///                  // namespace.
///   :: ~ C(); // Some Parser error like 'unexpected ~'.
/// }
///
/// We simplify the parser a bit and make it work like:
///
///       qualified-id:
///         '::'[opt] nested-name-specifier 'template'[opt] unqualified-id
///         '::' unqualified-id
///
/// That way Sema can handle and report similar errors for namespaces and the
/// global scope.
///
/// The isAddressOfOperand parameter indicates that this id-expression is a
/// direct operand of the address-of operator. This is, besides member contexts,
/// the only place where a qualified-id naming a non-static class member may
/// appear.
///
ExprResult Parser::ParseCXXIdExpression(bool isAddressOfOperand) {
  // qualified-id:
  //   '::'[opt] nested-name-specifier 'template'[opt] unqualified-id
  //   '::' unqualified-id
  //
  CXXScopeSpec SS;
  ParseOptionalCXXScopeSpecifier(SS, ParsedType(), false);
  
  UnqualifiedId Name;
  if (ParseUnqualifiedId(SS, 
                         /*EnteringContext=*/false, 
                         /*AllowDestructorName=*/false, 
                         /*AllowConstructorName=*/false, 
                         /*ObjectType=*/ ParsedType(),
                         Name))
    return ExprError();

  // This is only the direct operand of an & operator if it is not
  // followed by a postfix-expression suffix.
  if (isAddressOfOperand && isPostfixExpressionSuffixStart())
    isAddressOfOperand = false;
  
  return Actions.ActOnIdExpression(getCurScope(), SS, Name, Tok.is(tok::l_paren),
                                   isAddressOfOperand);
  
}

/// ParseCXXCasts - This handles the various ways to cast expressions to another
/// type.
///
///       postfix-expression: [C++ 5.2p1]
///         'dynamic_cast' '<' type-name '>' '(' expression ')'
///         'static_cast' '<' type-name '>' '(' expression ')'
///         'reinterpret_cast' '<' type-name '>' '(' expression ')'
///         'const_cast' '<' type-name '>' '(' expression ')'
///
ExprResult Parser::ParseCXXCasts() {
  tok::TokenKind Kind = Tok.getKind();
  const char *CastName = 0;     // For error messages

  switch (Kind) {
  default: assert(0 && "Unknown C++ cast!"); abort();
  case tok::kw_const_cast:       CastName = "const_cast";       break;
  case tok::kw_dynamic_cast:     CastName = "dynamic_cast";     break;
  case tok::kw_reinterpret_cast: CastName = "reinterpret_cast"; break;
  case tok::kw_static_cast:      CastName = "static_cast";      break;
  }

  SourceLocation OpLoc = ConsumeToken();
  SourceLocation LAngleBracketLoc = Tok.getLocation();

  // Check for "<::" which is parsed as "[:".  If found, fix token stream,
  // diagnose error, suggest fix, and recover parsing.
  Token Next = NextToken();
  if (Tok.is(tok::l_square) && Tok.getLength() == 2 && Next.is(tok::colon) &&
      AreTokensAdjacent(PP, Tok, Next))
    FixDigraph(*this, PP, Tok, Next, Kind, /*AtDigraph*/true);

  if (ExpectAndConsume(tok::less, diag::err_expected_less_after, CastName))
    return ExprError();

  TypeResult CastTy = ParseTypeName();
  SourceLocation RAngleBracketLoc = Tok.getLocation();

  if (ExpectAndConsume(tok::greater, diag::err_expected_greater))
    return ExprError(Diag(LAngleBracketLoc, diag::note_matching) << "<");

  SourceLocation LParenLoc = Tok.getLocation(), RParenLoc;

  if (ExpectAndConsume(tok::l_paren, diag::err_expected_lparen_after, CastName))
    return ExprError();

  ExprResult Result = ParseExpression();

  // Match the ')'.
  RParenLoc = MatchRHSPunctuation(tok::r_paren, LParenLoc);

  if (!Result.isInvalid() && !CastTy.isInvalid())
    Result = Actions.ActOnCXXNamedCast(OpLoc, Kind,
                                       LAngleBracketLoc, CastTy.get(),
                                       RAngleBracketLoc,
                                       LParenLoc, Result.take(), RParenLoc);

  return move(Result);
}

/// ParseCXXTypeid - This handles the C++ typeid expression.
///
///       postfix-expression: [C++ 5.2p1]
///         'typeid' '(' expression ')'
///         'typeid' '(' type-id ')'
///
ExprResult Parser::ParseCXXTypeid() {
  assert(Tok.is(tok::kw_typeid) && "Not 'typeid'!");

  SourceLocation OpLoc = ConsumeToken();
  SourceLocation LParenLoc = Tok.getLocation();
  SourceLocation RParenLoc;

  // typeid expressions are always parenthesized.
  if (ExpectAndConsume(tok::l_paren, diag::err_expected_lparen_after,
      "typeid"))
    return ExprError();

  ExprResult Result;

  if (isTypeIdInParens()) {
    TypeResult Ty = ParseTypeName();

    // Match the ')'.
    RParenLoc = MatchRHSPunctuation(tok::r_paren, LParenLoc);

    if (Ty.isInvalid() || RParenLoc.isInvalid())
      return ExprError();

    Result = Actions.ActOnCXXTypeid(OpLoc, LParenLoc, /*isType=*/true,
                                    Ty.get().getAsOpaquePtr(), RParenLoc);
  } else {
    // C++0x [expr.typeid]p3:
    //   When typeid is applied to an expression other than an lvalue of a
    //   polymorphic class type [...] The expression is an unevaluated
    //   operand (Clause 5).
    //
    // Note that we can't tell whether the expression is an lvalue of a
    // polymorphic class type until after we've parsed the expression, so
    // we the expression is potentially potentially evaluated.
    EnterExpressionEvaluationContext Unevaluated(Actions,
                                       Sema::PotentiallyPotentiallyEvaluated);
    Result = ParseExpression();

    // Match the ')'.
    if (Result.isInvalid())
      SkipUntil(tok::r_paren);
    else {
      RParenLoc = MatchRHSPunctuation(tok::r_paren, LParenLoc);
      if (RParenLoc.isInvalid())
        return ExprError();

      // If we are a foo<int> that identifies a single function, resolve it now...  
      Expr* e = Result.get();
      if (e->getType() == Actions.Context.OverloadTy) {
        ExprResult er = 
              Actions.ResolveAndFixSingleFunctionTemplateSpecialization(e);
        if (er.isUsable())
          Result = er.release();
      }
      Result = Actions.ActOnCXXTypeid(OpLoc, LParenLoc, /*isType=*/false,
                                      Result.release(), RParenLoc);
    }
  }

  return move(Result);
}

/// ParseCXXUuidof - This handles the Microsoft C++ __uuidof expression.
///
///         '__uuidof' '(' expression ')'
///         '__uuidof' '(' type-id ')'
///
ExprResult Parser::ParseCXXUuidof() {
  assert(Tok.is(tok::kw___uuidof) && "Not '__uuidof'!");

  SourceLocation OpLoc = ConsumeToken();
  SourceLocation LParenLoc = Tok.getLocation();
  SourceLocation RParenLoc;

  // __uuidof expressions are always parenthesized.
  if (ExpectAndConsume(tok::l_paren, diag::err_expected_lparen_after,
      "__uuidof"))
    return ExprError();

  ExprResult Result;

  if (isTypeIdInParens()) {
    TypeResult Ty = ParseTypeName();

    // Match the ')'.
    RParenLoc = MatchRHSPunctuation(tok::r_paren, LParenLoc);

    if (Ty.isInvalid())
      return ExprError();

    Result = Actions.ActOnCXXUuidof(OpLoc, LParenLoc, /*isType=*/true,
                                    Ty.get().getAsOpaquePtr(), RParenLoc);
  } else {
    EnterExpressionEvaluationContext Unevaluated(Actions, Sema::Unevaluated);
    Result = ParseExpression();

    // Match the ')'.
    if (Result.isInvalid())
      SkipUntil(tok::r_paren);
    else {
      RParenLoc = MatchRHSPunctuation(tok::r_paren, LParenLoc);

      Result = Actions.ActOnCXXUuidof(OpLoc, LParenLoc, /*isType=*/false,
                                      Result.release(), RParenLoc);
    }
  }

  return move(Result);
}

/// \brief Parse a C++ pseudo-destructor expression after the base,
/// . or -> operator, and nested-name-specifier have already been
/// parsed.
///
///       postfix-expression: [C++ 5.2]
///         postfix-expression . pseudo-destructor-name
///         postfix-expression -> pseudo-destructor-name
///
///       pseudo-destructor-name: 
///         ::[opt] nested-name-specifier[opt] type-name :: ~type-name 
///         ::[opt] nested-name-specifier template simple-template-id :: 
///                 ~type-name 
///         ::[opt] nested-name-specifier[opt] ~type-name
///       
ExprResult 
Parser::ParseCXXPseudoDestructor(ExprArg Base, SourceLocation OpLoc,
                                 tok::TokenKind OpKind,
                                 CXXScopeSpec &SS,
                                 ParsedType ObjectType) {
  // We're parsing either a pseudo-destructor-name or a dependent
  // member access that has the same form as a
  // pseudo-destructor-name. We parse both in the same way and let
  // the action model sort them out.
  //
  // Note that the ::[opt] nested-name-specifier[opt] has already
  // been parsed, and if there was a simple-template-id, it has
  // been coalesced into a template-id annotation token.
  UnqualifiedId FirstTypeName;
  SourceLocation CCLoc;
  if (Tok.is(tok::identifier)) {
    FirstTypeName.setIdentifier(Tok.getIdentifierInfo(), Tok.getLocation());
    ConsumeToken();
    assert(Tok.is(tok::coloncolon) &&"ParseOptionalCXXScopeSpecifier fail");
    CCLoc = ConsumeToken();
  } else if (Tok.is(tok::annot_template_id)) {
    FirstTypeName.setTemplateId(
                              (TemplateIdAnnotation *)Tok.getAnnotationValue());
    ConsumeToken();
    assert(Tok.is(tok::coloncolon) &&"ParseOptionalCXXScopeSpecifier fail");
    CCLoc = ConsumeToken();
  } else {
    FirstTypeName.setIdentifier(0, SourceLocation());
  }

  // Parse the tilde.
  assert(Tok.is(tok::tilde) && "ParseOptionalCXXScopeSpecifier fail");
  SourceLocation TildeLoc = ConsumeToken();
  if (!Tok.is(tok::identifier)) {
    Diag(Tok, diag::err_destructor_tilde_identifier);
    return ExprError();
  }
  
  // Parse the second type.
  UnqualifiedId SecondTypeName;
  IdentifierInfo *Name = Tok.getIdentifierInfo();
  SourceLocation NameLoc = ConsumeToken();
  SecondTypeName.setIdentifier(Name, NameLoc);
  
  // If there is a '<', the second type name is a template-id. Parse
  // it as such.
  if (Tok.is(tok::less) &&
      ParseUnqualifiedIdTemplateId(SS, Name, NameLoc, false, ObjectType,
                                   SecondTypeName, /*AssumeTemplateName=*/true,
                                   /*TemplateKWLoc*/SourceLocation()))
    return ExprError();

  return Actions.ActOnPseudoDestructorExpr(getCurScope(), Base,
                                           OpLoc, OpKind,
                                           SS, FirstTypeName, CCLoc,
                                           TildeLoc, SecondTypeName,
                                           Tok.is(tok::l_paren));
}

/// ParseCXXBoolLiteral - This handles the C++ Boolean literals.
///
///       boolean-literal: [C++ 2.13.5]
///         'true'
///         'false'
ExprResult Parser::ParseCXXBoolLiteral() {
  tok::TokenKind Kind = Tok.getKind();
  return Actions.ActOnCXXBoolLiteral(ConsumeToken(), Kind);
}

/// ParseThrowExpression - This handles the C++ throw expression.
///
///       throw-expression: [C++ 15]
///         'throw' assignment-expression[opt]
ExprResult Parser::ParseThrowExpression() {
  assert(Tok.is(tok::kw_throw) && "Not throw!");
  SourceLocation ThrowLoc = ConsumeToken();           // Eat the throw token.

  // If the current token isn't the start of an assignment-expression,
  // then the expression is not present.  This handles things like:
  //   "C ? throw : (void)42", which is crazy but legal.
  switch (Tok.getKind()) {  // FIXME: move this predicate somewhere common.
  case tok::semi:
  case tok::r_paren:
  case tok::r_square:
  case tok::r_brace:
  case tok::colon:
  case tok::comma:
    return Actions.ActOnCXXThrow(ThrowLoc, 0);

  default:
    ExprResult Expr(ParseAssignmentExpression());
    if (Expr.isInvalid()) return move(Expr);
    return Actions.ActOnCXXThrow(ThrowLoc, Expr.take());
  }
}

/// ParseCXXThis - This handles the C++ 'this' pointer.
///
/// C++ 9.3.2: In the body of a non-static member function, the keyword this is
/// a non-lvalue expression whose value is the address of the object for which
/// the function is called.
ExprResult Parser::ParseCXXThis() {
  assert(Tok.is(tok::kw_this) && "Not 'this'!");
  SourceLocation ThisLoc = ConsumeToken();
  return Actions.ActOnCXXThis(ThisLoc);
}

/// ParseCXXTypeConstructExpression - Parse construction of a specified type.
/// Can be interpreted either as function-style casting ("int(x)")
/// or class type construction ("ClassType(x,y,z)")
/// or creation of a value-initialized type ("int()").
///
///       postfix-expression: [C++ 5.2p1]
///         simple-type-specifier '(' expression-list[opt] ')'      [C++ 5.2.3]
///         typename-specifier '(' expression-list[opt] ')'         [TODO]
///
ExprResult
Parser::ParseCXXTypeConstructExpression(const DeclSpec &DS) {
  Declarator DeclaratorInfo(DS, Declarator::TypeNameContext);
  ParsedType TypeRep = Actions.ActOnTypeName(getCurScope(), DeclaratorInfo).get();

  assert(Tok.is(tok::l_paren) && "Expected '('!");
  GreaterThanIsOperatorScope G(GreaterThanIsOperator, true);

  SourceLocation LParenLoc = ConsumeParen();

  ExprVector Exprs(Actions);
  CommaLocsTy CommaLocs;

  if (Tok.isNot(tok::r_paren)) {
    if (ParseExpressionList(Exprs, CommaLocs)) {
      SkipUntil(tok::r_paren);
      return ExprError();
    }
  }

  // Match the ')'.
  SourceLocation RParenLoc = MatchRHSPunctuation(tok::r_paren, LParenLoc);

  // TypeRep could be null, if it references an invalid typedef.
  if (!TypeRep)
    return ExprError();

  assert((Exprs.size() == 0 || Exprs.size()-1 == CommaLocs.size())&&
         "Unexpected number of commas!");
  return Actions.ActOnCXXTypeConstructExpr(TypeRep, LParenLoc, move_arg(Exprs),
                                           RParenLoc);
}

/// ParseCXXCondition - if/switch/while condition expression.
///
///       condition:
///         expression
///         type-specifier-seq declarator '=' assignment-expression
/// [GNU]   type-specifier-seq declarator simple-asm-expr[opt] attributes[opt]
///             '=' assignment-expression
///
/// \param ExprResult if the condition was parsed as an expression, the
/// parsed expression.
///
/// \param DeclResult if the condition was parsed as a declaration, the
/// parsed declaration.
///
/// \param Loc The location of the start of the statement that requires this
/// condition, e.g., the "for" in a for loop.
///
/// \param ConvertToBoolean Whether the condition expression should be
/// converted to a boolean value.
///
/// \returns true if there was a parsing, false otherwise.
bool Parser::ParseCXXCondition(ExprResult &ExprOut,
                               Decl *&DeclOut,
                               SourceLocation Loc,
                               bool ConvertToBoolean) {
  if (Tok.is(tok::code_completion)) {
    Actions.CodeCompleteOrdinaryName(getCurScope(), Sema::PCC_Condition);
    ConsumeCodeCompletionToken();
  }

  if (!isCXXConditionDeclaration()) {
    // Parse the expression.
    ExprOut = ParseExpression(); // expression
    DeclOut = 0;
    if (ExprOut.isInvalid())
      return true;

    // If required, convert to a boolean value.
    if (ConvertToBoolean)
      ExprOut
        = Actions.ActOnBooleanCondition(getCurScope(), Loc, ExprOut.get());
    return ExprOut.isInvalid();
  }

  // type-specifier-seq
  DeclSpec DS(AttrFactory);
  ParseSpecifierQualifierList(DS);

  // declarator
  Declarator DeclaratorInfo(DS, Declarator::ConditionContext);
  ParseDeclarator(DeclaratorInfo);

  // simple-asm-expr[opt]
  if (Tok.is(tok::kw_asm)) {
    SourceLocation Loc;
    ExprResult AsmLabel(ParseSimpleAsm(&Loc));
    if (AsmLabel.isInvalid()) {
      SkipUntil(tok::semi);
      return true;
    }
    DeclaratorInfo.setAsmLabel(AsmLabel.release());
    DeclaratorInfo.SetRangeEnd(Loc);
  }

  // If attributes are present, parse them.
  MaybeParseGNUAttributes(DeclaratorInfo);

  // Type-check the declaration itself.
  DeclResult Dcl = Actions.ActOnCXXConditionDeclaration(getCurScope(), 
                                                        DeclaratorInfo);
  DeclOut = Dcl.get();
  ExprOut = ExprError();

  // '=' assignment-expression
  if (isTokenEqualOrMistypedEqualEqual(
                               diag::err_invalid_equalequal_after_declarator)) {
    ConsumeToken();
    ExprResult AssignExpr(ParseAssignmentExpression());
    if (!AssignExpr.isInvalid()) 
      Actions.AddInitializerToDecl(DeclOut, AssignExpr.take(), false,
                                   DS.getTypeSpecType() == DeclSpec::TST_auto);
  } else {
    // FIXME: C++0x allows a braced-init-list
    Diag(Tok, diag::err_expected_equal_after_declarator);
  }
  
  // FIXME: Build a reference to this declaration? Convert it to bool?
  // (This is currently handled by Sema).

  Actions.FinalizeDeclaration(DeclOut);
  
  return false;
}

/// \brief Determine whether the current token starts a C++
/// simple-type-specifier.
bool Parser::isCXXSimpleTypeSpecifier() const {
  switch (Tok.getKind()) {
  case tok::annot_typename:
  case tok::kw_short:
  case tok::kw_long:
  case tok::kw___int64:
  case tok::kw_signed:
  case tok::kw_unsigned:
  case tok::kw_void:
  case tok::kw_char:
  case tok::kw_int:
  case tok::kw_float:
  case tok::kw_double:
  case tok::kw_wchar_t:
  case tok::kw_char16_t:
  case tok::kw_char32_t:
  case tok::kw_bool:
  case tok::kw_decltype:
  case tok::kw_typeof:
    return true;

  default:
    break;
  }

  return false;
}

/// ParseCXXSimpleTypeSpecifier - [C++ 7.1.5.2] Simple type specifiers.
/// This should only be called when the current token is known to be part of
/// simple-type-specifier.
///
///       simple-type-specifier:
///         '::'[opt] nested-name-specifier[opt] type-name
///         '::'[opt] nested-name-specifier 'template' simple-template-id [TODO]
///         char
///         wchar_t
///         bool
///         short
///         int
///         long
///         signed
///         unsigned
///         float
///         double
///         void
/// [GNU]   typeof-specifier
/// [C++0x] auto               [TODO]
///
///       type-name:
///         class-name
///         enum-name
///         typedef-name
///
void Parser::ParseCXXSimpleTypeSpecifier(DeclSpec &DS) {
  DS.SetRangeStart(Tok.getLocation());
  const char *PrevSpec;
  unsigned DiagID;
  SourceLocation Loc = Tok.getLocation();

  switch (Tok.getKind()) {
  case tok::identifier:   // foo::bar
  case tok::coloncolon:   // ::foo::bar
    assert(0 && "Annotation token should already be formed!");
  default:
    assert(0 && "Not a simple-type-specifier token!");
    abort();

  // type-name
  case tok::annot_typename: {
    if (getTypeAnnotation(Tok))
      DS.SetTypeSpecType(DeclSpec::TST_typename, Loc, PrevSpec, DiagID,
                         getTypeAnnotation(Tok));
    else
      DS.SetTypeSpecError();
    
    DS.SetRangeEnd(Tok.getAnnotationEndLoc());
    ConsumeToken();
    
    // Objective-C supports syntax of the form 'id<proto1,proto2>' where 'id'
    // is a specific typedef and 'itf<proto1,proto2>' where 'itf' is an
    // Objective-C interface.  If we don't have Objective-C or a '<', this is
    // just a normal reference to a typedef name.
    if (Tok.is(tok::less) && getLang().ObjC1)
      ParseObjCProtocolQualifiers(DS);
    
    DS.Finish(Diags, PP);
    return;
  }

  // builtin types
  case tok::kw_short:
    DS.SetTypeSpecWidth(DeclSpec::TSW_short, Loc, PrevSpec, DiagID);
    break;
  case tok::kw_long:
    DS.SetTypeSpecWidth(DeclSpec::TSW_long, Loc, PrevSpec, DiagID);
    break;
  case tok::kw___int64:
    DS.SetTypeSpecWidth(DeclSpec::TSW_longlong, Loc, PrevSpec, DiagID);
    break;
  case tok::kw_signed:
    DS.SetTypeSpecSign(DeclSpec::TSS_signed, Loc, PrevSpec, DiagID);
    break;
  case tok::kw_unsigned:
    DS.SetTypeSpecSign(DeclSpec::TSS_unsigned, Loc, PrevSpec, DiagID);
    break;
  case tok::kw_void:
    DS.SetTypeSpecType(DeclSpec::TST_void, Loc, PrevSpec, DiagID);
    break;
  case tok::kw_char:
    DS.SetTypeSpecType(DeclSpec::TST_char, Loc, PrevSpec, DiagID);
    break;
  case tok::kw_int:
    DS.SetTypeSpecType(DeclSpec::TST_int, Loc, PrevSpec, DiagID);
    break;
  case tok::kw_float:
    DS.SetTypeSpecType(DeclSpec::TST_float, Loc, PrevSpec, DiagID);
    break;
  case tok::kw_double:
    DS.SetTypeSpecType(DeclSpec::TST_double, Loc, PrevSpec, DiagID);
    break;
  case tok::kw_wchar_t:
    DS.SetTypeSpecType(DeclSpec::TST_wchar, Loc, PrevSpec, DiagID);
    break;
  case tok::kw_char16_t:
    DS.SetTypeSpecType(DeclSpec::TST_char16, Loc, PrevSpec, DiagID);
    break;
  case tok::kw_char32_t:
    DS.SetTypeSpecType(DeclSpec::TST_char32, Loc, PrevSpec, DiagID);
    break;
  case tok::kw_bool:
    DS.SetTypeSpecType(DeclSpec::TST_bool, Loc, PrevSpec, DiagID);
    break;

    // FIXME: C++0x decltype support.
  // GNU typeof support.
  case tok::kw_typeof:
    ParseTypeofSpecifier(DS);
    DS.Finish(Diags, PP);
    return;
  }
  if (Tok.is(tok::annot_typename))
    DS.SetRangeEnd(Tok.getAnnotationEndLoc());
  else
    DS.SetRangeEnd(Tok.getLocation());
  ConsumeToken();
  DS.Finish(Diags, PP);
}

/// ParseCXXTypeSpecifierSeq - Parse a C++ type-specifier-seq (C++
/// [dcl.name]), which is a non-empty sequence of type-specifiers,
/// e.g., "const short int". Note that the DeclSpec is *not* finished
/// by parsing the type-specifier-seq, because these sequences are
/// typically followed by some form of declarator. Returns true and
/// emits diagnostics if this is not a type-specifier-seq, false
/// otherwise.
///
///   type-specifier-seq: [C++ 8.1]
///     type-specifier type-specifier-seq[opt]
///
bool Parser::ParseCXXTypeSpecifierSeq(DeclSpec &DS) {
  DS.SetRangeStart(Tok.getLocation());
  const char *PrevSpec = 0;
  unsigned DiagID;
  bool isInvalid = 0;

  // Parse one or more of the type specifiers.
  if (!ParseOptionalTypeSpecifier(DS, isInvalid, PrevSpec, DiagID,
      ParsedTemplateInfo(), /*SuppressDeclarations*/true)) {
    Diag(Tok, diag::err_expected_type);
    return true;
  }

  while (ParseOptionalTypeSpecifier(DS, isInvalid, PrevSpec, DiagID,
         ParsedTemplateInfo(), /*SuppressDeclarations*/true))
  {}

  DS.Finish(Diags, PP);
  return false;
}

/// \brief Finish parsing a C++ unqualified-id that is a template-id of
/// some form. 
///
/// This routine is invoked when a '<' is encountered after an identifier or
/// operator-function-id is parsed by \c ParseUnqualifiedId() to determine
/// whether the unqualified-id is actually a template-id. This routine will
/// then parse the template arguments and form the appropriate template-id to
/// return to the caller.
///
/// \param SS the nested-name-specifier that precedes this template-id, if
/// we're actually parsing a qualified-id.
///
/// \param Name for constructor and destructor names, this is the actual
/// identifier that may be a template-name.
///
/// \param NameLoc the location of the class-name in a constructor or 
/// destructor.
///
/// \param EnteringContext whether we're entering the scope of the 
/// nested-name-specifier.
///
/// \param ObjectType if this unqualified-id occurs within a member access
/// expression, the type of the base object whose member is being accessed.
///
/// \param Id as input, describes the template-name or operator-function-id
/// that precedes the '<'. If template arguments were parsed successfully,
/// will be updated with the template-id.
/// 
/// \param AssumeTemplateId When true, this routine will assume that the name
/// refers to a template without performing name lookup to verify. 
///
/// \returns true if a parse error occurred, false otherwise.
bool Parser::ParseUnqualifiedIdTemplateId(CXXScopeSpec &SS,
                                          IdentifierInfo *Name,
                                          SourceLocation NameLoc,
                                          bool EnteringContext,
                                          ParsedType ObjectType,
                                          UnqualifiedId &Id,
                                          bool AssumeTemplateId,
                                          SourceLocation TemplateKWLoc) {
  assert((AssumeTemplateId || Tok.is(tok::less)) &&
         "Expected '<' to finish parsing a template-id");
  
  TemplateTy Template;
  TemplateNameKind TNK = TNK_Non_template;
  switch (Id.getKind()) {
  case UnqualifiedId::IK_Identifier:
  case UnqualifiedId::IK_OperatorFunctionId:
  case UnqualifiedId::IK_LiteralOperatorId:
    if (AssumeTemplateId) {
      TNK = Actions.ActOnDependentTemplateName(getCurScope(), TemplateKWLoc, SS, 
                                               Id, ObjectType, EnteringContext,
                                               Template);
      if (TNK == TNK_Non_template)
        return true;
    } else {
      bool MemberOfUnknownSpecialization;
      TNK = Actions.isTemplateName(getCurScope(), SS,
                                   TemplateKWLoc.isValid(), Id,
                                   ObjectType, EnteringContext, Template,
                                   MemberOfUnknownSpecialization);
      
      if (TNK == TNK_Non_template && MemberOfUnknownSpecialization &&
          ObjectType && IsTemplateArgumentList()) {
        // We have something like t->getAs<T>(), where getAs is a 
        // member of an unknown specialization. However, this will only
        // parse correctly as a template, so suggest the keyword 'template'
        // before 'getAs' and treat this as a dependent template name.
        std::string Name;
        if (Id.getKind() == UnqualifiedId::IK_Identifier)
          Name = Id.Identifier->getName();
        else {
          Name = "operator ";
          if (Id.getKind() == UnqualifiedId::IK_OperatorFunctionId)
            Name += getOperatorSpelling(Id.OperatorFunctionId.Operator);
          else
            Name += Id.Identifier->getName();
        }
        Diag(Id.StartLocation, diag::err_missing_dependent_template_keyword)
          << Name
          << FixItHint::CreateInsertion(Id.StartLocation, "template ");
        TNK = Actions.ActOnDependentTemplateName(getCurScope(), TemplateKWLoc,
                                                 SS, Id, ObjectType,
                                                 EnteringContext, Template);
        if (TNK == TNK_Non_template)
          return true;              
      }
    }
    break;
      
  case UnqualifiedId::IK_ConstructorName: {
    UnqualifiedId TemplateName;
    bool MemberOfUnknownSpecialization;
    TemplateName.setIdentifier(Name, NameLoc);
    TNK = Actions.isTemplateName(getCurScope(), SS, TemplateKWLoc.isValid(),
                                 TemplateName, ObjectType, 
                                 EnteringContext, Template,
                                 MemberOfUnknownSpecialization);
    break;
  }
      
  case UnqualifiedId::IK_DestructorName: {
    UnqualifiedId TemplateName;
    bool MemberOfUnknownSpecialization;
    TemplateName.setIdentifier(Name, NameLoc);
    if (ObjectType) {
      TNK = Actions.ActOnDependentTemplateName(getCurScope(), TemplateKWLoc, SS, 
                                               TemplateName, ObjectType,
                                               EnteringContext, Template);
      if (TNK == TNK_Non_template)
        return true;
    } else {
      TNK = Actions.isTemplateName(getCurScope(), SS, TemplateKWLoc.isValid(),
                                   TemplateName, ObjectType, 
                                   EnteringContext, Template,
                                   MemberOfUnknownSpecialization);
      
      if (TNK == TNK_Non_template && !Id.DestructorName.get()) {
        Diag(NameLoc, diag::err_destructor_template_id)
          << Name << SS.getRange();
        return true;        
      }
    }
    break;
  }
      
  default:
    return false;
  }
  
  if (TNK == TNK_Non_template)
    return false;
  
  // Parse the enclosed template argument list.
  SourceLocation LAngleLoc, RAngleLoc;
  TemplateArgList TemplateArgs;
  if (Tok.is(tok::less) &&
      ParseTemplateIdAfterTemplateName(Template, Id.StartLocation,
                                       SS, true, LAngleLoc,
                                       TemplateArgs,
                                       RAngleLoc))
    return true;
  
  if (Id.getKind() == UnqualifiedId::IK_Identifier ||
      Id.getKind() == UnqualifiedId::IK_OperatorFunctionId ||
      Id.getKind() == UnqualifiedId::IK_LiteralOperatorId) {
    // Form a parsed representation of the template-id to be stored in the
    // UnqualifiedId.
    TemplateIdAnnotation *TemplateId
      = TemplateIdAnnotation::Allocate(TemplateArgs.size());

    if (Id.getKind() == UnqualifiedId::IK_Identifier) {
      TemplateId->Name = Id.Identifier;
      TemplateId->Operator = OO_None;
      TemplateId->TemplateNameLoc = Id.StartLocation;
    } else {
      TemplateId->Name = 0;
      TemplateId->Operator = Id.OperatorFunctionId.Operator;
      TemplateId->TemplateNameLoc = Id.StartLocation;
    }

    TemplateId->SS = SS;
    TemplateId->Template = Template;
    TemplateId->Kind = TNK;
    TemplateId->LAngleLoc = LAngleLoc;
    TemplateId->RAngleLoc = RAngleLoc;
    ParsedTemplateArgument *Args = TemplateId->getTemplateArgs();
    for (unsigned Arg = 0, ArgEnd = TemplateArgs.size(); 
         Arg != ArgEnd; ++Arg)
      Args[Arg] = TemplateArgs[Arg];
    
    Id.setTemplateId(TemplateId);
    return false;
  }

  // Bundle the template arguments together.
  ASTTemplateArgsPtr TemplateArgsPtr(Actions, TemplateArgs.data(),
                                     TemplateArgs.size());
  
  // Constructor and destructor names.
  TypeResult Type
    = Actions.ActOnTemplateIdType(SS, Template, NameLoc,
                                  LAngleLoc, TemplateArgsPtr,
                                  RAngleLoc);
  if (Type.isInvalid())
    return true;
  
  if (Id.getKind() == UnqualifiedId::IK_ConstructorName)
    Id.setConstructorName(Type.get(), NameLoc, RAngleLoc);
  else
    Id.setDestructorName(Id.StartLocation, Type.get(), RAngleLoc);
  
  return false;
}

/// \brief Parse an operator-function-id or conversion-function-id as part
/// of a C++ unqualified-id.
///
/// This routine is responsible only for parsing the operator-function-id or
/// conversion-function-id; it does not handle template arguments in any way.
///
/// \code
///       operator-function-id: [C++ 13.5]
///         'operator' operator
///
///       operator: one of
///            new   delete  new[]   delete[]
///            +     -    *  /    %  ^    &   |   ~
///            !     =    <  >    += -=   *=  /=  %=
///            ^=    &=   |= <<   >> >>= <<=  ==  !=
///            <=    >=   && ||   ++ --   ,   ->* ->
///            ()    []
///
///       conversion-function-id: [C++ 12.3.2]
///         operator conversion-type-id
///
///       conversion-type-id:
///         type-specifier-seq conversion-declarator[opt]
///
///       conversion-declarator:
///         ptr-operator conversion-declarator[opt]
/// \endcode
///
/// \param The nested-name-specifier that preceded this unqualified-id. If
/// non-empty, then we are parsing the unqualified-id of a qualified-id.
///
/// \param EnteringContext whether we are entering the scope of the 
/// nested-name-specifier.
///
/// \param ObjectType if this unqualified-id occurs within a member access
/// expression, the type of the base object whose member is being accessed.
///
/// \param Result on a successful parse, contains the parsed unqualified-id.
///
/// \returns true if parsing fails, false otherwise.
bool Parser::ParseUnqualifiedIdOperator(CXXScopeSpec &SS, bool EnteringContext,
                                        ParsedType ObjectType,
                                        UnqualifiedId &Result) {
  assert(Tok.is(tok::kw_operator) && "Expected 'operator' keyword");
  
  // Consume the 'operator' keyword.
  SourceLocation KeywordLoc = ConsumeToken();
  
  // Determine what kind of operator name we have.
  unsigned SymbolIdx = 0;
  SourceLocation SymbolLocations[3];
  OverloadedOperatorKind Op = OO_None;
  switch (Tok.getKind()) {
    case tok::kw_new:
    case tok::kw_delete: {
      bool isNew = Tok.getKind() == tok::kw_new;
      // Consume the 'new' or 'delete'.
      SymbolLocations[SymbolIdx++] = ConsumeToken();
      if (Tok.is(tok::l_square)) {
        // Consume the '['.
        SourceLocation LBracketLoc = ConsumeBracket();
        // Consume the ']'.
        SourceLocation RBracketLoc = MatchRHSPunctuation(tok::r_square,
                                                         LBracketLoc);
        if (RBracketLoc.isInvalid())
          return true;
        
        SymbolLocations[SymbolIdx++] = LBracketLoc;
        SymbolLocations[SymbolIdx++] = RBracketLoc;
        Op = isNew? OO_Array_New : OO_Array_Delete;
      } else {
        Op = isNew? OO_New : OO_Delete;
      }
      break;
    }
      
#define OVERLOADED_OPERATOR(Name,Spelling,Token,Unary,Binary,MemberOnly) \
    case tok::Token:                                                     \
      SymbolLocations[SymbolIdx++] = ConsumeToken();                     \
      Op = OO_##Name;                                                    \
      break;
#define OVERLOADED_OPERATOR_MULTI(Name,Spelling,Unary,Binary,MemberOnly)
#include "clang/Basic/OperatorKinds.def"
      
    case tok::l_paren: {
      // Consume the '('.
      SourceLocation LParenLoc = ConsumeParen();
      // Consume the ')'.
      SourceLocation RParenLoc = MatchRHSPunctuation(tok::r_paren,
                                                     LParenLoc);
      if (RParenLoc.isInvalid())
        return true;
      
      SymbolLocations[SymbolIdx++] = LParenLoc;
      SymbolLocations[SymbolIdx++] = RParenLoc;
      Op = OO_Call;
      break;
    }
      
    case tok::l_square: {
      // Consume the '['.
      SourceLocation LBracketLoc = ConsumeBracket();
      // Consume the ']'.
      SourceLocation RBracketLoc = MatchRHSPunctuation(tok::r_square,
                                                       LBracketLoc);
      if (RBracketLoc.isInvalid())
        return true;
      
      SymbolLocations[SymbolIdx++] = LBracketLoc;
      SymbolLocations[SymbolIdx++] = RBracketLoc;
      Op = OO_Subscript;
      break;
    }
      
    case tok::code_completion: {
      // Code completion for the operator name.
      Actions.CodeCompleteOperatorName(getCurScope());
      
      // Consume the operator token.
      ConsumeCodeCompletionToken();
      
      // Don't try to parse any further.
      return true;
    }
      
    default:
      break;
  }
  
  if (Op != OO_None) {
    // We have parsed an operator-function-id.
    Result.setOperatorFunctionId(KeywordLoc, Op, SymbolLocations);
    return false;
  }

  // Parse a literal-operator-id.
  //
  //   literal-operator-id: [C++0x 13.5.8]
  //     operator "" identifier

  if (getLang().CPlusPlus0x && Tok.is(tok::string_literal)) {
    if (Tok.getLength() != 2)
      Diag(Tok.getLocation(), diag::err_operator_string_not_empty);
    ConsumeStringToken();

    if (Tok.isNot(tok::identifier)) {
      Diag(Tok.getLocation(), diag::err_expected_ident);
      return true;
    }

    IdentifierInfo *II = Tok.getIdentifierInfo();
    Result.setLiteralOperatorId(II, KeywordLoc, ConsumeToken());
    return false;
  }
  
  // Parse a conversion-function-id.
  //
  //   conversion-function-id: [C++ 12.3.2]
  //     operator conversion-type-id
  //
  //   conversion-type-id:
  //     type-specifier-seq conversion-declarator[opt]
  //
  //   conversion-declarator:
  //     ptr-operator conversion-declarator[opt]
  
  // Parse the type-specifier-seq.
  DeclSpec DS(AttrFactory);
  if (ParseCXXTypeSpecifierSeq(DS)) // FIXME: ObjectType?
    return true;
  
  // Parse the conversion-declarator, which is merely a sequence of
  // ptr-operators.
  Declarator D(DS, Declarator::TypeNameContext);
  ParseDeclaratorInternal(D, /*DirectDeclParser=*/0);
  
  // Finish up the type.
  TypeResult Ty = Actions.ActOnTypeName(getCurScope(), D);
  if (Ty.isInvalid())
    return true;
  
  // Note that this is a conversion-function-id.
  Result.setConversionFunctionId(KeywordLoc, Ty.get(), 
                                 D.getSourceRange().getEnd());
  return false;  
}

/// \brief Parse a C++ unqualified-id (or a C identifier), which describes the
/// name of an entity.
///
/// \code
///       unqualified-id: [C++ expr.prim.general]
///         identifier
///         operator-function-id
///         conversion-function-id
/// [C++0x] literal-operator-id [TODO]
///         ~ class-name
///         template-id
///
/// \endcode
///
/// \param The nested-name-specifier that preceded this unqualified-id. If
/// non-empty, then we are parsing the unqualified-id of a qualified-id.
///
/// \param EnteringContext whether we are entering the scope of the 
/// nested-name-specifier.
///
/// \param AllowDestructorName whether we allow parsing of a destructor name.
///
/// \param AllowConstructorName whether we allow parsing a constructor name.
///
/// \param ObjectType if this unqualified-id occurs within a member access
/// expression, the type of the base object whose member is being accessed.
///
/// \param Result on a successful parse, contains the parsed unqualified-id.
///
/// \returns true if parsing fails, false otherwise.
bool Parser::ParseUnqualifiedId(CXXScopeSpec &SS, bool EnteringContext,
                                bool AllowDestructorName,
                                bool AllowConstructorName,
                                ParsedType ObjectType,
                                UnqualifiedId &Result) {

  // Handle 'A::template B'. This is for template-ids which have not
  // already been annotated by ParseOptionalCXXScopeSpecifier().
  bool TemplateSpecified = false;
  SourceLocation TemplateKWLoc;
  if (getLang().CPlusPlus && Tok.is(tok::kw_template) &&
      (ObjectType || SS.isSet())) {
    TemplateSpecified = true;
    TemplateKWLoc = ConsumeToken();
  }

  // unqualified-id:
  //   identifier
  //   template-id (when it hasn't already been annotated)
  if (Tok.is(tok::identifier)) {
    // Consume the identifier.
    IdentifierInfo *Id = Tok.getIdentifierInfo();
    SourceLocation IdLoc = ConsumeToken();

    if (!getLang().CPlusPlus) {
      // If we're not in C++, only identifiers matter. Record the
      // identifier and return.
      Result.setIdentifier(Id, IdLoc);
      return false;
    }

    if (AllowConstructorName && 
        Actions.isCurrentClassName(*Id, getCurScope(), &SS)) {
      // We have parsed a constructor name.
      Result.setConstructorName(Actions.getTypeName(*Id, IdLoc, getCurScope(),
                                                    &SS, false, false,
                                                    ParsedType(),
                                            /*NonTrivialTypeSourceInfo=*/true),
                                IdLoc, IdLoc);
    } else {
      // We have parsed an identifier.
      Result.setIdentifier(Id, IdLoc);      
    }

    // If the next token is a '<', we may have a template.
    if (TemplateSpecified || Tok.is(tok::less))
      return ParseUnqualifiedIdTemplateId(SS, Id, IdLoc, EnteringContext, 
                                          ObjectType, Result,
                                          TemplateSpecified, TemplateKWLoc);
    
    return false;
  }
  
  // unqualified-id:
  //   template-id (already parsed and annotated)
  if (Tok.is(tok::annot_template_id)) {
    TemplateIdAnnotation *TemplateId
      = static_cast<TemplateIdAnnotation*>(Tok.getAnnotationValue());

    // If the template-name names the current class, then this is a constructor 
    if (AllowConstructorName && TemplateId->Name &&
        Actions.isCurrentClassName(*TemplateId->Name, getCurScope(), &SS)) {
      if (SS.isSet()) {
        // C++ [class.qual]p2 specifies that a qualified template-name
        // is taken as the constructor name where a constructor can be
        // declared. Thus, the template arguments are extraneous, so
        // complain about them and remove them entirely.
        Diag(TemplateId->TemplateNameLoc, 
             diag::err_out_of_line_constructor_template_id)
          << TemplateId->Name
          << FixItHint::CreateRemoval(
                    SourceRange(TemplateId->LAngleLoc, TemplateId->RAngleLoc));
        Result.setConstructorName(Actions.getTypeName(*TemplateId->Name,
                                                  TemplateId->TemplateNameLoc, 
                                                      getCurScope(),
                                                      &SS, false, false,
                                                      ParsedType(),
                                            /*NontrivialTypeSourceInfo=*/true),
                                  TemplateId->TemplateNameLoc, 
                                  TemplateId->RAngleLoc);
        TemplateId->Destroy();
        ConsumeToken();
        return false;
      }

      Result.setConstructorTemplateId(TemplateId);
      ConsumeToken();
      return false;
    }

    // We have already parsed a template-id; consume the annotation token as
    // our unqualified-id.
    Result.setTemplateId(TemplateId);
    ConsumeToken();
    return false;
  }
  
  // unqualified-id:
  //   operator-function-id
  //   conversion-function-id
  if (Tok.is(tok::kw_operator)) {
    if (ParseUnqualifiedIdOperator(SS, EnteringContext, ObjectType, Result))
      return true;
    
    // If we have an operator-function-id or a literal-operator-id and the next
    // token is a '<', we may have a
    // 
    //   template-id:
    //     operator-function-id < template-argument-list[opt] >
    if ((Result.getKind() == UnqualifiedId::IK_OperatorFunctionId ||
         Result.getKind() == UnqualifiedId::IK_LiteralOperatorId) &&
        (TemplateSpecified || Tok.is(tok::less)))
      return ParseUnqualifiedIdTemplateId(SS, 0, SourceLocation(), 
                                          EnteringContext, ObjectType, 
                                          Result,
                                          TemplateSpecified, TemplateKWLoc);
    
    return false;
  }
  
  if (getLang().CPlusPlus && 
      (AllowDestructorName || SS.isSet()) && Tok.is(tok::tilde)) {
    // C++ [expr.unary.op]p10:
    //   There is an ambiguity in the unary-expression ~X(), where X is a 
    //   class-name. The ambiguity is resolved in favor of treating ~ as a 
    //    unary complement rather than treating ~X as referring to a destructor.
    
    // Parse the '~'.
    SourceLocation TildeLoc = ConsumeToken();
    
    // Parse the class-name.
    if (Tok.isNot(tok::identifier)) {
      Diag(Tok, diag::err_destructor_tilde_identifier);
      return true;
    }

    // Parse the class-name (or template-name in a simple-template-id).
    IdentifierInfo *ClassName = Tok.getIdentifierInfo();
    SourceLocation ClassNameLoc = ConsumeToken();
    
    if (TemplateSpecified || Tok.is(tok::less)) {
      Result.setDestructorName(TildeLoc, ParsedType(), ClassNameLoc);
      return ParseUnqualifiedIdTemplateId(SS, ClassName, ClassNameLoc,
                                          EnteringContext, ObjectType, Result,
                                          TemplateSpecified, TemplateKWLoc);
    }
    
    // Note that this is a destructor name.
    ParsedType Ty = Actions.getDestructorName(TildeLoc, *ClassName, 
                                              ClassNameLoc, getCurScope(),
                                              SS, ObjectType,
                                              EnteringContext);
    if (!Ty)
      return true;

    Result.setDestructorName(TildeLoc, Ty, ClassNameLoc);
    return false;
  }
  
  Diag(Tok, diag::err_expected_unqualified_id)
    << getLang().CPlusPlus;
  return true;
}

/// ParseCXXNewExpression - Parse a C++ new-expression. New is used to allocate
/// memory in a typesafe manner and call constructors.
///
/// This method is called to parse the new expression after the optional :: has
/// been already parsed.  If the :: was present, "UseGlobal" is true and "Start"
/// is its location.  Otherwise, "Start" is the location of the 'new' token.
///
///        new-expression:
///                   '::'[opt] 'new' new-placement[opt] new-type-id
///                                     new-initializer[opt]
///                   '::'[opt] 'new' new-placement[opt] '(' type-id ')'
///                                     new-initializer[opt]
///
///        new-placement:
///                   '(' expression-list ')'
///
///        new-type-id:
///                   type-specifier-seq new-declarator[opt]
/// [GNU]             attributes type-specifier-seq new-declarator[opt]
///
///        new-declarator:
///                   ptr-operator new-declarator[opt]
///                   direct-new-declarator
///
///        new-initializer:
///                   '(' expression-list[opt] ')'
/// [C++0x]           braced-init-list                                   [TODO]
///
ExprResult
Parser::ParseCXXNewExpression(bool UseGlobal, SourceLocation Start) {
  assert(Tok.is(tok::kw_new) && "expected 'new' token");
  ConsumeToken();   // Consume 'new'

  // A '(' now can be a new-placement or the '(' wrapping the type-id in the
  // second form of new-expression. It can't be a new-type-id.

  ExprVector PlacementArgs(Actions);
  SourceLocation PlacementLParen, PlacementRParen;

  SourceRange TypeIdParens;
  DeclSpec DS(AttrFactory);
  Declarator DeclaratorInfo(DS, Declarator::TypeNameContext);
  if (Tok.is(tok::l_paren)) {
    // If it turns out to be a placement, we change the type location.
    PlacementLParen = ConsumeParen();
    if (ParseExpressionListOrTypeId(PlacementArgs, DeclaratorInfo)) {
      SkipUntil(tok::semi, /*StopAtSemi=*/true, /*DontConsume=*/true);
      return ExprError();
    }

    PlacementRParen = MatchRHSPunctuation(tok::r_paren, PlacementLParen);
    if (PlacementRParen.isInvalid()) {
      SkipUntil(tok::semi, /*StopAtSemi=*/true, /*DontConsume=*/true);
      return ExprError();
    }

    if (PlacementArgs.empty()) {
      // Reset the placement locations. There was no placement.
      TypeIdParens = SourceRange(PlacementLParen, PlacementRParen);
      PlacementLParen = PlacementRParen = SourceLocation();
    } else {
      // We still need the type.
      if (Tok.is(tok::l_paren)) {
        TypeIdParens.setBegin(ConsumeParen());
        MaybeParseGNUAttributes(DeclaratorInfo);
        ParseSpecifierQualifierList(DS);
        DeclaratorInfo.SetSourceRange(DS.getSourceRange());
        ParseDeclarator(DeclaratorInfo);
        TypeIdParens.setEnd(MatchRHSPunctuation(tok::r_paren, 
                                                TypeIdParens.getBegin()));
      } else {
        MaybeParseGNUAttributes(DeclaratorInfo);
        if (ParseCXXTypeSpecifierSeq(DS))
          DeclaratorInfo.setInvalidType(true);
        else {
          DeclaratorInfo.SetSourceRange(DS.getSourceRange());
          ParseDeclaratorInternal(DeclaratorInfo,
                                  &Parser::ParseDirectNewDeclarator);
        }
      }
    }
  } else {
    // A new-type-id is a simplified type-id, where essentially the
    // direct-declarator is replaced by a direct-new-declarator.
    MaybeParseGNUAttributes(DeclaratorInfo);
    if (ParseCXXTypeSpecifierSeq(DS))
      DeclaratorInfo.setInvalidType(true);
    else {
      DeclaratorInfo.SetSourceRange(DS.getSourceRange());
      ParseDeclaratorInternal(DeclaratorInfo,
                              &Parser::ParseDirectNewDeclarator);
    }
  }
  if (DeclaratorInfo.isInvalidType()) {
    SkipUntil(tok::semi, /*StopAtSemi=*/true, /*DontConsume=*/true);
    return ExprError();
  }

  ExprVector ConstructorArgs(Actions);
  SourceLocation ConstructorLParen, ConstructorRParen;

  if (Tok.is(tok::l_paren)) {
    ConstructorLParen = ConsumeParen();
    if (Tok.isNot(tok::r_paren)) {
      CommaLocsTy CommaLocs;
      if (ParseExpressionList(ConstructorArgs, CommaLocs)) {
        SkipUntil(tok::semi, /*StopAtSemi=*/true, /*DontConsume=*/true);
        return ExprError();
      }
    }
    ConstructorRParen = MatchRHSPunctuation(tok::r_paren, ConstructorLParen);
    if (ConstructorRParen.isInvalid()) {
      SkipUntil(tok::semi, /*StopAtSemi=*/true, /*DontConsume=*/true);
      return ExprError();
    }
  }

  return Actions.ActOnCXXNew(Start, UseGlobal, PlacementLParen,
                             move_arg(PlacementArgs), PlacementRParen,
                             TypeIdParens, DeclaratorInfo, ConstructorLParen,
                             move_arg(ConstructorArgs), ConstructorRParen);
}

/// ParseDirectNewDeclarator - Parses a direct-new-declarator. Intended to be
/// passed to ParseDeclaratorInternal.
///
///        direct-new-declarator:
///                   '[' expression ']'
///                   direct-new-declarator '[' constant-expression ']'
///
void Parser::ParseDirectNewDeclarator(Declarator &D) {
  // Parse the array dimensions.
  bool first = true;
  while (Tok.is(tok::l_square)) {
    SourceLocation LLoc = ConsumeBracket();
    ExprResult Size(first ? ParseExpression()
                                : ParseConstantExpression());
    if (Size.isInvalid()) {
      // Recover
      SkipUntil(tok::r_square);
      return;
    }
    first = false;

    SourceLocation RLoc = MatchRHSPunctuation(tok::r_square, LLoc);

    ParsedAttributes attrs(AttrFactory);
    D.AddTypeInfo(DeclaratorChunk::getArray(0,
                                            /*static=*/false, /*star=*/false,
                                            Size.release(), LLoc, RLoc),
                  attrs, RLoc);

    if (RLoc.isInvalid())
      return;
  }
}

/// ParseExpressionListOrTypeId - Parse either an expression-list or a type-id.
/// This ambiguity appears in the syntax of the C++ new operator.
///
///        new-expression:
///                   '::'[opt] 'new' new-placement[opt] '(' type-id ')'
///                                     new-initializer[opt]
///
///        new-placement:
///                   '(' expression-list ')'
///
bool Parser::ParseExpressionListOrTypeId(
                                   llvm::SmallVectorImpl<Expr*> &PlacementArgs,
                                         Declarator &D) {
  // The '(' was already consumed.
  if (isTypeIdInParens()) {
    ParseSpecifierQualifierList(D.getMutableDeclSpec());
    D.SetSourceRange(D.getDeclSpec().getSourceRange());
    ParseDeclarator(D);
    return D.isInvalidType();
  }

  // It's not a type, it has to be an expression list.
  // Discard the comma locations - ActOnCXXNew has enough parameters.
  CommaLocsTy CommaLocs;
  return ParseExpressionList(PlacementArgs, CommaLocs);
}

/// ParseCXXDeleteExpression - Parse a C++ delete-expression. Delete is used
/// to free memory allocated by new.
///
/// This method is called to parse the 'delete' expression after the optional
/// '::' has been already parsed.  If the '::' was present, "UseGlobal" is true
/// and "Start" is its location.  Otherwise, "Start" is the location of the
/// 'delete' token.
///
///        delete-expression:
///                   '::'[opt] 'delete' cast-expression
///                   '::'[opt] 'delete' '[' ']' cast-expression
ExprResult
Parser::ParseCXXDeleteExpression(bool UseGlobal, SourceLocation Start) {
  assert(Tok.is(tok::kw_delete) && "Expected 'delete' keyword");
  ConsumeToken(); // Consume 'delete'

  // Array delete?
  bool ArrayDelete = false;
  if (Tok.is(tok::l_square)) {
    ArrayDelete = true;
    SourceLocation LHS = ConsumeBracket();
    SourceLocation RHS = MatchRHSPunctuation(tok::r_square, LHS);
    if (RHS.isInvalid())
      return ExprError();
  }

  ExprResult Operand(ParseCastExpression(false));
  if (Operand.isInvalid())
    return move(Operand);

  return Actions.ActOnCXXDelete(Start, UseGlobal, ArrayDelete, Operand.take());
}

static UnaryTypeTrait UnaryTypeTraitFromTokKind(tok::TokenKind kind) {
  switch(kind) {
  default: assert(false && "Not a known unary type trait.");
  case tok::kw___has_nothrow_assign:      return UTT_HasNothrowAssign;
  case tok::kw___has_nothrow_constructor: return UTT_HasNothrowConstructor;
  case tok::kw___has_nothrow_copy:           return UTT_HasNothrowCopy;
  case tok::kw___has_trivial_assign:      return UTT_HasTrivialAssign;
  case tok::kw___has_trivial_constructor: return UTT_HasTrivialConstructor;
  case tok::kw___has_trivial_copy:           return UTT_HasTrivialCopy;
  case tok::kw___has_trivial_destructor:  return UTT_HasTrivialDestructor;
  case tok::kw___has_virtual_destructor:  return UTT_HasVirtualDestructor;
  case tok::kw___is_abstract:             return UTT_IsAbstract;
  case tok::kw___is_arithmetic:              return UTT_IsArithmetic;
  case tok::kw___is_array:                   return UTT_IsArray;
  case tok::kw___is_class:                return UTT_IsClass;
  case tok::kw___is_complete_type:           return UTT_IsCompleteType;
  case tok::kw___is_compound:                return UTT_IsCompound;
  case tok::kw___is_const:                   return UTT_IsConst;
  case tok::kw___is_empty:                return UTT_IsEmpty;
  case tok::kw___is_enum:                 return UTT_IsEnum;
  case tok::kw___is_floating_point:          return UTT_IsFloatingPoint;
  case tok::kw___is_function:                return UTT_IsFunction;
  case tok::kw___is_fundamental:             return UTT_IsFundamental;
  case tok::kw___is_integral:                return UTT_IsIntegral;
  case tok::kw___is_lvalue_expr:             return UTT_IsLvalueExpr;
  case tok::kw___is_lvalue_reference:        return UTT_IsLvalueReference;
  case tok::kw___is_member_function_pointer: return UTT_IsMemberFunctionPointer;
  case tok::kw___is_member_object_pointer:   return UTT_IsMemberObjectPointer;
  case tok::kw___is_member_pointer:          return UTT_IsMemberPointer;
  case tok::kw___is_object:                  return UTT_IsObject;
  case tok::kw___is_literal:              return UTT_IsLiteral;
  case tok::kw___is_literal_type:         return UTT_IsLiteral;
  case tok::kw___is_pod:                  return UTT_IsPOD;
  case tok::kw___is_pointer:                 return UTT_IsPointer;
  case tok::kw___is_polymorphic:          return UTT_IsPolymorphic;
  case tok::kw___is_reference:               return UTT_IsReference;
  case tok::kw___is_rvalue_expr:             return UTT_IsRvalueExpr;
  case tok::kw___is_rvalue_reference:        return UTT_IsRvalueReference;
  case tok::kw___is_scalar:                  return UTT_IsScalar;
  case tok::kw___is_signed:                  return UTT_IsSigned;
  case tok::kw___is_standard_layout:         return UTT_IsStandardLayout;
  case tok::kw___is_trivial:                 return UTT_IsTrivial;
  case tok::kw___is_union:                return UTT_IsUnion;
  case tok::kw___is_unsigned:                return UTT_IsUnsigned;
  case tok::kw___is_void:                    return UTT_IsVoid;
  case tok::kw___is_volatile:                return UTT_IsVolatile;
  }
}

static BinaryTypeTrait BinaryTypeTraitFromTokKind(tok::TokenKind kind) {
  switch(kind) {
  default: llvm_unreachable("Not a known binary type trait");
  case tok::kw___is_base_of:                 return BTT_IsBaseOf;
  case tok::kw___is_convertible:             return BTT_IsConvertible;
  case tok::kw___is_same:                    return BTT_IsSame;
  case tok::kw___builtin_types_compatible_p: return BTT_TypeCompatible;
  case tok::kw___is_convertible_to:          return BTT_IsConvertibleTo;
  }
}

static ArrayTypeTrait ArrayTypeTraitFromTokKind(tok::TokenKind kind) {
  switch(kind) {
  default: llvm_unreachable("Not a known binary type trait");
  case tok::kw___array_rank:                 return ATT_ArrayRank;
  case tok::kw___array_extent:               return ATT_ArrayExtent;
  }
}

static ExpressionTrait ExpressionTraitFromTokKind(tok::TokenKind kind) {
  switch(kind) {
  default: assert(false && "Not a known unary expression trait.");
  case tok::kw___is_lvalue_expr:             return ET_IsLValueExpr;
  case tok::kw___is_rvalue_expr:             return ET_IsRValueExpr;
  }
}

/// ParseUnaryTypeTrait - Parse the built-in unary type-trait
/// pseudo-functions that allow implementation of the TR1/C++0x type traits
/// templates.
///
///       primary-expression:
/// [GNU]             unary-type-trait '(' type-id ')'
///
ExprResult Parser::ParseUnaryTypeTrait() {
  UnaryTypeTrait UTT = UnaryTypeTraitFromTokKind(Tok.getKind());
  SourceLocation Loc = ConsumeToken();

  SourceLocation LParen = Tok.getLocation();
  if (ExpectAndConsume(tok::l_paren, diag::err_expected_lparen))
    return ExprError();

  // FIXME: Error reporting absolutely sucks! If the this fails to parse a type
  // there will be cryptic errors about mismatched parentheses and missing
  // specifiers.
  TypeResult Ty = ParseTypeName();

  SourceLocation RParen = MatchRHSPunctuation(tok::r_paren, LParen);

  if (Ty.isInvalid())
    return ExprError();

  return Actions.ActOnUnaryTypeTrait(UTT, Loc, Ty.get(), RParen);
}

/// ParseBinaryTypeTrait - Parse the built-in binary type-trait
/// pseudo-functions that allow implementation of the TR1/C++0x type traits
/// templates.
///
///       primary-expression:
/// [GNU]             binary-type-trait '(' type-id ',' type-id ')'
///
ExprResult Parser::ParseBinaryTypeTrait() {
  BinaryTypeTrait BTT = BinaryTypeTraitFromTokKind(Tok.getKind());
  SourceLocation Loc = ConsumeToken();

  SourceLocation LParen = Tok.getLocation();
  if (ExpectAndConsume(tok::l_paren, diag::err_expected_lparen))
    return ExprError();

  TypeResult LhsTy = ParseTypeName();
  if (LhsTy.isInvalid()) {
    SkipUntil(tok::r_paren);
    return ExprError();
  }

  if (ExpectAndConsume(tok::comma, diag::err_expected_comma)) {
    SkipUntil(tok::r_paren);
    return ExprError();
  }

  TypeResult RhsTy = ParseTypeName();
  if (RhsTy.isInvalid()) {
    SkipUntil(tok::r_paren);
    return ExprError();
  }

  SourceLocation RParen = MatchRHSPunctuation(tok::r_paren, LParen);

  return Actions.ActOnBinaryTypeTrait(BTT, Loc, LhsTy.get(), RhsTy.get(), RParen);
}

/// ParseArrayTypeTrait - Parse the built-in array type-trait
/// pseudo-functions.
///
///       primary-expression:
/// [Embarcadero]     '__array_rank' '(' type-id ')'
/// [Embarcadero]     '__array_extent' '(' type-id ',' expression ')'
///
ExprResult Parser::ParseArrayTypeTrait() {
  ArrayTypeTrait ATT = ArrayTypeTraitFromTokKind(Tok.getKind());
  SourceLocation Loc = ConsumeToken();

  SourceLocation LParen = Tok.getLocation();
  if (ExpectAndConsume(tok::l_paren, diag::err_expected_lparen))
    return ExprError();

  TypeResult Ty = ParseTypeName();
  if (Ty.isInvalid()) {
    SkipUntil(tok::comma);
    SkipUntil(tok::r_paren);
    return ExprError();
  }

  switch (ATT) {
  case ATT_ArrayRank: {
    SourceLocation RParen = MatchRHSPunctuation(tok::r_paren, LParen);
    return Actions.ActOnArrayTypeTrait(ATT, Loc, Ty.get(), NULL, RParen);
  }
  case ATT_ArrayExtent: {
    if (ExpectAndConsume(tok::comma, diag::err_expected_comma)) {
      SkipUntil(tok::r_paren);
      return ExprError();
    }

    ExprResult DimExpr = ParseExpression();
    SourceLocation RParen = MatchRHSPunctuation(tok::r_paren, LParen);

    return Actions.ActOnArrayTypeTrait(ATT, Loc, Ty.get(), DimExpr.get(), RParen);
  }
  default:
    break;
  }
  return ExprError();
}

/// ParseExpressionTrait - Parse built-in expression-trait
/// pseudo-functions like __is_lvalue_expr( xxx ).
///
///       primary-expression:
/// [Embarcadero]     expression-trait '(' expression ')'
///
ExprResult Parser::ParseExpressionTrait() {
  ExpressionTrait ET = ExpressionTraitFromTokKind(Tok.getKind());
  SourceLocation Loc = ConsumeToken();

  SourceLocation LParen = Tok.getLocation();
  if (ExpectAndConsume(tok::l_paren, diag::err_expected_lparen))
    return ExprError();

  ExprResult Expr = ParseExpression();

  SourceLocation RParen = MatchRHSPunctuation(tok::r_paren, LParen);

  return Actions.ActOnExpressionTrait(ET, Loc, Expr.get(), RParen);
}


/// ParseCXXAmbiguousParenExpression - We have parsed the left paren of a
/// parenthesized ambiguous type-id. This uses tentative parsing to disambiguate
/// based on the context past the parens.
ExprResult
Parser::ParseCXXAmbiguousParenExpression(ParenParseOption &ExprType,
                                         ParsedType &CastTy,
                                         SourceLocation LParenLoc,
                                         SourceLocation &RParenLoc) {
  assert(getLang().CPlusPlus && "Should only be called for C++!");
  assert(ExprType == CastExpr && "Compound literals are not ambiguous!");
  assert(isTypeIdInParens() && "Not a type-id!");

  ExprResult Result(true);
  CastTy = ParsedType();

  // We need to disambiguate a very ugly part of the C++ syntax:
  //
  // (T())x;  - type-id
  // (T())*x; - type-id
  // (T())/x; - expression
  // (T());   - expression
  //
  // The bad news is that we cannot use the specialized tentative parser, since
  // it can only verify that the thing inside the parens can be parsed as
  // type-id, it is not useful for determining the context past the parens.
  //
  // The good news is that the parser can disambiguate this part without
  // making any unnecessary Action calls.
  //
  // It uses a scheme similar to parsing inline methods. The parenthesized
  // tokens are cached, the context that follows is determined (possibly by
  // parsing a cast-expression), and then we re-introduce the cached tokens
  // into the token stream and parse them appropriately.

  ParenParseOption ParseAs;
  CachedTokens Toks;

  // Store the tokens of the parentheses. We will parse them after we determine
  // the context that follows them.
  if (!ConsumeAndStoreUntil(tok::r_paren, Toks)) {
    // We didn't find the ')' we expected.
    MatchRHSPunctuation(tok::r_paren, LParenLoc);
    return ExprError();
  }

  if (Tok.is(tok::l_brace)) {
    ParseAs = CompoundLiteral;
  } else {
    bool NotCastExpr;
    // FIXME: Special-case ++ and --: "(S())++;" is not a cast-expression
    if (Tok.is(tok::l_paren) && NextToken().is(tok::r_paren)) {
      NotCastExpr = true;
    } else {
      // Try parsing the cast-expression that may follow.
      // If it is not a cast-expression, NotCastExpr will be true and no token
      // will be consumed.
      Result = ParseCastExpression(false/*isUnaryExpression*/,
                                   false/*isAddressofOperand*/,
                                   NotCastExpr,
                                   ParsedType()/*TypeOfCast*/);
    }

    // If we parsed a cast-expression, it's really a type-id, otherwise it's
    // an expression.
    ParseAs = NotCastExpr ? SimpleExpr : CastExpr;
  }

  // The current token should go after the cached tokens.
  Toks.push_back(Tok);
  // Re-enter the stored parenthesized tokens into the token stream, so we may
  // parse them now.
  PP.EnterTokenStream(Toks.data(), Toks.size(),
                      true/*DisableMacroExpansion*/, false/*OwnsTokens*/);
  // Drop the current token and bring the first cached one. It's the same token
  // as when we entered this function.
  ConsumeAnyToken();

  if (ParseAs >= CompoundLiteral) {
    TypeResult Ty = ParseTypeName();

    // Match the ')'.
    if (Tok.is(tok::r_paren))
      RParenLoc = ConsumeParen();
    else
      MatchRHSPunctuation(tok::r_paren, LParenLoc);

    if (ParseAs == CompoundLiteral) {
      ExprType = CompoundLiteral;
      return ParseCompoundLiteralExpression(Ty.get(), LParenLoc, RParenLoc);
    }

    // We parsed '(' type-id ')' and the thing after it wasn't a '{'.
    assert(ParseAs == CastExpr);

    if (Ty.isInvalid())
      return ExprError();

    CastTy = Ty.get();

    // Result is what ParseCastExpression returned earlier.
    if (!Result.isInvalid())
      Result = Actions.ActOnCastExpr(getCurScope(), LParenLoc, CastTy, RParenLoc,
                                     Result.take());
    return move(Result);
  }

  // Not a compound literal, and not followed by a cast-expression.
  assert(ParseAs == SimpleExpr);

  ExprType = SimpleExpr;
  Result = ParseExpression();
  if (!Result.isInvalid() && Tok.is(tok::r_paren))
    Result = Actions.ActOnParenExpr(LParenLoc, Tok.getLocation(), Result.take());

  // Match the ')'.
  if (Result.isInvalid()) {
    SkipUntil(tok::r_paren);
    return ExprError();
  }

  if (Tok.is(tok::r_paren))
    RParenLoc = ConsumeParen();
  else
    MatchRHSPunctuation(tok::r_paren, LParenLoc);

  return move(Result);
}
