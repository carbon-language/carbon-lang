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

#include "clang/Basic/Diagnostic.h"
#include "clang/Parse/Parser.h"
#include "clang/Parse/DeclSpec.h"
using namespace clang;

/// ParseCXXScopeSpecifier - Parse global scope or nested-name-specifier.
///
///       '::'[opt] nested-name-specifier
///       '::'
///
///       nested-name-specifier:
///         type-name '::'
///         namespace-name '::'
///         nested-name-specifier identifier '::'
///         nested-name-specifier 'template'[opt] simple-template-id '::' [TODO]
///
void Parser::ParseCXXScopeSpecifier(CXXScopeSpec &SS) {
  assert(isTokenCXXScopeSpecifier() && "Not scope specifier!");

  if (Tok.is(tok::annot_cxxscope)) {
    SS.setScopeRep(Tok.getAnnotationValue());
    SS.setRange(Tok.getAnnotationRange());
    ConsumeToken();
    return;
  }

  SS.setBeginLoc(Tok.getLocation());

  // '::'

  if (Tok.is(tok::coloncolon)) {
    // Global scope.
    SourceLocation CCLoc = ConsumeToken();
    SS.setScopeRep(Actions.ActOnCXXGlobalScopeSpecifier(CurScope, CCLoc));
    SS.setEndLoc(CCLoc);
  }

  // nested-name-specifier:
  //   type-name '::'
  //   namespace-name '::'
  //   nested-name-specifier identifier '::'
  //   nested-name-specifier 'template'[opt] simple-template-id '::' [TODO]

  while (Tok.is(tok::identifier) && NextToken().is(tok::coloncolon)) {
    IdentifierInfo *II = Tok.getIdentifierInfo();
    SourceLocation IdLoc = ConsumeToken();
    assert(Tok.is(tok::coloncolon) &&
           "NextToken() not working properly!");
    SourceLocation CCLoc = ConsumeToken();
    if (SS.isInvalid())
      continue;

    SS.setScopeRep(
         Actions.ActOnCXXNestedNameSpecifier(CurScope, SS, IdLoc, CCLoc, *II) );
    SS.setEndLoc(CCLoc);
  }
}

/// ParseCXXIdExpression - Handle id-expression.
///
///       id-expression:
///         unqualified-id
///         qualified-id
///
///       unqualified-id:
///         identifier
///         operator-function-id
///         conversion-function-id                [TODO]
///         '~' class-name                        [TODO]
///         template-id                           [TODO]
///
///       qualified-id:
///         '::'[opt] nested-name-specifier 'template'[opt] unqualified-id
///         '::' identifier
///         '::' operator-function-id
///         '::' template-id                      [TODO]
///
///       nested-name-specifier:
///         type-name '::'
///         namespace-name '::'
///         nested-name-specifier identifier '::'
///         nested-name-specifier 'template'[opt] simple-template-id '::' [TODO]
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
Parser::ExprResult Parser::ParseCXXIdExpression() {
  // qualified-id:
  //   '::'[opt] nested-name-specifier 'template'[opt] unqualified-id
  //   '::' unqualified-id
  //
  CXXScopeSpec SS;
  if (isTokenCXXScopeSpecifier())
    ParseCXXScopeSpecifier(SS);

  // unqualified-id:
  //   identifier
  //   operator-function-id
  //   conversion-function-id
  //   '~' class-name                        [TODO]
  //   template-id                           [TODO]
  //
  switch (Tok.getKind()) {
  default:
    return Diag(Tok, diag::err_expected_unqualified_id);

  case tok::identifier: {
    // Consume the identifier so that we can see if it is followed by a '('.
    IdentifierInfo &II = *Tok.getIdentifierInfo();
    SourceLocation L = ConsumeToken();
    return Actions.ActOnIdentifierExpr(CurScope, L, II,
                                       Tok.is(tok::l_paren), &SS);
  }

  case tok::kw_operator: {
    SourceLocation OperatorLoc = Tok.getLocation();
    if (OverloadedOperatorKind Op = TryParseOperatorFunctionId()) {
      return Actions.ActOnCXXOperatorFunctionIdExpr(CurScope, OperatorLoc, Op, 
                                                    Tok.is(tok::l_paren), SS);
    } else if (TypeTy *Type = ParseConversionFunctionId()) {
      return Actions.ActOnCXXConversionFunctionExpr(CurScope, OperatorLoc,
                                                    Type, Tok.is(tok::l_paren), 
                                                    SS);
    }
     
    // We already complained about a bad conversion-function-id,
    // above.
    return true;
  }

  } // switch.

  assert(0 && "The switch was supposed to take care everything.");
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
Parser::ExprResult Parser::ParseCXXCasts() {
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

  if (ExpectAndConsume(tok::less, diag::err_expected_less_after, CastName))
    return ExprResult(true);

  TypeTy *CastTy = ParseTypeName();
  SourceLocation RAngleBracketLoc = Tok.getLocation();

  if (ExpectAndConsume(tok::greater, diag::err_expected_greater))
    return Diag(LAngleBracketLoc, diag::note_matching) << "<";

  SourceLocation LParenLoc = Tok.getLocation(), RParenLoc;

  if (Tok.isNot(tok::l_paren))
    return Diag(Tok, diag::err_expected_lparen_after) << CastName;

  ExprResult Result = ParseSimpleParenExpression(RParenLoc);

  if (!Result.isInvalid)
    Result = Actions.ActOnCXXNamedCast(OpLoc, Kind,
                                       LAngleBracketLoc, CastTy, RAngleBracketLoc,
                                       LParenLoc, Result.Val, RParenLoc);

  return Result;
}

/// ParseCXXTypeid - This handles the C++ typeid expression.
///
///       postfix-expression: [C++ 5.2p1]
///         'typeid' '(' expression ')'
///         'typeid' '(' type-id ')'
///
Parser::ExprResult Parser::ParseCXXTypeid() {
  assert(Tok.is(tok::kw_typeid) && "Not 'typeid'!");

  SourceLocation OpLoc = ConsumeToken();
  SourceLocation LParenLoc = Tok.getLocation();
  SourceLocation RParenLoc;

  // typeid expressions are always parenthesized.
  if (ExpectAndConsume(tok::l_paren, diag::err_expected_lparen_after,
      "typeid"))
    return ExprResult(true);

  Parser::ExprResult Result;

  if (isTypeIdInParens()) {
    TypeTy *Ty = ParseTypeName();

    // Match the ')'.
    MatchRHSPunctuation(tok::r_paren, LParenLoc);

    if (!Ty)
      return ExprResult(true);

    Result = Actions.ActOnCXXTypeid(OpLoc, LParenLoc, /*isType=*/true,
                                    Ty, RParenLoc);
  } else {
    Result = ParseExpression();

    // Match the ')'.
    if (Result.isInvalid)
      SkipUntil(tok::r_paren);
    else {
      MatchRHSPunctuation(tok::r_paren, LParenLoc);

      Result = Actions.ActOnCXXTypeid(OpLoc, LParenLoc, /*isType=*/false,
                                      Result.Val, RParenLoc);
    }
  }

  return Result;
}

/// ParseCXXBoolLiteral - This handles the C++ Boolean literals.
///
///       boolean-literal: [C++ 2.13.5]
///         'true'
///         'false'
Parser::ExprResult Parser::ParseCXXBoolLiteral() {
  tok::TokenKind Kind = Tok.getKind();
  return Actions.ActOnCXXBoolLiteral(ConsumeToken(), Kind);
}

/// ParseThrowExpression - This handles the C++ throw expression.
///
///       throw-expression: [C++ 15]
///         'throw' assignment-expression[opt]
Parser::ExprResult Parser::ParseThrowExpression() {
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
    return Actions.ActOnCXXThrow(ThrowLoc);

  default:
    ExprResult Expr = ParseAssignmentExpression();
    if (Expr.isInvalid) return Expr;
    return Actions.ActOnCXXThrow(ThrowLoc, Expr.Val);
  }
}

/// ParseCXXThis - This handles the C++ 'this' pointer.
///
/// C++ 9.3.2: In the body of a non-static member function, the keyword this is
/// a non-lvalue expression whose value is the address of the object for which
/// the function is called.
Parser::ExprResult Parser::ParseCXXThis() {
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
Parser::ExprResult Parser::ParseCXXTypeConstructExpression(const DeclSpec &DS) {
  Declarator DeclaratorInfo(DS, Declarator::TypeNameContext);
  TypeTy *TypeRep = Actions.ActOnTypeName(CurScope, DeclaratorInfo).Val;

  assert(Tok.is(tok::l_paren) && "Expected '('!");
  SourceLocation LParenLoc = ConsumeParen();

  ExprListTy Exprs;
  CommaLocsTy CommaLocs;

  if (Tok.isNot(tok::r_paren)) {
    if (ParseExpressionList(Exprs, CommaLocs)) {
      SkipUntil(tok::r_paren);
      return ExprResult(true);
    }
  }

  // Match the ')'.
  SourceLocation RParenLoc = MatchRHSPunctuation(tok::r_paren, LParenLoc);

  assert((Exprs.size() == 0 || Exprs.size()-1 == CommaLocs.size())&&
         "Unexpected number of commas!");
  return Actions.ActOnCXXTypeConstructExpr(DS.getSourceRange(), TypeRep,
                                           LParenLoc,
                                           &Exprs[0], Exprs.size(),
                                           &CommaLocs[0], RParenLoc);
}

/// ParseCXXCondition - if/switch/while/for condition expression.
///
///       condition:
///         expression
///         type-specifier-seq declarator '=' assignment-expression
/// [GNU]   type-specifier-seq declarator simple-asm-expr[opt] attributes[opt]
///             '=' assignment-expression
///
Parser::ExprResult Parser::ParseCXXCondition() {
  if (!isCXXConditionDeclaration())
    return ParseExpression(); // expression

  SourceLocation StartLoc = Tok.getLocation();

  // type-specifier-seq
  DeclSpec DS;
  ParseSpecifierQualifierList(DS);

  // declarator
  Declarator DeclaratorInfo(DS, Declarator::ConditionContext);
  ParseDeclarator(DeclaratorInfo);

  // simple-asm-expr[opt]
  if (Tok.is(tok::kw_asm)) {
    ExprResult AsmLabel = ParseSimpleAsm();
    if (AsmLabel.isInvalid) {
      SkipUntil(tok::semi);
      return true;
    }
    DeclaratorInfo.setAsmLabel(AsmLabel.Val);
  }

  // If attributes are present, parse them.
  if (Tok.is(tok::kw___attribute))
    DeclaratorInfo.AddAttributes(ParseAttributes());

  // '=' assignment-expression
  if (Tok.isNot(tok::equal))
    return Diag(Tok, diag::err_expected_equal_after_declarator);
  SourceLocation EqualLoc = ConsumeToken();
  ExprResult AssignExpr = ParseAssignmentExpression();
  if (AssignExpr.isInvalid)
    return true;
  
  return Actions.ActOnCXXConditionDeclarationExpr(CurScope, StartLoc,
                                                  DeclaratorInfo,
                                                  EqualLoc, AssignExpr.Val);
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
  // Annotate typenames and C++ scope specifiers.
  TryAnnotateTypeOrScopeToken();

  DS.SetRangeStart(Tok.getLocation());
  const char *PrevSpec;
  SourceLocation Loc = Tok.getLocation();
  
  switch (Tok.getKind()) {
  default: 
    assert(0 && "Not a simple-type-specifier token!");
    abort();
      
  // type-name
  case tok::annot_qualtypename: {
    DS.SetTypeSpecType(DeclSpec::TST_typedef, Loc, PrevSpec,
                       Tok.getAnnotationValue());
    break;
  }
    
  // builtin types
  case tok::kw_short:
    DS.SetTypeSpecWidth(DeclSpec::TSW_short, Loc, PrevSpec);
    break;
  case tok::kw_long:
    DS.SetTypeSpecWidth(DeclSpec::TSW_long, Loc, PrevSpec);
    break;
  case tok::kw_signed:
    DS.SetTypeSpecSign(DeclSpec::TSS_signed, Loc, PrevSpec);
    break;
  case tok::kw_unsigned:
    DS.SetTypeSpecSign(DeclSpec::TSS_unsigned, Loc, PrevSpec);
    break;
  case tok::kw_void:
    DS.SetTypeSpecType(DeclSpec::TST_void, Loc, PrevSpec);
    break;
  case tok::kw_char:
    DS.SetTypeSpecType(DeclSpec::TST_char, Loc, PrevSpec);
    break;
  case tok::kw_int:
    DS.SetTypeSpecType(DeclSpec::TST_int, Loc, PrevSpec);
    break;
  case tok::kw_float:
    DS.SetTypeSpecType(DeclSpec::TST_float, Loc, PrevSpec);
    break;
  case tok::kw_double:
    DS.SetTypeSpecType(DeclSpec::TST_double, Loc, PrevSpec);
    break;
  case tok::kw_wchar_t:
    DS.SetTypeSpecType(DeclSpec::TST_wchar, Loc, PrevSpec);
    break;
  case tok::kw_bool:
    DS.SetTypeSpecType(DeclSpec::TST_bool, Loc, PrevSpec);
    break;
  
  // GNU typeof support.
  case tok::kw_typeof:
    ParseTypeofSpecifier(DS);
    DS.Finish(Diags, PP.getSourceManager(), getLang());
    return;
  }
  if (Tok.is(tok::annot_qualtypename))
    DS.SetRangeEnd(Tok.getAnnotationEndLoc());
  else
    DS.SetRangeEnd(Tok.getLocation());
  ConsumeToken();
  DS.Finish(Diags, PP.getSourceManager(), getLang());
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
  int isInvalid = 0;

  // Parse one or more of the type specifiers.
  if (!MaybeParseTypeSpecifier(DS, isInvalid, PrevSpec)) {
    Diag(Tok, diag::err_operator_missing_type_specifier);
    return true;
  }
  while (MaybeParseTypeSpecifier(DS, isInvalid, PrevSpec)) ;

  return false;
}

/// TryParseOperatorFunctionId - Attempts to parse a C++ overloaded
/// operator name (C++ [over.oper]). If successful, returns the
/// predefined identifier that corresponds to that overloaded
/// operator. Otherwise, returns NULL and does not consume any tokens.
///
///       operator-function-id: [C++ 13.5]
///         'operator' operator
///
/// operator: one of
///            new   delete  new[]   delete[]
///            +     -    *  /    %  ^    &   |   ~
///            !     =    <  >    += -=   *=  /=  %=
///            ^=    &=   |= <<   >> >>= <<=  ==  !=
///            <=    >=   && ||   ++ --   ,   ->* ->
///            ()    []
OverloadedOperatorKind Parser::TryParseOperatorFunctionId() {
  assert(Tok.is(tok::kw_operator) && "Expected 'operator' keyword");

  OverloadedOperatorKind Op = OO_None;
  switch (NextToken().getKind()) {
  case tok::kw_new:
    ConsumeToken(); // 'operator'
    ConsumeToken(); // 'new'
    if (Tok.is(tok::l_square)) {
      ConsumeBracket(); // '['
      ExpectAndConsume(tok::r_square, diag::err_expected_rsquare); // ']'
      Op = OO_Array_New;
    } else {
      Op = OO_New;
    }
    return Op;

  case tok::kw_delete:
    ConsumeToken(); // 'operator'
    ConsumeToken(); // 'delete'
    if (Tok.is(tok::l_square)) {
      ConsumeBracket(); // '['
      ExpectAndConsume(tok::r_square, diag::err_expected_rsquare); // ']'
      Op = OO_Array_Delete;
    } else {
      Op = OO_Delete;
    }
    return Op;

#define OVERLOADED_OPERATOR(Name,Spelling,Token,Unary,Binary,MemberOnly)  \
    case tok::Token:  Op = OO_##Name; break;
#define OVERLOADED_OPERATOR_MULTI(Name,Spelling,Unary,Binary,MemberOnly)
#include "clang/Basic/OperatorKinds.def"

  case tok::l_paren:
    ConsumeToken(); // 'operator'
    ConsumeParen(); // '('
    ExpectAndConsume(tok::r_paren, diag::err_expected_rparen); // ')'
    return OO_Call;

  case tok::l_square:
    ConsumeToken(); // 'operator'
    ConsumeBracket(); // '['
    ExpectAndConsume(tok::r_square, diag::err_expected_rsquare); // ']'
    return OO_Subscript;

  default:
    return OO_None;
  }

  ConsumeToken(); // 'operator'
  ConsumeAnyToken(); // the operator itself
  return Op;
}

/// ParseConversionFunctionId - Parse a C++ conversion-function-id,
/// which expresses the name of a user-defined conversion operator
/// (C++ [class.conv.fct]p1). Returns the type that this operator is
/// specifying a conversion for, or NULL if there was an error.
///
///        conversion-function-id: [C++ 12.3.2]
///                   operator conversion-type-id
///
///        conversion-type-id:
///                   type-specifier-seq conversion-declarator[opt]
///
///        conversion-declarator:
///                   ptr-operator conversion-declarator[opt]
Parser::TypeTy *Parser::ParseConversionFunctionId() {
  assert(Tok.is(tok::kw_operator) && "Expected 'operator' keyword");
  ConsumeToken(); // 'operator'

  // Parse the type-specifier-seq.
  DeclSpec DS;
  if (ParseCXXTypeSpecifierSeq(DS))
    return 0;

  // Parse the conversion-declarator, which is merely a sequence of
  // ptr-operators.
  Declarator D(DS, Declarator::TypeNameContext);
  ParseDeclaratorInternal(D, /*DirectDeclParser=*/0);

  // Finish up the type.
  Action::TypeResult Result = Actions.ActOnTypeName(CurScope, D);
  if (Result.isInvalid)
    return 0;
  else
    return Result.Val;
}

/// ParseCXXNewExpression - Parse a C++ new-expression. New is used to allocate
/// memory in a typesafe manner and call constructors.
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
///        new-initializer:
///                   '(' expression-list[opt] ')'
/// [C++0x]           braced-init-list                                   [TODO]
///
Parser::ExprResult Parser::ParseCXXNewExpression()
{
  assert((Tok.is(tok::coloncolon) || Tok.is(tok::kw_new)) &&
         "Expected :: or 'new' keyword");

  SourceLocation Start = Tok.getLocation();
  bool UseGlobal = false;
  if (Tok.is(tok::coloncolon)) {
    UseGlobal = true;
    ConsumeToken();
  }

  assert(Tok.is(tok::kw_new) && "Lookahead should have ensured 'new'");
  // Consume 'new'
  ConsumeToken();

  // A '(' now can be a new-placement or the '(' wrapping the type-id in the
  // second form of new-expression. It can't be a new-type-id.

  ExprListTy PlacementArgs;
  SourceLocation PlacementLParen, PlacementRParen;

  TypeTy *Ty = 0;
  SourceLocation TyStart, TyEnd;
  bool ParenTypeId;
  if (Tok.is(tok::l_paren)) {
    // If it turns out to be a placement, we change the type location.
    PlacementLParen = ConsumeParen();
    TyStart = Tok.getLocation();
    if (ParseExpressionListOrTypeId(PlacementArgs, Ty))
      return true;
    TyEnd = Tok.getLocation();

    PlacementRParen = MatchRHSPunctuation(tok::r_paren, PlacementLParen);
    if (PlacementRParen.isInvalid())
      return true;

    if (Ty) {
      // Reset the placement locations. There was no placement.
      PlacementLParen = PlacementRParen = SourceLocation();
      ParenTypeId = true;
    } else {
      // We still need the type.
      if (Tok.is(tok::l_paren)) {
        ConsumeParen();
        TyStart = Tok.getLocation();
        Ty = ParseTypeName(/*CXXNewMode=*/true);
        ParenTypeId = true;
      } else {
        TyStart = Tok.getLocation();
        Ty = ParseNewTypeId();
        ParenTypeId = false;
      }
      if (!Ty)
        return true;
      TyEnd = Tok.getLocation();
    }
  } else {
    TyStart = Tok.getLocation();
    Ty = ParseNewTypeId();
    if (!Ty)
      return true;
    TyEnd = Tok.getLocation();
    ParenTypeId = false;
  }

  ExprListTy ConstructorArgs;
  SourceLocation ConstructorLParen, ConstructorRParen;

  if (Tok.is(tok::l_paren)) {
    ConstructorLParen = ConsumeParen();
    if (Tok.isNot(tok::r_paren)) {
      CommaLocsTy CommaLocs;
      if (ParseExpressionList(ConstructorArgs, CommaLocs))
        return true;
    }
    ConstructorRParen = MatchRHSPunctuation(tok::r_paren, ConstructorLParen);
    if (ConstructorRParen.isInvalid())
      return true;
  }

  return Actions.ActOnCXXNew(Start, UseGlobal, PlacementLParen,
                             &PlacementArgs[0], PlacementArgs.size(),
                             PlacementRParen, ParenTypeId, TyStart, Ty, TyEnd,
                             ConstructorLParen, &ConstructorArgs[0],
                             ConstructorArgs.size(), ConstructorRParen);
}

/// ParseNewTypeId - Parses a type ID as it appears in a new expression.
/// The most interesting part of this is the new-declarator, which can be a
/// multi-dimensional array, of which the first has a non-constant expression as
/// the size, e.g.
/// @code new int[runtimeSize()][2][2] @endcode
///
///        new-type-id:
///                   type-specifier-seq new-declarator[opt]
///
///        new-declarator:
///                   ptr-operator new-declarator[opt]
///                   direct-new-declarator
///
Parser::TypeTy * Parser::ParseNewTypeId()
{
  DeclSpec DS;
  if (ParseCXXTypeSpecifierSeq(DS))
    return 0;

  // A new-declarator is a simplified version of a declarator. We use
  // ParseDeclaratorInternal, but pass our own direct declarator parser,
  // one that parses a direct-new-declarator.
  Declarator DeclaratorInfo(DS, Declarator::TypeNameContext);
  ParseDeclaratorInternal(DeclaratorInfo, &Parser::ParseDirectNewDeclarator);

  TypeTy *Ty = Actions.ActOnTypeName(CurScope, DeclaratorInfo,
                                     /*CXXNewMode=*/true).Val;
  return DeclaratorInfo.getInvalidType() ? 0 : Ty;
}

/// ParseDirectNewDeclarator - Parses a direct-new-declarator. Intended to be
/// passed to ParseDeclaratorInternal.
///
///        direct-new-declarator:
///                   '[' expression ']'
///                   direct-new-declarator '[' constant-expression ']'
///
void Parser::ParseDirectNewDeclarator(Declarator &D)
{
  // Parse the array dimensions.
  bool first = true;
  while (Tok.is(tok::l_square)) {
    SourceLocation LLoc = ConsumeBracket();
    ExprResult Size = first ? ParseExpression() : ParseConstantExpression();
    if (Size.isInvalid) {
      // Recover
      SkipUntil(tok::r_square);
      return;
    }
    first = false;

    D.AddTypeInfo(DeclaratorChunk::getArray(0, /*static=*/false, /*star=*/false,
                                            Size.Val, LLoc));

    if (MatchRHSPunctuation(tok::r_square, LLoc).isInvalid())
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
bool Parser::ParseExpressionListOrTypeId(ExprListTy &PlacementArgs, TypeTy *&Ty)
{
  // The '(' was already consumed.
  if (isTypeIdInParens()) {
    Ty = ParseTypeName(/*CXXNewMode=*/true);
    return Ty == 0;
  }

  // It's not a type, it has to be an expression list.
  // Discard the comma locations - ActOnCXXNew has enough parameters.
  CommaLocsTy CommaLocs;
  return ParseExpressionList(PlacementArgs, CommaLocs);
}

/// ParseCXXDeleteExpression - Parse a C++ delete-expression. Delete is used
/// to free memory allocated by new.
///
///        delete-expression:
///                   '::'[opt] 'delete' cast-expression
///                   '::'[opt] 'delete' '[' ']' cast-expression
Parser::ExprResult Parser::ParseCXXDeleteExpression()
{
  assert((Tok.is(tok::coloncolon) || Tok.is(tok::kw_delete)) &&
         "Expected :: or 'delete' keyword");

  SourceLocation Start = Tok.getLocation();
  bool UseGlobal = false;
  if (Tok.is(tok::coloncolon)) {
    UseGlobal = true;
    ConsumeToken();
  }

  assert(Tok.is(tok::kw_delete) && "Lookahead should have ensured 'delete'");
  // Consume 'delete'
  ConsumeToken();

  // Array delete?
  bool ArrayDelete = false;
  if (Tok.is(tok::l_square)) {
    ArrayDelete = true;
    SourceLocation LHS = ConsumeBracket();
    SourceLocation RHS = MatchRHSPunctuation(tok::r_square, LHS);
    if (RHS.isInvalid())
      return true;
  }

  ExprResult Operand = ParseCastExpression(false);
  if (Operand.isInvalid)
    return Operand;

  return Actions.ActOnCXXDelete(Start, UseGlobal, ArrayDelete, Operand.Val);
}
