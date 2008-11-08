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
  //   conversion-function-id                [TODO]
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
    if (IdentifierInfo *II = MaybeParseOperatorFunctionId()) {
      return Actions.ActOnIdentifierExpr(CurScope, OperatorLoc, *II, 
                                         Tok.is(tok::l_paren), &SS);
    }
    // FIXME: Handle conversion-function-id.
    unsigned DiagID = PP.getDiagnostics().getCustomDiagID(Diagnostic::Error,
                                    "expected operator-function-id");
    return Diag(Tok, DiagID);
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

  if (ExpectAndConsume(tok::greater, diag::err_expected_greater)) {
    Diag(LAngleBracketLoc, diag::err_matching, "<");
    return ExprResult(true);
  }

  SourceLocation LParenLoc = Tok.getLocation(), RParenLoc;

  if (Tok.isNot(tok::l_paren)) {
    Diag(Tok, diag::err_expected_lparen_after, CastName);
    return ExprResult(true);
  }

  ExprResult Result = ParseSimpleParenExpression(RParenLoc);

  if (!Result.isInvalid)
    Result = Actions.ActOnCXXNamedCast(OpLoc, Kind,
                                       LAngleBracketLoc, CastTy, RAngleBracketLoc,
                                       LParenLoc, Result.Val, RParenLoc);

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
    Diag(Tok.getLocation(), diag::err_operator_missing_type_specifier);
    return true;
  }
  while (MaybeParseTypeSpecifier(DS, isInvalid, PrevSpec)) ;

  return false;
}

/// MaybeParseOperatorFunctionId - Attempts to parse a C++ overloaded
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
IdentifierInfo *Parser::MaybeParseOperatorFunctionId() {
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
    return &PP.getIdentifierTable().getOverloadedOperator(Op);

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
    return &PP.getIdentifierTable().getOverloadedOperator(Op);

#define OVERLOADED_OPERATOR(Name,Spelling,Token) \
    case tok::Token:  Op = OO_##Name; break;
#define OVERLOADED_OPERATOR_MULTI(Name,Spelling)
#include "clang/Basic/OperatorKinds.def"

  case tok::l_paren:
    ConsumeToken(); // 'operator'
    ConsumeParen(); // '('
    ExpectAndConsume(tok::r_paren, diag::err_expected_rparen); // ')'
    return &PP.getIdentifierTable().getOverloadedOperator(OO_Call);

  case tok::l_square:
    ConsumeToken(); // 'operator'
    ConsumeBracket(); // '['
    ExpectAndConsume(tok::r_square, diag::err_expected_rsquare); // ']'
    return &PP.getIdentifierTable().getOverloadedOperator(OO_Subscript);

  default:
    break;
  }

  if (Op == OO_None)
    return 0;
  else {
    ConsumeToken(); // 'operator'
    ConsumeAnyToken(); // the operator itself
    return &PP.getIdentifierTable().getOverloadedOperator(Op);
  }
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
  ParseDeclaratorInternal(D, /*PtrOperator=*/true);

  // Finish up the type.
  Action::TypeResult Result = Actions.ActOnTypeName(CurScope, D);
  if (Result.isInvalid)
    return 0;
  else
    return Result.Val;
}
