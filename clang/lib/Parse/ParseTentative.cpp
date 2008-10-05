//===--- ParseTentative.cpp - Ambiguity Resolution Parsing ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the tentative parsing portions of the Parser
//  interfaces, for ambiguity resolution.
//
//===----------------------------------------------------------------------===//

#include "clang/Parse/Parser.h"
#include "clang/Basic/Diagnostic.h"
using namespace clang;

/// isCXXDeclarationStatement - C++-specialized function that disambiguates
/// between a declaration or an expression statement, when parsing function
/// bodies. Returns true for declaration, false for expression.
///
///         declaration-statement:
///           block-declaration
///
///         block-declaration:
///           simple-declaration
///           asm-definition
///           namespace-alias-definition
///           using-declaration
///           using-directive
/// [C++0x]   static_assert-declaration                          [TODO]
///
///         asm-definition:
///           'asm' '(' string-literal ')' ';'
///
///         namespace-alias-definition:
///           'namespace' identifier = qualified-namespace-specifier ';'
///
///         using-declaration:
///           'using' typename[opt] '::'[opt] nested-name-specifier
///                 unqualified-id ';'
///           'using' '::' unqualified-id ;
///
///         using-directive:
///           'using' 'namespace' '::'[opt] nested-name-specifier[opt]
///                 namespace-name ';'
///
/// [C++0x] static_assert-declaration:                           [TODO]
/// [C++0x]   static_assert '(' constant-expression ',' string-literal ')' ';'
///
bool Parser::isCXXDeclarationStatement() {
  switch (Tok.getKind()) {
    // asm-definition
  case tok::kw_asm:
    // namespace-alias-definition
  case tok::kw_namespace:
    // using-declaration
    // using-directive
  case tok::kw_using:
    return true;
  default:
    // simple-declaration
    return isCXXSimpleDeclaration();
  }
}

/// isCXXSimpleDeclaration - C++-specialized function that disambiguates
/// between a simple-declaration or an expression-statement.
/// If during the disambiguation process a parsing error is encountered,
/// the function returns true to let the declaration parsing code handle it.
/// Returns false if the statement is disambiguated as expression.
///
/// simple-declaration:
///   decl-specifier-seq init-declarator-list[opt] ';'
///
bool Parser::isCXXSimpleDeclaration() {
  // C++ 6.8p1:
  // There is an ambiguity in the grammar involving expression-statements and
  // declarations: An expression-statement with a function-style explicit type
  // conversion (5.2.3) as its leftmost subexpression can be indistinguishable
  // from a declaration where the first declarator starts with a '('. In those
  // cases the statement is a declaration. [Note: To disambiguate, the whole
  // statement might have to be examined to determine if it is an
  // expression-statement or a declaration].

  // C++ 6.8p3:
  // The disambiguation is purely syntactic; that is, the meaning of the names
  // occurring in such a statement, beyond whether they are type-names or not,
  // is not generally used in or changed by the disambiguation. Class
  // templates are instantiated as necessary to determine if a qualified name
  // is a type-name. Disambiguation precedes parsing, and a statement
  // disambiguated as a declaration may be an ill-formed declaration.

  // We don't have to parse all of the decl-specifier-seq part. There's only
  // an ambiguity if the first decl-specifier is
  // simple-type-specifier/typename-specifier followed by a '(', which may
  // indicate a function-style cast expression.
  // isCXXDeclarationSpecifier will return TPR_ambiguous only in such a case.

  TentativeParsingResult TPR = isCXXDeclarationSpecifier();
  if (TPR != TPR_ambiguous)
    return TPR != TPR_false; // Returns true for TPR_true or TPR_error.

  // FIXME: Add statistics about the number of ambiguous statements encountered
  // and how they were resolved (number of declarations+number of expressions).

  // Ok, we have a simple-type-specifier/typename-specifier followed by a '('.
  // We need tentative parsing...

  TentativeParsingAction PA(*this);

  TPR = TryParseSimpleDeclaration();
  SourceLocation TentativeParseLoc = Tok.getLocation();

  PA.Revert();

  // In case of an error, let the declaration parsing code handle it.
  if (TPR == TPR_error)
    return true;

  // Declarations take precedence over expressions.
  if (TPR == TPR_ambiguous)
    TPR = TPR_true;

  assert(TPR == TPR_true || TPR == TPR_false);
  if (TPR == TPR_true && Tok.isNot(tok::kw_void)) {
    // We have a declaration that looks like a functional cast; there's a high
    // chance that the author intended the statement to be an expression.
    // Emit a warning.
    Diag(Tok.getLocation(), diag::warn_statement_disambiguation,
      "declaration", SourceRange(Tok.getLocation(), TentativeParseLoc));
  } else if (TPR == TPR_false && Tok.is(tok::kw_void)) {
    // A functional cast to 'void' expression ? Warning..
    Diag(Tok.getLocation(), diag::warn_statement_disambiguation,
      "expression", SourceRange(Tok.getLocation(), TentativeParseLoc));
  }

  return TPR == TPR_true;
}

/// simple-declaration:
///   decl-specifier-seq init-declarator-list[opt] ';'
///
Parser::TentativeParsingResult Parser::TryParseSimpleDeclaration() {
  // We know that we have a simple-type-specifier/typename-specifier followed
  // by a '('.
  assert(isCXXDeclarationSpecifier() == TPR_ambiguous);

  if (Tok.is(tok::kw_typeof))
    TryParseTypeofSpecifier();
  else
    ConsumeToken();

  assert(Tok.is(tok::l_paren) && "Expected '('");

  TentativeParsingResult TPR = TryParseInitDeclaratorList();
  if (TPR != TPR_ambiguous)
    return TPR;

  if (Tok.isNot(tok::semi))
    return TPR_false;

  return TPR_ambiguous;
}

///       init-declarator-list:
///         init-declarator
///         init-declarator-list ',' init-declarator
///
///       init-declarator:
///         declarator initializer[opt]
/// [GNU]   declarator simple-asm-expr[opt] attributes[opt] initializer[opt]
///
/// initializer:
///   '=' initializer-clause
///   '(' expression-list ')'
///
/// initializer-clause:
///   assignment-expression
///   '{' initializer-list ','[opt] '}'
///   '{' '}'
///
Parser::TentativeParsingResult Parser::TryParseInitDeclaratorList() {
  // GCC only examines the first declarator for disambiguation:
  // i.e:
  // int(x), ++x; // GCC regards it as ill-formed declaration.
  //
  // Comeau and MSVC will regard the above statement as correct expression.
  // Clang examines all of the declarators and also regards the above statement
  // as correct expression.

  while (1) {
    // declarator
    TentativeParsingResult TPR = TryParseDeclarator(false/*mayBeAbstract*/);
    if (TPR != TPR_ambiguous)
      return TPR;

    // [GNU] simple-asm-expr[opt] attributes[opt]
    if (Tok.is(tok::kw_asm) || Tok.is(tok::kw___attribute))
      return TPR_true;

    // initializer[opt]
    if (Tok.is(tok::l_paren)) {
      // Parse through the parens.
      ConsumeParen();
      if (!SkipUntil(tok::r_paren))
        return TPR_error;
    } else if (Tok.is(tok::equal)) {
      // MSVC won't examine the rest of declarators if '=' is encountered, it
      // will conclude that it is a declaration.
      // Comeau and Clang will examine the rest of declarators.
      // Note that "int(x) = {0}, ++x;" will be interpreted as ill-formed
      // expression.
      //
      // Parse through the initializer-clause.
      SkipUntil(tok::comma, true/*StopAtSemi*/, true/*DontConsume*/);
    }

    if (Tok.isNot(tok::comma))
      break;
    ConsumeToken(); // the comma.
  }

  return TPR_ambiguous;
}

/// isCXXConditionDeclaration - Disambiguates between a declaration or an
/// expression for a condition of a if/switch/while/for statement.
/// If during the disambiguation process a parsing error is encountered,
/// the function returns true to let the declaration parsing code handle it.
///
///       condition:
///         expression
///         type-specifier-seq declarator '=' assignment-expression
/// [GNU]   type-specifier-seq declarator simple-asm-expr[opt] attributes[opt]
///             '=' assignment-expression
///
bool Parser::isCXXConditionDeclaration() {
  TentativeParsingResult TPR = isCXXDeclarationSpecifier();
  if (TPR != TPR_ambiguous)
    return TPR != TPR_false; // Returns true for TPR_true or TPR_error.

  // FIXME: Add statistics about the number of ambiguous statements encountered
  // and how they were resolved (number of declarations+number of expressions).

  // Ok, we have a simple-type-specifier/typename-specifier followed by a '('.
  // We need tentative parsing...

  TentativeParsingAction PA(*this);

  // type-specifier-seq
  if (Tok.is(tok::kw_typeof))
    TryParseTypeofSpecifier();
  else
    ConsumeToken();
  assert(Tok.is(tok::l_paren) && "Expected '('");

  // declarator
  TPR = TryParseDeclarator(false/*mayBeAbstract*/);

  PA.Revert();

  // In case of an error, let the declaration parsing code handle it.
  if (TPR == TPR_error)
    return true;

  if (TPR == TPR_ambiguous) {
    // '='
    // [GNU] simple-asm-expr[opt] attributes[opt]
    if (Tok.is(tok::equal)  ||
        Tok.is(tok::kw_asm) || Tok.is(tok::kw___attribute))
      TPR = TPR_true;
    else
      TPR = TPR_false;
  }

  assert(TPR == TPR_true || TPR == TPR_false);
  return TPR == TPR_true;
}

///         declarator:
///           direct-declarator
///           ptr-operator declarator
///
///         direct-declarator:
///           declarator-id
///           direct-declarator '(' parameter-declaration-clause ')'
///                 cv-qualifier-seq[opt] exception-specification[opt]
///           direct-declarator '[' constant-expression[opt] ']'
///           '(' declarator ')'
/// [GNU]     '(' attributes declarator ')'
///
///         abstract-declarator:
///           ptr-operator abstract-declarator[opt]
///           direct-abstract-declarator
///
///         direct-abstract-declarator:
///           direct-abstract-declarator[opt]
///           '(' parameter-declaration-clause ')' cv-qualifier-seq[opt]
///                 exception-specification[opt]
///           direct-abstract-declarator[opt] '[' constant-expression[opt] ']'
///           '(' abstract-declarator ')'
///
///         ptr-operator:
///           '*' cv-qualifier-seq[opt]
///           '&'
/// [C++0x]   '&&'                                                        [TODO]
///           '::'[opt] nested-name-specifier '*' cv-qualifier-seq[opt]   [TODO]
///
///         cv-qualifier-seq:
///           cv-qualifier cv-qualifier-seq[opt]
///
///         cv-qualifier:
///           'const'
///           'volatile'
///
///         declarator-id:
///           id-expression
///
///         id-expression:
///           unqualified-id
///           qualified-id                                                [TODO]
///
///         unqualified-id:
///           identifier
///           operator-function-id                                        [TODO]
///           conversion-function-id                                      [TODO]
///           '~' class-name                                              [TODO]
///           template-id                                                 [TODO]
///
Parser::TentativeParsingResult Parser::TryParseDeclarator(bool mayBeAbstract) {
  // declarator:
  //   direct-declarator
  //   ptr-operator declarator

  while (1) {
    if (Tok.is(tok::star) || Tok.is(tok::amp)) {
      // ptr-operator
      ConsumeToken();
      while (Tok.is(tok::kw_const)    ||
             Tok.is(tok::kw_volatile) ||
             Tok.is(tok::kw_restrict)   )
        ConsumeToken();
    } else {
      break;
    }
  }

  // direct-declarator:
  // direct-abstract-declarator:

  if (Tok.is(tok::identifier)) {
    // declarator-id
    ConsumeToken();
  } else if (Tok.is(tok::l_paren)) {
    if (mayBeAbstract && isCXXFunctionDeclarator()) {
      // '(' parameter-declaration-clause ')' cv-qualifier-seq[opt]
      //        exception-specification[opt]
      TentativeParsingResult TPR = TryParseFunctionDeclarator();
      if (TPR != TPR_ambiguous)
        return TPR;
    } else {
      // '(' declarator ')'
      // '(' attributes declarator ')'
      // '(' abstract-declarator ')'
      ConsumeParen();
      if (Tok.is(tok::kw___attribute))
        return TPR_true; // attributes indicate declaration
      TentativeParsingResult TPR = TryParseDeclarator(mayBeAbstract);
      if (TPR != TPR_ambiguous)
        return TPR;
      if (Tok.isNot(tok::r_paren))
        return TPR_false;
      ConsumeParen();
    }
  } else if (!mayBeAbstract) {
    return TPR_false;
  }

  while (1) {
    TentativeParsingResult TPR;

    if (Tok.is(tok::l_paren)) {
      // direct-declarator '(' parameter-declaration-clause ')'
      //        cv-qualifier-seq[opt] exception-specification[opt]
      if (!isCXXFunctionDeclarator())
        break;
      TPR = TryParseFunctionDeclarator();
    } else if (Tok.is(tok::l_square)) {
      // direct-declarator '[' constant-expression[opt] ']'
      // direct-abstract-declarator[opt] '[' constant-expression[opt] ']'
      TPR = TryParseBracketDeclarator();
    } else {
      break;
    }

    if (TPR != TPR_ambiguous)
      return TPR;
  }

  return TPR_ambiguous;
}

/// isCXXDeclarationSpecifier - Returns TPR_true if it is a declaration
/// specifier, TPR_false if it is not, TPR_ambiguous if it could be either
/// a decl-specifier or a function-style cast, and TPR_error if a parsing
/// error was found and reported.
///
///         decl-specifier:
///           storage-class-specifier
///           type-specifier
///           function-specifier
///           'friend'
///           'typedef'
/// [GNU]     attributes declaration-specifiers[opt]
///
///         storage-class-specifier:
///           'register'
///           'static'
///           'extern'
///           'mutable'
///           'auto'
/// [GNU]     '__thread'
///
///         function-specifier:
///           'inline'
///           'virtual'
///           'explicit'
///
///         typedef-name:
///           identifier
///
///         type-specifier:
///           simple-type-specifier
///           class-specifier
///           enum-specifier
///           elaborated-type-specifier
///           typename-specifier                                    [TODO]
///           cv-qualifier
///
///         simple-type-specifier:
///           '::'[opt] nested-name-specifier[opt] type-name        [TODO]
///           '::'[opt] nested-name-specifier 'template'
///                 simple-template-id                              [TODO]
///           'char'
///           'wchar_t'
///           'bool'
///           'short'
///           'int'
///           'long'
///           'signed'
///           'unsigned'
///           'float'
///           'double'
///           'void'
/// [GNU]     typeof-specifier
/// [GNU]     '_Complex'
/// [C++0x]   'auto'                                                [TODO]
///
///         type-name:
///           class-name
///           enum-name
///           typedef-name
///
///         elaborated-type-specifier:
///           class-key '::'[opt] nested-name-specifier[opt] identifier
///           class-key '::'[opt] nested-name-specifier[opt] 'template'[opt]
///               simple-template-id
///           'enum' '::'[opt] nested-name-specifier[opt] identifier
///
///         enum-name:
///           identifier
///
///         enum-specifier:
///           'enum' identifier[opt] '{' enumerator-list[opt] '}'
///           'enum' identifier[opt] '{' enumerator-list ',' '}'
///
///         class-specifier:
///           class-head '{' member-specification[opt] '}'
///
///         class-head:
///           class-key identifier[opt] base-clause[opt]
///           class-key nested-name-specifier identifier base-clause[opt]
///           class-key nested-name-specifier[opt] simple-template-id
///               base-clause[opt]
///
///         class-key:
///           'class'
///           'struct'
///           'union'
///
///         cv-qualifier:
///           'const'
///           'volatile'
/// [GNU]     restrict
///
Parser::TentativeParsingResult Parser::isCXXDeclarationSpecifier() {
  switch (Tok.getKind()) {
    // decl-specifier:
    //   storage-class-specifier
    //   type-specifier
    //   function-specifier
    //   'friend'
    //   'typedef'

  case tok::kw_friend:
  case tok::kw_typedef:
    // storage-class-specifier
  case tok::kw_register:
  case tok::kw_static:
  case tok::kw_extern:
  case tok::kw_mutable:
  case tok::kw_auto:
  case tok::kw___thread:
    // function-specifier
  case tok::kw_inline:
  case tok::kw_virtual:
  case tok::kw_explicit:

    // type-specifier:
    //   simple-type-specifier
    //   class-specifier
    //   enum-specifier
    //   elaborated-type-specifier
    //   typename-specifier
    //   cv-qualifier

    // class-specifier
    // elaborated-type-specifier
  case tok::kw_class:
  case tok::kw_struct:
  case tok::kw_union:
    // enum-specifier
  case tok::kw_enum:
    // cv-qualifier
  case tok::kw_const:
  case tok::kw_volatile:

    // GNU
  case tok::kw_restrict:
  case tok::kw__Complex:
  case tok::kw___attribute:
    return TPR_true;

    // The ambiguity resides in a simple-type-specifier/typename-specifier
    // followed by a '('. The '(' could either be the start of:
    //
    //   direct-declarator:
    //     '(' declarator ')'
    //
    //   direct-abstract-declarator:
    //     '(' parameter-declaration-clause ')' cv-qualifier-seq[opt]
    //              exception-specification[opt]
    //     '(' abstract-declarator ')'
    //
    // or part of a function-style cast expression:
    //
    //     simple-type-specifier '(' expression-list[opt] ')'
    //

    // simple-type-specifier:

  case tok::identifier:
    if (!Actions.isTypeName(*Tok.getIdentifierInfo(), CurScope))
      return TPR_false;
    // FALL THROUGH.

  case tok::kw_char:
  case tok::kw_wchar_t:
  case tok::kw_bool:
  case tok::kw_short:
  case tok::kw_int:
  case tok::kw_long:
  case tok::kw_signed:
  case tok::kw_unsigned:
  case tok::kw_float:
  case tok::kw_double:
  case tok::kw_void:
    if (NextToken().is(tok::l_paren))
      return TPR_ambiguous;

    return TPR_true;

    // GNU typeof support.
  case tok::kw_typeof: {
    if (NextToken().isNot(tok::l_paren))
      return TPR_true;

    TentativeParsingAction PA(*this);

    TentativeParsingResult TPR = TryParseTypeofSpecifier();
    bool isFollowedByParen = Tok.is(tok::l_paren);

    PA.Revert();

    if (TPR == TPR_error)
      return TPR_error;

    if (isFollowedByParen)
      return TPR_ambiguous;

    return TPR_true;
  }

  default:
    return TPR_false;
  }
}

/// [GNU] typeof-specifier:
///         'typeof' '(' expressions ')'
///         'typeof' '(' type-name ')'
///
Parser::TentativeParsingResult Parser::TryParseTypeofSpecifier() {
  assert(Tok.is(tok::kw_typeof) && "Expected 'typeof'!");
  ConsumeToken();

  assert(Tok.is(tok::l_paren) && "Expected '('");
  // Parse through the parens after 'typeof'.
  ConsumeParen();
  if (!SkipUntil(tok::r_paren))
    return TPR_error;

  return TPR_ambiguous;
}

Parser::TentativeParsingResult Parser::TryParseDeclarationSpecifier() {
  TentativeParsingResult TPR = isCXXDeclarationSpecifier();
  if (TPR != TPR_ambiguous)
    return TPR;

  if (Tok.is(tok::kw_typeof))
    TryParseTypeofSpecifier();
  else
    ConsumeToken();
  
  assert(Tok.is(tok::l_paren) && "Expected '('!");
  return TPR_ambiguous;
}

/// isCXXFunctionDeclarator - Disambiguates between a function declarator or
/// a constructor-style initializer, when parsing declaration statements.
/// Returns true for function declarator and false for constructor-style
/// initializer.
/// If during the disambiguation process a parsing error is encountered,
/// the function returns true to let the declaration parsing code handle it.
///
/// '(' parameter-declaration-clause ')' cv-qualifier-seq[opt]
///         exception-specification[opt]
///
bool Parser::isCXXFunctionDeclarator() {
  TentativeParsingAction PA(*this);

  ConsumeParen();
  TentativeParsingResult TPR = TryParseParameterDeclarationClause();
  if (TPR == TPR_ambiguous && Tok.isNot(tok::r_paren))
    TPR = TPR_false;

  PA.Revert();

  // In case of an error, let the declaration parsing code handle it.
  if (TPR == TPR_error)
    return true;

  // Function declarator has precedence over constructor-style initializer.
  if (TPR == TPR_ambiguous)
    return TPR_true;
  return TPR == TPR_true;
}

/// parameter-declaration-clause:
///   parameter-declaration-list[opt] '...'[opt]
///   parameter-declaration-list ',' '...'
///
/// parameter-declaration-list:
///   parameter-declaration
///   parameter-declaration-list ',' parameter-declaration
///
/// parameter-declaration:
///   decl-specifier-seq declarator
///   decl-specifier-seq declarator '=' assignment-expression
///   decl-specifier-seq abstract-declarator[opt]
///   decl-specifier-seq abstract-declarator[opt] '=' assignment-expression
///
Parser::TentativeParsingResult Parser::TryParseParameterDeclarationClause() {

  if (Tok.is(tok::r_paren))
    return TPR_true;

  //   parameter-declaration-list[opt] '...'[opt]
  //   parameter-declaration-list ',' '...'
  //
  // parameter-declaration-list:
  //   parameter-declaration
  //   parameter-declaration-list ',' parameter-declaration
  //
  while (1) {
    // '...'[opt]
    if (Tok.is(tok::ellipsis)) {
      ConsumeToken();
      return TPR_true; // '...' is a sign of a function declarator.
    }

    // decl-specifier-seq
    TentativeParsingResult TPR = TryParseDeclarationSpecifier();
    if (TPR != TPR_ambiguous)
      return TPR;

    // declarator
    // abstract-declarator[opt]
    TPR = TryParseDeclarator(true/*mayBeAbstract*/);
    if (TPR != TPR_ambiguous)
      return TPR;

    if (Tok.is(tok::equal)) {
      // '=' assignment-expression
      // Parse through assignment-expression.
      tok::TokenKind StopToks[3] ={ tok::comma, tok::ellipsis, tok::r_paren };
      if (!SkipUntil(StopToks, 3, true/*StopAtSemi*/, true/*DontConsume*/))
        return TPR_error;
    }

    if (Tok.is(tok::ellipsis)) {
      ConsumeToken();
      return TPR_true; // '...' is a sign of a function declarator.
    }

    if (Tok.isNot(tok::comma))
      break;
    ConsumeToken(); // the comma.
  }

  return TPR_ambiguous;
}

/// TryParseFunctionDeclarator - We previously determined (using
/// isCXXFunctionDeclarator) that we are at a function declarator. Now parse
/// through it.
/// 
/// '(' parameter-declaration-clause ')' cv-qualifier-seq[opt]
///         exception-specification[opt]
///
/// exception-specification:
///   'throw' '(' type-id-list[opt] ')'
///
Parser::TentativeParsingResult Parser::TryParseFunctionDeclarator() {
  assert(Tok.is(tok::l_paren));
  // Parse through the parens.
  ConsumeParen();
  if (!SkipUntil(tok::r_paren))
    return TPR_error;

  // cv-qualifier-seq
  while (Tok.is(tok::kw_const)    ||
         Tok.is(tok::kw_volatile) ||
         Tok.is(tok::kw_restrict)   )
    ConsumeToken();

  // exception-specification
  if (Tok.is(tok::kw_throw)) {
    ConsumeToken();
    if (Tok.isNot(tok::l_paren))
      return TPR_error;

    // Parse through the parens after 'throw'.
    ConsumeParen();
    if (!SkipUntil(tok::r_paren))
      return TPR_error;
  }

  return TPR_ambiguous;
}

/// '[' constant-expression[opt] ']'
///
Parser::TentativeParsingResult Parser::TryParseBracketDeclarator() {
  ConsumeBracket();
  if (!SkipUntil(tok::r_square))
    return TPR_error;

  return TPR_ambiguous;
}
