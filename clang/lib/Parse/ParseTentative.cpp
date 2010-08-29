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
#include "clang/Parse/ParseDiagnostic.h"
#include "clang/Sema/ParsedTemplate.h"
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
/// [C++0x]   static_assert-declaration
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
bool Parser::isCXXDeclarationStatement() {
  switch (Tok.getKind()) {
    // asm-definition
  case tok::kw_asm:
    // namespace-alias-definition
  case tok::kw_namespace:
    // using-declaration
    // using-directive
  case tok::kw_using:
    // static_assert-declaration
  case tok::kw_static_assert:
    return true;
    // simple-declaration
  default:
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
  // isCXXDeclarationSpecifier will return TPResult::Ambiguous() only in such
  // a case.

  TPResult TPR = isCXXDeclarationSpecifier();
  if (TPR != TPResult::Ambiguous())
    return TPR != TPResult::False(); // Returns true for TPResult::True() or
                                     // TPResult::Error().

  // FIXME: Add statistics about the number of ambiguous statements encountered
  // and how they were resolved (number of declarations+number of expressions).

  // Ok, we have a simple-type-specifier/typename-specifier followed by a '('.
  // We need tentative parsing...

  TentativeParsingAction PA(*this);

  TPR = TryParseSimpleDeclaration();
  SourceLocation TentativeParseLoc = Tok.getLocation();

  PA.Revert();

  // In case of an error, let the declaration parsing code handle it.
  if (TPR == TPResult::Error())
    return true;

  // Declarations take precedence over expressions.
  if (TPR == TPResult::Ambiguous())
    TPR = TPResult::True();

  assert(TPR == TPResult::True() || TPR == TPResult::False());
  return TPR == TPResult::True();
}

/// simple-declaration:
///   decl-specifier-seq init-declarator-list[opt] ';'
///
Parser::TPResult Parser::TryParseSimpleDeclaration() {
  // We know that we have a simple-type-specifier/typename-specifier followed
  // by a '('.
  assert(isCXXDeclarationSpecifier() == TPResult::Ambiguous());

  if (Tok.is(tok::kw_typeof))
    TryParseTypeofSpecifier();
  else
    ConsumeToken();

  assert(Tok.is(tok::l_paren) && "Expected '('");

  TPResult TPR = TryParseInitDeclaratorList();
  if (TPR != TPResult::Ambiguous())
    return TPR;

  if (Tok.isNot(tok::semi))
    return TPResult::False();

  return TPResult::Ambiguous();
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
Parser::TPResult Parser::TryParseInitDeclaratorList() {
  while (1) {
    // declarator
    TPResult TPR = TryParseDeclarator(false/*mayBeAbstract*/);
    if (TPR != TPResult::Ambiguous())
      return TPR;

    // [GNU] simple-asm-expr[opt] attributes[opt]
    if (Tok.is(tok::kw_asm) || Tok.is(tok::kw___attribute))
      return TPResult::True();

    // initializer[opt]
    if (Tok.is(tok::l_paren)) {
      // Parse through the parens.
      ConsumeParen();
      if (!SkipUntil(tok::r_paren))
        return TPResult::Error();
    } else if (Tok.is(tok::equal) || isTokIdentifier_in()) {
      // MSVC and g++ won't examine the rest of declarators if '=' is 
      // encountered; they just conclude that we have a declaration.
      // EDG parses the initializer completely, which is the proper behavior
      // for this case.
      //
      // At present, Clang follows MSVC and g++, since the parser does not have
      // the ability to parse an expression fully without recording the
      // results of that parse.
      // Also allow 'in' after on objective-c declaration as in: 
      // for (int (^b)(void) in array). Ideally this should be done in the 
      // context of parsing for-init-statement of a foreach statement only. But,
      // in any other context 'in' is invalid after a declaration and parser
      // issues the error regardless of outcome of this decision.
      // FIXME. Change if above assumption does not hold.
      return TPResult::True();
    }

    if (Tok.isNot(tok::comma))
      break;
    ConsumeToken(); // the comma.
  }

  return TPResult::Ambiguous();
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
  TPResult TPR = isCXXDeclarationSpecifier();
  if (TPR != TPResult::Ambiguous())
    return TPR != TPResult::False(); // Returns true for TPResult::True() or
                                     // TPResult::Error().

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

  // In case of an error, let the declaration parsing code handle it.
  if (TPR == TPResult::Error())
    TPR = TPResult::True();

  if (TPR == TPResult::Ambiguous()) {
    // '='
    // [GNU] simple-asm-expr[opt] attributes[opt]
    if (Tok.is(tok::equal)  ||
        Tok.is(tok::kw_asm) || Tok.is(tok::kw___attribute))
      TPR = TPResult::True();
    else
      TPR = TPResult::False();
  }

  PA.Revert();

  assert(TPR == TPResult::True() || TPR == TPResult::False());
  return TPR == TPResult::True();
}

  /// \brief Determine whether the next set of tokens contains a type-id.
  ///
  /// The context parameter states what context we're parsing right
  /// now, which affects how this routine copes with the token
  /// following the type-id. If the context is TypeIdInParens, we have
  /// already parsed the '(' and we will cease lookahead when we hit
  /// the corresponding ')'. If the context is
  /// TypeIdAsTemplateArgument, we've already parsed the '<' or ','
  /// before this template argument, and will cease lookahead when we
  /// hit a '>', '>>' (in C++0x), or ','. Returns true for a type-id
  /// and false for an expression.  If during the disambiguation
  /// process a parsing error is encountered, the function returns
  /// true to let the declaration parsing code handle it.
  ///
  /// type-id:
  ///   type-specifier-seq abstract-declarator[opt]
  ///
bool Parser::isCXXTypeId(TentativeCXXTypeIdContext Context, bool &isAmbiguous) {

  isAmbiguous = false;

  // C++ 8.2p2:
  // The ambiguity arising from the similarity between a function-style cast and
  // a type-id can occur in different contexts. The ambiguity appears as a
  // choice between a function-style cast expression and a declaration of a
  // type. The resolution is that any construct that could possibly be a type-id
  // in its syntactic context shall be considered a type-id.

  TPResult TPR = isCXXDeclarationSpecifier();
  if (TPR != TPResult::Ambiguous())
    return TPR != TPResult::False(); // Returns true for TPResult::True() or
                                     // TPResult::Error().

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
  TPR = TryParseDeclarator(true/*mayBeAbstract*/, false/*mayHaveIdentifier*/);

  // In case of an error, let the declaration parsing code handle it.
  if (TPR == TPResult::Error())
    TPR = TPResult::True();

  if (TPR == TPResult::Ambiguous()) {
    // We are supposed to be inside parens, so if after the abstract declarator
    // we encounter a ')' this is a type-id, otherwise it's an expression.
    if (Context == TypeIdInParens && Tok.is(tok::r_paren)) {
      TPR = TPResult::True();
      isAmbiguous = true;

    // We are supposed to be inside a template argument, so if after
    // the abstract declarator we encounter a '>', '>>' (in C++0x), or
    // ',', this is a type-id. Otherwise, it's an expression.
    } else if (Context == TypeIdAsTemplateArgument &&
               (Tok.is(tok::greater) || Tok.is(tok::comma) ||
                (getLang().CPlusPlus0x && Tok.is(tok::greatergreater)))) {
      TPR = TPResult::True();
      isAmbiguous = true;

    } else
      TPR = TPResult::False();
  }

  PA.Revert();

  assert(TPR == TPResult::True() || TPR == TPResult::False());
  return TPR == TPResult::True();
}

/// isCXX0XAttributeSpecifier - returns true if this is a C++0x
/// attribute-specifier. By default, unless in Obj-C++, only a cursory check is
/// performed that will simply return true if a [[ is seen. Currently C++ has no
/// syntactical ambiguities from this check, but it may inhibit error recovery.
/// If CheckClosing is true, a check is made for closing ]] brackets.
///
/// If given, After is set to the token after the attribute-specifier so that
/// appropriate parsing decisions can be made; it is left untouched if false is
/// returned.
///
/// FIXME: If an error is in the closing ]] brackets, the program assumes
/// the absence of an attribute-specifier, which can cause very yucky errors
/// to occur.
///
/// [C++0x] attribute-specifier:
///         '[' '[' attribute-list ']' ']'
///
/// [C++0x] attribute-list:
///         attribute[opt]
///         attribute-list ',' attribute[opt]
///
/// [C++0x] attribute:
///         attribute-token attribute-argument-clause[opt]
///
/// [C++0x] attribute-token:
///         identifier
///         attribute-scoped-token
///
/// [C++0x] attribute-scoped-token:
///         attribute-namespace '::' identifier
///
/// [C++0x] attribute-namespace:
///         identifier
///
/// [C++0x] attribute-argument-clause:
///         '(' balanced-token-seq ')'
///
/// [C++0x] balanced-token-seq:
///         balanced-token
///         balanced-token-seq balanced-token
///
/// [C++0x] balanced-token:
///         '(' balanced-token-seq ')'
///         '[' balanced-token-seq ']'
///         '{' balanced-token-seq '}'
///         any token but '(', ')', '[', ']', '{', or '}'
bool Parser::isCXX0XAttributeSpecifier (bool CheckClosing,
                                        tok::TokenKind *After) {
  if (Tok.isNot(tok::l_square) || NextToken().isNot(tok::l_square))
    return false;
  
  // No tentative parsing if we don't need to look for ]]
  if (!CheckClosing && !getLang().ObjC1)
    return true;
  
  struct TentativeReverter {
    TentativeParsingAction PA;

    TentativeReverter (Parser& P)
      : PA(P)
    {}
    ~TentativeReverter () {
      PA.Revert();
    }
  } R(*this);

  // Opening brackets were checked for above.
  ConsumeBracket();
  ConsumeBracket();

  // SkipUntil will handle balanced tokens, which are guaranteed in attributes.
  SkipUntil(tok::r_square, false);

  if (Tok.isNot(tok::r_square))
    return false;
  ConsumeBracket();

  if (After)
    *After = Tok.getKind();

  return true;
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
///           '::'[opt] nested-name-specifier '*' cv-qualifier-seq[opt]
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
Parser::TPResult Parser::TryParseDeclarator(bool mayBeAbstract,
                                            bool mayHaveIdentifier) {
  // declarator:
  //   direct-declarator
  //   ptr-operator declarator

  while (1) {
    if (Tok.is(tok::coloncolon) || Tok.is(tok::identifier))
      if (TryAnnotateCXXScopeToken(true))
        return TPResult::Error();

    if (Tok.is(tok::star) || Tok.is(tok::amp) || Tok.is(tok::caret) ||
        (Tok.is(tok::annot_cxxscope) && NextToken().is(tok::star))) {
      // ptr-operator
      ConsumeToken();
      while (Tok.is(tok::kw_const)    ||
             Tok.is(tok::kw_volatile) ||
             Tok.is(tok::kw_restrict))
        ConsumeToken();
    } else {
      break;
    }
  }

  // direct-declarator:
  // direct-abstract-declarator:

  if ((Tok.is(tok::identifier) ||
       (Tok.is(tok::annot_cxxscope) && NextToken().is(tok::identifier))) &&
      mayHaveIdentifier) {
    // declarator-id
    if (Tok.is(tok::annot_cxxscope))
      ConsumeToken();
    ConsumeToken();
  } else if (Tok.is(tok::l_paren)) {
    ConsumeParen();
    if (mayBeAbstract &&
        (Tok.is(tok::r_paren) ||       // 'int()' is a function.
         Tok.is(tok::ellipsis) ||      // 'int(...)' is a function.
         isDeclarationSpecifier())) {   // 'int(int)' is a function.
      // '(' parameter-declaration-clause ')' cv-qualifier-seq[opt]
      //        exception-specification[opt]
      TPResult TPR = TryParseFunctionDeclarator();
      if (TPR != TPResult::Ambiguous())
        return TPR;
    } else {
      // '(' declarator ')'
      // '(' attributes declarator ')'
      // '(' abstract-declarator ')'
      if (Tok.is(tok::kw___attribute))
        return TPResult::True(); // attributes indicate declaration
      TPResult TPR = TryParseDeclarator(mayBeAbstract, mayHaveIdentifier);
      if (TPR != TPResult::Ambiguous())
        return TPR;
      if (Tok.isNot(tok::r_paren))
        return TPResult::False();
      ConsumeParen();
    }
  } else if (!mayBeAbstract) {
    return TPResult::False();
  }

  while (1) {
    TPResult TPR(TPResult::Ambiguous());

    if (Tok.is(tok::l_paren)) {
      // Check whether we have a function declarator or a possible ctor-style
      // initializer that follows the declarator. Note that ctor-style
      // initializers are not possible in contexts where abstract declarators
      // are allowed.
      if (!mayBeAbstract && !isCXXFunctionDeclarator(false/*warnIfAmbiguous*/))
        break;

      // direct-declarator '(' parameter-declaration-clause ')'
      //        cv-qualifier-seq[opt] exception-specification[opt]
      ConsumeParen();
      TPR = TryParseFunctionDeclarator();
    } else if (Tok.is(tok::l_square)) {
      // direct-declarator '[' constant-expression[opt] ']'
      // direct-abstract-declarator[opt] '[' constant-expression[opt] ']'
      TPR = TryParseBracketDeclarator();
    } else {
      break;
    }

    if (TPR != TPResult::Ambiguous())
      return TPR;
  }

  return TPResult::Ambiguous();
}

/// isCXXDeclarationSpecifier - Returns TPResult::True() if it is a declaration
/// specifier, TPResult::False() if it is not, TPResult::Ambiguous() if it could
/// be either a decl-specifier or a function-style cast, and TPResult::Error()
/// if a parsing error was found and reported.
///
///         decl-specifier:
///           storage-class-specifier
///           type-specifier
///           function-specifier
///           'friend'
///           'typedef'
/// [C++0x]   'constexpr'
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
///           typename-specifier
///           cv-qualifier
///
///         simple-type-specifier:
///           '::'[opt] nested-name-specifier[opt] type-name
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
/// [C++0x]   'decltype' ( expression )
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
Parser::TPResult Parser::isCXXDeclarationSpecifier() {
  switch (Tok.getKind()) {
  case tok::identifier:   // foo::bar
    // Check for need to substitute AltiVec __vector keyword
    // for "vector" identifier.
    if (TryAltiVecVectorToken())
      return TPResult::True();
    // Fall through.
  case tok::kw_typename:  // typename T::type
    // Annotate typenames and C++ scope specifiers.  If we get one, just
    // recurse to handle whatever we get.
    if (TryAnnotateTypeOrScopeToken())
      return TPResult::Error();
    if (Tok.is(tok::identifier))
      return TPResult::False();
    return isCXXDeclarationSpecifier();

  case tok::coloncolon: {    // ::foo::bar
    const Token &Next = NextToken();
    if (Next.is(tok::kw_new) ||    // ::new
        Next.is(tok::kw_delete))   // ::delete
      return TPResult::False();

    // Annotate typenames and C++ scope specifiers.  If we get one, just
    // recurse to handle whatever we get.
    if (TryAnnotateTypeOrScopeToken())
      return TPResult::Error();
    return isCXXDeclarationSpecifier();
  }
      
    // decl-specifier:
    //   storage-class-specifier
    //   type-specifier
    //   function-specifier
    //   'friend'
    //   'typedef'
    //   'constexpr'
  case tok::kw_friend:
  case tok::kw_typedef:
  case tok::kw_constexpr:
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
    return TPResult::True();

    // Microsoft
  case tok::kw___declspec:
  case tok::kw___cdecl:
  case tok::kw___stdcall:
  case tok::kw___fastcall:
  case tok::kw___thiscall:
  case tok::kw___w64:
  case tok::kw___ptr64:
  case tok::kw___forceinline:
    return TPResult::True();
  
    // AltiVec
  case tok::kw___vector:
    return TPResult::True();

  case tok::annot_template_id: {
    TemplateIdAnnotation *TemplateId
      = static_cast<TemplateIdAnnotation *>(Tok.getAnnotationValue());
    if (TemplateId->Kind != TNK_Type_template)
      return TPResult::False();
    CXXScopeSpec SS;
    AnnotateTemplateIdTokenAsType(&SS);
    assert(Tok.is(tok::annot_typename));
    goto case_typename;
  }

  case tok::annot_cxxscope: // foo::bar or ::foo::bar, but already parsed
    // We've already annotated a scope; try to annotate a type.
    if (TryAnnotateTypeOrScopeToken())
      return TPResult::Error();
    if (!Tok.is(tok::annot_typename))
      return TPResult::False();
    // If that succeeded, fallthrough into the generic simple-type-id case.

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

  case tok::kw_char:
  case tok::kw_wchar_t:
  case tok::kw_char16_t:
  case tok::kw_char32_t:
  case tok::kw_bool:
  case tok::kw_short:
  case tok::kw_int:
  case tok::kw_long:
  case tok::kw_signed:
  case tok::kw_unsigned:
  case tok::kw_float:
  case tok::kw_double:
  case tok::kw_void:
  case tok::annot_typename:
  case_typename:
    if (NextToken().is(tok::l_paren))
      return TPResult::Ambiguous();

    return TPResult::True();

  // GNU typeof support.
  case tok::kw_typeof: {
    if (NextToken().isNot(tok::l_paren))
      return TPResult::True();

    TentativeParsingAction PA(*this);

    TPResult TPR = TryParseTypeofSpecifier();
    bool isFollowedByParen = Tok.is(tok::l_paren);

    PA.Revert();

    if (TPR == TPResult::Error())
      return TPResult::Error();

    if (isFollowedByParen)
      return TPResult::Ambiguous();

    return TPResult::True();
  }

  // C++0x decltype support.
  case tok::kw_decltype:
    return TPResult::True();

  default:
    return TPResult::False();
  }
}

/// [GNU] typeof-specifier:
///         'typeof' '(' expressions ')'
///         'typeof' '(' type-name ')'
///
Parser::TPResult Parser::TryParseTypeofSpecifier() {
  assert(Tok.is(tok::kw_typeof) && "Expected 'typeof'!");
  ConsumeToken();

  assert(Tok.is(tok::l_paren) && "Expected '('");
  // Parse through the parens after 'typeof'.
  ConsumeParen();
  if (!SkipUntil(tok::r_paren))
    return TPResult::Error();

  return TPResult::Ambiguous();
}

Parser::TPResult Parser::TryParseDeclarationSpecifier() {
  TPResult TPR = isCXXDeclarationSpecifier();
  if (TPR != TPResult::Ambiguous())
    return TPR;

  if (Tok.is(tok::kw_typeof))
    TryParseTypeofSpecifier();
  else
    ConsumeToken();

  assert(Tok.is(tok::l_paren) && "Expected '('!");
  return TPResult::Ambiguous();
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
bool Parser::isCXXFunctionDeclarator(bool warnIfAmbiguous) {

  // C++ 8.2p1:
  // The ambiguity arising from the similarity between a function-style cast and
  // a declaration mentioned in 6.8 can also occur in the context of a
  // declaration. In that context, the choice is between a function declaration
  // with a redundant set of parentheses around a parameter name and an object
  // declaration with a function-style cast as the initializer. Just as for the
  // ambiguities mentioned in 6.8, the resolution is to consider any construct
  // that could possibly be a declaration a declaration.

  TentativeParsingAction PA(*this);

  ConsumeParen();
  TPResult TPR = TryParseParameterDeclarationClause();
  if (TPR == TPResult::Ambiguous() && Tok.isNot(tok::r_paren))
    TPR = TPResult::False();

  SourceLocation TPLoc = Tok.getLocation();
  PA.Revert();

  // In case of an error, let the declaration parsing code handle it.
  if (TPR == TPResult::Error())
    return true;

  if (TPR == TPResult::Ambiguous()) {
    // Function declarator has precedence over constructor-style initializer.
    // Emit a warning just in case the author intended a variable definition.
    if (warnIfAmbiguous)
      Diag(Tok, diag::warn_parens_disambiguated_as_function_decl)
        << SourceRange(Tok.getLocation(), TPLoc);
    return true;
  }

  return TPR == TPResult::True();
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
Parser::TPResult Parser::TryParseParameterDeclarationClause() {

  if (Tok.is(tok::r_paren))
    return TPResult::True();

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
      return TPResult::True(); // '...' is a sign of a function declarator.
    }

    // decl-specifier-seq
    TPResult TPR = TryParseDeclarationSpecifier();
    if (TPR != TPResult::Ambiguous())
      return TPR;

    // declarator
    // abstract-declarator[opt]
    TPR = TryParseDeclarator(true/*mayBeAbstract*/);
    if (TPR != TPResult::Ambiguous())
      return TPR;

    if (Tok.is(tok::equal)) {
      // '=' assignment-expression
      // Parse through assignment-expression.
      tok::TokenKind StopToks[3] ={ tok::comma, tok::ellipsis, tok::r_paren };
      if (!SkipUntil(StopToks, 3, true/*StopAtSemi*/, true/*DontConsume*/))
        return TPResult::Error();
    }

    if (Tok.is(tok::ellipsis)) {
      ConsumeToken();
      return TPResult::True(); // '...' is a sign of a function declarator.
    }

    if (Tok.isNot(tok::comma))
      break;
    ConsumeToken(); // the comma.
  }

  return TPResult::Ambiguous();
}

/// TryParseFunctionDeclarator - We parsed a '(' and we want to try to continue
/// parsing as a function declarator.
/// If TryParseFunctionDeclarator fully parsed the function declarator, it will
/// return TPResult::Ambiguous(), otherwise it will return either False() or
/// Error().
///
/// '(' parameter-declaration-clause ')' cv-qualifier-seq[opt]
///         exception-specification[opt]
///
/// exception-specification:
///   'throw' '(' type-id-list[opt] ')'
///
Parser::TPResult Parser::TryParseFunctionDeclarator() {

  // The '(' is already parsed.

  TPResult TPR = TryParseParameterDeclarationClause();
  if (TPR == TPResult::Ambiguous() && Tok.isNot(tok::r_paren))
    TPR = TPResult::False();

  if (TPR == TPResult::False() || TPR == TPResult::Error())
    return TPR;

  // Parse through the parens.
  if (!SkipUntil(tok::r_paren))
    return TPResult::Error();

  // cv-qualifier-seq
  while (Tok.is(tok::kw_const)    ||
         Tok.is(tok::kw_volatile) ||
         Tok.is(tok::kw_restrict)   )
    ConsumeToken();

  // exception-specification
  if (Tok.is(tok::kw_throw)) {
    ConsumeToken();
    if (Tok.isNot(tok::l_paren))
      return TPResult::Error();

    // Parse through the parens after 'throw'.
    ConsumeParen();
    if (!SkipUntil(tok::r_paren))
      return TPResult::Error();
  }

  return TPResult::Ambiguous();
}

/// '[' constant-expression[opt] ']'
///
Parser::TPResult Parser::TryParseBracketDeclarator() {
  ConsumeBracket();
  if (!SkipUntil(tok::r_square))
    return TPResult::Error();

  return TPResult::Ambiguous();
}
