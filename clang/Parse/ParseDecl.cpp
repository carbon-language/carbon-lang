//===--- Declaration.cpp - Declaration Parsing ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the Declaration portions of the Parser interfaces.
//
//===----------------------------------------------------------------------===//

#include "clang/Parse/Parser.h"
#include "clang/Parse/Declarations.h"
using namespace llvm;
using namespace clang;

//===----------------------------------------------------------------------===//
// C99 6.7: Declarations.
//===----------------------------------------------------------------------===//

/// ParseTypeName
///       type-name: [C99 6.7.6]
///         specifier-qualifier-list abstract-declarator[opt]
void Parser::ParseTypeName() {
  // Parse the common declaration-specifiers piece.
  DeclSpec DS;
  ParseSpecifierQualifierList(DS);
  
  // Parse the abstract-declarator, if present.
  Declarator DeclaratorInfo(DS, Declarator::TypeNameContext);
  ParseDeclarator(DeclaratorInfo);
}


/// ParseDeclaration - Parse a full 'declaration', which consists of
/// declaration-specifiers, some number of declarators, and a semicolon.
/// 'Context' should be a Declarator::TheContext value.
void Parser::ParseDeclaration(unsigned Context) {
  // Parse the common declaration-specifiers piece.
  DeclSpec DS;
  ParseDeclarationSpecifiers(DS);
  
  // C99 6.7.2.3p6: Handle "struct-or-union identifier;", "enum { X };"
  // declaration-specifiers init-declarator-list[opt] ';'
  if (Tok.getKind() == tok::semi) {
    // TODO: emit error on 'int;' or 'const enum foo;'.
    // if (!DS.isMissingDeclaratorOk()) Diag(...);
    
    ConsumeToken();
    return;
  }
  
  Declarator DeclaratorInfo(DS, (Declarator::TheContext)Context);
  ParseDeclarator(DeclaratorInfo);
  
  ParseInitDeclaratorListAfterFirstDeclarator(DeclaratorInfo);
}

/// ParseInitDeclaratorListAfterFirstDeclarator - Parse 'declaration' after
/// parsing 'declaration-specifiers declarator'.  This method is split out this
/// way to handle the ambiguity between top-level function-definitions and
/// declarations.
///
///       declaration: [C99 6.7]
///         declaration-specifiers init-declarator-list[opt] ';' [TODO]
/// [!C99]  init-declarator-list ';'                             [TODO]
/// [OMP]   threadprivate-directive                              [TODO]
///
///       init-declarator-list: [C99 6.7]
///         init-declarator
///         init-declarator-list ',' init-declarator
///       init-declarator: [C99 6.7]
///         declarator
///         declarator '=' initializer
///
void Parser::ParseInitDeclaratorListAfterFirstDeclarator(Declarator &D) {
  // At this point, we know that it is not a function definition.  Parse the
  // rest of the init-declarator-list.
  while (1) {
    // must be: decl-spec[opt] declarator init-declarator-list
    // Parse declarator '=' initializer.
    ExprResult Init;
    if (Tok.getKind() == tok::equal) {
      ConsumeToken();
      Init = ParseInitializer();
      if (!Init.isInvalid) {
        SkipUntil(tok::semi);
        return;
      }
    }
    
    // Inform the current actions module that we just parsed a declarator.
    Actions.ParseDeclarator(Tok.getLocation(), CurScope, D, Init.Val);
    
    // If we don't have a comma, it is either the end of the list (a ';') or an
    // error, bail out.
    if (Tok.getKind() != tok::comma)
      break;
    
    // Consume the comma.
    ConsumeToken();
    
    // Parse the next declarator.
    D.clear();
    ParseDeclarator(D);
  }
  
  if (Tok.getKind() == tok::semi) {
    ConsumeToken();
  } else {
    Diag(Tok, diag::err_parse_error);
    // Skip to end of block or statement
    SkipUntil(tok::r_brace, true);
    if (Tok.getKind() == tok::semi)
      ConsumeToken();
  }
}

/// ParseSpecifierQualifierList
///        specifier-qualifier-list:
///          type-specifier specifier-qualifier-list[opt]
///          type-qualifier specifier-qualifier-list[opt]
///
void Parser::ParseSpecifierQualifierList(DeclSpec &DS) {
  /// specifier-qualifier-list is a subset of declaration-specifiers.  Just
  /// parse declaration-specifiers and complain about extra stuff.
  SourceLocation Loc = Tok.getLocation();
  ParseDeclarationSpecifiers(DS);
  
  // Validate declspec for type-name.
  unsigned Specs = DS.getParsedSpecifiers();
  if (Specs == DeclSpec::PQ_None)
    Diag(Tok, diag::err_typename_requires_specqual);
  
  if (Specs & DeclSpec::PQ_StorageClassSpecifier) {
    Diag(Loc, diag::err_typename_invalid_storageclass);
    // Remove storage class.
    DS.StorageClassSpec     = DeclSpec::SCS_unspecified;
    DS.SCS_thread_specified = false;
  }
  if (Specs & DeclSpec::PQ_FunctionSpecifier) {
    Diag(Loc, diag::err_typename_invalid_functionspec);
    DS.FS_inline_specified = false;
  }
}

/// ParseDeclarationSpecifiers
///       declaration-specifiers: [C99 6.7]
///         storage-class-specifier declaration-specifiers[opt]
///         type-specifier declaration-specifiers[opt]
///         type-qualifier declaration-specifiers[opt]
/// [C99]   function-specifier declaration-specifiers[opt]
/// [GNU]   attributes declaration-specifiers[opt]                [TODO]
///
///       storage-class-specifier: [C99 6.7.1]
///         'typedef'
///         'extern'
///         'static'
///         'auto'
///         'register'
/// [GNU]   '__thread'
///       type-specifier: [C99 6.7.2]
///         'void'
///         'char'
///         'short'
///         'int'
///         'long'
///         'float'
///         'double'
///         'signed'
///         'unsigned'
///         struct-or-union-specifier
///         enum-specifier
///         typedef-name                          [TODO]
/// [C99]   '_Bool'
/// [C99]   '_Complex'
/// [C99]   '_Imaginary'  // Removed in TC2?
/// [GNU]   '_Decimal32'
/// [GNU]   '_Decimal64'
/// [GNU]   '_Decimal128'
/// [GNU]   typeof-specifier                      [TODO]
/// [OBJC]  class-name objc-protocol-refs[opt]    [TODO]
/// [OBJC]  typedef-name objc-protocol-refs       [TODO]
/// [OBJC]  objc-protocol-refs                    [TODO]
///       type-qualifier:
///         'const'
///         'volatile'
/// [C99]   'restrict'
///       function-specifier: [C99 6.7.4]
/// [C99]   'inline'
///
void Parser::ParseDeclarationSpecifiers(DeclSpec &DS) {
  SourceLocation StartLoc = Tok.getLocation();
  while (1) {
    int isInvalid = false;
    const char *PrevSpec = 0;
    switch (Tok.getKind()) {
    default:
      // If this is not a declaration specifier token, we're done reading decl
      // specifiers.  First verify that DeclSpec's are consistent.
      DS.Finish(StartLoc, Diags, getLang());
      return;
      
    // storage-class-specifier
    case tok::kw_typedef:
      isInvalid = DS.SetStorageClassSpec(DeclSpec::SCS_typedef, PrevSpec);
      break;
    case tok::kw_extern:
      if (DS.SCS_thread_specified)
        Diag(Tok, diag::ext_thread_before, "extern");
      isInvalid = DS.SetStorageClassSpec(DeclSpec::SCS_extern, PrevSpec);
      break;
    case tok::kw_static:
      if (DS.SCS_thread_specified)
        Diag(Tok, diag::ext_thread_before, "static");
      isInvalid = DS.SetStorageClassSpec(DeclSpec::SCS_static, PrevSpec);
      break;
    case tok::kw_auto:
      isInvalid = DS.SetStorageClassSpec(DeclSpec::SCS_auto, PrevSpec);
      break;
    case tok::kw_register:
      isInvalid = DS.SetStorageClassSpec(DeclSpec::SCS_register, PrevSpec);
      break;
    case tok::kw___thread:
      if (DS.SCS_thread_specified)
        isInvalid = 2, PrevSpec = "__thread";
      else
        DS.SCS_thread_specified = true;
      break;
      
    // type-specifiers
    case tok::kw_short:
      isInvalid = DS.SetTypeSpecWidth(DeclSpec::TSW_short, PrevSpec);
      break;
    case tok::kw_long:
      if (DS.TypeSpecWidth != DeclSpec::TSW_long) {
        isInvalid = DS.SetTypeSpecWidth(DeclSpec::TSW_long, PrevSpec);
      } else {
        DS.TypeSpecWidth = DeclSpec::TSW_unspecified;
        isInvalid = DS.SetTypeSpecWidth(DeclSpec::TSW_longlong, PrevSpec);
      }
      break;
    case tok::kw_signed:
      isInvalid = DS.SetTypeSpecSign(DeclSpec::TSS_signed, PrevSpec);
      break;
    case tok::kw_unsigned:
      isInvalid = DS.SetTypeSpecSign(DeclSpec::TSS_unsigned, PrevSpec);
      break;
    case tok::kw__Complex:
      isInvalid = DS.SetTypeSpecComplex(DeclSpec::TSC_complex, PrevSpec);
      break;
    case tok::kw__Imaginary:
      isInvalid = DS.SetTypeSpecComplex(DeclSpec::TSC_imaginary, PrevSpec);
      break;
    case tok::kw_void:
      isInvalid = DS.SetTypeSpecType(DeclSpec::TST_void, PrevSpec);
      break;
    case tok::kw_char:
      isInvalid = DS.SetTypeSpecType(DeclSpec::TST_char, PrevSpec);
      break;
    case tok::kw_int:
      isInvalid = DS.SetTypeSpecType(DeclSpec::TST_int, PrevSpec);
      break;
    case tok::kw_float:
      isInvalid = DS.SetTypeSpecType(DeclSpec::TST_float, PrevSpec);
      break;
    case tok::kw_double:
      isInvalid = DS.SetTypeSpecType(DeclSpec::TST_double, PrevSpec);
      break;
    case tok::kw__Bool:
      isInvalid = DS.SetTypeSpecType(DeclSpec::TST_bool, PrevSpec);
      break;
    case tok::kw__Decimal32:
      isInvalid = DS.SetTypeSpecType(DeclSpec::TST_decimal32, PrevSpec);
      break;
    case tok::kw__Decimal64:
      isInvalid = DS.SetTypeSpecType(DeclSpec::TST_decimal64, PrevSpec);
      break;
    case tok::kw__Decimal128:
      isInvalid = DS.SetTypeSpecType(DeclSpec::TST_decimal128, PrevSpec);
      break;
      
    case tok::kw_struct:
    case tok::kw_union:
      ParseStructUnionSpecifier(DS);
      continue;
    case tok::kw_enum:
      ParseEnumSpecifier(DS);
      continue;
    
    //case tok::identifier:
    // TODO: handle typedef names.
      
    // type-qualifier
    case tok::kw_const:
      isInvalid = DS.SetTypeQual(DeclSpec::TQ_const   , PrevSpec, getLang())*2;
      break;
    case tok::kw_volatile:
      isInvalid = DS.SetTypeQual(DeclSpec::TQ_volatile, PrevSpec, getLang())*2;
      break;
    case tok::kw_restrict:
      isInvalid = DS.SetTypeQual(DeclSpec::TQ_restrict, PrevSpec, getLang())*2;
      break;
      
    // function-specifier
    case tok::kw_inline:
      // 'inline inline' is ok.
      DS.FS_inline_specified = true;
      break;
    }
    // If the specifier combination wasn't legal, issue a diagnostic.
    if (isInvalid) {
      assert(PrevSpec && "Method did not return previous specifier!");
      if (isInvalid == 1)  // Error.
        Diag(Tok, diag::err_invalid_decl_spec_combination, PrevSpec);
      else                 // extwarn.
        Diag(Tok, diag::ext_duplicate_declspec, PrevSpec);
    }
    ConsumeToken();
  }
}


/// ParseStructUnionSpecifier
///       struct-or-union-specifier: [C99 6.7.2.1]
///         struct-or-union identifier[opt] '{' struct-contents '}'
///         struct-or-union identifier
///       struct-or-union:
///         'struct'
///         'union'
///       struct-contents:
///         struct-declaration-list
/// [EXT]   empty
/// [GNU]   "struct-declaration-list" without terminatoring ';'   [TODO]
///       struct-declaration-list:
///         struct-declaration
///         struct-declaration-list struct-declaration
/// [OBC]   '@' 'defs' '(' class-name ')'                         [TODO]
///       struct-declaration:
///         specifier-qualifier-list struct-declarator-list ';'
/// [GNU]   __extension__ struct-declaration                       [TODO]
/// [GNU]   specifier-qualifier-list ';'                           [TODO]
///       struct-declarator-list:
///         struct-declarator
///         struct-declarator-list ',' struct-declarator
///       struct-declarator:
///         declarator
///         declarator[opt] ':' constant-expression
///
void Parser::ParseStructUnionSpecifier(DeclSpec &DS) {
  assert((Tok.getKind() == tok::kw_struct ||
          Tok.getKind() == tok::kw_union) && "Not a struct/union specifier");
  SourceLocation Start = Tok.getLocation();
  bool isUnion = Tok.getKind() == tok::kw_union;
  ConsumeToken();
  
  // Must have either 'struct name' or 'struct {...}'.
  if (Tok.getKind() != tok::identifier &&
      Tok.getKind() != tok::l_brace) {
    Diag(Tok, diag::err_expected_ident_lbrace);
    return;
  }
  
  if (Tok.getKind() == tok::identifier)
    ConsumeToken();
  
  if (Tok.getKind() == tok::l_brace) {
    SourceLocation LBraceLoc = Tok.getLocation();
    ConsumeBrace();

    if (Tok.getKind() == tok::r_brace)
      Diag(Tok, diag::ext_empty_struct_union_enum, isUnion ? "union":"struct");

    while (Tok.getKind() != tok::r_brace && 
           Tok.getKind() != tok::eof) {
      // Each iteration of this loop reads one struct-declaration.

      // Parse the common specifier-qualifiers-list piece.
      DeclSpec DS;
      SourceLocation SpecQualLoc = Tok.getLocation();
      ParseSpecifierQualifierList(DS);
      // TODO: Does specifier-qualifier list correctly check that *something* is
      // specified?
      
      Declarator DeclaratorInfo(DS, Declarator::MemberContext);

      // If there are no declarators, issue a warning.
      if (Tok.getKind() == tok::semi) {
        Diag(SpecQualLoc, diag::w_no_declarators);
      } else {
        // Read struct-declarators until we find the semicolon.
        while (1) {
          /// struct-declarator: declarator
          /// struct-declarator: declarator[opt] ':' constant-expression
          if (Tok.getKind() != tok::colon)
            ParseDeclarator(DeclaratorInfo);
          
          if (Tok.getKind() == tok::colon) {
            ConsumeToken();
            ExprResult Res = ParseConstantExpression();
            if (Res.isInvalid) {
              SkipUntil(tok::semi, true, true);
            } else {
              // Process it.
            }
          }

          // TODO: install declarator.
          
          // If we don't have a comma, it is either the end of the list (a ';')
          // or an error, bail out.
          if (Tok.getKind() != tok::comma)
            break;
          
          // Consume the comma.
          ConsumeToken();
          
          // Parse the next declarator.
          DeclaratorInfo.clear();
        }
      }
      
      if (Tok.getKind() == tok::semi) {
        ConsumeToken();
      } else {
        Diag(Tok, diag::err_expected_semi_decl_list);
        // Skip to end of block or statement
        SkipUntil(tok::r_brace, true, true);
      }
    }

    MatchRHSPunctuation(tok::r_brace, LBraceLoc, "{",diag::err_expected_rbrace);
  }

  const char *PrevSpec = 0;
  if (DS.SetTypeSpecType(isUnion ? DeclSpec::TST_union : DeclSpec::TST_struct,
                         PrevSpec))
    Diag(Start, diag::err_invalid_decl_spec_combination, PrevSpec);
}


/// ParseEnumSpecifier
///       enum-specifier: [C99 6.7.2.2]
///         'enum' identifier[opt] '{' enumerator-list '}'
/// [C99]   'enum' identifier[opt] '{' enumerator-list ',' '}'
/// [GNU]   'enum' identifier[opt] '{' enumerator-list '}' attributes [TODO]
/// [GNU]   'enum' identifier[opt] '{' enumerator-list ',' '}' attributes [TODO]
///         'enum' identifier
///       enumerator-list:
///         enumerator
///         enumerator-list ',' enumerator
///       enumerator:
///         enumeration-constant
///         enumeration-constant '=' constant-expression
///       enumeration-constant:
///         identifier
///
void Parser::ParseEnumSpecifier(DeclSpec &DS) {
  assert(Tok.getKind() == tok::kw_enum && "Not an enum specifier");
  SourceLocation Start = Tok.getLocation();
  ConsumeToken();
  
  // Must have either 'enum name' or 'enum {...}'.
  if (Tok.getKind() != tok::identifier &&
      Tok.getKind() != tok::l_brace) {
    Diag(Tok, diag::err_expected_ident_lbrace);
    return;
  }
  
  if (Tok.getKind() == tok::identifier)
    ConsumeToken();
  
  if (Tok.getKind() != tok::l_brace)
    return;
  
  SourceLocation LBraceLoc = Tok.getLocation();
  ConsumeBrace();
  
  if (Tok.getKind() == tok::r_brace)
    Diag(Tok, diag::ext_empty_struct_union_enum, "enum");
  
  // Parse the enumerator-list.
  while (Tok.getKind() == tok::identifier) {
    ConsumeToken();
    
    if (Tok.getKind() == tok::equal) {
      ConsumeToken();
      ExprResult Res = ParseConstantExpression();
      if (Res.isInvalid) SkipUntil(tok::comma, true, false);
    }
    
    if (Tok.getKind() != tok::comma)
      break;
    SourceLocation CommaLoc = Tok.getLocation();
    ConsumeToken();
    
    if (Tok.getKind() != tok::identifier && !getLang().C99)
      Diag(CommaLoc, diag::ext_c99_enumerator_list_comma);
  }
  
  // Eat the }.
  MatchRHSPunctuation(tok::r_brace, LBraceLoc, "{", 
                      diag::err_expected_rbrace);
  // TODO: semantic analysis on the declspec for enums.
  
  
  const char *PrevSpec = 0;
  if (DS.SetTypeSpecType(DeclSpec::TST_enum, PrevSpec))
    Diag(Start, diag::err_invalid_decl_spec_combination, PrevSpec);
}


/// isTypeSpecifierQualifier - Return true if the current token could be the
/// start of a specifier-qualifier-list.
bool Parser::isTypeSpecifierQualifier() const {
  switch (Tok.getKind()) {
  default: return false;
    // type-specifiers
  case tok::kw_short:
  case tok::kw_long:
  case tok::kw_signed:
  case tok::kw_unsigned:
  case tok::kw__Complex:
  case tok::kw__Imaginary:
  case tok::kw_void:
  case tok::kw_char:
  case tok::kw_int:
  case tok::kw_float:
  case tok::kw_double:
  case tok::kw__Bool:
  case tok::kw__Decimal32:
  case tok::kw__Decimal64:
  case tok::kw__Decimal128:
    
    // struct-or-union-specifier
  case tok::kw_struct:
  case tok::kw_union:
    // enum-specifier
  case tok::kw_enum:
    
    // type-qualifier
  case tok::kw_const:
  case tok::kw_volatile:
  case tok::kw_restrict:
    return true;
    
    // typedef-name
  case tok::identifier:
    // FIXME: if this is a typedef return true.
    return false;
    
    // TODO: Attributes.
  }
}

/// isDeclarationSpecifier() - Return true if the current token is part of a
/// declaration specifier.
bool Parser::isDeclarationSpecifier() const {
  switch (Tok.getKind()) {
  default: return false;
    // storage-class-specifier
  case tok::kw_typedef:
  case tok::kw_extern:
  case tok::kw_static:
  case tok::kw_auto:
  case tok::kw_register:
  case tok::kw___thread:
    
    // type-specifiers
  case tok::kw_short:
  case tok::kw_long:
  case tok::kw_signed:
  case tok::kw_unsigned:
  case tok::kw__Complex:
  case tok::kw__Imaginary:
  case tok::kw_void:
  case tok::kw_char:
  case tok::kw_int:
  case tok::kw_float:
  case tok::kw_double:
  case tok::kw__Bool:
  case tok::kw__Decimal32:
  case tok::kw__Decimal64:
  case tok::kw__Decimal128:
  
    // struct-or-union-specifier
  case tok::kw_struct:
  case tok::kw_union:
    // enum-specifier
  case tok::kw_enum:
    
    // type-qualifier
  case tok::kw_const:
  case tok::kw_volatile:
  case tok::kw_restrict:
    
    // function-specifier
  case tok::kw_inline:
    return true;
    
    // typedef-name
  case tok::identifier:
    // FIXME: if this is a typedef return true.
    return false;
    // TODO: Attributes.
  }
}


/// ParseTypeQualifierListOpt
///       type-qualifier-list: [C99 6.7.5]
///         type-qualifier
/// [GNU]   attributes                         [TODO]
///         type-qualifier-list type-qualifier
/// [GNU]   type-qualifier-list attributes     [TODO]
///
void Parser::ParseTypeQualifierListOpt(DeclSpec &DS) {
  SourceLocation StartLoc = Tok.getLocation();
  while (1) {
    int isInvalid = false;
    const char *PrevSpec = 0;

    switch (Tok.getKind()) {
    default:
      // If this is not a declaration specifier token, we're done reading decl
      // specifiers.  First verify that DeclSpec's are consistent.
      DS.Finish(StartLoc, Diags, getLang());
      return;
      // TODO: attributes.
    case tok::kw_const:
      isInvalid = DS.SetTypeQual(DeclSpec::TQ_const   , PrevSpec, getLang())*2;
      break;
    case tok::kw_volatile:
      isInvalid = DS.SetTypeQual(DeclSpec::TQ_volatile, PrevSpec, getLang())*2;
      break;
    case tok::kw_restrict:
      isInvalid = DS.SetTypeQual(DeclSpec::TQ_restrict, PrevSpec, getLang())*2;
      break;
    }
    
    // If the specifier combination wasn't legal, issue a diagnostic.
    if (isInvalid) {
      assert(PrevSpec && "Method did not return previous specifier!");
      if (isInvalid == 1)  // Error.
        Diag(Tok, diag::err_invalid_decl_spec_combination, PrevSpec);
      else                 // extwarn.
        Diag(Tok, diag::ext_duplicate_declspec, PrevSpec);
    }
    ConsumeToken();
  }
}


/// ParseDeclarator - Parse and verify a newly-initialized declarator.
///
void Parser::ParseDeclarator(Declarator &D) {
  /// This implements the 'declarator' production in the C grammar, then checks
  /// for well-formedness and issues diagnostics.
  ParseDeclaratorInternal(D);
  
  // TODO: validate D.

}

/// ParseDeclaratorInternal
///       declarator: [C99 6.7.5]
///         pointer[opt] direct-declarator
///
///       pointer: [C99 6.7.5]
///         '*' type-qualifier-list[opt]
///         '*' type-qualifier-list[opt] pointer
///
void Parser::ParseDeclaratorInternal(Declarator &D) {
  if (Tok.getKind() != tok::star)
    return ParseDirectDeclarator(D);
  
  // Otherwise, '*' -> pointer.
  SourceLocation Loc = Tok.getLocation();
  ConsumeToken();  // Eat the *.
  DeclSpec DS;
  ParseTypeQualifierListOpt(DS);
  
  // Recursively parse the declarator.
  ParseDeclaratorInternal(D);
  
  // Remember that we parsed a pointer type, and remember the type-quals.
  D.AddTypeInfo(DeclaratorTypeInfo::getPointer(DS.TypeQualifiers, Loc));
}


/// ParseDirectDeclarator
///       direct-declarator: [C99 6.7.5]
///         identifier
///         '(' declarator ')'
/// [GNU]   '(' attributes declarator ')'
/// [C90]   direct-declarator '[' constant-expression[opt] ']'
/// [C99]   direct-declarator '[' type-qual-list[opt] assignment-expr[opt] ']'
/// [C99]   direct-declarator '[' 'static' type-qual-list[opt] assign-expr ']'
/// [C99]   direct-declarator '[' type-qual-list 'static' assignment-expr ']'
/// [C99]   direct-declarator '[' type-qual-list[opt] '*' ']'
///         direct-declarator '(' parameter-type-list ')'
///         direct-declarator '(' identifier-list[opt] ')'
/// [GNU]   direct-declarator '(' parameter-forward-declarations
///                    parameter-type-list[opt] ')'
///
void Parser::ParseDirectDeclarator(Declarator &D) {
  // Parse the first direct-declarator seen.
  if (Tok.getKind() == tok::identifier && D.mayHaveIdentifier()) {
    assert(Tok.getIdentifierInfo() && "Not an identifier?");
    D.SetIdentifier(Tok.getIdentifierInfo(), Tok.getLocation());
    ConsumeToken();
  } else if (Tok.getKind() == tok::l_paren) {
    // direct-declarator: '(' declarator ')'
    // direct-declarator: '(' attributes declarator ')'   [TODO]
    // Example: 'char (*X)'   or 'int (*XX)(void)'
    ParseParenDeclarator(D);
  } else if (D.mayOmitIdentifier()) {
    // This could be something simple like "int" (in which case the declarator
    // portion is empty), if an abstract-declarator is allowed.
    D.SetIdentifier(0, Tok.getLocation());
  } else {
    // Expected identifier or '('.
    Diag(Tok, diag::err_expected_ident_lparen);
    D.SetIdentifier(0, Tok.getLocation());
  }
  
  assert(D.isPastIdentifier() &&
         "Haven't past the location of the identifier yet?");
  
  while (1) {
    if (Tok.getKind() == tok::l_paren) {
      ParseParenDeclarator(D);
    } else if (Tok.getKind() == tok::l_square) {
      ParseBracketDeclarator(D);
    } else {
      break;
    }
  }
}

/// ParseParenDeclarator - We parsed the declarator D up to a paren.  This may
/// either be before the identifier (in which case these are just grouping
/// parens for precedence) or it may be after the identifier, in which case
/// these are function arguments.
///
/// This method also handles this portion of the grammar:
///       parameter-type-list: [C99 6.7.5]
///         parameter-list
///         parameter-list ',' '...'
///
///       parameter-list: [C99 6.7.5]
///         parameter-declaration
///         parameter-list ',' parameter-declaration
///
///       parameter-declaration: [C99 6.7.5]
///         declaration-specifiers declarator
/// [GNU]   declaration-specifiers declarator attributes               [TODO]
///         declaration-specifiers abstract-declarator[opt] 
/// [GNU]   declaration-specifiers abstract-declarator[opt] attributes [TODO]
///
///       identifier-list: [C99 6.7.5]
///         identifier
///         identifier-list ',' identifier
///
void Parser::ParseParenDeclarator(Declarator &D) {
  SourceLocation StartLoc = Tok.getLocation();
  ConsumeParen();
  
  // If we haven't past the identifier yet (or where the identifier would be
  // stored, if this is an abstract declarator), then this is probably just
  // grouping parens.
  if (!D.isPastIdentifier()) {
    // Okay, this is probably a grouping paren.  However, if this could be an
    // abstract-declarator, then this could also be the start of function
    // arguments (consider 'void()').
    bool isGrouping;
    
    if (!D.mayOmitIdentifier()) {
      // If this can't be an abstract-declarator, this *must* be a grouping
      // paren, because we haven't seen the identifier yet.
      isGrouping = true;
    } else if (Tok.getKind() == tok::r_paren ||  // 'int()' is a function.
               isDeclarationSpecifier()) {       // 'int(int)' is a function.
      
      isGrouping = false;
    } else {
      // Otherwise, this is a grouping paren, e.g. 'int (*X)'.
      isGrouping = true;
    }
    
    // If this is a grouping paren, handle:
    // direct-declarator: '(' declarator ')'
    // direct-declarator: '(' attributes declarator ')'   [TODO]
    if (isGrouping) {
      ParseDeclaratorInternal(D);
      // Match the ')'.
      MatchRHSPunctuation(tok::r_paren, StartLoc, "(",
                          diag::err_expected_rparen);
      return;
    }
    
    // Okay, if this wasn't a grouping paren, it must be the start of a function
    // argument list.  Recognize that this declarator will never have an
    // identifier (and remember where it would have been), then fall through to
    // the handling of argument lists.
    D.SetIdentifier(0, Tok.getLocation());
  }
  
  // Okay, this is the parameter list of a function definition, or it is an
  // identifier list of a K&R-style function.

  // TODO: enter function-declaration scope, limiting any declarators for
  // arguments to the function scope.
  // NOTE: better to only create a scope if not '()'
  bool IsVariadic;
  bool HasPrototype;
  bool IsEmpty = false;
  bool ErrorEmitted = false;

  if (Tok.getKind() == tok::r_paren) {
    // int() -> no prototype, no '...'.
    IsVariadic   = false;
    HasPrototype = false;
    IsEmpty      = true;
  } else if (Tok.getKind() == tok::identifier &&
             1/*TODO: !isatypedefname(Tok.getIdentifierInfo())*/) {
    // Identifier list.  Note that '(' identifier-list ')' is only allowed for
    // normal declarators, not for abstract-declarators.
    assert(D.isPastIdentifier() && "Identifier (if present) must be passed!");
    
    // If there was no identifier specified, either we are in an
    // abstract-declarator, or we are in a parameter declarator which was found
    // to be abstract.  In abstract-declarators, identifier lists are not valid,
    // diagnose this.
    if (!D.getIdentifier())
      Diag(Tok, diag::ext_ident_list_in_param);
    
    // TODO: Remember token.
    ConsumeToken();
    while (Tok.getKind() == tok::comma) {
      // Eat the comma.
      ConsumeToken();
      
      if (ExpectAndConsume(tok::identifier, diag::err_expected_ident)) {
        ErrorEmitted = true;
        break;
      }
    }
    
    // K&R 'prototype'.
    IsVariadic = false;
    HasPrototype = false;
  } else {
    IsVariadic = false;
    bool ReadArg = false;
    // Finally, a normal, non-empty parameter type list.
    while (1) {
      if (Tok.getKind() == tok::ellipsis) {
        IsVariadic = true;

        // Check to see if this is "void(...)" which is not allowed.
        if (!ReadArg) {
          // Otherwise, parse parameter type list.  If it starts with an
          // ellipsis,  diagnose the malformed function.
          Diag(Tok, diag::err_ellipsis_first_arg);
          IsVariadic = false;       // Treat this like 'void()'.
        }

        // Consume the ellipsis.
        ConsumeToken();
        break;
      }
      
      ReadArg = true;

      // Parse the declaration-specifiers.
      DeclSpec DS;
      ParseDeclarationSpecifiers(DS);

      // Parse the declarator.  This is "PrototypeContext", because we must
      // accept either 'declarator' or 'abstract-declarator' here.
      Declarator DeclaratorInfo(DS, Declarator::PrototypeContext);
      ParseDeclarator(DeclaratorInfo);

      // TODO: do something with the declarator, if it is valid.
      
      // If the next token is a comma, consume it and keep reading arguments.
      if (Tok.getKind() != tok::comma) break;
      
      // Consume the comma.
      ConsumeToken();
    }
    
    HasPrototype = true;
  }
  
  // TODO: pop the scope.  

  // TODO: capture argument info.
  
  // Remember that we parsed a function type, and remember the attributes.
  D.AddTypeInfo(DeclaratorTypeInfo::getFunction(HasPrototype, IsVariadic,
                                                IsEmpty, StartLoc));
  
  
  // If we have the closing ')', eat it and we're done.
  if (Tok.getKind() == tok::r_paren) {
    ConsumeParen();
  } else {
    // If an error happened earlier parsing something else in the proto, don't
    // issue another error.
    if (!ErrorEmitted)
      Diag(Tok, diag::err_expected_rparen);
    SkipUntil(tok::r_paren);
  }
}


/// [C90]   direct-declarator '[' constant-expression[opt] ']'
/// [C99]   direct-declarator '[' type-qual-list[opt] assignment-expr[opt] ']'
/// [C99]   direct-declarator '[' 'static' type-qual-list[opt] assign-expr ']'
/// [C99]   direct-declarator '[' type-qual-list 'static' assignment-expr ']'
/// [C99]   direct-declarator '[' type-qual-list[opt] '*' ']'
void Parser::ParseBracketDeclarator(Declarator &D) {
  SourceLocation StartLoc = Tok.getLocation();
  ConsumeBracket();
  
  // If valid, this location is the position where we read the 'static' keyword.
  SourceLocation StaticLoc;
  if (Tok.getKind() == tok::kw_static) {
    StaticLoc = Tok.getLocation();
    ConsumeToken();
  }
  
  // If there is a type-qualifier-list, read it now.
  DeclSpec DS;
  ParseTypeQualifierListOpt(DS);
  
  // If we haven't already read 'static', check to see if there is one after the
  // type-qualifier-list.
  if (!StaticLoc.isValid() && Tok.getKind() == tok::kw_static) {
    StaticLoc = Tok.getLocation();
    ConsumeToken();
  }
  
  // Handle "direct-declarator [ type-qual-list[opt] * ]".
  bool isStar = false;
  ExprResult NumElements(false);
  if (Tok.getKind() == tok::star) {
    // Remember the '*' token, in case we have to un-get it.
    LexerToken StarTok = Tok;
    ConsumeToken();

    // Check that the ']' token is present to avoid incorrectly parsing
    // expressions starting with '*' as [*].
    if (Tok.getKind() == tok::r_square) {
      if (StaticLoc.isValid())
        Diag(StaticLoc, diag::err_unspecified_vla_size_with_static);
      StaticLoc = SourceLocation();  // Drop the static.
      isStar = true;
    } else {
      // Otherwise, the * must have been some expression (such as '*ptr') that
      // started an assignment-expr.  We already consumed the token, but now we
      // need to reparse it.  This handles cases like 'X[*p + 4]'
      NumElements = ParseAssignmentExpressionWithLeadingStar(StarTok);
    }
  } else if (Tok.getKind() != tok::r_square) {
    // Parse the assignment-expression now.
    NumElements = ParseAssignmentExpression();
  }
  
  // If there was an error parsing the assignment-expression, recover.
  if (NumElements.isInvalid) {
    // If the expression was invalid, skip it.
    SkipUntil(tok::r_square);
    return;
  }
  
  MatchRHSPunctuation(tok::r_square, StartLoc, "[", diag::err_expected_rsquare);
    
  // If C99 isn't enabled, emit an ext-warn if the arg list wasn't empty and if
  // it was not a constant expression.
  if (!getLang().C99) {
    // TODO: check C90 array constant exprness.
    if (isStar || StaticLoc.isValid() ||
        0/*TODO: NumElts is not a C90 constantexpr */)
      Diag(StartLoc, diag::ext_c99_array_usage);
  }
  
  // Remember that we parsed a pointer type, and remember the type-quals.
  D.AddTypeInfo(DeclaratorTypeInfo::getArray(DS.TypeQualifiers,
                                             StaticLoc.isValid(), isStar,
                                             NumElements.Val, StartLoc));
}

