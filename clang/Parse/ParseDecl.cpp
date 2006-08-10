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

/// ParseDeclaration - Parse a full 'declaration', which consists of
/// declaration-specifiers, some number of declarators, and a semicolon.
/// 'Context' should be a Declarator::TheContext value.
void Parser::ParseDeclaration(unsigned Context) {
  // Parse the common declaration-specifiers piece.
  DeclSpec DS;
  ParseDeclarationSpecifiers(DS);
  
  Declarator DeclaratorInfo(DS, (Declarator::TheContext)Context);
  ParseDeclarator(DeclaratorInfo);
  
  ParseInitDeclaratorListAfterFirstDeclarator(DeclaratorInfo);
}

void Parser::ParseInitDeclaratorListAfterFirstDeclarator(Declarator &D) {
  // At this point, we know that it is not a function definition.  Parse the
  // rest of the init-declarator-list.
  while (1) {
    // must be: decl-spec[opt] declarator init-declarator-list
    // Parse declarator '=' initializer.
    if (Tok.getKind() == tok::equal) {
      ConsumeToken();
      // FIXME: THIS IS WRONG: should ParseInitializer!!
      ParseExpression();
    }
    
    
    // TODO: install declarator.
    
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


/// ParseDeclarationSpecifiers
///       declaration-specifiers: [C99 6.7]
///         storage-class-specifier declaration-specifiers [opt]
///         type-specifier declaration-specifiers [opt]
///         type-qualifier declaration-specifiers [opt]
/// [C99]   function-specifier declaration-specifiers [opt]
/// [GNU]   attributes declaration-specifiers [opt]                [TODO]
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
///         struct-or-union-specifier             [TODO]
///         enum-specifier                        [TODO]
///         typedef-name                          [TODO]
/// [C99]   '_Bool'
/// [C99]   '_Complex'
/// [C99]   '_Imaginary'  // Removed in TC2?
/// [GNU]   '_Decimal32'
/// [GNU]   '_Decimal64'
/// [GNU]   '_Decimal128'
/// [GNU]   typeof-specifier                      [TODO]
/// [OBJC]  class-name objc-protocol-refs [opt]   [TODO]
/// [OBJC]  typedef-name objc-protocol-refs       [TODO]
/// [OBJC]  objc-protocol-refs                    [TODO]
///       type-qualifier:
///         const
///         volatile
/// [C99]   restrict
///       function-specifier: [C99 6.7.4]
/// [C99]   inline
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
      
    //case tok::kw_struct:
    //case tok::kw_union:
    //case tok::kw_enum:
    
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
  
  // FIXME: validate D.

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
      if (Tok.getKind() == tok::r_paren) {
        ConsumeParen();
      } else {
        // expected ')': skip until we find ')'.
        Diag(Tok, diag::err_expected_rparen);
        Diag(StartLoc, diag::err_matching);
        SkipUntil(tok::r_paren);
      }
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

  // FIXME: enter function-declaration scope, limiting any declarators for
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
    
    // FIXME: Remember token.
    ConsumeToken();
    while (Tok.getKind() == tok::comma) {
      // Eat the comma.
      ConsumeToken();
      
      if (Tok.getKind() != tok::identifier) {
        // If not identifier, diagnose the error.
        Diag(Tok, diag::err_expected_ident);
        ErrorEmitted = true;
        break;
      }

      // Eat the id.
      // FIXME: remember it!
      ConsumeToken();
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
  
  // FIXME: pop the scope.  

  // FIXME: capture argument info.
  
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
      // started an assign-expr.  We already consumed the token, but now we need
      // to reparse it.
      // FIXME: We must push 'StarTok' and Tok back into the preprocessor as a
      // macro expansion context, so they will be read again. It is basically
      // impossible to refudge the * in otherwise, due to cases like X[*p + 4].
      assert(0 && "FIXME: int X[*p] unimplemented");
    }
  }
  
  void *NumElts = 0;
  if (!isStar && Tok.getKind() != tok::r_square) {
    // Parse the assignment-expression now.
    NumElts = /*FIXME: parse array size expr*/0;
    assert(0 && "expr parsing not impl yet!");
  }
  
  ConsumeBracket();
  
  // If C99 isn't enabled, emit an ext-warn if the arg list wasn't empty and if
  // it was not a constant expression.
  if (!getLang().C99) {
    // TODO: check C90 array constant exprness.
    if (isStar || StaticLoc.isValid() || 0/*NumElts is constantexpr*/)
      Diag(StartLoc, diag::ext_c99_array_usage);
  }
  
  // Remember that we parsed a pointer type, and remember the type-quals.
  D.AddTypeInfo(DeclaratorTypeInfo::getArray(DS.TypeQualifiers,
                                             StaticLoc.isValid(), isStar,
                                             NumElts, StartLoc));
}

