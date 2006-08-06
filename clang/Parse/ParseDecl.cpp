//===--- ParserDeclarations.cpp - Declaration Parsing ---------------------===//
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


/// ParseDeclarator
///       declarator: [C99 6.7.5]
///         pointer[opt] direct-declarator
///
///       pointer: [C99 6.7.5]
///         '*' type-qualifier-list[opt]
///         '*' type-qualifier-list[opt] pointer
///
void Parser::ParseDeclarator(Declarator &D) {
  while (Tok.getKind() == tok::star) {  // '*' -> pointer.
    ConsumeToken();  // Eat the *.
    DeclSpec DS;
    ParseTypeQualifierListOpt(DS);
    // TODO: do something with DS.
  }
  
  ParseDirectDeclarator(D);
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


/// ParseDirectDeclarator
///       direct-declarator: [C99 6.7.5]
///         identifier
///         '(' declarator ')'
/// [GNU]   '(' attributes declarator ')'
/// [C90]   direct-declarator [ constant-expression[opt] ] 
/// [C99]   direct-declarator [ type-qual-list[opt] assignment-expr[opt] ]
/// [C99]   direct-declarator [ 'static' type-qual-list[opt] assignment-expr ]
/// [C99]   direct-declarator [ type-qual-list 'static' assignment-expr ]
/// [C99]   direct-declarator [ type-qual-list[opt] * ]
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
  } else if (Tok.getKind() == tok::l_square &&
             D.mayOmitIdentifier()) {
    // direct-abstract-declarator[opt] '[' assignment-expression[opt] ']'
    // direct-abstract-declarator[opt] '[' '*' ']'
    
    // direct-abstract-declarator was not specified.  Remember that this is the
    // place where the identifier would have been.
    D.SetIdentifier(0, Tok.getLocation());
    // Don't consume the '[', handle it below.
  } else if (D.mayOmitIdentifier()) {
    // This could be something simple like "int" (in which case the declarator
    // portion is empty), if an abstract-declarator is allowed.
    D.SetIdentifier(0, Tok.getLocation());
  } else {
    // expected identifier or '(' or '['.
    assert(0 && "ERROR: should recover!");
  }
  
  assert(D.isPastIdentifier() &&
         "Haven't past the location of the identifier yet?");
  
  while (1) {
    if (Tok.getKind() == tok::l_paren) {
      ParseParenDeclarator(D);
    } else if (Tok.getKind() == tok::l_square) {
      assert(0 && "Unimp!");
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
      // Otherwise, 'int (*X)', this is a grouping paren.
      isGrouping = true;
    }
    
    // If this is a grouping paren, handle:
    // direct-declarator: '(' declarator ')'
    // direct-declarator: '(' attributes declarator ')'   [TODO]
    if (isGrouping) {
      ParseDeclarator(D);
      // expected ')': skip until we find ')'.
     if (Tok.getKind() != tok::r_paren)
        assert(0 && "Recover!");
      ConsumeParen();
      return;
    }
    
    // Okay, if this wasn't a grouping paren, it must be the start of a function
    // argument list.  Recognize that this will never have an identifier (and
    // where it would be), then fall through to the handling of argument lists.
    D.SetIdentifier(0, Tok.getLocation());
  }
  
  // Okay, this is the parameter list of a function definition, or it is an
  // identifier list of a K&R-style function.

  // FIXME: enter function-declaration scope, limiting any declarators for
  // arguments to the function scope.
  // NOTE: better to only create a scope if not '()'
  bool isVariadic;
  bool HasPrototype;
  if (Tok.getKind() == tok::r_paren) {
    // int() -> no prototype, no '...'.
    isVariadic   = false;
    HasPrototype = false;
  } else if (Tok.getKind() == tok::identifier &&
             0/*TODO: !isatypedefname(Tok.getIdentifierInfo())*/) {
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
      
      // FIXME: if not identifier, consume until ')' then break.
      assert(Tok.getKind() == tok::identifier);

      // Eat the id.
      // FIXME: remember it!
      ConsumeToken();
    }
    
    // FIXME: if not identifier, consume until ')' then break.
    assert(Tok.getKind() == tok::r_paren);

    // K&R 'prototype'.
    isVariadic = false;
    HasPrototype = false;
  } else {
    isVariadic = false;
    bool ReadArg = false;
    // Finally, a normal, non-empty parameter type list.
    while (1) {
      if (Tok.getKind() == tok::ellipsis) {
        isVariadic = true;

        // Check to see if this is "void(...)" which is not allowed.
        if (!ReadArg) {
          // Otherwise, parse parameter type list.  If it starts with an ellipsis, 
          // diagnose the malformed function.
          Diag(Tok, diag::err_ellipsis_first_arg);
          isVariadic = false;       // Treat this like 'void()'.
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
  
  
  // expected ')': skip until we find ')'.
  if (Tok.getKind() != tok::r_paren)
    assert(0 && "Recover!");
  ConsumeParen();
}

