//===--- ParseDecl.cpp - Declaration Parsing ------------------------------===//
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
#include "clang/Parse/DeclSpec.h"
#include "llvm/ADT/SmallSet.h"
using namespace llvm;
using namespace clang;

//===----------------------------------------------------------------------===//
// C99 6.7: Declarations.
//===----------------------------------------------------------------------===//

/// ParseTypeName
///       type-name: [C99 6.7.6]
///         specifier-qualifier-list abstract-declarator[opt]
Parser::TypeTy *Parser::ParseTypeName() {
  // Parse the common declaration-specifiers piece.
  DeclSpec DS;
  ParseSpecifierQualifierList(DS);
  
  // Parse the abstract-declarator, if present.
  Declarator DeclaratorInfo(DS, Declarator::TypeNameContext);
  ParseDeclarator(DeclaratorInfo);
  
  return Actions.ParseTypeName(CurScope, DeclaratorInfo).Val;
}

/// ParseAttributes - Parse a non-empty attributes list.
///
/// [GNU] attributes:
///         attribute
///         attributes attribute
///
/// [GNU]  attribute:
///          '__attribute__' '(' '(' attribute-list ')' ')'
///
/// [GNU]  attribute-list:
///          attrib
///          attribute_list ',' attrib
///
/// [GNU]  attrib:
///          empty
///          any-word
///          any-word '(' identifier ')'
///          any-word '(' identifier ',' nonempty-expr-list ')'
///          any-word '(' expr-list ')'
///
void Parser::ParseAttributes() {
  assert(Tok.getKind() == tok::kw___attribute && "Not an attribute list!");
  ConsumeToken();
  
  if (ExpectAndConsume(tok::l_paren, diag::err_expected_lparen_after,
                       "attribute"))
    return;
  
  // TODO: Parse the attributes.
  SkipUntil(tok::r_paren, false);
}


/// ParseDeclaration - Parse a full 'declaration', which consists of
/// declaration-specifiers, some number of declarators, and a semicolon.
/// 'Context' should be a Declarator::TheContext value.
Parser::DeclTy *Parser::ParseDeclaration(unsigned Context) {
  // Parse the common declaration-specifiers piece.
  DeclSpec DS;
  ParseDeclarationSpecifiers(DS);
  
  // C99 6.7.2.3p6: Handle "struct-or-union identifier;", "enum { X };"
  // declaration-specifiers init-declarator-list[opt] ';'
  if (Tok.getKind() == tok::semi) {
    ConsumeToken();
    return Actions.ParsedFreeStandingDeclSpec(CurScope, DS);
  }
  
  Declarator DeclaratorInfo(DS, (Declarator::TheContext)Context);
  ParseDeclarator(DeclaratorInfo);
  
  return ParseInitDeclaratorListAfterFirstDeclarator(DeclaratorInfo);
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
/// [GNU]   declarator simple-asm-expr[opt] attributes[opt]
/// [GNU]   declarator simple-asm-expr[opt] attributes[opt] '=' initializer
///
Parser::DeclTy *Parser::
ParseInitDeclaratorListAfterFirstDeclarator(Declarator &D) {
  
  // Declarators may be grouped together ("int X, *Y, Z();").  Provide info so
  // that they can be chained properly if the actions want this.
  Parser::DeclTy *LastDeclInGroup = 0;
  
  // At this point, we know that it is not a function definition.  Parse the
  // rest of the init-declarator-list.
  while (1) {
    // If a simple-asm-expr is present, parse it.
    if (Tok.getKind() == tok::kw_asm)
      ParseSimpleAsm();
    
    // If attributes are present, parse them.
    if (Tok.getKind() == tok::kw___attribute)
      ParseAttributes();
    
    // Parse declarator '=' initializer.
    ExprResult Init;
    if (Tok.getKind() == tok::equal) {
      ConsumeToken();
      Init = ParseInitializer();
      if (Init.isInvalid) {
        SkipUntil(tok::semi);
        return 0;
      }
    }
    
    // Inform the current actions module that we just parsed this declarator.
    // FIXME: pass asm & attributes.
    LastDeclInGroup = Actions.ParseDeclarator(CurScope, D, Init.Val,
                                              LastDeclInGroup);
    
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
    return LastDeclInGroup;
  } else {
    Diag(Tok, diag::err_parse_error);
    // Skip to end of block or statement
    SkipUntil(tok::r_brace, true);
    if (Tok.getKind() == tok::semi)
      ConsumeToken();
    return 0;
  }
}

/// ParseSpecifierQualifierList
///        specifier-qualifier-list:
///          type-specifier specifier-qualifier-list[opt]
///          type-qualifier specifier-qualifier-list[opt]
/// [GNU]    attributes     specifier-qualifier-list[opt]
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
  
  // Issue diagnostic and remove storage class if present.
  if (Specs & DeclSpec::PQ_StorageClassSpecifier) {
    if (DS.getStorageClassSpecLoc().isValid())
      Diag(DS.getStorageClassSpecLoc(),diag::err_typename_invalid_storageclass);
    else
      Diag(DS.getThreadSpecLoc(), diag::err_typename_invalid_storageclass);
    DS.ClearStorageClassSpecs();
  }
  
  // Issue diagnostic and remove function specfier if present.
  if (Specs & DeclSpec::PQ_FunctionSpecifier) {
    Diag(DS.getInlineSpecLoc(), diag::err_typename_invalid_functionspec);
    DS.ClearFunctionSpecs();
  }
}

/// ParseDeclarationSpecifiers
///       declaration-specifiers: [C99 6.7]
///         storage-class-specifier declaration-specifiers[opt]
///         type-specifier declaration-specifiers[opt]
///         type-qualifier declaration-specifiers[opt]
/// [C99]   function-specifier declaration-specifiers[opt]
/// [GNU]   attributes declaration-specifiers[opt]
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
///         typedef-name
/// [C++]   'bool'
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
  while (1) {
    int isInvalid = false;
    const char *PrevSpec = 0;
    SourceLocation Loc = Tok.getLocation();
    
    switch (Tok.getKind()) {
      // typedef-name
    case tok::identifier:
      // This identifier can only be a typedef name if we haven't already seen
      // a type-specifier.  Without this check we misparse:
      //  typedef int X; struct Y { short X; };  as 'short int'.
      if (!DS.hasTypeSpecifier()) {
        // It has to be available as a typedef too!
        if (void *TypeRep = Actions.isTypeName(*Tok.getIdentifierInfo(),
                                               CurScope)) {
          isInvalid = DS.SetTypeSpecType(DeclSpec::TST_typedef, Loc, PrevSpec,
                                         TypeRep);
          break;
        }
      }
      // FALL THROUGH.
    default:
      // If this is not a declaration specifier token, we're done reading decl
      // specifiers.  First verify that DeclSpec's are consistent.
      DS.Finish(Diags, getLang());
      return;
    
    // GNU attributes support.
    case tok::kw___attribute:
      ParseAttributes();
      continue;
      
    // storage-class-specifier
    case tok::kw_typedef:
      isInvalid = DS.SetStorageClassSpec(DeclSpec::SCS_typedef, Loc, PrevSpec);
      break;
    case tok::kw_extern:
      if (DS.isThreadSpecified())
        Diag(Tok, diag::ext_thread_before, "extern");
      isInvalid = DS.SetStorageClassSpec(DeclSpec::SCS_extern, Loc, PrevSpec);
      break;
    case tok::kw_static:
      if (DS.isThreadSpecified())
        Diag(Tok, diag::ext_thread_before, "static");
      isInvalid = DS.SetStorageClassSpec(DeclSpec::SCS_static, Loc, PrevSpec);
      break;
    case tok::kw_auto:
      isInvalid = DS.SetStorageClassSpec(DeclSpec::SCS_auto, Loc, PrevSpec);
      break;
    case tok::kw_register:
      isInvalid = DS.SetStorageClassSpec(DeclSpec::SCS_register, Loc, PrevSpec);
      break;
    case tok::kw___thread:
      isInvalid = DS.SetStorageClassSpecThread(Loc, PrevSpec)*2;
      break;
      
    // type-specifiers
    case tok::kw_short:
      isInvalid = DS.SetTypeSpecWidth(DeclSpec::TSW_short, Loc, PrevSpec);
      break;
    case tok::kw_long:
      if (DS.getTypeSpecWidth() != DeclSpec::TSW_long)
        isInvalid = DS.SetTypeSpecWidth(DeclSpec::TSW_long, Loc, PrevSpec);
      else
        isInvalid = DS.SetTypeSpecWidth(DeclSpec::TSW_longlong, Loc, PrevSpec);
      break;
    case tok::kw_signed:
      isInvalid = DS.SetTypeSpecSign(DeclSpec::TSS_signed, Loc, PrevSpec);
      break;
    case tok::kw_unsigned:
      isInvalid = DS.SetTypeSpecSign(DeclSpec::TSS_unsigned, Loc, PrevSpec);
      break;
    case tok::kw__Complex:
      isInvalid = DS.SetTypeSpecComplex(DeclSpec::TSC_complex, Loc, PrevSpec);
      break;
    case tok::kw__Imaginary:
      isInvalid = DS.SetTypeSpecComplex(DeclSpec::TSC_imaginary, Loc, PrevSpec);
      break;
    case tok::kw_void:
      isInvalid = DS.SetTypeSpecType(DeclSpec::TST_void, Loc, PrevSpec);
      break;
    case tok::kw_char:
      isInvalid = DS.SetTypeSpecType(DeclSpec::TST_char, Loc, PrevSpec);
      break;
    case tok::kw_int:
      isInvalid = DS.SetTypeSpecType(DeclSpec::TST_int, Loc, PrevSpec);
      break;
    case tok::kw_float:
      isInvalid = DS.SetTypeSpecType(DeclSpec::TST_float, Loc, PrevSpec);
      break;
    case tok::kw_double:
      isInvalid = DS.SetTypeSpecType(DeclSpec::TST_double, Loc, PrevSpec);
      break;
    case tok::kw_bool:          // [C++ 2.11p1]
    case tok::kw__Bool:
      isInvalid = DS.SetTypeSpecType(DeclSpec::TST_bool, Loc, PrevSpec);
      break;
    case tok::kw__Decimal32:
      isInvalid = DS.SetTypeSpecType(DeclSpec::TST_decimal32, Loc, PrevSpec);
      break;
    case tok::kw__Decimal64:
      isInvalid = DS.SetTypeSpecType(DeclSpec::TST_decimal64, Loc, PrevSpec);
      break;
    case tok::kw__Decimal128:
      isInvalid = DS.SetTypeSpecType(DeclSpec::TST_decimal128, Loc, PrevSpec);
      break;
      
    case tok::kw_struct:
    case tok::kw_union:
      ParseStructUnionSpecifier(DS);
      continue;
    case tok::kw_enum:
      ParseEnumSpecifier(DS);
      continue;
    
    // type-qualifier
    case tok::kw_const:
      isInvalid = DS.SetTypeQual(DeclSpec::TQ_const   , Loc, PrevSpec,
                                 getLang())*2;
      break;
    case tok::kw_volatile:
      isInvalid = DS.SetTypeQual(DeclSpec::TQ_volatile, Loc, PrevSpec,
                                 getLang())*2;
      break;
    case tok::kw_restrict:
      isInvalid = DS.SetTypeQual(DeclSpec::TQ_restrict, Loc, PrevSpec,
                                 getLang())*2;
      break;
      
    // function-specifier
    case tok::kw_inline:
      isInvalid = DS.SetFunctionSpecInline(Loc, PrevSpec);
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

/// ParseTag - Parse "struct-or-union-or-class-or-enum identifier[opt]", where
/// the first token has already been read and has been turned into an instance
/// of DeclSpec::TST (TagType).  This returns true if there is an error parsing,
/// otherwise it returns false and fills in Decl.
bool Parser::ParseTag(DeclTy *&Decl, unsigned TagType, SourceLocation StartLoc){
  // If attributes exist after tag, parse them.
  if (Tok.getKind() == tok::kw___attribute)
    ParseAttributes();
  
  // Must have either 'struct name' or 'struct {...}'.
  if (Tok.getKind() != tok::identifier &&
      Tok.getKind() != tok::l_brace) {
    Diag(Tok, diag::err_expected_ident_lbrace);
    // TODO: better error recovery here.
    return true;
  }
  
  // If an identifier is present, consume and remember it.
  IdentifierInfo *Name = 0;
  SourceLocation NameLoc;
  if (Tok.getKind() == tok::identifier) {
    Name = Tok.getIdentifierInfo();
    NameLoc = ConsumeToken();
  }
  
  // There are three options here.  If we have 'struct foo;', then this is a
  // forward declaration.  If we have 'struct foo {...' then this is a
  // definition. Otherwise we have something like 'struct foo xyz', a reference.
  //
  // This is needed to handle stuff like this right (C99 6.7.2.3p11):
  // struct foo {..};  void bar() { struct foo; }    <- new foo in bar.
  // struct foo {..};  void bar() { struct foo x; }  <- use of old foo.
  //
  Action::TagKind TK;
  if (Tok.getKind() == tok::l_brace)
    TK = Action::TK_Definition;
  else if (Tok.getKind() == tok::semi)
    TK = Action::TK_Declaration;
  else
    TK = Action::TK_Reference;
  Decl = Actions.ParseTag(CurScope, TagType, TK, StartLoc, Name, NameLoc);
  return false;
}


/// ParseStructUnionSpecifier
///       struct-or-union-specifier: [C99 6.7.2.1]
///         struct-or-union identifier[opt] '{' struct-contents '}'
///         struct-or-union identifier
/// [GNU]   struct-or-union attributes[opt] identifier[opt] '{' struct-contents
///                                                         '}' attributes[opt]
/// [GNU]   struct-or-union attributes[opt] identifier
///       struct-or-union:
///         'struct'
///         'union'
///
void Parser::ParseStructUnionSpecifier(DeclSpec &DS) {
  assert((Tok.getKind() == tok::kw_struct ||
          Tok.getKind() == tok::kw_union) && "Not a struct/union specifier");
  DeclSpec::TST TagType =
    Tok.getKind() == tok::kw_union ? DeclSpec::TST_union : DeclSpec::TST_struct;
  SourceLocation StartLoc = ConsumeToken();

  // Parse the tag portion of this.
  DeclTy *TagDecl;
  if (ParseTag(TagDecl, TagType, StartLoc))
    return;
  
  // If there is a body, parse it and inform the actions module.
  if (Tok.getKind() == tok::l_brace)
    ParseStructUnionBody(StartLoc, TagType, TagDecl);

  const char *PrevSpec = 0;
  if (DS.SetTypeSpecType(TagType, StartLoc, PrevSpec, TagDecl))
    Diag(StartLoc, diag::err_invalid_decl_spec_combination, PrevSpec);
}


/// ParseStructUnionBody
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
/// [GNU]   struct-declarator-list ',' attributes[opt] struct-declarator
///       struct-declarator:
///         declarator
/// [GNU]   declarator attributes[opt]
///         declarator[opt] ':' constant-expression
/// [GNU]   declarator[opt] ':' constant-expression attributes[opt]
///
void Parser::ParseStructUnionBody(SourceLocation RecordLoc,
                                  unsigned TagType, DeclTy *TagDecl) {
  SourceLocation LBraceLoc = ConsumeBrace();
  
  // Empty structs are an extension in C (C99 6.7.2.1p7), but are allowed in
  // C++.
  if (Tok.getKind() == tok::r_brace)
    Diag(Tok, diag::ext_empty_struct_union_enum, 
         DeclSpec::getSpecifierName((DeclSpec::TST)TagType));

  SmallVector<DeclTy*, 32> FieldDecls;
  
  // While we still have something to read, read the declarations in the struct.
  while (Tok.getKind() != tok::r_brace && 
         Tok.getKind() != tok::eof) {
    // Each iteration of this loop reads one struct-declaration.
    
    // Parse the common specifier-qualifiers-list piece.
    DeclSpec DS;
    SourceLocation SpecQualLoc = Tok.getLocation();
    ParseSpecifierQualifierList(DS);
    // TODO: Does specifier-qualifier list correctly check that *something* is
    // specified?
    
    // If there are no declarators, issue a warning.
    if (Tok.getKind() == tok::semi) {
      Diag(SpecQualLoc, diag::w_no_declarators);
      ConsumeToken();
      continue;
    }

    // Read struct-declarators until we find the semicolon.
    Declarator DeclaratorInfo(DS, Declarator::MemberContext);

    while (1) {
      /// struct-declarator: declarator
      /// struct-declarator: declarator[opt] ':' constant-expression
      if (Tok.getKind() != tok::colon)
        ParseDeclarator(DeclaratorInfo);
      
      ExprTy *BitfieldSize = 0;
      if (Tok.getKind() == tok::colon) {
        ConsumeToken();
        ExprResult Res = ParseConstantExpression();
        if (Res.isInvalid) {
          SkipUntil(tok::semi, true, true);
        } else {
          BitfieldSize = Res.Val;
        }
      }
      
      // If attributes exist after the declarator, parse them.
      if (Tok.getKind() == tok::kw___attribute)
        ParseAttributes();
      
      // Install the declarator into the current TagDecl.
      DeclTy *Field = Actions.ParseField(CurScope, TagDecl, SpecQualLoc,
                                         DeclaratorInfo, BitfieldSize);
      FieldDecls.push_back(Field);
      
      // If we don't have a comma, it is either the end of the list (a ';')
      // or an error, bail out.
      if (Tok.getKind() != tok::comma)
        break;
      
      // Consume the comma.
      ConsumeToken();
      
      // Parse the next declarator.
      DeclaratorInfo.clear();
      
      // Attributes are only allowed on the second declarator.
      if (Tok.getKind() == tok::kw___attribute)
        ParseAttributes();
    }
    
    if (Tok.getKind() == tok::semi) {
      ConsumeToken();
    } else {
      Diag(Tok, diag::err_expected_semi_decl_list);
      // Skip to end of block or statement
      SkipUntil(tok::r_brace, true, true);
    }
  }
  
  MatchRHSPunctuation(tok::r_brace, LBraceLoc);
  
  Actions.ParseRecordBody(RecordLoc, TagDecl, &FieldDecls[0],FieldDecls.size());
  
  // If attributes exist after struct contents, parse them.
  if (Tok.getKind() == tok::kw___attribute)
    ParseAttributes();
}


/// ParseEnumSpecifier
///       enum-specifier: [C99 6.7.2.2]
///         'enum' identifier[opt] '{' enumerator-list '}'
/// [C99]   'enum' identifier[opt] '{' enumerator-list ',' '}'
/// [GNU]   'enum' attributes[opt] identifier[opt] '{' enumerator-list ',' [opt]
///                                                 '}' attributes[opt]
///         'enum' identifier
/// [GNU]   'enum' attributes[opt] identifier
void Parser::ParseEnumSpecifier(DeclSpec &DS) {
  assert(Tok.getKind() == tok::kw_enum && "Not an enum specifier");
  SourceLocation StartLoc = ConsumeToken();
  
  // Parse the tag portion of this.
  DeclTy *TagDecl;
  if (ParseTag(TagDecl, DeclSpec::TST_enum, StartLoc))
    return;
  
  if (Tok.getKind() == tok::l_brace)
    ParseEnumBody(StartLoc, TagDecl);
  
  // TODO: semantic analysis on the declspec for enums.
  const char *PrevSpec = 0;
  if (DS.SetTypeSpecType(DeclSpec::TST_enum, StartLoc, PrevSpec, TagDecl))
    Diag(StartLoc, diag::err_invalid_decl_spec_combination, PrevSpec);
}

/// ParseEnumBody - Parse a {} enclosed enumerator-list.
///       enumerator-list:
///         enumerator
///         enumerator-list ',' enumerator
///       enumerator:
///         enumeration-constant
///         enumeration-constant '=' constant-expression
///       enumeration-constant:
///         identifier
///
void Parser::ParseEnumBody(SourceLocation StartLoc, DeclTy *EnumDecl) {
  SourceLocation LBraceLoc = ConsumeBrace();
  
  if (Tok.getKind() == tok::r_brace)
    Diag(Tok, diag::ext_empty_struct_union_enum, "enum");
  
  SmallVector<DeclTy*, 32> EnumConstantDecls;

  // Parse the enumerator-list.
  while (Tok.getKind() == tok::identifier) {
    IdentifierInfo *Ident = Tok.getIdentifierInfo();
    SourceLocation IdentLoc = ConsumeToken();
    
    SourceLocation EqualLoc;
    ExprTy *AssignedVal = 0;
    if (Tok.getKind() == tok::equal) {
      EqualLoc = ConsumeToken();
      ExprResult Res = ParseConstantExpression();
      if (Res.isInvalid)
        SkipUntil(tok::comma, tok::r_brace, true, true);
      else
        AssignedVal = Res.Val;
    }
    
    // Install the enumerator constant into EnumDecl.
    DeclTy *ConstDecl = Actions.ParseEnumConstant(CurScope, EnumDecl,
                                                  IdentLoc, Ident,
                                                  EqualLoc, AssignedVal);
    EnumConstantDecls.push_back(ConstDecl);
    
    if (Tok.getKind() != tok::comma)
      break;
    SourceLocation CommaLoc = ConsumeToken();
    
    if (Tok.getKind() != tok::identifier && !getLang().C99)
      Diag(CommaLoc, diag::ext_c99_enumerator_list_comma);
  }
  
  // Eat the }.
  MatchRHSPunctuation(tok::r_brace, LBraceLoc);

  Actions.ParseEnumBody(StartLoc, EnumDecl, &EnumConstantDecls[0],
                        EnumConstantDecls.size());
  
  // If attributes exist after the identifier list, parse them.
  if (Tok.getKind() == tok::kw___attribute)
    ParseAttributes();
}

/// isTypeSpecifierQualifier - Return true if the current token could be the
/// start of a specifier-qualifier-list.
bool Parser::isTypeSpecifierQualifier() const {
  switch (Tok.getKind()) {
  default: return false;
    // GNU attributes support.
  case tok::kw___attribute:
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
    return Actions.isTypeName(*Tok.getIdentifierInfo(), CurScope) != 0;
    
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
    return Actions.isTypeName(*Tok.getIdentifierInfo(), CurScope) != 0;
    // TODO: Attributes.
  }
}


/// ParseTypeQualifierListOpt
///       type-qualifier-list: [C99 6.7.5]
///         type-qualifier
/// [GNU]   attributes
///         type-qualifier-list type-qualifier
/// [GNU]   type-qualifier-list attributes
///
void Parser::ParseTypeQualifierListOpt(DeclSpec &DS) {
  while (1) {
    int isInvalid = false;
    const char *PrevSpec = 0;
    SourceLocation Loc = Tok.getLocation();

    switch (Tok.getKind()) {
    default:
      // If this is not a type-qualifier token, we're done reading type
      // qualifiers.  First verify that DeclSpec's are consistent.
      DS.Finish(Diags, getLang());
      return;
    case tok::kw_const:
      isInvalid = DS.SetTypeQual(DeclSpec::TQ_const   , Loc, PrevSpec,
                                 getLang())*2;
      break;
    case tok::kw_volatile:
      isInvalid = DS.SetTypeQual(DeclSpec::TQ_volatile, Loc, PrevSpec,
                                 getLang())*2;
      break;
    case tok::kw_restrict:
      isInvalid = DS.SetTypeQual(DeclSpec::TQ_restrict, Loc, PrevSpec,
                                 getLang())*2;
      break;
      
    case tok::kw___attribute:
      ParseAttributes();
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
/// [C++]   reference direct-declarator [C++ 8p4, dcl.decl]
///
///       pointer: [C99 6.7.5]
///         '*' type-qualifier-list[opt]
///         '*' type-qualifier-list[opt] pointer
///
///       reference: [C++ 8p4, dcl.decl]
/// [C++]   '&' declarator
///
void Parser::ParseDeclaratorInternal(Declarator &D) {
  tok::TokenKind Kind = Tok.getKind();

  // Not a pointer or C++ reference.
  if (Kind != tok::star && !(Kind == tok::amp && getLang().CPlusPlus))
    return ParseDirectDeclarator(D);
  
  // Otherwise, '*' -> pointer or '&' -> reference.
  SourceLocation Loc = ConsumeToken();  // Eat the * or &.

  if (Kind == tok::star) {
    // Is a pointer
    DeclSpec DS;
    ParseTypeQualifierListOpt(DS);
  
    // Recursively parse the declarator.
    ParseDeclaratorInternal(D);

    // Remember that we parsed a pointer type, and remember the type-quals.
    D.AddTypeInfo(DeclaratorChunk::getPointer(DS.getTypeQualifiers(), Loc));
  } else {
    // Is a reference

    // Recursively parse the declarator.
    ParseDeclaratorInternal(D);

    // Remember that we parsed a reference type. It doesn't have type-quals.
    D.AddTypeInfo(DeclaratorChunk::getReference(Loc));
  }
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
    // direct-declarator: '(' attributes declarator ')'
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
/// [GNU]   declaration-specifiers declarator attributes
///         declaration-specifiers abstract-declarator[opt] 
/// [GNU]   declaration-specifiers abstract-declarator[opt] attributes
///
///       identifier-list: [C99 6.7.5]
///         identifier
///         identifier-list ',' identifier
///
void Parser::ParseParenDeclarator(Declarator &D) {
  SourceLocation StartLoc = ConsumeParen();
  
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
      // This handles C99 6.7.5.3p11: in "typedef int X; void foo(X)", X is
      // considered to be a type, not a K&R identifier-list.
      isGrouping = false;
    } else {
      // Otherwise, this is a grouping paren, e.g. 'int (*X)' or 'int(X)'.
      isGrouping = true;
    }
    
    // If this is a grouping paren, handle:
    // direct-declarator: '(' declarator ')'
    // direct-declarator: '(' attributes declarator ')'
    if (isGrouping) {
      if (Tok.getKind() == tok::kw___attribute)
        ParseAttributes();
      
      ParseDeclaratorInternal(D);
      // Match the ')'.
      MatchRHSPunctuation(tok::r_paren, StartLoc);
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
  bool IsVariadic;
  bool HasPrototype;
  bool ErrorEmitted = false;

  // Build up an array of information about the parsed arguments.
  SmallVector<DeclaratorChunk::ParamInfo, 16> ParamInfo;
  SmallSet<const IdentifierInfo*, 16> ParamsSoFar;
  
  if (Tok.getKind() == tok::r_paren) {
    // int() -> no prototype, no '...'.
    IsVariadic   = false;
    HasPrototype = false;
  } else if (Tok.getKind() == tok::identifier &&
             // K&R identifier lists can't have typedefs as identifiers, per
             // C99 6.7.5.3p11.
             !Actions.isTypeName(*Tok.getIdentifierInfo(), CurScope)) {
    // Identifier list.  Note that '(' identifier-list ')' is only allowed for
    // normal declarators, not for abstract-declarators.
    assert(D.isPastIdentifier() && "Identifier (if present) must be passed!");
    
    // If there was no identifier specified, either we are in an
    // abstract-declarator, or we are in a parameter declarator which was found
    // to be abstract.  In abstract-declarators, identifier lists are not valid,
    // diagnose this.
    if (!D.getIdentifier())
      Diag(Tok, diag::ext_ident_list_in_param);

    // Remember this identifier in ParamInfo.
    ParamInfo.push_back(DeclaratorChunk::ParamInfo(Tok.getIdentifierInfo(),
                                                   Tok.getLocation(), 0));

    ConsumeToken();
    while (Tok.getKind() == tok::comma) {
      // Eat the comma.
      ConsumeToken();
      
      if (Tok.getKind() != tok::identifier) {
        Diag(Tok, diag::err_expected_ident);
        ErrorEmitted = true;
        break;
      }
      
      IdentifierInfo *ParmII = Tok.getIdentifierInfo();
      
      // Verify that the argument identifier has not already been mentioned.
      if (!ParamsSoFar.insert(ParmII)) {
        Diag(Tok.getLocation(), diag::err_param_redefinition,ParmII->getName());
        ParmII = 0;
      }
          
      // Remember this identifier in ParamInfo.
      if (ParmII)
        ParamInfo.push_back(DeclaratorChunk::ParamInfo(ParmII,
                                                       Tok.getLocation(), 0));
      
      // Eat the identifier.
      ConsumeToken();
    }
    
    // K&R 'prototype'.
    IsVariadic = false;
    HasPrototype = false;
  } else {
    // Finally, a normal, non-empty parameter type list.
    
    // Enter function-declaration scope, limiting any declarators for struct
    // tags to the function prototype scope.
    // FIXME: is this needed?
    EnterScope(0);
    
    IsVariadic = false;
    while (1) {
      if (Tok.getKind() == tok::ellipsis) {
        IsVariadic = true;

        // Check to see if this is "void(...)" which is not allowed.
        if (ParamInfo.empty()) {
          // Otherwise, parse parameter type list.  If it starts with an
          // ellipsis,  diagnose the malformed function.
          Diag(Tok, diag::err_ellipsis_first_arg);
          IsVariadic = false;       // Treat this like 'void()'.
        }

        // Consume the ellipsis.
        ConsumeToken();
        break;
      }
      
      // Parse the declaration-specifiers.
      DeclSpec DS;
      ParseDeclarationSpecifiers(DS);

      // Parse the declarator.  This is "PrototypeContext", because we must
      // accept either 'declarator' or 'abstract-declarator' here.
      Declarator ParmDecl(DS, Declarator::PrototypeContext);
      ParseDeclarator(ParmDecl);

      // Parse GNU attributes, if present.
      if (Tok.getKind() == tok::kw___attribute)
        ParseAttributes();
      
      // Verify C99 6.7.5.3p2: The only SCS allowed is 'register'.
      // NOTE: we could trivially allow 'int foo(auto int X)' if we wanted.
      if (DS.getStorageClassSpec() != DeclSpec::SCS_unspecified &&
          DS.getStorageClassSpec() != DeclSpec::SCS_register) {
        Diag(DS.getStorageClassSpecLoc(),
             diag::err_invalid_storage_class_in_func_decl);
        DS.ClearStorageClassSpecs();
      }
      if (DS.isThreadSpecified()) {
        Diag(DS.getThreadSpecLoc(),
             diag::err_invalid_storage_class_in_func_decl);
        DS.ClearStorageClassSpecs();
      }
      
      // Inform the actions module about the parameter declarator, so it gets
      // added to the current scope.
      Action::TypeResult ParamTy =
        Actions.ParseParamDeclaratorType(CurScope, ParmDecl);
        
      // Remember this parsed parameter in ParamInfo.
      IdentifierInfo *ParmII = ParmDecl.getIdentifier();
      
      // Verify that the argument identifier has not already been mentioned.
      if (ParmII && !ParamsSoFar.insert(ParmII)) {
        Diag(ParmDecl.getIdentifierLoc(), diag::err_param_redefinition,
             ParmII->getName());
        ParmII = 0;
      }
        
      ParamInfo.push_back(DeclaratorChunk::ParamInfo(ParmII,
                                                    ParmDecl.getIdentifierLoc(),
                                                     ParamTy.Val));
      
      // If the next token is a comma, consume it and keep reading arguments.
      if (Tok.getKind() != tok::comma) break;
      
      // Consume the comma.
      ConsumeToken();
    }
    
    HasPrototype = true;
    
    // Leave prototype scope.
    ExitScope();
  }
  
  // Remember that we parsed a function type, and remember the attributes.
  if (!ErrorEmitted)
    D.AddTypeInfo(DeclaratorChunk::getFunction(HasPrototype, IsVariadic,
                                               &ParamInfo[0], ParamInfo.size(),
                                               StartLoc));
  
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
  SourceLocation StartLoc = ConsumeBracket();
  
  // If valid, this location is the position where we read the 'static' keyword.
  SourceLocation StaticLoc;
  if (Tok.getKind() == tok::kw_static)
    StaticLoc = ConsumeToken();
  
  // If there is a type-qualifier-list, read it now.
  DeclSpec DS;
  ParseTypeQualifierListOpt(DS);
  
  // If we haven't already read 'static', check to see if there is one after the
  // type-qualifier-list.
  if (!StaticLoc.isValid() && Tok.getKind() == tok::kw_static)
    StaticLoc = ConsumeToken();
  
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
  
  MatchRHSPunctuation(tok::r_square, StartLoc);
    
  // If C99 isn't enabled, emit an ext-warn if the arg list wasn't empty and if
  // it was not a constant expression.
  if (!getLang().C99) {
    // TODO: check C90 array constant exprness.
    if (isStar || StaticLoc.isValid() ||
        0/*TODO: NumElts is not a C90 constantexpr */)
      Diag(StartLoc, diag::ext_c99_array_usage);
  }
  
  // Remember that we parsed a pointer type, and remember the type-quals.
  D.AddTypeInfo(DeclaratorChunk::getArray(DS.getTypeQualifiers(),
                                          StaticLoc.isValid(), isStar,
                                          NumElements.Val, StartLoc));
}

