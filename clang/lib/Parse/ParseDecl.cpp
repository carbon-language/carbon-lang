//===--- ParseDecl.cpp - Declaration Parsing ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the Declaration portions of the Parser interfaces.
//
//===----------------------------------------------------------------------===//

#include "clang/Parse/Parser.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Parse/Scope.h"
#include "ExtensionRAIIObject.h"
#include "llvm/ADT/SmallSet.h"
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
  
  return Actions.ActOnTypeName(CurScope, DeclaratorInfo).Val;
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
///          attrib-name
///          attrib-name '(' identifier ')'
///          attrib-name '(' identifier ',' nonempty-expr-list ')'
///          attrib-name '(' argument-expression-list [C99 6.5.2] ')'
///
/// [GNU]  attrib-name:
///          identifier
///          typespec
///          typequal
///          storageclass
///          
/// FIXME: The GCC grammar/code for this construct implies we need two
/// token lookahead. Comment from gcc: "If they start with an identifier 
/// which is followed by a comma or close parenthesis, then the arguments 
/// start with that identifier; otherwise they are an expression list."
///
/// At the moment, I am not doing 2 token lookahead. I am also unaware of
/// any attributes that don't work (based on my limited testing). Most
/// attributes are very simple in practice. Until we find a bug, I don't see
/// a pressing need to implement the 2 token lookahead.

AttributeList *Parser::ParseAttributes() {
  assert(Tok.is(tok::kw___attribute) && "Not an attribute list!");
  
  AttributeList *CurrAttr = 0;
  
  while (Tok.is(tok::kw___attribute)) {
    ConsumeToken();
    if (ExpectAndConsume(tok::l_paren, diag::err_expected_lparen_after,
                         "attribute")) {
      SkipUntil(tok::r_paren, true); // skip until ) or ;
      return CurrAttr;
    }
    if (ExpectAndConsume(tok::l_paren, diag::err_expected_lparen_after, "(")) {
      SkipUntil(tok::r_paren, true); // skip until ) or ;
      return CurrAttr;
    }
    // Parse the attribute-list. e.g. __attribute__(( weak, alias("__f") ))
    while (Tok.is(tok::identifier) || isDeclarationSpecifier() ||
           Tok.is(tok::comma)) {
           
      if (Tok.is(tok::comma)) { 
        // allows for empty/non-empty attributes. ((__vector_size__(16),,,,))
        ConsumeToken();
        continue;
      }
      // we have an identifier or declaration specifier (const, int, etc.)
      IdentifierInfo *AttrName = Tok.getIdentifierInfo();
      SourceLocation AttrNameLoc = ConsumeToken();
      
      // check if we have a "paramterized" attribute
      if (Tok.is(tok::l_paren)) {
        ConsumeParen(); // ignore the left paren loc for now
        
        if (Tok.is(tok::identifier)) {
          IdentifierInfo *ParmName = Tok.getIdentifierInfo();
          SourceLocation ParmLoc = ConsumeToken();
          
          if (Tok.is(tok::r_paren)) { 
            // __attribute__(( mode(byte) ))
            ConsumeParen(); // ignore the right paren loc for now
            CurrAttr = new AttributeList(AttrName, AttrNameLoc, 
                                         ParmName, ParmLoc, 0, 0, CurrAttr);
          } else if (Tok.is(tok::comma)) {
            ConsumeToken();
            // __attribute__(( format(printf, 1, 2) ))
            llvm::SmallVector<ExprTy*, 8> ArgExprs;
            bool ArgExprsOk = true;
            
            // now parse the non-empty comma separated list of expressions
            while (1) {
              ExprResult ArgExpr = ParseAssignmentExpression();
              if (ArgExpr.isInvalid) {
                ArgExprsOk = false;
                SkipUntil(tok::r_paren);
                break;
              } else {
                ArgExprs.push_back(ArgExpr.Val);
              }
              if (Tok.isNot(tok::comma))
                break;
              ConsumeToken(); // Eat the comma, move to the next argument
            }
            if (ArgExprsOk && Tok.is(tok::r_paren)) {
              ConsumeParen(); // ignore the right paren loc for now
              CurrAttr = new AttributeList(AttrName, AttrNameLoc, ParmName, 
                           ParmLoc, &ArgExprs[0], ArgExprs.size(), CurrAttr);
            }
          }
        } else { // not an identifier
          // parse a possibly empty comma separated list of expressions
          if (Tok.is(tok::r_paren)) { 
            // __attribute__(( nonnull() ))
            ConsumeParen(); // ignore the right paren loc for now
            CurrAttr = new AttributeList(AttrName, AttrNameLoc, 
                                         0, SourceLocation(), 0, 0, CurrAttr);
          } else { 
            // __attribute__(( aligned(16) ))
            llvm::SmallVector<ExprTy*, 8> ArgExprs;
            bool ArgExprsOk = true;
            
            // now parse the list of expressions
            while (1) {
              ExprResult ArgExpr = ParseAssignmentExpression();
              if (ArgExpr.isInvalid) {
                ArgExprsOk = false;
                SkipUntil(tok::r_paren);
                break;
              } else {
                ArgExprs.push_back(ArgExpr.Val);
              }
              if (Tok.isNot(tok::comma))
                break;
              ConsumeToken(); // Eat the comma, move to the next argument
            }
            // Match the ')'.
            if (ArgExprsOk && Tok.is(tok::r_paren)) {
              ConsumeParen(); // ignore the right paren loc for now
              CurrAttr = new AttributeList(AttrName, AttrNameLoc, 0, 
                           SourceLocation(), &ArgExprs[0], ArgExprs.size(), 
                           CurrAttr);
            }
          }
        }
      } else {
        CurrAttr = new AttributeList(AttrName, AttrNameLoc, 
                                     0, SourceLocation(), 0, 0, CurrAttr);
      }
    }
    if (ExpectAndConsume(tok::r_paren, diag::err_expected_rparen))
      SkipUntil(tok::r_paren, false); 
    if (ExpectAndConsume(tok::r_paren, diag::err_expected_rparen))
      SkipUntil(tok::r_paren, false);
  }
  return CurrAttr;
}

/// ParseDeclaration - Parse a full 'declaration', which consists of
/// declaration-specifiers, some number of declarators, and a semicolon.
/// 'Context' should be a Declarator::TheContext value.
///
///       declaration: [C99 6.7]
///         block-declaration ->
///           simple-declaration
///           others                   [FIXME]
/// [C++]   namespace-definition
///         others... [FIXME]
///
Parser::DeclTy *Parser::ParseDeclaration(unsigned Context) {
  switch (Tok.getKind()) {
  case tok::kw_namespace:
    return ParseNamespace(Context);
  default:
    return ParseSimpleDeclaration(Context);
  }
}

///       simple-declaration: [C99 6.7: declaration] [C++ 7p1: dcl.dcl]
///         declaration-specifiers init-declarator-list[opt] ';'
///[C90/C++]init-declarator-list ';'                             [TODO]
/// [OMP]   threadprivate-directive                              [TODO]
Parser::DeclTy *Parser::ParseSimpleDeclaration(unsigned Context) {
  // Parse the common declaration-specifiers piece.
  DeclSpec DS;
  ParseDeclarationSpecifiers(DS);
  
  // C99 6.7.2.3p6: Handle "struct-or-union identifier;", "enum { X };"
  // declaration-specifiers init-declarator-list[opt] ';'
  if (Tok.is(tok::semi)) {
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
///       init-declarator-list: [C99 6.7]
///         init-declarator
///         init-declarator-list ',' init-declarator
///       init-declarator: [C99 6.7]
///         declarator
///         declarator '=' initializer
/// [GNU]   declarator simple-asm-expr[opt] attributes[opt]
/// [GNU]   declarator simple-asm-expr[opt] attributes[opt] '=' initializer
/// [C++]   declarator initializer[opt]
///
/// [C++] initializer:
/// [C++]   '=' initializer-clause
/// [C++]   '(' expression-list ')'
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
    if (Tok.is(tok::kw_asm)) {
      ExprResult AsmLabel = ParseSimpleAsm();
      if (AsmLabel.isInvalid) {
        SkipUntil(tok::semi);
        return 0;
      }
      
      D.setAsmLabel(AsmLabel.Val);
    }
    
    // If attributes are present, parse them.
    if (Tok.is(tok::kw___attribute))
      D.AddAttributes(ParseAttributes());

    // Inform the current actions module that we just parsed this declarator.
    LastDeclInGroup = Actions.ActOnDeclarator(CurScope, D, LastDeclInGroup);
        
    // Parse declarator '=' initializer.
    if (Tok.is(tok::equal)) {
      ConsumeToken();
      ExprResult Init = ParseInitializer();
      if (Init.isInvalid) {
        SkipUntil(tok::semi);
        return 0;
      }
      Actions.AddInitializerToDecl(LastDeclInGroup, Init.Val);
    } else if (Tok.is(tok::l_paren)) {
      // Parse C++ direct initializer: '(' expression-list ')'
      SourceLocation LParenLoc = ConsumeParen();
      ExprListTy Exprs;
      CommaLocsTy CommaLocs;

      bool InvalidExpr = false;
      if (ParseExpressionList(Exprs, CommaLocs)) {
        SkipUntil(tok::r_paren);
        InvalidExpr = true;
      }
      // Match the ')'.
      SourceLocation RParenLoc = MatchRHSPunctuation(tok::r_paren, LParenLoc);

      if (!InvalidExpr) {
        assert(!Exprs.empty() && Exprs.size()-1 == CommaLocs.size() &&
               "Unexpected number of commas!");
        Actions.AddCXXDirectInitializerToDecl(LastDeclInGroup, LParenLoc,
                                              &Exprs[0], Exprs.size(),
                                              &CommaLocs[0], RParenLoc);
      }
    } else {
      Actions.ActOnUninitializedDecl(LastDeclInGroup);
    }
    
    // If we don't have a comma, it is either the end of the list (a ';') or an
    // error, bail out.
    if (Tok.isNot(tok::comma))
      break;
    
    // Consume the comma.
    ConsumeToken();
    
    // Parse the next declarator.
    D.clear();
    
    // Accept attributes in an init-declarator.  In the first declarator in a
    // declaration, these would be part of the declspec.  In subsequent
    // declarators, they become part of the declarator itself, so that they
    // don't apply to declarators after *this* one.  Examples:
    //    short __attribute__((common)) var;    -> declspec
    //    short var __attribute__((common));    -> declarator
    //    short x, __attribute__((common)) var;    -> declarator
    if (Tok.is(tok::kw___attribute))
      D.AddAttributes(ParseAttributes());
    
    ParseDeclarator(D);
  }
  
  if (Tok.is(tok::semi)) {
    ConsumeToken();
    return Actions.FinalizeDeclaratorGroup(CurScope, LastDeclInGroup);
  }
  // If this is an ObjC2 for-each loop, this is a successful declarator
  // parse.  The syntax for these looks like:
  // 'for' '(' declaration 'in' expr ')' statement
  if (D.getContext()  == Declarator::ForContext && isTokIdentifier_in()) {
    return Actions.FinalizeDeclaratorGroup(CurScope, LastDeclInGroup);
  }
  Diag(Tok, diag::err_parse_error);
  // Skip to end of block or statement
  SkipUntil(tok::r_brace, true, true);
  if (Tok.is(tok::semi))
    ConsumeToken();
  return 0;
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
  ParseDeclarationSpecifiers(DS);
  
  // Validate declspec for type-name.
  unsigned Specs = DS.getParsedSpecifiers();
  if (Specs == DeclSpec::PQ_None && !DS.getNumProtocolQualifiers())
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
    if (DS.isInlineSpecified())
      Diag(DS.getInlineSpecLoc(), diag::err_typename_invalid_functionspec);
    if (DS.isVirtualSpecified())
      Diag(DS.getVirtualSpecLoc(), diag::err_typename_invalid_functionspec);
    if (DS.isExplicitSpecified())
      Diag(DS.getExplicitSpecLoc(), diag::err_typename_invalid_functionspec);
    DS.ClearFunctionSpecs();
  }
}

/// ParseDeclarationSpecifiers
///       declaration-specifiers: [C99 6.7]
///         storage-class-specifier declaration-specifiers[opt]
///         type-specifier declaration-specifiers[opt]
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
///       function-specifier: [C99 6.7.4]
/// [C99]   'inline'
/// [C++]   'virtual'
/// [C++]   'explicit'
///
void Parser::ParseDeclarationSpecifiers(DeclSpec &DS) {
  DS.SetRangeStart(Tok.getLocation());
  while (1) {
    int isInvalid = false;
    const char *PrevSpec = 0;
    SourceLocation Loc = Tok.getLocation();

    // Only annotate C++ scope. Allow class-name as an identifier in case
    // it's a constructor.
    TryAnnotateScopeToken();
    
    switch (Tok.getKind()) {
    default: 
      // Try to parse a type-specifier; if we found one, continue.
      if (MaybeParseTypeSpecifier(DS, isInvalid, PrevSpec))
        continue;

    DoneWithDeclSpec:
      // If this is not a declaration specifier token, we're done reading decl
      // specifiers.  First verify that DeclSpec's are consistent.
      DS.Finish(Diags, PP.getSourceManager(), getLang());
      return;

    case tok::annot_cxxscope: {
      if (DS.hasTypeSpecifier())
        goto DoneWithDeclSpec;

      // We are looking for a qualified typename.
      if (NextToken().isNot(tok::identifier))
        goto DoneWithDeclSpec;

      CXXScopeSpec SS;
      SS.setScopeRep(Tok.getAnnotationValue());
      SS.setRange(Tok.getAnnotationRange());

      // If the next token is the name of the class type that the C++ scope
      // denotes, followed by a '(', then this is a constructor declaration.
      // We're done with the decl-specifiers.
      if (Actions.isCurrentClassName(*NextToken().getIdentifierInfo(),
                                     CurScope, &SS) &&
          GetLookAheadToken(2).is(tok::l_paren))
        goto DoneWithDeclSpec;

      TypeTy *TypeRep = Actions.isTypeName(*NextToken().getIdentifierInfo(),
                                           CurScope, &SS);
      if (TypeRep == 0)
        goto DoneWithDeclSpec;

      ConsumeToken(); // The C++ scope.

      isInvalid = DS.SetTypeSpecType(DeclSpec::TST_typedef, Loc, PrevSpec,
                                     TypeRep);
      if (isInvalid)
        break;
      
      DS.SetRangeEnd(Tok.getLocation());
      ConsumeToken(); // The typename.

      continue;
    }

      // typedef-name
    case tok::identifier: {
      // This identifier can only be a typedef name if we haven't already seen
      // a type-specifier.  Without this check we misparse:
      //  typedef int X; struct Y { short X; };  as 'short int'.
      if (DS.hasTypeSpecifier())
        goto DoneWithDeclSpec;
      
      // It has to be available as a typedef too!
      TypeTy *TypeRep = Actions.isTypeName(*Tok.getIdentifierInfo(), CurScope);
      if (TypeRep == 0)
        goto DoneWithDeclSpec;
      
      // C++: If the identifier is actually the name of the class type
      // being defined and the next token is a '(', then this is a
      // constructor declaration. We're done with the decl-specifiers
      // and will treat this token as an identifier.
      if (getLang().CPlusPlus && 
          CurScope->isCXXClassScope() &&
          Actions.isCurrentClassName(*Tok.getIdentifierInfo(), CurScope) && 
          NextToken().getKind() == tok::l_paren)
        goto DoneWithDeclSpec;

      isInvalid = DS.SetTypeSpecType(DeclSpec::TST_typedef, Loc, PrevSpec,
                                     TypeRep);
      if (isInvalid)
        break;
      
      DS.SetRangeEnd(Tok.getLocation());
      ConsumeToken(); // The identifier

      // Objective-C supports syntax of the form 'id<proto1,proto2>' where 'id'
      // is a specific typedef and 'itf<proto1,proto2>' where 'itf' is an
      // Objective-C interface.  If we don't have Objective-C or a '<', this is
      // just a normal reference to a typedef name.
      if (!Tok.is(tok::less) || !getLang().ObjC1)
        continue;
      
      SourceLocation EndProtoLoc;
      llvm::SmallVector<DeclTy *, 8> ProtocolDecl;
      ParseObjCProtocolReferences(ProtocolDecl, false, EndProtoLoc);
      DS.setProtocolQualifiers(&ProtocolDecl[0], ProtocolDecl.size());
      
      DS.SetRangeEnd(EndProtoLoc);

      // Need to support trailing type qualifiers (e.g. "id<p> const").
      // If a type specifier follows, it will be diagnosed elsewhere.
      continue;
    }
    // GNU attributes support.
    case tok::kw___attribute:
      DS.AddAttributes(ParseAttributes());
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
    case tok::kw___private_extern__:
      isInvalid = DS.SetStorageClassSpec(DeclSpec::SCS_private_extern, Loc,
                                         PrevSpec);
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
      
      continue;
          
    // function-specifier
    case tok::kw_inline:
      isInvalid = DS.SetFunctionSpecInline(Loc, PrevSpec);
      break;

    case tok::kw_virtual:
      isInvalid = DS.SetFunctionSpecVirtual(Loc, PrevSpec);
      break;

    case tok::kw_explicit:
      isInvalid = DS.SetFunctionSpecExplicit(Loc, PrevSpec);
      break;
      
    case tok::less:
      // GCC ObjC supports types like "<SomeProtocol>" as a synonym for
      // "id<SomeProtocol>".  This is hopelessly old fashioned and dangerous,
      // but we support it.
      if (DS.hasTypeSpecifier() || !getLang().ObjC1)
        goto DoneWithDeclSpec;
        
      {
        SourceLocation EndProtoLoc;
        llvm::SmallVector<DeclTy *, 8> ProtocolDecl;
        ParseObjCProtocolReferences(ProtocolDecl, false, EndProtoLoc);
        DS.setProtocolQualifiers(&ProtocolDecl[0], ProtocolDecl.size());
        DS.SetRangeEnd(EndProtoLoc);

        Diag(Loc, diag::warn_objc_protocol_qualifier_missing_id,
             SourceRange(Loc, EndProtoLoc));
        // Need to support trailing type qualifiers (e.g. "id<p> const").
        // If a type specifier follows, it will be diagnosed elsewhere.
        continue;
      }
    }
    // If the specifier combination wasn't legal, issue a diagnostic.
    if (isInvalid) {
      assert(PrevSpec && "Method did not return previous specifier!");
      if (isInvalid == 1)  // Error.
        Diag(Tok, diag::err_invalid_decl_spec_combination, PrevSpec);
      else                 // extwarn.
        Diag(Tok, diag::ext_duplicate_declspec, PrevSpec);
    }
    DS.SetRangeEnd(Tok.getLocation());
    ConsumeToken();
  }
}
/// MaybeParseTypeSpecifier - Try to parse a single type-specifier. We
/// primarily follow the C++ grammar with additions for C99 and GNU,
/// which together subsume the C grammar. Note that the C++
/// type-specifier also includes the C type-qualifier (for const,
/// volatile, and C99 restrict). Returns true if a type-specifier was
/// found (and parsed), false otherwise.
///
///       type-specifier: [C++ 7.1.5]
///         simple-type-specifier
///         class-specifier
///         enum-specifier
///         elaborated-type-specifier  [TODO]
///         cv-qualifier
///
///       cv-qualifier: [C++ 7.1.5.1]
///         'const'
///         'volatile'
/// [C99]   'restrict'
///
///       simple-type-specifier: [ C++ 7.1.5.2]
///         '::'[opt] nested-name-specifier[opt] type-name [TODO]
///         '::'[opt] nested-name-specifier 'template' template-id [TODO]
///         'char'
///         'wchar_t'
///         'bool'
///         'short'
///         'int'
///         'long'
///         'signed'
///         'unsigned'
///         'float'
///         'double'
///         'void'
/// [C99]   '_Bool'
/// [C99]   '_Complex'
/// [C99]   '_Imaginary'  // Removed in TC2?
/// [GNU]   '_Decimal32'
/// [GNU]   '_Decimal64'
/// [GNU]   '_Decimal128'
/// [GNU]   typeof-specifier
/// [OBJC]  class-name objc-protocol-refs[opt]    [TODO]
/// [OBJC]  typedef-name objc-protocol-refs[opt]  [TODO]
bool Parser::MaybeParseTypeSpecifier(DeclSpec &DS, int& isInvalid,
                                     const char *&PrevSpec) {
  // Annotate typenames and C++ scope specifiers.
  TryAnnotateTypeOrScopeToken();

  SourceLocation Loc = Tok.getLocation();

  switch (Tok.getKind()) {
  // simple-type-specifier:
  case tok::annot_qualtypename: {
    isInvalid = DS.SetTypeSpecType(DeclSpec::TST_typedef, Loc, PrevSpec,
                                   Tok.getAnnotationValue());
    DS.SetRangeEnd(Tok.getAnnotationEndLoc());
    ConsumeToken(); // The typename
    
    // Objective-C supports syntax of the form 'id<proto1,proto2>' where 'id'
    // is a specific typedef and 'itf<proto1,proto2>' where 'itf' is an
    // Objective-C interface.  If we don't have Objective-C or a '<', this is
    // just a normal reference to a typedef name.
    if (!Tok.is(tok::less) || !getLang().ObjC1)
      return true;
    
    SourceLocation EndProtoLoc;
    llvm::SmallVector<DeclTy *, 8> ProtocolDecl;
    ParseObjCProtocolReferences(ProtocolDecl, false, EndProtoLoc);
    DS.setProtocolQualifiers(&ProtocolDecl[0], ProtocolDecl.size());
    
    DS.SetRangeEnd(EndProtoLoc);
    return true;
  }

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
  case tok::kw_wchar_t:
    isInvalid = DS.SetTypeSpecType(DeclSpec::TST_wchar, Loc, PrevSpec);
    break;
  case tok::kw_bool:
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

  // class-specifier:
  case tok::kw_class:
  case tok::kw_struct:
  case tok::kw_union:
    ParseClassSpecifier(DS);
    return true;

  // enum-specifier:
  case tok::kw_enum:
    ParseEnumSpecifier(DS);
    return true;

  // cv-qualifier:
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

  // GNU typeof support.
  case tok::kw_typeof:
    ParseTypeofSpecifier(DS);
    return true;

  default:
    // Not a type-specifier; do nothing.
    return false;
  }

  // If the specifier combination wasn't legal, issue a diagnostic.
  if (isInvalid) {
    assert(PrevSpec && "Method did not return previous specifier!");
    if (isInvalid == 1)  // Error.
      Diag(Tok, diag::err_invalid_decl_spec_combination, PrevSpec);
    else                 // extwarn.
      Diag(Tok, diag::ext_duplicate_declspec, PrevSpec);
  }
  DS.SetRangeEnd(Tok.getLocation());
  ConsumeToken(); // whatever we parsed above.
  return true;
}

/// ParseStructDeclaration - Parse a struct declaration without the terminating
/// semicolon.
///
///       struct-declaration:
///         specifier-qualifier-list struct-declarator-list
/// [GNU]   __extension__ struct-declaration
/// [GNU]   specifier-qualifier-list
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
void Parser::
ParseStructDeclaration(DeclSpec &DS,
                       llvm::SmallVectorImpl<FieldDeclarator> &Fields) {
  if (Tok.is(tok::kw___extension__)) {
    // __extension__ silences extension warnings in the subexpression.
    ExtensionRAIIObject O(Diags);  // Use RAII to do this.
    ConsumeToken();
    return ParseStructDeclaration(DS, Fields);
  }
  
  // Parse the common specifier-qualifiers-list piece.
  SourceLocation DSStart = Tok.getLocation();
  ParseSpecifierQualifierList(DS);
  
  // If there are no declarators, issue a warning.
  if (Tok.is(tok::semi)) {
    Diag(DSStart, diag::w_no_declarators);
    return;
  }

  // Read struct-declarators until we find the semicolon.
  Fields.push_back(FieldDeclarator(DS));
  while (1) {
    FieldDeclarator &DeclaratorInfo = Fields.back();
    
    /// struct-declarator: declarator
    /// struct-declarator: declarator[opt] ':' constant-expression
    if (Tok.isNot(tok::colon))
      ParseDeclarator(DeclaratorInfo.D);
    
    if (Tok.is(tok::colon)) {
      ConsumeToken();
      ExprResult Res = ParseConstantExpression();
      if (Res.isInvalid)
        SkipUntil(tok::semi, true, true);
      else
        DeclaratorInfo.BitfieldSize = Res.Val;
    }
    
    // If attributes exist after the declarator, parse them.
    if (Tok.is(tok::kw___attribute))
      DeclaratorInfo.D.AddAttributes(ParseAttributes());
    
    // If we don't have a comma, it is either the end of the list (a ';')
    // or an error, bail out.
    if (Tok.isNot(tok::comma))
      return;
    
    // Consume the comma.
    ConsumeToken();
    
    // Parse the next declarator.
    Fields.push_back(FieldDeclarator(DS));
    
    // Attributes are only allowed on the second declarator.
    if (Tok.is(tok::kw___attribute))
      Fields.back().D.AddAttributes(ParseAttributes());
  }
}

/// ParseStructUnionBody
///       struct-contents:
///         struct-declaration-list
/// [EXT]   empty
/// [GNU]   "struct-declaration-list" without terminatoring ';'
///       struct-declaration-list:
///         struct-declaration
///         struct-declaration-list struct-declaration
/// [OBC]   '@' 'defs' '(' class-name ')'
///
void Parser::ParseStructUnionBody(SourceLocation RecordLoc,
                                  unsigned TagType, DeclTy *TagDecl) {
  SourceLocation LBraceLoc = ConsumeBrace();
  
  // Empty structs are an extension in C (C99 6.7.2.1p7), but are allowed in
  // C++.
  if (Tok.is(tok::r_brace) && !getLang().CPlusPlus)
    Diag(Tok, diag::ext_empty_struct_union_enum, 
         DeclSpec::getSpecifierName((DeclSpec::TST)TagType));

  llvm::SmallVector<DeclTy*, 32> FieldDecls;
  llvm::SmallVector<FieldDeclarator, 8> FieldDeclarators;

  // While we still have something to read, read the declarations in the struct.
  while (Tok.isNot(tok::r_brace) && Tok.isNot(tok::eof)) {
    // Each iteration of this loop reads one struct-declaration.
    
    // Check for extraneous top-level semicolon.
    if (Tok.is(tok::semi)) {
      Diag(Tok, diag::ext_extra_struct_semi);
      ConsumeToken();
      continue;
    }

    // Parse all the comma separated declarators.
    DeclSpec DS;
    FieldDeclarators.clear();
    if (!Tok.is(tok::at)) {
      ParseStructDeclaration(DS, FieldDeclarators);
      
      // Convert them all to fields.
      for (unsigned i = 0, e = FieldDeclarators.size(); i != e; ++i) {
        FieldDeclarator &FD = FieldDeclarators[i];
        // Install the declarator into the current TagDecl.
        DeclTy *Field = Actions.ActOnField(CurScope,
                                           DS.getSourceRange().getBegin(),
                                           FD.D, FD.BitfieldSize);
        FieldDecls.push_back(Field);
      }
    } else { // Handle @defs
      ConsumeToken();
      if (!Tok.isObjCAtKeyword(tok::objc_defs)) {
        Diag(Tok, diag::err_unexpected_at);
        SkipUntil(tok::semi, true, true);
        continue;
      }
      ConsumeToken();
      ExpectAndConsume(tok::l_paren, diag::err_expected_lparen);
      if (!Tok.is(tok::identifier)) {
        Diag(Tok, diag::err_expected_ident);
        SkipUntil(tok::semi, true, true);
        continue;
      }
      llvm::SmallVector<DeclTy*, 16> Fields;
      Actions.ActOnDefs(CurScope, Tok.getLocation(), Tok.getIdentifierInfo(),
          Fields);
      FieldDecls.insert(FieldDecls.end(), Fields.begin(), Fields.end());
      ConsumeToken();
      ExpectAndConsume(tok::r_paren, diag::err_expected_rparen);
    } 

    if (Tok.is(tok::semi)) {
      ConsumeToken();
    } else if (Tok.is(tok::r_brace)) {
      Diag(Tok.getLocation(), diag::ext_expected_semi_decl_list);
      break;
    } else {
      Diag(Tok, diag::err_expected_semi_decl_list);
      // Skip to end of block or statement
      SkipUntil(tok::r_brace, true, true);
    }
  }
  
  SourceLocation RBraceLoc = MatchRHSPunctuation(tok::r_brace, LBraceLoc);
  
  AttributeList *AttrList = 0;
  // If attributes exist after struct contents, parse them.
  if (Tok.is(tok::kw___attribute))
    AttrList = ParseAttributes();

  Actions.ActOnFields(CurScope,
                      RecordLoc,TagDecl,&FieldDecls[0],FieldDecls.size(),
                      LBraceLoc, RBraceLoc,
                      AttrList);  
}


/// ParseEnumSpecifier
///       enum-specifier: [C99 6.7.2.2]
///         'enum' identifier[opt] '{' enumerator-list '}'
///[C99/C++]'enum' identifier[opt] '{' enumerator-list ',' '}'
/// [GNU]   'enum' attributes[opt] identifier[opt] '{' enumerator-list ',' [opt]
///                                                 '}' attributes[opt]
///         'enum' identifier
/// [GNU]   'enum' attributes[opt] identifier
///
/// [C++] elaborated-type-specifier:
/// [C++]   'enum' '::'[opt] nested-name-specifier[opt] identifier
///
void Parser::ParseEnumSpecifier(DeclSpec &DS) {
  assert(Tok.is(tok::kw_enum) && "Not an enum specifier");
  SourceLocation StartLoc = ConsumeToken();
  
  // Parse the tag portion of this.

  AttributeList *Attr = 0;
  // If attributes exist after tag, parse them.
  if (Tok.is(tok::kw___attribute))
    Attr = ParseAttributes();

  CXXScopeSpec SS;
  if (isTokenCXXScopeSpecifier()) {
    ParseCXXScopeSpecifier(SS);
    if (Tok.isNot(tok::identifier)) {
      Diag(Tok, diag::err_expected_ident);
      if (Tok.isNot(tok::l_brace)) {
        // Has no name and is not a definition.
        // Skip the rest of this declarator, up until the comma or semicolon.
        SkipUntil(tok::comma, true);
        return;
      }
    }
  }
  
  // Must have either 'enum name' or 'enum {...}'.
  if (Tok.isNot(tok::identifier) && Tok.isNot(tok::l_brace)) {
    Diag(Tok, diag::err_expected_ident_lbrace);
    
    // Skip the rest of this declarator, up until the comma or semicolon.
    SkipUntil(tok::comma, true);
    return;
  }
  
  // If an identifier is present, consume and remember it.
  IdentifierInfo *Name = 0;
  SourceLocation NameLoc;
  if (Tok.is(tok::identifier)) {
    Name = Tok.getIdentifierInfo();
    NameLoc = ConsumeToken();
  }
  
  // There are three options here.  If we have 'enum foo;', then this is a
  // forward declaration.  If we have 'enum foo {...' then this is a
  // definition. Otherwise we have something like 'enum foo xyz', a reference.
  //
  // This is needed to handle stuff like this right (C99 6.7.2.3p11):
  // enum foo {..};  void bar() { enum foo; }    <- new foo in bar.
  // enum foo {..};  void bar() { enum foo x; }  <- use of old foo.
  //
  Action::TagKind TK;
  if (Tok.is(tok::l_brace))
    TK = Action::TK_Definition;
  else if (Tok.is(tok::semi))
    TK = Action::TK_Declaration;
  else
    TK = Action::TK_Reference;
  DeclTy *TagDecl = Actions.ActOnTag(CurScope, DeclSpec::TST_enum, TK, StartLoc,
                                     SS, Name, NameLoc, Attr);
  
  if (Tok.is(tok::l_brace))
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
  
  // C does not allow an empty enumerator-list, C++ does [dcl.enum].
  if (Tok.is(tok::r_brace) && !getLang().CPlusPlus)
    Diag(Tok, diag::ext_empty_struct_union_enum, "enum");
  
  llvm::SmallVector<DeclTy*, 32> EnumConstantDecls;

  DeclTy *LastEnumConstDecl = 0;
  
  // Parse the enumerator-list.
  while (Tok.is(tok::identifier)) {
    IdentifierInfo *Ident = Tok.getIdentifierInfo();
    SourceLocation IdentLoc = ConsumeToken();
    
    SourceLocation EqualLoc;
    ExprTy *AssignedVal = 0;
    if (Tok.is(tok::equal)) {
      EqualLoc = ConsumeToken();
      ExprResult Res = ParseConstantExpression();
      if (Res.isInvalid)
        SkipUntil(tok::comma, tok::r_brace, true, true);
      else
        AssignedVal = Res.Val;
    }
    
    // Install the enumerator constant into EnumDecl.
    DeclTy *EnumConstDecl = Actions.ActOnEnumConstant(CurScope, EnumDecl,
                                                      LastEnumConstDecl,
                                                      IdentLoc, Ident,
                                                      EqualLoc, AssignedVal);
    EnumConstantDecls.push_back(EnumConstDecl);
    LastEnumConstDecl = EnumConstDecl;
    
    if (Tok.isNot(tok::comma))
      break;
    SourceLocation CommaLoc = ConsumeToken();
    
    if (Tok.isNot(tok::identifier) && !getLang().C99)
      Diag(CommaLoc, diag::ext_c99_enumerator_list_comma);
  }
  
  // Eat the }.
  MatchRHSPunctuation(tok::r_brace, LBraceLoc);

  Actions.ActOnEnumBody(StartLoc, EnumDecl, &EnumConstantDecls[0],
                        EnumConstantDecls.size());
  
  DeclTy *AttrList = 0;
  // If attributes exist after the identifier list, parse them.
  if (Tok.is(tok::kw___attribute))
    AttrList = ParseAttributes(); // FIXME: where do they do?
}

/// isTypeSpecifierQualifier - Return true if the current token could be the
/// start of a type-qualifier-list.
bool Parser::isTypeQualifier() const {
  switch (Tok.getKind()) {
  default: return false;
    // type-qualifier
  case tok::kw_const:
  case tok::kw_volatile:
  case tok::kw_restrict:
    return true;
  }
}

/// isTypeSpecifierQualifier - Return true if the current token could be the
/// start of a specifier-qualifier-list.
bool Parser::isTypeSpecifierQualifier() {
  // Annotate typenames and C++ scope specifiers.
  TryAnnotateTypeOrScopeToken();

  switch (Tok.getKind()) {
  default: return false;
    // GNU attributes support.
  case tok::kw___attribute:
    // GNU typeof support.
  case tok::kw_typeof:
  
    // type-specifiers
  case tok::kw_short:
  case tok::kw_long:
  case tok::kw_signed:
  case tok::kw_unsigned:
  case tok::kw__Complex:
  case tok::kw__Imaginary:
  case tok::kw_void:
  case tok::kw_char:
  case tok::kw_wchar_t:
  case tok::kw_int:
  case tok::kw_float:
  case tok::kw_double:
  case tok::kw_bool:
  case tok::kw__Bool:
  case tok::kw__Decimal32:
  case tok::kw__Decimal64:
  case tok::kw__Decimal128:
    
    // struct-or-union-specifier (C99) or class-specifier (C++)
  case tok::kw_class:
  case tok::kw_struct:
  case tok::kw_union:
    // enum-specifier
  case tok::kw_enum:
    
    // type-qualifier
  case tok::kw_const:
  case tok::kw_volatile:
  case tok::kw_restrict:

    // typedef-name
  case tok::annot_qualtypename:
    return true;
      
    // GNU ObjC bizarre protocol extension: <proto1,proto2> with implicit 'id'.
  case tok::less:
    return getLang().ObjC1;
  }
}

/// isDeclarationSpecifier() - Return true if the current token is part of a
/// declaration specifier.
bool Parser::isDeclarationSpecifier() {
  // Annotate typenames and C++ scope specifiers.
  TryAnnotateTypeOrScopeToken();

  switch (Tok.getKind()) {
  default: return false;
    // storage-class-specifier
  case tok::kw_typedef:
  case tok::kw_extern:
  case tok::kw___private_extern__:
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
  case tok::kw_wchar_t:
  case tok::kw_int:
  case tok::kw_float:
  case tok::kw_double:
  case tok::kw_bool:
  case tok::kw__Bool:
  case tok::kw__Decimal32:
  case tok::kw__Decimal64:
  case tok::kw__Decimal128:
  
    // struct-or-union-specifier (C99) or class-specifier (C++)
  case tok::kw_class:
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
  case tok::kw_virtual:
  case tok::kw_explicit:

    // typedef-name
  case tok::annot_qualtypename:

    // GNU typeof support.
  case tok::kw_typeof:
    
    // GNU attributes.
  case tok::kw___attribute:
    return true;
  
    // GNU ObjC bizarre protocol extension: <proto1,proto2> with implicit 'id'.
  case tok::less:
    return getLang().ObjC1;
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
      DS.Finish(Diags, PP.getSourceManager(), getLang());
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
      DS.AddAttributes(ParseAttributes());
      continue; // do *not* consume the next token!
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
}

/// ParseDeclaratorInternal - Parse a C or C++ declarator. If
/// PtrOperator is true, then this routine won't parse the final
/// direct-declarator; therefore, it effectively parses the C++
/// ptr-operator production.
///
///       declarator: [C99 6.7.5]
///         pointer[opt] direct-declarator
/// [C++]   '&' declarator [C++ 8p4, dcl.decl]
/// [GNU]   '&' restrict[opt] attributes[opt] declarator
///
///       pointer: [C99 6.7.5]
///         '*' type-qualifier-list[opt]
///         '*' type-qualifier-list[opt] pointer
///
///       ptr-operator:
///         '*' cv-qualifier-seq[opt]
///         '&'
/// [GNU]   '&' restrict[opt] attributes[opt]
///         '::'[opt] nested-name-specifier '*' cv-qualifier-seq[opt] [TODO]
void Parser::ParseDeclaratorInternal(Declarator &D, bool PtrOperator) {
  tok::TokenKind Kind = Tok.getKind();

  // Not a pointer, C++ reference, or block.
  if (Kind != tok::star && (Kind != tok::amp || !getLang().CPlusPlus) &&
      (Kind != tok::caret || !getLang().Blocks)) {
    if (!PtrOperator)
      ParseDirectDeclarator(D);
    return;
  }
  
  // Otherwise, '*' -> pointer, '^' -> block, '&' -> reference.
  SourceLocation Loc = ConsumeToken();  // Eat the * or &.

  if (Kind == tok::star || (Kind == tok::caret && getLang().Blocks)) {
    // Is a pointer.
    DeclSpec DS;
    
    ParseTypeQualifierListOpt(DS);
  
    // Recursively parse the declarator.
    ParseDeclaratorInternal(D, PtrOperator);
    if (Kind == tok::star)
      // Remember that we parsed a pointer type, and remember the type-quals.
      D.AddTypeInfo(DeclaratorChunk::getPointer(DS.getTypeQualifiers(), Loc,
                                                DS.TakeAttributes()));
    else
      // Remember that we parsed a Block type, and remember the type-quals.
      D.AddTypeInfo(DeclaratorChunk::getBlockPointer(DS.getTypeQualifiers(), 
                                                     Loc));	
  } else {
    // Is a reference
    DeclSpec DS;

    // C++ 8.3.2p1: cv-qualified references are ill-formed except when the
    // cv-qualifiers are introduced through the use of a typedef or of a
    // template type argument, in which case the cv-qualifiers are ignored.
    //
    // [GNU] Retricted references are allowed.
    // [GNU] Attributes on references are allowed.
    ParseTypeQualifierListOpt(DS);

    if (DS.getTypeQualifiers() != DeclSpec::TQ_unspecified) {
      if (DS.getTypeQualifiers() & DeclSpec::TQ_const)
        Diag(DS.getConstSpecLoc(),
             diag::err_invalid_reference_qualifier_application,
             "const");
      if (DS.getTypeQualifiers() & DeclSpec::TQ_volatile)
        Diag(DS.getVolatileSpecLoc(),
             diag::err_invalid_reference_qualifier_application,
             "volatile");
    }

    // Recursively parse the declarator.
    ParseDeclaratorInternal(D, PtrOperator);

    if (D.getNumTypeObjects() > 0) {
      // C++ [dcl.ref]p4: There shall be no references to references.
      DeclaratorChunk& InnerChunk = D.getTypeObject(D.getNumTypeObjects() - 1);
      if (InnerChunk.Kind == DeclaratorChunk::Reference) {
        Diag(InnerChunk.Loc, diag::err_illegal_decl_reference_to_reference,
             D.getIdentifier() ? D.getIdentifier()->getName() : "type name");

        // Once we've complained about the reference-to-referwnce, we
        // can go ahead and build the (technically ill-formed)
        // declarator: reference collapsing will take care of it.
      }
    }

    // Remember that we parsed a reference type. It doesn't have type-quals.
    D.AddTypeInfo(DeclaratorChunk::getReference(DS.getTypeQualifiers(), Loc,
                                                DS.TakeAttributes()));
  }
}

/// ParseDirectDeclarator
///       direct-declarator: [C99 6.7.5]
/// [C99]   identifier
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
/// [C++]   direct-declarator '(' parameter-declaration-clause ')'
///                    cv-qualifier-seq[opt] exception-specification[opt]
/// [C++]   declarator-id
///
///       declarator-id: [C++ 8]
///         id-expression
///         '::'[opt] nested-name-specifier[opt] type-name
///
///       id-expression: [C++ 5.1]
///         unqualified-id
///         qualified-id            [TODO]
///
///       unqualified-id: [C++ 5.1]
///         identifier 
///         operator-function-id
///         conversion-function-id  [TODO]
///          '~' class-name         
///         template-id             [TODO]
///
void Parser::ParseDirectDeclarator(Declarator &D) {
  CXXScopeSpec &SS = D.getCXXScopeSpec();
  DeclaratorScopeObj DeclScopeObj(*this, SS);

  if (D.mayHaveIdentifier() && isTokenCXXScopeSpecifier()) {
    ParseCXXScopeSpecifier(SS);
    // Change the declaration context for name lookup, until this function is
    // exited (and the declarator has been parsed).
    DeclScopeObj.EnterDeclaratorScope();
  }

  // Parse the first direct-declarator seen.
  if (Tok.is(tok::identifier) && D.mayHaveIdentifier()) {
    assert(Tok.getIdentifierInfo() && "Not an identifier?");
    // Determine whether this identifier is a C++ constructor name or
    // a normal identifier.
    if (getLang().CPlusPlus && 
        Actions.isCurrentClassName(*Tok.getIdentifierInfo(), CurScope))
      D.SetConstructor(Actions.isTypeName(*Tok.getIdentifierInfo(), CurScope),
                       Tok.getIdentifierInfo(), Tok.getLocation());
    else
      D.SetIdentifier(Tok.getIdentifierInfo(), Tok.getLocation());
    ConsumeToken();
  } else if (getLang().CPlusPlus &&
             Tok.is(tok::tilde) && D.mayHaveIdentifier()) {
    // This should be a C++ destructor.
    SourceLocation TildeLoc = ConsumeToken();
    if (Tok.is(tok::identifier)) {
      // Use the next identifier and "~" to form a name for the
      // destructor. This is useful both for diagnostics and for
      // correctness of the parser, since we use presence/absence of the
      // identifier to determine what we parsed.
      // FIXME: We could end up with a template-id here, once we parse
      // templates, and will have to do something different to form the
      // name of the destructor.
      IdentifierInfo *II = Tok.getIdentifierInfo();
      II = &PP.getIdentifierTable().get(std::string("~") + II->getName());

      if (TypeTy *Type = ParseClassName())
        D.SetDestructor(Type, II, TildeLoc);
      else
        D.SetIdentifier(0, TildeLoc);
    } else {
      Diag(Tok, diag::err_expected_class_name);
      D.SetIdentifier(0, TildeLoc);
    }
  } else if (Tok.is(tok::kw_operator)) {
    SourceLocation OperatorLoc = Tok.getLocation();

    // First try the name of an overloaded operator
    if (IdentifierInfo *II = MaybeParseOperatorFunctionId()) {
      D.SetIdentifier(II, OperatorLoc);
    } else {
      // This must be a conversion function (C++ [class.conv.fct]).
      if (TypeTy *ConvType = ParseConversionFunctionId()) {
        IdentifierInfo *II 
          = &PP.getIdentifierTable().get(std::string("operator ") + 
                                         Actions.getTypeAsString(ConvType));
        D.SetConversionFunction(ConvType, II, OperatorLoc);
      }
    }
  } else if (Tok.is(tok::l_paren) && SS.isEmpty()) {
    // direct-declarator: '(' declarator ')'
    // direct-declarator: '(' attributes declarator ')'
    // Example: 'char (*X)'   or 'int (*XX)(void)'
    ParseParenDeclarator(D);
  } else if (D.mayOmitIdentifier() && SS.isEmpty()) {
    // This could be something simple like "int" (in which case the declarator
    // portion is empty), if an abstract-declarator is allowed.
    D.SetIdentifier(0, Tok.getLocation());
  } else {
    if (getLang().CPlusPlus)
      Diag(Tok, diag::err_expected_unqualified_id);
    else
      Diag(Tok, diag::err_expected_ident_lparen); // Expected identifier or '('.
    D.SetIdentifier(0, Tok.getLocation());
  }
  
  assert(D.isPastIdentifier() &&
         "Haven't past the location of the identifier yet?");
  
  while (1) {
    if (Tok.is(tok::l_paren)) {
      // The paren may be part of a C++ direct initializer, eg. "int x(1);".
      // In such a case, check if we actually have a function declarator; if it
      // is not, the declarator has been fully parsed.
      if (getLang().CPlusPlus && D.mayBeFollowedByCXXDirectInit()) {
        // When not in file scope, warn for ambiguous function declarators, just
        // in case the author intended it as a variable definition.
        bool warnIfAmbiguous = D.getContext() != Declarator::FileContext;
        if (!isCXXFunctionDeclarator(warnIfAmbiguous))
          break;
      }
      ParseFunctionDeclarator(ConsumeParen(), D);
    } else if (Tok.is(tok::l_square)) {
      ParseBracketDeclarator(D);
    } else {
      break;
    }
  }
}

/// ParseParenDeclarator - We parsed the declarator D up to a paren.  This is
/// only called before the identifier, so these are most likely just grouping
/// parens for precedence.  If we find that these are actually function 
/// parameter parens in an abstract-declarator, we call ParseFunctionDeclarator.
///
///       direct-declarator:
///         '(' declarator ')'
/// [GNU]   '(' attributes declarator ')'
///         direct-declarator '(' parameter-type-list ')'
///         direct-declarator '(' identifier-list[opt] ')'
/// [GNU]   direct-declarator '(' parameter-forward-declarations
///                    parameter-type-list[opt] ')'
///
void Parser::ParseParenDeclarator(Declarator &D) {
  SourceLocation StartLoc = ConsumeParen();
  assert(!D.isPastIdentifier() && "Should be called before passing identifier");
  
  // Eat any attributes before we look at whether this is a grouping or function
  // declarator paren.  If this is a grouping paren, the attribute applies to
  // the type being built up, for example:
  //     int (__attribute__(()) *x)(long y)
  // If this ends up not being a grouping paren, the attribute applies to the
  // first argument, for example:
  //     int (__attribute__(()) int x)
  // In either case, we need to eat any attributes to be able to determine what
  // sort of paren this is.
  //
  AttributeList *AttrList = 0;
  bool RequiresArg = false;
  if (Tok.is(tok::kw___attribute)) {
    AttrList = ParseAttributes();
    
    // We require that the argument list (if this is a non-grouping paren) be
    // present even if the attribute list was empty.
    RequiresArg = true;
  }
  
  // If we haven't past the identifier yet (or where the identifier would be
  // stored, if this is an abstract declarator), then this is probably just
  // grouping parens. However, if this could be an abstract-declarator, then
  // this could also be the start of function arguments (consider 'void()').
  bool isGrouping;
  
  if (!D.mayOmitIdentifier()) {
    // If this can't be an abstract-declarator, this *must* be a grouping
    // paren, because we haven't seen the identifier yet.
    isGrouping = true;
  } else if (Tok.is(tok::r_paren) ||           // 'int()' is a function.
             (getLang().CPlusPlus && Tok.is(tok::ellipsis)) || // C++ int(...)
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
    bool hadGroupingParens = D.hasGroupingParens();
    D.setGroupingParens(true);
    if (AttrList)
      D.AddAttributes(AttrList);

    ParseDeclaratorInternal(D);
    // Match the ')'.
    MatchRHSPunctuation(tok::r_paren, StartLoc);

    D.setGroupingParens(hadGroupingParens);
    return;
  }
  
  // Okay, if this wasn't a grouping paren, it must be the start of a function
  // argument list.  Recognize that this declarator will never have an
  // identifier (and remember where it would have been), then call into
  // ParseFunctionDeclarator to handle of argument list.
  D.SetIdentifier(0, Tok.getLocation());

  ParseFunctionDeclarator(StartLoc, D, AttrList, RequiresArg);
}

/// ParseFunctionDeclarator - We are after the identifier and have parsed the
/// declarator D up to a paren, which indicates that we are parsing function
/// arguments.
///
/// If AttrList is non-null, then the caller parsed those arguments immediately
/// after the open paren - they should be considered to be the first argument of
/// a parameter.  If RequiresArg is true, then the first argument of the
/// function is required to be present and required to not be an identifier
/// list.
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
/// [C++]   declaration-specifiers declarator '=' assignment-expression
/// [GNU]   declaration-specifiers declarator attributes
///         declaration-specifiers abstract-declarator[opt] 
/// [C++]   declaration-specifiers abstract-declarator[opt] 
///           '=' assignment-expression
/// [GNU]   declaration-specifiers abstract-declarator[opt] attributes
///
/// For C++, after the parameter-list, it also parses "cv-qualifier-seq[opt]"
/// and "exception-specification[opt]"(TODO).
///
void Parser::ParseFunctionDeclarator(SourceLocation LParenLoc, Declarator &D,
                                     AttributeList *AttrList,
                                     bool RequiresArg) {
  // lparen is already consumed!
  assert(D.isPastIdentifier() && "Should not call before identifier!");
  
  // This parameter list may be empty.
  if (Tok.is(tok::r_paren)) {
    if (RequiresArg) {
      Diag(Tok.getLocation(), diag::err_argument_required_after_attribute);
      delete AttrList;
    }

    ConsumeParen();  // Eat the closing ')'.

    // cv-qualifier-seq[opt].
    DeclSpec DS;
    if (getLang().CPlusPlus) {
      ParseTypeQualifierListOpt(DS);
      // FIXME: Parse exception-specification[opt].
    }

    // Remember that we parsed a function type, and remember the attributes.
    // int() -> no prototype, no '...'.
    D.AddTypeInfo(DeclaratorChunk::getFunction(/*prototype*/getLang().CPlusPlus,
                                               /*variadic*/ false,
                                               /*arglist*/ 0, 0,
                                               DS.getTypeQualifiers(),
                                               LParenLoc));
    return;
  } 
  
  // Alternatively, this parameter list may be an identifier list form for a
  // K&R-style function:  void foo(a,b,c)
  if (!getLang().CPlusPlus && Tok.is(tok::identifier) &&
      // K&R identifier lists can't have typedefs as identifiers, per
      // C99 6.7.5.3p11.
      !Actions.isTypeName(*Tok.getIdentifierInfo(), CurScope)) {
    if (RequiresArg) {
      Diag(Tok.getLocation(), diag::err_argument_required_after_attribute);
      delete AttrList;
    }
    
    // Identifier list.  Note that '(' identifier-list ')' is only allowed for
    // normal declarators, not for abstract-declarators.
    return ParseFunctionDeclaratorIdentifierList(LParenLoc, D);
  }
  
  // Finally, a normal, non-empty parameter type list.
  
  // Build up an array of information about the parsed arguments.
  llvm::SmallVector<DeclaratorChunk::ParamInfo, 16> ParamInfo;

  // Enter function-declaration scope, limiting any declarators to the
  // function prototype scope, including parameter declarators.
  EnterScope(Scope::FnScope|Scope::DeclScope);
  
  bool IsVariadic = false;
  while (1) {
    if (Tok.is(tok::ellipsis)) {
      IsVariadic = true;
      
      // Check to see if this is "void(...)" which is not allowed.
      if (!getLang().CPlusPlus && ParamInfo.empty()) {
        // Otherwise, parse parameter type list.  If it starts with an
        // ellipsis,  diagnose the malformed function.
        Diag(Tok, diag::err_ellipsis_first_arg);
        IsVariadic = false;       // Treat this like 'void()'.
      }

      ConsumeToken();     // Consume the ellipsis.
      break;
    }
    
    SourceLocation DSStart = Tok.getLocation();
    
    // Parse the declaration-specifiers.
    DeclSpec DS;

    // If the caller parsed attributes for the first argument, add them now.
    if (AttrList) {
      DS.AddAttributes(AttrList);
      AttrList = 0;  // Only apply the attributes to the first parameter.
    }
    ParseDeclarationSpecifiers(DS);

    // Parse the declarator.  This is "PrototypeContext", because we must
    // accept either 'declarator' or 'abstract-declarator' here.
    Declarator ParmDecl(DS, Declarator::PrototypeContext);
    ParseDeclarator(ParmDecl);

    // Parse GNU attributes, if present.
    if (Tok.is(tok::kw___attribute))
      ParmDecl.AddAttributes(ParseAttributes());
    
    // Remember this parsed parameter in ParamInfo.
    IdentifierInfo *ParmII = ParmDecl.getIdentifier();
    
    // If no parameter was specified, verify that *something* was specified,
    // otherwise we have a missing type and identifier.
    if (DS.getParsedSpecifiers() == DeclSpec::PQ_None && 
        ParmDecl.getIdentifier() == 0 && ParmDecl.getNumTypeObjects() == 0) {
      // Completely missing, emit error.
      Diag(DSStart, diag::err_missing_param);
    } else {
      // Otherwise, we have something.  Add it and let semantic analysis try
      // to grok it and add the result to the ParamInfo we are building.
      
      // Inform the actions module about the parameter declarator, so it gets
      // added to the current scope.
      DeclTy *Param = Actions.ActOnParamDeclarator(CurScope, ParmDecl);

      // Parse the default argument, if any. We parse the default
      // arguments in all dialects; the semantic analysis in
      // ActOnParamDefaultArgument will reject the default argument in
      // C.
      if (Tok.is(tok::equal)) {
        SourceLocation EqualLoc = Tok.getLocation();
        
        // Consume the '='.
        ConsumeToken();
        
        // Parse the default argument
        ExprResult DefArgResult = ParseAssignmentExpression();
        if (DefArgResult.isInvalid) {
          SkipUntil(tok::comma, tok::r_paren, true, true);
        } else {
          // Inform the actions module about the default argument
          Actions.ActOnParamDefaultArgument(Param, EqualLoc, DefArgResult.Val);
        }
      }
      
      ParamInfo.push_back(DeclaratorChunk::ParamInfo(ParmII, 
                             ParmDecl.getIdentifierLoc(), Param));
    }

    // If the next token is a comma, consume it and keep reading arguments.
    if (Tok.isNot(tok::comma)) break;
    
    // Consume the comma.
    ConsumeToken();
  }
  
  // Leave prototype scope.
  ExitScope();
  
  // If we have the closing ')', eat it.
  MatchRHSPunctuation(tok::r_paren, LParenLoc);

  // cv-qualifier-seq[opt].
  DeclSpec DS;
  if (getLang().CPlusPlus) {
    ParseTypeQualifierListOpt(DS);
    // FIXME: Parse exception-specification[opt].
  }

  // Remember that we parsed a function type, and remember the attributes.
  D.AddTypeInfo(DeclaratorChunk::getFunction(/*proto*/true, IsVariadic,
                                             &ParamInfo[0], ParamInfo.size(),
                                             DS.getTypeQualifiers(),
                                             LParenLoc));
}

/// ParseFunctionDeclaratorIdentifierList - While parsing a function declarator
/// we found a K&R-style identifier list instead of a type argument list.  The
/// current token is known to be the first identifier in the list.
///
///       identifier-list: [C99 6.7.5]
///         identifier
///         identifier-list ',' identifier
///
void Parser::ParseFunctionDeclaratorIdentifierList(SourceLocation LParenLoc,
                                                   Declarator &D) {
  // Build up an array of information about the parsed arguments.
  llvm::SmallVector<DeclaratorChunk::ParamInfo, 16> ParamInfo;
  llvm::SmallSet<const IdentifierInfo*, 16> ParamsSoFar;
  
  // If there was no identifier specified for the declarator, either we are in
  // an abstract-declarator, or we are in a parameter declarator which was found
  // to be abstract.  In abstract-declarators, identifier lists are not valid:
  // diagnose this.
  if (!D.getIdentifier())
    Diag(Tok, diag::ext_ident_list_in_param);

  // Tok is known to be the first identifier in the list.  Remember this
  // identifier in ParamInfo.
  ParamsSoFar.insert(Tok.getIdentifierInfo());
  ParamInfo.push_back(DeclaratorChunk::ParamInfo(Tok.getIdentifierInfo(),
                                                 Tok.getLocation(), 0));
  
  ConsumeToken();  // eat the first identifier.
  
  while (Tok.is(tok::comma)) {
    // Eat the comma.
    ConsumeToken();
    
    // If this isn't an identifier, report the error and skip until ')'.
    if (Tok.isNot(tok::identifier)) {
      Diag(Tok, diag::err_expected_ident);
      SkipUntil(tok::r_paren);
      return;
    }

    IdentifierInfo *ParmII = Tok.getIdentifierInfo();

    // Reject 'typedef int y; int test(x, y)', but continue parsing.
    if (Actions.isTypeName(*ParmII, CurScope))
      Diag(Tok, diag::err_unexpected_typedef_ident, ParmII->getName());
    
    // Verify that the argument identifier has not already been mentioned.
    if (!ParamsSoFar.insert(ParmII)) {
      Diag(Tok.getLocation(), diag::err_param_redefinition, ParmII->getName());
    } else {
      // Remember this identifier in ParamInfo.
      ParamInfo.push_back(DeclaratorChunk::ParamInfo(ParmII,
                                                     Tok.getLocation(), 0));
    }
    
    // Eat the identifier.
    ConsumeToken();
  }
  
  // Remember that we parsed a function type, and remember the attributes.  This
  // function type is always a K&R style function type, which is not varargs and
  // has no prototype.
  D.AddTypeInfo(DeclaratorChunk::getFunction(/*proto*/false, /*varargs*/false,
                                             &ParamInfo[0], ParamInfo.size(),
                                             /*TypeQuals*/0, LParenLoc));
  
  // If we have the closing ')', eat it and we're done.
  MatchRHSPunctuation(tok::r_paren, LParenLoc);
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
  if (Tok.is(tok::kw_static))
    StaticLoc = ConsumeToken();
  
  // If there is a type-qualifier-list, read it now.
  DeclSpec DS;
  ParseTypeQualifierListOpt(DS);
  
  // If we haven't already read 'static', check to see if there is one after the
  // type-qualifier-list.
  if (!StaticLoc.isValid() && Tok.is(tok::kw_static))
    StaticLoc = ConsumeToken();
  
  // Handle "direct-declarator [ type-qual-list[opt] * ]".
  bool isStar = false;
  ExprResult NumElements(false);
  
  // Handle the case where we have '[*]' as the array size.  However, a leading
  // star could be the start of an expression, for example 'X[*p + 4]'.  Verify
  // the the token after the star is a ']'.  Since stars in arrays are
  // infrequent, use of lookahead is not costly here.
  if (Tok.is(tok::star) && GetLookAheadToken(1).is(tok::r_square)) {
    ConsumeToken();  // Eat the '*'.

    if (StaticLoc.isValid())
      Diag(StaticLoc, diag::err_unspecified_vla_size_with_static);
    StaticLoc = SourceLocation();  // Drop the static.
    isStar = true;
  } else if (Tok.isNot(tok::r_square)) {
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

/// [GNU]   typeof-specifier:
///           typeof ( expressions )
///           typeof ( type-name )
/// [GNU/C++] typeof unary-expression
///
void Parser::ParseTypeofSpecifier(DeclSpec &DS) {
  assert(Tok.is(tok::kw_typeof) && "Not a typeof specifier");
  const IdentifierInfo *BuiltinII = Tok.getIdentifierInfo();
  SourceLocation StartLoc = ConsumeToken();

  if (Tok.isNot(tok::l_paren)) {
    if (!getLang().CPlusPlus) {
      Diag(Tok, diag::err_expected_lparen_after, BuiltinII->getName());
      return;
    }

    ExprResult Result = ParseCastExpression(true/*isUnaryExpression*/);
    if (Result.isInvalid)
      return;

    const char *PrevSpec = 0;
    // Check for duplicate type specifiers.
    if (DS.SetTypeSpecType(DeclSpec::TST_typeofExpr, StartLoc, PrevSpec, 
                           Result.Val))
      Diag(StartLoc, diag::err_invalid_decl_spec_combination, PrevSpec);

    // FIXME: Not accurate, the range gets one token more than it should.
    DS.SetRangeEnd(Tok.getLocation());
    return;
  }

  SourceLocation LParenLoc = ConsumeParen(), RParenLoc;
  
  if (isTypeIdInParens()) {
    TypeTy *Ty = ParseTypeName();

    assert(Ty && "Parser::ParseTypeofSpecifier(): missing type");

    if (Tok.isNot(tok::r_paren)) {
      MatchRHSPunctuation(tok::r_paren, LParenLoc);
      return;
    }
    RParenLoc = ConsumeParen();
    const char *PrevSpec = 0;
    // Check for duplicate type specifiers (e.g. "int typeof(int)").
    if (DS.SetTypeSpecType(DeclSpec::TST_typeofType, StartLoc, PrevSpec, Ty))
      Diag(StartLoc, diag::err_invalid_decl_spec_combination, PrevSpec);
  } else { // we have an expression.
    ExprResult Result = ParseExpression();
    
    if (Result.isInvalid || Tok.isNot(tok::r_paren)) {
      MatchRHSPunctuation(tok::r_paren, LParenLoc);
      return;
    }
    RParenLoc = ConsumeParen();
    const char *PrevSpec = 0;
    // Check for duplicate type specifiers (e.g. "int typeof(int)").
    if (DS.SetTypeSpecType(DeclSpec::TST_typeofExpr, StartLoc, PrevSpec, 
                           Result.Val))
      Diag(StartLoc, diag::err_invalid_decl_spec_combination, PrevSpec);
  }
  DS.SetRangeEnd(RParenLoc);
}


