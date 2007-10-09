//===--- ParseObjc.cpp - Objective C Parsing ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Steve Naroff and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the Objective-C portions of the Parser interface.
//
//===----------------------------------------------------------------------===//

#include "clang/Parse/Parser.h"
#include "clang/Parse/DeclSpec.h"
#include "clang/Basic/Diagnostic.h"
#include "llvm/ADT/SmallVector.h"
using namespace clang;


/// ParseExternalDeclaration:
///       external-declaration: [C99 6.9]
/// [OBJC]  objc-class-definition
/// [OBJC]  objc-class-declaration     [TODO]
/// [OBJC]  objc-alias-declaration     [TODO]
/// [OBJC]  objc-protocol-definition   [TODO]
/// [OBJC]  objc-method-definition     [TODO]
/// [OBJC]  '@' 'end'                  [TODO]
Parser::DeclTy *Parser::ParseObjCAtDirectives() {
  SourceLocation AtLoc = ConsumeToken(); // the "@"
  
  switch (Tok.getObjCKeywordID()) {
    case tok::objc_class:
      return ParseObjCAtClassDeclaration(AtLoc);
    case tok::objc_interface:
      return ParseObjCAtInterfaceDeclaration(AtLoc);
    case tok::objc_protocol:
      return ParseObjCAtProtocolDeclaration(AtLoc);
    case tok::objc_implementation:
      return ObjcImpDecl = ParseObjCAtImplementationDeclaration(AtLoc);
    case tok::objc_end:
      return ParseObjCAtEndDeclaration(AtLoc);
    case tok::objc_compatibility_alias:
      return ParseObjCAtAliasDeclaration(AtLoc);
    case tok::objc_synthesize:
      return ParseObjCPropertySynthesize(AtLoc);
    case tok::objc_dynamic:
      return ParseObjCPropertyDynamic(AtLoc);
    default:
      Diag(AtLoc, diag::err_unexpected_at);
      SkipUntil(tok::semi);
      return 0;
  }
}

///
/// objc-class-declaration: 
///    '@' 'class' identifier-list ';'
///  
Parser::DeclTy *Parser::ParseObjCAtClassDeclaration(SourceLocation atLoc) {
  ConsumeToken(); // the identifier "class"
  llvm::SmallVector<IdentifierInfo *, 8> ClassNames;
  
  while (1) {
    if (Tok.isNot(tok::identifier)) {
      Diag(Tok, diag::err_expected_ident);
      SkipUntil(tok::semi);
      return 0;
    }
    ClassNames.push_back(Tok.getIdentifierInfo());
    ConsumeToken();
    
    if (Tok.isNot(tok::comma))
      break;
    
    ConsumeToken();
  }
  
  // Consume the ';'.
  if (ExpectAndConsume(tok::semi, diag::err_expected_semi_after, "@class"))
    return 0;
  
  return Actions.ActOnForwardClassDeclaration(CurScope, atLoc,
                                      &ClassNames[0], ClassNames.size());
}

///
///   objc-interface:
///     objc-class-interface-attributes[opt] objc-class-interface
///     objc-category-interface
///
///   objc-class-interface:
///     '@' 'interface' identifier objc-superclass[opt] 
///       objc-protocol-refs[opt]
///       objc-class-instance-variables[opt] 
///       objc-interface-decl-list
///     @end
///
///   objc-category-interface:
///     '@' 'interface' identifier '(' identifier[opt] ')' 
///       objc-protocol-refs[opt]
///       objc-interface-decl-list
///     @end
///
///   objc-superclass:
///     ':' identifier
///
///   objc-class-interface-attributes:
///     __attribute__((visibility("default")))
///     __attribute__((visibility("hidden")))
///     __attribute__((deprecated))
///     __attribute__((unavailable))
///     __attribute__((objc_exception)) - used by NSException on 64-bit
///
Parser::DeclTy *Parser::ParseObjCAtInterfaceDeclaration(
  SourceLocation atLoc, AttributeList *attrList) {
  assert(Tok.isObjCAtKeyword(tok::objc_interface) &&
         "ParseObjCAtInterfaceDeclaration(): Expected @interface");
  ConsumeToken(); // the "interface" identifier
  
  if (Tok.isNot(tok::identifier)) {
    Diag(Tok, diag::err_expected_ident); // missing class or category name.
    return 0;
  }
  // We have a class or category name - consume it.
  IdentifierInfo *nameId = Tok.getIdentifierInfo();
  SourceLocation nameLoc = ConsumeToken();
  
  if (Tok.is(tok::l_paren)) { // we have a category.
    SourceLocation lparenLoc = ConsumeParen();
    SourceLocation categoryLoc, rparenLoc;
    IdentifierInfo *categoryId = 0;
    llvm::SmallVector<IdentifierInfo *, 8> ProtocolRefs;
    
    // For ObjC2, the category name is optional (not an error).
    if (Tok.is(tok::identifier)) {
      categoryId = Tok.getIdentifierInfo();
      categoryLoc = ConsumeToken();
    } else if (!getLang().ObjC2) {
      Diag(Tok, diag::err_expected_ident); // missing category name.
      return 0;
    }
    if (Tok.isNot(tok::r_paren)) {
      Diag(Tok, diag::err_expected_rparen);
      SkipUntil(tok::r_paren, false); // don't stop at ';'
      return 0;
    }
    rparenLoc = ConsumeParen();
    // Next, we need to check for any protocol references.
    if (Tok.is(tok::less)) {
      if (ParseObjCProtocolReferences(ProtocolRefs))
        return 0;
    }
    if (attrList) // categories don't support attributes.
      Diag(Tok, diag::err_objc_no_attributes_on_category);
    
    DeclTy *CategoryType = Actions.ActOnStartCategoryInterface(CurScope, atLoc, 
                                     nameId, nameLoc, categoryId, categoryLoc,
                                     &ProtocolRefs[0], ProtocolRefs.size());
    
    ParseObjCInterfaceDeclList(CategoryType, tok::objc_not_keyword);

    // The @ sign was already consumed by ParseObjCInterfaceDeclList().
    if (Tok.isObjCAtKeyword(tok::objc_end)) {
      ConsumeToken(); // the "end" identifier
      return CategoryType;
    }
    Diag(Tok, diag::err_objc_missing_end);
    return 0;
  }
  // Parse a class interface.
  IdentifierInfo *superClassId = 0;
  SourceLocation superClassLoc;

  if (Tok.is(tok::colon)) { // a super class is specified.
    ConsumeToken();
    if (Tok.isNot(tok::identifier)) {
      Diag(Tok, diag::err_expected_ident); // missing super class name.
      return 0;
    }
    superClassId = Tok.getIdentifierInfo();
    superClassLoc = ConsumeToken();
  }
  // Next, we need to check for any protocol references.
  llvm::SmallVector<IdentifierInfo *, 8> ProtocolRefs;
  if (Tok.is(tok::less)) {
    if (ParseObjCProtocolReferences(ProtocolRefs))
      return 0;
  }
  DeclTy *ClsType = Actions.ActOnStartClassInterface(CurScope,
		      atLoc, nameId, nameLoc, 
                      superClassId, superClassLoc, &ProtocolRefs[0], 
                      ProtocolRefs.size(), attrList);
            
  if (Tok.is(tok::l_brace))
    ParseObjCClassInstanceVariables(ClsType);

  ParseObjCInterfaceDeclList(ClsType, tok::objc_interface);

  // The @ sign was already consumed by ParseObjCInterfaceDeclList().
  if (Tok.isObjCAtKeyword(tok::objc_end)) {
    ConsumeToken(); // the "end" identifier
    return ClsType;
  }
  Diag(Tok, diag::err_objc_missing_end);
  return 0;
}

///   objc-interface-decl-list:
///     empty
///     objc-interface-decl-list objc-property-decl [OBJC2]
///     objc-interface-decl-list objc-method-requirement [OBJC2]
///     objc-interface-decl-list objc-method-proto ';'
///     objc-interface-decl-list declaration
///     objc-interface-decl-list ';'
///
///   objc-method-requirement: [OBJC2]
///     @required
///     @optional
///
void Parser::ParseObjCInterfaceDeclList(DeclTy *interfaceDecl,
					tok::ObjCKeywordKind contextKey) {
  llvm::SmallVector<DeclTy*, 32>  allMethods;
  tok::ObjCKeywordKind MethodImplKind = tok::objc_not_keyword;
  while (1) {
    if (Tok.is(tok::at)) {
      SourceLocation AtLoc = ConsumeToken(); // the "@"
      tok::ObjCKeywordKind ocKind = Tok.getObjCKeywordID();
      
      if (ocKind == tok::objc_end) { // terminate list
        break;
      } else if (ocKind == tok::objc_required) { // protocols only
        ConsumeToken();
	MethodImplKind = ocKind;
	if (contextKey != tok::objc_protocol)
	  Diag(AtLoc, diag::err_objc_protocol_required);
      } else if (ocKind == tok::objc_optional) { // protocols only
        ConsumeToken();
	MethodImplKind = ocKind;
	if (contextKey != tok::objc_protocol)
	  Diag(AtLoc, diag::err_objc_protocol_optional);
      } else if (ocKind == tok::objc_property) {
        ParseObjCPropertyDecl(interfaceDecl);
        continue;
      } else {
        Diag(Tok, diag::err_objc_illegal_interface_qual);
        ConsumeToken();
      }
    }
    if (Tok.is(tok::minus) || Tok.is(tok::plus)) {
      DeclTy *methodPrototype = 
        ParseObjCMethodPrototype(interfaceDecl, MethodImplKind);
      allMethods.push_back(methodPrototype);
      // Consume the ';' here, since ParseObjCMethodPrototype() is re-used for
      // method definitions.
      ExpectAndConsume(tok::semi, diag::err_expected_semi_after,"method proto");
      continue;
    }
    if (Tok.is(tok::semi))
      ConsumeToken();
    else if (Tok.is(tok::eof))
      break;
    else {
      // FIXME: as the name implies, this rule allows function definitions.
      // We could pass a flag or check for functions during semantic analysis.
      ParseDeclarationOrFunctionDefinition();
    }
  }
  /// Insert collected methods declarations into the @interface object.
  Actions.ActOnAddMethodsToObjcDecl(CurScope, interfaceDecl,
                                    &allMethods[0], allMethods.size());
}

///   Parse property attribute declarations.
///
///   property-attr-decl: '(' property-attrlist ')'
///   property-attrlist:
///     property-attribute
///     property-attrlist ',' property-attribute
///   property-attribute:
///     getter '=' identifier
///     setter '=' identifier ':'
///     readonly
///     readwrite
///     assign
///     retain
///     copy
///     nonatomic
///
void Parser::ParseObjCPropertyAttribute (DeclTy *interfaceDecl) {
  SourceLocation loc = ConsumeParen(); // consume '('
  while (isObjCPropertyAttribute()) {
    const IdentifierInfo *II = Tok.getIdentifierInfo();
    // getter/setter require extra treatment.
    if (II == ObjcPropertyAttrs[objc_getter] || 
        II == ObjcPropertyAttrs[objc_setter]) {
      // skip getter/setter part.
      SourceLocation loc = ConsumeToken();
      if (Tok.is(tok::equal)) {
        loc = ConsumeToken();
        if (Tok.is(tok::identifier)) {
          if (II == ObjcPropertyAttrs[objc_setter]) {
            loc = ConsumeToken();  // consume method name
            if (Tok.isNot(tok::colon)) {
              Diag(loc, diag::err_expected_colon);
              SkipUntil(tok::r_paren,true,true);
              break;
            }
          }
        }
        else {
          Diag(loc, diag::err_expected_ident);
	  SkipUntil(tok::r_paren,true,true);
	  break;
	}
      }
      else {
        Diag(loc, diag::err_objc_expected_equal);    
        SkipUntil(tok::r_paren,true,true);
        break;
      }
    }
    ConsumeToken(); // consume last attribute token
    if (Tok.is(tok::comma)) {
      loc = ConsumeToken();
      continue;
    }
    if (Tok.is(tok::r_paren))
      break;
    Diag(loc, diag::err_expected_rparen);
    SkipUntil(tok::semi);
    return;
  }
  if (Tok.is(tok::r_paren))
    ConsumeParen();
  else {
    Diag(loc, diag::err_objc_expected_property_attr);
    SkipUntil(tok::r_paren); // recover from error inside attribute list
  }
}

///   Main routine to parse property declaration.
///
///   @property property-attr-decl[opt] property-component-decl ';'
///
void Parser::ParseObjCPropertyDecl(DeclTy *interfaceDecl) {
  assert(Tok.isObjCAtKeyword(tok::objc_property) &&
         "ParseObjCPropertyDecl(): Expected @property");
  ConsumeToken(); // the "property" identifier
  // Parse property attribute list, if any. 
  if (Tok.is(tok::l_paren)) {
    // property has attribute list.
    ParseObjCPropertyAttribute(0/*FIXME*/);
  }
  // Parse declaration portion of @property.
  llvm::SmallVector<DeclTy*, 32> PropertyDecls;
  ParseStructDeclaration(interfaceDecl, PropertyDecls);
  if (Tok.is(tok::semi)) 
    ConsumeToken();
  else {
    Diag(Tok, diag::err_expected_semi_decl_list);
    SkipUntil(tok::r_brace, true, true);
  }
}

///   objc-method-proto:
///     objc-instance-method objc-method-decl objc-method-attributes[opt] 
///     objc-class-method objc-method-decl objc-method-attributes[opt]
///
///   objc-instance-method: '-'
///   objc-class-method: '+'
///
///   objc-method-attributes:         [OBJC2]
///     __attribute__((deprecated))
///
Parser::DeclTy *Parser::ParseObjCMethodPrototype(DeclTy *IDecl, 
			  tok::ObjCKeywordKind MethodImplKind) {
  assert((Tok.is(tok::minus) || Tok.is(tok::plus)) && "expected +/-");

  tok::TokenKind methodType = Tok.getKind();  
  SourceLocation methodLoc = ConsumeToken();
  
  DeclTy *MDecl = ParseObjCMethodDecl(methodType, methodLoc, MethodImplKind);
  // Since this rule is used for both method declarations and definitions,
  // the caller is (optionally) responsible for consuming the ';'.
  return MDecl;
}

///   objc-selector:
///     identifier
///     one of
///       enum struct union if else while do for switch case default
///       break continue return goto asm sizeof typeof __alignof
///       unsigned long const short volatile signed restrict _Complex
///       in out inout bycopy byref oneway int char float double void _Bool
///
IdentifierInfo *Parser::ParseObjCSelector() {
  switch (Tok.getKind()) {
  default:
    return 0;
  case tok::identifier:
  case tok::kw_typeof:
  case tok::kw___alignof:
  case tok::kw_auto:
  case tok::kw_break:                    
  case tok::kw_case:                        
  case tok::kw_char:                        
  case tok::kw_const:                       
  case tok::kw_continue:                    
  case tok::kw_default:                     
  case tok::kw_do:                          
  case tok::kw_double:                      
  case tok::kw_else:                        
  case tok::kw_enum:                        
  case tok::kw_extern:                      
  case tok::kw_float:                       
  case tok::kw_for:                         
  case tok::kw_goto:                        
  case tok::kw_if:                       
  case tok::kw_inline:                     
  case tok::kw_int:                         
  case tok::kw_long:                        
  case tok::kw_register:                    
  case tok::kw_restrict:
  case tok::kw_return:                      
  case tok::kw_short:                       
  case tok::kw_signed:                      
  case tok::kw_sizeof:                      
  case tok::kw_static:                      
  case tok::kw_struct:                      
  case tok::kw_switch:                      
  case tok::kw_typedef:                     
  case tok::kw_union:                       
  case tok::kw_unsigned:                    
  case tok::kw_void:                        
  case tok::kw_volatile:                    
  case tok::kw_while:                       
  case tok::kw__Bool:
  case tok::kw__Complex:
    IdentifierInfo *II = Tok.getIdentifierInfo();
    ConsumeToken();
    return II;
  }
}

///   objc-type-qualifier: one of
///     in out inout bycopy byref oneway
///
bool Parser::isObjCTypeQualifier() {
  if (Tok.is(tok::identifier)) {
    const IdentifierInfo *II = Tok.getIdentifierInfo();
    for (unsigned i = 0; i < objc_NumQuals; ++i)
      if (II == ObjcTypeQuals[i]) return true;
  }
  return false;
}

///  property-attrlist: one of
///    readonly getter setter assign retain copy nonatomic
///
bool Parser::isObjCPropertyAttribute() {
  if (Tok.is(tok::identifier)) {
    const IdentifierInfo *II = Tok.getIdentifierInfo();
    for (unsigned i = 0; i < objc_NumAttrs; ++i)
      if (II == ObjcPropertyAttrs[i]) return true;
  }
  return false;
} 

///   objc-type-name:
///     '(' objc-type-qualifiers[opt] type-name ')'
///     '(' objc-type-qualifiers[opt] ')'
///
///   objc-type-qualifiers:
///     objc-type-qualifier
///     objc-type-qualifiers objc-type-qualifier
///
Parser::TypeTy *Parser::ParseObjCTypeName() {
  assert(Tok.is(tok::l_paren) && "expected (");
  
  SourceLocation LParenLoc = ConsumeParen(), RParenLoc;
  TypeTy *Ty = 0;
  
  while (isObjCTypeQualifier())
    ConsumeToken();

  if (isTypeSpecifierQualifier()) {
    Ty = ParseTypeName();
    // FIXME: back when Sema support is in place...
    // assert(Ty && "Parser::ParseObjCTypeName(): missing type");
  }
  if (Tok.isNot(tok::r_paren)) {
    MatchRHSPunctuation(tok::r_paren, LParenLoc);
    return 0; // FIXME: decide how we want to handle this error...
  }
  RParenLoc = ConsumeParen();
  return Ty;
}

///   objc-method-decl:
///     objc-selector
///     objc-keyword-selector objc-parmlist[opt]
///     objc-type-name objc-selector
///     objc-type-name objc-keyword-selector objc-parmlist[opt]
///
///   objc-keyword-selector:
///     objc-keyword-decl 
///     objc-keyword-selector objc-keyword-decl
///
///   objc-keyword-decl:
///     objc-selector ':' objc-type-name objc-keyword-attributes[opt] identifier
///     objc-selector ':' objc-keyword-attributes[opt] identifier
///     ':' objc-type-name objc-keyword-attributes[opt] identifier
///     ':' objc-keyword-attributes[opt] identifier
///
///   objc-parmlist:
///     objc-parms objc-ellipsis[opt]
///
///   objc-parms:
///     objc-parms , parameter-declaration
///
///   objc-ellipsis:
///     , ...
///
///   objc-keyword-attributes:         [OBJC2]
///     __attribute__((unused))
///
Parser::DeclTy *Parser::ParseObjCMethodDecl(tok::TokenKind mType, 
                                            SourceLocation mLoc,
					    tok::ObjCKeywordKind MethodImplKind)
{
  // Parse the return type.
  TypeTy *ReturnType = 0;
  if (Tok.is(tok::l_paren))
    ReturnType = ParseObjCTypeName();

  IdentifierInfo *SelIdent = ParseObjCSelector();
  if (Tok.isNot(tok::colon)) {
    if (!SelIdent) {
      Diag(Tok, diag::err_expected_ident); // missing selector name.
      // FIXME: this creates a unary selector with a null identifier, is this
      // ok??  Maybe we should skip to the next semicolon or something.
    }
    
    // If attributes exist after the method, parse them.
    AttributeList *MethodAttrs = 0;
    if (getLang().ObjC2 && Tok.is(tok::kw___attribute)) 
      MethodAttrs = ParseAttributes();
    
    Selector Sel = PP.getSelectorTable().getNullarySelector(SelIdent);
    return Actions.ActOnMethodDeclaration(mLoc, mType, ReturnType, Sel, 
                                          0, 0, MethodAttrs, MethodImplKind);
  }

  llvm::SmallVector<IdentifierInfo *, 12> KeyIdents;
  llvm::SmallVector<Action::TypeTy *, 12> KeyTypes;
  llvm::SmallVector<IdentifierInfo *, 12> ArgNames;
  
  Action::TypeTy *TypeInfo;
  while (1) {
    KeyIdents.push_back(SelIdent);
    
    // Each iteration parses a single keyword argument.
    if (Tok.isNot(tok::colon)) {
      Diag(Tok, diag::err_expected_colon);
      break;
    }
    ConsumeToken(); // Eat the ':'.
    if (Tok.is(tok::l_paren)) // Parse the argument type.
      TypeInfo = ParseObjCTypeName();
    else
      TypeInfo = 0;
    KeyTypes.push_back(TypeInfo);
    
    // If attributes exist before the argument name, parse them.
    if (getLang().ObjC2 && Tok.is(tok::kw___attribute))
      ParseAttributes(); // FIXME: pass attributes through.

    if (Tok.isNot(tok::identifier)) {
      Diag(Tok, diag::err_expected_ident); // missing argument name.
      break;
    }
    ArgNames.push_back(Tok.getIdentifierInfo());
    ConsumeToken(); // Eat the identifier.
    
    // Check for another keyword selector.
    SelIdent = ParseObjCSelector();
    if (!SelIdent && Tok.isNot(tok::colon))
      break;
    // We have a selector or a colon, continue parsing.
  }
  
  // Parse the (optional) parameter list.
  while (Tok.is(tok::comma)) {
    ConsumeToken();
    if (Tok.is(tok::ellipsis)) {
      ConsumeToken();
      break;
    }
    // Parse the c-style argument declaration-specifier.
    DeclSpec DS;
    ParseDeclarationSpecifiers(DS);
    // Parse the declarator. 
    Declarator ParmDecl(DS, Declarator::PrototypeContext);
    ParseDeclarator(ParmDecl);
  }
  
  // FIXME: Add support for optional parmameter list...
  // If attributes exist after the method, parse them.
  AttributeList *MethodAttrs = 0;
  if (getLang().ObjC2 && Tok.is(tok::kw___attribute)) 
    MethodAttrs = ParseAttributes();
  
  Selector Sel = PP.getSelectorTable().getSelector(KeyIdents.size(),
                                                   &KeyIdents[0]);
  return Actions.ActOnMethodDeclaration(mLoc, mType, ReturnType, Sel, 
                                        &KeyTypes[0], &ArgNames[0],
                                        MethodAttrs, MethodImplKind);
}

/// CmpProtocolVals - Comparison predicate for sorting protocols.
static bool CmpProtocolVals(const IdentifierInfo* const& lhs,
                            const IdentifierInfo* const& rhs) {
  return strcmp(lhs->getName(), rhs->getName()) < 0;
}

///   objc-protocol-refs:
///     '<' identifier-list '>'
///
bool Parser::ParseObjCProtocolReferences(
  llvm::SmallVectorImpl<IdentifierInfo*> &ProtocolRefs) {
  assert(Tok.is(tok::less) && "expected <");
  
  ConsumeToken(); // the "<"
  
  while (1) {
    if (Tok.isNot(tok::identifier)) {
      Diag(Tok, diag::err_expected_ident);
      SkipUntil(tok::greater);
      return true;
    }
    ProtocolRefs.push_back(Tok.getIdentifierInfo());
    ConsumeToken();
    
    if (Tok.isNot(tok::comma))
      break;
    ConsumeToken();
  }
  
  // Sort protocols, keyed by name.
  // Later on, we remove duplicates.
  std::stable_sort(ProtocolRefs.begin(), ProtocolRefs.end(), CmpProtocolVals);
  
  // Make protocol names unique.
  ProtocolRefs.erase(std::unique(ProtocolRefs.begin(), ProtocolRefs.end()), 
                     ProtocolRefs.end());

  // Consume the '>'.
  return ExpectAndConsume(tok::greater, diag::err_expected_greater);
}

///   objc-class-instance-variables:
///     '{' objc-instance-variable-decl-list[opt] '}'
///
///   objc-instance-variable-decl-list:
///     objc-visibility-spec
///     objc-instance-variable-decl ';'
///     ';'
///     objc-instance-variable-decl-list objc-visibility-spec
///     objc-instance-variable-decl-list objc-instance-variable-decl ';'
///     objc-instance-variable-decl-list ';'
///
///   objc-visibility-spec:
///     @private
///     @protected
///     @public
///     @package [OBJC2]
///
///   objc-instance-variable-decl:
///     struct-declaration 
///
void Parser::ParseObjCClassInstanceVariables(DeclTy *interfaceDecl) {
  assert(Tok.is(tok::l_brace) && "expected {");
  llvm::SmallVector<DeclTy*, 16> IvarDecls;
  llvm::SmallVector<DeclTy*, 32> AllIvarDecls;
  llvm::SmallVector<tok::ObjCKeywordKind, 32> AllVisibilities;
  
  SourceLocation LBraceLoc = ConsumeBrace(); // the "{"
  
  tok::ObjCKeywordKind visibility = tok::objc_private;
  // While we still have something to read, read the instance variables.
  while (Tok.isNot(tok::r_brace) && Tok.isNot(tok::eof)) {
    // Each iteration of this loop reads one objc-instance-variable-decl.
    
    // Check for extraneous top-level semicolon.
    if (Tok.is(tok::semi)) {
      Diag(Tok, diag::ext_extra_struct_semi);
      ConsumeToken();
      continue;
    }
    // Set the default visibility to private.
    if (Tok.is(tok::at)) { // parse objc-visibility-spec
      ConsumeToken(); // eat the @ sign
      switch (Tok.getObjCKeywordID()) {
      case tok::objc_private:
      case tok::objc_public:
      case tok::objc_protected:
      case tok::objc_package:
        visibility = Tok.getObjCKeywordID();
        ConsumeToken();
        continue; 
      default:
        Diag(Tok, diag::err_objc_illegal_visibility_spec);
        ConsumeToken();
        continue;
      }
    }
    ParseStructDeclaration(interfaceDecl, IvarDecls);
    for (unsigned i = 0; i < IvarDecls.size(); i++) {
	AllIvarDecls.push_back(IvarDecls[i]);
	AllVisibilities.push_back(visibility);
    }
    IvarDecls.clear();
    
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
  if (AllIvarDecls.size()) {  // Check for {} - no ivars in braces
    Actions.ActOnFields(CurScope, LBraceLoc, interfaceDecl, 
                        &AllIvarDecls[0], AllIvarDecls.size(),
                        &AllVisibilities[0]);
  }
  MatchRHSPunctuation(tok::r_brace, LBraceLoc);
  return;
}

///   objc-protocol-declaration:
///     objc-protocol-definition
///     objc-protocol-forward-reference
///
///   objc-protocol-definition:
///     @protocol identifier 
///       objc-protocol-refs[opt] 
///       objc-interface-decl-list 
///     @end
///
///   objc-protocol-forward-reference:
///     @protocol identifier-list ';'
///
///   "@protocol identifier ;" should be resolved as "@protocol
///   identifier-list ;": objc-interface-decl-list may not start with a
///   semicolon in the first alternative if objc-protocol-refs are omitted.

Parser::DeclTy *Parser::ParseObjCAtProtocolDeclaration(SourceLocation AtLoc) {
  assert(Tok.isObjCAtKeyword(tok::objc_protocol) &&
         "ParseObjCAtProtocolDeclaration(): Expected @protocol");
  ConsumeToken(); // the "protocol" identifier
  
  if (Tok.isNot(tok::identifier)) {
    Diag(Tok, diag::err_expected_ident); // missing protocol name.
    return 0;
  }
  // Save the protocol name, then consume it.
  IdentifierInfo *protocolName = Tok.getIdentifierInfo();
  SourceLocation nameLoc = ConsumeToken();
  
  llvm::SmallVector<IdentifierInfo *, 8> ProtocolRefs;
  if (Tok.is(tok::semi)) { // forward declaration of one protocol.
    ConsumeToken();
    ProtocolRefs.push_back(protocolName);
  }
  if (Tok.is(tok::comma)) { // list of forward declarations.
    // Parse the list of forward declarations.
    ProtocolRefs.push_back(protocolName);
    
    while (1) {
      ConsumeToken(); // the ','
      if (Tok.isNot(tok::identifier)) {
        Diag(Tok, diag::err_expected_ident);
        SkipUntil(tok::semi);
        return 0;
      }
      ProtocolRefs.push_back(Tok.getIdentifierInfo());
      ConsumeToken(); // the identifier
      
      if (Tok.isNot(tok::comma))
        break;
    }
    // Consume the ';'.
    if (ExpectAndConsume(tok::semi, diag::err_expected_semi_after, "@protocol"))
      return 0;
  }
  if (ProtocolRefs.size() > 0)
    return Actions.ActOnForwardProtocolDeclaration(CurScope, AtLoc,
                                                   &ProtocolRefs[0], 
                                                   ProtocolRefs.size());
  // Last, and definitely not least, parse a protocol declaration.
  if (Tok.is(tok::less)) {
    if (ParseObjCProtocolReferences(ProtocolRefs))
      return 0;
  }
  
  DeclTy *ProtoType = Actions.ActOnStartProtocolInterface(CurScope, AtLoc, 
                                protocolName, nameLoc,
                                &ProtocolRefs[0],
                                ProtocolRefs.size());
  ParseObjCInterfaceDeclList(ProtoType, tok::objc_protocol);

  // The @ sign was already consumed by ParseObjCInterfaceDeclList().
  if (Tok.isObjCAtKeyword(tok::objc_end)) {
    ConsumeToken(); // the "end" identifier
    return ProtoType;
  }
  Diag(Tok, diag::err_objc_missing_end);
  return 0;
}

///   objc-implementation:
///     objc-class-implementation-prologue
///     objc-category-implementation-prologue
///
///   objc-class-implementation-prologue:
///     @implementation identifier objc-superclass[opt]
///       objc-class-instance-variables[opt]
///
///   objc-category-implementation-prologue:
///     @implementation identifier ( identifier )

Parser::DeclTy *Parser::ParseObjCAtImplementationDeclaration(
  SourceLocation atLoc) {
  assert(Tok.isObjCAtKeyword(tok::objc_implementation) &&
         "ParseObjCAtImplementationDeclaration(): Expected @implementation");
  ConsumeToken(); // the "implementation" identifier
  
  if (Tok.isNot(tok::identifier)) {
    Diag(Tok, diag::err_expected_ident); // missing class or category name.
    return 0;
  }
  // We have a class or category name - consume it.
  IdentifierInfo *nameId = Tok.getIdentifierInfo();
  SourceLocation nameLoc = ConsumeToken(); // consume class or category name
  
  if (Tok.is(tok::l_paren)) { 
    // we have a category implementation.
    SourceLocation lparenLoc = ConsumeParen();
    SourceLocation categoryLoc, rparenLoc;
    IdentifierInfo *categoryId = 0;
    
    if (Tok.is(tok::identifier)) {
      categoryId = Tok.getIdentifierInfo();
      categoryLoc = ConsumeToken();
    } else {
      Diag(Tok, diag::err_expected_ident); // missing category name.
      return 0;
    }   
    if (Tok.isNot(tok::r_paren)) {
      Diag(Tok, diag::err_expected_rparen);
      SkipUntil(tok::r_paren, false); // don't stop at ';'
      return 0;
    }
    rparenLoc = ConsumeParen();
    DeclTy *ImplCatType = Actions.ActOnStartCategoryImplementation(CurScope,
                                    atLoc, nameId, nameLoc, categoryId, 
                                    categoryLoc);
    return ImplCatType;
  }
  // We have a class implementation
  SourceLocation superClassLoc;
  IdentifierInfo *superClassId = 0;
  if (Tok.is(tok::colon)) {
    // We have a super class
    ConsumeToken();
    if (Tok.isNot(tok::identifier)) {
      Diag(Tok, diag::err_expected_ident); // missing super class name.
      return 0;
    }
    superClassId = Tok.getIdentifierInfo();
    superClassLoc = ConsumeToken(); // Consume super class name
  }
  DeclTy *ImplClsType = Actions.ActOnStartClassImplementation(CurScope,
				  atLoc, nameId, nameLoc,
                                  superClassId, superClassLoc);
  
  if (Tok.is(tok::l_brace))
    ParseObjCClassInstanceVariables(ImplClsType/*FIXME*/); // we have ivars
  
  return ImplClsType;
}
Parser::DeclTy *Parser::ParseObjCAtEndDeclaration(SourceLocation atLoc) {
  assert(Tok.isObjCAtKeyword(tok::objc_end) &&
         "ParseObjCAtEndDeclaration(): Expected @end");
  ConsumeToken(); // the "end" identifier
  if (ObjcImpDecl) {
    // Checking is not necessary except that a parse error might have caused
    // @implementation not to have been parsed to completion and ObjcImpDecl 
    // could be 0.
    /// Insert collected methods declarations into the @interface object.
    Actions.ActOnAddMethodsToObjcDecl(CurScope, ObjcImpDecl,
                                      &AllImplMethods[0],AllImplMethods.size());
    ObjcImpDecl = 0;
    AllImplMethods.clear();
  }

  return 0;
}

///   compatibility-alias-decl:
///     @compatibility_alias alias-name  class-name ';'
///
Parser::DeclTy *Parser::ParseObjCAtAliasDeclaration(SourceLocation atLoc) {
  assert(Tok.isObjCAtKeyword(tok::objc_compatibility_alias) &&
         "ParseObjCAtAliasDeclaration(): Expected @compatibility_alias");
  ConsumeToken(); // consume compatibility_alias
  if (Tok.isNot(tok::identifier)) {
    Diag(Tok, diag::err_expected_ident);
    return 0;
  }
  ConsumeToken(); // consume alias-name
  if (Tok.isNot(tok::identifier)) {
    Diag(Tok, diag::err_expected_ident);
    return 0;
  }
  ConsumeToken(); // consume class-name;
  if (Tok.isNot(tok::semi))
    Diag(Tok, diag::err_expected_semi_after, "@compatibility_alias");
  return 0;
}

///   property-synthesis:
///     @synthesize property-ivar-list ';'
///
///   property-ivar-list:
///     property-ivar
///     property-ivar-list ',' property-ivar
///
///   property-ivar:
///     identifier
///     identifier '=' identifier
///
Parser::DeclTy *Parser::ParseObjCPropertySynthesize(SourceLocation atLoc) {
  assert(Tok.isObjCAtKeyword(tok::objc_synthesize) &&
         "ParseObjCPropertyDynamic(): Expected '@synthesize'");
  SourceLocation loc = ConsumeToken(); // consume dynamic
  if (Tok.isNot(tok::identifier)) {
    Diag(Tok, diag::err_expected_ident);
    return 0;
  }
  while (Tok.is(tok::identifier)) {
    ConsumeToken(); // consume property name
    if (Tok.is(tok::equal)) {
      // property '=' ivar-name
      ConsumeToken(); // consume '='
      if (Tok.isNot(tok::identifier)) {
        Diag(Tok, diag::err_expected_ident);
        break;
      }
      ConsumeToken(); // consume ivar-name
    }
    if (Tok.isNot(tok::comma))
      break;
    ConsumeToken(); // consume ','
  }
  if (Tok.isNot(tok::semi))
    Diag(Tok, diag::err_expected_semi_after, "@synthesize");
  return 0;
}

///   property-dynamic:
///     @dynamic  property-list
///
///   property-list:
///     identifier
///     property-list ',' identifier
///
Parser::DeclTy *Parser::ParseObjCPropertyDynamic(SourceLocation atLoc) {
  assert(Tok.isObjCAtKeyword(tok::objc_dynamic) &&
         "ParseObjCPropertyDynamic(): Expected '@dynamic'");
  SourceLocation loc = ConsumeToken(); // consume dynamic
  if (Tok.isNot(tok::identifier)) {
    Diag(Tok, diag::err_expected_ident);
    return 0;
  }
  while (Tok.is(tok::identifier)) {
    ConsumeToken(); // consume property name
    if (Tok.isNot(tok::comma))
      break;
    ConsumeToken(); // consume ','
  }
  if (Tok.isNot(tok::semi))
    Diag(Tok, diag::err_expected_semi_after, "@dynamic");
  return 0;
}
 
///  objc-throw-statement:
///    throw expression[opt];
///
Parser::DeclTy *Parser::ParseObjCThrowStmt(SourceLocation atLoc) {
  ConsumeToken(); // consume throw
  if (Tok.isNot(tok::semi)) {
    ExprResult Res = ParseExpression();
    if (Res.isInvalid) {
      SkipUntil(tok::semi);
      return 0;
    }
  }
  return 0;
}

///  objc-try-catch-statement:
///    @try compound-statement objc-catch-list[opt]
///    @try compound-statement objc-catch-list[opt] @finally compound-statement
///
///  objc-catch-list:
///    @catch ( parameter-declaration ) compound-statement
///    objc-catch-list @catch ( catch-parameter-declaration ) compound-statement
///  catch-parameter-declaration:
///     parameter-declaration
///     '...' [OBJC2]
///
Parser::DeclTy *Parser::ParseObjCTryStmt(SourceLocation atLoc) {
  bool catch_or_finally_seen = false;
  ConsumeToken(); // consume try
  if (Tok.isNot(tok::l_brace)) {
    Diag (Tok, diag::err_expected_lbrace);
    return 0;
  }
  StmtResult TryBody = ParseCompoundStatementBody();
  while (Tok.is(tok::at)) {
    ConsumeToken();
    if (Tok.getIdentifierInfo()->getObjCKeywordID() == tok::objc_catch) {
      SourceLocation catchLoc = ConsumeToken(); // consume catch
      if (Tok.is(tok::l_paren)) {
        ConsumeParen();
        if (Tok.isNot(tok::ellipsis)) {
          DeclSpec DS;
          ParseDeclarationSpecifiers(DS);
          // Parse the parameter-declaration. 
          // FIXME: BlockContext may not be the right context!
          Declarator ParmDecl(DS, Declarator::BlockContext);
          ParseDeclarator(ParmDecl);
        }
        else
          ConsumeToken(); // consume '...'
        ConsumeParen();
        StmtResult CatchMody = ParseCompoundStatementBody();
      }
      else {
        Diag(catchLoc, diag::err_expected_lparen_after, "@catch clause");
        return 0;
      }
      catch_or_finally_seen = true;
    }
    else if (Tok.getIdentifierInfo()->getObjCKeywordID() == tok::objc_finally) {
       ConsumeToken(); // consume finally
      StmtResult FinallyBody = ParseCompoundStatementBody();
      catch_or_finally_seen = true;
      break;
    }
  }
  if (!catch_or_finally_seen)
    Diag(atLoc, diag::err_missing_catch_finally);
  return 0;
}

///   objc-method-def: objc-method-proto ';'[opt] '{' body '}'
///
void Parser::ParseObjCInstanceMethodDefinition() {
  assert(Tok.is(tok::minus) && "Method definitions should start with '-'");
  
  // FIXME: @optional/@protocol??
  AllImplMethods.push_back(ParseObjCMethodPrototype(ObjcImpDecl));
  // parse optional ';'
  if (Tok.is(tok::semi))
    ConsumeToken();

  if (Tok.isNot(tok::l_brace)) {
    Diag (Tok, diag::err_expected_lbrace);
    return;
  }
    
  StmtResult FnBody = ParseCompoundStatementBody();
}

///   objc-method-def: objc-method-proto ';'[opt] '{' body '}'
///
void Parser::ParseObjCClassMethodDefinition() {
  assert(Tok.is(tok::plus) && "Class method definitions should start with '+'");
  // FIXME: @optional/@protocol??
  AllImplMethods.push_back(ParseObjCMethodPrototype(ObjcImpDecl));
  // parse optional ';'
  if (Tok.is(tok::semi))
    ConsumeToken();
  if (Tok.isNot(tok::l_brace)) {
    Diag (Tok, diag::err_expected_lbrace);
    return;
  }
  
  StmtResult FnBody = ParseCompoundStatementBody();
}

Parser::ExprResult Parser::ParseObjCExpression(SourceLocation AtLoc) {

  switch (Tok.getKind()) {
    case tok::string_literal:    // primary-expression: string-literal
    case tok::wide_string_literal:
      return ParsePostfixExpressionSuffix(ParseObjCStringLiteral());
    default:
      break;
  }
  
  switch (Tok.getIdentifierInfo()->getObjCKeywordID()) {
    case tok::objc_encode:
      return ParsePostfixExpressionSuffix(ParseObjCEncodeExpression());
    case tok::objc_protocol:
      return ParsePostfixExpressionSuffix(ParseObjCProtocolExpression());
    default:
      Diag(AtLoc, diag::err_unexpected_at);
      SkipUntil(tok::semi);
      break;
  }
  
  return 0;
}

///   objc-message-expr: 
///     '[' objc-receiver objc-message-args ']'
///
///   objc-receiver:
///     expression
///     class-name
///     type-name
///  
///   objc-message-args:
///     objc-selector
///     objc-keywordarg-list
///
///   objc-keywordarg-list:
///     objc-keywordarg
///     objc-keywordarg-list objc-keywordarg
///
///   objc-keywordarg: 
///     selector-name[opt] ':' objc-keywordexpr
///
///   objc-keywordexpr:
///     nonempty-expr-list
///
///   nonempty-expr-list:
///     assignment-expression
///     nonempty-expr-list , assignment-expression
///   
Parser::ExprResult Parser::ParseObjCMessageExpression() {
  assert(Tok.is(tok::l_square) && "'[' expected");
  SourceLocation LBracloc = ConsumeBracket(); // consume '['
  IdentifierInfo *ReceiverName = 0;
  ExprTy *ReceiverExpr = 0;
  // Parse receiver
  if (Tok.is(tok::identifier) &&
      Actions.isTypeName(*Tok.getIdentifierInfo(), CurScope)) {
    ReceiverName = Tok.getIdentifierInfo();
    ConsumeToken();
  } else {
    ExprResult Res = ParseAssignmentExpression();
    if (Res.isInvalid) {
      SkipUntil(tok::identifier);
      return Res;
    }
    ReceiverExpr = Res.Val;
  }
  // Parse objc-selector
  IdentifierInfo *selIdent = ParseObjCSelector();

  llvm::SmallVector<IdentifierInfo *, 12> KeyIdents;
  llvm::SmallVector<Action::ExprTy *, 12> KeyExprs;

  if (Tok.is(tok::colon)) {
    while (1) {
      // Each iteration parses a single keyword argument.
      KeyIdents.push_back(selIdent);

      if (Tok.isNot(tok::colon)) {
        Diag(Tok, diag::err_expected_colon);
        SkipUntil(tok::semi);
        return true;
      }
      ConsumeToken(); // Eat the ':'.
      ///  Parse the expression after ':' 
      ExprResult Res = ParseAssignmentExpression();
      if (Res.isInvalid) {
        SkipUntil(tok::identifier);
        return Res;
      }
      // We have a valid expression.
      KeyExprs.push_back(Res.Val);
      
      // Check for another keyword selector.
      selIdent = ParseObjCSelector();
      if (!selIdent && Tok.isNot(tok::colon))
        break;
      // We have a selector or a colon, continue parsing.
    }
    // Parse the, optional, argument list, comma separated.
    while (Tok.is(tok::comma)) {
      ConsumeToken();
      /// Parse the expression after ','
      ParseAssignmentExpression();
    }
  } else if (!selIdent) {
    Diag(Tok, diag::err_expected_ident); // missing selector name.
    SkipUntil(tok::semi);
    return 0;
  }
  
  if (Tok.isNot(tok::r_square)) {
    Diag(Tok, diag::err_expected_rsquare);
    SkipUntil(tok::semi);
    return 0;
  }
  SourceLocation RBracloc = ConsumeBracket(); // consume ']'
  
  unsigned nKeys = KeyIdents.size();
  if (nKeys == 0)
    KeyIdents.push_back(selIdent);
  Selector Sel = PP.getSelectorTable().getSelector(nKeys, &KeyIdents[0]);
  
  // We've just parsed a keyword message.
  if (ReceiverName) 
    return Actions.ActOnClassMessage(ReceiverName, Sel, LBracloc, RBracloc,
                                     &KeyExprs[0]);
  return Actions.ActOnInstanceMessage(ReceiverExpr, Sel, LBracloc, RBracloc,
                                      &KeyExprs[0]);
}

Parser::ExprResult Parser::ParseObjCStringLiteral() {
  ExprResult Res = ParseStringLiteralExpression();

  if (Res.isInvalid) return Res;

  return Actions.ParseObjCStringLiteral(Res.Val);
}

///    objc-encode-expression:
///      @encode ( type-name )
Parser::ExprResult Parser::ParseObjCEncodeExpression() {
  assert(Tok.isObjCAtKeyword(tok::objc_encode) && "Not an @encode expression!");
  
  SourceLocation EncLoc = ConsumeToken();
  
  if (Tok.isNot(tok::l_paren)) {
    Diag(Tok, diag::err_expected_lparen_after, "@encode");
    return true;
  }
   
  SourceLocation LParenLoc = ConsumeParen();
  
  TypeTy *Ty = ParseTypeName();
  
  SourceLocation RParenLoc = MatchRHSPunctuation(tok::r_paren, LParenLoc);
   
  return Actions.ParseObjCEncodeExpression(EncLoc, LParenLoc, Ty, 
                                           RParenLoc);
}

///     objc-protocol-expression
///       @protocol ( protocol-name )

Parser::ExprResult Parser::ParseObjCProtocolExpression()
{
  SourceLocation ProtoLoc = ConsumeToken();
  
  if (Tok.isNot(tok::l_paren)) {
    Diag(Tok, diag::err_expected_lparen_after, "@protocol");
    return true;
  }
  
  SourceLocation LParenLoc = ConsumeParen();
  
  if (Tok.isNot(tok::identifier)) {
    Diag(Tok, diag::err_expected_ident);
    return true;
  }

  // FIXME: Do something with the protocol name
  ConsumeToken();
  
  SourceLocation RParenLoc = MatchRHSPunctuation(tok::r_paren, LParenLoc);

  // FIXME 
  return 0;
}
