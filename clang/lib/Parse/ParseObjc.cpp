//===--- ParseObjC.cpp - Objective C Parsing ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the Objective-C portions of the Parser interface.
//
//===----------------------------------------------------------------------===//

#include "clang/Parse/Parser.h"
#include "clang/Parse/DeclSpec.h"
#include "clang/Parse/Scope.h"
#include "clang/Basic/Diagnostic.h"
#include "llvm/ADT/SmallVector.h"
using namespace clang;


/// ParseExternalDeclaration:
///       external-declaration: [C99 6.9]
/// [OBJC]  objc-class-definition
/// [OBJC]  objc-class-declaration
/// [OBJC]  objc-alias-declaration
/// [OBJC]  objc-protocol-definition
/// [OBJC]  objc-method-definition
/// [OBJC]  '@' 'end'
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
    return ParseObjCAtImplementationDeclaration(AtLoc);
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
  
  return Actions.ActOnForwardClassDeclaration(atLoc,
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
    SourceLocation EndProtoLoc;
    llvm::SmallVector<DeclTy *, 8> ProtocolRefs;
    if (Tok.is(tok::less) &&
        ParseObjCProtocolReferences(ProtocolRefs, true, EndProtoLoc))
      return 0;
    
    if (attrList) // categories don't support attributes.
      Diag(Tok, diag::err_objc_no_attributes_on_category);
    
    DeclTy *CategoryType = Actions.ActOnStartCategoryInterface(atLoc, 
                                     nameId, nameLoc, categoryId, categoryLoc,
                                     &ProtocolRefs[0], ProtocolRefs.size(),
                                     EndProtoLoc);
    
    ParseObjCInterfaceDeclList(CategoryType, tok::objc_not_keyword);
    return CategoryType;
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
  llvm::SmallVector<Action::DeclTy*, 8> ProtocolRefs;
  SourceLocation EndProtoLoc;
  if (Tok.is(tok::less) &&
      ParseObjCProtocolReferences(ProtocolRefs, true, EndProtoLoc))
    return 0;
  
  DeclTy *ClsType = 
    Actions.ActOnStartClassInterface(atLoc, nameId, nameLoc, 
                                     superClassId, superClassLoc,
                                     &ProtocolRefs[0], ProtocolRefs.size(),
                                     EndProtoLoc, attrList);
            
  if (Tok.is(tok::l_brace))
    ParseObjCClassInstanceVariables(ClsType, atLoc);

  ParseObjCInterfaceDeclList(ClsType, tok::objc_interface);
  return ClsType;
}

/// constructSetterName - Return the setter name for the given
/// identifier, i.e. "set" + Name where the initial character of Name
/// has been capitalized.
static IdentifierInfo *constructSetterName(IdentifierTable &Idents,
                                           const IdentifierInfo *Name) {
  unsigned N = Name->getLength();
  char *SelectorName = new char[3 + N];
  memcpy(SelectorName, "set", 3);
  memcpy(&SelectorName[3], Name->getName(), N);
  SelectorName[3] = toupper(SelectorName[3]);

  IdentifierInfo *Setter = 
    &Idents.get(SelectorName, &SelectorName[3 + N]);
  delete[] SelectorName;
  return Setter;
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
  llvm::SmallVector<DeclTy*, 32> allMethods;
  llvm::SmallVector<DeclTy*, 16> allProperties;
  tok::ObjCKeywordKind MethodImplKind = tok::objc_not_keyword;
  
  SourceLocation AtEndLoc;

  while (1) {
    // If this is a method prototype, parse it.
    if (Tok.is(tok::minus) || Tok.is(tok::plus)) {
      DeclTy *methodPrototype = 
        ParseObjCMethodPrototype(interfaceDecl, MethodImplKind);
      allMethods.push_back(methodPrototype);
      // Consume the ';' here, since ParseObjCMethodPrototype() is re-used for
      // method definitions.
      ExpectAndConsume(tok::semi, diag::err_expected_semi_after,"method proto");
      continue;
    }
    
    // Ignore excess semicolons.
    if (Tok.is(tok::semi)) {
      ConsumeToken();
      continue;
    }
    
    // If we got to the end of the file, exit the loop.
    if (Tok.is(tok::eof))
      break;
    
    // If we don't have an @ directive, parse it as a function definition.
    if (Tok.isNot(tok::at)) {
      // FIXME: as the name implies, this rule allows function definitions.
      // We could pass a flag or check for functions during semantic analysis.
      ParseDeclarationOrFunctionDefinition();
      continue;
    }
    
    // Otherwise, we have an @ directive, eat the @.
    SourceLocation AtLoc = ConsumeToken(); // the "@"
    tok::ObjCKeywordKind DirectiveKind = Tok.getObjCKeywordID();
    
    if (DirectiveKind == tok::objc_end) { // @end -> terminate list
      AtEndLoc = AtLoc;
      break;
    }
    
    // Eat the identifier.
    ConsumeToken();

    switch (DirectiveKind) {
    default:
      // FIXME: If someone forgets an @end on a protocol, this loop will
      // continue to eat up tons of stuff and spew lots of nonsense errors.  It
      // would probably be better to bail out if we saw an @class or @interface
      // or something like that.
      Diag(AtLoc, diag::err_objc_illegal_interface_qual);
      // Skip until we see an '@' or '}' or ';'.
      SkipUntil(tok::r_brace, tok::at);
      break;
      
    case tok::objc_required:
    case tok::objc_optional:
      // This is only valid on protocols.
      // FIXME: Should this check for ObjC2 being enabled?
      if (contextKey != tok::objc_protocol)
        Diag(AtLoc, diag::err_objc_directive_only_in_protocol);
      else
        MethodImplKind = DirectiveKind;
      break;
        
    case tok::objc_property:
      if (!getLang().ObjC2)
        Diag(AtLoc, diag::err_objc_propertoes_require_objc2);

      ObjCDeclSpec OCDS;
      // Parse property attribute list, if any. 
      if (Tok.is(tok::l_paren)) {
        ParseObjCPropertyAttribute(OCDS);
      }
        
      // Parse all the comma separated declarators.
      DeclSpec DS;
      llvm::SmallVector<FieldDeclarator, 8> FieldDeclarators;
      ParseStructDeclaration(DS, FieldDeclarators);
      
      ExpectAndConsume(tok::semi, diag::err_expected_semi_decl_list, "",
                       tok::at);
      
      // Convert them all to property declarations.
      for (unsigned i = 0, e = FieldDeclarators.size(); i != e; ++i) {
        FieldDeclarator &FD = FieldDeclarators[i];
        if (FD.D.getIdentifier() == 0) {
          Diag(AtLoc, diag::err_objc_property_requires_field_name,
               FD.D.getSourceRange());
          continue;
        }
        
        // Install the property declarator into interfaceDecl.
        IdentifierInfo *SelName =
          OCDS.getGetterName() ? OCDS.getGetterName() : FD.D.getIdentifier();
        
        Selector GetterSel = 
          PP.getSelectorTable().getNullarySelector(SelName);
        IdentifierInfo *SetterName = OCDS.getSetterName();
        if (!SetterName)
          SetterName = constructSetterName(PP.getIdentifierTable(),
                                           FD.D.getIdentifier());
        Selector SetterSel = 
          PP.getSelectorTable().getUnarySelector(SetterName);
        DeclTy *Property = Actions.ActOnProperty(CurScope, AtLoc, FD, OCDS,
                                                 GetterSel, SetterSel,
                                                 MethodImplKind);
        allProperties.push_back(Property);
      }
      break;
    }
  }

  // We break out of the big loop in two cases: when we see @end or when we see
  // EOF.  In the former case, eat the @end.  In the later case, emit an error.
  if (Tok.isObjCAtKeyword(tok::objc_end))
    ConsumeToken(); // the "end" identifier
  else
    Diag(Tok, diag::err_objc_missing_end);
  
  // Insert collected methods declarations into the @interface object.
  // This passes in an invalid SourceLocation for AtEndLoc when EOF is hit.
  Actions.ActOnAtEnd(AtEndLoc, interfaceDecl,
                     allMethods.empty() ? 0 : &allMethods[0],
                     allMethods.size(), 
                     allProperties.empty() ? 0 : &allProperties[0],
                     allProperties.size());
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
void Parser::ParseObjCPropertyAttribute(ObjCDeclSpec &DS) {
  SourceLocation LHSLoc = ConsumeParen(); // consume '('
  
  while (1) {
    const IdentifierInfo *II = Tok.getIdentifierInfo();
    
    // If this is not an identifier at all, bail out early.
    if (II == 0) {
      MatchRHSPunctuation(tok::r_paren, LHSLoc);
      return;
    }
    
    // getter/setter require extra treatment.
    if (II == ObjCPropertyAttrs[objc_getter] || 
        II == ObjCPropertyAttrs[objc_setter]) {
      // skip getter/setter part.
      SourceLocation loc = ConsumeToken();
      if (Tok.is(tok::equal)) {
        loc = ConsumeToken();
        if (Tok.is(tok::identifier)) {
          if (II == ObjCPropertyAttrs[objc_setter]) {
            DS.setPropertyAttributes(ObjCDeclSpec::DQ_PR_setter);
            DS.setSetterName(Tok.getIdentifierInfo());
            loc = ConsumeToken();  // consume method name
            if (Tok.isNot(tok::colon)) {
              Diag(loc, diag::err_expected_colon);
              SkipUntil(tok::r_paren);
              return;
            }
          } else {
            DS.setPropertyAttributes(ObjCDeclSpec::DQ_PR_getter);
            DS.setGetterName(Tok.getIdentifierInfo());
          }
        } else {
          Diag(loc, diag::err_expected_ident);
          SkipUntil(tok::r_paren);
          return;
        }
      }
      else {
        Diag(loc, diag::err_objc_expected_equal);    
        SkipUntil(tok::r_paren);
        return;
      }
    } else if (II == ObjCPropertyAttrs[objc_readonly])
      DS.setPropertyAttributes(ObjCDeclSpec::DQ_PR_readonly);
    else if (II == ObjCPropertyAttrs[objc_assign])
      DS.setPropertyAttributes(ObjCDeclSpec::DQ_PR_assign);
    else if (II == ObjCPropertyAttrs[objc_readwrite])
      DS.setPropertyAttributes(ObjCDeclSpec::DQ_PR_readwrite);
    else if (II == ObjCPropertyAttrs[objc_retain])
      DS.setPropertyAttributes(ObjCDeclSpec::DQ_PR_retain);
    else if (II == ObjCPropertyAttrs[objc_copy])
      DS.setPropertyAttributes(ObjCDeclSpec::DQ_PR_copy);
    else if (II == ObjCPropertyAttrs[objc_nonatomic])
      DS.setPropertyAttributes(ObjCDeclSpec::DQ_PR_nonatomic);
    else {
      Diag(Tok.getLocation(), diag::err_objc_expected_property_attr,
           II->getName());
      SkipUntil(tok::r_paren);
      return;
    }
    
    ConsumeToken(); // consume last attribute token
    if (Tok.is(tok::comma)) {
      ConsumeToken();
      continue;
    }
    
    if (Tok.is(tok::r_paren)) {
      ConsumeParen();
      return;
    }
    
    MatchRHSPunctuation(tok::r_paren, LHSLoc);
    return;
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
  SourceLocation mLoc = ConsumeToken();
  
  DeclTy *MDecl = ParseObjCMethodDecl(mLoc, methodType, IDecl, MethodImplKind);
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
IdentifierInfo *Parser::ParseObjCSelector(SourceLocation &SelectorLoc) {
  switch (Tok.getKind()) {
  default:
    return 0;
  case tok::identifier:
  case tok::kw_asm:
  case tok::kw_auto:
  case tok::kw_bool:
  case tok::kw_break:
  case tok::kw_case:
  case tok::kw_catch:
  case tok::kw_char:
  case tok::kw_class:
  case tok::kw_const:
  case tok::kw_const_cast:
  case tok::kw_continue:
  case tok::kw_default:
  case tok::kw_delete:
  case tok::kw_do:
  case tok::kw_double:
  case tok::kw_dynamic_cast:
  case tok::kw_else:
  case tok::kw_enum:
  case tok::kw_explicit:
  case tok::kw_export:
  case tok::kw_extern:
  case tok::kw_false:
  case tok::kw_float:
  case tok::kw_for:
  case tok::kw_friend:
  case tok::kw_goto:
  case tok::kw_if:
  case tok::kw_inline:
  case tok::kw_int:
  case tok::kw_long:
  case tok::kw_mutable:
  case tok::kw_namespace:
  case tok::kw_new:
  case tok::kw_operator:
  case tok::kw_private:
  case tok::kw_protected:
  case tok::kw_public:
  case tok::kw_register:
  case tok::kw_reinterpret_cast:
  case tok::kw_restrict:
  case tok::kw_return:
  case tok::kw_short:
  case tok::kw_signed:
  case tok::kw_sizeof:
  case tok::kw_static:
  case tok::kw_static_cast:
  case tok::kw_struct:
  case tok::kw_switch:
  case tok::kw_template:
  case tok::kw_this:
  case tok::kw_throw:
  case tok::kw_true:
  case tok::kw_try:
  case tok::kw_typedef:
  case tok::kw_typeid:
  case tok::kw_typename:
  case tok::kw_typeof:
  case tok::kw_union:
  case tok::kw_unsigned:
  case tok::kw_using:
  case tok::kw_virtual:
  case tok::kw_void:
  case tok::kw_volatile:
  case tok::kw_wchar_t:
  case tok::kw_while:
  case tok::kw__Bool:
  case tok::kw__Complex:
  case tok::kw___alignof:
    IdentifierInfo *II = Tok.getIdentifierInfo();
    SelectorLoc = ConsumeToken();
    return II;
  }
}

///  objc-for-collection-in: 'in'
///
bool Parser::isTokIdentifier_in() const {
  // FIXME: May have to do additional look-ahead to only allow for
  // valid tokens following an 'in'; such as an identifier, unary operators,
  // '[' etc.
  return (getLang().ObjC2 && Tok.is(tok::identifier) && 
          Tok.getIdentifierInfo() == ObjCTypeQuals[objc_in]);
}

/// ParseObjCTypeQualifierList - This routine parses the objective-c's type
/// qualifier list and builds their bitmask representation in the input
/// argument.
///
///   objc-type-qualifiers:
///     objc-type-qualifier
///     objc-type-qualifiers objc-type-qualifier
///
void Parser::ParseObjCTypeQualifierList(ObjCDeclSpec &DS) {
  while (1) {
    if (Tok.isNot(tok::identifier))
      return;
    
    const IdentifierInfo *II = Tok.getIdentifierInfo();
    for (unsigned i = 0; i != objc_NumQuals; ++i) {
      if (II != ObjCTypeQuals[i])
        continue;
      
      ObjCDeclSpec::ObjCDeclQualifier Qual;
      switch (i) {
      default: assert(0 && "Unknown decl qualifier");
      case objc_in:     Qual = ObjCDeclSpec::DQ_In; break;
      case objc_out:    Qual = ObjCDeclSpec::DQ_Out; break;
      case objc_inout:  Qual = ObjCDeclSpec::DQ_Inout; break;
      case objc_oneway: Qual = ObjCDeclSpec::DQ_Oneway; break;
      case objc_bycopy: Qual = ObjCDeclSpec::DQ_Bycopy; break;
      case objc_byref:  Qual = ObjCDeclSpec::DQ_Byref; break;
      }
      DS.setObjCDeclQualifier(Qual);
      ConsumeToken();
      II = 0;
      break;
    }
    
    // If this wasn't a recognized qualifier, bail out.
    if (II) return;
  }
}

///   objc-type-name:
///     '(' objc-type-qualifiers[opt] type-name ')'
///     '(' objc-type-qualifiers[opt] ')'
///
Parser::TypeTy *Parser::ParseObjCTypeName(ObjCDeclSpec &DS) {
  assert(Tok.is(tok::l_paren) && "expected (");
  
  SourceLocation LParenLoc = ConsumeParen(), RParenLoc;
  SourceLocation TypeStartLoc = Tok.getLocation();
  TypeTy *Ty = 0;
  
  // Parse type qualifiers, in, inout, etc.
  ParseObjCTypeQualifierList(DS);

  if (isTypeSpecifierQualifier()) {
    Ty = ParseTypeName();
    // FIXME: back when Sema support is in place...
    // assert(Ty && "Parser::ParseObjCTypeName(): missing type");
  }
  
  if (Tok.isNot(tok::r_paren)) {
    // If we didn't eat any tokens, then this isn't a type.
    if (Tok.getLocation() == TypeStartLoc) {
      Diag(Tok.getLocation(), diag::err_expected_type);
      SkipUntil(tok::r_brace);
    } else {
      // Otherwise, we found *something*, but didn't get a ')' in the right
      // place.  Emit an error then return what we have as the type.
      MatchRHSPunctuation(tok::r_paren, LParenLoc);
    }
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
Parser::DeclTy *Parser::ParseObjCMethodDecl(SourceLocation mLoc,
                                            tok::TokenKind mType,
                                            DeclTy *IDecl,
                                            tok::ObjCKeywordKind MethodImplKind)
{
  // Parse the return type if present.
  TypeTy *ReturnType = 0;
  ObjCDeclSpec DSRet;
  if (Tok.is(tok::l_paren))
    ReturnType = ParseObjCTypeName(DSRet);
  
  SourceLocation selLoc;
  IdentifierInfo *SelIdent = ParseObjCSelector(selLoc);

  if (!SelIdent) { // missing selector name.
    Diag(Tok.getLocation(), diag::err_expected_selector_for_method,
         SourceRange(mLoc, Tok.getLocation()));
    // Skip until we get a ; or {}.
    SkipUntil(tok::r_brace);
    return 0;
  }
  
  if (Tok.isNot(tok::colon)) {
    // If attributes exist after the method, parse them.
    AttributeList *MethodAttrs = 0;
    if (getLang().ObjC2 && Tok.is(tok::kw___attribute)) 
      MethodAttrs = ParseAttributes();
    
    Selector Sel = PP.getSelectorTable().getNullarySelector(SelIdent);
    return Actions.ActOnMethodDeclaration(mLoc, Tok.getLocation(),
                                          mType, IDecl, DSRet, ReturnType, Sel,
                                          0, 0, 0, MethodAttrs, MethodImplKind);
  }

  llvm::SmallVector<IdentifierInfo *, 12> KeyIdents;
  llvm::SmallVector<Action::TypeTy *, 12> KeyTypes;
  llvm::SmallVector<ObjCDeclSpec, 12> ArgTypeQuals;
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
    ObjCDeclSpec DSType;
    if (Tok.is(tok::l_paren)) // Parse the argument type.
      TypeInfo = ParseObjCTypeName(DSType);
    else
      TypeInfo = 0;
    KeyTypes.push_back(TypeInfo);
    ArgTypeQuals.push_back(DSType);
    
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
    SourceLocation Loc;
    SelIdent = ParseObjCSelector(Loc);
    if (!SelIdent && Tok.isNot(tok::colon))
      break;
    // We have a selector or a colon, continue parsing.
  }
  
  bool isVariadic = false;
  
  // Parse the (optional) parameter list.
  while (Tok.is(tok::comma)) {
    ConsumeToken();
    if (Tok.is(tok::ellipsis)) {
      isVariadic = true;
      ConsumeToken();
      break;
    }
    // FIXME: implement this...
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
  return Actions.ActOnMethodDeclaration(mLoc, Tok.getLocation(),
                                        mType, IDecl, DSRet, ReturnType, Sel, 
                                        &ArgTypeQuals[0], &KeyTypes[0], 
                                        &ArgNames[0], MethodAttrs, 
                                        MethodImplKind, isVariadic);
}

///   objc-protocol-refs:
///     '<' identifier-list '>'
///
bool Parser::
ParseObjCProtocolReferences(llvm::SmallVectorImpl<Action::DeclTy*> &Protocols,
                            bool WarnOnDeclarations, SourceLocation &EndLoc) {
  assert(Tok.is(tok::less) && "expected <");
  
  ConsumeToken(); // the "<"
  
  llvm::SmallVector<IdentifierLocPair, 8> ProtocolIdents;
  
  while (1) {
    if (Tok.isNot(tok::identifier)) {
      Diag(Tok, diag::err_expected_ident);
      SkipUntil(tok::greater);
      return true;
    }
    ProtocolIdents.push_back(std::make_pair(Tok.getIdentifierInfo(),
                                       Tok.getLocation()));
    ConsumeToken();
    
    if (Tok.isNot(tok::comma))
      break;
    ConsumeToken();
  }
  
  // Consume the '>'.
  if (Tok.isNot(tok::greater)) {
    Diag(Tok, diag::err_expected_greater);
    return true;
  }
  
  EndLoc = ConsumeAnyToken();
  
  // Convert the list of protocols identifiers into a list of protocol decls.
  Actions.FindProtocolDeclaration(WarnOnDeclarations,
                                  &ProtocolIdents[0], ProtocolIdents.size(),
                                  Protocols);
  return false;
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
void Parser::ParseObjCClassInstanceVariables(DeclTy *interfaceDecl,
                                             SourceLocation atLoc) {
  assert(Tok.is(tok::l_brace) && "expected {");
  llvm::SmallVector<DeclTy*, 32> AllIvarDecls;
  llvm::SmallVector<FieldDeclarator, 8> FieldDeclarators;

  SourceLocation LBraceLoc = ConsumeBrace(); // the "{"
  
  tok::ObjCKeywordKind visibility = tok::objc_protected;
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
        continue;
      }
    }
    
    // Parse all the comma separated declarators.
    DeclSpec DS;
    FieldDeclarators.clear();
    ParseStructDeclaration(DS, FieldDeclarators);
    
    // Convert them all to fields.
    for (unsigned i = 0, e = FieldDeclarators.size(); i != e; ++i) {
      FieldDeclarator &FD = FieldDeclarators[i];
      // Install the declarator into interfaceDecl.
      DeclTy *Field = Actions.ActOnIvar(CurScope,
                                         DS.getSourceRange().getBegin(),
                                         FD.D, FD.BitfieldSize, visibility);
      AllIvarDecls.push_back(Field);
    }
    
    if (Tok.is(tok::semi)) {
      ConsumeToken();
    } else {
      Diag(Tok, diag::err_expected_semi_decl_list);
      // Skip to end of block or statement
      SkipUntil(tok::r_brace, true, true);
    }
  }
  SourceLocation RBraceLoc = MatchRHSPunctuation(tok::r_brace, LBraceLoc);
  // Call ActOnFields() even if we don't have any decls. This is useful
  // for code rewriting tools that need to be aware of the empty list.
  Actions.ActOnFields(CurScope, atLoc, interfaceDecl,
                      &AllIvarDecls[0], AllIvarDecls.size(),
                      LBraceLoc, RBraceLoc, 0);
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
Parser::DeclTy *Parser::ParseObjCAtProtocolDeclaration(SourceLocation AtLoc,
                                               AttributeList *attrList) {
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
  
  if (Tok.is(tok::semi)) { // forward declaration of one protocol.
    IdentifierLocPair ProtoInfo(protocolName, nameLoc);
    ConsumeToken();
    return Actions.ActOnForwardProtocolDeclaration(AtLoc, &ProtoInfo, 1);
  }
  
  if (Tok.is(tok::comma)) { // list of forward declarations.
    llvm::SmallVector<IdentifierLocPair, 8> ProtocolRefs;
    ProtocolRefs.push_back(std::make_pair(protocolName, nameLoc));

    // Parse the list of forward declarations.
    while (1) {
      ConsumeToken(); // the ','
      if (Tok.isNot(tok::identifier)) {
        Diag(Tok, diag::err_expected_ident);
        SkipUntil(tok::semi);
        return 0;
      }
      ProtocolRefs.push_back(IdentifierLocPair(Tok.getIdentifierInfo(),
                                               Tok.getLocation()));
      ConsumeToken(); // the identifier
      
      if (Tok.isNot(tok::comma))
        break;
    }
    // Consume the ';'.
    if (ExpectAndConsume(tok::semi, diag::err_expected_semi_after, "@protocol"))
      return 0;
    
    return Actions.ActOnForwardProtocolDeclaration(AtLoc,
                                                   &ProtocolRefs[0], 
                                                   ProtocolRefs.size());
  }
  
  // Last, and definitely not least, parse a protocol declaration.
  SourceLocation EndProtoLoc;

  llvm::SmallVector<DeclTy *, 8> ProtocolRefs;
  if (Tok.is(tok::less) &&
      ParseObjCProtocolReferences(ProtocolRefs, true, EndProtoLoc))
    return 0;
  
  DeclTy *ProtoType =
    Actions.ActOnStartProtocolInterface(AtLoc, protocolName, nameLoc,
                                        &ProtocolRefs[0], ProtocolRefs.size(),
                                        EndProtoLoc, attrList);
  ParseObjCInterfaceDeclList(ProtoType, tok::objc_protocol);
  return ProtoType;
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
    DeclTy *ImplCatType = Actions.ActOnStartCategoryImplementation(
                                    atLoc, nameId, nameLoc, categoryId, 
                                    categoryLoc);
    ObjCImpDecl = ImplCatType;
    return 0;
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
  DeclTy *ImplClsType = Actions.ActOnStartClassImplementation(
                                  atLoc, nameId, nameLoc,
                                  superClassId, superClassLoc);
  
  if (Tok.is(tok::l_brace)) // we have ivars
    ParseObjCClassInstanceVariables(ImplClsType/*FIXME*/, atLoc);
  ObjCImpDecl = ImplClsType;
  
  return 0;
}

Parser::DeclTy *Parser::ParseObjCAtEndDeclaration(SourceLocation atLoc) {
  assert(Tok.isObjCAtKeyword(tok::objc_end) &&
         "ParseObjCAtEndDeclaration(): Expected @end");
  ConsumeToken(); // the "end" identifier
  if (ObjCImpDecl)
    Actions.ActOnAtEnd(atLoc, ObjCImpDecl);
  else
    Diag(atLoc, diag::warn_expected_implementation); // missing @implementation
  return ObjCImpDecl;
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
  IdentifierInfo *aliasId = Tok.getIdentifierInfo();
  SourceLocation aliasLoc = ConsumeToken(); // consume alias-name
  if (Tok.isNot(tok::identifier)) {
    Diag(Tok, diag::err_expected_ident);
    return 0;
  }
  IdentifierInfo *classId = Tok.getIdentifierInfo();
  SourceLocation classLoc = ConsumeToken(); // consume class-name;
  if (Tok.isNot(tok::semi)) {
    Diag(Tok, diag::err_expected_semi_after, "@compatibility_alias");
    return 0;
  }
  DeclTy *ClsType = Actions.ActOnCompatiblityAlias(atLoc, 
                                                   aliasId, aliasLoc,
                                                   classId, classLoc);
  return ClsType;
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
  SourceLocation loc = ConsumeToken(); // consume synthesize
  if (Tok.isNot(tok::identifier)) {
    Diag(Tok, diag::err_expected_ident);
    return 0;
  }
  while (Tok.is(tok::identifier)) {
    IdentifierInfo *propertyIvar = 0;
    IdentifierInfo *propertyId = Tok.getIdentifierInfo();
    SourceLocation propertyLoc = ConsumeToken(); // consume property name
    if (Tok.is(tok::equal)) {
      // property '=' ivar-name
      ConsumeToken(); // consume '='
      if (Tok.isNot(tok::identifier)) {
        Diag(Tok, diag::err_expected_ident);
        break;
      }
      propertyIvar = Tok.getIdentifierInfo();
      ConsumeToken(); // consume ivar-name
    }
    Actions.ActOnPropertyImplDecl(atLoc, propertyLoc, true, ObjCImpDecl,
                                  propertyId, propertyIvar);
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
    IdentifierInfo *propertyId = Tok.getIdentifierInfo();
    SourceLocation propertyLoc = ConsumeToken(); // consume property name
    Actions.ActOnPropertyImplDecl(atLoc, propertyLoc, false, ObjCImpDecl,
                                  propertyId, 0);

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
Parser::StmtResult Parser::ParseObjCThrowStmt(SourceLocation atLoc) {
  ExprResult Res;
  ConsumeToken(); // consume throw
  if (Tok.isNot(tok::semi)) {
    Res = ParseExpression();
    if (Res.isInvalid) {
      SkipUntil(tok::semi);
      return true;
    }
  }
  ConsumeToken(); // consume ';'
  return Actions.ActOnObjCAtThrowStmt(atLoc, Res.Val);
}

/// objc-synchronized-statement:
///   @synchronized '(' expression ')' compound-statement
///
Parser::StmtResult Parser::ParseObjCSynchronizedStmt(SourceLocation atLoc) {
  ConsumeToken(); // consume synchronized
  if (Tok.isNot(tok::l_paren)) {
    Diag (Tok, diag::err_expected_lparen_after, "@synchronized");
    return true;
  }
  ConsumeParen();  // '('
  ExprResult Res = ParseExpression();
  if (Res.isInvalid) {
    SkipUntil(tok::semi);
    return true;
  }
  if (Tok.isNot(tok::r_paren)) {
    Diag (Tok, diag::err_expected_lbrace);
    return true;
  }
  ConsumeParen();  // ')'
  if (Tok.isNot(tok::l_brace)) {
    Diag (Tok, diag::err_expected_lbrace);
    return true;
  }
  // Enter a scope to hold everything within the compound stmt.  Compound
  // statements can always hold declarations.
  EnterScope(Scope::DeclScope);

  StmtResult SynchBody = ParseCompoundStatementBody();
  
  ExitScope();
  if (SynchBody.isInvalid)
    SynchBody = Actions.ActOnNullStmt(Tok.getLocation());
  return Actions.ActOnObjCAtSynchronizedStmt(atLoc, Res.Val, SynchBody.Val);
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
Parser::StmtResult Parser::ParseObjCTryStmt(SourceLocation atLoc) {
  bool catch_or_finally_seen = false;
  
  ConsumeToken(); // consume try
  if (Tok.isNot(tok::l_brace)) {
    Diag (Tok, diag::err_expected_lbrace);
    return true;
  }
  StmtResult CatchStmts;
  StmtResult FinallyStmt;
  EnterScope(Scope::DeclScope);
  StmtResult TryBody = ParseCompoundStatementBody();
  ExitScope();
  if (TryBody.isInvalid)
    TryBody = Actions.ActOnNullStmt(Tok.getLocation());
  
  while (Tok.is(tok::at)) {
    // At this point, we need to lookahead to determine if this @ is the start
    // of an @catch or @finally.  We don't want to consume the @ token if this
    // is an @try or @encode or something else.
    Token AfterAt = GetLookAheadToken(1);
    if (!AfterAt.isObjCAtKeyword(tok::objc_catch) &&
        !AfterAt.isObjCAtKeyword(tok::objc_finally))
      break;
      
    SourceLocation AtCatchFinallyLoc = ConsumeToken();
    if (Tok.isObjCAtKeyword(tok::objc_catch)) {
      StmtTy *FirstPart = 0;
      ConsumeToken(); // consume catch
      if (Tok.is(tok::l_paren)) {
        ConsumeParen();
        EnterScope(Scope::DeclScope);
        if (Tok.isNot(tok::ellipsis)) {
          DeclSpec DS;
          ParseDeclarationSpecifiers(DS);
          // For some odd reason, the name of the exception variable is 
          // optional. As a result, we need to use PrototypeContext.
          Declarator DeclaratorInfo(DS, Declarator::PrototypeContext);
          ParseDeclarator(DeclaratorInfo);
          if (DeclaratorInfo.getIdentifier()) {
            DeclTy *aBlockVarDecl = Actions.ActOnDeclarator(CurScope, 
                                                          DeclaratorInfo, 0);
            StmtResult stmtResult =
              Actions.ActOnDeclStmt(aBlockVarDecl, 
                                    DS.getSourceRange().getBegin(),
                                    DeclaratorInfo.getSourceRange().getEnd());
            FirstPart = stmtResult.isInvalid ? 0 : stmtResult.Val;
          }
        } else
          ConsumeToken(); // consume '...'
        SourceLocation RParenLoc = ConsumeParen();
        
        StmtResult CatchBody(true);
        if (Tok.is(tok::l_brace))
          CatchBody = ParseCompoundStatementBody();
        else
          Diag(Tok, diag::err_expected_lbrace);
        if (CatchBody.isInvalid)
          CatchBody = Actions.ActOnNullStmt(Tok.getLocation());
        CatchStmts = Actions.ActOnObjCAtCatchStmt(AtCatchFinallyLoc, RParenLoc, 
          FirstPart, CatchBody.Val, CatchStmts.Val);
        ExitScope();
      } else {
        Diag(AtCatchFinallyLoc, diag::err_expected_lparen_after, 
             "@catch clause");
        return true;
      }
      catch_or_finally_seen = true;
    } else {
      assert(Tok.isObjCAtKeyword(tok::objc_finally) && "Lookahead confused?");
      ConsumeToken(); // consume finally
      EnterScope(Scope::DeclScope);

      
      StmtResult FinallyBody(true);
      if (Tok.is(tok::l_brace))
        FinallyBody = ParseCompoundStatementBody();
      else
        Diag(Tok, diag::err_expected_lbrace);
      if (FinallyBody.isInvalid)
        FinallyBody = Actions.ActOnNullStmt(Tok.getLocation());
      FinallyStmt = Actions.ActOnObjCAtFinallyStmt(AtCatchFinallyLoc, 
                                                   FinallyBody.Val);
      catch_or_finally_seen = true;
      ExitScope();
      break;
    }
  }
  if (!catch_or_finally_seen) {
    Diag(atLoc, diag::err_missing_catch_finally);
    return true;
  }
  return Actions.ActOnObjCAtTryStmt(atLoc, TryBody.Val, CatchStmts.Val, 
                                    FinallyStmt.Val);
}

///   objc-method-def: objc-method-proto ';'[opt] '{' body '}'
///
Parser::DeclTy *Parser::ParseObjCMethodDefinition() {
  DeclTy *MDecl = ParseObjCMethodPrototype(ObjCImpDecl);
  // parse optional ';'
  if (Tok.is(tok::semi))
    ConsumeToken();

  // We should have an opening brace now.
  if (Tok.isNot(tok::l_brace)) {
    Diag(Tok, diag::err_expected_method_body);
    
    // Skip over garbage, until we get to '{'.  Don't eat the '{'.
    SkipUntil(tok::l_brace, true, true);
    
    // If we didn't find the '{', bail out.
    if (Tok.isNot(tok::l_brace))
      return 0;
  }
  SourceLocation BraceLoc = Tok.getLocation();
  
  // Enter a scope for the method body.
  EnterScope(Scope::FnScope|Scope::DeclScope);
  
  // Tell the actions module that we have entered a method definition with the
  // specified Declarator for the method.
  Actions.ObjCActOnStartOfMethodDef(CurScope, MDecl);
  
  StmtResult FnBody = ParseCompoundStatementBody();
  
  // If the function body could not be parsed, make a bogus compoundstmt.
  if (FnBody.isInvalid)
    FnBody = Actions.ActOnCompoundStmt(BraceLoc, BraceLoc, 0, 0, false);
  
  // Leave the function body scope.
  ExitScope();
  
  // TODO: Pass argument information.
  Actions.ActOnFinishFunctionBody(MDecl, FnBody.Val);
  return MDecl;
}

Parser::StmtResult Parser::ParseObjCAtStatement(SourceLocation AtLoc) {
  if (Tok.isObjCAtKeyword(tok::objc_try)) {
    return ParseObjCTryStmt(AtLoc);
  } else if (Tok.isObjCAtKeyword(tok::objc_throw))
    return ParseObjCThrowStmt(AtLoc);
  else if (Tok.isObjCAtKeyword(tok::objc_synchronized))
    return ParseObjCSynchronizedStmt(AtLoc);
  ExprResult Res = ParseExpressionWithLeadingAt(AtLoc);
  if (Res.isInvalid) {
    // If the expression is invalid, skip ahead to the next semicolon. Not
    // doing this opens us up to the possibility of infinite loops if
    // ParseExpression does not consume any tokens.
    SkipUntil(tok::semi);
    return true;
  }
  // Otherwise, eat the semicolon.
  ExpectAndConsume(tok::semi, diag::err_expected_semi_after_expr);
  return Actions.ActOnExprStmt(Res.Val);
}

Parser::ExprResult Parser::ParseObjCAtExpression(SourceLocation AtLoc) {
  switch (Tok.getKind()) {
  case tok::string_literal:    // primary-expression: string-literal
  case tok::wide_string_literal:
    return ParsePostfixExpressionSuffix(ParseObjCStringLiteral(AtLoc));
  default:
    if (Tok.getIdentifierInfo() == 0)
      return Diag(AtLoc, diag::err_unexpected_at);
      
    switch (Tok.getIdentifierInfo()->getObjCKeywordID()) {
    case tok::objc_encode:
      return ParsePostfixExpressionSuffix(ParseObjCEncodeExpression(AtLoc));
    case tok::objc_protocol:
      return ParsePostfixExpressionSuffix(ParseObjCProtocolExpression(AtLoc));
    case tok::objc_selector:
      return ParsePostfixExpressionSuffix(ParseObjCSelectorExpression(AtLoc));
    default:
      return Diag(AtLoc, diag::err_unexpected_at);
    }
  }
}

///   objc-message-expr: 
///     '[' objc-receiver objc-message-args ']'
///
///   objc-receiver:
///     expression
///     class-name
///     type-name
Parser::ExprResult Parser::ParseObjCMessageExpression() {
  assert(Tok.is(tok::l_square) && "'[' expected");
  SourceLocation LBracLoc = ConsumeBracket(); // consume '['

  // Parse receiver
  if (isTokObjCMessageIdentifierReceiver()) {
    IdentifierInfo *ReceiverName = Tok.getIdentifierInfo();
    ConsumeToken();
    return ParseObjCMessageExpressionBody(LBracLoc, ReceiverName, 0);
  }

  ExprResult Res = ParseExpression();
  if (Res.isInvalid) {
    SkipUntil(tok::r_square);
    return Res;
  }
  
  return ParseObjCMessageExpressionBody(LBracLoc, 0, Res.Val);
}
  
/// ParseObjCMessageExpressionBody - Having parsed "'[' objc-receiver", parse
/// the rest of a message expression.
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
Parser::ExprResult
Parser::ParseObjCMessageExpressionBody(SourceLocation LBracLoc,
                                       IdentifierInfo *ReceiverName,
                                       ExprTy *ReceiverExpr) {
  // Parse objc-selector
  SourceLocation Loc;
  IdentifierInfo *selIdent = ParseObjCSelector(Loc);

  llvm::SmallVector<IdentifierInfo *, 12> KeyIdents;
  llvm::SmallVector<Action::ExprTy *, 12> KeyExprs;

  if (Tok.is(tok::colon)) {
    while (1) {
      // Each iteration parses a single keyword argument.
      KeyIdents.push_back(selIdent);

      if (Tok.isNot(tok::colon)) {
        Diag(Tok, diag::err_expected_colon);
        // We must manually skip to a ']', otherwise the expression skipper will
        // stop at the ']' when it skips to the ';'.  We want it to skip beyond
        // the enclosing expression.
        SkipUntil(tok::r_square);
        return true;
      }
      
      ConsumeToken(); // Eat the ':'.
      ///  Parse the expression after ':' 
      ExprResult Res = ParseAssignmentExpression();
      if (Res.isInvalid) {
        // We must manually skip to a ']', otherwise the expression skipper will
        // stop at the ']' when it skips to the ';'.  We want it to skip beyond
        // the enclosing expression.
        SkipUntil(tok::r_square);
        return Res;
      }
      
      // We have a valid expression.
      KeyExprs.push_back(Res.Val);
      
      // Check for another keyword selector.
      selIdent = ParseObjCSelector(Loc);
      if (!selIdent && Tok.isNot(tok::colon))
        break;
      // We have a selector or a colon, continue parsing.
    }
    // Parse the, optional, argument list, comma separated.
    while (Tok.is(tok::comma)) {
      ConsumeToken(); // Eat the ','.
      ///  Parse the expression after ',' 
      ExprResult Res = ParseAssignmentExpression();
      if (Res.isInvalid) {
        // We must manually skip to a ']', otherwise the expression skipper will
        // stop at the ']' when it skips to the ';'.  We want it to skip beyond
        // the enclosing expression.
        SkipUntil(tok::r_square);
        return Res;
      }
      
      // We have a valid expression.
      KeyExprs.push_back(Res.Val);
    }
  } else if (!selIdent) {
    Diag(Tok, diag::err_expected_ident); // missing selector name.
    
    // We must manually skip to a ']', otherwise the expression skipper will
    // stop at the ']' when it skips to the ';'.  We want it to skip beyond
    // the enclosing expression.
    SkipUntil(tok::r_square);
    return true;
  }
  
  if (Tok.isNot(tok::r_square)) {
    Diag(Tok, diag::err_expected_rsquare);
    // We must manually skip to a ']', otherwise the expression skipper will
    // stop at the ']' when it skips to the ';'.  We want it to skip beyond
    // the enclosing expression.
    SkipUntil(tok::r_square);
    return true;
  }
  
  SourceLocation RBracLoc = ConsumeBracket(); // consume ']'
  
  unsigned nKeys = KeyIdents.size();
  if (nKeys == 0)
    KeyIdents.push_back(selIdent);
  Selector Sel = PP.getSelectorTable().getSelector(nKeys, &KeyIdents[0]);
  
  // We've just parsed a keyword message.
  if (ReceiverName) 
    return Actions.ActOnClassMessage(CurScope,
                                     ReceiverName, Sel, LBracLoc, RBracLoc,
                                     &KeyExprs[0], KeyExprs.size());
  return Actions.ActOnInstanceMessage(ReceiverExpr, Sel, LBracLoc, RBracLoc,
                                      &KeyExprs[0], KeyExprs.size());
}

Parser::ExprResult Parser::ParseObjCStringLiteral(SourceLocation AtLoc) {
  ExprResult Res = ParseStringLiteralExpression();
  if (Res.isInvalid) return Res;
  
  // @"foo" @"bar" is a valid concatenated string.  Eat any subsequent string
  // expressions.  At this point, we know that the only valid thing that starts
  // with '@' is an @"".
  llvm::SmallVector<SourceLocation, 4> AtLocs;
  llvm::SmallVector<ExprTy*, 4> AtStrings;
  AtLocs.push_back(AtLoc);
  AtStrings.push_back(Res.Val);
  
  while (Tok.is(tok::at)) {
    AtLocs.push_back(ConsumeToken()); // eat the @.

    ExprResult Res(true);  // Invalid unless there is a string literal.
    if (isTokenStringLiteral())
      Res = ParseStringLiteralExpression();
    else
      Diag(Tok, diag::err_objc_concat_string);
    
    if (Res.isInvalid) {
      while (!AtStrings.empty()) {
        Actions.DeleteExpr(AtStrings.back());
        AtStrings.pop_back();
      }
      return Res;
    }

    AtStrings.push_back(Res.Val);
  }
  
  return Actions.ParseObjCStringLiteral(&AtLocs[0], &AtStrings[0],
                                        AtStrings.size());
}

///    objc-encode-expression:
///      @encode ( type-name )
Parser::ExprResult Parser::ParseObjCEncodeExpression(SourceLocation AtLoc) {
  assert(Tok.isObjCAtKeyword(tok::objc_encode) && "Not an @encode expression!");
  
  SourceLocation EncLoc = ConsumeToken();
  
  if (Tok.isNot(tok::l_paren))
    return Diag(Tok, diag::err_expected_lparen_after, "@encode");
   
  SourceLocation LParenLoc = ConsumeParen();
  
  TypeTy *Ty = ParseTypeName();
  
  SourceLocation RParenLoc = MatchRHSPunctuation(tok::r_paren, LParenLoc);
   
  return Actions.ParseObjCEncodeExpression(AtLoc, EncLoc, LParenLoc, Ty, 
                                           RParenLoc);
}

///     objc-protocol-expression
///       @protocol ( protocol-name )

Parser::ExprResult Parser::ParseObjCProtocolExpression(SourceLocation AtLoc)
{
  SourceLocation ProtoLoc = ConsumeToken();
  
  if (Tok.isNot(tok::l_paren))
    return Diag(Tok, diag::err_expected_lparen_after, "@protocol");
  
  SourceLocation LParenLoc = ConsumeParen();
  
  if (Tok.isNot(tok::identifier))
    return Diag(Tok, diag::err_expected_ident);
  
  IdentifierInfo *protocolId = Tok.getIdentifierInfo();
  ConsumeToken();
  
  SourceLocation RParenLoc = MatchRHSPunctuation(tok::r_paren, LParenLoc);

  return Actions.ParseObjCProtocolExpression(protocolId, AtLoc, ProtoLoc, 
                                             LParenLoc, RParenLoc);
}

///     objc-selector-expression
///       @selector '(' objc-keyword-selector ')'
Parser::ExprResult Parser::ParseObjCSelectorExpression(SourceLocation AtLoc)
{
  SourceLocation SelectorLoc = ConsumeToken();
  
  if (Tok.isNot(tok::l_paren))
    return Diag(Tok, diag::err_expected_lparen_after, "@selector");
  
  llvm::SmallVector<IdentifierInfo *, 12> KeyIdents;
  SourceLocation LParenLoc = ConsumeParen();
  SourceLocation sLoc;
  IdentifierInfo *SelIdent = ParseObjCSelector(sLoc);
  if (!SelIdent && Tok.isNot(tok::colon))
    return Diag(Tok, diag::err_expected_ident); // missing selector name.
  
  KeyIdents.push_back(SelIdent);
  unsigned nColons = 0;
  if (Tok.isNot(tok::r_paren)) {
    while (1) {
      if (Tok.isNot(tok::colon))
        return Diag(Tok, diag::err_expected_colon);
      
      nColons++;
      ConsumeToken(); // Eat the ':'.
      if (Tok.is(tok::r_paren))
        break;
      // Check for another keyword selector.
      SourceLocation Loc;
      SelIdent = ParseObjCSelector(Loc);
      KeyIdents.push_back(SelIdent);
      if (!SelIdent && Tok.isNot(tok::colon))
        break;
    }
  }
  SourceLocation RParenLoc = MatchRHSPunctuation(tok::r_paren, LParenLoc);
  Selector Sel = PP.getSelectorTable().getSelector(nColons, &KeyIdents[0]);
  return Actions.ParseObjCSelectorExpression(Sel, AtLoc, SelectorLoc, LParenLoc, 
                                             RParenLoc);
 }
