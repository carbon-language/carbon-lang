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
  
  IdentifierInfo *II = Tok.getIdentifierInfo();
  switch (II ? II->getObjCKeywordID() : tok::objc_not_keyword) {
    case tok::objc_class:
      return ParseObjCAtClassDeclaration(AtLoc);
    case tok::objc_interface:
      return ParseObjCAtInterfaceDeclaration(AtLoc);
    case tok::objc_protocol:
      return ParseObjCAtProtocolDeclaration();
    case tok::objc_implementation:
      return ParseObjCAtImplementationDeclaration();
    case tok::objc_end:
      return ParseObjCAtEndDeclaration();
    case tok::objc_compatibility_alias:
      return ParseObjCAtAliasDeclaration();
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
    if (Tok.getKind() != tok::identifier) {
      Diag(Tok, diag::err_expected_ident);
      SkipUntil(tok::semi);
      return 0;
    }
    
    ClassNames.push_back(Tok.getIdentifierInfo());
    ConsumeToken();
    
    if (Tok.getKind() != tok::comma)
      break;
    
    ConsumeToken();
  }
  
  // Consume the ';'.
  if (ExpectAndConsume(tok::semi, diag::err_expected_semi_after, "@class"))
    return 0;
  
  return Actions.ParsedObjcClassDeclaration(CurScope,
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
  assert(Tok.getIdentifierInfo()->getObjCKeywordID() == tok::objc_interface &&
         "ParseObjCAtInterfaceDeclaration(): Expected @interface");
  ConsumeToken(); // the "interface" identifier
  
  if (Tok.getKind() != tok::identifier) {
    Diag(Tok, diag::err_expected_ident); // missing class or category name.
    return 0;
  }
  // We have a class or category name - consume it.
  IdentifierInfo *nameId = Tok.getIdentifierInfo();
  SourceLocation nameLoc = ConsumeToken();
  
  if (Tok.getKind() == tok::l_paren) { // we have a category
    SourceLocation lparenLoc = ConsumeParen();
    SourceLocation categoryLoc, rparenLoc;
    IdentifierInfo *categoryId = 0;
    
    // OBJC2: The cateogry name is optional (not an error).
    if (Tok.getKind() == tok::identifier) {
      categoryId = Tok.getIdentifierInfo();
      categoryLoc = ConsumeToken();
    }
    if (Tok.getKind() != tok::r_paren) {
      Diag(Tok, diag::err_expected_rparen);
      SkipUntil(tok::r_paren, false); // don't stop at ';'
      return 0;
    }
    rparenLoc = ConsumeParen();
    // Next, we need to check for any protocol references.
    if (Tok.getKind() == tok::less) {
      if (ParseObjCProtocolReferences())
        return 0;
    }
    if (attrList) // categories don't support attributes.
      Diag(Tok, diag::err_objc_no_attributes_on_category);
    
    //ParseObjCInterfaceDeclList();

    if (Tok.getKind() != tok::at) { // check for @end
      Diag(Tok, diag::err_objc_missing_end);
      return 0;
    }
    SourceLocation atEndLoc = ConsumeToken(); // eat the @ sign
    if (Tok.getIdentifierInfo()->getObjCKeywordID() != tok::objc_end) {
      Diag(Tok, diag::err_objc_missing_end);
      return 0;
    }
    ConsumeToken(); // the "end" identifier
    return 0;
  }
  // Parse a class interface.
  IdentifierInfo *superClassId = 0;
  SourceLocation superClassLoc;
  
  if (Tok.getKind() == tok::colon) { // a super class is specified.
    ConsumeToken();
    if (Tok.getKind() != tok::identifier) {
      Diag(Tok, diag::err_expected_ident); // missing super class name.
      return 0;
    }
    superClassId = Tok.getIdentifierInfo();
    superClassLoc = ConsumeToken();
  }
  // Next, we need to check for any protocol references.
  if (Tok.getKind() == tok::less) {
    if (ParseObjCProtocolReferences())
      return 0;
  }
  // FIXME: add Actions.StartObjCClassInterface(nameId, superClassId, ...)
  if (Tok.getKind() == tok::l_brace)
    ParseObjCClassInstanceVariables(0/*FIXME*/);

  //ParseObjCInterfaceDeclList();
  
  if (Tok.getKind() != tok::at) { // check for @end
    Diag(Tok, diag::err_objc_missing_end);
    return 0;
  }
  SourceLocation atEndLoc = ConsumeToken(); // eat the @ sign
  if (Tok.getIdentifierInfo()->getObjCKeywordID() != tok::objc_end) {
    Diag(Tok, diag::err_objc_missing_end);
    return 0;
  }
  ConsumeToken(); // the "end" identifier
  return 0;
}

///   objc-interface-decl-list:
///     empty
///     objc-interface-decl-list objc-method-proto
///     objc-interface-decl-list objc-property-decl [OBJC2]
///     objc-interface-decl-list declaration
///     objc-interface-decl-list ';'
///
void Parser::ParseObjCInterfaceDeclList() {
  assert(0 && "Unimp");
}

///   objc-protocol-refs:
///     '<' identifier-list '>'
///
bool Parser::ParseObjCProtocolReferences() {
  assert(Tok.getKind() == tok::less && "expected <");
  
  ConsumeToken(); // the "<"
  llvm::SmallVector<IdentifierInfo *, 8> ProtocolRefs;
  
  while (1) {
    if (Tok.getKind() != tok::identifier) {
      Diag(Tok, diag::err_expected_ident);
      SkipUntil(tok::greater);
      return true;
    }
    ProtocolRefs.push_back(Tok.getIdentifierInfo());
    ConsumeToken();
    
    if (Tok.getKind() != tok::comma)
      break;
    ConsumeToken();
  }
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
  assert(Tok.getKind() == tok::l_brace && "expected {");
  
  SourceLocation LBraceLoc = ConsumeBrace(); // the "{"
  llvm::SmallVector<DeclTy*, 32> IvarDecls;
  
  // While we still have something to read, read the instance variables.
  while (Tok.getKind() != tok::r_brace && 
         Tok.getKind() != tok::eof) {
    // Each iteration of this loop reads one objc-instance-variable-decl.
    
    // Check for extraneous top-level semicolon.
    if (Tok.getKind() == tok::semi) {
      Diag(Tok, diag::ext_extra_struct_semi);
      ConsumeToken();
      continue;
    }
    // Set the default visibility to private.
    tok::ObjCKeywordKind visibility = tok::objc_private;
    if (Tok.getKind() == tok::at) { // parse objc-visibility-spec
      ConsumeToken(); // eat the @ sign
      IdentifierInfo *specId = Tok.getIdentifierInfo();
      switch (specId->getObjCKeywordID()) {
      case tok::objc_private:
      case tok::objc_public:
      case tok::objc_protected:
      case tok::objc_package:
        visibility = specId->getObjCKeywordID();
        ConsumeToken();
        continue; 
      default:
        Diag(Tok, diag::err_objc_illegal_visibility_spec);
        ConsumeToken();
        continue;
      }
    }
    ParseStructDeclaration(interfaceDecl, IvarDecls);

    if (Tok.getKind() == tok::semi) {
      ConsumeToken();
    } else if (Tok.getKind() == tok::r_brace) {
      Diag(Tok.getLocation(), diag::ext_expected_semi_decl_list);
      break;
    } else {
      Diag(Tok, diag::err_expected_semi_decl_list);
      // Skip to end of block or statement
      SkipUntil(tok::r_brace, true, true);
    }
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
///       objc-methodprotolist 
///     @end
///
///   objc-protocol-forward-reference:
///     @protocol identifier-list ';'
///
///   "@protocol identifier ;" should be resolved as "@protocol
///   identifier-list ;": objc-methodprotolist may not start with a
///   semicolon in the first alternative if objc-protocol-refs are omitted.

Parser::DeclTy *Parser::ParseObjCAtProtocolDeclaration() {
  assert(0 && "Unimp");
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

Parser::DeclTy *Parser::ParseObjCAtImplementationDeclaration() {
  assert(0 && "Unimp");
  return 0;
}
Parser::DeclTy *Parser::ParseObjCAtEndDeclaration() {
  assert(0 && "Unimp");
  return 0;
}
Parser::DeclTy *Parser::ParseObjCAtAliasDeclaration() {
  assert(0 && "Unimp");
  return 0;
}

void Parser::ParseObjCInstanceMethodDeclaration() {
  assert(0 && "Unimp");
}

void Parser::ParseObjCClassMethodDeclaration() {
  assert(0 && "Unimp");
}

Parser::ExprResult Parser::ParseObjCExpression() {
  SourceLocation AtLoc = ConsumeToken(); // the "@"

  switch (Tok.getKind()) {
    case tok::string_literal:    // primary-expression: string-literal
    case tok::wide_string_literal:
      return ParseObjCStringLiteral();
    case tok::objc_encode:
      return ParseObjCEncodeExpression();
      break;
    default:
      Diag(AtLoc, diag::err_unexpected_at);
      SkipUntil(tok::semi);
      break;
  }
  
  return 0;
}

Parser::ExprResult Parser::ParseObjCStringLiteral() {
  ExprResult Res = ParseStringLiteralExpression();

  if (Res.isInvalid) return Res;

  return Actions.ParseObjCStringLiteral(Res.Val);
}

///    objc-encode-expression:
///      @encode ( type-name )
Parser::ExprResult Parser::ParseObjCEncodeExpression() {
  assert(Tok.getIdentifierInfo()->getObjCKeywordID() == tok::objc_encode && 
         "Not an @encode expression!");
  
  SourceLocation EncLoc = ConsumeToken();
  
  if (Tok.getKind() != tok::l_paren) {
    Diag(Tok, diag::err_expected_lparen_after, "@encode");
    return true;
  }
   
  SourceLocation LParenLoc = ConsumeParen();
  
  TypeTy *Ty = ParseTypeName();
  
  if (Tok.getKind() != tok::r_paren) {
    Diag(Tok, diag::err_expected_rparen);
    return true;
  }
   
  return Actions.ParseObjCEncodeExpression(EncLoc, LParenLoc, Ty, 
                                           ConsumeParen());
}
