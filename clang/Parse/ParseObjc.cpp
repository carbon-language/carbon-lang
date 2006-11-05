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
using namespace llvm;
using namespace clang;


/// ParseExternalDeclaration:
///       external-declaration: [C99 6.9]
/// [OBJC]  objc-class-definition
/// [OBJC]  objc-class-declaration     [TODO]
/// [OBJC]  objc-alias-declaration     [TODO]
/// [OBJC]  objc-protocol-definition   [TODO]
/// [OBJC]  objc-method-definition     [TODO]
/// [OBJC]  '@' 'end'                  [TODO]
void Parser::ObjCParseAtDirectives() {
  SourceLocation AtLoc = ConsumeToken(); // the "@"
  
  IdentifierInfo *II = Tok.getIdentifierInfo();
  switch (II ? II->getObjCKeywordID() : tok::objc_not_keyword) {
    case tok::objc_class:
      return ObjCParseAtClassDeclaration(AtLoc);
    case tok::objc_interface:
      return ObjCParseAtInterfaceDeclaration();
    case tok::objc_protocol:
      return ObjCParseAtProtocolDeclaration();
    case tok::objc_implementation:
      return ObjCParseAtImplementationDeclaration();
    case tok::objc_end:
      return ObjCParseAtEndDeclaration();
    case tok::objc_compatibility_alias:
      return ObjCParseAtAliasDeclaration();
    default:
      Diag(AtLoc, diag::err_unexpected_at);
      SkipUntil(tok::semi);
  }
}

///
/// objc-class-declaration: 
///    '@' 'class' identifier-list ';'
///  
void Parser::ObjCParseAtClassDeclaration(SourceLocation atLoc) {
  ConsumeToken(); // the identifier "class"
  SmallVector<IdentifierInfo *, 8> ClassNames;
  
  while (1) {
    if (Tok.getKind() != tok::identifier) {
      Diag(diag::err_expected_ident);
      SkipUntil(tok::semi);
      return;
    }
    
    ClassNames.push_back(Tok.getIdentifierInfo());
    ConsumeToken();
    
    if (Tok.getKind() != tok::comma)
      break;
    
    ConsumeToken();
  }
  
  // Consume the ';'.
  if (ExpectAndConsume(tok::semi, diag::err_expected_semi_after, "@class"))
    return;
  
  Actions.ParsedClassDeclaration(CurScope, &ClassNames[0], ClassNames.size());
}

void Parser::ObjCParseAtInterfaceDeclaration() {
  assert(0 && "Unimp");
}
void Parser::ObjCParseAtProtocolDeclaration() {
  assert(0 && "Unimp");
}
void Parser::ObjCParseAtImplementationDeclaration() {
  assert(0 && "Unimp");
}
void Parser::ObjCParseAtEndDeclaration() {
  assert(0 && "Unimp");
}
void Parser::ObjCParseAtAliasDeclaration() {
  assert(0 && "Unimp");
}

void Parser::ObjCParseInstanceMethodDeclaration() {
  assert(0 && "Unimp");
}

void Parser::ObjCParseClassMethodDeclaration() {
  assert(0 && "Unimp");
}
