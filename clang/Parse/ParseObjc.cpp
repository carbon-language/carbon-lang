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
void Parser::ParseObjCAtDirectives() {
  SourceLocation AtLoc = ConsumeToken(); // the "@"
  
  IdentifierInfo *II = Tok.getIdentifierInfo();
  switch (II ? II->getObjCKeywordID() : tok::objc_not_keyword) {
    case tok::objc_class:
      return ParseObjCAtClassDeclaration(AtLoc);
    case tok::objc_interface:
      return ParseObjCAtInterfaceDeclaration();
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
  }
}

///
/// objc-class-declaration: 
///    '@' 'class' identifier-list ';'
///  
void Parser::ParseObjCAtClassDeclaration(SourceLocation atLoc) {
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

void Parser::ParseObjCAtInterfaceDeclaration() {
  assert(0 && "Unimp");
}
void Parser::ParseObjCAtProtocolDeclaration() {
  assert(0 && "Unimp");
}
void Parser::ParseObjCAtImplementationDeclaration() {
  assert(0 && "Unimp");
}
void Parser::ParseObjCAtEndDeclaration() {
  assert(0 && "Unimp");
}
void Parser::ParseObjCAtAliasDeclaration() {
  assert(0 && "Unimp");
}

void Parser::ParseObjCInstanceMethodDeclaration() {
  assert(0 && "Unimp");
}

void Parser::ParseObjCClassMethodDeclaration() {
  assert(0 && "Unimp");
}
