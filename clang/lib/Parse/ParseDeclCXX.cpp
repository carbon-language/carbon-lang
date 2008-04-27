//===--- ParseDeclCXX.cpp - C++ Declaration Parsing -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the C++ Declaration portions of the Parser interfaces.
//
//===----------------------------------------------------------------------===//

#include "clang/Parse/Parser.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Parse/DeclSpec.h"
#include "clang/Parse/Scope.h"
using namespace clang;

/// ParseNamespace - We know that the current token is a namespace keyword. This
/// may either be a top level namespace or a block-level namespace alias.
///
///       namespace-definition: [C++ 7.3: basic.namespace]
///         named-namespace-definition
///         unnamed-namespace-definition
///
///       unnamed-namespace-definition:
///         'namespace' attributes[opt] '{' namespace-body '}'
///
///       named-namespace-definition:
///         original-namespace-definition
///         extension-namespace-definition
///
///       original-namespace-definition:
///         'namespace' identifier attributes[opt] '{' namespace-body '}'
///
///       extension-namespace-definition:
///         'namespace' original-namespace-name '{' namespace-body '}'
///  
///       namespace-alias-definition:  [C++ 7.3.2: namespace.alias]
///         'namespace' identifier '=' qualified-namespace-specifier ';'
///
Parser::DeclTy *Parser::ParseNamespace(unsigned Context) {
  assert(Tok.is(tok::kw_namespace) && "Not a namespace!");
  SourceLocation NamespaceLoc = ConsumeToken();  // eat the 'namespace'.
  
  SourceLocation IdentLoc;
  IdentifierInfo *Ident = 0;
  
  if (Tok.is(tok::identifier)) {
    Ident = Tok.getIdentifierInfo();
    IdentLoc = ConsumeToken();  // eat the identifier.
  }
  
  // Read label attributes, if present.
  DeclTy *AttrList = 0;
  if (Tok.is(tok::kw___attribute))
    // FIXME: save these somewhere.
    AttrList = ParseAttributes();
  
  if (Tok.is(tok::equal)) {
    // FIXME: Verify no attributes were present.
    // FIXME: parse this.
  } else if (Tok.is(tok::l_brace)) {

    SourceLocation LBrace = ConsumeBrace();

    // Enter a scope for the namespace.
    EnterScope(Scope::DeclScope);

    DeclTy *NamespcDecl =
      Actions.ActOnStartNamespaceDef(CurScope, IdentLoc, Ident, LBrace);

    while (Tok.isNot(tok::r_brace) && Tok.isNot(tok::eof))
      ParseExternalDeclaration();
    
    // Leave the namespace scope.
    ExitScope();

    SourceLocation RBrace = MatchRHSPunctuation(tok::r_brace, LBrace);
    Actions.ActOnFinishNamespaceDef(NamespcDecl, RBrace);

    return NamespcDecl;
    
  } else {
    unsigned D = Ident ? diag::err_expected_lbrace : 
                         diag::err_expected_ident_lbrace;
    Diag(Tok.getLocation(), D);
  }
  
  return 0;
}

/// ParseLinkage - We know that the current token is a string_literal
/// and just before that, that extern was seen.
///
///       linkage-specification: [C++ 7.5p2: dcl.link]
///         'extern' string-literal '{' declaration-seq[opt] '}'
///         'extern' string-literal declaration
///
Parser::DeclTy *Parser::ParseLinkage(unsigned Context) {
  assert(Tok.is(tok::string_literal) && "Not a stringliteral!");
  llvm::SmallVector<char, 8> LangBuffer;
  // LangBuffer is guaranteed to be big enough.
  LangBuffer.resize(Tok.getLength());
  const char *LangBufPtr = &LangBuffer[0];
  unsigned StrSize = PP.getSpelling(Tok, LangBufPtr);

  SourceLocation Loc = ConsumeStringToken();
  DeclTy *D = 0;
  SourceLocation LBrace, RBrace;
  
  if (Tok.isNot(tok::l_brace)) {
    D = ParseDeclaration(Context);
  } else {
    LBrace = ConsumeBrace();
    while (Tok.isNot(tok::r_brace) && Tok.isNot(tok::eof)) {
      // FIXME capture the decls.
      D = ParseExternalDeclaration();
    }

    RBrace = MatchRHSPunctuation(tok::r_brace, LBrace);
  }

  if (!D)
    return 0;

  return Actions.ActOnLinkageSpec(Loc, LBrace, RBrace, LangBufPtr, StrSize, D);
}

/// ParseClassSpecifier - Parse a C++ class-specifier [C++ class] or
/// elaborated-type-specifier [C++ dcl.type.elab]; we can't tell which
/// until we reach the start of a definition or see a token that
/// cannot start a definition.
///
///       class-specifier: [C++ class]
///         class-head '{' member-specification[opt] '}'
///         class-head '{' member-specification[opt] '}' attributes[opt]
///       class-head:
///         class-key identifier[opt] base-clause[opt]
///         class-key nested-name-specifier identifier base-clause[opt]
///         class-key nested-name-specifier[opt] simple-template-id
///                          base-clause[opt]
/// [GNU]   class-key attributes[opt] identifier[opt] base-clause[opt]
/// [GNU]   class-key attributes[opt] nested-name-specifier 
///                          identifier base-clause[opt]
/// [GNU]   class-key attributes[opt] nested-name-specifier[opt] 
///                          simple-template-id base-clause[opt]
///       class-key:
///         'class'
///         'struct'
///         'union'
///
///       elaborated-type-specifier: [C++ dcl.type.elab]
///         class-key ::[opt] nested-name-specifier[opt] identifier 
///         class-key ::[opt] nested-name-specifier[opt] 'template'[opt] 
///                          simple-template-id 
///
///  Note that the C++ class-specifier and elaborated-type-specifier,
///  together, subsume the C99 struct-or-union-specifier:
///
///       struct-or-union-specifier: [C99 6.7.2.1]
///         struct-or-union identifier[opt] '{' struct-contents '}'
///         struct-or-union identifier
/// [GNU]   struct-or-union attributes[opt] identifier[opt] '{' struct-contents
///                                                         '}' attributes[opt]
/// [GNU]   struct-or-union attributes[opt] identifier
///       struct-or-union:
///         'struct'
///         'union'
void Parser::ParseClassSpecifier(DeclSpec &DS) {
  assert((Tok.is(tok::kw_class) || 
          Tok.is(tok::kw_struct) || 
          Tok.is(tok::kw_union)) &&
         "Not a class specifier");
  DeclSpec::TST TagType =
    Tok.is(tok::kw_class) ? DeclSpec::TST_class : 
    Tok.is(tok::kw_struct) ? DeclSpec::TST_struct : 
    DeclSpec::TST_union;

  SourceLocation StartLoc = ConsumeToken();

  AttributeList *Attr = 0;
  // If attributes exist after tag, parse them.
  if (Tok.is(tok::kw___attribute))
    Attr = ParseAttributes();

  // FIXME: Parse the (optional) nested-name-specifier.

  // Parse the (optional) class name.
  // FIXME: Alternatively, parse a simple-template-id.
  IdentifierInfo *Name = 0;
  SourceLocation NameLoc;
  if (Tok.is(tok::identifier)) {
    Name = Tok.getIdentifierInfo();
    NameLoc = ConsumeToken();
  }

  // There are three options here.  If we have 'struct foo;', then
  // this is a forward declaration.  If we have 'struct foo {...' or
  // 'struct fo :...' then this is a definition. Otherwise we have
  // something like 'struct foo xyz', a reference.
  Action::TagKind TK;
  if (Tok.is(tok::l_brace) || (getLang().CPlusPlus && Tok.is(tok::colon)))
    TK = Action::TK_Definition;
  else if (Tok.is(tok::semi))
    TK = Action::TK_Declaration;
  else
    TK = Action::TK_Reference;

  if (!Name && TK != Action::TK_Definition) {
    // We have a declaration or reference to an anonymous class.
    Diag(StartLoc, diag::err_anon_type_definition, 
         DeclSpec::getSpecifierName(TagType));

    // Skip the rest of this declarator, up until the comma or semicolon.
    SkipUntil(tok::comma, true);
    return;
  }

  // Parse the tag portion of this.
  DeclTy *TagDecl = Actions.ActOnTag(CurScope, TagType, TK, StartLoc, Name, 
                                     NameLoc, Attr);

  // Parse the optional base clause (C++ only).
  if (getLang().CPlusPlus && Tok.is(tok::colon)) {
    ParseBaseClause(TagDecl);
  }

  // If there is a body, parse it and inform the actions module.
  if (Tok.is(tok::l_brace))
    ParseStructUnionBody(StartLoc, TagType, TagDecl);
  else if (TK == Action::TK_Definition) {
    // FIXME: Complain that we have a base-specifier list but no
    // definition.
    Diag(Tok.getLocation(), diag::err_expected_lbrace);
  }

  const char *PrevSpec = 0;
  if (DS.SetTypeSpecType(TagType, StartLoc, PrevSpec, TagDecl))
    Diag(StartLoc, diag::err_invalid_decl_spec_combination, PrevSpec);
}

/// ParseBaseClause - Parse the base-clause of a C++ class [C++ class.derived]. 
///
///       base-clause : [C++ class.derived]
///         ':' base-specifier-list
///       base-specifier-list:
///         base-specifier '...'[opt]
///         base-specifier-list ',' base-specifier '...'[opt]
void Parser::ParseBaseClause(DeclTy *ClassDecl)
{
  assert(Tok.is(tok::colon) && "Not a base clause");
  ConsumeToken();

  while (true) {
    // Parse a base-specifier.
    if (ParseBaseSpecifier(ClassDecl)) {
      // Skip the rest of this base specifier, up until the comma or
      // opening brace.
      SkipUntil(tok::comma, tok::l_brace);
    }

    // If the next token is a comma, consume it and keep reading
    // base-specifiers.
    if (Tok.isNot(tok::comma)) break;
    
    // Consume the comma.
    ConsumeToken();
  }
}

/// ParseBaseSpecifier - Parse a C++ base-specifier. A base-specifier is
/// one entry in the base class list of a class specifier, for example:
///    class foo : public bar, virtual private baz {
/// 'public bar' and 'virtual private baz' are each base-specifiers.
///
///       base-specifier: [C++ class.derived]
///         ::[opt] nested-name-specifier[opt] class-name
///         'virtual' access-specifier[opt] ::[opt] nested-name-specifier[opt]
///                        class-name
///         access-specifier 'virtual'[opt] ::[opt] nested-name-specifier[opt]
///                        class-name
bool Parser::ParseBaseSpecifier(DeclTy *ClassDecl)
{
  bool IsVirtual = false;
  SourceLocation StartLoc = Tok.getLocation();

  // Parse the 'virtual' keyword.
  if (Tok.is(tok::kw_virtual))  {
    ConsumeToken();
    IsVirtual = true;
  }

  // Parse an (optional) access specifier.
  AccessSpecifier Access = getAccessSpecifierIfPresent();
  if (Access)
    ConsumeToken();
  
  // Parse the 'virtual' keyword (again!), in case it came after the
  // access specifier.
  if (Tok.is(tok::kw_virtual))  {
    SourceLocation VirtualLoc = ConsumeToken();
    if (IsVirtual) {
      // Complain about duplicate 'virtual'
      Diag(VirtualLoc, diag::err_dup_virtual);
    }

    IsVirtual = true;
  }

  // FIXME: Parse optional '::' and optional nested-name-specifier.

  // Parse the class-name.
  // FIXME: Alternatively, parse a simple-template-id.
  if (Tok.isNot(tok::identifier)) {
    Diag(Tok.getLocation(), diag::err_expected_class_name);
    return true;
  }

  // We have an identifier; check whether it is actually a type.
  DeclTy *BaseType = Actions.isTypeName(*Tok.getIdentifierInfo(), CurScope);
  if (!BaseType) {
    Diag(Tok.getLocation(), diag::err_expected_class_name);
    return true;
  }

  // The location of the base class itself.
  SourceLocation BaseLoc = Tok.getLocation();
  
  // Find the complete source range for the base-specifier.  
  SourceRange Range(StartLoc, BaseLoc);
  
  // Consume the identifier token (finally!).
  ConsumeToken();
  
  // Notify semantic analysis that we have parsed a complete
  // base-specifier.
  Actions.ActOnBaseSpecifier(ClassDecl, Range, IsVirtual, Access, BaseType,
                             BaseLoc);
  return false;
}

/// getAccessSpecifierIfPresent - Determine whether the next token is
/// a C++ access-specifier.
///
///       access-specifier: [C++ class.derived]
///         'private'
///         'protected'
///         'public'
AccessSpecifier Parser::getAccessSpecifierIfPresent() const
{
  switch (Tok.getKind()) {
  default: return AS_none;
  case tok::kw_private: return AS_private;
  case tok::kw_protected: return AS_protected;
  case tok::kw_public: return AS_public;
  }
}
