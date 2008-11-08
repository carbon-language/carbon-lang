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

/// ParseClassName - Parse a C++ class-name, which names a class. Note
/// that we only check that the result names a type; semantic analysis
/// will need to verify that the type names a class. The result is
/// either a type or NULL, dependending on whether a type name was
/// found.
///
///       class-name: [C++ 9.1]
///         identifier
///         template-id   [TODO]
/// 
Parser::TypeTy *Parser::ParseClassName(const CXXScopeSpec *SS) {
  // Parse the class-name.
  // FIXME: Alternatively, parse a simple-template-id.
  if (Tok.isNot(tok::identifier)) {
    Diag(Tok.getLocation(), diag::err_expected_class_name);
    return 0;
  }

  // We have an identifier; check whether it is actually a type.
  TypeTy *Type = Actions.isTypeName(*Tok.getIdentifierInfo(), CurScope, SS);
  if (!Type) {
    Diag(Tok.getLocation(), diag::err_expected_class_name);
    return 0;
  }

  // Consume the identifier.
  ConsumeToken();

  return Type;
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

  // Parse the (optional) nested-name-specifier.
  CXXScopeSpec SS;
  if (isTokenCXXScopeSpecifier()) {
    ParseCXXScopeSpecifier(SS);
    if (Tok.isNot(tok::identifier))
      Diag(Tok, diag::err_expected_ident);
  }

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
  DeclTy *TagDecl = Actions.ActOnTag(CurScope, TagType, TK, StartLoc, SS, Name, 
                                     NameLoc, Attr);

  // Parse the optional base clause (C++ only).
  if (getLang().CPlusPlus && Tok.is(tok::colon)) {
    ParseBaseClause(TagDecl);
  }

  // If there is a body, parse it and inform the actions module.
  if (Tok.is(tok::l_brace))
    if (getLang().CPlusPlus)
      ParseCXXMemberSpecification(StartLoc, TagType, TagDecl);
    else
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

  // Build up an array of parsed base specifiers.
  llvm::SmallVector<BaseTy *, 8> BaseInfo;

  while (true) {
    // Parse a base-specifier.
    BaseResult Result = ParseBaseSpecifier(ClassDecl);
    if (Result.isInvalid) {
      // Skip the rest of this base specifier, up until the comma or
      // opening brace.
      SkipUntil(tok::comma, tok::l_brace, true, true);
    } else {
      // Add this to our array of base specifiers.
      BaseInfo.push_back(Result.Val);
    }

    // If the next token is a comma, consume it and keep reading
    // base-specifiers.
    if (Tok.isNot(tok::comma)) break;
    
    // Consume the comma.
    ConsumeToken();
  }

  // Attach the base specifiers
  Actions.ActOnBaseSpecifiers(ClassDecl, &BaseInfo[0], BaseInfo.size());
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
Parser::BaseResult Parser::ParseBaseSpecifier(DeclTy *ClassDecl)
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
      Diag(VirtualLoc, diag::err_dup_virtual, 
           SourceRange(VirtualLoc, VirtualLoc));
    }

    IsVirtual = true;
  }

  // Parse optional '::' and optional nested-name-specifier.
  CXXScopeSpec SS;
  if (isTokenCXXScopeSpecifier())
    ParseCXXScopeSpecifier(SS);

  // The location of the base class itself.
  SourceLocation BaseLoc = Tok.getLocation();

  // Parse the class-name.
  TypeTy *BaseType = ParseClassName(&SS);
  if (!BaseType)
    return true;
  
  // Find the complete source range for the base-specifier.  
  SourceRange Range(StartLoc, BaseLoc);
  
  // Notify semantic analysis that we have parsed a complete
  // base-specifier.
  return Actions.ActOnBaseSpecifier(ClassDecl, Range, IsVirtual, Access, BaseType,
                                    BaseLoc);
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

/// ParseCXXClassMemberDeclaration - Parse a C++ class member declaration.
///
///       member-declaration:
///         decl-specifier-seq[opt] member-declarator-list[opt] ';'
///         function-definition ';'[opt]
///         ::[opt] nested-name-specifier template[opt] unqualified-id ';'[TODO]
///         using-declaration                                            [TODO]
/// [C++0x] static_assert-declaration                                    [TODO]
///         template-declaration                                         [TODO]
///
///       member-declarator-list:
///         member-declarator
///         member-declarator-list ',' member-declarator
///
///       member-declarator:
///         declarator pure-specifier[opt]
///         declarator constant-initializer[opt]
///         identifier[opt] ':' constant-expression
///
///       pure-specifier:   [TODO]
///         '= 0'
///
///       constant-initializer:
///         '=' constant-expression
///
Parser::DeclTy *Parser::ParseCXXClassMemberDeclaration(AccessSpecifier AS) {
  SourceLocation DSStart = Tok.getLocation();
  // decl-specifier-seq:
  // Parse the common declaration-specifiers piece.
  DeclSpec DS;
  ParseDeclarationSpecifiers(DS);

  if (Tok.is(tok::semi)) {
    ConsumeToken();
    // C++ 9.2p7: The member-declarator-list can be omitted only after a
    // class-specifier or an enum-specifier or in a friend declaration.
    // FIXME: Friend declarations.
    switch (DS.getTypeSpecType()) {
      case DeclSpec::TST_struct:
      case DeclSpec::TST_union:
      case DeclSpec::TST_class:
      case DeclSpec::TST_enum:
        return Actions.ParsedFreeStandingDeclSpec(CurScope, DS);
      default:
        Diag(DSStart, diag::err_no_declarators);
        return 0;
    }
  }

  Declarator DeclaratorInfo(DS, Declarator::MemberContext);

  if (Tok.isNot(tok::colon)) {
    // Parse the first declarator.
    ParseDeclarator(DeclaratorInfo);
    // Error parsing the declarator?
    if (DeclaratorInfo.getIdentifier() == 0) {
      // If so, skip until the semi-colon or a }.
      SkipUntil(tok::r_brace, true);
      if (Tok.is(tok::semi))
        ConsumeToken();
      return 0;
    }

    // function-definition:
    if (Tok.is(tok::l_brace)
        || (DeclaratorInfo.isFunctionDeclarator() && Tok.is(tok::colon))) {
      if (!DeclaratorInfo.isFunctionDeclarator()) {
        Diag(Tok, diag::err_func_def_no_params);
        ConsumeBrace();
        SkipUntil(tok::r_brace, true);
        return 0;
      }

      if (DS.getStorageClassSpec() == DeclSpec::SCS_typedef) {
        Diag(Tok, diag::err_function_declared_typedef);
        // This recovery skips the entire function body. It would be nice
        // to simply call ParseCXXInlineMethodDef() below, however Sema
        // assumes the declarator represents a function, not a typedef.
        ConsumeBrace();
        SkipUntil(tok::r_brace, true);
        return 0;
      }

      return ParseCXXInlineMethodDef(AS, DeclaratorInfo);
    }
  }

  // member-declarator-list:
  //   member-declarator
  //   member-declarator-list ',' member-declarator

  DeclTy *LastDeclInGroup = 0;
  ExprTy *BitfieldSize = 0;
  ExprTy *Init = 0;

  while (1) {

    // member-declarator:
    //   declarator pure-specifier[opt]
    //   declarator constant-initializer[opt]
    //   identifier[opt] ':' constant-expression

    if (Tok.is(tok::colon)) {
      ConsumeToken();
      ExprResult Res = ParseConstantExpression();
      if (Res.isInvalid)
        SkipUntil(tok::comma, true, true);
      else
        BitfieldSize = Res.Val;
    }
    
    // pure-specifier:
    //   '= 0'
    //
    // constant-initializer:
    //   '=' constant-expression

    if (Tok.is(tok::equal)) {
      ConsumeToken();
      ExprResult Res = ParseInitializer();
      if (Res.isInvalid)
        SkipUntil(tok::comma, true, true);
      else
        Init = Res.Val;
    }

    // If attributes exist after the declarator, parse them.
    if (Tok.is(tok::kw___attribute))
      DeclaratorInfo.AddAttributes(ParseAttributes());

    // NOTE: If Sema is the Action module and declarator is an instance field,
    // this call will *not* return the created decl; LastDeclInGroup will be
    // returned instead.
    // See Sema::ActOnCXXMemberDeclarator for details.
    LastDeclInGroup = Actions.ActOnCXXMemberDeclarator(CurScope, AS,
                                                       DeclaratorInfo,
                                                       BitfieldSize, Init,
                                                       LastDeclInGroup);

    // If we don't have a comma, it is either the end of the list (a ';')
    // or an error, bail out.
    if (Tok.isNot(tok::comma))
      break;
    
    // Consume the comma.
    ConsumeToken();
    
    // Parse the next declarator.
    DeclaratorInfo.clear();
    BitfieldSize = Init = 0;
    
    // Attributes are only allowed on the second declarator.
    if (Tok.is(tok::kw___attribute))
      DeclaratorInfo.AddAttributes(ParseAttributes());

    if (Tok.isNot(tok::colon))
      ParseDeclarator(DeclaratorInfo);
  }

  if (Tok.is(tok::semi)) {
    ConsumeToken();
    // Reverse the chain list.
    return Actions.FinalizeDeclaratorGroup(CurScope, LastDeclInGroup);
  }

  Diag(Tok, diag::err_expected_semi_decl_list);
  // Skip to end of block or statement
  SkipUntil(tok::r_brace, true, true);
  if (Tok.is(tok::semi))
    ConsumeToken();
  return 0;
}

/// ParseCXXMemberSpecification - Parse the class definition.
///
///       member-specification:
///         member-declaration member-specification[opt]
///         access-specifier ':' member-specification[opt]
///
void Parser::ParseCXXMemberSpecification(SourceLocation RecordLoc,
                                         unsigned TagType, DeclTy *TagDecl) {
  assert((TagType == DeclSpec::TST_struct ||
         TagType == DeclSpec::TST_union  ||
         TagType == DeclSpec::TST_class) && "Invalid TagType!");

  SourceLocation LBraceLoc = ConsumeBrace();

  if (!CurScope->isCXXClassScope() && // Not about to define a nested class.
      CurScope->isInCXXInlineMethodScope()) {
    // We will define a local class of an inline method.
    // Push a new LexedMethodsForTopClass for its inline methods.
    PushTopClassStack();
  }

  // Enter a scope for the class.
  EnterScope(Scope::CXXClassScope|Scope::DeclScope);

  Actions.ActOnStartCXXClassDef(CurScope, TagDecl, LBraceLoc);

  // C++ 11p3: Members of a class defined with the keyword class are private
  // by default. Members of a class defined with the keywords struct or union
  // are public by default.
  AccessSpecifier CurAS;
  if (TagType == DeclSpec::TST_class)
    CurAS = AS_private;
  else
    CurAS = AS_public;

  // While we still have something to read, read the member-declarations.
  while (Tok.isNot(tok::r_brace) && Tok.isNot(tok::eof)) {
    // Each iteration of this loop reads one member-declaration.
    
    // Check for extraneous top-level semicolon.
    if (Tok.is(tok::semi)) {
      Diag(Tok, diag::ext_extra_struct_semi);
      ConsumeToken();
      continue;
    }

    AccessSpecifier AS = getAccessSpecifierIfPresent();
    if (AS != AS_none) {
      // Current token is a C++ access specifier.
      CurAS = AS;
      ConsumeToken();
      ExpectAndConsume(tok::colon, diag::err_expected_colon);
      continue;
    }

    // Parse all the comma separated declarators.
    ParseCXXClassMemberDeclaration(CurAS);
  }
  
  SourceLocation RBraceLoc = MatchRHSPunctuation(tok::r_brace, LBraceLoc);
  
  AttributeList *AttrList = 0;
  // If attributes exist after class contents, parse them.
  if (Tok.is(tok::kw___attribute))
    AttrList = ParseAttributes(); // FIXME: where should I put them?

  Actions.ActOnFinishCXXMemberSpecification(CurScope, RecordLoc, TagDecl,
                                            LBraceLoc, RBraceLoc);

  // C++ 9.2p2: Within the class member-specification, the class is regarded as
  // complete within function bodies, default arguments,
  // exception-specifications, and constructor ctor-initializers (including
  // such things in nested classes).
  //
  // FIXME: Only function bodies are parsed correctly, fix the rest.
  if (!CurScope->getParent()->isCXXClassScope()) {
    // We are not inside a nested class. This class and its nested classes
    // are complete and we can parse the lexed inline method definitions.
    ParseLexedMethodDefs();

    // For a local class of inline method, pop the LexedMethodsForTopClass that
    // was previously pushed.

    assert((CurScope->isInCXXInlineMethodScope() ||
           TopClassStacks.size() == 1) &&
           "MethodLexers not getting popped properly!");
    if (CurScope->isInCXXInlineMethodScope())
      PopTopClassStack();
  }

  // Leave the class scope.
  ExitScope();

  Actions.ActOnFinishCXXClassDef(TagDecl);
}

/// ParseConstructorInitializer - Parse a C++ constructor initializer,
/// which explicitly initializes the members or base classes of a
/// class (C++ [class.base.init]). For example, the three initializers
/// after the ':' in the Derived constructor below:
///
/// @code
/// class Base { };
/// class Derived : Base {
///   int x;
///   float f;
/// public:
///   Derived(float f) : Base(), x(17), f(f) { }
/// };
/// @endcode
///
/// [C++]  ctor-initializer: 
///          ':' mem-initializer-list 
///
/// [C++]  mem-initializer-list: 
///          mem-initializer 
///          mem-initializer , mem-initializer-list 
void Parser::ParseConstructorInitializer(DeclTy *ConstructorDecl) {
  assert(Tok.is(tok::colon) && "Constructor initializer always starts with ':'");

  SourceLocation ColonLoc = ConsumeToken();
  
  llvm::SmallVector<MemInitTy*, 4> MemInitializers;
  
  do {
    MemInitResult MemInit = ParseMemInitializer(ConstructorDecl);
    if (!MemInit.isInvalid)
      MemInitializers.push_back(MemInit.Val);

    if (Tok.is(tok::comma))
      ConsumeToken();
    else if (Tok.is(tok::l_brace))
      break;
    else {
      // Skip over garbage, until we get to '{'.  Don't eat the '{'.
      SkipUntil(tok::l_brace, true, true);
      break;
    }
  } while (true);

  Actions.ActOnMemInitializers(ConstructorDecl, ColonLoc, 
                               &MemInitializers[0], MemInitializers.size());
}

/// ParseMemInitializer - Parse a C++ member initializer, which is
/// part of a constructor initializer that explicitly initializes one
/// member or base class (C++ [class.base.init]). See
/// ParseConstructorInitializer for an example.
///
/// [C++] mem-initializer:
///         mem-initializer-id '(' expression-list[opt] ')'
/// 
/// [C++] mem-initializer-id:
///         '::'[opt] nested-name-specifier[opt] class-name
///         identifier
Parser::MemInitResult Parser::ParseMemInitializer(DeclTy *ConstructorDecl) {
  // FIXME: parse '::'[opt] nested-name-specifier[opt]

  if (Tok.isNot(tok::identifier)) {
    Diag(Tok.getLocation(), diag::err_expected_member_or_base_name);
    return true;
  }

  // Get the identifier. This may be a member name or a class name,
  // but we'll let the semantic analysis determine which it is.
  IdentifierInfo *II = Tok.getIdentifierInfo();
  SourceLocation IdLoc = ConsumeToken();

  // Parse the '('.
  if (Tok.isNot(tok::l_paren)) {
    Diag(Tok.getLocation(), diag::err_expected_lparen);
    return true;
  }
  SourceLocation LParenLoc = ConsumeParen();

  // Parse the optional expression-list.
  ExprListTy ArgExprs;
  CommaLocsTy CommaLocs;
  if (Tok.isNot(tok::r_paren) && ParseExpressionList(ArgExprs, CommaLocs)) {
    SkipUntil(tok::r_paren);
    return true;
  }

  SourceLocation RParenLoc = MatchRHSPunctuation(tok::r_paren, LParenLoc);

  return Actions.ActOnMemInitializer(ConstructorDecl, CurScope, II, IdLoc, 
                                     LParenLoc, &ArgExprs[0], ArgExprs.size(), 
                                     &CommaLocs[0], RParenLoc);
}
