//===--- ParseTemplate.cpp - Template Parsing -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements parsing of C++ templates.
//
//===----------------------------------------------------------------------===//

#include "clang/Parse/Parser.h"
#include "clang/Parse/ParseDiagnostic.h"
#include "clang/Parse/DeclSpec.h"
#include "clang/Parse/Scope.h"
#include "clang/Parse/Template.h"
using namespace clang;

/// \brief Parse a template declaration, explicit instantiation, or
/// explicit specialization.
Parser::DeclPtrTy
Parser::ParseDeclarationStartingWithTemplate(unsigned Context,
                                             SourceLocation &DeclEnd,
                                             AccessSpecifier AS) {
  if (Tok.is(tok::kw_template) && NextToken().isNot(tok::less))
    return ParseExplicitInstantiation(SourceLocation(), ConsumeToken(),
                                      DeclEnd);

  return ParseTemplateDeclarationOrSpecialization(Context, DeclEnd, AS);
}

/// \brief RAII class that manages the template parameter depth.
namespace {
  class TemplateParameterDepthCounter {
    unsigned &Depth;
    unsigned AddedLevels;

  public:
    explicit TemplateParameterDepthCounter(unsigned &Depth)
      : Depth(Depth), AddedLevels(0) { }

    ~TemplateParameterDepthCounter() {
      Depth -= AddedLevels;
    }

    void operator++() {
      ++Depth;
      ++AddedLevels;
    }

    operator unsigned() const { return Depth; }
  };
}

/// \brief Parse a template declaration or an explicit specialization.
///
/// Template declarations include one or more template parameter lists
/// and either the function or class template declaration. Explicit
/// specializations contain one or more 'template < >' prefixes
/// followed by a (possibly templated) declaration. Since the
/// syntactic form of both features is nearly identical, we parse all
/// of the template headers together and let semantic analysis sort
/// the declarations from the explicit specializations.
///
///       template-declaration: [C++ temp]
///         'export'[opt] 'template' '<' template-parameter-list '>' declaration
///
///       explicit-specialization: [ C++ temp.expl.spec]
///         'template' '<' '>' declaration
Parser::DeclPtrTy
Parser::ParseTemplateDeclarationOrSpecialization(unsigned Context,
                                                 SourceLocation &DeclEnd,
                                                 AccessSpecifier AS) {
  assert((Tok.is(tok::kw_export) || Tok.is(tok::kw_template)) &&
         "Token does not start a template declaration.");

  // Enter template-parameter scope.
  ParseScope TemplateParmScope(this, Scope::TemplateParamScope);

  // Parse multiple levels of template headers within this template
  // parameter scope, e.g.,
  //
  //   template<typename T>
  //     template<typename U>
  //       class A<T>::B { ... };
  //
  // We parse multiple levels non-recursively so that we can build a
  // single data structure containing all of the template parameter
  // lists to easily differentiate between the case above and:
  //
  //   template<typename T>
  //   class A {
  //     template<typename U> class B;
  //   };
  //
  // In the first case, the action for declaring A<T>::B receives
  // both template parameter lists. In the second case, the action for
  // defining A<T>::B receives just the inner template parameter list
  // (and retrieves the outer template parameter list from its
  // context).
  bool isSpecialization = true;
  bool LastParamListWasEmpty = false;
  TemplateParameterLists ParamLists;
  TemplateParameterDepthCounter Depth(TemplateParameterDepth);
  do {
    // Consume the 'export', if any.
    SourceLocation ExportLoc;
    if (Tok.is(tok::kw_export)) {
      ExportLoc = ConsumeToken();
    }

    // Consume the 'template', which should be here.
    SourceLocation TemplateLoc;
    if (Tok.is(tok::kw_template)) {
      TemplateLoc = ConsumeToken();
    } else {
      Diag(Tok.getLocation(), diag::err_expected_template);
      return DeclPtrTy();
    }

    // Parse the '<' template-parameter-list '>'
    SourceLocation LAngleLoc, RAngleLoc;
    TemplateParameterList TemplateParams;
    if (ParseTemplateParameters(Depth, TemplateParams, LAngleLoc,
                                RAngleLoc)) {
      // Skip until the semi-colon or a }.
      SkipUntil(tok::r_brace, true, true);
      if (Tok.is(tok::semi))
        ConsumeToken();
      return DeclPtrTy();
    }

    ParamLists.push_back(
      Actions.ActOnTemplateParameterList(Depth, ExportLoc,
                                         TemplateLoc, LAngleLoc,
                                         TemplateParams.data(),
                                         TemplateParams.size(), RAngleLoc));

    if (!TemplateParams.empty()) {
      isSpecialization = false;
      ++Depth;
    } else {
      LastParamListWasEmpty = true;
    }
  } while (Tok.is(tok::kw_export) || Tok.is(tok::kw_template));

  // Parse the actual template declaration.
  return ParseSingleDeclarationAfterTemplate(Context,
                                             ParsedTemplateInfo(&ParamLists,
                                                             isSpecialization,
                                                         LastParamListWasEmpty),
                                             DeclEnd, AS);
}

/// \brief Parse a single declaration that declares a template,
/// template specialization, or explicit instantiation of a template.
///
/// \param TemplateParams if non-NULL, the template parameter lists
/// that preceded this declaration. In this case, the declaration is a
/// template declaration, out-of-line definition of a template, or an
/// explicit template specialization. When NULL, the declaration is an
/// explicit template instantiation.
///
/// \param TemplateLoc when TemplateParams is NULL, the location of
/// the 'template' keyword that indicates that we have an explicit
/// template instantiation.
///
/// \param DeclEnd will receive the source location of the last token
/// within this declaration.
///
/// \param AS the access specifier associated with this
/// declaration. Will be AS_none for namespace-scope declarations.
///
/// \returns the new declaration.
Parser::DeclPtrTy
Parser::ParseSingleDeclarationAfterTemplate(
                                       unsigned Context,
                                       const ParsedTemplateInfo &TemplateInfo,
                                       SourceLocation &DeclEnd,
                                       AccessSpecifier AS) {
  assert(TemplateInfo.Kind != ParsedTemplateInfo::NonTemplate &&
         "Template information required");

  if (Context == Declarator::MemberContext) {
    // We are parsing a member template.
    ParseCXXClassMemberDeclaration(AS, TemplateInfo);
    return DeclPtrTy::make((void*)0);
  }

  // Parse the declaration specifiers.
  ParsingDeclSpec DS(*this);

  if (getLang().CPlusPlus0x && isCXX0XAttributeSpecifier())
    DS.AddAttributes(ParseCXX0XAttributes().AttrList);

  ParseDeclarationSpecifiers(DS, TemplateInfo, AS);

  if (Tok.is(tok::semi)) {
    DeclEnd = ConsumeToken();
    DeclPtrTy Decl = Actions.ParsedFreeStandingDeclSpec(CurScope, DS);
    DS.complete(Decl);
    return Decl;
  }

  // Parse the declarator.
  ParsingDeclarator DeclaratorInfo(*this, DS, (Declarator::TheContext)Context);
  ParseDeclarator(DeclaratorInfo);
  // Error parsing the declarator?
  if (!DeclaratorInfo.hasName()) {
    // If so, skip until the semi-colon or a }.
    SkipUntil(tok::r_brace, true, true);
    if (Tok.is(tok::semi))
      ConsumeToken();
    return DeclPtrTy();
  }

  // If we have a declaration or declarator list, handle it.
  if (isDeclarationAfterDeclarator()) {
    // Parse this declaration.
    DeclPtrTy ThisDecl = ParseDeclarationAfterDeclarator(DeclaratorInfo,
                                                         TemplateInfo);

    if (Tok.is(tok::comma)) {
      Diag(Tok, diag::err_multiple_template_declarators)
        << (int)TemplateInfo.Kind;
      SkipUntil(tok::semi, true, false);
      return ThisDecl;
    }

    // Eat the semi colon after the declaration.
    ExpectAndConsume(tok::semi, diag::err_expected_semi_declaration);
    DS.complete(ThisDecl);
    return ThisDecl;
  }

  if (DeclaratorInfo.isFunctionDeclarator() &&
      isStartOfFunctionDefinition()) {
    if (DS.getStorageClassSpec() == DeclSpec::SCS_typedef) {
      Diag(Tok, diag::err_function_declared_typedef);

      if (Tok.is(tok::l_brace)) {
        // This recovery skips the entire function body. It would be nice
        // to simply call ParseFunctionDefinition() below, however Sema
        // assumes the declarator represents a function, not a typedef.
        ConsumeBrace();
        SkipUntil(tok::r_brace, true);
      } else {
        SkipUntil(tok::semi);
      }
      return DeclPtrTy();
    }
    return ParseFunctionDefinition(DeclaratorInfo, TemplateInfo);
  }

  if (DeclaratorInfo.isFunctionDeclarator())
    Diag(Tok, diag::err_expected_fn_body);
  else
    Diag(Tok, diag::err_invalid_token_after_toplevel_declarator);
  SkipUntil(tok::semi);
  return DeclPtrTy();
}

/// ParseTemplateParameters - Parses a template-parameter-list enclosed in
/// angle brackets. Depth is the depth of this template-parameter-list, which
/// is the number of template headers directly enclosing this template header.
/// TemplateParams is the current list of template parameters we're building.
/// The template parameter we parse will be added to this list. LAngleLoc and
/// RAngleLoc will receive the positions of the '<' and '>', respectively,
/// that enclose this template parameter list.
///
/// \returns true if an error occurred, false otherwise.
bool Parser::ParseTemplateParameters(unsigned Depth,
                                     TemplateParameterList &TemplateParams,
                                     SourceLocation &LAngleLoc,
                                     SourceLocation &RAngleLoc) {
  // Get the template parameter list.
  if (!Tok.is(tok::less)) {
    Diag(Tok.getLocation(), diag::err_expected_less_after) << "template";
    return true;
  }
  LAngleLoc = ConsumeToken();

  // Try to parse the template parameter list.
  if (Tok.is(tok::greater))
    RAngleLoc = ConsumeToken();
  else if (ParseTemplateParameterList(Depth, TemplateParams)) {
    if (!Tok.is(tok::greater)) {
      Diag(Tok.getLocation(), diag::err_expected_greater);
      return true;
    }
    RAngleLoc = ConsumeToken();
  }
  return false;
}

/// ParseTemplateParameterList - Parse a template parameter list. If
/// the parsing fails badly (i.e., closing bracket was left out), this
/// will try to put the token stream in a reasonable position (closing
/// a statement, etc.) and return false.
///
///       template-parameter-list:    [C++ temp]
///         template-parameter
///         template-parameter-list ',' template-parameter
bool
Parser::ParseTemplateParameterList(unsigned Depth,
                                   TemplateParameterList &TemplateParams) {
  while (1) {
    if (DeclPtrTy TmpParam
          = ParseTemplateParameter(Depth, TemplateParams.size())) {
      TemplateParams.push_back(TmpParam);
    } else {
      // If we failed to parse a template parameter, skip until we find
      // a comma or closing brace.
      SkipUntil(tok::comma, tok::greater, true, true);
    }

    // Did we find a comma or the end of the template parmeter list?
    if (Tok.is(tok::comma)) {
      ConsumeToken();
    } else if (Tok.is(tok::greater)) {
      // Don't consume this... that's done by template parser.
      break;
    } else {
      // Somebody probably forgot to close the template. Skip ahead and
      // try to get out of the expression. This error is currently
      // subsumed by whatever goes on in ParseTemplateParameter.
      // TODO: This could match >>, and it would be nice to avoid those
      // silly errors with template <vec<T>>.
      // Diag(Tok.getLocation(), diag::err_expected_comma_greater);
      SkipUntil(tok::greater, true, true);
      return false;
    }
  }
  return true;
}

/// \brief Determine whether the parser is at the start of a template
/// type parameter.
bool Parser::isStartOfTemplateTypeParameter() {
  if (Tok.is(tok::kw_class))
    return true;

  if (Tok.isNot(tok::kw_typename))
    return false;

  // C++ [temp.param]p2:
  //   There is no semantic difference between class and typename in a
  //   template-parameter. typename followed by an unqualified-id
  //   names a template type parameter. typename followed by a
  //   qualified-id denotes the type in a non-type
  //   parameter-declaration.
  Token Next = NextToken();

  // If we have an identifier, skip over it.
  if (Next.getKind() == tok::identifier)
    Next = GetLookAheadToken(2);

  switch (Next.getKind()) {
  case tok::equal:
  case tok::comma:
  case tok::greater:
  case tok::greatergreater:
  case tok::ellipsis:
    return true;

  default:
    return false;
  }
}

/// ParseTemplateParameter - Parse a template-parameter (C++ [temp.param]).
///
///       template-parameter: [C++ temp.param]
///         type-parameter
///         parameter-declaration
///
///       type-parameter: (see below)
///         'class' ...[opt][C++0x] identifier[opt]
///         'class' identifier[opt] '=' type-id
///         'typename' ...[opt][C++0x] identifier[opt]
///         'typename' identifier[opt] '=' type-id
///         'template' ...[opt][C++0x] '<' template-parameter-list '>' 'class' identifier[opt]
///         'template' '<' template-parameter-list '>' 'class' identifier[opt] = id-expression
Parser::DeclPtrTy
Parser::ParseTemplateParameter(unsigned Depth, unsigned Position) {
  if (isStartOfTemplateTypeParameter())
    return ParseTypeParameter(Depth, Position);

  if (Tok.is(tok::kw_template))
    return ParseTemplateTemplateParameter(Depth, Position);

  // If it's none of the above, then it must be a parameter declaration.
  // NOTE: This will pick up errors in the closure of the template parameter
  // list (e.g., template < ; Check here to implement >> style closures.
  return ParseNonTypeTemplateParameter(Depth, Position);
}

/// ParseTypeParameter - Parse a template type parameter (C++ [temp.param]).
/// Other kinds of template parameters are parsed in
/// ParseTemplateTemplateParameter and ParseNonTypeTemplateParameter.
///
///       type-parameter:     [C++ temp.param]
///         'class' ...[opt][C++0x] identifier[opt]
///         'class' identifier[opt] '=' type-id
///         'typename' ...[opt][C++0x] identifier[opt]
///         'typename' identifier[opt] '=' type-id
Parser::DeclPtrTy Parser::ParseTypeParameter(unsigned Depth, unsigned Position){
  assert((Tok.is(tok::kw_class) || Tok.is(tok::kw_typename)) &&
         "A type-parameter starts with 'class' or 'typename'");

  // Consume the 'class' or 'typename' keyword.
  bool TypenameKeyword = Tok.is(tok::kw_typename);
  SourceLocation KeyLoc = ConsumeToken();

  // Grab the ellipsis (if given).
  bool Ellipsis = false;
  SourceLocation EllipsisLoc;
  if (Tok.is(tok::ellipsis)) {
    Ellipsis = true;
    EllipsisLoc = ConsumeToken();

    if (!getLang().CPlusPlus0x)
      Diag(EllipsisLoc, diag::err_variadic_templates);
  }

  // Grab the template parameter name (if given)
  SourceLocation NameLoc;
  IdentifierInfo* ParamName = 0;
  if (Tok.is(tok::identifier)) {
    ParamName = Tok.getIdentifierInfo();
    NameLoc = ConsumeToken();
  } else if (Tok.is(tok::equal) || Tok.is(tok::comma) ||
            Tok.is(tok::greater)) {
    // Unnamed template parameter. Don't have to do anything here, just
    // don't consume this token.
  } else {
    Diag(Tok.getLocation(), diag::err_expected_ident);
    return DeclPtrTy();
  }

  DeclPtrTy TypeParam = Actions.ActOnTypeParameter(CurScope, TypenameKeyword,
                                                   Ellipsis, EllipsisLoc,
                                                   KeyLoc, ParamName, NameLoc,
                                                   Depth, Position);

  // Grab a default type id (if given).
  if (Tok.is(tok::equal)) {
    SourceLocation EqualLoc = ConsumeToken();
    SourceLocation DefaultLoc = Tok.getLocation();
    TypeResult DefaultType = ParseTypeName();
    if (!DefaultType.isInvalid())
      Actions.ActOnTypeParameterDefault(TypeParam, EqualLoc, DefaultLoc,
                                        DefaultType.get());
  }

  return TypeParam;
}

/// ParseTemplateTemplateParameter - Handle the parsing of template
/// template parameters.
///
///       type-parameter:    [C++ temp.param]
///         'template' '<' template-parameter-list '>' 'class' identifier[opt]
///         'template' '<' template-parameter-list '>' 'class' identifier[opt] = id-expression
Parser::DeclPtrTy
Parser::ParseTemplateTemplateParameter(unsigned Depth, unsigned Position) {
  assert(Tok.is(tok::kw_template) && "Expected 'template' keyword");

  // Handle the template <...> part.
  SourceLocation TemplateLoc = ConsumeToken();
  TemplateParameterList TemplateParams;
  SourceLocation LAngleLoc, RAngleLoc;
  {
    ParseScope TemplateParmScope(this, Scope::TemplateParamScope);
    if (ParseTemplateParameters(Depth + 1, TemplateParams, LAngleLoc,
                               RAngleLoc)) {
      return DeclPtrTy();
    }
  }

  // Generate a meaningful error if the user forgot to put class before the
  // identifier, comma, or greater.
  if (!Tok.is(tok::kw_class)) {
    Diag(Tok.getLocation(), diag::err_expected_class_before)
      << PP.getSpelling(Tok);
    return DeclPtrTy();
  }
  SourceLocation ClassLoc = ConsumeToken();

  // Get the identifier, if given.
  SourceLocation NameLoc;
  IdentifierInfo* ParamName = 0;
  if (Tok.is(tok::identifier)) {
    ParamName = Tok.getIdentifierInfo();
    NameLoc = ConsumeToken();
  } else if (Tok.is(tok::equal) || Tok.is(tok::comma) || Tok.is(tok::greater)) {
    // Unnamed template parameter. Don't have to do anything here, just
    // don't consume this token.
  } else {
    Diag(Tok.getLocation(), diag::err_expected_ident);
    return DeclPtrTy();
  }

  TemplateParamsTy *ParamList =
    Actions.ActOnTemplateParameterList(Depth, SourceLocation(),
                                       TemplateLoc, LAngleLoc,
                                       &TemplateParams[0],
                                       TemplateParams.size(),
                                       RAngleLoc);

  Parser::DeclPtrTy Param
    = Actions.ActOnTemplateTemplateParameter(CurScope, TemplateLoc,
                                             ParamList, ParamName,
                                             NameLoc, Depth, Position);

  // Get the a default value, if given.
  if (Tok.is(tok::equal)) {
    SourceLocation EqualLoc = ConsumeToken();
    ParsedTemplateArgument Default = ParseTemplateTemplateArgument();
    if (Default.isInvalid()) {
      Diag(Tok.getLocation(), 
           diag::err_default_template_template_parameter_not_template);
      static tok::TokenKind EndToks[] = { 
        tok::comma, tok::greater, tok::greatergreater
      };
      SkipUntil(EndToks, 3, true, true);
      return Param;
    } else if (Param)
      Actions.ActOnTemplateTemplateParameterDefault(Param, EqualLoc, Default);
  }

  return Param;
}

/// ParseNonTypeTemplateParameter - Handle the parsing of non-type
/// template parameters (e.g., in "template<int Size> class array;").
///
///       template-parameter:
///         ...
///         parameter-declaration
///
/// NOTE: It would be ideal to simply call out to ParseParameterDeclaration(),
/// but that didn't work out to well. Instead, this tries to recrate the basic
/// parsing of parameter declarations, but tries to constrain it for template
/// parameters.
/// FIXME: We need to make a ParseParameterDeclaration that works for
/// non-type template parameters and normal function parameters.
Parser::DeclPtrTy
Parser::ParseNonTypeTemplateParameter(unsigned Depth, unsigned Position) {
  SourceLocation StartLoc = Tok.getLocation();

  // Parse the declaration-specifiers (i.e., the type).
  // FIXME: The type should probably be restricted in some way... Not all
  // declarators (parts of declarators?) are accepted for parameters.
  DeclSpec DS;
  ParseDeclarationSpecifiers(DS);

  // Parse this as a typename.
  Declarator ParamDecl(DS, Declarator::TemplateParamContext);
  ParseDeclarator(ParamDecl);
  if (DS.getTypeSpecType() == DeclSpec::TST_unspecified && !DS.getTypeRep()) {
    // This probably shouldn't happen - and it's more of a Sema thing, but
    // basically we didn't parse the type name because we couldn't associate
    // it with an AST node. we should just skip to the comma or greater.
    // TODO: This is currently a placeholder for some kind of Sema Error.
    Diag(Tok.getLocation(), diag::err_parse_error);
    SkipUntil(tok::comma, tok::greater, true, true);
    return DeclPtrTy();
  }

  // Create the parameter.
  DeclPtrTy Param = Actions.ActOnNonTypeTemplateParameter(CurScope, ParamDecl,
                                                          Depth, Position);

  // If there is a default value, parse it.
  if (Tok.is(tok::equal)) {
    SourceLocation EqualLoc = ConsumeToken();

    // C++ [temp.param]p15:
    //   When parsing a default template-argument for a non-type
    //   template-parameter, the first non-nested > is taken as the
    //   end of the template-parameter-list rather than a greater-than
    //   operator.
    GreaterThanIsOperatorScope G(GreaterThanIsOperator, false);

    OwningExprResult DefaultArg = ParseAssignmentExpression();
    if (DefaultArg.isInvalid())
      SkipUntil(tok::comma, tok::greater, true, true);
    else if (Param)
      Actions.ActOnNonTypeTemplateParameterDefault(Param, EqualLoc,
                                                   move(DefaultArg));
  }

  return Param;
}

/// \brief Parses a template-id that after the template name has
/// already been parsed.
///
/// This routine takes care of parsing the enclosed template argument
/// list ('<' template-parameter-list [opt] '>') and placing the
/// results into a form that can be transferred to semantic analysis.
///
/// \param Template the template declaration produced by isTemplateName
///
/// \param TemplateNameLoc the source location of the template name
///
/// \param SS if non-NULL, the nested-name-specifier preceding the
/// template name.
///
/// \param ConsumeLastToken if true, then we will consume the last
/// token that forms the template-id. Otherwise, we will leave the
/// last token in the stream (e.g., so that it can be replaced with an
/// annotation token).
bool
Parser::ParseTemplateIdAfterTemplateName(TemplateTy Template,
                                         SourceLocation TemplateNameLoc,
                                         const CXXScopeSpec *SS,
                                         bool ConsumeLastToken,
                                         SourceLocation &LAngleLoc,
                                         TemplateArgList &TemplateArgs,
                                         SourceLocation &RAngleLoc) {
  assert(Tok.is(tok::less) && "Must have already parsed the template-name");

  // Consume the '<'.
  LAngleLoc = ConsumeToken();

  // Parse the optional template-argument-list.
  bool Invalid = false;
  {
    GreaterThanIsOperatorScope G(GreaterThanIsOperator, false);
    if (Tok.isNot(tok::greater))
      Invalid = ParseTemplateArgumentList(TemplateArgs);

    if (Invalid) {
      // Try to find the closing '>'.
      SkipUntil(tok::greater, true, !ConsumeLastToken);

      return true;
    }
  }

  if (Tok.isNot(tok::greater) && Tok.isNot(tok::greatergreater))
    return true;

  // Determine the location of the '>' or '>>'. Only consume this
  // token if the caller asked us to.
  RAngleLoc = Tok.getLocation();

  if (Tok.is(tok::greatergreater)) {
    if (!getLang().CPlusPlus0x) {
      const char *ReplaceStr = "> >";
      if (NextToken().is(tok::greater) || NextToken().is(tok::greatergreater))
        ReplaceStr = "> > ";

      Diag(Tok.getLocation(), diag::err_two_right_angle_brackets_need_space)
        << CodeModificationHint::CreateReplacement(
                                 SourceRange(Tok.getLocation()), ReplaceStr);
    }

    Tok.setKind(tok::greater);
    if (!ConsumeLastToken) {
      // Since we're not supposed to consume the '>>' token, we need
      // to insert a second '>' token after the first.
      PP.EnterToken(Tok);
    }
  } else if (ConsumeLastToken)
    ConsumeToken();

  return false;
}

/// \brief Replace the tokens that form a simple-template-id with an
/// annotation token containing the complete template-id.
///
/// The first token in the stream must be the name of a template that
/// is followed by a '<'. This routine will parse the complete
/// simple-template-id and replace the tokens with a single annotation
/// token with one of two different kinds: if the template-id names a
/// type (and \p AllowTypeAnnotation is true), the annotation token is
/// a type annotation that includes the optional nested-name-specifier
/// (\p SS). Otherwise, the annotation token is a template-id
/// annotation that does not include the optional
/// nested-name-specifier.
///
/// \param Template  the declaration of the template named by the first
/// token (an identifier), as returned from \c Action::isTemplateName().
///
/// \param TemplateNameKind the kind of template that \p Template
/// refers to, as returned from \c Action::isTemplateName().
///
/// \param SS if non-NULL, the nested-name-specifier that precedes
/// this template name.
///
/// \param TemplateKWLoc if valid, specifies that this template-id
/// annotation was preceded by the 'template' keyword and gives the
/// location of that keyword. If invalid (the default), then this
/// template-id was not preceded by a 'template' keyword.
///
/// \param AllowTypeAnnotation if true (the default), then a
/// simple-template-id that refers to a class template, template
/// template parameter, or other template that produces a type will be
/// replaced with a type annotation token. Otherwise, the
/// simple-template-id is always replaced with a template-id
/// annotation token.
///
/// If an unrecoverable parse error occurs and no annotation token can be
/// formed, this function returns true.
///
bool Parser::AnnotateTemplateIdToken(TemplateTy Template, TemplateNameKind TNK,
                                     const CXXScopeSpec *SS,
                                     UnqualifiedId &TemplateName,
                                     SourceLocation TemplateKWLoc,
                                     bool AllowTypeAnnotation) {
  assert(getLang().CPlusPlus && "Can only annotate template-ids in C++");
  assert(Template && Tok.is(tok::less) &&
         "Parser isn't at the beginning of a template-id");

  // Consume the template-name.
  SourceLocation TemplateNameLoc = TemplateName.getSourceRange().getBegin();

  // Parse the enclosed template argument list.
  SourceLocation LAngleLoc, RAngleLoc;
  TemplateArgList TemplateArgs;
  bool Invalid = ParseTemplateIdAfterTemplateName(Template, 
                                                  TemplateNameLoc,
                                                  SS, false, LAngleLoc,
                                                  TemplateArgs,
                                                  RAngleLoc);

  if (Invalid) {
    // If we failed to parse the template ID but skipped ahead to a >, we're not
    // going to be able to form a token annotation.  Eat the '>' if present.
    if (Tok.is(tok::greater))
      ConsumeToken();
    return true;
  }

  ASTTemplateArgsPtr TemplateArgsPtr(Actions, TemplateArgs.data(),
                                     TemplateArgs.size());

  // Build the annotation token.
  if (TNK == TNK_Type_template && AllowTypeAnnotation) {
    Action::TypeResult Type
      = Actions.ActOnTemplateIdType(Template, TemplateNameLoc,
                                    LAngleLoc, TemplateArgsPtr,
                                    RAngleLoc);
    if (Type.isInvalid()) {
      // If we failed to parse the template ID but skipped ahead to a >, we're not
      // going to be able to form a token annotation.  Eat the '>' if present.
      if (Tok.is(tok::greater))
        ConsumeToken();
      return true;
    }

    Tok.setKind(tok::annot_typename);
    Tok.setAnnotationValue(Type.get());
    if (SS && SS->isNotEmpty())
      Tok.setLocation(SS->getBeginLoc());
    else if (TemplateKWLoc.isValid())
      Tok.setLocation(TemplateKWLoc);
    else
      Tok.setLocation(TemplateNameLoc);
  } else {
    // Build a template-id annotation token that can be processed
    // later.
    Tok.setKind(tok::annot_template_id);
    TemplateIdAnnotation *TemplateId
      = TemplateIdAnnotation::Allocate(TemplateArgs.size());
    TemplateId->TemplateNameLoc = TemplateNameLoc;
    if (TemplateName.getKind() == UnqualifiedId::IK_Identifier) {
      TemplateId->Name = TemplateName.Identifier;
      TemplateId->Operator = OO_None;
    } else {
      TemplateId->Name = 0;
      TemplateId->Operator = TemplateName.OperatorFunctionId.Operator;
    }
    TemplateId->Template = Template.getAs<void*>();
    TemplateId->Kind = TNK;
    TemplateId->LAngleLoc = LAngleLoc;
    TemplateId->RAngleLoc = RAngleLoc;
    ParsedTemplateArgument *Args = TemplateId->getTemplateArgs();
    for (unsigned Arg = 0, ArgEnd = TemplateArgs.size(); Arg != ArgEnd; ++Arg)
      Args[Arg] = TemplateArgs[Arg];
    Tok.setAnnotationValue(TemplateId);
    if (TemplateKWLoc.isValid())
      Tok.setLocation(TemplateKWLoc);
    else
      Tok.setLocation(TemplateNameLoc);

    TemplateArgsPtr.release();
  }

  // Common fields for the annotation token
  Tok.setAnnotationEndLoc(RAngleLoc);

  // In case the tokens were cached, have Preprocessor replace them with the
  // annotation token.
  PP.AnnotateCachedTokens(Tok);
  return false;
}

/// \brief Replaces a template-id annotation token with a type
/// annotation token.
///
/// If there was a failure when forming the type from the template-id,
/// a type annotation token will still be created, but will have a
/// NULL type pointer to signify an error.
void Parser::AnnotateTemplateIdTokenAsType(const CXXScopeSpec *SS) {
  assert(Tok.is(tok::annot_template_id) && "Requires template-id tokens");

  TemplateIdAnnotation *TemplateId
    = static_cast<TemplateIdAnnotation *>(Tok.getAnnotationValue());
  assert((TemplateId->Kind == TNK_Type_template ||
          TemplateId->Kind == TNK_Dependent_template_name) &&
         "Only works for type and dependent templates");

  ASTTemplateArgsPtr TemplateArgsPtr(Actions,
                                     TemplateId->getTemplateArgs(),
                                     TemplateId->NumArgs);

  Action::TypeResult Type
    = Actions.ActOnTemplateIdType(TemplateTy::make(TemplateId->Template),
                                  TemplateId->TemplateNameLoc,
                                  TemplateId->LAngleLoc,
                                  TemplateArgsPtr,
                                  TemplateId->RAngleLoc);
  // Create the new "type" annotation token.
  Tok.setKind(tok::annot_typename);
  Tok.setAnnotationValue(Type.isInvalid()? 0 : Type.get());
  if (SS && SS->isNotEmpty()) // it was a C++ qualified type name.
    Tok.setLocation(SS->getBeginLoc());
  Tok.setAnnotationEndLoc(TemplateId->TemplateNameLoc);

  // Replace the template-id annotation token, and possible the scope-specifier
  // that precedes it, with the typename annotation token.
  PP.AnnotateCachedTokens(Tok);
  TemplateId->Destroy();
}

/// \brief Determine whether the given token can end a template argument.
static bool isEndOfTemplateArgument(Token Tok) {
  return Tok.is(tok::comma) || Tok.is(tok::greater) || 
         Tok.is(tok::greatergreater);
}

/// \brief Parse a C++ template template argument.
ParsedTemplateArgument Parser::ParseTemplateTemplateArgument() {
  if (!Tok.is(tok::identifier) && !Tok.is(tok::coloncolon) &&
      !Tok.is(tok::annot_cxxscope))
    return ParsedTemplateArgument();

  // C++0x [temp.arg.template]p1:
  //   A template-argument for a template template-parameter shall be the name
  //   of a class template or a template alias, expressed as id-expression.
  //   
  // We parse an id-expression that refers to a class template or template
  // alias. The grammar we parse is:
  //
  //   nested-name-specifier[opt] template[opt] identifier
  //
  // followed by a token that terminates a template argument, such as ',', 
  // '>', or (in some cases) '>>'.
  CXXScopeSpec SS; // nested-name-specifier, if present
  ParseOptionalCXXScopeSpecifier(SS, /*ObjectType=*/0, 
                                 /*EnteringContext=*/false);
  
  if (SS.isSet() && Tok.is(tok::kw_template)) {
    // Parse the optional 'template' keyword following the 
    // nested-name-specifier.
    SourceLocation TemplateLoc = ConsumeToken();
    
    if (Tok.is(tok::identifier)) {
      // We appear to have a dependent template name.
      UnqualifiedId Name;
      Name.setIdentifier(Tok.getIdentifierInfo(), Tok.getLocation());
      ConsumeToken(); // the identifier
      
      // If the next token signals the end of a template argument,
      // then we have a dependent template name that could be a template
      // template argument.
      if (isEndOfTemplateArgument(Tok)) {
        TemplateTy Template
        = Actions.ActOnDependentTemplateName(TemplateLoc, SS, Name, 
                                             /*ObjectType=*/0,
                                             /*EnteringContext=*/false);
        if (Template.get())
          return ParsedTemplateArgument(SS, Template, Name.StartLocation);
      }
    } 
  } else if (Tok.is(tok::identifier)) {
    // We may have a (non-dependent) template name.
    TemplateTy Template;
    UnqualifiedId Name;
    Name.setIdentifier(Tok.getIdentifierInfo(), Tok.getLocation());
    ConsumeToken(); // the identifier
    
    if (isEndOfTemplateArgument(Tok)) {
      TemplateNameKind TNK = Actions.isTemplateName(CurScope, SS, Name, 
                                                    /*ObjectType=*/0, 
                                                    /*EnteringContext=*/false, 
                                                    Template);
      if (TNK == TNK_Dependent_template_name || TNK == TNK_Type_template) {
        // We have an id-expression that refers to a class template or
        // (C++0x) template alias. 
        return ParsedTemplateArgument(SS, Template, Name.StartLocation);
      }
    }
  }
  
  // We don't have a template template argument.  
  return ParsedTemplateArgument();
}

/// ParseTemplateArgument - Parse a C++ template argument (C++ [temp.names]).
///
///       template-argument: [C++ 14.2]
///         constant-expression
///         type-id
///         id-expression
ParsedTemplateArgument Parser::ParseTemplateArgument() {
  // C++ [temp.arg]p2:
  //   In a template-argument, an ambiguity between a type-id and an
  //   expression is resolved to a type-id, regardless of the form of
  //   the corresponding template-parameter.
  //
  // Therefore, we initially try to parse a type-id.  
  if (isCXXTypeId(TypeIdAsTemplateArgument)) {
    SourceLocation Loc = Tok.getLocation();
    TypeResult TypeArg = ParseTypeName();
    if (TypeArg.isInvalid())
      return ParsedTemplateArgument();
    
    return ParsedTemplateArgument(ParsedTemplateArgument::Type, TypeArg.get(), 
                                  Loc);
  }
  
  // Try to parse a template template argument.
  {
    TentativeParsingAction TPA(*this);

    ParsedTemplateArgument TemplateTemplateArgument
      = ParseTemplateTemplateArgument();
    if (!TemplateTemplateArgument.isInvalid()) {
      TPA.Commit();
      return TemplateTemplateArgument;
    }
    
    // Revert this tentative parse to parse a non-type template argument.
    TPA.Revert();
  }
  
  // Parse a non-type template argument. 
  SourceLocation Loc = Tok.getLocation();
  OwningExprResult ExprArg = ParseConstantExpression();
  if (ExprArg.isInvalid() || !ExprArg.get())
    return ParsedTemplateArgument();

  return ParsedTemplateArgument(ParsedTemplateArgument::NonType, 
                                ExprArg.release(), Loc);
}

/// ParseTemplateArgumentList - Parse a C++ template-argument-list
/// (C++ [temp.names]). Returns true if there was an error.
///
///       template-argument-list: [C++ 14.2]
///         template-argument
///         template-argument-list ',' template-argument
bool
Parser::ParseTemplateArgumentList(TemplateArgList &TemplateArgs) {
  while (true) {
    ParsedTemplateArgument Arg = ParseTemplateArgument();
    if (Arg.isInvalid()) {
      SkipUntil(tok::comma, tok::greater, true, true);
      return true;
    }

    // Save this template argument.
    TemplateArgs.push_back(Arg);
      
    // If the next token is a comma, consume it and keep reading
    // arguments.
    if (Tok.isNot(tok::comma)) break;

    // Consume the comma.
    ConsumeToken();
  }

  return Tok.isNot(tok::greater) && Tok.isNot(tok::greatergreater);
}

/// \brief Parse a C++ explicit template instantiation
/// (C++ [temp.explicit]).
///
///       explicit-instantiation:
///         'extern' [opt] 'template' declaration
///
/// Note that the 'extern' is a GNU extension and C++0x feature.
Parser::DeclPtrTy
Parser::ParseExplicitInstantiation(SourceLocation ExternLoc,
                                   SourceLocation TemplateLoc,
                                   SourceLocation &DeclEnd) {
  return ParseSingleDeclarationAfterTemplate(Declarator::FileContext,
                                             ParsedTemplateInfo(ExternLoc,
                                                                TemplateLoc),
                                             DeclEnd, AS_none);
}
