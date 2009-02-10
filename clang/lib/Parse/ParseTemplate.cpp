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
#include "AstGuard.h"

using namespace clang;

/// ParseTemplateDeclaration - Parse a template declaration, which includes 
/// the template parameter list and either a function of class declaration.
///
///       template-declaration: [C++ temp]
///         'export'[opt] 'template' '<' template-parameter-list '>' declaration
Parser::DeclTy *Parser::ParseTemplateDeclaration(unsigned Context) {
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
  // lists easily differentiate between the case above and:
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
  TemplateParameterLists ParamLists;
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
      return 0;
    }
  
    // Parse the '<' template-parameter-list '>'
    SourceLocation LAngleLoc, RAngleLoc;
    TemplateParameterList TemplateParams;
    ParseTemplateParameters(ParamLists.size(), TemplateParams, LAngleLoc, 
                            RAngleLoc);

    ParamLists.push_back(
      Actions.ActOnTemplateParameterList(ParamLists.size(), ExportLoc, 
                                         TemplateLoc, LAngleLoc, 
                                         &TemplateParams[0],
                                         TemplateParams.size(), RAngleLoc));
  } while (Tok.is(tok::kw_export) || Tok.is(tok::kw_template));

  // Parse the actual template declaration.
  return ParseDeclarationOrFunctionDefinition(&ParamLists);
}

/// ParseTemplateParameters - Parses a template-parameter-list enclosed in
/// angle brackets. Depth is the depth of this template-parameter-list, which
/// is the number of template headers directly enclosing this template header.
/// TemplateParams is the current list of template parameters we're building.
/// The template parameter we parse will be added to this list. LAngleLoc and
/// RAngleLoc will receive the positions of the '<' and '>', respectively, 
/// that enclose this template parameter list.
bool Parser::ParseTemplateParameters(unsigned Depth,
                                     TemplateParameterList &TemplateParams,
                                     SourceLocation &LAngleLoc,
                                     SourceLocation &RAngleLoc) {
  // Get the template parameter list.
  if(!Tok.is(tok::less)) {
    Diag(Tok.getLocation(), diag::err_expected_less_after) << "template";
    return false;
  }
  LAngleLoc = ConsumeToken();
  
  // Try to parse the template parameter list.
  if (Tok.is(tok::greater))
    RAngleLoc = ConsumeToken();
  else if(ParseTemplateParameterList(Depth, TemplateParams)) {
    if(!Tok.is(tok::greater)) {
      Diag(Tok.getLocation(), diag::err_expected_greater);
      return false;
    }
    RAngleLoc = ConsumeToken();
  }
  return true;
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
  while(1) {
    if (DeclTy* TmpParam 
          = ParseTemplateParameter(Depth, TemplateParams.size())) {
      TemplateParams.push_back(TmpParam);
    } else {
      // If we failed to parse a template parameter, skip until we find
      // a comma or closing brace.
      SkipUntil(tok::comma, tok::greater, true, true);
    }
    
    // Did we find a comma or the end of the template parmeter list?
    if(Tok.is(tok::comma)) {
      ConsumeToken();
    } else if(Tok.is(tok::greater)) {
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

/// ParseTemplateParameter - Parse a template-parameter (C++ [temp.param]).
///
///       template-parameter: [C++ temp.param]
///         type-parameter
///         parameter-declaration
///
///       type-parameter: (see below)
///         'class' identifier[opt]
///         'class' identifier[opt] '=' type-id
///         'typename' identifier[opt]
///         'typename' identifier[opt] '=' type-id
///         'template' '<' template-parameter-list '>' 'class' identifier[opt]
///         'template' '<' template-parameter-list '>' 'class' identifier[opt] = id-expression
Parser::DeclTy *
Parser::ParseTemplateParameter(unsigned Depth, unsigned Position) {
  if(Tok.is(tok::kw_class) ||
     (Tok.is(tok::kw_typename) && 
         // FIXME: Next token has not been annotated!
	 NextToken().isNot(tok::annot_typename))) {
    return ParseTypeParameter(Depth, Position);
  }
  
  if(Tok.is(tok::kw_template))
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
///         'class' identifier[opt]
///         'class' identifier[opt] '=' type-id
///         'typename' identifier[opt]
///         'typename' identifier[opt] '=' type-id
Parser::DeclTy *Parser::ParseTypeParameter(unsigned Depth, unsigned Position) {
  assert((Tok.is(tok::kw_class) || Tok.is(tok::kw_typename)) &&
	 "A type-parameter starts with 'class' or 'typename'");

  // Consume the 'class' or 'typename' keyword.
  bool TypenameKeyword = Tok.is(tok::kw_typename);
  SourceLocation KeyLoc = ConsumeToken();

  // Grab the template parameter name (if given)
  SourceLocation NameLoc;
  IdentifierInfo* ParamName = 0;
  if(Tok.is(tok::identifier)) {
    ParamName = Tok.getIdentifierInfo();
    NameLoc = ConsumeToken();
  } else if(Tok.is(tok::equal) || Tok.is(tok::comma) ||
	    Tok.is(tok::greater)) {
    // Unnamed template parameter. Don't have to do anything here, just
    // don't consume this token.
  } else {
    Diag(Tok.getLocation(), diag::err_expected_ident);
    return 0;
  }
  
  DeclTy *TypeParam = Actions.ActOnTypeParameter(CurScope, TypenameKeyword,
                                                 KeyLoc, ParamName, NameLoc,
                                                 Depth, Position);

  // Grab a default type id (if given).
  if(Tok.is(tok::equal)) {
    SourceLocation EqualLoc = ConsumeToken();
    SourceLocation DefaultLoc = Tok.getLocation();
    if (TypeTy *DefaultType = ParseTypeName())
      Actions.ActOnTypeParameterDefault(TypeParam, EqualLoc, DefaultLoc,
                                        DefaultType);
  }
  
  return TypeParam;
}

/// ParseTemplateTemplateParameter - Handle the parsing of template
/// template parameters. 
///
///       type-parameter:    [C++ temp.param]
///         'template' '<' template-parameter-list '>' 'class' identifier[opt]
///         'template' '<' template-parameter-list '>' 'class' identifier[opt] = id-expression
Parser::DeclTy * 
Parser::ParseTemplateTemplateParameter(unsigned Depth, unsigned Position) {
  assert(Tok.is(tok::kw_template) && "Expected 'template' keyword");

  // Handle the template <...> part.
  SourceLocation TemplateLoc = ConsumeToken();
  TemplateParameterList TemplateParams; 
  SourceLocation LAngleLoc, RAngleLoc;
  {
    ParseScope TemplateParmScope(this, Scope::TemplateParamScope);
    if(!ParseTemplateParameters(Depth + 1, TemplateParams, LAngleLoc,
                                RAngleLoc)) {
      return 0;
    }
  }

  // Generate a meaningful error if the user forgot to put class before the
  // identifier, comma, or greater.
  if(!Tok.is(tok::kw_class)) {
    Diag(Tok.getLocation(), diag::err_expected_class_before) 
      << PP.getSpelling(Tok);
    return 0;
  }
  SourceLocation ClassLoc = ConsumeToken();

  // Get the identifier, if given.
  SourceLocation NameLoc;
  IdentifierInfo* ParamName = 0;
  if(Tok.is(tok::identifier)) {
    ParamName = Tok.getIdentifierInfo();
    NameLoc = ConsumeToken();
  } else if(Tok.is(tok::equal) || Tok.is(tok::comma) || Tok.is(tok::greater)) {
    // Unnamed template parameter. Don't have to do anything here, just
    // don't consume this token.
  } else {
    Diag(Tok.getLocation(), diag::err_expected_ident);
    return 0;
  }

  TemplateParamsTy *ParamList = 
    Actions.ActOnTemplateParameterList(Depth, SourceLocation(),
                                       TemplateLoc, LAngleLoc,
                                       &TemplateParams[0], 
                                       TemplateParams.size(),
                                       RAngleLoc);

  Parser::DeclTy * Param
    = Actions.ActOnTemplateTemplateParameter(CurScope, TemplateLoc,
                                             ParamList, ParamName,
                                             NameLoc, Depth, Position);

  // Get the a default value, if given.
  if (Tok.is(tok::equal)) {
    SourceLocation EqualLoc = ConsumeToken();
    OwningExprResult DefaultExpr = ParseCXXIdExpression();
    if (DefaultExpr.isInvalid())
      return Param;
    else if (Param)
      Actions.ActOnTemplateTemplateParameterDefault(Param, EqualLoc,
                                                    move(DefaultExpr));
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
Parser::DeclTy * 
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
    return 0;
  }

  // Create the parameter. 
  DeclTy *Param = Actions.ActOnNonTypeTemplateParameter(CurScope, ParamDecl,
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

/// AnnotateTemplateIdToken - The current token is an identifier that
/// refers to the template declaration Template, and is followed by a
/// '<'. Turn this template-id into a template-id annotation token.
void Parser::AnnotateTemplateIdToken(DeclTy *Template, TemplateNameKind TNK,
                                     const CXXScopeSpec *SS) {
  assert(getLang().CPlusPlus && "Can only annotate template-ids in C++");
  assert(Template && Tok.is(tok::identifier) && NextToken().is(tok::less) &&
         "Parser isn't at the beginning of a template-id");

  // Consume the template-name.
  SourceLocation TemplateNameLoc = ConsumeToken();

  // Consume the '<'.
  SourceLocation LAngleLoc = ConsumeToken();

  // Parse the optional template-argument-list.
  TemplateArgList TemplateArgs;
  TemplateArgIsTypeList TemplateArgIsType;
  TemplateArgLocationList TemplateArgLocations;

  {
    GreaterThanIsOperatorScope G(GreaterThanIsOperator, false);
    if (Tok.isNot(tok::greater) && 
        ParseTemplateArgumentList(TemplateArgs, TemplateArgIsType,
                                  TemplateArgLocations)) {
      // Try to find the closing '>'.
      SkipUntil(tok::greater, true, true);

      // Clean up any template arguments that we successfully parsed.
      ASTTemplateArgsPtr TemplateArgsPtr(Actions, &TemplateArgs[0],
                                         &TemplateArgIsType[0],
                                         TemplateArgs.size());
      
      return;
    }
  }

  if (Tok.isNot(tok::greater))
    return;

  // Determine the location of the '>'. We won't actually consume this
  // token, because we'll be replacing it with the template-id.
  SourceLocation RAngleLoc = Tok.getLocation();
  
  // Build the annotation token.
  if (TNK == Action::TNK_Function_template) {
    // This is a function template. We'll be building a template-id
    // annotation token.
    Tok.setKind(tok::annot_template_id);    
    TemplateIdAnnotation *TemplateId 
      = (TemplateIdAnnotation *)malloc(sizeof(TemplateIdAnnotation) + 
                                  sizeof(void*) * TemplateArgs.size());
    TemplateId->TemplateNameLoc = TemplateNameLoc;
    TemplateId->Template = Template;
    TemplateId->LAngleLoc = LAngleLoc;
    TemplateId->NumArgs = TemplateArgs.size();
    void **Args = (void**)(TemplateId + 1);
    for (unsigned Arg = 0, ArgEnd = TemplateArgs.size(); Arg != ArgEnd; ++Arg)
      Args[Arg] = TemplateArgs[Arg];
    Tok.setAnnotationValue(TemplateId);
  } else {
    // This is a type template, e.g., a class template, template
    // template parameter, or template alias. We'll be building a
    // "typename" annotation token.
    ASTTemplateArgsPtr TemplateArgsPtr(Actions, &TemplateArgs[0],
                                       &TemplateArgIsType[0],
                                       TemplateArgs.size());
    TypeTy *Ty 
      = Actions.ActOnClassTemplateSpecialization(Template, TemplateNameLoc,
                                                 LAngleLoc, TemplateArgsPtr,
                                                 &TemplateArgLocations[0],
                                                 RAngleLoc, SS);

    if (!Ty) // Something went wrong; don't annotate
      return;

    Tok.setKind(tok::annot_typename);
    Tok.setAnnotationValue(Ty);
  }

  // Common fields for the annotation token
  Tok.setAnnotationEndLoc(RAngleLoc);
  Tok.setLocation(TemplateNameLoc);
  if (SS && SS->isNotEmpty())
    Tok.setLocation(SS->getBeginLoc());

  // In case the tokens were cached, have Preprocessor replace them with the
  // annotation token.
  PP.AnnotateCachedTokens(Tok);
}

/// ParseTemplateArgument - Parse a C++ template argument (C++ [temp.names]).
///
///       template-argument: [C++ 14.2]
///         assignment-expression
///         type-id
///         id-expression
void *Parser::ParseTemplateArgument(bool &ArgIsType) {
  // C++ [temp.arg]p2:
  //   In a template-argument, an ambiguity between a type-id and an
  //   expression is resolved to a type-id, regardless of the form of
  //   the corresponding template-parameter.
  //
  // Therefore, we initially try to parse a type-id.
  if (isCXXTypeId(TypeIdAsTemplateArgument)) {
    ArgIsType = true;
    return ParseTypeName();
  }

  OwningExprResult ExprArg = ParseAssignmentExpression();
  if (ExprArg.isInvalid() || !ExprArg.get())
    return 0;

  ArgIsType = false;
  return ExprArg.release();
}

/// ParseTemplateArgumentList - Parse a C++ template-argument-list
/// (C++ [temp.names]). Returns true if there was an error.
///
///       template-argument-list: [C++ 14.2]
///         template-argument
///         template-argument-list ',' template-argument
bool 
Parser::ParseTemplateArgumentList(TemplateArgList &TemplateArgs,
                                  TemplateArgIsTypeList &TemplateArgIsType,
                              TemplateArgLocationList &TemplateArgLocations) {
  while (true) {
    bool IsType = false;
    SourceLocation Loc = Tok.getLocation();
    void *Arg = ParseTemplateArgument(IsType);
    if (Arg) {
      TemplateArgs.push_back(Arg);
      TemplateArgIsType.push_back(IsType);
      TemplateArgLocations.push_back(Loc);
    } else {
      SkipUntil(tok::comma, tok::greater, true, true);
      return true;
    }

    // If the next token is a comma, consume it and keep reading
    // arguments.
    if (Tok.isNot(tok::comma)) break;

    // Consume the comma.
    ConsumeToken();
  }

  return Tok.isNot(tok::greater);
}

