//===------- SemaTemplate.cpp - Semantic Analysis for C++ Templates -------===/

//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//+//===----------------------------------------------------------------------===/

//
//  This file implements semantic analysis for C++ templates.
//+//===----------------------------------------------------------------------===/

#include "Sema.h"
#include "clang/AST/Expr.h"
#include "clang/Parse/DeclSpec.h"
#include "clang/Basic/LangOptions.h"

using namespace clang;

/// isTemplateParameterDecl - Determines whether the given declaration
/// 'D' names a template parameter.
bool Sema::isTemplateParameterDecl(Decl *D) {
  return isa<TemplateTypeParmDecl>(D) || isa<NonTypeTemplateParmDecl>(D);
}

/// DiagnoseTemplateParameterShadow - Produce a diagnostic complaining
/// that the template parameter 'PrevDecl' is being shadowed by a new
/// declaration at location Loc. Returns true to indicate that this is
/// an error, and false otherwise.
bool Sema::DiagnoseTemplateParameterShadow(SourceLocation Loc, Decl *PrevDecl) {
  assert(isTemplateParameterDecl(PrevDecl) && "Not a template parameter");

  // Microsoft Visual C++ permits template parameters to be shadowed.
  if (getLangOptions().Microsoft)
    return false;

  // C++ [temp.local]p4:
  //   A template-parameter shall not be redeclared within its
  //   scope (including nested scopes).
  Diag(Loc, diag::err_template_param_shadow) 
    << cast<NamedDecl>(PrevDecl)->getDeclName();
  Diag(PrevDecl->getLocation(), diag::note_template_param_here);
  return true;
}

/// ActOnTypeParameter - Called when a C++ template type parameter
/// (e.g., "typename T") has been parsed. Typename specifies whether
/// the keyword "typename" was used to declare the type parameter
/// (otherwise, "class" was used), and KeyLoc is the location of the
/// "class" or "typename" keyword. ParamName is the name of the
/// parameter (NULL indicates an unnamed template parameter) and
/// ParamName is the location of the parameter name (if any). 
/// If the type parameter has a default argument, it will be added
/// later via ActOnTypeParameterDefault.
Sema::DeclTy *Sema::ActOnTypeParameter(Scope *S, bool Typename, 
				       SourceLocation KeyLoc,
				       IdentifierInfo *ParamName,
				       SourceLocation ParamNameLoc) {
  assert(S->isTemplateParamScope() && 
	 "Template type parameter not in template parameter scope!");
  bool Invalid = false;

  if (ParamName) {
    Decl *PrevDecl = LookupDecl(ParamName, Decl::IDNS_Tag, S);
    if (PrevDecl && isTemplateParameterDecl(PrevDecl))
      Invalid = Invalid || DiagnoseTemplateParameterShadow(ParamNameLoc,
							   PrevDecl);
  }

  TemplateTypeParmDecl *Param
    = TemplateTypeParmDecl::Create(Context, CurContext, 
				   ParamNameLoc, ParamName, Typename);
  if (Invalid)
    Param->setInvalidDecl();

  if (ParamName) {
    // Add the template parameter into the current scope.
    S->AddDecl(Param);
    IdResolver.AddDecl(Param);
  }

  return Param;
}

/// ActOnNonTypeTemplateParameter - Called when a C++ non-type
/// template parameter (e.g., "int Size" in "template<int Size>
/// class Array") has been parsed. S is the current scope and D is
/// the parsed declarator.
Sema::DeclTy *Sema::ActOnNonTypeTemplateParameter(Scope *S, Declarator &D) {
  QualType T = GetTypeForDeclarator(D, S);

  assert(S->isTemplateParamScope() && 
	 "Template type parameter not in template parameter scope!");
  bool Invalid = false;

  IdentifierInfo *ParamName = D.getIdentifier();
  if (ParamName) {
    Decl *PrevDecl = LookupDecl(ParamName, Decl::IDNS_Tag, S);
    if (PrevDecl && isTemplateParameterDecl(PrevDecl))
      Invalid = Invalid || DiagnoseTemplateParameterShadow(D.getIdentifierLoc(),
							   PrevDecl);
  }

  NonTypeTemplateParmDecl *Param
    = NonTypeTemplateParmDecl::Create(Context, CurContext, D.getIdentifierLoc(),
				      ParamName, T);
  if (Invalid)
    Param->setInvalidDecl();

  if (D.getIdentifier()) {
    // Add the template parameter into the current scope.
    S->AddDecl(Param);
    IdResolver.AddDecl(Param);
  }
  return Param;
}
