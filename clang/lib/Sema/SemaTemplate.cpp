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
#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/Parse/DeclSpec.h"
#include "clang/Basic/LangOptions.h"

using namespace clang;

/// isTemplateName - Determines whether the identifier II is a
/// template name in the current scope, and returns the template
/// declaration if II names a template. An optional CXXScope can be
/// passed to indicate the C++ scope in which the identifier will be
/// found. 
Sema::TemplateNameKind Sema::isTemplateName(IdentifierInfo &II, Scope *S,
                                            DeclTy *&Template,
                                            const CXXScopeSpec *SS) {
  NamedDecl *IIDecl = LookupParsedName(S, SS, &II, LookupOrdinaryName);

  if (IIDecl) {
    if (isa<TemplateDecl>(IIDecl)) {
      Template = IIDecl;
      if (isa<FunctionTemplateDecl>(IIDecl))
        return TNK_Function_template;
      else if (isa<ClassTemplateDecl>(IIDecl))
        return TNK_Class_template;
      else if (isa<TemplateTemplateParmDecl>(IIDecl))
        return TNK_Template_template_parm;
      else
        assert(false && "Unknown TemplateDecl");
    }

    // FIXME: What follows is a gross hack.
    if (FunctionDecl *FD = dyn_cast<FunctionDecl>(IIDecl)) {
      if (FD->getType()->isDependentType()) {
        Template = FD;
        return TNK_Function_template;
      }
    } else if (OverloadedFunctionDecl *Ovl 
                 = dyn_cast<OverloadedFunctionDecl>(IIDecl)) {
      for (OverloadedFunctionDecl::function_iterator F = Ovl->function_begin(),
                                                  FEnd = Ovl->function_end();
           F != FEnd; ++F) {
        if ((*F)->getType()->isDependentType()) {
          Template = Ovl;
          return TNK_Function_template;
        }
      }
    }
  }
  return TNK_Non_template;
}

/// DiagnoseTemplateParameterShadow - Produce a diagnostic complaining
/// that the template parameter 'PrevDecl' is being shadowed by a new
/// declaration at location Loc. Returns true to indicate that this is
/// an error, and false otherwise.
bool Sema::DiagnoseTemplateParameterShadow(SourceLocation Loc, Decl *PrevDecl) {
  assert(PrevDecl->isTemplateParameter() && "Not a template parameter");

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

/// AdjustDeclForTemplates - If the given decl happens to be a template, reset
/// the parameter D to reference the templated declaration and return a pointer
/// to the template declaration. Otherwise, do nothing to D and return null.
TemplateDecl *Sema::AdjustDeclIfTemplate(DeclTy *&D)
{
  if(TemplateDecl *Temp = dyn_cast<TemplateDecl>(static_cast<Decl*>(D))) {
    D = Temp->getTemplatedDecl();
    return Temp;
  }
  return 0;
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
				       SourceLocation ParamNameLoc,
                                       unsigned Depth, unsigned Position) {
  assert(S->isTemplateParamScope() && 
	 "Template type parameter not in template parameter scope!");
  bool Invalid = false;

  if (ParamName) {
    NamedDecl *PrevDecl = LookupName(S, ParamName, LookupTagName);
    if (PrevDecl && PrevDecl->isTemplateParameter())
      Invalid = Invalid || DiagnoseTemplateParameterShadow(ParamNameLoc,
							   PrevDecl);
  }

  SourceLocation Loc = ParamNameLoc;
  if (!ParamName)
    Loc = KeyLoc;

  TemplateTypeParmDecl *Param
    = TemplateTypeParmDecl::Create(Context, CurContext, Loc, 
                                   Depth, Position, ParamName, Typename);
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
Sema::DeclTy *Sema::ActOnNonTypeTemplateParameter(Scope *S, Declarator &D,
                                                  unsigned Depth, 
                                                  unsigned Position) {
  QualType T = GetTypeForDeclarator(D, S);

  assert(S->isTemplateParamScope() &&
         "Non-type template parameter not in template parameter scope!");
  bool Invalid = false;

  IdentifierInfo *ParamName = D.getIdentifier();
  if (ParamName) {
    NamedDecl *PrevDecl = LookupName(S, ParamName, LookupTagName);
    if (PrevDecl && PrevDecl->isTemplateParameter())
      Invalid = Invalid || DiagnoseTemplateParameterShadow(D.getIdentifierLoc(),
                                                           PrevDecl);
  }

  NonTypeTemplateParmDecl *Param
    = NonTypeTemplateParmDecl::Create(Context, CurContext, D.getIdentifierLoc(),
                                      Depth, Position, ParamName, T);
  if (Invalid)
    Param->setInvalidDecl();

  if (D.getIdentifier()) {
    // Add the template parameter into the current scope.
    S->AddDecl(Param);
    IdResolver.AddDecl(Param);
  }
  return Param;
}


/// ActOnTemplateTemplateParameter - Called when a C++ template template
/// parameter (e.g. T in template <template <typename> class T> class array)
/// has been parsed. S is the current scope.
Sema::DeclTy *Sema::ActOnTemplateTemplateParameter(Scope* S,
                                                   SourceLocation TmpLoc,
                                                   TemplateParamsTy *Params,
                                                   IdentifierInfo *Name,
                                                   SourceLocation NameLoc,
                                                   unsigned Depth,
                                                   unsigned Position)
{
  assert(S->isTemplateParamScope() &&
         "Template template parameter not in template parameter scope!");

  // Construct the parameter object.
  TemplateTemplateParmDecl *Param =
    TemplateTemplateParmDecl::Create(Context, CurContext, TmpLoc, Depth,
                                     Position, Name,
                                     (TemplateParameterList*)Params);

  // Make sure the parameter is valid.
  // FIXME: Decl object is not currently invalidated anywhere so this doesn't
  // do anything yet. However, if the template parameter list or (eventual)
  // default value is ever invalidated, that will propagate here.
  bool Invalid = false;
  if (Invalid) {
    Param->setInvalidDecl();
  }

  // If the tt-param has a name, then link the identifier into the scope
  // and lookup mechanisms.
  if (Name) {
    S->AddDecl(Param);
    IdResolver.AddDecl(Param);
  }

  return Param;
}

/// ActOnTemplateParameterList - Builds a TemplateParameterList that
/// contains the template parameters in Params/NumParams.
Sema::TemplateParamsTy *
Sema::ActOnTemplateParameterList(unsigned Depth,
                                 SourceLocation ExportLoc,
                                 SourceLocation TemplateLoc, 
                                 SourceLocation LAngleLoc,
                                 DeclTy **Params, unsigned NumParams,
                                 SourceLocation RAngleLoc) {
  if (ExportLoc.isValid())
    Diag(ExportLoc, diag::note_template_export_unsupported);

  return TemplateParameterList::Create(Context, TemplateLoc, LAngleLoc,
                                       (Decl**)Params, NumParams, RAngleLoc);
}

Sema::DeclTy *
Sema::ActOnClassTemplate(Scope *S, unsigned TagSpec, TagKind TK,
                         SourceLocation KWLoc, const CXXScopeSpec &SS,
                         IdentifierInfo *Name, SourceLocation NameLoc,
                         AttributeList *Attr,
                         MultiTemplateParamsArg TemplateParameterLists) {
  assert(TemplateParameterLists.size() > 0 && "No template parameter lists?");
  assert(TK != TK_Reference && "Can only declare or define class templates");

  // Check that we can declare a template here.
  if (CheckTemplateDeclScope(S, TemplateParameterLists))
    return 0;

  TagDecl::TagKind Kind;
  switch (TagSpec) {
  default: assert(0 && "Unknown tag type!");
  case DeclSpec::TST_struct: Kind = TagDecl::TK_struct; break;
  case DeclSpec::TST_union:  Kind = TagDecl::TK_union; break;
  case DeclSpec::TST_class:  Kind = TagDecl::TK_class; break;
  }

  // There is no such thing as an unnamed class template.
  if (!Name) {
    Diag(KWLoc, diag::err_template_unnamed_class);
    return 0;
  }

  // Find any previous declaration with this name.
  LookupResult Previous = LookupParsedName(S, &SS, Name, LookupOrdinaryName,
                                           true);
  assert(!Previous.isAmbiguous() && "Ambiguity in class template redecl?");
  NamedDecl *PrevDecl = 0;
  if (Previous.begin() != Previous.end())
    PrevDecl = *Previous.begin();

  DeclContext *SemanticContext = CurContext;
  if (SS.isNotEmpty() && !SS.isInvalid()) {
    SemanticContext = static_cast<DeclContext*>(SS.getScopeRep());

    // FIXME: need to match up several levels of template parameter
    // lists here.
  }

  // FIXME: member templates!
  TemplateParameterList *TemplateParams 
    = static_cast<TemplateParameterList *>(*TemplateParameterLists.release());

  // If there is a previous declaration with the same name, check
  // whether this is a valid redeclaration.
  ClassTemplateDecl *PrevClassTemplate 
    = dyn_cast_or_null<ClassTemplateDecl>(PrevDecl);
  if (PrevClassTemplate) {
    // Ensure that the template parameter lists are compatible.
    if (!TemplateParameterListsAreEqual(TemplateParams,
                                   PrevClassTemplate->getTemplateParameters(),
                                        /*Complain=*/true))
      return 0;

    // C++ [temp.class]p4:
    //   In a redeclaration, partial specialization, explicit
    //   specialization or explicit instantiation of a class template,
    //   the class-key shall agree in kind with the original class
    //   template declaration (7.1.5.3).
    RecordDecl *PrevRecordDecl = PrevClassTemplate->getTemplatedDecl();
    if (PrevRecordDecl->getTagKind() != Kind) {
      Diag(KWLoc, diag::err_use_with_wrong_tag) << Name;
      Diag(PrevRecordDecl->getLocation(), diag::note_previous_use);
      return 0;
    }


    // Check for redefinition of this class template.
    if (TK == TK_Definition) {
      if (TagDecl *Def = PrevRecordDecl->getDefinition(Context)) {
        Diag(NameLoc, diag::err_redefinition) << Name;
        Diag(Def->getLocation(), diag::note_previous_definition);
        // FIXME: Would it make sense to try to "forget" the previous
        // definition, as part of error recovery?
        return 0;
      }
    }
  } else if (PrevDecl && PrevDecl->isTemplateParameter()) {
    // Maybe we will complain about the shadowed template parameter.
    DiagnoseTemplateParameterShadow(NameLoc, PrevDecl);
    // Just pretend that we didn't see the previous declaration.
    PrevDecl = 0;
  } else if (PrevDecl) {
    // C++ [temp]p5:
    //   A class template shall not have the same name as any other
    //   template, class, function, object, enumeration, enumerator,
    //   namespace, or type in the same scope (3.3), except as specified
    //   in (14.5.4).
    Diag(NameLoc, diag::err_redefinition_different_kind) << Name;
    Diag(PrevDecl->getLocation(), diag::note_previous_definition);
    return 0;
  }

  // If we had a scope specifier, we better have a previous template
  // declaration!

  TagDecl *NewClass = 
    CXXRecordDecl::Create(Context, Kind, SemanticContext, NameLoc, Name,
                          PrevClassTemplate? 
                            PrevClassTemplate->getTemplatedDecl() : 0);

  ClassTemplateDecl *NewTemplate
    = ClassTemplateDecl::Create(Context, SemanticContext, NameLoc,
                                DeclarationName(Name), TemplateParams,
                                NewClass);
  
  // Set the lexical context of these templates
  NewClass->setLexicalDeclContext(CurContext);
  NewTemplate->setLexicalDeclContext(CurContext);

  if (TK == TK_Definition)
    NewClass->startDefinition();

  if (Attr)
    ProcessDeclAttributeList(NewClass, Attr);

  PushOnScopeChains(NewTemplate, S);

  return NewTemplate;
}



Action::TypeTy * 
Sema::ActOnClassTemplateSpecialization(DeclTy *TemplateD,
                                       SourceLocation TemplateLoc,
                                       SourceLocation LAngleLoc,
                                       ASTTemplateArgsPtr TemplateArgs,
                                       SourceLocation *TemplateArgLocs,
                                       SourceLocation RAngleLoc,
                                       const CXXScopeSpec *SS) {
  TemplateDecl *Template = cast<TemplateDecl>(static_cast<Decl *>(TemplateD));

  // Check that the template argument list is well-formed for this
  // template.
  if (!CheckTemplateArgumentList(Template, TemplateLoc, LAngleLoc, 
                                 TemplateArgs, TemplateArgLocs, RAngleLoc))
    return 0;

  // Yes, all class template specializations are just silly sugar for
  // 'int'. Gotta problem wit dat?
  QualType Result
    = Context.getClassTemplateSpecializationType(Template, 
                                                 TemplateArgs.size(),
                    reinterpret_cast<uintptr_t *>(TemplateArgs.getArgs()), 
                                                 TemplateArgs.getArgIsType(),
                                                 Context.IntTy);
  TemplateArgs.release();
  return Result.getAsOpaquePtr();
}

/// \brief Check that the given template argument list is well-formed
/// for specializing the given template.
bool Sema::CheckTemplateArgumentList(TemplateDecl *Template,
                                     SourceLocation TemplateLoc,
                                     SourceLocation LAngleLoc,
                                     ASTTemplateArgsPtr& Args,
                                     SourceLocation *TemplateArgLocs,
                                     SourceLocation RAngleLoc) {
  TemplateParameterList *Params = Template->getTemplateParameters();
  unsigned NumParams = Params->size();
  unsigned NumArgs = Args.size();
  bool Invalid = false;

  if (NumArgs > NumParams ||
      NumArgs < NumParams /*FIXME: default arguments! */) {
    // FIXME: point at either the first arg beyond what we can handle,
    // or the '>', depending on whether we have too many or too few
    // arguments.
    SourceRange Range;
    if (NumArgs > NumParams)
      Range = SourceRange(TemplateArgLocs[NumParams], RAngleLoc);
    Diag(TemplateLoc, diag::err_template_arg_list_different_arity)
      << (NumArgs > NumParams)
      << (isa<ClassTemplateDecl>(Template)? 0 :
          isa<FunctionTemplateDecl>(Template)? 1 :
          isa<TemplateTemplateParmDecl>(Template)? 2 : 3)
      << Template << Range;

    Invalid = true;
  }
  
  // C++ [temp.arg]p1: 
  //   [...] The type and form of each template-argument specified in
  //   a template-id shall match the type and form specified for the
  //   corresponding parameter declared by the template in its
  //   template-parameter-list.
  unsigned ArgIdx = 0;
  for (TemplateParameterList::iterator Param = Params->begin(),
                                       ParamEnd = Params->end();
       Param != ParamEnd; ++Param, ++ArgIdx) {
    // Decode the template argument
    QualType ArgType;
    Expr *ArgExpr = 0;
    SourceLocation ArgLoc;
    if (ArgIdx >= NumArgs) {
      // FIXME: Get the default argument here, which might
      // (eventually) require instantiation.
      break;
    } else
      ArgLoc = TemplateArgLocs[ArgIdx];

    if (Args.getArgIsType()[ArgIdx])
      ArgType = QualType::getFromOpaquePtr(Args.getArgs()[ArgIdx]);
    else
      ArgExpr = reinterpret_cast<Expr *>(Args.getArgs()[ArgIdx]);

    if (TemplateTypeParmDecl *TTP = dyn_cast<TemplateTypeParmDecl>(*Param)) {
      // Check template type parameters.
      if (!ArgType.isNull()) {
        if (!CheckTemplateArgument(TTP, ArgType, ArgLoc))
          Invalid = true;
        continue;
      }

      // C++ [temp.arg.type]p1:
      //   A template-argument for a template-parameter which is a
      //   type shall be a type-id.

      // We have a template type parameter but the template argument
      // is an expression.
      Diag(ArgExpr->getSourceRange().getBegin(), 
           diag::err_template_arg_must_be_type);
      Diag((*Param)->getLocation(), diag::note_template_parameter_here);
      Invalid = true;
    } else if (NonTypeTemplateParmDecl *NTTP 
                 = dyn_cast<NonTypeTemplateParmDecl>(*Param)) {
      // Check non-type template parameters.
      if (ArgExpr) {
        if (!CheckTemplateArgument(NTTP, ArgExpr))
          Invalid = true;
        continue;
      }

      // We have a non-type template parameter but the template
      // argument is a type.

      // C++ [temp.arg]p2:
      //   In a template-argument, an ambiguity between a type-id and
      //   an expression is resolved to a type-id, regardless of the
      //   form of the corresponding template-parameter.
      //
      // We warn specifically about this case, since it can be rather
      // confusing for users.
      if (ArgType->isFunctionType())
        Diag(ArgLoc, diag::err_template_arg_nontype_ambig)
          << ArgType;
      else
        Diag(ArgLoc, diag::err_template_arg_must_be_expr);
      Diag((*Param)->getLocation(), diag::note_template_parameter_here);
      Invalid = true;
    } else { 
      // Check template template parameters.
      TemplateTemplateParmDecl *TempParm 
        = cast<TemplateTemplateParmDecl>(*Param);
     
      if (ArgExpr && isa<DeclRefExpr>(ArgExpr) &&
          isa<TemplateDecl>(cast<DeclRefExpr>(ArgExpr)->getDecl())) {
        if (!CheckTemplateArgument(TempParm, cast<DeclRefExpr>(ArgExpr)))
          Invalid = true;
        continue;
      }

      // We have a template template parameter but the template
      // argument does not refer to a template.
      Diag(ArgLoc, diag::err_template_arg_must_be_template);
      Invalid = true;
    }
  }

  return Invalid;
}

/// \brief Check a template argument against its corresponding
/// template type parameter.
///
/// This routine implements the semantics of C++ [temp.arg.type]. It
/// returns true if an error occurred, and false otherwise.
bool Sema::CheckTemplateArgument(TemplateTypeParmDecl *Param, 
                                 QualType Arg, SourceLocation ArgLoc) {
  // C++ [temp.arg.type]p2:
  //   A local type, a type with no linkage, an unnamed type or a type
  //   compounded from any of these types shall not be used as a
  //   template-argument for a template type-parameter.
  //
  // FIXME: Perform the recursive and no-linkage type checks.
  const TagType *Tag = 0;
  if (const EnumType *EnumT = Arg->getAsEnumType())
    Tag = EnumT;
  else if (const RecordType *RecordT = Arg->getAsRecordType())
    Tag = RecordT;
  if (Tag && Tag->getDecl()->getDeclContext()->isFunctionOrMethod())
    return Diag(ArgLoc, diag::err_template_arg_local_type)
      << QualType(Tag, 0);
  else if (Tag && !Tag->getDecl()->getDeclName()) {
    Diag(ArgLoc, diag::err_template_arg_unnamed_type);
    Diag(Tag->getDecl()->getLocation(), diag::note_template_unnamed_type_here);
    return true;
  }

  return false;
}

/// \brief Check a template argument against its corresponding
/// non-type template parameter.
///
/// This routine implements the semantics of C++ [temp.arg.nontype]. 
/// It returns true if an error occurred, and false otherwise.
bool Sema::CheckTemplateArgument(NonTypeTemplateParmDecl *Param,
                                 Expr *Arg) {
  return false;
}

/// \brief Check a template argument against its corresponding
/// template template parameter.
///
/// This routine implements the semantics of C++ [temp.arg.template].
/// It returns true if an error occurred, and false otherwise.
bool Sema::CheckTemplateArgument(TemplateTemplateParmDecl *Param,
                                 DeclRefExpr *Arg) {
  return false;
}

/// \brief Determine whether the given template parameter lists are
/// equivalent.
///
/// \param New  The new template parameter list, typically written in the 
/// source code as part of a new template declaration.
///
/// \param Old  The old template parameter list, typically found via
/// name lookup of the template declared with this template parameter
/// list.
///
/// \param Complain  If true, this routine will produce a diagnostic if
/// the template parameter lists are not equivalent.
///
/// \returns True if the template parameter lists are equal, false
/// otherwise.
bool 
Sema::TemplateParameterListsAreEqual(TemplateParameterList *New,
                                     TemplateParameterList *Old,
                                     bool Complain,
                                     bool IsTemplateTemplateParm) {
  if (Old->size() != New->size()) {
    if (Complain) {
      Diag(New->getTemplateLoc(), diag::err_template_param_list_different_arity)
        << (New->size() > Old->size())
        << IsTemplateTemplateParm
        << SourceRange(New->getTemplateLoc(), New->getRAngleLoc());
      Diag(Old->getTemplateLoc(), diag::note_template_prev_declaration)
        << IsTemplateTemplateParm
        << SourceRange(Old->getTemplateLoc(), Old->getRAngleLoc());
    }

    return false;
  }

  for (TemplateParameterList::iterator OldParm = Old->begin(),
         OldParmEnd = Old->end(), NewParm = New->begin();
       OldParm != OldParmEnd; ++OldParm, ++NewParm) {
    if ((*OldParm)->getKind() != (*NewParm)->getKind()) {
      Diag((*NewParm)->getLocation(), diag::err_template_param_different_kind)
        << IsTemplateTemplateParm;
      Diag((*OldParm)->getLocation(), diag::note_template_prev_declaration)
        << IsTemplateTemplateParm;
      return false;
    }

    if (isa<TemplateTypeParmDecl>(*OldParm)) {
      // Okay; all template type parameters are equivalent (since we
      // know we're at the same depth/level).
#ifndef NDEBUG
      QualType OldParmType 
        = Context.getTypeDeclType(cast<TemplateTypeParmDecl>(*OldParm));
      QualType NewParmType 
        = Context.getTypeDeclType(cast<TemplateTypeParmDecl>(*NewParm));
      assert(Context.getCanonicalType(OldParmType) == 
             Context.getCanonicalType(NewParmType) && 
             "type parameter mismatch?");
#endif
    } else if (NonTypeTemplateParmDecl *OldNTTP 
                 = dyn_cast<NonTypeTemplateParmDecl>(*OldParm)) {
      // The types of non-type template parameters must agree.
      NonTypeTemplateParmDecl *NewNTTP
        = cast<NonTypeTemplateParmDecl>(*NewParm);
      if (Context.getCanonicalType(OldNTTP->getType()) !=
            Context.getCanonicalType(NewNTTP->getType())) {
        if (Complain) {
          Diag(NewNTTP->getLocation(), 
               diag::err_template_nontype_parm_different_type)
            << NewNTTP->getType()
            << IsTemplateTemplateParm;
          Diag(OldNTTP->getLocation(), 
               diag::note_template_nontype_parm_prev_declaration)
            << OldNTTP->getType();
        }
        return false;
      }
    } else {
      // The template parameter lists of template template
      // parameters must agree.
      // FIXME: Could we perform a faster "type" comparison here?
      assert(isa<TemplateTemplateParmDecl>(*OldParm) && 
             "Only template template parameters handled here");
      TemplateTemplateParmDecl *OldTTP 
        = cast<TemplateTemplateParmDecl>(*OldParm);
      TemplateTemplateParmDecl *NewTTP
        = cast<TemplateTemplateParmDecl>(*NewParm);
      if (!TemplateParameterListsAreEqual(NewTTP->getTemplateParameters(),
                                          OldTTP->getTemplateParameters(),
                                          Complain,
                                          /*IsTemplateTemplateParm=*/true))
        return false;
    }
  }

  return true;
}

/// \brief Check whether a template can be declared within this scope.
///
/// If the template declaration is valid in this scope, returns
/// false. Otherwise, issues a diagnostic and returns true.
bool 
Sema::CheckTemplateDeclScope(Scope *S, 
                             MultiTemplateParamsArg &TemplateParameterLists) {
  assert(TemplateParameterLists.size() > 0 && "Not a template");

  // Find the nearest enclosing declaration scope.
  while ((S->getFlags() & Scope::DeclScope) == 0 ||
         (S->getFlags() & Scope::TemplateParamScope) != 0)
    S = S->getParent();
  
  TemplateParameterList *TemplateParams = 
    static_cast<TemplateParameterList*>(*TemplateParameterLists.get());
  SourceLocation TemplateLoc = TemplateParams->getTemplateLoc();
  SourceRange TemplateRange 
    = SourceRange(TemplateLoc, TemplateParams->getRAngleLoc());

  // C++ [temp]p2:
  //   A template-declaration can appear only as a namespace scope or
  //   class scope declaration.
  DeclContext *Ctx = static_cast<DeclContext *>(S->getEntity());
  while (Ctx && isa<LinkageSpecDecl>(Ctx)) {
    if (cast<LinkageSpecDecl>(Ctx)->getLanguage() != LinkageSpecDecl::lang_cxx)
      return Diag(TemplateLoc, diag::err_template_linkage)
        << TemplateRange;

    Ctx = Ctx->getParent();
  }

  if (Ctx && (Ctx->isFileContext() || Ctx->isRecord()))
    return false;

  return Diag(TemplateLoc, diag::err_template_outside_namespace_or_class_scope)
    << TemplateRange;
}
