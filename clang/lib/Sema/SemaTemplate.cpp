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
#include "clang/AST/ExprCXX.h"
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

/// ActOnTypeParameterDefault - Adds a default argument (the type
/// Default) to the given template type parameter (TypeParam). 
void Sema::ActOnTypeParameterDefault(DeclTy *TypeParam, 
                                     SourceLocation EqualLoc,
                                     SourceLocation DefaultLoc, 
                                     TypeTy *DefaultT) {
  TemplateTypeParmDecl *Parm 
    = cast<TemplateTypeParmDecl>(static_cast<Decl *>(TypeParam));
  QualType Default = QualType::getFromOpaquePtr(DefaultT);

  // C++ [temp.param]p14:
  //   A template-parameter shall not be used in its own default argument.
  // FIXME: Implement this check! Needs a recursive walk over the types.
  
  // Check the template argument itself.
  if (CheckTemplateArgument(Parm, Default, DefaultLoc)) {
    Parm->setInvalidDecl();
    return;
  }

  Parm->setDefaultArgument(Default, DefaultLoc, false);
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

  // C++ [temp.param]p4:
  //
  // A non-type template-parameter shall have one of the following
  // (optionally cv-qualified) types:
  //
  //       -- integral or enumeration type,
  if (T->isIntegralType() || T->isEnumeralType() ||
      //   -- pointer to object or pointer to function, 
      (T->isPointerType() && 
       (T->getAsPointerType()->getPointeeType()->isObjectType() ||
        T->getAsPointerType()->getPointeeType()->isFunctionType())) ||
      //   -- reference to object or reference to function, 
      T->isReferenceType() ||
      //   -- pointer to member.
      T->isMemberPointerType() ||
      // If T is a dependent type, we can't do the check now, so we
      // assume that it is well-formed.
      T->isDependentType()) {
    // Okay: The template parameter is well-formed.
  } 
  // C++ [temp.param]p8:
  //
  //   A non-type template-parameter of type "array of T" or
  //   "function returning T" is adjusted to be of type "pointer to
  //   T" or "pointer to function returning T", respectively.
  else if (T->isArrayType())
    // FIXME: Keep the type prior to promotion?
    T = Context.getArrayDecayedType(T);
  else if (T->isFunctionType())
    // FIXME: Keep the type prior to promotion?
    T = Context.getPointerType(T);
  else {
    Diag(D.getIdentifierLoc(), diag::err_template_nontype_parm_bad_type)
      << T;
    return 0;
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

/// \brief Adds a default argument to the given non-type template
/// parameter.
void Sema::ActOnNonTypeTemplateParameterDefault(DeclTy *TemplateParamD,
                                                SourceLocation EqualLoc,
                                                ExprArg DefaultE) {
  NonTypeTemplateParmDecl *TemplateParm 
    = cast<NonTypeTemplateParmDecl>(static_cast<Decl *>(TemplateParamD));
  Expr *Default = static_cast<Expr *>(DefaultE.get());
  
  // C++ [temp.param]p14:
  //   A template-parameter shall not be used in its own default argument.
  // FIXME: Implement this check! Needs a recursive walk over the types.
  
  // Check the well-formedness of the default template argument.
  if (CheckTemplateArgument(TemplateParm, Default)) {
    TemplateParm->setInvalidDecl();
    return;
  }

  TemplateParm->setDefaultArgument(static_cast<Expr *>(DefaultE.release()));
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

/// \brief Adds a default argument to the given template template
/// parameter.
void Sema::ActOnTemplateTemplateParameterDefault(DeclTy *TemplateParamD,
                                                 SourceLocation EqualLoc,
                                                 ExprArg DefaultE) {
  TemplateTemplateParmDecl *TemplateParm 
    = cast<TemplateTemplateParmDecl>(static_cast<Decl *>(TemplateParamD));

  // Since a template-template parameter's default argument is an
  // id-expression, it must be a DeclRefExpr.
  DeclRefExpr *Default 
    = cast<DeclRefExpr>(static_cast<Expr *>(DefaultE.get()));

  // C++ [temp.param]p14:
  //   A template-parameter shall not be used in its own default argument.
  // FIXME: Implement this check! Needs a recursive walk over the types.

  // Check the well-formedness of the template argument.
  if (!isa<TemplateDecl>(Default->getDecl())) {
    Diag(Default->getSourceRange().getBegin(), 
         diag::err_template_arg_must_be_template)
      << Default->getSourceRange();
    TemplateParm->setInvalidDecl();
    return;
  } 
  if (CheckTemplateArgument(TemplateParm, Default)) {
    TemplateParm->setInvalidDecl();
    return;
  }

  DefaultE.release();
  TemplateParm->setDefaultArgument(Default);
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
  bool Invalid = false;

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

  // Check the template parameter list of this declaration, possibly
  // merging in the template parameter list from the previous class
  // template declaration.
  if (CheckTemplateParameterList(TemplateParams,
            PrevClassTemplate? PrevClassTemplate->getTemplateParameters() : 0))
    Invalid = true;
    
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

  if (Invalid) {
    NewTemplate->setInvalidDecl();
    NewClass->setInvalidDecl();
  }
  return NewTemplate;
}

/// \brief Checks the validity of a template parameter list, possibly
/// considering the template parameter list from a previous
/// declaration.
///
/// If an "old" template parameter list is provided, it must be
/// equivalent (per TemplateParameterListsAreEqual) to the "new"
/// template parameter list.
///
/// \param NewParams Template parameter list for a new template
/// declaration. This template parameter list will be updated with any
/// default arguments that are carried through from the previous
/// template parameter list.
///
/// \param OldParams If provided, template parameter list from a
/// previous declaration of the same template. Default template
/// arguments will be merged from the old template parameter list to
/// the new template parameter list.
///
/// \returns true if an error occurred, false otherwise.
bool Sema::CheckTemplateParameterList(TemplateParameterList *NewParams,
                                      TemplateParameterList *OldParams) {
  bool Invalid = false;
  
  // C++ [temp.param]p10:
  //   The set of default template-arguments available for use with a
  //   template declaration or definition is obtained by merging the
  //   default arguments from the definition (if in scope) and all
  //   declarations in scope in the same way default function
  //   arguments are (8.3.6).
  bool SawDefaultArgument = false;
  SourceLocation PreviousDefaultArgLoc;

  // Dummy initialization to avoid warnings.
  TemplateParameterList::iterator OldParam = NewParams->end();
  if (OldParams)
    OldParam = OldParams->begin();

  for (TemplateParameterList::iterator NewParam = NewParams->begin(),
                                    NewParamEnd = NewParams->end();
       NewParam != NewParamEnd; ++NewParam) {
    // Variables used to diagnose redundant default arguments
    bool RedundantDefaultArg = false;
    SourceLocation OldDefaultLoc;
    SourceLocation NewDefaultLoc;

    // Variables used to diagnose missing default arguments
    bool MissingDefaultArg = false;

    // Merge default arguments for template type parameters.
    if (TemplateTypeParmDecl *NewTypeParm
          = dyn_cast<TemplateTypeParmDecl>(*NewParam)) {
      TemplateTypeParmDecl *OldTypeParm 
          = OldParams? cast<TemplateTypeParmDecl>(*OldParam) : 0;
      
      if (OldTypeParm && OldTypeParm->hasDefaultArgument() && 
          NewTypeParm->hasDefaultArgument()) {
        OldDefaultLoc = OldTypeParm->getDefaultArgumentLoc();
        NewDefaultLoc = NewTypeParm->getDefaultArgumentLoc();
        SawDefaultArgument = true;
        RedundantDefaultArg = true;
        PreviousDefaultArgLoc = NewDefaultLoc;
      } else if (OldTypeParm && OldTypeParm->hasDefaultArgument()) {
        // Merge the default argument from the old declaration to the
        // new declaration.
        SawDefaultArgument = true;
        NewTypeParm->setDefaultArgument(OldTypeParm->getDefaultArgument(),
                                        OldTypeParm->getDefaultArgumentLoc(),
                                        true);
        PreviousDefaultArgLoc = OldTypeParm->getDefaultArgumentLoc();
      } else if (NewTypeParm->hasDefaultArgument()) {
        SawDefaultArgument = true;
        PreviousDefaultArgLoc = NewTypeParm->getDefaultArgumentLoc();
      } else if (SawDefaultArgument)
        MissingDefaultArg = true;
    } 
    // Merge default arguments for non-type template parameters
    else if (NonTypeTemplateParmDecl *NewNonTypeParm
               = dyn_cast<NonTypeTemplateParmDecl>(*NewParam)) {
      NonTypeTemplateParmDecl *OldNonTypeParm
        = OldParams? cast<NonTypeTemplateParmDecl>(*OldParam) : 0;
      if (OldNonTypeParm && OldNonTypeParm->hasDefaultArgument() && 
          NewNonTypeParm->hasDefaultArgument()) {
        OldDefaultLoc = OldNonTypeParm->getDefaultArgumentLoc();
        NewDefaultLoc = NewNonTypeParm->getDefaultArgumentLoc();
        SawDefaultArgument = true;
        RedundantDefaultArg = true;
        PreviousDefaultArgLoc = NewDefaultLoc;
      } else if (OldNonTypeParm && OldNonTypeParm->hasDefaultArgument()) {
        // Merge the default argument from the old declaration to the
        // new declaration.
        SawDefaultArgument = true;
        // FIXME: We need to create a new kind of "default argument"
        // expression that points to a previous template template
        // parameter.
        NewNonTypeParm->setDefaultArgument(
                                        OldNonTypeParm->getDefaultArgument());
        PreviousDefaultArgLoc = OldNonTypeParm->getDefaultArgumentLoc();
      } else if (NewNonTypeParm->hasDefaultArgument()) {
        SawDefaultArgument = true;
        PreviousDefaultArgLoc = NewNonTypeParm->getDefaultArgumentLoc();
      } else if (SawDefaultArgument)
        MissingDefaultArg = true;      
    }
    // Merge default arguments for template template parameters
    else {
      TemplateTemplateParmDecl *NewTemplateParm
        = cast<TemplateTemplateParmDecl>(*NewParam);
      TemplateTemplateParmDecl *OldTemplateParm
        = OldParams? cast<TemplateTemplateParmDecl>(*OldParam) : 0;
      if (OldTemplateParm && OldTemplateParm->hasDefaultArgument() && 
          NewTemplateParm->hasDefaultArgument()) {
        OldDefaultLoc = OldTemplateParm->getDefaultArgumentLoc();
        NewDefaultLoc = NewTemplateParm->getDefaultArgumentLoc();
        SawDefaultArgument = true;
        RedundantDefaultArg = true;
        PreviousDefaultArgLoc = NewDefaultLoc;
      } else if (OldTemplateParm && OldTemplateParm->hasDefaultArgument()) {
        // Merge the default argument from the old declaration to the
        // new declaration.
        SawDefaultArgument = true;
        // FIXME: We need to create a new kind of "default argument"
        // expression that points to a previous template template
        // parameter.
        NewTemplateParm->setDefaultArgument(
                                        OldTemplateParm->getDefaultArgument());
        PreviousDefaultArgLoc = OldTemplateParm->getDefaultArgumentLoc();
      } else if (NewTemplateParm->hasDefaultArgument()) {
        SawDefaultArgument = true;
        PreviousDefaultArgLoc = NewTemplateParm->getDefaultArgumentLoc();
      } else if (SawDefaultArgument)
        MissingDefaultArg = true;      
    }

    if (RedundantDefaultArg) {
      // C++ [temp.param]p12:
      //   A template-parameter shall not be given default arguments
      //   by two different declarations in the same scope.
      Diag(NewDefaultLoc, diag::err_template_param_default_arg_redefinition);
      Diag(OldDefaultLoc, diag::note_template_param_prev_default_arg);
      Invalid = true;
    } else if (MissingDefaultArg) {
      // C++ [temp.param]p11:
      //   If a template-parameter has a default template-argument,
      //   all subsequent template-parameters shall have a default
      //   template-argument supplied.
      Diag((*NewParam)->getLocation(), 
           diag::err_template_param_default_arg_missing);
      Diag(PreviousDefaultArgLoc, diag::note_template_param_prev_default_arg);
      Invalid = true;
    }

    // If we have an old template parameter list that we're merging
    // in, move on to the next parameter.
    if (OldParams)
      ++OldParam;
  }

  return Invalid;
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
  ClassTemplateDecl *ClassTemplate = cast<ClassTemplateDecl>(Template);

  // Check that the template argument list is well-formed for this
  // template.
  llvm::SmallVector<TemplateArgument, 16> ConvertedTemplateArgs;
  if (CheckTemplateArgumentList(Template, TemplateLoc, LAngleLoc, 
                                TemplateArgs, TemplateArgLocs, RAngleLoc,
                                ConvertedTemplateArgs))
    return 0;

  assert((ConvertedTemplateArgs.size() == 
            Template->getTemplateParameters()->size()) &&
         "Converted template argument list is too short!");

  // Find the class template specialization declaration that
  // corresponds to these arguments.
  llvm::FoldingSetNodeID ID;
  ClassTemplateSpecializationDecl::Profile(ID, &ConvertedTemplateArgs[0],
                                           ConvertedTemplateArgs.size());
  void *InsertPos = 0;
  ClassTemplateSpecializationDecl *Decl
    = ClassTemplate->getSpecializations().FindNodeOrInsertPos(ID, InsertPos);
  if (!Decl) {
    // This is the first time we have referenced this class template
    // specialization. Create an appropriate declaration node and add
    // it to the list of specializations. This is the canonical
    // declaration of the class template.
    Decl = ClassTemplateSpecializationDecl::Create(Context, 
                                           ClassTemplate->getDeclContext(),
                                                   TemplateLoc,
                                                   ClassTemplate,
                                                   &ConvertedTemplateArgs[0],
                                           ConvertedTemplateArgs.size());
    ClassTemplate->getSpecializations().InsertNode(Decl, InsertPos);
  }

  // Build the fully-sugared type for this class template
  // specialization, which refers back to the class template
  // specialization we created or found.
  QualType Result
    = Context.getClassTemplateSpecializationType(Template, 
                                                 TemplateArgs.size(),
                    reinterpret_cast<uintptr_t *>(TemplateArgs.getArgs()), 
                                                 TemplateArgs.getArgIsType(),
                                               Context.getTypeDeclType(Decl));
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
                                     SourceLocation RAngleLoc,
                          llvm::SmallVectorImpl<TemplateArgument> &Converted) {
  TemplateParameterList *Params = Template->getTemplateParameters();
  unsigned NumParams = Params->size();
  unsigned NumArgs = Args.size();
  bool Invalid = false;

  if (NumArgs > NumParams ||
      NumArgs < Params->getMinRequiredArguments()) {
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
    Diag(Template->getLocation(), diag::note_template_decl_here)
      << Params->getSourceRange();
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
      // Retrieve the default template argument from the template
      // parameter.
      if (TemplateTypeParmDecl *TTP = dyn_cast<TemplateTypeParmDecl>(*Param)) {
        if (!TTP->hasDefaultArgument())
          break;

        ArgType = TTP->getDefaultArgument();
        ArgLoc = TTP->getDefaultArgumentLoc();
      } else if (NonTypeTemplateParmDecl *NTTP 
                   = dyn_cast<NonTypeTemplateParmDecl>(*Param)) {
        if (!NTTP->hasDefaultArgument())
          break;

        ArgExpr = NTTP->getDefaultArgument();
        ArgLoc = NTTP->getDefaultArgumentLoc();
      } else {
        TemplateTemplateParmDecl *TempParm 
          = cast<TemplateTemplateParmDecl>(*Param);      

        if (!TempParm->hasDefaultArgument())
          break;

        ArgExpr = TempParm->getDefaultArgument();
        ArgLoc = TempParm->getDefaultArgumentLoc();
      }
    } else {
      // Retrieve the template argument produced by the user.
      ArgLoc = TemplateArgLocs[ArgIdx];

      if (Args.getArgIsType()[ArgIdx])
        ArgType = QualType::getFromOpaquePtr(Args.getArgs()[ArgIdx]);
      else
        ArgExpr = reinterpret_cast<Expr *>(Args.getArgs()[ArgIdx]);
    }


    if (TemplateTypeParmDecl *TTP = dyn_cast<TemplateTypeParmDecl>(*Param)) {
      // Check template type parameters.
      if (!ArgType.isNull()) {
        if (CheckTemplateArgument(TTP, ArgType, ArgLoc))
          Invalid = true;

        // Add the converted template type argument.
        Converted.push_back(
                      TemplateArgument(Context.getCanonicalType(ArgType)));
        continue;
      }

      // C++ [temp.arg.type]p1:
      //   A template-argument for a template-parameter which is a
      //   type shall be a type-id.

      // We have a template type parameter but the template argument
      // is an expression.
      Diag(ArgExpr->getSourceRange().getBegin(), 
           diag::err_template_arg_must_be_type);
      Diag((*Param)->getLocation(), diag::note_template_param_here);
      Invalid = true;
    } else if (NonTypeTemplateParmDecl *NTTP 
                 = dyn_cast<NonTypeTemplateParmDecl>(*Param)) {
      // Check non-type template parameters.
      if (ArgExpr) {
        if (CheckTemplateArgument(NTTP, ArgExpr, &Converted))
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
      Diag((*Param)->getLocation(), diag::note_template_param_here);
      Invalid = true;
    } else { 
      // Check template template parameters.
      TemplateTemplateParmDecl *TempParm 
        = cast<TemplateTemplateParmDecl>(*Param);
     
      if (ArgExpr && isa<DeclRefExpr>(ArgExpr) &&
          isa<TemplateDecl>(cast<DeclRefExpr>(ArgExpr)->getDecl())) {
        if (CheckTemplateArgument(TempParm, cast<DeclRefExpr>(ArgExpr)))
          Invalid = true;

        // Add the converted template argument.
        // FIXME: Need the "canonical" template declaration!
        Converted.push_back(
                   TemplateArgument(cast<DeclRefExpr>(ArgExpr)->getDecl()));
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

/// \brief Checks whether the given template argument is the address
/// of an object or function according to C++ [temp.arg.nontype]p1.
bool Sema::CheckTemplateArgumentAddressOfObjectOrFunction(Expr *Arg,
                                                          NamedDecl *&Entity) {
  bool Invalid = false;

  // See through any implicit casts we added to fix the type.
  if (ImplicitCastExpr *Cast = dyn_cast<ImplicitCastExpr>(Arg))
    Arg = Cast->getSubExpr();

  // C++ [temp.arg.nontype]p1:
  // 
  //   A template-argument for a non-type, non-template
  //   template-parameter shall be one of: [...]
  //
  //     -- the address of an object or function with external
  //        linkage, including function templates and function
  //        template-ids but excluding non-static class members,
  //        expressed as & id-expression where the & is optional if
  //        the name refers to a function or array, or if the
  //        corresponding template-parameter is a reference; or
  DeclRefExpr *DRE = 0;
  
  // Ignore (and complain about) any excess parentheses.
  while (ParenExpr *Parens = dyn_cast<ParenExpr>(Arg)) {
    if (!Invalid) {
      Diag(Arg->getSourceRange().getBegin(), 
           diag::err_template_arg_extra_parens)
        << Arg->getSourceRange();
      Invalid = true;
    }

    Arg = Parens->getSubExpr();
  }

  if (UnaryOperator *UnOp = dyn_cast<UnaryOperator>(Arg)) {
    if (UnOp->getOpcode() == UnaryOperator::AddrOf)
      DRE = dyn_cast<DeclRefExpr>(UnOp->getSubExpr());
  } else
    DRE = dyn_cast<DeclRefExpr>(Arg);

  if (!DRE || !isa<ValueDecl>(DRE->getDecl()))
    return Diag(Arg->getSourceRange().getBegin(), 
                diag::err_template_arg_not_object_or_func_form)
      << Arg->getSourceRange();

  // Cannot refer to non-static data members
  if (FieldDecl *Field = dyn_cast<FieldDecl>(DRE->getDecl()))
    return Diag(Arg->getSourceRange().getBegin(), diag::err_template_arg_field)
      << Field << Arg->getSourceRange();

  // Cannot refer to non-static member functions
  if (CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(DRE->getDecl()))
    if (!Method->isStatic())
      return Diag(Arg->getSourceRange().getBegin(), 
                  diag::err_template_arg_method)
        << Method << Arg->getSourceRange();
   
  // Functions must have external linkage.
  if (FunctionDecl *Func = dyn_cast<FunctionDecl>(DRE->getDecl())) {
    if (Func->getStorageClass() == FunctionDecl::Static) {
      Diag(Arg->getSourceRange().getBegin(), 
           diag::err_template_arg_function_not_extern)
        << Func << Arg->getSourceRange();
      Diag(Func->getLocation(), diag::note_template_arg_internal_object)
        << true;
      return true;
    }

    // Okay: we've named a function with external linkage.
    Entity = Func;
    return Invalid;
  }

  if (VarDecl *Var = dyn_cast<VarDecl>(DRE->getDecl())) {
    if (!Var->hasGlobalStorage()) {
      Diag(Arg->getSourceRange().getBegin(), 
           diag::err_template_arg_object_not_extern)
        << Var << Arg->getSourceRange();
      Diag(Var->getLocation(), diag::note_template_arg_internal_object)
        << true;
      return true;
    }

    // Okay: we've named an object with external linkage
    Entity = Var;
    return Invalid;
  }
  
  // We found something else, but we don't know specifically what it is.
  Diag(Arg->getSourceRange().getBegin(), 
       diag::err_template_arg_not_object_or_func)
      << Arg->getSourceRange();
  Diag(DRE->getDecl()->getLocation(), 
       diag::note_template_arg_refers_here);
  return true;
}

/// \brief Checks whether the given template argument is a pointer to
/// member constant according to C++ [temp.arg.nontype]p1.
bool 
Sema::CheckTemplateArgumentPointerToMember(Expr *Arg, NamedDecl *&Member) {
  bool Invalid = false;

  // See through any implicit casts we added to fix the type.
  if (ImplicitCastExpr *Cast = dyn_cast<ImplicitCastExpr>(Arg))
    Arg = Cast->getSubExpr();

  // C++ [temp.arg.nontype]p1:
  // 
  //   A template-argument for a non-type, non-template
  //   template-parameter shall be one of: [...]
  //
  //     -- a pointer to member expressed as described in 5.3.1.
  QualifiedDeclRefExpr *DRE = 0;

  // Ignore (and complain about) any excess parentheses.
  while (ParenExpr *Parens = dyn_cast<ParenExpr>(Arg)) {
    if (!Invalid) {
      Diag(Arg->getSourceRange().getBegin(), 
           diag::err_template_arg_extra_parens)
        << Arg->getSourceRange();
      Invalid = true;
    }

    Arg = Parens->getSubExpr();
  }

  if (UnaryOperator *UnOp = dyn_cast<UnaryOperator>(Arg))
    if (UnOp->getOpcode() == UnaryOperator::AddrOf)
      DRE = dyn_cast<QualifiedDeclRefExpr>(UnOp->getSubExpr());

  if (!DRE)
    return Diag(Arg->getSourceRange().getBegin(),
                diag::err_template_arg_not_pointer_to_member_form)
      << Arg->getSourceRange();

  if (isa<FieldDecl>(DRE->getDecl()) || isa<CXXMethodDecl>(DRE->getDecl())) {
    assert((isa<FieldDecl>(DRE->getDecl()) ||
            !cast<CXXMethodDecl>(DRE->getDecl())->isStatic()) &&
           "Only non-static member pointers can make it here");

    // Okay: this is the address of a non-static member, and therefore
    // a member pointer constant.
    Member = DRE->getDecl();
    return Invalid;
  }

  // We found something else, but we don't know specifically what it is.
  Diag(Arg->getSourceRange().getBegin(), 
       diag::err_template_arg_not_pointer_to_member_form)
      << Arg->getSourceRange();
  Diag(DRE->getDecl()->getLocation(), 
       diag::note_template_arg_refers_here);
  return true;
}

/// \brief Check a template argument against its corresponding
/// non-type template parameter.
///
/// This routine implements the semantics of C++ [temp.arg.nontype]. 
/// It returns true if an error occurred, and false otherwise.
///
/// If Converted is non-NULL and no errors occur, the value
/// of this argument will be added to the end of the Converted vector.
bool Sema::CheckTemplateArgument(NonTypeTemplateParmDecl *Param,
                                 Expr *&Arg, 
                         llvm::SmallVectorImpl<TemplateArgument> *Converted) {
  // If either the parameter has a dependent type or the argument is
  // type-dependent, there's nothing we can check now.
  // FIXME: Add template argument to Converted!
  if (Param->getType()->isDependentType() || Arg->isTypeDependent())
    return false;

  // C++ [temp.arg.nontype]p5:
  //   The following conversions are performed on each expression used
  //   as a non-type template-argument. If a non-type
  //   template-argument cannot be converted to the type of the
  //   corresponding template-parameter then the program is
  //   ill-formed.
  //
  //     -- for a non-type template-parameter of integral or
  //        enumeration type, integral promotions (4.5) and integral
  //        conversions (4.7) are applied.
  QualType ParamType = Param->getType();
  QualType ArgType = Arg->getType();
  if (ParamType->isIntegralType() || ParamType->isEnumeralType()) {
    // C++ [temp.arg.nontype]p1:
    //   A template-argument for a non-type, non-template
    //   template-parameter shall be one of:
    //
    //     -- an integral constant-expression of integral or enumeration
    //        type; or
    //     -- the name of a non-type template-parameter; or
    SourceLocation NonConstantLoc;
    llvm::APSInt Value;
    if (!ArgType->isIntegralType() && !ArgType->isEnumeralType()) {
      Diag(Arg->getSourceRange().getBegin(), 
           diag::err_template_arg_not_integral_or_enumeral)
        << ArgType << Arg->getSourceRange();
      Diag(Param->getLocation(), diag::note_template_param_here);
      return true;
    } else if (!Arg->isValueDependent() &&
               !Arg->isIntegerConstantExpr(Value, Context, &NonConstantLoc)) {
      Diag(NonConstantLoc, diag::err_template_arg_not_ice)
        << ArgType << Arg->getSourceRange();
      return true;
    }

    // FIXME: We need some way to more easily get the unqualified form
    // of the types without going all the way to the
    // canonical type.
    if (Context.getCanonicalType(ParamType).getCVRQualifiers())
      ParamType = Context.getCanonicalType(ParamType).getUnqualifiedType();
    if (Context.getCanonicalType(ArgType).getCVRQualifiers())
      ArgType = Context.getCanonicalType(ArgType).getUnqualifiedType();

    // Try to convert the argument to the parameter's type.
    if (ParamType == ArgType) {
      // Okay: no conversion necessary
    } else if (IsIntegralPromotion(Arg, ArgType, ParamType) ||
               !ParamType->isEnumeralType()) {
      // This is an integral promotion or conversion.
      ImpCastExprToType(Arg, ParamType);
    } else {
      // We can't perform this conversion.
      Diag(Arg->getSourceRange().getBegin(), 
           diag::err_template_arg_not_convertible)
        << Arg->getType() << Param->getType() << Arg->getSourceRange();
      Diag(Param->getLocation(), diag::note_template_param_here);
      return true;
    }

    // FIXME: Check overflow of template arguments?

    if (Converted) {
      // Add the value of this argument to the list of converted
      // arguments. We use the bitwidth and signedness of the template
      // parameter.
      QualType IntegerType = Context.getCanonicalType(ParamType);
      if (const EnumType *Enum = IntegerType->getAsEnumType())
        IntegerType = Enum->getDecl()->getIntegerType();
      
      llvm::APInt CanonicalArg(Context.getTypeSize(IntegerType), 0, 
                               IntegerType->isSignedIntegerType());
      CanonicalArg = Value;

      Converted->push_back(TemplateArgument(CanonicalArg));
    }

    return false;
  }

  // Handle pointer-to-function, reference-to-function, and
  // pointer-to-member-function all in (roughly) the same way.
  if (// -- For a non-type template-parameter of type pointer to
      //    function, only the function-to-pointer conversion (4.3) is
      //    applied. If the template-argument represents a set of
      //    overloaded functions (or a pointer to such), the matching
      //    function is selected from the set (13.4).
      (ParamType->isPointerType() &&
       ParamType->getAsPointerType()->getPointeeType()->isFunctionType()) ||
      // -- For a non-type template-parameter of type reference to
      //    function, no conversions apply. If the template-argument
      //    represents a set of overloaded functions, the matching
      //    function is selected from the set (13.4).
      (ParamType->isReferenceType() &&
       ParamType->getAsReferenceType()->getPointeeType()->isFunctionType()) ||
      // -- For a non-type template-parameter of type pointer to
      //    member function, no conversions apply. If the
      //    template-argument represents a set of overloaded member
      //    functions, the matching member function is selected from
      //    the set (13.4).
      (ParamType->isMemberPointerType() &&
       ParamType->getAsMemberPointerType()->getPointeeType()
         ->isFunctionType())) {
    if (Context.hasSameUnqualifiedType(ArgType, 
                                       ParamType.getNonReferenceType())) {
      // We don't have to do anything: the types already match.
    } else if (ArgType->isFunctionType() && ParamType->isPointerType()) {
      ArgType = Context.getPointerType(ArgType);
      ImpCastExprToType(Arg, ArgType);
    } else if (FunctionDecl *Fn 
                 = ResolveAddressOfOverloadedFunction(Arg, ParamType, true)) {
      FixOverloadedFunctionReference(Arg, Fn);
      ArgType = Arg->getType();
      if (ArgType->isFunctionType() && ParamType->isPointerType()) {
        ArgType = Context.getPointerType(Arg->getType());
        ImpCastExprToType(Arg, ArgType);
      }
    }

    if (!Context.hasSameUnqualifiedType(ArgType, 
                                        ParamType.getNonReferenceType())) {
      // We can't perform this conversion.
      Diag(Arg->getSourceRange().getBegin(), 
           diag::err_template_arg_not_convertible)
        << Arg->getType() << Param->getType() << Arg->getSourceRange();
      Diag(Param->getLocation(), diag::note_template_param_here);
      return true;
    }
    
    if (ParamType->isMemberPointerType()) {
      NamedDecl *Member = 0;
      if (CheckTemplateArgumentPointerToMember(Arg, Member))
        return true;

      if (Converted)
        Converted->push_back(TemplateArgument(Member));

      return false;
    }
    
    NamedDecl *Entity = 0;
    if (CheckTemplateArgumentAddressOfObjectOrFunction(Arg, Entity))
      return true;

    if (Converted)
      Converted->push_back(TemplateArgument(Entity));
    return false;
  }

  if (const PointerType *ParamPtrType = ParamType->getAsPointerType()) {
    //   -- for a non-type template-parameter of type pointer to
    //      object, qualification conversions (4.4) and the
    //      array-to-pointer conversion (4.2) are applied.
    assert(ParamPtrType->getPointeeType()->isObjectType() &&
           "Only object pointers allowed here");

    if (ArgType->isArrayType()) {
      ArgType = Context.getArrayDecayedType(ArgType);
      ImpCastExprToType(Arg, ArgType);
    }
    
    if (IsQualificationConversion(ArgType, ParamType)) {
      ArgType = ParamType;
      ImpCastExprToType(Arg, ParamType);
    }
    
    if (!Context.hasSameUnqualifiedType(ArgType, ParamType)) {
      // We can't perform this conversion.
      Diag(Arg->getSourceRange().getBegin(), 
           diag::err_template_arg_not_convertible)
        << Arg->getType() << Param->getType() << Arg->getSourceRange();
      Diag(Param->getLocation(), diag::note_template_param_here);
      return true;
    }
    
    NamedDecl *Entity = 0;
    if (CheckTemplateArgumentAddressOfObjectOrFunction(Arg, Entity))
      return true;

    if (Converted)
      Converted->push_back(TemplateArgument(Entity));

    return false;
  }
    
  if (const ReferenceType *ParamRefType = ParamType->getAsReferenceType()) {
    //   -- For a non-type template-parameter of type reference to
    //      object, no conversions apply. The type referred to by the
    //      reference may be more cv-qualified than the (otherwise
    //      identical) type of the template-argument. The
    //      template-parameter is bound directly to the
    //      template-argument, which must be an lvalue.
    assert(ParamRefType->getPointeeType()->isObjectType() &&
           "Only object references allowed here");

    if (!Context.hasSameUnqualifiedType(ParamRefType->getPointeeType(), ArgType)) {
      Diag(Arg->getSourceRange().getBegin(), 
           diag::err_template_arg_no_ref_bind)
        << Param->getType() << Arg->getType()
        << Arg->getSourceRange();
      Diag(Param->getLocation(), diag::note_template_param_here);
      return true;
    }

    unsigned ParamQuals 
      = Context.getCanonicalType(ParamType).getCVRQualifiers();
    unsigned ArgQuals = Context.getCanonicalType(ArgType).getCVRQualifiers();
    
    if ((ParamQuals | ArgQuals) != ParamQuals) {
      Diag(Arg->getSourceRange().getBegin(),
           diag::err_template_arg_ref_bind_ignores_quals)
        << Param->getType() << Arg->getType()
        << Arg->getSourceRange();
      Diag(Param->getLocation(), diag::note_template_param_here);
      return true;
    }
    
    NamedDecl *Entity = 0;
    if (CheckTemplateArgumentAddressOfObjectOrFunction(Arg, Entity))
      return true;

    if (Converted)
      Converted->push_back(TemplateArgument(Entity));

    return false;
  }

  //     -- For a non-type template-parameter of type pointer to data
  //        member, qualification conversions (4.4) are applied.
  assert(ParamType->isMemberPointerType() && "Only pointers to members remain");

  if (Context.hasSameUnqualifiedType(ParamType, ArgType)) {
    // Types match exactly: nothing more to do here.
  } else if (IsQualificationConversion(ArgType, ParamType)) {
    ImpCastExprToType(Arg, ParamType);
  } else {
    // We can't perform this conversion.
    Diag(Arg->getSourceRange().getBegin(), 
         diag::err_template_arg_not_convertible)
      << Arg->getType() << Param->getType() << Arg->getSourceRange();
    Diag(Param->getLocation(), diag::note_template_param_here);
    return true;    
  }

  NamedDecl *Member = 0;
  if (CheckTemplateArgumentPointerToMember(Arg, Member))
    return true;
  
  if (Converted)
    Converted->push_back(TemplateArgument(Member));
  
  return false;
}

/// \brief Check a template argument against its corresponding
/// template template parameter.
///
/// This routine implements the semantics of C++ [temp.arg.template].
/// It returns true if an error occurred, and false otherwise.
bool Sema::CheckTemplateArgument(TemplateTemplateParmDecl *Param,
                                 DeclRefExpr *Arg) {
  assert(isa<TemplateDecl>(Arg->getDecl()) && "Only template decls allowed");
  TemplateDecl *Template = cast<TemplateDecl>(Arg->getDecl());

  // C++ [temp.arg.template]p1:
  //   A template-argument for a template template-parameter shall be
  //   the name of a class template, expressed as id-expression. Only
  //   primary class templates are considered when matching the
  //   template template argument with the corresponding parameter;
  //   partial specializations are not considered even if their
  //   parameter lists match that of the template template parameter.
  if (!isa<ClassTemplateDecl>(Template)) {
    assert(isa<FunctionTemplateDecl>(Template) && 
           "Only function templates are possible here");
    Diag(Arg->getSourceRange().getBegin(), 
         diag::note_template_arg_refers_here_func)
      << Template;
  }

  return !TemplateParameterListsAreEqual(Template->getTemplateParameters(),
                                         Param->getTemplateParameters(),
                                         true, true,
                                         Arg->getSourceRange().getBegin());
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
/// \param IsTemplateTemplateParm  If true, this routine is being
/// called to compare the template parameter lists of a template
/// template parameter.
///
/// \param TemplateArgLoc If this source location is valid, then we
/// are actually checking the template parameter list of a template
/// argument (New) against the template parameter list of its
/// corresponding template template parameter (Old). We produce
/// slightly different diagnostics in this scenario.
///
/// \returns True if the template parameter lists are equal, false
/// otherwise.
bool 
Sema::TemplateParameterListsAreEqual(TemplateParameterList *New,
                                     TemplateParameterList *Old,
                                     bool Complain,
                                     bool IsTemplateTemplateParm,
                                     SourceLocation TemplateArgLoc) {
  if (Old->size() != New->size()) {
    if (Complain) {
      unsigned NextDiag = diag::err_template_param_list_different_arity;
      if (TemplateArgLoc.isValid()) {
        Diag(TemplateArgLoc, diag::err_template_arg_template_params_mismatch);
        NextDiag = diag::note_template_param_list_different_arity;
      } 
      Diag(New->getTemplateLoc(), NextDiag)
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
      unsigned NextDiag = diag::err_template_param_different_kind;
      if (TemplateArgLoc.isValid()) {
        Diag(TemplateArgLoc, diag::err_template_arg_template_params_mismatch);
        NextDiag = diag::note_template_param_different_kind;
      }
      Diag((*NewParm)->getLocation(), NextDiag)
        << IsTemplateTemplateParm;
      Diag((*OldParm)->getLocation(), diag::note_template_prev_declaration)
        << IsTemplateTemplateParm;
      return false;
    }

    if (isa<TemplateTypeParmDecl>(*OldParm)) {
      // Okay; all template type parameters are equivalent (since we
      // know we're at the same index).
#if 0
      // FIXME: Enable this code in debug mode *after* we properly go
      // through and "instantiate" the template parameter lists of
      // template template parameters. It's only after this
      // instantiation that (1) any dependent types within the
      // template parameter list of the template template parameter
      // can be checked, and (2) the template type parameter depths
      // will match up.
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
          unsigned NextDiag = diag::err_template_nontype_parm_different_type;
          if (TemplateArgLoc.isValid()) {
            Diag(TemplateArgLoc, 
                 diag::err_template_arg_template_params_mismatch);
            NextDiag = diag::note_template_nontype_parm_different_type;
          }
          Diag(NewNTTP->getLocation(), NextDiag)
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
                                          /*IsTemplateTemplateParm=*/true,
                                          TemplateArgLoc))
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
