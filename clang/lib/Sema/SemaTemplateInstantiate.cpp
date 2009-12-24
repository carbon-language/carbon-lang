//===------- SemaTemplateInstantiate.cpp - C++ Template Instantiation ------===/
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//===----------------------------------------------------------------------===/
//
//  This file implements C++ template instantiation.
//
//===----------------------------------------------------------------------===/

#include "Sema.h"
#include "TreeTransform.h"
#include "Lookup.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/Parse/DeclSpec.h"
#include "clang/Basic/LangOptions.h"

using namespace clang;

//===----------------------------------------------------------------------===/
// Template Instantiation Support
//===----------------------------------------------------------------------===/

/// \brief Retrieve the template argument list(s) that should be used to
/// instantiate the definition of the given declaration.
///
/// \param D the declaration for which we are computing template instantiation
/// arguments.
///
/// \param Innermost if non-NULL, the innermost template argument list.
MultiLevelTemplateArgumentList
Sema::getTemplateInstantiationArgs(NamedDecl *D, 
                                   const TemplateArgumentList *Innermost) {
  // Accumulate the set of template argument lists in this structure.
  MultiLevelTemplateArgumentList Result;

  if (Innermost)
    Result.addOuterTemplateArguments(Innermost);
  
  DeclContext *Ctx = dyn_cast<DeclContext>(D);
  if (!Ctx)
    Ctx = D->getDeclContext();

  while (!Ctx->isFileContext()) {
    // Add template arguments from a class template instantiation.
    if (ClassTemplateSpecializationDecl *Spec
          = dyn_cast<ClassTemplateSpecializationDecl>(Ctx)) {
      // We're done when we hit an explicit specialization.
      if (Spec->getSpecializationKind() == TSK_ExplicitSpecialization)
        break;

      Result.addOuterTemplateArguments(&Spec->getTemplateInstantiationArgs());
      
      // If this class template specialization was instantiated from a 
      // specialized member that is a class template, we're done.
      assert(Spec->getSpecializedTemplate() && "No class template?");
      if (Spec->getSpecializedTemplate()->isMemberSpecialization())
        break;
    }
    // Add template arguments from a function template specialization.
    else if (FunctionDecl *Function = dyn_cast<FunctionDecl>(Ctx)) {
      if (Function->getTemplateSpecializationKind() 
            == TSK_ExplicitSpecialization)
        break;
          
      if (const TemplateArgumentList *TemplateArgs
            = Function->getTemplateSpecializationArgs()) {
        // Add the template arguments for this specialization.
        Result.addOuterTemplateArguments(TemplateArgs);

        // If this function was instantiated from a specialized member that is
        // a function template, we're done.
        assert(Function->getPrimaryTemplate() && "No function template?");
        if (Function->getPrimaryTemplate()->isMemberSpecialization())
          break;
      }
      
      // If this is a friend declaration and it declares an entity at
      // namespace scope, take arguments from its lexical parent
      // instead of its semantic parent.
      if (Function->getFriendObjectKind() &&
          Function->getDeclContext()->isFileContext()) {
        Ctx = Function->getLexicalDeclContext();
        continue;
      }
    }

    Ctx = Ctx->getParent();
  }

  return Result;
}

bool Sema::ActiveTemplateInstantiation::isInstantiationRecord() const {
  switch (Kind) {
  case TemplateInstantiation:
  case DefaultTemplateArgumentInstantiation:
  case DefaultFunctionArgumentInstantiation:
    return true;
      
  case ExplicitTemplateArgumentSubstitution:
  case DeducedTemplateArgumentSubstitution:
  case PriorTemplateArgumentSubstitution:
  case DefaultTemplateArgumentChecking:
    return false;
  }
  
  return true;
}

Sema::InstantiatingTemplate::
InstantiatingTemplate(Sema &SemaRef, SourceLocation PointOfInstantiation,
                      Decl *Entity,
                      SourceRange InstantiationRange)
  :  SemaRef(SemaRef) {

  Invalid = CheckInstantiationDepth(PointOfInstantiation,
                                    InstantiationRange);
  if (!Invalid) {
    ActiveTemplateInstantiation Inst;
    Inst.Kind = ActiveTemplateInstantiation::TemplateInstantiation;
    Inst.PointOfInstantiation = PointOfInstantiation;
    Inst.Entity = reinterpret_cast<uintptr_t>(Entity);
    Inst.TemplateArgs = 0;
    Inst.NumTemplateArgs = 0;
    Inst.InstantiationRange = InstantiationRange;
    SemaRef.ActiveTemplateInstantiations.push_back(Inst);
  }
}

Sema::InstantiatingTemplate::InstantiatingTemplate(Sema &SemaRef,
                                         SourceLocation PointOfInstantiation,
                                         TemplateDecl *Template,
                                         const TemplateArgument *TemplateArgs,
                                         unsigned NumTemplateArgs,
                                         SourceRange InstantiationRange)
  : SemaRef(SemaRef) {

  Invalid = CheckInstantiationDepth(PointOfInstantiation,
                                    InstantiationRange);
  if (!Invalid) {
    ActiveTemplateInstantiation Inst;
    Inst.Kind
      = ActiveTemplateInstantiation::DefaultTemplateArgumentInstantiation;
    Inst.PointOfInstantiation = PointOfInstantiation;
    Inst.Entity = reinterpret_cast<uintptr_t>(Template);
    Inst.TemplateArgs = TemplateArgs;
    Inst.NumTemplateArgs = NumTemplateArgs;
    Inst.InstantiationRange = InstantiationRange;
    SemaRef.ActiveTemplateInstantiations.push_back(Inst);
  }
}

Sema::InstantiatingTemplate::InstantiatingTemplate(Sema &SemaRef,
                                         SourceLocation PointOfInstantiation,
                                      FunctionTemplateDecl *FunctionTemplate,
                                        const TemplateArgument *TemplateArgs,
                                                   unsigned NumTemplateArgs,
                         ActiveTemplateInstantiation::InstantiationKind Kind,
                                              SourceRange InstantiationRange)
: SemaRef(SemaRef) {

  Invalid = CheckInstantiationDepth(PointOfInstantiation,
                                    InstantiationRange);
  if (!Invalid) {
    ActiveTemplateInstantiation Inst;
    Inst.Kind = Kind;
    Inst.PointOfInstantiation = PointOfInstantiation;
    Inst.Entity = reinterpret_cast<uintptr_t>(FunctionTemplate);
    Inst.TemplateArgs = TemplateArgs;
    Inst.NumTemplateArgs = NumTemplateArgs;
    Inst.InstantiationRange = InstantiationRange;
    SemaRef.ActiveTemplateInstantiations.push_back(Inst);
    
    if (!Inst.isInstantiationRecord())
      ++SemaRef.NonInstantiationEntries;
  }
}

Sema::InstantiatingTemplate::InstantiatingTemplate(Sema &SemaRef,
                                         SourceLocation PointOfInstantiation,
                          ClassTemplatePartialSpecializationDecl *PartialSpec,
                                         const TemplateArgument *TemplateArgs,
                                         unsigned NumTemplateArgs,
                                         SourceRange InstantiationRange)
  : SemaRef(SemaRef) {

  Invalid = false;
    
  ActiveTemplateInstantiation Inst;
  Inst.Kind = ActiveTemplateInstantiation::DeducedTemplateArgumentSubstitution;
  Inst.PointOfInstantiation = PointOfInstantiation;
  Inst.Entity = reinterpret_cast<uintptr_t>(PartialSpec);
  Inst.TemplateArgs = TemplateArgs;
  Inst.NumTemplateArgs = NumTemplateArgs;
  Inst.InstantiationRange = InstantiationRange;
  SemaRef.ActiveTemplateInstantiations.push_back(Inst);
      
  assert(!Inst.isInstantiationRecord());
  ++SemaRef.NonInstantiationEntries;
}

Sema::InstantiatingTemplate::InstantiatingTemplate(Sema &SemaRef,
                                          SourceLocation PointOfInstantiation,
                                          ParmVarDecl *Param,
                                          const TemplateArgument *TemplateArgs,
                                          unsigned NumTemplateArgs,
                                          SourceRange InstantiationRange)
  : SemaRef(SemaRef) {

  Invalid = CheckInstantiationDepth(PointOfInstantiation, InstantiationRange);

  if (!Invalid) {
    ActiveTemplateInstantiation Inst;
    Inst.Kind
      = ActiveTemplateInstantiation::DefaultFunctionArgumentInstantiation;
    Inst.PointOfInstantiation = PointOfInstantiation;
    Inst.Entity = reinterpret_cast<uintptr_t>(Param);
    Inst.TemplateArgs = TemplateArgs;
    Inst.NumTemplateArgs = NumTemplateArgs;
    Inst.InstantiationRange = InstantiationRange;
    SemaRef.ActiveTemplateInstantiations.push_back(Inst);
  }
}

Sema::InstantiatingTemplate::
InstantiatingTemplate(Sema &SemaRef, SourceLocation PointOfInstantiation,
                      TemplateDecl *Template,
                      NonTypeTemplateParmDecl *Param,
                      const TemplateArgument *TemplateArgs,
                      unsigned NumTemplateArgs,
                      SourceRange InstantiationRange) : SemaRef(SemaRef) {
  Invalid = false;
  
  ActiveTemplateInstantiation Inst;
  Inst.Kind = ActiveTemplateInstantiation::PriorTemplateArgumentSubstitution;
  Inst.PointOfInstantiation = PointOfInstantiation;
  Inst.Template = Template;
  Inst.Entity = reinterpret_cast<uintptr_t>(Param);
  Inst.TemplateArgs = TemplateArgs;
  Inst.NumTemplateArgs = NumTemplateArgs;
  Inst.InstantiationRange = InstantiationRange;
  SemaRef.ActiveTemplateInstantiations.push_back(Inst);
  
  assert(!Inst.isInstantiationRecord());
  ++SemaRef.NonInstantiationEntries;
}

Sema::InstantiatingTemplate::
InstantiatingTemplate(Sema &SemaRef, SourceLocation PointOfInstantiation,
                      TemplateDecl *Template,
                      TemplateTemplateParmDecl *Param,
                      const TemplateArgument *TemplateArgs,
                      unsigned NumTemplateArgs,
                      SourceRange InstantiationRange) : SemaRef(SemaRef) {
  Invalid = false;
  ActiveTemplateInstantiation Inst;
  Inst.Kind = ActiveTemplateInstantiation::PriorTemplateArgumentSubstitution;
  Inst.PointOfInstantiation = PointOfInstantiation;
  Inst.Template = Template;
  Inst.Entity = reinterpret_cast<uintptr_t>(Param);
  Inst.TemplateArgs = TemplateArgs;
  Inst.NumTemplateArgs = NumTemplateArgs;
  Inst.InstantiationRange = InstantiationRange;
  SemaRef.ActiveTemplateInstantiations.push_back(Inst);
  
  assert(!Inst.isInstantiationRecord());
  ++SemaRef.NonInstantiationEntries;
}

Sema::InstantiatingTemplate::
InstantiatingTemplate(Sema &SemaRef, SourceLocation PointOfInstantiation,
                      TemplateDecl *Template,
                      NamedDecl *Param,
                      const TemplateArgument *TemplateArgs,
                      unsigned NumTemplateArgs,
                      SourceRange InstantiationRange) : SemaRef(SemaRef) {
  Invalid = false;
  
  ActiveTemplateInstantiation Inst;
  Inst.Kind = ActiveTemplateInstantiation::DefaultTemplateArgumentChecking;
  Inst.PointOfInstantiation = PointOfInstantiation;
  Inst.Template = Template;
  Inst.Entity = reinterpret_cast<uintptr_t>(Param);
  Inst.TemplateArgs = TemplateArgs;
  Inst.NumTemplateArgs = NumTemplateArgs;
  Inst.InstantiationRange = InstantiationRange;
  SemaRef.ActiveTemplateInstantiations.push_back(Inst);
  
  assert(!Inst.isInstantiationRecord());
  ++SemaRef.NonInstantiationEntries;
}

void Sema::InstantiatingTemplate::Clear() {
  if (!Invalid) {
    if (!SemaRef.ActiveTemplateInstantiations.back().isInstantiationRecord()) {
      assert(SemaRef.NonInstantiationEntries > 0);
      --SemaRef.NonInstantiationEntries;
    }
    
    SemaRef.ActiveTemplateInstantiations.pop_back();
    Invalid = true;
  }
}

bool Sema::InstantiatingTemplate::CheckInstantiationDepth(
                                        SourceLocation PointOfInstantiation,
                                           SourceRange InstantiationRange) {
  assert(SemaRef.NonInstantiationEntries <=
                                   SemaRef.ActiveTemplateInstantiations.size());
  if ((SemaRef.ActiveTemplateInstantiations.size() - 
          SemaRef.NonInstantiationEntries)
        <= SemaRef.getLangOptions().InstantiationDepth)
    return false;

  SemaRef.Diag(PointOfInstantiation,
               diag::err_template_recursion_depth_exceeded)
    << SemaRef.getLangOptions().InstantiationDepth
    << InstantiationRange;
  SemaRef.Diag(PointOfInstantiation, diag::note_template_recursion_depth)
    << SemaRef.getLangOptions().InstantiationDepth;
  return true;
}

/// \brief Prints the current instantiation stack through a series of
/// notes.
void Sema::PrintInstantiationStack() {
  // FIXME: In all of these cases, we need to show the template arguments
  for (llvm::SmallVector<ActiveTemplateInstantiation, 16>::reverse_iterator
         Active = ActiveTemplateInstantiations.rbegin(),
         ActiveEnd = ActiveTemplateInstantiations.rend();
       Active != ActiveEnd;
       ++Active) {
    switch (Active->Kind) {
    case ActiveTemplateInstantiation::TemplateInstantiation: {
      Decl *D = reinterpret_cast<Decl *>(Active->Entity);
      if (CXXRecordDecl *Record = dyn_cast<CXXRecordDecl>(D)) {
        unsigned DiagID = diag::note_template_member_class_here;
        if (isa<ClassTemplateSpecializationDecl>(Record))
          DiagID = diag::note_template_class_instantiation_here;
        Diags.Report(FullSourceLoc(Active->PointOfInstantiation, SourceMgr),
                     DiagID)
          << Context.getTypeDeclType(Record)
          << Active->InstantiationRange;
      } else if (FunctionDecl *Function = dyn_cast<FunctionDecl>(D)) {
        unsigned DiagID;
        if (Function->getPrimaryTemplate())
          DiagID = diag::note_function_template_spec_here;
        else
          DiagID = diag::note_template_member_function_here;
        Diags.Report(FullSourceLoc(Active->PointOfInstantiation, SourceMgr),
                     DiagID)
          << Function
          << Active->InstantiationRange;
      } else {
        Diags.Report(FullSourceLoc(Active->PointOfInstantiation, SourceMgr),
                     diag::note_template_static_data_member_def_here)
          << cast<VarDecl>(D)
          << Active->InstantiationRange;
      }
      break;
    }

    case ActiveTemplateInstantiation::DefaultTemplateArgumentInstantiation: {
      TemplateDecl *Template = cast<TemplateDecl>((Decl *)Active->Entity);
      std::string TemplateArgsStr
        = TemplateSpecializationType::PrintTemplateArgumentList(
                                                         Active->TemplateArgs,
                                                      Active->NumTemplateArgs,
                                                      Context.PrintingPolicy);
      Diags.Report(FullSourceLoc(Active->PointOfInstantiation, SourceMgr),
                   diag::note_default_arg_instantiation_here)
        << (Template->getNameAsString() + TemplateArgsStr)
        << Active->InstantiationRange;
      break;
    }

    case ActiveTemplateInstantiation::ExplicitTemplateArgumentSubstitution: {
      FunctionTemplateDecl *FnTmpl
        = cast<FunctionTemplateDecl>((Decl *)Active->Entity);
      Diags.Report(FullSourceLoc(Active->PointOfInstantiation, SourceMgr),
                   diag::note_explicit_template_arg_substitution_here)
        << FnTmpl << Active->InstantiationRange;
      break;
    }

    case ActiveTemplateInstantiation::DeducedTemplateArgumentSubstitution:
      if (ClassTemplatePartialSpecializationDecl *PartialSpec
            = dyn_cast<ClassTemplatePartialSpecializationDecl>(
                                                    (Decl *)Active->Entity)) {
        Diags.Report(FullSourceLoc(Active->PointOfInstantiation, SourceMgr),
                     diag::note_partial_spec_deduct_instantiation_here)
          << Context.getTypeDeclType(PartialSpec)
          << Active->InstantiationRange;
      } else {
        FunctionTemplateDecl *FnTmpl
          = cast<FunctionTemplateDecl>((Decl *)Active->Entity);
        Diags.Report(FullSourceLoc(Active->PointOfInstantiation, SourceMgr),
                     diag::note_function_template_deduction_instantiation_here)
          << FnTmpl << Active->InstantiationRange;
      }
      break;

    case ActiveTemplateInstantiation::DefaultFunctionArgumentInstantiation: {
      ParmVarDecl *Param = cast<ParmVarDecl>((Decl *)Active->Entity);
      FunctionDecl *FD = cast<FunctionDecl>(Param->getDeclContext());

      std::string TemplateArgsStr
        = TemplateSpecializationType::PrintTemplateArgumentList(
                                                         Active->TemplateArgs,
                                                      Active->NumTemplateArgs,
                                                      Context.PrintingPolicy);
      Diags.Report(FullSourceLoc(Active->PointOfInstantiation, SourceMgr),
                   diag::note_default_function_arg_instantiation_here)
        << (FD->getNameAsString() + TemplateArgsStr)
        << Active->InstantiationRange;
      break;
    }

    case ActiveTemplateInstantiation::PriorTemplateArgumentSubstitution: {
      NamedDecl *Parm = cast<NamedDecl>((Decl *)Active->Entity);
      std::string Name;
      if (!Parm->getName().empty())
        Name = std::string(" '") + Parm->getName().str() + "'";
                                        
      Diags.Report(FullSourceLoc(Active->PointOfInstantiation, SourceMgr),
                   diag::note_prior_template_arg_substitution)
        << isa<TemplateTemplateParmDecl>(Parm)
        << Name
        << getTemplateArgumentBindingsText(
                                    Active->Template->getTemplateParameters(), 
                                           Active->TemplateArgs, 
                                           Active->NumTemplateArgs)
        << Active->InstantiationRange;
      break;
    }

    case ActiveTemplateInstantiation::DefaultTemplateArgumentChecking: {
      Diags.Report(FullSourceLoc(Active->PointOfInstantiation, SourceMgr),
                   diag::note_template_default_arg_checking)
        << getTemplateArgumentBindingsText(
                                     Active->Template->getTemplateParameters(), 
                                           Active->TemplateArgs, 
                                           Active->NumTemplateArgs)
        << Active->InstantiationRange;
      break;
    }
    }
  }
}

bool Sema::isSFINAEContext() const {
  using llvm::SmallVector;
  for (SmallVector<ActiveTemplateInstantiation, 16>::const_reverse_iterator
         Active = ActiveTemplateInstantiations.rbegin(),
         ActiveEnd = ActiveTemplateInstantiations.rend();
       Active != ActiveEnd;
       ++Active) 
  {
    switch(Active->Kind) {
    case ActiveTemplateInstantiation::TemplateInstantiation:
    case ActiveTemplateInstantiation::DefaultFunctionArgumentInstantiation:
      // This is a template instantiation, so there is no SFINAE.
      return false;

    case ActiveTemplateInstantiation::DefaultTemplateArgumentInstantiation:
    case ActiveTemplateInstantiation::PriorTemplateArgumentSubstitution:
    case ActiveTemplateInstantiation::DefaultTemplateArgumentChecking:
      // A default template argument instantiation and substitution into
      // template parameters with arguments for prior parameters may or may 
      // not be a SFINAE context; look further up the stack.
      break;

    case ActiveTemplateInstantiation::ExplicitTemplateArgumentSubstitution:
    case ActiveTemplateInstantiation::DeducedTemplateArgumentSubstitution:
      // We're either substitution explicitly-specified template arguments
      // or deduced template arguments, so SFINAE applies.
      return true;
    }
  }

  return false;
}

//===----------------------------------------------------------------------===/
// Template Instantiation for Types
//===----------------------------------------------------------------------===/
namespace {
  class TemplateInstantiator
    : public TreeTransform<TemplateInstantiator> {
    const MultiLevelTemplateArgumentList &TemplateArgs;
    SourceLocation Loc;
    DeclarationName Entity;

  public:
    typedef TreeTransform<TemplateInstantiator> inherited;

    TemplateInstantiator(Sema &SemaRef,
                         const MultiLevelTemplateArgumentList &TemplateArgs,
                         SourceLocation Loc,
                         DeclarationName Entity)
      : inherited(SemaRef), TemplateArgs(TemplateArgs), Loc(Loc),
        Entity(Entity) { }

    /// \brief Determine whether the given type \p T has already been
    /// transformed.
    ///
    /// For the purposes of template instantiation, a type has already been
    /// transformed if it is NULL or if it is not dependent.
    bool AlreadyTransformed(QualType T) {
      return T.isNull() || !T->isDependentType();
    }

    /// \brief Returns the location of the entity being instantiated, if known.
    SourceLocation getBaseLocation() { return Loc; }

    /// \brief Returns the name of the entity being instantiated, if any.
    DeclarationName getBaseEntity() { return Entity; }

    /// \brief Sets the "base" location and entity when that
    /// information is known based on another transformation.
    void setBase(SourceLocation Loc, DeclarationName Entity) {
      this->Loc = Loc;
      this->Entity = Entity;
    }
      
    /// \brief Transform the given declaration by instantiating a reference to
    /// this declaration.
    Decl *TransformDecl(Decl *D);

    /// \brief Transform the definition of the given declaration by
    /// instantiating it.
    Decl *TransformDefinition(Decl *D);

    /// \bried Transform the first qualifier within a scope by instantiating the
    /// declaration.
    NamedDecl *TransformFirstQualifierInScope(NamedDecl *D, SourceLocation Loc);
      
    /// \brief Rebuild the exception declaration and register the declaration
    /// as an instantiated local.
    VarDecl *RebuildExceptionDecl(VarDecl *ExceptionDecl, QualType T,
                                  TypeSourceInfo *Declarator,
                                  IdentifierInfo *Name,
                                  SourceLocation Loc, SourceRange TypeRange);

    /// \brief Check for tag mismatches when instantiating an
    /// elaborated type.
    QualType RebuildElaboratedType(QualType T, ElaboratedType::TagKind Tag);

    Sema::OwningExprResult TransformPredefinedExpr(PredefinedExpr *E);
    Sema::OwningExprResult TransformDeclRefExpr(DeclRefExpr *E);
    Sema::OwningExprResult TransformCXXDefaultArgExpr(CXXDefaultArgExpr *E);

    /// \brief Transforms a template type parameter type by performing
    /// substitution of the corresponding template type argument.
    QualType TransformTemplateTypeParmType(TypeLocBuilder &TLB,
                                           TemplateTypeParmTypeLoc TL);
  };
}

Decl *TemplateInstantiator::TransformDecl(Decl *D) {
  if (!D)
    return 0;

  if (TemplateTemplateParmDecl *TTP = dyn_cast<TemplateTemplateParmDecl>(D)) {
    if (TTP->getDepth() < TemplateArgs.getNumLevels()) {
      TemplateName Template
        = TemplateArgs(TTP->getDepth(), TTP->getPosition()).getAsTemplate();
      assert(!Template.isNull() && Template.getAsTemplateDecl() &&
             "Wrong kind of template template argument");
      return Template.getAsTemplateDecl();
    }

    // If the corresponding template argument is NULL or non-existent, it's
    // because we are performing instantiation from explicitly-specified
    // template arguments in a function template, but there were some
    // arguments left unspecified.
    if (!TemplateArgs.hasTemplateArgument(TTP->getDepth(),
                                          TTP->getPosition()))
      return D;

    // Fall through to find the instantiated declaration for this template
    // template parameter.
  }

  return SemaRef.FindInstantiatedDecl(cast<NamedDecl>(D), TemplateArgs);
}

Decl *TemplateInstantiator::TransformDefinition(Decl *D) {
  Decl *Inst = getSema().SubstDecl(D, getSema().CurContext, TemplateArgs);
  if (!Inst)
    return 0;

  getSema().CurrentInstantiationScope->InstantiatedLocal(D, Inst);
  return Inst;
}

NamedDecl *
TemplateInstantiator::TransformFirstQualifierInScope(NamedDecl *D, 
                                                     SourceLocation Loc) {
  // If the first part of the nested-name-specifier was a template type 
  // parameter, instantiate that type parameter down to a tag type.
  if (TemplateTypeParmDecl *TTPD = dyn_cast_or_null<TemplateTypeParmDecl>(D)) {
    const TemplateTypeParmType *TTP 
      = cast<TemplateTypeParmType>(getSema().Context.getTypeDeclType(TTPD));
    if (TTP->getDepth() < TemplateArgs.getNumLevels()) {
      QualType T = TemplateArgs(TTP->getDepth(), TTP->getIndex()).getAsType();
      if (T.isNull())
        return cast_or_null<NamedDecl>(TransformDecl(D));
      
      if (const TagType *Tag = T->getAs<TagType>())
        return Tag->getDecl();
      
      // The resulting type is not a tag; complain.
      getSema().Diag(Loc, diag::err_nested_name_spec_non_tag) << T;
      return 0;
    }
  }
  
  return cast_or_null<NamedDecl>(TransformDecl(D));
}

VarDecl *
TemplateInstantiator::RebuildExceptionDecl(VarDecl *ExceptionDecl,
                                           QualType T,
                                           TypeSourceInfo *Declarator,
                                           IdentifierInfo *Name,
                                           SourceLocation Loc,
                                           SourceRange TypeRange) {
  VarDecl *Var = inherited::RebuildExceptionDecl(ExceptionDecl, T, Declarator,
                                                 Name, Loc, TypeRange);
  if (Var && !Var->isInvalidDecl())
    getSema().CurrentInstantiationScope->InstantiatedLocal(ExceptionDecl, Var);
  return Var;
}

QualType
TemplateInstantiator::RebuildElaboratedType(QualType T,
                                            ElaboratedType::TagKind Tag) {
  if (const TagType *TT = T->getAs<TagType>()) {
    TagDecl* TD = TT->getDecl();

    // FIXME: this location is very wrong;  we really need typelocs.
    SourceLocation TagLocation = TD->getTagKeywordLoc();

    // FIXME: type might be anonymous.
    IdentifierInfo *Id = TD->getIdentifier();

    // TODO: should we even warn on struct/class mismatches for this?  Seems
    // like it's likely to produce a lot of spurious errors.
    if (!SemaRef.isAcceptableTagRedeclaration(TD, Tag, TagLocation, *Id)) {
      SemaRef.Diag(TagLocation, diag::err_use_with_wrong_tag)
        << Id
        << CodeModificationHint::CreateReplacement(SourceRange(TagLocation),
                                                   TD->getKindName());
      SemaRef.Diag(TD->getLocation(), diag::note_previous_use);
    }
  }

  return TreeTransform<TemplateInstantiator>::RebuildElaboratedType(T, Tag);
}

Sema::OwningExprResult 
TemplateInstantiator::TransformPredefinedExpr(PredefinedExpr *E) {
  if (!E->isTypeDependent())
    return SemaRef.Owned(E->Retain());

  FunctionDecl *currentDecl = getSema().getCurFunctionDecl();
  assert(currentDecl && "Must have current function declaration when "
                        "instantiating.");

  PredefinedExpr::IdentType IT = E->getIdentType();

  unsigned Length =
    PredefinedExpr::ComputeName(getSema().Context, IT, currentDecl).length();

  llvm::APInt LengthI(32, Length + 1);
  QualType ResTy = getSema().Context.CharTy.withConst();
  ResTy = getSema().Context.getConstantArrayType(ResTy, LengthI, 
                                                 ArrayType::Normal, 0);
  PredefinedExpr *PE =
    new (getSema().Context) PredefinedExpr(E->getLocation(), ResTy, IT);
  return getSema().Owned(PE);
}

Sema::OwningExprResult
TemplateInstantiator::TransformDeclRefExpr(DeclRefExpr *E) {
  // FIXME: Clean this up a bit
  NamedDecl *D = E->getDecl();
  if (NonTypeTemplateParmDecl *NTTP = dyn_cast<NonTypeTemplateParmDecl>(D)) {
    if (NTTP->getDepth() < TemplateArgs.getNumLevels()) {
      // If the corresponding template argument is NULL or non-existent, it's
      // because we are performing instantiation from explicitly-specified
      // template arguments in a function template, but there were some
      // arguments left unspecified.
      if (!TemplateArgs.hasTemplateArgument(NTTP->getDepth(),
                                            NTTP->getPosition()))
        return SemaRef.Owned(E->Retain());

      const TemplateArgument &Arg = TemplateArgs(NTTP->getDepth(),
                                                 NTTP->getPosition());

      // The template argument itself might be an expression, in which
      // case we just return that expression.
      if (Arg.getKind() == TemplateArgument::Expression)
        return SemaRef.Owned(Arg.getAsExpr()->Retain());

      if (Arg.getKind() == TemplateArgument::Declaration) {
        ValueDecl *VD = cast<ValueDecl>(Arg.getAsDecl());

        VD = cast_or_null<ValueDecl>(
                              getSema().FindInstantiatedDecl(VD, TemplateArgs));
        if (!VD)
          return SemaRef.ExprError();

        if (VD->getDeclContext()->isRecord()) {
          // If the value is a class member, we might have a pointer-to-member.
          // Determine whether the non-type template template parameter is of
          // pointer-to-member type. If so, we need to build an appropriate
          // expression for a pointer-to-member, since a "normal" DeclRefExpr
          // would refer to the member itself.
          if (NTTP->getType()->isMemberPointerType()) {
            QualType ClassType
              = SemaRef.Context.getTypeDeclType(
                                        cast<RecordDecl>(VD->getDeclContext()));
            NestedNameSpecifier *Qualifier
              = NestedNameSpecifier::Create(SemaRef.Context, 0, false,
                                            ClassType.getTypePtr());
            CXXScopeSpec SS;
            SS.setScopeRep(Qualifier);
            OwningExprResult RefExpr 
              = SemaRef.BuildDeclRefExpr(VD, 
                                         VD->getType().getNonReferenceType(), 
                                         E->getLocation(), 
                                         &SS);
            if (RefExpr.isInvalid())
              return SemaRef.ExprError();
              
            return SemaRef.CreateBuiltinUnaryOp(E->getLocation(), 
                                                UnaryOperator::AddrOf, 
                                                move(RefExpr));
          }
        }

        return SemaRef.BuildDeclRefExpr(VD, VD->getType().getNonReferenceType(),
                                        E->getLocation());
      }

      assert(Arg.getKind() == TemplateArgument::Integral);
      QualType T = Arg.getIntegralType();
      if (T->isCharType() || T->isWideCharType())
        return SemaRef.Owned(new (SemaRef.Context) CharacterLiteral(
                                              Arg.getAsIntegral()->getZExtValue(),
                                              T->isWideCharType(),
                                              T,
                                              E->getSourceRange().getBegin()));
      if (T->isBooleanType())
        return SemaRef.Owned(new (SemaRef.Context) CXXBoolLiteralExpr(
                                            Arg.getAsIntegral()->getBoolValue(),
                                            T,
                                            E->getSourceRange().getBegin()));

      assert(Arg.getAsIntegral()->getBitWidth() == SemaRef.Context.getIntWidth(T));
      return SemaRef.Owned(new (SemaRef.Context) IntegerLiteral(
                                                *Arg.getAsIntegral(),
                                                T,
                                                E->getSourceRange().getBegin()));
    }
    
    // We have a non-type template parameter that isn't fully substituted;
    // FindInstantiatedDecl will find it in the local instantiation scope.
  }

  return TreeTransform<TemplateInstantiator>::TransformDeclRefExpr(E);
}

Sema::OwningExprResult TemplateInstantiator::TransformCXXDefaultArgExpr(
    CXXDefaultArgExpr *E) {
  assert(!cast<FunctionDecl>(E->getParam()->getDeclContext())->
             getDescribedFunctionTemplate() &&
         "Default arg expressions are never formed in dependent cases.");
  return SemaRef.BuildCXXDefaultArgExpr(E->getUsedLocation(),
                           cast<FunctionDecl>(E->getParam()->getDeclContext()), 
                                        E->getParam());
}


QualType
TemplateInstantiator::TransformTemplateTypeParmType(TypeLocBuilder &TLB,
                                                TemplateTypeParmTypeLoc TL) {
  TemplateTypeParmType *T = TL.getTypePtr();
  if (T->getDepth() < TemplateArgs.getNumLevels()) {
    // Replace the template type parameter with its corresponding
    // template argument.

    // If the corresponding template argument is NULL or doesn't exist, it's
    // because we are performing instantiation from explicitly-specified
    // template arguments in a function template class, but there were some
    // arguments left unspecified.
    if (!TemplateArgs.hasTemplateArgument(T->getDepth(), T->getIndex())) {
      TemplateTypeParmTypeLoc NewTL
        = TLB.push<TemplateTypeParmTypeLoc>(TL.getType());
      NewTL.setNameLoc(TL.getNameLoc());
      return TL.getType();
    }

    assert(TemplateArgs(T->getDepth(), T->getIndex()).getKind()
             == TemplateArgument::Type &&
           "Template argument kind mismatch");

    QualType Replacement
      = TemplateArgs(T->getDepth(), T->getIndex()).getAsType();

    // TODO: only do this uniquing once, at the start of instantiation.
    QualType Result
      = getSema().Context.getSubstTemplateTypeParmType(T, Replacement);
    SubstTemplateTypeParmTypeLoc NewTL
      = TLB.push<SubstTemplateTypeParmTypeLoc>(Result);
    NewTL.setNameLoc(TL.getNameLoc());
    return Result;
  }

  // The template type parameter comes from an inner template (e.g.,
  // the template parameter list of a member template inside the
  // template we are instantiating). Create a new template type
  // parameter with the template "level" reduced by one.
  QualType Result
    = getSema().Context.getTemplateTypeParmType(T->getDepth()
                                                 - TemplateArgs.getNumLevels(),
                                                T->getIndex(),
                                                T->isParameterPack(),
                                                T->getName());
  TemplateTypeParmTypeLoc NewTL = TLB.push<TemplateTypeParmTypeLoc>(Result);
  NewTL.setNameLoc(TL.getNameLoc());
  return Result;
}

/// \brief Perform substitution on the type T with a given set of template
/// arguments.
///
/// This routine substitutes the given template arguments into the
/// type T and produces the instantiated type.
///
/// \param T the type into which the template arguments will be
/// substituted. If this type is not dependent, it will be returned
/// immediately.
///
/// \param TemplateArgs the template arguments that will be
/// substituted for the top-level template parameters within T.
///
/// \param Loc the location in the source code where this substitution
/// is being performed. It will typically be the location of the
/// declarator (if we're instantiating the type of some declaration)
/// or the location of the type in the source code (if, e.g., we're
/// instantiating the type of a cast expression).
///
/// \param Entity the name of the entity associated with a declaration
/// being instantiated (if any). May be empty to indicate that there
/// is no such entity (if, e.g., this is a type that occurs as part of
/// a cast expression) or that the entity has no name (e.g., an
/// unnamed function parameter).
///
/// \returns If the instantiation succeeds, the instantiated
/// type. Otherwise, produces diagnostics and returns a NULL type.
TypeSourceInfo *Sema::SubstType(TypeSourceInfo *T,
                                const MultiLevelTemplateArgumentList &Args,
                                SourceLocation Loc,
                                DeclarationName Entity) {
  assert(!ActiveTemplateInstantiations.empty() &&
         "Cannot perform an instantiation without some context on the "
         "instantiation stack");
  
  if (!T->getType()->isDependentType())
    return T;

  TemplateInstantiator Instantiator(*this, Args, Loc, Entity);
  return Instantiator.TransformType(T);
}

/// Deprecated form of the above.
QualType Sema::SubstType(QualType T,
                         const MultiLevelTemplateArgumentList &TemplateArgs,
                         SourceLocation Loc, DeclarationName Entity) {
  assert(!ActiveTemplateInstantiations.empty() &&
         "Cannot perform an instantiation without some context on the "
         "instantiation stack");

  // If T is not a dependent type, there is nothing to do.
  if (!T->isDependentType())
    return T;

  TemplateInstantiator Instantiator(*this, TemplateArgs, Loc, Entity);
  return Instantiator.TransformType(T);
}

/// \brief Perform substitution on the base class specifiers of the
/// given class template specialization.
///
/// Produces a diagnostic and returns true on error, returns false and
/// attaches the instantiated base classes to the class template
/// specialization if successful.
bool
Sema::SubstBaseSpecifiers(CXXRecordDecl *Instantiation,
                          CXXRecordDecl *Pattern,
                          const MultiLevelTemplateArgumentList &TemplateArgs) {
  bool Invalid = false;
  llvm::SmallVector<CXXBaseSpecifier*, 4> InstantiatedBases;
  for (ClassTemplateSpecializationDecl::base_class_iterator
         Base = Pattern->bases_begin(), BaseEnd = Pattern->bases_end();
       Base != BaseEnd; ++Base) {
    if (!Base->getType()->isDependentType()) {
      const CXXRecordDecl *BaseDecl =
        cast<CXXRecordDecl>(Base->getType()->getAs<RecordType>()->getDecl());
      
      // Make sure to set the attributes from the base.
      SetClassDeclAttributesFromBase(Instantiation, BaseDecl, 
                                     Base->isVirtual());
      
      InstantiatedBases.push_back(new (Context) CXXBaseSpecifier(*Base));
      continue;
    }

    QualType BaseType = SubstType(Base->getType(),
                                  TemplateArgs,
                                  Base->getSourceRange().getBegin(),
                                  DeclarationName());
    if (BaseType.isNull()) {
      Invalid = true;
      continue;
    }

    if (CXXBaseSpecifier *InstantiatedBase
          = CheckBaseSpecifier(Instantiation,
                               Base->getSourceRange(),
                               Base->isVirtual(),
                               Base->getAccessSpecifierAsWritten(),
                               BaseType,
                               /*FIXME: Not totally accurate */
                               Base->getSourceRange().getBegin()))
      InstantiatedBases.push_back(InstantiatedBase);
    else
      Invalid = true;
  }

  if (!Invalid &&
      AttachBaseSpecifiers(Instantiation, InstantiatedBases.data(),
                           InstantiatedBases.size()))
    Invalid = true;

  return Invalid;
}

/// \brief Instantiate the definition of a class from a given pattern.
///
/// \param PointOfInstantiation The point of instantiation within the
/// source code.
///
/// \param Instantiation is the declaration whose definition is being
/// instantiated. This will be either a class template specialization
/// or a member class of a class template specialization.
///
/// \param Pattern is the pattern from which the instantiation
/// occurs. This will be either the declaration of a class template or
/// the declaration of a member class of a class template.
///
/// \param TemplateArgs The template arguments to be substituted into
/// the pattern.
///
/// \param TSK the kind of implicit or explicit instantiation to perform.
///
/// \param Complain whether to complain if the class cannot be instantiated due
/// to the lack of a definition.
///
/// \returns true if an error occurred, false otherwise.
bool
Sema::InstantiateClass(SourceLocation PointOfInstantiation,
                       CXXRecordDecl *Instantiation, CXXRecordDecl *Pattern,
                       const MultiLevelTemplateArgumentList &TemplateArgs,
                       TemplateSpecializationKind TSK,
                       bool Complain) {
  bool Invalid = false;

  CXXRecordDecl *PatternDef
    = cast_or_null<CXXRecordDecl>(Pattern->getDefinition(Context));
  if (!PatternDef) {
    if (!Complain) {
      // Say nothing
    } else if (Pattern == Instantiation->getInstantiatedFromMemberClass()) {
      Diag(PointOfInstantiation,
           diag::err_implicit_instantiate_member_undefined)
        << Context.getTypeDeclType(Instantiation);
      Diag(Pattern->getLocation(), diag::note_member_of_template_here);
    } else {
      Diag(PointOfInstantiation, diag::err_template_instantiate_undefined)
        << (TSK != TSK_ImplicitInstantiation)
        << Context.getTypeDeclType(Instantiation);
      Diag(Pattern->getLocation(), diag::note_template_decl_here);
    }
    return true;
  }
  Pattern = PatternDef;

  // \brief Record the point of instantiation.
  if (MemberSpecializationInfo *MSInfo 
        = Instantiation->getMemberSpecializationInfo()) {
    MSInfo->setTemplateSpecializationKind(TSK);
    MSInfo->setPointOfInstantiation(PointOfInstantiation);
  } else if (ClassTemplateSpecializationDecl *Spec 
               = dyn_cast<ClassTemplateSpecializationDecl>(Instantiation)) {
    Spec->setTemplateSpecializationKind(TSK);
    Spec->setPointOfInstantiation(PointOfInstantiation);
  }
  
  InstantiatingTemplate Inst(*this, PointOfInstantiation, Instantiation);
  if (Inst)
    return true;

  // Enter the scope of this instantiation. We don't use
  // PushDeclContext because we don't have a scope.
  DeclContext *PreviousContext = CurContext;
  CurContext = Instantiation;

  // Start the definition of this instantiation.
  Instantiation->startDefinition();

  // Do substitution on the base class specifiers.
  if (SubstBaseSpecifiers(Instantiation, Pattern, TemplateArgs))
    Invalid = true;

  llvm::SmallVector<DeclPtrTy, 4> Fields;
  for (RecordDecl::decl_iterator Member = Pattern->decls_begin(),
         MemberEnd = Pattern->decls_end();
       Member != MemberEnd; ++Member) {
    Decl *NewMember = SubstDecl(*Member, Instantiation, TemplateArgs);
    if (NewMember) {
      if (FieldDecl *Field = dyn_cast<FieldDecl>(NewMember))
        Fields.push_back(DeclPtrTy::make(Field));
      else if (NewMember->isInvalidDecl())
        Invalid = true;
    } else {
      // FIXME: Eventually, a NULL return will mean that one of the
      // instantiations was a semantic disaster, and we'll want to set Invalid =
      // true. For now, we expect to skip some members that we can't yet handle.
    }
  }

  // Finish checking fields.
  ActOnFields(0, Instantiation->getLocation(), DeclPtrTy::make(Instantiation),
              Fields.data(), Fields.size(), SourceLocation(), SourceLocation(),
              0);
  CheckCompletedCXXClass(Instantiation);
  if (Instantiation->isInvalidDecl())
    Invalid = true;
  
  // Exit the scope of this instantiation.
  CurContext = PreviousContext;

  if (!Invalid)
    Consumer.HandleTagDeclDefinition(Instantiation);

  return Invalid;
}

bool
Sema::InstantiateClassTemplateSpecialization(
                           SourceLocation PointOfInstantiation,
                           ClassTemplateSpecializationDecl *ClassTemplateSpec,
                           TemplateSpecializationKind TSK,
                           bool Complain) {
  // Perform the actual instantiation on the canonical declaration.
  ClassTemplateSpec = cast<ClassTemplateSpecializationDecl>(
                                         ClassTemplateSpec->getCanonicalDecl());

  // Check whether we have already instantiated or specialized this class
  // template specialization.
  if (ClassTemplateSpec->getSpecializationKind() != TSK_Undeclared) {
    if (ClassTemplateSpec->getSpecializationKind() == 
          TSK_ExplicitInstantiationDeclaration &&
        TSK == TSK_ExplicitInstantiationDefinition) {
      // An explicit instantiation definition follows an explicit instantiation
      // declaration (C++0x [temp.explicit]p10); go ahead and perform the
      // explicit instantiation.
      ClassTemplateSpec->setSpecializationKind(TSK);
      return false;
    }
    
    // We can only instantiate something that hasn't already been
    // instantiated or specialized. Fail without any diagnostics: our
    // caller will provide an error message.    
    return true;
  }

  if (ClassTemplateSpec->isInvalidDecl())
    return true;
  
  ClassTemplateDecl *Template = ClassTemplateSpec->getSpecializedTemplate();
  CXXRecordDecl *Pattern = 0;

  // C++ [temp.class.spec.match]p1:
  //   When a class template is used in a context that requires an
  //   instantiation of the class, it is necessary to determine
  //   whether the instantiation is to be generated using the primary
  //   template or one of the partial specializations. This is done by
  //   matching the template arguments of the class template
  //   specialization with the template argument lists of the partial
  //   specializations.
  typedef std::pair<ClassTemplatePartialSpecializationDecl *,
                    TemplateArgumentList *> MatchResult;
  llvm::SmallVector<MatchResult, 4> Matched;
  for (llvm::FoldingSet<ClassTemplatePartialSpecializationDecl>::iterator
         Partial = Template->getPartialSpecializations().begin(),
         PartialEnd = Template->getPartialSpecializations().end();
       Partial != PartialEnd;
       ++Partial) {
    TemplateDeductionInfo Info(Context);
    if (TemplateDeductionResult Result
          = DeduceTemplateArguments(&*Partial,
                                    ClassTemplateSpec->getTemplateArgs(),
                                    Info)) {
      // FIXME: Store the failed-deduction information for use in
      // diagnostics, later.
      (void)Result;
    } else {
      Matched.push_back(std::make_pair(&*Partial, Info.take()));
    }
  }

  if (Matched.size() >= 1) {
    llvm::SmallVector<MatchResult, 4>::iterator Best = Matched.begin();
    if (Matched.size() == 1) {
      //   -- If exactly one matching specialization is found, the
      //      instantiation is generated from that specialization.
      // We don't need to do anything for this.
    } else {
      //   -- If more than one matching specialization is found, the
      //      partial order rules (14.5.4.2) are used to determine
      //      whether one of the specializations is more specialized
      //      than the others. If none of the specializations is more
      //      specialized than all of the other matching
      //      specializations, then the use of the class template is
      //      ambiguous and the program is ill-formed.
      for (llvm::SmallVector<MatchResult, 4>::iterator P = Best + 1,
                                                    PEnd = Matched.end();
           P != PEnd; ++P) {
        if (getMoreSpecializedPartialSpecialization(P->first, Best->first) 
              == P->first)
          Best = P;
      }
      
      // Determine if the best partial specialization is more specialized than
      // the others.
      bool Ambiguous = false;
      for (llvm::SmallVector<MatchResult, 4>::iterator P = Matched.begin(),
                                                    PEnd = Matched.end();
           P != PEnd; ++P) {
        if (P != Best &&
            getMoreSpecializedPartialSpecialization(P->first, Best->first)
              != Best->first) {
          Ambiguous = true;
          break;
        }
      }
       
      if (Ambiguous) {
        // Partial ordering did not produce a clear winner. Complain.
        ClassTemplateSpec->setInvalidDecl();
        Diag(PointOfInstantiation, diag::err_partial_spec_ordering_ambiguous)
          << ClassTemplateSpec;
        
        // Print the matching partial specializations.
        for (llvm::SmallVector<MatchResult, 4>::iterator P = Matched.begin(),
                                                      PEnd = Matched.end();
             P != PEnd; ++P)
          Diag(P->first->getLocation(), diag::note_partial_spec_match)
            << getTemplateArgumentBindingsText(P->first->getTemplateParameters(),
                                               *P->second);

        return true;
      }
    }
    
    // Instantiate using the best class template partial specialization.
    ClassTemplatePartialSpecializationDecl *OrigPartialSpec = Best->first;
    while (OrigPartialSpec->getInstantiatedFromMember()) {
      // If we've found an explicit specialization of this class template,
      // stop here and use that as the pattern.
      if (OrigPartialSpec->isMemberSpecialization())
        break;
      
      OrigPartialSpec = OrigPartialSpec->getInstantiatedFromMember();
    }
    
    Pattern = OrigPartialSpec;
    ClassTemplateSpec->setInstantiationOf(Best->first, Best->second);
  } else {
    //   -- If no matches are found, the instantiation is generated
    //      from the primary template.
    ClassTemplateDecl *OrigTemplate = Template;
    while (OrigTemplate->getInstantiatedFromMemberTemplate()) {
      // If we've found an explicit specialization of this class template,
      // stop here and use that as the pattern.
      if (OrigTemplate->isMemberSpecialization())
        break;
      
      OrigTemplate = OrigTemplate->getInstantiatedFromMemberTemplate();
    }
    
    Pattern = OrigTemplate->getTemplatedDecl();
  }

  bool Result = InstantiateClass(PointOfInstantiation, ClassTemplateSpec, 
                                 Pattern,
                                getTemplateInstantiationArgs(ClassTemplateSpec),
                                 TSK,
                                 Complain);

  for (unsigned I = 0, N = Matched.size(); I != N; ++I) {
    // FIXME: Implement TemplateArgumentList::Destroy!
    //    if (Matched[I].first != Pattern)
    //      Matched[I].second->Destroy(Context);
  }

  return Result;
}

/// \brief Instantiates the definitions of all of the member
/// of the given class, which is an instantiation of a class template
/// or a member class of a template.
void
Sema::InstantiateClassMembers(SourceLocation PointOfInstantiation,
                              CXXRecordDecl *Instantiation,
                        const MultiLevelTemplateArgumentList &TemplateArgs,
                              TemplateSpecializationKind TSK) {
  for (DeclContext::decl_iterator D = Instantiation->decls_begin(),
                               DEnd = Instantiation->decls_end();
       D != DEnd; ++D) {
    bool SuppressNew = false;
    if (FunctionDecl *Function = dyn_cast<FunctionDecl>(*D)) {
      if (FunctionDecl *Pattern
            = Function->getInstantiatedFromMemberFunction()) {
        MemberSpecializationInfo *MSInfo 
          = Function->getMemberSpecializationInfo();
        assert(MSInfo && "No member specialization information?");
        if (CheckSpecializationInstantiationRedecl(PointOfInstantiation, TSK, 
                                                   Function, 
                                        MSInfo->getTemplateSpecializationKind(),
                                              MSInfo->getPointOfInstantiation(), 
                                                   SuppressNew) ||
            SuppressNew)
          continue;
        
        if (Function->getBody())
          continue;

        if (TSK == TSK_ExplicitInstantiationDefinition) {
          // C++0x [temp.explicit]p8:
          //   An explicit instantiation definition that names a class template
          //   specialization explicitly instantiates the class template 
          //   specialization and is only an explicit instantiation definition 
          //   of members whose definition is visible at the point of 
          //   instantiation.
          if (!Pattern->getBody())
            continue;
        
          Function->setTemplateSpecializationKind(TSK, PointOfInstantiation);
                      
          InstantiateFunctionDefinition(PointOfInstantiation, Function);
        } else {
          Function->setTemplateSpecializationKind(TSK, PointOfInstantiation);
        }
      }
    } else if (VarDecl *Var = dyn_cast<VarDecl>(*D)) {
      if (Var->isStaticDataMember()) {
        MemberSpecializationInfo *MSInfo = Var->getMemberSpecializationInfo();
        assert(MSInfo && "No member specialization information?");
        if (CheckSpecializationInstantiationRedecl(PointOfInstantiation, TSK, 
                                                   Var, 
                                        MSInfo->getTemplateSpecializationKind(),
                                              MSInfo->getPointOfInstantiation(), 
                                                   SuppressNew) ||
            SuppressNew)
          continue;
        
        if (TSK == TSK_ExplicitInstantiationDefinition) {
          // C++0x [temp.explicit]p8:
          //   An explicit instantiation definition that names a class template
          //   specialization explicitly instantiates the class template 
          //   specialization and is only an explicit instantiation definition 
          //   of members whose definition is visible at the point of 
          //   instantiation.
          if (!Var->getInstantiatedFromStaticDataMember()
                                                     ->getOutOfLineDefinition())
            continue;
          
          Var->setTemplateSpecializationKind(TSK, PointOfInstantiation);
          InstantiateStaticDataMemberDefinition(PointOfInstantiation, Var);
        } else {
          Var->setTemplateSpecializationKind(TSK, PointOfInstantiation);
        }
      }      
    } else if (CXXRecordDecl *Record = dyn_cast<CXXRecordDecl>(*D)) {
      if (Record->isInjectedClassName())
        continue;
      
      MemberSpecializationInfo *MSInfo = Record->getMemberSpecializationInfo();
      assert(MSInfo && "No member specialization information?");
      if (CheckSpecializationInstantiationRedecl(PointOfInstantiation, TSK, 
                                                 Record, 
                                        MSInfo->getTemplateSpecializationKind(),
                                              MSInfo->getPointOfInstantiation(), 
                                                 SuppressNew) ||
          SuppressNew)
        continue;
      
      CXXRecordDecl *Pattern = Record->getInstantiatedFromMemberClass();
      assert(Pattern && "Missing instantiated-from-template information");
      
      if (!Record->getDefinition(Context)) {
        if (!Pattern->getDefinition(Context)) {
          // C++0x [temp.explicit]p8:
          //   An explicit instantiation definition that names a class template
          //   specialization explicitly instantiates the class template 
          //   specialization and is only an explicit instantiation definition 
          //   of members whose definition is visible at the point of 
          //   instantiation.
          if (TSK == TSK_ExplicitInstantiationDeclaration) {
            MSInfo->setTemplateSpecializationKind(TSK);
            MSInfo->setPointOfInstantiation(PointOfInstantiation);
          }
          
          continue;
        }
        
        InstantiateClass(PointOfInstantiation, Record, Pattern,
                         TemplateArgs,
                         TSK);
      }
      
      Pattern = cast_or_null<CXXRecordDecl>(Record->getDefinition(Context));
      if (Pattern)
        InstantiateClassMembers(PointOfInstantiation, Pattern, TemplateArgs, 
                                TSK);
    }
  }
}

/// \brief Instantiate the definitions of all of the members of the
/// given class template specialization, which was named as part of an
/// explicit instantiation.
void
Sema::InstantiateClassTemplateSpecializationMembers(
                                           SourceLocation PointOfInstantiation,
                            ClassTemplateSpecializationDecl *ClassTemplateSpec,
                                               TemplateSpecializationKind TSK) {
  // C++0x [temp.explicit]p7:
  //   An explicit instantiation that names a class template
  //   specialization is an explicit instantion of the same kind
  //   (declaration or definition) of each of its members (not
  //   including members inherited from base classes) that has not
  //   been previously explicitly specialized in the translation unit
  //   containing the explicit instantiation, except as described
  //   below.
  InstantiateClassMembers(PointOfInstantiation, ClassTemplateSpec,
                          getTemplateInstantiationArgs(ClassTemplateSpec),
                          TSK);
}

Sema::OwningStmtResult
Sema::SubstStmt(Stmt *S, const MultiLevelTemplateArgumentList &TemplateArgs) {
  if (!S)
    return Owned(S);

  TemplateInstantiator Instantiator(*this, TemplateArgs,
                                    SourceLocation(),
                                    DeclarationName());
  return Instantiator.TransformStmt(S);
}

Sema::OwningExprResult
Sema::SubstExpr(Expr *E, const MultiLevelTemplateArgumentList &TemplateArgs) {
  if (!E)
    return Owned(E);

  TemplateInstantiator Instantiator(*this, TemplateArgs,
                                    SourceLocation(),
                                    DeclarationName());
  return Instantiator.TransformExpr(E);
}

/// \brief Do template substitution on a nested-name-specifier.
NestedNameSpecifier *
Sema::SubstNestedNameSpecifier(NestedNameSpecifier *NNS,
                               SourceRange Range,
                         const MultiLevelTemplateArgumentList &TemplateArgs) {
  TemplateInstantiator Instantiator(*this, TemplateArgs, Range.getBegin(),
                                    DeclarationName());
  return Instantiator.TransformNestedNameSpecifier(NNS, Range);
}

TemplateName
Sema::SubstTemplateName(TemplateName Name, SourceLocation Loc,
                        const MultiLevelTemplateArgumentList &TemplateArgs) {
  TemplateInstantiator Instantiator(*this, TemplateArgs, Loc,
                                    DeclarationName());
  return Instantiator.TransformTemplateName(Name);
}

bool Sema::Subst(const TemplateArgumentLoc &Input, TemplateArgumentLoc &Output,
                 const MultiLevelTemplateArgumentList &TemplateArgs) {
  TemplateInstantiator Instantiator(*this, TemplateArgs, SourceLocation(),
                                    DeclarationName());

  return Instantiator.TransformTemplateArgument(Input, Output);
}
