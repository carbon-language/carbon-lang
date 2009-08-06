//===--- SemaCXXScopeSpec.cpp - Semantic Analysis for C++ scope specifiers-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements C++ semantic analysis for scope specifiers.
//
//===----------------------------------------------------------------------===//

#include "Sema.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/NestedNameSpecifier.h"
#include "clang/Parse/DeclSpec.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
using namespace clang;

/// \brief Compute the DeclContext that is associated with the given
/// scope specifier.
///
/// \param SS the C++ scope specifier as it appears in the source
///
/// \param EnteringContext when true, we will be entering the context of
/// this scope specifier, so we can retrieve the declaration context of a
/// class template or class template partial specialization even if it is
/// not the current instantiation.
///
/// \returns the declaration context represented by the scope specifier @p SS,
/// or NULL if the declaration context cannot be computed (e.g., because it is
/// dependent and not the current instantiation).
DeclContext *Sema::computeDeclContext(const CXXScopeSpec &SS,
                                      bool EnteringContext) {
  if (!SS.isSet() || SS.isInvalid())
    return 0;

  NestedNameSpecifier *NNS 
    = static_cast<NestedNameSpecifier *>(SS.getScopeRep());
  if (NNS->isDependent()) {
    // If this nested-name-specifier refers to the current
    // instantiation, return its DeclContext.
    if (CXXRecordDecl *Record = getCurrentInstantiationOf(NNS))
      return Record;
    
    if (EnteringContext) {
      // We are entering the context of the nested name specifier, so try to
      // match the nested name specifier to either a primary class template
      // or a class template partial specialization.
      if (const TemplateSpecializationType *SpecType
            = dyn_cast_or_null<TemplateSpecializationType>(NNS->getAsType())) {
        if (ClassTemplateDecl *ClassTemplate 
              = dyn_cast_or_null<ClassTemplateDecl>(
                            SpecType->getTemplateName().getAsTemplateDecl())) {
          QualType ContextType
            = Context.getCanonicalType(QualType(SpecType, 0));

          // If the type of the nested name specifier is the same as the
          // injected class name of the named class template, we're entering
          // into that class template definition.
          QualType Injected = ClassTemplate->getInjectedClassNameType(Context);
          if (Context.hasSameType(Injected, ContextType))
            return ClassTemplate->getTemplatedDecl();
                
          // If the type of the nested name specifier is the same as the
          // type of one of the class template's class template partial
          // specializations, we're entering into the definition of that
          // class template partial specialization.
          if (ClassTemplatePartialSpecializationDecl *PartialSpec
                = ClassTemplate->findPartialSpecialization(ContextType))
            return PartialSpec;
        }
      }
      
      std::string NNSString;
      {
        llvm::raw_string_ostream OS(NNSString);
        NNS->print(OS, Context.PrintingPolicy);
      }
      
      // FIXME: Allow us to pass a nested-name-specifier to Diag?
      Diag(SS.getRange().getBegin(), 
           diag::err_template_qualified_declarator_no_match)
        << NNSString << SS.getRange();
    }
    
    return 0;
  }

  switch (NNS->getKind()) {
  case NestedNameSpecifier::Identifier:
    assert(false && "Dependent nested-name-specifier has no DeclContext");
    break;

  case NestedNameSpecifier::Namespace:
    return NNS->getAsNamespace();

  case NestedNameSpecifier::TypeSpec:
  case NestedNameSpecifier::TypeSpecWithTemplate: {
    const TagType *Tag = NNS->getAsType()->getAs<TagType>();
    assert(Tag && "Non-tag type in nested-name-specifier");
    return Tag->getDecl();
  } break;

  case NestedNameSpecifier::Global:
    return Context.getTranslationUnitDecl();
  }

  // Required to silence a GCC warning.
  return 0;
}

bool Sema::isDependentScopeSpecifier(const CXXScopeSpec &SS) {
  if (!SS.isSet() || SS.isInvalid())
    return false;

  NestedNameSpecifier *NNS 
    = static_cast<NestedNameSpecifier *>(SS.getScopeRep());
  return NNS->isDependent();
}

// \brief Determine whether this C++ scope specifier refers to an
// unknown specialization, i.e., a dependent type that is not the
// current instantiation.
bool Sema::isUnknownSpecialization(const CXXScopeSpec &SS) {
  if (!isDependentScopeSpecifier(SS))
    return false;

  NestedNameSpecifier *NNS 
    = static_cast<NestedNameSpecifier *>(SS.getScopeRep());
  return getCurrentInstantiationOf(NNS) == 0;
}

/// \brief If the given nested name specifier refers to the current
/// instantiation, return the declaration that corresponds to that
/// current instantiation (C++0x [temp.dep.type]p1).
///
/// \param NNS a dependent nested name specifier.
CXXRecordDecl *Sema::getCurrentInstantiationOf(NestedNameSpecifier *NNS) {
  assert(getLangOptions().CPlusPlus && "Only callable in C++");
  assert(NNS->isDependent() && "Only dependent nested-name-specifier allowed");

  if (!NNS->getAsType())
    return 0;
  
  QualType T = QualType(NNS->getAsType(), 0);
  // If the nested name specifier does not refer to a type, then it
  // does not refer to the current instantiation.
  if (T.isNull())
    return 0;

  T = Context.getCanonicalType(T);

  for (DeclContext *Ctx = CurContext; Ctx; Ctx = Ctx->getParent()) {
    // If we've hit a namespace or the global scope, then the
    // nested-name-specifier can't refer to the current instantiation.
    if (Ctx->isFileContext())
      return 0;

    // Skip non-class contexts.
    CXXRecordDecl *Record = dyn_cast<CXXRecordDecl>(Ctx);
    if (!Record)
      continue;

    // If this record type is not dependent, 
    if (!Record->isDependentType())
      return 0;

    // C++ [temp.dep.type]p1:
    //
    //   In the definition of a class template, a nested class of a
    //   class template, a member of a class template, or a member of a
    //   nested class of a class template, a name refers to the current
    //   instantiation if it is
    //     -- the injected-class-name (9) of the class template or
    //        nested class,
    //     -- in the definition of a primary class template, the name
    //        of the class template followed by the template argument
    //        list of the primary template (as described below)
    //        enclosed in <>,
    //     -- in the definition of a nested class of a class template,
    //        the name of the nested class referenced as a member of
    //        the current instantiation, or 
    //     -- in the definition of a partial specialization, the name
    //        of the class template followed by the template argument
    //        list of the partial specialization enclosed in <>. If
    //        the nth template parameter is a parameter pack, the nth
    //        template argument is a pack expansion (14.6.3) whose
    //        pattern is the name of the parameter pack. 
    //        (FIXME: parameter packs)
    //
    // All of these options come down to having the
    // nested-name-specifier type that is equivalent to the
    // injected-class-name of one of the types that is currently in
    // our context.
    if (Context.getCanonicalType(Context.getTypeDeclType(Record)) == T)
      return Record;
    
    if (ClassTemplateDecl *Template = Record->getDescribedClassTemplate()) {
      QualType InjectedClassName 
        = Template->getInjectedClassNameType(Context);
      if (T == Context.getCanonicalType(InjectedClassName))
        return Template->getTemplatedDecl();
    }
    // FIXME: check for class template partial specializations
  }

  return 0;
}

/// \brief Require that the context specified by SS be complete.
///
/// If SS refers to a type, this routine checks whether the type is
/// complete enough (or can be made complete enough) for name lookup
/// into the DeclContext. A type that is not yet completed can be
/// considered "complete enough" if it is a class/struct/union/enum
/// that is currently being defined. Or, if we have a type that names
/// a class template specialization that is not a complete type, we
/// will attempt to instantiate that class template.
bool Sema::RequireCompleteDeclContext(const CXXScopeSpec &SS) {
  if (!SS.isSet() || SS.isInvalid())
    return false;
  
  DeclContext *DC = computeDeclContext(SS, true);
  if (TagDecl *Tag = dyn_cast<TagDecl>(DC)) {
    // If we're currently defining this type, then lookup into the
    // type is okay: don't complain that it isn't complete yet.
    const TagType *TagT = Context.getTypeDeclType(Tag)->getAs<TagType>();
    if (TagT->isBeingDefined())
      return false;

    // The type must be complete.
    return RequireCompleteType(SS.getRange().getBegin(),
                               Context.getTypeDeclType(Tag),
                               diag::err_incomplete_nested_name_spec,
                               SS.getRange());
  }

  return false;
}

/// ActOnCXXGlobalScopeSpecifier - Return the object that represents the
/// global scope ('::').
Sema::CXXScopeTy *Sema::ActOnCXXGlobalScopeSpecifier(Scope *S,
                                                     SourceLocation CCLoc) {
  return NestedNameSpecifier::GlobalSpecifier(Context);
}

/// ActOnCXXNestedNameSpecifier - Called during parsing of a
/// nested-name-specifier. e.g. for "foo::bar::" we parsed "foo::" and now
/// we want to resolve "bar::". 'SS' is empty or the previously parsed
/// nested-name part ("foo::"), 'IdLoc' is the source location of 'bar',
/// 'CCLoc' is the location of '::' and 'II' is the identifier for 'bar'.
/// Returns a CXXScopeTy* object representing the C++ scope.
Sema::CXXScopeTy *Sema::ActOnCXXNestedNameSpecifier(Scope *S,
                                                    const CXXScopeSpec &SS,
                                                    SourceLocation IdLoc,
                                                    SourceLocation CCLoc,
                                                    IdentifierInfo &II) {
  NestedNameSpecifier *Prefix 
    = static_cast<NestedNameSpecifier *>(SS.getScopeRep());

  // If the prefix already refers to an unknown specialization, there
  // is no name lookup to perform. Just build the resulting
  // nested-name-specifier.
  if (Prefix && isUnknownSpecialization(SS))
    return NestedNameSpecifier::Create(Context, Prefix, &II);

  NamedDecl *SD = LookupParsedName(S, &SS, &II, LookupNestedNameSpecifierName);

  if (SD) {
    if (NamespaceDecl *Namespace = dyn_cast<NamespaceDecl>(SD))
      return NestedNameSpecifier::Create(Context, Prefix, Namespace);

    if (TypeDecl *Type = dyn_cast<TypeDecl>(SD)) {
      // Determine whether we have a class (or, in C++0x, an enum) or
      // a typedef thereof. If so, build the nested-name-specifier.
      QualType T = Context.getTypeDeclType(Type);
      bool AcceptableType = false;
      if (T->isDependentType())
        AcceptableType = true;
      else if (TypedefDecl *TD = dyn_cast<TypedefDecl>(SD)) {
        if (TD->getUnderlyingType()->isRecordType() ||
            (getLangOptions().CPlusPlus0x && 
             TD->getUnderlyingType()->isEnumeralType()))
          AcceptableType = true;
      } else if (isa<RecordDecl>(Type) || 
                 (getLangOptions().CPlusPlus0x && isa<EnumDecl>(Type)))
        AcceptableType = true;

      if (AcceptableType)
        return NestedNameSpecifier::Create(Context, Prefix, false, 
                                           T.getTypePtr());
    }
    
    if (NamespaceAliasDecl *Alias = dyn_cast<NamespaceAliasDecl>(SD))
      return NestedNameSpecifier::Create(Context, Prefix,
                                         Alias->getNamespace());

    // Fall through to produce an error: we found something that isn't
    // a class or a namespace.
  }

  // If we didn't find anything during our lookup, try again with
  // ordinary name lookup, which can help us produce better error
  // messages.
  if (!SD)
    SD = LookupParsedName(S, &SS, &II, LookupOrdinaryName);
  unsigned DiagID;
  if (SD)
    DiagID = diag::err_expected_class_or_namespace;
  else if (SS.isSet())
    DiagID = diag::err_typecheck_no_member;
  else
    DiagID = diag::err_undeclared_var_use;

  if (SS.isSet())
    Diag(IdLoc, DiagID) << &II << SS.getRange();
  else
    Diag(IdLoc, DiagID) << &II;

  return 0;
}

Sema::CXXScopeTy *Sema::ActOnCXXNestedNameSpecifier(Scope *S,
                                                    const CXXScopeSpec &SS,
                                                    TypeTy *Ty,
                                                    SourceRange TypeRange,
                                                    SourceLocation CCLoc) {
  NestedNameSpecifier *Prefix 
    = static_cast<NestedNameSpecifier *>(SS.getScopeRep());
  QualType T = QualType::getFromOpaquePtr(Ty);
  return NestedNameSpecifier::Create(Context, Prefix, /*FIXME:*/false,
                                     T.getTypePtr());
}

Action::OwningExprResult
Sema::ActOnCXXEnterMemberScope(Scope *S, CXXScopeSpec &SS, ExprArg Base,
                               tok::TokenKind OpKind) {
  Expr *BaseExpr = (Expr*)Base.get();
  assert(BaseExpr && "no record expansion");

  QualType BaseType = BaseExpr->getType();
  // FIXME: handle dependent types
  if (BaseType->isDependentType())
    return move(Base);

  // C++ [over.match.oper]p8:
  //   [...] When operator->returns, the operator-> is applied  to the value 
  //   returned, with the original second operand.
  if (OpKind == tok::arrow) {
    while (BaseType->isRecordType()) {
      Base = BuildOverloadedArrowExpr(S, move(Base), BaseExpr->getExprLoc());
      BaseExpr = (Expr*)Base.get();
      if (BaseExpr == NULL)
          return ExprError();
      BaseType = BaseExpr->getType();
    }
  }

  if (BaseType->isPointerType())
    BaseType = BaseType->getPointeeType();

  // We could end up with various non-record types here, such as extended 
  // vector types or Objective-C interfaces. Just return early and let
  // ActOnMemberReferenceExpr do the work.
  if (!BaseType->isRecordType())
    return move(Base);

  SS.setRange(BaseExpr->getSourceRange());
  SS.setScopeRep(
    NestedNameSpecifier::Create(Context, 0, false, BaseType.getTypePtr())
    );

  if (S)
    ActOnCXXEnterDeclaratorScope(S,SS);
  return move(Base);
}

void Sema::ActOnCXXExitMemberScope(Scope *S, const CXXScopeSpec &SS) {
  if (S && SS.isSet())
    ActOnCXXExitDeclaratorScope(S,SS);
}


/// ActOnCXXEnterDeclaratorScope - Called when a C++ scope specifier (global
/// scope or nested-name-specifier) is parsed, part of a declarator-id.
/// After this method is called, according to [C++ 3.4.3p3], names should be
/// looked up in the declarator-id's scope, until the declarator is parsed and
/// ActOnCXXExitDeclaratorScope is called.
/// The 'SS' should be a non-empty valid CXXScopeSpec.
void Sema::ActOnCXXEnterDeclaratorScope(Scope *S, const CXXScopeSpec &SS) {
  assert(SS.isSet() && "Parser passed invalid CXXScopeSpec.");
  if (DeclContext *DC = computeDeclContext(SS, true))
    EnterDeclaratorContext(S, DC);
  else
    const_cast<CXXScopeSpec&>(SS).setScopeRep(0);
}

/// ActOnCXXExitDeclaratorScope - Called when a declarator that previously
/// invoked ActOnCXXEnterDeclaratorScope(), is finished. 'SS' is the same
/// CXXScopeSpec that was passed to ActOnCXXEnterDeclaratorScope as well.
/// Used to indicate that names should revert to being looked up in the
/// defining scope.
void Sema::ActOnCXXExitDeclaratorScope(Scope *S, const CXXScopeSpec &SS) {
  assert(SS.isSet() && "Parser passed invalid CXXScopeSpec.");
  assert((SS.isInvalid() || S->getEntity() == computeDeclContext(SS, true)) && 
         "Context imbalance!");
  if (!SS.isInvalid())
    ExitDeclaratorContext(S);
}
