//===--- SemaDecl.cpp - Semantic Analysis for Declarations ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements semantic analysis for declarations.
//
//===----------------------------------------------------------------------===//

#include "Sema.h"
#include "clang/AST/APValue.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/ExprCXX.h"
#include "clang/Parse/DeclSpec.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/SourceManager.h"
// FIXME: layering (ideally, Sema shouldn't be dependent on Lex API's)
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/HeaderSearch.h" 
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <functional>

using namespace clang;

Sema::TypeTy *Sema::isTypeName(IdentifierInfo &II, Scope *S,
                               const CXXScopeSpec *SS) {
  DeclContext *DC = 0;
  if (SS) {
    if (SS->isInvalid())
      return 0;
    DC = static_cast<DeclContext*>(SS->getScopeRep());
  }
  LookupResult Result = LookupDecl(&II, Decl::IDNS_Ordinary, S, DC, false);

  Decl *IIDecl = 0;
  switch (Result.getKind()) {
  case LookupResult::NotFound:
  case LookupResult::FoundOverloaded:
  case LookupResult::AmbiguousBaseSubobjectTypes:
  case LookupResult::AmbiguousBaseSubobjects:
    // FIXME: In the event of an ambiguous lookup, we could visit all of
    // the entities found to determine whether they are all types. This
    // might provide better diagnostics.
    return 0;

  case LookupResult::Found:
    IIDecl = Result.getAsDecl();
    break;
  }

  if (isa<TypedefDecl>(IIDecl) || 
      isa<ObjCInterfaceDecl>(IIDecl) ||
      isa<TagDecl>(IIDecl) ||
      isa<TemplateTypeParmDecl>(IIDecl))
    return IIDecl;
  return 0;
}

DeclContext *Sema::getContainingDC(DeclContext *DC) {
  if (CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(DC)) {
    // A C++ out-of-line method will return to the file declaration context.
    if (MD->isOutOfLineDefinition())
      return MD->getLexicalDeclContext();

    // A C++ inline method is parsed *after* the topmost class it was declared in
    // is fully parsed (it's "complete").
    // The parsing of a C++ inline method happens at the declaration context of
    // the topmost (non-nested) class it is lexically declared in.
    assert(isa<CXXRecordDecl>(MD->getParent()) && "C++ method not in Record.");
    DC = MD->getParent();
    while (CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(DC->getLexicalParent()))
      DC = RD;

    // Return the declaration context of the topmost class the inline method is
    // declared in.
    return DC;
  }

  if (isa<ObjCMethodDecl>(DC))
    return Context.getTranslationUnitDecl();

  if (Decl *D = dyn_cast<Decl>(DC))
    return D->getLexicalDeclContext();

  return DC->getLexicalParent();
}

void Sema::PushDeclContext(Scope *S, DeclContext *DC) {
  assert(getContainingDC(DC) == CurContext &&
      "The next DeclContext should be lexically contained in the current one.");
  CurContext = DC;
  S->setEntity(DC);
}

void Sema::PopDeclContext() {
  assert(CurContext && "DeclContext imbalance!");

  CurContext = getContainingDC(CurContext);
}

/// Add this decl to the scope shadowed decl chains.
void Sema::PushOnScopeChains(NamedDecl *D, Scope *S) {
  // Move up the scope chain until we find the nearest enclosing
  // non-transparent context. The declaration will be introduced into this
  // scope.
  while (S->getEntity() && 
         ((DeclContext *)S->getEntity())->isTransparentContext())
    S = S->getParent();

  S->AddDecl(D);

  // Add scoped declarations into their context, so that they can be
  // found later. Declarations without a context won't be inserted
  // into any context.
  CurContext->addDecl(D);

  // C++ [basic.scope]p4:
  //   -- exactly one declaration shall declare a class name or
  //   enumeration name that is not a typedef name and the other
  //   declarations shall all refer to the same object or
  //   enumerator, or all refer to functions and function templates;
  //   in this case the class name or enumeration name is hidden.
  if (TagDecl *TD = dyn_cast<TagDecl>(D)) {
    // We are pushing the name of a tag (enum or class).
    if (CurContext->getLookupContext() 
          == TD->getDeclContext()->getLookupContext()) {
      // We're pushing the tag into the current context, which might
      // require some reshuffling in the identifier resolver.
      IdentifierResolver::iterator
        I = IdResolver.begin(TD->getDeclName(), CurContext, 
                             false/*LookInParentCtx*/),
        IEnd = IdResolver.end();
      if (I != IEnd && isDeclInScope(*I, CurContext, S)) {
        NamedDecl *PrevDecl = *I;
        for (; I != IEnd && isDeclInScope(*I, CurContext, S); 
             PrevDecl = *I, ++I) {
          if (TD->declarationReplaces(*I)) {
            // This is a redeclaration. Remove it from the chain and
            // break out, so that we'll add in the shadowed
            // declaration.
            S->RemoveDecl(*I);
            if (PrevDecl == *I) {
              IdResolver.RemoveDecl(*I);
              IdResolver.AddDecl(TD);
              return;
            } else {
              IdResolver.RemoveDecl(*I);
              break;
            }
          }
        }

        // There is already a declaration with the same name in the same
        // scope, which is not a tag declaration. It must be found
        // before we find the new declaration, so insert the new
        // declaration at the end of the chain.
        IdResolver.AddShadowedDecl(TD, PrevDecl);
        
        return;
      }
    }
  } else if (getLangOptions().CPlusPlus && isa<FunctionDecl>(D)) {
    // We are pushing the name of a function, which might be an
    // overloaded name.
    FunctionDecl *FD = cast<FunctionDecl>(D);
    DeclContext *DC = FD->getDeclContext()->getLookupContext();
    IdentifierResolver::iterator Redecl
      = std::find_if(IdResolver.begin(FD->getDeclName(), DC, 
                                      false/*LookInParentCtx*/),
                     IdResolver.end(),
                     std::bind1st(std::mem_fun(&NamedDecl::declarationReplaces),
                                  FD));
    if (Redecl != IdResolver.end()) {
      // There is already a declaration of a function on our
      // IdResolver chain. Replace it with this declaration.
      S->RemoveDecl(*Redecl);
      IdResolver.RemoveDecl(*Redecl);
    }
  }

  IdResolver.AddDecl(D);
}

void Sema::ActOnPopScope(SourceLocation Loc, Scope *S) {
  if (S->decl_empty()) return;
  assert((S->getFlags() & (Scope::DeclScope | Scope::TemplateParamScope)) &&
	 "Scope shouldn't contain decls!");

  for (Scope::decl_iterator I = S->decl_begin(), E = S->decl_end();
       I != E; ++I) {
    Decl *TmpD = static_cast<Decl*>(*I);
    assert(TmpD && "This decl didn't get pushed??");

    assert(isa<NamedDecl>(TmpD) && "Decl isn't NamedDecl?");
    NamedDecl *D = cast<NamedDecl>(TmpD);

    if (!D->getDeclName()) continue;

    // Remove this name from our lexical scope.
    IdResolver.RemoveDecl(D);
  }
}

/// getObjCInterfaceDecl - Look up a for a class declaration in the scope.
/// return 0 if one not found.
ObjCInterfaceDecl *Sema::getObjCInterfaceDecl(IdentifierInfo *Id) {
  // The third "scope" argument is 0 since we aren't enabling lazy built-in
  // creation from this context.
  Decl *IDecl = LookupDecl(Id, Decl::IDNS_Ordinary, 0, false);
  
  return dyn_cast_or_null<ObjCInterfaceDecl>(IDecl);
}

/// getNonFieldDeclScope - Retrieves the innermost scope, starting
/// from S, where a non-field would be declared. This routine copes
/// with the difference between C and C++ scoping rules in structs and
/// unions. For example, the following code is well-formed in C but
/// ill-formed in C++:
/// @code
/// struct S6 {
///   enum { BAR } e;
/// };
/// 
/// void test_S6() {
///   struct S6 a;
///   a.e = BAR;
/// }
/// @endcode
/// For the declaration of BAR, this routine will return a different
/// scope. The scope S will be the scope of the unnamed enumeration
/// within S6. In C++, this routine will return the scope associated
/// with S6, because the enumeration's scope is a transparent
/// context but structures can contain non-field names. In C, this
/// routine will return the translation unit scope, since the
/// enumeration's scope is a transparent context and structures cannot
/// contain non-field names.
Scope *Sema::getNonFieldDeclScope(Scope *S) {
  while (((S->getFlags() & Scope::DeclScope) == 0) ||
         (S->getEntity() && 
          ((DeclContext *)S->getEntity())->isTransparentContext()) ||
         (S->isClassScope() && !getLangOptions().CPlusPlus))
    S = S->getParent();
  return S;
}

/// LookupDecl - Look up the inner-most declaration in the specified
/// namespace. NamespaceNameOnly - during lookup only namespace names
/// are considered as required in C++ [basic.lookup.udir] 3.4.6.p1
/// 'When looking up a namespace-name in a using-directive or
/// namespace-alias-definition, only namespace names are considered.'
///
/// Note: The use of this routine is deprecated. Please use
/// LookupName, LookupQualifiedName, or LookupParsedName instead.
Sema::LookupResult
Sema::LookupDecl(DeclarationName Name, unsigned NSI, Scope *S,
                 const DeclContext *LookupCtx,
                 bool enableLazyBuiltinCreation,
                 bool LookInParent,
                 bool NamespaceNameOnly) {
  LookupCriteria::NameKind Kind;
  if (NSI == Decl::IDNS_Ordinary) {
    if (NamespaceNameOnly)
      Kind = LookupCriteria::Namespace;
    else
      Kind = LookupCriteria::Ordinary;
  } else if (NSI == Decl::IDNS_Tag) 
    Kind = LookupCriteria::Tag;
  else {
    assert(NSI == Decl::IDNS_Member &&"Unable to grok LookupDecl NSI argument");
    Kind = LookupCriteria::Member;
  }
  
  if (LookupCtx)
    return LookupQualifiedName(const_cast<DeclContext *>(LookupCtx), Name, 
                               LookupCriteria(Kind, !LookInParent,
                                              getLangOptions().CPlusPlus));

  // Unqualified lookup
  return LookupName(S, Name, 
                    LookupCriteria(Kind, !LookInParent,
                                   getLangOptions().CPlusPlus));
}

void Sema::InitBuiltinVaListType() {
  if (!Context.getBuiltinVaListType().isNull())
    return;
  
  IdentifierInfo *VaIdent = &Context.Idents.get("__builtin_va_list");
  Decl *VaDecl = LookupDecl(VaIdent, Decl::IDNS_Ordinary, TUScope);
  TypedefDecl *VaTypedef = cast<TypedefDecl>(VaDecl);
  Context.setBuiltinVaListType(Context.getTypedefType(VaTypedef));
}

/// LazilyCreateBuiltin - The specified Builtin-ID was first used at file scope.
/// lazily create a decl for it.
NamedDecl *Sema::LazilyCreateBuiltin(IdentifierInfo *II, unsigned bid,
                                     Scope *S) {
  Builtin::ID BID = (Builtin::ID)bid;

  if (Context.BuiltinInfo.hasVAListUse(BID))
    InitBuiltinVaListType();
    
  QualType R = Context.BuiltinInfo.GetBuiltinType(BID, Context);  
  FunctionDecl *New = FunctionDecl::Create(Context,
                                           Context.getTranslationUnitDecl(),
                                           SourceLocation(), II, R,
                                           FunctionDecl::Extern, false);
  
  // Create Decl objects for each parameter, adding them to the
  // FunctionDecl.
  if (FunctionTypeProto *FT = dyn_cast<FunctionTypeProto>(R)) {
    llvm::SmallVector<ParmVarDecl*, 16> Params;
    for (unsigned i = 0, e = FT->getNumArgs(); i != e; ++i)
      Params.push_back(ParmVarDecl::Create(Context, New, SourceLocation(), 0,
                                           FT->getArgType(i), VarDecl::None, 0));
    New->setParams(Context, &Params[0], Params.size());
  }
  
  
  
  // TUScope is the translation-unit scope to insert this function into.
  // FIXME: This is hideous. We need to teach PushOnScopeChains to
  // relate Scopes to DeclContexts, and probably eliminate CurContext
  // entirely, but we're not there yet.
  DeclContext *SavedContext = CurContext;
  CurContext = Context.getTranslationUnitDecl();
  PushOnScopeChains(New, TUScope);
  CurContext = SavedContext;
  return New;
}

/// GetStdNamespace - This method gets the C++ "std" namespace. This is where
/// everything from the standard library is defined.
NamespaceDecl *Sema::GetStdNamespace() {
  if (!StdNamespace) {
    IdentifierInfo *StdIdent = &PP.getIdentifierTable().get("std");
    DeclContext *Global = Context.getTranslationUnitDecl();
    Decl *Std = LookupDecl(StdIdent, Decl::IDNS_Ordinary,
                           0, Global, /*enableLazyBuiltinCreation=*/false);
    StdNamespace = dyn_cast_or_null<NamespaceDecl>(Std);
  }
  return StdNamespace;
}

/// MergeTypeDefDecl - We just parsed a typedef 'New' which has the same name
/// and scope as a previous declaration 'Old'.  Figure out how to resolve this
/// situation, merging decls or emitting diagnostics as appropriate.
///
TypedefDecl *Sema::MergeTypeDefDecl(TypedefDecl *New, Decl *OldD) {
  bool objc_types = false;
  // Allow multiple definitions for ObjC built-in typedefs.
  // FIXME: Verify the underlying types are equivalent!
  if (getLangOptions().ObjC1) {
    const IdentifierInfo *TypeID = New->getIdentifier();
    switch (TypeID->getLength()) {
    default: break;
    case 2: 
      if (!TypeID->isStr("id"))
        break;
      Context.setObjCIdType(New);
      objc_types = true;
      break;
    case 5:
      if (!TypeID->isStr("Class"))
        break;
      Context.setObjCClassType(New);
      objc_types = true;
      return New;
    case 3:
      if (!TypeID->isStr("SEL"))
        break;
      Context.setObjCSelType(New);
      objc_types = true;
      return New;
    case 8:
      if (!TypeID->isStr("Protocol"))
        break;
      Context.setObjCProtoType(New->getUnderlyingType());
      objc_types = true;
      return New;
    }
    // Fall through - the typedef name was not a builtin type.
  }
  // Verify the old decl was also a typedef.
  TypedefDecl *Old = dyn_cast<TypedefDecl>(OldD);
  if (!Old) {
    Diag(New->getLocation(), diag::err_redefinition_different_kind)
      << New->getDeclName();
    if (!objc_types)
      Diag(OldD->getLocation(), diag::note_previous_definition);
    return New;
  }
  
  // If the typedef types are not identical, reject them in all languages and
  // with any extensions enabled.
  if (Old->getUnderlyingType() != New->getUnderlyingType() && 
      Context.getCanonicalType(Old->getUnderlyingType()) != 
      Context.getCanonicalType(New->getUnderlyingType())) {
    Diag(New->getLocation(), diag::err_redefinition_different_typedef)
      << New->getUnderlyingType() << Old->getUnderlyingType();
    if (!objc_types)
      Diag(Old->getLocation(), diag::note_previous_definition);
    return New;
  }
  if (objc_types) return New;
  if (getLangOptions().Microsoft) return New;

  // C++ [dcl.typedef]p2:
  //   In a given non-class scope, a typedef specifier can be used to
  //   redefine the name of any type declared in that scope to refer
  //   to the type to which it already refers.
  if (getLangOptions().CPlusPlus && !isa<CXXRecordDecl>(CurContext))
    return New;

  // In C, redeclaration of a type is a constraint violation (6.7.2.3p1).
  // Apparently GCC, Intel, and Sun all silently ignore the redeclaration if
  // *either* declaration is in a system header. The code below implements
  // this adhoc compatibility rule. FIXME: The following code will not
  // work properly when compiling ".i" files (containing preprocessed output).
  if (PP.getDiagnostics().getSuppressSystemWarnings()) {
    SourceManager &SrcMgr = Context.getSourceManager();
    if (SrcMgr.isInSystemHeader(Old->getLocation()))
      return New;
    if (SrcMgr.isInSystemHeader(New->getLocation()))
      return New;
  }

  Diag(New->getLocation(), diag::err_redefinition) << New->getDeclName();
  Diag(Old->getLocation(), diag::note_previous_definition);
  return New;
}

/// DeclhasAttr - returns true if decl Declaration already has the target
/// attribute.
static bool DeclHasAttr(const Decl *decl, const Attr *target) {
  for (const Attr *attr = decl->getAttrs(); attr; attr = attr->getNext())
    if (attr->getKind() == target->getKind())
      return true;

  return false;
}

/// MergeAttributes - append attributes from the Old decl to the New one.
static void MergeAttributes(Decl *New, Decl *Old) {
  Attr *attr = const_cast<Attr*>(Old->getAttrs()), *tmp;

  while (attr) {
     tmp = attr;
     attr = attr->getNext();

    if (!DeclHasAttr(New, tmp)) {
       tmp->setInherited(true);
       New->addAttr(tmp);
    } else {
       tmp->setNext(0);
       delete(tmp);
    }
  }

  Old->invalidateAttrs();
}

/// MergeFunctionDecl - We just parsed a function 'New' from
/// declarator D which has the same name and scope as a previous
/// declaration 'Old'.  Figure out how to resolve this situation,
/// merging decls or emitting diagnostics as appropriate.
/// Redeclaration will be set true if this New is a redeclaration OldD.
///
/// In C++, New and Old must be declarations that are not
/// overloaded. Use IsOverload to determine whether New and Old are
/// overloaded, and to select the Old declaration that New should be
/// merged with.
FunctionDecl *
Sema::MergeFunctionDecl(FunctionDecl *New, Decl *OldD, bool &Redeclaration) {
  assert(!isa<OverloadedFunctionDecl>(OldD) && 
         "Cannot merge with an overloaded function declaration");

  Redeclaration = false;
  // Verify the old decl was also a function.
  FunctionDecl *Old = dyn_cast<FunctionDecl>(OldD);
  if (!Old) {
    Diag(New->getLocation(), diag::err_redefinition_different_kind)
      << New->getDeclName();
    Diag(OldD->getLocation(), diag::note_previous_definition);
    return New;
  }

  // Determine whether the previous declaration was a definition,
  // implicit declaration, or a declaration.
  diag::kind PrevDiag;
  if (Old->isThisDeclarationADefinition())
    PrevDiag = diag::note_previous_definition;
  else if (Old->isImplicit())
    PrevDiag = diag::note_previous_implicit_declaration;
  else 
    PrevDiag = diag::note_previous_declaration;
  
  QualType OldQType = Context.getCanonicalType(Old->getType());
  QualType NewQType = Context.getCanonicalType(New->getType());
  
  if (getLangOptions().CPlusPlus) {
    // (C++98 13.1p2):
    //   Certain function declarations cannot be overloaded:
    //     -- Function declarations that differ only in the return type 
    //        cannot be overloaded.
    QualType OldReturnType 
      = cast<FunctionType>(OldQType.getTypePtr())->getResultType();
    QualType NewReturnType 
      = cast<FunctionType>(NewQType.getTypePtr())->getResultType();
    if (OldReturnType != NewReturnType) {
      Diag(New->getLocation(), diag::err_ovl_diff_return_type);
      Diag(Old->getLocation(), PrevDiag);
      Redeclaration = true;
      return New;
    }

    const CXXMethodDecl* OldMethod = dyn_cast<CXXMethodDecl>(Old);
    const CXXMethodDecl* NewMethod = dyn_cast<CXXMethodDecl>(New);
    if (OldMethod && NewMethod) {
      //    -- Member function declarations with the same name and the 
      //       same parameter types cannot be overloaded if any of them 
      //       is a static member function declaration.
      if (OldMethod->isStatic() || NewMethod->isStatic()) {
        Diag(New->getLocation(), diag::err_ovl_static_nonstatic_member);
        Diag(Old->getLocation(), PrevDiag);
        return New;
      }

      // C++ [class.mem]p1:
      //   [...] A member shall not be declared twice in the
      //   member-specification, except that a nested class or member
      //   class template can be declared and then later defined.
      if (OldMethod->getLexicalDeclContext() == 
            NewMethod->getLexicalDeclContext()) {
        unsigned NewDiag;
        if (isa<CXXConstructorDecl>(OldMethod))
          NewDiag = diag::err_constructor_redeclared;
        else if (isa<CXXDestructorDecl>(NewMethod))
          NewDiag = diag::err_destructor_redeclared;
        else if (isa<CXXConversionDecl>(NewMethod))
          NewDiag = diag::err_conv_function_redeclared;
        else
          NewDiag = diag::err_member_redeclared;

        Diag(New->getLocation(), NewDiag);
        Diag(Old->getLocation(), PrevDiag);
      }
    }

    // (C++98 8.3.5p3):
    //   All declarations for a function shall agree exactly in both the
    //   return type and the parameter-type-list.
    if (OldQType == NewQType) {
      // We have a redeclaration.
      MergeAttributes(New, Old);
      Redeclaration = true;
      return MergeCXXFunctionDecl(New, Old);
    } 

    // Fall through for conflicting redeclarations and redefinitions.
  }

  // C: Function types need to be compatible, not identical. This handles
  // duplicate function decls like "void f(int); void f(enum X);" properly.
  if (!getLangOptions().CPlusPlus &&
      Context.typesAreCompatible(OldQType, NewQType)) {
    MergeAttributes(New, Old);
    Redeclaration = true;
    return New;
  }

  // A function that has already been declared has been redeclared or defined
  // with a different type- show appropriate diagnostic

  // TODO: CHECK FOR CONFLICTS, multiple decls with same name in one scope.
  // TODO: This is totally simplistic.  It should handle merging functions
  // together etc, merging extern int X; int X; ...
  Diag(New->getLocation(), diag::err_conflicting_types) << New->getDeclName();
  Diag(Old->getLocation(), PrevDiag);
  return New;
}

/// Predicate for C "tentative" external object definitions (C99 6.9.2).
static bool isTentativeDefinition(VarDecl *VD) {
  if (VD->isFileVarDecl())
    return (!VD->getInit() &&
            (VD->getStorageClass() == VarDecl::None ||
             VD->getStorageClass() == VarDecl::Static));
  return false;
}

/// CheckForFileScopedRedefinitions - Make sure we forgo redefinition errors
/// when dealing with C "tentative" external object definitions (C99 6.9.2).
void Sema::CheckForFileScopedRedefinitions(Scope *S, VarDecl *VD) {
  bool VDIsTentative = isTentativeDefinition(VD);
  bool VDIsIncompleteArray = VD->getType()->isIncompleteArrayType();
  
  // FIXME: I don't think this will actually see all of the
  // redefinitions. Can't we check this property on-the-fly?
  for (IdentifierResolver::iterator
       I = IdResolver.begin(VD->getIdentifier(), 
                            VD->getDeclContext(), false/*LookInParentCtx*/), 
       E = IdResolver.end(); I != E; ++I) {
    if (*I != VD && isDeclInScope(*I, VD->getDeclContext(), S)) {
      VarDecl *OldDecl = dyn_cast<VarDecl>(*I);
      
      // Handle the following case:
      //   int a[10];
      //   int a[];   - the code below makes sure we set the correct type. 
      //   int a[11]; - this is an error, size isn't 10.
      if (OldDecl && VDIsTentative && VDIsIncompleteArray && 
          OldDecl->getType()->isConstantArrayType())
        VD->setType(OldDecl->getType());
      
      // Check for "tentative" definitions. We can't accomplish this in 
      // MergeVarDecl since the initializer hasn't been attached.
      if (!OldDecl || isTentativeDefinition(OldDecl) || VDIsTentative)
        continue;
  
      // Handle __private_extern__ just like extern.
      if (OldDecl->getStorageClass() != VarDecl::Extern &&
          OldDecl->getStorageClass() != VarDecl::PrivateExtern &&
          VD->getStorageClass() != VarDecl::Extern &&
          VD->getStorageClass() != VarDecl::PrivateExtern) {
        Diag(VD->getLocation(), diag::err_redefinition) << VD->getDeclName();
        Diag(OldDecl->getLocation(), diag::note_previous_definition);
      }
    }
  }
}

/// MergeVarDecl - We just parsed a variable 'New' which has the same name
/// and scope as a previous declaration 'Old'.  Figure out how to resolve this
/// situation, merging decls or emitting diagnostics as appropriate.
///
/// Tentative definition rules (C99 6.9.2p2) are checked by 
/// FinalizeDeclaratorGroup. Unfortunately, we can't analyze tentative 
/// definitions here, since the initializer hasn't been attached.
/// 
VarDecl *Sema::MergeVarDecl(VarDecl *New, Decl *OldD) {
  // Verify the old decl was also a variable.
  VarDecl *Old = dyn_cast<VarDecl>(OldD);
  if (!Old) {
    Diag(New->getLocation(), diag::err_redefinition_different_kind)
      << New->getDeclName();
    Diag(OldD->getLocation(), diag::note_previous_definition);
    return New;
  }

  MergeAttributes(New, Old);

  // Verify the types match.
  QualType OldCType = Context.getCanonicalType(Old->getType());
  QualType NewCType = Context.getCanonicalType(New->getType());
  if (OldCType != NewCType && !Context.typesAreCompatible(OldCType, NewCType)) {
    Diag(New->getLocation(), diag::err_redefinition_different_type) 
      << New->getDeclName();
    Diag(Old->getLocation(), diag::note_previous_definition);
    return New;
  }
  // C99 6.2.2p4: Check if we have a static decl followed by a non-static.
  if (New->getStorageClass() == VarDecl::Static &&
      (Old->getStorageClass() == VarDecl::None ||
       Old->getStorageClass() == VarDecl::Extern)) {
    Diag(New->getLocation(), diag::err_static_non_static) << New->getDeclName();
    Diag(Old->getLocation(), diag::note_previous_definition);
    return New;
  }
  // C99 6.2.2p4: Check if we have a non-static decl followed by a static.
  if (New->getStorageClass() != VarDecl::Static &&
      Old->getStorageClass() == VarDecl::Static) {
    Diag(New->getLocation(), diag::err_non_static_static) << New->getDeclName();
    Diag(Old->getLocation(), diag::note_previous_definition);
    return New;
  }
  // Variables with external linkage are analyzed in FinalizeDeclaratorGroup.
  if (New->getStorageClass() != VarDecl::Extern && !New->isFileVarDecl()) {
    Diag(New->getLocation(), diag::err_redefinition) << New->getDeclName();
    Diag(Old->getLocation(), diag::note_previous_definition);
  }
  return New;
}

/// CheckParmsForFunctionDef - Check that the parameters of the given
/// function are appropriate for the definition of a function. This
/// takes care of any checks that cannot be performed on the
/// declaration itself, e.g., that the types of each of the function
/// parameters are complete.
bool Sema::CheckParmsForFunctionDef(FunctionDecl *FD) {
  bool HasInvalidParm = false;
  for (unsigned p = 0, NumParams = FD->getNumParams(); p < NumParams; ++p) {
    ParmVarDecl *Param = FD->getParamDecl(p);

    // C99 6.7.5.3p4: the parameters in a parameter type list in a
    // function declarator that is part of a function definition of
    // that function shall not have incomplete type.
    if (!Param->isInvalidDecl() &&
        DiagnoseIncompleteType(Param->getLocation(), Param->getType(),
                               diag::err_typecheck_decl_incomplete_type)) {
      Param->setInvalidDecl();
      HasInvalidParm = true;
    }
    
    // C99 6.9.1p5: If the declarator includes a parameter type list, the
    // declaration of each parameter shall include an identifier.
    if (Param->getIdentifier() == 0 && !getLangOptions().CPlusPlus)
      Diag(Param->getLocation(), diag::err_parameter_name_omitted);
  }

  return HasInvalidParm;
}

/// ParsedFreeStandingDeclSpec - This method is invoked when a declspec with
/// no declarator (e.g. "struct foo;") is parsed.
Sema::DeclTy *Sema::ParsedFreeStandingDeclSpec(Scope *S, DeclSpec &DS) {
  TagDecl *Tag 
    = dyn_cast_or_null<TagDecl>(static_cast<Decl *>(DS.getTypeRep()));
  if (RecordDecl *Record = dyn_cast_or_null<RecordDecl>(Tag)) {
    if (!Record->getDeclName() && Record->isDefinition() &&
        DS.getStorageClassSpec() != DeclSpec::SCS_typedef)
      return BuildAnonymousStructOrUnion(S, DS, Record);

    // Microsoft allows unnamed struct/union fields. Don't complain
    // about them.
    // FIXME: Should we support Microsoft's extensions in this area?
    if (Record->getDeclName() && getLangOptions().Microsoft)
      return Tag;
  }

  if (!DS.isMissingDeclaratorOk()) {
    // Warn about typedefs of enums without names, since this is an
    // extension in both Microsoft an GNU.
    if (DS.getStorageClassSpec() == DeclSpec::SCS_typedef &&
        Tag && isa<EnumDecl>(Tag)) {
      Diag(DS.getSourceRange().getBegin(), diag::ext_typedef_without_a_name)
        << DS.getSourceRange();
      return Tag;
    }

    // FIXME: This diagnostic is emitted even when various previous
    // errors occurred (see e.g. test/Sema/decl-invalid.c). However,
    // DeclSpec has no means of communicating this information, and the
    // responsible parser functions are quite far apart.
    Diag(DS.getSourceRange().getBegin(), diag::err_no_declarators)
      << DS.getSourceRange();
    return 0;
  }
  
  return Tag;
}

/// InjectAnonymousStructOrUnionMembers - Inject the members of the
/// anonymous struct or union AnonRecord into the owning context Owner
/// and scope S. This routine will be invoked just after we realize
/// that an unnamed union or struct is actually an anonymous union or
/// struct, e.g.,
///
/// @code
/// union {
///   int i;
///   float f;
/// }; // InjectAnonymousStructOrUnionMembers called here to inject i and
///    // f into the surrounding scope.x
/// @endcode
///
/// This routine is recursive, injecting the names of nested anonymous
/// structs/unions into the owning context and scope as well.
bool Sema::InjectAnonymousStructOrUnionMembers(Scope *S, DeclContext *Owner,
                                               RecordDecl *AnonRecord) {
  bool Invalid = false;
  for (RecordDecl::field_iterator F = AnonRecord->field_begin(),
                               FEnd = AnonRecord->field_end();
       F != FEnd; ++F) {
    if ((*F)->getDeclName()) {
      Decl *PrevDecl = LookupDecl((*F)->getDeclName(), Decl::IDNS_Ordinary,
                                  S, Owner, false, false, false);
      if (PrevDecl && !isa<TagDecl>(PrevDecl)) {
        // C++ [class.union]p2:
        //   The names of the members of an anonymous union shall be
        //   distinct from the names of any other entity in the
        //   scope in which the anonymous union is declared.
        unsigned diagKind 
          = AnonRecord->isUnion()? diag::err_anonymous_union_member_redecl
                                 : diag::err_anonymous_struct_member_redecl;
        Diag((*F)->getLocation(), diagKind)
          << (*F)->getDeclName();
        Diag(PrevDecl->getLocation(), diag::note_previous_declaration);
        Invalid = true;
      } else {
        // C++ [class.union]p2:
        //   For the purpose of name lookup, after the anonymous union
        //   definition, the members of the anonymous union are
        //   considered to have been defined in the scope in which the
        //   anonymous union is declared.
        Owner->makeDeclVisibleInContext(*F);
        S->AddDecl(*F);
        IdResolver.AddDecl(*F);
      }
    } else if (const RecordType *InnerRecordType
                 = (*F)->getType()->getAsRecordType()) {
      RecordDecl *InnerRecord = InnerRecordType->getDecl();
      if (InnerRecord->isAnonymousStructOrUnion())
        Invalid = Invalid || 
          InjectAnonymousStructOrUnionMembers(S, Owner, InnerRecord);
    }
  }

  return Invalid;
}

/// ActOnAnonymousStructOrUnion - Handle the declaration of an
/// anonymous structure or union. Anonymous unions are a C++ feature
/// (C++ [class.union]) and a GNU C extension; anonymous structures
/// are a GNU C and GNU C++ extension. 
Sema::DeclTy *Sema::BuildAnonymousStructOrUnion(Scope *S, DeclSpec &DS,
                                                RecordDecl *Record) {
  DeclContext *Owner = Record->getDeclContext();

  // Diagnose whether this anonymous struct/union is an extension.
  if (Record->isUnion() && !getLangOptions().CPlusPlus)
    Diag(Record->getLocation(), diag::ext_anonymous_union);
  else if (!Record->isUnion())
    Diag(Record->getLocation(), diag::ext_anonymous_struct);
  
  // C and C++ require different kinds of checks for anonymous
  // structs/unions.
  bool Invalid = false;
  if (getLangOptions().CPlusPlus) {
    const char* PrevSpec = 0;
    // C++ [class.union]p3:
    //   Anonymous unions declared in a named namespace or in the
    //   global namespace shall be declared static.
    if (DS.getStorageClassSpec() != DeclSpec::SCS_static &&
        (isa<TranslationUnitDecl>(Owner) ||
         (isa<NamespaceDecl>(Owner) && 
          cast<NamespaceDecl>(Owner)->getDeclName()))) {
      Diag(Record->getLocation(), diag::err_anonymous_union_not_static);
      Invalid = true;

      // Recover by adding 'static'.
      DS.SetStorageClassSpec(DeclSpec::SCS_static, SourceLocation(), PrevSpec);
    } 
    // C++ [class.union]p3:
    //   A storage class is not allowed in a declaration of an
    //   anonymous union in a class scope.
    else if (DS.getStorageClassSpec() != DeclSpec::SCS_unspecified &&
             isa<RecordDecl>(Owner)) {
      Diag(DS.getStorageClassSpecLoc(), 
           diag::err_anonymous_union_with_storage_spec);
      Invalid = true;

      // Recover by removing the storage specifier.
      DS.SetStorageClassSpec(DeclSpec::SCS_unspecified, SourceLocation(),
                             PrevSpec);
    }

    // C++ [class.union]p2: 
    //   The member-specification of an anonymous union shall only
    //   define non-static data members. [Note: nested types and
    //   functions cannot be declared within an anonymous union. ]
    for (DeclContext::decl_iterator Mem = Record->decls_begin(),
                                 MemEnd = Record->decls_end();
         Mem != MemEnd; ++Mem) {
      if (FieldDecl *FD = dyn_cast<FieldDecl>(*Mem)) {
        // C++ [class.union]p3:
        //   An anonymous union shall not have private or protected
        //   members (clause 11).
        if (FD->getAccess() == AS_protected || FD->getAccess() == AS_private) {
          Diag(FD->getLocation(), diag::err_anonymous_record_nonpublic_member)
            << (int)Record->isUnion() << (int)(FD->getAccess() == AS_protected);
          Invalid = true;
        }
      } else if ((*Mem)->isImplicit()) {
        // Any implicit members are fine.
      } else if (RecordDecl *MemRecord = dyn_cast<RecordDecl>(*Mem)) {
        if (!MemRecord->isAnonymousStructOrUnion() &&
            MemRecord->getDeclName()) {
          // This is a nested type declaration.
          Diag(MemRecord->getLocation(), diag::err_anonymous_record_with_type)
            << (int)Record->isUnion();
          Invalid = true;
        }
      } else {
        // We have something that isn't a non-static data
        // member. Complain about it.
        unsigned DK = diag::err_anonymous_record_bad_member;
        if (isa<TypeDecl>(*Mem))
          DK = diag::err_anonymous_record_with_type;
        else if (isa<FunctionDecl>(*Mem))
          DK = diag::err_anonymous_record_with_function;
        else if (isa<VarDecl>(*Mem))
          DK = diag::err_anonymous_record_with_static;
        Diag((*Mem)->getLocation(), DK)
            << (int)Record->isUnion();
          Invalid = true;
      }
    }
  } else {
    // FIXME: Check GNU C semantics
    if (Record->isUnion() && !Owner->isRecord()) {
      Diag(Record->getLocation(), diag::err_anonymous_union_not_member)
        << (int)getLangOptions().CPlusPlus;
      Invalid = true;
    }
  }

  if (!Record->isUnion() && !Owner->isRecord()) {
    Diag(Record->getLocation(), diag::err_anonymous_struct_not_member)
      << (int)getLangOptions().CPlusPlus;
    Invalid = true;
  }

  // Create a declaration for this anonymous struct/union. 
  NamedDecl *Anon = 0;
  if (RecordDecl *OwningClass = dyn_cast<RecordDecl>(Owner)) {
    Anon = FieldDecl::Create(Context, OwningClass, Record->getLocation(),
                             /*IdentifierInfo=*/0, 
                             Context.getTypeDeclType(Record),
                             /*BitWidth=*/0, /*Mutable=*/false);
    Anon->setAccess(AS_public);
    if (getLangOptions().CPlusPlus)
      FieldCollector->Add(cast<FieldDecl>(Anon));
  } else {
    VarDecl::StorageClass SC;
    switch (DS.getStorageClassSpec()) {
    default: assert(0 && "Unknown storage class!");
    case DeclSpec::SCS_unspecified:    SC = VarDecl::None; break;
    case DeclSpec::SCS_extern:         SC = VarDecl::Extern; break;
    case DeclSpec::SCS_static:         SC = VarDecl::Static; break;
    case DeclSpec::SCS_auto:           SC = VarDecl::Auto; break;
    case DeclSpec::SCS_register:       SC = VarDecl::Register; break;
    case DeclSpec::SCS_private_extern: SC = VarDecl::PrivateExtern; break;
    case DeclSpec::SCS_mutable:
      // mutable can only appear on non-static class members, so it's always
      // an error here
      Diag(Record->getLocation(), diag::err_mutable_nonmember);
      Invalid = true;
      SC = VarDecl::None;
      break;
    }

    Anon = VarDecl::Create(Context, Owner, Record->getLocation(),
                           /*IdentifierInfo=*/0, 
                           Context.getTypeDeclType(Record),
                           SC, DS.getSourceRange().getBegin());
  }
  Anon->setImplicit();

  // Add the anonymous struct/union object to the current
  // context. We'll be referencing this object when we refer to one of
  // its members.
  Owner->addDecl(Anon);

  // Inject the members of the anonymous struct/union into the owning
  // context and into the identifier resolver chain for name lookup
  // purposes.
  if (InjectAnonymousStructOrUnionMembers(S, Owner, Record))
    Invalid = true;

  // Mark this as an anonymous struct/union type. Note that we do not
  // do this until after we have already checked and injected the
  // members of this anonymous struct/union type, because otherwise
  // the members could be injected twice: once by DeclContext when it
  // builds its lookup table, and once by
  // InjectAnonymousStructOrUnionMembers. 
  Record->setAnonymousStructOrUnion(true);

  if (Invalid)
    Anon->setInvalidDecl();

  return Anon;
}

bool Sema::CheckSingleInitializer(Expr *&Init, QualType DeclType, 
                                  bool DirectInit) {  
  // Get the type before calling CheckSingleAssignmentConstraints(), since
  // it can promote the expression.
  QualType InitType = Init->getType(); 

  if (getLangOptions().CPlusPlus) {
    // FIXME: I dislike this error message. A lot.
    if (PerformImplicitConversion(Init, DeclType, "initializing", DirectInit))
      return Diag(Init->getSourceRange().getBegin(),
                  diag::err_typecheck_convert_incompatible)
        << DeclType << Init->getType() << "initializing" 
        << Init->getSourceRange();

    return false;
  }

  AssignConvertType ConvTy = CheckSingleAssignmentConstraints(DeclType, Init);
  return DiagnoseAssignmentResult(ConvTy, Init->getLocStart(), DeclType,
                                  InitType, Init, "initializing");
}

bool Sema::CheckStringLiteralInit(StringLiteral *strLiteral, QualType &DeclT) {
  const ArrayType *AT = Context.getAsArrayType(DeclT);
  
  if (const IncompleteArrayType *IAT = dyn_cast<IncompleteArrayType>(AT)) {
    // C99 6.7.8p14. We have an array of character type with unknown size 
    // being initialized to a string literal.
    llvm::APSInt ConstVal(32);
    ConstVal = strLiteral->getByteLength() + 1;
    // Return a new array type (C99 6.7.8p22).
    DeclT = Context.getConstantArrayType(IAT->getElementType(), ConstVal, 
                                         ArrayType::Normal, 0);
  } else {
    const ConstantArrayType *CAT = cast<ConstantArrayType>(AT);
    // C99 6.7.8p14. We have an array of character type with known size.
    // FIXME: Avoid truncation for 64-bit length strings.
    if (strLiteral->getByteLength() > (unsigned)CAT->getSize().getZExtValue())
      Diag(strLiteral->getSourceRange().getBegin(),
           diag::warn_initializer_string_for_char_array_too_long)
        << strLiteral->getSourceRange();
  }
  // Set type from "char *" to "constant array of char".
  strLiteral->setType(DeclT);
  // For now, we always return false (meaning success).
  return false;
}

StringLiteral *Sema::IsStringLiteralInit(Expr *Init, QualType DeclType) {
  const ArrayType *AT = Context.getAsArrayType(DeclType);
  if (AT && AT->getElementType()->isCharType()) {
    return dyn_cast<StringLiteral>(Init->IgnoreParens());
  }
  return 0;
}

bool Sema::CheckInitializerTypes(Expr *&Init, QualType &DeclType,
                                 SourceLocation InitLoc,
                                 DeclarationName InitEntity,
                                 bool DirectInit) {
  if (DeclType->isDependentType() || Init->isTypeDependent())
    return false;

  // C++ [dcl.init.ref]p1:
  //   A variable declared to be a T&, that is "reference to type T"
  //   (8.3.2), shall be initialized by an object, or function, of
  //   type T or by an object that can be converted into a T.
  if (DeclType->isReferenceType())
    return CheckReferenceInit(Init, DeclType, 0, false, DirectInit);

  // C99 6.7.8p3: The type of the entity to be initialized shall be an array
  // of unknown size ("[]") or an object type that is not a variable array type.
  if (const VariableArrayType *VAT = Context.getAsVariableArrayType(DeclType))
    return Diag(InitLoc,  diag::err_variable_object_no_init)
      << VAT->getSizeExpr()->getSourceRange();
  
  InitListExpr *InitList = dyn_cast<InitListExpr>(Init);
  if (!InitList) {
    // FIXME: Handle wide strings
    if (StringLiteral *strLiteral = IsStringLiteralInit(Init, DeclType))
      return CheckStringLiteralInit(strLiteral, DeclType);

    // C++ [dcl.init]p14:
    //   -- If the destination type is a (possibly cv-qualified) class
    //      type:
    if (getLangOptions().CPlusPlus && DeclType->isRecordType()) {
      QualType DeclTypeC = Context.getCanonicalType(DeclType);
      QualType InitTypeC = Context.getCanonicalType(Init->getType());

      //   -- If the initialization is direct-initialization, or if it is
      //      copy-initialization where the cv-unqualified version of the
      //      source type is the same class as, or a derived class of, the
      //      class of the destination, constructors are considered.
      if ((DeclTypeC.getUnqualifiedType() == InitTypeC.getUnqualifiedType()) ||
          IsDerivedFrom(InitTypeC, DeclTypeC)) {
        CXXConstructorDecl *Constructor 
          = PerformInitializationByConstructor(DeclType, &Init, 1,
                                               InitLoc, Init->getSourceRange(),
                                               InitEntity, 
                                               DirectInit? IK_Direct : IK_Copy);
        return Constructor == 0;
      }

      //   -- Otherwise (i.e., for the remaining copy-initialization
      //      cases), user-defined conversion sequences that can
      //      convert from the source type to the destination type or
      //      (when a conversion function is used) to a derived class
      //      thereof are enumerated as described in 13.3.1.4, and the
      //      best one is chosen through overload resolution
      //      (13.3). If the conversion cannot be done or is
      //      ambiguous, the initialization is ill-formed. The
      //      function selected is called with the initializer
      //      expression as its argument; if the function is a
      //      constructor, the call initializes a temporary of the
      //      destination type.
      // FIXME: We're pretending to do copy elision here; return to
      // this when we have ASTs for such things.
      if (!PerformImplicitConversion(Init, DeclType, "initializing"))
        return false;
      
      if (InitEntity)
        return Diag(InitLoc, diag::err_cannot_initialize_decl)
          << InitEntity << (int)(Init->isLvalue(Context) == Expr::LV_Valid)
          << Init->getType() << Init->getSourceRange();
      else
        return Diag(InitLoc, diag::err_cannot_initialize_decl_noname)
          << DeclType << (int)(Init->isLvalue(Context) == Expr::LV_Valid)
          << Init->getType() << Init->getSourceRange();
    }

    // C99 6.7.8p16.
    if (DeclType->isArrayType())
      return Diag(Init->getLocStart(), diag::err_array_init_list_required)
        << Init->getSourceRange();

    return CheckSingleInitializer(Init, DeclType, DirectInit);
  } else if (getLangOptions().CPlusPlus) {
    // C++ [dcl.init]p14:
    //   [...] If the class is an aggregate (8.5.1), and the initializer
    //   is a brace-enclosed list, see 8.5.1.
    //
    // Note: 8.5.1 is handled below; here, we diagnose the case where
    // we have an initializer list and a destination type that is not
    // an aggregate.
    // FIXME: In C++0x, this is yet another form of initialization.
    if (const RecordType *ClassRec = DeclType->getAsRecordType()) {
      const CXXRecordDecl *ClassDecl = cast<CXXRecordDecl>(ClassRec->getDecl());
      if (!ClassDecl->isAggregate())
        return Diag(InitLoc, diag::err_init_non_aggr_init_list)
           << DeclType << Init->getSourceRange();
    }
  }

  InitListChecker CheckInitList(this, InitList, DeclType);
  return CheckInitList.HadError();
}

/// GetNameForDeclarator - Determine the full declaration name for the
/// given Declarator.
DeclarationName Sema::GetNameForDeclarator(Declarator &D) {
  switch (D.getKind()) {
  case Declarator::DK_Abstract:
    assert(D.getIdentifier() == 0 && "abstract declarators have no name");
    return DeclarationName();

  case Declarator::DK_Normal:
    assert (D.getIdentifier() != 0 && "normal declarators have an identifier");
    return DeclarationName(D.getIdentifier());

  case Declarator::DK_Constructor: {
    QualType Ty = Context.getTypeDeclType((TypeDecl *)D.getDeclaratorIdType());
    Ty = Context.getCanonicalType(Ty);
    return Context.DeclarationNames.getCXXConstructorName(Ty);
  }

  case Declarator::DK_Destructor: {
    QualType Ty = Context.getTypeDeclType((TypeDecl *)D.getDeclaratorIdType());
    Ty = Context.getCanonicalType(Ty);
    return Context.DeclarationNames.getCXXDestructorName(Ty);
  }

  case Declarator::DK_Conversion: {
    QualType Ty = QualType::getFromOpaquePtr(D.getDeclaratorIdType());
    Ty = Context.getCanonicalType(Ty);
    return Context.DeclarationNames.getCXXConversionFunctionName(Ty);
  }

  case Declarator::DK_Operator:
    assert(D.getIdentifier() == 0 && "operator names have no identifier");
    return Context.DeclarationNames.getCXXOperatorName(
                                                D.getOverloadedOperator());
  }

  assert(false && "Unknown name kind");
  return DeclarationName();
}

/// isNearlyMatchingMemberFunction - Determine whether the C++ member
/// functions Declaration and Definition are "nearly" matching. This
/// heuristic is used to improve diagnostics in the case where an
/// out-of-line member function definition doesn't match any
/// declaration within the class.
static bool isNearlyMatchingMemberFunction(ASTContext &Context,
                                           FunctionDecl *Declaration,
                                           FunctionDecl *Definition) {
  if (Declaration->param_size() != Definition->param_size())
    return false;
  for (unsigned Idx = 0; Idx < Declaration->param_size(); ++Idx) {
    QualType DeclParamTy = Declaration->getParamDecl(Idx)->getType();
    QualType DefParamTy = Definition->getParamDecl(Idx)->getType();

    DeclParamTy = Context.getCanonicalType(DeclParamTy.getNonReferenceType());
    DefParamTy = Context.getCanonicalType(DefParamTy.getNonReferenceType());
    if (DeclParamTy.getUnqualifiedType() != DefParamTy.getUnqualifiedType())
      return false;
  }

  return true;
}

Sema::DeclTy *
Sema::ActOnDeclarator(Scope *S, Declarator &D, DeclTy *lastDecl,
                      bool IsFunctionDefinition) {
  NamedDecl *LastDeclarator = dyn_cast_or_null<NamedDecl>((Decl *)lastDecl);
  DeclarationName Name = GetNameForDeclarator(D);

  // All of these full declarators require an identifier.  If it doesn't have
  // one, the ParsedFreeStandingDeclSpec action should be used.
  if (!Name) {
    if (!D.getInvalidType())  // Reject this if we think it is valid.
      Diag(D.getDeclSpec().getSourceRange().getBegin(),
           diag::err_declarator_need_ident)
        << D.getDeclSpec().getSourceRange() << D.getSourceRange();
    return 0;
  }
  
  // The scope passed in may not be a decl scope.  Zip up the scope tree until
  // we find one that is.
  while ((S->getFlags() & Scope::DeclScope) == 0 ||
        (S->getFlags() & Scope::TemplateParamScope) != 0)
    S = S->getParent();
  
  DeclContext *DC;
  Decl *PrevDecl;
  NamedDecl *New;
  bool InvalidDecl = false;
 
  // See if this is a redefinition of a variable in the same scope.
  if (!D.getCXXScopeSpec().isSet()) {
    DC = CurContext;
    PrevDecl = LookupDecl(Name, Decl::IDNS_Ordinary, S);
  } else { // Something like "int foo::x;"
    DC = static_cast<DeclContext*>(D.getCXXScopeSpec().getScopeRep());
    PrevDecl = LookupDecl(Name, Decl::IDNS_Ordinary, S, DC);

    // C++ 7.3.1.2p2:
    // Members (including explicit specializations of templates) of a named
    // namespace can also be defined outside that namespace by explicit
    // qualification of the name being defined, provided that the entity being
    // defined was already declared in the namespace and the definition appears
    // after the point of declaration in a namespace that encloses the
    // declarations namespace.
    //
    // Note that we only check the context at this point. We don't yet
    // have enough information to make sure that PrevDecl is actually
    // the declaration we want to match. For example, given:
    //
    //   class X {
    //     void f();
    //     void f(float);
    //   };
    //
    //   void X::f(int) { } // ill-formed
    //
    // In this case, PrevDecl will point to the overload set
    // containing the two f's declared in X, but neither of them
    // matches. 
    if (!CurContext->Encloses(DC)) {
      // The qualifying scope doesn't enclose the original declaration.
      // Emit diagnostic based on current scope.
      SourceLocation L = D.getIdentifierLoc();
      SourceRange R = D.getCXXScopeSpec().getRange();
      if (isa<FunctionDecl>(CurContext)) {
        Diag(L, diag::err_invalid_declarator_in_function) << Name << R;
      } else {
        Diag(L, diag::err_invalid_declarator_scope)
          << Name << cast<NamedDecl>(DC)->getDeclName() << R;
      }
      InvalidDecl = true;
    }
  }

  if (PrevDecl && PrevDecl->isTemplateParameter()) {
    // Maybe we will complain about the shadowed template parameter.
    InvalidDecl = InvalidDecl 
      || DiagnoseTemplateParameterShadow(D.getIdentifierLoc(), PrevDecl);
    // Just pretend that we didn't see the previous declaration.
    PrevDecl = 0;
  }

  // In C++, the previous declaration we find might be a tag type
  // (class or enum). In this case, the new declaration will hide the
  // tag type. 
  if (PrevDecl && PrevDecl->getIdentifierNamespace() == Decl::IDNS_Tag)
    PrevDecl = 0;

  QualType R = GetTypeForDeclarator(D, S);
  assert(!R.isNull() && "GetTypeForDeclarator() returned null type");

  if (D.getDeclSpec().getStorageClassSpec() == DeclSpec::SCS_typedef) {
    New = ActOnTypedefDeclarator(S, D, DC, R, LastDeclarator, PrevDecl,
                                 InvalidDecl);
  } else if (R.getTypePtr()->isFunctionType()) {
    New = ActOnFunctionDeclarator(S, D, DC, R, LastDeclarator, PrevDecl, 
                                  IsFunctionDefinition, InvalidDecl);
  } else {
    New = ActOnVariableDeclarator(S, D, DC, R, LastDeclarator, PrevDecl, 
                                  InvalidDecl);
  }

  if (New == 0)
    return 0;
  
  // Set the lexical context. If the declarator has a C++ scope specifier, the
  // lexical context will be different from the semantic context.
  New->setLexicalDeclContext(CurContext);

  // If this has an identifier, add it to the scope stack.
  if (Name)
    PushOnScopeChains(New, S);
  // If any semantic error occurred, mark the decl as invalid.
  if (D.getInvalidType() || InvalidDecl)
    New->setInvalidDecl();
  
  return New;
}

NamedDecl*
Sema::ActOnTypedefDeclarator(Scope* S, Declarator& D, DeclContext* DC,
                             QualType R, Decl* LastDeclarator,
                             Decl* PrevDecl, bool& InvalidDecl) {
  // Typedef declarators cannot be qualified (C++ [dcl.meaning]p1).
  if (D.getCXXScopeSpec().isSet()) {
    Diag(D.getIdentifierLoc(), diag::err_qualified_typedef_declarator)
      << D.getCXXScopeSpec().getRange();
    InvalidDecl = true;
    // Pretend we didn't see the scope specifier.
    DC = 0;
  }

  // Check that there are no default arguments (C++ only).
  if (getLangOptions().CPlusPlus)
    CheckExtraCXXDefaultArguments(D);

  TypedefDecl *NewTD = ParseTypedefDecl(S, D, R, LastDeclarator);
  if (!NewTD) return 0;

  // Handle attributes prior to checking for duplicates in MergeVarDecl
  ProcessDeclAttributes(NewTD, D);
  // Merge the decl with the existing one if appropriate. If the decl is
  // in an outer scope, it isn't the same thing.
  if (PrevDecl && isDeclInScope(PrevDecl, DC, S)) {
    NewTD = MergeTypeDefDecl(NewTD, PrevDecl);
    if (NewTD == 0) return 0;
  }

  if (S->getFnParent() == 0) {
    // C99 6.7.7p2: If a typedef name specifies a variably modified type
    // then it shall have block scope.
    if (NewTD->getUnderlyingType()->isVariablyModifiedType()) {
      if (NewTD->getUnderlyingType()->isVariableArrayType())
        Diag(D.getIdentifierLoc(), diag::err_vla_decl_in_file_scope);
      else
        Diag(D.getIdentifierLoc(), diag::err_vm_decl_in_file_scope);

      InvalidDecl = true;
    }
  }
  return NewTD;
}

NamedDecl*
Sema::ActOnVariableDeclarator(Scope* S, Declarator& D, DeclContext* DC,
                              QualType R, Decl* LastDeclarator,
                              Decl* PrevDecl, bool& InvalidDecl) {
  DeclarationName Name = GetNameForDeclarator(D);

  // Check that there are no default arguments (C++ only).
  if (getLangOptions().CPlusPlus)
    CheckExtraCXXDefaultArguments(D);

  if (R.getTypePtr()->isObjCInterfaceType()) {
    Diag(D.getIdentifierLoc(), diag::err_statically_allocated_object)
      << D.getIdentifier();
    InvalidDecl = true;
  }

  VarDecl *NewVD;
  VarDecl::StorageClass SC;
  switch (D.getDeclSpec().getStorageClassSpec()) {
  default: assert(0 && "Unknown storage class!");
  case DeclSpec::SCS_unspecified:    SC = VarDecl::None; break;
  case DeclSpec::SCS_extern:         SC = VarDecl::Extern; break;
  case DeclSpec::SCS_static:         SC = VarDecl::Static; break;
  case DeclSpec::SCS_auto:           SC = VarDecl::Auto; break;
  case DeclSpec::SCS_register:       SC = VarDecl::Register; break;
  case DeclSpec::SCS_private_extern: SC = VarDecl::PrivateExtern; break;
  case DeclSpec::SCS_mutable:
    // mutable can only appear on non-static class members, so it's always
    // an error here
    Diag(D.getIdentifierLoc(), diag::err_mutable_nonmember);
    InvalidDecl = true;
    SC = VarDecl::None;
    break;
  }

  IdentifierInfo *II = Name.getAsIdentifierInfo();
  if (!II) {
    Diag(D.getIdentifierLoc(), diag::err_bad_variable_name)
      << Name.getAsString();
    return 0;
  }

  if (DC->isRecord()) {
    // This is a static data member for a C++ class.
    NewVD = CXXClassVarDecl::Create(Context, cast<CXXRecordDecl>(DC),
                                    D.getIdentifierLoc(), II,
                                    R);
  } else {
    bool ThreadSpecified = D.getDeclSpec().isThreadSpecified();
    if (S->getFnParent() == 0) {
      // C99 6.9p2: The storage-class specifiers auto and register shall not
      // appear in the declaration specifiers in an external declaration.
      if (SC == VarDecl::Auto || SC == VarDecl::Register) {
        Diag(D.getIdentifierLoc(), diag::err_typecheck_sclass_fscope);
        InvalidDecl = true;
      }
    }
    NewVD = VarDecl::Create(Context, DC, D.getIdentifierLoc(), 
                            II, R, SC, 
                            // FIXME: Move to DeclGroup...
                            D.getDeclSpec().getSourceRange().getBegin());
    NewVD->setThreadSpecified(ThreadSpecified);
  }
  NewVD->setNextDeclarator(LastDeclarator);

  // Handle attributes prior to checking for duplicates in MergeVarDecl
  ProcessDeclAttributes(NewVD, D);

  // Handle GNU asm-label extension (encoded as an attribute).
  if (Expr *E = (Expr*) D.getAsmLabel()) {
    // The parser guarantees this is a string.
    StringLiteral *SE = cast<StringLiteral>(E);  
    NewVD->addAttr(new AsmLabelAttr(std::string(SE->getStrData(),
                                                SE->getByteLength())));
  }

  // Emit an error if an address space was applied to decl with local storage.
  // This includes arrays of objects with address space qualifiers, but not
  // automatic variables that point to other address spaces.
  // ISO/IEC TR 18037 S5.1.2
  if (NewVD->hasLocalStorage() && (NewVD->getType().getAddressSpace() != 0)) {
    Diag(D.getIdentifierLoc(), diag::err_as_qualified_auto_decl);
    InvalidDecl = true;
  }
  // Merge the decl with the existing one if appropriate. If the decl is
  // in an outer scope, it isn't the same thing.
  if (PrevDecl && isDeclInScope(PrevDecl, DC, S)) {
    if (isa<FieldDecl>(PrevDecl) && D.getCXXScopeSpec().isSet()) {
      // The user tried to define a non-static data member
      // out-of-line (C++ [dcl.meaning]p1).
      Diag(NewVD->getLocation(), diag::err_nonstatic_member_out_of_line)
        << D.getCXXScopeSpec().getRange();
      NewVD->Destroy(Context);
      return 0;
    }

    NewVD = MergeVarDecl(NewVD, PrevDecl);
    if (NewVD == 0) return 0;

    if (D.getCXXScopeSpec().isSet()) {
      // No previous declaration in the qualifying scope.
      Diag(D.getIdentifierLoc(), diag::err_typecheck_no_member)
        << Name << D.getCXXScopeSpec().getRange();
      InvalidDecl = true;
    }
  }
  return NewVD;
}

NamedDecl* 
Sema::ActOnFunctionDeclarator(Scope* S, Declarator& D, DeclContext* DC,
                              QualType R, Decl *LastDeclarator,
                              Decl* PrevDecl, bool IsFunctionDefinition,
                              bool& InvalidDecl) {
  assert(R.getTypePtr()->isFunctionType());

  DeclarationName Name = GetNameForDeclarator(D);
  FunctionDecl::StorageClass SC = FunctionDecl::None;
  switch (D.getDeclSpec().getStorageClassSpec()) {
  default: assert(0 && "Unknown storage class!");
  case DeclSpec::SCS_auto:
  case DeclSpec::SCS_register:
  case DeclSpec::SCS_mutable:
    Diag(D.getIdentifierLoc(), diag::err_typecheck_sclass_func);
    InvalidDecl = true;
    break;
  case DeclSpec::SCS_unspecified: SC = FunctionDecl::None; break;
  case DeclSpec::SCS_extern:      SC = FunctionDecl::Extern; break;
  case DeclSpec::SCS_static:      SC = FunctionDecl::Static; break;
  case DeclSpec::SCS_private_extern: SC = FunctionDecl::PrivateExtern;break;
  }

  bool isInline = D.getDeclSpec().isInlineSpecified();
  // bool isVirtual = D.getDeclSpec().isVirtualSpecified();
  bool isExplicit = D.getDeclSpec().isExplicitSpecified();

  FunctionDecl *NewFD;
  if (D.getKind() == Declarator::DK_Constructor) {
    // This is a C++ constructor declaration.
    assert(DC->isRecord() &&
           "Constructors can only be declared in a member context");

    InvalidDecl = InvalidDecl || CheckConstructorDeclarator(D, R, SC);

    // Create the new declaration
    NewFD = CXXConstructorDecl::Create(Context, 
                                       cast<CXXRecordDecl>(DC),
                                       D.getIdentifierLoc(), Name, R,
                                       isExplicit, isInline,
                                       /*isImplicitlyDeclared=*/false);

    if (InvalidDecl)
      NewFD->setInvalidDecl();
  } else if (D.getKind() == Declarator::DK_Destructor) {
    // This is a C++ destructor declaration.
    if (DC->isRecord()) {
      InvalidDecl = InvalidDecl || CheckDestructorDeclarator(D, R, SC);

      NewFD = CXXDestructorDecl::Create(Context,
                                        cast<CXXRecordDecl>(DC),
                                        D.getIdentifierLoc(), Name, R, 
                                        isInline,
                                        /*isImplicitlyDeclared=*/false);

      if (InvalidDecl)
        NewFD->setInvalidDecl();
    } else {
      Diag(D.getIdentifierLoc(), diag::err_destructor_not_member);

      // Create a FunctionDecl to satisfy the function definition parsing
      // code path.
      NewFD = FunctionDecl::Create(Context, DC, D.getIdentifierLoc(),
                                   Name, R, SC, isInline, 
                                   // FIXME: Move to DeclGroup...
                                   D.getDeclSpec().getSourceRange().getBegin());
      InvalidDecl = true;
      NewFD->setInvalidDecl();
    }
  } else if (D.getKind() == Declarator::DK_Conversion) {
    if (!DC->isRecord()) {
      Diag(D.getIdentifierLoc(),
           diag::err_conv_function_not_member);
      return 0;
    } else {
      InvalidDecl = InvalidDecl || CheckConversionDeclarator(D, R, SC);

      NewFD = CXXConversionDecl::Create(Context, cast<CXXRecordDecl>(DC),
                                        D.getIdentifierLoc(), Name, R,
                                        isInline, isExplicit);
        
      if (InvalidDecl)
        NewFD->setInvalidDecl();
    }
  } else if (DC->isRecord()) {
    // This is a C++ method declaration.
    NewFD = CXXMethodDecl::Create(Context, cast<CXXRecordDecl>(DC),
                                  D.getIdentifierLoc(), Name, R,
                                  (SC == FunctionDecl::Static), isInline);
  } else {
    NewFD = FunctionDecl::Create(Context, DC,
                                 D.getIdentifierLoc(),
                                 Name, R, SC, isInline, 
                                 // FIXME: Move to DeclGroup...
                                 D.getDeclSpec().getSourceRange().getBegin());
  }
  NewFD->setNextDeclarator(LastDeclarator);

  // Set the lexical context. If the declarator has a C++
  // scope specifier, the lexical context will be different
  // from the semantic context.
  NewFD->setLexicalDeclContext(CurContext);

  // Handle GNU asm-label extension (encoded as an attribute).
  if (Expr *E = (Expr*) D.getAsmLabel()) {
    // The parser guarantees this is a string.
    StringLiteral *SE = cast<StringLiteral>(E);  
    NewFD->addAttr(new AsmLabelAttr(std::string(SE->getStrData(),
                                                SE->getByteLength())));
  }

  // Copy the parameter declarations from the declarator D to
  // the function declaration NewFD, if they are available.
  if (D.getNumTypeObjects() > 0) {
    DeclaratorChunk::FunctionTypeInfo &FTI = D.getTypeObject(0).Fun;

    // Create Decl objects for each parameter, adding them to the
    // FunctionDecl.
    llvm::SmallVector<ParmVarDecl*, 16> Params;
  
    // Check for C99 6.7.5.3p10 - foo(void) is a non-varargs
    // function that takes no arguments, not a function that takes a
    // single void argument.
    // We let through "const void" here because Sema::GetTypeForDeclarator
    // already checks for that case.
    if (FTI.NumArgs == 1 && !FTI.isVariadic && FTI.ArgInfo[0].Ident == 0 &&
        FTI.ArgInfo[0].Param &&
        ((ParmVarDecl*)FTI.ArgInfo[0].Param)->getType()->isVoidType()) {
      // empty arg list, don't push any params.
      ParmVarDecl *Param = (ParmVarDecl*)FTI.ArgInfo[0].Param;

      // In C++, the empty parameter-type-list must be spelled "void"; a
      // typedef of void is not permitted.
      if (getLangOptions().CPlusPlus &&
          Param->getType().getUnqualifiedType() != Context.VoidTy) {
        Diag(Param->getLocation(), diag::ext_param_typedef_of_void);
      }
    } else if (FTI.NumArgs > 0 && FTI.ArgInfo[0].Param != 0) {
      for (unsigned i = 0, e = FTI.NumArgs; i != e; ++i)
        Params.push_back((ParmVarDecl *)FTI.ArgInfo[i].Param);
    }
  
    NewFD->setParams(Context, &Params[0], Params.size());
  } else if (R->getAsTypedefType()) {
    // When we're declaring a function with a typedef, as in the
    // following example, we'll need to synthesize (unnamed)
    // parameters for use in the declaration.
    //
    // @code
    // typedef void fn(int);
    // fn f;
    // @endcode
    const FunctionTypeProto *FT = R->getAsFunctionTypeProto();
    if (!FT) {
      // This is a typedef of a function with no prototype, so we
      // don't need to do anything.
    } else if ((FT->getNumArgs() == 0) ||
               (FT->getNumArgs() == 1 && !FT->isVariadic() &&
                FT->getArgType(0)->isVoidType())) {
      // This is a zero-argument function. We don't need to do anything.
    } else {
      // Synthesize a parameter for each argument type.
      llvm::SmallVector<ParmVarDecl*, 16> Params;
      for (FunctionTypeProto::arg_type_iterator ArgType = FT->arg_type_begin();
           ArgType != FT->arg_type_end(); ++ArgType) {
        Params.push_back(ParmVarDecl::Create(Context, DC,
                                             SourceLocation(), 0,
                                             *ArgType, VarDecl::None,
                                             0));
      }

      NewFD->setParams(Context, &Params[0], Params.size());
    }
  }

  if (CXXConstructorDecl *Constructor = dyn_cast<CXXConstructorDecl>(NewFD))
    InvalidDecl = InvalidDecl || CheckConstructor(Constructor);
  else if (isa<CXXDestructorDecl>(NewFD)) {
    CXXRecordDecl *Record = cast<CXXRecordDecl>(NewFD->getParent());
    Record->setUserDeclaredDestructor(true);
    // C++ [class]p4: A POD-struct is an aggregate class that has [...] no
    // user-defined destructor.
    Record->setPOD(false);
  } else if (CXXConversionDecl *Conversion =
             dyn_cast<CXXConversionDecl>(NewFD))
    ActOnConversionDeclarator(Conversion);

  // Extra checking for C++ overloaded operators (C++ [over.oper]).
  if (NewFD->isOverloadedOperator() &&
      CheckOverloadedOperatorDeclaration(NewFD))
    NewFD->setInvalidDecl();
    
  // Merge the decl with the existing one if appropriate. Since C functions
  // are in a flat namespace, make sure we consider decls in outer scopes.
  if (PrevDecl &&
      (!getLangOptions().CPlusPlus||isDeclInScope(PrevDecl, DC, S))) {
    bool Redeclaration = false;

    // If C++, determine whether NewFD is an overload of PrevDecl or
    // a declaration that requires merging. If it's an overload,
    // there's no more work to do here; we'll just add the new
    // function to the scope.
    OverloadedFunctionDecl::function_iterator MatchedDecl;
    if (!getLangOptions().CPlusPlus ||
        !IsOverload(NewFD, PrevDecl, MatchedDecl)) {
      Decl *OldDecl = PrevDecl;

      // If PrevDecl was an overloaded function, extract the
      // FunctionDecl that matched.
      if (isa<OverloadedFunctionDecl>(PrevDecl))
        OldDecl = *MatchedDecl;

      // NewFD and PrevDecl represent declarations that need to be
      // merged. 
      NewFD = MergeFunctionDecl(NewFD, OldDecl, Redeclaration);

      if (NewFD == 0) return 0;
      if (Redeclaration) {
        NewFD->setPreviousDeclaration(cast<FunctionDecl>(OldDecl));

        // An out-of-line member function declaration must also be a
        // definition (C++ [dcl.meaning]p1).
        if (!IsFunctionDefinition && D.getCXXScopeSpec().isSet() &&
            !InvalidDecl) {
          Diag(NewFD->getLocation(), diag::err_out_of_line_declaration)
            << D.getCXXScopeSpec().getRange();
          NewFD->setInvalidDecl();
        }
      }
    }

    if (!Redeclaration && D.getCXXScopeSpec().isSet()) {
      // The user tried to provide an out-of-line definition for a
      // member function, but there was no such member function
      // declared (C++ [class.mfct]p2). For example:
      // 
      // class X {
      //   void f() const;
      // }; 
      //
      // void X::f() { } // ill-formed
      //
      // Complain about this problem, and attempt to suggest close
      // matches (e.g., those that differ only in cv-qualifiers and
      // whether the parameter types are references).
      Diag(D.getIdentifierLoc(), diag::err_member_def_does_not_match)
        << cast<CXXRecordDecl>(DC)->getDeclName() 
        << D.getCXXScopeSpec().getRange();
      InvalidDecl = true;
        
      PrevDecl = LookupDecl(Name, Decl::IDNS_Ordinary, S, DC);
      if (!PrevDecl) {
        // Nothing to suggest.
      } else if (OverloadedFunctionDecl *Ovl 
                 = dyn_cast<OverloadedFunctionDecl>(PrevDecl)) {
        for (OverloadedFunctionDecl::function_iterator 
               Func = Ovl->function_begin(),
               FuncEnd = Ovl->function_end();
             Func != FuncEnd; ++Func) {
          if (isNearlyMatchingMemberFunction(Context, *Func, NewFD))
            Diag((*Func)->getLocation(), diag::note_member_def_close_match);
            
        }
      } else if (CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(PrevDecl)) {
        // Suggest this no matter how mismatched it is; it's the only
        // thing we have.
        unsigned diag;
        if (isNearlyMatchingMemberFunction(Context, Method, NewFD))
          diag = diag::note_member_def_close_match;
        else if (Method->getBody())
          diag = diag::note_previous_definition;
        else
          diag = diag::note_previous_declaration;
        Diag(Method->getLocation(), diag);
      }
        
      PrevDecl = 0;
    }
  }
  // Handle attributes. We need to have merged decls when handling attributes
  // (for example to check for conflicts, etc).
  ProcessDeclAttributes(NewFD, D);

  if (getLangOptions().CPlusPlus) {
    // In C++, check default arguments now that we have merged decls.
    CheckCXXDefaultArguments(NewFD);

    // An out-of-line member function declaration must also be a
    // definition (C++ [dcl.meaning]p1).
    if (!IsFunctionDefinition && D.getCXXScopeSpec().isSet() && !InvalidDecl) {
      Diag(NewFD->getLocation(), diag::err_out_of_line_declaration)
        << D.getCXXScopeSpec().getRange();
      InvalidDecl = true;
    }
  }
  return NewFD;
}

void Sema::InitializerElementNotConstant(const Expr *Init) {
  Diag(Init->getExprLoc(), diag::err_init_element_not_constant)
    << Init->getSourceRange();
}

bool Sema::CheckAddressConstantExpressionLValue(const Expr* Init) {
  switch (Init->getStmtClass()) {
  default:
    InitializerElementNotConstant(Init);
    return true;
  case Expr::ParenExprClass: {
    const ParenExpr* PE = cast<ParenExpr>(Init);
    return CheckAddressConstantExpressionLValue(PE->getSubExpr());
  }
  case Expr::CompoundLiteralExprClass:
    return cast<CompoundLiteralExpr>(Init)->isFileScope();
  case Expr::DeclRefExprClass: 
  case Expr::QualifiedDeclRefExprClass: {
    const Decl *D = cast<DeclRefExpr>(Init)->getDecl();
    if (const VarDecl *VD = dyn_cast<VarDecl>(D)) {
      if (VD->hasGlobalStorage())
        return false;
      InitializerElementNotConstant(Init);
      return true;
    }
    if (isa<FunctionDecl>(D))
      return false;
    InitializerElementNotConstant(Init);
    return true;
  }
  case Expr::MemberExprClass: {
    const MemberExpr *M = cast<MemberExpr>(Init);
    if (M->isArrow())
      return CheckAddressConstantExpression(M->getBase());
    return CheckAddressConstantExpressionLValue(M->getBase());
  }
  case Expr::ArraySubscriptExprClass: {
    // FIXME: Should we pedwarn for "x[0+0]" (where x is a pointer)?
    const ArraySubscriptExpr *ASE = cast<ArraySubscriptExpr>(Init);
    return CheckAddressConstantExpression(ASE->getBase()) ||
           CheckArithmeticConstantExpression(ASE->getIdx());
  }
  case Expr::StringLiteralClass:
  case Expr::PredefinedExprClass:
    return false;
  case Expr::UnaryOperatorClass: {
    const UnaryOperator *Exp = cast<UnaryOperator>(Init);

    // C99 6.6p9
    if (Exp->getOpcode() == UnaryOperator::Deref)
      return CheckAddressConstantExpression(Exp->getSubExpr());

    InitializerElementNotConstant(Init);
    return true;
  }
  }
}

bool Sema::CheckAddressConstantExpression(const Expr* Init) {
  switch (Init->getStmtClass()) {
  default:
    InitializerElementNotConstant(Init);
    return true;
  case Expr::ParenExprClass:
    return CheckAddressConstantExpression(cast<ParenExpr>(Init)->getSubExpr());
  case Expr::StringLiteralClass:
  case Expr::ObjCStringLiteralClass:
    return false;
  case Expr::CallExprClass:
  case Expr::CXXOperatorCallExprClass:
    // __builtin___CFStringMakeConstantString is a valid constant l-value.
    if (cast<CallExpr>(Init)->isBuiltinCall() == 
           Builtin::BI__builtin___CFStringMakeConstantString)
      return false;
      
    InitializerElementNotConstant(Init);
    return true;
      
  case Expr::UnaryOperatorClass: {
    const UnaryOperator *Exp = cast<UnaryOperator>(Init);

    // C99 6.6p9
    if (Exp->getOpcode() == UnaryOperator::AddrOf)
      return CheckAddressConstantExpressionLValue(Exp->getSubExpr());

    if (Exp->getOpcode() == UnaryOperator::Extension)
      return CheckAddressConstantExpression(Exp->getSubExpr());
  
    InitializerElementNotConstant(Init);
    return true;
  }
  case Expr::BinaryOperatorClass: {
    // FIXME: Should we pedwarn for expressions like "a + 1 + 2"?
    const BinaryOperator *Exp = cast<BinaryOperator>(Init);

    Expr *PExp = Exp->getLHS();
    Expr *IExp = Exp->getRHS();
    if (IExp->getType()->isPointerType())
      std::swap(PExp, IExp);

    // FIXME: Should we pedwarn if IExp isn't an integer constant expression?
    return CheckAddressConstantExpression(PExp) ||
           CheckArithmeticConstantExpression(IExp);
  }
  case Expr::ImplicitCastExprClass:
  case Expr::CStyleCastExprClass: {
    const Expr* SubExpr = cast<CastExpr>(Init)->getSubExpr();
    if (Init->getStmtClass() == Expr::ImplicitCastExprClass) {
      // Check for implicit promotion
      if (SubExpr->getType()->isFunctionType() ||
          SubExpr->getType()->isArrayType())
        return CheckAddressConstantExpressionLValue(SubExpr);
    }

    // Check for pointer->pointer cast
    if (SubExpr->getType()->isPointerType())
      return CheckAddressConstantExpression(SubExpr);

    if (SubExpr->getType()->isIntegralType()) {
      // Check for the special-case of a pointer->int->pointer cast;
      // this isn't standard, but some code requires it. See
      // PR2720 for an example.
      if (const CastExpr* SubCast = dyn_cast<CastExpr>(SubExpr)) {
        if (SubCast->getSubExpr()->getType()->isPointerType()) {
          unsigned IntWidth = Context.getIntWidth(SubCast->getType());
          unsigned PointerWidth = Context.getTypeSize(Context.VoidPtrTy);
          if (IntWidth >= PointerWidth) {
            return CheckAddressConstantExpression(SubCast->getSubExpr());
          }
        }
      }
    }
    if (SubExpr->getType()->isArithmeticType()) {
      return CheckArithmeticConstantExpression(SubExpr);
    }

    InitializerElementNotConstant(Init);
    return true;
  }
  case Expr::ConditionalOperatorClass: {
    // FIXME: Should we pedwarn here?
    const ConditionalOperator *Exp = cast<ConditionalOperator>(Init);
    if (!Exp->getCond()->getType()->isArithmeticType()) {
      InitializerElementNotConstant(Init);
      return true;
    }
    if (CheckArithmeticConstantExpression(Exp->getCond()))
      return true;
    if (Exp->getLHS() &&
        CheckAddressConstantExpression(Exp->getLHS()))
      return true;
    return CheckAddressConstantExpression(Exp->getRHS());
  }
  case Expr::AddrLabelExprClass:
    return false;
  }
}

static const Expr* FindExpressionBaseAddress(const Expr* E);

static const Expr* FindExpressionBaseAddressLValue(const Expr* E) {
  switch (E->getStmtClass()) {
  default:
    return E;
  case Expr::ParenExprClass: {
    const ParenExpr* PE = cast<ParenExpr>(E);
    return FindExpressionBaseAddressLValue(PE->getSubExpr());
  }
  case Expr::MemberExprClass: {
    const MemberExpr *M = cast<MemberExpr>(E);
    if (M->isArrow())
      return FindExpressionBaseAddress(M->getBase());
    return FindExpressionBaseAddressLValue(M->getBase());
  }
  case Expr::ArraySubscriptExprClass: {
    const ArraySubscriptExpr *ASE = cast<ArraySubscriptExpr>(E);
    return FindExpressionBaseAddress(ASE->getBase());
  }
  case Expr::UnaryOperatorClass: {
    const UnaryOperator *Exp = cast<UnaryOperator>(E);

    if (Exp->getOpcode() == UnaryOperator::Deref)
      return FindExpressionBaseAddress(Exp->getSubExpr());

    return E;
  }
  }
}

static const Expr* FindExpressionBaseAddress(const Expr* E) {
  switch (E->getStmtClass()) {
  default:
    return E;
  case Expr::ParenExprClass: {
    const ParenExpr* PE = cast<ParenExpr>(E);
    return FindExpressionBaseAddress(PE->getSubExpr());
  }
  case Expr::UnaryOperatorClass: {
    const UnaryOperator *Exp = cast<UnaryOperator>(E);

    // C99 6.6p9
    if (Exp->getOpcode() == UnaryOperator::AddrOf)
      return FindExpressionBaseAddressLValue(Exp->getSubExpr());

    if (Exp->getOpcode() == UnaryOperator::Extension)
      return FindExpressionBaseAddress(Exp->getSubExpr());
  
    return E;
  }
  case Expr::BinaryOperatorClass: {
    const BinaryOperator *Exp = cast<BinaryOperator>(E);

    Expr *PExp = Exp->getLHS();
    Expr *IExp = Exp->getRHS();
    if (IExp->getType()->isPointerType())
      std::swap(PExp, IExp);

    return FindExpressionBaseAddress(PExp);
  }
  case Expr::ImplicitCastExprClass: {
    const Expr* SubExpr = cast<ImplicitCastExpr>(E)->getSubExpr();

    // Check for implicit promotion
    if (SubExpr->getType()->isFunctionType() ||
        SubExpr->getType()->isArrayType())
      return FindExpressionBaseAddressLValue(SubExpr);

    // Check for pointer->pointer cast
    if (SubExpr->getType()->isPointerType())
      return FindExpressionBaseAddress(SubExpr);

    // We assume that we have an arithmetic expression here;
    // if we don't, we'll figure it out later
    return 0;
  }
  case Expr::CStyleCastExprClass: {
    const Expr* SubExpr = cast<CastExpr>(E)->getSubExpr();

    // Check for pointer->pointer cast
    if (SubExpr->getType()->isPointerType())
      return FindExpressionBaseAddress(SubExpr);

    // We assume that we have an arithmetic expression here;
    // if we don't, we'll figure it out later
    return 0;
  }
  }
}

bool Sema::CheckArithmeticConstantExpression(const Expr* Init) {  
  switch (Init->getStmtClass()) {
  default:
    InitializerElementNotConstant(Init);
    return true;
  case Expr::ParenExprClass: {
    const ParenExpr* PE = cast<ParenExpr>(Init);
    return CheckArithmeticConstantExpression(PE->getSubExpr());
  }
  case Expr::FloatingLiteralClass:
  case Expr::IntegerLiteralClass:
  case Expr::CharacterLiteralClass:
  case Expr::ImaginaryLiteralClass:
  case Expr::TypesCompatibleExprClass:
  case Expr::CXXBoolLiteralExprClass:
    return false;
  case Expr::CallExprClass: 
  case Expr::CXXOperatorCallExprClass: {
    const CallExpr *CE = cast<CallExpr>(Init);

    // Allow any constant foldable calls to builtins.
    if (CE->isBuiltinCall() && CE->isEvaluatable(Context))
      return false;
    
    InitializerElementNotConstant(Init);
    return true;
  }
  case Expr::DeclRefExprClass:
  case Expr::QualifiedDeclRefExprClass: {
    const Decl *D = cast<DeclRefExpr>(Init)->getDecl();
    if (isa<EnumConstantDecl>(D))
      return false;
    InitializerElementNotConstant(Init);
    return true;
  }
  case Expr::CompoundLiteralExprClass:
    // Allow "(vector type){2,4}"; normal C constraints don't allow this,
    // but vectors are allowed to be magic.
    if (Init->getType()->isVectorType())
      return false;
    InitializerElementNotConstant(Init);
    return true;
  case Expr::UnaryOperatorClass: {
    const UnaryOperator *Exp = cast<UnaryOperator>(Init);
  
    switch (Exp->getOpcode()) {
    // Address, indirect, pre/post inc/dec, etc are not valid constant exprs.
    // See C99 6.6p3.
    default:
      InitializerElementNotConstant(Init);
      return true;
    case UnaryOperator::OffsetOf:
      if (Exp->getSubExpr()->getType()->isConstantSizeType())
        return false;
      InitializerElementNotConstant(Init);
      return true;
    case UnaryOperator::Extension:
    case UnaryOperator::LNot:
    case UnaryOperator::Plus:
    case UnaryOperator::Minus:
    case UnaryOperator::Not:
      return CheckArithmeticConstantExpression(Exp->getSubExpr());
    }
  }
  case Expr::SizeOfAlignOfExprClass: {
    const SizeOfAlignOfExpr *Exp = cast<SizeOfAlignOfExpr>(Init);
    // Special check for void types, which are allowed as an extension
    if (Exp->getTypeOfArgument()->isVoidType())
      return false;
    // alignof always evaluates to a constant.
    // FIXME: is sizeof(int[3.0]) a constant expression?
    if (Exp->isSizeOf() && !Exp->getTypeOfArgument()->isConstantSizeType()) {
      InitializerElementNotConstant(Init);
      return true;
    }
    return false;
  }
  case Expr::BinaryOperatorClass: {
    const BinaryOperator *Exp = cast<BinaryOperator>(Init);

    if (Exp->getLHS()->getType()->isArithmeticType() &&
        Exp->getRHS()->getType()->isArithmeticType()) {
      return CheckArithmeticConstantExpression(Exp->getLHS()) ||
             CheckArithmeticConstantExpression(Exp->getRHS());
    }

    if (Exp->getLHS()->getType()->isPointerType() &&
        Exp->getRHS()->getType()->isPointerType()) {
      const Expr* LHSBase = FindExpressionBaseAddress(Exp->getLHS());
      const Expr* RHSBase = FindExpressionBaseAddress(Exp->getRHS());

      // Only allow a null (constant integer) base; we could
      // allow some additional cases if necessary, but this
      // is sufficient to cover offsetof-like constructs.
      if (!LHSBase && !RHSBase) {
        return CheckAddressConstantExpression(Exp->getLHS()) ||
               CheckAddressConstantExpression(Exp->getRHS());
      }
    }

    InitializerElementNotConstant(Init);
    return true;
  }
  case Expr::ImplicitCastExprClass:
  case Expr::CStyleCastExprClass: {
    const Expr *SubExpr = cast<CastExpr>(Init)->getSubExpr();
    if (SubExpr->getType()->isArithmeticType())
      return CheckArithmeticConstantExpression(SubExpr);

    if (SubExpr->getType()->isPointerType()) {
      const Expr* Base = FindExpressionBaseAddress(SubExpr);
      // If the pointer has a null base, this is an offsetof-like construct
      if (!Base)
        return CheckAddressConstantExpression(SubExpr);
    }

    InitializerElementNotConstant(Init);
    return true;
  }
  case Expr::ConditionalOperatorClass: {
    const ConditionalOperator *Exp = cast<ConditionalOperator>(Init);
    
    // If GNU extensions are disabled, we require all operands to be arithmetic
    // constant expressions.
    if (getLangOptions().NoExtensions) {
      return CheckArithmeticConstantExpression(Exp->getCond()) ||
          (Exp->getLHS() && CheckArithmeticConstantExpression(Exp->getLHS())) ||
             CheckArithmeticConstantExpression(Exp->getRHS());
    }

    // Otherwise, we have to emulate some of the behavior of fold here.
    // Basically GCC treats things like "4 ? 1 : somefunc()" as a constant
    // because it can constant fold things away.  To retain compatibility with
    // GCC code, we see if we can fold the condition to a constant (which we
    // should always be able to do in theory).  If so, we only require the
    // specified arm of the conditional to be a constant.  This is a horrible
    // hack, but is require by real world code that uses __builtin_constant_p.
    Expr::EvalResult EvalResult;
    if (!Exp->getCond()->Evaluate(EvalResult, Context) || 
        EvalResult.HasSideEffects) {
      // If Evaluate couldn't fold it, CheckArithmeticConstantExpression
      // won't be able to either.  Use it to emit the diagnostic though.
      bool Res = CheckArithmeticConstantExpression(Exp->getCond());
      assert(Res && "Evaluate couldn't evaluate this constant?");
      return Res;
    }
    
    // Verify that the side following the condition is also a constant.
    const Expr *TrueSide = Exp->getLHS(), *FalseSide = Exp->getRHS();
    if (EvalResult.Val.getInt() == 0) 
      std::swap(TrueSide, FalseSide);
    
    if (TrueSide && CheckArithmeticConstantExpression(TrueSide))
      return true;
      
    // Okay, the evaluated side evaluates to a constant, so we accept this.
    // Check to see if the other side is obviously not a constant.  If so, 
    // emit a warning that this is a GNU extension.
    if (FalseSide && !FalseSide->isEvaluatable(Context))
      Diag(Init->getExprLoc(), 
           diag::ext_typecheck_expression_not_constant_but_accepted)
        << FalseSide->getSourceRange();
    return false;
  }
  }
}

bool Sema::CheckForConstantInitializer(Expr *Init, QualType DclT) {
  if (DesignatedInitExpr *DIE = dyn_cast<DesignatedInitExpr>(Init))
    Init = DIE->getInit();

  Init = Init->IgnoreParens();

  if (Init->isEvaluatable(Context))
    return false;

  // Look through CXXDefaultArgExprs; they have no meaning in this context.
  if (CXXDefaultArgExpr* DAE = dyn_cast<CXXDefaultArgExpr>(Init))
    return CheckForConstantInitializer(DAE->getExpr(), DclT);

  if (CompoundLiteralExpr *e = dyn_cast<CompoundLiteralExpr>(Init))
    return CheckForConstantInitializer(e->getInitializer(), DclT);

  if (InitListExpr *Exp = dyn_cast<InitListExpr>(Init)) {
    unsigned numInits = Exp->getNumInits();
    for (unsigned i = 0; i < numInits; i++) {
      // FIXME: Need to get the type of the declaration for C++,
      // because it could be a reference?
      if (CheckForConstantInitializer(Exp->getInit(i),
                                      Exp->getInit(i)->getType()))
        return true;
    }
    return false;
  }

  // FIXME: We can probably remove some of this code below, now that 
  // Expr::Evaluate is doing the heavy lifting for scalars.
  
  if (Init->isNullPointerConstant(Context))
    return false;
  if (Init->getType()->isArithmeticType()) {
    QualType InitTy = Context.getCanonicalType(Init->getType())
                             .getUnqualifiedType();
    if (InitTy == Context.BoolTy) {
      // Special handling for pointers implicitly cast to bool;
      // (e.g. "_Bool rr = &rr;"). This is only legal at the top level.
      if (ImplicitCastExpr* ICE = dyn_cast<ImplicitCastExpr>(Init)) {
        Expr* SubE = ICE->getSubExpr();
        if (SubE->getType()->isPointerType() ||
            SubE->getType()->isArrayType() ||
            SubE->getType()->isFunctionType()) {
          return CheckAddressConstantExpression(Init);
        }
      }
    } else if (InitTy->isIntegralType()) {
      Expr* SubE = 0;
      if (CastExpr* CE = dyn_cast<CastExpr>(Init))
        SubE = CE->getSubExpr();
      // Special check for pointer cast to int; we allow as an extension
      // an address constant cast to an integer if the integer
      // is of an appropriate width (this sort of code is apparently used
      // in some places).
      // FIXME: Add pedwarn?
      // FIXME: Don't allow bitfields here!  Need the FieldDecl for that.
      if (SubE && (SubE->getType()->isPointerType() ||
                   SubE->getType()->isArrayType() ||
                   SubE->getType()->isFunctionType())) {
        unsigned IntWidth = Context.getTypeSize(Init->getType());
        unsigned PointerWidth = Context.getTypeSize(Context.VoidPtrTy);
        if (IntWidth >= PointerWidth)
          return CheckAddressConstantExpression(Init);
      }
    }

    return CheckArithmeticConstantExpression(Init);
  }

  if (Init->getType()->isPointerType())
    return CheckAddressConstantExpression(Init);

  // An array type at the top level that isn't an init-list must
  // be a string literal
  if (Init->getType()->isArrayType())
    return false;

  if (Init->getType()->isFunctionType())
    return false;

  // Allow block exprs at top level.
  if (Init->getType()->isBlockPointerType())
    return false;

  // GCC cast to union extension
  // note: the validity of the cast expr is checked by CheckCastTypes()
  if (CastExpr *C = dyn_cast<CastExpr>(Init)) {
    QualType T = C->getType();
    return T->isUnionType() && CheckForConstantInitializer(C->getSubExpr(), T);
  }

  InitializerElementNotConstant(Init);
  return true;
}

void Sema::AddInitializerToDecl(DeclTy *dcl, ExprArg init) {
  AddInitializerToDecl(dcl, move(init), /*DirectInit=*/false);
}

/// AddInitializerToDecl - Adds the initializer Init to the
/// declaration dcl. If DirectInit is true, this is C++ direct
/// initialization rather than copy initialization.
void Sema::AddInitializerToDecl(DeclTy *dcl, ExprArg init, bool DirectInit) {
  Decl *RealDecl = static_cast<Decl *>(dcl);
  Expr *Init = static_cast<Expr *>(init.release());
  assert(Init && "missing initializer");
  
  // If there is no declaration, there was an error parsing it.  Just ignore
  // the initializer.
  if (RealDecl == 0) {
    delete Init;
    return;
  }
  
  VarDecl *VDecl = dyn_cast<VarDecl>(RealDecl);
  if (!VDecl) {
    Diag(RealDecl->getLocation(), diag::err_illegal_initializer);
    RealDecl->setInvalidDecl();
    return;
  }  
  // Get the decls type and save a reference for later, since
  // CheckInitializerTypes may change it.
  QualType DclT = VDecl->getType(), SavT = DclT;
  if (VDecl->isBlockVarDecl()) {
    VarDecl::StorageClass SC = VDecl->getStorageClass();
    if (SC == VarDecl::Extern) { // C99 6.7.8p5
      Diag(VDecl->getLocation(), diag::err_block_extern_cant_init);
      VDecl->setInvalidDecl();
    } else if (!VDecl->isInvalidDecl()) {
      if (CheckInitializerTypes(Init, DclT, VDecl->getLocation(),
                                VDecl->getDeclName(), DirectInit))
        VDecl->setInvalidDecl();
      
      // C++ 3.6.2p2, allow dynamic initialization of static initializers.
      if (!getLangOptions().CPlusPlus) {
        if (SC == VarDecl::Static) // C99 6.7.8p4.
          CheckForConstantInitializer(Init, DclT);
      }
    }
  } else if (VDecl->isFileVarDecl()) {
    if (VDecl->getStorageClass() == VarDecl::Extern)
      Diag(VDecl->getLocation(), diag::warn_extern_init);
    if (!VDecl->isInvalidDecl())
      if (CheckInitializerTypes(Init, DclT, VDecl->getLocation(),
                                VDecl->getDeclName(), DirectInit))
        VDecl->setInvalidDecl();
    
    // C++ 3.6.2p2, allow dynamic initialization of static initializers.
    if (!getLangOptions().CPlusPlus) {
      // C99 6.7.8p4. All file scoped initializers need to be constant.
      CheckForConstantInitializer(Init, DclT);
    }
  }
  // If the type changed, it means we had an incomplete type that was
  // completed by the initializer. For example: 
  //   int ary[] = { 1, 3, 5 };
  // "ary" transitions from a VariableArrayType to a ConstantArrayType.
  if (!VDecl->isInvalidDecl() && (DclT != SavT)) {
    VDecl->setType(DclT);
    Init->setType(DclT);
  }
    
  // Attach the initializer to the decl.
  VDecl->setInit(Init);
  return;
}

void Sema::ActOnUninitializedDecl(DeclTy *dcl) {
  Decl *RealDecl = static_cast<Decl *>(dcl);

  // If there is no declaration, there was an error parsing it. Just ignore it.
  if (RealDecl == 0)
    return;

  if (VarDecl *Var = dyn_cast<VarDecl>(RealDecl)) {
    QualType Type = Var->getType();
    // C++ [dcl.init.ref]p3:
    //   The initializer can be omitted for a reference only in a
    //   parameter declaration (8.3.5), in the declaration of a
    //   function return type, in the declaration of a class member
    //   within its class declaration (9.2), and where the extern
    //   specifier is explicitly used.
    if (Type->isReferenceType() && 
        Var->getStorageClass() != VarDecl::Extern &&
        Var->getStorageClass() != VarDecl::PrivateExtern) {
      Diag(Var->getLocation(), diag::err_reference_var_requires_init)
        << Var->getDeclName()
        << SourceRange(Var->getLocation(), Var->getLocation());
      Var->setInvalidDecl();
      return;
    }

    // C++ [dcl.init]p9:
    //
    //   If no initializer is specified for an object, and the object
    //   is of (possibly cv-qualified) non-POD class type (or array
    //   thereof), the object shall be default-initialized; if the
    //   object is of const-qualified type, the underlying class type
    //   shall have a user-declared default constructor.
    if (getLangOptions().CPlusPlus) {
      QualType InitType = Type;
      if (const ArrayType *Array = Context.getAsArrayType(Type))
        InitType = Array->getElementType();
      if (Var->getStorageClass() != VarDecl::Extern &&
          Var->getStorageClass() != VarDecl::PrivateExtern &&
          InitType->isRecordType()) {
        const CXXConstructorDecl *Constructor
          = PerformInitializationByConstructor(InitType, 0, 0, 
                                               Var->getLocation(),
                                               SourceRange(Var->getLocation(),
                                                           Var->getLocation()),
                                               Var->getDeclName(),
                                               IK_Default);
        if (!Constructor)
          Var->setInvalidDecl();
      }
    }

#if 0
    // FIXME: Temporarily disabled because we are not properly parsing
    // linkage specifications on declarations, e.g., 
    //
    //   extern "C" const CGPoint CGPointerZero;
    //
    // C++ [dcl.init]p9:
    //
    //     If no initializer is specified for an object, and the
    //     object is of (possibly cv-qualified) non-POD class type (or
    //     array thereof), the object shall be default-initialized; if
    //     the object is of const-qualified type, the underlying class
    //     type shall have a user-declared default
    //     constructor. Otherwise, if no initializer is specified for
    //     an object, the object and its subobjects, if any, have an
    //     indeterminate initial value; if the object or any of its
    //     subobjects are of const-qualified type, the program is
    //     ill-formed.
    //
    // This isn't technically an error in C, so we don't diagnose it.
    //
    // FIXME: Actually perform the POD/user-defined default
    // constructor check.
    if (getLangOptions().CPlusPlus &&
        Context.getCanonicalType(Type).isConstQualified() &&
        Var->getStorageClass() != VarDecl::Extern)
      Diag(Var->getLocation(),  diag::err_const_var_requires_init)
        << Var->getName()
        << SourceRange(Var->getLocation(), Var->getLocation());
#endif
  }
}

/// The declarators are chained together backwards, reverse the list.
Sema::DeclTy *Sema::FinalizeDeclaratorGroup(Scope *S, DeclTy *group) {
  // Often we have single declarators, handle them quickly.
  Decl *GroupDecl = static_cast<Decl*>(group);
  if (GroupDecl == 0)
    return 0;
  
  Decl *Group = dyn_cast<Decl>(GroupDecl);
  Decl *NewGroup = 0;
  if (Group->getNextDeclarator() == 0) 
    NewGroup = Group;
  else { // reverse the list.
    while (Group) {
      Decl *Next = Group->getNextDeclarator();
      Group->setNextDeclarator(NewGroup);
      NewGroup = Group;
      Group = Next;
    }
  }
  // Perform semantic analysis that depends on having fully processed both
  // the declarator and initializer.
  for (Decl *ID = NewGroup; ID; ID = ID->getNextDeclarator()) {
    VarDecl *IDecl = dyn_cast<VarDecl>(ID);
    if (!IDecl)
      continue;
    QualType T = IDecl->getType();
    
    if (T->isVariableArrayType()) {
      const VariableArrayType *VAT = Context.getAsVariableArrayType(T);
      
      // FIXME: This won't give the correct result for 
      // int a[10][n];      
      SourceRange SizeRange = VAT->getSizeExpr()->getSourceRange();
      if (IDecl->isFileVarDecl()) {
        Diag(IDecl->getLocation(), diag::err_vla_decl_in_file_scope) <<
          SizeRange;
          
        IDecl->setInvalidDecl();
      } else {
        // C99 6.7.5.2p2: If an identifier is declared to be an object with 
        // static storage duration, it shall not have a variable length array.
        if (IDecl->getStorageClass() == VarDecl::Static) {
          Diag(IDecl->getLocation(), diag::err_vla_decl_has_static_storage)
            << SizeRange;
          IDecl->setInvalidDecl();
        } else if (IDecl->getStorageClass() == VarDecl::Extern) {
          Diag(IDecl->getLocation(), diag::err_vla_decl_has_extern_linkage)
            << SizeRange;
          IDecl->setInvalidDecl();
        }
      }
    } else if (T->isVariablyModifiedType()) {
      if (IDecl->isFileVarDecl()) {
        Diag(IDecl->getLocation(), diag::err_vm_decl_in_file_scope);
        IDecl->setInvalidDecl();
      } else {
        if (IDecl->getStorageClass() == VarDecl::Extern) {
          Diag(IDecl->getLocation(), diag::err_vm_decl_has_extern_linkage);
          IDecl->setInvalidDecl();
        }
      }
    }
     
    // Block scope. C99 6.7p7: If an identifier for an object is declared with
    // no linkage (C99 6.2.2p6), the type for the object shall be complete...
    if (IDecl->isBlockVarDecl() && 
        IDecl->getStorageClass() != VarDecl::Extern) {
      if (!IDecl->isInvalidDecl() &&
          DiagnoseIncompleteType(IDecl->getLocation(), T, 
                                 diag::err_typecheck_decl_incomplete_type))
        IDecl->setInvalidDecl();
    }
    // File scope. C99 6.9.2p2: A declaration of an identifier for and 
    // object that has file scope without an initializer, and without a
    // storage-class specifier or with the storage-class specifier "static",
    // constitutes a tentative definition. Note: A tentative definition with
    // external linkage is valid (C99 6.2.2p5).
    if (isTentativeDefinition(IDecl)) {
      if (T->isIncompleteArrayType()) {
        // C99 6.9.2 (p2, p5): Implicit initialization causes an incomplete
        // array to be completed. Don't issue a diagnostic.
      } else if (!IDecl->isInvalidDecl() &&
                 DiagnoseIncompleteType(IDecl->getLocation(), T,
                                        diag::err_typecheck_decl_incomplete_type))
        // C99 6.9.2p3: If the declaration of an identifier for an object is
        // a tentative definition and has internal linkage (C99 6.2.2p3), the  
        // declared type shall not be an incomplete type.
        IDecl->setInvalidDecl();
    }
    if (IDecl->isFileVarDecl())
      CheckForFileScopedRedefinitions(S, IDecl);
  }
  return NewGroup;
}

/// ActOnParamDeclarator - Called from Parser::ParseFunctionDeclarator()
/// to introduce parameters into function prototype scope.
Sema::DeclTy *
Sema::ActOnParamDeclarator(Scope *S, Declarator &D) {
  const DeclSpec &DS = D.getDeclSpec();

  // Verify C99 6.7.5.3p2: The only SCS allowed is 'register'.
  VarDecl::StorageClass StorageClass = VarDecl::None;
  if (DS.getStorageClassSpec() == DeclSpec::SCS_register) {
    StorageClass = VarDecl::Register;
  } else if (DS.getStorageClassSpec() != DeclSpec::SCS_unspecified) {
    Diag(DS.getStorageClassSpecLoc(),
         diag::err_invalid_storage_class_in_func_decl);
    D.getMutableDeclSpec().ClearStorageClassSpecs();
  }
  if (DS.isThreadSpecified()) {
    Diag(DS.getThreadSpecLoc(),
         diag::err_invalid_storage_class_in_func_decl);
    D.getMutableDeclSpec().ClearStorageClassSpecs();
  }
  
  // Check that there are no default arguments inside the type of this
  // parameter (C++ only).
  if (getLangOptions().CPlusPlus)
    CheckExtraCXXDefaultArguments(D);
 
  // In this context, we *do not* check D.getInvalidType(). If the declarator
  // type was invalid, GetTypeForDeclarator() still returns a "valid" type,
  // though it will not reflect the user specified type.
  QualType parmDeclType = GetTypeForDeclarator(D, S);
  
  assert(!parmDeclType.isNull() && "GetTypeForDeclarator() returned null type");

  // TODO: CHECK FOR CONFLICTS, multiple decls with same name in one scope.
  // Can this happen for params?  We already checked that they don't conflict
  // among each other.  Here they can only shadow globals, which is ok.
  IdentifierInfo *II = D.getIdentifier();
  if (II) {
    if (Decl *PrevDecl = LookupDecl(II, Decl::IDNS_Ordinary, S)) {
      if (PrevDecl->isTemplateParameter()) {
        // Maybe we will complain about the shadowed template parameter.
        DiagnoseTemplateParameterShadow(D.getIdentifierLoc(), PrevDecl);
        // Just pretend that we didn't see the previous declaration.
        PrevDecl = 0;
      } else if (S->isDeclScope(PrevDecl)) {
        Diag(D.getIdentifierLoc(), diag::err_param_redefinition) << II;

        // Recover by removing the name
        II = 0;
        D.SetIdentifier(0, D.getIdentifierLoc());
      }
    }
  }

  // Perform the default function/array conversion (C99 6.7.5.3p[7,8]).
  // Doing the promotion here has a win and a loss. The win is the type for
  // both Decl's and DeclRefExpr's will match (a convenient invariant for the
  // code generator). The loss is the orginal type isn't preserved. For example:
  //
  // void func(int parmvardecl[5]) { // convert "int [5]" to "int *"
  //    int blockvardecl[5];
  //    sizeof(parmvardecl);  // size == 4
  //    sizeof(blockvardecl); // size == 20
  // }
  //
  // For expressions, all implicit conversions are captured using the
  // ImplicitCastExpr AST node (we have no such mechanism for Decl's).
  //
  // FIXME: If a source translation tool needs to see the original type, then
  // we need to consider storing both types (in ParmVarDecl)...
  // 
  if (parmDeclType->isArrayType()) {
    // int x[restrict 4] ->  int *restrict
    parmDeclType = Context.getArrayDecayedType(parmDeclType);
  } else if (parmDeclType->isFunctionType())
    parmDeclType = Context.getPointerType(parmDeclType);

  ParmVarDecl *New = ParmVarDecl::Create(Context, CurContext, 
                                         D.getIdentifierLoc(), II,
                                         parmDeclType, StorageClass, 
                                         0);
  
  if (D.getInvalidType())
    New->setInvalidDecl();

  // Parameter declarators cannot be qualified (C++ [dcl.meaning]p1).
  if (D.getCXXScopeSpec().isSet()) {
    Diag(D.getIdentifierLoc(), diag::err_qualified_param_declarator)
      << D.getCXXScopeSpec().getRange();
    New->setInvalidDecl();
  }

  // Add the parameter declaration into this scope.
  S->AddDecl(New);
  if (II)
    IdResolver.AddDecl(New);

  ProcessDeclAttributes(New, D);
  return New;

}

void Sema::ActOnFinishKNRParamDeclarations(Scope *S, Declarator &D) {
  assert(D.getTypeObject(0).Kind == DeclaratorChunk::Function &&
         "Not a function declarator!");
  DeclaratorChunk::FunctionTypeInfo &FTI = D.getTypeObject(0).Fun;

  // Verify 6.9.1p6: 'every identifier in the identifier list shall be declared'
  // for a K&R function.
  if (!FTI.hasPrototype) {
    for (unsigned i = 0, e = FTI.NumArgs; i != e; ++i) {
      if (FTI.ArgInfo[i].Param == 0) {
        Diag(FTI.ArgInfo[i].IdentLoc, diag::ext_param_not_declared)
          << FTI.ArgInfo[i].Ident;
        // Implicitly declare the argument as type 'int' for lack of a better
        // type.
        DeclSpec DS;
        const char* PrevSpec; // unused
        DS.SetTypeSpecType(DeclSpec::TST_int, FTI.ArgInfo[i].IdentLoc, 
                           PrevSpec);
        Declarator ParamD(DS, Declarator::KNRTypeListContext);
        ParamD.SetIdentifier(FTI.ArgInfo[i].Ident, FTI.ArgInfo[i].IdentLoc);
        FTI.ArgInfo[i].Param = ActOnParamDeclarator(S, ParamD);
      }
    }
  } 
}

Sema::DeclTy *Sema::ActOnStartOfFunctionDef(Scope *FnBodyScope, Declarator &D) {
  assert(getCurFunctionDecl() == 0 && "Function parsing confused");
  assert(D.getTypeObject(0).Kind == DeclaratorChunk::Function &&
         "Not a function declarator!");
  DeclaratorChunk::FunctionTypeInfo &FTI = D.getTypeObject(0).Fun;

  if (FTI.hasPrototype) {
    // FIXME: Diagnose arguments without names in C. 
  }
  
  Scope *ParentScope = FnBodyScope->getParent();

  return ActOnStartOfFunctionDef(FnBodyScope,
                                 ActOnDeclarator(ParentScope, D, 0, 
                                                 /*IsFunctionDefinition=*/true));
}

Sema::DeclTy *Sema::ActOnStartOfFunctionDef(Scope *FnBodyScope, DeclTy *D) {
  Decl *decl = static_cast<Decl*>(D);
  FunctionDecl *FD = cast<FunctionDecl>(decl);

  // See if this is a redefinition.
  const FunctionDecl *Definition;
  if (FD->getBody(Definition)) {
    Diag(FD->getLocation(), diag::err_redefinition) << FD->getDeclName();
    Diag(Definition->getLocation(), diag::note_previous_definition);
  }

  PushDeclContext(FnBodyScope, FD);

  // Check the validity of our function parameters
  CheckParmsForFunctionDef(FD);

  // Introduce our parameters into the function scope
  for (unsigned p = 0, NumParams = FD->getNumParams(); p < NumParams; ++p) {
    ParmVarDecl *Param = FD->getParamDecl(p);
    Param->setOwningFunction(FD);

    // If this has an identifier, add it to the scope stack.
    if (Param->getIdentifier())
      PushOnScopeChains(Param, FnBodyScope);
  }

  // Checking attributes of current function definition
  // dllimport attribute.
  if (FD->getAttr<DLLImportAttr>() && (!FD->getAttr<DLLExportAttr>())) {
    // dllimport attribute cannot be applied to definition.
    if (!(FD->getAttr<DLLImportAttr>())->isInherited()) {
      Diag(FD->getLocation(),
           diag::err_attribute_can_be_applied_only_to_symbol_declaration)
        << "dllimport";
      FD->setInvalidDecl();
      return FD;
    } else {
      // If a symbol previously declared dllimport is later defined, the
      // attribute is ignored in subsequent references, and a warning is
      // emitted.
      Diag(FD->getLocation(),
           diag::warn_redeclaration_without_attribute_prev_attribute_ignored)
        << FD->getNameAsCString() << "dllimport";
    }
  }
  return FD;
}

Sema::DeclTy *Sema::ActOnFinishFunctionBody(DeclTy *D, StmtArg BodyArg) {
  Decl *dcl = static_cast<Decl *>(D);
  Stmt *Body = static_cast<Stmt*>(BodyArg.release());
  if (FunctionDecl *FD = dyn_cast_or_null<FunctionDecl>(dcl)) {
    FD->setBody(Body);
    assert(FD == getCurFunctionDecl() && "Function parsing confused");
  } else if (ObjCMethodDecl *MD = dyn_cast_or_null<ObjCMethodDecl>(dcl)) {
    MD->setBody((Stmt*)Body);
  } else
    return 0;
  PopDeclContext();
  // Verify and clean out per-function state.
  
  // Check goto/label use.
  for (llvm::DenseMap<IdentifierInfo*, LabelStmt*>::iterator
       I = LabelMap.begin(), E = LabelMap.end(); I != E; ++I) {
    // Verify that we have no forward references left.  If so, there was a goto
    // or address of a label taken, but no definition of it.  Label fwd
    // definitions are indicated with a null substmt.
    if (I->second->getSubStmt() == 0) {
      LabelStmt *L = I->second;
      // Emit error.
      Diag(L->getIdentLoc(), diag::err_undeclared_label_use) << L->getName();
      
      // At this point, we have gotos that use the bogus label.  Stitch it into
      // the function body so that they aren't leaked and that the AST is well
      // formed.
      if (Body) {
        L->setSubStmt(new NullStmt(L->getIdentLoc()));
        cast<CompoundStmt>(Body)->push_back(L);
      } else {
        // The whole function wasn't parsed correctly, just delete this.
        delete L;
      }
    }
  }
  LabelMap.clear();
  
  return D;
}

/// ImplicitlyDefineFunction - An undeclared identifier was used in a function
/// call, forming a call to an implicitly defined function (per C99 6.5.1p2).
NamedDecl *Sema::ImplicitlyDefineFunction(SourceLocation Loc, 
                                          IdentifierInfo &II, Scope *S) {
  // Extension in C99.  Legal in C90, but warn about it.
  if (getLangOptions().C99)
    Diag(Loc, diag::ext_implicit_function_decl) << &II;
  else
    Diag(Loc, diag::warn_implicit_function_decl) << &II;
  
  // FIXME: handle stuff like:
  // void foo() { extern float X(); }
  // void bar() { X(); }  <-- implicit decl for X in another scope.

  // Set a Declarator for the implicit definition: int foo();
  const char *Dummy;
  DeclSpec DS;
  bool Error = DS.SetTypeSpecType(DeclSpec::TST_int, Loc, Dummy);
  Error = Error; // Silence warning.
  assert(!Error && "Error setting up implicit decl!");
  Declarator D(DS, Declarator::BlockContext);
  D.AddTypeInfo(DeclaratorChunk::getFunction(false, false, 0, 0, 0, Loc, D));
  D.SetIdentifier(&II, Loc);
  
  // Insert this function into translation-unit scope.

  DeclContext *PrevDC = CurContext;
  CurContext = Context.getTranslationUnitDecl();
 
  FunctionDecl *FD = 
    dyn_cast<FunctionDecl>(static_cast<Decl*>(ActOnDeclarator(TUScope, D, 0)));
  FD->setImplicit();

  CurContext = PrevDC;

  return FD;
}


TypedefDecl *Sema::ParseTypedefDecl(Scope *S, Declarator &D, QualType T,
                                    Decl *LastDeclarator) {
  assert(D.getIdentifier() && "Wrong callback for declspec without declarator");
  assert(!T.isNull() && "GetTypeForDeclarator() returned null type");
  
  // Scope manipulation handled by caller.
  TypedefDecl *NewTD = TypedefDecl::Create(Context, CurContext,
                                           D.getIdentifierLoc(),
                                           D.getIdentifier(), 
                                           T);
  NewTD->setNextDeclarator(LastDeclarator);
  if (D.getInvalidType())
    NewTD->setInvalidDecl();
  return NewTD;
}

/// ActOnTag - This is invoked when we see 'struct foo' or 'struct {'.  In the
/// former case, Name will be non-null.  In the later case, Name will be null.
/// TagSpec indicates what kind of tag this is. TK indicates whether this is a
/// reference/declaration/definition of a tag.
Sema::DeclTy *Sema::ActOnTag(Scope *S, unsigned TagSpec, TagKind TK,
                             SourceLocation KWLoc, const CXXScopeSpec &SS,
                             IdentifierInfo *Name, SourceLocation NameLoc,
                             AttributeList *Attr,
                             MultiTemplateParamsArg TemplateParameterLists) {
  // If this is not a definition, it must have a name.
  assert((Name != 0 || TK == TK_Definition) &&
         "Nameless record must be a definition!");
  
  TagDecl::TagKind Kind;
  switch (TagSpec) {
  default: assert(0 && "Unknown tag type!");
  case DeclSpec::TST_struct: Kind = TagDecl::TK_struct; break;
  case DeclSpec::TST_union:  Kind = TagDecl::TK_union; break;
  case DeclSpec::TST_class:  Kind = TagDecl::TK_class; break;
  case DeclSpec::TST_enum:   Kind = TagDecl::TK_enum; break;
  }
  
  DeclContext *SearchDC = CurContext;
  DeclContext *DC = CurContext;
  DeclContext *LexicalContext = CurContext;
  Decl *PrevDecl = 0;

  bool Invalid = false;

  if (Name && SS.isNotEmpty()) {
    // We have a nested-name tag ('struct foo::bar').

    // Check for invalid 'foo::'.
    if (SS.isInvalid()) {
      Name = 0;
      goto CreateNewDecl;
    }

    DC = static_cast<DeclContext*>(SS.getScopeRep());
    // Look-up name inside 'foo::'.
    PrevDecl = dyn_cast_or_null<TagDecl>(LookupDecl(Name, Decl::IDNS_Tag,S,DC)
                                           .getAsDecl());

    // A tag 'foo::bar' must already exist.
    if (PrevDecl == 0) {
      Diag(NameLoc, diag::err_not_tag_in_scope) << Name << SS.getRange();
      Name = 0;
      goto CreateNewDecl;
    }
  } else if (Name) {
    // If this is a named struct, check to see if there was a previous forward
    // declaration or definition.
    PrevDecl = dyn_cast_or_null<NamedDecl>(LookupDecl(Name, Decl::IDNS_Tag,S)
                                           .getAsDecl());

    if (!getLangOptions().CPlusPlus && TK != TK_Reference) {
      // FIXME: This makes sure that we ignore the contexts associated
      // with C structs, unions, and enums when looking for a matching
      // tag declaration or definition. See the similar lookup tweak
      // in Sema::LookupDecl; is there a better way to deal with this?
      while (isa<RecordDecl>(SearchDC) || isa<EnumDecl>(SearchDC))
        SearchDC = SearchDC->getParent();
    }
  }

  if (PrevDecl && PrevDecl->isTemplateParameter()) {
    // Maybe we will complain about the shadowed template parameter.
    DiagnoseTemplateParameterShadow(NameLoc, PrevDecl);
    // Just pretend that we didn't see the previous declaration.
    PrevDecl = 0;
  }

  if (PrevDecl) {    
    assert((isa<TagDecl>(PrevDecl) || isa<NamespaceDecl>(PrevDecl)) &&
            "unexpected Decl type");
    if (TagDecl *PrevTagDecl = dyn_cast<TagDecl>(PrevDecl)) {
      // If this is a use of a previous tag, or if the tag is already declared
      // in the same scope (so that the definition/declaration completes or
      // rementions the tag), reuse the decl.
      if (TK == TK_Reference || isDeclInScope(PrevDecl, SearchDC, S)) {
        // Make sure that this wasn't declared as an enum and now used as a
        // struct or something similar.
        if (PrevTagDecl->getTagKind() != Kind) {
          Diag(KWLoc, diag::err_use_with_wrong_tag) << Name;
          Diag(PrevDecl->getLocation(), diag::note_previous_use);
          // Recover by making this an anonymous redefinition.
          Name = 0;
          PrevDecl = 0;
          Invalid = true;
        } else {
          // If this is a use, just return the declaration we found.

          // FIXME: In the future, return a variant or some other clue
          // for the consumer of this Decl to know it doesn't own it.
          // For our current ASTs this shouldn't be a problem, but will
          // need to be changed with DeclGroups.
          if (TK == TK_Reference)
            return PrevDecl;
          
          // Diagnose attempts to redefine a tag.
          if (TK == TK_Definition) {
            if (TagDecl *Def = PrevTagDecl->getDefinition(Context)) {
              Diag(NameLoc, diag::err_redefinition) << Name;
              Diag(Def->getLocation(), diag::note_previous_definition);
              // If this is a redefinition, recover by making this
              // struct be anonymous, which will make any later
              // references get the previous definition.
              Name = 0;
              PrevDecl = 0;
              Invalid = true;
            } else {
              // If the type is currently being defined, complain
              // about a nested redefinition.
              TagType *Tag = cast<TagType>(Context.getTagDeclType(PrevTagDecl));
              if (Tag->isBeingDefined()) {
                Diag(NameLoc, diag::err_nested_redefinition) << Name;
                Diag(PrevTagDecl->getLocation(), 
                     diag::note_previous_definition);
                Name = 0;
                PrevDecl = 0;
                Invalid = true;
              }
            }

            // Okay, this is definition of a previously declared or referenced
            // tag PrevDecl. We're going to create a new Decl for it.
          }
        }
        // If we get here we have (another) forward declaration or we
        // have a definition.  Just create a new decl.        
      } else {
        // If we get here, this is a definition of a new tag type in a nested
        // scope, e.g. "struct foo; void bar() { struct foo; }", just create a 
        // new decl/type.  We set PrevDecl to NULL so that the entities
        // have distinct types.
        PrevDecl = 0;
      }
      // If we get here, we're going to create a new Decl. If PrevDecl
      // is non-NULL, it's a definition of the tag declared by
      // PrevDecl. If it's NULL, we have a new definition.
    } else {
      // PrevDecl is a namespace.
      if (isDeclInScope(PrevDecl, SearchDC, S)) {
        // The tag name clashes with a namespace name, issue an error and
        // recover by making this tag be anonymous.
        Diag(NameLoc, diag::err_redefinition_different_kind) << Name;
        Diag(PrevDecl->getLocation(), diag::note_previous_definition);
        Name = 0;
        PrevDecl = 0;
        Invalid = true;
      } else {
        // The existing declaration isn't relevant to us; we're in a
        // new scope, so clear out the previous declaration.
        PrevDecl = 0;
      }
    }
  } else if (TK == TK_Reference && SS.isEmpty() && Name &&
             (Kind != TagDecl::TK_enum))  {
    // C++ [basic.scope.pdecl]p5:
    //   -- for an elaborated-type-specifier of the form 
    //
    //          class-key identifier
    //
    //      if the elaborated-type-specifier is used in the
    //      decl-specifier-seq or parameter-declaration-clause of a
    //      function defined in namespace scope, the identifier is
    //      declared as a class-name in the namespace that contains
    //      the declaration; otherwise, except as a friend
    //      declaration, the identifier is declared in the smallest
    //      non-class, non-function-prototype scope that contains the
    //      declaration.
    //
    // C99 6.7.2.3p8 has a similar (but not identical!) provision for
    // C structs and unions.

    // Find the context where we'll be declaring the tag.
    // FIXME: We would like to maintain the current DeclContext as the
    // lexical context, 
    while (DC->isRecord())
      DC = DC->getParent();
    LexicalContext = DC;

    // Find the scope where we'll be declaring the tag.
    while (S->isClassScope() || 
           (getLangOptions().CPlusPlus && S->isFunctionPrototypeScope()) ||
           ((S->getFlags() & Scope::DeclScope) == 0) ||
           (S->getEntity() && 
            ((DeclContext *)S->getEntity())->isTransparentContext()))
      S = S->getParent();
  }

CreateNewDecl:
  
  // If there is an identifier, use the location of the identifier as the
  // location of the decl, otherwise use the location of the struct/union
  // keyword.
  SourceLocation Loc = NameLoc.isValid() ? NameLoc : KWLoc;
  
  // Otherwise, create a new declaration. If there is a previous
  // declaration of the same entity, the two will be linked via
  // PrevDecl.
  TagDecl *New;

  if (Kind == TagDecl::TK_enum) {
    // FIXME: Tag decls should be chained to any simultaneous vardecls, e.g.:
    // enum X { A, B, C } D;    D should chain to X.
    New = EnumDecl::Create(Context, DC, Loc, Name, 
                           cast_or_null<EnumDecl>(PrevDecl));
    // If this is an undefined enum, warn.
    if (TK != TK_Definition) Diag(Loc, diag::ext_forward_ref_enum);
  } else {
    // struct/union/class

    // FIXME: Tag decls should be chained to any simultaneous vardecls, e.g.:
    // struct X { int A; } D;    D should chain to X.
    if (getLangOptions().CPlusPlus)
      // FIXME: Look for a way to use RecordDecl for simple structs.
      New = CXXRecordDecl::Create(Context, Kind, DC, Loc, Name,
                                  cast_or_null<CXXRecordDecl>(PrevDecl));
    else
      New = RecordDecl::Create(Context, Kind, DC, Loc, Name,
                               cast_or_null<RecordDecl>(PrevDecl));
  }

  if (Kind != TagDecl::TK_enum) {
    // Handle #pragma pack: if the #pragma pack stack has non-default
    // alignment, make up a packed attribute for this decl. These
    // attributes are checked when the ASTContext lays out the
    // structure.
    //
    // It is important for implementing the correct semantics that this
    // happen here (in act on tag decl). The #pragma pack stack is
    // maintained as a result of parser callbacks which can occur at
    // many points during the parsing of a struct declaration (because
    // the #pragma tokens are effectively skipped over during the
    // parsing of the struct).
    if (unsigned Alignment = PackContext.getAlignment())
      New->addAttr(new PackedAttr(Alignment * 8));
  }

  if (Invalid)
    New->setInvalidDecl();

  if (Attr)
    ProcessDeclAttributeList(New, Attr);

  // If we're declaring or defining a tag in function prototype scope
  // in C, note that this type can only be used within the function.
  if (Name && S->isFunctionPrototypeScope() && !getLangOptions().CPlusPlus)
    Diag(Loc, diag::warn_decl_in_param_list) << Context.getTagDeclType(New);

  // Set the lexical context. If the tag has a C++ scope specifier, the
  // lexical context will be different from the semantic context.
  New->setLexicalDeclContext(LexicalContext);

  if (TK == TK_Definition)
    New->startDefinition();
  
  // If this has an identifier, add it to the scope stack.
  if (Name) {
    S = getNonFieldDeclScope(S);
    
    // Add it to the decl chain.
    if (LexicalContext != CurContext) {
      // FIXME: PushOnScopeChains should not rely on CurContext!
      DeclContext *OldContext = CurContext;
      CurContext = LexicalContext;
      PushOnScopeChains(New, S);
      CurContext = OldContext;
    } else 
      PushOnScopeChains(New, S);
  } else {
    LexicalContext->addDecl(New);
  }

  return New;
}

void Sema::ActOnTagStartDefinition(Scope *S, DeclTy *TagD) {
  TagDecl *Tag = cast<TagDecl>((Decl *)TagD);

  // Enter the tag context.
  PushDeclContext(S, Tag);

  if (CXXRecordDecl *Record = dyn_cast<CXXRecordDecl>(Tag)) {
    FieldCollector->StartClass();

    if (Record->getIdentifier()) {
      // C++ [class]p2: 
      //   [...] The class-name is also inserted into the scope of the
      //   class itself; this is known as the injected-class-name. For
      //   purposes of access checking, the injected-class-name is treated
      //   as if it were a public member name.
      RecordDecl *InjectedClassName
        = CXXRecordDecl::Create(Context, Record->getTagKind(),
                                CurContext, Record->getLocation(),
                                Record->getIdentifier(), Record);
      InjectedClassName->setImplicit();
      PushOnScopeChains(InjectedClassName, S);
    }
  }
}

void Sema::ActOnTagFinishDefinition(Scope *S, DeclTy *TagD) {
  TagDecl *Tag = cast<TagDecl>((Decl *)TagD);

  if (isa<CXXRecordDecl>(Tag))
    FieldCollector->FinishClass();

  // Exit this scope of this tag's definition.
  PopDeclContext();

  // Notify the consumer that we've defined a tag.
  Consumer.HandleTagDeclDefinition(Tag);
}

/// TryToFixInvalidVariablyModifiedType - Helper method to turn variable array
/// types into constant array types in certain situations which would otherwise
/// be errors (for GCC compatibility).
static QualType TryToFixInvalidVariablyModifiedType(QualType T,
                                                    ASTContext &Context) {
  // This method tries to turn a variable array into a constant
  // array even when the size isn't an ICE.  This is necessary
  // for compatibility with code that depends on gcc's buggy
  // constant expression folding, like struct {char x[(int)(char*)2];}
  const VariableArrayType* VLATy = dyn_cast<VariableArrayType>(T);
  if (!VLATy) return QualType();
  
  Expr::EvalResult EvalResult;
  if (!VLATy->getSizeExpr() ||
      !VLATy->getSizeExpr()->Evaluate(EvalResult, Context))
    return QualType();
    
  assert(EvalResult.Val.isInt() && "Size expressions must be integers!");
  llvm::APSInt &Res = EvalResult.Val.getInt();
  if (Res > llvm::APSInt(Res.getBitWidth(), Res.isUnsigned()))
    return Context.getConstantArrayType(VLATy->getElementType(),
                                        Res, ArrayType::Normal, 0);
  return QualType();
}

bool Sema::VerifyBitField(SourceLocation FieldLoc, IdentifierInfo *FieldName, 
                          QualType FieldTy, const Expr *BitWidth) {
  // FIXME: 6.7.2.1p4 - verify the field type.
  
  llvm::APSInt Value;
  if (VerifyIntegerConstantExpression(BitWidth, &Value))
    return true;

  // Zero-width bitfield is ok for anonymous field.
  if (Value == 0 && FieldName)
    return Diag(FieldLoc, diag::err_bitfield_has_zero_width) << FieldName;
  
  if (Value.isNegative())
    return Diag(FieldLoc, diag::err_bitfield_has_negative_width) << FieldName;

  uint64_t TypeSize = Context.getTypeSize(FieldTy);
  // FIXME: We won't need the 0 size once we check that the field type is valid.
  if (TypeSize && Value.getZExtValue() > TypeSize)
    return Diag(FieldLoc, diag::err_bitfield_width_exceeds_type_size)
       << FieldName << (unsigned)TypeSize;

  return false;
}

/// ActOnField - Each field of a struct/union/class is passed into this in order
/// to create a FieldDecl object for it.
Sema::DeclTy *Sema::ActOnField(Scope *S, DeclTy *TagD,
                               SourceLocation DeclStart, 
                               Declarator &D, ExprTy *BitfieldWidth) {
  IdentifierInfo *II = D.getIdentifier();
  Expr *BitWidth = (Expr*)BitfieldWidth;
  SourceLocation Loc = DeclStart;
  RecordDecl *Record = (RecordDecl *)TagD;
  if (II) Loc = D.getIdentifierLoc();
  
  // FIXME: Unnamed fields can be handled in various different ways, for
  // example, unnamed unions inject all members into the struct namespace!

  QualType T = GetTypeForDeclarator(D, S);
  assert(!T.isNull() && "GetTypeForDeclarator() returned null type");
  bool InvalidDecl = false;

  // C99 6.7.2.1p8: A member of a structure or union may have any type other
  // than a variably modified type.
  if (T->isVariablyModifiedType()) {
    QualType FixedTy = TryToFixInvalidVariablyModifiedType(T, Context);
    if (!FixedTy.isNull()) {
      Diag(Loc, diag::warn_illegal_constant_array_size);
      T = FixedTy;
    } else {
      Diag(Loc, diag::err_typecheck_field_variable_size);
      T = Context.IntTy;
      InvalidDecl = true;
    }
  }
  
  if (BitWidth) {
    if (VerifyBitField(Loc, II, T, BitWidth))
      InvalidDecl = true;
  } else {
    // Not a bitfield.

    // validate II.
    
  }
  
  // FIXME: Chain fielddecls together.
  FieldDecl *NewFD;

  NewFD = FieldDecl::Create(Context, Record,
                            Loc, II, T, BitWidth,
                            D.getDeclSpec().getStorageClassSpec() ==
                              DeclSpec::SCS_mutable);

  if (II) {
    Decl *PrevDecl 
      = LookupDecl(II, Decl::IDNS_Member, S, 0, false, false, false);
    if (PrevDecl && isDeclInScope(PrevDecl, CurContext, S)
        && !isa<TagDecl>(PrevDecl)) {
      Diag(Loc, diag::err_duplicate_member) << II;
      Diag(PrevDecl->getLocation(), diag::note_previous_declaration);
      NewFD->setInvalidDecl();
      Record->setInvalidDecl();
    }
  }

  if (getLangOptions().CPlusPlus) {
    CheckExtraCXXDefaultArguments(D);
    if (!T->isPODType())
      cast<CXXRecordDecl>(Record)->setPOD(false);
  }

  ProcessDeclAttributes(NewFD, D);

  if (D.getInvalidType() || InvalidDecl)
    NewFD->setInvalidDecl();

  if (II) {
    PushOnScopeChains(NewFD, S);
  } else
    Record->addDecl(NewFD);

  return NewFD;
}

/// TranslateIvarVisibility - Translate visibility from a token ID to an 
///  AST enum value.
static ObjCIvarDecl::AccessControl
TranslateIvarVisibility(tok::ObjCKeywordKind ivarVisibility) {
  switch (ivarVisibility) {
  default: assert(0 && "Unknown visitibility kind");
  case tok::objc_private: return ObjCIvarDecl::Private;
  case tok::objc_public: return ObjCIvarDecl::Public;
  case tok::objc_protected: return ObjCIvarDecl::Protected;
  case tok::objc_package: return ObjCIvarDecl::Package;
  }
}

/// ActOnIvar - Each ivar field of an objective-c class is passed into this 
/// in order to create an IvarDecl object for it.
Sema::DeclTy *Sema::ActOnIvar(Scope *S,
                              SourceLocation DeclStart, 
                              Declarator &D, ExprTy *BitfieldWidth,
                              tok::ObjCKeywordKind Visibility) {
  
  IdentifierInfo *II = D.getIdentifier();
  Expr *BitWidth = (Expr*)BitfieldWidth;
  SourceLocation Loc = DeclStart;
  if (II) Loc = D.getIdentifierLoc();
  
  // FIXME: Unnamed fields can be handled in various different ways, for
  // example, unnamed unions inject all members into the struct namespace!
  
  QualType T = GetTypeForDeclarator(D, S);
  assert(!T.isNull() && "GetTypeForDeclarator() returned null type");
  bool InvalidDecl = false;
  
  if (BitWidth) {
    // TODO: Validate.
    //printf("WARNING: BITFIELDS IGNORED!\n");
    
    // 6.7.2.1p3
    // 6.7.2.1p4
    
  } else {
    // Not a bitfield.
    
    // validate II.
    
  }
  
  // C99 6.7.2.1p8: A member of a structure or union may have any type other
  // than a variably modified type.
  if (T->isVariablyModifiedType()) {
    Diag(Loc, diag::err_typecheck_ivar_variable_size);
    InvalidDecl = true;
  }
  
  // Get the visibility (access control) for this ivar.
  ObjCIvarDecl::AccessControl ac = 
    Visibility != tok::objc_not_keyword ? TranslateIvarVisibility(Visibility)
                                        : ObjCIvarDecl::None;

  // Construct the decl.
  ObjCIvarDecl *NewID = ObjCIvarDecl::Create(Context, Loc, II, T, ac,                                             
                                             (Expr *)BitfieldWidth);
  
  if (II) {
    Decl *PrevDecl 
      = LookupDecl(II, Decl::IDNS_Member, S, 0, false, false, false);
    if (PrevDecl && isDeclInScope(PrevDecl, CurContext, S)
        && !isa<TagDecl>(PrevDecl)) {
      Diag(Loc, diag::err_duplicate_member) << II;
      Diag(PrevDecl->getLocation(), diag::note_previous_declaration);
      NewID->setInvalidDecl();
    }
  }

  // Process attributes attached to the ivar.
  ProcessDeclAttributes(NewID, D);
  
  if (D.getInvalidType() || InvalidDecl)
    NewID->setInvalidDecl();

  if (II) {
    // FIXME: When interfaces are DeclContexts, we'll need to add
    // these to the interface.
    S->AddDecl(NewID);
    IdResolver.AddDecl(NewID);
  }

  return NewID;
}

void Sema::ActOnFields(Scope* S,
                       SourceLocation RecLoc, DeclTy *RecDecl,
                       DeclTy **Fields, unsigned NumFields,
                       SourceLocation LBrac, SourceLocation RBrac,
                       AttributeList *Attr) {
  Decl *EnclosingDecl = static_cast<Decl*>(RecDecl);
  assert(EnclosingDecl && "missing record or interface decl");
  RecordDecl *Record = dyn_cast<RecordDecl>(EnclosingDecl);
  
  // Verify that all the fields are okay.
  unsigned NumNamedMembers = 0;
  llvm::SmallVector<FieldDecl*, 32> RecFields;

  for (unsigned i = 0; i != NumFields; ++i) {
    FieldDecl *FD = cast_or_null<FieldDecl>(static_cast<Decl*>(Fields[i]));
    assert(FD && "missing field decl");
    
    // Get the type for the field.
    Type *FDTy = FD->getType().getTypePtr();

    if (!FD->isAnonymousStructOrUnion()) {
      // Remember all fields written by the user.
      RecFields.push_back(FD);
    }
      
    // C99 6.7.2.1p2 - A field may not be a function type.
    if (FDTy->isFunctionType()) {
      Diag(FD->getLocation(), diag::err_field_declared_as_function)
        << FD->getDeclName();
      FD->setInvalidDecl();
      EnclosingDecl->setInvalidDecl();
      continue;
    }
    // C99 6.7.2.1p2 - A field may not be an incomplete type except...
    if (FDTy->isIncompleteType()) {
      if (!Record) {  // Incomplete ivar type is always an error.
        DiagnoseIncompleteType(FD->getLocation(), FD->getType(), 
                               diag::err_field_incomplete);
        FD->setInvalidDecl();
        EnclosingDecl->setInvalidDecl();
        continue;
      }
      if (i != NumFields-1 ||                   // ... that the last member ...
          !Record->isStruct() ||  // ... of a structure ...
          !FDTy->isArrayType()) {         //... may have incomplete array type.
        DiagnoseIncompleteType(FD->getLocation(), FD->getType(), 
                               diag::err_field_incomplete);
        FD->setInvalidDecl();
        EnclosingDecl->setInvalidDecl();
        continue;
      }
      if (NumNamedMembers < 1) {  //... must have more than named member ...
        Diag(FD->getLocation(), diag::err_flexible_array_empty_struct)
          << FD->getDeclName();
        FD->setInvalidDecl();
        EnclosingDecl->setInvalidDecl();
        continue;
      }
      // Okay, we have a legal flexible array member at the end of the struct.
      if (Record)
        Record->setHasFlexibleArrayMember(true);
    }
    /// C99 6.7.2.1p2 - a struct ending in a flexible array member cannot be the
    /// field of another structure or the element of an array.
    if (const RecordType *FDTTy = FDTy->getAsRecordType()) {
      if (FDTTy->getDecl()->hasFlexibleArrayMember()) {
        // If this is a member of a union, then entire union becomes "flexible".
        if (Record && Record->isUnion()) {
          Record->setHasFlexibleArrayMember(true);
        } else {
          // If this is a struct/class and this is not the last element, reject
          // it.  Note that GCC supports variable sized arrays in the middle of
          // structures.
          if (i != NumFields-1) {
            Diag(FD->getLocation(), diag::err_variable_sized_type_in_struct)
              << FD->getDeclName();
            FD->setInvalidDecl();
            EnclosingDecl->setInvalidDecl();
            continue;
          }
          // We support flexible arrays at the end of structs in other structs
          // as an extension.
          Diag(FD->getLocation(), diag::ext_flexible_array_in_struct)
            << FD->getDeclName();
          if (Record)
            Record->setHasFlexibleArrayMember(true);
        }
      }
    }
    /// A field cannot be an Objective-c object
    if (FDTy->isObjCInterfaceType()) {
      Diag(FD->getLocation(), diag::err_statically_allocated_object)
        << FD->getDeclName();
      FD->setInvalidDecl();
      EnclosingDecl->setInvalidDecl();
      continue;
    }
    // Keep track of the number of named members.
    if (FD->getIdentifier())
      ++NumNamedMembers;
  }

  // Okay, we successfully defined 'Record'.
  if (Record) {
    Record->completeDefinition(Context);
  } else {
    ObjCIvarDecl **ClsFields = reinterpret_cast<ObjCIvarDecl**>(&RecFields[0]);
    if (ObjCInterfaceDecl *ID = dyn_cast<ObjCInterfaceDecl>(EnclosingDecl)) {
      ID->addInstanceVariablesToClass(ClsFields, RecFields.size(), RBrac);
      // Must enforce the rule that ivars in the base classes may not be
      // duplicates.
      if (ID->getSuperClass()) {
        for (ObjCInterfaceDecl::ivar_iterator IVI = ID->ivar_begin(), 
             IVE = ID->ivar_end(); IVI != IVE; ++IVI) {
          ObjCIvarDecl* Ivar = (*IVI);
          IdentifierInfo *II = Ivar->getIdentifier();
          ObjCIvarDecl* prevIvar = ID->getSuperClass()->FindIvarDeclaration(II);
          if (prevIvar) {
            Diag(Ivar->getLocation(), diag::err_duplicate_member) << II;
            Diag(prevIvar->getLocation(), diag::note_previous_declaration);
          }
        }
      }
    }
    else if (ObjCImplementationDecl *IMPDecl = 
               dyn_cast<ObjCImplementationDecl>(EnclosingDecl)) {
      assert(IMPDecl && "ActOnFields - missing ObjCImplementationDecl");
      IMPDecl->ObjCAddInstanceVariablesToClassImpl(ClsFields, RecFields.size());
      CheckImplementationIvars(IMPDecl, ClsFields, RecFields.size(), RBrac);
    }
  }

  if (Attr)
    ProcessDeclAttributeList(Record, Attr);
}

Sema::DeclTy *Sema::ActOnEnumConstant(Scope *S, DeclTy *theEnumDecl,
                                      DeclTy *lastEnumConst,
                                      SourceLocation IdLoc, IdentifierInfo *Id,
                                      SourceLocation EqualLoc, ExprTy *val) {
  EnumDecl *TheEnumDecl = cast<EnumDecl>(static_cast<Decl*>(theEnumDecl));
  EnumConstantDecl *LastEnumConst =
    cast_or_null<EnumConstantDecl>(static_cast<Decl*>(lastEnumConst));
  Expr *Val = static_cast<Expr*>(val);

  // The scope passed in may not be a decl scope.  Zip up the scope tree until
  // we find one that is.
  S = getNonFieldDeclScope(S);
  
  // Verify that there isn't already something declared with this name in this
  // scope.
  Decl *PrevDecl = LookupDecl(Id, Decl::IDNS_Ordinary, S);
  if (PrevDecl && PrevDecl->isTemplateParameter()) {
    // Maybe we will complain about the shadowed template parameter.
    DiagnoseTemplateParameterShadow(IdLoc, PrevDecl);
    // Just pretend that we didn't see the previous declaration.
    PrevDecl = 0;
  }

  if (PrevDecl) {
    // When in C++, we may get a TagDecl with the same name; in this case the
    // enum constant will 'hide' the tag.
    assert((getLangOptions().CPlusPlus || !isa<TagDecl>(PrevDecl)) &&
           "Received TagDecl when not in C++!");
    if (!isa<TagDecl>(PrevDecl) && isDeclInScope(PrevDecl, CurContext, S)) {
      if (isa<EnumConstantDecl>(PrevDecl))
        Diag(IdLoc, diag::err_redefinition_of_enumerator) << Id;
      else
        Diag(IdLoc, diag::err_redefinition) << Id;
      Diag(PrevDecl->getLocation(), diag::note_previous_definition);
      delete Val;
      return 0;
    }
  }

  llvm::APSInt EnumVal(32);
  QualType EltTy;
  if (Val) {
    // Make sure to promote the operand type to int.
    UsualUnaryConversions(Val);
    
    // C99 6.7.2.2p2: Make sure we have an integer constant expression.
    SourceLocation ExpLoc;
    if (VerifyIntegerConstantExpression(Val, &EnumVal)) {
      delete Val;
      Val = 0;  // Just forget about it.
    } else {
      EltTy = Val->getType();
    }
  }
  
  if (!Val) {
    if (LastEnumConst) {
      // Assign the last value + 1.
      EnumVal = LastEnumConst->getInitVal();
      ++EnumVal;

      // Check for overflow on increment.
      if (EnumVal < LastEnumConst->getInitVal())
        Diag(IdLoc, diag::warn_enum_value_overflow);
      
      EltTy = LastEnumConst->getType();
    } else {
      // First value, set to zero.
      EltTy = Context.IntTy;
      EnumVal.zextOrTrunc(static_cast<uint32_t>(Context.getTypeSize(EltTy)));
    }
  }
  
  EnumConstantDecl *New = 
    EnumConstantDecl::Create(Context, TheEnumDecl, IdLoc, Id, EltTy,
                             Val, EnumVal);
  
  // Register this decl in the current scope stack.
  PushOnScopeChains(New, S);

  return New;
}

// FIXME: For consistency with ActOnFields(), we should have the parser
// pass in the source location for the left/right braces.
void Sema::ActOnEnumBody(SourceLocation EnumLoc, DeclTy *EnumDeclX,
                         DeclTy **Elements, unsigned NumElements) {
  EnumDecl *Enum = cast<EnumDecl>(static_cast<Decl*>(EnumDeclX));
  QualType EnumType = Context.getTypeDeclType(Enum);
  
  // TODO: If the result value doesn't fit in an int, it must be a long or long
  // long value.  ISO C does not support this, but GCC does as an extension,
  // emit a warning.
  unsigned IntWidth = Context.Target.getIntWidth();
  
  // Verify that all the values are okay, compute the size of the values, and
  // reverse the list.
  unsigned NumNegativeBits = 0;
  unsigned NumPositiveBits = 0;
  
  // Keep track of whether all elements have type int.
  bool AllElementsInt = true;
  
  for (unsigned i = 0; i != NumElements; ++i) {
    EnumConstantDecl *ECD =
      cast_or_null<EnumConstantDecl>(static_cast<Decl*>(Elements[i]));
    if (!ECD) continue;  // Already issued a diagnostic.
    
    // If the enum value doesn't fit in an int, emit an extension warning.
    const llvm::APSInt &InitVal = ECD->getInitVal();
    assert(InitVal.getBitWidth() >= IntWidth &&
           "Should have promoted value to int");
    if (InitVal.getBitWidth() > IntWidth) {
      llvm::APSInt V(InitVal);
      V.trunc(IntWidth);
      V.extend(InitVal.getBitWidth());
      if (V != InitVal)
        Diag(ECD->getLocation(), diag::ext_enum_value_not_int)
          << InitVal.toString(10);
    }
    
    // Keep track of the size of positive and negative values.
    if (InitVal.isUnsigned() || InitVal.isNonNegative())
      NumPositiveBits = std::max(NumPositiveBits,
                                 (unsigned)InitVal.getActiveBits());
    else
      NumNegativeBits = std::max(NumNegativeBits,
                                 (unsigned)InitVal.getMinSignedBits());

    // Keep track of whether every enum element has type int (very commmon).
    if (AllElementsInt)
      AllElementsInt = ECD->getType() == Context.IntTy; 
  }
  
  // Figure out the type that should be used for this enum.
  // FIXME: Support attribute(packed) on enums and -fshort-enums.
  QualType BestType;
  unsigned BestWidth;
  
  if (NumNegativeBits) {
    // If there is a negative value, figure out the smallest integer type (of 
    // int/long/longlong) that fits.
    if (NumNegativeBits <= IntWidth && NumPositiveBits < IntWidth) {
      BestType = Context.IntTy;
      BestWidth = IntWidth;
    } else {
      BestWidth = Context.Target.getLongWidth();
      
      if (NumNegativeBits <= BestWidth && NumPositiveBits < BestWidth)
        BestType = Context.LongTy;
      else {
        BestWidth = Context.Target.getLongLongWidth();
        
        if (NumNegativeBits > BestWidth || NumPositiveBits >= BestWidth)
          Diag(Enum->getLocation(), diag::warn_enum_too_large);
        BestType = Context.LongLongTy;
      }
    }
  } else {
    // If there is no negative value, figure out which of uint, ulong, ulonglong
    // fits.
    if (NumPositiveBits <= IntWidth) {
      BestType = Context.UnsignedIntTy;
      BestWidth = IntWidth;
    } else if (NumPositiveBits <=
               (BestWidth = Context.Target.getLongWidth())) {
      BestType = Context.UnsignedLongTy;
    } else {
      BestWidth = Context.Target.getLongLongWidth();
      assert(NumPositiveBits <= BestWidth &&
             "How could an initializer get larger than ULL?");
      BestType = Context.UnsignedLongLongTy;
    }
  }
  
  // Loop over all of the enumerator constants, changing their types to match
  // the type of the enum if needed.
  for (unsigned i = 0; i != NumElements; ++i) {
    EnumConstantDecl *ECD =
      cast_or_null<EnumConstantDecl>(static_cast<Decl*>(Elements[i]));
    if (!ECD) continue;  // Already issued a diagnostic.

    // Standard C says the enumerators have int type, but we allow, as an
    // extension, the enumerators to be larger than int size.  If each
    // enumerator value fits in an int, type it as an int, otherwise type it the
    // same as the enumerator decl itself.  This means that in "enum { X = 1U }"
    // that X has type 'int', not 'unsigned'.
    if (ECD->getType() == Context.IntTy) {
      // Make sure the init value is signed.
      llvm::APSInt IV = ECD->getInitVal();
      IV.setIsSigned(true);
      ECD->setInitVal(IV);

      if (getLangOptions().CPlusPlus)
        // C++ [dcl.enum]p4: Following the closing brace of an
        // enum-specifier, each enumerator has the type of its
        // enumeration. 
        ECD->setType(EnumType);
      continue;  // Already int type.
    }

    // Determine whether the value fits into an int.
    llvm::APSInt InitVal = ECD->getInitVal();
    bool FitsInInt;
    if (InitVal.isUnsigned() || !InitVal.isNegative())
      FitsInInt = InitVal.getActiveBits() < IntWidth;
    else
      FitsInInt = InitVal.getMinSignedBits() <= IntWidth;

    // If it fits into an integer type, force it.  Otherwise force it to match
    // the enum decl type.
    QualType NewTy;
    unsigned NewWidth;
    bool NewSign;
    if (FitsInInt) {
      NewTy = Context.IntTy;
      NewWidth = IntWidth;
      NewSign = true;
    } else if (ECD->getType() == BestType) {
      // Already the right type!
      if (getLangOptions().CPlusPlus)
        // C++ [dcl.enum]p4: Following the closing brace of an
        // enum-specifier, each enumerator has the type of its
        // enumeration. 
        ECD->setType(EnumType);
      continue;
    } else {
      NewTy = BestType;
      NewWidth = BestWidth;
      NewSign = BestType->isSignedIntegerType();
    }

    // Adjust the APSInt value.
    InitVal.extOrTrunc(NewWidth);
    InitVal.setIsSigned(NewSign);
    ECD->setInitVal(InitVal);
    
    // Adjust the Expr initializer and type.
    if (ECD->getInitExpr())
      ECD->setInitExpr(new ImplicitCastExpr(NewTy, ECD->getInitExpr(), 
                                            /*isLvalue=*/false));
    if (getLangOptions().CPlusPlus)
      // C++ [dcl.enum]p4: Following the closing brace of an
      // enum-specifier, each enumerator has the type of its
      // enumeration. 
      ECD->setType(EnumType);
    else
      ECD->setType(NewTy);
  }
  
  Enum->completeDefinition(Context, BestType);
}

Sema::DeclTy *Sema::ActOnFileScopeAsmDecl(SourceLocation Loc,
                                          ExprArg expr) {
  StringLiteral *AsmString = cast<StringLiteral>((Expr*)expr.release());

  return FileScopeAsmDecl::Create(Context, CurContext, Loc, AsmString);
}


void Sema::ActOnPragmaPack(PragmaPackKind Kind, IdentifierInfo *Name, 
                           ExprTy *alignment, SourceLocation PragmaLoc, 
                           SourceLocation LParenLoc, SourceLocation RParenLoc) {
  Expr *Alignment = static_cast<Expr *>(alignment);

  // If specified then alignment must be a "small" power of two.
  unsigned AlignmentVal = 0;
  if (Alignment) {
    llvm::APSInt Val;
    if (!Alignment->isIntegerConstantExpr(Val, Context) ||
        !Val.isPowerOf2() ||
        Val.getZExtValue() > 16) {
      Diag(PragmaLoc, diag::warn_pragma_pack_invalid_alignment);
      delete Alignment;
      return; // Ignore
    }

    AlignmentVal = (unsigned) Val.getZExtValue();
  }

  switch (Kind) {
  case Action::PPK_Default: // pack([n])
    PackContext.setAlignment(AlignmentVal);
    break;

  case Action::PPK_Show: // pack(show)
    // Show the current alignment, making sure to show the right value
    // for the default.
    AlignmentVal = PackContext.getAlignment();
    // FIXME: This should come from the target.
    if (AlignmentVal == 0)
      AlignmentVal = 8;
    Diag(PragmaLoc, diag::warn_pragma_pack_show) << AlignmentVal;
    break;

  case Action::PPK_Push: // pack(push [, id] [, [n])
    PackContext.push(Name);
    // Set the new alignment if specified.
    if (Alignment)
      PackContext.setAlignment(AlignmentVal);    
    break;

  case Action::PPK_Pop: // pack(pop [, id] [,  n])
    // MSDN, C/C++ Preprocessor Reference > Pragma Directives > pack:
    // "#pragma pack(pop, identifier, n) is undefined"
    if (Alignment && Name)
      Diag(PragmaLoc, diag::warn_pragma_pack_pop_identifer_and_alignment);    
    
    // Do the pop.
    if (!PackContext.pop(Name)) {
      // If a name was specified then failure indicates the name
      // wasn't found. Otherwise failure indicates the stack was
      // empty.
      Diag(PragmaLoc, diag::warn_pragma_pack_pop_failed)
        << (Name ? "no record matching name" : "stack empty");

      // FIXME: Warn about popping named records as MSVC does.
    } else {
      // Pop succeeded, set the new alignment if specified.
      if (Alignment)
        PackContext.setAlignment(AlignmentVal);
    }
    break;

  default:
    assert(0 && "Invalid #pragma pack kind.");
  }
}

bool PragmaPackStack::pop(IdentifierInfo *Name) {
  if (Stack.empty())
    return false;

  // If name is empty just pop top.
  if (!Name) {
    Alignment = Stack.back().first;
    Stack.pop_back();
    return true;
  } 

  // Otherwise, find the named record.
  for (unsigned i = Stack.size(); i != 0; ) {
    --i;
    if (Stack[i].second == Name) {
      // Found it, pop up to and including this record.
      Alignment = Stack[i].first;
      Stack.erase(Stack.begin() + i, Stack.end());
      return true;
    }
  }

  return false;
}
