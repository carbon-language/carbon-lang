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
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExprCXX.h"
#include "clang/Parse/DeclSpec.h"
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

/// getDeclName - Return a pretty name for the specified decl if possible, or
/// an empty string if not.  This is used for pretty crash reporting. 
std::string Sema::getDeclName(DeclTy *d) {
  Decl *D = static_cast<Decl *>(d);
  if (NamedDecl *DN = dyn_cast_or_null<NamedDecl>(D))
    return DN->getQualifiedNameAsString();
  return "";
}

/// \brief If the identifier refers to a type name within this scope,
/// return the declaration of that type.
///
/// This routine performs ordinary name lookup of the identifier II
/// within the given scope, with optional C++ scope specifier SS, to
/// determine whether the name refers to a type. If so, returns an
/// opaque pointer (actually a QualType) corresponding to that
/// type. Otherwise, returns NULL.
///
/// If name lookup results in an ambiguity, this routine will complain
/// and then return NULL.
Sema::TypeTy *Sema::getTypeName(IdentifierInfo &II, SourceLocation NameLoc,
                                Scope *S, const CXXScopeSpec *SS) {
  // C++ [temp.res]p3:
  //   A qualified-id that refers to a type and in which the
  //   nested-name-specifier depends on a template-parameter (14.6.2)
  //   shall be prefixed by the keyword typename to indicate that the
  //   qualified-id denotes a type, forming an
  //   elaborated-type-specifier (7.1.5.3).
  //
  // We therefore do not perform any name lookup up SS is a dependent
  // scope name. FIXME: we will need to perform a special kind of
  // lookup if the scope specifier names a member of the current
  // instantiation.
  if (SS && isDependentScopeSpecifier(*SS))
    return 0;

  NamedDecl *IIDecl = 0;
  LookupResult Result = LookupParsedName(S, SS, &II, LookupOrdinaryName, 
                                         false, false);
  switch (Result.getKind()) {
  case LookupResult::NotFound:
  case LookupResult::FoundOverloaded:
    return 0;

  case LookupResult::AmbiguousBaseSubobjectTypes:
  case LookupResult::AmbiguousBaseSubobjects:
  case LookupResult::AmbiguousReference:
    DiagnoseAmbiguousLookup(Result, DeclarationName(&II), NameLoc);
    return 0;

  case LookupResult::Found:
    IIDecl = Result.getAsDecl();
    break;
  }

  if (IIDecl) {
    QualType T;
  
    if (TypeDecl *TD = dyn_cast<TypeDecl>(IIDecl)) {
      // Check whether we can use this type
      (void)DiagnoseUseOfDecl(IIDecl, NameLoc);
      
      T = Context.getTypeDeclType(TD);
    } else if (ObjCInterfaceDecl *IDecl = dyn_cast<ObjCInterfaceDecl>(IIDecl)) {
      // Check whether we can use this interface.
      (void)DiagnoseUseOfDecl(IIDecl, NameLoc);
      
      T = Context.getObjCInterfaceType(IDecl);
    } else
      return 0;

    if (SS)
      T = getQualifiedNameType(*SS, T);

    return T.getAsOpaquePtr();
  }

  return 0;
}

DeclContext *Sema::getContainingDC(DeclContext *DC) {
  if (CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(DC)) {
    // A C++ out-of-line method will return to the file declaration context.
    if (MD->isOutOfLineDefinition())
      return MD->getLexicalDeclContext();

    // A C++ inline method is parsed *after* the topmost class it was declared
    // in is fully parsed (it's "complete").
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

/// \brief Determine whether we allow overloading of the function
/// PrevDecl with another declaration.
///
/// This routine determines whether overloading is possible, not
/// whether some new function is actually an overload. It will return
/// true in C++ (where we can always provide overloads) or, as an
/// extension, in C when the previous function is already an
/// overloaded function declaration or has the "overloadable"
/// attribute.
static bool AllowOverloadingOfFunction(Decl *PrevDecl, ASTContext &Context) {
  if (Context.getLangOptions().CPlusPlus)
    return true;

  if (isa<OverloadedFunctionDecl>(PrevDecl))
    return true;

  return PrevDecl->getAttr<OverloadableAttr>() != 0;
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
        I = IdResolver.begin(TD->getDeclName()),
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
  } else if (isa<FunctionDecl>(D) &&
             AllowOverloadingOfFunction(D, Context)) {
    // We are pushing the name of a function, which might be an
    // overloaded name.
    FunctionDecl *FD = cast<FunctionDecl>(D);
    IdentifierResolver::iterator Redecl
      = std::find_if(IdResolver.begin(FD->getDeclName()),
                     IdResolver.end(),
                     std::bind1st(std::mem_fun(&NamedDecl::declarationReplaces),
                                  FD));
    if (Redecl != IdResolver.end() && S->isDeclScope(*Redecl)) {
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
  NamedDecl *IDecl = LookupName(TUScope, Id, LookupOrdinaryName);
  
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

void Sema::InitBuiltinVaListType() {
  if (!Context.getBuiltinVaListType().isNull())
    return;
  
  IdentifierInfo *VaIdent = &Context.Idents.get("__builtin_va_list");
  NamedDecl *VaDecl = LookupName(TUScope, VaIdent, LookupOrdinaryName);
  TypedefDecl *VaTypedef = cast<TypedefDecl>(VaDecl);
  Context.setBuiltinVaListType(Context.getTypedefType(VaTypedef));
}

/// LazilyCreateBuiltin - The specified Builtin-ID was first used at
/// file scope.  lazily create a decl for it. ForRedeclaration is true
/// if we're creating this built-in in anticipation of redeclaring the
/// built-in.
NamedDecl *Sema::LazilyCreateBuiltin(IdentifierInfo *II, unsigned bid,
                                     Scope *S, bool ForRedeclaration,
                                     SourceLocation Loc) {
  Builtin::ID BID = (Builtin::ID)bid;

  if (Context.BuiltinInfo.hasVAListUse(BID))
    InitBuiltinVaListType();

  Builtin::Context::GetBuiltinTypeError Error;
  QualType R = Context.BuiltinInfo.GetBuiltinType(BID, Context, Error);  
  switch (Error) {
  case Builtin::Context::GE_None:
    // Okay
    break;

  case Builtin::Context::GE_Missing_FILE:
    if (ForRedeclaration)
      Diag(Loc, diag::err_implicit_decl_requires_stdio)
        << Context.BuiltinInfo.GetName(BID);
    return 0;
  }

  if (!ForRedeclaration && Context.BuiltinInfo.isPredefinedLibFunction(BID)) {
    Diag(Loc, diag::ext_implicit_lib_function_decl)
      << Context.BuiltinInfo.GetName(BID)
      << R;
    if (Context.BuiltinInfo.getHeaderName(BID) &&
        Diags.getDiagnosticMapping(diag::ext_implicit_lib_function_decl)
          != diag::MAP_IGNORE)
      Diag(Loc, diag::note_please_include_header)
        << Context.BuiltinInfo.getHeaderName(BID)
        << Context.BuiltinInfo.GetName(BID);
  }

  FunctionDecl *New = FunctionDecl::Create(Context,
                                           Context.getTranslationUnitDecl(),
                                           Loc, II, R,
                                           FunctionDecl::Extern, false,
                                           /*hasPrototype=*/true);
  New->setImplicit();

  // Create Decl objects for each parameter, adding them to the
  // FunctionDecl.
  if (FunctionProtoType *FT = dyn_cast<FunctionProtoType>(R)) {
    llvm::SmallVector<ParmVarDecl*, 16> Params;
    for (unsigned i = 0, e = FT->getNumArgs(); i != e; ++i)
      Params.push_back(ParmVarDecl::Create(Context, New, SourceLocation(), 0,
                                           FT->getArgType(i), VarDecl::None, 0));
    New->setParams(Context, &Params[0], Params.size());
  }
  
  AddKnownFunctionAttributes(New);  
  
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
    Decl *Std = LookupQualifiedName(Global, StdIdent, LookupNamespaceName);
    StdNamespace = dyn_cast_or_null<NamespaceDecl>(Std);
  }
  return StdNamespace;
}

/// MergeTypeDefDecl - We just parsed a typedef 'New' which has the
/// same name and scope as a previous declaration 'Old'.  Figure out
/// how to resolve this situation, merging decls or emitting
/// diagnostics as appropriate. Returns true if there was an error,
/// false otherwise.
///
bool Sema::MergeTypeDefDecl(TypedefDecl *New, Decl *OldD) {
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
      return false;
    case 3:
      if (!TypeID->isStr("SEL"))
        break;
      Context.setObjCSelType(New);
      objc_types = true;
      return false;
    case 8:
      if (!TypeID->isStr("Protocol"))
        break;
      Context.setObjCProtoType(New->getUnderlyingType());
      objc_types = true;
      return false;
    }
    // Fall through - the typedef name was not a builtin type.
  }
  // Verify the old decl was also a type.
  TypeDecl *Old = dyn_cast<TypeDecl>(OldD);
  if (!Old) {
    Diag(New->getLocation(), diag::err_redefinition_different_kind) 
      << New->getDeclName();
    if (!objc_types)
      Diag(OldD->getLocation(), diag::note_previous_definition);
    return true;
  }

  // Determine the "old" type we'll use for checking and diagnostics.  
  QualType OldType;
  if (TypedefDecl *OldTypedef = dyn_cast<TypedefDecl>(Old))
    OldType = OldTypedef->getUnderlyingType();
  else
    OldType = Context.getTypeDeclType(Old);

  // If the typedef types are not identical, reject them in all languages and
  // with any extensions enabled.

  if (OldType != New->getUnderlyingType() && 
      Context.getCanonicalType(OldType) != 
      Context.getCanonicalType(New->getUnderlyingType())) {
    Diag(New->getLocation(), diag::err_redefinition_different_typedef)
      << New->getUnderlyingType() << OldType;
    if (!objc_types)
      Diag(Old->getLocation(), diag::note_previous_definition);
    return true;
  }
  if (objc_types) return false;
  if (getLangOptions().Microsoft) return false;

  // C++ [dcl.typedef]p2:
  //   In a given non-class scope, a typedef specifier can be used to
  //   redefine the name of any type declared in that scope to refer
  //   to the type to which it already refers.
  if (getLangOptions().CPlusPlus && !isa<CXXRecordDecl>(CurContext))
    return false;

  // In C, redeclaration of a type is a constraint violation (6.7.2.3p1).
  // Apparently GCC, Intel, and Sun all silently ignore the redeclaration if
  // *either* declaration is in a system header. The code below implements
  // this adhoc compatibility rule. FIXME: The following code will not
  // work properly when compiling ".i" files (containing preprocessed output).
  if (PP.getDiagnostics().getSuppressSystemWarnings()) {
    SourceManager &SrcMgr = Context.getSourceManager();
    if (SrcMgr.isInSystemHeader(Old->getLocation()))
      return false;
    if (SrcMgr.isInSystemHeader(New->getLocation()))
      return false;
  }

  Diag(New->getLocation(), diag::err_redefinition) << New->getDeclName();
  Diag(Old->getLocation(), diag::note_previous_definition);
  return true;
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
static void MergeAttributes(Decl *New, Decl *Old, ASTContext &C) {
  Attr *attr = const_cast<Attr*>(Old->getAttrs());

  while (attr) {
    Attr *tmp = attr;
    attr = attr->getNext();

    if (!DeclHasAttr(New, tmp) && tmp->isMerged()) {
      tmp->setInherited(true);
      New->addAttr(tmp);
    } else {
      tmp->setNext(0);
      tmp->Destroy(C);
    }
  }

  Old->invalidateAttrs();
}

/// Used in MergeFunctionDecl to keep track of function parameters in
/// C.
struct GNUCompatibleParamWarning {
  ParmVarDecl *OldParm;
  ParmVarDecl *NewParm;
  QualType PromotedType;
};

/// MergeFunctionDecl - We just parsed a function 'New' from
/// declarator D which has the same name and scope as a previous
/// declaration 'Old'.  Figure out how to resolve this situation,
/// merging decls or emitting diagnostics as appropriate.
///
/// In C++, New and Old must be declarations that are not
/// overloaded. Use IsOverload to determine whether New and Old are
/// overloaded, and to select the Old declaration that New should be
/// merged with.
///
/// Returns true if there was an error, false otherwise.
bool Sema::MergeFunctionDecl(FunctionDecl *New, Decl *OldD) {
  assert(!isa<OverloadedFunctionDecl>(OldD) && 
         "Cannot merge with an overloaded function declaration");

  // Verify the old decl was also a function.
  FunctionDecl *Old = dyn_cast<FunctionDecl>(OldD);
  if (!Old) {
    Diag(New->getLocation(), diag::err_redefinition_different_kind)
      << New->getDeclName();
    Diag(OldD->getLocation(), diag::note_previous_definition);
    return true;
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
  
  if (!isa<CXXMethodDecl>(New) && !isa<CXXMethodDecl>(Old) &&
      New->getStorageClass() == FunctionDecl::Static &&
      Old->getStorageClass() != FunctionDecl::Static) {
    Diag(New->getLocation(), diag::err_static_non_static)
      << New;
    Diag(Old->getLocation(), PrevDiag);
    return true;
  }

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
      Diag(Old->getLocation(), PrevDiag) << Old << Old->getType();
      return true;
    }

    const CXXMethodDecl* OldMethod = dyn_cast<CXXMethodDecl>(Old);
    const CXXMethodDecl* NewMethod = dyn_cast<CXXMethodDecl>(New);
    if (OldMethod && NewMethod && 
        OldMethod->getLexicalDeclContext() == 
          NewMethod->getLexicalDeclContext()) {
      //    -- Member function declarations with the same name and the 
      //       same parameter types cannot be overloaded if any of them 
      //       is a static member function declaration.
      if (OldMethod->isStatic() || NewMethod->isStatic()) {
        Diag(New->getLocation(), diag::err_ovl_static_nonstatic_member);
        Diag(Old->getLocation(), PrevDiag) << Old << Old->getType();
        return true;
      }

      // C++ [class.mem]p1:
      //   [...] A member shall not be declared twice in the
      //   member-specification, except that a nested class or member
      //   class template can be declared and then later defined.
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
      Diag(Old->getLocation(), PrevDiag) << Old << Old->getType();
    }

    // (C++98 8.3.5p3):
    //   All declarations for a function shall agree exactly in both the
    //   return type and the parameter-type-list.
    if (OldQType == NewQType)
      return MergeCompatibleFunctionDecls(New, Old);

    // Fall through for conflicting redeclarations and redefinitions.
  }

  // C: Function types need to be compatible, not identical. This handles
  // duplicate function decls like "void f(int); void f(enum X);" properly.
  if (!getLangOptions().CPlusPlus &&
      Context.typesAreCompatible(OldQType, NewQType)) {
    const FunctionType *OldFuncType = OldQType->getAsFunctionType();
    const FunctionType *NewFuncType = NewQType->getAsFunctionType();
    const FunctionProtoType *OldProto = 0;
    if (isa<FunctionNoProtoType>(NewFuncType) &&
        (OldProto = dyn_cast<FunctionProtoType>(OldFuncType))) {
      // The old declaration provided a function prototype, but the
      // new declaration does not. Merge in the prototype.
      llvm::SmallVector<QualType, 16> ParamTypes(OldProto->arg_type_begin(),
                                                 OldProto->arg_type_end());
      NewQType = Context.getFunctionType(NewFuncType->getResultType(),
                                         &ParamTypes[0], ParamTypes.size(),
                                         OldProto->isVariadic(),
                                         OldProto->getTypeQuals());
      New->setType(NewQType);
      New->setInheritedPrototype();

      // Synthesize a parameter for each argument type.
      llvm::SmallVector<ParmVarDecl*, 16> Params;
      for (FunctionProtoType::arg_type_iterator 
             ParamType = OldProto->arg_type_begin(), 
             ParamEnd = OldProto->arg_type_end();
           ParamType != ParamEnd; ++ParamType) {
        ParmVarDecl *Param = ParmVarDecl::Create(Context, New,
                                                 SourceLocation(), 0,
                                                 *ParamType, VarDecl::None,
                                                 0);
        Param->setImplicit();
        Params.push_back(Param);
      }

      New->setParams(Context, &Params[0], Params.size());
    } 

    return MergeCompatibleFunctionDecls(New, Old);
  }

  // GNU C permits a K&R definition to follow a prototype declaration
  // if the declared types of the parameters in the K&R definition
  // match the types in the prototype declaration, even when the
  // promoted types of the parameters from the K&R definition differ
  // from the types in the prototype. GCC then keeps the types from
  // the prototype.
  if (!getLangOptions().CPlusPlus &&
      !getLangOptions().NoExtensions &&
      Old->hasPrototype() && !New->hasPrototype() &&
      New->getType()->getAsFunctionProtoType() &&
      Old->getNumParams() == New->getNumParams()) {
    llvm::SmallVector<QualType, 16> ArgTypes;
    llvm::SmallVector<GNUCompatibleParamWarning, 16> Warnings;
    const FunctionProtoType *OldProto 
      = Old->getType()->getAsFunctionProtoType();
    const FunctionProtoType *NewProto 
      = New->getType()->getAsFunctionProtoType();
    
    // Determine whether this is the GNU C extension.
    bool GNUCompatible = 
      Context.typesAreCompatible(OldProto->getResultType(),
                                 NewProto->getResultType()) &&
      (OldProto->isVariadic() == NewProto->isVariadic());
    for (unsigned Idx = 0, End = Old->getNumParams(); 
         GNUCompatible && Idx != End; ++Idx) {
      ParmVarDecl *OldParm = Old->getParamDecl(Idx);
      ParmVarDecl *NewParm = New->getParamDecl(Idx);
      if (Context.typesAreCompatible(OldParm->getType(), 
                                     NewProto->getArgType(Idx))) {
        ArgTypes.push_back(NewParm->getType());
      } else if (Context.typesAreCompatible(OldParm->getType(),
                                            NewParm->getType())) {
        GNUCompatibleParamWarning Warn 
          = { OldParm, NewParm, NewProto->getArgType(Idx) };
        Warnings.push_back(Warn);
        ArgTypes.push_back(NewParm->getType());
      } else
        GNUCompatible = false;
    }

    if (GNUCompatible) {
      for (unsigned Warn = 0; Warn < Warnings.size(); ++Warn) {
        Diag(Warnings[Warn].NewParm->getLocation(),
             diag::ext_param_promoted_not_compatible_with_prototype)
          << Warnings[Warn].PromotedType
          << Warnings[Warn].OldParm->getType();
        Diag(Warnings[Warn].OldParm->getLocation(), 
             diag::note_previous_declaration);
      }

      New->setType(Context.getFunctionType(NewProto->getResultType(),
                                           &ArgTypes[0], ArgTypes.size(),
                                           NewProto->isVariadic(),
                                           NewProto->getTypeQuals()));
      return MergeCompatibleFunctionDecls(New, Old);
    }

    // Fall through to diagnose conflicting types.
  }

  // A function that has already been declared has been redeclared or defined
  // with a different type- show appropriate diagnostic
  if (unsigned BuiltinID = Old->getBuiltinID(Context)) {
    // The user has declared a builtin function with an incompatible
    // signature.
    if (Context.BuiltinInfo.isPredefinedLibFunction(BuiltinID)) {
      // The function the user is redeclaring is a library-defined
      // function like 'malloc' or 'printf'. Warn about the
      // redeclaration, then ignore it.
      Diag(New->getLocation(), diag::warn_redecl_library_builtin) << New;
      Diag(Old->getLocation(), diag::note_previous_builtin_declaration)
        << Old << Old->getType();
      return true;
    }

    PrevDiag = diag::note_previous_builtin_declaration;
  }

  Diag(New->getLocation(), diag::err_conflicting_types) << New->getDeclName();
  Diag(Old->getLocation(), PrevDiag) << Old << Old->getType();
  return true;
}

/// \brief Completes the merge of two function declarations that are
/// known to be compatible. 
///
/// This routine handles the merging of attributes and other
/// properties of function declarations form the old declaration to
/// the new declaration, once we know that New is in fact a
/// redeclaration of Old.
///
/// \returns false
bool Sema::MergeCompatibleFunctionDecls(FunctionDecl *New, FunctionDecl *Old) {
  // Merge the attributes
  MergeAttributes(New, Old, Context);

  // Merge the storage class.
  New->setStorageClass(Old->getStorageClass());

  // FIXME: need to implement inline semantics

  // Merge "pure" flag.
  if (Old->isPure())
    New->setPure();

  // Merge the "deleted" flag.
  if (Old->isDeleted())
    New->setDeleted();
  
  if (getLangOptions().CPlusPlus)
    return MergeCXXFunctionDecl(New, Old);

  return false;
}

/// MergeVarDecl - We just parsed a variable 'New' which has the same name
/// and scope as a previous declaration 'Old'.  Figure out how to resolve this
/// situation, merging decls or emitting diagnostics as appropriate.
///
/// Tentative definition rules (C99 6.9.2p2) are checked by 
/// FinalizeDeclaratorGroup. Unfortunately, we can't analyze tentative 
/// definitions here, since the initializer hasn't been attached.
/// 
bool Sema::MergeVarDecl(VarDecl *New, Decl *OldD) {
  // Verify the old decl was also a variable.
  VarDecl *Old = dyn_cast<VarDecl>(OldD);
  if (!Old) {
    Diag(New->getLocation(), diag::err_redefinition_different_kind)
      << New->getDeclName();
    Diag(OldD->getLocation(), diag::note_previous_definition);
    return true;
  }

  MergeAttributes(New, Old, Context);

  // Merge the types
  QualType MergedT = Context.mergeTypes(New->getType(), Old->getType());
  if (MergedT.isNull()) {
    Diag(New->getLocation(), diag::err_redefinition_different_type) 
      << New->getDeclName();
    Diag(Old->getLocation(), diag::note_previous_definition);
    return true;
  }
  New->setType(MergedT);

  // C99 6.2.2p4: Check if we have a static decl followed by a non-static.
  if (New->getStorageClass() == VarDecl::Static &&
      (Old->getStorageClass() == VarDecl::None ||
       Old->getStorageClass() == VarDecl::Extern)) {
    Diag(New->getLocation(), diag::err_static_non_static) << New->getDeclName();
    Diag(Old->getLocation(), diag::note_previous_definition);
    return true;
  }
  // C99 6.2.2p4: 
  //   For an identifier declared with the storage-class specifier
  //   extern in a scope in which a prior declaration of that
  //   identifier is visible,23) if the prior declaration specifies
  //   internal or external linkage, the linkage of the identifier at
  //   the later declaration is the same as the linkage specified at
  //   the prior declaration. If no prior declaration is visible, or
  //   if the prior declaration specifies no linkage, then the
  //   identifier has external linkage.
  if (New->hasExternalStorage() && Old->hasLinkage())
    /* Okay */;
  else if (New->getStorageClass() != VarDecl::Static &&
           Old->getStorageClass() == VarDecl::Static) {
    Diag(New->getLocation(), diag::err_non_static_static) << New->getDeclName();
    Diag(Old->getLocation(), diag::note_previous_definition);
    return true;
  }
  // Variables with external linkage are analyzed in FinalizeDeclaratorGroup.
  if (New->getStorageClass() != VarDecl::Extern && !New->isFileVarDecl() &&
      // Don't complain about out-of-line definitions of static members.
      !(Old->getLexicalDeclContext()->isRecord() &&
        !New->getLexicalDeclContext()->isRecord())) {
    Diag(New->getLocation(), diag::err_redefinition) << New->getDeclName();
    Diag(Old->getLocation(), diag::note_previous_definition);
    return true;
  }

  // Keep a chain of previous declarations.
  New->setPreviousDeclaration(Old);

  return false;
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
        RequireCompleteType(Param->getLocation(), Param->getType(),
                               diag::err_typecheck_decl_incomplete_type)) {
      Param->setInvalidDecl();
      HasInvalidParm = true;
    }
    
    // C99 6.9.1p5: If the declarator includes a parameter type list, the
    // declaration of each parameter shall include an identifier.
    if (Param->getIdentifier() == 0 && 
        !Param->isImplicit() &&
        !getLangOptions().CPlusPlus)
      Diag(Param->getLocation(), diag::err_parameter_name_omitted);
  }

  return HasInvalidParm;
}

/// ParsedFreeStandingDeclSpec - This method is invoked when a declspec with
/// no declarator (e.g. "struct foo;") is parsed.
Sema::DeclTy *Sema::ParsedFreeStandingDeclSpec(Scope *S, DeclSpec &DS) {
  TagDecl *Tag = 0;
  if (DS.getTypeSpecType() == DeclSpec::TST_class ||
      DS.getTypeSpecType() == DeclSpec::TST_struct ||
      DS.getTypeSpecType() == DeclSpec::TST_union ||
      DS.getTypeSpecType() == DeclSpec::TST_enum)
    Tag = dyn_cast<TagDecl>(static_cast<Decl *>(DS.getTypeRep()));

  if (RecordDecl *Record = dyn_cast_or_null<RecordDecl>(Tag)) {
    if (!Record->getDeclName() && Record->isDefinition() &&
        DS.getStorageClassSpec() != DeclSpec::SCS_typedef) {
      if (getLangOptions().CPlusPlus ||
          Record->getDeclContext()->isRecord())
        return BuildAnonymousStructOrUnion(S, DS, Record);

      Diag(DS.getSourceRange().getBegin(), diag::err_no_declarators)
        << DS.getSourceRange();
    }

    // Microsoft allows unnamed struct/union fields. Don't complain
    // about them.
    // FIXME: Should we support Microsoft's extensions in this area?
    if (Record->getDeclName() && getLangOptions().Microsoft)
      return Tag;
  }

  if (!DS.isMissingDeclaratorOk() && 
      DS.getTypeSpecType() != DeclSpec::TST_error) {
    // Warn about typedefs of enums without names, since this is an
    // extension in both Microsoft an GNU.
    if (DS.getStorageClassSpec() == DeclSpec::SCS_typedef &&
        Tag && isa<EnumDecl>(Tag)) {
      Diag(DS.getSourceRange().getBegin(), diag::ext_typedef_without_a_name)
        << DS.getSourceRange();
      return Tag;
    }

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
      NamedDecl *PrevDecl = LookupQualifiedName(Owner, (*F)->getDeclName(),
                                                LookupOrdinaryName, true);
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
      } else if (isa<TagDecl>(*Mem) && (*Mem)->getDeclContext() != Record) {
        // This is a type that showed up in an
        // elaborated-type-specifier inside the anonymous struct or
        // union, but which actually declares a type outside of the
        // anonymous struct or union. It's okay.
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
    QualType Ty = QualType::getFromOpaquePtr(D.getDeclaratorIdType());
    Ty = Context.getCanonicalType(Ty);
    return Context.DeclarationNames.getCXXConstructorName(Ty);
  }

  case Declarator::DK_Destructor: {
    QualType Ty = QualType::getFromOpaquePtr(D.getDeclaratorIdType());
    Ty = Context.getCanonicalType(Ty);
    return Context.DeclarationNames.getCXXDestructorName(Ty);
  }

  case Declarator::DK_Conversion: {
    // FIXME: We'd like to keep the non-canonical type for diagnostics!
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

/// isNearlyMatchingFunction - Determine whether the C++ functions
/// Declaration and Definition are "nearly" matching. This heuristic
/// is used to improve diagnostics in the case where an out-of-line
/// function definition doesn't match any declaration within
/// the class or namespace.
static bool isNearlyMatchingFunction(ASTContext &Context,
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
  NamedDecl *PrevDecl;
  NamedDecl *New;
  bool InvalidDecl = false;

  QualType R = GetTypeForDeclarator(D, S);
  if (R.isNull()) {
    InvalidDecl = true;
    R = Context.IntTy;
  }

  // See if this is a redefinition of a variable in the same scope.
  if (D.getCXXScopeSpec().isInvalid()) {
    DC = CurContext;
    PrevDecl = 0;
    InvalidDecl = true;
  } else if (!D.getCXXScopeSpec().isSet()) {
    LookupNameKind NameKind = LookupOrdinaryName;

    // If the declaration we're planning to build will be a function
    // or object with linkage, then look for another declaration with
    // linkage (C99 6.2.2p4-5 and C++ [basic.link]p6).
    if (D.getDeclSpec().getStorageClassSpec() == DeclSpec::SCS_typedef)
      /* Do nothing*/;
    else if (R->isFunctionType()) {
      if (CurContext->isFunctionOrMethod())
        NameKind = LookupRedeclarationWithLinkage;
    } else if (D.getDeclSpec().getStorageClassSpec() == DeclSpec::SCS_extern)
      NameKind = LookupRedeclarationWithLinkage;

    DC = CurContext;
    PrevDecl = LookupName(S, Name, NameKind, true, 
                          D.getDeclSpec().getStorageClassSpec() != 
                            DeclSpec::SCS_static,
                          D.getIdentifierLoc());
  } else { // Something like "int foo::x;"
    DC = computeDeclContext(D.getCXXScopeSpec());
    // FIXME: RequireCompleteDeclContext(D.getCXXScopeSpec()); ?
    PrevDecl = LookupQualifiedName(DC, Name, LookupOrdinaryName, true);

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

    // First check whether we named the global scope.
    if (isa<TranslationUnitDecl>(DC)) {
      Diag(D.getIdentifierLoc(), diag::err_invalid_declarator_global_scope) 
        << Name << D.getCXXScopeSpec().getRange();
    } else if (!CurContext->Encloses(DC)) {
      // The qualifying scope doesn't enclose the original declaration.
      // Emit diagnostic based on current scope.
      SourceLocation L = D.getIdentifierLoc();
      SourceRange R = D.getCXXScopeSpec().getRange();
      if (isa<FunctionDecl>(CurContext))
        Diag(L, diag::err_invalid_declarator_in_function) << Name << R;
      else 
        Diag(L, diag::err_invalid_declarator_scope)
          << Name << cast<NamedDecl>(DC) << R;
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
  // tag type. Note that this does does not apply if we're declaring a
  // typedef (C++ [dcl.typedef]p4).
  if (PrevDecl && PrevDecl->getIdentifierNamespace() == Decl::IDNS_Tag &&
      D.getDeclSpec().getStorageClassSpec() != DeclSpec::SCS_typedef)
    PrevDecl = 0;

  bool Redeclaration = false;
  if (D.getDeclSpec().getStorageClassSpec() == DeclSpec::SCS_typedef) {
    New = ActOnTypedefDeclarator(S, D, DC, R, LastDeclarator, PrevDecl,
                                 InvalidDecl, Redeclaration);
  } else if (R->isFunctionType()) {
    New = ActOnFunctionDeclarator(S, D, DC, R, LastDeclarator, PrevDecl, 
                                  IsFunctionDefinition, InvalidDecl,
                                  Redeclaration);
  } else {
    New = ActOnVariableDeclarator(S, D, DC, R, LastDeclarator, PrevDecl, 
                                  InvalidDecl, Redeclaration);
  }

  if (New == 0)
    return 0;
  
  // If this has an identifier and is not an invalid redeclaration,
  // add it to the scope stack.
  if (Name && !(Redeclaration && InvalidDecl))
    PushOnScopeChains(New, S);
  // If any semantic error occurred, mark the decl as invalid.
  if (D.getInvalidType() || InvalidDecl)
    New->setInvalidDecl();
  
  return New;
}

/// TryToFixInvalidVariablyModifiedType - Helper method to turn variable array
/// types into constant array types in certain situations which would otherwise
/// be errors (for GCC compatibility).
static QualType TryToFixInvalidVariablyModifiedType(QualType T,
                                                    ASTContext &Context,
                                                    bool &SizeIsNegative) {
  // This method tries to turn a variable array into a constant
  // array even when the size isn't an ICE.  This is necessary
  // for compatibility with code that depends on gcc's buggy
  // constant expression folding, like struct {char x[(int)(char*)2];}
  SizeIsNegative = false;

  if (const PointerType* PTy = dyn_cast<PointerType>(T)) {
    QualType Pointee = PTy->getPointeeType();
    QualType FixedType =
        TryToFixInvalidVariablyModifiedType(Pointee, Context, SizeIsNegative);
    if (FixedType.isNull()) return FixedType;
    FixedType = Context.getPointerType(FixedType);
    FixedType.setCVRQualifiers(T.getCVRQualifiers());
    return FixedType;
  }

  const VariableArrayType* VLATy = dyn_cast<VariableArrayType>(T);
  if (!VLATy)
    return QualType();
  // FIXME: We should probably handle this case
  if (VLATy->getElementType()->isVariablyModifiedType())
    return QualType();
  
  Expr::EvalResult EvalResult;
  if (!VLATy->getSizeExpr() ||
      !VLATy->getSizeExpr()->Evaluate(EvalResult, Context) ||
      !EvalResult.Val.isInt())
    return QualType();

  llvm::APSInt &Res = EvalResult.Val.getInt();
  if (Res >= llvm::APSInt(Res.getBitWidth(), Res.isUnsigned()))
    return Context.getConstantArrayType(VLATy->getElementType(),
                                        Res, ArrayType::Normal, 0);

  SizeIsNegative = true;
  return QualType();
}

/// \brief Register the given locally-scoped external C declaration so
/// that it can be found later for redeclarations
void 
Sema::RegisterLocallyScopedExternCDecl(NamedDecl *ND, NamedDecl *PrevDecl,
                                       Scope *S) {
  assert(ND->getLexicalDeclContext()->isFunctionOrMethod() &&
         "Decl is not a locally-scoped decl!");
  // Note that we have a locally-scoped external with this name.
  LocallyScopedExternalDecls[ND->getDeclName()] = ND;

  if (!PrevDecl)
    return;

  // If there was a previous declaration of this variable, it may be
  // in our identifier chain. Update the identifier chain with the new
  // declaration.
  if (IdResolver.ReplaceDecl(PrevDecl, ND)) {
    // The previous declaration was found on the identifer resolver
    // chain, so remove it from its scope.
    while (S && !S->isDeclScope(PrevDecl))
      S = S->getParent();

    if (S)
      S->RemoveDecl(PrevDecl);
  }
}

NamedDecl*
Sema::ActOnTypedefDeclarator(Scope* S, Declarator& D, DeclContext* DC,
                             QualType R, Decl* LastDeclarator,
                             Decl* PrevDecl, bool& InvalidDecl, 
                             bool &Redeclaration) {
  // Typedef declarators cannot be qualified (C++ [dcl.meaning]p1).
  if (D.getCXXScopeSpec().isSet()) {
    Diag(D.getIdentifierLoc(), diag::err_qualified_typedef_declarator)
      << D.getCXXScopeSpec().getRange();
    InvalidDecl = true;
    // Pretend we didn't see the scope specifier.
    DC = 0;
  }

  if (getLangOptions().CPlusPlus) {
    // Check that there are no default arguments (C++ only).
    CheckExtraCXXDefaultArguments(D);

    if (D.getDeclSpec().isVirtualSpecified())
      Diag(D.getDeclSpec().getVirtualSpecLoc(), 
           diag::err_virtual_non_function);
  }

  TypedefDecl *NewTD = ParseTypedefDecl(S, D, R, LastDeclarator);
  if (!NewTD) return 0;

  // Handle attributes prior to checking for duplicates in MergeVarDecl
  ProcessDeclAttributes(NewTD, D);
  // Merge the decl with the existing one if appropriate. If the decl is
  // in an outer scope, it isn't the same thing.
  if (PrevDecl && isDeclInScope(PrevDecl, DC, S)) {
    Redeclaration = true;
    if (MergeTypeDefDecl(NewTD, PrevDecl))
      InvalidDecl = true;
  }

  if (S->getFnParent() == 0) {
    QualType T = NewTD->getUnderlyingType();
    // C99 6.7.7p2: If a typedef name specifies a variably modified type
    // then it shall have block scope.
    if (T->isVariablyModifiedType()) {
      bool SizeIsNegative;
      QualType FixedTy =
          TryToFixInvalidVariablyModifiedType(T, Context, SizeIsNegative);
      if (!FixedTy.isNull()) {
        Diag(D.getIdentifierLoc(), diag::warn_illegal_constant_array_size);
        NewTD->setUnderlyingType(FixedTy);
      } else {
        if (SizeIsNegative)
          Diag(D.getIdentifierLoc(), diag::err_typecheck_negative_array_size);
        else if (T->isVariableArrayType())
          Diag(D.getIdentifierLoc(), diag::err_vla_decl_in_file_scope);
        else
          Diag(D.getIdentifierLoc(), diag::err_vm_decl_in_file_scope);
        InvalidDecl = true;
      }
    }
  }
  return NewTD;
}

/// \brief Determines whether the given declaration is an out-of-scope
/// previous declaration.
///
/// This routine should be invoked when name lookup has found a
/// previous declaration (PrevDecl) that is not in the scope where a
/// new declaration by the same name is being introduced. If the new
/// declaration occurs in a local scope, previous declarations with
/// linkage may still be considered previous declarations (C99
/// 6.2.2p4-5, C++ [basic.link]p6).
///
/// \param PrevDecl the previous declaration found by name
/// lookup
/// 
/// \param DC the context in which the new declaration is being
/// declared.
///
/// \returns true if PrevDecl is an out-of-scope previous declaration
/// for a new delcaration with the same name.
static bool 
isOutOfScopePreviousDeclaration(NamedDecl *PrevDecl, DeclContext *DC,
                                ASTContext &Context) {
  if (!PrevDecl)
    return 0;

  // FIXME: PrevDecl could be an OverloadedFunctionDecl, in which
  // case we need to check each of the overloaded functions.
  if (!PrevDecl->hasLinkage())
    return false;

  if (Context.getLangOptions().CPlusPlus) {
    // C++ [basic.link]p6:
    //   If there is a visible declaration of an entity with linkage
    //   having the same name and type, ignoring entities declared
    //   outside the innermost enclosing namespace scope, the block
    //   scope declaration declares that same entity and receives the
    //   linkage of the previous declaration.
    DeclContext *OuterContext = DC->getLookupContext();
    if (!OuterContext->isFunctionOrMethod())
      // This rule only applies to block-scope declarations.
      return false;
    else {
      DeclContext *PrevOuterContext = PrevDecl->getDeclContext();
      if (PrevOuterContext->isRecord())
        // We found a member function: ignore it.
        return false;
      else {
        // Find the innermost enclosing namespace for the new and
        // previous declarations.
        while (!OuterContext->isFileContext())
          OuterContext = OuterContext->getParent();
        while (!PrevOuterContext->isFileContext())
          PrevOuterContext = PrevOuterContext->getParent();
          
        // The previous declaration is in a different namespace, so it
        // isn't the same function.
        if (OuterContext->getPrimaryContext() != 
            PrevOuterContext->getPrimaryContext())
          return false;
      }
    }
  }

  return true;
}

NamedDecl*
Sema::ActOnVariableDeclarator(Scope* S, Declarator& D, DeclContext* DC,
                              QualType R, Decl* LastDeclarator,
                              NamedDecl* PrevDecl, bool& InvalidDecl,
                              bool &Redeclaration) {
  DeclarationName Name = GetNameForDeclarator(D);

  // Check that there are no default arguments (C++ only).
  if (getLangOptions().CPlusPlus)
    CheckExtraCXXDefaultArguments(D);

  if (R.getTypePtr()->isObjCInterfaceType()) {
    Diag(D.getIdentifierLoc(), diag::err_statically_allocated_object);
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

  if (D.getDeclSpec().isVirtualSpecified())
    Diag(D.getDeclSpec().getVirtualSpecLoc(), 
         diag::err_virtual_non_function);

  bool ThreadSpecified = D.getDeclSpec().isThreadSpecified();
  if (!DC->isRecord() && S->getFnParent() == 0) {
    // C99 6.9p2: The storage-class specifiers auto and register shall not
    // appear in the declaration specifiers in an external declaration.
    if (SC == VarDecl::Auto || SC == VarDecl::Register) {
      Diag(D.getIdentifierLoc(), diag::err_typecheck_sclass_fscope);
      InvalidDecl = true;
    }
  }
  if (DC->isRecord() && !CurContext->isRecord()) {
    // This is an out-of-line definition of a static data member.
    if (SC == VarDecl::Static) {
      Diag(D.getDeclSpec().getStorageClassSpecLoc(), 
           diag::err_static_out_of_line)
        << CodeModificationHint::CreateRemoval(
                       SourceRange(D.getDeclSpec().getStorageClassSpecLoc()));
    } else if (SC == VarDecl::None)
      SC = VarDecl::Static;
  }

  // The variable can not have an abstract class type.
  if (RequireNonAbstractType(D.getIdentifierLoc(), R, 2 /* variable type */))
    InvalidDecl = true;
    
  // The variable can not 
  NewVD = VarDecl::Create(Context, DC, D.getIdentifierLoc(), 
                          II, R, SC, 
                          // FIXME: Move to DeclGroup...
                          D.getDeclSpec().getSourceRange().getBegin());
  NewVD->setThreadSpecified(ThreadSpecified);
  NewVD->setNextDeclarator(LastDeclarator);

  // Set the lexical context. If the declarator has a C++ scope specifier, the
  // lexical context will be different from the semantic context.
  NewVD->setLexicalDeclContext(CurContext);

  // Handle attributes prior to checking for duplicates in MergeVarDecl
  ProcessDeclAttributes(NewVD, D);

  // Handle GNU asm-label extension (encoded as an attribute).
  if (Expr *E = (Expr*) D.getAsmLabel()) {
    // The parser guarantees this is a string.
    StringLiteral *SE = cast<StringLiteral>(E);  
    NewVD->addAttr(::new (Context) AsmLabelAttr(std::string(SE->getStrData(),
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

  if (NewVD->hasLocalStorage() && NewVD->getType().isObjCGCWeak()) {
    Diag(D.getIdentifierLoc(), diag::warn_attribute_weak_on_local);
  }

  bool isIllegalVLA = R->isVariableArrayType() && NewVD->hasGlobalStorage();
  bool isIllegalVM = R->isVariablyModifiedType() && NewVD->hasLinkage();
  if (isIllegalVLA || isIllegalVM) {
    bool SizeIsNegative;
    QualType FixedTy =
        TryToFixInvalidVariablyModifiedType(R, Context, SizeIsNegative);
    if (!FixedTy.isNull()) {
      Diag(NewVD->getLocation(), diag::warn_illegal_constant_array_size);
      NewVD->setType(FixedTy);
    } else if (R->isVariableArrayType()) {
      NewVD->setInvalidDecl();

      const VariableArrayType *VAT = Context.getAsVariableArrayType(R);
      // FIXME: This won't give the correct result for 
      // int a[10][n];      
      SourceRange SizeRange = VAT->getSizeExpr()->getSourceRange();

      if (NewVD->isFileVarDecl())
        Diag(NewVD->getLocation(), diag::err_vla_decl_in_file_scope)
          << SizeRange;
      else if (NewVD->getStorageClass() == VarDecl::Static)
        Diag(NewVD->getLocation(), diag::err_vla_decl_has_static_storage)
          << SizeRange;
      else
        Diag(NewVD->getLocation(), diag::err_vla_decl_has_extern_linkage)
            << SizeRange;
    } else {
      InvalidDecl = true;
      
      if (NewVD->isFileVarDecl())
        Diag(NewVD->getLocation(), diag::err_vm_decl_in_file_scope);
      else
        Diag(NewVD->getLocation(), diag::err_vm_decl_has_extern_linkage);
    }
  }

  // If name lookup finds a previous declaration that is not in the
  // same scope as the new declaration, this may still be an
  // acceptable redeclaration.
  if (PrevDecl && !isDeclInScope(PrevDecl, DC, S) &&
      !(NewVD->hasLinkage() &&
        isOutOfScopePreviousDeclaration(PrevDecl, DC, Context)))
    PrevDecl = 0;     

  if (!PrevDecl && NewVD->isExternC(Context)) {
    // Since we did not find anything by this name and we're declaring
    // an extern "C" variable, look for a non-visible extern "C"
    // declaration with the same name.
    llvm::DenseMap<DeclarationName, NamedDecl *>::iterator Pos
      = LocallyScopedExternalDecls.find(Name);
    if (Pos != LocallyScopedExternalDecls.end())
      PrevDecl = Pos->second;
  }

  // Merge the decl with the existing one if appropriate.
  if (PrevDecl) {
    if (isa<FieldDecl>(PrevDecl) && D.getCXXScopeSpec().isSet()) {
      // The user tried to define a non-static data member
      // out-of-line (C++ [dcl.meaning]p1).
      Diag(NewVD->getLocation(), diag::err_nonstatic_member_out_of_line)
        << D.getCXXScopeSpec().getRange();
      NewVD->Destroy(Context);
      return 0;
    }

    Redeclaration = true;
    if (MergeVarDecl(NewVD, PrevDecl))
      InvalidDecl = true;
  } else if (D.getCXXScopeSpec().isSet()) {
    // No previous declaration in the qualifying scope.
    Diag(D.getIdentifierLoc(), diag::err_typecheck_no_member)
      << Name << D.getCXXScopeSpec().getRange();
    InvalidDecl = true;
  }

  if (!InvalidDecl && R->isVoidType() && !NewVD->hasExternalStorage()) {
    Diag(NewVD->getLocation(), diag::err_typecheck_decl_incomplete_type)
      << R;
    InvalidDecl = true;
  }


  // If this is a locally-scoped extern C variable, update the map of
  // such variables.
  if (CurContext->isFunctionOrMethod() && NewVD->isExternC(Context) &&
      !InvalidDecl)
    RegisterLocallyScopedExternCDecl(NewVD, PrevDecl, S);

  return NewVD;
}

NamedDecl* 
Sema::ActOnFunctionDeclarator(Scope* S, Declarator& D, DeclContext* DC,
                              QualType R, Decl *LastDeclarator,
                              NamedDecl* PrevDecl, bool IsFunctionDefinition,
                              bool& InvalidDecl, bool &Redeclaration) {
  assert(R.getTypePtr()->isFunctionType());

  DeclarationName Name = GetNameForDeclarator(D);
  FunctionDecl::StorageClass SC = FunctionDecl::None;
  switch (D.getDeclSpec().getStorageClassSpec()) {
  default: assert(0 && "Unknown storage class!");
  case DeclSpec::SCS_auto:
  case DeclSpec::SCS_register:
  case DeclSpec::SCS_mutable:
    Diag(D.getDeclSpec().getStorageClassSpecLoc(), 
         diag::err_typecheck_sclass_func);
    InvalidDecl = true;
    break;
  case DeclSpec::SCS_unspecified: SC = FunctionDecl::None; break;
  case DeclSpec::SCS_extern:      SC = FunctionDecl::Extern; break;
  case DeclSpec::SCS_static: {
    if (CurContext->getLookupContext()->isFunctionOrMethod()) {
      // C99 6.7.1p5:
      //   The declaration of an identifier for a function that has
      //   block scope shall have no explicit storage-class specifier
      //   other than extern
      // See also (C++ [dcl.stc]p4).
      Diag(D.getDeclSpec().getStorageClassSpecLoc(), 
           diag::err_static_block_func);
      SC = FunctionDecl::None;
    } else
      SC = FunctionDecl::Static; 
    break;
  }
  case DeclSpec::SCS_private_extern: SC = FunctionDecl::PrivateExtern;break;
  }

  bool isInline = D.getDeclSpec().isInlineSpecified();
  bool isVirtual = D.getDeclSpec().isVirtualSpecified();
  bool isExplicit = D.getDeclSpec().isExplicitSpecified();

  // Check that the return type is not an abstract class type.
  if (RequireNonAbstractType(D.getIdentifierLoc(), 
                             R->getAsFunctionType()->getResultType(),
                             0 /* return type */))
    InvalidDecl = true;
  
  bool isVirtualOkay = false;
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

      isVirtualOkay = true;
    } else {
      Diag(D.getIdentifierLoc(), diag::err_destructor_not_member);

      // Create a FunctionDecl to satisfy the function definition parsing
      // code path.
      NewFD = FunctionDecl::Create(Context, DC, D.getIdentifierLoc(),
                                   Name, R, SC, isInline, 
                                   /*hasPrototype=*/true,
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

      isVirtualOkay = true;
    }
  } else if (DC->isRecord()) {
    // This is a C++ method declaration.
    NewFD = CXXMethodDecl::Create(Context, cast<CXXRecordDecl>(DC),
                                  D.getIdentifierLoc(), Name, R,
                                  (SC == FunctionDecl::Static), isInline);

    isVirtualOkay = (SC != FunctionDecl::Static);
  } else {
    // Determine whether the function was written with a
    // prototype. This true when:
    //   - we're in C++ (where every function has a prototype),
    //   - there is a prototype in the declarator, or
    //   - the type R of the function is some kind of typedef or other reference
    //     to a type name (which eventually refers to a function type).
    bool HasPrototype = 
       getLangOptions().CPlusPlus ||
       (D.getNumTypeObjects() && D.getTypeObject(0).Fun.hasPrototype) ||
       (!isa<FunctionType>(R.getTypePtr()) && R->isFunctionProtoType());
    
    NewFD = FunctionDecl::Create(Context, DC,
                                 D.getIdentifierLoc(),
                                 Name, R, SC, isInline, HasPrototype,
                                 // FIXME: Move to DeclGroup...
                                 D.getDeclSpec().getSourceRange().getBegin());
  }
  NewFD->setNextDeclarator(LastDeclarator);

  // Set the lexical context. If the declarator has a C++
  // scope specifier, the lexical context will be different
  // from the semantic context.
  NewFD->setLexicalDeclContext(CurContext);

  // C++ [dcl.fct.spec]p5:
  //   The virtual specifier shall only be used in declarations of
  //   nonstatic class member functions that appear within a
  //   member-specification of a class declaration; see 10.3.
  //
  // FIXME: Checking the 'virtual' specifier is not sufficient. A
  // function is also virtual if it overrides an already virtual
  // function. This is important to do here because it's part of the
  // declaration.
  if (isVirtual && !InvalidDecl) {
    if (!isVirtualOkay) {
       Diag(D.getDeclSpec().getVirtualSpecLoc(), 
           diag::err_virtual_non_function);
    } else if (!CurContext->isRecord()) {
      // 'virtual' was specified outside of the class.
      Diag(D.getDeclSpec().getVirtualSpecLoc(), diag::err_virtual_out_of_class)
        << CodeModificationHint::CreateRemoval(
                             SourceRange(D.getDeclSpec().getVirtualSpecLoc()));
    } else {
      // Okay: Add virtual to the method.
      cast<CXXMethodDecl>(NewFD)->setVirtual();
      CXXRecordDecl *CurClass = cast<CXXRecordDecl>(DC);
      CurClass->setAggregate(false);
      CurClass->setPOD(false);
      CurClass->setPolymorphic(true);
    }
  }

  if (SC == FunctionDecl::Static && isa<CXXMethodDecl>(NewFD) && 
      !CurContext->isRecord()) {
    // C++ [class.static]p1:
    //   A data or function member of a class may be declared static
    //   in a class definition, in which case it is a static member of
    //   the class.

    // Complain about the 'static' specifier if it's on an out-of-line
    // member function definition.
    Diag(D.getDeclSpec().getStorageClassSpecLoc(), 
         diag::err_static_out_of_line)
      << CodeModificationHint::CreateRemoval(
                      SourceRange(D.getDeclSpec().getStorageClassSpecLoc()));
  }

  // Handle GNU asm-label extension (encoded as an attribute).
  if (Expr *E = (Expr*) D.getAsmLabel()) {
    // The parser guarantees this is a string.
    StringLiteral *SE = cast<StringLiteral>(E);  
    NewFD->addAttr(::new (Context) AsmLabelAttr(std::string(SE->getStrData(),
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
      for (unsigned i = 0, e = FTI.NumArgs; i != e; ++i) {
        ParmVarDecl *PVD = (ParmVarDecl *)FTI.ArgInfo[i].Param;
        
        // Function parameters cannot have abstract class types.
        if (RequireNonAbstractType(PVD->getLocation(), PVD->getType(),
                                   1 /* parameter type */))
            InvalidDecl = true;
        Params.push_back(PVD);
      }
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
    const FunctionProtoType *FT = R->getAsFunctionProtoType();
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
      for (FunctionProtoType::arg_type_iterator ArgType = FT->arg_type_begin();
           ArgType != FT->arg_type_end(); ++ArgType) {
        ParmVarDecl *Param = ParmVarDecl::Create(Context, DC,
                                                 SourceLocation(), 0,
                                                 *ArgType, VarDecl::None,
                                                 0);
        Param->setImplicit();
        Params.push_back(Param);
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
    
  // If name lookup finds a previous declaration that is not in the
  // same scope as the new declaration, this may still be an
  // acceptable redeclaration.
  if (PrevDecl && !isDeclInScope(PrevDecl, DC, S) &&
      !(NewFD->hasLinkage() &&
        isOutOfScopePreviousDeclaration(PrevDecl, DC, Context)))
    PrevDecl = 0;

  if (!PrevDecl && NewFD->isExternC(Context)) {
    // Since we did not find anything by this name and we're declaring
    // an extern "C" function, look for a non-visible extern "C"
    // declaration with the same name.
    llvm::DenseMap<DeclarationName, NamedDecl *>::iterator Pos
      = LocallyScopedExternalDecls.find(Name);
    if (Pos != LocallyScopedExternalDecls.end())
      PrevDecl = Pos->second;
  }

  // Merge or overload the declaration with an existing declaration of
  // the same name, if appropriate.
  bool OverloadableAttrRequired = false;
  if (PrevDecl) {
    // Determine whether NewFD is an overload of PrevDecl or
    // a declaration that requires merging. If it's an overload,
    // there's no more work to do here; we'll just add the new
    // function to the scope.
    OverloadedFunctionDecl::function_iterator MatchedDecl;

    if (!getLangOptions().CPlusPlus &&
        AllowOverloadingOfFunction(PrevDecl, Context)) {
      OverloadableAttrRequired = true;

      // Functions marked "overloadable" must have a prototype (that
      // we can't get through declaration merging).
      if (!R->getAsFunctionProtoType()) {
        Diag(NewFD->getLocation(), diag::err_attribute_overloadable_no_prototype)
          << NewFD;
        InvalidDecl = true;
        Redeclaration = true;

        // Turn this into a variadic function with no parameters.
        R = Context.getFunctionType(R->getAsFunctionType()->getResultType(),
                                    0, 0, true, 0);
        NewFD->setType(R);
      }
    }

    if (PrevDecl && 
        (!AllowOverloadingOfFunction(PrevDecl, Context) || 
         !IsOverload(NewFD, PrevDecl, MatchedDecl))) {
      Redeclaration = true;
      Decl *OldDecl = PrevDecl;

      // If PrevDecl was an overloaded function, extract the
      // FunctionDecl that matched.
      if (isa<OverloadedFunctionDecl>(PrevDecl))
        OldDecl = *MatchedDecl;

      // NewFD and PrevDecl represent declarations that need to be
      // merged. 
      if (MergeFunctionDecl(NewFD, OldDecl))
        InvalidDecl = true;

      if (!InvalidDecl) {
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
  }

  if (D.getCXXScopeSpec().isSet() &&
      (!PrevDecl || !Redeclaration)) {
    // The user tried to provide an out-of-line definition for a
    // function that is a member of a class or namespace, but there
    // was no such member function declared (C++ [class.mfct]p2, 
    // C++ [namespace.memdef]p2). For example:
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
      << cast<NamedDecl>(DC) << D.getCXXScopeSpec().getRange();
    InvalidDecl = true;
    
    LookupResult Prev = LookupQualifiedName(DC, Name, LookupOrdinaryName, 
                                            true);
    assert(!Prev.isAmbiguous() && 
           "Cannot have an ambiguity in previous-declaration lookup");
    for (LookupResult::iterator Func = Prev.begin(), FuncEnd = Prev.end();
         Func != FuncEnd; ++Func) {
      if (isa<FunctionDecl>(*Func) &&
          isNearlyMatchingFunction(Context, cast<FunctionDecl>(*Func), NewFD))
        Diag((*Func)->getLocation(), diag::note_member_def_close_match);
    }
    
    PrevDecl = 0;
  }

  // Handle attributes. We need to have merged decls when handling attributes
  // (for example to check for conflicts, etc).
  ProcessDeclAttributes(NewFD, D);
  AddKnownFunctionAttributes(NewFD);

  if (OverloadableAttrRequired && !NewFD->getAttr<OverloadableAttr>()) {
    // If a function name is overloadable in C, then every function
    // with that name must be marked "overloadable".
    Diag(NewFD->getLocation(), diag::err_attribute_overloadable_missing)
      << Redeclaration << NewFD;
    if (PrevDecl)
      Diag(PrevDecl->getLocation(), 
           diag::note_attribute_overloadable_prev_overload);
    NewFD->addAttr(::new (Context) OverloadableAttr());
  }

  if (getLangOptions().CPlusPlus) {
    // In C++, check default arguments now that we have merged decls. Unless
    // the lexical context is the class, because in this case this is done
    // during delayed parsing anyway.
    if (!CurContext->isRecord())
      CheckCXXDefaultArguments(NewFD);

    // An out-of-line member function declaration must also be a
    // definition (C++ [dcl.meaning]p1).
    if (!IsFunctionDefinition && D.getCXXScopeSpec().isSet() && !InvalidDecl) {
      Diag(NewFD->getLocation(), diag::err_out_of_line_declaration)
        << D.getCXXScopeSpec().getRange();
      InvalidDecl = true;
    }
  }

  // If this is a locally-scoped extern C function, update the
  // map of such names.
  if (CurContext->isFunctionOrMethod() && NewFD->isExternC(Context)
      && !InvalidDecl)
    RegisterLocallyScopedExternCDecl(NewFD, PrevDecl, S);

  return NewFD;
}

bool Sema::CheckForConstantInitializer(Expr *Init, QualType DclT) {
  // FIXME: Need strict checking.  In C89, we need to check for
  // any assignment, increment, decrement, function-calls, or
  // commas outside of a sizeof.  In C99, it's the same list,
  // except that the aforementioned are allowed in unevaluated
  // expressions.  Everything else falls under the
  // "may accept other forms of constant expressions" exception.
  // (We never end up here for C++, so the constant expression
  // rules there don't matter.)
  if (Init->isConstantInitializer(Context))
    return false;
  Diag(Init->getExprLoc(), diag::err_init_element_not_constant)
    << Init->getSourceRange();
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
  // If there is no declaration, there was an error parsing it.  Just ignore
  // the initializer.
  if (RealDecl == 0)
    return;
  
  if (CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(RealDecl)) {
    // With declarators parsed the way they are, the parser cannot
    // distinguish between a normal initializer and a pure-specifier.
    // Thus this grotesque test.
    IntegerLiteral *IL;
    Expr *Init = static_cast<Expr *>(init.get());
    if ((IL = dyn_cast<IntegerLiteral>(Init)) && IL->getValue() == 0 &&
        Context.getCanonicalType(IL->getType()) == Context.IntTy) {
      if (Method->isVirtual()) {
        Method->setPure();

        // A class is abstract if at least one function is pure virtual.
        cast<CXXRecordDecl>(CurContext)->setAbstract(true);
      } else {
        Diag(Method->getLocation(), diag::err_non_virtual_pure)
          << Method->getDeclName() << Init->getSourceRange();
        Method->setInvalidDecl();
      }
    } else {
      Diag(Method->getLocation(), diag::err_member_function_initialization)
        << Method->getDeclName() << Init->getSourceRange();
      Method->setInvalidDecl();
    }
    return;
  }

  VarDecl *VDecl = dyn_cast<VarDecl>(RealDecl);
  if (!VDecl) {
    if (getLangOptions().CPlusPlus &&
        RealDecl->getLexicalDeclContext()->isRecord() &&
        isa<NamedDecl>(RealDecl))
      Diag(RealDecl->getLocation(), diag::err_member_initialization)
        << cast<NamedDecl>(RealDecl)->getDeclName();
    else
      Diag(RealDecl->getLocation(), diag::err_illegal_initializer);
    RealDecl->setInvalidDecl();
    return;
  }

  const VarDecl *Def = 0;
  if (VDecl->getDefinition(Def)) {
    Diag(VDecl->getLocation(), diag::err_redefinition) 
      << VDecl->getDeclName();
    Diag(Def->getLocation(), diag::note_previous_definition);
    VDecl->setInvalidDecl();
    return;
  }

  // Take ownership of the expression, now that we're sure we have somewhere
  // to put it.
  Expr *Init = static_cast<Expr *>(init.release());
  assert(Init && "missing initializer");

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
      // Don't check invalid declarations to avoid emitting useless diagnostics.
      if (!getLangOptions().CPlusPlus && !VDecl->isInvalidDecl()) {
        if (SC == VarDecl::Static) // C99 6.7.8p4.
          CheckForConstantInitializer(Init, DclT);
      }
    }
  } else if (VDecl->isStaticDataMember() && 
             VDecl->getLexicalDeclContext()->isRecord()) {
    // This is an in-class initialization for a static data member, e.g.,
    //
    // struct S {
    //   static const int value = 17;
    // };

    // Attach the initializer
    VDecl->setInit(Init);

    // C++ [class.mem]p4:
    //   A member-declarator can contain a constant-initializer only
    //   if it declares a static member (9.4) of const integral or
    //   const enumeration type, see 9.4.2.
    QualType T = VDecl->getType();
    if (!T->isDependentType() && 
        (!Context.getCanonicalType(T).isConstQualified() ||
         !T->isIntegralType())) {
      Diag(VDecl->getLocation(), diag::err_member_initialization)
        << VDecl->getDeclName() << Init->getSourceRange();
      VDecl->setInvalidDecl();
    } else {
      // C++ [class.static.data]p4:
      //   If a static data member is of const integral or const
      //   enumeration type, its declaration in the class definition
      //   can specify a constant-initializer which shall be an
      //   integral constant expression (5.19).
      if (!Init->isTypeDependent() &&
          !Init->getType()->isIntegralType()) {
        // We have a non-dependent, non-integral or enumeration type.
        Diag(Init->getSourceRange().getBegin(), 
             diag::err_in_class_initializer_non_integral_type)
          << Init->getType() << Init->getSourceRange();
        VDecl->setInvalidDecl();
      } else if (!Init->isTypeDependent() && !Init->isValueDependent()) {
        // Check whether the expression is a constant expression.
        llvm::APSInt Value;
        SourceLocation Loc;
        if (!Init->isIntegerConstantExpr(Value, Context, &Loc)) {
          Diag(Loc, diag::err_in_class_initializer_non_constant)
            << Init->getSourceRange();
          VDecl->setInvalidDecl();
        }
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
    // Don't check invalid declarations to avoid emitting useless diagnostics.
    if (!getLangOptions().CPlusPlus && !VDecl->isInvalidDecl()) {
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
        const CXXConstructorDecl *Constructor = 0;
        if (!RequireCompleteType(Var->getLocation(), InitType, 
                                    diag::err_invalid_incomplete_type_use))
          Constructor
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
  Decl *Group = static_cast<Decl*>(group);
  if (Group == 0)
    return 0;
  
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

    // Block scope. C99 6.7p7: If an identifier for an object is declared with
    // no linkage (C99 6.2.2p6), the type for the object shall be complete...
    if (IDecl->isBlockVarDecl() && 
        IDecl->getStorageClass() != VarDecl::Extern) {
      if (!IDecl->isInvalidDecl() &&
          RequireCompleteType(IDecl->getLocation(), T, 
                                 diag::err_typecheck_decl_incomplete_type))
        IDecl->setInvalidDecl();
    }
    // File scope. C99 6.9.2p2: A declaration of an identifier for and 
    // object that has file scope without an initializer, and without a
    // storage-class specifier or with the storage-class specifier "static",
    // constitutes a tentative definition. Note: A tentative definition with
    // external linkage is valid (C99 6.2.2p5).
    if (IDecl->isTentativeDefinition(Context)) {
      QualType CheckType = T;
      unsigned DiagID = diag::err_typecheck_decl_incomplete_type;

      const IncompleteArrayType *ArrayT = Context.getAsIncompleteArrayType(T);
      if (ArrayT) {
        CheckType = ArrayT->getElementType();
        DiagID = diag::err_illegal_decl_array_incomplete_type;
      }

      if (IDecl->isInvalidDecl()) {
        // Do nothing with invalid declarations
      } else if ((ArrayT || IDecl->getStorageClass() == VarDecl::Static) &&
                 RequireCompleteType(IDecl->getLocation(), CheckType, DiagID)) {
        // C99 6.9.2p3: If the declaration of an identifier for an object is
        // a tentative definition and has internal linkage (C99 6.2.2p3), the  
        // declared type shall not be an incomplete type.
        IDecl->setInvalidDecl();
      }
    }
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
    if (NamedDecl *PrevDecl = LookupName(S, II, LookupOrdinaryName)) {
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
  // Parameter declarators cannot be interface types. All ObjC objects are
  // passed by reference.
  if (parmDeclType->isObjCInterfaceType()) {
    Diag(D.getIdentifierLoc(), diag::err_object_cannot_be_by_value) 
         << "passed";
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

  // Builtin functions cannot be defined.
  if (unsigned BuiltinID = FD->getBuiltinID(Context)) {
    if (!Context.BuiltinInfo.isPredefinedLibFunction(BuiltinID)) {
      Diag(FD->getLocation(), diag::err_builtin_definition) << FD;
      FD->setInvalidDecl();
    }
  }

  // The return type of a function definition must be complete
  // (C99 6.9.1p3)
  if (FD->getResultType()->isIncompleteType() &&
      !FD->getResultType()->isVoidType()) {
    Diag(FD->getLocation(), diag::err_func_def_incomplete_result) << FD;
    FD->setInvalidDecl();
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

static bool StatementCreatesScope(Stmt* S) {
  bool result = false;
  if (DeclStmt* DS = dyn_cast<DeclStmt>(S)) {
    for (DeclStmt::decl_iterator i = DS->decl_begin();
         i != DS->decl_end(); ++i) {
      if (VarDecl* D = dyn_cast<VarDecl>(*i)) {
        result |= D->getType()->isVariablyModifiedType();
        result |= !!D->getAttr<CleanupAttr>();
      } else if (TypedefDecl* D = dyn_cast<TypedefDecl>(*i)) {
        result |= D->getUnderlyingType()->isVariablyModifiedType();
      }
    }
  }

  return result;
}

void Sema::RecursiveCalcLabelScopes(llvm::DenseMap<Stmt*, void*>& LabelScopeMap,
                                    llvm::DenseMap<void*, Stmt*>& PopScopeMap,
                                    std::vector<void*>& ScopeStack,
                                    Stmt* CurStmt,
                                    Stmt* ParentCompoundStmt) {
  for (Stmt::child_iterator i = CurStmt->child_begin();
       i != CurStmt->child_end(); ++i) {
    if (!*i) continue;
    if (StatementCreatesScope(*i))  {
      ScopeStack.push_back(*i);
      PopScopeMap[*i] = ParentCompoundStmt;
    } else if (isa<LabelStmt>(CurStmt)) {
      LabelScopeMap[CurStmt] = ScopeStack.size() ? ScopeStack.back() : 0;
    }
    if (isa<DeclStmt>(*i)) continue;
    Stmt* CurCompound = isa<CompoundStmt>(*i) ? *i : ParentCompoundStmt;
    RecursiveCalcLabelScopes(LabelScopeMap, PopScopeMap, ScopeStack,
                             *i, CurCompound);
  }

  while (ScopeStack.size() && PopScopeMap[ScopeStack.back()] == CurStmt) {
    ScopeStack.pop_back();
  }
}

void Sema::RecursiveCalcJumpScopes(llvm::DenseMap<Stmt*, void*>& LabelScopeMap,
                                   llvm::DenseMap<void*, Stmt*>& PopScopeMap,
                                   llvm::DenseMap<Stmt*, void*>& GotoScopeMap,
                                   std::vector<void*>& ScopeStack,
                                   Stmt* CurStmt) {
  for (Stmt::child_iterator i = CurStmt->child_begin();
       i != CurStmt->child_end(); ++i) {
    if (!*i) continue;
    if (StatementCreatesScope(*i))  {
      ScopeStack.push_back(*i);
    } else if (GotoStmt* GS = dyn_cast<GotoStmt>(*i)) {
      void* LScope = LabelScopeMap[GS->getLabel()];
      if (LScope) {
        bool foundScopeInStack = false;
        for (unsigned i = ScopeStack.size(); i > 0; --i) {
          if (LScope == ScopeStack[i-1]) {
            foundScopeInStack = true;
            break;
          }
        }
        if (!foundScopeInStack) {
          Diag(GS->getSourceRange().getBegin(), diag::err_goto_into_scope);
        }
      }
    }
    if (isa<DeclStmt>(*i)) continue;
    RecursiveCalcJumpScopes(LabelScopeMap, PopScopeMap, GotoScopeMap,
                            ScopeStack, *i);
  }

  while (ScopeStack.size() && PopScopeMap[ScopeStack.back()] == CurStmt) {
    ScopeStack.pop_back();
  }
}

Sema::DeclTy *Sema::ActOnFinishFunctionBody(DeclTy *D, StmtArg BodyArg) {
  Decl *dcl = static_cast<Decl *>(D);
  Stmt *Body = static_cast<Stmt*>(BodyArg.release());
  if (FunctionDecl *FD = dyn_cast_or_null<FunctionDecl>(dcl)) {
    FD->setBody(cast<CompoundStmt>(Body));
    assert(FD == getCurFunctionDecl() && "Function parsing confused");
  } else if (ObjCMethodDecl *MD = dyn_cast_or_null<ObjCMethodDecl>(dcl)) {
    assert(MD == getCurMethodDecl() && "Method parsing confused");
    MD->setBody(cast<CompoundStmt>(Body));
  } else {
    Body->Destroy(Context);
    return 0;
  }
  PopDeclContext();
  // Verify and clean out per-function state.

  bool HaveLabels = !LabelMap.empty();
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
#if 0
        // FIXME: Why do this?  Having a 'push_back' in CompoundStmt is ugly,
        // and the AST is malformed anyway.  We should just blow away 'L'.
        L->setSubStmt(new (Context) NullStmt(L->getIdentLoc()));
        cast<CompoundStmt>(Body)->push_back(L);        
#else
        L->Destroy(Context);
#endif
      } else {
        // The whole function wasn't parsed correctly, just delete this.
        L->Destroy(Context);
      }
    }
  }
  LabelMap.clear();

  if (!Body) return D;

  if (HaveLabels) {
    llvm::DenseMap<Stmt*, void*> LabelScopeMap;
    llvm::DenseMap<void*, Stmt*> PopScopeMap;
    llvm::DenseMap<Stmt*, void*> GotoScopeMap;
    std::vector<void*> ScopeStack;
    RecursiveCalcLabelScopes(LabelScopeMap, PopScopeMap, ScopeStack, Body, Body);
    RecursiveCalcJumpScopes(LabelScopeMap, PopScopeMap, GotoScopeMap, ScopeStack, Body);
  }

  return D;
}

/// ImplicitlyDefineFunction - An undeclared identifier was used in a function
/// call, forming a call to an implicitly defined function (per C99 6.5.1p2).
NamedDecl *Sema::ImplicitlyDefineFunction(SourceLocation Loc, 
                                          IdentifierInfo &II, Scope *S) {
  // Before we produce a declaration for an implicitly defined
  // function, see whether there was a locally-scoped declaration of
  // this name as a function or variable. If so, use that
  // (non-visible) declaration, and complain about it.
  llvm::DenseMap<DeclarationName, NamedDecl *>::iterator Pos
    = LocallyScopedExternalDecls.find(&II);
  if (Pos != LocallyScopedExternalDecls.end()) {
    Diag(Loc, diag::warn_use_out_of_scope_declaration) << Pos->second;
    Diag(Pos->second->getLocation(), diag::note_previous_declaration);
    return Pos->second;
  }

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
  D.AddTypeInfo(DeclaratorChunk::getFunction(false, false, SourceLocation(),
                                             0, 0, 0, Loc, D),
                SourceLocation());
  D.SetIdentifier(&II, Loc);

  // Insert this function into translation-unit scope.

  DeclContext *PrevDC = CurContext;
  CurContext = Context.getTranslationUnitDecl();
 
  FunctionDecl *FD = 
    dyn_cast<FunctionDecl>(static_cast<Decl*>(ActOnDeclarator(TUScope, D, 0)));
  FD->setImplicit();

  CurContext = PrevDC;

  AddKnownFunctionAttributes(FD);

  return FD;
}

/// \brief Adds any function attributes that we know a priori based on
/// the declaration of this function.
///
/// These attributes can apply both to implicitly-declared builtins
/// (like __builtin___printf_chk) or to library-declared functions
/// like NSLog or printf.
void Sema::AddKnownFunctionAttributes(FunctionDecl *FD) {
  if (FD->isInvalidDecl())
    return;

  // If this is a built-in function, map its builtin attributes to
  // actual attributes.
  if (unsigned BuiltinID = FD->getBuiltinID(Context)) {
    // Handle printf-formatting attributes.
    unsigned FormatIdx;
    bool HasVAListArg;
    if (Context.BuiltinInfo.isPrintfLike(BuiltinID, FormatIdx, HasVAListArg)) {
      if (!FD->getAttr<FormatAttr>())
        FD->addAttr(::new (Context) FormatAttr("printf", FormatIdx + 1,
                                               FormatIdx + 2));
    }

    // Mark const if we don't care about errno and that is the only
    // thing preventing the function from being const. This allows
    // IRgen to use LLVM intrinsics for such functions.
    if (!getLangOptions().MathErrno &&
        Context.BuiltinInfo.isConstWithoutErrno(BuiltinID)) {
      if (!FD->getAttr<ConstAttr>())
        FD->addAttr(::new (Context) ConstAttr());
    }
  }

  IdentifierInfo *Name = FD->getIdentifier();
  if (!Name)
    return;
  if ((!getLangOptions().CPlusPlus && 
       FD->getDeclContext()->isTranslationUnit()) ||
      (isa<LinkageSpecDecl>(FD->getDeclContext()) &&
       cast<LinkageSpecDecl>(FD->getDeclContext())->getLanguage() == 
       LinkageSpecDecl::lang_c)) {
    // Okay: this could be a libc/libm/Objective-C function we know
    // about.
  } else
    return;

  unsigned KnownID;
  for (KnownID = 0; KnownID != id_num_known_functions; ++KnownID)
    if (KnownFunctionIDs[KnownID] == Name)
      break;

  switch (KnownID) {
  case id_NSLog:
  case id_NSLogv:
    if (const FormatAttr *Format = FD->getAttr<FormatAttr>()) {
      // FIXME: We known better than our headers.
      const_cast<FormatAttr *>(Format)->setType("printf");
    } else 
      FD->addAttr(::new (Context) FormatAttr("printf", 1, 2));
    break;

  case id_asprintf:
  case id_vasprintf:
    if (!FD->getAttr<FormatAttr>())
      FD->addAttr(::new (Context) FormatAttr("printf", 2, 3));
    break;

  default:
    // Unknown function or known function without any attributes to
    // add. Do nothing.
    break;
  }
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
  
  if (TagType *TT = dyn_cast<TagType>(T)) {
    TagDecl *TD = TT->getDecl();
    
    // If the TagDecl that the TypedefDecl points to is an anonymous decl
    // keep track of the TypedefDecl.
    if (!TD->getIdentifier() && !TD->getTypedefForAnonDecl())
      TD->setTypedefForAnonDecl(NewTD);
  }

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
                             AttributeList *Attr) {
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
  NamedDecl *PrevDecl = 0;

  bool Invalid = false;

  if (Name && SS.isNotEmpty()) {
    // We have a nested-name tag ('struct foo::bar').

    // Check for invalid 'foo::'.
    if (SS.isInvalid()) {
      Name = 0;
      goto CreateNewDecl;
    }

    // FIXME: RequireCompleteDeclContext(SS)?
    DC = computeDeclContext(SS);
    SearchDC = DC;
    // Look-up name inside 'foo::'.
    PrevDecl = dyn_cast_or_null<TagDecl>(
                 LookupQualifiedName(DC, Name, LookupTagName, true).getAsDecl());

    // A tag 'foo::bar' must already exist.
    if (PrevDecl == 0) {
      Diag(NameLoc, diag::err_not_tag_in_scope) << Name << SS.getRange();
      Name = 0;
      goto CreateNewDecl;
    }
  } else if (Name) {
    // If this is a named struct, check to see if there was a previous forward
    // declaration or definition.
    // FIXME: We're looking into outer scopes here, even when we
    // shouldn't be. Doing so can result in ambiguities that we
    // shouldn't be diagnosing.
    LookupResult R = LookupName(S, Name, LookupTagName,
                                /*RedeclarationOnly=*/(TK != TK_Reference));
    if (R.isAmbiguous()) {
      DiagnoseAmbiguousLookup(R, Name, NameLoc);
      // FIXME: This is not best way to recover from case like:
      //
      // struct S s;
      //
      // causes needless err_ovl_no_viable_function_in_init latter.
      Name = 0;
      PrevDecl = 0;
      Invalid = true;
    }
    else
      PrevDecl = R;

    if (!getLangOptions().CPlusPlus && TK != TK_Reference) {
      // FIXME: This makes sure that we ignore the contexts associated
      // with C structs, unions, and enums when looking for a matching
      // tag declaration or definition. See the similar lookup tweak
      // in Sema::LookupName; is there a better way to deal with this?
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
    // Check whether the previous declaration is usable.
    (void)DiagnoseUseOfDecl(PrevDecl, NameLoc);
    
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
      // PrevDecl is a namespace, template, or anything else
      // that lives in the IDNS_Tag identifier namespace.
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
             (Kind != TagDecl::TK_enum || !getLangOptions().CPlusPlus)) {
    // C.scope.pdecl]p5:
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
    //
    // GNU C also supports this behavior as part of its incomplete
    // enum types extension, while GNU C++ does not.
    //
    // Find the context where we'll be declaring the tag.
    // FIXME: We would like to maintain the current DeclContext as the
    // lexical context, 
    while (SearchDC->isRecord())
      SearchDC = SearchDC->getParent();

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
    New = EnumDecl::Create(Context, SearchDC, Loc, Name, 
                           cast_or_null<EnumDecl>(PrevDecl));
    // If this is an undefined enum, warn.
    if (TK != TK_Definition && !Invalid)  {
      unsigned DK = getLangOptions().CPlusPlus? diag::err_forward_ref_enum
                                              : diag::ext_forward_ref_enum;
      Diag(Loc, DK);
    }
  } else {
    // struct/union/class

    // FIXME: Tag decls should be chained to any simultaneous vardecls, e.g.:
    // struct X { int A; } D;    D should chain to X.
    if (getLangOptions().CPlusPlus)
      // FIXME: Look for a way to use RecordDecl for simple structs.
      New = CXXRecordDecl::Create(Context, Kind, SearchDC, Loc, Name,
                                  cast_or_null<CXXRecordDecl>(PrevDecl));
    else
      New = RecordDecl::Create(Context, Kind, SearchDC, Loc, Name,
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
    if (unsigned Alignment = getPragmaPackAlignment())
      New->addAttr(::new (Context) PackedAttr(Alignment * 8));
  }

  if (getLangOptions().CPlusPlus && SS.isEmpty() && Name && !Invalid) {
    // C++ [dcl.typedef]p3:
    //   [...] Similarly, in a given scope, a class or enumeration
    //   shall not be declared with the same name as a typedef-name
    //   that is declared in that scope and refers to a type other
    //   than the class or enumeration itself.
    LookupResult Lookup = LookupName(S, Name, LookupOrdinaryName, true);
    TypedefDecl *PrevTypedef = 0;
    if (Lookup.getKind() == LookupResult::Found)
      PrevTypedef = dyn_cast<TypedefDecl>(Lookup.getAsDecl());

    if (PrevTypedef && isDeclInScope(PrevTypedef, SearchDC, S) &&
        Context.getCanonicalType(Context.getTypeDeclType(PrevTypedef)) !=
          Context.getCanonicalType(Context.getTypeDeclType(New))) {
      Diag(Loc, diag::err_tag_definition_of_typedef)
        << Context.getTypeDeclType(New)
        << PrevTypedef->getUnderlyingType();
      Diag(PrevTypedef->getLocation(), diag::note_previous_definition);
      Invalid = true;
    }
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
  New->setLexicalDeclContext(CurContext);

  if (TK == TK_Definition)
    New->startDefinition();
  
  // If this has an identifier, add it to the scope stack.
  if (Name) {
    S = getNonFieldDeclScope(S);
    PushOnScopeChains(New, S);
  } else {
    CurContext->addDecl(New);
  }

  return New;
}

void Sema::ActOnTagStartDefinition(Scope *S, DeclTy *TagD) {
  AdjustDeclIfTemplate(TagD);
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
  AdjustDeclIfTemplate(TagD);
  TagDecl *Tag = cast<TagDecl>((Decl *)TagD);

  if (isa<CXXRecordDecl>(Tag))
    FieldCollector->FinishClass();

  // Exit this scope of this tag's definition.
  PopDeclContext();

  // Notify the consumer that we've defined a tag.
  Consumer.HandleTagDeclDefinition(Tag);
}

bool Sema::VerifyBitField(SourceLocation FieldLoc, IdentifierInfo *FieldName, 
                          QualType FieldTy, const Expr *BitWidth) {
  // C99 6.7.2.1p4 - verify the field type.
  // C++ 9.6p3: A bit-field shall have integral or enumeration type.
  if (!FieldTy->isDependentType() && !FieldTy->isIntegralType()) {
    // Handle incomplete types with specific error.
    if (RequireCompleteType(FieldLoc, FieldTy, diag::err_field_incomplete))
      return true;
    return Diag(FieldLoc, diag::err_not_integral_type_bitfield)
      << FieldName << FieldTy << BitWidth->getSourceRange();
  }

  // If the bit-width is type- or value-dependent, don't try to check
  // it now.
  if (BitWidth->isValueDependent() || BitWidth->isTypeDependent())
    return false;

  llvm::APSInt Value;
  if (VerifyIntegerConstantExpression(BitWidth, &Value))
    return true;

  // Zero-width bitfield is ok for anonymous field.
  if (Value == 0 && FieldName)
    return Diag(FieldLoc, diag::err_bitfield_has_zero_width) << FieldName;
  
  if (Value.isSigned() && Value.isNegative())
    return Diag(FieldLoc, diag::err_bitfield_has_negative_width) 
             << FieldName << Value.toString(10);

  if (!FieldTy->isDependentType()) {
    uint64_t TypeSize = Context.getTypeSize(FieldTy);
    // FIXME: We won't need the 0 size once we check that the field type is valid.
    if (TypeSize && Value.getZExtValue() > TypeSize)
      return Diag(FieldLoc, diag::err_bitfield_width_exceeds_type_size)
        << FieldName << (unsigned)TypeSize;
  }

  return false;
}

/// ActOnField - Each field of a struct/union/class is passed into this in order
/// to create a FieldDecl object for it.
Sema::DeclTy *Sema::ActOnField(Scope *S, DeclTy *TagD,
                               SourceLocation DeclStart, 
                               Declarator &D, ExprTy *BitfieldWidth) {

  return HandleField(S, static_cast<RecordDecl*>(TagD), DeclStart, D,
                     static_cast<Expr*>(BitfieldWidth),
                     AS_public);
}

/// HandleField - Analyze a field of a C struct or a C++ data member.
///
FieldDecl *Sema::HandleField(Scope *S, RecordDecl *Record,
                             SourceLocation DeclStart,
                             Declarator &D, Expr *BitWidth,
                             AccessSpecifier AS) {
  IdentifierInfo *II = D.getIdentifier();
  SourceLocation Loc = DeclStart;
  if (II) Loc = D.getIdentifierLoc();
 
  QualType T = GetTypeForDeclarator(D, S);

  if (getLangOptions().CPlusPlus) {
    CheckExtraCXXDefaultArguments(D);

    if (D.getDeclSpec().isVirtualSpecified())
      Diag(D.getDeclSpec().getVirtualSpecLoc(), 
           diag::err_virtual_non_function);
  }

  NamedDecl *PrevDecl = LookupName(S, II, LookupMemberName, true);
  if (PrevDecl && !isDeclInScope(PrevDecl, Record, S))
    PrevDecl = 0;

  FieldDecl *NewFD 
    = CheckFieldDecl(II, T, Record, Loc,
               D.getDeclSpec().getStorageClassSpec() == DeclSpec::SCS_mutable,
                     BitWidth, AS, PrevDecl, &D);
  if (NewFD->isInvalidDecl() && PrevDecl) {
    // Don't introduce NewFD into scope; there's already something
    // with the same name in the same scope.
  } else if (II) {
    PushOnScopeChains(NewFD, S);
  } else
    Record->addDecl(NewFD);

  return NewFD;
}

/// \brief Build a new FieldDecl and check its well-formedness.
///
/// This routine builds a new FieldDecl given the fields name, type,
/// record, etc. \p PrevDecl should refer to any previous declaration
/// with the same name and in the same scope as the field to be
/// created.
///
/// \returns a new FieldDecl.
///
/// \todo The Declarator argument is a hack. It will be removed once 
FieldDecl *Sema::CheckFieldDecl(DeclarationName Name, QualType T, 
                                RecordDecl *Record, SourceLocation Loc,
                                bool Mutable, Expr *BitWidth, 
                                AccessSpecifier AS, NamedDecl *PrevDecl,
                                Declarator *D) {
  IdentifierInfo *II = Name.getAsIdentifierInfo();
  bool InvalidDecl = false;

  // If we receive a broken type, recover by assuming 'int' and
  // marking this declaration as invalid.
  if (T.isNull()) {
    InvalidDecl = true;
    T = Context.IntTy;
  }

  // C99 6.7.2.1p8: A member of a structure or union may have any type other
  // than a variably modified type.
  if (T->isVariablyModifiedType()) {
    bool SizeIsNegative;
    QualType FixedTy = TryToFixInvalidVariablyModifiedType(T, Context,
                                                           SizeIsNegative);
    if (!FixedTy.isNull()) {
      Diag(Loc, diag::warn_illegal_constant_array_size);
      T = FixedTy;
    } else {
      if (SizeIsNegative)
        Diag(Loc, diag::err_typecheck_negative_array_size);
      else
        Diag(Loc, diag::err_typecheck_field_variable_size);
      T = Context.IntTy;
      InvalidDecl = true;
    }
  }
  
  // Fields can not have abstract class types
  if (RequireNonAbstractType(Loc, T, 3 /* field type */))
    InvalidDecl = true;
  
  // If this is declared as a bit-field, check the bit-field.
  if (BitWidth && VerifyBitField(Loc, II, T, BitWidth)) {
    InvalidDecl = true;
    DeleteExpr(BitWidth);
    BitWidth = 0;
  }
  
  FieldDecl *NewFD = FieldDecl::Create(Context, Record, Loc, II, T, BitWidth,
                                       Mutable);

  if (PrevDecl && !isa<TagDecl>(PrevDecl)) {
    Diag(Loc, diag::err_duplicate_member) << II;
    Diag(PrevDecl->getLocation(), diag::note_previous_declaration);
    NewFD->setInvalidDecl();
    Record->setInvalidDecl();
  }

  if (getLangOptions().CPlusPlus && !T->isPODType())
    cast<CXXRecordDecl>(Record)->setPOD(false);

  // FIXME: We need to pass in the attributes given an AST
  // representation, not a parser representation.
  if (D)
    ProcessDeclAttributes(NewFD, *D);

  if (T.isObjCGCWeak())
    Diag(Loc, diag::warn_attribute_weak_on_field);

  if (InvalidDecl)
    NewFD->setInvalidDecl();

  NewFD->setAccess(AS);

  // C++ [dcl.init.aggr]p1:
  //   An aggregate is an array or a class (clause 9) with [...] no
  //   private or protected non-static data members (clause 11).
  // A POD must be an aggregate.
  if (getLangOptions().CPlusPlus &&
      (AS == AS_private || AS == AS_protected)) {
    CXXRecordDecl *CXXRecord = cast<CXXRecordDecl>(Record);
    CXXRecord->setAggregate(false);
    CXXRecord->setPOD(false);
  }

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
    // 6.7.2.1p3, 6.7.2.1p4
    if (VerifyBitField(Loc, II, T, BitWidth)) {
      InvalidDecl = true;
      DeleteExpr(BitWidth);
      BitWidth = 0;
    }
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
  ObjCIvarDecl *NewID = ObjCIvarDecl::Create(Context, CurContext, Loc, II, T,ac,
                                             (Expr *)BitfieldWidth);
  
  if (II) {
    NamedDecl *PrevDecl = LookupName(S, II, LookupMemberName, true);
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
  
  // If the decl this is being inserted into is invalid, then it may be a
  // redeclaration or some other bogus case.  Don't try to add fields to it.
  if (EnclosingDecl->isInvalidDecl()) {
    // FIXME: Deallocate fields?
    return;
  }

  
  // Verify that all the fields are okay.
  unsigned NumNamedMembers = 0;
  llvm::SmallVector<FieldDecl*, 32> RecFields;

  RecordDecl *Record = dyn_cast<RecordDecl>(EnclosingDecl);
  for (unsigned i = 0; i != NumFields; ++i) {
    FieldDecl *FD = cast_or_null<FieldDecl>(static_cast<Decl*>(Fields[i]));
    assert(FD && "missing field decl");
    
    // Get the type for the field.
    Type *FDTy = FD->getType().getTypePtr();

    if (!FD->isAnonymousStructOrUnion()) {
      // Remember all fields written by the user.
      RecFields.push_back(FD);
    }
    
    // If the field is already invalid for some reason, don't emit more
    // diagnostics about it.
    if (FD->isInvalidDecl())
      continue;
      
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
        RequireCompleteType(FD->getLocation(), FD->getType(), 
                               diag::err_field_incomplete);
        FD->setInvalidDecl();
        EnclosingDecl->setInvalidDecl();
        continue;
      }
      if (i != NumFields-1 ||                   // ... that the last member ...
          !Record->isStruct() ||  // ... of a structure ...
          !FDTy->isArrayType()) {         //... may have incomplete array type.
        RequireCompleteType(FD->getLocation(), FD->getType(), 
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
          if (i != NumFields-1)
            Diag(FD->getLocation(), diag::ext_variable_sized_type_in_struct)
              << FD->getDeclName();
          else {
            // We support flexible arrays at the end of structs in
            // other structs as an extension.
            Diag(FD->getLocation(), diag::ext_flexible_array_in_struct)
              << FD->getDeclName();
            if (Record)
              Record->setHasFlexibleArrayMember(true);
          }
        }
      }
    }
    /// A field cannot be an Objective-c object
    if (FDTy->isObjCInterfaceType()) {
      Diag(FD->getLocation(), diag::err_statically_allocated_object);
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
      ID->setIVarList(ClsFields, RecFields.size(), Context);
      ID->setLocEnd(RBrac);
      
      // Must enforce the rule that ivars in the base classes may not be
      // duplicates.
      if (ID->getSuperClass()) {
        for (ObjCInterfaceDecl::ivar_iterator IVI = ID->ivar_begin(), 
             IVE = ID->ivar_end(); IVI != IVE; ++IVI) {
          ObjCIvarDecl* Ivar = (*IVI);
          IdentifierInfo *II = Ivar->getIdentifier();
          ObjCIvarDecl* prevIvar =
            ID->getSuperClass()->lookupInstanceVariable(II);
          if (prevIvar) {
            Diag(Ivar->getLocation(), diag::err_duplicate_member) << II;
            Diag(prevIvar->getLocation(), diag::note_previous_declaration);
          }
        }
      }
    } else if (ObjCImplementationDecl *IMPDecl = 
                  dyn_cast<ObjCImplementationDecl>(EnclosingDecl)) {
      assert(IMPDecl && "ActOnFields - missing ObjCImplementationDecl");
      IMPDecl->setIVarList(ClsFields, RecFields.size(), Context);
      CheckImplementationIvars(IMPDecl, ClsFields, RecFields.size(), RBrac);
    }
  }

  if (Attr)
    ProcessDeclAttributeList(Record, Attr);
}

EnumConstantDecl *Sema::CheckEnumConstant(EnumDecl *Enum,
                                          EnumConstantDecl *LastEnumConst,
                                          SourceLocation IdLoc,
                                          IdentifierInfo *Id,
                                          ExprArg val) {
  Expr *Val = (Expr *)val.get();

  llvm::APSInt EnumVal(32);
  QualType EltTy;
  if (Val && !Val->isTypeDependent()) {
    // Make sure to promote the operand type to int.
    UsualUnaryConversions(Val);
    if (Val != val.get()) {
      val.release();
      val = Val;
    }

    // C99 6.7.2.2p2: Make sure we have an integer constant expression.
    SourceLocation ExpLoc;
    if (!Val->isValueDependent() &&
        VerifyIntegerConstantExpression(Val, &EnumVal)) {
      Val = 0;
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
  
  val.release();
  return EnumConstantDecl::Create(Context, Enum, IdLoc, Id, EltTy,
                                  Val, EnumVal);  
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
  NamedDecl *PrevDecl = LookupName(S, Id, LookupOrdinaryName);
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
      if (Val) Val->Destroy(Context);
      return 0;
    }
  }

  EnumConstantDecl *New = CheckEnumConstant(TheEnumDecl, LastEnumConst,
                                            IdLoc, Id, Owned(Val));

  // Register this decl in the current scope stack.
  if (New)
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
      ECD->setInitExpr(new (Context) ImplicitCastExpr(NewTy, ECD->getInitExpr(), 
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

