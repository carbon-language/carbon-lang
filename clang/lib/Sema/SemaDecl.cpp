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
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Builtins.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"
#include "clang/Parse/DeclSpec.h"
#include "clang/Parse/Scope.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/SourceManager.h"
#include "clang/AST/ExprCXX.h"
// FIXME: layering (ideally, Sema shouldn't be dependent on Lex API's)
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/HeaderSearch.h" 
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/DenseSet.h"
using namespace clang;

Sema::DeclTy *Sema::isTypeName(const IdentifierInfo &II, Scope *S) {
  Decl *IIDecl = LookupDecl(&II, Decl::IDNS_Ordinary, S, false);

  if (IIDecl && (isa<TypedefDecl>(IIDecl) || 
                 isa<ObjCInterfaceDecl>(IIDecl) ||
                 isa<TagDecl>(IIDecl)))
    return IIDecl;
  return 0;
}

void Sema::PushDeclContext(DeclContext *DC) {
  assert( ( (isa<ObjCMethodDecl>(DC) && isa<TranslationUnitDecl>(CurContext))
            || DC->getParent() == CurContext ) &&
      "The next DeclContext should be directly contained in the current one.");
  CurContext = DC;
}

void Sema::PopDeclContext() {
  assert(CurContext && "DeclContext imbalance!");
  // If CurContext is a ObjC method, getParent() will return NULL.
  CurContext = isa<ObjCMethodDecl>(CurContext)
               ? Context.getTranslationUnitDecl()
                 :  CurContext->getParent();
}

/// Add this decl to the scope shadowed decl chains.
void Sema::PushOnScopeChains(NamedDecl *D, Scope *S) {
  S->AddDecl(D);

  // C++ [basic.scope]p4:
  //   -- exactly one declaration shall declare a class name or
  //   enumeration name that is not a typedef name and the other
  //   declarations shall all refer to the same object or
  //   enumerator, or all refer to functions and function templates;
  //   in this case the class name or enumeration name is hidden.
  if (TagDecl *TD = dyn_cast<TagDecl>(D)) {
    // We are pushing the name of a tag (enum or class).
    IdentifierResolver::ctx_iterator
      CIT = IdResolver.ctx_begin(TD->getIdentifier(), TD->getDeclContext());
    if (CIT != IdResolver.ctx_end(TD->getIdentifier()) &&
        IdResolver.isDeclInScope(*CIT, TD->getDeclContext(), S)) {
      // There is already a declaration with the same name in the same
      // scope. It must be found before we find the new declaration,
      // so swap the order on the shadowed declaration chain.

      IdResolver.AddShadowedDecl(TD, *CIT);
      return;
    }
  }

  IdResolver.AddDecl(D);
}

void Sema::ActOnPopScope(SourceLocation Loc, Scope *S) {
  if (S->decl_empty()) return;
  assert((S->getFlags() & Scope::DeclScope) &&"Scope shouldn't contain decls!");

  for (Scope::decl_iterator I = S->decl_begin(), E = S->decl_end();
       I != E; ++I) {
    Decl *TmpD = static_cast<Decl*>(*I);
    assert(TmpD && "This decl didn't get pushed??");

    if (isa<CXXFieldDecl>(TmpD)) continue;

    assert(isa<ScopedDecl>(TmpD) && "Decl isn't ScopedDecl?");
    ScopedDecl *D = cast<ScopedDecl>(TmpD);
    
    IdentifierInfo *II = D->getIdentifier();
    if (!II) continue;
    
    // We only want to remove the decls from the identifier decl chains for local
    // scopes, when inside a function/method.
    if (S->getFnParent() != 0)
      IdResolver.RemoveDecl(D);

    // Chain this decl to the containing DeclContext.
    D->setNext(CurContext->getDeclChain());
    CurContext->setDeclChain(D);
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

/// LookupDecl - Look up the inner-most declaration in the specified
/// namespace.
Decl *Sema::LookupDecl(const IdentifierInfo *II, unsigned NSI,
                       Scope *S, bool enableLazyBuiltinCreation) {
  if (II == 0) return 0;
  unsigned NS = NSI;
  if (getLangOptions().CPlusPlus && (NS & Decl::IDNS_Ordinary))
    NS |= Decl::IDNS_Tag;

  // Scan up the scope chain looking for a decl that matches this identifier
  // that is in the appropriate namespace.  This search should not take long, as
  // shadowing of names is uncommon, and deep shadowing is extremely uncommon.
  for (IdentifierResolver::iterator
       I = IdResolver.begin(II, CurContext), E = IdResolver.end(II); I != E; ++I)
    if ((*I)->getIdentifierNamespace() & NS)
      return *I;

  // If we didn't find a use of this identifier, and if the identifier
  // corresponds to a compiler builtin, create the decl object for the builtin
  // now, injecting it into translation unit scope, and return it.
  if (NS & Decl::IDNS_Ordinary) {
    if (enableLazyBuiltinCreation) {
      // If this is a builtin on this (or all) targets, create the decl.
      if (unsigned BuiltinID = II->getBuiltinID())
        return LazilyCreateBuiltin((IdentifierInfo *)II, BuiltinID, S);
    }
    if (getLangOptions().ObjC1) {
      // @interface and @compatibility_alias introduce typedef-like names.
      // Unlike typedef's, they can only be introduced at file-scope (and are 
      // therefore not scoped decls). They can, however, be shadowed by
      // other names in IDNS_Ordinary.
      ObjCInterfaceDeclsTy::iterator IDI = ObjCInterfaceDecls.find(II);
      if (IDI != ObjCInterfaceDecls.end())
        return IDI->second;
      ObjCAliasTy::iterator I = ObjCAliasDecls.find(II);
      if (I != ObjCAliasDecls.end())
        return I->second->getClassInterface();
    }
  }
  return 0;
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
ScopedDecl *Sema::LazilyCreateBuiltin(IdentifierInfo *II, unsigned bid,
                                      Scope *S) {
  Builtin::ID BID = (Builtin::ID)bid;

  if (BID == Builtin::BI__builtin_va_start ||
      BID == Builtin::BI__builtin_va_copy ||
      BID == Builtin::BI__builtin_va_end)
    InitBuiltinVaListType();
    
  QualType R = Context.BuiltinInfo.GetBuiltinType(BID, Context);  
  FunctionDecl *New = FunctionDecl::Create(Context,
                                           Context.getTranslationUnitDecl(),
                                           SourceLocation(), II, R,
                                           FunctionDecl::Extern, false, 0);
  
  // Create Decl objects for each parameter, adding them to the
  // FunctionDecl.
  if (FunctionTypeProto *FT = dyn_cast<FunctionTypeProto>(R)) {
    llvm::SmallVector<ParmVarDecl*, 16> Params;
    for (unsigned i = 0, e = FT->getNumArgs(); i != e; ++i)
      Params.push_back(ParmVarDecl::Create(Context, New, SourceLocation(), 0,
                                           FT->getArgType(i), VarDecl::None, 0,
                                           0));
    New->setParams(&Params[0], Params.size());
  }
  
  
  
  // TUScope is the translation-unit scope to insert this function into.
  PushOnScopeChains(New, TUScope);
  return New;
}

/// MergeTypeDefDecl - We just parsed a typedef 'New' which has the same name
/// and scope as a previous declaration 'Old'.  Figure out how to resolve this
/// situation, merging decls or emitting diagnostics as appropriate.
///
TypedefDecl *Sema::MergeTypeDefDecl(TypedefDecl *New, Decl *OldD) {
  // Verify the old decl was also a typedef.
  TypedefDecl *Old = dyn_cast<TypedefDecl>(OldD);
  if (!Old) {
    Diag(New->getLocation(), diag::err_redefinition_different_kind,
         New->getName());
    Diag(OldD->getLocation(), diag::err_previous_definition);
    return New;
  }
  
  // Allow multiple definitions for ObjC built-in typedefs.
  // FIXME: Verify the underlying types are equivalent!
  if (getLangOptions().ObjC1 && isBuiltinObjCType(New))
    return Old;

  if (getLangOptions().Microsoft) return New;

  // Redeclaration of a type is a constraint violation (6.7.2.3p1).
  // Apparently GCC, Intel, and Sun all silently ignore the redeclaration if
  // *either* declaration is in a system header. The code below implements
  // this adhoc compatibility rule. FIXME: The following code will not
  // work properly when compiling ".i" files (containing preprocessed output).
  SourceManager &SrcMgr = Context.getSourceManager();
  HeaderSearch &HdrInfo = PP.getHeaderSearchInfo();
  const FileEntry *OldDeclFile = SrcMgr.getFileEntryForLoc(Old->getLocation());
  if (OldDeclFile) {
    DirectoryLookup::DirType OldDirType = HdrInfo.getFileDirFlavor(OldDeclFile);
    // Allow reclarations in both SystemHeaderDir and ExternCSystemHeaderDir.
    if (OldDirType != DirectoryLookup::NormalHeaderDir)
      return New;
  }
  const FileEntry *NewDeclFile = SrcMgr.getFileEntryForLoc(New->getLocation());
  if (NewDeclFile) {
    DirectoryLookup::DirType NewDirType = HdrInfo.getFileDirFlavor(NewDeclFile);
    // Allow reclarations in both SystemHeaderDir and ExternCSystemHeaderDir.
    if (NewDirType != DirectoryLookup::NormalHeaderDir)
      return New;
  }

  Diag(New->getLocation(), diag::err_redefinition, New->getName());
  Diag(Old->getLocation(), diag::err_previous_definition);
  return New;
}

/// DeclhasAttr - returns true if decl Declaration already has the target attribute.
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
/// Redeclaration will be set true if thisNew is a redeclaration OldD.
FunctionDecl *
Sema::MergeFunctionDecl(FunctionDecl *New, Decl *OldD, bool &Redeclaration) {
  Redeclaration = false;
  // Verify the old decl was also a function.
  FunctionDecl *Old = dyn_cast<FunctionDecl>(OldD);
  if (!Old) {
    Diag(New->getLocation(), diag::err_redefinition_different_kind,
         New->getName());
    Diag(OldD->getLocation(), diag::err_previous_definition);
    return New;
  }
  
  QualType OldQType = Context.getCanonicalType(Old->getType());
  QualType NewQType = Context.getCanonicalType(New->getType());
  
  // C++ [dcl.fct]p3:
  //   All declarations for a function shall agree exactly in both the
  //   return type and the parameter-type-list.
  if (getLangOptions().CPlusPlus && OldQType == NewQType) {
    MergeAttributes(New, Old);
    Redeclaration = true;
    return MergeCXXFunctionDecl(New, Old);
  }

  // C: Function types need to be compatible, not identical. This handles
  // duplicate function decls like "void f(int); void f(enum X);" properly.
  if (!getLangOptions().CPlusPlus &&
      Context.functionTypesAreCompatible(OldQType, NewQType)) {
    MergeAttributes(New, Old);
    Redeclaration = true;
    return New;
  }

  // A function that has already been declared has been redeclared or defined
  // with a different type- show appropriate diagnostic
  diag::kind PrevDiag;
  if (Old->isThisDeclarationADefinition())
    PrevDiag = diag::err_previous_definition;
  else if (Old->isImplicit())
    PrevDiag = diag::err_previous_implicit_declaration;
  else 
    PrevDiag = diag::err_previous_declaration;

  // TODO: CHECK FOR CONFLICTS, multiple decls with same name in one scope.
  // TODO: This is totally simplistic.  It should handle merging functions
  // together etc, merging extern int X; int X; ...
  Diag(New->getLocation(), diag::err_conflicting_types, New->getName());
  Diag(Old->getLocation(), PrevDiag);
  return New;
}

/// equivalentArrayTypes - Used to determine whether two array types are 
/// equivalent.
/// We need to check this explicitly as an incomplete array definition is
/// considered a VariableArrayType, so will not match a complete array 
/// definition that would be otherwise equivalent.
static bool areEquivalentArrayTypes(QualType NewQType, QualType OldQType) {
  const ArrayType *NewAT = NewQType->getAsArrayType();
  const ArrayType *OldAT = OldQType->getAsArrayType();

  if (!NewAT || !OldAT)
    return false;
  
  // If either (or both) array types in incomplete we need to strip off the
  // outer VariableArrayType.  Once the outer VAT is removed the remaining
  // types must be identical if the array types are to be considered 
  // equivalent.
  // eg. int[][1] and int[1][1] become
  //     VAT(null, CAT(1, int)) and CAT(1, CAT(1, int))
  // removing the outermost VAT gives
  //     CAT(1, int) and CAT(1, int)
  // which are equal, therefore the array types are equivalent.
  if (NewAT->isIncompleteArrayType() || OldAT->isIncompleteArrayType()) {
    if (NewAT->getIndexTypeQualifier() != OldAT->getIndexTypeQualifier())
      return false;
    NewQType = NewAT->getElementType().getCanonicalType();
    OldQType = OldAT->getElementType().getCanonicalType();
  }
  
  return NewQType == OldQType;
}

/// MergeVarDecl - We just parsed a variable 'New' which has the same name
/// and scope as a previous declaration 'Old'.  Figure out how to resolve this
/// situation, merging decls or emitting diagnostics as appropriate.
///
/// FIXME: Need to carefully consider tentative definition rules (C99 6.9.2p2).
/// For example, we incorrectly complain about i1, i4 from C99 6.9.2p4.
/// 
VarDecl *Sema::MergeVarDecl(VarDecl *New, Decl *OldD) {
  // Verify the old decl was also a variable.
  VarDecl *Old = dyn_cast<VarDecl>(OldD);
  if (!Old) {
    Diag(New->getLocation(), diag::err_redefinition_different_kind,
         New->getName());
    Diag(OldD->getLocation(), diag::err_previous_definition);
    return New;
  }

  MergeAttributes(New, Old);

  // Verify the types match.
  QualType OldCType = Context.getCanonicalType(Old->getType());
  QualType NewCType = Context.getCanonicalType(New->getType());
  if (OldCType != NewCType && !areEquivalentArrayTypes(NewCType, OldCType)) {
    Diag(New->getLocation(), diag::err_redefinition, New->getName());
    Diag(Old->getLocation(), diag::err_previous_definition);
    return New;
  }
  // C99 6.2.2p4: Check if we have a static decl followed by a non-static.
  if (New->getStorageClass() == VarDecl::Static &&
      (Old->getStorageClass() == VarDecl::None ||
       Old->getStorageClass() == VarDecl::Extern)) {
    Diag(New->getLocation(), diag::err_static_non_static, New->getName());
    Diag(Old->getLocation(), diag::err_previous_definition);
    return New;
  }
  // C99 6.2.2p4: Check if we have a non-static decl followed by a static.
  if (New->getStorageClass() != VarDecl::Static &&
      Old->getStorageClass() == VarDecl::Static) {
    Diag(New->getLocation(), diag::err_non_static_static, New->getName());
    Diag(Old->getLocation(), diag::err_previous_definition);
    return New;
  }
  // We've verified the types match, now handle "tentative" definitions.
  if (Old->isFileVarDecl() && New->isFileVarDecl()) {
    // Handle C "tentative" external object definitions (C99 6.9.2).
    bool OldIsTentative = false;
    bool NewIsTentative = false;
    
    if (!Old->getInit() &&
        (Old->getStorageClass() == VarDecl::None ||
         Old->getStorageClass() == VarDecl::Static))
      OldIsTentative = true;
      
    // FIXME: this check doesn't work (since the initializer hasn't been
    // attached yet). This check should be moved to FinalizeDeclaratorGroup.
    // Unfortunately, by the time we get to FinializeDeclaratorGroup, we've 
    // thrown out the old decl.
    if (!New->getInit() &&
        (New->getStorageClass() == VarDecl::None ||
         New->getStorageClass() == VarDecl::Static))
      ; // change to NewIsTentative = true; once the code is moved.
    
    if (NewIsTentative || OldIsTentative)
      return New;
  }
  // Handle __private_extern__ just like extern.
  if (Old->getStorageClass() != VarDecl::Extern &&
      Old->getStorageClass() != VarDecl::PrivateExtern &&
      New->getStorageClass() != VarDecl::Extern &&
      New->getStorageClass() != VarDecl::PrivateExtern) {
    Diag(New->getLocation(), diag::err_redefinition, New->getName());
    Diag(Old->getLocation(), diag::err_previous_definition);
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
    if (Param->getType()->isIncompleteType() &&
        !Param->isInvalidDecl()) {
      Diag(Param->getLocation(), diag::err_typecheck_decl_incomplete_type,
           Param->getType().getAsString());
      Param->setInvalidDecl();
      HasInvalidParm = true;
    }
  }

  return HasInvalidParm;
}

/// CreateImplicitParameter - Creates an implicit function parameter
/// in the scope S and with the given type. This routine is used, for
/// example, to create the implicit "self" parameter in an Objective-C
/// method.
ImplicitParamDecl *
Sema::CreateImplicitParameter(Scope *S, IdentifierInfo *Id, 
                              SourceLocation IdLoc, QualType Type) {
  ImplicitParamDecl *New = ImplicitParamDecl::Create(Context, CurContext,
      IdLoc, Id, Type, 0);
  if (Id)
    PushOnScopeChains(New, S);

  return New;
}

/// ParsedFreeStandingDeclSpec - This method is invoked when a declspec with
/// no declarator (e.g. "struct foo;") is parsed.
Sema::DeclTy *Sema::ParsedFreeStandingDeclSpec(Scope *S, DeclSpec &DS) {
  // TODO: emit error on 'int;' or 'const enum foo;'.
  // TODO: emit error on 'typedef int;'
  // if (!DS.isMissingDeclaratorOk()) Diag(...);
  
  return dyn_cast_or_null<TagDecl>(static_cast<Decl *>(DS.getTypeRep()));
}

bool Sema::CheckSingleInitializer(Expr *&Init, QualType DeclType) {  
  // Get the type before calling CheckSingleAssignmentConstraints(), since
  // it can promote the expression.
  QualType InitType = Init->getType(); 
  
  AssignConvertType ConvTy = CheckSingleAssignmentConstraints(DeclType, Init);
  return DiagnoseAssignmentResult(ConvTy, Init->getLocStart(), DeclType,
                                  InitType, Init, "initializing");
}

bool Sema::CheckStringLiteralInit(StringLiteral *strLiteral, QualType &DeclT) {
  if (const IncompleteArrayType *IAT = DeclT->getAsIncompleteArrayType()) {
    // C99 6.7.8p14. We have an array of character type with unknown size 
    // being initialized to a string literal.
    llvm::APSInt ConstVal(32);
    ConstVal = strLiteral->getByteLength() + 1;
    // Return a new array type (C99 6.7.8p22).
    DeclT = Context.getConstantArrayType(IAT->getElementType(), ConstVal, 
                                         ArrayType::Normal, 0);
  } else if (const ConstantArrayType *CAT = DeclT->getAsConstantArrayType()) {
    // C99 6.7.8p14. We have an array of character type with known size.
    if (strLiteral->getByteLength() > (unsigned)CAT->getMaximumElements())
      Diag(strLiteral->getSourceRange().getBegin(),
           diag::warn_initializer_string_for_char_array_too_long,
           strLiteral->getSourceRange());
  } else {
    assert(0 && "HandleStringLiteralInit(): Invalid array type");
  }
  // Set type from "char *" to "constant array of char".
  strLiteral->setType(DeclT);
  // For now, we always return false (meaning success).
  return false;
}

StringLiteral *Sema::IsStringLiteralInit(Expr *Init, QualType DeclType) {
  const ArrayType *AT = DeclType->getAsArrayType();
  if (AT && AT->getElementType()->isCharType()) {
    return dyn_cast<StringLiteral>(Init);
  }
  return 0;
}

bool Sema::CheckInitializerTypes(Expr *&Init, QualType &DeclType) {  
  // C99 6.7.8p3: The type of the entity to be initialized shall be an array
  // of unknown size ("[]") or an object type that is not a variable array type.
  if (const VariableArrayType *VAT = DeclType->getAsVariableArrayType())
    return Diag(VAT->getSizeExpr()->getLocStart(), 
                diag::err_variable_object_no_init, 
                VAT->getSizeExpr()->getSourceRange());
  
  InitListExpr *InitList = dyn_cast<InitListExpr>(Init);
  if (!InitList) {
    // FIXME: Handle wide strings
    if (StringLiteral *strLiteral = IsStringLiteralInit(Init, DeclType))
      return CheckStringLiteralInit(strLiteral, DeclType);

    if (DeclType->isArrayType())
      return Diag(Init->getLocStart(),
                  diag::err_array_init_list_required, 
                  Init->getSourceRange());

    return CheckSingleInitializer(Init, DeclType);
  }

  InitListChecker CheckInitList(this, InitList, DeclType);
  return CheckInitList.HadError();
}

Sema::DeclTy *
Sema::ActOnDeclarator(Scope *S, Declarator &D, DeclTy *lastDecl) {
  ScopedDecl *LastDeclarator = dyn_cast_or_null<ScopedDecl>((Decl *)lastDecl);
  IdentifierInfo *II = D.getIdentifier();
  
  // All of these full declarators require an identifier.  If it doesn't have
  // one, the ParsedFreeStandingDeclSpec action should be used.
  if (II == 0) {
    Diag(D.getDeclSpec().getSourceRange().getBegin(),
         diag::err_declarator_need_ident,
         D.getDeclSpec().getSourceRange(), D.getSourceRange());
    return 0;
  }
  
  // The scope passed in may not be a decl scope.  Zip up the scope tree until
  // we find one that is.
  while ((S->getFlags() & Scope::DeclScope) == 0)
    S = S->getParent();
  
  // See if this is a redefinition of a variable in the same scope.
  Decl *PrevDecl = LookupDecl(II, Decl::IDNS_Ordinary, S);
  ScopedDecl *New;
  bool InvalidDecl = false;
 
  // In C++, the previous declaration we find might be a tag type
  // (class or enum). In this case, the new declaration will hide the
  // tag type. 
  if (PrevDecl && PrevDecl->getIdentifierNamespace() == Decl::IDNS_Tag)
    PrevDecl = 0;

  QualType R = GetTypeForDeclarator(D, S);
  assert(!R.isNull() && "GetTypeForDeclarator() returned null type");

  if (D.getDeclSpec().getStorageClassSpec() == DeclSpec::SCS_typedef) {
    // Check that there are no default arguments (C++ only).
    if (getLangOptions().CPlusPlus)
      CheckExtraCXXDefaultArguments(D);

    TypedefDecl *NewTD = ParseTypedefDecl(S, D, R, LastDeclarator);
    if (!NewTD) return 0;

    // Handle attributes prior to checking for duplicates in MergeVarDecl
    HandleDeclAttributes(NewTD, D.getDeclSpec().getAttributes(),
                         D.getAttributes());
    // Merge the decl with the existing one if appropriate. If the decl is
    // in an outer scope, it isn't the same thing.
    if (PrevDecl && IdResolver.isDeclInScope(PrevDecl, CurContext, S)) {
      NewTD = MergeTypeDefDecl(NewTD, PrevDecl);
      if (NewTD == 0) return 0;
    }
    New = NewTD;
    if (S->getFnParent() == 0) {
      // C99 6.7.7p2: If a typedef name specifies a variably modified type
      // then it shall have block scope.
      if (NewTD->getUnderlyingType()->isVariablyModifiedType()) {
        // FIXME: Diagnostic needs to be fixed.
        Diag(D.getIdentifierLoc(), diag::err_typecheck_illegal_vla);
        InvalidDecl = true;
      }
    }
  } else if (R.getTypePtr()->isFunctionType()) {
    FunctionDecl::StorageClass SC = FunctionDecl::None;
    switch (D.getDeclSpec().getStorageClassSpec()) {
      default: assert(0 && "Unknown storage class!");
      case DeclSpec::SCS_auto:        
      case DeclSpec::SCS_register:
        Diag(D.getIdentifierLoc(), diag::err_typecheck_sclass_func,
             R.getAsString());
        InvalidDecl = true;
        break;
      case DeclSpec::SCS_unspecified: SC = FunctionDecl::None; break;
      case DeclSpec::SCS_extern:      SC = FunctionDecl::Extern; break;
      case DeclSpec::SCS_static:      SC = FunctionDecl::Static; break;
      case DeclSpec::SCS_private_extern: SC = FunctionDecl::PrivateExtern;break;
    }

    bool isInline = D.getDeclSpec().isInlineSpecified();
    FunctionDecl *NewFD = FunctionDecl::Create(Context, CurContext,
                                               D.getIdentifierLoc(),
                                               II, R, SC, isInline,
                                               LastDeclarator);
    // Handle attributes.
    HandleDeclAttributes(NewFD, D.getDeclSpec().getAttributes(),
                         D.getAttributes());

    // Copy the parameter declarations from the declarator D to
    // the function declaration NewFD, if they are available.
    if (D.getNumTypeObjects() > 0 &&
        D.getTypeObject(0).Fun.hasPrototype) {
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

      } else {
        for (unsigned i = 0, e = FTI.NumArgs; i != e; ++i)
          Params.push_back((ParmVarDecl *)FTI.ArgInfo[i].Param);
      }
  
      NewFD->setParams(&Params[0], Params.size());
    }

    // Merge the decl with the existing one if appropriate. Since C functions
    // are in a flat namespace, make sure we consider decls in outer scopes.
    if (PrevDecl &&
        (!getLangOptions().CPlusPlus ||
         IdResolver.isDeclInScope(PrevDecl, CurContext, S)) ) {
      bool Redeclaration = false;
      NewFD = MergeFunctionDecl(NewFD, PrevDecl, Redeclaration);
      if (NewFD == 0) return 0;
      if (Redeclaration) {
        NewFD->setPreviousDeclaration(cast<FunctionDecl>(PrevDecl));
      }
    }
    New = NewFD;

    // In C++, check default arguments now that we have merged decls.
    if (getLangOptions().CPlusPlus)
      CheckCXXDefaultArguments(NewFD);
  } else {
    // Check that there are no default arguments (C++ only).
    if (getLangOptions().CPlusPlus)
      CheckExtraCXXDefaultArguments(D);

    if (R.getTypePtr()->isObjCInterfaceType()) {
      Diag(D.getIdentifierLoc(), diag::err_statically_allocated_object,
           D.getIdentifier()->getName());
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
    }    
    if (S->getFnParent() == 0) {
      // C99 6.9p2: The storage-class specifiers auto and register shall not
      // appear in the declaration specifiers in an external declaration.
      if (SC == VarDecl::Auto || SC == VarDecl::Register) {
        Diag(D.getIdentifierLoc(), diag::err_typecheck_sclass_fscope,
             R.getAsString());
        InvalidDecl = true;
      }
      NewVD = VarDecl::Create(Context, CurContext, D.getIdentifierLoc(),
                              II, R, SC, LastDeclarator);
    } else {
      NewVD = VarDecl::Create(Context, CurContext, D.getIdentifierLoc(),
                              II, R, SC, LastDeclarator);
    }
    // Handle attributes prior to checking for duplicates in MergeVarDecl
    HandleDeclAttributes(NewVD, D.getDeclSpec().getAttributes(),
                         D.getAttributes());

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
    if (PrevDecl && IdResolver.isDeclInScope(PrevDecl, CurContext, S)) {
      NewVD = MergeVarDecl(NewVD, PrevDecl);
      if (NewVD == 0) return 0;
    }
    New = NewVD;
  }
  
  // If this has an identifier, add it to the scope stack.
  if (II)
    PushOnScopeChains(New, S);
  // If any semantic error occurred, mark the decl as invalid.
  if (D.getInvalidType() || InvalidDecl)
    New->setInvalidDecl();
  
  return New;
}

bool Sema::CheckAddressConstantExpressionLValue(const Expr* Init) {
  switch (Init->getStmtClass()) {
  default:
    Diag(Init->getExprLoc(),
         diag::err_init_element_not_constant, Init->getSourceRange());
    return true;
  case Expr::ParenExprClass: {
    const ParenExpr* PE = cast<ParenExpr>(Init);
    return CheckAddressConstantExpressionLValue(PE->getSubExpr());
  }
  case Expr::CompoundLiteralExprClass:
    return cast<CompoundLiteralExpr>(Init)->isFileScope();
  case Expr::DeclRefExprClass: {
    const Decl *D = cast<DeclRefExpr>(Init)->getDecl();
    if (const VarDecl *VD = dyn_cast<VarDecl>(D)) {
      if (VD->hasGlobalStorage())
        return false;
      Diag(Init->getExprLoc(),
           diag::err_init_element_not_constant, Init->getSourceRange());
      return true;
    }
    if (isa<FunctionDecl>(D))
      return false;
    Diag(Init->getExprLoc(),
         diag::err_init_element_not_constant, Init->getSourceRange());
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
  case Expr::PreDefinedExprClass:
    return false;
  case Expr::UnaryOperatorClass: {
    const UnaryOperator *Exp = cast<UnaryOperator>(Init);

    // C99 6.6p9
    if (Exp->getOpcode() == UnaryOperator::Deref)
      return CheckAddressConstantExpression(Exp->getSubExpr());

    Diag(Init->getExprLoc(),
         diag::err_init_element_not_constant, Init->getSourceRange());
    return true;
  }
  }
}

bool Sema::CheckAddressConstantExpression(const Expr* Init) {
  switch (Init->getStmtClass()) {
  default:
    Diag(Init->getExprLoc(),
         diag::err_init_element_not_constant, Init->getSourceRange());
    return true;
  case Expr::ParenExprClass: {
    const ParenExpr* PE = cast<ParenExpr>(Init);
    return CheckAddressConstantExpression(PE->getSubExpr());
  }
  case Expr::StringLiteralClass:
  case Expr::ObjCStringLiteralClass:
    return false;
  case Expr::CallExprClass: {
    const CallExpr *CE = cast<CallExpr>(Init);
    if (CE->isBuiltinConstantExpr())
      return false;
    Diag(Init->getExprLoc(),
         diag::err_init_element_not_constant, Init->getSourceRange());
    return true;
  }
  case Expr::UnaryOperatorClass: {
    const UnaryOperator *Exp = cast<UnaryOperator>(Init);

    // C99 6.6p9
    if (Exp->getOpcode() == UnaryOperator::AddrOf)
      return CheckAddressConstantExpressionLValue(Exp->getSubExpr());

    if (Exp->getOpcode() == UnaryOperator::Extension)
      return CheckAddressConstantExpression(Exp->getSubExpr());
  
    Diag(Init->getExprLoc(),
         diag::err_init_element_not_constant, Init->getSourceRange());
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
  case Expr::ImplicitCastExprClass: {
    const Expr* SubExpr = cast<ImplicitCastExpr>(Init)->getSubExpr();

    // Check for implicit promotion
    if (SubExpr->getType()->isFunctionType() ||
        SubExpr->getType()->isArrayType())
      return CheckAddressConstantExpressionLValue(SubExpr);

    // Check for pointer->pointer cast
    if (SubExpr->getType()->isPointerType())
      return CheckAddressConstantExpression(SubExpr);

    if (SubExpr->getType()->isArithmeticType())
      return CheckArithmeticConstantExpression(SubExpr);

    Diag(Init->getExprLoc(),
         diag::err_init_element_not_constant, Init->getSourceRange());
    return true;
  }
  case Expr::CastExprClass: {
    const Expr* SubExpr = cast<CastExpr>(Init)->getSubExpr();

    // Check for pointer->pointer cast
    if (SubExpr->getType()->isPointerType())
      return CheckAddressConstantExpression(SubExpr);

    // FIXME: Should we pedwarn for (int*)(0+0)?
    if (SubExpr->getType()->isArithmeticType())
      return CheckArithmeticConstantExpression(SubExpr);

    Diag(Init->getExprLoc(),
         diag::err_init_element_not_constant, Init->getSourceRange());
    return true;
  }
  case Expr::ConditionalOperatorClass: {
    // FIXME: Should we pedwarn here?
    const ConditionalOperator *Exp = cast<ConditionalOperator>(Init);
    if (!Exp->getCond()->getType()->isArithmeticType()) {
      Diag(Init->getExprLoc(),
           diag::err_init_element_not_constant, Init->getSourceRange());
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
  case Expr::CastExprClass: {
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
    Diag(Init->getExprLoc(),
         diag::err_init_element_not_constant, Init->getSourceRange());
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
  case Expr::CallExprClass: {
    const CallExpr *CE = cast<CallExpr>(Init);
    if (CE->isBuiltinConstantExpr())
      return false;
    Diag(Init->getExprLoc(),
         diag::err_init_element_not_constant, Init->getSourceRange());
    return true;
  }
  case Expr::DeclRefExprClass: {
    const Decl *D = cast<DeclRefExpr>(Init)->getDecl();
    if (isa<EnumConstantDecl>(D))
      return false;
    Diag(Init->getExprLoc(),
         diag::err_init_element_not_constant, Init->getSourceRange());
    return true;
  }
  case Expr::CompoundLiteralExprClass:
    // Allow "(vector type){2,4}"; normal C constraints don't allow this,
    // but vectors are allowed to be magic.
    if (Init->getType()->isVectorType())
      return false;
    Diag(Init->getExprLoc(),
         diag::err_init_element_not_constant, Init->getSourceRange());
    return true;
  case Expr::UnaryOperatorClass: {
    const UnaryOperator *Exp = cast<UnaryOperator>(Init);
  
    switch (Exp->getOpcode()) {
    // Address, indirect, pre/post inc/dec, etc are not valid constant exprs.
    // See C99 6.6p3.
    default:
      Diag(Init->getExprLoc(),
           diag::err_init_element_not_constant, Init->getSourceRange());
      return true;
    case UnaryOperator::SizeOf:
    case UnaryOperator::AlignOf:
    case UnaryOperator::OffsetOf:
      // sizeof(E) is a constantexpr if and only if E is not evaluted.
      // See C99 6.5.3.4p2 and 6.6p3.
      if (Exp->getSubExpr()->getType()->isConstantSizeType())
        return false;
      Diag(Init->getExprLoc(),
           diag::err_init_element_not_constant, Init->getSourceRange());
      return true;
    case UnaryOperator::Extension:
    case UnaryOperator::LNot:
    case UnaryOperator::Plus:
    case UnaryOperator::Minus:
    case UnaryOperator::Not:
      return CheckArithmeticConstantExpression(Exp->getSubExpr());
    }
  }
  case Expr::SizeOfAlignOfTypeExprClass: {
    const SizeOfAlignOfTypeExpr *Exp = cast<SizeOfAlignOfTypeExpr>(Init);
    // Special check for void types, which are allowed as an extension
    if (Exp->getArgumentType()->isVoidType())
      return false;
    // alignof always evaluates to a constant.
    // FIXME: is sizeof(int[3.0]) a constant expression?
    if (Exp->isSizeOf() && !Exp->getArgumentType()->isConstantSizeType()) {
      Diag(Init->getExprLoc(),
           diag::err_init_element_not_constant, Init->getSourceRange());
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

    Diag(Init->getExprLoc(),
         diag::err_init_element_not_constant, Init->getSourceRange());
    return true;
  }
  case Expr::ImplicitCastExprClass:
  case Expr::CastExprClass: {
    const Expr *SubExpr;
    if (const CastExpr *C = dyn_cast<CastExpr>(Init)) {
      SubExpr = C->getSubExpr();
    } else {
      SubExpr = cast<ImplicitCastExpr>(Init)->getSubExpr();
    }

    if (SubExpr->getType()->isArithmeticType())
      return CheckArithmeticConstantExpression(SubExpr);

    Diag(Init->getExprLoc(),
         diag::err_init_element_not_constant, Init->getSourceRange());
    return true;
  }
  case Expr::ConditionalOperatorClass: {
    const ConditionalOperator *Exp = cast<ConditionalOperator>(Init);
    if (CheckArithmeticConstantExpression(Exp->getCond()))
      return true;
    if (Exp->getLHS() &&
        CheckArithmeticConstantExpression(Exp->getLHS()))
      return true;
    return CheckArithmeticConstantExpression(Exp->getRHS());
  }
  }
}

bool Sema::CheckForConstantInitializer(Expr *Init, QualType DclT) {
  // Look through CXXDefaultArgExprs; they have no meaning in this context.
  if (CXXDefaultArgExpr* DAE = dyn_cast<CXXDefaultArgExpr>(Init))
    return CheckForConstantInitializer(DAE->getExpr(), DclT);

  if (Init->getType()->isReferenceType()) {
    // FIXME: Work out how the heck reference types work
    return false;
#if 0
    // A reference is constant if the address of the expression
    // is constant
    // We look through initlists here to simplify
    // CheckAddressConstantExpressionLValue.
    if (InitListExpr *Exp = dyn_cast<InitListExpr>(Init)) {
      assert(Exp->getNumInits() > 0 &&
             "Refernce initializer cannot be empty");
      Init = Exp->getInit(0);
    }
    return CheckAddressConstantExpressionLValue(Init);
#endif
  }

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

  if (Init->isNullPointerConstant(Context))
    return false;
  if (Init->getType()->isArithmeticType()) {
    QualType InitTy = Init->getType().getCanonicalType().getUnqualifiedType();
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
      if (ImplicitCastExpr* ICE = dyn_cast<ImplicitCastExpr>(Init))
        SubE = ICE->getSubExpr();
      else if (CastExpr* CE = dyn_cast<CastExpr>(Init))
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

  Diag(Init->getExprLoc(), diag::err_init_element_not_constant,
       Init->getSourceRange());
  return true;
}

void Sema::AddInitializerToDecl(DeclTy *dcl, ExprTy *init) {
  Decl *RealDecl = static_cast<Decl *>(dcl);
  Expr *Init = static_cast<Expr *>(init);
  assert(Init && "missing initializer");
  
  // If there is no declaration, there was an error parsing it.  Just ignore
  // the initializer.
  if (RealDecl == 0) {
    delete Init;
    return;
  }
  
  VarDecl *VDecl = dyn_cast<VarDecl>(RealDecl);
  if (!VDecl) {
    Diag(dyn_cast<ScopedDecl>(RealDecl)->getLocation(), 
         diag::err_illegal_initializer);
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
      if (CheckInitializerTypes(Init, DclT))
        VDecl->setInvalidDecl();
      if (SC == VarDecl::Static) // C99 6.7.8p4.
        CheckForConstantInitializer(Init, DclT);
    }
  } else if (VDecl->isFileVarDecl()) {
    if (VDecl->getStorageClass() == VarDecl::Extern)
      Diag(VDecl->getLocation(), diag::warn_extern_init);
    if (!VDecl->isInvalidDecl())
      if (CheckInitializerTypes(Init, DclT))
        VDecl->setInvalidDecl();
    
    // C99 6.7.8p4. All file scoped initializers need to be constant.
    CheckForConstantInitializer(Init, DclT);
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

/// The declarators are chained together backwards, reverse the list.
Sema::DeclTy *Sema::FinalizeDeclaratorGroup(Scope *S, DeclTy *group) {
  // Often we have single declarators, handle them quickly.
  Decl *GroupDecl = static_cast<Decl*>(group);
  if (GroupDecl == 0)
    return 0;
  
  ScopedDecl *Group = dyn_cast<ScopedDecl>(GroupDecl);
  ScopedDecl *NewGroup = 0;
  if (Group->getNextDeclarator() == 0) 
    NewGroup = Group;
  else { // reverse the list.
    while (Group) {
      ScopedDecl *Next = Group->getNextDeclarator();
      Group->setNextDeclarator(NewGroup);
      NewGroup = Group;
      Group = Next;
    }
  }
  // Perform semantic analysis that depends on having fully processed both
  // the declarator and initializer.
  for (ScopedDecl *ID = NewGroup; ID; ID = ID->getNextDeclarator()) {
    VarDecl *IDecl = dyn_cast<VarDecl>(ID);
    if (!IDecl)
      continue;
    QualType T = IDecl->getType();
    
    // C99 6.7.5.2p2: If an identifier is declared to be an object with 
    // static storage duration, it shall not have a variable length array.
    if ((IDecl->isFileVarDecl() || IDecl->isBlockVarDecl()) && 
        IDecl->getStorageClass() == VarDecl::Static) {
      if (T->getAsVariableArrayType()) {
        Diag(IDecl->getLocation(), diag::err_typecheck_illegal_vla);
        IDecl->setInvalidDecl();
      }
    }
    // Block scope. C99 6.7p7: If an identifier for an object is declared with
    // no linkage (C99 6.2.2p6), the type for the object shall be complete...
    if (IDecl->isBlockVarDecl() && 
        IDecl->getStorageClass() != VarDecl::Extern) {
      if (T->isIncompleteType() && !IDecl->isInvalidDecl()) {
        Diag(IDecl->getLocation(), diag::err_typecheck_decl_incomplete_type,
             T.getAsString());
        IDecl->setInvalidDecl();
      }
    }
    // File scope. C99 6.9.2p2: A declaration of an identifier for and 
    // object that has file scope without an initializer, and without a
    // storage-class specifier or with the storage-class specifier "static",
    // constitutes a tentative definition. Note: A tentative definition with
    // external linkage is valid (C99 6.2.2p5).
    if (IDecl && !IDecl->getInit() && 
        (IDecl->getStorageClass() == VarDecl::Static || 
         IDecl->getStorageClass() == VarDecl::None)) {
      if (T->isIncompleteArrayType()) {
        // C99 6.9.2 (p2, p5): Implicit initialization causes an incomplete
        // array to be completed. Don't issue a diagnostic.
      } else if (T->isIncompleteType() && !IDecl->isInvalidDecl()) {
        // C99 6.9.2p3: If the declaration of an identifier for an object is
        // a tentative definition and has internal linkage (C99 6.2.2p3), the  
        // declared type shall not be an incomplete type.
        Diag(IDecl->getLocation(), diag::err_typecheck_decl_incomplete_type,
             T.getAsString());
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
  DeclSpec &DS = D.getDeclSpec();
  
  // Verify C99 6.7.5.3p2: The only SCS allowed is 'register'.
  if (DS.getStorageClassSpec() != DeclSpec::SCS_unspecified &&
      DS.getStorageClassSpec() != DeclSpec::SCS_register) {
    Diag(DS.getStorageClassSpecLoc(),
         diag::err_invalid_storage_class_in_func_decl);
    DS.ClearStorageClassSpecs();
  }
  if (DS.isThreadSpecified()) {
    Diag(DS.getThreadSpecLoc(),
         diag::err_invalid_storage_class_in_func_decl);
    DS.ClearStorageClassSpecs();
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
  if (Decl *PrevDecl = LookupDecl(II, Decl::IDNS_Ordinary, S)) {
    if (S->isDeclScope(PrevDecl)) {
      Diag(D.getIdentifierLoc(), diag::err_param_redefinition,
           dyn_cast<NamedDecl>(PrevDecl)->getName());

      // Recover by removing the name
      II = 0;
      D.SetIdentifier(0, D.getIdentifierLoc());
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
                                         parmDeclType, VarDecl::None, 
                                         0, 0);
  
  if (D.getInvalidType())
    New->setInvalidDecl();
    
  if (II)
    PushOnScopeChains(New, S);

  HandleDeclAttributes(New, D.getDeclSpec().getAttributes(),
                       D.getAttributes());
  return New;

}

Sema::DeclTy *Sema::ActOnStartOfFunctionDef(Scope *FnBodyScope, Declarator &D) {
  assert(CurFunctionDecl == 0 && "Function parsing confused");
  assert(D.getTypeObject(0).Kind == DeclaratorChunk::Function &&
         "Not a function declarator!");
  DeclaratorChunk::FunctionTypeInfo &FTI = D.getTypeObject(0).Fun;

  // Verify 6.9.1p6: 'every identifier in the identifier list shall be declared'
  // for a K&R function.
  if (!FTI.hasPrototype) {
    for (unsigned i = 0, e = FTI.NumArgs; i != e; ++i) {
      if (FTI.ArgInfo[i].Param == 0) {
        Diag(FTI.ArgInfo[i].IdentLoc, diag::ext_param_not_declared,
             FTI.ArgInfo[i].Ident->getName());
        // Implicitly declare the argument as type 'int' for lack of a better
        // type.
        DeclSpec DS;
        const char* PrevSpec; // unused
        DS.SetTypeSpecType(DeclSpec::TST_int, FTI.ArgInfo[i].IdentLoc, 
                           PrevSpec);
        Declarator ParamD(DS, Declarator::KNRTypeListContext);
        ParamD.SetIdentifier(FTI.ArgInfo[i].Ident, FTI.ArgInfo[i].IdentLoc);
        FTI.ArgInfo[i].Param = ActOnParamDeclarator(FnBodyScope, ParamD);
      }
    }

    // Since this is a function definition, act as though we have information
    // about the arguments.
    if (FTI.NumArgs)
      FTI.hasPrototype = true;
  } else {
    // FIXME: Diagnose arguments without names in C. 
  }
  
  Scope *GlobalScope = FnBodyScope->getParent();

  // See if this is a redefinition.
  Decl *PrevDcl = LookupDecl(D.getIdentifier(), Decl::IDNS_Ordinary,
                             GlobalScope);
  if (PrevDcl && IdResolver.isDeclInScope(PrevDcl, CurContext)) {
    if (FunctionDecl *FD = dyn_cast<FunctionDecl>(PrevDcl)) {
      const FunctionDecl *Definition;
      if (FD->getBody(Definition)) {
        Diag(D.getIdentifierLoc(), diag::err_redefinition, 
             D.getIdentifier()->getName());
        Diag(Definition->getLocation(), diag::err_previous_definition);
      }
    }
  }
  Decl *decl = static_cast<Decl*>(ActOnDeclarator(GlobalScope, D, 0));
  FunctionDecl *FD = cast<FunctionDecl>(decl);
  CurFunctionDecl = FD;
  PushDeclContext(FD);
    
  // Check the validity of our function parameters
  CheckParmsForFunctionDef(FD);

  // Introduce our parameters into the function scope
  for (unsigned p = 0, NumParams = FD->getNumParams(); p < NumParams; ++p) {
    ParmVarDecl *Param = FD->getParamDecl(p);
    // If this has an identifier, add it to the scope stack.
    if (Param->getIdentifier())
      PushOnScopeChains(Param, FnBodyScope);
  }

  return FD;
}

Sema::DeclTy *Sema::ActOnFinishFunctionBody(DeclTy *D, StmtTy *Body) {
  Decl *dcl = static_cast<Decl *>(D);
  if (FunctionDecl *FD = dyn_cast<FunctionDecl>(dcl)) {
    FD->setBody((Stmt*)Body);
    assert(FD == CurFunctionDecl && "Function parsing confused");
    CurFunctionDecl = 0;
  } else if (ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(dcl)) {
    MD->setBody((Stmt*)Body);
    CurMethodDecl = 0;
  }  
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
      Diag(L->getIdentLoc(), diag::err_undeclared_label_use, L->getName());
      
      // At this point, we have gotos that use the bogus label.  Stitch it into
      // the function body so that they aren't leaked and that the AST is well
      // formed.
      if (Body) {
        L->setSubStmt(new NullStmt(L->getIdentLoc()));
        cast<CompoundStmt>((Stmt*)Body)->push_back(L);
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
ScopedDecl *Sema::ImplicitlyDefineFunction(SourceLocation Loc, 
                                           IdentifierInfo &II, Scope *S) {
  // Extension in C99.  Legal in C90, but warn about it.
  if (getLangOptions().C99)
    Diag(Loc, diag::ext_implicit_function_decl, II.getName());
  else
    Diag(Loc, diag::warn_implicit_function_decl, II.getName());
  
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
  D.AddTypeInfo(DeclaratorChunk::getFunction(false, false, 0, 0, Loc));
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
                                    ScopedDecl *LastDeclarator) {
  assert(D.getIdentifier() && "Wrong callback for declspec without declarator");
  assert(!T.isNull() && "GetTypeForDeclarator() returned null type");
  
  // Scope manipulation handled by caller.
  TypedefDecl *NewTD = TypedefDecl::Create(Context, CurContext,
                                           D.getIdentifierLoc(),
                                           D.getIdentifier(), 
                                           T, LastDeclarator);
  if (D.getInvalidType())
    NewTD->setInvalidDecl();
  return NewTD;
}

/// ActOnTag - This is invoked when we see 'struct foo' or 'struct {'.  In the
/// former case, Name will be non-null.  In the later case, Name will be null.
/// TagType indicates what kind of tag this is. TK indicates whether this is a
/// reference/declaration/definition of a tag.
Sema::DeclTy *Sema::ActOnTag(Scope *S, unsigned TagType, TagKind TK,
                             SourceLocation KWLoc, IdentifierInfo *Name,
                             SourceLocation NameLoc, AttributeList *Attr) {
  // If this is a use of an existing tag, it must have a name.
  assert((Name != 0 || TK == TK_Definition) &&
         "Nameless record must be a definition!");
  
  TagDecl::TagKind Kind;
  switch (TagType) {
  default: assert(0 && "Unknown tag type!");
  case DeclSpec::TST_struct: Kind = TagDecl::TK_struct; break;
  case DeclSpec::TST_union:  Kind = TagDecl::TK_union; break;
  case DeclSpec::TST_class:  Kind = TagDecl::TK_class; break;
  case DeclSpec::TST_enum:   Kind = TagDecl::TK_enum; break;
  }
  
  // If this is a named struct, check to see if there was a previous forward
  // declaration or definition.
  // Use ScopedDecl instead of TagDecl, because a NamespaceDecl may come up.
  if (ScopedDecl *PrevDecl = 
          dyn_cast_or_null<ScopedDecl>(LookupDecl(Name, Decl::IDNS_Tag, S))) {
    
    assert((isa<TagDecl>(PrevDecl) || isa<NamespaceDecl>(PrevDecl)) &&
            "unexpected Decl type");
    if (TagDecl *PrevTagDecl = dyn_cast<TagDecl>(PrevDecl)) {
      // If this is a use of a previous tag, or if the tag is already declared in
      // the same scope (so that the definition/declaration completes or
      // rementions the tag), reuse the decl.
      if (TK == TK_Reference ||
          IdResolver.isDeclInScope(PrevDecl, CurContext, S)) {
        // Make sure that this wasn't declared as an enum and now used as a struct
        // or something similar.
        if (PrevTagDecl->getTagKind() != Kind) {
          Diag(KWLoc, diag::err_use_with_wrong_tag, Name->getName());
          Diag(PrevDecl->getLocation(), diag::err_previous_use);
        }
        
        // If this is a use or a forward declaration, we're good.
        if (TK != TK_Definition)
          return PrevDecl;

        // Diagnose attempts to redefine a tag.
        if (PrevTagDecl->isDefinition()) {
          Diag(NameLoc, diag::err_redefinition, Name->getName());
          Diag(PrevDecl->getLocation(), diag::err_previous_definition);
          // If this is a redefinition, recover by making this struct be
          // anonymous, which will make any later references get the previous
          // definition.
          Name = 0;
        } else {
          // Okay, this is definition of a previously declared or referenced tag.
          // Move the location of the decl to be the definition site.
          PrevDecl->setLocation(NameLoc);
          return PrevDecl;
        }
      }
      // If we get here, this is a definition of a new struct type in a nested
      // scope, e.g. "struct foo; void bar() { struct foo; }", just create a new
      // type.
    } else {
      // The tag name clashes with a namespace name, issue an error and recover
      // by making this tag be anonymous.
      Diag(NameLoc, diag::err_redefinition_different_kind, Name->getName());
      Diag(PrevDecl->getLocation(), diag::err_previous_definition);
      Name = 0;
    }
  }
  
  // If there is an identifier, use the location of the identifier as the
  // location of the decl, otherwise use the location of the struct/union
  // keyword.
  SourceLocation Loc = NameLoc.isValid() ? NameLoc : KWLoc;
  
  // Otherwise, if this is the first time we've seen this tag, create the decl.
  TagDecl *New;
  if (Kind == TagDecl::TK_enum) {
    // FIXME: Tag decls should be chained to any simultaneous vardecls, e.g.:
    // enum X { A, B, C } D;    D should chain to X.
    New = EnumDecl::Create(Context, CurContext, Loc, Name, 0);
    // If this is an undefined enum, warn.
    if (TK != TK_Definition) Diag(Loc, diag::ext_forward_ref_enum);
  } else {
    // struct/union/class

    // FIXME: Tag decls should be chained to any simultaneous vardecls, e.g.:
    // struct X { int A; } D;    D should chain to X.
    New = RecordDecl::Create(Context, Kind, CurContext, Loc, Name, 0);
  }    
  
  // If this has an identifier, add it to the scope stack.
  if (Name) {
    // The scope passed in may not be a decl scope.  Zip up the scope tree until
    // we find one that is.
    while ((S->getFlags() & Scope::DeclScope) == 0)
      S = S->getParent();
    
    // Add it to the decl chain.
    PushOnScopeChains(New, S);
  }
  
  HandleDeclAttributes(New, Attr, 0);
  return New;
}

/// Collect the instance variables declared in an Objective-C object.  Used in
/// the creation of structures from objects using the @defs directive.
static void CollectIvars(ObjCInterfaceDecl *Class,
                         llvm::SmallVector<Sema::DeclTy*, 16> &ivars) {
  if (Class->getSuperClass())
    CollectIvars(Class->getSuperClass(), ivars);
  ivars.append(Class->ivar_begin(), Class->ivar_end());
}

/// Called whenever @defs(ClassName) is encountered in the source.  Inserts the
/// instance variables of ClassName into Decls.
void Sema::ActOnDefs(Scope *S, SourceLocation DeclStart, 
                     IdentifierInfo *ClassName,
                     llvm::SmallVector<DeclTy*, 16> &Decls) {
  // Check that ClassName is a valid class
  ObjCInterfaceDecl *Class = getObjCInterfaceDecl(ClassName);
  if (!Class) {
    Diag(DeclStart, diag::err_undef_interface, ClassName->getName());
    return;
  }
  // Collect the instance variables
  CollectIvars(Class, Decls);
}


static bool CalcFakeICEVal(const Expr* Expr,
                           llvm::APSInt& Result,
                           ASTContext& Context) {
  // Calculate the value of an expression that has a calculatable
  // value, but isn't an ICE. Currently, this only supports
  // a very narrow set of extensions, but it can be expanded if needed.
  if (const ParenExpr *PE = dyn_cast<ParenExpr>(Expr))
    return CalcFakeICEVal(PE->getSubExpr(), Result, Context);

  if (const CastExpr *CE = dyn_cast<CastExpr>(Expr)) {
    QualType CETy = CE->getType();
    if ((CETy->isIntegralType() && !CETy->isBooleanType()) ||
        CETy->isPointerType()) {
      if (CalcFakeICEVal(CE->getSubExpr(), Result, Context)) {
        Result.extOrTrunc(Context.getTypeSize(CETy));
        // FIXME: This assumes pointers are signed.
        Result.setIsSigned(CETy->isSignedIntegerType() ||
                           CETy->isPointerType());
        return true;
      }
    }
  }

  if (Expr->getType()->isIntegralType())
    return Expr->isIntegerConstantExpr(Result, Context);

  return false;
}

QualType Sema::TryFixInvalidVariablyModifiedType(QualType T) {
  // This method tries to turn a variable array into a constant
  // array even when the size isn't an ICE.  This is necessary
  // for compatibility with code that depends on gcc's buggy
  // constant expression folding, like struct {char x[(int)(char*)2];}
  if (const VariableArrayType* VLATy = dyn_cast<VariableArrayType>(T)) {
    llvm::APSInt Result(32);
    if (VLATy->getSizeExpr() &&
        CalcFakeICEVal(VLATy->getSizeExpr(), Result, Context) &&
        Result > llvm::APSInt(Result.getBitWidth(), Result.isUnsigned())) {
      return Context.getConstantArrayType(VLATy->getElementType(),
                                          Result, ArrayType::Normal, 0);
    }
  }
  return QualType();
}

/// ActOnField - Each field of a struct/union/class is passed into this in order
/// to create a FieldDecl object for it.
Sema::DeclTy *Sema::ActOnField(Scope *S,
                               SourceLocation DeclStart, 
                               Declarator &D, ExprTy *BitfieldWidth) {
  IdentifierInfo *II = D.getIdentifier();
  Expr *BitWidth = (Expr*)BitfieldWidth;
  SourceLocation Loc = DeclStart;
  if (II) Loc = D.getIdentifierLoc();
  
  // FIXME: Unnamed fields can be handled in various different ways, for
  // example, unnamed unions inject all members into the struct namespace!
  
  
  if (BitWidth) {
    // TODO: Validate.
    //printf("WARNING: BITFIELDS IGNORED!\n");
    
    // 6.7.2.1p3
    // 6.7.2.1p4
    
  } else {
    // Not a bitfield.

    // validate II.
    
  }
  
  QualType T = GetTypeForDeclarator(D, S);
  assert(!T.isNull() && "GetTypeForDeclarator() returned null type");
  bool InvalidDecl = false;

  // C99 6.7.2.1p8: A member of a structure or union may have any type other
  // than a variably modified type.
  if (T->isVariablyModifiedType()) {
    QualType FixedTy = TryFixInvalidVariablyModifiedType(T);
    if (!FixedTy.isNull()) {
      Diag(Loc, diag::warn_illegal_constant_array_size, Loc);
      T = FixedTy;
    } else {
      // FIXME: This diagnostic needs work
      Diag(Loc, diag::err_typecheck_illegal_vla, Loc);
      InvalidDecl = true;
    }
  }
  // FIXME: Chain fielddecls together.
  FieldDecl *NewFD = FieldDecl::Create(Context, Loc, II, T, BitWidth);
  
  HandleDeclAttributes(NewFD, D.getDeclSpec().getAttributes(),
                       D.getAttributes());

  if (D.getInvalidType() || InvalidDecl)
    NewFD->setInvalidDecl();
  return NewFD;
}

/// TranslateIvarVisibility - Translate visibility from a token ID to an 
///  AST enum value.
static ObjCIvarDecl::AccessControl
TranslateIvarVisibility(tok::ObjCKeywordKind ivarVisibility) {
  switch (ivarVisibility) {
    case tok::objc_private: return ObjCIvarDecl::Private;
    case tok::objc_public: return ObjCIvarDecl::Public;
    case tok::objc_protected: return ObjCIvarDecl::Protected;
    case tok::objc_package: return ObjCIvarDecl::Package;
    default: assert(false && "Unknown visitibility kind");
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
  
  
  if (BitWidth) {
    // TODO: Validate.
    //printf("WARNING: BITFIELDS IGNORED!\n");
    
    // 6.7.2.1p3
    // 6.7.2.1p4
    
  } else {
    // Not a bitfield.
    
    // validate II.
    
  }
  
  QualType T = GetTypeForDeclarator(D, S);
  assert(!T.isNull() && "GetTypeForDeclarator() returned null type");
  bool InvalidDecl = false;
  
  // C99 6.7.2.1p8: A member of a structure or union may have any type other
  // than a variably modified type.
  if (T->isVariablyModifiedType()) {
    // FIXME: This diagnostic needs work
    Diag(Loc, diag::err_typecheck_illegal_vla, Loc);
    InvalidDecl = true;
  }
  
  ObjCIvarDecl *NewID = ObjCIvarDecl::Create(Context, Loc, II, T);
  
  HandleDeclAttributes(NewID, D.getDeclSpec().getAttributes(),
                       D.getAttributes());
  
  if (D.getInvalidType() || InvalidDecl)
    NewID->setInvalidDecl();
  // If we have visibility info, make sure the AST is set accordingly.
  if (Visibility != tok::objc_not_keyword)
    NewID->setAccessControl(TranslateIvarVisibility(Visibility));
  return NewID;
}

void Sema::ActOnFields(Scope* S,
                       SourceLocation RecLoc, DeclTy *RecDecl,
                       DeclTy **Fields, unsigned NumFields,
                       SourceLocation LBrac, SourceLocation RBrac) {
  Decl *EnclosingDecl = static_cast<Decl*>(RecDecl);
  assert(EnclosingDecl && "missing record or interface decl");
  RecordDecl *Record = dyn_cast<RecordDecl>(EnclosingDecl);
  
  if (Record && Record->isDefinition()) {
    // Diagnose code like:
    //     struct S { struct S {} X; };
    // We discover this when we complete the outer S.  Reject and ignore the
    // outer S.
    Diag(Record->getLocation(), diag::err_nested_redefinition,
         Record->getKindName());
    Diag(RecLoc, diag::err_previous_definition);
    Record->setInvalidDecl();
    return;
  }
  // Verify that all the fields are okay.
  unsigned NumNamedMembers = 0;
  llvm::SmallVector<FieldDecl*, 32> RecFields;
  llvm::SmallSet<const IdentifierInfo*, 32> FieldIDs;
  
  for (unsigned i = 0; i != NumFields; ++i) {
    
    FieldDecl *FD = cast_or_null<FieldDecl>(static_cast<Decl*>(Fields[i]));
    assert(FD && "missing field decl");
    
    // Remember all fields.
    RecFields.push_back(FD);
    
    // Get the type for the field.
    Type *FDTy = FD->getType().getTypePtr();
      
    // C99 6.7.2.1p2 - A field may not be a function type.
    if (FDTy->isFunctionType()) {
      Diag(FD->getLocation(), diag::err_field_declared_as_function, 
           FD->getName());
      FD->setInvalidDecl();
      EnclosingDecl->setInvalidDecl();
      continue;
    }
    // C99 6.7.2.1p2 - A field may not be an incomplete type except...
    if (FDTy->isIncompleteType()) {
      if (!Record) {  // Incomplete ivar type is always an error.
        Diag(FD->getLocation(), diag::err_field_incomplete, FD->getName());
        FD->setInvalidDecl();
        EnclosingDecl->setInvalidDecl();
        continue;
      }
      if (i != NumFields-1 ||                   // ... that the last member ...
          !Record->isStruct() ||  // ... of a structure ...
          !FDTy->isArrayType()) {         //... may have incomplete array type.
        Diag(FD->getLocation(), diag::err_field_incomplete, FD->getName());
        FD->setInvalidDecl();
        EnclosingDecl->setInvalidDecl();
        continue;
      }
      if (NumNamedMembers < 1) {  //... must have more than named member ...
        Diag(FD->getLocation(), diag::err_flexible_array_empty_struct,
             FD->getName());
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
            Diag(FD->getLocation(), diag::err_variable_sized_type_in_struct,
                 FD->getName());
            FD->setInvalidDecl();
            EnclosingDecl->setInvalidDecl();
            continue;
          }
          // We support flexible arrays at the end of structs in other structs
          // as an extension.
          Diag(FD->getLocation(), diag::ext_flexible_array_in_struct,
               FD->getName());
          if (Record)
            Record->setHasFlexibleArrayMember(true);
        }
      }
    }
    /// A field cannot be an Objective-c object
    if (FDTy->isObjCInterfaceType()) {
      Diag(FD->getLocation(), diag::err_statically_allocated_object,
           FD->getName());
      FD->setInvalidDecl();
      EnclosingDecl->setInvalidDecl();
      continue;
    }
    // Keep track of the number of named members.
    if (IdentifierInfo *II = FD->getIdentifier()) {
      // Detect duplicate member names.
      if (!FieldIDs.insert(II)) {
        Diag(FD->getLocation(), diag::err_duplicate_member, II->getName());
        // Find the previous decl.
        SourceLocation PrevLoc;
        for (unsigned i = 0, e = RecFields.size(); ; ++i) {
          assert(i != e && "Didn't find previous def!");
          if (RecFields[i]->getIdentifier() == II) {
            PrevLoc = RecFields[i]->getLocation();
            break;
          }
        }
        Diag(PrevLoc, diag::err_previous_definition);
        FD->setInvalidDecl();
        EnclosingDecl->setInvalidDecl();
        continue;
      }
      ++NumNamedMembers;
    }
  }
 
  // Okay, we successfully defined 'Record'.
  if (Record) {
    Record->defineBody(&RecFields[0], RecFields.size());
    Consumer.HandleTagDeclDefinition(Record);
  } else {
    ObjCIvarDecl **ClsFields = reinterpret_cast<ObjCIvarDecl**>(&RecFields[0]);
    if (ObjCInterfaceDecl *ID = dyn_cast<ObjCInterfaceDecl>(EnclosingDecl))
      ID->addInstanceVariablesToClass(ClsFields, RecFields.size(), RBrac);
    else if (ObjCImplementationDecl *IMPDecl = 
               dyn_cast<ObjCImplementationDecl>(EnclosingDecl)) {
      assert(IMPDecl && "ActOnFields - missing ObjCImplementationDecl");
      IMPDecl->ObjCAddInstanceVariablesToClassImpl(ClsFields, RecFields.size());
      CheckImplementationIvars(IMPDecl, ClsFields, RecFields.size(), RBrac);
    }
  }
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
  while ((S->getFlags() & Scope::DeclScope) == 0)
    S = S->getParent();
  
  // Verify that there isn't already something declared with this name in this
  // scope.
  if (Decl *PrevDecl = LookupDecl(Id, Decl::IDNS_Ordinary, S)) {
    if (IdResolver.isDeclInScope(PrevDecl, CurContext, S)) {
      if (isa<EnumConstantDecl>(PrevDecl))
        Diag(IdLoc, diag::err_redefinition_of_enumerator, Id->getName());
      else
        Diag(IdLoc, diag::err_redefinition, Id->getName());
      Diag(PrevDecl->getLocation(), diag::err_previous_definition);
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
    if (!Val->isIntegerConstantExpr(EnumVal, Context, &ExpLoc)) {
      Diag(ExpLoc, diag::err_enum_value_not_integer_constant_expr, 
           Id->getName());
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
                             Val, EnumVal,
                             LastEnumConst);
  
  // Register this decl in the current scope stack.
  PushOnScopeChains(New, S);
  return New;
}

void Sema::ActOnEnumBody(SourceLocation EnumLoc, DeclTy *EnumDeclX,
                         DeclTy **Elements, unsigned NumElements) {
  EnumDecl *Enum = cast<EnumDecl>(static_cast<Decl*>(EnumDeclX));
  assert(!Enum->isDefinition() && "Enum redefinitions can't reach here");
  
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
  
  EnumConstantDecl *EltList = 0;
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
        Diag(ECD->getLocation(), diag::ext_enum_value_not_int,
             InitVal.toString());
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
    
    ECD->setNextDeclarator(EltList);
    EltList = ECD;
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
    ECD->setInitExpr(new ImplicitCastExpr(NewTy, ECD->getInitExpr()));
    ECD->setType(NewTy);
  }
  
  Enum->defineElements(EltList, BestType);
  Consumer.HandleTagDeclDefinition(Enum);
}

Sema::DeclTy *Sema::ActOnFileScopeAsmDecl(SourceLocation Loc,
                                          ExprTy *expr) {
  StringLiteral *AsmString = cast<StringLiteral>((Expr*)expr);
  
  return FileScopeAsmDecl::Create(Context, Loc, AsmString);
}

Sema::DeclTy* Sema::ActOnLinkageSpec(SourceLocation Loc,
                                     SourceLocation LBrace,
                                     SourceLocation RBrace,
                                     const char *Lang,
                                     unsigned StrSize,
                                     DeclTy *D) {
  LinkageSpecDecl::LanguageIDs Language;
  Decl *dcl = static_cast<Decl *>(D);
  if (strncmp(Lang, "\"C\"", StrSize) == 0)
    Language = LinkageSpecDecl::lang_c;
  else if (strncmp(Lang, "\"C++\"", StrSize) == 0)
    Language = LinkageSpecDecl::lang_cxx;
  else {
    Diag(Loc, diag::err_bad_language);
    return 0;
  }

  // FIXME: Add all the various semantics of linkage specifications
  return LinkageSpecDecl::Create(Context, Loc, Language, dcl);
}

void Sema::HandleDeclAttribute(Decl *New, AttributeList *Attr) {
  
  switch (Attr->getKind()) {
  case AttributeList::AT_vector_size:
    if (ValueDecl *vDecl = dyn_cast<ValueDecl>(New)) {
      QualType newType = HandleVectorTypeAttribute(vDecl->getType(), Attr);
      if (!newType.isNull()) // install the new vector type into the decl
        vDecl->setType(newType);
    } 
    if (TypedefDecl *tDecl = dyn_cast<TypedefDecl>(New)) {
      QualType newType = HandleVectorTypeAttribute(tDecl->getUnderlyingType(), 
                                                   Attr);
      if (!newType.isNull()) // install the new vector type into the decl
        tDecl->setUnderlyingType(newType);
    }
    break;
  case AttributeList::AT_ext_vector_type:
    if (TypedefDecl *tDecl = dyn_cast<TypedefDecl>(New))
      HandleExtVectorTypeAttribute(tDecl, Attr);
    else
      Diag(Attr->getLoc(), 
           diag::err_typecheck_ext_vector_not_typedef);
    break;
  case AttributeList::AT_address_space:
    // Ignore this, this is a type attribute, handled by ProcessTypeAttributes.
    break;
  case AttributeList::AT_mode:
    // Despite what would be logical, the mode attribute is a decl attribute,
    // not a type attribute: 'int ** __attribute((mode(HI))) *G;' tries to make
    // 'G' be HImode, not an intermediate pointer.
    if (TypedefDecl *tDecl = dyn_cast<TypedefDecl>(New)) {
      QualType newType = HandleModeTypeAttribute(tDecl->getUnderlyingType(),
                                                 Attr);
      tDecl->setUnderlyingType(newType);
    } else if (ValueDecl *vDecl = dyn_cast<ValueDecl>(New)) {
      QualType newType = HandleModeTypeAttribute(vDecl->getType(), Attr);
      vDecl->setType(newType);
    }
    // FIXME: Diagnostic?
    break;
  case AttributeList::AT_alias:
    HandleAliasAttribute(New, Attr);
    break;
  case AttributeList::AT_deprecated:
    HandleDeprecatedAttribute(New, Attr);
    break;
  case AttributeList::AT_visibility:
    HandleVisibilityAttribute(New, Attr);
    break;
  case AttributeList::AT_weak:
    HandleWeakAttribute(New, Attr);
    break;
  case AttributeList::AT_dllimport:
    HandleDLLImportAttribute(New, Attr);
    break;
  case AttributeList::AT_dllexport:
    HandleDLLExportAttribute(New, Attr);
    break;
  case AttributeList::AT_nothrow:
    HandleNothrowAttribute(New, Attr);
    break;
  case AttributeList::AT_stdcall:
    HandleStdCallAttribute(New, Attr);
    break;
  case AttributeList::AT_fastcall:
    HandleFastCallAttribute(New, Attr);
    break;
  case AttributeList::AT_aligned:
    HandleAlignedAttribute(New, Attr);
    break;
  case AttributeList::AT_packed:
    HandlePackedAttribute(New, Attr);
    break;
  case AttributeList::AT_annotate:
    HandleAnnotateAttribute(New, Attr);
    break;
  case AttributeList::AT_noreturn:
    HandleNoReturnAttribute(New, Attr);
    break;
  case AttributeList::AT_format:
    HandleFormatAttribute(New, Attr);
    break;
  case AttributeList::AT_transparent_union:
    HandleTransparentUnionAttribute(New, Attr);
    break;
  default:
#if 0
    // TODO: when we have the full set of attributes, warn about unknown ones.
    Diag(Attr->getLoc(), diag::warn_attribute_ignored,
         Attr->getName()->getName());
#endif
    break;
  }
}

void Sema::HandleDeclAttributes(Decl *New, AttributeList *declspec_prefix,
                                AttributeList *declarator_postfix) {
  if (declspec_prefix == 0 && declarator_postfix == 0) return;
  
  while (declspec_prefix) {
    HandleDeclAttribute(New, declspec_prefix);
    declspec_prefix = declspec_prefix->getNext();
  }
  
  // If there are any type attributes that were in the declarator, apply them to
  // its top level type.
  if (ValueDecl *VD = dyn_cast<ValueDecl>(New)) {
    QualType DT = VD->getType();
    ProcessTypeAttributes(DT, declarator_postfix);
    VD->setType(DT);
  } else if (TypedefDecl *TD = dyn_cast<TypedefDecl>(New)) {
    QualType DT = TD->getUnderlyingType();
    ProcessTypeAttributes(DT, declarator_postfix);
    TD->setUnderlyingType(DT);
  }
  
  while (declarator_postfix) {
    HandleDeclAttribute(New, declarator_postfix);
    declarator_postfix = declarator_postfix->getNext();
  }
}

void Sema::HandleExtVectorTypeAttribute(TypedefDecl *tDecl, 
                                        AttributeList *rawAttr) {
  QualType curType = tDecl->getUnderlyingType();
  // check the attribute arguments.
  if (rawAttr->getNumArgs() != 1) {
    Diag(rawAttr->getLoc(), diag::err_attribute_wrong_number_arguments,
         std::string("1"));
    return;
  }
  Expr *sizeExpr = static_cast<Expr *>(rawAttr->getArg(0));
  llvm::APSInt vecSize(32);
  if (!sizeExpr->isIntegerConstantExpr(vecSize, Context)) {
    Diag(rawAttr->getLoc(), diag::err_attribute_argument_not_int,
         "ext_vector_type", sizeExpr->getSourceRange());
    return;
  }
  // unlike gcc's vector_size attribute, we do not allow vectors to be defined
  // in conjunction with complex types (pointers, arrays, functions, etc.).
  Type *canonType = curType.getCanonicalType().getTypePtr();
  if (!(canonType->isIntegerType() || canonType->isRealFloatingType())) {
    Diag(rawAttr->getLoc(), diag::err_attribute_invalid_vector_type,
         curType.getCanonicalType().getAsString());
    return;
  }
  // unlike gcc's vector_size attribute, the size is specified as the 
  // number of elements, not the number of bytes.
  unsigned vectorSize = static_cast<unsigned>(vecSize.getZExtValue()); 
  
  if (vectorSize == 0) {
    Diag(rawAttr->getLoc(), diag::err_attribute_zero_size,
         sizeExpr->getSourceRange());
    return;
  }
  // Instantiate/Install the vector type, the number of elements is > 0.
  tDecl->setUnderlyingType(Context.getExtVectorType(curType, vectorSize));
  // Remember this typedef decl, we will need it later for diagnostics.
  ExtVectorDecls.push_back(tDecl);
}

QualType Sema::HandleVectorTypeAttribute(QualType curType, 
                                         AttributeList *rawAttr) {
  // check the attribute arugments.
  if (rawAttr->getNumArgs() != 1) {
    Diag(rawAttr->getLoc(), diag::err_attribute_wrong_number_arguments,
         std::string("1"));
    return QualType();
  }
  Expr *sizeExpr = static_cast<Expr *>(rawAttr->getArg(0));
  llvm::APSInt vecSize(32);
  if (!sizeExpr->isIntegerConstantExpr(vecSize, Context)) {
    Diag(rawAttr->getLoc(), diag::err_attribute_argument_not_int,
         "vector_size", sizeExpr->getSourceRange());
    return QualType();
  }
  // navigate to the base type - we need to provide for vector pointers, 
  // vector arrays, and functions returning vectors.
  Type *canonType = curType.getCanonicalType().getTypePtr();
  
  if (canonType->isPointerType() || canonType->isArrayType() ||
      canonType->isFunctionType()) {
    assert(0 && "HandleVector(): Complex type construction unimplemented");
    /* FIXME: rebuild the type from the inside out, vectorizing the inner type.
        do {
          if (PointerType *PT = dyn_cast<PointerType>(canonType))
            canonType = PT->getPointeeType().getTypePtr();
          else if (ArrayType *AT = dyn_cast<ArrayType>(canonType))
            canonType = AT->getElementType().getTypePtr();
          else if (FunctionType *FT = dyn_cast<FunctionType>(canonType))
            canonType = FT->getResultType().getTypePtr();
        } while (canonType->isPointerType() || canonType->isArrayType() ||
                 canonType->isFunctionType());
    */
  }
  // the base type must be integer or float.
  if (!(canonType->isIntegerType() || canonType->isRealFloatingType())) {
    Diag(rawAttr->getLoc(), diag::err_attribute_invalid_vector_type,
         curType.getCanonicalType().getAsString());
    return QualType();
  }
  unsigned typeSize = static_cast<unsigned>(Context.getTypeSize(curType));
  // vecSize is specified in bytes - convert to bits.
  unsigned vectorSize = static_cast<unsigned>(vecSize.getZExtValue() * 8); 
  
  // the vector size needs to be an integral multiple of the type size.
  if (vectorSize % typeSize) {
    Diag(rawAttr->getLoc(), diag::err_attribute_invalid_size,
         sizeExpr->getSourceRange());
    return QualType();
  }
  if (vectorSize == 0) {
    Diag(rawAttr->getLoc(), diag::err_attribute_zero_size,
         sizeExpr->getSourceRange());
    return QualType();
  }
  // Instantiate the vector type, the number of elements is > 0, and not
  // required to be a power of 2, unlike GCC.
  return Context.getVectorType(curType, vectorSize/typeSize);
}

void Sema::HandlePackedAttribute(Decl *d, AttributeList *rawAttr) {
  // check the attribute arguments.
  if (rawAttr->getNumArgs() > 0) {
    Diag(rawAttr->getLoc(), diag::err_attribute_wrong_number_arguments,
         std::string("0"));
    return;
  }
  
  if (TagDecl *TD = dyn_cast<TagDecl>(d))
    TD->addAttr(new PackedAttr);
  else if (FieldDecl *FD = dyn_cast<FieldDecl>(d)) {
    // If the alignment is less than or equal to 8 bits, the packed attribute
    // has no effect.
    if (!FD->getType()->isIncompleteType() &&
        Context.getTypeAlign(FD->getType()) <= 8)
      Diag(rawAttr->getLoc(), 
           diag::warn_attribute_ignored_for_field_of_type,
           rawAttr->getName()->getName(), FD->getType().getAsString());
    else
      FD->addAttr(new PackedAttr);
  } else
    Diag(rawAttr->getLoc(), diag::warn_attribute_ignored,
         rawAttr->getName()->getName());
}

void Sema::HandleAliasAttribute(Decl *d, AttributeList *rawAttr) {
  // check the attribute arguments.
  if (rawAttr->getNumArgs() != 1) {
    Diag(rawAttr->getLoc(), diag::err_attribute_wrong_number_arguments,
         std::string("1"));
    return;
  }

  Expr *Arg = static_cast<Expr*>(rawAttr->getArg(0));
  Arg = Arg->IgnoreParenCasts();
  StringLiteral *Str = dyn_cast<StringLiteral>(Arg);

  if (Str == 0 || Str->isWide()) {
    Diag(rawAttr->getLoc(), diag::err_attribute_argument_n_not_string,
         "alias", std::string("1"));
    return;
  }

  const char *Alias = Str->getStrData();
  unsigned AliasLen = Str->getByteLength();

  // FIXME: check if target symbol exists in current file

  d->addAttr(new AliasAttr(std::string(Alias, AliasLen)));
}

void Sema::HandleNoReturnAttribute(Decl *d, AttributeList *rawAttr) {
  // check the attribute arguments.
  if (rawAttr->getNumArgs() != 0) {
    Diag(rawAttr->getLoc(), diag::err_attribute_wrong_number_arguments,
         std::string("0"));
    return;
  }
  
  FunctionDecl *Fn = dyn_cast<FunctionDecl>(d);
  
  if (!Fn) {
    Diag(rawAttr->getLoc(), diag::warn_attribute_wrong_decl_type,
         "noreturn", "function");
    return;
  }
  
  d->addAttr(new NoReturnAttr());
}

void Sema::HandleDeprecatedAttribute(Decl *d, AttributeList *rawAttr) {
  // check the attribute arguments.
  if (rawAttr->getNumArgs() != 0) {
    Diag(rawAttr->getLoc(), diag::err_attribute_wrong_number_arguments,
         std::string("0"));
    return;
  }

  d->addAttr(new DeprecatedAttr());
}

void Sema::HandleVisibilityAttribute(Decl *d, AttributeList *rawAttr) {
  // check the attribute arguments.
  if (rawAttr->getNumArgs() != 1) {
    Diag(rawAttr->getLoc(), diag::err_attribute_wrong_number_arguments,
         std::string("1"));
    return;
  }

  Expr *Arg = static_cast<Expr*>(rawAttr->getArg(0));
  Arg = Arg->IgnoreParenCasts();
  StringLiteral *Str = dyn_cast<StringLiteral>(Arg);
  
  if (Str == 0 || Str->isWide()) {
    Diag(rawAttr->getLoc(), diag::err_attribute_argument_n_not_string,
         "visibility", std::string("1"));
    return;
  }

  const char *TypeStr = Str->getStrData();
  unsigned TypeLen = Str->getByteLength();
  VisibilityAttr::VisibilityTypes type;

  if (TypeLen == 7 && !memcmp(TypeStr, "default", 7))
    type = VisibilityAttr::DefaultVisibility;
  else if (TypeLen == 6 && !memcmp(TypeStr, "hidden", 6))
    type = VisibilityAttr::HiddenVisibility;
  else if (TypeLen == 8 && !memcmp(TypeStr, "internal", 8))
    type = VisibilityAttr::HiddenVisibility; // FIXME
  else if (TypeLen == 9 && !memcmp(TypeStr, "protected", 9))
    type = VisibilityAttr::ProtectedVisibility;
  else {
    Diag(rawAttr->getLoc(), diag::warn_attribute_type_not_supported,
           "visibility", TypeStr);
    return;
  }

  d->addAttr(new VisibilityAttr(type));
}

void Sema::HandleWeakAttribute(Decl *d, AttributeList *rawAttr) {
  // check the attribute arguments.
  if (rawAttr->getNumArgs() != 0) {
    Diag(rawAttr->getLoc(), diag::err_attribute_wrong_number_arguments,
         std::string("0"));
    return;
  }

  d->addAttr(new WeakAttr());
}

void Sema::HandleDLLImportAttribute(Decl *d, AttributeList *rawAttr) {
  // check the attribute arguments.
  if (rawAttr->getNumArgs() != 0) {
    Diag(rawAttr->getLoc(), diag::err_attribute_wrong_number_arguments,
         std::string("0"));
    return;
  }

  d->addAttr(new DLLImportAttr());
}

void Sema::HandleDLLExportAttribute(Decl *d, AttributeList *rawAttr) {
  // check the attribute arguments.
  if (rawAttr->getNumArgs() != 0) {
    Diag(rawAttr->getLoc(), diag::err_attribute_wrong_number_arguments,
         std::string("0"));
    return;
  }

  d->addAttr(new DLLExportAttr());
}

void Sema::HandleStdCallAttribute(Decl *d, AttributeList *rawAttr) {
  // check the attribute arguments.
  if (rawAttr->getNumArgs() != 0) {
    Diag(rawAttr->getLoc(), diag::err_attribute_wrong_number_arguments,
         std::string("0"));
    return;
  }

  d->addAttr(new StdCallAttr());
}

void Sema::HandleFastCallAttribute(Decl *d, AttributeList *rawAttr) {
  // check the attribute arguments.
  if (rawAttr->getNumArgs() != 0) {
    Diag(rawAttr->getLoc(), diag::err_attribute_wrong_number_arguments,
         std::string("0"));
    return;
  }

  d->addAttr(new FastCallAttr());
}

void Sema::HandleNothrowAttribute(Decl *d, AttributeList *rawAttr) {
  // check the attribute arguments.
  if (rawAttr->getNumArgs() != 0) {
    Diag(rawAttr->getLoc(), diag::err_attribute_wrong_number_arguments,
         std::string("0"));
    return;
  }

  d->addAttr(new NoThrowAttr());
}

static const FunctionTypeProto *getFunctionProto(Decl *d) {
  QualType Ty;

  if (ValueDecl *decl = dyn_cast<ValueDecl>(d))
    Ty = decl->getType();
  else if (FieldDecl *decl = dyn_cast<FieldDecl>(d))
    Ty = decl->getType();
  else if (TypedefDecl* decl = dyn_cast<TypedefDecl>(d))
    Ty = decl->getUnderlyingType();
  else
    return 0;

  if (Ty->isFunctionPointerType()) {
    const PointerType *PtrTy = Ty->getAsPointerType();
    Ty = PtrTy->getPointeeType();
  }

  if (const FunctionType *FnTy = Ty->getAsFunctionType())
    return dyn_cast<FunctionTypeProto>(FnTy->getAsFunctionType());

  return 0;
}

static inline bool isNSStringType(QualType T, ASTContext &Ctx) {
  if (!T->isPointerType())
    return false;
  
  T = T->getAsPointerType()->getPointeeType().getCanonicalType();
  ObjCInterfaceType* ClsT = dyn_cast<ObjCInterfaceType>(T.getTypePtr());
  
  if (!ClsT)
    return false;
  
  IdentifierInfo* ClsName = ClsT->getDecl()->getIdentifier();
  
  // FIXME: Should we walk the chain of classes?
  return ClsName == &Ctx.Idents.get("NSString") ||
         ClsName == &Ctx.Idents.get("NSMutableString");
}

/// Handle __attribute__((format(type,idx,firstarg))) attributes
/// based on http://gcc.gnu.org/onlinedocs/gcc/Function-Attributes.html
void Sema::HandleFormatAttribute(Decl *d, AttributeList *rawAttr) {

  if (!rawAttr->getParameterName()) {
    Diag(rawAttr->getLoc(), diag::err_attribute_argument_n_not_string,
           "format", std::string("1"));
    return;
  }

  if (rawAttr->getNumArgs() != 2) {
    Diag(rawAttr->getLoc(), diag::err_attribute_wrong_number_arguments,
         std::string("3"));
    return;
  }

  // GCC ignores the format attribute on K&R style function
  // prototypes, so we ignore it as well
  const FunctionTypeProto *proto = getFunctionProto(d);

  if (!proto) {
    Diag(rawAttr->getLoc(), diag::warn_attribute_wrong_decl_type,
           "format", "function");
    return;
  }

  // FIXME: in C++ the implicit 'this' function parameter also counts.
  // this is needed in order to be compatible with GCC
  // the index must start in 1 and the limit is numargs+1
  unsigned NumArgs  = proto->getNumArgs();
  unsigned FirstIdx = 1;

  const char *Format = rawAttr->getParameterName()->getName();
  unsigned FormatLen = rawAttr->getParameterName()->getLength();

  // Normalize the argument, __foo__ becomes foo.
  if (FormatLen > 4 && Format[0] == '_' && Format[1] == '_' &&
      Format[FormatLen - 2] == '_' && Format[FormatLen - 1] == '_') {
    Format += 2;
    FormatLen -= 4;
  }

  bool Supported = false;
  bool is_NSString = false;
  bool is_strftime = false;
  
  switch (FormatLen) {
    default: break;
    case 5:
      Supported = !memcmp(Format, "scanf", 5);
      break;
    case 6:
      Supported = !memcmp(Format, "printf", 6);
      break;
    case 7:
      Supported = !memcmp(Format, "strfmon", 7);
      break;
    case 8:
      Supported = (is_strftime = !memcmp(Format, "strftime", 8)) || 
                  (is_NSString = !memcmp(Format, "NSString", 8));
      break;
  }
      
  if (!Supported) {
    Diag(rawAttr->getLoc(), diag::warn_attribute_type_not_supported,
           "format", rawAttr->getParameterName()->getName());
    return;
  }

  // checks for the 2nd argument
  Expr *IdxExpr = static_cast<Expr *>(rawAttr->getArg(0));
  llvm::APSInt Idx(Context.getTypeSize(IdxExpr->getType()));
  if (!IdxExpr->isIntegerConstantExpr(Idx, Context)) {
    Diag(rawAttr->getLoc(), diag::err_attribute_argument_n_not_int,
           "format", std::string("2"), IdxExpr->getSourceRange());
    return;
  }

  if (Idx.getZExtValue() < FirstIdx || Idx.getZExtValue() > NumArgs) {
    Diag(rawAttr->getLoc(), diag::err_attribute_argument_out_of_bounds,
           "format", std::string("2"), IdxExpr->getSourceRange());
    return;
  }

  // FIXME: Do we need to bounds check?
  unsigned ArgIdx = Idx.getZExtValue() - 1;
  
  // make sure the format string is really a string
  QualType Ty = proto->getArgType(ArgIdx);

  if (is_NSString) {
    // FIXME: do we need to check if the type is NSString*?  What are
    //  the semantics?
    if (!isNSStringType(Ty, Context)) {
      // FIXME: Should highlight the actual expression that has the
      // wrong type.
      Diag(rawAttr->getLoc(), diag::err_format_attribute_not_NSString,
           IdxExpr->getSourceRange());
      return;
    }    
  }
  else if (!Ty->isPointerType() ||
      !Ty->getAsPointerType()->getPointeeType()->isCharType()) {
    // FIXME: Should highlight the actual expression that has the
    // wrong type.
    Diag(rawAttr->getLoc(), diag::err_format_attribute_not_string,
         IdxExpr->getSourceRange());
    return;
  }

  // check the 3rd argument
  Expr *FirstArgExpr = static_cast<Expr *>(rawAttr->getArg(1));
  llvm::APSInt FirstArg(Context.getTypeSize(FirstArgExpr->getType()));
  if (!FirstArgExpr->isIntegerConstantExpr(FirstArg, Context)) {
    Diag(rawAttr->getLoc(), diag::err_attribute_argument_n_not_int,
           "format", std::string("3"), FirstArgExpr->getSourceRange());
    return;
  }

  // check if the function is variadic if the 3rd argument non-zero
  if (FirstArg != 0) {
    if (proto->isVariadic()) {
      ++NumArgs; // +1 for ...
    } else {
      Diag(d->getLocation(), diag::err_format_attribute_requires_variadic);
      return;
    }
  }

  // strftime requires FirstArg to be 0 because it doesn't read from any variable
  // the input is just the current time + the format string
  if (is_strftime) {
    if (FirstArg != 0) {
      Diag(rawAttr->getLoc(), diag::err_format_strftime_third_parameter,
             FirstArgExpr->getSourceRange());
      return;
    }
  // if 0 it disables parameter checking (to use with e.g. va_list)
  } else if (FirstArg != 0 && FirstArg != NumArgs) {
    Diag(rawAttr->getLoc(), diag::err_attribute_argument_out_of_bounds,
           "format", std::string("3"), FirstArgExpr->getSourceRange());
    return;
  }

  d->addAttr(new FormatAttr(std::string(Format, FormatLen),
                            Idx.getZExtValue(), FirstArg.getZExtValue()));
}

void Sema::HandleTransparentUnionAttribute(Decl *d, AttributeList *rawAttr) {
  // check the attribute arguments.
  if (rawAttr->getNumArgs() != 0) {
    Diag(rawAttr->getLoc(), diag::err_attribute_wrong_number_arguments,
         std::string("0"));
    return;
  }

  TypeDecl *decl = dyn_cast<TypeDecl>(d);

  if (!decl || !Context.getTypeDeclType(decl)->isUnionType()) {
    Diag(rawAttr->getLoc(), diag::warn_attribute_wrong_decl_type,
         "transparent_union", "union");
    return;
  }

  //QualType QTy = Context.getTypeDeclType(decl);
  //const RecordType *Ty = QTy->getAsUnionType();

// FIXME
// Ty->addAttr(new TransparentUnionAttr());
}

void Sema::HandleAnnotateAttribute(Decl *d, AttributeList *rawAttr) {
  // check the attribute arguments.
  if (rawAttr->getNumArgs() != 1) {
    Diag(rawAttr->getLoc(), diag::err_attribute_wrong_number_arguments,
         std::string("1"));
    return;
  }
  Expr *argExpr = static_cast<Expr *>(rawAttr->getArg(0));
  StringLiteral *SE = dyn_cast<StringLiteral>(argExpr);
  
  // Make sure that there is a string literal as the annotation's single
  // argument.
  if (!SE) {
    Diag(rawAttr->getLoc(), diag::err_attribute_annotate_no_string);
    return;
  }
  d->addAttr(new AnnotateAttr(std::string(SE->getStrData(),
                                          SE->getByteLength())));
}

void Sema::HandleAlignedAttribute(Decl *d, AttributeList *rawAttr)
{
  // check the attribute arguments.
  if (rawAttr->getNumArgs() > 1) {
    Diag(rawAttr->getLoc(), diag::err_attribute_wrong_number_arguments,
         std::string("1"));
    return;
  }

  unsigned Align = 0;
  
  if (rawAttr->getNumArgs() == 0) {
    // FIXME: This should be the target specific maximum alignment.
    // (For now we just use 128 bits which is the maximum on X86.
    Align = 128;
    return;
  } else {
    Expr *alignmentExpr = static_cast<Expr *>(rawAttr->getArg(0));
    llvm::APSInt alignment(32);
    if (!alignmentExpr->isIntegerConstantExpr(alignment, Context)) {
      Diag(rawAttr->getLoc(), diag::err_attribute_argument_not_int,
           "aligned", alignmentExpr->getSourceRange());
      return;
    }
    
    Align = alignment.getZExtValue() * 8;
  }

  d->addAttr(new AlignedAttr(Align));
}
