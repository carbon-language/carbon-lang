//===------ SemaDeclCXX.cpp - Semantic Analysis for C++ Declarations ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements semantic analysis for C++ declarations.
//
//===----------------------------------------------------------------------===//

#include "Sema.h"
#include "clang/Basic/LangOptions.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/AST/Type.h"
#include "clang/Parse/Scope.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/Support/Compiler.h"

using namespace clang;

//===----------------------------------------------------------------------===//
// CheckDefaultArgumentVisitor
//===----------------------------------------------------------------------===//

namespace {
  /// CheckDefaultArgumentVisitor - C++ [dcl.fct.default] Traverses
  /// the default argument of a parameter to determine whether it
  /// contains any ill-formed subexpressions. For example, this will
  /// diagnose the use of local variables or parameters within the
  /// default argument expression.
  class VISIBILITY_HIDDEN CheckDefaultArgumentVisitor 
    : public StmtVisitor<CheckDefaultArgumentVisitor, bool>
  {
    Expr *DefaultArg;
    Sema *S;

  public:
    CheckDefaultArgumentVisitor(Expr *defarg, Sema *s) 
      : DefaultArg(defarg), S(s) {}

    bool VisitExpr(Expr *Node);
    bool VisitDeclRefExpr(DeclRefExpr *DRE);
  };

  /// VisitExpr - Visit all of the children of this expression.
  bool CheckDefaultArgumentVisitor::VisitExpr(Expr *Node) {
    bool IsInvalid = false;
    for (Stmt::child_iterator first = Node->child_begin(), 
           last = Node->child_end();
         first != last; ++first)
      IsInvalid |= Visit(*first);

    return IsInvalid;
  }

  /// VisitDeclRefExpr - Visit a reference to a declaration, to
  /// determine whether this declaration can be used in the default
  /// argument expression.
  bool CheckDefaultArgumentVisitor::VisitDeclRefExpr(DeclRefExpr *DRE) {
    ValueDecl *Decl = DRE->getDecl();
    if (ParmVarDecl *Param = dyn_cast<ParmVarDecl>(Decl)) {
      // C++ [dcl.fct.default]p9
      //   Default arguments are evaluated each time the function is
      //   called. The order of evaluation of function arguments is
      //   unspecified. Consequently, parameters of a function shall not
      //   be used in default argument expressions, even if they are not
      //   evaluated. Parameters of a function declared before a default
      //   argument expression are in scope and can hide namespace and
      //   class member names.
      return S->Diag(DRE->getSourceRange().getBegin(), 
                     diag::err_param_default_argument_references_param,
                     Param->getName(), DefaultArg->getSourceRange());
    } else if (VarDecl *VDecl = dyn_cast<VarDecl>(Decl)) {
      // C++ [dcl.fct.default]p7
      //   Local variables shall not be used in default argument
      //   expressions.
      if (VDecl->isBlockVarDecl())
        return S->Diag(DRE->getSourceRange().getBegin(), 
                       diag::err_param_default_argument_references_local,
                       VDecl->getName(), DefaultArg->getSourceRange());
    }

    // FIXME: when Clang has support for member functions, "this"
    // will also need to be diagnosed.

    return false;
  }
}

/// ActOnParamDefaultArgument - Check whether the default argument
/// provided for a function parameter is well-formed. If so, attach it
/// to the parameter declaration.
void
Sema::ActOnParamDefaultArgument(DeclTy *param, SourceLocation EqualLoc, 
                                ExprTy *defarg) {
  ParmVarDecl *Param = (ParmVarDecl *)param;
  llvm::OwningPtr<Expr> DefaultArg((Expr *)defarg);
  QualType ParamType = Param->getType();

  // Default arguments are only permitted in C++
  if (!getLangOptions().CPlusPlus) {
    Diag(EqualLoc, diag::err_param_default_argument, 
         DefaultArg->getSourceRange());
    return;
  }

  // C++ [dcl.fct.default]p5
  //   A default argument expression is implicitly converted (clause
  //   4) to the parameter type. The default argument expression has
  //   the same semantic constraints as the initializer expression in
  //   a declaration of a variable of the parameter type, using the
  //   copy-initialization semantics (8.5).
  //
  // FIXME: CheckSingleAssignmentConstraints has the wrong semantics
  // for C++ (since we want copy-initialization, not copy-assignment),
  // but we don't have the right semantics implemented yet. Because of
  // this, our error message is also very poor.
  QualType DefaultArgType = DefaultArg->getType();   
  Expr *DefaultArgPtr = DefaultArg.get();
  AssignConvertType ConvTy = CheckSingleAssignmentConstraints(ParamType,
                                                              DefaultArgPtr);
  if (DefaultArgPtr != DefaultArg.get()) {
    DefaultArg.take();
    DefaultArg.reset(DefaultArgPtr);
  }
  if (DiagnoseAssignmentResult(ConvTy, DefaultArg->getLocStart(), 
                               ParamType, DefaultArgType, DefaultArg.get(), 
                               "in default argument")) {
    return;
  }

  // Check that the default argument is well-formed
  CheckDefaultArgumentVisitor DefaultArgChecker(DefaultArg.get(), this);
  if (DefaultArgChecker.Visit(DefaultArg.get()))
    return;

  // Okay: add the default argument to the parameter
  Param->setDefaultArg(DefaultArg.take());
}

/// CheckExtraCXXDefaultArguments - Check for any extra default
/// arguments in the declarator, which is not a function declaration
/// or definition and therefore is not permitted to have default
/// arguments. This routine should be invoked for every declarator
/// that is not a function declaration or definition.
void Sema::CheckExtraCXXDefaultArguments(Declarator &D) {
  // C++ [dcl.fct.default]p3
  //   A default argument expression shall be specified only in the
  //   parameter-declaration-clause of a function declaration or in a
  //   template-parameter (14.1). It shall not be specified for a
  //   parameter pack. If it is specified in a
  //   parameter-declaration-clause, it shall not occur within a
  //   declarator or abstract-declarator of a parameter-declaration.
  for (unsigned i = 0; i < D.getNumTypeObjects(); ++i) {
    DeclaratorChunk &chunk = D.getTypeObject(i);
    if (chunk.Kind == DeclaratorChunk::Function) {
      for (unsigned argIdx = 0; argIdx < chunk.Fun.NumArgs; ++argIdx) {
        ParmVarDecl *Param = (ParmVarDecl *)chunk.Fun.ArgInfo[argIdx].Param;
        if (Param->getDefaultArg()) {
          Diag(Param->getLocation(), diag::err_param_default_argument_nonfunc,
               Param->getDefaultArg()->getSourceRange());
          Param->setDefaultArg(0);
        }
      }
    }
  }
}

// MergeCXXFunctionDecl - Merge two declarations of the same C++
// function, once we already know that they have the same
// type. Subroutine of MergeFunctionDecl.
FunctionDecl * 
Sema::MergeCXXFunctionDecl(FunctionDecl *New, FunctionDecl *Old) {
  // C++ [dcl.fct.default]p4:
  //
  //   For non-template functions, default arguments can be added in
  //   later declarations of a function in the same
  //   scope. Declarations in different scopes have completely
  //   distinct sets of default arguments. That is, declarations in
  //   inner scopes do not acquire default arguments from
  //   declarations in outer scopes, and vice versa. In a given
  //   function declaration, all parameters subsequent to a
  //   parameter with a default argument shall have default
  //   arguments supplied in this or previous declarations. A
  //   default argument shall not be redefined by a later
  //   declaration (not even to the same value).
  for (unsigned p = 0, NumParams = Old->getNumParams(); p < NumParams; ++p) {
    ParmVarDecl *OldParam = Old->getParamDecl(p);
    ParmVarDecl *NewParam = New->getParamDecl(p);

    if(OldParam->getDefaultArg() && NewParam->getDefaultArg()) {
      Diag(NewParam->getLocation(), 
           diag::err_param_default_argument_redefinition,
           NewParam->getDefaultArg()->getSourceRange());
      Diag(OldParam->getLocation(), diag::err_previous_definition);
    } else if (OldParam->getDefaultArg()) {
      // Merge the old default argument into the new parameter
      NewParam->setDefaultArg(OldParam->getDefaultArg());
    }
  }

  return New;  
}

/// CheckCXXDefaultArguments - Verify that the default arguments for a
/// function declaration are well-formed according to C++
/// [dcl.fct.default].
void Sema::CheckCXXDefaultArguments(FunctionDecl *FD) {
  unsigned NumParams = FD->getNumParams();
  unsigned p;

  // Find first parameter with a default argument
  for (p = 0; p < NumParams; ++p) {
    ParmVarDecl *Param = FD->getParamDecl(p);
    if (Param->getDefaultArg())
      break;
  }

  // C++ [dcl.fct.default]p4:
  //   In a given function declaration, all parameters
  //   subsequent to a parameter with a default argument shall
  //   have default arguments supplied in this or previous
  //   declarations. A default argument shall not be redefined
  //   by a later declaration (not even to the same value).
  unsigned LastMissingDefaultArg = 0;
  for(; p < NumParams; ++p) {
    ParmVarDecl *Param = FD->getParamDecl(p);
    if (!Param->getDefaultArg()) {
      if (Param->getIdentifier())
        Diag(Param->getLocation(), 
             diag::err_param_default_argument_missing_name,
             Param->getIdentifier()->getName());
      else
        Diag(Param->getLocation(), 
             diag::err_param_default_argument_missing);
    
      LastMissingDefaultArg = p;
    }
  }

  if (LastMissingDefaultArg > 0) {
    // Some default arguments were missing. Clear out all of the
    // default arguments up to (and including) the last missing
    // default argument, so that we leave the function parameters
    // in a semantically valid state.
    for (p = 0; p <= LastMissingDefaultArg; ++p) {
      ParmVarDecl *Param = FD->getParamDecl(p);
      if (Param->getDefaultArg()) {
        delete Param->getDefaultArg();
        Param->setDefaultArg(0);
      }
    }
  }
}

/// ActOnBaseSpecifier - Parsed a base specifier. A base specifier is
/// one entry in the base class list of a class specifier, for
/// example: 
///    class foo : public bar, virtual private baz { 
/// 'public bar' and 'virtual private baz' are each base-specifiers.
void Sema::ActOnBaseSpecifier(DeclTy *classdecl, SourceRange SpecifierRange,
                              bool Virtual, AccessSpecifier Access,
                              DeclTy *basetype, SourceLocation BaseLoc) {
  RecordDecl *Decl = (RecordDecl*)classdecl;
  QualType BaseType = Context.getTypeDeclType((TypeDecl*)basetype);

  // Base specifiers must be record types.
  if (!BaseType->isRecordType()) {
    Diag(BaseLoc, diag::err_base_must_be_class, SpecifierRange);
    return;
  }

  // C++ [class.union]p1:
  //   A union shall not be used as a base class.
  if (BaseType->isUnionType()) {
    Diag(BaseLoc, diag::err_union_as_base_class, SpecifierRange);
    return;
  }

  // C++ [class.union]p1:
  //   A union shall not have base classes.
  if (Decl->isUnion()) {
    Diag(Decl->getLocation(), diag::err_base_clause_on_union,
         SpecifierRange);
    Decl->setInvalidDecl();
    return;
  }

  // C++ [class.derived]p2:
  //   The class-name in a base-specifier shall not be an incompletely
  //   defined class.
  if (BaseType->isIncompleteType()) {
    Diag(BaseLoc, diag::err_incomplete_base_class, SpecifierRange);
    return;
  }

  // FIXME: C++ [class.mi]p3:
  //   A class shall not be specified as a direct base class of a
  //   derived class more than once.

  // FIXME: Attach base class to the record.
}

//===----------------------------------------------------------------------===//
// C++ class member Handling
//===----------------------------------------------------------------------===//

/// ActOnStartCXXClassDef - This is called at the start of a class/struct/union
/// definition, when on C++.
void Sema::ActOnStartCXXClassDef(Scope *S, DeclTy *D, SourceLocation LBrace) {
  Decl *Dcl = static_cast<Decl *>(D);
  PushDeclContext(cast<CXXRecordDecl>(Dcl));
  FieldCollector->StartClass();
}

/// ActOnCXXMemberDeclarator - This is invoked when a C++ class member
/// declarator is parsed. 'AS' is the access specifier, 'BW' specifies the
/// bitfield width if there is one and 'InitExpr' specifies the initializer if
/// any. 'LastInGroup' is non-null for cases where one declspec has multiple
/// declarators on it.
///
/// NOTE: Because of CXXFieldDecl's inability to be chained like ScopedDecls, if
/// an instance field is declared, a new CXXFieldDecl is created but the method
/// does *not* return it; it returns LastInGroup instead. The other C++ members
/// (which are all ScopedDecls) are returned after appending them to
/// LastInGroup.
Sema::DeclTy *
Sema::ActOnCXXMemberDeclarator(Scope *S, AccessSpecifier AS, Declarator &D,
                               ExprTy *BW, ExprTy *InitExpr,
                               DeclTy *LastInGroup) {
  const DeclSpec &DS = D.getDeclSpec();
  IdentifierInfo *II = D.getIdentifier();
  Expr *BitWidth = static_cast<Expr*>(BW);
  Expr *Init = static_cast<Expr*>(InitExpr);
  SourceLocation Loc = D.getIdentifierLoc();

  // C++ 9.2p6: A member shall not be declared to have automatic storage
  // duration (auto, register) or with the extern storage-class-specifier.
  switch (DS.getStorageClassSpec()) {
    case DeclSpec::SCS_unspecified:
    case DeclSpec::SCS_typedef:
    case DeclSpec::SCS_static:
      // FALL THROUGH.
      break;
    default:
      if (DS.getStorageClassSpecLoc().isValid())
        Diag(DS.getStorageClassSpecLoc(),
             diag::err_storageclass_invalid_for_member);
      else
        Diag(DS.getThreadSpecLoc(), diag::err_storageclass_invalid_for_member);
      D.getMutableDeclSpec().ClearStorageClassSpecs();
  }

  QualType T = GetTypeForDeclarator(D, S);

  // T->isFunctionType() is used instead of D.isFunctionDeclarator() to cover
  // this case:
  //
  // typedef int f();
  // f a;
  bool isInstField = (DS.getStorageClassSpec() == DeclSpec::SCS_unspecified &&
                      !T->isFunctionType());

  Decl *Member;
  bool InvalidDecl = false;

  if (isInstField)
    Member = static_cast<Decl*>(ActOnField(S, Loc, D, BitWidth));
  else
    Member = static_cast<Decl*>(ActOnDeclarator(S, D, LastInGroup));

  if (!Member) return LastInGroup;

  assert(II || isInstField && "No identifier for non-field ?");

  // set/getAccess is not part of Decl's interface to avoid bloating it with C++
  // specific methods. Use a wrapper class that can be used with all C++ class
  // member decls.
  CXXClassMemberWrapper(Member).setAccess(AS);

  if (BitWidth) {
    // C++ 9.6p2: Only when declaring an unnamed bit-field may the
    // constant-expression be a value equal to zero.
    // FIXME: Check this.

    if (D.isFunctionDeclarator()) {
      // FIXME: Emit diagnostic about only constructors taking base initializers
      // or something similar, when constructor support is in place.
      Diag(Loc, diag::err_not_bitfield_type,
           II->getName(), BitWidth->getSourceRange());
      InvalidDecl = true;

    } else if (isInstField || isa<FunctionDecl>(Member)) {
      // An instance field or a function typedef ("typedef int f(); f a;").
      // C++ 9.6p3: A bit-field shall have integral or enumeration type.
      if (!T->isIntegralType()) {
        Diag(Loc, diag::err_not_integral_type_bitfield,
             II->getName(), BitWidth->getSourceRange());
        InvalidDecl = true;
      }

    } else if (isa<TypedefDecl>(Member)) {
      // "cannot declare 'A' to be a bit-field type"
      Diag(Loc, diag::err_not_bitfield_type, II->getName(), 
           BitWidth->getSourceRange());
      InvalidDecl = true;

    } else {
      assert(isa<CXXClassVarDecl>(Member) &&
             "Didn't we cover all member kinds?");
      // C++ 9.6p3: A bit-field shall not be a static member.
      // "static member 'A' cannot be a bit-field"
      Diag(Loc, diag::err_static_not_bitfield, II->getName(), 
           BitWidth->getSourceRange());
      InvalidDecl = true;
    }
  }

  if (Init) {
    // C++ 9.2p4: A member-declarator can contain a constant-initializer only
    // if it declares a static member of const integral or const enumeration
    // type.
    if (CXXClassVarDecl *CVD =
              dyn_cast<CXXClassVarDecl>(Member)) { // ...static member of...
      CVD->setInit(Init);
      QualType MemberTy = CVD->getType().getCanonicalType();
      // ...const integral or const enumeration type.
      if (MemberTy.isConstQualified() && MemberTy->isIntegralType()) {
        if (CheckForConstantInitializer(Init, MemberTy)) // constant-initializer
          InvalidDecl = true;

      } else {
        // not const integral.
        Diag(Loc, diag::err_member_initialization,
             II->getName(), Init->getSourceRange());
        InvalidDecl = true;
      }

    } else {
      // not static member.
      Diag(Loc, diag::err_member_initialization,
           II->getName(), Init->getSourceRange());
      InvalidDecl = true;
    }
  }

  if (InvalidDecl)
    Member->setInvalidDecl();

  if (isInstField) {
    FieldCollector->Add(cast<CXXFieldDecl>(Member));
    return LastInGroup;
  }
  return Member;
}

void Sema::ActOnFinishCXXMemberSpecification(Scope* S, SourceLocation RLoc,
                                             DeclTy *TagDecl,
                                             SourceLocation LBrac,
                                             SourceLocation RBrac) {
  ActOnFields(S, RLoc, TagDecl,
              (DeclTy**)FieldCollector->getCurFields(),
              FieldCollector->getCurNumFields(), LBrac, RBrac);
}

void Sema::ActOnFinishCXXClassDef(DeclTy *D,SourceLocation RBrace) {
  Decl *Dcl = static_cast<Decl *>(D);
  assert(isa<CXXRecordDecl>(Dcl) &&
         "Invalid parameter, expected CXXRecordDecl");
  FieldCollector->FinishClass();
  PopDeclContext();
}

//===----------------------------------------------------------------------===//
// Namespace Handling
//===----------------------------------------------------------------------===//

/// ActOnStartNamespaceDef - This is called at the start of a namespace
/// definition.
Sema::DeclTy *Sema::ActOnStartNamespaceDef(Scope *NamespcScope,
                                           SourceLocation IdentLoc,
                                           IdentifierInfo *II,
                                           SourceLocation LBrace) {
  NamespaceDecl *Namespc =
      NamespaceDecl::Create(Context, CurContext, IdentLoc, II);
  Namespc->setLBracLoc(LBrace);

  Scope *DeclRegionScope = NamespcScope->getParent();

  if (II) {
    // C++ [namespace.def]p2:
    // The identifier in an original-namespace-definition shall not have been
    // previously defined in the declarative region in which the
    // original-namespace-definition appears. The identifier in an
    // original-namespace-definition is the name of the namespace. Subsequently
    // in that declarative region, it is treated as an original-namespace-name.

    Decl *PrevDecl =
        LookupDecl(II, Decl::IDNS_Tag | Decl::IDNS_Ordinary, DeclRegionScope,
                   /*enableLazyBuiltinCreation=*/false);

    if (PrevDecl &&
        IdResolver.isDeclInScope(PrevDecl, CurContext, DeclRegionScope)) {
      if (NamespaceDecl *OrigNS = dyn_cast<NamespaceDecl>(PrevDecl)) {
        // This is an extended namespace definition.
        // Attach this namespace decl to the chain of extended namespace
        // definitions.
        NamespaceDecl *NextNS = OrigNS;
        while (NextNS->getNextNamespace())
          NextNS = NextNS->getNextNamespace();

        NextNS->setNextNamespace(Namespc);
        Namespc->setOriginalNamespace(OrigNS);

        // We won't add this decl to the current scope. We want the namespace
        // name to return the original namespace decl during a name lookup.
      } else {
        // This is an invalid name redefinition.
        Diag(Namespc->getLocation(), diag::err_redefinition_different_kind,
          Namespc->getName());
        Diag(PrevDecl->getLocation(), diag::err_previous_definition);
        Namespc->setInvalidDecl();
        // Continue on to push Namespc as current DeclContext and return it.
      }
    } else {
      // This namespace name is declared for the first time.
      PushOnScopeChains(Namespc, DeclRegionScope);
    }
  }
  else {
    // FIXME: Handle anonymous namespaces
  }

  // Although we could have an invalid decl (i.e. the namespace name is a
  // redefinition), push it as current DeclContext and try to continue parsing.
  PushDeclContext(Namespc->getOriginalNamespace());
  return Namespc;
}

/// ActOnFinishNamespaceDef - This callback is called after a namespace is
/// exited. Decl is the DeclTy returned by ActOnStartNamespaceDef.
void Sema::ActOnFinishNamespaceDef(DeclTy *D, SourceLocation RBrace) {
  Decl *Dcl = static_cast<Decl *>(D);
  NamespaceDecl *Namespc = dyn_cast_or_null<NamespaceDecl>(Dcl);
  assert(Namespc && "Invalid parameter, expected NamespaceDecl");
  Namespc->setRBracLoc(RBrace);
  PopDeclContext();
}
