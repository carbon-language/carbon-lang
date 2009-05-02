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
#include "SemaInherit.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/TypeOrdering.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Parse/DeclSpec.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Compiler.h"
#include <algorithm> // for std::equal
#include <map>

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
    : public StmtVisitor<CheckDefaultArgumentVisitor, bool> {
    Expr *DefaultArg;
    Sema *S;

  public:
    CheckDefaultArgumentVisitor(Expr *defarg, Sema *s) 
      : DefaultArg(defarg), S(s) {}

    bool VisitExpr(Expr *Node);
    bool VisitDeclRefExpr(DeclRefExpr *DRE);
    bool VisitCXXThisExpr(CXXThisExpr *ThisE);
  };

  /// VisitExpr - Visit all of the children of this expression.
  bool CheckDefaultArgumentVisitor::VisitExpr(Expr *Node) {
    bool IsInvalid = false;
    for (Stmt::child_iterator I = Node->child_begin(), 
         E = Node->child_end(); I != E; ++I)
      IsInvalid |= Visit(*I);
    return IsInvalid;
  }

  /// VisitDeclRefExpr - Visit a reference to a declaration, to
  /// determine whether this declaration can be used in the default
  /// argument expression.
  bool CheckDefaultArgumentVisitor::VisitDeclRefExpr(DeclRefExpr *DRE) {
    NamedDecl *Decl = DRE->getDecl();
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
                     diag::err_param_default_argument_references_param)
         << Param->getDeclName() << DefaultArg->getSourceRange();
    } else if (VarDecl *VDecl = dyn_cast<VarDecl>(Decl)) {
      // C++ [dcl.fct.default]p7
      //   Local variables shall not be used in default argument
      //   expressions.
      if (VDecl->isBlockVarDecl())
        return S->Diag(DRE->getSourceRange().getBegin(), 
                       diag::err_param_default_argument_references_local)
          << VDecl->getDeclName() << DefaultArg->getSourceRange();
    }

    return false;
  }

  /// VisitCXXThisExpr - Visit a C++ "this" expression.
  bool CheckDefaultArgumentVisitor::VisitCXXThisExpr(CXXThisExpr *ThisE) {
    // C++ [dcl.fct.default]p8:
    //   The keyword this shall not be used in a default argument of a
    //   member function.
    return S->Diag(ThisE->getSourceRange().getBegin(),
                   diag::err_param_default_argument_references_this)
               << ThisE->getSourceRange();
  }
}

/// ActOnParamDefaultArgument - Check whether the default argument
/// provided for a function parameter is well-formed. If so, attach it
/// to the parameter declaration.
void
Sema::ActOnParamDefaultArgument(DeclPtrTy param, SourceLocation EqualLoc, 
                                ExprArg defarg) {
  ParmVarDecl *Param = cast<ParmVarDecl>(param.getAs<Decl>());
  ExprOwningPtr<Expr> DefaultArg(this, defarg.takeAs<Expr>());
  QualType ParamType = Param->getType();

  // Default arguments are only permitted in C++
  if (!getLangOptions().CPlusPlus) {
    Diag(EqualLoc, diag::err_param_default_argument)
      << DefaultArg->getSourceRange();
    Param->setInvalidDecl();
    return;
  }

  // C++ [dcl.fct.default]p5
  //   A default argument expression is implicitly converted (clause
  //   4) to the parameter type. The default argument expression has
  //   the same semantic constraints as the initializer expression in
  //   a declaration of a variable of the parameter type, using the
  //   copy-initialization semantics (8.5).
  Expr *DefaultArgPtr = DefaultArg.get();
  bool DefaultInitFailed = CheckInitializerTypes(DefaultArgPtr, ParamType,
                                                 EqualLoc,
                                                 Param->getDeclName(),
                                                 /*DirectInit=*/false);
  if (DefaultArgPtr != DefaultArg.get()) {
    DefaultArg.take();
    DefaultArg.reset(DefaultArgPtr);
  }
  if (DefaultInitFailed) {
    return;
  }

  // Check that the default argument is well-formed
  CheckDefaultArgumentVisitor DefaultArgChecker(DefaultArg.get(), this);
  if (DefaultArgChecker.Visit(DefaultArg.get())) {
    Param->setInvalidDecl();
    return;
  }

  // Okay: add the default argument to the parameter
  Param->setDefaultArg(DefaultArg.take());
}

/// ActOnParamUnparsedDefaultArgument - We've seen a default
/// argument for a function parameter, but we can't parse it yet
/// because we're inside a class definition. Note that this default
/// argument will be parsed later.
void Sema::ActOnParamUnparsedDefaultArgument(DeclPtrTy param, 
                                             SourceLocation EqualLoc) {
  ParmVarDecl *Param = cast<ParmVarDecl>(param.getAs<Decl>());
  if (Param)
    Param->setUnparsedDefaultArg();
}

/// ActOnParamDefaultArgumentError - Parsing or semantic analysis of
/// the default argument for the parameter param failed.
void Sema::ActOnParamDefaultArgumentError(DeclPtrTy param) {
  cast<ParmVarDecl>(param.getAs<Decl>())->setInvalidDecl();
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
  for (unsigned i = 0, e = D.getNumTypeObjects(); i != e; ++i) {
    DeclaratorChunk &chunk = D.getTypeObject(i);
    if (chunk.Kind == DeclaratorChunk::Function) {
      for (unsigned argIdx = 0, e = chunk.Fun.NumArgs; argIdx != e; ++argIdx) {
        ParmVarDecl *Param =
          cast<ParmVarDecl>(chunk.Fun.ArgInfo[argIdx].Param.getAs<Decl>());
        if (Param->hasUnparsedDefaultArg()) {
          CachedTokens *Toks = chunk.Fun.ArgInfo[argIdx].DefaultArgTokens;
          Diag(Param->getLocation(), diag::err_param_default_argument_nonfunc)
            << SourceRange((*Toks)[1].getLocation(), Toks->back().getLocation());
          delete Toks;
          chunk.Fun.ArgInfo[argIdx].DefaultArgTokens = 0;
        } else if (Param->getDefaultArg()) {
          Diag(Param->getLocation(), diag::err_param_default_argument_nonfunc)
            << Param->getDefaultArg()->getSourceRange();
          Param->setDefaultArg(0);
        }
      }
    }
  }
}

// MergeCXXFunctionDecl - Merge two declarations of the same C++
// function, once we already know that they have the same
// type. Subroutine of MergeFunctionDecl. Returns true if there was an
// error, false otherwise.
bool Sema::MergeCXXFunctionDecl(FunctionDecl *New, FunctionDecl *Old) {
  bool Invalid = false;

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
           diag::err_param_default_argument_redefinition)
        << NewParam->getDefaultArg()->getSourceRange();
      Diag(OldParam->getLocation(), diag::note_previous_definition);
      Invalid = true;
    } else if (OldParam->getDefaultArg()) {
      // Merge the old default argument into the new parameter
      NewParam->setDefaultArg(OldParam->getDefaultArg());
    }
  }

  return Invalid;
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
      if (Param->isInvalidDecl())
        /* We already complained about this parameter. */;
      else if (Param->getIdentifier())
        Diag(Param->getLocation(), 
             diag::err_param_default_argument_missing_name)
          << Param->getIdentifier();
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
        if (!Param->hasUnparsedDefaultArg())
          Param->getDefaultArg()->Destroy(Context);
        Param->setDefaultArg(0);
      }
    }
  }
}

/// isCurrentClassName - Determine whether the identifier II is the
/// name of the class type currently being defined. In the case of
/// nested classes, this will only return true if II is the name of
/// the innermost class.
bool Sema::isCurrentClassName(const IdentifierInfo &II, Scope *,
                              const CXXScopeSpec *SS) {
  CXXRecordDecl *CurDecl;
  if (SS && SS->isSet() && !SS->isInvalid()) {
    DeclContext *DC = computeDeclContext(*SS);
    CurDecl = dyn_cast_or_null<CXXRecordDecl>(DC);
  } else
    CurDecl = dyn_cast_or_null<CXXRecordDecl>(CurContext);

  if (CurDecl)
    return &II == CurDecl->getIdentifier();
  else
    return false;
}

/// \brief Check the validity of a C++ base class specifier. 
///
/// \returns a new CXXBaseSpecifier if well-formed, emits diagnostics
/// and returns NULL otherwise.
CXXBaseSpecifier *
Sema::CheckBaseSpecifier(CXXRecordDecl *Class,
                         SourceRange SpecifierRange,
                         bool Virtual, AccessSpecifier Access,
                         QualType BaseType, 
                         SourceLocation BaseLoc) {
  // C++ [class.union]p1:
  //   A union shall not have base classes.
  if (Class->isUnion()) {
    Diag(Class->getLocation(), diag::err_base_clause_on_union)
      << SpecifierRange;
    return 0;
  }

  if (BaseType->isDependentType())
    return new CXXBaseSpecifier(SpecifierRange, Virtual, 
                                Class->getTagKind() == RecordDecl::TK_class,
                                Access, BaseType);

  // Base specifiers must be record types.
  if (!BaseType->isRecordType()) {
    Diag(BaseLoc, diag::err_base_must_be_class) << SpecifierRange;
    return 0;
  }

  // C++ [class.union]p1:
  //   A union shall not be used as a base class.
  if (BaseType->isUnionType()) {
    Diag(BaseLoc, diag::err_union_as_base_class) << SpecifierRange;
    return 0;
  }

  // C++ [class.derived]p2:
  //   The class-name in a base-specifier shall not be an incompletely
  //   defined class.
  if (RequireCompleteType(BaseLoc, BaseType, diag::err_incomplete_base_class,
                          SpecifierRange))
    return 0;

  // If the base class is polymorphic, the new one is, too.
  RecordDecl *BaseDecl = BaseType->getAsRecordType()->getDecl();
  assert(BaseDecl && "Record type has no declaration");
  BaseDecl = BaseDecl->getDefinition(Context);
  assert(BaseDecl && "Base type is not incomplete, but has no definition");
  if (cast<CXXRecordDecl>(BaseDecl)->isPolymorphic())
    Class->setPolymorphic(true);

  // C++ [dcl.init.aggr]p1:
  //   An aggregate is [...] a class with [...] no base classes [...].
  Class->setAggregate(false);
  Class->setPOD(false);

  if (Virtual) {
    // C++ [class.ctor]p5:
    //   A constructor is trivial if its class has no virtual base classes.
    Class->setHasTrivialConstructor(false);
  } else {
    // C++ [class.ctor]p5:
    //   A constructor is trivial if all the direct base classes of its 
    //   class have trivial constructors.
    Class->setHasTrivialConstructor(cast<CXXRecordDecl>(BaseDecl)->
                                    hasTrivialConstructor());
  }

  // C++ [class.ctor]p3:
  //   A destructor is trivial if all the direct base classes of its class
  //   have trivial destructors.
  Class->setHasTrivialDestructor(cast<CXXRecordDecl>(BaseDecl)->
                                 hasTrivialDestructor());
  
  // Create the base specifier.
  // FIXME: Allocate via ASTContext?
  return new CXXBaseSpecifier(SpecifierRange, Virtual, 
                              Class->getTagKind() == RecordDecl::TK_class, 
                              Access, BaseType);
}

/// ActOnBaseSpecifier - Parsed a base specifier. A base specifier is
/// one entry in the base class list of a class specifier, for
/// example: 
///    class foo : public bar, virtual private baz { 
/// 'public bar' and 'virtual private baz' are each base-specifiers.
Sema::BaseResult 
Sema::ActOnBaseSpecifier(DeclPtrTy classdecl, SourceRange SpecifierRange,
                         bool Virtual, AccessSpecifier Access,
                         TypeTy *basetype, SourceLocation BaseLoc) {
  AdjustDeclIfTemplate(classdecl);
  CXXRecordDecl *Class = cast<CXXRecordDecl>(classdecl.getAs<Decl>());
  QualType BaseType = QualType::getFromOpaquePtr(basetype);
  if (CXXBaseSpecifier *BaseSpec = CheckBaseSpecifier(Class, SpecifierRange,
                                                      Virtual, Access,
                                                      BaseType, BaseLoc))
    return BaseSpec;
  
  return true;
}

/// \brief Performs the actual work of attaching the given base class
/// specifiers to a C++ class.
bool Sema::AttachBaseSpecifiers(CXXRecordDecl *Class, CXXBaseSpecifier **Bases,
                                unsigned NumBases) {
 if (NumBases == 0)
    return false;

  // Used to keep track of which base types we have already seen, so
  // that we can properly diagnose redundant direct base types. Note
  // that the key is always the unqualified canonical type of the base
  // class.
  std::map<QualType, CXXBaseSpecifier*, QualTypeOrdering> KnownBaseTypes;

  // Copy non-redundant base specifiers into permanent storage.
  unsigned NumGoodBases = 0;
  bool Invalid = false;
  for (unsigned idx = 0; idx < NumBases; ++idx) {
    QualType NewBaseType 
      = Context.getCanonicalType(Bases[idx]->getType());
    NewBaseType = NewBaseType.getUnqualifiedType();

    if (KnownBaseTypes[NewBaseType]) {
      // C++ [class.mi]p3:
      //   A class shall not be specified as a direct base class of a
      //   derived class more than once.
      Diag(Bases[idx]->getSourceRange().getBegin(),
           diag::err_duplicate_base_class)
        << KnownBaseTypes[NewBaseType]->getType()
        << Bases[idx]->getSourceRange();

      // Delete the duplicate base class specifier; we're going to
      // overwrite its pointer later.
      delete Bases[idx];

      Invalid = true;
    } else {
      // Okay, add this new base class.
      KnownBaseTypes[NewBaseType] = Bases[idx];
      Bases[NumGoodBases++] = Bases[idx];
    }
  }

  // Attach the remaining base class specifiers to the derived class.
  Class->setBases(Bases, NumGoodBases);

  // Delete the remaining (good) base class specifiers, since their
  // data has been copied into the CXXRecordDecl.
  for (unsigned idx = 0; idx < NumGoodBases; ++idx)
    delete Bases[idx];

  return Invalid;
}

/// ActOnBaseSpecifiers - Attach the given base specifiers to the
/// class, after checking whether there are any duplicate base
/// classes.
void Sema::ActOnBaseSpecifiers(DeclPtrTy ClassDecl, BaseTy **Bases, 
                               unsigned NumBases) {
  if (!ClassDecl || !Bases || !NumBases)
    return;

  AdjustDeclIfTemplate(ClassDecl);
  AttachBaseSpecifiers(cast<CXXRecordDecl>(ClassDecl.getAs<Decl>()),
                       (CXXBaseSpecifier**)(Bases), NumBases);
}

//===----------------------------------------------------------------------===//
// C++ class member Handling
//===----------------------------------------------------------------------===//

/// ActOnCXXMemberDeclarator - This is invoked when a C++ class member
/// declarator is parsed. 'AS' is the access specifier, 'BW' specifies the
/// bitfield width if there is one and 'InitExpr' specifies the initializer if
/// any.
Sema::DeclPtrTy
Sema::ActOnCXXMemberDeclarator(Scope *S, AccessSpecifier AS, Declarator &D,
                               ExprTy *BW, ExprTy *InitExpr, bool Deleted) {
  const DeclSpec &DS = D.getDeclSpec();
  DeclarationName Name = GetNameForDeclarator(D);
  Expr *BitWidth = static_cast<Expr*>(BW);
  Expr *Init = static_cast<Expr*>(InitExpr);
  SourceLocation Loc = D.getIdentifierLoc();

  bool isFunc = D.isFunctionDeclarator();

  // C++ 9.2p6: A member shall not be declared to have automatic storage
  // duration (auto, register) or with the extern storage-class-specifier.
  // C++ 7.1.1p8: The mutable specifier can be applied only to names of class
  // data members and cannot be applied to names declared const or static,
  // and cannot be applied to reference members.
  switch (DS.getStorageClassSpec()) {
    case DeclSpec::SCS_unspecified:
    case DeclSpec::SCS_typedef:
    case DeclSpec::SCS_static:
      // FALL THROUGH.
      break;
    case DeclSpec::SCS_mutable:
      if (isFunc) {
        if (DS.getStorageClassSpecLoc().isValid())
          Diag(DS.getStorageClassSpecLoc(), diag::err_mutable_function);
        else
          Diag(DS.getThreadSpecLoc(), diag::err_mutable_function);
        
        // FIXME: It would be nicer if the keyword was ignored only for this
        // declarator. Otherwise we could get follow-up errors.
        D.getMutableDeclSpec().ClearStorageClassSpecs();
      } else {
        QualType T = GetTypeForDeclarator(D, S);
        diag::kind err = static_cast<diag::kind>(0);
        if (T->isReferenceType())
          err = diag::err_mutable_reference;
        else if (T.isConstQualified())
          err = diag::err_mutable_const;
        if (err != 0) {
          if (DS.getStorageClassSpecLoc().isValid())
            Diag(DS.getStorageClassSpecLoc(), err);
          else
            Diag(DS.getThreadSpecLoc(), err);
          // FIXME: It would be nicer if the keyword was ignored only for this
          // declarator. Otherwise we could get follow-up errors.
          D.getMutableDeclSpec().ClearStorageClassSpecs();
        }
      }
      break;
    default:
      if (DS.getStorageClassSpecLoc().isValid())
        Diag(DS.getStorageClassSpecLoc(),
             diag::err_storageclass_invalid_for_member);
      else
        Diag(DS.getThreadSpecLoc(), diag::err_storageclass_invalid_for_member);
      D.getMutableDeclSpec().ClearStorageClassSpecs();
  }

  if (!isFunc &&
      D.getDeclSpec().getTypeSpecType() == DeclSpec::TST_typename &&
      D.getNumTypeObjects() == 0) {
    // Check also for this case:
    //
    // typedef int f();
    // f a;
    //
    QualType TDType = QualType::getFromOpaquePtr(DS.getTypeRep());
    isFunc = TDType->isFunctionType();
  }

  bool isInstField = ((DS.getStorageClassSpec() == DeclSpec::SCS_unspecified ||
                       DS.getStorageClassSpec() == DeclSpec::SCS_mutable) &&
                      !isFunc);

  Decl *Member;
  if (isInstField) {
    Member = HandleField(S, cast<CXXRecordDecl>(CurContext), Loc, D, BitWidth,
                         AS);
    assert(Member && "HandleField never returns null");
  } else {
    Member = ActOnDeclarator(S, D).getAs<Decl>();
    if (!Member) {
      if (BitWidth) DeleteExpr(BitWidth);
      return DeclPtrTy();
    }

    // Non-instance-fields can't have a bitfield.
    if (BitWidth) {
      if (Member->isInvalidDecl()) {
        // don't emit another diagnostic.
      } else if (isa<VarDecl>(Member)) {
        // C++ 9.6p3: A bit-field shall not be a static member.
        // "static member 'A' cannot be a bit-field"
        Diag(Loc, diag::err_static_not_bitfield)
          << Name << BitWidth->getSourceRange();
      } else if (isa<TypedefDecl>(Member)) {
        // "typedef member 'x' cannot be a bit-field"
        Diag(Loc, diag::err_typedef_not_bitfield)
          << Name << BitWidth->getSourceRange();
      } else {
        // A function typedef ("typedef int f(); f a;").
        // C++ 9.6p3: A bit-field shall have integral or enumeration type.
        Diag(Loc, diag::err_not_integral_type_bitfield)
          << Name << cast<ValueDecl>(Member)->getType() 
          << BitWidth->getSourceRange();
      }
      
      DeleteExpr(BitWidth);
      BitWidth = 0;
      Member->setInvalidDecl();
    }

    Member->setAccess(AS);
  }

  assert((Name || isInstField) && "No identifier for non-field ?");

  if (Init)
    AddInitializerToDecl(DeclPtrTy::make(Member), ExprArg(*this, Init), false);
  if (Deleted) // FIXME: Source location is not very good.
    SetDeclDeleted(DeclPtrTy::make(Member), D.getSourceRange().getBegin());

  if (isInstField) {
    FieldCollector->Add(cast<FieldDecl>(Member));
    return DeclPtrTy();
  }
  return DeclPtrTy::make(Member);
}

/// ActOnMemInitializer - Handle a C++ member initializer.
Sema::MemInitResult 
Sema::ActOnMemInitializer(DeclPtrTy ConstructorD,
                          Scope *S,
                          IdentifierInfo *MemberOrBase,
                          SourceLocation IdLoc,
                          SourceLocation LParenLoc,
                          ExprTy **Args, unsigned NumArgs,
                          SourceLocation *CommaLocs,
                          SourceLocation RParenLoc) {
  CXXConstructorDecl *Constructor 
    = dyn_cast<CXXConstructorDecl>(ConstructorD.getAs<Decl>());
  if (!Constructor) {
    // The user wrote a constructor initializer on a function that is
    // not a C++ constructor. Ignore the error for now, because we may
    // have more member initializers coming; we'll diagnose it just
    // once in ActOnMemInitializers.
    return true;
  }

  CXXRecordDecl *ClassDecl = Constructor->getParent();

  // C++ [class.base.init]p2:
  //   Names in a mem-initializer-id are looked up in the scope of the
  //   constructor’s class and, if not found in that scope, are looked
  //   up in the scope containing the constructor’s
  //   definition. [Note: if the constructor’s class contains a member
  //   with the same name as a direct or virtual base class of the
  //   class, a mem-initializer-id naming the member or base class and
  //   composed of a single identifier refers to the class member. A
  //   mem-initializer-id for the hidden base class may be specified
  //   using a qualified name. ]
  // Look for a member, first.
  FieldDecl *Member = 0;
  DeclContext::lookup_result Result 
    = ClassDecl->lookup(Context, MemberOrBase);
  if (Result.first != Result.second)
    Member = dyn_cast<FieldDecl>(*Result.first);

  // FIXME: Handle members of an anonymous union.

  if (Member) {
    // FIXME: Perform direct initialization of the member.
    return new CXXBaseOrMemberInitializer(Member, (Expr **)Args, NumArgs);
  }

  // It didn't name a member, so see if it names a class.
  TypeTy *BaseTy = getTypeName(*MemberOrBase, IdLoc, S, 0/*SS*/);
  if (!BaseTy)
    return Diag(IdLoc, diag::err_mem_init_not_member_or_class)
      << MemberOrBase << SourceRange(IdLoc, RParenLoc);
  
  QualType BaseType = QualType::getFromOpaquePtr(BaseTy);
  if (!BaseType->isRecordType())
    return Diag(IdLoc, diag::err_base_init_does_not_name_class)
      << BaseType << SourceRange(IdLoc, RParenLoc);

  // C++ [class.base.init]p2:
  //   [...] Unless the mem-initializer-id names a nonstatic data
  //   member of the constructor’s class or a direct or virtual base
  //   of that class, the mem-initializer is ill-formed. A
  //   mem-initializer-list can initialize a base class using any
  //   name that denotes that base class type.
  
  // First, check for a direct base class.
  const CXXBaseSpecifier *DirectBaseSpec = 0;
  for (CXXRecordDecl::base_class_const_iterator Base = ClassDecl->bases_begin();
       Base != ClassDecl->bases_end(); ++Base) {
    if (Context.getCanonicalType(BaseType).getUnqualifiedType() == 
        Context.getCanonicalType(Base->getType()).getUnqualifiedType()) {
      // We found a direct base of this type. That's what we're
      // initializing.
      DirectBaseSpec = &*Base;
      break;
    }
  }
  
  // Check for a virtual base class.
  // FIXME: We might be able to short-circuit this if we know in
  // advance that there are no virtual bases.
  const CXXBaseSpecifier *VirtualBaseSpec = 0;
  if (!DirectBaseSpec || !DirectBaseSpec->isVirtual()) {
    // We haven't found a base yet; search the class hierarchy for a
    // virtual base class.
    BasePaths Paths(/*FindAmbiguities=*/true, /*RecordPaths=*/true,
                    /*DetectVirtual=*/false);
    if (IsDerivedFrom(Context.getTypeDeclType(ClassDecl), BaseType, Paths)) {
      for (BasePaths::paths_iterator Path = Paths.begin(); 
           Path != Paths.end(); ++Path) {
        if (Path->back().Base->isVirtual()) {
          VirtualBaseSpec = Path->back().Base;
          break;
        }
      }
    }
  }

  // C++ [base.class.init]p2:
  //   If a mem-initializer-id is ambiguous because it designates both
  //   a direct non-virtual base class and an inherited virtual base
  //   class, the mem-initializer is ill-formed.
  if (DirectBaseSpec && VirtualBaseSpec)
    return Diag(IdLoc, diag::err_base_init_direct_and_virtual)
      << MemberOrBase << SourceRange(IdLoc, RParenLoc);

  return new CXXBaseOrMemberInitializer(BaseType, (Expr **)Args, NumArgs);
}

void Sema::ActOnMemInitializers(DeclPtrTy ConstructorDecl, 
                                SourceLocation ColonLoc,
                                MemInitTy **MemInits, unsigned NumMemInits) {
  CXXConstructorDecl *Constructor = 
  dyn_cast<CXXConstructorDecl>(ConstructorDecl.getAs<Decl>());
  
  if (!Constructor) {
    Diag(ColonLoc, diag::err_only_constructors_take_base_inits);
    return;
  }
}

namespace {
  /// PureVirtualMethodCollector - traverses a class and its superclasses
  /// and determines if it has any pure virtual methods.
  class VISIBILITY_HIDDEN PureVirtualMethodCollector {
    ASTContext &Context;

  public:
    typedef llvm::SmallVector<const CXXMethodDecl*, 8> MethodList;

  private:
    MethodList Methods;
    
    void Collect(const CXXRecordDecl* RD, MethodList& Methods);
    
  public:
    PureVirtualMethodCollector(ASTContext &Ctx, const CXXRecordDecl* RD) 
      : Context(Ctx) {
        
      MethodList List;
      Collect(RD, List);
        
      // Copy the temporary list to methods, and make sure to ignore any
      // null entries.
      for (size_t i = 0, e = List.size(); i != e; ++i) {
        if (List[i])
          Methods.push_back(List[i]);
      }          
    }
    
    bool empty() const { return Methods.empty(); }
    
    MethodList::const_iterator methods_begin() { return Methods.begin(); }
    MethodList::const_iterator methods_end() { return Methods.end(); }
  };
  
  void PureVirtualMethodCollector::Collect(const CXXRecordDecl* RD, 
                                           MethodList& Methods) {
    // First, collect the pure virtual methods for the base classes.
    for (CXXRecordDecl::base_class_const_iterator Base = RD->bases_begin(),
         BaseEnd = RD->bases_end(); Base != BaseEnd; ++Base) {
      if (const RecordType *RT = Base->getType()->getAsRecordType()) {
        const CXXRecordDecl *BaseDecl = cast<CXXRecordDecl>(RT->getDecl());
        if (BaseDecl && BaseDecl->isAbstract())
          Collect(BaseDecl, Methods);
      }
    }
    
    // Next, zero out any pure virtual methods that this class overrides.
    for (size_t i = 0, e = Methods.size(); i != e; ++i) {
      const CXXMethodDecl *VMD = dyn_cast_or_null<CXXMethodDecl>(Methods[i]);
      if (!VMD)
        continue;
      
      DeclContext::lookup_const_iterator I, E;
      for (llvm::tie(I, E) = RD->lookup(Context, VMD->getDeclName()); 
           I != E; ++I) {
        if (const CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(*I)) {
          if (Context.getCanonicalType(MD->getType()) == 
              Context.getCanonicalType(VMD->getType())) {
            // We did find a matching method, which means that this is not a
            // pure virtual method in the current class. Zero it out.
            Methods[i] = 0;
          }
        }
      }
    }
    
    // Finally, add pure virtual methods from this class.
    for (RecordDecl::decl_iterator i = RD->decls_begin(Context), 
                                   e = RD->decls_end(Context); 
         i != e; ++i) {
      if (CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(*i)) {
        if (MD->isPure())
          Methods.push_back(MD);
      }
    }
  }
}

bool Sema::RequireNonAbstractType(SourceLocation Loc, QualType T, 
                                  unsigned DiagID, AbstractDiagSelID SelID,
                                  const CXXRecordDecl *CurrentRD) {
  
  if (!getLangOptions().CPlusPlus)
    return false;
  
  if (const ArrayType *AT = Context.getAsArrayType(T))
    return RequireNonAbstractType(Loc, AT->getElementType(), DiagID, SelID,
                                  CurrentRD);
  
  if (const PointerType *PT = T->getAsPointerType()) {
    // Find the innermost pointer type.
    while (const PointerType *T = PT->getPointeeType()->getAsPointerType())
      PT = T;
    
    if (const ArrayType *AT = Context.getAsArrayType(PT->getPointeeType()))
      return RequireNonAbstractType(Loc, AT->getElementType(), DiagID, SelID,
                                    CurrentRD);
  }
  
  const RecordType *RT = T->getAsRecordType();
  if (!RT)
    return false;
  
  const CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(RT->getDecl());
  if (!RD)
    return false;

  if (CurrentRD && CurrentRD != RD)
    return false;
  
  if (!RD->isAbstract())
    return false;
  
  Diag(Loc, DiagID) << RD->getDeclName() << SelID;
  
  // Check if we've already emitted the list of pure virtual functions for this
  // class.
  if (PureVirtualClassDiagSet && PureVirtualClassDiagSet->count(RD))
    return true;
  
  PureVirtualMethodCollector Collector(Context, RD);
  
  for (PureVirtualMethodCollector::MethodList::const_iterator I = 
       Collector.methods_begin(), E = Collector.methods_end(); I != E; ++I) {
    const CXXMethodDecl *MD = *I;
    
    Diag(MD->getLocation(), diag::note_pure_virtual_function) << 
      MD->getDeclName();
  }

  if (!PureVirtualClassDiagSet)
    PureVirtualClassDiagSet.reset(new RecordDeclSetTy);
  PureVirtualClassDiagSet->insert(RD);
  
  return true;
}

namespace {
  class VISIBILITY_HIDDEN AbstractClassUsageDiagnoser 
    : public DeclVisitor<AbstractClassUsageDiagnoser, bool> {
    Sema &SemaRef;
    CXXRecordDecl *AbstractClass;
  
    bool VisitDeclContext(const DeclContext *DC) {
      bool Invalid = false;

      for (CXXRecordDecl::decl_iterator I = DC->decls_begin(SemaRef.Context),
           E = DC->decls_end(SemaRef.Context); I != E; ++I)
        Invalid |= Visit(*I);

      return Invalid;
    }
      
  public:
    AbstractClassUsageDiagnoser(Sema& SemaRef, CXXRecordDecl *ac)
      : SemaRef(SemaRef), AbstractClass(ac) {
        Visit(SemaRef.Context.getTranslationUnitDecl());
    }

    bool VisitFunctionDecl(const FunctionDecl *FD) {
      if (FD->isThisDeclarationADefinition()) {
        // No need to do the check if we're in a definition, because it requires
        // that the return/param types are complete.
        // because that requires 
        return VisitDeclContext(FD);
      }
      
      // Check the return type.
      QualType RTy = FD->getType()->getAsFunctionType()->getResultType();
      bool Invalid = 
        SemaRef.RequireNonAbstractType(FD->getLocation(), RTy,
                                       diag::err_abstract_type_in_decl,
                                       Sema::AbstractReturnType,
                                       AbstractClass);

      for (FunctionDecl::param_const_iterator I = FD->param_begin(), 
           E = FD->param_end(); I != E; ++I) {
        const ParmVarDecl *VD = *I;
        Invalid |= 
          SemaRef.RequireNonAbstractType(VD->getLocation(),
                                         VD->getOriginalType(), 
                                         diag::err_abstract_type_in_decl, 
                                         Sema::AbstractParamType,
                                         AbstractClass);
      }

      return Invalid;
    }
    
    bool VisitDecl(const Decl* D) {
      if (const DeclContext *DC = dyn_cast<DeclContext>(D))
        return VisitDeclContext(DC);
      
      return false;
    }
  };
}

void Sema::ActOnFinishCXXMemberSpecification(Scope* S, SourceLocation RLoc,
                                             DeclPtrTy TagDecl,
                                             SourceLocation LBrac,
                                             SourceLocation RBrac) {
  TemplateDecl *Template = AdjustDeclIfTemplate(TagDecl);
  ActOnFields(S, RLoc, TagDecl,
              (DeclPtrTy*)FieldCollector->getCurFields(),
              FieldCollector->getCurNumFields(), LBrac, RBrac, 0);

  CXXRecordDecl *RD = cast<CXXRecordDecl>(TagDecl.getAs<Decl>());
  if (!RD->isAbstract()) {
    // Collect all the pure virtual methods and see if this is an abstract
    // class after all.
    PureVirtualMethodCollector Collector(Context, RD);
    if (!Collector.empty()) 
      RD->setAbstract(true);
  }
  
  if (RD->isAbstract()) 
    AbstractClassUsageDiagnoser(*this, RD);
    
  if (RD->hasTrivialConstructor() || RD->hasTrivialDestructor()) {
    for (RecordDecl::field_iterator i = RD->field_begin(Context), 
         e = RD->field_end(Context); i != e; ++i) {
      // All the nonstatic data members must have trivial constructors.
      QualType FTy = i->getType();
      while (const ArrayType *AT = Context.getAsArrayType(FTy))
        FTy = AT->getElementType();
      
      if (const RecordType *RT = FTy->getAsRecordType()) {
        CXXRecordDecl *FieldRD = cast<CXXRecordDecl>(RT->getDecl());
        
        if (!FieldRD->hasTrivialConstructor())
          RD->setHasTrivialConstructor(false);
        if (!FieldRD->hasTrivialDestructor())
          RD->setHasTrivialDestructor(false);
        
        // If RD has neither a trivial constructor nor a trivial destructor
        // we don't need to continue checking.
        if (!RD->hasTrivialConstructor() && !RD->hasTrivialDestructor())
          break;
      }
    }
  }
      
  if (!Template)
    AddImplicitlyDeclaredMembersToClass(RD);
}

/// AddImplicitlyDeclaredMembersToClass - Adds any implicitly-declared
/// special functions, such as the default constructor, copy
/// constructor, or destructor, to the given C++ class (C++
/// [special]p1).  This routine can only be executed just before the
/// definition of the class is complete.
void Sema::AddImplicitlyDeclaredMembersToClass(CXXRecordDecl *ClassDecl) {
  QualType ClassType = Context.getTypeDeclType(ClassDecl);
  ClassType = Context.getCanonicalType(ClassType);

  // FIXME: Implicit declarations have exception specifications, which are
  // the union of the specifications of the implicitly called functions.

  if (!ClassDecl->hasUserDeclaredConstructor()) {
    // C++ [class.ctor]p5:
    //   A default constructor for a class X is a constructor of class X
    //   that can be called without an argument. If there is no
    //   user-declared constructor for class X, a default constructor is
    //   implicitly declared. An implicitly-declared default constructor
    //   is an inline public member of its class.
    DeclarationName Name 
      = Context.DeclarationNames.getCXXConstructorName(ClassType);
    CXXConstructorDecl *DefaultCon = 
      CXXConstructorDecl::Create(Context, ClassDecl,
                                 ClassDecl->getLocation(), Name,
                                 Context.getFunctionType(Context.VoidTy,
                                                         0, 0, false, 0),
                                 /*isExplicit=*/false,
                                 /*isInline=*/true,
                                 /*isImplicitlyDeclared=*/true);
    DefaultCon->setAccess(AS_public);
    DefaultCon->setImplicit();
    ClassDecl->addDecl(Context, DefaultCon);

    // Notify the class that we've added a constructor.
    ClassDecl->addedConstructor(Context, DefaultCon);
  }

  if (!ClassDecl->hasUserDeclaredCopyConstructor()) {
    // C++ [class.copy]p4:
    //   If the class definition does not explicitly declare a copy
    //   constructor, one is declared implicitly.

    // C++ [class.copy]p5:
    //   The implicitly-declared copy constructor for a class X will
    //   have the form
    //
    //       X::X(const X&)
    //
    //   if
    bool HasConstCopyConstructor = true;

    //     -- each direct or virtual base class B of X has a copy
    //        constructor whose first parameter is of type const B& or
    //        const volatile B&, and
    for (CXXRecordDecl::base_class_iterator Base = ClassDecl->bases_begin();
         HasConstCopyConstructor && Base != ClassDecl->bases_end(); ++Base) {
      const CXXRecordDecl *BaseClassDecl
        = cast<CXXRecordDecl>(Base->getType()->getAsRecordType()->getDecl());
      HasConstCopyConstructor 
        = BaseClassDecl->hasConstCopyConstructor(Context);
    }

    //     -- for all the nonstatic data members of X that are of a
    //        class type M (or array thereof), each such class type
    //        has a copy constructor whose first parameter is of type
    //        const M& or const volatile M&.
    for (CXXRecordDecl::field_iterator Field = ClassDecl->field_begin(Context);
         HasConstCopyConstructor && Field != ClassDecl->field_end(Context);
         ++Field) {
      QualType FieldType = (*Field)->getType();
      if (const ArrayType *Array = Context.getAsArrayType(FieldType))
        FieldType = Array->getElementType();
      if (const RecordType *FieldClassType = FieldType->getAsRecordType()) {
        const CXXRecordDecl *FieldClassDecl 
          = cast<CXXRecordDecl>(FieldClassType->getDecl());
        HasConstCopyConstructor 
          = FieldClassDecl->hasConstCopyConstructor(Context);
      }
    }

    //   Otherwise, the implicitly declared copy constructor will have
    //   the form
    //
    //       X::X(X&)
    QualType ArgType = ClassType;
    if (HasConstCopyConstructor)
      ArgType = ArgType.withConst();
    ArgType = Context.getLValueReferenceType(ArgType);

    //   An implicitly-declared copy constructor is an inline public
    //   member of its class.
    DeclarationName Name 
      = Context.DeclarationNames.getCXXConstructorName(ClassType);
    CXXConstructorDecl *CopyConstructor
      = CXXConstructorDecl::Create(Context, ClassDecl,
                                   ClassDecl->getLocation(), Name,
                                   Context.getFunctionType(Context.VoidTy,
                                                           &ArgType, 1,
                                                           false, 0),
                                   /*isExplicit=*/false,
                                   /*isInline=*/true,
                                   /*isImplicitlyDeclared=*/true);
    CopyConstructor->setAccess(AS_public);
    CopyConstructor->setImplicit();

    // Add the parameter to the constructor.
    ParmVarDecl *FromParam = ParmVarDecl::Create(Context, CopyConstructor,
                                                 ClassDecl->getLocation(),
                                                 /*IdentifierInfo=*/0,
                                                 ArgType, VarDecl::None, 0);
    CopyConstructor->setParams(Context, &FromParam, 1);

    ClassDecl->addedConstructor(Context, CopyConstructor);
    ClassDecl->addDecl(Context, CopyConstructor);
  }

  if (!ClassDecl->hasUserDeclaredCopyAssignment()) {
    // Note: The following rules are largely analoguous to the copy
    // constructor rules. Note that virtual bases are not taken into account
    // for determining the argument type of the operator. Note also that
    // operators taking an object instead of a reference are allowed.
    //
    // C++ [class.copy]p10:
    //   If the class definition does not explicitly declare a copy
    //   assignment operator, one is declared implicitly.
    //   The implicitly-defined copy assignment operator for a class X
    //   will have the form
    //
    //       X& X::operator=(const X&)
    //
    //   if
    bool HasConstCopyAssignment = true;

    //       -- each direct base class B of X has a copy assignment operator
    //          whose parameter is of type const B&, const volatile B& or B,
    //          and
    for (CXXRecordDecl::base_class_iterator Base = ClassDecl->bases_begin();
         HasConstCopyAssignment && Base != ClassDecl->bases_end(); ++Base) {
      const CXXRecordDecl *BaseClassDecl
        = cast<CXXRecordDecl>(Base->getType()->getAsRecordType()->getDecl());
      HasConstCopyAssignment = BaseClassDecl->hasConstCopyAssignment(Context);
    }

    //       -- for all the nonstatic data members of X that are of a class
    //          type M (or array thereof), each such class type has a copy
    //          assignment operator whose parameter is of type const M&,
    //          const volatile M& or M.
    for (CXXRecordDecl::field_iterator Field = ClassDecl->field_begin(Context);
         HasConstCopyAssignment && Field != ClassDecl->field_end(Context);
         ++Field) {
      QualType FieldType = (*Field)->getType();
      if (const ArrayType *Array = Context.getAsArrayType(FieldType))
        FieldType = Array->getElementType();
      if (const RecordType *FieldClassType = FieldType->getAsRecordType()) {
        const CXXRecordDecl *FieldClassDecl
          = cast<CXXRecordDecl>(FieldClassType->getDecl());
        HasConstCopyAssignment
          = FieldClassDecl->hasConstCopyAssignment(Context);
      }
    }

    //   Otherwise, the implicitly declared copy assignment operator will
    //   have the form
    //
    //       X& X::operator=(X&)
    QualType ArgType = ClassType;
    QualType RetType = Context.getLValueReferenceType(ArgType);
    if (HasConstCopyAssignment)
      ArgType = ArgType.withConst();
    ArgType = Context.getLValueReferenceType(ArgType);

    //   An implicitly-declared copy assignment operator is an inline public
    //   member of its class.
    DeclarationName Name =
      Context.DeclarationNames.getCXXOperatorName(OO_Equal);
    CXXMethodDecl *CopyAssignment =
      CXXMethodDecl::Create(Context, ClassDecl, ClassDecl->getLocation(), Name,
                            Context.getFunctionType(RetType, &ArgType, 1,
                                                    false, 0),
                            /*isStatic=*/false, /*isInline=*/true);
    CopyAssignment->setAccess(AS_public);
    CopyAssignment->setImplicit();

    // Add the parameter to the operator.
    ParmVarDecl *FromParam = ParmVarDecl::Create(Context, CopyAssignment,
                                                 ClassDecl->getLocation(),
                                                 /*IdentifierInfo=*/0,
                                                 ArgType, VarDecl::None, 0);
    CopyAssignment->setParams(Context, &FromParam, 1);

    // Don't call addedAssignmentOperator. There is no way to distinguish an
    // implicit from an explicit assignment operator.
    ClassDecl->addDecl(Context, CopyAssignment);
  }

  if (!ClassDecl->hasUserDeclaredDestructor()) {
    // C++ [class.dtor]p2:
    //   If a class has no user-declared destructor, a destructor is
    //   declared implicitly. An implicitly-declared destructor is an
    //   inline public member of its class.
    DeclarationName Name 
      = Context.DeclarationNames.getCXXDestructorName(ClassType);
    CXXDestructorDecl *Destructor 
      = CXXDestructorDecl::Create(Context, ClassDecl,
                                  ClassDecl->getLocation(), Name,
                                  Context.getFunctionType(Context.VoidTy,
                                                          0, 0, false, 0),
                                  /*isInline=*/true,
                                  /*isImplicitlyDeclared=*/true);
    Destructor->setAccess(AS_public);
    Destructor->setImplicit();
    ClassDecl->addDecl(Context, Destructor);
  }
}

/// ActOnStartDelayedCXXMethodDeclaration - We have completed
/// parsing a top-level (non-nested) C++ class, and we are now
/// parsing those parts of the given Method declaration that could
/// not be parsed earlier (C++ [class.mem]p2), such as default
/// arguments. This action should enter the scope of the given
/// Method declaration as if we had just parsed the qualified method
/// name. However, it should not bring the parameters into scope;
/// that will be performed by ActOnDelayedCXXMethodParameter.
void Sema::ActOnStartDelayedCXXMethodDeclaration(Scope *S, DeclPtrTy MethodD) {
  CXXScopeSpec SS;
  FunctionDecl *Method = cast<FunctionDecl>(MethodD.getAs<Decl>());
  QualType ClassTy 
    = Context.getTypeDeclType(cast<RecordDecl>(Method->getDeclContext()));
  SS.setScopeRep(
    NestedNameSpecifier::Create(Context, 0, false, ClassTy.getTypePtr()));
  ActOnCXXEnterDeclaratorScope(S, SS);
}

/// ActOnDelayedCXXMethodParameter - We've already started a delayed
/// C++ method declaration. We're (re-)introducing the given
/// function parameter into scope for use in parsing later parts of
/// the method declaration. For example, we could see an
/// ActOnParamDefaultArgument event for this parameter.
void Sema::ActOnDelayedCXXMethodParameter(Scope *S, DeclPtrTy ParamD) {
  ParmVarDecl *Param = cast<ParmVarDecl>(ParamD.getAs<Decl>());

  // If this parameter has an unparsed default argument, clear it out
  // to make way for the parsed default argument.
  if (Param->hasUnparsedDefaultArg())
    Param->setDefaultArg(0);

  S->AddDecl(DeclPtrTy::make(Param));
  if (Param->getDeclName())
    IdResolver.AddDecl(Param);
}

/// ActOnFinishDelayedCXXMethodDeclaration - We have finished
/// processing the delayed method declaration for Method. The method
/// declaration is now considered finished. There may be a separate
/// ActOnStartOfFunctionDef action later (not necessarily
/// immediately!) for this method, if it was also defined inside the
/// class body.
void Sema::ActOnFinishDelayedCXXMethodDeclaration(Scope *S, DeclPtrTy MethodD) {
  FunctionDecl *Method = cast<FunctionDecl>(MethodD.getAs<Decl>());
  CXXScopeSpec SS;
  QualType ClassTy 
    = Context.getTypeDeclType(cast<RecordDecl>(Method->getDeclContext()));
  SS.setScopeRep(
    NestedNameSpecifier::Create(Context, 0, false, ClassTy.getTypePtr()));
  ActOnCXXExitDeclaratorScope(S, SS);

  // Now that we have our default arguments, check the constructor
  // again. It could produce additional diagnostics or affect whether
  // the class has implicitly-declared destructors, among other
  // things.
  if (CXXConstructorDecl *Constructor = dyn_cast<CXXConstructorDecl>(Method))
    CheckConstructor(Constructor);

  // Check the default arguments, which we may have added.
  if (!Method->isInvalidDecl())
    CheckCXXDefaultArguments(Method);
}

/// CheckConstructorDeclarator - Called by ActOnDeclarator to check
/// the well-formedness of the constructor declarator @p D with type @p
/// R. If there are any errors in the declarator, this routine will
/// emit diagnostics and set the invalid bit to true.  In any case, the type
/// will be updated to reflect a well-formed type for the constructor and
/// returned.
QualType Sema::CheckConstructorDeclarator(Declarator &D, QualType R,
                                          FunctionDecl::StorageClass &SC) {
  bool isVirtual = D.getDeclSpec().isVirtualSpecified();

  // C++ [class.ctor]p3:
  //   A constructor shall not be virtual (10.3) or static (9.4). A
  //   constructor can be invoked for a const, volatile or const
  //   volatile object. A constructor shall not be declared const,
  //   volatile, or const volatile (9.3.2).
  if (isVirtual) {
    if (!D.isInvalidType())
      Diag(D.getIdentifierLoc(), diag::err_constructor_cannot_be)
        << "virtual" << SourceRange(D.getDeclSpec().getVirtualSpecLoc())
        << SourceRange(D.getIdentifierLoc());
    D.setInvalidType();
  }
  if (SC == FunctionDecl::Static) {
    if (!D.isInvalidType())
      Diag(D.getIdentifierLoc(), diag::err_constructor_cannot_be)
        << "static" << SourceRange(D.getDeclSpec().getStorageClassSpecLoc())
        << SourceRange(D.getIdentifierLoc());
    D.setInvalidType();
    SC = FunctionDecl::None;
  }
  
  DeclaratorChunk::FunctionTypeInfo &FTI = D.getTypeObject(0).Fun;
  if (FTI.TypeQuals != 0) {
    if (FTI.TypeQuals & QualType::Const)
      Diag(D.getIdentifierLoc(), diag::err_invalid_qualified_constructor)
        << "const" << SourceRange(D.getIdentifierLoc());
    if (FTI.TypeQuals & QualType::Volatile)
      Diag(D.getIdentifierLoc(), diag::err_invalid_qualified_constructor)
        << "volatile" << SourceRange(D.getIdentifierLoc());
    if (FTI.TypeQuals & QualType::Restrict)
      Diag(D.getIdentifierLoc(), diag::err_invalid_qualified_constructor)
        << "restrict" << SourceRange(D.getIdentifierLoc());
  }
      
  // Rebuild the function type "R" without any type qualifiers (in
  // case any of the errors above fired) and with "void" as the
  // return type, since constructors don't have return types. We
  // *always* have to do this, because GetTypeForDeclarator will
  // put in a result type of "int" when none was specified.
  const FunctionProtoType *Proto = R->getAsFunctionProtoType();
  return Context.getFunctionType(Context.VoidTy, Proto->arg_type_begin(),
                                 Proto->getNumArgs(),
                                 Proto->isVariadic(), 0);
}

/// CheckConstructor - Checks a fully-formed constructor for
/// well-formedness, issuing any diagnostics required. Returns true if
/// the constructor declarator is invalid.
void Sema::CheckConstructor(CXXConstructorDecl *Constructor) {
  CXXRecordDecl *ClassDecl 
    = dyn_cast<CXXRecordDecl>(Constructor->getDeclContext());
  if (!ClassDecl)
    return Constructor->setInvalidDecl();

  // C++ [class.copy]p3:
  //   A declaration of a constructor for a class X is ill-formed if
  //   its first parameter is of type (optionally cv-qualified) X and
  //   either there are no other parameters or else all other
  //   parameters have default arguments.
  if (!Constructor->isInvalidDecl() &&
      ((Constructor->getNumParams() == 1) || 
       (Constructor->getNumParams() > 1 && 
        Constructor->getParamDecl(1)->getDefaultArg() != 0))) {
    QualType ParamType = Constructor->getParamDecl(0)->getType();
    QualType ClassTy = Context.getTagDeclType(ClassDecl);
    if (Context.getCanonicalType(ParamType).getUnqualifiedType() == ClassTy) {
      SourceLocation ParamLoc = Constructor->getParamDecl(0)->getLocation();
      Diag(ParamLoc, diag::err_constructor_byvalue_arg)
        << CodeModificationHint::CreateInsertion(ParamLoc, " const &");
      Constructor->setInvalidDecl();
    }
  }
  
  // Notify the class that we've added a constructor.
  ClassDecl->addedConstructor(Context, Constructor);
}

static inline bool 
FTIHasSingleVoidArgument(DeclaratorChunk::FunctionTypeInfo &FTI) {
  return (FTI.NumArgs == 1 && !FTI.isVariadic && FTI.ArgInfo[0].Ident == 0 &&
          FTI.ArgInfo[0].Param &&
          FTI.ArgInfo[0].Param.getAs<ParmVarDecl>()->getType()->isVoidType());
}

/// CheckDestructorDeclarator - Called by ActOnDeclarator to check
/// the well-formednes of the destructor declarator @p D with type @p
/// R. If there are any errors in the declarator, this routine will
/// emit diagnostics and set the declarator to invalid.  Even if this happens,
/// will be updated to reflect a well-formed type for the destructor and
/// returned.
QualType Sema::CheckDestructorDeclarator(Declarator &D,
                                         FunctionDecl::StorageClass& SC) {
  // C++ [class.dtor]p1:
  //   [...] A typedef-name that names a class is a class-name
  //   (7.1.3); however, a typedef-name that names a class shall not
  //   be used as the identifier in the declarator for a destructor
  //   declaration.
  QualType DeclaratorType = QualType::getFromOpaquePtr(D.getDeclaratorIdType());
  if (isa<TypedefType>(DeclaratorType)) {
    Diag(D.getIdentifierLoc(), diag::err_destructor_typedef_name)
      << DeclaratorType;
    D.setInvalidType();
  }

  // C++ [class.dtor]p2:
  //   A destructor is used to destroy objects of its class type. A
  //   destructor takes no parameters, and no return type can be
  //   specified for it (not even void). The address of a destructor
  //   shall not be taken. A destructor shall not be static. A
  //   destructor can be invoked for a const, volatile or const
  //   volatile object. A destructor shall not be declared const,
  //   volatile or const volatile (9.3.2).
  if (SC == FunctionDecl::Static) {
    if (!D.isInvalidType())
      Diag(D.getIdentifierLoc(), diag::err_destructor_cannot_be)
        << "static" << SourceRange(D.getDeclSpec().getStorageClassSpecLoc())
        << SourceRange(D.getIdentifierLoc());
    SC = FunctionDecl::None;
    D.setInvalidType();
  }
  if (D.getDeclSpec().hasTypeSpecifier() && !D.isInvalidType()) {
    // Destructors don't have return types, but the parser will
    // happily parse something like:
    //
    //   class X {
    //     float ~X();
    //   };
    //
    // The return type will be eliminated later.
    Diag(D.getIdentifierLoc(), diag::err_destructor_return_type)
      << SourceRange(D.getDeclSpec().getTypeSpecTypeLoc())
      << SourceRange(D.getIdentifierLoc());
  }
  
  DeclaratorChunk::FunctionTypeInfo &FTI = D.getTypeObject(0).Fun;
  if (FTI.TypeQuals != 0 && !D.isInvalidType()) {
    if (FTI.TypeQuals & QualType::Const)
      Diag(D.getIdentifierLoc(), diag::err_invalid_qualified_destructor)
        << "const" << SourceRange(D.getIdentifierLoc());
    if (FTI.TypeQuals & QualType::Volatile)
      Diag(D.getIdentifierLoc(), diag::err_invalid_qualified_destructor)
        << "volatile" << SourceRange(D.getIdentifierLoc());
    if (FTI.TypeQuals & QualType::Restrict)
      Diag(D.getIdentifierLoc(), diag::err_invalid_qualified_destructor)
        << "restrict" << SourceRange(D.getIdentifierLoc());
    D.setInvalidType();
  }

  // Make sure we don't have any parameters.
  if (FTI.NumArgs > 0 && !FTIHasSingleVoidArgument(FTI)) {
    Diag(D.getIdentifierLoc(), diag::err_destructor_with_params);

    // Delete the parameters.
    FTI.freeArgs();
    D.setInvalidType();
  }

  // Make sure the destructor isn't variadic.  
  if (FTI.isVariadic) {
    Diag(D.getIdentifierLoc(), diag::err_destructor_variadic);
    D.setInvalidType();
  }

  // Rebuild the function type "R" without any type qualifiers or
  // parameters (in case any of the errors above fired) and with
  // "void" as the return type, since destructors don't have return
  // types. We *always* have to do this, because GetTypeForDeclarator
  // will put in a result type of "int" when none was specified.
  return Context.getFunctionType(Context.VoidTy, 0, 0, false, 0);
}

/// CheckConversionDeclarator - Called by ActOnDeclarator to check the
/// well-formednes of the conversion function declarator @p D with
/// type @p R. If there are any errors in the declarator, this routine
/// will emit diagnostics and return true. Otherwise, it will return
/// false. Either way, the type @p R will be updated to reflect a
/// well-formed type for the conversion operator.
void Sema::CheckConversionDeclarator(Declarator &D, QualType &R,
                                     FunctionDecl::StorageClass& SC) {
  // C++ [class.conv.fct]p1:
  //   Neither parameter types nor return type can be specified. The
  //   type of a conversion function (8.3.5) is “function taking no
  //   parameter returning conversion-type-id.” 
  if (SC == FunctionDecl::Static) {
    if (!D.isInvalidType())
      Diag(D.getIdentifierLoc(), diag::err_conv_function_not_member)
        << "static" << SourceRange(D.getDeclSpec().getStorageClassSpecLoc())
        << SourceRange(D.getIdentifierLoc());
    D.setInvalidType();
    SC = FunctionDecl::None;
  }
  if (D.getDeclSpec().hasTypeSpecifier() && !D.isInvalidType()) {
    // Conversion functions don't have return types, but the parser will
    // happily parse something like:
    //
    //   class X {
    //     float operator bool();
    //   };
    //
    // The return type will be changed later anyway.
    Diag(D.getIdentifierLoc(), diag::err_conv_function_return_type)
      << SourceRange(D.getDeclSpec().getTypeSpecTypeLoc())
      << SourceRange(D.getIdentifierLoc());
  }

  // Make sure we don't have any parameters.
  if (R->getAsFunctionProtoType()->getNumArgs() > 0) {
    Diag(D.getIdentifierLoc(), diag::err_conv_function_with_params);

    // Delete the parameters.
    D.getTypeObject(0).Fun.freeArgs();
    D.setInvalidType();
  }

  // Make sure the conversion function isn't variadic.  
  if (R->getAsFunctionProtoType()->isVariadic() && !D.isInvalidType()) {
    Diag(D.getIdentifierLoc(), diag::err_conv_function_variadic);
    D.setInvalidType();
  }

  // C++ [class.conv.fct]p4:
  //   The conversion-type-id shall not represent a function type nor
  //   an array type.
  QualType ConvType = QualType::getFromOpaquePtr(D.getDeclaratorIdType());
  if (ConvType->isArrayType()) {
    Diag(D.getIdentifierLoc(), diag::err_conv_function_to_array);
    ConvType = Context.getPointerType(ConvType);
    D.setInvalidType();
  } else if (ConvType->isFunctionType()) {
    Diag(D.getIdentifierLoc(), diag::err_conv_function_to_function);
    ConvType = Context.getPointerType(ConvType);
    D.setInvalidType();
  }

  // Rebuild the function type "R" without any parameters (in case any
  // of the errors above fired) and with the conversion type as the
  // return type. 
  R = Context.getFunctionType(ConvType, 0, 0, false, 
                              R->getAsFunctionProtoType()->getTypeQuals());

  // C++0x explicit conversion operators.
  if (D.getDeclSpec().isExplicitSpecified() && !getLangOptions().CPlusPlus0x)
    Diag(D.getDeclSpec().getExplicitSpecLoc(), 
         diag::warn_explicit_conversion_functions)
      << SourceRange(D.getDeclSpec().getExplicitSpecLoc());
}

/// ActOnConversionDeclarator - Called by ActOnDeclarator to complete
/// the declaration of the given C++ conversion function. This routine
/// is responsible for recording the conversion function in the C++
/// class, if possible.
Sema::DeclPtrTy Sema::ActOnConversionDeclarator(CXXConversionDecl *Conversion) {
  assert(Conversion && "Expected to receive a conversion function declaration");

  // Set the lexical context of this conversion function
  Conversion->setLexicalDeclContext(CurContext);

  CXXRecordDecl *ClassDecl = cast<CXXRecordDecl>(Conversion->getDeclContext());

  // Make sure we aren't redeclaring the conversion function.
  QualType ConvType = Context.getCanonicalType(Conversion->getConversionType());

  // C++ [class.conv.fct]p1:
  //   [...] A conversion function is never used to convert a
  //   (possibly cv-qualified) object to the (possibly cv-qualified)
  //   same object type (or a reference to it), to a (possibly
  //   cv-qualified) base class of that type (or a reference to it),
  //   or to (possibly cv-qualified) void.
  // FIXME: Suppress this warning if the conversion function ends up
  // being a virtual function that overrides a virtual function in a 
  // base class.
  QualType ClassType 
    = Context.getCanonicalType(Context.getTypeDeclType(ClassDecl));
  if (const ReferenceType *ConvTypeRef = ConvType->getAsReferenceType())
    ConvType = ConvTypeRef->getPointeeType();
  if (ConvType->isRecordType()) {
    ConvType = Context.getCanonicalType(ConvType).getUnqualifiedType();
    if (ConvType == ClassType)
      Diag(Conversion->getLocation(), diag::warn_conv_to_self_not_used)
        << ClassType;
    else if (IsDerivedFrom(ClassType, ConvType))
      Diag(Conversion->getLocation(), diag::warn_conv_to_base_not_used)
        <<  ClassType << ConvType;
  } else if (ConvType->isVoidType()) {
    Diag(Conversion->getLocation(), diag::warn_conv_to_void_not_used)
      << ClassType << ConvType;
  }

  if (Conversion->getPreviousDeclaration()) {
    OverloadedFunctionDecl *Conversions = ClassDecl->getConversionFunctions();
    for (OverloadedFunctionDecl::function_iterator 
           Conv = Conversions->function_begin(),
           ConvEnd = Conversions->function_end();
         Conv != ConvEnd; ++Conv) {
      if (*Conv == Conversion->getPreviousDeclaration()) {
        *Conv = Conversion;
        return DeclPtrTy::make(Conversion);
      }
    }
    assert(Conversion->isInvalidDecl() && "Conversion should not get here.");
  } else 
    ClassDecl->addConversionFunction(Context, Conversion);

  return DeclPtrTy::make(Conversion);
}

//===----------------------------------------------------------------------===//
// Namespace Handling
//===----------------------------------------------------------------------===//

/// ActOnStartNamespaceDef - This is called at the start of a namespace
/// definition.
Sema::DeclPtrTy Sema::ActOnStartNamespaceDef(Scope *NamespcScope,
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

    NamedDecl *PrevDecl = LookupName(DeclRegionScope, II, LookupOrdinaryName,
                                     true);
    
    if (NamespaceDecl *OrigNS = dyn_cast_or_null<NamespaceDecl>(PrevDecl)) {
      // This is an extended namespace definition.
      // Attach this namespace decl to the chain of extended namespace
      // definitions.
      OrigNS->setNextNamespace(Namespc);
      Namespc->setOriginalNamespace(OrigNS->getOriginalNamespace());

      // Remove the previous declaration from the scope.      
      if (DeclRegionScope->isDeclScope(DeclPtrTy::make(OrigNS))) {
        IdResolver.RemoveDecl(OrigNS);
        DeclRegionScope->RemoveDecl(DeclPtrTy::make(OrigNS));
      }
    } else if (PrevDecl) {
      // This is an invalid name redefinition.
      Diag(Namespc->getLocation(), diag::err_redefinition_different_kind)
       << Namespc->getDeclName();
      Diag(PrevDecl->getLocation(), diag::note_previous_definition);
      Namespc->setInvalidDecl();
      // Continue on to push Namespc as current DeclContext and return it.
    } 

    PushOnScopeChains(Namespc, DeclRegionScope);
  } else {
    // FIXME: Handle anonymous namespaces
  }

  // Although we could have an invalid decl (i.e. the namespace name is a
  // redefinition), push it as current DeclContext and try to continue parsing.
  // FIXME: We should be able to push Namespc here, so that the
  // each DeclContext for the namespace has the declarations
  // that showed up in that particular namespace definition.
  PushDeclContext(NamespcScope, Namespc);
  return DeclPtrTy::make(Namespc);
}

/// ActOnFinishNamespaceDef - This callback is called after a namespace is
/// exited. Decl is the DeclTy returned by ActOnStartNamespaceDef.
void Sema::ActOnFinishNamespaceDef(DeclPtrTy D, SourceLocation RBrace) {
  Decl *Dcl = D.getAs<Decl>();
  NamespaceDecl *Namespc = dyn_cast_or_null<NamespaceDecl>(Dcl);
  assert(Namespc && "Invalid parameter, expected NamespaceDecl");
  Namespc->setRBracLoc(RBrace);
  PopDeclContext();
}

Sema::DeclPtrTy Sema::ActOnUsingDirective(Scope *S,
                                          SourceLocation UsingLoc,
                                          SourceLocation NamespcLoc,
                                          const CXXScopeSpec &SS,
                                          SourceLocation IdentLoc,
                                          IdentifierInfo *NamespcName,
                                          AttributeList *AttrList) {
  assert(!SS.isInvalid() && "Invalid CXXScopeSpec.");
  assert(NamespcName && "Invalid NamespcName.");
  assert(IdentLoc.isValid() && "Invalid NamespceName location.");
  assert(S->getFlags() & Scope::DeclScope && "Invalid Scope.");

  UsingDirectiveDecl *UDir = 0;

  // Lookup namespace name.
  LookupResult R = LookupParsedName(S, &SS, NamespcName,
                                    LookupNamespaceName, false);
  if (R.isAmbiguous()) {
    DiagnoseAmbiguousLookup(R, NamespcName, IdentLoc);
    return DeclPtrTy();
  }
  if (NamedDecl *NS = R) {
    assert(isa<NamespaceDecl>(NS) && "expected namespace decl");
    // C++ [namespace.udir]p1:
    //   A using-directive specifies that the names in the nominated
    //   namespace can be used in the scope in which the
    //   using-directive appears after the using-directive. During
    //   unqualified name lookup (3.4.1), the names appear as if they
    //   were declared in the nearest enclosing namespace which
    //   contains both the using-directive and the nominated
    //   namespace. [Note: in this context, “contains” means “contains
    //   directly or indirectly”. ]

    // Find enclosing context containing both using-directive and
    // nominated namespace.
    DeclContext *CommonAncestor = cast<DeclContext>(NS);
    while (CommonAncestor && !CommonAncestor->Encloses(CurContext))
      CommonAncestor = CommonAncestor->getParent();

    UDir = UsingDirectiveDecl::Create(Context, CurContext, UsingLoc,
                                      NamespcLoc, IdentLoc,
                                      cast<NamespaceDecl>(NS),
                                      CommonAncestor);
    PushUsingDirective(S, UDir);
  } else {
    Diag(IdentLoc, diag::err_expected_namespace_name) << SS.getRange();
  }

  // FIXME: We ignore attributes for now.
  delete AttrList;
  return DeclPtrTy::make(UDir);
}

void Sema::PushUsingDirective(Scope *S, UsingDirectiveDecl *UDir) {
  // If scope has associated entity, then using directive is at namespace
  // or translation unit scope. We add UsingDirectiveDecls, into
  // it's lookup structure.
  if (DeclContext *Ctx = static_cast<DeclContext*>(S->getEntity()))
    Ctx->addDecl(Context, UDir);
  else
    // Otherwise it is block-sope. using-directives will affect lookup
    // only to the end of scope.
    S->PushUsingDirective(DeclPtrTy::make(UDir));
}

/// getNamespaceDecl - Returns the namespace a decl represents. If the decl
/// is a namespace alias, returns the namespace it points to.
static inline NamespaceDecl *getNamespaceDecl(NamedDecl *D) {
  if (NamespaceAliasDecl *AD = dyn_cast_or_null<NamespaceAliasDecl>(D))
    return AD->getNamespace();
  return dyn_cast_or_null<NamespaceDecl>(D);
}

Sema::DeclPtrTy Sema::ActOnNamespaceAliasDef(Scope *S, 
                                             SourceLocation NamespaceLoc,
                                             SourceLocation AliasLoc,
                                             IdentifierInfo *Alias,
                                             const CXXScopeSpec &SS,
                                             SourceLocation IdentLoc,
                                             IdentifierInfo *Ident) {
  
  // Lookup the namespace name.
  LookupResult R = LookupParsedName(S, &SS, Ident, LookupNamespaceName, false);

  // Check if we have a previous declaration with the same name.
  if (NamedDecl *PrevDecl = LookupName(S, Alias, LookupOrdinaryName, true)) {
    if (NamespaceAliasDecl *AD = dyn_cast<NamespaceAliasDecl>(PrevDecl)) {
      // We already have an alias with the same name that points to the same 
      // namespace, so don't create a new one.
      if (!R.isAmbiguous() && AD->getNamespace() == getNamespaceDecl(R))
        return DeclPtrTy();
    }
    
    unsigned DiagID = isa<NamespaceDecl>(PrevDecl) ? diag::err_redefinition :
      diag::err_redefinition_different_kind;
    Diag(AliasLoc, DiagID) << Alias;
    Diag(PrevDecl->getLocation(), diag::note_previous_definition);
    return DeclPtrTy();
  }

  if (R.isAmbiguous()) {
    DiagnoseAmbiguousLookup(R, Ident, IdentLoc);
    return DeclPtrTy();
  }
  
  if (!R) {
    Diag(NamespaceLoc, diag::err_expected_namespace_name) << SS.getRange();
    return DeclPtrTy();
  }
  
  NamespaceAliasDecl *AliasDecl = 
    NamespaceAliasDecl::Create(Context, CurContext, NamespaceLoc, AliasLoc, Alias, 
                               IdentLoc, R);
  
  CurContext->addDecl(Context, AliasDecl);
  return DeclPtrTy::make(AliasDecl);
}

void Sema::InitializeVarWithConstructor(VarDecl *VD, 
                                        CXXConstructorDecl *Constructor,
                                        QualType DeclInitType, 
                                        Expr **Exprs, unsigned NumExprs) {
  Expr *Temp = CXXConstructExpr::Create(Context, VD, DeclInitType, Constructor, 
                                        false, Exprs, NumExprs);
  VD->setInit(Temp);
}

/// AddCXXDirectInitializerToDecl - This action is called immediately after 
/// ActOnDeclarator, when a C++ direct initializer is present.
/// e.g: "int x(1);"
void Sema::AddCXXDirectInitializerToDecl(DeclPtrTy Dcl,
                                         SourceLocation LParenLoc,
                                         MultiExprArg Exprs,
                                         SourceLocation *CommaLocs,
                                         SourceLocation RParenLoc) {
  unsigned NumExprs = Exprs.size();
  assert(NumExprs != 0 && Exprs.get() && "missing expressions");
  Decl *RealDecl = Dcl.getAs<Decl>();

  // If there is no declaration, there was an error parsing it.  Just ignore
  // the initializer.
  if (RealDecl == 0)
    return;
  
  VarDecl *VDecl = dyn_cast<VarDecl>(RealDecl);
  if (!VDecl) {
    Diag(RealDecl->getLocation(), diag::err_illegal_initializer);
    RealDecl->setInvalidDecl();
    return;
  }

  // FIXME: Need to handle dependent types and expressions here.

  // We will treat direct-initialization as a copy-initialization:
  //    int x(1);  -as-> int x = 1;
  //    ClassType x(a,b,c); -as-> ClassType x = ClassType(a,b,c);
  //
  // Clients that want to distinguish between the two forms, can check for
  // direct initializer using VarDecl::hasCXXDirectInitializer().
  // A major benefit is that clients that don't particularly care about which
  // exactly form was it (like the CodeGen) can handle both cases without
  // special case code.

  // C++ 8.5p11:
  // The form of initialization (using parentheses or '=') is generally
  // insignificant, but does matter when the entity being initialized has a
  // class type.
  QualType DeclInitType = VDecl->getType();
  if (const ArrayType *Array = Context.getAsArrayType(DeclInitType))
    DeclInitType = Array->getElementType();

  // FIXME: This isn't the right place to complete the type.
  if (RequireCompleteType(VDecl->getLocation(), VDecl->getType(),
                          diag::err_typecheck_decl_incomplete_type)) {
    VDecl->setInvalidDecl();
    return;
  }

  if (VDecl->getType()->isRecordType()) {
    CXXConstructorDecl *Constructor
      = PerformInitializationByConstructor(DeclInitType,
                                           (Expr **)Exprs.get(), NumExprs,
                                           VDecl->getLocation(),
                                           SourceRange(VDecl->getLocation(),
                                                       RParenLoc),
                                           VDecl->getDeclName(),
                                           IK_Direct);
    if (!Constructor)
      RealDecl->setInvalidDecl();
    else {
      VDecl->setCXXDirectInitializer(true);
      InitializeVarWithConstructor(VDecl, Constructor, DeclInitType, 
                                   (Expr**)Exprs.release(), NumExprs);
    }
    return;
  }

  if (NumExprs > 1) {
    Diag(CommaLocs[0], diag::err_builtin_direct_init_more_than_one_arg)
      << SourceRange(VDecl->getLocation(), RParenLoc);
    RealDecl->setInvalidDecl();
    return;
  }

  // Let clients know that initialization was done with a direct initializer.
  VDecl->setCXXDirectInitializer(true);

  assert(NumExprs == 1 && "Expected 1 expression");
  // Set the init expression, handles conversions.
  AddInitializerToDecl(Dcl, ExprArg(*this, Exprs.release()[0]),
                       /*DirectInit=*/true);
}

/// PerformInitializationByConstructor - Perform initialization by
/// constructor (C++ [dcl.init]p14), which may occur as part of
/// direct-initialization or copy-initialization. We are initializing
/// an object of type @p ClassType with the given arguments @p
/// Args. @p Loc is the location in the source code where the
/// initializer occurs (e.g., a declaration, member initializer,
/// functional cast, etc.) while @p Range covers the whole
/// initialization. @p InitEntity is the entity being initialized,
/// which may by the name of a declaration or a type. @p Kind is the
/// kind of initialization we're performing, which affects whether
/// explicit constructors will be considered. When successful, returns
/// the constructor that will be used to perform the initialization;
/// when the initialization fails, emits a diagnostic and returns
/// null.
CXXConstructorDecl *
Sema::PerformInitializationByConstructor(QualType ClassType,
                                         Expr **Args, unsigned NumArgs,
                                         SourceLocation Loc, SourceRange Range,
                                         DeclarationName InitEntity,
                                         InitializationKind Kind) {
  const RecordType *ClassRec = ClassType->getAsRecordType();
  assert(ClassRec && "Can only initialize a class type here");

  // C++ [dcl.init]p14: 
  //
  //   If the initialization is direct-initialization, or if it is
  //   copy-initialization where the cv-unqualified version of the
  //   source type is the same class as, or a derived class of, the
  //   class of the destination, constructors are considered. The
  //   applicable constructors are enumerated (13.3.1.3), and the
  //   best one is chosen through overload resolution (13.3). The
  //   constructor so selected is called to initialize the object,
  //   with the initializer expression(s) as its argument(s). If no
  //   constructor applies, or the overload resolution is ambiguous,
  //   the initialization is ill-formed.
  const CXXRecordDecl *ClassDecl = cast<CXXRecordDecl>(ClassRec->getDecl());
  OverloadCandidateSet CandidateSet;

  // Add constructors to the overload set.
  DeclarationName ConstructorName 
    = Context.DeclarationNames.getCXXConstructorName(
                       Context.getCanonicalType(ClassType.getUnqualifiedType()));
  DeclContext::lookup_const_iterator Con, ConEnd;
  for (llvm::tie(Con, ConEnd) = ClassDecl->lookup(Context, ConstructorName);
       Con != ConEnd; ++Con) {
    CXXConstructorDecl *Constructor = cast<CXXConstructorDecl>(*Con);
    if ((Kind == IK_Direct) ||
        (Kind == IK_Copy && Constructor->isConvertingConstructor()) ||
        (Kind == IK_Default && Constructor->isDefaultConstructor()))
      AddOverloadCandidate(Constructor, Args, NumArgs, CandidateSet);
  }

  // FIXME: When we decide not to synthesize the implicitly-declared
  // constructors, we'll need to make them appear here.

  OverloadCandidateSet::iterator Best;
  switch (BestViableFunction(CandidateSet, Best)) {
  case OR_Success:
    // We found a constructor. Return it.
    return cast<CXXConstructorDecl>(Best->Function);
    
  case OR_No_Viable_Function:
    if (InitEntity)
      Diag(Loc, diag::err_ovl_no_viable_function_in_init)
        << InitEntity << Range;
    else
      Diag(Loc, diag::err_ovl_no_viable_function_in_init)
        << ClassType << Range;
    PrintOverloadCandidates(CandidateSet, /*OnlyViable=*/false);
    return 0;
    
  case OR_Ambiguous:
    if (InitEntity)
      Diag(Loc, diag::err_ovl_ambiguous_init) << InitEntity << Range;
    else
      Diag(Loc, diag::err_ovl_ambiguous_init) << ClassType << Range;
    PrintOverloadCandidates(CandidateSet, /*OnlyViable=*/true);
    return 0;

  case OR_Deleted:
    if (InitEntity)
      Diag(Loc, diag::err_ovl_deleted_init)
        << Best->Function->isDeleted()
        << InitEntity << Range;
    else
      Diag(Loc, diag::err_ovl_deleted_init)
        << Best->Function->isDeleted()
        << InitEntity << Range;
    PrintOverloadCandidates(CandidateSet, /*OnlyViable=*/true);
    return 0;
  }
  
  return 0;
}

/// CompareReferenceRelationship - Compare the two types T1 and T2 to
/// determine whether they are reference-related,
/// reference-compatible, reference-compatible with added
/// qualification, or incompatible, for use in C++ initialization by
/// reference (C++ [dcl.ref.init]p4). Neither type can be a reference
/// type, and the first type (T1) is the pointee type of the reference
/// type being initialized.
Sema::ReferenceCompareResult 
Sema::CompareReferenceRelationship(QualType T1, QualType T2, 
                                   bool& DerivedToBase) {
  assert(!T1->isReferenceType() &&
    "T1 must be the pointee type of the reference type");
  assert(!T2->isReferenceType() && "T2 cannot be a reference type");

  T1 = Context.getCanonicalType(T1);
  T2 = Context.getCanonicalType(T2);
  QualType UnqualT1 = T1.getUnqualifiedType();
  QualType UnqualT2 = T2.getUnqualifiedType();

  // C++ [dcl.init.ref]p4:
  //   Given types “cv1 T1” and “cv2 T2,” “cv1 T1” is
  //   reference-related to “cv2 T2” if T1 is the same type as T2, or 
  //   T1 is a base class of T2.
  if (UnqualT1 == UnqualT2)
    DerivedToBase = false;
  else if (IsDerivedFrom(UnqualT2, UnqualT1))
    DerivedToBase = true;
  else
    return Ref_Incompatible;

  // At this point, we know that T1 and T2 are reference-related (at
  // least).

  // C++ [dcl.init.ref]p4:
  //   "cv1 T1” is reference-compatible with “cv2 T2” if T1 is
  //   reference-related to T2 and cv1 is the same cv-qualification
  //   as, or greater cv-qualification than, cv2. For purposes of
  //   overload resolution, cases for which cv1 is greater
  //   cv-qualification than cv2 are identified as
  //   reference-compatible with added qualification (see 13.3.3.2).
  if (T1.getCVRQualifiers() == T2.getCVRQualifiers())
    return Ref_Compatible;
  else if (T1.isMoreQualifiedThan(T2))
    return Ref_Compatible_With_Added_Qualification;
  else
    return Ref_Related;
}

/// CheckReferenceInit - Check the initialization of a reference
/// variable with the given initializer (C++ [dcl.init.ref]). Init is
/// the initializer (either a simple initializer or an initializer
/// list), and DeclType is the type of the declaration. When ICS is
/// non-null, this routine will compute the implicit conversion
/// sequence according to C++ [over.ics.ref] and will not produce any
/// diagnostics; when ICS is null, it will emit diagnostics when any
/// errors are found. Either way, a return value of true indicates
/// that there was a failure, a return value of false indicates that
/// the reference initialization succeeded.
///
/// When @p SuppressUserConversions, user-defined conversions are
/// suppressed.
/// When @p AllowExplicit, we also permit explicit user-defined
/// conversion functions.
/// When @p ForceRValue, we unconditionally treat the initializer as an rvalue.
bool 
Sema::CheckReferenceInit(Expr *&Init, QualType DeclType,
                         ImplicitConversionSequence *ICS,
                         bool SuppressUserConversions,
                         bool AllowExplicit, bool ForceRValue) {
  assert(DeclType->isReferenceType() && "Reference init needs a reference");

  QualType T1 = DeclType->getAsReferenceType()->getPointeeType();
  QualType T2 = Init->getType();

  // If the initializer is the address of an overloaded function, try
  // to resolve the overloaded function. If all goes well, T2 is the
  // type of the resulting function.
  if (Context.getCanonicalType(T2) == Context.OverloadTy) {
    FunctionDecl *Fn = ResolveAddressOfOverloadedFunction(Init, DeclType, 
                                                          ICS != 0);
    if (Fn) {
      // Since we're performing this reference-initialization for
      // real, update the initializer with the resulting function.
      if (!ICS) {
        if (DiagnoseUseOfDecl(Fn, Init->getSourceRange().getBegin()))
          return true;

        FixOverloadedFunctionReference(Init, Fn);
      }

      T2 = Fn->getType();
    }
  }

  // Compute some basic properties of the types and the initializer.
  bool isRValRef = DeclType->isRValueReferenceType();
  bool DerivedToBase = false;
  Expr::isLvalueResult InitLvalue = ForceRValue ? Expr::LV_InvalidExpression :
                                                  Init->isLvalue(Context);
  ReferenceCompareResult RefRelationship 
    = CompareReferenceRelationship(T1, T2, DerivedToBase);

  // Most paths end in a failed conversion.
  if (ICS)
    ICS->ConversionKind = ImplicitConversionSequence::BadConversion;

  // C++ [dcl.init.ref]p5:
  //   A reference to type “cv1 T1” is initialized by an expression
  //   of type “cv2 T2” as follows:

  //     -- If the initializer expression

  // Rvalue references cannot bind to lvalues (N2812).
  // There is absolutely no situation where they can. In particular, note that
  // this is ill-formed, even if B has a user-defined conversion to A&&:
  //   B b;
  //   A&& r = b;
  if (isRValRef && InitLvalue == Expr::LV_Valid) {
    if (!ICS)
      Diag(Init->getSourceRange().getBegin(), diag::err_lvalue_to_rvalue_ref)
        << Init->getSourceRange();
    return true;
  }

  bool BindsDirectly = false;
  //       -- is an lvalue (but is not a bit-field), and “cv1 T1” is
  //          reference-compatible with “cv2 T2,” or
  //
  // Note that the bit-field check is skipped if we are just computing
  // the implicit conversion sequence (C++ [over.best.ics]p2).
  if (InitLvalue == Expr::LV_Valid && (ICS || !Init->getBitField()) &&
      RefRelationship >= Ref_Compatible_With_Added_Qualification) {
    BindsDirectly = true;

    if (ICS) {
      // C++ [over.ics.ref]p1:
      //   When a parameter of reference type binds directly (8.5.3)
      //   to an argument expression, the implicit conversion sequence
      //   is the identity conversion, unless the argument expression
      //   has a type that is a derived class of the parameter type,
      //   in which case the implicit conversion sequence is a
      //   derived-to-base Conversion (13.3.3.1).
      ICS->ConversionKind = ImplicitConversionSequence::StandardConversion;
      ICS->Standard.First = ICK_Identity;
      ICS->Standard.Second = DerivedToBase? ICK_Derived_To_Base : ICK_Identity;
      ICS->Standard.Third = ICK_Identity;
      ICS->Standard.FromTypePtr = T2.getAsOpaquePtr();
      ICS->Standard.ToTypePtr = T1.getAsOpaquePtr();
      ICS->Standard.ReferenceBinding = true;
      ICS->Standard.DirectBinding = true;
      ICS->Standard.RRefBinding = false;
      ICS->Standard.CopyConstructor = 0;

      // Nothing more to do: the inaccessibility/ambiguity check for
      // derived-to-base conversions is suppressed when we're
      // computing the implicit conversion sequence (C++
      // [over.best.ics]p2).
      return false;
    } else {
      // Perform the conversion.
      // FIXME: Binding to a subobject of the lvalue is going to require
      // more AST annotation than this.
      ImpCastExprToType(Init, T1, /*isLvalue=*/true);    
    }
  }

  //       -- has a class type (i.e., T2 is a class type) and can be
  //          implicitly converted to an lvalue of type “cv3 T3,”
  //          where “cv1 T1” is reference-compatible with “cv3 T3”
  //          92) (this conversion is selected by enumerating the
  //          applicable conversion functions (13.3.1.6) and choosing
  //          the best one through overload resolution (13.3)),
  if (!isRValRef && !SuppressUserConversions && T2->isRecordType()) {
    // FIXME: Look for conversions in base classes!
    CXXRecordDecl *T2RecordDecl 
      = dyn_cast<CXXRecordDecl>(T2->getAsRecordType()->getDecl());

    OverloadCandidateSet CandidateSet;
    OverloadedFunctionDecl *Conversions 
      = T2RecordDecl->getConversionFunctions();
    for (OverloadedFunctionDecl::function_iterator Func 
           = Conversions->function_begin();
         Func != Conversions->function_end(); ++Func) {
      CXXConversionDecl *Conv = cast<CXXConversionDecl>(*Func);

      // If the conversion function doesn't return a reference type,
      // it can't be considered for this conversion.
      if (Conv->getConversionType()->isLValueReferenceType() &&
          (AllowExplicit || !Conv->isExplicit()))
        AddConversionCandidate(Conv, Init, DeclType, CandidateSet);
    }

    OverloadCandidateSet::iterator Best;
    switch (BestViableFunction(CandidateSet, Best)) {
    case OR_Success:
      // This is a direct binding.
      BindsDirectly = true;

      if (ICS) {
        // C++ [over.ics.ref]p1:
        //
        //   [...] If the parameter binds directly to the result of
        //   applying a conversion function to the argument
        //   expression, the implicit conversion sequence is a
        //   user-defined conversion sequence (13.3.3.1.2), with the
        //   second standard conversion sequence either an identity
        //   conversion or, if the conversion function returns an
        //   entity of a type that is a derived class of the parameter
        //   type, a derived-to-base Conversion.
        ICS->ConversionKind = ImplicitConversionSequence::UserDefinedConversion;
        ICS->UserDefined.Before = Best->Conversions[0].Standard;
        ICS->UserDefined.After = Best->FinalConversion;
        ICS->UserDefined.ConversionFunction = Best->Function;
        assert(ICS->UserDefined.After.ReferenceBinding &&
               ICS->UserDefined.After.DirectBinding &&
               "Expected a direct reference binding!");
        return false;
      } else {
        // Perform the conversion.
        // FIXME: Binding to a subobject of the lvalue is going to require
        // more AST annotation than this.
        ImpCastExprToType(Init, T1, /*isLvalue=*/true);
      }
      break;

    case OR_Ambiguous:
      assert(false && "Ambiguous reference binding conversions not implemented.");
      return true;
      
    case OR_No_Viable_Function:
    case OR_Deleted:
      // There was no suitable conversion, or we found a deleted
      // conversion; continue with other checks.
      break;
    }
  }
      
  if (BindsDirectly) {
    // C++ [dcl.init.ref]p4:
    //   [...] In all cases where the reference-related or
    //   reference-compatible relationship of two types is used to
    //   establish the validity of a reference binding, and T1 is a
    //   base class of T2, a program that necessitates such a binding
    //   is ill-formed if T1 is an inaccessible (clause 11) or
    //   ambiguous (10.2) base class of T2.
    //
    // Note that we only check this condition when we're allowed to
    // complain about errors, because we should not be checking for
    // ambiguity (or inaccessibility) unless the reference binding
    // actually happens.
    if (DerivedToBase) 
      return CheckDerivedToBaseConversion(T2, T1, 
                                          Init->getSourceRange().getBegin(),
                                          Init->getSourceRange());
    else
      return false;
  }

  //     -- Otherwise, the reference shall be to a non-volatile const
  //        type (i.e., cv1 shall be const), or the reference shall be an
  //        rvalue reference and the initializer expression shall be an rvalue.
  if (!isRValRef && T1.getCVRQualifiers() != QualType::Const) {
    if (!ICS)
      Diag(Init->getSourceRange().getBegin(),
           diag::err_not_reference_to_const_init)
        << T1 << (InitLvalue != Expr::LV_Valid? "temporary" : "value")
        << T2 << Init->getSourceRange();
    return true;
  }

  //       -- If the initializer expression is an rvalue, with T2 a
  //          class type, and “cv1 T1” is reference-compatible with
  //          “cv2 T2,” the reference is bound in one of the
  //          following ways (the choice is implementation-defined):
  //
  //          -- The reference is bound to the object represented by
  //             the rvalue (see 3.10) or to a sub-object within that
  //             object.
  //
  //          -- A temporary of type “cv1 T2” [sic] is created, and
  //             a constructor is called to copy the entire rvalue
  //             object into the temporary. The reference is bound to
  //             the temporary or to a sub-object within the
  //             temporary.
  //
  //          The constructor that would be used to make the copy
  //          shall be callable whether or not the copy is actually
  //          done.
  //
  // Note that C++0x [dcl.init.ref]p5 takes away this implementation
  // freedom, so we will always take the first option and never build
  // a temporary in this case. FIXME: We will, however, have to check
  // for the presence of a copy constructor in C++98/03 mode.
  if (InitLvalue != Expr::LV_Valid && T2->isRecordType() &&
      RefRelationship >= Ref_Compatible_With_Added_Qualification) {
    if (ICS) {
      ICS->ConversionKind = ImplicitConversionSequence::StandardConversion;
      ICS->Standard.First = ICK_Identity;
      ICS->Standard.Second = DerivedToBase? ICK_Derived_To_Base : ICK_Identity;
      ICS->Standard.Third = ICK_Identity;
      ICS->Standard.FromTypePtr = T2.getAsOpaquePtr();
      ICS->Standard.ToTypePtr = T1.getAsOpaquePtr();
      ICS->Standard.ReferenceBinding = true;
      ICS->Standard.DirectBinding = false;
      ICS->Standard.RRefBinding = isRValRef;
      ICS->Standard.CopyConstructor = 0;
    } else {
      // FIXME: Binding to a subobject of the rvalue is going to require
      // more AST annotation than this.
      ImpCastExprToType(Init, T1, /*isLvalue=*/true);
    }
    return false;
  }

  //       -- Otherwise, a temporary of type “cv1 T1” is created and
  //          initialized from the initializer expression using the
  //          rules for a non-reference copy initialization (8.5). The
  //          reference is then bound to the temporary. If T1 is
  //          reference-related to T2, cv1 must be the same
  //          cv-qualification as, or greater cv-qualification than,
  //          cv2; otherwise, the program is ill-formed.
  if (RefRelationship == Ref_Related) {
    // If cv1 == cv2 or cv1 is a greater cv-qualified than cv2, then
    // we would be reference-compatible or reference-compatible with
    // added qualification. But that wasn't the case, so the reference
    // initialization fails.
    if (!ICS)
      Diag(Init->getSourceRange().getBegin(),
           diag::err_reference_init_drops_quals)
        << T1 << (InitLvalue != Expr::LV_Valid? "temporary" : "value")
        << T2 << Init->getSourceRange();
    return true;
  }

  // If at least one of the types is a class type, the types are not
  // related, and we aren't allowed any user conversions, the
  // reference binding fails. This case is important for breaking
  // recursion, since TryImplicitConversion below will attempt to
  // create a temporary through the use of a copy constructor.
  if (SuppressUserConversions && RefRelationship == Ref_Incompatible &&
      (T1->isRecordType() || T2->isRecordType())) {
    if (!ICS)
      Diag(Init->getSourceRange().getBegin(),
           diag::err_typecheck_convert_incompatible)
        << DeclType << Init->getType() << "initializing" << Init->getSourceRange();
    return true;
  }

  // Actually try to convert the initializer to T1.
  if (ICS) {
    // C++ [over.ics.ref]p2:
    // 
    //   When a parameter of reference type is not bound directly to
    //   an argument expression, the conversion sequence is the one
    //   required to convert the argument expression to the
    //   underlying type of the reference according to
    //   13.3.3.1. Conceptually, this conversion sequence corresponds
    //   to copy-initializing a temporary of the underlying type with
    //   the argument expression. Any difference in top-level
    //   cv-qualification is subsumed by the initialization itself
    //   and does not constitute a conversion.
    *ICS = TryImplicitConversion(Init, T1, SuppressUserConversions);
    // Of course, that's still a reference binding.
    if (ICS->ConversionKind == ImplicitConversionSequence::StandardConversion) {
      ICS->Standard.ReferenceBinding = true;
      ICS->Standard.RRefBinding = isRValRef;
    } else if(ICS->ConversionKind ==
              ImplicitConversionSequence::UserDefinedConversion) {
      ICS->UserDefined.After.ReferenceBinding = true;
      ICS->UserDefined.After.RRefBinding = isRValRef;
    }
    return ICS->ConversionKind == ImplicitConversionSequence::BadConversion;
  } else {
    return PerformImplicitConversion(Init, T1, "initializing");
  }
}

/// CheckOverloadedOperatorDeclaration - Check whether the declaration
/// of this overloaded operator is well-formed. If so, returns false;
/// otherwise, emits appropriate diagnostics and returns true.
bool Sema::CheckOverloadedOperatorDeclaration(FunctionDecl *FnDecl) {
  assert(FnDecl && FnDecl->isOverloadedOperator() &&
         "Expected an overloaded operator declaration");

  OverloadedOperatorKind Op = FnDecl->getOverloadedOperator();

  // C++ [over.oper]p5: 
  //   The allocation and deallocation functions, operator new,
  //   operator new[], operator delete and operator delete[], are
  //   described completely in 3.7.3. The attributes and restrictions
  //   found in the rest of this subclause do not apply to them unless
  //   explicitly stated in 3.7.3.
  // FIXME: Write a separate routine for checking this. For now, just 
  // allow it.
  if (Op == OO_New || Op == OO_Array_New ||
      Op == OO_Delete || Op == OO_Array_Delete)
    return false;

  // C++ [over.oper]p6:
  //   An operator function shall either be a non-static member
  //   function or be a non-member function and have at least one
  //   parameter whose type is a class, a reference to a class, an
  //   enumeration, or a reference to an enumeration.
  if (CXXMethodDecl *MethodDecl = dyn_cast<CXXMethodDecl>(FnDecl)) {
    if (MethodDecl->isStatic())
      return Diag(FnDecl->getLocation(),
                  diag::err_operator_overload_static) << FnDecl->getDeclName();
  } else {
    bool ClassOrEnumParam = false;
    for (FunctionDecl::param_iterator Param = FnDecl->param_begin(),
                                   ParamEnd = FnDecl->param_end();
         Param != ParamEnd; ++Param) {
      QualType ParamType = (*Param)->getType().getNonReferenceType();
      if (ParamType->isRecordType() || ParamType->isEnumeralType()) {
        ClassOrEnumParam = true;
        break;
      }
    }

    if (!ClassOrEnumParam)
      return Diag(FnDecl->getLocation(),
                  diag::err_operator_overload_needs_class_or_enum)
        << FnDecl->getDeclName();
  }

  // C++ [over.oper]p8:
  //   An operator function cannot have default arguments (8.3.6),
  //   except where explicitly stated below.
  //
  // Only the function-call operator allows default arguments 
  // (C++ [over.call]p1).
  if (Op != OO_Call) {
    for (FunctionDecl::param_iterator Param = FnDecl->param_begin();
         Param != FnDecl->param_end(); ++Param) {
      if ((*Param)->hasUnparsedDefaultArg())
        return Diag((*Param)->getLocation(), 
                    diag::err_operator_overload_default_arg)
          << FnDecl->getDeclName();
      else if (Expr *DefArg = (*Param)->getDefaultArg())
        return Diag((*Param)->getLocation(),
                    diag::err_operator_overload_default_arg)
          << FnDecl->getDeclName() << DefArg->getSourceRange();
    }
  }

  static const bool OperatorUses[NUM_OVERLOADED_OPERATORS][3] = {
    { false, false, false }
#define OVERLOADED_OPERATOR(Name,Spelling,Token,Unary,Binary,MemberOnly) \
    , { Unary, Binary, MemberOnly }
#include "clang/Basic/OperatorKinds.def"
  };

  bool CanBeUnaryOperator = OperatorUses[Op][0];
  bool CanBeBinaryOperator = OperatorUses[Op][1];
  bool MustBeMemberOperator = OperatorUses[Op][2];

  // C++ [over.oper]p8:
  //   [...] Operator functions cannot have more or fewer parameters
  //   than the number required for the corresponding operator, as
  //   described in the rest of this subclause.
  unsigned NumParams = FnDecl->getNumParams() 
                     + (isa<CXXMethodDecl>(FnDecl)? 1 : 0);
  if (Op != OO_Call &&
      ((NumParams == 1 && !CanBeUnaryOperator) ||
       (NumParams == 2 && !CanBeBinaryOperator) ||
       (NumParams < 1) || (NumParams > 2))) {
    // We have the wrong number of parameters.
    unsigned ErrorKind;
    if (CanBeUnaryOperator && CanBeBinaryOperator) {
      ErrorKind = 2;  // 2 -> unary or binary.
    } else if (CanBeUnaryOperator) {
      ErrorKind = 0;  // 0 -> unary
    } else {
      assert(CanBeBinaryOperator &&
             "All non-call overloaded operators are unary or binary!");
      ErrorKind = 1;  // 1 -> binary
    }

    return Diag(FnDecl->getLocation(), diag::err_operator_overload_must_be)
      << FnDecl->getDeclName() << NumParams << ErrorKind;
  }

  // Overloaded operators other than operator() cannot be variadic.
  if (Op != OO_Call &&
      FnDecl->getType()->getAsFunctionProtoType()->isVariadic()) {
    return Diag(FnDecl->getLocation(), diag::err_operator_overload_variadic)
      << FnDecl->getDeclName();
  }

  // Some operators must be non-static member functions.
  if (MustBeMemberOperator && !isa<CXXMethodDecl>(FnDecl)) {
    return Diag(FnDecl->getLocation(),
                diag::err_operator_overload_must_be_member)
      << FnDecl->getDeclName();
  }

  // C++ [over.inc]p1:
  //   The user-defined function called operator++ implements the
  //   prefix and postfix ++ operator. If this function is a member
  //   function with no parameters, or a non-member function with one
  //   parameter of class or enumeration type, it defines the prefix
  //   increment operator ++ for objects of that type. If the function
  //   is a member function with one parameter (which shall be of type
  //   int) or a non-member function with two parameters (the second
  //   of which shall be of type int), it defines the postfix
  //   increment operator ++ for objects of that type.
  if ((Op == OO_PlusPlus || Op == OO_MinusMinus) && NumParams == 2) {
    ParmVarDecl *LastParam = FnDecl->getParamDecl(FnDecl->getNumParams() - 1);
    bool ParamIsInt = false;
    if (const BuiltinType *BT = LastParam->getType()->getAsBuiltinType())
      ParamIsInt = BT->getKind() == BuiltinType::Int;

    if (!ParamIsInt)
      return Diag(LastParam->getLocation(),
                  diag::err_operator_overload_post_incdec_must_be_int) 
        << LastParam->getType() << (Op == OO_MinusMinus);
  }

  // Notify the class if it got an assignment operator.
  if (Op == OO_Equal) {
    // Would have returned earlier otherwise.
    assert(isa<CXXMethodDecl>(FnDecl) &&
      "Overloaded = not member, but not filtered.");
    CXXMethodDecl *Method = cast<CXXMethodDecl>(FnDecl);
    Method->getParent()->addedAssignmentOperator(Context, Method);
  }

  return false;
}

/// ActOnStartLinkageSpecification - Parsed the beginning of a C++
/// linkage specification, including the language and (if present)
/// the '{'. ExternLoc is the location of the 'extern', LangLoc is
/// the location of the language string literal, which is provided
/// by Lang/StrSize. LBraceLoc, if valid, provides the location of
/// the '{' brace. Otherwise, this linkage specification does not
/// have any braces.
Sema::DeclPtrTy Sema::ActOnStartLinkageSpecification(Scope *S,
                                                     SourceLocation ExternLoc,
                                                     SourceLocation LangLoc,
                                                     const char *Lang,
                                                     unsigned StrSize,
                                                     SourceLocation LBraceLoc) {
  LinkageSpecDecl::LanguageIDs Language;
  if (strncmp(Lang, "\"C\"", StrSize) == 0)
    Language = LinkageSpecDecl::lang_c;
  else if (strncmp(Lang, "\"C++\"", StrSize) == 0)
    Language = LinkageSpecDecl::lang_cxx;
  else {
    Diag(LangLoc, diag::err_bad_language);
    return DeclPtrTy();
  }
  
  // FIXME: Add all the various semantics of linkage specifications
  
  LinkageSpecDecl *D = LinkageSpecDecl::Create(Context, CurContext,
                                               LangLoc, Language, 
                                               LBraceLoc.isValid());
  CurContext->addDecl(Context, D);
  PushDeclContext(S, D);
  return DeclPtrTy::make(D);
}

/// ActOnFinishLinkageSpecification - Completely the definition of
/// the C++ linkage specification LinkageSpec. If RBraceLoc is
/// valid, it's the position of the closing '}' brace in a linkage
/// specification that uses braces.
Sema::DeclPtrTy Sema::ActOnFinishLinkageSpecification(Scope *S,
                                                      DeclPtrTy LinkageSpec,
                                                      SourceLocation RBraceLoc) {
  if (LinkageSpec)
    PopDeclContext();
  return LinkageSpec;
}

/// ActOnExceptionDeclarator - Parsed the exception-declarator in a C++ catch
/// handler.
Sema::DeclPtrTy Sema::ActOnExceptionDeclarator(Scope *S, Declarator &D) {
  QualType ExDeclType = GetTypeForDeclarator(D, S);
  SourceLocation Begin = D.getDeclSpec().getSourceRange().getBegin();

  bool Invalid = D.isInvalidType();

  // Arrays and functions decay.
  if (ExDeclType->isArrayType())
    ExDeclType = Context.getArrayDecayedType(ExDeclType);
  else if (ExDeclType->isFunctionType())
    ExDeclType = Context.getPointerType(ExDeclType);

  // C++ 15.3p1: The exception-declaration shall not denote an incomplete type.
  // The exception-declaration shall not denote a pointer or reference to an
  // incomplete type, other than [cv] void*.
  // N2844 forbids rvalue references.
  if(ExDeclType->isRValueReferenceType()) {
    Diag(Begin, diag::err_catch_rvalue_ref) << D.getSourceRange();
    Invalid = true;
  }
  QualType BaseType = ExDeclType;
  int Mode = 0; // 0 for direct type, 1 for pointer, 2 for reference
  unsigned DK = diag::err_catch_incomplete;
  if (const PointerType *Ptr = BaseType->getAsPointerType()) {
    BaseType = Ptr->getPointeeType();
    Mode = 1;
    DK = diag::err_catch_incomplete_ptr;
  } else if(const ReferenceType *Ref = BaseType->getAsReferenceType()) {
    // For the purpose of error recovery, we treat rvalue refs like lvalue refs.
    BaseType = Ref->getPointeeType();
    Mode = 2;
    DK = diag::err_catch_incomplete_ref;
  }
  if (!Invalid && (Mode == 0 || !BaseType->isVoidType()) &&
      RequireCompleteType(Begin, BaseType, DK))
    Invalid = true;

  if (!Invalid && RequireNonAbstractType(Begin, ExDeclType,
                                         diag::err_abstract_type_in_decl,
                                         AbstractVariableType))
    Invalid = true;

  // FIXME: Need to test for ability to copy-construct and destroy the
  // exception variable.
  // FIXME: Need to check for abstract classes.

  IdentifierInfo *II = D.getIdentifier();
  if (NamedDecl *PrevDecl = LookupName(S, II, LookupOrdinaryName)) {
    // The scope should be freshly made just for us. There is just no way
    // it contains any previous declaration.
    assert(!S->isDeclScope(DeclPtrTy::make(PrevDecl)));
    if (PrevDecl->isTemplateParameter()) {
      // Maybe we will complain about the shadowed template parameter.
      DiagnoseTemplateParameterShadow(D.getIdentifierLoc(), PrevDecl);
    }
  }

  VarDecl *ExDecl = VarDecl::Create(Context, CurContext, D.getIdentifierLoc(),
                                    II, ExDeclType, VarDecl::None, Begin);
  if (D.getCXXScopeSpec().isSet() && !Invalid) {
    Diag(D.getIdentifierLoc(), diag::err_qualified_catch_declarator)
      << D.getCXXScopeSpec().getRange();
    Invalid = true;
  }

  if (Invalid)
    ExDecl->setInvalidDecl();
  
  // Add the exception declaration into this scope.
  S->AddDecl(DeclPtrTy::make(ExDecl));
  if (II)
    IdResolver.AddDecl(ExDecl);

  ProcessDeclAttributes(ExDecl, D);
  return DeclPtrTy::make(ExDecl);
}

Sema::DeclPtrTy Sema::ActOnStaticAssertDeclaration(SourceLocation AssertLoc, 
                                                   ExprArg assertexpr,
                                                   ExprArg assertmessageexpr) {
  Expr *AssertExpr = (Expr *)assertexpr.get();
  StringLiteral *AssertMessage = 
    cast<StringLiteral>((Expr *)assertmessageexpr.get());

  if (!AssertExpr->isTypeDependent() && !AssertExpr->isValueDependent()) {
    llvm::APSInt Value(32);
    if (!AssertExpr->isIntegerConstantExpr(Value, Context)) {
      Diag(AssertLoc, diag::err_static_assert_expression_is_not_constant) <<
        AssertExpr->getSourceRange();
      return DeclPtrTy();
    }

    if (Value == 0) {
      std::string str(AssertMessage->getStrData(), 
                      AssertMessage->getByteLength());
      Diag(AssertLoc, diag::err_static_assert_failed) 
        << str << AssertExpr->getSourceRange();
    }
  }
  
  assertexpr.release();
  assertmessageexpr.release();
  Decl *Decl = StaticAssertDecl::Create(Context, CurContext, AssertLoc, 
                                        AssertExpr, AssertMessage);
  
  CurContext->addDecl(Context, Decl);
  return DeclPtrTy::make(Decl);
}

void Sema::SetDeclDeleted(DeclPtrTy dcl, SourceLocation DelLoc) {
  Decl *Dcl = dcl.getAs<Decl>();
  FunctionDecl *Fn = dyn_cast<FunctionDecl>(Dcl);
  if (!Fn) {
    Diag(DelLoc, diag::err_deleted_non_function);
    return;
  }
  if (const FunctionDecl *Prev = Fn->getPreviousDeclaration()) {
    Diag(DelLoc, diag::err_deleted_decl_not_first);
    Diag(Prev->getLocation(), diag::note_previous_declaration);
    // If the declaration wasn't the first, we delete the function anyway for
    // recovery.
  }
  Fn->setDeleted();
}

static void SearchForReturnInStmt(Sema &Self, Stmt *S) {
  for (Stmt::child_iterator CI = S->child_begin(), E = S->child_end(); CI != E;
       ++CI) {
    Stmt *SubStmt = *CI;
    if (!SubStmt)
      continue;
    if (isa<ReturnStmt>(SubStmt))
      Self.Diag(SubStmt->getSourceRange().getBegin(),
           diag::err_return_in_constructor_handler);
    if (!isa<Expr>(SubStmt))
      SearchForReturnInStmt(Self, SubStmt);
  }
}

void Sema::DiagnoseReturnInConstructorExceptionHandler(CXXTryStmt *TryBlock) {
  for (unsigned I = 0, E = TryBlock->getNumHandlers(); I != E; ++I) {
    CXXCatchStmt *Handler = TryBlock->getHandler(I);
    SearchForReturnInStmt(*this, Handler);
  }
}
