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
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/TypeOrdering.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Parse/DeclSpec.h"
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

/// isCurrentClassName - Determine whether the identifier II is the
/// name of the class type currently being defined. In the case of
/// nested classes, this will only return true if II is the name of
/// the innermost class.
bool Sema::isCurrentClassName(const IdentifierInfo &II, Scope *) {
  if (CXXRecordDecl *CurDecl = dyn_cast_or_null<CXXRecordDecl>(CurContext))
    return &II == CurDecl->getIdentifier();
  else
    return false;
}

/// ActOnBaseSpecifier - Parsed a base specifier. A base specifier is
/// one entry in the base class list of a class specifier, for
/// example: 
///    class foo : public bar, virtual private baz { 
/// 'public bar' and 'virtual private baz' are each base-specifiers.
Sema::BaseResult 
Sema::ActOnBaseSpecifier(DeclTy *classdecl, SourceRange SpecifierRange,
                         bool Virtual, AccessSpecifier Access,
                         TypeTy *basetype, SourceLocation BaseLoc) {
  RecordDecl *Decl = (RecordDecl*)classdecl;
  QualType BaseType = Context.getTypeDeclType((TypeDecl*)basetype);

  // Base specifiers must be record types.
  if (!BaseType->isRecordType()) {
    Diag(BaseLoc, diag::err_base_must_be_class, SpecifierRange);
    return true;
  }

  // C++ [class.union]p1:
  //   A union shall not be used as a base class.
  if (BaseType->isUnionType()) {
    Diag(BaseLoc, diag::err_union_as_base_class, SpecifierRange);
    return true;
  }

  // C++ [class.union]p1:
  //   A union shall not have base classes.
  if (Decl->isUnion()) {
    Diag(Decl->getLocation(), diag::err_base_clause_on_union,
         SpecifierRange);
    return true;
  }

  // C++ [class.derived]p2:
  //   The class-name in a base-specifier shall not be an incompletely
  //   defined class.
  if (BaseType->isIncompleteType()) {
    Diag(BaseLoc, diag::err_incomplete_base_class, SpecifierRange);
    return true;
  }

  // Create the base specifier.
  return new CXXBaseSpecifier(SpecifierRange, Virtual, 
                              BaseType->isClassType(), Access, BaseType);
}

/// ActOnBaseSpecifiers - Attach the given base specifiers to the
/// class, after checking whether there are any duplicate base
/// classes.
void Sema::ActOnBaseSpecifiers(DeclTy *ClassDecl, BaseTy **Bases, 
                               unsigned NumBases) {
  if (NumBases == 0)
    return;

  // Used to keep track of which base types we have already seen, so
  // that we can properly diagnose redundant direct base types. Note
  // that the key is always the unqualified canonical type of the base
  // class.
  std::map<QualType, CXXBaseSpecifier*, QualTypeOrdering> KnownBaseTypes;

  // Copy non-redundant base specifiers into permanent storage.
  CXXBaseSpecifier **BaseSpecs = (CXXBaseSpecifier **)Bases;
  unsigned NumGoodBases = 0;
  for (unsigned idx = 0; idx < NumBases; ++idx) {
    QualType NewBaseType 
      = Context.getCanonicalType(BaseSpecs[idx]->getType());
    NewBaseType = NewBaseType.getUnqualifiedType();

    if (KnownBaseTypes[NewBaseType]) {
      // C++ [class.mi]p3:
      //   A class shall not be specified as a direct base class of a
      //   derived class more than once.
      Diag(BaseSpecs[idx]->getSourceRange().getBegin(),
           diag::err_duplicate_base_class, 
           KnownBaseTypes[NewBaseType]->getType().getAsString(),
           BaseSpecs[idx]->getSourceRange());

      // Delete the duplicate base class specifier; we're going to
      // overwrite its pointer later.
      delete BaseSpecs[idx];
    } else {
      // Okay, add this new base class.
      KnownBaseTypes[NewBaseType] = BaseSpecs[idx];
      BaseSpecs[NumGoodBases++] = BaseSpecs[idx];
    }
  }

  // Attach the remaining base class specifiers to the derived class.
  CXXRecordDecl *Decl = (CXXRecordDecl*)ClassDecl;
  Decl->setBases(BaseSpecs, NumGoodBases);

  // Delete the remaining (good) base class specifiers, since their
  // data has been copied into the CXXRecordDecl.
  for (unsigned idx = 0; idx < NumGoodBases; ++idx)
    delete BaseSpecs[idx];
}

//===----------------------------------------------------------------------===//
// C++ class member Handling
//===----------------------------------------------------------------------===//

/// ActOnStartCXXClassDef - This is called at the start of a class/struct/union
/// definition, when on C++.
void Sema::ActOnStartCXXClassDef(Scope *S, DeclTy *D, SourceLocation LBrace) {
  CXXRecordDecl *Dcl = cast<CXXRecordDecl>(static_cast<Decl *>(D));
  PushDeclContext(Dcl);
  FieldCollector->StartClass();

  if (Dcl->getIdentifier()) {
    // C++ [class]p2: 
    //   [...] The class-name is also inserted into the scope of the
    //   class itself; this is known as the injected-class-name. For
    //   purposes of access checking, the injected-class-name is treated
    //   as if it were a public member name.
    TypedefDecl *InjectedClassName 
      = TypedefDecl::Create(Context, Dcl, LBrace, Dcl->getIdentifier(),
                            Context.getTypeDeclType(Dcl), /*PrevDecl=*/0);
    PushOnScopeChains(InjectedClassName, S);
  }
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

  bool isFunc = D.isFunctionDeclarator();
  if (!isFunc &&
      D.getDeclSpec().getTypeSpecType() == DeclSpec::TST_typedef &&
      D.getNumTypeObjects() == 0) {
    // Check also for this case:
    //
    // typedef int f();
    // f a;
    //
    Decl *TD = static_cast<Decl *>(DS.getTypeRep());
    isFunc = Context.getTypeDeclType(cast<TypeDecl>(TD))->isFunctionType();
  }

  bool isInstField = (DS.getStorageClassSpec() == DeclSpec::SCS_unspecified &&
                      !isFunc);

  Decl *Member;
  bool InvalidDecl = false;

  if (isInstField)
    Member = static_cast<Decl*>(ActOnField(S, Loc, D, BitWidth));
  else
    Member = static_cast<Decl*>(ActOnDeclarator(S, D, LastInGroup));

  if (!Member) return LastInGroup;

  assert((II || isInstField) && "No identifier for non-field ?");

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

    } else if (isInstField) {
      // C++ 9.6p3: A bit-field shall have integral or enumeration type.
      if (!cast<FieldDecl>(Member)->getType()->isIntegralType()) {
        Diag(Loc, diag::err_not_integral_type_bitfield,
             II->getName(), BitWidth->getSourceRange());
        InvalidDecl = true;
      }

    } else if (isa<FunctionDecl>(Member)) {
      // A function typedef ("typedef int f(); f a;").
      // C++ 9.6p3: A bit-field shall have integral or enumeration type.
      Diag(Loc, diag::err_not_integral_type_bitfield,
           II->getName(), BitWidth->getSourceRange());
      InvalidDecl = true;

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
    if (CXXClassVarDecl *CVD = dyn_cast<CXXClassVarDecl>(Member)) {
      // ...static member of...
      CVD->setInit(Init);
      // ...const integral or const enumeration type.
      if (Context.getCanonicalType(CVD->getType()).isConstQualified() &&
          CVD->getType()->isIntegralType()) {
        // constant-initializer
        if (CheckForConstantInitializer(Init, CVD->getType()))
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
              FieldCollector->getCurNumFields(), LBrac, RBrac, 0);
}

/// AddImplicitlyDeclaredMembersToClass - Adds any implicitly-declared
/// special functions, such as the default constructor, copy
/// constructor, or destructor, to the given C++ class (C++
/// [special]p1).  This routine can only be executed just before the
/// definition of the class is complete.
void Sema::AddImplicitlyDeclaredMembersToClass(CXXRecordDecl *ClassDecl) {
  if (!ClassDecl->hasUserDeclaredConstructor()) {
    // C++ [class.ctor]p5:
    //   A default constructor for a class X is a constructor of class X
    //   that can be called without an argument. If there is no
    //   user-declared constructor for class X, a default constructor is
    //   implicitly declared. An implicitly-declared default constructor
    //   is an inline public member of its class.
    CXXConstructorDecl *DefaultCon = 
      CXXConstructorDecl::Create(Context, ClassDecl,
                                 ClassDecl->getLocation(),
                                 ClassDecl->getIdentifier(),
                                 Context.getFunctionType(Context.VoidTy,
                                                         0, 0, false, 0),
                                 /*isExplicit=*/false,
                                 /*isInline=*/true,
                                 /*isImplicitlyDeclared=*/true);
    DefaultCon->setAccess(AS_public);
    ClassDecl->addConstructor(Context, DefaultCon);
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
    for (CXXRecordDecl::field_iterator Field = ClassDecl->field_begin();
         HasConstCopyConstructor && Field != ClassDecl->field_end(); ++Field) {
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

    //  Otherwise, the implicitly declared copy constructor will have
    //  the form
    //
    //       X::X(X&)
    QualType ArgType = Context.getTypeDeclType(ClassDecl);
    if (HasConstCopyConstructor)
      ArgType = ArgType.withConst();
    ArgType = Context.getReferenceType(ArgType);

    //  An implicitly-declared copy constructor is an inline public
    //  member of its class.
    CXXConstructorDecl *CopyConstructor
      = CXXConstructorDecl::Create(Context, ClassDecl,
                                   ClassDecl->getLocation(),
                                   ClassDecl->getIdentifier(),
                                   Context.getFunctionType(Context.VoidTy,
                                                           &ArgType, 1,
                                                           false, 0),
                                   /*isExplicit=*/false,
                                   /*isInline=*/true,
                                   /*isImplicitlyDeclared=*/true);
    CopyConstructor->setAccess(AS_public);

    // Add the parameter to the constructor.
    ParmVarDecl *FromParam = ParmVarDecl::Create(Context, CopyConstructor,
                                                 ClassDecl->getLocation(),
                                                 /*IdentifierInfo=*/0,
                                                 ArgType, VarDecl::None, 0, 0);
    CopyConstructor->setParams(&FromParam, 1);

    ClassDecl->addConstructor(Context, CopyConstructor);
  }

  // FIXME: Implicit destructor
  // FIXME: Implicit copy assignment operator
}

void Sema::ActOnFinishCXXClassDef(DeclTy *D) {
  CXXRecordDecl *Rec = cast<CXXRecordDecl>(static_cast<Decl *>(D));
  FieldCollector->FinishClass();
  AddImplicitlyDeclaredMembersToClass(Rec);
  PopDeclContext();

  // Everything, including inline method definitions, have been parsed.
  // Let the consumer know of the new TagDecl definition.
  Consumer.HandleTagDeclDefinition(Rec);
}

/// ActOnConstructorDeclarator - Called by ActOnDeclarator to complete
/// the declaration of the given C++ constructor ConDecl that was
/// built from declarator D. This routine is responsible for checking
/// that the newly-created constructor declaration is well-formed and
/// for recording it in the C++ class. Example:
///
/// @code 
/// class X {
///   X(); // X::X() will be the ConDecl.
/// };
/// @endcode
Sema::DeclTy *Sema::ActOnConstructorDeclarator(CXXConstructorDecl *ConDecl) {
  assert(ConDecl && "Expected to receive a constructor declaration");

  // Check default arguments on the constructor
  CheckCXXDefaultArguments(ConDecl);

  CXXRecordDecl *ClassDecl = dyn_cast_or_null<CXXRecordDecl>(CurContext);
  if (!ClassDecl) {
    ConDecl->setInvalidDecl();
    return ConDecl;
  }

  // Make sure this constructor is an overload of the existing
  // constructors.
  OverloadedFunctionDecl::function_iterator MatchedDecl;
  if (!IsOverload(ConDecl, ClassDecl->getConstructors(), MatchedDecl)) {
    Diag(ConDecl->getLocation(),
         diag::err_constructor_redeclared,
         SourceRange(ConDecl->getLocation()));
    Diag((*MatchedDecl)->getLocation(),
         diag::err_previous_declaration,
         SourceRange((*MatchedDecl)->getLocation()));
    ConDecl->setInvalidDecl();
    return ConDecl;
  }


  // C++ [class.copy]p3:
  //   A declaration of a constructor for a class X is ill-formed if
  //   its first parameter is of type (optionally cv-qualified) X and
  //   either there are no other parameters or else all other
  //   parameters have default arguments.
  if ((ConDecl->getNumParams() == 1) || 
      (ConDecl->getNumParams() > 1 && 
       ConDecl->getParamDecl(1)->getDefaultArg() != 0)) {
    QualType ParamType = ConDecl->getParamDecl(0)->getType();
    QualType ClassTy = Context.getTagDeclType(
                         const_cast<CXXRecordDecl*>(ConDecl->getParent()));
    if (Context.getCanonicalType(ParamType).getUnqualifiedType() == ClassTy) {
      Diag(ConDecl->getLocation(),
           diag::err_constructor_byvalue_arg,
           SourceRange(ConDecl->getParamDecl(0)->getLocation()));
      ConDecl->setInvalidDecl();
      return 0;
    }
  }
      
  // Add this constructor to the set of constructors of the current
  // class.
  ClassDecl->addConstructor(Context, ConDecl);

  return (DeclTy *)ConDecl;
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

    if (PrevDecl && isDeclInScope(PrevDecl, CurContext, DeclRegionScope)) {
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


/// AddCXXDirectInitializerToDecl - This action is called immediately after 
/// ActOnDeclarator, when a C++ direct initializer is present.
/// e.g: "int x(1);"
void Sema::AddCXXDirectInitializerToDecl(DeclTy *Dcl, SourceLocation LParenLoc,
                                         ExprTy **ExprTys, unsigned NumExprs,
                                         SourceLocation *CommaLocs,
                                         SourceLocation RParenLoc) {
  assert(NumExprs != 0 && ExprTys && "missing expressions");
  Decl *RealDecl = static_cast<Decl *>(Dcl);

  // If there is no declaration, there was an error parsing it.  Just ignore
  // the initializer.
  if (RealDecl == 0) {
    for (unsigned i = 0; i != NumExprs; ++i)
      delete static_cast<Expr *>(ExprTys[i]);
    return;
  }
  
  VarDecl *VDecl = dyn_cast<VarDecl>(RealDecl);
  if (!VDecl) {
    Diag(RealDecl->getLocation(), diag::err_illegal_initializer);
    RealDecl->setInvalidDecl();
    return;
  }

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

  if (VDecl->getType()->isRecordType()) {
    // FIXME: When constructors for class types are supported, determine how 
    // exactly semantic checking will be done for direct initializers.
    unsigned DiagID = PP.getDiagnostics().getCustomDiagID(Diagnostic::Error,
                           "initialization for class types is not handled yet");
    Diag(VDecl->getLocation(), DiagID);
    RealDecl->setInvalidDecl();
    return;
  }

  if (NumExprs > 1) {
    Diag(CommaLocs[0], diag::err_builtin_direct_init_more_than_one_arg,
         SourceRange(VDecl->getLocation(), RParenLoc));
    RealDecl->setInvalidDecl();
    return;
  }

  // Let clients know that initialization was done with a direct initializer.
  VDecl->setCXXDirectInitializer(true);

  assert(NumExprs == 1 && "Expected 1 expression");
  // Set the init expression, handles conversions.
  AddInitializerToDecl(Dcl, ExprTys[0]);
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
  assert(!T1->isReferenceType() && "T1 must be the pointee type of the reference type");
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
bool 
Sema::CheckReferenceInit(Expr *&Init, QualType &DeclType, 
                         ImplicitConversionSequence *ICS) {
  assert(DeclType->isReferenceType() && "Reference init needs a reference");

  QualType T1 = DeclType->getAsReferenceType()->getPointeeType();
  QualType T2 = Init->getType();

  // Compute some basic properties of the types and the initializer.
  bool DerivedToBase = false;
  Expr::isLvalueResult InitLvalue = Init->isLvalue(Context);
  ReferenceCompareResult RefRelationship 
    = CompareReferenceRelationship(T1, T2, DerivedToBase);

  // Most paths end in a failed conversion.
  if (ICS)
    ICS->ConversionKind = ImplicitConversionSequence::BadConversion;

  // C++ [dcl.init.ref]p5:
  //   A reference to type “cv1 T1” is initialized by an expression
  //   of type “cv2 T2” as follows:

  //     -- If the initializer expression

  bool BindsDirectly = false;
  //       -- is an lvalue (but is not a bit-field), and “cv1 T1” is
  //          reference-compatible with “cv2 T2,” or
  //
  // Note that the bit-field check is skipped if we are just computing
  // the implicit conversion sequence (C++ [over.best.ics]p2).
  if (InitLvalue == Expr::LV_Valid && (ICS || !Init->isBitField()) &&
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

      // Nothing more to do: the inaccessibility/ambiguity check for
      // derived-to-base conversions is suppressed when we're
      // computing the implicit conversion sequence (C++
      // [over.best.ics]p2).
      return false;
    } else {
      // Perform the conversion.
      // FIXME: Binding to a subobject of the lvalue is going to require
      // more AST annotation than this.
      ImpCastExprToType(Init, T1);    
    }
  }

  //       -- has a class type (i.e., T2 is a class type) and can be
  //          implicitly converted to an lvalue of type “cv3 T3,”
  //          where “cv1 T1” is reference-compatible with “cv3 T3”
  //          92) (this conversion is selected by enumerating the
  //          applicable conversion functions (13.3.1.6) and choosing
  //          the best one through overload resolution (13.3)),
  // FIXME: Implement this second bullet, once we have conversion
  //        functions. Also remember C++ [over.ics.ref]p1, second part.

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
  //        type (i.e., cv1 shall be const).
  if (T1.getCVRQualifiers() != QualType::Const) {
    if (!ICS)
      Diag(Init->getSourceRange().getBegin(),
           diag::err_not_reference_to_const_init,
           T1.getAsString(), 
           InitLvalue != Expr::LV_Valid? "temporary" : "value",
           T2.getAsString(), Init->getSourceRange());
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
  //
  //          The constructor that would be used to make the copy
  //          shall be callable whether or not the copy is actually
  //          done.
  //
  // Note that C++0x [dcl.ref.init]p5 takes away this implementation
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
    } else {
      // FIXME: Binding to a subobject of the rvalue is going to require
      // more AST annotation than this.
      ImpCastExprToType(Init, T1);
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
           diag::err_reference_init_drops_quals,
           T1.getAsString(), 
           InitLvalue != Expr::LV_Valid? "temporary" : "value",
           T2.getAsString(), Init->getSourceRange());
    return true;
  }

  // Actually try to convert the initializer to T1.
  if (ICS) {
    /// C++ [over.ics.ref]p2:
    /// 
    ///   When a parameter of reference type is not bound directly to
    ///   an argument expression, the conversion sequence is the one
    ///   required to convert the argument expression to the
    ///   underlying type of the reference according to
    ///   13.3.3.1. Conceptually, this conversion sequence corresponds
    ///   to copy-initializing a temporary of the underlying type with
    ///   the argument expression. Any difference in top-level
    ///   cv-qualification is subsumed by the initialization itself
    ///   and does not constitute a conversion.
    *ICS = TryImplicitConversion(Init, T1);
    return ICS->ConversionKind == ImplicitConversionSequence::BadConversion;
  } else {
    return PerformImplicitConversion(Init, T1);
  }
}
