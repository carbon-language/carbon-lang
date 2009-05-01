//===--- SemaTemplateInstantiateDecl.cpp - C++ Template Decl Instantiation ===/
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//===----------------------------------------------------------------------===/
//
//  This file implements C++ template instantiation for declarations.
//
//===----------------------------------------------------------------------===/
#include "Sema.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/Expr.h"
#include "llvm/Support/Compiler.h"

using namespace clang;

namespace {
  class VISIBILITY_HIDDEN TemplateDeclInstantiator 
    : public DeclVisitor<TemplateDeclInstantiator, Decl *> {
    Sema &SemaRef;
    DeclContext *Owner;
    const TemplateArgument *TemplateArgs;
    unsigned NumTemplateArgs;
    
  public:
    typedef Sema::OwningExprResult OwningExprResult;

    TemplateDeclInstantiator(Sema &SemaRef, DeclContext *Owner,
                             const TemplateArgument *TemplateArgs,
                             unsigned NumTemplateArgs)
      : SemaRef(SemaRef), Owner(Owner), TemplateArgs(TemplateArgs), 
        NumTemplateArgs(NumTemplateArgs) { }
    
    // FIXME: Once we get closer to completion, replace these
    // manually-written declarations with automatically-generated ones
    // from clang/AST/DeclNodes.def.
    Decl *VisitTranslationUnitDecl(TranslationUnitDecl *D);
    Decl *VisitNamespaceDecl(NamespaceDecl *D);
    Decl *VisitTypedefDecl(TypedefDecl *D);
    Decl *VisitVarDecl(VarDecl *D);
    Decl *VisitFieldDecl(FieldDecl *D);
    Decl *VisitStaticAssertDecl(StaticAssertDecl *D);
    Decl *VisitEnumDecl(EnumDecl *D);
    Decl *VisitEnumConstantDecl(EnumConstantDecl *D);
    Decl *VisitCXXRecordDecl(CXXRecordDecl *D);
    Decl *VisitCXXMethodDecl(CXXMethodDecl *D);
    Decl *VisitCXXConstructorDecl(CXXConstructorDecl *D);
    Decl *VisitCXXDestructorDecl(CXXDestructorDecl *D);
    Decl *VisitCXXConversionDecl(CXXConversionDecl *D);
    ParmVarDecl *VisitParmVarDecl(ParmVarDecl *D);
    Decl *VisitOriginalParmVarDecl(OriginalParmVarDecl *D);

    // Base case. FIXME: Remove once we can instantiate everything.
    Decl *VisitDecl(Decl *) { 
      assert(false && "Template instantiation of unknown declaration kind!");
      return 0;
    }

    // Helper functions for instantiating methods.
    QualType InstantiateFunctionType(FunctionDecl *D,
                             llvm::SmallVectorImpl<ParmVarDecl *> &Params);
    bool InitMethodInstantiation(CXXMethodDecl *New, CXXMethodDecl *Tmpl);
  };
}

Decl *
TemplateDeclInstantiator::VisitTranslationUnitDecl(TranslationUnitDecl *D) {
  assert(false && "Translation units cannot be instantiated");
  return D;
}

Decl *
TemplateDeclInstantiator::VisitNamespaceDecl(NamespaceDecl *D) {
  assert(false && "Namespaces cannot be instantiated");
  return D;
}

Decl *TemplateDeclInstantiator::VisitTypedefDecl(TypedefDecl *D) {
  bool Invalid = false;
  QualType T = D->getUnderlyingType();
  if (T->isDependentType()) {
    T = SemaRef.InstantiateType(T, TemplateArgs, NumTemplateArgs,
                                D->getLocation(),
                                D->getDeclName());
    if (T.isNull()) {
      Invalid = true;
      T = SemaRef.Context.IntTy;
    }
  }
       
  // Create the new typedef
  TypedefDecl *Typedef
    = TypedefDecl::Create(SemaRef.Context, Owner, D->getLocation(),
                          D->getIdentifier(), T);
  if (Invalid)
    Typedef->setInvalidDecl();

  Owner->addDecl(SemaRef.Context, Typedef);
  return Typedef;
}

Decl *TemplateDeclInstantiator::VisitVarDecl(VarDecl *D) {
  // Instantiate the type of the declaration
  QualType T = SemaRef.InstantiateType(D->getType(), TemplateArgs,
                                       NumTemplateArgs, 
                                       D->getTypeSpecStartLoc(),
                                       D->getDeclName());
  if (T.isNull())
    return 0;

  // Build the instantiataed declaration
  VarDecl *Var = VarDecl::Create(SemaRef.Context, Owner,
                                 D->getLocation(), D->getIdentifier(),
                                 T, D->getStorageClass(),
                                 D->getTypeSpecStartLoc());
  Var->setThreadSpecified(D->isThreadSpecified());
  Var->setCXXDirectInitializer(D->hasCXXDirectInitializer());
  Var->setDeclaredInCondition(D->isDeclaredInCondition());
 
  // FIXME: In theory, we could have a previous declaration for
  // variables that are not static data members.
  bool Redeclaration = false;
  SemaRef.CheckVariableDeclaration(Var, 0, Redeclaration);
  Owner->addDecl(SemaRef.Context, Var);

  if (D->getInit()) {
    OwningExprResult Init 
      = SemaRef.InstantiateExpr(D->getInit(), TemplateArgs, NumTemplateArgs);
    if (Init.isInvalid())
      Var->setInvalidDecl();
    else
      SemaRef.AddInitializerToDecl(Sema::DeclPtrTy::make(Var), move(Init),
                                   D->hasCXXDirectInitializer());
  }

  return Var;
}

Decl *TemplateDeclInstantiator::VisitFieldDecl(FieldDecl *D) {
  bool Invalid = false;
  QualType T = D->getType();
  if (T->isDependentType())  {
    T = SemaRef.InstantiateType(T, TemplateArgs, NumTemplateArgs,
                                D->getLocation(),
                                D->getDeclName());
    if (!T.isNull() && T->isFunctionType()) {
      // C++ [temp.arg.type]p3:
      //   If a declaration acquires a function type through a type
      //   dependent on a template-parameter and this causes a
      //   declaration that does not use the syntactic form of a
      //   function declarator to have function type, the program is
      //   ill-formed.
      SemaRef.Diag(D->getLocation(), diag::err_field_instantiates_to_function)
        << T;
      T = QualType();
      Invalid = true;
    }
  }

  Expr *BitWidth = D->getBitWidth();
  if (Invalid)
    BitWidth = 0;
  else if (BitWidth) {
    OwningExprResult InstantiatedBitWidth
      = SemaRef.InstantiateExpr(BitWidth, TemplateArgs, NumTemplateArgs);
    if (InstantiatedBitWidth.isInvalid()) {
      Invalid = true;
      BitWidth = 0;
    } else
      BitWidth = InstantiatedBitWidth.takeAs<Expr>();
  }

  FieldDecl *Field = SemaRef.CheckFieldDecl(D->getDeclName(), T,
                                            cast<RecordDecl>(Owner), 
                                            D->getLocation(),
                                            D->isMutable(),
                                            BitWidth,
                                            D->getAccess(),
                                            0);
  if (Field) {
    if (Invalid)
      Field->setInvalidDecl();
    
    Owner->addDecl(SemaRef.Context, Field);
  }

  return Field;
}

Decl *TemplateDeclInstantiator::VisitStaticAssertDecl(StaticAssertDecl *D) {
  Expr *AssertExpr = D->getAssertExpr();
      
  OwningExprResult InstantiatedAssertExpr
    = SemaRef.InstantiateExpr(AssertExpr, TemplateArgs, NumTemplateArgs);
  if (InstantiatedAssertExpr.isInvalid())
    return 0;

  OwningExprResult Message = SemaRef.Clone(D->getMessage());
  Decl *StaticAssert 
    = SemaRef.ActOnStaticAssertDeclaration(D->getLocation(), 
                                           move(InstantiatedAssertExpr),
                                           move(Message)).getAs<Decl>();
  return StaticAssert;
}

Decl *TemplateDeclInstantiator::VisitEnumDecl(EnumDecl *D) {
  EnumDecl *Enum = EnumDecl::Create(SemaRef.Context, Owner, 
                                    D->getLocation(), D->getIdentifier(),
                                    /*PrevDecl=*/0);
  Enum->setAccess(D->getAccess());
  Owner->addDecl(SemaRef.Context, Enum);
  Enum->startDefinition();

  llvm::SmallVector<Sema::DeclPtrTy, 16> Enumerators;

  EnumConstantDecl *LastEnumConst = 0;
  for (EnumDecl::enumerator_iterator EC = D->enumerator_begin(SemaRef.Context),
         ECEnd = D->enumerator_end(SemaRef.Context);
       EC != ECEnd; ++EC) {
    // The specified value for the enumerator.
    OwningExprResult Value = SemaRef.Owned((Expr *)0);
    if (Expr *UninstValue = EC->getInitExpr())
      Value = SemaRef.InstantiateExpr(UninstValue, 
                                      TemplateArgs, NumTemplateArgs);

    // Drop the initial value and continue.
    bool isInvalid = false;
    if (Value.isInvalid()) {
      Value = SemaRef.Owned((Expr *)0);
      isInvalid = true;
    }

    EnumConstantDecl *EnumConst 
      = SemaRef.CheckEnumConstant(Enum, LastEnumConst,
                                  EC->getLocation(), EC->getIdentifier(),
                                  move(Value));

    if (isInvalid) {
      if (EnumConst)
        EnumConst->setInvalidDecl();
      Enum->setInvalidDecl();
    }

    if (EnumConst) {
      Enum->addDecl(SemaRef.Context, EnumConst);
      Enumerators.push_back(Sema::DeclPtrTy::make(EnumConst));
      LastEnumConst = EnumConst;
    }
  }
      
  SemaRef.ActOnEnumBody(Enum->getLocation(), Sema::DeclPtrTy::make(Enum),
                        &Enumerators[0], Enumerators.size());

  return Enum;
}

Decl *TemplateDeclInstantiator::VisitEnumConstantDecl(EnumConstantDecl *D) {
  assert(false && "EnumConstantDecls can only occur within EnumDecls.");
  return 0;
}

Decl *TemplateDeclInstantiator::VisitCXXRecordDecl(CXXRecordDecl *D) {
  CXXRecordDecl *PrevDecl = 0;
  if (D->isInjectedClassName())
    PrevDecl = cast<CXXRecordDecl>(Owner);

  CXXRecordDecl *Record
    = CXXRecordDecl::Create(SemaRef.Context, D->getTagKind(), Owner, 
                            D->getLocation(), D->getIdentifier(), PrevDecl);
  Record->setImplicit(D->isImplicit());
  Record->setAccess(D->getAccess());

  if (!D->isInjectedClassName())
    Record->setInstantiationOfMemberClass(D);
  else
    Record->setDescribedClassTemplate(D->getDescribedClassTemplate());

  Owner->addDecl(SemaRef.Context, Record);
  return Record;
}

Decl *TemplateDeclInstantiator::VisitCXXMethodDecl(CXXMethodDecl *D) {
  // Only handle actual methods; we'll deal with constructors,
  // destructors, etc. separately.
  if (D->getKind() != Decl::CXXMethod)
    return 0;

  llvm::SmallVector<ParmVarDecl *, 16> Params;
  QualType T = InstantiateFunctionType(D, Params);
  if (T.isNull())
    return 0;

  // Build the instantiated method declaration.
  CXXRecordDecl *Record = cast<CXXRecordDecl>(Owner);
  CXXMethodDecl *Method
    = CXXMethodDecl::Create(SemaRef.Context, Record, D->getLocation(), 
                            D->getDeclName(), T, D->isStatic(), 
                            D->isInline());

  // Attach the parameters
  for (unsigned P = 0; P < Params.size(); ++P)
    Params[P]->setOwningFunction(Method);
  Method->setParams(SemaRef.Context, &Params[0], Params.size());

  if (InitMethodInstantiation(Method, D))
    Method->setInvalidDecl();

  NamedDecl *PrevDecl 
    = SemaRef.LookupQualifiedName(Owner, Method->getDeclName(), 
                                  Sema::LookupOrdinaryName, true);
  // In C++, the previous declaration we find might be a tag type
  // (class or enum). In this case, the new declaration will hide the
  // tag type. Note that this does does not apply if we're declaring a
  // typedef (C++ [dcl.typedef]p4).
  if (PrevDecl && PrevDecl->getIdentifierNamespace() == Decl::IDNS_Tag)
    PrevDecl = 0;
  bool Redeclaration = false;
  bool OverloadableAttrRequired = false;
  SemaRef.CheckFunctionDeclaration(Method, PrevDecl, Redeclaration,
                                   /*FIXME:*/OverloadableAttrRequired);

  if (!Method->isInvalidDecl() || !PrevDecl)
    Owner->addDecl(SemaRef.Context, Method);
  return Method;
}

Decl *TemplateDeclInstantiator::VisitCXXConstructorDecl(CXXConstructorDecl *D) {
  llvm::SmallVector<ParmVarDecl *, 16> Params;
  QualType T = InstantiateFunctionType(D, Params);
  if (T.isNull())
    return 0;

  // Build the instantiated method declaration.
  CXXRecordDecl *Record = cast<CXXRecordDecl>(Owner);
  QualType ClassTy = SemaRef.Context.getTypeDeclType(Record);
  DeclarationName Name
    = SemaRef.Context.DeclarationNames.getCXXConstructorName(ClassTy);
  CXXConstructorDecl *Constructor
    = CXXConstructorDecl::Create(SemaRef.Context, Record, D->getLocation(), 
                                 Name, T, D->isExplicit(), D->isInline(), 
                                 false);

  // Attach the parameters
  for (unsigned P = 0; P < Params.size(); ++P)
    Params[P]->setOwningFunction(Constructor);
  Constructor->setParams(SemaRef.Context, &Params[0], Params.size());

  if (InitMethodInstantiation(Constructor, D))
    Constructor->setInvalidDecl();

  NamedDecl *PrevDecl 
    = SemaRef.LookupQualifiedName(Owner, Name, Sema::LookupOrdinaryName, true);

  // In C++, the previous declaration we find might be a tag type
  // (class or enum). In this case, the new declaration will hide the
  // tag type. Note that this does does not apply if we're declaring a
  // typedef (C++ [dcl.typedef]p4).
  if (PrevDecl && PrevDecl->getIdentifierNamespace() == Decl::IDNS_Tag)
    PrevDecl = 0;
  bool Redeclaration = false;
  bool OverloadableAttrRequired = false;
  SemaRef.CheckFunctionDeclaration(Constructor, PrevDecl, Redeclaration,
                                   /*FIXME:*/OverloadableAttrRequired);

  Owner->addDecl(SemaRef.Context, Constructor);
  return Constructor;
}

Decl *TemplateDeclInstantiator::VisitCXXDestructorDecl(CXXDestructorDecl *D) {
  llvm::SmallVector<ParmVarDecl *, 16> Params;
  QualType T = InstantiateFunctionType(D, Params);
  if (T.isNull())
    return 0;
  assert(Params.size() == 0 && "Destructor with parameters?");

  // Build the instantiated destructor declaration.
  CXXRecordDecl *Record = cast<CXXRecordDecl>(Owner);
  QualType ClassTy = SemaRef.Context.getTypeDeclType(Record);
  CXXDestructorDecl *Destructor
    = CXXDestructorDecl::Create(SemaRef.Context, Record,
                                D->getLocation(),
             SemaRef.Context.DeclarationNames.getCXXDestructorName(ClassTy),
                                T, D->isInline(), false);
  if (InitMethodInstantiation(Destructor, D))
    Destructor->setInvalidDecl();

  bool Redeclaration = false;
  bool OverloadableAttrRequired = false;
  NamedDecl *PrevDecl = 0;
  SemaRef.CheckFunctionDeclaration(Destructor, PrevDecl, Redeclaration,
                                   /*FIXME:*/OverloadableAttrRequired);
  Owner->addDecl(SemaRef.Context, Destructor);
  return Destructor;
}

Decl *TemplateDeclInstantiator::VisitCXXConversionDecl(CXXConversionDecl *D) {
  llvm::SmallVector<ParmVarDecl *, 16> Params;
  QualType T = InstantiateFunctionType(D, Params);
  if (T.isNull())
    return 0;
  assert(Params.size() == 0 && "Destructor with parameters?");

  // Build the instantiated conversion declaration.
  CXXRecordDecl *Record = cast<CXXRecordDecl>(Owner);
  QualType ClassTy = SemaRef.Context.getTypeDeclType(Record);
  QualType ConvTy 
    = SemaRef.Context.getCanonicalType(T->getAsFunctionType()->getResultType());
  CXXConversionDecl *Conversion
    = CXXConversionDecl::Create(SemaRef.Context, Record,
                                D->getLocation(),
         SemaRef.Context.DeclarationNames.getCXXConversionFunctionName(ConvTy),
                                T, D->isInline(), D->isExplicit());
  if (InitMethodInstantiation(Conversion, D))
    Conversion->setInvalidDecl();

  bool Redeclaration = false;
  bool OverloadableAttrRequired = false;
  NamedDecl *PrevDecl = 0;
  SemaRef.CheckFunctionDeclaration(Conversion, PrevDecl, Redeclaration,
                                   /*FIXME:*/OverloadableAttrRequired);
  Owner->addDecl(SemaRef.Context, Conversion);
  return Conversion;  
}

ParmVarDecl *TemplateDeclInstantiator::VisitParmVarDecl(ParmVarDecl *D) {
  QualType OrigT = SemaRef.InstantiateType(D->getOriginalType(), TemplateArgs,
                                           NumTemplateArgs, D->getLocation(),
                                           D->getDeclName());
  if (OrigT.isNull())
    return 0;

  QualType T = SemaRef.adjustParameterType(OrigT);

  if (D->getDefaultArg()) {
    // FIXME: Leave a marker for "uninstantiated" default
    // arguments. They only get instantiated on demand at the call
    // site.
    unsigned DiagID = SemaRef.Diags.getCustomDiagID(Diagnostic::Warning,
        "sorry, dropping default argument during template instantiation");
    SemaRef.Diag(D->getDefaultArg()->getSourceRange().getBegin(), DiagID)
      << D->getDefaultArg()->getSourceRange();
  }

  // Allocate the parameter
  ParmVarDecl *Param = 0;
  if (T == OrigT)
    Param = ParmVarDecl::Create(SemaRef.Context, Owner, D->getLocation(),
                                D->getIdentifier(), T, D->getStorageClass(), 
                                0);
  else
    Param = OriginalParmVarDecl::Create(SemaRef.Context, Owner, 
                                        D->getLocation(), D->getIdentifier(),
                                        T, OrigT, D->getStorageClass(), 0);

  // Note: we don't try to instantiate function parameters until after
  // we've instantiated the function's type. Therefore, we don't have
  // to check for 'void' parameter types here.
  return Param;
}

Decl *
TemplateDeclInstantiator::VisitOriginalParmVarDecl(OriginalParmVarDecl *D) {
  // Since parameter types can decay either before or after
  // instantiation, we simply treat OriginalParmVarDecls as
  // ParmVarDecls the same way, and create one or the other depending
  // on what happens after template instantiation.
  return VisitParmVarDecl(D);
}

Decl *Sema::InstantiateDecl(Decl *D, DeclContext *Owner,
                            const TemplateArgument *TemplateArgs,
                            unsigned NumTemplateArgs) {
  TemplateDeclInstantiator Instantiator(*this, Owner, TemplateArgs,
                                        NumTemplateArgs);
  return Instantiator.Visit(D);
}

/// \brief Instantiates the type of the given function, including
/// instantiating all of the function parameters.
///
/// \param D The function that we will be instantiated
///
/// \param Params the instantiated parameter declarations

/// \returns the instantiated function's type if successfull, a NULL
/// type if there was an error.
QualType 
TemplateDeclInstantiator::InstantiateFunctionType(FunctionDecl *D,
                              llvm::SmallVectorImpl<ParmVarDecl *> &Params) {
  bool InvalidDecl = false;

  // Instantiate the function parameters
  TemplateDeclInstantiator ParamInstantiator(SemaRef, 0,
                                             TemplateArgs, NumTemplateArgs);
  llvm::SmallVector<QualType, 16> ParamTys;
  for (FunctionDecl::param_iterator P = D->param_begin(), 
                                 PEnd = D->param_end();
       P != PEnd; ++P) {
    if (ParmVarDecl *PInst = ParamInstantiator.VisitParmVarDecl(*P)) {
      if (PInst->getType()->isVoidType()) {
        SemaRef.Diag(PInst->getLocation(), diag::err_param_with_void_type);
        PInst->setInvalidDecl();
      }
      else if (SemaRef.RequireNonAbstractType(PInst->getLocation(), 
                                              PInst->getType(),
                                              diag::err_abstract_type_in_decl,
                                              Sema::AbstractParamType))
        PInst->setInvalidDecl();

      Params.push_back(PInst);
      ParamTys.push_back(PInst->getType());

      if (PInst->isInvalidDecl())
        InvalidDecl = true;
    } else 
      InvalidDecl = true;
  }

  // FIXME: Deallocate dead declarations.
  if (InvalidDecl)
    return QualType();

  const FunctionProtoType *Proto = D->getType()->getAsFunctionProtoType();
  assert(Proto && "Missing prototype?");
  QualType ResultType 
    = SemaRef.InstantiateType(Proto->getResultType(),
                              TemplateArgs, NumTemplateArgs,
                              D->getLocation(), D->getDeclName());
  if (ResultType.isNull())
    return QualType();

  return SemaRef.BuildFunctionType(ResultType, &ParamTys[0], ParamTys.size(),
                                   Proto->isVariadic(), Proto->getTypeQuals(),
                                   D->getLocation(), D->getDeclName());
}

/// \brief Initializes common fields of an instantiated method
/// declaration (New) from the corresponding fields of its template
/// (Tmpl).
///
/// \returns true if there was an error
bool 
TemplateDeclInstantiator::InitMethodInstantiation(CXXMethodDecl *New, 
                                                  CXXMethodDecl *Tmpl) {
  CXXRecordDecl *Record = cast<CXXRecordDecl>(Owner);
  New->setAccess(Tmpl->getAccess());
  if (Tmpl->isVirtual()) {
    New->setVirtual();
    Record->setAggregate(false);
    Record->setPOD(false);
    Record->setPolymorphic(true);
  }
  if (Tmpl->isDeleted())
    New->setDeleted();
  if (Tmpl->isPure()) {
    New->setPure();
    Record->setAbstract(true);
  }

  // FIXME: attributes
  // FIXME: New needs a pointer to Tmpl
  return false;
}
