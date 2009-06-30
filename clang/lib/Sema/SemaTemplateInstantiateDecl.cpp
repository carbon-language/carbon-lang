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
#include "clang/AST/ASTConsumer.h"
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
    const TemplateArgumentList &TemplateArgs;
    
  public:
    typedef Sema::OwningExprResult OwningExprResult;

    TemplateDeclInstantiator(Sema &SemaRef, DeclContext *Owner,
                             const TemplateArgumentList &TemplateArgs)
      : SemaRef(SemaRef), Owner(Owner), TemplateArgs(TemplateArgs) { }
    
    // FIXME: Once we get closer to completion, replace these manually-written
    // declarations with automatically-generated ones from
    // clang/AST/DeclNodes.def.
    Decl *VisitTranslationUnitDecl(TranslationUnitDecl *D);
    Decl *VisitNamespaceDecl(NamespaceDecl *D);
    Decl *VisitTypedefDecl(TypedefDecl *D);
    Decl *VisitVarDecl(VarDecl *D);
    Decl *VisitFieldDecl(FieldDecl *D);
    Decl *VisitStaticAssertDecl(StaticAssertDecl *D);
    Decl *VisitEnumDecl(EnumDecl *D);
    Decl *VisitEnumConstantDecl(EnumConstantDecl *D);
    Decl *VisitFunctionDecl(FunctionDecl *D);
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
    bool InitFunctionInstantiation(FunctionDecl *New, FunctionDecl *Tmpl);
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
    T = SemaRef.InstantiateType(T, TemplateArgs, 
                                D->getLocation(), D->getDeclName());
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
                                       D->getTypeSpecStartLoc(),
                                       D->getDeclName());
  if (T.isNull())
    return 0;

  // Build the instantiated declaration
  VarDecl *Var = VarDecl::Create(SemaRef.Context, Owner,
                                 D->getLocation(), D->getIdentifier(),
                                 T, D->getStorageClass(),
                                 D->getTypeSpecStartLoc());
  Var->setThreadSpecified(D->isThreadSpecified());
  Var->setCXXDirectInitializer(D->hasCXXDirectInitializer());
  Var->setDeclaredInCondition(D->isDeclaredInCondition());
 
  // FIXME: In theory, we could have a previous declaration for variables that
  // are not static data members.
  bool Redeclaration = false;
  SemaRef.CheckVariableDeclaration(Var, 0, Redeclaration);
  Owner->addDecl(SemaRef.Context, Var);

  if (D->getInit()) {
    OwningExprResult Init 
      = SemaRef.InstantiateExpr(D->getInit(), TemplateArgs);
    if (Init.isInvalid())
      Var->setInvalidDecl();
    else
      SemaRef.AddInitializerToDecl(Sema::DeclPtrTy::make(Var), move(Init),
                                   D->hasCXXDirectInitializer());
  } else {
    // FIXME: Call ActOnUninitializedDecl? (Not always)
  }

  return Var;
}

Decl *TemplateDeclInstantiator::VisitFieldDecl(FieldDecl *D) {
  bool Invalid = false;
  QualType T = D->getType();
  if (T->isDependentType())  {
    T = SemaRef.InstantiateType(T, TemplateArgs,
                                D->getLocation(), D->getDeclName());
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
    // The bit-width expression is not potentially evaluated.
    EnterExpressionEvaluationContext Unevaluated(SemaRef, Action::Unevaluated);
    
    OwningExprResult InstantiatedBitWidth
      = SemaRef.InstantiateExpr(BitWidth, TemplateArgs);
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
      
  // The expression in a static assertion is not potentially evaluated.
  EnterExpressionEvaluationContext Unevaluated(SemaRef, Action::Unevaluated);
  
  OwningExprResult InstantiatedAssertExpr
    = SemaRef.InstantiateExpr(AssertExpr, TemplateArgs);
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
  Enum->setInstantiationOfMemberEnum(D);
  Enum->setAccess(D->getAccess());
  Owner->addDecl(SemaRef.Context, Enum);
  Enum->startDefinition();

  llvm::SmallVector<Sema::DeclPtrTy, 4> Enumerators;

  EnumConstantDecl *LastEnumConst = 0;
  for (EnumDecl::enumerator_iterator EC = D->enumerator_begin(SemaRef.Context),
         ECEnd = D->enumerator_end(SemaRef.Context);
       EC != ECEnd; ++EC) {
    // The specified value for the enumerator.
    OwningExprResult Value = SemaRef.Owned((Expr *)0);
    if (Expr *UninstValue = EC->getInitExpr()) {
      // The enumerator's value expression is not potentially evaluated.
      EnterExpressionEvaluationContext Unevaluated(SemaRef, 
                                                   Action::Unevaluated);
      
      Value = SemaRef.InstantiateExpr(UninstValue, TemplateArgs);
    }

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
      
  // FIXME: Fixup LBraceLoc and RBraceLoc
  SemaRef.ActOnEnumBody(Enum->getLocation(), SourceLocation(), SourceLocation(),
                        Sema::DeclPtrTy::make(Enum),
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

  Owner->addDecl(SemaRef.Context, Record);
  return Record;
}

Decl *TemplateDeclInstantiator::VisitFunctionDecl(FunctionDecl *D) {
  // Check whether there is already a function template specialization for
  // this declaration.
  FunctionTemplateDecl *FunctionTemplate = D->getDescribedFunctionTemplate();
  void *InsertPos = 0;
  if (FunctionTemplate) {
    llvm::FoldingSetNodeID ID;
    FunctionTemplateSpecializationInfo::Profile(ID, 
                                          TemplateArgs.getFlatArgumentList(),
                                                TemplateArgs.flat_size());
    
    FunctionTemplateSpecializationInfo *Info 
      = FunctionTemplate->getSpecializations().FindNodeOrInsertPos(ID, 
                                                                   InsertPos);
    
    // If we already have a function template specialization, return it.
    if (Info)
      return Info->Function;
  }
  
  Sema::LocalInstantiationScope Scope(SemaRef);
  
  llvm::SmallVector<ParmVarDecl *, 4> Params;
  QualType T = InstantiateFunctionType(D, Params);
  if (T.isNull())
    return 0;
  
  // Build the instantiated method declaration.
  FunctionDecl *Function
    = FunctionDecl::Create(SemaRef.Context, Owner, D->getLocation(), 
                           D->getDeclName(), T, D->getStorageClass(),
                           D->isInline(), D->hasWrittenPrototype(),
                           D->getTypeSpecStartLoc());
  
  // FIXME: friend functions
  
  // Attach the parameters
  for (unsigned P = 0; P < Params.size(); ++P)
    Params[P]->setOwningFunction(Function);
  Function->setParams(SemaRef.Context, Params.data(), Params.size());
  
  if (InitFunctionInstantiation(Function, D))
    Function->setInvalidDecl();
  
  bool Redeclaration = false;
  bool OverloadableAttrRequired = false;
  NamedDecl *PrevDecl = 0;
  SemaRef.CheckFunctionDeclaration(Function, PrevDecl, Redeclaration,
                                   /*FIXME:*/OverloadableAttrRequired);

  if (FunctionTemplate) {
    // Record this function template specialization.
    Function->setFunctionTemplateSpecialization(SemaRef.Context,
                                                FunctionTemplate,
                                                &TemplateArgs,
                                                InsertPos);
   }

  return Function;
}

Decl *TemplateDeclInstantiator::VisitCXXMethodDecl(CXXMethodDecl *D) {
  // FIXME: Look for existing, explicit specializations.
  Sema::LocalInstantiationScope Scope(SemaRef);

  llvm::SmallVector<ParmVarDecl *, 4> Params;
  QualType T = InstantiateFunctionType(D, Params);
  if (T.isNull())
    return 0;

  // Build the instantiated method declaration.
  CXXRecordDecl *Record = cast<CXXRecordDecl>(Owner);
  CXXMethodDecl *Method
    = CXXMethodDecl::Create(SemaRef.Context, Record, D->getLocation(), 
                            D->getDeclName(), T, D->isStatic(), 
                            D->isInline());
  Method->setInstantiationOfMemberFunction(D);

  // Attach the parameters
  for (unsigned P = 0; P < Params.size(); ++P)
    Params[P]->setOwningFunction(Method);
  Method->setParams(SemaRef.Context, Params.data(), Params.size());

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
  // FIXME: Look for existing, explicit specializations.
  Sema::LocalInstantiationScope Scope(SemaRef);

  llvm::SmallVector<ParmVarDecl *, 4> Params;
  QualType T = InstantiateFunctionType(D, Params);
  if (T.isNull())
    return 0;

  // Build the instantiated method declaration.
  CXXRecordDecl *Record = cast<CXXRecordDecl>(Owner);
  QualType ClassTy = SemaRef.Context.getTypeDeclType(Record);
  DeclarationName Name
    = SemaRef.Context.DeclarationNames.getCXXConstructorName(
                                 SemaRef.Context.getCanonicalType(ClassTy));
  CXXConstructorDecl *Constructor
    = CXXConstructorDecl::Create(SemaRef.Context, Record, D->getLocation(), 
                                 Name, T, D->isExplicit(), D->isInline(), 
                                 false);
  Constructor->setInstantiationOfMemberFunction(D);

  // Attach the parameters
  for (unsigned P = 0; P < Params.size(); ++P)
    Params[P]->setOwningFunction(Constructor);
  Constructor->setParams(SemaRef.Context, Params.data(), Params.size());

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

  Record->addedConstructor(SemaRef.Context, Constructor);
  Owner->addDecl(SemaRef.Context, Constructor);
  return Constructor;
}

Decl *TemplateDeclInstantiator::VisitCXXDestructorDecl(CXXDestructorDecl *D) {
  // FIXME: Look for existing, explicit specializations.
  Sema::LocalInstantiationScope Scope(SemaRef);

  llvm::SmallVector<ParmVarDecl *, 4> Params;
  QualType T = InstantiateFunctionType(D, Params);
  if (T.isNull())
    return 0;
  assert(Params.size() == 0 && "Destructor with parameters?");

  // Build the instantiated destructor declaration.
  CXXRecordDecl *Record = cast<CXXRecordDecl>(Owner);
  QualType ClassTy = 
    SemaRef.Context.getCanonicalType(SemaRef.Context.getTypeDeclType(Record));
  CXXDestructorDecl *Destructor
    = CXXDestructorDecl::Create(SemaRef.Context, Record,
                                D->getLocation(),
             SemaRef.Context.DeclarationNames.getCXXDestructorName(ClassTy),
                                T, D->isInline(), false);
  Destructor->setInstantiationOfMemberFunction(D);
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
  // FIXME: Look for existing, explicit specializations.
  Sema::LocalInstantiationScope Scope(SemaRef);

  llvm::SmallVector<ParmVarDecl *, 4> Params;
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
  Conversion->setInstantiationOfMemberFunction(D);
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
                                           D->getLocation(), D->getDeclName());
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
  SemaRef.CurrentInstantiationScope->InstantiatedLocal(D, Param);
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
                            const TemplateArgumentList &TemplateArgs) {
  TemplateDeclInstantiator Instantiator(*this, Owner, TemplateArgs);
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
  TemplateDeclInstantiator ParamInstantiator(SemaRef, 0, TemplateArgs);
  llvm::SmallVector<QualType, 4> ParamTys;
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
    = SemaRef.InstantiateType(Proto->getResultType(), TemplateArgs,
                              D->getLocation(), D->getDeclName());
  if (ResultType.isNull())
    return QualType();

  return SemaRef.BuildFunctionType(ResultType, ParamTys.data(), ParamTys.size(),
                                   Proto->isVariadic(), Proto->getTypeQuals(),
                                   D->getLocation(), D->getDeclName());
}

/// \brief Initializes the common fields of an instantiation function 
/// declaration (New) from the corresponding fields of its template (Tmpl).
///
/// \returns true if there was an error
bool 
TemplateDeclInstantiator::InitFunctionInstantiation(FunctionDecl *New, 
                                                    FunctionDecl *Tmpl) {
  if (Tmpl->isDeleted())
    New->setDeleted();
  return false;
}

/// \brief Initializes common fields of an instantiated method
/// declaration (New) from the corresponding fields of its template
/// (Tmpl).
///
/// \returns true if there was an error
bool 
TemplateDeclInstantiator::InitMethodInstantiation(CXXMethodDecl *New, 
                                                  CXXMethodDecl *Tmpl) {
  if (InitFunctionInstantiation(New, Tmpl))
    return true;
  
  CXXRecordDecl *Record = cast<CXXRecordDecl>(Owner);
  New->setAccess(Tmpl->getAccess());
  if (Tmpl->isVirtualAsWritten()) {
    New->setVirtualAsWritten(true);
    Record->setAggregate(false);
    Record->setPOD(false);
    Record->setPolymorphic(true);
  }
  if (Tmpl->isPure()) {
    New->setPure();
    Record->setAbstract(true);
  }

  // FIXME: attributes
  // FIXME: New needs a pointer to Tmpl
  return false;
}

/// \brief Instantiate the definition of the given function from its
/// template.
///
/// \param Function the already-instantiated declaration of a
/// function.
void Sema::InstantiateFunctionDefinition(SourceLocation PointOfInstantiation,
                                         FunctionDecl *Function) {
  if (Function->isInvalidDecl())
    return;

  assert(!Function->getBody() && "Already instantiated!");
  
  // Find the function body that we'll be substituting.
  const FunctionDecl *PatternDecl = 0;
  if (FunctionTemplateDecl *Primary = Function->getPrimaryTemplate())
    PatternDecl = Primary->getTemplatedDecl();
  else 
    PatternDecl = Function->getInstantiatedFromMemberFunction();
  Stmt *Pattern = 0;
  if (PatternDecl)
    Pattern = PatternDecl->getBody(PatternDecl);

  if (!Pattern)
    return;

  InstantiatingTemplate Inst(*this, PointOfInstantiation, Function);
  if (Inst)
    return;

  ActOnStartOfFunctionDef(0, DeclPtrTy::make(Function));

  // Introduce a new scope where local variable instantiations will be
  // recorded.
  LocalInstantiationScope Scope(*this);
  
  // Introduce the instantiated function parameters into the local
  // instantiation scope.
  for (unsigned I = 0, N = PatternDecl->getNumParams(); I != N; ++I)
    Scope.InstantiatedLocal(PatternDecl->getParamDecl(I),
                            Function->getParamDecl(I));

  // Enter the scope of this instantiation. We don't use
  // PushDeclContext because we don't have a scope.
  DeclContext *PreviousContext = CurContext;
  CurContext = Function;

  // Instantiate the function body.
  OwningStmtResult Body 
    = InstantiateStmt(Pattern, getTemplateInstantiationArgs(Function));

  ActOnFinishFunctionBody(DeclPtrTy::make(Function), move(Body), 
                          /*IsInstantiation=*/true);

  CurContext = PreviousContext;

  DeclGroupRef DG(Function);
  Consumer.HandleTopLevelDecl(DG);
}

/// \brief Instantiate the definition of the given variable from its
/// template.
///
/// \param Var the already-instantiated declaration of a variable.
void Sema::InstantiateVariableDefinition(VarDecl *Var) {
  // FIXME: Implement this!
}

static bool isInstantiationOf(ASTContext &Ctx, NamedDecl *D, Decl *Other) {
  if (D->getKind() != Other->getKind())
    return false;

  if (CXXRecordDecl *Record = dyn_cast<CXXRecordDecl>(Other))
    return Ctx.getCanonicalDecl(Record->getInstantiatedFromMemberClass())
             == Ctx.getCanonicalDecl(D);

  if (FunctionDecl *Function = dyn_cast<FunctionDecl>(Other))
    return Ctx.getCanonicalDecl(Function->getInstantiatedFromMemberFunction())
             == Ctx.getCanonicalDecl(D);

  if (EnumDecl *Enum = dyn_cast<EnumDecl>(Other))
    return Ctx.getCanonicalDecl(Enum->getInstantiatedFromMemberEnum())
             == Ctx.getCanonicalDecl(D);

  // FIXME: How can we find instantiations of anonymous unions?

  return D->getDeclName() && isa<NamedDecl>(Other) &&
    D->getDeclName() == cast<NamedDecl>(Other)->getDeclName();
}

template<typename ForwardIterator>
static NamedDecl *findInstantiationOf(ASTContext &Ctx, 
                                      NamedDecl *D,
                                      ForwardIterator first,
                                      ForwardIterator last) {
  for (; first != last; ++first)
    if (isInstantiationOf(Ctx, D, *first))
      return cast<NamedDecl>(*first);

  return 0;
}

/// \brief Find the instantiation of the given declaration within the
/// current instantiation.
///
/// This routine is intended to be used when \p D is a declaration
/// referenced from within a template, that needs to mapped into the
/// corresponding declaration within an instantiation. For example,
/// given:
///
/// \code
/// template<typename T>
/// struct X {
///   enum Kind {
///     KnownValue = sizeof(T)
///   };
///
///   bool getKind() const { return KnownValue; }
/// };
///
/// template struct X<int>;
/// \endcode
///
/// In the instantiation of X<int>::getKind(), we need to map the
/// EnumConstantDecl for KnownValue (which refers to
/// X<T>::<Kind>::KnownValue) to its instantiation
/// (X<int>::<Kind>::KnownValue). InstantiateCurrentDeclRef() performs
/// this mapping from within the instantiation of X<int>.
NamedDecl * Sema::InstantiateCurrentDeclRef(NamedDecl *D) {
  DeclContext *ParentDC = D->getDeclContext();
  if (isa<ParmVarDecl>(D) || ParentDC->isFunctionOrMethod()) {
    // D is a local of some kind. Look into the map of local
    // declarations to their instantiations.
    return cast<NamedDecl>(CurrentInstantiationScope->getInstantiationOf(D));
  }

  if (NamedDecl *ParentDecl = dyn_cast<NamedDecl>(ParentDC)) {
    ParentDecl = InstantiateCurrentDeclRef(ParentDecl);
    if (!ParentDecl)
      return 0;

    ParentDC = cast<DeclContext>(ParentDecl);
  }

  if (ParentDC != D->getDeclContext()) {
    // We performed some kind of instantiation in the parent context,
    // so now we need to look into the instantiated parent context to
    // find the instantiation of the declaration D.
    NamedDecl *Result = 0;
    if (D->getDeclName()) {
      DeclContext::lookup_result Found
        = ParentDC->lookup(Context, D->getDeclName());
      Result = findInstantiationOf(Context, D, Found.first, Found.second);
    } else {
      // Since we don't have a name for the entity we're looking for,
      // our only option is to walk through all of the declarations to
      // find that name. This will occur in a few cases:
      //
      //   - anonymous struct/union within a template
      //   - unnamed class/struct/union/enum within a template
      //
      // FIXME: Find a better way to find these instantiations!
      Result = findInstantiationOf(Context, D, 
                                   ParentDC->decls_begin(Context),
                                   ParentDC->decls_end(Context));
    }
    assert(Result && "Unable to find instantiation of declaration!");
    D = Result;
  }

  if (CXXRecordDecl *Record = dyn_cast<CXXRecordDecl>(D))
    if (ClassTemplateDecl *ClassTemplate 
          = Record->getDescribedClassTemplate()) {
      // When the declaration D was parsed, it referred to the current
      // instantiation. Therefore, look through the current context,
      // which contains actual instantiations, to find the
      // instantiation of the "current instantiation" that D refers
      // to. Alternatively, we could just instantiate the
      // injected-class-name with the current template arguments, but
      // such an instantiation is far more expensive.
      for (DeclContext *DC = CurContext; !DC->isFileContext(); 
           DC = DC->getParent()) {
        if (ClassTemplateSpecializationDecl *Spec 
              = dyn_cast<ClassTemplateSpecializationDecl>(DC))
          if (Context.getCanonicalDecl(Spec->getSpecializedTemplate())
              == Context.getCanonicalDecl(ClassTemplate))
            return Spec;
      }

      assert(false && 
             "Unable to find declaration for the current instantiation");
    }

  return D;
}

/// \brief Performs template instantiation for all implicit template 
/// instantiations we have seen until this point.
void Sema::PerformPendingImplicitInstantiations() {
  while (!PendingImplicitInstantiations.empty()) {
    PendingImplicitInstantiation Inst = PendingImplicitInstantiations.front();
    PendingImplicitInstantiations.pop();
    
    if (FunctionDecl *Function = dyn_cast<FunctionDecl>(Inst.first))
      if (!Function->getBody())
        InstantiateFunctionDefinition(/*FIXME:*/Inst.second, Function);
    
    // FIXME: instantiation static member variables
  }
}
