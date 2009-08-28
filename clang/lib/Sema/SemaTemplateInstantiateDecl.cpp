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
#include "clang/Lex/Preprocessor.h"
#include "llvm/Support/Compiler.h"

using namespace clang;

namespace {
  class VISIBILITY_HIDDEN TemplateDeclInstantiator 
    : public DeclVisitor<TemplateDeclInstantiator, Decl *> {
    Sema &SemaRef;
    DeclContext *Owner;
    const MultiLevelTemplateArgumentList &TemplateArgs;
    
  public:
    typedef Sema::OwningExprResult OwningExprResult;

    TemplateDeclInstantiator(Sema &SemaRef, DeclContext *Owner,
                             const MultiLevelTemplateArgumentList &TemplateArgs)
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
    Decl *VisitFriendDecl(FriendDecl *D);
    Decl *VisitFunctionDecl(FunctionDecl *D);
    Decl *VisitCXXRecordDecl(CXXRecordDecl *D);
    Decl *VisitCXXMethodDecl(CXXMethodDecl *D,
                             TemplateParameterList *TemplateParams = 0);
    Decl *VisitCXXConstructorDecl(CXXConstructorDecl *D);
    Decl *VisitCXXDestructorDecl(CXXDestructorDecl *D);
    Decl *VisitCXXConversionDecl(CXXConversionDecl *D);
    ParmVarDecl *VisitParmVarDecl(ParmVarDecl *D);
    Decl *VisitOriginalParmVarDecl(OriginalParmVarDecl *D);
    Decl *VisitClassTemplateDecl(ClassTemplateDecl *D);
    Decl *VisitFunctionTemplateDecl(FunctionTemplateDecl *D);
    Decl *VisitTemplateTypeParmDecl(TemplateTypeParmDecl *D);
    Decl *VisitUnresolvedUsingDecl(UnresolvedUsingDecl *D);
      
    // Base case. FIXME: Remove once we can instantiate everything.
    Decl *VisitDecl(Decl *) { 
      assert(false && "Template instantiation of unknown declaration kind!");
      return 0;
    }

    const LangOptions &getLangOptions() {
      return SemaRef.getLangOptions();
    }

    // Helper functions for instantiating methods.
    QualType SubstFunctionType(FunctionDecl *D,
                             llvm::SmallVectorImpl<ParmVarDecl *> &Params);
    bool InitFunctionInstantiation(FunctionDecl *New, FunctionDecl *Tmpl);
    bool InitMethodInstantiation(CXXMethodDecl *New, CXXMethodDecl *Tmpl);

    TemplateParameterList *
      SubstTemplateParams(TemplateParameterList *List);
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
    T = SemaRef.SubstType(T, TemplateArgs, 
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

  Owner->addDecl(Typedef);
    
  return Typedef;
}

Decl *TemplateDeclInstantiator::VisitVarDecl(VarDecl *D) {
  // Do substitution on the type of the declaration
  QualType T = SemaRef.SubstType(D->getType(), TemplateArgs,
                                 D->getTypeSpecStartLoc(),
                                 D->getDeclName());
  if (T.isNull())
    return 0;

  // Build the instantiated declaration
  VarDecl *Var = VarDecl::Create(SemaRef.Context, Owner,
                                 D->getLocation(), D->getIdentifier(),
                                 T, D->getDeclaratorInfo(),
                                 D->getStorageClass());
  Var->setThreadSpecified(D->isThreadSpecified());
  Var->setCXXDirectInitializer(D->hasCXXDirectInitializer());
  Var->setDeclaredInCondition(D->isDeclaredInCondition());
 
  // If we are instantiating a static data member defined 
  // out-of-line, the instantiation will have the same lexical
  // context (which will be a namespace scope) as the template.
  if (D->isOutOfLine())
    Var->setLexicalDeclContext(D->getLexicalDeclContext());
  
  // FIXME: In theory, we could have a previous declaration for variables that
  // are not static data members.
  bool Redeclaration = false;
  SemaRef.CheckVariableDeclaration(Var, 0, Redeclaration);
  
  if (D->isOutOfLine()) {
    D->getLexicalDeclContext()->addDecl(Var);
    Owner->makeDeclVisibleInContext(Var);
  } else {
    Owner->addDecl(Var);
  }
  
  if (D->getInit()) {
    OwningExprResult Init 
      = SemaRef.SubstExpr(D->getInit(), TemplateArgs);
    if (Init.isInvalid())
      Var->setInvalidDecl();
    else if (ParenListExpr *PLE = dyn_cast<ParenListExpr>((Expr *)Init.get())) {
      // FIXME: We're faking all of the comma locations, which is suboptimal. 
      // Do we even need these comma locations?
      llvm::SmallVector<SourceLocation, 4> FakeCommaLocs;
      if (PLE->getNumExprs() > 0) {
        FakeCommaLocs.reserve(PLE->getNumExprs() - 1);
        for (unsigned I = 0, N = PLE->getNumExprs() - 1; I != N; ++I) {
          Expr *E = PLE->getExpr(I)->Retain();
          FakeCommaLocs.push_back(
                                SemaRef.PP.getLocForEndOfToken(E->getLocEnd()));
        }
        PLE->getExpr(PLE->getNumExprs() - 1)->Retain();
      }
      
      // Add the direct initializer to the declaration.
      SemaRef.AddCXXDirectInitializerToDecl(Sema::DeclPtrTy::make(Var),
                                            PLE->getLParenLoc(), 
                                            Sema::MultiExprArg(SemaRef,
                                                       (void**)PLE->getExprs(),
                                                           PLE->getNumExprs()),
                                            FakeCommaLocs.data(),
                                            PLE->getRParenLoc());
      
      // When Init is destroyed, it will destroy the instantiated ParenListExpr;
      // we've explicitly retained all of its subexpressions already.
    } else
      SemaRef.AddInitializerToDecl(Sema::DeclPtrTy::make(Var), move(Init),
                                   D->hasCXXDirectInitializer());
  } else if (!Var->isStaticDataMember() || Var->isOutOfLine())
    SemaRef.ActOnUninitializedDecl(Sema::DeclPtrTy::make(Var), false);

  // Link instantiations of static data members back to the template from
  // which they were instantiated.
  if (Var->isStaticDataMember())
    SemaRef.Context.setInstantiatedFromStaticDataMember(Var, D);
    
  return Var;
}

Decl *TemplateDeclInstantiator::VisitFieldDecl(FieldDecl *D) {
  bool Invalid = false;
  QualType T = D->getType();
  if (T->isDependentType())  {
    T = SemaRef.SubstType(T, TemplateArgs,
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
      = SemaRef.SubstExpr(BitWidth, TemplateArgs);
    if (InstantiatedBitWidth.isInvalid()) {
      Invalid = true;
      BitWidth = 0;
    } else
      BitWidth = InstantiatedBitWidth.takeAs<Expr>();
  }

  FieldDecl *Field = SemaRef.CheckFieldDecl(D->getDeclName(), T,
                                            D->getDeclaratorInfo(),
                                            cast<RecordDecl>(Owner), 
                                            D->getLocation(),
                                            D->isMutable(),
                                            BitWidth,
                                            D->getTypeSpecStartLoc(),
                                            D->getAccess(),
                                            0);
  if (Field) {
    if (Invalid)
      Field->setInvalidDecl();
    
    Owner->addDecl(Field);
  }

  return Field;
}

Decl *TemplateDeclInstantiator::VisitFriendDecl(FriendDecl *D) {
  FriendDecl::FriendUnion FU;

  // Handle friend type expressions by simply substituting template
  // parameters into the pattern type.
  if (Type *Ty = D->getFriendType()) {
    QualType T = SemaRef.SubstType(QualType(Ty,0), TemplateArgs,
                                   D->getLocation(), DeclarationName());
    if (T.isNull()) return 0;

    assert(getLangOptions().CPlusPlus0x || T->isRecordType());
    FU = T.getTypePtr();

  // Handle everything else by appropriate substitution.
  } else {
    NamedDecl *ND = D->getFriendDecl();
    assert(ND && "friend decl must be a decl or a type!");

    Decl *NewND = Visit(ND);
    if (!NewND) return 0;

    FU = cast<NamedDecl>(NewND);
  }
  
  FriendDecl *FD =
    FriendDecl::Create(SemaRef.Context, Owner, D->getLocation(), FU,
                       D->getFriendLoc());
  Owner->addDecl(FD);
  return FD;
}

Decl *TemplateDeclInstantiator::VisitStaticAssertDecl(StaticAssertDecl *D) {
  Expr *AssertExpr = D->getAssertExpr();
      
  // The expression in a static assertion is not potentially evaluated.
  EnterExpressionEvaluationContext Unevaluated(SemaRef, Action::Unevaluated);
  
  OwningExprResult InstantiatedAssertExpr
    = SemaRef.SubstExpr(AssertExpr, TemplateArgs);
  if (InstantiatedAssertExpr.isInvalid())
    return 0;

  OwningExprResult Message(SemaRef, D->getMessage());
  D->getMessage()->Retain();
  Decl *StaticAssert 
    = SemaRef.ActOnStaticAssertDeclaration(D->getLocation(), 
                                           move(InstantiatedAssertExpr),
                                           move(Message)).getAs<Decl>();
  return StaticAssert;
}

Decl *TemplateDeclInstantiator::VisitEnumDecl(EnumDecl *D) {
  EnumDecl *Enum = EnumDecl::Create(SemaRef.Context, Owner, 
                                    D->getLocation(), D->getIdentifier(),
                                    D->getTagKeywordLoc(),
                                    /*PrevDecl=*/0);
  Enum->setInstantiationOfMemberEnum(D);
  Enum->setAccess(D->getAccess());
  Owner->addDecl(Enum);
  Enum->startDefinition();

  llvm::SmallVector<Sema::DeclPtrTy, 4> Enumerators;

  EnumConstantDecl *LastEnumConst = 0;
  for (EnumDecl::enumerator_iterator EC = D->enumerator_begin(),
         ECEnd = D->enumerator_end();
       EC != ECEnd; ++EC) {
    // The specified value for the enumerator.
    OwningExprResult Value = SemaRef.Owned((Expr *)0);
    if (Expr *UninstValue = EC->getInitExpr()) {
      // The enumerator's value expression is not potentially evaluated.
      EnterExpressionEvaluationContext Unevaluated(SemaRef, 
                                                   Action::Unevaluated);
      
      Value = SemaRef.SubstExpr(UninstValue, TemplateArgs);
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
      Enum->addDecl(EnumConst);
      Enumerators.push_back(Sema::DeclPtrTy::make(EnumConst));
      LastEnumConst = EnumConst;
    }
  }
      
  // FIXME: Fixup LBraceLoc and RBraceLoc
  // FIXME: Empty Scope and AttributeList (required to handle attribute packed).
  SemaRef.ActOnEnumBody(Enum->getLocation(), SourceLocation(), SourceLocation(),
                        Sema::DeclPtrTy::make(Enum),
                        &Enumerators[0], Enumerators.size(),
                        0, 0);

  return Enum;
}

Decl *TemplateDeclInstantiator::VisitEnumConstantDecl(EnumConstantDecl *D) {
  assert(false && "EnumConstantDecls can only occur within EnumDecls.");
  return 0;
}

Decl *TemplateDeclInstantiator::VisitClassTemplateDecl(ClassTemplateDecl *D) {
  TemplateParameterList *TempParams = D->getTemplateParameters();
  TemplateParameterList *InstParams = SubstTemplateParams(TempParams);
  if (!InstParams) 
    return NULL;

  CXXRecordDecl *Pattern = D->getTemplatedDecl();
  CXXRecordDecl *RecordInst
    = CXXRecordDecl::Create(SemaRef.Context, Pattern->getTagKind(), Owner,
                            Pattern->getLocation(), Pattern->getIdentifier(),
                            Pattern->getTagKeywordLoc(), /*PrevDecl=*/ NULL);

  ClassTemplateDecl *Inst
    = ClassTemplateDecl::Create(SemaRef.Context, Owner, D->getLocation(),
                                D->getIdentifier(), InstParams, RecordInst, 0);
  RecordInst->setDescribedClassTemplate(Inst);
  Inst->setAccess(D->getAccess());
  Inst->setInstantiatedFromMemberTemplate(D);

  Owner->addDecl(Inst);
  return Inst;
}

Decl *
TemplateDeclInstantiator::VisitFunctionTemplateDecl(FunctionTemplateDecl *D) {
  TemplateParameterList *TempParams = D->getTemplateParameters();
  TemplateParameterList *InstParams = SubstTemplateParams(TempParams);
  if (!InstParams) 
    return NULL;
  
  // FIXME: Handle instantiation of nested function templates that aren't 
  // member function templates. This could happen inside a FriendDecl.
  assert(isa<CXXMethodDecl>(D->getTemplatedDecl()));
  CXXMethodDecl *InstMethod 
    = cast_or_null<CXXMethodDecl>(
                 VisitCXXMethodDecl(cast<CXXMethodDecl>(D->getTemplatedDecl()), 
                                    InstParams));
  if (!InstMethod)
    return 0;

  // Link the instantiated function template declaration to the function 
  // template from which it was instantiated.
  FunctionTemplateDecl *InstTemplate = InstMethod->getDescribedFunctionTemplate();
  assert(InstTemplate && "VisitCXXMethodDecl didn't create a template!");
  InstTemplate->setInstantiatedFromMemberTemplate(D);
  Owner->addDecl(InstTemplate);
  return InstTemplate;
}

Decl *TemplateDeclInstantiator::VisitCXXRecordDecl(CXXRecordDecl *D) {
  CXXRecordDecl *PrevDecl = 0;
  if (D->isInjectedClassName())
    PrevDecl = cast<CXXRecordDecl>(Owner);

  CXXRecordDecl *Record
    = CXXRecordDecl::Create(SemaRef.Context, D->getTagKind(), Owner, 
                            D->getLocation(), D->getIdentifier(),
                            D->getTagKeywordLoc(), PrevDecl);
  Record->setImplicit(D->isImplicit());
  // FIXME: Check against AS_none is an ugly hack to work around the issue that
  // the tag decls introduced by friend class declarations don't have an access
  // specifier. Remove once this area of the code gets sorted out.
  if (D->getAccess() != AS_none)
    Record->setAccess(D->getAccess());
  if (!D->isInjectedClassName())
    Record->setInstantiationOfMemberClass(D);

  // If the original function was part of a friend declaration,
  // inherit its namespace state.
  if (Decl::FriendObjectKind FOK = D->getFriendObjectKind())
    Record->setObjectOfFriendDecl(FOK == Decl::FOK_Declared);

  Owner->addDecl(Record);
  return Record;
}

/// Normal class members are of more specific types and therefore
/// don't make it here.  This function serves two purposes:
///   1) instantiating function templates
///   2) substituting friend declarations
/// FIXME: preserve function definitions in case #2
Decl *TemplateDeclInstantiator::VisitFunctionDecl(FunctionDecl *D) {
  // Check whether there is already a function template specialization for
  // this declaration.
  FunctionTemplateDecl *FunctionTemplate = D->getDescribedFunctionTemplate();
  void *InsertPos = 0;
  if (FunctionTemplate) {
    llvm::FoldingSetNodeID ID;
    FunctionTemplateSpecializationInfo::Profile(ID, 
                             TemplateArgs.getInnermost().getFlatArgumentList(),
                                       TemplateArgs.getInnermost().flat_size(),
                                                SemaRef.Context);
    
    FunctionTemplateSpecializationInfo *Info 
      = FunctionTemplate->getSpecializations().FindNodeOrInsertPos(ID, 
                                                                   InsertPos);
    
    // If we already have a function template specialization, return it.
    if (Info)
      return Info->Function;
  }
  
  Sema::LocalInstantiationScope Scope(SemaRef);
  
  llvm::SmallVector<ParmVarDecl *, 4> Params;
  QualType T = SubstFunctionType(D, Params);
  if (T.isNull())
    return 0;

  // Build the instantiated method declaration.
  DeclContext *DC = SemaRef.FindInstantiatedContext(D->getDeclContext());
  FunctionDecl *Function =
      FunctionDecl::Create(SemaRef.Context, DC, D->getLocation(), 
                           D->getDeclName(), T, D->getDeclaratorInfo(),
                           D->getStorageClass(),
                           D->isInline(), D->hasWrittenPrototype());
  Function->setLexicalDeclContext(Owner);

  // Attach the parameters
  for (unsigned P = 0; P < Params.size(); ++P)
    Params[P]->setOwningFunction(Function);
  Function->setParams(SemaRef.Context, Params.data(), Params.size());

  // If the original function was part of a friend declaration,
  // inherit its namespace state and add it to the owner.
  if (Decl::FriendObjectKind FOK = D->getFriendObjectKind()) {
    bool WasDeclared = (FOK == Decl::FOK_Declared);
    Function->setObjectOfFriendDecl(WasDeclared);
    if (!Owner->isDependentContext())
      DC->makeDeclVisibleInContext(Function);
  }
  
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
                                                &TemplateArgs.getInnermost(),
                                                InsertPos);
  }

  return Function;
}

Decl *
TemplateDeclInstantiator::VisitCXXMethodDecl(CXXMethodDecl *D,
                                      TemplateParameterList *TemplateParams) {
  FunctionTemplateDecl *FunctionTemplate = D->getDescribedFunctionTemplate();
  void *InsertPos = 0;
  if (FunctionTemplate && !TemplateParams) {
    // We are creating a function template specialization from a function 
    // template. Check whether there is already a function template 
    // specialization for this particular set of template arguments.
    llvm::FoldingSetNodeID ID;
    FunctionTemplateSpecializationInfo::Profile(ID, 
                            TemplateArgs.getInnermost().getFlatArgumentList(),
                                      TemplateArgs.getInnermost().flat_size(),
                                                SemaRef.Context);
    
    FunctionTemplateSpecializationInfo *Info 
      = FunctionTemplate->getSpecializations().FindNodeOrInsertPos(ID, 
                                                                   InsertPos);
    
    // If we already have a function template specialization, return it.
    if (Info)
      return Info->Function;
  }

  Sema::LocalInstantiationScope Scope(SemaRef);

  llvm::SmallVector<ParmVarDecl *, 4> Params;
  QualType T = SubstFunctionType(D, Params);
  if (T.isNull())
    return 0;

  // Build the instantiated method declaration.
  CXXRecordDecl *Record = cast<CXXRecordDecl>(Owner);
  CXXMethodDecl *Method = 0;
  
  DeclarationName Name = D->getDeclName();
  if (CXXConstructorDecl *Constructor = dyn_cast<CXXConstructorDecl>(D)) {
    QualType ClassTy = SemaRef.Context.getTypeDeclType(Record);
    Name = SemaRef.Context.DeclarationNames.getCXXConstructorName(
                                    SemaRef.Context.getCanonicalType(ClassTy));
    Method = CXXConstructorDecl::Create(SemaRef.Context, Record, 
                                        Constructor->getLocation(), 
                                        Name, T, 
                                        Constructor->getDeclaratorInfo(),
                                        Constructor->isExplicit(), 
                                        Constructor->isInline(), false);
  } else if (CXXDestructorDecl *Destructor = dyn_cast<CXXDestructorDecl>(D)) {
    QualType ClassTy = SemaRef.Context.getTypeDeclType(Record);
    Name = SemaRef.Context.DeclarationNames.getCXXDestructorName(
                                   SemaRef.Context.getCanonicalType(ClassTy));
    Method = CXXDestructorDecl::Create(SemaRef.Context, Record,
                                       Destructor->getLocation(), Name,
                                       T, Destructor->isInline(), false);
  } else if (CXXConversionDecl *Conversion = dyn_cast<CXXConversionDecl>(D)) {
    CanQualType ConvTy 
      = SemaRef.Context.getCanonicalType(
                                      T->getAsFunctionType()->getResultType());
    Name = SemaRef.Context.DeclarationNames.getCXXConversionFunctionName(
                                                                      ConvTy);
    Method = CXXConversionDecl::Create(SemaRef.Context, Record,
                                       Conversion->getLocation(), Name,
                                       T, Conversion->getDeclaratorInfo(),
                                       Conversion->isInline(), 
                                       Conversion->isExplicit());
  } else {
    Method = CXXMethodDecl::Create(SemaRef.Context, Record, D->getLocation(), 
                                   D->getDeclName(), T, D->getDeclaratorInfo(),
                                   D->isStatic(), D->isInline());
  }

  if (TemplateParams) {
    // Our resulting instantiation is actually a function template, since we
    // are substituting only the outer template parameters. For example, given
    // 
    //   template<typename T>
    //   struct X {
    //     template<typename U> void f(T, U);
    //   };
    //
    //   X<int> x;
    //
    // We are instantiating the member template "f" within X<int>, which means
    // substituting int for T, but leaving "f" as a member function template.
    // Build the function template itself.
    FunctionTemplate = FunctionTemplateDecl::Create(SemaRef.Context, Record,
                                                    Method->getLocation(),
                                                    Method->getDeclName(), 
                                                    TemplateParams, Method);
    if (D->isOutOfLine())
      FunctionTemplate->setLexicalDeclContext(D->getLexicalDeclContext());
    Method->setDescribedFunctionTemplate(FunctionTemplate);
  } else if (!FunctionTemplate)
    Method->setInstantiationOfMemberFunction(D);

  // If we are instantiating a member function defined 
  // out-of-line, the instantiation will have the same lexical
  // context (which will be a namespace scope) as the template.
  if (D->isOutOfLine())
    Method->setLexicalDeclContext(D->getLexicalDeclContext());
  
  // Attach the parameters
  for (unsigned P = 0; P < Params.size(); ++P)
    Params[P]->setOwningFunction(Method);
  Method->setParams(SemaRef.Context, Params.data(), Params.size());

  if (InitMethodInstantiation(Method, D))
    Method->setInvalidDecl();

  NamedDecl *PrevDecl = 0;
  
  if (!FunctionTemplate || TemplateParams) {
    PrevDecl = SemaRef.LookupQualifiedName(Owner, Name, 
                                           Sema::LookupOrdinaryName, true);
  
    // In C++, the previous declaration we find might be a tag type
    // (class or enum). In this case, the new declaration will hide the
    // tag type. Note that this does does not apply if we're declaring a
    // typedef (C++ [dcl.typedef]p4).
    if (PrevDecl && PrevDecl->getIdentifierNamespace() == Decl::IDNS_Tag)
      PrevDecl = 0;
  }

  if (FunctionTemplate && !TemplateParams)
    // Record this function template specialization.
    Method->setFunctionTemplateSpecialization(SemaRef.Context,
                                              FunctionTemplate,
                                              &TemplateArgs.getInnermost(),
                                              InsertPos);
  
  bool Redeclaration = false;
  bool OverloadableAttrRequired = false;
  SemaRef.CheckFunctionDeclaration(Method, PrevDecl, Redeclaration,
                                   /*FIXME:*/OverloadableAttrRequired);

  if (!FunctionTemplate && (!Method->isInvalidDecl() || !PrevDecl))
    Owner->addDecl(Method);
  
  return Method;
}

Decl *TemplateDeclInstantiator::VisitCXXConstructorDecl(CXXConstructorDecl *D) {
  return VisitCXXMethodDecl(D);
}

Decl *TemplateDeclInstantiator::VisitCXXDestructorDecl(CXXDestructorDecl *D) {
  return VisitCXXMethodDecl(D);
}

Decl *TemplateDeclInstantiator::VisitCXXConversionDecl(CXXConversionDecl *D) {
  return VisitCXXMethodDecl(D);
}

ParmVarDecl *TemplateDeclInstantiator::VisitParmVarDecl(ParmVarDecl *D) {
  QualType OrigT = SemaRef.SubstType(D->getOriginalType(), TemplateArgs,
                                           D->getLocation(), D->getDeclName());
  if (OrigT.isNull())
    return 0;

  QualType T = SemaRef.adjustParameterType(OrigT);

  // Allocate the parameter
  ParmVarDecl *Param = 0;
  if (T == OrigT)
    Param = ParmVarDecl::Create(SemaRef.Context, Owner, D->getLocation(),
                                D->getIdentifier(), T, D->getDeclaratorInfo(),
                                D->getStorageClass(), 0);
  else
    Param = OriginalParmVarDecl::Create(SemaRef.Context, Owner, 
                                        D->getLocation(), D->getIdentifier(),
                                        T, D->getDeclaratorInfo(), OrigT,
                                        D->getStorageClass(), 0);

  // Mark the default argument as being uninstantiated.
  if (Expr *Arg = D->getDefaultArg())
    Param->setUninstantiatedDefaultArg(Arg);
  
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

Decl *TemplateDeclInstantiator::VisitTemplateTypeParmDecl(
                                                    TemplateTypeParmDecl *D) {
  // TODO: don't always clone when decls are refcounted.
  const Type* T = D->getTypeForDecl();
  assert(T->isTemplateTypeParmType());
  const TemplateTypeParmType *TTPT = T->getAs<TemplateTypeParmType>();
  
  TemplateTypeParmDecl *Inst =
    TemplateTypeParmDecl::Create(SemaRef.Context, Owner, D->getLocation(),
                                 TTPT->getDepth(), TTPT->getIndex(),
                                 TTPT->getName(),
                                 D->wasDeclaredWithTypename(),
                                 D->isParameterPack());

  if (D->hasDefaultArgument()) {
    QualType DefaultPattern = D->getDefaultArgument();
    QualType DefaultInst
      = SemaRef.SubstType(DefaultPattern, TemplateArgs,
                          D->getDefaultArgumentLoc(),
                          D->getDeclName());
    
    Inst->setDefaultArgument(DefaultInst,
                             D->getDefaultArgumentLoc(),
                             D->defaultArgumentWasInherited() /* preserve? */);
  }

  return Inst;
}

Decl *
TemplateDeclInstantiator::VisitUnresolvedUsingDecl(UnresolvedUsingDecl *D) {
  NestedNameSpecifier *NNS = 
    SemaRef.SubstNestedNameSpecifier(D->getTargetNestedNameSpecifier(), 
                                     D->getTargetNestedNameRange(), 
                                     TemplateArgs);
  if (!NNS)
    return 0;
  
  CXXScopeSpec SS;
  SS.setRange(D->getTargetNestedNameRange());
  SS.setScopeRep(NNS);
  
  return SemaRef.BuildUsingDeclaration(D->getLocation(), SS, 
                                       D->getTargetNameLocation(), 
                                       D->getTargetName(), 0, D->isTypeName());
}

Decl *Sema::SubstDecl(Decl *D, DeclContext *Owner,
                      const MultiLevelTemplateArgumentList &TemplateArgs) {
  TemplateDeclInstantiator Instantiator(*this, Owner, TemplateArgs);
  return Instantiator.Visit(D);
}

/// \brief Instantiates a nested template parameter list in the current
/// instantiation context.
///
/// \param L The parameter list to instantiate
///
/// \returns NULL if there was an error
TemplateParameterList *
TemplateDeclInstantiator::SubstTemplateParams(TemplateParameterList *L) {
  // Get errors for all the parameters before bailing out.
  bool Invalid = false;

  unsigned N = L->size();
  typedef llvm::SmallVector<Decl*,8> ParamVector;
  ParamVector Params;
  Params.reserve(N);
  for (TemplateParameterList::iterator PI = L->begin(), PE = L->end();
       PI != PE; ++PI) {
    Decl *D = Visit(*PI);
    Params.push_back(D);
    Invalid = Invalid || !D;
  }

  // Clean up if we had an error.
  if (Invalid) {
    for (ParamVector::iterator PI = Params.begin(), PE = Params.end();
         PI != PE; ++PI)
      if (*PI)
        (*PI)->Destroy(SemaRef.Context);
    return NULL;
  }

  TemplateParameterList *InstL
    = TemplateParameterList::Create(SemaRef.Context, L->getTemplateLoc(),
                                    L->getLAngleLoc(), &Params.front(), N,
                                    L->getRAngleLoc());
  return InstL;
} 

/// \brief Does substitution on the type of the given function, including
/// all of the function parameters.
///
/// \param D The function whose type will be the basis of the substitution
///
/// \param Params the instantiated parameter declarations

/// \returns the instantiated function's type if successful, a NULL
/// type if there was an error.
QualType 
TemplateDeclInstantiator::SubstFunctionType(FunctionDecl *D,
                              llvm::SmallVectorImpl<ParmVarDecl *> &Params) {
  bool InvalidDecl = false;

  // Substitute all of the function's formal parameter types.
  TemplateDeclInstantiator ParamInstantiator(SemaRef, 0, TemplateArgs);
  llvm::SmallVector<QualType, 4> ParamTys;
  for (FunctionDecl::param_iterator P = D->param_begin(), 
                                 PEnd = D->param_end();
       P != PEnd; ++P) {
    if (ParmVarDecl *PInst = ParamInstantiator.VisitParmVarDecl(*P)) {
      if (PInst->getType()->isVoidType()) {
        SemaRef.Diag(PInst->getLocation(), diag::err_param_with_void_type);
        PInst->setInvalidDecl();
      } else if (SemaRef.RequireNonAbstractType(PInst->getLocation(), 
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
    = SemaRef.SubstType(Proto->getResultType(), TemplateArgs,
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
  
  // If we are performing substituting explicitly-specified template arguments
  // or deduced template arguments into a function template and we reach this
  // point, we are now past the point where SFINAE applies and have committed
  // to keeping the new function template specialization. We therefore 
  // convert the active template instantiation for the function template 
  // into a template instantiation for this specific function template
  // specialization, which is not a SFINAE context, so that we diagnose any
  // further errors in the declaration itself.
  typedef Sema::ActiveTemplateInstantiation ActiveInstType;
  ActiveInstType &ActiveInst = SemaRef.ActiveTemplateInstantiations.back();
  if (ActiveInst.Kind == ActiveInstType::ExplicitTemplateArgumentSubstitution ||
      ActiveInst.Kind == ActiveInstType::DeducedTemplateArgumentSubstitution) {
    if (FunctionTemplateDecl *FunTmpl 
          = dyn_cast<FunctionTemplateDecl>((Decl *)ActiveInst.Entity)) {
      assert(FunTmpl->getTemplatedDecl() == Tmpl && 
             "Deduction from the wrong function template?");
      (void) FunTmpl;
      ActiveInst.Kind = ActiveInstType::TemplateInstantiation;
      ActiveInst.Entity = reinterpret_cast<uintptr_t>(New);
    }
  }
    
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
    Record->setEmpty(false);
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
/// \param PointOfInstantiation the point at which the instantiation was
/// required. Note that this is not precisely a "point of instantiation"
/// for the function, but it's close.
///
/// \param Function the already-instantiated declaration of a
/// function template specialization or member function of a class template
/// specialization.
///
/// \param Recursive if true, recursively instantiates any functions that
/// are required by this instantiation.
void Sema::InstantiateFunctionDefinition(SourceLocation PointOfInstantiation,
                                         FunctionDecl *Function,
                                         bool Recursive) {
  if (Function->isInvalidDecl())
    return;

  assert(!Function->getBody() && "Already instantiated!");
  
  // Find the function body that we'll be substituting.
  const FunctionDecl *PatternDecl = 0;
  if (FunctionTemplateDecl *Primary = Function->getPrimaryTemplate()) {
    while (Primary->getInstantiatedFromMemberTemplate())
      Primary = Primary->getInstantiatedFromMemberTemplate();
    
    PatternDecl = Primary->getTemplatedDecl();
  } else 
    PatternDecl = Function->getInstantiatedFromMemberFunction();
  Stmt *Pattern = 0;
  if (PatternDecl)
    Pattern = PatternDecl->getBody(PatternDecl);

  if (!Pattern)
    return;

  InstantiatingTemplate Inst(*this, PointOfInstantiation, Function);
  if (Inst)
    return;

  // If we're performing recursive template instantiation, create our own
  // queue of pending implicit instantiations that we will instantiate later,
  // while we're still within our own instantiation context.
  std::deque<PendingImplicitInstantiation> SavedPendingImplicitInstantiations;
  if (Recursive)
    PendingImplicitInstantiations.swap(SavedPendingImplicitInstantiations);
  
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
    = SubstStmt(Pattern, getTemplateInstantiationArgs(Function));

  ActOnFinishFunctionBody(DeclPtrTy::make(Function), move(Body), 
                          /*IsInstantiation=*/true);

  CurContext = PreviousContext;

  DeclGroupRef DG(Function);
  Consumer.HandleTopLevelDecl(DG);
  
  if (Recursive) {
    // Instantiate any pending implicit instantiations found during the
    // instantiation of this template. 
    PerformPendingImplicitInstantiations();
    
    // Restore the set of pending implicit instantiations.
    PendingImplicitInstantiations.swap(SavedPendingImplicitInstantiations);
  }
}

/// \brief Instantiate the definition of the given variable from its
/// template.
///
/// \param PointOfInstantiation the point at which the instantiation was
/// required. Note that this is not precisely a "point of instantiation"
/// for the function, but it's close.
///
/// \param Var the already-instantiated declaration of a static member
/// variable of a class template specialization.
///
/// \param Recursive if true, recursively instantiates any functions that
/// are required by this instantiation.
void Sema::InstantiateStaticDataMemberDefinition(
                                          SourceLocation PointOfInstantiation,
                                                 VarDecl *Var,
                                                 bool Recursive) {
  if (Var->isInvalidDecl())
    return;
  
  // Find the out-of-line definition of this static data member.
  // FIXME: Do we have to look for specializations separately?
  VarDecl *Def = Var->getInstantiatedFromStaticDataMember();
  bool FoundOutOfLineDef = false;
  assert(Def && "This data member was not instantiated from a template?");
  assert(Def->isStaticDataMember() && "Not a static data member?"); 
  for (VarDecl::redecl_iterator RD = Def->redecls_begin(), 
                             RDEnd = Def->redecls_end();
       RD != RDEnd; ++RD) {
    if (RD->getLexicalDeclContext()->isFileContext()) {
      Def = *RD;
      FoundOutOfLineDef = true;
    }
  }
  
  if (!FoundOutOfLineDef) {
    // We did not find an out-of-line definition of this static data member,
    // so we won't perform any instantiation. Rather, we rely on the user to
    // instantiate this definition (or provide a specialization for it) in 
    // another translation unit. 
    return;
  }

  InstantiatingTemplate Inst(*this, PointOfInstantiation, Var);
  if (Inst)
    return;
  
  // If we're performing recursive template instantiation, create our own
  // queue of pending implicit instantiations that we will instantiate later,
  // while we're still within our own instantiation context.
  std::deque<PendingImplicitInstantiation> SavedPendingImplicitInstantiations;
  if (Recursive)
    PendingImplicitInstantiations.swap(SavedPendingImplicitInstantiations);
    
  // Enter the scope of this instantiation. We don't use
  // PushDeclContext because we don't have a scope.
  DeclContext *PreviousContext = CurContext;
  CurContext = Var->getDeclContext();
  
  Var = cast_or_null<VarDecl>(SubstDecl(Def, Var->getDeclContext(),
                                          getTemplateInstantiationArgs(Var)));
  
  CurContext = PreviousContext;

  if (Var) {
    DeclGroupRef DG(Var);
    Consumer.HandleTopLevelDecl(DG);
  }
  
  if (Recursive) {
    // Instantiate any pending implicit instantiations found during the
    // instantiation of this template. 
    PerformPendingImplicitInstantiations();
    
    // Restore the set of pending implicit instantiations.
    PendingImplicitInstantiations.swap(SavedPendingImplicitInstantiations);
  }  
}

static bool isInstantiationOf(ASTContext &Ctx, NamedDecl *D, Decl *Other) {
  if (D->getKind() != Other->getKind())
    return false;

  if (CXXRecordDecl *Record = dyn_cast<CXXRecordDecl>(Other)) {
    if (CXXRecordDecl *Pattern = Record->getInstantiatedFromMemberClass())
      return Pattern->getCanonicalDecl() == D->getCanonicalDecl();
    else
      return false;
  }
  
  if (FunctionDecl *Function = dyn_cast<FunctionDecl>(Other)) {
    if (FunctionDecl *Pattern = Function->getInstantiatedFromMemberFunction())
      return Pattern->getCanonicalDecl() == D->getCanonicalDecl();
    else
      return false;
  }

  if (EnumDecl *Enum = dyn_cast<EnumDecl>(Other)) {
    if (EnumDecl *Pattern = Enum->getInstantiatedFromMemberEnum())
      return Pattern->getCanonicalDecl() == D->getCanonicalDecl();
    else
      return false;
  }

  if (VarDecl *Var = dyn_cast<VarDecl>(Other))
    if (Var->isStaticDataMember()) {
      if (VarDecl *Pattern = Var->getInstantiatedFromStaticDataMember())
        return Pattern->getCanonicalDecl() == D->getCanonicalDecl();
      else
        return false;
    }

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

/// \brief Finds the instantiation of the given declaration context
/// within the current instantiation.
///
/// \returns NULL if there was an error
DeclContext *Sema::FindInstantiatedContext(DeclContext* DC) {
  if (NamedDecl *D = dyn_cast<NamedDecl>(DC)) {
    Decl* ID = FindInstantiatedDecl(D);
    return cast_or_null<DeclContext>(ID);
  } else return DC;
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
NamedDecl * Sema::FindInstantiatedDecl(NamedDecl *D) {
  DeclContext *ParentDC = D->getDeclContext();
  if (isa<ParmVarDecl>(D) || ParentDC->isFunctionOrMethod()) {
    // D is a local of some kind. Look into the map of local
    // declarations to their instantiations.
    return cast<NamedDecl>(CurrentInstantiationScope->getInstantiationOf(D));
  }

  ParentDC = FindInstantiatedContext(ParentDC);
  if (!ParentDC) return 0;

  if (ParentDC != D->getDeclContext()) {
    // We performed some kind of instantiation in the parent context,
    // so now we need to look into the instantiated parent context to
    // find the instantiation of the declaration D.
    NamedDecl *Result = 0;
    if (D->getDeclName()) {
      DeclContext::lookup_result Found = ParentDC->lookup(D->getDeclName());
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
                                   ParentDC->decls_begin(),
                                   ParentDC->decls_end());
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
          if (Spec->getSpecializedTemplate()->getCanonicalDecl()
              == ClassTemplate->getCanonicalDecl())
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
    PendingImplicitInstantiations.pop_front();
    
    // Instantiate function definitions
    if (FunctionDecl *Function = dyn_cast<FunctionDecl>(Inst.first)) {
      if (!Function->getBody())
        InstantiateFunctionDefinition(/*FIXME:*/Inst.second, Function, true);
      continue;
    }
    
    // Instantiate static data member definitions.
    VarDecl *Var = cast<VarDecl>(Inst.first);
    assert(Var->isStaticDataMember() && "Not a static data member?");
    InstantiateStaticDataMemberDefinition(/*FIXME:*/Inst.second, Var, true);
  }
}
