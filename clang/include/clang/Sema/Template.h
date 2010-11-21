//===------- SemaTemplate.h - C++ Templates ---------------------*- C++ -*-===/
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//===----------------------------------------------------------------------===/
//
//  This file provides types used in the semantic analysis of C++ templates.
//
//===----------------------------------------------------------------------===/
#ifndef LLVM_CLANG_SEMA_TEMPLATE_H
#define LLVM_CLANG_SEMA_TEMPLATE_H

#include "clang/AST/DeclTemplate.h"
#include "clang/AST/DeclVisitor.h"
#include "llvm/ADT/SmallVector.h"
#include <cassert>
#include <utility>

namespace clang {
  /// \brief Data structure that captures multiple levels of template argument
  /// lists for use in template instantiation.
  ///
  /// Multiple levels of template arguments occur when instantiating the 
  /// definitions of member templates. For example:
  ///
  /// \code
  /// template<typename T>
  /// struct X {
  ///   template<T Value>
  ///   struct Y {
  ///     void f();
  ///   };
  /// };
  /// \endcode
  ///
  /// When instantiating X<int>::Y<17>::f, the multi-level template argument
  /// list will contain a template argument list (int) at depth 0 and a
  /// template argument list (17) at depth 1.
  class MultiLevelTemplateArgumentList {
  public:
    typedef std::pair<const TemplateArgument *, unsigned> ArgList;
    
  private:
    /// \brief The template argument lists, stored from the innermost template
    /// argument list (first) to the outermost template argument list (last).
    llvm::SmallVector<ArgList, 4> TemplateArgumentLists;
    
  public:
    /// \brief Construct an empty set of template argument lists.
    MultiLevelTemplateArgumentList() { }
    
    /// \brief Construct a single-level template argument list.
    explicit 
    MultiLevelTemplateArgumentList(const TemplateArgumentList &TemplateArgs) {
      addOuterTemplateArguments(&TemplateArgs);
    }
    
    /// \brief Determine the number of levels in this template argument
    /// list.
    unsigned getNumLevels() const { return TemplateArgumentLists.size(); }
    
    /// \brief Retrieve the template argument at a given depth and index.
    const TemplateArgument &operator()(unsigned Depth, unsigned Index) const {
      assert(Depth < TemplateArgumentLists.size());
      assert(Index < TemplateArgumentLists[getNumLevels() - Depth - 1].second);
      return TemplateArgumentLists[getNumLevels() - Depth - 1].first[Index];
    }
    
    /// \brief Determine whether there is a non-NULL template argument at the
    /// given depth and index.
    ///
    /// There must exist a template argument list at the given depth.
    bool hasTemplateArgument(unsigned Depth, unsigned Index) const {
      assert(Depth < TemplateArgumentLists.size());
      
      if (Index >= TemplateArgumentLists[getNumLevels() - Depth - 1].second)
        return false;
      
      return !(*this)(Depth, Index).isNull();
    }
    
    /// \brief Add a new outermost level to the multi-level template argument 
    /// list.
    void addOuterTemplateArguments(const TemplateArgumentList *TemplateArgs) {
      TemplateArgumentLists.push_back(ArgList(TemplateArgs->data(),
                                              TemplateArgs->size()));
    }
    
    /// \brief Add a new outmost level to the multi-level template argument
    /// list.
    void addOuterTemplateArguments(const TemplateArgument *Args, 
                                   unsigned NumArgs) {
      TemplateArgumentLists.push_back(ArgList(Args, NumArgs));
    }
    
    /// \brief Retrieve the innermost template argument list.
    const ArgList &getInnermost() const { 
      return TemplateArgumentLists.front(); 
    }
  };
  
  /// \brief The context in which partial ordering of function templates occurs.
  enum TPOC {
    /// \brief Partial ordering of function templates for a function call.
    TPOC_Call,
    /// \brief Partial ordering of function templates for a call to a 
    /// conversion function.
    TPOC_Conversion,
    /// \brief Partial ordering of function templates in other contexts, e.g.,
    /// taking the address of a function template or matching a function 
    /// template specialization to a function template.
    TPOC_Other
  };

  // This is lame but unavoidable in a world without forward
  // declarations of enums.  The alternatives are to either pollute
  // Sema.h (by including this file) or sacrifice type safety (by
  // making Sema.h declare things as enums).
  class TemplatePartialOrderingContext {
    TPOC Value;
  public:
    TemplatePartialOrderingContext(TPOC Value) : Value(Value) {}
    operator TPOC() const { return Value; }
  };

  /// \brief Captures a template argument whose value has been deduced
  /// via c++ template argument deduction.
  class DeducedTemplateArgument : public TemplateArgument {
    /// \brief For a non-type template argument, whether the value was
    /// deduced from an array bound.
    bool DeducedFromArrayBound;

  public:
    DeducedTemplateArgument()
      : TemplateArgument(), DeducedFromArrayBound(false) { }

    DeducedTemplateArgument(const TemplateArgument &Arg,
                            bool DeducedFromArrayBound = false)
      : TemplateArgument(Arg), DeducedFromArrayBound(DeducedFromArrayBound) { }

    /// \brief Construct an integral non-type template argument that
    /// has been deduced, possible from an array bound.
    DeducedTemplateArgument(const llvm::APSInt &Value,
                            QualType ValueType,
                            bool DeducedFromArrayBound)
      : TemplateArgument(Value, ValueType), 
        DeducedFromArrayBound(DeducedFromArrayBound) { }

    /// \brief For a non-type template argument, determine whether the
    /// template argument was deduced from an array bound.
    bool wasDeducedFromArrayBound() const { return DeducedFromArrayBound; }

    /// \brief Specify whether the given non-type template argument
    /// was deduced from an array bound.
    void setDeducedFromArrayBound(bool Deduced) {
      DeducedFromArrayBound = Deduced;
    }
  };

  /// \brief A stack-allocated class that identifies which local
  /// variable declaration instantiations are present in this scope.
  ///
  /// A new instance of this class type will be created whenever we
  /// instantiate a new function declaration, which will have its own
  /// set of parameter declarations.
  class LocalInstantiationScope {
    /// \brief Reference to the semantic analysis that is performing
    /// this template instantiation.
    Sema &SemaRef;

    /// \brief A mapping from local declarations that occur
    /// within a template to their instantiations.
    ///
    /// This mapping is used during instantiation to keep track of,
    /// e.g., function parameter and variable declarations. For example,
    /// given:
    ///
    /// \code
    ///   template<typename T> T add(T x, T y) { return x + y; }
    /// \endcode
    ///
    /// when we instantiate add<int>, we will introduce a mapping from
    /// the ParmVarDecl for 'x' that occurs in the template to the
    /// instantiated ParmVarDecl for 'x'.
    llvm::DenseMap<const Decl *, Decl *> LocalDecls;

    /// \brief The outer scope, which contains local variable
    /// definitions from some other instantiation (that may not be
    /// relevant to this particular scope).
    LocalInstantiationScope *Outer;

    /// \brief Whether we have already exited this scope.
    bool Exited;

    /// \brief Whether to combine this scope with the outer scope, such that
    /// lookup will search our outer scope.
    bool CombineWithOuterScope;
    
    // This class is non-copyable
    LocalInstantiationScope(const LocalInstantiationScope &);
    LocalInstantiationScope &operator=(const LocalInstantiationScope &);

  public:
    LocalInstantiationScope(Sema &SemaRef, bool CombineWithOuterScope = false)
      : SemaRef(SemaRef), Outer(SemaRef.CurrentInstantiationScope),
        Exited(false), CombineWithOuterScope(CombineWithOuterScope)
    {
      SemaRef.CurrentInstantiationScope = this;
    }

    ~LocalInstantiationScope() {
      Exit();
    }

    /// \brief Exit this local instantiation scope early.
    void Exit() {
      if (Exited)
        return;
      
      SemaRef.CurrentInstantiationScope = Outer;
      Exited = true;
    }

    Decl *getInstantiationOf(const Decl *D);

    VarDecl *getInstantiationOf(const VarDecl *Var) {
      return cast<VarDecl>(getInstantiationOf(cast<Decl>(Var)));
    }

    ParmVarDecl *getInstantiationOf(const ParmVarDecl *Var) {
      return cast<ParmVarDecl>(getInstantiationOf(cast<Decl>(Var)));
    }

    NonTypeTemplateParmDecl *getInstantiationOf(
                                          const NonTypeTemplateParmDecl *Var) {
      return cast<NonTypeTemplateParmDecl>(getInstantiationOf(cast<Decl>(Var)));
    }

    void InstantiatedLocal(const Decl *D, Decl *Inst);
  };

  class TemplateDeclInstantiator
    : public DeclVisitor<TemplateDeclInstantiator, Decl *> {
    Sema &SemaRef;
    DeclContext *Owner;
    const MultiLevelTemplateArgumentList &TemplateArgs;

    /// \brief A list of out-of-line class template partial
    /// specializations that will need to be instantiated after the
    /// enclosing class's instantiation is complete.
    llvm::SmallVector<std::pair<ClassTemplateDecl *,
                                ClassTemplatePartialSpecializationDecl *>, 4>
      OutOfLinePartialSpecs;

  public:
    TemplateDeclInstantiator(Sema &SemaRef, DeclContext *Owner,
                             const MultiLevelTemplateArgumentList &TemplateArgs)
      : SemaRef(SemaRef), Owner(Owner), TemplateArgs(TemplateArgs) { }

    // FIXME: Once we get closer to completion, replace these manually-written
    // declarations with automatically-generated ones from
    // clang/AST/DeclNodes.inc.
    Decl *VisitTranslationUnitDecl(TranslationUnitDecl *D);
    Decl *VisitNamespaceDecl(NamespaceDecl *D);
    Decl *VisitNamespaceAliasDecl(NamespaceAliasDecl *D);
    Decl *VisitTypedefDecl(TypedefDecl *D);
    Decl *VisitVarDecl(VarDecl *D);
    Decl *VisitAccessSpecDecl(AccessSpecDecl *D);
    Decl *VisitFieldDecl(FieldDecl *D);
    Decl *VisitIndirectFieldDecl(IndirectFieldDecl *D);
    Decl *VisitStaticAssertDecl(StaticAssertDecl *D);
    Decl *VisitEnumDecl(EnumDecl *D);
    Decl *VisitEnumConstantDecl(EnumConstantDecl *D);
    Decl *VisitFriendDecl(FriendDecl *D);
    Decl *VisitFunctionDecl(FunctionDecl *D,
                            TemplateParameterList *TemplateParams = 0);
    Decl *VisitCXXRecordDecl(CXXRecordDecl *D);
    Decl *VisitCXXMethodDecl(CXXMethodDecl *D,
                             TemplateParameterList *TemplateParams = 0);
    Decl *VisitCXXConstructorDecl(CXXConstructorDecl *D);
    Decl *VisitCXXDestructorDecl(CXXDestructorDecl *D);
    Decl *VisitCXXConversionDecl(CXXConversionDecl *D);
    ParmVarDecl *VisitParmVarDecl(ParmVarDecl *D);
    Decl *VisitClassTemplateDecl(ClassTemplateDecl *D);
    Decl *VisitClassTemplatePartialSpecializationDecl(
                                    ClassTemplatePartialSpecializationDecl *D);
    Decl *VisitFunctionTemplateDecl(FunctionTemplateDecl *D);
    Decl *VisitTemplateTypeParmDecl(TemplateTypeParmDecl *D);
    Decl *VisitNonTypeTemplateParmDecl(NonTypeTemplateParmDecl *D);
    Decl *VisitTemplateTemplateParmDecl(TemplateTemplateParmDecl *D);
    Decl *VisitUsingDirectiveDecl(UsingDirectiveDecl *D);
    Decl *VisitUsingDecl(UsingDecl *D);
    Decl *VisitUsingShadowDecl(UsingShadowDecl *D);
    Decl *VisitUnresolvedUsingValueDecl(UnresolvedUsingValueDecl *D);
    Decl *VisitUnresolvedUsingTypenameDecl(UnresolvedUsingTypenameDecl *D);

    // Base case. FIXME: Remove once we can instantiate everything.
    Decl *VisitDecl(Decl *D) {
      unsigned DiagID = SemaRef.getDiagnostics().getCustomDiagID(
                                                            Diagnostic::Error,
                                                   "cannot instantiate %0 yet");
      SemaRef.Diag(D->getLocation(), DiagID)
        << D->getDeclKindName();
      
      return 0;
    }
    
    typedef 
      llvm::SmallVectorImpl<std::pair<ClassTemplateDecl *,
                                     ClassTemplatePartialSpecializationDecl *> >
        ::iterator
      delayed_partial_spec_iterator;

    /// \brief Return an iterator to the beginning of the set of
    /// "delayed" partial specializations, which must be passed to
    /// InstantiateClassTemplatePartialSpecialization once the class
    /// definition has been completed.
    delayed_partial_spec_iterator delayed_partial_spec_begin() {
      return OutOfLinePartialSpecs.begin();
    }

    /// \brief Return an iterator to the end of the set of
    /// "delayed" partial specializations, which must be passed to
    /// InstantiateClassTemplatePartialSpecialization once the class
    /// definition has been completed.
    delayed_partial_spec_iterator delayed_partial_spec_end() {
      return OutOfLinePartialSpecs.end();
    }

    // Helper functions for instantiating methods.
    TypeSourceInfo *SubstFunctionType(FunctionDecl *D,
                             llvm::SmallVectorImpl<ParmVarDecl *> &Params);
    bool InitFunctionInstantiation(FunctionDecl *New, FunctionDecl *Tmpl);
    bool InitMethodInstantiation(CXXMethodDecl *New, CXXMethodDecl *Tmpl);

    TemplateParameterList *
      SubstTemplateParams(TemplateParameterList *List);

    bool SubstQualifier(const DeclaratorDecl *OldDecl,
                        DeclaratorDecl *NewDecl);
    bool SubstQualifier(const TagDecl *OldDecl,
                        TagDecl *NewDecl);
      
    ClassTemplatePartialSpecializationDecl *
    InstantiateClassTemplatePartialSpecialization(
                                              ClassTemplateDecl *ClassTemplate,
                           ClassTemplatePartialSpecializationDecl *PartialSpec);
  };
}

#endif // LLVM_CLANG_SEMA_TEMPLATE_H
