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
#include "llvm/ADT/SmallVector.h"
#include <cassert>

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
      TemplateArgumentLists.push_back(
                                    ArgList(TemplateArgs->getFlatArgumentList(),
                                            TemplateArgs->flat_size()));
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
  enum TemplatePartialOrderingContext {
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
}

#endif // LLVM_CLANG_SEMA_TEMPLATE_H
