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
  struct MultiLevelTemplateArgumentList {
    /// \brief The template argument lists, stored from the innermost template
    /// argument list (first) to the outermost template argument list (last)
    llvm::SmallVector<const TemplateArgumentList *, 4> TemplateArgumentLists;
    
  public:
    /// \brief Construct an empty set of template argument lists.
    MultiLevelTemplateArgumentList() { }
    
    /// \brief Construct a single-level template argument list.
    MultiLevelTemplateArgumentList(const TemplateArgumentList &TemplateArgs) {
      TemplateArgumentLists.push_back(&TemplateArgs);
    }
    
    /// \brief Determine the number of levels in this template argument
    /// list.
    unsigned getNumLevels() const { return TemplateArgumentLists.size(); }
    
    /// \brief Retrieve the template argument at a given depth and index.
    const TemplateArgument &operator()(unsigned Depth, unsigned Index) const {
      assert(Depth < TemplateArgumentLists.size());
      assert(Index < TemplateArgumentLists[getNumLevels() - Depth - 1]->size());
      return TemplateArgumentLists[getNumLevels() - Depth - 1]->get(Index);
    }
    
    /// \brief Determine whether there is a non-NULL template argument at the
    /// given depth and index.
    ///
    /// There must exist a template argument list at the given depth.
    bool hasTemplateArgument(unsigned Depth, unsigned Index) const {
      assert(Depth < TemplateArgumentLists.size());
      
      if (Index >= TemplateArgumentLists[getNumLevels() - Depth - 1]->size())
        return false;
      
      return !(*this)(Depth, Index).isNull();
    }
    
    /// \brief Add a new outermost level to the multi-level template argument 
    /// list.
    void addOuterTemplateArguments(const TemplateArgumentList *TemplateArgs) {
      TemplateArgumentLists.push_back(TemplateArgs);
    }
    
    /// \brief Retrieve the innermost template argument list.
    const TemplateArgumentList &getInnermost() const {
      return *TemplateArgumentLists.front();
    }
    
    // Implicit conversion to a single template argument list, to facilitate a
    // gradual transition to MultiLevelTemplateArgumentLists.
    operator const TemplateArgumentList &() const {
      assert(getNumLevels() == 1 && 
             "Conversion only works with a single level of template arguments");
      return *TemplateArgumentLists.front();
    }
  };
}

#endif // LLVM_CLANG_SEMA_TEMPLATE_H
