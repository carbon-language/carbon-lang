//===--- VariantValue.h - Polymorphic value type -*- C++ -*-===/
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Polymorphic value type.
///
/// Supports all the types required for dynamic Matcher construction.
///  Used by the registry to construct matchers in a generic way.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_MATCHERS_DYNAMIC_VARIANT_VALUE_H
#define LLVM_CLANG_AST_MATCHERS_DYNAMIC_VARIANT_VALUE_H

#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchersInternal.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/type_traits.h"

namespace clang {
namespace ast_matchers {
namespace dynamic {

using ast_matchers::internal::DynTypedMatcher;

/// \brief Variant value class.
///
/// Basically, a tagged union with value type semantics.
/// It is used by the registry as the return value and argument type for the
/// matcher factory methods.
/// It can be constructed from any of the supported types. It supports
/// copy/assignment.
///
/// Supported types:
///  - \c unsigned
///  - \c std::string
///  - \c DynTypedMatcher, and any \c Matcher<T>
class VariantValue {
public:
  VariantValue() : Type(VT_Nothing) {}

  VariantValue(const VariantValue &Other);
  ~VariantValue();
  VariantValue &operator=(const VariantValue &Other);

  /// \brief Specific constructors for each supported type.
  VariantValue(unsigned Unsigned);
  VariantValue(const std::string &String);
  VariantValue(const DynTypedMatcher &Matcher);

  /// \brief Unsigned value functions.
  bool isUnsigned() const;
  unsigned getUnsigned() const;
  void setUnsigned(unsigned Unsigned);

  /// \brief String value functions.
  bool isString() const;
  const std::string &getString() const;
  void setString(const std::string &String);

  /// \brief Matcher value functions.
  bool isMatcher() const;
  const DynTypedMatcher &getMatcher() const;
  void setMatcher(const DynTypedMatcher &Matcher);
  /// \brief Set the value to be \c Matcher by taking ownership of the object.
  void takeMatcher(DynTypedMatcher *Matcher);

  /// \brief Specialized Matcher<T> is/get functions.
  template <class T>
  bool isTypedMatcher() const {
    // TODO: Add some logic to test if T is actually valid for the underlying
    // type of the matcher.
    return isMatcher();
  }

  template <class T>
  ast_matchers::internal::Matcher<T> getTypedMatcher() const {
    return ast_matchers::internal::makeMatcher(
        new DerivedTypeMatcher<T>(getMatcher()));
  }

private:
  void reset();

  /// \brief Matcher bridge between a Matcher<T> and a generic DynTypedMatcher.
  template <class T>
  class DerivedTypeMatcher :
      public ast_matchers::internal::MatcherInterface<T> {
  public:
    explicit DerivedTypeMatcher(const DynTypedMatcher &DynMatcher)
        : DynMatcher(DynMatcher.clone()) {}
    virtual ~DerivedTypeMatcher() {}

    typedef ast_matchers::internal::ASTMatchFinder ASTMatchFinder;
    typedef ast_matchers::internal::BoundNodesTreeBuilder BoundNodesTreeBuilder;
    bool matches(const T &Node, ASTMatchFinder *Finder,
                 BoundNodesTreeBuilder *Builder) const {
      return DynMatcher->matches(ast_type_traits::DynTypedNode::create(Node),
                                 Finder, Builder);
    }

  private:
    const OwningPtr<DynTypedMatcher> DynMatcher;
  };

  /// \brief All supported value types.
  enum ValueType {
    VT_Nothing,
    VT_Unsigned,
    VT_String,
    VT_Matcher
  };

  /// \brief All supported value types.
  union AllValues {
    unsigned Unsigned;
    std::string *String;
    DynTypedMatcher *Matcher;
  };

  ValueType Type;
  AllValues Value;
};

} // end namespace dynamic
} // end namespace ast_matchers
} // end namespace clang

#endif  // LLVM_CLANG_AST_MATCHERS_DYNAMIC_VARIANT_VALUE_H
