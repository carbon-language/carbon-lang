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

#include <vector>

#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchersInternal.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/type_traits.h"

namespace clang {
namespace ast_matchers {
namespace dynamic {

using ast_matchers::internal::DynTypedMatcher;

/// \brief A list of \c DynTypedMatcher objects.
///
/// The purpose of this list is to wrap multiple different matchers and
/// provide the right one when calling \c hasTypedMatcher/getTypedMatcher.
class MatcherList {
public:
  /// \brief An empty list.
  MatcherList();
  /// \brief Clones the matcher objects.
  MatcherList(const MatcherList &Other);
  /// \brief Clones the provided matcher.
  MatcherList(const DynTypedMatcher &Matcher);
  ~MatcherList();

  MatcherList &operator=(const MatcherList &Other);

  /// \brief Add a matcher to this list. The matcher is cloned.
  void add(const DynTypedMatcher &Matcher);

  /// \brief Empties the list.
  void reset();

  /// \brief Whether the list is empty.
  bool empty() const { return List.empty(); }

  ArrayRef<const DynTypedMatcher *> matchers() const { return List; }

  /// \brief Determines if any of the contained matchers can be converted
  ///   to \c Matcher<T>.
  ///
  /// Returns true if one, and only one, of the contained matchers can be
  /// converted to \c Matcher<T>. If there are more than one that can, the
  /// result would be ambigous and false is returned.
  template <class T>
  bool hasTypedMatcher() const {
    size_t Matches = 0;
    for (size_t I = 0, E = List.size(); I != E; ++I) {
      Matches += ast_matchers::internal::Matcher<T>::canConstructFrom(*List[I]);
    }
    return Matches == 1;
  }

  /// \brief Wrap the correct matcher as a \c Matcher<T>.
  ///
  /// Selects the appropriate matcher from the list and returns it as a
  /// \c Matcher<T>.
  /// Asserts that \c hasTypedMatcher<T>() is true.
  template <class T>
  ast_matchers::internal::Matcher<T> getTypedMatcher() const {
    assert(hasTypedMatcher<T>());
    for (size_t I = 0, E = List.size(); I != E; ++I) {
      if (ast_matchers::internal::Matcher<T>::canConstructFrom(*List[I]))
        return ast_matchers::internal::Matcher<T>::constructFrom(*List[I]);
    }
    llvm_unreachable("!hasTypedMatcher<T>()");
  }

  /// \brief String representation of the type of the value.
  ///
  /// If there are more than one matcher on the list, the string will show all
  /// the types.
  std::string getTypeAsString() const;

private:
  std::vector<const DynTypedMatcher *> List;
};

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
///  - \c MatcherList (\c DynTypedMatcher / \c Matcher<T>)
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
  VariantValue(const MatcherList &Matchers);

  /// \brief Unsigned value functions.
  bool isUnsigned() const;
  unsigned getUnsigned() const;
  void setUnsigned(unsigned Unsigned);

  /// \brief String value functions.
  bool isString() const;
  const std::string &getString() const;
  void setString(const std::string &String);

  /// \brief Matcher value functions.
  bool isMatchers() const;
  const MatcherList &getMatchers() const;
  void setMatchers(const MatcherList &Matchers);

  /// \brief Shortcut functions.
  template <class T>
  bool hasTypedMatcher() const {
    return isMatchers() && getMatchers().hasTypedMatcher<T>();
  }

  template <class T>
  ast_matchers::internal::Matcher<T> getTypedMatcher() const {
    return getMatchers().getTypedMatcher<T>();
  }

  /// \brief String representation of the type of the value.
  std::string getTypeAsString() const;

private:
  void reset();

  /// \brief All supported value types.
  enum ValueType {
    VT_Nothing,
    VT_Unsigned,
    VT_String,
    VT_Matchers
  };

  /// \brief All supported value types.
  union AllValues {
    unsigned Unsigned;
    std::string *String;
    MatcherList *Matchers;
  };

  ValueType Type;
  AllValues Value;
};

} // end namespace dynamic
} // end namespace ast_matchers
} // end namespace clang

#endif  // LLVM_CLANG_AST_MATCHERS_DYNAMIC_VARIANT_VALUE_H
