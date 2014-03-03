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
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/type_traits.h"
#include <vector>

namespace clang {
namespace ast_matchers {
namespace dynamic {

using ast_matchers::internal::DynTypedMatcher;

/// \brief A variant matcher object.
///
/// The purpose of this object is to abstract simple and polymorphic matchers
/// into a single object type.
/// Polymorphic matchers might be implemented as a list of all the possible
/// overloads of the matcher. \c VariantMatcher knows how to select the
/// appropriate overload when needed.
/// To get a real matcher object out of a \c VariantMatcher you can do:
///  - getSingleMatcher() which returns a matcher, only if it is not ambiguous
///    to decide which matcher to return. Eg. it contains only a single
///    matcher, or a polymorphic one with only one overload.
///  - hasTypedMatcher<T>()/getTypedMatcher<T>(): These calls will determine if
///    the underlying matcher(s) can unambiguously return a Matcher<T>.
class VariantMatcher {
  /// \brief Methods that depend on T from hasTypedMatcher/getTypedMatcher.
  class MatcherOps {
  public:
    virtual ~MatcherOps();
    virtual bool canConstructFrom(const DynTypedMatcher &Matcher) const = 0;
    virtual void constructFrom(const DynTypedMatcher &Matcher) = 0;
    virtual void constructVariadicOperator(
        ast_matchers::internal::VariadicOperatorFunction Func,
        ArrayRef<VariantMatcher> InnerMatchers) = 0;
  };

  /// \brief Payload interface to be specialized by each matcher type.
  ///
  /// It follows a similar interface as VariantMatcher itself.
  class Payload : public RefCountedBaseVPTR {
  public:
    virtual ~Payload();
    virtual llvm::Optional<DynTypedMatcher> getSingleMatcher() const = 0;
    virtual std::string getTypeAsString() const = 0;
    virtual void makeTypedMatcher(MatcherOps &Ops) const = 0;
  };

public:
  /// \brief A null matcher.
  VariantMatcher();

  /// \brief Clones the provided matcher.
  static VariantMatcher SingleMatcher(const DynTypedMatcher &Matcher);

  /// \brief Clones the provided matchers.
  ///
  /// They should be the result of a polymorphic matcher.
  static VariantMatcher PolymorphicMatcher(ArrayRef<DynTypedMatcher> Matchers);

  /// \brief Creates a 'variadic' operator matcher.
  ///
  /// It will bind to the appropriate type on getTypedMatcher<T>().
  static VariantMatcher VariadicOperatorMatcher(
      ast_matchers::internal::VariadicOperatorFunction Func,
      ArrayRef<VariantMatcher> Args);

  /// \brief Makes the matcher the "null" matcher.
  void reset();

  /// \brief Whether the matcher is null.
  bool isNull() const { return !Value; }

  /// \brief Return a single matcher, if there is no ambiguity.
  ///
  /// \returns the matcher, if there is only one matcher. An empty Optional, if
  /// the underlying matcher is a polymorphic matcher with more than one
  /// representation.
  llvm::Optional<DynTypedMatcher> getSingleMatcher() const;

  /// \brief Determines if the contained matcher can be converted to
  ///   \c Matcher<T>.
  ///
  /// For the Single case, it returns true if it can be converted to
  /// \c Matcher<T>.
  /// For the Polymorphic case, it returns true if one, and only one, of the
  /// overloads can be converted to \c Matcher<T>. If there are more than one
  /// that can, the result would be ambiguous and false is returned.
  template <class T>
  bool hasTypedMatcher() const {
    TypedMatcherOps<T> Ops;
    if (Value) Value->makeTypedMatcher(Ops);
    return Ops.hasMatcher();
  }

  /// \brief Return this matcher as a \c Matcher<T>.
  ///
  /// Handles the different types (Single, Polymorphic) accordingly.
  /// Asserts that \c hasTypedMatcher<T>() is true.
  template <class T>
  ast_matchers::internal::Matcher<T> getTypedMatcher() const {
    TypedMatcherOps<T> Ops;
    Value->makeTypedMatcher(Ops);
    assert(Ops.hasMatcher() && "hasTypedMatcher<T>() == false");
    return Ops.matcher();
  }

  /// \brief String representation of the type of the value.
  ///
  /// If the underlying matcher is a polymorphic one, the string will show all
  /// the types.
  std::string getTypeAsString() const;

private:
  explicit VariantMatcher(Payload *Value) : Value(Value) {}

  class SinglePayload;
  class PolymorphicPayload;
  class VariadicOpPayload;

  template <typename T>
  class TypedMatcherOps : public MatcherOps {
  public:
    typedef ast_matchers::internal::Matcher<T> MatcherT;

    virtual bool canConstructFrom(const DynTypedMatcher &Matcher) const {
      return Matcher.canConvertTo<T>();
    }

    virtual void constructFrom(const DynTypedMatcher& Matcher) {
      Out.reset(new MatcherT(Matcher.convertTo<T>()));
    }

    virtual void constructVariadicOperator(
        ast_matchers::internal::VariadicOperatorFunction Func,
        ArrayRef<VariantMatcher> InnerMatchers) {
      std::vector<DynTypedMatcher> DynMatchers;
      for (size_t i = 0, e = InnerMatchers.size(); i != e; ++i) {
        // Abort if any of the inner matchers can't be converted to
        // Matcher<T>.
        if (!InnerMatchers[i].hasTypedMatcher<T>()) {
          return;
        }
        DynMatchers.push_back(InnerMatchers[i].getTypedMatcher<T>());
      }
      Out.reset(new MatcherT(
          new ast_matchers::internal::VariadicOperatorMatcherInterface<T>(
              Func, DynMatchers)));
    }

    bool hasMatcher() const { return Out.get() != NULL; }
    const MatcherT &matcher() const { return *Out; }

  private:
    OwningPtr<MatcherT> Out;
  };

  IntrusiveRefCntPtr<const Payload> Value;
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
///  - \c VariantMatcher (\c DynTypedMatcher / \c Matcher<T>)
class VariantValue {
public:
  VariantValue() : Type(VT_Nothing) {}

  VariantValue(const VariantValue &Other);
  ~VariantValue();
  VariantValue &operator=(const VariantValue &Other);

  /// \brief Specific constructors for each supported type.
  VariantValue(unsigned Unsigned);
  VariantValue(const std::string &String);
  VariantValue(const VariantMatcher &Matchers);

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
  const VariantMatcher &getMatcher() const;
  void setMatcher(const VariantMatcher &Matcher);

  /// \brief String representation of the type of the value.
  std::string getTypeAsString() const;

private:
  void reset();

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
    VariantMatcher *Matcher;
  };

  ValueType Type;
  AllValues Value;
};

} // end namespace dynamic
} // end namespace ast_matchers
} // end namespace clang

#endif  // LLVM_CLANG_AST_MATCHERS_DYNAMIC_VARIANT_VALUE_H
