// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_HASHTABLE_KEY_CONTEXT_H_
#define CARBON_COMMON_HASHTABLE_KEY_CONTEXT_H_

#include <concepts>

#include "common/hashing.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"

namespace Carbon {

// The equality comparison used by the the hashtable key contexts in this file,
// and suitable for using in other hashtable key contexts.
//
// This provides a hashtable-specific extension point to implement equality
// comparison within a hashtable key context. By default, it will use
// `operator==` on the LHS and RHS operands. However, types can provide a
// dedicated customization point by implementing a free function that can be
// found by ADL for your type called `CarbonHashtableEq` with the following
// signature:
//
// ```cpp
// auto CarbonHashtableEq(const YourType& lhs, const YourType& rhs) -> bool;
// ```
//
// Any such overload will be able to override the default we provide for types
// that can compare with `==`.
//
// This library also provides any customization points for LLVM or standard
// library types either lacking `operator==` or where that operator is not
// suitable for hashtables. For example, `llvm::APInt` and `llvm::APFloat` have
// custom equality comparisons provided through this extension point.
template <typename LeftT, typename RightT>
auto HashtableEq(const LeftT& lhs, const RightT& rhs) -> bool;

// Customizable context for keys in hashtables.
//
// This type or customizations matching its API are used with the data
// structures in `map.h` and `set.h`. By providing a custom version of the
// `KeyContext` type parameter to those data structures, users can provide
// either stateless or stateful customization of the two core hashtable key
// operations: hashing and comparison.
//
// The default for hashing uses Carbon's `hashing.h`. Customizations must still
// return a `HashCode` as defined there, and it needs to have the same core
// properties of hashes produced by the `hashing.h` infrastructure.
//
// The default for comparison is `operator==`. The `KeyEq` method is always
// called with a key *stored in the hashtable* as the second or "RHS" parameter.
// This is to allow simplifying the set of overloads needed for heterogeneous
// contexts: only the first, LHS, parameter needs to support different lookup
// key types.
//
// Custom KeyContext types should have the the same API as the default type.
// They can choose to use templates to support heterogeneous key types or not as
// appropriate. The default context can also be used as a base class with only
// one or the other APIs customized.
//
// An important consideration is how the key context is constructed. When the
// key context can be default constructed, hashtable APIs trafficking in keys
// will have overloads that provide a default constructed key context. When the
// context is *not* default constructible, every API that accepts a key will
// also require a context argument to be called, and that argument will be used
// throughout that operation. The intent is to allow callers to provide stateful
// contexts to each API where it would be needed, while managing that state
// outside the hashtable. Often the needed state is trivially part of the
// caller's existing state and needn't be stored separately.
//
// Example for a stateful, customized key context for interned strings:
// ```cpp
// class InternedStringIndexKeyContext {
//  public:
//   InternedStringIndexKeyContext(
//       llvm::ArrayRef<llvm::StringRef> interned_strings)
//       : interned_strings_(interned_strings) {}
//
//   auto HashKey(llvm::StringRef s, uint64_t seed) const -> HashCode {
//     return HashValue(s);
//   }
//   auto HashKey(int index_key, uint64_t seed) const -> HashCode {
//     return HashKey(interned_strings_[index_key]);
//   }
//
//   auto KeyEq(llvm::StringRef lhs, int rhs_index) const -> bool {
//     return lhs == interned_strings_[rhs_index];
//   }
//   auto KeyEq(int lhs_index, int rhs_index) const -> bool {
//     return KeyEq(interned_strings_[lhs_index], rhs_index);
//   }
//
//  private:
//   llvm::ArrayRef<llvm::StringRef> interned_strings_;
// };
// ```
struct DefaultKeyContext {
  template <typename AnyKeyT>
  auto HashKey(const AnyKeyT& key, uint64_t seed) const -> HashCode;

  template <typename AnyKeyT, typename TableKeyT>
  auto KeyEq(const AnyKeyT& lhs_key, const TableKeyT& rhs_key) const -> bool;
};

// A CRTP mixin for a custom key context type that first translates keys to a
// different type, possibly using some state.
//
// Derived types should publicly inherit from this mixin and define overloads of
// the `TranslateKey` method as indicated below in its comment.
template <typename DerivedT>
class TranslatingKeyContext {
 public:
  // Derived types should provide one or more overloads that hide this function
  // and perform translation for the key types which need it.
  //
  // For any key type, the context will check if there exists a callable
  // `TranslateKey` function on the derived type. If so, that function will be
  // called and the result used for hashing or comparison. If not, the key will
  // be used directly. The derived type doesn't need to and shouldn't provide a
  // default no-op overload. Instead, for any types that need no translation, it
  // should ensure no overload is viable.
  //
  // Note that this function should be *hidden* by the derived overloads. It is
  // provided here to help detect typos or misspellings or cases where no
  // overload is provided at all.
  template <typename TranslateKeyT>
  auto TranslateKey(const TranslateKeyT& /*key*/) const -> int {
    // A static_assert that will fail on any actual instantiation (it can't be
    // instantiated with a void type). We have to make this dependent as
    // Clang-16 will fail to compile even when the definition is never
    // instantiated otherwise.
    static_assert(
        std::same_as<TranslateKeyT, void>,
        "No `TranslateKey` overload was provided by the derived type!");
  }

  template <typename AnyKeyT>
  auto HashKey(const AnyKeyT& key, uint64_t seed) const -> HashCode;

  template <typename AnyKeyT, typename TableKeyT>
  auto KeyEq(const AnyKeyT& lhs_key, const TableKeyT& rhs_key) const -> bool;
};

////////////////////////////////////////////////////////////////////////////////
//
// Only implementation details below this point.
//
////////////////////////////////////////////////////////////////////////////////

namespace InternalHashtableEqDispatch {

inline auto CarbonHashtableEq(const llvm::APInt& lhs, const llvm::APInt& rhs)
    -> bool {
  return lhs.getBitWidth() == rhs.getBitWidth() && lhs == rhs;
}

inline auto CarbonHashtableEq(const llvm::APFloat& lhs,
                              const llvm::APFloat& rhs) -> bool {
  return lhs.bitwiseIsEqual(rhs);
}

template <typename LeftT, typename RightT>
inline auto CarbonHashtableEq(const LeftT& lhs, const RightT& rhs) -> bool
  requires(requires {
    { lhs == rhs } -> std::convertible_to<bool>;
  })
{
  return lhs == rhs;
}

template <typename LeftT, typename RightT>
inline auto DispatchImpl(const LeftT& lhs, const RightT& rhs) -> bool {
  // This unqualified call will find both the overloads in our internal
  // namespace above and ADL-found functions within an associated namespace for
  // either `LeftT` or `RightT`.
  return CarbonHashtableEq(lhs, rhs);
}

}  // namespace InternalHashtableEqDispatch

template <typename LeftT, typename RightT>
inline auto HashtableEq(const LeftT& lhs, const RightT& rhs) -> bool {
  return InternalHashtableEqDispatch::DispatchImpl(lhs, rhs);
}

template <typename AnyKeyT>
auto DefaultKeyContext::HashKey(const AnyKeyT& key, uint64_t seed) const
    -> HashCode {
  return HashValue(key, seed);
}

template <typename AnyKeyT, typename TableKeyT>
auto DefaultKeyContext::KeyEq(const AnyKeyT& lhs_key,
                              const TableKeyT& rhs_key) const -> bool {
  return HashtableEq(lhs_key, rhs_key);
}

template <typename DerivedT>
template <typename AnyKeyT>
auto TranslatingKeyContext<DerivedT>::HashKey(const AnyKeyT& key,
                                              uint64_t seed) const -> HashCode {
  const DerivedT& self = *static_cast<const DerivedT*>(this);
  if constexpr (requires { self.TranslateKey(key); }) {
    return HashValue(self.TranslateKey(key), seed);
  } else {
    return HashValue(key, seed);
  }
}

template <typename DerivedT>
template <typename AnyKeyT, typename TableKeyT>
auto TranslatingKeyContext<DerivedT>::KeyEq(const AnyKeyT& lhs_key,
                                            const TableKeyT& rhs_key) const
    -> bool {
  const DerivedT& self = *static_cast<const DerivedT*>(this);
  // Because we don't want to make no-op calls and potentially struggle with
  // temporary lifetimes at runtime we have to fully expand the 4 states.
  constexpr bool TranslateLHS = requires { self.TranslateKey(lhs_key); };
  constexpr bool TranslateRHS = requires { self.TranslateKey(rhs_key); };
  if constexpr (TranslateLHS && TranslateRHS) {
    return HashtableEq(self.TranslateKey(lhs_key), self.TranslateKey(rhs_key));
  } else if constexpr (TranslateLHS) {
    return HashtableEq(self.TranslateKey(lhs_key), rhs_key);
  } else if constexpr (TranslateRHS) {
    return HashtableEq(lhs_key, self.TranslateKey(rhs_key));
  } else {
    return HashtableEq(lhs_key, rhs_key);
  }
}

}  // namespace Carbon

#endif  // CARBON_COMMON_HASHTABLE_KEY_CONTEXT_H_
