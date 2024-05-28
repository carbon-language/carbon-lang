// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_HASHTABLE_KEY_CONTEXT_H_
#define CARBON_COMMON_HASHTABLE_KEY_CONTEXT_H_

#include "common/hashing.h"

namespace Carbon {

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
  template <typename KeyT>
  auto HashKey(const KeyT& key, uint64_t seed) const -> HashCode {
    return HashValue(key, seed);
  }

  template <typename LHSKeyT, typename RHSKeyT>
  auto KeyEq(const LHSKeyT& lhs_key, const RHSKeyT& rhs_key) const -> bool {
    return lhs_key == rhs_key;
  }
};

}  // namespace Carbon

#endif  // CARBON_COMMON_HASHTABLE_KEY_CONTEXT_H_
