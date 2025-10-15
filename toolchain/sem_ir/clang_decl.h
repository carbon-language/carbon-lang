// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_CLANG_DECL_H_
#define CARBON_TOOLCHAIN_SEM_IR_CLANG_DECL_H_

#include <concepts>

#include "clang/AST/Decl.h"
#include "common/hashtable_key_context.h"
#include "common/ostream.h"
#include "toolchain/base/canonical_value_store.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

// A key describing a Clang declaration that can be looked up in the value
// store. This is a `clang::Decl*` pointing to a canonical declaration, plus any
// other information that affects the mapping into Carbon. Currently this
// includes the number of imported parameters for a function with default
// arguments.
//
// A canonical declaration pointer is used so that we can perform direct address
// comparisons and hash this structure based on its contents.
struct ClangDeclKey : public Printable<ClangDeclKey> {
  // For declaration classes that are unrelated to FunctionDecl, no parameter
  // count is expected.
  template <typename DeclT>
    requires(std::derived_from<DeclT, clang::Decl> &&
             !std::derived_from<clang::FunctionDecl, DeclT> &&
             !std::derived_from<DeclT, clang::FunctionDecl>)
  explicit ClangDeclKey(DeclT* decl) : ClangDeclKey(decl, -1, UncheckedTag()) {}

  // For declaration classes that are derived from FunctionDecl, a parameter
  // count is required.
  static auto ForFunctionDecl(clang::FunctionDecl* decl, int num_params)
      -> ClangDeclKey {
    return ClangDeclKey(decl, num_params, UncheckedTag());
  }

  // Factory function for clang declaration that is dynamically known to not be
  // a function declaration.
  static auto ForNonFunctionDecl(clang::Decl* decl) -> ClangDeclKey {
    CARBON_CHECK(!isa<clang::FunctionDecl>(decl));
    return ClangDeclKey(decl, -1, UncheckedTag());
  }

  auto Print(llvm::raw_ostream& out) const -> void;

  auto operator==(const ClangDeclKey& rhs) const -> bool {
    return decl == rhs.decl && num_params == rhs.num_params;
  }

  // Hashing for ClangDecl. See common/hashing.h.
  friend auto CarbonHashValue(const ClangDeclKey& value, uint64_t seed)
      -> HashCode {
    // Manual hashing support is required because this type has tail padding in
    // 64-bit compilations.
    return HashValue(std::pair{value.decl, value.num_params}, seed);
  }

  // The Clang declaration pointing to the Clang AST.
  // TODO: Ensure we can easily serialize/deserialize this. Consider
  // `clang::LazyDeclPtr`.
  clang::Decl* decl = nullptr;

  // The number of parameters to import for a function declaration. Excludes the
  // implicit object parameter, if there is one. Always -1 for a non-function
  // declaration.
  int32_t num_params = -1;

 private:
  struct UncheckedTag {
    explicit UncheckedTag() = default;
  };
  ClangDeclKey(clang::Decl* decl, int num_params, UncheckedTag /*_*/)
      : decl(decl->getCanonicalDecl()), num_params(num_params) {}
};

// A Clang declaration mapped to a Carbon instruction.
//
// Instances of this type are managed by a `ClangDeclStore`, which ensures that
// a single `ClangDecl` exists for each `ClangDeclKey` used.
struct ClangDecl : public Printable<ClangDecl> {
  auto Print(llvm::raw_ostream& out) const -> void;

  // The key by which this declaration can be looked up.
  ClangDeclKey key;

  // The instruction the Clang declaration is mapped to.
  InstId inst_id;

  auto GetAsKey() const -> ClangDeclKey { return key; }
};

// The ID of a `ClangDecl`.
//
// These IDs are importantly distinct from the `inst_id` associated with each
// declaration. These form a dense range of IDs that is used to reference the
// AST node pointers without storing those pointers directly into SemIR and
// needing space to hold a full pointer. We can't avoid having these IDs without
// embedding pointers directly into the storage of SemIR as part of an
// instruction.
struct ClangDeclId : public IdBase<ClangDeclId> {
  static constexpr llvm::StringLiteral Label = "clang_decl_id";

  using IdBase::IdBase;
};

// Use the AST node pointer directly when doing `Lookup` to find an ID.
using ClangDeclStore =
    CanonicalValueStore<ClangDeclId, ClangDeclKey, ClangDecl>;

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_CLANG_DECL_H_
