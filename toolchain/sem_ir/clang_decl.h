// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_CLANG_DECL_H_
#define CARBON_TOOLCHAIN_SEM_IR_CLANG_DECL_H_

#include "common/hashtable_key_context.h"
#include "common/ostream.h"
#include "toolchain/sem_ir/ids.h"

// NOLINTNEXTLINE(readability-identifier-naming)
namespace clang {

// Forward declare indexed types, for integration with ValueStore.
class Decl;

}  // namespace clang

namespace Carbon::SemIR {

// A Clang declaration mapped to a Carbon instruction.
//
// Note that Clang's AST uses address-identity for nodes, which means the
// pointer is the canonical way to represent a specific AST node and is expected
// to be sufficient for comparison, hashing, etc.
struct ClangDecl : public Printable<ClangDecl> {
  auto Print(llvm::raw_ostream& out) const -> void;

  // Equality comparison uses the address-identity property of the Clang AST and
  // just compares the `decl` pointers. We can disregard `inst_id` because `ClangDecl`s
  // are only used as entries in `SemIR::clang_decls`, which canonicalizes them to
  // ensure that the same `decl` can't be associated with more than one `inst_id`.
  auto operator==(const ClangDecl& rhs) const -> bool {
    return decl == rhs.decl;
  }
  // Support direct comparison with the Clang AST node pointer.
  auto operator==(const clang::Decl* rhs_decl) const -> bool {
    return decl == rhs_decl;
  }

  // Hashing for ClangDecl. See common/hashing.h.
  friend auto CarbonHashValue(const ClangDecl& value, uint64_t seed)
      -> HashCode {
    return HashValue(value.decl, seed);
  }

  // The Clang declaration pointing to the Clang AST.
  // TODO: Ensure we can easily serialize/deserialize this. Consider
  // `clang::LazyDeclPtr`.
  clang::Decl* decl = nullptr;

  // The instruction the Clang declaration is mapped to.
  //
  // This is stored along side the `decl` pointer to avoid having to lookup both
  // the pointer and the instruction ID in two separate areas of storage.
  InstId inst_id;
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

  using ValueType = ClangDecl;

  // Use the AST node pointer directly when doing `Lookup` to find an ID.
  using KeyType = clang::Decl*;

  using IdBase::IdBase;
};

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_CLANG_DECL_H_
