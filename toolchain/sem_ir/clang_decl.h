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
// Using custom hashing since the declaration is keyed by the `decl` member for
// lookup.
// TODO: Avoid custom hashing by either having the data structure support keying
// or create a dedicated mapping. See
// https://discord.com/channels/655572317891461132/768530752592805919/1384999468293947537
struct ClangDecl : public Printable<ClangDecl> {
  auto Print(llvm::raw_ostream& out) const -> void;

  friend auto CarbonHashtableEq(const ClangDecl& lhs, const ClangDecl& rhs)
      -> bool {
    return HashtableEq(lhs.decl, rhs.decl);
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
  InstId inst_id;
};

// The ID of a `ClangDecl`.
struct ClangDeclId : public IdBase<ClangDeclId> {
  static constexpr llvm::StringLiteral Label = "clang_decl_id";

  using ValueType = ClangDecl;

  using IdBase::IdBase;
};

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_CLANG_DECL_H_
