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

// Information about how to form the Carbon function signature from the Clang
// function declaration.
struct Signature : public Printable<Signature> {
  // A passing mode for a parameter in a C++ function signature.
  enum class PassingMode : int8_t {
    Copy,
    Move,
  };

  enum Kind : int8_t {
    // A normal function signature: each C++ parameter maps into a Carbon
    // parameter.
    Normal,
    // A function signature taking a tuple pattern that contains the C++
    // parameters. This is used when importing a constructor that is used for
    // list initialization from a Carbon tuple.
    TuplePattern,
  };
  // The kind of function signature being imported.
  Kind kind = Normal;
  // The number of parameters to import. This can be less than the number of
  // parameters in the Clang declaration if the Clang declaration has default
  // arguments. Excludes the implicit object parameter, if there is one.
  int32_t num_params = -1;

  // The passing mode for each parameter.
  llvm::SmallVector<PassingMode, 4> passing_modes;

  // Returns the passing mode for the i-th parameter.
  // TODO: Require that `passing_modes.size()` is always `num_params`.
  auto GetPassingMode(int32_t i) const -> PassingMode {
    return i < static_cast<int32_t>(passing_modes.size()) ? passing_modes[i]
                                                          : PassingMode::Copy;
  }

  auto Print(llvm::raw_ostream& out) const -> void;

  auto operator==(const Signature& rhs) const -> bool {
    return kind == rhs.kind && num_params == rhs.num_params &&
           passing_modes == rhs.passing_modes;
  }

  // Hashing for Signature.
  friend auto CarbonHashValue(const Signature& value, uint64_t seed)
      -> HashCode {
    HashCode code = HashValue(std::tuple{value.kind, value.num_params}, seed);
    for (auto mode : value.passing_modes) {
      code = HashValue(static_cast<int8_t>(mode), static_cast<uint64_t>(code));
    }
    return code;
  }
};

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
  explicit ClangDeclKey(DeclT* decl)
      : ClangDeclKey(decl, SignatureId::None, UncheckedTag()) {}

  // For declaration classes that are derived from FunctionDecl, a parameter
  // count is required.
  static auto ForFunctionDecl(clang::FunctionDecl* decl, SignatureId signature_id)
      -> ClangDeclKey {
    return ClangDeclKey(decl, signature_id, UncheckedTag());
  }

  // Factory function for clang declaration that is dynamically known to not be
  // a function declaration.
  static auto ForNonFunctionDecl(clang::Decl* decl) -> ClangDeclKey {
    CARBON_CHECK(!isa<clang::FunctionDecl>(decl));
    return ClangDeclKey(decl, SignatureId::None, UncheckedTag());
  }

  auto Print(llvm::raw_ostream& out) const -> void;

  auto operator==(const ClangDeclKey& rhs) const -> bool {
    return decl == rhs.decl && signature_id == rhs.signature_id;
  }

  // Hashing for ClangDecl. See common/hashing.h.
  friend auto CarbonHashValue(const ClangDeclKey& value, uint64_t seed)
      -> HashCode {
    return HashValue(std::tuple{value.decl, value.signature_id}, seed);
  }

  // The Clang declaration pointing to the Clang AST.
  // TODO: Ensure we can easily serialize/deserialize this. Consider
  // `clang::LazyDeclPtr`.
  clang::Decl* decl;

  // The parameters to import for a function declaration. Otherwise
  // SignatureId::None.
  SignatureId signature_id;

 private:
  struct UncheckedTag {
    explicit UncheckedTag() = default;
  };
  ClangDeclKey(clang::Decl* decl, SignatureId signature_id, UncheckedTag /*_*/)
      : decl(decl->getCanonicalDecl()), signature_id(signature_id) {}
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

// Use the AST node pointer directly when doing `Lookup` to find an ID.
using ClangDeclStore =
    CanonicalValueStore<ClangDeclId, ClangDeclKey, Tag<CheckIRId>, ClangDecl>;

// A Signature mapped to an ID.
using SignatureStore =
    CanonicalValueStore<SignatureId, Signature, Tag<CheckIRId>, Signature>;

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_CLANG_DECL_H_
