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
  // Information about how to form the Carbon function signature from the Clang
  // function declaration.
  struct Signature {
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

    friend auto operator==(const Signature& lhs, const Signature& rhs)
        -> bool = default;
  };

  // For declaration classes that are unrelated to FunctionDecl, no parameter
  // count is expected.
  template <typename DeclT>
    requires(std::derived_from<DeclT, clang::Decl> &&
             !std::derived_from<clang::FunctionDecl, DeclT> &&
             !std::derived_from<DeclT, clang::FunctionDecl>)
  explicit ClangDeclKey(DeclT* decl)
      : ClangDeclKey(decl, Signature{}, UncheckedTag()) {}

  // For declaration classes that are derived from FunctionDecl, a parameter
  // count is required.
  static auto ForFunctionDecl(clang::FunctionDecl* decl, Signature signature)
      -> ClangDeclKey {
    return ClangDeclKey(decl, signature, UncheckedTag());
  }

  // Factory function for clang declaration that is dynamically known to not be
  // a function declaration.
  static auto ForNonFunctionDecl(clang::Decl* decl) -> ClangDeclKey {
    CARBON_CHECK(!isa<clang::FunctionDecl>(decl));
    return ClangDeclKey(decl, Signature{}, UncheckedTag());
  }

  auto Print(llvm::raw_ostream& out) const -> void;

  auto operator==(const ClangDeclKey& rhs) const -> bool {
    return decl == rhs.decl && signature == rhs.signature;
  }

  // Hashing for ClangDecl. See common/hashing.h.
  friend auto CarbonHashValue(const ClangDeclKey& value, uint64_t seed)
      -> HashCode {
    // Manual hashing support is required because `Signature` has padding.
    return HashValue(std::tuple{value.decl, value.signature.num_params,
                                value.signature.kind},
                     seed);
  }

  // The Clang declaration pointing to the Clang AST.
  // TODO: Ensure we can easily serialize/deserialize this. Consider
  // `clang::LazyDeclPtr`.
  clang::Decl* decl;

  // The parameters to import for a function declaration. Otherwise a
  // default-constructed value.
  Signature signature;

 private:
  struct UncheckedTag {
    explicit UncheckedTag() = default;
  };
  ClangDeclKey(clang::Decl* decl, Signature signature, UncheckedTag /*_*/)
      : decl(decl->getCanonicalDecl()), signature(signature) {}
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

// Canonical storage for `ClangDecl`s. Provides bidirectional mapping
// between `ClangDeclId`s and `InstId`s.
class ClangDeclStore {
 public:
  explicit ClangDeclStore(CheckIRId check_ir_id);

  // Adds a `ClangDecl`, returning an ID to reference it.
  auto Add(ClangDecl value) -> ClangDeclId;

  // Looks up a `ClangDecl` by `ClangDeclId`.
  auto Get(ClangDeclId id) const -> const ClangDecl& { return values_.Get(id); }

  // Looks up a `ClangDeclId` by `ClangDeclKey`.
  auto Lookup(ClangDeclKey key) const -> ClangDeclId;

  // Looks up a `ClangDeclId` by `InstId`.
  auto Lookup(InstId inst_id) const -> ClangDeclId;

  auto OutputYaml() const -> Yaml::OutputMapping;

  auto CollectMemUsage(MemUsage& mem_usage, llvm::StringRef label) const
      -> void;

 private:
  // Canonical storage for `ClangDecl`s. Allows mapping from a
  // `ClangDeclId` to an `InstId`.
  CanonicalValueStore<ClangDeclId, ClangDeclKey, Tag<CheckIRId>, ClangDecl>
      values_;

  // Map from `InstId` to `ClangDeclId`.
  Map<InstId, ClangDeclId> inst_id_to_clang_decl_id_;
};

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_CLANG_DECL_H_
