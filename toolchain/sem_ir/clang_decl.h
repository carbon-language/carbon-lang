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
struct ClangDeclSignature : public Printable<ClangDeclSignature> {
  // A passing mode for a parameter in a C++ function signature.
  enum class PassingMode : int8_t {
    // This parameter is passed by Carbon value. This is used for a C++
    // non-reference parameter that would be copied at the call site, and for a
    // C++ const reference parameter (either lvalue or rvalue reference).
    ByValue,
    // This parameter is passed as a Carbon var parameter. This is used for a
    // C++ non-reference parameter that would be moved or constructed in-place
    // at the call site, or for a C++ non-const rvalue reference parameter.
    ByVar,
    // This parameter is passed as a Carbon ref parameter. This is used for a
    // C++ non-const lvalue reference parameter.
    ByRef,
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
  // arguments. Excludes the (implicit or explicit) object parameter, if there
  // is one.
  // TODO: Remove in favor of `passing_modes`.
  int32_t num_params = -1;

  // The passing mode for the (implicit or explicit) object parameter, if there
  // is one. Otherwise PassingMode::ByRef.
  PassingMode self_passing_mode = PassingMode::ByRef;

  // The passing mode for each parameter. Excludes the (implicit or explicit)
  // object parameter, if there is one. This must be the same size as
  // `num_params`.
  // TODO: Generalize this to be parameter info, not just passing mode.
  llvm::SmallVector<PassingMode, 4> passing_modes;

  // Convenience function to make a fixed signature.
  static auto Make(
      std::initializer_list<SemIR::ClangDeclSignature::PassingMode> modes,
      Kind kind = Normal, PassingMode self_passing_mode = PassingMode::ByRef)
      -> ClangDeclSignature {
    ClangDeclSignature signature;
    signature.kind = kind;
    signature.num_params = static_cast<int32_t>(modes.size());
    signature.self_passing_mode = self_passing_mode;
    signature.passing_modes.assign(modes.begin(), modes.end());
    return signature;
  }

  // Returns the passing mode for the i-th parameter.
  auto GetPassingMode(int32_t i) const -> PassingMode {
    return i < static_cast<int32_t>(passing_modes.size()) ? passing_modes[i]
                                                          : PassingMode::ByVar;
  }

  auto Print(llvm::raw_ostream& out) const -> void;

  auto operator==(const ClangDeclSignature& rhs) const -> bool {
    return kind == rhs.kind && num_params == rhs.num_params &&
           passing_modes == rhs.passing_modes &&
           self_passing_mode == rhs.self_passing_mode;
  }

  // Hashing for ClangDeclSignature.
  friend auto CarbonHashValue(const ClangDeclSignature& value, uint64_t seed)
      -> HashCode {
    HashCode code =
        HashValue(std::tuple{value.kind, value.num_params,
                             static_cast<int8_t>(value.self_passing_mode)},
                  seed);
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
      : ClangDeclKey(decl, ClangDeclSignatureId::None, UncheckedTag()) {}

  // For declaration classes that are derived from FunctionDecl, a parameter
  // count is required.
  static auto ForFunctionDecl(clang::FunctionDecl* decl,
                              ClangDeclSignatureId signature_id)
      -> ClangDeclKey {
    return ClangDeclKey(decl, signature_id, UncheckedTag());
  }

  // Factory function for clang declaration that is dynamically known to not be
  // a function declaration.
  static auto ForNonFunctionDecl(clang::Decl* decl) -> ClangDeclKey {
    CARBON_CHECK(!isa<clang::FunctionDecl>(decl));
    return ClangDeclKey(decl, ClangDeclSignatureId::None, UncheckedTag());
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
  // ClangDeclSignatureId::None.
  ClangDeclSignatureId signature_id;

 private:
  struct UncheckedTag {
    explicit UncheckedTag() = default;
  };
  ClangDeclKey(clang::Decl* decl, ClangDeclSignatureId signature_id,
               UncheckedTag /*_*/)
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

  // True if this declaration originated from C++. False if this declaration was
  // created by exporting some Carbon declaration to C++.
  bool is_imported = false;

  auto GetAsKey() const -> ClangDeclKey { return key; }
};

// Canonical storage for `ClangDecl`s. Provides bidirectional mapping
// between `ClangDeclId`s and `InstId`s.
class ClangDeclStore {
 public:
  explicit ClangDeclStore(CheckIRId check_ir_id);

  // Adds a `ClangDecl`, returning an ID to reference it.
  auto Add(ClangDecl value) -> ClangDeclId;

  // Same as `Add`, but for `VarStorage` that maps to a `clang::VarDecl`.
  //
  // When looking up via `InstId`, the pattern's `InstId` must be used
  // instead of the `InstId` corresponding to the `VarStorage`. Note however
  // that the `value.inst_id` is still the `VarStorage` `InstId`.
  //
  // The pattern's `InstId` is used because it provides a more stable
  // lookup key than the `VarStorage` `InstId`. For example, a call to
  // `Convert` may cause a new `VarStorage` instruction to be created,
  // but the pattern will remain the same.
  auto AddVar(ClangDecl value, InstId pattern_id) -> ClangDeclId;

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

// A ClangDeclSignature mapped to an ID.
using ClangDeclSignatureStore =
    CanonicalValueStore<ClangDeclSignatureId, ClangDeclSignature,
                        Tag<CheckIRId>, ClangDeclSignature>;

}  // namespace Carbon::SemIR

namespace Carbon {
extern template class CanonicalValueStore<
    SemIR::ClangDeclId, SemIR::ClangDeclKey, Tag<SemIR::CheckIRId>,
    SemIR::ClangDecl>;
extern template class ValueStore<SemIR::ClangDeclId, SemIR::ClangDecl,
                                 Tag<SemIR::CheckIRId>>;
extern template class CanonicalValueStore<
    SemIR::ClangDeclSignatureId, SemIR::ClangDeclSignature,
    Tag<SemIR::CheckIRId>, SemIR::ClangDeclSignature>;
extern template class ValueStore<SemIR::ClangDeclSignatureId,
                                 SemIR::ClangDeclSignature,
                                 Tag<SemIR::CheckIRId>>;
}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_SEM_IR_CLANG_DECL_H_
