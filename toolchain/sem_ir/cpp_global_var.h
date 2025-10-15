// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_CPP_GLOBAL_VAR_H_
#define CARBON_TOOLCHAIN_SEM_IR_CPP_GLOBAL_VAR_H_

#include "common/hashing.h"
#include "common/ostream.h"
#include "toolchain/sem_ir/clang_decl.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

// A key describing a C++ global variable imported into Carbon, identified by
// its entity name.
struct CppGlobalVarKey : public Printable<CppGlobalVarKey> {
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "{entity_name_id: " << entity_name_id << "}";
  }

  // TODO: Use default when `Printable` supports it.
  friend auto operator==(const CppGlobalVarKey& lhs, const CppGlobalVarKey& rhs)
      -> bool {
    return lhs.entity_name_id == rhs.entity_name_id;
  }

  // The name of the variable.
  EntityNameId entity_name_id;
};

// A C++ global variable imported into Carbon. This is used to map the entity
// name to the Clang declaration so we can use Clang mangling.
struct CppGlobalVar : public Printable<CppGlobalVar> {
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "{key: " << key << ", clang_decl_id: " << clang_decl_id << "}";
  }

  // The key by which this variable can be looked up.
  CppGlobalVarKey key;

  // The Clang declaration for this variable, if any.
  // This is ignored for equality and hashing, since it's always unique for a
  // given key, in order to store it in `CanonicalValueStore` and allow lookup
  // by `CppGlobalVarKey`.
  ClangDeclId clang_decl_id;
};

// Use the name of a C++ global variable when doing `Lookup` to find an ID.
using CppGlobalVarStore =
    CanonicalValueStore<CppGlobalVarId, CppGlobalVarKey, CppGlobalVar,
                        [](const CppGlobalVar& value)
                            -> const CppGlobalVarKey& { return value.key; }>;

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_CPP_GLOBAL_VAR_H_
