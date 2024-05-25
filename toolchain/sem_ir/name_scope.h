// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_NAME_SCOPE_H_
#define CARBON_TOOLCHAIN_SEM_IR_NAME_SCOPE_H_

#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

struct NameScope : Printable<NameScope> {
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "{inst: " << inst_id << ", enclosing_scope: " << enclosing_scope_id
        << ", has_error: " << (has_error ? "true" : "false");

    out << ", extended_scopes: [";
    llvm::ListSeparator scope_sep;
    for (auto id : extended_scopes) {
      out << scope_sep << id;
    }
    out << "]";

    out << ", names: {";
    // Sort name keys to get stable output.
    llvm::SmallVector<NameId> keys;
    for (auto [key, _] : names) {
      keys.push_back(key);
    }
    llvm::sort(keys,
               [](NameId lhs, NameId rhs) { return lhs.index < rhs.index; });
    llvm::ListSeparator key_sep;
    for (auto key : keys) {
      out << key_sep << key << ": " << names.find(key)->second;
    }
    out << "}";

    out << "}";
  }

  // Names in the scope.
  llvm::DenseMap<NameId, InstId> names = llvm::DenseMap<NameId, InstId>();

  // Scopes extended by this scope.
  //
  // TODO: A `NameScopeId` is currently insufficient to describe an extended
  // scope in general. For example:
  //
  //   class A(T:! type) {
  //     extend base: B(T*);
  //   }
  //
  // needs to describe the `T*` argument.
  //
  // Small vector size is set to 1: we expect that there will rarely be more
  // than a single extended scope. Currently the only kind of extended scope is
  // a base class, and there can be only one of those per scope.
  // TODO: Revisit this once we have more kinds of extended scope and data.
  // TODO: Consider using something like `TinyPtrVector` for this.
  llvm::SmallVector<NameScopeId, 1> extended_scopes;

  // The instruction which owns the scope.
  InstId inst_id;

  // When the scope is a namespace, the name. Otherwise, invalid.
  NameId name_id;

  // The scope enclosing this one.
  NameScopeId enclosing_scope_id;

  // Whether we have diagnosed an error in a construct that would have added
  // names to this scope. For example, this can happen if an `import` failed or
  // an `extend` declaration was ill-formed. If true, the `names` map is assumed
  // to be missing names as a result of the error, and no further errors are
  // produced for lookup failures in this scope.
  bool has_error = false;

  // True if this is a closed namespace created by importing a package.
  bool is_closed_import = false;

  // Imported IR scopes that compose this namespace. This will be empty for
  // scopes that correspond to the current package.
  llvm::SmallVector<std::pair<SemIR::ImportIRId, SemIR::NameScopeId>, 0>
      import_ir_scopes;
};

// Provides a ValueStore wrapper for an API specific to name scopes.
class NameScopeStore {
 public:
  // Adds a name scope, returning an ID to reference it.
  auto Add(InstId inst_id, NameId name_id, NameScopeId enclosing_scope_id)
      -> NameScopeId {
    return values_.Add({.inst_id = inst_id,
                        .name_id = name_id,
                        .enclosing_scope_id = enclosing_scope_id});
  }

  // Returns the requested name scope.
  auto Get(NameScopeId scope_id) -> NameScope& { return values_.Get(scope_id); }

  // Returns the requested name scope.
  auto Get(NameScopeId scope_id) const -> const NameScope& {
    return values_.Get(scope_id);
  }

  // Returns the instruction owning the requested name scope, or an invalid
  // instruction if the scope is either invalid or has no associated
  // instruction.
  auto GetInstIdIfValid(NameScopeId scope_id) const -> InstId {
    if (!scope_id.is_valid()) {
      return InstId::Invalid;
    }
    return Get(scope_id).inst_id;
  }

  auto OutputYaml() const -> Yaml::OutputMapping {
    return values_.OutputYaml();
  }

 private:
  ValueStore<NameScopeId> values_;
};

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_NAME_SCOPE_H_
