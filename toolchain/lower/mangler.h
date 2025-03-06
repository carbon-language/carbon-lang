// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LOWER_MANGLER_H_
#define CARBON_TOOLCHAIN_LOWER_MANGLER_H_

#include <string>

#include "clang/AST/Mangle.h"
#include "toolchain/lower/file_context.h"
#include "toolchain/sem_ir/constant.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst_fingerprinter.h"

namespace Carbon::Lower {

// A class for producing mangled (deterministically unique, at least partially
// human readable) names for externally referenceable entities such as
// functions.
class Mangler {
 public:
  // Initialize a new Mangler instance for mangling entities within the
  // specified `FileContext`.
  explicit Mangler(FileContext& file_context) : file_context_(file_context) {}

  // Produce a deterministically unique mangled name for the function specified
  // by `function_id` and `specific_id`.
  auto Mangle(SemIR::FunctionId function_id, SemIR::SpecificId specific_id)
      -> std::string;

 private:
  // Mangle this qualified name with inner scope first, working outwards. This
  // may reduce the incidence of common prefixes in the name mangling. (i.e.:
  // every standard library name won't have a common prefix that has to be
  // skipped and compared before getting to the interesting part)
  auto MangleInverseQualifiedNameScope(llvm::raw_ostream& os,
                                       SemIR::NameScopeId name_scope_id)
      -> void;

  // Generates a mangled name using Clang mangling for imported C++ functions.
  auto MangleCppClang(const clang::NamedDecl* decl) -> std::string;

  auto sem_ir() const -> const SemIR::File& { return file_context_.sem_ir(); }

  auto names() const -> SemIR::NameStoreWrapper { return sem_ir().names(); }

  auto insts() const -> const SemIR::InstStore& { return sem_ir().insts(); }

  auto types() const -> const SemIR::TypeStore& { return sem_ir().types(); }

  auto constant_values() const -> const SemIR::ConstantValueStore& {
    return sem_ir().constant_values();
  }

  FileContext& file_context_;

  // TODO: If `file_context_` has an `InstNamer`, we could share its
  // fingerprinter.
  SemIR::InstFingerprinter fingerprinter_;

  // Clang Mangler lazily initialized when necessary. We create it once under
  // the assumption all declarations we need to mangle can use the same Mangler
  // (same AST Context).
  std::unique_ptr<clang::MangleContext> cpp_mangle_context_;
};

}  // namespace Carbon::Lower

#endif  // CARBON_TOOLCHAIN_LOWER_MANGLER_H_
