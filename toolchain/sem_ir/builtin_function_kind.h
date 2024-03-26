// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_BUILTIN_FUNCTION_KIND_H_
#define CARBON_TOOLCHAIN_SEM_IR_BUILTIN_FUNCTION_KIND_H_

#include <cstdint>

#include "common/enum_base.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

class File;

CARBON_DEFINE_RAW_ENUM_CLASS(BuiltinFunctionKind, std::uint8_t) {
#define CARBON_SEM_IR_BUILTIN_FUNCTION_KIND(Name) \
  CARBON_RAW_ENUM_ENUMERATOR(Name)
#include "toolchain/sem_ir/builtin_function_kind.def"
};

// A kind of builtin function.
class BuiltinFunctionKind : public CARBON_ENUM_BASE(BuiltinFunctionKind) {
 public:
#define CARBON_SEM_IR_BUILTIN_FUNCTION_KIND(Name) \
  CARBON_ENUM_CONSTANT_DECL(Name)
#include "toolchain/sem_ir/builtin_function_kind.def"

  // Returns the builtin function kind with the given name, or None if the name
  // is unknown.
  static auto ForBuiltinName(llvm::StringRef name) -> BuiltinFunctionKind;

  // Returns the builtin function kind corresponding to the given function
  // callee, or None if the callee is not known to be a builtin.
  static auto ForCallee(const File& sem_ir, InstId callee_id)
      -> BuiltinFunctionKind;

  // Determines whether this builtin function kind can have the specified
  // function type.
  auto IsValidType(const File& sem_ir, llvm::ArrayRef<TypeId> arg_types,
                   TypeId return_type) const -> bool;
};

#define CARBON_SEM_IR_BUILTIN_FUNCTION_KIND(Name) \
  CARBON_ENUM_CONSTANT_DEFINITION(BuiltinFunctionKind, Name)
#include "toolchain/sem_ir/builtin_function_kind.def"

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_BUILTIN_FUNCTION_KIND_H_
