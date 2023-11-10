// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_INTERPRETER_BUILTINS_H_
#define CARBON_EXPLORER_INTERPRETER_BUILTINS_H_

#include <array>
#include <optional>
#include <string_view>

#include "common/enum_base.h"
#include "common/error.h"
#include "explorer/ast/declaration.h"
#include "explorer/ast/expression.h"
#include "explorer/ast/value.h"
#include "explorer/base/nonnull.h"
#include "explorer/base/source_location.h"

namespace Carbon {

CARBON_DEFINE_RAW_ENUM_CLASS(Builtin, int) {
#define CARBON_BUILTIN(Name) CARBON_RAW_ENUM_ENUMERATOR(Name)
#include "explorer/interpreter/builtins.def"
};

class Builtin : public CARBON_ENUM_BASE(Builtin) {
 public:
#define CARBON_BUILTIN(Name) CARBON_ENUM_CONSTANT_DECL(Name)
#include "explorer/interpreter/builtins.def"

  static const int NumBuiltins;

  // Support conversion to and from an int for array indexing.
  using EnumBase::AsInt;
  using EnumBase::FromInt;
};

#define CARBON_BUILTIN(Name) CARBON_ENUM_CONSTANT_DEFINITION(Builtin, Name)
#include "explorer/interpreter/builtins.def"

constexpr int Builtin::NumBuiltins = Invalid.AsInt();

class Builtins {
 public:
  explicit Builtins() = default;

  // Register a declaration that might be a builtin.
  void Register(Nonnull<const Declaration*> decl);

  // Get a registered builtin.
  auto Get(SourceLocation source_loc, Builtin builtin) const
      -> ErrorOr<Nonnull<const Declaration*>>;

 private:
  std::optional<Nonnull<const Declaration*>> builtins_[Builtin::NumBuiltins] =
      {};
};

}  // namespace Carbon

#endif  // CARBON_EXPLORER_INTERPRETER_BUILTINS_H_
