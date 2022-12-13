// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_PARSER_PARSER_STATE_H_
#define CARBON_TOOLCHAIN_PARSER_PARSER_STATE_H_

#include "toolchain/common/enum_base.h"

namespace Carbon {

namespace Internal {
enum class ParserStateEnum : uint8_t {
#define CARBON_PARSER_STATE(Name) CARBON_ENUM_BASE_LITERAL(Name)
#include "toolchain/parser/parser_state.def"
};
}  // namespace Internal

class ParserState : public EnumBase<ParserState, Internal::ParserStateEnum> {
 public:
#define CARBON_PARSER_STATE(Name) CARBON_ENUM_BASE_FACTORY(ParserState, Name)
#include "toolchain/parser/parser_state.def"

  // Gets a friendly name for the token for logging or debugging.
  [[nodiscard]] inline auto name() const -> llvm::StringRef {
    static constexpr llvm::StringLiteral Names[] = {
#define CARBON_PARSER_STATE(Name) CARBON_ENUM_BASE_STRING(Name)
#include "toolchain/parser/parser_state.def"
    };
    return Names[static_cast<int>(val_)];
  }

 private:
  using EnumBase::EnumBase;
};

// We expect ParserState to fit compactly into 8 bits.
static_assert(sizeof(ParserState) == 1, "ParserState includes padding!");

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_PARSER_PARSER_STATE_H_
