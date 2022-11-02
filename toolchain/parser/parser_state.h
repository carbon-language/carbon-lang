// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_PARSER_PARSER_STATE_H_
#define CARBON_TOOLCHAIN_PARSER_PARSER_STATE_H_

#include <cstdint>
#include <iterator>

#include "common/ostream.h"
#include "llvm/ADT/StringRef.h"

namespace Carbon {

class ParserState {
  // Note that this must be declared earlier in the class so that its type can
  // be used, for example in the conversion operator.
  enum class StateEnum : uint8_t {
#define CARBON_PARSER_STATE(Name) Name,
#include "toolchain/parser/parser_state.def"
  };

 public:
  // `clang-format` has a bug with spacing around `->` returns in macros. See
  // https://bugs.llvm.org/show_bug.cgi?id=48320 for details.
#define CARBON_PARSER_STATE(Name)             \
  static constexpr auto Name()->ParserState { \
    return ParserState(StateEnum::Name);      \
  }
#include "toolchain/parser/parser_state.def"

  // The default constructor is deleted because objects of this type should
  // always be constructed using the above factory functions for each unique
  // kind.
  ParserState() = delete;

  friend auto operator==(ParserState lhs, ParserState rhs) -> bool {
    return lhs.state_ == rhs.state_;
  }
  friend auto operator!=(ParserState lhs, ParserState rhs) -> bool {
    return lhs.state_ != rhs.state_;
  }

  // Gets a friendly name for the token for logging or debugging.
  [[nodiscard]] auto name() const -> llvm::StringRef;

  // Enable conversion to our private enum, including in a `constexpr` context,
  // to enable usage in `switch` and `case`. The enum remains private and
  // nothing else should be using this function.
  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr operator StateEnum() const { return state_; }

  void Print(llvm::raw_ostream& out) const { out << name(); }

 private:
  constexpr explicit ParserState(StateEnum k) : state_(k) {}

  StateEnum state_;
};

// We expect the parse node kind to fit compactly into 8 bits.
static_assert(sizeof(ParserState) == 1, "ParserState objects include padding!");

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_PARSER_PARSER_STATE_H_
