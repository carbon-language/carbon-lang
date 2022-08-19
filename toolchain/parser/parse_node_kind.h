// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_PARSER_PARSE_NODE_KIND_H_
#define CARBON_TOOLCHAIN_PARSER_PARSE_NODE_KIND_H_

#include <cstdint>
#include <iterator>

#include "common/ostream.h"
#include "llvm/ADT/StringRef.h"

namespace Carbon {

// A class wrapping an enumeration of the different kinds of nodes in the parse
// tree.
//
// Rather than using a raw enumerator for each distinct kind of node produced by
// the parser, we wrap the enumerator in a class to expose a more rich API
// including bidirectional mappings to string spellings of the different kinds
// and any relevant classification.
//
// Instances of this type should always be created using the `constexpr` static
// member functions. These instances are designed specifically to be usable in
// `case` labels of `switch` statements just like an enumerator would.
class ParseNodeKind {
  // Note that this must be declared earlier in the class so that its type can
  // be used, for example in the conversion operator.
  enum class KindEnum : uint8_t {
#define CARBON_PARSE_NODE_KIND(Name) Name,
#include "toolchain/parser/parse_node_kind.def"
  };

 public:
  // The formatting for this macro is weird due to a `clang-format` bug. See
  // https://bugs.llvm.org/show_bug.cgi?id=48320 for details.
#define CARBON_PARSE_NODE_KIND(Name)            \
  static constexpr auto Name()->ParseNodeKind { \
    return ParseNodeKind(KindEnum::Name);       \
  }
#include "toolchain/parser/parse_node_kind.def"

  // The default constructor is deleted because objects of this type should
  // always be constructed using the above factory functions for each unique
  // kind.
  ParseNodeKind() = delete;

  friend auto operator==(ParseNodeKind lhs, ParseNodeKind rhs) -> bool {
    return lhs.kind_ == rhs.kind_;
  }
  friend auto operator!=(ParseNodeKind lhs, ParseNodeKind rhs) -> bool {
    return lhs.kind_ != rhs.kind_;
  }

  // Gets a friendly name for the token for logging or debugging.
  [[nodiscard]] auto name() const -> llvm::StringRef;

  // Enable conversion to our private enum, including in a `constexpr` context,
  // to enable usage in `switch` and `case`. The enum remains private and
  // nothing else should be using this function.
  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr operator KindEnum() const { return kind_; }

  void Print(llvm::raw_ostream& out) const { out << name(); }

 private:
  constexpr explicit ParseNodeKind(KindEnum k) : kind_(k) {}

  KindEnum kind_;
};

// We expect the parse node kind to fit compactly into 8 bits.
static_assert(sizeof(ParseNodeKind) == 1, "Kind objects include padding!");

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_PARSER_PARSE_NODE_KIND_H_
