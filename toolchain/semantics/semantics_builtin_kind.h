// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_BUILTIN_KIND_H_
#define CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_BUILTIN_KIND_H_

#include <cstdint>

#include "common/ostream.h"

namespace Carbon {

class SemanticsBuiltinKind {
 private:
  // Note that this must be declared earlier in the class so that its type can
  // be used, for example in the conversion operator.
  enum class KindEnum : uint8_t {
#define CARBON_SEMANTICS_BUILTIN_KIND(Name) Name,
#include "toolchain/semantics/semantics_builtin_kind.def"
  };

 public:
  // The count of enum values excluding Invalid.
  static constexpr uint8_t ValidCount = static_cast<uint8_t>(KindEnum::Invalid);

  // `clang-format` has a bug with spacing around `->` returns in macros. See
  // https://bugs.llvm.org/show_bug.cgi?id=48320 for details.
#define CARBON_SEMANTICS_BUILTIN_KIND(Name)            \
  static constexpr auto Name()->SemanticsBuiltinKind { \
    return SemanticsBuiltinKind(KindEnum::Name);       \
  }
#include "toolchain/semantics/semantics_builtin_kind.def"

  // The default constructor is deleted because objects of this type should
  // always be constructed using the above factory functions for each unique
  // kind.
  SemanticsBuiltinKind() = delete;

  friend auto operator==(SemanticsBuiltinKind lhs, SemanticsBuiltinKind rhs)
      -> bool {
    return lhs.kind_ == rhs.kind_;
  }
  friend auto operator!=(SemanticsBuiltinKind lhs, SemanticsBuiltinKind rhs)
      -> bool {
    return lhs.kind_ != rhs.kind_;
  }

  // Gets a friendly name for the token for logging or debugging.
  [[nodiscard]] auto name() const -> llvm::StringRef;

  // Support conversion to and from an int32_t for SemanticNode storage.
  auto AsInt() -> int32_t { return static_cast<int32_t>(kind_); }
  static auto FromInt(int32_t val) -> SemanticsBuiltinKind {
    return SemanticsBuiltinKind(static_cast<KindEnum>(val));
  }

  // Enable conversion to our private enum, including in a `constexpr`
  // context, to enable usage in `switch` and `case`. The enum remains
  // private and nothing else should be using this function.
  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr operator KindEnum() const { return kind_; }

  void Print(llvm::raw_ostream& out) const { out << name(); }

 private:
  constexpr explicit SemanticsBuiltinKind(KindEnum k) : kind_(k) {}

  KindEnum kind_;
};

// We expect the builtin kind to fit compactly into 8 bits.
static_assert(sizeof(SemanticsBuiltinKind) == 1,
              "Kind objects include padding!");

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_BUILTIN_KIND_H_
