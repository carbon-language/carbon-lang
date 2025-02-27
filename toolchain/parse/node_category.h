// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_PARSE_NODE_CATEGORY_H_
#define CARBON_TOOLCHAIN_PARSE_NODE_CATEGORY_H_

#include "common/ostream.h"
#include "llvm/ADT/BitmaskEnum.h"

namespace Carbon::Parse {

LLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();

// An X-macro for node categories. Uses should look like:
//
//   #define CARBON_NODE_CATEGORY_FOR_XYZ(Name) ...
//   CARBON_NODE_CATEGORY(CARBON_NODE_CATEGORY_FOR_XYZ)
//   #undef CARBON_NODE_CATEGORY_FOR_XYZ
#define CARBON_NODE_CATEGORY(X) \
  X(Decl)                       \
  X(Expr)                       \
  X(ImplAs)                     \
  X(IntConst)                   \
  X(MemberExpr)                 \
  X(MemberName)                 \
  X(Modifier)                   \
  X(NonExprIdentifierName)      \
  X(PackageName)                \
  X(Pattern)                    \
  X(Requirement)                \
  X(Statement)

// Represents a set of keyword modifiers, using a separate bit per modifier.
class NodeCategory : public Printable<NodeCategory> {
 private:
  // Use an enum to get incremental bit shifts.
  enum class BitShift : uint8_t {
#define CARBON_NODE_CATEGORY_FOR_BIT_SHIFT(Name) Name,
    CARBON_NODE_CATEGORY(CARBON_NODE_CATEGORY_FOR_BIT_SHIFT)
#undef CARBON_NODE_CATEGORY_FOR_BIT_SHIFT

    // For `LLVM_MARK_AS_BITMASK_ENUM`.
    LargestValueMarker,
  };

 public:
  // Provide values as an enum. This doesn't expose these as NodeCategory
  // instances just due to the duplication of declarations that would cause.
  //
  // We expect this to grow, so are using a bigger size than needed.
  // NOLINTNEXTLINE(performance-enum-size)
  enum RawEnumType : uint32_t {
#define CARBON_NODE_CATEGORY_FOR_BIT_MASK(Name) \
  Name = 1 << static_cast<uint8_t>(BitShift::Name),
    CARBON_NODE_CATEGORY(CARBON_NODE_CATEGORY_FOR_BIT_MASK)
#undef CARBON_NODE_CATEGORY_FOR_BIT_MASK
    // If you add a new category here, also add it to the Print function.
    None = 0,

    LLVM_MARK_AS_BITMASK_ENUM(
        /*LargestValue=*/1
        << (static_cast<uint8_t>(BitShift::LargestValueMarker) - 1))
  };

  // Support implicit conversion so that the difference with the member enum is
  // opaque.
  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr NodeCategory(RawEnumType value) : value_(value) {}

  // Returns true if there's a non-empty set intersection.
  constexpr auto HasAnyOf(NodeCategory other) const -> bool {
    return value_ & other.value_;
  }

  // Returns the set inverse.
  constexpr auto operator~() const -> NodeCategory { return ~value_; }

  friend auto operator==(NodeCategory lhs, NodeCategory rhs) -> bool {
    return lhs.value_ == rhs.value_;
  }

  auto Print(llvm::raw_ostream& out) const -> void;

 private:
  RawEnumType value_;
};

}  // namespace Carbon::Parse

#endif  // CARBON_TOOLCHAIN_PARSE_NODE_CATEGORY_H_
