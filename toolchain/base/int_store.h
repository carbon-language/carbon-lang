// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_BASE_INT_STORE_H_
#define CARBON_TOOLCHAIN_BASE_INT_STORE_H_

#include "common/check.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "toolchain/base/mem_usage.h"
#include "toolchain/base/value_ids.h"
#include "toolchain/base/value_store.h"
#include "toolchain/base/yaml.h"

namespace Carbon {

// Forward declare a testing peer so we can friend it.
namespace Testing {
struct IntStoreTestPeer;
}  // namespace Testing

// A canonicalizing value store with deep optimizations for integers.
//
// This stores integers as abstract, signed mathematical integers. The bit width
// of specific `APInt` values, either as inputs or outputs, is disregarded for
// the purpose of canonicalization and the returned integer may use a very
// different bit width `APInt` than was used when adding. There are also
// optimized paths for adding integer values representable using native integer
// types.
//
// Because the integers in the store are canonicalized without a specific bit
// width there are helper functions to coerce them to a specific desired bit
// width for use.
//
// This leverages a significant optimization for small integer values -- rather
// than canonicalizing and making them unique in a `ValueStore`, they are
// directly embedded in the `IntId` itself. Only larger integers are stored in
// an array of `APInt` values and represented as an index in the ID.
class IntStore {
 public:
  // Accepts a signed `int64_t` and uses the mathematical signed integer value
  // of it as the added integer value.
  //
  // Returns the ID corresponding to this integer value, storing an `APInt` if
  // necessary to represent it.
  auto Add(int64_t value) -> IntId {
    // First try directly making this into an ID.
    if (IntId id = IntId::TryMakeValue(value); id.is_valid()) [[likely]] {
      return id;
    }

    // Fallback for larger values.
    return AddLarge(value);
  }

  // Returns the ID corresponding to this integer value, storing an `APInt` if
  // necessary to represent it.
  auto AddSigned(llvm::APInt value) -> IntId {
    // First try directly making this into an ID.
    if (IntId id = IntId::TryMakeSignedValue(value); id.is_valid()) [[likely]] {
      return id;
    }

    // Fallback for larger values.
    return AddSignedLarge(std::move(value));
  }

  // Returns the ID corresponding to an equivalent signed integer value, storing an `APInt` if necessary to represent it.
  auto AddUnsigned(llvm::APInt value) -> IntId {
    // First try directly making this into an ID.
    if (IntId id = IntId::TryMakeUnsignedValue(value); id.is_valid())
        [[likely]] {
      return id;
    }

    // Fallback for larger values.
    return AddUnsignedLarge(std::move(value));
  }

  // Returns the value for an ID.
  //
  // This will always be a signed `APInt` with a canonical bit width for the
  // specific integer value in question.
  auto Get(IntId id) const -> llvm::APInt {
    if (id.is_value()) [[likely]] {
      return llvm::APInt(MinAPWidth, id.AsValue(), /*isSigned=*/true);
    }
    return values_.Get(APIntId(id.AsIndex()));
  }

  // Returns the value for an ID adjusted to a specific bit width.
  //
  // Note that because we store canonical mathematical integers as signed
  // integers, this always sign extends or truncates to the target width. The
  // caller can then use that as a signed or unsigned integer as needed.
  auto GetAtWidth(IntId id, int bit_width) const -> llvm::APInt {
    llvm::APInt value = Get(id);
    if (static_cast<int>(value.getBitWidth()) != bit_width) {
      value = value.sextOrTrunc(bit_width);
    }
    return value;
  }

  // Returns the value for an ID adjusted to the bit width specified with
  // another integer ID.
  //
  // This simply looks up the width integer ID, and then calls the above
  // `GetAtWidth` overload using the value found for it. See that overload for
  // more details.
  auto GetAtWidth(IntId id, IntId bit_width_id) const -> llvm::APInt {
    const llvm::APInt bit_width = Get(bit_width_id);
    CARBON_CHECK(bit_width.isStrictlyPositive() &&
                     bit_width.isSignedIntN(MinAPWidth),
                 "Invalid bit width value: {0}", bit_width);
    return GetAtWidth(id, bit_width.getSExtValue());
  }

  // Looks up the canonical ID for a value, or returns invalid if not in the
  // store.
  auto LookupSigned(llvm::APInt value) const -> IntId {
    if (IntId id = IntId::TryMakeSignedValue(value); id.is_valid()) [[likely]] {
      return id;
    }

    // Fallback for larger values.
    return LookupSignedLarge(std::move(value));
  }

  // Output a YAML description of this data structure. Note that this will only
  // include the integers that required storing, not those successfully embedded
  // into the ID space.
  auto OutputYaml() const -> Yaml::OutputMapping;

  auto array_ref() const -> llvm::ArrayRef<llvm::APInt> {
    return values_.array_ref();
  }
  auto size() const -> size_t { return values_.size(); }

  // Collects the memory usage of the separately stored integers.
  auto CollectMemUsage(MemUsage& mem_usage, llvm::StringRef label) const
      -> void;

 private:
  friend struct Testing::IntStoreTestPeer;

  struct APIntId : IdBase, Printable<APIntId> {
    using ValueType = llvm::APInt;
    static const APIntId Invalid;
    using IdBase::IdBase;
    auto Print(llvm::raw_ostream& out) const -> void {
      out << "ap_int";
      IdBase::Print(out);
    }
  };

  static constexpr int MinAPWidth = 64;

  // Pick a canonical bit width for the provided number of significant bits.
  static auto CanonicalBitWidth(int significant_bits) -> int;

  // Canonicalize an incoming signed APInt to the correct bit width.
  static auto CanonicalizeSigned(llvm::APInt value) -> llvm::APInt;

  // Canonicalize an incoming unsigned APInt to the correct bit width.
  static auto CanonicalizeUnsigned(llvm::APInt value) -> llvm::APInt;

  // Helper functions for handling values that are large enough to require an
  // allocated `APInt` for storage. Creating or manipulating that storage is
  // only a few lines of code, but it ends up expensive and a lot of code so we
  // move these out-of-line.
  auto AddLarge(int64_t value) -> IntId;
  auto AddSignedLarge(llvm::APInt value) -> IntId;
  auto AddUnsignedLarge(llvm::APInt value) -> IntId;
  auto LookupSignedLarge(llvm::APInt value) const -> IntId;

  CanonicalValueStore<APIntId> values_;
};

constexpr IntStore::APIntId IntStore::APIntId::Invalid(
    IntId::Invalid.AsIndex());

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_BASE_INT_STORE_H_
