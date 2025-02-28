// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/base/int.h"

namespace Carbon {

auto IntStore::CanonicalBitWidth(int significant_bits) -> int {
  // For larger integers, we store them in as a signed APInt with a canonical
  // width that is the smallest multiple of the word type's bits, but no
  // smaller than a minimum of 64 bits to avoid spurious resizing of the most
  // common cases (<= 64 bits).
  static constexpr int WordWidth = llvm::APInt::APINT_BITS_PER_WORD;

  return std::max<int>(
      MinAPWidth, ((significant_bits + WordWidth - 1) / WordWidth) * WordWidth);
}

auto IntStore::CanonicalizeSigned(llvm::APInt value) -> llvm::APInt {
  return value.sextOrTrunc(CanonicalBitWidth(value.getSignificantBits()));
}

auto IntStore::CanonicalizeUnsigned(llvm::APInt value) -> llvm::APInt {
  // We need the width to include a zero sign bit as we canonicalize to a
  // signed representation.
  return value.zextOrTrunc(CanonicalBitWidth(value.getActiveBits() + 1));
}

auto IntStore::AddLarge(int64_t value) -> IntId {
  auto ap_id =
      values_.Add(llvm::APInt(CanonicalBitWidth(64), value, /*isSigned=*/true));
  return MakeIndexOrNone(ap_id.index);
}

auto IntStore::AddSignedLarge(llvm::APInt value) -> IntId {
  auto ap_id = values_.Add(CanonicalizeSigned(value));
  return MakeIndexOrNone(ap_id.index);
}

auto IntStore::AddUnsignedLarge(llvm::APInt value) -> IntId {
  auto ap_id = values_.Add(CanonicalizeUnsigned(value));
  return MakeIndexOrNone(ap_id.index);
}

auto IntStore::LookupLarge(int64_t value) const -> IntId {
  auto ap_id = values_.Lookup(
      llvm::APInt(CanonicalBitWidth(64), value, /*isSigned=*/true));
  return MakeIndexOrNone(ap_id.index);
}

auto IntStore::LookupSignedLarge(llvm::APInt value) const -> IntId {
  auto ap_id = values_.Lookup(CanonicalizeSigned(value));
  return MakeIndexOrNone(ap_id.index);
}

auto IntStore::OutputYaml() const -> Yaml::OutputMapping {
  return values_.OutputYaml();
}

auto IntStore::CollectMemUsage(MemUsage& mem_usage, llvm::StringRef label) const
    -> void {
  mem_usage.Collect(std::string(label), values_);
}

}  // namespace Carbon
