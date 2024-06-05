// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/raw_hashtable_metadata_group.h"

#include "llvm/ADT/StringExtras.h"

namespace Carbon::RawHashtable {

auto MetadataGroup::Print(llvm::raw_ostream& out) const -> void {
  out << "[";
  llvm::ListSeparator sep;
  for (uint8_t byte : metadata_bytes) {
    out << sep << llvm::formatv("{0:x2}", byte);
  }
  out << "]";
}

}  // namespace Carbon::RawHashtable
