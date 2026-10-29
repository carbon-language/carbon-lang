// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/format/format.h"

#include <cstdint>
#include <string>
#include <utility>

#include "toolchain/format/formatter.h"

namespace Carbon::Format {

auto Format(const Parse::Tree& tree, llvm::raw_ostream& out, const Style& style)
    -> bool {
  Formatter formatter(&tree, style);
  bool formatted_cleanly = formatter.Run();
  out << formatter.TakeOutput();
  return formatted_cleanly;
}

auto FormatReplacements(const Parse::Tree& tree,
                        llvm::SmallVectorImpl<Replacement>& replacements,
                        std::optional<LineRange> lines, const Style& style)
    -> bool {
  Formatter formatter(&tree, style);
  bool formatted_cleanly = formatter.Run();
  for (Replacement& replacement : formatter.ComputeReplacements(lines)) {
    replacements.push_back(std::move(replacement));
  }
  return formatted_cleanly;
}

auto ApplyReplacements(llvm::StringRef source,
                       llvm::ArrayRef<Replacement> replacements)
    -> std::string {
  std::string result;
  int32_t pos = 0;
  for (const Replacement& replacement : replacements) {
    result.append(source.data() + pos, replacement.offset - pos);
    result.append(replacement.text);
    pos = replacement.offset + replacement.length;
  }
  result.append(source.data() + pos, source.size() - pos);
  return result;
}

}  // namespace Carbon::Format
