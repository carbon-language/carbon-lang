// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_COMMON_SOURCE_LOCATION_H_
#define CARBON_EXPLORER_COMMON_SOURCE_LOCATION_H_

#include <string>
#include <string_view>

#include "common/ostream.h"
#include "explorer/common/nonnull.h"

namespace Carbon {

class SourceLocation {
 public:
  // The filename should be eternal or arena-allocated to eliminate copies.
  constexpr SourceLocation(const char* filename, int line_num)
      : filename_(filename), line_num_(line_num) {}
  SourceLocation(Nonnull<const std::string*> filename, int line_num)
      : filename_(filename->c_str()), line_num_(line_num) {}

  SourceLocation(const SourceLocation&) = default;
  SourceLocation(SourceLocation&&) = default;
  auto operator=(const SourceLocation&) -> SourceLocation& = default;
  auto operator=(SourceLocation&&) -> SourceLocation& = default;

  auto operator==(SourceLocation other) const -> bool {
    return filename_ == other.filename_ && line_num_ == other.line_num_;
  }

  void Print(llvm::raw_ostream& out) const {
    out << filename_ << ":" << line_num_;
  }
  auto ToString() const -> std::string {
    std::string result;
    llvm::raw_string_ostream out(result);
    Print(out);
    return result;
  }
  LLVM_DUMP_METHOD void Dump() const { Print(llvm::errs()); }

 private:
  std::string_view filename_;
  int line_num_;
};

}  // namespace Carbon

#endif  // CARBON_EXPLORER_COMMON_SOURCE_LOCATION_H_
