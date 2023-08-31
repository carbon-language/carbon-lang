// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_BASE_SOURCE_LOCATION_H_
#define CARBON_EXPLORER_BASE_SOURCE_LOCATION_H_

#include <string>
#include <string_view>

#include "common/ostream.h"
#include "explorer/base/nonnull.h"

namespace Carbon {

// Describes the kind of file that the source location is within.
enum class FileKind { Main, Prelude, Import, Unknown, Last = Unknown };

class SourceLocation : public Printable<SourceLocation> {
 public:
  // Produce a source location that is known to not be used, because it is fed
  // into an operation that creates no AST nodes and whose diagnostics are
  // discarded.
  static auto DiagnosticsIgnored() -> SourceLocation {
    return SourceLocation("", 0, FileKind::Unknown);
  }

  // The filename should be eternal or arena-allocated to eliminate copies.
  explicit constexpr SourceLocation(std::string_view filename, int line_num,
                                    FileKind file_kind)
      : filename_(filename), line_num_(line_num), file_kind_(file_kind) {}
  explicit SourceLocation(Nonnull<const std::string*> filename, int line_num,
                          FileKind file_kind)
      : filename_(*filename), line_num_(line_num), file_kind_(file_kind) {}

  SourceLocation(const SourceLocation&) = default;
  SourceLocation(SourceLocation&&) = default;
  auto operator=(const SourceLocation&) -> SourceLocation& = default;
  auto operator=(SourceLocation&&) -> SourceLocation& = default;

  auto operator==(SourceLocation other) const -> bool {
    return filename_ == other.filename_ && line_num_ == other.line_num_ &&
           file_kind_ == other.file_kind_;
  }

  auto filename() const -> std::string_view { return filename_; }

  auto file_kind() const -> FileKind { return file_kind_; }

  void Print(llvm::raw_ostream& out) const {
    if (file_kind_ == FileKind::Prelude) {
      out << llvm::StringRef(filename_).rsplit("/").second << ":" << line_num_;
    } else {
      out << filename_ << ":" << line_num_;
    }
  }

  auto ToString() const -> std::string {
    std::string result;
    llvm::raw_string_ostream out(result);
    Print(out);
    return result;
  }

 private:
  std::string_view filename_;
  int line_num_;
  FileKind file_kind_;
};

}  // namespace Carbon

#endif  // CARBON_EXPLORER_BASE_SOURCE_LOCATION_H_
