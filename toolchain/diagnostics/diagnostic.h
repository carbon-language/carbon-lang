// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DIAGNOSTICS_DIAGNOSTIC_H_
#define CARBON_TOOLCHAIN_DIAGNOSTICS_DIAGNOSTIC_H_

#include <cstdint>
#include <functional>
#include <string>

#include "common/check.h"
#include "llvm/ADT/Any.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "toolchain/diagnostics/diagnostic_kind.h"

namespace Carbon::Diagnostics {

enum class Level : int8_t {
  // Information about the location of another diagnostic, showing how we
  // reached that location. This is currently only used for the "in import"
  // message.
  LocationInfo,
  // A note, not indicating an error on its own, but possibly providing
  // additional information for an error or warning.
  Note,
  // A warning diagnostic, indicating a likely problem with the program.
  Warning,
  // An error diagnostic, indicating that the program is not valid.
  Error,
};

// Provides a definition of a diagnostic. For example:
//   CARBON_DIAGNOSTIC(MyDiagnostic, Error, "invalid code!");
//   CARBON_DIAGNOSTIC(MyDiagnostic, Warning, "found {0}, expected {1}",
//                     std::string, std::string);
//
// Arguments are passed to llvm::formatv; see:
// https://llvm.org/doxygen/FormatVariadic_8h_source.html
//
// See `Diagnostics::Emitter::Emit` for comments about argument lifetimes.
#define CARBON_DIAGNOSTIC(DiagnosticName, LevelValue, Format, ...) \
  static constexpr auto DiagnosticName =                           \
      ::Carbon::Diagnostics::DiagnosticBase<__VA_ARGS__>(          \
          ::Carbon::Diagnostics::Kind::DiagnosticName,             \
          ::Carbon::Diagnostics::Level::LevelValue, Format)

// A location for a diagnostic in a file. The lifetime of a Loc
// is required to be less than SourceBuffer that it refers to due to the
// contained filename and line references.
struct Loc {
  // Writes the location to the given stream. It will be formatted as
  // `<filename>:<line_number>:<column_number>: ` with parts dropped when
  // unknown.
  auto FormatLocation(llvm::raw_ostream& out) const -> void;

  // Write the source snippet corresponding to this location to the given
  // stream.
  auto FormatSnippet(llvm::raw_ostream& out, int indent = 0) const -> void;

  // Name of the file or buffer that this diagnostic refers to.
  llvm::StringRef filename;

  // A reference to the line of the error.
  llvm::StringRef line;

  // 1-based line number. -1 indicates unknown; other values are unused.
  int32_t line_number = -1;

  // 1-based column number. -1 indicates unknown; other values are unused.
  int32_t column_number = -1;

  // The number of characters corresponding to the location in the line,
  // starting at column_number. Should always be at least 1.
  int32_t length = 1;
};

// A message composing a diagnostic. This may be the main message, but can also
// be notes providing more information.
struct Message {
  // Helper for calling `format_fn`.
  auto Format() const -> std::string { return format_fn(*this); }

  // The diagnostic's kind.
  Kind kind;

  // The diagnostic's level.
  Level level;

  // The calculated location of the diagnostic.
  Loc loc;

  // The diagnostic's format string. This, along with format_args, will be
  // passed to format_fn.
  llvm::StringLiteral format;

  // A list of format arguments.
  //
  // These may be used by non-standard consumers to inspect diagnostic details
  // without needing to parse the formatted string; however, it should be
  // understood that diagnostic formats are subject to change and the llvm::Any
  // offers limited compile-time type safety. Integration tests are required.
  llvm::SmallVector<llvm::Any> format_args;

  // Returns the formatted string. By default, this uses llvm::formatv.
  std::function<auto(const Message&)->std::string> format_fn;
};

// An instance of a single error or warning.  Information about the diagnostic
// can be recorded into it for more complex consumers.
struct Diagnostic {
  // The diagnostic's level.
  Level level;

  // The byte offset of the final token which is associated with the diagnostic.
  // This is used by `SortingConsumer`. This is separate from the
  // `Loc` because it must refer to a position in the primary file
  // being processed by a consumer, and has no use cross-file or in notes.
  //
  // This will usually be the start position (not end) of the last lexed token
  // processed before the diagnostic; it could also be `-1` when no source code
  // needs to be processed for a diagnostic, or an appropriate byte offset when
  // we specifically want a different diagnostic ordering than when a diagnostic
  // is issued.
  int32_t last_byte_offset = -1;

  // Messages related to the diagnostic. Only one should be a warning or error;
  // other messages provide context.
  llvm::SmallVector<Message> messages;
};

// Use the DIAGNOSTIC macro to instantiate this.
// This stores static information about a diagnostic category.
template <typename... Args>
struct DiagnosticBase {
  explicit constexpr DiagnosticBase(Kind kind, Level level,
                                    llvm::StringLiteral format)
      : Kind(kind), Level(level), Format(format) {
    static_assert((... && !(std::is_same_v<Args, llvm::StringRef> ||
                            std::is_same_v<Args, llvm::StringLiteral>)),
                  "String type disallowed in diagnostics. See "
                  "https://github.com/carbon-language/carbon-lang/blob/trunk/"
                  "toolchain/docs/diagnostics.md#diagnostic-parameter-types");
  }

  // The diagnostic's kind.
  Kind Kind;
  // The diagnostic's level.
  Level Level;
  // The diagnostic's format for llvm::formatv.
  llvm::StringLiteral Format;
};

}  // namespace Carbon::Diagnostics

#endif  // CARBON_TOOLCHAIN_DIAGNOSTICS_DIAGNOSTIC_H_
