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

namespace Carbon {

enum class DiagnosticLevel : int8_t {
  // A note, not indicating an error on its own, but possibly providing context
  // for an error.
  Note,
  // A warning diagnostic, indicating a likely problem with the program.
  Warning,
  // An error diagnostic, indicating that the program is not valid.
  Error,
};

// Provides a definition of a diagnostic. For example:
//   CARBON_DIAGNOSTIC(MyDiagnostic, Error, "Invalid code!");
//   CARBON_DIAGNOSTIC(MyDiagnostic, Warning, "Found {0}, expected {1}.",
//                     std::string, std::string);
//
// Arguments are passed to llvm::formatv; see:
// https://llvm.org/doxygen/FormatVariadic_8h_source.html
//
// See `DiagnosticEmitter::Emit` for comments about argument lifetimes.
#define CARBON_DIAGNOSTIC(DiagnosticName, Level, Format, ...) \
  static constexpr auto DiagnosticName =                      \
      ::Carbon::Internal::DiagnosticBase<__VA_ARGS__>(        \
          ::Carbon::DiagnosticKind::DiagnosticName,           \
          ::Carbon::DiagnosticLevel::Level, Format)

// A location for a diagnostic in a file. The lifetime of a DiagnosticLoc
// is required to be less than SourceBuffer that it refers to due to the
// contained filename and line references.
struct DiagnosticLoc {
  // Name of the file or buffer that this diagnostic refers to.
  llvm::StringRef filename;
  // A reference to the line of the error.
  llvm::StringRef line;
  // 1-based line number.
  int32_t line_number = -1;
  // 1-based column number.
  int32_t column_number = -1;
  // A location can represent a range of text if set to >1 value.
  int32_t length = 1;
};

// A message composing a diagnostic. This may be the main message, but can also
// be notes providing more information.
struct DiagnosticMessage {
  // The diagnostic's kind.
  DiagnosticKind kind;

  // The diagnostic's level.
  DiagnosticLevel level;

  // The calculated location of the diagnostic.
  DiagnosticLoc loc;

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
  std::function<std::string(const DiagnosticMessage&)> format_fn;
};

// An instance of a single error or warning.  Information about the diagnostic
// can be recorded into it for more complex consumers.
struct Diagnostic {
  // The diagnostic's level.
  DiagnosticLevel level;

  // Messages related to the diagnostic. Only one should be a warning or error;
  // other messages provide context.
  llvm::SmallVector<DiagnosticMessage> messages;
};

namespace Internal {

// Use the DIAGNOSTIC macro to instantiate this.
// This stores static information about a diagnostic category.
template <typename... Args>
struct DiagnosticBase {
  explicit constexpr DiagnosticBase(DiagnosticKind kind, DiagnosticLevel level,
                                    llvm::StringLiteral format)
      : Kind(kind), Level(level), Format(format) {
    static_assert((... && !std::is_same_v<Args, llvm::StringRef>),
                  "Use std::string or llvm::StringLiteral for diagnostics to "
                  "avoid lifetime issues.");
  }

  // The diagnostic's kind.
  DiagnosticKind Kind;
  // The diagnostic's level.
  DiagnosticLevel Level;
  // The diagnostic's format for llvm::formatv.
  llvm::StringLiteral Format;
};

}  // namespace Internal

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_DIAGNOSTICS_DIAGNOSTIC_H_
