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
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "toolchain/diagnostics/kind.h"

namespace Carbon::Diagnostics {

// The severity of a diagnostic.
//
// A diagnostic reports one problem and its message says what that problem is,
// so these are the only levels there are. Everything else a diagnostic has to
// say is a label attached to it.
enum class Level : int8_t {
  // A likely problem with the program.
  Warning,
  // The program is not valid.
  Error,
};

// What a labeled range of source has to do with the diagnostic it is attached
// to.
//
// A label is read against the code it marks rather than on its own, which is
// what separates one from a message: `declared here` is not a sentence anyone
// would want to read by itself. Anything that doesn't mark source that way is
// not a label: see `CARBON_DIAGNOSTIC_CONTEXT` and
// `CARBON_DIAGNOSTIC_LOCATION_INFO`.
enum class LabelCategory : int8_t {
  // Source that explains why the problem happened without being part of it.
  Info,
  // Source that is directly part of the problem.
  Primary,
};

// Provides a definition of a diagnostic: the message that reports a problem
// and stands alone as the sentence describing it. For example:
//   CARBON_DIAGNOSTIC(MyDiagnostic, Error, "invalid code!");
//   CARBON_DIAGNOSTIC(MyDiagnostic, Warning, "found {0}, expected {1}",
//                     std::string, std::string);
//
// Arguments are passed to llvm::formatv; see:
// https://llvm.org/doxygen/FormatVariadic_8h_source.html
//
// See `Diagnostics::Emitter::Emit` for comments about argument lifetimes.
#define CARBON_DIAGNOSTIC(DiagnosticName, LevelValue, Format, ...)         \
  static constexpr auto DiagnosticName =                                   \
      ::Carbon::Diagnostics::DiagnosticBase<__VA_ARGS__>(                  \
          ::Carbon::Diagnostics::Kind::DiagnosticName,                     \
          ::Carbon::Diagnostics::Level::LevelValue, /*is_on_scope=*/false, \
          Format)

// Provides a definition of a label: a phrase attached to a range of source,
// saying what that range has to do with the diagnostic. For example:
//   CARBON_DIAGNOSTIC_LABEL(ArgsPassedHere, Primary,
//                           "{0} argument{0:s} passed here", IntAsSelect);
//   CARBON_DIAGNOSTIC_LABEL(FunctionDeclaredHere, Info,
//                           "calling function declared here");
//
// The category says what the range is to the diagnostic; see `LabelCategory`.
// A label has a format and arguments of its own, so it says only what it needs
// to about the place it marks, and both are validated as a message's are.
//
// It has no level, because it is read against the code rather than on its own,
// and no kind, because there is no central registry of labels: a label is
// declared where it is attached and nowhere else. What the registry does for a
// diagnostic is done here by `check_diagnostics.py`, which reports a label that
// is never attached, and by the compiler, which warns about most of them.
//
// It still carries its own name, so that `--include-diagnostic-kind` can say
// which label produced a line and a test can match on it. See
// `Emitter::Builder::Attach`.
#define CARBON_DIAGNOSTIC_LABEL(LabelName, CategoryValue, Format, ...)     \
  static constexpr auto LabelName =                                        \
      ::Carbon::Diagnostics::LabelBase<__VA_ARGS__>(                       \
          ::Carbon::Diagnostics::LabelCategory::CategoryValue, #LabelName, \
          Format)

// Provides a definition of a context: the operation a problem happened inside.
// For example:
//   CARBON_DIAGNOSTIC_CONTEXT(InCallToFunctionParam,
//                             "initializing parameter {0}", int);
//
// A context is not a label. It is a sentence in its own right, the way a
// message is, so where a diagnostic has one it leads and the message is read
// against the code like anything else explaining it; see `LeadingContext`. It
// is attached by a `ContextScope` rather than by the code emitting a
// diagnostic.
#define CARBON_DIAGNOSTIC_CONTEXT(ContextName, Format, ...)              \
  static constexpr auto ContextName =                                    \
      ::Carbon::Diagnostics::ContextBase<__VA_ARGS__>(/*is_soft=*/false, \
                                                      #ContextName, Format)

// Like `CARBON_DIAGNOSTIC_CONTEXT`, for a context used only as a fallback: it
// is dropped when the diagnostic already has one, which is assumed to describe
// the failure better.
#define CARBON_DIAGNOSTIC_SOFT_CONTEXT(ContextName, Format, ...)        \
  static constexpr auto ContextName =                                   \
      ::Carbon::Diagnostics::ContextBase<__VA_ARGS__>(/*is_soft=*/true, \
                                                      #ContextName, Format)

// Provides a definition of a step in the path by which a location was reached
// -- an import, an include, a macro expansion. For example:
//   CARBON_DIAGNOSTIC_LOCATION_INFO(InImport, "imported from");
//
// This is not a label either: it marks no source, and is drawn as
// `<text>: <location>` above the file it leads to. It is attached by an
// emitter's `ConvertLoc`, and takes no arguments, because that is all the
// conversion has to give it.
#define CARBON_DIAGNOSTIC_LOCATION_INFO(StepName, Format) \
  static constexpr auto StepName =                        \
      ::Carbon::Diagnostics::LocationInfoBase(#StepName, Format)

// Similar to `CARBON_DIAGNOSTIC`, but for diagnostics that are generated on a
// scope; see `Diagnostic::is_on_scope` for details.
#define CARBON_DIAGNOSTIC_ON_SCOPE(DiagnosticName, LevelValue, Format, ...) \
  static constexpr auto DiagnosticName =                                    \
      ::Carbon::Diagnostics::DiagnosticBase<__VA_ARGS__>(                   \
          ::Carbon::Diagnostics::Kind::DiagnosticName,                      \
          ::Carbon::Diagnostics::Level::LevelValue, /*is_on_scope=*/true,   \
          Format)

// A location for a diagnostic in a file. A `Loc` must not outlive the
// `SourceBuffer` it refers into: `filename`, `line`, and `file_text` are all
// references into it.
//
// This is data only; see `Diagnostics::Renderer` for how it is drawn.
struct Loc {
  // Name of the file or buffer that this diagnostic refers to.
  llvm::StringRef filename;

  // A reference to the line of the error.
  llvm::StringRef line;

  // The whole file `line` points into, when there is one. Rendering uses it to
  // show the lines between two spans that are close together, and `line` is a
  // slice of it, so the two can't disagree.
  llvm::StringRef file_text;

  // 1-based line number. -1 indicates unknown; other values are unused.
  int32_t line_number = -1;

  // 1-based byte offset into `line`, despite the name. -1 indicates unknown;
  // other values are unused.
  //
  // Bytes rather than columns because that is what every producer has: the
  // lexer subtracts source pointers, and Clang's columns are byte offsets too.
  // A location may name a byte inside a multi-byte character or a tab, so
  // rendering converts to columns rather than assuming the two agree.
  int32_t column_number = -1;

  // The number of bytes the location covers in `line`, starting at
  // `column_number`. Should always be at least 1.
  //
  // This is the range underlined in the source snippet. A message's own
  // location is a point that an editor puts a cursor on, so where a range was
  // attached to the same line to say what the problem covers, that range is
  // drawn in its place rather than this one. See `Emitter::Builder::Attach`.
  int32_t length = 1;
};

// Applies a format to its arguments.
//
// Formatting is deferred so that a consumer can inspect a diagnostic's
// arguments rather than parsing its formatted text; see
// `Message::format_args`.
using FormatFn = std::function<auto(llvm::StringLiteral format,
                                    llvm::ArrayRef<llvm::Any> args)
                                   ->std::string>;

// A step in the path by which a location was reached; see
// `CARBON_DIAGNOSTIC_LOCATION_INFO`.
//
// Several steps stack into a path, outermost first, so they are held in the
// order they were walked by whatever they lead to.
struct LocationInfo {
  // Returns the step's text, matching `Message` and `Label`. It is the declared
  // format string verbatim, since a step has no arguments to apply to it.
  auto Format() const -> std::string { return format.str(); }

  // The location this step names, which is where the step was taken from
  // rather than where it arrived.
  Loc loc;

  // The step's own name; see `CARBON_DIAGNOSTIC_LOCATION_INFO`.
  llvm::StringLiteral name;

  // The step's format, which with no arguments to apply is its text.
  llvm::StringLiteral format;
};

// The message reporting a problem, which is what starts a diagnostic.
//
// A message stands alone: it is the sentence that says what is wrong, and
// makes sense read by itself with nothing around it. Everything that only
// makes sense against a piece of code is a `Label` attached to the diagnostic
// instead.
struct Message {
  // Helper for calling `format_fn`. A message always has text; the empty result
  // for one that doesn't is so that rendering a `Diagnostic` nothing filled in
  // is never the reason a compiler dies while reporting a problem.
  auto Format() const -> std::string {
    return format_fn ? format_fn(format, format_args) : std::string();
  }

  // The diagnostic's kind.
  Kind kind;

  // The message's level, which is the diagnostic's.
  Level level;

  // The primary location to associate with the message: where an editor puts a
  // cursor, and the range underlined in the snippet. See `Loc::length`.
  Loc loc;

  // How `loc` was reached, when it was reached through an import, an include,
  // or a macro expansion. Inline storage is off because a step is large and
  // almost nothing is reached through one.
  llvm::SmallVector<LocationInfo, 0> location_info;

  // The message's format string. This, along with format_args, will be
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
  FormatFn format_fn;
};

// A range of source attached to a diagnostic, and optionally a phrase saying
// what that range has to do with it.
//
// A label with no text marks its range and says nothing, which is how a
// diagnostic points at the code its message is about without repeating the
// message against it.
struct Label {
  // Returns the label's text, which is empty when it only marks a range.
  auto Format() const -> std::string {
    return format_fn ? format_fn(format, format_args) : std::string();
  }

  // What this range has to do with the diagnostic.
  LabelCategory category;

  // The range of source this marks, from `Loc::column_number` across
  // `Loc::length`.
  Loc loc;

  // How `loc` was reached, when it was reached through an import, an include,
  // or a macro expansion. Inline storage is off because a step is large and
  // almost nothing is reached through one.
  llvm::SmallVector<LocationInfo, 0> location_info;

  // The label's own name; see `CARBON_DIAGNOSTIC_LABEL`. Empty for a label that
  // only marks its range, which nothing needs to name because it says nothing.
  llvm::StringLiteral name = "";

  // The label's format for llvm::formatv, and the arguments to apply to it.
  // Both are empty for a label that only marks its range.
  llvm::StringLiteral format = "";
  llvm::SmallVector<llvm::Any> format_args;

  // Empty for a label that only marks its range.
  FormatFn format_fn;
};

// The operation a problem happened inside; see `CARBON_DIAGNOSTIC_CONTEXT`.
//
// This is not a label: its text stands alone as a sentence rather than being
// read against the source it names, which is what lets it lead a diagnostic in
// place of the message.
struct Context {
  // Returns the context's text.
  auto Format() const -> std::string {
    return format_fn ? format_fn(format, format_args) : std::string();
  }

  // Where the operation was entered.
  Loc loc;

  // How `loc` was reached, when it was reached through an import, an include,
  // or a macro expansion. Inline storage is off because a step is large and
  // almost nothing is reached through one.
  llvm::SmallVector<LocationInfo, 0> location_info;

  // The context's own name; see `CARBON_DIAGNOSTIC_CONTEXT`.
  llvm::StringLiteral name;

  // The context's format for llvm::formatv, and the arguments to apply to it.
  llvm::StringLiteral format;
  llvm::SmallVector<llvm::Any> format_args;

  // Returns the formatted string. By default, this uses llvm::formatv.
  FormatFn format_fn;
};

// An instance of a single error or warning. Information about the diagnostic
// can be recorded into it for more complex consumers.
struct Diagnostic {
  // The diagnostic's level.
  Level level;

  // Whether a diagnostic should only sort by `last_byte_offset` (which is
  // normal), or if it's generated on a scope and should be sorted based on the
  // message's line and column when the `last_byte_offset` is equal. This is
  // used by `SortingConsumer`.
  bool is_on_scope;

  // The byte offset of the final token which is associated with the diagnostic.
  // This is used by `SortingConsumer`. This is separate from the
  // `Loc` because it must refer to a position in the primary file
  // being processed by a consumer, and has no use cross-file or in labels.
  //
  // This will usually be the start position (not end) of the last lexed token
  // processed before the diagnostic; it could also be `-1` when no source code
  // needs to be processed for a diagnostic, or an appropriate byte offset when
  // we specifically want a different diagnostic ordering than when a diagnostic
  // is issued.
  int32_t last_byte_offset = -1;

  // The one problem being reported.
  Message message;

  // The operations the problem happened inside, outermost first, which is the
  // order their `ContextScope`s were entered in.
  //
  // A `Context` is undesirably large for inline storage by SmallVector, so we
  // specify 0.
  llvm::SmallVector<Context, 0> contexts;

  // The source attached to the message, in the order it was attached.
  //
  // A `Label` is undesirably large for inline storage by SmallVector, so we
  // specify 0.
  llvm::SmallVector<Label, 0> labels;
};

// Returns the context that leads the diagnostic in place of its message, or
// null when there is none.
//
// A context names the operation that failed, and when a diagnostic has one that
// is what the reader is told about; the message is then read against the code
// like anything else explaining it. Anything presenting a diagnostic as a
// single sentence -- the headline of the rendering, an editor's hover text --
// wants this rather than the message.
inline auto LeadingContext(const Diagnostic& diagnostic) -> const Context* {
  return diagnostic.contexts.empty() ? nullptr : &diagnostic.contexts.front();
}

// Rejects `Args` a diagnostic or label may not be formatted with, when the
// enclosing constructor is instantiated.
template <typename... Args>
constexpr auto CheckArgTypes() -> void {
  static_assert((... && !(std::is_same_v<Args, llvm::StringRef> ||
                          std::is_same_v<Args, llvm::StringLiteral>)),
                "String type disallowed in diagnostics. See "
                "https://github.com/carbon-language/carbon-lang/blob/trunk/"
                "toolchain/docs/diagnostics.md#diagnostic-parameter-types");
}

// Use the DIAGNOSTIC macro to instantiate this.
// This stores static information about a diagnostic category.
template <typename... Args>
struct DiagnosticBase {
  explicit constexpr DiagnosticBase(Kind kind, Level level, bool is_on_scope,
                                    llvm::StringLiteral format)
      : Kind(kind), Level(level), IsOnScope(is_on_scope), Format(format) {
    CheckArgTypes<Args...>();
  }

  // The diagnostic's kind.
  Kind Kind;
  // The diagnostic's level.
  Level Level;
  // See `Diagnostic::is_on_scope`.
  bool IsOnScope;
  // The diagnostic's format for llvm::formatv.
  llvm::StringLiteral Format;
};

// Use the CARBON_DIAGNOSTIC_LABEL macro to instantiate this. This stores the
// static information about a label: what it is to the diagnostic, what it is
// called, its format, and the types of its arguments.
template <typename... Args>
struct LabelBase {
  explicit constexpr LabelBase(LabelCategory category, llvm::StringLiteral name,
                               llvm::StringLiteral format)
      : Category(category), Name(name), Format(format) {
    CheckArgTypes<Args...>();
  }

  // What the labeled range is to the diagnostic.
  LabelCategory Category;
  // The label's own name, which stands in for a diagnostic's kind when naming
  // what produced a line of output.
  llvm::StringLiteral Name;
  // The label's format for llvm::formatv.
  llvm::StringLiteral Format;
};

// Use the CARBON_DIAGNOSTIC_CONTEXT macro to instantiate this. This stores the
// static information about a context: whether it is only a fallback, what it is
// called, its format, and the types of its arguments.
template <typename... Args>
struct ContextBase {
  explicit constexpr ContextBase(bool is_soft, llvm::StringLiteral name,
                                 llvm::StringLiteral format)
      : IsSoft(is_soft), Name(name), Format(format) {
    CheckArgTypes<Args...>();
  }

  // Whether this is dropped when the diagnostic already has a context; see
  // `CARBON_DIAGNOSTIC_SOFT_CONTEXT`.
  bool IsSoft;
  // The context's own name, which stands in for a diagnostic's kind when naming
  // what produced a line of output.
  llvm::StringLiteral Name;
  // The context's format for llvm::formatv.
  llvm::StringLiteral Format;
};

// Use the CARBON_DIAGNOSTIC_LOCATION_INFO macro to instantiate this. This
// stores the static information about a step in a path: what it is called and
// its text. It is not a template, because a step has no arguments.
struct LocationInfoBase {
  explicit constexpr LocationInfoBase(llvm::StringLiteral name,
                                      llvm::StringLiteral format)
      : Name(name), Format(format) {}

  // The step's own name, which stands in for a diagnostic's kind when naming
  // what produced a line of output.
  llvm::StringLiteral Name;
  // The step's text.
  llvm::StringLiteral Format;
};

}  // namespace Carbon::Diagnostics

#endif  // CARBON_TOOLCHAIN_DIAGNOSTICS_DIAGNOSTIC_H_
