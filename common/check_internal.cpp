// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/check_internal.h"

#include <cstdlib>
#include <string>

#include "common/ostream.h"
#include "llvm/Support/FormatCommon.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"

namespace Carbon::Internal {

namespace {
// Renders `fmt` over the externally-built, type-erased `adapters` into `out`,
// with the same semantics as `llvm::formatv` (including runtime format-string
// validation).
//
// TODO: We should add a type-erased helper to upstream LLVM instead of rolling
// our own type-erased version of `format` here.
auto FormatvInto(
    llvm::raw_ostream& out, llvm::StringRef format_str,
    llvm::ArrayRef<llvm::support::detail::format_adapter*> adapters) -> void {
  for (const llvm::ReplacementItem& replacement :
       llvm::formatv_object_base::parseFormatString(format_str, adapters.size(),
                                                    /*Validate=*/true)) {
    if (replacement.Type == llvm::ReplacementType::Literal ||
        replacement.Index >= adapters.size()) {
      out << replacement.Spec;
      continue;
    }
    llvm::FmtAlign(*adapters[replacement.Index], replacement.Where,
                   replacement.Width, replacement.Pad)
        .format(out, replacement.Options);
  }
}
}  // namespace

auto CheckFailImpl(
    const char* kind, const char* file, int line, const char* condition_str,
    const char* extra_format,
    llvm::ArrayRef<llvm::support::detail::format_adapter*> extra_adapters)
    -> void {
  // Render the final check string directly into one stream. The extra message
  // is rendered in place from its format string and type-erased adapters, so
  // we never materialize a separate string just for it.
  //
  // `llvm::raw_string_ostream` (rather than `common/raw_string_ostream.h`) is
  // used to avoid a dependency cycle: `RawStringOstream` itself uses
  // `CARBON_CHECK`. It is unbuffered, so `message` is populated directly.
  std::string message;
  llvm::raw_string_ostream message_stream(message);
  message_stream << kind << " failure at " << file << ":" << line;
  if (*condition_str != '\0') {
    message_stream << ": " << condition_str;
  }
  if (*extra_format != '\0') {
    message_stream << ": ";
    FormatvInto(message_stream, extra_format, extra_adapters);
  }
  message_stream << "\n";

  // This macro is defined by `--config=non-fatal-checks`.
#ifdef CARBON_NON_FATAL_CHECKS
#ifdef NDEBUG
#error "--config=non-fatal-checks is incompatible with -c opt"
#endif
  // TODO: It'd be nice to print the LLVM PrettyStackTrace, but LLVM doesn't
  // expose functionality to do so.
  llvm::sys::PrintStackTrace(llvm::errs());

  llvm::errs() << message;
#else
  // Register another signal handler to print the message. This is because we
  // want it at the bottom of output, after LLVM's builtin stack output, rather
  // than the top.
  llvm::sys::AddSignalHandler(
      [](void* str) { llvm::errs() << reinterpret_cast<char*>(str); },
      const_cast<char*>(message.c_str()));

  // It's useful to exit the program with `std::abort()` for integration with
  // debuggers and other tools. We also assume LLVM's exit handling is
  // installed, which will stack trace on `std::abort()`.
  std::abort();
#endif
}

}  // namespace Carbon::Internal
