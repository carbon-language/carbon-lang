// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DIAGNOSTICS_FORMAT_PROVIDERS_H_
#define CARBON_TOOLCHAIN_DIAGNOSTICS_FORMAT_PROVIDERS_H_

#include "common/check.h"
#include "common/ostream.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"

namespace Carbon {

// Selects a formatv string based on the value.
//
// When used, the true and false outputs are separated by a `|`.
//
// For example, `{0:true|false}` would yield standard bool formatting.
//
// If needed, the _full_ style string can be wrapped with `'` in order to
// preserve prefix or suffix whitespace (which is stripped by formatv). For
// example, `{0:' true | false '}` retains whitespace which would be dropped
// before `true` and after `false`.
struct FormatBool {
 public:
  bool value;
};

// Selects a formatv string based on the value.
//
// The style is a series of match cases, separated by `|`. Each case is a pair
// formatted as `<selector>:<output string>`.
//
// Supported selectors are:
// - `=<value>`: Matches when the value is correct.
// - Empty for the default. This is optional, although it's a fatal error to not
//   handle a value. If provided, it must be last.
//
// For example, `{0:=0:zero|=1:one|:other}` breaks down into:
// - `=0` -> `zero`
// - `=1` -> `one`
// - default -> `other`
//
// As another example, `{0:=1:is|:are}` is a way to handle plural-based output.
//
// If needed, the _full_ style string can be wrapped with `'` in order to
// preserve prefix or suffix whitespace (which is stripped by formatv). For
// example, `{0:'=0: zero |=1: one '}` retains whitespace which would be dropped
// after `one`.
struct FormatInt {
 public:
  int value;
};

}  // namespace Carbon

// See FormatBool.
template <>
struct llvm::format_provider<Carbon::FormatBool> {
  static void format(const Carbon::FormatBool& wrapper, raw_ostream& out,
                     StringRef style) {
    // Remove wrapping quotes if present.
    if (style.starts_with('\'') && style.ends_with('\'')) {
      style = style.drop_front().drop_back();
    }

    auto sep = style.find('|');
    CARBON_CHECK(
        sep != llvm::StringRef::npos,
        "FormatBool requires a `|` separating true and false results: `{0}`",
        style);
    if (wrapper.value) {
      out << style.take_front(sep);
    } else {
      out << style.drop_front(sep + 1);
    }
  }
};

// See FormatInt.
template <>
struct llvm::format_provider<Carbon::FormatInt> {
  static void format(const Carbon::FormatInt& wrapper, raw_ostream& out,
                     StringRef style) {
    // Remove wrapping quotes if present.
    if (style.starts_with('\'') && style.ends_with('\'')) {
      style = style.drop_front().drop_back();
    }

    auto cursor = style;
    while (!cursor.empty()) {
      auto case_sep = cursor.find("|");
      auto token = cursor.substr(0, case_sep);
      if (case_sep == llvm::StringRef::npos) {
        cursor = llvm::StringRef();
      } else {
        cursor = cursor.drop_front(case_sep + 1);
      }

      auto pair_sep = token.find(':');
      CARBON_CHECK(pair_sep != llvm::StringRef::npos,
                   "FormatInt requires a `:` separating each comparison and "
                   "output string: `{0}`",
                   style);

      auto comp = token.take_front(pair_sep);
      auto output_string = token.drop_front(pair_sep + 1);

      if (comp.empty()) {
        // Default case.
        CARBON_CHECK(cursor.empty(),
                     "FormatInt requires the default case be last: `{0}`",
                     style);
        out << output_string;
        return;
      } else if (comp.consume_front("=")) {
        // Equality comparison.
        int value;
        CARBON_CHECK(to_integer(comp, value),
                     "FormatInt has invalid value in comparison: `{0}`", style);
        if (value == wrapper.value) {
          out << output_string;
          return;
        }
      } else {
        CARBON_FATAL("FormatInt has unrecognized comparison: `{0}`", style);
      }
    }

    CARBON_FATAL("FormatInt doesn't handle `{0}`: `{1}`", wrapper.value, style);
  }
};

#endif  // CARBON_TOOLCHAIN_DIAGNOSTICS_FORMAT_PROVIDERS_H_
