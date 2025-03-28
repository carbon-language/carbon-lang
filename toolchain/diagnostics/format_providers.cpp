// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/diagnostics/format_providers.h"

#include "common/check.h"
#include "llvm/ADT/StringExtras.h"

auto llvm::format_provider<Carbon::Diagnostics::BoolAsSelect>::format(
    const Carbon::Diagnostics::BoolAsSelect& wrapper, raw_ostream& out,
    StringRef style) -> void {
  if (style.empty()) {
    llvm::format_provider<bool>::format(wrapper.value, out, style);
    return;
  }

  auto sep = style.find('|');
  CARBON_CHECK(
      sep != llvm::StringRef::npos,
      "BoolAsSelect requires a `|` separating true and false results: `{0}`",
      style);
  CARBON_CHECK(style.find('|', sep + 1) == llvm::StringRef::npos,
               "BoolAsSelect only allows one `|`: `{0}`", style);

  if (wrapper.value) {
    out << style.take_front(sep);
  } else {
    out << style.drop_front(sep + 1);
  }
}

auto llvm::format_provider<Carbon::Diagnostics::IntAsSelect>::format(
    const Carbon::Diagnostics::IntAsSelect& wrapper, raw_ostream& out,
    StringRef style) -> void {
  if (style == "s") {
    if (wrapper.value != 1) {
      out << "s";
    }
    return;
  } else if (style.empty()) {
    llvm::format_provider<int>::format(wrapper.value, out, style);
    return;
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
                 "IntAsSelect requires a `:` separating each comparison and "
                 "output string: `{0}`",
                 style);

    auto comp = token.take_front(pair_sep);
    auto output_string = token.drop_front(pair_sep + 1);

    if (comp.empty()) {
      // Default case.
      CARBON_CHECK(cursor.empty(),
                   "IntAsSelect requires the default case be last: `{0}`",
                   style);
      out << output_string;
      return;
    } else if (comp.consume_front("=")) {
      // Equality comparison.
      int value;
      CARBON_CHECK(to_integer(comp, value),
                   "IntAsSelect has invalid value in comparison: `{0}`", style);
      if (value == wrapper.value) {
        out << output_string;
        return;
      }
    } else {
      CARBON_FATAL("IntAsSelect has unrecognized comparison: `{0}`", style);
    }
  }

  CARBON_FATAL("IntAsSelect doesn't handle `{0}`: `{1}`", wrapper.value, style);
}
