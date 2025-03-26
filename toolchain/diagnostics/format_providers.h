// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DIAGNOSTICS_FORMAT_PROVIDERS_H_
#define CARBON_TOOLCHAIN_DIAGNOSTICS_FORMAT_PROVIDERS_H_

#include "common/ostream.h"
#include "llvm/Support/FormatVariadicDetails.h"

namespace Carbon::Diagnostics {

// Selects a formatv string based on the value.
//
// Supported format styles are:
// - None, as in `{0}`. This uses standard integer formatting.
// - Selector, as in `{0:true|false}`. The output string used is separated by a
//   `|`, with the true case first. the example would yield standard bool
//   formatting.
struct BoolAsSelect {
  // NOLINTNEXTLINE(google-explicit-constructor)
  BoolAsSelect(bool value) : value(value) {}

  bool value;
};

// Selects a formatv string based on the value.
//
// Supported format styles are:
// - None, as in `{0}`. This uses standard integer formatting.
// - Selector, as in `{0:=0:zero|:default}`. This is detailed below.
// - Plural `s`, as in `{0:s}`. This outputs an `s` when the value is not 1,
//   equivalent to `{0:=1:|:s}`.
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
struct IntAsSelect {
  // NOLINTNEXTLINE(google-explicit-constructor)
  IntAsSelect(int value) : value(value) {}

  int value;
};

}  // namespace Carbon::Diagnostics

// See Diagnostics::BoolAsSelect.
template <>
struct llvm::format_provider<Carbon::Diagnostics::BoolAsSelect> {
  static auto format(const Carbon::Diagnostics::BoolAsSelect& wrapper,
                     raw_ostream& out, StringRef style) -> void;
};

// See Diagnostics::IntAsSelect.
template <>
struct llvm::format_provider<Carbon::Diagnostics::IntAsSelect> {
  static auto format(const Carbon::Diagnostics::IntAsSelect& wrapper,
                     raw_ostream& out, StringRef style) -> void;
};

#endif  // CARBON_TOOLCHAIN_DIAGNOSTICS_FORMAT_PROVIDERS_H_
