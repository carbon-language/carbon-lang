// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DIAGNOSTICS_FORMAT_PROVIDERS_H_
#define CARBON_TOOLCHAIN_DIAGNOSTICS_FORMAT_PROVIDERS_H_

#include "common/ostream.h"
#include "llvm/Support/FormatVariadicDetails.h"

namespace Carbon {

// Selects a formatv string based on the value. If the format style is not
// provided, as in `{0}`, the value uses standard formatting.
//
// When used, the true and false outputs are separated by a `|`.
//
// For example, `{0:true|false}` would yield standard bool formatting.
//
// If needed, the _full_ style string can be wrapped with `'` in order to
// preserve prefix or suffix whitespace (which is stripped by formatv). For
// example, `{0:' true | false '}` retains whitespace which would be dropped
// before `true` and after `false`.
struct BoolAsSelect {
  // NOLINTNEXTLINE(google-explicit-constructor)
  BoolAsSelect(bool value) : value(value) {}

  bool value;
};

// Selects a formatv string based on the value. If the format style is not
// provided, as in `{0}`, the value uses standard formatting.
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
struct IntAsSelect {
  // NOLINTNEXTLINE(google-explicit-constructor)
  IntAsSelect(int value) : value(value) {}

  int value;
};

}  // namespace Carbon

// See BoolAsSelect.
template <>
struct llvm::format_provider<Carbon::BoolAsSelect> {
  static auto format(const Carbon::BoolAsSelect& wrapper, raw_ostream& out,
                     StringRef style) -> void;
};

// See IntAsSelect.
template <>
struct llvm::format_provider<Carbon::IntAsSelect> {
  static auto format(const Carbon::IntAsSelect& wrapper, raw_ostream& out,
                     StringRef style) -> void;
};

#endif  // CARBON_TOOLCHAIN_DIAGNOSTICS_FORMAT_PROVIDERS_H_
