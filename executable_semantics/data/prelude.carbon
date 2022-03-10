// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package Carbon api;

// Note that Print is experimental, and not part of an accepted proposal, but
// is included here for printing state in tests.
fn Print(format_str: String) {
  __intrinsic_print(format_str);
}
