// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/default.carbon
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/destroy.carbon
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/form.carbon

// --- min_prelude/parts/as.carbon

package Core library "prelude/parts/as";

export import library "prelude/parts/default";
export import library "prelude/parts/destroy";
export import library "prelude/parts/form";

interface UnsafeAs(Dest:! type) {
  fn Convert[self: Self]() -> Dest;
}

interface As(Dest:! type) {
  // TODO: extend UnsafeAs(Dest);
  fn Convert[self: Self]() -> Dest;
}

interface ImplicitAs(Dest:! type) {
  // TODO: extend As(Dest);
  fn Convert[self: Self]() -> Dest;
}

interface BitAndWith(Other:! type) {
  fn Op[self: Self](other: Other) -> Self;
}

impl type as BitAndWith(type) {
  fn Op[self: Self](other: Self) -> Self = "type.and";
}
