// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package Core library "prelude/operators/deref";

// TODO: Align with https://docs.carbon-lang.dev/docs/design/values.html#dereferencing-customization.
interface CppUnsafeDeref {
  let Result:! type;
  fn Op(self) -> Result;
}
