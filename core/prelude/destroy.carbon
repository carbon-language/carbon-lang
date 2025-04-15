// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package Core library "prelude/destroy";

// Object destruction, including running `fn destroy()`. Note this is distinct
// from memory deallocation.
interface Destroy {
  fn Op[self: Self]();
}

// Provide a default blanket impl for trivial destruction.
impl forall [T:! type] T as Destroy {
  fn Op[self: Self]() = "no_op";
}
