// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// --- min_prelude/parts/destroy.carbon

package Core library "prelude/parts/destroy";

// TODO: Add `Destructor`, as in:
// interface Destructor {
//   private fn Op[ref self: Self]();
// }

// Destroys objects. This will invoke `Destructor` impls recursively on members;
// it does not deallocate memory.
interface Destroy {
  // TODO: This should be `final fn Op[ref self: Self]() = "type.destroy"`.
  fn Op[ref self: Self]();
}
