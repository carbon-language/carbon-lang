// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package Core library "prelude/destroy";

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

// Returns a constraint that matches all types that should have a `Destroy` impl.
private fn CanDestroy() -> type = "type.can_destroy";

// Destroys an instance of `DestroyT`. This is also used for trivial types.
final impl forall [DestroyT:! CanDestroy()] DestroyT as Destroy {
  fn Op[ref self: Self]() = "type.destroy";
}
