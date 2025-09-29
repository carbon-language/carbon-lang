// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// --- min_prelude/parts/destroy.carbon

package Core library "prelude/parts/destroy";

// TODO: Add `Destructor`, as in:
// interface Destructor {
//   private fn Op[ref self: Self]();
// }

// Destroys objects. Note this is distinct from memory deallocation.
interface Destroy {
  // TODO: This should be `final fn Op[ref self: Self]() = "type.destroy"`.
  fn Op[addr self: Self*]();
}

// True for all types that should have a `Destroy` impl.
private fn CanDestroy() -> type = "type.can_destroy";

// Destroys an instance of `DestroyT`. This is also used for trivial types.
final impl forall [DestroyT:! CanDestroy()] DestroyT as Destroy {
  fn Op[addr self: Self*]() = "type.destroy";
}
