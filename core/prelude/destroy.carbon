// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package Core library "prelude/destroy";

import library "prelude/types/bool";
import library "prelude/types/int_literal";

// Object destruction, including running `fn destroy()`. Note this is distinct
// from memory deallocation.
interface Destroy {
  fn Op[addr self: Self*]();
}

// Add destruction for essential builtin types. This can't be done earlier
// because `Destroy` is using them.
// TODO: This should match trivial destruction.
impl bool as Destroy {
  fn Op[addr self: Self*]() = "no_op";
}
impl type as Destroy {
  fn Op[addr self: Self*]() = "no_op";
}
impl forall [PointeeT:! type] PointeeT* as Destroy {
  fn Op[addr self: Self*]() = "no_op";
}

// Handles builtin aggregate type destruction.
private fn CanAggregateDestroy() -> type = "type.can_aggregate_destroy";
impl forall [AggregateT:! CanAggregateDestroy()] AggregateT as Destroy {
  fn Op[addr self: Self*]() = "type.aggregate_destroy";
}

// `const` instances always use the non-`const` destructor.
impl forall [NonConstT:! Destroy] const NonConstT as Destroy {
  fn Op[addr self: Self*]() { (self unsafe as NonConstT*)->(Destroy.Op)(); }
}
