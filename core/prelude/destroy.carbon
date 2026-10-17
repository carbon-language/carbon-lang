// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
package Core library "prelude/destroy";

import library "prelude/operators/as";
import library "prelude/operators/bitwise";

interface SubobjectDestroy {
  fn Op(ref self);
}

// TODO: rename to `Destroy` when `require impls` adds a witness table entry.
interface DestroyImpl {
  // TODO: uncomment when `require impls` adds a witness table entry.
  // require impls SubobjectDestroy;

  // TODO: change `Self` to `partial Self`.
  fn Op(ref self);
}

impl forall [T: SubobjectDestroy] T as DestroyImpl {
  fn Op(ref self) = "no_op";
}

alias Destroy = SubobjectDestroy & DestroyImpl;

// TODO: uncomment when `require impls` adds a witness table entry.
// impl forall [T: SubobjectDestroy] partial T as Destroy {
//   alias Op = T.Op;
// }

fn SelfDestruct[T: Destroy](ref t: T) {
    t.(DestroyImpl.Op)();
    t.(SubobjectDestroy.Op)();
}

interface TrivialDestroy {}
