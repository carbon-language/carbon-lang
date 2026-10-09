// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
package Core library "prelude/destroy";

import library "prelude/operators/as";
import library "prelude/operators/bitwise";

// TODO: uncomment when `require impls` adds a witness table entry.
interface SubobjectDestroy {
  fn Op(ref self);
}

interface Destroy {
  // TODO: uncomment when `require impls` adds a witness table entry.
  // require impls SubobjectDestroy;

  // TODO: change `Self` to `partial Self`.
  fn Op(ref self);
}

// TODO: uncomment when `require impls` adds a witness table entry.
// impl forall [T: SubobjectDestroy] partial T as Destroy {
//   alias Op = T.Op;
// }

fn SelfDestruct[T: SubobjectDestroy & Destroy](ref t: T) {
    t.(Destroy.Op)();
    t.(SubobjectDestroy.Op)();
}
