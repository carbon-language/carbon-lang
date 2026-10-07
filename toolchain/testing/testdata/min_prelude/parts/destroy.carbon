// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/as.carbon

// --- min_prelude/parts/destroy.carbon

package Core library "prelude/parts/destroy";

export import library "prelude/parts/as";

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

// TODO: move `SelfDestruct.Op` to `Destroy.SelfDestruct`
// when `require impls` adds a witness table entry.
interface SelfDestruct {
  fn Op(ref self);
}

impl forall [T: SubobjectDestroy & Destroy] T as SelfDestruct {
  fn Op(ref self) {
    self.(Destroy.Op)();
    self.(SubobjectDestroy.Op)();
  }
}
