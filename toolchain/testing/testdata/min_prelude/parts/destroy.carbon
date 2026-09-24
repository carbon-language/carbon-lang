// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// --- min_prelude/parts/destroy.carbon

package Core library "prelude/parts/destroy";

// TODO: uncomment when `require impls` adds a witness table entry.
// interface SubobjectDestroy {
//   final fn Op(ref self) = "subobject.destroy";
// }

interface Destroy {
  // TODO: uncomment when `require impls` adds a witness table entry.
  // require impls SubobjectDestroy;

  // TODO: change `Self` to `partial Self`.
  fn Op(ref self);

  // TODO: remove when `require impls` adds a witness table entry.
  final fn SubobjectDestroy(unused ref self) {
    // This function body is ignored by the toolchain. We can't use
    // a built-in because the compiler treats it as an error. We
    // should address this at some point, but it's not a high
    // priority right now.
  }

  final fn SelfDestruct(ref self) {
    self.Op();
    self.SubobjectDestroy();
  }
}

// TODO: uncomment when `require impls` adds a witness table entry.
// impl forall [T: SubobjectDestroy] partial T as Destroy {
//   alias Op = T.Op;
// }
