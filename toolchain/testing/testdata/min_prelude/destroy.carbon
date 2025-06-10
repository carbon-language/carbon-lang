// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --custom-core --exclude-dump-file-prefix=min_prelude/

// --- min_prelude/destroy.carbon

// A minimal prelude for tests that require destruction.
package Core library "prelude";

// TODO: Uncomment once the interface is used.
// interface Destroy {
//   fn Op[self: Self]();
// }
// impl forall [T:! type] T as Destroy {
//   fn Op[self: Self]() = "no_op";
// }
