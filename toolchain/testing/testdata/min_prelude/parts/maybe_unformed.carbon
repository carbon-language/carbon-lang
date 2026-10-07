// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/default.carbon
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/destroy.carbon
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/form.carbon

// --- min_prelude/parts/maybe_unformed.carbon

package Core library "prelude/parts/maybe_unformed";

export import library "prelude/parts/default";
export import library "prelude/parts/destroy";
export import library "prelude/parts/form";

private fn Make(t: type) -> type = "maybe_unformed.make_type";

class MaybeUnformed(T: type) {
  adapt Make(T);
}
