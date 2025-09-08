// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package Core library "prelude/types/maybe_unformed";

import library "prelude/destroy";

private fn MakeMaybeUnformed(t: type) -> type = "maybe_unformed.make_type";

class MaybeUnformed(T:! type) {
  adapt MakeMaybeUnformed(T);
}
