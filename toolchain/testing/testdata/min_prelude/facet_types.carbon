// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/as.carbon
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/copy.carbon
// EXTRA-ARGS: --custom-core --exclude-dump-file-prefix=min_prelude/

// --- min_prelude/facet_types.carbon

// A minimal prelude for testing facet types, their use as generic types, and
// conversions to and from them.
package Core library "prelude";

export import library "prelude/parts/as";
export import library "prelude/parts/copy";

interface BitAndWith(Other:! type) {
  fn Op[self: Self](other: Other) -> Self;
}

impl type as BitAndWith(type) {
  fn Op[self: Self](other: Self) -> Self = "type.and";
}
