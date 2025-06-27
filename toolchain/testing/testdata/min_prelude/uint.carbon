// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/int.carbon
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/uint.carbon
// EXTRA-ARGS: --custom-core --exclude-dump-file-prefix=min_prelude/

// --- min_prelude/uint.carbon

// A minimal prelude for testing using `UInt` or `u32`, and includes `Int` and
// `i32` as well.
package Core library "prelude";

export import library "prelude/parts/int";
export import library "prelude/parts/uint";
