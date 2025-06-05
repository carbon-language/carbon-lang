// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --custom-core --exclude-dump-file-prefix=min_prelude/
// INCLUDE-FILE: toolchain/testing/min_prelude/parts/as.carbon

// --- min_prelude/convert.carbon

// A minimal prelude for testing conversion between types.
package Core library "prelude";

export import library "prelude/parts/as";
