// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/array.carbon
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/int.carbon
// EXTRA-ARGS: --custom-core --exclude-dump-file-prefix=min_prelude/

// --- min_prelude/array.carbon

// A minimal prelude for testing arrays. Also includes integer support since
// that's generally necessary when using arrays.
package Core library "prelude";

export import library "prelude/parts/array";
export import library "prelude/parts/int";
