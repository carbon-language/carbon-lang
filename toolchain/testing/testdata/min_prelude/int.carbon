// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// TODO: This should move to the place where it's imported, but it ends up
// included more than once and that fails to build.
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/as.carbon
// TODO: This should move to the place where it's imported, but it ends up
// included more than once and that fails to build.
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/int_literal.carbon
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/int.carbon
// EXTRA-ARGS: --custom-core --exclude-dump-file-prefix=min_prelude/

// --- min_prelude/int.carbon

// A minimal prelude for testing using `Int` or `i32`; required for arrays.
package Core library "prelude";

export import library "prelude/parts/int";
