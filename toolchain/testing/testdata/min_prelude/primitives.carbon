// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --custom-core --exclude-dump-file-prefix=min_prelude/
// TODO: This should move to the place where it's imported, but it ends up
// included more than once and that fails to build.
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/as.carbon
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/bool.carbon
// TODO: This should move to the place where it's imported, but it ends up
// included more than once and that fails to build.
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/float.carbon
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/int_literal.carbon
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/int.carbon
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/uint.carbon

// --- min_prelude/primitives.carbon

// A minimal prelude for testing with primitive types.
package Core library "prelude";

export import library "prelude/parts/bool";
export import library "prelude/parts/float";
export import library "prelude/parts/int";
export import library "prelude/parts/uint";
