// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/bool.carbon
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/float.carbon
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/int.carbon
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/string.carbon
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/uint.carbon
// EXTRA-ARGS: --custom-core --exclude-dump-file-prefix=min_prelude/

// --- min_prelude/primitives.carbon

// A minimal prelude for testing with primitive types.
package Core library "prelude";

export import library "prelude/parts/bool";
export import library "prelude/parts/float";
export import library "prelude/parts/int";
export import library "prelude/parts/string";
export import library "prelude/parts/uint";
