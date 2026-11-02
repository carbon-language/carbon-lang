// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/bool.carbon
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/destroy.carbon
// EXTRA-ARGS: --custom-core --exclude-dump-file-prefix=min_prelude/

// --- min_prelude/bool.carbon

// A minimal prelude for testing conversion between types.
package Core library "prelude";

export import library "prelude/parts/bool";
export import library "prelude/parts/destroy";
