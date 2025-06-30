// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --custom-core --exclude-dump-file-prefix=min_prelude/
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/destroy.carbon

// --- min_prelude/destroy.carbon

// A minimal prelude for tests that require destruction.
package Core library "prelude";

export import library "prelude/parts/destroy";
