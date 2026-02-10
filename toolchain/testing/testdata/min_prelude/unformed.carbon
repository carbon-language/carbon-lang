// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/as.carbon
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/maybe_unformed.carbon
// EXTRA-ARGS: --custom-core --exclude-dump-file-prefix=min_prelude/

// --- min_prelude/uint.carbon

// A minimal prelude for test involving `Core.MaybeUnformed(T)`.
package Core library "prelude";

export import library "prelude/parts/as";
export import library "prelude/parts/maybe_unformed";
