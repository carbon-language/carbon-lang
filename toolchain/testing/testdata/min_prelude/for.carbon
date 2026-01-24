// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --custom-core --exclude-dump-file-prefix=min_prelude/
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/as.carbon
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/bool.carbon
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/copy.carbon
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/iterate.carbon
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/optional.carbon

// --- min_prelude/bool.carbon

// A minimal prelude for testing `for` loops.
package Core library "prelude";

export import library "prelude/parts/as";
export import library "prelude/parts/bool";
export import library "prelude/parts/copy";
export import library "prelude/parts/iterate";
export import library "prelude/parts/optional";
