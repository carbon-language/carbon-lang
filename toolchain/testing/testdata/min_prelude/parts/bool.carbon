// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/copy.carbon
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/default.carbon
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/form.carbon

// --- min_prelude/parts/bool.carbon

package Core library "prelude/parts/bool";

export import library "prelude/parts/copy";
export import library "prelude/parts/default";
export import library "prelude/parts/form";

private fn MakeBool() -> type = "bool.make_type";
alias Bool = MakeBool();
