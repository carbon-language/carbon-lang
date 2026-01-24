// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/copy.carbon

// --- min_prelude/parts/bool.carbon

package Core library "prelude/parts/bool";

export import library "prelude/parts/copy";

fn Bool() -> type = "bool.make_type";
