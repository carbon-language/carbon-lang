// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// --- min_prelude/parts/float.carbon

package Core library "prelude/parts/float";

export import library "prelude/parts/int_literal";

fn Float(size: IntLiteral()) -> type = "float.make_type";
