// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package Core library "prelude/types";

export import library "prelude/types/bool";
export import library "prelude/types/int_literal";
export import library "prelude/types/int";
export import library "prelude/types/uint";

fn Float(size: IntLiteral()) -> type = "float.make_type";
