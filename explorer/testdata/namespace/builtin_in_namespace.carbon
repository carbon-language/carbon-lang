// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: result: 2

package ExplorerTest api;

namespace Namespace;

// This does not interfere with the builtin `AddWith`, nor with the declaration
// of `F` below.
interface Namespace.AddWith {}

fn F[T:! AddWith(i32) where .Result = i32](n: T) -> i32 { return n + 1; }

fn Main() -> i32 { return F(1); }
