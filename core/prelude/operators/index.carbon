// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package Core library "prelude/operators/index";

interface IndexWith(SubscriptType:! type, ElementType:! type) {
  fn At[self: Self](subscript: SubscriptType) -> ElementType;
}
