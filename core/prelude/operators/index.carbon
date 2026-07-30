// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package Core library "prelude/operators/index";

export import library "prelude/copy";

interface IndexWith(SubscriptType: type) {
  let ElementType: type;
  fn At(self, subscript: SubscriptType) -> ElementType;
}

interface IndirectIndexWith(SubscriptType: type) {
  let ElementType: type;
  fn Addr(self, subscript: SubscriptType) -> ElementType*;
}

final impl forall [
        SubscriptType: type,
        T: IndirectIndexWith(SubscriptType)
        where .ElementType impls Copy
    ] T as IndexWith(SubscriptType)
        where .ElementType = T.ElementType {
  fn At(self, subscript: SubscriptType) -> ElementType {
    return *self.Addr(subscript);
  }
}
