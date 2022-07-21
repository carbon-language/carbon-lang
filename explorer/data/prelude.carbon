// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package Carbon api;

// Explicitly convert `Self` to `T`.
interface As(T:! Type) {
  fn Convert[me: Self]() -> T;
}

// Implicitly convert `Self` to `T`.
interface ImplicitAs(T:! Type) {
  fn Convert[me: Self]() -> T;
}

// TODO: ImplicitAs(T) should extend As(T).
impl forall [T:! Type, U:! ImplicitAs(T)] U as As(T) {
  fn Convert[me: Self]() -> T { return me.Convert(); }
}

// Every type implicitly converts to itself.
impl forall [T:! Type] T as ImplicitAs(T) {
  fn Convert[me: Self]() -> T { return me; }
}

// TODO: Simplify this once we have variadics.
// TODO: Should these be final?
impl forall [U1:! Type, T1:! ImplicitAs(U1)]
    (T1,) as ImplicitAs((U1,)) {
  fn Convert[me: Self]() -> (U1,) {
    let (v1: T1,) = me;
    return (v1.Convert(),);
  }
}
impl forall [U1:! Type, U2:! Type, T1:! ImplicitAs(U1), T2:! ImplicitAs(U2)]
    (T1, T2) as ImplicitAs((U1, U2)) {
  fn Convert[me: Self]() -> (U1, U2) {
    let (v1: T1, v2: T2) = me;
    return (v1.Convert(), v2.Convert());
  }
}
impl forall [U1:! Type, U2:! Type, U3:! Type,
             T1:! ImplicitAs(U1), T2:! ImplicitAs(U2), T3:! ImplicitAs(U3)]
    (T1, T2, T3) as ImplicitAs((U1, U2, U3)) {
  fn Convert[me: Self]() -> (U1, U2, U3) {
    let (v1: T1, v2: T2, v3: T3) = me;
    return (v1.Convert(), v2.Convert(), v3.Convert());
  }
}

// Note that Print is experimental, and not part of an accepted proposal, but
// is included here for printing state in tests.
// TODO: Remove Print special casing once we have variadics or overloads.
// fn Print(format_str: String) {
//   __intrinsic_print(format_str);
// }

class Heap {
  fn New[T:! Type, me: Self](x : T) -> T* {
    return __intrinsic_new(x);
  }
  fn Delete[T:! Type, me: Self](p : T*) {
    __intrinsic_delete(p);
  }
}

var heap: Heap = {};
