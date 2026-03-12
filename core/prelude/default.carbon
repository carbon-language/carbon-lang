// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package Core library "prelude/default";

// Provides the default value of an object. If implemented for a type `T`, this
// is used to initialize declarations without an explicit initializer, such as
// `var x: T;`, and leaves them in a fully-formed state.
interface Default { fn Op() -> Self; }

// Indicates that a type permits unformed initialization, which leaves the
// object in a state where calling the destructor is valid but optional, and no
// other operations on the object except for reinitialization are permitted.
interface UnformedInit {
  // TODO: This should probably be:
  //   let StructT:! type;
  //   fn Op() -> StructT;
  // and should be able to initialize a subset of the fields. For now we always
  // leave the object uninitialized when it is in an unformed state.
  // See https://github.com/carbon-language/carbon-lang/pull/5913
}

// Provides a default, possibly unformed, value of an object. This should not be
// implemented directly. Instead, implement `Default` to provide a fully-formed
// state or (eventually) `UnformedInit` to provide an unformed state.
interface DefaultOrUnformed {
  // TODO: This should return `MaybeUnformed(Self)` once that is supported.
  fn Op() -> Self;
}

final impl forall [T:! Default] T as DefaultOrUnformed {
  fn Op() -> Self {
    return T.Op();
  }
}

impl forall [T:! UnformedInit] T as DefaultOrUnformed {
  fn Op() -> Self = "make_uninitialized";
}
