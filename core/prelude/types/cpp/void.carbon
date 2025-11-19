// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package Core library "prelude/types/cpp/void";

namespace CppCompat;

// C++ `void` as a Carbon type.
//
// This type represents a notional "base-most" object at the root of every
// inheritance hierarchy. There are no objects of type `VoidType`, but any value
// can be converted to a value of type `VoidType`.
//
// This type is used as the default mapping for C++ `void`, which is used in
// most contexts in which `void` can appear: for example, as a pointee of a
// pointer type, as the type of a typedef, or as a type template argument.
// However, in a function signature, a `void` return type maps to `()` instead,
// and a `(void)` parameter list maps to `()`.
//
// This type is also available via the alias `Cpp.void`.
abstract class CppCompat.VoidBase {
  adapt ();
}

// TODO: An unqualified pointer to `T` should implicitly convert to a pointer to
// `<qualifiers> void` if `<qualifiers>` includes all of the qualifiers for `T`.
// Similarly, a value of type `T` should implicitly convert to a value of type
// `void`.
