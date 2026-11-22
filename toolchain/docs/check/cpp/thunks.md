<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# C++ Interoperability: Thunks

<!-- toc -->

## Table of contents

-   [Overview](#overview)
    -   [Simplified ABI](#simplified-abi)
    -   [Thunk inlining](#thunk-inlining)
    -   [Signature adaptation thunks](#signature-adaptation-thunks)
-   [Carbon calling C++](#carbon-calling-c)
-   [C++ calling Carbon](#c-calling-carbon)
-   [Carbon overriding C++ virtual functions](#carbon-overriding-c-virtual-functions)
    -   [What goes in the vtables](#what-goes-in-the-vtables)
    -   [Example of synthesized functions](#example-of-synthesized-functions)
    -   [Deferred thunk generation](#deferred-thunk-generation)

<!-- tocstop -->

## Overview

When C++ code calls Carbon functions, or when Carbon calls C++ code, we need to
be able to interoperate between the Carbon ABI and the C++ ABI. We don't want to
hardcode all the minutiae of the various C++ ABIs into the Carbon toolchain, so
instead we generate thunks with intentionally simple ABIs.

Example:

```carbon
inline Cpp '''
class X { ... };
X f(X x);
''';

fn G(x: Cpp.X) -> Cpp.X {
  return Cpp.f(x);
}
```

The calling conventions used to pass and return an `X` object in C++ are very
varied, and depend on various aspects of both the target and of the definition
of `X`. In order to call `f` from Carbon, we generate a thunk on the C++ side:

```cpp
__attribute__((always_inline))
inline void f__carbon_thunk__(void *result, void *x) {
  new (result) X(f(*static_cast<X*>(x)));
}
```

... and notionally create a second thunk on the Carbon side to call it:

```carbon
fn F:thunk(var x: Cpp.X) -> Cpp.X {
  returned var result: Cpp.X;
  Cpp.f__carbon_thunk__(&result, &x);
  return var;
}
```

### Simplified ABI

The simple ABI that we use for cross-language calls supports only the following:

-   Pointer and reference parameter and return types.
-   32- and 64-bit integer parameter and return types.
-   `void` return types.

We generate calls with these types by using the corresponding LLVM function
type. We assume that this matches the calling convention for these types on the
C++ side; in practice, it does for the ABIs that we care about.

### Thunk inlining

We aggressively inline thunks. This happens in two ways:

1.  Thunks are marked as "always inline" on both the C++ side and the Carbon
    side. In C++ this happens by adding an `always_inline` attribute in the AST;
    in Carbon it happens by emitting the LLVM `alwaysinline` attribute directly.
2.  When building SemIR, if we try to build a `call` instruction whose target is
    a Carbon-defined thunk, we instead inline the body of the thunk directly
    into the SemIR. This means that we can usually avoid the Carbon-side thunk
    entirely.

### Signature adaptation thunks

Carbon has another kind of thunk beyond those used for C++ interoperability.
When a function in an interface has a different signature than the corresponding
function in an impl, or a virtual function has a different signature than an
overrider, a thunk is generated to adapt the signature of the function. This
thunk simply implicitly converts each argument to the parameter type, and
implicitly converts the return value to the return type, with one exception: for
a virtual function, the `self` parameter is converted from the base class type
to the derived class type.

```carbon
base class A {
  virtual fn F(self, n: i32) -> i32;
}
class B {
  extend base: A;
  override fn F(self, n: i64) -> i16;
}

// Behaves as if this function is in the vtable:
fn B.OverrideF(self: A, n: i32) -> i32 {
  // Implicitly converts n from i32 to i64
  // Implicitly converts result from i16 to i32
  return (self unsafe as B).F(n);
}
```

These thunks are not the topic of this document, but understanding them is
important for understanding the behavior of
[Carbon overriders of C++ virtual functions](#carbon-overriding-c-virtual-functions).

## Carbon calling C++

If a C++ function already has a simple ABI, we don't generate a thunk, and
instead we call it directly. Otherwise, we generate a thunk as follows.

On the C++ side, we have two `clang::FunctionDecl`s:

-   The original callee.
-   The thunk with a simplified ABI, which is defined to call the original
    callee.

On the Carbon side, we have two `SemIR::Function`s:

-   The function representing the original C++ function signature. This is
    marked as `SpecialFunctionKind::HasCppThunk`. Attempts to call this function
    generate a call through the thunk instead. This is returned when Carbon
    invokes C++ overload resolution.
-   The function representing the C++ thunk. This is the target of SemIR `call`
    instructions, and is marked as `SpecialFunctionKind::CppThunk`. This has the
    same symbol name as the C++ thunk.

`Context::clang_decls` can be used to map between the corresponding C++ and
Carbon functions above, and `Function::cpp_thunk_decl_id` and
`Function::cpp_thunk_callee` can be used to map between the two
`SemIR::Function`s.

The `HasCppThunk` function on the Carbon side is only ever directly invoked. It
can't be placed into a witness table or a vtable. Therefore the thunk is
[always inlined](#thunk-inlining), and we never generate a Carbon-side
definition for it.

The full story is a little more involved than this: in order to support C++
default arguments (and some other call quirks), each C++ function can map to
multiple different `SemIR::Function`s with different Carbon-side signatures,
such as having different numbers of parameters. This leads to there being up to
2N `SemIR::Function`s per C++ function rather than only 2, where N is the number
of function variants in use.

## C++ calling Carbon

When C++ code calls into Carbon, we always generate a thunk on each side.

On the Carbon side, we have two `SemIR::Function`s:

-   The original callee.
-   The thunk with a simplified ABI, which is defined to call the original
    callee. This is marked as `SpecialFunctionKind::CppThunk`.

On the C++ side, we have two `clang::FunctionDecl`s:

-   A C++ function representing the original Carbon function signature. This
    function is defined in the C++ AST with a body that calls the thunk; from
    Clang's perspective this is a normal C++ function.
-   A C++ function representing the Carbon thunk. This has the same symbol name
    as the Carbon thunk.

`Context::clang_decls` can be used to map between the corresponding C++ and
Carbon functions above. `Function::cpp_thunk_callee` can be used to map from the
`CppThunk` to the original Carbon callee.

## Carbon overriding C++ virtual functions

When a Carbon class extends a C++ base class and overrides a C++ virtual
function, we generate a set of thunks in order to produce a virtual override
with the correct signature. Broadly we use a similar pattern to C++ calling
Carbon, but with extra complexity because we need to fine-tune the signature of
the C++ function, and we need to import it back into Carbon so it can be
referenced from the Carbon vtable representation.

Consider the following example of a Carbon class overriding a C++ virtual
function:

```carbon
import Cpp;

inline Cpp '''
struct Base {
  virtual void f(int x) = 0;
};
void CallBase(Base& b) { b.f(42); }
''';

class Derived {
  extend base: Cpp.Base;
  override fn f(self, x: i64) {
    ...
  }
}
```

This is implemented by combining three other kinds of thunk, as follows:

-   A declaration of the C++ function in the base class is imported into Carbon
    as a member of the derived class. This function is marked as being a
    [signature adaptation thunk](#signature-adaptation-thunks) for the Carbon
    overrider.
-   The signature adaptation thunk is [exported to C++](#c-calling-carbon). This
    generates a Carbon-side `CppThunk` definition that calls the signature
    adaptation thunk, and a C++-side definition.
-   The Carbon-side thunk's call to the signature adaptation thunk is inlined in
    SemIR.
-   The C++-side function definition uses the exact signature of the original
    C++ function. We know to do this because it is a thunk generated for a
    signature adaptation thunk whose signature is itself imported from C++.

In `clang_decls`, the signature adaptation thunk corresponds to the C++ virtual
override function.

We generate a signature adaptation thunk in Carbon so that any conversions
required in order to call the Carbon derived function from the base function
signature are performed using Carbon rules, not C++ rules.

### What goes in the vtables

We are concerned with two vtables:

-   The Carbon-side vtable representation. This is notionally the authoritative
    vtable, as the class `Derived` is a Carbon class, but in practice it is only
    used for constant evaluation on the Carbon side.
-   The C++-side vtable representation. In this case, because the vptr was
    originally introduced by a C++ class, this will be used for code generation.
    This is generated by Clang based on our exporting a suitable set of
    overriding functions when we export the Carbon class to C++.

The Carbon-side vtable contains the signature adaptation thunk. The C++-side
vtable contains the corresponding C++ virtual override function.

### Example of synthesized functions

In the C++ AST:

```cpp
namespace Carbon {
struct Derived : Base {
  // The C++ vtable entry for Derived::f.
  // Placed into `Derived`'s C++ vtable. When called from C++,
  // it forwards directly to the Carbon thunk.
  virtual void f(int x) override {
    // The C++ declaration of the Carbon-side thunk.
    extern void _Cf__carbon_thunk_Derived_Main(Base* self, int x);
    _Cf__carbon_thunk_Derived_Main(this, x);
  }
};
}
```

In Carbon SemIR:

```carbon
// The user-written Carbon override function.
fn Derived.f(self: Derived, x: i64) {
  ...
}

// Signature adaptation thunk.
fn Derived.OverrideF(self: Cpp.Base, x: i32) {
  // Explicitly converts `self` from `Base` to `Derived`.
  // Implicitly converts x from i32 to i64.
  (self unsafe as Derived).F(x);
}

// The synthesized Carbon thunk.
fn _Cf__carbon_thunk.Derived.Main(self_ptr: Base*, x: i32) {
  // Notionally: `Derived.OverrideF(*self_ptr, x)`, but call is inlined:
  (*self_ptr unsafe as Derived).F(x);
}
```

### Deferred thunk generation

We can't synthesize the definition of the C++-side virtual overrider until the
enclosing class is complete in the C++ AST. Therefore we split the
responsibility for generating the thunks in two:

-   When we complete the Carbon class, we generate the signature adaptation
    thunk.
-   When we form a corresponding complete C++ class type, we generate the
    C++-side virtual overrider thunk.
