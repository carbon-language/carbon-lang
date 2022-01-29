<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

The code in this directory defines the AST that represents Carbon code in the
rest of `executable-semantics`.

All node types in the AST are derived from [`AstNode`](ast_node.h), and use
[LLVM-style RTTI](https://llvm.org/docs/HowToSetUpLLVMStyleRTTI.html) to support
safe down-casting and similar operations. Each abstract class `Foo` in the
hierarchy has a `kind` method which returns a enum `FooKind` that identifies the
concrete type of the object, and a `FooKind` value can be safely `static_cast`ed
to `BarKind` if that value represents a type that's derived from both `Foo` and
`Bar`.

We rely on code generation to help enforce those invariants, so every node type
must be described in [`ast_rtti.txt`](ast_rtti.txt). See the documentation in
(`gen_rtti.py`)[../gen_rtti.py], the code generation script, for details about
the file format and generated code.

The AST class hierarchy is structured in a fairly unsurprising way, with
abstract classes such as `Statement` and `Expression`, and concrete classes
representing individual syntactic constructs, such as `If` for if-statements.

Sometimes it is useful to work with a subset of node types that "cuts across"
the primary class hierarchy. Rather than deal with the pitfalls of multiple
inheritance, we handle these cases using a form of type erasure: we specify a
notional interface that those types conform to, and then define a "view" class
that behaves like a pointer to an instance of that interface. Types declare that
they model an interface `Foo` by defining a public static member named
`ImplementsCarbonFoo`. See [NamedEntityView](static_scope.h) for an example of
this pattern.
