# Clang API usage survey

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Different uses of Clang's APIs](#different-uses-of-clangs-apis)
    -   [Clang](#clang)
    -   [Swift](#swift)
    -   [LLDB](#lldb)
    -   [Carbon previously](#carbon-previously)
    -   [Carbon approach](#carbon-approach)
-   [Alternatives considered](#alternatives-considered)

<!-- tocstop -->

## Overview

Clang performs the equivalent of Carbon's `lower` progressively, interleaved
with parsing/semantic analysis. This is in conflict with Carbon's phase-based
approach and leads to bugs in missing functionality in Clang's generated IR
during Carbon/C++ interop.

We analyze different uses of Clang's APIs to better understand the tradeoffs
between them, and propose a direction for Carbon.

## Different uses of Clang's APIs

There are several different users of Clang's APIs that take different approaches
to address their needs (needs more or less similar to Carbon's), below is a
survey of those approaches:

### Clang

[`clang::CodeGeneratorImpl`](https://github.com/llvm/llvm-project/blob/b2880eac7c09c1f3238d77c5a3356451178d7b8e/clang/lib/CodeGen/ModuleBuilder.cpp#L34)
is registered during clang’s parsing/semantic analysis, receiving callbacks
during that process rather than in a batch phase afterwards. Specifically, the
`CodeGeneratorImpl` has several virtual function callbacks that handle various
features. I tested clang by disabling or asserting in the various callbacks to
identify tests/examples that rely on the callback. I was then able to verify
that Carbon had missing functionality due to not implementing the callback using
a test like this:

`test.cpp`

```cpp
// some code
```

`test.carbon`

```carbon
import Cpp inline '''
#include "test.cpp"
''';
```

```shell
$ diff <(clang++ test.cpp -c && nm test.o) \
       <(carbon compile interop.carbon --optimize=none && nm interop.o)
```

Any difference should be a bug in the Carbon compiler's interop support. Here
are some (non-exhaustive) examples I found, based on different callbacks in the
`ASTConsumer` API:

-   `HandleCXXStaticMemberVarInstantiation` handles instantiating C++ static
    member variables in template contexts like this:

```cpp
template<typename T>struct t3 {
  static int i;
};
template<typename T>int t3<T>::i;
void f1() {
  // Without the callback, t3<int>::i is not emitted.  t3<int>::i = 42;
}
```

-   `HandleTopLevelDecl` this is the main callback that handles each top level
    (nested only within namespaces \- not within another class or function)
    declarations for code generation
-   `EmitDeferredDecls`\+`HandleInlineFunctionDefinition` for emitting inline
    function definitions in certain situations, like this:

```cpp
struct t2 {
  // Without the callback, `func`'s definition is not emitted.
  __attribute__((used)) void func() {}
};
```

-   `HandleTagDeclDefinition` updates types in the IR when a definition is
    provided later (not relevant to Carbon or Swift since they only generate the
    IR once the AST is complete anyway), eg:

```cpp
struct S;
extern S a[10];
S(*b)[10] = &a;
struct S {
  int x;
};
// Without the callback, this code still compiles,
// but uses a gep over a raw byte array, whereas
// with the callback it uses a gep over the `struct S` type.
int f() { return a[3].x; }
```

-   `HandleTagDeclRequiredDefinition` seems to be just for Microsoft debug info.
-   `HandleTranslationUnit` handles finishing things up after the translation
    unit \- Carbon can call this & get the same behavior.
-   `AssignInheritanceModel` related to the Microsoft inheritance attribute for.
-   `CompleteTentativeDefinition` seems to be only relevant to C code, not C++.
-   `CompleteExternalDeclaration` seems to be only relevant to the BPF target.
-   `HandleVTable` emits vtables as needed, eg:

```cpp
struct t1 {
  virtual void f1();
};
// Without the callback, the vtable is not emitted despite
// the appearance of this key function definition.
void t1::f1() { }
```

### Swift

-   [Swift supports C++ \-\> Swift interop](https://www.swift.org/documentation/cxx-interop/#exposing-swift-apis-to-c)
    by generating a Swift library with a matched `MyLib-Swift.h` header file,
    unmodified Clang can then parse that header for calling into the Swift
    library
-   Swift-\>C++ can’t do template instantiation, only interacting with class
    templates already instantiated in C++ with C++ parameters \- so nothing like
    Carbon’s closer interop (that already allows new instantiations from Carbon
    using C++ types as parameters, and will allow instantiating a C++ type with
    a Carbon type as a parameter)
-   Swift constructs the `clang::CodeGenerator` itself (rather than by way of
    `clang`\-the-compiler-like use) in `swift::IRGenModule`
    -   Dealing with the inline function problem, Swift uses
        [`IRGenModule::emitClangDecl`](https://github.com/swiftlang/swift/blob/6d4c516a32a597f5a06f021363ac0d6ab4c5adc5/lib/IRGen/GenClangDecl.cpp#L204)
        whenever it needs a decl from Clang for a call from Swift.
    -   It recurses through decls in the decl that swift requires searching for
        other decls that might need to be emitted \- this search is all done in
        Swift’s `IRGenModule::emitClangDecl`.
    -   Ultimately any decls found by way of this recursion are passed to
        `clang::CodeGenerator::HandleTopLevelDecl`

### LLDB

-   `clang::ParseAST` with the `clang::CodeGenerator` already registered
    -   looks basically like Clang, doesn’t need to separate parsing from IRGen,
        so it doesn’t have the problems Carbon and Swift do

### Carbon previously

-   `check` used a [Clang Tooling](https://clang.llvm.org/docs/LibTooling.html)
    based API, `clang::tooling::buildASTFromCodeWithArgs`
-   `check` does things to the AST, trigger template instantiation, etc
-   `lower` uses `clang::CreateLLVMCodeGen` to create a code generator for the
    AST
-   Carbon handles passing ASTs to this `clang::CodeGenerator`
-   Limitations have been partially addressed by
    [PR6237](https://github.com/carbon-language/carbon-lang/pull/6237)
    -   Clang’s Sema does at least keep a list of top level decls that need to
        be visited by the code generator, so this solves the inline-calls-inline
        situation by essentially replaying the `HandleTopLevelDecl` callback.
    -   Expected that the clang\<\>carbon divergence is still an outstanding
        risk.
    -   This work laid more of a foundation for doing something like PR5543
        (keeping the clang::CodeGenerator attached through Sema/SemIR) but
        without the need for multithreading, because it’s effectively inlined
        the FrontendAction execution into Clang, which is relatively little
        code/risk of divergence. This ended up landing in
        [PR6569](https://github.com/carbon-language/carbon-lang/pull/6569)

### Carbon approach

Since PR5543, several changes (especially
[PR6237](https://github.com/carbon-language/carbon-lang/pull/6237)) have been
made to Clang for related but incremental reasons. This has resulted in what was
indivisible work that motivated PR5543's multithreading to be inlined into
Carbon.

With that code inlined, we're now able to address the underlying desire - have a
`clang::CodeGenerator` attached to Clang's Sema throughout
Carbon`s `check`phase - allowing Clang to lower as it does in the native`clang`
compilation. This avoids the divergence without the multithreading complexity.

The main cost is the inherent difference between Clang's continuous lowering and
Carbon's phase based lowering, though that seems to be an acceptable cost to
avoid friction trying to otherwise wedge Clang into Carbon's phase based
approach.

## Alternatives considered

-   [PR5543 More closely mimic the Clang compilation](/proposals/p6641.md#pr5543-more-closely-mimic-the-clang-compilation)
-   [Status Quo with Improvements](/proposals/p6641.md#status-quo-with-improvements)
-   [Upstream Clang Changes to use Phase Based Lowering](/proposals/p6641.md#upstream-clang-changes-to-use-phase-based-lowering)
