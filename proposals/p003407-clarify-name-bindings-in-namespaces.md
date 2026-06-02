# Clarify name bindings in namespaces.

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/3407)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Allow prefixing a tuple binding pattern with a namespace](#allow-prefixing-a-tuple-binding-pattern-with-a-namespace)
    -   [Allow binding patterns to declare names in multiple namespaces](#allow-binding-patterns-to-declare-names-in-multiple-namespaces)
    -   [Allow declaring names in namespaces not owned by the current scope](#allow-declaring-names-in-namespaces-not-owned-by-the-current-scope)
    -   [Allow declaring namespaces in scopes other than the file scope](#allow-declaring-namespaces-in-scopes-other-than-the-file-scope)

<!-- tocstop -->

## Abstract

-   Require namespace members be declared in the same name scope as the
    namespace is declared.
    -   Declaring namespaces outside file scope is disallowed, so this means
        only at file scope.
-   Allow binding patterns to directly declare names in namespaces.
-   Disallow introducing bindings in different namespaces in the same pattern.

## Problem

While the trivial case of `class NS.C` seems to be supported as a consequence of
proposal
[#107: Code and name organization](https://github.com/carbon-language/carbon-lang/pull/107),
it lacks detail. For example, there are a couple syntactic options when binding
multiple names.

Also, there's no clear decision around code such as:

```carbon
namespace NS;
class ClassT {
  // Is this a class member accessed through `NS`, or a file scope member inside
  // `NS`? What is its lifetime?
  var NS.a: i32 = 0;
}
```

This proposal mainly aims to remove ambiguities.

## Background

Namespaces are covered in
[code and name organization](/docs/design/code_and_name_organization/#namespaces).
Binding patterns are covered in
[pattern matching](/docs/design/pattern_matching.md#binding-patterns).

There's some discussion of `var` in
[values, variables, and pointers](/docs/design/values.md), but it's specific to
locals. That doesn't address other use cases, such as globals or member
variables.

## Proposal

When used to declare names in binding patterns, as in `var` or `let`, all names
must be in the same namespace. `namespace` members must be declared from within
the same scope that declared the `namespace`.

There was uncertainty about whether namespaces could be declared outside of file
scopes; for now, disallow it.

See the changes to
[code and name organization](/docs/design/code_and_name_organization/#namespaces)
for reference.

## Rationale

-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)

    -   Requiring that declarations of multiple names use `NS.a` syntax is
        consistent with the single variable case.
    -   Requiring namespace members be declared while in the same name scope as
        the namespace itself makes lifetimes clearer.

## Alternatives considered

### Allow prefixing a tuple binding pattern with a namespace

We could use the namespace to prefix the binding tuple. For example:

```carbon
var NS.(a: i32, b: i32) = (3, 4);
```

It's rare that we would have a single statement declare multiple names. As a
consequence, the separation of the namespace qualifier from the declared
identifier might end up unique to this syntax. In that context, we prefer `NS.a`
for consistency with other cases, such as `class NS.class`.

### Allow binding patterns to declare names in multiple namespaces

We could allow binding patterns to declare names in multiple namespaces. For
example:

```carbon
namespace NS;
var (NS.a: i32, b: i32) = InitData();
```

Mixing namespaces could be confusing: for example, `b` could be misunderstood to
be declared in `NS`. We lack data that would demonstrate benefits to offset
that.

We disallow mixing namespaces in a single declaration for simplicity.

### Allow declaring names in namespaces not owned by the current scope

We could allow declaring names in namespaces not owned by the current scope. For
example:

```carbon
namespace NS;
class ClassT {
  var NS.val: i32;

  class NS.ChildT {}
}
```

Here, `package.NS.val` would be a global, but `ClassT.NS.val` looks more like an
instance member. It's also unclear whether `ClassT.NS.val` (or
`instance.NS.val`) could be used to reference the produced variable, since `NS`
is not inside `ClassT`'s name scope. The naming problems extend to non-binding
declarations such as `NS.ChildT`, too.

Disallowing using namespaces to cross name scopes is consistent with rules that
generally prevent declaring names in other name scopes, such as:

```carbon
class A {
    class B {
        // `C` must be declared directly inside `A`.
        class A.C;
    }
}

// `D` must be declared within `A`, even if separately defined.
class A.D {}
```

Both the `namespace` declaration and names declared within the `namespace` must
be written in the same name scope. This avoids name lookup ambiguities, and
builds consistency in name scope boundaries across declarations.

### Allow declaring namespaces in scopes other than the file scope

We could allow declaring namespaces in scopes other than the file scope. It's
ambiguous what path should have resulted from proposal
[#107: Code and name organization](https://github.com/carbon-language/carbon-lang/pull/107),
although examples all focus on file scope, so other scopes weren't carefully
considered.

It might prove useful in some situations. For example, perhaps a complex class
would find a member namespace useful:

```carbon
class Complex {
   namespace OptionSet1;
   class OptionSet1.MemberClassA;
   class OptionSet1.MemberClassB;

   namespace OptionSet2;
   class OptionSet2.MemberClassC;
   class OptionSet2.MemberClassD;

   namespace Vars;
   var Vars.a;
}
```

This proposal takes a stance against declaring namespaces other than the file
scope because:

-   Proposal
    [#107: Code and name organization](https://github.com/carbon-language/carbon-lang/pull/107)
    only mentions file scope namespaces, implicitly disallowing it in other
    scopes.
-   Disallowing namespaces in other scopes is consistent with C++.

If allowed, it would be necessary to decide whether `Complex.Vars.a` would have
instance or global lifetime.

For now, namespaces may only be declared at file scope, which gives consistency
with C++. This decision may be reevaluated in a future proposal.
