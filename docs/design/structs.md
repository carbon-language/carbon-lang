# Structs

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Table of contents

<!-- toc -->

- [TODO](#todo)
- [Overview](#overview)
- [Open questions](#open-questions)
  - [`self` type](#self-type)
  - [Default access control level](#default-access-control-level)

<!-- tocstop -->

## TODO

This is a skeletal design, added to support [the overview](README.md). It should
not be treated as accepted by the core team; rather, it is a placeholder until
we have more time to examine this detail. Please feel welcome to rewrite and
update as appropriate.

## Overview

Beyond simple tuples, Carbon of course allows defining named product types. This
is the primary mechanism for users to extend the Carbon type system and
fundamentally is deeply rooted in C++ and its history (C and Simula). We simply
call them `struct`s rather than other terms as it is both familiar to existing
programmers and accurately captures their essence: they are a mechanism for
structuring data:

```
struct Widget {
  var Int: x;
  var Int: y;
  var Int: z;

  var String: payload;
}
```

Most of the core features of structures from C++ remain present in Carbon, but
often using different syntax:

```
struct AdvancedWidget {
  // Do a thing!
  fn DoSomething(AdvancedWidget: self, Int: x, Int: y);

  // A nested type.
  struct Subtype {
    // ...
  }

  private var Int: x;
  private var Int: y;
}

fn Foo(AdvancedWidget: thing) {
  thing.DoSomething(1, 2);
}
```

Here we provide a public object method and two private data members. The method
explicitly indicates how the object parameter is passed to it, and there is no
automatic scoping - you have to use `self` here. The `self` name is also a
keyword, though, that explains how to invoke this method on an object. This
member function accepts the object _by value_, which is easily expressed here
along with other constraints on the object parameter. Private members work the
same as in C++, providing a layer of easy validation of the most basic interface
constraints.

The type itself is a compile-time constant value. All name access is done with
the `.` notation. Constant members (including member types and member functions
which do not need an implicit object parameter) can be accessed via that
constant: `AdvancedWidget.Subtype`. Other members and member functions needing
an implicit object parameter (or "methods") must be accessed from an object of
the type.

Some things in C++ are notably absent or orthogonally handled:

- No need for `static` functions, they simply don't accept an implicit object
  parameter.
- No `static` variables because there are no global variables. Instead, can have
  scoped constants.

## Open questions

### `self` type

Requiring the type of `self` makes method declarations quite verbose. Unclear
what is the best way to mitigate this, there are many options. One is to have a
special `Self` type.

It may be interesting to consider separating the `self` syntax from the rest of
the parameter pattern as it doesn't seem necessary to inject all of the special
rules (covariance vs. contravariance, special pointer handling) for `self` into
the general pattern matching system.

### Default access control level

The default access control level, and the options for access control, are
pretty large open questions. Swift and C++ (especially w/ modules) provide a lot
of options and a pretty wide space to explore here. If the default isn't right
most of the time, access control runs the risk of becoming a significant
ceremony burden that we may want to alleviate with grouped access regions
instead of per-entity specifiers. Grouped access regions have some other
advantages in terms of pulling the public interface into a specific area of the
type.
