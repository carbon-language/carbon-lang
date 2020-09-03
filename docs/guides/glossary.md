# Glossary

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## entity

An _entity_ is a named item with an associated name path, such as a function,
type, interface, or namespace. For example, in `fn GetTime()`, `GetTime` refers
to an entity which is a function.

## identifier

An _identifier_ is the token which names an entity, and is also used in code to
refer to the entity. For example, in `fn GetTime()`, `GetTime` is the identifier
for the function.

## library

A _library_ is a group of files that form an importable API and its
implementation. Carbon encourages small libraries, bundled into larger packages.
For example, given `package Geometry library Shapes;`, `Shapes` is a library in
the `Geometry` package.

## name path

A _name path_ is the dot-separated identifier list that indicates a relative or
full path of a name. For example, given `fn GetArea(var Geometry.Circle: x)`,
`Geometry.Circle` is a name path. `GetArea` is also a name path, albeit with
only one identifier needed.

## namespace

A _namespace_ is a entity that contains entities, and may be nested. For
example, given a name path of `Geometry.Circle`, `Geometry` is a namespace
containing `Circle`.

## package

A _package_ is a group of libraries in Carbon, and is the standard unit for
distribution. The package name also serves as the root namespace for all name
paths in its libraries. The package name should be a single, globally-unique
identifier. For example, given `package Geometry;` in a file, `Geometry` is the
package and root namespace.
