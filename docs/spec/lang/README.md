# Carbon language specification

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Program structure

1.  A _program_ is a collection of one or more linkage units that are
    [linked](#linkage) together.

2.  A _Carbon linkage unit_ is the result of [translating](#translation) a
    source file. A _foreign linkage unit_ is an artifact produced by a
    translation process for some other programming language. A linkage unit is
    either a Carbon linkage unit or a foreign linkage unit.

3.  A _source file_ is a sequence of Unicode code points.

    > Note: Source files are typically stored on disk in files with a `.carbon`
    > file extension, encoded in UTF-8.

## Conformance

1.  A program is _valid_ if it contains no constructs that violate "shall"
    constraints in this specification. Otherwise, the program is _invalid_.

2.  An implementation is _conforming_ if it accepts all valid programs, it
    rejects all invalid programs for which a diagnostic is required, and the
    [execution](execution.md) semantics of all accepted programs is as specified
    in this specification.

## Translation

1.  Translation of a source file into a Carbon linkage unit proceeds as follows:

    -   [Lexical analysis](lex.md) decomposes the sequence of code points into a
        sequence of lexical elements.
    -   Whitespace and text comments are discarded, leaving a sequence of
        [tokens](lex.md).
    -   The tokens are [parsed](parsing.md) into an abstract syntax tree.
    -   [Unqualified names are bound](names.md) to declarations in the abstract
        syntax tree.
    -   A translated form of each imported [library](libs.md) is located and
        loaded.
    -   [Semantic analysis](semantics.md) is performed: types are determined and
        semantic checks are performed for all non-template-dependent constructs
        in the abstract syntax tree, constant expressions are evaluated, and
        templates are instantiated and semantically analyzed.

2.  > Note: After semantic analysis, an implementation may optionally
    > monomorphize generics by a process similar to template instantiation.

3.  The resulting linkage unit comprises all entities in the translated source
    file that are either [external](#linkage) or are reachable from an external
    entity.

    > Note: A linkage unit can include non-monomorphized generics, but never
    > includes templates. Constant evaluation can eliminate references to
    > entities.

## Linkage

1.  Two declarations declare the same entity if both declarations are in the
    same library and the same [scope](names.md#scopes) and declare the same
    [name](names.md).

    TODO: Linkage rules for foreign entities. TODO: Ability to declare
    file-local entities.

2.  All declarations of an entity shall use the same type.

3.  Every entity that is reachable from a linkage unit in a program shall be
    defined by a linkage unit in the program; no diagnostic is required unless
    an entity that can be referenced during the [execution](execution.md) of the
    program is not defined.

4.  There shall not be more than one definition of an entity in a program.
