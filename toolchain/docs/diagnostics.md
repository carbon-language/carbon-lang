# Diagnostics

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [DiagnosticEmitter](#diagnosticemitter)
-   [DiagnosticConsumers](#diagnosticconsumers)
-   [Producing diagnostics](#producing-diagnostics)
-   [Diagnostic registry](#diagnostic-registry)
-   [CARBON_DIAGNOSTIC placement](#carbon_diagnostic-placement)
-   [Diagnostic context](#diagnostic-context)
-   [Diagnostic parameter types](#diagnostic-parameter-types)
-   [Diagnostic message style guide](#diagnostic-message-style-guide)

<!-- tocstop -->

## Overview

The diagnostic code is used by the toolchain to produce output.

## DiagnosticEmitter

[DiagnosticEmitters](/toolchain/diagnostics/diagnostic_emitter.h) handle the
main formatting of a message. It's parameterized on a location type, for which a
DiagnosticLocationTranslator must be provided that can translate the location
type into a standardized DiagnosticLocation of file, line, and column.

When emitting, the resulting formatted message is passed to a
DiagnosticConsumer.

## DiagnosticConsumers

DiagnosticConsumers handle output of diagnostic messages after they've been
formatted by an Emitter. Important consumers are:

-   [ConsoleDiagnosticConsumer](/toolchain/diagnostics/diagnostic_consumer.cpp):
    prints diagnostics to console.

-   [ErrorTrackingDiagnosticConsumer](/toolchain/diagnostics/diagnostic_consumer.h):
    counts the number of errors produced, particularly so that it can be
    determined whether any errors were encountered.

-   [SortingDiagnosticConsumer](/toolchain/diagnostics/sorting_diagnostic_consumer.h):
    sorts diagnostics by line so that diagnostics are seen in terminal based on
    their order in the file rather than the order they were produced.

-   [NullDiagnosticConsumer](/toolchain/diagnostics/null_diagnostics.h):
    suppresses diagnostics, particularly for tests.

Note that `SortingDiagnosticConsumer` is used by default by `carbon compile`. In
cases where one error leads to another error at an earlier location, for example
if an error in a function call argument leads to an error in the function call,
this can result in confusing diagnostic output where a consequence of the error
is reported before the cause. Usually this should be handled by tracking that an
error occurred and suppressing the follow-on diagnostic. During toolchain
development, it can be useful to disable the sorting so that the diagnostic
order matches the order in which the file was processed. This can be done using
`carbon compile –stream-errors`.

## Producing diagnostics

Diagnostics are used to surface issues from compilation. A simple diagnostic
looks like:

```cpp
CARBON_DIAGNOSTIC(InvalidCode, Error, "code is invalid");
emitter.Emit(location, InvalidCode);
```

Here, `CARBON_DIAGNOSTIC` defines a static instance of a diagnostic named
`InvalidCode` with the associated severity (`Error` or `Warning`).

The `Emit` call produces a single instance of the diagnostic. When emitted,
`"code is invalid"` will be the message used. The type of `location` depends on
the `DiagnosticEmitter`.

A diagnostic with an argument looks like:

```cpp
CARBON_DIAGNOSTIC(InvalidCharacter, Error, "invalid character {0}", char);
emitter.Emit(location, InvalidCharacter, invalid_char);
```

Here, the additional `char` argument to `CARBON_DIAGNOSTIC` specifies the type
of an argument to expect for message formatting. The `invalid_char` argument to
`Emit` provides the matching value. It's then passed along with the diagnostic
message format to `llvm::formatv` to produce the final diagnostic message.

## Diagnostic registry

There is a [registry](/toolchain/diagnostics/diagnostic_kind.def) which all
diagnostics must be added to. Each diagnostic has a line like:

```cpp
CARBON_DIAGNOSTIC_KIND(InvalidCode)
```

This produces a central enumeration of all diagnostics. The eventual intent is
to require tests for every diagnostic that can be produced, but that isn't
currently implemented.

## CARBON_DIAGNOSTIC placement

Idiomatically, `CARBON_DIAGNOSTIC` will be adjacent to the `Emit` call. However,
this is only because many diagnostics can only be produced in one code location.
If they can be produced in multiple locations, they will be at a higher scope so
that multiple `Emit` calls can reference them. When in a function,
`CARBON_DIAGNOSTIC` should be placed as close as possible to the usage so that
it's easier to see the associated output.

## Diagnostic context

Diagnostics can provide additional context for errors by attaching notes, which
have their own location information. A diagnostic with a note looks like:

```cpp
CARBON_DIAGNOSTIC(CallArgCountMismatch, Error,
                  "{0} argument(s) passed to function expecting "
                  "{1} argument(s).",
                  int, int);
CARBON_DIAGNOSTIC(InCallToFunction, Note,
                  "calling function declared here");
context.emitter()
    .Build(call_parse_node, CallArgCountMismatch, arg_refs.size(),
           param_refs.size())
    .Note(param_parse_node, InCallToFunction)
    .Emit();
```

The error and the note are registered as two separate diagnostics, but a single
overall diagnostic object is built and emitted, so that the error and the note
can be treated as a single unit.

Diagnostic context information can also be registered in a scope, so that all
diagnostics produced in that scope attach a specific note. For example:

```cpp
DiagnosticAnnotationScope annotate_diagnostics(
    &context.emitter(), [&](auto& builder) {
      CARBON_DIAGNOSTIC(
          InCallToFunctionParam, Note,
          "initializing parameter {0} of function declared here", int);
      builder.Note(param_parse_node, InCallToFunctionParam,
                   diag_param_index + 1);
    });
```

This is useful when delegating to another part of Check that may produce many
different kinds of diagnostic.

## Diagnostic parameter types

Diagnostic parameters should have informative types. We rely on three different
methods for formatting arguments:

-   Builtin
    [llvm::formatv](https://llvm.org/doxygen/FormatVariadic_8h_source.html)
    support.
    -   This includes `char` and integer types (`int`, `int32_t`, and so on).
    -   String types can be added as needed, but stringifying values using the
        methods noted below is preferred.
        -   Use `std::string` when allocations are required.
        -   `llvm::StringRef` is disallowed due to lifetime issues.
        -   `llvm::StringLiteral` is disallowed because format providers such as
            `BoolAsSelect` should work in cases where a `StringLiteral` could be
            used, and because string literal parameters tend to make the
            resulting diagnostics hard to translate.
-   `llvm::format_provider<...>` specializations.
    -   `BoolAsSelect` and `IntAsSelect` from
        [format_providers.h](/toolchain/diagnostics/format_providers.h) are
        recommended for many cases, because they allow putting the output string
        in the format.
        -   `IntAsSelect` can also be used to support pluralization.
    -   Custom providers can also be added for non-translated values. For
        example, `Lex::TokenKind` refers to syntax elements, and so can safely
        have its own format provider.
-   `DiagnosticConverter::ConvertArg` overrides.
    -   This can provide additional context to a formatter.
    -   For example, formatting `SemIR::NameId` accesses the IR's name list.

For `Check`, a custom diagnostic converter is provided that can convert some
common argument types. This includes some types defined in
[`check/diagnostic_helpers.h`](/toolchain/check/diagnostic_helpers.h) that exist
solely to be used as diagnostic parameter types. The types specifically
supported in `Check` diagnostics are:

-   For formatting names:
    -   `NameId` for a general name. This automatically uses raw identifier
        syntax for names that would collide with keywords.
    -   `LibraryNameId` for a library name string, which is formatted as either
        `default library` or `library "foo"`.
-   For formatting types, use the following, in order of preference:

    -   A `TypeOfInstId` parameter takes an `InstId` and formats the type of
        that instruction.
    -   An `InstIdAsType` parameter takes an `InstId` for a type expression and
        formats that type expression.
    -   A `TypeId` parameter is formatted as a canonical description of the
        type. This should be avoided when possible: `TypeId` has no context
        information, so any information about how the type was written in the
        source program will be lost.

    The above all include enclosing `` ` ``s around the formatted types. They
    may also include additional information about the type, such as the names
    bound to any aliases in the type, although at present they do not.

    When a type is formatted within a larger snippet of Carbon code, it can be
    desirable to instead just format the type itself; for this, `*AsRawType`
    parameter types are supported:

    -   `InstIdAsRawType`
    -   `TypeIdAsRawType`

-   For integer constants, `TypedInt` can be used to format an `APInt` given its
    type. The type is used to determine the signedness to use for the value.

## Diagnostic message style guide

We want Carbon's diagnostics to be helpful for developers when they run into an
error, and phrased consistently across diagnostics. In addition, Carbon
diagnostics may be mixed with Clang diagnostics when compiling interoperable
code, so we are borrowing some features of Clang's
[Diagnostic Wording](https://clang.llvm.org/docs/InternalsManual.html#diagnostic-wording).
Carbon's diagnostic style aims to balance these concerns. Our style is:

-   Start diagnostics with a lower case letter or quoted code, and omit trailing
    periods.

-   Quoted code should be enclosed in backticks, for example: ``"`{0}` is bad"``

-   Phrase diagnostics as bullet points rather than full sentences. Leave out
    articles unless they're necessary for clarity.

    -   Semicolons can be used to separate sentence fragments.

-   Diagnostics should describe the situation the toolchain observed. The
    language rule violated can be mentioned if it wouldn't otherwise be clear.
    For example:

    -   `"redeclaration of X"` describes the situation and implies that
        redeclarations are not permitted.

    -   ``"`self` declared in invalid context; can only be declared in implicit parameter list"``
        describes the language rule.

    -   It's OK for a diagnostic to guess at the developer's intent and provide
        a hint after explaining the situation and the rule, but not as a
        substitute for that. For example,
        ``"add `as String` to convert `i32` to `String`"`` is not sufficient as
        an error message, but
        ``"cannot implicitly convert `i32` to `String`; add `as String` for explicit conversion"``
        could be acceptable.

-   Use "cannot" if needed, but try to use phrasing that doesn't require it.
    Avoid "allowed", "legal", "permitted", "valid", and related wording. For
    example:

    -   ``"`export` in `impl` file"`` rather than
        ``"`export` is only allowed in API files"``.
    -   ``"`extern library` specifies current library"`` rather than
        `` "`extern library` cannot specify the current library"``.

-   Try to structure diagnostics such that inputs can be extracted without
    string parsing; prefer [typed parameters](#diagnostic-parameter-types). We
    would like to keep a path for diagnostics to be an API. There can be
    exceptions where this is particularly difficult.

-   TODO: Should diagnostics be atemporal and non-sequential ("multiple
    declarations of X", "additional declaration here"), present tense but
    sequential ("redeclaration of X", "previous declaration is here"), or
    temporal ("redeclaration of X", "previous declaration was here")? We could
    try to sidestep difference between the latter two by avoiding verbs with
    tense ("previously declared here", "Y declared here", with no is/was).

-   TODO: When do we put identifiers or expressions in diagnostics, versus
    requiring notes pointing at relevant code? Is it only avoided for values, or
    only allowed for types?

-   TODO: Lots more things to decide, give examples.
