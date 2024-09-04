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

-   [ConsoleDiagnosticConsumer](/toolchain/diagnostics/diagnostic_emitter.h):
    prints diagnostics to console.

-   [ErrorTrackingDiagnosticConsumer](/toolchain/diagnostics/diagnostic_emitter.h):
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
CARBON_DIAGNOSTIC(InvalidCode, Error, "Code is invalid");
emitter.Emit(location, InvalidCode);
```

Here, `CARBON_DIAGNOSTIC` defines a static instance of a diagnostic named
`InvalidCode` with the associated severity (`Error` or `Warning`).

The `Emit` call produces a single instance of the diagnostic. When emitted,
`"Code is invalid"` will be the message used. The type of `location` depends on
the `DiagnosticEmitter`.

A diagnostic with an argument looks like:

```cpp
CARBON_DIAGNOSTIC(InvalidCharacter, Error, "Invalid character {0}.", char);
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
                  "Calling function declared here.");
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
          "Initializing parameter {0} of function declared here.", int);
      builder.Note(param_parse_node, InCallToFunctionParam,
                   diag_param_index + 1);
    });
```

This is useful when delegating to another part of Check that may produce many
different kinds of diagnostic.

## Diagnostic parameter types

Here are some types you might consider for the parameters to a diagnostic:

-   `llvm::StringLiteral`. Note that we don't use `llvm::StringRef` to avoid
    lifetime issues.
-   `std::string`
-   Carbon types `T` that implement `llvm::format_provider<T>` like:
    -   `Lex::TokenKind`
    -   `Lex::NumericLiteral::Radix`
    -   `Parse::RelativeLocation`
-   integer types: `int`, `uint64_t`, `int64_t`, `size_t`
-   `char`
-   Other
    [types supported by llvm::formatv](https://llvm.org/doxygen/FormatVariadic_8h_source.html)

## Diagnostic message style guide

In order to provide a consistent experience, Carbon diagnostics should be
written in the following style:

-   Start diagnostics with a capital letter or quoted code, and end them with a
    period.

-   Quoted code should be enclosed in backticks, for example:
    ``"`{0}` is bad."``

-   Phrase diagnostics as bullet points rather than full sentences. Leave out
    articles unless they're necessary for clarity.

-   Diagnostics should describe the situation the toolchain observed and the
    language rule that was violated, although either can be omitted if it's
    clear from the other. For example:

    -   `"Redeclaration of X."` describes the situation and implies that
        redeclarations are not permitted.

    -   ``"`self` can only be declared in an implicit parameter list."``
        describes the language rule and implies that you declared `self`
        somewhere else.

    -   It's OK for a diagnostic to guess at the developer's intent and provide
        a hint after explaining the situation and the rule, but not as a
        substitute for that. For example,
        ``"Add an `as String` cast to format this integer as a string."`` is not
        sufficient as an error message, but
        ``"Cannot add i32 to String. Add an `as String` cast to format this integer as a string."``
        could be acceptable.

-   TODO: Should diagnostics be atemporal and non-sequential ("multiple
    declarations of X", "additional declaration here"), present tense but
    sequential ("redeclaration of X", "previous declaration is here"), or
    temporal ("redeclaration of X", "previous declaration was here")? We could
    try to sidestep difference between the latter two by avoiding verbs with
    tense ("previously declared here", "Y declared here", with no is/was).

-   TODO: Word choices:

    -   For disallowed constructs, do we say they're not permitted / not allowed
        / not valid / not legal / illegal / ill-formed / disallowed? Do we say
        "X cannot be Y" or "X may not be Y" or "X must not be Y" or "X shall not
        be Y"?

-   TODO: Is structuring diagnostics such that inputs can be parsed without
    string parsing important? that is, when is passing strings in as part of the
    message templating okay?

-   TODO: When do we put identifiers or expressions in diagnostics, versus
    requiring notes pointing at relevant code? Is it only avoided for values, or
    only allowed for types?

-   TODO: Lots more things to decide, give examples.
