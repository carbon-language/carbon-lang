# Diagnostics

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Emitters](#emitters)
-   [Consumers](#consumers)
    -   [SortingConsumer](#sortingconsumer)
-   [Producing diagnostics](#producing-diagnostics)
-   [Attaching labels](#attaching-labels)
    -   [Context](#context)
-   [Diagnostic registry](#diagnostic-registry)
-   [CARBON_DIAGNOSTIC placement](#carbon_diagnostic-placement)
-   [Choosing what a label marks](#choosing-what-a-label-marks)
-   [Diagnostic parameter types](#diagnostic-parameter-types)
-   [Diagnostic message style guide](#diagnostic-message-style-guide)
-   [Alternatives considered](#alternatives-considered)
-   [References](#references)

<!-- tocstop -->

## Overview

The diagnostic code is used by the toolchain to produce output.

## Emitters

[`Emitter`s](/toolchain/diagnostics/emitter.h) handle the main formatting of a
message. It's parameterized on a location type, which `ConvertLoc` translates
into a standardized DiagnosticLocation of file, line, and column.

When emitting, the resulting formatted message is passed to a `Consumer`.

## Consumers

`Consumer`s handle output of diagnostic messages after they've been formatted by
an Emitter. Important consumers are:

-   [ConsoleConsumer](/toolchain/diagnostics/stream_consumer.cpp): prints
    diagnostics to console.

-   [ErrorTrackingConsumer](/toolchain/diagnostics/consumer.h): counts the
    number of errors produced, particularly so that it can be determined whether
    any errors were encountered.

-   [SortingConsumer](/toolchain/diagnostics/sorting_consumer.h): buffers and
    sorts diagnostics to provide a more human-understandable order while
    maintaining causal consistency.

### SortingConsumer

`SortingConsumer` is used by default by `carbon compile`. To see the actual
emitted order, use `carbon compile --stream-errors`.

The current `SortingConsumer` implementation sorts diagnostics based on the
`last_byte_offset`, which represents the latest token handled by the phase
emitting the diagnostic. This maintains the causal order of the toolchain's
traversal.

We expect cases where multiple diagnostics are emitted at the same offset,
particularly when they're emitted at the end of a scope. These can have attached
locations earlier in the scope, such as variable declarations. In these cases,
we put non-on-scope diagnostics first, and then sort on-scope diagnostics by
their start position (line and column).

Diagnostic sorting is stable, so that diagnostics from earlier phases are
printed first if all else is equal.

The sorting approach balances several competing needs:

-   **Causal order**: Developers generally want to fix errors in the order they
    are printed. If fixing error A could also fix error B, A should be printed
    first.

-   **Human-understandable order**: A human expects diagnostics to follow the
    flow of the file. If all parse errors in a file are printed before any
    semantic check errors, the developer may find it confusing to jump back and
    forth through the file.

-   **Performance where possible**: On fully correct code with no diagnostics,
    which is our performance priority, this has negligible overhead. When there
    are diagnostics, we try to only sort within the `SortingConsumer`. When
    sorting is not desired (such as tools and IDEs that provide their own
    ordering), it's easy to disable.

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

An on-scope diagnostic uses `CARBON_DIAGNOSTIC_ON_SCOPE` which is identical
other than the macro name. For example:

```cpp
CARBON_DIAGNOSTIC_ON_SCOPE(InvalidInScope, Error, "error inside scope");
emitter.Emit(location, InvalidInScope);
```

## Attaching labels

A label is a range of source attached to a diagnostic, with optional text saying
what that range has to do with the problem. It is declared with
`CARBON_DIAGNOSTIC_LABEL`, which takes the category before the format, and is
attached with `Attach`:

```cpp
CARBON_DIAGNOSTIC(CallArgCountMismatch, Error,
                  "{0} argument{0:s} passed to function expecting "
                  "{1} argument{1:s}",
                  IntAsSelect, IntAsSelect);
CARBON_DIAGNOSTIC_LABEL(ArgsPassedHere, Primary,
                        "{0} argument{0:s} passed here", IntAsSelect);
CARBON_DIAGNOSTIC_LABEL(InCallToFunction, Info,
                        "calling function declared here");
context.emitter()
    .Build(call_parse_node, CallArgCountMismatch,
           arg_refs.size(), param_refs.size())
    .Attach(args_parse_node, ArgsPassedHere, arg_refs.size())
    .Attach(param_parse_node, InCallToFunction)
    .Emit();
```

A label has a format and arguments of its own, validated the way a message's
are, so it says only what it needs to about the place it marks. It has no level,
because it belongs to the message it is attached to.

The category is `Primary` for source directly part of the problem and `Info` for
source that explains it without being part of it. Those two are all there is:
anything a diagnostic has to say that isn't read against the code it names is
not a label, and is declared with `CARBON_DIAGNOSTIC_CONTEXT` or
`CARBON_DIAGNOSTIC_LOCATION_INFO` instead.

`Attach` also takes a location alone, with no label:

```cpp
emitter.Build(loc, InvalidCode).Attach(operand_loc).Emit();
```

That marks a range as part of the problem and says nothing about it, which is
how a diagnostic points at the code its message is about without repeating the
message against it.

Attach every label at a location that reaches source: one that converts to only
a filename -- an invalid node, code the compiler generated -- leaves its words
on a row of their own with nothing to point at. Some diagnostics still do that
today; the goal is to restructure them until none do, and to give a diagnostic
genuinely about a file as a whole a start-of-file or end-of-file location with
a rendering of its own.

### Context

A context names the operation a problem happened inside. It is not a label: its
text is a sentence in its own right, so where a diagnostic has one it leads and
the message is read against the code like anything else explaining it.

A context is declared with `CARBON_DIAGNOSTIC_CONTEXT` and registered in a
scope, so that every diagnostic produced inside it says what larger operation
was being performed:

```cpp
Diagnostics::ContextScope diagnostic_context(
    &context.emitter(), [&](auto& builder) {
      CARBON_DIAGNOSTIC_CONTEXT(
          QualifiedDeclInIncompleteClassScope,
          "cannot declare a member of incomplete class {0}", SemIR::TypeId);
      builder.Attach(loc_id, QualifiedDeclInIncompleteClassScope,
                     context.classes().Get(class_id).self_type_id);
    });
```

This is useful when delegating to another part of Check that may produce many
different kinds of diagnostic. `CARBON_DIAGNOSTIC_SOFT_CONTEXT` declares a
fallback: it is dropped when the diagnostic already has a context, which is
assumed to describe the failure better.

An `AnnotationScope` works the same way for labels, attaching one to every
diagnostic emitted inside it.

## Diagnostic registry

There is a [registry](/toolchain/diagnostics/kind.def) which all diagnostics
must be added to. Each diagnostic has a line like:

```cpp
CARBON_DIAGNOSTIC_KIND(InvalidCode)
```

This produces a central enumeration of all diagnostics. The eventual intent is
to require tests for every diagnostic that can be produced, but that isn't
currently implemented.

Labels, contexts, and location info are not registered. Each is declared where
it is attached and nowhere else, so there is nothing for a registry to make
unique and nothing that needs to name one from a distance. Each still carries
its own name, so `--include-diagnostic-kind` says which one produced a line and
a test can match on it as it does on a kind.

What the registry does for a diagnostic is done for those by
`check_diagnostics.py`, which reads the declarations and the testdata directly
-- `label` below stands for any of the three:

-   Every declared label is attached somewhere. The compiler warns about most of
    these, but not about one declared in a header.
-   Every label is exercised by a `file_test`, which is what `coverage_test`
    does for kinds. `UNCOVERED_LABELS` is the exemption list, and an entry that
    gains a test has to come back out of it.
-   Every name a test matches on is a kind or a label that exists, so renaming
    one doesn't leave a test matching nothing.

Being attached does not imply being covered, which is why both are checked. A
label attached only on a branch no test takes is referenced by the code and
drawn by nothing, and the kind of the diagnostic it hangs off can be covered
while the branch never runs.

## CARBON_DIAGNOSTIC placement

Idiomatically, `CARBON_DIAGNOSTIC` will be adjacent to the `Emit` call. However,
this is only because many diagnostics can only be produced in one code location.
If they can be produced in multiple locations, they will be at a higher scope so
that multiple `Emit` calls can reference them. When in a function,
`CARBON_DIAGNOSTIC` should be placed as close as possible to the usage so that
it's easier to see the associated output.

## Choosing what a label marks

A message's location is underlined across the range it covers, so a diagnostic
with no label still shows the reader something. What it shows is whatever
location the code emitting it happened to have, which is usually the whole
construct the diagnostic was raised on -- and the part that is actually wrong is
often narrower.

So the question for each diagnostic is not whether it marks anything but whether
it marks the right thing:

-   Where the message's own location is what the reader should look at, leave it
    alone. Attaching a label with the same range and no text says nothing the
    message's own range didn't.

-   Where something narrower is what went wrong, attach a primary label naming
    it: the operand rather than the operator, the one argument rather than the
    call.

-   Give the label words when they point at where a part of the message became
    relevant, and leave it wordless otherwise. A message that says two things
    about two different places should say each of them where it happened:
    `CallArgCountMismatch` marks the call with `1 argument passed here` and the
    declaration with the arity it was declared with. Repeating the message that
    way is worth it, because the reader is being shown where each half of the
    sentence came from.

-   Where the right range isn't reachable, attach the closest one that is and
    leave a TODO saying which range it should be and what would make it
    available. A diagnostic marking its whole call because nothing names its
    argument list is better than one marking nothing, but the gap is worth
    recording where the next person will find it.

-   A message that names two types is the clearest case for two labels. An
    operation whose operands don't match reports the interface it looked up not
    being implemented, which names both types and points at neither, so each
    operand is marked with the type it contributed. `BuildBinaryOperator` takes
    a `MissingImplDiagnostic` for this, and installs it as an `AnnotationScope`,
    because the diagnostic is emitted further down where only one operand is
    still in hand.

    The words belong to the syntax, not to the helper the syntax shares.
    `a * b` has a left and a right operand, `a[i]` has an object and an index,
    and `for (x in r)` has only a range, so each names its own labels rather
    than everything that reaches `BuildBinaryOperator` saying "left" and
    "right".

    Once the operands carry labels, the message points at whatever they leave
    over rather than at the expression they make up. For an infix operator that
    is the operator itself, which `LocIdForDiagnostics::TokenOnly` names since
    the parse node is the operator token. For `a[i]` the parse node names the
    closing bracket, so the message points at the index -- the operand the
    object failed to accept.

-   Mark the syntax that required an `impl`, and name the interface in the same
    breath. Which interface a piece of syntax needs is a language rule the
    reader may not know, and the message reports only that one is missing, so
    `AttachOperatorSyntax` marks the operator with `` `*` requires an impl of
    `Core.MulWith` `` and lets the two be read together. That also gives the
    message somewhere to point that is neither operand.

    A message reporting the same missing `impl` twice for one construct is
    worse than either half of it. A `for` loop looks up `Iterate` to make its
    cursor and again to advance it, so the second is diagnosed only when the
    first succeeded.

-   Say what the developer wrote, not what it desugared to. Impl lookup for
    `a * b` runs through member access, and reporting that a member of
    `Core.MulWith` can't be accessed describes code nobody typed. Where a
    caller passes the syntax down -- `desugared_loc_id` in
    `PerformCompoundMemberAccess` -- the message drops to the part the syntax
    can't say: which type failed to implement the interface.

-   Don't narrate the code the label sits under. `` `var` nested within another
    `var` `` marks the keyword, and a label reading ``the invalid `var`
    keyword`` repeats both the source and the message while saying nothing
    about where anything came from. Most parse diagnostics are like this: the
    message already names the token, so the range marks it and says nothing.

Some ranges cannot be attached at all, and the reason is worth knowing before
trying. A diagnostic can only mark source that something in hand still names,
and several layers discard that on the way:

-   **Desugaring drops where each operand was written.** An operator becomes a
    call whose arguments are conversion insts created at the operator's
    location, so a diagnostic raised while evaluating that call -- division by
    zero, integer overflow, a shift out of range -- can name the operation but
    not the operand. `SemIR::Converted` keeps an `original_id` for tooling, but
    the argument on this path is not a `Converted`, so following it does not
    reach the operand. `Convert` works around the same gap for
    `ConversionFailure` by holding the expression it was given. Closing it
    properly means the desugaring recording where each operand was written.

-   **Some scopes have no declaration.** `extend impl` and `impl as` outside a
    class are only ever reached from a namespace -- the file or package -- or
    from a scope with no instruction, so there is nothing to mark as the scope
    they ended up in.

Where that is the case, attach nothing and leave a TODO saying which range it
should be and what would make it available. A label whose location names a file
but no line draws no anchor and reads as pointing at nothing, which is worse
than the message alone. `check-toolchain-diagnostics` fails a label no testdata
reaches, which is how to find one whose location never resolves.

The obvious source being a dead end is not the same as no source existing, and
it is worth one more look before writing the TODO. Deduction's inputs are
substituted insts with no source, but the argument each one descended from can
be carried alongside it. `UsedBeforeInitialization` is handed an `InitTombstone`
rather than the binding, but the binding is still on `FullPatternStack`. In both
cases the range is somewhere else on a stack that has not been unwound yet.

Until a diagnostic is migrated its message location supplies the range, which is
usually the one column an editor would put a cursor on and rarely an extent
worth looking at.

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

    -   ``"`self` declared in invalid context; can only be declared in implicit
        parameter list"`` describes the language rule.

    -   It's OK for a diagnostic to guess at the developer's intent and provide
        a hint after explaining the situation and the rule, but not as a
        substitute for that. For example, ``"add `as String` to convert `i32` to
        `String`"`` is not sufficient as an error message, but ``"cannot
        implicitly convert `i32` to `String`; add `as String` for explicit
        conversion"`` could be acceptable.

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
    requiring labels pointing at relevant code? Is it only avoided for values,
    or only allowed for types?

-   TODO: Lots more things to decide, give examples.

## Alternatives considered

-   [Don't sort diagnostics](/proposals/p006699-diagnostic-sorting.md#dont-sort-diagnostics)
-   [Sort by line and column](/proposals/p006699-diagnostic-sorting.md#sort-by-line-and-column)
-   [Sort by last processed token](/proposals/p006699-diagnostic-sorting.md#sort-by-last-processed-token)

## References

-   Proposal
    [#6699: Sort diagnostics](https://github.com/carbon-language/carbon-lang/pull/6699)
