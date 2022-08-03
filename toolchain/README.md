# Toolchain

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

The toolchain represents the production portion of Carbon. The toolchain's top
priority is performance: it needs to generate code quickly and scale well.

The main compiler is `//toolchain/driver:carbon`. When compiling, the current
flow of data is:

1. Load the file into a [SourceBuffer](source/source_buffer.h).
2. Lex a SourceBuffer into a [TokenizedBuffer](lexer/tokenized_buffer.h).
3. Parse a TokenizedBuffer into a [ParseTree](parser/parse_tree.h).
4. Transform a ParseTree into a [SemanticsIR](semantics/semantics_ir.h).
5. This flow is still incomplete: code generation, using LLVM, is still
   required.

## Lexing

The [TokenizedBuffer](lexer/tokenized_buffer.h) is the central point of lexing.

The entire source buffer is converted into tokens before parsing begins. Tokens are referred to by an opaque handle, `TokenizedBuffer::Token`, which is represented as a 32-bit integer into the token array. The tokenized buffer can be queried to discover information about a token, such as its token kind, its location in the source file, and its spelling.

The lexer ensures that all forms of brackets are matched, and is intended to recover from missing brackets based on contextual cues such as indentation (although this is not yet implemented), inserting matching close bracket tokens where it thinks they belong. After the lexer completes, every opening bracket token has a matching closing bracket token.

## Parsing

The [ParseTree](parser/parse_tree.h) is the output of parsing, but most logic is
in [ParserImpl](parser/parser_impl.h).

The produced ParseTree is in reverse postorder. For example, given the code:

```carbon
fn foo() -> f64 {
  return 42;
}
```

The node order is (with indentation to indicate nesting):

```carbon
  Index 0: kind DeclaredName
    Index 1: kind ParameterListEnd
  Index 2: kind ParameterList
    Index 3: kind Literal
  Index 4: kind ReturnType
      Index 5: kind Literal
      Index 6: kind StatementEnd
    Index 7: kind ReturnStatement
    Index 8: kind CodeBlockEnd
  Index 9: kind CodeBlock
Index 10: kind FunctionDeclaration
Index 11: kind FileEnd
```

This is done this way in order to allow for more efficient processing of a file.
As a consequence, the SemanticsIR does a lot of reversal of the ParseTree
ordering in order to visit code in source order.

### Stack overflow

The ParseTree has been prone to stack overflows. As a consequence,
`CARBON_RETURN_IF_STACK_LIMITED` is checked at the start of most functions in
order to avoid errors. This manages depth increments and, when the scope exits,
decrements.

## Semantics

The [SemanticsIR](semantics/semantics_ir.h) is the output of semantic
processing. It's currently built using
[a factory](semantics/semantics_ir_factory.h).

The intent is that a SemanticsIR looks closer to a series of instructions than a
tree. This is in order to better align with the LLVM IR structure which will be
used for code generation.

This phase should eventually include semantic checking of the SemanticsIR, but
it's a work in progress.

## Diagnostics

### DiagnosticEmitter

[DiagnosticEmitters](diagnostics/diagnostic_emitter.h) handle the main
formatting of a message. It's parameterized on a location type, for which a
`DiagnosticLocationTranslator` must be provided that can translate the location
type into a standardized `DiagnosticLocation` of file, line, and column.

When emitting, the resulting formatted message is passed to a
`DiagnosticConsumer`.

### DiagnosticConsumers

`DiagnosticConsumers` handle output of diagnostic messages after they've been
formatted by an `Emitter`. Important consumers are:

-   [ConsoleDiagnosticConsumer](diagnostics/diagnostic_emitter.h): prints
    diagnostics to console.
-   [ErrorTrackingDiagnosticConsumer](diagnostics/diagnostic_emitter.h): counts
    the number of errors produced, particularly so that it can be determined
    whether any errors were encountered.
-   [SortingDiagnosticConsumer](diagnostics/sorting_diagnostic_consumer.h):
    sorts diagnostics by line so that diagnostics are seen in terminal based on
    their order in the file rather than the order they were produced.
-   [NullDiagnosticConsumer](diagnostics/null_diagnostics.h): suppresses
    diagnostics, particularly for tests.

### Producing diagnostics

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
the DiagnosticEmitter.

A diagnostic with an argument looks like:

```cpp
CARBON_DIAGNOSTIC(InvalidCharacter, Error, "Invalid character `{0}`.", char);
emitter.Emit(location, InvalidCharacter, invalid_char);
```

Here, the additional `char` argument to `CARBON_DIAGNOSTIC` specifies the type
of an argument to expect for message formatting. The `invalid_char` argument to
`Emit` provides the matching value. It's then passed along with the diagnostic
message format to `llvm::formatv` in order to produce the final diagnostic
message.

#### Diagnostic registry

There is a [registry](diagnostics/diagnostic_registry.def) which all diagnostics
must be added to. Each diagnostic has a line like:

```cppp
CARBON_DIAGNOSTIC_KIND(InvalidCode)
```

This produces a central enumeration of all diagnostics. The eventual intent is
to require tests for every diagnostic that can be produced, but that isn't
currently implemented.

#### `CARBON_DIAGNOSTIC` placement

Idiomatically, `CARBON_DIAGNOSTIC` will be adjacent to the `Emit` call. However,
this is only because many diagnostics can only be produced in one code location.
If they can be produced in multiple locations, they will be at a higher scope so
that multiple `Emit` calls can reference them. When in a function,
`CARBON_DIAGNOSTIC` should be placed as close as possible to the usage so that
it's easier to see the associated output.

### Diagnostic context

In the future, we'll want to provide additional context for errors. For example,
if there's a function parameter mismatch, it may be useful to point both at the
caller and function signature compared. However, at present the emitter only
produces errors on one location. This is something that we need to consider
further, and will probably involve further changes to diagnostic handling.
