# Toolchain

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

The toolchain represents the production portion of Carbon. At a high level, the
toolchain's top priorities are:

-   Correctness.
-   Quality of generated code, including its performance.
-   Compilation performance.
-   Quality of diagnostics for incorrect or questionable code.

TODO: Add an expanded document that fully explains the goals and priorities and
link to it here.

The compiler is organized into a collection of libraries that can be used
independently. This includes the `//toolchain/driver` libraries that orchestrate
the typical and expected compilation flow using the other libraries. The driver
also includes the primary command-line tool: `//toolchain/driver:carbon`.

The typical compilation flow of data is:

1. Load the file into a [SourceBuffer](source/source_buffer.h).
2. Lex a `SourceBuffer` into a [TokenizedBuffer](lexer/tokenized_buffer.h).
3. Parse a `TokenizedBuffer` into a [ParseTree](parser/parse_tree.h).
4. Transform a `ParseTree` into a [SemanticsIR](semantics/semantics_ir.h).
5. This flow is still incomplete: code generation, using LLVM, is still
   required.

## Lexing

The [TokenizedBuffer](lexer/tokenized_buffer.h) is the central point of lexing.

The entire source buffer is converted into tokens before parsing begins. Tokens
are referred to by an opaque handle, `TokenizedBuffer::Token`, which is
represented as a dense integer index into the buffer. The tokenized buffer can
be queried to discover information about a token, such as its token kind, its
location in the source file, and its spelling.

The lexer ensures that all forms of brackets are matched, and is intended to
recover from missing brackets based on contextual cues such as indentation
(although this is not yet implemented), inserting matching close bracket tokens
where it thinks they belong. After the lexer completes, every opening bracket
token has a matching closing bracket token.

## Parsing

The [ParseTree](parser/parse_tree.h) is the output of parsing, but most logic is
in [ParserImpl](parser/parser_impl.h).

The parse tree faithfully represents the tree structure of the source program,
interpreted according to the Carbon grammar. No semantics are associated with
the tree structure at this level, and no name lookup is performed.

Each parse tree node has an expected structure, corresponding to the grammar of
the Carbon language, and the parser ensures that a valid parse tree node always
has a valid structure. However, any parse tree node can be marked as invalid,
and an invalid parse tree node can contain child nodes of any kind in any order.
This is intended to model the situation where parsing failed because the code
did not match the grammar, but we were still able to parse some subexpressions,
as an aid for non-compiler tools such as syntax highlighters or refactoring
tools.

Many functions in the parser return `llvm::Optional<T>`. A return value of
`llvm::None` indicates that parsing has failed and an error diagnostic has
already been produced, and that the current region of the parse tree might not
meet its invariants so that the caller should create an invalid parse tree node.
Other return values indicate that parsing was either successful or that any
encountered errors have been recovered from, so the caller can create a valid
parse tree node.

The produced `ParseTree` is in postorder. For example, given the code:

```carbon
fn foo() -> f64 {
  return 42;
}
```

The node order is (with indentation to indicate nesting):

```
    {node_index: 0, kind: 'FunctionIntroducer', text: 'fn'}
    {node_index: 1, kind: 'DeclaredName', text: 'foo'}
      {node_index: 2, kind: 'ParameterListEnd', text: ')'}
    {node_index: 3, kind: 'ParameterList', text: '(', subtree_size: 2}
      {node_index: 4, kind: 'Literal', text: 'f64'}
    {node_index: 5, kind: 'ReturnType', text: '->', subtree_size: 2}
  {node_index: 6, kind: 'FunctionDefinitionStart', text: '{', subtree_size: 7}
    {node_index: 7, kind: 'Literal', text: '42'}
    {node_index: 8, kind: 'StatementEnd', text: ';'}
  {node_index: 9, kind: 'ReturnStatement', text: 'return', subtree_size: 3}
{node_index: 10, kind: 'FunctionDefinition', text: '}', subtree_size: 11}
{node_index: 11, kind: 'FileEnd', text: ''}
```

This ordering is focused on efficient translation into the SemanticsIR.
Non-template code should be type-checked as soon as nodes are encountered,
decreasing SemanticsIR mutations.

While sometimes the beginning of the grammatical construct will be the parent,
where introducer keywords are used, it will often be the _end_ of the
grammatical construct that is the parent: this is so that a postorder traversal
of the tree can see the kind of grammatical construct being built first, and
handle child nodes taking that into account.

### Stack overflow

The `ParseTree` has been prone to stack overflows. As a consequence,
`CARBON_RETURN_IF_STACK_LIMITED` is checked at the start of most functions in
order to avoid errors. This manages depth increments and, when the scope exits,
decrements.

#### Future work

We are interested in eventually exploring ways to adjust the parser design to be
non-recursive and remove this limitation, but it hasn't yet been a priority and
keeping the code simple seems better until the language design stabilizes.

## Semantics

The [SemanticsIR](semantics/semantics_ir.h) is the output of semantic
processing.

The intent is that a `SemanticsIR` looks closer to a series of instructions than
a tree. This is in order to better align with the LLVM IR structure which will
be used for code generation.

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
the `DiagnosticEmitter`.

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

```cpp
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
