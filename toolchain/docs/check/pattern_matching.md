# Pattern matching

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Pattern instructions](#pattern-instructions)
-   [Instruction ordering](#instruction-ordering)
-   [Parser-driven pattern block pushing](#parser-driven-pattern-block-pushing)
-   [Function parameters](#function-parameters)
    -   [`Call` parameters and arguments](#call-parameters-and-arguments)
    -   [Caller and callee matching](#caller-and-callee-matching)
    -   [The return slot](#the-return-slot)

<!-- tocstop -->

## Overview

This document focuses on the implementation of pattern matching. See
[here](/docs/design/pattern_matching.md) for more on the design and fundamental
concepts.

The SemIR for a pattern-matching operation is emitted in three steps:

1. **Pattern:** Traverse the parse tree of the pattern to emit SemIR that
   abstractly describes the pattern.
2. **Scrutinee:** Traverse the parse tree of the scrutinee expression to emit
   SemIR that evaluates it.
3. **Match:** Traverse the pattern SemIR from step 1 (sometimes in conjunction
   with the scrutinee SemIR) to emit SemIR that actually performs pattern
   matching.

## Pattern instructions

The SemIR emitted in the pattern step primarily consists of _pattern
instructions_, which are instructions that describe the pattern itself. For
example, given the pattern `(x: i32, y:i32)`, the pattern step might emit the
following SemIR:

```
%x.patt: %pattern_type.7ce = binding_pattern x [concrete]
%y.patt: %pattern_type.7ce = binding_pattern y [concrete]
%.loc4_21: %pattern_type.511 = tuple_pattern (%x.patt, %y.patt) [concrete]
```

Pattern instructions do not represent executable code, and are generally ignored
during lowering. Instead, they descriptively represent the pattern itself as a
kind of constant value, and their primary consumer is the match step. The type
of a pattern instruction is a _pattern type_, which is represented by a
`PatternType` instruction. For example, the `constants` block might define the
types in the above SemIR like so:

```
%i32: type = class_type @Int, @Int(%int_32) [concrete]
%pattern_type.7ce: type = pattern_type %i32 [concrete]
%tuple.type: type = tuple_type (%i32, %i32) [concrete]
%pattern_type.511: type = pattern_type %tuple.type [concrete]
```

We can read this as saying that the type of `%x.patt` and `%y.patt` is "pattern
that matches an `i32` scrutinee", and the type of `%.loc4_21` is "pattern that
matches a `(i32, i32)` scrutinee".

Pattern instructions are only emitted during the pattern step, but that step can
emit non-pattern instructions as well. For example, in a pattern like
`(x: i32, a + b)`, `i32` and `a + b` are ordinary expressions, and so their
SemIR must be emitted during the initial traversal of the parse tree, as with
any other expression.

All the pattern instructions for a given full-pattern are grouped together in a
distinct block that contains only pattern instructions. Consequently,
`Check::Context` maintains `pattern_block_stack` as a separate `InstBlockStack`
for pattern blocks, and provides separate methods like `AddPatternInst` for
adding instructions to it.

## Instruction ordering

The SemIR produced in the first two steps is (like most SemIR) generally in
post-order, reflecting the order of the parse tree. However, the match step
traversal is performed pre-order, starting with the root instruction of the
pattern and traversing into its dependencies.

In some cases it is necessary for the pattern step to allocate instructions that
won't actually be emitted until the match step, because they are responsible for
performing pattern matching. When that happens, they are allocated but not added
to a block, and their IDs are stored in the `Check::Context` so that they can be
spliced into the current block at the appropriate point in the match step.

Currently this happens in two cases, which are handled using two maps in
`Check::Context` from pattern instruction IDs to the corresponding match
instruction IDs:

-   A name binding can be used within the same pattern that declares it:
    ```carbon
    match (x) {
      case (n: i32, n) => ...
    ```
    For this to work, the name `n` needs to be added to the scope as soon as we
    handle its declaration, and it needs to resolve to the `BindName`
    instruction that binds a value to that name. This means that the `BindName`
    instruction needs to be allocated during the pattern step, even though it is
    part of matching, not part of the pattern. `Context::bind_name_map` stores
    these `BindName`s, keyed by the corresponding `BindingPattern` instruction.
-   A `var` pattern allocates storage during matching, which is represented by a
    `VarStorage` instruction. This instruction must be allocated during the
    pattern step, so that it can be used as the output parameter of scrutinee
    expression evaluation during the scrutinee step. `Context::var_storage_map`
    stores these `VarStorage` instructions, keyed by the corresponding
    `VarPattern` instruction.

As noted earlier, the pattern step can also emit non-pattern instructions to
evaluate expressions that are embedded in the pattern, such as the type
expressions of binding patterns, and expressions that are used as patterns
themselves (although those have not been implemented yet). The parse tree
doesn't mark these situations in advance: any given subpattern might turn out to
be one that emits non-pattern instructions. To handle these situations, we
speculatively push an instruction block onto the (non-pattern) stack whenever we
are about to begin handling a subpattern, and then pop it at the end of the
subpattern, with different treatment depending on whether the subpattern turned
out to be a subexpression. This is handled by `BeginSubpattern`,
`EndSubpatternAsExpr`, and `EndSubpatternAsNonExpr`.

One further complication here is that the type expression can contain control
flow (such as an `if` expression). Consequently, we can't represent the type
expression SemIR as a single block; instead, we represent the SemIR for a given
type expression as a
[single-entry, single-exit (SE/SE) region](https://en.wikipedia.org/wiki/Single-entry_single-exit),
potentially consisting of multiple blocks.

> **Note:** The original motivation for rigorously excluding non-pattern
> instructions from the pattern block may no longer apply. In particular, it may
> make sense to put non-pattern instructions in the pattern block when they
> represent an expression that is part of the pattern. If so, substantial parts
> of this design might change. See
> [issue #5351](https://github.com/carbon-language/carbon-lang/issues/5351).

## Parser-driven pattern block pushing

At the same time as all of that, we have to manage the _pattern_ block stack as
well. We attempt to do this precisely rather than speculatively, by leveraging
the parser to precisely mark the nodes immediately before full-patterns, and
pushing the pattern block stack when we handle those nodes. We then rely on
signals from both the parser and the node stack to determine when to pop from
the pattern block stack.

In the case of `let` and `var` decls, this is fairly straightforward: the
beginning is marked by the `LetIntroducer` or `VarIntroducer` node, and the end
is marked by the `LetInitializer` or `VarInitializer`, or by the `VarDecl` in
the case of a `var` decl with no initializer. Similarly, the beginning of an
`impl forall` parameter list is marked by the `Forall` node, and the end is
marked by the `ImplDecl` or `ImplDefinitionStart`.

The case of a parameterized name (such as `Bar(y: i32)`) is more challenging.
The node immediately before the start of the full-pattern is an identifier, but
an identifier doesn't necessarily mark the start of a full-pattern. We've solved
that by having the parser mark identifier nodes that are followed by
full-patterns (using lookahead). Rather than use additional storage for what is
logically a single bit of data, we effectively smuggle that bit into the kind
enum by having separate node kinds `IdentifierNameBeforeParams` and
`IdentifierNameNotBeforeParams`.

If the parameterized name is a name qualifier (such as the first part of
`Foo(X:! i32).Bar(y: i32)`), the node immediately after it will be the qualifier
node. As of this writing, we bifurcate qualifier nodes into
`NameQualifierWithParams` and `NameQualifierWithoutParams`, much like we do with
identifier names, but we don't actually use that information, and instead use
the presence of parameters on the node stack to determine whether to pop the
pattern block stack.

> **Open question:** should we re-combine the two qualifier node kinds?

If the parameterized name is not part of a name qualifier, the node immediately
after it will be a `*Decl` or `*DefinitionStart` node of the appropriate kind
(for example `FunctionDecl` or `FunctionDefinitionStart` if the introducer was
`fn`). Note that this means the pattern block is still on the stack while
handling the return type of a function. This is intentional, because we model
the return type as declaring an output parameter (see below), which makes it
functionally part of the parameter pattern.

## Function parameters

### `Call` parameters and arguments

SemIR models a function call as a `Call` instruction, which has an instruction
block consisting of one instruction per argument. Correspondingly, the SemIR
representation of a function has a block consisting of one instruction per
parameter. We refer to these as _`Call` arguments_ and _`Call` parameters_,
because they don't necessarily correspond to the colloquial meaning of
"arguments" and "parameters" (which are sometimes referred to as _syntactic_
arguments and parameters).

For example, consider this function:

```carbon
fn F(T:! type, U:! type) -> Core.String;
```

The `Call` instruction is a runtime-phase operation, so it notionally runs after
compile-time parameters have already been bound to values. As a result, a `Call`
instruction calling `F` does not pass values for either `T` or `U`. On the other
hand, it does pass a reference to the storage that `F` should construct the
return value in. So although we would colloquially say that `F` takes two
parameters of type `type`, it has a single `Call` parameter of type
`Core.String`.

If Carbon supports general patterns in function parameter lists, that introduces
additional ways that `Call` parameters can diverge from the colloquial meaning.
For example:

```carbon
fn G(x: i32, var (y: i32, z: i32));
fn H(x: i32, (y: i32, var z: i32));
```

A `var` pattern converts the scrutinee to a durable reference expression, and
then performs further pattern matching on the object it refers to. As a result,
`G` has two `Call` parameters: a value corresponding to `x`, and a reference to
an object of type `(i32, i32)`, corresponding to both `y` and `z`. On the other
hand, `H` has 3 `Call` parameters: values corresponding to `x` and `y`, and a
reference corresponding to `z`.

### Caller and callee matching

The `Call` parameters define the API boundary between the caller and callee at
the SemIR level. As a result, responsibility for matching the arguments against
the parameter list is split between the caller and the callee. Continuing the
example from above, given the call `G(0, (x, y))`, the caller is responsible for
converting `0` to `i32`, and for initializing a new `(i32, i32)` object from
`(x, y)`, but the callee is responsible for binding the name `x` to its first
`Call` parameter, and for destructuring its second `Call` parameter and binding
the names `y` and `z` to its elements.

In SemIR we represent this situation with special `ParamPattern` instructions,
which mark the boundary: there is exactly one `ParamPattern` instruction for
each `Call` parameter, which matches the entire corresponding `Call` argument.
The subpatterns of the `ParamPattern`s are matched on the callee side, and
everything above them is matched on the caller side. There are multiple kinds of
`ParamPattern` instruction, which correspond to different ways of passing a
parameter (such as by reference or by value).

When performing callee-side pattern matching, we do not have an actual scrutinee
expression. Instead, for each `ParamPattern` instruction we generate a
corresponding `Param` instruction, which reads from the corresponding entry in
the `Call` argument list, and we use that as the scrutinee of the
`ParamPattern`. Every `ParamPattern` kind has a corresponding `Param` kind.

### The return slot

If a function has a declared return type, the function takes an additional
`Call` parameter, which points to the storage that should be initialized with
the return value. This `Call` parameter is represented as an `OutParamPattern`
instruction with a `ReturnSlotPattern` instruction as a subpattern. The
`ReturnSlotPattern` also represents the return type declaration itself, such as
in `FunctionFields`. The SemIR that matches these patterns consists of a
`ReturnSlot` instruction, which binds the special name `NameId::ReturnSlot` to
the `OutParam` instruction representing the storage passed by the caller.

This structure is analogous to the handling of an ordinary by-value parameter,
which is represented in the `Call` parameters as a `ValueParamPattern`
instruction with a `BindingPattern`, and in the pattern-matching SemIR as a
`BindName` instruction that binds the parameter name to the `ValueParam`
instruction representing the argument passed by the caller.

Note that if the return type does not have an in-place value representation
(meaning that the return value should not be passed in memory), these
instructions will all still be generated, but the SemIR for `return` statements
will not access the `ReturnSlot`, and the `Call` argument list will not contain
an argument corresponding to the `OutParamPattern` (and so it will be one
element shorter than the `Call` parameter list). However, the
`ReturnSlotPattern` is still used, in its other role as a representation of the
return type declaration. This leads to a potentially confusing situation, where
the term "return slot" sometimes refers to the `ReturnSlotPattern` (for example
in `FunctionFields::return_slot_pattern`), which is present for any function
with a declared return type, and sometimes refers to the actual storage provided
by the caller (for example in `ReturnTypeInfo::has_return_slot`), which is
present only if the return type has an in-place value representation.

> **TODO:** When the return type isn't in-place, the `OutParamPattern` should
> probably not be in the `Call` parameter list (for consistency with the `Call`
> argument list), and possibly the `OutParamPattern`, `OutParam`, and
> `ReturnSlot` instructions should not be emitted in the first place.
> Furthermore, we should find a way to resolve the inconsistent "return slot"
> terminology.
