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
    -   [Name references](#name-references)
    -   [Storage and initializing expressions](#storage-and-initializing-expressions)
    -   [The final ordering](#the-final-ordering)
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

1.  **Pattern:** Traverse the parse tree of the pattern to emit SemIR that
    abstractly describes the pattern.
2.  **Scrutinee:** Traverse the parse tree of the scrutinee expression to emit
    SemIR that evaluates it.
3.  **Match:** Traverse the pattern SemIR from step 1 (sometimes in conjunction
    with the scrutinee SemIR) to emit SemIR that actually performs pattern
    matching.

Note that steps 1 and 2 emit insts in bottom-up order (as usual for SemIR), but
step 3 will traverse the pattern and scrutinee insts in top-down order.

However, the resulting insts do not necessarily appear in that order in the
SemIR, and in some cases instructions that belong to the later steps are
emitted in earlier steps, for reasons discussed [below](#instruction-ordering).

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
distinct block that contains only pattern instructions, for reasons discussed
[below](#name-references). Consequently,
`Check::Context` maintains `pattern_block_stack` as a separate `InstBlockStack`
for pattern blocks, and operations like `AddInst` automatically put
newly-created pattern insts on that stack.

## Instruction ordering

Consider the following Carbon code (using the currently-hypothetical `if let`):

```carbon
if (let (var n: i32, n) = (f(), x)) ...
```

You might expect the SemIR for that code to reflect the 3-step process of
creating it, for example:

```
// Step 1: traverse the pattern.
%n_patt: %i32_pattern = ref_binding_pattern n
%n_var_patt: %i32_pattern = var_pattern %n_patt
%n_ref: ref i32 = name_ref n, %n
%n_expr_patt: %i32_pattern = expr_pattern %n_ref
%pattern: %i32_pair_pattern = tuple_pattern (%n_var_patt, %n_expr_patt)

// Step 2: evaluate the scrutinee.
%call: init i32 to %n_var = call %F
%x_ref: i32 = name_ref x
%scrutinee: %i32_pair = tuple_literal (%call, %x_ref)

// Step 3: match the pattern with the scrutinee
%n_var: ref i32 = var_storage
%n: ref i32 = ref_binding %n_var
%equal: init bool = call %Core.EqWith.Op(%n_ref, %x_ref)
if %equal then br !if.then else br !if.else
```

However, we require SemIR to be topologically ordered, and the above code
violates that in two places:

-   `%n_ref` takes the value of the referenced name (in this case `%n`) as an
    operand. `%n_ref` is part of the pattern, so it belongs in step 1, but `%n`
    is a name binding created during pattern matching in step 3. In general
    this can happen any time a pattern uses the name of a binding that was
    declared earlier in the same pattern.
-   `%call` is an initializing expression, so it takes the storage to initialize
    (`%n_ref` in this case) as an output parameter. `%call` is part of the
    initializer, so it belongs in step 2, but `%n_ref` is part of step 3. In
    general this can be a problem any time a `var` pattern is initialized from
    a local initializing expression.

> **Note:** In practice `%call` actually won't have an output parameter because
> `i32` doesn't have a pointer initializing representation, but we're ignoring
> that to keep the example concrete.

In both cases, the non-topological order reflects a deeper problem: at the point
where we want to create the instruction, one of its operands doesn't yet exist.
Furthermore, to the extent that we solve that problem by creating the operand
inst earlier, we still have the problem of how to _find_ that operand inst when
we need it.

We solve these two classes of problems in different ways.

### Name references

To solve the topological ordering problem with name references, we emit pattern
insts into a separate block (on a separate stack of pattern blocks), and then
splice it into the SemIR at the end of pattern matching, which must be after the
inst that the `name_ref` refers to. For a local pattern match like our example,
this splicing is performed by a `name_binding_decl` inst. We can do this without
violating the topological ordering, because within a pattern-matching operation,
non-pattern insts may be generated from the pattern insts, but can't actually
depend on them.

> **TODO:** As of this writing, a `var_storage` inst takes the `var_pattern`
> it was generated from as an operand, which violates this requirement and can
> lead to violations of the topological ordering. We need to fix this.

That still leaves the problem that we're creating a `name_ref` in step 1, but
don't know its operand until step 3. We solve this as follows:

During step 1:

-   When we are about to handle an expression within a pattern (such as an
    expression pattern or the type part of a binding pattern), we push an
    `ExprRegion` onto the `inst_block_stack` to capture the expression insts,
    and then pop it and store its ID at the end of the expression, so that we
    can splice the expression evaluation into the pattern matching SemIR later.
    This is handled by the `ExprRegionForPattern` functions in
    `toolchain/check/pattern.h`.
-   When we handle a binding pattern, we eagerly create a binding inst (in
    addition to a binding pattern inst), and add its ID to name lookup, but we
    create it in a placeholder state with no value, and we do not add it to any
    block yet. We also add the binding inst and the `ExprRegion` for its type
    expression to `bind_name_map` so that they can later be looked up using the
    binding pattern as a key.
-   When we handle a name expression (during step 1 or at any other time), we
    look up its name to find the ID of the binding, create a `name_ref` with
    that binding as its value operand, and add it to the `inst_block_stack`.

Then, during step 3:

-   When we match a binding pattern inst with its scrutinee, we look up the
    corresponding binding inst and its type's `ExprRegion` in `bind_name_map`,
    splice the `ExprRegion` onto the top of the `inst_block_stack`, overwrite
    the binding's value operand with the scrutinee ID, and add the binding inst
    to the top of the `inst_block_stack` (as if we had just created it).
-   When we match an expression pattern inst, we splice the `ExprRegion` onto
    the top of the `inst_block_stack`, and then compare it with the scrutinee
    (Note: expression pattern matching is not yet implemented).

### Storage and initializing expressions

To solve the ordering problems with initializing expressions like `%call`, we
create and emit `var_storage` insts during the initial traversal of the pattern
in step 1, so that they are sequenced before the insts in step 2, and track them
in the `FullPatternStack` for later reuse. Then, when evaluating the initializer
in step 2, we set its output operand to a placeholder ID. Finally, when we bring
the `var_pattern` and its initializer together in step 3, we look up the
corresponding `var_storage` inst in the `FullPatternStack`, and then rewrite the
initializer inst (or make a rewritten copy of it) to have the `var_storage` inst
as its output operand (see the `Initialize` in `convert.h` for details about
that).

Note that when the full pattern is part of a parameter list, we create the
`var_storage` inst on demand in step 3, because parameters currently can't have
initializers, so this problem doesn't come up.

### The final ordering

Combining the solutions to those two problems, the emitted SemIR for our example
will actually look something like this:

```
// Step 1: traverse the pattern.
%n_var: ref i32 = var_storage

// Step 2: traverse the scrutinee.
%call: init i32 to %n_var = call %F
%x_ref: i32 = name_ref x, %x
%scrutinee: %i32_pair = tuple_literal (%call, %x_ref)

// Step 3: match the pattern with the scrutinee
%n: ref i32 = ref_binding %n_var
%n_ref: ref i32 = name_ref n, %n
%equal: init bool = call %Core.EqWith.Op(%n_ref, %x_ref)

name_binding_decl {
  // Step 1: traverse the pattern.
  %n_patt: %i32_pattern = ref_binding_pattern n
  %n_var_patt: %i32_pattern = var_pattern %n_patt
  %n_expr_patt: %i32_pattern = expr_pattern %n_ref
  %pattern: %i32_pair_pattern = tuple_pattern (%n_var_patt, %n_expr_patt)
}
if %equal then br !if.then else br !if.else
```

## Parser-driven pattern block pushing

In order to produce correct pattern blocks, we need to ensure that a new pattern
block is pushed onto the stack at the start of every full-pattern, and popped
at the end. We attempt to do this precisely rather than speculatively, by leveraging
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
enum by having separate node kinds `IdentifierNameMaybeBeforeSignature` and
`IdentifierNameNotBeforeSignature`.

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
If a `ParamPattern` has a subpattern, it is matched on the callee side, and
everything above it is matched primarily on the caller side. There are multiple
kinds of `ParamPattern` instruction, which correspond to different ways of
passing a parameter (such as by reference or by value).

When performing callee-side pattern matching, we do not have an actual scrutinee
expression. Instead, for each `ParamPattern` instruction we generate a
corresponding `Param` instruction, which reads from the corresponding entry in
the `Call` argument list, and we use that as the scrutinee of the
`ParamPattern`. Every `ParamPattern` kind has a corresponding `Param` kind.

### The return slot

If a function has a declared return type, the function takes an additional
`Call` parameter, which points to the storage that should be initialized with
the return value. This `Call` parameter is represented as `ReturnSlotPattern`
instruction with an `OutParamPattern` instruction as a subpattern. The
`ReturnSlotPattern` also represents the return type declaration itself, such as
in `FunctionFields`. The SemIR that matches these patterns consists of a
`ReturnSlot` instruction, which binds the special name `NameId::ReturnSlot` to
the `OutParam` instruction representing the storage passed by the caller.

This structure is analogous to the handling of an ordinary by-value parameter,
which is represented in the `Call` parameters as an `WrapperBindingPattern`
instruction with a `ValueParamPattern` subpattern, and in the pattern-matching
SemIR as a `ValueBinding` instruction that binds the parameter name to the
`ValueParam` instruction representing the argument passed by the caller.

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
