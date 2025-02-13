# Check

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Subtopics](#subtopics)
-   [Postorder processing](#postorder-processing)
-   [Key IR concepts](#key-ir-concepts)
    -   [Instruction operands](#instruction-operands)
    -   [Parameters and arguments](#parameters-and-arguments)
-   [SemIR textual format](#semir-textual-format)
    -   [Raw form](#raw-form)
    -   [Formatted IR](#formatted-ir)
        -   [Instructions](#instructions)
        -   [Top-level entities](#top-level-entities)
-   [Core loop](#core-loop)
    -   [Node stack](#node-stack)
    -   [Delayed evaluation (not yet implemented)](#delayed-evaluation-not-yet-implemented)
    -   [Templates (not yet implemented)](#templates-not-yet-implemented)
    -   [Rewrites](#rewrites)
-   [Types](#types)
    -   [Type printing (not yet implemented)](#type-printing-not-yet-implemented)
-   [Expression categories](#expression-categories)
    -   [ExprCategory::NotExpression](#exprcategorynotexpression)
    -   [ExprCategory::Value](#exprcategoryvalue)
    -   [ExprCategory::DurableReference and ExprCategory::EphemeralReference](#exprcategorydurablereference-and-exprcategoryephemeralreference)
    -   [ExprCategory::Initializing](#exprcategoryinitializing)
    -   [ExprCategory::Mixed](#exprcategorymixed)
    -   [Value bindings](#value-bindings)
-   [Handling Parse::Tree errors (not yet implemented)](#handling-parsetree-errors-not-yet-implemented)
-   [Alternatives considered](#alternatives-considered)
    -   [Using a traditional AST representation](#using-a-traditional-ast-representation)

<!-- tocstop -->

## Overview

Check takes the parse tree and generates a semantic intermediate representation,
or SemIR. This will look closer to a series of instructions, in preparation for
transformation to LLVM IR. Semantic analysis and type checking occurs during the
production of SemIR. It also does any validation that requires context.

## Subtopics

Some particular topics have their own documentation:

-   [Associated constants](associated_constant.md)

## Postorder processing

The checking step is oriented on postorder processing on the `Parse::Tree` to
iterate through the `Parse::NodeImpl` vectorized storage once, in order, as much
as possible. This is primarily for performance, but also relies on the
[information accumulation principle](/docs/project/principles/information_accumulation.md):
that is, when that principle applies, we should be able to generate IR
immediately because we can rely on the principle that when a line is processed,
the information necessary to semantically check that line is already available.

Indirectly, what this really means is that we should be able to go from a
Parse::Tree (which cannot be used for name lookups) to a SemIR with name lookups
completed in a single pass. The SemIR should not need to be re-processed to add
more information outside of templates. By doing this, we avoid an additional
processing pass with associated storage needs.

This single-pass approach also means that the checking step does not make use of
the tree structure of the `Parse::Tree`. In cases where the actions performed
for a parse tree node depend on the context in which that node appears, a node
that is visited earlier in the postorder traversal, such as a bracketing node,
needs to establish the necessary context. In this respect, the sequence of
`Parse::Node`s can be thought of as a byte code input that the check step
interprets to build the `SemIR`.

## Key IR concepts

A `SemIR::Inst` is the basic building block that represents a simple
instruction, such as an operator or declaring a literal. For each kind of
instruction, a struct for that specific kind of instruction is provided in the
`SemIR` namespace. For example, `SemIR::Assign` represents an assignment
instruction, and `SemIR::PointerType` represents a pointer type instruction.

Each instruction class has up to four public data members describing the
instruction, as described in
[sem_ir/typed_insts.h](/toolchain/sem_ir/typed_insts.h) (also see
[adding features for Check](/toolchain/docs/adding_features.md#check)):

-   An `InstKind kind;` member if the instruction has a `Kinds` constant making
    it a shorthand for multiple individual instructions.

-   A `SemIR::TypeId type_id;` member that describes the type of the instruction
    is present on all instructions that produce a value. This includes namespace
    instructions, which are modeled as producing a value of "namespace" type,
    even though they can't be used as a first-class value in Carbon expressions.

-   Up to two additional [kind-specific members](#instruction-operands). For
    example `SemIR::Assign` has members `InstId lhs_id` and `InstId rhs_id`. And
    `SemIR::ClassElementAccess` has an `ElementIndex` into the fields of the
    class to define which field it is accessing.

Instructions are stored as type-erased `SemIR::Inst` objects, which store the
instruction kind and the (up to) four fields described above. This balances the
size of `SemIR::Inst` against the overhead of indirection. All instructions have
with them a `LocId` that tracks their source location, and is stored separately
on the `InstStore` and retrieved by `GetLocId()`.

A `SemIR::InstBlockId`, used to index into `SemIR::InstBlockStore`, can
represent a code block. However, it can also be created when a series of
instructions needs to be closely associated, such as a parameter list.

A number of instruction types in
[sem_ir/typed_insts.h](/toolchain/sem_ir/typed_insts.h) are builtin
instructions, such as `SemIR::TypeType` which represents the unconstrained facet
type `type`. Builtins have stable ids in the `SemIR::InstStore` across `SemIR`
instances.

### Instruction operands

The kind-specific members on a typed instruction struct can be of any type
listed in the `SemIR::IdKind` enumeration defined in
[sem_ir/id_kind.h](/toolchain/sem_ir/id_kind.h), typically derived from
`IdBase`. The most commonly used kinds refer to other instructions:

-   `SemIR::InstId`: Refers to a specific instance of an instruction. For
    example, this should be used if the operand may have side-effects or a
    meaningful location.
-   `SemIR::ConstantId`: An abstract reference to a known constant value. This
    should be used instead of `InstId` if you care only about the identity of
    the value and not how it was formed.
-   `SemIR::TypeId`: An abstract reference to a known constant value of type
    `type`. This should be used instead of `ConstantId` if you know that the
    type of the constant value is always `type` (or the value is the "error"
    constant, whose type is also the "error" constant).

Other ID types are used for more specialized purposes. Refer to the class
documentation for the ID type for more details.

### Parameters and arguments

Parameters and arguments will be stored as two blocks (referred to by
`SemIR::InstBlockId`) each. The first will contain the full IR, while the second
will contain references to the last instruction for each parameter or argument.
The references block will have a size equal to the number of parameters or
arguments, allowing for quick size comparisons and indexed access.

## SemIR textual format

There are two textual ways to view `SemIR`.

### Raw form

The raw form of SemIR shows the details of the representation, such as numeric
instruction and block IDs. The representation is intended to very closely match
the `SemIR::File` and `SemIR::Inst` representations. This can be useful when
debugging low-level issues with the `SemIR` representation.

The driver will print this when passed `--dump-raw-sem-ir`.

### Formatted IR

In addition to the raw form, there is a higher-level formatted IR that aims to
be human readable. This is used in most `check` tests to validate the output,
and also expected to be used regularly by toolchain developers to inspect the
result of checking the parse tree.

The driver will print this when passed `--dump-sem-ir`.

Unlike the raw form, certain representational choices in the `SemIR` data may
not be visible in this form. However, it is intended to be possible to parse the
`SemIR` output and form an equivalent – but not necessarily identical – `SemIR`
representation, although no such parser currently exists.

As an example, given the program:

```carbon
fn Cond() -> bool;
fn Run() -> i32 { return if Cond() then 1 else 2; }
```

The formatted IR is currently:

```
constants {
  %.1: i32 = int_literal 1 [template]
  %.2: i32 = int_literal 2 [template]
}

file {
  package: <namespace> = namespace [template] {
    .Cond = %Cond
    .Run = %Run
  }
  %Cond: <function> = fn_decl @Cond [template] {
    %return.var.loc1: ref bool = var <return slot>
  }
  %Run: <function> = fn_decl @Run [template] {
    %return.var.loc2: ref i32 = var <return slot>
  }
}

fn @Cond() -> bool;

fn @Run() -> i32 {
!entry:
  %Cond.ref: <function> = name_ref Cond, file.%Cond [template = file.%Cond]
  %.loc2_33.1: init bool = call %Cond.ref()
  %.loc2_26.1: bool = value_of_initializer %.loc2_33.1
  %.loc2_33.2: bool = converted %.loc2_33.1, %.loc2_26.1
  if %.loc2_33.2 br !if.expr.then else br !if.expr.else

!if.expr.then:
  %.loc2_41: i32 = int_literal 1 [template = constants.%.1]
  br !if.expr.result(%.loc2_41)

!if.expr.else:
  %.loc2_48: i32 = int_literal 2 [template = constants.%.2]
  br !if.expr.result(%.loc2_48)

!if.expr.result:
  %.loc2_26.2: i32 = block_arg !if.expr.result
  return %.loc2_26.2
}
```

There are three kinds of names in formatted IR, which are distinguished by their
leading sigils:

-   `%name` denotes a value produced by an instruction. These names are
    introduced by a line of the form `%name: <category> <type> = <instruction>`,
    and are scoped to the enclosing top-level entity. `<category>` describes the
    [expression category](#expression-categories), which is `init` for an
    initializing expression, `ref` for a reference expression, or omitted for a
    value expression. Typically, values can only be referenced by instructions
    that their introduction
    [dominates](<https://en.wikipedia.org/wiki/Dominator_(graph_theory)>), but
    some kinds of instruction might have other rules. Names in the `file` block
    can be referenced as `file.%<name>`.

-   `!name` denotes a label, and `!name:` appears as a prefix of each
    `InstBlock` in a `Function`. These names are scoped to their enclosing
    function, and can be referenced anywhere in that function, but not outside.

-   `@name` denotes a top-level entity, such as a function, class, or interface.
    The SemIR view of these entities is flattened, so member functions are
    treated as top-level entities.

Names in formatted IR are all invented by the formatter, and generally are of
the form `<base_name>[.loc<line>[_<col>[.<counter>]]]` where `<line>` and
`<col>` describe the location of the instruction, and `<counter>` is used as a
disambiguator if multiple instructions appear at the same location. Trailing
name components are only included if they are necessary to disambiguate the
name. `<base_name>` is a guessed good name for the instruction, often derived
from source-level identifiers, and is empty if no guess was made.

#### Instructions

There is usually one line in a `InstBlock` for each `Inst`. You can find the
documentation for the different kinds of instructions in
[toolchain/sem_ir/typed_insts.h](/toolchain/sem_ir/typed_insts.h). For example,
given a formatted SemIR line like:

```
%N: i32 = assoc_const_decl N [template]
```

you would look for a `struct` definition that uses `"assoc_const_decl"` as its
`ir_name`. In this case, this is the `AssociatedConstantDecl` instruction:

```cpp
// An associated constant declaration in an interface, such as `let T:! type;`.
struct AssociatedConstantDecl {
  static constexpr auto Kind =
      InstKind::AssociatedConstantDecl.Define<Parse::NodeId>(
          {.ir_name = "assoc_const_decl", .is_lowered = false});

  TypeId type_id;
  NameId name_id;
};
```

Since this instruction produces a value, it has a `TypeId type_id` field, which
corresponds to the type written between the `:` and the `=`. In the example
above, that type is `i32`. The other arguments to the instruction are written
after the `ir_name` -- in this example the `name_id` is `N`. From this we find
that the instruction corresponds to an associated constant declaration in an
interface like `let N:! i32;`.

Instructions producing a constant value, like `assoc_const_decl` above, are
followed by their phase, either `[symbolic]` or `[template]`, and then `=` the
value if it is the value of a different instruction.

Instructions that do not produce a value, such as the `br` and `return`
instructions above, omit the leading `%name: ... =` prefix, as they cannot be
named by other instructions. These instructions do not have a `TypeId type_id`
field, like the `AdaptDecl` instruction:

```cpp
// An adapted type declaration in a class, of the form `adapt T;`.
struct AdaptDecl {
  static constexpr auto Kind = InstKind::AdaptDecl.Define<Parse::AdaptDeclId>(
      {.ir_name = "adapt_decl", .is_lowered = false});

  // No type_id; this is not a value.
  TypeId adapted_type_id;
};
```

An `adapt SomeClass;` declaration would have the corresponding SemIR formatted
as:

```
adapt_decl %SomeClass
```

Some instructions have special argument handling. For example, some invalid
arguments will be omitted. Or an `InstBlockId` argument will be rendered inline,
commonly enclosed in braces `{`...`}` or parens `(`...`)`. In other cases, the
formatter will combine instructions together to make the IR more readable:

-   A terminator sequence in a block, comprising a sequence of `BranchIf`
    instructions followed by a `Branch` or `BranchWithArg` instruction, is
    collapsed into a single
    `if %cond br !label1 else if ... else br !labelN(%arg)` line.
-   A struct type, formed by a sequence of `StructTypeField` instructions
    followed by a `StructType` instruction, is collapsed into a single
    `struct_type{.field1: %value1, ..., .fieldN: %valueN}` line.

These exceptions may be found in
[toolchain/sem_ir/formatter.cpp](/toolchain/sem_ir/formatter.cpp).

#### Top-level entities

**Question:** Are these too in flux to document at this time?

-   `constants`: TODO
-   `imports`: TODO
-   `file`: TODO
-   entities
    -   TODO: may be preceded by `extern`.
    -   TODO: may be preceded by `generic`.
        -   These may have an optional `!definition:` section containing the
            generic's `definition_block_id`.
    -   `fn`: TODO; followed by `= "`...`"` for builtins
    -   `class`: TODO
    -   `interface`: TODO
    -   `impl`: TODO
-   `specific`: TODO
    -   body in braces `{`...`}` has a bunch of
        ``<generic parameter> => <specific value>` assignment lines
    -   The first lines of the body describe the declaration
    -   If there is a valid definition, there are additional definition
        assignments after a `!definition:` line.

## Core loop

The core loop is `Check::CheckParseTree`. This loops through the `Parse::Tree`
and calls a `Handle`... function corresponding to the `NodeKind` of each node.
Communication between these functions for different nodes working together is
through the `Context` object defined in
[check/context.h](/toolchain/check/context.h), which stores things in a
collection of stacks. The common pattern is that the children of a node are
processed first. They produce information that is then consumed when processing
the parent node.

One example of this pattern is expressions. Each subexpression outputs SemIR
instructions to compute the value of that subexpression to the current
instruction block, added to the top of the `InstBlockStack` stored in the
`Context` object. It leaves an instruction id on the top of the
[node stack](#node-stack) pointing to the instruction that produces the value of
that subexpression. Those are consumed by parent operations, like an
[RPN](https://en.wikipedia.org/wiki/Reverse_Polish_notation) calculator. For
example, the expression `1 * 2 + 3` corresponds to this parse tree:

```yaml
    {kind: 'IntegerLiteral', text: '1'},
    {kind: 'IntegerLiteral', text: '2'},
  {kind: 'InfixOperator', text: '*', subtree_size: 3},
  {kind: 'IntegerLiteral', text: '3'},
{kind: 'InfixOperator', text: '+', subtree_size: 5},
```

This parse tree is processed by one call to a `Handle` function per node:

-   The first node is an integer literal, so the core loop calls
    `HandleIntegerLiteral`.

    -   It calls `context::AddInstAndPush` to output a `SemIR::IntegerLiteral`
        instruction to the current instruction block, and pushes the parse node
        along with the instruction id to the [node stack](#node-stack).

-   The second node is also an integer literal, which outputs a second
    instruction and pushes another entry onto the node stack.

-   `HandleInfixOperator` pops the two entries off of the node stack, outputs
    any conversion instructions that are needed, and uses
    `context::AddInstAndPush` to create and push the instruction id representing
    the output of a multiplication instruction. That multiplication instruction
    takes the instruction ids it popped off the stack at the beginning as
    arguments.

-   Another integer literal instruction is created for `3` and pushed onto the
    stack.

-   `HandleInfixOperator` is called again. It pops the two instruction ids off
    the stack to use as the arguments to the multiplication instruction it
    creates and pushes.

In this way, the handle functions coordinate producing their output using the
instruction block stack and node block stack from the context.

A similar pattern uses bracketing nodes to support parent nodes that can have a
variable number of children. For example, a `return` statement can produce parse
trees following a few different patterns:

-   `return;`

    ```yaml
      {kind: 'ReturnStatementStart', text: 'return'},
    {kind: 'ReturnStatement', text: ';', subtree_size: 2},
    ```

-   `return x;`

    ```yaml
      {kind: 'ReturnStatementStart', text: 'return'},
      {kind: 'NameExpr', text: 'x'},
    {kind: 'ReturnStatement', text: ';', subtree_size: 3},
    ```

-   `return var;`

    ```yaml
      {kind: 'ReturnStatementStart', text: 'return'},
      {kind: 'ReturnVarModifier', text: 'var'},
    {kind: 'ReturnStatement', text: ';', subtree_size: 3},
    ```

In all three cases, the introducer node `ReturnStatementStart` pushes an entry
on the [node stack](#node-stack) with just the parse node and no id, called a
_solo parse node_. The handler for the parent `ReturnStatement` node can pop and
process entries from the node stack until it finds that solo parse node from
`ReturnStatementStart` that indicates it is done.

Another pattern that arises is state is set up by an introducer node, updated by
its siblings, and then consumed by the bracketing parent node. TODO: example

### Node stack

The node stack, defined in [check/node_stack.h](/toolchain/check/node_stack.h),
stores pairs of a `Parse::Node` and an id. The type of the id is determined by
the `NodeKind` of the parse node. It is the default, general-purpose stack used
by `Handle`... functions in the check stage. Using a single stack is beneficial
since it improves locality of reference and reduces allocations. However,
additional stacks are used to ensure we never need to search through the stack
to find data -- we always want to be operating on the top of the stack (or a
fixed offset).

The node stack contains any state pushed by siblings of the current
`Parse::Node` at the top, and state pushed by siblings of ancestors below. The
boundaries between what is a sibling of the current `Parse::Node` versus what is
a sibling of an ancestor are not explicitly determined. Instead, the handler for
the parent node knows how many nodes it must pop from the stack based either on
knowing the fixed number of children for that node kind or popping nodes until
it reaches a bracketing node. The arity or bracketing node kind for each parent
node is documented in [parse/node_kind.def](/toolchain/parse/node_kind.def).

When each `Parse::Node` is evaluated, the SemIR for it is typically immediately
generated as `SemIR::Inst`s. To help generate the IR to an appropriate context,
scopes have separate `SemIR::InstBlock`s.

### Delayed evaluation (not yet implemented)

Sometimes, nodes will need to have delayed evaluation; for example, an inline
definition of a class member function needs to be evaluated after the class is
fully declared. The `SemIR::Inst`s cannot be immediately generated because they
may include name references to the class. We're likely to store a reference to
the relevant `Parse::Node` for each definition for re-evaluation after the class
scope completes. This means that nodes in a definition would be traversed twice,
once while determining that they're inline and without full checking or IR
generation, then again with full checking and IR generation.

### Templates (not yet implemented)

Templates need to have partial semantic checking when declared, but can't be
fully implemented before they're instantiated against a specific type.

We are likely to generate a partial IR for templates, allowing for checking with
the incomplete information in the IR. Instantiation will likely use that IR and
fill in the missing information, but it could also reevaluate the original
`Parse::Node`s with the known template state.

### Rewrites

Carbon relies on rewrites of code, such as rewriting the destination of an
initializer to a specific target object once that object is known.

We have two ways to achieve this. One is to track the IR location of a
placeholder instruction and, if it needs updating, replace it with a "rewrite"
`SemIR::Inst` that points to a new `SemIR::InstBlock` containing the required IR
and specifying which value is the result of that rewrite. This is expressed in
SemIR as a `splice_block` instruction. Another is to track the list of
instructions to be created separately from the node block stack, and merge those
instructions into the current block once we have decided on their contents.

## Types

Type expressions are treated like any other expression, and are modeled as
`SemIR::Inst`s. The types computed by type expressions are deduplicated,
resulting in a canonical `SemIR::TypeId` for each distinct type.

### Type printing (not yet implemented)

The `TypeId` preserves only the identity of the type, not its spelling, and so
printing it will produce a fully-resolved type name, which isn't a great user
experience as it doesn't reflect how the type was written in the source code.

Instead, when printing a type name for use in a diagnostic, we will start with
one of two `InstId`s:

-   A `InstId` for a type expression that describes the way the type was
    computed.
-   A `InstId` for an expression that has the given type.

In the former case, the type is pretty-printed by walking the type expression
and printing it. In the latter case, the type of the expression is reconstructed
based on the form of the expression: for example, to print the type of `&x`, we
print the type of `x` and append a `*`, being careful to take potential
precedence issues into account.

TODO: This requires being able to print the type of, for example,
`x.foo[0].bar`, by printing only the desired portion of the type of `x`, and
similarly may require handling the case where the type of an expression involves
generic parameters whose arguments are specified by that expression. In effect,
the type computation performed when checking an operation is duplicated into the
type printing logic, but is simpler because errors don't need to be detected.

This approach means we don't need to preserve a fully-sugared type for each
expression instruction. Instead, we compute that type when we need to print it.

## Expression categories

Each `SemIR::Inst` that has an associated type also has an expression category,
which describes how it produces a value of that type. These
`SemIR::ExprCategory` values correspond to the Carbon expression categories
defined in proposal
[#2006](https://github.com/carbon-language/carbon-lang/pull/2006):

### ExprCategory::NotExpression

This instruction is not an expression instruction, and doesn't have an
expression category. This is used for namespaces, control flow instructions, and
other constructs that represent some non-expression-level semantics.

### ExprCategory::Value

This instruction produces a value using the type's value representation.
Lowering the instruction will produce an LLVM value using that value
representation.

### ExprCategory::DurableReference and ExprCategory::EphemeralReference

This instruction produces a reference to an object. Lowering will produce a
pointer to an object representation.

### ExprCategory::Initializing

This instruction represents the initialization of an object. Depending on the
initializing representation for the type, the initializing expression
instruction will do one of the following:

-   For an in-place initializing representation, the instruction will store a
    value to the target of the initialization.

-   For a by-copy initializing representation, the instruction will produce an
    object representation by value that can be stored into the target. This is
    currently only used in cases where the object representation and the value
    representation are the same.

-   For a type with no initializing representation, such as an empty struct or
    tuple, it does neither of the above things.

Regardless of the initializing representation, an initializing expression should
be consumed by another instruction that finishes the initialization. For a
by-copy initialization, this final instruction represents the store into the
target, whereas in the other cases it is only used to track in SemIR how the
initialization was used. When an in-place initializer uses a by-copy initializer
as a subexpression, an `initialize_from` instruction is inserted to perform this
final store.

### ExprCategory::Mixed

This instruction represents a language construct that doesn't have a single
expression category. This is used for struct and tuple literals, where the
elements of the literal can have different expression categories. Instructions
with a mixed expression category are treated as a special case in conversion,
which recurses into the elements of those instructions before performing
conversions.

### Value bindings

A value binding represents a conversion from a reference expression to the value
stored in that expression. There are three important cases here:

-   For types with a by-copy value representation, such as `i32`, a value
    binding represents a load from the address indicated by the reference
    expression.

-   For types with a by-pointer value representation, such as arrays and large
    structs and tuples, a value binding implicitly takes the address of the
    reference expression.

-   For structs and tuples, the value representation is a struct or tuple of the
    elements' value representations, which is not necessarily the same as a
    struct or tuple of the elements' object representations. In the case where
    the value representation is not a copy of, or pointer to, the object
    representation, `value_binding` instructions are not used, and a
    `tuple_value` or `struct_value` instruction is used to construct a value
    representation instead. `value_binding` should still be used in the case
    where the value and object representation are the same, but this is not yet
    implemented.

## Handling Parse::Tree errors (not yet implemented)

`Parse::Tree` errors will typically indicate that checking would error for a
given context. We'll want to be careful about how this is handled, but we'll
likely want to generate diagnostics for valid child nodes, then reduce
diagnostics once invalid nodes are encountered. We should be able to reasonably
abandon generated IR of the valid children when we encounter an invalid parent,
without severe effects on surrounding checks.

For example, an invalid line of code in a function might generate some
incomplete IR in the function's `SemIR::InstBlock`, but that IR won't negatively
interfere with checking later valid lines in the same function.

## Alternatives considered

### Using a traditional AST representation

Clang creates an AST as part of compilation. In Carbon, it's something we could
do as a step between parsing and checking, possibly replacing the SemIR. It's
likely that doing so would be simpler, amongst other possible trade-offs.
However, we think the SemIR approach is going to yield higher performance,
enough so that it's the chosen approach.
