# Parse

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Parse stack](#parse-stack)
-   [Postorder tree](#postorder-tree)
-   [Bracketing inside the tree](#bracketing-inside-the-tree)
-   [Visual example](#visual-example)
-   [Handling invalid parses](#handling-invalid-parses)
-   [How is this accomplished?](#how-is-this-accomplished)
    -   [Introducer](#introducer)
    -   [Optional modifiers before an introducer](#optional-modifiers-before-an-introducer)
    -   [Something required in context](#something-required-in-context)
    -   [Optional clauses](#optional-clauses)
        -   [Case 1: introducer to optional clause is used as parent node](#case-1-introducer-to-optional-clause-is-used-as-parent-node)
        -   [Case 2: parent node is required token after optional clause, with different parent node kinds for different options](#case-2-parent-node-is-required-token-after-optional-clause-with-different-parent-node-kinds-for-different-options)
        -   [Case 3: optional sibling](#case-3-optional-sibling)
    -   [Operators](#operators)

<!-- tocstop -->

## Overview

Parsing uses tokens to produce a parse tree that faithfully represents the tree
structure of the source program, interpreted according to the Carbon grammar. No
semantics are associated with the tree structure at this level, and no name
lookup is performed.

The parse tree's structure corresponds to the grammar of the Carbon language. On
valid input, there will be a 1:1 correspondence between parse tree nodes and
tokens.

A parse tree is considered _structurally valid_ if all nodes have the number of
children that their node kind requires. On invalid input, nodes may be added
that don't correspond to a token to maintain a structurally valid parse tree.
When a parse tree node is marked as having an error, it will still be
structurally valid, but its children may not match a valid grammar. Code trying
to handle children of erroneous nodes must be prepared to handle atypical
structures, but it may still be helpful for tools such as syntax highlighters or
refactoring tools.

In general, we favor doing the checking for whether something is allowed _in a
particular context_ in [the check stage](check.md) instead of the parse stage,
unless the context is very local. This is for a few reasons:

-   We anticipate that the parse stage will be used to operate on invalid code
    while still preserving as much of the intent of the author as possible, for
    example in an IDE or a code formatter.
-   To keep as much code out of the parse stage as possible, so it is simple and
    fast.
-   We are building all the infrastructure to keep track of context in the check
    stage.

These reasons explain what local context is okay: where we already have the
contextual information at hand so there is no performance cost, and we can
output a parse tree that still captures faithfully what the user wrote.
Examples:

-   All declaration modifiers are allowed in any order on any declaration in the
    parse stage. Diagnosing duplicated modifiers, modifiers that conflict with
    other modifiers, or modifiers that can't be used on a particular declaration
    is postponed until the check stage.
-   Rejecting a keyword after `fn` where a name is expected is done at the parse
    stage.

## Parse stack

The core parser loop is `Parse::Tree::Parse`. In the loop, it pops the next
state off the stack, and dispatches to the appropriate `Handle` function.

A typical handler function pops the state first, leaving the stack ready for the
next state. It may add nodes to the parse tree, based on the current code. If it
needs to trigger other states, it will push them onto the stack; because it's a
stack, the _next_ state is always pushed _last_.

Operator expressions store information about current operator precedence in the
stack as well. While this isn't necessary for most parser states, and could be
stored separately, it's currently together because it has no impact on the size
of a stack entry and is thus more efficient to store in one place.

## Postorder tree

The parse tree's storage layout is in postorder. For example, given the code:

```carbon
fn foo() -> f64 {
  return 42;
}
```

The node order is (with indentation to indicate nesting):

<!-- Prevent prettier from changing indents. -->
<!-- prettier-ignore-start -->

```yaml
[
  {kind: 'FileStart', text: ''},
      {kind: 'FunctionIntroducer', text: 'fn'},
      {kind: 'Name', text: 'foo'},
        {kind: 'ParamListStart', text: '('},
      {kind: 'ParamList', text: ')', subtree_size: 2},
        {kind: 'Literal', text: 'f64'},
      {kind: 'ReturnType', text: '->', subtree_size: 2},
    {kind: 'FunctionDefinitionStart', text: '{', subtree_size: 7},
      {kind: 'ReturnStatementStart', text: 'return'},
      {kind: 'Literal', text: '42'},
    {kind: 'ReturnStatement', text: ';', subtree_size: 3},
  {kind: 'FunctionDefinition', text: '}', subtree_size: 11},
  {kind: 'FileEnd', text: ''},
]
```

<!-- prettier-ignore-end -->

In this example, `FileStart`, `FunctionDefinition`, and `FileEnd` are "root"
nodes for the tree. Function components are children of `FunctionDefinition`.

It's produced in this way because it's an efficient layout to produce with
vectorized storage, requiring little context to be maintained during parsing.
Because it's stored in postorder, it's also most efficient to process the parsed
output in postorder; this affects checking.

The parse tree is printed in postorder by default because it matches how the
parse tree is expected to be processed within the toolchain , and so can make it
easier to reason about. However, the `--preorder` flag may be used in contexts
where a preorder representation would be easier to handle.

## Bracketing inside the tree

The parse tree is designed to be walked in postorder by checking, allowing
checking to be more efficient. To support this, checking sometimes requires
context on the meaning of a node when it is encountered.

Each `ParseNodeKind` has either a bracketing node, or a specific child count.
This helps document and enforce the expected tree structure.

When a bracketing node is indicated, it is the opening bracket: it will always
be the first child of the parent, and that will be the only time it occurs in
the parent's children (it may still occur in children of children). When
checking encounters the opening bracket, this means it can make contextual
decisions for the later children of the node.

Nodes can also have a specific child count, for example, infix operators always
have two children: the lhs and rhs expressions. Many nodes have a child count of
0; this just means they're leaf nodes, and will never have children.

Because the tree structure is always valid, these are treated as contracts. Some
nodes exist only to be used to construct valid tree structures for invalid
input, such as `StructFieldUnknown`.

Although each subtree's size is also tracked as part of the node, we're
currently trying to avoid relying on it and may eliminate it if it turns out to
be unnecessary and a meaningful cost for the compiler.

## Visual example

To try to explain the transition from code to Parse Tree, consider the
statement:

```carbon
var x: i32 = y + 1;
```

Lexing creates distinct tokens for each syntactic element, which will form the
basis of the parse tree:

<!--
These charts are using a hidden "root" node in order to adjust alignment.
We're trying to keep the "lexed" tokens separate from "parse tree" nodes.
-->

```mermaid
flowchart BT
    subgraph tokens["Tokens"]
        token1[var]
        token2[x]
        token3[:]
        token4[i32]
        token5[=]
        token6[y]
        token7[+]
        token8[1]
        token9[;]
    end
```

First the `var` keyword is used as a "bracketing" node (VariableIntroducer).
When this is seen in a postorder traversal, it tells us to expect the basics of
a variable declaration structure.

```mermaid
flowchart BT
    subgraph tokens["Remaining tokens"]
        token1[var]:::used
        token2[x]
        token3[:]
        token4[i32]
        token5[=]
        token6[y]
        token7[+]
        token8[1]
        token9[;]
    end

    subgraph nodes["Parsed nodes"]
        direction BT
        node1[var]:::moved
        node2[x]:::pending
        node3[:]:::pending
        node4[i32]:::pending
        node5[=]:::pending
        node6[y]:::pending
        node7[+]:::pending
        node8[1]:::pending
        node9[;]:::pending
    end

    %% A token which has been used.
    classDef used visibility:hidden
    %% A node which will be used, but hasn't yet been.
    classDef pending visibility:hidden
    %% A token or node which is actively being used.
    classDef moved fill:#0F0,color:#000

    nodes ~~~ tokens

    node1 ~~~~ tokens
    node3 ~~~ node2 & node4
    node3 ~~~ tokens
    node5 ~~~~ tokens
    node7 ~~~ node6 & node8
    node7 ~~~ tokens
    node9 ~~~ node1 & node3 & node5 & node7
    node9 ~~~ tokens
```

Next, we can consider the pattern binding. Here, `x` is the identifier and `i32`
is the type expression. The `:` provides a parent node that must always contain
two children, the name and type expression. Because it always has two direct
children, it doesn't need to be bracketed.

```mermaid
flowchart BT
    subgraph tokens["Remaining tokens"]
        token1[var]:::used
        token2[x]:::used
        token3[:]:::used
        token4[i32]:::used
        token5[=]
        token6[y]
        token7[+]
        token8[1]
        token9[;]
    end

    classDef used visibility:hidden
```

```mermaid
flowchart BT
    root:::hidden
    subgraph nodes["Parsed nodes"]
        direction BT
        node1[var]
        node2[x]:::moved
        node3[:]:::moved
        node4[i32]:::moved
        node5[=]:::pending
        node6[y]:::pending
        node7[+]:::pending
        node8[1]:::pending
        node9[;]:::pending
    end

    classDef pending visibility:hidden
    classDef moved fill:#0F0,color:#000
    classDef hidden visibility:hidden,display:none

    node1 ~~~~ root
    node3 --- node2 & node4
    node3 ~~~ root
    node5 ~~~~ root
    node7 ~~~ node6 & node8
    node7 ~~~ root
    node9 ~~~ node1 & node3 & node5 & node7
    node9 ~~~ root
```

We use the `=` as a separator (instead of a node with children like `:`) to help
indicate the transition from binding to assignment expression, which is
important for expression parsing during checking.

```mermaid
flowchart BT
    subgraph tokens["Remaining tokens"]
        token1[var]:::used
        token2[x]:::used
        token3[:]:::used
        token4[i32]:::used
        token5[=]:::used
        token6[y]
        token7[+]
        token8[1]
        token9[;]
    end

    classDef used visibility:hidden
```

```mermaid
flowchart BT
    root:::hidden
    subgraph nodes["Parsed nodes"]
        direction BT
        node1[var]
        node2[x]
        node3[:]
        node4[i32]
        node5[=]:::moved
        node6[y]:::pending
        node7[+]:::pending
        node8[1]:::pending
        node9[;]:::pending
    end

    classDef pending visibility:hidden
    classDef moved fill:#0F0,color:#000
    classDef hidden visibility:hidden,display:none

    node1 ~~~~ root
    node3 --- node2 & node4
    node3 ~~~ root
    node5 ~~~~ root
    node7 ~~~ node6 & node8
    node7 ~~~ root
    node9 ~~~ node1 & node3 & node5 & node7
    node9 ~~~ root
```

The expression is a subtree with `+` as the parent, and the two operands as
child nodes.

```mermaid
flowchart BT
    subgraph tokens["Remaining tokens"]
        token1[var]:::used
        token2[x]:::used
        token3[:]:::used
        token4[i32]:::used
        token5[=]:::used
        token6[y]:::used
        token7[+]:::used
        token8[1]:::used
        token9[;]
    end

    classDef used visibility:hidden
```

```mermaid
flowchart BT
    root:::hidden
    subgraph nodes["Parsed nodes"]
        direction BT
        node1[var]
        node2[x]
        node3[:]
        node4[i32]
        node5[=]
        node6[y]:::moved
        node7[+]:::moved
        node8[1]:::moved
        node9[;]:::pending
    end

    classDef pending visibility:hidden
    classDef moved fill:#0F0,color:#000
    classDef hidden visibility:hidden,display:none

    node1 ~~~~ root
    node3 --- node2 & node4
    node3 ~~~ root
    node5 ~~~~ root
    node7 --- node6 & node8
    node7 ~~~ root
    node9 ~~~ node1 & node3 & node5 & node7
    node9 ~~~ root
```

Finally, the `;` is used as the "root" of the variable declaration. It's
explicitly tracked as the `;` for a variable declaration so that it's
unambiguously bracketed by `var`.

```mermaid
flowchart BT
    root:::hidden
    subgraph nodes["Parsed nodes"]
        direction BT
        node1[var]
        node2[x]
        node3[:]
        node4[i32]
        node5[=]
        node6[y]
        node7[+]
        node8[1]
        node9[;]:::moved
    end

    classDef pending visibility:hidden
    classDef moved fill:#0F0,color:#000
    classDef hidden visibility:hidden,display:none

    node1 ~~~~ root
    node3 --- node2 & node4
    node3 ~~~ root
    node5 ~~~~ root
    node7 --- node6 & node8
    node7 ~~~ root
    node9 --- node1 & node3 & node5 & node7
    node9 ~~~ root
```

Thus we have the parse tree:

```mermaid
flowchart BT
    root:::hidden
    subgraph nodes["Parsed nodes"]
        direction BT
        node1[var]
        node2[x]
        node3[:]
        node4[i32]
        node5[=]
        node6[y]
        node7[+]
        node8[1]
        node9[;]
    end

    classDef pending visibility:hidden
    classDef moved fill:#0F0,color:#000
    classDef hidden visibility:hidden,display:none

    node1 ~~~~ root
    node3 --- node2 & node4
    node3 ~~~ root
    node5 ~~~~ root
    node7 --- node6 & node8
    node7 ~~~ root
    node9 --- node1 & node3 & node5 & node7
    node9 ~~~ root
```

In storage, this tree will be flat and in postorder. Because the order hasn't
changed much from the original code, we can do the reordering for postorder with
a minimal number of nodes being delayed for later output: it will be linear with
respect to the depth of the parse tree.

```mermaid
flowchart BT
    subgraph tokens["Tokens"]
        token1[var]
        token2[x]
        token3[:]
        token4[i32]
        token5[=]
        token6[y]
        token7[+]
        token8[1]
        token9[;]
    end
```

```mermaid
flowchart BT
    root:::hidden
    subgraph nodes["Parsed nodes"]
        direction BT
        node1[var]
        node2[x]
        node3[:]
        node4[i32]
        node5[=]
        node6[y]
        node7[+]
        node8[1]
        node9[;]
    end

    classDef used visibility:hidden
    classDef moved fill:#0F0,color:#000
    classDef hidden visibility:hidden,display:none

    node1 ~~~~ root
    node3 --- node2 & node4
    node3 ~~~ root
    node5 ~~~~ root
    node7 --- node6 & node8
    node7 ~~~ root
    node9 --- node1 & node3 & node5 & node7
    node9 ~~~ root
```

```mermaid
flowchart BT
    subgraph storage["Storage"]
        storage1[var]
        storage2[x]
        storage4[i32]:::moved
        storage3[:]:::moved
        storage5[=]
        storage6[y]
        storage8[1]:::moved
        storage7[+]:::moved
        storage9[;]
    end

    classDef moved fill:#0F0,color:#000
```

The structural concepts of bracketing nodes (`var` and `;`) and parent nodes
with a known child count (`:` and `+` with 2 children, but also `=` with 0
children) will allow checking to reconstruct the tree as it encounters nodes
during the postorder.

There are other structures that could have been used here, such as `=` being
parent of the `var` and pattern nodes, and `;` being the parent of the `=` and
assignment expression nodes. In that example alternative, the storage order
would be the same; it would only change the tree representation. The current
structure is influenced by choices in checking.

## Handling invalid parses

On an invalid parse, the output tree should still try to mirror the intended
tree structure when possible. There's a balance here, and it's not expected to
try too hard to make things correct, but outputting nodes is preferred. There
are `InvalidParse` nodes which may be used to provide a node when the planned
node kind is too difficult to get correct child counts (bracketed subtrees may
not need an `InvalidParse` node).

When marking a child node with `has_error=true`, parent nodes may also be marked
with `has_error=true`, but try to be conservative about this. As a rule of
thumb, if checking could continue on a parent node without needing the child
node to be fully checked (possibly with incomplete information), then the parent
node should not be marked as `has_error=true`. The goal remains providing
something similar to a well-formed parse tree.

In general, a parent node must have the immediate children described in
[parse/typed_nodes.h](/toolchain/parse/typed_nodes.h), unless it is marked
`has_error=true`. If this is violated for a particular parse tree, an error will
be raised in `Tree::Verify`. Note that an `InvalidParse` node is allowed as a
declaration or expression, and an `InvalidParseSubtree` is allowed as a
declaration. These invalid nodes can be added to more node categories as needed.

Child states may indicate an error to their parent using `ReturnErrorOnState`.
This is particularly intended for when a child state emits a diagnostic, to
prevent the parent state from emitting redundant diagnostics; for example, an
invalid expression might have more invalid tokens following it, and the parent
might skip those without emitting diagnostics.

## How is this accomplished?

The specific approach to producing the desired tree depends on the kind of
grammar rule being implemented, as well as the desired output tree structure.

### Introducer

**Example:** `if (c) { ... }`

Here `if` is the introducer. Many other possible introducers could occur in that
position, such as `while` or `var`, and we want to dispatch based on which token
is present. See
[parse/handle_statement.cpp](/toolchain/parse/handle_statement.cpp).

The first step is to identify the introducer token, typically using a `switch`
or `if` on the `Lex::TokenKind` at the current position:

```cpp
switch (context.PositionKind()) {
  case Lex::TokenKind::___: {
    ...
    break;
  }
  ...
}
```

There should be a `default:` (or `else`) case so every kind of token is handled.
This may be an error, in which case:

-   A [diagnostic](diagnostics.md) should be emitted.

-   An invalid parse node should be added, using something like:

    ```cpp
    context.AddLeafNode(NodeKind::InvalidParse, context.Consume(),
                        /*has_error=*/true);
    ```

-   At least one node should be consumed, particularly if it will continue with
    this state at this position, to avoid an infinite loop.

The default case may also be delegated to another state. For example, in the
state where a statement is expected, if no keyword introducer is recognized, it
switches to the expression-statement state.

Depending on the introducer, different actions can be taken. The most common
case is to:

-   Call `context.PushState(State::___);` to mark the beginning of the statement
    or declaration and indicate the state that will handle the tokens after the
    introducer.

-   Call `context.AddLeafNode(NodeKind::___, context.Consume());` to output a
    bracketing node for this introducer.

The next state can then add sibling nodes until it gets to the end of the
declaration or statement. The last token, often a semicolon `;`, is used as a
parent node to match the bracketing node of the introducer.

If the introducer token won't be used as a bracketing node, it can be
temporarily skipped after `context.PushState` by calling
`context.ConsumeAndDiscard()` instead of `context.AddLeafNode`. It must be added
to the output tree as a node by some later state, unless an error occurs. For
example, a `for` statement uses the `for` token as the root of the tree -- it
doesn't need a bracketing node since it has a fixed child count. Note that the
token was saved when the state was pushed, and can be retrieved when adding a
node as in this example:

```cpp
auto state = context.PopState();
context.AddNode(NodeKind::ForStatement, state.token, state.subtree_start,
                state.has_error);
```

If this state is for an element of a scope like the statements in a code block,
most introducer tokens indicate that the current state should be repeated, to
handle the next statement, but some other token, like a close curly brace (`}`)
means that the state should be exited.

### Optional modifiers before an introducer

**Example:** `virtual fn Foo();`

Here `fn` is the introducer, and `virtual` is an optional modifier that appears
before. See
[parse/handle_decl_scope_loop.cpp](/toolchain/parse/handle_decl_scope_loop.cpp).

Use this pattern when the goal is to produce a subtree that starts with the
introducer as a bracketing node, as in the previous case, followed by nodes for
any modifiers. Note that bracketing is needed here, since the optional modifier
nodes mean that there is not a fixed child count for the parent node. This means
shuffling the introducer node before an unknown number of modifier nodes. This
is accomplished by emitting a placeholder node for the introducer, processing
all the modifiers until reaching the introducer, filling in the placeholder with
the information about the introducer, and then finishing the rest of the
declaration or statement.

-   **Step 1**: Save the current value of `context.tree().size()`. This could be
    accomplished by calling `context.PushState()`, which saves that value in the
    `subtree_start` field of `Context::StateStackEntry`; or by constructing a
    `Context::StateStackEntry` value directly, as is done in
    [parse/handle_decl_scope_loop.cpp](/toolchain/parse/handle_decl_scope_loop.cpp).
    This marks the position of the placeholder node we are going to replace, as
    well as the beginning of the subtree we are eventually going to emit for
    this declaration or statement.

-   **Step 2**: Emit the placeholder node using
    `context.AddLeafNode(NodeKind::Placeholder, *context.position());`. The
    `NodeKind` and `Lex::TokenIndex` values will be overwritten later.

-   **Step 3**: Process tokens until we hit the introducer. All of the nodes we
    emit at this point will appear as siblings after the introducer token in the
    output tree.

-   **Step 4 - success**: If an introducer token is found, replace the
    placeholder node using something like:

    ```cpp
    context.ReplacePlaceholderNode(state.subtree_start, introducer_kind,
                                   context.Consume());
    ```

    -   `state.subtree_start` is the value of `context.tree().size()` saved in
        step 1, which marks the position of the placeholder node in the output
        parse tree.

    -   `introducer_kind` is the `NodeKind` for the introducer of this
        declaration or statement, a leaf node that will act as a bracketing node
        at the beginning of the subtree for this declaration or statement

-   **Step 4 - error**: If we run into something other than a modifier or
    introducer before finding an introducer, we need to do error handling:

    ```cpp
    context.ReplacePlaceholderNode(subtree_start, NodeKind::InvalidParseStart,
                                   *context.position(), /*has_error=*/true);
    ```

    -   Emit a [diagnostic](diagnostics.md).

    -   Replace the placeholder node (similar to step 4) with an
        `InvalidParseStart` node. It will be associated with the unexpected
        token that triggered this error.

    -   Consume input token up to the likely end of the end of the current
        statement or declaration. For example, we might consume up to a `;` or a
        token at a lesser indent level using `context.SkipPastLikelyEnd(...)`.
        It is important that we consume at least one token in the error case,
        otherwise we could have an infinite loop of generating the same error on
        the same token.

    -   Emit a `InvalidParseSubtree` node. This will be the parent of any
        emitted modifier nodes, and will be bracketed by the `InvalidParseStart`
        node emitted above. It should be associated with the last token
        consumed.

        ```cpp
        // Set `iter` to the last token consumed, one before the current position.
        auto iter = context.position();
        --iter;
        context.AddNode(NodeKind::InvalidParseSubtree, *iter, subtree_start,
                        /*has_error=*/true);
        ```

-   **Step 5**: (If success at step 4) Push whatever states are to be used to
    parse the rest of the declaration. The first state pushed (the last state to
    be processed) will handle the end of this declaration. That pushed state
    should have a `subtree_start` field set to the value of
    `context.tree().size()` saved in step 1.

-   **Step 6**: When handling the state for the end of the declaration, emit the
    root node of subtree:

    ```cpp
    state = context.PopState();
    context.AddNode(NodeKind::___, context.Consume(),
                    state.subtree_start, state.has_error);
    ```

    -   This `state.subtree_start` will mark everything since the bracketing
        introducer node as the children of this node.

### Something required in context

FIXME

Example: name after introducer
[parse/handle_decl_name_and_params.cpp](/toolchain/parse/handle_decl_name_and_params.cpp)

Example: "`[` _implicit parameter list_ `]`" after `impl forall`
[parse/handle_impl.cpp](/toolchain/parse/handle_impl.cpp)

### Optional clauses

#### Case 1: introducer to optional clause is used as parent node

**Example:** The optional `-> <return type expression>` in a function signature
uses this pattern, so `fn foo() -> u32;` is transformed to:

```yaml
  {kind: 'FunctionIntroducer', text: 'fn'},
  {kind: 'IdentifierName', text: 'foo'},
    {kind: 'TuplePatternStart', text: '('},
  {kind: 'TuplePattern', text: ')', subtree_size: 2},
    {kind: 'UnsignedIntTypeLiteral', text: 'u32'},
  {kind: 'ReturnType', text: '->', subtree_size: 2},
{kind: 'FunctionDecl', text: ';', subtree_size: 7},
```

Note how the `->` token becomes a `ReturnType` node in the output tree, and is
moved after the `u32` type expression that becomes its child. Compare with the
parse tree output for `fn foo();` which has no `ReturnType` node:

```yaml
  {kind: 'FunctionIntroducer', text: 'fn'},
  {kind: 'IdentifierName', text: 'foo'},
    {kind: 'TuplePatternStart', text: '('},
  {kind: 'TuplePattern', text: ')', subtree_size: 2},
{kind: 'FunctionDecl', text: ';', subtree_size: 5},
```

Here is the code from
[parse/handle_function.cpp](/toolchain/parse/handle_function.cpp) that does
this:

```cpp
auto HandleFunctionAfterParams(Context& context) -> void {
  ...
  // If there is a return type, parse the expression before adding the return
  // type node.
  if (context.PositionIs(Lex::TokenKind::MinusGreater)) {
    context.PushState(State::FunctionReturnTypeFinish);
    context.ConsumeAndDiscard();
    context.PushStateForExpr(PrecedenceGroup::ForType());
  }
}

auto HandleFunctionReturnTypeFinish(Context& context) -> void {
  auto state = context.PopState();

  context.AddNode(NodeKind::ReturnType, state.token, state.subtree_start,
                  state.has_error);
}
```

The `->` token is saved by `context.PushState(`...`)`, so it is available as
`state.token` when calling
`context.AddNode(NodeKind::ReturnType, state.token,`...`)` later in
`HandleFunctionReturnTypeFinish`.

Also see how the optional initializer is handled on `var`, treating the `=` as
its introducer in `HandleVarAfterPattern` and `HandleVarInitializer` in
[parse/handle_var.cpp](/toolchain/parse/handle_var.cpp).

#### Case 2: parent node is required token after optional clause, with different parent node kinds for different options

**Example:** The optional type expression before `as` in `impl as` is
represented by producing two different output parse nodes for `as`. It outputs a
`DefaultSelfImplAs` node with no children when the type expression is absent,
and otherwise a `TypeImplAs` parse node with the type expression as its child.

So `impl bool as Interface;` is transformed to:

```yaml
  {kind: 'ImplIntroducer', text: 'impl'},
    {kind: 'BoolTypeLiteral', text: 'bool'},
  {kind: 'TypeImplAs', text: 'as', subtree_size: 2},
  {kind: 'IdentifierNameExpr', text: 'Interface'},
{kind: 'ImplDecl', text: ';', subtree_size: 5},
```

while `impl as Interface;` is transformed to:

```yaml
  {kind: 'ImplIntroducer', text: 'impl'},
  {kind: 'DefaultSelfImplAs', text: 'as'},
  {kind: 'IdentifierNameExpr', text: 'Interface'},
{kind: 'ImplDecl', text: ';', subtree_size: 4},
```

This is handled by the `ExpectAsOrTypeExpression` code from
[parse/handle_impl.cpp](/toolchain/parse/handle_impl.cpp):

```cpp
if (context.PositionIs(Lex::TokenKind::As)) {
  // as <expression> ...
  context.AddLeafNode(NodeKind::DefaultSelfImplAs, context.Consume());
  context.PushState(State::Expr);
} else {
  // <expression> as <expression>...
  context.PushState(State::ImplBeforeAs);
  context.PushStateForExpr(PrecedenceGroup::ForImplAs());
}
```

and then `HandleImplBeforeAs` creates the parent node in the second case:

```cpp
auto state = context.PopState();
if (auto as = context.ConsumeIf(Lex::TokenKind::As)) {
  context.AddNode(NodeKind::TypeImplAs, *as, state.subtree_start,
                  state.has_error);
  context.PushState(State::Expr);
} else {
  if (!state.has_error) {
    CARBON_DIAGNOSTIC(ImplExpectedAs, Error,
                      "Expected `as` in `impl` declaration.");
    context.emitter().Emit(*context.position(), ImplExpectedAs);
  }
  context.ReturnErrorOnState();
}
```

Note (1) that the `state.subtree_start` value comes from the
`context.PushState(State::ImplBeforeAs);` before parsing the type expression,
and that is how that type expression ends up as the child of the created
`TypeImplAs` node. Unlike
[the previous case 1](#case-1-introducer-to-optional-clause-is-used-as-parent-node),
though, the parent node uses the token after the optional expression, rather
than an introducer token for the optional clause.

Note (2) how `HandleImplBeforeAs` handles three cases of errors:

-   `as` present but an error in the child type expression -> error on the
    output `TypeImplAs` node, but not propagated to the parent.
-   Error from no `as` present but the type expression was okay -> create a new
    error.
-   There was error from the child type expression and no `as` present -> no new
    diagnostic, we suppress errors once one is emitted until we can recover.

If there is no `as` token, we don't output either a `TypeImplAs` or a
`DefaultSelfImplAs` node, as required by the parent node, so in those cases we
mark the parent as having an error.

#### Case 3: optional sibling

> TODO: This was changed by
> [#3678](https://github.com/carbon-language/carbon-lang/pull/3678) and needs to
> be updated.

**Example:** The optional type expression before `as` in `impl as` is output as
an optional sibling subtree between the `ImplIntroducer` node for the `impl`
introducer and the `ImplAs` node for the required `as` keyword.

`impl bool as Interface;` is transformed to:

```yaml
  {kind: 'ImplIntroducer', text: 'impl'},
  {kind: 'BoolTypeLiteral', text: 'bool'},
  {kind: 'ImplAs', text: 'as'},
  {kind: 'IdentifierNameExpr', text: 'Interface'},
{kind: 'ImplDecl', text: ';', subtree_size: 5},
```

while `impl as Interface;` is transformed to:

```yaml
  {kind: 'ImplIntroducer', text: 'impl'},
  {kind: 'ImplAs', text: 'as'},
  {kind: 'IdentifierNameExpr', text: 'Interface'},
{kind: 'ImplDecl', text: ';', subtree_size: 4},
```

This is handled by the `ExpectAsOrTypeExpression` code from
[parse/handle_impl.cpp](/toolchain/parse/handle_impl.cpp):

```cpp
if (context.PositionIs(Lex::TokenKind::As)) {
  // as <expression> ...
  context.AddLeafNode(NodeKind::ImplAs, context.Consume());
  context.PushState(State::Expr);
} else {
  // <expression> as <expression>...
  context.PushState(State::ImplBeforeAs);
  context.PushStateForExpr(PrecedenceGroup::ForImplAs());
}
```

and then `HandleImplBeforeAs` follows
[the "something required in context" pattern](#something-required-in-context) to
deal with the `as` that follows when the type expression is present.

### Operators

FIXME

An independent description of our approach:
["Better operator precedence" on scattered-thoughts.net](https://www.scattered-thoughts.net/writing/better-operator-precedence/)
