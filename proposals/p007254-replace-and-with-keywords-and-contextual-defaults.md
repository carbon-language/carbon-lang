# Replace `:!` and `:?` with keywords and contextual defaults

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/7254)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
-   [Details](#details)
    -   [Phase Keywords and Contextual Defaults](#phase-keywords-and-contextual-defaults)
        -   [Contextual Defaults](#contextual-defaults)
    -   [Associated Constants](#associated-constants)
    -   [Extended Types](#extended-types)
        -   [Future Work on Extended Types](#future-work-on-extended-types)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Keep the `:!` syntax](#keep-the--syntax)
    -   [Alternative keyword names](#alternative-keyword-names)
    -   [Use `template generic` instead of just `template`](#use-template-generic-instead-of-just-template)
    -   [Context-independent syntax](#context-independent-syntax)
    -   [Erased model for generics](#erased-model-for-generics)
    -   [Context-sensitive defaults based on parameter type](#context-sensitive-defaults-based-on-parameter-type)
    -   [Allow redundant phase keywords](#allow-redundant-phase-keywords)
    -   [Use `exprtype` and `expr` keywords](#use-exprtype-and-expr-keywords)

<!-- tocstop -->

## Abstract

This proposal removes the `:!` syntax for generics and templates in favor of
keywords (`generic`, `template`, `runtime`) and contextual defaults for phase.
It also suggests replacing `:?` from proposal #5389 with `fwd` and renaming
"forms" to "extended types".

## Problem

The `:!` syntax for generics and templates has several issues:

-   It doesn't work well for controlling the phase for functions.
-   The connection between generics/templates and `!` is tenuous and not an
    effective mnemonic.
-   It is very inventive syntax with little familiarity from other languages.
-   It makes Carbon code using generics start to look like ASCII-art due to
    dense punctuation.
-   It is in tension with more compelling use cases for `!`, such as for
    operations that are required to succeed or terminate (for example,
    unwrapping optionals).

## Background

The `:!` syntax was originally chosen to evoke the idea of "phase", using `!` to
mark parameters that belong to an earlier (compile-time) phase of evaluation.
Similarly, the `:?` syntax in the current revision of proposal #5389 is intended
to mark deduced _form bindings_: parameters that capture extended type
information (what was called a "form") about an expression, such as its value
category and phase, rather than just its object type.

These issues were discussed in leads issue #6932, and a direction was decided to
move away from punctuation and towards keywords and contextual defaults.

## Proposal

We propose to:

1.  Remove `:!` syntax for generics and templates.
2.  Introduce contextual defaults for phase:
    -   Parameters to compile-time entities (interfaces, impls, classes) are
        checked generic parameters by default.
    -   Deduced function parameters are checked generic parameters by default.
    -   Explicit function parameters are runtime by default.
3.  Allow overriding defaults with keywords `template`, `generic`, and
    `runtime`.
4.  Disallow keywords that match the contextual default to ensure consistency.
5.  Change the underlying terminology from "forms" to "extended types" and
    introduce `exttype`. Also suggest replacing the `:?` and `->?` syntax from
    pending proposal #5389 with a binding modifier `fwd` and corresponding
    return syntax.

## Details

### Phase Keywords and Contextual Defaults

Parameter phase is primarily determined by the context of the parameter:

-   Parameters to compile-time entities (interfaces, impls, classes) are checked
    generic parameters by default.
-   Deduced function parameters are checked generic parameters by default.
-   Explicit function parameters are runtime by default.

These defaults can be overridden where meaningful by using one of the following
keywords:

-   `runtime`: Causes a parameter to be a runtime parameter in the deduced
    parameter context, if we ever decide to support runtime deduced parameters.
-   `generic`: Causes a parameter to be a checked generic when in an explicit
    function parameter context.
-   `template`: Causes a parameter to be a template generic in any of the three
    contexts.

#### Contextual Defaults

-   **Compile-time entities**: Parameters to entities like `interface`, `impl`,
    and `class` are checked generic parameters by default.

    ```carbon
    interface I(T: type) { ... } // T is a checked generic parameter
    ```

    They can be marked as `template`:

    ```carbon
    class C(template T: type) { ... } // T is a template generic parameter
    ```

-   **Deduced function parameters**: Parameters in `[]` for functions default to
    checked generic parameters.

    ```carbon
    fn F[T: type](arg: T); // T is a checked generic parameter
    ```

    They can be marked as `template`:

    ```carbon
    fn F[template T: type](arg: T); // T is a template generic parameter
    ```

    If we ever add deduced runtime parameters (anticipated for scoped parameters
    like allocators), they would be marked with the `runtime` keyword:

    ```carbon
    fn F[runtime heap: Heap](T: type, arg: T) -> T*; // heap is a runtime parameter
    ```

-   **Explicit function parameters**: Parameters in `()` for functions default
    to runtime parameters.

    ```carbon
    fn F(arg: i32); // arg is a runtime parameter
    ```

    They can be marked as `generic` or `template`:

    ```carbon
    fn F(generic T: type, arg: T); // T is a checked generic parameter
    ```

Keywords are only allowed where needed to override the contextual default. This
avoids confusion and ensures consistency.

The checked generic default for deduced parameters applies only to declared
parameters in the `[]` list. Lambda captures, which also appear in `[]` but are
syntactically distinguished (they are not declared names), are not affected by
this default. Instead, a capture retains the phase of the expression being
captured, which we expect to be important for the usability of lambdas.

### Associated Constants

Associated constants in interfaces require no extra keywords. Their meaning is
guided by the context of the interface definition itself.

The conceptual model is that an interface is essentially a class whose phase is
inherently the symbolic compile-time (generic) phase. As a consequence, its
fields (the associated constants) act as generic constants naturally, and
placing an additional phase keyword on them would be redundant and disallowed.
The same logic applies when implementing those constants in an `impl`, which
already uses distinct syntax to assign them.

### Extended Types

This proposal replaces the concept of "forms" (as described in
[`/docs/design/values.md`](/docs/design/values.md)) with **extended types**.

The term "forms" was originally used to generalize types to include expression
category, phase, and value. However, this terminology was found to conflict
confusingly with the concept of "unformed state". To resolve this, we move to a
model where these are considered "extended types", connecting them more directly
to the type system while preserving `type` for standard object types.

Under this new design:

-   The literal expression `form(expr)` is renamed to `exttype(expr)`.
-   The type of extended types is `Core.ExtType` (replacing `Core.Form`).
-   The previous `:?` syntax for deduced form bindings is suggested to be
    replaced by a binding modifier `fwd`. This modifier causes the
    right-hand-side of the binding's `:` to be converted to `Core.ExtType`
    rather than `type`. This is a suggested (but not fully decided) direction
    for pending proposal #5389 to go with the syntax.
-   `fwd` is also suggested for use in the return signature (for example,
    `-> fwd T`) to forward the extended type information. Note that this may end
    up being more significant than just a syntactic replacement: it remains to
    be decided in proposal #5389 whether `fwd` must appear directly after the
    `->` (matching how `->?` works in that proposal currently) or if it can be
    used within tuple syntax in the return type, similar to how `ref` is
    allowed.

This approach allows us to reclaim high-value punctuation like `?` for other
uses (such as optional types) while providing a more explicit and less
punctuated syntax for advanced generic programming.

Example:

```carbon
fn F[T: Core.ExtType](fwd arg: T) -> fwd T;
```

> **Open question:** Should we require the `fwd` modifier on call arguments as
> well, analogously to how `ref` is required on arguments for reference
> parameters?

#### Future Work on Extended Types

Once the design for extended types in proposal #5389 is more complete, we may
also want to replace `Core.ExtType` with a new built-in keyword `exttype` for
the type of extended types, and potentially replace `exttype(expr)` literals
with library entities. This would make `exttype` analogous to `type` in the
grammar.

## Rationale

This proposal advances the following Carbon goals and principles:

-   [**Code that is easy to read, understand, and write**](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write):
    Removing dense punctuation in favor of keywords with meaningful names makes
    code less like ASCII-art and more immediately readable. The contextual
    defaults are carefully chosen to match what nearly all code uses in
    practice, keeping keywords sparse while remaining explicit when they are
    needed.

-   [**Software and language evolution**](/docs/project/goals.md#software-and-language-evolution):
    Reclaiming `!` and `?` as punctuation opens up syntax space for other
    high-value features. In particular, `!` is a strong candidate for operations
    that are required to succeed or terminate (for example, unwrapping
    optionals), which would have been visually ambiguous if `!` were also used
    for generics.

-   [**Progressive disclosure**](/docs/project/principles/progressive_disclosure.md):
    The contextual defaults allow learners to work with generic interfaces and
    classes without needing to understand or type phase keywords at first.
    Keywords only become relevant when departing from the defaults, which is a
    rarer, more advanced case. This mirrors how Carbon teaches other concepts
    progressively.

-   [**Prefer only one way to do a given thing**](/docs/project/principles/one_way.md):
    Disallowing redundant phase keywords (those that match the contextual
    default) ensures there is exactly one canonical way to write each parameter
    declaration, consistent with how Carbon handles other defaults such as
    `public` access.

## Alternatives considered

### Keep the `:!` syntax

One alternative was to retain the existing punctuation-based syntax where `:!`
is used to denote checked generic parameters and template generic parameters.

-   **Advantages**:
    -   Maintains continuity with the previously established design.
    -   Is very concise, requiring no keywords.
    -   A parameter's phase is encoded explicitly in its syntax and is
        independent of its position, so moving a parameter between the `[]` and
        `()` lists does not change its meaning. Under the proposed contextual
        defaults, such a move changes the default phase and requires adding a
        keyword to preserve it, making this kind of refactoring of a function
        signature slightly less straightforward.
-   **Disadvantages**:
    -   The syntax makes code look like "ASCII-art" due to the high density of
        punctuation.
    -   The connection between `!` and generics is not an effective mnemonic.
    -   It blocks other potential uses for `!`, such as for operations that are
        required to succeed or terminate (for example, unwrapping optionals).
    -   It does not scale well to controlling the phase of function parameters.
-   **Decision**: This alternative was rejected because the disadvantages in
    readability and extensibility outweigh the benefit of conciseness, including
    the modest refactoring cost noted above. The leads decided to move towards
    keywords and contextual defaults.

### Alternative keyword names

Several alternative keywords were considered for the three phase keywords.

For `generic`, the key candidates considered were:

-   **`symbolic`**: Reflects the technical description of symbolic compile-time
    evaluation.
-   **`comptime`**: Reflects when the value is known (compile time).
-   **`checked`**: Reflects the semantic behavior that these parameters are
    type-checked at the definition site.

For `runtime`, the main candidate discussed was:

-   **`dynamic`**: Reflects that values are dynamically determined at runtime.

Looking across these options:

-   **Advantages**:
    -   `symbolic` is more technically precise for compiler experts as it
        reflects the symbolic evaluation phase.
    -   `comptime` is a familiar pattern from other modern systems languages.
    -   `checked` is highly precise about the checking model used, matches the
        terminology we use in the design, and matches the structure of
        `template`.
    -   `dynamic` uses a term that is recognizable for runtime behavior.
-   **Disadvantages**:
    -   None of the alternatives offer as strong a mnemonic connection to the
        _programming concepts_ they represent as the chosen keywords.
    -   `symbolic` is inaccessible jargon and less teachable to developers not
        familiar with compiler or type theory terminology.
    -   `comptime` describes _when_ the value is known, not _why_ or _how_ it is
        used, lacking a connection to generic programming.
    -   `checked` focuses on the implementation mechanism (checking) rather than
        the programmer's intent (generic programming) and loses the immediate
        familiarity of the term `generic`.
    -   `dynamic` conflicts with the well-established use in dynamic dispatch
        (for example, Rust's `dyn`), making it a poor fit for Carbon.
-   **Decision**: The chosen keywords (`generic`, `template`, `runtime`) were
    found to best balance accessibility with precision. `generic` in particular
    connects directly to the well-known concept of generic programming, making
    it both familiar and teachable.

### Use `template generic` instead of just `template`

An alternative considered was to require `template generic` (two keywords) for
template generic parameters, and `generic` for checked generic parameters, to
make it clear that templates _are_ generics.

Under this model, the terminology is that we have "generic parameters" that come
in two semantic forms: "checked generic parameters" and "template generic
parameters". Both of these are considered "generic parameters". The _default_
semantic is checked generic parameters, so when a parameter is marked `generic`
(or defaults to it), it gets that semantic. The rejected alternative would be to
use both keywords as `template generic` for the template case, rather than
omitting the `generic` keyword and just using `template`.

-   **Advantages**:
    -   The syntax would more strictly reflect the terminology that templates
        are a kind of generic.
-   **Disadvantages**:
    -   It makes the syntax significantly more verbose in a case where there is
        nothing else that could be meant. The _only_ way to have the `template`
        keyword on a parameter is for it to be a generic parameter, so adding
        `generic` provides no additional information.
-   **Decision**: Rejected in favor of using just `template` to avoid
    unnecessary verbosity.

### Context-independent syntax

An alternative approach proposed making the phase of every parameter fully
explicit in its declaration, without any contextual defaults. The specific
proposal from the discussion used a `static` modifier for compile-time value
parameters, so that the phase could always be read directly from the declaration
without needing to know whether the parameter appears in `()` or `[]`:

```carbon
fn MakeArray(T: type, static Length: i32) -> Array(T, Length);
fn ReverseArray[T: type, static Length: i32](ref a: Array(T, Length));
```

This is analogous to how `ref` and `val` modifiers make value categories
explicit today, with the goal of making each parameter declaration
self-contained.

-   **Advantages**:
    -   Each parameter declaration contains all the information needed to
        determine its phase, without requiring knowledge of the surrounding
        syntactic context.
    -   Avoids any cognitive overhead from remembering contextual defaults.
-   **Disadvantages**:
    -   `static` is heavily overloaded in C++, covering storage duration, class
        membership, and file-scope linkage, which creates significant confusion
        for C++ developers migrating to Carbon.
    -   Types like integers can be used in both runtime and compile-time
        contexts (for example, as array sizes). Requiring an explicit `static`
        keyword for compile-time integers creates pressure towards having
        separate compile-time and runtime vocabulary types, which Carbon has
        aimed to avoid to keep the type vocabulary compact.
-   **Decision**: Rejected in favor of contextual defaults. The chosen defaults
    align with what nearly all code does in practice (most explicit function
    parameters are runtime, and most parameters to interfaces and classes are
    checked generics), so keywords remain sparse while still being explicit when
    non-default behavior is needed. The `static` keyword in particular was found
    to have significant overloading concerns coming from C++.

### Erased model for generics

An alternative approach proposed using _type erasure_ as the foundational mental
model for generic parameters, paralleling the way languages like Java implement
generics. Under this model, a generic type parameter is said to be "erased" at
runtime: the type information is available at compile time but not preserved in
the runtime representation. This would use `erased` as the keyword instead of
`generic`:

```carbon
// T is erased (available at compile time, erased at runtime)
interface I(T: type) {
  fn Op(self, arg: T) -> T;
}

// Explicit erased parameter in a function
fn ScopedParams[runtime heap: Heap](erased T: type) -> T*;
```

This model has particular implications for _associated constants_ in interfaces.
Under the erased model, associated constants would be thought of as values that
are erased from the runtime representation (present at compile time but not
available at runtime), rather than as fields of a compile-time class that are
inherently generic by context.

-   **Advantages**:
    -   "Erased" is technically accurate in certain respects: when using Carbon
        checked generics (as opposed to template generics), the specific type
        bound to a checked generic parameter is not available at runtime.
    -   Connects to a concept familiar from type erasure literature and
        languages like Java, where this is the standard implementation model for
        generics.
-   **Disadvantages**:
    -   The term "erased" focuses on what is _lost_ at runtime rather than what
        the concept _enables_; it describes an implementation detail rather than
        the programming paradigm. The keyword `generic` more directly connects
        to the reason a developer reaches for this feature.
    -   Carbon's generics are not purely erasure-based: checked generics may use
        erasure techniques, but templates generate fully specialized code. Using
        "erased" would imply a single implementation strategy that doesn't
        capture the full picture of Carbon's compile-time programming model.
    -   The interface-as-compile-time-class model chosen for Carbon makes
        associated constants more naturally generic: an interface is treated as
        a class whose "evaluation time" is inherently the symbolic compile-time
        phase, so its fields act as generic constants by context, with no extra
        keyword required. The erased model framing fits less cleanly with this
        interface design.
    -   The `generic` versus `template` terminology split, which is the clearest
        way to distinguish the two distinct kinds of compile-time parameters in
        Carbon, is obscured by "erased" framing, since templates are not erased.
-   **Decision**: Rejected in favor of the `generic`/`template` split and the
    interface-as-compile-time-class model. The team preferred a terminology that
    describes the _programming concept_ rather than an implementation detail,
    and found the model where interface fields are inherently generic by context
    to be more intuitive and consistent with the rest of the design.

### Context-sensitive defaults based on parameter type

One alternative suggested was to make explicit function parameters default to
checked generic if they cannot be represented at runtime (such as types). This
would allow omitting `generic` even in the explicit `()` parameter list when the
parameter type makes the phase unambiguous:

```carbon
fn F1[Q: type](arg1: Q, QQ: type, arg2: QQ) -> (Q, QQ);
```

-   **Advantages**:
    -   Allows omitting keywords in more cases, reducing verbosity further.
    -   Creates a natural feel where `T: type` always implies compile-time,
        regardless of position.
-   **Disadvantages**:
    -   Types like integers can be used in both runtime and compile-time
        contexts, for example as array size parameters. Requiring `generic` for
        compile-time integers but not for compile-time types creates an
        inconsistent rule that would be difficult to learn.
    -   This creates pressure towards having separate compile-time and runtime
        vocabulary types (for example, a compile-time integer versus a runtime
        integer), which Carbon has aimed to avoid to keep the type vocabulary
        compact.
    -   Determining whether a keyword is required depends on the type of the
        parameter, which requires resolving imports before the parser can
        determine the meaning. This loses the benefit of a simple, purely local
        syntactic rule, the same benefit that `:!` provided today.
-   **Decision**: Rejected due to the added complexity, the inconsistency
    introduced by types usable in both phases, and the loss of a simple local
    syntactic rule. The chosen defaults (based on syntactic position, not
    parameter type) are easier to explain and implement.

### Allow redundant phase keywords

Another alternative was to allow keywords matching the contextual default to be
used optionally, for example allowing `generic T: type` in a deduced parameter
list where checked generic is already the default semantic.

-   **Advantages**:
    -   Provides a simpler mental model for beginners: the rule would be "just
        always write the keyword if you want to be explicit" rather than "write
        it only when non-default."
    -   Would allow users to treat the shorthand as a style rule enforced by a
        linter, rather than a language rule enforced by the compiler.
    -   Supports a progressive learning path where users learn the explicit form
        first and adopt the shorthand later.
-   **Disadvantages**:
    -   Creates two syntactically valid ways to say the same thing, which
        confuses readers who may wonder why the author was explicit about the
        default, suggesting intentionality where there is none.
    -   Inconsistent with how Carbon handles other defaults. For example, Carbon
        does not allow writing `public` in a context where `public` is already
        the default access, for the same reason: explicit statement of a default
        implies it was chosen deliberately, which is misleading.
-   **Decision**: Rejected to ensure consistency and avoid confusion, following
    the established Carbon pattern of not allowing redundant keywords that match
    a contextual default. The compiler enforcing this as an error (rather than a
    linter warning) means the rule is consistently applied and the absence of a
    keyword reliably communicates "this is the default" across all code.

### Use `exprtype` and `expr` keywords

One alternative considered for replacing "forms" was to use the terminology
"expression types" with `exprtype` as the bottom type and `expr` as the binding
modifier.

-   **Advantages**:
    -   Maintains progressive disclosure by keeping `type` as the primary term
        for object types and qualifying it as `exprtype` for expression types.
    -   Connects directly to the concept of expression metadata.
-   **Disadvantages**:
    -   It has a slightly awkward construction where the narrower term ("type")
        is the base term, and the broader term ("expression type") is qualified.
    -   It confusingly implies that it refers to the _type of the expression_,
        while we want that use of the term "type" to not include the extended
        information.
    -   It also implies with `expr` on a binding that the expression itself is
        bound and captured, rather than being evaluated first. Hard to explain
        that this matches the _evaluated_ expression.
-   **Decision**: This alternative was rejected in favor of the **Extended
    Types** model. The team preferred "extended types" as the terminology anchor
    (yielding `Core.ExtType`). For the binding modifier, `fwd` was chosen
    because it connects to the use case of forwarding extended type information
    (similar to C++ `std::forward`) and fits well as a three-letter keyword
    similar to `ref`, `var`, and `val`.
