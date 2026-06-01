# Principle: information accumulation

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/875)

<!-- toc -->

## Table of contents

-   [Problem](#problem)
-   [Background](#background)
    -   [Single-pass "splat" compilation](#single-pass-splat-compilation)
    -   [Global consistency](#global-consistency)
    -   [The C++ compromise](#the-c-compromise)
    -   [Separate declarations and definitions](#separate-declarations-and-definitions)
-   [Proposal](#proposal)
-   [Details](#details)
    -   [Goals](#goals)
    -   [Choice of alternative](#choice-of-alternative)
-   [Rationale based on Carbon's goals](#rationale-based-on-carbons-goals)
-   [Alternatives considered](#alternatives-considered)
    -   [Separate declaration and definition](#separate-declaration-and-definition)
        -   [Allow separate declaration and definition](#allow-separate-declaration-and-definition)
            -   [Analysis](#analysis)
        -   [Disallow separate declaration and definition](#disallow-separate-declaration-and-definition)
            -   [Analysis](#analysis-1)
    -   [Information flow](#information-flow)
        -   [Strict top-down](#strict-top-down)
            -   [Analysis](#analysis-2)
        -   [Strict global consistency](#strict-global-consistency)
            -   [Block-scope exception](#block-scope-exception)
            -   [Analysis](#analysis-3)
        -   [Top-down with minimally deferred type checking](#top-down-with-minimally-deferred-type-checking)
            -   [Analysis](#analysis-4)
        -   [Top-down with deferred method bodies](#top-down-with-deferred-method-bodies)
            -   [Analysis](#analysis-5)
        -   [Context-sensitive local consistency](#context-sensitive-local-consistency)
            -   [Analysis](#analysis-6)

<!-- tocstop -->

## Problem

We should have consistent rules describing what information about a program is
visible where.

## Background

Information in a source file is provided incrementally, with each source
utterance providing a small piece of the overall picture. Different languages
have different rules for which information is available where.

### Single-pass "splat" compilation

In C and other languages of a similar age, single-pass compilation was highly
desirable, due to resource limits and performance concerns. In these languages:

-   Information is accumulated top-down, and can only be used lexically after it
    appears.
-   Most information can be discarded soon after it is provided: function bodies
    don't need to be kept around once they've been converted to the output
    format, and no information on local variable or parameter names needs to
    persist past the end of the variable's scope. However, the types of globals
    and the contents of type definitions must be retained.
-   The behavior of an entity can be different at different places in the same
    source file. An early use may fail if it depends on information that's
    provided later, and in some cases a later use may fail when an earlier use
    succeeded because the use is invalid in a way that was not visible at the
    point of an earlier use.

### Global consistency

In more modern languages such as C#, Rust, Java, and Swift, there is no lexical
information ordering. In these languages:

-   Information is effectively accumulated and processed in separate passes.
-   The language design and implementation ensure that the behavior of an entity
    is the same everywhere: both before its definition, after its definition,
    within its definition, and in any other source file in which it was made
    visible.
-   Dependency cycles between program properties are carefully avoided by the
    language designers.

### The C++ compromise

In C++, a hybrid approach is taken. There is a C-like lexical information
ordering rule, but this rule is subverted within classes by -- effectively --
reordering certain parts of a class that appear within the class definition so
that they are processed after the class definition. This primarily applies to
the bodies of member functions. Here:

-   Information is mostly accumulated top-down, and is accumulated fully
    top-down after the reordering step.
-   The behavior of a class is the same within member function bodies that are
    defined inside the class as it is within member function bodies defined
    lexically after the class.
-   The language designers need to ensure that the bounds of the member function
    bodies and similar constructs can be determined without parsing them, so
    that the late-parsed portions can be separated from the early-parsed
    portions. In C++, this was not done successfully, and there are constructs
    for which this determination is very hard or impossible.

### Separate declarations and definitions

Somewhat separate from the direction of information flow is the ability to
separate the information about an entity into multiple distinct regions of
source files. In C and C++, entities can be separately declared and defined. As
a consequence, these languages need rules to determine whether two declarations
declare the same entity.

In C++, especially for templated declarations, these rules can be incredibly
complex, and even now, more than 30 years after the introduction of templates in
C++, [basic questions are not fully answered](https://wg21.link/cwg2), and
implementations disagree about which declarations declare the same entity in
fairly simple examples.

One key benefit of this separation is in reduction of _physical dependencies_:
in order to validate a usage of an entity, we need only see a source file
containing a declaration of that entity, and need never consider the source file
containing its definition. This both reduces the number of steps required for an
incremental rebuild and reduces the input information and processing required
for each individual step.

The ability to break physical dependencies is limited to the cases where
information can actually be hidden from the users of the entity. For example, if
the user actually needs a function body, either because they will evaluate a
call to the function during compilation or because they will inline it prior to
linking, it cannot be physically isolated from the user of that information. As
a consequence, in C++, a programmer must carefully manage which information they
put in the source files that are exposed to client code and which information is
kept separate.

Another key benefit is that the exported interface of a source file can become
more readable, by presenting an interface that contains only the facts that are
salient to a user and not the implementation details.

## Proposal

Carbon will use a
[top-down approach with deferred method bodies](#top-down-with-deferred-method-bodies),
with [forward declarations](#allow-separate-declaration-and-definition).

In any case where the [strict global consistency](#strict-global-consistency)
model with the [block-scope exception](#block-scope-exception) would consider
the program to be invalid or would assign it a different meaning, the program is
invalid.

Put another way: we accept exactly the cases where those two models both accept
and give the same result, and reject all other cases.

## Details

Each entity has only one meaning and only one set of properties; that meaning
and those properties do not depend on which source file you are within or where
in that source file you are. However, not all information will be available
everywhere, and it is an error to attempt to use information that is not
available -- either because it's in a different source file and not exported or
not imported, or because it appears later in the same source file.

For example, a full definition of a class `Widget` may be available in some
parts of the program, with only a forward declaration available in other parts
of the program. In this case, any attempt to use `Widget` where only a forward
declaration is available will always either result in the program being invalid
or doing the same thing that would happen if the full information were
available. This is a stronger guarantee than is provided by C++'s One Definition
Rule, which guarantees that every use sees an equivalent definition or no
definition, but does not guarantee that every use has the same behavior -- in
C++ a function could do one thing if `Widget` is defined and a different thing
if `Widget` is only forward declared, and that will not be possible in Carbon.

Name lookup only provides earlier-declared names. However, the bodies of class
member functions that are defined inside the class are reordered to be
separately defined after the class, so all class member names are declared
before the class body is seen. Unlike in C++, classes are either incomplete or
complete, with no intermediate state. Any attempt to perform name lookup into an
incomplete class is an error, even if the name being looked up has already been
declared.

```
class A {
  fn F() {
    // OK: A and B are complete class types here.
    var b: A.B;
  }

  class B {}

  // Error: cannot look up B within incomplete class A.
  var p: A.B*;
}
```

Forward declarations are permitted in order to allow separation of interface and
implementation whenever the developer wishes to do so, and introduce names early
in order to break cycles or when the developer does not wish to define entities
in topological order. All declarations of an entity are required to be part of
the same library, but can be in different source files. If a point within a
source file can only see a declaration of an entity and not its definition, some
uses of that entity will be invalid that would be valid if the definition were
visible. For example, without a class definition, we may be unable to create
instances of the class, and without a function definition, we may be unable to
use the function to compute a compile-time constant.

### Goals

For this proposal, we have the following goals as refinements of the overall
Carbon goals:

-   _Comprehensibility._ Our rules should be understandable, and should minimize
    surprise and gotchas. Our behavior should be self-consistent, and
    explainable in only a few sentences.
-   _Ergonomics._ It should be easy to express common developer desires, without
    a lot of boilerplate or repetitive code.
-   _Readability._ Code written using our rules should be as straightforward as
    possible for Carbon developers to read and reason about.
-   _Efficient and simple compilation._ It should be relatively straightforward
    to implement our semantic rules. Implementation heroics shouldn't be
    required, and the number of special cases required should be minimized.
-   _Diagnosability._ An implementation should be able to explain coding errors
    in ways that are easy to understand and are well-correlated with the error
    and its remedy. Diagnostics should appear in an order and style that guides
    the developer through logical steps to fix their mistakes.
-   _Toolability._ Relatively simple tools should be able to understand simpler
    properties of Carbon code. It should ideally be possible to identify which
    names can be used in a particular context and what those names mean without
    full processing. It should ideally be possible to gather useful and mostly
    complete information about a potentially-invalid source file that is
    currently being edited, for which it may be desirable to assume there is a
    "hole" in the source file at the cursor position that will be filled by
    unknown code.

### Choice of alternative

We have a large space of options here, all of which have both benefits and
drawbacks. The rationale for our choice is as follows:

The first consideration is whether to provide a
[globally consistent view](#strict-global-consistency) of code. Providing such a
view would require us to find a compatible build process that can scale to large
projects; see the [detailed analysis](#analysis-3) for details. Moreover, this
is likely to be a choice that is relatively hard to evolve away from. A
non-scalable build process would be an existential threat to the Carbon language
under its stated goals, so we set this option aside. This should be revisited if
a complete design for a scalable build process with a globally-consistent
semantic model is proposed.

Once we step away from strict global consistency, there will be cases where the
meaning of an entity differs in different parts of the program. In order for
programs to behave consistently and predictably, especially in the presence of
templates and generics, we do not want to allow the same operation to have two
or more different behaviors depending on which information is available. So we
choose to make programs invalid if they would make use of information that is
not available.

Even though we have chosen not to provide a globally consistent view, we could
still choose to provide a consistent view throughout a package, library, or
source file, such as in the
[context-sensitive local consistency](#context-sensitive-local-consistency)
model. However, again for build performance reasons, we want to avoid
package-at-a-time or even library-at-a-time compilation, and would like a
file-at-a-time compilation strategy. The cost of having different views in
different files is substantially reduced given that we have already chosen to
have different views at least in different packages. So we allow the known
information about an entity to vary between source files, even in the same
library.

This implies that we at least need forward declarations for entities that are
declared in an `api` file for a library and defined in an `impl` file. However,
this leaves the question of what happens within a single source file: do
entities need to be manually written in dependency order, or is the
implementation expected to use information that appears after its point of
declaration? Neither option guarantees that an entity will behave consistently
wherever it's used, because we already gave up that guarantee for entities
visible across source files, and we need to cope with the resulting risk of
incoherence anyway.

The simplest option for implementation, and the one most similar to C++, would
then be to accumulate information top-down. This is also the choice that is most
consistent with the cross-source-file behavior: moving a prefix of a source file
into a separate, imported file should generally preserve the validity of the
program. Further, this choice gives us the most opportunity for future language
evolution, as it is the most restrictive option, and hence it is
forward-compatible with the other rules under consideration, as well as
supporting paths for evolution that require a top-down view, such as some
approaches to metaprogramming.

This choice implies that we need forward declarations, both for declarations in
different source files from their definitions and to resolve cycles or cases
where the developer cannot or does not wish to write code in topological order.
Allowing separate declaration and definition may also be valuable to allow the
code author to hide implementation details from readers of the interface, and
present a condensed catalog or table of contents for the library. This is also
an important aspect in providing a language that is familiar to C++ developers.
So we choose to
[allow separate declaration and definition](#allow-separate-declaration-and-definition)
in general.

Given the above decisions, class member functions could not be meaningfully
defined inline under a strict top-down rule, because the class would not yet be
complete. We make an ergonomic affordance of reordering the bodies of such
functions as if they appear after the class definition. This also improves
consistency with C++, which has a similar rule.

## Rationale based on Carbon's goals

-   [Language tools and ecosystem](/docs/project/goals.md#language-tools-and-ecosystem)
    -   See "Toolability" goal.
-   [Software and language evolution](/docs/project/goals.md#software-and-language-evolution)
    -   Ensuring that the program interpretation is consistent with an
        interpretation with full information makes it easier to evolve code, as
        changing the amount of visible information -- for example, by reordering
        code or by adding or removing imports -- cannot change the meaning of a
        program from one valid meaning to another.
    -   Selecting a rule that disallows information from flowing backwards keeps
        open more language evolution paths:
        -   We could allow more information to flow backwards.
        -   We could expose metaprogramming constructs that can inspect the
            state of a partial source file without creating inconsistency in our
            model.
-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)
    -   See "Readability", "Ergonomics", and "Comprehensibility" gaols.
-   [Fast and scalable development](/docs/project/goals.md#fast-and-scalable-development)
    -   See "Efficient and simple compilation" and "Diagnosability" goals.
-   [Interoperability with and migration from existing C++ code](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code)
    -   The chosen rule is mostly the same as the corresponding rule in C++,
        both in accumulating information top-down and in treating inline method
        bodies as a special case. This should keep migration simple and improve
        familiarity for experienced C++ developers.

It should be noted that the decision between the model presented here and the
leading alternative, which was
[context-sensitive local consistency](#context-sensitive-local-consistency), was
very close, and came down to minor details. For each of our goals, neither
alternative had a significant advantage. The similarity to C++ and evolvability
of the alternative in this proposal were ultimately the deciding factors.

## Alternatives considered

Below, various alternatives are presented and rated according to the
[goals](#goals) for this proposal.

Generally we need to pick a rule for allowing or disallowing separate
declaration and definition, and a rule for information flow. These choices are
not independent: some information flow rules require separate declaration and
definition.

### Separate declaration and definition

#### Allow separate declaration and definition

We could allow two or more declarations of each entity, with at most one
declaration providing a definition and the other declarations merely introducing
the existence of the entity and its basic properties:

```
// Forward declaration.
fn F();
// ...
// Another forward declaration of the same function.
fn F();
// One and only definition.
fn F() {}
// Yet another forward declaration.
fn F();
```

This could be done in either a free-for-all fashion as in the above example,
where any number of declarations is permitted, or in a more restricted fashion,
where there can be at most one declaration introducing an entity, and, if that
declaration is not a definition, at most one separate implementation providing
the definition. In this case, the separate definition could contain a syntactic
marker indicating that a prior declaration should exist:

```
// Forward declaration.
fn F();
// ...
// Separate definition, with `impl` marker to indicate that this is just an
// implementation of an entity that was already declared.
impl fn F() {}
```

##### Analysis

_Comprehensibility:_ In general, determining whether two declarations declare
the same entity may not be straightforward, particularly for a redeclaration of
an overloaded function. We have some options, but they all have challenges for
comprehensibility or ergonomics of the rule:

-   We could require token-for-token identical declarations, but this may result
    in ergonomic problems -- developers may wish to leave out information in an
    interface that isn't relevant for consumers of that interface, or spell the
    same type or parameter name differently in an implementation.
-   We could allow the declarations to differ so long as they have the same
    meaning. However, there may be cases, as there are in C++, where it's
    unclear whether two declarations are "sufficiently similar" so as to declare
    the same entity.
-   We could require all overloads of a function to be declared together in a
    group, perhaps with some dedicated syntax, and then match up redeclarations
    based on position in the group rather than signature. Such an approach is
    likely to lead to developers, especially those familiar with the C++ rule,
    being surprised. It will likely also interact poorly with cases where some
    but not all of the overloads are defined in their first declaration and the
    rest are defined separately, or where it is desirable for different
    overloads to be defined in different implementation files.

_Ergonomics:_ Any case where the declaration and definition are separated
results in repetitive code. However, this repetition may serve a purpose to the
extent that it's presenting an API description or acting as a compilation
barrier.

_Readability:_ Readability of code may be substantially improved by separating a
declaration of an API from its definition, and permitting an API user to read,
in a small localized portion of a source file, only the information they care
about, without regard for implementation details.

_Efficient and simple compilation:_ Allowing multiple declarations of an entity
introduces some implementation complexity, as the implementation must look up,
validate, and link together multiple declarations of the same entity. If
redeclarations are permitted across source files -- such as between the
interface and implementation of an API, or between multiple implementation files
-- then this may also require some stable mechanism for identifying the
declaration, such as a name mangling scheme. This may be especially complex in
the case of overloaded function templates, which might differ in arbitrary
Carbon expressions appearing in the function's declaration. This complexity
could be greatly reduced if we require all overloads to be declared together --
at least in the same source file -- as we can then identify them by name and
index.

This approach supports physical separation of implementation from interface,
potentially leading to build time wins through reduced recompilation and through
reducing the amount of information fed into each compilation step.

_Diagnosability:_ Identifying errors where a redeclaration doesn't exactly match
a prior declaration is not completely straightforward. This is especially the
case when function overloads can be added by the same syntax with which a
definition of a prior declaration would be introduced. Even if declarations and
definitions use a different syntax, diagnosing a mismatch between a definition
of an overloaded function and a prior set of declarations requires some amount
of fuzzy matching to infer which one was probably intended.

_Toolability:_ Supporting separate declarations and definitions will present the
same kinds of complexity for non-compiler tools as for compilers. Tools will
need to be able to reason about multiple declarations providing distinct
information about an entity and the effect of that on how the entity can be used
at different source locations.

#### Disallow separate declaration and definition

We could require each entity to be declared in exactly one place, and reject any
cases where the same entity is declared more than once.

##### Analysis

_Comprehensibility:_ This rule is easy to explain and easy to reason about.

_Ergonomics:_ Minimizes the work required by the developer to implement new
functionality and maintain existing functionality. Reduces the chance that a
change in one place will be forgotten in another.

_Readability:_ Each entity has only one declaration, giving a single canonical
location to include documentation, and no question as to where information about
the entity might be found. This will likely improve the navigability of source
code and the ease with which all the information about an entity can be
gathered, and reduce the chance that information in the declaration or
definition will be incomplete or that they will be inconsistent with each other.

However, there would be no ability to present a function or class as source code
without also including implementation details. Such implementation details
cannot be physically separated from the API as presented to clients. The ability
to read and understand an interface will depend more heavily on tools that can
hide implementation details, such as documentation generators and IDEs with
outlining, and implementation techniques such as using `interface`s to present
the API of a class separate from its implementation.

For developers familiar with C++ in particular, the absence of the ability to
separate declaration from definition may create friction, due to the familiarity
of that feature and having their thinking about code layout shaped around it.
Given that Carbon aims to be familiar to those coming from C++, an absence of
forward declarations will work against that goal to some extent, although
empirical evidence suggests that the extent to which this is an issue varies
widely between existing C++ developers.

_Efficient and simple compilation:_ This approach leads to a simpler compilation
strategy. However, in the absence of forward declarations, every compilation
referring to an entity needs to have a build dependency on the full definition
of that entity, meaning there is no physical separation between the APIs of
entities and their definitions. This would impose a potentially-significant
build time cost due to increased recompilations when the definition of an entity
changes.

There are possibilities to reduce this build time cost. For example, instead of
each declaration identifying whether it is public or private to a library, we
could add a third visibility level to say that the declaration but not the
definition is public, and use a stripping tool as part of the build process to
avoid dependencies on non-public definitions from cascading into unnecessary
rebuilds.

This approach is also unlikely to be acceptable from an ergonomic standpoint
unless declarations within a source file are visible throughout at least the
entire library containing that file, although this is strictly an independent
choice. Making definitions from a source file visible throughout its library
would presumably require a library-at-a-time build strategy, which is also
expected to introduce build time costs due to changes to a library requiring
more code in that library to be recompiled. This cost could be reduced by using
a reactive compilation model, recompiling only the portions of a library that
are affected by a code change, at some complexity cost.

_Diagnosability:_ Because each entity can only be declared once, there is no
significant challenge in diagnosing redeclaration issues. However, there is
still some work required to diagnose conflicts between similar or functionally
identical function overloads.

_Toolability:_ Having a unique source location for each entity allows for
somewhat simpler tooling. For example, there is no need to distinguish between
"jump to declaration" and "jump to definition", or to decide which declaration
should be consulted to find documentation comments, parameter names, and so on.

### Information flow

#### Strict top-down

Carbon could accumulate information top-down. We could require that each program
utterance is type-checked and fully validated before any later code is
considered.

In order to support this and still permit cyclic references between entities, we
would need to
[allow separate declaration and definition](#allow-separate-declaration-and-definition).

##### Analysis

_Comprehensibility:_ This rule is simple to explain, and has no special cases.
However, the inability to look at information from later in the source file is
likely to result in gotchas:

```
class Base {
  var n: i32;
}
class Derived extends Base {
  // Returns me.(Base.n), not me.(Derived.n), because the latter has not
  // been declared yet.
  fn Get[me: Self]() -> i32 { return me.n; }
  var n: i32;
}
```

It might be possible to require a diagnostic in such cases, when we find a
declaration that would change the meaning of prior code if it had appeared
earlier, but that would result in implementation complexity, and the fact that
such cases are rejected would still be a surprise.

_Ergonomics:_ The developer is required to topologically sort their source files
in dependency order, manually breaking cycles with forward declarations. Common
refactoring tasks such as reorganizing code may require effort or tooling
assistance in order to preserve a topological order.

Developers could adapt to this ruleset by beginning each source file with a
collection of forward declarations. This would mitigate the need to produce a
topological ordering, except within those forward declarations themselves, and
other declarations required to provide those forward declarations. For example,
a forward declaration of a class member will likely only be possible within a
class definition, and the order in which that class definition is given can be
relevant to the validity of other class definitions. However, our experience
with C++ indicates that this will not be done, and instead an ad-hoc and
undisciplined combination of a topological sort and occasional forward
declarations will be used. Indeed, including these additional forward
declarations would increase the maintenance burden and introduce an additional
opportunity for errors due to mismatches between declaration and definition.

_Readability:_ Developers wishing to understand code have the advantage that
they need only consider prior code, and there is no possibility that a later
source utterance could change the meaning of the code they're reading. However,
it is rare to read code top-down, so the effect of this advantage may be modest.

This advantage leads to a significant disadvantage: the behaviour of an entity
can be different at different places within a source file. For example, a type
can be incomplete in one place and complete in another, or can fail to implement
an interface when inspected early and then found to implement that interface
later. This can lead to very subtle incoherent behavior.

In practice, the topological ordering constraint tends to lead to good locality
of information: helpers for particular functionality are often located near to
the functionality. However, this is not a universal advantage, and the
topological constraint sometimes leads to internal helpers being ordered
immediately before their first use instead of in a more logical position near
correlated functionality.

_Efficient and simple compilation:_ This rule is mostly simple and efficient to
implement, and even allows single-pass processing of source files.

_Diagnosability:_ Because information is provided top-down, diagnostics can also
be provided top-down and in every case the diagnostic will be caused by an error
at the given location or earlier. Fixing errors should require little or no
backtracking by the developer.

However, an implementation that strictly confines its processing to top-down
order and produces diagnostics eagerly cannot deliver diagnostics that react
intelligently to contextual cues that appear after the point of the diagnostic.
This approach diminishes the ability for an implementation to pinpoint the cause
of the error and describe it in a developer-oriented fashion.

_Toolability:_ Limiting information flow to top-down means that tools such as
code completion tools need only consider context prior to the cursor, and they
can be confident that if all the code prior to the cursor is valid that it can
be type-checked and suitable completions offered.

However, in the case where the user wants to refer to a later-declared entity,
such tools would not be able to use this strategy. They would need to parse as
if there were not a top-down rule in order to find such later-declared entities,
and would likely additionally need the ability to add forward declarations or to
reorder declarations in order to satisfy the ordering requirement.

#### Strict global consistency

Carbon could follow an approach of requiring the behavior of every entity to be
globally consistent. In this approach, the behavior of every entity would be as
if the entire program could be consulted to determine facts about that entity.

In practice, to make this work, we would need to limit where those facts can be
declared. For example, we limit implementations of interfaces to appear only in
source files that must already be in use wherever the question "does this type
implement that interface?" can be asked.

In addition, we need to reject at least the case where some property of the
program recursively depends upon itself:

```
struct X {
  var a: Array(sizeof(X), i8);
}
```

In order to give globally consistent semantics to, for example, a package name,
we would likely need to process all source files comprising a package at the
same time. This is likely to encourage packages to be small, whereas we have
designed package support on the assumption that the scope of a package matches
that of a complete source code repository.

This alternative can be considered either with or without the ability to
separate declarations from definitions. If we permit separate declaration and
definition, we would likely require declarations of the same entity to appear in
the same library; however, that limitation is desirable regardless of which
strategy we choose.

##### Block-scope exception

Applying this rule to local name lookup in block scope does result in some
surprises. For example, C# uses this approach, and combined with its
disallowance of shadowing of local variables, this
[confuses some developers](https://stackoverflow.com/questions/1196941/variable-scope-confusion-in-c-sharp).
As a variant of this alternative, it would be reasonable and in line with likely
programmer expectations to not apply this rule to names in block scope. We call
this the _block-scope exception_.

For example:

```
var m: i32 = 1;
fn F(b: bool) -> i32 {
  if (b) {
    var n: i32 = 2;
    // With the block-scope exception:
    // * `n` is unambiguously the variable declared on the line above, and
    // * `m` is unambiguously the global.
    //
    // Without the block-scope exception:
    // * depending on the rules we choose for shadowing, `n` might name the
    //   variable above, might be ambiguous, or might be invalid because it
    //   shadows the variable `n` declared below, and
    // * `m` would name the local variable declared below.
    return n + m;
  }
  var n: i32 = 3;
  var m: i32 = 4;
  return n * m;
}
```

##### Analysis

_Comprehensibility:_ This rule is simple to explain, and has no special cases,
other than perhaps the block scope variant. The disallowance of semantic cycles
is likely to be unsurprising as it is a logical necessity in any rule.

_Ergonomics:_ The developer can organize or arrange their code in any way they
desire. There is never a requirement to forward-declare or repeat an interface
declaration. Refactoring and code reorganization do not require any non-obvious
changes, because the same code means the same thing regardless of how it is
located relative to other code.

_Readability:_ Reasoning about code is simple in this model, as such reasoning
is largely not context-sensitivity. Instead of questioning "what does this do
here?" we can instead consider "what does this do?". Some context sensitivity
may remain, for example due to access and name bindings differing in different
contexts.

However, to developers accustomed to a top-down semantic model, the ability to
defer giving key information about an entity -- or even declaring it at all --
until long after it is first used may hinder readability in some circumstances,
particularly when reading code top-down.

_Efficient and simple compilation:_ This model forces the compilation process to
operate in multiple stages rather than as a single pass.

Some form of cycle detection is necessary if cycles are possible. However, such
a mechanism is likely to be necessary for template instantiation too, so this is
likely not a novel requirement for Carbon.

Forcing all files within a package to be compiled together in order to provide
consistent semantics for the package name may place an undesirable scalability
barrier on the build system. This will also tend to push up build latency,
especially for incremental builds, by decreasing the granule size of
compilation. It may increase the scope of recompilations due to additional
physical dependencies unless we implement a mechanism to detect whether a
library has changed in an "important" way.

We do not have an existence proof of an efficient build strategy for a
Carbon-like language that follows this model. For example, Rust has known build
scalability issues, as does Java for large projects unless a tool like
[ijar](https://github.com/bazelbuild/bazel/tree/master/third_party/ijar) is used
to automatically create something resembling a C++ header file from a source
file. There is a significant risk that such an automated tool would be very hard
to create for Carbon: because the definitions of classes and functions can be
exposed in various ways through compile-time function evaluation, templates, and
metaprogramming, it seems challenging to put any useful bound on which entities
can be stripped without restricting the capabilities of client code.

_Diagnosability:_ The implementation is likely to have more contextual
information when providing diagnostics, improving their quality. However, the
diagnostics may appear in a confusing order: if an early declaration needs
information from a later declaration in order to type-check, diagnostics
associated with that later declaration may be produced first, or may be
interleaved with diagnostics for the earlier declaration, leading to the
programmer potentially revisiting the same code multiple times during a
compile-edit cycle.

_Toolability:_ This model requires tools to consider the whole file as context,
because code may refer to entities that are only introduced later. For an IDE
scenario, where the cursor represents a location where an arbitrary chunk of
code may be missing, this presents a challenge of determining how to
resynchronize the input in order to determine how to interpret the portion of
the source file following the cursor.

Sophisticated tooling for a top-down model may wish to inspect the trailing
portion of the file anyway, in order to provide a better developer experience,
but this complexity would be forced upon tools with this model.

#### Top-down with minimally deferred type checking

We could follow a top-down approach generally, but defer type-checking some or
all top-level entities until we reach the end of that entity. For example, we
would check an entire class as a single unit, following the same principles as
in the globally-consistent rule, but using only information provided prior to
the end of the class definition. This would allow class members to use other
members that have not yet been declared, while not permitting a function
definition preceding the class definition to use such members.

Following C++'s lead, we would apply this to at least classes and interfaces,
but perhaps to nothing else. We may still want to apply the top-down rule to
function definitions appearing within a class or interface, as the C# behavior
that the scope of a local variable extends backwards before its declaration may
be surprising.

##### Analysis

This approach generally has the properties as
[strict top-down](#strict-top-down), except as follows:

_Comprehensibility:_ Slightly reduced due to additional special-casing of
top-level declarations. Gotchas would be rarer, but may not be fully eliminated.
For example:

```
import X;
// G from package X, not G from class X below.
fn F() { X.G(); }
class X {
  fn G() {}
}
class Y {
  // G from class X below, not G from package X.
  fn F() { X.G(); }
  class X {
    fn G() {}
  }
}
```

_Ergonomics:_ Improved within classes, as members can now be named everywhere
within the class.

_Readability:_ The ability to read and understand code may be somewhat harmed,
as the rules for behavior across top-level declarations aren't the same as the
rules for behavior within a top-level declaration. It may be especially jarring
that code that is valid within a class definition is not valid at the top level:

```
class A {
  // OK
  var b: B;
  class B {}
}

// Error, I have not heard of B.
var b: B;
class B {}
```

However, the behavior of an entity no longer differs at different points within
the same declaration of that entity, so the ability to reason about the behavior
of an entity is somewhat improved compared to the strict top-down approach. For
entities without a separate declaration and definition, the behavior of those
entities is the same everywhere.

_Efficient and simple compilation:_ This has the same complications as the
strict global consistency approach, except that it does not apply to contexts
that span multiple files, so it doesn't have the build time disadvantages of
requiring all source files in a library to be built together.

_Diagnosability:_ Diagnostics should be encountered by the implementation in the
top-down order of the top-level declaration containing the error, but may in
principle appear in any order within that top-level declaration, if there are
forward references that require later code to be analyzed before earlier code
is.

Diagnostics can be caused by errors appearing both before and after the location
reported, but not arbitrarily far after: later top-level declarations cannot
affect an earlier diagnostic.

_Toolability:_ Tools that want to correctly model Carbon semantics would be
required to deal with incomplete source code in the vicinity of the cursor. This
is probably the hardest kind of incomplete source code to handle, as the region
of "damaged" code is most likely to be relevant context. In practice, this is
likely to present the same difficulties as tooling in the
[strict global consistency](#strict-global-consistency) model.

#### Top-down with deferred method bodies

**This is the proposed approach.**

We could follow a top-down approach generally, but defer all processing of
bodies of class member functions until we have finished processing the class, at
which point we would process those member function bodies in the order they
appeared within the class. For example, we would reinterpret:

```carbon
class A {
  fn F[me: Self]() -> i32 { return me.n; }
  fn Make() -> A { return {.n = 5}; }
  var n: i32;
}
```

exactly as if it were written with the member function bodies out of line:

```carbon
class A {
  fn F[me: Self]() -> i32;
  fn Make() -> A;
  var n: i32;
}
fn A.F[me: Self]() -> i32 { return me.n; }
fn A.Make() -> A { return {.n = 5}; }
```

The [strict top-down](#strict-top-down) would be applied to this rewritten form
of the program.

As in C++, member functions of nested classes would be deferred until the
outermost class is complete. This deferral would apply only to the bodies of
member functions, unlike in C++ where it also applies to default arguments,
default member initializers, and exception specifications, and unlike
[top-down with minimally deferred type checking](#top-down-with-minimally-deferred-type-checking),
where it also applies to member function signatures and the declarations and
definitions of non-function members.

##### Analysis

This approach generally has the properties as
[strict top-down](#strict-top-down), except as follows:

_Comprehensibility:_ Slightly reduced due to additional special-casing of class
member functions, but these rules are well-aligned with the rules from C++,
which do not seem to suffer from major comprehensibility problems in this area.

_Ergonomics:_ Improved within classes, as members can now be named within member
function bodies.

_Readability:_ The ability to read and understand code may be somewhat harmed,
as the rules for behavior within a member function body are different from the
rules everywhere else in the language. However, the very similar rule in C++ is
not known to cause readability problems, so no major concerns are anticipated.

_Efficient and simple compilation:_ The complexity of this approach is not
substantially greater than the simple top-down approach. Processing of member
function bodies must be deferred, but this can be done either by storing and
replaying the tokens, or by forming a parse tree but deferring the semantic
analysis and type-checking until the end of the class.

_Diagnosability:_ Diagnostics for errors in member functions may appear after
diagnostics for later code in the same class.

Developers may be confused by diagnostics claiming that a class member function
body is not available yet and referring to a point in the program text that
lexically follows the member function definition. It is likely possible to
special-case such diagnostics to explain the nature of the problem.

_Toolability:_ Tools that want to parse incomplete Carbon code would need to
cope with member function bodies containing errors. For the most part, skipping
a brace-balanced function body should be straightforward, but tools will also
need to consider whether they attempt to detect and recover from mismatched
braces within member functions.

#### Context-sensitive local consistency

We could use different behaviors in different contexts, as follows:

-   For contexts that are fundamentally ordered, such as function bodies, a
    top-down rule is used.
-   For contexts that are defined across multiple source files, such as packages
    and namespaces, we guarantee consistent behavior within each source file,
    but the behavior may be inconsistent across source files: different source
    files may see different sets of names within a package or namespace,
    depending on what they have imported.
-   For contexts that are defined within a single source file, such as a class
    or an interface, we guarantee globally consistent behavior.

Compared to the [strict global consistency](#strict-global-consistency) model,
this would not guarantee that the contents of a package or namespace are the
same everywhere: you only see the names that you declare or import into a
package or namespace. Also, a top-down rule is used within function bodies, as
that likely better matches programmer expectations.

Compared to the
[top-down with minimally deferred type checking](#top-down-with-minimally-deferred-type-checking)
and
[top-down with deferred method bodies](#top-down-with-deferred-method-bodies)
models, this would not require forward declarations for namespace and package
members that are declared or defined later in the same source file.

##### Analysis

_Comprehensibility:_ This model is probably a little easier to understand than
the minimally-deferred type-checking model, because there is no difference
between top-level declarations and nested declarations.

Because the behavior of every entity is consistent within a file, and every name
lookup is consistent within a scope -- other than the top-down behavior within
function scopes -- there are fewer opportunities for surprises than in the
top-down approaches, and nearly as few as in the strict global consistency
model.

_Ergonomics:_ As with the strict global consistency model, code can be organized
as the developer desires, and refactoring doesn't have any surprises due to
disrupting a topological order.

_Readability:_ As with the strict global consistency model, readability is
improved by not being context-sensitive. However, it is file-sensitive, and a
different set of imports may lead to different behavior.

Similarly, as with the strict global consistency model, developers accustomed to
the top-down model may find it harder to understand code in which a name can be
referenced before it is declared.

_Efficient and simple compilation:_ This model has the same general costs as the
top-down with minimal deferred type-checking model. The implementation needs to
be able to separate parsing from type-checking, and type-check lazily in order
to handle forward references. However, more state needs to be accumulated before
type-checking can begin.

This model supports separate compilation of source files, as no attempt is made
to ensure consistency across source files, only within a source file.

_Diagnosability:_ The compilation process will be split up into multiple phases,
and it is likely that diagnostics from one phase will precede those from
another. For example, a parsing error for a later construct may precede a type
error for an earlier one.

If an earlier phase fails, it may not be feasible to diagnose later phases. For
example, if parsing encounters an unrecoverable failure, type errors for the
successfully-parsed portion cannot be generated, because they may depend on
constructs appearing later in the same source file. However, this may also mean
that better diagnostics are produced for cases such as a missing `}`, where the
syntactic error will precede or prevent diagnostics for semantic errors caused
by the misinterpretation of the program resulting from that missing `}`.

Within a particular phase of processing, diagnostics will generally be produced
in a topological order, and the processing can be arranged such that they are
produced in top-down order except where a forward reference requires that a
later declaration is checked earlier.

_Toolability:_ Largely the same as the strict global consistency model, except
that tools only need to use the current source file as context. Also largely the
same as the top-down with minimally deferred type checking model.
