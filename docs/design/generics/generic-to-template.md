# Passing generic arguments to template parameters

<!--
Part of the Carbon Language, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Table of contents

<!-- toc -->

-   [Problem statement {#problem-statement}](#problem-statement-%23problem-statement)
-   [Possible approaches {#possible-approaches}](#possible-approaches-%23possible-approaches)
    -   [Adopting none of the below options {#adopting-none-of-the-below-options}](#adopting-none-of-the-below-options-%23adopting-none-of-the-below-options)
    -   [Option #1: instantiate templates immediately on an archetype {#option-#1-instantiate-templates-immediately-on-an-archetype}](#option-%231-instantiate-templates-immediately-on-an-archetype-%23option-%231-instantiate-templates-immediately-on-an-archetype)
        -   [Recommendation: no to option #1 {#recommendation-no-to-option-#1}](#recommendation-no-to-option-%231-%23recommendation-no-to-option-%231)
        -   [Simulating option #1 {#simulating-option-#1}](#simulating-option-%231-%23simulating-option-%231)
    -   [Option #2: provide a generic interface for the template {#option-#2-provide-a-generic-interface-for-the-template}](#option-%232-provide-a-generic-interface-for-the-template-%23option-%232-provide-a-generic-interface-for-the-template)
        -   [Recommendation: support templated impl of interfaces {#recommendation-support-templated-impl-of-interfaces}](#recommendation-support-templated-impl-of-interfaces-%23recommendation-support-templated-impl-of-interfaces)
    -   [Option #3: generically typecheck template code {#option-#3-generically-typecheck-template-code}](#option-%233-generically-typecheck-template-code-%23option-%233-generically-typecheck-template-code)
        -   [Allowed {#allowed}](#allowed-%23allowed)
        -   [Trickier cases {#trickier-cases}](#trickier-cases-%23trickier-cases)
            -   [Option A: reject comparisons we can't answer {#option-a-reject-comparisons-we-can't-answer}](#option-a-reject-comparisons-we-cant-answer-%23option-a-reject-comparisons-we-cant-answer)
            -   [Option B: typecheck considering all code paths {#option-b-typecheck-considering-all-code-paths}](#option-b-typecheck-considering-all-code-paths-%23option-b-typecheck-considering-all-code-paths)
                -   [Applications {#applications}](#applications-%23applications)
        -   [Recommendation: defer option #3 {#recommendation-defer-option-#3}](#recommendation-defer-option-%233-%23recommendation-defer-option-%233)
-   [Conclusion {#conclusion}](#conclusion-%23conclusion)
-   [Broken links footnote](#broken-links-footnote)

<!-- tocstop -->

## Problem statement {#problem-statement}

Most of the time, templates and generics co-exist simply and without complex
interactions. They have their own rules and operate accordingly. There is a
narrow case where we need to define the precise interaction semantics and it is
difficult to do so:

**Within a parameterized function or type with a generic parameter, how can we
use a template parameterized by types or values dependent on that generic
parameter?**

Consider the following Carbon code (please ignore the specific syntax used) as a
representative example:

```
fn TemplateFunction[Type:$$ TemplateParameter](Ptr(TemplateParameter): pointer) {
  ...
}

// `SomeInterface` here constrains `GenericParameter` to be some type implementing
// `SomeInterface`.
fn GenericFunction[SomeInterface:$ GenericParameter]
    (Ptr(GenericParameter): pointer) {
  // What type is `TemplateParameter` is set to when we instantiate and
  // type-check `TemplateFunction` here?
  TemplateFunction(pointer);
}
```

Here the `pointer` has a type that has some generic component, it could be that
the type is a generic value, is parameterized by a value only known generically,
or as in the above example it could be a pointer-to-generic type. When `pointer`
is passed to `TemplateFunction`, we need to assign some type to
`TemplateParameter`.

A generic function is type-checked exactly once (when defined), regardless of
the specific types that will be passed in at the call sites. Therefore, the call
`TemplateFunction(pointer)` must be type checked exactly once, and must work for
all types that satisfy the constraints in the function signature.

For templates, though, call sites influence how a template is instantiated and
type-checked.

Therefore, we have a contradiction: when we type-check `GenericFunction`, we
don’t know the specific type that it will be called with, and therefore, we
can’t instantiate and type-check `TemplateFunction` using that type.

## Possible approaches {#possible-approaches}

Note that these options are somewhat independent. We could adopt none of these,
or a combination of multiple options.

### Adopting none of the below options {#adopting-none-of-the-below-options}

It is valuable to first consider what happens if we simply forbid the above from
happening, since this is the simplest approach.

In the case of non-type values, I believe that it is the best we can do. If the
parameter could have been a generic instead of a template, it almost certainly
would be, unless the code is in C++. Attempts to use the template with a generic
value will invariably trip over some use of the value which we can't compile
without violating the generic abstraction, requiring knowing the value for type
checking.

For type parameters, consider a C++ code base using templates being converted to
Carbon, but the result of the conversion could be written using Carbon generics.
As long as the conversion is performed "bottom up", that is the leaves of the
call tree before the callers, you can avoid setting a template parameter to
something that depends on a generic.

It does result in some restrictions which may be problematic:

-   Cannot use templates even when the desired behavior would easily be achieved
    without using any part of the actual type parameters to the generic. The
    classic example here is using templates with pointers to generic type
    parameters that would work equally well with opaque pointers or `void`
    pointers. The specific value of the generic type parameter in this case is
    immaterial, but still blocks the usage because it is technically possible
    for the template to use it in some way -- even if it happens not to in most
    cases.
-   Cannot use C++ vocabulary types which are templates with generic Carbon
    types (likely most containers and algorithms). This seems likely to be a
    very significant barrier to interoperability given the prevalence of APIs
    accepting `std::optional`, `std::unique_ptr`, and the like.

As a consequence, we also consider more complex options where the usage is
allowed.

### Option #1: instantiate templates immediately on an archetype {#option-#1-instantiate-templates-immediately-on-an-archetype}

Here, we use the term _archetype_ to mean a type synthesized to embody the known
properties of the generic type parameter. To implement this option, our
archetype would also implement these properties using the exact same mechanisms
as the generic code.

In our example above, we might name the archetype `GenericParameter`, to match
the generic type value. Much like a facet type (and in fact facet types were
inspired by this idea), that type would have exactly the functions defined by
its constraints. In our example, it would have the methods defined in
`SomeInterface`. The implementation of those methods would not be known, but the
signature information in `SomeInterface` would (hopefully) be enough to
typecheck the template instantiation of `TemplateFunction`.

This option attempts to most fully address the first of the two use cases, as
when the specific generic type parameter is immaterial to the template this will
work exceptionally well. The archetype will essentially be a placeholder that
allows forming a stable derived type (such as a pointer) for use by the template
code.

Pros:

-   Directly supports an anticipated common case where the type is essentially
    immaterial (and could equally be opaque) to the template.
-   Easily implementable without added complexity to the type-checking of
    generics.
-   Does not require any deferred work when compiling users of the generic, all
    type checking and code generation can be done in advance, as with all other
    generic code.
-   Does not require writing a modular description of templates’ interfaces.

Cons:

-   Generics can fail to use template specializations intended for the actual
    type used with the generic. This may cause surprising semantic issues where
    the primary template is not valid for the specific type (for which a
    specialization is defined), memory layout issues where the specialization of
    the template has a different memory layout, or it may cause performance
    cliffs where an important optimization is not employed in a data structure
    such as `std::optional` or `std::variant` reusing bits.
-   May be a difficult model to teach as it affirms a distinction between the
    type within the generic and the type argument to the generic.
-   Does not enable use of C++ vocabulary types which are templates with generic
    Carbon types (the second use case above). This doesn't work because within
    the generic context, the C++ template would use some archetype and not the
    desired (and required for interop) actual type.

#### Recommendation: no to option #1 {#recommendation-no-to-option-#1}

I recommend against option #1. Since it hides the type identity from the
template, it can cause a situation where a refactoring we want to be safe
instead can silently change behavior or performance. The situation is this: you
start with two templated functions, `F` and `G`, where `F` calls `G`. You then
change `F` to take a generic parameter instead of its template parameter. You
would hope that if the result compiles, then nothing has changed, but this is
not the case. `G` may depend on the identity of its type parameter:

-   It could have an explicit type test to detect and handle types that require
    or benefit from special treatment.
-   It could call an overloaded function, and the overload that gets selected
    depends on the specifics of the type not exposed by the archetype.
-   It could test the type for properties or expanded capabilities which allow a
    more efficient algorithm to be used. For example, it could test whether a
    container's iterators support random access, and if so, select an algorithm
    with a lower complexity class.

Furthermore, if you do want this behavior, it should in principle be something
users could do on their own, without specific support from the Carbon language
itself. If we see users finding important use cases best solved by this option,
we can revisit -- though likely with an explicit syntax that opts into this
behavior rather than what you get by default.

#### Simulating option #1 {#simulating-option-#1}

Option #1 works by separating the generic parameter from the template argument
by interposing the archetype between. The archetype can be a single real type
_not dependent on generic type parameters_. We can do this manually, for
example:

```
// Assume `SomeInterface` has functions `F` and `G`.
struct Archetype {
  var Ptr(Void): p;
  var fn(Ptr(Void))->Int: f;
  var fn(Ptr(Void))->Bool: g;
  fn F(Ptr(Self): this)->Int {
    return this->f(this->p);
  }
  fn G(Ptr(Self): this)->Bool {
    return this->g(this->p);
  }
}

fn GenericFunction[SomeInterface:$ GenericParameter]
    (Ptr(GenericParameter): pointer) {
  // Need some casts from `Ptr(GenericParameter)` to `Ptr(Void)` here.
  var Archetype: archetype =
    (.p = pointer, .f = GenericParameter.F, .g = GenericParameter.G);
  TemplateFunction(&archetype);
}
```

This would likely introduce some performance penalty (assuming it wouldn't all
be optimized away), in addition to being a lot of cumbersome additional code to
write. On the plus side, this explicit formulation makes the semantics very
clear -- `Archetype` is clearly a distinct type from the generic type and its
capabilities visibly don't vary with `GenericParameter`.

If we wanted to support option #1 via an explicit syntax, I feel we should adopt
an approach that creates a specific named type compatible with the passed-in
generic type to act as the archetype. We might reuse
[the adapt concept](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/designs/generics-interface-type-types.md#adapting-types),
something along the lines of:

```
fn GenericFunction[SomeInterface:$ GenericParameter]
    (Ptr(GenericParameter): pointer) {
  adaptor Archetype for GenericParameter;
  TemplateFunction(&archetype as Ptr(Archetype));
}
```

### Option #2: provide a generic interface for the template {#option-#2-provide-a-generic-interface-for-the-template}

To enable using C++ vocabulary templates in Carbon, we have to instantiate the
C++ template with the underlying type argument to the generic. This is
particularly difficult as we do not want to undermine the modular definition
checking of generics. Consider a template whose interface is defined in terms of
the template argument like the following example:

```
extern "C++" {  // C++ code
template <typename T>
class my_vector_iterator {
  // ...
  using value_type = T;
  auto operator*() -> value_type { ... }
};

template <typename T>
class vector {
  // ...
  using iterator = my_vector_iterator<T>;
  auto begin() -> iterator { ... }
};
}

// Carbon code
fn CarbonGeneric[Type:$ T](T*: thing) {
  // ...
  var Cpp.vector(T): my_vector;
  var auto: first_element = *my_vector.begin();
}
```

Here, we cannot type check the expression `*my_vector.begin()` and thus cannot
determine the type of `first_element` without first instantiating `vector` and
then in turn instantiating `my_vector_iterator` to pass the type parameter all
the way through the template.

Generics can use other generics and maintain their modular type checking because
other generics provide a modular definition of their interface (in other words,
interface that can be obtained without instantiation). Therefore, we could add a
mechanism to templates to provide a modular description of their interface to
enable modular type-checking of users (regardless of whether they are generics,
templates, or concrete code).

Instantiating the template would still be non-modular and require substitution.
If the result of the substitution does not satisfy the declared interface, the
program is ill-formed.

The next challenge is to trigger the actual instantiation of the necessary
template. This can be done by including the generic description of the template
interface in the required interfaces of the generic (as would typically be
needed for any dependent generic). The root usage of generics (the transition
from concrete code into generics, or from a template into a generic) with
concrete types will then intrinsically have the transitive closure of templates
required to be instantiated for its types, just as each generic reached will
have an abstract interface definition to type check against. The root usage can
even validate these generic interfaces as correctly matching the instantiated
template interface.

Pros:

-   Fully models the actual desired construct: a template instantiated directly
    on the actual type passed as a generic type parameter.
-   Specializations of templates are used according to actual types passed as
    the type parameters.
-   Does not force generics to be monomorphized or otherwise restricted.
-   Supports complete definition checking of generic code.
-   Definition checking of generic code is modular.
-   Introduces interface checking for templates (which can be a desirable
    feature regardless).
-   Intrinsically models the necessary dependency information between generics.
-   Can address complex interoperability concerns with C++ interfaces accepting
    templated types.

Cons:

-   Requires writing modular descriptions of templates’ interfaces, potentially
    duplicating a large portion of an API already described in the template.
-   Requires instantiating transitive closure of templates at the root of any
    used generic.

#### Recommendation: support templated impl of interfaces {#recommendation-support-templated-impl-of-interfaces}

This feature has already been recommended
[in the current generics proposal](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/designs/generics-interface-type-types.md#templated-impls-for-generic-interfaces),
specifically as
[a bridge for C++ templates](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/designs/generics-interface-type-types.md#bridge-for-c-templates).
For example, we might define a Carbon interface to wrap the C++
`std::optional<T>` template type, like so:

```
interface GenericCppOptional(Type:$ T) {
  // Here, `T` is generic and not a template.
  fn Get(Self*: this) -> T;
}
```

and then implement it for `std::optional<T>`:

```
impl[Type:$$ T] GenericCppOptional(T) for extern "C++" { std::optional<T> } {
  // Here `T` is a template parameter, and `Self` is `std::optional<T>`.
  fn Get(Self*: this) -> T { this->get(); }
}
```

This demonstrates the first "con" -- this option doesn't let us directly use
`std::optional<T>` from a generic without first creating this wrapper interface
and impl. We could potentially make this less onerous with the help of
additional language features, but we don't recommend any at this time.

The second "con" is any generic function using this wrapper will need to mention
it in its signature. For example:

```
fn UsesOptional[Type:$ T, GenericCppOptional(T):$ OptT](...) { ... };
```

This is true even if the use is some function called by `UsesOptional`.

### Option #3: generically typecheck template code {#option-#3-generically-typecheck-template-code}

The big problem with
[option #1](#option-#1-instantiate-templates-immediately-on-an-archetype) is
that it can silently do something undesirable. A possible way to repair this
problem is to say that calling a template with a generic may fail, but if it
succeeds, it works as you expect. Of course calling a template is generally
something that can fail, since templates are only typechecked as the result of
instantiation triggered by a call, so it shouldn't be a big surprise that they
can fail in this case too.

The question is now, what works and what fails? In principle what we need to
assure is that we can do enough checking when the generic caller is defined that
no new failures will occur when we substitute in actual type arguments from the
call during code generation.

#### Allowed {#allowed}

Some questions can be resolved definitively using just the information known
from the type information available in the generic caller. consider a templated
`Vector` type with a special case for `Bool`. For _some_ generic code, this
should be a non-issue:

```
// This function parameterized by a generic should be fine:
fn MakeShortVectorOfPointers[Type:$ T](T*: x) -> Vector(T*) {
  var Vector(T*): v;
  v.PushBack(x);
  return v;
}

var Bool: b = True;
var Vector(Bool*): v = MakeShortVector(&b);
```

As long as we can say that `T* != Bool` independent of `T`, the `Vector(Bool)`
specialization is irrelevant. (There is some concern about how reliably we can
answer that question for arbitrary type expressions involving generics.) In this
case, the `Vector` template is actually generic (in the sense of not doing
anything invalid for generics) for the subset of types that are pointers. Note
that this relies on
[closed function overloading (TODO)](#broken-links-footnote)<!-- T:Carbon closed function overloading proposal -->,
to be able to have a complete list of all the overloads being considered when
resolving a function call.

If we adopt option #3, we should definitely allow straightforward cases like
this.

#### Trickier cases {#trickier-cases}

The remaining question is whether we should allow code to probe properties about
the generic type beyond what we can resolve at type checking time. For example,
if `Vector(T)` is overloaded to have a separate implementation for
`Vector(Bool)`, then this code relies on determining whether the generic type
value `T` is `Bool`:

```
// Seems like it maybe should be fine:
fn MakeShortVector[Type:$ T](T: x) -> Vector(T) {
  var Vector(T): v;
  v.PushBack(x);
  return v;
}

// Trouble?
var Vector(Bool): v = MakeShortVector(True);
```

What we don't want is the option #1 behavior of instantiating `Vector(T)` using
some type `T != Bool`. Instead we have two choices: option A rejects this code,
and option B accepts it.

##### Option A: reject comparisons we can't answer {#option-a-reject-comparisons-we-can't-answer}

The first option is to reject this code. Since `Vector(T)` is overloaded, it
introduces a comparison between `T` and `Bool`. Since `T` is unknown, we can't
answer the question, and so we report an error during type checking time. This
is a conservative option, but potentially rejects more code than necessary.

##### Option B: typecheck considering all code paths {#option-b-typecheck-considering-all-code-paths}

The second option is to consider both cases of how the type comparison could go.
Since we can't answer the question of whether `T == Bool` when type checking
`MakeShortVector`'s definition, we instead consider it to be a runtime type
test. We then need to consider both possibilities and make sure the function
type checks either way. Then, during code generation, the answer to the question
becomes a constant and the code can be optimized to just the relevant case. This
will accept more code, but in some sense allows "piercing the veil": using more
type information than is declared in the signature.

Under this proposal, `MakeShortVector(T)` above is legal and works for all `T`,
`Bool` or not, as long as the differences between those two cases don't cause
any type errors in `MakeShortVector` or in the template expansion of `Vector(T)`
methods invoked from `MakeShortVector`. However, these functions would both
trigger compile errors:

```diff
  fn FlipVector[Type:$ T](T: x) -> Vector(T) {
    var Vector(T): v;
    v.PushBack(x);
    // Error: method only present in Vector(Bool)
-   v.Flip();
    return v;
  }

  fn AddressOfElementOfVector[Type:$ T](T: x) -> Vector(T) {
    var Vector(T): v;
    v.PushBack(x);
    // Error: illegal for Vector(Bool)
-   var Ptr(T): p = &v[0];
    DoSomethingWithPointer(p);
    return v;
  }
```

One concern with this approach is if we want to use a dynamic strategy for
compiling generics that only generates one copy of the function. In this case,
the dynamic type test will be left in the code at runtime, with the associated
runtime costs.

If we allow template code to perform type tests on generic type values, we very
likely want to allow those tests directly without having to call a separate
templated function. This would allow generic code to include optimizations that
could be very important to performance.

###### Applications {#applications}

In C++, `std::vector<T>::resize()` can use a more efficient algorithm if `T` has
a `noexcept` move constructor. We could allow this optimization from generic
code since it does not affect the signature of `resize()`, and therefore type
checking.

We also expect some algorithms to have a special case putting values on the
stack instead of the heap when they are below some size threshold. Others may
have more efficient code paths for types with specific capabilities such as
random access iterators, or an efficient swap or move operation.

These all seem like cases that would be allowed and enabled by this option.
Perhaps slightly less clear is whether this would allow `auto` to be defined as
`(Type:$$ _)`, even when it is matching a generic type value.

#### Recommendation: defer option #3 {#recommendation-defer-option-#3}

Option #3 definitely introduces complexity and implementation work into the
compiler, which we aren't sure we need. Once we have greater experience
implementing with generics, we will be better able to say whether we need the
capabilities of option #3 and what variation on it.

## Conclusion {#conclusion}

After discussing these in depth, we came to a clear realization: there are many
use cases here and we may well need to support multiple options at the same
time. With all three options there is uncertainty about how much language
support we want to provide, particularly where we would be making things easy
instead of just possible. The usage patterns seem difficult to predict in this
area: currently no widely used existing languages are trying to interface these
technical patterns (generics and templates) or interface with C++ templates in
this manner. It seems reasonable to delay until we get more experience to see
where the _[desire paths](https://en.wikipedia.org/wiki/Desire_path)_ are before
paving them.

## Broken links footnote

Some links in this document aren't yet available, and so have been directed here
until we can do the work to make them available.

We thank you for your patience.
