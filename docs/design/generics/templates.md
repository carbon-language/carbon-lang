# Carbon templates

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

### Non-type templates

For `CreateArray` function, we could change the parameter to be a template
parameter by replacing "`UInt:$ N`" with "`UInt:$$ N`", but there would not be a
difference that you would observe. However, a generic function would not be able
to type check if the parameter was changed to an array size that could be
negative, as in "`Int:$ N`". For a template, this would only be a problem when a
negative value was passed in.

```
// Compile error: array size can't be negative.
fn CreateArray_Error(Int:$ N, Int: value) -> FixedArray(Int, N) { ... }

// No compile error.
fn CreateArray_Template(Int:$$ N, Int: value) -> FixedArray(Int, N) { ... }

// No compile error.
CreateArray_Template(3, 7);
CreateArray_Template(6, 12);

// Compile error: array size can't be negative.
CreateArray_Template(-2, 12);
```

Similarly, we could call an overloaded function from a templated version of
`PrintXs`:

```
fn NumXs(1) -> Char {
  return 'X';
}
fn NumXs(2) -> String {
  return "XX";
}
fn PrintXs_Template(Int:$$ n) {
  Print(NumXs(n));
}

PrintXs_Template(1);  // Prints: X (using Print(Char))
PrintXs_Template(2);  // Prints: XX (using Print(String))
var Int: m = 3;
PrintXs_Template(m);  // Compile error: value for template parameter `n`
                      // unknown at compile time.
PrintXs_Template(3);  // Compile error: NumXs(3) undefined.
```

Since type checking is delayed until `n` is known, we don't need the return type
of `NumXs` to be consistent across different values of `n`.

**Comparison with other languages:** These are called
[non-type template parameters in C++](https://en.cppreference.com/w/cpp/language/template_parameters#Non-type_template_parameter).

#### Difference between templates and generics

For generics, the body of the function is fully checked when it is defined; it
is an error to perform an operation the compiler can't verify. For templates,
name lookup and type checking may only be able to be resolved using information
from the call site.

#### Substitution failure is an error

Note: This is a difference from C++, and the rules may be different when calling
C++ code from Carbon.

In Carbon, when you call a function, the corresponding implementation (function
body) is resolved using name lookup and overload resolution rules, which use
information in the function signature but not the function body. The function
signature can include arbitrary code to determine if a function is applicable,
but once it is selected it won't ever switch to another function body. This
means that if substituting in templated arguments into a function triggers an
error, that error will be reported to the user instead of trying another
function body (say for a different overload of the same name that matches but
isn't preferred, perhaps because it is less specific).

**Open question:** Determine an alternative mechanism for determining when a
templated function is applicable, to replace the use cases of
[SFINAE in C++](https://en.wikipedia.org/wiki/Substitution_failure_is_not_an_error).

### Generic type parameters versus templated type parameters

Recall, from
[the "Difference between templates and generics" section above](#difference-between-templates-and-generics),
that we fully check functions with generic parameters at the time they are
defined, while functions with template parameters can use information from the
caller.

If you have a value of a generic type, you need to provide constraints on that
type that define what you can do with values of that type. However when using a
templated type, you can perform any operation on values of that type, and what
happens will be resolved once that type is known. This may be an error if that
type doesn't support that operation, but that will be reported at the call site
not the function body; other call sites that call the same function with
different types may be fine.

So while you can define constraints for template type parameters, they are
needed for generic type parameters. In fact, type constraints are the main thing
we need to add to support generic type parameters, beyond what is described in
[the "non-type generics" section above](#non-type-generics).

### Calling templated code

See ["Passing generic arguments to template parameter"](generic-to-template.md).

### `auto`

**Aside:** We can define `auto` as syntactic sugar for `(_:$$ Type)`. This
definition allows you to use `auto` as the type for a local variable whose type
can be statically determined by the compiler. It also allows you to use `auto`
as the type of a function parameter, to mean "accepts a value of any type, and
this function will be instantiated separately for every different type." This is
consistent with the
[use of `auto` in the C++20 Abbreviated function template feature](https://en.cppreference.com/w/cpp/language/function_template#Abbreviated_function_template).

### Future work: method constraints

FIXME: skipped for now

Structural interfaces are a reasonable mechanism for describing other structural
type constraints, which we will likely want for template constraints. For
example, a method definition in a structural interface would match any type that
has a method with that name and signature. This is only for templates, not
generics, since "the method with a given name and signature" can change when
casting to a facet type. For example:

```
structural interface ShowPrintable {
  impl Printable;
  alias Show = Printable.Print;
}

structural interface ShowRenderable {
  impl Renderable;
  alias Show = Renderable.Draw;
}

structural interface HasShow {
  method (this: Self) Show();
}

// Template, not generic, since this relies on structural typing.
fn CallShow[T:$$ HasShow](x: T) {
  x.Show();
}

fn ViaPrintable[T:$ ShowPrintable](x: T) {
  // Calls Printable.Print().
  CallShow(x);
}

fn ViaRenderable[T:$ ShowRenderable](x: T) {
  // Calls Renderable.Draw().
  CallShow(x);
}

struct Sprite {
  impl Printable { ... }
  impl Renderable { ... }
}

var x: Sprite = ();
ViaPrintable(x);
ViaRenderable(x);
// Not allowed, no method `Show`:
CallShow(x);
```

We could similarly support associated constant and
[instance data field](#field-requirements) requirements. This is future work
though, as it does not directly impact generics in Carbon.

### Structural conformance

**Question:** How do you say: "restrict this impl to types that have a member
function with a specific name & signature"?

An important use case is to restrict templated definitions to an appropriate set
of types.

**Rejected alternative:** We don't want to support the
[SFINAE rule](https://en.wikipedia.org/wiki/Substitution_failure_is_not_an_error)
of C++ because it does not let the user clearly express the intent of which
substitution failures are meant to be constraints and which are bugs.
Furthermore, the
[SFINAE rule](https://en.wikipedia.org/wiki/Substitution_failure_is_not_an_error)
leads to problems where the constraints can change accidentally as part of
modifications to the body that were not intended to affect the constraints at
all. As such, constraints should only be in the impl signature rather than be
determined by anything in the body.

**Rejected alternative:** We don't want anything like `LegalExpression(...)` for
turning substitution success/failure into True/False at this time, since we
believe that it introduces a lot of complexity, and we would rather lean on
conforming to an interface or the reflection APIs. However, we feel less
strongly about this position than the previous position and we may revisit (say
because of needing a bridge for C++ users). One nice property of the
`LegalExpression(...)` paradigm for expressing a constraint is that it would be
easy for the constraint to mirror code from the body of the function.

**Additional concern:** We want to be able to express "method has a signature"
in terms of the types involved, without necessarily any legal example values for
those types. For example, we want to be able to express that "`T` is
constructible from `U` if it has a `operator create` method that takes a `U`
value", without any way to write down an particular value for `U` in general:

```
interface ConstructibleFrom(Args:$ ...) { ... }
external impl [T:$$ Type] T as ConstructibleFrom[U:$$ Type](U)
    if (LegalExpression(T.operator create(???))) {
  ...
}
```

This is a problem for the `LegalExpression(...)` model, another reason to avoid
it.

**Answer:** We will use the planned
[method constraints extension](#future-work-method-constraints) to
[structural interfaces](#structural-interfaces).

This is similar to the structural interface matching used in the Go language. It
isn't clear how much we want to encourage it, but it does have some advantages
with respect to decoupling dependencies, breaking cycles, cleaning up layering.
Example: two libraries can be combined without knowing about each other as long
as they use methods with the same names and signatures.

```
structural interface HasFooAndBar {
  method (this: Self) Foo(_: Int) -> String;
  method (this: Self) Bar(_: String) -> Bool;
}

fn CallsFooAndBar[T:$$ HasFooAndBar]
    (x: T, y: Int) -> Bool {
  return x.Bar(x.Foo(y));
}
```

One downside of this approach is that it nails down the type of `this`, even
though multiple options would work in a template. We might need to introduce
additional option in the syntax only for use with templates:

```
structural interface HasFooAndBar {
  method (_: _) Foo(Int) -> String;
  method (_: _) Bar(String) -> Bool;
}
```

Note that this would be awkward for generics to support the dynamic compilation
strategy, and we don't expect to want to hide the difference between read-only
and mutable in a generic context.
