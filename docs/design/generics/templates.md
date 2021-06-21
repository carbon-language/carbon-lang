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
