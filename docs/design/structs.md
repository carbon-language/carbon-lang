# Carbon struct types

<!--
Part of the Carbon Language, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Table of contents

<!-- toc -->

- [Background](#background)
- [Goals](#goals)
- [Approach](#approach)
  - [Simple data records](#simple-data-records)
  - [Defaults](#defaults)
  - [Constants](#constants)
  - [Member functions](#member-functions)
    - [Using dot to get values from the struct's namespace](#using-dot-to-get-values-from-the-structs-namespace)
  - [Customizing an object's lifecycle](#customizing-an-objects-lifecycle)
    - [Control over creation](#control-over-creation)
    - [Control over allocation](#control-over-allocation)
    - [Control over destruction / RAII](#control-over-destruction--raii)
    - [Reference counting](#reference-counting)
    - [Moving](#moving)
    - [Copying](#copying)
  - [Other special member functions / operators](#other-special-member-functions--operators)
    - [Equality](#equality)
    - [Ordering](#ordering)
    - [Properties](#properties)
    - [Other operator overloading](#other-operator-overloading)
  - [Const Storage](#const-storage)
  - [QUESTION: Static storage?](#question-static-storage)
  - [QUESTION: Extension / Inheritance?](#question-extension--inheritance)
  - [Mixins](#mixins)
  - [Templating](#templating)
  - [Visibility](#visibility)
  - [Appendix: Construction and Inheritance](#appendix-construction-and-inheritance)
    - [Failure handling](#failure-handling)
      - [Option A: "destructor runs if phase 1 succeeds"](#option-a-destructor-runs-if-phase-1-succeeds)
      - [Option B: "destructor runs if phase 2 succeeds"](#option-b-destructor-runs-if-phase-2-succeeds)
    - [Alternative proposal (requested by Chandler): C++ constructor order](#alternative-proposal-requested-by-chandler-c-constructor-order)

<!-- tocstop -->

## Background

In
["Carbon Language design overview"](https://github.com/jonmeow/carbon-lang/blob/proposal-design-overview/docs/design/README.md),
it says:

    Beyond simple tuples, Carbon of course allows defining named product types. This is the primary mechanism for users to extend the Carbon type system and fundamentally is deeply rooted in C++ and its history (C and Simula). We simply call them structs rather than other terms as it is both familiar to existing programmers and accurately captures their essence: they are a mechanism for structuring data...

This document builds on and extends the definition of tuples found in
["Carbon tuples and variadics" (TODO)](#broken-links-footnote)<!-- T:Carbon tuples and variadics -->.

For more background, see how other languages tackle this problem:

- [Swift](https://docs.swift.org/swift-book/LanguageGuide/ClassesAndStructures.html)
  - has two different concepts: classes support
    [inheritance](https://docs.swift.org/swift-book/LanguageGuide/Inheritance.html)
    and use
    [reference counting](https://docs.swift.org/swift-book/LanguageGuide/AutomaticReferenceCounting.html)
    while structs have value semantics
  - may have
    [constructor functions called "initializers"](https://docs.swift.org/swift-book/LanguageGuide/Initialization.html)
    and
    [destructors called "deinitializers"](https://docs.swift.org/swift-book/LanguageGuide/Deinitialization.html)
  - supports
    [properties](https://docs.swift.org/swift-book/LanguageGuide/Properties.html),
    including computed & lazy properties
  - methods are const by default
    [unless marked mutating](https://docs.swift.org/swift-book/LanguageGuide/Methods.html#ID239)
  - supports
    [extensions](https://docs.swift.org/swift-book/LanguageGuide/Extensions.html)
  - has per-field
    [access control](https://docs.swift.org/swift-book/LanguageGuide/AccessControl.html)
- [Rust](https://doc.rust-lang.org/book/ch05-01-defining-structs.html)
  - has no support for inheritance
  - has no special constructor functions, instead has literal syntax
  - has some convenience syntax for common cases:
    [variable and field names matching](https://doc.rust-lang.org/book/ch05-01-defining-structs.html#using-the-field-init-shorthand-when-variables-and-fields-have-the-same-name),
    [updating a subset of fields](https://doc.rust-lang.org/book/ch05-01-defining-structs.html#creating-instances-from-other-instances-with-struct-update-syntax)
  - [can have unnamed fields](https://doc.rust-lang.org/book/ch05-01-defining-structs.html#using-tuple-structs-without-named-fields-to-create-different-types)
  - [supports structs with size 0](https://doc.rust-lang.org/book/ch05-01-defining-structs.html#unit-like-structs-without-any-fields)
- [Zig](https://ziglang.org/documentation/0.6.0/#struct)
  - [explicitly mark structs as packed to manually control layout](https://ziglang.org/documentation/0.6.0/#packed-struct)
  - has a struct literal syntax,
    [including for anonymous structs](https://ziglang.org/documentation/0.6.0/#Anonymous-Struct-Literals)
  - no special constructor functions
  - supports fields with undefined values
  - supports structs with size 0
  - supports generics via memoized compile time functions accepting and
    returning types
  - [supports default field values](https://ziglang.org/documentation/0.6.0/#toc-Default-Field-Values)
  - [has no properties or operator overloading -- Zig does not like hidden control flow](https://ziglang.org/#Small-simple-language)

## Goals

We wish to provide the Carbon programmer with a way of forming record types that
is [simple](https://www.infoq.com/presentations/Simple-Made-Easy/) (made up of
orthogonal mechanisms that are individually each controlling only one aspect),
has few pitfalls, and supports the high-performance use cases that Carbon is
targeting.

We also want to effectively address common use cases in existing C++ code to
ensure easy and effective migration. Some specific examples that significantly
influence the design:

- RAII patterns
- Non-movable types
- Movable types that are not copyable
- Inheritance (excluding complex multiple inheritance schemes, virtual
  inheritance, etc.)

There are other features we will need to support, but that are not addressed in
this document:

- On in-memory layout, we expect to follow C here by providing a completely
  unsurprising linear layout in memory. We do want to preclude raw pointer
  manipulation to move between data members.
- Can conveniently match common simple on-disk or on-wire binary
  representations.
  - Will need control over packing and defining bitfields (important in practice
    for in-memory performance).
- Control over memory alignment. Alignment can affect things like: speed of
  access, memory usage overhead, which bits in an address/pointer are used vs.
  known to be zero (see
  [https://en.wikipedia.org/wiki/Tagged_pointer](https://en.wikipedia.org/wiki/Tagged_pointer)),
  and which SIMD/vectorization instructions you can use.
- Specification of which bit patterns are illegal for this type, so that we can
  efficiently implement things like `Optional&lt;T>`.

## Approach

### Simple data records

First we need to support standard product types, with a collection of variables.
Here is our current syntax:

```
struct Widget {
  var Int: x;
  var Int: y;
  var Int: z;
  var String: payload;
}
```

This defines a new type named `Widget` that contains four _fields_, three with
type `Int` and the last with type `String`. These fields have names (`x`, `y`,
`z`, `payload`). Now that this type is defined, you may use the name `Widget` in
future variable declarations.

```
var Widget: w = ...;  // Initialization syntax discussed below
```

And then you may access the fields using the `.` syntax on the variable:

```
w.y += 2;
Print(w.payload);
```

**Note:** This scope plays by the same rules as other scopes in that we forbid
name clashes.

**Note:** We will likely need to make the name of the struct available inside
the body/definition of the struct. In C++ this is used in a few different
places:

- [Curiously recurring template pattern](https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern)
- Defining pointers to the same type in e.g. linked lists or tree data
  structures
- The definition of member functions

This implies we will need to support "incomplete" types. These should hopefully
not include weird C++ behavior:

- ["injected class name": where the name of a struct means something different inside the struct definition than outside](https://en.cppreference.com/w/cpp/language/injected-class-name).
  We will need some other way of addressing the issue of naming types in nested
  parameterized structs.

      ```

  struct List(Type: T) { struct Node(ListPolicy: P) { var T: data; var
  Ptr(Self): next; // ok, Self is List(T).Node(P) var Ptr(???): list; // how do
  you write this without repeating the arguments for List? } var
  Ptr(Node(ComputePolicy())): first; }

```



    One solution, assuming we think this is rare enough, is to require the programmer to add an alias `type ListSelf = Self` to the outer struct.

*   [In C++, if D and B are incomplete, the compiler will assume D* a not a subtype of B*, when in fact the answer should be "don't know"](https://google.github.io/styleguide/cppguide.html#Forward_Declarations).

This is an area we will have to define for Carbon more rigorously, perhaps by restricting forward declarations to things defined in the same library. Note that C++ also uses incomplete types to forbid recursive types that can't be given a finite size, but other languages resolve that problem with independent mechanisms.

**Concern:** Incomplete types introduce complexity in the language that could possibly be resolved in other ways, such as two-pass parsing of the file to allow declarations to be in any order.

**Note:** We should probably allow structs to be created without being named. Maybe the resulting name is essentially a structural description of the fields, as is done with [tuples (TODO)](#broken-links-footnote)<!-- T:Carbon tuples and variadics -->? This would make `struct {...}` an expression whose value is an anonymous type.  Example:


```

var struct { var Int: x; var Int: y }: point = ...; point.x = 2; point.y = 3;

```


**Proposal:** All fields of a struct must be named.

This is a difference from tuples. This is something we could potentially change, but as of now we don't have a use case. Another possibility is that names get auto-assigned in tuples (`_0`, `_1`, etc.)?

**Alternative considered:** We could support [anonymous fields as is done in Go](http://golangtutorials.blogspot.com/2011/06/anonymous-fields-in-structs-like-object.html). This raises issues about name collisions, particularly as types change. The policy Go uses of only promoting field names from anonymous nested struct that don't conflict with any field of the containing struct is reasonable (see [https://play.golang.org/p/b3tp6pgjbKW](https://play.golang.org/p/b3tp6pgjbKW), for example). However, we at this time are avoiding the complexity of including this feature.


### Tuples with named fields

By allowing users to specify the names for fields for tuples, we bring structs and tuples into much greater alignment. This will allow us to [say that they both records](#tuples-and-structs-are-both-records), and [provide a literal syntax for initializing structs](#simple-initialization-from-a-tuple). Also, the [Google C++ style guide](https://google.github.io/styleguide/cppguide.html#Structs_vs._Tuples) recommends structs over C++'s pair & tuple precisely because naming the components is so important.

**Proposed syntax:**


```

var auto: proposed_value_syntax = (.x = 1, .y = 2); var (.x = Int, .y = Int):
proposed_type_syntax = (.x = 3, .y = 4); var struct { var Int: x; var Int: y; }:
equivalent_struct = (.x = 5, .y = 6);

var (.z = Int): tuple_with_single_named_field = (.z = 7);

```


The `.` here means "in the tuple's namespace"; so it is clear that `x` and `y` in that first line don't refer to anything in the current namespace. The intent is that `.x` reminds us of field access, as in `foo.x = 3`.

Notice that the type of `(.x = 3, .y = 4)` is represented by another tuple, `(.x = Int, .y = Int)`, containing types instead of values. This is to mirror the same pattern for tuples with unnamed components: `(Int, Int)` (a tuple containing two types) is the type of `(3, 4)`, as seen in [the tuple doc (TODO)](#broken-links-footnote)<!-- T:Carbon tuples and variadics -->.

**Open question:** We could drop the convenience of the type of a tuple being another tuple of types. We would instead make the type of a tuple an anonymous struct with unnamed fields. Alternatively, we could get rid of the anonymous struct syntax, so there is only one way to do things.

**Proposal:** Order still matters when using named fields.

Note that my first design was that two tuples with the same (name, type) pairs (as a set) were compatible no matter the order they were listed. In particular, these statements would not be errors:


```

assert(proposed_type_syntax == (.y = 4, .x = 3)); // Error! Order doesn't match.
proposed_type_syntax = (.y = 7, .x = 8); // Error! Order doesn't match.

```


This has a problem though, when the order of fields of the type doesn't match the order of the initializer, what order are the components evaluated? A reader of the code will expect the evaluation to match the most visible order written in the code: the order used for the initializer. But the ordering of the fields of the type is what will determine the destruction order -- which really should be the opposite of the order that values are constructed.

**Concern:** Field order affects hidden implementation details like alignment and padding, and yet the field order is also API-visible in a way that clients almost can't avoid depending on.

We may need to provide a way for users to explicitly specify a struct layout and an explicit (different) order of initialization for the rare structs where this matters.

**Proposal:**  Just as tuples without names are used to provide positional arguments when calling a function, see [Carbon tuples and variadics (TODO)](#broken-links-footnote)<!-- T:Carbon tuples and variadics -->, tuples with names should be used to call functions that take keyword arguments. Tuples may contain a mix of positional and named members; the positional members will always come first (and similarly positional arguments will always come before keyword arguments in functions).

Example:


```

// Define a function that takes keyword arguments. // Keyword arguments must be
defined after positional arguments. fn f(Int: p1, Int: p2, .key1 = Int: key1,
.key2 = Int: key2) { ... }

// Call the function. f(positional_1, positional_2, .key1 = keyword_1, .key2 =
keyword_2); // Keyword arguments must be provided in the same order. // ERROR!
Order mismatch: f(positional_1, positional_2, .key2 = keyword_2, .key1 =
keyword_1);

```


We are currently making order always matter for consistency, even though the implementation concerns for function calling may not require the same constraint.

The additional benefit of a tuple with named components is that it provides a natural syntax for literals we can use to initialize structs.


### Tuples and structs are both "records"

**Proposal:** For simplicity, we should make tuples and structs two ways of making record data types, with slightly different restrictions.


```

var struct { var Int: a; var Int: b }: as_struct = (.a = 1, .b = 2); var (.a =
Int, .b = Int): as_tuple = (.a = 3, .b = 4); static_assert(typeof(as_struct) ==
typeof(as_tuple));

```


Tuples are different from structs in that they:



*   can only have data fields (no methods, constructors, or static members), and those data fields can not have default initial values;
*   can not have a name (and therefore tuples use [structural equality](#record-type-equality));
*   have only public members (no access control);
*   may have anonymous fields, indexed by position.

Their benefit is that they have a simple literal syntax, and a short-hand for specifying a particular tuple type. These features make them convenient to use as part of the function call protocol, among other places.

**Reasoning:**

We should have fewer, more general mechanisms where possible. This means fewer places where we would have to convert between tuples and structs, and fewer "islands" of functionality. For example, by making tuples and structs the same thing, we will only need a single reflection API to handle both.

**Concern:** It would be even better to say tuples are a kind of struct, and so we might be able to simplify this even further. In particular, it would nice not to have to define a "record" to encompass the union of things that can be expressed using tuple and record syntax.

The only tricky part are positional fields, which are currently only available in tuples not structs. Two possible solutions:



1. The conceptual model of a struct includes fields without names, but we don't provide any syntax for writing such a struct down.
2. We add a syntax for positional fields in a struct.

[chandlerc](https://github.com/chandlerc) says:


    I somewhat like #2 as it means you can define custom, named types that expose the exact same API as a tuple which seems quite useful. I've not thought about it enough to know if there are problems here though.


#### Record type equality

For types with names, we say that two types are equal if their fully qualified names are equal. Here fully qualified includes both the namespace and any parameters. For example, let's say we have three types, `A`, `B`, and `C(T)` defined as follows:


```

struct A { var Int: x; var Int: y; }

struct B { var Int: x; var Int: y; }

struct C(Type:\$\$ T) { var T: x; var T: y; }

```


Here `A`, `B`, and `C(Int)` all define different types even though they are the same structurally. Further `C(Int)` is different from `C(Bool)` since their names include any parameters.

This implies that you can't have two types with the same name in the same namespace. In some cases, a struct may be given a name when it is defined, indicating that the user wants to use name-based type equality, but that name may not be visible in some other scope. In this example:


```

fn ReturnsAType(bool: x, type: T) -> Type { if (x) { struct S(type: U) { ... }
return S(T); } else { struct S { ... } return S; } }

```


there are in fact two types named `S` in different scopes, neither of which is visible outside the function. In this case, the types get names that are sufficient to disambiguate them, like `ReturnsAType(True, Int)` might return `ReturnsAType.S#1(U=Int)`.

For types without names, type equality is structural. Effectively the name of a type like `(Int, Bool, .s = String)` includes all the type-specific information, like `"(Int, Bool, .s = String)"`. If the type includes other things beyond slots (member functions, etc.), those must match exactly.

Types with names are never equal to types without names. So the tuple type `(.x = Int, .y = Int)` does not match the types `A` and `B` above, but is equal to this struct without a name:


```

struct { var Int: x; var Int: y; }

```



#### Type deduction and type functions

Context: [Carbon chat Oct 31, 2019: Higher-kinded types, normative types, and type deduction (TODO)](#broken-links-footnote)<!-- T:Carbon chat Oct 31, 2019: Higher-kinded types, normative types, and type deduction --><!-- A:#heading=h.r48w6htktgjf -->

TODO: Move this section to [Carbon pattern matching](https://github.com/josh11b/carbon-lang/blob/pattern-matching/docs/design/pattern-matching.md#deduced-specification-match-rules).

Since parameterized types have names that include their parameters, this means that parameter may be deduced when calling a function. For example,


```

struct Vec(Type: T) { ... }

fn F[Type: T](Vec(T): v) { ... }

var Vec(Int): x; F(x); // `T` is deduced to be `Int`.

```


In addition, a parameterized type can actually be thought of as a function, which can also be deduced:


```

// Continued from above fn G[fn(Type)->Type: V](V(Int): v) { ... } G(x); // `V`
is deduced to be `Vec`

fn H[Type: T, fn(Type)->Type: V](V(T): v) { ... } H(x); // `T` is deduced to be
`Int` and `V` is deduced to be `Vec`.

```


This would be used in the same situations as [C++'s template-template parameters](https://stackoverflow.com/questions/213761/what-are-some-uses-of-template-template-parameters).

**Proposal:** The above deductions are only available based on a type's name, not arbitrary functions returning types.

If we write some other function that returns a type:


```

fn I(Type: T) -> Type { if (T != Bool) { return Vec(T); } else { return BitVec;
} }

```


In theory, since the function is injective, it might be possible to deduce its input from a specific output value, as in:


```

// Not allowed: `I` is an arbitrary function, can't be involved in deduction: fn
J[Type: T](I(T): z) { ... } var I(Int): y; // We do not attempt to figure out
that if `T` was `Int`, then `I(T)` is equal // to the type of `y`. J(y);

```


If we wanted to support this case, we would require the type function to satisfy two conditions:



*   It must be _injective_, so different inputs are guaranteed to produce different outputs. For example, `F(T) = Pair(T, T)` is injective but `F(T) = Int` is not.
*   It must not have any conditional logic depending on the input. We could enforce this by requiring it to take arguments generically using the `:$` syntax.

If both those conditions are met, then in principle the compiler can symbolically evaluate the function. The result should be a type expression we could pattern match with the input type to determine the inputs to the function. In general, this might be difficult so we need to determine if this feature is important and possibly some other restrictions we may want to place. A hard example would be deducing `N` in `F(T, N) = Array(T, N * N * N)` given `Array(Int, 27)`. This makes me think this feature should be deprioritized until there are compelling use cases which can guide what sort of restrictions would make sense.


#### Type immutability

Types should generally be immutable once they are defined. (Caveat: we do need to support transitioning from a forward-declared incomplete type, to a type in the process of being defined, to defined type that is no longer mutable.) There should not be any operation that adds a field or function to an existing type, or does monkey patching -- unless it does this by creating a new type instead of modifying an existing one. The one exception, which we are still uncertain about, would be if we want to support [slots with static storage, see the section below](#question-static-storage).


### Simple initialization from a tuple

Let's say we have a struct and we want to initialize a variable of that struct type.


```

struct Point { var Int: x; var Int: y; }

var Point: p = ???;

```


**Goals & Principles:**



*   Every field will be initialized unless the `uninit` keyword is used in initialization (or you will get a compiler error).

**Proposal:** we allow you to initialize a struct from a tuple with names that match the names of the fields, in the same order ([see above](#bookmark=id.opejfo1k8om9)).


```

var Point: p1 = (.x = 1, .y = 2); var Point: p2 = (.y = 4, .x = 3); // Error:
order matters.

// No need for a trailing comma in tuples with a single named field. struct
OneMember { var Int: x; } var OneMember : a = (.x = 12);

```


**Proposal:** We do not allow positional initialization from a tuple without names:


```

// Error: Tuple missing names: var Point: p = (1, 2); assert(p.x == 1);
assert(p.y == 2);

````


The rule here is:



*   The tuple to the right of the `=` is passed as the argument to the factory function.
*   The default factory function for a struct has one keyword argument per field, with the same names and order as the field definitions.

So just like we require positional & keyword arguments to match between the call and the declaration for functions (see ["Carbon pattern matching"](https://github.com/josh11b/carbon-lang/blob/pattern-matching/docs/design/pattern-matching.md)), the names are required when initializing. If for some reason it makes sense for a specific type to support initialization from a tuple with unnamed fields, the definition of that type can add a factory function that takes positional arguments (see [the section on object creation below](#control-over-creation)).

**Proposal:** Every field of the struct will be initialized unless there is a visible indication otherwise in the source code.

For example, what should happen if you say `var Point: p;` without any initializer?



*   If the `Point` struct defines defaults (see next section) for every field, then \
`var Point: p;` just gives you the default values.
*   **Rejected alternative:** We require you to specify a value to indicate that the value is initialized, which could be the empty tuple to get the default values: `var Point: p = ();`.
*   We likely also define default initialization ("default defaults") for some types as part of the language (`Int` defaults to 0, optional values default to not present, etc.) or via a default constructor for that type. If every field has a default (either from the struct definition (preferred) or from the field's type), then `var Point: p;` will initialize `p` with those defaults. Note however, there will be some types for which we have no default value (e.g. a non-nullable pointer, see [Carbon pointers and references (TODO)](#broken-links-footnote)<!-- T:Carbon pointers and references -->).
*   **Question:** Do we want to define default initialization for arrays, or should we require the user to be explicit to avoid hiding a possibly large performance cost?
*   Otherwise we forbid `var Point: p;` with a compile error. If you truly want an uninitialized `Point`, you would have to explicitly say something like:

    ```
var Point: p = uninit;
````

- The question of how future code should operate on a possibly uninitialized
  value is addressed in a separate document:
  ["Uninitialized variables and lifetime" (TODO)](#broken-links-footnote)<!-- T:Uninitialized variables and lifetime -->
  or
  [carbon-uninit-v2 (TODO)](#broken-links-footnote)<!-- T:Carbon Uninitialized variables and lifetime v2 -->.
- If `Point` does have a default value, users should still be able to write \
  `var Point: p = uninit;` to skip the work of initialization when they know this
  value will be overwritten later.
- **Rejected alternative:** We allow `var Point: p;` without initializing every
  field as long as we can statically determine that the fields of `p` are
  initialized before use. The `uninit` syntax would be an escape hatch saying "I
  know what I'm doing." \

- We do not reflect the possibly uninitialized status in the type itself, but we
  require that the compiler can statitically determine when the value is
  initialized. If it can't, we require the user to use `Optional(T)` instead of
  `T`. We do not want the compiler silently associating a `bool` to track
  initialization status. \

- You may specify that specific fields are not initialized, via something like

      ```

  var Point: p = (.x = 2, .y = uninit);

````



    However, we don't want to silently leave some fields uninitialized, so


    ```
var Point: p = (.x = 2);
````

    should either do some default initialization of the `y` field or trigger a compile error.

**Proposal:** It is legal to pass `uninit` as an argument to a type using the
default factory function.

```
var auto: p = Point(.x = 10, .y = uninit)
```

User-written factory functions can opt in to allowing this by accepting some
sort of `MaybeInit(T)` type.

I envision `uninit` is going to be to initialization as `nil` is to `T?` /
`Optional(T)`. That is, there is a sum type, let us call it `MaybeInit(T)`, that
represents either a `T` value or `uninit`. Constructors (and the default factory
function you get for structs with just data members) in particular take that sum
type. If you want to write a factory function that supports `uninit` arguments,
you can accept that type as well. This may in fact just be optional -- see the
discussion in the
[Carbon-uninit-v2 doc (TODO)](#broken-links-footnote)<!-- T:Carbon Uninitialized variables and lifetime v2 --><!-- A:# -->.

**Rejected alternative:** There is no initialization of structs (with names)
from tuples. Instead you'd have to write something like

```
var auto: p = Point(.x = 10, .y = 20)
```

Here `Point(.x = 10, .y = 20)` would be whatever our syntax is for invoking a
type's constructor / factory function.

Discussion: TODO. [chandlerc](https://github.com/chandlerc) please put your
reasoning here!

### Defaults

**Proposal:** We support specifying a default value for struct data members.
Members with a default need not be explicitly initialized.

At first we may only want to support simple cases where the default value is a
compile-time constant:

```
struct Point {
  var Int: x = 0;
  var Int: y = 0;
}
var Point: p;
assert(p.x == 0);
assert(p.y == 0);

var Point: p2 = (.x = 1);
assert(p2.x == 1);
assert(p2.y == 0);
```

**Proposal:** We should allow initializer expressions that refer to names
introduced earlier in the struct definition. Initializers should definitely be
allowed to refer to constants ([see below](#constants)), but we could also
support member variables. The semantics would be that members are initialized in
the order that they are declared in the `struct` declaration, and later members
see the (possibly default) value that the earlier members were initialized with.

```
struct Point {
  var Int: x = 0;
  var Int: y = x + 2;
}
var Point: p;
assert(p.x == 0);
assert(p.y == 2);
var Point: p2 = (.x = 1);
assert(p.x == 1);
assert(p.y == 3);
```

**Rejected alternative:** We could possibly allow initializers to refer to any
name declared in the struct. This would mean we would have to deal with forward
references and the possibility of cycles (which might still be okay in some
cases!).

```
struct Point {
  var Int: x = y + 1;
  var Int: y = x + 2;
}
var Point: p = (.x = 0);
assert(p.x == 0);
assert(p.y == 2);
var Point: p2 = (.y = 1);
assert(p.x == 2);
assert(p.y == 1);
```

Rationale: We have been consistently only allowing backward references in Carbon
to simplify the parser and tooling such as code completion.

Concern: Only allowing backward references may actually make tooling and code
generation more complex. Tooling would have to enforce this constraint, and code
generators would have to be careful to follow these rules.

### Constants

In addition to data members, a struct is a namespace which we can put constants.
Here, I'm specifically talking about:

- Values with no associated storage (like `Int: 3`), or only storage in
  read-only / immutable memory (like `String: "foo"`).
- These values have names associated with the type rather than the instance.
  They may still be accessed through the instance (like other members of the
  type like [member functions discussed below](#member-functions)), but that is
  not a requirement.

These values might be constants, other types, functions, etc.

There may be actually two concepts here: aliases and constants, see the
description of the `alias` keyword in
["Carbon language design / Aliases"](https://github.com/jonmeow/carbon-lang/blob/proposal-design-overview/docs/design/README.md#aliases).
Resolving that question is outside the scope of this document.

**Proposed syntax:** To be consistent with the
[generic and template](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/designs/generics-overview.md)
syntax for designating something as a constant, there are actually two syntaxes.

- Normal constants, where the value is given in the source, would use the
  template syntax: `var` &lt;type>`:$$` &lt;id> `=` &lt;value>`;`
- Constants that depend on generic parameters and so are only known at
  codegen-time would use the generic syntax: `var` &lt;type>`:$` &lt;id> `=`
  &lt;value>`;`

In both cases, the `=` and a value would be required, unlike with variables.

```
struct MathConstants {
  var Float64:$$ Pi = 3.1415926535897932384626433832795;
  var Float64:$$ E = 2.7182818284590452353602874713527;
}

var Complex : one = Math.Exp(Complex(0, 2 * MathConstants.Pi));

struct ErrorCodes {
  var auto:$$ OK = 0;
  var auto:$$ FAILURE = 1;
}

struct IteratorImpl(Type:$ T) { ... }
struct Container(Type:$ T) {
  // This constant depends on a generic parameter, so we use `:$`.
  var Type:$ Iterator = IteratorImpl(T);
  var Iterator: begin;  // The `Iterator` name is visible here
}
```

**Proposal:** The name introduced by the type would be visible within the struct
definition after it is declared. Note this is in contrast with C++ where I
believe it is visible everywhere within the struct even before its definition.

In particular, future constants can use previous constants to compute their
values.

**Proposal:** You may access these constants either through the name of the type
or an instance of that type.

```
struct Foo {
  var Int:$$ c = 1;
  var Int: x = 2;
}

var Foo: y;
assert(Foo.c == y.c);
```

**Concern:** this introduces non-uniformity into the model, and perhaps should
be forbidden until we have a compelling use case.

**Rejected alternative:** Constants always need to be assigned a value when they
are defined. We could consider skipping the type, implicitly always making it
`auto`. This however is not consistent with how we are using this syntax
elsewhere in Carbon.

**Proposal:** Just like at the global scope, we should allow users to introduce
a declaration with `struct` or `fn`, which also defines a constant.

```
struct Foo {
  struct Bar {
    var Int: value;
  }
  var Bar: data;
}
```

Note that this is different from:

```
struct Foo {
  var Type:$$ Bar = struct {
    var Int: value;
  };
  var Bar: data;
}
```

In the first case the `Foo.Bar` type is considered named, and in the second case
it is an anonymous type (and so uses structural equality).

**Proposal:** Structs are allowed to have size 0 by default.

If we determine there is a need, we can provide a way to for the user to
explicitly request that the members of a struct have different addresses. For
now, that is accomplished by adding a data member so that objects of that type
would have size > 0.

In C++, a struct with no data members will still have a size > 0, to guarantee
that every object has a different address. This leads to the
[empty base class optimization](https://www.google.com/search?q=empty+base+class+optimization+c%2B%2B),
which just seems like a pitfall and extra complexity.
[A C++20 proposal](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p0840r2.html)
includes an attribute to get this behavior: `[[no_unique_address]]`, but we
believe that C++20 has the wrong default behavior.

**Concern:** Rust allows empty objects, and it seems to cause a fair amount of
complexity and special-casing in generic code. In Rust that complexity is
limited to code that's explicitly tagged as unsafe, but I don't know if we'll be
able to get away with that in Carbon. See
[Handling Zero-Sized Types - The Rustonomicon](https://doc.rust-lang.org/nomicon/vec-zsts.html).

### Member functions

The previous section proposes two equivalent ways of defining a function as a
member of a struct type:

```
struct Foo {
  fn AddOne(Int : x) -> Int { return x + 1; }
  // Equivalent to:
  var auto:$$ AddOne = fn(Int : x) -> Int { return x + 1; }
}

assert(Foo.AddOne(2) == 3);
```

This is just like "static" functions in C++. As mentioned above in
[the constants section](#constants), and similar to C++, you also may refer to
`AddOne` through an instance of `Foo`.

The more interesting use case here is for functions that take a pointer to the
type as their first argument:

```
struct Counter {
  var Int: current = 0;
  fn Increment(Ptr(Counter): this) -> Int {
    this->current += 1;
    return this->current;
  }
}
```

(Note that `Counter` is in the process of being defined when it is being used in
the definition of the `Increment` function.** Question:** Do we want to restrict
it to only operating on names that have already been defined? Or do we want to
postpone type checking the body until the `Counter` type is done being defined?)

We have a specific syntax for calling such functions on an object of that type
using the dot (`.`) member access operator:

```
var Counter: c;
assert(c.Increment() == 1);
assert(c.Increment() == 2);
```

This is equivalent to looking up the `Increment` function in `c`'s type, and
calling it with `c`'s address, effectively `c.Increment()` is treated something
like `Counter.Increment(&c)`. So the model is:

- Functions may be defined as members of the type.
- Functions that take a pointer to the containing type (or a const pointer, once
  we have such a thing) as their first argument support a convenient call syntax
  that looks like member access. **Question:** Does the argument have to have a
  special name (`this` or `self`) to get the special treatment?
- Just like in the outer / module level scope, we only allow multiple
  definitions of the same symbol if they all define functions. In this case, we
  resolve them into a single function at the call site using pattern matching
  using the same rules. (See:
  ["Carbon pattern matching"](https://github.com/josh11b/carbon-lang/blob/pattern-matching/docs/design/pattern-matching.md)
  and
  ["Carbon tuples and variadics" (TODO)](#broken-links-footnote)<!-- T:Carbon tuples and variadics -->.)

**Question:** What is the type of `foo.Bar`? Is it a closure,
`...: args -> foo.Bar(Args...)`, capturing the value of `foo`? If you don't want
to capture the value of `foo`, perhaps you have to do something like
`typeof(foo).Bar`?

**Open question:** Instead of making this member function calling syntax
available for any member with a first parameter having a special type or name,
we could explicitly mark the function as being called in a different way.
Perhaps something like:

```
struct Counter {
  var Int: current = 0;
  method (Ptr(Counter): this) Increment() -> Int { ... }
}
```

**Proposal:** We support defining functions out of line.

Benefits:

- We would definitely have resolved enough of the struct definition to do type
  checking of the body of the function.
- It is easier for readers to get an overview of the entire struct definition
  (though maybe IDEs, codesearch, etc. can collapse function definitions down by
  default?).
- The code will have one less level of indenting.
- It would allow the definition of a struct to be spread out across multiple
  files within the same library (which we've been told is important).
- Supports a programming style that avoids unnecessary physical dependencies
  (changing a function definition provably doesn't affect compilations that
  didn't even consider the file containing that definition), potentially
  allowing a more minimal rebuild. **Concern:** We should figure out to what
  extent we care about this. Do we want to design the language such that it's
  easy to determine whether the changes to a particular file require its
  dependents to be recompiled? If so, how, and what constraints does that
  impose?

Downsides:

- Less convenient to write.
- Have to repeat the signature (see example below).
- Multiple ways to write the same code.
- More burden for tooling teams (e.g., cross-references, “jump to
  definition”/”jump to declaration” navigation commands, “define inline”/”define
  out of line” boilerplate generation, “move inline”/”move out of line”
  refactorings — clangd team is working on all of this for C++, whereas it could
  be avoided in Carbon).
- More things to match up and cross-check during semantic analysis.
- More difficult to find the function definition.
- Opens up questions about the extent to which the function signature should be
  repeated (e.g., do we repeat attributes? etc.)

Proposed syntax:

```
struct Counter {
  var Int: current = 0;
  fn Increment(Ptr(Counter): this) -> Int;
}

// Issue: repeats signature already written above. Needed, though, when the
// function is overloaded, so there are multiple functions with the same name.
fn Counter.Increment(Ptr(Counter): this) -> Int {
  this->current += 1;
  return this->current;
}
```

All members of a struct would have to be defined in the same library (not
necessarily the same file) as the struct itself.

**Proposal:** Chandler has proposed that the `self`/`this` argument could be
passed by value (instead of via a pointer) in the
[Carbon language design doc](https://github.com/jonmeow/carbon-lang/blob/proposal-design-overview/docs/design/README.md#structs).
This is incompatible with inheritance (due to the
[slicing problem](https://en.wikipedia.org/wiki/Object_slicing)), so we should
make an effort to make sure these methods are not called with a descendant type
with compile time checks:

- it is safe if the struct is "final" (currently we are saying this is the
  default, see [the inheritance section](#question-extension--inheritance))
- it is safe if accessed via an object on the stack
- it is safe if accessed via a "pointer to exactly T" instead of a "pointer to T
  or a subtype" (but
  [we are currently not making that distinction in the type system](#control-over-destruction--raii))

Geoffrey Romer says: "The use cases are the same as the use cases for passing by
value in any other argument position- you pass by value because it's simpler,
and it establishes that the object isn't shared with any other code. This makes
the code much simpler to reason about, both for humans and for the optimizer."

**Question (minor):** You can also have functions as `var` members of a type.
Should member access with a dot (`.`) falls back to the ordinary field access
semantics, or should it also treat functions with a `this`/`self` first argument
as special, passing in the calling object as the first argument?

Example:

```
struct Foo {
  var Int: a = 0;
  var auto: bar = fn(Int: x) -> Int { return x + 1; };
  var auto: baz = fn(Ptr(Foo): this) -> Int { return this->a + 1; };
}

var Foo: f;
// f.bar is an ordinary function (Int) -> Int.
assert(f.bar(3) == 4);
// Since f.bar is declared as a `var`, it can be changed from its initial
// default value.
f.bar = fn(Int: x) -> Int { return x + 2; };
assert(f.bar(3) == 5);
// Case 1: No special treatment for functions taking a pointer to the
// containing struct type as a first argument if they are variables.
assert(f.baz(f) == 1);
f.a = 2;
assert(f.baz(f) == 3);
f.a = 0;
// Case 2: Same special treatment, whether it is a var or not.
assert(f.baz() == 1);
f.a = 2;
assert(f.baz() == 3);
```

One issue is this might force us to include whether "the first argument is named
`this`" (or `self` if we go that way instead) as part of a function's type.

#### Using dot to get values from the struct's namespace

**Proposal:** We provide a convenient syntax, when calling a member function,
for accessing other members of that struct. Example:

```
struct Fruit {
  var Int:$$ Apple = 0;
  var Int:$$ Banana = 1;
  var Int: c;
  fn MixWith(Ptr(Fruit): this, Int: fruit) { ... }
}
var Fruit: f = (.c = Fruit.Banana);
f.MixWith(.Apple);
// same as:
f.MixWith(Fruit.Apple);
```

Here the `.` means "look up the symbol `Apple` in the context of the function
being called, rather than the caller's namespace." So it would find the `Apple`
member of `Fruit`.

**Rejected:** We could also imagine that since initialization of a struct should
be thought of as a call to the struct's constructor, and should support the same
syntax.

```
var Fruit: g = (.c = .Banana);
```

Reasoning: ([chandlerc](https://github.com/chandlerc)) The issue I see is that
we want the expression after the `=` to stand on its own as a valid expression.
But it can't if things like name lookup rely on the context of the declaration.

([josh11b](https://github.com/josh11b)) One problem here is that the `Fruit`
context is a bit removed -- really the right side is making a tuple and if we
separated that out we'd certainly lose that context.

```
var auto: initial = (.c = .Banana);  // Error, what "Banana"?
var Fruit: g = initial;
```

This would be an advantage of initializing using a `Fruit` constructor \
(`var auto: g = Fruit(.c = .Banana);`) instead of allowing initialization from a
tuple.

**Question (minor):** should the context be the `Fruit` type, or also the `f`
object? For example, should this be legal?

```
f.MixWith(.c);
// same as:
f.MixWith(f.c);
// One possible difference: "f.MixWith(.c)" may not have to re-evaluate f?
```

### Customizing an object's lifecycle

**Goals/principles:**

- We want to give users a great deal of fine-grained control over what code is
  executed at each of an object's lifetime events: creation, copying, moving,
  and destruction. This is to allow high-performance techniques like
  thread-specific memory pools, arena allocators, free lists, etc.
- Users should be able to use "move" operations avoid unnecessary copying or
  duplication of resources when e.g. returning a value from a function.
- The lifecycle should be deterministic and predictable, so users can rely on it
  for things like holding resources / mutex locks
  ([RAII](https://en.wikipedia.org/wiki/Resource_acquisition_is_initialization)).
  **Question:** Possibly we could give the compiler more freedom to optimize if
  we require users to explicitly mark which types have precise lifetime
  requirements.
- We want to avoid pitfalls that users would have to avoid, such as operating on
  a partially-constructed object with surprising or undefined semantics.
- We want to reduce boilerplate & code duplication to define similar operations
  like C++'s copy constructor & assignment operator.
- Initializing a struct from a tuple uses exactly the same rules as passing a
  tuple as the argument list to a function, since it is actually just passing
  that tuple to a factory function.

#### Control over creation

**Note:** This section was quite long, particularly to deal with the interaction
between creation and inheritance. To make reading this document more manageable,
I've moved most of the details related to inheritance to
[an appendix](#appendix-construction-and-inheritance), though there is a little
bit of duplication between the two sections.

**Problem:** Normally the job of a constructor is to establish the invariants
expected of a type. We want to avoid exposing partially-constructed objects to
user-written code. For example, calling a member function on an object during
its construction process can lead to surprising or undefined results. It is well
worth reading
[Perils of Constructors](https://matklad.github.io/2019/07/16/perils-of-constructors.html)
which describes the problem and surveys various solutions used by existing
languages. Here is a quick summary of the existing alternatives:

- C++: has a special, limited syntax for setting initial values for member
  variables. This is complicated (separate syntax, separate rules), awkward
  (constructors look different from other code), non-orthogonal. For example,
  the arguments to a parent's class constructor must be simple expressions;
  there is no provision for writing arbitrary code that does control flow or
  saves common values in a temporary variable beforehand. It also doesn't handle
  failure well. The one bright spot is it does address the question of how to
  deal with "const" data member that you wouldn't otherwise be able to assign to
  using the normal assignment syntax.
- C: Allocation gives you an unsafe pointer to uninitialized memory. This is a
  very simple model, which we could easily improve by allowing the type to
  define an initialization function. However, it would still be too easy to
  misuse the uninitialized `this` pointer during that initialization function.
  It also has no support for "const" members, for that we would need some
  mechanism for allowing those values to only be assigned to once (or at least,
  only in the constructor).
- Swift: Like previous option, but with two-phase initialization. Basically:
  language rules to make sure you initialize everything, and don't use anything
  before its initialized. See
  [Swift's doc on Initialization](https://docs.swift.org/swift-book/LanguageGuide/Initialization.html).
- Rust: There is a single constructor defined by the language, not by the user
  (see
  [Rust's doc on constructors](https://doc.rust-lang.org/nomicon/constructors.html)).
  Upside: user is never exposed to a partially constructed object. Downsides:
  you may not be able to optimize away copying, you need a story for
  inheritance, and you need some way to handle any initialization that requires
  knowledge of the address of the resulting object (registration, linked lists,
  etc.).

**Problem:** Support failure of user code during object creation. Currently this
adds a lot of complexity to C++, and untested code paths. It would be better if
there was a transition where the object goes from not existing to constructed
that did not have any user code that could fail.

(There is also the problem of allocation failure, but that is for the next
section.)

**Problem:** We'd like to support multiple constructors, where it is clear at
the call site what you are getting. The main issue is that a list of arguments
on its own may not be a very clear signal as to what behavior you are going to
get, see
[the Constructor’s Signature section of the Perils of Constructors article](https://matklad.github.io/2019/07/16/perils-of-constructors.html#constructors-signature).
Keyword arguments and named constructors / factory functions are ways of
addressing this.

**Proposal:** We adopt a Rust-style construction protocol which we change to
support access to the address of the constructed object.

Our starting point is that for a given type `struct Foo` with fields `a` and
`b`. The compiler gives us a constructor function (which I'm writing `construct`
below) that takes arguments `a` and `b` (with defaults matching the defaults
defined in the declaration of `Foo`, if any) and returns a `Foo` object, but for
our case lets say that function is a private implementation detail used by any
factory functions we want to define.

**Concern:** The "constructor" vs. "factory function" terminology here could be
improved, and in particular diverges from the C++ usage of the word
"constructor".

For example, we might declare `struct Foo` with one factory function like so
(making up a syntax):

```
struct Foo {
  var Int: a;
  var Int: b;
  fn operator create(Int: sum, Int: difference) -> Ptr(Foo) {
    return construct(
        .a = (sum + difference) / 2, .b = (sum - difference) / 2);
  }
}

var Foo: x = (7, 1);
assert(x.a == 4);
assert(x.b == 3);
```

**Bikeshed:** What is a good syntax here? Instead of `create` we could use
`init` or `factory` or the name of the class (to match C++). The word `operator`
here could easily be replaced too.

Under some conditions, Carbon will provide the default factory function which
just matches `construct`. So for example, this declaration:

```
struct Foo {
  var Int: a = 2;
  var Int: b;
}
```

would be equivalent to:

```
struct Foo {
  var Int: a = 2;
  var Int: b;
  fn operator create(.a = Int: a_ = 2, .b = Int: b_) -> Ptr(Foo) {
    return construct(.a = a_, .b = b_);
  }
}
```

This matches the behavior described in
[earlier](#simple-initialization-from-a-tuple) [sections](#defaults). **TBD:**
exact condition when this function will be provided -- maybe if all data members
are publicly writable and/or if there are no explicit factory functions
provided? See
[Swift's Default Initializers](https://docs.swift.org/swift-book/LanguageGuide/Initialization.html#ID213)
or
[the C++ situation](https://stackoverflow.com/questions/4943958/conditions-for-automatic-generation-of-default-copy-move-ctor-and-copy-move-assi).

Note that the default factory function will always used named arguments. If you
want to support initialization without having to specify those names, you would
define a custom factory function with the appropriate signature yourself, as in:

```
struct Point {
  var Float32: x;
  var Float32: y;
  fn operator create(Float32: x_, Float32: y_) -> Ptr(Point) {
    return construct(.x = x_, .y = y_);
  }
  // Could also define a factor function that takes (.x = ..., .y = ...),
  // if desired.
}

var Point: p = (1.0, 2.0);
```

Let's consider the situation where a factory function needs to use the address
of the constructed object. We don't know the address of the constructed `Foo`
until the `construct` function returns, so none of the arguments to that
function can use the address. If we just need to register that address in some
other object, we can do that in the factory function after the `construct` call.

```
struct Foo {
  var Int: a;
  var Ptr(Foo)?: child;
  fn operator create(Int: value, Ptr(Foo): parent) -> Ptr(Foo) {
    // Question: If `this` is not a reserved word, then maybe you would be able to
    // write `this` instead of `result` here.
    var Ptr(Foo): result = construct(.a = value, .child = nil);
    parent->child = result;
    return result;
  }
}
```

If instead some field needs the address of the constructed object to initialize
some field. What if instead we wanted to make a node in a circularly linked list
that when it was first constructed just pointed to itself? In this case we could
either assign a temporary value that we overwrite or use the `uninit` keyword to
delay assigning a value to those fields until after construction:

```
struct Bar {
  var Int: a;
  var Ptr(Bar): next;  // not optional, so no easy value to init with.
  fn operator create(Int: value) -> Ptr(Bar) {
    var Ptr(Bar): result = construct(.a = value, .next = uninit);
    result->next = result;
    return result;
  }
}
```

The `uninit` syntax would be a signal that you needed to be careful -- so you
might want to avoid calling any member functions on the object until that field
is properly set. Using a temporary value would be safer (e.g. no problems with
the object being in an invalid state for calling member functions or dealing
with failures during the factory function), but won't be available for all types
(e.g. non-nullable pointers).

More on this issue in
["Uninitialized variables and lifetime" (TODO)](#broken-links-footnote)<!-- T:Uninitialized variables and lifetime -->
or
[carbon-uninit-v2 (TODO)](#broken-links-footnote)<!-- T:Carbon Uninitialized variables and lifetime v2 -->

**Rejected alternative:** We also considered the Swift model, where the factory
function is given a `this` pointer that is set to allocated-but-uninitialized
memory at the start of the function.

```
struct Bar {  // Swift-style
  var Int: a;
  var Ptr(Bar): next;
  fn operator create(Ptr(Bar): this, Int: value) {
    this->a = value;
    this->next = this;  // Address of created object available at the start.
  }
}
```

Advantages of the Swift model:

- Code is more concise, since you skip the "construct call" step. Languages like
  Dart go to great lengths to provide short ways of writing constructors (see
  e.g.
  [Deconstructing Dart Constructors](https://medium.com/flutter-community/deconstructing-dart-constructors-e3b553f583ef)).
- Easier for the optimizer to avoid copies, since the code is already explicitly
  putting values in their final place.

Disadvantages:

- More reliant on code analysis to verify safety rules like every field is
  initialized before any member functions are called. In the proposed model
  there will typically be no object pointer to call methods on until fields have
  been initialized (and cases where care is needed are visible in the source
  since you need to use the `uninit` keyword).
- Fewer edge cases, particularly when considering
  [inheritance](#appendix-construction-and-inheritance).

Mixed:

- Const data members are supported in the Swift model, but it creates an
  inconsistency where they are allowed to be assigned in initializer functions,
  but not elsewhere. It is also unclear if the compiler should try and enforce
  them only being assigned to once. In the proposed model there is a natural
  place to give const members their value without ever assigning to them (but
  they can't depend on the address the object is given).

**Proposal:** Factory functions will be allowed to fail, with the factory
function's signature indicating whether it returns an optional value or may
raise an exception (see
[Swift's Failable Initializers](https://docs.swift.org/swift-book/LanguageGuide/Initialization.html#ID224)).

Syntax: This should be made consistent with how we mark any other function
returns an error, or make the return type an optional value. [TODO: link to doc
on errors.]

**Proposal:** My current preference is that factory functions don't have their
own names, we just rely on using keyword arguments to distinguish different
overloads (as is done in Swift).

This makes it easier for an intermediary (like a container) to construct
something on your behalf (like C++'s `emplace_back`) by just taking a tuple
representing the arguments to the factory function (and the tuple's names are
used to select the factory function overload with the same keyword arguments).
Note: Chandler disagrees.

**Proposal:** Eventually we will support factory functions that call other
factory functions in place of calling `construct` (like
[C++11 "delegating constructors"](https://en.wikipedia.org/wiki/C++11#Object_construction_improvement)
or
[Swift "convenience initializers"](https://docs.swift.org/swift-book/LanguageGuide/Initialization.html#ID217)).

**Question:** Should we infer template arguments to the struct type based on the
arguments to a factory function?

Goal would be to avoid having to have a separate `make_pair` or `make_tuple`
function that just exists to infer the type parameters. Seems like this would
also be useful for containers when you have some data to initialize with.
Problem is that we would like to be explicit about which arguments are deduced
instead of having them sometimes being deduced and sometimes not, and we won't
always be referring to the type in the context of a factory function with an
argument of the template type. Note that C++17 added this feature:
[https://en.cppreference.com/w/cpp/language/class_template_argument_deduction](https://en.cppreference.com/w/cpp/language/class_template_argument_deduction).
This allows you to write `std::array a = {1, 2, 3};` instead of
`std::array&lt;int, 3> a = {1, 2, 3};`.

For now, my assumption is that this not something we want to support,
particularly not as a first cut.

**Question:** Perhaps Carbon will use specific factory function signatures in
specific situations? Basically, do we need the concepts of copy constructor,
move constructor, default constructor from C++?

We still need to define how you invoke factory functions to construct objects.
The caller, though, doesn't just need to select the factory function arguments,
they also need to decide how the object gets allocated.

#### Control over allocation

**Background:**
[Beef Programming Language: Memory management](https://www.beeflang.org/docs/language-guide/memory/)

**Problem:** There are many use cases that involve changing how allocation is
done, particularly related to strategies for reducing total memory usage or
total time taken, or making those costs more predictable. Use cases:

- thread-specific memory allocator
- [arena allocators](https://en.wikipedia.org/wiki/Region-based_memory_management)
- type-specific memory pool (which can be good for locality, memory budgeting,
  fragmentation; also see
  [Slab allocation](https://en.wikipedia.org/wiki/Slab_allocation))
- an allocator that enforces a particular type's alignment restrictions
- There are a set of use cases around controlling the allocation of stack
  frames, e.g. for coroutines, that is related but in some ways very different.
  (Discuss with [geoffromer](https://github.com/geoffromer) if interested.)

This is in addition to the stack and heap allocation strategies that are
provided by Carbon directly, and the placement allocator that has a number of
uses.

**Problem:** Some types need to control the size of their allocation.

For example a `FixedArray(T)` object may want to store the array of objects in
an additional data region at the end of the object, based on a runtime
`num_elements` argument passed to the constructor (and the size of `T`). In C,
this is no obstacle: you can get any amount of memory from a call to `malloc()`,
but in C++ normal usage of `new` only takes a type, not a size. Of course if you
want to allocate an array of objects, they will all need to be the same size. We
likely also want to restrict stack objects to be a fixed size -- this is
definitely a restriction we need inside stackless coroutines, if we have them.

**Possible solution:** For this case, we could just have a data member with a
special type indicating that it ideally would have a variable size. In this
case, the factory function will be able to query if this the fixed-size case
(e.g. array or stack allocation). The factory function will specify a size to
initialize the special member. In the case that the object needs to be
fixed-size, the special member will always be a pointer to the variable-sized
region. Otherwise that member will itself be allocated the requested size.

**Problem:** Restricting which allocators may be used with a particular type.

For example, a type may want to enforce that it is always on the stack or heap.
Or that its destructor does important work and can't be in an arena. Or using a
type-specific memory pool or an allocator that performs special alignment.

**Question: **Are there other cases where the type wants to control how it is
allocated?

If there are other cases, we may want to have a special (static) member function
structs could implement to customize what allocation is done. It would take an
`Allocator`
[generic interface (TODO)](#broken-links-footnote)<!-- T:Carbon templates and generics --><!-- A:#heading=h.8wggk7rc2ziv -->
argument, along with the constructor arguments, and possibly the type being
instantiated (which may be a descendant of the type where the special member
function is defined).

For example:

```
operator fn allocate_single(
  Allocator: allocator, Type extends Foo:$ Self, ...: init_tuple) -> Self
```

With a default implementation that is something like:

```
{ return allocator.Create(Self, init_tuple); }
```

`Allocator` can be one of the defaults provided by Carbon: `HeapAllocator`,
`StackAllocator`, `PlacementAllocator`; or custom provided by the user:
`ArenaAllocator`, `FreeListAllocator`, ... The `Allocator` interface would have
an API for saying whether you can allocate a different size. We would also need
a story for allocating an array: maybe via a closure (`(Int) -> Tuple`) that can
be used to return a tuple for element `i`? That would be useful for copying, in
addition to explicitly specified initializers.

**Question:** Would a [free list](https://en.wikipedia.org/wiki/Free_list) be an
allocator or something a type would control?

Issue: normally you would only want to use a free list when the alternative is
allocating from the heap. You wouldn't want a type to decide to use a free list
when the user is allocating an object on the stack. This suggests that it should
be under the allocator's control.

**Problem:** How do we handle allocation failures?

To handle failure, we provide two APIs for allocating on the heap: one that
aborts the program if allocation fails that is used by almost all users, and
another that can report an error, but has a more awkward interface. We likely
want this second interface anyway since we want to support constructors that can
fail as well.

**Problem:** How are allocators specified?

One simple case is local variables in a function: they use Carbon's provided
`StackAllocator`. Otherwise, Carbon's `new` operator (or whatever we replace
`new` with) should take anything that conforms to the `Allocator`
[generic interface (TODO)](#broken-links-footnote)<!-- T:Carbon templates and generics --><!-- A:#heading=h.8wggk7rc2ziv -->.
For example, the equivalent of C++ placement new is to pass the
`PlacementAllocator` to `new`. I don't want to suggest this is a good syntax,
but here is how you might write it:

```
var Foo : x = uninit;
new(PlacementAllocator(&x)) Foo(.a = 1, .b = 2);
```

**Question:** Should the `new` operator default to using the default
`HeapAllocator`?

Probably, but this is getting into syntax questions that I don't think we should
get into yet.

**Question:** Should the allocator being used be available to the factory
function?

My main concern is that uses of `StackAllocator` and `PlacementAllocator` might
be pretty restrictive (e.g. they may only have space for this one allocation
available), and so it seems like it may be of too-limited value to be worth
supporting. If you have a container that is going to do allocation, possibly you
should just use an explicit allocator argument to the factory function.

#### Control over destruction / RAII

**Goals/principles:**

- Preference is for predictability -- users should know when destructors are
  executed. This precludes things like concurrent garbage collection determining
  when destructors run. Exception: User's can opt-into garbage collection via
  libraries, such as
  [C++ proposal: An RAII Interface for Deferred Reclamation](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p0561r4.html).
- Predictability should support
  [RAII](https://en.wikipedia.org/wiki/Resource_acquisition_is_initialization)
  programming patterns, so resources beyond just memory can be released when the
  programmer expects.

**Proposal:** We will allow each struct to optionally define a destructor. Every
class has at most one destructor, which takes no arguments and returns nothing.
If not provided, the destructor is equivalent to a function that does nothing.
The destructor will be executed:

- For a stack object, when control exits the scope in which the object was
  declared.
- For a heap object, when the object is explicitly deallocated (in C++, this
  would be the result of a `delete`).
- For an object allocated with a non-default allocator, you will need to
  deallocate using the same allocator.

In the case that the object has a type that uses inheritance, every type in the
inheritance chain may define a destructor. The order of operations will be:

1. The descendant destructor is executed. Calls to member functions are allowed.
2. Destructors for data members defined in the descendant are executed.
3. The parent destructor is executed. Calls to member functions ignore any
   overrides from the descendant.
4. Destructors for data members defined in the parent are executed.

If the object does not use inheritance, only the last two steps will be
executed.

**Problem:** Deleting an object of a derived type via a pointer to the base
class.

In particular, how does the compiler know which destructors to run? In C++, you
may declare a destructor as virtual, which ensures the object stores a vtable,
and that vtable has a pointer to a function that can do all the cleanup needed
based on the runtime/dynamic (vs. compile-time/static) type. This leads many C++
style guides to require all destructors to be virtual.

**Proposal:** Structs are final by default. If explicitly declared as a base
class, then they require a `virtual` destructor or it is an error to delete such
a pointer.

If this proves insufficient with experience, we would consider the following
alternative.

**Rejected alternative:** We distinguish two pointer types:

- Pointer to exactly type `T` (the default).
- Pointer to type `T` or some descendant.

Pointer of the first kind should implicitly convert to the second kind as
needed. It should be a compiler error to delete a pointer of the second kind
unless it defines its destructor as `virtual`. See the
[meeting notes where we discussed this point (TODO)](#broken-links-footnote)<!-- T:Archived Carbon meeting notes --><!-- A:#heading=h.984sgbf7cf1l -->.

We may want some way to generically say "a pointer that can be deleted" for
types like whatever Carbon's equivalent of `unique_ptr` is (`OwnedPtr(T)`?)

Note: the pointer returned by the `construct` call in a factory function (see
[section on creation](#control-over-creation)) will be of the second kind unless
the type is declared `final` (if we support that). The return value of a call to
`new` will be of the first kind, unless you cast that to a pointer to a parent
type, in which case it will have to be the second kind. We may want to support a
version of `dynamic_cast` for going from the second kind to an optional version
of the first kind, that will be `nil` if the pointer is actually dynamically
pointing to a descendant type.

**Proposal:** We don't allow destructors to generate errors.

This is definitely a source of complexity in C++, and is in general discouraged:

- [isocpp.org: Exceptions and Error Handling: How can I handle a destructor that fails?](https://isocpp.org/wiki/faq/exceptions#dtors-shouldnt-throw)
- [Andrzej's C++ blog: Destructors that throw](https://akrzemi1.wordpress.com/2011/09/21/destructors-that-throw/)

One problem is that you could generate an exception while unwinding the stack
for another exception.

Another example, let's say you are destroying a value whose type uses many
levels of inheritance, and some middle destructor fails. The object is now in a
partially destroyed state -- how could you recover?

So that leaves the suggestion: we only allow users to abort the program on error
in a destructor. This would encourage a style where you first instruct the
object to clean up, and deal with any errors then, before destroying the object.
For example, if you wanted to handle errors on closing a file, you would
explicitly call `Close()` on the file object, otherwise errors would either be
ignored or abort the program.

**Question:** Do we have some mechanism to run a destructor early?

On the surface, this seems good: we are giving the programmer control. This
control may be important for some use cases: for example, one obstacle to
performing the tail call optimization is having to run destructors after what
would otherwise be a tail call. Another use case is [destructive move](#moving).
The bad news is how can we know statically when exiting the scope if a
destructor has already executed? It is much simpler if that is actually encoded
in the dynamic state of the object (for example how `unique_ptr` holds a
`nullptr` if it has already freed its contents). The answer here should mirror
what we do for uninitialized values
(["Uninitialized variables and lifetime" (TODO)](#broken-links-footnote)<!-- T:Uninitialized variables and lifetime -->
or
[carbon-uninit-v2 (TODO)](#broken-links-footnote)<!-- T:Carbon Uninitialized variables and lifetime v2 -->)
-- running a destructor early should either:

- prevent the destructor from being run again when the object leaves scope,
  because the compiler can determine statically at every exit point whether it
  is already destroyed; or
- the type provides a way to put itself in a state which is safe to destroy
  again.

Otherwise the compiler should issue an error.

Also: how hard would it be for the compiler to automatically determine when it
was safe to move destructor execution earlier? It wouldn't be good for something
that holds a mutex or is recording how long this block of code takes to execute.

**Side question:** Do we want a special syntax for making a tail call?

Seems good for a performance-oriented language:

- Guarantees that the programmer is getting the performance they expect.
- Could give a compile error if it won't happen, with an explanation of how to
  fix.
- Sets expectations that some calls won't appear in a stack trace.

Note: this feature is also important for coroutines, see
[C++ proposal: Core Coroutines](http://wg21.link/P1063).

#### Reference counting

**Question:** How can we support reference counts, in the cases where you want
them?

**Feel free to skip this section:** This section really isn't about structs, but
some future standard library facility. I was mostly just working through this to
see what issues came up that might be relevant to structs in general. For
example, I've heard people say that being able to override `operator=` is pretty
surprising in C++, but it seems essential if you are to support reference
counting. It does motivate support for "moving" and "borrowing" as part of the
function call protocol.

**Why we care:**
[Reference counting](https://en.wikipedia.org/wiki/Reference_counting) is an
appealing alternative to manual memory management when there isn't a single
clear owner responsible for deallocating an object, since it is deterministic
and does not require extensive or intrusive runtime support. For example, when
one system makes an async cancellable request of another system, both systems
may need the request object to live until they are done with it. The system that
is last to use the request object can depend on whether the request completes or
is cancelled first.

**Approaches:** We need a few ingredients:

- an object whose lifetime is under control of the reference count
- an integer with the reference count; this may need to be an atomic type if the
  object will be referenced from multiple threads. Rust has a strength here --
  you can use a non-atomic reference counting type (`Rc`) safely since the
  compiler will raise an error if it is not safe so only in that situation would
  you use the atomic version (`Arc`). **Concern:** Using an atomic counter
  scales poorly with the number of processor cores.
- a handle or smart pointer type holding the reference; the intent is the value
  of the reference count equals the number of smart pointers referencing the
  object.

There are a few different approaches:

- With a C++ `shared_ptr`, the object is separate from the reference count. This
  can be convenient, since any method of allocating an object that gives you
  pointer can then be managed by a `shared_ptr`, including pointers to
  incomplete types. They also support
  [weak references](https://en.wikipedia.org/wiki/Weak_reference) via
  `weak_ptr`. But it is easy to misuse, if for example the same pointer is given
  to two `shared_ptr` objects to manage, they will both delete it. It also
  [allocates the reference count separately from the object](https://stackoverflow.com/questions/2802953/how-do-shared-pointers-work),
  which adds overhead.
- The reference count (either atomic or not) could be a data member of an object
  (possibly via inheritance or a mix-in). The handle / smart-pointer type would
  require access to that reference count via a
  [generic interface (TODO)](#broken-links-footnote)<!-- T:Carbon templates and generics --><!-- A:#heading=h.8wggk7rc2ziv -->.
- The smart pointer class could also be responsible for creation of the object
  (in addition to destruction), and when it creates the object it creates it as
  part of a struct that also contains the reference count. This is probably the
  hardest to misuse, but reduces flexibility somewhat. This seems more
  convenient than the previous option, and safe, but doesn't naturally support
  inheritance.

I think all three are viable, though I think we might not want to support the
first one in the standard library except as part of the C++ integration. Here is
a sketch of the code for the last option.

```
struct RC(Type:$ T,
          Type implements Allocator:$ AllocatorType = Carbon.HeapAllocator) {
  private struct Counted {
    var int64: count = 1;  // ARC would use Atomic(int64) instead here.
    var T: value;
  };
  private var Ptr(Counted) : p;
  // Hopefully the default heap allocator will have size 0?
  private var AllocatorType : allocator;
  // We may want to provide this type alias by default.
  private var Type:$ Self = RC(T, AllocatorType);

  // Create a new T.
  fn operator create(
      ...: args,
      .allocator = AllocatorType: allocator = Carbon.default_allocator)
      -> Ptr(Self) {
    // We forward the arguments passed to this factory function to T's
    // factory function.
    return construct(
        .p = new(Counted(.value = args), .allocator = allocator),
        .allocator = allocator);
  }

  // Copy constructor increments the reference count.
  // TODO: `in` here means const reference
  fn operator create(in Self : to_copy) -> Ptr(Self) {
    to_copy.p->count += 1;
    return construct(.p = to_copy.p, .allocator = to_copy.allocator);
  }

  private fn dec_ref(Ptr(Self): this) {
    this->p->count -= 1;
    if (this->p->count == 0) {
      delete(this->p, .allocator = this->allocator);
    }
  }

  // Destructor decrements the reference count and deletes if zero.
  fn operator destroy(Ptr(Self): this) {
    this->dec_ref();
  }

  // Access stored `T` object.
  // TODO: some way of marking this as const?
  fn get(Ptr(Self): this) -> Ptr(T) {
    return &(this->p->value);
  }

  // Assignment.
  fn operator=(Ptr(Self): this, in Self: rhs) {
    rhs.p->count += 1;
    this->dec_ref();
    this->p = rhs.p;
    this->allocator = rhs.allocator;
  }

  // Plus smart-pointer operator overloads (like ->).
}
```

Of course we'd like to maximize the performance of reference counts by reducing
how often the counts are updated. There are two common cases:

- "**Moving**": if we are transferring ownership, say because we are returning
  an RC value from a function, we don't need to do a copy, incrementing the
  count, and then a destroy, decrementing the count. We should just have some
  way of marking the object as movable and skip all that unnecessary work. See
  next section.
- "**Borrowing**": if we have a reference to an object using an RC handle, we
  should not need to increment & decrement the reference count just to call a
  function that won't persistently store the reference we provide.

**Proposal:** Functions taking a pointer should, in their signature, indicate if
they save a copy of pointers they are provided so the compiler can verify
whether a borrow is safe ("capture" the pointer). In fact, the default should be
that functions do _not_ capture a pointer unless they (say) use a `capture`
keyword by the pointer type in the function signature (this would be most common
in factory functions).

This would be helpful for the reference counted case. It would also allow the
compiler to know that passing a pointer to an object on the stack is safe. And
it seems useful to the optimizer's alias analysis and the sanitizer.

**Question:** Would this just be to generate faster code, or is there some way
we can use this to improve safety to statically detect errors we can't diagnose
in C++?

Extension: Maybe a return type of "borrowed pointer" means "returned pointer
must not outlive this object", which means, among other things, safe to pass to
a function that takes a "borrowed pointer" type. In general we should consider
the question of whether we want to be able to express "has lifetime of object
x", or if just being able to express "scoped to this function call" is enough.

#### Moving

TODO

Chandler says:

- What moves are in C++: transfer of resources without copy.

- What C++ moves are not, but people want: transfer of lifetime and location.

Context for destructive moves:
[About Move](https://sean-parent.stlab.cc/2014/05/30/about-move.html),
[Google search [c++ destructive move]](https://www.google.com/search?q=c%2B%2B+destructive+move)

Discussion about moves in Carbon ending the variable's lifetime: TODO

Discussion about uninitialized values also affecting lifetime:
[Carbon Uninitialized variables and lifetime v2 (TODO)](#broken-links-footnote)<!-- T:Carbon Uninitialized variables and lifetime v2 -->

#### Copying

TODO

In general, the default assignment for a type should be a combination of
destruction and copying, with some checking to make sure self-assignment is
safe.

[geoffromer](https://github.com/geoffromer) says:

    I certainly agree that these are the right semantics, but it's not clear to me how much the language can do here, since destroy-and-reconstruct is likely to be somewhat inefficient compared to e.g. recursive assignment.


    It might be interesting to explore making this an unenforced language-level requirement, so that the language and libraries can optimize on the basis of this equivalence.

### Other special member functions / operators

**General principle:** User-defined types have no implicitly-defined operations
whatsoever. We should make it as easy as possible to _explicitly_ ask for
defaulted operations, but every one of the type's operations should have some
visible trace in the type definition.

**Proposal:** The way for a type to overload a Carbon operator is to
[implement an interface](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/designs/generics-interface-type-types.md#implementing-interfaces).
This has a number of benefits:

- For binary operations, the implementation can be defined with either the left
  or right type.
- We can benefit from the existing interface mechanisms for providing standard /
  common implementations.
- [Supports defining many operators by implementing one or a few operations.](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/designs/generics-interface-type-types.md#example-defining-an-impl-for-use-by-other-types)
  This is both more convenient and provides greater consistency. Examples:

  - `!=` should always be the opposite of `==`
  - all comparison operations could be derived from `&lt;=>`
  - perhaps assignment (`=`) could be derived from a destructor and copy
    construction

  However, these operations may be individually implemented if desired. For
  example, you might be able to code a cheaper implementation of `==` than
  `&lt;=>`.

- Conversely, there are consistency requirements between operations. For
  example, any type implementing `==` must also implement `!=`. Similarly for
  `&lt;` vs. `>`. This is accomplished via the specific choices of what goes
  into the operator-overloading interfaces Carbon provides.
- Can implement different interfaces to convey different semantics.
  - Whether operations are associative or commutative.
  - Distinguish between strong vs. weak. vs. partial ordering.
- It is convenient for parameterized functions and types if constraints they
  want to express like "supports comparisons" (e.g. for a sort function or
  sorted container) are already reflected in what interfaces a type has
  implemented.

#### Equality

**Proposal:** We have a `Carbon.EqualComparableTo(T)` that types can implement.
It defines two API endpoints `Equal` and `NotEqual` taking `Self` and `T` and
returning a `Bool`. There is also a `Carbon.EqualComparable` interface that is
equivalent to `Carbon.EqualComparableTo(Self)`. The Carbon standard library will
provide an implementation of `Carbon.EqualComparable` that structs can use if
all data members themselves implement `Carbon.EqualComparable` that types
[may opt into using](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/designs/generics-interface-type-types.md#example-defining-an-impl-for-use-by-other-types).

There should be a straightforward way to implement `Carbon.EqualComparableTo(T)`
for `U` and `Carbon.EqualComparableTo(U)` for `T` at the same time via a single
comparison function. It would be nice if there were some way to enforce
consistency here.

#### Ordering

**Proposal:** Define the ordering operators (`&lt;`, `&lt;=`, `>=`, `>`) and the
(spaceship) `&lt;=>` from
[C++20](https://en.cppreference.com/w/cpp/language/default_comparisons) via
comparison interfaces. In general we would expect to implement those interfaces
using a single comparison function that matches the semantics of `&lt;=>`. The
idea is that `a &lt; b` is defined to be `(a &lt;=> b) &lt; 0`.

There are different possible ordering relations:

- 0 only when actually equal (C++ calls this "Strong ordering"). Example:
  lexicographical ordering of a tuple of numerical types.
- (a &lt;=> b) == 0 defines an equivalence class (C++ calls this "Weak
  ordering"). Example: case-insensitive string comparison.
- (a &lt;=> b) == 0 means "unrelated". (C++ calls this "Partial ordering").
  Example: subtyping relation; if you have three types: `Child &lt; Parent`, but
  neither is related to `Other`. From this you can see "not related" is not
  transitive.

**Proposal:** We distinguish between these cases by providing different
interfaces, with a subtyping relationship between them. If you implement the
`Carbon.StrongOrdering` or `Carbon.WeakOrdering` you get a definition for the
equality operators as well (`==` and `!=`). Further, `Carbon.StrongOrdering`
will
[extend](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/designs/generics-interface-type-types.md#interface-extension-optional-feature)
`Carbon.WeakOrdering` (and similarly with `Carbon.PartialOrdering`) so values of
types implementing `Carbon.StrongOrdering` may be passed to functions expecting
values of types implementing `Carbon.WeakOrdering`.

**Alternative considered:** C++ distinguishes these cases by having three
different return types from `operator&lt;=>`.

**Question:** C++ actually defines the equality operators even in the last case
by having two different return values for "equivalent" vs. "unrelated". Do we
want that?

Example: floating-point numbers have
[NaN](https://en.wikipedia.org/wiki/NaN#Comparison_with_NaN) which are
"unrelated" to other floats (and to _itself_, which is a bit unusual from a
mathematical perspective), but also has values that are "equivalent", such as
[signed zero](https://en.wikipedia.org/wiki/Signed_zero#Comparisons) (`0` and
`-0` should compare equal according to the IEEE standard).

It seems important that Carbon represents IEEE semantics with as much fidelity
as possible, which suggests that we do need to make this distinction. One
additional wrinkle here is that IEEE actually defines two ways of comparing
floats: the
[Comparison predicates](https://en.wikipedia.org/wiki/IEEE_754#Comparison_predicates)
and the
[Total-ordering predicate](https://en.wikipedia.org/wiki/IEEE_754#Total-ordering_predicate).

**Proposal:** There are standard library implementations of the comparison
interfaces that do lexicographical comparison of the members in a specified
order.

**Question:** In C++ it can infer the return type (and therefore whether it is a
strong, weak, or partial ordering) as the strongest common return type
supported. Do we want users to have to say which ordering interface is
implemented, or do we want to allow that to be inferred?

**Question:** C++ also has two different return types for defining equality via
`&lt;=>` without defining an ordering. Do we want that? This seems mainly useful
as part of C++'s story for inferring the comparison supported by the struct from
the components.

#### Properties

**Goals/principles:**

- Allow evolution of how a struct is represented and implemented without having
  to modify all callers (see the discussion of properties in
  [Uniform access principle](https://en.wikipedia.org/wiki/Uniform_access_principle)).
- Avoid users having to write a lot of boilerplate getters & setters to
  preemptively support future evolution, which may never happen. This results in
  style guides and coding practices that make new types very expensive the
  amount of code to be written and read.

**Context:**

- "Computed properties" (as opposed to "stored properties") in Swift:
  [https://docs.swift.org/swift-book/LanguageGuide/Properties.html#ID259](https://docs.swift.org/swift-book/LanguageGuide/Properties.html#ID259)
- [https://belkadan.com/blog/2009/03/Const-Correctness/](https://belkadan.com/blog/2009/03/Const-Correctness/)
- [https://www.beeflang.org/docs/language-guide/datatypes/members/#properties](https://www.beeflang.org/docs/language-guide/datatypes/members/#properties)

**Concern:**
[The Google C++ style guide](https://google.github.io/styleguide/cppguide.html#Structs_vs._Classes)
recommends distinguishing between "structs" that have no private data members
and no invariants and "classes" with no public data members and may have
invariants, even when C++ itself does not make that distinction. In that world,
Google has found evolution from struct to class is quite rare, and so that need
for properties is not strong. Furthermore, when writing classes, getters and
setters are a [code smell](https://en.wikipedia.org/wiki/Code_smell) indicating
the API is leaking its implementation details rather than providing a useful
abstraction boundary.

**Concern:** Properties introduce both a complexity cost in the language, and
also a readability cost, where you are not sure what happens when you access a
member of a type without looking up the definition of that type. This is a large
enough concern that languages like [Zig](https://ziglang.org/) have put "no
hidden control flow" as
[the first highlight on its homepage](https://ziglang.org/#Small-simple-language),
citing a lack of properties as an important feature.

**Problem:** A single function that returns a value is probably enough for read
use cases, but what about all the different ways to mutate (assignment, +=,
moving)?

**Proposal:** At first we only support read-only properties. This is the more
common case, and easier to support. They can be a replacement for `public_read`
data members (see [the visibility section below](#visibility)).

```
struct Foo {
  private var Int: x = 2;
  fn operator .positive(Ptr(const Self): this) -> bool {
    return this->x > 0;
  }
}

var Foo: f;
assert(f.positive);
```

**Future:** The next step would be to support a property with a "setter".
Operations like `a.b += c`, where `b` is a property with both a getter and a
setter, would be rewritten to be `a.b = a.b + c`. Another approach would be to
define some sort of proxy object (maybe via a template type in the standard
library) that could be returned by a property function that would have all the
overloaded operators. Swift has recently started experimenting with supporting
"modify" operations on computed properties, see:

- [Generalized accessors](https://github.com/apple/swift/blob/master/docs/OwnershipManifesto.md#generalized-accessors)
  in the Swift Ownership Manifesto
- [Implement generalized accessors using yield-once coroutines](https://github.com/apple/swift/pull/18156)
- Use in Swift's dictionary subscript operator
  [https://github.com/apple/swift/blob/master/stdlib/public/core/Dictionary.swift#L893](https://github.com/apple/swift/blob/master/stdlib/public/core/Dictionary.swift#L893)

#### Other operator overloading

We will need to define operator overloads for the standard Carbon operations.
Use cases include:

- Smart pointers
- Numerical types like "Matrix"
- Associative container types

This generally seems straightforward, but may require us to define some
protocols. For example, in C++ you could imagine the return type of a
dereference operator (`*p`) being a reference type, and we would need some sort
of alternative in Carbon which (currently) doesn't have references.

In practice it is valuable from a performance perspective to have an efficient
story for updating entries in a hash map. For example, just being able to write
`foo[key] += 1;` and have it work even for keys not already in the hash map is a
big win (using a default of 0), especially if it means not having to do two
lookups. I saw this as one big part of the performance difference in
[straightforward Swift code](https://benchmarksgame-team.pages.debian.net/benchmarksgame/program/knucleotide-swift-2.html)
vs.
[C++ code](https://benchmarksgame-team.pages.debian.net/benchmarksgame/program/knucleotide-gpp-2.html)
in
[one benchmark](https://benchmarksgame-team.pages.debian.net/benchmarksgame/performance/knucleotide.html)
in the Computer language benchmarks game.

**Concerns:**

- Not necessary for this update operations to be spelled with indexing syntax to
  get the benefit.
- By making the update happen via the indexing syntax, it makes the default to
  use when the key is not available hard to express.
- The fact that this leads to C++ automatically inserting entries into the hash
  map when key lookup fails is frequently unfortunate, and prevents using the
  index syntax on const objects.

One possible approach would be to have `foo[key]` return a proxy type that could
be converted into an `Optional(ValueType)` if used as an RValue, but also
supported the assignment operators of `ValueType`. This would be a bit magical,
and alternative suggestions would be welcome. In particular, in C++ it doesn't
seem possible to define such a proxy type, and efforts to extend C++ to support
it have been problematic.

### Const Storage

**Question:** Do we want to have a modifier / alternative to `var` to indicate
"const" storage (not modified after initialization, in contrast with
[type constants described above](#constants))? If we want to support it, I think
it should be pretty straightforward, since all user construction code happens
before the object is actually instantiated and const values would be frozen. One
stumbling block is having any const member prevents moving or copying the value,
making it hard to give value semantics for that type.

It is at least vital that the distinction between reads and writes be visible at
the language level, because that's the foundation of safely sharing data across
threads.

We really need a separate document focused solely on how Carbon should approach
const. As a brief summary, there has been some discussion about how `const` has
caused grief within C++:

- Const in C++ causes some functions to be implemented in const and non-const
  versions, or would have to become generic in order to preserve constness.
- Const in C++ introduces incompatibility between things that should logically
  be compatible: like a function that doesn't modify its argument should be able
  to be passed a `const vector&lt;Foo>&` or a `const vector&lt;const Foo>&`.

See some discussion here:
[https://nim-lang.org/araq/writetracking.html](https://nim-lang.org/araq/writetracking.html),
which talks about the alternative of explicitly modeling what is mutated by a
function. This is mostly an argument about how functions should be annotated --
we still may want a concept of "const" for variables.

**Note:** in the thought-experiment construction model, there would be no way to
represent a const member that depended on the address of the object.

Contrast this with the `public_read` modifier as described in
[the visibility section below](#visibility).

### QUESTION: Static storage?

C++ provides a way to declare that data members of struct or class have "static"
storage, which are variables that have storage and scope associated with the
type rather than the instance, but are (in contrast with
[type constants](#constants)) mutable. It is unclear if we want to support this
in Carbon. We could easily have some syntax marking variables as having static
storage, but (a) we would need define how initialization and destruction works
(a large source of complexity in C++, see
["Static and Global Variables" in the Google C++ style guide](https://google.github.io/styleguide/cppguide.html#Static_and_Global_Variables))
and (b) it conflicts with goals expressed elsewhere such as:

- [Carbon variables](https://github.com/jonmeow/carbon-lang/blob/proposal-design-overview/docs/design/variables.md):
  "no global variables."
- [Carbon Sanitizing, Fuzzing, Hardening (TODO)](#broken-links-footnote)<!-- T:Carbon Sanitizing, Fuzzing, Hardening -->
  /
  [Global variables (TODO)](#broken-links-footnote)<!-- T:Carbon Sanitizing, Fuzzing, Hardening --><!-- A:#heading=h.d6g0qwy9uyo0 -->:
  "The current expectation is that Carbon will not have mutable global
  variables."

### QUESTION: Extension / Inheritance?

What sort of extension / inheritance mechanisms do we want to support in Carbon?
C++ inheritance has two aspects: implementation reuse and subtyping. Subtyping
means that a subclass implements the interface of its superclass and so the
subclass can be passed to functions expecting the superclass. However, you can
achieve this flexibility using Carbon interfaces, either
[at compile time (TODO)](#broken-links-footnote)<!-- T:Carbon templates and generics --><!-- A:#heading=h.8wggk7rc2ziv -->
or
[at runtime (TODO)](#broken-links-footnote)<!-- T:Carbon templates and generics --><!-- A:#heading=h.nt307l3mhsng -->.

- Rust does not support inheritance, and instead uses interfaces exclusively (I
  think they are called
  "[traits](https://doc.rust-lang.org/1.7.0/book/traits.html)" and
  "[trait objects](https://doc.rust-lang.org/1.7.0/book/trait-objects.html)" in
  Rust).
- Single inheritance is commonly used in C++ and dropping it entirely may make
  for a rough transition for C++ users and codebases.
- Multiple inheritance is supported by C++, but comes with a performance and
  complexity cost and so is used much less frequently. Choices here include
  [virtual inheritance](https://en.wikipedia.org/wiki/Virtual_inheritance), and
  Python's super facility (which I think is a failed experiment, see
  [https://fuhm.net/super-harmful/](https://fuhm.net/super-harmful/)).
- There are also
  [Swift Extensions](https://docs.swift.org/swift-book/LanguageGuide/Extensions.html),
  but I expect Carbon interfaces provide enough here.

**Proposal:** Single inheritance.

The situation as I see it: supporting just interfaces and not inheritance is
significantly simpler from a language perspective. In particular, it makes most
object construction issues disappear. However, this would be a significantly
more opinionated choice, and likely in conflict with Carbon's goal of being an
easy transition from C++. I do not think we should support multiple inheritance,
our interface story is good enough and multiple implementation inheritance is
already strongly discouraged within Google, see
[http://"Google C++ Style Guide"#Inheritance](https://google.github.io/styleguide/cppguide.html#Inheritance).

By single inheritance we mean there is a single parent class for which there is
a subtyping relationship. This means avoiding issues like:

- needing thunks for adjusting the `this` pointer when calling methods,
- having to change addresses when casting, such as with covariant return types,
  and
- a general host of other issues and complexities that come with multiple
  inheritance.

We only want to support at most one vtable pointer, and a tree subtyping
relationship. Casting a pointer to a different type (but one still compatible
with the runtime value) should not change the value of that pointer. For cases
where you might want to use multiple parents, we will support
[mixins instead, see below](#mixins).

The model is:

- A descendant may specify a single parent type.
- The fields / data members / vars of the struct are the concatenation of the
  vars of the parent with those listed in the descendant's definition.
- Parent names are in scope for the descendant, and so no name collisions are
  allowed, with the exception of functions marked `virtual` in the parent.
  **Question:** Could we rename `virtual` into something more meaningful like
  `overridable`?
- Functions marked `virtual` in the parent may be implemented in the descendant
  by using the `override` keyword. Functions with `override` are also `virtual`
  and may be overridden again by further descendants. The keyword `final` is
  like `override` but prevents descendants from overriding.
- Overridden functions in the descendant may take more general (contravariant)
  inputs and return more specific (covariant) outputs.
- Functions marked `required` are `virtual` but have no implementation
  (equivalent to C++'s "pure virtual"). Types with required members that have
  not been overridden can not be instantiated. **Question: **is this the only
  way to mark a type as abstract (as in can't be instantiated)? C++ has a weird
  corner case where you sometimes want to prevent instantiation of a class but
  it has no member functions that would naturally be marked pure virtual, and so
  you mark the destructor as pure virtual even though it still has to be
  implemented. It may be better to just support a syntax like
  `struct Foo abstract { ... }`, though some on Carbon team disagree this rare
  case merits its own language feature.
- To delete a value of a derived type via a pointer to a base type, the
  destructor must be marked virtual in the base type (see
  [the "Control over destruction" section above](#control-over-destruction--raii)).
- We will likely support some rules for inheriting factory
  functions/initializers from parent types, see
  [the "Control over creation" section above](#control-over-creation).
- Structs may implement any number of interfaces. See
  [the "Implementing interfaces" section of the generics design docs](https://github.com/josh11b/carbon-lang/blob/generics-docs/docs/designs/generics-interface-type-types.md#implementing-interfaces).

**Rejected alternative:** We distinguish two pointer types:

- Pointer to exactly type `T`.
- Pointer to type `T` or some descendant.

See the [destruction section above](#control-over-destruction--raii) and the
[meeting notes where we discussed this point (TODO)](#broken-links-footnote)<!-- T:Archived Carbon meeting notes --><!-- A:#heading=h.984sgbf7cf1l -->.

**Current experiment:** there is only one kind of pointer, but we will instead
say that most types can't be extended.

**Proposal:** By default, types are "final" and you have to mark types as `base`
to allow structs to descend from them.

```
struct A base { ... }
struct B base extends A { ... }
struct C { ... }
struct D extends C { ... }  // Compile error: C not a base type.
struct E extends B { ... }
struct F extends E { ... }  // Compile error: E not a base type.
```

**Rejected alternative:** Allow users to mark types as `final` to indicate that
you may not descend from them.

```
struct Foo { ... }
struct Bar extends Foo final { ... }
struct Baz final { ... }
struct Quux extends Baz { ... }  // Compile error: Baz is final.
```

**Rationale:** We want to favor coding patterns that don't use inheritance, and
we think C++ chooses the wrong default here.

Knowing a type is final would allow us to avoid the indirection of virtual
calls, including for the destructor. It would also allow us to pass `this` in by
value, which actually could just be a consequence of changing the `Self` type
from `extends T` to `exactly T`. Also some types are not designed to be
inherited from, and right now that is hard to express in C++.

**Question:** Do we want to support C++'s `private` or `protected` inheritance,
or just `public`?

Public inheritance is the common case, and we maybe address the `private` /
`protected` uses via other mechanisms. Seems like we could add these other forms
later as the need is demonstrated.

**Question:** Do we want a descendant type to be able to replace the definition
of a constant defined in the parent?

It would be simpler to say no, but that may mean that we want to define special
"`Self`" and "`Super`" types automatically, that do vary. The `Self` type
changes covariantly, that is it is always a refinement (more specific) in the
descendant of its value in the parent.

**Question:** Can a descendant type override the default value used to
initialize data members?

My best guess is that we would only allow this via writing code in the factory
function, not via any explicit syntax.

### Mixins

Context:
[https://en.wikipedia.org/wiki/Mixin](https://en.wikipedia.org/wiki/Mixin)

**Goal:** Support implementation reuse in structs without the baggage of
subtyping that you get from inheritance. In particular using multiple mixins
should not introduce any performance overhead, unlike multiple inheritance.

The basic idea is that you can define some data fields and functions operating
on those fields, and any struct ("destination") that wants those can easily add
them.

Things get more exciting if the mixin wants to access data fields or functions
that it doesn't define, and instead requires are defined in the destination
struct being added to. It seems like we might reuse the `required` keyword
(mentioned [above](#bookmark=id.a13yodpnvzyd)) to define functions and their
signatures that must be present.

**Background:** We might want to adopt the concept of a "protocol" that the
destination has to "match" according to:
[On Subtyping and Matching by Abadi and Cardelli](http://lucacardelli.name/Papers/PrimObjMatching.pdf).

I would expect mixins that define data fields would frequently need to define
factory functions for initializing them. Factory functions for the destination
type would have to specify the arguments to one of the mixin's factory functions
to complete initialization.

One implementation strategy is that mixin functions have a single instantiation,
and that instantiation gets passed some destination-specific constants that
specify the offsets to data members and anything else needed. I'm not sure
though how to make the mixin agnostic to whether the function is virtual or not.

Another strategy is the mixin is generic or templated in the destination struct
type -- which seems simpler to me, but could increase compile time and binary
size. In either implementation strategy, I suspect the mixin definition will
need to be visible at the point the destination type is defined.

**Background:** In Dart, the mixin defines an interface that the destination
type ends up implementing, which restores a form of subtyping. See
[Dart: What are mixins?](https://medium.com/flutter-community/dart-what-are-mixins-3a72344011f3).

TODO: propose syntax.

### Templating

See
[Carbon templates and generics / Template types (TODO)](#broken-links-footnote)<!-- T:Carbon templates and generics --><!-- A:#heading=h.i92flm67b33b -->.

### Visibility

**Proposal:** We should support `public`, `private`, and likely `protected` as
in C++. The syntax will be a little different, where the visibility keywords are
modifiers to individual declarations, instead of labels affecting everything
afterwards:

```
struct AdvancedWidget {
  private var Int: x;
  private var Int: y;
  protected virtual fn DoSomething(AdvancedWidget: this);
}
```

This specific syntactic choice was chosen for syntactic uniformity and ease of
tooling.

**Alternative considered:** Separate public, protected, and private sections, as
is done in C++.

This seems tied to the fundamental question of whether we intend the source code
itself to be the primary API documentation (favoring this alternative and
grouping the public API together), or the output of a documentation generator
(like JavaDoc). Right now we are aiming for first class reference document
generation that is ideal for API consumers and so the source code itself should
be optimized for people trying to understand/modify the implementation (ideally
with help from IDEs such as expand/collapse, etc.).

Another factor here is that declaration order matters in Carbon, unlike C++,
which may make it harder to group the declarations by their visibility. This
comes up for fields (memory layout and the calling convention) and for defaults
initializers & methods (since we allow only backwards not forward references
unless you forward declare your function).

**Question:** What is the scope of things that should be allowed to access
`private` members? Anything defined in the same object? same type? same file?
same library?

In my opinion, the important thing is that the code with access to `private`
members is bounded. This allows you to audit the bounded area and determine if
uses of those members satisfy any constraints you might want to verify. We don't
want to create roadblocks, though, to accessing private members in tests or in
parts of the code that are conceptually part of the implementation of that type.

**Concern:** Some might argue that tests should test a stable, documented
contract, and in practice those are usually (but not always) public. Perhaps we
should not make it too easy to test private members to discourage bad test
practices?

**Counterargument:** Perhaps that is being too opinionated about test practices?
For example, you might extract the "algorithm" part of a public API into a
private method just so you can test that part without the side effects or calls
into other components that the full implementation of that API might entail.

Really testing is a large topic which deserves its own document (what about
testing code snippets in your doc strings? what about intrusive tests that are
intermixed with your implementation code, documenting what values intermediate
variables take on for one example input?).

**Proposal:** I think we should also support marking data members as
`public_read`. This makes reading the variable part of the public API, but
unlike `const` members they may be written by member functions.

```
struct Foo {
  public_read var Int: x = 3;
  fn AddOne(Ptr(Foo): this) {
    // May mutate this->x inside member functions.
    this->x += 1;
  }
}

var Foo: y;
// &(y.x) is of type Ptr(const Int)
assert(y.x == 3);
y.AddOne();
assert(y.x == 4);
// Would be a compile error: y.x += 1;
```

This would allow for a progression from plain data struct, to one where only
member functions mutate the (`public_read`) state variables, to one where the
data members are actually [properties](#properties) (and are therefore computed
instead of stored).

### Appendix: Construction and Inheritance

**Question:** What sort of interaction should be allowed between the
constructors when using inheritance? For example, let's look at how C++ and
Swift handle construction:

C++

1. Descendant calls Base class constructor.
2. Base class initializes base data members.
3. Base class constructor code executes. Any calls to virtual member functions
   get the base class versions.
4. Descendant class initializes descendant data members.
5. Descendant constructor code executes. Can modify any data member or call any
   member function.

Swift

1. Descendant class initialized descendant data members.
2. Descendant calls Base class constructor.
3. Base class initializes base data members.
4. Base class constructor code executes. Any calls to virtual member functions
   get the overridden versions.
5. Descendant constructor code executes. Can modify any data member or call any
   member function.

This affects what is available when initializing any particular value. For
example, in C++ you can use the values computed in the base class constructor
when providing initial values to descendant data members, but in Swift you can
call overridden virtual functions in the base class constructor. In Swift, you
might be able to modify the values of the descendant data members after the base
class constructor completes, but there is a danger that those values may have
been needed during the base class constructor as part of an overridden method
call. The C++ approach also has the downside that the vtable pointer may need to
be written once per layer in the class hierarchy, instead of just once.

**Alternative considered:** I ([josh11b](https://github.com/josh11b)) was in
favor of the Swift construction order since it avoids C++'s pitfall w.r.t.
calling virtual functions in the base class constructor, and I have wanted to
get the descendant's version of a virtual function in a base class constructor
before. However, this may not be viable since we want to allow inheritance
across the C++/Carbon boundary. This option is considered first.

**Proposal:** Chandler prefers the C++ constructor order, but forbidding calls
to virtual member functions in constructors to avoid confusion. In particular,
he thinks it is common to want to access base members in the descendant.
[This option is considered in more detail after the Swift model below](#alternative-proposal-requested-by-chandler-c-constructor-order).

**Background:**
[https://www.beeflang.org/docs/language-guide/datatypes/initialization/#initialization](https://www.beeflang.org/docs/language-guide/datatypes/initialization/#initialization)

**Swift-construction-order thought experiment:** What would a Rust-style
construction protocol [described above](#control-over-creation) look like if it
were changed to support inheritance?

NOTE: This section is quite long and you can skip to
[the "preliminary conclusion" below](#preliminary-conclusion) unless you want to
see my thought process for getting there.

Recall that this model uses user-defined "factory functions" (equivalent to
Swift's "initializers") that call the language-provided `construct`.

We somehow want to coordinate between each layer's factory function to divide
everything up into two phases -- before the `construct` function is called and
after, something like:

- Descendant factory function is called by the user.
- Descendant factory function determines (a) the parent factory function to call
  and (b) the initial values for all variables added in the descendant.
- Parent factory function determines initial values for all variables in the
  parent.
- Parent factory function says it is time to invoke `construct`, but we somehow
  arrange to construct the descendant type instead, with the concatenation of
  the parent and descendant initial values.
- The `construct` function returns a pointer to the parent class to the parent's
  factory function, but it actually is a pointer to the descendant class (which
  is okay due to subtyping). The parent's factory function does any
  modifications to the resulting object, which can use the final address and
  call any methods. If those methods are overridden in the descendant, the
  descendant's methods are used.
- The `construct` call in the descendant factory function returns the same
  object, but this time as a pointer to the descendant type. Descendant's
  factory function can make any modifications, including doing anything that
  depends on knowing the results of running the parent's factory function.

Ideally, the optimizer would automatically rewrite the code to match what is
written in the Swift case:

- Allocator allocates memory for the size of the object.
- As the descendant and parent types compute values that are eventually passed
  to the `construct` function, the optimizer arranges to actually store those
  values in their final location instead.
- By the point in the code where `construct` is actually called, there is
  nothing left to do except return the pointer to the (now fully initialized)
  new object.
- Code after the `construct` function proceeds as normal.

Strawman syntax:

```
struct Bar {  // Same `Bar` as from "Control over creation"
  ...
  fn operator create(Int: value) -> Ptr(Bar) { ... }
}

struct Baz extends Bar {  // Baz descends from Bar
  var Ptr(Baz): prev;
  fn operator create(Int: value) -> Ptr(Baz) {
    // Somehow need to specify both the initial values for data members defined in
    // the descendant and the args to the factory function in the parent class.
    var Ptr(Baz): result = construct(.parent = (value), .prev = uninit);
    result->prev = result;
    return result;
  }
}
```

Once we've made all these changes to the Rust model, the
Swift-construction-order-thought-experiment model ends up quite similar (in
spirit) to the Swift model. For reference, here is the same code following a
Swift model:

```
struct Bar {  // See Swift-style version above.
  ...
  fn operator create(Ptr(Bar): this, Int: value) { ... }
}

struct Baz extends Bar {  // Baz descends from Bar
  var Ptr(Baz): prev;
  fn operator create(Ptr(Baz): this, Int: value) {
    this->prev = prev;
    // Phase 1 done once we've assigned all descendant fields and call the parent's
    // factory function.
    Bar.operator create(this, value);  // or super.init() or whatever
  }
}
```

There are a few, relatively minor, differences:

- One difference is that in the Swift-construction-order-thought-experiment
  model even the base class has to call the `construct` function.
- In the Swift model there are a number of safety rules about which values must
  be initialized and which must not be in phase 1. In the
  Swift-construction-order-thought-experiment model this is enforced implicitly
  by not having access to the object at all in phase 1. This also naturally
  prevents calls to member functions in phase 1 in the
  Swift-construction-order-thought-experiment model.
- There is no copying of values through the `construct` call in the Swift model.
  In the Swift-construction-order-thought-experiment model, it would be an
  optimization to avoid that copy.
- Both cases support initializer/factory functions that can fail: either by
  returning an optional type or raising an exception. However, the clean up
  story for the Swift-construction-order-thought-experiment model is simpler --
  more on that below.

Weighing the options: from a syntax perspective, the Swift model is more concise
and direct, but the failure story isn't as nice and generally has more edge
cases and rules that only apply to factory functions. As a result, I'm generally
leaning toward the Swift-construction-order-thought-experiment model. If we
decide to forgo inheritance support in Carbon, I would support that model more
strongly. If we go the Swift route, we'd have to do a bunch of work on the
compiler side to statically verify the code written by users satisfies the
safety rules -- which might be a bit fragile in the presence of control flow,
pointers to members, etc. The only such analysis needed in the
Swift-construction-order-thought-experiment case is to verify that there is a
single call to `construct()` or another factory function.

#### Failure handling

Consider the situation where there is an inheritance hierarchy of three classes:
`D2` extends `D1` extends `B`, and there is a failure generated in `D1`'s
factory function. There are two cases, either the failure occurs before ("phase
1") or after ("phase 2") the call to `construct`.

If the failure happens in phase 1, in the
Swift-construction-order-thought-experiment models the recovery story is
straightforward:

1. `D2` phase 1 runs, determining initial values for `D2` variables (which the
   optimizer may be storing in object being constructed as an optimization but
   conceptually are just local variables) and calls
   `super.init`/`construct`/whatever.
2. `D1` phase 1 runs, creating some local variables and then returning an error.
   The only thing created in `D1` were local variables, which are destroyed as
   normal as part of returning the error.
3. `D2`'s factory function receives an error instead of an object from
   `construct`. It presumably just passes it on, possibly after running some
   clean up code, running any destructors for local variables as normal for any
   return.
4. The object is never fully constructed, and in fact in the
   Swift-construction-order-thought-experiment model was never a named entity in
   any executed code. It gets deallocated using the allocator object.

The story in the Swift model is almost the same, except that values were
actually set in some of the fields in the object being created, according to the
code the user wrote. We would presumably still run the same set of destructors,
just now it also includes the fields of the object that have been assigned to at
that point in the code (a bit more awkward, since that may be hard to know
statically, but we have to solve that problem anyway for the Swift model). The
main other wrinkle is that the address of the object being constructed was
exposed to the user's code in the Swift model, and so it could have been stashed
somewhere -- it would be nice to prevent that (or hope that the user clean up
all such references in the error handling code).

Now consider a failure in phase 2, Swift-construction-order-thought-experiment
model:

1. `D2`, `D1`, and `B` phase 1 runs, determining initial values (or `uninit`)
   for all data members.
2. Object is constructed, using the specified initial values.
3. `B` phase 2 runs successfully, perhaps modifying the object, changing other
   objects to point to this object, or other side effects. Local variables in
   `B` are destroyed, but in many cases those values were already moved into the
   object being constructed.
4. `D1` phase 2 runs partially and then fails, returning an error. Local
   variables are destroyed, but again many values were moved to the object being
   constructed by this point.
5. Optional (see below): run destructor for `D1`
6. Run destructors for any data members of `D1` that were initialized.
7. Run the destructor for `B`.
8. Run destructors for all data members of `B`.
9. `D2` phase 2 receives the error instead of an object from the `construct`
   call. From its perspective this is identical to the `D1` phase 1 failure case
   described above, except that the initial values are no longer local variables
   they've been moved into the object. It may run some clean up code, and
   destroy any local variables as before.

**Aside:** Steps 5-8 could possibly be moved to the end (after `D2` phase 2),
but I think the generated code would be simpler this way, and there is not much
other difference.

**Commentary:** One way to look at this process is as follows:

- Before phase1 finishes, there is no object, only local variables. The rules,
  as you say above, are simple.
- Once phase1 finishes, for any subobject whose phase2 has **finished**, we run
  the destructor for that subobject. For any members of a subobject whose phase2
  has not finished, we run the destructors for the members of the subobject.
- In this example, that would result in the B destructor running and the
  destructor for each member in D1 and D2 running.

**Requirement:** We need to run the destructor for every member variable that
was initialized. This means it needs to be a _compile error_ if the compiler
can't figure out what is initialized at the point an error is returned (and this
would affect the generated code, since the values have nontrivial destructors),
e.g.:

```
struct Foo {
  var Bar : a;
  var Bar : b;
  fn operator create(Boolean : c) -> Ptr(Foo) throws {
    var Ptr(Foo): result = construct(.a = uninit, .b = uninit);
    if (c) {
      result->a = Bar(1);
    } else {
      result->b = Bar(2);
    }
    try OperationThatCanFail();
    // Error: We can't know if result->a or result->b are initialized, so the
    // compiler doesn't know what destructors to run.
    return result;
  }
}
```

More on this issue in
["Uninitialized variables and lifetime" (TODO)](#broken-links-footnote)<!-- T:Uninitialized variables and lifetime -->
or
[carbon-uninit-v2 (TODO)](#broken-links-footnote)<!-- T:Carbon Uninitialized variables and lifetime v2 -->.

We _may_ want to run the destructor for `D1`, and absolutely need to run the
destructor for `B`. However, the code in `D1`'s factory function and its
destructor could change quite a bit depending on whether the destructor runs or
not.

##### Option A: "destructor runs if phase 1 succeeds"

Here the model is that `construct` returns an owning pointer (like C++
`unique_ptr`), that will be destroyed unless the factory function returns it. If
the destructor runs when a factory function fails in phase 2, then the
error-handling code in the factory function needs to get the object into a state
that the destructor can handle. This includes making sure any uninitialized
fields are set, if the destructor is going to do anything with them. Concern:

```
struct Baz extends Bar {  // Baz descends from Bar
  var Ptr(Foo): x;
  fn operator create() -> OwnedPtr(Baz) throws {
    var OwnedPtr(Baz): result = construct(.x = uninit);
    try {
      // Couldn't do this in phase 1 when it was safe since `Foo` needs
      // a pointer to this object (`result`) as a construct argument.
      result->x = new Foo(result);
    } except {
      // Uh oh! `result->x` is uninitialized and not optional. We don't have any
      // value we can assign to `result->x` to make the destructor happy.
    }
    return result;
  }
  fn operator destroy(Ptr(Baz): this) {
    delete this->x;
  }
}
```

Here the type of `x` would need to be changed to be optional and the destructor
would have to have extra checking that is otherwise unnecessary.

##### Option B: "destructor runs if phase 2 succeeds"

If the destructor never runs unless the factory function completes, then we
avoid the concerns about option A, but we instead need to duplicate the
destructor code in the factory function's error handling code. If there are
multiple failure points, we might need some mechanism like
[D's scope guard statements](https://digitalmars.com/d/2.0/statement.html#ScopeGuardStatement)
to specify the clean up code in a concise way.

**Note:** [chandlerc](https://github.com/chandlerc) prefers option B over option
A. He thinks we get one advantage by being able to destroy the members
individually.

I did not see anything in the Swift manual spelling out how failure is handled
during construction. I did find online that the compiler used to emit an error
saying "All stored properties of a class instance must be initialized before
throwing from an initializer", though this restriction was
[lifted in Swift 2.2](https://stackoverflow.com/questions/26495586/best-practice-to-implement-a-failable-initializer-in-swift/26496022#26496022).

**Question:** What about the case where the factory function calls another
function instead of phase 1 + `construct()`? In that case, we have a second
phase 2 after the object was fully constructed by the other factory function.
Should we run the destructor in that case?

I can see arguments both ways.

#### Alternative proposal (requested by Chandler): C++ constructor order

Like the thought experiment above, but with C++'s constructor order so you can
access the base type's member variables and functions in phase 1 of the parent.

This means that in a descendant you would have two special calls instead of one:

- initialize the parent, passing in arguments that determine which of the
  parent's factory functions to select and returns a pointer to the parent's
  type (which could be ignored if you don't need to access any of the values
  from the parent).
- initialize the descendant

      ```

  struct Baz extends Bar { // Baz descends from Bar var Int: y; fn operator
  create(Int: value) -> Ptr(Baz) { // This call can fail if the parent's factory
  function generates an error. // This call must come before construct(...), but
  unlike C++ doesn't have // to be the first statement in the function. var
  Ptr(Bar): parent = create_super(value);

      // Baz phase 1. Can use `parent` to compute the value of members.

      // Can we say that `construct()` will never fail?
      var Ptr(Baz): result = construct(.y = parent->f() + 2);
      // Actually `parent` and `result` store the same address. Should we allow
      // users to rely on this? Alternatively we can make `construct`
      // consume `parent` as a move argument so it doesn't remain valid
      // after this call.

      // Baz phase 2. Can modify *result, call methods on it, etc.

      return result;

  } }

```



If you wanted to call another factory function in this class, it would have to be instead of *both* of these special calls.


##### Comparison



*   Two calls instead of one.
*   Can't call virtual functions in base constructor, even in phase 2.
*   Can access base members in descendant's phase 1.
*   A bit more control over when the base constructor gets called, but it isn't clear what you would use it for.
*   Ending up with two pointers to the same object (but with different types) seems a bit weird.
*   A clearer story for invoking a mixin's factory function to initialize its fields.
*   **Big advantage:** Much clearer story for interop with C++ classes, assuming we have to support Carbon structs inheriting from C++ classes, and vice versa.


#### Alternative proposal (suggested by Richard): named factory functions

Building on the "C++ constructor order" proposal, imagine that any function returning a value of that type can be used as a factory function. Unfortunately, I haven't figured out how to make this work with mixins, where multiple functions work together to construct the object, each defining the value of a subset of the fields.

If we ignore mixins, what we are aiming for is something like:


```

struct Foo { private var Int: x; // Totally ordinary "static" function. No
special meaning to the name "Create". public fn Create(Int: x) -> Foo { // We
can call Foo() as a member of Foo. return Foo(.x = x); } public fn
CreateInPhases(Int: x) -> Foo { // Phase 1 var Foo: ret = (.x = x); // Phase 2,
can call methods on `ret` return ret; } } // Factory functions don't need to be
members of the type. fn MakeFoo(Int: x) -> Foo { return Foo.CreateInPhases(x); }

var Foo: f_illegal = Foo(.x = 3); // Error, Foo has private fields. var Foo:
f_allowed = Foo.Create(3);

struct Bar extends Foo { private var Int: y; public fn Create(Int: x, Int: y) ->
Bar { // First argument is value for the parent. return Bar(.Foo =
Foo.Create(x), .y = y); } } var Bar: b = Bar.Create(4, 5);

```


**Nice:** Since the special compiler-generated constructors never fail, the error recovery story is pretty straightforward. The only problem is if the code can't establish invariants that the destructor relies on before a failure in phase 2.

**Questions:**



*   Can this be made efficient, e.g. without a lot of copying? Without inheritance, this seems straightforward -- the space for `b` is allocated up front, and a pointer to that space is passed to `Bar.Create` which passes it to the `Bar(...)` constructor. The problem is arranging for the `Foo.Create()` call to construct its value at the beginning of the `Bar` allocation, especially as the factory function code gets more complex. For example, what if `Foo.Create()` can possibly generate an error? It would have to reason about control flow.
*   In the above proposal, there is a single special constructor function created by the compiler, which user's can't replace. They can only make it private and provide alternatives that are spelled differently. Is this a problem? I'm worried that evolving a type to one that needs logic in its constructor will involve changing all users.
*   What about parent types, like abstract base classes, that can't be instantiated directly?
*   How does this fit in with mixins?
*   Less obvious how to make an API like C++'s `std::vector&lt;T>::emplace_back()` that constructs a value in place.
*   What about types that don't have value semantics? In particular, what if `Foo.Create()` is in a doubly linked list and saves its `this` pointer?


#### Preliminary conclusion

Carbon will have some way of defining factory/initializer functions that will most likely follow the [C++-constructor-order version of the thought-experiment model](#alternative-proposal-requested-by-chandler-c-constructor-order), but could follow the Swift model or the Swift-construction-order-thought-experiment model. In any of these cases:



*   Factory functions are not ordinary functions, and will need some special syntax marking them as different. (In Swift they are defined starting with the `init` keyword, in C++ they don't have a return type and have a name matching that of the class.)
*   Factory functions will be allowed to fail. If any factory function in the inheritance hierarchy fails, any member variables and types that finished being constructed will be destroyed (option B above).
*   Eventually we will have some rules for inheriting factory functions from your parent class to avoid boilerplate (both [C++11](https://en.wikipedia.org/wiki/C++11#Object_construction_improvement) and [Swift](https://docs.swift.org/swift-book/LanguageGuide/Initialization.html#ID222) have this).

In the C++ constructor order model (currently preferred due to C++ interop concerns):



*   Factory functions will have a three-phase structure.
*   If the type is a descendant, then in phase 0, arguments for the parent's factory function are determined. Phase 0 ends with calling the parent's factory function, giving you a parent instance, or failure. If the type is not in an inheritance hierarchy, phase 0 is skipped.
*   In phase 1, you may call parent methods on a parent instance (if any) and prepare the values for the data members for the current type. Phase 1 ends by calling a function that transforms the parent instance into the current type. This call will never fail.

In the two Swift-like models (currently disfavored):



*   Factory functions will have a two-phase structure.
*   In phase 1, the initial values for the data members in the current class will be determined. If this class has a parent, you will also determine the arguments to a parent's factory function in phase 1.
*   If this class has a parent, the parent's factory function will be called at the boundary between phase 1 and phase 2. This call will probably have its own syntax (e.g. in Swift this is `super.init(...)`, in C++ you use the name of the parent class invoked in the initializer list). If a parent factory function's fails, the error will be reported at this point.

In either case, in phase 2 you can modify any data members (including those from the parent), and call member functions. **Concern:** [chandlerc](https://github.com/chandlerc) has reservations about allowing virtual functions to be called (particularly if they can invoke a descendant's override implementation) and would prefer to disallow it.


## Broken links footnote

Some links in this document aren't yet available,
and so have been directed here until we can do the
work to make them available.

We thank you for your patience.
```
