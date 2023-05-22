# `for` statement and user types

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/1885)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
    -   [Other languages](#other-languages)
        -   [C++](#c)
        -   [Python](#python)
        -   [Rust](#rust)
        -   [Typescript](#typescript)
        -   [Go](#go)
-   [Proposal](#proposal)
-   [Details](#details)
    -   [`for` with immutable values](#for-with-immutable-values)
        -   [`Iterate` interface](#iterate-interface)
        -   [`let` by default](#let-by-default)
        -   [Lifetime of rvalues on right-hand side of `:`](#lifetime-of-rvalues-on-right-hand-side-of-)
    -   [Use cases](#use-cases)
        -   [Mutable and immutable elements](#mutable-and-immutable-elements)
        -   [Random access containers](#random-access-containers)
        -   [Maps, hash maps, and other containers](#maps-hash-maps-and-other-containers)
        -   [R-value containers](#r-value-containers)
        -   [Synthetic data](#synthetic-data)
        -   [Large containers](#large-containers)
        -   [Filtering, transforming, and reversing](#filtering-transforming-and-reversing)
        -   [Interoperability with C++ types and containers](#interoperability-with-c-types-and-containers)
-   [Future work](#future-work)
    -   [`for` with mutable values](#for-with-mutable-values)
    -   [Speculative mixin usage](#speculative-mixin-usage)
        -   [Mutable values](#mutable-values)
        -   [Alternative views](#alternative-views)
    -   [Large elements and Optional](#large-elements-and-optional)
    -   [C++ interoperability](#c-interoperability)
        -   [Iterating over C++ types in Carbon](#iterating-over-c-types-in-carbon)
        -   [Iterating over Carbon types in C++](#iterating-over-carbon-types-in-c)
    -   [Inversion of control](#inversion-of-control)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Atomic methods for `Iterate`](#atomic-methods-for-iterate)
    -   [Using an iterator instead of a cursor](#using-an-iterator-instead-of-a-cursor)
    -   [Support getter for both `T` and `T*` with `Iterate`](#support-getter-for-both-t-and-t-with-iterate)

<!-- tocstop -->

## Abstract

This proposes an interface that can be implemented by user types to support
range-based iteration with `for`.

## Problem

The current `for` proposal does not define a way for types to support
range-based `for` loops.

Examples of use cases for `for` loops include iterating over:

-   Mutable and immutable elements
-   Random access containers
-   Maps, hash maps, and complex containers
-   R-value containers
-   Synthetic data
-   Large containers, and large elements
-   Filtering, transforming, and reversing
-   Interoperability with C++ types and containers

The goals for the solution includes:

-   Easy to support for user types, with minimal requirements
-   Simple and clear usage
-   Minimal performance overhead

## Background

The original proposal of `for` loops were discussed in
[p0353](/proposals/p0353.md), and the basic design documented in
[control_flow/loops#for](/docs/design/control_flow/loops.md#for).

The resulting syntax was:

> `for (` _var declaration_ `in` _expression_ `) {` _statements_ `}`

We now need a way to allow supporting this syntax for user and library container
types.

### Other languages

This section highlights some examples of how iterators and range-based for
user-defined types is addressed in different languages.

#### C++

C++ requires either `.begin()`/`.end()` methods, or
`begin(<container>)`/`end(<container>)` to be supported for
[range-based `for`](https://en.cppreference.com/w/cpp/language/range-for).

#### Python

Python requires 2 dunder methods, `__iter__` on the container and `__next__` on
the iterator, to be usable with a `for <element> in <container>:` construct.

[Generator functions](https://peps.python.org/pep-0255/) act like iterators.
They are executed until the next `yield` statement, whose argument is returned
as the next value to the `for` loop. Iteration ends when the function returns.

```python
def generate_fibonacci():
  yield 0;
  n1, n2 = 0, 1;
  while True:
    n1, n2 = n2, n1+n2;
    yield n1;

for n in generate():
  print(n);
```

#### Rust

Rust supports iterating over a container with
[for](https://doc.rust-lang.org/rust-by-example/flow_control/for.html) using

-   ranges `for n in 1..100`
-   iterators `for element in container.iter()` or `.iter_mut()`

The `Iterator` type can be used with custom types using the
[following syntax](https://doc.rust-lang.org/rust-by-example/trait/iter.html)
(though specific):

```rust
struct Fibonacci {
    curr: u32,
    next: u32,
}

impl Iterator for Fibonacci {
    type Item = u32;
    fn next(&mut self) -> Option<Self::Item> {
        // calculate and yield value
        [...]
    }
}

for i in fibonacci().take(4) {
    ...
}
```

#### Typescript

Typescript uses the
[`.iterator` property](https://www.typescriptlang.org/docs/handbook/iterators-and-generators.html#iterables),
along with
[two forms](https://www.typescriptlang.org/docs/handbook/iterators-and-generators.html#forof-statements)
of range-based for loops: `for..in` and `for..of`.

```ts
let pets = new Set(['Cat', 'Dog', 'Hamster']);
pets['species'] = 'mammals';
for (let pet in pets) {
    console.log(pet); // "species"
}
for (let pet of pets) {
    console.log(pet); // "Cat", "Dog", "Hamster"
}
```

Using the `.iterator` property, Typescript allow to lazily generate data easily
using functions, classes, or objects:

```ts
const reverse = (arr) => ({
    [Symbol.iterator]() {
        let i = arr.length;
        return {
            next: () => ({
                value: arr[--i],
                done: i < 0,
            }),
        };
    },
});
```

#### Go

Go does not officially support customizing `for` for user types.

## Proposal

Provide interfaces that can be implemented by user types to enable support for
ranged-for loops.

We propose:

-   A new `Iterate` interface to support for loops
-   Using a cursor-based approach
-   Making the `for` loop value `let` by default, that is, non-reassignable

Below is a high-level example of this concept:

```carbon
// Add support for `for`
class MyRange {
  impl as Iterate where .ElementType = i32 and .CursorType = i32 {
    // Implement `Iterate`
  }
}
// Usage
let my_range: MyRange = {...};
for (item: auto in my_range) {
  // Do something with `item`
}
```

## Details

### `for` with immutable values

#### `Iterate` interface

This proposal exposes a new `Iterate`. This interface:

-   Relies on a cursor-based approach: minimize implementation effort for
    developers, could support R-values.
-   Expects an `ElementType` and `CursorType`.
-   Combines "advance", "get", and "bounds check" into a single `Next(cursor)`
    method that returns an optional value.

The `Iterate` interface is:

```carbon
interface Iterate {
  let ElementType:! type;
  let CursorType:! type;
  fn NewCursor[self: Self]() -> CursorType;
  fn Next[self: Self](cursor: CursorType*) -> Optional(ElementType);
}
```

A naive Carbon implementation of the for loop could be:

```carbon
var cursor: range.(Iterate.CursorType) = range.(Iterate.NewCursor)();
var iter: Optional(range.(Iterate.ElementType)) = range.(Iterate.Next)(&cursor);

// A. Possible implementation
while (iter.HasValue()) {
  ExecuteForBlock(iter.Get());
  iter = container.(Iterate.Next)(&cursor);
}

// B. Possible alternative depending on the API for Optional(T):
while (true) {
  match (iter) {
    case .None => { break; }
    case .Some(let value: range.(Iterate.ElementType)) => {
      ExecuteForBlock(value);
      iter = container.(Iterate.Next)(&cursor);
    }
  }
}
```

The cursor tracks progression, and the `Next` method advances the cursor and
returns an `Optional` value. An empty `Optional` indicates that we have reached
the end.

Below is a sample implementation of that interface for a user type.

```carbon
class MyIntContainer {
  var data: ...
  fn Size[self: Self]() -> i32 {
      return 4;
  }
  impl as Iterate where .ElementType = i32 and .CursorType = i32 {
    fn NewCursor[self: Self]() -> i32 { return 0; }
    fn Next[self: Self](cursor: i32*) -> Optional(i32) {
      // Advance and return value, or return empty
      if (*cursor < self.Size()) {
        let index: i32 = *cursor;
        *cursor = index + 1;
        return Optional(i32).Create(self.data[index]);
      } else {
        return Optional(i32).CreateEmpty();
      }
    }
  }
}
```

Which can be used as follow:

```carbon
fn Main() -> i32 {
  var container: MyIntContainer = {.data = (1, 2, 3, 4)};
  // Implicit `let` value declaration
  for (value: i32 in container) {
    Print("{0}", value);
  }
  return 0;
}
```

#### `let` by default

To keep usage simple and non-surprising, we propose defaulting variable
declarations to `let` declarations. This is consistent with function parameters
and scrutinees for
[pattern matching](https://github.com/carbon-language/carbon-lang/pull/2188),
which are immutable by default.

```carbon
// Implicitly `let item: T`
for (item: T in my_range) {
  // `item` cannot be reassigned
}
```

This default can be overridden by adding the `var` keyword:

```carbon
for (var item: T in my_range) {
  // `item` can be modified, but this will not modify `my_range`
  // because it is a copy of the element with its own storage
}
```

#### Lifetime of rvalues on right-hand side of `:`

It will be important to ensure that temporary entities on the right-hand side of
`:` are alive during the execution of the `for` loop to prevent invalid memory
access (Note: this was only recently addressed for C++ by
[P2644](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p2644r0.pdf)).
This applies to Carbon and C++ interoperability.

```carbon
// Example using a temporary range, with `GetElements()` returning an object that implements `Iterate`
for (item: T in bucket.GetElements()) {
  // We need to make sure the object is alive for the duration of the `for` loop
}
```

### Use cases

This section covers the use cases highlighted in the introduction, and usage
with the proposed design.

#### Mutable and immutable elements

This proposal covers strictly immutable elements, and proposes defaulting
element declaration to `let`, to minimize surprises and clarify the intent.
Mutable elements are mentioned in the [Future work](#future-work) section.

#### Random access containers

For random access containers, the cursor would represent a numerical index,
incremented within `Next()`.

#### Maps, hash maps, and other containers

Because the cursor type is defined by the developer, and opaque to the user, it
could be a hashmap index, a string, or pointer to a node, without changing the
usage for users.

The `ElementType` can be a tuple, such as a `(key, value)` for maps, or a single
value. See [Future work][#future-work] for other examples.

#### R-value containers

R-values are likely to be limited in terms of addressing and mutation. The
impact is undefined until the
[corresponding design](https://github.com/carbon-language/carbon-lang/pull/2006)
is finalized.

The cursor approach used for `Iterate` has the slight advantage of not requiring
pointers (and addressing) for some container types. This may be beneficial as a
first iteration, until the dependent design is updated, at which point we will
have to define how to handle this special case.

#### Synthetic data

`Next()` can be trivially implemented to return a single computed value
on-demand based on the cursor, without the need for storage. If the sequence is
infinite, `Next()` will never return an empty `Optional`, and the `for` loop
will continue until interrupted by the user.

```carbon
class SquaredSequence {
  impl as Iterate where .ElementType = i32 and .CursorType = i32 {
    fn NewCursor[self: Self]() -> i32 { return -1; }
    fn Next[self: Self](cursor: i32*) -> Optional(i32) {
      // Advance and return computed value
      *cursor = *cursor + 1;
      return Optional(i32).Create((*cursor) * (*cursor));
    }
  }
}
```

#### Large containers

The proposal does not present any limitation regarding large collections. The
cursor can be adapted to the collection size, and no copy of the container
should occur both for immutable collections, and future mutable elements.

#### Filtering, transforming, and reversing

Provided the traversal order is the same, generic filters and transformers are
supported by implementing a proxy `Iterate` interface, internally or externally.

This example illustrates a simple proxy interface.

```carbon
class BasicFilter(T:! Iterate) {
  var seq: T*;
  var value: T.ElementType;
  impl as Iterate where .ElementType = T.ElementType and .CursorType = T.CursorType {
    fn NewCursor[self: Self]() -> T.ElementType { return self.seq->NewCursor(); }
    fn Next[self: Self](cursor: T.CursorType*) -> Optional(T.ElementType) {
      var res: auto = self.seq->Next(cursor);
      while (res.has_value() && res.get() == value) {
        res = self.seq->Next(cursor);
      }
      return res;
    }
  }
}

fn Main() -> i32 {
  let container: MyIntContainer = MyIntContainer.Create(0,1,0,2,3,0);
  let no_zeros: BasicFilter(MyIntContainer) =
      { .seq = &container, .value = 0 };
  // Prints "123"
  for (v: i32 in no_zeros) {
    Print("{0}", v);
  }
  return 0;
}
```

On the other hand, changing how the collection is traversed (for example
reverse), requires internal knowledge about the data layout, and a custom
implementation. One approach for containers that support reverse iteration would
be to implement a different interface with a `Prev()` method. A proxy could be
exposed to make it usable with `for` directly.

Below is an example of what such an interface could look like:

```carbon
interface IterateBidirectional;
class ReverseProxy(T:! IterateBidirectional);
interface IterateBidirectional {
  extends Iterate;
  fn NewEndCursor[self: Self]() -> CursorType;
  fn Prev[self: Self](cursor: CursorType*) -> Optional(ElementType);
  final fn Reversed[self: Self]() -> ReverseProxy(Self);
}
class ReverseProxy(T:! IterateBidirectional) {
  var container: T;
  impl as Iterate where .ElementType = T.ElementType and .CursorType = T.CursorType {
    fn NewCursor[self: Self]() -> CursorType {
      return self.container.NewEndCursor();
    }
    fn Next[self: Self](cursor: CursorType*) -> Optional(ElementType) {
      return self.container.Prev(cursor);
    }
  }
}

fn IterateBidirectional.Reversed[self: Self]() -> ReverseProxy(Self) {
  return {.container = self};
}

/// Usage
class MyContainer {
  fn Create(...) { ... }
  impl as Iterate ... { ... }
  impl as IterateBidirectional ... { ... }
}
var container: MyContainer = MyContainer.Create(...);
for (v: container.(Iterate.ElementType) in container.Reversed()) {...}
```

#### Interoperability with C++ types and containers

The cursor approach deviates from the C++ iterator approach, requiring some
adaptation.

Given the current state of the design, C++ interoperability is yet to be
defined, and suggestions have been highlighted in the
[Future work](#future-work) section.

## Future work

### `for` with mutable values

Supporting mutable values could be achieved using a second interface that
provides a proxy or view for the data, exposing an `Iterate` interface with
`ElementType*`

```carbon
// Object implementing `MutableIterate` returned by a function
for (p: T* in my_range.AddrRange()) {}
```

The example `MutableIterate` interface below acts as a proxy for the collection,
and implements our `Iterate` interface.

```carbon
interface MutableIterate {
  impl as Iterate;
  alias ElementType = Iterate.ElementType;
  let ProxyType:! Iterate where .ElementType = ElementType*
      and .CursorType is ImplicitAs((Self as Iterate).CursorType);
  fn AddrRange[addr self: Self*]() -> ProxyType;
}

class MyIntContainer {
  //...
  impl as Iterate ...
  external impl as MutateIterate where .ProxyType = Self {
    ...
  }
}
```

### Speculative mixin usage

Mixins are currently in early design stages. This section highlights possible
uses speculating on the final design. Some may include:

-   Improved semantics for views compared to getter methods: no direct side
    effect from using the view
-   Facilitate code reuse, compared to reimplementing an interface
-   Direct access to `self`, limiting needs for pointers and address resolution

#### Mutable values

Named mixins, used as programmable members, could expose a view with mutable
values:

```carbon
// Using a named mixin to expose mutable values
for (p: my_range.(Iterate.ElementType)* in my_range.mutable) {}
```

Used in this way, this clarifies that this a view of the data, like an
attribute, with no direct side effects, when compared to a method.

#### Alternative views

Similarly, mixins could allow supporting other views into the container.

For example, maps could expose a view with "key" and "value" only, in addition
to the default (key, value) tuple.

```carbon
// Using a named mixin to expose mutable values
for (k: my_dict.(Iterate.CursorType) in my_dict.keys) {}
```

And "reversed" view could also be used for reverse iteration:

```carbon
// Reversing
for (v: my_dict.(Iterate.ElementType) in my_dict.reversed) {}
```

### Large elements and Optional

Large elements would have a negative impact if the `Optional` cannot leverage
unused bit patterns within the type, nor present a view instead of a copy.

If this proves difficult, we can let implementers of the container choose
whether to expose values or pointers to values. When electing to "pointers to
values", an adapter can be provided to support transforming the range of
pointers into a range a values. This choice allows supporting either iteration
over container r-values (`ElementType = T`), or efficient iteration
(`ElementType = T*`), but not both. While acceptable as an intermediate
solution, a better alternative will be needed in the future.

```carbon
adapter IterateByValue(T:! Iterate where .CursorType impls Deref) for T {
  impl as Iterate
      where .CursorType = T.CursorType
        and .ElementType = T.CursorType.(Deref.Result) {
        ...
  }
}

// Used like:
var frames: NetworkFramesContainerByPtr = ...
for (i: NetworkFrame in frames as IterateByValue(NetworkFramesContainerByPtr))
{
  ...
}
```

Alternatively, we can allow the binding in the `for` loop to be an `addr`
pattern, implemented by calling into a separate interface on the container. This
puts the choice of using values or pointers to values in the hands of the `for`
loop author. While this deviates from the ergonomics of `let` (where the
compiler can choose to copy or alias the value depending on what is more
efficient), this option would provide parity with C++ `for` loops (where the
author chooses between value and reference).

### C++ interoperability

#### Iterating over C++ types in Carbon

For reference, range-based `for` loops in C++ requires:

-   `begin()` and `end()` methods or free functions, and
-   the type returned supports pre-increment `++`, indirection `*`, and
    inequality `!=` operations

See [range-based for statement](https://eel.is/c++draft/stmt.iter#stmt.ranged)
for more details.

A limitation of the approach proposed in this document is the need for
adaptation to interoperate with the C++ iterator model. The adapter would be a
Carbon type that satisfies the Cursor interface, implemented in terms of a C++
iterator.

-   `NewCursor` providing the "start iterator" returned by `begin()`
-   `Next` doing the "increment, bounds check, and returning value"

Two different approaches are identified:

-   One is to have a templated `impl` of `Iterate`, for anything with the right
    method signatures (`begin`, `end`, and so on)
-   Another is a templated adapter that implements `Iterate` when requested

```carbon
// Template impl approach
template constraint CppIterator { ... }
template constraint CppContainer {
  // Speculative
  for_unique IteratorType: CppIterator;
  fn begin() -> IteratorType;
  fn end() -> IteratorType;
}
external impl forall [template T:! CppContainer] T as Iterate
    where .CursorType = T.(CppContainer.IteratorType)
      and .ElementType = .CursorType.ElementType {
  fn NewCursor[self: Self]() -> CursorType {
    return self.(CppContainer.begin)();
  }
  fn Next[self: Self](cursor: CursorType*) -> Optional(ElementType) {
    if (*cursor == self.(CppContainer.end)()) {
      return Optional(ElementType).MakeEmpty();
    }
    let value: ElementType = **cursor;
    ++*cursor;
    return Optional(ElementType).MakeValue(value);
  }
}
// Used like:
var v: Cpp.std.vector(i32) = ...
for (i: i32 in v) { ... }
```

```carbon
// Adapter approach
adapter CppIterate(template T:! type) for T {
  impl as Iterate
      // Speculative syntax. Alternatively, a cursor type parameter
      // would avoid the need for deduction.
      where .CursorType = typeof(T.begin())
        and .ElementType = .CursorType.(Deref.Result) {
    fn NewCursor[self: Self]() -> CursorType {
      return self.begin();
    }
    fn Next[self: Self](cursor: CursorType*) -> Optional(ElementType) {
      if (*cursor == self.end()) {
        return Optional(ElementType).MakeEmpty();
      }
      let value: ElementType = **cursor;
      ++*cursor;
      return Optional(ElementType).MakeValue(value);
    }
  }
}
// Used like:
var v: Cpp.std.vector(i32) = ...
for (i: i32 in v as CppIterate(Cpp.std.vector(i32))) { ... }
```

These two options could allow interoperability with a compatible C++ type,
either implicitly (template approach) or explicitly (adapter approach).

#### Iterating over Carbon types in C++

There need to be `begin()` and `end()` methods or free functions accessible to
C++ for any Carbon type that implements the `Iterate` interface, in addition to
overloads of the `*`, `++`, and `!=` operators for the types returned.

The `Iterate` interface does not expose a way to have a cursor nor an iterator
to the past-last element. An option would be to have `end()` return a predefined
invalid value, to satisfy iterator comparison when `Next()` returns an empty
`Optional`.

Below is sample implementation to iterate on a Carbon `Iterate`. Note that we
need to copy and cache the last value, to allow for `*it` to be called. This
puts a requirement on `ElementType` to be copyable for this interoperability to
be usable.

```cpp
namespace Carbon {
// A C++ input iterator for any type that implements the Carbon `Iterate` interface.
template <typename Iterate>
class IterateIterator {
 public:
  // Returns an "Invalid" iterator representing `end()`
  static auto Invalid(const Iterate& container) -> IterateIterator<Iterate> {
    return IterateIterator<Iterate>(container, std::nullopt);
  }

  IterateIterator(const Iterate& container,
                  std::optional<typename Iterate::CursorType> cursor)
      : container_(&container), cursor_(cursor) {
    if (cursor_) {
      next();
    }
  }
  IterateIterator(const IterateIterator& other)
      : container_(other.container_), cursor_(other.cursor_),
      value_(other) {}

  auto operator*() const -> const typename Iterate::ElementType& {
    assert(cursor_);
    return value_;
  }
  auto operator++() -> IterateIterator<Iterate>& {
    next();
    return *this;
  }
  auto operator==(const IterateIterator<Iterate>& rhs) -> bool {
    return container_ == rhs.container_ && cursor_ == rhs.cursor_;
  }
  auto operator!=(const IterateIterator<Iterate>& rhs) -> bool {
    return !(*this == rhs);
  }

 private:
  void next() {
    assert(cursor_);
    if (const auto v = container_->Next(*cursor_); v) {
      value_ = *std::move(v);
    } else {
      cursor_ = std::nullopt;
    }
  }

  const Iterate* container_;
  std::optional<typename Iterate::CursorType> cursor_;
  typename Iterate::ElementType value_;
};

// The C++ side of a custom Carbon type implementing
// `Iterate` with CursorType `C`, and ElementType `T`
// could have `begin()` and `end()` methods defined as:
class MyIterateType {
  // [...]
  auto begin() -> IterateIterator<Iterate<C, T>> {
    return IterateIterator<Iterate<C, T>>(*this, NewCursor());
  }
  auto end() -> IterateIterator<Iterate<C, T>> {
    return IterateIterator<Iterate<C, T>>::Invalid(*this);
  }
};
}  // namespace Carbon

// Usage:
Carbon::MyIterateType container;
for (auto s : container) { /*...*/ }
```

### Inversion of control

Using an `Optional` has downsides, discussed in the
["alternatives considered"](#alternatives-considered) section. A possible
approach would be to invert the control of the `for` loop, by having the
container call in the `for` block for each value, passing the value as argument.
This would work similarly to [Python generator functions](#python).

This has the following advantages:

-   Removes the need for an `Optional`, and the associated overheads, copies,
    and unwrapping.
-   No boundary checks needed at the `for` level
-   Compatible with R-value containers

Limitations:

-   The context needed by the `for` body will needs to be available or provided
-   No optimizer known to the team can reliably produce efficient code for this
    construct.

## Rationale

This proposal offers a first step to support a needed way to iterate using a
`for` loop. More specifically:

-   Requires a single interface to be implemented with a combined `Next` method,
    uses cursors instead of their more complex iterator counterpart, and defines
    loop values as `let` by default to be less error prone
    ([Code that is easy to read, understand, and write](https://github.com/carbon-language/carbon-lang/blob/trunk/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)).
-   Makes use of
    [Library APIs only](https://github.com/carbon-language/carbon-lang/blob/trunk/docs/project/principles/library_apis_only.md).

## Alternatives considered

### Atomic methods for `Iterate`

An option was to define methods such as `Get`, `Advance`, `IsAtEnd` on
`Iterate`, instead of a unique `Next` methods. While being more atomic, the
concern is that these individual methods will all need to be called separately
while iterating, while a single call to `Next` performs these three operations
together, providing more opportunities for optimization.

Using a single operation to return an optional is also consistent with Rust’s
approach.

The combined `Next` method forces us to return an `Optional`, which has
downsides:

-   potential performance overhead to wrapping and unwrapping,
-   storage overhead for `Optional` itself, and
-   may require additional copying of the value wrapped.

Those may be mitigated by leveraging unused bit patterns, passing a view of the
value, or [inverting control](#inversion-of-control).

### Using an iterator instead of a cursor

The iterator approach was considered but proved to have key limitations that a
cursor approach does not have:

-   It requires implementing 2 interfaces instead of 1, and is more complex due
    to having the iteration logic separated from the container itself
-   More difficult to harden or troubleshoot, compared to a cursor that allows
    bounds checking in debug & hardened build modes
-   Can pose problems with ranges that consumes their input (See Barry Revzin’s
    "take(5)" presentation from C++Now 2023), when compared to a combined
    `Next()` approach.
-   Higher overhead
-   Proves to be difficult to support for R-values

On the other hand, the cursor approach deviates more from C++ than iterator, and
may require some more effort for C++ interoperability.

### Support getter for both `T` and `T*` with `Iterate`

Because some collections will be read-only, we want to separate mutable and
immutable interfaces. A single interface would force developers to implement or
stub the mutable portion of the interface, even if it is not applicable.
