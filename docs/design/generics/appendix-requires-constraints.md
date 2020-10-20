<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Rejected: Constraints using "requires" clauses

## Description

Another approach for introducing constraints between associated types in an
interface is using a `requires` clause, as in:

```
interface Iterator {
  var Type:$ ElementType;
  ...
}
interface Container {
  var Type:$ ElementType;
  var Iterator:$ IteratorType;
  requires IteratorType.ElementType == ElementType;
  ...
}
```

Functions accepting a generic type might also want to constrain an associated
type. For example, we might want to have a function only accept stacks
containing integers:

```
fn SumIntStack[Stack:$ T](Ptr(T): s) -> Int requires T.ElementType == Int {
  var Int: sum = 0;
  while (!s->IsEmpty()) {
    sum += s->Pop();
  }
  return sum;
}
```

There are a lot of constraints you might want to express, such as that two
associated types are equal or that an associated type satisfies an interface.

```
fn EqualContainers[Container:$ CT1, Container:$ CT2]
    (Ptr(CT1): c1, Ptr(CT2): c2) -> Bool
  requires CT1.ElementType == CT2.ElementType &&
           CT1.ElementType as HasEquality { ... }
```

## Concerns

This makes determining if two types are equal into an undecidable problem. This
[arises in Swift](https://forums.swift.org/t/swift-type-checking-is-undecidable/39024).

We would need a little miniature language just for describing constraints in
`requires` clauses.

These constraints do not compose with other language features as well as things
represented directly in type-types.

## Extended discussion

Originally we were comparing the approaches of associated types with `requires`
constraints to interface parameters. This is the discussion that eventually led
me to say we should instead constrain associated types by making them optional,
named parameters, unifying them (to an extent) with interface parameters.

### Associated types vs. interface parameters: ergonomics

Associated types provide an advantage in ergonomics because constraints in
generic types and functions don't need to specify associated types that they
don't care about.

```
interface CollectionParam(Type:$ Element) { ... }
interface CollectionAssoc {
  var Type:$ Element;
}

// When `Element` is a parameter, every user of `Collection` has to specify it.
// Users that want to constrain `Element` to a concrete type get a nice syntax:
fn UseAnyCollectionOfInt[CollectionParam(Int):$ ManyInts](ManyInts: ints) { ... }

// Users who want to constrain `Element` in more complex ways have to first bind
// it to a type variable, to be able to specify the constraints for that type
// variable:
fn UseAnyCollectionOfPrintables[
  Printable:$,
  CollectionParam(P):$ ManyPrintables
](ManyPrintables: printables) { ... }

// Users who don't care about what `Element` is get boilerplate because they still
// have to bind it to a type variable:
fn Map[Type:$ OrigElement, CollectionParam(OrigElement):$ C, Type:$ NewElement](
  C: collection, fn (OrigElement) -> NewElement: mapper
) -> Array<NewElement> { ... }
//
// JoshL: Counter point here is that this function probably needs to refer to the
// OrigElement type somewhere anyway -- for example, here it is needed in the
// declaration of `mapper` and likely in the body of the function at least once.
// In this example, the differences in length between associated types and
// interface parameters has as much to do with choosing a short name for the
// collection type and a long name for the element type.
//
// DmitriG: Maybe it is not the greatest example. Consider std::vector's iterator
// or allocator type instead.
// Nevertheless, I think having to introduce a name for the generic type parameter
// (OrigElement) is somewhat of a burden even if it is used. The user already
// introduced a name for the type of the whole collection, why do they have to name
// its element type separately?

// When `Element` is an associated type, only users who care to constrain it
// have to specify it.
fn UseAnyCollectionOfInt[CollectionAssoc:$ ManyInts](ManyInts: ints)
    requires ManyInts.Element == Int { ... }

fn UseAnyCollectionOfPrintables[
  CollectionAssoc:$ ManyPrintables
](ManyPrintables: printables)
    requires ManyPrintables.Element: Printable { ... }

fn Map[CollectionAssoc:$ C, Type:$ NewElement](
  C: collection, fn (C.Element) -> NewElement: mapper) -> Array<NewElement> { ... }
```

### Associated types vs. interface parameters: existential types (dynamic types)

Interface parameters make us create separate types for different
"instantiations" of interfaces. Because of that, those instantiations can act as
existential types on their own.

```

interface CollectionParam(Type:$ Element) {
  method (Ptr(Self): this) GetByIndex(Int: i) -> Element;
  method (Ptr(Self): this) size() -> Int;
}
var DynPtr(CollectionParam(Int)): ints1 = &Array(Int)::make(...); // ok!
var DynPtr(CollectionParam(Int)): ints2 = &Set(Int)::make(...); // ok!
ints1.size(); // ok
ints1.GetByIndex(123); // ok!

interface CollectionAssoc {
  var Type:$ Element;
  method (Ptr(Self): this) GetByIndex(Int: i) -> Element;
  method (Ptr(Self): this) size() -> Int;
}
var DynPtr(CollectionAssoc): ints3 = &Array<Int>::make(...); // ok?
ints3.size() // ok -- returns an Int
ints3.GetByIndex(123) // what is the return type?
```

If we make `CollectionAssoc` into an existential, we don't bind the `Element` to
a concrete type. Therefore, we have a problem retrieving elements. How does the
compiler know that when we get an element out of `ints3` that it is an `Int`?
With the information provided in the type of `ints3`, it does not. Therefore,
`ints3.GetByIndex()` would either become inaccessible, or it would return a
completely opaque type, which would not be very useful.

Calling `ints3.size()` is perfectly fine though, because the signature of
`size()` does not mention any associated types.

### Design advice: use interface parameters (?)

An obvious reaction to difficulties above is to recommend API designers to use
interface parameters for type variables that users might want to constrain.
However, this recommendation has drawbacks.

First of all, users who don't want to constrain interface parameters have to
introduce useless type variables in their generic signatures. (Examples -- see
above.)

Second, for some type variables it is hard to predict if users will want to
constrain them or not (see the Collection.SubSequence example below). What if
only a small minority of users wants to constrain them? We would be choosing
between interface parameters and burdening every user with extra syntax, or
choosing associated types and not serving the use cases where these type
variables need to be constrained.

Finally, making some type variables into interface parameters and then having
users constrain them is harmful for genericity (see both the
Collection.SubSequence and Index examples below).

### Example: SubSequence in Collection

A subsequence of a collection is a view into the elements of this collection. We
can define it as either an interface parameter or as an associated type:

```
interface CollectionParam[Type:$ Element, Type:$ SubSequence]
    requires SubSequence = CollectionParam<Element, SubSequence> {
  method (Ptr(Self): this) Slice(Int: begin, Int: end) -> SubSequence;
}
interface CollectionAssoc {
  var Collection:$ SubSequence requires SubSequence.Element == Self.Element;
  method (Ptr(Self): this) Slice(Int: begin, Int: end) -> SubSequence;
}
```

Generally, users don't care about a specific type of the SubSequence. So then it
should be an associated type, according to the design advice? Well, but some
users care about the SubSequence and want to constrain it. For example, a
DropFirst API can be efficiently implemented for collections that are a
subsequence of themselves ("efficiently sliceable"). For example, we can
efficiently DropFirst from a `std::span` (whose SubSequence is `std::span`), but
not from a `std::vector` (whose subsequence is `std::span`).

```
fn DropFirstParam[Type:$ Element, Type:$ SliceableCollection]
    (Ptr(SliceableCollection): c)
    requires SliceableCollection = CollectionParam<Element, SliceableCollection> ->
        SliceableCollection {
  *c = c.Slice(1, c.size());
}
fn DropFirstAssoc[CollectionAssoc:$ SliceableCollection](Ptr(SliceableCollection): c)
    requires CollectionAssoc.SubSequence = CollectionAssoc {
  *c = c.Slice(1, c.size());
}
```

Notice how the `DropFirstParam` function has to thread the `Element` through
constraints even though it does not care about the specific element type. Notice
how it also has to declare all type variables before it can finally specify the
equality constraint. It can't even specify that `SliceableCollection` is a
`CollectionParam` within square brackets because not all type variables are
visible yet.

Notice how `DropFirstAssoc` specifies only things that it cares about. It does
not mention `Element`, only `SubSequence`.

Okay, so now we have established that some users want to constrain
`SubSequence`. Then, according to our design advice, it has to become an
interface parameter. But now users have to constrain `SubSequence` in
existential types. What are they going to constrain `SubSequence` to, though?

```
var DynPtr(CollectionParam<Int, ???>): ints1 = &Array<Int>::make(...);
var DynPtr(CollectionParam<Int, ???>): ints2 = &Set<Int>::make(...);
```

What do we put in place of the question marks? For `ints1` the subsequence would
be some equivalent of C++'s `std::span`. But `ints2` is backed by a set, it
won't be able to expose a `std::span` subsequence because it is organized
differently internally!

So, `SubSequence` will often expose some implementation details, and it would
not be desirable to constrain it to something specific in existential types.
Maybe we should type erase associated types in existentials? See the example
below to see where that fails.

### Example: Index in Collection

In the examples above, a simplified `Collection` interface was presented that
assumed that every collection can be efficiently indexed by an `Int`. However,
that is not true in practice. So Collection's index, just like in C++
collection's iterator, has to be an opaque type that is potentially different in
every collection.

Only very rarely users would want to constrain the Index of a collection. Every
collection can have its own index type, so constraining the index would make
code a lot less generic. So we should make it an associated type, right?

```
interface CollectionIndex {
  method (Ptr(Self): this) increment()
}
interface Collection[Type:$ Element] {
  var CollectionIndex:$ Index;
  method (Ptr(Self): this) GetByIndex(Int: Index) -> Element;
  method (Ptr(Self): this) StartIndex() -> Index;
  method (Ptr(Self): this) EndIndex() -> Index;
}
```

Alright, let's make an existential out of this collection:

```
var DynPtr(Collection<Int>): ints1 = &Array<Int>::make(...); // ok!
var DynPtr(Collection<Int>): ints2 = &Set<Int>::make(...); // ok!
ints1.GetByIndex(ints1.StartIndex()); // Whoops, what is the return type of
                                      // StartIndex()?
```

Maybe we should add another level of type erasure? Then, the return type of
`ints.StartIndex()` would be a `CollectionIndex` existential:

```
var DynPtr(CollectionIndex): i = ints1.StartIndex();
var DynPtr(CollectionIndex): j = ints2.StartIndex();
ints1.GetByIndex(i) // ok maybe?
ints1.GetByIndex(j) // Incorrect: j is an index of a Set, not an index of an Array!
                    // But the type system does not know.
```

Indeed, ints1.Index and ints2.Index can be completely different types. So we
can't just pass any existential into GetByIndex(), it has to be an existential
that internally contains the correct type.

The only way to model it in a sound way is to say that the type of
`ints1.StartIndex()` is dependent on the value of `ints1`:

```
var DynPtr(ints1.Index): i = ints1.StartIndex();
var DynPtr(ints2.Index): j = ints2.StartIndex();
ints1.GetByIndex(i) // ok!
ints1.GetByIndex(j) // not ok, and the type system can tell that.
```

Therefore, dmitrig@ thinks that we would be better off choosing associated types
only, and supporting all use cases within that model. Allowing interfaces to be
parameterized solves some first-order design problems that people will hit early
by allowing to switch an associated type into an interface parameter, and
therefore allowing to constrain it in an existential. However, this does not
solve harder issues where the API designer does not _want_ to make a type
variable into an interface parameter.

Swift's solution to constraining associated types in existentials is a feature
called "generalized existentials" (which is not implemented yet):

```
var DynPtr(CollectionAssoc): ints3 =
    &Array<Int>::make(...); // ok? But useless if allowed
var DynPtr(CollectionAssoc requires Element = Int) ints3 =
    &Array<Int>::make(...); // ok!
```

### joshl@ Reaction: Let's make associated types into "optional keyword parameters"

There definitely seem to be some differences between types that you might
typically want to parameterize vs. those you would want to be associated types:

-   For things like "element type" in a "container" interface, it is going to be
    more convenient to use interface parameters since every user of the
    interface is going to want to have a name for it anyway, and many will want
    to constrain it. You expect to see the element type as parameter of the
    interface and you want to see it clearly in the code. You would expect most
    (but not all, e.g. strings) container types to be parameterized by element
    type anyway. Frequently the interface will not constrain the type at all.
-   For things like "iterator type" or "slice type", you expect that to be
    determined by the container type rather than a parameter to the container
    type, and associated types are a more natural fit. Furthermore, we expect to
    be able to say quite a lot about the API of the "iterator type" or "slice
    type" as part of the "container" interface contract. You could imagine
    writing a lot of generic-container algorithms in terms of just the
    "iterator" API promised by the "container" interface without further
    constraints.

The trick is that we may still want to (relatively rarely) write a function that
constrains something that might naturally be an associated type, as you have
described in your examples. Having a way to constrain an interface parameter and
a different way to constrain an associated type sounds too heavyweight. But it
seems like we get most of the benefits of associated types by making them
optional named parameters, which are only specified when you want to express an
additional constraint on them. Then we have everything is a type parameter, but
some are required & positional, and some are named & optional.

```
interface IteratorType(Type:$ ElementType) { ... }
interface RandomAccessIterator(Type:$ ElementType)
    extends IteratorType(ElementType) { ... }
interface ContainerInterface(Type:$ ElementType) {
  var IteratorIterface(ElementType):$ IteratorType;
  var ContainerIterface(ElementType):$ SliceType;
...
}

// The following expressions could be FOO in `fn MyFunction[FOO](...) { ... }`,
// that is these are examples of function generic parameter lists.

ContainerInterface(Int):$ IntContainerType

Type:$ T, ContainerInterface(T):$ AnyContainerType

Type:$ T, RandomAccessIterator(T):$ RandomIteratorType,
ContainerInterface(T, .IteratorType=RandomIteratorType):$ RandomAccessContainerType

// Question: how can we define a ContainerInterface with SliceType == ContainerType?
// Perhaps:
Type:$ T, ContainerInterface(T, .SliceType=Self):$ ContainerWithSlicesType
// Answer(dmitrig): I think the obvious answer is that one should write a "requires"
// clause that specifies a same-type constraint, but I feel like that is not
// applicable here somehow?
Type:$ T, ContainerInterface(T):$ MyContainer
    requires MyContainer.SliceType == MyContainer
// dmitrig: We could put the same-type constraint into the parentheses of
// (arbitrarily) either left-hand-side or the right-hand-side of the
// same-type constraint:
Type:$ T, ContainerInterface(T, .SliceType = .):$ MyContainer
// dmitrig: for example, two containers with the same slice type:
Type:$ T,
ContainerInterface(T):$ MyContainer1,
ContainerInterface(T, .SliceType = MyContainer1.SliceType):$ MyContainer2
// dmitrig: however, I feel like we are getting into the "angle bracket blindness"
// problem here. In Swift, we on purpose changed the syntax to move the constraints
// to a separate section in a function declaration at the tail end of the signature,
// after all type parameters, value parameters, and the return type. We did this in
// response to the user feedback that users mostly didn't care about generic
// constraints when reading the signature, but those constraints required a lot of
// characters _at the very beginning of the function declaration_ and were obscuring
// the more important information like the names of the generic type parameters and
// value parameters (according to users' opinion).
```
