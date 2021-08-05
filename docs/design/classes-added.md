# Classes new material

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Nominal class types

The declarations for nominal class types will have:

-   `class` introducer
-   the name of the class
-   in the future, we will have optional modifiers for inheritance
-   `{`, an open curly brace
-   a sequence of declarations
-   `}`, a close curly brace

Declarations should generally match declarations that can be declared in other
contexts, for example variable declarations with `var` will define
[instance variables](https://en.wikipedia.org/wiki/Instance_variable):

```
class TextLabel {
  var x: i32;
  var y: i32;

  var text: String = "default";
}
```

The main difference here is that `"default"` is a default instead of an
initializer, and will be ignored if another value is supplied for that field
when constructing a value.

### Forward declaration

To support circular references between class types, we allow
[forward declaration](https://en.wikipedia.org/wiki/Forward_declaration) of
types. A type that is forward declared is considered incomplete until the end of
a definition with the same name.

```
class GraphNode;

class GraphEdge {
  var head: GraphNode*;
  var tail: GraphNode*;
}

class GraphNode {
  var edges: Vector(GraphEdge*);
}
```

**Open question:** What is specifically allowed and forbidden with an incomplete
type has not yet been decided.

### Self

A `class` definition may provisionally include references to its own name in
limited ways. These limitations arise from the type not being complete until the
end of its definition is reached.

```
class IntListNode {
  var data: i32;
  var next: IntListNode*;
}
```

An equivalent definition of `IntListNode`, since `Self` is an alias for the
current type, is:

```
class IntListNode {
  var data: i32;
  var next: Self*;
}
```

`Self` refers to the innermost type declaration:

```
class IntList {
  class IntListNode {
    var data: i32;
    var next: Self*;
  }
  var first: IntListNode*;
}
```

### Construction

Any function with access to all the data fields of a class can construct one by
converting a [struct value](#struct-types) to the class type:

```
var tl1: TextLabel = {.x = 1, .y = 2};
var tl2: auto = {.x = 1, .y = 2} as TextLabel;

Assert(tl1.x == tl2.x);

fn ReturnsATextLabel() -> TextLabel {
  return {.x = 1, .y = 2};
}
var tl3: TextLabel = ReturnsATextLabel();

fn AcceptsATextLabel(tl: TextLabel) -> i32 {
  return tl.x + tl.y;
}
Assert(AcceptsATextLabel({.x = 2, .y = 4}) == 6);
```

### Associated functions

An associated function is like a
[C++ static member function or method](<https://en.wikipedia.org/wiki/Static_(keyword)#Static_method>),
and is declared like a function at file scope. The declaration can include a
definition of the function body, or that definition can be provided out of line
after the class definition is finished. The most common use is for constructor
functions.

```
class Point {
  fn Origin() -> Self {
    return {.x = 0, .y = 0};
  }
  fn CreateCentered() -> Self;

  var x: i32;
  var y: i32;
}

fn Point.CreateCentered() -> Self {
  return {.x = ScreenWidth() / 2, .y = ScreenHeight() / 2};
}
```

Associated functions are members of the type, and may be accessed as using dot
`.` member access either the type or any instance.

```
var p1: Point = Point.Origin();
var p2: Point = p1.CreateCentered();
```

### Methods

[Method](<https://en.wikipedia.org/wiki/Method_(computer_programming)>)
declarations are distinguished from other
[function declarations](#associated-functions) by having a `me` parameter in
square brackets `[`...`]` before the explicit parameter list in parens
`(`...`)`. There is no implicit member access in methods, so inside the method
body members are accessed through the `me` parameter. Methods may be written
lexically inline or after the class declaration.

```carbon
class Circle {
  fn Diameter[me: Self]() -> f32 {
    return me.radius * 2;
  }
  fn Expand[addr me: Self*](distance: f32);

  var center: Point;
  var radius: f32;
}

fn Circle.Expand[addr me: Self*](distance: f32) {
  me->radius += distance;
}

var c: Circle = {.center = Point.Origin(), .radius = 1.5 };
Assert(Math.Abs(c.Diameter() - 3.0) < 0.001);
c.Expand(0.5);
Assert(Math.Abs(c.Diameter() - 4.0) < 0.001);
```

-   Methods are called using using the dot `.` member syntax, `c.Diameter()` and
    `c.Expand(`...`)`.
-   `Diameter` computes and returns the diameter of the circle without modifying
    the `Circle` instance. This is signified using `[me: Self]` in the method
    declaration.
-   `c.Expand(...)` does modify the value of `c`. This is signified using
    `[addr me: Self*]` in the method declaration.

FIXME: Meaning of `addr`

#### Name lookup in method definitions

FIXME

### Nominal data classes

We will mark [data classes](#data-classes) with an `impl as Data {}` line.

```
class TextLabel {
  var x: i32;
  var y: i32;

  var text: String;

  // This line makes `TextLabel` a data class, which defines
  // a number of operations field-wise.
  impl as Data {}
}
```

The fields of data classes must all be public. That line will add
[field-wise implementations and operations of all interfaces that a struct with the same fields would get by default](#operations-performed-field-wise).

### Member type

Additional types may be defined in the scope of a class definition.

```
class StringCounts {
  class Node {
    var key: String;
    var count: i32;
  }
  var counts: Vector(Node);
}
```

The inner type is a member of the type, and is given the name
`StringCounts.Node`.

### Let

Other type constants can be defined using a `let` declaration:

```
class MyClass {
  let Pi: f32 = 3.141592653589793;
  let IndexType: Type = i32;
}
```

These do not affect the storage of instances of that class.

**Open question:** Should these use the `:!` generic syntax decided in
[issue #565](https://github.com/carbon-language/carbon-lang/issues/565)?

### Alias

You may declare aliases of the names of class members. This is to allow them to
be renamed in multiple steps or support alternate names.

```
class StringPair {
  var key: String;
  var value: String;
  alias first = key;
  alias second = value;
}

var sp1: StringPair = {.key = "K", .value = "1"};
var sp2: StringPair = {.first = "K", .second = "2"};
Assert(sp1.first == sp2.key);
Assert(&sp1.first == &sp1.key);
```

### Access control

FIXME:

-   `private`: just accessible to members of the class, like C++
-   `internal`: works within the library + tests, does not provide linkage, does
    not export symbols through an `import` boundary; if accessed through a
    template or an inlined function body, needs to be `private` instead
    (probably based on the compiler telling you that you need to)
-   `friend`: works just within a package, does not introduce a new name

LATER: `protected`, maybe `package`

We will need some way of controlling access to the members of classes. By
default, all members are fully publicly accessible, as decided in
[issue #665](https://github.com/carbon-language/carbon-lang/issues/665).

The set of access control options Carbon will support is an open question. Swift
and C++ (especially w/ modules) provide a lot of options and a pretty wide space
to explore here.
