# Classes new material

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Nominal class types

The declarations for nominal class types will have a different format.

```
class TextLabel {
  var x: Int;
  var y: Int;

  var text: String;
}
```

### Forward declaration

To support circular references between class types, we allow forward declaration
of types. A type that is forward declared is considered incomplete until the end
of a definition with the same name.

```
class GraphNode;

class GraphEdge {
  var head: GraphNode*;
  var tail: GraphNode*;
}

class GraphNode {
  var edges: Vector(GraphEdge);
}
```

**Open question:** What is specifically allowed and forbidden with an incomplete
type, either from a forward declaration or referencing a type inside its
definition, has not yet been decided.

### Self

A `class` definition may provisionally include references to its own name in
limited ways. These limitations arise from the type not being complete until the
end of its definition is reached.

```
class IntListNode {
  var data: Int;
  var next: IntListNode*;
}
```

An equivalent definition of `IntListNode`, since `Self` is an alias for the
current type, is:

```
class IntListNode {
  var data: Int;
  var next: Self*;
}
```

`Self` refers to the innermost type declaration:

```
class IntList {
  class IntListNode {
    var data: Int;
    var next: Self*;
  }
  var first: IntListNode*;
}
```

### Construction

Any function with access to all the data fields of a class can construct one by
converting a [struct value](#struct-types) to the class type:

```
var p1: Point2D = {.x = 1, .y = 2};
var p2: auto = {.x = 1, .y = 2} as Point2D;
```

### Associated functions

FIXME

### Methods

FIXME

A future proposal will incorporate
[method](<https://en.wikipedia.org/wiki/Method_(computer_programming)>)
declaration, definition, and calling into classes. The syntax for declaring
methods has been decided in
[question-for-leads issue #494](https://github.com/carbon-language/carbon-lang/issues/494).
Summarizing that issue:

-   Accessors are written: `fn Diameter[me: Self]() -> Float { ... }`
-   Mutators are written: `fn Expand[addr me: Self*](distance: Float) { ... }`
-   Associated functions that don't take a receiver at all, like
    [C++'s static methods](<https://en.wikipedia.org/wiki/Static_(keyword)#Static_method>),
    are written: `fn Create() -> Self { ... }`

We do not expect to have implicit member access in methods, so inside the method
body members will be accessed through the `me` parameter.

### Nominal data classes

We will mark [data classes](#data-classes) with an `impl as Data {}` line.

```
class TextLabel {
  var x: Int;
  var y: Int;

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
    var count: Int;
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
  let Pi: Float32 = 3.141592653589793;
  let IndexType: Type = Int;
}
```

**Open question:** Should these use the `:!` generic syntax decided in
[issue #565](https://github.com/carbon-language/carbon-lang/issues/565)?

### Alias

FIXME

### Access control

FIXME

We will need some way of controlling access to the members of classes. By
default, all members are fully publicly accessible, as decided in
[issue #665](https://github.com/carbon-language/carbon-lang/issues/665).

The set of access control options Carbon will support is an open question. Swift
and C++ (especially w/ modules) provide a lot of options and a pretty wide space
to explore here.
