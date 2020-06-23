# User-defined types

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Table of contents

<!-- toc -->

- [Inheritance](#inheritance)
  - [Using C++ types as mixins for Carbon structs](#using-c-types-as-mixins-for-carbon-structs)
    - [Caveat: Abstract methods](#caveat-abstract-methods)
  - [Inheriting from C++ types with Carbon structs](#inheriting-from-c-types-with-carbon-structs)
    - [Caveat: Missing or conflicting method declarations](#caveat-missing-or-conflicting-method-declarations)
    - [Caveat: Concrete methods on parents](#caveat-concrete-methods-on-parents)
    - [Alternative: Require bridge code for inheritance](#alternative-require-bridge-code-for-inheritance)
  - [Implementing Carbon interfaces in C++](#implementing-carbon-interfaces-in-c)
  - [Public non-virtual inheritance](#public-non-virtual-inheritance)
    - [Alternative: Simulate interfaces for C++ types](#alternative-simulate-interfaces-for-c-types)
  - [Virtual and Non-Public Inheritance](#virtual-and-non-public-inheritance)
- [Unions and transparent union members](#unions-and-transparent-union-members)

<!-- tocstop -->

All (non-template) user-defined C++ class types (and C struct types) are
directly available within Carbon with the exact C++ layout. Member function
semantics (including special member function semantics) are outlined below, but
the data is directly available.

i.e., given a simple C/C++ class:

```cc
class Circle {
 public:
  double GetArea();

 private:
  double radius_;
};
```

We expect this to behave as a similar Carbon class:

```carbon
package Cpp;

struct Circle {
  fn GetArea() -> Float64;
  private var Float64: radius_;
};
```

### Inheritance

We expect to have a different object inheritance model than C++. For example,
Carbon structs may use [interfaces and mixins](http://go/carbon-struct). In
order to avoid having C++ inheritance constrain Carbon's design, cross-language
type inheritance will be restricted: C++ types won't be able to directly inherit
from Carbon types, and vice versa.

Even if a C++ class is able to inherit from a Carbon type, Carbon will not
recognize the subclass. Therefore, all Carbon types exported to C++ will be
marked as `final`.

For use cases like implementing interfaces required by C++ APIs, embedded C++
code within the Carbon code should be used as the bridge. For an example of
addressing the lack of cross-language inheritance, see the
[Framework API](#bookmark=kix.jrmn54jubgcz) migration example.

#### Using C++ types as mixins for Carbon structs

Assuming we provide mixins on Carbon structs, we should be able to safely allow
C++ types as mixins on Carbon structs. This would allow C++ types to provide
APIs directly, without a Carbon wrapper.

This assumes there are no other implications for Carbon or C++
inheritance-related features. If there are, we should reconsider this.

For example, consider the C++ code:

```cc
class Shape {
 public:
  double GetArea();
};
```

The Carbon code should be able to reuse the class's implementation:

```carbon
struct ShapeWrapper {
  mixin Cpp.Shape;
}

ShapeWrapper x = MakeShapeWrapper();
x.GetArea();
```

However, this has no interface implications for ShapeWrapper. Also, if used from
C++, ShapeWrapper would not be considered to inherit from Shape. Additional
language-specific wrappers would be required to get inheritance functionality.

##### Caveat: Abstract methods

C++ abstract methods will not have a direct equivalent in Carbon. Although a
Carbon interface is similar to a C++ class with only abstract methods, it's not
clear that it's useful or helpful to translate such types as interfaces. A
Carbon struct should only have concrete methods.

A C++ class with abstract methods cannot be instantiated, as in a mixin. Users
should expect a compiler error in such a case, similar to if they attempted to
instantiate the class in Carbon code in general.

#### Inheriting from C++ types with Carbon structs

Inheritance from C++ types cannot be straightforward because Carbon will not
have matching inheritance concepts. However, we can augment Carbon structs to
indicate an inheritance chain that works similarly.

For example, consider the C++ code:

```cc
class Shape {
 public:
  virtual double GetArea() = 0;
  Shape* MakeUnion(Shape* other) { ... };
};
```

In order to provide an implementation in Carbon, we can then write code like:

```carbon
import Cpp "project/shape.h"

$extern("Cpp", parent="Cpp.Shape") struct Circle {
  fn GetArea() -> Float64 { ... };
}
```

This will create code like:

```cc
namespace Carbon {
class Circle : public Shape {
  double GetArea() override;
};
}  // namespace Carbon
```

To use the resulting class, C++ code can write:

```cc
#include "project/circle.carbon.h"

Shape* shape = new Carbon::Circle();
```

Note this parent inheritance will _only_ be visible to C++ code, so any attempts
to call a C++ function that takes the parent class, such as
`void Draw(Shape* shape)`, will need to be called from a C++ bridge function,
not Carbon.

##### Caveat: Missing or conflicting method declarations

Note that, in this example, the Carbon `Circle` type definition is not natively
aware of the `Shape` type inheritance. For example, suppose `GetArea()` returned
an `Int64` on `Shape`, but `Float64` on `Circle`. This would mean the `Shape`
type had conflicting method declarations, and the conflict only becomes apparent
when the `Circle` type is compiled for C++. Similarly, if `Shape` had an
abstract method that `Circle` did not implement, it would only be seen by C++.

Users could address this by declaring the C++ parent as a mixin on the Carbon
type (and this may generally be desirable, for consistent functionality). That
would pull C++ `Shape` method declarations over to Carbon. However, that doesn't
work for C++ classes with abstract methods, per
[the above caveat](#bookmark=kix.2nvh3xt3blqe).

Instead, the Carbon compiler should detect this situation. The indicated parent
class is given, so it should be able to ensure method declarations match.

##### Caveat: Concrete methods on parents

If the parent has a concrete method definition, note the \$extern chain does
_not_ result in that method being present on the Carbon-native version of
`Circle`. i.e., `Shape::MakeUnion` is only present on the C++ version of
`Circle`, not the Carbon version of `Circle`. This is a trade-off to allow use
of C++ inheritance for interoperability.

##### Alternative: Require bridge code for inheritance

Instead of providing parent support in \$extern, we could instead require users
to write bridge code.

In the `Shape` example, that would require Carbon code:

```
$extern("Cpp") struct Circle {
  fn GetArea() -> Float64 { ... };
}
```

And bridging C++ code:

```
#include "project/circle.carbon.h"

class CircleWrapper : public Shape {
 public:
  double GetArea() override { return circle_.GetArea(); }

 private:
  ::Carbon::Circle circle_;
};
```

Pros:

- Reduces complexity of the interop layer.
- Avoids caveats about missing/conflicting method declaration and concrete
  parent method.

Cons:

- Inheritance is a common operation in C++, so users should be expected to write
  this bridge code frequently.
- May cause significant friction for inheritance with multiple parent methods.

#### Implementing Carbon interfaces in C++

Carbon interfaces expected to be implemented in C++ should have an explicit C++
interface to operate as a bridge. Similarly, a layer of wrapping should be used
to bridge between languages when carrying over a type hierarchy from one to the
other.

e.g., given the Carbon code:

```
package Art;

interface Shape {
  fn Draw();
}
```

To implement a Square in C++, write the corresponding C++ implementation:

```
class Square {
 public:
  void Draw() { ... }
};
```

Then provide a bridge struct in Carbon:

```
package Art;

struct SquareBridge {
  impl Shape;
  mixin Cpp.Square;
};
```

#### Public non-virtual inheritance

C++ type hierarchies formed with public non-virtual inheritance should be
modeled in the types exposed to Carbon with equivalent behavior where
appropriate (conversions, etc). Since extending the type hierarchy from Carbon
isn't supported, this is expected to be a minimal surface. Carbon types exported
to C++ simply aren't allowed to have any visible type hierarchy. When exposing a
C++ hierarchy is necessary, it should be defined using C++ wrapper types in
bridge code.

e.g., consider the C++ code:

```
class Shape {
 public:
  virtual double GetArea() = 0;
};
class Circle : public Shape {
 public:
  double GetArea() override;
  double GetRadius();
};
class Square : public Shape {};

void Draw(Shape* shape);
```

This will be imported to Carbon, based on references, as:

```
package Cpp;

struct Shape {
  fn GetArea() -> Float64;
}
struct Circle {
  fn GetArea() -> Float64;
  fn GetRadius() -> Float64;
}
struct Square {
  fn GetArea() -> Float64;
}

fn Draw(var Shape*?: shape);
// Overrides are generated to model the inheritance if they're called.
fn Draw(var Circle*?: shape);
fn Draw(var Square*?: shape);
```

##### Alternative: Simulate interfaces for C++ types

We could simulate interfaces for all C++ types, generating (for example) a
`NAME_CppInterface` interface for all classes and structs. This must be done for
every non-final class because we don't know if the class may be inherited from
later.

In order to work correctly, this would require using the interface type for all
non-owning references and pointers. However, that may yield compatibility
issues, e.g. if a pointer is dereferenced to create a copy passed into an API.

e.g., given the C++ code:

```
class Shape {
 public:
  virtual double GetArea() = 0;
};
class Circle : public Shape {
 public:
  double GetArea() override;
  double GetRadius();
};
class Square : public Shape {};

void Draw(Shape* shape);
Circle* MakeCircle(int radius);
void Print(Circle circle);
```

This would mean the imported results would look more like:

```
package Cpp;

interface Shape_CppInterface {
  fn GetArea() -> Float64;
}
struct Shape {
  impl ShapeInterface { ... }
}

interface Circle_CppInterface extends Shape_CppInterface {
  fn GetRadius() -> Float64;
}
struct Circle {
  impl Circle_CppInterface { ... }
}

interface Square_CppInterface extends Shape_CppInterface {}
struct Square {
  impl Square_CppInterface { ... }
}

fn Draw(var Shape_CppInterface*?: shape);
Circle_CppInterface* MakeCircle(Int64 radius);
void Print(Circle circle);
```

Pros:

- Interfaces echo C++ inheritance chain.

Cons:

- The separation of the concrete type and the interface used in APIs may confuse
  users and cause problems for APIs. e.g., `Print(*MakeCircle(radius))` would be
  valid for C++, but not possible through the Carbon wrapper because of the
  `Circle` vs `Circle_CppInterface` difference.

#### Virtual and Non-Public Inheritance

Virtual inheritance and non-public inheritance from C++ are not made visible in
Carbon in any way. For virtual inheritance, the type API is flattened into each
derived type used from Carbon code and conversions are not supported.

### Unions and transparent union members

C/C++ includes both unions and transparent union members within classes. Carbon
doesn't have unions, creating an incompatibility.

We will expose C/C++ union structs through method-based APIs. If we add a
similar Carbon feature, we would expect it to similarly be encapsulated by an
API instead of being directly exposed.

e.g., `signal.h` has a union API:

```
union sigval {
  int sival_int;
  void *sival_ptr;
};
struct sigevent {
  ...
  union sigval sigev_value;
  ...
};
```

This will be accessible in Carbon through wrapper getter/setter APIs:

```
package C;

struct sigval {
  fn get_sival_int() -> Int64;
  fn set_sival_int(var Int64: val);
  fn get_sival_ptr() -> Void*?;
  fn set_sival_ptr(Void*?);

  // The size of the struct needs to match, but the actual type shouldn't matter.
  private var Byte[8]: storage;
};
struct sigevent {
  ...
  sigval sigev_value;
  ...
};
```
