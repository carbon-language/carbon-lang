# Type mapping

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Table of contents

<!-- toc -->

- [Overview](#overview)
- [Primitive types](#primitive-types)
  - [32-bit vs 64-bit and platform compatibility](#32-bit-vs-64-bit-and-platform-compatibility)
    - [**Alternative: Supplement mappings with platform-compatible conversion APIs**](#alternative-supplement-mappings-with-platform-compatible-conversion-apis)
    - [**Alternative: Provide variable size types**](#alternative-provide-variable-size-types)
  - [size_t and signed vs unsigned](#size_t-and-signed-vs-unsigned)
    - [**Alternative: Map size_t to UInt64**](#alternative-map-size_t-to-uint64)
  - [char/unsigned char and byte vs character](#charunsigned-char-and-byte-vs-character)
    - [**Alternative: Support + and - on Byte**](#alternative-support--and---on-byte)
    - [**Alternative: Create a Char8 type**](#alternative-create-a-char8-type)
    - [**Alternative: Use Int8**](#alternative-use-int8)
- [User-defined class types](#user-defined-class-types)
  - [Inheritance](#inheritance)
    - [_Using C++ types as mixins for Carbon structs_](#_using-c-types-as-mixins-for-carbon-structs_)
      - [**Caveat: Abstract methods**](#caveat-abstract-methods)
    - [_Inheriting from C++ types with Carbon structs_](#_inheriting-from-c-types-with-carbon-structs_)
      - [**Caveat: Missing or conflicting method declarations**](#caveat-missing-or-conflicting-method-declarations)
      - [**Caveat: Concrete methods on parents**](#caveat-concrete-methods-on-parents)
      - [**Alternative: Require bridge code for inheritance**](#alternative-require-bridge-code-for-inheritance)
    - [_Implementing Carbon interfaces in C++_](#_implementing-carbon-interfaces-in-c_)
    - [_Public non-virtual inheritance_](#_public-non-virtual-inheritance_)
      - [**Alternative: Simulate interfaces for C++ types**](#alternative-simulate-interfaces-for-c-types)
    - [_Virtual and Non-Public Inheritance_](#_virtual-and-non-public-inheritance_)
  - [Templates](#templates)
    - [_C++ templates_](#_c-templates_)
    - [_Using Carbon generics/templates with C++ types in Carbon code_](#_using-carbon-genericstemplates-with-c-types-in-carbon-code_)
    - [_Using Carbon templates from C++_](#_using-carbon-templates-from-c_)
      - [**Alternative: Require bridge code**](#alternative-require-bridge-code)
    - [_Using Carbon generics from C++_](#_using-carbon-generics-from-c_)
  - [Unions and transparent union members](#unions-and-transparent-union-members)
- [User-defined enumerations](#user-defined-enumerations)
  - [C/C++ enums in Carbon](#cc-enums-in-carbon)
  - [Carbon enums in C/C++](#carbon-enums-in-cc)
- [Vocabulary types](#vocabulary-types)
  - [Non-owning value types](#non-owning-value-types)
  - [Non-owning references and pointers](#non-owning-references-and-pointers)
    - [_Slice special-casing_](#_slice-special-casing_)
    - [_Mapping similar built-in types_](#_mapping-similar-built-in-types_)
  - [Ownership transfer types](#ownership-transfer-types)
    - [**Alternative: Bind tightly to particular C++ libraries**](#alternative-bind-tightly-to-particular-c-libraries)
  - [Copying vocabulary types](#copying-vocabulary-types)

<!-- tocstop -->

## Overview

Carbon and C++ (as well as the C subset of C++) will have a number of types with
direct mappings between the languages. The existence of these mappings allow
switching from one type to another across any interface boundary between the
languages. However, this only works across the interface boundary to avoid any
aliasing or other concerns. Transitioning in or out of Carbon code is what
provides special permission for these type aliases to be used.

Also note that the behavior of these types will not always be identical between
the languages. It is only the values that transparently map from one to the
other. Mapping operations is significantly different. e.g., Carbon may have
`Float32` match the C++ `float` storage while making subtle changes to
arithmetic and/or comparison behaviors. We will prioritize reflecting the intent
of type choices.

Last but not least, in some cases, there are multiple C or C++ types that can
map to a single Carbon type. This is generally "fine" but may impose important
constraints around overload resolution and other C++ features that would
otherwise "just work" due to these mappings.

## Primitive types

Note: In all cases where the C / C++ type comes in both `::T` and `std::T` forms
in C++, both are implied by writing `T` below. Also, when mapping from Carbon to
C/C++ and there are multiple C/C++ type candidates, the first one in the list is
intended to be the canonical answer.

Note: this is not intended to be exhaustive, but indicative.

<table>
  <tr>
   <td>Carbon
   </td>
   <td>C / C++
   </td>
  </tr>
  <tr>
   <td><code>Void</code>
   </td>
   <td><code>void</code>
   </td>
  </tr>
  <tr>
   <td><code>Byte</code>
   </td>
   <td><code>unsigned char</code>, <code>char</code>, <code>std::byte</code>
   </td>
  </tr>
  <tr>
   <td><code>Bool</code>
   </td>
   <td><code>_Bool</code>, <code>bool</code>
   </td>
  </tr>
  <tr>
   <td><code>Int8</code>
   </td>
   <td><code>int8_t</code>, <code>signed char</code>
   </td>
  </tr>
  <tr>
   <td><code>Int16</code>
   </td>
   <td><code>int16_t</code>, <code>short</code>
   </td>
  </tr>
  <tr>
   <td><code>Int32</code>
   </td>
   <td><code>int32_t, int</code>
<p>
If 32-bit: <code>long</code>, <code>ptrdiff_t</code>, <code>size_t</code>, <code>rsize_t</code>, <code>ssize_t</code>
   </td>
  </tr>
  <tr>
   <td><code>Int64</code>
   </td>
   <td><code>int64_t</code>, <code>long long</code>, <code>intptr_t</code>
<p>
If 64-bit: <code>long</code>, <code>ptrdiff_t</code>, <code>size_t</code>, <code>rsize_t</code>, <code>ssize_t</code>
   </td>
  </tr>
  <tr>
   <td><code>Int128</code>
   </td>
   <td><code>int128_t</code>
   </td>
  </tr>
  <tr>
   <td><code>UInt8</code>
   </td>
   <td><code>uint8_t</code>
   </td>
  </tr>
  <tr>
   <td><code>UInt16</code>
   </td>
   <td><code>uint16_t</code>, <code>unsigned short</code>
   </td>
  </tr>
  <tr>
   <td><code>UInt32</code>
   </td>
   <td><code>uint32_t</code>, <code>unsigned int</code>
<p>
If 32-bit: <code>unsigned long</code>
   </td>
  </tr>
  <tr>
   <td><code>UInt64</code>
   </td>
   <td><code>uint64_t</code>, <code>unsigned long long</code>, <code>uintptr_t</code>
<p>
If 62-bit: <code>unsigned long, unsigned long</code>
   </td>
  </tr>
  <tr>
   <td><code>Float16</code>
   </td>
   <td><code>short float</code> (hopefully)
   </td>
  </tr>
  <tr>
   <td><code>Float32</code>
   </td>
   <td><code>float</code>
   </td>
  </tr>
  <tr>
   <td><code>Float64</code>
   </td>
   <td><code>double</code>
   </td>
  </tr>
</table>

### 32-bit vs 64-bit and platform compatibility

At present, the proposed translation for these types to Carbon is based on the
corresponding platform-specific size that C++ uses. e.g.:

<table>
  <tr>
   <td>
   </td>
   <td>LP32
   </td>
   <td>ILP32
   </td>
   <td>LLP64
   </td>
   <td>LP64
   </td>
  </tr>
  <tr>
   <td><code>int</code>
   </td>
   <td><code>Int16</code>
   </td>
   <td><code>Int32</code>
   </td>
   <td><code>Int32</code>
   </td>
   <td><code>Int32</code>
   </td>
  </tr>
  <tr>
   <td><code>long</code>
   </td>
   <td><code>Int32</code>
   </td>
   <td><code>Int32</code>
   </td>
   <td><code>Int32</code>
   </td>
   <td><code>Int64</code>
   </td>
  </tr>
  <tr>
   <td><code>pointer</code>
   </td>
   <td><code>Int32</code>
   </td>
   <td><code>Int32</code>
   </td>
   <td><code>Int64</code>
   </td>
   <td><code>Int64</code>
   </td>
  </tr>
  <tr>
   <td><code>size_t</code>
   </td>
   <td><code>Int32</code>
   </td>
   <td><code>Int32</code>
   </td>
   <td><code>Int64</code>
   </td>
   <td><code>Int64</code>
   </td>
  </tr>
</table>

In practice, we are most worried about differences between LP64 platforms (the
vast majority of 64-bit Unix-like operating systems, including Linux) and LLP64
(Windows x86_64). If Carbon will support 32-bit CPUs, we are interested in the
ILP32 model, which is adopted by the vast majority of 32-bit platforms.

Similarly, `float` and `double` may end up being different sizes on particular
platforms.

When writing cross-platform, cross-language code, engineers will need to be
sensitive to what size they use in Carbon vs what size C++ would use.

Pros:

- Types always map to a Carbon type of an explicit, equal size.

Cons:

- Portability issues due to differences between C types across platforms. As a
  result, a given C or C++ API will be imported into Carbon differently
  depending on the selected target. Carbon code would have to be written in such
  a way as to compile in all modes.

#### **Alternative: Supplement mappings with platform-compatible conversion APIs**

Carbon could provide conversion APIs, e.g. `ToCLong`, to improve portability.

i.e.:

```
package CppCompat;

$if platform == LP64
fn ToCLong(var Int64: val) -> Int64 { return val; }
$else
fn ToCLong(var Int64: val) -> Int32 { return (Int32)val; }
$endif

var Int64: myVal = Foo();
// retVal is always safe because an Int32 can always become an Int64.
var Int64: retVal = Cpp.ApiUsingLong(CppCompat.ToCLong(val));
```

Pros:

- Reduces the amount of platform-specific code that authors need to provide.
- Platform-specific conversions are clearly annotated as being related to
  compatibility with specific C types.

Cons:

- Implies type conversions on certain platforms, with performance overhead.
  - Users may still need to write platform-specific code.
- The code uses explicitly-sized types, so users have to compile code for all
  target platforms to see all possible errors.

#### **Alternative: Provide variable size types**

Carbon could provide compatibility types with matching sizes to the C++
implementation.

i.e.:

```
package CppCompat;

$if platform == LP64
struct CLong { private var Bytes[8]: data; }
$else
struct CLong { private var Bytes[4]: data; }
$endif

// This line will fail to compile if, e.g., Foo returns an Int64 while
// CLong is 32-bit.
var CLong: myVal = (CLong)Foo();
var CLong: retVal = Cpp.ApiUsingLong(val);
```

Pros:

- Reduces the amount of platform-specific code that authors need to provide.

Cons:

- Most Carbon code is still expected to use explicit sizes. Platform-specific
  code should still be expected to crop up around APIs that expect a particular
  size.

### size_t and signed vs unsigned

At present, the proposal is that `size_t` will map to the signed `Int64` type.

Pros:

- Idiomatically represent memory sizes, container lengths, etc as `Int64`.

Cons:

- Does not match `size_t` unsigned semantics.

#### **Alternative: Map size_t to UInt64**

We could alternatively use `UInt64` for the mapping.

Pros:

- Matches C++ semantics.

Cons:

- Pushes engineers to use unsigned types when talking about lengths, risking
  errors with negative values and/or comparisons.

### char/unsigned char and byte vs character

At present, the proposal is that `char`/`unsigned char` should map to `Byte`.
`Byte` is distinct because, while `Int8`/`UInt8` has arithmetic, `Byte` is
intended to not have arithmetic.

Pros:

- C/C++ use char types in some cases when dealing with memory, because there
  hasn't historically been a dedicated byte type.

Cons:

- Users may do character arithmetic on `char`.
  - Using an integer offset to get a letter. e.g., `'A' + 15` as a way to get
    the value `'P'`.
  - Using `32` to capitalize. e.g., `'a' + 32` as a way to get the value `'A'`.

#### **Alternative: Support + and - on Byte**

We could plan on supporting basic `+` and `-` on `Byte`.

Pros:

- Keeps the `Byte` translation, which may be more appropriate for some APIs.

Cons:

- Adds arithmetic operations to the `Byte` type, which may be inappropriate for
  actual memory representation.

#### **Alternative: Create a Char8 type**

We could add a `Char8` type, specifically limiting it to a single byte,
mirroring C++. Note this is `Char8` because we'll presumably have `Char32` for
UTF-32, and possibly `Char` to indicate a multi-byte Unicode character.

Pros:

- Mirrors the C++ semantic.
- Allows for character-specific behaviors, e.g. when printing values to stdout.

Cons:

- Prevents us from representing C++ memory operations as `Byte` without a
  specific type mapping.

#### **Alternative: Use Int8**

We could convert `char` to `Int8`.

Pros:

- Makes all `char` types an `Int8`.

Cons:

- It's likely that we want to provide more character-like behaviors than `Int8`
  could offer.
- We probably don't want operators like `*` or `/` to provide integer arithmetic
  for a character.

## User-defined class types

All (non-template) user-defined C++ class types (and C struct types) are
directly available within Carbon with the exact C++ layout. Member function
semantics (including special member function semantics) are outlined below, but
the data is directly available.

i.e., given a simple C/C++ class:

```
class Circle {
 public:
  double GetArea();

 private:
  double radius_;
};
```

We expect this to behave as a similar Carbon class:

```
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

#### _Using C++ types as mixins for Carbon structs_

Assuming we provide mixins on Carbon structs, we should be able to safely allow
C++ types as mixins on Carbon structs. This would allow C++ types to provide
APIs directly, without a Carbon wrapper.

This assumes there are no other implications for Carbon or C++
inheritance-related features. If there are, we should reconsider this.

e.g., consider the C++ code:

```
class Shape {
 public:
  double GetArea();
};
```

The Carbon code should be able to reuse the class's implementation:

```
struct ShapeWrapper {
  mixin Cpp.Shape;
}

ShapeWrapper x = MakeShapeWrapper();
x.GetArea();
```

However, this has no interface implications for ShapeWrapper. Also, if used from
C++, ShapeWrapper would not be considered to inherit from Shape. Additional
language-specific wrappers would be required to get inheritance functionality.

##### **Caveat: Abstract methods**

C++ abstract methods will not have a direct equivalent in Carbon. Although a
Carbon interface is similar to a C++ class with only abstract methods, it's not
clear that it's useful or helpful to translate such types as interfaces. A
Carbon struct should only have concrete methods.

A C++ class with abstract methods cannot be instantiated, as in a mixin. Users
should expect a compiler error in such a case, similar to if they attempted to
instantiate the class in Carbon code in general.

#### _Inheriting from C++ types with Carbon structs_

Inheritance from C++ types cannot be straightforward because Carbon will not
have matching inheritance concepts. However, we can augment Carbon structs to
indicate an inheritance chain that works similarly.

e.g., consider the C++ code:

```
class Shape {
 public:
  virtual double GetArea() = 0;
  Shape* MakeUnion(Shape* other) { ... };
};
```

In order to provide an implementation in Carbon, we can then write code like:

```
import Cpp "project/shape.h"

$extern("Cpp", parent="Cpp.Shape") struct Circle {
  fn GetArea() -> Float64 { ... };
}
```

This will create code like:

```
namespace Carbon {
class Circle : public Shape {
  double GetArea() override;
};
}  // namespace Carbon
```

To use the resulting class, C++ code can write:

```
#include "project/circle.carbon.h"

Shape* shape = new Carbon::Circle();
```

Note this parent inheritance will _only_ be visible to C++ code, so any attempts
to call a C++ function that takes the parent class (e.g.,
`void Draw(Shape* shape)`) will need to be called from a C++ bridge function,
not Carbon.

##### **Caveat: Missing or conflicting method declarations**

Note that, in this example, the Carbon Circle type definition is not natively
aware of the Shape type inheritance. That means, if the Shape type had
conflicting method declarations (e.g., GetArea() returned an Int64 on Shape, but
Float64 on Circle), the conflict only becomes apparent when the Circle type is
compiled for C++. Similarly, if Shape had an abstract method that Circle did not
implement, it would only be seen by C++.

Users could address this by declaring the C++ parent as a mixin on the Carbon
type (and this may generally be desirable, for consistent functionality). That
would pull C++ Shape method declarations over to Carbon. However, that doesn't
work for C++ classes with abstract methods, per
[the above caveat](#bookmark=kix.2nvh3xt3blqe).

Instead, the Carbon compiler should detect this situation. The indicated parent
class is given, so it should be able to ensure method declarations match.

##### **Caveat: Concrete methods on parents**

If the parent has a concrete method definition, note the \$extern chain does
_not_ result in that method being present on the Carbon-native version of
Circle. i.e., `Shape::MakeUnion` is only present on the C++ version of Circle,
not the Carbon version of Circle. This is a trade-off to allow use of C++
inheritance for interoperability.

##### **Alternative: Require bridge code for inheritance**

Instead of providing parent support in \$extern, we could instead require users
to write bridge code.

In the Shape example, that would require Carbon code:

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

#### _Implementing Carbon interfaces in C++_

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

#### _Public non-virtual inheritance_

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

##### **Alternative: Simulate interfaces for C++ types**

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

#### _Virtual and Non-Public Inheritance_

Virtual inheritance and non-public inheritance from C++ are not made visible in
Carbon in any way. For virtual inheritance, the type API is flattened into each
derived type used from Carbon code and conversions are not supported.

### Templates

#### _C++ templates_

Simple C++ class templates are directly made available as Carbon templates. For
example, ignoring allocators and their associated complexity,
`std::vector&lt;T>` in C++ would be available as `Cpp.std.vector(T)` in Carbon.

Instantiating C++ templates with a Carbon type requires that type to be made
available to C++ code and the instantiation occurs against the C++ interface to
that Carbon type. e.g.:

```
package Art;

import Cpp "<vector>";

// Extern the interface so that template code can see it:
$extern("Cpp") interface Shape {
  fn Draw();
}

// Then instantiate the template:
var Cpp.std.vector(Shape): shapes;
```

More complex C++ class templates may need to be explicitly instantiated using
bridge C++ code to explicitly provide Carbon types visible within C++ to the
appropriate C++ template parameters. The key principle is that C++ templates are
instantiated within C++ against a C++-visible API for a given Carbon type.

#### _Using Carbon generics/templates with C++ types in Carbon code_

Any C++ type can be used as a type parameter in Carbon. However, it will be
interpreted as Carbon code; e.g., if there are any requirements for Carbon
interfaces, [bridge code will be required](#bookmark=kix.8fx2t4lplthb).

#### _Using Carbon templates from C++_

We plan to modify Clang to allow for extensions that will use Carbon to compile
the template then insert the results into Clang's AST for expansion.

This assumes low-level modifications to LLVM. We acknowledge this would be
necessary, and may gate such a feature.

##### **Alternative: Require bridge code**

We could require bridge code that explicitly instantiates versions of the
template for use with C++ types.

Pros:

- Avoids modifications to Clang.

Cons:

- Requires extra code to use templates from C++, making it harder to migrate
  code to Carbon.

#### _Using Carbon generics from C++_

Using Carbon generics from C++ code will require bridge Carbon code that hides
the generic. Note this could be wrapping the generic with a template.

For example, given the Carbon code:

```
fn GenericAPI[Foo:$ T](T*: x) { ... }

fn TemplateAPI[Foo:$$ T](T* x) { GenericAPI(x); }
```

We could have C++ code that uses the template wrapper to use the generic:

```
CppType y;
::Carbon::TemplateAPI(&y);
```

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

## User-defined enumerations

We expect enums can be represented directly in the other language. All values in
the copy should be assumed to be explicit, to prevent any possible issues in
enum semantics.

### C/C++ enums in Carbon

Given an enum:

```
enum Direction {
  East,
  West = 20,
  North,
  South,
};
```

We would expect to generate equivalent Carbon code:

```
enum Direction {
  East = 0,
  West = 20,
  North = 21,
  South = 22,
}

// Calling semantic:
var Direction: x = Direction.East;
```

Sometimes enum names may repeat the enum identifier, e.g., `DIRECTION_EAST`
instead of `East`. To help with this case, we may want to support renaming of
enum entries. e.g., to rename in a way that results in a match to the above
Carbon calling convention:

```
enum Direction {
  DIRECTION_EAST,
  DIRECTION_WEST,
  DIRECTION_NORTH,
  DIRECTION_SOUTH,
} __attribute__((carbon_enum("East:West:North:South"));
```

If using enum class, we'd expect similar behavior:

```
enum class Direction : char {
  East = 'E',
  West = 'W',
  North = 'N',
  South = 'S',
};
```

With Carbon code:

```
enum(Byte) Direction {
  East = 'E',
  West = 'W',
  North = 'N',
  South = 'S',
};
```

### Carbon enums in C/C++

Given an enum:

```
$extern("Cpp") enum Direction {
  East,
  West = 20,
  North,
  South,
}
```

Because Carbon automatically groups enums, we would expect to generate
equivalent C++ code:

```
enum class Direction {
  East = 0,
  West = 20,
  North = 21,
  South = 22,
};
```

## Vocabulary types

There are several cases of vocabulary types that are important to consider and
offer different degrees of flexibility in support:

- Non-owning types passed by value (`std::string_view`, `std::span`, ...)
- Non-owning types passed by reference (or pointer) (`std::vector&lt;T> &`, ...)
- Owning types signaling a move of ownership (`std::unique_ptr`,
  `std::vector&lt;T> &&`, ...)
- Owning types signaling a copy of data (`std::vector&lt;T>`, ...)

Each of these lends itself to different strategies of interoperation between
Carbon and C++.

### Non-owning value types

These are some of the most important types to have direct support for in Carbon
because they are complex and opaque C++ types that are expected to come at zero
runtime cost and refer back to data that will remain managed by C/C++ code. As
such, Carbon prioritizes direct mappings for the most important of these types
at no cost.

The primary idiom from C++ that Carbon attempts to directly support are types
which represent contiguous data in memory. The dominant case here is
`std::span&lt;T>`, but there are a wide variety of similar vocabulary types.
Here, Carbon directly maps these types to a slice type: `T[]`. These types in
Carbon have the same core semantics and can be trivially built from any
pointer-and-size formulation.

There is an important special case: `std::string_view` and similar views into
C++ string data. It is an open question in Carbon whether there is a dedicated
`StringSlice` type or instead it simply uses a direct slice of the underlying
`CodeUnit` (or `Char`) type. Whatever is the canonical vocabulary used to convey
a slice of a Carbon `String` should also be used as the idiomatic mapping for
`std::string_view`.

Other non-owning value types will get automatic mappings as a use case is
understood to be sufficiently important. We expect the vast majority of
performance critical mappings to end up devolving to slices.

### Non-owning references and pointers

Non-owning references and pointers are perhaps the simplest non-owning types in
C and C++ and have the same critical performance requirements as value types.
Mapping them into Carbon is simple because they have limited and well-known
semantics. By default, these are both mapped to pointers in Carbon, with
references mapped to non-null pointers (`T*`) and C++ pointers mapped to
nullable pointers (`T*?`).

e.g., given a C++ API:

```
Resource* LoadResource(const Location& name);
ResourceEntry& SelectResource(const Selector& sel);
```

We would expect a Carbon call to look like:

```
var Cpp.Location: loc = ...;
This maps the C++ * to a nullable pointer.
var Cpp.Resource*?: res = Cpp.LoadResource(&loc);

var Cpp.Selector: sel = ...;
// This maps the C++ & to a non-nullable pointer.
var Cpp.ResourceEntry*: entry = Cpp.SelectResource(sel);
```

However, there are interesting special cases where it will be advantageous to
promote these types to higher-level types in Carbon to make the interface
boundary cleaner. The currently planned special cases are listed here, and more
can be added as we discover both a compelling need and an effective strategy.

#### _Slice special-casing_

Where possible to convert a reference or pointer (typically to a `const` type)
to a slice, Carbon will do so automatically. This should cover common patterns
such as `const std::vector&lt;T> &` -> `T[]` and `const std::vector&lt;T> *` ->
`T[]?`. Specific types that should provide this conversion:

- `const std::vector&lt;T>`
- `std::array&lt;T, N>` (This loses some info, can build a compile-time-length
  slice if needed)

#### _Mapping similar built-in types_

When it is not possible to convert a non-owning reference or pointer to a C++
data structure or vocabulary type into a suitable Carbon type, the actual C++
type will be used. However, its API may not match Carbon idioms or patterns, and
may not integrate with generic Carbon code written against those idioms (or vice
versa).

For sufficiently widely used C++ types, Carbon will provide non-owning wrappers
(preferably using generics) that map between the relevant idioms. This will be a
Carbon wrapper to map from a C++ data type like `std::map` into a Carbon
idiomatic interface, or a C++ wrapper to map from a Carbon data type to the C++
idiomatic interface.

The result is that Carbon data structures and vocabulary types should be no more
foreign in C++ code than Boost or other framework libraries that carefully
adhere to C++ idioms, and similarly C++ types in Carbon code.

### Ownership transfer types

Another special case that is important to optimize for is when ownership of data
is being transferred between C++ and Carbon. This can be tricky to recognize due
to reasonable use of pass-by-value when doing ownership transfer in C++, but
Carbon should recognize as many idioms as possible.

The most fundamental case to handle is `std::unique_ptr`, which fortunately is
easily recognized. It can only signify a transfer of ownership. Here Carbon
should completely translate this transfer of ownership from the C++ heap to the
Carbon heap, including to a heap-owned pointer in Carbon. This in turn requires
the C++ heap to be implemented as part of the Carbon heap, allowing allocations
in one to be deallocated in the other and vice versa. TODO(chandlerc): need to
spell out how the heap works in Carbon to pin down the details here.

The next case is `std::vector&lt;T> &&`. This should get translated to a
transfer of ownership with a Carbon `Array(T)`. These types may not have the
same layout, but it should be easy to migrate data from one to the other, even
in the presence of a small-size-optimized `Array(T)` by copying the data if
necessary. **NB:** that means that when transferring ownership into or out of
Carbon, there is the possibility of an extra copy. This does not precisely match
the contract of `std::vector` but should be a documented requirement for using
Carbon's automatic type mapping as it is expected to be correct in the
overwhelming majority of cases.

Other vocabulary types similar to `std::vector&lt;T>`, or more generally where
the allocation can be transferred or a small copy be performed, should get
similar automatic mapping with Carbon.

Vocabulary types with significantly more complex data structures are unlikely to
be efficiently convertible and should remain C++ types for efficient access.
Some Carbon types can provide explicit conversion routines (which may be
significantly more expensive such as re-allocating) when useful. A good example
here is `boost::unordered_map&lt;Key, Value>`. We likely do not want to
constrain a Carbon language-provided `Map(Key, Value)` type to match a
particular C++ library's layout and representation to allow trivial translation
from C++ to Carbon. However, this also likely isn't necessary. As we outline in
the philosophy above, neither C++ nor Carbon aim to reduce the prevalence of
custom C++ data structures. And we can still provide explicit (but potentially
expensive) conversions when ownership transfer is necessary (rather than using
the non-owning wrappers described previously).

#### **Alternative: Bind tightly to particular C++ libraries**

As an alternative, Carbon's language-provided types could precisely match the
internal representation and implementation of particular C++ libraries. In this
scenario, Carbon's `Map(Key, Value)` would need to precisely match
`boost::unordered_map&lt;Key, Value>`.

Note that this doesn't affect the ability of a Carbon program to use
`boost::unordered_map` independently; it only affects the ability of a Carbon
program to use `boost::unordered_map` in place of `Map`.

Pros:

- Users can convert a popular C++ type to an idiomatic Carbon type without
  conversion overhead.

Cons:

- Carbon could not break cross-language type compatibility without
  unpredictability affecting performance of applications that rely on
  compatibility.
- Carbon would be restricted when trying to evolve the API unless we could get
  the particular library (e.g., Boost) to change their implementation in a
  matching manner.
  - Carbon and C++ should be expected to have slightly different performance
    nuances: a performance improvement for Carbon might be a slowdown for C++.
- If the particular library (e.g., Boost) changed their implementation, it would
  either break compiles or corrupt Carbon programs.
  - Users would need to bind to specific releases of the library. We would
    become responsible for matching all development.

### Copying vocabulary types

When a vocabulary type crosses between C++ and Carbon and a copy is a valid
option, an extremely good interoperability story can be provided. Here, we can
in almost all cases completely convert common data structures and vocabulary
types between the languages. The data is being copied anyways and so any
necessary changes to the representation and layout are unlikely to be an
unacceptable overhead.

This strategy should be available for essentially all containers and copyable
vocabulary types in the C++ STL, Abseil, and any other sufficiently widely used
libraries.
