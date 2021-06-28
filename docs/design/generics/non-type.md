# Carbon non-type generics

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Non-type generics v1

Imagine we had a regular function that printed some number of 'X' characters:

```
fn PrintXs_Regular(n: Int) {
  var i: Int = 0;
  while (i < n) {
    Print("X");
    i += 1;
  }
}

PrintXs_Regular(1); // Prints: X
PrintXs_Regular(2); // Prints: XX
var n: Int = 3;
PrintXs_Regular(n); // Prints: XXX
```

### Basic generics

What would it mean to change the parameter to be a generic parameter?

```
fn PrintXs_Generic(N:$ Int) {
  var i: Int = 0;
  while (i < N) {
    Print("X");
    i += 1;
  }
}

PrintXs_Generic(1);  // Prints: X
PrintXs_Generic(2);  // Prints: XX
var m: Int = 3;
PrintXs_Generic(m);  // Compile error: value for generic parameter `n`
                     // unknown at compile time.
```

For the definition of the function there is only one difference: we added a `$`
to indicate that the parameter named `n` is generic. The body of the function
type checks using the same logic as `PrintXs_Regular`. However, callers must be
able to know the value of the argument at compile time. This allows the compiler
to adopt a code generation strategy that creates a separate copy of the
`PrintXs_Generic` function for each combination of values of the generic (and
template) arguments, called [static specialization](goals.md#dispatch-control).
In this case, this means that the compiler can generate different binary code
for the calls passing `n==1` and `n==2`. Knowing the value of `n` at code
generation time allows the optimizer to unroll the loop, so that the call
`PrintXs_Generic(2)` could be transformed into:

```
Print("X");
Print("X");
```

Since we know the generic parameter is restricted to values known at compile
time, we can use the generic parameter in places we would expect a compile-time
constant value, such as in types.

```
fn CreateArray(N:$ UInt, value: Int) -> FixedArray(Int, N) {
  var ret: FixedArray(Int, N);
  var i: Int = 0;
  while (i < N) {
    ret[i] = value;
    i += 1;
  }
  return ret;
}
```

**Comparison with other languages:** This feature is part of
[const generics in Rust](https://blog.rust-lang.org/2021/02/26/const-generics-mvp-beta.html).

## v2

### Non-type generics have utility

Although the examples above mostly use deduced parameters and interfaces,
without them generics can also provide useful behavior.

Imagine we had a regular function that printed some number of 'X' characters:

```
fn PrintXs_Regular(n: Int) {
  var i: Int = 0;
  while (i < n) {
    Print("X");
    i += 1;
  }
}

PrintXs_Regular(1); // Prints: X
PrintXs_Regular(2); // Prints: XX
var n: Int = 3;
PrintXs_Regular(n); // Prints: XXX
```

What would it mean to change the parameter to be a generic parameter?

```
fn PrintXs_Generic(N:$ Int) {
  var i: Int = 0;
  while (i < N) {
    Print("X");
    i += 1;
  }
}

PrintXs_Generic(1);  // Prints: X
PrintXs_Generic(2);  // Prints: XX
var m: Int = 3;
PrintXs_Generic(m);  // Compile error: value for generic parameter `n`
                     // unknown at compile time.
```

The body of the function type checks using the same logic as `PrintXs_Regular`.
However, callers must be able to know the value of the argument at compile time.
This allows the compiler to adopt a code generation strategy that creates a
separate copy of the `PrintXs_Generic` function for each combination of values
of the generic (and template) arguments, called
[static specialization](goals.md#dispatch-control). In this case, this means
that the compiler can generate different binary code for the calls passing `1`
or `2` for `N`. Knowing the value of `N` at code generation time allows the
optimizer to unroll the loop, so that the call `PrintXs_Generic(2)` could be
transformed into:

```
Print("X");
Print("X");
```

Since the generic parameter is restricted to values known at compile time, the
generic parameter can be used in places a compile-time constant value would be
expected, including in types.

```
fn CreateArray(N:$ UInt, value: Int) -> FixedArray(Int, N) {
  var ret: FixedArray(Int, N);
  var i: Int = 0;
  while (i < N) {
    ret[i] = value;
    i += 1;
  }
  return ret;
}
```

**Comparison with other languages:** This feature is part of
[const generics in Rust](https://blog.rust-lang.org/2021/02/26/const-generics-mvp-beta.html).

### Dynamic parameters may be used as explicit types

**Open question:** Are regular dynamic parameters like `n` here allowed to be
used in type expressions?

In this hypothetical example, `n` is not a generic despite being deduced:

```
fn PrintArraySize[n: Int](array: FixedArray(String, n)*) {
  Print(n);
}

var a: FixedArray(String, 3) = ...;
PrintArraySize(&a);  // Prints: 3
```

FIXME: Garden-path sentence: What happens here is the type for the `array`
parameter is determined from the value passed in, and the pattern-matching
process used to see if the types match finds that it does match if `n` is set to
`3`. So this is equivalent to:

```
fn PrintN(n: Int) {
  Print(n);
}
PrintN(a.Size);
```

FIXME: Fix transition: Normally you would declare an implicit parameter as a
generic or template instead of a regular parameter. This avoids overhead from
having to support types (like the type of `array` inside the `PrintArraySize`
function body) that are only fully known with dynamic information. For example:

```
fn PrintStringArray[n:$ Int](array: FixedArray(String, n)*) {
  var i: Int = 0;
  while (i < n) {
    Print(array->get(i));
    ++i;
  }
}
```

## ALSO: Local constants

You may also have local generic constants as members of types. Just like generic
parameters, they have compile-time, not runtime, storage. You may also have
template constant members, with the difference that template constant members
can use the actual value of the member in type checking. In both cases, these
can be initialized with values computed from generic/template parameters, or
other things that are effectively constant and/or available at compile time.

We also support local generic constants in functions:

```
fn PrintOddNumbers(Int:$ N) {
  // last_odd is computed and stored at compile time.
  var Int:$ LastOdd = 2 * N - 1;
  var Int: i = 1;
  while (i <= LastOdd) {
    Print(i);
    i += 2;
  }
}
```

FIXME Local template constants may be used in type checking:

```
fn PrimesLessThan(Int:$$ N) {
  var Int:$$ MaxNumPrimes = N / 2;
  // Value of MaxNumPrimes is available at type checking time.
  var FixedArray(Int, MaxNumPrimes): primes;
  var Int: num_primes_found = 0;
  // ...
}
```

Interfaces may include requirements that a type's implementation of that
interface have local constants with a particular type and name. These are called
FIXME: [associated constants](details.md#associated-constants).
