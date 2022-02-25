# MLIR Rationale

This document is intended to capture some of the alternatives considered and
open debates in the design of MLIR, along with the rationale for certain
decisions we made. This is not intended to be a "finely groomed" document - we
prefer the ability to dump in interesting tidbits without worrying too much
about their consistency or readability.

[TOC]

## Abstract

MLIR is a compiler intermediate representation with similarities to traditional
three-address SSA representations (like
[LLVM IR](http://llvm.org/docs/LangRef.html) or
[SIL](https://github.com/apple/swift/blob/master/docs/SIL.rst)), but which
introduces notions from the polyhedral loop optimization works as first class
concepts. This hybrid design is optimized to represent, analyze, and transform
high level dataflow graphs as well as target-specific code generated for high
performance data parallel systems. Beyond its representational capabilities, its
single continuous design provides a framework to lower from dataflow graphs to
high performance target specific code.

MLIR stands for one of "Multi-Level IR" or "Multi-dimensional Loop IR" or
"Machine Learning IR" or "Mid Level IR", we prefer the first. This document only
provides the rationale behind MLIR -- its actual
[specification document](../LangRef.md) and other content is hosted elsewhere.

## Introduction and Motivation

The Multi-Level Intermediate Representation (MLIR) is intended for easy
expression and optimization of computations involving deep loop nests and dense
matrices of high dimensionality. It is thus well-suited to deep learning
computations in particular. Yet it is general enough to also represent arbitrary
sequential computation. The representation allows high-level optimization and
parallelization for a wide range of parallel architectures including those with
deep memory hierarchies --- general-purpose multicores, GPUs, and specialized
neural network accelerators.

MLIR uses ideas drawn from IRs of LLVM and Swift for lower level constructs
while combining them with ideas from the polyhedral abstraction to represent
loop nests, multidimensional data (tensors), and transformations on these
entities as first class concepts in the IR.

MLIR is a multi-level IR, i.e., it represents code at a domain-specific
representation such as HLO or TensorFlow graphs, all the way down to the machine
level. MLIR is able to represent arbitrary control flow and arbitrary data
accesses, and is general enough to represent nearly all sequential computation.
This is a key distinction from existing polyhedral representation
implementations (such as LLVM [Polly](https://polly.llvm.org/)) that are able to
use the polyhedral abstraction in a way isolated from the LLVM IR and only for
affine loop nests, i.e., portions of the code where array accesses, loop bounds,
and conditionals are regular (involve linear functions of loop iterators and
constant symbols). The presence of statically unpredictable data accesses or
control flow does not preclude representation in MLIR, but only limits to a
certain extent the ability to reason about and apply transformations using the
polyhedral abstraction.

Maps, sets, and relations with affine constraints are the core structures
underlying a polyhedral representation of high-dimensional loop nests and
multidimensional arrays. These structures are represented as textual
expressions in a form close to their mathematical form. These structures are
used to capture loop nests, tensor data structures, and how they are reordered
and mapped for a target architecture. All structured or "conforming" loops are
captured as part of the polyhedral information, and so are tensor variables,
their layouts, and subscripted accesses to these tensors in memory.

The information captured in the IR allows a compact expression of all loop
transformations, data remappings, explicit copying necessary for explicitly
addressed memory in accelerators, mapping to pre-tuned expert-written
primitives, and mapping to specialized vector instructions. Loop transformations
that can be easily implemented include the body of affine transformations: these
subsume all traditional loop transformations (unimodular and non-unimodular)
such as loop tiling, interchange, permutation, skewing, scaling, relative
shifting, reversal, fusion, and distribution/fission. Transformations on data
layout such as padding and transforming to blocked layouts are also represented
well via affine layout maps.

MLIR's design allows a progressive lowering to target-specific forms. Besides
high-level transformations for loop nests and data layouts that a typical
mid-level optimizer is expected to deal with, MLIR is also designed to perform
certain low-level scheduling and mapping decisions that a typical backend IR is
entrusted with: these include mapping to specialized vector instructions,
auto-vectorization, and software pipelining. The need to support these
transformations stems from the fact that neural network accelerators have
specialized units that deal with large chunks of data whose computation maps
back to chunks of more than one loop of the loop nests as viewed by a program at
a level closer to the original specification. Such specialized units or
instructions operate on multidimensional data chunks from a programmer's
viewpoint. It thus makes it hard or infeasible for a backend operating on a very
low-level IR close to assembly to lift and reconstruct loops and perform such a
mapping. This is in contrast to classic instruction selection and scheduling in
today's compilers that primarily only deals with the body of the innermost loop.
MLIR also facilitates automatic mapping to expert pre-tuned primitives or vendor
libraries operating on data at higher levels (or at the highest level) of the
memory hierarchy.

In summary, MLIR is convenient for and closed under the kind of transformations
needed to lower to general-purpose as well as specialized accelerators. It also
allows one to build modular and reusable target independent and target dependent
passes.

## Design Decisions

This section sheds light on some of the design decisions -- some of these are
indirectly implied by the specification document.

### Loads and stores

The 'load' and 'store' instructions are specifically crafted to fully resolve to
an element of a memref. These instructions take as arguments n+1 indices for an
n-ranked tensor. This disallows the equivalent of pointer arithmetic or the
ability to index into the same memref in other ways (something which C arrays
allow for example). Furthermore, for the affine constructs, the compiler can
follow use-def chains (e.g. through
[affine.apply operations](../Dialects/Affine.md/#affineapply-affineapplyop)) or through
the map attributes of [affine operations](../Dialects/Affine.md/#operations)) to
precisely analyze references at compile-time using polyhedral techniques. This
is possible because of the [restrictions on dimensions and symbols](../Dialects/Affine.md/#restrictions-on-dimensions-and-symbols).

A scalar of element-type (a primitive type or a vector type) that is stored in
memory is modeled as a 0-d memref. This is also necessary for scalars that are
live out of for loops and if conditionals in a function, for which we don't yet
have an SSA representation --
[an extension](#affineif-and-affinefor-extensions-for-escaping-scalars) to allow that is
described later in this doc.

### Symbols and types

The current MLIR disallows use of symbols in types. For example, when a tensor
or memref dimension is statically unknown, it is denoted in the type as '?'. An
SSA symbol is then bound to it when a memref is created. The actual value of the
unknown dimension can be queried using the "dim" builtin as shown below.

Example:

```mlir
func foo(...) {
  %A = alloc <8x?xf32, #lmap> (%N)
  ...
  call bar(%A) : (memref<8x?xf32, #lmap>)
}

func bar(%A : memref<8x?xf32, #lmap>) {
  // Type of %A indicates that %A has dynamic shape with 8 rows
  // and unknown number of columns. The number of columns is queried
  // dynamically using dim instruction.
  %N = dim %A, 1 : memref<8x?xf32, #lmap>

  affine.for %i = 0 to 8 {
    affine.for %j = 0 to %N {
      // A[i,j] += 1
      %s1 = affine.load %A[%i, %j] : memref<8x?xf32, #lmap>
      %s2 = add %s1, 1
      affine.store %s2, %A[%i, %j] : memref<8x?xf32, #lmap>
    }
  }
  return
}

```

An alternative design is to embed the reference to symbols directly in the
type - memref<8x%Nxf32>. We went for the current approach in MLIR because it
simplifies the design --- types remain immutable when the values of symbols
change.

### Block Arguments vs PHI nodes

MLIR Regions represent SSA using "[block arguments](../LangRef.md/#blocks)" rather
than [PHI instructions](http://llvm.org/docs/LangRef.html#i-phi) used in LLVM.
This choice is representationally identical (the same constructs can be
represented in either form) but block arguments have several advantages:

1.  LLVM PHI nodes always have to be kept at the top of a block, and
    transformations frequently have to manually skip over them. This is defined
    away with BB arguments.
1.  LLVM has a separate function Argument node. This is defined away with BB
    arguments, because the arguments to the entry block serve this purpose.
1.  Blocks of PHI nodes in LLVM execute atomically, which is surprising and
    super confusing to compiler engineers and it is easy to introduce bugs with
    this (very related to the
    "[lost copy](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.524.5461&rep=rep1&type=pdf)"
    problem in SSA lowering literature.) With the BB argument representation,
    this confusion is defined away.
1.  The entry list of PHI nodes in LLVM are unordered, and some blocks have
    thousands of predecessors (e.g. unwind blocks). This can cause long compile
    time problems because transformations have to linearly scan this list. This
    is defined away with BB argument representation.
1.  LLVM has no way to represent values that are available only in one successor
    but not the other, e.g. its invoke instruction cannot produce the exception
    value JUST on the exception edge. Instead, the
    [landingpad instruction](http://llvm.org/docs/LangRef.html#landingpad-instruction)
    is a hack used to represent this. MLIR doesn't make use of this capability,
    but SIL uses it extensively, e.g. in the
    [switch_enum instruction](https://github.com/apple/swift/blob/master/docs/SIL.rst#switch-enum).

For more context, block arguments were previously used in the Swift
[SIL Intermediate Representation](https://github.com/apple/swift/blob/master/docs/SIL.rst),
and described in
[a talk on YouTube](https://www.youtube.com/watch?v=Ntj8ab-5cvE). The section of
interest
[starts here](https://www.google.com/url?q=https://youtu.be/Ntj8ab-5cvE?t%3D596&sa=D&ust=1529450150971000&usg=AFQjCNFQHEWL7m8q3eO-1DiKw9zqC2v24Q).

### Index type usage and limitations

Index types are intended to be used for platform-specific "size" values and may
appear in subscripts, sizes of aggregate types and affine expressions. They are
also tightly coupled with `affine.apply` and affine.load/store operations;
having `index` type is a necessary precondition of a value to be acceptable by
these operations.

We allow `index` types in tensors, vectors, and memrefs as a code generation
strategy has to map `index` to an implementation type and hence needs to be able
to materialize corresponding values. However, the target might lack support for
`vector` values with the target specific equivalent of the `index` type.

### Data layout of non-primitive types

Data layout information such as the bit width or the alignment of types may be
target and ABI-specific and thus should be configurable rather than imposed by
the compiler. Especially, the layout of compound or `index` types may vary. MLIR
specifies default bit widths for certain primitive _types_, in particular for
integers and floats. It is equal to the number that appears in the type
definition, e.g. the bit width of `i32` is `32`, so is the bit width of `f32`.
The bit width is not _necessarily_ related to the amount of memory (in bytes) or
the register size (in bits) that is necessary to store the value of the given
type. For example, `vector<3xi57>` is likely to be lowered to a vector of four
64-bit integers, so that its storage requirement is `4 x 64 / 8 = 32` bytes,
rather than `(3 x 57) ceildiv 8 = 22` bytes as can be naively computed from the
bit width. MLIR makes such [data layout information](../DataLayout.md)
configurable using attributes that can be queried during lowering, for example,
when allocating a compound type.

The data layout of dialect-specific types is undefined at MLIR level. Yet
dialects are free to define their own quantities and make them available via the
data layout infrastructure.

### Integer signedness semantics

Integers in the builtin MLIR type system have a bitwidth (note that the `index`
type has a symbolic width equal to the machine word size), and they *may*
additionally have signedness semantics. The purpose is to satisfy the needs of
different dialects, which can model different levels of abstractions. Certain
abstraction, especially closer to source language, might want to differentiate
signedness with integer types; while others, especially closer to machine
instruction, might want signless integers. Instead of forcing each abstraction
to adopt the same integer modelling or develop its own one in house, Integer
type provides this as an option to help code reuse and consistency.

For the standard dialect, the choice is to have signless integer types. An
integer value does not have an intrinsic sign, and it's up to the specific op
for interpretation. For example, ops like `addi` and `muli` do two's complement
arithmetic, but some other operations get a sign, e.g. `divis` vs `diviu`.

LLVM uses the [same design](http://llvm.org/docs/LangRef.html#integer-type),
which was introduced in a revamp rolled out
[in the LLVM 2.0 integer type](http://releases.llvm.org/2.0/docs/LangRef.html#t_derived).
Prior to that, from
[LLVM 1.0](http://releases.llvm.org/1.0/docs/LangRef.html#t_classifications) to
[1.9](http://releases.llvm.org/1.9/docs/LangRef.html#t_classifications), LLVM
uses signed types like "sbyte" and "ubyte". This shift was important and has
served LLVM well over the years. The reason this is important is that it is a
good thing for an intermediate representation to represent the same computation
with the same instruction. Signed types got in the way, because (e.g.) an "add
of an sbyte" does the same computation as an "add of a ubyte", but the type
system made them look artificially different. This split also required casts
like "cast from sbyte to ubyte" which do nothing at the machine level. Removing
signs from the type system eliminated these problems, making the compiler
simpler.

More information about this split is available in an old
[talk on youtube](https://www.youtube.com/watch?v=VeRaLPupGks) talking about
LLVM 2.0.

Note that this rationale only applies to the "standard ops" dialect in which we
can express an opinion about its design. Other dialects generally try to model
an external system, and should aim to reflect its design as closely as possible.

### Splitting floating point vs integer operations

The MLIR "standard" operation set splits many integer and floating point
operations into different categories, for example `addf` vs `addi` and `cmpf` vs
`cmpi`
([following the design of LLVM](http://llvm.org/docs/LangRef.html#binary-operations)).
These instructions _are_ polymorphic on the number of elements in the type
though, for example `addf` is used with scalar floats, vectors of floats, and
tensors of floats (LLVM does the same thing with its scalar/vector types).

This split is important because floating point and integer operations are quite
different in practice: for example, floating point values include NaN's, so
[integer comparisons](http://llvm.org/docs/LangRef.html#icmp-instruction) and
[floating point comparisons](http://llvm.org/docs/LangRef.html#fcmp-instruction)
should use different comparison opcodes. On the arithmetic side of things,
floating point operations support rounding modes, floating point contractions,
["fast math"](http://llvm.org/docs/LangRef.html#fadd-instruction), and integers
may want to have two's complement overflow behavior or be undefined on
[various forms of wrapping](http://llvm.org/docs/LangRef.html#add-instruction)
for performance.

We are a long way from this sort of thing being a priority to care about in
MLIR, but since we have experience and know the right way to do this, we'd
rather design it in from the beginning.

Note that this rationale only applies to the "standard ops" dialect in which we
can express an opinion about its design. Other dialects generally try to model
an external system, and should aim to reflect its design as closely as possible.

### Specifying sign in integer comparison operations

Since integers are [signless](#integer-signedness-semantics), it is necessary to define the
sign for integer comparison operations. This sign indicates how to treat the
foremost bit of the integer: as sign bit or as most significant bit. For
example, comparing two `i4` values `0b1000` and `0b0010` yields different
results for unsigned (`8 > 3`) and signed (`-8 < 3`) interpretations. This
difference is only significant for _order_ comparisons, but not for _equality_
comparisons. Indeed, for the latter all bits must have the same value
independently of the sign. Since both arguments have exactly the same bit width
and cannot be padded by this operation, it is impossible to compare two values
whose bit representations would differ while the values are interpreted as
equal.

### Specifying comparison kind as attribute

Unlike arithmetic, comparison operators share several common properties, e.g.
they cannot be considered associative. In practice, comparisons are sometimes
implemented by the same instruction or its variants so it makes sense to group
them together at the IR level.

An alternative would be introducing ten distinct operators for all currently
supported kinds of integer comparisons. These operators would have increased the
number of "reserved" names used by standard operations as well as the size of
the C++ API while their implementations would have been mostly identical.

The comparison kind is internally an integer attribute. However, for the sake of
readability by humans, custom assembly form accepts string literals that are
mapped to the underlying integer values: `cmpi "eq", %lhs, %rhs` better implies
integer equality comparison than `cmpi 0, %lhs, %rhs` where it is unclear what
gets compared to what else. This syntactic sugar is possible thanks to parser
logic redefinitions for custom assembly form of non-builtin operations.
Supporting it in the full notation would have required changing how the main
parsing algorithm works and may have unexpected repercussions. While it had been
possible to store the predicate as string attribute, it would have rendered
impossible to implement switching logic based on the comparison kind and made
attribute validity checks (one out of ten possible kinds) more complex.

### 'select' operation to implement min/max

Although `min` and `max` operations are likely to occur as a result of
transforming affine loops in ML functions, we did not make them first-class
operations. Instead, we provide the `select` operation that can be combined with
`cmpi` to implement the minimum and maximum computation. Although they now
require two operations, they are likely to be emitted automatically during the
transformation inside MLIR. On the other hand, there are multiple benefits of
introducing `select`: standalone min/max would concern themselves with the
signedness of the comparison, already taken into account by `cmpi`; `select` can
support floats transparently if used after a float-comparison operation; the
lower-level targets provide `select`-like instructions making the translation
trivial.

This operation could have been implemented with additional control flow: `%r =
select %cond, %t, %f` is equivalent to

```mlir
^bb0:
  cond_br %cond, ^bb1(%t), ^bb1(%f)
^bb1(%r):
```

However, this control flow granularity is not available in the ML functions
where min/max, and thus `select`, are likely to appear. In addition, simpler
control flow may be beneficial for optimization in general.

### Regions

#### Attributes of type 'Block'

We considered representing regions through `ArrayAttr`s containing a list of a
special type `IRBlockAttr`, which in turn would contain a list of operations.
All attributes in MLIR are unique’d within the context, which would make the IR
inside the regions immortal for no good reason.

#### Use "inlined" functions as regions

We considered attaching a "force-inline" attribute on a function and/or a
function `call` operation. Even the minimal region support (use cases in
affine.for and affine.if existing before the regions) requires access to the
values defined in the dominating block, which is not supported by functions.
Conceptually, function bodies are instances of regions rather than the inverse;
regions can also be device kernels, alternative sections, etc.

#### Dedicated `region` operation

This would mean we have a special kind of operation that is allowed to have
regions while other operations are not. Such distinction is similar to the
Stmt/Op difference we have had and chose to remove to make the IR simpler and
more flexible. It would also require analyses and passes to consider the
interplay between operations (e.g., an `affine.for` operation must be followed
by a region operation). Finally, a region operation can be introduced using the
current implementation, among other operations and without being special in any
sense.

#### Explicit capture of the values used in a region

Being able to use values defined outside the region implies that use-def chains
may contain uses from different nested regions. Consequently, IR transformations
and analyses can pull the instruction defining the value across region
boundaries, for example in case of TableGen-defined canonicalization patterns.
This would not be the case if all used values had been passed as region
arguments. One of the motivations for introducing regions in the IR is precisely
to enable cross-region analyses and transformations that are simpler than
inter-procedural transformations. Having uses from different regions appear in
the same use-def chain, contrary to an additional data structure maintaining
correspondence between function call arguments as uses of the original
definitions and formal arguments as new definitions, enables such
simplification. Since individual operations now belong to blocks, which belong
to regions, it is always possible to check if the definition of the value
belongs to the same region as its particular use. The risk is that any IR
traversal will need to handle explicitly this situation and it is easy to forget
a check (or conversely it isn’t easy to design the right check in a tablegen
pattern for example): traversing use-def chains potentially crosses implicitly
semantic barriers, making it possible to unknowingly break region semantics.
This is expected to be caught in the verifier after the transformation.

At the same time, one may choose to pass certain or all values as region
arguments to explicitly break the use-def chains in the current proposal. This
can be combined with an attribute-imposed semantic requirement disallowing the
body of the region to refer to any value from outside it.

### Dialect type extensions

This section describes the design decisions that shaped the dialect extensible
type system present in MLIR.

#### Interactions between dialects

There are two different interactions between dialects that are important to
understand. When types of a dialect are:

*   In operations of other dialects

    -   For standard/builtin operations, only builtin types are allowed. This
        restriction allows for operations to clearly understand the invariants
        that they are working under.
    -   Outside of standard/builtin operations, dialects are expected to verify
        the allowable operation types per operation.

*   In types of other dialects

    -   For builtin types, these types are allowed to contain types from other
        dialects. This simplifies the type system and removes the need for
        dialects to redefine all of the builtin aggregate types, e.g. tensor, as
        well as the memref type. Dialects are expected to verify that a specific
        type is valid within a builtin type, e.g. if a type can be an element of
        a tensor.
    -   For dialect types, the dialect is expected to verify any type
        invariants, e.g. if the tensor type can contain a specific type of that
        dialect.

#### Separating builtin and standard types

Following the separation between the built-in and standard dialect, it makes
sense to separate built-in types and standard dialect types. Built-in types are
required for the validity of the IR itself, e.g. the function type (which
appears in function signatures and generic assembly forms of operations).
Integer, float, vector, memref and tensor types, while important, are not
necessary for IR validity.

#### Unregistered types

MLIR supports unregistered operations in generic assembly form. MLIR also
supports a similar concept for types. When parsing, if the dialect for dialect
type has not been registered the type is modeled as an 'OpaqueType'. This allows
for types to be round-tripped without needing to link in the dialect library
that defined them. No additional information about opaque types, outside of
parsing/printing, will be available.

#### Dialect type syntax

Dialect extended types are represented as string literals wrapped inside of the
dialect namespace. This means that the parser delegates to the dialect for
parsing specific type instances. This differs from the representation of dialect
defined operations, of which have an identifier name that the parser uses to
identify and parse them.

This representation was chosen for several reasons:

##### Dialects must provide custom type parsers

Dialect type parsing cannot plug into the existing parser infrastructure as
operations do with the OpAsmParser/Printer. Operations have a defined syntax
structure that is the same across all dialects. Types, on the other hand, may
have many different, and sometimes conflicting, parsing constraints that would
be difficult/unmaintainable to provide within a single interface.

This also has the added benefit of encouraging dialects to reuse existing
external type parsers. For example, an LLVM dialect may provide an MLIR LLVM
type that is simply a wrapper around LLVM types. The LLVM dialect would then use
the existing LLVM type parsing infrastructure.

Example:

```mlir
%s = "foo"() : () -> !llvm<"i32*">
```

##### Types do not always have canonical names

Unlike operations, types generally do not have a formal canonical name. For
example, function types have no defined keyword and integer types are defined by
a regular expression to support arbitrary bitwidth. Dialects with existing type
systems, e.g. LLVM, are likely to provide wrappers around their existing type
systems. For these wrapper types there is no simple canonical name, it's logical
to think of these types as existing within the namespace of the dialect. If a
dialect wishes to assign a canonical name to a type, it can be done via
[type aliases](../LangRef.md/#type-aliases).

### Tuple types

The MLIR type system provides first class support for defining
[tuple types](../Dialects/Builtin/#tupletype). This is due to the fact that `Tuple`
represents a universal concept that is likely to, and has already begun to,
present itself in many different dialects. Though this type is first class in
the type system, it merely serves to provide a common mechanism in which to
represent this concept in MLIR. As such, MLIR provides no standard operations
for interfacing with `tuple` types. It is up to dialect authors to provide
operations, e.g. extract_tuple_element, to interpret and manipulate them. When
possible, operations should prefer to use multiple results instead. These
provide a myriad of benefits, such as alleviating any need for tuple-extract
operations that merely get in the way of analysis and transformation.

### Assembly forms

MLIR decides to support both generic and custom assembly forms under the
following considerations:

MLIR is an open system; it is designed to support modular and pluggable
dialects. Depending on whether there exists a corresponding dialect and whether
the dialect is plugged in, operations may or may not be registered into MLIR
system. Yet we still need a way to investigate these operations. So the generic
assembly form is mandated by this aspect of MLIR system. It provides a default
textual form for operations.

On the other hand, an assembly form is for assisting developers to investigate
the IR. The generic form serves as a safe fallback but it can be too verbose for
certain ops. Therefore, MLIR gives each dialect the choice to define a custom
assembly form for each operation according to the operation's semantics and
specific needs. The custom assembly form can de-duplicate information from the
operation to derive a more concise form, thus better facilitating the
comprehension of the IR.

## Examples

This section describes a few very simple examples that help understand how MLIR
represents computation.

### Non-affine control flow

```mlir
// A simple linear search in every row of a matrix
for (i = 0; i < N; i++) {
  for (j = 0; j < N; j++) {
    // dynamic control flow
    if (a[i][j] == key) {
      s[i] = j;
      break;
    }
  }
}
```

The presence of dynamic control flow leads to an inner non-affine function
nested in an outer function that uses affine loops.

```mlir
func @search(%A: memref<?x?xi32>, %S: <?xi32>, %key : i32) {
  %ni = dim %A, 0 : memref<?x?xi32>
  // This loop can be parallelized
  affine.for %i = 0 to %ni {
    call @search_body (%A, %S, %key, %i) : (memref<?x?xi32>, memref<?xi32>, i32, i32)
  }
  return
}

func @search_body(%A: memref<?x?xi32>, %S: memref<?xi32>, %key: i32, %i : i32) {
  %nj = dim %A, 1 : memref<?x?xi32>
  br ^bb1(0)

^bb1(%j: i32)
  %p1 = cmpi "lt", %j, %nj : i32
  cond_br %p1, ^bb2, ^bb5

^bb2:
  %v = affine.load %A[%i, %j] : memref<?x?xi32>
  %p2 = cmpi "eq", %v, %key : i32
  cond_br %p2, ^bb3(%j), ^bb4

^bb3(%j: i32)
  affine.store %j, %S[%i] : memref<?xi32>
  br ^bb5

^bb4:
  %jinc = addi %j, 1 : i32
  br ^bb1(%jinc)

^bb5:
  return
}
```

As per the [MLIR spec](../LangRef.md), the restrictions on dimensions and symbol
identifiers to be used with the affine.apply operation only apply to accesses
inside `affine.for` and `affine.if` operations. However, an analysis of accesses
inside the called function (`@search_body`) is necessary to determine if the
`%i` loop could be parallelized: such function access analysis is calling
context sensitive.

### Non-affine loop bounds

Loop bounds that are not affine lead to a nesting of functions as shown below.

```c
for (i = 0; i < N; i++)
  for (j = 0; j < N; j++)
    // Non-affine loop bound for k loop.
    for (k = 0; k < pow(2, j); k++)
       for (l = 0; l < N; l++) {
        // block loop body
        ...
       }
```

```mlir
func @outer_nest(%n : index) {
  affine.for %i = 0 to %n {
    affine.for %j = 0 to %n {
      %pow = call @pow(2, %j) : (index, index) ->  index
      call @inner_nest(%pow, %n) : ...
    }
  }
  return
}

func @inner_nest(%m : index, %n : index) {
  affine.for %k = 0 to %m {
    affine.for %l = 0 to %n {
      ...
    }
  }
  return
}
```

### Reference 2D Convolution

The following example illustrates a reference implementation of a 2D
convolution, which uses an integer set `#domain` to represent valid input data
in a dilated convolution.

```mlir
// Dilation factors S0 and S1 can be constant folded if constant at compile time.
#domain = (d0, d1)[S0,S1,S2,S3]: (d0 % S0 == 0, d1 % S1 == 0, d0 >= 0, d1 >= 0,
                                   S3 - d0 - 1 >= 0, S4 - d1 - 1 >= 0)
// Identity map (shown here for illustration).
#map0 = (d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)

// Affine map from output to input coordinate space.
// d0 = output_h, d1 = output_w, d2 = kernel_h, d3 = kernel_w
// S0 = h_stride, S1 = w_stride, S2 = h_kernel_dilation, S3 = w_kernel_dilation
// S4 = h_pad_low, S5 = w_pad_low
//     %out0 =  %0#1 * %h_stride + %0#4 * %h_kernel_dilation - %h_pad_low
//     %out1=  %0#2 * %w_stride + %0#5 * %w_kernel_dilation - %w_pad_low
#map1_0 = (d0, d1, d2, d3) [S0, S1, S2, S3, S4, S5] -> (d0 * S0 + d2 * S2 - %S4)
#map1_1 = (d0, d1, d2, d3) [S0, S1, S2, S3, S4, S5] -> (d1 * S1 + d3 * S3 - %S5)

// Semi-affine map to undilated input coordinate space.
// d0 = input_h, d1 = input_w, S0 = h_base_dilation, S1 = w_base_dilation.
#map2_0 = (d0, d1) [S0, S1] -> (d0 / S0)
#map2_1 = (d0, d1) [S0, S1] -> (d1 / S1)

// Conv2D shapes:
// input:   [batch, input_height, input_width, input_feature]
// kernel: [kernel_height, kernel_width, input_feature, output_feature]
// output: [batch, output_height, output_width, output_feature]
func @conv2d(%input: memref<16x1024x1024x3xf32, #lm0, /*scratchpad=*/1>,
             %kernel: memref<5x5x3x32xf32, #lm0, /*scratchpad=*/1>,
             %output: memref<16x512x512x32xf32, #lm0, /*scratchpad=*/1>) {
  affine.for %b = 0 to %batch {
    affine.for %oh = 0 to %output_height {
      affine.for %ow = 0 to %output_width {
        affine.for %of = 0 to %output_feature {
          affine.for %kh = 0 to %kernel_height {
            affine.for %kw = 0 to %kernel_width {
              affine.for %if = 0 to %input_feature {
                // Calculate input indices.
                %1_0 = affine.apply #map1_0 (%0#1, %0#2, %0#4, %0#5)
                  [%h_stride, %w_stride, %h_kernel_dilation, %w_kernel_dilation,
                   %h_pad_low, %w_pad_low]
                %1_1 = affine.apply #map1_1 (%0#1, %0#2, %0#4, %0#5)
                  [%h_stride, %w_stride, %h_kernel_dilation, %w_kernel_dilation,
                   %h_pad_low, %w_pad_low]

                // Check if access is not in padding.
                affine.if #domain(%1_0, %1_1)
                                       [%h_base_dilation, %w_kernel_dilation, %h_bound, %w_bound] {
                  %2_0 = affine.apply #map2 (%1_0, %1_1)
                  %2_1 = affine.apply #map2 (%1_0, %1_1)
                  // Compute: output[output_indices] += input[input_indices] * kernel[kernel_indices]
                  call @multiply_accumulate(%input, %kernel, %output, %b, %oh, %ow, %of, %kh, %kw, %if, %2_0, %2_1)
                }
              }
            }
          }
        }
      }
    }
  }
  return
}
```

TODO: (Add more examples showing the IR for a variety of interesting cases)

## Design alternatives and extensions

This is a list of some design alternatives and extensions that we discussed in
detail but did not include in the spec or postponed them for future
consideration on demand. We will revisit these discussions when we have more
implementation experience and learn more about the challenges and limitations of
our current design in practice.

### Polyhedral code representation alternatives: schedule lists vs schedules trees vs affine loop/if forms

The current MLIR uses a representation of polyhedral schedules using a tree of
if/for loops. We extensively debated the tradeoffs involved in the typical
unordered polyhedral instruction representation (where each instruction has
multidimensional schedule information), discussed the benefits of schedule tree
forms, and eventually decided to go with a syntactic tree of affine if/else
conditionals and affine for loops. Discussion of the tradeoff was captured in
this document:
[ MLIR: The case for a simplified polyhedral form](RationaleSimplifiedPolyhedralForm.md).

At a high level, we have two alternatives here:

1.  Schedule tree representation instead of an affine loop AST form: The current
    proposal uses an affine loop and conditional tree form, which is syntactic
    and with no separation of domains as sets and schedules as multidimensional
    affine functions. A schedule tree form however makes polyhedral domains and
    schedules a first class concept in the IR allowing compact expression of
    transformations through the schedule tree without changing the domains of
    instructions. Such a representation also hides prologues, epilogues, partial
    tiles, complex loop bounds and conditionals making loop nests free of
    "syntax". Cost models instead look at domains and schedules. In addition, if
    necessary such a domain schedule representation can be normalized to
    explicitly propagate the schedule into domains and model all the cleanup
    code. An example and more detail on the schedule tree form is in the next
    section.
1.  Having two different forms of "affine regions": an affine loop tree form
    and a polyhedral schedule tree form. In the latter, ops could carry
    attributes capturing domain, scheduling, and other polyhedral code
    generation options with IntegerSet, AffineMap, and other attributes.

#### Schedule Tree Representation for Affine Regions

This representation is based on a simplified form of the domain/schedule
representation used by the polyhedral compiler community. Domains represent what
has to be executed while schedules represent the order in which domain elements
are interleaved. We model domains as non-piece-wise convex integer sets, and
schedules as affine functions; however, the former can be disjunctive, and the
latter can be piece-wise affine relations. In the schedule tree representation,
domain and schedules for instructions are represented in a tree-like structure
which is called a schedule tree. Each non-leaf node of the tree is an abstract
polyhedral dimension corresponding to an abstract fused loop for each ML
instruction that appears in that branch. Each leaf node is an ML Instruction.

```mlir
// A tiled matmul code (128x128x128) represented in schedule tree form

// #map0 = (d0, d1, d2, d3, d4, d5) -> (128*d0 + d3, 128*d1 + d4, 128*d2 + d5)
#intset_ij = (i, j) [M, N, K]  : i >= 0, -i + N - 1 >= 0, j >= 0, -j + N-1 >= 0
#intset_ijk = (i, j, k) [M, N, K] : i >= 0, -i + N - 1 >= 0, j >= 0,
                                     -j +  M-1 >= 0, k >= 0, -k + N - 1 >= 0)
func @matmul(%A, %B, %C, %M, %N, %K) : (...)  { // %M, N, K are symbols
  // t1, t2, t3, t4, t5, t6  are abstract polyhedral loops
  mldim %t1 : {S1,S2,S3,S4,S5}  floordiv (i, 128) {
    mldim %t2 : {S1,S2,S3,S4,S5}  floordiv (j, 128) {
      // (%i, %j) = affine.apply (d0, d1) -> (128*d0, 128*d1) (%t1, %t2)
      call dma_mem_to_scratchpad(%C, %i, %j, %M, %N, %K)
          with @intset_ij(%i, %j) [%M, %N, %K]
      mldim %t3 :   {S2,S3,S4,S5} floordiv (k, 128) {
        // (%i, %j, %k) = affine.apply (d0, d1, d2)
        //                          -> (128*d0, 128*d1, 128*d2)  (%t1, %t2, %t3)
        call dma_mem_to_scratchpad(%A, ...) with #inset_ijk (%i, %j, %k) [%M, %N, %K]
        // (%i, %j, %k) = affine.apply (d0, d1, d2)
        //                          -> (128*d0, 128*d1, 128*d2)  (%t1, %t2, %t3)
        call dma_mem_to_scratchpad(%B, ...) with #inset_ijk (%i, %j, %k) [%M, %N, %K]
        mldim %t4 : {S4} i mod 128 {
          mldim %t5 : {S4} j mod 128 {
            mldim %t6 : {S4} k mod 128 {
              // (%i, %j, %k) = affine.apply #map0 (%t1, %t2, %t3, %t4, %t5, %t6)
              call matmul_body(A, B, C, %i, %j, %k, %M, %N, %K)
                  with #inset_ijk(%i, %j, %k) [%M, %N, %K]
            } // end mld4im t6
          } // end mldim t5
        } // end mldim t4
      } // end mldim t3
      // (%i, %j) = affine.apply (d0, d1) -> (128*d0, 128*d1) (%t1, %t2)
      call $dma_scratchpad_to_mem_C ... with #intset(%i, %j) [%M, %N, %K]
    }  // end mldim t2
  } // end mldim t1
  return
}

```

### Affine Relations

The current MLIR spec includes affine maps and integer sets, but not
affine relations. Affine relations are a natural way to model read and
write access information, which can be very useful to capture the
behavior of external library calls where no implementation is
available, high-performance vendor libraries, or user-provided /
user-tuned routines.

An affine relation is a relation between input and output dimension identifiers
while being symbolic on a list of symbolic identifiers and with affine
constraints on the identifiers.

Syntax:

```
// Affine relation definition at the top of file
affine-rel-def ::= affine-rel-id `=` affine-relation-inline

affine-rel-id ::= `##` prefixed-id

affine-relation-inline ::=
       `(` input-dims `)` (`[` symbols `]`)? `->`
       `(` output-dims `)` :  affine-constraint-conjunction

input-dims ::= bare-id-list
output-dims ::= bare-id-list
symbols ::= bare-id-list

affine-rel ::= affine-rel-id | affine-relation-inline

// Usage
affine-rel-spec ::= affine-rel dim-and-symbol-use-list
```

All identifiers appearing in input-dims, output-dims, and symbol-dims are
pairwise distinct. All affine-constraint non-terminals in the above syntax are
allowed to contain identifiers only from input-dims, output-dims, and
symbol-dims.

Affine relations are used to model read, write, may_read, and may_write sets of
functions in the IR. The output dimension identifiers correspond to the data
dimensions.

Example:

```mlir
// read relation: two elements ( d0 <= r0 <= d0+1 )
##aff_rel9 = (d0) -> (r0) : r0 - d0 >= 0, d0 - r0 + 1 >= 0

func @count (%A : memref<128xf32>, %pos : i32) -> f32
  reads: {%A ##aff_rel9 (%pos)}
  writes: /* empty */
  may_reads: /* empty */
  may_writes: /* empty */ {
bb0 (%0, %1: memref<128xf32>, i64):
  %val = affine.load %A [%pos]
  %val = affine.load %A [%pos + 1]
  %p = mulf %val, %val : f32
  return %p : f32
}
```

### Regions

#### Making function definition an operation

MLIR supports values of a Function type. Instead of having first-class IR
concept for functions, one could define an operation with a body region that
defines a function value. The particularity of functions is that their names are
globally visible and can be referred to before being defined, unlike SSA values
that must be defined first. Implementing a "function definition" operation would
require to relax some of the SSA constraints in a region, and also make the IR
Module a region as well. It would also affect the core infrastructure (e.g.,
function passes) only for the sake of concept unification.

#### Having types on a region

Instead of inspecting the types of arguments of the first block, one could give
the region itself a type. This type would be redundant with block argument
types, which must have values and create room for type mismatches. While
functions do have types that are partly redundant with the arguments of the
first block in the function, this is necessary to support function declarations
that do not have a body which we can refer to in order to obtain the argument
types. A region is always contained in an operation or a function that can be
queried to obtain the “type” of the region if necessary.

A type on a region can be justified if Regions were to be considered separately
from the enclosing entity (operation or function) and had their own semantics
that should be checked.

#### Attaching attributes to regions

Regions could be annotated with dialect attributes to use attribute verification
hooks. An operation could take multiple regions as arguments, and each of them
may require different attributes. However, there are currently very few
practical cases where this would be necessary. Instead, one could simulate
per-region attributes with array attributes attached to the entity containing
the region (operation or function). This decreases the overall complexity of the
IR and enables more concise and op-specific forms, e.g., when all regions of an
op have the same attribute that can be only mentioned once. Since the semantics
of the region is entirely defined by the enclosing entity, it also makes sense
to have attributes attached to that entity rather than to the region itself.

This can be reconsidered in the future if we see a non-neglectable amount of use
cases.

### Read/Write/May_Read/May_Write sets for External Functions

Having read, write, may_read, and may_write sets for external functions which
include opaque ones, high-performance vendor libraries such as CuDNN, CuB, MKL,
FFT libraries, user-provided/optimized functions, or data movement runtimes such
as DMA ones is a powerful feature. It allows the compiler to perform analysis,
composition/transformation in the presence of such calls and with loops around
such calls on sub-tensors. For user-provided or custom hand-tuned functions, the
read/write/may_read/may_write sets could be provided a-priori by a user as part
of the external function signature or they could be part of a database.

TODO: Design this, and update to use function attribute syntax.

Example:

```mlir
##rel9 ( ) [s0] -> (r0, r1) : 0 <= r0 <= 1023, 0 <= r1 <= s0 - 1

func @cblas_reduce_ffi(%M: memref<1024 x ? x f32, #layout_map0, /*mem=*/0>)
  -> f32 [
  reads: {%M, ##rel9() }
  writes: /* empty */
  may_reads: /* empty */
  may_writes: /* empty */
]

func @dma_mem_to_scratchpad(%a : memref<1024 x f32, #layout_map0, /*mem=*/0>,
    %b : memref<1024 x f32, #layout_map0, 1>, %c : memref<1024 x f32,
    #layout_map0>) [
  reads: {%M, ##rel9() }
  writes: /* empty */
  may_reads: /* empty */
  may_writes: /* empty */
 ]

```

### Memref Extensions

1.  Arbitrary polyhedral shapes for tensors: e.g., triangular shapes in tensor
    dimensions where there is symmetry: use integer set (affine constraints) to
    model tensor data space (instead of just extents). Requires some changes to
    the IR and the in-memory form.
1.  Layout maps

    1.  Allow piece-wise affine maps for layouts: allows clean modeling of
        boundary cases for images/tensors through padding, wrapping, mirroring,
        padding where padded values are the results of computation as opposed to
        data, padding in the interior as opposed to just boundaries.
    1.  Allow many-to-one layout maps: Index and layout maps in the current
        proposal are bijective. Extending them to many-to-one layout maps allows
        cleaner(?) modeling of broadcast/reduce style computations while reusing
        memory.

    Proposal 2(a) requires non-trivial changes to the IR and the in-memory
    representation. 2(b) requires no change, but impacts how cost models look at
    index and layout maps.

### `affine.if` and `affine.for` Extensions for "Escaping Scalars"

We considered providing a representation for SSA values that are live out of
`if/else` conditional bodies and loop carried in `affine.for` loops. We
ultimately abandoned this approach due to its complexity. In the current design
of MLIR, scalar variables cannot escape for loops or if instructions. In
situations, where escaping is necessary, we use zero-dimensional tensors and
memrefs instead of scalars.

**TODO**: This whole section is obsolete and should be updated to use block
arguments and a yield like terminator in for/if instructions.

The abandoned design of supporting escaping scalars is as follows:

#### affine.for Instruction

Syntax:

```
[<out-var-list> =]
for %<index-variable-name> = <lower-bound> ... <upper-bound> step <step>
   [with <in-var-list>] { <loop-instruction-list> }
```

out-var-list is a comma separated list of SSA values defined in the loop body
and used outside the loop body. in-var-list is a comma separated list of SSA
values used inside the loop body and their initializers. loop-instruction-list
is a list of instructions that may also include a yield instruction.

Example:

```mlir
// Return sum of elements in 1-dimensional mref A
func i32 @sum(%A : memref<?xi32>, %N : i32) -> (i32) {
   %init = 0
   %result = affine.for %i = 0 to N with %tmp(%init) {
      %value = affine.load %A[%i]
      %sum = %value + %tmp
      yield %sum
   }
   return %result : i32
}
```

#### affine.if/else Instruction

Syntax:

```
<out-var-list> = affine.if (<cond-list>) {...} [else {...}]
```

Out-var-list is a list of SSA values defined by the if-instruction. The values
are arguments to the yield-instruction that occurs in both then and else clauses
when else clause is present. When if instruction contains only if clause, the
escaping value defined in the then clause should be merged with the value the
variable had before the if instruction. The design captured here does not handle
this situation.

Example:

```mlir
// Compute sum of half of the array
func i32 @sum_half(%A : memref<?xi32>, %N : i32) -> (i32) {
   %s0 = 0
   %s1 = affine.for %i = 1 ... N step 1 with %s2 (%s0) {
       %s3 = if (%i >= %N / 2) {
          %v0 = affine.load %A[%i]
          %s4 = %s2 + %v0
          yield %s4
       }
       yield %s3
   }
   return %s1 : i32
}
```

### Multithreading the compiler

People want compilers to go fast, and one simple way to do that is to
multi-thread them. There are multiple strategies for this, but a simple one is
to optimize and compile separate functions in parallel. LLVM's original pass
manager anticipated this demand, and the CallGraphSCCPass manager is even
designed to support this as well, but unfortunately, a few early design
decisions in LLVM prevent this from ever happening. Instead, things like ThinLTO
are forced to split programs into separate LLVM modules/context and optimize
those chunks independently.

The problem is that LLVM has several objects in its IR that are globally uniqued
and also mutable: notably constants like `i32 0`. In LLVM, these constants are
`Value`'s, which allow them to be used as operands to instructions, and that
they also have SSA use lists. Because these things are uniqued, every `i32 0` in
any function shares a use list. This means that optimizing multiple functions in
parallel won't work (at least without some sort of synchronization on the use
lists, which would be unbearably inefficient).

MLIR now supports a multithreaded pass manager. We do this through several
design choices:

1.  MLIR makes use of extensive uniqued immutable data structures (affine
    expressions, types, etc are all immutable, uniqued, and immortal).
2.  Constants are defined in per-function pools, instead of being globally
    uniqued.
3.  Functions themselves are not SSA values either, so they don't have the same
    problem as constants.
4.  FunctionPasses are copied (through their copy ctor) into one instance per
    thread, avoiding sharing of local state across threads.

This allows MLIR function passes to support efficient multithreaded compilation
and code generation.
