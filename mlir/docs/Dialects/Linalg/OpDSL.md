# Linalg OpDSL

Python based DSL for authoring Linalg op definitions and generating
`linalg.generic` IR based on them for samples.

The Linalg OpDSL is a high level DSL for constructing structured op definitions
in a way that can be exported to built-in, named structured ops via
[YAML-based definitions](_index.md/#yaml-gen) or used interactively to emit
corresponding `linalg.generic` IR for the composition.

## Basic usage

The tool is bundled with the MLIR Python bindings. To use from the CMake build
tree, MLIR must be build with Python bindings enabled
(`-DMLIR_ENALBE_BINDINGS_PYTHON=ON`). Then add the `python` directory in the
build tree to your `PYTHONPATH` environment variable (i.e. `export
PYTHONPATH=$PWD/build/python`). Optionally, use an installed MLIR package, if
available, to avoid building.

```shell
# Dump the `core_named_ops.py` module as YAML.
python -m mlir.dialects.linalg.opdsl.dump_oplib .ops.core_named_ops
```

The tool is meant for use during both development and runtime, but not as a
build tool of the core compiler: in order to export static named op definitions
to be built as part of the compiler, the corresponding Linalg dialect YAML file
must be updated and reviewed. TODO: Develop a script to automate op updates to
these files.

## Language Guide

The language presented here is loosely inspired from the
[Tensor Comprehensions](https://arxiv.org/pdf/1802.04730.pdf) work, adapted to
represent linalg structured ops.

This tool is new and rapidly evolving. For language examples, refer to the
built-in ops in the `mlir.tools.linalg_opdsl.ops` package
(`lib/Bindings/Python/mlir/tools/linalg_opdsl/ops` in the repository).

Using a matmul as an example, we will decompose the language:

```python
T1 = TV.T1
T2 = TV.T2

@linalg_structured_op
def matmul(A=TensorDef(T1, S.M, S.K),
           B=TensorDef(T2, S.K, S.N),
           C=TensorDef(U, S.M, S.N, output=True)):
  """Performs a matrix multiplication of two 2D inputs.

  Numeric casting is performed on the operands to the inner multiply, promoting
  them to the same data type as the accumulator/output.
  """
  domain(D.m, D.n, D.k)
  implements(ContractionOpInterface)
  C[D.m, D.n] += cast(U, A[D.m, D.k]) * cast(U, B[D.k, D.n])
```

Here we have a simple type polymorphic contraction that takes arguments `A` and
`B` and outputs `C`. Each is bound to a `TensorDef`, which specifies:

*   The symbolic element type (`T1`, `T2`, `U` above).
*   Symbolic shape expressions with symbols that are bound globally for the op (
    note that in this simple example, the shape expressions are just symbol
    references, but they are permitted to be a constrained set of affine
    expressions).
*   Usage (`output=True`).

The docstring will be transferred to the op definition verbatim.

An explicit iteration domain dimension order can be declared for the op via
`domain(D.d0[, D.d1...])`.

Special identifying op interfaces can be declared for the op via
`implements(interface1[, interface2...])`.

## Parameters

Structured operations take two types of runtime parameters namely scalars and
tensors. While scalars are inputs only, a tensor may be marked as an output.
Assignment expressions index the tensor parameters to access the individual
elements, while scalars can be accessed directly.

The following example demonstrates the use of the two parameter types:

```python
@linalg_structured_op
def copy_and_scale(val=ScalarDef(T),
                   I=TensorDef(T, S.M, S.K),
                   O=TensorDef(T, S.M, S.K, output=True)):
  """Scale the input by the scalar value and store the result"""
  O[D.m, D.n] = I[D.m, D.n] * val
```

The operation scales the input tensor `I` scales its elements by the value `val`
and writes the result to the output tensor `out`. The scalar `val` is bound to a
`ScalarDef`, which specifies the type of the scalar operand. The tensors are
bound to a `TensorDef` as demonstrated by the matmul example. All parameters
appear in the parameter list of the operation:

```python
fill(val, in_tensor, outs=[out_tensor])
```

## Attributes

Attributes are compile-time constant parameters only accessible in index
expressions. They can be used to parameterize the access pattern of a structured
operation, for example, by setting its strides. They cannot take part in the
actual computation.

The following example demonstrates the use of attributes:

```python
@linalg_structured_op
def strided_copy(I=TensorDef(T, S.IH, S.IW),
                 O=TensorDef(T, S.OH, S.OW, output=True),
                 strides=AttributeDef(S.SH, S.SW)):
  """Copy a subset of the input tensor elements to the output tensor"""
  O[D.oh, D.ow] = I[D.oh * S.SH, D.ow * S.SW]
```

The operation implements a strided copy from the input tensor `I` to the output
tensor `O`. The `strides` attribute is bound to an `AttributeDef`. It defines
the symbols `S.SH` and `S.SW`, which are used to index the input tensor `I`.
When instantiating the operation, the attribute is set using a named argument:

```python
strided_copy(in_tensor, outs=[out_tensor], strides=[1,2])
```

The `strides` vector elements substitute the symbols `S.SH` and `S.SW` in the
index expressions of the operation instance.

Attributes are currently limited to integer vectors and only accessible in index
expressions. An operation may have multiple attributes all of them placed at the
end of the parameter list after the output tensors.

## Shape-Only Tensors

Structured operations derive the iteration space given the sizes of the input
and output tensors. Certain operations need shape-only tensors that are not
accessed and exist purely for the sake of specifying the iteration domain. An
example is the pooling operation that takes a shape-only tensor to define the
iteration space of the reduction. As shape-only tensors have no uses, the
`TensorDef` takes an additional optional `index_dims` parameter to map the shape
to index dimensions.

The following example demonstrates the index dimension annotation:

```python
@linalg_structured_op
def pooling_poly(
    I=TensorDef(T1, S.N, S.H, S.W, S.C),
    K=TensorDef(T2, S.KH, S.KW, index_dims=[D.kh, D.kw]),
    O=TensorDef(U, S.N, S.OH, S.OW, S.C, output=True),
    strides=AttributeDef(S.SH, S.SW),
    dilations=AttributeDef(S.DH, S.DW)):
  O[D.n, D.oh, D.ow, D.c] += \
      cast(U, I[D.n, D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW, D.c])
```

The pooling operation does not access the shape-only tensor `K`. Instead, the
shapes `S.KH` and `S.KW` specify the iteration domain for the reduction
dimensions `D.kh` and `D.kw`.

## Assignments

The bulk of language consists of assignment expressions of the form above. The
iteration dimension order is determined lexically based on the order encountered
in the expression (following operator precedence if math operators are used).
TODO: Introduce a directive to fix the dimension bindings.

Reduction dimensions are inferred to be any dimensions on the RHS that are not
on the LHS.

A number of arithmetic primitive functions are supported:

*   `PrimFn.add(a, b)` (also via overloading the binary `+` operator)
*   `PrimFn.exp(a)`
*   `PrimFn.log(a)`
*   `PrimFn.mul(a, b)` (also via overloading the binary `*` operator)
*   `PrimFn.max(a, b)`
*   `PrimFn.sub(a, b)` (also via overloading the binary `-` operator)

Reduction functions can appear as the outer-most function on the RHS:

*   `ReduceFn.add` (also overloading the inplace `+=` on a LHS)
*   `ReduceFn.mul`
*   `ReduceFn.max`

There are also special forms:

*   `cast(TypeVar, operand)` casts the `operand` to the target type `TypeVar`.
*   `const(TypeVar, value)` returns a constant value of type `TypeVar`.
*   `index(dim)` returns the iteration index in the given dimension `dim`.

## Types

All types in assignment expressions are late bound based on actual input and
output types of constructed ops. An exception are predefined types such as
`I32`, `I64`, `F32`, and `F64`. These hardwired types enable intermediate
computations with a type that is independent of the input and output types. For
example, parts of floating point computation may require double precision
arithmetic despite all inputs and outputs being single precision values.
Assignment expressions with no `cast` calls will generally require uniform types
throughout and will fail to verify if violated. The presence of a `cast` allows
for a limited form of numeric type conversion between element types that can be
derived from inputs and outputs (and in the future, attributes). `cast` calls
with a `TypeVar` first argument are emitted as `symbolic_cast` primitives in the
YAML definition.

Casting will perform `int<->float` and `index->int` type conversions and will
perform any necessary extension or truncation within type family. Note that
presently, any integer type is assumed to be signed for the purpose of
determining how to extend or truncate. Supporting unsigned integer types is left
for future work.

Not all functions are applicable for all numeric types, and on mismatch, op
verification will fail.
