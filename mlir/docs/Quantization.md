# Quantization

This document outlines the design of the MLIR quantization system. While the
term "quantization" is highly overloaded, in this case, it refers to a fairly
narrow scope of techniques in use to enable conversion of floating-point
computations to corresponding and plausible variants expressed in integer math
for inference, as has historically been supported by low-bit depth inference
engines such as TFLite, various accelerator hardware, and many DSPs.

Much of this is inspired by the approach taken
[in this paper](https://arxiv.org/abs/1712.05877) with many extensions and
adaptations folded in. It specifically documents the positions that MLIR has
taken on the topic, and is not a general reference.

[TOC]

## Uniform quantization

The primary quantization mechanism supported by MLIR is a scheme which can
express fixed point and affine transformations via uniformly spaced point on the
[Real](https://en.wikipedia.org/wiki/Real_number) number line.

Further, the scheme can be applied:

*   *per-layer* : Applying to every value within the target type.
*   *per-axis* (also called *per-channel*) : Applying individually to each index
    along a specific axis of a tensor type.

### Fixed point values

[Fixed point](https://en.wikipedia.org/wiki/Fixed-point_arithmetic) values are a
[Real](https://en.wikipedia.org/wiki/Real_number) number divided by a *scale*.
We will call the result of the divided real the *scaled value*.

$$ real\\_value = scaled\\_value * scale $$

The scale can be interpreted as the distance, in real units, between neighboring
scaled values. For example, if the scale is $ \pi $, then fixed point values
with this scale can only represent multiples of $ \pi $, and nothing in
between. The maximum rounding error to convert an arbitrary Real to a fixed
point value with a given $ scale $ is $ \frac{scale}{2} $. Continuing the
previous example, when $ scale = \pi $, the maximum rounding error will be $
\frac{\pi}{2} $.

Multiplication can be performed on scaled values with different scales, using
the same algorithm as multiplication of real values (note that product scaled
value has $$ scale_{product} = scale_{left \mbox{ } operand} * scale_{right
\mbox{ } operand} $$). Addition can be performed on scaled values, so long as
they have the same scale, using the same algorithm for addition of real values.
This makes it convenient to represent scaled values on a computer as signed
integers, and perform arithmetic on those signed integers, because the results
will be correct scaled values.

### Affine values

Mathematically speaking, affine values are the result of
[adding a Real-valued *zero point*, to a scaled value](https://en.wikipedia.org/wiki/Affine_transformation#Representation).
Alternatively (and equivalently), subtracting a zero point from an affine value results in a
scaled value:

$$ real\\_value = scaled\\_value * scale = (affine\\_value - zero\\_point) * scale $$

Essentially, affine values are a shift of the scaled values by some constant
amount. Arithmetic (i.e., addition, subtraction, multiplication, division)
cannot, in general, be directly performed on affine values; they must first be
[converted](#affine-to-fixed-point) to the equivalent scaled values.

As alluded to above, the motivation for using affine values is to more
efficiently represent real values that will actually be encountered during
computation. Frequently, real values that will be encountered are not
symmetric around the real zero. We also make the assumption that the real zero
is encountered during computation, and should thus be represented.

In this case, it is inefficient to store scaled values represented by signed
integers, as some of the signed integers will never be used. In effect, the bit patterns
corresponding to those signed integers are going to waste.

In order to exactly represent the real zero with an integral-valued affine
value, the zero point must be an integer between the minimum and maximum affine
value (inclusive). For example, given an affine value represented by an 8 bit
unsigned integer, we have: $ 0 \leq zero\\_point \leq 255 $. This is important,
because in convolution-like operations of deep neural networks, we frequently
need to zero-pad inputs and outputs, so zero must be exactly representable, or
the result will be biased.

### Relation

Real values, fixed point values, and affine values relate through the following
equation, which demonstrates how to convert one type of number to another:

$$ real\\_value = scaled\\_value * scale = (affine\\_value - zero\\_point) * scale $$

Note that computers generally store mathematical values using a finite number of
bits. Thus, while the above conversions are exact, to store the result in a
finite number of bits, we must, in general, round the result of the conversion
(this applies to both cases: storing using floating point and storing using
fixed point). Note that a full discussion of rounding behavior is outside the
scope of this document, and it is safe to assume unless otherwise stated that
rounding should be according to the IEEE754 default of RNE (where hardware
permits).

### Converting between real and fixed point or affine

To convert a real value to a fixed point value, we must know the scale. To
convert a real value to an affine value, we must know the scale and the zero point.

#### Real to affine

To convert an input tensor of real-valued elements (usually represented by a
floating point format, frequently
[Single precision](https://en.wikipedia.org/wiki/Single-precision_floating-point_format))
to a tensor of affine elements represented by an integral type (e.g. 8-bit
unsigned integer), the following conversion can be performed (note that it is
not required that all representable values of the integral type are used):

$$
\begin{align*}
af&fine\\_value_{uint8 \\, or \\, uint16} \\\\
      &= clampToTargetSize(roundToNearestInteger( \frac{real\\_value_{Single}}{scale_{Single}})_{sint32} + zero\\_point_{uint8 \, or \, uint16})
\end{align*}
$$

In the above, we assume that $real\\_value$ is a Single, $scale$ is a Single,
$roundToNearestInteger$ returns a signed 32-bit integer, and $zero\\_point$
is an unsigned 8-bit or 16-bit integer. Note that bit depth and number of fixed
point values are indicative of common types on typical hardware but is not
constrained to particular bit depths or a requirement that the entire range of
an N-bit integer is used.

#### Affine to real

To convert an output tensor of affine elements represented by uint8
or uint16 to a tensor of real-valued elements (usually represented with a
floating point format, frequently Single precision), the following conversion
can be performed:

$$
\begin{align*}
re&al\\_value_{Single} \\\\
      &= roundToNearestFloat((affine\\_value_{uint8 \\, or \\, uint16} - zero\\_point_{uint8 \\, or \\, uint16})_{sint32})_{Single} * scale_{Single}
\end{align*}
$$

In the above, we assume that the result of subtraction is in 32-bit signed
integer format, and that $roundToNearestFloat$ returns a Single.

#### Affine to fixed point

When the affine and fixed point scales are the same, subtract the zero point
from the affine value to get the equivalent fixed point value.

$$
\begin{align*}
  scaled\\_value = affine\\_value_{non\mbox{-}negative} - zero\\_point_{non\mbox{-}negative}
\end{align*}
$$

#### Fixed point to affine

When the affine and fixed point scales are the same, add the zero point to the
fixed point value to get the equivalent affine value.

$$
\begin{align*}
  affine\\_value_{non\mbox{-}negative} = scaled\\_value + zero\\_point_{non\mbox{-}negative}
\end{align*}
$$

## Usage within MLIR

There are several components to the quantization system being developed within
MLIR:

*   *Quantization* dialect containing:

    *   A family of [QuantizedTypes](#quantized-type) which represent the
        mapping between *expressed* values (typically of a floating point
        computer type) and *storage* values (typically of an integral computer
        type).
    *   [Type conversion ops](#quantized-type-conversion-operations) for converting
        between types based on a QuantizedType and its *expressed* and *storage*
        sub-types.
    *   [Instrumentation ops](#instrumentation-and-constraint-operations) for assigning
        instrumentation points within the computation where runtime statistics
        may help guide the quantization process.

*   [Integration with simulated quantization at training time](#integration-with-simulated-quantization-at-training-time)

*   [TFLite native quantization](#tflite-native-quantization)

    *   The TFLite op-set natively supports uniform-quantized variants.
    *   Passes and tools exist to convert directly from the *TensorFlow* dialect
        to the TFLite quantized operation set.

Not every application of quantization will use all of these facilities. Specifically, the
TensorFlow to TensorFlow Lite conversion uses the QuantizedTypes but has its own
operations for type conversion and expression of the supporting math.

## Quantization Dialect

### Quantized type

TODO: Flesh this section out.

*   QuantizedType base class
*   UniformQuantizedType

### Quantized type conversion operations

*   qcast : Convert from an expressed type to QuantizedType
*   dcast : Convert from a QuantizedType to its expressed type
*   scast : Convert between a QuantizedType and its storage type

### Instrumentation and constraint operations

*   const_fake_quant : Emulates the logic of the historic TensorFlow
    fake_quant_with_min_max_args operation.
*   stats_ref : Declares that statistics should be gathered at this point with a
    unique key and made available to future passes of the solver.
*   stats : Declares inline statistics (per layer and per axis) for the point in
    the computation. stats_ref ops are generally converted to statistical operations once
    trial runs have been performed.
*   coupled_ref : Declares points in the computation to be coupled from a type
    inference perspective based on a unique key.

## Integration with simulated quantization at training time

TensorFlow has historically used the
[tf.quantization.fake_quant_\*](https://www.tensorflow.org/api_docs/python/tf/quantization/fake_quant_with_min_max_args)
family of operations to simulate the effect of quantization at training time.

As originally implemented, TensorFlow Lite was the primary user of such
operations at inference time. When quantized inference was enabled, if every
eligible tensor passed through an appropriate fake_quant node (the rules of
which tensors can have fake_quant applied are somewhat involved), then
TensorFlow Lite would use the attributes of the fake_quant operations to make a
judgment about how to convert to use kernels from its quantized operations subset.

In MLIR-based quantization, fake_quant_\* operations are handled by converting them to
a sequence of *qcast* (quantize) followed by *dcast* (dequantize) with an
appropriate *UniformQuantizedType* as the target of the qcast operation.

This allows subsequent compiler passes to preserve the knowledge that
quantization was simulated in a certain way, while giving the compiler
flexibility to move the casts as it simplifies the computation and converts it
to a form based on integral arithmetic.

This scheme also naturally allows computations that are *partially quantized*
where the parts which could not be reduced to integral operations are still carried out
in floating point with appropriate conversions at the boundaries.

## TFLite native quantization

TODO: Flesh this out

### General algorithm

1.  Take input min/max information and set the ArrayInfo (which really is
    InputOrOutputArrayInfo.
1.  In LegalizeTF, convert ArrayInfo min/max to tf.Quantize and tf.Dequantize
    nodes. (or tf.FakeQuant) Convert all constant FakeQuants to (tf.FQ -> tfl.Q
    -> tfl.DQ).
1.  Hardcode logic/propagation needs to happen here.
1.  Run TF constant folding.
1.  In PrepareTFL, convert all tf.FQ to (tfl.Q -> tfl.DQ).
1.  Run quantization pass that take (tfl.DQ (for both input and weights) -> op
    -> tfl.Q) and replaces with (op). Also replace (constant_float -> tfl.Q)
    with (constant_quant).
