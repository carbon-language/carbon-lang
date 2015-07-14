// WebAssemblyInstrFloat.td-WebAssembly Float codegen support ---*- tablegen -*-
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief WebAssembly Floating-point operand code-gen constructs.
///
//===----------------------------------------------------------------------===//

defm FADD : BinaryFP<fadd>;
defm FSUB : BinaryFP<fsub>;
defm FMUL : BinaryFP<fmul>;
defm FDIV : BinaryFP<fdiv>;
defm FABS : UnaryFP<fabs>;
defm FNEG : UnaryFP<fneg>;
defm COPYSIGN : BinaryFP<fcopysign>;
defm CEIL : UnaryFP<fceil>;
defm FLOOR : UnaryFP<ffloor>;
defm TRUNC : UnaryFP<ftrunc>;
defm NEARESTINT : UnaryFP<fnearbyint>;

/*
 * TODO(jfb): Add the following for 32-bit and 64-bit.
 *
 * float32.eq: compare equal
 * float32.lt: less than
 * float32.le: less than or equal
 * float32.gt: greater than
 * float32.ge: greater than or equal
 */

defm SQRT : UnaryFP<fsqrt>;

/*
 * TODO(jfb): Add the following for 32-bit and 64-bit.
 *
 * float32.min: minimum (binary operator); if either operand is NaN, returns NaN
 * float32.max: maximum (binary operator); if either operand is NaN, returns NaN
 */
