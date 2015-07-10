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

/*
 * TODO(jfb): Add the following for 32-bit and 64-bit.
 *
 * float32.add: addition
 * float32.sub: subtraction
 * float32.mul: multiplication
 * float32.div: division
 * float32.abs: absolute value
 * float32.neg: negation
 * float32.copysign: copysign
 * float32.ceil: ceiling operation
 * float32.floor: floor operation
 * float32.trunc: round to nearest integer towards zero
 * float32.nearestint: round to nearest integer, ties to even
 * float32.eq: compare equal
 * float32.lt: less than
 * float32.le: less than or equal
 * float32.gt: greater than
 * float32.ge: greater than or equal
 * float32.sqrt: square root
 * float32.min: minimum (binary operator); if either operand is NaN, returns NaN
 * float32.max: maximum (binary operator); if either operand is NaN, returns NaN
 */
