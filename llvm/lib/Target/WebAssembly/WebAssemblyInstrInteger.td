// WebAssemblyInstrInteger.td-WebAssembly Integer codegen -------*- tablegen -*-
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief WebAssembly Integer operand code-gen constructs.
///
//===----------------------------------------------------------------------===//

/*
 * TODO(jfb): Add the following for 32-bit and 64-bit.
 *
 * int32.add: signed-less addition
 * int32.sub: signed-less subtraction
 * int32.mul: signed-less multiplication (lower 32-bits)
 * int32.sdiv: signed division
 * int32.udiv: unsigned division
 * int32.srem: signed remainder
 * int32.urem: unsigned remainder
 * int32.and: signed-less logical and
 * int32.ior: signed-less inclusive or
 * int32.xor: signed-less exclusive or
 * int32.shl: signed-less shift left
 * int32.shr: signed-less logical shift right
 * int32.sar: signed-less arithmetic shift right
 * int32.eq: signed-less compare equal
 * int32.slt: signed less than
 * int32.sle: signed less than or equal
 * int32.ult: unsigned less than
 * int32.ule: unsigned less than or equal
 * int32.sgt: signed greater than
 * int32.sge: signed greater than or equal
 * int32.ugt: unsigned greater than
 * int32.uge: unsigned greater than or equal
 * int32.clz: count leading zeroes (defined for all values, including zero)
 * int32.ctz: count trailing zeroes (defined for all values, including zero)
 * int32.popcnt: count number of ones
 */
