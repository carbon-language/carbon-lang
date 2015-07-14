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

defm ADD : BinaryInt<add>;
defm SUB : BinaryInt<sub>;
defm MUL : BinaryInt<mul>;
defm SDIV : BinaryInt<sdiv>;
defm UDIV : BinaryInt<udiv>;
defm SREM : BinaryInt<srem>;
defm UREM : BinaryInt<urem>;
defm AND : BinaryInt<and>;
defm IOR : BinaryInt<or>;
defm XOR : BinaryInt<xor>;
defm SHL : BinaryInt<shl>;
defm SHR : BinaryInt<srl>;
defm SAR : BinaryInt<sra>;

/*
 * TODO(jfb): Add the following for 32-bit and 64-bit.
 *
 * int32.eq: signed-less compare equal
 * int32.slt: signed less than
 * int32.sle: signed less than or equal
 * int32.ult: unsigned less than
 * int32.ule: unsigned less than or equal
 * int32.sgt: signed greater than
 * int32.sge: signed greater than or equal
 * int32.ugt: unsigned greater than
 * int32.uge: unsigned greater than or equal
 */

defm CLZ : UnaryInt<ctlz>;
defm CTZ : UnaryInt<cttz>;
defm POPCNT : UnaryInt<ctpop>;
