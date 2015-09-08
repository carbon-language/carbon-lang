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
defm SHR_U : BinaryInt<srl>;
defm SHR_S : BinaryInt<sra>;

defm EQ : ComparisonInt<SETEQ>;
defm NE : ComparisonInt<SETNE>;
defm SLT : ComparisonInt<SETLT>;
defm SLE : ComparisonInt<SETLE>;
defm ULT : ComparisonInt<SETULT>;
defm ULE : ComparisonInt<SETULE>;
defm SGT : ComparisonInt<SETGT>;
defm SGE : ComparisonInt<SETGE>;
defm UGT : ComparisonInt<SETUGT>;
defm UGE : ComparisonInt<SETUGE>;

defm CLZ : UnaryInt<ctlz>;
defm CTZ : UnaryInt<cttz>;
defm POPCNT : UnaryInt<ctpop>;

// Expand the "don't care" operations to supported operations.
def : Pat<(ctlz_zero_undef Int32:$src), (CLZ_I32 Int32:$src)>;
def : Pat<(ctlz_zero_undef Int64:$src), (CLZ_I64 Int64:$src)>;
def : Pat<(cttz_zero_undef Int32:$src), (CTZ_I32 Int32:$src)>;
def : Pat<(cttz_zero_undef Int64:$src), (CTZ_I64 Int64:$src)>;
