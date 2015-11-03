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
defm DIV_S : BinaryInt<sdiv>;
defm DIV_U : BinaryInt<udiv>;
defm REM_S : BinaryInt<srem>;
defm REM_U : BinaryInt<urem>;
defm AND : BinaryInt<and>;
defm IOR : BinaryInt<or>;
defm XOR : BinaryInt<xor>;
defm SHL : BinaryInt<shl>;
defm SHR_U : BinaryInt<srl>;
defm SHR_S : BinaryInt<sra>;

defm EQ : ComparisonInt<SETEQ>;
defm NE : ComparisonInt<SETNE>;
defm LT_S : ComparisonInt<SETLT>;
defm LE_S : ComparisonInt<SETLE>;
defm LT_U : ComparisonInt<SETULT>;
defm LE_U : ComparisonInt<SETULE>;
defm GT_S : ComparisonInt<SETGT>;
defm GE_S : ComparisonInt<SETGE>;
defm GT_U : ComparisonInt<SETUGT>;
defm GE_U : ComparisonInt<SETUGE>;

defm CLZ : UnaryInt<ctlz>;
defm CTZ : UnaryInt<cttz>;
defm POPCNT : UnaryInt<ctpop>;

// Expand the "don't care" operations to supported operations.
def : Pat<(ctlz_zero_undef I32:$src), (CLZ_I32 I32:$src)>;
def : Pat<(ctlz_zero_undef I64:$src), (CLZ_I64 I64:$src)>;
def : Pat<(cttz_zero_undef I32:$src), (CTZ_I32 I32:$src)>;
def : Pat<(cttz_zero_undef I64:$src), (CTZ_I64 I64:$src)>;

def SELECT_I32 : I<(outs I32:$dst), (ins I32:$cond, I32:$lhs, I32:$rhs),
                   [(set I32:$dst, (select I32:$cond, I32:$lhs, I32:$rhs))]>;
def SELECT_I64 : I<(outs I64:$dst), (ins I32:$cond, I64:$lhs, I64:$rhs),
                   [(set I64:$dst, (select I32:$cond, I64:$lhs, I64:$rhs))]>;
