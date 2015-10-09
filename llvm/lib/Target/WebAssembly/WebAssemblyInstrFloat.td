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

defm ADD : BinaryFP<fadd>;
defm SUB : BinaryFP<fsub>;
defm MUL : BinaryFP<fmul>;
defm DIV : BinaryFP<fdiv>;
defm SQRT : UnaryFP<fsqrt>;

defm ABS : UnaryFP<fabs>;
defm NEG : UnaryFP<fneg>;
defm COPYSIGN : BinaryFP<fcopysign>;

defm CEIL : UnaryFP<fceil>;
defm FLOOR : UnaryFP<ffloor>;
defm TRUNC : UnaryFP<ftrunc>;
defm NEAREST : UnaryFP<fnearbyint>;

// WebAssembly doesn't expose inexact exceptions, so map frint to fnearbyint.
def : Pat<(frint f32:$src), (NEAREST_F32 f32:$src)>;
def : Pat<(frint f64:$src), (NEAREST_F64 f64:$src)>;

defm EQ : ComparisonFP<SETOEQ>;
defm NE : ComparisonFP<SETUNE>;
defm LT : ComparisonFP<SETOLT>;
defm LE : ComparisonFP<SETOLE>;
defm GT : ComparisonFP<SETOGT>;
defm GE : ComparisonFP<SETOGE>;

// Don't care floating-point comparisons, supported via other comparisons.
def : Pat<(seteq f32:$lhs, f32:$rhs), (EQ_F32 f32:$lhs, f32:$rhs)>;
def : Pat<(setne f32:$lhs, f32:$rhs), (NE_F32 f32:$lhs, f32:$rhs)>;
def : Pat<(setlt f32:$lhs, f32:$rhs), (LT_F32 f32:$lhs, f32:$rhs)>;
def : Pat<(setle f32:$lhs, f32:$rhs), (LE_F32 f32:$lhs, f32:$rhs)>;
def : Pat<(setgt f32:$lhs, f32:$rhs), (GT_F32 f32:$lhs, f32:$rhs)>;
def : Pat<(setge f32:$lhs, f32:$rhs), (GE_F32 f32:$lhs, f32:$rhs)>;
def : Pat<(seteq f64:$lhs, f64:$rhs), (EQ_F64 f64:$lhs, f64:$rhs)>;
def : Pat<(setne f64:$lhs, f64:$rhs), (NE_F64 f64:$lhs, f64:$rhs)>;
def : Pat<(setlt f64:$lhs, f64:$rhs), (LT_F64 f64:$lhs, f64:$rhs)>;
def : Pat<(setle f64:$lhs, f64:$rhs), (LE_F64 f64:$lhs, f64:$rhs)>;
def : Pat<(setgt f64:$lhs, f64:$rhs), (GT_F64 f64:$lhs, f64:$rhs)>;
def : Pat<(setge f64:$lhs, f64:$rhs), (GE_F64 f64:$lhs, f64:$rhs)>;

/*
 * TODO(jfb): Add the following for 32-bit and 64-bit.
 *
 * f32.min: minimum (binary operator); if either operand is NaN, returns NaN
 * f32.max: maximum (binary operator); if either operand is NaN, returns NaN
 */
