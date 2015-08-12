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

/*
 * TODO(jfb): add and test these:
 * defm CEIL : UnaryFP<fceil>;
 * defm FLOOR : UnaryFP<ffloor>;
 * defm TRUNC : UnaryFP<ftrunc>;
 * defm NEARESTINT : UnaryFP<fnearbyint>;
 */

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

defm SQRT : UnaryFP<fsqrt>;

/*
 * TODO(jfb): Add the following for 32-bit and 64-bit.
 *
 * float32.min: minimum (binary operator); if either operand is NaN, returns NaN
 * float32.max: maximum (binary operator); if either operand is NaN, returns NaN
 */
