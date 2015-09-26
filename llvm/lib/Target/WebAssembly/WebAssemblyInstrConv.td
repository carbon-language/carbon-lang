//===-- WebAssemblyInstrConv.td-WebAssembly Conversion support -*- tablegen -*-=
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief WebAssembly datatype conversions, truncations, reinterpretations,
/// promotions, and demotions operand code-gen constructs.
///
//===----------------------------------------------------------------------===//

/*
 * TODO(jfb): Add the following.
 *
 * i32.wrap[i64]: wrap a 64-bit integer to a 32-bit integer
 * i32.trunc_s[f32]: truncate a 32-bit float to a signed 32-bit integer
 * i32.trunc_s[f64]: truncate a 64-bit float to a signed 32-bit integer
 * i32.trunc_u[f32]: truncate a 32-bit float to an unsigned 32-bit integer
 * i32.trunc_u[f64]: truncate a 64-bit float to an unsigned 32-bit integer
 * i32.reinterpret[f32]: reinterpret the bits of a 32-bit float as a 32-bit integer
 * i64.extend_s[i32]: extend a signed 32-bit integer to a 64-bit integer
 * i64.extend_u[i32]: extend an unsigned 32-bit integer to a 64-bit integer
 * i64.trunc_s[f32]: truncate a 32-bit float to a signed 64-bit integer
 * i64.trunc_s[f64]: truncate a 64-bit float to a signed 64-bit integer
 * i64.trunc_u[f32]: truncate a 32-bit float to an unsigned 64-bit integer
 * i64.trunc_u[f64]: truncate a 64-bit float to an unsigned 64-bit integer
 * i64.reinterpret[f64]: reinterpret the bits of a 64-bit float as a 64-bit integer
 * f32.demote[f64]: demote a 64-bit float to a 32-bit float
 * f32.convert_s[i32]: convert a signed 32-bit integer to a 32-bit float
 * f32.convert_s[i64]: convert a signed 64-bit integer to a 32-bit float
 * f32.convert_u[i32]: convert an unsigned 32-bit integer to a 32-bit float
 * f32.convert_u[i64]: convert an unsigned 64-bit integer to a 32-bit float
 * f32.reinterpret[i32]: reinterpret the bits of a 32-bit integer as a 32-bit float
 * f64.promote[f32]: promote a 32-bit float to a 64-bit float
 * f64.convert_s[i32]: convert a signed 32-bit integer to a 64-bit float
 * f64.convert_s[i64]: convert a signed 64-bit integer to a 64-bit float
 * f64.convert_u[i32]: convert an unsigned 32-bit integer to a 64-bit float
 * f64.convert_u[i64]: convert an unsigned 64-bit integer to a 64-bit float
 * f64.reinterpret[i64]: reinterpret the bits of a 64-bit integer as a 64-bit float
 */

def WRAP_I64_I32 : I<(outs I32:$dst), (ins I64:$src),
                     [(set I32:$dst, (trunc I64:$src))]>;
