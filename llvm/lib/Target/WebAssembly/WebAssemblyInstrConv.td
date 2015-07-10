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
 * int32.wrap[int64]: wrap a 64-bit integer to a 32-bit integer
 * int32.trunc_signed[float32]: truncate a 32-bit float to a signed 32-bit integer
 * int32.trunc_signed[float64]: truncate a 64-bit float to a signed 32-bit integer
 * int32.trunc_unsigned[float32]: truncate a 32-bit float to an unsigned 32-bit integer
 * int32.trunc_unsigned[float64]: truncate a 64-bit float to an unsigned 32-bit integer
 * int32.reinterpret[float32]: reinterpret the bits of a 32-bit float as a 32-bit integer
 * int64.extend_signed[int32]: extend a signed 32-bit integer to a 64-bit integer
 * int64.extend_unsigned[int32]: extend an unsigned 32-bit integer to a 64-bit integer
 * int64.trunc_signed[float32]: truncate a 32-bit float to a signed 64-bit integer
 * int64.trunc_signed[float64]: truncate a 64-bit float to a signed 64-bit integer
 * int64.trunc_unsigned[float32]: truncate a 32-bit float to an unsigned 64-bit integer
 * int64.trunc_unsigned[float64]: truncate a 64-bit float to an unsigned 64-bit integer
 * int64.reinterpret[float64]: reinterpret the bits of a 64-bit float as a 64-bit integer
 * float32.demote[float64]: demote a 64-bit float to a 32-bit float
 * float32.cvt_signed[int32]: convert a signed 32-bit integer to a 32-bit float
 * float32.cvt_signed[int64]: convert a signed 64-bit integer to a 32-bit float
 * float32.cvt_unsigned[int32]: convert an unsigned 32-bit integer to a 32-bit float
 * float32.cvt_unsigned[int64]: convert an unsigned 64-bit integer to a 32-bit float
 * float32.reinterpret[int32]: reinterpret the bits of a 32-bit integer as a 32-bit float
 * float64.promote[float32]: promote a 32-bit float to a 64-bit float
 * float64.cvt_signed[int32]: convert a signed 32-bit integer to a 64-bit float
 * float64.cvt_signed[int64]: convert a signed 64-bit integer to a 64-bit float
 * float64.cvt_unsigned[int32]: convert an unsigned 32-bit integer to a 64-bit float
 * float64.cvt_unsigned[int64]: convert an unsigned 64-bit integer to a 64-bit float
 * float64.reinterpret[int64]: reinterpret the bits of a 64-bit integer as a 64-bit float
 */
