// WebAssemblyInstrMemory.td-WebAssembly Memory codegen support -*- tablegen -*-
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief WebAssembly Memory operand code-gen constructs.
///
//===----------------------------------------------------------------------===//

/*
 * TODO(jfb): Add the following.
 * Each has optional alignment and immediate byte offset.
 *
 * int32.load_sx[int8]: sign-extend to int32
 * int32.load_sx[int16]: sign-extend to int32
 * int32.load_zx[int8]: zero-extend to int32
 * int32.load_zx[int16]: zero-extend to int32
 * int32.load[int32]: (no conversion)
 * int64.load_sx[int8]: sign-extend to int64
 * int64.load_sx[int16]: sign-extend to int64
 * int64.load_sx[int32]: sign-extend to int64
 * int64.load_zx[int8]: zero-extend to int64
 * int64.load_zx[int16]: zero-extend to int64
 * int64.load_zx[int32]: zero-extend to int64
 * int64.load[int64]: (no conversion)
 * float32.load[float32]: (no conversion)
 * float64.load[float64]: (no conversion)
 * 
 * int32.store[int8]: wrap int32 to int8
 * int32.store[int16]: wrap int32 to int16
 * int32.store[int32]: (no conversion)
 * int64.store[int8]: wrap int64 to int8
 * int64.store[int16]: wrap int64 to int16
 * int64.store[int32]: wrap int64 to int32
 * int64.store[int64]: (no conversion)
 * float32.store[float32]: (no conversion)
 * float64.store[float64]: (no conversion)
 * 
 * load_global: load the value of a given global variable
 * store_global: store a given value to a given global variable
 */
