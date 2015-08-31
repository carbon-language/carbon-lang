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
 * 
 * load_global: load the value of a given global variable
 * store_global: store a given value to a given global variable
 */

// FIXME:
//  - HasAddr64
//  - WebAssemblyTargetLowering::isLegalAddressingMode
//  - WebAssemblyTargetLowering having to do with atomics
//  - Each has optional alignment and immediate byte offset.

// WebAssembly has i8/i16/i32/i64/f32/f64 memory types, but doesn't have i8/i16
// local types. These memory-only types instead zero- or sign-extend into local
// types when loading, and truncate when storing.

// Basic load.
def LOAD_I32_ : I<(outs Int32:$dst), (ins Int32:$addr),
                  [(set Int32:$dst, (load Int32:$addr))]>;
def LOAD_I64_ : I<(outs Int64:$dst), (ins Int32:$addr),
                  [(set Int64:$dst, (load Int32:$addr))]>;
def LOAD_F32_ : I<(outs Float32:$dst), (ins Int32:$addr),
                  [(set Float32:$dst, (load Int32:$addr))]>;
def LOAD_F64_ : I<(outs Float64:$dst), (ins Int32:$addr),
                  [(set Float64:$dst, (load Int32:$addr))]>;

// Extending load.
def LOAD_SX_I8_I32_  : I<(outs Int32:$dst), (ins Int32:$addr),
                         [(set Int32:$dst, (sextloadi8 Int32:$addr))]>;
def LOAD_ZX_I8_I32_  : I<(outs Int32:$dst), (ins Int32:$addr),
                         [(set Int32:$dst, (zextloadi8 Int32:$addr))]>;
def LOAD_SX_I16_I32_ : I<(outs Int32:$dst), (ins Int32:$addr),
                         [(set Int32:$dst, (sextloadi16 Int32:$addr))]>;
def LOAD_ZX_I16_I32_ : I<(outs Int32:$dst), (ins Int32:$addr),
                         [(set Int32:$dst, (zextloadi16 Int32:$addr))]>;
def LOAD_SX_I8_I64_  : I<(outs Int64:$dst), (ins Int32:$addr),
                         [(set Int64:$dst, (sextloadi8 Int32:$addr))]>;
def LOAD_ZX_I8_I64_  : I<(outs Int64:$dst), (ins Int32:$addr),
                         [(set Int64:$dst, (zextloadi8 Int32:$addr))]>;
def LOAD_SX_I16_I64_ : I<(outs Int64:$dst), (ins Int32:$addr),
                         [(set Int64:$dst, (sextloadi16 Int32:$addr))]>;
def LOAD_ZX_I16_I64_ : I<(outs Int64:$dst), (ins Int32:$addr),
                         [(set Int64:$dst, (zextloadi16 Int32:$addr))]>;
def LOAD_SX_I32_I64_ : I<(outs Int64:$dst), (ins Int32:$addr),
                         [(set Int64:$dst, (sextloadi32 Int32:$addr))]>;
def LOAD_ZX_I32_I64_ : I<(outs Int64:$dst), (ins Int32:$addr),
                         [(set Int64:$dst, (zextloadi32 Int32:$addr))]>;

// "Don't care" extending load become zero-extending load.
def : Pat<(i32 (extloadi8 Int32:$addr)),  (LOAD_ZX_I8_I32_ $addr)>;
def : Pat<(i32 (extloadi16 Int32:$addr)), (LOAD_ZX_I16_I32_ $addr)>;
def : Pat<(i64 (extloadi8 Int32:$addr)),  (LOAD_ZX_I8_I64_ $addr)>;
def : Pat<(i64 (extloadi16 Int32:$addr)), (LOAD_ZX_I16_I64_ $addr)>;
def : Pat<(i64 (extloadi32 Int32:$addr)), (LOAD_ZX_I32_I64_ $addr)>;

// Basic store.
// Note: WebAssembly inverts SelectionDAG's usual operand order.
def STORE_I32_  : I<(outs), (ins Int32:$addr, Int32:$val),
                    [(store Int32:$val, Int32:$addr)]>;
def STORE_I64_  : I<(outs), (ins Int32:$addr, Int64:$val),
                    [(store Int64:$val, Int32:$addr)]>;
def STORE_F32_  : I<(outs), (ins Int32:$addr, Float32:$val),
                    [(store Float32:$val, Int32:$addr)]>;
def STORE_F64_  : I<(outs), (ins Int32:$addr, Float64:$val),
                    [(store Float64:$val, Int32:$addr)]>;

// Truncating store.
def STORE_I8_I32  : I<(outs), (ins Int32:$addr, Int32:$val),
                      [(truncstorei8 Int32:$val, Int32:$addr)]>;
def STORE_I16_I32 : I<(outs), (ins Int32:$addr, Int32:$val),
                      [(truncstorei16 Int32:$val, Int32:$addr)]>;
def STORE_I8_I64  : I<(outs), (ins Int32:$addr, Int64:$val),
                      [(truncstorei8 Int64:$val, Int32:$addr)]>;
def STORE_I16_I64 : I<(outs), (ins Int32:$addr, Int64:$val),
                      [(truncstorei16 Int64:$val, Int32:$addr)]>;
def STORE_I32_I64 : I<(outs), (ins Int32:$addr, Int64:$val),
                      [(truncstorei32 Int64:$val, Int32:$addr)]>;

// Page size.
def page_size_I32 : I<(outs Int32:$dst), (ins),
                      [(set Int32:$dst, (int_wasm_page_size))]>,
                    Requires<[HasAddr32]>;
def page_size_I64 : I<(outs Int64:$dst), (ins),
                      [(set Int64:$dst, (int_wasm_page_size))]>,
                    Requires<[HasAddr64]>;
