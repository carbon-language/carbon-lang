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
def LOAD_I32_ : I<(outs I32:$dst), (ins I32:$addr),
                  [(set I32:$dst, (load I32:$addr))]>;
def LOAD_I64_ : I<(outs I64:$dst), (ins I32:$addr),
                  [(set I64:$dst, (load I32:$addr))]>;
def LOAD_F32_ : I<(outs F32:$dst), (ins I32:$addr),
                  [(set F32:$dst, (load I32:$addr))]>;
def LOAD_F64_ : I<(outs F64:$dst), (ins I32:$addr),
                  [(set F64:$dst, (load I32:$addr))]>;

// Extending load.
def LOAD_S_i8_I32_  : I<(outs I32:$dst), (ins I32:$addr),
                        [(set I32:$dst, (sextloadi8 I32:$addr))]>;
def LOAD_U_i8_I32_  : I<(outs I32:$dst), (ins I32:$addr),
                        [(set I32:$dst, (zextloadi8 I32:$addr))]>;
def LOAD_S_i16_I32_ : I<(outs I32:$dst), (ins I32:$addr),
                        [(set I32:$dst, (sextloadi16 I32:$addr))]>;
def LOAD_U_i16_I32_ : I<(outs I32:$dst), (ins I32:$addr),
                        [(set I32:$dst, (zextloadi16 I32:$addr))]>;
def LOAD_S_i8_I64_  : I<(outs I64:$dst), (ins I32:$addr),
                        [(set I64:$dst, (sextloadi8 I32:$addr))]>;
def LOAD_U_i8_I64_  : I<(outs I64:$dst), (ins I32:$addr),
                        [(set I64:$dst, (zextloadi8 I32:$addr))]>;
def LOAD_S_i16_I64_ : I<(outs I64:$dst), (ins I32:$addr),
                        [(set I64:$dst, (sextloadi16 I32:$addr))]>;
def LOAD_U_i16_I64_ : I<(outs I64:$dst), (ins I32:$addr),
                        [(set I64:$dst, (zextloadi16 I32:$addr))]>;
def LOAD_S_I32_I64_ : I<(outs I64:$dst), (ins I32:$addr),
                        [(set I64:$dst, (sextloadi32 I32:$addr))]>;
def LOAD_U_I32_I64_ : I<(outs I64:$dst), (ins I32:$addr),
                        [(set I64:$dst, (zextloadi32 I32:$addr))]>;

// "Don't care" extending load become zero-extending load.
def : Pat<(i32 (extloadi8 I32:$addr)),  (LOAD_U_i8_I32_ $addr)>;
def : Pat<(i32 (extloadi16 I32:$addr)), (LOAD_U_i16_I32_ $addr)>;
def : Pat<(i64 (extloadi8 I32:$addr)),  (LOAD_U_i8_I64_ $addr)>;
def : Pat<(i64 (extloadi16 I32:$addr)), (LOAD_U_i16_I64_ $addr)>;
def : Pat<(i64 (extloadi32 I32:$addr)), (LOAD_U_I32_I64_ $addr)>;

// Basic store.
// Note: WebAssembly inverts SelectionDAG's usual operand order.
def STORE_I32_  : I<(outs), (ins I32:$addr, I32:$val),
                    [(store i32:$val, I32:$addr)]>;
def STORE_I64_  : I<(outs), (ins I32:$addr, I64:$val),
                    [(store i64:$val, I32:$addr)]>;
def STORE_F32_  : I<(outs), (ins I32:$addr, F32:$val),
                    [(store f32:$val, I32:$addr)]>;
def STORE_F64_  : I<(outs), (ins I32:$addr, F64:$val),
                    [(store f64:$val, I32:$addr)]>;

// Truncating store.
def STORE_i8_I32  : I<(outs), (ins I32:$addr, I32:$val),
                      [(truncstorei8 I32:$val, I32:$addr)]>;
def STORE_i16_I32 : I<(outs), (ins I32:$addr, I32:$val),
                      [(truncstorei16 I32:$val, I32:$addr)]>;
def STORE_i8_I64  : I<(outs), (ins I32:$addr, I64:$val),
                      [(truncstorei8 I64:$val, I32:$addr)]>;
def STORE_i16_I64 : I<(outs), (ins I32:$addr, I64:$val),
                      [(truncstorei16 I64:$val, I32:$addr)]>;
def STORE_I32_I64 : I<(outs), (ins I32:$addr, I64:$val),
                      [(truncstorei32 I64:$val, I32:$addr)]>;

// Page size.
def page_size_I32 : I<(outs I32:$dst), (ins),
                      [(set I32:$dst, (int_wasm_page_size))]>,
                    Requires<[HasAddr32]>;
def page_size_I64 : I<(outs I64:$dst), (ins),
                      [(set I64:$dst, (int_wasm_page_size))]>,
                    Requires<[HasAddr64]>;

// Memory size.
def memory_size_I32 : I<(outs I32:$dst), (ins),
                        [(set I32:$dst, (int_wasm_memory_size))]>,
                      Requires<[HasAddr32]>;
def memory_size_I64 : I<(outs I64:$dst), (ins),
                        [(set I64:$dst, (int_wasm_memory_size))]>,
                      Requires<[HasAddr64]>;
