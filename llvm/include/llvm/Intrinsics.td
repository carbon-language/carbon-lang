//===- Intrinsics.td - Defines all LLVM intrinsics ---------*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines properties of all LLVM intrinsics.
//
//===----------------------------------------------------------------------===//

include "llvm/CodeGen/ValueTypes.td"

//===----------------------------------------------------------------------===//
//  Properties we keep track of for intrinsics.
//===----------------------------------------------------------------------===//

class IntrinsicProperty;

// Intr*Mem - Memory properties.  An intrinsic is allowed to have exactly one of
// these properties set.  They are listed from the most aggressive (best to use
// if correct) to the least aggressive.  If no property is set, the worst case 
// is assumed (IntrWriteMem).

// InstrNoMem - The intrinsic does not access memory or have any other side
// effects.  It may be CSE'd deleted if dead, etc.
def InstrNoMem : IntrinsicProperty;

// InstrReadArgMem - This intrinsic reads only from memory that one of its
// arguments points to, but may read an unspecified amount.
def InstrReadArgMem : IntrinsicProperty;

// IntrReadMem - This intrinsic reads from unspecified memory, so it cannot be
// moved across stores.  However, it can be reordered otherwise and can be
// deleted if dead.
def IntrReadMem : IntrinsicProperty;

// InstrWriteArgMem - This intrinsic reads and writes only from memory that one
// of its arguments points to, but may access an unspecified amount.  It has no
// other side effects.  This may only be used if the intrinsic doesn't "capture"
// the argument pointer (e.g. storing it someplace).
def InstrWriteArgMem : IntrinsicProperty;

// IntrWriteMem - This intrinsic may read or modify unspecified memory or has 
// other side effects.  It cannot be modified by the optimizer.  This is the
// default if the intrinsic has no other Intr*Mem property.
def IntrWriteMem : IntrinsicProperty;

//===----------------------------------------------------------------------===//
// Types used by intrinsics.
//===----------------------------------------------------------------------===//

class LLVMType<ValueType vt, string typeval> {
  ValueType VT = vt;
  string TypeVal = typeval;
}

class LLVMPackedType<ValueType VT, int numelts, LLVMType elty>
  : LLVMType<VT, "Type::PackedTyID">{
  int NumElts = numelts;
  LLVMType ElTy = elty;
} 

def llvm_void_ty       : LLVMType<isVoid, "Type::VoidTyID">;
def llvm_bool_ty       : LLVMType<i1 , "Type::BoolTyID">;
def llvm_sbyte_ty      : LLVMType<i8 , "Type::SByteTyID">;
def llvm_short_ty      : LLVMType<i16, "Type::ShortTyID">;
def llvm_int_ty        : LLVMType<i32, "Type::IntTyID">;
def llvm_long_ty       : LLVMType<i64, "Type::LongTyID">;
def llvm_ubyte_ty      : LLVMType<i8,  "Type::UByteTyID">;
def llvm_ushort_ty     : LLVMType<i16, "Type::UShortTyID">;
def llvm_uint_ty       : LLVMType<i32, "Type::UIntTyID">;
def llvm_ulong_ty      : LLVMType<i64, "Type::ULongTyID">;
def llvm_float_ty      : LLVMType<f32, "Type::FloatTyID">;
def llvm_double_ty     : LLVMType<f64, "Type::DoubleTyID">;
def llvm_ptr_ty        : LLVMType<OtherVT, "Type::PointerTyID">;     // sbyte*
def llvm_ptrptr_ty     : LLVMType<OtherVT, "Type::PointerTyID">;     // sbyte**
def llvm_descriptor_ty : LLVMType<OtherVT, "Type::PointerTyID">;     // global*

def llvm_v2i32_ty      : LLVMPackedType<v2i32, 2, llvm_int_ty>;    // 2 x int
def llvm_v4i32_ty      : LLVMPackedType<v4i32, 4, llvm_int_ty>;    // 4 x int
def llvm_v4f32_ty      : LLVMPackedType<v4f32, 4, llvm_float_ty>;  // 4 x float
def llvm_v2f64_ty      : LLVMPackedType<v2f64, 2, llvm_double_ty>; // 2 x double

//===----------------------------------------------------------------------===//
// Intrinsic Definitions.
//===----------------------------------------------------------------------===//

// Intrinsic class - This is used to define one LLVM intrinsic.  The name of the
// intrinsic definition should start with "int_", then match the LLVM intrinsic
// name with the "llvm." prefix removed, and all "."s turned into "_"s.  For
// example, llvm.bswap.i16 -> int_bswap_i16.
//
//  * Types is a list containing the return type and the argument types
//    expected for the intrinsic.
//  * Properties can be set to describe the behavior of the intrinsic.
//
class Intrinsic<list<LLVMType> types,
                list<IntrinsicProperty> properties = [],
                string name = ""> {
  string LLVMName = name;
  string TargetPrefix = "";   // Set to a prefix for target-specific intrinsics.
  list<LLVMType> Types = types;
  list<IntrinsicProperty> Properties = properties;
}

/// GCCBuiltin - If this intrinsic exactly corresponds to a GCC builtin, this
/// specifies the name of the builtin.  This provides automatic CBE and CFE
/// support.
class GCCBuiltin<string name> {
  string GCCBuiltinName = name;
}


//===--------------- Variable Argument Handling Intrinsics ----------------===//
//  

def int_vastart : Intrinsic<[llvm_void_ty, llvm_ptrptr_ty], [], "llvm.va_start">;
def int_vacopy  : Intrinsic<[llvm_void_ty, llvm_ptrptr_ty, llvm_ptrptr_ty], [],
                            "llvm.va_copy">;
def int_vaend   : Intrinsic<[llvm_void_ty, llvm_ptrptr_ty], [], "llvm.va_end">;

//===------------------- Garbage Collection Intrinsics --------------------===//
//  
def int_gcroot  : Intrinsic<[llvm_void_ty, llvm_ptrptr_ty, llvm_ptr_ty]>;
def int_gcread  : Intrinsic<[llvm_ptr_ty, llvm_ptr_ty, llvm_ptrptr_ty],
                            [InstrReadArgMem]>;
def int_gcwrite : Intrinsic<[llvm_void_ty, llvm_ptr_ty, llvm_ptr_ty,
                             llvm_ptrptr_ty], [InstrWriteArgMem]>;

//===--------------------- Code Generator Intrinsics ----------------------===//
//  
def int_returnaddress : Intrinsic<[llvm_ptr_ty, llvm_uint_ty], [InstrNoMem]>;
def int_frameaddress  : Intrinsic<[llvm_ptr_ty, llvm_uint_ty], [InstrNoMem]>;
def int_stacksave     : Intrinsic<[llvm_ptr_ty], [IntrReadMem]>;
def int_stackrestore  : Intrinsic<[llvm_void_ty, llvm_ptr_ty]>;
def int_prefetch      : Intrinsic<[llvm_void_ty, llvm_ptr_ty, 
                                   llvm_uint_ty, llvm_uint_ty]>;
def int_pcmarker      : Intrinsic<[llvm_void_ty, llvm_uint_ty]>;

def int_readcyclecounter : Intrinsic<[llvm_ulong_ty]>;

//===------------------- Standard C Library Intrinsics --------------------===//
//

let Properties = [InstrWriteArgMem] in {
  def int_memcpy_i32  : Intrinsic<[llvm_void_ty, llvm_ptr_ty, llvm_ptr_ty,
                                   llvm_uint_ty, llvm_uint_ty]>;
  def int_memcpy_i64  : Intrinsic<[llvm_void_ty, llvm_ptr_ty, llvm_ptr_ty,
                                   llvm_ulong_ty, llvm_uint_ty]>;
  def int_memmove_i32 : Intrinsic<[llvm_void_ty, llvm_ptr_ty, llvm_ptr_ty,
                                   llvm_uint_ty, llvm_uint_ty]>;
  def int_memmove_i64 : Intrinsic<[llvm_void_ty, llvm_ptr_ty, llvm_ptr_ty,
                                   llvm_ulong_ty, llvm_uint_ty]>;
  def int_memset_i32  : Intrinsic<[llvm_void_ty, llvm_ptr_ty, llvm_ubyte_ty,
                                   llvm_uint_ty, llvm_uint_ty]>;
  def int_memset_i64  : Intrinsic<[llvm_void_ty, llvm_ptr_ty, llvm_ubyte_ty,
                                   llvm_ulong_ty, llvm_uint_ty]>;
}

let Properties = [InstrNoMem] in {
  def int_isunordered_f32 : Intrinsic<[llvm_bool_ty, 
                                       llvm_float_ty,  llvm_float_ty]>;
  def int_isunordered_f64 : Intrinsic<[llvm_bool_ty, 
                                       llvm_double_ty, llvm_double_ty]>;
  def int_sqrt_f32 : Intrinsic<[llvm_float_ty , llvm_float_ty]>;
  def int_sqrt_f64 : Intrinsic<[llvm_double_ty, llvm_double_ty]>;
}

// NOTE: these are internal interfaces.
def int_setjmp     : Intrinsic<[llvm_int_ty , llvm_ptr_ty]>;
def int_longjmp    : Intrinsic<[llvm_void_ty, llvm_ptr_ty, llvm_int_ty]>;
def int_sigsetjmp  : Intrinsic<[llvm_int_ty , llvm_ptr_ty]>;
def int_siglongjmp : Intrinsic<[llvm_void_ty, llvm_ptr_ty, llvm_int_ty]>;

//===-------------------- Bit Manipulation Intrinsics ---------------------===//
//

// None of these intrinsics accesses memory at all.
let Properties = [InstrNoMem] in {
  def int_bswap_i16 : Intrinsic<[llvm_ushort_ty, llvm_ushort_ty]>;
  def int_bswap_i32 : Intrinsic<[llvm_uint_ty,   llvm_uint_ty]>;
  def int_bswap_i64 : Intrinsic<[llvm_ulong_ty,  llvm_ulong_ty]>;

  def int_ctpop_i8  : Intrinsic<[llvm_ubyte_ty,  llvm_ubyte_ty]>;
  def int_ctpop_i16 : Intrinsic<[llvm_ushort_ty, llvm_ushort_ty]>;
  def int_ctpop_i32 : Intrinsic<[llvm_uint_ty,   llvm_uint_ty]>;
  def int_ctpop_i64 : Intrinsic<[llvm_ulong_ty,  llvm_ulong_ty]>;
  
  def int_ctlz_i8  : Intrinsic<[llvm_ubyte_ty,  llvm_ubyte_ty]>;
  def int_ctlz_i16 : Intrinsic<[llvm_ushort_ty, llvm_ushort_ty]>;
  def int_ctlz_i32 : Intrinsic<[llvm_uint_ty,   llvm_uint_ty]>;
  def int_ctlz_i64 : Intrinsic<[llvm_ulong_ty,  llvm_ulong_ty]>;

  def int_cttz_i8  : Intrinsic<[llvm_ubyte_ty,  llvm_ubyte_ty]>;
  def int_cttz_i16 : Intrinsic<[llvm_ushort_ty, llvm_ushort_ty]>;
  def int_cttz_i32 : Intrinsic<[llvm_uint_ty,   llvm_uint_ty]>;
  def int_cttz_i64 : Intrinsic<[llvm_ulong_ty,  llvm_ulong_ty]>;
} 

//===------------------------ Debugger Intrinsics -------------------------===//
//

def int_dbg_stoppoint    : Intrinsic<[llvm_void_ty,
                                      llvm_uint_ty, llvm_uint_ty, 
                                      llvm_descriptor_ty]>;
def int_dbg_region_start : Intrinsic<[llvm_void_ty, llvm_descriptor_ty]>;
def int_dbg_region_end   : Intrinsic<[llvm_void_ty, llvm_descriptor_ty]>;
def int_dbg_func_start   : Intrinsic<[llvm_void_ty, llvm_descriptor_ty]>;
def int_dbg_declare      : Intrinsic<[llvm_void_ty, llvm_ptr_ty,
                                                    llvm_descriptor_ty]>;

//===----------------------------------------------------------------------===//
// Target-specific intrinsics
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// PowerPC Intrinsics
//
let TargetPrefix = "ppc" in {  // All intrinsics start with "llvm.ppc.".
  def int_ppc_altivec_lvx : GCCBuiltin<"__builtin_altivec_lvx">,
              Intrinsic<[llvm_v4i32_ty, llvm_int_ty, llvm_ptr_ty],
                        [IntrReadMem]>;
  def int_ppc_altivec_stvx : GCCBuiltin<"__builtin_altivec_stvx">,
              Intrinsic<[llvm_void_ty, llvm_v4i32_ty, llvm_int_ty, llvm_ptr_ty],
                        [IntrWriteMem]>;
                            
  def int_ppc_altivec_vmaddfp : GCCBuiltin<"__builtin_altivec_vmaddfp">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty, llvm_v4f32_ty], [InstrNoMem]>;
  def int_ppc_altivec_vadduwm : GCCBuiltin<"__builtin_altivec_vadduwm">,
              Intrinsic<[llvm_v4i32_ty, llvm_v4i32_ty, llvm_v4i32_ty],
                        [InstrNoMem]>;
  
}


//===----------------------------------------------------------------------===//
// X86 Intrinsics
//
// SSE1

// Arithmetic ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_addss : GCCBuiltin<"__builtin_ia32_addss">,
              Intrinsic<[llvm_float_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_addps : GCCBuiltin<"__builtin_ia32_addps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_subss : GCCBuiltin<"__builtin_ia32_subss">,
              Intrinsic<[llvm_float_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_subps : GCCBuiltin<"__builtin_ia32_subps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_mulss : GCCBuiltin<"__builtin_ia32_mulss">,
              Intrinsic<[llvm_float_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_mulps : GCCBuiltin<"__builtin_ia32_mulps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_divss : GCCBuiltin<"__builtin_ia32_divss">,
              Intrinsic<[llvm_float_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_divps : GCCBuiltin<"__builtin_ia32_divps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_sqrtss : GCCBuiltin<"__builtin_ia32_sqrtss">,
              Intrinsic<[llvm_float_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_sqrtps : GCCBuiltin<"__builtin_ia32_sqrtps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_rcpss : GCCBuiltin<"__builtin_ia32_rcpss">,
              Intrinsic<[llvm_float_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_rcpps : GCCBuiltin<"__builtin_ia32_rcpps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_rsqrtss : GCCBuiltin<"__builtin_ia32_rsqrtss">,
              Intrinsic<[llvm_float_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_rsqrtps : GCCBuiltin<"__builtin_ia32_rsqrtps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_minss : GCCBuiltin<"__builtin_ia32_minss">,
              Intrinsic<[llvm_float_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_minps : GCCBuiltin<"__builtin_ia32_minps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_maxss : GCCBuiltin<"__builtin_ia32_maxss">,
              Intrinsic<[llvm_float_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_maxps : GCCBuiltin<"__builtin_ia32_maxps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
}

// Logical ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_andps : GCCBuiltin<"__builtin_ia32_andps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_andnotps : GCCBuiltin<"__builtin_ia32_andnotps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_orps : GCCBuiltin<"__builtin_ia32_orps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_xorps : GCCBuiltin<"__builtin_ia32_xorps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
}

// Comparison ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_cmpeqss : GCCBuiltin<"__builtin_ia32_cmpeqss">,
              Intrinsic<[llvm_float_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_cmpeqps : GCCBuiltin<"__builtin_ia32_cmpeqps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_cmpltss : GCCBuiltin<"__builtin_ia32_cmpltss">,
              Intrinsic<[llvm_float_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_cmpltps : GCCBuiltin<"__builtin_ia32_cmpltps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_cmpless : GCCBuiltin<"__builtin_ia32_cmpless">,
              Intrinsic<[llvm_float_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_cmpleps : GCCBuiltin<"__builtin_ia32_cmpleps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_cmpgtss : GCCBuiltin<"__builtin_ia32_cmpgtss">,
              Intrinsic<[llvm_float_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_cmpgtps : GCCBuiltin<"__builtin_ia32_cmpgtps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_cmpgess : GCCBuiltin<"__builtin_ia32_cmpgess">,
              Intrinsic<[llvm_float_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_cmpgeps : GCCBuiltin<"__builtin_ia32_cmpgeps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_cmpneqss : GCCBuiltin<"__builtin_ia32_cmpneqss">,
              Intrinsic<[llvm_float_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_cmpneqps : GCCBuiltin<"__builtin_ia32_cmpneqps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_cmpnltss : GCCBuiltin<"__builtin_ia32_cmpnltss">,
              Intrinsic<[llvm_float_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_cmpnltps : GCCBuiltin<"__builtin_ia32_cmpnltps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_cmpnless : GCCBuiltin<"__builtin_ia32_cmpnless">,
              Intrinsic<[llvm_float_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_cmpnleps : GCCBuiltin<"__builtin_ia32_cmpnleps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_cmpngtss : GCCBuiltin<"__builtin_ia32_cmpngtss">,
              Intrinsic<[llvm_float_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_cmpngtps : GCCBuiltin<"__builtin_ia32_cmpngtps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_cmpngess : GCCBuiltin<"__builtin_ia32_cmpngess">,
              Intrinsic<[llvm_float_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_cmpngeps : GCCBuiltin<"__builtin_ia32_cmpngeps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_cmpordss : GCCBuiltin<"__builtin_ia32_cmpordss">,
              Intrinsic<[llvm_float_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_cmpordps : GCCBuiltin<"__builtin_ia32_cmpordps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_cmpunordss : GCCBuiltin<"__builtin_ia32_cmpunordss">,
              Intrinsic<[llvm_float_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_cmpunordps : GCCBuiltin<"__builtin_ia32_cmpunordps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v4f32_ty,
                         llvm_v4f32_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_comieqss : GCCBuiltin<"__builtin_ia32_comieq">,
              Intrinsic<[llvm_int_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_comiltss : GCCBuiltin<"__builtin_ia32_comilt">,
              Intrinsic<[llvm_int_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_comiless : GCCBuiltin<"__Builtin_ia32_comile">,
              Intrinsic<[llvm_int_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_comigtss : GCCBuiltin<"__builtin_ia32_comigt">,
              Intrinsic<[llvm_int_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_comigess : GCCBuiltin<"__builtin_ia32_comige">,
              Intrinsic<[llvm_int_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_comineqss : GCCBuiltin<"__builtin_ia32_comineq">,
              Intrinsic<[llvm_int_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_ucomieqss : GCCBuiltin<"__builtin_ia32_ucomieq">,
              Intrinsic<[llvm_int_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_ucomiltss : GCCBuiltin<"__builtin_ia32_ucomilt">,
              Intrinsic<[llvm_int_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_ucomiless : GCCBuiltin<"__Builtin_ia32_ucomile">,
              Intrinsic<[llvm_int_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_ucomigtss : GCCBuiltin<"__builtin_ia32_ucomigt">,
              Intrinsic<[llvm_int_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_ucomigess : GCCBuiltin<"__builtin_ia32_ucomige">,
              Intrinsic<[llvm_int_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_ucomineqss : GCCBuiltin<"__builtin_ia32_ucomineq">,
              Intrinsic<[llvm_int_ty, llvm_float_ty,
                         llvm_float_ty], [InstrNoMem]>;
}


// Conversion ops
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_cvtss2si : GCCBuiltin<"__builtin_ia32_cvtss2si">,
              Intrinsic<[llvm_int_ty, llvm_float_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_cvtps2pi : GCCBuiltin<"__builtin_ia32_cvtps2pi">,
              Intrinsic<[llvm_v2i32_ty, llvm_v4i32_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_cvttss2si : GCCBuiltin<"__builtin_ia32_cvttss2si">,
              Intrinsic<[llvm_int_ty, llvm_float_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_cvttps2pi : GCCBuiltin<"__builtin_ia32_cvttps2pi">,
              Intrinsic<[llvm_v2i32_ty, llvm_v4i32_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_cvtsi2ss : GCCBuiltin<"__builtin_ia32_cvtsi2ss">,
              Intrinsic<[llvm_float_ty, llvm_int_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_cvtpi2ps : GCCBuiltin<"__builtin_ia32_cvtpi2ps">,
              Intrinsic<[llvm_v4f32_ty, llvm_v2i32_ty], [InstrNoMem]>;
}

// SIMD load ops

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_loadhps : GCCBuiltin<"__builtin_ia32_loadhps">,
              Intrinsic<[llvm_v4f32_ty, llvm_ptr_ty], [IntrReadMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_loadlps : GCCBuiltin<"__builtin_ia32_loadlps">,
              Intrinsic<[llvm_v4f32_ty, llvm_ptr_ty], [IntrReadMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_loadups : GCCBuiltin<"__builtin_ia32_loadups">,
              Intrinsic<[llvm_v4f32_ty, llvm_ptr_ty], [IntrReadMem]>;
}

// SIMD store ops

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_storehps : GCCBuiltin<"__builtin_ia32_storehps">,
              Intrinsic<[llvm_ptr_ty, llvm_v4f32_ty], [IntrWriteMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_storelps : GCCBuiltin<"__builtin_ia32_storelps">,
              Intrinsic<[llvm_ptr_ty, llvm_v4f32_ty], [IntrWriteMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_storeups : GCCBuiltin<"__builtin_ia32_storeups">,
              Intrinsic<[llvm_ptr_ty, llvm_v4f32_ty], [IntrWriteMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_movmskps : GCCBuiltin<"__builtin_ia32_movmskps">,
              Intrinsic<[llvm_int_ty, llvm_v4f32_ty], [InstrNoMem]>;
}

let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse_ldmxcsr : GCCBuiltin<"__builtin_ia32_ldmxcsr">,
              Intrinsic<[llvm_void_ty, llvm_ptr_ty], [IntrWriteMem]>;
}

// SSE2
let TargetPrefix = "x86" in {  // All intrinsics start with "llvm.x86.".
  def int_x86_sse2_movmskpd : GCCBuiltin<"__builtin_ia32_movmskpd">,
              Intrinsic<[llvm_int_ty, llvm_v2f64_ty], [InstrNoMem]>;
}
