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

// IntrNoMem - The intrinsic does not access memory or have any other side
// effects.  It may be CSE'd deleted if dead, etc.
def IntrNoMem : IntrinsicProperty;

// IntrReadArgMem - This intrinsic reads only from memory that one of its
// arguments points to, but may read an unspecified amount.
def IntrReadArgMem : IntrinsicProperty;

// IntrReadMem - This intrinsic reads from unspecified memory, so it cannot be
// moved across stores.  However, it can be reordered otherwise and can be
// deleted if dead.
def IntrReadMem : IntrinsicProperty;

// IntrWriteArgMem - This intrinsic reads and writes only from memory that one
// of its arguments points to, but may access an unspecified amount.  It has no
// other side effects.  This may only be used if the intrinsic doesn't "capture"
// the argument pointer (e.g. storing it someplace).
def IntrWriteArgMem : IntrinsicProperty;

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

class LLVMIntegerType<ValueType VT, int width>
  : LLVMType<VT, "Type::IntegerTyID"> {
  int Width = width;
}

class LLVMVectorType<ValueType VT, int numelts, LLVMType elty>
  : LLVMType<VT, "Type::VectorTyID">{
  int NumElts = numelts;
  LLVMType ElTy = elty;
} 

class LLVMPointerType<LLVMType elty>
  : LLVMType<iPTR, "Type::PointerTyID">{
  LLVMType ElTy = elty;
} 

class LLVMEmptyStructType
  : LLVMType<OtherVT, "Type::StructTyID">{
} 

def llvm_void_ty       : LLVMType<isVoid, "Type::VoidTyID">;
def llvm_int_ty        : LLVMIntegerType<iAny, 0>;
def llvm_i1_ty         : LLVMIntegerType<i1 , 1>;
def llvm_i8_ty         : LLVMIntegerType<i8 , 8>;
def llvm_i16_ty        : LLVMIntegerType<i16, 16>;
def llvm_i32_ty        : LLVMIntegerType<i32, 32>;
def llvm_i64_ty        : LLVMIntegerType<i64, 64>;
def llvm_float_ty      : LLVMType<f32, "Type::FloatTyID">;
def llvm_double_ty     : LLVMType<f64, "Type::DoubleTyID">;
def llvm_ptr_ty        : LLVMPointerType<llvm_i8_ty>;             // i8*
def llvm_ptrptr_ty     : LLVMPointerType<llvm_ptr_ty>;            // i8**
def llvm_empty_ty      : LLVMEmptyStructType;                     // { }
def llvm_descriptor_ty : LLVMPointerType<llvm_empty_ty>;          // { }*

def llvm_v16i8_ty      : LLVMVectorType<v16i8,16, llvm_i8_ty>;    // 16 x i8
def llvm_v8i16_ty      : LLVMVectorType<v8i16, 8, llvm_i16_ty>;   //  8 x i16
def llvm_v2i64_ty      : LLVMVectorType<v2i64, 2, llvm_i64_ty>;   //  2 x i64
def llvm_v2i32_ty      : LLVMVectorType<v2i32, 2, llvm_i32_ty>;   //  2 x i32
def llvm_v1i64_ty      : LLVMVectorType<v1i64, 1, llvm_i64_ty>;   //  1 x i64
def llvm_v4i32_ty      : LLVMVectorType<v4i32, 4, llvm_i32_ty>;   //  4 x i32
def llvm_v4f32_ty      : LLVMVectorType<v4f32, 4, llvm_float_ty>; //  4 x float
def llvm_v2f64_ty      : LLVMVectorType<v2f64, 2, llvm_double_ty>;//  2 x double

// MMX Vector Types
def llvm_v8i8_ty       : LLVMVectorType<v8i8,  8, llvm_i8_ty>;    //  8 x i8
def llvm_v4i16_ty      : LLVMVectorType<v4i16, 4, llvm_i16_ty>;   //  4 x i16

def llvm_vararg_ty     : LLVMType<isVoid, "...">; // vararg

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

def int_vastart : Intrinsic<[llvm_void_ty, llvm_ptr_ty], [], "llvm.va_start">;
def int_vacopy  : Intrinsic<[llvm_void_ty, llvm_ptr_ty, llvm_ptr_ty], [],
                            "llvm.va_copy">;
def int_vaend   : Intrinsic<[llvm_void_ty, llvm_ptr_ty], [], "llvm.va_end">;

//===------------------- Garbage Collection Intrinsics --------------------===//
//  
def int_gcroot  : Intrinsic<[llvm_void_ty, llvm_ptrptr_ty, llvm_ptr_ty]>;
def int_gcread  : Intrinsic<[llvm_ptr_ty, llvm_ptr_ty, llvm_ptrptr_ty],
                            [IntrReadArgMem]>;
def int_gcwrite : Intrinsic<[llvm_void_ty, llvm_ptr_ty, llvm_ptr_ty,
                             llvm_ptrptr_ty], [IntrWriteArgMem]>;

//===--------------------- Code Generator Intrinsics ----------------------===//
//  
def int_returnaddress : Intrinsic<[llvm_ptr_ty, llvm_i32_ty], [IntrNoMem]>;
def int_frameaddress  : Intrinsic<[llvm_ptr_ty, llvm_i32_ty], [IntrNoMem]>;
def int_stacksave     : Intrinsic<[llvm_ptr_ty], [IntrReadMem]>,
                        GCCBuiltin<"__builtin_stack_save">;
def int_stackrestore  : Intrinsic<[llvm_void_ty, llvm_ptr_ty]>,
                        GCCBuiltin<"__builtin_stack_restore">;
def int_prefetch      : Intrinsic<[llvm_void_ty, llvm_ptr_ty, 
                                   llvm_i32_ty, llvm_i32_ty]>;
def int_pcmarker      : Intrinsic<[llvm_void_ty, llvm_i32_ty]>;

def int_readcyclecounter : Intrinsic<[llvm_i64_ty]>;

//===------------------- Standard C Library Intrinsics --------------------===//
//

let Properties = [IntrWriteArgMem] in {
  def int_memcpy_i32  : Intrinsic<[llvm_void_ty, llvm_ptr_ty, llvm_ptr_ty,
                                   llvm_i32_ty, llvm_i32_ty]>;
  def int_memcpy_i64  : Intrinsic<[llvm_void_ty, llvm_ptr_ty, llvm_ptr_ty,
                                   llvm_i64_ty, llvm_i32_ty]>;
  def int_memmove_i32 : Intrinsic<[llvm_void_ty, llvm_ptr_ty, llvm_ptr_ty,
                                   llvm_i32_ty, llvm_i32_ty]>;
  def int_memmove_i64 : Intrinsic<[llvm_void_ty, llvm_ptr_ty, llvm_ptr_ty,
                                   llvm_i64_ty, llvm_i32_ty]>;
  def int_memset_i32  : Intrinsic<[llvm_void_ty, llvm_ptr_ty, llvm_i8_ty,
                                   llvm_i32_ty, llvm_i32_ty]>;
  def int_memset_i64  : Intrinsic<[llvm_void_ty, llvm_ptr_ty, llvm_i8_ty,
                                   llvm_i64_ty, llvm_i32_ty]>;
}

let Properties = [IntrNoMem] in {
  def int_sqrt_f32 : Intrinsic<[llvm_float_ty , llvm_float_ty]>;
  def int_sqrt_f64 : Intrinsic<[llvm_double_ty, llvm_double_ty]>;

  def int_powi_f32 : Intrinsic<[llvm_float_ty , llvm_float_ty, llvm_i32_ty]>;
  def int_powi_f64 : Intrinsic<[llvm_double_ty, llvm_double_ty, llvm_i32_ty]>;
}

// NOTE: these are internal interfaces.
def int_setjmp     : Intrinsic<[llvm_i32_ty , llvm_ptr_ty]>;
def int_longjmp    : Intrinsic<[llvm_void_ty, llvm_ptr_ty, llvm_i32_ty]>;
def int_sigsetjmp  : Intrinsic<[llvm_i32_ty , llvm_ptr_ty, llvm_i32_ty]>;
def int_siglongjmp : Intrinsic<[llvm_void_ty, llvm_ptr_ty, llvm_i32_ty]>;

//===-------------------- Bit Manipulation Intrinsics ---------------------===//
//

// None of these intrinsics accesses memory at all.
let Properties = [IntrNoMem] in {
  def int_bswap: Intrinsic<[llvm_int_ty, llvm_int_ty]>;
  def int_ctpop: Intrinsic<[llvm_i32_ty, llvm_int_ty]>;
  def int_ctlz : Intrinsic<[llvm_i32_ty, llvm_int_ty]>;
  def int_cttz : Intrinsic<[llvm_i32_ty, llvm_int_ty]>;
  def int_part_select : 
     Intrinsic<[llvm_int_ty, llvm_int_ty, llvm_i32_ty, llvm_i32_ty]>;
  def int_part_set :
     Intrinsic<[llvm_int_ty, llvm_int_ty, llvm_int_ty, llvm_i32_ty, 
                llvm_i32_ty]>;
}

//===------------------------ Debugger Intrinsics -------------------------===//
//

def int_dbg_stoppoint    : Intrinsic<[llvm_void_ty,
                                      llvm_i32_ty, llvm_i32_ty, 
                                      llvm_descriptor_ty]>;
def int_dbg_region_start : Intrinsic<[llvm_void_ty, llvm_descriptor_ty]>;
def int_dbg_region_end   : Intrinsic<[llvm_void_ty, llvm_descriptor_ty]>;
def int_dbg_func_start   : Intrinsic<[llvm_void_ty, llvm_descriptor_ty]>;
def int_dbg_declare      : Intrinsic<[llvm_void_ty, llvm_descriptor_ty,
                                                    llvm_descriptor_ty]>;

//===------------------ Exception Handling Intrinsics----------------------===//
//
def int_eh_exception  : Intrinsic<[llvm_ptr_ty]>;
def int_eh_selector   : Intrinsic<[llvm_i32_ty, llvm_ptr_ty, llvm_ptr_ty,
                                                llvm_vararg_ty]>;
def int_eh_filter     : Intrinsic<[llvm_i32_ty, llvm_ptr_ty, llvm_ptr_ty,
                                                llvm_vararg_ty]>;
def int_eh_typeid_for : Intrinsic<[llvm_i32_ty, llvm_ptr_ty]>;

//===----------------------------------------------------------------------===//
// Target-specific intrinsics
//===----------------------------------------------------------------------===//

include "llvm/IntrinsicsPowerPC.td"
include "llvm/IntrinsicsX86.td"
