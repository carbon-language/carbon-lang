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

class LLVMType<string typeval> {
  string TypeVal = typeval;
}

def llvm_void_ty       : LLVMType<"Type::VoidTyID">;
def llvm_bool_ty       : LLVMType<"Type::BoolTyID">;
def llvm_sbyte_ty      : LLVMType<"Type::SByteTyID">;
def llvm_short_ty      : LLVMType<"Type::ShortTyID">;
def llvm_int_ty        : LLVMType<"Type::IntTyID">;
def llvm_long_ty       : LLVMType<"Type::LongTyID">;
def llvm_ubyte_ty      : LLVMType<"Type::UByteTyID">;
def llvm_ushort_ty     : LLVMType<"Type::UShortTyID">;
def llvm_uint_ty       : LLVMType<"Type::UIntTyID">;
def llvm_ulong_ty      : LLVMType<"Type::ULongTyID">;
def llvm_float_ty      : LLVMType<"Type::FloatTyID">;
def llvm_double_ty     : LLVMType<"Type::DoubleTyID">;
def llvm_ptr_ty        : LLVMType<"Type::PointerTyID">;     // sbyte*
def llvm_ptrptr_ty     : LLVMType<"Type::PointerTyID">;     // sbyte**
def llvm_anchor_ty     : LLVMType<"Type::PointerTyID">;     // {}*
def llvm_descriptor_ty : LLVMType<"Type::PointerTyID">;     // global*

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
  list<LLVMType> Types = types;
  list<IntrinsicProperty> Properties = properties;
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
def int_gcread  : Intrinsic<[llvm_ptr_ty, llvm_ptrptr_ty], [InstrReadArgMem]>;
def int_gcwrite : Intrinsic<[llvm_void_ty, llvm_ptr_ty, llvm_ptrptr_ty],
                            [InstrWriteArgMem]>;

//===--------------------- Code Generator Intrinsics ----------------------===//
//  
def int_returnaddress : Intrinsic<[llvm_ptr_ty, llvm_uint_ty], [InstrNoMem]>;
def int_frameaddress  : Intrinsic<[llvm_ptr_ty, llvm_uint_ty], [InstrNoMem]>;
def int_stacksave     : Intrinsic<[llvm_ptr_ty]>;
def int_stackrestore  : Intrinsic<[llvm_ptr_ty]>;
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
  def int_isunordered_f32 : Intrinsic<[llvm_float_ty,  llvm_float_ty]>;
  def int_isunordered_f64 : Intrinsic<[llvm_double_ty, llvm_double_ty]>;
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

def int_dbg_stoppoint    : Intrinsic<[llvm_anchor_ty, llvm_uint_ty, 
                                      llvm_uint_ty, llvm_descriptor_ty]>;
def int_dbg_region_start : Intrinsic<[llvm_anchor_ty, llvm_anchor_ty]>;
def int_dbg_region_end   : Intrinsic<[llvm_anchor_ty, llvm_anchor_ty]>;
def int_dbg_func_start   : Intrinsic<[llvm_anchor_ty, llvm_descriptor_ty]>;
//    dbg_declare,      // Declare a local object

