/*===-- target_ocaml.c - LLVM OCaml Glue ------------------------*- C++ -*-===*\
|*                                                                            *|
|*                     The LLVM Compiler Infrastructure                       *|
|*                                                                            *|
|* This file is distributed under the University of Illinois Open Source      *|
|* License. See LICENSE.TXT for details.                                      *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This file glues LLVM's OCaml interface to its C interface. These functions *|
|* are by and large transparent wrappers to the corresponding C functions.    *|
|*                                                                            *|
|* Note that these functions intentionally take liberties with the CAMLparamX *|
|* macros, since most of the parameters are not GC heap objects.              *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#include "llvm-c/Target.h"
#include "caml/alloc.h"
#include "caml/custom.h"

#define TargetData_val(v)  (*(LLVMTargetDataRef *)(Data_custom_val(v)))

static void llvm_finalize_target_data(value TargetData) {
  LLVMDisposeTargetData(TargetData_val(TargetData));
}

static struct custom_operations llvm_target_data_ops = {
  (char *) "LLVMTargetData",
  llvm_finalize_target_data,
  custom_compare_default,
  custom_hash_default,
  custom_serialize_default,
  custom_deserialize_default
#ifdef custom_compare_ext_default
  , custom_compare_ext_default
#endif
};

value llvm_alloc_target_data(LLVMTargetDataRef TargetData) {
  value V = alloc_custom(&llvm_target_data_ops, sizeof(LLVMTargetDataRef), 0, 1);
  TargetData_val(V) = TargetData;
  return V;
}

/* string -> DataLayout.t */
CAMLprim value llvm_targetdata_create(value StringRep) {
  return llvm_alloc_target_data(LLVMCreateTargetData(String_val(StringRep)));
}

/* DataLayout.t -> [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
CAMLprim value llvm_targetdata_add(value TD, LLVMPassManagerRef PM){
  LLVMAddTargetData(TargetData_val(TD), PM);
  return Val_unit;
}

/* DataLayout.t -> string */
CAMLprim value llvm_targetdata_as_string(value TD) {
  char *StringRep = LLVMCopyStringRepOfTargetData(TargetData_val(TD));
  value Copy = copy_string(StringRep);
  LLVMDisposeMessage(StringRep);
  return Copy;
}

/* DataLayout.t -> Endian.t */
CAMLprim value llvm_byte_order(value TD) {
  return Val_int(LLVMByteOrder(TargetData_val(TD)));
}

/* DataLayout.t -> int */
CAMLprim value llvm_pointer_size(value TD) {
  return Val_int(LLVMPointerSize(TargetData_val(TD)));
}

/* DataLayout.t -> Llvm.llcontext -> Llvm.lltype */
CAMLprim LLVMTypeRef llvm_intptr_type(value TD, LLVMContextRef C) {
  return LLVMIntPtrTypeInContext(C, TargetData_val(TD));;
}

/* DataLayout.t -> Llvm.lltype -> Int64.t */
CAMLprim value llvm_size_in_bits(value TD, LLVMTypeRef Ty) {
  return caml_copy_int64(LLVMSizeOfTypeInBits(TargetData_val(TD), Ty));
}

/* DataLayout.t -> Llvm.lltype -> Int64.t */
CAMLprim value llvm_store_size(value TD, LLVMTypeRef Ty) {
  return caml_copy_int64(LLVMStoreSizeOfType(TargetData_val(TD), Ty));
}

/* DataLayout.t -> Llvm.lltype -> Int64.t */
CAMLprim value llvm_abi_size(value TD, LLVMTypeRef Ty) {
  return caml_copy_int64(LLVMABISizeOfType(TargetData_val(TD), Ty));
}

/* DataLayout.t -> Llvm.lltype -> int */
CAMLprim value llvm_abi_align(value TD, LLVMTypeRef Ty) {
  return Val_int(LLVMABIAlignmentOfType(TargetData_val(TD), Ty));
}

/* DataLayout.t -> Llvm.lltype -> int */
CAMLprim value llvm_stack_align(value TD, LLVMTypeRef Ty) {
  return Val_int(LLVMCallFrameAlignmentOfType(TargetData_val(TD), Ty));
}

/* DataLayout.t -> Llvm.lltype -> int */
CAMLprim value llvm_preferred_align(value TD, LLVMTypeRef Ty) {
  return Val_int(LLVMPreferredAlignmentOfType(TargetData_val(TD), Ty));
}

/* DataLayout.t -> Llvm.llvalue -> int */
CAMLprim value llvm_preferred_align_of_global(value TD,
                                              LLVMValueRef GlobalVar) {
  return Val_int(LLVMPreferredAlignmentOfGlobal(TargetData_val(TD), GlobalVar));
}

/* DataLayout.t -> Llvm.lltype -> Int64.t -> int */
CAMLprim value llvm_element_at_offset(value TD, LLVMTypeRef Ty,
                                      value Offset) {
  return Val_int(LLVMElementAtOffset(TargetData_val(TD), Ty, Int64_val(Offset)));
}

/* DataLayout.t -> Llvm.lltype -> int -> Int64.t */
CAMLprim value llvm_offset_of_element(value TD, LLVMTypeRef Ty,
                                      value Index) {
  return caml_copy_int64(LLVMOffsetOfElement(TargetData_val(TD), Ty, Int_val(Index)));
}
