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

#define DataLayout_val(v)  (*(LLVMTargetDataRef *)(Data_custom_val(v)))

static void llvm_finalize_data_layout(value DataLayout) {
  LLVMDisposeTargetData(DataLayout_val(DataLayout));
}

static struct custom_operations llvm_data_layout_ops = {
  (char *) "LLVMDataLayout",
  llvm_finalize_data_layout,
  custom_compare_default,
  custom_hash_default,
  custom_serialize_default,
  custom_deserialize_default
#ifdef custom_compare_ext_default
  , custom_compare_ext_default
#endif
};

value llvm_alloc_data_layout(LLVMTargetDataRef DataLayout) {
  value V = alloc_custom(&llvm_data_layout_ops, sizeof(LLVMTargetDataRef), 0, 1);
  DataLayout_val(V) = DataLayout;
  return V;
}

/* string -> DataLayout.t */
CAMLprim value llvm_datalayout_of_string(value StringRep) {
  return llvm_alloc_data_layout(LLVMCreateTargetData(String_val(StringRep)));
}

/* DataLayout.t -> string */
CAMLprim value llvm_datalayout_as_string(value TD) {
  char *StringRep = LLVMCopyStringRepOfTargetData(DataLayout_val(TD));
  value Copy = copy_string(StringRep);
  LLVMDisposeMessage(StringRep);
  return Copy;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> DataLayout.t -> unit */
CAMLprim value llvm_datalayout_add_to_pass_manager(LLVMPassManagerRef PM,
                                                   value DL) {
  LLVMAddTargetData(DataLayout_val(DL), PM);
  return Val_unit;
}

/* DataLayout.t -> Endian.t */
CAMLprim value llvm_datalayout_byte_order(value DL) {
  return Val_int(LLVMByteOrder(DataLayout_val(DL)));
}

/* DataLayout.t -> int */
CAMLprim value llvm_datalayout_pointer_size(value DL) {
  return Val_int(LLVMPointerSize(DataLayout_val(DL)));
}

/* Llvm.llcontext -> DataLayout.t -> Llvm.lltype */
CAMLprim LLVMTypeRef llvm_datalayout_intptr_type(LLVMContextRef C, value DL) {
  return LLVMIntPtrTypeInContext(C, DataLayout_val(DL));;
}

/* int -> DataLayout.t -> int */
CAMLprim value llvm_datalayout_qualified_pointer_size(value AS, value DL) {
  return Val_int(LLVMPointerSizeForAS(DataLayout_val(DL), Int_val(AS)));
}

/* Llvm.llcontext -> int -> DataLayout.t -> Llvm.lltype */
CAMLprim LLVMTypeRef llvm_datalayout_qualified_intptr_type(LLVMContextRef C,
                                                           value AS,
                                                           value DL) {
  return LLVMIntPtrTypeForASInContext(C, DataLayout_val(DL), Int_val(AS));
}

/* Llvm.lltype -> DataLayout.t -> Int64.t */
CAMLprim value llvm_datalayout_size_in_bits(LLVMTypeRef Ty, value DL) {
  return caml_copy_int64(LLVMSizeOfTypeInBits(DataLayout_val(DL), Ty));
}

/* Llvm.lltype -> DataLayout.t -> Int64.t */
CAMLprim value llvm_datalayout_store_size(LLVMTypeRef Ty, value DL) {
  return caml_copy_int64(LLVMStoreSizeOfType(DataLayout_val(DL), Ty));
}

/* Llvm.lltype -> DataLayout.t -> Int64.t */
CAMLprim value llvm_datalayout_abi_size(LLVMTypeRef Ty, value DL) {
  return caml_copy_int64(LLVMABISizeOfType(DataLayout_val(DL), Ty));
}

/* Llvm.lltype -> DataLayout.t -> int */
CAMLprim value llvm_datalayout_abi_align(LLVMTypeRef Ty, value DL) {
  return Val_int(LLVMABIAlignmentOfType(DataLayout_val(DL), Ty));
}

/* Llvm.lltype -> DataLayout.t -> int */
CAMLprim value llvm_datalayout_stack_align(LLVMTypeRef Ty, value DL) {
  return Val_int(LLVMCallFrameAlignmentOfType(DataLayout_val(DL), Ty));
}

/* Llvm.lltype -> DataLayout.t -> int */
CAMLprim value llvm_datalayout_preferred_align(LLVMTypeRef Ty, value DL) {
  return Val_int(LLVMPreferredAlignmentOfType(DataLayout_val(DL), Ty));
}

/* Llvm.llvalue -> DataLayout.t -> int */
CAMLprim value llvm_datalayout_preferred_align_of_global(LLVMValueRef GlobalVar,
                                                         value DL) {
  return Val_int(LLVMPreferredAlignmentOfGlobal(DataLayout_val(DL), GlobalVar));
}

/* Llvm.lltype -> Int64.t -> DataLayout.t -> int */
CAMLprim value llvm_datalayout_element_at_offset(LLVMTypeRef Ty, value Offset,
                                                 value DL) {
  return Val_int(LLVMElementAtOffset(DataLayout_val(DL), Ty,
                                     Int64_val(Offset)));
}

/* Llvm.lltype -> int -> DataLayout.t -> Int64.t */
CAMLprim value llvm_datalayout_offset_of_element(LLVMTypeRef Ty, value Index,
                                                 value DL) {
  return caml_copy_int64(LLVMOffsetOfElement(DataLayout_val(DL), Ty,
                                             Int_val(Index)));
}
