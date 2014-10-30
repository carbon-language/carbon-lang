/*===-- bitwriter_ocaml.c - LLVM OCaml Glue ---------------------*- C++ -*-===*\
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

#include "llvm-c/BitWriter.h"
#include "llvm-c/Core.h"
#include "caml/alloc.h"
#include "caml/mlvalues.h"
#include "caml/memory.h"

/* Llvm.llmodule -> string -> bool */
CAMLprim value llvm_write_bitcode_file(LLVMModuleRef M, value Path) {
  int Result = LLVMWriteBitcodeToFile(M, String_val(Path));
  return Val_bool(Result == 0);
}

/* ?unbuffered:bool -> Llvm.llmodule -> Unix.file_descr -> bool */
CAMLprim value llvm_write_bitcode_to_fd(value U, LLVMModuleRef M, value FD) {
  int Unbuffered;
  int Result;

  if (U == Val_int(0)) {
    Unbuffered = 0;
  } else {
    Unbuffered = Bool_val(Field(U, 0));
  }

  Result = LLVMWriteBitcodeToFD(M, Int_val(FD), 0, Unbuffered);
  return Val_bool(Result == 0);
}

/* Llvm.llmodule -> Llvm.llmemorybuffer */
CAMLprim LLVMMemoryBufferRef llvm_write_bitcode_to_memory_buffer(LLVMModuleRef M) {
  return LLVMWriteBitcodeToMemoryBuffer(M);
}
