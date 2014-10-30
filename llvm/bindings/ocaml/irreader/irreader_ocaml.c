/*===-- irreader_ocaml.c - LLVM OCaml Glue ----------------------*- C++ -*-===*\
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
\*===----------------------------------------------------------------------===*/

#include "llvm-c/IRReader.h"
#include "caml/alloc.h"
#include "caml/fail.h"
#include "caml/memory.h"
#include "caml/callback.h"

void llvm_raise(value Prototype, char *Message);

/* Llvm.llcontext -> Llvm.llmemorybuffer -> Llvm.llmodule */
CAMLprim value llvm_parse_ir(LLVMContextRef C,
                             LLVMMemoryBufferRef MemBuf) {
  CAMLparam0();
  CAMLlocal2(Variant, MessageVal);
  LLVMModuleRef M;
  char *Message;

  if (LLVMParseIRInContext(C, MemBuf, &M, &Message))
    llvm_raise(*caml_named_value("Llvm_irreader.Error"), Message);

  CAMLreturn((value) M);
}
