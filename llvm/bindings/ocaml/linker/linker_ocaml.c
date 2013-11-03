/*===-- linker_ocaml.c - LLVM Ocaml Glue ------------------------*- C++ -*-===*\
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

#include "llvm-c/Linker.h"
#include "caml/alloc.h"
#include "caml/memory.h"
#include "caml/fail.h"

static value llvm_linker_error_exn;

CAMLprim value llvm_register_linker_exns(value Error) {
  llvm_linker_error_exn = Field(Error, 0);
  register_global_root(&llvm_linker_error_exn);
  return Val_unit;
}

static void llvm_raise(value Prototype, char *Message) {
  CAMLparam1(Prototype);
  CAMLlocal1(CamlMessage);

  CamlMessage = copy_string(Message);
  LLVMDisposeMessage(Message);

  raise_with_arg(Prototype, CamlMessage);
  abort(); /* NOTREACHED */
#ifdef CAMLnoreturn
  CAMLnoreturn; /* Silences warnings, but is missing in some versions. */
#endif
}

/* llmodule -> llmodule -> Mode.t -> unit
   raises Error msg on error */
CAMLprim value llvm_link_modules(LLVMModuleRef Dst, LLVMModuleRef Src, value Mode) {
  char* Message;

  if (LLVMLinkModules(Dst, Src, Int_val(Mode), &Message))
    llvm_raise(llvm_linker_error_exn, Message);

  return Val_unit;
}
