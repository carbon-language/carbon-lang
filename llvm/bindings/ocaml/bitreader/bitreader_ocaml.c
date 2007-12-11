/*===-- bitwriter_ocaml.c - LLVM Ocaml Glue ---------------------*- C++ -*-===*\
|*                                                                            *|
|*                     The LLVM Compiler Infrastructure                       *|
|*                                                                            *|
|* This file was developed by Gordon Henriksen and is distributed under the   *|
|* University of Illinois Open Source License. See LICENSE.TXT for details.   *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This file glues LLVM's ocaml interface to its C interface. These functions *|
|* are by and large transparent wrappers to the corresponding C functions.    *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#include "llvm-c/BitReader.h"
#include "caml/alloc.h"
#include "caml/mlvalues.h"
#include "caml/memory.h"

/*===-- Modules -----------------------------------------------------------===*/

/* string -> bitreader_result

   type bitreader_result =
   | Bitreader_success of Llvm.llmodule
   | Bitreader_failure of string
 */
CAMLprim value llvm_read_bitcode_file(value Path) {
  LLVMModuleRef M;
  char *Message;
  CAMLparam1(Path);
  CAMLlocal2(Variant, MessageVal);
  
  if (LLVMReadBitcodeFromFile(String_val(Path), &M, &Message)) {
    MessageVal = copy_string(Message);
    LLVMDisposeBitcodeReaderMessage(Message);
    
    Variant = alloc(1, 1);
    Field(Variant, 0) = MessageVal;
  } else {
    Variant = alloc(1, 0);
    Field(Variant, 0) = Val_op(M);
  }
  
  CAMLreturn(Variant);
}
