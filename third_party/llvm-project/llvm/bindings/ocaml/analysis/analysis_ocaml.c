/*===-- analysis_ocaml.c - LLVM OCaml Glue ----------------------*- C++ -*-===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
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

#include "llvm-c/Analysis.h"
#include "llvm-c/Core.h"
#include "caml/alloc.h"
#include "caml/mlvalues.h"
#include "caml/memory.h"
#include "llvm_ocaml.h"

/* Llvm.llmodule -> string option */
value llvm_verify_module(LLVMModuleRef M) {
  CAMLparam0();
  CAMLlocal2(String, Option);

  char *Message;
  int Result = LLVMVerifyModule(M, LLVMReturnStatusAction, &Message);

  if (0 == Result) {
    Option = Val_none;
  } else {
    String = copy_string(Message);
    Option = caml_alloc_some(String);
  }

  LLVMDisposeMessage(Message);

  CAMLreturn(Option);
}

/* Llvm.llvalue -> bool */
value llvm_verify_function(LLVMValueRef Fn) {
  return Val_bool(LLVMVerifyFunction(Fn, LLVMReturnStatusAction) == 0);
}

/* Llvm.llmodule -> unit */
value llvm_assert_valid_module(LLVMModuleRef M) {
  LLVMVerifyModule(M, LLVMAbortProcessAction, 0);
  return Val_unit;
}

/* Llvm.llvalue -> unit */
value llvm_assert_valid_function(LLVMValueRef Fn) {
  LLVMVerifyFunction(Fn, LLVMAbortProcessAction);
  return Val_unit;
}

/* Llvm.llvalue -> unit */
value llvm_view_function_cfg(LLVMValueRef Fn) {
  LLVMViewFunctionCFG(Fn);
  return Val_unit;
}

/* Llvm.llvalue -> unit */
value llvm_view_function_cfg_only(LLVMValueRef Fn) {
  LLVMViewFunctionCFGOnly(Fn);
  return Val_unit;
}
