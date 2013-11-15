/*===-- backend_ocaml.c - LLVM OCaml Glue -----------------------*- C++ -*-===*\
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
#include "caml/memory.h"

// TODO: Figure out how to call these only for targets which support them.
// LLVMInitialize ## target ## AsmPrinter();
// LLVMInitialize ## target ## AsmParser();
// LLVMInitialize ## target ## Disassembler();

#define INITIALIZER1(target) \
  CAMLprim value llvm_initialize_ ## target(value Unit) {  \
    LLVMInitialize ## target ## TargetInfo();              \
    LLVMInitialize ## target ## Target();                  \
    LLVMInitialize ## target ## TargetMC();                \
    return Val_unit;                                       \
  }

#define INITIALIZER(target) INITIALIZER1(target)

INITIALIZER(TARGET)
