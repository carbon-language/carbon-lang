/*===-- llvm-c/Support.h - Support C Interface --------------------*- C -*-===*\
|*                                                                            *|
|*                     The LLVM Compiler Infrastructure                       *|
|*                                                                            *|
|* This file is distributed under the University of Illinois Open Source      *|
|* License. See LICENSE.TXT for details.                                      *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This file defines the C interface to the LLVM support library.             *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef LLVM_C_SUPPORT_H
#define LLVM_C_SUPPORT_H

#include "llvm-c/Core.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * This function permanently loads the dynamic library at the given path.
 * It is safe to call this function multiple times for the same library.
 *
 * @see sys::DynamicLibrary::LoadLibraryPermanently()
  */
LLVMBool LLVMLoadLibraryPermanently(const char* Filename);

#ifdef __cplusplus
}
#endif

#endif
