/*===-- llvm-c/BitReader.h - BitReader Library C Interface ------*- C++ -*-===*\
|*                                                                            *|
|*                     The LLVM Compiler Infrastructure                       *|
|*                                                                            *|
|* This file was developed by Gordon Henriksen and is distributed under the   *|
|* University of Illinois Open Source License. See LICENSE.TXT for details.   *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This header declares the C interface to libLLVMBitReader.a, which          *|
|* implements input of the LLVM bitcode format.                               *|
|*                                                                            *|
|* Many exotic languages can interoperate with C code but have a harder time  *|
|* with C++ due to name mangling. So in addition to C, this interface enables *|
|* tools written in such languages.                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef LLVM_C_BITCODEREADER_H
#define LLVM_C_BITCODEREADER_H

#include "llvm-c/Core.h"

#ifdef __cplusplus
extern "C" {
#endif


/* Reads a module from the specified path, returning a reference to the module
   via the OutModule parameter. Returns 0 on success. Optionally returns a
   human-readable error message. */ 
int LLVMReadBitcodeFromFile(const char *Path, LLVMModuleRef *OutModule,
                            char **OutMessage);

/* Disposes of the message allocated by the bitcode reader, if any. */ 
void LLVMDisposeBitcodeReaderMessage(char *Message);


#ifdef __cplusplus
}
#endif

#endif
