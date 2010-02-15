//===-- BitReader.cpp -----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm-c/BitReader.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/LLVMContext.h"
#include "llvm/Support/MemoryBuffer.h"
#include <string>
#include <cstring>

using namespace llvm;

/* Builds a module from the bitcode in the specified memory buffer, returning a
   reference to the module via the OutModule parameter. Returns 0 on success.
   Optionally returns a human-readable error message via OutMessage. */
LLVMBool LLVMParseBitcode(LLVMMemoryBufferRef MemBuf,
                          LLVMModuleRef *OutModule, char **OutMessage) {
  return LLVMParseBitcodeInContext(wrap(&getGlobalContext()), MemBuf, OutModule,
                                   OutMessage);
}

LLVMBool LLVMParseBitcodeInContext(LLVMContextRef ContextRef,
                                   LLVMMemoryBufferRef MemBuf,
                                   LLVMModuleRef *OutModule,
                                   char **OutMessage) {
  std::string Message;
  
  *OutModule = wrap(ParseBitcodeFile(unwrap(MemBuf), *unwrap(ContextRef),
                                     &Message));
  if (!*OutModule) {
    if (OutMessage)
      *OutMessage = strdup(Message.c_str());
    return 1;
  }
  
  return 0;
}

/* Reads a module from the specified path, returning via the OutModule parameter
   a module provider which performs lazy deserialization. Returns 0 on success.
   Optionally returns a human-readable error message via OutMessage. */ 
LLVMBool LLVMGetBitcodeModuleProvider(LLVMMemoryBufferRef MemBuf,
                                      LLVMModuleProviderRef *OutMP,
                                      char **OutMessage) {
  return LLVMGetBitcodeModuleProviderInContext(wrap(&getGlobalContext()),
                                               MemBuf, OutMP, OutMessage);
}

LLVMBool LLVMGetBitcodeModuleProviderInContext(LLVMContextRef ContextRef,
                                               LLVMMemoryBufferRef MemBuf,
                                               LLVMModuleProviderRef *OutMP,
                                               char **OutMessage) {
  std::string Message;
  
  *OutMP = reinterpret_cast<LLVMModuleProviderRef>(
    getLazyBitcodeModule(unwrap(MemBuf), *unwrap(ContextRef), &Message));
  if (!*OutMP) {
    if (OutMessage)
      *OutMessage = strdup(Message.c_str());
    return 1;
  }
  
  return 0;
}
