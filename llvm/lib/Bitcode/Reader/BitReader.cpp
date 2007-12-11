//===-- BitReader.cpp -----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Gordon Henriksen and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm-c/BitReader.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Support/MemoryBuffer.h"
#include <string>

using namespace llvm;


int LLVMReadBitcodeFromFile(const char *Path, LLVMModuleRef *OutModule,
                            char **OutMessage) {
  std::string Message;
  
  MemoryBuffer *buf = MemoryBuffer::getFile(Path, strlen(Path), &Message);
  if (!buf) {
    if (!OutMessage)
      *OutMessage = strdup(Message.c_str());
    return 1;
  }
  
  *OutModule = wrap(ParseBitcodeFile(buf, &Message));
  if (!*OutModule) {
    if (OutMessage)
      *OutMessage = strdup(Message.c_str());
    return 1;
  }
  
  return 0;
}

void LLVMDisposeBitcodeReaderMessage(char *Message) {
  if (Message)
    free(Message);
}
