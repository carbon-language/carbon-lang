//===- llvm/Bytecode/WriteBytecodePass.h - Bytecode Writer Pass -*- C++ -*-===//
//
// This file defines a simple pass to write the working module to a file after
// pass processing is completed.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_BYTECODE_WRITEBYTECODEPASS_H
#define LLVM_BYTECODE_WRITEBYTECODEPASS_H

#include "llvm/Pass.h"
#include "llvm/Bytecode/Writer.h"
#include <iostream>

class WriteBytecodePass : public Pass {
  std::ostream *Out;           // ostream to print on
  bool DeleteStream;
public:
  WriteBytecodePass() : Out(&std::cout), DeleteStream(false) {}
  WriteBytecodePass(std::ostream *o, bool DS = false) 
    : Out(o), DeleteStream(DS) {
  }

  inline ~WriteBytecodePass() {
    if (DeleteStream) delete Out;
  }
  
  bool run(Module &M) {
    WriteBytecodeToFile(&M, *Out);    
    return false;
  }
};

#endif
