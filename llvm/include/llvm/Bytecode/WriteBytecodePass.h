//===- llvm/Bytecode/WriteBytecodePass.h - Bytecode Writer Pass -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a simple pass to write the working module to a file after
// pass processing is completed.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_BYTECODE_WRITEBYTECODEPASS_H
#define LLVM_BYTECODE_WRITEBYTECODEPASS_H

#include "llvm/Pass.h"
#include "llvm/Bytecode/Writer.h"
#include "llvm/Support/Streams.h"

namespace llvm {

class WriteBytecodePass : public ModulePass {
  OStream *Out;                 // ostream to print on
  bool DeleteStream;
  bool CompressFile;
public:
  static const char ID; // Pass identifcation, replacement for typeid
  WriteBytecodePass()
    : ModulePass((intptr_t) &ID), Out(&cout), DeleteStream(false), 
      CompressFile(false) {}
  WriteBytecodePass(OStream *o, bool DS = false, bool CF = false)
    : ModulePass((intptr_t) &ID), Out(o), DeleteStream(DS), CompressFile(CF) {}

  inline ~WriteBytecodePass() {
    if (DeleteStream) delete Out;
  }

  bool runOnModule(Module &M) {
    WriteBytecodeToFile(&M, *Out, CompressFile);
    return false;
  }
};

} // End llvm namespace

#endif
