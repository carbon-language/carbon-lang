//===--- yaml2obj.h - -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// \brief Common declarations for yaml2obj
//===----------------------------------------------------------------------===//
#ifndef LLVM_TOOLS_YAML2OBJ_H
#define LLVM_TOOLS_YAML2OBJ_H

namespace llvm {
  class raw_ostream;
  class MemoryBuffer;
}
int yaml2coff(llvm::raw_ostream &Out, llvm::MemoryBuffer *Buf);
int yaml2elf(llvm::raw_ostream &Out, llvm::MemoryBuffer *Buf);

#endif
