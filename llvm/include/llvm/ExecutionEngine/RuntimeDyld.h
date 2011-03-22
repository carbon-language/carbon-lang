//===-- RuntimeDyld.h - Run-time dynamic linker for MC-JIT ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Interface for the runtime dynamic linker facilities of the MC-JIT.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_RUNTIME_DYLD_H
#define LLVM_RUNTIME_DYLD_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Memory.h"

namespace llvm {

class RuntimeDyldImpl;
class MemoryBuffer;

class RuntimeDyld {
  RuntimeDyld(const RuntimeDyld &);     // DO NOT IMPLEMENT
  void operator=(const RuntimeDyld &);  // DO NOT IMPLEMENT

  // RuntimeDyldImpl is the actual class. RuntimeDyld is just the public
  // interface.
  RuntimeDyldImpl *Dyld;
public:
  RuntimeDyld();
  ~RuntimeDyld();

  bool loadObject(MemoryBuffer *InputBuffer);
  void *getSymbolAddress(StringRef Name);
  // FIXME: Should be parameterized to get the memory block associated with
  // a particular loaded object.
  sys::MemoryBlock getMemoryBlock();
  StringRef getErrorString();
};

} // end namespace llvm

#endif
