//===-- llvm/Assembly/CachedWriter.h - Printer Accellerator ------*- C++ -*--=//
//
// This file defines a 'CacheWriter' class that is used to accelerate printing
// chunks of LLVM.  This is used when a module is not being changed, but random
// parts of it need to be printed.  This can greatly speed up printing of LLVM
// output.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ASSEMBLY_CACHED_WRITER_H
#define LLVM_ASSEMBLY_CACHED_WRITER_H

#include "llvm/Assembly/Writer.h"

class AssemblyWriter;  // Internal private class
class SlotCalculator;

class CachedWriter {
  AssemblyWriter *AW;
  SlotCalculator *SC;
  ostream &Out;
public:
  CachedWriter(ostream &O = cout) : AW(0), SC(0), Out(O) { }
  CachedWriter(const Module *M, ostream &O = cout) : AW(0), SC(0), Out(O) {
    setModule(M);
  }
  ~CachedWriter();

  // setModule - Invalidate internal state, use the new module instead.
  void setModule(const Module *M);

  CachedWriter &operator<<(const Value *V);

  template<class X>
  inline CachedWriter &operator<<(X &v) {
    Out << v;
    return *this;
  }
};

#endif
