//===-- llvm/Assembly/CachedWriter.h - Printer Accellerator -----*- C++ -*-===//
//
// This file defines a 'CacheWriter' class that is used to accelerate printing
// chunks of LLVM.  This is used when a module is not being changed, but random
// parts of it need to be printed.  This can greatly speed up printing of LLVM
// output.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ASSEMBLY_CACHED_WRITER_H
#define LLVM_ASSEMBLY_CACHED_WRITER_H

#include "llvm/Value.h"
#include <iostream>

class Module;
class PointerType;
class SlotCalculator;
class AssemblyWriter;  // Internal private class

class CachedWriter {
  AssemblyWriter *AW;
  SlotCalculator *SC;
public:
  std::ostream &Out;
public:
  CachedWriter(std::ostream &O = std::cout) : AW(0), SC(0), Out(O) { }
  CachedWriter(const Module *M, std::ostream &O = std::cout)
    : AW(0), SC(0), Out(O) {
    setModule(M);
  }
  ~CachedWriter();

  // setModule - Invalidate internal state, use the new module instead.
  void setModule(const Module *M);

  CachedWriter &operator<<(const Value *V);

  inline CachedWriter &operator<<(Value *X) {
    return *this << (const Value*)X;
  }
  inline CachedWriter &operator<<(const GlobalVariable *X) {
    return *this << (const Value*)X;
  }
  inline CachedWriter &operator<<(const Function *X) {
    return *this << (const Value*)X;
  }
  inline CachedWriter &operator<<(const Argument *X) {
    return *this << (const Value*)X;
  }
  inline CachedWriter &operator<<(const BasicBlock *X) {
    return *this << (const Value*)X;
  }
  inline CachedWriter &operator<<(const Instruction *X) {
    return *this << (const Value*)X; 
  }
  inline CachedWriter &operator<<(const Constant *X) {
    return *this << (const Value*)X; 
  }
  inline CachedWriter &operator<<(const Type *X) {
    return *this << (const Value*)X;
  }
  inline CachedWriter &operator<<(const PointerType *X) {
    return *this << (const Value*)X; 
  }

  inline CachedWriter &operator<<(std::ostream &(&Manip)(std::ostream &)) {
    Out << Manip; return *this;
  }

  template<class X>
  inline CachedWriter &operator<<(const X &v) {
    Out << v;
    return *this;
  }
};

#endif
