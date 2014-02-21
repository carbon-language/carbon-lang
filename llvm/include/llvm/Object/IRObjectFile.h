//===- IRObjectFile.h - LLVM IR object file implementation ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the IRObjectFile template class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_IR_OBJECT_FILE_H
#define LLVM_OBJECT_IR_OBJECT_FILE_H

#include "llvm/Object/SymbolicFile.h"

namespace llvm {
class Module;
class GlobalValue;

namespace object {
class IRObjectFile : public SymbolicFile {
  OwningPtr<Module> M;
public:
  IRObjectFile(MemoryBuffer *Object, error_code &EC, LLVMContext &Context,
               bool BufferOwned);
  void moveSymbolNext(DataRefImpl &Symb) const LLVM_OVERRIDE;
  error_code printSymbolName(raw_ostream &OS, DataRefImpl Symb) const
      LLVM_OVERRIDE;
  uint32_t getSymbolFlags(DataRefImpl Symb) const LLVM_OVERRIDE;
  const GlobalValue &getSymbolGV(DataRefImpl Symb) const;
  basic_symbol_iterator symbol_begin_impl() const LLVM_OVERRIDE;
  basic_symbol_iterator symbol_end_impl() const LLVM_OVERRIDE;

  static inline bool classof(const Binary *v) {
    return v->isIR();
  }
};
}
}

#endif
