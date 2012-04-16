//===-- RuntimeDyldELF.h - Run-time dynamic linker for MC-JIT ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// ELF support for MC-JIT runtime dynamic linker.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_RUNTIME_DYLD_ELF_H
#define LLVM_RUNTIME_DYLD_ELF_H

#include "RuntimeDyldImpl.h"

using namespace llvm;


namespace llvm {
class RuntimeDyldELF : public RuntimeDyldImpl {
protected:
  ObjectImage *LoadedObject;

  void resolveX86_64Relocation(uint8_t *LocalAddress,
                               uint64_t FinalAddress,
                               uint64_t Value,
                               uint32_t Type,
                               int64_t Addend);

  void resolveX86Relocation(uint8_t *LocalAddress,
                            uint32_t FinalAddress,
                            uint32_t Value,
                            uint32_t Type,
                            int32_t Addend);

  void resolveARMRelocation(uint8_t *LocalAddress,
                            uint32_t FinalAddress,
                            uint32_t Value,
                            uint32_t Type,
                            int32_t Addend);

  virtual void resolveRelocation(uint8_t *LocalAddress,
                                 uint64_t FinalAddress,
                                 uint64_t Value,
                                 uint32_t Type,
                                 int64_t Addend);

  virtual void processRelocationRef(const ObjRelocationInfo &Rel,
                                    ObjectImage &Obj,
                                    ObjSectionToIDMap &ObjSectionToID,
                                    LocalSymbolMap &Symbols, StubMap &Stubs);

  virtual ObjectImage *createObjectImage(const MemoryBuffer *InputBuffer);
  virtual void handleObjectLoaded(ObjectImage *Obj);

public:
  RuntimeDyldELF(RTDyldMemoryManager *mm)
      : RuntimeDyldImpl(mm), LoadedObject(0) {}

  virtual ~RuntimeDyldELF();

  bool isCompatibleFormat(const MemoryBuffer *InputBuffer) const;
};

} // end namespace llvm

#endif
