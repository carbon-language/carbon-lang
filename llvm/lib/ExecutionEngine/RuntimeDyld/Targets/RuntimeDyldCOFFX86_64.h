//===-- RuntimeDyldCOFFX86_64.h --- COFF/X86_64 specific code ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// COFF x86_x64 support for MC-JIT runtime dynamic linker.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_EXECUTIONENGINE_RUNTIMEDYLD_TARGETS_RUNTIMEDYLDCOFF86_64_H
#define LLVM_LIB_EXECUTIONENGINE_RUNTIMEDYLD_TARGETS_RUNTIMEDYLDCOFF86_64_H

#include "llvm/Object/COFF.h"
#include "llvm/Support/COFF.h"
#include "../RuntimeDyldCOFF.h"

using namespace llvm;

namespace llvm {

class RuntimeDyldCOFFX86_64 : public RuntimeDyldCOFF {

private:
  // When a module is loaded we save the SectionID of the unwind
  // sections in a table until we receive a request to register all
  // unregisteredEH frame sections with the memory manager.
  SmallVector<SID, 2> UnregisteredEHFrameSections;
  SmallVector<SID, 2> RegisteredEHFrameSections;

public:
  RuntimeDyldCOFFX86_64(RTDyldMemoryManager *MM) : RuntimeDyldCOFF(MM) {}

  unsigned getMaxStubSize() override {
    return 6; // 2-byte jmp instruction + 32-bit relative address
  }

  void resolveRelocation(const RelocationEntry &RE, uint64_t Value) override;

  relocation_iterator processRelocationRef(unsigned SectionID,
                                           relocation_iterator RelI,
                                           const ObjectFile &Obj,
                                           ObjSectionToIDMap &ObjSectionToID,
                                           StubMap &Stubs) override;

  unsigned getStubAlignment() override { return 1; }
  void registerEHFrames() override;
  void deregisterEHFrames() override;
  void finalizeLoad(const ObjectFile &Obj,
                    ObjSectionToIDMap &SectionMap) override;
};

} // end namespace llvm

#undef DEBUG_TYPE

#endif
