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
    // For each symbol, keep a list of relocations based on it. Anytime
    // its address is reassigned (the JIT re-compiled the function, e.g.),
    // the relocations get re-resolved.
    struct RelocationEntry {
      // Function or section this relocation is contained in.
      std::string Target;
      // Offset into the target function or section for the relocation.
      uint32_t    Offset;
      // Relocation type
      uint32_t    Type;
      // Addend encoded in the instruction itself, if any.
      int32_t     Addend;
      // Has the relocation been recalcuated as an offset within a function?
      bool        IsFunctionRelative;
      // Has this relocation been resolved previously?
      bool        isResolved;

      RelocationEntry(StringRef t,
                      uint32_t offset,
                      uint32_t type,
                      int32_t addend,
                      bool isFunctionRelative)
        : Target(t)
        , Offset(offset)
        , Type(type)
        , Addend(addend)
        , IsFunctionRelative(isFunctionRelative)
        , isResolved(false) { }
    };
    typedef SmallVector<RelocationEntry, 4> RelocationList;
    StringMap<RelocationList> Relocations;
    unsigned Arch;

    void resolveRelocations();

    void resolveX86_64Relocation(StringRef Name,
                                 uint8_t *Addr,
                                 const RelocationEntry &RE);

    void resolveX86Relocation(StringRef Name,
                              uint8_t *Addr,
                              const RelocationEntry &RE);

    void resolveArmRelocation(StringRef Name,
                              uint8_t *Addr,
                              const RelocationEntry &RE);

    void resolveRelocation(StringRef Name,
                           uint8_t *Addr,
                           const RelocationEntry &RE);

public:
  RuntimeDyldELF(RTDyldMemoryManager *mm) : RuntimeDyldImpl(mm) {}

  bool loadObject(MemoryBuffer *InputBuffer);

  void reassignSymbolAddress(StringRef Name, uint8_t *Addr);
  void reassignSectionAddress(unsigned SectionID, uint64_t Addr);

  bool isCompatibleFormat(const MemoryBuffer *InputBuffer) const;
};

} // end namespace llvm

#endif 

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
    // For each symbol, keep a list of relocations based on it. Anytime
    // its address is reassigned (the JIT re-compiled the function, e.g.),
    // the relocations get re-resolved.
    struct RelocationEntry {
      // Function or section this relocation is contained in.
      std::string Target;
      // Offset into the target function or section for the relocation.
      uint32_t    Offset;
      // Relocation type
      uint32_t    Type;
      // Addend encoded in the instruction itself, if any.
      int32_t     Addend;
      // Has the relocation been recalcuated as an offset within a function?
      bool        IsFunctionRelative;
      // Has this relocation been resolved previously?
      bool        isResolved;

      RelocationEntry(StringRef t,
                      uint32_t offset,
                      uint32_t type,
                      int32_t addend,
                      bool isFunctionRelative)
        : Target(t)
        , Offset(offset)
        , Type(type)
        , Addend(addend)
        , IsFunctionRelative(isFunctionRelative)
        , isResolved(false) { }
    };
    typedef SmallVector<RelocationEntry, 4> RelocationList;
    StringMap<RelocationList> Relocations;
    unsigned Arch;

    void resolveRelocations();

    void resolveX86_64Relocation(StringRef Name,
                                 uint8_t *Addr,
                                 const RelocationEntry &RE);

    void resolveX86Relocation(StringRef Name,
                              uint8_t *Addr,
                              const RelocationEntry &RE);

    void resolveArmRelocation(StringRef Name,
                              uint8_t *Addr,
                              const RelocationEntry &RE);

    void resolveRelocation(StringRef Name,
                           uint8_t *Addr,
                           const RelocationEntry &RE);

public:
  RuntimeDyldELF(RTDyldMemoryManager *mm) : RuntimeDyldImpl(mm) {}

  bool loadObject(MemoryBuffer *InputBuffer);

  void reassignSymbolAddress(StringRef Name, uint8_t *Addr);
  void reassignSectionAddress(unsigned SectionID, uint64_t Addr);

  bool isCompatibleFormat(const MemoryBuffer *InputBuffer) const;
};

} // end namespace llvm

#endif 

