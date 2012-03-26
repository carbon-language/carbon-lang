//===-- RuntimeDyldMachO.h - Run-time dynamic linker for MC-JIT ---*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// MachO support for MC-JIT runtime dynamic linker.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_RUNTIME_DYLD_MACHO_H
#define LLVM_RUNTIME_DYLD_MACHO_H

#include "llvm/ADT/IndexedMap.h"
#include "llvm/Object/MachOObject.h"
#include "llvm/Support/Format.h"
#include "RuntimeDyldImpl.h"

using namespace llvm;
using namespace llvm::object;


namespace llvm {
class RuntimeDyldMachO : public RuntimeDyldImpl {

  // For each symbol, keep a list of relocations based on it. Anytime
  // its address is reassigned (the JIT re-compiled the function, e.g.),
  // the relocations get re-resolved.
  // The symbol (or section) the relocation is sourced from is the Key
  // in the relocation list where it's stored.
  struct RelocationEntry {
    unsigned    SectionID;  // Section the relocation is contained in.
    uint64_t    Offset;     // Offset into the section for the relocation.
    uint32_t    Data;       // Second word of the raw macho relocation entry.
    int64_t     Addend;     // Addend encoded in the instruction itself, if any,
                            // plus the offset into the source section for
                            // the symbol once the relocation is resolvable.

    RelocationEntry(unsigned id, uint64_t offset, uint32_t data, int64_t addend)
      : SectionID(id), Offset(offset), Data(data), Addend(addend) {}
  };
  typedef SmallVector<RelocationEntry, 4> RelocationList;

  // For each section, keep a list of referrers in that section that are clients
  // of relocations in other sections.  Whenever a relocation gets created,
  // create a corresponding referrer.  Whenever relocations are re-resolved,
  // re-resolve the referrers' relocations as well.
  struct Referrer {
    unsigned    SectionID;  // Section whose RelocationList contains the relocation.
    uint32_t    Index;      // Index of the RelocatonEntry in that RelocationList.

    Referrer(unsigned id, uint32_t index)
      : SectionID(id), Index(index) {}
  };
  typedef SmallVector<Referrer, 4> ReferrerList;

  // Relocations to sections already loaded. Indexed by SectionID which is the
  // source of the address. The target where the address will be writen is
  // SectionID/Offset in the relocation itself.
  IndexedMap<RelocationList> Relocations;
  // Referrers corresponding to Relocations.
  IndexedMap<ReferrerList> Referrers;
  // Relocations to symbols that are not yet resolved. Must be external
  // relocations by definition. Indexed by symbol name.
  StringMap<RelocationList> UnresolvedRelocations;

  bool resolveRelocation(uint8_t *LocalAddress,
                         uint64_t FinalAddress,
                         uint64_t Value,
                         bool isPCRel,
                         unsigned Type,
                         unsigned Size,
                         int64_t Addend);
  bool resolveI386Relocation(uint8_t *LocalAddress,
                             uint64_t FinalAddress,
                             uint64_t Value,
                             bool isPCRel,
                             unsigned Type,
                             unsigned Size,
                             int64_t Addend);
  bool resolveX86_64Relocation(uint8_t *LocalAddress,
                               uint64_t FinalAddress,
                               uint64_t Value,
                               bool isPCRel,
                               unsigned Type,
                               unsigned Size,
                               int64_t Addend);
  bool resolveARMRelocation(uint8_t *LocalAddress,
                            uint64_t FinalAddress,
                            uint64_t Value,
                            bool isPCRel,
                            unsigned Type,
                            unsigned Size,
                            int64_t Addend);

  bool loadSegment32(const MachOObject *Obj,
                     const MachOObject::LoadCommandInfo *SegmentLCI,
                     const InMemoryStruct<macho::SymtabLoadCommand> &SymtabLC);
  bool loadSegment64(const MachOObject *Obj,
                     const MachOObject::LoadCommandInfo *SegmentLCI,
                     const InMemoryStruct<macho::SymtabLoadCommand> &SymtabLC);
  bool processSymbols32(const MachOObject *Obj,
                      SmallVectorImpl<unsigned> &SectionMap,
                      SmallVectorImpl<StringRef> &SymbolNames,
                      const InMemoryStruct<macho::SymtabLoadCommand> &SymtabLC);
  bool processSymbols64(const MachOObject *Obj,
                      SmallVectorImpl<unsigned> &SectionMap,
                      SmallVectorImpl<StringRef> &SymbolNames,
                      const InMemoryStruct<macho::SymtabLoadCommand> &SymtabLC);

  void resolveSymbol(StringRef Name);

public:
  RuntimeDyldMachO(RTDyldMemoryManager *mm) : RuntimeDyldImpl(mm) {}

  bool loadObject(MemoryBuffer *InputBuffer);

  void reassignSectionAddress(unsigned SectionID, uint64_t Addr);

  static bool isKnownFormat(const MemoryBuffer *InputBuffer);

  bool isCompatibleFormat(const MemoryBuffer *InputBuffer) const {
    return isKnownFormat(InputBuffer);
  }
};

} // end namespace llvm

#endif
