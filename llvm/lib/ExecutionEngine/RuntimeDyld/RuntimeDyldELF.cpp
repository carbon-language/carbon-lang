//===-- RuntimeDyldELF.cpp - Run-time dynamic linker for MC-JIT ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implementation of ELF support for the MC-JIT runtime dynamic linker.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "dyld"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/IntervalMap.h"
#include "RuntimeDyldImpl.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/ELF.h"
#include "llvm/ADT/Triple.h"
using namespace llvm;
using namespace llvm::object;

namespace llvm {

namespace {

// FIXME: this function should probably not live here...
//
// Returns the name and address of an unrelocated symbol in an ELF section
void getSymbolInfo(symbol_iterator Sym, uint64_t &Addr, StringRef &Name) {
  //FIXME: error checking here required to catch corrupt ELF objects...
  error_code Err = Sym->getName(Name);

  uint64_t AddrInSection;
  Err = Sym->getAddress(AddrInSection);

  SectionRef empty_section;
  section_iterator Section(empty_section);
  Err = Sym->getSection(Section);

  StringRef SectionContents;
  Section->getContents(SectionContents);

  Addr = reinterpret_cast<uint64_t>(SectionContents.data()) + AddrInSection;
}

}

bool RuntimeDyldELF::loadObject(MemoryBuffer *InputBuffer) {
  if (!isCompatibleFormat(InputBuffer))
    return true;

  OwningPtr<ObjectFile> Obj(ObjectFile::createELFObjectFile(InputBuffer));

  Arch = Obj->getArch();

  // Map address in the Object file image to function names
  IntervalMap<uint64_t, StringRef>::Allocator A;
  IntervalMap<uint64_t, StringRef> FuncMap(A);

  // This is a bit of a hack.  The ObjectFile we've just loaded reports
  // section addresses as 0 and doesn't provide access to the section
  // offset (from which we could calculate the address.  Instead,
  // we're storing the address when it comes up in the ST_Debug case
  // below.
  //
  StringMap<uint64_t> DebugSymbolMap;

  symbol_iterator SymEnd = Obj->end_symbols();
  error_code Err;
  for (symbol_iterator Sym = Obj->begin_symbols();
       Sym != SymEnd; Sym.increment(Err)) {
    SymbolRef::Type Type;
    Sym->getType(Type);
    if (Type == SymbolRef::ST_Function) {
      StringRef Name;
      uint64_t Addr;
      getSymbolInfo(Sym, Addr, Name);

      uint64_t Size;
      Err = Sym->getSize(Size);

      uint8_t *Start;
      uint8_t *End;
      Start = reinterpret_cast<uint8_t*>(Addr);
      End   = reinterpret_cast<uint8_t*>(Addr + Size - 1);

      extractFunction(Name, Start, End);
      FuncMap.insert(Addr, Addr + Size - 1, Name);
    } else if (Type == SymbolRef::ST_Debug) {
      // This case helps us find section addresses
      StringRef Name;
      uint64_t Addr;
      getSymbolInfo(Sym, Addr, Name);
      DebugSymbolMap[Name] = Addr;
    }
  }

  // Iterate through the relocations for this object
  section_iterator SecEnd = Obj->end_sections();
  for (section_iterator Sec = Obj->begin_sections();
       Sec != SecEnd; Sec.increment(Err)) {
    StringRef SecName;
    uint64_t  SecAddr;
    Sec->getName(SecName);
    // Ignore sections that aren't in our map
    if (DebugSymbolMap.find(SecName) == DebugSymbolMap.end()) {
      continue;
    }
    SecAddr = DebugSymbolMap[SecName];
    relocation_iterator RelEnd = Sec->end_relocations();
    for (relocation_iterator Rel = Sec->begin_relocations();
         Rel != RelEnd; Rel.increment(Err)) {
      uint64_t RelOffset;
      uint64_t RelType;
      int64_t RelAddend;
      SymbolRef RelSym;
      StringRef SymName;
      uint64_t SymAddr;
      uint64_t SymOffset;

      Rel->getAddress(RelOffset);
      Rel->getType(RelType);
      Rel->getAdditionalInfo(RelAddend);
      Rel->getSymbol(RelSym);
      RelSym.getName(SymName);
      RelSym.getAddress(SymAddr);
      RelSym.getFileOffset(SymOffset);

      // If this relocation is inside a function, we want to store the
      // function name and a function-relative offset
      IntervalMap<uint64_t, StringRef>::iterator ContainingFunc
        = FuncMap.find(SecAddr + RelOffset);
      if (ContainingFunc.valid()) {
        // Re-base the relocation to make it relative to the target function
        RelOffset = (SecAddr + RelOffset) - ContainingFunc.start();
        Relocations[SymName].push_back(RelocationEntry(ContainingFunc.value(),
                                                       RelOffset,
                                                       RelType,
                                                       RelAddend,
                                                       true));
      } else {
        Relocations[SymName].push_back(RelocationEntry(SecName,
                                                       RelOffset,
                                                       RelType,
                                                       RelAddend,
                                                       false));
      }
    }
  }
  return false;
}

void RuntimeDyldELF::resolveRelocations() {
  // FIXME: deprecated. should be changed to use the by-section
  // allocation and relocation scheme.

  // Just iterate over the symbols in our symbol table and assign their
  // addresses.
  StringMap<SymbolLoc>::iterator i = SymbolTable.begin();
  StringMap<SymbolLoc>::iterator e = SymbolTable.end();
  for (;i != e; ++i) {
    assert (i->getValue().second == 0 && "non-zero offset in by-function sym!");
    reassignSymbolAddress(i->getKey(),
                          (uint8_t*)Sections[i->getValue().first].base());
  }
}

void RuntimeDyldELF::resolveX86_64Relocation(StringRef Name,
                                             uint8_t *Addr,
                                             const RelocationEntry &RE) {
  uint8_t *TargetAddr;
  if (RE.IsFunctionRelative) {
    StringMap<SymbolLoc>::const_iterator Loc = SymbolTable.find(RE.Target);
    assert(Loc != SymbolTable.end() && "Function for relocation not found");
    TargetAddr =
      reinterpret_cast<uint8_t*>(Sections[Loc->second.first].base()) +
      Loc->second.second + RE.Offset;
  } else {
    // FIXME: Get the address of the target section and add that to RE.Offset
    assert(0 && ("Non-function relocation not implemented yet!"));
  }

  switch (RE.Type) {
  default:
    assert(0 && ("Relocation type not implemented yet!"));
  break;
  case ELF::R_X86_64_64: {
    uint8_t **Target = reinterpret_cast<uint8_t**>(TargetAddr);
    *Target = Addr + RE.Addend;
    break;
  }
  case ELF::R_X86_64_32:
  case ELF::R_X86_64_32S: {
    uint64_t Value = reinterpret_cast<uint64_t>(Addr) + RE.Addend;
    // FIXME: Handle the possibility of this assertion failing
    assert((RE.Type == ELF::R_X86_64_32 && !(Value & 0xFFFFFFFF00000000ULL)) ||
           (RE.Type == ELF::R_X86_64_32S &&
            (Value & 0xFFFFFFFF00000000ULL) == 0xFFFFFFFF00000000ULL));
    uint32_t TruncatedAddr = (Value & 0xFFFFFFFF);
    uint32_t *Target = reinterpret_cast<uint32_t*>(TargetAddr);
    *Target = TruncatedAddr;
    break;
  }
  case ELF::R_X86_64_PC32: {
    uint32_t *Placeholder = reinterpret_cast<uint32_t*>(TargetAddr);
    uint64_t RealOffset = *Placeholder +
                           reinterpret_cast<uint64_t>(Addr) +
                           RE.Addend - reinterpret_cast<uint64_t>(TargetAddr);
    assert((RealOffset & 0xFFFFFFFF) == RealOffset);
    uint32_t TruncOffset = (RealOffset & 0xFFFFFFFF);
    *Placeholder = TruncOffset;
    break;
  }
  }
}

void RuntimeDyldELF::resolveX86Relocation(StringRef Name,
                                          uint8_t *Addr,
                                          const RelocationEntry &RE) {
  uint8_t *TargetAddr;
  if (RE.IsFunctionRelative) {
    StringMap<SymbolLoc>::const_iterator Loc = SymbolTable.find(RE.Target);
    assert(Loc != SymbolTable.end() && "Function for relocation not found");
    TargetAddr =
      reinterpret_cast<uint8_t*>(Sections[Loc->second.first].base()) +
      Loc->second.second + RE.Offset;
  } else {
    // FIXME: Get the address of the target section and add that to RE.Offset
    assert(0 && ("Non-function relocation not implemented yet!"));
  }

  switch (RE.Type) {
  case ELF::R_386_32: {
    uint8_t **Target = reinterpret_cast<uint8_t**>(TargetAddr);
    *Target = Addr + RE.Addend;
    break;
  }
  case ELF::R_386_PC32: {
    uint32_t *Placeholder = reinterpret_cast<uint32_t*>(TargetAddr);
    uint32_t RealOffset = *Placeholder + reinterpret_cast<uintptr_t>(Addr) +
                           RE.Addend - reinterpret_cast<uintptr_t>(TargetAddr);
    *Placeholder = RealOffset;
    break;
    }
    default:
      // There are other relocation types, but it appears these are the
      //  only ones currently used by the LLVM ELF object writer
      assert(0 && ("Relocation type not implemented yet!"));
      break;
  }
}

void RuntimeDyldELF::resolveArmRelocation(StringRef Name,
                                          uint8_t *Addr,
                                          const RelocationEntry &RE) {
}

void RuntimeDyldELF::resolveRelocation(StringRef Name,
                                       uint8_t *Addr,
                                       const RelocationEntry &RE) {
  switch (Arch) {
  case Triple::x86_64:
    resolveX86_64Relocation(Name, Addr, RE);
    break;
  case Triple::x86:
    resolveX86Relocation(Name, Addr, RE);
    break;
  case Triple::arm:
    resolveArmRelocation(Name, Addr, RE);
    break;
  default:
    assert(0 && "Unsupported CPU type!");
    break;
  }
}

void RuntimeDyldELF::reassignSymbolAddress(StringRef Name, uint8_t *Addr) {
  // FIXME: deprecated. switch to reassignSectionAddress() instead.
  //
  // Actually moving the symbol address requires by-section mapping.
  assert(Sections[SymbolTable.lookup(Name).first].base() == (void*)Addr &&
         "Unable to relocate section in by-function JIT allocation model!");

  RelocationList &Relocs = Relocations[Name];
  for (unsigned i = 0, e = Relocs.size(); i != e; ++i) {
    RelocationEntry &RE = Relocs[i];
    resolveRelocation(Name, Addr, RE);
  }
}

// Assign an address to a symbol name and resolve all the relocations
// associated with it.
void RuntimeDyldELF::reassignSectionAddress(unsigned SectionID, uint64_t Addr) {
  // The address to use for relocation resolution is not
  // the address of the local section buffer. We must be doing
  // a remote execution environment of some sort. Re-apply any
  // relocations referencing this section with the given address.
  //
  // Addr is a uint64_t because we can't assume the pointer width
  // of the target is the same as that of the host. Just use a generic
  // "big enough" type.
  assert(0);
}

bool RuntimeDyldELF::isCompatibleFormat(const MemoryBuffer *InputBuffer) const {
  StringRef Magic = InputBuffer->getBuffer().slice(0, ELF::EI_NIDENT);
  return (memcmp(Magic.data(), ELF::ElfMagic, strlen(ELF::ElfMagic))) == 0;
}
} // namespace llvm
