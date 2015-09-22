//===- Target.cpp ---------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Target.h"
#include "Error.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Object/ELF.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/ELF.h"

using namespace llvm;
using namespace llvm::object;
using namespace llvm::ELF;

namespace lld {
namespace elf2 {

std::unique_ptr<TargetInfo> Target;

TargetInfo::~TargetInfo() {}

X86TargetInfo::X86TargetInfo() { PCRelReloc = R_386_PC32; }

void X86TargetInfo::writePltEntry(uint8_t *Buf, uint64_t GotEntryAddr,
                                  uint64_t PltEntryAddr) const {
  ArrayRef<uint8_t> Jmp = {0xff, 0x25}; // jmpl *val
  memcpy(Buf, Jmp.data(), Jmp.size());
  Buf += Jmp.size();

  assert(isUInt<32>(GotEntryAddr));
  support::endian::write32le(Buf, GotEntryAddr);
  Buf += 4;

  ArrayRef<uint8_t> Nops = {0x90, 0x90};
  memcpy(Buf, Nops.data(), Nops.size());
}

bool X86TargetInfo::relocNeedsGot(uint32_t Type) const {
  if (relocNeedsPlt(Type))
    return true;
  switch (Type) {
  default:
    return false;
  case R_386_GOT32:
    return true;
  }
}

bool X86TargetInfo::relocNeedsPlt(uint32_t Type) const {
  switch (Type) {
  default:
    return false;
  case R_386_PLT32:
    return true;
  }
}

void X86TargetInfo::relocateOne(uint8_t *Buf, const void *RelP, uint32_t Type,
                                uint64_t BaseAddr, uint64_t SymVA) const {
  typedef ELFFile<ELF32LE>::Elf_Rel Elf_Rel;
  auto &Rel = *reinterpret_cast<const Elf_Rel *>(RelP);

  uint32_t Offset = Rel.r_offset;
  uint8_t *Location = Buf + Offset;
  uint32_t Addend = *(support::ulittle32_t *)Location;
  switch (Type) {
  case R_386_PC32:
    support::endian::write32le(Location, SymVA + Addend - (BaseAddr + Offset));
    break;
  case R_386_32:
    support::endian::write32le(Location, SymVA + Addend);
    break;
  default:
    error(Twine("unrecognized reloc ") + Twine(Type));
    break;
  }
}

X86_64TargetInfo::X86_64TargetInfo() { PCRelReloc = R_X86_64_PC32; }

void X86_64TargetInfo::writePltEntry(uint8_t *Buf, uint64_t GotEntryAddr,
                                     uint64_t PltEntryAddr) const {
  ArrayRef<uint8_t> Jmp = {0xff, 0x25}; // jmpq *val(%rip)
  memcpy(Buf, Jmp.data(), Jmp.size());
  Buf += Jmp.size();

  uintptr_t NextPC = PltEntryAddr + 6;
  intptr_t Delta = GotEntryAddr - NextPC;
  assert(isInt<32>(Delta));
  support::endian::write32le(Buf, Delta);
  Buf += 4;

  ArrayRef<uint8_t> Nops = {0x90, 0x90};
  memcpy(Buf, Nops.data(), Nops.size());
}

bool X86_64TargetInfo::relocNeedsGot(uint32_t Type) const {
  if (relocNeedsPlt(Type))
    return true;
  switch (Type) {
  default:
    return false;
  case R_X86_64_GOTPCREL:
    return true;
  }
}

bool X86_64TargetInfo::relocNeedsPlt(uint32_t Type) const {
  switch (Type) {
  default:
    return false;
  case R_X86_64_PLT32:
    return true;
  }
}

void X86_64TargetInfo::relocateOne(uint8_t *Buf, const void *RelP,
                                   uint32_t Type, uint64_t BaseAddr,
                                   uint64_t SymVA) const {
  typedef ELFFile<ELF64LE>::Elf_Rela Elf_Rela;
  auto &Rel = *reinterpret_cast<const Elf_Rela *>(RelP);

  uint64_t Offset = Rel.r_offset;
  uint8_t *Location = Buf + Offset;
  switch (Type) {
  case R_X86_64_PC32:
    support::endian::write32le(Location,
                               SymVA + Rel.r_addend - (BaseAddr + Offset));
    break;
  case R_X86_64_64:
    support::endian::write64le(Location, SymVA + Rel.r_addend);
    break;
  case R_X86_64_32: {
  case R_X86_64_32S:
    uint64_t VA = SymVA + Rel.r_addend;
    if (Type == R_X86_64_32 && !isUInt<32>(VA))
      error("R_X86_64_32 out of range");
    else if (!isInt<32>(VA))
      error("R_X86_64_32S out of range");

    support::endian::write32le(Location, VA);
    break;
  }
  default:
    error(Twine("unrecognized reloc ") + Twine(Type));
    break;
  }
}

PPC64TargetInfo::PPC64TargetInfo() {
  // PCRelReloc = FIXME
}
void PPC64TargetInfo::writePltEntry(uint8_t *Buf, uint64_t GotEntryAddr,
                                    uint64_t PltEntryAddr) const {}
bool PPC64TargetInfo::relocNeedsGot(uint32_t Type) const { return false; }
bool PPC64TargetInfo::relocNeedsPlt(uint32_t Type) const { return false; }
void PPC64TargetInfo::relocateOne(uint8_t *Buf, const void *RelP, uint32_t Type,
                                  uint64_t BaseAddr, uint64_t SymVA) const {
  typedef ELFFile<ELF64BE>::Elf_Rela Elf_Rela;
  auto &Rel = *reinterpret_cast<const Elf_Rela *>(RelP);

  uint64_t Offset = Rel.r_offset;
  uint8_t *Location = Buf + Offset;
  switch (Type) {
  case R_PPC64_ADDR64:
    support::endian::write64be(Location, SymVA + Rel.r_addend);
    break;
  case R_PPC64_TOC:
    // We don't create a TOC yet.
    break;
  default:
    error(Twine("unrecognized reloc ") + Twine(Type));
    break;
  }
}
}
}
