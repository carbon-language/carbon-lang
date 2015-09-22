//===- Target.cpp ---------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Target.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/ELF.h"

using namespace llvm;
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

X86_64TargetInfo::X86_64TargetInfo() { PCRelReloc = R_X86_64_PC32; }

void X86_64TargetInfo::writePltEntry(uint8_t *Buf, uint64_t GotEntryAddr,
                                     uint64_t PltEntryAddr) const {
  ArrayRef<uint8_t> Jmp = {0xff, 0x25}; // jmpq *val(%rip)
  memcpy(Buf, Jmp.data(), Jmp.size());
  Buf += Jmp.size();

  uintptr_t NextPC = PltEntryAddr + 6;
  uintptr_t Delta = GotEntryAddr - NextPC;
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
}
}
