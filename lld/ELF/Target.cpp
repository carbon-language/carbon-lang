//===- Target.cpp ---------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Machine-specific stuff, such as applying relocations, creation of
// GOT or PLT entries, etc., are is handled in this file.
//
//===----------------------------------------------------------------------===//

#include "Target.h"
#include "Error.h"
#include "OutputSections.h"
#include "Symbols.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Object/ELF.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/ELF.h"

using namespace llvm;
using namespace llvm::object;
using namespace llvm::support::endian;
using namespace llvm::ELF;

namespace lld {
namespace elf2 {

std::unique_ptr<TargetInfo> Target;

TargetInfo *createTarget() {
  switch (Config->EMachine) {
  case EM_386:
    return new X86TargetInfo();
  case EM_AARCH64:
    return new AArch64TargetInfo();
  case EM_MIPS:
    return new MipsTargetInfo();
  case EM_PPC:
    return new PPCTargetInfo();
  case EM_PPC64:
    return new PPC64TargetInfo();
  case EM_X86_64:
    return new X86_64TargetInfo();
  }
  error("Unknown target machine");
}

TargetInfo::~TargetInfo() {}

bool TargetInfo::relocPointsToGot(uint32_t Type) const { return false; }

bool TargetInfo::isRelRelative(uint32_t Type) const { return true; }

X86TargetInfo::X86TargetInfo() {
  PCRelReloc = R_386_PC32;
  GotReloc = R_386_GLOB_DAT;
  GotRefReloc = R_386_GOT32;
}

void X86TargetInfo::writePltEntry(uint8_t *Buf, uint64_t GotEntryAddr,
                                  uint64_t PltEntryAddr) const {
  // jmpl *val; nop; nop
  const uint8_t Inst[] = {0xff, 0x25, 0, 0, 0, 0, 0x90, 0x90};
  memcpy(Buf, Inst, sizeof(Inst));
  assert(isUInt<32>(GotEntryAddr));
  write32le(Buf + 2, GotEntryAddr);
}

bool X86TargetInfo::relocNeedsGot(uint32_t Type, const SymbolBody &S) const {
  return Type == R_386_GOT32 || relocNeedsPlt(Type, S);
}

bool X86TargetInfo::relocPointsToGot(uint32_t Type) const {
  return Type == R_386_GOTPC;
}

bool X86TargetInfo::relocNeedsPlt(uint32_t Type, const SymbolBody &S) const {
  return Type == R_386_PLT32 || (Type == R_386_PC32 && S.isShared());
}

static void add32le(uint8_t *L, int32_t V) { write32le(L, read32le(L) + V); }
static void or32le(uint8_t *L, int32_t V) { write32le(L, read32le(L) | V); }

void X86TargetInfo::relocateOne(uint8_t *Buf, uint8_t *BufEnd, const void *RelP,
                                uint32_t Type, uint64_t BaseAddr,
                                uint64_t SymVA) const {
  typedef ELFFile<ELF32LE>::Elf_Rel Elf_Rel;
  auto &Rel = *reinterpret_cast<const Elf_Rel *>(RelP);

  uint32_t Offset = Rel.r_offset;
  uint8_t *Loc = Buf + Offset;
  switch (Type) {
  case R_386_GOT32:
    add32le(Loc, SymVA - Out<ELF32LE>::Got->getVA());
    break;
  case R_386_PC32:
    add32le(Loc, SymVA - (BaseAddr + Offset));
    break;
  case R_386_32:
    add32le(Loc, SymVA);
    break;
  default:
    error("unrecognized reloc " + Twine(Type));
  }
}

X86_64TargetInfo::X86_64TargetInfo() {
  PCRelReloc = R_X86_64_PC32;
  GotReloc = R_X86_64_GLOB_DAT;
  GotRefReloc = R_X86_64_PC32;
  RelativeReloc = R_X86_64_RELATIVE;
}

void X86_64TargetInfo::writePltEntry(uint8_t *Buf, uint64_t GotEntryAddr,
                                     uint64_t PltEntryAddr) const {
  // jmpq *val(%rip); nop; nop
  const uint8_t Inst[] = {0xff, 0x25, 0, 0, 0, 0, 0x90, 0x90};
  memcpy(Buf, Inst, sizeof(Inst));

  uint64_t NextPC = PltEntryAddr + 6;
  int64_t Delta = GotEntryAddr - NextPC;
  assert(isInt<32>(Delta));
  write32le(Buf + 2, Delta);
}

bool X86_64TargetInfo::relocNeedsGot(uint32_t Type, const SymbolBody &S) const {
  return Type == R_X86_64_GOTPCREL || relocNeedsPlt(Type, S);
}

bool X86_64TargetInfo::relocNeedsPlt(uint32_t Type, const SymbolBody &S) const {
  switch (Type) {
  default:
    return false;
  case R_X86_64_PC32:
    // This relocation is defined to have a value of (S + A - P).
    // The problems start when a non PIC program calls a function in a shared
    // library.
    // In an ideal world, we could just report an error saying the relocation
    // can overflow at runtime.
    // In the real world with glibc, crt1.o has a R_X86_64_PC32 pointing to
    // libc.so.
    //
    // The general idea on how to handle such cases is to create a PLT entry
    // and use that as the function value.
    //
    // For the static linking part, we just return true and everything else
    // will use the the PLT entry as the address.
    //
    // The remaining (unimplemented) problem is making sure pointer equality
    // still works. We need the help of the dynamic linker for that. We
    // let it know that we have a direct reference to a so symbol by creating
    // an undefined symbol with a non zero st_value. Seeing that, the
    // dynamic linker resolves the symbol to the value of the symbol we created.
    // This is true even for got entries, so pointer equality is maintained.
    // To avoid an infinite loop, the only entry that points to the
    // real function is a dedicated got entry used by the plt. That is
    // identified by special relocation types (R_X86_64_JUMP_SLOT,
    // R_386_JMP_SLOT, etc).
    return S.isShared();
  case R_X86_64_PLT32:
    return true;
  }
}

bool X86_64TargetInfo::isRelRelative(uint32_t Type) const {
  switch (Type) {
  default:
    return false;
  case R_X86_64_PC64:
  case R_X86_64_PC32:
  case R_X86_64_PC16:
  case R_X86_64_PC8:
    return true;
  }
}

void X86_64TargetInfo::relocateOne(uint8_t *Buf, uint8_t *BufEnd,
                                   const void *RelP, uint32_t Type,
                                   uint64_t BaseAddr, uint64_t SymVA) const {
  typedef ELFFile<ELF64LE>::Elf_Rela Elf_Rela;
  auto &Rel = *reinterpret_cast<const Elf_Rela *>(RelP);

  uint64_t Offset = Rel.r_offset;
  uint8_t *Loc = Buf + Offset;
  switch (Type) {
  case R_X86_64_PC32:
  case R_X86_64_GOTPCREL:
    write32le(Loc, SymVA + Rel.r_addend - (BaseAddr + Offset));
    break;
  case R_X86_64_64:
    write64le(Loc, SymVA + Rel.r_addend);
    break;
  case R_X86_64_32: {
  case R_X86_64_32S:
    uint64_t VA = SymVA + Rel.r_addend;
    if (Type == R_X86_64_32 && !isUInt<32>(VA))
      error("R_X86_64_32 out of range");
    else if (!isInt<32>(VA))
      error("R_X86_64_32S out of range");

    write32le(Loc, VA);
    break;
  }
  default:
    error("unrecognized reloc " + Twine(Type));
  }
}

// Relocation masks following the #lo(value), #hi(value), #ha(value),
// #higher(value), #highera(value), #highest(value), and #highesta(value)
// macros defined in section 4.5.1. Relocation Types of the PPC-elf64abi
// document.

static uint16_t applyPPCLo(uint64_t V) { return V & 0xffff; }

static uint16_t applyPPCHi(uint64_t V) { return (V >> 16) & 0xffff; }

static uint16_t applyPPCHa(uint64_t V) { return ((V + 0x8000) >> 16) & 0xffff; }

static uint16_t applyPPCHigher(uint64_t V) { return (V >> 32) & 0xffff; }

static uint16_t applyPPCHighera(uint64_t V) {
  return ((V + 0x8000) >> 32) & 0xffff;
}

static uint16_t applyPPCHighest(uint64_t V) { return V >> 48; }

static uint16_t applyPPCHighesta(uint64_t V) { return (V + 0x8000) >> 48; }

PPC64TargetInfo::PPC64TargetInfo() {
  PCRelReloc = R_PPC64_REL24;
  GotReloc = R_PPC64_GLOB_DAT;
  GotRefReloc = R_PPC64_REL64;
  RelativeReloc = R_PPC64_RELATIVE;
  PltEntrySize = 32;

  // We need 64K pages (at least under glibc/Linux, the loader won't
  // set different permissions on a finer granularity than that).
  PageSize = 65536;
}

static uint64_t getPPC64TocBase() {
  // The TOC consists of sections .got, .toc, .tocbss, .plt in that
  // order. The TOC starts where the first of these sections starts.

  // FIXME: This obviously does not do the right thing when there is no .got
  // section, but there is a .toc or .tocbss section.
  uint64_t TocVA = Out<ELF64BE>::Got->getVA();
  if (!TocVA)
    TocVA = Out<ELF64BE>::Plt->getVA();

  // Per the ppc64-elf-linux ABI, The TOC base is TOC value plus 0x8000
  // thus permitting a full 64 Kbytes segment. Note that the glibc startup
  // code (crt1.o) assumes that you can get from the TOC base to the
  // start of the .toc section with only a single (signed) 16-bit relocation.
  return TocVA + 0x8000;
}

void PPC64TargetInfo::writePltEntry(uint8_t *Buf, uint64_t GotEntryAddr,
                                    uint64_t PltEntryAddr) const {
  uint64_t Off = GotEntryAddr - getPPC64TocBase();

  // FIXME: What we should do, in theory, is get the offset of the function
  // descriptor in the .opd section, and use that as the offset from %r2 (the
  // TOC-base pointer). Instead, we have the GOT-entry offset, and that will
  // be a pointer to the function descriptor in the .opd section. Using
  // this scheme is simpler, but requires an extra indirection per PLT dispatch.

  write32be(Buf,      0xf8410000);                   // std %r2, 40(%r1)
  write32be(Buf + 4,  0x3d620000 | applyPPCHa(Off)); // addis %r11, %r2, X@ha
  write32be(Buf + 8,  0xe98b0000 | applyPPCLo(Off)); // ld %r12, X@l(%r11)
  write32be(Buf + 12, 0xe96c0000);                   // ld %r11,0(%r12)
  write32be(Buf + 16, 0x7d6903a6);                   // mtctr %r11
  write32be(Buf + 20, 0xe84c0008);                   // ld %r2,8(%r12)
  write32be(Buf + 24, 0xe96c0010);                   // ld %r11,16(%r12)
  write32be(Buf + 28, 0x4e800420);                   // bctr
}

bool PPC64TargetInfo::relocNeedsGot(uint32_t Type, const SymbolBody &S) const {
  if (relocNeedsPlt(Type, S))
    return true;

  switch (Type) {
  default: return false;
  case R_PPC64_GOT16:
  case R_PPC64_GOT16_LO:
  case R_PPC64_GOT16_HI:
  case R_PPC64_GOT16_HA:
  case R_PPC64_GOT16_DS:
  case R_PPC64_GOT16_LO_DS:
    return true;
  }
}

bool PPC64TargetInfo::relocNeedsPlt(uint32_t Type, const SymbolBody &S) const {
  if (Type != R_PPC64_REL24)
    return false;

  // These are function calls that need to be redirected through a PLT stub.
  return S.isShared() || (S.isUndefined() && S.isWeak());
}

bool PPC64TargetInfo::isRelRelative(uint32_t Type) const {
  switch (Type) {
  default:
    return false;
  case R_PPC64_REL24:
  case R_PPC64_REL14:
  case R_PPC64_REL14_BRTAKEN:
  case R_PPC64_REL14_BRNTAKEN:
  case R_PPC64_REL32:
  case R_PPC64_REL64:
    return true;
  }
}

void PPC64TargetInfo::relocateOne(uint8_t *Buf, uint8_t *BufEnd,
                                  const void *RelP, uint32_t Type,
                                  uint64_t BaseAddr, uint64_t SymVA) const {
  typedef ELFFile<ELF64BE>::Elf_Rela Elf_Rela;
  auto &Rel = *reinterpret_cast<const Elf_Rela *>(RelP);

  uint8_t *L = Buf + Rel.r_offset;
  uint64_t S = SymVA;
  int64_t A = Rel.r_addend;
  uint64_t P = BaseAddr + Rel.r_offset;
  uint64_t TB = getPPC64TocBase();

  if (Type == R_PPC64_TOC) {
    write64be(L, TB);
    return;
  }

  // For a TOC-relative relocation, adjust the addend and proceed in terms of
  // the corresponding ADDR16 relocation type.
  switch (Type) {
  case R_PPC64_TOC16:       Type = R_PPC64_ADDR16;       A -= TB; break;
  case R_PPC64_TOC16_DS:    Type = R_PPC64_ADDR16_DS;    A -= TB; break;
  case R_PPC64_TOC16_LO:    Type = R_PPC64_ADDR16_LO;    A -= TB; break;
  case R_PPC64_TOC16_LO_DS: Type = R_PPC64_ADDR16_LO_DS; A -= TB; break;
  case R_PPC64_TOC16_HI:    Type = R_PPC64_ADDR16_HI;    A -= TB; break;
  case R_PPC64_TOC16_HA:    Type = R_PPC64_ADDR16_HA;    A -= TB; break;
  default: break;
  }

  uint64_t R = S + A;

  switch (Type) {
  case R_PPC64_ADDR16:
    write16be(L, applyPPCLo(R));
    break;
  case R_PPC64_ADDR16_DS:
    if (!isInt<16>(R))
      error("Relocation R_PPC64_ADDR16_DS overflow");
    write16be(L, (read16be(L) & 3) | (R & ~3));
    break;
  case R_PPC64_ADDR16_LO:
    write16be(L, applyPPCLo(R));
    break;
  case R_PPC64_ADDR16_LO_DS:
    write16be(L, (read16be(L) & 3) | (applyPPCLo(R) & ~3));
    break;
  case R_PPC64_ADDR16_HI:
    write16be(L, applyPPCHi(R));
    break;
  case R_PPC64_ADDR16_HA:
    write16be(L, applyPPCHa(R));
    break;
  case R_PPC64_ADDR16_HIGHER:
    write16be(L, applyPPCHigher(R));
    break;
  case R_PPC64_ADDR16_HIGHERA:
    write16be(L, applyPPCHighera(R));
    break;
  case R_PPC64_ADDR16_HIGHEST:
    write16be(L, applyPPCHighest(R));
    break;
  case R_PPC64_ADDR16_HIGHESTA:
    write16be(L, applyPPCHighesta(R));
    break;
  case R_PPC64_ADDR14: {
    if ((R & 3) != 0)
      error("Improper alignment for relocation R_PPC64_ADDR14");

    // Preserve the AA/LK bits in the branch instruction
    uint8_t AALK = L[3];
    write16be(L + 2, (AALK & 3) | (R & 0xfffc));
    break;
  }
  case R_PPC64_REL16_LO:
    write16be(L, applyPPCLo(R - P));
    break;
  case R_PPC64_REL16_HI:
    write16be(L, applyPPCHi(R - P));
    break;
  case R_PPC64_REL16_HA:
    write16be(L, applyPPCHa(R - P));
    break;
  case R_PPC64_ADDR32:
    if (!isInt<32>(R))
      error("Relocation R_PPC64_ADDR32 overflow");
    write32be(L, R);
    break;
  case R_PPC64_REL24: {
    uint64_t PltStart = Out<ELF64BE>::Plt->getVA();
    uint64_t PltEnd = PltStart + Out<ELF64BE>::Plt->getSize();
    bool InPlt = PltStart <= S + A && S + A < PltEnd;

    if (!InPlt && Out<ELF64BE>::Opd) {
      // If this is a local call, and we currently have the address of a
      // function-descriptor, get the underlying code address instead.
      uint64_t OpdStart = Out<ELF64BE>::Opd->getVA();
      uint64_t OpdEnd = OpdStart + Out<ELF64BE>::Opd->getSize();
      bool InOpd = OpdStart <= S + A && S + A < OpdEnd;

      if (InOpd)
        R = read64be(&Out<ELF64BE>::OpdBuf[S + A - OpdStart]);
    }

    uint32_t Mask = 0x03FFFFFC;
    if (!isInt<24>(R - P))
      error("Relocation R_PPC64_REL24 overflow");
    write32be(L, (read32be(L) & ~Mask) | ((R - P) & Mask));

    if (InPlt && L + 8 <= BufEnd &&
        read32be(L + 4) == 0x60000000 /* nop */)
      write32be(L + 4, 0xe8410028); // ld %r2, 40(%r1)
    break;
  }
  case R_PPC64_REL32:
    if (!isInt<32>(R - P))
      error("Relocation R_PPC64_REL32 overflow");
    write32be(L, R - P);
    break;
  case R_PPC64_REL64:
    write64be(L, R - P);
    break;
  case R_PPC64_ADDR64:
    write64be(L, R);
    break;
  default:
    error("unrecognized reloc " + Twine(Type));
  }
}

PPCTargetInfo::PPCTargetInfo() {
  // PCRelReloc = FIXME
  // GotReloc = FIXME
  PageSize = 65536;
}
void PPCTargetInfo::writePltEntry(uint8_t *Buf, uint64_t GotEntryAddr,
                                  uint64_t PltEntryAddr) const {}
bool PPCTargetInfo::relocNeedsGot(uint32_t Type, const SymbolBody &S) const {
  return false;
}
bool PPCTargetInfo::relocNeedsPlt(uint32_t Type, const SymbolBody &S) const {
  return false;
}
void PPCTargetInfo::relocateOne(uint8_t *Buf, uint8_t *BufEnd,
                                const void *RelP, uint32_t Type,
                                uint64_t BaseAddr, uint64_t SymVA) const {}

AArch64TargetInfo::AArch64TargetInfo() {
  // PCRelReloc = FIXME
  // GotReloc = FIXME
}
void AArch64TargetInfo::writePltEntry(uint8_t *Buf, uint64_t GotEntryAddr,
                                      uint64_t PltEntryAddr) const {}
bool AArch64TargetInfo::relocNeedsGot(uint32_t Type,
                                      const SymbolBody &S) const {
  return false;
}
bool AArch64TargetInfo::relocNeedsPlt(uint32_t Type,
                                      const SymbolBody &S) const {
  return false;
}

static void updateAArch64Adr(uint8_t *L, uint64_t Imm) {
  uint32_t ImmLo = (Imm & 0x3) << 29;
  uint32_t ImmHi = ((Imm & 0x1FFFFC) >> 2) << 5;
  uint64_t Mask = (0x3 << 29) | (0x7FFFF << 5);
  write32le(L, (read32le(L) & ~Mask) | ImmLo | ImmHi);
}

// Page(Expr) is the page address of the expression Expr, defined
// as (Expr & ~0xFFF). (This applies even if the machine page size
// supported by the platform has a different value.)
static uint64_t getAArch64Page(uint64_t Expr) {
  return Expr & (~static_cast<uint64_t>(0xFFF));
}

void AArch64TargetInfo::relocateOne(uint8_t *Buf, uint8_t *BufEnd,
                                    const void *RelP, uint32_t Type,
                                    uint64_t BaseAddr, uint64_t SymVA) const {
  typedef ELFFile<ELF64LE>::Elf_Rela Elf_Rela;
  auto &Rel = *reinterpret_cast<const Elf_Rela *>(RelP);

  uint8_t *L = Buf + Rel.r_offset;
  uint64_t S = SymVA;
  int64_t A = Rel.r_addend;
  uint64_t P = BaseAddr + Rel.r_offset;
  switch (Type) {
  case R_AARCH64_ABS16:
    if (!isInt<16>(S + A))
      error("Relocation R_AARCH64_ABS16 out of range");
    write16le(L, S + A);
    break;
  case R_AARCH64_ABS32:
    if (!isInt<32>(S + A))
      error("Relocation R_AARCH64_ABS32 out of range");
    write32le(L, S + A);
    break;
  case R_AARCH64_ABS64:
    // No overflow check needed.
    write64le(L, S + A);
    break;
  case R_AARCH64_ADD_ABS_LO12_NC:
    // No overflow check needed.
    or32le(L, ((S + A) & 0xFFF) << 10);
    break;
  case R_AARCH64_ADR_PREL_LO21: {
    uint64_t X = S + A - P;
    if (!isInt<21>(X))
      error("Relocation R_AARCH64_ADR_PREL_LO21 out of range");
    updateAArch64Adr(L, X & 0x1FFFFF);
    break;
  }
  case R_AARCH64_ADR_PREL_PG_HI21: {
    uint64_t X = getAArch64Page(S + A) - getAArch64Page(P);
    if (!isInt<33>(X))
      error("Relocation R_AARCH64_ADR_PREL_PG_HI21 out of range");
    updateAArch64Adr(L, (X >> 12) & 0x1FFFFF); // X[32:12]
    break;
  }
  default:
    error("unrecognized reloc " + Twine(Type));
  }
}

MipsTargetInfo::MipsTargetInfo() {
  // PCRelReloc = FIXME
  // GotReloc = FIXME
  PageSize = 65536;
}

void MipsTargetInfo::writePltEntry(uint8_t *Buf, uint64_t GotEntryAddr,
                                   uint64_t PltEntryAddr) const {}

bool MipsTargetInfo::relocNeedsGot(uint32_t Type, const SymbolBody &S) const {
  return false;
}

bool MipsTargetInfo::relocNeedsPlt(uint32_t Type, const SymbolBody &S) const {
  return false;
}

void MipsTargetInfo::relocateOne(uint8_t *Buf, uint8_t *BufEnd,
                                 const void *RelP, uint32_t Type,
                                 uint64_t BaseAddr, uint64_t SymVA) const {
  typedef ELFFile<ELF32LE>::Elf_Rel Elf_Rel;
  auto &Rel = *reinterpret_cast<const Elf_Rel *>(RelP);

  switch (Type) {
  case R_MIPS_32:
    add32le(Buf + Rel.r_offset, SymVA);
    break;
  default:
    error("unrecognized reloc " + Twine(Type));
  }
}
}
}
