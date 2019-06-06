//===- PPC.cpp ------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OutputSections.h"
#include "Symbols.h"
#include "SyntheticSections.h"
#include "Target.h"
#include "lld/Common/ErrorHandler.h"
#include "llvm/Support/Endian.h"

using namespace llvm;
using namespace llvm::support::endian;
using namespace llvm::ELF;
using namespace lld;
using namespace lld::elf;

namespace {
class PPC final : public TargetInfo {
public:
  PPC();
  void writeGotHeader(uint8_t *Buf) const override;
  void writePltHeader(uint8_t *Buf) const override {
    llvm_unreachable("should call writePPC32GlinkSection() instead");
  }
  void writePlt(uint8_t *Buf, uint64_t GotPltEntryAddr, uint64_t PltEntryAddr,
    int32_t Index, unsigned RelOff) const override {
    llvm_unreachable("should call writePPC32GlinkSection() instead");
  }
  void writeGotPlt(uint8_t *Buf, const Symbol &S) const override;
  bool needsThunk(RelExpr Expr, RelType RelocType, const InputFile *File,
                  uint64_t BranchAddr, const Symbol &S) const override;
  uint32_t getThunkSectionSpacing() const override;
  bool inBranchRange(RelType Type, uint64_t Src, uint64_t Dst) const override;
  RelExpr getRelExpr(RelType Type, const Symbol &S,
                     const uint8_t *Loc) const override;
  void relocateOne(uint8_t *Loc, RelType Type, uint64_t Val) const override;
};
} // namespace

static uint16_t lo(uint32_t V) { return V; }
static uint16_t ha(uint32_t V) { return (V + 0x8000) >> 16; }

void elf::writePPC32GlinkSection(uint8_t *Buf, size_t NumEntries) {
  // On PPC Secure PLT ABI, bl foo@plt jumps to a call stub, which loads an
  // absolute address from a specific .plt slot (usually called .got.plt on
  // other targets) and jumps there.
  //
  // a) With immediate binding (BIND_NOW), the .plt entry is resolved at load
  // time. The .glink section is not used.
  // b) With lazy binding, the .plt entry points to a `b PLTresolve`
  // instruction in .glink, filled in by PPC::writeGotPlt().

  // Write N `b PLTresolve` first.
  for (size_t I = 0; I != NumEntries; ++I)
    write32(Buf + 4 * I, 0x48000000 | 4 * (NumEntries - I));
  Buf += 4 * NumEntries;

  // Then write PLTresolve(), which has two forms: PIC and non-PIC. PLTresolve()
  // computes the PLT index (by computing the distance from the landing b to
  // itself) and calls _dl_runtime_resolve() (in glibc).
  uint32_t GOT = In.Got->getVA();
  uint32_t Glink = In.Plt->getVA(); // VA of .glink
  const uint8_t *End = Buf + 64;
  if (Config->Pic) {
    uint32_t AfterBcl = In.Plt->getSize() - Target->PltHeaderSize + 12;
    uint32_t GotBcl = GOT + 4 - (Glink + AfterBcl);
    write32(Buf + 0, 0x3d6b0000 | ha(AfterBcl));  // addis r11,r11,1f-glink@ha
    write32(Buf + 4, 0x7c0802a6);                 // mflr r0
    write32(Buf + 8, 0x429f0005);                 // bcl 20,30,.+4
    write32(Buf + 12, 0x396b0000 | lo(AfterBcl)); // 1: addi r11,r11,1b-.glink@l
    write32(Buf + 16, 0x7d8802a6);                // mflr r12
    write32(Buf + 20, 0x7c0803a6);                // mtlr r0
    write32(Buf + 24, 0x7d6c5850);                // sub r11,r11,r12
    write32(Buf + 28, 0x3d8c0000 | ha(GotBcl));   // addis 12,12,GOT+4-1b@ha
    if (ha(GotBcl) == ha(GotBcl + 4)) {
      write32(Buf + 32, 0x800c0000 | lo(GotBcl)); // lwz r0,r12,GOT+4-1b@l(r12)
      write32(Buf + 36,
              0x818c0000 | lo(GotBcl + 4));       // lwz r12,r12,GOT+8-1b@l(r12)
    } else {
      write32(Buf + 32, 0x840c0000 | lo(GotBcl)); // lwzu r0,r12,GOT+4-1b@l(r12)
      write32(Buf + 36, 0x818c0000 | 4);          // lwz r12,r12,4(r12)
    }
    write32(Buf + 40, 0x7c0903a6);                // mtctr 0
    write32(Buf + 44, 0x7c0b5a14);                // add r0,11,11
    write32(Buf + 48, 0x7d605a14);                // add r11,0,11
    write32(Buf + 52, 0x4e800420);                // bctr
    Buf += 56;
  } else {
    write32(Buf + 0, 0x3d800000 | ha(GOT + 4));   // lis     r12,GOT+4@ha
    write32(Buf + 4, 0x3d6b0000 | ha(-Glink));    // addis   r11,r11,-Glink@ha
    if (ha(GOT + 4) == ha(GOT + 8))
      write32(Buf + 8, 0x800c0000 | lo(GOT + 4)); // lwz r0,GOT+4@l(r12)
    else
      write32(Buf + 8, 0x840c0000 | lo(GOT + 4)); // lwzu r0,GOT+4@l(r12)
    write32(Buf + 12, 0x396b0000 | lo(-Glink));   // addi    r11,r11,-Glink@l
    write32(Buf + 16, 0x7c0903a6);                // mtctr   r0
    write32(Buf + 20, 0x7c0b5a14);                // add     r0,r11,r11
    if (ha(GOT + 4) == ha(GOT + 8))
      write32(Buf + 24, 0x818c0000 | lo(GOT + 8)); // lwz r12,GOT+8@ha(r12)
    else
      write32(Buf + 24, 0x818c0000 | 4);          // lwz r12,4(r12)
    write32(Buf + 28, 0x7d605a14);                // add     r11,r0,r11
    write32(Buf + 32, 0x4e800420);                // bctr
    Buf += 36;
  }

  // Pad with nop. They should not be executed.
  for (; Buf < End; Buf += 4)
    write32(Buf, 0x60000000);
}

PPC::PPC() {
  GotRel = R_PPC_GLOB_DAT;
  NoneRel = R_PPC_NONE;
  PltRel = R_PPC_JMP_SLOT;
  RelativeRel = R_PPC_RELATIVE;
  IRelativeRel = R_PPC_IRELATIVE;
  GotBaseSymInGotPlt = false;
  GotHeaderEntriesNum = 3;
  GotPltHeaderEntriesNum = 0;
  PltHeaderSize = 64; // size of PLTresolve in .glink
  PltEntrySize = 4;

  NeedsThunks = true;

  DefaultMaxPageSize = 65536;
  DefaultImageBase = 0x10000000;

  write32(TrapInstr.data(), 0x7fe00008);
}

void PPC::writeGotHeader(uint8_t *Buf) const {
  // _GLOBAL_OFFSET_TABLE_[0] = _DYNAMIC
  // glibc stores _dl_runtime_resolve in _GLOBAL_OFFSET_TABLE_[1],
  // link_map in _GLOBAL_OFFSET_TABLE_[2].
  write32(Buf, In.Dynamic->getVA());
}

void PPC::writeGotPlt(uint8_t *Buf, const Symbol &S) const {
  // Address of the symbol resolver stub in .glink .
  write32(Buf, In.Plt->getVA() + 4 * S.PltIndex);
}

bool PPC::needsThunk(RelExpr Expr, RelType Type, const InputFile *File,
                     uint64_t BranchAddr, const Symbol &S) const {
  if (Type != R_PPC_REL24 && Type != R_PPC_PLTREL24)
    return false;
  if (S.isInPlt())
    return true;
  if (S.isUndefWeak())
    return false;
  return !(Expr == R_PC && PPC::inBranchRange(Type, BranchAddr, S.getVA()));
}

uint32_t PPC::getThunkSectionSpacing() const { return 0x2000000; }

bool PPC::inBranchRange(RelType Type, uint64_t Src, uint64_t Dst) const {
  uint64_t Offset = Dst - Src;
  if (Type == R_PPC_REL24 || R_PPC_PLTREL24)
    return isInt<26>(Offset);
  llvm_unreachable("unsupported relocation type used in branch");
}

RelExpr PPC::getRelExpr(RelType Type, const Symbol &S,
                        const uint8_t *Loc) const {
  switch (Type) {
  case R_PPC_REL14:
  case R_PPC_REL32:
  case R_PPC_LOCAL24PC:
  case R_PPC_REL16_LO:
  case R_PPC_REL16_HI:
  case R_PPC_REL16_HA:
    return R_PC;
  case R_PPC_GOT16:
    return R_GOT_OFF;
  case R_PPC_REL24:
    return R_PLT_PC;
  case R_PPC_PLTREL24:
    return R_PPC32_PLTREL;
  default:
    return R_ABS;
  }
}

void PPC::relocateOne(uint8_t *Loc, RelType Type, uint64_t Val) const {
  switch (Type) {
  case R_PPC_ADDR16:
  case R_PPC_GOT16:
    checkInt(Loc, Val, 16, Type);
    write16(Loc, Val);
    break;
  case R_PPC_ADDR16_HA:
  case R_PPC_REL16_HA:
    write16(Loc, ha(Val));
    break;
  case R_PPC_ADDR16_HI:
  case R_PPC_REL16_HI:
    write16(Loc, Val >> 16);
    break;
  case R_PPC_ADDR16_LO:
  case R_PPC_REL16_LO:
    write16(Loc, Val);
    break;
  case R_PPC_ADDR32:
  case R_PPC_GLOB_DAT:
  case R_PPC_REL32:
    write32(Loc, Val);
    break;
  case R_PPC_REL14: {
    uint32_t Mask = 0x0000FFFC;
    checkInt(Loc, Val, 16, Type);
    checkAlignment(Loc, Val, 4, Type);
    write32(Loc, (read32(Loc) & ~Mask) | (Val & Mask));
    break;
  }
  case R_PPC_REL24:
  case R_PPC_LOCAL24PC:
  case R_PPC_PLTREL24: {
    uint32_t Mask = 0x03FFFFFC;
    checkInt(Loc, Val, 26, Type);
    checkAlignment(Loc, Val, 4, Type);
    write32(Loc, (read32(Loc) & ~Mask) | (Val & Mask));
    break;
  }
  default:
    error(getErrorLocation(Loc) + "unrecognized relocation " + toString(Type));
  }
}

TargetInfo *elf::getPPCTargetInfo() {
  static PPC Target;
  return &Target;
}
