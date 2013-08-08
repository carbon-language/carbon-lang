//===-- PPCMachObjectWriter.cpp - PPC Mach-O Writer -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/PPCMCTargetDesc.h"
#include "MCTargetDesc/PPCFixupKinds.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCAsmLayout.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCMachObjectWriter.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Object/MachOFormat.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"

using namespace llvm;
using namespace llvm::object;

namespace {
class PPCMachObjectWriter : public MCMachObjectTargetWriter {
  bool RecordScatteredRelocation(MachObjectWriter *Writer,
                                 const MCAssembler &Asm,
                                 const MCAsmLayout &Layout,
                                 const MCFragment *Fragment,
                                 const MCFixup &Fixup, MCValue Target,
                                 unsigned Log2Size, uint64_t &FixedValue);

  void RecordPPCRelocation(MachObjectWriter *Writer, const MCAssembler &Asm,
                           const MCAsmLayout &Layout,
                           const MCFragment *Fragment, const MCFixup &Fixup,
                           MCValue Target, uint64_t &FixedValue);

public:
  PPCMachObjectWriter(bool Is64Bit, uint32_t CPUType, uint32_t CPUSubtype)
      : MCMachObjectTargetWriter(Is64Bit, CPUType, CPUSubtype,
                                 /*UseAggressiveSymbolFolding=*/Is64Bit) {}

  void RecordRelocation(MachObjectWriter *Writer, const MCAssembler &Asm,
                        const MCAsmLayout &Layout, const MCFragment *Fragment,
                        const MCFixup &Fixup, MCValue Target,
                        uint64_t &FixedValue) {
    if (Writer->is64Bit()) {
      report_fatal_error("Relocation emission for MachO/PPC64 unimplemented.");
    } else
      RecordPPCRelocation(Writer, Asm, Layout, Fragment, Fixup, Target,
                          FixedValue);
  }
};
}

/// computes the log2 of the size of the relocation,
/// used for relocation_info::r_length.
static unsigned getFixupKindLog2Size(unsigned Kind) {
  switch (Kind) {
  default:
    report_fatal_error("log2size(FixupKind): Unhandled fixup kind!");
  case FK_PCRel_1:
  case FK_Data_1:
    return 0;
  case FK_PCRel_2:
  case FK_Data_2:
    return 1;
  case FK_PCRel_4:
  case PPC::fixup_ppc_brcond14:
  case PPC::fixup_ppc_half16:
  case PPC::fixup_ppc_br24:
  case FK_Data_4:
    return 2;
  case FK_PCRel_8:
  case FK_Data_8:
    return 3;
  }
  return 0;
}

/// Translates generic PPC fixup kind to Mach-O/PPC relocation type enum.
/// Outline based on PPCELFObjectWriter::getRelocTypeInner().
static unsigned getRelocType(const MCValue &Target,
                             const MCFixupKind FixupKind, // from
                                                          // Fixup.getKind()
                             const bool IsPCRel) {
  const MCSymbolRefExpr::VariantKind Modifier =
      Target.isAbsolute() ? MCSymbolRefExpr::VK_None
                          : Target.getSymA()->getKind();
  // determine the type of the relocation
  unsigned Type = macho::RIT_Vanilla;
  if (IsPCRel) { // relative to PC
    switch ((unsigned)FixupKind) {
    default:
      report_fatal_error("Unimplemented fixup kind (relative)");
    case PPC::fixup_ppc_br24:
      Type = macho::RIT_PPC_BR24; // R_PPC_REL24
      break;
    case PPC::fixup_ppc_brcond14:
      Type = macho::RIT_PPC_BR14;
      break;
    case PPC::fixup_ppc_half16:
      switch (Modifier) {
      default:
        llvm_unreachable("Unsupported modifier for half16 fixup");
      case MCSymbolRefExpr::VK_PPC_HA:
        Type = macho::RIT_PPC_HA16;
        break;
      case MCSymbolRefExpr::VK_PPC_LO:
        Type = macho::RIT_PPC_LO16;
        break;
      case MCSymbolRefExpr::VK_PPC_HI:
        Type = macho::RIT_PPC_HI16;
        break;
      }
      break;
    }
  } else {
    switch ((unsigned)FixupKind) {
    default:
      report_fatal_error("Unimplemented fixup kind (absolute)!");
    case PPC::fixup_ppc_half16:
      switch (Modifier) {
      default:
        llvm_unreachable("Unsupported modifier for half16 fixup");
      case MCSymbolRefExpr::VK_PPC_HA:
        Type = macho::RIT_PPC_HA16_SECTDIFF;
        break;
      case MCSymbolRefExpr::VK_PPC_LO:
        Type = macho::RIT_PPC_LO16_SECTDIFF;
        break;
      case MCSymbolRefExpr::VK_PPC_HI:
        Type = macho::RIT_PPC_HI16_SECTDIFF;
        break;
      }
      break;
    case FK_Data_4:
      break;
    case FK_Data_2:
      break;
    }
  }
  return Type;
}

static void makeRelocationInfo(macho::RelocationEntry &MRE,
                               const uint32_t FixupOffset, const uint32_t Index,
                               const unsigned IsPCRel, const unsigned Log2Size,
                               const unsigned IsExtern, const unsigned Type) {
  MRE.Word0 = FixupOffset;
  // The bitfield offsets that work (as determined by trial-and-error)
  // are different than what is documented in the mach-o manuals.
  // Is this an endianness issue w/ PPC?
  MRE.Word1 = ((Index << 8) |    // was << 0
               (IsPCRel << 7) |  // was << 24
               (Log2Size << 5) | // was << 25
               (IsExtern << 4) | // was << 27
               (Type << 0));     // was << 28
}

static void
makeScatteredRelocationInfo(macho::RelocationEntry &MRE, const uint32_t Addr,
                            const unsigned Type, const unsigned Log2Size,
                            const unsigned IsPCRel, const uint32_t Value2) {
  // For notes on bitfield positions and endianness, see:
  // https://developer.apple.com/library/mac/documentation/developertools/conceptual/MachORuntime/Reference/reference.html#//apple_ref/doc/uid/20001298-scattered_relocation_entry
  MRE.Word0 = ((Addr << 0) | (Type << 24) | (Log2Size << 28) | (IsPCRel << 30) |
               macho::RF_Scattered);
  MRE.Word1 = Value2;
}

/// Compute fixup offset (address).
static uint32_t getFixupOffset(const MCAsmLayout &Layout,
                               const MCFragment *Fragment,
                               const MCFixup &Fixup) {
  uint32_t FixupOffset = Layout.getFragmentOffset(Fragment) + Fixup.getOffset();
  // On Mach-O, ppc_fixup_half16 relocations must refer to the
  // start of the instruction, not the second halfword, as ELF does
  if (Fixup.getKind() == PPC::fixup_ppc_half16)
    FixupOffset &= ~uint32_t(3);
  return FixupOffset;
}

/// \return false if falling back to using non-scattered relocation,
/// otherwise true for normal scattered relocation.
/// based on X86MachObjectWriter::RecordScatteredRelocation
/// and ARMMachObjectWriter::RecordScatteredRelocation
bool PPCMachObjectWriter::RecordScatteredRelocation(
    MachObjectWriter *Writer, const MCAssembler &Asm, const MCAsmLayout &Layout,
    const MCFragment *Fragment, const MCFixup &Fixup, MCValue Target,
    unsigned Log2Size, uint64_t &FixedValue) {
  // caller already computes these, can we just pass and reuse?
  const uint32_t FixupOffset = getFixupOffset(Layout, Fragment, Fixup);
  const MCFixupKind FK = Fixup.getKind();
  const unsigned IsPCRel = Writer->isFixupKindPCRel(Asm, FK);
  const unsigned Type = getRelocType(Target, FK, IsPCRel);

  // Is this a local or SECTDIFF relocation entry?
  // SECTDIFF relocation entries have symbol subtractions,
  // and require two entries, the first for the add-symbol value,
  // the second for the subtract-symbol value.

  // See <reloc.h>.
  const MCSymbol *A = &Target.getSymA()->getSymbol();
  MCSymbolData *A_SD = &Asm.getSymbolData(*A);

  if (!A_SD->getFragment())
    report_fatal_error("symbol '" + A->getName() +
                       "' can not be undefined in a subtraction expression");

  uint32_t Value = Writer->getSymbolAddress(A_SD, Layout);
  uint64_t SecAddr =
      Writer->getSectionAddress(A_SD->getFragment()->getParent());
  FixedValue += SecAddr;
  uint32_t Value2 = 0;

  if (const MCSymbolRefExpr *B = Target.getSymB()) {
    MCSymbolData *B_SD = &Asm.getSymbolData(B->getSymbol());

    if (!B_SD->getFragment())
      report_fatal_error("symbol '" + B->getSymbol().getName() +
                         "' can not be undefined in a subtraction expression");

    // FIXME: is Type correct? see include/llvm/Object/MachOFormat.h
    Value2 = Writer->getSymbolAddress(B_SD, Layout);
    FixedValue -= Writer->getSectionAddress(B_SD->getFragment()->getParent());
  }
  // FIXME: does FixedValue get used??

  // Relocations are written out in reverse order, so the PAIR comes first.
  if (Type == macho::RIT_PPC_SECTDIFF || Type == macho::RIT_PPC_HI16_SECTDIFF ||
      Type == macho::RIT_PPC_LO16_SECTDIFF ||
      Type == macho::RIT_PPC_HA16_SECTDIFF ||
      Type == macho::RIT_PPC_LO14_SECTDIFF ||
      Type == macho::RIT_PPC_LOCAL_SECTDIFF) {
    // X86 had this piece, but ARM does not
    // If the offset is too large to fit in a scattered relocation,
    // we're hosed. It's an unfortunate limitation of the MachO format.
    if (FixupOffset > 0xffffff) {
      char Buffer[32];
      format("0x%x", FixupOffset).print(Buffer, sizeof(Buffer));
      Asm.getContext().FatalError(Fixup.getLoc(),
                                  Twine("Section too large, can't encode "
                                        "r_address (") +
                                      Buffer + ") into 24 bits of scattered "
                                               "relocation entry.");
      llvm_unreachable("fatal error returned?!");
    }

    // Is this supposed to follow MCTarget/PPCAsmBackend.cpp:adjustFixupValue()?
    // see PPCMCExpr::EvaluateAsRelocatableImpl()
    uint32_t other_half = 0;
    switch (Type) {
    case macho::RIT_PPC_LO16_SECTDIFF:
      other_half = (FixedValue >> 16) & 0xffff;
      // applyFixupOffset longer extracts the high part because it now assumes
      // this was already done.
      // It looks like this is not true for the FixedValue needed with Mach-O
      // relocs.
      // So we need to adjust FixedValue again here.
      FixedValue &= 0xffff;
      break;
    case macho::RIT_PPC_HA16_SECTDIFF:
      other_half = FixedValue & 0xffff;
      FixedValue =
          ((FixedValue >> 16) + ((FixedValue & 0x8000) ? 1 : 0)) & 0xffff;
      break;
    case macho::RIT_PPC_HI16_SECTDIFF:
      other_half = FixedValue & 0xffff;
      FixedValue = (FixedValue >> 16) & 0xffff;
      break;
    default:
      llvm_unreachable("Invalid PPC scattered relocation type.");
      break;
    }

    macho::RelocationEntry MRE;
    makeScatteredRelocationInfo(MRE, other_half, macho::RIT_Pair, Log2Size,
                                IsPCRel, Value2);
    Writer->addRelocation(Fragment->getParent(), MRE);
  } else {
    // If the offset is more than 24-bits, it won't fit in a scattered
    // relocation offset field, so we fall back to using a non-scattered
    // relocation. This is a bit risky, as if the offset reaches out of
    // the block and the linker is doing scattered loading on this
    // symbol, things can go badly.
    //
    // Required for 'as' compatibility.
    if (FixupOffset > 0xffffff)
      return false;
  }
  macho::RelocationEntry MRE;
  makeScatteredRelocationInfo(MRE, FixupOffset, Type, Log2Size, IsPCRel, Value);
  Writer->addRelocation(Fragment->getParent(), MRE);
  return true;
}

// see PPCELFObjectWriter for a general outline of cases
void PPCMachObjectWriter::RecordPPCRelocation(
    MachObjectWriter *Writer, const MCAssembler &Asm, const MCAsmLayout &Layout,
    const MCFragment *Fragment, const MCFixup &Fixup, MCValue Target,
    uint64_t &FixedValue) {
  const MCFixupKind FK = Fixup.getKind(); // unsigned
  const unsigned Log2Size = getFixupKindLog2Size(FK);
  const bool IsPCRel = Writer->isFixupKindPCRel(Asm, FK);
  const unsigned RelocType = getRelocType(Target, FK, IsPCRel);

  // If this is a difference or a defined symbol plus an offset, then we need a
  // scattered relocation entry. Differences always require scattered
  // relocations.
  if (Target.getSymB() &&
      // Q: are branch targets ever scattered?
      RelocType != macho::RIT_PPC_BR24 && RelocType != macho::RIT_PPC_BR14) {
    RecordScatteredRelocation(Writer, Asm, Layout, Fragment, Fixup, Target,
                              Log2Size, FixedValue);
    return;
  }

  // this doesn't seem right for RIT_PPC_BR24
  // Get the symbol data, if any.
  MCSymbolData *SD = 0;
  if (Target.getSymA())
    SD = &Asm.getSymbolData(Target.getSymA()->getSymbol());

  // See <reloc.h>.
  const uint32_t FixupOffset = getFixupOffset(Layout, Fragment, Fixup);
  unsigned Index = 0;
  unsigned IsExtern = 0;
  unsigned Type = RelocType;

  if (Target.isAbsolute()) { // constant
                             // SymbolNum of 0 indicates the absolute section.
                             //
    // FIXME: Currently, these are never generated (see code below). I cannot
    // find a case where they are actually emitted.
    report_fatal_error("FIXME: relocations to absolute targets "
                       "not yet implemented");
    // the above line stolen from ARM, not sure
  } else {
    // Resolve constant variables.
    if (SD->getSymbol().isVariable()) {
      int64_t Res;
      if (SD->getSymbol().getVariableValue()->EvaluateAsAbsolute(
              Res, Layout, Writer->getSectionAddressMap())) {
        FixedValue = Res;
        return;
      }
    }

    // Check whether we need an external or internal relocation.
    if (Writer->doesSymbolRequireExternRelocation(SD)) {
      IsExtern = 1;
      Index = SD->getIndex();
      // For external relocations, make sure to offset the fixup value to
      // compensate for the addend of the symbol address, if it was
      // undefined. This occurs with weak definitions, for example.
      if (!SD->Symbol->isUndefined())
        FixedValue -= Layout.getSymbolOffset(SD);
    } else {
      // The index is the section ordinal (1-based).
      const MCSectionData &SymSD =
          Asm.getSectionData(SD->getSymbol().getSection());
      Index = SymSD.getOrdinal() + 1;
      FixedValue += Writer->getSectionAddress(&SymSD);
    }
    if (IsPCRel)
      FixedValue -= Writer->getSectionAddress(Fragment->getParent());
  }

  // struct relocation_info (8 bytes)
  macho::RelocationEntry MRE;
  makeRelocationInfo(MRE, FixupOffset, Index, IsPCRel, Log2Size, IsExtern,
                     Type);
  Writer->addRelocation(Fragment->getParent(), MRE);
}

MCObjectWriter *llvm::createPPCMachObjectWriter(raw_ostream &OS, bool Is64Bit,
                                                uint32_t CPUType,
                                                uint32_t CPUSubtype) {
  return createMachObjectWriter(
      new PPCMachObjectWriter(Is64Bit, CPUType, CPUSubtype), OS,
      /*IsLittleEndian=*/false);
}
