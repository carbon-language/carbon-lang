//===-- ARMAsmBackend.cpp - ARM Assembler Backend -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/ARMMCTargetDesc.h"
#include "MCTargetDesc/ARMBaseInfo.h"
#include "MCTargetDesc/ARMFixupKinds.h"
#include "MCTargetDesc/ARMAddressingModes.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCDirectives.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCMachObjectWriter.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Object/MachOFormat.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

namespace {
class ARMELFObjectWriter : public MCELFObjectTargetWriter {
public:
  ARMELFObjectWriter(uint8_t OSABI)
    : MCELFObjectTargetWriter(/*Is64Bit*/ false, OSABI, ELF::EM_ARM,
                              /*HasRelocationAddend*/ false) {}
};

class ARMAsmBackend : public MCAsmBackend {
  const MCSubtargetInfo* STI;
  bool isThumbMode;  // Currently emitting Thumb code.
public:
  ARMAsmBackend(const Target &T, const StringRef TT)
    : MCAsmBackend(), STI(ARM_MC::createARMMCSubtargetInfo(TT, "", "")),
      isThumbMode(TT.startswith("thumb")) {}

  ~ARMAsmBackend() {
    delete STI;
  }

  unsigned getNumFixupKinds() const { return ARM::NumTargetFixupKinds; }

  bool hasNOP() const {
    return (STI->getFeatureBits() & ARM::HasV6T2Ops) != 0;
  }

  const MCFixupKindInfo &getFixupKindInfo(MCFixupKind Kind) const {
    const static MCFixupKindInfo Infos[ARM::NumTargetFixupKinds] = {
// This table *must* be in the order that the fixup_* kinds are defined in
// ARMFixupKinds.h.
//
// Name                      Offset (bits) Size (bits)     Flags
{ "fixup_arm_ldst_pcrel_12", 0,            32,  MCFixupKindInfo::FKF_IsPCRel },
{ "fixup_t2_ldst_pcrel_12",  0,            32,  MCFixupKindInfo::FKF_IsPCRel |
                                   MCFixupKindInfo::FKF_IsAlignedDownTo32Bits},
{ "fixup_arm_pcrel_10_unscaled", 0,        32,  MCFixupKindInfo::FKF_IsPCRel },
{ "fixup_arm_pcrel_10",      0,            32,  MCFixupKindInfo::FKF_IsPCRel },
{ "fixup_t2_pcrel_10",       0,            32,  MCFixupKindInfo::FKF_IsPCRel |
                                   MCFixupKindInfo::FKF_IsAlignedDownTo32Bits},
{ "fixup_thumb_adr_pcrel_10",0,            8,   MCFixupKindInfo::FKF_IsPCRel |
                                   MCFixupKindInfo::FKF_IsAlignedDownTo32Bits},
{ "fixup_arm_adr_pcrel_12",  0,            32,  MCFixupKindInfo::FKF_IsPCRel },
{ "fixup_t2_adr_pcrel_12",   0,            32,  MCFixupKindInfo::FKF_IsPCRel |
                                   MCFixupKindInfo::FKF_IsAlignedDownTo32Bits},
{ "fixup_arm_condbranch",    0,            24,  MCFixupKindInfo::FKF_IsPCRel },
{ "fixup_arm_uncondbranch",  0,            24,  MCFixupKindInfo::FKF_IsPCRel },
{ "fixup_t2_condbranch",     0,            32,  MCFixupKindInfo::FKF_IsPCRel },
{ "fixup_t2_uncondbranch",   0,            32,  MCFixupKindInfo::FKF_IsPCRel },
{ "fixup_arm_thumb_br",      0,            16,  MCFixupKindInfo::FKF_IsPCRel },
{ "fixup_arm_thumb_bl",      0,            32,  MCFixupKindInfo::FKF_IsPCRel },
{ "fixup_arm_thumb_blx",     0,            32,  MCFixupKindInfo::FKF_IsPCRel },
{ "fixup_arm_thumb_cb",      0,            16,  MCFixupKindInfo::FKF_IsPCRel },
{ "fixup_arm_thumb_cp",      0,             8,  MCFixupKindInfo::FKF_IsPCRel },
{ "fixup_arm_thumb_bcc",     0,             8,  MCFixupKindInfo::FKF_IsPCRel },
// movw / movt: 16-bits immediate but scattered into two chunks 0 - 12, 16 - 19.
{ "fixup_arm_movt_hi16",     0,            20,  0 },
{ "fixup_arm_movw_lo16",     0,            20,  0 },
{ "fixup_t2_movt_hi16",      0,            20,  0 },
{ "fixup_t2_movw_lo16",      0,            20,  0 },
{ "fixup_arm_movt_hi16_pcrel", 0,          20,  MCFixupKindInfo::FKF_IsPCRel },
{ "fixup_arm_movw_lo16_pcrel", 0,          20,  MCFixupKindInfo::FKF_IsPCRel },
{ "fixup_t2_movt_hi16_pcrel", 0,           20,  MCFixupKindInfo::FKF_IsPCRel },
{ "fixup_t2_movw_lo16_pcrel", 0,           20,  MCFixupKindInfo::FKF_IsPCRel },
    };

    if (Kind < FirstTargetFixupKind)
      return MCAsmBackend::getFixupKindInfo(Kind);

    assert(unsigned(Kind - FirstTargetFixupKind) < getNumFixupKinds() &&
           "Invalid kind!");
    return Infos[Kind - FirstTargetFixupKind];
  }

  bool MayNeedRelaxation(const MCInst &Inst) const;

  bool fixupNeedsRelaxation(const MCFixup &Fixup,
                            uint64_t Value,
                            const MCInstFragment *DF,
                            const MCAsmLayout &Layout) const;

  void RelaxInstruction(const MCInst &Inst, MCInst &Res) const;

  bool WriteNopData(uint64_t Count, MCObjectWriter *OW) const;

  void HandleAssemblerFlag(MCAssemblerFlag Flag) {
    switch (Flag) {
    default: break;
    case MCAF_Code16:
      setIsThumb(true);
      break;
    case MCAF_Code32:
      setIsThumb(false);
      break;
    }
  }

  unsigned getPointerSize() const { return 4; }
  bool isThumb() const { return isThumbMode; }
  void setIsThumb(bool it) { isThumbMode = it; }
};
} // end anonymous namespace

static unsigned getRelaxedOpcode(unsigned Op) {
  switch (Op) {
  default: return Op;
  case ARM::tBcc: return ARM::t2Bcc;
  }
}

bool ARMAsmBackend::MayNeedRelaxation(const MCInst &Inst) const {
  if (getRelaxedOpcode(Inst.getOpcode()) != Inst.getOpcode())
    return true;
  return false;
}

bool ARMAsmBackend::fixupNeedsRelaxation(const MCFixup &Fixup,
                                         uint64_t Value,
                                         const MCInstFragment *DF,
                                         const MCAsmLayout &Layout) const {
  // Relaxing tBcc to t2Bcc. tBcc has a signed 9-bit displacement with the
  // low bit being an implied zero. There's an implied +4 offset for the
  // branch, so we adjust the other way here to determine what's
  // encodable.
  //
  // Relax if the value is too big for a (signed) i8.
  int64_t Offset = int64_t(Value) - 4;
  return Offset > 254 || Offset < -256;
}

void ARMAsmBackend::RelaxInstruction(const MCInst &Inst, MCInst &Res) const {
  unsigned RelaxedOp = getRelaxedOpcode(Inst.getOpcode());

  // Sanity check w/ diagnostic if we get here w/ a bogus instruction.
  if (RelaxedOp == Inst.getOpcode()) {
    SmallString<256> Tmp;
    raw_svector_ostream OS(Tmp);
    Inst.dump_pretty(OS);
    OS << "\n";
    report_fatal_error("unexpected instruction to relax: " + OS.str());
  }

  // The instructions we're relaxing have (so far) the same operands.
  // We just need to update to the proper opcode.
  Res = Inst;
  Res.setOpcode(RelaxedOp);
}

bool ARMAsmBackend::WriteNopData(uint64_t Count, MCObjectWriter *OW) const {
  const uint16_t Thumb1_16bitNopEncoding = 0x46c0; // using MOV r8,r8
  const uint16_t Thumb2_16bitNopEncoding = 0xbf00; // NOP
  const uint32_t ARMv4_NopEncoding = 0xe1a0000; // using MOV r0,r0
  const uint32_t ARMv6T2_NopEncoding = 0xe320f000; // NOP
  if (isThumb()) {
    const uint16_t nopEncoding = hasNOP() ? Thumb2_16bitNopEncoding
                                          : Thumb1_16bitNopEncoding;
    uint64_t NumNops = Count / 2;
    for (uint64_t i = 0; i != NumNops; ++i)
      OW->Write16(nopEncoding);
    if (Count & 1)
      OW->Write8(0);
    return true;
  }
  // ARM mode
  const uint32_t nopEncoding = hasNOP() ? ARMv6T2_NopEncoding
                                        : ARMv4_NopEncoding;
  uint64_t NumNops = Count / 4;
  for (uint64_t i = 0; i != NumNops; ++i)
    OW->Write32(nopEncoding);
  // FIXME: should this function return false when unable to write exactly
  // 'Count' bytes with NOP encodings?
  switch (Count % 4) {
  default: break; // No leftover bytes to write
  case 1: OW->Write8(0); break;
  case 2: OW->Write16(0); break;
  case 3: OW->Write16(0); OW->Write8(0xa0); break;
  }

  return true;
}

static unsigned adjustFixupValue(unsigned Kind, uint64_t Value) {
  switch (Kind) {
  default:
    llvm_unreachable("Unknown fixup kind!");
  case FK_Data_1:
  case FK_Data_2:
  case FK_Data_4:
    return Value;
  case ARM::fixup_arm_movt_hi16:
    Value >>= 16;
    // Fallthrough
  case ARM::fixup_arm_movw_lo16:
  case ARM::fixup_arm_movt_hi16_pcrel:
  case ARM::fixup_arm_movw_lo16_pcrel: {
    unsigned Hi4 = (Value & 0xF000) >> 12;
    unsigned Lo12 = Value & 0x0FFF;
    // inst{19-16} = Hi4;
    // inst{11-0} = Lo12;
    Value = (Hi4 << 16) | (Lo12);
    return Value;
  }
  case ARM::fixup_t2_movt_hi16:
    Value >>= 16;
    // Fallthrough
  case ARM::fixup_t2_movw_lo16:
  case ARM::fixup_t2_movt_hi16_pcrel:  //FIXME: Shouldn't this be shifted like
                                       // the other hi16 fixup?
  case ARM::fixup_t2_movw_lo16_pcrel: {
    unsigned Hi4 = (Value & 0xF000) >> 12;
    unsigned i = (Value & 0x800) >> 11;
    unsigned Mid3 = (Value & 0x700) >> 8;
    unsigned Lo8 = Value & 0x0FF;
    // inst{19-16} = Hi4;
    // inst{26} = i;
    // inst{14-12} = Mid3;
    // inst{7-0} = Lo8;
    Value = (Hi4 << 16) | (i << 26) | (Mid3 << 12) | (Lo8);
    uint64_t swapped = (Value & 0xFFFF0000) >> 16;
    swapped |= (Value & 0x0000FFFF) << 16;
    return swapped;
  }
  case ARM::fixup_arm_ldst_pcrel_12:
    // ARM PC-relative values are offset by 8.
    Value -= 4;
    // FALLTHROUGH
  case ARM::fixup_t2_ldst_pcrel_12: {
    // Offset by 4, adjusted by two due to the half-word ordering of thumb.
    Value -= 4;
    bool isAdd = true;
    if ((int64_t)Value < 0) {
      Value = -Value;
      isAdd = false;
    }
    assert ((Value < 4096) && "Out of range pc-relative fixup value!");
    Value |= isAdd << 23;

    // Same addressing mode as fixup_arm_pcrel_10,
    // but with 16-bit halfwords swapped.
    if (Kind == ARM::fixup_t2_ldst_pcrel_12) {
      uint64_t swapped = (Value & 0xFFFF0000) >> 16;
      swapped |= (Value & 0x0000FFFF) << 16;
      return swapped;
    }

    return Value;
  }
  case ARM::fixup_thumb_adr_pcrel_10:
    return ((Value - 4) >> 2) & 0xff;
  case ARM::fixup_arm_adr_pcrel_12: {
    // ARM PC-relative values are offset by 8.
    Value -= 8;
    unsigned opc = 4; // bits {24-21}. Default to add: 0b0100
    if ((int64_t)Value < 0) {
      Value = -Value;
      opc = 2; // 0b0010
    }
    assert(ARM_AM::getSOImmVal(Value) != -1 &&
           "Out of range pc-relative fixup value!");
    // Encode the immediate and shift the opcode into place.
    return ARM_AM::getSOImmVal(Value) | (opc << 21);
  }

  case ARM::fixup_t2_adr_pcrel_12: {
    Value -= 4;
    unsigned opc = 0;
    if ((int64_t)Value < 0) {
      Value = -Value;
      opc = 5;
    }

    uint32_t out = (opc << 21);
    out |= (Value & 0x800) << 15;
    out |= (Value & 0x700) << 4;
    out |= (Value & 0x0FF);

    uint64_t swapped = (out & 0xFFFF0000) >> 16;
    swapped |= (out & 0x0000FFFF) << 16;
    return swapped;
  }

  case ARM::fixup_arm_condbranch:
  case ARM::fixup_arm_uncondbranch:
    // These values don't encode the low two bits since they're always zero.
    // Offset by 8 just as above.
    return 0xffffff & ((Value - 8) >> 2);
  case ARM::fixup_t2_uncondbranch: {
    Value = Value - 4;
    Value >>= 1; // Low bit is not encoded.

    uint32_t out = 0;
    bool I =  Value & 0x800000;
    bool J1 = Value & 0x400000;
    bool J2 = Value & 0x200000;
    J1 ^= I;
    J2 ^= I;

    out |= I  << 26; // S bit
    out |= !J1 << 13; // J1 bit
    out |= !J2 << 11; // J2 bit
    out |= (Value & 0x1FF800)  << 5; // imm6 field
    out |= (Value & 0x0007FF);        // imm11 field

    uint64_t swapped = (out & 0xFFFF0000) >> 16;
    swapped |= (out & 0x0000FFFF) << 16;
    return swapped;
  }
  case ARM::fixup_t2_condbranch: {
    Value = Value - 4;
    Value >>= 1; // Low bit is not encoded.

    uint64_t out = 0;
    out |= (Value & 0x80000) << 7; // S bit
    out |= (Value & 0x40000) >> 7; // J2 bit
    out |= (Value & 0x20000) >> 4; // J1 bit
    out |= (Value & 0x1F800) << 5; // imm6 field
    out |= (Value & 0x007FF);      // imm11 field

    uint32_t swapped = (out & 0xFFFF0000) >> 16;
    swapped |= (out & 0x0000FFFF) << 16;
    return swapped;
  }
  case ARM::fixup_arm_thumb_bl: {
    // The value doesn't encode the low bit (always zero) and is offset by
    // four. The value is encoded into disjoint bit positions in the destination
    // opcode. x = unchanged, I = immediate value bit, S = sign extension bit
    //
    //   BL:  xxxxxSIIIIIIIIII xxxxxIIIIIIIIIII
    //
    // Note that the halfwords are stored high first, low second; so we need
    // to transpose the fixup value here to map properly.
    unsigned isNeg = (int64_t(Value - 4) < 0) ? 1 : 0;
    uint32_t Binary = 0;
    Value = 0x3fffff & ((Value - 4) >> 1);
    Binary  = (Value & 0x7ff) << 16;    // Low imm11 value.
    Binary |= (Value & 0x1ffc00) >> 11; // High imm10 value.
    Binary |= isNeg << 10;              // Sign bit.
    return Binary;
  }
  case ARM::fixup_arm_thumb_blx: {
    // The value doesn't encode the low two bits (always zero) and is offset by
    // four (see fixup_arm_thumb_cp). The value is encoded into disjoint bit
    // positions in the destination opcode. x = unchanged, I = immediate value
    // bit, S = sign extension bit, 0 = zero.
    //
    //   BLX: xxxxxSIIIIIIIIII xxxxxIIIIIIIIII0
    //
    // Note that the halfwords are stored high first, low second; so we need
    // to transpose the fixup value here to map properly.
    unsigned isNeg = (int64_t(Value-4) < 0) ? 1 : 0;
    uint32_t Binary = 0;
    Value = 0xfffff & ((Value - 2) >> 2);
    Binary  = (Value & 0x3ff) << 17;    // Low imm10L value.
    Binary |= (Value & 0xffc00) >> 10;  // High imm10H value.
    Binary |= isNeg << 10;              // Sign bit.
    return Binary;
  }
  case ARM::fixup_arm_thumb_cp:
    // Offset by 4, and don't encode the low two bits. Two bytes of that
    // 'off by 4' is implicitly handled by the half-word ordering of the
    // Thumb encoding, so we only need to adjust by 2 here.
    return ((Value - 2) >> 2) & 0xff;
  case ARM::fixup_arm_thumb_cb: {
    // Offset by 4 and don't encode the lower bit, which is always 0.
    uint32_t Binary = (Value - 4) >> 1;
    return ((Binary & 0x20) << 4) | ((Binary & 0x1f) << 3);
  }
  case ARM::fixup_arm_thumb_br:
    // Offset by 4 and don't encode the lower bit, which is always 0.
    return ((Value - 4) >> 1) & 0x7ff;
  case ARM::fixup_arm_thumb_bcc:
    // Offset by 4 and don't encode the lower bit, which is always 0.
    return ((Value - 4) >> 1) & 0xff;
  case ARM::fixup_arm_pcrel_10_unscaled: {
    Value = Value - 8; // ARM fixups offset by an additional word and don't
                       // need to adjust for the half-word ordering.
    bool isAdd = true;
    if ((int64_t)Value < 0) {
      Value = -Value;
      isAdd = false;
    }
    assert ((Value < 256) && "Out of range pc-relative fixup value!");
    return Value | (isAdd << 23);
  }
  case ARM::fixup_arm_pcrel_10:
    Value = Value - 4; // ARM fixups offset by an additional word and don't
                       // need to adjust for the half-word ordering.
    // Fall through.
  case ARM::fixup_t2_pcrel_10: {
    // Offset by 4, adjusted by two due to the half-word ordering of thumb.
    Value = Value - 4;
    bool isAdd = true;
    if ((int64_t)Value < 0) {
      Value = -Value;
      isAdd = false;
    }
    // These values don't encode the low two bits since they're always zero.
    Value >>= 2;
    assert ((Value < 256) && "Out of range pc-relative fixup value!");
    Value |= isAdd << 23;

    // Same addressing mode as fixup_arm_pcrel_10, but with 16-bit halfwords
    // swapped.
    if (Kind == ARM::fixup_t2_pcrel_10) {
      uint32_t swapped = (Value & 0xFFFF0000) >> 16;
      swapped |= (Value & 0x0000FFFF) << 16;
      return swapped;
    }

    return Value;
  }
  }
}

namespace {

// FIXME: This should be in a separate file.
// ELF is an ELF of course...
class ELFARMAsmBackend : public ARMAsmBackend {
public:
  uint8_t OSABI;
  ELFARMAsmBackend(const Target &T, const StringRef TT,
                   uint8_t _OSABI)
    : ARMAsmBackend(T, TT), OSABI(_OSABI) { }

  void ApplyFixup(const MCFixup &Fixup, char *Data, unsigned DataSize,
                  uint64_t Value) const;

  MCObjectWriter *createObjectWriter(raw_ostream &OS) const {
    return createELFObjectWriter(new ARMELFObjectWriter(OSABI), OS,
                              /*IsLittleEndian*/ true);
  }
};

// FIXME: Raise this to share code between Darwin and ELF.
void ELFARMAsmBackend::ApplyFixup(const MCFixup &Fixup, char *Data,
                                  unsigned DataSize, uint64_t Value) const {
  unsigned NumBytes = 4;        // FIXME: 2 for Thumb
  Value = adjustFixupValue(Fixup.getKind(), Value);
  if (!Value) return;           // Doesn't change encoding.

  unsigned Offset = Fixup.getOffset();

  // For each byte of the fragment that the fixup touches, mask in the bits from
  // the fixup value. The Value has been "split up" into the appropriate
  // bitfields above.
  for (unsigned i = 0; i != NumBytes; ++i)
    Data[Offset + i] |= uint8_t((Value >> (i * 8)) & 0xff);
}

// FIXME: This should be in a separate file.
class DarwinARMAsmBackend : public ARMAsmBackend {
public:
  const object::mach::CPUSubtypeARM Subtype;
  DarwinARMAsmBackend(const Target &T, const StringRef TT,
                      object::mach::CPUSubtypeARM st)
    : ARMAsmBackend(T, TT), Subtype(st) { }

  MCObjectWriter *createObjectWriter(raw_ostream &OS) const {
    return createARMMachObjectWriter(OS, /*Is64Bit=*/false,
                                     object::mach::CTM_ARM,
                                     Subtype);
  }

  void ApplyFixup(const MCFixup &Fixup, char *Data, unsigned DataSize,
                  uint64_t Value) const;

  virtual bool doesSectionRequireSymbols(const MCSection &Section) const {
    return false;
  }
};

/// getFixupKindNumBytes - The number of bytes the fixup may change.
static unsigned getFixupKindNumBytes(unsigned Kind) {
  switch (Kind) {
  default:
    llvm_unreachable("Unknown fixup kind!");

  case FK_Data_1:
  case ARM::fixup_arm_thumb_bcc:
  case ARM::fixup_arm_thumb_cp:
  case ARM::fixup_thumb_adr_pcrel_10:
    return 1;

  case FK_Data_2:
  case ARM::fixup_arm_thumb_br:
  case ARM::fixup_arm_thumb_cb:
    return 2;

  case ARM::fixup_arm_pcrel_10_unscaled:
  case ARM::fixup_arm_ldst_pcrel_12:
  case ARM::fixup_arm_pcrel_10:
  case ARM::fixup_arm_adr_pcrel_12:
  case ARM::fixup_arm_condbranch:
  case ARM::fixup_arm_uncondbranch:
    return 3;

  case FK_Data_4:
  case ARM::fixup_t2_ldst_pcrel_12:
  case ARM::fixup_t2_condbranch:
  case ARM::fixup_t2_uncondbranch:
  case ARM::fixup_t2_pcrel_10:
  case ARM::fixup_t2_adr_pcrel_12:
  case ARM::fixup_arm_thumb_bl:
  case ARM::fixup_arm_thumb_blx:
  case ARM::fixup_arm_movt_hi16:
  case ARM::fixup_arm_movw_lo16:
  case ARM::fixup_arm_movt_hi16_pcrel:
  case ARM::fixup_arm_movw_lo16_pcrel:
  case ARM::fixup_t2_movt_hi16:
  case ARM::fixup_t2_movw_lo16:
  case ARM::fixup_t2_movt_hi16_pcrel:
  case ARM::fixup_t2_movw_lo16_pcrel:
    return 4;
  }
}

void DarwinARMAsmBackend::ApplyFixup(const MCFixup &Fixup, char *Data,
                                     unsigned DataSize, uint64_t Value) const {
  unsigned NumBytes = getFixupKindNumBytes(Fixup.getKind());
  Value = adjustFixupValue(Fixup.getKind(), Value);
  if (!Value) return;           // Doesn't change encoding.

  unsigned Offset = Fixup.getOffset();
  assert(Offset + NumBytes <= DataSize && "Invalid fixup offset!");

  // For each byte of the fragment that the fixup touches, mask in the
  // bits from the fixup value.
  for (unsigned i = 0; i != NumBytes; ++i)
    Data[Offset + i] |= uint8_t((Value >> (i * 8)) & 0xff);
}

} // end anonymous namespace

MCAsmBackend *llvm::createARMAsmBackend(const Target &T, StringRef TT) {
  Triple TheTriple(TT);

  if (TheTriple.isOSDarwin()) {
    if (TheTriple.getArchName() == "armv4t" ||
        TheTriple.getArchName() == "thumbv4t")
      return new DarwinARMAsmBackend(T, TT, object::mach::CSARM_V4T);
    else if (TheTriple.getArchName() == "armv5e" ||
        TheTriple.getArchName() == "thumbv5e")
      return new DarwinARMAsmBackend(T, TT, object::mach::CSARM_V5TEJ);
    else if (TheTriple.getArchName() == "armv6" ||
        TheTriple.getArchName() == "thumbv6")
      return new DarwinARMAsmBackend(T, TT, object::mach::CSARM_V6);
    return new DarwinARMAsmBackend(T, TT, object::mach::CSARM_V7);
  }

  if (TheTriple.isOSWindows())
    assert(0 && "Windows not supported on ARM");

  uint8_t OSABI = MCELFObjectTargetWriter::getOSABI(Triple(TT).getOS());
  return new ELFARMAsmBackend(T, TT, OSABI);
}
