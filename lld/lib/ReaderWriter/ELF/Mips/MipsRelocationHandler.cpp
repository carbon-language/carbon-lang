//===- lib/ReaderWriter/ELF/Mips/MipsRelocationHandler.cpp ----------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MipsLinkingContext.h"
#include "MipsRelocationHandler.h"
#include "MipsTargetLayout.h"
#include "llvm/Support/Format.h"

using namespace lld;
using namespace elf;
using namespace llvm::ELF;
using namespace llvm::support;

namespace {
enum class CrossJumpMode {
  None,      // Not a jump or non-isa-cross jump
  ToRegular, // cross isa jump to regular symbol
  ToMicro    // cross isa jump to microMips symbol
};

typedef std::function<std::error_code(int64_t, bool)> OverflowChecker;

static std::error_code dummyCheck(int64_t, bool) {
  return std::error_code();
}

template <int BITS> static std::error_code signedCheck(int64_t res, bool) {
  if (llvm::isInt<BITS>(res))
    return std::error_code();
  return make_out_of_range_reloc_error();
}

template <int BITS>
static std::error_code gpDispCheck(int64_t res, bool isGpDisp) {
  if (!isGpDisp || llvm::isInt<BITS>(res))
    return std::error_code();
  return make_out_of_range_reloc_error();
}

struct MipsRelocationParams {
  uint8_t _size;  // Relocations's size in bytes
  uint64_t _mask; // Read/write mask of relocation
  uint8_t _shift; // Relocation's addendum left shift size
  bool _shuffle;  // Relocation's addendum/result needs to be shuffled
  OverflowChecker _overflow; // Check the relocation result
};

template <class ELFT> class RelocationHandler : public TargetRelocationHandler {
public:
  RelocationHandler(MipsLinkingContext &ctx, MipsTargetLayout<ELFT> &layout)
      : _ctx(ctx), _targetLayout(layout) {}

  std::error_code applyRelocation(ELFWriter &writer,
                                  llvm::FileOutputBuffer &buf,
                                  const AtomLayout &atom,
                                  const Reference &ref) const override;

private:
  MipsLinkingContext &_ctx;
  MipsTargetLayout<ELFT> &_targetLayout;
};
}

static MipsRelocationParams getRelocationParams(uint32_t rType) {
  switch (rType) {
  case R_MIPS_NONE:
    return {4, 0x0, 0, false, dummyCheck};
  case R_MIPS_64:
  case R_MIPS_SUB:
    return {8, 0xffffffffffffffffull, 0, false, dummyCheck};
  case R_MIPS_32:
  case R_MIPS_GPREL32:
  case R_MIPS_REL32:
  case R_MIPS_PC32:
  case R_MIPS_EH:
    return {4, 0xffffffff, 0, false, dummyCheck};
  case LLD_R_MIPS_32_HI16:
    return {4, 0xffff0000, 0, false, dummyCheck};
  case LLD_R_MIPS_64_HI16:
    return {8, 0xffffffffffff0000ull, 0, false, dummyCheck};
  case R_MIPS_26:
  case LLD_R_MIPS_GLOBAL_26:
    return {4, 0x3ffffff, 2, false, dummyCheck};
  case R_MIPS_PC16:
    return {4, 0xffff, 2, false, signedCheck<18>};
  case R_MIPS_PC18_S3:
    return {4, 0x3ffff, 3, false, signedCheck<21>};
  case R_MIPS_PC19_S2:
    return {4, 0x7ffff, 2, false, signedCheck<21>};
  case R_MIPS_PC21_S2:
    return {4, 0x1fffff, 2, false, signedCheck<23>};
  case R_MIPS_PC26_S2:
    return {4, 0x3ffffff, 2, false, signedCheck<28>};
  case R_MIPS_HI16:
    return {4, 0xffff, 0, false, gpDispCheck<16>};
  case R_MIPS_LO16:
    return {4, 0xffff, 0, false, dummyCheck};
  case R_MIPS_PCHI16:
  case R_MIPS_PCLO16:
  case R_MIPS_GOT16:
  case R_MIPS_CALL16:
  case R_MIPS_GOT_DISP:
  case R_MIPS_GOT_PAGE:
  case R_MIPS_GOT_OFST:
  case R_MIPS_GPREL16:
  case R_MIPS_TLS_GD:
  case R_MIPS_TLS_LDM:
  case R_MIPS_TLS_GOTTPREL:
    return {4, 0xffff, 0, false, signedCheck<16>};
  case R_MIPS_GOT_HI16:
  case R_MIPS_GOT_LO16:
  case R_MIPS_CALL_HI16:
  case R_MIPS_CALL_LO16:
  case R_MIPS_TLS_DTPREL_HI16:
  case R_MIPS_TLS_DTPREL_LO16:
  case R_MIPS_TLS_TPREL_HI16:
  case R_MIPS_TLS_TPREL_LO16:
    return {4, 0xffff, 0, false, dummyCheck};
  case R_MICROMIPS_GOT_HI16:
  case R_MICROMIPS_GOT_LO16:
  case R_MICROMIPS_CALL_HI16:
  case R_MICROMIPS_CALL_LO16:
  case R_MICROMIPS_TLS_DTPREL_HI16:
  case R_MICROMIPS_TLS_DTPREL_LO16:
  case R_MICROMIPS_TLS_TPREL_HI16:
  case R_MICROMIPS_TLS_TPREL_LO16:
    return {4, 0xffff, 0, true, dummyCheck};
  case R_MICROMIPS_26_S1:
  case LLD_R_MICROMIPS_GLOBAL_26_S1:
    return {4, 0x3ffffff, 1, true, dummyCheck};
  case R_MICROMIPS_HI16:
    return {4, 0xffff, 0, true, gpDispCheck<16>};
  case R_MICROMIPS_LO16:
    return {4, 0xffff, 0, true, dummyCheck};
  case R_MICROMIPS_PC16_S1:
    return {4, 0xffff, 1, true, signedCheck<17>};
  case R_MICROMIPS_PC7_S1:
    return {4, 0x7f, 1, false, signedCheck<8>};
  case R_MICROMIPS_PC10_S1:
    return {4, 0x3ff, 1, false, signedCheck<11>};
  case R_MICROMIPS_PC23_S2:
    return {4, 0x7fffff, 2, true, signedCheck<25>};
  case R_MICROMIPS_PC18_S3:
    return {4, 0x3ffff, 3, true, signedCheck<21>};
  case R_MICROMIPS_PC19_S2:
    return {4, 0x7ffff, 2, true, signedCheck<21>};
  case R_MICROMIPS_PC21_S2:
    return {4, 0x1fffff, 2, true, signedCheck<23>};
  case R_MICROMIPS_PC26_S2:
    return {4, 0x3ffffff, 2, true, signedCheck<28>};
  case R_MICROMIPS_GOT16:
  case R_MICROMIPS_CALL16:
  case R_MICROMIPS_TLS_GD:
  case R_MICROMIPS_TLS_LDM:
  case R_MICROMIPS_TLS_GOTTPREL:
  case R_MICROMIPS_GOT_DISP:
  case R_MICROMIPS_GOT_PAGE:
  case R_MICROMIPS_GOT_OFST:
    return {4, 0xffff, 0, true, signedCheck<16>};
  case R_MIPS_JALR:
    return {4, 0x0, 0, false, dummyCheck};
  case R_MICROMIPS_JALR:
    return {4, 0x0, 0, true, dummyCheck};
  case R_MIPS_JUMP_SLOT:
  case R_MIPS_COPY:
  case R_MIPS_TLS_DTPMOD32:
  case R_MIPS_TLS_DTPREL32:
  case R_MIPS_TLS_TPREL32:
    return {4, 0xffffffff, 0, false, dummyCheck};
  case R_MIPS_TLS_DTPMOD64:
  case R_MIPS_TLS_DTPREL64:
  case R_MIPS_TLS_TPREL64:
    return {8, 0xffffffffffffffffull, 0, false, dummyCheck};
  case LLD_R_MIPS_GLOBAL_GOT:
  case LLD_R_MIPS_STO_PLT:
    // Do nothing.
    return {4, 0x0, 0, false, dummyCheck};
  default:
    llvm_unreachable("Unknown relocation");
  }
}

static int64_t getHi16(int64_t value) {
  return ((value + 0x8000) >> 16) & 0xffff;
}

static int64_t maskLow16(int64_t value) {
  return (value + 0x8000) & ~0xffff;
}

/// \brief R_MIPS_32
/// local/external: word32 S + A (truncate)
static int32_t reloc32(uint64_t S, int64_t A) { return S + A; }

/// \brief R_MIPS_64
/// local/external: word64 S + A (truncate)
static int64_t reloc64(uint64_t S, int64_t A) { return S + A; }

/// \brief R_MIPS_SUB
/// local/external: word64 S - A (truncate)
static int64_t relocSub(uint64_t S, int64_t A) { return S - A; }

/// \brief R_MIPS_PC32
/// local/external: word32 S + A - P (truncate)
static int32_t relocpc32(uint64_t P, uint64_t S, int64_t A) {
  return S + A - P;
}

/// \brief R_MIPS_26, R_MICROMIPS_26_S1
/// local   : ((A | ((P + 4) & 0x3F000000)) + S) >> 2
static int32_t reloc26loc(uint64_t P, uint64_t S, int32_t A, uint32_t shift) {
  return (A | ((P + 4) & (0xfc000000 << shift))) + S;
}

/// \brief LLD_R_MIPS_GLOBAL_26, LLD_R_MICROMIPS_GLOBAL_26_S1
/// external: (sign-extend(A) + S) >> 2
static int32_t reloc26ext(uint64_t S, int32_t A, uint32_t shift) {
  A = shift == 1 ? llvm::SignExtend32<27>(A) : llvm::SignExtend32<28>(A);
  return A + S;
}

/// \brief R_MIPS_HI16, R_MIPS_TLS_DTPREL_HI16, R_MIPS_TLS_TPREL_HI16,
/// R_MICROMIPS_HI16, R_MICROMIPS_TLS_DTPREL_HI16, R_MICROMIPS_TLS_TPREL_HI16
/// local/external: hi16 (AHL + S) - (short)(AHL + S) (truncate)
/// _gp_disp      : hi16 (AHL + GP - P) - (short)(AHL + GP - P) (verify)
static int32_t relocHi16(uint64_t P, uint64_t S, int64_t AHL, bool isGPDisp) {
  return getHi16(isGPDisp ? AHL + S - P : AHL + S);
}

/// \brief R_MIPS_PCHI16
/// local/external: hi16 (S + AHL - P)
static int32_t relocPcHi16(uint64_t P, uint64_t S, int64_t AHL) {
  return getHi16(S + AHL - P);
}

/// \brief R_MIPS_LO16, R_MIPS_TLS_DTPREL_LO16, R_MIPS_TLS_TPREL_LO16,
/// R_MICROMIPS_LO16, R_MICROMIPS_TLS_DTPREL_LO16, R_MICROMIPS_TLS_TPREL_LO16
/// local/external: lo16 AHL + S (truncate)
/// _gp_disp      : lo16 AHL + GP - P + 4 (verify)
static int32_t relocLo16(uint64_t P, uint64_t S, int64_t AHL, bool isGPDisp,
                         bool micro) {
  return isGPDisp ? AHL + S - P + (micro ? 3 : 4) : AHL + S;
}

/// \brief R_MIPS_PCLO16
/// local/external: lo16 (S + AHL - P)
static int32_t relocPcLo16(uint64_t P, uint64_t S, int64_t AHL) {
  AHL = llvm::SignExtend32<16>(AHL);
  return S + AHL - P;
}

/// \brief R_MIPS_GOT16, R_MIPS_CALL16, R_MICROMIPS_GOT16, R_MICROMIPS_CALL16
/// rel16 G (verify)
static int64_t relocGOT(uint64_t S, uint64_t GP) {
  return S - GP;
}

/// \brief R_MIPS_GOT_LO16, R_MIPS_CALL_LO16
/// R_MICROMIPS_GOT_LO16, R_MICROMIPS_CALL_LO16
/// rel16 G (truncate)
static int64_t relocGOTLo16(uint64_t S, uint64_t GP) {
  return S - GP;
}

/// \brief R_MIPS_GOT_HI16, R_MIPS_CALL_HI16,
/// R_MICROMIPS_GOT_HI16, R_MICROMIPS_CALL_HI16
/// rel16 %high(G) (truncate)
static int64_t relocGOTHi16(uint64_t S, uint64_t GP) {
  return getHi16(S - GP);
}

/// R_MIPS_GOT_OFST, R_MICROMIPS_GOT_OFST
/// rel16 offset of (S+A) from the page pointer (verify)
static int32_t relocGOTOfst(uint64_t S, int64_t A) {
  int64_t page = maskLow16(S + A);
  return S + A - page;
}

/// \brief R_MIPS_GPREL16
/// local: sign-extend(A) + S + GP0 - GP
/// external: sign-extend(A) + S - GP
static int64_t relocGPRel16(uint64_t S, int64_t A, uint64_t GP) {
  // We added GP0 to addendum for a local symbol during a Relocation pass.
  return A + S - GP;
}

/// \brief R_MIPS_GPREL32
/// local: rel32 A + S + GP0 - GP (truncate)
static int64_t relocGPRel32(uint64_t S, int64_t A, uint64_t GP) {
  // We added GP0 to addendum for a local symbol during a Relocation pass.
  return A + S - GP;
}

/// \brief R_MIPS_PC16
/// local/external: (S + A - P) >> 2
static ErrorOr<int64_t> relocPc16(uint64_t P, uint64_t S, int64_t A) {
  A = llvm::SignExtend32<18>(A);
  if ((S + A) & 3)
    return make_unaligned_range_reloc_error();
  return S + A - P;
}

/// \brief R_MIPS_PC18_S3, R_MICROMIPS_PC18_S3
/// local/external: (S + A - P) >> 3 (P with cleared 3 less significant bits)
static ErrorOr<int64_t> relocPc18(uint64_t P, uint64_t S, int64_t A) {
  A = llvm::SignExtend32<21>(A);
  if ((S + A) & 6)
    return make_unaligned_range_reloc_error();
  return S + A - ((P | 7) ^ 7);
}

/// \brief R_MIPS_PC19_S2, R_MICROMIPS_PC19_S2
/// local/external: (S + A - P) >> 2
static ErrorOr<int64_t> relocPc19(uint64_t P, uint64_t S, int64_t A) {
  A = llvm::SignExtend32<21>(A);
  if ((S + A) & 2)
    return make_unaligned_range_reloc_error();
  return S + A - P;
}

/// \brief R_MIPS_PC21_S2, R_MICROMIPS_PC21_S2
/// local/external: (S + A - P) >> 2
static ErrorOr<int64_t> relocPc21(uint64_t P, uint64_t S, int64_t A) {
  A = llvm::SignExtend32<23>(A);
  if ((S + A) & 2)
    return make_unaligned_range_reloc_error();
  return S + A - P;
}

/// \brief R_MIPS_PC26_S2, R_MICROMIPS_PC26_S2
/// local/external: (S + A - P) >> 2
static ErrorOr<int64_t> relocPc26(uint64_t P, uint64_t S, int64_t A) {
  A = llvm::SignExtend32<28>(A);
  if ((S + A) & 2)
    return make_unaligned_range_reloc_error();
  return S + A - P;
}

/// \brief R_MICROMIPS_PC7_S1
static int32_t relocPc7(uint64_t P, uint64_t S, int64_t A) {
  A = llvm::SignExtend32<8>(A);
  return S + A - P;
}

/// \brief R_MICROMIPS_PC10_S1
static int32_t relocPc10(uint64_t P, uint64_t S, int64_t A) {
  A = llvm::SignExtend32<11>(A);
  return S + A - P;
}

/// \brief R_MICROMIPS_PC16_S1
static int32_t relocPc16Micro(uint64_t P, uint64_t S, int64_t A) {
  A = llvm::SignExtend32<17>(A);
  return S + A - P;
}

/// \brief R_MICROMIPS_PC23_S2
static uint32_t relocPc23(uint64_t P, uint64_t S, int64_t A) {
  A = llvm::SignExtend32<25>(A);
  return S + A - P;
}

/// \brief LLD_R_MIPS_32_HI16, LLD_R_MIPS_64_HI16
static int64_t relocMaskLow16(uint64_t S, int64_t A) {
  return maskLow16(S + A);
}

/// R_MIPS_TLS_TPREL32, R_MIPS_TLS_TPREL64
static int64_t relocTlsTpRel(uint64_t S, int64_t A) {
  return S + A - 0x7000;
}

/// R_MIPS_TLS_DTPREL32, R_MIPS_TLS_DTPREL64
static int64_t relocTlsDTpRel(uint64_t S, int64_t A) {
  return S + A - 0x8000;
}

static int64_t relocRel32(int64_t A) {
  // If output relocation format is REL and the input one is RELA, the only
  // method to transfer the relocation addend from the input relocation
  // to the output dynamic relocation is to save this addend to the location
  // modified by R_MIPS_REL32.
  return A;
}

static std::error_code adjustJumpOpCode(uint64_t &ins, uint64_t tgt,
                                        CrossJumpMode mode) {
  if (mode == CrossJumpMode::None)
    return std::error_code();

  bool toMicro = mode == CrossJumpMode::ToMicro;
  uint32_t opNative = toMicro ? 0x03 : 0x3d;
  uint32_t opCross = toMicro ? 0x1d : 0x3c;

  if ((tgt & 1) != toMicro)
    return make_dynamic_error_code("Incorrect bit 0 for the jalx target");

  if (tgt & 2)
    return make_dynamic_error_code(Twine("The jalx target 0x") +
                                   Twine::utohexstr(tgt) +
                                   " is not word-aligned");
  uint8_t op = ins >> 26;
  if (op != opNative && op != opCross)
    return make_dynamic_error_code(Twine("Unsupported jump opcode (0x") +
                                   Twine::utohexstr(op) +
                                   ") for ISA modes cross call");

  ins = (ins & ~(0x3f << 26)) | (opCross << 26);
  return std::error_code();
}

static bool isMicroMipsAtom(const Atom *a) {
  if (const auto *da = dyn_cast<DefinedAtom>(a))
    return da->codeModel() == DefinedAtom::codeMipsMicro ||
           da->codeModel() == DefinedAtom::codeMipsMicroPIC;
  return false;
}

static CrossJumpMode getCrossJumpMode(const Reference &ref) {
  if (!isa<DefinedAtom>(ref.target()))
    return CrossJumpMode::None;
  bool isTgtMicro = isMicroMipsAtom(ref.target());
  switch (ref.kindValue()) {
  case R_MIPS_26:
  case LLD_R_MIPS_GLOBAL_26:
    return isTgtMicro ? CrossJumpMode::ToMicro : CrossJumpMode::None;
  case R_MICROMIPS_26_S1:
  case LLD_R_MICROMIPS_GLOBAL_26_S1:
    return isTgtMicro ? CrossJumpMode::None : CrossJumpMode::ToRegular;
  default:
    return CrossJumpMode::None;
  }
}

static uint32_t microShuffle(uint32_t ins) {
  return ((ins & 0xffff) << 16) | ((ins & 0xffff0000) >> 16);
}

static ErrorOr<int64_t> calculateRelocation(Reference::KindValue kind,
                                            Reference::Addend addend,
                                            uint64_t tgtAddr, uint64_t relAddr,
                                            uint64_t gpAddr, bool isGP,
                                            bool isCrossJump, bool isDynamic) {
  switch (kind) {
  case R_MIPS_NONE:
    return 0;
  case R_MIPS_32:
    return reloc32(tgtAddr, addend);
  case R_MIPS_64:
    return reloc64(tgtAddr, addend);
  case R_MIPS_SUB:
    return relocSub(tgtAddr, addend);
  case R_MIPS_26:
    return reloc26loc(relAddr, tgtAddr, addend, 2);
  case R_MICROMIPS_26_S1:
    return reloc26loc(relAddr, tgtAddr, addend, isCrossJump ? 2 : 1);
  case R_MIPS_HI16:
  case R_MICROMIPS_HI16:
    return relocHi16(relAddr, tgtAddr, addend, isGP);
  case R_MIPS_PCHI16:
    return relocPcHi16(relAddr, tgtAddr, addend);
  case R_MIPS_LO16:
    return relocLo16(relAddr, tgtAddr, addend, isGP, false);
  case R_MIPS_PCLO16:
    return relocPcLo16(relAddr, tgtAddr, addend);
  case R_MICROMIPS_LO16:
    return relocLo16(relAddr, tgtAddr, addend, isGP, true);
  case R_MIPS_GOT_LO16:
  case R_MIPS_CALL_LO16:
  case R_MICROMIPS_GOT_LO16:
  case R_MICROMIPS_CALL_LO16:
    return relocGOTLo16(tgtAddr, gpAddr);
  case R_MIPS_GOT_HI16:
  case R_MIPS_CALL_HI16:
  case R_MICROMIPS_GOT_HI16:
  case R_MICROMIPS_CALL_HI16:
    return relocGOTHi16(tgtAddr, gpAddr);
  case R_MIPS_EH:
  case R_MIPS_GOT16:
  case R_MIPS_CALL16:
  case R_MIPS_GOT_DISP:
  case R_MIPS_GOT_PAGE:
  case R_MICROMIPS_GOT_DISP:
  case R_MICROMIPS_GOT_PAGE:
  case R_MICROMIPS_GOT16:
  case R_MICROMIPS_CALL16:
  case R_MIPS_TLS_GD:
  case R_MIPS_TLS_LDM:
  case R_MIPS_TLS_GOTTPREL:
  case R_MICROMIPS_TLS_GD:
  case R_MICROMIPS_TLS_LDM:
  case R_MICROMIPS_TLS_GOTTPREL:
    return relocGOT(tgtAddr, gpAddr);
  case R_MIPS_GOT_OFST:
  case R_MICROMIPS_GOT_OFST:
    return relocGOTOfst(tgtAddr, addend);
  case R_MIPS_PC16:
    return relocPc16(relAddr, tgtAddr, addend);
  case R_MIPS_PC18_S3:
  case R_MICROMIPS_PC18_S3:
    return relocPc18(relAddr, tgtAddr, addend);
  case R_MIPS_PC19_S2:
  case R_MICROMIPS_PC19_S2:
    return relocPc19(relAddr, tgtAddr, addend);
  case R_MIPS_PC21_S2:
  case R_MICROMIPS_PC21_S2:
    return relocPc21(relAddr, tgtAddr, addend);
  case R_MIPS_PC26_S2:
  case R_MICROMIPS_PC26_S2:
    return relocPc26(relAddr, tgtAddr, addend);
  case R_MICROMIPS_PC7_S1:
    return relocPc7(relAddr, tgtAddr, addend);
  case R_MICROMIPS_PC10_S1:
    return relocPc10(relAddr, tgtAddr, addend);
  case R_MICROMIPS_PC16_S1:
    return relocPc16Micro(relAddr, tgtAddr, addend);
  case R_MICROMIPS_PC23_S2:
    return relocPc23(relAddr, tgtAddr, addend);
  case R_MIPS_TLS_DTPREL_HI16:
  case R_MIPS_TLS_TPREL_HI16:
  case R_MICROMIPS_TLS_DTPREL_HI16:
  case R_MICROMIPS_TLS_TPREL_HI16:
    return relocHi16(0, tgtAddr, addend, false);
  case R_MIPS_TLS_DTPREL_LO16:
  case R_MIPS_TLS_TPREL_LO16:
    return relocLo16(0, tgtAddr, addend, false, false);
  case R_MICROMIPS_TLS_DTPREL_LO16:
  case R_MICROMIPS_TLS_TPREL_LO16:
    return relocLo16(0, tgtAddr, addend, false, true);
  case R_MIPS_GPREL16:
    return relocGPRel16(tgtAddr, addend, gpAddr);
  case R_MIPS_GPREL32:
    return relocGPRel32(tgtAddr, addend, gpAddr);
  case R_MIPS_JALR:
  case R_MICROMIPS_JALR:
    // We do not do JALR optimization now.
    return 0;
  case R_MIPS_REL32:
    return relocRel32(addend);
  case R_MIPS_JUMP_SLOT:
  case R_MIPS_COPY:
    // Ignore runtime relocations.
    return 0;
  case R_MIPS_TLS_DTPMOD32:
  case R_MIPS_TLS_DTPMOD64:
    return isDynamic ? 0 : 1;
  case R_MIPS_TLS_DTPREL32:
  case R_MIPS_TLS_DTPREL64:
    if (isDynamic)
      return 0;
    return relocTlsDTpRel(tgtAddr, addend);
  case R_MIPS_TLS_TPREL32:
  case R_MIPS_TLS_TPREL64:
    if (isDynamic)
      return 0;
    return relocTlsTpRel(tgtAddr, addend);
  case R_MIPS_PC32:
    return relocpc32(relAddr, tgtAddr, addend);
  case LLD_R_MIPS_32_HI16:
  case LLD_R_MIPS_64_HI16:
    return relocMaskLow16(tgtAddr, addend);
  case LLD_R_MIPS_GLOBAL_26:
    return reloc26ext(tgtAddr, addend, 2);
  case LLD_R_MICROMIPS_GLOBAL_26_S1:
    return reloc26ext(tgtAddr, addend, isCrossJump ? 2 : 1);
  case LLD_R_MIPS_STO_PLT:
  case LLD_R_MIPS_GLOBAL_GOT:
    // Do nothing.
    return 0;
  default:
    return make_unhandled_reloc_error();
  }
}

static uint64_t relocRead(const MipsRelocationParams &params,
                          const uint8_t *loc) {
  uint64_t data;
  switch (params._size) {
  case 4:
    data = endian::read32le(loc);
    break;
  case 8:
    data = endian::read64le(loc);
    break;
  default:
    llvm_unreachable("Unexpected size");
  }
  if (params._shuffle)
    data = microShuffle(data);
  return data;
}

template <class ELFT>
static void relocWrite(uint64_t data, const MipsRelocationParams &params,
                       uint8_t *loc) {
  if (params._shuffle)
    data = microShuffle(data);
  switch (params._size) {
  case 4:
    endian::write<uint32_t, ELFT::TargetEndianness, unaligned>(loc, data);
    break;
  case 8:
    endian::write<uint64_t, ELFT::TargetEndianness, unaligned>(loc, data);
    break;
  default:
    llvm_unreachable("Unexpected size");
  }
}

static uint32_t getRelKind(const Reference &ref, size_t num) {
  if (num == 0)
    return ref.kindValue();
  if (num > 2)
    return R_MIPS_NONE;
  return (ref.tag() >> (8 * (num - 1))) & 0xff;
}

static uint8_t getRelShift(Reference::KindValue kind,
                           const MipsRelocationParams &params,
                           bool isCrossJump) {
  uint8_t shift = params._shift;
  if (isCrossJump &&
      (kind == R_MICROMIPS_26_S1 || kind == LLD_R_MICROMIPS_GLOBAL_26_S1))
    return 2;
  return shift;
}

template <class ELFT>
std::error_code RelocationHandler<ELFT>::applyRelocation(
    ELFWriter &writer, llvm::FileOutputBuffer &buf, const AtomLayout &atom,
    const Reference &ref) const {
  if (ref.kindNamespace() != Reference::KindNamespace::ELF)
    return std::error_code();
  assert(ref.kindArch() == Reference::KindArch::Mips);

  uint64_t gpAddr = _targetLayout.getGPAddr();
  bool isGpDisp = ref.target()->name() == "_gp_disp";

  uint8_t *atomContent = buf.getBufferStart() + atom._fileOffset;
  uint8_t *location = atomContent + ref.offsetInAtom();
  uint64_t tgtAddr = writer.addressOfAtom(ref.target());
  uint64_t relAddr = atom._virtualAddr + ref.offsetInAtom();

  if (isMicroMipsAtom(ref.target()))
    tgtAddr |= 1;

  CrossJumpMode jumpMode = getCrossJumpMode(ref);
  bool isCrossJump = jumpMode != CrossJumpMode::None;

  uint64_t sym = tgtAddr;
  ErrorOr<int64_t> res = ref.addend();
  Reference::KindValue lastRel = R_MIPS_NONE;

  for (size_t relNum = 0; relNum < 3; ++relNum) {
    Reference::KindValue kind = getRelKind(ref, relNum);
    if (kind == R_MIPS_NONE)
      break;
    auto params = getRelocationParams(kind);
    res = calculateRelocation(kind, *res, sym, relAddr, gpAddr, isGpDisp,
                              isCrossJump, _ctx.isDynamic());
    if (auto ec = res.getError())
      return ec;
    // Check result for the last relocation only.
    if (getRelKind(ref, relNum + 1) == R_MIPS_NONE) {
      if (auto ec = params._overflow(*res, isGpDisp))
        return ec;
    }
    res = *res >> getRelShift(kind, params, isCrossJump);
    // FIXME (simon): Handle r_ssym value.
    sym = 0;
    isGpDisp = false;
    isCrossJump = false;
    lastRel = kind;
  }

  auto params = getRelocationParams(lastRel);
  uint64_t ins = relocRead(params, location);
  if (auto ec = adjustJumpOpCode(ins, tgtAddr, jumpMode))
    return ec;

  ins = (ins & ~params._mask) | (*res & params._mask);
  relocWrite<ELFT>(ins, params, location);

  return std::error_code();
}

namespace lld {
namespace elf {

template <>
std::unique_ptr<TargetRelocationHandler>
createMipsRelocationHandler<ELF32LE>(MipsLinkingContext &ctx,
                                     MipsTargetLayout<ELF32LE> &layout) {
  return llvm::make_unique<RelocationHandler<ELF32LE>>(ctx, layout);
}

template <>
std::unique_ptr<TargetRelocationHandler>
createMipsRelocationHandler<ELF64LE>(MipsLinkingContext &ctx,
                                     MipsTargetLayout<ELF64LE> &layout) {
  return llvm::make_unique<RelocationHandler<ELF64LE>>(ctx, layout);
}

Reference::Addend readMipsRelocAddend(Reference::KindValue kind,
                                      const uint8_t *content) {
  auto params = getRelocationParams(kind);
  uint64_t ins = relocRead(params, content);
  int64_t res = (ins & params._mask) << params._shift;
  switch (kind) {
  case R_MIPS_GPREL16:
    return llvm::SignExtend32<16>(res);
  default:
    // Nothing to do
    break;
  }
  return res;
}

} // elf
} // lld
