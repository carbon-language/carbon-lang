//===- lib/ReaderWriter/ELF/Mips/MipsRelocationHandler.cpp ----------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MipsTargetHandler.h"
#include "MipsLinkingContext.h"
#include "MipsRelocationHandler.h"

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

struct MipsRelocationParams {
  uint8_t _size;  // Relocations's size in bytes
  uint64_t _mask; // Read/write mask of relocation
  uint8_t _shift; // Relocation's addendum left shift size
  bool _shuffle;  // Relocation's addendum/result needs to be shuffled
};

template <class ELFT> class RelocationHandler : public MipsRelocationHandler {
public:
  RelocationHandler(MipsLinkingContext &ctx) : _ctx(ctx) {}

  std::error_code applyRelocation(ELFWriter &writer,
                                  llvm::FileOutputBuffer &buf,
                                  const lld::AtomLayout &atom,
                                  const Reference &ref) const override;

  Reference::Addend readAddend(Reference::KindValue kind,
                               const uint8_t *content) const override;

private:
  MipsLinkingContext &_ctx;
};
}

static MipsRelocationParams getRelocationParams(uint32_t rType) {
  switch (rType) {
  case R_MIPS_NONE:
    return {4, 0x0, 0, false};
  case R_MIPS_64:
  case R_MIPS_SUB:
    return {8, 0xffffffffffffffffull, 0, false};
  case R_MIPS_32:
  case R_MIPS_GPREL32:
  case R_MIPS_PC32:
    return {4, 0xffffffff, 0, false};
  case LLD_R_MIPS_32_HI16:
    return {4, 0xffff0000, 0, false};
  case LLD_R_MIPS_64_HI16:
    return {8, 0xffffffffffff0000ull, 0, false};
  case R_MIPS_26:
  case LLD_R_MIPS_GLOBAL_26:
    return {4, 0x3ffffff, 2, false};
  case R_MIPS_PC21_S2:
    return {4, 0x1fffff, 2, false};
  case R_MIPS_HI16:
  case R_MIPS_LO16:
  case R_MIPS_GPREL16:
  case R_MIPS_GOT16:
  case R_MIPS_GOT_DISP:
  case R_MIPS_GOT_PAGE:
  case R_MIPS_GOT_OFST:
  case R_MIPS_TLS_DTPREL_HI16:
  case R_MIPS_TLS_DTPREL_LO16:
  case R_MIPS_TLS_TPREL_HI16:
  case R_MIPS_TLS_TPREL_LO16:
  case LLD_R_MIPS_HI16:
  case LLD_R_MIPS_LO16:
    return {4, 0xffff, 0, false};
  case R_MICROMIPS_TLS_DTPREL_HI16:
  case R_MICROMIPS_TLS_DTPREL_LO16:
  case R_MICROMIPS_TLS_TPREL_HI16:
  case R_MICROMIPS_TLS_TPREL_LO16:
    return {4, 0xffff, 0, true};
  case R_MICROMIPS_26_S1:
  case LLD_R_MICROMIPS_GLOBAL_26_S1:
    return {4, 0x3ffffff, 1, true};
  case R_MICROMIPS_HI16:
  case R_MICROMIPS_LO16:
  case R_MICROMIPS_GOT16:
    return {4, 0xffff, 0, true};
  case R_MICROMIPS_PC16_S1:
    return {4, 0xffff, 1, true};
  case R_MICROMIPS_PC7_S1:
    return {4, 0x7f, 1, false};
  case R_MICROMIPS_PC10_S1:
    return {4, 0x3ff, 1, false};
  case R_MICROMIPS_PC23_S2:
    return {4, 0x7fffff, 2, true};
  case R_MIPS_CALL16:
  case R_MIPS_TLS_GD:
  case R_MIPS_TLS_LDM:
  case R_MIPS_TLS_GOTTPREL:
    return {4, 0xffff, 0, false};
  case R_MICROMIPS_CALL16:
  case R_MICROMIPS_TLS_GD:
  case R_MICROMIPS_TLS_LDM:
  case R_MICROMIPS_TLS_GOTTPREL:
    return {4, 0xffff, 0, true};
  case R_MIPS_JALR:
    return {4, 0x0, 0, false};
  case R_MICROMIPS_JALR:
    return {4, 0x0, 0, true};
  case R_MIPS_REL32:
  case R_MIPS_JUMP_SLOT:
  case R_MIPS_COPY:
  case R_MIPS_TLS_DTPMOD32:
  case R_MIPS_TLS_DTPREL32:
  case R_MIPS_TLS_TPREL32:
    // Ignore runtime relocations.
    return {4, 0x0, 0, false};
  case R_MIPS_TLS_DTPMOD64:
  case R_MIPS_TLS_DTPREL64:
  case R_MIPS_TLS_TPREL64:
    return {8, 0x0, 0, false};
  case LLD_R_MIPS_GLOBAL_GOT:
  case LLD_R_MIPS_STO_PLT:
    // Do nothing.
    return {4, 0x0, 0, false};
  default:
    llvm_unreachable("Unknown relocation");
  }
}

/// \brief R_MIPS_32
/// local/external: word32 S + A (truncate)
static uint32_t reloc32(uint64_t S, int64_t A) { return S + A; }

/// \brief R_MIPS_64
/// local/external: word64 S + A (truncate)
static uint64_t reloc64(uint64_t S, int64_t A) { return S + A; }

/// \brief R_MIPS_SUB
/// local/external: word64 S - A (truncate)
static uint64_t relocSub(uint64_t S, int64_t A) { return S - A; }

/// \brief R_MIPS_PC32
/// local/external: word32 S + A i- P (truncate)
static uint32_t relocpc32(uint64_t P, uint64_t S, int64_t A) {
  return S + A - P;
}

/// \brief R_MIPS_26, R_MICROMIPS_26_S1
/// local   : ((A | ((P + 4) & 0x3F000000)) + S) >> 2
static uint32_t reloc26loc(uint64_t P, uint64_t S, int32_t A, uint32_t shift) {
  uint32_t result = (A | ((P + 4) & (0xfc000000 << shift))) + S;
  return result >> shift;
}

/// \brief LLD_R_MIPS_GLOBAL_26, LLD_R_MICROMIPS_GLOBAL_26_S1
/// external: (sign-extend(A) + S) >> 2
static uint32_t reloc26ext(uint64_t S, int32_t A, uint32_t shift) {
  int32_t result =
      shift == 1 ? llvm::SignExtend32<27>(A) : llvm::SignExtend32<28>(A);
  return (result + S) >> shift;
}

/// \brief R_MIPS_HI16, R_MIPS_TLS_DTPREL_HI16, R_MIPS_TLS_TPREL_HI16,
/// R_MICROMIPS_HI16, R_MICROMIPS_TLS_DTPREL_HI16, R_MICROMIPS_TLS_TPREL_HI16,
/// LLD_R_MIPS_HI16
/// local/external: hi16 (AHL + S) - (short)(AHL + S) (truncate)
/// _gp_disp      : hi16 (AHL + GP - P) - (short)(AHL + GP - P) (verify)
static uint32_t relocHi16(uint64_t P, uint64_t S, int64_t AHL, bool isGPDisp) {
  int32_t result = isGPDisp ? AHL + S - P : AHL + S;
  return (result + 0x8000) >> 16;
}

/// \brief R_MIPS_LO16, R_MIPS_TLS_DTPREL_LO16, R_MIPS_TLS_TPREL_LO16,
/// R_MICROMIPS_LO16, R_MICROMIPS_TLS_DTPREL_LO16, R_MICROMIPS_TLS_TPREL_LO16,
/// LLD_R_MIPS_LO16
/// local/external: lo16 AHL + S (truncate)
/// _gp_disp      : lo16 AHL + GP - P + 4 (verify)
static uint32_t relocLo16(uint64_t P, uint64_t S, int64_t AHL, bool isGPDisp,
                          bool micro) {
  int32_t result = isGPDisp ? AHL + S - P + (micro ? 3 : 4) : AHL + S;
  return result;
}

/// \brief R_MIPS_GOT16, R_MIPS_CALL16, R_MICROMIPS_GOT16, R_MICROMIPS_CALL16
/// rel16 G (verify)
static uint64_t relocGOT(uint64_t S, uint64_t GP) {
  int64_t G = (int64_t)(S - GP);
  return G;
}

/// R_MIPS_GOT_OFST
/// rel16 offset of (S+A) from the page pointer (verify)
static uint32_t relocGOTOfst(uint64_t S, int64_t A) {
  uint64_t page = (S + A + 0x8000) & ~0xffff;
  return S + A - page;
}

/// \brief R_MIPS_GPREL16
/// local: sign-extend(A) + S + GP0 - GP
/// external: sign-extend(A) + S - GP
static uint32_t relocGPRel16(uint64_t S, int64_t A, uint64_t GP) {
  // We added GP0 to addendum for a local symbol during a Relocation pass.
  int32_t result = llvm::SignExtend32<16>(A) + S - GP;
  return result;
}

/// \brief R_MIPS_GPREL32
/// local: rel32 A + S + GP0 - GP (truncate)
static uint32_t relocGPRel32(uint64_t S, int64_t A, uint64_t GP) {
  // We added GP0 to addendum for a local symbol during a Relocation pass.
  int32_t result = A + S - GP;
  return result;
}

/// \brief R_MIPS_PC21_S2
static uint32_t relocPc21(uint64_t P, uint64_t S, int64_t A) {
  A = llvm::SignExtend32<23>(A);
  // FIXME (simon): Check that S + A has 4-byte alignment
  int32_t result = S + A - P;
  return result >> 2;
}

/// \brief R_MICROMIPS_PC7_S1
static uint32_t relocPc7(uint64_t P, uint64_t S, int64_t A) {
  A = llvm::SignExtend32<8>(A);
  int32_t result = S + A - P;
  return result >> 1;
}

/// \brief R_MICROMIPS_PC10_S1
static uint32_t relocPc10(uint64_t P, uint64_t S, int64_t A) {
  A = llvm::SignExtend32<11>(A);
  int32_t result = S + A - P;
  return result >> 1;
}

/// \brief R_MICROMIPS_PC16_S1
static uint32_t relocPc16(uint64_t P, uint64_t S, int64_t A) {
  A = llvm::SignExtend32<17>(A);
  int32_t result = S + A - P;
  return result >> 1;
}

/// \brief R_MICROMIPS_PC23_S2
static uint32_t relocPc23(uint64_t P, uint64_t S, int64_t A) {
  A = llvm::SignExtend32<25>(A);
  int32_t result = S + A - P;

  // Check addiupc 16MB range.
  if (result + 0x1000000 >= 0x2000000)
    llvm::errs() << "The addiupc instruction immediate "
                 << llvm::format_hex(result, 10) << " is out of range.\n";

  return result >> 2;
}

/// \brief LLD_R_MIPS_32_HI16, LLD_R_MIPS_64_HI16
static uint64_t relocMaskLow16(uint64_t S, int64_t A) {
  return S + A + 0x8000;
}

static std::error_code adjustJumpOpCode(uint64_t &ins, uint64_t tgt,
                                        CrossJumpMode mode) {
  if (mode == CrossJumpMode::None)
    return std::error_code();

  bool toMicro = mode == CrossJumpMode::ToMicro;
  uint32_t opNative = toMicro ? 0x03 : 0x3d;
  uint32_t opCross = toMicro ? 0x1d : 0x3c;

  if ((tgt & 1) != toMicro)
    return make_dynamic_error_code(
        Twine("Incorrect bit 0 for the jalx target"));

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

static ErrorOr<uint64_t> calculateRelocation(const Reference &ref,
                                             uint64_t tgtAddr, uint64_t relAddr,
                                             uint64_t gpAddr, bool isGP) {
  bool isCrossJump = getCrossJumpMode(ref) != CrossJumpMode::None;
  switch (ref.kindValue()) {
  case R_MIPS_NONE:
    return 0;
  case R_MIPS_32:
    return reloc32(tgtAddr, ref.addend());
  case R_MIPS_64:
    return reloc64(tgtAddr, ref.addend());
  case R_MIPS_SUB:
    return relocSub(tgtAddr, ref.addend());
  case R_MIPS_26:
    return reloc26loc(relAddr, tgtAddr, ref.addend(), 2);
  case R_MICROMIPS_26_S1:
    return reloc26loc(relAddr, tgtAddr, ref.addend(), isCrossJump ? 2 : 1);
  case R_MIPS_HI16:
  case R_MICROMIPS_HI16:
    return relocHi16(relAddr, tgtAddr, ref.addend(), isGP);
  case R_MIPS_LO16:
    return relocLo16(relAddr, tgtAddr, ref.addend(), isGP, false);
  case R_MICROMIPS_LO16:
    return relocLo16(relAddr, tgtAddr, ref.addend(), isGP, true);
  case R_MIPS_GOT16:
  case R_MIPS_CALL16:
  case R_MIPS_GOT_DISP:
  case R_MIPS_GOT_PAGE:
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
    return relocGOTOfst(tgtAddr, ref.addend());
  case R_MIPS_PC21_S2:
    return relocPc21(relAddr, tgtAddr, ref.addend());
  case R_MICROMIPS_PC7_S1:
    return relocPc7(relAddr, tgtAddr, ref.addend());
  case R_MICROMIPS_PC10_S1:
    return relocPc10(relAddr, tgtAddr, ref.addend());
  case R_MICROMIPS_PC16_S1:
    return relocPc16(relAddr, tgtAddr, ref.addend());
  case R_MICROMIPS_PC23_S2:
    return relocPc23(relAddr, tgtAddr, ref.addend());
  case R_MIPS_TLS_DTPREL_HI16:
  case R_MIPS_TLS_TPREL_HI16:
  case R_MICROMIPS_TLS_DTPREL_HI16:
  case R_MICROMIPS_TLS_TPREL_HI16:
    return relocHi16(0, tgtAddr, ref.addend(), false);
  case R_MIPS_TLS_DTPREL_LO16:
  case R_MIPS_TLS_TPREL_LO16:
    return relocLo16(0, tgtAddr, ref.addend(), false, false);
  case R_MICROMIPS_TLS_DTPREL_LO16:
  case R_MICROMIPS_TLS_TPREL_LO16:
    return relocLo16(0, tgtAddr, ref.addend(), false, true);
  case R_MIPS_GPREL16:
    return relocGPRel16(tgtAddr, ref.addend(), gpAddr);
  case R_MIPS_GPREL32:
    return relocGPRel32(tgtAddr, ref.addend(), gpAddr);
  case R_MIPS_JALR:
  case R_MICROMIPS_JALR:
    // We do not do JALR optimization now.
    return 0;
  case R_MIPS_REL32:
  case R_MIPS_JUMP_SLOT:
  case R_MIPS_COPY:
  case R_MIPS_TLS_DTPMOD32:
  case R_MIPS_TLS_DTPREL32:
  case R_MIPS_TLS_TPREL32:
  case R_MIPS_TLS_DTPMOD64:
  case R_MIPS_TLS_DTPREL64:
  case R_MIPS_TLS_TPREL64:
    // Ignore runtime relocations.
    return 0;
  case R_MIPS_PC32:
    return relocpc32(relAddr, tgtAddr, ref.addend());
  case LLD_R_MIPS_GLOBAL_GOT:
    // Do nothing.
  case LLD_R_MIPS_32_HI16:
  case LLD_R_MIPS_64_HI16:
    return relocMaskLow16(tgtAddr, ref.addend());
  case LLD_R_MIPS_GLOBAL_26:
    return reloc26ext(tgtAddr, ref.addend(), 2);
  case LLD_R_MICROMIPS_GLOBAL_26_S1:
    return reloc26ext(tgtAddr, ref.addend(), isCrossJump ? 2 : 1);
  case LLD_R_MIPS_HI16:
    return relocHi16(0, tgtAddr, 0, false);
  case LLD_R_MIPS_LO16:
    return relocLo16(0, tgtAddr, 0, false, false);
  case LLD_R_MIPS_STO_PLT:
    // Do nothing.
    return 0;
  default:
    return make_unhandled_reloc_error();
  }
}

template <class ELFT>
static uint64_t relocRead(const MipsRelocationParams &params,
                          const uint8_t *loc) {
  uint64_t data;
  switch (params._size) {
  case 4:
    data = endian::read<uint32_t, ELFT::TargetEndianness, unaligned>(loc);
    break;
  case 8:
    data = endian::read<uint64_t, ELFT::TargetEndianness, unaligned>(loc);
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

template <class ELFT>
std::error_code RelocationHandler<ELFT>::applyRelocation(
    ELFWriter &writer, llvm::FileOutputBuffer &buf, const lld::AtomLayout &atom,
    const Reference &ref) const {
  if (ref.kindNamespace() != lld::Reference::KindNamespace::ELF)
    return std::error_code();
  assert(ref.kindArch() == Reference::KindArch::Mips);

  auto &targetLayout = static_cast<MipsTargetLayout<ELFT> &>(
      _ctx.getTargetHandler<ELFT>().getTargetLayout());

  AtomLayout *gpAtom = targetLayout.getGP();
  uint64_t gpAddr = gpAtom ? gpAtom->_virtualAddr : 0;

  AtomLayout *gpDispAtom = targetLayout.getGPDisp();
  bool isGpDisp = gpDispAtom && ref.target() == gpDispAtom->_atom;

  uint8_t *atomContent = buf.getBufferStart() + atom._fileOffset;
  uint8_t *location = atomContent + ref.offsetInAtom();
  uint64_t tgtAddr = writer.addressOfAtom(ref.target());
  uint64_t relAddr = atom._virtualAddr + ref.offsetInAtom();

  if (isMicroMipsAtom(ref.target()))
    tgtAddr |= 1;

  auto res = calculateRelocation(ref, tgtAddr, relAddr, gpAddr, isGpDisp);
  if (auto ec = res.getError())
    return ec;

  auto params = getRelocationParams(ref.kindValue());
  uint64_t ins = relocRead<ELFT>(params, location);

  if (auto ec = adjustJumpOpCode(ins, tgtAddr, getCrossJumpMode(ref)))
    return ec;

  ins = (ins & ~params._mask) | (*res & params._mask);
  relocWrite<ELFT>(ins, params, location);

  return std::error_code();
}

template <class ELFT>
Reference::Addend
RelocationHandler<ELFT>::readAddend(Reference::KindValue kind,
                                    const uint8_t *content) const {
  auto params = getRelocationParams(kind);
  uint64_t ins = relocRead<ELFT>(params, content);
  return (ins & params._mask) << params._shift;
}

namespace lld {
namespace elf {

template <>
std::unique_ptr<TargetRelocationHandler>
createMipsRelocationHandler<Mips32ELType>(MipsLinkingContext &ctx) {
  return std::unique_ptr<TargetRelocationHandler>(
      new RelocationHandler<Mips32ELType>(ctx));
}

template <>
std::unique_ptr<TargetRelocationHandler>
createMipsRelocationHandler<Mips64ELType>(MipsLinkingContext &ctx) {
  return std::unique_ptr<TargetRelocationHandler>(
      new RelocationHandler<Mips64ELType>(ctx));
}

} // elf
} // lld
