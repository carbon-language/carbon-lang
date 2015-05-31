//===- lib/ReaderWriter/ELF/MipsAbiInfoHandler.cpp ------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MipsAbiInfoHandler.h"
#include "lld/Core/Error.h"
#include "lld/ReaderWriter/ELFLinkingContext.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/MipsABIFlags.h"
#include "llvm/Support/raw_ostream.h"

using namespace lld;
using namespace lld::elf;
using namespace llvm;
using namespace llvm::ELF;
using namespace llvm::Mips;

namespace {

// The joined set of MIPS ISAs and MIPS ISA extensions.
enum MipsISAs {
  ArchNone,

  // General ISAs
  Arch1,
  Arch2,
  Arch3,
  Arch4,
  Arch5,
  Arch32,
  Arch32r2,
  Arch32r3,
  Arch32r5,
  Arch32r6,
  Arch64,
  Arch64r2,
  Arch64r3,
  Arch64r5,
  Arch64r6,

  // CPU specific ISAs
  Arch3900,
  Arch4010,
  Arch4100,
  Arch4111,
  Arch4120,
  Arch4650,
  Arch5400,
  Arch5500,
  Arch5900,
  Arch9000,
  Arch10000,
  ArchLs2e,
  ArchLs2f,
  ArchLs3a,
  ArchOcteon,
  ArchOcteonP,
  ArchOcteon2,
  ArchOcteon3,
  ArchSB1,
  ArchXLR
};

struct MipsISATreeEdge {
  MipsISAs child;
  MipsISAs parent;
};

struct ElfArchPair {
  uint32_t _elfFlag;
  MipsISAs _arch;
};

struct AbiIsaArchPair {
  uint8_t _isaLevel;
  uint8_t _isaRev;
  uint8_t _isaExt;
  MipsISAs _arch;
};
}

static const MipsISATreeEdge isaTree[] = {
  // MIPS32R6 and MIPS64R6 are not compatible with other extensions

  // MIPS64R2 extensions.
  {ArchOcteon3, ArchOcteon2},
  {ArchOcteon2, ArchOcteonP},
  {ArchOcteonP, ArchOcteon},
  {ArchOcteon,  Arch64r2},
  {ArchLs3a,    Arch64r2},

  // MIPS64 extensions.
  {Arch64r2, Arch64},
  {ArchSB1,  Arch64},
  {ArchXLR,  Arch64},

  // MIPS V extensions.
  {Arch64, Arch5},

  // R5000 extensions.
  {Arch5500, Arch5400},

  // MIPS IV extensions.
  {Arch5, Arch4},
  {Arch5400, Arch4},
  {Arch9000, Arch4},

  // VR4100 extensions.
  {Arch4120, Arch4100},
  {Arch4111, Arch4100},

  // MIPS III extensions.
  {ArchLs2e, Arch3},
  {ArchLs2f, Arch3},
  {Arch4650, Arch3},
  {Arch4100, Arch3},
  {Arch4010, Arch3},
  {Arch5900, Arch3},
  {Arch4,    Arch3},

  // MIPS32 extensions.
  {Arch32r2, Arch32},

  // MIPS II extensions.
  {Arch3, Arch2},
  {Arch32, Arch2},

  // MIPS I extensions.
  {Arch3900, Arch1},
  {Arch2,    Arch1},
};

// Conversion ELF arch flags => MipsISAs
static const ElfArchPair elfArchPairs[] = {
  {EF_MIPS_ARCH_1    | EF_MIPS_MACH_3900,    Arch3900},
  {EF_MIPS_ARCH_3    | EF_MIPS_MACH_4010,    Arch4010},
  {EF_MIPS_ARCH_3    | EF_MIPS_MACH_4100,    Arch4100},
  {EF_MIPS_ARCH_3    | EF_MIPS_MACH_4111,    Arch4111},
  {EF_MIPS_ARCH_3    | EF_MIPS_MACH_4120,    Arch4120},
  {EF_MIPS_ARCH_3    | EF_MIPS_MACH_4650,    Arch4650},
  {EF_MIPS_ARCH_4    | EF_MIPS_MACH_5400,    Arch5400},
  {EF_MIPS_ARCH_4    | EF_MIPS_MACH_5500,    Arch5500},
  {EF_MIPS_ARCH_3    | EF_MIPS_MACH_5900,    Arch5900},
  {EF_MIPS_ARCH_4    | EF_MIPS_MACH_9000,    Arch9000},
  {EF_MIPS_ARCH_3    | EF_MIPS_MACH_LS2E,    ArchLs2e},
  {EF_MIPS_ARCH_3    | EF_MIPS_MACH_LS2F,    ArchLs2f},
  {EF_MIPS_ARCH_64R2 | EF_MIPS_MACH_LS3A,    ArchLs3a},
  {EF_MIPS_ARCH_64R2 | EF_MIPS_MACH_OCTEON,  ArchOcteon},
  {EF_MIPS_ARCH_64R2 | EF_MIPS_MACH_OCTEON2, ArchOcteon2},
  {EF_MIPS_ARCH_64R2 | EF_MIPS_MACH_OCTEON3, ArchOcteon3},
  {EF_MIPS_ARCH_64   | EF_MIPS_MACH_SB1,     ArchSB1},
  {EF_MIPS_ARCH_64   | EF_MIPS_MACH_XLR,     ArchXLR},
  {EF_MIPS_ARCH_1,    Arch1},
  {EF_MIPS_ARCH_2,    Arch2},
  {EF_MIPS_ARCH_3,    Arch3},
  {EF_MIPS_ARCH_4,    Arch4},
  {EF_MIPS_ARCH_5,    Arch5},
  {EF_MIPS_ARCH_32,   Arch32},
  {EF_MIPS_ARCH_32R2, Arch32r2},
  {EF_MIPS_ARCH_32R6, Arch32r6},
  {EF_MIPS_ARCH_64,   Arch64},
  {EF_MIPS_ARCH_64R2, Arch64r2},
  {EF_MIPS_ARCH_64R6, Arch64r6}
};

// Conversion MipsISAs => ELF arch flags
static const ElfArchPair archElfPairs[] = {
  {EF_MIPS_ARCH_1,    Arch1},
  {EF_MIPS_ARCH_2,    Arch2},
  {EF_MIPS_ARCH_3,    Arch3},
  {EF_MIPS_ARCH_4,    Arch4},
  {EF_MIPS_ARCH_5,    Arch5},
  {EF_MIPS_ARCH_32,   Arch32},
  {EF_MIPS_ARCH_32R2, Arch32r2},
  {EF_MIPS_ARCH_32R2, Arch32r3},
  {EF_MIPS_ARCH_32R2, Arch32r5},
  {EF_MIPS_ARCH_32R6, Arch32r6},
  {EF_MIPS_ARCH_64,   Arch64},
  {EF_MIPS_ARCH_64R2, Arch64r2},
  {EF_MIPS_ARCH_64R2, Arch64r3},
  {EF_MIPS_ARCH_64R2, Arch64r5},
  {EF_MIPS_ARCH_64R6, Arch64r6},
  {EF_MIPS_ARCH_1    | EF_MIPS_MACH_3900,    Arch3900},
  {EF_MIPS_ARCH_3    | EF_MIPS_MACH_4010,    Arch4010},
  {EF_MIPS_ARCH_3    | EF_MIPS_MACH_4100,    Arch4100},
  {EF_MIPS_ARCH_3    | EF_MIPS_MACH_4111,    Arch4111},
  {EF_MIPS_ARCH_3    | EF_MIPS_MACH_4120,    Arch4120},
  {EF_MIPS_ARCH_3    | EF_MIPS_MACH_4650,    Arch4650},
  {EF_MIPS_ARCH_4    | EF_MIPS_MACH_5400,    Arch5400},
  {EF_MIPS_ARCH_4    | EF_MIPS_MACH_5500,    Arch5500},
  {EF_MIPS_ARCH_3    | EF_MIPS_MACH_5900,    Arch5900},
  {EF_MIPS_ARCH_4    | EF_MIPS_MACH_9000,    Arch9000},
  {EF_MIPS_ARCH_4,                           Arch10000},
  {EF_MIPS_ARCH_3    | EF_MIPS_MACH_LS2E,    ArchLs2e},
  {EF_MIPS_ARCH_3    | EF_MIPS_MACH_LS2F,    ArchLs2f},
  {EF_MIPS_ARCH_64R2 | EF_MIPS_MACH_LS3A,    ArchLs3a},
  {EF_MIPS_ARCH_64R2 | EF_MIPS_MACH_OCTEON,  ArchOcteon},
  {EF_MIPS_ARCH_64R2 | EF_MIPS_MACH_OCTEON,  ArchOcteonP},
  {EF_MIPS_ARCH_64R2 | EF_MIPS_MACH_OCTEON2, ArchOcteon2},
  {EF_MIPS_ARCH_64R2 | EF_MIPS_MACH_OCTEON3, ArchOcteon3},
  {EF_MIPS_ARCH_64   | EF_MIPS_MACH_SB1,     ArchSB1},
  {EF_MIPS_ARCH_64   | EF_MIPS_MACH_SB1,     ArchXLR}
};

// Conversion .MIPS.abiflags isa/level/extension <=> MipsISAs
static const AbiIsaArchPair abiIsaArchPair[] = {
  { 0, 0, 0, ArchNone},
  { 1, 0, 0, Arch1},
  { 2, 0, 0, Arch2},
  { 3, 0, 0, Arch3},
  { 4, 0, 0, Arch4},
  { 5, 0, 0, Arch5},
  {32, 1, 0, Arch32},
  {32, 2, 0, Arch32r2},
  {32, 3, 0, Arch32r3},
  {32, 5, 0, Arch32r5},
  {32, 6, 0, Arch32r6},
  {64, 1, 0, Arch64},
  {64, 2, 0, Arch64r2},
  {64, 3, 0, Arch64r3},
  {64, 5, 0, Arch64r5},
  {64, 6, 0, Arch64r6},
  { 1, 0, AFL_EXT_3900,  Arch3900},
  { 3, 0, AFL_EXT_4010,  Arch4010},
  { 3, 0, AFL_EXT_4100,  Arch4100},
  { 3, 0, AFL_EXT_4111,  Arch4111},
  { 3, 0, AFL_EXT_4120,  Arch4120},
  { 3, 0, AFL_EXT_4650,  Arch4650},
  { 4, 0, AFL_EXT_5400,  Arch5400},
  { 4, 0, AFL_EXT_5500,  Arch5500},
  { 3, 0, AFL_EXT_5900,  Arch5900},
  { 4, 0, AFL_EXT_10000, Arch10000},
  { 3, 0, AFL_EXT_LOONGSON_2E, ArchLs2e},
  { 3, 0, AFL_EXT_LOONGSON_2F, ArchLs2f},
  {64, 2, AFL_EXT_LOONGSON_3A, ArchLs3a},
  {64, 2, AFL_EXT_OCTEON,  ArchOcteon},
  {64, 2, AFL_EXT_OCTEON2, ArchOcteon2},
  {64, 2, AFL_EXT_OCTEON3, ArchOcteon3},
  {64, 1, AFL_EXT_SB1,     ArchSB1},
  {64, 1, AFL_EXT_XLR,     ArchXLR}
};

static bool matchMipsISA(MipsISAs base, MipsISAs ext) {
  if (base == ext)
    return true;
  if (base == Arch32 && matchMipsISA(Arch64, ext))
    return true;
  if (base == Arch32r2 && matchMipsISA(Arch64r2, ext))
    return true;
  for (const auto &edge : isaTree) {
    if (ext == edge.child) {
      ext = edge.parent;
      if (ext == base)
        return true;
    }
  }
  return false;
}

static bool is32BitElfFlags(unsigned flags) {
  if (flags & EF_MIPS_32BITMODE)
    return true;

  unsigned arch = flags & EF_MIPS_ARCH;
  if (arch == EF_MIPS_ARCH_1 || arch == EF_MIPS_ARCH_2 ||
      arch == EF_MIPS_ARCH_32 || arch == EF_MIPS_ARCH_32R2 ||
      arch == EF_MIPS_ARCH_32R6)
    return true;

  unsigned abi = flags & EF_MIPS_ABI;
  if (abi == EF_MIPS_ABI_O32 || abi == EF_MIPS_ABI_EABI32)
    return true;

  return false;
}

static ErrorOr<MipsISAs> headerFlagsToIsa(uint32_t flags) {
  uint32_t arch = flags & (EF_MIPS_ARCH | EF_MIPS_MACH);
  for (const auto &p : elfArchPairs)
    if (p._elfFlag == arch)
      return p._arch;
  return make_dynamic_error_code(
      StringRef("Unknown EF_MIPS_ARCH | EF_MIPS_MACH flags (0x") +
      Twine::utohexstr(arch) + ")");
}

static uint32_t isaToHeaderFlags(unsigned isa) {
  for (const auto &p : archElfPairs)
    if (p._arch == isa)
      return p._elfFlag;
  llvm_unreachable("Unknown MIPS ISA");
}

static ErrorOr<uint32_t> flagsToAses(uint32_t flags) {
  uint32_t ases = flags & EF_MIPS_ARCH_ASE;
  switch (ases) {
  case 0:
    return 0;
  case EF_MIPS_MICROMIPS:
    return AFL_ASE_MICROMIPS;
  case EF_MIPS_ARCH_ASE_M16:
    return AFL_ASE_MIPS16;
  case EF_MIPS_ARCH_ASE_MDMX:
    return AFL_ASE_MDMX;
  default:
    return make_dynamic_error_code(
        StringRef("Unknown EF_MIPS_ARCH_ASE flag (0x") +
        Twine::utohexstr(ases) + ")");
  }
}

static uint32_t asesToFlags(uint32_t ases) {
  switch (ases) {
  case AFL_ASE_MICROMIPS:
    return EF_MIPS_MICROMIPS;
  case AFL_ASE_MIPS16:
    return EF_MIPS_ARCH_ASE_M16;
  case AFL_ASE_MDMX:
    return EF_MIPS_ARCH_ASE_MDMX;
  default:
    return 0;
  }
}

static ErrorOr<MipsISAs> sectionFlagsToIsa(uint8_t isaLevel, uint8_t isaRev,
                                           uint8_t isaExt) {
  for (const auto &p : abiIsaArchPair)
    if (p._isaLevel == isaLevel && p._isaRev == isaRev && p._isaExt == isaExt)
      return p._arch;
  return make_dynamic_error_code(
      StringRef("Unknown ISA level/revision/extension ") + Twine(isaLevel) +
      "/" + Twine(isaRev) + "/" + Twine(isaExt));
}

static std::tuple<uint8_t, uint8_t, uint32_t> isaToSectionFlags(unsigned isa) {
  for (const auto &p : abiIsaArchPair)
    if (p._arch == isa)
      return std::make_tuple(p._isaLevel, p._isaRev, p._isaExt);
  llvm_unreachable("Unknown MIPS ISA");
}

static bool checkCompatibility(const MipsAbiFlags &hdr,
                               const MipsAbiFlags &sec) {
  uint32_t secIsa = ArchNone;
  switch (sec._isa) {
  case Arch32r3:
  case Arch32r5:
    secIsa = Arch32r2;
    break;
  case Arch64r3:
  case Arch64r5:
    secIsa = Arch64r2;
    break;
  default:
    secIsa = sec._isa;
    break;
  }
  if (secIsa != hdr._isa) {
    llvm::errs() << "inconsistent ISA between .MIPS.abiflags "
                    "and ELF header e_flags field\n";
    return false;
  }
  if ((sec._ases & hdr._ases) != hdr._ases) {
    llvm::errs() << "inconsistent ASEs between .MIPS.abiflags "
                    "and ELF header e_flags field\n";
    return false;
  }
  return true;
}

static int compareFpAbi(uint32_t fpA, uint32_t fpB) {
  if (fpA == fpB)
    return 0;
  if (fpB == Val_GNU_MIPS_ABI_FP_ANY)
    return 1;
  if (fpB == Val_GNU_MIPS_ABI_FP_64A && fpA == Val_GNU_MIPS_ABI_FP_64)
    return 1;
  if (fpB != Val_GNU_MIPS_ABI_FP_XX)
    return -1;
  if (fpA == Val_GNU_MIPS_ABI_FP_DOUBLE || fpA == Val_GNU_MIPS_ABI_FP_64 ||
      fpA == Val_GNU_MIPS_ABI_FP_64A)
    return 1;
  return -1;
}

static StringRef getFpAbiName(uint32_t fpAbi) {
  switch (fpAbi) {
  case Val_GNU_MIPS_ABI_FP_ANY:
    return "<any>";
  case Val_GNU_MIPS_ABI_FP_DOUBLE:
    return "-mdouble-float";
  case Val_GNU_MIPS_ABI_FP_SINGLE:
    return "-msingle-float";
  case Val_GNU_MIPS_ABI_FP_SOFT:
    return "-msoft-float";
  case Val_GNU_MIPS_ABI_FP_OLD_64:
    return "-mips32r2 -mfp64 (old)";
  case Val_GNU_MIPS_ABI_FP_XX:
    return "-mfpxx";
  case Val_GNU_MIPS_ABI_FP_64:
    return "-mgp32 -mfp64";
  case Val_GNU_MIPS_ABI_FP_64A:
    return "-mgp32 -mfp64 -mno-odd-spreg";
  default:
    return "<unknown>";
  }
}

static uint32_t selectFpAbiFlag(uint32_t oldFp, uint32_t newFp) {
  if (compareFpAbi(newFp, oldFp) >= 0)
    return newFp;
  if (compareFpAbi(oldFp, newFp) < 0)
    llvm::errs() << "FP ABI " << getFpAbiName(oldFp) << " is incompatible with "
                 << getFpAbiName(newFp) << "\n";
  return oldFp;
}

namespace lld {
namespace elf {

template <class ELFT> bool MipsAbiInfoHandler<ELFT>::isMicroMips() const {
  assert(_abiFlags.hasValue());
  return _abiFlags->_ases & AFL_ASE_MICROMIPS;
}

template <class ELFT> bool MipsAbiInfoHandler<ELFT>::isMipsR6() const {
  assert(_abiFlags.hasValue());
  return _abiFlags->_isa == Arch32r6 || _abiFlags->_isa == Arch64r6;
}

template <class ELFT> uint32_t MipsAbiInfoHandler<ELFT>::getFlags() const {
  std::lock_guard<std::mutex> lock(_mutex);
  uint32_t flags = 0;
  if (_abiFlags.hasValue()) {
    flags |= isaToHeaderFlags(_abiFlags->_isa);
    flags |= asesToFlags(_abiFlags->_ases);
    flags |= _abiFlags->_abi;
    flags |= _abiFlags->_isPic ? EF_MIPS_PIC : 0;
    flags |= _abiFlags->_isCPic ? EF_MIPS_CPIC : 0;
    flags |= _abiFlags->_isNoReorder ? EF_MIPS_NOREORDER : 0;
    flags |= _abiFlags->_is32BitMode ? EF_MIPS_32BITMODE : 0;
    flags |= _abiFlags->_isNan2008 ? EF_MIPS_NAN2008 : 0;
  }
  return flags;
}

template <class ELFT>
llvm::Optional<typename MipsAbiInfoHandler<ELFT>::Elf_Mips_RegInfo>
MipsAbiInfoHandler<ELFT>::getRegistersMask() const {
  std::lock_guard<std::mutex> lock(_mutex);
  return _regMask;
}

template <class ELFT>
llvm::Optional<typename MipsAbiInfoHandler<ELFT>::Elf_Mips_ABIFlags>
MipsAbiInfoHandler<ELFT>::getAbiFlags() const {
  std::lock_guard<std::mutex> lock(_mutex);
  if (!_hasAbiSection)
    return llvm::Optional<Elf_Mips_ABIFlags>();

  Elf_Mips_ABIFlags sec;
  sec.version = 0;
  std::tie(sec.isa_level, sec.isa_rev, sec.isa_ext) =
      isaToSectionFlags(_abiFlags->_isa);
  sec.gpr_size = _abiFlags->_gprSize;
  sec.cpr1_size = _abiFlags->_cpr1Size;
  sec.cpr2_size = _abiFlags->_cpr2Size;
  sec.fp_abi = _abiFlags->_fpAbi;
  sec.ases = _abiFlags->_ases;
  sec.flags1 = _abiFlags->_flags1;
  sec.flags2 = 0;
  return sec;
}

template <class ELFT>
std::error_code
MipsAbiInfoHandler<ELFT>::mergeFlags(uint32_t newFlags,
                                     const Elf_Mips_ABIFlags *newSec) {
  std::lock_guard<std::mutex> lock(_mutex);

  ErrorOr<MipsAbiFlags> abiFlags = createAbiFlags(newFlags, newSec);
  if (auto ec = abiFlags.getError())
    return ec;

  // We support two ABI: O32 and N64. The last one does not have
  // the corresponding ELF flag.
  uint32_t supportedAbi = ELFT::Is64Bits ? 0 : uint32_t(EF_MIPS_ABI_O32);
  if (abiFlags->_abi != supportedAbi)
    return make_dynamic_error_code("Unsupported ABI");

  // ... and still do not support MIPS-16 extension.
  if (abiFlags->_ases & AFL_ASE_MIPS16)
    return make_dynamic_error_code("Unsupported extension: MIPS16");

  // PIC code is inherently CPIC and may not set CPIC flag explicitly.
  // Ensure that this flag will exist in the linked file.
  if (abiFlags->_isPic)
    abiFlags->_isCPic = true;

  // If the old set of flags is empty, use the new one as a result.
  if (!_abiFlags.hasValue()) {
    _abiFlags = *abiFlags;
    return std::error_code();
  }

  // Check PIC / CPIC flags compatibility.
  if (abiFlags->_isCPic != _abiFlags->_isCPic)
    llvm::errs() << "lld warning: linking abicalls and non-abicalls files\n";

  if (!abiFlags->_isPic)
    _abiFlags->_isPic = false;
  if (abiFlags->_isCPic)
    _abiFlags->_isCPic = true;

  // Check mixing -mnan=2008 / -mnan=legacy modules.
  if (abiFlags->_isNan2008 != _abiFlags->_isNan2008)
    return make_dynamic_error_code(
        "Linking -mnan=2008 and -mnan=legacy modules");

  // Check ISA compatibility and update the extension flag.
  if (!matchMipsISA(MipsISAs(abiFlags->_isa), MipsISAs(_abiFlags->_isa))) {
    if (!matchMipsISA(MipsISAs(_abiFlags->_isa), MipsISAs(abiFlags->_isa)))
      return make_dynamic_error_code("Linking modules with incompatible ISA");
    _abiFlags->_isa = abiFlags->_isa;
  }

  _abiFlags->_ases |= abiFlags->_ases;
  _abiFlags->_isNoReorder = _abiFlags->_isNoReorder || abiFlags->_isNoReorder;
  _abiFlags->_is32BitMode = _abiFlags->_is32BitMode || abiFlags->_is32BitMode;

  _abiFlags->_fpAbi = selectFpAbiFlag(_abiFlags->_fpAbi, abiFlags->_fpAbi);
  _abiFlags->_gprSize = std::max(_abiFlags->_gprSize, abiFlags->_gprSize);
  _abiFlags->_cpr1Size = std::max(_abiFlags->_cpr1Size, abiFlags->_cpr1Size);
  _abiFlags->_cpr2Size = std::max(_abiFlags->_cpr2Size, abiFlags->_cpr2Size);
  _abiFlags->_flags1 |= abiFlags->_flags1;

  return std::error_code();
}

template <class ELFT>
void MipsAbiInfoHandler<ELFT>::mergeRegistersMask(
    const Elf_Mips_RegInfo &info) {
  std::lock_guard<std::mutex> lock(_mutex);
  if (!_regMask.hasValue()) {
    _regMask = info;
    return;
  }
  _regMask->ri_gprmask = _regMask->ri_gprmask | info.ri_gprmask;
  _regMask->ri_cprmask[0] = _regMask->ri_cprmask[0] | info.ri_cprmask[0];
  _regMask->ri_cprmask[1] = _regMask->ri_cprmask[1] | info.ri_cprmask[1];
  _regMask->ri_cprmask[2] = _regMask->ri_cprmask[2] | info.ri_cprmask[2];
  _regMask->ri_cprmask[3] = _regMask->ri_cprmask[3] | info.ri_cprmask[3];
}

template <class ELFT>
ErrorOr<MipsAbiFlags>
MipsAbiInfoHandler<ELFT>::createAbiFlags(uint32_t flags,
                                         const Elf_Mips_ABIFlags *sec) {
  ErrorOr<MipsAbiFlags> hdrFlags = createAbiFromHeaderFlags(flags);
  if (auto ec = hdrFlags.getError())
    return ec;
  if (!sec)
    return *hdrFlags;
  ErrorOr<MipsAbiFlags> secFlags = createAbiFromSection(*sec);
  if (auto ec = secFlags.getError())
    return ec;
  if (!checkCompatibility(*hdrFlags, *secFlags))
    return *hdrFlags;

  _hasAbiSection = true;

  secFlags->_abi = hdrFlags->_abi;
  secFlags->_isPic = hdrFlags->_isPic;
  secFlags->_isCPic = hdrFlags->_isCPic;
  secFlags->_isNoReorder = hdrFlags->_isNoReorder;
  secFlags->_is32BitMode = hdrFlags->_is32BitMode;
  secFlags->_isNan2008 = hdrFlags->_isNan2008;
  return *secFlags;
}

template <class ELFT>
ErrorOr<MipsAbiFlags>
MipsAbiInfoHandler<ELFT>::createAbiFromHeaderFlags(uint32_t flags) {
  MipsAbiFlags abi;
  ErrorOr<MipsISAs> isa = headerFlagsToIsa(flags);
  if (auto ec = isa.getError())
    return ec;
  abi._isa = *isa;

  abi._fpAbi = Val_GNU_MIPS_ABI_FP_ANY;
  abi._cpr1Size = AFL_REG_NONE;
  abi._cpr2Size = AFL_REG_NONE;
  abi._gprSize = is32BitElfFlags(flags) ? AFL_REG_32 : AFL_REG_64;

  ErrorOr<uint32_t> ases = flagsToAses(flags);
  if (auto ec = ases.getError())
    return ec;
  abi._ases = *ases;
  abi._flags1 = 0;
  abi._abi = flags & EF_MIPS_ABI;
  abi._isPic = flags & EF_MIPS_PIC;
  abi._isCPic = flags & EF_MIPS_CPIC;
  abi._isNoReorder = flags & EF_MIPS_NOREORDER;
  abi._is32BitMode = flags & EF_MIPS_32BITMODE;
  abi._isNan2008 = flags & EF_MIPS_NAN2008;
  return abi;
}

template <class ELFT>
ErrorOr<MipsAbiFlags>
MipsAbiInfoHandler<ELFT>::createAbiFromSection(const Elf_Mips_ABIFlags &sec) {
  MipsAbiFlags abi;
  ErrorOr<MipsISAs> isa =
      sectionFlagsToIsa(sec.isa_level, sec.isa_rev, sec.isa_ext);
  if (auto ec = isa.getError())
    return ec;
  abi._isa = *isa;
  abi._fpAbi = sec.fp_abi;
  abi._cpr1Size = sec.cpr1_size;
  abi._cpr2Size = sec.cpr2_size;
  abi._gprSize = sec.gpr_size;
  abi._ases = sec.ases;
  abi._flags1 = sec.flags1;
  if (sec.flags2 != 0)
    return make_dynamic_error_code("unexpected non-zero 'flags2' value");
  return abi;
}

template class MipsAbiInfoHandler<ELF32LE>;
template class MipsAbiInfoHandler<ELF64LE>;

}
}
