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
#include "llvm/Support/raw_ostream.h"

using namespace llvm::ELF;

struct MipsISATreeEdge {
  unsigned child;
  unsigned parent;
};

static MipsISATreeEdge isaTree[] = {
    // MIPS32R6 and MIPS64R6 are not compatible with other extensions

    // MIPS64 extensions.
    {EF_MIPS_ARCH_64R2, EF_MIPS_ARCH_64},
    // MIPS V extensions.
    {EF_MIPS_ARCH_64, EF_MIPS_ARCH_5},
    // MIPS IV extensions.
    {EF_MIPS_ARCH_5, EF_MIPS_ARCH_4},
    // MIPS III extensions.
    {EF_MIPS_ARCH_4, EF_MIPS_ARCH_3},
    // MIPS32 extensions.
    {EF_MIPS_ARCH_32R2, EF_MIPS_ARCH_32},
    // MIPS II extensions.
    {EF_MIPS_ARCH_3, EF_MIPS_ARCH_2},
    {EF_MIPS_ARCH_32, EF_MIPS_ARCH_2},
    // MIPS I extensions.
    {EF_MIPS_ARCH_2, EF_MIPS_ARCH_1},
};

static bool matchMipsISA(unsigned base, unsigned ext) {
  if (base == ext)
    return true;
  if (base == EF_MIPS_ARCH_32 && matchMipsISA(EF_MIPS_ARCH_64, ext))
    return true;
  if (base == EF_MIPS_ARCH_32R2 && matchMipsISA(EF_MIPS_ARCH_64R2, ext))
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

namespace lld {
namespace elf {

template <class ELFT>
std::error_code MipsAbiInfoHandler<ELFT>::mergeFlags(uint32_t newFlags) {
  // We support two ABI: O32 and N64. The last one does not have
  // the corresponding ELF flag.
  uint32_t inAbi = newFlags & EF_MIPS_ABI;
  uint32_t supportedAbi = ELFT::Is64Bits ? 0 : uint32_t(EF_MIPS_ABI_O32);
  if (inAbi != supportedAbi)
    return make_dynamic_error_code("Unsupported ABI");

  // ... and reduced set of architectures ...
  uint32_t newArch = newFlags & EF_MIPS_ARCH;
  switch (newArch) {
  case EF_MIPS_ARCH_1:
  case EF_MIPS_ARCH_2:
  case EF_MIPS_ARCH_3:
  case EF_MIPS_ARCH_4:
  case EF_MIPS_ARCH_5:
  case EF_MIPS_ARCH_32:
  case EF_MIPS_ARCH_64:
  case EF_MIPS_ARCH_32R2:
  case EF_MIPS_ARCH_64R2:
  case EF_MIPS_ARCH_32R6:
  case EF_MIPS_ARCH_64R6:
    break;
  default:
    return make_dynamic_error_code("Unsupported instruction set");
  }

  // ... and still do not support MIPS-16 extension.
  if (newFlags & EF_MIPS_ARCH_ASE_M16)
    return make_dynamic_error_code("Unsupported extension: MIPS16");

  // PIC code is inherently CPIC and may not set CPIC flag explicitly.
  // Ensure that this flag will exist in the linked file.
  if (newFlags & EF_MIPS_PIC)
    newFlags |= EF_MIPS_CPIC;

  std::lock_guard<std::mutex> lock(_mutex);

  // If the old set of flags is empty, use the new one as a result.
  if (!_flags) {
    _flags = newFlags;
    return std::error_code();
  }

  // Check PIC / CPIC flags compatibility.
  uint32_t newPic = newFlags & (EF_MIPS_PIC | EF_MIPS_CPIC);
  uint32_t oldPic = _flags & (EF_MIPS_PIC | EF_MIPS_CPIC);

  if ((newPic != 0) != (oldPic != 0))
    llvm::errs() << "lld warning: linking abicalls and non-abicalls files\n";

  if (!(newPic & EF_MIPS_PIC))
    _flags &= ~EF_MIPS_PIC;
  if (newPic)
    _flags |= EF_MIPS_CPIC;

  // Check mixing -mnan=2008 / -mnan=legacy modules.
  if ((newFlags & EF_MIPS_NAN2008) != (_flags & EF_MIPS_NAN2008))
    return make_dynamic_error_code(
        "Linking -mnan=2008 and -mnan=legacy modules");

  // Check ISA compatibility and update the extension flag.
  uint32_t oldArch = _flags & EF_MIPS_ARCH;
  if (!matchMipsISA(newArch, oldArch)) {
    if (!matchMipsISA(oldArch, newArch))
      return make_dynamic_error_code("Linking modules with incompatible ISA");
    _flags &= ~EF_MIPS_ARCH;
    _flags |= newArch;
  }

  _flags |= newFlags & EF_MIPS_NOREORDER;
  _flags |= newFlags & EF_MIPS_MICROMIPS;
  _flags |= newFlags & EF_MIPS_NAN2008;
  _flags |= newFlags & EF_MIPS_32BITMODE;

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

template class MipsAbiInfoHandler<ELF32LE>;
template class MipsAbiInfoHandler<ELF64LE>;

}
}
