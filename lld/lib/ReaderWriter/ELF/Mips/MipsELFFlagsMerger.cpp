//===- lib/ReaderWriter/ELF/MipsELFFlagsMerger.cpp ------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MipsELFFlagsMerger.h"
#include "lld/Core/Error.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/raw_ostream.h"

using namespace lld;
using namespace lld::elf;

MipsELFFlagsMerger::MipsELFFlagsMerger() : _flags(0) {}

uint32_t MipsELFFlagsMerger::getMergedELFFlags() const { return _flags; }

std::error_code MipsELFFlagsMerger::merge(uint8_t newClass, uint32_t newFlags) {
  // Reject 64-bit binaries.
  if (newClass != llvm::ELF::ELFCLASS32)
    return make_dynamic_error_code(
        Twine("Bitness is incompatible with that of the selected target"));

  // We support the only ABI - O32 ...
  uint32_t abi = newFlags & llvm::ELF::EF_MIPS_ABI;
  if (abi != llvm::ELF::EF_MIPS_ABI_O32)
    return make_dynamic_error_code(Twine("Unsupported ABI"));

  // ... and reduced set of architectures ...
  uint32_t newArch = newFlags & llvm::ELF::EF_MIPS_ARCH;
  switch (newArch) {
  case llvm::ELF::EF_MIPS_ARCH_1:
  case llvm::ELF::EF_MIPS_ARCH_2:
  case llvm::ELF::EF_MIPS_ARCH_32:
  case llvm::ELF::EF_MIPS_ARCH_32R2:
  case llvm::ELF::EF_MIPS_ARCH_32R6:
    break;
  default:
    return make_dynamic_error_code(Twine("Unsupported architecture"));
  }

  // ... and still do not support MIPS-16 extension.
  if (newFlags & llvm::ELF::EF_MIPS_ARCH_ASE_M16)
    return make_dynamic_error_code(Twine("Unsupported extension: MIPS16"));

  // PIC code is inherently CPIC and may not set CPIC flag explicitly.
  // Ensure that this flag will exist in the linked file.
  if (newFlags & llvm::ELF::EF_MIPS_PIC)
    newFlags |= llvm::ELF::EF_MIPS_CPIC;

  std::lock_guard<std::mutex> lock(_mutex);

  // If the old set of flags is empty, use the new one as a result.
  if (!_flags) {
    _flags = newFlags;
    return std::error_code();
  }

  // Check PIC / CPIC flags compatibility.
  uint32_t newPic =
      newFlags & (llvm::ELF::EF_MIPS_PIC | llvm::ELF::EF_MIPS_CPIC);
  uint32_t oldPic = _flags & (llvm::ELF::EF_MIPS_PIC | llvm::ELF::EF_MIPS_CPIC);

  if ((newPic != 0) != (oldPic != 0))
    llvm::errs() << "lld warning: linking abicalls and non-abicalls files\n";

  if (!(newPic & llvm::ELF::EF_MIPS_PIC))
    _flags &= ~llvm::ELF::EF_MIPS_PIC;
  if (newPic)
    _flags |= llvm::ELF::EF_MIPS_CPIC;

  // Check mixing -mnan=2008 / -mnan=legacy modules.
  if ((newFlags & llvm::ELF::EF_MIPS_NAN2008) !=
      (_flags & llvm::ELF::EF_MIPS_NAN2008))
    return make_dynamic_error_code(
        Twine("Linking -mnan=2008 and -mnan=legacy modules"));

  // Set the "largest" ISA.
  uint32_t oldArch = _flags & llvm::ELF::EF_MIPS_ARCH;
  _flags |= std::max(newArch, oldArch);

  _flags |= newFlags & llvm::ELF::EF_MIPS_NOREORDER;
  _flags |= newFlags & llvm::ELF::EF_MIPS_MICROMIPS;
  _flags |= newFlags & llvm::ELF::EF_MIPS_NAN2008;

  return std::error_code();
}
