//===- lib/ReaderWriter/ELF/MipsAbiInfoHandler.h --------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLD_READER_WRITER_ELF_MIPS_MIPS_ABI_INFO_HANDLER_H
#define LLD_READER_WRITER_ELF_MIPS_MIPS_ABI_INFO_HANDLER_H

#include "llvm/ADT/Optional.h"
#include "llvm/Object/ELFTypes.h"
#include "llvm/Support/ErrorOr.h"
#include <mutex>
#include <system_error>

namespace lld {
namespace elf {

struct MipsAbiFlags {
  unsigned _isa = 0;
  unsigned _fpAbi = 0;
  unsigned _ases = 0;
  unsigned _flags1 = 0;
  unsigned _gprSize = 0;
  unsigned _cpr1Size = 0;
  unsigned _cpr2Size = 0;

  unsigned _abi = 0;

  bool _isPic = false;
  bool _isCPic = false;
  bool _isNoReorder = false;
  bool _is32BitMode = false;
  bool _isNan2008 = false;
};

template <class ELFT> class MipsAbiInfoHandler {
public:
  typedef llvm::object::Elf_Mips_RegInfo<ELFT> Elf_Mips_RegInfo;
  typedef llvm::object::Elf_Mips_ABIFlags<ELFT> Elf_Mips_ABIFlags;

  MipsAbiInfoHandler() = default;

  bool hasMipsAbiSection() const { return _hasAbiSection; }
  bool isMicroMips() const;
  bool isMipsR6() const;

  uint32_t getFlags() const;
  llvm::Optional<Elf_Mips_RegInfo> getRegistersMask() const;
  llvm::Optional<Elf_Mips_ABIFlags> getAbiFlags() const;

  /// \brief Merge saved ELF header flags and the new set of flags.
  std::error_code mergeFlags(uint32_t newFlags,
                             const Elf_Mips_ABIFlags *newAbi);

  /// \brief Merge saved and new sets of registers usage masks.
  void mergeRegistersMask(const Elf_Mips_RegInfo &info);

private:
  mutable std::mutex _mutex;
  bool _hasAbiSection = false;
  llvm::Optional<MipsAbiFlags> _abiFlags;
  llvm::Optional<Elf_Mips_RegInfo> _regMask;

  llvm::ErrorOr<MipsAbiFlags> createAbiFlags(uint32_t flags,
                                             const Elf_Mips_ABIFlags *sec);
  static llvm::ErrorOr<MipsAbiFlags> createAbiFromHeaderFlags(uint32_t flags);
  static llvm::ErrorOr<MipsAbiFlags>
  createAbiFromSection(const Elf_Mips_ABIFlags &sec);
};

} // namespace elf
} // namespace lld

#endif
