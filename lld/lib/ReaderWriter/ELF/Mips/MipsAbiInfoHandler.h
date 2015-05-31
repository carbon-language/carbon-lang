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
#include <mutex>
#include <system_error>

namespace lld {
namespace elf {

template <class ELFT> class MipsAbiInfoHandler {
public:
  typedef llvm::object::Elf_Mips_RegInfo<ELFT> Elf_Mips_RegInfo;

  MipsAbiInfoHandler() = default;

  uint32_t getFlags() const { return _flags; }
  const llvm::Optional<Elf_Mips_RegInfo> &getRegistersMask() const {
    return _regMask;
  }

  /// \brief Merge saved ELF header flags and the new set of flags.
  std::error_code mergeFlags(uint32_t newFlags);

  /// \brief Merge saved and new sets of registers usage masks.
  void mergeRegistersMask(const Elf_Mips_RegInfo &info);

private:
  std::mutex _mutex;
  uint32_t _flags = 0;
  llvm::Optional<Elf_Mips_RegInfo> _regMask;
};

} // namespace elf
} // namespace lld

#endif
