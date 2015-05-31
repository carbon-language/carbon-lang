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

#include "MipsReginfo.h"
#include "llvm/ADT/Optional.h"
#include <mutex>
#include <system_error>

namespace lld {
namespace elf {

template <class ELFT> class MipsAbiInfoHandler {
public:
  MipsAbiInfoHandler() = default;

  uint32_t getFlags() const;
  const llvm::Optional<MipsReginfo> &getRegistersMask() const;

  /// \brief Merge saved ELF header flags and the new set of flags.
  std::error_code mergeFlags(uint32_t newFlags);

  /// \brief Merge saved and new sets of registers usage masks.
  void mergeRegistersMask(const MipsReginfo &info);

private:
  std::mutex _mutex;
  uint32_t _flags = 0;
  llvm::Optional<MipsReginfo> _regMask;
};

} // namespace elf
} // namespace lld

#endif
