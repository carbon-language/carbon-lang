//===- lld/ReaderWriter/MachOTargetInfo.h ---------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_MACHO_TARGET_INFO_H
#define LLD_READER_WRITER_MACHO_TARGET_INFO_H

#include "lld/Core/TargetInfo.h"

#include <memory>

namespace lld {
class MachOTargetInfo : public TargetInfo {
protected:
  MachOTargetInfo(const LinkerOptions &lo) : TargetInfo(lo) {}

public:
  uint32_t getCPUType() const;
  uint32_t getCPUSubType() const;

  bool addEntryPointLoadCommand() const;
  bool addUnixThreadLoadCommand() const;

  virtual uint64_t getPageZeroSize() const = 0;

  static std::unique_ptr<MachOTargetInfo> create(const LinkerOptions &lo);
};
} // end namespace lld

#endif
