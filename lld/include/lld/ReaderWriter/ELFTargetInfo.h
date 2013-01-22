//===- lld/ReaderWriter/ELFTargetInfo.h -----------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_TARGET_INFO_H
#define LLD_READER_WRITER_ELF_TARGET_INFO_H

#include "lld/Core/TargetInfo.h"

#include <memory>

namespace lld {
class ELFTargetInfo : public TargetInfo {
protected:
  ELFTargetInfo(const LinkerOptions &lo) : TargetInfo(lo) {}

public:
  uint16_t getOutputType() const;
  uint16_t getOutputMachine() const;

  static std::unique_ptr<ELFTargetInfo> create(const LinkerOptions &lo);
};
} // end namespace lld

#endif
