//===- Target.h -------------------------------------------------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_TARGET_H
#define LLD_ELF_TARGET_H

#include "llvm/ADT/StringRef.h"

#include <memory>

namespace lld {
namespace elf2 {
class SymbolBody;

class TargetInfo {
public:
  unsigned getPageSize() const { return PageSize; }
  unsigned getPCRelReloc() const { return PCRelReloc; }
  unsigned getGotReloc() const { return GotReloc; }
  unsigned getGotRefReloc() const { return GotRefReloc; }
  unsigned getRelativeReloc() const { return RelativeReloc; }
  unsigned getPltEntrySize() const { return PltEntrySize; }
  virtual unsigned getPLTRefReloc(unsigned Type) const;
  virtual void writePltEntry(uint8_t *Buf, uint64_t GotEntryAddr,
                             uint64_t PltEntryAddr) const = 0;
  virtual bool isRelRelative(uint32_t Type) const;
  virtual bool relocNeedsGot(uint32_t Type, const SymbolBody &S) const = 0;
  virtual bool relocPointsToGot(uint32_t Type) const;
  virtual bool relocNeedsPlt(uint32_t Type, const SymbolBody &S) const = 0;
  virtual void relocateOne(uint8_t *Buf, uint8_t *BufEnd, const void *RelP,
                           uint32_t Type, uint64_t BaseAddr,
                           uint64_t SymVA) const = 0;

  virtual ~TargetInfo();

protected:
  unsigned PageSize = 4096;
  unsigned PCRelReloc;
  unsigned GotRefReloc;
  unsigned GotReloc;
  unsigned RelativeReloc;
  unsigned PltEntrySize = 8;
};

extern std::unique_ptr<TargetInfo> Target;
TargetInfo *createTarget();
}
}

#endif
