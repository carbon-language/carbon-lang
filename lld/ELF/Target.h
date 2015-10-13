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
  uint64_t getVAStart() const { return VAStart; }
  unsigned getPCRelReloc() const { return PCRelReloc; }
  unsigned getGotReloc() const { return GotReloc; }
  unsigned getPltReloc() const { return PltReloc; }
  unsigned getGotRefReloc() const { return GotRefReloc; }
  unsigned getRelativeReloc() const { return RelativeReloc; }
  unsigned getPltZeroEntrySize() const { return PltZeroEntrySize; }
  unsigned getPltEntrySize() const { return PltEntrySize; }
  virtual void writeGotPltEntry(uint8_t *Buf, uint64_t Plt) const = 0;
  virtual void writePltZeroEntry(uint8_t *Buf, uint64_t GotEntryAddr,
                              uint64_t PltEntryAddr) const = 0;
  virtual void writePltEntry(uint8_t *Buf, uint64_t GotEntryAddr,
                             uint64_t PltEntryAddr, int32_t Index) const = 0;
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
  uint64_t VAStart;
  unsigned PCRelReloc;
  unsigned GotRefReloc;
  unsigned GotReloc;
  unsigned PltReloc;
  unsigned RelativeReloc;
  unsigned PltEntrySize = 8;
  unsigned PltZeroEntrySize = 16;
  llvm::StringRef DefaultEntry = "_start";
};

class X86TargetInfo final : public TargetInfo {
public:
  X86TargetInfo();
  void writeGotPltEntry(uint8_t *Buf, uint64_t Plt) const override;
  void writePltZeroEntry(uint8_t *Buf, uint64_t GotEntryAddr,
                      uint64_t PltEntryAddr) const override;
  void writePltEntry(uint8_t *Buf, uint64_t GotEntryAddr,
                     uint64_t PltEntryAddr, int32_t Index) const override;
  bool relocNeedsGot(uint32_t Type, const SymbolBody &S) const override;
  bool relocPointsToGot(uint32_t Type) const override;
  bool relocNeedsPlt(uint32_t Type, const SymbolBody &S) const override;
  void relocateOne(uint8_t *Buf, uint8_t *BufEnd, const void *RelP,
                   uint32_t Type, uint64_t BaseAddr,
                   uint64_t SymVA) const override;
};

class X86_64TargetInfo final : public TargetInfo {
public:
  X86_64TargetInfo();
  void writeGotPltEntry(uint8_t *Buf, uint64_t Plt) const override;
  void writePltZeroEntry(uint8_t *Buf, uint64_t GotEntryAddr,
                      uint64_t PltEntryAddr) const override;
  void writePltEntry(uint8_t *Buf, uint64_t GotEntryAddr,
                     uint64_t PltEntryAddr, int32_t Index) const override;
  bool relocNeedsGot(uint32_t Type, const SymbolBody &S) const override;
  bool relocNeedsPlt(uint32_t Type, const SymbolBody &S) const override;
  void relocateOne(uint8_t *Buf, uint8_t *BufEnd, const void *RelP,
                   uint32_t Type, uint64_t BaseAddr,
                   uint64_t SymVA) const override;
  bool isRelRelative(uint32_t Type) const override;
};

class PPC64TargetInfo final : public TargetInfo {
public:
  PPC64TargetInfo();
  void writeGotPltEntry(uint8_t *Buf, uint64_t Plt) const override;
  void writePltZeroEntry(uint8_t *Buf, uint64_t GotEntryAddr,
                      uint64_t PltEntryAddr) const override;
  void writePltEntry(uint8_t *Buf, uint64_t GotEntryAddr,
                     uint64_t PltEntryAddr, int32_t Index) const override;
  bool relocNeedsGot(uint32_t Type, const SymbolBody &S) const override;
  bool relocNeedsPlt(uint32_t Type, const SymbolBody &S) const override;
  void relocateOne(uint8_t *Buf, uint8_t *BufEnd, const void *RelP,
                   uint32_t Type, uint64_t BaseAddr,
                   uint64_t SymVA) const override;
  bool isRelRelative(uint32_t Type) const override;
};

class PPCTargetInfo final : public TargetInfo {
public:
  PPCTargetInfo();
  void writeGotPltEntry(uint8_t *Buf, uint64_t Plt) const override;
  void writePltZeroEntry(uint8_t *Buf, uint64_t GotEntryAddr,
                      uint64_t PltEntryAddr) const override;
  void writePltEntry(uint8_t *Buf, uint64_t GotEntryAddr,
                     uint64_t PltEntryAddr, int32_t Index) const override;
  bool relocNeedsGot(uint32_t Type, const SymbolBody &S) const override;
  bool relocNeedsPlt(uint32_t Type, const SymbolBody &S) const override;
  void relocateOne(uint8_t *Buf, uint8_t *BufEnd, const void *RelP,
                   uint32_t Type, uint64_t BaseAddr,
                   uint64_t SymVA) const override;
};

class AArch64TargetInfo final : public TargetInfo {
public:
  AArch64TargetInfo();
  void writeGotPltEntry(uint8_t *Buf, uint64_t Plt) const override;
  void writePltZeroEntry(uint8_t *Buf, uint64_t GotEntryAddr,
                      uint64_t PltEntryAddr) const override;
  void writePltEntry(uint8_t *Buf, uint64_t GotEntryAddr,
                     uint64_t PltEntryAddr, int32_t Index) const override;
  bool relocNeedsGot(uint32_t Type, const SymbolBody &S) const override;
  bool relocNeedsPlt(uint32_t Type, const SymbolBody &S) const override;
  void relocateOne(uint8_t *Buf, uint8_t *BufEnd, const void *RelP,
                   uint32_t Type, uint64_t BaseAddr,
                   uint64_t SymVA) const override;
};

class MipsTargetInfo final : public TargetInfo {
public:
  MipsTargetInfo();
  void writeGotPltEntry(uint8_t *Buf, uint64_t Plt) const override;
  void writePltZeroEntry(uint8_t *Buf, uint64_t GotEntryAddr,
                      uint64_t PltEntryAddr) const override;
  void writePltEntry(uint8_t *Buf, uint64_t GotEntryAddr,
                     uint64_t PltEntryAddr, int32_t Index) const override;
  bool relocNeedsGot(uint32_t Type, const SymbolBody &S) const override;
  bool relocNeedsPlt(uint32_t Type, const SymbolBody &S) const override;
  void relocateOne(uint8_t *Buf, uint8_t *BufEnd, const void *RelP,
                   uint32_t Type, uint64_t BaseAddr,
                   uint64_t SymVA) const override;
};

extern std::unique_ptr<TargetInfo> Target;
TargetInfo *createTarget();
}
}

#endif
