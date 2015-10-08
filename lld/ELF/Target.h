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
  llvm::StringRef getDefaultEntry() const { return DefaultEntry; }
  unsigned getPageSize() const { return PageSize; }
  uint64_t getVAStart() const { return VAStart; }
  unsigned getPCRelReloc() const { return PCRelReloc; }
  unsigned getGotReloc() const { return GotReloc; }
  unsigned getGotRefReloc() const { return GotRefReloc; }
  unsigned getRelativeReloc() const { return RelativeReloc; }
  unsigned getPltEntrySize() const { return PltEntrySize; }
  virtual void writePltEntry(uint8_t *Buf, uint64_t GotEntryAddr,
                             uint64_t PltEntryAddr) const = 0;
  virtual bool isRelRelative(uint32_t Type) const;
  virtual bool relocNeedsGot(uint32_t Type, const SymbolBody &S) const = 0;
  virtual bool relocPointsToGot(uint32_t Type) const;
  virtual bool relocNeedsPlt(uint32_t Type, const SymbolBody &S) const = 0;
  virtual void relocateOne(uint8_t *Buf, const void *RelP, uint32_t Type,
                           uint64_t BaseAddr, uint64_t SymVA) const = 0;

  virtual ~TargetInfo();

protected:
  unsigned PageSize = 4096;
  uint64_t VAStart;
  unsigned PCRelReloc;
  unsigned GotRefReloc;
  unsigned GotReloc;
  unsigned RelativeReloc;
  unsigned PltEntrySize = 8;
  llvm::StringRef DefaultEntry = "_start";
};

class X86TargetInfo final : public TargetInfo {
public:
  X86TargetInfo();
  void writePltEntry(uint8_t *Buf, uint64_t GotEntryAddr,
                     uint64_t PltEntryAddr) const override;
  bool relocNeedsGot(uint32_t Type, const SymbolBody &S) const override;
  bool relocPointsToGot(uint32_t Type) const override;
  bool relocNeedsPlt(uint32_t Type, const SymbolBody &S) const override;
  void relocateOne(uint8_t *Buf, const void *RelP, uint32_t Type,
                   uint64_t BaseAddr, uint64_t SymVA) const override;
};

class X86_64TargetInfo final : public TargetInfo {
public:
  X86_64TargetInfo();
  void writePltEntry(uint8_t *Buf, uint64_t GotEntryAddr,
                     uint64_t PltEntryAddr) const override;
  bool relocNeedsGot(uint32_t Type, const SymbolBody &S) const override;
  bool relocNeedsPlt(uint32_t Type, const SymbolBody &S) const override;
  void relocateOne(uint8_t *Buf, const void *RelP, uint32_t Type,
                   uint64_t BaseAddr, uint64_t SymVA) const override;
  bool isRelRelative(uint32_t Type) const override;
};

class PPC64TargetInfo final : public TargetInfo {
public:
  PPC64TargetInfo();
  void writePltEntry(uint8_t *Buf, uint64_t GotEntryAddr,
                     uint64_t PltEntryAddr) const override;
  bool relocNeedsGot(uint32_t Type, const SymbolBody &S) const override;
  bool relocNeedsPlt(uint32_t Type, const SymbolBody &S) const override;
  void relocateOne(uint8_t *Buf, const void *RelP, uint32_t Type,
                   uint64_t BaseAddr, uint64_t SymVA) const override;
};

class PPCTargetInfo final : public TargetInfo {
public:
  PPCTargetInfo();
  void writePltEntry(uint8_t *Buf, uint64_t GotEntryAddr,
                     uint64_t PltEntryAddr) const override;
  bool relocNeedsGot(uint32_t Type, const SymbolBody &S) const override;
  bool relocNeedsPlt(uint32_t Type, const SymbolBody &S) const override;
  void relocateOne(uint8_t *Buf, const void *RelP, uint32_t Type,
                   uint64_t BaseAddr, uint64_t SymVA) const override;
};

class ARMTargetInfo final : public TargetInfo {
public:
  ARMTargetInfo();
  void writePltEntry(uint8_t *Buf, uint64_t GotEntryAddr,
                     uint64_t PltEntryAddr) const override;
  bool relocNeedsGot(uint32_t Type, const SymbolBody &S) const override;
  bool relocNeedsPlt(uint32_t Type, const SymbolBody &S) const override;
  void relocateOne(uint8_t *Buf, const void *RelP, uint32_t Type,
                   uint64_t BaseAddr, uint64_t SymVA) const override;
};

class AArch64TargetInfo final : public TargetInfo {
public:
  AArch64TargetInfo();
  void writePltEntry(uint8_t *Buf, uint64_t GotEntryAddr,
                     uint64_t PltEntryAddr) const override;
  bool relocNeedsGot(uint32_t Type, const SymbolBody &S) const override;
  bool relocNeedsPlt(uint32_t Type, const SymbolBody &S) const override;
  void relocateOne(uint8_t *Buf, const void *RelP, uint32_t Type,
                   uint64_t BaseAddr, uint64_t SymVA) const override;
};

class MipsTargetInfo final : public TargetInfo {
public:
  MipsTargetInfo();
  void writePltEntry(uint8_t *Buf, uint64_t GotEntryAddr,
                     uint64_t PltEntryAddr) const override;
  bool relocNeedsGot(uint32_t Type, const SymbolBody &S) const override;
  bool relocNeedsPlt(uint32_t Type, const SymbolBody &S) const override;
  void relocateOne(uint8_t *Buf, const void *RelP, uint32_t Type,
                   uint64_t BaseAddr, uint64_t SymVA) const override;
};

extern std::unique_ptr<TargetInfo> Target;
}
}

#endif
