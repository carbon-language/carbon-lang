//===-- DWARFTypeUnit.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_DEBUGINFO_DWARFTYPEUNIT_H
#define LLVM_LIB_DEBUGINFO_DWARFTYPEUNIT_H

#include "DWARFUnit.h"

namespace llvm {

class DWARFTypeUnit : public DWARFUnit {
private:
  uint64_t TypeHash;
  uint32_t TypeOffset;
public:
  DWARFTypeUnit(DWARFContext &Context, const DWARFDebugAbbrev *DA,
                StringRef IS, StringRef RS, StringRef SS, StringRef SOS,
                StringRef AOS, const RelocAddrMap *M, bool LE,
                const DWARFUnitSectionBase &UnitSection)
    : DWARFUnit(Context, DA, IS, RS, SS, SOS, AOS, M, LE, UnitSection) {}
  uint32_t getHeaderSize() const override {
    return DWARFUnit::getHeaderSize() + 12;
  }
  void dump(raw_ostream &OS);
protected:
  bool extractImpl(DataExtractor debug_info, uint32_t *offset_ptr) override;
};

}

#endif

