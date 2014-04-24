//===-- DWARFTypeUnit.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_DWARFTYPEUNIT_H
#define LLVM_DEBUGINFO_DWARFTYPEUNIT_H

#include "DWARFUnit.h"

namespace llvm {

class DWARFTypeUnit : public DWARFUnit {
private:
  uint64_t TypeHash;
  uint32_t TypeOffset;
public:
  DWARFTypeUnit(const DWARFDebugAbbrev *DA, StringRef IS, StringRef RS,
                StringRef SS, StringRef SOS, StringRef AOS,
                const RelocAddrMap *M, bool LE)
      : DWARFUnit(DA, IS, RS, SS, SOS, AOS, M, LE) {}
  uint32_t getSize() const override { return DWARFUnit::getSize() + 12; }
  void dump(raw_ostream &OS);
protected:
  bool extractImpl(DataExtractor debug_info, uint32_t *offset_ptr) override;
};

}

#endif

