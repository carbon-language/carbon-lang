//===-- DWARFCompileUnit.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_DWARFCOMPILEUNIT_H
#define LLVM_DEBUGINFO_DWARFCOMPILEUNIT_H

#include "DWARFUnit.h"

namespace llvm {

class DWARFCompileUnit : public DWARFUnit {
public:
  DWARFCompileUnit(const DWARFDebugAbbrev *DA, StringRef IS, StringRef AS,
                   StringRef RS, StringRef SS, StringRef SOS, StringRef AOS,
                   const RelocAddrMap *M, bool LE)
      : DWARFUnit(DA, IS, AS, RS, SS, SOS, AOS, M, LE) {}
  void dump(raw_ostream &OS);
  // VTable anchor.
  ~DWARFCompileUnit() override;
};

}

#endif
