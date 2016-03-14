//===--- DebugLineTableRowRef.h - Identifies a row in a .debug_line table -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Class that references a row in a DWARFDebugLine::LineTable by the DWARF
// Context index of the DWARF Compile Unit that owns the Line Table and the row
// index. This is tied to our IR during disassembly so that we can later update
// .debug_line information. The RowIndex has a base of 1, which means a RowIndex
// of 1 maps to the first row of the line table and a RowIndex of 0 is invalid.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_DEBUGLINETABLEROWREF_H
#define LLVM_TOOLS_LLVM_BOLT_DEBUGLINETABLEROWREF_H

#include "llvm/DebugInfo/DWARF/DWARFCompileUnit.h"
#include "llvm/Support/SMLoc.h"

namespace llvm {
namespace bolt {

struct DebugLineTableRowRef {
  uint32_t DwCompileUnitIndex;
  uint32_t RowIndex;

  const static DebugLineTableRowRef NULL_ROW;

  bool operator==(const DebugLineTableRowRef &Rhs) const {
    return DwCompileUnitIndex == Rhs.DwCompileUnitIndex &&
      RowIndex == Rhs.RowIndex;
  }

  bool operator!=(const DebugLineTableRowRef &Rhs) const {
    return !(*this == Rhs);
  }

  static DebugLineTableRowRef fromSMLoc(const SMLoc &Loc) {
    union {
      decltype(Loc.getPointer()) Ptr;
      DebugLineTableRowRef Ref;
    } U;
    U.Ptr = Loc.getPointer();
    return U.Ref;
  }

  SMLoc toSMLoc() const {
    union {
      decltype(SMLoc().getPointer()) Ptr;
      DebugLineTableRowRef Ref;
    } U;
    U.Ref = *this;
    return SMLoc::getFromPointer(U.Ptr);
  }
};

} // namespace bolt
} // namespace llvm

#endif
