//===--- lib/CodeGen/DebugLocList.h - DWARF debug_loc list ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_ASMPRINTER_DEBUGLOCLIST_H
#define LLVM_LIB_CODEGEN_ASMPRINTER_DEBUGLOCLIST_H

#include "DebugLocEntry.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {
class DwarfCompileUnit;
class MCSymbol;
struct DebugLocList {
  MCSymbol *Label;
  DwarfCompileUnit *CU;
  SmallVector<DebugLocEntry, 4> List;
};
}
#endif
