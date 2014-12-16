//===- tools/dsymutil/DwarfLinker.cpp - Dwarf debug info linker -----------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "DebugMap.h"
#include "dsymutil.h"

namespace llvm {
namespace dsymutil {

bool linkDwarf(StringRef OutputFilename, const DebugMap &DM, bool Verbose) {
  // Do nothing for now.
  return true;
}
}
}
