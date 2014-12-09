//===- tools/dsymutil/DwarfLinker.cpp - Dwarf debug info linker -----------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "DwarfLinker.h"
#include "DebugMap.h"

namespace llvm {

DwarfLinker::DwarfLinker(StringRef OutputFilename)
  : OutputFilename(OutputFilename)
{}

bool DwarfLinker::link(const DebugMap &Map) {
  return true;
}

}
