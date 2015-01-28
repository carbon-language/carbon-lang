//===- tools/dsymutil/DwarfLinker.cpp - Dwarf debug info linker -----------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "DebugMap.h"

#include "BinaryHolder.h"
#include "DebugMap.h"
#include "dsymutil.h"
#include "llvm/DebugInfo/DWARFContext.h"
#include "llvm/DebugInfo/DWARFDebugInfoEntry.h"
#include <string>

namespace llvm {
namespace dsymutil {

namespace {

/// \brief The core of the Dwarf linking logic.
class DwarfLinker {
public:
  DwarfLinker(StringRef OutputFilename, bool Verbose)
      : OutputFilename(OutputFilename), Verbose(Verbose), BinHolder(Verbose) {}

  /// \brief Link the contents of the DebugMap.
  bool link(const DebugMap &);

private:
  std::string OutputFilename;
  bool Verbose;
  BinaryHolder BinHolder;
};

bool DwarfLinker::link(const DebugMap &Map) {

  if (Map.begin() == Map.end()) {
    errs() << "Empty debug map.\n";
    return false;
  }

  for (const auto &Obj : Map.objects()) {
    if (Verbose)
      outs() << "DEBUG MAP OBJECT: " << Obj->getObjectFilename() << "\n";
    auto ErrOrObj = BinHolder.GetObjectFile(Obj->getObjectFilename());
    if (std::error_code EC = ErrOrObj.getError()) {
      errs() << Obj->getObjectFilename() << ": " << EC.message() << "\n";
      continue;
    }

    DWARFContextInMemory DwarfContext(*ErrOrObj);

    for (const auto &CU : DwarfContext.compile_units()) {
      auto *CUDie = CU->getCompileUnitDIE(false);
      if (Verbose) {
        outs() << "Input compilation unit:";
        CUDie->dump(outs(), CU.get(), 0);
      }
    }
  }

  return true;
}
}

bool linkDwarf(StringRef OutputFilename, const DebugMap &DM, bool Verbose) {
  DwarfLinker Linker(OutputFilename, Verbose);
  return Linker.link(DM);
}
}
}
