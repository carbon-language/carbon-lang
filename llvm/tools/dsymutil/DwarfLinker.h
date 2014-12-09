//===- tools/dsymutil/DwarfLinker.h - Dwarf debug info linker -------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
///
/// This file contains the class declaration of the DwarfLinker
/// object. A DwarfLinker takes a DebugMap as input and links the
/// debug information of all the referenced object files together. It
/// may drop and rewrite some parts of the debug info tree in the
/// process.
///
//===----------------------------------------------------------------------===//
#ifndef DSYMUTIL_DWARFLINKER_H
#define DSYMUTIL_DWARFLINKER_H

#include "llvm/ADT/StringRef.h"

namespace llvm {

class DebugMap;

class DwarfLinker {
  std::string OutputFilename;
public:
  DwarfLinker(StringRef OutputFilename);

  /// \brief Link the passed debug map into the ouptut file.
  /// \returns false if the link encountered a fatal error.
  bool link(const DebugMap&);
};

}

#endif
