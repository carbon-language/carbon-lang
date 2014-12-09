//===- tools/dsymutil/MachODebugMapParser.h - Parse STABS debug maps ------===//
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
/// This file contains the class declaration for the code that parses STABS
/// debug maps that are embedded in the binaries symbol tables.
///
//===----------------------------------------------------------------------===//
#ifndef DSYMUTIL_MACHODEBUGMAPPARSER_H
#define DSYMUTIL_MACHODEBUGMAPPARSER_H

#include "DebugMap.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/Object/MachO.h"
#include "llvm/Object/Error.h"

namespace llvm {

class MachODebugMapParser {
public:
 MachODebugMapParser(StringRef BinaryPath)
   : BinaryPath(BinaryPath) {}

  /// \brief Add a prefix to every object file path before trying to
  /// open it.
  void setPreprendPath(StringRef Prefix) { PathPrefix = Prefix; }

  /// \brief Parses and returns the DebugMap of the input binary.
  /// \returns an error in case the provided BinaryPath doesn't exist
  /// or isn't of a supported type.
  ErrorOr<std::unique_ptr<DebugMap>> parse();

private:
  std::string BinaryPath;
  std::string PathPrefix;

  /// OwningBinary constructed from the BinaryPath.
  object::OwningBinary<object::MachOObjectFile> MainOwningBinary;
  /// Map of the binary symbol addresses.
  StringMap<uint64_t> MainBinarySymbolAddresses;
  /// The constructed DebugMap.
  std::unique_ptr<DebugMap> Result;

  /// Handle to the currently processed object file.
  object::OwningBinary<object::MachOObjectFile> CurrentObjectFile;
  /// Map of the currently processed object file symbol addresses.
  StringMap<uint64_t> CurrentObjectAddresses;
  /// Element of the debug map corresponfing to the current object file.
  DebugMapObject *CurrentDebugMapObject;

  void switchToNewDebugMapObject(StringRef Filename);
  void resetParserState();
  uint64_t getMainBinarySymbolAddress(StringRef Name);
  void loadMainBinarySymbols();
  void loadCurrentObjectFileSymbols();
  void handleStabSymbolTableEntry(uint32_t StringIndex, uint8_t Type,
                                  uint8_t SectionIndex, uint16_t Flags,
                                  uint64_t Value);

  template <typename STEType> void handleStabDebugMapEntry(const STEType &STE) {
    handleStabSymbolTableEntry(STE.n_strx, STE.n_type, STE.n_sect, STE.n_desc,
                               STE.n_value);
  }
};

}

#endif // DSYMUTIL_MACHODEBUGMAPPARSER_H
