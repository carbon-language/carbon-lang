//===- MCDwarf.h - Machine Code Dwarf support -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the MCDwarfFile to support the dwarf
// .file directive.
// TODO: add the support needed for the .loc directive.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCDWARF_H
#define LLVM_MC_MCDWARF_H

#include "llvm/ADT/StringRef.h"

namespace llvm {
  class MCContext;
  class raw_ostream;

  /// MCDwarfFile - Instances of this class represent the name of the dwarf
  /// .file directive and its associated dwarf file number in the MC file,
  /// and MCDwarfFile's are created and unique'd by the MCContext class where
  /// the file number for each is its index into the vector of DwarfFiles (note
  /// index 0 is not used and not a valid dwarf file number).
  class MCDwarfFile {
    // Name - the base name of the file without its directory path.
    // The StringRef references memory allocated in the MCContext.
    StringRef Name;

    // DirIndex - the index into the list of directory names for this file name.
    unsigned DirIndex;

  private:  // MCContext creates and uniques these.
    friend class MCContext;
    MCDwarfFile(StringRef name, unsigned dirIndex)
      : Name(name), DirIndex(dirIndex) {}

    MCDwarfFile(const MCDwarfFile&);       // DO NOT IMPLEMENT
    void operator=(const MCDwarfFile&); // DO NOT IMPLEMENT
  public:
    /// getName - Get the base name of this MCDwarfFile.
    StringRef getName() const { return Name; }

    /// print - Print the value to the stream \arg OS.
    void print(raw_ostream &OS) const;

    /// dump - Print the value to stderr.
    void dump() const;
  };

  inline raw_ostream &operator<<(raw_ostream &OS, const MCDwarfFile &DwarfFile){
    DwarfFile.print(OS);
    return OS;
  }
} // end namespace llvm

#endif
