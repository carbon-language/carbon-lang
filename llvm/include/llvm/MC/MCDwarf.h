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

    /// getDirIndex - Get the dirIndex of this MCDwarfFile.
    unsigned getDirIndex() const { return DirIndex; }


    /// print - Print the value to the stream \arg OS.
    void print(raw_ostream &OS) const;

    /// dump - Print the value to stderr.
    void dump() const;
  };

  /// MCDwarfLoc - Instances of this class represent the information from a
  /// dwarf .loc directive.
  class MCDwarfLoc {
    // FileNum - the file number.
    unsigned FileNum;
    // Line - the line number.
    unsigned Line;
    // Column - the column position.
    unsigned Column;
    // Flags (see #define's below)
    unsigned Flags;
    // Isa
    unsigned Isa;

#define DWARF2_FLAG_IS_STMT        (1 << 0)
#define DWARF2_FLAG_BASIC_BLOCK    (1 << 1)
#define DWARF2_FLAG_PROLOGUE_END   (1 << 2)
#define DWARF2_FLAG_EPILOGUE_BEGIN (1 << 3)

  private:  // MCContext manages these
    friend class MCContext;
    MCDwarfLoc(unsigned fileNum, unsigned line, unsigned column, unsigned flags,
               unsigned isa)
      : FileNum(fileNum), Line(line), Column(column), Flags(flags), Isa(isa) {}

    MCDwarfLoc(const MCDwarfLoc&);       // DO NOT IMPLEMENT
    void operator=(const MCDwarfLoc&); // DO NOT IMPLEMENT
  public:
    /// setFileNum - Set the FileNum of this MCDwarfLoc.
    void setFileNum(unsigned fileNum) { FileNum = fileNum; }

    /// setLine - Set the Line of this MCDwarfLoc.
    void setLine(unsigned line) { Line = line; }

    /// setColumn - Set the Column of this MCDwarfLoc.
    void setColumn(unsigned column) { Column = column; }

    /// setFlags - Set the Flags of this MCDwarfLoc.
    void setFlags(unsigned flags) { Flags = flags; }

    /// setIsa - Set the Isa of this MCDwarfLoc.
    void setIsa(unsigned isa) { Isa = isa; }
  };

  inline raw_ostream &operator<<(raw_ostream &OS, const MCDwarfFile &DwarfFile){
    DwarfFile.print(OS);
    return OS;
  }
} // end namespace llvm

#endif
