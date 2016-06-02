//===- RawTypes.h -----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_RAW_RAWTYPES_H
#define LLVM_DEBUGINFO_PDB_RAW_RAWTYPES_H

#include "llvm/Support/Endian.h"

namespace llvm {
namespace pdb {
// This struct is defined as "SO" in langapi/include/pdb.h.
struct SectionOffset {
  support::ulittle32_t Off;
  support::ulittle16_t Isect;
  char Padding[2];
};

// This is HRFile.
struct PSHashRecord {
  support::ulittle32_t Off; // Offset in the symbol record stream
  support::ulittle32_t CRef;
};

// This struct is defined as `SC` in include/dbicommon.h
struct SectionContrib {
  support::ulittle16_t ISect;
  char Padding[2];
  support::little32_t Off;
  support::little32_t Size;
  support::ulittle32_t Characteristics;
  support::ulittle16_t Imod;
  char Padding2[2];
  support::ulittle32_t DataCrc;
  support::ulittle32_t RelocCrc;
};

// This struct is defined as `SC2` in include/dbicommon.h
struct SectionContrib2 {
  // To guarantee SectionContrib2 is standard layout, we cannot use inheritance.
  SectionContrib Base;
  support::ulittle32_t ISectCoff;
};

// This corresponds to the `OMFSegMap` structure.
struct SecMapHeader {
  support::ulittle16_t SecCount;    // Number of segment descriptors in table
  support::ulittle16_t SecCountLog; // Number of logical segment descriptors
};

// This corresponds to the `OMFSegMapDesc` structure.  The definition is not
// present in the reference implementation, but the layout is derived from
// code that accesses the fields.
struct SecMapEntry {
  support::ulittle16_t Flags; // Descriptor flags.  See OMFSegDescFlags
  support::ulittle16_t Ovl;   // Logical overlay number.
  support::ulittle16_t Group; // Group index into descriptor array.
  support::ulittle16_t Frame;
  support::ulittle16_t SecName;       // Byte index of the segment or group name
                                      // in the sstSegName table, or 0xFFFF.
  support::ulittle16_t ClassName;     // Byte index of the class name in the
                                      // sstSegName table, or 0xFFFF.
  support::ulittle32_t Offset;        // Byte offset of the logical segment
                                      // within the specified physical segment.
                                      // If group is set in flags, offset is the
                                      // offset of the group.
  support::ulittle32_t SecByteLength; // Byte count of the segment or group.
};

// Corresponds to the `CV_DebugSSubsectionHeader_t` structure.
struct ModuleSubsectionHeader {
  support::ulittle32_t Kind;   // codeview::ModuleSubstreamKind enum
  support::ulittle32_t Length; // number of bytes occupied by this record.
};

// Corresponds to the `CV_DebugSLinesHeader_t` structure.
struct LineTableSubsectionHeader {
  support::ulittle32_t OffCon;
  support::ulittle16_t SegCon;
  support::ulittle16_t Flags;
  support::ulittle32_t CbCon;
};

// Corresponds to the `CV_DebugSLinesFileBlockHeader_t` structure.
struct SourceFileBlockHeader {
  support::ulittle32_t offFile;
  support::ulittle32_t nLines;
  support::ulittle32_t cbBlock;
  // LineInfo      lines[nLines];
  // ColumnInfo    columns[nColumns];
};

// Corresponds to `CV_Line_t` structure
struct LineInfo {
  unsigned long Offset; // Offset to start of code bytes for line number
  unsigned long LinenumStart : 24; // line where statement/expression starts
  unsigned long
      DeltaLineEnd : 7;         // delta to line where statement ends (optional)
  unsigned long FStatement : 1; // true if a statement linenumber, else an
                                // expression line num
};

// Corresponds to `CV_Column_t` structure
struct ColumnInfo {
  support::ulittle16_t OffColumnStart;
  support::ulittle16_t OffColumnEnd;
};

} // namespace pdb
} // namespace llvm

#endif
