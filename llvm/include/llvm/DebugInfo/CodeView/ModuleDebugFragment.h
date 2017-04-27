//===- ModuleDebugFragment.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_MODULEDEBUGFRAGMENT_H
#define LLVM_DEBUGINFO_CODEVIEW_MODULEDEBUGFRAGMENT_H

#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/Support/BinaryStreamArray.h"
#include "llvm/Support/BinaryStreamRef.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"

namespace llvm {
namespace codeview {

// Corresponds to the `CV_DebugSSubsectionHeader_t` structure.
struct ModuleDebugFragmentHeader {
  support::ulittle32_t Kind;   // codeview::ModuleDebugFragmentKind enum
  support::ulittle32_t Length; // number of bytes occupied by this record.
};

// Corresponds to the `CV_DebugSLinesHeader_t` structure.
struct LineFragmentHeader {
  support::ulittle32_t RelocOffset;  // Code offset of line contribution.
  support::ulittle16_t RelocSegment; // Code segment of line contribution.
  support::ulittle16_t Flags;        // See LineFlags enumeration.
  support::ulittle32_t CodeSize;     // Code size of this line contribution.
};

// Corresponds to the `CV_DebugSLinesFileBlockHeader_t` structure.
struct LineBlockFragmentHeader {
  support::ulittle32_t NameIndex; // Index in DBI name buffer of filename.
  support::ulittle32_t NumLines;  // Number of lines
  support::ulittle32_t BlockSize; // Code size of block, in bytes.
  // The following two variable length arrays appear immediately after the
  // header.  The structure definitions follow.
  // LineNumberEntry   Lines[NumLines];
  // ColumnNumberEntry Columns[NumLines];
};

// Corresponds to `CV_Line_t` structure
struct LineNumberEntry {
  support::ulittle32_t Offset; // Offset to start of code bytes for line number
  support::ulittle32_t Flags;  // Start:24, End:7, IsStatement:1
};

// Corresponds to `CV_Column_t` structure
struct ColumnNumberEntry {
  support::ulittle16_t StartColumn;
  support::ulittle16_t EndColumn;
};

class ModuleDebugFragment {
public:
  ModuleDebugFragment();
  ModuleDebugFragment(ModuleDebugFragmentKind Kind, BinaryStreamRef Data);
  static Error initialize(BinaryStreamRef Stream, ModuleDebugFragment &Info);
  uint32_t getRecordLength() const;
  ModuleDebugFragmentKind kind() const;
  BinaryStreamRef getRecordData() const;

private:
  ModuleDebugFragmentKind Kind;
  BinaryStreamRef Data;
};

typedef VarStreamArray<ModuleDebugFragment> ModuleDebugFragmentArray;
} // namespace codeview

template <> struct VarStreamArrayExtractor<codeview::ModuleDebugFragment> {
  typedef void ContextType;

  static Error extract(BinaryStreamRef Stream, uint32_t &Length,
                       codeview::ModuleDebugFragment &Info, void *Ctx) {
    if (auto EC = codeview::ModuleDebugFragment::initialize(Stream, Info))
      return EC;
    Length = Info.getRecordLength();
    return Error::success();
  }
};
} // namespace llvm

#endif // LLVM_DEBUGINFO_CODEVIEW_MODULEDEBUGFRAGMENT_H
