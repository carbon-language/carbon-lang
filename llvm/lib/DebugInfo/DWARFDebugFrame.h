//===-- DWARFDebugFrame.h - Parsing of .debug_frame -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_DWARFDEBUGFRAME_H
#define LLVM_DEBUGINFO_DWARFDEBUGFRAME_H

#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/raw_ostream.h"
#include <vector>


namespace llvm {

class FrameEntry;


/// \brief A parsed .debug_frame section
///
class DWARFDebugFrame {
public:
  DWARFDebugFrame();
  ~DWARFDebugFrame();

  /// \brief Dump the section data into the given stream.
  void dump(raw_ostream &OS) const;

  /// \brief Parse the section from raw data.
  /// data is assumed to be pointing to the beginning of the section.
  void parse(DataExtractor Data);

private:
  typedef std::vector<FrameEntry *> EntryVector;
  EntryVector Entries;
};


} // namespace llvm

#endif
