//===-- StringTableBuilder.h - String table building utility ------*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_STRINGTABLEBUILDER_H
#define LLVM_MC_STRINGTABLEBUILDER_H

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/DenseMap.h"
#include <cassert>

namespace llvm {

/// \brief Utility for building string tables with deduplicated suffixes.
class StringTableBuilder {
  SmallString<256> StringTable;
  DenseMap<StringRef, size_t> StringIndexMap;

public:
  /// \brief Add a string to the builder. Returns a StringRef to the internal
  /// copy of s. Can only be used before the table is finalized.
  void add(StringRef S);

  enum Kind {
    ELF,
    WinCOFF,
    MachO
  };

  /// \brief Analyze the strings and build the final table. No more strings can
  /// be added after this point.
  void finalize(Kind K);

  /// \brief Retrieve the string table data. Can only be used after the table
  /// is finalized.
  StringRef data() const {
    assert(isFinalized());
    return StringTable;
  }

  /// \brief Get the offest of a string in the string table. Can only be used
  /// after the table is finalized.
  size_t getOffset(StringRef S) const;

  void clear();

private:
  bool isFinalized() const {
    return !StringTable.empty();
  }
};

} // end llvm namespace

#endif
