//===-- DIContext.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines DIContext, an abstract data structure that holds
// debug information data.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_DICONTEXT_H
#define LLVM_DEBUGINFO_DICONTEXT_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/DataTypes.h"
#include <cstring>

namespace llvm {

class raw_ostream;

/// DILineInfo - a format-neutral container for source line information.
class DILineInfo {
  const char *FileName;
  const char *FunctionName;
  uint32_t Line;
  uint32_t Column;
public:
  DILineInfo()
    : FileName("<invalid>"), FunctionName("<invalid>"),
      Line(0), Column(0) {}
  DILineInfo(const char *fileName, const char *functionName,
             uint32_t line, uint32_t column)
    : FileName(fileName), FunctionName(functionName),
      Line(line), Column(column) {}

  const char *getFileName() const { return FileName; }
  const char *getFunctionName() const { return FunctionName; }
  uint32_t getLine() const { return Line; }
  uint32_t getColumn() const { return Column; }

  bool operator==(const DILineInfo &RHS) const {
    return Line == RHS.Line && Column == RHS.Column &&
           std::strcmp(FileName, RHS.FileName) == 0 &&
           std::strcmp(FunctionName, RHS.FunctionName) == 0;
  }
  bool operator!=(const DILineInfo &RHS) const {
    return !(*this == RHS);
  }
};

/// DILineInfoSpecifier - controls which fields of DILineInfo container
/// should be filled with data.
class DILineInfoSpecifier {
  const uint32_t Flags;  // Or'ed flags that set the info we want to fetch.
public:
  enum Specification {
    FileLineInfo = 1 << 0,
    FunctionName = 1 << 1
  };
  // Use file/line info by default.
  DILineInfoSpecifier(uint32_t flags = FileLineInfo) : Flags(flags) {}
  bool needs(Specification spec) const {
    return (Flags & spec) > 0;
  }
};

class DIContext {
public:
  virtual ~DIContext();

  /// getDWARFContext - get a context for binary DWARF data.
  static DIContext *getDWARFContext(bool isLittleEndian,
                                    StringRef infoSection,
                                    StringRef abbrevSection,
                                    StringRef aRangeSection = StringRef(),
                                    StringRef lineSection = StringRef(),
                                    StringRef stringSection = StringRef());

  virtual void dump(raw_ostream &OS) = 0;

  virtual DILineInfo getLineInfoForAddress(uint64_t address,
      DILineInfoSpecifier specifier = DILineInfoSpecifier()) = 0;
};

}

#endif
