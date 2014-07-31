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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Object/RelocVisitor.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/DataTypes.h"

#include <string>

namespace llvm {

class raw_ostream;

/// DILineInfo - a format-neutral container for source line information.
struct DILineInfo {
  std::string FileName;
  std::string FunctionName;
  uint32_t Line;
  uint32_t Column;

  DILineInfo()
      : FileName("<invalid>"), FunctionName("<invalid>"), Line(0), Column(0) {}

  bool operator==(const DILineInfo &RHS) const {
    return Line == RHS.Line && Column == RHS.Column &&
           FileName == RHS.FileName && FunctionName == RHS.FunctionName;
  }
  bool operator!=(const DILineInfo &RHS) const {
    return !(*this == RHS);
  }
};

typedef SmallVector<std::pair<uint64_t, DILineInfo>, 16> DILineInfoTable;

/// DIInliningInfo - a format-neutral container for inlined code description.
class DIInliningInfo {
  SmallVector<DILineInfo, 4> Frames;
 public:
  DIInliningInfo() {}
  DILineInfo getFrame(unsigned Index) const {
    assert(Index < Frames.size());
    return Frames[Index];
  }
  uint32_t getNumberOfFrames() const {
    return Frames.size();
  }
  void addFrame(const DILineInfo &Frame) {
    Frames.push_back(Frame);
  }
};

/// DILineInfoSpecifier - controls which fields of DILineInfo container
/// should be filled with data.
struct DILineInfoSpecifier {
  enum class FileLineInfoKind { None, Default, AbsoluteFilePath };
  enum class FunctionNameKind { None, ShortName, LinkageName };

  FileLineInfoKind FLIKind;
  FunctionNameKind FNKind;

  DILineInfoSpecifier(FileLineInfoKind FLIKind = FileLineInfoKind::Default,
                      FunctionNameKind FNKind = FunctionNameKind::None)
      : FLIKind(FLIKind), FNKind(FNKind) {}
};

/// Selects which debug sections get dumped.
enum DIDumpType {
  DIDT_Null,
  DIDT_All,
  DIDT_Abbrev,
  DIDT_AbbrevDwo,
  DIDT_Aranges,
  DIDT_Frames,
  DIDT_Info,
  DIDT_InfoDwo,
  DIDT_Types,
  DIDT_TypesDwo,
  DIDT_Line,
  DIDT_LineDwo,
  DIDT_Loc,
  DIDT_LocDwo,
  DIDT_Ranges,
  DIDT_Pubnames,
  DIDT_Pubtypes,
  DIDT_GnuPubnames,
  DIDT_GnuPubtypes,
  DIDT_Str,
  DIDT_StrDwo,
  DIDT_StrOffsetsDwo
};

// In place of applying the relocations to the data we've read from disk we use
// a separate mapping table to the side and checking that at locations in the
// dwarf where we expect relocated values. This adds a bit of complexity to the
// dwarf parsing/extraction at the benefit of not allocating memory for the
// entire size of the debug info sections.
typedef DenseMap<uint64_t, std::pair<uint8_t, int64_t> > RelocAddrMap;

class DIContext {
public:
  enum DIContextKind {
    CK_DWARF
  };
  DIContextKind getKind() const { return Kind; }

  DIContext(DIContextKind K) : Kind(K) {}
  virtual ~DIContext();

  /// getDWARFContext - get a context for binary DWARF data.
  static DIContext *getDWARFContext(object::ObjectFile &);

  virtual void dump(raw_ostream &OS, DIDumpType DumpType = DIDT_All) = 0;

  virtual DILineInfo getLineInfoForAddress(uint64_t Address,
      DILineInfoSpecifier Specifier = DILineInfoSpecifier()) = 0;
  virtual DILineInfoTable getLineInfoForAddressRange(uint64_t Address,
      uint64_t Size, DILineInfoSpecifier Specifier = DILineInfoSpecifier()) = 0;
  virtual DIInliningInfo getInliningInfoForAddress(uint64_t Address,
      DILineInfoSpecifier Specifier = DILineInfoSpecifier()) = 0;
private:
  const DIContextKind Kind;
};

}

#endif
