//===-- Reader/DataReader.h - Perf data reader ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This family of functions reads profile data written by the perf2bolt
// utility and stores it in memory for llvm-bolt consumption.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_DATA_READER_H
#define LLVM_TOOLS_LLVM_BOLT_DATA_READER_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
namespace bolt {

struct Location {
  bool IsSymbol;
  StringRef Name;
  uint64_t Offset;

  Location(bool IsSymbol, StringRef Name, uint64_t Offset)
      : IsSymbol(IsSymbol), Name(Name), Offset(Offset) {}

  bool operator==(const Location &RHS) const {
    return IsSymbol == RHS.IsSymbol &&
           Name == RHS.Name &&
           (Name == "[heap]" || Offset == RHS.Offset);
  }

  bool operator<(const Location &RHS) const {
    if (IsSymbol < RHS.IsSymbol)
      return true;

    if (Name < RHS.Name)
      return true;

    return IsSymbol == RHS.IsSymbol &&
           Name == RHS.Name &&
           Name != "[heap]" &&
           Offset < RHS.Offset;
  }
};

struct BranchInfo {
  Location From;
  Location To;
  int64_t Mispreds;
  int64_t Branches;

  BranchInfo(Location From, Location To, int64_t Mispreds, int64_t Branches)
      : From(std::move(From)), To(std::move(To)), Mispreds(Mispreds),
        Branches(Branches) {}

  bool operator==(const BranchInfo &RHS) const {
    return From == RHS.From &&
           To == RHS.To;
  }

  bool operator<(const BranchInfo &RHS) const {
    if (From < RHS.From)
      return true;

    if (From == RHS.From)
      return (To < RHS.To);

    return false;
  }
};

struct FuncBranchData {
  typedef std::vector<BranchInfo> ContainerTy;

  StringRef Name;
  ContainerTy Data;

  /// Total execution count for the function.
  int64_t ExecutionCount{0};

  FuncBranchData(StringRef Name, ContainerTy Data)
      : Name(Name), Data(std::move(Data)) {}

  ErrorOr<const BranchInfo &> getBranch(uint64_t From, uint64_t To) const;
};

//===----------------------------------------------------------------------===//
//
/// DataReader Class
///
class DataReader {
public:
  explicit DataReader(raw_ostream &Diag) : Diag(Diag) {}

  DataReader(std::unique_ptr<MemoryBuffer> MemBuf, raw_ostream &Diag)
      : FileBuf(std::move(MemBuf)), Diag(Diag), ParsingBuf(FileBuf->getBuffer()),
        Line(0), Col(0) {}

  static ErrorOr<std::unique_ptr<DataReader>> readPerfData(StringRef Path,
                                                           raw_ostream &Diag);

  /// Parses the input bolt data file into internal data structures. We expect
  /// the file format to follow the syntax below.
  ///
  /// <is symbol?> <closest elf symbol or DSO name> <relative FROM address>
  /// <is symbol?> <closest elf symbol or DSO name> <relative TO address>
  /// <number of mispredictions> <number of branches>
  ///
  /// In <is symbol?> field we record 0 if our closest address is a DSO load
  /// address or 1 if our closest address is an ELF symbol.
  ///
  /// Example:
  ///
  ///  1 main 3fb 0 /lib/ld-2.21.so 12 4 221
  ///
  /// The example records branches from symbol main, offset 3fb, to DSO ld-2.21,
  /// offset 12, with 4 mispredictions and 221 branches
  std::error_code parse();

  ErrorOr<const FuncBranchData &> getFuncBranchData(StringRef FuncName) const;

  using FuncsMapType = StringMap<FuncBranchData>;

  FuncsMapType &getAllFuncsData() { return FuncsMap; }

  /// Dumps the entire data structures parsed. Used for debugging.
  void dump() const;

private:

  void reportError(StringRef ErrorMsg);
  bool expectAndConsumeFS();
  ErrorOr<StringRef> parseString(char EndChar);
  ErrorOr<int64_t> parseNumberField(char EndChar);
  ErrorOr<Location> parseLocation();
  ErrorOr<BranchInfo> parseBranchInfo();
  bool hasData();

  // An in-memory copy of the input data file - owns strings used in reader
  std::unique_ptr<MemoryBuffer> FileBuf;
  raw_ostream &Diag;
  StringRef ParsingBuf;
  unsigned Line;
  unsigned Col;
  FuncsMapType FuncsMap;
  static const char FieldSeparator = ' ';
};



}
}

#endif
