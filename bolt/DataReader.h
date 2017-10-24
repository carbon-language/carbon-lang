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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <vector>

namespace llvm {
namespace bolt {

/// LTO-generated function names take a form:
//
///   <function_name>.lto_priv.<decimal_number>/...
///     or
///   <function_name>.constprop.<decimal_number>/...
///
/// they can also be:
///
///   <function_name>.lto_priv.<decimal_number1>.lto_priv.<decimal_number2>/...
///
/// The <decimal_number> is a global counter used for the whole program. As a
/// result, a tiny change in a program may affect the naming of many LTO
/// functions. For us this means that if we do a precise name matching, then
/// a large set of functions could be left without a profile.
///
/// To solve this issue, we try to match a function to a regex profile:
///
///   <function_name>.(lto_priv|consprop).*
///
/// The name before an asterisk above represents a common LTO name for a family
/// of functions. Later out of all matching profiles we pick the one with the
/// best match.

/// Return a common part of LTO name for a given \p Name.
Optional<StringRef> getLTOCommonName(const StringRef Name);

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
    if (IsSymbol != RHS.IsSymbol)
      return IsSymbol < RHS.IsSymbol;

    if (Name != RHS.Name)
      return Name < RHS.Name;

    return Name != "[heap]" && Offset < RHS.Offset;
  }
};

typedef std::vector<std::pair<Location, Location>> BranchContext;

struct BranchHistory {
  int64_t Mispreds;
  int64_t Branches;
  BranchContext Context;

  BranchHistory(int64_t Mispreds, int64_t Branches, BranchContext Context)
      : Mispreds(Mispreds), Branches(Branches), Context(std::move(Context)) {}
};

typedef std::vector<BranchHistory> BranchHistories;

struct BranchInfo {
  Location From;
  Location To;
  int64_t Mispreds;
  int64_t Branches;
  BranchHistories Histories;

  BranchInfo(Location From, Location To, int64_t Mispreds, int64_t Branches,
             BranchHistories Histories)
      : From(std::move(From)), To(std::move(To)), Mispreds(Mispreds),
        Branches(Branches), Histories(std::move(Histories)) {}

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

  /// Merges the branch and misprediction counts as well as the histories of BI
  /// with those of this objetc.
  void mergeWith(const BranchInfo &BI);

  void print(raw_ostream &OS) const;
};

struct FuncBranchData {
  typedef std::vector<BranchInfo> ContainerTy;

  StringRef Name;
  ContainerTy Data;
  ContainerTy EntryData;

  /// Total execution count for the function.
  int64_t ExecutionCount{0};

  /// Indicate if the data was used.
  bool Used{false};

  FuncBranchData() {}

  FuncBranchData(StringRef Name, ContainerTy Data)
      : Name(Name), Data(std::move(Data)) {}

  FuncBranchData(StringRef Name, ContainerTy Data, ContainerTy EntryData)
      : Name(Name), Data(std::move(Data)), EntryData(std::move(EntryData)) {}

  ErrorOr<const BranchInfo &> getBranch(uint64_t From, uint64_t To) const;

  /// Returns the branch info object associated with a direct call originating
  /// from the given offset. If no branch info object is found, an error is
  /// returned. If the offset corresponds to an indirect call the behavior is
  /// undefined.
  ErrorOr<const BranchInfo &> getDirectCallBranch(uint64_t From) const;

  /// Find all the branches originating at From.
  iterator_range<ContainerTy::const_iterator> getBranchRange(
    uint64_t From) const;

  /// Append the branch data of another function located \p Offset bytes away
  /// from the entry of this function.
  void appendFrom(const FuncBranchData &FBD, uint64_t Offset);

  /// Aggregation helpers
  DenseMap<uint64_t, DenseMap<uint64_t, size_t>> IntraIndex;
  DenseMap<uint64_t, DenseMap<Location, size_t>> InterIndex;
  DenseMap<uint64_t, DenseMap<Location, size_t>> EntryIndex;

  void bumpBranchCount(uint64_t OffsetFrom, uint64_t OffsetTo, bool Mispred);
  void bumpCallCount(uint64_t OffsetFrom, const Location &To, bool Mispred);
  void bumpEntryCount(const Location &From, uint64_t OffsetTo, bool Mispred);
};

/// Similar to BranchInfo, but instead of recording from-to address (an edge),
/// it records the address of a perf event and the number of times samples hit
/// this address.
struct SampleInfo {
  Location Address; // FIXME: Change this name to Loc
  int64_t Occurrences; // FIXME: Variable name is horrible

  SampleInfo(Location Address, int64_t Occurrences)
      : Address(std::move(Address)), Occurrences(Occurrences) {}

  bool operator==(const SampleInfo &RHS) const {
    return Address == RHS.Address;
  }

  bool operator<(const SampleInfo &RHS) const {
    if (Address < RHS.Address)
      return true;

    return false;
  }

  void print(raw_ostream &OS) const;

  void mergeWith(const SampleInfo &SI);
};

/// Helper class to store samples recorded in the address space of a given
/// function, analogous to FuncBranchData but for samples instead of branches.
struct FuncSampleData {
  typedef std::vector<SampleInfo> ContainerTy;

  StringRef Name;
  ContainerTy Data;

  FuncSampleData(StringRef Name, ContainerTy Data)
      : Name(Name), Data(std::move(Data)) {}

  /// Get the number of samples recorded in [Start, End)
  uint64_t getSamples(uint64_t Start, uint64_t End) const;
};

//===----------------------------------------------------------------------===//
//
/// DataReader Class
///
class DataReader {
public:
  explicit DataReader(raw_ostream &Diag) : Diag(Diag) {}

  DataReader(std::unique_ptr<MemoryBuffer> MemBuf, raw_ostream &Diag)
      : FileBuf(std::move(MemBuf)), Diag(Diag),
        ParsingBuf(FileBuf->getBuffer()), Line(0), Col(0) {}

  static ErrorOr<std::unique_ptr<DataReader>> readPerfData(StringRef Path,
                                                           raw_ostream &Diag);

  /// Parses the input bolt data file into internal data structures. We expect
  /// the file format to follow the syntax below.
  ///
  /// <is symbol?> <closest elf symbol or DSO name> <relative FROM address>
  /// <is symbol?> <closest elf symbol or DSO name> <relative TO address>
  /// <number of mispredictions> <number of branches> [<number of histories>
  /// <history entry>
  /// <history entry>
  /// ...]
  ///
  /// Each history entry follows the syntax below.
  ///
  /// <number of mispredictions> <number of branches> <history length>
  /// <is symbol?> <closest elf symbol or DSO name> <relative FROM address>
  /// <is symbol?> <closest elf symbol or DSO name> <relative TO address>
  /// ...
  ///
  /// In <is symbol?> field we record 0 if our closest address is a DSO load
  /// address or 1 if our closest address is an ELF symbol.
  ///
  /// Examples:
  ///
  ///  1 main 3fb 0 /lib/ld-2.21.so 12 4 221
  ///
  /// The example records branches from symbol main, offset 3fb, to DSO ld-2.21,
  /// offset 12, with 4 mispredictions and 221 branches. No history is provided.
  ///
  ///  2 t2.c/func 11 1 globalfunc 1d 0 1775 2
  ///  0 1002 2
  ///  2 t2.c/func 31 2 t2.c/func d
  ///  2 t2.c/func 18 2 t2.c/func 20
  ///  0 773 2
  ///  2 t2.c/func 71 2 t2.c/func d
  ///  2 t2.c/func 18 2 t2.c/func 60
  ///
  /// The examples records branches from local symbol func (from t2.c), offset
  /// 11, to global symbol globalfunc, offset 1d, with 1775 branches, no
  /// mispreds. Of these branches, 1002 were preceeded by a sequence of
  /// branches from func, offset 18 to offset 20 and then from offset 31 to
  /// offset d. The rest 773 branches were preceeded by a different sequence
  /// of branches, from func, offset 18 to offset 60 and then from offset 71 to
  /// offset d.
  std::error_code parse();

  /// When no_lbr is the first line of the file, activate No LBR mode. In this
  /// mode we read the addresses where samples were recorded directly instead of
  /// LBR entries. The line format is almost the same, except for a missing <to>
  /// triple and a missing mispredictions field:
  ///
  /// no_lbr
  /// <is symbol?> <closest elf symbol or DSO name> <relative address> <count>
  /// ...
  ///
  /// Example:
  ///
  /// no_lbr                           # First line of fdata file
  ///  1 BZ2_compressBlock 466c 3
  ///  1 BZ2_hbMakeCodeLengths 29c 1
  ///
  std::error_code parseInNoLBRMode();

  /// Return branch data matching one of the names in \p FuncNames.
  FuncBranchData *
  getFuncBranchData(const std::vector<std::string> &FuncNames);

  FuncSampleData *
  getFuncSampleData(const std::vector<std::string> &FuncNames);

  /// Return a vector of all FuncBranchData matching the list of names.
  /// Internally use fuzzy matching to match special names like LTO-generated
  /// function names.
  std::vector<FuncBranchData *>
  getFuncBranchDataRegex(const std::vector<std::string> &FuncNames);

  using FuncsToBranchesMapTy = StringMap<FuncBranchData>;
  using FuncsToSamplesMapTy = StringMap<FuncSampleData>;

  FuncsToBranchesMapTy &getAllFuncsBranchData() { return FuncsToBranches; }
  FuncsToSamplesMapTy &getAllFuncsSampleData() { return FuncsToSamples; }

  const FuncsToBranchesMapTy &getAllFuncsData() const {
    return FuncsToBranches;
  }

  /// Return true if profile contains an entry for a local function
  /// that has a non-empty associated file name.
  bool hasLocalsWithFileName() const;

  /// Dumps the entire data structures parsed. Used for debugging.
  void dump() const;

  /// Return false only if we are running with profiling data that lacks LBR.
  bool hasLBR() const { return !NoLBRMode; }

  /// Return true if event named \p Name was used to collect this profile data.
  bool usesEvent(StringRef Name) const {
    for (auto I = EventNames.begin(), E = EventNames.end(); I != E; ++I) {
      StringRef Event = I->getKey();
      if (Event.find(Name) != StringRef::npos)
        return true;
    }
    return false;
  }

  /// Return all event names used to collect this profile
  const StringSet<> &getEventNames() const {
    return EventNames;
  }

protected:

  void reportError(StringRef ErrorMsg);
  bool expectAndConsumeFS();
  bool checkAndConsumeNewLine();
  ErrorOr<StringRef> parseString(char EndChar, bool EndNl=false);
  ErrorOr<int64_t> parseNumberField(char EndChar, bool EndNl=false);
  ErrorOr<Location> parseLocation(char EndChar, bool EndNl=false);
  ErrorOr<BranchHistory> parseBranchHistory();
  ErrorOr<BranchInfo> parseBranchInfo();
  ErrorOr<SampleInfo> parseSampleInfo();
  ErrorOr<bool> maybeParseNoLBRFlag();
  bool hasData();

  /// Build suffix map once the profile data is parsed.
  void buildLTONameMap();

  /// An in-memory copy of the input data file - owns strings used in reader.
  std::unique_ptr<MemoryBuffer> FileBuf;
  raw_ostream &Diag;
  StringRef ParsingBuf;
  unsigned Line;
  unsigned Col;
  FuncsToBranchesMapTy FuncsToBranches;
  FuncsToSamplesMapTy FuncsToSamples;
  bool NoLBRMode{false};
  StringSet<> EventNames;
  static const char FieldSeparator = ' ';

  /// Map of common LTO names to possible matching profiles.
  StringMap<std::vector<FuncBranchData *>> LTOCommonNameMap;
};

}

/// DenseMapInfo allows us to use the DenseMap LLVM data structure to store
/// Locations
template<> struct DenseMapInfo<bolt::Location> {
  static inline bolt::Location getEmptyKey() {
    return bolt::Location(true, StringRef(), static_cast<uint64_t>(-1LL));
  }
  static inline bolt::Location getTombstoneKey() {
    return bolt::Location(true, StringRef(), static_cast<uint64_t>(-2LL));;
  }
  static unsigned getHashValue(const bolt::Location &L) {
    return (unsigned(DenseMapInfo<StringRef>::getHashValue(L.Name)) >> 4) ^
           (unsigned(L.Offset));
  }
  static bool isEqual(const bolt::Location &LHS,
                      const bolt::Location &RHS) {
    return LHS.IsSymbol == RHS.IsSymbol && LHS.Name == RHS.Name &&
           LHS.Offset == RHS.Offset;
  }
};


}

#endif
