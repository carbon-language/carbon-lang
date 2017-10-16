//===-- DataReader.cpp - Perf data reader -----------------------*- C++ -*-===//
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


#include "DataReader.h"
#include "llvm/Support/Debug.h"
#include <map>

namespace llvm {
namespace bolt {

Optional<StringRef> getLTOCommonName(const StringRef Name) {
  auto LTOSuffixPos = Name.find(".lto_priv.");
  if (LTOSuffixPos != StringRef::npos) {
    return Name.substr(0, LTOSuffixPos + 10);
  } else if ((LTOSuffixPos = Name.find(".constprop.")) != StringRef::npos) {
    return Name.substr(0, LTOSuffixPos + 11);
  } else {
    return NoneType();
  }
}

namespace {

/// Return standard name of the function possibly renamed by BOLT.
StringRef normalizeName(StringRef Name) {
  // Strip "PG." prefix used for globalized locals.
  return Name.startswith("PG.") ? Name.substr(2) : Name;
}

} // anonymous namespace

raw_ostream &operator<<(raw_ostream &OS, const Location &Loc) {
  if (Loc.IsSymbol) {
    OS << Loc.Name;
    if (Loc.Offset)
      OS << "+" << Twine::utohexstr(Loc.Offset);
  } else {
    OS << Twine::utohexstr(Loc.Offset);
  }
  return OS;
}

iterator_range<FuncBranchData::ContainerTy::const_iterator>
FuncBranchData::getBranchRange(uint64_t From) const {
  assert(std::is_sorted(Data.begin(), Data.end()));
  struct Compare {
    bool operator()(const BranchInfo &BI, const uint64_t Val) const {
      return BI.From.Offset < Val;
    }
    bool operator()(const uint64_t Val, const BranchInfo &BI) const {
      return Val < BI.From.Offset;
    }
  };
  auto Range = std::equal_range(Data.begin(), Data.end(), From, Compare());
  return iterator_range<ContainerTy::const_iterator>(Range.first, Range.second);
}

void FuncBranchData::appendFrom(const FuncBranchData &FBD, uint64_t Offset) {
  Data.insert(Data.end(), FBD.Data.begin(), FBD.Data.end());
  for (auto I = Data.begin(), E = Data.end(); I != E; ++I) {
    if (I->From.Name == FBD.Name) {
      I->From.Name = this->Name;
      I->From.Offset += Offset;
    }
    if (I->To.Name == FBD.Name) {
      I->To.Name = this->Name;
      I->To.Offset += Offset;
    }
  }
  std::stable_sort(Data.begin(), Data.end());
  ExecutionCount += FBD.ExecutionCount;
  for (auto I = FBD.EntryData.begin(), E = FBD.EntryData.end(); I != E; ++I) {
    assert(I->To.Name == FBD.Name);
    auto NewElmt = EntryData.insert(EntryData.end(), *I);
    NewElmt->To.Name = this->Name;
    NewElmt->To.Offset += Offset;
  }
}

void SampleInfo::mergeWith(const SampleInfo &SI) {
  Occurrences += SI.Occurrences;
}

void SampleInfo::print(raw_ostream &OS) const {
  OS << Address.IsSymbol << " " << Address.Name << " "
     << Twine::utohexstr(Address.Offset) << " "
     << Occurrences << "\n";
}

uint64_t
FuncSampleData::getSamples(uint64_t Start, uint64_t End) const {
  assert(std::is_sorted(Data.begin(), Data.end()));
  struct Compare {
    bool operator()(const SampleInfo &SI, const uint64_t Val) const {
      return SI.Address.Offset < Val;
    }
    bool operator()(const uint64_t Val, const SampleInfo &SI) const {
      return Val < SI.Address.Offset;
    }
  };
  uint64_t Result{0};
  for (auto I = std::lower_bound(Data.begin(), Data.end(), Start, Compare()),
            E = std::lower_bound(Data.begin(), Data.end(), End, Compare());
       I != E; ++I) {
    Result += I->Occurrences;
  }
  return Result;
}

void FuncBranchData::bumpBranchCount(uint64_t OffsetFrom, uint64_t OffsetTo,
                                     bool Mispred) {
  auto Iter = IntraIndex[OffsetFrom].find(OffsetTo);
  if (Iter == IntraIndex[OffsetFrom].end()) {
    Data.emplace_back(Location(true, Name, OffsetFrom),
                      Location(true, Name, OffsetTo), Mispred, 1,
                      BranchHistories());
    IntraIndex[OffsetFrom][OffsetTo] = Data.size() - 1;
    return;
  }
  auto &BI = Data[Iter->second];
  ++BI.Branches;
  if (Mispred)
    ++BI.Mispreds;
}

void FuncBranchData::bumpCallCount(uint64_t OffsetFrom, const Location &To,
                                   bool Mispred) {
  auto Iter = InterIndex[OffsetFrom].find(To);
  if (Iter == InterIndex[OffsetFrom].end()) {
    Data.emplace_back(Location(true, Name, OffsetFrom), To, Mispred, 1,
                      BranchHistories());
    InterIndex[OffsetFrom][To] = Data.size() - 1;
    return;
  }
  auto &BI = Data[Iter->second];
  ++BI.Branches;
  if (Mispred)
    ++BI.Mispreds;
}

void FuncBranchData::bumpEntryCount(const Location &From, uint64_t OffsetTo,
                                    bool Mispred) {
  auto Iter = EntryIndex[OffsetTo].find(From);
  if (Iter == EntryIndex[OffsetTo].end()) {
    EntryData.emplace_back(From, Location(true, Name, OffsetTo), Mispred, 1,
                           BranchHistories());
    EntryIndex[OffsetTo][From] = EntryData.size() - 1;
    return;
  }
  auto &BI = EntryData[Iter->second];
  ++BI.Branches;
  if (Mispred)
    ++BI.Mispreds;
}

void BranchInfo::mergeWith(const BranchInfo &BI) {

  // Merge branch and misprediction counts.
  Branches += BI.Branches;
  Mispreds += BI.Mispreds;

  // Trivial cases
  if (BI.Histories.size() == 0)
    return;

  if (Histories.size() == 0) {
    Histories = BI.Histories;
    return;
  }

  // map BranchContext -> (mispreds, count), used to merge histories
  std::map<BranchContext, std::pair<uint64_t, uint64_t>> HistMap;

  // Add histories of this BranchInfo into HistMap.
  for (const auto &H : Histories) {
    BranchContext C;
    for (const auto &LocPair : H.Context) {
      C.emplace_back(LocPair);
      const auto I = HistMap.find(C);
      if (I == HistMap.end()) {
        HistMap.insert(
            std::make_pair(C, std::make_pair(H.Mispreds, H.Branches)));
      }
      else {
        I->second.first += H.Mispreds;
        I->second.second += H.Branches;
      }
    }
  }

  // Add histories of BI into HistMap.
  for (const auto &H : BI.Histories) {
    BranchContext C;
    for (const auto &LocPair : H.Context) {
      C.emplace_back(LocPair);
      const auto I = HistMap.find(C);
      if (I == HistMap.end()) {
        HistMap.insert(
            std::make_pair(C, std::make_pair(H.Mispreds, H.Branches)));
      }
      else {
        I->second.first += H.Mispreds;
        I->second.second += H.Branches;
      }
    }
  }

  // Helper function that checks whether context A is a prefix of context B.
  auto isPrefix = [] (const BranchContext &A, const BranchContext &B) -> bool {
    for (unsigned i = 0; i < A.size(); ++i) {
      if (i >= B.size() || A[i] != B[i])
        return false;
    }
    return true;
  };

  // Extract merged histories from HistMap. Keep only the longest history
  // between histories that share a common prefix.
  Histories.clear();
  auto I = HistMap.begin(), E = HistMap.end();
  auto NextI = I;
  ++NextI;
  for ( ; I != E; ++I, ++NextI) {
    if (NextI != E && isPrefix(I->first, NextI->first))
      continue;

    Histories.emplace_back(BranchHistory(I->second.first,
                                         I->second.second,
                                         I->first));
  }
}

void BranchInfo::print(raw_ostream &OS) const {
  OS << From.IsSymbol << " " << From.Name << " "
     << Twine::utohexstr(From.Offset) << " "
     << To.IsSymbol << " " << To.Name << " "
     << Twine::utohexstr(To.Offset) << " "
     << Mispreds << " " << Branches;

  if (Histories.size() == 0) {
    OS << "\n";
    return;
  }

  OS << " " << Histories.size() << "\n";
  for (const auto &H : Histories) {
    OS << H.Mispreds << " " << H.Branches << " " << H.Context.size() << "\n";
    for (const auto &C : H.Context) {
      OS << C.first.IsSymbol << " " << C.first.Name << " "
         << Twine::utohexstr(C.first.Offset) << " "
         << C.second.IsSymbol << " " << C.second.Name << " "
         << Twine::utohexstr(C.second.Offset) << "\n";
    }
  }
}

ErrorOr<const BranchInfo &> FuncBranchData::getBranch(uint64_t From,
                                                      uint64_t To) const {
  for (const auto &I : Data) {
    if (I.From.Offset == From && I.To.Offset == To &&
        I.From.Name == I.To.Name)
      return I;
  }
  return make_error_code(llvm::errc::invalid_argument);
}

ErrorOr<const BranchInfo &>
FuncBranchData::getDirectCallBranch(uint64_t From) const {
  // Commented out because it can be expensive.
  // assert(std::is_sorted(Data.begin(), Data.end()));
  struct Compare {
    bool operator()(const BranchInfo &BI, const uint64_t Val) const {
      return BI.From.Offset < Val;
    }
    bool operator()(const uint64_t Val, const BranchInfo &BI) const {
      return Val < BI.From.Offset;
    }
  };
  auto Range = std::equal_range(Data.begin(), Data.end(), From, Compare());
  for (auto I = Range.first; I != Range.second; ++I) {
    if (I->From.Name != I->To.Name)
      return *I;
  }
  return make_error_code(llvm::errc::invalid_argument);
}

void MemInfo::print(raw_ostream &OS) const {
  OS << (Offset.IsSymbol + 3) << " " << Offset.Name << " "
     << Twine::utohexstr(Offset.Offset) << " "
     << (Addr.IsSymbol + 3) << " " << Addr.Name << " "
     << Twine::utohexstr(Addr.Offset) << " "
     << Count << "\n";
}

iterator_range<FuncMemData::ContainerTy::const_iterator>
FuncMemData::getMemInfoRange(uint64_t Offset) const {
  assert(std::is_sorted(Data.begin(), Data.end()));
  struct Compare {
    bool operator()(const MemInfo &MI, const uint64_t Val) const {
      return MI.Offset.Offset < Val;
    }
    bool operator()(const uint64_t Val, const MemInfo &MI) const {
      return Val < MI.Offset.Offset;
    }
  };
  auto Range = std::equal_range(Data.begin(), Data.end(), Offset, Compare());
  return iterator_range<ContainerTy::const_iterator>(Range.first, Range.second);
}

void FuncMemData::update(const Location &Offset, const Location &Addr) {
  auto Iter = EventIndex[Offset.Offset].find(Addr);
  if (Iter == EventIndex[Offset.Offset].end()) {
    Data.emplace_back(MemInfo(Offset, Addr, 1));
    EventIndex[Offset.Offset][Addr] = Data.size() - 1;
    return;
  }
  ++Data[Iter->second].Count;
}

ErrorOr<std::unique_ptr<DataReader>>
DataReader::readPerfData(StringRef Path, raw_ostream &Diag) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> MB =
      MemoryBuffer::getFileOrSTDIN(Path);
  if (std::error_code EC = MB.getError()) {
    Diag << "Cannot open " << Path << ": " << EC.message() << "\n";
    return EC;
  }
  auto DR = make_unique<DataReader>(std::move(MB.get()), Diag);
  DR->parse();
  DR->buildLTONameMaps();
  return std::move(DR);
}

void DataReader::reportError(StringRef ErrorMsg) {
  Diag << "Error reading bolt data input file: line " << Line << ", column "
       << Col << ": " << ErrorMsg << '\n';
}

bool DataReader::expectAndConsumeFS() {
  if (ParsingBuf[0] != FieldSeparator) {
    reportError("expected field separator");
    return false;
  }
  ParsingBuf = ParsingBuf.drop_front(1);
  Col += 1;
  return true;
}

bool DataReader::checkAndConsumeNewLine() {
  if (ParsingBuf[0] != '\n')
    return false;

  ParsingBuf = ParsingBuf.drop_front(1);
  Col = 0;
  Line += 1;
  return true;
}

ErrorOr<StringRef> DataReader::parseString(char EndChar, bool EndNl) {
  std::string EndChars(1, EndChar);
  if (EndNl)
    EndChars.push_back('\n');
  auto StringEnd = ParsingBuf.find_first_of(EndChars);
  if (StringEnd == StringRef::npos || StringEnd == 0) {
    reportError("malformed field");
    return make_error_code(llvm::errc::io_error);
  }

  StringRef Str = ParsingBuf.substr(0, StringEnd);

  // If EndNl was set and nl was found instead of EndChar, do not consume the
  // new line.
  bool EndNlInstreadOfEndChar =
    ParsingBuf[StringEnd] == '\n' && EndChar != '\n';
  unsigned End = EndNlInstreadOfEndChar ? StringEnd : StringEnd + 1;

  ParsingBuf = ParsingBuf.drop_front(End);
  if (EndChar == '\n') {
    Col = 0;
    Line += 1;
  } else {
    Col += End;
  }
  return Str;
}

ErrorOr<int64_t> DataReader::parseNumberField(char EndChar, bool EndNl) {
  auto NumStrRes = parseString(EndChar, EndNl);
  if (std::error_code EC = NumStrRes.getError())
    return EC;
  StringRef NumStr = NumStrRes.get();
  int64_t Num;
  if (NumStr.getAsInteger(10, Num)) {
    reportError("expected decimal number");
    Diag << "Found: " << NumStr << "\n";
    return make_error_code(llvm::errc::io_error);
  }
  return Num;
}

ErrorOr<uint64_t> DataReader::parseHexField(char EndChar, bool EndNl) {
  auto NumStrRes = parseString(EndChar, EndNl);
  if (std::error_code EC = NumStrRes.getError())
    return EC;
  StringRef NumStr = NumStrRes.get();
  uint64_t Num;
  if (NumStr.getAsInteger(16, Num)) {
    reportError("expected hexidecimal number");
    Diag << "Found: " << NumStr << "\n";
    return make_error_code(llvm::errc::io_error);
  }
  return Num;
}

ErrorOr<Location> DataReader::parseLocation(char EndChar,
                                            bool EndNl,
                                            bool ExpectMemLoc) {
  // Read whether the location of the branch should be DSO or a symbol
  // 0 means it is a DSO. 1 means it is a global symbol. 2 means it is a local
  // symbol.
  // The symbol flag is also used to tag memory load events by adding 3 to the
  // base values, i.e. 3 not a symbol, 4 global symbol and 5 local symbol.
  if (!ExpectMemLoc &&
      ParsingBuf[0] != '0' && ParsingBuf[0] != '1' && ParsingBuf[0] != '2') {
    reportError("expected 0, 1 or 2");
    return make_error_code(llvm::errc::io_error);
  }

  if (ExpectMemLoc &&
      ParsingBuf[0] != '3' && ParsingBuf[0] != '4' && ParsingBuf[0] != '5') {
    reportError("expected 3, 4 or 5");
    return make_error_code(llvm::errc::io_error);
  }

  bool IsSymbol =
    (!ExpectMemLoc && (ParsingBuf[0] == '1' || ParsingBuf[0] == '2')) ||
    (ExpectMemLoc && (ParsingBuf[0] == '4' || ParsingBuf[0] == '5'));
  ParsingBuf = ParsingBuf.drop_front(1);
  Col += 1;

  if (!expectAndConsumeFS())
    return make_error_code(llvm::errc::io_error);

  // Read the string containing the symbol or the DSO name
  auto NameRes = parseString(FieldSeparator);
  if (std::error_code EC = NameRes.getError())
    return EC;
  StringRef Name = NameRes.get();

  // Read the offset
  auto Offset = parseHexField(EndChar, EndNl);
  if (std::error_code EC = Offset.getError())
    return EC;

  return Location(IsSymbol, Name, Offset.get());
}

ErrorOr<BranchHistory> DataReader::parseBranchHistory() {
  auto MRes = parseNumberField(FieldSeparator);
  if (std::error_code EC = MRes.getError())
    return EC;
  int64_t NumMispreds = MRes.get();

  auto BRes = parseNumberField(FieldSeparator);
  if (std::error_code EC = BRes.getError())
    return EC;
  int64_t NumBranches = BRes.get();

  auto LRes = parseNumberField('\n');
  if (std::error_code EC = LRes.getError())
    return EC;
  int64_t ContextLength = LRes.get();
  assert(ContextLength > 0 && "found branch context with length 0");

  BranchContext Context;
  for (unsigned i = 0; i < ContextLength; ++i) {
    auto Res = parseLocation(FieldSeparator);
    if (std::error_code EC = Res.getError())
      return EC;
    Location CtxFrom = Res.get();

    Res = parseLocation('\n');
    if (std::error_code EC = Res.getError())
      return EC;
    Location CtxTo = Res.get();

    Context.emplace_back(std::make_pair(std::move(CtxFrom),
                                        std::move(CtxTo)));
  }

  return BranchHistory(NumMispreds, NumBranches, std::move(Context));
}

ErrorOr<BranchInfo> DataReader::parseBranchInfo() {
  auto Res = parseLocation(FieldSeparator);
  if (std::error_code EC = Res.getError())
    return EC;
  Location From = Res.get();

  Res = parseLocation(FieldSeparator);
  if (std::error_code EC = Res.getError())
    return EC;
  Location To = Res.get();

  auto MRes = parseNumberField(FieldSeparator);
  if (std::error_code EC = MRes.getError())
    return EC;
  int64_t NumMispreds = MRes.get();

  auto BRes = parseNumberField(FieldSeparator, /* EndNl = */ true);
  if (std::error_code EC = BRes.getError())
    return EC;
  int64_t NumBranches = BRes.get();

  BranchHistories Histories;

  if (!checkAndConsumeNewLine()) {
    auto HRes = parseNumberField('\n');
    if (std::error_code EC = HRes.getError())
      return EC;
    int64_t NumHistories = HRes.get();
    assert(NumHistories > 0 && "found branch history list with length 0");

    for (unsigned i = 0; i < NumHistories; ++i) {
      auto Res = parseBranchHistory();
      if (std::error_code EC = Res.getError())
        return EC;
      BranchHistory Hist = Res.get();

      Histories.emplace_back(std::move(Hist));
    }
  }

  return BranchInfo(std::move(From), std::move(To), NumMispreds, NumBranches,
                    std::move(Histories));
}

ErrorOr<MemInfo> DataReader::parseMemInfo() {
  auto Res = parseMemLocation(FieldSeparator);
  if (std::error_code EC = Res.getError())
    return EC;
  Location Offset = Res.get();

  Res = parseMemLocation(FieldSeparator);
  if (std::error_code EC = Res.getError())
    return EC;
  Location Addr = Res.get();

  auto CountRes = parseNumberField(FieldSeparator, true);
  if (std::error_code EC = CountRes.getError())
    return EC;

  checkAndConsumeNewLine();

  return MemInfo(Offset, Addr, CountRes.get());
}
  
ErrorOr<SampleInfo> DataReader::parseSampleInfo() {
  auto Res = parseLocation(FieldSeparator);
  if (std::error_code EC = Res.getError())
    return EC;
  Location Address = Res.get();

  auto BRes = parseNumberField(FieldSeparator, /* EndNl = */ true);
  if (std::error_code EC = BRes.getError())
    return EC;
  int64_t Occurrences = BRes.get();

  if (!checkAndConsumeNewLine()) {
    reportError("expected end of line");
    return make_error_code(llvm::errc::io_error);
  }

  return SampleInfo(std::move(Address), Occurrences);
}

ErrorOr<bool> DataReader::maybeParseNoLBRFlag() {
  if (ParsingBuf.size() < 6 || ParsingBuf.substr(0, 6) != "no_lbr")
    return false;
  ParsingBuf = ParsingBuf.drop_front(6);
  Col += 6;

  if (ParsingBuf.size() > 0 && ParsingBuf[0] == ' ')
    ParsingBuf = ParsingBuf.drop_front(1);

  while (ParsingBuf.size() > 0 && ParsingBuf[0] != '\n') {
    auto EventName = parseString(' ', true);
    if (!EventName)
      return make_error_code(llvm::errc::io_error);
    EventNames.insert(EventName.get());
  }

  if (!checkAndConsumeNewLine()) {
    reportError("malformed no_lbr line");
    return make_error_code(llvm::errc::io_error);
  }
  return true;
}

bool DataReader::hasBranchData() {
  if (ParsingBuf.size() == 0)
    return false;

  if (ParsingBuf[0] == '0' || ParsingBuf[0] == '1' || ParsingBuf[0] == '2')
    return true;
  return false;
}

bool DataReader::hasMemData() {
  if (ParsingBuf.size() == 0)
    return false;

  if (ParsingBuf[0] == '3' || ParsingBuf[0] == '4' || ParsingBuf[0] == '5')
    return true;
  return false;
}

std::error_code DataReader::parseInNoLBRMode() {
  auto GetOrCreateFuncEntry = [&](StringRef Name) {
    auto I = FuncsToSamples.find(Name);
    if (I == FuncsToSamples.end()) {
      bool success;
      std::tie(I, success) = FuncsToSamples.insert(std::make_pair(
          Name, FuncSampleData(Name, FuncSampleData::ContainerTy())));

      assert(success && "unexpected result of insert");
    }
    return I;
  };

  auto GetOrCreateFuncMemEntry = [&](StringRef Name) {
    auto I = FuncsToMemEvents.find(Name);
    if (I == FuncsToMemEvents.end()) {
      bool success;
      std::tie(I, success) = FuncsToMemEvents.insert(
          std::make_pair(Name, FuncMemData(Name, FuncMemData::ContainerTy())));
      assert(success && "unexpected result of insert");
    }
    return I;
  };

  while (hasBranchData()) {
    auto Res = parseSampleInfo();
    if (std::error_code EC = Res.getError())
      return EC;

    SampleInfo SI = Res.get();

    // Ignore samples not involving known locations
    if (!SI.Address.IsSymbol)
      continue;

    auto I = GetOrCreateFuncEntry(SI.Address.Name);
    I->getValue().Data.emplace_back(std::move(SI));
  }

  while (hasMemData()) {
    auto Res = parseMemInfo();
    if (std::error_code EC = Res.getError())
      return EC;

    MemInfo MI = Res.get();

    // Ignore memory events not involving known pc.
    if (!MI.Offset.IsSymbol)
      continue;

    auto I = GetOrCreateFuncMemEntry(MI.Offset.Name);
    I->getValue().Data.emplace_back(std::move(MI));
  }

  for (auto &FuncSamples : FuncsToSamples) {
    std::stable_sort(FuncSamples.second.Data.begin(),
                     FuncSamples.second.Data.end());
  }

  for (auto &MemEvents : FuncsToMemEvents) {
    std::stable_sort(MemEvents.second.Data.begin(),
                     MemEvents.second.Data.end());
  }

  return std::error_code();
}

std::error_code DataReader::parse() {
  auto GetOrCreateFuncEntry = [&](StringRef Name) {
    auto I = FuncsToBranches.find(Name);
    if (I == FuncsToBranches.end()) {
      bool success;
      std::tie(I, success) = FuncsToBranches.insert(
          std::make_pair(Name, FuncBranchData(Name,
                                              FuncBranchData::ContainerTy(),
                                              FuncBranchData::ContainerTy())));
      assert(success && "unexpected result of insert");
    }
    return I;
  };

  auto GetOrCreateFuncMemEntry = [&](StringRef Name) {
    auto I = FuncsToMemEvents.find(Name);
    if (I == FuncsToMemEvents.end()) {
      bool success;
      std::tie(I, success) = FuncsToMemEvents.insert(
          std::make_pair(Name, FuncMemData(Name, FuncMemData::ContainerTy())));
      assert(success && "unexpected result of insert");
    }
    return I;
  };

  Col = 0;
  Line = 1;
  auto FlagOrErr = maybeParseNoLBRFlag();
  if (!FlagOrErr)
    return FlagOrErr.getError();
  NoLBRMode = *FlagOrErr;
  if (NoLBRMode)
    return parseInNoLBRMode();

  while (hasBranchData()) {
    auto Res = parseBranchInfo();
    if (std::error_code EC = Res.getError())
      return EC;

    BranchInfo BI = Res.get();

    // Ignore branches not involving known location.
    if (!BI.From.IsSymbol && !BI.To.IsSymbol)
      continue;

    auto I = GetOrCreateFuncEntry(BI.From.Name);
    I->getValue().Data.emplace_back(std::move(BI));

    // Add entry data for branches to another function or branches
    // to entry points (including recursive calls)
    if (BI.To.IsSymbol &&
        (!BI.From.Name.equals(BI.To.Name) || BI.To.Offset == 0)) {
      I = GetOrCreateFuncEntry(BI.To.Name);
      I->getValue().EntryData.emplace_back(std::move(BI));
    }

    // If destination is the function start - update execution count.
    // NB: the data is skewed since we cannot tell tail recursion from
    //     branches to the function start.
    if (BI.To.IsSymbol && BI.To.Offset == 0) {
      I = GetOrCreateFuncEntry(BI.To.Name);
      I->getValue().ExecutionCount += BI.Branches;
    }
  }

  while (hasMemData()) {
    auto Res = parseMemInfo();
    if (std::error_code EC = Res.getError())
      return EC;

    MemInfo MI = Res.get();

    // Ignore memory events not involving known pc.
    if (!MI.Offset.IsSymbol)
      continue;

    auto I = GetOrCreateFuncMemEntry(MI.Offset.Name);
    I->getValue().Data.emplace_back(std::move(MI));
  }

  for (auto &FuncBranches : FuncsToBranches) {
    std::stable_sort(FuncBranches.second.Data.begin(),
                     FuncBranches.second.Data.end());
  }

  for (auto &MemEvents : FuncsToMemEvents) {
    std::stable_sort(MemEvents.second.Data.begin(),
                     MemEvents.second.Data.end());
  }

  return std::error_code();
}

void DataReader::buildLTONameMaps() {
  for (auto &FuncData : FuncsToBranches) {
    const auto FuncName = FuncData.getKey();
    const auto CommonName = getLTOCommonName(FuncName);
    if (CommonName)
      LTOCommonNameMap[*CommonName].push_back(&FuncData.getValue());
  }

  for (auto &FuncData : FuncsToMemEvents) {
    const auto FuncName = FuncData.getKey();
    const auto CommonName = getLTOCommonName(FuncName);
    if (CommonName)
      LTOCommonNameMemMap[*CommonName].push_back(&FuncData.getValue());
  }
}

namespace {
template <typename MapTy>
decltype(MapTy::MapEntryTy::second) *
fetchMapEntry(MapTy &Map, const std::vector<std::string> &FuncNames) {
  // Do a reverse order iteration since the name in profile has a higher chance
  // of matching a name at the end of the list.
  for (auto FI = FuncNames.rbegin(), FE = FuncNames.rend(); FI != FE; ++FI) {
    auto I = Map.find(normalizeName(*FI));
    if (I != Map.end())
      return &I->getValue();
  }
  return nullptr;
}

template <typename MapTy>
std::vector<decltype(MapTy::MapEntryTy::second) *>
fetchMapEntriesRegex(
  MapTy &Map,
  const StringMap<std::vector<decltype(MapTy::MapEntryTy::second) *>> &LTOCommonNameMap,
  const std::vector<std::string> &FuncNames) {
  std::vector<decltype(MapTy::MapEntryTy::second) *> AllData;
  // Do a reverse order iteration since the name in profile has a higher chance
  // of matching a name at the end of the list.
  for (auto FI = FuncNames.rbegin(), FE = FuncNames.rend(); FI != FE; ++FI) {
    StringRef Name = *FI;
    Name = normalizeName(Name);
    const auto LTOCommonName = getLTOCommonName(Name);
    if (LTOCommonName) {
      auto I = LTOCommonNameMap.find(*LTOCommonName);
      if (I != LTOCommonNameMap.end()) {
        auto &CommonData = I->getValue();
        AllData.insert(AllData.end(), CommonData.begin(), CommonData.end());
      }
    } else {
      auto I = Map.find(Name);
      if (I != Map.end()) {
        return {&I->getValue()};
      }
    }
  }
  return AllData;
}

}

FuncBranchData *
DataReader::getFuncBranchData(const std::vector<std::string> &FuncNames) {
  return fetchMapEntry<FuncsToBranchesMapTy>(FuncsToBranches, FuncNames);
}

FuncMemData *
DataReader::getFuncMemData(const std::vector<std::string> &FuncNames) {
  return fetchMapEntry<FuncsToMemEventsMapTy>(FuncsToMemEvents, FuncNames);
}

FuncSampleData *
DataReader::getFuncSampleData(const std::vector<std::string> &FuncNames) {
  return fetchMapEntry<FuncsToSamplesMapTy>(FuncsToSamples, FuncNames);
}

std::vector<FuncBranchData *>
DataReader::getFuncBranchDataRegex(const std::vector<std::string> &FuncNames) {
  return fetchMapEntriesRegex(FuncsToBranches, LTOCommonNameMap, FuncNames);
}

std::vector<FuncMemData *>
DataReader::getFuncMemDataRegex(const std::vector<std::string> &FuncNames) {
  return fetchMapEntriesRegex(FuncsToMemEvents, LTOCommonNameMemMap, FuncNames);
}

bool DataReader::hasLocalsWithFileName() const {
  for (const auto &Func : FuncsToBranches) {
    const auto &FuncName = Func.getKey();
    if (FuncName.count('/') == 2 && FuncName[0] != '/')
      return true;
  }
  return false;
}

void DataReader::dump() const {
  for (const auto &Func : FuncsToBranches) {
    Diag << Func.getKey() << " branches:\n";
    for (const auto &BI : Func.getValue().Data) {
      Diag << BI.From.Name << " " << BI.From.Offset << " " << BI.To.Name << " "
           << BI.To.Offset << " " << BI.Mispreds << " " << BI.Branches << "\n";
      for (const auto &HI : BI.Histories) {
        Diag << "\thistory " << HI.Mispreds << " " << HI.Branches << "\n";
        for (const auto &CI : HI.Context) {
          Diag << "\t" <<  CI.first.Name << " " << CI.first.Offset << " "
                       << CI.second.Name << " " << CI.second.Offset << "\n";
        }
      }
    }
    Diag << Func.getKey() << " entry points:\n";
    for (const auto &BI : Func.getValue().EntryData) {
      Diag << BI.From.Name << " " << BI.From.Offset << " " << BI.To.Name << " "
           << BI.To.Offset << " " << BI.Mispreds << " " << BI.Branches << "\n";
      for (const auto &HI : BI.Histories) {
        Diag << "\thistory " << HI.Mispreds << " " << HI.Branches << "\n";
        for (const auto &CI : HI.Context) {
          Diag << "\t" <<  CI.first.Name << " " << CI.first.Offset << " "
                       << CI.second.Name << " " << CI.second.Offset << "\n";
        }
      }
    }
  }

  for (auto I = EventNames.begin(), E = EventNames.end(); I != E; ++I) {
    StringRef Event = I->getKey();
    Diag << "Data was collected with event: " << Event << "\n";
  }
  for (const auto &Func : FuncsToSamples) {
    Diag << Func.getKey() << " samples:\n";
    for (const auto &SI : Func.getValue().Data) {
      Diag << SI.Address.Name << " " << SI.Address.Offset << " "
           << SI.Occurrences << "\n";
    }
  }

  for (const auto &Func : FuncsToMemEvents) {
    Diag << "Memory events for " << Func.getValue().Name;
    Location LastOffset(0);
    for (auto &MI : Func.getValue().Data) {
      if (MI.Offset == LastOffset) {
        Diag << ", " << MI.Addr << "/" << MI.Count;
      } else {
        Diag << "\n" << MI.Offset << ": " << MI.Addr << "/" << MI.Count;
      }
      LastOffset = MI.Offset;
    }
    Diag << "\n";
  }
}

} // namespace bolt
} // namespace llvm
