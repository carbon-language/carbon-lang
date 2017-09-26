//===-- DataAggregator.cpp - Perf data aggregator ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This family of functions reads profile data written by perf record,
// aggregate it and then write it back to an output file.
//
//===----------------------------------------------------------------------===//

#include "BinaryContext.h"
#include "BinaryFunction.h"
#include "DataAggregator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Options.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Timer.h"

#include <unistd.h>

#define DEBUG_TYPE "aggregator"

using namespace llvm;
using namespace bolt;

namespace opts {

extern cl::OptionCategory AggregatorCategory;

static llvm::cl::opt<bool>
TimeAggregator("time-aggr",
  cl::desc("time BOLT aggregator"),
  cl::init(false),
  cl::ZeroOrMore,
  cl::cat(AggregatorCategory));

}

namespace {

const char TimerGroupName[] = "Aggregator";

}

void DataAggregator::findPerfExecutable() {
  auto PerfExecutable = sys::Process::FindInEnvPath("PATH", "perf");
  if (!PerfExecutable) {
    outs() << "PERF2BOLT: No perf executable found!\n";
    exit(1);
  }
  PerfPath = *PerfExecutable;
}

void DataAggregator::start(StringRef PerfDataFilename) {
  Enabled = true;
  outs() << "PERF2BOLT: Starting data aggregation job for " << PerfDataFilename
         << "\n";
  findPerfExecutable();
  launchPerfEventsNoWait(PerfDataFilename);
  launchPerfTasksNoWait(PerfDataFilename);
}

bool DataAggregator::launchPerfEventsNoWait(StringRef PerfDataFilename) {
  SmallVector<const char*, 4> Argv;
  SmallVector<StringRef, 3> Redirects;
  SmallVector<const StringRef*, 3> RedirectPtrs;

  outs() << "PERF2BOLT: Spawning perf-script job to read events\n";
  Argv.push_back(PerfPath.data());
  Argv.push_back("script");
  Argv.push_back("-F");
  Argv.push_back("pid,brstack");
  Argv.push_back("-i");
  Argv.push_back(PerfDataFilename.data());
  Argv.push_back(nullptr);

  if (auto Errc = sys::fs::createTemporaryFile("perf.script", "out",
                                               PerfEventsOutputPath)) {
    outs() << "PERF2BOLT: Failed to create temporary file "
           << PerfEventsOutputPath << " with error " << Errc.message() << "\n";
    exit(1);
  }

  if (auto Errc = sys::fs::createTemporaryFile("perf.script", "err",
                                               PerfEventsErrPath)) {
    outs() << "PERF2BOLT: Failed to create temporary file "
           << PerfEventsErrPath << " with error " << Errc.message() << "\n";
    exit(1);
  }

  Redirects.push_back("");                                     // Stdin
  Redirects.push_back(StringRef(PerfEventsOutputPath.data())); // Stdout
  Redirects.push_back(StringRef(PerfEventsErrPath.data()));    // Stderr
  RedirectPtrs.push_back(&Redirects[0]);
  RedirectPtrs.push_back(&Redirects[1]);
  RedirectPtrs.push_back(&Redirects[2]);

  DEBUG(dbgs() << "Launching perf: " << PerfPath.data() << " 1> "
               << PerfEventsOutputPath.data() << " 2> "
               << PerfEventsErrPath.data() << "\n");

  EventsPI = sys::ExecuteNoWait(PerfPath.data(), Argv.data(),
                                /*envp*/ nullptr, &RedirectPtrs[0]);

  return true;
}

bool DataAggregator::launchPerfTasksNoWait(StringRef PerfDataFilename) {
  SmallVector<const char*, 4> Argv;
  SmallVector<StringRef, 3> Redirects;
  SmallVector<const StringRef*, 3> RedirectPtrs;

  outs() << "PERF2BOLT: Spawning perf-script job to read tasks\n";
  Argv.push_back(PerfPath.data());
  Argv.push_back("script");
  Argv.push_back("--show-task-events");
  Argv.push_back("-i");
  Argv.push_back(PerfDataFilename.data());
  Argv.push_back(nullptr);

  if (auto Errc = sys::fs::createTemporaryFile("perf.script", "out",
                                               PerfTasksOutputPath)) {
    outs() << "PERF2BOLT: Failed to create temporary file "
           << PerfTasksOutputPath << " with error " << Errc.message() << "\n";
    exit(1);
  }

  if (auto Errc = sys::fs::createTemporaryFile("perf.script", "err",
                                               PerfTasksErrPath)) {
    outs() << "PERF2BOLT: Failed to create temporary file "
           << PerfTasksErrPath << " with error " << Errc.message() << "\n";
    exit(1);
  }

  Redirects.push_back("");                                    // Stdin
  Redirects.push_back(StringRef(PerfTasksOutputPath.data())); // Stdout
  Redirects.push_back(StringRef(PerfTasksErrPath.data()));    // Stderr
  RedirectPtrs.push_back(&Redirects[0]);
  RedirectPtrs.push_back(&Redirects[1]);
  RedirectPtrs.push_back(&Redirects[2]);

  DEBUG(dbgs() << "Launching perf: " << PerfPath.data() << " 1> "
               << PerfTasksOutputPath.data() << " 2> "
               << PerfTasksErrPath.data() << "\n");

  TasksPI = sys::ExecuteNoWait(PerfPath.data(), Argv.data(),
                               /*envp*/ nullptr, &RedirectPtrs[0]);

  return true;
}

bool DataAggregator::checkPerfDataMagic(StringRef FileName) {
  int FD;
  if (sys::fs::openFileForRead(FileName, FD)) {
    return false;
  }

  char Buf[7] = {0, 0, 0, 0, 0, 0, 0};

  if (::read(FD, Buf, 7) == -1) {
    ::close(FD);
    return false;
  }
  ::close(FD);

  if (strncmp(Buf, "PERFILE", 7) == 0)
    return true;
  return false;
}

void DataAggregator::deleteTempFiles() {
  if (auto Errc = sys::fs::remove(PerfEventsErrPath.data())) {
    outs() << "PERF2BOLT: Failed to delete temporary file "
           << PerfEventsErrPath << " with error " << Errc.message() << "\n";
  }

  if (auto Errc = sys::fs::remove(PerfEventsOutputPath.data())) {
    outs() << "PERF2BOLT: Failed to delete temporary file "
           << PerfEventsOutputPath << " with error " << Errc.message() << "\n";
  }

  if (auto Errc = sys::fs::remove(PerfTasksErrPath.data())) {
    outs() << "PERF2BOLT: Failed to delete temporary file "
           << PerfTasksErrPath << " with error " << Errc.message() << "\n";
  }

  if (auto Errc = sys::fs::remove(PerfTasksOutputPath.data())) {
    outs() << "PERF2BOLT: Failed to delete temporary file "
           << PerfTasksOutputPath << " with error " << Errc.message() << "\n";
  }
}

bool DataAggregator::aggregate(BinaryContext &BC,
                               std::map<uint64_t, BinaryFunction> &BFs) {
  std::string Error;

  this->BC = &BC;
  this->BFs = &BFs;

  outs() << "PERF2BOLT: Waiting for perf tasks collection to finish...\n";
  auto PI1 = sys::Wait(TasksPI, 0, true, &Error);

  if (!Error.empty()) {
    errs() << "PERF-ERROR: " << Error << "\n";
    deleteTempFiles();
    exit(1);
  }

  if (PI1.ReturnCode != 0) {
    ErrorOr<std::unique_ptr<MemoryBuffer>> MB =
      MemoryBuffer::getFileOrSTDIN(PerfTasksErrPath.data());
    StringRef ErrBuf = (*MB)->getBuffer();

    errs() << "PERF-ERROR: Return code " << PI1.ReturnCode << "\n";
    errs() << ErrBuf;
    deleteTempFiles();
    exit(1);
  }

  ErrorOr<std::unique_ptr<MemoryBuffer>> MB1 =
    MemoryBuffer::getFileOrSTDIN(PerfTasksOutputPath.data());
  if (std::error_code EC = MB1.getError()) {
    errs() << "Cannot open " << PerfTasksOutputPath.data() << ": "
           << EC.message() << "\n";
    deleteTempFiles();
    exit(1);
  }

  FileBuf.reset(MB1->release());
  ParsingBuf = FileBuf->getBuffer();
  Col = 0;
  Line = 1;
  if (parseTasks()) {
    outs() << "PERF2BOLT: Failed to parse tasks\n";
  }

  outs()
      << "PERF2BOLT: Waiting for perf events collection to finish...\n";
  auto PI2 = sys::Wait(EventsPI, 0, true, &Error);

  if (!Error.empty()) {
    errs() << "PERF-ERROR: " << Error << "\n";
    deleteTempFiles();
    exit(1);
  }

  if (PI2.ReturnCode != 0) {
    ErrorOr<std::unique_ptr<MemoryBuffer>> MB =
      MemoryBuffer::getFileOrSTDIN(PerfEventsErrPath.data());
    StringRef ErrBuf = (*MB)->getBuffer();

    errs() << "PERF-ERROR: Return code " << PI2.ReturnCode << "\n";
    errs() << ErrBuf;
    deleteTempFiles();
    exit(1);
  }

  ErrorOr<std::unique_ptr<MemoryBuffer>> MB2 =
    MemoryBuffer::getFileOrSTDIN(PerfEventsOutputPath.data());
  if (std::error_code EC = MB2.getError()) {
    errs() << "Cannot open " << PerfEventsOutputPath.data() << ": "
           << EC.message() << "\n";
    deleteTempFiles();
    exit(1);
  }

  FileBuf.reset(MB2->release());
  deleteTempFiles();
  ParsingBuf = FileBuf->getBuffer();
  Col = 0;
  Line = 1;
  if (parseEvents()) {
    outs() << "PERF2BOLT: Failed to parse events\n";
  }

  return true;
}

BinaryFunction *
DataAggregator::getBinaryFunctionContainingAddress(uint64_t Address) {
  auto FI = BFs->upper_bound(Address);
  if (FI == BFs->begin())
    return nullptr;
  --FI;

  const auto UsedSize = FI->second.getMaxSize();
  if (Address >= FI->first + UsedSize)
    return nullptr;
  return &FI->second;
}

bool DataAggregator::doIntraBranch(BinaryFunction *Func, uint64_t From,
                                   uint64_t To, bool Mispred) {
  FuncBranchData *AggrData = Func->getBranchData();
  if (!AggrData) {
    AggrData = &FuncsToBranches[Func->getNames()[0]];
    AggrData->Name = Func->getNames()[0];
    Func->setBranchData(AggrData);
  }

  From -= Func->getAddress();
  To -= Func->getAddress();
  AggrData->bumpBranchCount(From, To, Mispred);
  return true;
}

bool DataAggregator::doInterBranch(BinaryFunction *FromFunc,
                                   BinaryFunction *ToFunc, uint64_t From,
                                   uint64_t To, bool Mispred) {
  FuncBranchData *FromAggrData{nullptr};
  FuncBranchData *ToAggrData{nullptr};
  StringRef SrcFunc;
  StringRef DstFunc;
  if (FromFunc) {
    SrcFunc = FromFunc->getNames()[0];
    FromAggrData = FromFunc->getBranchData();
    if (!FromAggrData) {
      FromAggrData = &FuncsToBranches[SrcFunc];
      FromAggrData->Name = SrcFunc;
      FromFunc->setBranchData(FromAggrData);
    }
    From -= FromFunc->getAddress();
  }
  if (ToFunc) {
    DstFunc = ToFunc->getNames()[0];
    ToAggrData = ToFunc->getBranchData();
    if (!ToAggrData) {
      ToAggrData = &FuncsToBranches[DstFunc];
      ToAggrData->Name = DstFunc;
      ToFunc->setBranchData(ToAggrData);
    }
    To -= ToFunc->getAddress();
  }

  if (FromAggrData)
    FromAggrData->bumpCallCount(From, Location(!DstFunc.empty(), DstFunc, To),
                                Mispred);
  if (ToAggrData)
    ToAggrData->bumpEntryCount(Location(!SrcFunc.empty(), SrcFunc, From), To,
                               Mispred);
  return true;
}

bool DataAggregator::doBranch(uint64_t From, uint64_t To, bool Mispred) {
  auto *FromFunc = getBinaryFunctionContainingAddress(From);
  auto *ToFunc = getBinaryFunctionContainingAddress(To);
  if (!FromFunc && !ToFunc)
    return false;

  if (FromFunc == ToFunc)
    return doIntraBranch(FromFunc, From, To, Mispred);

  return doInterBranch(FromFunc, ToFunc, From, To, Mispred);
}

bool DataAggregator::doTrace(uint64_t From, uint64_t To) {
  auto *FromFunc = getBinaryFunctionContainingAddress(From);
  auto *ToFunc = getBinaryFunctionContainingAddress(To);
  if (!FromFunc || !ToFunc) {
    ++NumLongRangeTraces;
    return false;
  }
  if (FromFunc != ToFunc) {
    ++NumInvalidTraces;
    DEBUG(dbgs() << "Trace starting in " << FromFunc->getPrintName() << " @ "
                 << Twine::utohexstr(From - FromFunc->getAddress())
                 << " and ending in " << ToFunc->getPrintName() << " @ "
                 << ToFunc->getPrintName() << " @ "
                 << Twine::utohexstr(To - ToFunc->getAddress()) << "\n");
    return false;
  }
  if (FromFunc) {
    From -= FromFunc->getAddress();
    To -= ToFunc->getAddress();
  }

  auto FTs = FromFunc->getFallthroughsInTrace(From, To);
  if (!FTs) {
    ++NumInvalidTraces;
    return false;
  }

  for (const auto &Pair : *FTs) {
    doIntraBranch(FromFunc, Pair.first + FromFunc->getAddress(),
                  Pair.second + FromFunc->getAddress(), false);
  }

  return true;
}

ErrorOr<LBREntry> DataAggregator::parseLBREntry() {
  LBREntry Res;
  auto FromStrRes = parseString('/');
  if (std::error_code EC = FromStrRes.getError())
    return EC;
  StringRef OffsetStr = FromStrRes.get();
  if (OffsetStr.getAsInteger(0, Res.From)) {
    reportError("expected hexadecimal number with From address");
    Diag << "Found: " << OffsetStr << "\n";
    return make_error_code(llvm::errc::io_error);
  }

  auto ToStrRes = parseString('/');
  if (std::error_code EC = ToStrRes.getError())
    return EC;
  OffsetStr = ToStrRes.get();
  if (OffsetStr.getAsInteger(0, Res.To)) {
    reportError("expected hexadecimal number with To address");
    Diag << "Found: " << OffsetStr << "\n";
    return make_error_code(llvm::errc::io_error);
  }

  auto MispredStrRes = parseString('/');
  if (std::error_code EC = MispredStrRes.getError())
    return EC;
  StringRef MispredStr = MispredStrRes.get();
  if (MispredStr.size() != 1 ||
      (MispredStr[0] != 'P' && MispredStr[0] != 'M')) {
    reportError("expected single char for mispred bit");
    Diag << "Found: " << OffsetStr << "\n";
    return make_error_code(llvm::errc::io_error);
  }
  Res.Mispred = MispredStr[0] == 'M';

  auto Rest = parseString(FieldSeparator, true);
  if (std::error_code EC = Rest.getError())
    return EC;
  if (Rest.get().size() < 5) {
    reportError("expected rest of LBR entry");
    Diag << "Found: " << OffsetStr << "\n";
    return make_error_code(llvm::errc::io_error);
  }
  return Res;
}

bool DataAggregator::checkAndConsumeFS() {
  if (ParsingBuf[0] != FieldSeparator) {
    return false;
  }
  ParsingBuf = ParsingBuf.drop_front(1);
  Col += 1;
  return true;
}

void DataAggregator::consumeRestOfLine() {
  auto LineEnd = ParsingBuf.find_first_of('\n');
  if (LineEnd == StringRef::npos) {
    ParsingBuf = StringRef();
    Col = 0;
    Line += 1;
    return;
  }
  ParsingBuf = ParsingBuf.drop_front(LineEnd + 1);
  Col = 0;
  Line += 1;
}

ErrorOr<PerfSample> DataAggregator::parseSample() {
  PerfSample Res;

  while (checkAndConsumeFS()) {}

  auto PIDRes = parseNumberField(FieldSeparator, true);
  if (std::error_code EC = PIDRes.getError())
    return EC;
  if (!PIDs.empty() && !PIDs.count(PIDRes.get())) {
    consumeRestOfLine();
    return Res;
  }

  while (!checkAndConsumeNewLine()) {
    if (!expectAndConsumeFS())
      return make_error_code(llvm::errc::io_error);

    auto LBRRes = parseLBREntry();
    if (std::error_code EC = LBRRes.getError())
      return EC;
    Res.LBR.push_back(LBRRes.get());
  }

  return Res;
}

bool DataAggregator::hasData() {
  if (ParsingBuf.size() == 0)
    return false;

  return true;
}

std::error_code DataAggregator::parseEvents() {
  outs() << "PERF2BOLT: Aggregating...\n";
  NamedRegionTimer T("Samples parsing", TimerGroupName, opts::TimeAggregator);
  uint64_t NumEntries{0};
  uint64_t NumSamples{0};
  uint64_t NumTraces{0};
  while (hasData()) {
    auto SampleRes = parseSample();
    if (std::error_code EC = SampleRes.getError())
      return EC;

    auto &Sample = SampleRes.get();
    if (Sample.LBR.empty())
      continue;

    ++NumSamples;
    NumEntries += Sample.LBR.size();

    // Parser semantic actions
    uint64_t Last{0};
    for (const auto &LBR : Sample.LBR) {
      if (Last) {
        doTrace(LBR.To, Last);
        ++NumTraces;
      }
      doBranch(LBR.From, LBR.To, LBR.Mispred);
      Last = LBR.From;
    }
  }
  outs() << "PERF2BOLT: Read " << NumSamples << " samples and "
         << NumEntries << " LBR entries\n";
  outs() << "PERF2BOLT: Traces mismatching disassembled function contents: "
         << NumInvalidTraces;
  float Perc{0.0f};
  if (NumTraces > 0) {
    outs() << " (";
    Perc = NumInvalidTraces * 100.0f / NumTraces;
    if (outs().has_colors()) {
      if (Perc > 10.0f) {
        outs().changeColor(raw_ostream::RED);
      } else if (Perc > 5.0f) {
        outs().changeColor(raw_ostream::YELLOW);
      } else {
        outs().changeColor(raw_ostream::GREEN);
      }
    }
    outs() << format("%.1f%%", Perc);
    outs().resetColor();
    outs() << ")";
  }
  outs() << "\n";
  if (Perc > 10.0f) {
    outs() << "\n !! WARNING !! This high mismatch ratio indicates the input "
              "binary is probably not the same binary used during profiling "
              "collection. The generated data may be ineffective for improving "
              "performance.\n\n";
  }

  outs() << "PERF2BOLT: Out of range traces involving unknown regions: "
         << NumLongRangeTraces;
  if (NumTraces > 0) {
    outs() << format(" (%.1f%%)", NumLongRangeTraces * 100.0f / NumTraces);
  }
  outs() << "\n";

  return std::error_code();
}

ErrorOr<int64_t> DataAggregator::parseTaskPID() {
  while (checkAndConsumeFS()) {}

  auto CommNameStr = parseString(FieldSeparator, true);
  if (std::error_code EC = CommNameStr.getError())
    return EC;
  if (CommNameStr.get() != BinaryName) {
    consumeRestOfLine();
    return -1;
  }

  auto LineEnd = ParsingBuf.find_first_of("\n");
  if (LineEnd == StringRef::npos) {
    reportError("expected rest of line");
    Diag << "Found: " << ParsingBuf << "\n";
    return make_error_code(llvm::errc::io_error);
  }

  StringRef Line = ParsingBuf.substr(0, LineEnd);

  if (Line.find("PERF_RECORD_COMM") != StringRef::npos) {
    int64_t PID;
    StringRef PIDStr = Line.rsplit(':').second.split('/').first;
    if (PIDStr.getAsInteger(10, PID)) {
      reportError("expected PID");
      Diag << "Found: " << PIDStr << "\n";
      return make_error_code(llvm::errc::io_error);
    }
    return PID;
  }

  consumeRestOfLine();
  return -1;
}

std::error_code DataAggregator::parseTasks() {
  outs() << "PERF2BOLT: Parsing perf-script tasks output\n";
  NamedRegionTimer T("Tasks parsing", TimerGroupName, opts::TimeAggregator);

  while (hasData()) {
    auto PIDRes = parseTaskPID();
    if (std::error_code EC = PIDRes.getError())
      return EC;

    auto PID = PIDRes.get();
    if (PID == -1) {
      continue;
    }

    PIDs.insert(PID);
  }
  if (!PIDs.empty())
    outs() << "PERF2BOLT: Input binary is associated with " << PIDs.size()
           << " PID(s)\n";
  else
    outs() << "PERF2BOLT: Could not bind input binary to a PID - will parse "
              "all samples in perf data.\n";

  return std::error_code();
}

std::error_code DataAggregator::writeAggregatedFile() const {
  std::error_code EC;
  raw_fd_ostream OutFile(OutputFDataName, EC, sys::fs::OpenFlags::F_None);
  if (EC)
    return EC;

  uint64_t Values{0};
  for (const auto &Func : FuncsToBranches) {
    for (const auto &BI : Func.getValue().Data) {
      OutFile << (BI.From.IsSymbol ? "1 " : "0 ")
              << (BI.From.Name.empty() ? "[unknown]" : BI.From.Name)  << " "
              << Twine::utohexstr(BI.From.Offset) << " "
              << (BI.To.IsSymbol ? "1 " : "0 ")
              << (BI.To.Name.empty() ? "[unknown]" : BI.To.Name)  << " "
              << Twine::utohexstr(BI.To.Offset) << " " << BI.Mispreds << " "
              << BI.Branches << "\n";
      ++Values;
    }
    for (const auto &BI : Func.getValue().EntryData) {
      // Do not output if source is a known symbol, since this was already
      // accounted for in the source function
      if (BI.From.IsSymbol)
        continue;
      OutFile << (BI.From.IsSymbol ? "1 " : "0 ")
              << (BI.From.Name.empty() ? "[unknown]" : BI.From.Name)  << " "
              << Twine::utohexstr(BI.From.Offset) << " "
              << (BI.To.IsSymbol ? "1 " : "0 ")
              << (BI.To.Name.empty() ? "[unknown]" : BI.To.Name)  << " "
              << Twine::utohexstr(BI.To.Offset) << " " << BI.Mispreds << " "
              << BI.Branches << "\n";
      ++Values;
    }
  }
  outs() << "PERF2BOLT: Wrote " << Values << " objects to "
         << OutputFDataName << "\n";

  return std::error_code();
}

void DataAggregator::dump() const {
  DataReader::dump();
}

void DataAggregator::dump(const LBREntry &LBR) const {
  Diag << "From: " << Twine::utohexstr(LBR.From)
       << " To: " << Twine::utohexstr(LBR.To) << " Mispred? " << LBR.Mispred
       << "\n";
}

void DataAggregator::dump(const PerfSample &Sample) const {
  Diag << "Sample LBR entries: " << Sample.LBR.size() << "\n";
  for (const auto &LBR : Sample.LBR) {
    dump(LBR);
  }
}
