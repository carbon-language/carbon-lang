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
#include "llvm/Support/Regex.h"
#include "llvm/Support/Timer.h"
#include <map>

#include <unistd.h>

#define DEBUG_TYPE "aggregator"

using namespace llvm;
using namespace bolt;

namespace opts {

extern cl::OptionCategory AggregatorCategory;

static cl::opt<bool>
BasicAggregation("nl",
  cl::desc("aggregate basic samples (without LBR info)"),
  cl::init(false),
  cl::ZeroOrMore,
  cl::cat(AggregatorCategory));

static cl::opt<bool>
ReadPreAggregated("pa",
  cl::desc("skip perf and read data from a pre-aggregated file format"),
  cl::init(false),
  cl::ZeroOrMore,
  cl::cat(AggregatorCategory));

static cl::opt<bool>
IgnoreBuildID("ignore-build-id",
  cl::desc("continue even if build-ids in input binary and perf.data mismatch"),
  cl::init(false),
  cl::cat(AggregatorCategory));

static cl::opt<bool>
TimeAggregator("time-aggr",
  cl::desc("time BOLT aggregator"),
  cl::init(false),
  cl::ZeroOrMore,
  cl::cat(AggregatorCategory));

}

namespace {

const char TimerGroupName[] = "aggregator";
const char TimerGroupDesc[] = "Aggregator";

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
  this->PerfDataFilename = PerfDataFilename;
  outs() << "PERF2BOLT: Starting data aggregation job for " << PerfDataFilename
         << "\n";

  // Don't launch perf for pre-aggregated files
  if (opts::ReadPreAggregated)
    return;

  findPerfExecutable();
  launchPerfBranchEventsNoWait();
  launchPerfMemEventsNoWait();
  launchPerfMMapEventsNoWait();
}

void DataAggregator::abort() {
  if (opts::ReadPreAggregated)
    return;

  std::string Error;

  // Kill subprocesses in case they are not finished
  sys::Wait(MMapEventsPI, 1, false, &Error);
  sys::Wait(BranchEventsPI, 1, false, &Error);
  sys::Wait(MemEventsPI, 1, false, &Error);

  deleteTempFiles();
}

bool DataAggregator::launchPerfBranchEventsNoWait() {
  SmallVector<const char*, 4> Argv;

  if (opts::BasicAggregation)
    outs()
        << "PERF2BOLT: Spawning perf-script job to read events without LBR\n";
  else
    outs() << "PERF2BOLT: Spawning perf-script job to read branch events\n";
  Argv.push_back(PerfPath.data());
  Argv.push_back("script");
  Argv.push_back("-F");
  if (opts::BasicAggregation)
    Argv.push_back("pid,event,ip");
  else
    Argv.push_back("pid,brstack");
  Argv.push_back("-i");
  Argv.push_back(PerfDataFilename.data());
  Argv.push_back(nullptr);

  if (auto Errc = sys::fs::createTemporaryFile("perf.script", "out",
                                               PerfBranchEventsOutputPath)) {
    outs() << "PERF2BOLT: Failed to create temporary file "
           << PerfBranchEventsOutputPath << " with error " << Errc.message()
           << "\n";
    exit(1);
  }

  if (auto Errc = sys::fs::createTemporaryFile("perf.script", "err",
                                               PerfBranchEventsErrPath)) {
    outs() << "PERF2BOLT: Failed to create temporary file "
           << PerfBranchEventsErrPath << " with error " << Errc.message()
           << "\n";
    exit(1);
  }
  Optional<StringRef> Redirects[] = {
      llvm::None,                                   // Stdin
      StringRef(PerfBranchEventsOutputPath.data()), // Stdout
      StringRef(PerfBranchEventsErrPath.data())};   // Stderr

  DEBUG(dbgs() << "Launching perf: " << PerfPath.data() << " 1> "
               << PerfBranchEventsOutputPath.data() << " 2> "
               << PerfBranchEventsErrPath.data() << "\n");

  BranchEventsPI = sys::ExecuteNoWait(PerfPath.data(), Argv.data(),
                                      /*envp*/ nullptr, Redirects);

  return true;
}

bool DataAggregator::launchPerfMemEventsNoWait() {
  SmallVector<const char*, 4> Argv;

  outs() << "PERF2BOLT: Spawning perf-script job to read mem events\n";
  Argv.push_back(PerfPath.data());
  Argv.push_back("script");
  Argv.push_back("-F");
  Argv.push_back("pid,event,addr,ip");
  Argv.push_back("-i");
  Argv.push_back(PerfDataFilename.data());
  Argv.push_back(nullptr);

  if (auto Errc = sys::fs::createTemporaryFile("perf.script", "out",
                                               PerfMemEventsOutputPath)) {
    outs() << "PERF2BOLT: Failed to create temporary file "
           << PerfMemEventsOutputPath << " with error " << Errc.message() << "\n";
    exit(1);
  }

  if (auto Errc = sys::fs::createTemporaryFile("perf.script", "err",
                                               PerfMemEventsErrPath)) {
    outs() << "PERF2BOLT: Failed to create temporary file "
           << PerfMemEventsErrPath << " with error " << Errc.message() << "\n";
    exit(1);
  }

  Optional<StringRef> Redirects[] = {
      llvm::None,                                // Stdin
      StringRef(PerfMemEventsOutputPath.data()), // Stdout
      StringRef(PerfMemEventsErrPath.data())};   // Stderr

  DEBUG(dbgs() << "Launching perf: " << PerfPath.data() << " 1> "
               << PerfMemEventsOutputPath.data() << " 2> "
               << PerfMemEventsErrPath.data() << "\n");

  MemEventsPI = sys::ExecuteNoWait(PerfPath.data(), Argv.data(),
                                   /*envp*/ nullptr, Redirects);

  return true;
}

bool DataAggregator::launchPerfMMapEventsNoWait() {
  SmallVector<const char*, 4> Argv;

  outs() << "PERF2BOLT: Spawning perf-script job to read process info\n";
  Argv.push_back(PerfPath.data());
  Argv.push_back("script");
  Argv.push_back("--show-mmap-events");
  Argv.push_back("-i");
  Argv.push_back(PerfDataFilename.data());
  Argv.push_back(nullptr);

  if (auto Errc = sys::fs::createTemporaryFile("perf.script", "out",
                                               PerfMMapEventsOutputPath)) {
    outs() << "PERF2BOLT: Failed to create temporary file "
           << PerfMMapEventsOutputPath << " with error " << Errc.message()
           << "\n";
    exit(1);
  }

  if (auto Errc = sys::fs::createTemporaryFile("perf.script", "err",
                                               PerfMMapEventsErrPath)) {
    outs() << "PERF2BOLT: Failed to create temporary file "
           << PerfMMapEventsErrPath << " with error " << Errc.message() << "\n";
    exit(1);
  }

  Optional<StringRef> Redirects[] = {
      llvm::None,                                 // Stdin
      StringRef(PerfMMapEventsOutputPath.data()), // Stdout
      StringRef(PerfMMapEventsErrPath.data())};   // Stderr

  DEBUG(dbgs() << "Launching perf: " << PerfPath.data() << " 1> "
               << PerfMMapEventsOutputPath.data() << " 2> "
               << PerfMMapEventsErrPath.data() << "\n");

  MMapEventsPI = sys::ExecuteNoWait(PerfPath.data(), Argv.data(),
                                    /*envp*/ nullptr, Redirects);

  return true;
}

void DataAggregator::processFileBuildID(StringRef FileBuildID) {
  if (opts::ReadPreAggregated)
    return;

  SmallVector<const char *, 4> Argv;
  SmallVector<char, 256> OutputPath;
  SmallVector<char, 256> ErrPath;

  Argv.push_back(PerfPath.data());
  Argv.push_back("buildid-list");
  Argv.push_back("-i");
  Argv.push_back(PerfDataFilename.data());
  Argv.push_back(nullptr);

  if (auto Errc = sys::fs::createTemporaryFile("perf.buildid", "out",
                                               OutputPath)) {
    outs() << "PERF2BOLT: Failed to create temporary file "
           << OutputPath << " with error " << Errc.message() << "\n";
    exit(1);
  }

  if (auto Errc = sys::fs::createTemporaryFile("perf.script", "err",
                                               ErrPath)) {
    outs() << "PERF2BOLT: Failed to create temporary file "
           << ErrPath << " with error " << Errc.message() << "\n";
    exit(1);
  }

  Optional<StringRef> Redirects[] = {
      llvm::None,                   // Stdin
      StringRef(OutputPath.data()), // Stdout
      StringRef(ErrPath.data())};   // Stderr

  DEBUG(dbgs() << "Launching perf: " << PerfPath.data() << " 1> "
               << OutputPath.data() << " 2> "
               << ErrPath.data() << "\n");

  auto RetCode = sys::ExecuteAndWait(PerfPath.data(), Argv.data(),
                                     /*envp*/ nullptr, Redirects);

  if (RetCode != 0) {
    ErrorOr<std::unique_ptr<MemoryBuffer>> MB =
      MemoryBuffer::getFileOrSTDIN(ErrPath.data());
    StringRef ErrBuf = (*MB)->getBuffer();

    errs() << "PERF-ERROR: Return code " << RetCode << "\n";
    errs() << ErrBuf;
    deleteTempFile(ErrPath.data());
    deleteTempFile(OutputPath.data());
    return;
  }

  ErrorOr<std::unique_ptr<MemoryBuffer>> MB =
    MemoryBuffer::getFileOrSTDIN(OutputPath.data());
  if (std::error_code EC = MB.getError()) {
    errs() << "Cannot open " << PerfMMapEventsOutputPath.data() << ": "
           << EC.message() << "\n";
    deleteTempFile(ErrPath.data());
    deleteTempFile(OutputPath.data());
    return;
  }

  FileBuf.reset(MB->release());
  ParsingBuf = FileBuf->getBuffer();
  if (ParsingBuf.empty()) {
    errs() << "PERF2BOLT-WARNING: build-id will not be checked because perf "
              "data was recorded without it\n";
    deleteTempFile(ErrPath.data());
    deleteTempFile(OutputPath.data());
    return;
  }

  Col = 0;
  Line = 1;
  auto FileName = getFileNameForBuildID(FileBuildID);
  if (!FileName) {
    errs() << "PERF2BOLT-ERROR: failed to match build-id from perf output. "
              "This indicates the input binary supplied for data aggregation "
              "is not the same recorded by perf when collecting profiling "
              "data, or there were no samples recorded for the binary. "
              "Use -ignore-build-id option to override.\n";
    if (!opts::IgnoreBuildID) {
      deleteTempFile(ErrPath.data());
      deleteTempFile(OutputPath.data());
      abort();
      exit(1);
    }
  } else if (*FileName != BinaryName) {
    errs() << "PERF2BOLT-WARNING: build-id matched a different file name\n";
    BuildIDBinaryName = *FileName;
  } else {
    outs() << "PERF2BOLT: matched build-id and file name\n";
  }

  deleteTempFile(ErrPath.data());
  deleteTempFile(OutputPath.data());
  return;
}

bool DataAggregator::checkPerfDataMagic(StringRef FileName) {
  if (opts::ReadPreAggregated)
    return true;

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

void DataAggregator::deleteTempFile(StringRef File) {
  if (auto Errc = sys::fs::remove(File.data())) {
    outs() << "PERF2BOLT: Failed to delete temporary file "
           << File << " with error " << Errc.message() << "\n";
  }
}

void DataAggregator::deleteTempFiles() {
  deleteTempFile(PerfBranchEventsErrPath.data());
  deleteTempFile(PerfBranchEventsOutputPath.data());
  deleteTempFile(PerfMemEventsErrPath.data());
  deleteTempFile(PerfMemEventsOutputPath.data());
  deleteTempFile(PerfMMapEventsErrPath.data());
  deleteTempFile(PerfMMapEventsOutputPath.data());
}

bool DataAggregator::processPreAggregated() {
  std::string Error;

  auto MB = MemoryBuffer::getFileOrSTDIN(PerfDataFilename);
  if (std::error_code EC = MB.getError()) {
    errs() << "PERF2BOLT-ERROR: cannot open " << PerfDataFilename << ": "
           << EC.message() << "\n";
    exit(1);
  }

  FileBuf.reset(MB->release());
  ParsingBuf = FileBuf->getBuffer();
  Col = 0;
  Line = 1;
  if (parseAggregatedLBRSamples()) {
    outs() << "PERF2BOLT: Failed to parse samples\n";
    exit(1);
  }

  // Mark all functions with registered events as having a valid profile.
  for (auto &BFI : *BFs) {
    auto &BF = BFI.second;
    if (BF.getBranchData()) {
      const auto Flags = opts::BasicAggregation ? BinaryFunction::PF_SAMPLE
                                                : BinaryFunction::PF_LBR;
      BF.markProfiled(Flags);
    }
  }

  return true;
}

bool DataAggregator::aggregate(BinaryContext &BC,
                               std::map<uint64_t, BinaryFunction> &BFs) {
  std::string Error;

  this->BC = &BC;
  this->BFs = &BFs;

  if (opts::ReadPreAggregated)
    return processPreAggregated();

  outs() << "PERF2BOLT: Waiting for perf mmap events collection to finish...\n";
  auto PI1 = sys::Wait(MMapEventsPI, 0, true, &Error);

  if (!Error.empty()) {
    errs() << "PERF-ERROR: " << Error << "\n";
    deleteTempFiles();
    exit(1);
  }

  if (PI1.ReturnCode != 0) {
    ErrorOr<std::unique_ptr<MemoryBuffer>> MB =
      MemoryBuffer::getFileOrSTDIN(PerfMMapEventsErrPath.data());
    StringRef ErrBuf = (*MB)->getBuffer();

    errs() << "PERF-ERROR: Return code " << PI1.ReturnCode << "\n";
    errs() << ErrBuf;
    deleteTempFiles();
    exit(1);
  }

  ErrorOr<std::unique_ptr<MemoryBuffer>> MB1 =
    MemoryBuffer::getFileOrSTDIN(PerfMMapEventsOutputPath.data());
  if (std::error_code EC = MB1.getError()) {
    errs() << "Cannot open " << PerfMMapEventsOutputPath.data() << ": "
           << EC.message() << "\n";
    deleteTempFiles();
    exit(1);
  }

  FileBuf.reset(MB1->release());
  ParsingBuf = FileBuf->getBuffer();
  Col = 0;
  Line = 1;
  if (parseMMapEvents()) {
    outs() << "PERF2BOLT: Failed to parse mmap events\n";
  }

  outs()
      << "PERF2BOLT: Waiting for perf events collection to finish...\n";
  auto PI2 = sys::Wait(BranchEventsPI, 0, true, &Error);

  if (!Error.empty()) {
    errs() << "PERF-ERROR: " << Error << "\n";
    deleteTempFiles();
    exit(1);
  }

  if (PI2.ReturnCode != 0) {
    ErrorOr<std::unique_ptr<MemoryBuffer>> MB =
      MemoryBuffer::getFileOrSTDIN(PerfBranchEventsErrPath.data());
    StringRef ErrBuf = (*MB)->getBuffer();

    errs() << "PERF-ERROR: Return code " << PI2.ReturnCode << "\n";
    errs() << ErrBuf;
    deleteTempFiles();
    exit(1);
  }

  ErrorOr<std::unique_ptr<MemoryBuffer>> MB2 =
    MemoryBuffer::getFileOrSTDIN(PerfBranchEventsOutputPath.data());
  if (std::error_code EC = MB2.getError()) {
    errs() << "Cannot open " << PerfBranchEventsOutputPath.data() << ": "
           << EC.message() << "\n";
    deleteTempFiles();
    exit(1);
  }

  FileBuf.reset(MB2->release());
  ParsingBuf = FileBuf->getBuffer();
  Col = 0;
  Line = 1;
  if ((!opts::BasicAggregation && parseBranchEvents()) ||
      (opts::BasicAggregation && parseBasicEvents())) {
    outs() << "PERF2BOLT: Failed to parse samples\n";
  }

  // Mark all functions with registered events as having a valid profile.
  for (auto &BFI : BFs) {
    auto &BF = BFI.second;
    if (BF.getBranchData()) {
      const auto Flags = opts::BasicAggregation ? BinaryFunction::PF_SAMPLE
                                                : BinaryFunction::PF_LBR;
      BF.markProfiled(Flags);
    }
  }

  auto PI3 = sys::Wait(MemEventsPI, 0, true, &Error);
  if (PI3.ReturnCode != 0) {
    ErrorOr<std::unique_ptr<MemoryBuffer>> MB =
      MemoryBuffer::getFileOrSTDIN(PerfMemEventsErrPath.data());
    StringRef ErrBuf = (*MB)->getBuffer();

    deleteTempFiles();

    Regex NoData("Samples for '.*' event do not have ADDR attribute set. "
                 "Cannot print 'addr' field.");
    if (!NoData.match(ErrBuf)) {
      errs() << "PERF-ERROR: Return code " << PI3.ReturnCode << "\n";
      errs() << ErrBuf;
      exit(1);
    }
    return true;
  }

  ErrorOr<std::unique_ptr<MemoryBuffer>> MB3 =
    MemoryBuffer::getFileOrSTDIN(PerfMemEventsOutputPath.data());
  if (std::error_code EC = MB3.getError()) {
    errs() << "Cannot open " << PerfMemEventsOutputPath.data() << ": "
           << EC.message() << "\n";
    deleteTempFiles();
    exit(1);
  }

  FileBuf.reset(MB3->release());
  ParsingBuf = FileBuf->getBuffer();
  Col = 0;
  Line = 1;
  if (const auto EC = parseMemEvents()) {
    errs() << "PERF2BOLT: Failed to parse memory events: "
           << EC.message() << '\n';
  }

  deleteTempFiles();

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

bool
DataAggregator::doSample(BinaryFunction &Func, uint64_t Address) {
  auto I = FuncsToSamples.find(Func.getNames()[0]);
  if (I == FuncsToSamples.end()) {
    bool Success;
    std::tie(I, Success) = FuncsToSamples.insert(std::make_pair(
        Func.getNames()[0],
        FuncSampleData(Func.getNames()[0], FuncSampleData::ContainerTy())));
  }

  I->second.bumpCount(Address - Func.getAddress());
  return true;
}

bool DataAggregator::doIntraBranch(BinaryFunction &Func, uint64_t From,
                                   uint64_t To, uint64_t Count,
                                   uint64_t Mispreds) {
  FuncBranchData *AggrData = Func.getBranchData();
  if (!AggrData) {
    AggrData = &FuncsToBranches[Func.getNames()[0]];
    AggrData->Name = Func.getNames()[0];
    Func.setBranchData(AggrData);
  }

  AggrData->bumpBranchCount(From - Func.getAddress(), To - Func.getAddress(),
                            Count, Mispreds);
  return true;
}

bool DataAggregator::doInterBranch(BinaryFunction *FromFunc,
                                   BinaryFunction *ToFunc, uint64_t From,
                                   uint64_t To, uint64_t Count,
                                   uint64_t Mispreds) {
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

    FromFunc->recordExit(From, Mispreds, Count);
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

    ToFunc->recordEntry(To, Mispreds, Count);
  }

  if (FromAggrData)
    FromAggrData->bumpCallCount(From, Location(!DstFunc.empty(), DstFunc, To),
                                Count, Mispreds);
  if (ToAggrData)
    ToAggrData->bumpEntryCount(Location(!SrcFunc.empty(), SrcFunc, From), To,
                               Count, Mispreds);
  return true;
}

bool DataAggregator::doBranch(uint64_t From, uint64_t To, uint64_t Count,
                              uint64_t Mispreds) {
  auto *FromFunc = getBinaryFunctionContainingAddress(From);
  auto *ToFunc = getBinaryFunctionContainingAddress(To);
  if (!FromFunc && !ToFunc)
    return false;

  if (FromFunc == ToFunc) {
    FromFunc->recordBranch(From - FromFunc->getAddress(),
                           To - FromFunc->getAddress(), Count, Mispreds);
    return doIntraBranch(*FromFunc, From, To, Count, Mispreds);
  }

  return doInterBranch(FromFunc, ToFunc, From, To, Count, Mispreds);
}

bool DataAggregator::doTrace(const LBREntry &First, const LBREntry &Second,
                             uint64_t Count) {
  auto *FromFunc = getBinaryFunctionContainingAddress(First.To);
  auto *ToFunc = getBinaryFunctionContainingAddress(Second.From);
  if (!FromFunc || !ToFunc) {
    NumLongRangeTraces += Count;
    return false;
  }
  if (FromFunc != ToFunc) {
    NumInvalidTraces += Count;
    DEBUG(dbgs() << "Trace starting in " << FromFunc->getPrintName() << " @ "
                 << Twine::utohexstr(First.To - FromFunc->getAddress())
                 << " and ending in " << ToFunc->getPrintName() << " @ "
                 << ToFunc->getPrintName() << " @ "
                 << Twine::utohexstr(Second.From - ToFunc->getAddress())
                 << '\n');
    return false;
  }

  auto FTs = FromFunc->getFallthroughsInTrace(First, Second, Count);
  if (!FTs) {
    NumInvalidTraces += Count;
    return false;
  }

  for (const auto &Pair : *FTs) {
    doIntraBranch(*FromFunc, Pair.first + FromFunc->getAddress(),
                  Pair.second + FromFunc->getAddress(), Count, false);
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
      (MispredStr[0] != 'P' && MispredStr[0] != 'M' && MispredStr[0] != '-')) {
    reportError("expected single char for mispred bit");
    Diag << "Found: " << MispredStr << "\n";
    return make_error_code(llvm::errc::io_error);
  }
  Res.Mispred = MispredStr[0] == 'M';

  static bool MispredWarning = true;;
  if (MispredStr[0] == '-' && MispredWarning) {
    errs() << "PERF2BOLT-WARNING: misprediction bit is missing in profile\n";
    MispredWarning = false;
  }

  auto Rest = parseString(FieldSeparator, true);
  if (std::error_code EC = Rest.getError())
    return EC;
  if (Rest.get().size() < 5) {
    reportError("expected rest of LBR entry");
    Diag << "Found: " << Rest.get() << "\n";
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

ErrorOr<PerfBranchSample> DataAggregator::parseBranchSample() {
  PerfBranchSample Res;

  while (checkAndConsumeFS()) {}

  auto PIDRes = parseNumberField(FieldSeparator, true);
  if (std::error_code EC = PIDRes.getError())
    return EC;
  auto MMapInfoIter = BinaryMMapInfo.find(*PIDRes);
  if (MMapInfoIter == BinaryMMapInfo.end()) {
    consumeRestOfLine();
    return Res;
  }

  while (!checkAndConsumeNewLine()) {
    checkAndConsumeFS();

    auto LBRRes = parseLBREntry();
    if (std::error_code EC = LBRRes.getError())
      return EC;
    auto LBR = LBRRes.get();
    if (!BC->HasFixedLoadAddress)
      adjustLBR(LBR, MMapInfoIter->second);
    Res.LBR.push_back(LBR);
  }

  return Res;
}

ErrorOr<PerfBasicSample> DataAggregator::parseBasicSample() {
  while (checkAndConsumeFS()) {}

  auto PIDRes = parseNumberField(FieldSeparator, true);
  if (std::error_code EC = PIDRes.getError())
    return EC;

  auto MMapInfoIter = BinaryMMapInfo.find(*PIDRes);
  if (MMapInfoIter == BinaryMMapInfo.end()) {
    consumeRestOfLine();
    return PerfBasicSample{StringRef(), 0};
  }

  while (checkAndConsumeFS()) {}

  auto Event = parseString(FieldSeparator);
  if (std::error_code EC = Event.getError())
    return EC;

  while (checkAndConsumeFS()) {}

  auto AddrRes = parseHexField(FieldSeparator, true);
  if (std::error_code EC = AddrRes.getError()) {
    return EC;
  }

  if (!checkAndConsumeNewLine()) {
    reportError("expected end of line");
    return make_error_code(llvm::errc::io_error);
  }

  auto Address = *AddrRes;
  if (!BC->HasFixedLoadAddress)
    adjustAddress(Address, MMapInfoIter->second);

  return PerfBasicSample{Event.get(), Address};
}

ErrorOr<PerfMemSample> DataAggregator::parseMemSample() {
  PerfMemSample Res{0,0};

  while (checkAndConsumeFS()) {}

  auto PIDRes = parseNumberField(FieldSeparator, true);
  if (std::error_code EC = PIDRes.getError())
    return EC;

  auto MMapInfoIter = BinaryMMapInfo.find(*PIDRes);
  if (MMapInfoIter == BinaryMMapInfo.end()) {
    consumeRestOfLine();
    return Res;
  }

  while (checkAndConsumeFS()) {}

  auto Event = parseString(FieldSeparator);
  if (std::error_code EC = Event.getError())
    return EC;
  if (Event.get().find("mem-loads") == StringRef::npos) {
    consumeRestOfLine();
    return Res;
  }

  while (checkAndConsumeFS()) {}

  auto AddrRes = parseHexField(FieldSeparator);
  if (std::error_code EC = AddrRes.getError()) {
    return EC;
  }

  while (checkAndConsumeFS()) {}

  auto PCRes = parseHexField(FieldSeparator, true);
  if (std::error_code EC = PCRes.getError()) {
    consumeRestOfLine();
    return EC;
  }

  if (!checkAndConsumeNewLine()) {
    reportError("expected end of line");
    return make_error_code(llvm::errc::io_error);
  }

  auto Address = *AddrRes;
  if (!BC->HasFixedLoadAddress)
    adjustAddress(Address, MMapInfoIter->second);

  return PerfMemSample{PCRes.get(), Address};
}

ErrorOr<Location> DataAggregator::parseLocationOrOffset() {
  auto parseOffset = [this]() -> ErrorOr<Location> {
    auto Res = parseHexField(FieldSeparator);
    if (std::error_code EC = Res.getError())
      return EC;
    return Location(Res.get());
  };

  auto Sep = ParsingBuf.find_first_of(" \n");
  if (Sep == StringRef::npos)
    return parseOffset();
  auto LookAhead = ParsingBuf.substr(0, Sep);
  if (LookAhead.find_first_of(":") == StringRef::npos)
    return parseOffset();

  auto BuildID = parseString(':');
  if (std::error_code EC = BuildID.getError())
    return EC;
  auto Offset = parseHexField(FieldSeparator);
  if (std::error_code EC = Offset.getError())
    return EC;
  return Location(true, BuildID.get(), Offset.get());
}

ErrorOr<AggregatedLBREntry> DataAggregator::parseAggregatedLBREntry() {
  while (checkAndConsumeFS()) {}

  auto TypeOrErr = parseString(FieldSeparator);
  if (std::error_code EC = TypeOrErr.getError())
    return EC;
  auto Type{AggregatedLBREntry::BRANCH};
  if (TypeOrErr.get() == "B") {
    Type = AggregatedLBREntry::BRANCH;
  } else if (TypeOrErr.get() == "F") {
    Type = AggregatedLBREntry::FT;
  } else if (TypeOrErr.get() == "f") {
    Type = AggregatedLBREntry::FT_EXTERNAL_ORIGIN;
  } else {
    reportError("expected B, F or f");
    return make_error_code(llvm::errc::io_error);
  }

  while (checkAndConsumeFS()) {}
  auto From = parseLocationOrOffset();
  if (std::error_code EC = From.getError())
    return EC;

  while (checkAndConsumeFS()) {}
  auto To = parseLocationOrOffset();
  if (std::error_code EC = To.getError())
    return EC;

  while (checkAndConsumeFS()) {}
  auto Frequency = parseNumberField(FieldSeparator,
                                    Type != AggregatedLBREntry::BRANCH);
  if (std::error_code EC = Frequency.getError())
    return EC;

  uint64_t Mispreds{0};
  if (Type == AggregatedLBREntry::BRANCH) {
    while (checkAndConsumeFS()) {}
    auto MispredsOrErr = parseNumberField(FieldSeparator, true);
    if (std::error_code EC = MispredsOrErr.getError())
      return EC;
    Mispreds = static_cast<uint64_t>(MispredsOrErr.get());
  }

  if (!checkAndConsumeNewLine()) {
    reportError("expected end of line");
    return make_error_code(llvm::errc::io_error);
  }

  return AggregatedLBREntry{From.get(), To.get(),
                            static_cast<uint64_t>(Frequency.get()), Mispreds,
                            Type};
}

bool DataAggregator::hasData() {
  if (ParsingBuf.size() == 0)
    return false;

  return true;
}

std::error_code DataAggregator::parseBranchEvents() {
  outs() << "PERF2BOLT: Aggregating branch events...\n";
  NamedRegionTimer T("parseBranch", "Branch samples parsing", TimerGroupName,
                     TimerGroupDesc, opts::TimeAggregator);
  uint64_t NumEntries{0};
  uint64_t NumSamples{0};
  uint64_t NumTraces{0};
  while (hasData()) {
    auto SampleRes = parseBranchSample();
    if (std::error_code EC = SampleRes.getError())
      return EC;

    auto &Sample = SampleRes.get();
    if (Sample.LBR.empty())
      continue;

    ++NumSamples;
    NumEntries += Sample.LBR.size();

    // LBRs are stored in reverse execution order. NextLBR refers to the next
    // executed branch record.
    const LBREntry *NextLBR{nullptr};
    for (const auto &LBR : Sample.LBR) {
      if (NextLBR) {
        doTrace(LBR, *NextLBR);
        ++NumTraces;
      }
      doBranch(LBR.From, LBR.To, 1, LBR.Mispred);
      NextLBR = &LBR;
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
    if (outs().has_colors())
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

std::error_code DataAggregator::parseBasicEvents() {
  outs() << "PERF2BOLT: Aggregating basic events (without LBR)...\n";
  NamedRegionTimer T("parseBasic", "Perf samples parsing", TimerGroupName,
                     TimerGroupDesc, opts::TimeAggregator);
  uint64_t NumSamples{0};
  uint64_t OutOfRangeSamples{0};
  while (hasData()) {
    auto SampleRes = parseBasicSample();
    if (std::error_code EC = SampleRes.getError())
      return EC;

    auto &Sample = SampleRes.get();
    if (!Sample.PC)
      continue;

    ++NumSamples;
    auto *Func = getBinaryFunctionContainingAddress(Sample.PC);
    if (!Func) {
      ++OutOfRangeSamples;
      continue;
    }

    doSample(*Func, Sample.PC);
    EventNames.insert(Sample.EventName);
  }
  outs() << "PERF2BOLT: Read " << NumSamples << " samples\n";

  outs() << "PERF2BOLT: Out of range samples recorded in unknown regions: "
         << OutOfRangeSamples;
  float Perc{0.0f};
  if (NumSamples > 0) {
    outs() << " (";
    Perc = OutOfRangeSamples * 100.0f / NumSamples;
    if (outs().has_colors()) {
      if (Perc > 60.0f) {
        outs().changeColor(raw_ostream::RED);
      } else if (Perc > 40.0f) {
        outs().changeColor(raw_ostream::YELLOW);
      } else {
        outs().changeColor(raw_ostream::GREEN);
      }
    }
    outs() << format("%.1f%%", Perc);
    if (outs().has_colors())
      outs().resetColor();
    outs() << ")";
  }
  outs() << "\n";
  if (Perc > 80.0f) {
    outs() << "\n !! WARNING !! This high mismatch ratio indicates the input "
              "binary is probably not the same binary used during profiling "
              "collection. The generated data may be ineffective for improving "
              "performance.\n\n";
  }

  return std::error_code();
}

std::error_code DataAggregator::parseMemEvents() {
  outs() << "PERF2BOLT: Aggregating memory events...\n";
  NamedRegionTimer T("memevents", "Mem samples parsing", TimerGroupName,
                     TimerGroupDesc, opts::TimeAggregator);

  while (hasData()) {
    auto SampleRes = parseMemSample();
    if (std::error_code EC = SampleRes.getError())
      return EC;

    auto PC = SampleRes.get().PC;
    auto Addr = SampleRes.get().Addr;
    StringRef FuncName;
    StringRef MemName;

    // Try to resolve symbol for PC
    auto *Func = getBinaryFunctionContainingAddress(PC);
    if (Func) {
      FuncName = Func->getNames()[0];
      PC -= Func->getAddress();
    }

    // Try to resolve symbol for memory load
    auto *MemFunc = getBinaryFunctionContainingAddress(Addr);
    if (MemFunc) {
      MemName = MemFunc->getNames()[0];
      Addr -= MemFunc->getAddress();
    } else if (Addr) {  // TODO: filter heap/stack/nulls here?
      if (auto *BD = BC->getBinaryDataContainingAddress(Addr)) {
        MemName = BD->getName();
        Addr -= BD->getAddress();
      }
    }

    const Location FuncLoc(!FuncName.empty(), FuncName, PC);
    const Location AddrLoc(!MemName.empty(), MemName, Addr);

    // TODO what does it mean when PC is 0 (or not a known function)?
    DEBUG(if (!Func && PC != 0) {
      dbgs() << "Skipped mem event: " << FuncLoc << " = " << AddrLoc << "\n";
    });

    if (Func) {
      auto *MemData = &FuncsToMemEvents[FuncName];
      Func->setMemData(MemData);
      MemData->update(FuncLoc, AddrLoc);
      DEBUG(dbgs() << "Mem event: " << FuncLoc << " = " << AddrLoc << "\n");
    }
  }

  return std::error_code();
}

std::error_code DataAggregator::parseAggregatedLBRSamples() {
  outs() << "PERF2BOLT: Aggregating...\n";
  NamedRegionTimer T("parseAggregated", "Aggregated LBR parsing", TimerGroupName,
                     TimerGroupDesc, opts::TimeAggregator);
  uint64_t NumAggrEntries{0};
  uint64_t NumTraces{0};
  while (hasData()) {
    auto AggrEntryRes = parseAggregatedLBREntry();
    if (std::error_code EC = AggrEntryRes.getError())
      return EC;

    auto &AggrEntry = AggrEntryRes.get();

    ++NumAggrEntries;
    switch (AggrEntry.EntryType) {
    case AggregatedLBREntry::BRANCH:
      doBranch(AggrEntry.From.Offset, AggrEntry.To.Offset, AggrEntry.Count,
               AggrEntry.Mispreds);
      break;
    case AggregatedLBREntry::FT:
    case AggregatedLBREntry::FT_EXTERNAL_ORIGIN: {
      LBREntry First{AggrEntry.EntryType == AggregatedLBREntry::FT
                         ? AggrEntry.From.Offset
                         : 0,
                     AggrEntry.From.Offset, false};
      LBREntry Second{AggrEntry.To.Offset, AggrEntry.To.Offset, false};
      doTrace(First, Second, AggrEntry.Count);
      ++NumTraces;
      break;
    }
    }
  }
  outs() << "PERF2BOLT: Read " << NumAggrEntries << " aggregated LBR entries\n";
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
    if (outs().has_colors())
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

  dump();

  return std::error_code();
}

ErrorOr<std::pair<StringRef, DataAggregator::MMapInfo>>
DataAggregator::parseMMapEvent() {
  while (checkAndConsumeFS()) {}

  MMapInfo ParsedInfo;

  auto LineEnd = ParsingBuf.find_first_of("\n");
  if (LineEnd == StringRef::npos) {
    reportError("expected rest of line");
    Diag << "Found: " << ParsingBuf << "\n";
    return make_error_code(llvm::errc::io_error);
  }
  StringRef Line = ParsingBuf.substr(0, LineEnd);

  auto Pos = Line.find("PERF_RECORD_MMAP2");
  if (Pos == StringRef::npos) {
    consumeRestOfLine();
    return std::make_pair(StringRef(), ParsedInfo);
  }
  Line = Line.drop_front(Pos);

  auto FileName = Line.rsplit(FieldSeparator).second;
  if (FileName.startswith("//") || FileName.startswith("[")) {
    consumeRestOfLine();
    return std::make_pair(StringRef(), ParsedInfo);
  }
  FileName = sys::path::filename(FileName);

  StringRef PIDStr = Line.split(FieldSeparator).second.split('/').first;
  if (PIDStr.getAsInteger(10, ParsedInfo.PID)) {
    reportError("expected PID");
    Diag << "Found: " << PIDStr << "in '" << Line << "'\n";
    return make_error_code(llvm::errc::io_error);
  }

  StringRef BaseAddressStr = Line.split('[').second.split('(').first;
  if (BaseAddressStr.getAsInteger(0, ParsedInfo.BaseAddress)) {
    reportError("expected base address");
    Diag << "Found: " << BaseAddressStr << "in '" << Line << "'\n";
    return make_error_code(llvm::errc::io_error);
  }

  StringRef SizeStr = Line.split('(').second.split(')').first;
  if (SizeStr.getAsInteger(0, ParsedInfo.Size)) {
    reportError("expected mmaped size");
    Diag << "Found: " << SizeStr << "in '" << Line << "'\n";
    return make_error_code(llvm::errc::io_error);
  }

  consumeRestOfLine();

  return std::make_pair(FileName, ParsedInfo);
}

std::error_code DataAggregator::parseMMapEvents() {
  outs() << "PERF2BOLT: Parsing perf-script mmap events output\n";
  NamedRegionTimer T("parseMMapEvents", "Parsing mmap events", TimerGroupName,
                     TimerGroupDesc, opts::TimeAggregator);

  std::multimap<StringRef, MMapInfo> GlobalMMapInfo;
  while (hasData()) {
    auto FileMMapInfoRes = parseMMapEvent();
    if (std::error_code EC = FileMMapInfoRes.getError())
      return EC;

    auto FileMMapInfo = FileMMapInfoRes.get();
    if (FileMMapInfo.second.PID == -1)
      continue;

    // Consider only the first mapping of the file for any given PID
    bool PIDExists = false;
    auto Range = GlobalMMapInfo.equal_range(FileMMapInfo.first);
    for (auto MI = Range.first; MI != Range.second; ++MI) {
      if (MI->second.PID == FileMMapInfo.second.PID) {
        PIDExists = true;
        break;
      }
    }
    if (PIDExists)
      continue;

    GlobalMMapInfo.insert(FileMMapInfo);
  }

  DEBUG(
    dbgs() << "FileName -> mmap info:\n";
    for (const auto &Pair : GlobalMMapInfo) {
      dbgs() << "  " << Pair.first << " : " << Pair.second.PID << " [0x"
             << Twine::utohexstr(Pair.second.BaseAddress) << ", "
             << Twine::utohexstr(Pair.second.Size) << "]\n";
    }
  );

  auto NameToUse = BinaryName;
  if (GlobalMMapInfo.count(NameToUse) == 0 && !BuildIDBinaryName.empty()) {
    errs() << "PERF2BOLT-WARNING: using \"" << BuildIDBinaryName
           << "\" for profile matching\n";
    NameToUse = BuildIDBinaryName;
  }

  auto Range = GlobalMMapInfo.equal_range(NameToUse);
  for (auto I = Range.first; I != Range.second; ++I) {
    BinaryMMapInfo.insert(std::make_pair(I->second.PID, I->second));
  }

  if (BinaryMMapInfo.empty()) {
    if (errs().has_colors())
      errs().changeColor(raw_ostream::RED);
    errs() << "PERF2BOLT-ERROR: could not find a profile matching binary \""
           << BinaryName << "\".";
    if (!GlobalMMapInfo.empty()) {
      errs() << " Profile for the following binary name(s) is available:\n";
      for (auto I = GlobalMMapInfo.begin(), IE = GlobalMMapInfo.end(); I != IE;
           I = GlobalMMapInfo.upper_bound(I->first)) {
        errs() << "  " << I->first << '\n';
      }
      errs() << "Please rename the input binary.\n";
    } else {
      errs() << " Failed to extract any binary name from a profile.\n";
    }
    if (errs().has_colors())
      errs().resetColor();

    exit(1);
  }

  outs() << "PERF2BOLT: Input binary is associated with "
         << BinaryMMapInfo.size() << " PID(s)\n";

  return std::error_code();
}

Optional<std::pair<StringRef, StringRef>>
DataAggregator::parseNameBuildIDPair() {
  while (checkAndConsumeFS()) {}

  auto BuildIDStr = parseString(FieldSeparator, true);
  if (std::error_code EC = BuildIDStr.getError())
    return NoneType();

  auto NameStr = parseString(FieldSeparator, true);
  if (std::error_code EC = NameStr.getError())
    return NoneType();

  consumeRestOfLine();
  return std::make_pair(NameStr.get(), BuildIDStr.get());
}

Optional<StringRef>
DataAggregator::getFileNameForBuildID(StringRef FileBuildID) {
  while (hasData()) {
    auto IDPair = parseNameBuildIDPair();
    if (!IDPair)
      return NoneType();

    if (IDPair->second.startswith(FileBuildID))
      return sys::path::filename(IDPair->first);
  }
  return NoneType();
}

std::error_code DataAggregator::writeAggregatedFile() const {
  std::error_code EC;
  raw_fd_ostream OutFile(OutputFDataName, EC, sys::fs::OpenFlags::F_None);
  if (EC)
    return EC;

  bool WriteMemLocs = false;

  auto writeLocation = [&OutFile,&WriteMemLocs](const Location &Loc) {
    if (WriteMemLocs)
      OutFile << (Loc.IsSymbol ? "4 " : "3 ");
    else
      OutFile << (Loc.IsSymbol ? "1 " : "0 ");
    OutFile << (Loc.Name.empty() ? "[unknown]" : Loc.Name)  << " "
            << Twine::utohexstr(Loc.Offset)
            << FieldSeparator;
  };

  uint64_t BranchValues{0};
  uint64_t MemValues{0};

  if (opts::BasicAggregation) {
    OutFile << "no_lbr";
    for (const auto &Entry : EventNames) {
      OutFile << " " << Entry.getKey();
    }
    OutFile << "\n";

    for (const auto &Func : FuncsToSamples) {
      for (const auto &SI : Func.getValue().Data) {
        writeLocation(SI.Loc);
        OutFile << SI.Hits << "\n";
        ++BranchValues;
      }
    }
  } else {
    for (const auto &Func : FuncsToBranches) {
      for (const auto &BI : Func.getValue().Data) {
        writeLocation(BI.From);
        writeLocation(BI.To);
        OutFile << BI.Mispreds << " " << BI.Branches << "\n";
        ++BranchValues;
      }
      for (const auto &BI : Func.getValue().EntryData) {
        // Do not output if source is a known symbol, since this was already
        // accounted for in the source function
        if (BI.From.IsSymbol)
          continue;
        writeLocation(BI.From);
        writeLocation(BI.To);
        OutFile << BI.Mispreds << " " << BI.Branches << "\n";
        ++BranchValues;
      }
    }

    WriteMemLocs = true;
    for (const auto &Func : FuncsToMemEvents) {
      for (const auto &MemEvent : Func.getValue().Data) {
        writeLocation(MemEvent.Offset);
        writeLocation(MemEvent.Addr);
        OutFile << MemEvent.Count << "\n";
        ++MemValues;
      }
    }
  }

  outs() << "PERF2BOLT: Wrote " << BranchValues << " objects and "
         << MemValues << " memory objects to " << OutputFDataName << "\n";

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

void DataAggregator::dump(const PerfBranchSample &Sample) const {
  Diag << "Sample LBR entries: " << Sample.LBR.size() << "\n";
  for (const auto &LBR : Sample.LBR) {
    dump(LBR);
  }
}

void DataAggregator::dump(const PerfMemSample &Sample) const {
  Diag << "Sample mem entries: " << Sample.PC << ": " << Sample.Addr << "\n";
}
