//===- bolt/tools/driver/llvm-bolt.cpp - Feedback-directed optimizer ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a binary optimizer that will take 'perf' output and change
// basic block layout for better performance (a.k.a. branch straightening),
// plus some other optimizations that are better performed on a binary.
//
//===----------------------------------------------------------------------===//

#include "bolt/Profile/DataAggregator.h"
#include "bolt/Rewrite/MachORewriteInstance.h"
#include "bolt/Rewrite/RewriteInstance.h"
#include "bolt/Utils/CommandLineOpts.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/Binary.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/TargetSelect.h"

#define DEBUG_TYPE "bolt"

using namespace llvm;
using namespace object;
using namespace bolt;

namespace opts {

static cl::OptionCategory *BoltCategories[] = {&BoltCategory,
                                               &BoltOptCategory,
                                               &BoltRelocCategory,
                                               &BoltInstrCategory,
                                               &BoltOutputCategory};

static cl::OptionCategory *BoltDiffCategories[] = {&BoltDiffCategory};

static cl::OptionCategory *Perf2BoltCategories[] = {&AggregatorCategory,
                                                    &BoltOutputCategory};

static cl::opt<std::string>
InputDataFilename("data",
  cl::desc("<data file>"),
  cl::Optional,
  cl::cat(BoltCategory));

static cl::alias
BoltProfile("b",
  cl::desc("alias for -data"),
  cl::aliasopt(InputDataFilename),
  cl::cat(BoltCategory));

static cl::opt<std::string>
InputDataFilename2("data2",
  cl::desc("<data file>"),
  cl::Optional,
  cl::cat(BoltCategory));

static cl::opt<std::string>
InputFilename2(
  cl::Positional,
  cl::desc("<executable>"),
  cl::Optional,
  cl::cat(BoltDiffCategory));

} // namespace opts

static StringRef ToolName;

static void report_error(StringRef Message, std::error_code EC) {
  assert(EC);
  errs() << ToolName << ": '" << Message << "': " << EC.message() << ".\n";
  exit(1);
}

static void report_error(StringRef Message, Error E) {
  assert(E);
  errs() << ToolName << ": '" << Message << "': " << toString(std::move(E))
         << ".\n";
  exit(1);
}

static void printBoltRevision(llvm::raw_ostream &OS) {
  OS << "BOLT revision " << BoltRevision << "\n";
}

void perf2boltMode(int argc, char **argv) {
  cl::HideUnrelatedOptions(makeArrayRef(opts::Perf2BoltCategories));
  cl::AddExtraVersionPrinter(printBoltRevision);
  cl::ParseCommandLineOptions(
      argc, argv,
      "perf2bolt - BOLT data aggregator\n"
      "\nEXAMPLE: perf2bolt -p=perf.data executable -o data.fdata\n");
  if (opts::PerfData.empty()) {
    errs() << ToolName << ": expected -perfdata=<filename> option.\n";
    exit(1);
  }
  if (!opts::InputDataFilename.empty()) {
    errs() << ToolName << ": unknown -data option.\n";
    exit(1);
  }
  if (!sys::fs::exists(opts::PerfData))
    report_error(opts::PerfData, errc::no_such_file_or_directory);
  if (!DataAggregator::checkPerfDataMagic(opts::PerfData)) {
    errs() << ToolName << ": '" << opts::PerfData
           << "': expected valid perf.data file.\n";
    exit(1);
  }
  if (opts::OutputFilename.empty()) {
    errs() << ToolName << ": expected -o=<output file> option.\n";
    exit(1);
  }
  opts::AggregateOnly = true;
}

void heatmapMode(int argc, char **argv) {
  // Insert a fake subcommand if invoked via a command alias.
  std::unique_ptr<char *[]> FakeArgv;
  if (argc == 1 || strcmp(argv[1], "heatmap")) {
    ++argc;
    FakeArgv.reset(new char *[argc + 1]);
    FakeArgv[0] = argv[0];
    FakeArgv[1] = const_cast<char *>("heatmap");
    for (int I = 2; I < argc; ++I)
      FakeArgv[I] = argv[I - 1];
    FakeArgv[argc] = nullptr;
    argv = FakeArgv.get();
  }

  cl::ParseCommandLineOptions(argc, argv, "");

  if (!sys::fs::exists(opts::InputFilename))
    report_error(opts::InputFilename, errc::no_such_file_or_directory);

  if (opts::PerfData.empty()) {
    errs() << ToolName << ": expected -perfdata=<filename> option.\n";
    exit(1);
  }

  opts::HeatmapMode = true;
  opts::AggregateOnly = true;
}

void boltDiffMode(int argc, char **argv) {
  cl::HideUnrelatedOptions(makeArrayRef(opts::BoltDiffCategories));
  cl::AddExtraVersionPrinter(printBoltRevision);
  cl::ParseCommandLineOptions(
      argc, argv,
      "llvm-boltdiff - BOLT binary diff tool\n"
      "\nEXAMPLE: llvm-boltdiff -data=a.fdata -data2=b.fdata exec1 exec2\n");
  if (opts::InputDataFilename2.empty()) {
    errs() << ToolName << ": expected -data2=<filename> option.\n";
    exit(1);
  }
  if (opts::InputDataFilename.empty()) {
    errs() << ToolName << ": expected -data=<filename> option.\n";
    exit(1);
  }
  if (opts::InputFilename2.empty()) {
    errs() << ToolName << ": expected second binary name.\n";
    exit(1);
  }
  if (opts::InputFilename.empty()) {
    errs() << ToolName << ": expected binary.\n";
    exit(1);
  }
  opts::DiffOnly = true;
}

void boltMode(int argc, char **argv) {
  cl::HideUnrelatedOptions(makeArrayRef(opts::BoltCategories));
  // Register the target printer for --version.
  cl::AddExtraVersionPrinter(printBoltRevision);
  cl::AddExtraVersionPrinter(TargetRegistry::printRegisteredTargetsForVersion);

  cl::ParseCommandLineOptions(argc, argv,
                              "BOLT - Binary Optimization and Layout Tool\n");

  if (opts::OutputFilename.empty()) {
    errs() << ToolName << ": expected -o=<output file> option.\n";
    exit(1);
  }
}

std::string GetExecutablePath(const char *Argv0) {
  SmallString<128> ExecutablePath(Argv0);
  // Do a PATH lookup if Argv0 isn't a valid path.
  if (!llvm::sys::fs::exists(ExecutablePath))
    if (llvm::ErrorOr<std::string> P =
            llvm::sys::findProgramByName(ExecutablePath))
      ExecutablePath = *P;
  return std::string(ExecutablePath.str());
}

int main(int argc, char **argv) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal(argv[0]);
  PrettyStackTraceProgram X(argc, argv);

  llvm_shutdown_obj Y; // Call llvm_shutdown() on exit.

  std::string ToolPath = GetExecutablePath(argv[0]);

  // Initialize targets and assembly printers/parsers.
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllDisassemblers();

  llvm::InitializeAllTargets();
  llvm::InitializeAllAsmPrinters();

  ToolName = argv[0];

  // Pre-process subcommands.
  if (argc > 1 && *argv[1] != '-') {
    if (!strcmp(argv[1], "heatmap"))
      opts::HeatmapMode = true;
  }

  if (llvm::sys::path::filename(ToolName) == "perf2bolt")
    perf2boltMode(argc, argv);
  else if (llvm::sys::path::filename(ToolName) == "llvm-boltdiff")
    boltDiffMode(argc, argv);
  else if (llvm::sys::path::filename(ToolName) == "llvm-bolt-heatmap" ||
           opts::HeatmapMode)
    heatmapMode(argc, argv);
  else
    boltMode(argc, argv);

  if (!sys::fs::exists(opts::InputFilename))
    report_error(opts::InputFilename, errc::no_such_file_or_directory);

  // Attempt to open the binary.
  if (!opts::DiffOnly) {
    Expected<OwningBinary<Binary>> BinaryOrErr =
        createBinary(opts::InputFilename);
    if (Error E = BinaryOrErr.takeError())
      report_error(opts::InputFilename, std::move(E));
    Binary &Binary = *BinaryOrErr.get().getBinary();

    if (auto *e = dyn_cast<ELFObjectFileBase>(&Binary)) {
      RewriteInstance RI(e, argc, argv, ToolPath);
      if (!opts::PerfData.empty()) {
        if (!opts::AggregateOnly) {
          errs() << ToolName
                 << ": WARNING: reading perf data directly is unsupported, "
                    "please use "
                    "-aggregate-only or perf2bolt.\n!!! Proceed on your own "
                    "risk. !!!\n";
        }
        if (Error E = RI.setProfile(opts::PerfData))
          report_error(opts::PerfData, std::move(E));
      }
      if (!opts::InputDataFilename.empty()) {
        if (Error E = RI.setProfile(opts::InputDataFilename))
          report_error(opts::InputDataFilename, std::move(E));
      }
      if (opts::AggregateOnly && opts::PerfData.empty()) {
        errs() << ToolName << ": missing required -perfdata option.\n";
        exit(1);
      }

      RI.run();
    } else if (auto *O = dyn_cast<MachOObjectFile>(&Binary)) {
      MachORewriteInstance MachORI(O, ToolPath);

      if (!opts::InputDataFilename.empty())
        if (Error E = MachORI.setProfile(opts::InputDataFilename))
          report_error(opts::InputDataFilename, std::move(E));

      MachORI.run();
    } else {
      report_error(opts::InputFilename, object_error::invalid_file_type);
    }

    return EXIT_SUCCESS;
  }

  // Bolt-diff
  Expected<OwningBinary<Binary>> BinaryOrErr1 =
      createBinary(opts::InputFilename);
  Expected<OwningBinary<Binary>> BinaryOrErr2 =
      createBinary(opts::InputFilename2);
  if (Error E = BinaryOrErr1.takeError())
    report_error(opts::InputFilename, std::move(E));
  if (Error E = BinaryOrErr2.takeError())
    report_error(opts::InputFilename2, std::move(E));
  Binary &Binary1 = *BinaryOrErr1.get().getBinary();
  Binary &Binary2 = *BinaryOrErr2.get().getBinary();
  if (auto *ELFObj1 = dyn_cast<ELFObjectFileBase>(&Binary1)) {
    if (auto *ELFObj2 = dyn_cast<ELFObjectFileBase>(&Binary2)) {
      RewriteInstance RI1(ELFObj1, argc, argv, ToolPath);
      if (Error E = RI1.setProfile(opts::InputDataFilename))
        report_error(opts::InputDataFilename, std::move(E));
      RewriteInstance RI2(ELFObj2, argc, argv, ToolPath);
      if (Error E = RI2.setProfile(opts::InputDataFilename2))
        report_error(opts::InputDataFilename2, std::move(E));
      outs() << "BOLT-DIFF: *** Analyzing binary 1: " << opts::InputFilename
             << "\n";
      outs() << "BOLT-DIFF: *** Binary 1 fdata:     " << opts::InputDataFilename
             << "\n";
      RI1.run();
      outs() << "BOLT-DIFF: *** Analyzing binary 2: " << opts::InputFilename2
             << "\n";
      outs() << "BOLT-DIFF: *** Binary 2 fdata:     "
             << opts::InputDataFilename2 << "\n";
      RI2.run();
      RI1.compare(RI2);
    } else {
      report_error(opts::InputFilename2, object_error::invalid_file_type);
    }
  } else {
    report_error(opts::InputFilename, object_error::invalid_file_type);
  }

  return EXIT_SUCCESS;
}
