//===-- llvm-bolt.cpp - Feedback-directed layout optimizer ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a binary optimizer that will take 'perf' output and change
// basic block layout for better performance (a.k.a. branch straightening),
// plus some other optimizations that are better performed on a binary.
//
//===----------------------------------------------------------------------===//

#include "DataAggregator.h"
#include "DataReader.h"
#include "RewriteInstance.h"
#include "llvm/Object/Binary.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/TargetRegistry.h"

#undef  DEBUG_TYPE
#define DEBUG_TYPE "bolt"

using namespace llvm;
using namespace object;
using namespace bolt;

namespace opts {

cl::OptionCategory BoltCategory("BOLT generic options");
cl::OptionCategory BoltDiffCategory("BOLTDIFF generic options");
cl::OptionCategory BoltOptCategory("BOLT optimization options");
cl::OptionCategory BoltRelocCategory("BOLT options in relocation mode");
cl::OptionCategory BoltOutputCategory("Output options");
cl::OptionCategory AggregatorCategory("Data aggregation options");

static cl::OptionCategory *BoltCategories[] = {&BoltCategory,
                                               &BoltOptCategory,
                                               &BoltRelocCategory,
                                               &BoltOutputCategory};

static cl::OptionCategory *BoltDiffCategories[] = {&BoltDiffCategory};

static cl::OptionCategory *Perf2BoltCategories[] = {&AggregatorCategory,
                                                    &BoltOutputCategory};

extern cl::opt<std::string> OutputFilename;
extern cl::opt<bool> AggregateOnly;
extern cl::opt<bool> DiffOnly;

static cl::opt<bool>
DumpData("dump-data",
  cl::desc("dump parsed bolt data and exit (debugging)"),
  cl::Hidden,
  cl::cat(BoltCategory));

static cl::opt<std::string>
InputDataFilename("data",
  cl::desc("<data file>"),
  cl::Optional,
  cl::cat(BoltCategory));

static cl::opt<std::string>
InputDataFilename2("data2",
  cl::desc("<data file>"),
  cl::Optional,
  cl::cat(BoltCategory));

static cl::opt<std::string>
InputFilename(
  cl::Positional,
  cl::desc("<executable>"),
  cl::Required,
  cl::cat(BoltDiffCategory));

static cl::opt<std::string>
InputFilename2(
  cl::Positional,
  cl::desc("<executable>"),
  cl::Optional,
  cl::cat(BoltDiffCategory));

static cl::opt<std::string>
PerfData("perfdata",
  cl::desc("<data file>"),
  cl::Optional,
  cl::cat(AggregatorCategory));

static cl::alias
PerfDataA("p",
  cl::desc("Alias for -perfdata"),
  cl::aliasopt(PerfData),
  cl::cat(AggregatorCategory));

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

namespace llvm {
namespace bolt {
const char *BoltRevision =
#include "BoltRevision.inc"
;
}
}

static void printBoltRevision(llvm::raw_ostream &OS) {
  OS << "BOLT revision " << BoltRevision << "\n";
}

void perf2boltMode(int argc, char **argv) {
  cl::HideUnrelatedOptions(makeArrayRef(opts::Perf2BoltCategories));
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

void boltDiffMode(int argc, char **argv) {
  cl::HideUnrelatedOptions(makeArrayRef(opts::BoltDiffCategories));
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

int main(int argc, char **argv) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal(argv[0]);
  PrettyStackTraceProgram X(argc, argv);

  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.

  // Initialize targets and assembly printers/parsers.
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllDisassemblers();

  llvm::InitializeAllTargets();
  llvm::InitializeAllAsmPrinters();

  ToolName = argv[0];

  if (llvm::sys::path::filename(ToolName) == "perf2bolt")
    perf2boltMode(argc, argv);
  else if (llvm::sys::path::filename(ToolName) == "llvm-boltdiff")
    boltDiffMode(argc, argv);
  else
    boltMode(argc, argv);


  if (!sys::fs::exists(opts::InputFilename))
    report_error(opts::InputFilename, errc::no_such_file_or_directory);

  std::unique_ptr<bolt::DataReader> DR(new DataReader(errs()));
  std::unique_ptr<bolt::DataAggregator> DA(
      new DataAggregator(errs(), opts::InputFilename));

  if (opts::AggregateOnly) {
    DA->setOutputFDataName(opts::OutputFilename);
    if (opts::PerfData.empty()) {
      errs() << ToolName << ": missing required -perfdata option.\n";
      exit(1);
    }
  }
  if (!opts::PerfData.empty()) {
    if (!opts::AggregateOnly) {
      errs() << ToolName
        << ": WARNING: reading perf data directly is unsupported, please use "
        "-aggregate-only or perf2bolt.\n!!! Proceed on your own risk. !!!\n";
    }
    DA->start(opts::PerfData);
  } else if (!opts::InputDataFilename.empty()) {
    if (!sys::fs::exists(opts::InputDataFilename))
      report_error(opts::InputDataFilename, errc::no_such_file_or_directory);

    auto ReaderOrErr =
        bolt::DataReader::readPerfData(opts::InputDataFilename, errs());
    if (std::error_code EC = ReaderOrErr.getError())
      report_error(opts::InputDataFilename, EC);
    DR.reset(ReaderOrErr.get().release());
    if (opts::DumpData) {
      DR->dump();
      return EXIT_SUCCESS;
    }
  }

  // Attempt to open the binary.
  if (!opts::DiffOnly) {
    Expected<OwningBinary<Binary>> BinaryOrErr =
        createBinary(opts::InputFilename);
    if (auto E = BinaryOrErr.takeError())
      report_error(opts::InputFilename, std::move(E));
    Binary &Binary = *BinaryOrErr.get().getBinary();

    if (auto *e = dyn_cast<ELFObjectFileBase>(&Binary)) {
      RewriteInstance RI(e, *DR.get(), *DA.get(), argc, argv);
      RI.run();
    } else {
      report_error(opts::InputFilename, object_error::invalid_file_type);
    }

    return EXIT_SUCCESS;
  }

  // Bolt-diff
  std::unique_ptr<bolt::DataReader> DR2(new DataReader(errs()));
  if (!sys::fs::exists(opts::InputDataFilename2))
    report_error(opts::InputDataFilename2, errc::no_such_file_or_directory);

  auto ReaderOrErr2 =
    bolt::DataReader::readPerfData(opts::InputDataFilename2, errs());
  if (std::error_code EC = ReaderOrErr2.getError())
    report_error(opts::InputDataFilename2, EC);
  DR2.reset(ReaderOrErr2.get().release());

  Expected<OwningBinary<Binary>> BinaryOrErr1 =
      createBinary(opts::InputFilename);
  Expected<OwningBinary<Binary>> BinaryOrErr2 =
      createBinary(opts::InputFilename2);
  if (auto E = BinaryOrErr1.takeError())
    report_error(opts::InputFilename, std::move(E));
  if (auto E = BinaryOrErr2.takeError())
    report_error(opts::InputFilename2, std::move(E));
  Binary &Binary1 = *BinaryOrErr1.get().getBinary();
  Binary &Binary2 = *BinaryOrErr2.get().getBinary();

  if (auto *ELFObj1 = dyn_cast<ELFObjectFileBase>(&Binary1)) {
    if (auto *ELFObj2 = dyn_cast<ELFObjectFileBase>(&Binary2)) {
      RewriteInstance RI1(ELFObj1, *DR.get(), *DA.get(), argc, argv);
      RewriteInstance RI2(ELFObj2, *DR2.get(), *DA.get(), argc, argv);
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
