//===-- llvm-flo.cpp - Feedback-directed layout optimizer -----------------===//
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
#define DEBUG_TYPE "flo"

using namespace llvm;
using namespace object;
using namespace flo;

namespace opts {

static cl::opt<std::string>
InputFilename(cl::Positional, cl::desc("<executable>"), cl::Required);

static cl::opt<std::string>
InputDataFilename("data", cl::desc("<data file>"), cl::Optional);

static cl::opt<bool>
DumpData("dump-data", cl::desc("dump parsed flo data and exit (debugging)"),
         cl::Hidden);

} // namespace opts

static StringRef ToolName;

static void report_error(StringRef Message, std::error_code EC) {
  assert(EC);
  errs() << ToolName << ": '" << Message << "': " << EC.message() << ".\n";
  exit(1);
}

int main(int argc, char **argv) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);

  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.

  // Initialize targets and assembly printers/parsers.
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllDisassemblers();

  llvm::InitializeAllTargets();
  llvm::InitializeAllAsmPrinters();

  // Register the target printer for --version.
  cl::AddExtraVersionPrinter(TargetRegistry::printRegisteredTargetsForVersion);

  cl::ParseCommandLineOptions(argc, argv,
                              "llvm feedback-directed layout optimizer\n");

  ToolName = argv[0];

  if (!sys::fs::exists(opts::InputFilename))
    report_error(opts::InputFilename, errc::no_such_file_or_directory);

  std::unique_ptr<flo::DataReader> DR(new DataReader(errs()));
  if (!opts::InputDataFilename.empty()) {
    if (!sys::fs::exists(opts::InputDataFilename))
      report_error(opts::InputDataFilename, errc::no_such_file_or_directory);

    // Attempt to read input flo data
    auto ReaderOrErr =
      flo::DataReader::readPerfData(opts::InputDataFilename, errs());
    if (std::error_code EC = ReaderOrErr.getError())
      report_error(opts::InputDataFilename, EC);
    DR.reset(ReaderOrErr.get().release());
    if (opts::DumpData) {
      DR->dump();
      return EXIT_SUCCESS;
    }
  }

  // Attempt to open the binary.
  ErrorOr<OwningBinary<Binary>> BinaryOrErr = createBinary(opts::InputFilename);
  if (std::error_code EC = BinaryOrErr.getError())
    report_error(opts::InputFilename, EC);
  Binary &Binary = *BinaryOrErr.get().getBinary();

  if (auto *e = dyn_cast<ELFObjectFileBase>(&Binary)) {
    RewriteInstance RI(e, *DR.get());
    RI.run();
  } else {
    report_error(opts::InputFilename, object_error::invalid_file_type);
  }

  return EXIT_SUCCESS;
}
