//===-- llvm-ranlib.cpp - LLVM archive index generator --------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// Adds or updates an index (symbol table) for an LLVM archive file.
//
//===----------------------------------------------------------------------===//

#include "llvm/Module.h"
#include "llvm/Bytecode/Archive.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/System/Signals.h"
#include <iostream>
#include <algorithm>
#include <iomanip>

using namespace llvm;

// llvm-ar operation code and modifier flags
cl::opt<std::string> 
ArchiveName(cl::Positional, cl::Optional, cl::desc("<archive-file>..."));

cl::opt<bool>
Verbose("verbose",cl::Optional,cl::init(false),
        cl::desc("Print the symbol table"));

sys::Path TmpArchive;

void cleanup() {
  if (TmpArchive.exists())
    TmpArchive.destroyFile();
}

int main(int argc, char **argv) {

  // Have the command line options parsed and handle things
  // like --help and --version.
  cl::ParseCommandLineOptions(argc, argv,
    " LLVM Archive Index Generator (llvm-ranlib)\n\n"
    "  This program adds or updates an index of bytecode symbols\n"
    "  to an LLVM archive file."
  );

  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal();

  int exitCode = 0;

  // Make sure we don't exit with "unhandled exception".
  try {

    // Check the path name of the archive
    sys::Path ArchivePath;
    if (!ArchivePath.setFile(ArchiveName))
      throw std::string("Archive name invalid: ") + ArchiveName;

    // Make sure it exists, we don't create empty archives
    if (!ArchivePath.exists())
      throw "Archive file does not exist";

    // Archive* TheArchive = Archive::OpenAndLoad(ArchivePath);
    Archive* TheArchive = Archive::OpenAndLoad(ArchivePath);

    assert(TheArchive && "Unable to instantiate the archive");

    TheArchive->writeToDisk(true,false,false,Verbose);

    delete TheArchive;

  } catch (const char*msg) {
    std::cerr << argv[0] << ": " << msg << "\n\n";
    exitCode = 1;
  } catch (const std::string& msg) {
    std::cerr << argv[0] << ": " << msg << "\n";
    exitCode = 2;
  } catch (...) {
    std::cerr << argv[0] << ": An nexpected unknown exception occurred.\n";
    exitCode = 3;
  }
  cleanup();
  return exitCode;
}
