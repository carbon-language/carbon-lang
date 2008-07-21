//===- lto-bugpoing.cpp - The lto-bugpoint driver -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// lto-bugpoint tool identifies minmal set of bitcode files that is causing
// failure when Link Time Optimization is enabled. The failure is identified
// using developer provided validation script.
//
//===----------------------------------------------------------------------===//

#include "LTOBugPoint.h"
#include <iostream>
#include <fstream>

int main(int argc, char **argv) {
  try {

    if (argc !=  4) {
      std::cerr << "Invalid number of lto-bugpoint arguments!\n";
      return 1;
    }
    
    std::ios::openmode input_mode = std::ios::in;

    // First argument is linker command line options file. This text file
    // is a list of linker command line options, one option per line.
    // First line always list the absolute path to invoke the linker.
    std::istream *LinkerArgsFile = new std::ifstream(argv[1], input_mode);
    if (!LinkerArgsFile->good()) {
      std::cerr << argv[0] << ": error opening " << argv[1] << "!\n";
      return 1;
    }

    // Second argment is a text file that includes the linker input
    // file paths, one input file path per line. 
    std::istream *LinkerInputsFile = new std::ifstream(argv[2], input_mode);
    if (!LinkerInputsFile->good()) {
      std::cerr << argv[0] << ": error opening " << argv[2] << "!\n";
      delete LinkerArgsFile;
      return 1;
    }

    // Third argument is absolute path to the validation script. This
    // script is used to validate LTO error under investigation.
    std::string ValidationScript = argv[3];
    LTOBugPoint bugFinder(*LinkerArgsFile, *LinkerInputsFile);

    llvm::SmallVector<std::string, 4> TroubleMakers;
    if (!bugFinder.findTroubleMakers(TroubleMakers, ValidationScript)) {
      std::cerr << "lto-bugpoint:" << bugFinder.getErrMsg() << "\n";
      return 1;
    }

    return 0;
  } catch (const std::string& msg) {
    std::cerr << argv[0] << ": " << msg << "\n";
  } catch (...) {
    std::cerr << argv[0] << ": Unexpected unknown exception occurred.\n";
  }
  return 1;
}
