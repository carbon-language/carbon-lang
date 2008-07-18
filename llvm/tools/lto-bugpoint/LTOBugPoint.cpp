//===- LTOBugPoint.cpp - Top-Level LTO BugPoint class ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class contains all of the shared state and information that is used by
// the LTO BugPoint tool to track down bit code files that cause errors.
//
//===----------------------------------------------------------------------===//

#include "LTOBugPoint.h"
#include "llvm/Support/SystemUtils.h"
#include "llvm/System/Path.h"
#include <iostream>

/// LTOBugPoint -- Constructor. Popuate list of linker options and
/// list of linker input files.
LTOBugPoint::LTOBugPoint(std::istream &args, std::istream &ins) {

  // Read linker options. Order is important here.
  std::string option;
  while (getline(args, option))
    LinkerOptions.push_back(option);
  
  // Read linker input files. Order is important here.
  std::string inFile;
  while(getline(ins, inFile))
    LinkerInputFiles.push_back(inFile);
}

/// findTroubleMakers - Find minimum set of input files that causes error
/// identified by the script.
bool
LTOBugPoint::findTroubleMakers(llvm::SmallVector<std::string, 4> &TroubleMakers,
			       std::string &Script) {

  // First, build native object files set.
  bool bitcodeFileSeen = false;
  for(llvm::SmallVector<std::string, 16>::iterator I = LinkerInputFiles.begin(),
	E = LinkerInputFiles.end(); I != E; ++I) {
    std::string &path = *I;
    if (llvm::sys::Path(path.c_str()).isBitcodeFile()) 
      bitcodeFileSeen = true;
  }

  if (!bitcodeFileSeen) {
    std::cerr << "lto-bugpoint: Error: Unable to help!"; 
    std::cerr << " Need at least one input file that contains llvm bitcode\n";
    return false;
  }

  return true;
}
