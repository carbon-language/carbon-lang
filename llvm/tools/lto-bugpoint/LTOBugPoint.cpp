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
