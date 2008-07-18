//===- LTOBugPoint.h - Top-Level LTO BugPoint class -------------*- C++ -*-===//
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

#include "llvm/ADT/SmallVector.h"
#include <string>
#include <fstream>

class LTOBugPoint {
 public:

  LTOBugPoint(std::istream &args, std::istream &ins);

 private:
  /// LinkerInputFiles - This is a list of linker input files. Once populated
  /// this list is not modified.
  llvm::SmallVector<std::string, 16> LinkerInputFiles;
  llvm::SmallVector<std::string, 16> LinkerOptions;

};
