//===- ProfileInfo.h - Represents profile information -----------*- C++ -*-===//
// 
//                      The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// The ProfileInfo class is used to represent profiling information read in from
// the dump file.
//
//===----------------------------------------------------------------------===//

#ifndef PROFILEINFO_H
#define PROFILEINFO_H

#include <vector>
#include <string>

class ProfileInfo {
  std::vector<std::string> CommandLines;
  std::vector<unsigned>    FunctionCounts;
  std::vector<unsigned>    BlockCounts;
public:
  // ProfileInfo ctor - Read the specified profiling data file, exiting the
  // program if the file is invalid or broken.
  ProfileInfo(const char *ToolName, const std::string &Filename);
};

#endif
