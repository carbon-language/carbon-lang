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
#include <utility>
class Module;
class Function;

class ProfileInfo {
  Module &M;
  std::vector<std::string> CommandLines;
  std::vector<unsigned>    FunctionCounts;
  std::vector<unsigned>    BlockCounts;
public:
  // ProfileInfo ctor - Read the specified profiling data file, exiting the
  // program if the file is invalid or broken.
  ProfileInfo(const char *ToolName, const std::string &Filename, Module &M);

  // getFunctionCounts - This method is used by consumers of function counting
  // information.  If we do not directly have function count information, we
  // compute it from other, more refined, types of profile information.
  //
  void getFunctionCounts(std::vector<std::pair<Function*, unsigned> > &Counts);

};

#endif
