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

namespace llvm {

class Module;
class Function;
class BasicBlock;

class ProfileInfo {
  Module &M;
  std::vector<std::string> CommandLines;
  std::vector<unsigned>    FunctionCounts;
  std::vector<unsigned>    BlockCounts;
public:
  // ProfileInfo ctor - Read the specified profiling data file, exiting the
  // program if the file is invalid or broken.
  ProfileInfo(const char *ToolName, const std::string &Filename, Module &M);

  unsigned getNumExecutions() const { return CommandLines.size(); }
  const std::string &getExecution(unsigned i) const { return CommandLines[i]; }

  // getFunctionCounts - This method is used by consumers of function counting
  // information.  If we do not directly have function count information, we
  // compute it from other, more refined, types of profile information.
  //
  void getFunctionCounts(std::vector<std::pair<Function*, unsigned> > &Counts);

  // hasAccurateBlockCounts - Return true if we can synthesize accurate block
  // frequency information from whatever we have.
  //
  bool hasAccurateBlockCounts() const {
    return !BlockCounts.empty();
  }

  // getBlockCounts - This method is used by consumers of block counting
  // information.  If we do not directly have block count information, we
  // compute it from other, more refined, types of profile information.
  //
  void getBlockCounts(std::vector<std::pair<BasicBlock*, unsigned> > &Counts);
};

} // End llvm namespace

#endif
