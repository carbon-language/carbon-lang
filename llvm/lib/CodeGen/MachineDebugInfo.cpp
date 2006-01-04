//===-- llvm/CodeGen/MachineDebugInfo.cpp -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by James M. Laskey and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Collect debug information for a module.  This information should be in a
// neutral form that can be used by different debugging schemes.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineDebugInfo.h"

using namespace llvm;

// Handle the Pass registration stuff necessary to use TargetData's.
namespace {
  RegisterPass<MachineDebugInfo> X("machinedebuginfo", "Debug Information");
}
  
/// doInitialization - Initialize the debug state for a new module.
///
bool MachineDebugInfo::doInitialization() {
  return false;
}

/// doFinalization - Tear down the debug state after completion of a module.
///
bool MachineDebugInfo::doFinalization() {
  return false;
}

/// getUniqueSourceID - Register a source file with debug info.  Returns an id.
///
unsigned MachineDebugInfo::getUniqueSourceID(const std::string &fname,
                                             const std::string &dirname) {
  // Compose a key
  const std::string path = dirname + "/" + fname;
  // Check if the source file is already recorded
  std::map<std::string, unsigned>::iterator
      SMI = SourceMap.lower_bound(path);
  // If already there return existing id
  if (SMI != SourceMap.end() && SMI->first == path) return SMI->second;
  // Bump up the count
  ++SourceCount;
  // Record the count
  SourceMap.insert(SMI, std::make_pair(path, SourceCount));
  // Return id
  return SourceCount;
}

/// getSourceFiles - Return a vector of files.  Vector index + 1 equals id.
///
std::vector<std::string> MachineDebugInfo::getSourceFiles() const {
  std::vector<std::string> Sources(SourceCount);
  
  for (std::map<std::string, unsigned>::const_iterator SMI = SourceMap.begin(),
                                                       E = SourceMap.end();
                                                       SMI != E; SMI++) {
    unsigned Index = SMI->second - 1;
    const std::string &Path = SMI->first;
    Sources[Index] = Path;
  }
  return Sources;
}

