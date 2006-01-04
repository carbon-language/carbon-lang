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
  RegisterPass<MachineDebugInfo> X("machinedebuginfo", "Debug Information",
                                  PassInfo::Analysis | PassInfo::Optimization);
}

namespace llvm {
  
  /// DebugInfo - Keep track of debug information for the function.
  ///
  // FIXME - making it global until we can find a proper place to hang it from.
  MachineDebugInfo *DebugInfo;

  // FIXME - temporary hack until we can find a place to hand debug info from.
  ModulePass *createDebugInfoPass() {
    if (!DebugInfo) DebugInfo = new MachineDebugInfo();
    return (ModulePass *)DebugInfo;
  }
  
  /// getDebugInfo - Returns the DebugInfo.
  MachineDebugInfo &getMachineDebugInfo() {
    assert(DebugInfo && "DebugInfo pass not created");
    return *DebugInfo;
  }

  /// doInitialization - Initialize the debug state for a new module.
  ///
  bool MachineDebugInfo::doInitialization() {
    return true;
  }

  /// doFinalization - Tear down the debug state after completion of a module.
  ///
  bool MachineDebugInfo::doFinalization() {
    return true;
  }

  /// RecordSource - Register a source file with debug info.  Returns an id.
  ///
  unsigned MachineDebugInfo::RecordSource(std::string fname,
                                          std::string dirname) {
    // Compose a key
    std::string path = dirname + "/" + fname;
    // Check if the source file is already recorded
    StrIntMapIter SMI = SourceMap.find(path);
    // If already there return existing id
    if (SMI != SourceMap.end()) return SMI->second;
    // Bump up the count
    ++SourceCount;
    // Record the count
    SourceMap[path] = SourceCount;
    // Return id
    return SourceCount;
  }

  /// getSourceFiles - Return a vector of files.  Vector index + 1 equals id.
  ///
  std::vector<std::string> MachineDebugInfo::getSourceFiles() {
    std::vector<std::string> Sources(SourceCount);
    
    for (StrIntMapIter SMI = SourceMap.begin(), E = SourceMap.end(); SMI != E;
                       SMI++) {
      unsigned Index = SMI->second - 1;
      std::string Path = SMI->first;
      Sources[Index] = Path;
    }
    return Sources;
  }


};

