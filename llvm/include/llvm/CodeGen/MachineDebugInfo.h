//===-- llvm/CodeGen/MachineDebugInfo.h -------------------------*- C++ -*-===//
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

#ifndef LLVM_CODEGEN_MACHINEDEBUGINFO_H
#define LLVM_CODEGEN_MACHINEDEBUGINFO_H

#include <string>
#include <map>
#include <vector>

namespace llvm {
//===----------------------------------------------------------------------===//
/// MachineDebugInfo - This class contains debug information specific to a
/// module.  Queries can be made by different debugging schemes and reformated
/// for specific use.
///
class MachineDebugInfo {
private:
  // convenience types
  typedef std::map<std::string, unsigned> StrIntMap;
  typedef StrIntMap::iterator StrIntMapIter;
  
  StrIntMap SourceMap;                  // Map of source file path to id
  unsigned SourceCount;                 // Number of source files (used to
                                        // generate id)

public:
  // Ctor.
  MachineDebugInfo() : SourceMap(), SourceCount(0) {}
  
  /// RecordSource - Register a source file with debug info.  Returns an id.
  ///
  unsigned RecordSource(std::string fname, std::string dirname) {
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
  std::vector<std::string> getSourceFiles() {
    std::vector<std::string> Sources(SourceCount);
    
    for (StrIntMapIter SMI = SourceMap.begin(), E = SourceMap.end(); SMI != E;
                       SMI++) {
      unsigned Index = SMI->second - 1;
      std::string Path = SMI->first;
      Sources[Index] = Path;
    }
    return Sources;
  }
  
}; // End class MachineDebugInfo
//===----------------------------------------------------------------------===//



} // End llvm namespace

#endif
