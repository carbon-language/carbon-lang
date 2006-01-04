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

#include "llvm/Pass.h"
#include <string>
#include <map>
#include <vector>

namespace llvm {
//===----------------------------------------------------------------------===//
/// MachineDebugInfo - This class contains debug information specific to a
/// module.  Queries can be made by different debugging schemes and reformated
/// for specific use.
///
class MachineDebugInfo : public ImmutablePass {
private:
  std::map<std::string, unsigned> SourceMap; // Map of source file path to id
  unsigned SourceCount;                 // Number of source files (used to
                                        // generate id)
  unsigned UniqueID;                    // Number used to unique labels used
                                        // by debugger.

public:
  // Ctor.
  MachineDebugInfo()
  : SourceMap()
  , SourceCount(0)
  , UniqueID(1)
  {}
  ~MachineDebugInfo() { }
  
  /// hasInfo - Returns true if debug info is present.
  ///
  // FIXME - need scheme to suppress debug output.
  bool hasInfo() const { return SourceCount != 0; }
  
  /// getNextUniqueID - Returns a unique number for labels used by debugger.
  ///
  unsigned getNextUniqueID() { return UniqueID++; }
  
  bool doInitialization();
  bool doFinalization();
  
  /// getUniqueSourceID - Register a source file with debug info. Returns an id.
  ///
  unsigned getUniqueSourceID(const std::string &fname, 
                             const std::string &dirname);
  
  /// getSourceFiles - Return a vector of files.  Vector index + 1 equals id.
  ///
  std::vector<std::string> getSourceFiles() const;
  
}; // End class MachineDebugInfo

} // End llvm namespace

#endif
