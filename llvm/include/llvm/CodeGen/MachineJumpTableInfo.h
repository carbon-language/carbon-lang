//===-- CodeGen/MachineJumpTableInfo.h - Abstract Jump Tables  --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Nate Begeman and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The MachineJumpTableInfo class keeps track of jump tables referenced by
// lowered switch instructions in the MachineFunction.
//
// Instructions reference the address of these jump tables through the use of 
// MO_JumpTableIndex values.  When emitting assembly or machine code, these 
// virtual address references are converted to refer to the address of the 
// function jump tables.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEJUMPTABLEINFO_H
#define LLVM_CODEGEN_MACHINEJUMPTABLEINFO_H

#include "llvm/Target/TargetData.h"
#include <vector>
#include <iosfwd>

namespace llvm {

class MachineBasicBlock;

/// MachineJumpTableEntry - One jump table in the jump table info.
///
struct MachineJumpTableEntry {
  /// MBBs - The vector of basic blocks from which to create the jump table.
  std::vector<MachineBasicBlock*> MBBs;
  
  MachineJumpTableEntry(std::vector<MachineBasicBlock*> &M) : MBBs(M) {}
};
  
class MachineJumpTableInfo {
  const TargetData &TD;
  std::vector<MachineJumpTableEntry> JumpTables;
public:
  MachineJumpTableInfo(const TargetData &td) : TD(td) {}
    
  /// getJumpTableIndex - Create a new jump table or return an existing one.
  ///
  unsigned getJumpTableIndex(std::vector<MachineBasicBlock*> &DestBBs);
  
  /// isEmpty - Return true if there are no jump tables.
  ///
  bool isEmpty() const { return JumpTables.empty(); }

  const std::vector<MachineJumpTableEntry> &getJumpTables() const {
    return JumpTables;
  }
  
  unsigned getEntrySize() const { return TD.getPointerSize(); }
  unsigned getAlignment() const { return TD.getPointerAlignment(); }
  
  /// print - Used by the MachineFunction printer to print information about
  /// jump tables.  Implemented in MachineFunction.cpp
  ///
  void print(std::ostream &OS) const;

  /// dump - Call print(std::cerr) to be called from the debugger.
  ///
  void dump() const;
};

} // End llvm namespace

#endif
