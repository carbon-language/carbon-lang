//===-- MachOEmitter.h - Target-independent Mach-O Emitter class ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef MACHOCODEEMITTER_H
#define MACHOCODEEMITTER_H

#include "llvm/CodeGen/ObjectCodeEmitter.h"
#include <map>

namespace llvm {

class MachOWriter;

/// MachOCodeEmitter - This class is used by the MachOWriter to emit the code 
/// for functions to the Mach-O file.

class MachOCodeEmitter : public ObjectCodeEmitter {
  MachOWriter &MOW;

  /// Target machine description.
  TargetMachine &TM;

  /// is64Bit/isLittleEndian - This information is inferred from the target
  /// machine directly, indicating what header values and flags to set.
  bool is64Bit, isLittleEndian;

  const MCAsmInfo *TAI;

  /// Relocations - These are the relocations that the function needs, as
  /// emitted.
  std::vector<MachineRelocation> Relocations;

  std::map<uint64_t, uintptr_t> Labels;

public:
  MachOCodeEmitter(MachOWriter &mow, MachOSection &mos);

  virtual void startFunction(MachineFunction &MF);
  virtual bool finishFunction(MachineFunction &MF);

  virtual void addRelocation(const MachineRelocation &MR) {
    Relocations.push_back(MR);
  }

  void emitConstantPool(MachineConstantPool *MCP);
  void emitJumpTables(MachineJumpTableInfo *MJTI);

  virtual void emitLabel(uint64_t LabelID) {
    Labels[LabelID] = getCurrentPCOffset();
  }

  virtual uintptr_t getLabelAddress(uint64_t Label) const {
    return Labels.find(Label)->second;
  }

  virtual void setModuleInfo(llvm::MachineModuleInfo* MMI) { }

}; // end class MachOCodeEmitter

} // end namespace llvm

#endif

