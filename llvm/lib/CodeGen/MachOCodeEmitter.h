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

#include "MachOWriter.h"
#include "llvm/CodeGen/MachineCodeEmitter.h"
#include <vector>

namespace llvm {

/// MachOCodeEmitter - This class is used by the MachOWriter to emit the code 
/// for functions to the Mach-O file.

class MachOCodeEmitter : public MachineCodeEmitter {
  MachOWriter &MOW;

  /// Target machine description.
  TargetMachine &TM;

  /// is64Bit/isLittleEndian - This information is inferred from the target
  /// machine directly, indicating what header values and flags to set.
  bool is64Bit, isLittleEndian;

  const TargetAsmInfo *TAI;

  /// Relocations - These are the relocations that the function needs, as
  /// emitted.
  std::vector<MachineRelocation> Relocations;
  
  /// CPLocations - This is a map of constant pool indices to offsets from the
  /// start of the section for that constant pool index.
  std::vector<uintptr_t> CPLocations;

  /// CPSections - This is a map of constant pool indices to the MachOSection
  /// containing the constant pool entry for that index.
  std::vector<unsigned> CPSections;

  /// JTLocations - This is a map of jump table indices to offsets from the
  /// start of the section for that jump table index.
  std::vector<uintptr_t> JTLocations;

  /// MBBLocations - This vector is a mapping from MBB ID's to their address.
  /// It is filled in by the StartMachineBasicBlock callback and queried by
  /// the getMachineBasicBlockAddress callback.
  std::vector<uintptr_t> MBBLocations;
  
public:
  MachOCodeEmitter(MachOWriter &mow) : MOW(mow), TM(MOW.TM)
  {
    is64Bit = TM.getTargetData()->getPointerSizeInBits() == 64;
    isLittleEndian = TM.getTargetData()->isLittleEndian();
    TAI = TM.getTargetAsmInfo();  
  }

  virtual void startFunction(MachineFunction &MF);
  virtual bool finishFunction(MachineFunction &MF);

  virtual void addRelocation(const MachineRelocation &MR) {
    Relocations.push_back(MR);
  }
  
  void emitConstantPool(MachineConstantPool *MCP);
  void emitJumpTables(MachineJumpTableInfo *MJTI);
  
  virtual uintptr_t getConstantPoolEntryAddress(unsigned Index) const {
    assert(CPLocations.size() > Index && "CP not emitted!");
    return CPLocations[Index];
  }
  virtual uintptr_t getJumpTableEntryAddress(unsigned Index) const {
    assert(JTLocations.size() > Index && "JT not emitted!");
    return JTLocations[Index];
  }

  virtual void StartMachineBasicBlock(MachineBasicBlock *MBB) {
    if (MBBLocations.size() <= (unsigned)MBB->getNumber())
      MBBLocations.resize((MBB->getNumber()+1)*2);
    MBBLocations[MBB->getNumber()] = getCurrentPCOffset();
  }

  virtual uintptr_t getMachineBasicBlockAddress(MachineBasicBlock *MBB) const {
    assert(MBBLocations.size() > (unsigned)MBB->getNumber() && 
           MBBLocations[MBB->getNumber()] && "MBB not emitted!");
    return MBBLocations[MBB->getNumber()];
  }

  virtual uintptr_t getLabelAddress(uint64_t Label) const {
    assert(0 && "get Label not implemented");
    abort();
    return 0;
  }

  virtual void emitLabel(uint64_t LabelID) {
    assert(0 && "emit Label not implemented");
    abort();
  }

  virtual void setModuleInfo(llvm::MachineModuleInfo* MMI) { }

  /// JIT SPECIFIC FUNCTIONS - DO NOT IMPLEMENT THESE HERE!
  virtual void startGVStub(const GlobalValue* F, unsigned StubSize,
                           unsigned Alignment = 1) {
    assert(0 && "JIT specific function called!");
    abort();
  }
  virtual void startGVStub(const GlobalValue* F, void *Buffer, 
                           unsigned StubSize) {
    assert(0 && "JIT specific function called!");
    abort();
  }
  virtual void *finishGVStub(const GlobalValue* F) {
    assert(0 && "JIT specific function called!");
    abort();
    return 0;
  }

}; // end class MachOCodeEmitter

} // end namespace llvm

#endif

