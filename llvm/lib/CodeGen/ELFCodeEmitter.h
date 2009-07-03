//===-- lib/CodeGen/ELFCodeEmitter.h ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef ELFCODEEMITTER_H
#define ELFCODEEMITTER_H

#include "llvm/CodeGen/MachineCodeEmitter.h"
#include <vector>

namespace llvm {
  class ELFWriter;
  class ELFSection;

  /// ELFCodeEmitter - This class is used by the ELFWriter to 
  /// emit the code for functions to the ELF file.
  class ELFCodeEmitter : public MachineCodeEmitter {
    ELFWriter &EW;

    /// Target machine description
    TargetMachine &TM;

    /// Section containing code for functions
    ELFSection *ES;

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

    /// FnStartPtr - Pointer to the start location of the current function
    /// in the buffer
    uint8_t *FnStartPtr;

    /// JumpTableSectionIdx - Holds the index of the Jump Table Section 
    unsigned JumpTableSectionIdx;
  public:
    explicit ELFCodeEmitter(ELFWriter &ew) : EW(ew), TM(EW.TM),
                                             JumpTableSectionIdx(0) {}

    void startFunction(MachineFunction &F);
    bool finishFunction(MachineFunction &F);

    void addRelocation(const MachineRelocation &MR) {
      Relocations.push_back(MR);
    }

    virtual void StartMachineBasicBlock(MachineBasicBlock *MBB) {
      if (MBBLocations.size() <= (unsigned)MBB->getNumber())
        MBBLocations.resize((MBB->getNumber()+1)*2);
      MBBLocations[MBB->getNumber()] = getCurrentPCOffset();
    }

    virtual uintptr_t getConstantPoolEntryAddress(unsigned Index) const {
      assert(CPLocations.size() > Index && "CP not emitted!");
      return CPLocations[Index];
    }

    virtual uintptr_t getJumpTableEntryAddress(unsigned Index) const {
      assert(JTLocations.size() > Index && "JT not emitted!");
      return JTLocations[Index];
    }

    virtual uintptr_t getMachineBasicBlockAddress(MachineBasicBlock *MBB) const {
      assert(MBBLocations.size() > (unsigned)MBB->getNumber() && 
             MBBLocations[MBB->getNumber()] && "MBB not emitted!");
      return MBBLocations[MBB->getNumber()];
    }

    virtual uintptr_t getLabelAddress(uint64_t Label) const {
      assert(0 && "Label address not implementated yet!");
      abort();
      return 0;
    }

    virtual void emitLabel(uint64_t LabelID) {
      assert(0 && "emit Label not implementated yet!");
      abort();
    }

    /// emitConstantPool - For each constant pool entry, figure out which section
    /// the constant should live in and emit the constant.
    void emitConstantPool(MachineConstantPool *MCP);

    /// emitJumpTables - Emit all the jump tables for a given jump table info
    /// record to the appropriate section.
    void emitJumpTables(MachineJumpTableInfo *MJTI);

    virtual void setModuleInfo(llvm::MachineModuleInfo* MMI) {}

    /// JIT SPECIFIC FUNCTIONS - DO NOT IMPLEMENT THESE HERE!
    void startGVStub(const GlobalValue* F, unsigned StubSize,
                     unsigned Alignment = 1) {
      assert(0 && "JIT specific function called!");
      abort();
    }
    void startGVStub(const GlobalValue* F,  void *Buffer, unsigned StubSize) {
      assert(0 && "JIT specific function called!");
      abort();
    }
    void *finishGVStub(const GlobalValue *F) {
      assert(0 && "JIT specific function called!");
      abort();
      return 0;
    }
};  // end class ELFCodeEmitter

} // end namespace llvm

#endif

