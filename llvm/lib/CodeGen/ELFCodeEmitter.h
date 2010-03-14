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

#include "llvm/CodeGen/ObjectCodeEmitter.h"
#include <vector>

namespace llvm {
  class ELFWriter;
  class ELFSection;

  /// ELFCodeEmitter - This class is used by the ELFWriter to 
  /// emit the code for functions to the ELF file.
  class ELFCodeEmitter : public ObjectCodeEmitter {
    ELFWriter &EW;

    /// Target machine description
    TargetMachine &TM;

    /// Section containing code for functions
    ELFSection *ES;

    /// Relocations - Record relocations needed by the current function 
    std::vector<MachineRelocation> Relocations;

    /// JTRelocations - Record relocations needed by the relocation
    /// section.
    std::vector<MachineRelocation> JTRelocations;

    /// FnStartPtr - Function offset from the beginning of ELFSection 'ES'
    uintptr_t FnStartOff;
  public:
    explicit ELFCodeEmitter(ELFWriter &ew) : EW(ew), TM(EW.TM) {}

    /// addRelocation - Register new relocations for this function
    void addRelocation(const MachineRelocation &MR) {
      Relocations.push_back(MR);
    }

    /// emitConstantPool - For each constant pool entry, figure out which
    /// section the constant should live in and emit data to it
    void emitConstantPool(MachineConstantPool *MCP);

    /// emitJumpTables - Emit all the jump tables for a given jump table
    /// info and record them to the appropriate section.
    void emitJumpTables(MachineJumpTableInfo *MJTI);

    void startFunction(MachineFunction &F);
    bool finishFunction(MachineFunction &F);

    /// emitLabel - Emits a label
    virtual void emitLabel(MCSymbol *Label) {
      assert("emitLabel not implemented");
    }

    /// getLabelAddress - Return the address of the specified LabelID, 
    /// only usable after the LabelID has been emitted.
    virtual uintptr_t getLabelAddress(MCSymbol *Label) const {
      assert("getLabelAddress not implemented");
      return 0;
    }

    virtual void setModuleInfo(llvm::MachineModuleInfo* MMI) {}

};  // end class ELFCodeEmitter

} // end namespace llvm

#endif

