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

#include "ELFWriter.h"
#include "llvm/CodeGen/MachineCodeEmitter.h"
#include <vector>

namespace llvm {

  /// ELFCodeEmitter - This class is used by the ELFWriter to 
  /// emit the code for functions to the ELF file.
  class ELFCodeEmitter : public MachineCodeEmitter {
    ELFWriter &EW;
    TargetMachine &TM;
    ELFWriter::ELFSection *ES;  // Section to write to.
    uint8_t *FnStartPtr;
  public:
    explicit ELFCodeEmitter(ELFWriter &ew) : EW(ew), TM(EW.TM) {}

    void startFunction(MachineFunction &F);
    bool finishFunction(MachineFunction &F);

    void addRelocation(const MachineRelocation &MR) {
      assert(0 && "relo not handled yet!");
    }

    virtual void StartMachineBasicBlock(MachineBasicBlock *MBB) {
    }

    virtual uintptr_t getConstantPoolEntryAddress(unsigned Index) const {
      assert(0 && "CP not implementated yet!");
      return 0;
    }
    virtual uintptr_t getJumpTableEntryAddress(unsigned Index) const {
      assert(0 && "JT not implementated yet!");
      return 0;
    }

    virtual uintptr_t getMachineBasicBlockAddress(MachineBasicBlock *MBB) const {
      assert(0 && "JT not implementated yet!");
      return 0;
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

    virtual void setModuleInfo(llvm::MachineModuleInfo* MMI) { }

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

