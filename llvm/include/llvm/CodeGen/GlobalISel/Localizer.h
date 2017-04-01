//== llvm/CodeGen/GlobalISel/Localizer.h - Localizer -------------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file This file describes the interface of the 
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_GLOBALISEL_LOCALIZER_H
#define LLVM_CODEGEN_GLOBALISEL_LOCALIZER_H

#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/MachineFunctionPass.h"

namespace llvm {
// Forward declarations.
class MachineRegisterInfo;

/// This pass implements the reg bank selector pass used in the GlobalISel
/// pipeline. At the end of this pass, all register operands have been assigned
class Localizer : public MachineFunctionPass {
public:
  static char ID;

private:
  /// MRI contains all the register class/bank information that this
  /// pass uses and updates.
  MachineRegisterInfo *MRI;

  static bool shouldLocalize(const MachineInstr &MI);

  static bool isLocalUse(MachineOperand &MOUse, const MachineInstr &Def, MachineBasicBlock **InsertMBB);

  /// Initialize the field members using \p MF.
  void init(MachineFunction &MF);

public:
  Localizer();

  StringRef getPassName() const override { return "Localizer"; }

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties()
        .set(MachineFunctionProperties::Property::IsSSA)
        .set(MachineFunctionProperties::Property::Legalized)
        .set(MachineFunctionProperties::Property::RegBankSelected);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
};

} // End namespace llvm.

#endif
