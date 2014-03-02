//===- SystemZConstantPoolValue.h - SystemZ constant-pool value -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef SYSTEMZCONSTANTPOOLVALUE_H
#define SYSTEMZCONSTANTPOOLVALUE_H

#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/Support/ErrorHandling.h"

namespace llvm {

class GlobalValue;

namespace SystemZCP {
  enum SystemZCPModifier {
    NTPOFF
  };
}

/// A SystemZ-specific constant pool value.  At present, the only
/// defined constant pool values are offsets of thread-local variables
/// (written x@NTPOFF).
class SystemZConstantPoolValue : public MachineConstantPoolValue {
  const GlobalValue *GV;
  SystemZCP::SystemZCPModifier Modifier;

protected:
  SystemZConstantPoolValue(const GlobalValue *GV,
                           SystemZCP::SystemZCPModifier Modifier);

public:
  static SystemZConstantPoolValue *
    Create(const GlobalValue *GV, SystemZCP::SystemZCPModifier Modifier);

  // Override MachineConstantPoolValue.
  virtual unsigned getRelocationInfo() const override;
  virtual int getExistingMachineCPValue(MachineConstantPool *CP,
                                        unsigned Alignment) override;
  virtual void addSelectionDAGCSEId(FoldingSetNodeID &ID) override;
  virtual void print(raw_ostream &O) const override;

  // Access SystemZ-specific fields.
  const GlobalValue *getGlobalValue() const { return GV; }
  SystemZCP::SystemZCPModifier getModifier() const { return Modifier; }
};

} // End llvm namespace

#endif
