//===- PIC16ConstantPoolValue.h - PIC16 constantpool value ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source 
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the PIC16 specific constantpool value class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_PIC16_CONSTANTPOOLVALUE_H
#define LLVM_TARGET_PIC16_CONSTANTPOOLVALUE_H

#include "llvm/CodeGen/MachineConstantPool.h"

namespace llvm {

class GlobalValue;

namespace PIC16CP {
  enum PIC16CPKind {
    CPValue,
    CPNonLazyPtr,
    CPStub
  };
}

/// PIC16ConstantPoolValue - PIC16 specific constantpool value. This is used to
/// represent PC relative displacement between the address of the load
/// instruction and the global value being loaded, i.e. (&GV-(LPIC+8)).
class PIC16ConstantPoolValue : public MachineConstantPoolValue {
  GlobalValue *GV;         // GlobalValue being loaded.
  const char *S;           // ExtSymbol being loaded.
  unsigned LabelId;        // Label id of the load.
  PIC16CP::PIC16CPKind Kind;   // non_lazy_ptr or stub?
  unsigned char PCAdjust;  // Extra adjustment if constantpool is pc relative.
                           // 8 for PIC16
  const char *Modifier;    // GV modifier i.e. (&GV(modifier)-(LPIC+8))
  bool AddCurrentAddress;

public:
  PIC16ConstantPoolValue(GlobalValue *gv, unsigned id,
                         PIC16CP::PIC16CPKind Kind = PIC16CP::CPValue,
                         unsigned char PCAdj = 0, const char *Modifier = NULL,
                         bool AddCurrentAddress = false);
  PIC16ConstantPoolValue(const char *s, unsigned id,
                         PIC16CP::PIC16CPKind Kind = PIC16CP::CPValue,
                         unsigned char PCAdj = 0, const char *Modifier = NULL,
                         bool AddCurrentAddress = false);
  PIC16ConstantPoolValue(GlobalValue *GV, PIC16CP::PIC16CPKind Kind,
                         const char *Modifier);


  GlobalValue *getGV() const { return GV; }
  const char *getSymbol() const { return S; }
  const char *getModifier() const { return Modifier; }
  bool hasModifier() const { return Modifier != NULL; }
  bool mustAddCurrentAddress() const { return AddCurrentAddress; }
  unsigned getLabelId() const { return LabelId; }
  bool isNonLazyPointer() const { return Kind == PIC16CP::CPNonLazyPtr; }
  bool isStub() const { return Kind == PIC16CP::CPStub; }
  unsigned char getPCAdjustment() const { return PCAdjust; }

  virtual int getExistingMachineCPValue(MachineConstantPool *CP,
                                        unsigned Alignment);

  virtual void AddSelectionDAGCSEId(FoldingSetNodeID &ID);

  virtual void print(std::ostream &O) const;
};
  
}

#endif
