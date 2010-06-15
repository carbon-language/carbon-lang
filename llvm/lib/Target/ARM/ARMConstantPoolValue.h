//===- ARMConstantPoolValue.h - ARM constantpool value ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the ARM specific constantpool value class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_ARM_CONSTANTPOOLVALUE_H
#define LLVM_TARGET_ARM_CONSTANTPOOLVALUE_H

#include "llvm/CodeGen/MachineConstantPool.h"
#include <cstddef>

namespace llvm {

class Constant;
class BlockAddress;
class GlobalValue;
class LLVMContext;

namespace ARMCP {
  enum ARMCPKind {
    CPValue,
    CPExtSymbol,
    CPBlockAddress,
    CPLSDA
  };
}

/// ARMConstantPoolValue - ARM specific constantpool value. This is used to
/// represent PC-relative displacement between the address of the load
/// instruction and the constant being loaded, i.e. (&GV-(LPIC+8)).
class ARMConstantPoolValue : public MachineConstantPoolValue {
  const Constant *CVal;    // Constant being loaded.
  const char *S;           // ExtSymbol being loaded.
  unsigned LabelId;        // Label id of the load.
  ARMCP::ARMCPKind Kind;   // Kind of constant.
  unsigned char PCAdjust;  // Extra adjustment if constantpool is pc-relative.
                           // 8 for ARM, 4 for Thumb.
  const char *Modifier;    // GV modifier i.e. (&GV(modifier)-(LPIC+8))
  bool AddCurrentAddress;

public:
  ARMConstantPoolValue(const Constant *cval, unsigned id,
                       ARMCP::ARMCPKind Kind = ARMCP::CPValue,
                       unsigned char PCAdj = 0, const char *Modifier = NULL,
                       bool AddCurrentAddress = false);
  ARMConstantPoolValue(LLVMContext &C, const char *s, unsigned id,
                       unsigned char PCAdj = 0, const char *Modifier = NULL,
                       bool AddCurrentAddress = false);
  ARMConstantPoolValue(const GlobalValue *GV, const char *Modifier);
  ARMConstantPoolValue();
  ~ARMConstantPoolValue();

  const GlobalValue *getGV() const;
  const char *getSymbol() const { return S; }
  const BlockAddress *getBlockAddress() const;
  const char *getModifier() const { return Modifier; }
  bool hasModifier() const { return Modifier != NULL; }
  bool mustAddCurrentAddress() const { return AddCurrentAddress; }
  unsigned getLabelId() const { return LabelId; }
  unsigned char getPCAdjustment() const { return PCAdjust; }
  bool isGlobalValue() const { return Kind == ARMCP::CPValue; }
  bool isExtSymbol() const { return Kind == ARMCP::CPExtSymbol; }
  bool isBlockAddress() { return Kind == ARMCP::CPBlockAddress; }
  bool isLSDA() { return Kind == ARMCP::CPLSDA; }

  virtual unsigned getRelocationInfo() const {
    // FIXME: This is conservatively claiming that these entries require a
    // relocation, we may be able to do better than this.
    return 2;
  }

  virtual int getExistingMachineCPValue(MachineConstantPool *CP,
                                        unsigned Alignment);

  virtual void AddSelectionDAGCSEId(FoldingSetNodeID &ID);

  /// hasSameValue - Return true if this ARM constpool value
  /// can share the same constantpool entry as another ARM constpool value.
  bool hasSameValue(ARMConstantPoolValue *ACPV);

  void print(raw_ostream *O) const { if (O) print(*O); }
  void print(raw_ostream &O) const;
  void dump() const;
};

inline raw_ostream &operator<<(raw_ostream &O, const ARMConstantPoolValue &V) {
  V.print(O);
  return O;
}

} // End llvm namespace

#endif
