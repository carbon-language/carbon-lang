//===- ARMBaseRegisterInfo.h - ARM Register Information Impl ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the base ARM implementation of TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef ARMBASEREGISTERINFO_H
#define ARMBASEREGISTERINFO_H

#include "ARM.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "ARMGenRegisterInfo.h.inc"

namespace llvm {
  class ARMSubtarget;
  class ARMBaseInstrInfo;
  class Type;

/// Register allocation hints.
namespace ARMRI {
  enum {
    RegPairOdd  = 1,
    RegPairEven = 2
  };
}

/// isARMLowRegister - Returns true if the register is low register r0-r7.
///
static inline bool isARMLowRegister(unsigned Reg) {
  using namespace ARM;
  switch (Reg) {
  case R0:  case R1:  case R2:  case R3:
  case R4:  case R5:  case R6:  case R7:
    return true;
  default:
    return false;
  }
}

/// isARMArea1Register - Returns true if the register is a low register (r0-r7)
/// or a stack/pc register that we should push/pop.
static inline bool isARMArea1Register(unsigned Reg, bool isDarwin) {
  using namespace ARM;
  switch (Reg) {
    case R0:  case R1:  case R2:  case R3:
    case R4:  case R5:  case R6:  case R7:
    case LR:  case SP:  case PC:
      return true;
    case R8:  case R9:  case R10: case R11:
      // For darwin we want r7 and lr to be next to each other.
      return !isDarwin;
    default:
      return false;
  }
}

static inline bool isARMArea2Register(unsigned Reg, bool isDarwin) {
  using namespace ARM;
  switch (Reg) {
    case R8: case R9: case R10: case R11:
      // Darwin has this second area.
      return isDarwin;
    default:
      return false;
  }
}

static inline bool isARMArea3Register(unsigned Reg, bool isDarwin) {
  using namespace ARM;
  switch (Reg) {
    case D15: case D14: case D13: case D12:
    case D11: case D10: case D9:  case D8:
      return true;
    default:
      return false;
  }
}

class ARMBaseRegisterInfo : public ARMGenRegisterInfo {
protected:
  const ARMBaseInstrInfo &TII;
  const ARMSubtarget &STI;

  /// FramePtr - ARM physical register used as frame ptr.
  unsigned FramePtr;

  /// BasePtr - ARM physical register used as a base ptr in complex stack
  /// frames. I.e., when we need a 3rd base, not just SP and FP, due to
  /// variable size stack objects.
  unsigned BasePtr;

  // Can be only subclassed.
  explicit ARMBaseRegisterInfo(const ARMBaseInstrInfo &tii,
                               const ARMSubtarget &STI);

  // Return the opcode that implements 'Op', or 0 if no opcode
  unsigned getOpcode(int Op) const;

public:
  /// Code Generation virtual methods...
  const unsigned *getCalleeSavedRegs(const MachineFunction *MF = 0) const;

  BitVector getReservedRegs(const MachineFunction &MF) const;

  /// getMatchingSuperRegClass - Return a subclass of the specified register
  /// class A so that each register in it has a sub-register of the
  /// specified sub-register index which is in the specified register class B.
  virtual const TargetRegisterClass *
  getMatchingSuperRegClass(const TargetRegisterClass *A,
                           const TargetRegisterClass *B, unsigned Idx) const;

  /// canCombineSubRegIndices - Given a register class and a list of
  /// subregister indices, return true if it's possible to combine the
  /// subregister indices into one that corresponds to a larger
  /// subregister. Return the new subregister index by reference. Note the
  /// new index may be zero if the given subregisters can be combined to
  /// form the whole register.
  virtual bool canCombineSubRegIndices(const TargetRegisterClass *RC,
                                       SmallVectorImpl<unsigned> &SubIndices,
                                       unsigned &NewSubIdx) const;

  const TargetRegisterClass *getPointerRegClass(unsigned Kind = 0) const;

  std::pair<TargetRegisterClass::iterator,TargetRegisterClass::iterator>
  getAllocationOrder(const TargetRegisterClass *RC,
                     unsigned HintType, unsigned HintReg,
                     const MachineFunction &MF) const;

  unsigned ResolveRegAllocHint(unsigned Type, unsigned Reg,
                               const MachineFunction &MF) const;

  void UpdateRegAllocHint(unsigned Reg, unsigned NewReg,
                          MachineFunction &MF) const;

  bool hasBasePointer(const MachineFunction &MF) const;

  bool canRealignStack(const MachineFunction &MF) const;
  bool needsStackRealignment(const MachineFunction &MF) const;
  int64_t getFrameIndexInstrOffset(const MachineInstr *MI, int Idx) const;
  bool needsFrameBaseReg(MachineInstr *MI, int64_t Offset) const;
  void materializeFrameBaseRegister(MachineBasicBlock *MBB,
                                    unsigned BaseReg, int FrameIdx,
                                    int64_t Offset) const;
  void resolveFrameIndex(MachineBasicBlock::iterator I,
                         unsigned BaseReg, int64_t Offset) const;
  bool isFrameOffsetLegal(const MachineInstr *MI, int64_t Offset) const;

  bool cannotEliminateFrame(const MachineFunction &MF) const;

  // Debug information queries.
  unsigned getRARegister() const;
  unsigned getFrameRegister(const MachineFunction &MF) const;
  unsigned getBaseRegister() const { return BasePtr; }

  // Exception handling queries.
  unsigned getEHExceptionRegister() const;
  unsigned getEHHandlerRegister() const;

  int getDwarfRegNum(unsigned RegNum, bool isEH) const;

  bool isLowRegister(unsigned Reg) const;


  /// emitLoadConstPool - Emits a load from constpool to materialize the
  /// specified immediate.
  virtual void emitLoadConstPool(MachineBasicBlock &MBB,
                                 MachineBasicBlock::iterator &MBBI,
                                 DebugLoc dl,
                                 unsigned DestReg, unsigned SubIdx,
                                 int Val,
                                 ARMCC::CondCodes Pred = ARMCC::AL,
                                 unsigned PredReg = 0) const;

  /// Code Generation virtual methods...
  virtual bool isReservedReg(const MachineFunction &MF, unsigned Reg) const;

  virtual bool requiresRegisterScavenging(const MachineFunction &MF) const;

  virtual bool requiresFrameIndexScavenging(const MachineFunction &MF) const;

  virtual bool requiresVirtualBaseRegisters(const MachineFunction &MF) const;

  virtual void eliminateCallFramePseudoInstr(MachineFunction &MF,
                                           MachineBasicBlock &MBB,
                                           MachineBasicBlock::iterator I) const;

  virtual void eliminateFrameIndex(MachineBasicBlock::iterator II,
                                   int SPAdj, RegScavenger *RS = NULL) const;

private:
  unsigned getRegisterPairEven(unsigned Reg, const MachineFunction &MF) const;

  unsigned getRegisterPairOdd(unsigned Reg, const MachineFunction &MF) const;
};

} // end namespace llvm

#endif
