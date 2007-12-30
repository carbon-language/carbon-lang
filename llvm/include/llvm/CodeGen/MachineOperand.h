//===-- llvm/CodeGen/MachineOperand.h - MachineOperand class ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the MachineOperand class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEOPERAND_H
#define LLVM_CODEGEN_MACHINEOPERAND_H

#include "llvm/Support/DataTypes.h"
#include <vector>
#include <cassert>
#include <iosfwd>

namespace llvm {
  
class MachineBasicBlock;
class GlobalValue;
  class MachineInstr;
  
/// MachineOperand class - Representation of each machine instruction operand.
///
class MachineOperand {
public:
  enum MachineOperandType {
    MO_Register,                // Register operand.
    MO_Immediate,               // Immediate Operand
    MO_MachineBasicBlock,       // MachineBasicBlock reference
    MO_FrameIndex,              // Abstract Stack Frame Index
    MO_ConstantPoolIndex,       // Address of indexed Constant in Constant Pool
    MO_JumpTableIndex,          // Address of indexed Jump Table for switch
    MO_ExternalSymbol,          // Name of external global symbol
    MO_GlobalAddress            // Address of a global value
  };

private:
  union {
    GlobalValue *GV;          // For MO_GlobalAddress.
    MachineBasicBlock *MBB;   // For MO_MachineBasicBlock.
    const char *SymbolName;   // For MO_ExternalSymbol.
    unsigned RegNo;           // For MO_Register.
    int64_t ImmVal;           // For MO_Immediate.
    int Index;                // For MO_FrameIndex/CPI/JTI.
  } contents;

  /// ParentMI - This is the instruction that this operand is embedded into.
  MachineInstr *ParentMI;
  
  MachineOperandType opType:8; // Discriminate the union.
  bool IsDef : 1;              // True if this is a def, false if this is a use.
  bool IsImp : 1;              // True if this is an implicit def or use.

  bool IsKill : 1;             // True if this is a reg use and the reg is dead
                               // immediately after the read.
  bool IsDead : 1;             // True if this is a reg def and the reg is dead
                               // immediately after the write. i.e. A register
                               // that is defined but never used.

  /// SubReg - Subregister number, only valid for MO_Register.  A value of 0
  /// indicates the MO_Register has no subReg.
  unsigned char SubReg;
  
  /// auxInfo - auxiliary information used by the MachineOperand
  union {
    /// offset - Offset to address of global or external, only valid for
    /// MO_GlobalAddress, MO_ExternalSym and MO_ConstantPoolIndex
    int offset;

  } auxInfo;
  
  MachineOperand() : ParentMI(0) {}

  void print(std::ostream &os) const;
  void print(std::ostream *os) const { if (os) print(*os); }

public:
  MachineOperand(const MachineOperand &M) {
    *this = M;
  }
  
  ~MachineOperand() {}
  
  /// getType - Returns the MachineOperandType for this operand.
  ///
  MachineOperandType getType() const { return opType; }

  /// getParent - Return the instruction that this operand belongs to.
  ///
  MachineInstr *getParent() { return ParentMI; }
  const MachineInstr *getParent() const { return ParentMI; }
  
  /// Accessors that tell you what kind of MachineOperand you're looking at.
  ///
  bool isRegister() const { return opType == MO_Register; }
  bool isImmediate() const { return opType == MO_Immediate; }
  bool isMachineBasicBlock() const { return opType == MO_MachineBasicBlock; }
  bool isFrameIndex() const { return opType == MO_FrameIndex; }
  bool isConstantPoolIndex() const { return opType == MO_ConstantPoolIndex; }
  bool isJumpTableIndex() const { return opType == MO_JumpTableIndex; }
  bool isGlobalAddress() const { return opType == MO_GlobalAddress; }
  bool isExternalSymbol() const { return opType == MO_ExternalSymbol; }

  int64_t getImm() const {
    assert(isImmediate() && "Wrong MachineOperand accessor");
    return contents.ImmVal;
  }
  
  MachineBasicBlock *getMBB() const {
    assert(isMachineBasicBlock() && "Wrong MachineOperand accessor");
    return contents.MBB;
  }
  MachineBasicBlock *getMachineBasicBlock() const {
    assert(isMachineBasicBlock() && "Wrong MachineOperand accessor");
    return contents.MBB;
  }
  void setMachineBasicBlock(MachineBasicBlock *MBB) {
    assert(isMachineBasicBlock() && "Wrong MachineOperand accessor");
    contents.MBB = MBB;
  }
  int getFrameIndex() const {
    assert(isFrameIndex() && "Wrong MachineOperand accessor");
    return (int)contents.Index;
  }
  unsigned getConstantPoolIndex() const {
    assert(isConstantPoolIndex() && "Wrong MachineOperand accessor");
    return (unsigned)contents.Index;
  }
  unsigned getJumpTableIndex() const {
    assert(isJumpTableIndex() && "Wrong MachineOperand accessor");
    return (unsigned)contents.Index;
  }
  GlobalValue *getGlobal() const {
    assert(isGlobalAddress() && "Wrong MachineOperand accessor");
    return contents.GV;
  }
  int getOffset() const {
    assert((isGlobalAddress() || isExternalSymbol() || isConstantPoolIndex()) &&
        "Wrong MachineOperand accessor");
    return auxInfo.offset;
  }
  unsigned getSubReg() const {
    assert(isRegister() && "Wrong MachineOperand accessor");
    return (unsigned)SubReg;
  }
  const char *getSymbolName() const {
    assert(isExternalSymbol() && "Wrong MachineOperand accessor");
    return contents.SymbolName;
  }

  bool isUse() const { 
    assert(isRegister() && "Wrong MachineOperand accessor");
    return !IsDef;
  }
  bool isDef() const {
    assert(isRegister() && "Wrong MachineOperand accessor");
    return IsDef;
  }
  void setIsUse() {
    assert(isRegister() && "Wrong MachineOperand accessor");
    IsDef = false;
  }
  void setIsDef() {
    assert(isRegister() && "Wrong MachineOperand accessor");
    IsDef = true;
  }

  bool isImplicit() const { 
    assert(isRegister() && "Wrong MachineOperand accessor");
    return IsImp;
  }
  void setImplicit() { 
    assert(isRegister() && "Wrong MachineOperand accessor");
    IsImp = true;
  }

  bool isKill() const {
    assert(isRegister() && "Wrong MachineOperand accessor");
    return IsKill;
  }
  bool isDead() const {
    assert(isRegister() && "Wrong MachineOperand accessor");
    return IsDead;
  }
  void setIsKill() {
    assert(isRegister() && !IsDef && "Wrong MachineOperand accessor");
    IsKill = true;
  }
  void setIsDead() {
    assert(isRegister() && IsDef && "Wrong MachineOperand accessor");
    IsDead = true;
  }
  void unsetIsKill() {
    assert(isRegister() && !IsDef && "Wrong MachineOperand accessor");
    IsKill = false;
  }
  void unsetIsDead() {
    assert(isRegister() && IsDef && "Wrong MachineOperand accessor");
    IsDead = false;
  }

  /// getReg - Returns the register number.
  ///
  unsigned getReg() const {
    assert(isRegister() && "This is not a register operand!");
    return contents.RegNo;
  }

  /// MachineOperand mutators.
  ///
  void setReg(unsigned Reg) {
    assert(isRegister() && "This is not a register operand!");
    contents.RegNo = Reg;
  }

  void setImm(int64_t immVal) {
    assert(isImmediate() && "Wrong MachineOperand mutator");
    contents.ImmVal = immVal;
  }

  void setOffset(int Offset) {
    assert((isGlobalAddress() || isExternalSymbol() || isConstantPoolIndex()) &&
        "Wrong MachineOperand accessor");
    auxInfo.offset = Offset;
  }
  void setSubReg(unsigned subReg) {
    assert(isRegister() && "Wrong MachineOperand accessor");
    SubReg = (unsigned char)subReg;
  }
  void setConstantPoolIndex(unsigned Idx) {
    assert(isConstantPoolIndex() && "Wrong MachineOperand accessor");
    contents.Index = Idx;
  }
  void setJumpTableIndex(unsigned Idx) {
    assert(isJumpTableIndex() && "Wrong MachineOperand accessor");
    contents.Index = Idx;
  }
  
  /// isIdenticalTo - Return true if this operand is identical to the specified
  /// operand. Note: This method ignores isKill and isDead properties.
  bool isIdenticalTo(const MachineOperand &Other) const;
  
  /// ChangeToImmediate - Replace this operand with a new immediate operand of
  /// the specified value.  If an operand is known to be an immediate already,
  /// the setImm method should be used.
  void ChangeToImmediate(int64_t ImmVal) {
    opType = MO_Immediate;
    contents.ImmVal = ImmVal;
  }

  /// ChangeToRegister - Replace this operand with a new register operand of
  /// the specified value.  If an operand is known to be an register already,
  /// the setReg method should be used.
  void ChangeToRegister(unsigned Reg, bool isDef, bool isImp = false,
                        bool isKill = false, bool isDead = false) {
    opType = MO_Register;
    contents.RegNo = Reg;
    IsDef = isDef;
    IsImp = isImp;
    IsKill = isKill;
    IsDead = isDead;
    SubReg = 0;
  }
  
  static MachineOperand CreateImm(int64_t Val) {
    MachineOperand Op;
    Op.opType = MachineOperand::MO_Immediate;
    Op.contents.ImmVal = Val;
    return Op;
  }
  static MachineOperand CreateReg(unsigned Reg, bool isDef, bool isImp = false,
                                  bool isKill = false, bool isDead = false,
                                  unsigned SubReg = 0) {
    MachineOperand Op;
    Op.opType = MachineOperand::MO_Register;
    Op.IsDef = isDef;
    Op.IsImp = isImp;
    Op.IsKill = isKill;
    Op.IsDead = isDead;
    Op.contents.RegNo = Reg;
    Op.SubReg = SubReg;
    return Op;
  }
  static MachineOperand CreateMBB(MachineBasicBlock *MBB) {
    MachineOperand Op;
    Op.opType = MachineOperand::MO_MachineBasicBlock;
    Op.contents.MBB = MBB;
    return Op;
  }
  static MachineOperand CreateFI(unsigned Idx) {
    MachineOperand Op;
    Op.opType = MachineOperand::MO_FrameIndex;
    Op.contents.Index = Idx;
    return Op;
  }
  static MachineOperand CreateCPI(unsigned Idx, int Offset) {
    MachineOperand Op;
    Op.opType = MachineOperand::MO_ConstantPoolIndex;
    Op.contents.Index = Idx;
    Op.auxInfo.offset = Offset;
    return Op;
  }
  static MachineOperand CreateJTI(unsigned Idx) {
    MachineOperand Op;
    Op.opType = MachineOperand::MO_JumpTableIndex;
    Op.contents.Index = Idx;
    return Op;
  }
  static MachineOperand CreateGA(GlobalValue *GV, int Offset) {
    MachineOperand Op;
    Op.opType = MachineOperand::MO_GlobalAddress;
    Op.contents.GV = GV;
    Op.auxInfo.offset = Offset;
    return Op;
  }
  static MachineOperand CreateES(const char *SymName, int Offset = 0) {
    MachineOperand Op;
    Op.opType = MachineOperand::MO_ExternalSymbol;
    Op.contents.SymbolName = SymName;
    Op.auxInfo.offset = Offset;
    return Op;
  }
  const MachineOperand &operator=(const MachineOperand &MO) {
    contents = MO.contents;
    IsDef    = MO.IsDef;
    IsImp    = MO.IsImp;
    IsKill   = MO.IsKill;
    IsDead   = MO.IsDead;
    opType   = MO.opType;
    auxInfo  = MO.auxInfo;
    SubReg   = MO.SubReg;
    ParentMI = MO.ParentMI;
    return *this;
  }

  friend std::ostream& operator<<(std::ostream& os, const MachineOperand& mop) {
    mop.print(os);
    return os;
  }

  friend class MachineInstr;
};

std::ostream& operator<<(std::ostream &OS, const MachineOperand &MO);

} // End llvm namespace

#endif
