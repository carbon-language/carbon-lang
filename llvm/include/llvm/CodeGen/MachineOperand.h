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
#include <cassert>
#include <iosfwd>

namespace llvm {
  
class ConstantFP;
class MachineBasicBlock;
class GlobalValue;
class MachineInstr;
class TargetMachine;
class MachineRegisterInfo;
class raw_ostream;
  
/// MachineOperand class - Representation of each machine instruction operand.
///
class MachineOperand {
public:
  enum MachineOperandType {
    MO_Register,                // Register operand.
    MO_Immediate,               // Immediate Operand
    MO_FPImmediate,
    MO_MachineBasicBlock,       // MachineBasicBlock reference
    MO_FrameIndex,              // Abstract Stack Frame Index
    MO_ConstantPoolIndex,       // Address of indexed Constant in Constant Pool
    MO_JumpTableIndex,          // Address of indexed Jump Table for switch
    MO_ExternalSymbol,          // Name of external global symbol
    MO_GlobalAddress            // Address of a global value
  };

private:
  /// OpKind - Specify what kind of operand this is.  This discriminates the
  /// union.
  MachineOperandType OpKind : 8;
  
  /// IsDef/IsImp/IsKill/IsDead flags - These are only valid for MO_Register
  /// operands.
  
  /// IsDef - True if this is a def, false if this is a use of the register.
  ///
  bool IsDef : 1;
  
  /// IsImp - True if this is an implicit def or use, false if it is explicit.
  ///
  bool IsImp : 1;

  /// IsKill - True if this instruction is the last use of the register on this
  /// path through the function.  This is only valid on uses of registers.
  bool IsKill : 1;

  /// IsDead - True if this register is never used by a subsequent instruction.
  /// This is only valid on definitions of registers.
  bool IsDead : 1;

  /// IsEarlyClobber - True if this MO_Register operand is marked earlyclobber
  /// in an inline asm.  Flag is not valid for any other case.   See gcc doc
  /// for description of earlyclobber.
  bool IsEarlyClobber : 1;

  /// OverlapsEarlyClobber - True if this MO_Register operand is used as an
  /// input to an inline asm that has the earlyclobber bit set on some other
  /// operand.  Flag is not valid for any other case.   See gcc doc
  /// for description of earlyclobber.
  bool OverlapsEarlyClobber : 1;

  /// SubReg - Subregister number, only valid for MO_Register.  A value of 0
  /// indicates the MO_Register has no subReg.
  unsigned char SubReg;
  
  /// ParentMI - This is the instruction that this operand is embedded into. 
  /// This is valid for all operand types, when the operand is in an instr.
  MachineInstr *ParentMI;

  /// Contents union - This contains the payload for the various operand types.
  union {
    MachineBasicBlock *MBB;   // For MO_MachineBasicBlock.
    const ConstantFP *CFP;    // For MO_FPImmediate.
    int64_t ImmVal;           // For MO_Immediate.

    struct {                  // For MO_Register.
      unsigned RegNo;
      MachineOperand **Prev;  // Access list for register.
      MachineOperand *Next;
    } Reg;
    
    /// OffsetedInfo - This struct contains the offset and an object identifier.
    /// this represent the object as with an optional offset from it.
    struct {
      union {
        int Index;                // For MO_*Index - The index itself.
        const char *SymbolName;   // For MO_ExternalSymbol.
        GlobalValue *GV;          // For MO_GlobalAddress.
      } Val;
      int Offset;   // An offset from the object.
    } OffsetedInfo;
  } Contents;
  
  explicit MachineOperand(MachineOperandType K) : OpKind(K), ParentMI(0) {}
public:
  MachineOperand(const MachineOperand &M) {
    *this = M;
  }
  
  ~MachineOperand() {}
  
  /// getType - Returns the MachineOperandType for this operand.
  ///
  MachineOperandType getType() const { return OpKind; }

  /// getParent - Return the instruction that this operand belongs to.
  ///
  MachineInstr *getParent() { return ParentMI; }
  const MachineInstr *getParent() const { return ParentMI; }
  
  void print(std::ostream &os, const TargetMachine *TM = 0) const;
  void print(raw_ostream &os, const TargetMachine *TM = 0) const;

  /// Accessors that tell you what kind of MachineOperand you're looking at.
  ///
  bool isRegister() const { return OpKind == MO_Register; }
  bool isImmediate() const { return OpKind == MO_Immediate; }
  bool isFPImmediate() const { return OpKind == MO_FPImmediate; }
  bool isMachineBasicBlock() const { return OpKind == MO_MachineBasicBlock; }
  bool isFrameIndex() const { return OpKind == MO_FrameIndex; }
  bool isConstantPoolIndex() const { return OpKind == MO_ConstantPoolIndex; }
  bool isJumpTableIndex() const { return OpKind == MO_JumpTableIndex; }
  bool isGlobalAddress() const { return OpKind == MO_GlobalAddress; }
  bool isExternalSymbol() const { return OpKind == MO_ExternalSymbol; }

  //===--------------------------------------------------------------------===//
  // Accessors for Register Operands
  //===--------------------------------------------------------------------===//

  /// getReg - Returns the register number.
  unsigned getReg() const {
    assert(isRegister() && "This is not a register operand!");
    return Contents.Reg.RegNo;
  }
  
  unsigned getSubReg() const {
    assert(isRegister() && "Wrong MachineOperand accessor");
    return (unsigned)SubReg;
  }
  
  bool isUse() const { 
    assert(isRegister() && "Wrong MachineOperand accessor");
    return !IsDef;
  }
  
  bool isDef() const {
    assert(isRegister() && "Wrong MachineOperand accessor");
    return IsDef;
  }
  
  bool isImplicit() const { 
    assert(isRegister() && "Wrong MachineOperand accessor");
    return IsImp;
  }
  
  bool isDead() const {
    assert(isRegister() && "Wrong MachineOperand accessor");
    return IsDead;
  }
  
  bool isKill() const {
    assert(isRegister() && "Wrong MachineOperand accessor");
    return IsKill;
  }
  
  bool isEarlyClobber() const {
    assert(isRegister() && "Wrong MachineOperand accessor");
    return IsEarlyClobber;
  }

  bool overlapsEarlyClobber() const {
    assert(isRegister() && "Wrong MachineOperand accessor");
    return OverlapsEarlyClobber;
  }

  /// getNextOperandForReg - Return the next MachineOperand in the function that
  /// uses or defines this register.
  MachineOperand *getNextOperandForReg() const {
    assert(isRegister() && "This is not a register operand!");
    return Contents.Reg.Next;
  }

  //===--------------------------------------------------------------------===//
  // Mutators for Register Operands
  //===--------------------------------------------------------------------===//
  
  /// Change the register this operand corresponds to.
  ///
  void setReg(unsigned Reg);
  
  void setSubReg(unsigned subReg) {
    assert(isRegister() && "Wrong MachineOperand accessor");
    SubReg = (unsigned char)subReg;
  }
  
  void setIsUse(bool Val = true) {
    assert(isRegister() && "Wrong MachineOperand accessor");
    IsDef = !Val;
  }
  
  void setIsDef(bool Val = true) {
    assert(isRegister() && "Wrong MachineOperand accessor");
    IsDef = Val;
  }

  void setImplicit(bool Val = true) { 
    assert(isRegister() && "Wrong MachineOperand accessor");
    IsImp = Val;
  }

  void setIsKill(bool Val = true) {
    assert(isRegister() && !IsDef && "Wrong MachineOperand accessor");
    IsKill = Val;
  }
  
  void setIsDead(bool Val = true) {
    assert(isRegister() && IsDef && "Wrong MachineOperand accessor");
    IsDead = Val;
  }

  void setIsEarlyClobber(bool Val = true) {
    assert(isRegister() && IsDef && "Wrong MachineOperand accessor");
    IsEarlyClobber = Val;
  }

  void setOverlapsEarlyClobber(bool Val = true) {
    assert(isRegister() && "Wrong MachineOperand accessor");
    OverlapsEarlyClobber = Val;
  }

  //===--------------------------------------------------------------------===//
  // Accessors for various operand types.
  //===--------------------------------------------------------------------===//
  
  int64_t getImm() const {
    assert(isImmediate() && "Wrong MachineOperand accessor");
    return Contents.ImmVal;
  }
  
  const ConstantFP *getFPImm() const {
    assert(isFPImmediate() && "Wrong MachineOperand accessor");
    return Contents.CFP;
  }
  
  MachineBasicBlock *getMBB() const {
    assert(isMachineBasicBlock() && "Wrong MachineOperand accessor");
    return Contents.MBB;
  }

  int getIndex() const {
    assert((isFrameIndex() || isConstantPoolIndex() || isJumpTableIndex()) &&
           "Wrong MachineOperand accessor");
    return Contents.OffsetedInfo.Val.Index;
  }
  
  GlobalValue *getGlobal() const {
    assert(isGlobalAddress() && "Wrong MachineOperand accessor");
    return Contents.OffsetedInfo.Val.GV;
  }
  
  int getOffset() const {
    assert((isGlobalAddress() || isExternalSymbol() || isConstantPoolIndex()) &&
           "Wrong MachineOperand accessor");
    return Contents.OffsetedInfo.Offset;
  }
  
  const char *getSymbolName() const {
    assert(isExternalSymbol() && "Wrong MachineOperand accessor");
    return Contents.OffsetedInfo.Val.SymbolName;
  }
  
  //===--------------------------------------------------------------------===//
  // Mutators for various operand types.
  //===--------------------------------------------------------------------===//
  
  void setImm(int64_t immVal) {
    assert(isImmediate() && "Wrong MachineOperand mutator");
    Contents.ImmVal = immVal;
  }

  void setOffset(int Offset) {
    assert((isGlobalAddress() || isExternalSymbol() || isConstantPoolIndex()) &&
        "Wrong MachineOperand accessor");
    Contents.OffsetedInfo.Offset = Offset;
  }
  
  void setIndex(int Idx) {
    assert((isFrameIndex() || isConstantPoolIndex() || isJumpTableIndex()) &&
           "Wrong MachineOperand accessor");
    Contents.OffsetedInfo.Val.Index = Idx;
  }
  
  void setMBB(MachineBasicBlock *MBB) {
    assert(isMachineBasicBlock() && "Wrong MachineOperand accessor");
    Contents.MBB = MBB;
  }
  
  //===--------------------------------------------------------------------===//
  // Other methods.
  //===--------------------------------------------------------------------===//
  
  /// isIdenticalTo - Return true if this operand is identical to the specified
  /// operand. Note: This method ignores isKill and isDead properties.
  bool isIdenticalTo(const MachineOperand &Other) const;
  
  /// ChangeToImmediate - Replace this operand with a new immediate operand of
  /// the specified value.  If an operand is known to be an immediate already,
  /// the setImm method should be used.
  void ChangeToImmediate(int64_t ImmVal);
  
  /// ChangeToRegister - Replace this operand with a new register operand of
  /// the specified value.  If an operand is known to be an register already,
  /// the setReg method should be used.
  void ChangeToRegister(unsigned Reg, bool isDef, bool isImp = false,
                        bool isKill = false, bool isDead = false);
  
  //===--------------------------------------------------------------------===//
  // Construction methods.
  //===--------------------------------------------------------------------===//
  
  static MachineOperand CreateImm(int64_t Val) {
    MachineOperand Op(MachineOperand::MO_Immediate);
    Op.setImm(Val);
    return Op;
  }
  
  static MachineOperand CreateFPImm(const ConstantFP *CFP) {
    MachineOperand Op(MachineOperand::MO_FPImmediate);
    Op.Contents.CFP = CFP;
    return Op;
  }
  
  static MachineOperand CreateReg(unsigned Reg, bool isDef, bool isImp = false,
                                  bool isKill = false, bool isDead = false,
                                  unsigned SubReg = 0,
                                  bool isEarlyClobber = false,
                                  bool overlapsEarlyClobber = false) {
    MachineOperand Op(MachineOperand::MO_Register);
    Op.IsDef = isDef;
    Op.IsImp = isImp;
    Op.IsKill = isKill;
    Op.IsDead = isDead;
    Op.IsEarlyClobber = isEarlyClobber;
    Op.OverlapsEarlyClobber = overlapsEarlyClobber;
    Op.Contents.Reg.RegNo = Reg;
    Op.Contents.Reg.Prev = 0;
    Op.Contents.Reg.Next = 0;
    Op.SubReg = SubReg;
    return Op;
  }
  static MachineOperand CreateMBB(MachineBasicBlock *MBB) {
    MachineOperand Op(MachineOperand::MO_MachineBasicBlock);
    Op.setMBB(MBB);
    return Op;
  }
  static MachineOperand CreateFI(unsigned Idx) {
    MachineOperand Op(MachineOperand::MO_FrameIndex);
    Op.setIndex(Idx);
    return Op;
  }
  static MachineOperand CreateCPI(unsigned Idx, int Offset) {
    MachineOperand Op(MachineOperand::MO_ConstantPoolIndex);
    Op.setIndex(Idx);
    Op.setOffset(Offset);
    return Op;
  }
  static MachineOperand CreateJTI(unsigned Idx) {
    MachineOperand Op(MachineOperand::MO_JumpTableIndex);
    Op.setIndex(Idx);
    return Op;
  }
  static MachineOperand CreateGA(GlobalValue *GV, int Offset) {
    MachineOperand Op(MachineOperand::MO_GlobalAddress);
    Op.Contents.OffsetedInfo.Val.GV = GV;
    Op.setOffset(Offset);
    return Op;
  }
  static MachineOperand CreateES(const char *SymName, int Offset = 0) {
    MachineOperand Op(MachineOperand::MO_ExternalSymbol);
    Op.Contents.OffsetedInfo.Val.SymbolName = SymName;
    Op.setOffset(Offset);
    return Op;
  }
  const MachineOperand &operator=(const MachineOperand &MO) {
    OpKind   = MO.OpKind;
    IsDef    = MO.IsDef;
    IsImp    = MO.IsImp;
    IsKill   = MO.IsKill;
    IsDead   = MO.IsDead;
    IsEarlyClobber = MO.IsEarlyClobber;
    OverlapsEarlyClobber = MO.OverlapsEarlyClobber;
    SubReg   = MO.SubReg;
    ParentMI = MO.ParentMI;
    Contents = MO.Contents;
    return *this;
  }

  friend class MachineInstr;
  friend class MachineRegisterInfo;
private:
  //===--------------------------------------------------------------------===//
  // Methods for handling register use/def lists.
  //===--------------------------------------------------------------------===//

  /// isOnRegUseList - Return true if this operand is on a register use/def list
  /// or false if not.  This can only be called for register operands that are
  /// part of a machine instruction.
  bool isOnRegUseList() const {
    assert(isRegister() && "Can only add reg operand to use lists");
    return Contents.Reg.Prev != 0;
  }
  
  /// AddRegOperandToRegInfo - Add this register operand to the specified
  /// MachineRegisterInfo.  If it is null, then the next/prev fields should be
  /// explicitly nulled out.
  void AddRegOperandToRegInfo(MachineRegisterInfo *RegInfo);

  void RemoveRegOperandFromRegInfo() {
    assert(isOnRegUseList() && "Reg operand is not on a use list");
    // Unlink this from the doubly linked list of operands.
    MachineOperand *NextOp = Contents.Reg.Next;
    *Contents.Reg.Prev = NextOp; 
    if (NextOp) {
      assert(NextOp->getReg() == getReg() && "Corrupt reg use/def chain!");
      NextOp->Contents.Reg.Prev = Contents.Reg.Prev;
    }
    Contents.Reg.Prev = 0;
    Contents.Reg.Next = 0;
  }
};

inline std::ostream &operator<<(std::ostream &OS, const MachineOperand &MO) {
  MO.print(OS, 0);
  return OS;
}

inline raw_ostream &operator<<(raw_ostream &OS, const MachineOperand& MO) {
  MO.print(OS, 0);
  return OS;
}

} // End llvm namespace

#endif
