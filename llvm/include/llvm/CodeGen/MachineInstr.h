//===-- llvm/CodeGen/MachineInstr.h - MachineInstr class --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the MachineInstr class, which is the
// basic representation for all target dependent machine instructions used by
// the back end.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEINSTR_H
#define LLVM_CODEGEN_MACHINEINSTR_H

#include "llvm/ADT/iterator"
#include "llvm/Support/DataTypes.h"
#include <vector>
#include <cassert>

namespace llvm {

class Value;
class Function;
class MachineBasicBlock;
class TargetMachine;
class GlobalValue;

template <typename T> struct ilist_traits;
template <typename T> struct ilist;

//===----------------------------------------------------------------------===//
// class MachineOperand
//
//   Representation of each machine instruction operand.
//
struct MachineOperand {
private:
  // Bit fields of the flags variable used for different operand properties
  enum {
    DEFFLAG     = 0x01,       // this is a def of the operand
    USEFLAG     = 0x02,       // this is a use of the operand
  };

public:
  // UseType - This enum describes how the machine operand is used by
  // the instruction. Note that the MachineInstr/Operator class
  // currently uses bool arguments to represent this information
  // instead of an enum.  Eventually this should change over to use
  // this _easier to read_ representation instead.
  //
  enum UseType {
    Use = USEFLAG,        /// only read
    Def = DEFFLAG,        /// only written
    UseAndDef = Use | Def /// read AND written
  };

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
    int64_t immedVal;         // For MO_Immediate and MO_*Index.
  } contents;

  char flags;                   // see bit field definitions above
  MachineOperandType opType:8;  // Pack into 8 bits efficiently after flags.
  
  /// offset - Offset to address of global or external, only valid for
  /// MO_GlobalAddress, MO_ExternalSym and MO_ConstantPoolIndex
  int offset;

  MachineOperand() {}
public:
  MachineOperand(const MachineOperand &M) {
    *this = M;
  }

  ~MachineOperand() {}

  const MachineOperand &operator=(const MachineOperand &MO) {
    contents = MO.contents;
    flags    = MO.flags;
    opType   = MO.opType;
    offset   = MO.offset;
    return *this;
  }

  /// getType - Returns the MachineOperandType for this operand.
  ///
  MachineOperandType getType() const { return opType; }

  /// getUseType - Returns the MachineOperandUseType of this operand.
  ///
  UseType getUseType() const { return UseType(flags & (USEFLAG|DEFFLAG)); }

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

  int64_t getImmedValue() const {
    assert(isImmediate() && "Wrong MachineOperand accessor");
    return contents.immedVal;
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
    return (int)contents.immedVal;
  }
  unsigned getConstantPoolIndex() const {
    assert(isConstantPoolIndex() && "Wrong MachineOperand accessor");
    return (unsigned)contents.immedVal;
  }
  unsigned getJumpTableIndex() const {
    assert(isJumpTableIndex() && "Wrong MachineOperand accessor");
    return (unsigned)contents.immedVal;
  }
  GlobalValue *getGlobal() const {
    assert(isGlobalAddress() && "Wrong MachineOperand accessor");
    return contents.GV;
  }
  int getOffset() const {
    assert((isGlobalAddress() || isExternalSymbol() || isConstantPoolIndex()) &&
        "Wrong MachineOperand accessor");
    return offset;
  }
  const char *getSymbolName() const {
    assert(isExternalSymbol() && "Wrong MachineOperand accessor");
    return contents.SymbolName;
  }

  /// MachineOperand methods for testing that work on any kind of
  /// MachineOperand...
  ///
  bool            isUse           () const { return flags & USEFLAG; }
  MachineOperand& setUse          ()       { flags |= USEFLAG; return *this; }
  bool            isDef           () const { return flags & DEFFLAG; }
  MachineOperand& setDef          ()       { flags |= DEFFLAG; return *this; }

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

  void setImmedValue(int64_t immVal) {
    assert(isImmediate() && "Wrong MachineOperand mutator");
    contents.immedVal = immVal;
  }

  void setOffset(int Offset) {
    assert((isGlobalAddress() || isExternalSymbol() || isConstantPoolIndex() ||
            isJumpTableIndex()) &&
        "Wrong MachineOperand accessor");
    offset = Offset;
  }
  
  /// ChangeToImmediate - Replace this operand with a new immediate operand of
  /// the specified value.  If an operand is known to be an immediate already,
  /// the setImmedValue method should be used.
  void ChangeToImmediate(int64_t ImmVal) {
    opType = MO_Immediate;
    contents.immedVal = ImmVal;
  }

  /// ChangeToRegister - Replace this operand with a new register operand of
  /// the specified value.  If an operand is known to be an register already,
  /// the setReg method should be used.
  void ChangeToRegister(unsigned Reg) {
    opType = MO_Register;
    contents.RegNo = Reg;
  }

  friend std::ostream& operator<<(std::ostream& os, const MachineOperand& mop);

  friend class MachineInstr;
};


//===----------------------------------------------------------------------===//
/// MachineInstr - Representation of each machine instruction.
///
class MachineInstr {
  short Opcode;                         // the opcode
  std::vector<MachineOperand> Operands; // the operands
  MachineInstr* prev, *next;            // links for our intrusive list
  MachineBasicBlock* parent;            // pointer to the owning basic block

  // OperandComplete - Return true if it's illegal to add a new operand
  bool OperandsComplete() const;

  MachineInstr(const MachineInstr&);
  void operator=(const MachineInstr&); // DO NOT IMPLEMENT

  // Intrusive list support
  //
  friend struct ilist_traits<MachineInstr>;

public:
  /// MachineInstr ctor - This constructor reserve's space for numOperand
  /// operands.
  MachineInstr(short Opcode, unsigned numOperands);

  /// MachineInstr ctor - Work exactly the same as the ctor above, except that
  /// the MachineInstr is created and added to the end of the specified basic
  /// block.
  ///
  MachineInstr(MachineBasicBlock *MBB, short Opcode, unsigned numOps);

  ~MachineInstr();

  const MachineBasicBlock* getParent() const { return parent; }
  MachineBasicBlock* getParent() { return parent; }

  /// getOpcode - Returns the opcode of this MachineInstr.
  ///
  const int getOpcode() const { return Opcode; }

  /// Access to explicit operands of the instruction.
  ///
  unsigned getNumOperands() const { return Operands.size(); }

  const MachineOperand& getOperand(unsigned i) const {
    assert(i < getNumOperands() && "getOperand() out of range!");
    return Operands[i];
  }
  MachineOperand& getOperand(unsigned i) {
    assert(i < getNumOperands() && "getOperand() out of range!");
    return Operands[i];
  }


  /// clone - Create a copy of 'this' instruction that is identical in
  /// all ways except the the instruction has no parent, prev, or next.
  MachineInstr* clone() const { return new MachineInstr(*this); }
  
  /// removeFromParent - This method unlinks 'this' from the containing basic
  /// block, and returns it, but does not delete it.
  MachineInstr *removeFromParent();
  
  /// eraseFromParent - This method unlinks 'this' from the containing basic
  /// block and deletes it.
  void eraseFromParent() {
    delete removeFromParent();
  }

  //
  // Debugging support
  //
  void print(std::ostream &OS, const TargetMachine *TM) const;
  void dump() const;
  friend std::ostream& operator<<(std::ostream& os, const MachineInstr& minstr);

  //===--------------------------------------------------------------------===//
  // Accessors to add operands when building up machine instructions.
  //

  /// addRegOperand - Add a register operand.
  ///
  void addRegOperand(unsigned Reg,
                     MachineOperand::UseType UTy = MachineOperand::Use) {
    MachineOperand &Op = AddNewOperand();
    Op.opType = MachineOperand::MO_Register;
    Op.flags = UTy;
    Op.contents.RegNo = Reg;
    Op.offset = 0;
  }

  /// addImmOperand - Add a zero extended constant argument to the
  /// machine instruction.
  ///
  void addImmOperand(int64_t Val) {
    MachineOperand &Op = AddNewOperand();
    Op.opType = MachineOperand::MO_Immediate;
    Op.flags = 0;
    Op.contents.immedVal = Val;
    Op.offset = 0;
  }

  void addMachineBasicBlockOperand(MachineBasicBlock *MBB) {
    MachineOperand &Op = AddNewOperand();
    Op.opType = MachineOperand::MO_MachineBasicBlock;
    Op.flags = 0;
    Op.contents.MBB = MBB;
    Op.offset = 0;
  }

  /// addFrameIndexOperand - Add an abstract frame index to the instruction
  ///
  void addFrameIndexOperand(unsigned Idx) {
    MachineOperand &Op = AddNewOperand();
    Op.opType = MachineOperand::MO_FrameIndex;
    Op.flags = 0;
    Op.contents.immedVal = Idx;
    Op.offset = 0;
  }

  /// addConstantPoolndexOperand - Add a constant pool object index to the
  /// instruction.
  ///
  void addConstantPoolIndexOperand(unsigned Idx, int Offset) {
    MachineOperand &Op = AddNewOperand();
    Op.opType = MachineOperand::MO_ConstantPoolIndex;
    Op.flags = 0;
    Op.contents.immedVal = Idx;
    Op.offset = Offset;
  }

  /// addJumpTableIndexOperand - Add a jump table object index to the
  /// instruction.
  ///
  void addJumpTableIndexOperand(unsigned Idx) {
    MachineOperand &Op = AddNewOperand();
    Op.opType = MachineOperand::MO_JumpTableIndex;
    Op.flags = 0;
    Op.contents.immedVal = Idx;
    Op.offset = 0;
  }
  
  void addGlobalAddressOperand(GlobalValue *GV, int Offset) {
    MachineOperand &Op = AddNewOperand();
    Op.opType = MachineOperand::MO_GlobalAddress;
    Op.flags = 0;
    Op.contents.GV = GV;
    Op.offset = Offset;
  }

  /// addExternalSymbolOperand - Add an external symbol operand to this instr
  ///
  void addExternalSymbolOperand(const char *SymName) {
    MachineOperand &Op = AddNewOperand();
    Op.opType = MachineOperand::MO_ExternalSymbol;
    Op.flags = 0;
    Op.contents.SymbolName = SymName;
    Op.offset = 0;
  }

  //===--------------------------------------------------------------------===//
  // Accessors used to modify instructions in place.
  //

  /// setOpcode - Replace the opcode of the current instruction with a new one.
  ///
  void setOpcode(unsigned Op) { Opcode = Op; }

  /// RemoveOperand - Erase an operand  from an instruction, leaving it with one
  /// fewer operand than it started with.
  ///
  void RemoveOperand(unsigned i) {
    Operands.erase(Operands.begin()+i);
  }
private:
  MachineOperand &AddNewOperand() {
    assert(!OperandsComplete() &&
           "Trying to add an operand to a machine instr that is already done!");
    Operands.push_back(MachineOperand());
    return Operands.back();
  }
};

//===----------------------------------------------------------------------===//
// Debugging Support

std::ostream& operator<<(std::ostream &OS, const MachineInstr &MI);
std::ostream& operator<<(std::ostream &OS, const MachineOperand &MO);

} // End llvm namespace

#endif
