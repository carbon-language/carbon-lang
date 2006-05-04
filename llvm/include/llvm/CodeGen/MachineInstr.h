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

typedef short MachineOpCode;

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
    MO_VirtualRegister,         // virtual register for *value
    MO_SignExtendedImmed,
    MO_UnextendedImmed,
    MO_MachineBasicBlock,       // MachineBasicBlock reference
    MO_FrameIndex,              // Abstract Stack Frame Index
    MO_ConstantPoolIndex,       // Address of indexed Constant in Constant Pool
    MO_JumpTableIndex,          // Address of indexed Jump Table for switch
    MO_ExternalSymbol,          // Name of external global symbol
    MO_GlobalAddress            // Address of a global value
  };

private:
  union {
    Value*  value;      // BasicBlockVal for a label operand.
                        // ConstantVal for a non-address immediate.
                        // Virtual register for an SSA operand,
                        //   including hidden operands required for
                        //   the generated machine code.
                        // LLVM global for MO_GlobalAddress.

    int64_t immedVal;   // Constant value for an explicit constant

    MachineBasicBlock *MBB;     // For MO_MachineBasicBlock type
    const char *SymbolName;     // For MO_ExternalSymbol type
  } contents;

  char flags;                   // see bit field definitions above
  MachineOperandType opType:8;  // Pack into 8 bits efficiently after flags.
  union {
    int regNum;                 // register number for an explicit register
                                // will be set for a value after reg allocation

    int offset;                 // Offset to address of global or external, only
                                // valid for MO_GlobalAddress, MO_ExternalSym
                                // and MO_ConstantPoolIndex
  } extra;

  void zeroContents () {
    memset (&contents, 0, sizeof (contents));
    memset (&extra, 0, sizeof (extra));
  }

  MachineOperand(int64_t ImmVal, MachineOperandType OpTy, int Offset = 0)
    : flags(0), opType(OpTy) {
    zeroContents ();
    contents.immedVal = ImmVal;
    extra.offset = Offset;
  }

  MachineOperand(int Reg, MachineOperandType OpTy, UseType UseTy)
    : flags(UseTy), opType(OpTy) {
    zeroContents ();
    extra.regNum = Reg;
  }

  MachineOperand(GlobalValue *V, int Offset = 0)
    : flags(MachineOperand::Use), opType(MachineOperand::MO_GlobalAddress) {
    zeroContents ();
    contents.value = (Value*)V;
    extra.offset = Offset;
  }

  MachineOperand(MachineBasicBlock *mbb)
    : flags(0), opType(MO_MachineBasicBlock) {
    zeroContents ();
    contents.MBB = mbb;
  }

  MachineOperand(const char *SymName, int Offset)
    : flags(0), opType(MO_ExternalSymbol) {
    zeroContents ();
    contents.SymbolName = SymName;
    extra.offset = Offset;
  }

public:
  MachineOperand(const MachineOperand &M)
    : flags(M.flags), opType(M.opType) {
    zeroContents ();
    contents = M.contents;
    extra = M.extra;
  }

  ~MachineOperand() {}

  const MachineOperand &operator=(const MachineOperand &MO) {
    contents = MO.contents;
    flags    = MO.flags;
    opType   = MO.opType;
    extra    = MO.extra;
    return *this;
  }

  /// getType - Returns the MachineOperandType for this operand.
  ///
  MachineOperandType getType() const { return opType; }

  /// getUseType - Returns the MachineOperandUseType of this operand.
  ///
  UseType getUseType() const { return UseType(flags & (USEFLAG|DEFFLAG)); }

  /// isRegister - Return true if this operand is a register operand.
  ///
  bool isRegister() const {
    return opType == MO_VirtualRegister;
  }

  /// Accessors that tell you what kind of MachineOperand you're looking at.
  ///
  bool isMachineBasicBlock() const { return opType == MO_MachineBasicBlock; }
  bool isImmediate() const {
    return opType == MO_SignExtendedImmed || opType == MO_UnextendedImmed;
  }
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
    return (GlobalValue*)contents.value;
  }
  int getOffset() const {
    assert((isGlobalAddress() || isExternalSymbol() || isConstantPoolIndex()) &&
        "Wrong MachineOperand accessor");
    return extra.offset;
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

  /// hasAllocatedReg - Returns true iff a machine register has been
  /// allocated to this operand.
  ///
  bool hasAllocatedReg() const {
    return extra.regNum >= 0 && opType == MO_VirtualRegister;
  }

  /// getReg - Returns the register number. It is a runtime error to call this
  /// if a register is not allocated.
  ///
  unsigned getReg() const {
    assert(hasAllocatedReg());
    return extra.regNum;
  }

  /// MachineOperand mutators.
  ///
  void setReg(unsigned Reg) {
    assert(hasAllocatedReg() && "This operand cannot have a register number!");
    extra.regNum = Reg;
  }

  void setImmedValue(int immVal) {
    assert(isImmediate() && "Wrong MachineOperand mutator");
    contents.immedVal = immVal;
  }

  void setOffset(int Offset) {
    assert((isGlobalAddress() || isExternalSymbol() || isConstantPoolIndex() ||
            isJumpTableIndex()) &&
        "Wrong MachineOperand accessor");
    extra.offset = Offset;
  }

  friend std::ostream& operator<<(std::ostream& os, const MachineOperand& mop);

  friend class MachineInstr;
};


//===----------------------------------------------------------------------===//
// class MachineInstr
//
// Purpose:
//   Representation of each machine instruction.
//
//   MachineOpCode must be an enum, defined separately for each target.
//   E.g., It is defined in SparcInstructionSelection.h for the SPARC.
//
//  There are 2 kinds of operands:
//
//  (1) Explicit operands of the machine instruction in vector operands[]
//
//  (2) "Implicit operands" are values implicitly used or defined by the
//      machine instruction, such as arguments to a CALL, return value of
//      a CALL (if any), and return value of a RETURN.
//===----------------------------------------------------------------------===//

class MachineInstr {
  short Opcode;                         // the opcode
  std::vector<MachineOperand> operands; // the operands
  MachineInstr* prev, *next;            // links for our intrusive list
  MachineBasicBlock* parent;            // pointer to the owning basic block

  // OperandComplete - Return true if it's illegal to add a new operand
  bool OperandsComplete() const;

  //Constructor used by clone() method
  MachineInstr(const MachineInstr&);

  void operator=(const MachineInstr&); // DO NOT IMPLEMENT

  // Intrusive list support
  //
  friend struct ilist_traits<MachineInstr>;

public:
  /// MachineInstr ctor - This constructor only does a _reserve_ of the
  /// operands, not a resize for them.  It is expected that if you use this that
  /// you call add* methods below to fill up the operands, instead of the Set
  /// methods.  Eventually, the "resizing" ctors will be phased out.
  ///
  MachineInstr(short Opcode, unsigned numOperands, bool XX, bool YY);

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
  unsigned getNumOperands() const { return operands.size(); }

  const MachineOperand& getOperand(unsigned i) const {
    assert(i < getNumOperands() && "getOperand() out of range!");
    return operands[i];
  }
  MachineOperand& getOperand(unsigned i) {
    assert(i < getNumOperands() && "getOperand() out of range!");
    return operands[i];
  }


  /// clone - Create a copy of 'this' instruction that is identical in
  /// all ways except the the instruction has no parent, prev, or next.
  MachineInstr* clone() const;
  
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
  // Accessors to add operands when building up machine instructions
  //

  /// addRegOperand - Add a symbolic virtual register reference...
  ///
  void addRegOperand(int reg, bool isDef) {
    assert(!OperandsComplete() &&
           "Trying to add an operand to a machine instr that is already done!");
    operands.push_back(
      MachineOperand(reg, MachineOperand::MO_VirtualRegister,
                     isDef ? MachineOperand::Def : MachineOperand::Use));
  }

  /// addRegOperand - Add a symbolic virtual register reference...
  ///
  void addRegOperand(int reg,
                     MachineOperand::UseType UTy = MachineOperand::Use) {
    assert(!OperandsComplete() &&
           "Trying to add an operand to a machine instr that is already done!");
    operands.push_back(
      MachineOperand(reg, MachineOperand::MO_VirtualRegister, UTy));
  }

  /// addZeroExtImmOperand - Add a zero extended constant argument to the
  /// machine instruction.
  ///
  void addZeroExtImmOperand(int intValue) {
    assert(!OperandsComplete() &&
           "Trying to add an operand to a machine instr that is already done!");
    operands.push_back(
      MachineOperand(intValue, MachineOperand::MO_UnextendedImmed));
  }

  /// addZeroExtImm64Operand - Add a zero extended 64-bit constant argument
  /// to the machine instruction.
  ///
  void addZeroExtImm64Operand(uint64_t intValue) {
    assert(!OperandsComplete() &&
           "Trying to add an operand to a machine instr that is already done!");
    operands.push_back(
      MachineOperand(intValue, MachineOperand::MO_UnextendedImmed));
  }

  /// addSignExtImmOperand - Add a zero extended constant argument to the
  /// machine instruction.
  ///
  void addSignExtImmOperand(int intValue) {
    assert(!OperandsComplete() &&
           "Trying to add an operand to a machine instr that is already done!");
    operands.push_back(
      MachineOperand(intValue, MachineOperand::MO_SignExtendedImmed));
  }

  void addMachineBasicBlockOperand(MachineBasicBlock *MBB) {
    assert(!OperandsComplete() &&
           "Trying to add an operand to a machine instr that is already done!");
    operands.push_back(MachineOperand(MBB));
  }

  /// addFrameIndexOperand - Add an abstract frame index to the instruction
  ///
  void addFrameIndexOperand(unsigned Idx) {
    assert(!OperandsComplete() &&
           "Trying to add an operand to a machine instr that is already done!");
    operands.push_back(MachineOperand(Idx, MachineOperand::MO_FrameIndex));
  }

  /// addConstantPoolndexOperand - Add a constant pool object index to the
  /// instruction.
  ///
  void addConstantPoolIndexOperand(unsigned I, int Offset=0) {
    assert(!OperandsComplete() &&
           "Trying to add an operand to a machine instr that is already done!");
    operands.push_back(MachineOperand(I, MachineOperand::MO_ConstantPoolIndex));
  }

  /// addJumpTableIndexOperand - Add a jump table object index to the
  /// instruction.
  ///
  void addJumpTableIndexOperand(unsigned I) {
    assert(!OperandsComplete() &&
           "Trying to add an operand to a machine instr that is already done!");
    operands.push_back(MachineOperand(I, MachineOperand::MO_JumpTableIndex));
  }
  
  void addGlobalAddressOperand(GlobalValue *GV, int Offset) {
    assert(!OperandsComplete() &&
           "Trying to add an operand to a machine instr that is already done!");
    operands.push_back(MachineOperand(GV, Offset));
  }

  /// addExternalSymbolOperand - Add an external symbol operand to this instr
  ///
  void addExternalSymbolOperand(const char *SymName) {
    operands.push_back(MachineOperand(SymName, 0));
  }

  //===--------------------------------------------------------------------===//
  // Accessors used to modify instructions in place.
  //
  // FIXME: Move this stuff to MachineOperand itself!

  /// setOpcode - Replace the opcode of the current instruction with a new one.
  ///
  void setOpcode(unsigned Op) { Opcode = Op; }

  /// RemoveOperand - Erase an operand  from an instruction, leaving it with one
  /// fewer operand than it started with.
  ///
  void RemoveOperand(unsigned i) {
    operands.erase(operands.begin()+i);
  }

  // Access to set the operands when building the machine instruction
  //
  void SetMachineOperandVal(unsigned i,
                            MachineOperand::MachineOperandType operandType,
                            Value* V);

  void SetMachineOperandConst(unsigned i,
                              MachineOperand::MachineOperandType operandType,
                              int intValue);

  void SetMachineOperandReg(unsigned i, int regNum);
};

//===----------------------------------------------------------------------===//
// Debugging Support

std::ostream& operator<<(std::ostream &OS, const MachineInstr &MI);
std::ostream& operator<<(std::ostream &OS, const MachineOperand &MO);

} // End llvm namespace

#endif
