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

#include "Support/iterator"
#include <string>
#include <vector>
#include <cassert>

namespace llvm {

class Value;
class Function;
class MachineBasicBlock;
class TargetMachine;
class GlobalValue;

template <typename T> class ilist_traits;
template <typename T> class ilist;

typedef short MachineOpCode;

//===----------------------------------------------------------------------===//
// class MachineOperand 
// 
// Purpose:
//   Representation of each machine instruction operand.
//   This class is designed so that you can allocate a vector of operands
//   first and initialize each one later.
//
//   E.g, for this VM instruction:
//		ptr = alloca type, numElements
//   we generate 2 machine instructions on the SPARC:
// 
//		mul Constant, Numelements -> Reg
//		add %sp, Reg -> Ptr
// 
//   Each instruction has 3 operands, listed above.  Of those:
//   -	Reg, NumElements, and Ptr are of operand type MO_Register.
//   -	Constant is of operand type MO_SignExtendedImmed on the SPARC.
//	
//   For the register operands, the virtual register type is as follows:
//	
//   -  Reg will be of virtual register type MO_MInstrVirtualReg.  The field
//	MachineInstr* minstr will point to the instruction that computes reg.
// 
//   -	%sp will be of virtual register type MO_MachineReg.
//	The field regNum identifies the machine register.
// 
//   -	NumElements will be of virtual register type MO_VirtualReg.
//	The field Value* value identifies the value.
// 
//   -	Ptr will also be of virtual register type MO_VirtualReg.
//	Again, the field Value* value identifies the value.
// 
//===----------------------------------------------------------------------===//

struct MachineOperand {
private:
  // Bit fields of the flags variable used for different operand properties
  enum {
    DEFFLAG     = 0x01,       // this is a def of the operand
    USEFLAG     = 0x02,       // this is a use of the operand
    HIFLAG32    = 0x04,       // operand is %hi32(value_or_immedVal)
    LOFLAG32    = 0x08,       // operand is %lo32(value_or_immedVal)
    HIFLAG64    = 0x10,       // operand is %hi64(value_or_immedVal)
    LOFLAG64    = 0x20,       // operand is %lo64(value_or_immedVal)
    PCRELATIVE  = 0x40,       // Operand is relative to PC, not a global address
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
    MO_VirtualRegister,		// virtual register for *value
    MO_MachineRegister,		// pre-assigned machine register `regNum'
    MO_CCRegister,
    MO_SignExtendedImmed,
    MO_UnextendedImmed,
    MO_PCRelativeDisp,
    MO_MachineBasicBlock,       // MachineBasicBlock reference
    MO_FrameIndex,              // Abstract Stack Frame Index
    MO_ConstantPoolIndex,       // Address of indexed Constant in Constant Pool
    MO_ExternalSymbol,          // Name of external global symbol
    MO_GlobalAddress,           // Address of a global value
  };
  
private:
  union {
    Value*  value;      // BasicBlockVal for a label operand.
                        // ConstantVal for a non-address immediate.
                        // Virtual register for an SSA operand,
                        //   including hidden operands required for
                        //   the generated machine code.     
                        // LLVM global for MO_GlobalAddress.

    int immedVal;		// Constant value for an explicit constant

    MachineBasicBlock *MBB;     // For MO_MachineBasicBlock type
    std::string *SymbolName;    // For MO_ExternalSymbol type
  } contents;

  char flags;                   // see bit field definitions above
  MachineOperandType opType:8;  // Pack into 8 bits efficiently after flags.
  int regNum;	                // register number for an explicit register
                                // will be set for a value after reg allocation
private:
  void zeroContents () { 
    memset (&contents, 0, sizeof (contents));
  }

  MachineOperand(int ImmVal = 0, MachineOperandType OpTy = MO_VirtualRegister)
    : flags(0), opType(OpTy), regNum(-1) {
    zeroContents ();
    contents.immedVal = ImmVal;
  }

  MachineOperand(int Reg, MachineOperandType OpTy, UseType UseTy)
    : flags(UseTy), opType(OpTy), regNum(Reg) {
    zeroContents ();
  }

  MachineOperand(Value *V, MachineOperandType OpTy, UseType UseTy,
		 bool isPCRelative = false)
    : flags(UseTy | (isPCRelative?PCRELATIVE:0)), opType(OpTy), regNum(-1) {
    zeroContents ();
    contents.value = V;
  }

  MachineOperand(MachineBasicBlock *mbb)
    : flags(0), opType(MO_MachineBasicBlock), regNum(-1) {
    zeroContents ();
    contents.MBB = mbb;
  }

  MachineOperand(const std::string &SymName, bool isPCRelative)
    : flags(isPCRelative?PCRELATIVE:0), opType(MO_ExternalSymbol), regNum(-1) {
    zeroContents ();
    contents.SymbolName = new std::string (SymName);
  }

public:
  MachineOperand(const MachineOperand &M)
    : flags(M.flags), opType(M.opType), regNum(M.regNum) {
    zeroContents ();
    contents = M.contents;
    if (isExternalSymbol())
      contents.SymbolName = new std::string(M.getSymbolName());
  }

 
  ~MachineOperand() {
    if (isExternalSymbol())
      delete contents.SymbolName;
  }
  
  const MachineOperand &operator=(const MachineOperand &MO) {
    if (isExternalSymbol())             // if old operand had a symbol name,
      delete contents.SymbolName;       // release old memory
    contents = MO.contents;
    flags    = MO.flags;
    opType   = MO.opType;
    regNum   = MO.regNum;
    if (isExternalSymbol())
      contents.SymbolName = new std::string(MO.getSymbolName());
    return *this;
  }

  /// getType - Returns the MachineOperandType for this operand.
  /// 
  MachineOperandType getType() const { return opType; }

  /// getUseType - Returns the MachineOperandUseType of this operand.
  ///
  UseType getUseType() const { return UseType(flags & (USEFLAG|DEFFLAG)); }

  /// isPCRelative - This returns the value of the PCRELATIVE flag, which
  /// indicates whether this operand should be emitted as a PC relative value
  /// instead of a global address.  This is used for operands of the forms:
  /// MachineBasicBlock, GlobalAddress, ExternalSymbol
  ///
  bool isPCRelative() const { return (flags & PCRELATIVE) != 0; }

  /// isRegister - Return true if this operand is a register operand.  The X86
  /// backend currently can't decide whether to use MO_MR or MO_VR to represent
  /// them, so we accept both.
  ///
  /// Note: The sparc backend should not use this method.
  ///
  bool isRegister() const {
    return opType == MO_MachineRegister || opType == MO_VirtualRegister;
  }

  /// Accessors that tell you what kind of MachineOperand you're looking at.
  ///
  bool isMachineBasicBlock() const { return opType == MO_MachineBasicBlock; }
  bool isPCRelativeDisp() const { return opType == MO_PCRelativeDisp; }
  bool isImmediate() const {
    return opType == MO_SignExtendedImmed || opType == MO_UnextendedImmed;
  }
  bool isFrameIndex() const { return opType == MO_FrameIndex; }
  bool isConstantPoolIndex() const { return opType == MO_ConstantPoolIndex; }
  bool isGlobalAddress() const { return opType == MO_GlobalAddress; }
  bool isExternalSymbol() const { return opType == MO_ExternalSymbol; }

  /// getVRegValueOrNull - Get the Value* out of a MachineOperand if it
  /// has one. This is deprecated and only used by the SPARC v9 backend.
  ///
  Value* getVRegValueOrNull() const {
    return (opType == MO_VirtualRegister || opType == MO_CCRegister || 
            isPCRelativeDisp()) ? contents.value : NULL;
  }

  /// MachineOperand accessors that only work on certain types of
  /// MachineOperand...
  ///
  Value* getVRegValue() const {
    assert ((opType == MO_VirtualRegister || opType == MO_CCRegister
             || isPCRelativeDisp()) && "Wrong MachineOperand accessor");
    return contents.value;
  }
  int getMachineRegNum() const {
    assert(opType == MO_MachineRegister && "Wrong MachineOperand accessor");
    return regNum;
  }
  int getImmedValue() const {
    assert(isImmediate() && "Wrong MachineOperand accessor");
    return contents.immedVal;
  }
  MachineBasicBlock *getMachineBasicBlock() const {
    assert(isMachineBasicBlock() && "Wrong MachineOperand accessor");
    return contents.MBB;
  }
  int getFrameIndex() const {
    assert(isFrameIndex() && "Wrong MachineOperand accessor");
    return contents.immedVal;
  }
  unsigned getConstantPoolIndex() const {
    assert(isConstantPoolIndex() && "Wrong MachineOperand accessor");
    return contents.immedVal;
  }
  GlobalValue *getGlobal() const {
    assert(isGlobalAddress() && "Wrong MachineOperand accessor");
    return (GlobalValue*)contents.value;
  }
  const std::string &getSymbolName() const {
    assert(isExternalSymbol() && "Wrong MachineOperand accessor");
    return *contents.SymbolName;
  }

  /// MachineOperand methods for testing that work on any kind of
  /// MachineOperand...
  ///
  bool            isUse           () const { return flags & USEFLAG; }
  MachineOperand& setUse          ()       { flags |= USEFLAG; return *this; }
  bool            isDef           () const { return flags & DEFFLAG; }
  MachineOperand& setDef          ()       { flags |= DEFFLAG; return *this; }
  bool            isHiBits32      () const { return flags & HIFLAG32; }
  bool            isLoBits32      () const { return flags & LOFLAG32; }
  bool            isHiBits64      () const { return flags & HIFLAG64; }
  bool            isLoBits64      () const { return flags & LOFLAG64; }

  /// hasAllocatedReg - Returns true iff a machine register has been
  /// allocated to this operand.
  ///
  bool hasAllocatedReg() const {
    return (regNum >= 0 &&
            (opType == MO_VirtualRegister || opType == MO_CCRegister || 
             opType == MO_MachineRegister));
  }

  /// getReg - Returns the register number. It is a runtime error to call this
  /// if a register is not allocated.
  ///
  unsigned getReg() const {
    assert(hasAllocatedReg());
    return regNum;
  }

  /// MachineOperand mutators...
  ///
  void setReg(unsigned Reg) {
    // This method's comment used to say: 'TODO: get rid of this duplicate
    // code.' It's not clear where the duplication is.
    assert(hasAllocatedReg() && "This operand cannot have a register number!");
    regNum = Reg;
  }    
  void setImmedValue(int immVal) {
    assert(isImmediate() && "Wrong MachineOperand mutator");
    contents.immedVal = immVal;
  }

  friend std::ostream& operator<<(std::ostream& os, const MachineOperand& mop);

private:
  /// markHi32, markLo32, etc. - These methods must be accessed via
  /// corresponding methods in MachineInstr.  These methods are deprecated
  /// and only used by the SPARC v9 back-end.
  ///
  void markHi32()      { flags |= HIFLAG32; }
  void markLo32()      { flags |= LOFLAG32; }
  void markHi64()      { flags |= HIFLAG64; }
  void markLo64()      { flags |= LOFLAG64; }
  
  /// setRegForValue - Replaces the Value with its corresponding physical
  /// register after register allocation is complete. This is deprecated
  /// and only used by the SPARC v9 back-end.
  ///
  void setRegForValue(int reg) {
    assert(opType == MO_VirtualRegister || opType == MO_CCRegister || 
	   opType == MO_MachineRegister);
    regNum = reg;
  }
  
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
  unsigned char numImplicitRefs;        // number of implicit operands
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
  friend class ilist_traits<MachineInstr>;

public:
  MachineInstr(short Opcode, unsigned numOperands);

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
  unsigned getNumOperands() const { return operands.size() - numImplicitRefs; }
  
  const MachineOperand& getOperand(unsigned i) const {
    assert(i < getNumOperands() && "getOperand() out of range!");
    return operands[i];
  }
  MachineOperand& getOperand(unsigned i) {
    assert(i < getNumOperands() && "getOperand() out of range!");
    return operands[i];
  }

  //
  // Access to explicit or implicit operands of the instruction
  // This returns the i'th entry in the operand vector.
  // That represents the i'th explicit operand or the (i-N)'th implicit operand,
  // depending on whether i < N or i >= N.
  // 
  const MachineOperand& getExplOrImplOperand(unsigned i) const {
    assert(i < operands.size() && "getExplOrImplOperand() out of range!");
    return (i < getNumOperands()? getOperand(i)
                                : getImplicitOp(i - getNumOperands()));
  }

  //
  // Access to implicit operands of the instruction
  // 
  unsigned getNumImplicitRefs() const{ return numImplicitRefs; }
  
  MachineOperand& getImplicitOp(unsigned i) {
    assert(i < numImplicitRefs && "implicit ref# out of range!");
    return operands[i + operands.size() - numImplicitRefs];
  }
  const MachineOperand& getImplicitOp(unsigned i) const {
    assert(i < numImplicitRefs && "implicit ref# out of range!");
    return operands[i + operands.size() - numImplicitRefs];
  }

  Value* getImplicitRef(unsigned i) {
    return getImplicitOp(i).getVRegValue();
  }
  const Value* getImplicitRef(unsigned i) const {
    return getImplicitOp(i).getVRegValue();
  }

  void addImplicitRef(Value* V, bool isDef = false, bool isDefAndUse = false) {
    ++numImplicitRefs;
    addRegOperand(V, isDef, isDefAndUse);
  }
  void setImplicitRef(unsigned i, Value* V) {
    assert(i < getNumImplicitRefs() && "setImplicitRef() out of range!");
    SetMachineOperandVal(i + getNumOperands(),
                         MachineOperand::MO_VirtualRegister, V);
  }

  /// clone - Create a copy of 'this' instruction that is identical in
  /// all ways except the the instruction has no parent, prev, or next.
  MachineInstr* clone();

  //
  // Debugging support
  //
  void print(std::ostream &OS, const TargetMachine &TM) const;
  void dump() const;
  friend std::ostream& operator<<(std::ostream& os, const MachineInstr& minstr);

  //
  // Define iterators to access the Value operands of the Machine Instruction.
  // Note that these iterators only enumerate the explicit operands.
  // begin() and end() are defined to produce these iterators...
  //
  template<class _MI, class _V> class ValOpIterator;
  typedef ValOpIterator<const MachineInstr*,const Value*> const_val_op_iterator;
  typedef ValOpIterator<      MachineInstr*,      Value*> val_op_iterator;


  //===--------------------------------------------------------------------===//
  // Accessors to add operands when building up machine instructions
  //

  /// addRegOperand - Add a MO_VirtualRegister operand to the end of the
  /// operands list...
  ///
  void addRegOperand(Value *V, bool isDef, bool isDefAndUse=false) {
    assert(!OperandsComplete() &&
           "Trying to add an operand to a machine instr that is already done!");
    operands.push_back(
      MachineOperand(V, MachineOperand::MO_VirtualRegister,
                     !isDef ? MachineOperand::Use :
                     (isDefAndUse ? MachineOperand::UseAndDef :
                      MachineOperand::Def)));
  }

  void addRegOperand(Value *V,
                     MachineOperand::UseType UTy = MachineOperand::Use,
                     bool isPCRelative = false) {
    assert(!OperandsComplete() &&
           "Trying to add an operand to a machine instr that is already done!");
    operands.push_back(MachineOperand(V, MachineOperand::MO_VirtualRegister,
                                      UTy, isPCRelative));
  }

  void addCCRegOperand(Value *V,
                       MachineOperand::UseType UTy = MachineOperand::Use) {
    assert(!OperandsComplete() &&
           "Trying to add an operand to a machine instr that is already done!");
    operands.push_back(MachineOperand(V, MachineOperand::MO_CCRegister, UTy,
                                      false));
  }


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

  /// addPCDispOperand - Add a PC relative displacement operand to the MI
  ///
  void addPCDispOperand(Value *V) {
    assert(!OperandsComplete() &&
           "Trying to add an operand to a machine instr that is already done!");
    operands.push_back(
      MachineOperand(V, MachineOperand::MO_PCRelativeDisp,MachineOperand::Use));
  }

  /// addMachineRegOperand - Add a virtual register operand to this MachineInstr
  ///
  void addMachineRegOperand(int reg, bool isDef) {
    assert(!OperandsComplete() &&
           "Trying to add an operand to a machine instr that is already done!");
    operands.push_back(
      MachineOperand(reg, MachineOperand::MO_MachineRegister,
                     isDef ? MachineOperand::Def : MachineOperand::Use));
  }

  /// addMachineRegOperand - Add a virtual register operand to this MachineInstr
  ///
  void addMachineRegOperand(int reg,
                            MachineOperand::UseType UTy = MachineOperand::Use) {
    assert(!OperandsComplete() &&
           "Trying to add an operand to a machine instr that is already done!");
    operands.push_back(
      MachineOperand(reg, MachineOperand::MO_MachineRegister, UTy));
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
  void addConstantPoolIndexOperand(unsigned I) {
    assert(!OperandsComplete() &&
           "Trying to add an operand to a machine instr that is already done!");
    operands.push_back(MachineOperand(I, MachineOperand::MO_ConstantPoolIndex));
  }

  void addGlobalAddressOperand(GlobalValue *GV, bool isPCRelative) {
    assert(!OperandsComplete() &&
           "Trying to add an operand to a machine instr that is already done!");
    operands.push_back(
      MachineOperand((Value*)GV, MachineOperand::MO_GlobalAddress,
                     MachineOperand::Use, isPCRelative));
  }

  /// addExternalSymbolOperand - Add an external symbol operand to this instr
  ///
  void addExternalSymbolOperand(const std::string &SymName, bool isPCRelative) {
    operands.push_back(MachineOperand(SymName, isPCRelative));
  }

  //===--------------------------------------------------------------------===//
  // Accessors used to modify instructions in place.
  //
  // FIXME: Move this stuff to MachineOperand itself!

  /// replace - Support to rewrite a machine instruction in place: for now,
  /// simply replace() and then set new operands with Set.*Operand methods
  /// below.
  /// 
  void replace(short Opcode, unsigned numOperands);

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


  unsigned substituteValue(const Value* oldVal, Value* newVal,
                           bool defsOnly, bool notDefsAndUses,
                           bool& someArgsWereIgnored);

  void setOperandHi32(unsigned i) { operands[i].markHi32(); }
  void setOperandLo32(unsigned i) { operands[i].markLo32(); }
  void setOperandHi64(unsigned i) { operands[i].markHi64(); }
  void setOperandLo64(unsigned i) { operands[i].markLo64(); }
  
  
  // SetRegForOperand -
  // SetRegForImplicitRef -
  // Mark an explicit or implicit operand with its allocated physical register.
  // 
  void SetRegForOperand(unsigned i, int regNum);
  void SetRegForImplicitRef(unsigned i, int regNum);

  //
  // Iterator to enumerate machine operands.
  // 
  template<class MITy, class VTy>
  class ValOpIterator : public forward_iterator<VTy, ptrdiff_t> {
    unsigned i;
    MITy MI;
    
    void skipToNextVal() {
      while (i < MI->getNumOperands() &&
             !( (MI->getOperand(i).getType() == MachineOperand::MO_VirtualRegister ||
                 MI->getOperand(i).getType() == MachineOperand::MO_CCRegister)
                && MI->getOperand(i).getVRegValue() != 0))
        ++i;
    }
  
    inline ValOpIterator(MITy mi, unsigned I) : i(I), MI(mi) {
      skipToNextVal();
    }
  
  public:
    typedef ValOpIterator<MITy, VTy> _Self;
    
    inline VTy operator*() const {
      return MI->getOperand(i).getVRegValue();
    }

    const MachineOperand &getMachineOperand() const { return MI->getOperand(i);}
          MachineOperand &getMachineOperand()       { return MI->getOperand(i);}

    inline VTy operator->() const { return operator*(); }

    inline bool isUse()   const { return MI->getOperand(i).isUse(); } 
    inline bool isDef()   const { return MI->getOperand(i).isDef(); } 

    inline _Self& operator++() { i++; skipToNextVal(); return *this; }
    inline _Self  operator++(int) { _Self tmp = *this; ++*this; return tmp; }

    inline bool operator==(const _Self &y) const { 
      return i == y.i;
    }
    inline bool operator!=(const _Self &y) const { 
      return !operator==(y);
    }

    static _Self begin(MITy MI) {
      return _Self(MI, 0);
    }
    static _Self end(MITy MI) {
      return _Self(MI, MI->getNumOperands());
    }
  };

  // define begin() and end()
  val_op_iterator begin() { return val_op_iterator::begin(this); }
  val_op_iterator end()   { return val_op_iterator::end(this); }

  const_val_op_iterator begin() const {
    return const_val_op_iterator::begin(this);
  }
  const_val_op_iterator end() const {
    return const_val_op_iterator::end(this);
  }
};

//===----------------------------------------------------------------------===//
// Debugging Support

std::ostream& operator<<(std::ostream &OS, const MachineInstr &MI);
std::ostream& operator<<(std::ostream &OS, const MachineOperand &MO);
void PrintMachineInstructions(const Function *F);

} // End llvm namespace

#endif
