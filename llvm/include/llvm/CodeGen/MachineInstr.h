//===-- llvm/CodeGen/MachineInstr.h - MachineInstr class ---------*- C++ -*--=//
//
// This file contains the declaration of the MachineInstr class, which is the
// basic representation for all target dependant machine instructions used by
// the back end.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEINSTR_H
#define LLVM_CODEGEN_MACHINEINSTR_H

#include "llvm/Target/MachineInstrInfo.h"
#include "llvm/Annotation.h"
#include <Support/iterator>
#include <Support/hash_set>
class Instruction;
using std::vector;

//---------------------------------------------------------------------------
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
//---------------------------------------------------------------------------


class MachineOperand {
public:
  enum MachineOperandType {
    MO_VirtualRegister,		// virtual register for *value
    MO_MachineRegister,		// pre-assigned machine register `regNum'
    MO_CCRegister,
    MO_SignExtendedImmed,
    MO_UnextendedImmed,
    MO_PCRelativeDisp,
  };
  
private:
  // Bit fields of the flags variable used for different operand properties
  static const char DEFFLAG    = 0x1;  // this is a def of the operand
  static const char DEFUSEFLAG = 0x2;  // this is both a def and a use
  static const char HIFLAG32   = 0x4;  // operand is %hi32(value_or_immedVal)
  static const char LOFLAG32   = 0x8;  // operand is %lo32(value_or_immedVal)
  static const char HIFLAG64   = 0x10; // operand is %hi64(value_or_immedVal)
  static const char LOFLAG64   = 0x20; // operand is %lo64(value_or_immedVal)
  
private:
  MachineOperandType opType;
  
  union {
    Value*	value;		// BasicBlockVal for a label operand.
				// ConstantVal for a non-address immediate.
				// Virtual register for an SSA operand,
				// including hidden operands required for
				// the generated machine code.     
    int64_t immedVal;		// constant value for an explicit constant
  };

  int regNum;	                // register number for an explicit register
                                // will be set for a value after reg allocation
  char flags;                   // see bit field definitions above
  
public:
  /*ctor*/		MachineOperand	();
  /*ctor*/		MachineOperand	(MachineOperandType operandType,
					 Value* _val);
  /*copy ctor*/		MachineOperand	(const MachineOperand&);
  /*dtor*/		~MachineOperand	() {}
  
  // Accessor methods.  Caller is responsible for checking the
  // operand type before invoking the corresponding accessor.
  // 
  inline MachineOperandType getOperandType() const {
    return opType;
  }
  inline Value*		getVRegValue	() const {
    assert(opType == MO_VirtualRegister || opType == MO_CCRegister || 
	   opType == MO_PCRelativeDisp);
    return value;
  }
  inline Value*		getVRegValueOrNull() const {
    return (opType == MO_VirtualRegister || opType == MO_CCRegister || 
            opType == MO_PCRelativeDisp)? value : NULL;
  }
  inline int            getMachineRegNum() const {
    assert(opType == MO_MachineRegister);
    return regNum;
  }
  inline int64_t	getImmedValue	() const {
    assert(opType == MO_SignExtendedImmed || opType == MO_UnextendedImmed);
    return immedVal;
  }
  inline bool		opIsDef		() const {
    return flags & DEFFLAG;
  }
  inline bool		opIsDefAndUse	() const {
    return flags & DEFUSEFLAG;
  }
  inline bool           opHiBits32      () const {
    return flags & HIFLAG32;
  }
  inline bool           opLoBits32      () const {
    return flags & LOFLAG32;
  }
  inline bool           opHiBits64      () const {
    return flags & HIFLAG64;
  }
  inline bool           opLoBits64      () const {
    return flags & LOFLAG64;
  }

  // used to check if a machine register has been allocated to this operand
  inline bool   hasAllocatedReg() const {
    return (regNum >= 0 &&
            (opType == MO_VirtualRegister || opType == MO_CCRegister || 
             opType == MO_MachineRegister));
  }

  // used to get the reg number if when one is allocated
  inline int  getAllocatedRegNum() const {
    assert(opType == MO_VirtualRegister || opType == MO_CCRegister || 
	   opType == MO_MachineRegister);
    return regNum;
  }

  
public:
  friend std::ostream& operator<<(std::ostream& os, const MachineOperand& mop);

private:
  // These functions are provided so that a vector of operands can be
  // statically allocated and individual ones can be initialized later.
  // Give class MachineInstr access to these functions.
  // 
  void			Initialize	(MachineOperandType operandType,
					 Value* _val);
  void			InitializeConst	(MachineOperandType operandType,
					 int64_t intValue);
  void			InitializeReg	(int regNum,
                                         bool isCCReg);

  // Construction methods needed for fine-grain control.
  // These must be accessed via coresponding methods in MachineInstr.
  void markDef()       { flags |= DEFFLAG; }
  void markDefAndUse() { flags |= DEFUSEFLAG; }
  void markHi32()      { flags |= HIFLAG32; }
  void markLo32()      { flags |= LOFLAG32; }
  void markHi64()      { flags |= HIFLAG64; }
  void markLo64()      { flags |= LOFLAG64; }
  
  // Replaces the Value with its corresponding physical register after
  // register allocation is complete
  void setRegForValue(int reg) {
    assert(opType == MO_VirtualRegister || opType == MO_CCRegister || 
	   opType == MO_MachineRegister);
    regNum = reg;
  }
  
  friend class MachineInstr;
};


inline
MachineOperand::MachineOperand()
  : opType(MO_VirtualRegister),
    immedVal(0),
    regNum(-1),
    flags(0)
{}

inline
MachineOperand::MachineOperand(MachineOperandType operandType,
			       Value* _val)
  : opType(operandType),
    immedVal(0),
    regNum(-1),
    flags(0)
{}

inline
MachineOperand::MachineOperand(const MachineOperand& mo)
  : opType(mo.opType),
    flags(mo.flags)
{
  switch(opType) {
  case MO_VirtualRegister:
  case MO_CCRegister:		value = mo.value; break;
  case MO_MachineRegister:	regNum = mo.regNum; break;
  case MO_SignExtendedImmed:
  case MO_UnextendedImmed:
  case MO_PCRelativeDisp:	immedVal = mo.immedVal; break;
  default: assert(0);
  }
}

inline void
MachineOperand::Initialize(MachineOperandType operandType,
			   Value* _val)
{
  opType = operandType;
  value = _val;
  regNum = -1;
  flags = 0;
}

inline void
MachineOperand::InitializeConst(MachineOperandType operandType,
				int64_t intValue)
{
  opType = operandType;
  value = NULL;
  immedVal = intValue;
  regNum = -1;
  flags = 0;
}

inline void
MachineOperand::InitializeReg(int _regNum, bool isCCReg)
{
  opType = isCCReg? MO_CCRegister : MO_MachineRegister;
  value = NULL;
  regNum = (int) _regNum;
  flags = 0;
}


//---------------------------------------------------------------------------
// class MachineInstr 
// 
// Purpose:
//   Representation of each machine instruction.
// 
//   MachineOpCode must be an enum, defined separately for each target.
//   E.g., It is defined in SparcInstructionSelection.h for the SPARC.
// 
//   opCodeMask is used to record variants of an instruction.
//   E.g., each branch instruction on SPARC has 2 flags (i.e., 4 variants):
//	ANNUL:		   if 1: Annul delay slot instruction.
//	PREDICT-NOT-TAKEN: if 1: predict branch not taken.
//   Instead of creating 4 different opcodes for BNZ, we create a single
//   opcode and set bits in opCodeMask for each of these flags.
//
//  There are 2 kinds of operands:
// 
//  (1) Explicit operands of the machine instruction in vector operands[] 
// 
//  (2) "Implicit operands" are values implicitly used or defined by the
//      machine instruction, such as arguments to a CALL, return value of
//      a CALL (if any), and return value of a RETURN.
//---------------------------------------------------------------------------

class MachineInstr :  public Annotable,         // Values are annotable
                      public NonCopyable {      // Disable copy operations
  MachineOpCode    opCode;              // the opcode
  OpCodeMask       opCodeMask;          // extra bits for variants of an opcode
  vector<MachineOperand> operands;      // the operands
  vector<Value*>   implicitRefs;        // values implicitly referenced by this
  vector<bool>     implicitIsDef;       //  machine instruction (eg, call args)
  vector<bool>     implicitIsDefAndUse; //
  hash_set<int>    regsUsed;            // all machine registers used for this
                                        //  instruction, including regs used
                                        //  to save values across the instr.
public:
  /*ctor*/		MachineInstr	(MachineOpCode _opCode,
					 OpCodeMask    _opCodeMask = 0x0);
  /*ctor*/		MachineInstr	(MachineOpCode _opCode,
					 unsigned	numOperands,
					 OpCodeMask    _opCodeMask = 0x0);
  inline           	~MachineInstr	() {}
  const MachineOpCode	getOpCode	() const { return opCode; }

  //
  // Information about explicit operands of the instruction
  // 
  unsigned int		getNumOperands	() const { return operands.size(); }
  
  bool			operandIsDefined(unsigned i) const;
  bool			operandIsDefinedAndUsed(unsigned i) const;
  
  const MachineOperand& getOperand	(unsigned i) const;
        MachineOperand& getOperand	(unsigned i);
  
  //
  // Information about implicit operands of the instruction
  // 
  unsigned             	getNumImplicitRefs() const{return implicitRefs.size();}
  
  bool			implicitRefIsDefined(unsigned i) const;
  bool			implicitRefIsDefinedAndUsed(unsigned i) const;
  
  const Value*          getImplicitRef  (unsigned i) const;
        Value*          getImplicitRef  (unsigned i);
  
  //
  // Information about registers used in this instruction
  // 
  const hash_set<int>&  getRegsUsed    () const { return regsUsed; }
        hash_set<int>&  getRegsUsed    ()       { return regsUsed; }
  
  //
  // Debugging support
  // 
  void			dump		() const;
  friend std::ostream& operator<<       (std::ostream& os,
                                         const MachineInstr& minstr);

  //
  // Define iterators to access the Value operands of the Machine Instruction.
  // begin() and end() are defined to produce these iterators...
  //
  template<class _MI, class _V> class ValOpIterator;
  typedef ValOpIterator<const MachineInstr*,const Value*> const_val_op_iterator;
  typedef ValOpIterator<      MachineInstr*,      Value*> val_op_iterator;


  // Access to set the operands when building the machine instruction
  // 
  void			SetMachineOperandVal(unsigned i,
                                             MachineOperand::MachineOperandType
                                               operandType,
                                             Value* _val,
                                             bool isDef=false,
                                             bool isDefAndUse=false);
  void			SetMachineOperandConst(unsigned i,
                                           MachineOperand::MachineOperandType
                                                 operandType,
                                               int64_t intValue);
  void			SetMachineOperandReg(unsigned i, int regNum, 
                                             bool isDef=false,
                                             bool isDefAndUse=false,
                                             bool isCCReg=false);
  
  void                  addImplicitRef	 (Value* val, 
                                          bool isDef=false,
                                          bool isDefAndUse=false);
  
  void                  setImplicitRef	 (unsigned i,
                                          Value* val, 
                                          bool isDef=false,
                                          bool isDefAndUse=false);

  unsigned              substituteValue  (const Value* oldVal,
                                          Value* newVal,
                                          bool defsOnly = true);

  void                  setOperandHi32   (unsigned i);
  void                  setOperandLo32   (unsigned i);
  void                  setOperandHi64   (unsigned i);
  void                  setOperandLo64   (unsigned i);
  
  
  // Replaces the Value for the operand with its allocated
  // physical register after register allocation is complete.
  // 
  void                  SetRegForOperand(unsigned i, int regNum);
  
  //
  // Iterator to enumerate machine operands.
  // 
  template<class MITy, class VTy>
  class ValOpIterator : public forward_iterator<VTy, ptrdiff_t> {
    unsigned i;
    MITy MI;
    
    inline void skipToNextVal() {
      while (i < MI->getNumOperands() &&
             !((MI->getOperand(i).getOperandType() == MachineOperand::MO_VirtualRegister ||
                MI->getOperand(i).getOperandType() == MachineOperand::MO_CCRegister)
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

    inline bool isDef()       const { return MI->getOperand(i).opIsDef(); } 
    inline bool isDefAndUse() const { return MI->getOperand(i).opIsDefAndUse();}

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


inline MachineOperand&
MachineInstr::getOperand(unsigned int i)
{
  assert(i < operands.size() && "getOperand() out of range!");
  return operands[i];
}

inline const MachineOperand&
MachineInstr::getOperand(unsigned int i) const
{
  assert(i < operands.size() && "getOperand() out of range!");
  return operands[i];
}

inline bool
MachineInstr::operandIsDefined(unsigned int i) const
{
  return getOperand(i).opIsDef();
}

inline bool
MachineInstr::operandIsDefinedAndUsed(unsigned int i) const
{
  return getOperand(i).opIsDefAndUse();
}

inline bool
MachineInstr::implicitRefIsDefined(unsigned int i) const
{
  assert(i < implicitIsDef.size() && "operand out of range!");
  return implicitIsDef[i];
}

inline bool
MachineInstr::implicitRefIsDefinedAndUsed(unsigned int i) const
{
  assert(i < implicitIsDefAndUse.size() && "operand out of range!");
  return implicitIsDefAndUse[i];
}

inline const Value*
MachineInstr::getImplicitRef(unsigned int i) const
{
  assert(i < implicitRefs.size() && "getImplicitRef() out of range!");
  return implicitRefs[i];
}

inline Value*
MachineInstr::getImplicitRef(unsigned int i)
{
  assert(i < implicitRefs.size() && "getImplicitRef() out of range!");
  return implicitRefs[i];
}

inline void
MachineInstr::addImplicitRef(Value* val, 
                             bool isDef,
                             bool isDefAndUse)
{
  implicitRefs.push_back(val);
  implicitIsDef.push_back(isDef);
  implicitIsDefAndUse.push_back(isDefAndUse);
}

inline void
MachineInstr::setImplicitRef(unsigned int i,
                             Value* val, 
                             bool isDef,
                             bool isDefAndUse)
{
  assert(i < implicitRefs.size() && "setImplicitRef() out of range!");
  implicitRefs[i] = val;
  implicitIsDef[i] = isDef;
  implicitIsDefAndUse[i] = isDefAndUse;
}

inline void
MachineInstr::setOperandHi32(unsigned i)
{
  operands[i].markHi32();
}

inline void
MachineInstr::setOperandLo32(unsigned i)
{
  operands[i].markLo32();
}

inline void
MachineInstr::setOperandHi64(unsigned i)
{
  operands[i].markHi64();
}

inline void
MachineInstr::setOperandLo64(unsigned i)
{
  operands[i].markLo64();
}


//---------------------------------------------------------------------------
// Debugging Support
//---------------------------------------------------------------------------

std::ostream& operator<<    (std::ostream& os, const MachineInstr& minstr);

std::ostream& operator<<    (std::ostream& os, const MachineOperand& mop);
					 
void	PrintMachineInstructions(const Function *F);

#endif
