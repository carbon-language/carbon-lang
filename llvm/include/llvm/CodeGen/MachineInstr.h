// $Id$ -*-c++-*-
//***************************************************************************
// File:
//	MachineInstr.h
// 
// Purpose:
//	
// 
// Strategy:
// 
// History:
//	7/2/01	 -  Vikram Adve  -  Created
//**************************************************************************/

#ifndef LLVM_CODEGEN_MACHINEINSTR_H
#define LLVM_CODEGEN_MACHINEINSTR_H

#include <iterator>
#include "llvm/CodeGen/InstrForest.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/NonCopyable.h"
#include "llvm/CodeGen/TargetMachine.h"

template<class _MI, class _V> class ValOpIterator;


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
  MachineOperandType opType;
  
  union {
    Value*	value;		// BasicBlockVal for a label operand.
				// ConstantVal for a non-address immediate.
				// Virtual register for an SSA operand,
				// including hidden operands required for
				// the generated machine code.
    
    unsigned int regNum;	// register number for an explicit register
  
    int64_t immedVal;		// constant value for an explicit constant
  };

  bool isDef;                   // is this a defition for the value
                                // made public for faster access
  
public:
  /*ctor*/		MachineOperand	();
  /*ctor*/		MachineOperand	(MachineOperandType operandType,
					 Value* _val);
  /*copy ctor*/		MachineOperand	(const MachineOperand&);
  /*dtor*/		~MachineOperand	() {}
  
  // Accessor methods.  Caller is responsible for checking the
  // operand type before invoking the corresponding accessor.
  // 
  inline MachineOperandType getOperandType	() const {
    return opType;
  }
  inline Value*		getVRegValue	() const {
    assert(opType == MO_VirtualRegister || opType == MO_CCRegister || 
	   opType == MO_PCRelativeDisp);
    return value;
  }
  inline unsigned int		getMachineRegNum() const {
    assert(opType == MO_MachineRegister);
    return regNum;
  }
  inline int64_t	getImmedValue	() const {
    assert(opType >= MO_SignExtendedImmed || opType <= MO_PCRelativeDisp);
    return immedVal;
  }
  inline bool		opIsDef		() const {
    return isDef;
  }
  
public:
  friend ostream& operator<<(ostream& os, const MachineOperand& mop);

  
private:
  // These functions are provided so that a vector of operands can be
  // statically allocated and individual ones can be initialized later.
  // Give class MachineInstr gets access to these functions.
  // 
  void			Initialize	(MachineOperandType operandType,
					 Value* _val);
  void			InitializeConst	(MachineOperandType operandType,
					 int64_t intValue);
  void			InitializeReg	(unsigned int regNum);

  friend class MachineInstr;
  friend class ValOpIterator<const MachineInstr, const Value>;
  friend class ValOpIterator<      MachineInstr,       Value>;


public:

 
};


inline
MachineOperand::MachineOperand()
  : opType(MO_VirtualRegister),
    value(NULL),
    regNum(0),
    immedVal(0),
    isDef(false)
{}

inline
MachineOperand::MachineOperand(MachineOperandType operandType,
			       Value* _val)
  : opType(operandType),
    value(_val),
    regNum(0),
    immedVal(0),
    isDef(false)
{}

inline
MachineOperand::MachineOperand(const MachineOperand& mo)
  : opType(mo.opType),
    isDef(false)
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
}

inline void
MachineOperand::InitializeConst(MachineOperandType operandType,
				int64_t intValue)
{
  opType = operandType;
  value = NULL;
  immedVal = intValue;
}

inline void
MachineOperand::InitializeReg(unsigned int _regNum)
{
  opType = MO_MachineRegister;
  value = NULL;
  regNum = _regNum;
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
//---------------------------------------------------------------------------

class MachineInstr : public NonCopyable {
private:
  MachineOpCode	opCode;
  OpCodeMask	opCodeMask;		// extra bits for variants of an opcode
  vector<MachineOperand> operands;
  
public:
  typedef ValOpIterator<const MachineInstr, const Value> val_op_const_iterator;
  typedef ValOpIterator<const MachineInstr,       Value> val_op_iterator;
  
public:
  /*ctor*/		MachineInstr	(MachineOpCode _opCode,
					 OpCodeMask    _opCodeMask = 0x0);
  /*ctor*/		MachineInstr	(MachineOpCode _opCode,
					 unsigned	numOperands,
					 OpCodeMask    _opCodeMask = 0x0);
  inline           	~MachineInstr	() {}
  
  const MachineOpCode	getOpCode	() const;
  
  unsigned int		getNumOperands	() const;
  
  const MachineOperand& getOperand	(unsigned int i) const;
        MachineOperand& getOperand	(unsigned int i);
  
  bool			operandIsDefined(unsigned int i) const;
  
  void			dump		(unsigned int indent = 0) const;




  
public:
  friend ostream& operator<<(ostream& os, const MachineInstr& minstr);
  friend val_op_const_iterator;
  friend val_op_iterator;

public:
  // Access to set the operands when building the machine instruction
  void			SetMachineOperand(unsigned int i,
			      MachineOperand::MachineOperandType operandType,
			      Value* _val, bool isDef=false);
  void			SetMachineOperand(unsigned int i,
			      MachineOperand::MachineOperandType operandType,
			      int64_t intValue, bool isDef=false);
  void			SetMachineOperand(unsigned int i,
					  unsigned int regNum, 
					  bool isDef=false);
};

inline const MachineOpCode
MachineInstr::getOpCode() const
{
  return opCode;
}

inline unsigned int
MachineInstr::getNumOperands() const
{
  return operands.size();
}

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


template<class _MI, class _V>
class ValOpIterator : public std::forward_iterator<_V, ptrdiff_t> {
private:
  unsigned int i;
  int resultPos;
  _MI* minstr;
  
  inline void	skipToNextVal() {
    while (i < minstr->getNumOperands() &&
	   ! ((minstr->operands[i].opType == MachineOperand::MO_VirtualRegister
	       || minstr->operands[i].opType == MachineOperand::MO_CCRegister)
	      && minstr->operands[i].value != NULL))
      ++i;
  }
  
public:
  typedef ValOpIterator<_MI, _V> _Self;
  
  inline ValOpIterator(_MI* _minstr) : i(0), minstr(_minstr) {
    resultPos = TargetInstrDescriptors[minstr->opCode].resultPos;
    skipToNextVal();
  };
  
  inline _V*	operator*()  const { return minstr->getOperand(i).getVRegValue();}
  inline _V*	operator->() const { return operator*(); }
  //  inline bool	isDef	()   const { return (((int) i) == resultPos); }
  
  inline bool	isDef	()   const { return minstr->getOperand(i).isDef; } 
  inline bool	done	()   const { return (i == minstr->getNumOperands()); }
  
  inline _Self& operator++()	   { i++; skipToNextVal(); return *this; }
  inline _Self  operator++(int)	   { _Self tmp = *this; ++*this; return tmp; }
};


//---------------------------------------------------------------------------
// class MachineCodeForVMInstr
// 
// Purpose:
//   Representation of the sequence of machine instructions created
//   for a single VM instruction.  Additionally records any temporary 
//   "values" used as intermediate values in this sequence.
//   Note that such values should be treated as pure SSA values with
//   no interpretation of their operands (i.e., as a TmpInstruction object
//   which actually represents such a value).
// 
//---------------------------------------------------------------------------

class MachineCodeForVMInstr: public vector<MachineInstr*>
{
private:
  vector<Value*> tempVec;
  
public:
  /*ctor*/	MachineCodeForVMInstr	()	{}
  /*ctor*/	~MachineCodeForVMInstr	();
  
  const vector<Value*>&
		getTempValues		() const { return tempVec; }
  
  void		addTempValue		(Value* val)
						 { tempVec.push_back(val); }

  // dropAllReferences() - This function drops all references within
  // temporary (hidden) instructions created in implementing the original
  // VM intruction.  This ensures there are no remaining "uses" within
  // these hidden instructions, before the values of a method are freed.
  //
  // Make this inline because it has to be called from class Instruction
  // and inlining it avoids a serious circurality in link order.
  inline void dropAllReferences() {
    for (unsigned i=0, N=tempVec.size(); i < N; i++)
      if (Instruction *I = tempVec[i]->castInstruction())
	I->dropAllReferences();
  }
};

inline
MachineCodeForVMInstr::~MachineCodeForVMInstr()
{
  // Free the Value objects created to hold intermediate values
  for (unsigned i=0, N=tempVec.size(); i < N; i++)
    delete tempVec[i];
  
  // Free the MachineInstr objects allocated, if any.
  for (unsigned i=0, N=this->size(); i < N; i++)
    delete (*this)[i];
}


//---------------------------------------------------------------------------
// class MachineCodeForBasicBlock
// 
// Purpose:
//   Representation of the sequence of machine instructions created
//   for a basic block.
//---------------------------------------------------------------------------


class MachineCodeForBasicBlock: public vector<MachineInstr*> {
public:
  typedef vector<MachineInstr*>::iterator iterator;
  typedef vector<const MachineInstr*>::const_iterator const_iterator;
};


//---------------------------------------------------------------------------
// Target-independent utility routines for creating machine instructions
//---------------------------------------------------------------------------


//------------------------------------------------------------------------ 
// Function Set2OperandsFromInstr
// Function Set3OperandsFromInstr
// 
// For the common case of 2- and 3-operand arithmetic/logical instructions,
// set the m/c instr. operands directly from the VM instruction's operands.
// Check whether the first or second operand is 0 and can use a dedicated
// "0" register.
// Check whether the second operand should use an immediate field or register.
// (First and third operands are never immediates for such instructions.)
// 
// Arguments:
// canDiscardResult: Specifies that the result operand can be discarded
//		     by using the dedicated "0"
// 
// op1position, op2position and resultPosition: Specify in which position
//		     in the machine instruction the 3 operands (arg1, arg2
//		     and result) should go.
// 
// RETURN VALUE: unsigned int flags, where
//	flags & 0x01	=> operand 1 is constant and needs a register
//	flags & 0x02	=> operand 2 is constant and needs a register
//------------------------------------------------------------------------ 

void		Set2OperandsFromInstr	(MachineInstr* minstr,
					 InstructionNode* vmInstrNode,
					 const TargetMachine& targetMachine,
					 bool canDiscardResult = false,
					 int op1Position = 0,
					 int resultPosition = 1);

void		Set3OperandsFromInstr	(MachineInstr* minstr,
					 InstructionNode* vmInstrNode,
					 const TargetMachine& targetMachine,
					 bool canDiscardResult = false,
					 int op1Position = 0,
					 int op2Position = 1,
					 int resultPosition = 2);

MachineOperand::MachineOperandType
		ChooseRegOrImmed(Value* val,
				 MachineOpCode opCode,
				 const TargetMachine& targetMachine,
				 bool canUseImmed,
				 unsigned int& getMachineRegNum,
				 int64_t& getImmedValue);


ostream& operator<<(ostream& os, const MachineInstr& minstr);


ostream& operator<<(ostream& os, const MachineOperand& mop);
					 

void	PrintMachineInstructions	(const Method *method);


//**************************************************************************/

#endif
