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

//************************** System Include Files **************************/

#include <string>
#include <vector>

//*************************** User Include Files ***************************/

#include "llvm/Tools/DataTypes.h"
#include "llvm/Instruction.h"
#include "llvm/Support/Unique.h"
#include "llvm/CodeGen/TargetMachine.h"


//************************* Opaque Declarations ****************************/

class Value;
class InstrTreeNode;
class InstructionNode;
class MachineInstr;
class MachineInstrInfo;
class MachineOperand;


//************************ Exported Data Types *****************************/

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
  friend ostream& operator<<(ostream& os, const MachineOperand& mop);

public:
  enum MachineOperandType {
    MO_Register,
    MO_CCRegister,
    MO_SignExtendedImmed,
    MO_UnextendedImmed,
    MO_PCRelativeDisp,
  };
  
  enum VirtualRegisterType {
    MO_VirtualReg,		// virtual register for *value
    MO_MachineReg		// pre-assigned machine register `regNum'
  };
  
  MachineOperandType machineOperandType;
  
  VirtualRegisterType vregType;
  
  Value*	value;		// BasicBlockVal for a label operand.
				// ConstantVal for a non-address immediate.
				// Virtual register for a register operand.
  
  unsigned int regNum;		// register number for an explicit register
  
  int64_t immedVal;		// constant value for an explicit constant
  
  /*ctor*/		MachineOperand	();
  /*ctor*/		MachineOperand	(MachineOperandType operandType,
					 Value* _val);
  /*copy ctor*/		MachineOperand	(const MachineOperand&);
  /*dtor*/		~MachineOperand	() {}
  
  // These functions are provided so that a vector of operands can be
  // statically allocated and individual ones can be initialized later.
  // 
  void			Initialize	(MachineOperandType operandType,
					 Value* _val);
  void			InitializeConst	(MachineOperandType operandType,
					 int64_t intValue);
  void			InitializeReg	(unsigned int regNum);
};


inline
MachineOperand::MachineOperand()
  : machineOperandType(MO_Register),
    vregType(MO_VirtualReg),
    value(NULL),
    regNum(0),
    immedVal(0)
{}

inline
MachineOperand::MachineOperand(MachineOperandType operandType,
			       Value* _val)
  : machineOperandType(operandType),
    vregType(MO_VirtualReg),
    value(_val),
    regNum(0),
    immedVal(0)
{}

inline
MachineOperand::MachineOperand(const MachineOperand& mo)
  : machineOperandType(mo.machineOperandType),
    vregType(mo.vregType),
    value(mo.value),
    regNum(mo.regNum),
    immedVal(mo.immedVal)
{
}

inline void
MachineOperand::Initialize(MachineOperandType operandType,
			   Value* _val)
{
  machineOperandType = operandType;
  value = _val;
}

inline void
MachineOperand::InitializeConst(MachineOperandType operandType,
				int64_t intValue)
{
  machineOperandType = operandType;
  value = NULL;
  immedVal = intValue;
}

inline void
MachineOperand::InitializeReg(unsigned int _regNum)
{
  machineOperandType = MO_Register;
  vregType = MO_MachineReg;
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
//   The array MachineInstrInfo TargetMachineInstrInfo[] objects
//   (indexed by opCode) provides information about each target instruction.
// 
//   opCodeMask is used to record variants of an instruction.
//   E.g., each branch instruction on SPARC has 2 flags (i.e., 4 variants):
//	ANNUL:		   if 1: Annul delay slot instruction.
//	PREDICT-NOT-TAKEN: if 1: predict branch not taken.
//   Instead of creating 4 different opcodes for BNZ, we create a single
//   opcode and set bits in opCodeMask for each of these flags.
//---------------------------------------------------------------------------

class MachineInstr : public Unique {
private:
  MachineOpCode	opCode;
  OpCodeMask	opCodeMask;		// extra bits for variants of an opcode
  vector<MachineOperand> operands;	// operand 0 is the result
  
public:
  /*ctor*/		MachineInstr	(MachineOpCode _opCode,
					 OpCodeMask    _opCodeMask = 0x0);
  
  /*dtor*/ virtual	~MachineInstr	();
  
  const MachineOpCode	getOpCode	() const;
  
  unsigned int		getNumOperands	() const;
  
  const MachineOperand& getOperand	(unsigned int i) const;
  
  void			dump		(unsigned int indent = 0);
  
public:
  friend ostream& operator<<(ostream& os, const MachineInstr& minstr);

public:
  // Access to set the operands when building the machine instruction
  void			SetMachineOperand(unsigned int i,
			      MachineOperand::MachineOperandType operandType,
			      Value* _val);
  void			SetMachineOperand(unsigned int i,
			      MachineOperand::MachineOperandType operandType,
			      int64_t intValue);
  void			SetMachineOperand(unsigned int i,
					  unsigned int regNum);
};

inline const MachineOpCode
MachineInstr::getOpCode() const
{
  return opCode;
}

inline unsigned int
MachineInstr::getNumOperands() const
{
  assert(operands.size() == TargetMachineInstrInfo[opCode].numOperands);
  return operands.size();
}

inline const MachineOperand&
MachineInstr::getOperand(unsigned int i) const
{
  return operands[i];
}


//---------------------------------------------------------------------------
// class MachineInstructionsForVMInstr
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
    if (tempVec[i]->getValueType() == Value::InstructionVal)
      ((Instruction*) tempVec[i])->dropAllReferences();
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
			     MachineOperand::VirtualRegisterType& getVRegType,
			     unsigned int& getMachineRegNum,
			     int64_t& getImmedValue);

ostream& operator<<(ostream& os, const MachineInstr& minstr);


ostream& operator<<(ostream& os, const MachineOperand& mop);
					 

//**************************************************************************/

#endif
