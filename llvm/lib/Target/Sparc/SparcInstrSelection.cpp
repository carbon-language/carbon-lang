// $Id$
//***************************************************************************
// File:
//	SparcInstrSelection.cpp
// 
// Purpose:
//      BURS instruction selection for SPARC V9 architecture.      
//	
// History:
//	7/02/01	 -  Vikram Adve  -  Created
//**************************************************************************/

#include "SparcInternals.h"
#include "SparcInstrSelectionSupport.h"
#include "SparcRegClassInfo.h"
#include "llvm/CodeGen/InstrSelectionSupport.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/InstrForest.h"
#include "llvm/CodeGen/InstrSelection.h"
#include "llvm/CodeGen/MachineCodeForMethod.h"
#include "llvm/CodeGen/MachineCodeForInstruction.h"
#include "llvm/DerivedTypes.h"
#include "llvm/iTerminators.h"
#include "llvm/iMemory.h"
#include "llvm/iOther.h"
#include "llvm/BasicBlock.h"
#include "llvm/Method.h"
#include "llvm/ConstantVals.h"
#include "Support/MathExtras.h"
#include <math.h>
using std::vector;

//************************* Forward Declarations ***************************/


static void SetMemOperands_Internal     (vector<MachineInstr*>& mvec,
                                         vector<MachineInstr*>::iterator mvecI,
                                         const InstructionNode* vmInstrNode,
                                         Value* ptrVal,
                                         const std::vector<Value*>& idxVec,
                                         const TargetMachine& target);


//************************ Internal Functions ******************************/


static inline MachineOpCode 
ChooseBprInstruction(const InstructionNode* instrNode)
{
  MachineOpCode opCode;
  
  Instruction* setCCInstr =
    ((InstructionNode*) instrNode->leftChild())->getInstruction();
  
  switch(setCCInstr->getOpcode())
    {
    case Instruction::SetEQ: opCode = BRZ;   break;
    case Instruction::SetNE: opCode = BRNZ;  break;
    case Instruction::SetLE: opCode = BRLEZ; break;
    case Instruction::SetGE: opCode = BRGEZ; break;
    case Instruction::SetLT: opCode = BRLZ;  break;
    case Instruction::SetGT: opCode = BRGZ;  break;
    default:
      assert(0 && "Unrecognized VM instruction!");
      opCode = INVALID_OPCODE;
      break; 
    }
  
  return opCode;
}


static inline MachineOpCode 
ChooseBpccInstruction(const InstructionNode* instrNode,
                      const BinaryOperator* setCCInstr)
{
  MachineOpCode opCode = INVALID_OPCODE;
  
  bool isSigned = setCCInstr->getOperand(0)->getType()->isSigned();
  
  if (isSigned)
    {
      switch(setCCInstr->getOpcode())
        {
        case Instruction::SetEQ: opCode = BE;  break;
        case Instruction::SetNE: opCode = BNE; break;
        case Instruction::SetLE: opCode = BLE; break;
        case Instruction::SetGE: opCode = BGE; break;
        case Instruction::SetLT: opCode = BL;  break;
        case Instruction::SetGT: opCode = BG;  break;
        default:
          assert(0 && "Unrecognized VM instruction!");
          break; 
        }
    }
  else
    {
      switch(setCCInstr->getOpcode())
        {
        case Instruction::SetEQ: opCode = BE;   break;
        case Instruction::SetNE: opCode = BNE;  break;
        case Instruction::SetLE: opCode = BLEU; break;
        case Instruction::SetGE: opCode = BCC;  break;
        case Instruction::SetLT: opCode = BCS;  break;
        case Instruction::SetGT: opCode = BGU;  break;
        default:
          assert(0 && "Unrecognized VM instruction!");
          break; 
        }
    }
  
  return opCode;
}

static inline MachineOpCode 
ChooseBFpccInstruction(const InstructionNode* instrNode,
                       const BinaryOperator* setCCInstr)
{
  MachineOpCode opCode = INVALID_OPCODE;
  
  switch(setCCInstr->getOpcode())
    {
    case Instruction::SetEQ: opCode = FBE;  break;
    case Instruction::SetNE: opCode = FBNE; break;
    case Instruction::SetLE: opCode = FBLE; break;
    case Instruction::SetGE: opCode = FBGE; break;
    case Instruction::SetLT: opCode = FBL;  break;
    case Instruction::SetGT: opCode = FBG;  break;
    default:
      assert(0 && "Unrecognized VM instruction!");
      break; 
    }
  
  return opCode;
}


// Create a unique TmpInstruction for a boolean value,
// representing the CC register used by a branch on that value.
// For now, hack this using a little static cache of TmpInstructions.
// Eventually the entire BURG instruction selection should be put
// into a separate class that can hold such information.
// The static cache is not too bad because the memory for these
// TmpInstructions will be freed along with the rest of the Method anyway.
// 
static TmpInstruction*
GetTmpForCC(Value* boolVal, const Method* method, const Type* ccType)
{
  typedef std::hash_map<const Value*, TmpInstruction*> BoolTmpCache;
  static BoolTmpCache boolToTmpCache;     // Map boolVal -> TmpInstruction*
  static const Method* lastMethod = NULL; // Use to flush cache between methods
  
  assert(boolVal->getType() == Type::BoolTy && "Weird but ok! Delete assert");
  
  if (lastMethod != method)
    {
      lastMethod = method;
      boolToTmpCache.clear();
    }
  
  // Look for tmpI and create a new one otherwise.  The new value is
  // directly written to map using the ref returned by operator[].
  TmpInstruction*& tmpI = boolToTmpCache[boolVal];
  if (tmpI == NULL)
    tmpI = new TmpInstruction(ccType, boolVal);
  
  return tmpI;
}


static inline MachineOpCode 
ChooseBccInstruction(const InstructionNode* instrNode,
                     bool& isFPBranch)
{
  InstructionNode* setCCNode = (InstructionNode*) instrNode->leftChild();
  BinaryOperator* setCCInstr = (BinaryOperator*) setCCNode->getInstruction();
  const Type* setCCType = setCCInstr->getOperand(0)->getType();
  
  isFPBranch = (setCCType == Type::FloatTy || setCCType == Type::DoubleTy); 
  
  if (isFPBranch) 
    return ChooseBFpccInstruction(instrNode, setCCInstr);
  else
    return ChooseBpccInstruction(instrNode, setCCInstr);
}


static inline MachineOpCode 
ChooseMovFpccInstruction(const InstructionNode* instrNode)
{
  MachineOpCode opCode = INVALID_OPCODE;
  
  switch(instrNode->getInstruction()->getOpcode())
    {
    case Instruction::SetEQ: opCode = MOVFE;  break;
    case Instruction::SetNE: opCode = MOVFNE; break;
    case Instruction::SetLE: opCode = MOVFLE; break;
    case Instruction::SetGE: opCode = MOVFGE; break;
    case Instruction::SetLT: opCode = MOVFL;  break;
    case Instruction::SetGT: opCode = MOVFG;  break;
    default:
      assert(0 && "Unrecognized VM instruction!");
      break; 
    }
  
  return opCode;
}


// Assumes that SUBcc v1, v2 -> v3 has been executed.
// In most cases, we want to clear v3 and then follow it by instruction
// MOVcc 1 -> v3.
// Set mustClearReg=false if v3 need not be cleared before conditional move.
// Set valueToMove=0 if we want to conditionally move 0 instead of 1
//                      (i.e., we want to test inverse of a condition)
// (The latter two cases do not seem to arise because SetNE needs nothing.)
// 
static MachineOpCode
ChooseMovpccAfterSub(const InstructionNode* instrNode,
                     bool& mustClearReg,
                     int& valueToMove)
{
  MachineOpCode opCode = INVALID_OPCODE;
  mustClearReg = true;
  valueToMove = 1;
  
  switch(instrNode->getInstruction()->getOpcode())
    {
    case Instruction::SetEQ: opCode = MOVE;  break;
    case Instruction::SetLE: opCode = MOVLE; break;
    case Instruction::SetGE: opCode = MOVGE; break;
    case Instruction::SetLT: opCode = MOVL;  break;
    case Instruction::SetGT: opCode = MOVG;  break;
    case Instruction::SetNE: assert(0 && "No move required!"); break;
    default:		     assert(0 && "Unrecognized VM instr!"); break; 
    }
  
  return opCode;
}

static inline MachineOpCode
ChooseConvertToFloatInstr(const InstructionNode* instrNode,
                          const Type* opType)
{
  MachineOpCode opCode = INVALID_OPCODE;
  
  switch(instrNode->getOpLabel())
    {
    case ToFloatTy: 
      if (opType == Type::SByteTy || opType == Type::ShortTy || opType == Type::IntTy)
        opCode = FITOS;
      else if (opType == Type::LongTy)
        opCode = FXTOS;
      else if (opType == Type::DoubleTy)
        opCode = FDTOS;
      else if (opType == Type::FloatTy)
        ;
      else
        assert(0 && "Cannot convert this type to FLOAT on SPARC");
      break;
      
    case ToDoubleTy: 
      // This is usually used in conjunction with CreateCodeToCopyIntToFloat().
      // Both functions should treat the integer as a 32-bit value for types
      // of 4 bytes or less, and as a 64-bit value otherwise.
      if (opType == Type::SByteTy || opType == Type::ShortTy || opType == Type::IntTy)
        opCode = FITOD;
      else if (opType == Type::LongTy)
        opCode = FXTOD;
      else if (opType == Type::FloatTy)
        opCode = FSTOD;
      else if (opType == Type::DoubleTy)
        ;
      else
        assert(0 && "Cannot convert this type to DOUBLE on SPARC");
      break;
      
    default:
      break;
    }
  
  return opCode;
}

static inline MachineOpCode 
ChooseConvertToIntInstr(const InstructionNode* instrNode,
                        const Type* opType)
{
  MachineOpCode opCode = INVALID_OPCODE;;
  
  int instrType = (int) instrNode->getOpLabel();
  
  if (instrType == ToSByteTy || instrType == ToShortTy || instrType == ToIntTy)
    {
      switch (opType->getPrimitiveID())
        {
        case Type::FloatTyID:   opCode = FSTOI; break;
        case Type::DoubleTyID:  opCode = FDTOI; break;
        default:
          assert(0 && "Non-numeric non-bool type cannot be converted to Int");
          break;
        }
    }
  else if (instrType == ToLongTy)
    {
      switch (opType->getPrimitiveID())
        {
        case Type::FloatTyID:   opCode = FSTOX; break;
        case Type::DoubleTyID:  opCode = FDTOX; break;
        default:
          assert(0 && "Non-numeric non-bool type cannot be converted to Long");
          break;
        }
    }
  else
      assert(0 && "Should not get here, Mo!");
  
  return opCode;
}


static inline MachineOpCode 
ChooseAddInstructionByType(const Type* resultType)
{
  MachineOpCode opCode = INVALID_OPCODE;
  
  if (resultType->isIntegral() ||
      resultType->isPointerType() ||
      resultType->isLabelType() ||
      isa<MethodType>(resultType) ||
      resultType == Type::BoolTy)
    {
      opCode = ADD;
    }
  else
    switch(resultType->getPrimitiveID())
      {
      case Type::FloatTyID:  opCode = FADDS; break;
      case Type::DoubleTyID: opCode = FADDD; break;
      default: assert(0 && "Invalid type for ADD instruction"); break; 
      }
  
  return opCode;
}


static inline MachineOpCode 
ChooseAddInstruction(const InstructionNode* instrNode)
{
  return ChooseAddInstructionByType(instrNode->getInstruction()->getType());
}


static inline MachineInstr* 
CreateMovFloatInstruction(const InstructionNode* instrNode,
                          const Type* resultType)
{
  MachineInstr* minstr = new MachineInstr((resultType == Type::FloatTy)
                                          ? FMOVS : FMOVD);
  minstr->SetMachineOperandVal(0, MachineOperand::MO_VirtualRegister,
                               instrNode->leftChild()->getValue());
  minstr->SetMachineOperandVal(1, MachineOperand::MO_VirtualRegister,
                               instrNode->getValue());
  return minstr;
}

static inline MachineInstr* 
CreateAddConstInstruction(const InstructionNode* instrNode)
{
  MachineInstr* minstr = NULL;
  
  Value* constOp = ((InstrTreeNode*) instrNode->rightChild())->getValue();
  assert(isa<Constant>(constOp));
  
  // Cases worth optimizing are:
  // (1) Add with 0 for float or double: use an FMOV of appropriate type,
  //	 instead of an FADD (1 vs 3 cycles).  There is no integer MOV.
  // 
  const Type* resultType = instrNode->getInstruction()->getType();
  
  if (resultType == Type::FloatTy ||
      resultType == Type::DoubleTy)
    {
      double dval = cast<ConstantFP>(constOp)->getValue();
      if (dval == 0.0)
        minstr = CreateMovFloatInstruction(instrNode, resultType);
    }
  
  return minstr;
}


static inline MachineOpCode 
ChooseSubInstructionByType(const Type* resultType)
{
  MachineOpCode opCode = INVALID_OPCODE;
  
  if (resultType->isIntegral() ||
      resultType->isPointerType())
    {
      opCode = SUB;
    }
  else
    switch(resultType->getPrimitiveID())
      {
      case Type::FloatTyID:  opCode = FSUBS; break;
      case Type::DoubleTyID: opCode = FSUBD; break;
      default: assert(0 && "Invalid type for SUB instruction"); break; 
      }
  
  return opCode;
}


static inline MachineInstr* 
CreateSubConstInstruction(const InstructionNode* instrNode)
{
  MachineInstr* minstr = NULL;
  
  Value* constOp = ((InstrTreeNode*) instrNode->rightChild())->getValue();
  assert(isa<Constant>(constOp));
  
  // Cases worth optimizing are:
  // (1) Sub with 0 for float or double: use an FMOV of appropriate type,
  //	 instead of an FSUB (1 vs 3 cycles).  There is no integer MOV.
  // 
  const Type* resultType = instrNode->getInstruction()->getType();
  
  if (resultType == Type::FloatTy ||
      resultType == Type::DoubleTy)
    {
      double dval = cast<ConstantFP>(constOp)->getValue();
      if (dval == 0.0)
        minstr = CreateMovFloatInstruction(instrNode, resultType);
    }
  
  return minstr;
}


static inline MachineOpCode 
ChooseFcmpInstruction(const InstructionNode* instrNode)
{
  MachineOpCode opCode = INVALID_OPCODE;
  
  Value* operand = ((InstrTreeNode*) instrNode->leftChild())->getValue();
  switch(operand->getType()->getPrimitiveID()) {
  case Type::FloatTyID:  opCode = FCMPS; break;
  case Type::DoubleTyID: opCode = FCMPD; break;
  default: assert(0 && "Invalid type for FCMP instruction"); break; 
  }
  
  return opCode;
}


// Assumes that leftArg and rightArg are both cast instructions.
//
static inline bool
BothFloatToDouble(const InstructionNode* instrNode)
{
  InstrTreeNode* leftArg = instrNode->leftChild();
  InstrTreeNode* rightArg = instrNode->rightChild();
  InstrTreeNode* leftArgArg = leftArg->leftChild();
  InstrTreeNode* rightArgArg = rightArg->leftChild();
  assert(leftArg->getValue()->getType() == rightArg->getValue()->getType());
  
  // Check if both arguments are floats cast to double
  return (leftArg->getValue()->getType() == Type::DoubleTy &&
          leftArgArg->getValue()->getType() == Type::FloatTy &&
          rightArgArg->getValue()->getType() == Type::FloatTy);
}


static inline MachineOpCode 
ChooseMulInstructionByType(const Type* resultType)
{
  MachineOpCode opCode = INVALID_OPCODE;
  
  if (resultType->isIntegral())
    opCode = MULX;
  else
    switch(resultType->getPrimitiveID())
      {
      case Type::FloatTyID:  opCode = FMULS; break;
      case Type::DoubleTyID: opCode = FMULD; break;
      default: assert(0 && "Invalid type for MUL instruction"); break; 
      }
  
  return opCode;
}



static inline MachineInstr*
CreateIntNegInstruction(const TargetMachine& target,
                        Value* vreg)
{
  MachineInstr* minstr = new MachineInstr(SUB);
  minstr->SetMachineOperandReg(0, target.getRegInfo().getZeroRegNum());
  minstr->SetMachineOperandVal(1, MachineOperand::MO_VirtualRegister, vreg);
  minstr->SetMachineOperandVal(2, MachineOperand::MO_VirtualRegister, vreg);
  return minstr;
}


// Does not create any instructions if we cannot exploit constant to
// create a cheaper instruction
static inline void
CreateMulConstInstruction(const TargetMachine &target,
                          Value* lval, Value* rval, Value* destVal,
                          vector<MachineInstr*>& mvec)
{
  MachineInstr* minstr1 = NULL;
  MachineInstr* minstr2 = NULL;
  
  Value* constOp = rval;
  if (! isa<Constant>(constOp))
    return;
  
  // Cases worth optimizing are:
  // (1) Multiply by 0 or 1 for any type: replace with copy (ADD or FMOV)
  // (2) Multiply by 2^x for integer types: replace with Shift
  // 
  const Type* resultType = destVal->getType();
  
  if (resultType->isIntegral() || resultType->isPointerType())
    {
      unsigned pow;
      bool isValidConst;
      int64_t C = GetConstantValueAsSignedInt(constOp, isValidConst);
      if (isValidConst)
        {
          bool needNeg = false;
          if (C < 0)
            {
              needNeg = true;
              C = -C;
            }
          
          if (C == 0 || C == 1)
            {
              minstr1 = new MachineInstr(ADD);
              
              if (C == 0)
                minstr1->SetMachineOperandReg(0,
                                          target.getRegInfo().getZeroRegNum());
              else
                minstr1->SetMachineOperandVal(0,MachineOperand::MO_VirtualRegister,
                                          lval);
              minstr1->SetMachineOperandReg(1,target.getRegInfo().getZeroRegNum());
            }
          else if (IsPowerOf2(C, pow))
            {
              minstr1 = new MachineInstr((resultType == Type::LongTy)
                                        ? SLLX : SLL);
              minstr1->SetMachineOperandVal(0, MachineOperand::MO_VirtualRegister,
                                           lval);
              minstr1->SetMachineOperandConst(1, MachineOperand::MO_UnextendedImmed,
                                              pow);
            }
          
          if (minstr1 && needNeg)
            { // insert <reg = SUB 0, reg> after the instr to flip the sign
              minstr2 = CreateIntNegInstruction(target, destVal);
            }
        }
    }
  else
    {
      if (resultType == Type::FloatTy ||
          resultType == Type::DoubleTy)
        {
          double dval = cast<ConstantFP>(constOp)->getValue();
          if (fabs(dval) == 1)
            {
              bool needNeg = (dval < 0);
              
              MachineOpCode opCode = needNeg
                ? (resultType == Type::FloatTy? FNEGS : FNEGD)
                : (resultType == Type::FloatTy? FMOVS : FMOVD);
              
              minstr1 = new MachineInstr(opCode);
              minstr1->SetMachineOperandVal(0,
                                            MachineOperand::MO_VirtualRegister,
                                            lval);
            } 
        }
    }
  
  if (minstr1 != NULL)
    minstr1->SetMachineOperandVal(2, MachineOperand::MO_VirtualRegister,
                                  destVal);   
  
  if (minstr1)
    mvec.push_back(minstr1);
  if (minstr2)
    mvec.push_back(minstr2);
}


// Return NULL if we cannot exploit constant to create a cheaper instruction
static inline void
CreateMulInstruction(const TargetMachine &target,
                     Value* lval, Value* rval, Value* destVal,
                     vector<MachineInstr*>& mvec,
                     MachineOpCode forceMulOp = INVALID_MACHINE_OPCODE)
{
  unsigned int L = mvec.size();
  CreateMulConstInstruction(target, lval, rval, destVal, mvec);
  if (mvec.size() == L)
    { // no instructions were added so create MUL reg, reg, reg.
      // Use FSMULD if both operands are actually floats cast to doubles.
      // Otherwise, use the default opcode for the appropriate type.
      MachineOpCode mulOp = ((forceMulOp != INVALID_MACHINE_OPCODE)
                             ? forceMulOp 
                             : ChooseMulInstructionByType(destVal->getType()));
      MachineInstr* M = new MachineInstr(mulOp);
      M->SetMachineOperandVal(0, MachineOperand::MO_VirtualRegister, lval);
      M->SetMachineOperandVal(1, MachineOperand::MO_VirtualRegister, rval);
      M->SetMachineOperandVal(2, MachineOperand::MO_VirtualRegister, destVal);
      mvec.push_back(M);
    }
}


// Generate a divide instruction for Div or Rem.
// For Rem, this assumes that the operand type will be signed if the result
// type is signed.  This is correct because they must have the same sign.
// 
static inline MachineOpCode 
ChooseDivInstruction(TargetMachine &target,
                     const InstructionNode* instrNode)
{
  MachineOpCode opCode = INVALID_OPCODE;
  
  const Type* resultType = instrNode->getInstruction()->getType();
  
  if (resultType->isIntegral())
    opCode = resultType->isSigned()? SDIVX : UDIVX;
  else
    switch(resultType->getPrimitiveID())
      {
      case Type::FloatTyID:  opCode = FDIVS; break;
      case Type::DoubleTyID: opCode = FDIVD; break;
      default: assert(0 && "Invalid type for DIV instruction"); break; 
      }
  
  return opCode;
}


// Return NULL if we cannot exploit constant to create a cheaper instruction
static inline void
CreateDivConstInstruction(TargetMachine &target,
                          const InstructionNode* instrNode,
                          vector<MachineInstr*>& mvec)
{
  MachineInstr* minstr1 = NULL;
  MachineInstr* minstr2 = NULL;
  
  Value* constOp = ((InstrTreeNode*) instrNode->rightChild())->getValue();
  if (! isa<Constant>(constOp))
    return;
  
  // Cases worth optimizing are:
  // (1) Divide by 1 for any type: replace with copy (ADD or FMOV)
  // (2) Divide by 2^x for integer types: replace with SR[L or A]{X}
  // 
  const Type* resultType = instrNode->getInstruction()->getType();
  
  if (resultType->isIntegral())
    {
      unsigned pow;
      bool isValidConst;
      int64_t C = GetConstantValueAsSignedInt(constOp, isValidConst);
      if (isValidConst)
        {
          bool needNeg = false;
          if (C < 0)
            {
              needNeg = true;
              C = -C;
            }
          
          if (C == 1)
            {
              minstr1 = new MachineInstr(ADD);
              minstr1->SetMachineOperandVal(0,MachineOperand::MO_VirtualRegister,
                                          instrNode->leftChild()->getValue());
              minstr1->SetMachineOperandReg(1,target.getRegInfo().getZeroRegNum());
            }
          else if (IsPowerOf2(C, pow))
            {
              MachineOpCode opCode= ((resultType->isSigned())
                                     ? (resultType==Type::LongTy)? SRAX : SRA
                                     : (resultType==Type::LongTy)? SRLX : SRL);
              minstr1 = new MachineInstr(opCode);
              minstr1->SetMachineOperandVal(0, MachineOperand::MO_VirtualRegister,
                                           instrNode->leftChild()->getValue());
              minstr1->SetMachineOperandConst(1, MachineOperand::MO_UnextendedImmed,
                                           pow);
            }
          
          if (minstr1 && needNeg)
            { // insert <reg = SUB 0, reg> after the instr to flip the sign
              minstr2 = CreateIntNegInstruction(target,
                                                   instrNode->getValue());
            }
        }
    }
  else
    {
      if (resultType == Type::FloatTy ||
          resultType == Type::DoubleTy)
        {
          double dval = cast<ConstantFP>(constOp)->getValue();
          if (fabs(dval) == 1)
            {
              bool needNeg = (dval < 0);
              
              MachineOpCode opCode = needNeg
                ? (resultType == Type::FloatTy? FNEGS : FNEGD)
                : (resultType == Type::FloatTy? FMOVS : FMOVD);
              
              minstr1 = new MachineInstr(opCode);
              minstr1->SetMachineOperandVal(0, MachineOperand::MO_VirtualRegister,
                                           instrNode->leftChild()->getValue());
            } 
        }
    }
  
  if (minstr1 != NULL)
    minstr1->SetMachineOperandVal(2, MachineOperand::MO_VirtualRegister,
                                 instrNode->getValue());   
  
  if (minstr1)
    mvec.push_back(minstr1);
  if (minstr2)
    mvec.push_back(minstr2);
}


static void
CreateCodeForVariableSizeAlloca(const TargetMachine& target,
                                Instruction* result,
                                unsigned int tsize,
                                Value* numElementsVal,
                                vector<MachineInstr*>& getMvec)
{
  MachineInstr* M;
  
  // Create a Value to hold the (constant) element size
  Value* tsizeVal = ConstantSInt::get(Type::IntTy, tsize);

  // Get the constant offset from SP for dynamically allocated storage
  // and create a temporary Value to hold it.
  assert(result && result->getParent() && "Result value is not part of a method?");
  Method* method = result->getParent()->getParent();
  MachineCodeForMethod& mcInfo = MachineCodeForMethod::get(method);
  bool growUp;
  ConstantSInt* dynamicAreaOffset =
    ConstantSInt::get(Type::IntTy,
                      target.getFrameInfo().getDynamicAreaOffset(mcInfo,growUp));
  assert(! growUp && "Has SPARC v9 stack frame convention changed?");

  // Create a temporary value to hold the result of MUL
  TmpInstruction* tmpProd = new TmpInstruction(numElementsVal, tsizeVal);
  MachineCodeForInstruction::get(result).addTemp(tmpProd);
  
  // Instruction 1: mul numElements, typeSize -> tmpProd
  M = new MachineInstr(MULX);
  M->SetMachineOperandVal(0, MachineOperand::MO_VirtualRegister, numElementsVal);
  M->SetMachineOperandVal(1, MachineOperand::MO_VirtualRegister, tsizeVal);
  M->SetMachineOperandVal(2, MachineOperand::MO_VirtualRegister, tmpProd);
  getMvec.push_back(M);
        
  // Instruction 2: sub %sp, tmpProd -> %sp
  M = new MachineInstr(SUB);
  M->SetMachineOperandReg(0, target.getRegInfo().getStackPointer());
  M->SetMachineOperandVal(1, MachineOperand::MO_VirtualRegister, tmpProd);
  M->SetMachineOperandReg(2, target.getRegInfo().getStackPointer());
  getMvec.push_back(M);
  
  // Instruction 3: add %sp, frameSizeBelowDynamicArea -> result
  M = new MachineInstr(ADD);
  M->SetMachineOperandReg(0, target.getRegInfo().getStackPointer());
  M->SetMachineOperandVal(1, MachineOperand::MO_VirtualRegister, dynamicAreaOffset);
  M->SetMachineOperandVal(2, MachineOperand::MO_VirtualRegister, result);
  getMvec.push_back(M);
}        


static void
CreateCodeForFixedSizeAlloca(const TargetMachine& target,
                             Instruction* result,
                             unsigned int tsize,
                             unsigned int numElements,
                             vector<MachineInstr*>& getMvec)
{
  assert(result && result->getParent() && "Result value is not part of a method?");
  Method* method = result->getParent()->getParent();
  MachineCodeForMethod& mcInfo = MachineCodeForMethod::get(method);

  // Check if the offset would small enough to use as an immediate in load/stores
  // (check LDX because all load/stores have the same-size immediate field).
  // If not, put the variable in the dynamically sized area of the frame.
  int offsetFromFP = mcInfo.computeOffsetforLocalVar(target, result,
                                                     tsize * numElements);
  if (! target.getInstrInfo().constantFitsInImmedField(LDX, offsetFromFP))
    {
      CreateCodeForVariableSizeAlloca(target, result, tsize, 
                                      ConstantSInt::get(Type::IntTy,numElements),
                                      getMvec);
      return;
    }
  
  // else offset fits in immediate field so go ahead and allocate it.
  offsetFromFP = mcInfo.allocateLocalVar(target, result, tsize * numElements);
  
  // Create a temporary Value to hold the constant offset.
  // This is needed because it may not fit in the immediate field.
  ConstantSInt* offsetVal = ConstantSInt::get(Type::IntTy, offsetFromFP);
  
  // Instruction 1: add %fp, offsetFromFP -> result
  MachineInstr* M = new MachineInstr(ADD);
  M->SetMachineOperandReg(0, target.getRegInfo().getFramePointer());
  M->SetMachineOperandVal(1, MachineOperand::MO_VirtualRegister, offsetVal); 
  M->SetMachineOperandVal(2, MachineOperand::MO_VirtualRegister, result);
  
  getMvec.push_back(M);
}



//------------------------------------------------------------------------ 
// Function SetOperandsForMemInstr
//
// Choose addressing mode for the given load or store instruction.
// Use [reg+reg] if it is an indexed reference, and the index offset is
//		 not a constant or if it cannot fit in the offset field.
// Use [reg+offset] in all other cases.
// 
// This assumes that all array refs are "lowered" to one of these forms:
//	%x = load (subarray*) ptr, constant	; single constant offset
//	%x = load (subarray*) ptr, offsetVal	; single non-constant offset
// Generally, this should happen via strength reduction + LICM.
// Also, strength reduction should take care of using the same register for
// the loop index variable and an array index, when that is profitable.
//------------------------------------------------------------------------ 

static void
SetOperandsForMemInstr(vector<MachineInstr*>& mvec,
                       vector<MachineInstr*>::iterator mvecI,
                       const InstructionNode* vmInstrNode,
                       const TargetMachine& target)
{
  MemAccessInst* memInst = (MemAccessInst*) vmInstrNode->getInstruction();
  
  // Variables to hold the index vector, ptr value, and offset value.
  // The major work here is to extract these for all 3 instruction types
  // and then call the common function SetMemOperands_Internal().
  // 
  vector<Value*> idxVec;
  Value* ptrVal = memInst->getPointerOperand();
  
  // Test if a GetElemPtr instruction is being folded into this mem instrn.
  // If so, it will be in the left child for Load and GetElemPtr,
  // and in the right child for Store instructions.
  // 
  InstrTreeNode* ptrChild = (vmInstrNode->getOpLabel() == Instruction::Store
                             ? vmInstrNode->rightChild()
                             : vmInstrNode->leftChild()); 
  
  // We can only fold a chain of GetElemPtr instructions for structure references
  // 
  if (isa<StructType>(cast<PointerType>(ptrVal->getType())->getElementType())
      && (ptrChild->getOpLabel() == Instruction::GetElementPtr ||
          ptrChild->getOpLabel() == GetElemPtrIdx))
    {
      // There is a GetElemPtr instruction and there may be a chain of
      // more than one.  Use the pointer value of the last one in the chain.
      // Fold the index vectors from the entire chain and from the mem
      // instruction into one single index vector.
      // 
      ptrVal = FoldGetElemChain((InstructionNode*) ptrChild, idxVec);

      assert (! cast<PointerType>(ptrVal->getType())->getElementType()->isArrayType()
              && "GetElemPtr cannot be folded into array refs in selection");
    }
  
  // Append the index vector of this instruction (may be none) to the indexes
  // folded in previous getElementPtr's (may be none)
  idxVec.insert(idxVec.end(), memInst->idx_begin(), memInst->idx_end());
  
  SetMemOperands_Internal(mvec, mvecI, vmInstrNode, ptrVal, idxVec, target);
}


// Generate the correct operands (and additional instructions if needed)
// for the given pointer and given index vector.
//
static void
SetMemOperands_Internal(vector<MachineInstr*>& mvec,
                        vector<MachineInstr*>::iterator mvecI,
                        const InstructionNode* vmInstrNode,
                        Value* ptrVal,
                        const vector<Value*>& idxVec,
                        const TargetMachine& target)
{
  MemAccessInst* memInst = (MemAccessInst*) vmInstrNode->getInstruction();
  
  // Initialize so we default to storing the offset in a register.
  int64_t smallConstOffset = 0;
  Value* valueForRegOffset = NULL;
  MachineOperand::MachineOperandType offsetOpType =MachineOperand::MO_VirtualRegister;

  // Check if there is an index vector and if so, compute the
  // right offset for structures and for arrays 
  // 
  if (idxVec.size() > 0)
    {
      unsigned offset = 0;
      
      const PointerType* ptrType = cast<PointerType>(ptrVal->getType());
      
      if (ptrType->getElementType()->isStructType())
        {
          // Compute the offset value using the index vector,
          // and create a virtual register for it.
          unsigned offset = target.DataLayout.getIndexedOffset(ptrType,idxVec);
          valueForRegOffset = ConstantSInt::get(Type::IntTy, offset);
        }
      else
        {
          // It must be an array ref.  Check that the indexing has been
          // lowered to a single offset.
          assert((memInst->getNumOperands()
                  == (unsigned) 1 + memInst->getFirstIndexOperandNumber())
                 && "Array refs must be lowered before Instruction Selection");
          
          Value* arrayOffsetVal =  * memInst->idx_begin();
          
          // Generate a MUL instruction to compute address from index
          // The call to getTypeSize() will fail if size is not constant
          vector<MachineInstr*> mulVec;
          Instruction* addr = new TmpInstruction(Type::UIntTy, memInst);
          MachineCodeForInstruction::get(memInst).addTemp(addr);
          unsigned int eltSize =
            target.DataLayout.getTypeSize(ptrType->getElementType());
          assert(eltSize > 0 && "Invalid or non-constant array element size");
          ConstantUInt* eltVal = ConstantUInt::get(Type::UIntTy, eltSize);
          
          CreateMulInstruction(target,
                               arrayOffsetVal, /* lval, not likely constant */
                               eltVal,         /* rval, likely constant */
                               addr,           /* result*/
                               mulVec, INVALID_MACHINE_OPCODE);
          assert(mulVec.size() > 0 && "No multiply instruction created?");
          for (vector<MachineInstr*>::const_iterator I = mulVec.begin();
               I != mulVec.end(); ++I)
            {
              mvecI = mvec.insert(mvecI, *I);	// get ptr to inserted value
              ++mvecI;                          // get ptr to mem. instr.
            }
          
          valueForRegOffset = addr;
          
          // Check if the offset is a constant,
          // if (Constant *CPV = dyn_cast<Constant>(arrayOffsetVal))
          //   {
          //     isConstantOffset = true;  // always constant for structs
          //     assert(arrayOffsetVal->getType()->isIntegral());
          //     offset = (CPV->getType()->isSigned()
          //               ? cast<ConstantSInt>(CPV)->getValue()
          //               : (int64_t) cast<ConstantUInt>(CPV)->getValue());
          //   }
          // else
          //  {
          //    valueForRegOffset = arrayOffsetVal;
          //  }
        }
    }
  else
    {
      offsetOpType = MachineOperand::MO_SignExtendedImmed;
      smallConstOffset = 0;
    }
  
  // Operand 0 is value for STORE, ptr for LOAD or GET_ELEMENT_PTR
  // It is the left child in the instruction tree in all cases.
  Value* leftVal = vmInstrNode->leftChild()->getValue();
  (*mvecI)->SetMachineOperandVal(0, MachineOperand::MO_VirtualRegister,
                                 leftVal);
  
  // Operand 1 is ptr for STORE, offset for LOAD or GET_ELEMENT_PTR
  // Operand 2 is offset for STORE, result reg for LOAD or GET_ELEMENT_PTR
  //
  unsigned offsetOpNum = (memInst->getOpcode() == Instruction::Store)? 2 : 1;
  if (offsetOpType == MachineOperand::MO_VirtualRegister)
    {
      assert(valueForRegOffset != NULL);
      (*mvecI)->SetMachineOperandVal(offsetOpNum, offsetOpType,
                                     valueForRegOffset); 
    }
  else
    (*mvecI)->SetMachineOperandConst(offsetOpNum, offsetOpType,
                                     smallConstOffset);
  
  if (memInst->getOpcode() == Instruction::Store)
    (*mvecI)->SetMachineOperandVal(1, MachineOperand::MO_VirtualRegister,
                                   ptrVal);
  else
    (*mvecI)->SetMachineOperandVal(2, MachineOperand::MO_VirtualRegister,
                                   vmInstrNode->getValue());
}


// 
// Substitute operand `operandNum' of the instruction in node `treeNode'
// in place of the use(s) of that instruction in node `parent'.
// Check both explicit and implicit operands!
// Also make sure to skip over a parent who:
// (1) is a list node in the Burg tree, or
// (2) itself had its results forwarded to its parent
// 
static void
ForwardOperand(InstructionNode* treeNode,
               InstrTreeNode*   parent,
               int operandNum)
{
  assert(treeNode && parent && "Invalid invocation of ForwardOperand");
  
  Instruction* unusedOp = treeNode->getInstruction();
  Value* fwdOp = unusedOp->getOperand(operandNum);

  // The parent itself may be a list node, so find the real parent instruction
  while (parent->getNodeType() != InstrTreeNode::NTInstructionNode)
    {
      parent = parent->parent();
      assert(parent && "ERROR: Non-instruction node has no parent in tree.");
    }
  InstructionNode* parentInstrNode = (InstructionNode*) parent;
  
  Instruction* userInstr = parentInstrNode->getInstruction();
  MachineCodeForInstruction &mvec = MachineCodeForInstruction::get(userInstr);

  // The parent's mvec would be empty if it was itself forwarded.
  // Recursively call ForwardOperand in that case...
  //
  if (mvec.size() == 0)
    {
      assert(parent->parent() != NULL &&
             "Parent could not have been forwarded, yet has no instructions?");
      ForwardOperand(treeNode, parent->parent(), operandNum);
    }
  else
    {
      bool fwdSuccessful = false;
      for (unsigned i=0, N=mvec.size(); i < N; i++)
        {
          MachineInstr* minstr = mvec[i];
          for (unsigned i=0, numOps=minstr->getNumOperands(); i < numOps; ++i)
            {
              const MachineOperand& mop = minstr->getOperand(i);
              if (mop.getOperandType() == MachineOperand::MO_VirtualRegister &&
                  mop.getVRegValue() == unusedOp)
                {
                  minstr->SetMachineOperandVal(i,
                                MachineOperand::MO_VirtualRegister, fwdOp);
                  fwdSuccessful = true;
                }
            }
          
          for (unsigned i=0,numOps=minstr->getNumImplicitRefs(); i<numOps; ++i)
            if (minstr->getImplicitRef(i) == unusedOp)
              {
                minstr->setImplicitRef(i, fwdOp,
                                       minstr->implicitRefIsDefined(i));
                fwdSuccessful = true;
              }
        }
      assert(fwdSuccessful && "Value to be forwarded is never used!");
    }
}


void UltraSparcInstrInfo::
CreateCopyInstructionsByType(const TargetMachine& target,
                             Method* method,
                             Value* src,
                             Instruction* dest,
                             vector<MachineInstr*>& minstrVec) const
{
  bool loadConstantToReg = false;
  
  const Type* resultType = dest->getType();
  
  MachineOpCode opCode = ChooseAddInstructionByType(resultType);
  if (opCode == INVALID_OPCODE)
    {
      assert(0 && "Unsupported result type in CreateCopyInstructionsByType()");
      return;
    }
  
  // if `src' is a constant that doesn't fit in the immed field or if it is
  // a global variable (i.e., a constant address), generate a load
  // instruction instead of an add
  // 
  if (isa<Constant>(src))
    {
      unsigned int machineRegNum;
      int64_t immedValue;
      MachineOperand::MachineOperandType opType =
        ChooseRegOrImmed(src, opCode, target, /*canUseImmed*/ true,
                         machineRegNum, immedValue);
      
      if (opType == MachineOperand::MO_VirtualRegister)
        loadConstantToReg = true;
    }
  else if (isa<GlobalValue>(src))
    loadConstantToReg = true;
  
  if (loadConstantToReg)
    { // `src' is constant and cannot fit in immed field for the ADD
      // Insert instructions to "load" the constant into a register
      vector<TmpInstruction*> tempVec;
      target.getInstrInfo().CreateCodeToLoadConst(method, src, dest,minstrVec,tempVec);
      for (unsigned i=0; i < tempVec.size(); i++)
        MachineCodeForInstruction::get(dest).addTemp(tempVec[i]);
    }
  else
    { // Create an add-with-0 instruction of the appropriate type.
      // Make `src' the second operand, in case it is a constant
      // Use (unsigned long) 0 for a NULL pointer value.
      // 
      const Type* zeroValueType =
        (resultType->getPrimitiveID() == Type::PointerTyID)? Type::ULongTy
                                                           : resultType;
      MachineInstr* minstr = new MachineInstr(opCode);
      minstr->SetMachineOperandVal(0, MachineOperand::MO_VirtualRegister,
                                   Constant::getNullConstant(zeroValueType));
      minstr->SetMachineOperandVal(1, MachineOperand::MO_VirtualRegister, src);
      minstr->SetMachineOperandVal(2, MachineOperand::MO_VirtualRegister,dest);
      minstrVec.push_back(minstr);
    }
}



//******************* Externally Visible Functions *************************/


//------------------------------------------------------------------------ 
// External Function: GetInstructionsForProlog
// External Function: GetInstructionsForEpilog
//
// Purpose:
//   Create prolog and epilog code for procedure entry and exit
//------------------------------------------------------------------------ 

extern unsigned
GetInstructionsForProlog(BasicBlock* entryBB,
                         TargetMachine &target,
                         MachineInstr** mvec)
{
  MachineInstr* M;
  const MachineFrameInfo& frameInfo = target.getFrameInfo();
  unsigned int N = 0;
  
  // The second operand is the stack size. If it does not fit in the
  // immediate field, we have to use a free register to hold the size.
  // We will assume that local register `l0' is unused since the SAVE
  // instruction must be the first instruction in each procedure.
  // 
  Method* method = entryBB->getParent();
  MachineCodeForMethod& mcInfo = MachineCodeForMethod::get(method);
  unsigned int staticStackSize = mcInfo.getStaticStackSize();
  
  if (staticStackSize < (unsigned) frameInfo.getMinStackFrameSize())
    staticStackSize = (unsigned) frameInfo.getMinStackFrameSize();
  
  if (unsigned padsz = (staticStackSize %
                        (unsigned) frameInfo.getStackFrameSizeAlignment()))
    staticStackSize += frameInfo.getStackFrameSizeAlignment() - padsz;
  
  if (target.getInstrInfo().constantFitsInImmedField(SAVE, staticStackSize))
    {
      M = new MachineInstr(SAVE);
      M->SetMachineOperandReg(0, target.getRegInfo().getStackPointer());
      M->SetMachineOperandConst(1, MachineOperand::MO_SignExtendedImmed,
                                   - (int) staticStackSize);
      M->SetMachineOperandReg(2, target.getRegInfo().getStackPointer());
      mvec[N++] = M;
    }
  else
    {
      M = new MachineInstr(SETSW);
      M->SetMachineOperandReg(0, MachineOperand::MO_SignExtendedImmed,
                                 - staticStackSize);
      M->SetMachineOperandReg(1, MachineOperand::MO_MachineRegister,
                                 target.getRegInfo().getUnifiedRegNum(
                                  target.getRegInfo().getRegClassIDOfType(Type::IntTy),
                                  SparcIntRegOrder::l0));
      mvec[N++] = M;
      
      M = new MachineInstr(SAVE);
      M->SetMachineOperandReg(0, target.getRegInfo().getStackPointer());
      M->SetMachineOperandReg(1, MachineOperand::MO_MachineRegister,
                                 target.getRegInfo().getUnifiedRegNum(
                                  target.getRegInfo().getRegClassIDOfType(Type::IntTy),
                                  SparcIntRegOrder::l0));
      M->SetMachineOperandReg(2, target.getRegInfo().getStackPointer());
      mvec[N++] = M;
    }
  
  return N;
}


extern unsigned
GetInstructionsForEpilog(BasicBlock* anExitBB,
                         TargetMachine &target,
                         MachineInstr** mvec)
{
  mvec[0] = new MachineInstr(RESTORE);
  mvec[0]->SetMachineOperandReg(0, target.getRegInfo().getZeroRegNum());
  mvec[0]->SetMachineOperandConst(1, MachineOperand::MO_SignExtendedImmed,
                             (int64_t)0);
  mvec[0]->SetMachineOperandReg(2, target.getRegInfo().getZeroRegNum());
  
  return 1;
}


//------------------------------------------------------------------------ 
// External Function: ThisIsAChainRule
//
// Purpose:
//   Check if a given BURG rule is a chain rule.
//------------------------------------------------------------------------ 

extern bool
ThisIsAChainRule(int eruleno)
{
  switch(eruleno)
    {
    case 111:	// stmt:  reg
    case 113:	// stmt:  bool
    case 123:
    case 124:
    case 125:
    case 126:
    case 127:
    case 128:
    case 129:
    case 130:
    case 131:
    case 132:
    case 133:
    case 155:
    case 221:
    case 222:
    case 241:
    case 242:
    case 243:
    case 244:
      return true; break;
      
    default:
      return false; break;
    }
}


//------------------------------------------------------------------------ 
// External Function: GetInstructionsByRule
//
// Purpose:
//   Choose machine instructions for the SPARC according to the
//   patterns chosen by the BURG-generated parser.
//------------------------------------------------------------------------ 

void
GetInstructionsByRule(InstructionNode* subtreeRoot,
                      int ruleForNode,
                      short* nts,
                      TargetMachine &target,
                      vector<MachineInstr*>& mvec)
{
  bool checkCast = false;		// initialize here to use fall-through
  int nextRule;
  int forwardOperandNum = -1;
  unsigned int allocaSize = 0;
  MachineInstr* M, *M2;
  unsigned int L;

  mvec.clear(); 
  
  // 
  // Let's check for chain rules outside the switch so that we don't have
  // to duplicate the list of chain rule production numbers here again
  // 
  if (ThisIsAChainRule(ruleForNode))
    {
      // Chain rules have a single nonterminal on the RHS.
      // Get the rule that matches the RHS non-terminal and use that instead.
      // 
      assert(nts[0] && ! nts[1]
             && "A chain rule should have only one RHS non-terminal!");
      nextRule = burm_rule(subtreeRoot->state, nts[0]);
      nts = burm_nts[nextRule];
      GetInstructionsByRule(subtreeRoot, nextRule, nts, target, mvec);
    }
  else
    {
      switch(ruleForNode) {
      case 1:	// stmt:   Ret
      case 2:	// stmt:   RetValue(reg)
      {         // NOTE: Prepass of register allocation is responsible
                //	 for moving return value to appropriate register.
                // Mark the return-address register as a hidden virtual reg.
                // Mark the return value   register as an implicit ref of
                // the machine instruction.
         	// Finally put a NOP in the delay slot.
        ReturnInst *returnInstr =
          cast<ReturnInst>(subtreeRoot->getInstruction());
        assert(returnInstr->getOpcode() == Instruction::Ret);
        
        Instruction* returnReg = new TmpInstruction(returnInstr);
        MachineCodeForInstruction::get(returnInstr).addTemp(returnReg);
        
        M = new MachineInstr(JMPLRET);
        M->SetMachineOperandReg(0, MachineOperand::MO_VirtualRegister,
                                      returnReg);
        M->SetMachineOperandConst(1,MachineOperand::MO_SignExtendedImmed,
                                   (int64_t)8);
        M->SetMachineOperandReg(2, target.getRegInfo().getZeroRegNum());
        
        if (returnInstr->getReturnValue() != NULL)
          M->addImplicitRef(returnInstr->getReturnValue());
        
        mvec.push_back(M);
        mvec.push_back(new MachineInstr(NOP));
        
        break;
      }  
        
      case 3:	// stmt:   Store(reg,reg)
      case 4:	// stmt:   Store(reg,ptrreg)
        mvec.push_back(new MachineInstr(
                         ChooseStoreInstruction(
                            subtreeRoot->leftChild()->getValue()->getType())));
        SetOperandsForMemInstr(mvec, mvec.end()-1, subtreeRoot, target);
        break;

      case 5:	// stmt:   BrUncond
        M = new MachineInstr(BA);
        M->SetMachineOperandVal(0, MachineOperand::MO_CCRegister,
                                      (Value*)NULL);
        M->SetMachineOperandVal(1, MachineOperand::MO_PCRelativeDisp,
             cast<BranchInst>(subtreeRoot->getInstruction())->getSuccessor(0));
        mvec.push_back(M);
        
        // delay slot
        mvec.push_back(new MachineInstr(NOP));
        break;

      case 206:	// stmt:   BrCond(setCCconst)
      { // setCCconst => boolean was computed with `%b = setCC type reg1 const'
        // If the constant is ZERO, we can use the branch-on-integer-register
        // instructions and avoid the SUBcc instruction entirely.
        // Otherwise this is just the same as case 5, so just fall through.
        // 
        InstrTreeNode* constNode = subtreeRoot->leftChild()->rightChild();
        assert(constNode &&
               constNode->getNodeType() ==InstrTreeNode::NTConstNode);
        Constant *constVal = cast<Constant>(constNode->getValue());
        bool isValidConst;

        if ((constVal->getType()->isIntegral()
             || constVal->getType()->isPointerType())
            && GetConstantValueAsSignedInt(constVal, isValidConst) == 0
            && isValidConst)
          {
            BranchInst* brInst=cast<BranchInst>(subtreeRoot->getInstruction());
            
            // That constant is a zero after all...
            // Use the left child of setCC as the first argument!
            M = new MachineInstr(ChooseBprInstruction(subtreeRoot));
            M->SetMachineOperandVal(0, MachineOperand::MO_VirtualRegister,
                          subtreeRoot->leftChild()->leftChild()->getValue());
            M->SetMachineOperandVal(1, MachineOperand::MO_PCRelativeDisp,
                                    brInst->getSuccessor(0));
            mvec.push_back(M);

            // delay slot
            mvec.push_back(new MachineInstr(NOP));

            // false branch
            M = new MachineInstr(BA);
            M->SetMachineOperandVal(0, MachineOperand::MO_CCRegister,
                                    (Value*) NULL);
            M->SetMachineOperandVal(1, MachineOperand::MO_PCRelativeDisp,
                                          brInst->getSuccessor(1));
            mvec.push_back(M);
            
            // delay slot
            mvec.push_back(new MachineInstr(NOP));
            
            break;
          }
        // ELSE FALL THROUGH
      }

      case 6:	// stmt:   BrCond(bool)
      { // bool => boolean was computed with some boolean operator
        // (SetCC, Not, ...).  We need to check whether the type was a FP,
        // signed int or unsigned int, and check the branching condition in
        // order to choose the branch to use.
        // If it is an integer CC, we also need to find the unique
        // TmpInstruction representing that CC.
        // 
        BranchInst* brInst = cast<BranchInst>(subtreeRoot->getInstruction());
        bool isFPBranch;
        M = new MachineInstr(ChooseBccInstruction(subtreeRoot, isFPBranch));
        
        Value* ccValue = GetTmpForCC(subtreeRoot->leftChild()->getValue(),
                                     brInst->getParent()->getParent(),
                                     isFPBranch? Type::FloatTy : Type::IntTy);
        
        M->SetMachineOperandVal(0, MachineOperand::MO_CCRegister, ccValue);
        M->SetMachineOperandVal(1, MachineOperand::MO_PCRelativeDisp,
                                   brInst->getSuccessor(0));
        mvec.push_back(M);
        
        // delay slot
        mvec.push_back(new MachineInstr(NOP));
        
        // false branch
        M = new MachineInstr(BA);
        M->SetMachineOperandVal(0, MachineOperand::MO_CCRegister,
                                   (Value*) NULL);
        M->SetMachineOperandVal(1, MachineOperand::MO_PCRelativeDisp,
                                   brInst->getSuccessor(1));
        mvec.push_back(M);
        
        // delay slot
        mvec.push_back(new MachineInstr(NOP));
        break;
      }
        
      case 208:	// stmt:   BrCond(boolconst)
      {
        // boolconst => boolean is a constant; use BA to first or second label
        Constant* constVal = 
          cast<Constant>(subtreeRoot->leftChild()->getValue());
        unsigned dest = cast<ConstantBool>(constVal)->getValue()? 0 : 1;
        
        M = new MachineInstr(BA);
        M->SetMachineOperandVal(0, MachineOperand::MO_CCRegister,
                                (Value*) NULL);
        M->SetMachineOperandVal(1, MachineOperand::MO_PCRelativeDisp,
          ((BranchInst*) subtreeRoot->getInstruction())->getSuccessor(dest));
        mvec.push_back(M);
        
        // delay slot
        mvec.push_back(new MachineInstr(NOP));
        break;
      }
        
      case   8:	// stmt:   BrCond(boolreg)
      { // boolreg   => boolean is stored in an existing register.
        // Just use the branch-on-integer-register instruction!
        // 
        M = new MachineInstr(BRNZ);
        M->SetMachineOperandVal(0, MachineOperand::MO_VirtualRegister,
                                      subtreeRoot->leftChild()->getValue());
        M->SetMachineOperandVal(1, MachineOperand::MO_PCRelativeDisp,
              ((BranchInst*) subtreeRoot->getInstruction())->getSuccessor(0));
        mvec.push_back(M);

        // delay slot
        mvec.push_back(new MachineInstr(NOP));

        // false branch
        M = new MachineInstr(BA);
        M->SetMachineOperandVal(0, MachineOperand::MO_CCRegister,
                                (Value*) NULL);
        M->SetMachineOperandVal(1, MachineOperand::MO_PCRelativeDisp,
              ((BranchInst*) subtreeRoot->getInstruction())->getSuccessor(1));
        mvec.push_back(M);
        
        // delay slot
        mvec.push_back(new MachineInstr(NOP));
        break;
      }  
      
      case 9:	// stmt:   Switch(reg)
        assert(0 && "*** SWITCH instruction is not implemented yet.");
        break;

      case 10:	// reg:   VRegList(reg, reg)
        assert(0 && "VRegList should never be the topmost non-chain rule");
        break;

      case 21:	// bool:  Not(bool):	Both these are implemented as:
      case 321:	// reg:   BNot(reg) :	     reg = reg XOR-NOT 0
        M = new MachineInstr(XNOR);
        M->SetMachineOperandVal(0, MachineOperand::MO_VirtualRegister,
                                subtreeRoot->leftChild()->getValue());
        M->SetMachineOperandReg(1, target.getRegInfo().getZeroRegNum());
        M->SetMachineOperandVal(2, MachineOperand::MO_VirtualRegister,
                                subtreeRoot->getValue());
        mvec.push_back(M);
        break;

      case 322:	// reg:   ToBoolTy(bool):
      case 22:	// reg:   ToBoolTy(reg):
      {
        const Type* opType = subtreeRoot->leftChild()->getValue()->getType();
        assert(opType->isIntegral() || opType->isPointerType()
               || opType == Type::BoolTy);
        forwardOperandNum = 0;          // forward first operand to user
        break;
      }
      
      case 23:	// reg:   ToUByteTy(reg)
      case 25:	// reg:   ToUShortTy(reg)
      case 27:	// reg:   ToUIntTy(reg)
      case 29:	// reg:   ToULongTy(reg)
      {
        const Type* opType = subtreeRoot->leftChild()->getValue()->getType();
        assert(opType->isIntegral() ||
               opType->isPointerType() ||
               opType == Type::BoolTy && "Cast is illegal for other types");
        forwardOperandNum = 0;          // forward first operand to user
        break;
      }
      
      case 24:	// reg:   ToSByteTy(reg)
      case 26:	// reg:   ToShortTy(reg)
      case 28:	// reg:   ToIntTy(reg)
      case 30:	// reg:   ToLongTy(reg)
      {
        const Type* opType = subtreeRoot->leftChild()->getValue()->getType();
        if (opType->isIntegral()
            || opType->isPointerType()
            || opType == Type::BoolTy)
          {
            forwardOperandNum = 0;          // forward first operand to user
          }
        else
          {
            // If the source operand is an FP type, the int result must be
            // copied from float to int register via memory!
            Instruction *dest = subtreeRoot->getInstruction();
            Value* leftVal = subtreeRoot->leftChild()->getValue();
            Value* destForCast;
            vector<MachineInstr*> minstrVec;
            
            if (opType == Type::FloatTy || opType == Type::DoubleTy)
              {
                // Create a temporary to represent the INT register
                // into which the FP value will be copied via memory.
                // The type of this temporary will determine the FP
                // register used: single-prec for a 32-bit int or smaller,
                // double-prec for a 64-bit int.
                // 
                const Type* destTypeToUse =
                  (dest->getType() == Type::LongTy)? Type::DoubleTy
                                                   : Type::FloatTy;
                destForCast = new TmpInstruction(destTypeToUse, leftVal);
                MachineCodeForInstruction &MCFI = 
                  MachineCodeForInstruction::get(dest);
                MCFI.addTemp(destForCast);
                
                vector<TmpInstruction*> tempVec;
                target.getInstrInfo().CreateCodeToCopyFloatToInt(
                    dest->getParent()->getParent(),
                    (TmpInstruction*) destForCast, dest,
                    minstrVec, tempVec, target);
                
                for (unsigned i=0; i < tempVec.size(); ++i)
                  MCFI.addTemp(tempVec[i]);
              }
            else
              destForCast = leftVal;
            
            MachineOpCode opCode=ChooseConvertToIntInstr(subtreeRoot, opType);
            assert(opCode != INVALID_OPCODE && "Expected to need conversion!");
            
            M = new MachineInstr(opCode);
            M->SetMachineOperandVal(0, MachineOperand::MO_VirtualRegister,
                                    leftVal);
            M->SetMachineOperandVal(1, MachineOperand::MO_VirtualRegister,
                                    destForCast);
            mvec.push_back(M);

            // Append the copy code, if any, after the conversion instr.
            mvec.insert(mvec.end(), minstrVec.begin(), minstrVec.end());
          }
        break;
      }  
      
      case  31:	// reg:   ToFloatTy(reg):
      case  32:	// reg:   ToDoubleTy(reg):
      case 232:	// reg:   ToDoubleTy(Constant):
        
        // If this instruction has a parent (a user) in the tree 
        // and the user is translated as an FsMULd instruction,
        // then the cast is unnecessary.  So check that first.
        // In the future, we'll want to do the same for the FdMULq instruction,
        // so do the check here instead of only for ToFloatTy(reg).
        // 
        if (subtreeRoot->parent() != NULL &&
            MachineCodeForInstruction::get(((InstructionNode*)subtreeRoot->parent())->getInstruction())[0]->getOpCode() == FSMULD)
          {
            forwardOperandNum = 0;          // forward first operand to user
          }
        else
          {
            Value* leftVal = subtreeRoot->leftChild()->getValue();
            const Type* opType = leftVal->getType();
            MachineOpCode opCode=ChooseConvertToFloatInstr(subtreeRoot,opType);
            if (opCode == INVALID_OPCODE)	// no conversion needed
              {
                forwardOperandNum = 0;      // forward first operand to user
              }
            else
              {
                // If the source operand is a non-FP type it must be
                // first copied from int to float register via memory!
                Instruction *dest = subtreeRoot->getInstruction();
                Value* srcForCast;
                int n = 0;
                if (opType != Type::FloatTy && opType != Type::DoubleTy)
                  {
                    // Create a temporary to represent the FP register
                    // into which the integer will be copied via memory.
                    // The type of this temporary will determine the FP
                    // register used: single-prec for a 32-bit int or smaller,
                    // double-prec for a 64-bit int.
                    // 
                    const Type* srcTypeToUse =
                      (leftVal->getType() == Type::LongTy)? Type::DoubleTy
                                                          : Type::FloatTy;
                    
                    srcForCast = new TmpInstruction(srcTypeToUse, dest);
                    MachineCodeForInstruction &DestMCFI = 
                      MachineCodeForInstruction::get(dest);
                    DestMCFI.addTemp(srcForCast);
                    
                    vector<MachineInstr*> minstrVec;
                    vector<TmpInstruction*> tempVec;
                    target.getInstrInfo().CreateCodeToCopyIntToFloat(
                         dest->getParent()->getParent(),
                         leftVal, (TmpInstruction*) srcForCast,
                         minstrVec, tempVec, target);
                    
                    mvec.insert(mvec.end(), minstrVec.begin(),minstrVec.end());
                    
                    for (unsigned i=0; i < tempVec.size(); ++i)
                       DestMCFI.addTemp(tempVec[i]);
                  }
                else
                  srcForCast = leftVal;
                
                M = new MachineInstr(opCode);
                M->SetMachineOperandVal(0, MachineOperand::MO_VirtualRegister,
                                           srcForCast);
                M->SetMachineOperandVal(1, MachineOperand::MO_VirtualRegister,
                                           dest);
                mvec.push_back(M);
              }
          }
        break;

      case 19:	// reg:   ToArrayTy(reg):
      case 20:	// reg:   ToPointerTy(reg):
        forwardOperandNum = 0;          // forward first operand to user
        break;

      case 233:	// reg:   Add(reg, Constant)
        M = CreateAddConstInstruction(subtreeRoot);
        if (M != NULL)
          {
            mvec.push_back(M);
            break;
          }
        // ELSE FALL THROUGH
        
      case 33:	// reg:   Add(reg, reg)
        mvec.push_back(new MachineInstr(ChooseAddInstruction(subtreeRoot)));
        Set3OperandsFromInstr(mvec.back(), subtreeRoot, target);
        break;

      case 234:	// reg:   Sub(reg, Constant)
        M = CreateSubConstInstruction(subtreeRoot);
        if (M != NULL)
          {
            mvec.push_back(M);
            break;
          }
        // ELSE FALL THROUGH
        
      case 34:	// reg:   Sub(reg, reg)
        mvec.push_back(new MachineInstr(ChooseSubInstructionByType(
                                   subtreeRoot->getInstruction()->getType())));
        Set3OperandsFromInstr(mvec.back(), subtreeRoot, target);
        break;

      case 135:	// reg:   Mul(todouble, todouble)
        checkCast = true;
        // FALL THROUGH 

      case 35:	// reg:   Mul(reg, reg)
      {
        MachineOpCode forceOp = ((checkCast && BothFloatToDouble(subtreeRoot))
                                 ? FSMULD
                                 : INVALID_MACHINE_OPCODE);
        CreateMulInstruction(target,
                             subtreeRoot->leftChild()->getValue(),
                             subtreeRoot->rightChild()->getValue(),
                             subtreeRoot->getInstruction(),
                             mvec, forceOp);
        break;
      }
      case 335:	// reg:   Mul(todouble, todoubleConst)
        checkCast = true;
        // FALL THROUGH 

      case 235:	// reg:   Mul(reg, Constant)
      {
        MachineOpCode forceOp = ((checkCast && BothFloatToDouble(subtreeRoot))
                                 ? FSMULD
                                 : INVALID_MACHINE_OPCODE);
        CreateMulInstruction(target,
                             subtreeRoot->leftChild()->getValue(),
                             subtreeRoot->rightChild()->getValue(),
                             subtreeRoot->getInstruction(),
                             mvec, forceOp);
        break;
      }
      case 236:	// reg:   Div(reg, Constant)
        L = mvec.size();
        CreateDivConstInstruction(target, subtreeRoot, mvec);
        if (mvec.size() > L)
          break;
        // ELSE FALL THROUGH
      
      case 36:	// reg:   Div(reg, reg)
        mvec.push_back(new MachineInstr(ChooseDivInstruction(target, subtreeRoot)));
        Set3OperandsFromInstr(mvec.back(), subtreeRoot, target);
        break;

      case  37:	// reg:   Rem(reg, reg)
      case 237:	// reg:   Rem(reg, Constant)
      {
        Instruction* remInstr = subtreeRoot->getInstruction();
        
        TmpInstruction* quot = new TmpInstruction(
                                        subtreeRoot->leftChild()->getValue(),
                                        subtreeRoot->rightChild()->getValue());
        TmpInstruction* prod = new TmpInstruction(
                                        quot,
                                        subtreeRoot->rightChild()->getValue());
        MachineCodeForInstruction::get(remInstr).addTemp(quot).addTemp(prod); 
        
        M = new MachineInstr(ChooseDivInstruction(target, subtreeRoot));
        Set3OperandsFromInstr(M, subtreeRoot, target);
        M->SetMachineOperandVal(2, MachineOperand::MO_VirtualRegister,quot);
        mvec.push_back(M);
        
        M = new MachineInstr(ChooseMulInstructionByType(
                                   subtreeRoot->getInstruction()->getType()));
        M->SetMachineOperandVal(0, MachineOperand::MO_VirtualRegister,quot);
        M->SetMachineOperandVal(1, MachineOperand::MO_VirtualRegister,
                                      subtreeRoot->rightChild()->getValue());
        M->SetMachineOperandVal(2, MachineOperand::MO_VirtualRegister,prod);
        mvec.push_back(M);
        
        M = new MachineInstr(ChooseSubInstructionByType(
                                   subtreeRoot->getInstruction()->getType()));
        Set3OperandsFromInstr(M, subtreeRoot, target);
        M->SetMachineOperandVal(1, MachineOperand::MO_VirtualRegister,prod);
        mvec.push_back(M);
        
        break;
      }
      
      case  38:	// bool:   And(bool, bool)
      case 238:	// bool:   And(bool, boolconst)
      case 338:	// reg :   BAnd(reg, reg)
      case 538:	// reg :   BAnd(reg, Constant)
        mvec.push_back(new MachineInstr(AND));
        Set3OperandsFromInstr(mvec.back(), subtreeRoot, target);
        break;

      case 138:	// bool:   And(bool, not)
      case 438:	// bool:   BAnd(bool, not)
        mvec.push_back(new MachineInstr(ANDN));
        Set3OperandsFromInstr(mvec.back(), subtreeRoot, target);
        break;

      case  39:	// bool:   Or(bool, bool)
      case 239:	// bool:   Or(bool, boolconst)
      case 339:	// reg :   BOr(reg, reg)
      case 539:	// reg :   BOr(reg, Constant)
        mvec.push_back(new MachineInstr(ORN));
        Set3OperandsFromInstr(mvec.back(), subtreeRoot, target);
        break;

      case 139:	// bool:   Or(bool, not)
      case 439:	// bool:   BOr(bool, not)
        mvec.push_back(new MachineInstr(ORN));
        Set3OperandsFromInstr(mvec.back(), subtreeRoot, target);
        break;

      case  40:	// bool:   Xor(bool, bool)
      case 240:	// bool:   Xor(bool, boolconst)
      case 340:	// reg :   BXor(reg, reg)
      case 540:	// reg :   BXor(reg, Constant)
        mvec.push_back(new MachineInstr(XOR));
        Set3OperandsFromInstr(mvec.back(), subtreeRoot, target);
        break;

      case 140:	// bool:   Xor(bool, not)
      case 440:	// bool:   BXor(bool, not)
        mvec.push_back(new MachineInstr(XNOR));
        Set3OperandsFromInstr(mvec.back(), subtreeRoot, target);
        break;

      case 41:	// boolconst:   SetCC(reg, Constant)
        // Check if this is an integer comparison, and
        // there is a parent, and the parent decided to use
        // a branch-on-integer-register instead of branch-on-condition-code.
        // If so, the SUBcc instruction is not required.
        // (However, we must still check for constants to be loaded from
        // the constant pool so that such a load can be associated with
        // this instruction.)
        // 
        // Otherwise this is just the same as case 42, so just fall through.
        // 
        if ((subtreeRoot->leftChild()->getValue()->getType()->isIntegral() ||
             subtreeRoot->leftChild()->getValue()->getType()->isPointerType())
            && subtreeRoot->parent() != NULL)
          {
            InstructionNode* parent = (InstructionNode*) subtreeRoot->parent();
            assert(parent->getNodeType() == InstrTreeNode::NTInstructionNode);
            const MachineCodeForInstruction &minstrVec =
              MachineCodeForInstruction::get(parent->getInstruction());
            MachineOpCode parentOpCode;
            if (parent->getInstruction()->getOpcode() == Instruction::Br &&
                (parentOpCode = minstrVec[0]->getOpCode()) >= BRZ &&
                parentOpCode <= BRGEZ)
              {
                break;                  // don't forward the operand!
              }
          }
        // ELSE FALL THROUGH

      case 42:	// bool:   SetCC(reg, reg):
      {
        // This generates a SUBCC instruction, putting the difference in
        // a result register, and setting a condition code.
        // 
        // If the boolean result of the SetCC is used by anything other
        // than a single branch instruction, the boolean must be
        // computed and stored in the result register.  Otherwise, discard
        // the difference (by using %g0) and keep only the condition code.
        // 
        // To compute the boolean result in a register we use a conditional
        // move, unless the result of the SUBCC instruction can be used as
        // the bool!  This assumes that zero is FALSE and any non-zero
        // integer is TRUE.
        // 
        InstructionNode* parentNode = (InstructionNode*) subtreeRoot->parent();
        Instruction* setCCInstr = subtreeRoot->getInstruction();
        bool keepBoolVal = (parentNode == NULL ||
                            parentNode->getInstruction()->getOpcode()
                                != Instruction::Br);
        bool subValIsBoolVal = setCCInstr->getOpcode() == Instruction::SetNE;
        bool keepSubVal = keepBoolVal && subValIsBoolVal;
        bool computeBoolVal = keepBoolVal && ! subValIsBoolVal;
        
        bool mustClearReg;
        int valueToMove;
        MachineOpCode movOpCode = 0;

        // Mark the 4th operand as being a CC register, and as a def
        // A TmpInstruction is created to represent the CC "result".
        // Unlike other instances of TmpInstruction, this one is used
        // by machine code of multiple LLVM instructions, viz.,
        // the SetCC and the branch.  Make sure to get the same one!
        // Note that we do this even for FP CC registers even though they
        // are explicit operands, because the type of the operand
        // needs to be a floating point condition code, not an integer
        // condition code.  Think of this as casting the bool result to
        // a FP condition code register.
        // 
        Value* leftVal = subtreeRoot->leftChild()->getValue();
        bool isFPCompare = (leftVal->getType() == Type::FloatTy || 
                            leftVal->getType() == Type::DoubleTy);
        
        TmpInstruction* tmpForCC = GetTmpForCC(setCCInstr,
                                     setCCInstr->getParent()->getParent(),
                                     isFPCompare? Type::FloatTy : Type::IntTy);
        MachineCodeForInstruction::get(setCCInstr).addTemp(tmpForCC);
        
        if (! isFPCompare)
          {
            // Integer condition: dest. should be %g0 or an integer register.
            // If result must be saved but condition is not SetEQ then we need
            // a separate instruction to compute the bool result, so discard
            // result of SUBcc instruction anyway.
            // 
            M = new MachineInstr(SUBcc);
            Set3OperandsFromInstr(M, subtreeRoot, target, ! keepSubVal);
            M->SetMachineOperandVal(3, MachineOperand::MO_CCRegister,
                                    tmpForCC, /*def*/true);
            mvec.push_back(M);
            
            if (computeBoolVal)
              { // recompute bool using the integer condition codes
                movOpCode =
                  ChooseMovpccAfterSub(subtreeRoot,mustClearReg,valueToMove);
              }
          }
        else
          {
            // FP condition: dest of FCMP should be some FCCn register
            M = new MachineInstr(ChooseFcmpInstruction(subtreeRoot));
            M->SetMachineOperandVal(0, MachineOperand::MO_CCRegister,
                                          tmpForCC);
            M->SetMachineOperandVal(1,MachineOperand::MO_VirtualRegister,
                                         subtreeRoot->leftChild()->getValue());
            M->SetMachineOperandVal(2,MachineOperand::MO_VirtualRegister,
                                        subtreeRoot->rightChild()->getValue());
            mvec.push_back(M);
            
            if (computeBoolVal)
              {// recompute bool using the FP condition codes
                mustClearReg = true;
                valueToMove = 1;
                movOpCode = ChooseMovFpccInstruction(subtreeRoot);
              }
          }
        
        if (computeBoolVal)
          {
            if (mustClearReg)
              {// Unconditionally set register to 0
                M = new MachineInstr(SETHI);
                M->SetMachineOperandConst(0,MachineOperand::MO_UnextendedImmed,
                                          (int64_t)0);
                M->SetMachineOperandVal(1, MachineOperand::MO_VirtualRegister,
                                        setCCInstr);
                mvec.push_back(M);
              }
            
            // Now conditionally move `valueToMove' (0 or 1) into the register
            M = new MachineInstr(movOpCode);
            M->SetMachineOperandVal(0, MachineOperand::MO_CCRegister,
                                    tmpForCC);
            M->SetMachineOperandConst(1, MachineOperand::MO_UnextendedImmed,
                                      valueToMove);
            M->SetMachineOperandVal(2, MachineOperand::MO_VirtualRegister,
                                    setCCInstr);
            mvec.push_back(M);
          }
        break;
      }    

      case 43:	// boolreg: VReg
      case 44:	// boolreg: Constant
        break;

      case 51:	// reg:   Load(reg)
      case 52:	// reg:   Load(ptrreg)
      case 53:	// reg:   LoadIdx(reg,reg)
      case 54:	// reg:   LoadIdx(ptrreg,reg)
        mvec.push_back(new MachineInstr(ChooseLoadInstruction(
                                     subtreeRoot->getValue()->getType())));
        SetOperandsForMemInstr(mvec, mvec.end()-1, subtreeRoot, target);
        break;

      case 55:	// reg:   GetElemPtr(reg)
      case 56:	// reg:   GetElemPtrIdx(reg,reg)
        if (subtreeRoot->parent() != NULL)
          {
            // If the parent was a memory operation and not an array access,
            // the parent will fold this instruction in so generate nothing.
            // 
            Instruction* parent =
              cast<Instruction>(subtreeRoot->parent()->getValue());
            if (parent->getOpcode() == Instruction::Load ||
                parent->getOpcode() == Instruction::Store ||
                parent->getOpcode() == Instruction::GetElementPtr)
              {
                // Check if the parent is an array access,
                // If so, we still need to generate this instruction.
                GetElementPtrInst* getElemInst =
                  cast<GetElementPtrInst>(subtreeRoot->getInstruction());
                const PointerType* ptrType =
                  cast<PointerType>(getElemInst->getPointerOperand()->getType());
                if (! ptrType->getElementType()->isArrayType())
                  {// we don't need a separate instr
                    break;      		// don't forward operand!
                  }
              }
          }
        // else in all other cases we need to a separate ADD instruction
        mvec.push_back(new MachineInstr(ADD));
        SetOperandsForMemInstr(mvec, mvec.end()-1, subtreeRoot, target);
        break;

      case 57:	// reg:  Alloca: Implement as 1 instruction:
      {         //	    add %fp, offsetFromFP -> result
        AllocationInst* instr = cast<AllocationInst>(subtreeRoot->getInstruction());
        unsigned int tsize =target.findOptimalStorageSize(instr->getAllocatedType());
        assert(tsize != 0);
        CreateCodeForFixedSizeAlloca(target, instr, tsize, 1, mvec);
        break;
      }
      
      case 58:	// reg:   Alloca(reg): Implement as 3 instructions:
                //	mul num, typeSz -> tmp
                //	sub %sp, tmp    -> %sp
      {         //	add %sp, frameSizeBelowDynamicArea -> result
        AllocationInst* instr = cast<AllocationInst>(subtreeRoot->getInstruction());
        const Type* eltType = instr->getAllocatedType();
        
        // If the #elements is a constant, use simpler code for fixed-size allocas
        int tsize = (int) target.findOptimalStorageSize(eltType);
        if (isa<Constant>(instr->getArraySize()))
          // total size is constant: generate code for fixed-size alloca
          CreateCodeForFixedSizeAlloca(target, instr, tsize,
                         cast<ConstantUInt>(instr->getArraySize())->getValue(), mvec);
        else // total size is not constant.
          CreateCodeForVariableSizeAlloca(target, instr, tsize,
                                          instr->getArraySize(), mvec);
        break;
      }
      
      case 61:	// reg:   Call
      {         // Generate a call-indirect (i.e., jmpl) for now to expose
                // the potential need for registers.  If an absolute address
                // is available, replace this with a CALL instruction.
                // Mark both the indirection register and the return-address
        	// register as hidden virtual registers.
                // Also, mark the operands of the Call and return value (if
                // any) as implicit operands of the CALL machine instruction.
                // 
        CallInst *callInstr = cast<CallInst>(subtreeRoot->getInstruction());
        Value *callee = callInstr->getCalledValue();
        
        Instruction* retAddrReg = new TmpInstruction(callInstr);
        
        // Note temporary values in the machineInstrVec for the VM instr.
        //
        // WARNING: Operands 0..N-1 must go in slots 0..N-1 of implicitUses.
        //          The result value must go in slot N.  This is assumed
        //          in register allocation.
        // 
        MachineCodeForInstruction::get(callInstr).addTemp(retAddrReg);
        
        
        // Generate the machine instruction and its operands.
        // Use CALL for direct function calls; this optimistically assumes
        // the PC-relative address fits in the CALL address field (22 bits).
        // Use JMPL for indirect calls.
        // 
        if (callee->getValueType() == Value::MethodVal)
          { // direct function call
            M = new MachineInstr(CALL);
            M->SetMachineOperandVal(0, MachineOperand::MO_PCRelativeDisp,
                                    callee);
          } 
        else
          { // indirect function call
            M = new MachineInstr(JMPLCALL);
            M->SetMachineOperandVal(0, MachineOperand::MO_VirtualRegister,
                                    callee);
            M->SetMachineOperandConst(1, MachineOperand::MO_SignExtendedImmed,
                                      (int64_t) 0);
            M->SetMachineOperandVal(2, MachineOperand::MO_VirtualRegister,
                                    retAddrReg);
          }
        
        mvec.push_back(M);
        
        // Add the call operands and return value as implicit refs
        for (unsigned i=0, N=callInstr->getNumOperands(); i < N; ++i)
          if (callInstr->getOperand(i) != callee)
            mvec.back()->addImplicitRef(callInstr->getOperand(i));
        
        if (callInstr->getType() != Type::VoidTy)
          mvec.back()->addImplicitRef(callInstr, /*isDef*/ true);
        
        // For the CALL instruction, the ret. addr. reg. is also implicit
        if (callee->getValueType() == Value::MethodVal)
          mvec.back()->addImplicitRef(retAddrReg, /*isDef*/ true);
        
        // delay slot
        mvec.push_back(new MachineInstr(NOP));
        break;
      }

      case 62:	// reg:   Shl(reg, reg)
      { const Type* opType = subtreeRoot->leftChild()->getValue()->getType();
        assert(opType->isIntegral()
               || opType == Type::BoolTy
               || opType->isPointerType()&& "Shl unsupported for other types");
        mvec.push_back(new MachineInstr((opType == Type::LongTy)? SLLX : SLL));
        Set3OperandsFromInstr(mvec.back(), subtreeRoot, target);
        break;
      }
      
      case 63:	// reg:   Shr(reg, reg)
      { const Type* opType = subtreeRoot->leftChild()->getValue()->getType();
        assert(opType->isIntegral()
               || opType == Type::BoolTy
               || opType->isPointerType() &&"Shr unsupported for other types");
        mvec.push_back(new MachineInstr((opType->isSigned()
                                   ? ((opType == Type::LongTy)? SRAX : SRA)
                                   : ((opType == Type::LongTy)? SRLX : SRL))));
        Set3OperandsFromInstr(mvec.back(), subtreeRoot, target);
        break;
      }
      
      case 64:	// reg:   Phi(reg,reg)
        break;                          // don't forward the value

#undef NEED_PHI_MACHINE_INSTRS
#ifdef NEED_PHI_MACHINE_INSTRS
      {		// This instruction has variable #operands, so resultPos is 0.
        Instruction* phi = subtreeRoot->getInstruction();
        M = new MachineInstr(PHI, 1 + phi->getNumOperands());
        M->SetMachineOperandVal(0, MachineOperand::MO_VirtualRegister,
                                      subtreeRoot->getValue());
        for (unsigned i=0, N=phi->getNumOperands(); i < N; i++)
          M->SetMachineOperandVal(i+1, MachineOperand::MO_VirtualRegister,
                                  phi->getOperand(i));
        mvec.push_back(M);
        break;
      }  
#endif // NEED_PHI_MACHINE_INSTRS
      
      
      case 71:	// reg:     VReg
      case 72:	// reg:     Constant
        break;                          // don't forward the value

      default:
        assert(0 && "Unrecognized BURG rule");
        break;
      }
    }
  
  if (forwardOperandNum >= 0)
    { // We did not generate a machine instruction but need to use operand.
      // If user is in the same tree, replace Value in its machine operand.
      // If not, insert a copy instruction which should get coalesced away
      // by register allocation.
      if (subtreeRoot->parent() != NULL)
        ForwardOperand(subtreeRoot, subtreeRoot->parent(), forwardOperandNum);
      else
        {
          vector<MachineInstr*> minstrVec;
          target.getInstrInfo().CreateCopyInstructionsByType(target, 
                subtreeRoot->getInstruction()->getParent()->getParent(),
                subtreeRoot->getInstruction()->getOperand(forwardOperandNum),
                subtreeRoot->getInstruction(), minstrVec);
          assert(minstrVec.size() > 0);
          mvec.insert(mvec.end(), minstrVec.begin(), minstrVec.end());
        }
    }
}


