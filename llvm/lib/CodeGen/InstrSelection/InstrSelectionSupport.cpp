//===-- InstrSelectionSupport.cpp -----------------------------------------===//
//
// Target-independent instruction selection code.  See SparcInstrSelection.cpp
// for usage.
// 
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/InstrSelectionSupport.h"
#include "llvm/CodeGen/InstrSelection.h"
#include "llvm/CodeGen/MachineInstrAnnot.h"
#include "llvm/CodeGen/MachineCodeForInstruction.h"
#include "llvm/CodeGen/InstrForest.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Constants.h"
#include "llvm/BasicBlock.h"
#include "llvm/DerivedTypes.h"
using std::vector;

//*************************** Local Functions ******************************/


// Generate code to load the constant into a TmpInstruction (virtual reg) and
// returns the virtual register.
// 
static TmpInstruction*
InsertCodeToLoadConstant(Function *F,
                         Value* opValue,
                         Instruction* vmInstr,
                         vector<MachineInstr*>& loadConstVec,
                         TargetMachine& target)
{
  // Create a tmp virtual register to hold the constant.
  TmpInstruction* tmpReg = new TmpInstruction(opValue);
  MachineCodeForInstruction &mcfi = MachineCodeForInstruction::get(vmInstr);
  mcfi.addTemp(tmpReg);
  
  target.getInstrInfo().CreateCodeToLoadConst(target, F, opValue, tmpReg,
                                              loadConstVec, mcfi);
  
  // Record the mapping from the tmp VM instruction to machine instruction.
  // Do this for all machine instructions that were not mapped to any
  // other temp values created by 
  // tmpReg->addMachineInstruction(loadConstVec.back());
  
  return tmpReg;
}


MachineOperand::MachineOperandType
ChooseRegOrImmed(int64_t intValue,
                 bool isSigned,
		 MachineOpCode opCode,
		 const TargetMachine& target,
		 bool canUseImmed,
		 unsigned int& getMachineRegNum,
		 int64_t& getImmedValue)
{
  MachineOperand::MachineOperandType opType=MachineOperand::MO_VirtualRegister;
  getMachineRegNum = 0;
  getImmedValue = 0;

  if (canUseImmed &&
      target.getInstrInfo().constantFitsInImmedField(opCode, intValue))
    {
      opType = isSigned? MachineOperand::MO_SignExtendedImmed
                       : MachineOperand::MO_UnextendedImmed;
      getImmedValue = intValue;
    }
  else if (intValue == 0 && target.getRegInfo().getZeroRegNum() >= 0)
    {
      opType = MachineOperand::MO_MachineRegister;
      getMachineRegNum = target.getRegInfo().getZeroRegNum();
    }

  return opType;
}


MachineOperand::MachineOperandType
ChooseRegOrImmed(Value* val,
		 MachineOpCode opCode,
		 const TargetMachine& target,
		 bool canUseImmed,
		 unsigned int& getMachineRegNum,
		 int64_t& getImmedValue)
{
  getMachineRegNum = 0;
  getImmedValue = 0;

  // To use reg or immed, constant needs to be integer, bool, or a NULL pointer
  Constant *CPV = dyn_cast<Constant>(val);
  if (CPV == NULL ||
      (! CPV->getType()->isIntegral() &&
       ! (isa<PointerType>(CPV->getType()) && CPV->isNullValue())))
    return MachineOperand::MO_VirtualRegister;

  // Now get the constant value and check if it fits in the IMMED field.
  // Take advantage of the fact that the max unsigned value will rarely
  // fit into any IMMED field and ignore that case (i.e., cast smaller
  // unsigned constants to signed).
  // 
  int64_t intValue;
  if (isa<PointerType>(CPV->getType()))
    intValue = 0;                       // We checked above that it is NULL 
  else if (ConstantBool* CB = dyn_cast<ConstantBool>(CPV))
    intValue = (int64_t) CB->getValue();
  else if (CPV->getType()->isSigned())
    intValue = cast<ConstantSInt>(CPV)->getValue();
  else
    { // get the int value and sign-extend if original was less than 64 bits
      intValue = (int64_t) cast<ConstantUInt>(CPV)->getValue();
      switch(CPV->getType()->getPrimitiveID())
        {
        case Type::UByteTyID:  intValue = (int64_t) (int8_t) intValue; break;
        case Type::UShortTyID: intValue = (int64_t) (short)  intValue; break;
        case Type::UIntTyID:   intValue = (int64_t) (int)    intValue; break;
        default: break;
        }
    }

  return ChooseRegOrImmed(intValue, CPV->getType()->isSigned(),
                          opCode, target, canUseImmed,
                          getMachineRegNum, getImmedValue);
}



//---------------------------------------------------------------------------
// Function: FixConstantOperandsForInstr
// 
// Purpose:
// Special handling for constant operands of a machine instruction
// -- if the constant is 0, use the hardwired 0 register, if any;
// -- if the constant fits in the IMMEDIATE field, use that field;
// -- else create instructions to put the constant into a register, either
//    directly or by loading explicitly from the constant pool.
// 
// In the first 2 cases, the operand of `minstr' is modified in place.
// Returns a vector of machine instructions generated for operands that
// fall under case 3; these must be inserted before `minstr'.
//---------------------------------------------------------------------------

vector<MachineInstr*>
FixConstantOperandsForInstr(Instruction* vmInstr,
                            MachineInstr* minstr,
                            TargetMachine& target)
{
  vector<MachineInstr*> MVec;
  
  MachineOpCode opCode = minstr->getOpCode();
  const TargetInstrInfo& instrInfo = target.getInstrInfo();
  int resultPos = instrInfo.getResultPos(opCode);
  int immedPos = instrInfo.getImmedConstantPos(opCode);

  Function *F = vmInstr->getParent()->getParent();

  for (unsigned op=0; op < minstr->getNumOperands(); op++)
    {
      const MachineOperand& mop = minstr->getOperand(op);
          
      // Skip the result position, preallocated machine registers, or operands
      // that cannot be constants (CC regs or PC-relative displacements)
      if (resultPos == (int)op ||
          mop.getType() == MachineOperand::MO_MachineRegister ||
          mop.getType() == MachineOperand::MO_CCRegister ||
          mop.getType() == MachineOperand::MO_PCRelativeDisp)
        continue;

      bool constantThatMustBeLoaded = false;
      unsigned int machineRegNum = 0;
      int64_t immedValue = 0;
      Value* opValue = NULL;
      MachineOperand::MachineOperandType opType =
        MachineOperand::MO_VirtualRegister;

      // Operand may be a virtual register or a compile-time constant
      if (mop.getType() == MachineOperand::MO_VirtualRegister)
        {
          assert(mop.getVRegValue() != NULL);
          opValue = mop.getVRegValue();
          if (Constant *opConst = dyn_cast<Constant>(opValue)) {
            opType = ChooseRegOrImmed(opConst, opCode, target,
                                      (immedPos == (int)op), machineRegNum,
                                      immedValue);
            if (opType == MachineOperand::MO_VirtualRegister)
              constantThatMustBeLoaded = true;
          }
        }
      else
        {
          assert(mop.isImmediate());
          bool isSigned = mop.getType() == MachineOperand::MO_SignExtendedImmed;

          // Bit-selection flags indicate an instruction that is extracting
          // bits from its operand so ignore this even if it is a big constant.
          if (mop.opHiBits32() || mop.opLoBits32() ||
              mop.opHiBits64() || mop.opLoBits64())
            continue;

          opType = ChooseRegOrImmed(mop.getImmedValue(), isSigned,
                                    opCode, target, (immedPos == (int)op), 
                                    machineRegNum, immedValue);

          if (opType == mop.getType()) 
            continue;           // no change: this is the most common case

          if (opType == MachineOperand::MO_VirtualRegister)
            {
              constantThatMustBeLoaded = true;
              opValue = isSigned
                ? (Value*)ConstantSInt::get(Type::LongTy, immedValue)
                : (Value*)ConstantUInt::get(Type::ULongTy,(uint64_t)immedValue);
            }
        }

      if (opType == MachineOperand::MO_MachineRegister)
        minstr->SetMachineOperandReg(op, machineRegNum);
      else if (opType == MachineOperand::MO_SignExtendedImmed ||
               opType == MachineOperand::MO_UnextendedImmed)
        minstr->SetMachineOperandConst(op, opType, immedValue);
      else if (constantThatMustBeLoaded ||
               (opValue && isa<GlobalValue>(opValue)))
        { // opValue is a constant that must be explicitly loaded into a reg
          assert(opValue);
          TmpInstruction* tmpReg = InsertCodeToLoadConstant(F, opValue, vmInstr,
                                                            MVec, target);
          minstr->SetMachineOperandVal(op, MachineOperand::MO_VirtualRegister,
                                       tmpReg);
        }
    }
  
  // Also, check for implicit operands used by the machine instruction
  // (no need to check those defined since they cannot be constants).
  // These include:
  // -- arguments to a Call
  // -- return value of a Return
  // Any such operand that is a constant value needs to be fixed also.
  // The current instructions with implicit refs (viz., Call and Return)
  // have no immediate fields, so the constant always needs to be loaded
  // into a register.
  // 
  bool isCall = instrInfo.isCall(opCode);
  unsigned lastCallArgNum = 0;          // unused if not a call
  CallArgsDescriptor* argDesc = NULL;   // unused if not a call
  if (isCall)
    argDesc = CallArgsDescriptor::get(minstr);
  
  for (unsigned i=0, N=minstr->getNumImplicitRefs(); i < N; ++i)
    if (isa<Constant>(minstr->getImplicitRef(i)) ||
        isa<GlobalValue>(minstr->getImplicitRef(i)))
      {
        Value* oldVal = minstr->getImplicitRef(i);
        TmpInstruction* tmpReg =
          InsertCodeToLoadConstant(F, oldVal, vmInstr, MVec, target);
        minstr->setImplicitRef(i, tmpReg);
        
        if (isCall)
          { // find and replace the argument in the CallArgsDescriptor
            unsigned i=lastCallArgNum;
            while (argDesc->getArgInfo(i).getArgVal() != oldVal)
              ++i;
            assert(i < argDesc->getNumArgs() &&
                   "Constant operands to a call *must* be in the arg list");
            lastCallArgNum = i;
            argDesc->getArgInfo(i).replaceArgVal(tmpReg);
          }
      }
  
  return MVec;
}
