// $Id$
//***************************************************************************
// File:
//	SparcInstrInfo.cpp
// 
// Purpose:
//	
// History:
//	10/15/01	 -  Vikram Adve  -  Created
//**************************************************************************/


#include "SparcInternals.h"
#include "SparcInstrSelectionSupport.h"
#include "llvm/Target/Sparc.h"
#include "llvm/CodeGen/InstrSelection.h"
#include "llvm/CodeGen/InstrSelectionSupport.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineCodeForMethod.h"
#include "llvm/Function.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"


//************************ Internal Functions ******************************/


static inline MachineInstr*
CreateIntSetInstruction(int64_t C, Value* dest,
                        std::vector<TmpInstruction*>& tempVec)
{
  MachineInstr* minstr;
  uint64_t absC = (C >= 0)? C : -C;
  if (absC > (unsigned int) ~0)
    { // C does not fit in 32 bits
      TmpInstruction* tmpReg = new TmpInstruction(Type::IntTy);
      tempVec.push_back(tmpReg);
      
      minstr = new MachineInstr(SETX);
      minstr->SetMachineOperandConst(0, MachineOperand::MO_SignExtendedImmed, C);
      minstr->SetMachineOperandVal(1, MachineOperand::MO_VirtualRegister, tmpReg,
                                   /*isdef*/ true);
      minstr->SetMachineOperandVal(2, MachineOperand::MO_VirtualRegister,dest);
    }
  else
    {
      minstr = new MachineInstr(SETSW);
      minstr->SetMachineOperandConst(0,MachineOperand::MO_SignExtendedImmed,C);
      minstr->SetMachineOperandVal(1, MachineOperand::MO_VirtualRegister,dest);
    }
  
  return minstr;
}

static inline MachineInstr*
CreateUIntSetInstruction(uint64_t C, Value* dest,
                         std::vector<TmpInstruction*>& tempVec)
{
  MachineInstr* minstr;
  if (C > (unsigned int) ~0)
    { // C does not fit in 32 bits
      assert(dest->getType() == Type::ULongTy && "Sign extension problems");
      TmpInstruction *tmpReg = new TmpInstruction(Type::IntTy);
      tempVec.push_back(tmpReg);
      
      minstr = new MachineInstr(SETX);
      minstr->SetMachineOperandConst(0,MachineOperand::MO_SignExtendedImmed,C);
      minstr->SetMachineOperandVal(1, MachineOperand::MO_VirtualRegister,
                                   tmpReg, /*isdef*/ true);
      minstr->SetMachineOperandVal(2, MachineOperand::MO_VirtualRegister,dest);
    }
  else if (dest->getType() == Type::ULongTy)
    {
      minstr = new MachineInstr(SETUW);
      minstr->SetMachineOperandConst(0, MachineOperand::MO_UnextendedImmed, C);
      minstr->SetMachineOperandVal(1, MachineOperand::MO_VirtualRegister,dest);
    }
  else
    { // cast to signed type of the right length and use signed op (SETSW)
      // to get correct sign extension
      // 
      minstr = new MachineInstr(SETSW);
      minstr->SetMachineOperandVal(1, MachineOperand::MO_VirtualRegister,dest);
      
      switch (dest->getType()->getPrimitiveID())
        {
        case Type::UIntTyID:
          minstr->SetMachineOperandConst(0,
                                         MachineOperand::MO_SignExtendedImmed,
                                         (int) C);
          break;
        case Type::UShortTyID:
          minstr->SetMachineOperandConst(0,
                                         MachineOperand::MO_SignExtendedImmed,
                                         (short) C);
          break;
        case Type::UByteTyID:
          minstr->SetMachineOperandConst(0,
                                         MachineOperand::MO_SignExtendedImmed,
                                         (char) C);
          break;
        default:
          assert(0 && "Unexpected unsigned type");
          break;
        }
    }
  
  return minstr;
}

//************************* External Classes *******************************/

//---------------------------------------------------------------------------
// class UltraSparcInstrInfo 
// 
// Purpose:
//   Information about individual instructions.
//   Most information is stored in the SparcMachineInstrDesc array above.
//   Other information is computed on demand, and most such functions
//   default to member functions in base class MachineInstrInfo. 
//---------------------------------------------------------------------------

/*ctor*/
UltraSparcInstrInfo::UltraSparcInstrInfo(const TargetMachine& tgt)
  : MachineInstrInfo(tgt, SparcMachineInstrDesc,
		     /*descSize = */ NUM_TOTAL_OPCODES,
		     /*numRealOpCodes = */ NUM_REAL_OPCODES)
{
}

// 
// Create an instruction sequence to put the constant `val' into
// the virtual register `dest'.  `val' may be a Constant or a
// GlobalValue, viz., the constant address of a global variable or function.
// The generated instructions are returned in `minstrVec'.
// Any temp. registers (TmpInstruction) created are returned in `tempVec'.
// 
void
UltraSparcInstrInfo::CreateCodeToLoadConst(Function *F, Value* val,
                                           Instruction* dest,
                                           std::vector<MachineInstr*>&minstrVec,
                                    std::vector<TmpInstruction*>& tempVec) const
{
  MachineInstr* minstr;
  
  assert(isa<Constant>(val) || isa<GlobalValue>(val) &&
         "I only know about constant values and global addresses");
  
  // Use a "set" instruction for known constants that can go in an integer reg.
  // Use a "load" instruction for all other constants, in particular,
  // floating point constants and addresses of globals.
  // 
  const Type* valType = val->getType();
  
  if (valType->isIntegral() || valType == Type::BoolTy)
    {
      if (ConstantUInt* uval = dyn_cast<ConstantUInt>(val))
        {
          uint64_t C = uval->getValue();
          minstr = CreateUIntSetInstruction(C, dest, tempVec);
        }
      else
        {
          bool isValidConstant;
          int64_t C = GetConstantValueAsSignedInt(val, isValidConstant);
          assert(isValidConstant && "Unrecognized constant");
          minstr = CreateIntSetInstruction(C, dest, tempVec);
        }
      minstrVec.push_back(minstr);
    }
  else
    {
      // Make an instruction sequence to load the constant, viz:
      //            SETX <addr-of-constant>, tmpReg, addrReg
      //            LOAD  /*addr*/ addrReg, /*offset*/ 0, dest
      // Only the SETX is needed if `val' is a GlobalValue, i.e,. it is
      // itself a constant address.  Otherwise, both are needed.
      
      Value* addrVal;
      int64_t zeroOffset = 0; // to avoid ambiguity with (Value*) 0
      
      TmpInstruction* tmpReg =
        new TmpInstruction(PointerType::get(val->getType()), val);
      tempVec.push_back(tmpReg);
      
      if (isa<Constant>(val))
        {
          // Create another TmpInstruction for the hidden integer register
          TmpInstruction* addrReg =
            new TmpInstruction(PointerType::get(val->getType()), val);
          tempVec.push_back(addrReg);
          addrVal = addrReg;
        }
      else
        addrVal = dest;
      
      minstr = new MachineInstr(SETX);
      minstr->SetMachineOperandVal(0, MachineOperand::MO_PCRelativeDisp, val);
      minstr->SetMachineOperandVal(1, MachineOperand::MO_VirtualRegister,tmpReg,
                                   /*isdef*/ true);
      minstr->SetMachineOperandVal(2, MachineOperand::MO_VirtualRegister,
                                   addrVal);
      minstrVec.push_back(minstr);
      
      if (isa<Constant>(val))
        {
          // Make sure constant is emitted to constant pool in assembly code.
          MachineCodeForMethod& mcinfo = MachineCodeForMethod::get(F);
          mcinfo.addToConstantPool(cast<Constant>(val));
          
          // Generate the load instruction
          minstr = new MachineInstr(ChooseLoadInstruction(val->getType()));
          minstr->SetMachineOperandVal(0, MachineOperand::MO_VirtualRegister,
                                       addrVal);
          minstr->SetMachineOperandConst(1,MachineOperand::MO_SignExtendedImmed,
                                       zeroOffset);
          minstr->SetMachineOperandVal(2, MachineOperand::MO_VirtualRegister,
                                       dest);
          minstrVec.push_back(minstr);
        }
    }
}


// Create an instruction sequence to copy an integer value `val'
// to a floating point value `dest' by copying to memory and back.
// val must be an integral type.  dest must be a Float or Double.
// The generated instructions are returned in `minstrVec'.
// Any temp. registers (TmpInstruction) created are returned in `tempVec'.
// 
void
UltraSparcInstrInfo::CreateCodeToCopyIntToFloat(Function *F,
                                         Value* val,
                                         Instruction* dest,
                                         std::vector<MachineInstr*>& minstrVec,
                                         std::vector<TmpInstruction*>& tempVec,
                                         TargetMachine& target) const
{
  assert((val->getType()->isIntegral() || val->getType()->isPointerType())
         && "Source type must be integral");
  assert((dest->getType() == Type::FloatTy || dest->getType() == Type::DoubleTy)
         && "Dest type must be float/double");
  
  MachineCodeForMethod& mcinfo = MachineCodeForMethod::get(F);
  int offset = mcinfo.allocateLocalVar(target, val); 
  
  // Store instruction stores `val' to [%fp+offset].
  // The store and load opCodes are based on the value being copied, and
  // they use integer and float types that accomodate the
  // larger of the source type and the destination type:
  // On SparcV9: int for float, long for double.
  // 
  Type* tmpType = (dest->getType() == Type::FloatTy)? Type::IntTy
                                                    : Type::LongTy;
  MachineInstr* store = new MachineInstr(ChooseStoreInstruction(tmpType));
  store->SetMachineOperandVal(0, MachineOperand::MO_VirtualRegister, val);
  store->SetMachineOperandReg(1, target.getRegInfo().getFramePointer());
  store->SetMachineOperandConst(2,MachineOperand::MO_SignExtendedImmed, offset);
  minstrVec.push_back(store);

  // Load instruction loads [%fp+offset] to `dest'.
  // 
  MachineInstr* load =new MachineInstr(ChooseLoadInstruction(dest->getType()));
  load->SetMachineOperandReg(0, target.getRegInfo().getFramePointer());
  load->SetMachineOperandConst(1, MachineOperand::MO_SignExtendedImmed,offset);
  load->SetMachineOperandVal(2, MachineOperand::MO_VirtualRegister, dest);
  minstrVec.push_back(load);
}


// Similarly, create an instruction sequence to copy an FP value
// `val' to an integer value `dest' by copying to memory and back.
// See the previous function for information about return values.
// 
void
UltraSparcInstrInfo::CreateCodeToCopyFloatToInt(Function *F,
                                        Value* val,
                                        Instruction* dest,
                                        std::vector<MachineInstr*>& minstrVec,
                                        std::vector<TmpInstruction*>& tempVec,
                                        TargetMachine& target) const
{
  assert((val->getType() ==Type::FloatTy || val->getType() ==Type::DoubleTy)
         && "Source type must be float/double");
  assert((dest->getType()->isIntegral() || dest->getType()->isPointerType())
         && "Dest type must be integral");
  
  MachineCodeForMethod& mcinfo = MachineCodeForMethod::get(F);
  int offset = mcinfo.allocateLocalVar(target, val); 
  
  // Store instruction stores `val' to [%fp+offset].
  // The store and load opCodes are based on the value being copied, and
  // they use the integer type that matches the source type in size:
  // On SparcV9: int for float, long for double.
  // 
  Type* tmpType = (val->getType() == Type::FloatTy)? Type::IntTy
                                                   : Type::LongTy;
  MachineInstr* store=new MachineInstr(ChooseStoreInstruction(val->getType()));
  store->SetMachineOperandVal(0, MachineOperand::MO_VirtualRegister, val);
  store->SetMachineOperandReg(1, target.getRegInfo().getFramePointer());
  store->SetMachineOperandConst(2,MachineOperand::MO_SignExtendedImmed,offset);
  minstrVec.push_back(store);
  
  // Load instruction loads [%fp+offset] to `dest'.
  // 
  MachineInstr* load = new MachineInstr(ChooseLoadInstruction(tmpType));
  load->SetMachineOperandReg(0, target.getRegInfo().getFramePointer());
  load->SetMachineOperandConst(1, MachineOperand::MO_SignExtendedImmed, offset);
  load->SetMachineOperandVal(2, MachineOperand::MO_VirtualRegister, dest);
  minstrVec.push_back(load);
}
