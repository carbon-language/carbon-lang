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
#include "llvm/CodeGen/MachineCodeForMethod.h"
#include "llvm/CodeGen/MachineCodeForInstruction.h"
#include "llvm/Function.h"
#include "llvm/BasicBlock.h"
#include "llvm/Instruction.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"


//************************ Internal Functions ******************************/


static inline void
CreateIntSetInstruction(const TargetMachine& target, Function* F,
                        int64_t C, Instruction* dest,
                        std::vector<MachineInstr*>& mvec,
                        MachineCodeForInstruction& mcfi)
{
  assert(dest->getType()->isSigned() && "Use CreateUIntSetInstruction()");
  
  MachineInstr* M;
  uint64_t absC = (C >= 0)? C : -C;
  if (absC > (unsigned int) ~0)
    { // C does not fit in 32 bits
      TmpInstruction* tmpReg = new TmpInstruction(Type::IntTy);
      mcfi.addTemp(tmpReg);
      
      M = new MachineInstr(SETX);
      M->SetMachineOperandConst(0,MachineOperand::MO_SignExtendedImmed,C);
      M->SetMachineOperandVal(1, MachineOperand::MO_VirtualRegister, tmpReg,
                                 /*isdef*/ true);
      M->SetMachineOperandVal(2, MachineOperand::MO_VirtualRegister,dest);
      mvec.push_back(M);
    }
  else
    {
      M = Create2OperandInstr_SImmed(SETSW, C, dest);
      mvec.push_back(M);
    }
}

static inline void
CreateUIntSetInstruction(const TargetMachine& target, Function* F,
                         uint64_t C, Instruction* dest,
                         std::vector<MachineInstr*>& mvec,
                         MachineCodeForInstruction& mcfi)
{
  assert(! dest->getType()->isSigned() && "Use CreateIntSetInstruction()");
  unsigned destSize = target.DataLayout.getTypeSize(dest->getType());
  MachineInstr* M;
  
  if (C > (unsigned int) ~0)
    { // C does not fit in 32 bits
      assert(dest->getType() == Type::ULongTy && "Sign extension problems");
      TmpInstruction *tmpReg = new TmpInstruction(Type::IntTy);
      mcfi.addTemp(tmpReg);
      
      M = new MachineInstr(SETX);
      M->SetMachineOperandConst(0, MachineOperand::MO_UnextendedImmed, C);
      M->SetMachineOperandVal(1, MachineOperand::MO_VirtualRegister, tmpReg,
                              /*isdef*/ true);
      M->SetMachineOperandVal(2, MachineOperand::MO_VirtualRegister, dest);
      mvec.push_back(M);
    }
  else
    {
      // If the destination is smaller than the standard integer reg. size,
      // we have to extend the sign-bit into upper bits of dest, so we
      // need to put the result of the SETUW into a temporary.
      // 
      Value* setuwDest = dest;
      if (destSize < target.DataLayout.getIntegerRegize())
        {
          setuwDest = new TmpInstruction(dest, NULL, "setTmp");
          mcfi.addTemp(setuwDest);
        }
      
      M = Create2OperandInstr_UImmed(SETUW, C, setuwDest);
      mvec.push_back(M);
      
      if (setuwDest != dest)
        { // extend the sign-bit of the result into all upper bits of dest
          assert(8*destSize <= 32 &&
                 "Unexpected type size > 4 and < IntRegSize?");
          target.getInstrInfo().
            CreateSignExtensionInstructions(target, F,
                                            setuwDest, 8*destSize, dest,
                                            mvec, mcfi);
        }
    }
  
#define USE_DIRECT_SIGN_EXTENSION_INSTRS
#ifndef USE_DIRECT_SIGN_EXTENSION_INSTRS
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
#endif USE_DIRECT_SIGN_EXTENSION_INSTRS
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
// The generated instructions are returned in `mvec'.
// Any temp. registers (TmpInstruction) created are recorded in mcfi.
// Any stack space required is allocated via MachineCodeForMethod.
// 
void
UltraSparcInstrInfo::CreateCodeToLoadConst(const TargetMachine& target,
                                           Function* F,
                                           Value* val,
                                           Instruction* dest,
                                           std::vector<MachineInstr*>& mvec,
                                       MachineCodeForInstruction& mcfi) const
{
  assert(isa<Constant>(val) || isa<GlobalValue>(val) &&
         "I only know about constant values and global addresses");
  
  // Use a "set" instruction for known constants that can go in an integer reg.
  // Use a "load" instruction for all other constants, in particular,
  // floating point constants and addresses of globals.
  // 
  const Type* valType = val->getType();
  
  if (valType->isIntegral() || valType == Type::BoolTy)
    {
      if (! val->getType()->isSigned())
        {
          uint64_t C = cast<ConstantUInt>(val)->getValue();
          CreateUIntSetInstruction(target, F, C, dest, mvec, mcfi);
        }
      else
        {
          bool isValidConstant;
          int64_t C = GetConstantValueAsSignedInt(val, isValidConstant);
          assert(isValidConstant && "Unrecognized constant");
          CreateIntSetInstruction(target, F, C, dest, mvec, mcfi);
        }
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
      mcfi.addTemp(tmpReg);
      
      if (isa<Constant>(val))
        {
          // Create another TmpInstruction for the hidden integer register
          TmpInstruction* addrReg =
            new TmpInstruction(PointerType::get(val->getType()), val);
          mcfi.addTemp(addrReg);
          addrVal = addrReg;
        }
      else
        addrVal = dest;
      
      MachineInstr* M = new MachineInstr(SETX);
      M->SetMachineOperandVal(0, MachineOperand::MO_PCRelativeDisp, val);
      M->SetMachineOperandVal(1, MachineOperand::MO_VirtualRegister, tmpReg,
                              /*isdef*/ true);
      M->SetMachineOperandVal(2, MachineOperand::MO_VirtualRegister, addrVal);
      mvec.push_back(M);
      
      if (isa<Constant>(val))
        {
          // Make sure constant is emitted to constant pool in assembly code.
          MachineCodeForMethod::get(F).addToConstantPool(cast<Constant>(val));
          
          // Generate the load instruction
          M = Create3OperandInstr_SImmed(ChooseLoadInstruction(val->getType()),
                                         addrVal, zeroOffset, dest);
          mvec.push_back(M);
        }
    }
}


// Create an instruction sequence to copy an integer value `val'
// to a floating point value `dest' by copying to memory and back.
// val must be an integral type.  dest must be a Float or Double.
// The generated instructions are returned in `mvec'.
// Any temp. registers (TmpInstruction) created are recorded in mcfi.
// Any stack space required is allocated via MachineCodeForMethod.
// 
void
UltraSparcInstrInfo::CreateCodeToCopyIntToFloat(const TargetMachine& target,
                                        Function* F,
                                        Value* val,
                                        Instruction* dest,
                                        std::vector<MachineInstr*>& mvec,
                                        MachineCodeForInstruction& mcfi) const
{
  assert((val->getType()->isIntegral() || isa<PointerType>(val->getType()))
         && "Source type must be integral");
  assert(dest->getType()->isFloatingPoint()
         && "Dest type must be float/double");
  
  int offset = MachineCodeForMethod::get(F).allocateLocalVar(target, val); 
  
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
  store->SetMachineOperandConst(2,MachineOperand::MO_SignExtendedImmed,offset);
  mvec.push_back(store);

  // Load instruction loads [%fp+offset] to `dest'.
  // 
  MachineInstr* load =new MachineInstr(ChooseLoadInstruction(dest->getType()));
  load->SetMachineOperandReg(0, target.getRegInfo().getFramePointer());
  load->SetMachineOperandConst(1, MachineOperand::MO_SignExtendedImmed,offset);
  load->SetMachineOperandVal(2, MachineOperand::MO_VirtualRegister, dest);
  mvec.push_back(load);
}


// Similarly, create an instruction sequence to copy an FP value
// `val' to an integer value `dest' by copying to memory and back.
// The generated instructions are returned in `mvec'.
// Any temp. registers (TmpInstruction) created are recorded in mcfi.
// Any stack space required is allocated via MachineCodeForMethod.
// 
void
UltraSparcInstrInfo::CreateCodeToCopyFloatToInt(const TargetMachine& target,
                                        Function* F,
                                        Value* val,
                                        Instruction* dest,
                                        std::vector<MachineInstr*>& mvec,
                                        MachineCodeForInstruction& mcfi) const
{
  assert(val->getType()->isFloatingPoint()
         && "Source type must be float/double");
  assert((dest->getType()->isIntegral() || isa<PointerType>(dest->getType()))
         && "Dest type must be integral");
  
  int offset = MachineCodeForMethod::get(F).allocateLocalVar(target, val); 
  
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
  mvec.push_back(store);
  
  // Load instruction loads [%fp+offset] to `dest'.
  // 
  MachineInstr* load = new MachineInstr(ChooseLoadInstruction(tmpType));
  load->SetMachineOperandReg(0, target.getRegInfo().getFramePointer());
  load->SetMachineOperandConst(1, MachineOperand::MO_SignExtendedImmed,offset);
  load->SetMachineOperandVal(2, MachineOperand::MO_VirtualRegister, dest);
  mvec.push_back(load);
}


// Create instruction(s) to copy src to dest, for arbitrary types
// The generated instructions are returned in `mvec'.
// Any temp. registers (TmpInstruction) created are recorded in mcfi.
// Any stack space required is allocated via MachineCodeForMethod.
// 
void
UltraSparcInstrInfo::CreateCopyInstructionsByType(const TargetMachine& target,
                                                  Function *F,
                                                  Value* src,
                                                  Instruction* dest,
                                                  vector<MachineInstr*>& mvec,
                                          MachineCodeForInstruction& mcfi) const
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
      target.getInstrInfo().CreateCodeToLoadConst(target, F, src, dest,
                                                  mvec, mcfi);
    }
  else
    { // Create an add-with-0 instruction of the appropriate type.
      // Make `src' the second operand, in case it is a constant
      // Use (unsigned long) 0 for a NULL pointer value.
      // 
      const Type* zeroValueType =
        isa<PointerType>(resultType) ? Type::ULongTy : resultType;
      MachineInstr* minstr =
        Create3OperandInstr(opCode, Constant::getNullValue(zeroValueType),
                            src, dest);
      mvec.push_back(minstr);
    }
}


// Create instruction sequence to produce a sign-extended register value
// from an arbitrary sized value (sized in bits, not bytes).
// For SPARC v9, we sign-extend the given unsigned operand using SLL; SRA.
// The generated instructions are returned in `mvec'.
// Any temp. registers (TmpInstruction) created are recorded in mcfi.
// Any stack space required is allocated via MachineCodeForMethod.
// 
void
UltraSparcInstrInfo::CreateSignExtensionInstructions(
                                        const TargetMachine& target,
                                        Function* F,
                                        Value* unsignedSrcVal,
                                        unsigned int srcSizeInBits,
                                        Value* dest,
                                        vector<MachineInstr*>& mvec,
                                        MachineCodeForInstruction& mcfi) const
{
  MachineInstr* M;
  
  assert(srcSizeInBits > 0 && srcSizeInBits <= 32
     && "Hmmm... srcSizeInBits > 32 unexpected but could be handled here.");
  
  if (srcSizeInBits < 32)
    { // SLL is needed since operand size is < 32 bits.
      TmpInstruction *tmpI = new TmpInstruction(dest->getType(),
                                                unsignedSrcVal, dest,"make32");
      mcfi.addTemp(tmpI);
      M = Create3OperandInstr_UImmed(SLL,unsignedSrcVal,32-srcSizeInBits,tmpI);
      mvec.push_back(M);
      unsignedSrcVal = tmpI;
    }
  
  M = Create3OperandInstr_UImmed(SRA, unsignedSrcVal, 32-srcSizeInBits, dest);
  mvec.push_back(M);
}
