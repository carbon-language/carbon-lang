//===-- InstSelectSimple.cpp - A simple instruction selector for SparcV8 --===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines a simple peephole instruction selector for the V8 target
//
//===----------------------------------------------------------------------===//

#include "SparcV8.h"
#include "SparcV8InstrInfo.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicLowering.h"
#include "llvm/Pass.h"
#include "llvm/Constants.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/Support/InstVisitor.h"
#include "llvm/Support/CFG.h"
using namespace llvm;

namespace {
  struct V8ISel : public FunctionPass, public InstVisitor<V8ISel> {
    TargetMachine &TM;
    MachineFunction *F;                 // The function we are compiling into
    MachineBasicBlock *BB;              // The current MBB we are compiling

    std::map<Value*, unsigned> RegMap;  // Mapping between Val's and SSA Regs

    // MBBMap - Mapping between LLVM BB -> Machine BB
    std::map<const BasicBlock*, MachineBasicBlock*> MBBMap;

    V8ISel(TargetMachine &tm) : TM(tm), F(0), BB(0) {}

    /// runOnFunction - Top level implementation of instruction selection for
    /// the entire function.
    ///
    bool runOnFunction(Function &Fn);

    virtual const char *getPassName() const {
      return "SparcV8 Simple Instruction Selection";
    }

    /// visitBasicBlock - This method is called when we are visiting a new basic
    /// block.  This simply creates a new MachineBasicBlock to emit code into
    /// and adds it to the current MachineFunction.  Subsequent visit* for
    /// instructions will be invoked for all instructions in the basic block.
    ///
    void visitBasicBlock(BasicBlock &LLVM_BB) {
      BB = MBBMap[&LLVM_BB];
    }

	void visitBinaryOperator(BinaryOperator &I);
	void visitCallInst(CallInst &I);
	void visitReturnInst(ReturnInst &RI);

    void visitInstruction(Instruction &I) {
      std::cerr << "Unhandled instruction: " << I;
      abort();
    }

    /// LowerUnknownIntrinsicFunctionCalls - This performs a prepass over the
    /// function, lowering any calls to unknown intrinsic functions into the
    /// equivalent LLVM code.
    void LowerUnknownIntrinsicFunctionCalls(Function &F);
    void visitIntrinsicCall(Intrinsic::ID ID, CallInst &CI);

    /// copyConstantToRegister - Output the instructions required to put the
    /// specified constant into the specified register.
    ///
    void copyConstantToRegister(MachineBasicBlock *MBB,
                                MachineBasicBlock::iterator IP,
                                Constant *C, unsigned R);

    /// makeAnotherReg - This method returns the next register number we haven't
    /// yet used.
    ///
    /// Long values are handled somewhat specially.  They are always allocated
    /// as pairs of 32 bit integer values.  The register number returned is the
    /// lower 32 bits of the long value, and the regNum+1 is the upper 32 bits
    /// of the long value.
    ///
    unsigned makeAnotherReg(const Type *Ty) {
      assert(dynamic_cast<const SparcV8RegisterInfo*>(TM.getRegisterInfo()) &&
             "Current target doesn't have SparcV8 reg info??");
      const SparcV8RegisterInfo *MRI =
        static_cast<const SparcV8RegisterInfo*>(TM.getRegisterInfo());
      if (Ty == Type::LongTy || Ty == Type::ULongTy) {
        const TargetRegisterClass *RC = MRI->getRegClassForType(Type::IntTy);
        // Create the lower part
        F->getSSARegMap()->createVirtualRegister(RC);
        // Create the upper part.
        return F->getSSARegMap()->createVirtualRegister(RC)-1;
      }

      // Add the mapping of regnumber => reg class to MachineFunction
      const TargetRegisterClass *RC = MRI->getRegClassForType(Ty);
      return F->getSSARegMap()->createVirtualRegister(RC);
    }

    unsigned getReg(Value &V) { return getReg (&V); } // allow refs.
    unsigned getReg(Value *V) {
      // Just append to the end of the current bb.
      MachineBasicBlock::iterator It = BB->end();
      return getReg(V, BB, It);
    }
    unsigned getReg(Value *V, MachineBasicBlock *MBB,
                    MachineBasicBlock::iterator IPt) {
      unsigned &Reg = RegMap[V];
      if (Reg == 0) {
        Reg = makeAnotherReg(V->getType());
        RegMap[V] = Reg;
      }
      // If this operand is a constant, emit the code to copy the constant into
      // the register here...
      //
      if (Constant *C = dyn_cast<Constant>(V)) {
        copyConstantToRegister(MBB, IPt, C, Reg);
        RegMap.erase(V);  // Assign a new name to this constant if ref'd again
      } else if (GlobalValue *GV = dyn_cast<GlobalValue>(V)) {
        // Move the address of the global into the register
        unsigned TmpReg = makeAnotherReg(V->getType());
        BuildMI (*MBB, IPt, V8::SETHIi, 1, TmpReg).addGlobalAddress (GV);
        BuildMI (*MBB, IPt, V8::ORri, 2, Reg).addReg (TmpReg)
          .addGlobalAddress (GV);
        RegMap.erase(V);  // Assign a new name to this address if ref'd again
      }

      return Reg;
    }

  };
}

FunctionPass *llvm::createSparcV8SimpleInstructionSelector(TargetMachine &TM) {
  return new V8ISel(TM);
}

enum TypeClass {
  cByte, cShort, cInt, cLong, cFloat, cDouble
};

static TypeClass getClass (const Type *T) {
  switch (T->getPrimitiveID ()) {
    case Type::UByteTyID:  case Type::SByteTyID:  return cByte;
    case Type::UShortTyID: case Type::ShortTyID:  return cShort;
    case Type::UIntTyID:   case Type::IntTyID:    return cInt;
    case Type::ULongTyID:  case Type::LongTyID:   return cLong;
    case Type::FloatTyID:                         return cFloat;
    case Type::DoubleTyID:                        return cDouble;
    default:
      assert (0 && "Type of unknown class passed to getClass?");
      return cByte;
  }
}

/// copyConstantToRegister - Output the instructions required to put the
/// specified constant into the specified register.
///
void V8ISel::copyConstantToRegister(MachineBasicBlock *MBB,
                                    MachineBasicBlock::iterator IP,
                                    Constant *C, unsigned R) {
  if (ConstantInt *CI = dyn_cast<ConstantInt> (C)) {
    unsigned Class = getClass(C->getType());
    switch (Class) {
      case cByte:
        BuildMI (*MBB, IP, V8::ORri, 2, R).addReg (V8::G0).addImm ((uint8_t) CI->getRawValue ());
        return;
      case cShort: {
        unsigned TmpReg = makeAnotherReg (C->getType ());
        BuildMI (*MBB, IP, V8::SETHIi, 1, TmpReg).addImm (((uint16_t) CI->getRawValue ()) >> 10);
        BuildMI (*MBB, IP, V8::ORri, 2, R).addReg (TmpReg).addImm (((uint16_t) CI->getRawValue ()) & 0x03ff);
        return;
      }
      case cInt: {
        unsigned TmpReg = makeAnotherReg (C->getType ());
        BuildMI (*MBB, IP, V8::SETHIi, 1, TmpReg).addImm (((uint32_t) CI->getRawValue ()) >> 10);
        BuildMI (*MBB, IP, V8::ORri, 2, R).addReg (TmpReg).addImm (((uint32_t) CI->getRawValue ()) & 0x03ff);
        return;
      }
      case cLong: {
        unsigned TmpReg = makeAnotherReg (Type::UIntTy);
        uint32_t topHalf, bottomHalf;
        topHalf = (uint32_t) (CI->getRawValue () >> 32);
        bottomHalf = (uint32_t) (CI->getRawValue () & 0x0ffffffffULL);
        BuildMI (*MBB, IP, V8::SETHIi, 1, TmpReg).addImm (topHalf >> 10);
        BuildMI (*MBB, IP, V8::ORri, 2, R).addReg (TmpReg).addImm (topHalf & 0x03ff);
        BuildMI (*MBB, IP, V8::SETHIi, 1, TmpReg).addImm (bottomHalf >> 10);
        BuildMI (*MBB, IP, V8::ORri, 2, R).addReg (TmpReg).addImm (bottomHalf & 0x03ff);
        return;
      }
      default:
        std::cerr << "Offending constant: " << *C << "\n";
        assert (0 && "Can't copy this kind of constant into register yet");
        return;
    }
  }

  std::cerr << "Offending constant: " << *C << "\n";
  assert (0 && "Can't copy this kind of constant into register yet");
}

bool V8ISel::runOnFunction(Function &Fn) {
  // First pass over the function, lower any unknown intrinsic functions
  // with the IntrinsicLowering class.
  LowerUnknownIntrinsicFunctionCalls(Fn);
  
  F = &MachineFunction::construct(&Fn, TM);
  
  // Create all of the machine basic blocks for the function...
  for (Function::iterator I = Fn.begin(), E = Fn.end(); I != E; ++I)
    F->getBasicBlockList().push_back(MBBMap[I] = new MachineBasicBlock(I));
  
  BB = &F->front();
  
  // Set up a frame object for the return address.  This is used by the
  // llvm.returnaddress & llvm.frameaddress intrinisics.
  //ReturnAddressIndex = F->getFrameInfo()->CreateFixedObject(4, -4);
  
  // Copy incoming arguments off of the stack and out of fixed registers.
  //LoadArgumentsToVirtualRegs(Fn);
  
  // Instruction select everything except PHI nodes
  visit(Fn);
  
  // Select the PHI nodes
  //SelectPHINodes();
  
  RegMap.clear();
  MBBMap.clear();
  F = 0;
  // We always build a machine code representation for the function
  return true;
}

void V8ISel::visitCallInst(CallInst &I) {
  assert (I.getNumOperands () == 1 && "Can't handle call args yet");
  unsigned DestReg = getReg (I);
  BuildMI (BB, V8::CALL, 1).addPCDisp (I.getOperand (0));
  if (I.getType ()->getPrimitiveID () == Type::VoidTyID)
    return;
  // Deal w/ return value
  switch (getClass (I.getType ())) {
    case cByte:
    case cShort:
    case cInt:
      // Schlep it over into the destination register
      BuildMI (BB, V8::ORrr, 2, DestReg).addReg(V8::G0).addReg(V8::O0);
      break;
    default:
      visitInstruction (I);
      return;
  }
}

void V8ISel::visitReturnInst(ReturnInst &I) {
  if (I.getNumOperands () == 1) {
    unsigned RetValReg = getReg (I.getOperand (0));
    switch (getClass (I.getOperand (0)->getType ())) {
      case cByte:
      case cShort:
      case cInt:
        // Schlep it over into i0 (where it will become o0 after restore).
        BuildMI (BB, V8::ORrr, 2, V8::I0).addReg(V8::G0).addReg(RetValReg);
        break;
      default:
        visitInstruction (I);
        return;
    }
  } else if (I.getNumOperands () != 1) {
    visitInstruction (I);
  }
  // Just emit a 'retl' instruction to return.
  BuildMI(BB, V8::RETL, 0);
  return;
}

void V8ISel::visitBinaryOperator (BinaryOperator &I) {
  unsigned DestReg = getReg (I);
  unsigned Op0Reg = getReg (I.getOperand (0));
  unsigned Op1Reg = getReg (I.getOperand (1));

  unsigned ResultReg = makeAnotherReg (I.getType ());
  unsigned OpCase = ~0;

  // FIXME: support long, ulong, fp.
  switch (I.getOpcode ()) {
  case Instruction::Add: OpCase = 0; break;
  case Instruction::Sub: OpCase = 1; break;
  case Instruction::Mul: OpCase = 2; break;
  case Instruction::And: OpCase = 3; break;
  case Instruction::Or:  OpCase = 4; break;
  case Instruction::Xor: OpCase = 5; break;

  case Instruction::Div:
  case Instruction::Rem: {
    unsigned Dest = ResultReg;
    if (I.getOpcode() == Instruction::Rem)
      Dest = makeAnotherReg(I.getType());

    // FIXME: this is probably only right for 32 bit operands.
    if (I.getType ()->isSigned()) {
      unsigned Tmp = makeAnotherReg (I.getType ());
      // Sign extend into the Y register
      BuildMI (BB, V8::SRAri, 2, Tmp).addReg (Op0Reg).addZImm (31);
      BuildMI (BB, V8::WRrr, 2, V8::Y).addReg (Tmp).addReg (V8::G0);
      BuildMI (BB, V8::SDIVrr, 2, Dest).addReg (Op0Reg).addReg (Op1Reg);
    } else {
      // Zero extend into the Y register, ie, just set it to zero
      BuildMI (BB, V8::WRrr, 2, V8::Y).addReg (V8::G0).addReg (V8::G0);
      BuildMI (BB, V8::UDIVrr, 2, Dest).addReg (Op0Reg).addReg (Op1Reg);
    }

    if (I.getOpcode() == Instruction::Rem) {
      unsigned Tmp = makeAnotherReg (I.getType ());
      BuildMI (BB, V8::SMULrr, 2, Tmp).addReg(Dest).addReg(Op1Reg);
      BuildMI (BB, V8::SUBrr, 2, ResultReg).addReg(Op0Reg).addReg(Tmp);
    }
    break;
  }
  default:
    visitInstruction (I);
    return;
  }

  if (OpCase != ~0U) {
    static const unsigned Opcodes[] = {
      V8::ADDrr, V8::SUBrr, V8::SMULrr, V8::ANDrr, V8::ORrr, V8::XORrr
    };
    BuildMI (BB, Opcodes[OpCase], 2, ResultReg).addReg (Op0Reg).addReg (Op1Reg);
  }

  switch (getClass (I.getType ())) {
    case cByte: 
      if (I.getType ()->isSigned ()) { // add byte
        BuildMI (BB, V8::ANDri, 2, DestReg).addReg (ResultReg).addZImm (0xff);
      } else { // add ubyte
        unsigned TmpReg = makeAnotherReg (I.getType ());
        BuildMI (BB, V8::SLLri, 2, TmpReg).addReg (ResultReg).addZImm (24);
        BuildMI (BB, V8::SRAri, 2, DestReg).addReg (TmpReg).addZImm (24);
      }
      break;
    case cShort:
      if (I.getType ()->isSigned ()) { // add short
        unsigned TmpReg = makeAnotherReg (I.getType ());
        BuildMI (BB, V8::SLLri, 2, TmpReg).addReg (ResultReg).addZImm (16);
        BuildMI (BB, V8::SRAri, 2, DestReg).addReg (TmpReg).addZImm (16);
      } else { // add ushort
        unsigned TmpReg = makeAnotherReg (I.getType ());
        BuildMI (BB, V8::SLLri, 2, TmpReg).addReg (ResultReg).addZImm (16);
        BuildMI (BB, V8::SRLri, 2, DestReg).addReg (TmpReg).addZImm (16);
      }
      break;
    case cInt:
      BuildMI (BB, V8::ORrr, 2, DestReg).addReg (V8::G0).addReg (ResultReg);
      break;
    default:
      visitInstruction (I);
      return;
  }
}


/// LowerUnknownIntrinsicFunctionCalls - This performs a prepass over the
/// function, lowering any calls to unknown intrinsic functions into the
/// equivalent LLVM code.
void V8ISel::LowerUnknownIntrinsicFunctionCalls(Function &F) {
  for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB)
    for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; )
      if (CallInst *CI = dyn_cast<CallInst>(I++))
        if (Function *F = CI->getCalledFunction())
          switch (F->getIntrinsicID()) {
          case Intrinsic::not_intrinsic: break;
          default:
            // All other intrinsic calls we must lower.
            Instruction *Before = CI->getPrev();
            TM.getIntrinsicLowering().LowerIntrinsicCall(CI);
            if (Before) {        // Move iterator to instruction after call
              I = Before;  ++I;
            } else {
              I = BB->begin();
            }
          }
}


void V8ISel::visitIntrinsicCall(Intrinsic::ID ID, CallInst &CI) {
  unsigned TmpReg1, TmpReg2;
  switch (ID) {
  default: assert(0 && "Intrinsic not supported!");
  }
}
