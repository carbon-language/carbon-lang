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
#include "llvm/Support/Debug.h"
#include "llvm/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Constants.h"
#include "llvm/CodeGen/IntrinsicLowering.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineConstantPool.h"
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

    /// emitGEPOperation - Common code shared between visitGetElementPtrInst and
    /// constant expression GEP support.
    ///
    void emitGEPOperation(MachineBasicBlock *BB, MachineBasicBlock::iterator IP,
                          Value *Src, User::op_iterator IdxBegin,
                          User::op_iterator IdxEnd, unsigned TargetReg);

    /// emitCastOperation - Common code shared between visitCastInst and
    /// constant expression cast support.
    ///
    void emitCastOperation(MachineBasicBlock *BB,MachineBasicBlock::iterator IP,
                           Value *Src, const Type *DestTy, unsigned TargetReg);

    /// emitIntegerCast, emitFPToIntegerCast - Helper methods for
    /// emitCastOperation.
    ///
    void emitIntegerCast (MachineBasicBlock *BB, MachineBasicBlock::iterator IP,
                          const Type *oldTy, unsigned SrcReg, const Type *newTy,
                          unsigned DestReg);
    void emitFPToIntegerCast (MachineBasicBlock *BB,
                              MachineBasicBlock::iterator IP, const Type *oldTy,
                              unsigned SrcReg, const Type *newTy,
                              unsigned DestReg);

    /// visitBasicBlock - This method is called when we are visiting a new basic
    /// block.  This simply creates a new MachineBasicBlock to emit code into
    /// and adds it to the current MachineFunction.  Subsequent visit* for
    /// instructions will be invoked for all instructions in the basic block.
    ///
    void visitBasicBlock(BasicBlock &LLVM_BB) {
      BB = MBBMap[&LLVM_BB];
    }

    void visitBinaryOperator(Instruction &I);
    void visitShiftInst (ShiftInst &SI) { visitBinaryOperator (SI); }
    void visitSetCondInst(SetCondInst &I);
    void visitCallInst(CallInst &I);
    void visitReturnInst(ReturnInst &I);
    void visitBranchInst(BranchInst &I);
    void visitUnreachableInst(UnreachableInst &I) {}
    void visitCastInst(CastInst &I);
    void visitLoadInst(LoadInst &I);
    void visitStoreInst(StoreInst &I);
    void visitPHINode(PHINode &I) {}      // PHI nodes handled by second pass
    void visitGetElementPtrInst(GetElementPtrInst &I);
    void visitAllocaInst(AllocaInst &I);

    void visitInstruction(Instruction &I) {
      std::cerr << "Unhandled instruction: " << I;
      abort();
    }

    /// LowerUnknownIntrinsicFunctionCalls - This performs a prepass over the
    /// function, lowering any calls to unknown intrinsic functions into the
    /// equivalent LLVM code.
    void LowerUnknownIntrinsicFunctionCalls(Function &F);
    void visitIntrinsicCall(Intrinsic::ID ID, CallInst &CI);

    void LoadArgumentsToVirtualRegs(Function *F);

    /// SelectPHINodes - Insert machine code to generate phis.  This is tricky
    /// because we have to generate our sources into the source basic blocks,
    /// not the current one.
    ///
    void SelectPHINodes();

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
  switch (T->getTypeID()) {
    case Type::UByteTyID:  case Type::SByteTyID:  return cByte;
    case Type::UShortTyID: case Type::ShortTyID:  return cShort;
    case Type::PointerTyID:
    case Type::UIntTyID:   case Type::IntTyID:    return cInt;
    case Type::ULongTyID:  case Type::LongTyID:   return cLong;
    case Type::FloatTyID:                         return cFloat;
    case Type::DoubleTyID:                        return cDouble;
    default:
      assert (0 && "Type of unknown class passed to getClass?");
      return cByte;
  }
}

static TypeClass getClassB(const Type *T) {
  if (T == Type::BoolTy) return cByte;
  return getClass(T);
}

/// copyConstantToRegister - Output the instructions required to put the
/// specified constant into the specified register.
///
void V8ISel::copyConstantToRegister(MachineBasicBlock *MBB,
                                    MachineBasicBlock::iterator IP,
                                    Constant *C, unsigned R) {
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
    switch (CE->getOpcode()) {
    case Instruction::GetElementPtr:
      emitGEPOperation(MBB, IP, CE->getOperand(0),
                       CE->op_begin()+1, CE->op_end(), R);
      return;
    case Instruction::Cast:
      emitCastOperation(MBB, IP, CE->getOperand(0), CE->getType(), R);
      return;
    default:
      std::cerr << "Copying this constant expr not yet handled: " << *CE;
      abort();
    }
  } else if (isa<UndefValue>(C)) {
    BuildMI(*MBB, IP, V8::IMPLICIT_DEF, 0, R);
    if (getClassB (C->getType ()) == cLong)
      BuildMI(*MBB, IP, V8::IMPLICIT_DEF, 0, R+1);
    return;
  }

  if (C->getType()->isIntegral ()) {
    uint64_t Val;
    unsigned Class = getClassB (C->getType ());
    if (Class == cLong) {
      unsigned TmpReg = makeAnotherReg (Type::IntTy);
      unsigned TmpReg2 = makeAnotherReg (Type::IntTy);
      // Copy the value into the register pair.
      // R = top(more-significant) half, R+1 = bottom(less-significant) half
      uint64_t Val = cast<ConstantInt>(C)->getRawValue();
      copyConstantToRegister(MBB, IP, ConstantUInt::get(Type::UIntTy,
                             Val >> 32), R);
      copyConstantToRegister(MBB, IP, ConstantUInt::get(Type::UIntTy,
                             Val & 0xffffffffU), R+1);
      return;
    }

    assert(Class <= cInt && "Type not handled yet!");

    if (C->getType() == Type::BoolTy) {
      Val = (C == ConstantBool::True);
    } else {
      ConstantInt *CI = cast<ConstantInt> (C);
      Val = CI->getRawValue ();
    }
    switch (Class) {
      case cByte:  Val =  (int8_t) Val; break;
      case cShort: Val = (int16_t) Val; break;
      case cInt:   Val = (int32_t) Val; break;
      default:
        std::cerr << "Offending constant: " << *C << "\n";
        assert (0 && "Can't copy this kind of constant into register yet");
        return;
    }
    if (Val == 0) {
      BuildMI (*MBB, IP, V8::ORrr, 2, R).addReg (V8::G0).addReg(V8::G0);
    } else if (((int64_t)Val >= -4096) && ((int64_t)Val <= 4095)) {
      BuildMI (*MBB, IP, V8::ORri, 2, R).addReg (V8::G0).addSImm(Val);
    } else {
      unsigned TmpReg = makeAnotherReg (C->getType ());
      BuildMI (*MBB, IP, V8::SETHIi, 1, TmpReg)
        .addSImm (((uint32_t) Val) >> 10);
      BuildMI (*MBB, IP, V8::ORri, 2, R).addReg (TmpReg)
        .addSImm (((uint32_t) Val) & 0x03ff);
      return;
    }
  } else if (ConstantFP *CFP = dyn_cast<ConstantFP>(C)) {
    // We need to spill the constant to memory...
    MachineConstantPool *CP = F->getConstantPool();
    unsigned CPI = CP->getConstantPoolIndex(CFP);
    const Type *Ty = CFP->getType();
    unsigned TmpReg = makeAnotherReg (Type::UIntTy);
    unsigned AddrReg = makeAnotherReg (Type::UIntTy);

    assert(Ty == Type::FloatTy || Ty == Type::DoubleTy && "Unknown FP type!");
    unsigned LoadOpcode = Ty == Type::FloatTy ? V8::LDFri : V8::LDDFri;
    BuildMI (*MBB, IP, V8::SETHIi, 1, TmpReg).addConstantPoolIndex (CPI);
    BuildMI (*MBB, IP, V8::ORri, 2, AddrReg).addReg (TmpReg)
      .addConstantPoolIndex (CPI);
    BuildMI (*MBB, IP, LoadOpcode, 2, R).addReg (AddrReg).addSImm (0);
  } else if (isa<ConstantPointerNull>(C)) {
    // Copy zero (null pointer) to the register.
    BuildMI (*MBB, IP, V8::ORri, 2, R).addReg (V8::G0).addSImm (0);
  } else if (GlobalValue *GV = dyn_cast<GlobalValue>(C)) {
    // Copy it with a SETHI/OR pair; the JIT + asmwriter should recognize
    // that SETHI %reg,global == SETHI %reg,%hi(global) and 
    // OR %reg,global,%reg == OR %reg,%lo(global),%reg.
    unsigned TmpReg = makeAnotherReg (C->getType ());
    BuildMI (*MBB, IP, V8::SETHIi, 1, TmpReg).addGlobalAddress(GV);
    BuildMI (*MBB, IP, V8::ORri, 2, R).addReg(TmpReg).addGlobalAddress(GV);
  } else {
    std::cerr << "Offending constant: " << *C << "\n";
    assert (0 && "Can't copy this kind of constant into register yet");
  }
}

void V8ISel::LoadArgumentsToVirtualRegs (Function *LF) {
  static const unsigned IncomingArgRegs[] = { V8::I0, V8::I1, V8::I2,
    V8::I3, V8::I4, V8::I5 };

  // Add IMPLICIT_DEFs of input regs.
  unsigned ArgNo = 0;
  for (Function::aiterator I = LF->abegin(), E = LF->aend();
       I != E && ArgNo < 6; ++I, ++ArgNo) {
    switch (getClassB(I->getType())) {
    case cByte:
    case cShort:
    case cInt:
    case cFloat:
      BuildMI(BB, V8::IMPLICIT_DEF, 0, IncomingArgRegs[ArgNo]);
      break;
    case cDouble:
    case cLong:
      // Double and Long use register pairs.
      BuildMI(BB, V8::IMPLICIT_DEF, 0, IncomingArgRegs[ArgNo]);
      ++ArgNo;
      if (ArgNo < 6)
        BuildMI(BB, V8::IMPLICIT_DEF, 0, IncomingArgRegs[ArgNo]);
      break;
    default:
      assert (0 && "type not handled");
      return;
    }
  }

  // Copy args out of their incoming hard regs or stack slots into virtual regs.
  const unsigned *IAREnd = &IncomingArgRegs[6];
  const unsigned *IAR = &IncomingArgRegs[0];
  unsigned ArgOffset = 68;
  for (Function::aiterator I = LF->abegin(), E = LF->aend(); I != E; ++I) {
    Argument &A = *I;
    unsigned ArgReg = getReg (A);
    if (getClassB (A.getType ()) < cLong) {
      // Get it out of the incoming arg register
      if (ArgOffset < 92) {
        assert (IAR != IAREnd
                && "About to dereference past end of IncomingArgRegs");
        BuildMI (BB, V8::ORrr, 2, ArgReg).addReg (V8::G0).addReg (*IAR++);
      } else {
        int FI = F->getFrameInfo()->CreateFixedObject(4, ArgOffset);
        BuildMI (BB, V8::LD, 3, ArgReg).addFrameIndex (FI).addSImm (0);
      }
      ArgOffset += 4;
    } else if (getClassB (A.getType ()) == cFloat) {
      if (ArgOffset < 92) {
        // Single-fp args are passed in integer registers; go through
        // memory to get them out of integer registers and back into fp. (Bleh!)
        unsigned FltAlign = TM.getTargetData().getFloatAlignment();
        int FI = F->getFrameInfo()->CreateStackObject(4, FltAlign);
        assert (IAR != IAREnd
                && "About to dereference past end of IncomingArgRegs");
        BuildMI (BB, V8::ST, 3).addFrameIndex (FI).addSImm (0).addReg (*IAR++);
        BuildMI (BB, V8::LDFri, 2, ArgReg).addFrameIndex (FI).addSImm (0);
      } else {
        int FI = F->getFrameInfo()->CreateFixedObject(4, ArgOffset);
        BuildMI (BB, V8::LDFri, 3, ArgReg).addFrameIndex (FI).addSImm (0);
      }
      ArgOffset += 4;
    } else if (getClassB (A.getType ()) == cDouble) {
      // Double-fp args are passed in pairs of integer registers; go through
      // memory to get them out of integer registers and back into fp. (Bleh!)
      // We'd like to 'ldd' these right out of the incoming-args area,
      // but it might not be 8-byte aligned (e.g., call x(int x, double d)).
      unsigned DblAlign = TM.getTargetData().getDoubleAlignment();
      int FI = F->getFrameInfo()->CreateStackObject(8, DblAlign);
      if (ArgOffset < 92 && IAR != IAREnd) {
        BuildMI (BB, V8::ST, 3).addFrameIndex (FI).addSImm (0).addReg (*IAR++);
      } else {
        unsigned TempReg = makeAnotherReg (Type::IntTy);
        BuildMI (BB, V8::LD, 2, TempReg).addFrameIndex (FI).addSImm (0);
        BuildMI (BB, V8::ST, 3).addFrameIndex (FI).addSImm (0).addReg (TempReg);
      }
      ArgOffset += 4;
      if (ArgOffset < 92 && IAR != IAREnd) {
        BuildMI (BB, V8::ST, 3).addFrameIndex (FI).addSImm (4).addReg (*IAR++);
      } else {
        unsigned TempReg = makeAnotherReg (Type::IntTy);
        BuildMI (BB, V8::LD, 2, TempReg).addFrameIndex (FI).addSImm (4);
        BuildMI (BB, V8::ST, 3).addFrameIndex (FI).addSImm (4).addReg (TempReg);
      }
      ArgOffset += 4;
      BuildMI (BB, V8::LDDFri, 2, ArgReg).addFrameIndex (FI).addSImm (0);
    } else if (getClassB (A.getType ()) == cLong) {
      // do the first half...
      if (ArgOffset < 92) {
        assert (IAR != IAREnd
                && "About to dereference past end of IncomingArgRegs");
        BuildMI (BB, V8::ORrr, 2, ArgReg).addReg (V8::G0).addReg (*IAR++);
      } else {
        int FI = F->getFrameInfo()->CreateFixedObject(4, ArgOffset);
        BuildMI (BB, V8::LD, 2, ArgReg).addFrameIndex (FI).addSImm (0);
      }
      ArgOffset += 4;
      // ...then do the second half
      if (ArgOffset < 92) {
        assert (IAR != IAREnd
                && "About to dereference past end of IncomingArgRegs");
        BuildMI (BB, V8::ORrr, 2, ArgReg+1).addReg (V8::G0).addReg (*IAR++);
      } else {
        int FI = F->getFrameInfo()->CreateFixedObject(4, ArgOffset);
        BuildMI (BB, V8::LD, 2, ArgReg+1).addFrameIndex (FI).addSImm (0);
      }
      ArgOffset += 4;
    } else {
      assert (0 && "Unknown class?!");
    }
  }
}

void V8ISel::SelectPHINodes() {
  const TargetInstrInfo &TII = *TM.getInstrInfo();
  const Function &LF = *F->getFunction();  // The LLVM function...
  for (Function::const_iterator I = LF.begin(), E = LF.end(); I != E; ++I) {
    const BasicBlock *BB = I;
    MachineBasicBlock &MBB = *MBBMap[I];

    // Loop over all of the PHI nodes in the LLVM basic block...
    MachineBasicBlock::iterator PHIInsertPoint = MBB.begin();
    for (BasicBlock::const_iterator I = BB->begin();
         PHINode *PN = const_cast<PHINode*>(dyn_cast<PHINode>(I)); ++I) {

      // Create a new machine instr PHI node, and insert it.
      unsigned PHIReg = getReg(*PN);
      MachineInstr *PhiMI = BuildMI(MBB, PHIInsertPoint,
                                    V8::PHI, PN->getNumOperands(), PHIReg);

      MachineInstr *LongPhiMI = 0;
      if (PN->getType() == Type::LongTy || PN->getType() == Type::ULongTy)
        LongPhiMI = BuildMI(MBB, PHIInsertPoint,
                            V8::PHI, PN->getNumOperands(), PHIReg+1);

      // PHIValues - Map of blocks to incoming virtual registers.  We use this
      // so that we only initialize one incoming value for a particular block,
      // even if the block has multiple entries in the PHI node.
      //
      std::map<MachineBasicBlock*, unsigned> PHIValues;

      for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
        MachineBasicBlock *PredMBB = 0;
        for (MachineBasicBlock::pred_iterator PI = MBB.pred_begin (),
             PE = MBB.pred_end (); PI != PE; ++PI)
          if (PN->getIncomingBlock(i) == (*PI)->getBasicBlock()) {
            PredMBB = *PI;
            break;
          }
        assert (PredMBB && "Couldn't find incoming machine-cfg edge for phi");
        
        unsigned ValReg;
        std::map<MachineBasicBlock*, unsigned>::iterator EntryIt =
          PHIValues.lower_bound(PredMBB);

        if (EntryIt != PHIValues.end() && EntryIt->first == PredMBB) {
          // We already inserted an initialization of the register for this
          // predecessor.  Recycle it.
          ValReg = EntryIt->second;

        } else {        
          // Get the incoming value into a virtual register.
          //
          Value *Val = PN->getIncomingValue(i);

          // If this is a constant or GlobalValue, we may have to insert code
          // into the basic block to compute it into a virtual register.
          if ((isa<Constant>(Val) && !isa<ConstantExpr>(Val)) ||
              isa<GlobalValue>(Val)) {
            // Simple constants get emitted at the end of the basic block,
            // before any terminator instructions.  We "know" that the code to
            // move a constant into a register will never clobber any flags.
            ValReg = getReg(Val, PredMBB, PredMBB->getFirstTerminator());
          } else {
            // Because we don't want to clobber any values which might be in
            // physical registers with the computation of this constant (which
            // might be arbitrarily complex if it is a constant expression),
            // just insert the computation at the top of the basic block.
            MachineBasicBlock::iterator PI = PredMBB->begin();
            
            // Skip over any PHI nodes though!
            while (PI != PredMBB->end() && PI->getOpcode() == V8::PHI)
              ++PI;
            
            ValReg = getReg(Val, PredMBB, PI);
          }

          // Remember that we inserted a value for this PHI for this predecessor
          PHIValues.insert(EntryIt, std::make_pair(PredMBB, ValReg));
        }

        PhiMI->addRegOperand(ValReg);
        PhiMI->addMachineBasicBlockOperand(PredMBB);
        if (LongPhiMI) {
          LongPhiMI->addRegOperand(ValReg+1);
          LongPhiMI->addMachineBasicBlockOperand(PredMBB);
        }
      }

      // Now that we emitted all of the incoming values for the PHI node, make
      // sure to reposition the InsertPoint after the PHI that we just added.
      // This is needed because we might have inserted a constant into this
      // block, right after the PHI's which is before the old insert point!
      PHIInsertPoint = LongPhiMI ? LongPhiMI : PhiMI;
      ++PHIInsertPoint;
    }
  }
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
  LoadArgumentsToVirtualRegs(&Fn);
  
  // Instruction select everything except PHI nodes
  visit(Fn);
  
  // Select the PHI nodes
  SelectPHINodes();
  
  RegMap.clear();
  MBBMap.clear();
  F = 0;
  // We always build a machine code representation for the function
  return true;
}

void V8ISel::visitCastInst(CastInst &I) {
  Value *Op = I.getOperand(0);
  unsigned DestReg = getReg(I);
  MachineBasicBlock::iterator MI = BB->end();
  emitCastOperation(BB, MI, Op, I.getType(), DestReg);
}


void V8ISel::emitIntegerCast (MachineBasicBlock *BB,
                              MachineBasicBlock::iterator IP, const Type *oldTy,
                              unsigned SrcReg, const Type *newTy,
                              unsigned DestReg) {
  if (oldTy == newTy) {
    // No-op cast - just emit a copy; assume the reg. allocator will zap it.
    BuildMI (*BB, IP, V8::ORrr, 2, DestReg).addReg (V8::G0).addReg(SrcReg);
    return;
  }
  // Emit left-shift, then right-shift to sign- or zero-extend.
  unsigned TmpReg = makeAnotherReg (newTy);
  unsigned shiftWidth = 32 - (8 * TM.getTargetData ().getTypeSize (newTy));
  BuildMI (*BB, IP, V8::SLLri, 2, TmpReg).addZImm (shiftWidth).addReg(SrcReg);
  if (newTy->isSigned ()) { // sign-extend with SRA
    BuildMI(*BB, IP, V8::SRAri, 2, DestReg).addZImm (shiftWidth).addReg(TmpReg);
  } else { // zero-extend with SRL
    BuildMI(*BB, IP, V8::SRLri, 2, DestReg).addZImm (shiftWidth).addReg(TmpReg);
  }
}

void V8ISel::emitFPToIntegerCast (MachineBasicBlock *BB,
                                  MachineBasicBlock::iterator IP,
                                  const Type *oldTy, unsigned SrcReg,
                                  const Type *newTy, unsigned DestReg) {
  unsigned FPCastOpcode, FPStoreOpcode, FPSize, FPAlign;
  unsigned oldTyClass = getClassB(oldTy);
  if (oldTyClass == cFloat) { 
    FPCastOpcode = V8::FSTOI; FPStoreOpcode = V8::STFri; FPSize = 4; 
    FPAlign = TM.getTargetData().getFloatAlignment();
  } else { // it's a double
    FPCastOpcode = V8::FDTOI; FPStoreOpcode = V8::STDFri; FPSize = 8; 
    FPAlign = TM.getTargetData().getDoubleAlignment();
  }
  unsigned TempReg = makeAnotherReg (oldTy);
  BuildMI (*BB, IP, FPCastOpcode, 1, TempReg).addReg (SrcReg);
  int FI = F->getFrameInfo()->CreateStackObject(FPSize, FPAlign);
  BuildMI (*BB, IP, FPStoreOpcode, 3).addFrameIndex (FI).addSImm (0)
    .addReg (TempReg);
  unsigned TempReg2 = makeAnotherReg (newTy);
  BuildMI (*BB, IP, V8::LD, 3, TempReg2).addFrameIndex (FI).addSImm (0);
  emitIntegerCast (BB, IP, Type::IntTy, TempReg2, newTy, DestReg);
}

/// emitCastOperation - Common code shared between visitCastInst and constant
/// expression cast support.
///
void V8ISel::emitCastOperation(MachineBasicBlock *BB,
                               MachineBasicBlock::iterator IP, Value *Src,
                               const Type *DestTy, unsigned DestReg) {
  const Type *SrcTy = Src->getType();
  unsigned SrcClass = getClassB(SrcTy);
  unsigned DestClass = getClassB(DestTy);
  unsigned SrcReg = getReg(Src, BB, IP);

  const Type *oldTy = SrcTy;
  const Type *newTy = DestTy;
  unsigned oldTyClass = SrcClass;
  unsigned newTyClass = DestClass;

  if (oldTyClass < cLong && newTyClass < cLong) {
    emitIntegerCast (BB, IP, oldTy, SrcReg, newTy, DestReg);
  } else switch (newTyClass) {
    case cByte:
    case cShort:
    case cInt:
      switch (oldTyClass) {
      case cFloat: 
      case cDouble:
        emitFPToIntegerCast (BB, IP, oldTy, SrcReg, newTy, DestReg);
        break;
      default: goto not_yet;
      }
      return;

    case cFloat:
      switch (oldTyClass) {
      case cLong: goto not_yet;
      case cFloat:
        BuildMI (*BB, IP, V8::FMOVS, 1, DestReg).addReg (SrcReg);
        break;
      case cDouble:
        BuildMI (*BB, IP, V8::FDTOS, 1, DestReg).addReg (SrcReg);
        break;
      default: {
        unsigned FltAlign = TM.getTargetData().getFloatAlignment();
        // cast integer type to float.  Store it to a stack slot and then load
        // it using ldf into a floating point register. then do fitos.
        unsigned TmpReg = makeAnotherReg (newTy);
        int FI = F->getFrameInfo()->CreateStackObject(4, FltAlign);
        BuildMI (*BB, IP, V8::ST, 3).addFrameIndex (FI).addSImm (0)
          .addReg (SrcReg);
        BuildMI (*BB, IP, V8::LDFri, 2, TmpReg).addFrameIndex (FI).addSImm (0);
        BuildMI (*BB, IP, V8::FITOS, 1, DestReg).addReg(TmpReg);
        break;
      }
      }
      return;

    case cDouble:
      switch (oldTyClass) {
      case cLong: goto not_yet;
      case cFloat:
        BuildMI (*BB, IP, V8::FSTOD, 1, DestReg).addReg (SrcReg);
        break;
      case cDouble: // use double move pseudo-instr
        BuildMI (*BB, IP, V8::FpMOVD, 1, DestReg).addReg (SrcReg);
        break;
      default: {
        unsigned DoubleAlignment = TM.getTargetData().getDoubleAlignment();
        unsigned TmpReg = makeAnotherReg (newTy);
        int FI = F->getFrameInfo()->CreateStackObject(8, DoubleAlignment);
        BuildMI (*BB, IP, V8::ST, 3).addFrameIndex (FI).addSImm (0)
          .addReg (SrcReg);
        BuildMI (*BB, IP, V8::LDDFri, 2, TmpReg).addFrameIndex (FI).addSImm (0);
        BuildMI (*BB, IP, V8::FITOD, 1, DestReg).addReg(TmpReg);
        break;
      }
      }
      return;

    case cLong:
      switch (oldTyClass) {
      case cByte:
      case cShort:
      case cInt:
        // Just copy it to the bottom half, and put a zero in the top half.
        BuildMI (*BB, IP, V8::ORrr, 2, DestReg).addReg (V8::G0)
          .addReg (V8::G0);
        BuildMI (*BB, IP, V8::ORrr, 2, DestReg+1).addReg (V8::G0)
          .addReg (SrcReg);
        break;
      case cLong:
        // Just copy both halves.
        BuildMI (*BB, IP, V8::ORrr, 2, DestReg).addReg (V8::G0).addReg (SrcReg);
        BuildMI (*BB, IP, V8::ORrr, 2, DestReg+1).addReg (V8::G0)
          .addReg (SrcReg+1);
        break;
      default: goto not_yet;
      }
      return;

    default: goto not_yet;
  }
  return;
not_yet:
  std::cerr << "Sorry, cast still unsupported: SrcTy = " << *SrcTy
            << ", DestTy = " << *DestTy << "\n";
  abort ();
}

void V8ISel::visitLoadInst(LoadInst &I) {
  unsigned DestReg = getReg (I);
  unsigned PtrReg = getReg (I.getOperand (0));
  switch (getClassB (I.getType ())) {
   case cByte:
    if (I.getType ()->isSigned ())
      BuildMI (BB, V8::LDSB, 2, DestReg).addReg (PtrReg).addSImm(0);
    else
      BuildMI (BB, V8::LDUB, 2, DestReg).addReg (PtrReg).addSImm(0);
    return;
   case cShort:
    if (I.getType ()->isSigned ())
      BuildMI (BB, V8::LDSH, 2, DestReg).addReg (PtrReg).addSImm(0);
    else
      BuildMI (BB, V8::LDUH, 2, DestReg).addReg (PtrReg).addSImm(0);
    return;
   case cInt:
    BuildMI (BB, V8::LD, 2, DestReg).addReg (PtrReg).addSImm(0);
    return;
   case cLong:
    BuildMI (BB, V8::LD, 2, DestReg).addReg (PtrReg).addSImm(0);
    BuildMI (BB, V8::LD, 2, DestReg+1).addReg (PtrReg).addSImm(4);
    return;
   case cFloat:
    BuildMI (BB, V8::LDFri, 2, DestReg).addReg (PtrReg).addSImm(0);
    return;
   case cDouble:
    BuildMI (BB, V8::LDDFri, 2, DestReg).addReg (PtrReg).addSImm(0);
    return;
   default:
    std::cerr << "Load instruction not handled: " << I;
    abort ();
    return;
  }
}

void V8ISel::visitStoreInst(StoreInst &I) {
  Value *SrcVal = I.getOperand (0);
  unsigned SrcReg = getReg (SrcVal);
  unsigned PtrReg = getReg (I.getOperand (1));
  switch (getClassB (SrcVal->getType ())) {
   case cByte:
    BuildMI (BB, V8::STB, 3).addReg (PtrReg).addSImm (0).addReg (SrcReg);
    return;
   case cShort:
    BuildMI (BB, V8::STH, 3).addReg (PtrReg).addSImm (0).addReg (SrcReg);
    return;
   case cInt:
    BuildMI (BB, V8::ST, 3).addReg (PtrReg).addSImm (0).addReg (SrcReg);
    return;
   case cLong:
    BuildMI (BB, V8::ST, 3).addReg (PtrReg).addSImm (0).addReg (SrcReg);
    BuildMI (BB, V8::ST, 3).addReg (PtrReg).addSImm (4).addReg (SrcReg+1);
    return;
   case cFloat:
    BuildMI (BB, V8::STFri, 3).addReg (PtrReg).addSImm (0).addReg (SrcReg);
    return;
   case cDouble:
    BuildMI (BB, V8::STDFri, 3).addReg (PtrReg).addSImm (0).addReg (SrcReg);
    return;
   default:
    std::cerr << "Store instruction not handled: " << I;
    abort ();
    return;
  }
}

void V8ISel::visitCallInst(CallInst &I) {
  MachineInstr *TheCall;
  // Is it an intrinsic function call?
  if (Function *F = I.getCalledFunction()) {
    if (Intrinsic::ID ID = (Intrinsic::ID)F->getIntrinsicID()) {
      visitIntrinsicCall(ID, I);   // Special intrinsics are not handled here
      return;
    }
  }

  unsigned extraStack = 0;
  // How much extra call stack will we need?
  for (unsigned i = 7; i < I.getNumOperands (); ++i) {
    switch (getClassB (I.getOperand (i)->getType ())) {
      case cLong: extraStack += 8; break;
      case cFloat: extraStack += 4; break;
      case cDouble: extraStack += 8; break;
      default: extraStack += 4; break;
    }
  }
  // Round up extra stack size to the nearest doubleword.
  if (extraStack) { extraStack = (extraStack + 7) & ~7; }

  // Deal with args
  static const unsigned OutgoingArgRegs[] = { V8::O0, V8::O1, V8::O2, V8::O3,
    V8::O4, V8::O5 };
  const unsigned *OAREnd = &OutgoingArgRegs[6];
  const unsigned *OAR = &OutgoingArgRegs[0];
  unsigned ArgOffset = 68;
  if (extraStack) BuildMI (BB, V8::ADJCALLSTACKDOWN, 1).addImm (extraStack);
  for (unsigned i = 1; i < I.getNumOperands (); ++i) {
    unsigned ArgReg = getReg (I.getOperand (i));
    if (getClassB (I.getOperand (i)->getType ()) < cLong) {
      // Schlep it over into the incoming arg register
      if (ArgOffset < 92) {
	assert (OAR != OAREnd && "About to dereference past end of OutgoingArgRegs");
	BuildMI (BB, V8::ORrr, 2, *OAR++).addReg (V8::G0).addReg (ArgReg);
      } else {
	BuildMI (BB, V8::ST, 3).addReg (V8::SP).addSImm (ArgOffset).addReg (ArgReg);
      }
      ArgOffset += 4;
    } else if (getClassB (I.getOperand (i)->getType ()) == cFloat) {
      if (ArgOffset < 92) {
	// Single-fp args are passed in integer registers; go through
	// memory to get them out of FP registers. (Bleh!)
	unsigned FltAlign = TM.getTargetData().getFloatAlignment();
	int FI = F->getFrameInfo()->CreateStackObject(4, FltAlign);
	BuildMI (BB, V8::STFri, 3).addFrameIndex (FI).addSImm (0).addReg (ArgReg);
	assert (OAR != OAREnd && "About to dereference past end of OutgoingArgRegs");
	BuildMI (BB, V8::LD, 2, *OAR++).addFrameIndex (FI).addSImm (0);
      } else {
	BuildMI (BB, V8::STFri, 3).addReg (V8::SP).addSImm (ArgOffset).addReg (ArgReg);
      }
      ArgOffset += 4;
    } else if (getClassB (I.getOperand (i)->getType ()) == cDouble) {
      // Double-fp args are passed in pairs of integer registers; go through
      // memory to get them out of FP registers. (Bleh!)
      // We'd like to 'std' these right onto the outgoing-args area, but it might
      // not be 8-byte aligned (e.g., call x(int x, double d)). sigh.
      unsigned DblAlign = TM.getTargetData().getDoubleAlignment();
      int FI = F->getFrameInfo()->CreateStackObject(8, DblAlign);
      BuildMI (BB, V8::STDFri, 3).addFrameIndex (FI).addSImm (0).addReg (ArgReg);
      if (ArgOffset < 92 && OAR != OAREnd) {
	assert (OAR != OAREnd && "About to dereference past end of OutgoingArgRegs");
	BuildMI (BB, V8::LD, 2, *OAR++).addFrameIndex (FI).addSImm (0);
      } else {
        unsigned TempReg = makeAnotherReg (Type::IntTy);
	BuildMI (BB, V8::LD, 2, TempReg).addFrameIndex (FI).addSImm (0);
	BuildMI (BB, V8::ST, 3).addReg (V8::SP).addSImm (ArgOffset).addReg (TempReg);
      }
      ArgOffset += 4;
      if (ArgOffset < 92 && OAR != OAREnd) {
	assert (OAR != OAREnd && "About to dereference past end of OutgoingArgRegs");
	BuildMI (BB, V8::LD, 2, *OAR++).addFrameIndex (FI).addSImm (4);
      } else {
        unsigned TempReg = makeAnotherReg (Type::IntTy);
	BuildMI (BB, V8::LD, 2, TempReg).addFrameIndex (FI).addSImm (4);
	BuildMI (BB, V8::ST, 3).addReg (V8::SP).addSImm (ArgOffset).addReg (TempReg);
      }
      ArgOffset += 4;
    } else if (getClassB (I.getOperand (i)->getType ()) == cLong) {
      // do the first half...
      if (ArgOffset < 92) {
	assert (OAR != OAREnd && "About to dereference past end of OutgoingArgRegs");
	BuildMI (BB, V8::ORrr, 2, *OAR++).addReg (V8::G0).addReg (ArgReg);
      } else {
	BuildMI (BB, V8::ST, 3).addReg (V8::SP).addSImm (ArgOffset).addReg (ArgReg);
      }
      ArgOffset += 4;
      // ...then do the second half
      if (ArgOffset < 92) {
	assert (OAR != OAREnd && "About to dereference past end of OutgoingArgRegs");
	BuildMI (BB, V8::ORrr, 2, *OAR++).addReg (V8::G0).addReg (ArgReg+1);
      } else {
	BuildMI (BB, V8::ST, 3).addReg (V8::SP).addSImm (ArgOffset).addReg (ArgReg+1);
      }
      ArgOffset += 4;
    } else {
      assert (0 && "Unknown class?!");
    }
  }

  // Emit call instruction
  if (Function *F = I.getCalledFunction ()) {
    BuildMI (BB, V8::CALL, 1).addGlobalAddress (F, true);
  } else {  // Emit an indirect call...
    unsigned Reg = getReg (I.getCalledValue ());
    BuildMI (BB, V8::JMPLrr, 3, V8::O7).addReg (Reg).addReg (V8::G0);
  }

  if (extraStack) BuildMI (BB, V8::ADJCALLSTACKUP, 1).addImm (extraStack);

  // Deal w/ return value: schlep it over into the destination register
  if (I.getType () == Type::VoidTy)
    return;
  unsigned DestReg = getReg (I);
  switch (getClassB (I.getType ())) {
    case cByte:
    case cShort:
    case cInt:
      BuildMI (BB, V8::ORrr, 2, DestReg).addReg(V8::G0).addReg(V8::O0);
      break;
    case cFloat:
      BuildMI (BB, V8::FMOVS, 2, DestReg).addReg(V8::F0);
      break;
    case cDouble:
      BuildMI (BB, V8::FpMOVD, 2, DestReg).addReg(V8::D0);
      break;
    case cLong:
      BuildMI (BB, V8::ORrr, 2, DestReg).addReg(V8::G0).addReg(V8::O0);
      BuildMI (BB, V8::ORrr, 2, DestReg+1).addReg(V8::G0).addReg(V8::O1);
      break;
    default:
      std::cerr << "Return type of call instruction not handled: " << I;
      abort ();
  }
}

void V8ISel::visitReturnInst(ReturnInst &I) {
  if (I.getNumOperands () == 1) {
    unsigned RetValReg = getReg (I.getOperand (0));
    switch (getClassB (I.getOperand (0)->getType ())) {
      case cByte:
      case cShort:
      case cInt:
        // Schlep it over into i0 (where it will become o0 after restore).
        BuildMI (BB, V8::ORrr, 2, V8::I0).addReg(V8::G0).addReg(RetValReg);
        break;
      case cFloat:
        BuildMI (BB, V8::FMOVS, 1, V8::F0).addReg(RetValReg);
        break;
      case cDouble:
        BuildMI (BB, V8::FpMOVD, 1, V8::D0).addReg(RetValReg);
        break;
      case cLong:
        BuildMI (BB, V8::ORrr, 2, V8::I0).addReg(V8::G0).addReg(RetValReg);
        BuildMI (BB, V8::ORrr, 2, V8::I1).addReg(V8::G0).addReg(RetValReg+1);
        break;
      default:
        std::cerr << "Return instruction of this type not handled: " << I;
        abort ();
    }
  }

  // Just emit a 'retl' instruction to return.
  BuildMI(BB, V8::RETL, 0);
  return;
}

static inline BasicBlock *getBlockAfter(BasicBlock *BB) {
  Function::iterator I = BB; ++I;  // Get iterator to next block
  return I != BB->getParent()->end() ? &*I : 0;
}

/// visitBranchInst - Handles conditional and unconditional branches.
///
void V8ISel::visitBranchInst(BranchInst &I) {
  BasicBlock *takenSucc = I.getSuccessor (0);
  MachineBasicBlock *takenSuccMBB = MBBMap[takenSucc];
  BB->addSuccessor (takenSuccMBB);
  if (I.isConditional()) {  // conditional branch
    BasicBlock *notTakenSucc = I.getSuccessor (1);
    MachineBasicBlock *notTakenSuccMBB = MBBMap[notTakenSucc];
    BB->addSuccessor (notTakenSuccMBB);

    // CondReg=(<condition>);
    // If (CondReg==0) goto notTakenSuccMBB;
    unsigned CondReg = getReg (I.getCondition ());
    BuildMI (BB, V8::CMPri, 2).addSImm (0).addReg (CondReg);
    BuildMI (BB, V8::BE, 1).addMBB (notTakenSuccMBB);
  }
  // goto takenSuccMBB;
  BuildMI (BB, V8::BA, 1).addMBB (takenSuccMBB);
}

/// emitGEPOperation - Common code shared between visitGetElementPtrInst and
/// constant expression GEP support.
///
void V8ISel::emitGEPOperation (MachineBasicBlock *MBB,
                               MachineBasicBlock::iterator IP,
		               Value *Src, User::op_iterator IdxBegin,
		               User::op_iterator IdxEnd, unsigned TargetReg) {
  const TargetData &TD = TM.getTargetData ();
  const Type *Ty = Src->getType ();
  unsigned basePtrReg = getReg (Src, MBB, IP);

  // GEPs have zero or more indices; we must perform a struct access
  // or array access for each one.
  for (GetElementPtrInst::op_iterator oi = IdxBegin, oe = IdxEnd; oi != oe;
       ++oi) {
    Value *idx = *oi;
    unsigned nextBasePtrReg = makeAnotherReg (Type::UIntTy);
    if (const StructType *StTy = dyn_cast<StructType> (Ty)) {
      // It's a struct access.  idx is the index into the structure,
      // which names the field. Use the TargetData structure to
      // pick out what the layout of the structure is in memory.
      // Use the (constant) structure index's value to find the
      // right byte offset from the StructLayout class's list of
      // structure member offsets.
      unsigned fieldIndex = cast<ConstantUInt> (idx)->getValue ();
      unsigned memberOffset =
        TD.getStructLayout (StTy)->MemberOffsets[fieldIndex];
      // Emit an ADD to add memberOffset to the basePtr.
      BuildMI (*MBB, IP, V8::ADDri, 2,
               nextBasePtrReg).addReg (basePtrReg).addZImm (memberOffset);
      // The next type is the member of the structure selected by the
      // index.
      Ty = StTy->getElementType (fieldIndex);
    } else if (const SequentialType *SqTy = dyn_cast<SequentialType> (Ty)) {
      // It's an array or pointer access: [ArraySize x ElementType].
      // We want to add basePtrReg to (idxReg * sizeof ElementType). First, we
      // must find the size of the pointed-to type (Not coincidentally, the next
      // type is the type of the elements in the array).
      Ty = SqTy->getElementType ();
      unsigned elementSize = TD.getTypeSize (Ty);
      unsigned idxReg = getReg (idx, MBB, IP);
      unsigned OffsetReg = makeAnotherReg (Type::IntTy);
      unsigned elementSizeReg = makeAnotherReg (Type::UIntTy);
      copyConstantToRegister (MBB, IP,
        ConstantUInt::get(Type::UIntTy, elementSize), elementSizeReg);
      // Emit a SMUL to multiply the register holding the index by
      // elementSize, putting the result in OffsetReg.
      BuildMI (*MBB, IP, V8::SMULrr, 2,
               OffsetReg).addReg (elementSizeReg).addReg (idxReg);
      // Emit an ADD to add OffsetReg to the basePtr.
      BuildMI (*MBB, IP, V8::ADDrr, 2,
               nextBasePtrReg).addReg (basePtrReg).addReg (OffsetReg);
    }
    basePtrReg = nextBasePtrReg;
  }
  // After we have processed all the indices, the result is left in
  // basePtrReg.  Move it to the register where we were expected to
  // put the answer.
  BuildMI (BB, V8::ORrr, 1, TargetReg).addReg (V8::G0).addReg (basePtrReg);
}

void V8ISel::visitGetElementPtrInst (GetElementPtrInst &I) {
  unsigned outputReg = getReg (I);
  emitGEPOperation (BB, BB->end (), I.getOperand (0),
                    I.op_begin ()+1, I.op_end (), outputReg);
}


void V8ISel::visitBinaryOperator (Instruction &I) {
  unsigned DestReg = getReg (I);
  unsigned Op0Reg = getReg (I.getOperand (0));
  unsigned Op1Reg = getReg (I.getOperand (1));

  unsigned Class = getClassB (I.getType());
  unsigned OpCase = ~0;

  if (Class > cLong) {
    switch (I.getOpcode ()) {
    case Instruction::Add: OpCase = 0; break;
    case Instruction::Sub: OpCase = 1; break;
    case Instruction::Mul: OpCase = 2; break;
    case Instruction::Div: OpCase = 3; break;
    default: visitInstruction (I); return;
    }
    static unsigned Opcodes[] = { V8::FADDS, V8::FADDD,
                                  V8::FSUBS, V8::FSUBD,
                                  V8::FMULS, V8::FMULD,
                                  V8::FDIVS, V8::FDIVD };
    BuildMI (BB, Opcodes[2*OpCase + (Class - cFloat)], 2, DestReg)
      .addReg (Op0Reg).addReg (Op1Reg);
    return;
  }

  unsigned ResultReg = DestReg;
  if (Class != cInt && Class != cLong)
    ResultReg = makeAnotherReg (I.getType ());

  if (Class == cLong) {
    DEBUG (std::cerr << "Class = cLong\n");
    DEBUG (std::cerr << "Op0Reg = " << Op0Reg << ", " << Op0Reg+1 << "\n");
    DEBUG (std::cerr << "Op1Reg = " << Op1Reg << ", " << Op1Reg+1 << "\n");
    DEBUG (std::cerr << "ResultReg = " << ResultReg << ", " << ResultReg+1 << "\n");
    DEBUG (std::cerr << "DestReg = " << DestReg << ", " << DestReg+1 <<  "\n");
  }

  // FIXME: support long, ulong.
  switch (I.getOpcode ()) {
  case Instruction::Add: OpCase = 0; break;
  case Instruction::Sub: OpCase = 1; break;
  case Instruction::Mul: OpCase = 2; break;
  case Instruction::And: OpCase = 3; break;
  case Instruction::Or:  OpCase = 4; break;
  case Instruction::Xor: OpCase = 5; break;
  case Instruction::Shl: OpCase = 6; break;
  case Instruction::Shr: OpCase = 7+I.getType()->isSigned(); break;

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

  static const unsigned Opcodes[] = {
    V8::ADDrr, V8::SUBrr, V8::SMULrr, V8::ANDrr, V8::ORrr, V8::XORrr,
    V8::SLLrr, V8::SRLrr, V8::SRArr
  };
  if (OpCase != ~0U) {
    BuildMI (BB, Opcodes[OpCase], 2, ResultReg).addReg (Op0Reg).addReg (Op1Reg);
  }

  switch (getClassB (I.getType ())) {
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
      // Nothing to do here.
      break;
    case cLong:
      // Only support and, or, xor.
      if (OpCase < 3 || OpCase > 5) {
        visitInstruction (I);
        return;
      }
      // Do the other half of the value:
      BuildMI (BB, Opcodes[OpCase], 2, ResultReg+1).addReg (Op0Reg+1)
        .addReg (Op1Reg+1);
      break;
    default:
      visitInstruction (I);
  }
}

void V8ISel::visitSetCondInst(SetCondInst &I) {
  unsigned Op0Reg = getReg (I.getOperand (0));
  unsigned Op1Reg = getReg (I.getOperand (1));
  unsigned DestReg = getReg (I);
  const Type *Ty = I.getOperand (0)->getType ();
  
  // Compare the two values.
  assert (getClass (Ty) != cLong && "can't setcc on longs yet");
  if (getClass (Ty) < cLong) {
    BuildMI(BB, V8::SUBCCrr, 2, V8::G0).addReg(Op0Reg).addReg(Op1Reg);
  } else if (getClass (Ty) == cFloat) {
    BuildMI(BB, V8::FCMPS, 2).addReg(Op0Reg).addReg(Op1Reg);
  } else if (getClass (Ty) == cDouble) {
    BuildMI(BB, V8::FCMPD, 2).addReg(Op0Reg).addReg(Op1Reg);
  }

  unsigned BranchIdx;
  switch (I.getOpcode()) {
  default: assert(0 && "Unknown setcc instruction!");
  case Instruction::SetEQ: BranchIdx = 0; break;
  case Instruction::SetNE: BranchIdx = 1; break;
  case Instruction::SetLT: BranchIdx = 2; break;
  case Instruction::SetGT: BranchIdx = 3; break;
  case Instruction::SetLE: BranchIdx = 4; break;
  case Instruction::SetGE: BranchIdx = 5; break;
  }
  unsigned Column = 0;
  if (Ty->isSigned() && !Ty->isFloatingPoint()) Column = 1;
  if (Ty->isFloatingPoint()) Column = 2;
  static unsigned OpcodeTab[3*6] = {
                                 // LLVM            SparcV8
                                 //        unsigned signed  fp
    V8::BE,   V8::BE,  V8::FBE,  // seteq = be      be      fbe
    V8::BNE,  V8::BNE, V8::FBNE, // setne = bne     bne     fbne
    V8::BCS,  V8::BL,  V8::FBL,  // setlt = bcs     bl      fbl
    V8::BGU,  V8::BG,  V8::FBG,  // setgt = bgu     bg      fbg
    V8::BLEU, V8::BLE, V8::FBLE, // setle = bleu    ble     fble
    V8::BCC,  V8::BGE, V8::FBGE  // setge = bcc     bge     fbge
  };
  unsigned Opcode = OpcodeTab[3*BranchIdx + Column];

  MachineBasicBlock *thisMBB = BB;
  const BasicBlock *LLVM_BB = BB->getBasicBlock ();
  //  thisMBB:
  //  ...
  //   subcc %reg0, %reg1, %g0
  //   bCC copy1MBB
  //   ba copy0MBB

  // FIXME: we wouldn't need copy0MBB (we could fold it into thisMBB)
  // if we could insert other, non-terminator instructions after the
  // bCC. But MBB->getFirstTerminator() can't understand this.
  MachineBasicBlock *copy1MBB = new MachineBasicBlock (LLVM_BB);
  F->getBasicBlockList ().push_back (copy1MBB);
  BuildMI (BB, Opcode, 1).addMBB (copy1MBB);
  MachineBasicBlock *copy0MBB = new MachineBasicBlock (LLVM_BB);
  F->getBasicBlockList ().push_back (copy0MBB);
  BuildMI (BB, V8::BA, 1).addMBB (copy0MBB);
  // Update machine-CFG edges
  BB->addSuccessor (copy1MBB);
  BB->addSuccessor (copy0MBB);

  //  copy0MBB:
  //   %FalseValue = or %G0, 0
  //   ba sinkMBB
  BB = copy0MBB;
  unsigned FalseValue = makeAnotherReg (I.getType ());
  BuildMI (BB, V8::ORri, 2, FalseValue).addReg (V8::G0).addZImm (0);
  MachineBasicBlock *sinkMBB = new MachineBasicBlock (LLVM_BB);
  F->getBasicBlockList ().push_back (sinkMBB);
  BuildMI (BB, V8::BA, 1).addMBB (sinkMBB);
  // Update machine-CFG edges
  BB->addSuccessor (sinkMBB);

  DEBUG (std::cerr << "thisMBB is at " << (void*)thisMBB << "\n");
  DEBUG (std::cerr << "copy1MBB is at " << (void*)copy1MBB << "\n");
  DEBUG (std::cerr << "copy0MBB is at " << (void*)copy0MBB << "\n");
  DEBUG (std::cerr << "sinkMBB is at " << (void*)sinkMBB << "\n");

  //  copy1MBB:
  //   %TrueValue = or %G0, 1
  //   ba sinkMBB
  BB = copy1MBB;
  unsigned TrueValue = makeAnotherReg (I.getType ());
  BuildMI (BB, V8::ORri, 2, TrueValue).addReg (V8::G0).addZImm (1);
  BuildMI (BB, V8::BA, 1).addMBB (sinkMBB);
  // Update machine-CFG edges
  BB->addSuccessor (sinkMBB);

  //  sinkMBB:
  //   %Result = phi [ %FalseValue, copy0MBB ], [ %TrueValue, copy1MBB ]
  //  ...
  BB = sinkMBB;
  BuildMI (BB, V8::PHI, 4, DestReg).addReg (FalseValue)
    .addMBB (copy0MBB).addReg (TrueValue).addMBB (copy1MBB);
}

void V8ISel::visitAllocaInst(AllocaInst &I) {
  // Find the data size of the alloca inst's getAllocatedType.
  const Type *Ty = I.getAllocatedType();
  unsigned TySize = TM.getTargetData().getTypeSize(Ty);

  unsigned ArraySizeReg = getReg (I.getArraySize ());
  unsigned TySizeReg = getReg (ConstantUInt::get (Type::UIntTy, TySize));
  unsigned TmpReg1 = makeAnotherReg (Type::UIntTy);
  unsigned TmpReg2 = makeAnotherReg (Type::UIntTy);
  unsigned StackAdjReg = makeAnotherReg (Type::UIntTy);

  // StackAdjReg = (ArraySize * TySize) rounded up to nearest doubleword boundary
  BuildMI (BB, V8::UMULrr, 2, TmpReg1).addReg (ArraySizeReg).addReg (TySizeReg);

  // Round up TmpReg1 to nearest doubleword boundary:
  BuildMI (BB, V8::ADDri, 2, TmpReg2).addReg (TmpReg1).addSImm (7);
  BuildMI (BB, V8::ANDri, 2, StackAdjReg).addReg (TmpReg2).addSImm (-8);

  // Subtract size from stack pointer, thereby allocating some space.
  BuildMI (BB, V8::SUBrr, 2, V8::SP).addReg (V8::SP).addReg (StackAdjReg);

  // Put a pointer to the space into the result register, by copying
  // the stack pointer.
  BuildMI (BB, V8::ADDri, 2, getReg(I)).addReg (V8::SP).addSImm (96);

  // Inform the Frame Information that we have just allocated a variable-sized
  // object.
  F->getFrameInfo()->CreateVariableSizedObject();
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
