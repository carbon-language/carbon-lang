//===- PreSelection.cpp - Specialize LLVM code for target machine ---------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines the PreSelection pass which specializes LLVM code for a
// target machine, while remaining in legal portable LLVM form and
// preserving type information and type safety.  This is meant to enable
// dataflow optimizations on target-specific operations such as accesses to
// constants, globals, and array indexing.
//
//===----------------------------------------------------------------------===//

#include "SparcInternals.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/iMemory.h"
#include "llvm/iPHINode.h"
#include "llvm/iOther.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/InstVisitor.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Scalar.h"
#include <algorithm>

namespace llvm {

namespace {

  //===--------------------------------------------------------------------===//
  // PreSelection Pass - Specialize LLVM code for the current target machine.
  // 
  class PreSelection : public FunctionPass, public InstVisitor<PreSelection> {
    const TargetInstrInfo &instrInfo;

  public:
    PreSelection(const TargetMachine &T)
      : instrInfo(T.getInstrInfo()) {}

    // runOnFunction - apply this pass to each Function
    bool runOnFunction(Function &F) {
      visit(F);
      return true;
    }

    // These methods do the actual work of specializing code
    void visitInstruction(Instruction &I);   // common work for every instr. 
    void visitGetElementPtrInst(GetElementPtrInst &I);
    void visitCallInst(CallInst &I);
    void visitPHINode(PHINode &PN);

    // Helper functions for visiting operands of every instruction
    // 
    // visitOperands() works on every operand in [firstOp, lastOp-1].
    // If lastOp==0, lastOp defaults to #operands or #incoming Phi values.
    // 
    // visitOneOperand() does all the work for one operand.
    // 
    void visitOperands(Instruction &I, int firstOp=0);
    void visitOneOperand(Instruction &I, Value* Op, unsigned opNum,
                         Instruction& insertBefore);
  };

#if 0
  // Register the pass...
  RegisterPass<PreSelection> X("preselect",
                               "Specialize LLVM code for a target machine"
                               createPreselectionPass);
#endif

}  // end anonymous namespace


//------------------------------------------------------------------------------
// Helper functions used by methods of class PreSelection
//------------------------------------------------------------------------------


// getGlobalAddr(): Put address of a global into a v. register.
static GetElementPtrInst* getGlobalAddr(Value* ptr, Instruction& insertBefore) {
  if (isa<ConstantPointerRef>(ptr))
    ptr = cast<ConstantPointerRef>(ptr)->getValue();

  return (isa<GlobalVariable>(ptr))
    ? new GetElementPtrInst(ptr,
                    std::vector<Value*>(1, ConstantSInt::get(Type::LongTy, 0U)),
                    "addrOfGlobal", &insertBefore)
    : NULL;
}

// Wrapper on Constant::classof to use in find_if
inline static bool nonConstant(const Use& U) {
  return ! isa<Constant>(U);
}

static Instruction* DecomposeConstantExpr(ConstantExpr* CE,
                                          Instruction& insertBefore)
{
  Value *getArg1, *getArg2;

  switch(CE->getOpcode())
    {
    case Instruction::Cast:
      getArg1 = CE->getOperand(0);
      if (ConstantExpr* CEarg = dyn_cast<ConstantExpr>(getArg1))
        getArg1 = DecomposeConstantExpr(CEarg, insertBefore);
      return new CastInst(getArg1, CE->getType(), "constantCast",&insertBefore);

    case Instruction::GetElementPtr:
      assert(find_if(CE->op_begin()+1, CE->op_end(),nonConstant) == CE->op_end()
             && "All indices in ConstantExpr getelementptr must be constant!");
      getArg1 = CE->getOperand(0);
      if (ConstantExpr* CEarg = dyn_cast<ConstantExpr>(getArg1))
        getArg1 = DecomposeConstantExpr(CEarg, insertBefore);
      else if (GetElementPtrInst* gep = getGlobalAddr(getArg1, insertBefore))
        getArg1 = gep;
      return new GetElementPtrInst(getArg1,
                          std::vector<Value*>(CE->op_begin()+1, CE->op_end()),
                          "constantGEP", &insertBefore);

    default:                            // must be a binary operator
      assert(CE->getOpcode() >= Instruction::BinaryOpsBegin &&
             CE->getOpcode() <  Instruction::BinaryOpsEnd &&
             "Unrecognized opcode in ConstantExpr");
      getArg1 = CE->getOperand(0);
      if (ConstantExpr* CEarg = dyn_cast<ConstantExpr>(getArg1))
        getArg1 = DecomposeConstantExpr(CEarg, insertBefore);
      getArg2 = CE->getOperand(1);
      if (ConstantExpr* CEarg = dyn_cast<ConstantExpr>(getArg2))
        getArg2 = DecomposeConstantExpr(CEarg, insertBefore);
      return BinaryOperator::create((Instruction::BinaryOps) CE->getOpcode(),
                                    getArg1, getArg2,
                                    "constantBinaryOp", &insertBefore);
    }
}


//------------------------------------------------------------------------------
// Instruction visitor methods to perform instruction-specific operations
//------------------------------------------------------------------------------
inline void
PreSelection::visitOneOperand(Instruction &I, Value* Op, unsigned opNum,
                              Instruction& insertBefore)
{
  assert(&insertBefore != NULL && "Must have instruction to insert before.");

  if (GetElementPtrInst* gep = getGlobalAddr(Op, insertBefore)) {
    I.setOperand(opNum, gep);           // replace global operand
    return;                             // nothing more to do for this op.
  }

  Constant* CV  = dyn_cast<Constant>(Op);
  if (CV == NULL)
    return;

  if (ConstantExpr* CE = dyn_cast<ConstantExpr>(CV)) {
    // load-time constant: factor it out so we optimize as best we can
    Instruction* computeConst = DecomposeConstantExpr(CE, insertBefore);
    I.setOperand(opNum, computeConst); // replace expr operand with result
  } else if (instrInfo.ConstantTypeMustBeLoaded(CV)) {
    // load address of constant into a register, then load the constant
    // this is now done during instruction selection
    // the constant will live in the MachineConstantPool later on
  } else if (instrInfo.ConstantMayNotFitInImmedField(CV, &I)) {
    // put the constant into a virtual register using a cast
    CastInst* castI = new CastInst(CV, CV->getType(), "copyConst",
                                   &insertBefore);
    I.setOperand(opNum, castI);      // replace operand with copy in v.reg.
  }
}

/// visitOperands - transform individual operands of all instructions:
/// -- Load "large" int constants into a virtual register.  What is large
///    depends on the type of instruction and on the target architecture.
/// -- For any constants that cannot be put in an immediate field,
///    load address into virtual register first, and then load the constant.
/// 
/// firstOp and lastOp can be used to skip leading and trailing operands.
/// If lastOp is 0, it defaults to #operands or #incoming Phi values.
///  
inline void PreSelection::visitOperands(Instruction &I, int firstOp) {
  // For any instruction other than PHI, copies go just before the instr.
  for (unsigned i = firstOp, e = I.getNumOperands(); i != e; ++i)
    visitOneOperand(I, I.getOperand(i), i, I);
}


void PreSelection::visitPHINode(PHINode &PN) {
  // For a PHI, operand copies must be before the terminator of the
  // appropriate predecessor basic block.  Remaining logic is simple
  // so just handle PHIs and other instructions separately.
  // 
  for (unsigned i = 0, e = PN.getNumIncomingValues(); i != e; ++i)
    visitOneOperand(PN, PN.getIncomingValue(i),
                    PN.getOperandNumForIncomingValue(i),
                    *PN.getIncomingBlock(i)->getTerminator());
  // do not call visitOperands!
}

// Common work for *all* instructions.  This needs to be called explicitly
// by other visit<InstructionType> functions.
inline void PreSelection::visitInstruction(Instruction &I) { 
  visitOperands(I);              // Perform operand transformations
}

// GetElementPtr instructions: check if pointer is a global
void PreSelection::visitGetElementPtrInst(GetElementPtrInst &I) { 
  Instruction* curI = &I;

  // Decompose multidimensional array references
  if (I.getNumIndices() >= 2) {
    // DecomposeArrayRef() replaces I and deletes it, if successful,
    // so remember predecessor in order to find the replacement instruction.
    // Also remember the basic block in case there is no predecessor.
    Instruction* prevI = I.getPrev();
    BasicBlock* bb = I.getParent();
    if (DecomposeArrayRef(&I))
      // first instr. replacing I
      curI = cast<GetElementPtrInst>(prevI? prevI->getNext() : &bb->front());
  }

  // Perform other transformations common to all instructions
  visitInstruction(*curI);
}

void PreSelection::visitCallInst(CallInst &I) {
  // Tell visitOperands to ignore the function name if this is a direct call.
  visitOperands(I, (/*firstOp=*/ I.getCalledFunction()? 1 : 0));
}

/// createPreSelectionPass - Public entry point for the PreSelection pass
///
FunctionPass* createPreSelectionPass(const TargetMachine &TM) {
  return new PreSelection(TM);
}

} // End llvm namespace
