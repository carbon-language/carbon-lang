//===- PreSelection.cpp - Specialize LLVM code for target machine ---------===//
//
// This file defines the PreSelection pass which specializes LLVM code for a
// target machine, while remaining in legal portable LLVM form and
// preserving type information and type safety.  This is meant to enable
// dataflow optimizations on target-specific operations such as accesses to
// constants, globals, and array indexing.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/PreSelection.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Support/InstVisitor.h"
#include "llvm/Module.h"
#include "llvm/Constants.h"
#include "llvm/iMemory.h"
#include "llvm/iPHINode.h"
#include "llvm/iOther.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Pass.h"
#include "Support/CommandLine.h"
#include <algorithm>

namespace {
  //===--------------------------------------------------------------------===//
  // SelectDebugLevel - Allow command line control over debugging.
  //
  enum PreSelectDebugLevel_t {
    PreSelect_NoDebugInfo,
    PreSelect_PrintOutput, 
  };

  // Enable Debug Options to be specified on the command line
  cl::opt<PreSelectDebugLevel_t>
  PreSelectDebugLevel("dpreselect", cl::Hidden,
     cl::desc("debug information for target-dependent pre-selection"),
     cl::values(
       clEnumValN(PreSelect_NoDebugInfo, "n", "disable debug output (default)"),
       clEnumValN(PreSelect_PrintOutput, "y", "print generated machine code"),
       /* default level = */ PreSelect_NoDebugInfo));


  //===--------------------------------------------------------------------===//
  // class ConstantPoolForModule:
  // 
  // The pool of constants that must be emitted for a module.
  // This is a single pool for the entire module and is shared by
  // all invocations of the PreSelection pass for this module by putting
  // this as an annotation on the Module object.
  // A single GlobalVariable is created for each constant in the pool
  // representing the memory for that constant.  
  // 
  AnnotationID CPFM_AID(
                 AnnotationManager::getID("CodeGen::ConstantPoolForModule"));

  class ConstantPoolForModule : private Annotation {
    Module* myModule;
    std::map<const Constant*, GlobalVariable*> gvars;
    std::map<const Constant*, GlobalVariable*> origGVars;
    ConstantPoolForModule(Module* M);   // called only by annotation builder
    ConstantPoolForModule();                      // DO NOT IMPLEMENT
    void operator=(const ConstantPoolForModule&); // DO NOT IMPLEMENT
  public:
    static ConstantPoolForModule& get(Module* M) {
      ConstantPoolForModule* cpool =
        (ConstantPoolForModule*) M->getAnnotation(CPFM_AID);
      if (cpool == NULL) // create a new annotation and add it to the Module
        M->addAnnotation(cpool = new ConstantPoolForModule(M));
      return *cpool;
    }

    GlobalVariable* getGlobalForConstant(Constant* CV) {
      std::map<const Constant*, GlobalVariable*>::iterator I = gvars.find(CV);
      if (I != gvars.end())
        return I->second;               // global exists so return it
      return addToConstantPool(CV);     // create a new global and return it
    }

    GlobalVariable*  addToConstantPool(Constant* CV) {
      GlobalVariable*& GV = gvars[CV];  // handle to global var entry in map
      if (GV == NULL)
        { // check if a global constant already existed; otherwise create one
          std::map<const Constant*, GlobalVariable*>::iterator PI =
            origGVars.find(CV);
          if (PI != origGVars.end())
            GV = PI->second;            // put in map
          else
            {
              GV = new GlobalVariable(CV->getType(), true, //put in map
                                      GlobalValue::InternalLinkage, CV);
              myModule->getGlobalList().push_back(GV); // GV owned by module now
            }
        }
      return GV;
    }
  };

  /* ctor */
  ConstantPoolForModule::ConstantPoolForModule(Module* M)
    : Annotation(CPFM_AID), myModule(M)
  {
    // Build reverse map for pre-existing global constants so we can find them
    for (Module::giterator GI = M->gbegin(), GE = M->gend(); GI != GE; ++GI)
      if (GI->hasInitializer() && GI->isConstant())
        origGVars[GI->getInitializer()] = GI;
  }

  //===--------------------------------------------------------------------===//
  // PreSelection Pass - Specialize LLVM code for the current target machine.
  // This was and will be a basicblock pass, but make it a FunctionPass until
  // BasicBlockPass ::doFinalization(Function&) is available.
  // 
  class PreSelection : public BasicBlockPass, public InstVisitor<PreSelection>
  {
    const TargetMachine &target;
    const TargetInstrInfo &instrInfo;
    Function* function;

    GlobalVariable* getGlobalForConstant(Constant* CV) {
      Module* M = function->getParent();
      return ConstantPoolForModule::get(M).getGlobalForConstant(CV);
    }

  public:
    PreSelection (const TargetMachine &T):
      target(T), instrInfo(T.getInstrInfo()), function(NULL) {}

    // runOnBasicBlock - apply this pass to each BB
    bool runOnBasicBlock(BasicBlock &BB) {
      function = BB.getParent();
      this->visit(BB);
      return true;
    }

    bool doFinalization(Function &F) {
      if (PreSelectDebugLevel >= PreSelect_PrintOutput)
        std::cerr << "\n\n*** LLVM code after pre-selection for function "
                  << F.getName() << ":\n\n" << F;
      return false;
    }

    // These methods do the actual work of specializing code
    void visitInstruction(Instruction &I);   // common work for every instr. 
    void visitGetElementPtrInst(GetElementPtrInst &I);
    void visitLoadInst(LoadInst &I);
    void visitCastInst(CastInst &I);
    void visitCallInst(CallInst &I);
    void visitStoreInst(StoreInst &I);

    // Helper functions for visiting operands of every instruction
    // 
    // visitOperands() works on every operand in [firstOp, lastOp-1].
    // If lastOp==0, lastOp defaults to #operands or #incoming Phi values.
    // 
    // visitOneOperand() does all the work for one operand.
    // 
    void visitOperands(Instruction &I, int firstOp=0, int lastOp=0);
    void visitOneOperand(Instruction &I, Value* Op, unsigned opNum,
                         Instruction& insertBefore);
  };

  // Register the pass...
  RegisterOpt<PreSelection> X("preselect",
                              "Specialize LLVM code for a target machine",
                              createPreSelectionPass);
}  // end anonymous namespace


//------------------------------------------------------------------------------
// Helper functions used by methods of class PreSelection
//------------------------------------------------------------------------------


// getGlobalAddr(): Put address of a global into a v. register.
static GetElementPtrInst* getGlobalAddr(Value* ptr, Instruction& insertBefore)
{
  if (isa<ConstantPointerRef>(ptr))
    ptr = cast<ConstantPointerRef>(ptr)->getValue();

  return (isa<GlobalVariable>(ptr))
    ? new GetElementPtrInst(ptr,
                    std::vector<Value*>(1, ConstantSInt::get(Type::LongTy, 0U)),
                    "addrOfGlobal", &insertBefore)
    : NULL;
}


// Wrapper on Constant::classof to use in find_if :-(
inline static bool nonConstant(const Use& U)
{
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
  if (GetElementPtrInst* gep = getGlobalAddr(Op, insertBefore)) {
    I.setOperand(opNum, gep);           // replace global operand
    return;
  }

  Constant* CV  = dyn_cast<Constant>(Op);
  if (CV == NULL)
    return;

  if (ConstantExpr* CE = dyn_cast<ConstantExpr>(CV))
    { // load-time constant: factor it out so we optimize as best we can
      Instruction* computeConst = DecomposeConstantExpr(CE, insertBefore);
      I.setOperand(opNum, computeConst); // replace expr operand with result
    }
  else if (instrInfo.ConstantTypeMustBeLoaded(CV))
    { // load address of constant into a register, then load the constant
      GetElementPtrInst* gep = getGlobalAddr(getGlobalForConstant(CV),
                                             insertBefore);
      LoadInst* ldI = new LoadInst(gep, "loadConst", &insertBefore);
      I.setOperand(opNum, ldI);        // replace operand with copy in v.reg.
    }
  else if (instrInfo.ConstantMayNotFitInImmedField(CV, &I))
    { // put the constant into a virtual register using a cast
      CastInst* castI = new CastInst(CV, CV->getType(), "copyConst",
                                     &insertBefore);
      I.setOperand(opNum, castI);      // replace operand with copy in v.reg.
    }
}

// visitOperands() transforms individual operands of all instructions:
// -- Load "large" int constants into a virtual register.  What is large
//    depends on the type of instruction and on the target architecture.
// -- For any constants that cannot be put in an immediate field,
//    load address into virtual register first, and then load the constant.
// 
// firstOp and lastOp can be used to skip leading and trailing operands.
// If lastOp is 0, it defaults to #operands or #incoming Phi values.
//  
inline void
PreSelection::visitOperands(Instruction &I, int firstOp, int lastOp)
{
  // For any instruction other than PHI, copies go just before the instr.
  // For a PHI, operand copies must be before the terminator of the
  // appropriate predecessor basic block.  Remaining logic is simple
  // so just handle PHIs and other instructions separately.
  // 
  if (PHINode* phi = dyn_cast<PHINode>(&I))
    {
      if (lastOp == 0)
        lastOp = phi->getNumIncomingValues();
      for (unsigned i=firstOp, N=lastOp; i < N; ++i)
        this->visitOneOperand(I, phi->getIncomingValue(i),
                              phi->getOperandNumForIncomingValue(i),
                              * phi->getIncomingBlock(i)->getTerminator());
    }
  else
    {
      if (lastOp == 0)
        lastOp = I.getNumOperands();
      for (unsigned i=firstOp, N=lastOp; i < N; ++i)
        this->visitOneOperand(I, I.getOperand(i), i, I);
    }
}



// Common work for *all* instructions.  This needs to be called explicitly
// by other visit<InstructionType> functions.
inline void
PreSelection::visitInstruction(Instruction &I)
{ 
  visitOperands(I);              // Perform operand transformations
}


// GetElementPtr instructions: check if pointer is a global
void
PreSelection::visitGetElementPtrInst(GetElementPtrInst &I)
{ 
  // Check for a global and put its address into a register before this instr
  if (GetElementPtrInst* gep = getGlobalAddr(I.getPointerOperand(), I))
    I.setOperand(I.getPointerOperandIndex(), gep); // replace pointer operand

  // Decompose multidimensional array references
  DecomposeArrayRef(&I);

  // Perform other transformations common to all instructions
  visitInstruction(I);
}


// Load instructions: check if pointer is a global
void
PreSelection::visitLoadInst(LoadInst &I)
{ 
  // Check for a global and put its address into a register before this instr
  if (GetElementPtrInst* gep = getGlobalAddr(I.getPointerOperand(), I))
    I.setOperand(I.getPointerOperandIndex(), gep); // replace pointer operand

  // Perform other transformations common to all instructions
  visitInstruction(I);
}


// Store instructions: check if pointer is a global
void
PreSelection::visitStoreInst(StoreInst &I)
{ 
  // Check for a global and put its address into a register before this instr
  if (GetElementPtrInst* gep = getGlobalAddr(I.getPointerOperand(), I))
    I.setOperand(I.getPointerOperandIndex(), gep); // replace pointer operand

  // Perform other transformations common to all instructions
  visitInstruction(I);
}


// Cast instructions:
// -- check if argument is a global
// -- make multi-step casts explicit:
//    -- float/double to uint32_t:
//         If target does not have a float-to-unsigned instruction, we
//         need to convert to uint64_t and then to uint32_t, or we may
//         overflow the signed int representation for legal uint32_t
//         values.  Expand this without checking target.
// 
void
PreSelection::visitCastInst(CastInst &I)
{ 
  CastInst* castI = NULL;

  // Check for a global and put its address into a register before this instr
  if (GetElementPtrInst* gep = getGlobalAddr(I.getOperand(0), I))
    {
      I.setOperand(0, gep);             // replace pointer operand
    }
  else if (I.getType() == Type::UIntTy &&
           I.getOperand(0)->getType()->isFloatingPoint())
    { // insert a cast-fp-to-long before I, and then replace the operand of I
      castI = new CastInst(I.getOperand(0), Type::LongTy, "fp2Long2Uint", &I);
      I.setOperand(0, castI);           // replace fp operand with long
    }

  // Perform other transformations common to all instructions
  visitInstruction(I);
  if (castI)
    visitInstruction(*castI);
}

void
PreSelection::visitCallInst(CallInst &I)
{
  // Tell visitOperands to ignore the function name if this is a direct call.
  visitOperands(I, (/*firstOp=*/ I.getCalledFunction()? 1 : 0));
}


//===----------------------------------------------------------------------===//
// createPreSelectionPass - Public entrypoint for pre-selection pass
// and this file as a whole...
//
Pass*
createPreSelectionPass(TargetMachine &T)
{
  return new PreSelection(T);
}

