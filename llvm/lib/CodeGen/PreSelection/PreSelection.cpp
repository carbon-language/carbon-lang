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
#include "llvm/Target/MachineInstrInfo.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Support/InstVisitor.h"
#include "llvm/Module.h"
#include "llvm/Function.h"
#include "llvm/Constants.h"
#include "llvm/GlobalVariable.h"
#include "llvm/iMemory.h"
#include "llvm/iPHINode.h"
#include "llvm/iOther.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Pass.h"
#include "llvm/Annotation.h"
#include "Support/CommandLine.h"
#include "Support/NonCopyable.h"
using std::map;
using std::cerr;

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
  // this as as annotation on the Module object.
  // A single GlobalVariable is created for each constant in the pool
  // representing the memory for that constant.  
  // 
  static AnnotationID CPFM_AID(
                 AnnotationManager::getID("CodeGen::ConstantPoolForModule"));

  class ConstantPoolForModule: private Annotation, public NonCopyable {
    Module* myModule;
    std::map<const Constant*, GlobalVariable*> gvars;
    std::map<const Constant*, GlobalVariable*> origGVars;
    ConstantPoolForModule(Module* M);   // called only by annotation builder
    ConstantPoolForModule();            // do not implement
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
              GV = new GlobalVariable(CV->getType(), true,true,CV); //put in map
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
    Function* function;

    GetElementPtrInst* getGlobalAddr(Value* ptr, Instruction* insertBefore = 0);

    GlobalVariable* getGlobalForConstant(Constant* CV) {
      Module* M = function->getParent();
      return ConstantPoolForModule::get(M).getGlobalForConstant(CV);
    }

  public:
    PreSelection (const TargetMachine &T): target(T), function(NULL) {}

    // runOnBasicBlock - apply this pass to each BB
    bool runOnBasicBlock(BasicBlock &BB) {
      function = BB.getParent();
      this->visit(BB);
      return true;
    }

    bool doFinalization(Function &F) {
      if (PreSelectDebugLevel >= PreSelect_PrintOutput)
        cerr << "\n\n*** LLVM code after pre-selection for function "
             << F.getName() << ":\n\n" << F;
      return false;
    }

    // These methods do the actual work of specializing code
    void visitInstruction(Instruction &I);   // common work for every instr. 
    void visitGetElementPtrInst(GetElementPtrInst &I);
    void visitLoadInst(LoadInst &I);
    void visitStoreInst(StoreInst &I);

    // Helper functions for visiting operands of every instruction
    void visitOperands(Instruction &I);    // work on all operands of instr.
    void visitOneOperand(Instruction &I, Constant* CV, unsigned opNum,
                         Instruction& insertBefore); // iworks on one operand
  };
}  // end anonymous namespace


// Register the pass...
static RegisterOpt<PreSelection> X("preselect",
                                   "Specialize LLVM code for a target machine",
                                   createPreSelectionPass);

// PreSelection::getGlobalAddr: Put address of a global into a v. register.
GetElementPtrInst* 
PreSelection::getGlobalAddr(Value* ptr, Instruction* insertBefore = 0)
{
  return (isa<GlobalValue>(ptr))
    ? new GetElementPtrInst(ptr,
                    std::vector<Value*>(1, ConstantSInt::get(Type::LongTy, 0U)),
                    "addrOfGlobal", insertBefore)
    : NULL;
}


//------------------------------------------------------------------------------
// Instruction visitor methods to perform instruction-specific operations
//------------------------------------------------------------------------------

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
  if (GetElementPtrInst* gep = getGlobalAddr(I.getPointerOperand(), &I))
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
  if (GetElementPtrInst* gep = getGlobalAddr(I.getPointerOperand(), &I))
    I.setOperand(I.getPointerOperandIndex(), gep); // replace pointer operand

  // Perform other transformations common to all instructions
  visitInstruction(I);
}


// Store instructions: check if pointer is a global
void
PreSelection::visitStoreInst(StoreInst &I)
{ 
  // Check for a global and put its address into a register before this instr
  if (GetElementPtrInst* gep = getGlobalAddr(I.getPointerOperand(), &I))
    I.setOperand(I.getPointerOperandIndex(), gep); // replace pointer operand

  // Perform other transformations common to all instructions
  visitInstruction(I);
}


// visitOperands() transforms individual operands of all instructions:
// -- Load "large" int constants into a virtual register.  What is large
//    depends on the type of instruction and on the target architecture.
// -- For any constants that cannot be put in an immediate field,
//    load address into virtual register first, and then load the constant.
// 
void
PreSelection::visitOperands(Instruction &I)
{
  // For any instruction other than PHI, copies go just before the instr.
  // For a PHI, operand copies must be before the terminator of the
  // appropriate predecessor basic block.  Remaining logic is simple
  // so just handle PHIs and other instructions separately.
  // 
  if (PHINode* phi = dyn_cast<PHINode>(&I))
    {
      for (unsigned i=0, N=phi->getNumIncomingValues(); i < N; ++i)
        if (Constant* CV = dyn_cast<Constant>(phi->getIncomingValue(i)))
          this->visitOneOperand(I, CV, phi->getOperandNumForIncomingValue(i),
                                * phi->getIncomingBlock(i)->getTerminator());
    }
  else
    for (unsigned i=0, N=I.getNumOperands(); i < N; ++i)
      if (Constant* CV = dyn_cast<Constant>(I.getOperand(i)))
        this->visitOneOperand(I, CV, i, I);
}

void
PreSelection::visitOneOperand(Instruction &I, Constant* CV, unsigned opNum,
                              Instruction& insertBefore)
{
  if (target.getInstrInfo().ConstantTypeMustBeLoaded(CV))
    { // load address of constant into a register, then load the constant
      GetElementPtrInst* gep = getGlobalAddr(getGlobalForConstant(CV),
                                             &insertBefore);
      LoadInst* ldI = new LoadInst(gep, "loadConst", &insertBefore);
      I.setOperand(opNum, ldI);        // replace operand with copy in v.reg.
    }
  else if (target.getInstrInfo().ConstantMayNotFitInImmedField(CV, &I))
    { // put the constant into a virtual register using a cast
      CastInst* castI = new CastInst(CV, CV->getType(), "copyConst",
                                     &insertBefore);
      I.setOperand(opNum, castI);      // replace operand with copy in v.reg.
    }
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

