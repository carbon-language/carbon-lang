//===- TraceValues.cpp - Value Tracing for debugging -------------*- C++ -*--=//
//
// Support for inserting LLVM code to print values at basic block and function
// exits.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/TraceValues.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/iMemory.h"
#include "llvm/iTerminators.h"
#include "llvm/iOther.h"
#include "llvm/BasicBlock.h"
#include "llvm/Function.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Assembly/Writer.h"
#include "Support/StringExtras.h"
#include <sstream>
using std::vector;
using std::string;

namespace {
  class InsertTraceCode : public FunctionPass {
    bool TraceBasicBlockExits, TraceFunctionExits;
    Function *PrintfFunc;
  public:
    InsertTraceCode(bool traceBasicBlockExits, bool traceFunctionExits)
      : TraceBasicBlockExits(traceBasicBlockExits), 
        TraceFunctionExits(traceFunctionExits) {}

    const char *getPassName() const { return "Trace Code Insertion"; }
    
    // Add a prototype for printf if it is not already in the program.
    //
    bool doInitialization(Module *M);
    
    //--------------------------------------------------------------------------
    // Function InsertCodeToTraceValues
    // 
    // Inserts tracing code for all live values at basic block and/or function
    // exits as specified by `traceBasicBlockExits' and `traceFunctionExits'.
    //
    static bool doit(Function *M, bool traceBasicBlockExits,
                     bool traceFunctionExits, Function *Printf);
    
    // runOnFunction - This method does the work.
    //
    bool runOnFunction(Function *F) {
      return doit(F, TraceBasicBlockExits, TraceFunctionExits, PrintfFunc);
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.preservesCFG();
    }
  };
} // end anonymous namespace


Pass *createTraceValuesPassForFunction() {     // Just trace functions
  return new InsertTraceCode(false, true);
}

Pass *createTraceValuesPassForBasicBlocks() {  // Trace BB's and functions
  return new InsertTraceCode(true, true);
}




// Add a prototype for printf if it is not already in the program.
//
bool InsertTraceCode::doInitialization(Module *M) {
  const Type *SBP = PointerType::get(Type::SByteTy);
  const FunctionType *MTy =
    FunctionType::get(Type::IntTy, vector<const Type*>(1, SBP), true);

  PrintfFunc = M->getOrInsertFunction("printf", MTy);
  return false;
}


static inline GlobalVariable *getStringRef(Module *M, const string &str) {
  // Create a constant internal string reference...
  Constant *Init = ConstantArray::get(str);

  // Create the global variable and record it in the module
  // The GV will be renamed to a unique name if needed.
  GlobalVariable *GV = new GlobalVariable(Init->getType(), true, true, Init,
                                          "trstr");
  M->getGlobalList().push_back(GV);
  return GV;
}


// 
// Check if this instruction has any uses outside its basic block,
// or if it used by either a Call or Return instruction.
// 
static inline bool LiveAtBBExit(const Instruction* I) {
  const BasicBlock *BB = I->getParent();
  for (Value::use_const_iterator U = I->use_begin(); U != I->use_end(); ++U)
    if (const Instruction *UI = dyn_cast<Instruction>(*U))
      if (UI->getParent() != BB || isa<ReturnInst>(UI))
        return true;

  return false;
}


static inline bool TraceThisOpCode(unsigned opCode) {
  // Explicitly test for opCodes *not* to trace so that any new opcodes will
  // be traced by default (VoidTy's are already excluded)
  // 
  return (opCode  < Instruction::FirstOtherOp &&
          opCode != Instruction::Alloca &&
          opCode != Instruction::PHINode &&
          opCode != Instruction::Cast);
}


static bool ShouldTraceValue(const Instruction *I) {
  return
    I->getType() != Type::VoidTy && LiveAtBBExit(I) &&
    TraceThisOpCode(I->getOpcode());
}

static string getPrintfCodeFor(const Value *V) {
  if (V == 0) return "";
  if (V->getType()->isFloatingPoint())
    return "%g";
  else if (V->getType() == Type::LabelTy || isa<PointerType>(V->getType()))
    return "0x%p";
  else if (V->getType()->isIntegral() || V->getType() == Type::BoolTy)
    return "%d";
    
  assert(0 && "Illegal value to print out...");
  return "";
}


static void InsertPrintInst(Value *V, BasicBlock *BB, BasicBlock::iterator &BBI,
                            string Message, Function *Printf) {
  // Escape Message by replacing all % characters with %% chars.
  unsigned Offset = 0;
  while ((Offset = Message.find('%', Offset)) != string::npos) {
    Message.replace(Offset, 1, "%%");
    Offset += 2;  // Skip over the new %'s
  }

  Module *Mod = BB->getParent()->getParent();

  // Turn the marker string into a global variable...
  GlobalVariable *fmtVal = getStringRef(Mod, Message+getPrintfCodeFor(V)+"\n");

  // Turn the format string into an sbyte *
  Instruction *GEP = 
    new GetElementPtrInst(fmtVal,
                          vector<Value*>(2,ConstantUInt::get(Type::UIntTy, 0)),
                          "trstr");
  BBI = BB->getInstList().insert(BBI, GEP)+1;
  
  // Insert the first print instruction to print the string flag:
  vector<Value*> PrintArgs;
  PrintArgs.push_back(GEP);
  if (V) PrintArgs.push_back(V);
  Instruction *I = new CallInst(Printf, PrintArgs, "trace");
  BBI = BB->getInstList().insert(BBI, I)+1;
}
                            

static void InsertVerbosePrintInst(Value *V, BasicBlock *BB,
                                   BasicBlock::iterator &BBI,
                                   const string &Message, Function *Printf) {
  std::ostringstream OutStr;
  if (V) WriteAsOperand(OutStr, V);
  InsertPrintInst(V, BB, BBI, Message+OutStr.str()+" = ", Printf);
}


// Insert print instructions at the end of the basic block *bb
// for each value in valueVec[] that is live at the end of that basic block,
// or that is stored to memory in this basic block.
// If the value is stored to memory, we load it back before printing
// We also return all such loaded values in the vector valuesStoredInFunction
// for printing at the exit from the function.  (Note that in each invocation
// of the function, this will only get the last value stored for each static
// store instruction).
// *bb must be the block in which the value is computed;
// this is not checked here.
// 
static void TraceValuesAtBBExit(BasicBlock *BB, Function *Printf,
                                vector<Instruction*> *valuesStoredInFunction) {
  // Get an iterator to point to the insertion location, which is
  // just before the terminator instruction.
  // 
  BasicBlock::iterator InsertPos = BB->end()-1;
  assert((*InsertPos)->isTerminator());
  
  // If the terminator is a conditional branch, insert the trace code just
  // before the instruction that computes the branch condition (just to
  // avoid putting a call between the CC-setting instruction and the branch).
  // Use laterInstrSet to mark instructions that come after the setCC instr
  // because those cannot be traced at the location we choose.
  // 
  Instruction *SetCC = 0;
  if (BranchInst *Branch = dyn_cast<BranchInst>(BB->getTerminator()))
    if (!Branch->isUnconditional())
      if (Instruction *I = dyn_cast<Instruction>(Branch->getCondition()))
        if (I->getParent() == BB) {
          SetCC = I;
          while (*InsertPos != SetCC)
            --InsertPos;        // Back up until we can insert before the setcc
        }

  // Copy all of the instructions into a vector to avoid problems with Setcc
  const vector<Instruction*> Insts(BB->begin(), InsertPos);

  std::ostringstream OutStr;
  WriteAsOperand(OutStr, BB, false);
  InsertPrintInst(0, BB, InsertPos, "LEAVING BB:" + OutStr.str(), Printf);

  // Insert a print instruction for each value.
  // 
  for (vector<Instruction*>::const_iterator II = Insts.begin(),
         IE = Insts.end(); II != IE; ++II) {
    Instruction *I = *II;
    if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
      assert(valuesStoredInFunction &&
             "Should not be printing a store instruction at function exit");
      LoadInst *LI = new LoadInst(SI->getPointerOperand(), SI->copyIndices(),
                                  "reload");
      InsertPos = BB->getInstList().insert(InsertPos, LI) + 1;
      valuesStoredInFunction->push_back(LI);
    }
    if (ShouldTraceValue(I))
      InsertVerbosePrintInst(I, BB, InsertPos, "  ", Printf);
  }
}

static inline void InsertCodeToShowFunctionEntry(Function *M, Function *Printf){
  // Get an iterator to point to the insertion location
  BasicBlock *BB = M->getEntryNode();
  BasicBlock::iterator BBI = BB->begin();

  std::ostringstream OutStr;
  WriteAsOperand(OutStr, M, true);
  InsertPrintInst(0, BB, BBI, "ENTERING FUNCTION: " + OutStr.str(), Printf);

  // Now print all the incoming arguments
  const Function::ArgumentListType &argList = M->getArgumentList();
  unsigned ArgNo = 0;
  for (Function::ArgumentListType::const_iterator
         I = argList.begin(), E = argList.end(); I != E; ++I, ++ArgNo) {
    InsertVerbosePrintInst((Value*)*I, BB, BBI,
                           "  Arg #" + utostr(ArgNo), Printf);
  }
}


static inline void InsertCodeToShowFunctionExit(BasicBlock *BB,
                                                Function *Printf) {
  // Get an iterator to point to the insertion location
  BasicBlock::iterator BBI = BB->end()-1;
  ReturnInst *Ret = cast<ReturnInst>(*BBI);
  
  std::ostringstream OutStr;
  WriteAsOperand(OutStr, BB->getParent(), true);
  InsertPrintInst(0, BB, BBI, "LEAVING  FUNCTION: " + OutStr.str(), Printf);
  
  // print the return value, if any
  if (BB->getParent()->getReturnType() != Type::VoidTy)
    InsertPrintInst(Ret->getReturnValue(), BB, BBI, "  Returning: ", Printf);
}


bool InsertTraceCode::doit(Function *M, bool traceBasicBlockExits,
                           bool traceFunctionEvents, Function *Printf) {
  if (!traceBasicBlockExits && !traceFunctionEvents)
    return false;

  vector<Instruction*> valuesStoredInFunction;
  vector<BasicBlock*>  exitBlocks;

  if (traceFunctionEvents)
    InsertCodeToShowFunctionEntry(M, Printf);
  
  for (Function::iterator BI = M->begin(); BI != M->end(); ++BI) {
    BasicBlock *BB = *BI;
    if (isa<ReturnInst>(BB->getTerminator()))
      exitBlocks.push_back(BB); // record this as an exit block
    
    if (traceBasicBlockExits)
      TraceValuesAtBBExit(BB, Printf, &valuesStoredInFunction);
  }

  if (traceFunctionEvents)
    for (unsigned i=0; i < exitBlocks.size(); ++i) {
#if 0
      TraceValuesAtBBExit(valuesStoredInFunction, exitBlocks[i], module,
                          /*indent*/ 0, /*isFunctionExit*/ true,
                          /*valuesStoredInFunction*/ NULL);
#endif
      InsertCodeToShowFunctionExit(exitBlocks[i], Printf);
    }

  return true;
}
