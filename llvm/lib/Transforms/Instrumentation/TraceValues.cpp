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
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Assembly/Writer.h"
#include "Support/CommandLine.h"
#include "Support/StringExtras.h"
#include <algorithm>
#include <sstream>
using std::vector;
using std::string;

static cl::opt<bool>
DisablePtrHashing("tracedisablehashdisable", cl::Hidden,
                  cl::desc("Disable pointer hashing"));

static cl::list<string>
TraceFuncName("tracefunc", cl::desc("trace only specific functions"),
              cl::value_desc("function"));


// We trace a particular function if no functions to trace were specified
// or if the function is in the specified list.
// 
inline bool
TraceThisFunction(Function* func)
{
  if (TraceFuncName.size() == 0)
    return true;

  return std::find(TraceFuncName.begin(), TraceFuncName.end(), func->getName())
                  != TraceFuncName.end();
}


namespace {
  struct ExternalFuncs {
    Function *PrintfFunc, *HashPtrFunc, *ReleasePtrFunc;
    Function *RecordPtrFunc, *PushOnEntryFunc, *ReleaseOnReturnFunc;
    void doInitialization(Module &M); // Add prototypes for external functions
  };
  
  class InsertTraceCode : public FunctionPass {
    bool TraceBasicBlockExits, TraceFunctionExits;
    ExternalFuncs externalFuncs;
  public:
    InsertTraceCode(bool traceBasicBlockExits, bool traceFunctionExits)
      : TraceBasicBlockExits(traceBasicBlockExits), 
        TraceFunctionExits(traceFunctionExits) {}

    const char *getPassName() const { return "Trace Code Insertion"; }
    
    // Add a prototype for runtime functions not already in the program.
    //
    bool doInitialization(Module &M);
    
    //--------------------------------------------------------------------------
    // Function InsertCodeToTraceValues
    // 
    // Inserts tracing code for all live values at basic block and/or function
    // exits as specified by `traceBasicBlockExits' and `traceFunctionExits'.
    //
    static bool doit(Function *M, bool traceBasicBlockExits,
                     bool traceFunctionExits, ExternalFuncs& externalFuncs);

    // runOnFunction - This method does the work.
    //
    bool runOnFunction(Function &F) {
      return doit(&F, TraceBasicBlockExits, TraceFunctionExits, externalFuncs);
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

// Add a prototype for external functions used by the tracing code.
//
void ExternalFuncs::doInitialization(Module &M) {
  const Type *SBP = PointerType::get(Type::SByteTy);
  const FunctionType *MTy =
    FunctionType::get(Type::IntTy, vector<const Type*>(1, SBP), true);
  PrintfFunc = M.getOrInsertFunction("printf", MTy);

  // uint (sbyte*)
  const FunctionType *hashFuncTy =
    FunctionType::get(Type::UIntTy, vector<const Type*>(1, SBP), false);
  HashPtrFunc = M.getOrInsertFunction("HashPointerToSeqNum", hashFuncTy);
  
  // void (sbyte*)
  const FunctionType *voidSBPFuncTy =
    FunctionType::get(Type::VoidTy, vector<const Type*>(1, SBP), false);
  
  ReleasePtrFunc = M.getOrInsertFunction("ReleasePointerSeqNum", voidSBPFuncTy);
  RecordPtrFunc  = M.getOrInsertFunction("RecordPointer", voidSBPFuncTy);
  
  const FunctionType *voidvoidFuncTy =
    FunctionType::get(Type::VoidTy, vector<const Type*>(), false);
  
  PushOnEntryFunc = M.getOrInsertFunction("PushPointerSet", voidvoidFuncTy);
  ReleaseOnReturnFunc = M.getOrInsertFunction("ReleasePointersPopSet",
                                               voidvoidFuncTy);
}


// Add a prototype for external functions used by the tracing code.
//
bool InsertTraceCode::doInitialization(Module &M) {
  externalFuncs.doInitialization(M);
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
  else if (V->getType() == Type::LabelTy)
    return "0x%p";
  else if (isa<PointerType>(V->getType()))
    return DisablePtrHashing ? "0x%p" : "%d";
  else if (V->getType()->isIntegral() || V->getType() == Type::BoolTy)
    return "%d";
  
  assert(0 && "Illegal value to print out...");
  return "";
}


static void InsertPrintInst(Value *V,BasicBlock *BB, BasicBlock::iterator &BBI,
                            string Message,
                            Function *Printf, Function* HashPtrToSeqNum) {
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
  BBI = ++BB->getInstList().insert(BBI, GEP);
  
  // Insert a call to the hash function if this is a pointer value
  if (V && isa<PointerType>(V->getType()) && !DisablePtrHashing) {
    const Type *SBP = PointerType::get(Type::SByteTy);
    if (V->getType() != SBP) {   // Cast pointer to be sbyte*
      Instruction *I = new CastInst(V, SBP, "Hash_cast");
      BBI = ++BB->getInstList().insert(BBI, I);
      V = I;
    }

    vector<Value*> HashArgs(1, V);
    V = new CallInst(HashPtrToSeqNum, HashArgs, "ptrSeqNum");
    BBI = ++BB->getInstList().insert(BBI, cast<Instruction>(V));
  }
  
  // Insert the first print instruction to print the string flag:
  vector<Value*> PrintArgs;
  PrintArgs.push_back(GEP);
  if (V) PrintArgs.push_back(V);
  Instruction *I = new CallInst(Printf, PrintArgs, "trace");
  BBI = ++BB->getInstList().insert(BBI, I);
}
                            

static void InsertVerbosePrintInst(Value *V, BasicBlock *BB,
                                   BasicBlock::iterator &BBI,
                                   const string &Message, Function *Printf,
                                   Function* HashPtrToSeqNum) {
  std::ostringstream OutStr;
  if (V) WriteAsOperand(OutStr, V);
  InsertPrintInst(V, BB, BBI, Message+OutStr.str()+" = ",
                  Printf, HashPtrToSeqNum);
}

static void 
InsertReleaseInst(Value *V, BasicBlock *BB,
                  BasicBlock::iterator &BBI,
                  Function* ReleasePtrFunc) {
  
  const Type *SBP = PointerType::get(Type::SByteTy);
  if (V->getType() != SBP) {   // Cast pointer to be sbyte*
    Instruction *I = new CastInst(V, SBP, "RPSN_cast");
    BBI = ++BB->getInstList().insert(BBI, I);
    V = I;
  }
  vector<Value*> releaseArgs(1, V);
  Instruction *I = new CallInst(ReleasePtrFunc, releaseArgs);
  BBI = ++BB->getInstList().insert(BBI, I);
}

static void 
InsertRecordInst(Value *V, BasicBlock *BB,
                 BasicBlock::iterator &BBI,
                 Function* RecordPtrFunc) {
    const Type *SBP = PointerType::get(Type::SByteTy);
  if (V->getType() != SBP) {   // Cast pointer to be sbyte*
    Instruction *I = new CastInst(V, SBP, "RP_cast");
    BBI = ++BB->getInstList().insert(BBI, I);
    V = I;
  }
  vector<Value*> releaseArgs(1, V);
  Instruction *I = new CallInst(RecordPtrFunc, releaseArgs);
  BBI = ++BB->getInstList().insert(BBI, I);
}

static void
InsertPushOnEntryFunc(Function *M,
                      Function* PushOnEntryFunc) {
  // Get an iterator to point to the insertion location
  BasicBlock &BB = M->getEntryNode();
  BB.getInstList().insert(BB.begin(), new CallInst(PushOnEntryFunc,
                                                   vector<Value*>()));
}

static void 
InsertReleaseRecordedInst(BasicBlock *BB,
                          Function* ReleaseOnReturnFunc) {
  BasicBlock::iterator BBI = --BB->end();
  BBI = ++BB->getInstList().insert(BBI, new CallInst(ReleaseOnReturnFunc,
                                                     vector<Value*>()));
}

// Look for alloca and free instructions. These are the ptrs to release.
// Release the free'd pointers immediately.  Record the alloca'd pointers
// to be released on return from the current function.
// 
static void
ReleasePtrSeqNumbers(BasicBlock *BB,
                     ExternalFuncs& externalFuncs) {
  
  for (BasicBlock::iterator II=BB->begin(); II != BB->end(); ++II) {
    if (FreeInst *FI = dyn_cast<FreeInst>(&*II))
      InsertReleaseInst(FI->getOperand(0), BB,II,externalFuncs.ReleasePtrFunc);
    else if (AllocaInst *AI = dyn_cast<AllocaInst>(&*II))
      {
        BasicBlock::iterator nextI = ++II;
        InsertRecordInst(AI, BB, nextI, externalFuncs.RecordPtrFunc);     
        II = --nextI;
      }
  }
}  


// Insert print instructions at the end of basic block BB for each value
// computed in BB that is live at the end of BB,
// or that is stored to memory in BB.
// If the value is stored to memory, we load it back before printing it
// We also return all such loaded values in the vector valuesStoredInFunction
// for printing at the exit from the function.  (Note that in each invocation
// of the function, this will only get the last value stored for each static
// store instruction).
// 
static void TraceValuesAtBBExit(BasicBlock *BB,
                                Function *Printf, Function* HashPtrToSeqNum,
                                vector<Instruction*> *valuesStoredInFunction) {
  // Get an iterator to point to the insertion location, which is
  // just before the terminator instruction.
  // 
  BasicBlock::iterator InsertPos = --BB->end();
  assert(InsertPos->isTerminator());
  
#undef CANNOT_SAVE_CCR_ACROSS_CALLS
#ifdef CANNOT_SAVE_CCR_ACROSS_CALLS
  // 
  // *** DISABLING THIS BECAUSE SAVING %CCR ACROSS CALLS WORKS NOW.
  // *** DELETE THIS CODE AFTER SOME TESTING.
  // *** NOTE: THIS CODE IS BROKEN ANYWAY WHEN THE SETCC IS NOT JUST
  // ***       BEFORE THE BRANCH.
  // -- Vikram Adve, 7/7/02.
  // 
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
          InsertPos = SetCC = I; // Back up until we can insert before the setcc
        }
#endif CANNOT_SAVE_CCR_ACROSS_CALLS

  std::ostringstream OutStr;
  WriteAsOperand(OutStr, BB, false);
  InsertPrintInst(0, BB, InsertPos, "LEAVING BB:" + OutStr.str(),
                  Printf, HashPtrToSeqNum);

  // Insert a print instruction for each instruction preceding InsertPos.
  // The print instructions must go before InsertPos, so we use the
  // instruction *preceding* InsertPos to check when to terminate the loop.
  // 
  if (InsertPos != BB->begin()) { // there's at least one instr before InsertPos
    BasicBlock::iterator II = BB->begin(), IEincl = InsertPos;
    --IEincl;
    do {                          // do from II up to IEincl, inclusive
      if (StoreInst *SI = dyn_cast<StoreInst>(&*II)) {
        assert(valuesStoredInFunction &&
               "Should not be printing a store instruction at function exit");
        LoadInst *LI = new LoadInst(SI->getPointerOperand(), SI->copyIndices(),
                                  "reload."+SI->getPointerOperand()->getName());
        InsertPos = ++BB->getInstList().insert(InsertPos, LI);
        valuesStoredInFunction->push_back(LI);
      }
      if (ShouldTraceValue(II))
        InsertVerbosePrintInst(II, BB, InsertPos, "  ", Printf,HashPtrToSeqNum);
    } while (II++ != IEincl);
  }
}

static inline void InsertCodeToShowFunctionEntry(Function *M, Function *Printf,
                                                 Function* HashPtrToSeqNum){
  // Get an iterator to point to the insertion location
  BasicBlock &BB = M->getEntryNode();
  BasicBlock::iterator BBI = BB.begin();

  std::ostringstream OutStr;
  WriteAsOperand(OutStr, M, true);
  InsertPrintInst(0, &BB, BBI, "ENTERING FUNCTION: " + OutStr.str(),
                  Printf, HashPtrToSeqNum);

  // Now print all the incoming arguments
  unsigned ArgNo = 0;
  for (Function::aiterator I = M->abegin(), E = M->aend(); I != E; ++I,++ArgNo){
    InsertVerbosePrintInst(I, &BB, BBI,
                           "  Arg #" + utostr(ArgNo) + ": ", Printf,
                           HashPtrToSeqNum);
  }
}


static inline void InsertCodeToShowFunctionExit(BasicBlock *BB,
                                                Function *Printf,
                                                Function* HashPtrToSeqNum) {
  // Get an iterator to point to the insertion location
  BasicBlock::iterator BBI = --BB->end();
  ReturnInst &Ret = cast<ReturnInst>(BB->back());
  
  std::ostringstream OutStr;
  WriteAsOperand(OutStr, BB->getParent(), true);
  InsertPrintInst(0, BB, BBI, "LEAVING  FUNCTION: " + OutStr.str(),
                  Printf, HashPtrToSeqNum);
  
  // print the return value, if any
  if (BB->getParent()->getReturnType() != Type::VoidTy)
    InsertPrintInst(Ret.getReturnValue(), BB, BBI, "  Returning: ",
                    Printf, HashPtrToSeqNum);
}


bool InsertTraceCode::doit(Function *M, bool traceBasicBlockExits,
                           bool traceFunctionEvents,
                           ExternalFuncs& externalFuncs) {
  if (!traceBasicBlockExits && !traceFunctionEvents)
    return false;

  if (!TraceThisFunction(M))
    return false;
  
  vector<Instruction*> valuesStoredInFunction;
  vector<BasicBlock*>  exitBlocks;

  // Insert code to trace values at function entry
  if (traceFunctionEvents)
    InsertCodeToShowFunctionEntry(M, externalFuncs.PrintfFunc,
                                  externalFuncs.HashPtrFunc);
  
  // Push a pointer set for recording alloca'd pointers at entry.
  if (!DisablePtrHashing)
    InsertPushOnEntryFunc(M, externalFuncs.PushOnEntryFunc);
  
  for (Function::iterator BB = M->begin(); BB != M->end(); ++BB) {
    if (isa<ReturnInst>(BB->getTerminator()))
      exitBlocks.push_back(BB); // record this as an exit block
    
    if (traceBasicBlockExits)
      TraceValuesAtBBExit(BB, externalFuncs.PrintfFunc,
                          externalFuncs.HashPtrFunc, &valuesStoredInFunction);
    
    if (!DisablePtrHashing)          // release seq. numbers on free/ret
      ReleasePtrSeqNumbers(BB, externalFuncs);
  }
  
  for (unsigned i=0; i < exitBlocks.size(); ++i)
    {
      // Insert code to trace values at function exit
      if (traceFunctionEvents)
        InsertCodeToShowFunctionExit(exitBlocks[i], externalFuncs.PrintfFunc,
                                     externalFuncs.HashPtrFunc);
      
      // Release all recorded pointers before RETURN.  Do this LAST!
      if (!DisablePtrHashing)
        InsertReleaseRecordedInst(exitBlocks[i],
                                  externalFuncs.ReleaseOnReturnFunc);
    }
  
  return true;
}
