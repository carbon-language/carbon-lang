//===- TraceValues.cpp - Value Tracing for debugging -------------*- C++ -*--=//
//
// Support for inserting LLVM code to print values at basic block and method
// exits.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/TraceValues.h"
#include "llvm/GlobalVariable.h"
#include "llvm/ConstantVals.h"
#include "llvm/DerivedTypes.h"
#include "llvm/iMemory.h"
#include "llvm/iTerminators.h"
#include "llvm/iOther.h"
#include "llvm/Method.h"
#include "llvm/Module.h"
#include "llvm/SymbolTable.h"
#include "llvm/Assembly/Writer.h"
#include "Support/StringExtras.h"
#include <sstream>
using std::vector;
using std::string;

// Add a prototype for printf if it is not already in the program.
//
bool InsertTraceCode::doInitialization(Module *M) {
  SymbolTable *ST = M->getSymbolTable();
  const Type *SBP = PointerType::get(Type::SByteTy);
  const MethodType *MTy =
    MethodType::get(Type::IntTy, vector<const Type*>(1, SBP), true);

  if (Value *Meth = ST->lookup(PointerType::get(MTy), "printf")) {
    PrintfMeth = cast<Method>(Meth);
    return false;
  }

  // Create a new method and add it to the module
  PrintfMeth = new Method(MTy, false, "printf");
  M->getMethodList().push_back(PrintfMeth);
  return true;
}


static inline GlobalVariable *getStringRef(Module *M, const string &str) {
  // Create a constant internal string reference...
  Constant *Init = ConstantArray::get(str);
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
  switch (V->getType()->getPrimitiveID()) {
  case Type::BoolTyID:
  case Type::UByteTyID: case Type::UShortTyID:
  case Type::UIntTyID:  case Type::ULongTyID:
  case Type::SByteTyID: case Type::ShortTyID:
  case Type::IntTyID:   case Type::LongTyID:
    return "%d";
    
  case Type::FloatTyID: case Type::DoubleTyID:
    return "%g";

  case Type::LabelTyID: case Type::PointerTyID:
    return "%p";
    
  default:
    assert(0 && "Illegal value to print out...");
    return "";
  }
}


static void InsertPrintInst(Value *V, BasicBlock *BB, BasicBlock::iterator &BBI,
                            string Message, Method *Printf) {
  // Escape Message by replacing all % characters with %% chars.
  unsigned Offset = 0;
  while ((Offset = Message.find('%', Offset)) != string::npos) {
    Message.replace(Offset, 2, "%%");
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
                                   const string &Message, Method *Printf) {
  std::ostringstream OutStr;
  if (V) WriteAsOperand(OutStr, V);
  InsertPrintInst(V, BB, BBI, Message+OutStr.str()+" = ", Printf);
}


// Insert print instructions at the end of the basic block *bb
// for each value in valueVec[] that is live at the end of that basic block,
// or that is stored to memory in this basic block.
// If the value is stored to memory, we load it back before printing
// We also return all such loaded values in the vector valuesStoredInMethod
// for printing at the exit from the method.  (Note that in each invocation
// of the method, this will only get the last value stored for each static
// store instruction).
// *bb must be the block in which the value is computed;
// this is not checked here.
// 
static void TraceValuesAtBBExit(BasicBlock *BB, Method *Printf,
                                vector<Instruction*> *valuesStoredInMethod) {
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
      assert(valuesStoredInMethod &&
             "Should not be printing a store instruction at method exit");
      LoadInst *LI = new LoadInst(SI->getPointerOperand(), SI->copyIndices(),
                                  "reload");
      InsertPos = BB->getInstList().insert(InsertPos, LI) + 1;
      valuesStoredInMethod->push_back(LI);
    }
    if (ShouldTraceValue(I))
      InsertVerbosePrintInst(I, BB, InsertPos, "  ", Printf);
  }
}

static inline void InsertCodeToShowMethodEntry(Method *M, Method *Printf) {
  // Get an iterator to point to the insertion location
  BasicBlock *BB = M->getEntryNode();
  BasicBlock::iterator BBI = BB->begin();

  std::ostringstream OutStr;
  WriteAsOperand(OutStr, M, true);
  InsertPrintInst(0, BB, BBI, "ENTERING METHOD: " + OutStr.str(), Printf);

  // Now print all the incoming arguments
  const Method::ArgumentListType &argList = M->getArgumentList();
  unsigned ArgNo = 0;
  for (Method::ArgumentListType::const_iterator
         I = argList.begin(), E = argList.end(); I != E; ++I, ++ArgNo) {
    InsertVerbosePrintInst(*I, BB, BBI,
                           "  Arg #" + utostr(ArgNo), Printf);
  }
}


static inline void InsertCodeToShowMethodExit(BasicBlock *BB, Method *Printf) {
  // Get an iterator to point to the insertion location
  BasicBlock::iterator BBI = BB->end()-1;
  ReturnInst *Ret = cast<ReturnInst>(*BBI);
  
  std::ostringstream OutStr;
  WriteAsOperand(OutStr, BB->getParent(), true);
  InsertPrintInst(0, BB, BBI, "LEAVING  METHOD: " + OutStr.str(), Printf);
  
  // print the return value, if any
  if (BB->getParent()->getReturnType() != Type::VoidTy)
    InsertPrintInst(Ret->getReturnValue(), BB, BBI, "  Returning: ", Printf);
}


bool InsertTraceCode::doit(Method *M, bool traceBasicBlockExits,
                           bool traceMethodEvents, Method *Printf) {
  if (!traceBasicBlockExits && !traceMethodEvents)
    return false;

  vector<Instruction*> valuesStoredInMethod;
  vector<BasicBlock*>  exitBlocks;

  if (traceMethodEvents)
    InsertCodeToShowMethodEntry(M, Printf);
  
  for (Method::iterator BI = M->begin(); BI != M->end(); ++BI) {
    BasicBlock *BB = *BI;
    if (isa<ReturnInst>(BB->getTerminator()))
      exitBlocks.push_back(BB); // record this as an exit block
    
    if (traceBasicBlockExits)
      TraceValuesAtBBExit(BB, Printf, &valuesStoredInMethod);
  }

  if (traceMethodEvents)
    for (unsigned i=0; i < exitBlocks.size(); ++i) {
#if 0
      TraceValuesAtBBExit(valuesStoredInMethod, exitBlocks[i], module,
                          /*indent*/ 0, /*isMethodExit*/ true,
                          /*valuesStoredInMethod*/ NULL);
#endif
      InsertCodeToShowMethodExit(exitBlocks[i], Printf);
    }

  return true;
}
