// $Id$
//***************************************************************************
// File:
//	TraceValues.cpp
// 
// Purpose:
//      Support for inserting LLVM code to print values at basic block
//      and method exits.  Also exports functions to create a call
//      "printf" instruction with one of the signatures listed below.
// 
// History:
//	10/11/01	 -  Vikram Adve  -  Created
//**************************************************************************/


#include "llvm/Transforms/Instrumentation/TraceValues.h"
#include "llvm/GlobalVariable.h"
#include "llvm/ConstPoolVals.h"
#include "llvm/Type.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instruction.h"
#include "llvm/iTerminators.h"
#include "llvm/iOther.h"
#include "llvm/BasicBlock.h"
#include "llvm/Method.h"
#include "llvm/Module.h"
#include "llvm/SymbolTable.h"
#include <strstream>
#include "llvm/Assembly/Writer.h"

static inline GlobalVariable *
GetStringRef(Module *M, const string &str)
{
  ConstPoolArray *Init = ConstPoolArray::get(str);
  GlobalVariable *V = new GlobalVariable(Init->getType(), /*Const*/true, Init);
  M->getGlobalList().push_back(V);

  return V;
}

static inline bool
TraceThisOpCode(unsigned opCode)
{
  // Explicitly test for opCodes *not* to trace so that any new opcodes will
  // be traced by default (VoidTy's are already excluded)
  // 
  return (opCode  < Instruction::FirstOtherOp &&
          opCode != Instruction::Alloca &&
          opCode != Instruction::PHINode &&
          opCode != Instruction::Cast);
}


static void 
FindValuesToTraceInBB(BasicBlock* bb, vector<Value*>& valuesToTraceInBB)
{
  for (BasicBlock::iterator II = bb->begin(); II != bb->end(); ++II)
    if ((*II)->getType()->isPrimitiveType() && 
        (*II)->getType() != Type::VoidTy &&
        TraceThisOpCode((*II)->getOpcode()))
      {
        valuesToTraceInBB.push_back(*II);
      }
}

// The invocation should be:
//       call "printVal"(value).
// 
static Value *GetPrintMethodForType(Module *Mod, const Type *VTy) {
  MethodType *MTy = MethodType::get(Type::VoidTy, vector<const Type*>(1, VTy),
                                    /*isVarArg*/ false);
  
  SymbolTable *ST = Mod->getSymbolTableSure();
  if (Value *V = ST->lookup(PointerType::get(MTy), "printVal"))
    return V;

  // Create a new method and add it to the module
  Method *M = new Method(MTy, "printVal");
  Mod->getMethodList().push_back(M);
  return M;
}


static void InsertPrintInsts(Value *Val,
                             BasicBlock::iterator &BBI,
                             Module *Mod,
                             unsigned int indent,
                             bool isMethodExit) {
  const Type* ValTy = Val->getType();
  BasicBlock *BB = (*BBI)->getParent();
  
  assert(ValTy->isPrimitiveType() &&
         ValTy->getPrimitiveID() != Type::VoidTyID &&
         ValTy->getPrimitiveID() != Type::TypeTyID &&
         ValTy->getPrimitiveID() != Type::LabelTyID && 
         "Unsupported type for printing");
  
  const Value* scopeToUse = 
    isMethodExit ? (const Value*)BB->getParent() : (const Value*)BB;

  // Create the marker string...
  strstream scopeNameString;
  WriteAsOperand(scopeNameString, scopeToUse) << " : ";
  WriteAsOperand(scopeNameString, Val) << " = " << ends;
  string fmtString(indent, ' ');
  
  fmtString += string(" At exit of") + scopeNameString.str();

  // Turn the marker string into a global variable...
  GlobalVariable *fmtVal = GetStringRef(Mod, fmtString);
  
  // Insert the first print instruction to print the string flag:
  Instruction *I = new CallInst(GetPrintMethodForType(Mod, fmtVal->getType()),
                                vector<Value*>(1, fmtVal));
  BBI = BB->getInstList().insert(BBI, I)+1;

  // Insert the next print instruction to print the value:
  I = new CallInst(GetPrintMethodForType(Mod, ValTy),
                   vector<Value*>(1, Val));
  BBI = BB->getInstList().insert(BBI, I)+1;

  // Print out a newline
  fmtVal = GetStringRef(Mod, "\n");
  I = new CallInst(GetPrintMethodForType(Mod, fmtVal->getType()),
                   vector<Value*>(1, fmtVal));
  BBI = BB->getInstList().insert(BBI, I)+1;
}



// 
// Insert print instructions at the end of the basic block *bb
// for each value in valueVec[].  *bb must postdominate the block
// in which the value is computed; this is not checked here.
// 
static void
TraceValuesAtBBExit(const vector<Value*>& valueVec,
                    BasicBlock* bb,
                    Module* module,
                    unsigned int indent,
                    bool isMethodExit)
{
  // Get an iterator to point to the insertion location
  // 
  BasicBlock::InstListType& instList = bb->getInstList();
  TerminatorInst* termInst = bb->getTerminator(); 
  BasicBlock::iterator here = instList.end()-1;
  assert((*here)->isTerminator());
  
  // Insert a print instruction for each value.
  // 
  for (unsigned i=0, N=valueVec.size(); i < N; i++)
    InsertPrintInsts(valueVec[i], here, module, indent, isMethodExit);
}

static void
InsertCodeToShowMethodEntry(BasicBlock* entryBB)
{
}

static void
InsertCodeToShowMethodExit(BasicBlock* exitBB)
{
}


bool InsertTraceCode::doInsertTraceCode(Method *M, bool traceBasicBlockExits,
                                        bool traceMethodExits) {
  vector<Value*> valuesToTraceInMethod;
  Module* module = M->getParent();
  BasicBlock* exitBB = NULL;
  
  if (M->isExternal() ||
      (! traceBasicBlockExits && ! traceMethodExits))
    return false;

  if (traceMethodExits) {
    InsertCodeToShowMethodEntry(M->getEntryNode());
    exitBB = M->getBasicBlocks().front(); //getExitNode();
  }

  for (Method::iterator BI = M->begin(); BI != M->end(); ++BI) {
    BasicBlock* bb = *BI;

    vector<Value*> valuesToTraceInBB;
    FindValuesToTraceInBB(bb, valuesToTraceInBB);

    if (traceBasicBlockExits && bb != exitBB)
      TraceValuesAtBBExit(valuesToTraceInBB, bb, module,
                          /*indent*/ 4, /*isMethodExit*/ false);

    if (traceMethodExits) {
      valuesToTraceInMethod.insert(valuesToTraceInMethod.end(),
                                   valuesToTraceInBB.begin(),
                                   valuesToTraceInBB.end());
    }
  }

#if 0
  // Disable this code until we have a proper exit node.
  if (traceMethodExits) {
    TraceValuesAtBBExit(valuesToTraceInMethod, exitBB, module,
                        /*indent*/ 0, /*isMethodExit*/ true);
    InsertCodeToShowMethodExit(exitBB);
  }
#endif
  return true;
}
