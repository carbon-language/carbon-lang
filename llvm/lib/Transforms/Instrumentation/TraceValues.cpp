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


const char* const PRINTF = "printVal";

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
//       call "printf"(fmt, value).
// 
static Value *GetPrintMethodForType(Module *Mod, const Type *valueType) {
  vector<const Type*> ArgTys;
  ArgTys.reserve(2);
  ArgTys.push_back(PointerType::get(ArrayType::get(Type::UByteTy)));
  ArgTys.push_back(valueType);
  
  MethodType *printMethodTy = MethodType::get(Type::VoidTy, ArgTys,
                                              /*isVarArg*/ false);
  
  SymbolTable *ST = Mod->getSymbolTableSure();
  if (Value *V = ST->lookup(PointerType::get(printMethodTy), PRINTF))
    return V;

  // Create a new method and add it to the module
  Method *M = new Method(printMethodTy, PRINTF);
  Mod->getMethodList().push_back(M);
  return M;
}


static Instruction*
CreatePrintInstr(Value* val,
                 const BasicBlock* bb,
                 Module* module,
                 unsigned int indent,
                 bool isMethodExit)
{
  strstream scopeNameString;
  const Type* valueType = val->getType();
  
  assert(valueType->isPrimitiveType() &&
         valueType->getPrimitiveID() != Type::VoidTyID &&
         valueType->getPrimitiveID() != Type::TypeTyID &&
         valueType->getPrimitiveID() != Type::LabelTyID && 
         "Unsupported type for printing");
  
  const Value* scopeToUse = (isMethodExit)? (const Value*) bb->getParent()
                                          : (const Value*) bb;
  WriteAsOperand(scopeNameString, scopeToUse) << " : ";
  WriteAsOperand(scopeNameString, val) << " = "
                                       << val->getType()->getDescription()
                                       << ends;
  string fmtString(indent, ' ');
  
  fmtString += " At exit of " + string(isMethodExit ? "Method " : "BB ") +
    scopeNameString.str();
  
  switch(valueType->getPrimitiveID()) {
  case Type::BoolTyID:
  case Type::UByteTyID: case Type::UShortTyID:
  case Type::UIntTyID:  case Type::ULongTyID:
  case Type::SByteTyID: case Type::ShortTyID:
  case Type::IntTyID:   case Type::LongTyID:
    fmtString += " %d\0A";
    break;
    
  case Type::FloatTyID:     case Type::DoubleTyID:
    fmtString += " %g\0A";
    break;
    
  case Type::PointerTyID:
    fmtString += " %p\0A";
    break;
    
  default:
    assert(0 && "Should not get here.  Check the IF expression above");
    return NULL;
  }
  
  GlobalVariable *fmtVal = GetStringRef(module, fmtString);
  
  vector<Value*> paramList;
  paramList.push_back(fmtVal);
  paramList.push_back(val);

  return new CallInst(GetPrintMethodForType(module, valueType), paramList);
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
  BasicBlock::InstListType::iterator here = instList.end()-1;
  assert((*here)->isTerminator());
  
  // Insert a print instruction for each value.
  // 
  for (unsigned i=0, N=valueVec.size(); i < N; i++)
    {
      Instruction* traceInstr =
        CreatePrintInstr(valueVec[i], bb, module, indent, isMethodExit);
      here = instList.insert(here, traceInstr);
    }
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
  
  if (traceMethodExits) {
    TraceValuesAtBBExit(valuesToTraceInMethod, exitBB, module,
                        /*indent*/ 0, /*isMethodExit*/ true);
    InsertCodeToShowMethodExit(exitBB);
  }
  return true;
}
