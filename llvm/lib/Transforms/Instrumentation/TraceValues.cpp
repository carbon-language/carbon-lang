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
#include "llvm/iMemory.h"
#include "llvm/iTerminators.h"
#include "llvm/iOther.h"
#include "llvm/BasicBlock.h"
#include "llvm/Method.h"
#include "llvm/Module.h"
#include "llvm/SymbolTable.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Support/HashExtras.h"
#include <strstream>
#include <hash_map>


//*********************** Internal Data Structures *************************/

const char* const PRINTF = "printf";


//************************** Internal Functions ****************************/


static inline GlobalVariable *
GetStringRef(Module *M, const string &str)
{
  static hash_map<string, GlobalVariable*> stringRefCache;
  static Module* lastModule = NULL;
  
  if (lastModule != M)
    { // Let's make sure we create separate global references in each module
      stringRefCache.clear();
      lastModule = M;
    }
  
  GlobalVariable* result = stringRefCache[str];
  if (result == NULL)
    {
      ConstPoolArray *Init = ConstPoolArray::get(str);
      result = new GlobalVariable(Init->getType(), /*Const*/true, Init);
      M->getGlobalList().push_back(result);
      stringRefCache[str] = result;
    }
  
  return result;
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

// 
// Check if this instruction has any uses outside its basic block
// 
static inline bool
LiveAtBBExit(Instruction* I)
{
  BasicBlock* bb = I->getParent();
  bool isLive = false;
  for (Value::use_const_iterator U = I->use_begin(); U != I->use_end(); ++U)
    {
      const Instruction* userI = dyn_cast<Instruction>(*U);
      if (userI == NULL || userI->getParent() != bb)
        isLive = true;
    }
  
  return isLive;
}


static void 
FindValuesToTraceInBB(BasicBlock* bb, vector<Instruction*>& valuesToTraceInBB)
{
  for (BasicBlock::iterator II = bb->begin(); II != bb->end(); ++II)
    if ((*II)->getOpcode() == Instruction::Store
        || (LiveAtBBExit(*II) &&
            (*II)->getType()->isPrimitiveType() && 
            (*II)->getType() != Type::VoidTy &&
            TraceThisOpCode((*II)->getOpcode())))
      {
        valuesToTraceInBB.push_back(*II);
      }
}


// 
// Let's save this code for future use; it has been tested and works:
// 
// The signatures of the printf methods supported are:
//   int printf(ubyte*,  ubyte*,  ubyte*,  ubyte*,  int      intValue)
//   int printf(ubyte*,  ubyte*,  ubyte*,  ubyte*,  unsigned uintValue)
//   int printf(ubyte*,  ubyte*,  ubyte*,  ubyte*,  float    floatValue)
//   int printf(ubyte*,  ubyte*,  ubyte*,  ubyte*,  double   doubleValue)
//   int printf(ubyte*,  ubyte*,  ubyte*,  ubyte*,  char*    stringValue)
//   int printf(ubyte*,  ubyte*,  ubyte*,  ubyte*,  void*    ptrValue)
// 
// The invocation should be:
//       call "printf"(fmt, bbName, valueName, valueTypeName, value).
// 
Method*
GetPrintfMethodForType(Module* module, const Type* valueType)
{
  static const int LASTARGINDEX = 4;
  static PointerType* ubytePtrTy = NULL;
  static vector<const Type*> argTypesVec(LASTARGINDEX + 1);
  
  if (ubytePtrTy == NULL)
    { // create these once since they are invariant
      ubytePtrTy = PointerType::get(ArrayType::get(Type::UByteTy));
      argTypesVec[0] = ubytePtrTy;
      argTypesVec[1] = ubytePtrTy;
      argTypesVec[2] = ubytePtrTy;
      argTypesVec[3] = ubytePtrTy;
    }
  
  SymbolTable* symtab = module->getSymbolTable();
  argTypesVec[LASTARGINDEX] = valueType;
  MethodType* printMethodTy = MethodType::get(Type::IntTy, argTypesVec,
                                              /*isVarArg*/ false);
  
  Method* printMethod =
    cast<Method>(symtab->lookup(PointerType::get(printMethodTy), PRINTF));
  if (printMethod == NULL)
    { // Create a new method and add it to the module
      printMethod = new Method(printMethodTy, PRINTF);
      module->getMethodList().push_back(printMethod);
      
      // Create the argument list for the method so that the full signature
      // can be declared.  The args can be anonymous.
      Method::ArgumentListType &argList = printMethod->getArgumentList();
      for (unsigned i=0; i < argTypesVec.size(); ++i)
        argList.push_back(new MethodArgument(argTypesVec[i]));
    }
  
  return printMethod;
}


Instruction*
CreatePrintfInstr(Value* val,
                  const BasicBlock* bb,
                  Module* module,
                  unsigned int indent,
                  bool isMethodExit)
{
  strstream fmtString, scopeNameString, valNameString;
  vector<Value*> paramList;
  const Type* valueType = val->getType();
  Method* printMethod = GetPrintfMethodForType(module, valueType);
  
  if (! valueType->isPrimitiveType() ||
      valueType->getPrimitiveID() == Type::VoidTyID ||
      valueType->getPrimitiveID() == Type::TypeTyID ||
      valueType->getPrimitiveID() == Type::LabelTyID)
    {
      assert(0 && "Unsupported type for printing");
      return NULL;
    }
  
  const Value* scopeToUse = (isMethodExit)? (const Value*) bb->getParent()
                                          : (const Value*) bb;
  if (scopeToUse->hasName())
    scopeNameString << scopeToUse->getName() << ends;
  else
    scopeNameString << scopeToUse << ends;
  
  if (val->hasName())
    valNameString << val->getName() << ends;
  else
    valNameString << val << ends;
    
  for (unsigned i=0; i < indent; i++)
    fmtString << " ";
  
  fmtString << " At exit of "
            << ((isMethodExit)? "Method " : "BB ")
            << "%s : val %s = %s ";
  
  GlobalVariable* scopeNameVal = GetStringRef(module, scopeNameString.str());
  GlobalVariable* valNameVal   = GetStringRef(module,valNameString.str());
  GlobalVariable* typeNameVal  = GetStringRef(module,
                                     val->getType()->getDescription().c_str());
  
  switch(valueType->getPrimitiveID())
    {
    case Type::BoolTyID:
    case Type::UByteTyID: case Type::UShortTyID:
    case Type::UIntTyID:  case Type::ULongTyID:
    case Type::SByteTyID: case Type::ShortTyID:
    case Type::IntTyID:   case Type::LongTyID:
      fmtString << " %d\0A";
      break;
      
    case Type::FloatTyID:     case Type::DoubleTyID:
      fmtString << " %g\0A";
      break;
      
    case Type::PointerTyID:
      fmtString << " %p\0A";
      break;
      
    default:
      assert(0 && "Should not get here.  Check the IF expression above");
      return NULL;
    }
  
  fmtString << ends;
  GlobalVariable* fmtVal = GetStringRef(module, fmtString.str());
  
  paramList.push_back(fmtVal);
  paramList.push_back(scopeNameVal);
  paramList.push_back(valNameVal);
  paramList.push_back(typeNameVal);
  paramList.push_back(val);
  
  free(fmtString.str());
  free(scopeNameString.str());
  free(valNameString.str());
  
  return new CallInst(printMethod, paramList);
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


static void
InsertPrintInsts(Value *Val,
                 BasicBlock* BB,
                 BasicBlock::iterator &BBI,
                 Module *Mod,
                 unsigned int indent,
                 bool isMethodExit)
{
  const Type* ValTy = Val->getType();
  
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
  free(scopeNameString.str());
  
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


static LoadInst*
InsertLoadInst(StoreInst* storeInst,
               BasicBlock *bb,
               BasicBlock::iterator &BBI)
{
  LoadInst* loadInst = new LoadInst(storeInst->getPtrOperand(),
                                    storeInst->getIndexVec());
  BBI = bb->getInstList().insert(BBI, loadInst) + 1;
  return loadInst;
}


// 
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
static void
TraceValuesAtBBExit(const vector<Instruction*>& valueVec,
                    BasicBlock* bb,
                    Module* module,
                    unsigned int indent,
                    bool isMethodExit,
                    vector<Instruction*>* valuesStoredInMethod)
{
  // Get an iterator to point to the insertion location
  // 
  BasicBlock::InstListType& instList = bb->getInstList();
  BasicBlock::iterator here = instList.end()-1;
  assert((*here)->isTerminator());
  
  // Insert a print instruction for each value.
  // 
  for (unsigned i=0, N=valueVec.size(); i < N; i++)
    {
      Instruction* I = valueVec[i];
      if (I->getOpcode() == Instruction::Store)
        {
          assert(valuesStoredInMethod != NULL &&
                 "Should not be printing a store instruction at method exit");
          I = InsertLoadInst((StoreInst*) I, bb, here);
          valuesStoredInMethod->push_back(I);
        }
      InsertPrintInsts(I, bb, here, module, indent, isMethodExit);
    }
}



static Instruction*
CreateMethodTraceInst(
                      Method* method,
                      unsigned int indent,
                      const string& msg)
{
  string fmtString(indent, ' ');
  strstream methodNameString;
  WriteAsOperand(methodNameString, method) << ends;
  fmtString += msg + string(" METHOD ") + methodNameString.str();
  free(methodNameString.str());
  
  GlobalVariable *fmtVal = GetStringRef(method->getParent(), fmtString);
  Instruction *printInst =
    new CallInst(GetPrintMethodForType(method->getParent(), fmtVal->getType()),
                 vector<Value*>(1, fmtVal));

  return printInst;
}


static inline void
InsertCodeToShowMethodEntry(Method* method,
                            BasicBlock* entryBB,
                            unsigned int indent)
{
  // Get an iterator to point to the insertion location
  BasicBlock::InstListType& instList = entryBB->getInstList();
  BasicBlock::iterator here = instList.begin();
  
  Instruction *printInst = CreateMethodTraceInst(method, indent, "ENTERING"); 
  
  here = entryBB->getInstList().insert(here, printInst) + 1;
}


static inline void
InsertCodeToShowMethodExit(Method* method,
                           BasicBlock* exitBB,
                           unsigned int indent)
{
  // Get an iterator to point to the insertion location
  BasicBlock::InstListType& instList = exitBB->getInstList();
  BasicBlock::iterator here = instList.end()-1;
  assert((*here)->isTerminator());
  
  Instruction *printInst = CreateMethodTraceInst(method, indent, "LEAVING "); 
  
  here = exitBB->getInstList().insert(here, printInst) + 1;
}


//************************** External Functions ****************************/


bool
InsertTraceCode::doInsertTraceCode(Method *M,
                                   bool traceBasicBlockExits,
                                   bool traceMethodExits)
{
  vector<Instruction*> valuesStoredInMethod;
  Module* module = M->getParent();
  vector<BasicBlock*> exitBlocks;

  if (M->isExternal() ||
      (! traceBasicBlockExits && ! traceMethodExits))
    return false;

  if (traceMethodExits)
    InsertCodeToShowMethodEntry(M, M->getEntryNode(), /*indent*/ 0);
  
  for (Method::iterator BI = M->begin(); BI != M->end(); ++BI)
    {
      BasicBlock* bb = *BI;
      bool isExitBlock = false;
      vector<Instruction*> valuesToTraceInBB;
      
      FindValuesToTraceInBB(bb, valuesToTraceInBB);
      
      if (bb->succ_begin() == bb->succ_end())
        { // record this as an exit block
          exitBlocks.push_back(bb);
          isExitBlock = true;
        }
      
      if (traceBasicBlockExits)
        TraceValuesAtBBExit(valuesToTraceInBB, bb, module,
                            /*indent*/ 4, /*isMethodExit*/ false,
                            &valuesStoredInMethod);
    }

  if (traceMethodExits)
    for (unsigned i=0; i < exitBlocks.size(); ++i)
      {
        TraceValuesAtBBExit(valuesStoredInMethod, exitBlocks[i], module,
                            /*indent*/ 0, /*isMethodExit*/ true,
                            /*valuesStoredInMethod*/ NULL);
        InsertCodeToShowMethodExit(M, exitBlocks[i], /*indent*/ 0);
      }

  return true;
}
