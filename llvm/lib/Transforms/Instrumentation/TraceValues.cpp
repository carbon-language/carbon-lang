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
#include <hash_set>
#include <sstream>


static const char*
PrintMethodNameForType(const Type* type)
{
  if (PointerType* pty = dyn_cast<PointerType>(type))
    {
      const Type* elemTy;
      if (ArrayType* aty = dyn_cast<ArrayType>(pty->getValueType()))
        elemTy = aty->getElementType();
      else
        elemTy = pty->getValueType();
      if (elemTy == Type::SByteTy || elemTy == Type::UByteTy)
        return "printString";
    }

  switch (type->getPrimitiveID())
    {
    case Type::BoolTyID:    return "printBool";
    case Type::UByteTyID:   return "printUByte";
    case Type::SByteTyID:   return "printSByte";
    case Type::UShortTyID:  return "printUShort";
    case Type::ShortTyID:   return "printShort";
    case Type::UIntTyID:    return "printUInt";
    case Type::IntTyID:     return "printInt";
    case Type::ULongTyID:   return "printULong";
    case Type::LongTyID:    return "printLong";
    case Type::FloatTyID:   return "printFloat";
    case Type::DoubleTyID:  return "printDouble";
    case Type::PointerTyID: return "printPointer";
    default:
      assert(0 && "Unsupported type for printing");
      return NULL;
    }
}

static inline GlobalVariable *GetStringRef(Module *M, const string &str) {
  ConstPoolArray *Init = ConstPoolArray::get(str);
  GlobalVariable *GV = new GlobalVariable(Init->getType(), /*Const*/true, Init);
  M->getGlobalList().push_back(GV);
  return GV;
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
// Check if this instruction has any uses outside its basic block,
// or if it used by either a Call or Return instruction.
// 
static inline bool
LiveAtBBExit(Instruction* I)
{
  BasicBlock* bb = I->getParent();
  bool isLive = false;
  for (Value::use_const_iterator U = I->use_begin(); U != I->use_end(); ++U)
    {
      const Instruction* userI = dyn_cast<Instruction>(*U);
      if (userI == NULL
          || userI->getParent() != bb
          || userI->getOpcode() == Instruction::Call
          || userI->getOpcode() == Instruction::Ret)
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
Value *GetPrintfMethodForType(Module* module, const Type* valueType)
{
  PointerType *ubytePtrTy = PointerType::get(ArrayType::get(Type::UByteTy));
  vector<const Type*> argTypesVec(4, ubytePtrTy);
  argTypesVec.push_back(valueType);
    
  MethodType *printMethodTy = MethodType::get(Type::IntTy, argTypesVec,
                                              /*isVarArg*/ false);
  
  SymbolTable *ST = module->getSymbolTable();
  if (Value *Meth = ST->lookup(PointerType::get(printMethodTy), "printf"))
    return Meth;

  // Create a new method and add it to the module
  Method *printMethod = new Method(printMethodTy, "printf");
  module->getMethodList().push_back(printMethod);
  
  return printMethod;
}


Instruction*
CreatePrintfInstr(Value* val,
                  const BasicBlock* bb,
                  Module* module,
                  unsigned int indent,
                  bool isMethodExit)
{
  ostringstream fmtString, scopeNameString, valNameString;
  vector<Value*> paramList;
  const Type* valueType = val->getType();
  Method* printMethod = cast<Method>(GetPrintfMethodForType(module,valueType));
  
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
      fmtString << " %d\n";
      break;
      
    case Type::FloatTyID:     case Type::DoubleTyID:
      fmtString << " %g\n";
      break;
      
    case Type::PointerTyID:
      fmtString << " %p\n";
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
  
  return new CallInst(printMethod, paramList);
}


// The invocation should be:
//       call "printString"([ubyte*] or [sbyte*] or ubyte* or sbyte*).
//       call "printLong"(long)
//       call "printInt"(int) ...
// 
static Value *GetPrintMethodForType(Module *Mod, const Type *VTy) {
  MethodType *MTy = MethodType::get(Type::VoidTy, vector<const Type*>(1, VTy),
                                    /*isVarArg*/ false);
  
  const char* printMethodName = PrintMethodNameForType(VTy);
  SymbolTable *ST = Mod->getSymbolTableSure();
  if (Value *V = ST->lookup(PointerType::get(MTy), printMethodName))
    return V;

  // Create a new method and add it to the module
  Method *M = new Method(MTy, printMethodName);
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
  
  assert((ValTy->isPrimitiveType() || isa<PointerType>(ValTy)) &&
         ValTy != Type::VoidTy && ValTy != Type::TypeTy &&
         ValTy != Type::LabelTy && "Unsupported type for printing");
  
  const Value* scopeToUse = 
    isMethodExit ? (const Value*)BB->getParent() : (const Value*)BB;

  // Create the marker string...
  ostringstream scopeNameString;
  WriteAsOperand(scopeNameString, scopeToUse) << " : ";
  WriteAsOperand(scopeNameString, Val) << " = ";
  
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


static LoadInst*
InsertLoadInst(StoreInst* storeInst,
               BasicBlock *bb,
               BasicBlock::iterator &BBI)
{
  LoadInst* loadInst = new LoadInst(storeInst->getPtrOperand(),
                                    storeInst->getIndices());
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
  // Get an iterator to point to the insertion location, which is
  // just before the terminator instruction.
  // 
  BasicBlock::InstListType& instList = bb->getInstList();
  BasicBlock::iterator here = instList.end()-1;
  assert((*here)->isTerminator());
  
  // If the terminator is a conditional branch, insert the trace code just
  // before the instruction that computes the branch condition (just to
  // avoid putting a call between the CC-setting instruction and the branch).
  // Use laterInstrSet to mark instructions that come after the setCC instr
  // because those cannot be traced at the location we choose.
  // 
  hash_set<Instruction*> laterInstrSet;
  if (BranchInst* brInst = dyn_cast<BranchInst>(*here))
    if (! brInst->isUnconditional())
      if (Instruction* setCC = dyn_cast<Instruction>(brInst->getCondition()))
        if (setCC->getParent() == bb)
          {
            while ((*here) != setCC && here != instList.begin())
              {
                --here;
                laterInstrSet.insert(*here);
              }
            assert((*here) == setCC && "Missed the setCC instruction?");
            laterInstrSet.insert(*here);
          }
  
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
      if (laterInstrSet.find(I) == laterInstrSet.end())
        InsertPrintInsts(I, bb, here, module, indent, isMethodExit);
    }
}



static Instruction*
CreateMethodTraceInst(Method* method,
                      unsigned int indent,
                      const string& msg)
{
  string fmtString(indent, ' ');
  ostringstream methodNameString;
  WriteAsOperand(methodNameString, method);
  fmtString += msg + methodNameString.str() + '\n';
  
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
  
  Instruction *printInst = CreateMethodTraceInst(method, indent, 
                                                 "Entering Method"); 
  
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
  
  Instruction *printInst = CreateMethodTraceInst(method, indent,
                                                 "Leaving Method"); 
  
  exitBB->getInstList().insert(here, printInst) + 1;
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
