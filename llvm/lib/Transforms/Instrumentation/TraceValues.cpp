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
#include "llvm/Support/HashExtras.h"
#include <hash_map>
#include <strstream.h>


//*********************** Internal Data Structures *************************/

const char* const PRINTF = "printf";

#undef DONT_EMBED_STRINGS_IN_FMT


//************************** Internal Functions ****************************/

#undef USE_PTRREF
#ifdef USE_PTRREF
static inline ConstPoolPointerRef*
GetStringRef(Module* module, const char* str)
{
  static hash_map<string, ConstPoolPointerRef*> stringRefCache;
  static Module* lastModule = NULL;
  
  if (lastModule != module)
    { // Let's make sure we create separate global references in each module
      stringRefCache.clear();
      lastModule = module;
    }
  
  ConstPoolPointerRef* result = stringRefCache[str];
  if (result == NULL)
    {
      ConstPoolArray* charArray = ConstPoolArray::get(str);
      GlobalVariable* stringVar =
        new GlobalVariable(charArray->getType(),/*isConst*/true,charArray,str);
      module->getGlobalList().push_back(stringVar);
      result = ConstPoolPointerRef::get(stringVar);
      assert(result && "Failed to create reference to string constant");
      stringRefCache[str] = result;
    }
  
  return result;
}
#endif USE_PTRREF

static inline GlobalVariable*
GetStringRef(Module* module, const char* str)
{
  static hash_map<string, GlobalVariable*> stringRefCache;
  static Module* lastModule = NULL;
  
  if (lastModule != module)
    { // Let's make sure we create separate global references in each module
      stringRefCache.clear();
      lastModule = module;
    }
  
  GlobalVariable* result = stringRefCache[str];
  if (result == NULL)
    {
      ConstPoolArray* charArray = ConstPoolArray::get(str);
      GlobalVariable* stringVar =
        new GlobalVariable(charArray->getType(),/*isConst*/true,charArray);
      module->getGlobalList().push_back(stringVar);
      result = stringVar;
      // result = ConstPoolPointerRef::get(stringVar);
      assert(result && "Failed to create reference to string constant");
      stringRefCache[str] = result;
    }
  
  return result;
}


static inline bool
TraceThisOpCode(unsigned opCode)
{
  // Explicitly test for opCodes *not* to trace so that any new opcodes will
  // be traced by default (or will fail in a later assertion on VoidTy)
  // 
  return (opCode  < Instruction::FirstOtherOp &&
          opCode != Instruction::Ret &&
          opCode != Instruction::Br &&
          opCode != Instruction::Switch &&
          opCode != Instruction::Free &&
          opCode != Instruction::Alloca &&
          opCode != Instruction::Store &&
          opCode != Instruction::PHINode &&
          opCode != Instruction::Cast);
}


static void
FindValuesToTraceInBB(BasicBlock* bb,
                      vector<Value*>& valuesToTraceInBB)
{
  for (BasicBlock::iterator II = bb->begin(); II != bb->end(); ++II)
    if ((*II)->getType()->isPrimitiveType() &&
        TraceThisOpCode((*II)->getOpcode()))
      {
        valuesToTraceInBB.push_back(*II);
      }
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
  BasicBlock::InstListType::iterator here = instList.end();
  while ((*here) != termInst && here != instList.begin())
    --here;
  assert((*here) == termInst);
  
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


//************************** External Functions ****************************/

// 
// The signatures of the print methods supported are:
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
GetPrintMethodForType(Module* module, const Type* valueType)
{
#ifdef DONT_EMBED_STRINGS_IN_FMT
  static const int LASTARGINDEX = 4;
#else
  static const int LASTARGINDEX = 1;
#endif
  static PointerType* ubytePtrTy = NULL;
  static vector<const Type*> argTypesVec(LASTARGINDEX + 1);
  
  if (ubytePtrTy == NULL)
    { // create these once since they are invariant
      ubytePtrTy = PointerType::get(ArrayType::get(Type::UByteTy));
      argTypesVec[0] = ubytePtrTy;
#ifdef DONT_EMBED_STRINGS_IN_FMT
      argTypesVec[1] = ubytePtrTy;
      argTypesVec[2] = ubytePtrTy;
      argTypesVec[3] = ubytePtrTy;
#endif DONT_EMBED_STRINGS_IN_FMT
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
CreatePrintInstr(Value* val,
                 const BasicBlock* bb,
                 Module* module,
                 unsigned int indent,
                 bool isMethodExit)
{
  strstream fmtString, scopeNameString, valNameString;
  vector<Value*> paramList;
  const Type* valueType = val->getType();
  Method* printMethod = GetPrintMethodForType(module, valueType);
  
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
  
#undef DONT_EMBED_STRINGS_IN_FMT
#ifdef DONT_EMBED_STRINGS_IN_FMT
  fmtString << " At exit of "
            << ((isMethodExit)? "Method " : "BB ")
            << "%s : val %s = %s ";
  
  GlobalVariable* scopeNameVal = GetStringRef(module, scopeNameString.str());
  GlobalVariable* valNameVal   = GetStringRef(module,valNameString.str());
  GlobalVariable* typeNameVal  = GetStringRef(module,
                                     val->getType()->getDescription().c_str());
#else
  fmtString << " At exit of "
            << ((isMethodExit)? "Method " : "BB ")
            << scopeNameString.str() << " : "
            << valNameString.str()   << " = "
            << val->getType()->getDescription().c_str();
#endif DONT_EMBED_STRINGS_IN_FMT
  
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
  
#ifdef DONT_EMBED_STRINGS_IN_FMT
  paramList.push_back(fmtVal);
  paramList.push_back(scopeNameVal);
  paramList.push_back(valNameVal);
  paramList.push_back(typeNameVal);
  paramList.push_back(val);
#else
  paramList.push_back(fmtVal);
  paramList.push_back(val);
#endif DONT_EMBED_STRINGS_IN_FMT
  
  free(fmtString.str());
  free(scopeNameString.str());
  free(valNameString.str());
  
  return new CallInst(printMethod, paramList);
}


void
InsertCodeToTraceValues(Method* method,
                        bool traceBasicBlockExits,
                        bool traceMethodExits)
{
  vector<Value*> valuesToTraceInMethod;
  Module* module = method->getParent();
  BasicBlock* exitBB = NULL;
  
  if (method->isExternal() ||
      (! traceBasicBlockExits && ! traceMethodExits))
    return;
  
  if (traceMethodExits)
    {
      InsertCodeToShowMethodEntry(method->getEntryNode());
#ifdef TODO_LATER
      exitBB = method->getExitNode();
#endif
    }
  
  for (Method::iterator BI = method->begin(); BI != method->end(); ++BI)
    {
      BasicBlock* bb = *BI;
      vector<Value*> valuesToTraceInBB;
      FindValuesToTraceInBB(bb, valuesToTraceInBB);
      
      if (traceBasicBlockExits && bb != exitBB)
        TraceValuesAtBBExit(valuesToTraceInBB, bb, module,
                            /*indent*/ 4, /*isMethodExit*/ false);
      
      if (traceMethodExits)
        valuesToTraceInMethod.insert(valuesToTraceInMethod.end(),
                                     valuesToTraceInBB.begin(),
                                     valuesToTraceInBB.end());
    }
  
  if (traceMethodExits)
    {
      TraceValuesAtBBExit(valuesToTraceInMethod, exitBB, module,
                          /*indent*/ 0, /*isMethodExit*/ true);
      InsertCodeToShowMethodExit(exitBB);
    }
}
