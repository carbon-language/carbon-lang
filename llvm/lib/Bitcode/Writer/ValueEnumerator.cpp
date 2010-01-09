//===-- ValueEnumerator.cpp - Number values and types for bitcode writer --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the ValueEnumerator class.
//
//===----------------------------------------------------------------------===//

#include "ValueEnumerator.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/TypeSymbolTable.h"
#include "llvm/ValueSymbolTable.h"
#include "llvm/Instructions.h"
#include <algorithm>
using namespace llvm;

static bool isSingleValueType(const std::pair<const llvm::Type*,
                              unsigned int> &P) {
  return P.first->isSingleValueType();
}

static bool isIntegerValue(const std::pair<const Value*, unsigned> &V) {
  return isa<IntegerType>(V.first->getType());
}

static bool CompareByFrequency(const std::pair<const llvm::Type*,
                               unsigned int> &P1,
                               const std::pair<const llvm::Type*,
                               unsigned int> &P2) {
  return P1.second > P2.second;
}

/// ValueEnumerator - Enumerate module-level information.
ValueEnumerator::ValueEnumerator(const Module *M) {
  InstructionCount = 0;

  // Enumerate the global variables.
  for (Module::const_global_iterator I = M->global_begin(),
         E = M->global_end(); I != E; ++I)
    EnumerateValue(I);

  // Enumerate the functions.
  for (Module::const_iterator I = M->begin(), E = M->end(); I != E; ++I) {
    EnumerateValue(I);
    EnumerateAttributes(cast<Function>(I)->getAttributes());
  }

  // Enumerate the aliases.
  for (Module::const_alias_iterator I = M->alias_begin(), E = M->alias_end();
       I != E; ++I)
    EnumerateValue(I);

  // Remember what is the cutoff between globalvalue's and other constants.
  unsigned FirstConstant = Values.size();

  // Enumerate the global variable initializers.
  for (Module::const_global_iterator I = M->global_begin(),
         E = M->global_end(); I != E; ++I)
    if (I->hasInitializer())
      EnumerateValue(I->getInitializer());

  // Enumerate the aliasees.
  for (Module::const_alias_iterator I = M->alias_begin(), E = M->alias_end();
       I != E; ++I)
    EnumerateValue(I->getAliasee());

  // Enumerate types used by the type symbol table.
  EnumerateTypeSymbolTable(M->getTypeSymbolTable());

  // Insert constants and metadata  that are named at module level into the slot 
  // pool so that the module symbol table can refer to them...
  EnumerateValueSymbolTable(M->getValueSymbolTable());
  EnumerateMDSymbolTable(M->getMDSymbolTable());

  SmallVector<std::pair<unsigned, MDNode*>, 8> MDs;

  // Enumerate types used by function bodies and argument lists.
  for (Module::const_iterator F = M->begin(), E = M->end(); F != E; ++F) {

    for (Function::const_arg_iterator I = F->arg_begin(), E = F->arg_end();
         I != E; ++I)
      EnumerateType(I->getType());

    for (Function::const_iterator BB = F->begin(), E = F->end(); BB != E; ++BB)
      for (BasicBlock::const_iterator I = BB->begin(), E = BB->end(); I!=E;++I){
        for (User::const_op_iterator OI = I->op_begin(), E = I->op_end();
             OI != E; ++OI)
          EnumerateOperandType(*OI);
        EnumerateType(I->getType());
        if (const CallInst *CI = dyn_cast<CallInst>(I))
          EnumerateAttributes(CI->getAttributes());
        else if (const InvokeInst *II = dyn_cast<InvokeInst>(I))
          EnumerateAttributes(II->getAttributes());

        // Enumerate metadata attached with this instruction.
        MDs.clear();
        I->getAllMetadata(MDs);
        for (unsigned i = 0, e = MDs.size(); i != e; ++i)
          EnumerateMetadata(MDs[i].second);
      }
  }

  // Optimize constant ordering.
  OptimizeConstants(FirstConstant, Values.size());

  // Sort the type table by frequency so that most commonly used types are early
  // in the table (have low bit-width).
  std::stable_sort(Types.begin(), Types.end(), CompareByFrequency);

  // Partition the Type ID's so that the single-value types occur before the
  // aggregate types.  This allows the aggregate types to be dropped from the
  // type table after parsing the global variable initializers.
  std::partition(Types.begin(), Types.end(), isSingleValueType);

  // Now that we rearranged the type table, rebuild TypeMap.
  for (unsigned i = 0, e = Types.size(); i != e; ++i)
    TypeMap[Types[i].first] = i+1;
}

unsigned ValueEnumerator::getInstructionID(const Instruction *Inst) const {
  InstructionMapType::const_iterator I = InstructionMap.find(Inst);
  assert (I != InstructionMap.end() && "Instruction is not mapped!");
    return I->second;
}

void ValueEnumerator::setInstructionID(const Instruction *I) {
  InstructionMap[I] = InstructionCount++;
}

unsigned ValueEnumerator::getValueID(const Value *V) const {
  if (isa<MetadataBase>(V) || isa<NamedMDNode>(V)) {
    ValueMapType::const_iterator I = MDValueMap.find(V);
    assert(I != MDValueMap.end() && "Value not in slotcalculator!");
    return I->second-1;
  }

  ValueMapType::const_iterator I = ValueMap.find(V);
  assert(I != ValueMap.end() && "Value not in slotcalculator!");
  return I->second-1;
}

// Optimize constant ordering.
namespace {
  struct CstSortPredicate {
    ValueEnumerator &VE;
    explicit CstSortPredicate(ValueEnumerator &ve) : VE(ve) {}
    bool operator()(const std::pair<const Value*, unsigned> &LHS,
                    const std::pair<const Value*, unsigned> &RHS) {
      // Sort by plane.
      if (LHS.first->getType() != RHS.first->getType())
        return VE.getTypeID(LHS.first->getType()) <
               VE.getTypeID(RHS.first->getType());
      // Then by frequency.
      return LHS.second > RHS.second;
    }
  };
}

/// OptimizeConstants - Reorder constant pool for denser encoding.
void ValueEnumerator::OptimizeConstants(unsigned CstStart, unsigned CstEnd) {
  if (CstStart == CstEnd || CstStart+1 == CstEnd) return;

  CstSortPredicate P(*this);
  std::stable_sort(Values.begin()+CstStart, Values.begin()+CstEnd, P);

  // Ensure that integer constants are at the start of the constant pool.  This
  // is important so that GEP structure indices come before gep constant exprs.
  std::partition(Values.begin()+CstStart, Values.begin()+CstEnd,
                 isIntegerValue);

  // Rebuild the modified portion of ValueMap.
  for (; CstStart != CstEnd; ++CstStart)
    ValueMap[Values[CstStart].first] = CstStart+1;
}


/// EnumerateTypeSymbolTable - Insert all of the types in the specified symbol
/// table.
void ValueEnumerator::EnumerateTypeSymbolTable(const TypeSymbolTable &TST) {
  for (TypeSymbolTable::const_iterator TI = TST.begin(), TE = TST.end();
       TI != TE; ++TI)
    EnumerateType(TI->second);
}

/// EnumerateValueSymbolTable - Insert all of the values in the specified symbol
/// table into the values table.
void ValueEnumerator::EnumerateValueSymbolTable(const ValueSymbolTable &VST) {
  for (ValueSymbolTable::const_iterator VI = VST.begin(), VE = VST.end();
       VI != VE; ++VI)
    EnumerateValue(VI->getValue());
}

/// EnumerateMDSymbolTable - Insert all of the values in the specified metadata
/// table.
void ValueEnumerator::EnumerateMDSymbolTable(const MDSymbolTable &MST) {
  for (MDSymbolTable::const_iterator MI = MST.begin(), ME = MST.end();
       MI != ME; ++MI)
    EnumerateValue(MI->getValue());
}

void ValueEnumerator::EnumerateNamedMDNode(const NamedMDNode *MD) {
  // Check to see if it's already in!
  unsigned &MDValueID = MDValueMap[MD];
  if (MDValueID) {
    // Increment use count.
    MDValues[MDValueID-1].second++;
    return;
  }

  // Enumerate the type of this value.
  EnumerateType(MD->getType());

  for (unsigned i = 0, e = MD->getNumOperands(); i != e; ++i)
    if (MDNode *E = MD->getOperand(i))
      EnumerateValue(E);
  MDValues.push_back(std::make_pair(MD, 1U));
  MDValueMap[MD] = Values.size();
}

void ValueEnumerator::EnumerateMetadata(const MetadataBase *MD) {
  // Check to see if it's already in!
  unsigned &MDValueID = MDValueMap[MD];
  if (MDValueID) {
    // Increment use count.
    MDValues[MDValueID-1].second++;
    return;
  }

  // Enumerate the type of this value.
  EnumerateType(MD->getType());

  if (const MDNode *N = dyn_cast<MDNode>(MD)) {
    MDValues.push_back(std::make_pair(MD, 1U));
    MDValueMap[MD] = MDValues.size();
    MDValueID = MDValues.size();
    for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i) {    
      if (Value *V = N->getOperand(i))
        EnumerateValue(V);
      else
        EnumerateType(Type::getVoidTy(MD->getContext()));
    }
    return;
  }
  
  // Add the value.
  assert(isa<MDString>(MD) && "Unknown metadata kind");
  MDValues.push_back(std::make_pair(MD, 1U));
  MDValueID = MDValues.size();
}

void ValueEnumerator::EnumerateValue(const Value *V) {
  assert(!V->getType()->isVoidTy() && "Can't insert void values!");
  if (const MetadataBase *MB = dyn_cast<MetadataBase>(V))
    return EnumerateMetadata(MB);
  else if (const NamedMDNode *NMD = dyn_cast<NamedMDNode>(V))
    return EnumerateNamedMDNode(NMD);

  // Check to see if it's already in!
  unsigned &ValueID = ValueMap[V];
  if (ValueID) {
    // Increment use count.
    Values[ValueID-1].second++;
    return;
  }

  // Enumerate the type of this value.
  EnumerateType(V->getType());

  if (const Constant *C = dyn_cast<Constant>(V)) {
    if (isa<GlobalValue>(C)) {
      // Initializers for globals are handled explicitly elsewhere.
    } else if (isa<ConstantArray>(C) && cast<ConstantArray>(C)->isString()) {
      // Do not enumerate the initializers for an array of simple characters.
      // The initializers just polute the value table, and we emit the strings
      // specially.
    } else if (C->getNumOperands()) {
      // If a constant has operands, enumerate them.  This makes sure that if a
      // constant has uses (for example an array of const ints), that they are
      // inserted also.

      // We prefer to enumerate them with values before we enumerate the user
      // itself.  This makes it more likely that we can avoid forward references
      // in the reader.  We know that there can be no cycles in the constants
      // graph that don't go through a global variable.
      for (User::const_op_iterator I = C->op_begin(), E = C->op_end();
           I != E; ++I)
        if (!isa<BasicBlock>(*I)) // Don't enumerate BB operand to BlockAddress.
          EnumerateValue(*I);

      // Finally, add the value.  Doing this could make the ValueID reference be
      // dangling, don't reuse it.
      Values.push_back(std::make_pair(V, 1U));
      ValueMap[V] = Values.size();
      return;
    }
  }

  // Add the value.
  Values.push_back(std::make_pair(V, 1U));
  ValueID = Values.size();
}


void ValueEnumerator::EnumerateType(const Type *Ty) {
  unsigned &TypeID = TypeMap[Ty];

  if (TypeID) {
    // If we've already seen this type, just increase its occurrence count.
    Types[TypeID-1].second++;
    return;
  }

  // First time we saw this type, add it.
  Types.push_back(std::make_pair(Ty, 1U));
  TypeID = Types.size();

  // Enumerate subtypes.
  for (Type::subtype_iterator I = Ty->subtype_begin(), E = Ty->subtype_end();
       I != E; ++I)
    EnumerateType(*I);
}

// Enumerate the types for the specified value.  If the value is a constant,
// walk through it, enumerating the types of the constant.
void ValueEnumerator::EnumerateOperandType(const Value *V) {
  EnumerateType(V->getType());
  if (const Constant *C = dyn_cast<Constant>(V)) {
    // If this constant is already enumerated, ignore it, we know its type must
    // be enumerated.
    if (ValueMap.count(V)) return;

    // This constant may have operands, make sure to enumerate the types in
    // them.
    for (unsigned i = 0, e = C->getNumOperands(); i != e; ++i) {
      const User *Op = C->getOperand(i);
      
      // Don't enumerate basic blocks here, this happens as operands to
      // blockaddress.
      if (isa<BasicBlock>(Op)) continue;
      
      EnumerateOperandType(cast<Constant>(Op));
    }

    if (const MDNode *N = dyn_cast<MDNode>(V)) {
      for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i)
        if (Value *Elem = N->getOperand(i))
          EnumerateOperandType(Elem);
    }
  } else if (isa<MDString>(V) || isa<MDNode>(V))
    EnumerateValue(V);
}

void ValueEnumerator::EnumerateAttributes(const AttrListPtr &PAL) {
  if (PAL.isEmpty()) return;  // null is always 0.
  // Do a lookup.
  unsigned &Entry = AttributeMap[PAL.getRawPointer()];
  if (Entry == 0) {
    // Never saw this before, add it.
    Attributes.push_back(PAL);
    Entry = Attributes.size();
  }
}


void ValueEnumerator::incorporateFunction(const Function &F) {
  NumModuleValues = Values.size();

  // Adding function arguments to the value table.
  for(Function::const_arg_iterator I = F.arg_begin(), E = F.arg_end();
      I != E; ++I)
    EnumerateValue(I);

  FirstFuncConstantID = Values.size();

  // Add all function-level constants to the value table.
  for (Function::const_iterator BB = F.begin(), E = F.end(); BB != E; ++BB) {
    for (BasicBlock::const_iterator I = BB->begin(), E = BB->end(); I!=E; ++I)
      for (User::const_op_iterator OI = I->op_begin(), E = I->op_end();
           OI != E; ++OI) {
        if ((isa<Constant>(*OI) && !isa<GlobalValue>(*OI)) ||
            isa<InlineAsm>(*OI))
          EnumerateValue(*OI);
      }
    BasicBlocks.push_back(BB);
    ValueMap[BB] = BasicBlocks.size();
  }

  // Optimize the constant layout.
  OptimizeConstants(FirstFuncConstantID, Values.size());

  // Add the function's parameter attributes so they are available for use in
  // the function's instruction.
  EnumerateAttributes(F.getAttributes());

  FirstInstID = Values.size();

  // Add all of the instructions.
  for (Function::const_iterator BB = F.begin(), E = F.end(); BB != E; ++BB) {
    for (BasicBlock::const_iterator I = BB->begin(), E = BB->end(); I!=E; ++I) {
      if (!I->getType()->isVoidTy())
        EnumerateValue(I);
    }
  }
}

void ValueEnumerator::purgeFunction() {
  /// Remove purged values from the ValueMap.
  for (unsigned i = NumModuleValues, e = Values.size(); i != e; ++i)
    ValueMap.erase(Values[i].first);
  for (unsigned i = 0, e = BasicBlocks.size(); i != e; ++i)
    ValueMap.erase(BasicBlocks[i]);

  Values.resize(NumModuleValues);
  BasicBlocks.clear();
}

static void IncorporateFunctionInfoGlobalBBIDs(const Function *F,
                                 DenseMap<const BasicBlock*, unsigned> &IDMap) {
  unsigned Counter = 0;
  for (Function::const_iterator BB = F->begin(), E = F->end(); BB != E; ++BB)
    IDMap[BB] = ++Counter;
}

/// getGlobalBasicBlockID - This returns the function-specific ID for the
/// specified basic block.  This is relatively expensive information, so it
/// should only be used by rare constructs such as address-of-label.
unsigned ValueEnumerator::getGlobalBasicBlockID(const BasicBlock *BB) const {
  unsigned &Idx = GlobalBasicBlockIDs[BB];
  if (Idx != 0)
    return Idx-1;

  IncorporateFunctionInfoGlobalBBIDs(BB->getParent(), GlobalBasicBlockIDs);
  return getGlobalBasicBlockID(BB);
}

