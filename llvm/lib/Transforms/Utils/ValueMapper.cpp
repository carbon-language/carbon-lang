//===- ValueMapper.cpp - Interface shared by lib/Transforms/Utils ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the MapValue function, which is shared by various parts of
// the lib/Transforms/Utils library.
//
//===----------------------------------------------------------------------===//

#include "ValueMapper.h"
#include "llvm/Type.h"
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/Metadata.h"
#include "llvm/ADT/SmallVector.h"
using namespace llvm;

Value *llvm::MapValue(const Value *V, ValueToValueMapTy &VM) {
  ValueToValueMapTy::iterator VMI = VM.find(V);
  if (VMI != VM.end()) 
    return VMI->second;      // Does it exist in the map yet?
  
  // Global values, metadata strings and inline asm do not need to be seeded into
  // the ValueMap if they are using the identity mapping.
  if (isa<GlobalValue>(V) || isa<InlineAsm>(V) || isa<MDString>(V)) {
    VM.insert(std::make_pair(V, const_cast<Value*>(V)));
    return const_cast<Value*>(V);
  }

  if (const MDNode *MD = dyn_cast<MDNode>(V)) {
    // Insert a place holder in map to handle mdnode cycles.
    Value *TmpV = MDString::get(V->getContext(),
                                std::string("llvm.md.clone.tmp." + VM.size()));
    VM.insert(std::make_pair(V, MDNode::get(V->getContext(), &TmpV, 1)));
    
    bool ReuseMD = true;
    SmallVector<Value*, 4> Elts;
    // If metadata element is mapped to a new value then seed metadata 
    // in the map.
    for (unsigned i = 0, e = MD->getNumOperands(); i != e; ++i) {
      if (!MD->getOperand(i))
        Elts.push_back(0);
      else {
        Value *MappedOp = MapValue(MD->getOperand(i), VM);
        if (MappedOp != MD->getOperand(i))
          ReuseMD = false;
        Elts.push_back(MappedOp);
      }
    }
    if (ReuseMD) {
      VM.insert(std::make_pair(V, const_cast<Value*>(V)));
      return const_cast<Value*>(V);
    }
    MDNode *NewMD = MDNode::get(V->getContext(), Elts.data(), Elts.size());
    VM.insert(std::make_pair(V, NewMD));
    return NewMD;
  }

  Constant *C = const_cast<Constant*>(dyn_cast<Constant>(V));
  if (C == 0) return 0;
  
  if (isa<ConstantInt>(C) || isa<ConstantFP>(C) ||
      isa<ConstantPointerNull>(C) || isa<ConstantAggregateZero>(C) ||
      isa<UndefValue>(C) || isa<MDString>(C))
    return VM[V] = C;           // Primitive constants map directly
  
  if (ConstantArray *CA = dyn_cast<ConstantArray>(C)) {
    for (User::op_iterator b = CA->op_begin(), i = b, e = CA->op_end();
         i != e; ++i) {
      Value *MV = MapValue(*i, VM);
      if (MV != *i) {
        // This array must contain a reference to a global, make a new array
        // and return it.
        //
        std::vector<Constant*> Values;
        Values.reserve(CA->getNumOperands());
        for (User::op_iterator j = b; j != i; ++j)
          Values.push_back(cast<Constant>(*j));
        Values.push_back(cast<Constant>(MV));
        for (++i; i != e; ++i)
          Values.push_back(cast<Constant>(MapValue(*i, VM)));
        return VM[V] = ConstantArray::get(CA->getType(), Values);
      }
    }
    return VM[V] = C;
  }
  
  if (ConstantStruct *CS = dyn_cast<ConstantStruct>(C)) {
    for (User::op_iterator b = CS->op_begin(), i = b, e = CS->op_end();
         i != e; ++i) {
      Value *MV = MapValue(*i, VM);
      if (MV != *i) {
        // This struct must contain a reference to a global, make a new struct
        // and return it.
        //
        std::vector<Constant*> Values;
        Values.reserve(CS->getNumOperands());
        for (User::op_iterator j = b; j != i; ++j)
          Values.push_back(cast<Constant>(*j));
        Values.push_back(cast<Constant>(MV));
        for (++i; i != e; ++i)
          Values.push_back(cast<Constant>(MapValue(*i, VM)));
        return VM[V] = ConstantStruct::get(CS->getType(), Values);
      }
    }
    return VM[V] = C;
  }
  
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
    std::vector<Constant*> Ops;
    for (User::op_iterator i = CE->op_begin(), e = CE->op_end(); i != e; ++i)
      Ops.push_back(cast<Constant>(MapValue(*i, VM)));
    return VM[V] = CE->getWithOperands(Ops);
  }
  
  if (ConstantVector *CV = dyn_cast<ConstantVector>(C)) {
    for (User::op_iterator b = CV->op_begin(), i = b, e = CV->op_end();
         i != e; ++i) {
      Value *MV = MapValue(*i, VM);
      if (MV != *i) {
        // This vector value must contain a reference to a global, make a new
        // vector constant and return it.
        //
        std::vector<Constant*> Values;
        Values.reserve(CV->getNumOperands());
        for (User::op_iterator j = b; j != i; ++j)
          Values.push_back(cast<Constant>(*j));
        Values.push_back(cast<Constant>(MV));
        for (++i; i != e; ++i)
          Values.push_back(cast<Constant>(MapValue(*i, VM)));
        return VM[V] = ConstantVector::get(Values);
      }
    }
    return VM[V] = C;
  }
  
  BlockAddress *BA = cast<BlockAddress>(C);
  Function *F = cast<Function>(MapValue(BA->getFunction(), VM));
  BasicBlock *BB = cast_or_null<BasicBlock>(MapValue(BA->getBasicBlock(),VM));
  return VM[V] = BlockAddress::get(F, BB ? BB : BA->getBasicBlock());
}

/// RemapInstruction - Convert the instruction operands from referencing the
/// current values into those specified by ValueMap.
///
void llvm::RemapInstruction(Instruction *I, ValueToValueMapTy &ValueMap) {
  for (User::op_iterator op = I->op_begin(), E = I->op_end(); op != E; ++op) {
    Value *V = MapValue(*op, ValueMap);
    assert(V && "Referenced value not in value map!");
    *op = V;
  }
}

