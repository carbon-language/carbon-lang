//===- ValueMapper.cpp - Interface shared by lib/Transforms/Utils ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the MapValue function, which is shared by various parts of
// the lib/Transforms/Utils library.
//
//===----------------------------------------------------------------------===//

#include "ValueMapper.h"
#include "llvm/Constants.h"
#include "llvm/GlobalValue.h"
#include "llvm/Instruction.h"
using namespace llvm;

Value *llvm::MapValue(const Value *V, std::map<const Value*, Value*> &VM) {
  Value *&VMSlot = VM[V];
  if (VMSlot) return VMSlot;      // Does it exist in the map yet?

  // Global values do not need to be seeded into the ValueMap if they are using
  // the identity mapping.
  if (isa<GlobalValue>(V) || isa<InlineAsm>(V))
    return VMSlot = const_cast<Value*>(V);

  if (Constant *C = const_cast<Constant*>(dyn_cast<Constant>(V))) {
    if (isa<ConstantIntegral>(C) || isa<ConstantFP>(C) ||
        isa<ConstantPointerNull>(C) || isa<ConstantAggregateZero>(C) ||
        isa<UndefValue>(C))
      return VMSlot = C;           // Primitive constants map directly
    else if (ConstantArray *CA = dyn_cast<ConstantArray>(C)) {
      for (unsigned i = 0, e = CA->getNumOperands(); i != e; ++i) {
        Value *MV = MapValue(CA->getOperand(i), VM);
        if (MV != CA->getOperand(i)) {
          // This array must contain a reference to a global, make a new array
          // and return it.
          //
          std::vector<Constant*> Values;
          Values.reserve(CA->getNumOperands());
          for (unsigned j = 0; j != i; ++j)
            Values.push_back(CA->getOperand(j));
          Values.push_back(cast<Constant>(MV));
          for (++i; i != e; ++i)
            Values.push_back(cast<Constant>(MapValue(CA->getOperand(i), VM)));
          return VMSlot = ConstantArray::get(CA->getType(), Values);
        }
      }
      return VMSlot = C;

    } else if (ConstantStruct *CS = dyn_cast<ConstantStruct>(C)) {
      for (unsigned i = 0, e = CS->getNumOperands(); i != e; ++i) {
        Value *MV = MapValue(CS->getOperand(i), VM);
        if (MV != CS->getOperand(i)) {
          // This struct must contain a reference to a global, make a new struct
          // and return it.
          //
          std::vector<Constant*> Values;
          Values.reserve(CS->getNumOperands());
          for (unsigned j = 0; j != i; ++j)
            Values.push_back(CS->getOperand(j));
          Values.push_back(cast<Constant>(MV));
          for (++i; i != e; ++i)
            Values.push_back(cast<Constant>(MapValue(CS->getOperand(i), VM)));
          return VMSlot = ConstantStruct::get(CS->getType(), Values);
        }
      }
      return VMSlot = C;

    } else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
      std::vector<Constant*> Ops;
      for (unsigned i = 0, e = CE->getNumOperands(); i != e; ++i)
        Ops.push_back(cast<Constant>(MapValue(CE->getOperand(i), VM)));
      return VMSlot = CE->getWithOperands(Ops);
    } else if (ConstantPacked *CP = dyn_cast<ConstantPacked>(C)) {
      for (unsigned i = 0, e = CP->getNumOperands(); i != e; ++i) {
        Value *MV = MapValue(CP->getOperand(i), VM);
        if (MV != CP->getOperand(i)) {
          // This packed value must contain a reference to a global, make a new
          // packed constant and return it.
          //
          std::vector<Constant*> Values;
          Values.reserve(CP->getNumOperands());
          for (unsigned j = 0; j != i; ++j)
            Values.push_back(CP->getOperand(j));
          Values.push_back(cast<Constant>(MV));
          for (++i; i != e; ++i)
            Values.push_back(cast<Constant>(MapValue(CP->getOperand(i), VM)));
          return VMSlot = ConstantPacked::get(Values);
        }
      }
      return VMSlot = C;
      
    } else {
      assert(0 && "Unknown type of constant!");
    }
  }

  return 0;
}

/// RemapInstruction - Convert the instruction operands from referencing the
/// current values into those specified by ValueMap.
///
void llvm::RemapInstruction(Instruction *I,
                            std::map<const Value *, Value*> &ValueMap) {
  for (unsigned op = 0, E = I->getNumOperands(); op != E; ++op) {
    const Value *Op = I->getOperand(op);
    Value *V = MapValue(Op, ValueMap);
    assert(V && "Referenced value not in value map!");
    I->setOperand(op, V);
  }
}
