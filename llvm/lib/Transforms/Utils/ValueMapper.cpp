//===- ValueMapper.cpp - Interface shared by lib/Transforms/Utils ---------===//
//
// This file defines the MapValue function, which is shared by various parts of
// the lib/Transforms/Utils library.
//
//===----------------------------------------------------------------------===//

#include "ValueMapper.h"
#include "llvm/Constants.h"
#include "llvm/Instruction.h"

Value *MapValue(const Value *V, std::map<const Value*, Value*> &VM) {
  Value *&VMSlot = VM[V];
  if (VMSlot) return VMSlot;      // Does it exist in the map yet?
  
  if (Constant *C = const_cast<Constant*>(dyn_cast<Constant>(V))) {
    if (isa<ConstantIntegral>(C) || isa<ConstantFP>(C) ||
        isa<ConstantPointerNull>(C))
      return VMSlot = C;           // Primitive constants map directly
    else if (ConstantPointerRef *CPR = dyn_cast<ConstantPointerRef>(C)) {
      GlobalValue *MV = cast<GlobalValue>(MapValue((Value*)CPR->getValue(),VM));
      return VMSlot = ConstantPointerRef::get(MV);
    } else if (ConstantArray *CA = dyn_cast<ConstantArray>(C)) {
      const std::vector<Use> &Vals = CA->getValues();
      for (unsigned i = 0, e = Vals.size(); i != e; ++i) {
        Value *MV = MapValue(Vals[i], VM);
        if (MV != Vals[i]) {
          // This array must contain a reference to a global, make a new array
          // and return it.
          //
          std::vector<Constant*> Values;
          Values.reserve(Vals.size());
          for (unsigned j = 0; j != i; ++j)
            Values.push_back(cast<Constant>(Vals[j]));
          Values.push_back(cast<Constant>(MV));
          for (++i; i != e; ++i)
            Values.push_back(cast<Constant>(MapValue(Vals[i], VM)));
          return VMSlot = ConstantArray::get(CA->getType(), Values);
        }
      }
      return VMSlot = C;

    } else if (ConstantStruct *CS = dyn_cast<ConstantStruct>(C)) {
      const std::vector<Use> &Vals = CS->getValues();
      for (unsigned i = 0, e = Vals.size(); i != e; ++i) {
        Value *MV = MapValue(Vals[i], VM);
        if (MV != Vals[i]) {
          // This struct must contain a reference to a global, make a new struct
          // and return it.
          //
          std::vector<Constant*> Values;
          Values.reserve(Vals.size());
          for (unsigned j = 0; j != i; ++j)
            Values.push_back(cast<Constant>(Vals[j]));
          Values.push_back(cast<Constant>(MV));
          for (++i; i != e; ++i)
            Values.push_back(cast<Constant>(MapValue(Vals[i], VM)));
          return VMSlot = ConstantStruct::get(CS->getType(), Values);
        }
      }
      return VMSlot = C;

    } else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
      if (CE->getOpcode() == Instruction::Cast) {
        Constant *MV = cast<Constant>(MapValue(CE->getOperand(0), VM));
        return VMSlot = ConstantExpr::getCast(MV, CE->getType());
      } else if (CE->getOpcode() == Instruction::GetElementPtr) {
        std::vector<Constant*> Idx;
        Constant *MV = cast<Constant>(MapValue(CE->getOperand(0), VM));
        for (unsigned i = 1, e = CE->getNumOperands(); i != e; ++i)
          Idx.push_back(cast<Constant>(MapValue(CE->getOperand(i), VM)));
        return VMSlot = ConstantExpr::getGetElementPtr(MV, Idx);
      } else {
        assert(CE->getNumOperands() == 2 && "Must be binary operator?");
        Constant *MV1 = cast<Constant>(MapValue(CE->getOperand(0), VM));
        Constant *MV2 = cast<Constant>(MapValue(CE->getOperand(1), VM));
        return VMSlot = ConstantExpr::get(CE->getOpcode(), MV1, MV2);
      }

    } else {
      assert(0 && "Unknown type of constant!");
    }
  }

  V->dump();
  assert(0 && "Unknown value type: why didn't it get resolved?!");
  return 0;
}

