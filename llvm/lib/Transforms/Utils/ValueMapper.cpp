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

#include "llvm/Transforms/Utils/ValueMapper.h"
#include "llvm/Type.h"
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/Metadata.h"
#include "llvm/ADT/SmallVector.h"
using namespace llvm;

Value *llvm::MapValue(const Value *V, ValueToValueMapTy &VM,
                      RemapFlags Flags) {
  ValueToValueMapTy::iterator I = VM.find(V);
  
  // If the value already exists in the map, use it.
  if (I != VM.end() && I->second) return I->second;
  
  // Global values do not need to be seeded into the VM if they
  // are using the identity mapping.
  if (isa<GlobalValue>(V) || isa<InlineAsm>(V) || isa<MDString>(V))
    return VM[V] = const_cast<Value*>(V);

  if (const MDNode *MD = dyn_cast<MDNode>(V)) {
    // If this is a module-level metadata and we know that nothing at the module
    // level is changing, then use an identity mapping.
    if (!MD->isFunctionLocal() && (Flags & RF_NoModuleLevelChanges))
      return VM[V] = const_cast<Value*>(V);
    
    // Check all operands to see if any need to be remapped.
    for (unsigned i = 0, e = MD->getNumOperands(); i != e; ++i) {
      Value *OP = MD->getOperand(i);
      if (OP == 0 || MapValue(OP, VM, Flags) == OP) continue;

      // Ok, at least one operand needs remapping.  Create a dummy node in case
      // we have a metadata cycle.
      MDNode *Dummy = MDNode::getTemporary(V->getContext(), 0, 0);
      VM[V] = Dummy;
      SmallVector<Value*, 4> Elts;
      Elts.reserve(MD->getNumOperands());
      for (i = 0; i != e; ++i) {
        Value *Op = MD->getOperand(i);
        Elts.push_back(Op ? MapValue(Op, VM, Flags) : 0);
      }
      MDNode *NewMD = MDNode::get(V->getContext(), Elts.data(), Elts.size());
      Dummy->replaceAllUsesWith(NewMD);
      MDNode::deleteTemporary(Dummy);
      return VM[V] = NewMD;
    }

    // No operands needed remapping.  Use an identity mapping.
    return VM[V] = const_cast<Value*>(V);
  }

  // Okay, this either must be a constant (which may or may not be mappable) or
  // is something that is not in the mapping table.
  Constant *C = const_cast<Constant*>(dyn_cast<Constant>(V));
  if (C == 0)
    return 0;
  
  if (BlockAddress *BA = dyn_cast<BlockAddress>(C)) {
    Function *F = cast<Function>(MapValue(BA->getFunction(), VM, Flags));
    BasicBlock *BB = cast_or_null<BasicBlock>(MapValue(BA->getBasicBlock(), VM,
                                                       Flags));
    return VM[V] = BlockAddress::get(F, BB ? BB : BA->getBasicBlock());
  }
  
  for (unsigned i = 0, e = C->getNumOperands(); i != e; ++i) {
    Value *Op = C->getOperand(i);
    Value *Mapped = MapValue(Op, VM, Flags);
    if (Mapped == C) continue;
    
    // Okay, the operands don't all match.  We've already processed some or all
    // of the operands, set them up now.
    std::vector<Constant*> Ops;
    Ops.reserve(C->getNumOperands());
    for (unsigned j = 0; j != i; ++j)
      Ops.push_back(cast<Constant>(C->getOperand(i)));
    Ops.push_back(cast<Constant>(Mapped));
    
    // Map the rest of the operands that aren't processed yet.
    for (++i; i != e; ++i)
      Ops.push_back(cast<Constant>(MapValue(C->getOperand(i), VM, Flags)));
    
    if (ConstantExpr *CE = dyn_cast<ConstantExpr>(C))
      return VM[V] = CE->getWithOperands(Ops);
    if (ConstantArray *CA = dyn_cast<ConstantArray>(C))
      return VM[V] = ConstantArray::get(CA->getType(), Ops);
    if (ConstantStruct *CS = dyn_cast<ConstantStruct>(C))
      return VM[V] = ConstantStruct::get(CS->getType(), Ops);
    assert(isa<ConstantVector>(C) && "Unknown mapped constant type");
    return VM[V] = ConstantVector::get(Ops);
  }

  // If we reach here, all of the operands of the constant match.
  return VM[V] = C;
}

/// RemapInstruction - Convert the instruction operands from referencing the
/// current values into those specified by VMap.
///
void llvm::RemapInstruction(Instruction *I, ValueToValueMapTy &VMap,
                            RemapFlags Flags) {
  // Remap operands.
  for (User::op_iterator op = I->op_begin(), E = I->op_end(); op != E; ++op) {
    Value *V = MapValue(*op, VMap, Flags);
    // If we aren't ignoring missing entries, assert that something happened.
    if (V != 0)
      *op = V;
    else
      assert((Flags & RF_IgnoreMissingEntries) &&
             "Referenced value not in value map!");
  }

  // Remap attached metadata.
  SmallVector<std::pair<unsigned, MDNode *>, 4> MDs;
  I->getAllMetadata(MDs);
  for (SmallVectorImpl<std::pair<unsigned, MDNode *> >::iterator
       MI = MDs.begin(), ME = MDs.end(); MI != ME; ++MI) {
    Value *Old = MI->second;
    Value *New = MapValue(Old, VMap, Flags);
    if (New != Old)
      I->setMetadata(MI->first, cast<MDNode>(New));
  }
}
