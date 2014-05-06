//===- CtorUtils.cpp - Helpers for working with global_ctors ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines functions that are used to process llvm.global_ctors.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/CtorUtils.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "ctor_utils"

namespace llvm {

namespace {
/// Given a specified llvm.global_ctors list, install the
/// specified array.
void installGlobalCtors(GlobalVariable *GCL,
                        const std::vector<Function *> &Ctors) {
  // If we made a change, reassemble the initializer list.
  Constant *CSVals[2];
  CSVals[0] = ConstantInt::get(Type::getInt32Ty(GCL->getContext()), 65535);
  CSVals[1] = nullptr;

  StructType *StructTy =
      cast<StructType>(GCL->getType()->getElementType()->getArrayElementType());

  // Create the new init list.
  std::vector<Constant *> CAList;
  for (unsigned i = 0, e = Ctors.size(); i != e; ++i) {
    if (Ctors[i]) {
      CSVals[1] = Ctors[i];
    } else {
      Type *FTy = FunctionType::get(Type::getVoidTy(GCL->getContext()), false);
      PointerType *PFTy = PointerType::getUnqual(FTy);
      CSVals[1] = Constant::getNullValue(PFTy);
      CSVals[0] =
          ConstantInt::get(Type::getInt32Ty(GCL->getContext()), 0x7fffffff);
    }
    CAList.push_back(ConstantStruct::get(StructTy, CSVals));
  }

  // Create the array initializer.
  Constant *CA =
      ConstantArray::get(ArrayType::get(StructTy, CAList.size()), CAList);

  // If we didn't change the number of elements, don't create a new GV.
  if (CA->getType() == GCL->getInitializer()->getType()) {
    GCL->setInitializer(CA);
    return;
  }

  // Create the new global and insert it next to the existing list.
  GlobalVariable *NGV =
      new GlobalVariable(CA->getType(), GCL->isConstant(), GCL->getLinkage(),
                         CA, "", GCL->getThreadLocalMode());
  GCL->getParent()->getGlobalList().insert(GCL, NGV);
  NGV->takeName(GCL);

  // Nuke the old list, replacing any uses with the new one.
  if (!GCL->use_empty()) {
    Constant *V = NGV;
    if (V->getType() != GCL->getType())
      V = ConstantExpr::getBitCast(V, GCL->getType());
    GCL->replaceAllUsesWith(V);
  }
  GCL->eraseFromParent();
}

/// Given a llvm.global_ctors list that we can understand,
/// return a list of the functions and null terminator as a vector.
std::vector<Function*> parseGlobalCtors(GlobalVariable *GV) {
  if (GV->getInitializer()->isNullValue())
    return std::vector<Function *>();
  ConstantArray *CA = cast<ConstantArray>(GV->getInitializer());
  std::vector<Function *> Result;
  Result.reserve(CA->getNumOperands());
  for (User::op_iterator i = CA->op_begin(), e = CA->op_end(); i != e; ++i) {
    ConstantStruct *CS = cast<ConstantStruct>(*i);
    Result.push_back(dyn_cast<Function>(CS->getOperand(1)));
  }
  return Result;
}

/// Find the llvm.global_ctors list, verifying that all initializers have an
/// init priority of 65535.
GlobalVariable *findGlobalCtors(Module &M) {
  GlobalVariable *GV = M.getGlobalVariable("llvm.global_ctors");
  if (!GV)
    return nullptr;

  // Verify that the initializer is simple enough for us to handle. We are
  // only allowed to optimize the initializer if it is unique.
  if (!GV->hasUniqueInitializer())
    return nullptr;

  if (isa<ConstantAggregateZero>(GV->getInitializer()))
    return GV;
  ConstantArray *CA = cast<ConstantArray>(GV->getInitializer());

  for (User::op_iterator i = CA->op_begin(), e = CA->op_end(); i != e; ++i) {
    if (isa<ConstantAggregateZero>(*i))
      continue;
    ConstantStruct *CS = cast<ConstantStruct>(*i);
    if (isa<ConstantPointerNull>(CS->getOperand(1)))
      continue;

    // Must have a function or null ptr.
    if (!isa<Function>(CS->getOperand(1)))
      return nullptr;

    // Init priority must be standard.
    ConstantInt *CI = cast<ConstantInt>(CS->getOperand(0));
    if (CI->getZExtValue() != 65535)
      return nullptr;
  }

  return GV;
}
} // namespace

/// Call "ShouldRemove" for every entry in M's global_ctor list and remove the
/// entries for which it returns true.  Return true if anything changed.
bool optimizeGlobalCtorsList(Module &M,
                             function_ref<bool(Function *)> ShouldRemove) {
  GlobalVariable *GlobalCtors = findGlobalCtors(M);
  if (!GlobalCtors)
    return false;

  std::vector<Function *> Ctors = parseGlobalCtors(GlobalCtors);
  if (Ctors.empty())
    return false;

  bool MadeChange = false;

  // Loop over global ctors, optimizing them when we can.
  for (unsigned i = 0; i != Ctors.size(); ++i) {
    Function *F = Ctors[i];
    // Found a null terminator in the middle of the list, prune off the rest of
    // the list.
    if (!F) {
      if (i != Ctors.size() - 1) {
        Ctors.resize(i + 1);
        MadeChange = true;
      }
      break;
    }
    DEBUG(dbgs() << "Optimizing Global Constructor: " << *F << "\n");

    // We cannot simplify external ctor functions.
    if (F->empty())
      continue;

    // If we can evaluate the ctor at compile time, do.
    if (ShouldRemove(F)) {
      Ctors.erase(Ctors.begin() + i);
      MadeChange = true;
      --i;
      continue;
    }
  }

  if (!MadeChange)
    return false;

  installGlobalCtors(GlobalCtors, Ctors);
  return true;
}

} // End llvm namespace
