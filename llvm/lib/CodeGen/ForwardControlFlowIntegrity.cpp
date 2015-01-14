//===-- ForwardControlFlowIntegrity.cpp: Forward-Edge CFI -----------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief A pass that instruments code with fast checks for indirect calls and
/// hooks for a function to check violations.
///
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "cfi"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/JumpInstrTableInfo.h"
#include "llvm/CodeGen/ForwardControlFlowIntegrity.h"
#include "llvm/CodeGen/JumpInstrTables.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

STATISTIC(NumCFIIndirectCalls,
          "Number of indirect call sites rewritten by the CFI pass");

char ForwardControlFlowIntegrity::ID = 0;
INITIALIZE_PASS_BEGIN(ForwardControlFlowIntegrity, "forward-cfi",
                      "Control-Flow Integrity", true, true)
INITIALIZE_PASS_DEPENDENCY(JumpInstrTableInfo);
INITIALIZE_PASS_DEPENDENCY(JumpInstrTables);
INITIALIZE_PASS_END(ForwardControlFlowIntegrity, "forward-cfi",
                    "Control-Flow Integrity", true, true)

ModulePass *llvm::createForwardControlFlowIntegrityPass() {
  return new ForwardControlFlowIntegrity();
}

ModulePass *llvm::createForwardControlFlowIntegrityPass(
    JumpTable::JumpTableType JTT, CFIntegrity CFIType, bool CFIEnforcing,
    StringRef CFIFuncName) {
  return new ForwardControlFlowIntegrity(JTT, CFIType, CFIEnforcing,
                                         CFIFuncName);
}

// Checks to see if a given CallSite is making an indirect call, including
// cases where the indirect call is made through a bitcast.
static bool isIndirectCall(CallSite &CS) {
  if (CS.getCalledFunction())
    return false;

  // Check the value to see if it is merely a bitcast of a function. In
  // this case, it will translate to a direct function call in the resulting
  // assembly, so we won't treat it as an indirect call here.
  const Value *V = CS.getCalledValue();
  if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(V)) {
    return !(CE->isCast() && isa<Function>(CE->getOperand(0)));
  }

  // Otherwise, since we know it's a call, it must be an indirect call
  return true;
}

static const char cfi_failure_func_name[] = "__llvm_cfi_pointer_warning";

ForwardControlFlowIntegrity::ForwardControlFlowIntegrity()
    : ModulePass(ID), IndirectCalls(), JTType(JumpTable::Single),
      CFIType(CFIntegrity::Sub), CFIEnforcing(false), CFIFuncName("") {
  initializeForwardControlFlowIntegrityPass(*PassRegistry::getPassRegistry());
}

ForwardControlFlowIntegrity::ForwardControlFlowIntegrity(
    JumpTable::JumpTableType JTT, CFIntegrity CFIType, bool CFIEnforcing,
    std::string CFIFuncName)
    : ModulePass(ID), IndirectCalls(), JTType(JTT), CFIType(CFIType),
      CFIEnforcing(CFIEnforcing), CFIFuncName(CFIFuncName) {
  initializeForwardControlFlowIntegrityPass(*PassRegistry::getPassRegistry());
}

ForwardControlFlowIntegrity::~ForwardControlFlowIntegrity() {}

void ForwardControlFlowIntegrity::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<JumpInstrTableInfo>();
  AU.addRequired<JumpInstrTables>();
}

void ForwardControlFlowIntegrity::getIndirectCalls(Module &M) {
  // To get the indirect calls, we iterate over all functions and iterate over
  // the list of basic blocks in each. We extract a total list of indirect calls
  // before modifying any of them, since our modifications will modify the list
  // of basic blocks.
  for (Function &F : M) {
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        CallSite CS(&I);
        if (!(CS && isIndirectCall(CS)))
          continue;

        Value *CalledValue = CS.getCalledValue();

        // Don't rewrite this instruction if the indirect call is actually just
        // inline assembly, since our transformation will generate an invalid
        // module in that case.
        if (isa<InlineAsm>(CalledValue))
          continue;

        IndirectCalls.push_back(&I);
      }
    }
  }
}

void ForwardControlFlowIntegrity::updateIndirectCalls(Module &M,
                                                      CFITables &CFIT) {
  Type *Int64Ty = Type::getInt64Ty(M.getContext());
  for (Instruction *I : IndirectCalls) {
    CallSite CS(I);
    Value *CalledValue = CS.getCalledValue();

    // Get the function type for this call and look it up in the tables.
    Type *VTy = CalledValue->getType();
    PointerType *PTy = dyn_cast<PointerType>(VTy);
    Type *EltTy = PTy->getElementType();
    FunctionType *FunTy = dyn_cast<FunctionType>(EltTy);
    FunctionType *TransformedTy = JumpInstrTables::transformType(JTType, FunTy);
    ++NumCFIIndirectCalls;
    Constant *JumpTableStart = nullptr;
    Constant *JumpTableMask = nullptr;
    Constant *JumpTableSize = nullptr;

    // Some call sites have function types that don't correspond to any
    // address-taken function in the module. This happens when function pointers
    // are passed in from external code.
    auto it = CFIT.find(TransformedTy);
    if (it == CFIT.end()) {
      // In this case, make sure that the function pointer will change by
      // setting the mask and the start to be 0 so that the transformed
      // function is 0.
      JumpTableStart = ConstantInt::get(Int64Ty, 0);
      JumpTableMask = ConstantInt::get(Int64Ty, 0);
      JumpTableSize = ConstantInt::get(Int64Ty, 0);
    } else {
      JumpTableStart = it->second.StartValue;
      JumpTableMask = it->second.MaskValue;
      JumpTableSize = it->second.Size;
    }

    rewriteFunctionPointer(M, I, CalledValue, JumpTableStart, JumpTableMask,
                           JumpTableSize);
  }

  return;
}

bool ForwardControlFlowIntegrity::runOnModule(Module &M) {
  JumpInstrTableInfo *JITI = &getAnalysis<JumpInstrTableInfo>();
  Type *Int64Ty = Type::getInt64Ty(M.getContext());
  Type *VoidPtrTy = Type::getInt8PtrTy(M.getContext());

  // JumpInstrTableInfo stores information about the alignment of each entry.
  // The alignment returned by JumpInstrTableInfo is alignment in bytes, not
  // in the exponent.
  ByteAlignment = JITI->entryByteAlignment();
  LogByteAlignment = llvm::Log2_64(ByteAlignment);

  // Set up tables for control-flow integrity based on information about the
  // jump-instruction tables.
  CFITables CFIT;
  for (const auto &KV : JITI->getTables()) {
    uint64_t Size = static_cast<uint64_t>(KV.second.size());
    uint64_t TableSize = NextPowerOf2(Size);

    int64_t MaskValue = ((TableSize << LogByteAlignment) - 1) & -ByteAlignment;
    Constant *JumpTableMaskValue = ConstantInt::get(Int64Ty, MaskValue);
    Constant *JumpTableSize = ConstantInt::get(Int64Ty, Size);

    // The base of the table is defined to be the first jumptable function in
    // the table.
    Function *First = KV.second.begin()->second;
    Constant *JumpTableStartValue = ConstantExpr::getBitCast(First, VoidPtrTy);
    CFIT[KV.first].StartValue = JumpTableStartValue;
    CFIT[KV.first].MaskValue = JumpTableMaskValue;
    CFIT[KV.first].Size = JumpTableSize;
  }

  if (CFIT.empty())
    return false;

  getIndirectCalls(M);

  if (!CFIEnforcing) {
    addWarningFunction(M);
  }

  // Update the instructions with the check and the indirect jump through our
  // table.
  updateIndirectCalls(M, CFIT);

  return true;
}

void ForwardControlFlowIntegrity::addWarningFunction(Module &M) {
  PointerType *CharPtrTy = Type::getInt8PtrTy(M.getContext());

  // Get the type of the Warning Function: void (i8*, i8*),
  // where the first argument is the name of the function in which the violation
  // occurs, and the second is the function pointer that violates CFI.
  SmallVector<Type *, 2> WarningFunArgs;
  WarningFunArgs.push_back(CharPtrTy);
  WarningFunArgs.push_back(CharPtrTy);
  FunctionType *WarningFunTy =
      FunctionType::get(Type::getVoidTy(M.getContext()), WarningFunArgs, false);

  if (!CFIFuncName.empty()) {
    Constant *FailureFun = M.getOrInsertFunction(CFIFuncName, WarningFunTy);
    if (!FailureFun)
      report_fatal_error("Could not get or insert the function specified by"
                         " -cfi-func-name");
  } else {
    // The default warning function swallows the warning and lets the call
    // continue, since there's no generic way for it to print out this
    // information.
    Function *WarningFun = M.getFunction(cfi_failure_func_name);
    if (!WarningFun) {
      WarningFun =
          Function::Create(WarningFunTy, GlobalValue::LinkOnceAnyLinkage,
                           cfi_failure_func_name, &M);
    }

    BasicBlock *Entry =
        BasicBlock::Create(M.getContext(), "entry", WarningFun, 0);
    ReturnInst::Create(M.getContext(), Entry);
  }
}

void ForwardControlFlowIntegrity::rewriteFunctionPointer(
    Module &M, Instruction *I, Value *FunPtr, Constant *JumpTableStart,
    Constant *JumpTableMask, Constant *JumpTableSize) {
  IRBuilder<> TempBuilder(I);

  Type *OrigFunType = FunPtr->getType();

  BasicBlock *CurBB = cast<BasicBlock>(I->getParent());
  Function *CurF = cast<Function>(CurBB->getParent());
  Type *Int64Ty = Type::getInt64Ty(M.getContext());

  Value *TI = TempBuilder.CreatePtrToInt(FunPtr, Int64Ty);
  Value *TStartInt = TempBuilder.CreatePtrToInt(JumpTableStart, Int64Ty);

  Value *NewFunPtr = nullptr;
  Value *Check = nullptr;
  switch (CFIType) {
  case CFIntegrity::Sub: {
    // This is the subtract, mask, and add version.
    // Subtract from the base.
    Value *Sub = TempBuilder.CreateSub(TI, TStartInt);

    // Mask the difference to force this to be a table offset.
    Value *And = TempBuilder.CreateAnd(Sub, JumpTableMask);

    // Add it back to the base.
    Value *Result = TempBuilder.CreateAdd(And, TStartInt);

    // Convert it back into a function pointer that we can call.
    NewFunPtr = TempBuilder.CreateIntToPtr(Result, OrigFunType);
    break;
  }
  case CFIntegrity::Ror: {
    // This is the subtract and rotate version.
    // Rotate right by the alignment value. The optimizer should recognize
    // this sequence as a rotation.

    // This cast is safe, since unsigned is always a subset of uint64_t.
    uint64_t LogByteAlignment64 = static_cast<uint64_t>(LogByteAlignment);
    Constant *RightShift = ConstantInt::get(Int64Ty, LogByteAlignment64);
    Constant *LeftShift = ConstantInt::get(Int64Ty, 64 - LogByteAlignment64);

    // Subtract from the base.
    Value *Sub = TempBuilder.CreateSub(TI, TStartInt);

    // Create the equivalent of a rotate-right instruction.
    Value *Shr = TempBuilder.CreateLShr(Sub, RightShift);
    Value *Shl = TempBuilder.CreateShl(Sub, LeftShift);
    Value *Or = TempBuilder.CreateOr(Shr, Shl);

    // Perform unsigned comparison to check for inclusion in the table.
    Check = TempBuilder.CreateICmpULT(Or, JumpTableSize);
    NewFunPtr = FunPtr;
    break;
  }
  case CFIntegrity::Add: {
    // This is the mask and add version.
    // Mask the function pointer to turn it into an offset into the table.
    Value *And = TempBuilder.CreateAnd(TI, JumpTableMask);

    // Then or this offset to the base and get the pointer value.
    Value *Result = TempBuilder.CreateAdd(And, TStartInt);

    // Convert it back into a function pointer that we can call.
    NewFunPtr = TempBuilder.CreateIntToPtr(Result, OrigFunType);
    break;
  }
  }

  if (!CFIEnforcing) {
    // If a check hasn't been added (in the rotation version), then check to see
    // if it's the same as the original function. This check determines whether
    // or not we call the CFI failure function.
    if (!Check)
      Check = TempBuilder.CreateICmpEQ(NewFunPtr, FunPtr);
    BasicBlock *InvalidPtrBlock =
        BasicBlock::Create(M.getContext(), "invalid.ptr", CurF, 0);
    BasicBlock *ContinuationBB = CurBB->splitBasicBlock(I);

    // Remove the unconditional branch that connects the two blocks.
    TerminatorInst *TermInst = CurBB->getTerminator();
    TermInst->eraseFromParent();

    // Add a conditional branch that depends on the Check above.
    BranchInst::Create(ContinuationBB, InvalidPtrBlock, Check, CurBB);

    // Call the warning function for this pointer, then continue.
    Instruction *BI = BranchInst::Create(ContinuationBB, InvalidPtrBlock);
    insertWarning(M, InvalidPtrBlock, BI, FunPtr);
  } else {
    // Modify the instruction to call this value.
    CallSite CS(I);
    CS.setCalledFunction(NewFunPtr);
  }
}

void ForwardControlFlowIntegrity::insertWarning(Module &M, BasicBlock *Block,
                                                Instruction *I, Value *FunPtr) {
  Function *ParentFun = cast<Function>(Block->getParent());

  // Get the function to call right before the instruction.
  Function *WarningFun = nullptr;
  if (CFIFuncName.empty()) {
    WarningFun = M.getFunction(cfi_failure_func_name);
  } else {
    WarningFun = M.getFunction(CFIFuncName);
  }

  assert(WarningFun && "Could not find the CFI failure function");

  Type *VoidPtrTy = Type::getInt8PtrTy(M.getContext());

  IRBuilder<> WarningInserter(I);
  // Create a mergeable GlobalVariable containing the name of the function.
  Value *ParentNameGV =
      WarningInserter.CreateGlobalString(ParentFun->getName());
  Value *ParentNamePtr = WarningInserter.CreateBitCast(ParentNameGV, VoidPtrTy);
  Value *FunVoidPtr = WarningInserter.CreateBitCast(FunPtr, VoidPtrTy);
  WarningInserter.CreateCall2(WarningFun, ParentNamePtr, FunVoidPtr);
}
