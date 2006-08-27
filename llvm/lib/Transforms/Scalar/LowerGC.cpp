//===-- LowerGC.cpp - Provide GC support for targets that don't -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering for the llvm.gc* intrinsics for targets that do
// not natively support them (which includes the C backend).  Note that the code
// generated is not as efficient as it would be for targets that natively
// support the GC intrinsics, but it is useful for getting new targets
// up-and-running quickly.
//
// This pass implements the code transformation described in this paper:
//   "Accurate Garbage Collection in an Uncooperative Environment"
//   Fergus Henderson, ISMM, 2002
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "lowergc"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/Compiler.h"
using namespace llvm;

namespace {
  class VISIBILITY_HIDDEN LowerGC : public FunctionPass {
    /// GCRootInt, GCReadInt, GCWriteInt - The function prototypes for the
    /// llvm.gcread/llvm.gcwrite/llvm.gcroot intrinsics.
    Function *GCRootInt, *GCReadInt, *GCWriteInt;

    /// GCRead/GCWrite - These are the functions provided by the garbage
    /// collector for read/write barriers.
    Function *GCRead, *GCWrite;

    /// RootChain - This is the global linked-list that contains the chain of GC
    /// roots.
    GlobalVariable *RootChain;

    /// MainRootRecordType - This is the type for a function root entry if it
    /// had zero roots.
    const Type *MainRootRecordType;
  public:
    LowerGC() : GCRootInt(0), GCReadInt(0), GCWriteInt(0),
                GCRead(0), GCWrite(0), RootChain(0), MainRootRecordType(0) {}
    virtual bool doInitialization(Module &M);
    virtual bool runOnFunction(Function &F);

  private:
    const StructType *getRootRecordType(unsigned NumRoots);
  };

  RegisterOpt<LowerGC>
  X("lowergc", "Lower GC intrinsics, for GCless code generators");
}

/// createLowerGCPass - This function returns an instance of the "lowergc"
/// pass, which lowers garbage collection intrinsics to normal LLVM code.
FunctionPass *llvm::createLowerGCPass() {
  return new LowerGC();
}

/// getRootRecordType - This function creates and returns the type for a root
/// record containing 'NumRoots' roots.
const StructType *LowerGC::getRootRecordType(unsigned NumRoots) {
  // Build a struct that is a type used for meta-data/root pairs.
  std::vector<const Type *> ST;
  ST.push_back(GCRootInt->getFunctionType()->getParamType(0));
  ST.push_back(GCRootInt->getFunctionType()->getParamType(1));
  StructType *PairTy = StructType::get(ST);

  // Build the array of pairs.
  ArrayType *PairArrTy = ArrayType::get(PairTy, NumRoots);

  // Now build the recursive list type.
  PATypeHolder RootListH =
    MainRootRecordType ? (Type*)MainRootRecordType : (Type*)OpaqueType::get();
  ST.clear();
  ST.push_back(PointerType::get(RootListH));         // Prev pointer
  ST.push_back(Type::UIntTy);                        // NumElements in array
  ST.push_back(PairArrTy);                           // The pairs
  StructType *RootList = StructType::get(ST);
  if (MainRootRecordType)
    return RootList;

  assert(NumRoots == 0 && "The main struct type should have zero entries!");
  cast<OpaqueType>((Type*)RootListH.get())->refineAbstractTypeTo(RootList);
  MainRootRecordType = RootListH;
  return cast<StructType>(RootListH.get());
}

/// doInitialization - If this module uses the GC intrinsics, find them now.  If
/// not, this pass does not do anything.
bool LowerGC::doInitialization(Module &M) {
  GCRootInt  = M.getNamedFunction("llvm.gcroot");
  GCReadInt  = M.getNamedFunction("llvm.gcread");
  GCWriteInt = M.getNamedFunction("llvm.gcwrite");
  if (!GCRootInt && !GCReadInt && !GCWriteInt) return false;

  PointerType *VoidPtr = PointerType::get(Type::SByteTy);
  PointerType *VoidPtrPtr = PointerType::get(VoidPtr);

  // If the program is using read/write barriers, find the implementations of
  // them from the GC runtime library.
  if (GCReadInt)        // Make:  sbyte* %llvm_gc_read(sbyte**)
    GCRead = M.getOrInsertFunction("llvm_gc_read", VoidPtr, VoidPtr, VoidPtrPtr,
                                   (Type *)0);
  if (GCWriteInt)       // Make:  void %llvm_gc_write(sbyte*, sbyte**)
    GCWrite = M.getOrInsertFunction("llvm_gc_write", Type::VoidTy,
                                    VoidPtr, VoidPtr, VoidPtrPtr, (Type *)0);

  // If the program has GC roots, get or create the global root list.
  if (GCRootInt) {
    const StructType *RootListTy = getRootRecordType(0);
    const Type *PRLTy = PointerType::get(RootListTy);
    M.addTypeName("llvm_gc_root_ty", RootListTy);

    // Get the root chain if it already exists.
    RootChain = M.getGlobalVariable("llvm_gc_root_chain", PRLTy);
    if (RootChain == 0) {
      // If the root chain does not exist, insert a new one with linkonce
      // linkage!
      RootChain = new GlobalVariable(PRLTy, false,
                                     GlobalValue::LinkOnceLinkage,
                                     Constant::getNullValue(PRLTy),
                                     "llvm_gc_root_chain", &M);
    } else if (RootChain->hasExternalLinkage() && RootChain->isExternal()) {
      RootChain->setInitializer(Constant::getNullValue(PRLTy));
      RootChain->setLinkage(GlobalValue::LinkOnceLinkage);
    }
  }
  return true;
}

/// Coerce - If the specified operand number of the specified instruction does
/// not have the specified type, insert a cast.
static void Coerce(Instruction *I, unsigned OpNum, Type *Ty) {
  if (I->getOperand(OpNum)->getType() != Ty) {
    if (Constant *C = dyn_cast<Constant>(I->getOperand(OpNum)))
      I->setOperand(OpNum, ConstantExpr::getCast(C, Ty));
    else {
      CastInst *CI = new CastInst(I->getOperand(OpNum), Ty, "", I);
      I->setOperand(OpNum, CI);
    }
  }
}

/// runOnFunction - If the program is using GC intrinsics, replace any
/// read/write intrinsics with the appropriate read/write barrier calls, then
/// inline them.  Finally, build the data structures for
bool LowerGC::runOnFunction(Function &F) {
  // Quick exit for programs that are not using GC mechanisms.
  if (!GCRootInt && !GCReadInt && !GCWriteInt) return false;

  PointerType *VoidPtr    = PointerType::get(Type::SByteTy);
  PointerType *VoidPtrPtr = PointerType::get(VoidPtr);

  // If there are read/write barriers in the program, perform a quick pass over
  // the function eliminating them.  While we are at it, remember where we see
  // calls to llvm.gcroot.
  std::vector<CallInst*> GCRoots;
  std::vector<CallInst*> NormalCalls;

  bool MadeChange = false;
  for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB)
    for (BasicBlock::iterator II = BB->begin(), E = BB->end(); II != E;)
      if (CallInst *CI = dyn_cast<CallInst>(II++)) {
        if (!CI->getCalledFunction() ||
            !CI->getCalledFunction()->getIntrinsicID())
          NormalCalls.push_back(CI);   // Remember all normal function calls.

        if (Function *F = CI->getCalledFunction())
          if (F == GCRootInt)
            GCRoots.push_back(CI);
          else if (F == GCReadInt || F == GCWriteInt) {
            if (F == GCWriteInt) {
              // Change a llvm.gcwrite call to call llvm_gc_write instead.
              CI->setOperand(0, GCWrite);
              // Insert casts of the operands as needed.
              Coerce(CI, 1, VoidPtr);
              Coerce(CI, 2, VoidPtr);
              Coerce(CI, 3, VoidPtrPtr);
            } else {
              Coerce(CI, 1, VoidPtr);
              Coerce(CI, 2, VoidPtrPtr);
              if (CI->getType() == VoidPtr) {
                CI->setOperand(0, GCRead);
              } else {
                // Create a whole new call to replace the old one.
                CallInst *NC = new CallInst(GCRead, CI->getOperand(1),
                                            CI->getOperand(2),
                                            CI->getName(), CI);
                Value *NV = new CastInst(NC, CI->getType(), "", CI);
                CI->replaceAllUsesWith(NV);
                BB->getInstList().erase(CI);
                CI = NC;
              }
            }

            MadeChange = true;
          }
      }

  // If there are no GC roots in this function, then there is no need to create
  // a GC list record for it.
  if (GCRoots.empty()) return MadeChange;

  // Okay, there are GC roots in this function.  On entry to the function, add a
  // record to the llvm_gc_root_chain, and remove it on exit.

  // Create the alloca, and zero it out.
  const StructType *RootListTy = getRootRecordType(GCRoots.size());
  AllocaInst *AI = new AllocaInst(RootListTy, 0, "gcroots", F.begin()->begin());

  // Insert the memset call after all of the allocas in the function.
  BasicBlock::iterator IP = AI;
  while (isa<AllocaInst>(IP)) ++IP;

  Constant *Zero = ConstantUInt::get(Type::UIntTy, 0);
  Constant *One  = ConstantUInt::get(Type::UIntTy, 1);

  // Get a pointer to the prev pointer.
  std::vector<Value*> Par;
  Par.push_back(Zero);
  Par.push_back(Zero);
  Value *PrevPtrPtr = new GetElementPtrInst(AI, Par, "prevptrptr", IP);

  // Load the previous pointer.
  Value *PrevPtr = new LoadInst(RootChain, "prevptr", IP);
  // Store the previous pointer into the prevptrptr
  new StoreInst(PrevPtr, PrevPtrPtr, IP);

  // Set the number of elements in this record.
  Par[1] = ConstantUInt::get(Type::UIntTy, 1);
  Value *NumEltsPtr = new GetElementPtrInst(AI, Par, "numeltsptr", IP);
  new StoreInst(ConstantUInt::get(Type::UIntTy, GCRoots.size()), NumEltsPtr,IP);

  Par[1] = ConstantUInt::get(Type::UIntTy, 2);
  Par.resize(4);

  const PointerType *PtrLocTy =
    cast<PointerType>(GCRootInt->getFunctionType()->getParamType(0));
  Constant *Null = ConstantPointerNull::get(PtrLocTy);

  // Initialize all of the gcroot records now, and eliminate them as we go.
  for (unsigned i = 0, e = GCRoots.size(); i != e; ++i) {
    // Initialize the meta-data pointer.
    Par[2] = ConstantUInt::get(Type::UIntTy, i);
    Par[3] = One;
    Value *MetaDataPtr = new GetElementPtrInst(AI, Par, "MetaDataPtr", IP);
    assert(isa<Constant>(GCRoots[i]->getOperand(2)) && "Must be a constant");
    new StoreInst(GCRoots[i]->getOperand(2), MetaDataPtr, IP);

    // Initialize the root pointer to null on entry to the function.
    Par[3] = Zero;
    Value *RootPtrPtr = new GetElementPtrInst(AI, Par, "RootEntPtr", IP);
    new StoreInst(Null, RootPtrPtr, IP);

    // Each occurrance of the llvm.gcroot intrinsic now turns into an
    // initialization of the slot with the address and a zeroing out of the
    // address specified.
    new StoreInst(Constant::getNullValue(PtrLocTy->getElementType()),
                  GCRoots[i]->getOperand(1), GCRoots[i]);
    new StoreInst(GCRoots[i]->getOperand(1), RootPtrPtr, GCRoots[i]);
    GCRoots[i]->getParent()->getInstList().erase(GCRoots[i]);
  }

  // Now that the record is all initialized, store the pointer into the global
  // pointer.
  Value *C = new CastInst(AI, PointerType::get(MainRootRecordType), "", IP);
  new StoreInst(C, RootChain, IP);

  // On exit from the function we have to remove the entry from the GC root
  // chain.  Doing this is straight-forward for return and unwind instructions:
  // just insert the appropriate copy.
  for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB)
    if (isa<UnwindInst>(BB->getTerminator()) ||
        isa<ReturnInst>(BB->getTerminator())) {
      // We could reuse the PrevPtr loaded on entry to the function, but this
      // would make the value live for the whole function, which is probably a
      // bad idea.  Just reload the value out of our stack entry.
      PrevPtr = new LoadInst(PrevPtrPtr, "prevptr", BB->getTerminator());
      new StoreInst(PrevPtr, RootChain, BB->getTerminator());
    }

  // If an exception is thrown from a callee we have to make sure to
  // unconditionally take the record off the stack.  For this reason, we turn
  // all call instructions into invoke whose cleanup pops the entry off the
  // stack.  We only insert one cleanup block, which is shared by all invokes.
  if (!NormalCalls.empty()) {
    // Create the shared cleanup block.
    BasicBlock *Cleanup = new BasicBlock("gc_cleanup", &F);
    UnwindInst *UI = new UnwindInst(Cleanup);
    PrevPtr = new LoadInst(PrevPtrPtr, "prevptr", UI);
    new StoreInst(PrevPtr, RootChain, UI);

    // Loop over all of the function calls, turning them into invokes.
    while (!NormalCalls.empty()) {
      CallInst *CI = NormalCalls.back();
      BasicBlock *CBB = CI->getParent();
      NormalCalls.pop_back();

      // Split the basic block containing the function call.
      BasicBlock *NewBB = CBB->splitBasicBlock(CI, CBB->getName()+".cont");

      // Remove the unconditional branch inserted at the end of the CBB.
      CBB->getInstList().pop_back();
      NewBB->getInstList().remove(CI);

      // Create a new invoke instruction.
      Value *II = new InvokeInst(CI->getCalledValue(), NewBB, Cleanup,
                                 std::vector<Value*>(CI->op_begin()+1,
                                                     CI->op_end()),
                                 CI->getName(), CBB);
      CI->replaceAllUsesWith(II);
      delete CI;
    }
  }

  return true;
}
