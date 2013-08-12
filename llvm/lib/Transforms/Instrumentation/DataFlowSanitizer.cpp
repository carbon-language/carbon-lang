//===-- DataFlowSanitizer.cpp - dynamic data flow analysis ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file is a part of DataFlowSanitizer, a generalised dynamic data flow
/// analysis.
///
/// Unlike other Sanitizer tools, this tool is not designed to detect a specific
/// class of bugs on its own.  Instead, it provides a generic dynamic data flow
/// analysis framework to be used by clients to help detect application-specific
/// issues within their own code.
///
/// The analysis is based on automatic propagation of data flow labels (also
/// known as taint labels) through a program as it performs computation.  Each
/// byte of application memory is backed by two bytes of shadow memory which
/// hold the label.  On Linux/x86_64, memory is laid out as follows:
///
/// +--------------------+ 0x800000000000 (top of memory)
/// | application memory |
/// +--------------------+ 0x700000008000 (kAppAddr)
/// |                    |
/// |       unused       |
/// |                    |
/// +--------------------+ 0x200200000000 (kUnusedAddr)
/// |    union table     |
/// +--------------------+ 0x200000000000 (kUnionTableAddr)
/// |   shadow memory    |
/// +--------------------+ 0x000000010000 (kShadowAddr)
/// | reserved by kernel |
/// +--------------------+ 0x000000000000
///
/// To derive a shadow memory address from an application memory address,
/// bits 44-46 are cleared to bring the address into the range
/// [0x000000008000,0x100000000000).  Then the address is shifted left by 1 to
/// account for the double byte representation of shadow labels and move the
/// address into the shadow memory range.  See the function
/// DataFlowSanitizer::getShadowAddress below.
///
/// For more information, please refer to the design document:
/// http://clang.llvm.org/docs/DataFlowSanitizerDesign.html

#include "llvm/Transforms/Instrumentation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/InstVisitor.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/SpecialCaseList.h"
#include <iterator>

using namespace llvm;

// The -dfsan-preserve-alignment flag controls whether this pass assumes that
// alignment requirements provided by the input IR are correct.  For example,
// if the input IR contains a load with alignment 8, this flag will cause
// the shadow load to have alignment 16.  This flag is disabled by default as
// we have unfortunately encountered too much code (including Clang itself;
// see PR14291) which performs misaligned access.
static cl::opt<bool> ClPreserveAlignment(
    "dfsan-preserve-alignment",
    cl::desc("respect alignment requirements provided by input IR"), cl::Hidden,
    cl::init(false));

// The greylist file controls how shadow parameters are passed.
// The program acts as though every function in the greylist is passed
// parameters with zero shadow and that its return value also has zero shadow.
// This avoids the use of TLS or extra function parameters to pass shadow state
// and essentially makes the function conform to the "native" (i.e. unsanitized)
// ABI.
static cl::opt<std::string> ClGreylistFile(
    "dfsan-greylist",
    cl::desc("File containing the list of functions with a native ABI"),
    cl::Hidden);

static cl::opt<bool> ClArgsABI(
    "dfsan-args-abi",
    cl::desc("Use the argument ABI rather than the TLS ABI"),
    cl::Hidden);

namespace {

class DataFlowSanitizer : public ModulePass {
  friend struct DFSanFunction;
  friend class DFSanVisitor;

  enum {
    ShadowWidth = 16
  };

  enum InstrumentedABI {
    IA_None,
    IA_MemOnly,
    IA_Args,
    IA_TLS
  };

  DataLayout *DL;
  Module *Mod;
  LLVMContext *Ctx;
  IntegerType *ShadowTy;
  PointerType *ShadowPtrTy;
  IntegerType *IntptrTy;
  ConstantInt *ZeroShadow;
  ConstantInt *ShadowPtrMask;
  ConstantInt *ShadowPtrMul;
  Constant *ArgTLS;
  Constant *RetvalTLS;
  void *(*GetArgTLSPtr)();
  void *(*GetRetvalTLSPtr)();
  Constant *GetArgTLS;
  Constant *GetRetvalTLS;
  FunctionType *DFSanUnionFnTy;
  FunctionType *DFSanUnionLoadFnTy;
  Constant *DFSanUnionFn;
  Constant *DFSanUnionLoadFn;
  MDNode *ColdCallWeights;
  OwningPtr<SpecialCaseList> Greylist;
  DenseMap<Value *, Function *> UnwrappedFnMap;

  Value *getShadowAddress(Value *Addr, Instruction *Pos);
  Value *combineShadows(Value *V1, Value *V2, Instruction *Pos);
  FunctionType *getInstrumentedFunctionType(FunctionType *T);
  InstrumentedABI getInstrumentedABI(Function *F);
  InstrumentedABI getDefaultInstrumentedABI();

public:
  DataFlowSanitizer(void *(*getArgTLS)() = 0, void *(*getRetValTLS)() = 0);
  static char ID;
  bool doInitialization(Module &M);
  bool runOnModule(Module &M);
};

struct DFSanFunction {
  DataFlowSanitizer &DFS;
  Function *F;
  DataFlowSanitizer::InstrumentedABI IA;
  Value *ArgTLSPtr;
  Value *RetvalTLSPtr;
  DenseMap<Value *, Value *> ValShadowMap;
  DenseMap<AllocaInst *, AllocaInst *> AllocaShadowMap;
  std::vector<std::pair<PHINode *, PHINode *> > PHIFixups;
  DenseSet<Instruction *> SkipInsts;

  DFSanFunction(DataFlowSanitizer &DFS, Function *F)
      : DFS(DFS), F(F), IA(DFS.getInstrumentedABI(F)), ArgTLSPtr(0),
        RetvalTLSPtr(0) {}
  Value *getArgTLSPtr();
  Value *getArgTLS(unsigned Index, Instruction *Pos);
  Value *getRetvalTLS();
  Value *getShadow(Value *V);
  void setShadow(Instruction *I, Value *Shadow);
  Value *combineOperandShadows(Instruction *Inst);
  Value *loadShadow(Value *ShadowAddr, uint64_t Size, uint64_t Align,
                    Instruction *Pos);
  void storeShadow(Value *Addr, uint64_t Size, uint64_t Align, Value *Shadow,
                   Instruction *Pos);
};

class DFSanVisitor : public InstVisitor<DFSanVisitor> {
public:
  DFSanFunction &DFSF;
  DFSanVisitor(DFSanFunction &DFSF) : DFSF(DFSF) {}

  void visitOperandShadowInst(Instruction &I);

  void visitBinaryOperator(BinaryOperator &BO);
  void visitCastInst(CastInst &CI);
  void visitCmpInst(CmpInst &CI);
  void visitGetElementPtrInst(GetElementPtrInst &GEPI);
  void visitLoadInst(LoadInst &LI);
  void visitStoreInst(StoreInst &SI);
  void visitReturnInst(ReturnInst &RI);
  void visitCallSite(CallSite CS);
  void visitPHINode(PHINode &PN);
  void visitExtractElementInst(ExtractElementInst &I);
  void visitInsertElementInst(InsertElementInst &I);
  void visitShuffleVectorInst(ShuffleVectorInst &I);
  void visitExtractValueInst(ExtractValueInst &I);
  void visitInsertValueInst(InsertValueInst &I);
  void visitAllocaInst(AllocaInst &I);
  void visitSelectInst(SelectInst &I);
  void visitMemTransferInst(MemTransferInst &I);
};

}

char DataFlowSanitizer::ID;
INITIALIZE_PASS(DataFlowSanitizer, "dfsan",
                "DataFlowSanitizer: dynamic data flow analysis.", false, false)

ModulePass *llvm::createDataFlowSanitizerPass(void *(*getArgTLS)(),
                                              void *(*getRetValTLS)()) {
  return new DataFlowSanitizer(getArgTLS, getRetValTLS);
}

DataFlowSanitizer::DataFlowSanitizer(void *(*getArgTLS)(),
                                     void *(*getRetValTLS)())
    : ModulePass(ID), GetArgTLSPtr(getArgTLS), GetRetvalTLSPtr(getRetValTLS),
      Greylist(SpecialCaseList::createOrDie(ClGreylistFile)) {}

FunctionType *DataFlowSanitizer::getInstrumentedFunctionType(FunctionType *T) {
  llvm::SmallVector<Type *, 4> ArgTypes;
  std::copy(T->param_begin(), T->param_end(), std::back_inserter(ArgTypes));
  for (unsigned i = 0, e = T->getNumParams(); i != e; ++i)
    ArgTypes.push_back(ShadowTy);
  if (T->isVarArg())
    ArgTypes.push_back(ShadowPtrTy);
  Type *RetType = T->getReturnType();
  if (!RetType->isVoidTy())
    RetType = StructType::get(RetType, ShadowTy, (Type *)0);
  return FunctionType::get(RetType, ArgTypes, T->isVarArg());
}

bool DataFlowSanitizer::doInitialization(Module &M) {
  DL = getAnalysisIfAvailable<DataLayout>();
  if (!DL)
    return false;

  Mod = &M;
  Ctx = &M.getContext();
  ShadowTy = IntegerType::get(*Ctx, ShadowWidth);
  ShadowPtrTy = PointerType::getUnqual(ShadowTy);
  IntptrTy = DL->getIntPtrType(*Ctx);
  ZeroShadow = ConstantInt::getSigned(ShadowTy, 0);
  ShadowPtrMask = ConstantInt::getSigned(IntptrTy, ~0x700000000000LL);
  ShadowPtrMul = ConstantInt::getSigned(IntptrTy, ShadowWidth / 8);

  Type *DFSanUnionArgs[2] = { ShadowTy, ShadowTy };
  DFSanUnionFnTy =
      FunctionType::get(ShadowTy, DFSanUnionArgs, /*isVarArg=*/ false);
  Type *DFSanUnionLoadArgs[2] = { ShadowPtrTy, IntptrTy };
  DFSanUnionLoadFnTy =
      FunctionType::get(ShadowTy, DFSanUnionLoadArgs, /*isVarArg=*/ false);

  if (GetArgTLSPtr) {
    Type *ArgTLSTy = ArrayType::get(ShadowTy, 64);
    ArgTLS = 0;
    GetArgTLS = ConstantExpr::getIntToPtr(
        ConstantInt::get(IntptrTy, uintptr_t(GetArgTLSPtr)),
        PointerType::getUnqual(
            FunctionType::get(PointerType::getUnqual(ArgTLSTy), (Type *)0)));
  }
  if (GetRetvalTLSPtr) {
    RetvalTLS = 0;
    GetRetvalTLS = ConstantExpr::getIntToPtr(
        ConstantInt::get(IntptrTy, uintptr_t(GetRetvalTLSPtr)),
        PointerType::getUnqual(
            FunctionType::get(PointerType::getUnqual(ShadowTy), (Type *)0)));
  }

  ColdCallWeights = MDBuilder(*Ctx).createBranchWeights(1, 1000);
  return true;
}

DataFlowSanitizer::InstrumentedABI
DataFlowSanitizer::getInstrumentedABI(Function *F) {
  if (Greylist->isIn(*F))
    return IA_MemOnly;
  else
    return getDefaultInstrumentedABI();
}

DataFlowSanitizer::InstrumentedABI
DataFlowSanitizer::getDefaultInstrumentedABI() {
  return ClArgsABI ? IA_Args : IA_TLS;
}

bool DataFlowSanitizer::runOnModule(Module &M) {
  if (!DL)
    return false;

  if (!GetArgTLSPtr) {
    Type *ArgTLSTy = ArrayType::get(ShadowTy, 64);
    ArgTLS = Mod->getOrInsertGlobal("__dfsan_arg_tls", ArgTLSTy);
    if (GlobalVariable *G = dyn_cast<GlobalVariable>(ArgTLS))
      G->setThreadLocalMode(GlobalVariable::InitialExecTLSModel);
  }
  if (!GetRetvalTLSPtr) {
    RetvalTLS = Mod->getOrInsertGlobal("__dfsan_retval_tls", ShadowTy);
    if (GlobalVariable *G = dyn_cast<GlobalVariable>(RetvalTLS))
      G->setThreadLocalMode(GlobalVariable::InitialExecTLSModel);
  }

  DFSanUnionFn = Mod->getOrInsertFunction("__dfsan_union", DFSanUnionFnTy);
  if (Function *F = dyn_cast<Function>(DFSanUnionFn)) {
    F->addAttribute(AttributeSet::FunctionIndex, Attribute::ReadNone);
    F->addAttribute(AttributeSet::ReturnIndex, Attribute::ZExt);
    F->addAttribute(1, Attribute::ZExt);
    F->addAttribute(2, Attribute::ZExt);
  }
  DFSanUnionLoadFn =
      Mod->getOrInsertFunction("__dfsan_union_load", DFSanUnionLoadFnTy);
  if (Function *F = dyn_cast<Function>(DFSanUnionLoadFn)) {
    F->addAttribute(AttributeSet::ReturnIndex, Attribute::ZExt);
  }

  std::vector<Function *> FnsToInstrument;
  for (Module::iterator i = M.begin(), e = M.end(); i != e; ++i) {
    if (!i->isIntrinsic() && i != DFSanUnionFn && i != DFSanUnionLoadFn)
      FnsToInstrument.push_back(&*i);
  }

  // First, change the ABI of every function in the module.  Greylisted
  // functions keep their original ABI and get a wrapper function.
  for (std::vector<Function *>::iterator i = FnsToInstrument.begin(),
                                         e = FnsToInstrument.end();
       i != e; ++i) {
    Function &F = **i;

    FunctionType *FT = F.getFunctionType();
    FunctionType *NewFT = getInstrumentedFunctionType(FT);
    // If the function types are the same (i.e. void()), we don't need to do
    // anything here.
    if (FT != NewFT) {
      switch (getInstrumentedABI(&F)) {
      case IA_Args: {
        Function *NewF = Function::Create(NewFT, F.getLinkage(), "", &M);
        NewF->setCallingConv(F.getCallingConv());
        NewF->setAttributes(F.getAttributes().removeAttributes(
            *Ctx, AttributeSet::ReturnIndex,
            AttributeFuncs::typeIncompatible(NewFT->getReturnType(),
                                             AttributeSet::ReturnIndex)));
        for (Function::arg_iterator FArg = F.arg_begin(),
                                    NewFArg = NewF->arg_begin(),
                                    FArgEnd = F.arg_end();
             FArg != FArgEnd; ++FArg, ++NewFArg) {
          FArg->replaceAllUsesWith(NewFArg);
        }
        NewF->getBasicBlockList().splice(NewF->begin(), F.getBasicBlockList());

        for (Function::use_iterator ui = F.use_begin(), ue = F.use_end();
             ui != ue;) {
          BlockAddress *BA = dyn_cast<BlockAddress>(ui.getUse().getUser());
          ++ui;
          if (BA) {
            BA->replaceAllUsesWith(
                BlockAddress::get(NewF, BA->getBasicBlock()));
            delete BA;
          }
        }
        F.replaceAllUsesWith(
            ConstantExpr::getBitCast(NewF, PointerType::getUnqual(FT)));
        NewF->takeName(&F);
        F.eraseFromParent();
        *i = NewF;
        break;
      }
      case IA_MemOnly: {
        assert(!FT->isVarArg() && "varargs not handled here yet");
        assert(getDefaultInstrumentedABI() == IA_Args);
        Function *NewF =
            Function::Create(NewFT, GlobalValue::LinkOnceODRLinkage,
                             std::string("dfsw$") + F.getName(), &M);
        NewF->setCallingConv(F.getCallingConv());
        NewF->setAttributes(F.getAttributes());

        BasicBlock *BB = BasicBlock::Create(*Ctx, "entry", NewF);
        std::vector<Value *> Args;
        unsigned n = FT->getNumParams();
        for (Function::arg_iterator i = NewF->arg_begin(); n != 0; ++i, --n)
          Args.push_back(&*i);
        CallInst *CI = CallInst::Create(&F, Args, "", BB);
        if (FT->getReturnType()->isVoidTy())
          ReturnInst::Create(*Ctx, BB);
        else {
          Value *InsVal = InsertValueInst::Create(
              UndefValue::get(NewFT->getReturnType()), CI, 0, "", BB);
          Value *InsShadow =
              InsertValueInst::Create(InsVal, ZeroShadow, 1, "", BB);
          ReturnInst::Create(*Ctx, InsShadow, BB);
        }

        Value *WrappedFnCst =
            ConstantExpr::getBitCast(NewF, PointerType::getUnqual(FT));
        F.replaceAllUsesWith(WrappedFnCst);
        UnwrappedFnMap[WrappedFnCst] = &F;
        break;
      }
      default:
        break;
      }
    }
  }

  for (std::vector<Function *>::iterator i = FnsToInstrument.begin(),
                                         e = FnsToInstrument.end();
       i != e; ++i) {
    if ((*i)->isDeclaration())
      continue;

    removeUnreachableBlocks(**i);

    DFSanFunction DFSF(*this, *i);

    // DFSanVisitor may create new basic blocks, which confuses df_iterator.
    // Build a copy of the list before iterating over it.
    llvm::SmallVector<BasicBlock *, 4> BBList;
    std::copy(df_begin(&(*i)->getEntryBlock()), df_end(&(*i)->getEntryBlock()),
              std::back_inserter(BBList));

    for (llvm::SmallVector<BasicBlock *, 4>::iterator i = BBList.begin(),
                                                      e = BBList.end();
         i != e; ++i) {
      Instruction *Inst = &(*i)->front();
      while (1) {
        // DFSanVisitor may split the current basic block, changing the current
        // instruction's next pointer and moving the next instruction to the
        // tail block from which we should continue.
        Instruction *Next = Inst->getNextNode();
        if (!DFSF.SkipInsts.count(Inst))
          DFSanVisitor(DFSF).visit(Inst);
        if (isa<TerminatorInst>(Inst))
          break;
        Inst = Next;
      }
    }

    for (std::vector<std::pair<PHINode *, PHINode *> >::iterator
             i = DFSF.PHIFixups.begin(),
             e = DFSF.PHIFixups.end();
         i != e; ++i) {
      for (unsigned val = 0, n = i->first->getNumIncomingValues(); val != n;
           ++val) {
        i->second->setIncomingValue(
            val, DFSF.getShadow(i->first->getIncomingValue(val)));
      }
    }
  }

  return false;
}

Value *DFSanFunction::getArgTLSPtr() {
  if (ArgTLSPtr)
    return ArgTLSPtr;
  if (DFS.ArgTLS)
    return ArgTLSPtr = DFS.ArgTLS;

  IRBuilder<> IRB(F->getEntryBlock().begin());
  return ArgTLSPtr = IRB.CreateCall(DFS.GetArgTLS);
}

Value *DFSanFunction::getRetvalTLS() {
  if (RetvalTLSPtr)
    return RetvalTLSPtr;
  if (DFS.RetvalTLS)
    return RetvalTLSPtr = DFS.RetvalTLS;

  IRBuilder<> IRB(F->getEntryBlock().begin());
  return RetvalTLSPtr = IRB.CreateCall(DFS.GetRetvalTLS);
}

Value *DFSanFunction::getArgTLS(unsigned Idx, Instruction *Pos) {
  IRBuilder<> IRB(Pos);
  return IRB.CreateConstGEP2_64(getArgTLSPtr(), 0, Idx);
}

Value *DFSanFunction::getShadow(Value *V) {
  if (!isa<Argument>(V) && !isa<Instruction>(V))
    return DFS.ZeroShadow;
  Value *&Shadow = ValShadowMap[V];
  if (!Shadow) {
    if (Argument *A = dyn_cast<Argument>(V)) {
      switch (IA) {
      case DataFlowSanitizer::IA_TLS: {
        Value *ArgTLSPtr = getArgTLSPtr();
        Instruction *ArgTLSPos =
            DFS.ArgTLS ? &*F->getEntryBlock().begin()
                       : cast<Instruction>(ArgTLSPtr)->getNextNode();
        IRBuilder<> IRB(ArgTLSPos);
        Shadow = IRB.CreateLoad(getArgTLS(A->getArgNo(), ArgTLSPos));
        break;
      }
      case DataFlowSanitizer::IA_Args: {
        unsigned ArgIdx = A->getArgNo() + F->getArgumentList().size() / 2;
        Function::arg_iterator i = F->arg_begin();
        while (ArgIdx--)
          ++i;
        Shadow = i;
        break;
      }
      default:
        Shadow = DFS.ZeroShadow;
        break;
      }
    } else {
      Shadow = DFS.ZeroShadow;
    }
  }
  return Shadow;
}

void DFSanFunction::setShadow(Instruction *I, Value *Shadow) {
  assert(!ValShadowMap.count(I));
  assert(Shadow->getType() == DFS.ShadowTy);
  ValShadowMap[I] = Shadow;
}

Value *DataFlowSanitizer::getShadowAddress(Value *Addr, Instruction *Pos) {
  assert(Addr != RetvalTLS && "Reinstrumenting?");
  IRBuilder<> IRB(Pos);
  return IRB.CreateIntToPtr(
      IRB.CreateMul(
          IRB.CreateAnd(IRB.CreatePtrToInt(Addr, IntptrTy), ShadowPtrMask),
          ShadowPtrMul),
      ShadowPtrTy);
}

// Generates IR to compute the union of the two given shadows, inserting it
// before Pos.  Returns the computed union Value.
Value *DataFlowSanitizer::combineShadows(Value *V1, Value *V2,
                                         Instruction *Pos) {
  if (V1 == ZeroShadow)
    return V2;
  if (V2 == ZeroShadow)
    return V1;
  if (V1 == V2)
    return V1;
  IRBuilder<> IRB(Pos);
  BasicBlock *Head = Pos->getParent();
  Value *Ne = IRB.CreateICmpNE(V1, V2);
  Instruction *NeInst = dyn_cast<Instruction>(Ne);
  if (NeInst) {
    BranchInst *BI = cast<BranchInst>(SplitBlockAndInsertIfThen(
        NeInst, /*Unreachable=*/ false, ColdCallWeights));
    IRBuilder<> ThenIRB(BI);
    CallInst *Call = ThenIRB.CreateCall2(DFSanUnionFn, V1, V2);
    Call->addAttribute(AttributeSet::ReturnIndex, Attribute::ZExt);
    Call->addAttribute(1, Attribute::ZExt);
    Call->addAttribute(2, Attribute::ZExt);

    BasicBlock *Tail = BI->getSuccessor(0);
    PHINode *Phi = PHINode::Create(ShadowTy, 2, "", Tail->begin());
    Phi->addIncoming(Call, Call->getParent());
    Phi->addIncoming(ZeroShadow, Head);
    Pos = Phi;
    return Phi;
  } else {
    assert(0 && "todo");
    return 0;
  }
}

// A convenience function which folds the shadows of each of the operands
// of the provided instruction Inst, inserting the IR before Inst.  Returns
// the computed union Value.
Value *DFSanFunction::combineOperandShadows(Instruction *Inst) {
  if (Inst->getNumOperands() == 0)
    return DFS.ZeroShadow;

  Value *Shadow = getShadow(Inst->getOperand(0));
  for (unsigned i = 1, n = Inst->getNumOperands(); i != n; ++i) {
    Shadow = DFS.combineShadows(Shadow, getShadow(Inst->getOperand(i)), Inst);
  }
  return Shadow;
}

void DFSanVisitor::visitOperandShadowInst(Instruction &I) {
  Value *CombinedShadow = DFSF.combineOperandShadows(&I);
  DFSF.setShadow(&I, CombinedShadow);
}

// Generates IR to load shadow corresponding to bytes [Addr, Addr+Size), where
// Addr has alignment Align, and take the union of each of those shadows.
Value *DFSanFunction::loadShadow(Value *Addr, uint64_t Size, uint64_t Align,
                                 Instruction *Pos) {
  if (AllocaInst *AI = dyn_cast<AllocaInst>(Addr)) {
    llvm::DenseMap<AllocaInst *, AllocaInst *>::iterator i =
        AllocaShadowMap.find(AI);
    if (i != AllocaShadowMap.end()) {
      IRBuilder<> IRB(Pos);
      return IRB.CreateLoad(i->second);
    }
  }

  uint64_t ShadowAlign = Align * DFS.ShadowWidth / 8;
  SmallVector<Value *, 2> Objs;
  GetUnderlyingObjects(Addr, Objs, DFS.DL);
  bool AllConstants = true;
  for (SmallVector<Value *, 2>::iterator i = Objs.begin(), e = Objs.end();
       i != e; ++i) {
    if (isa<Function>(*i) || isa<BlockAddress>(*i))
      continue;
    if (isa<GlobalVariable>(*i) && cast<GlobalVariable>(*i)->isConstant())
      continue;

    AllConstants = false;
    break;
  }
  if (AllConstants)
    return DFS.ZeroShadow;

  Value *ShadowAddr = DFS.getShadowAddress(Addr, Pos);
  switch (Size) {
  case 0:
    return DFS.ZeroShadow;
  case 1: {
    LoadInst *LI = new LoadInst(ShadowAddr, "", Pos);
    LI->setAlignment(ShadowAlign);
    return LI;
  }
  case 2: {
    IRBuilder<> IRB(Pos);
    Value *ShadowAddr1 =
        IRB.CreateGEP(ShadowAddr, ConstantInt::get(DFS.IntptrTy, 1));
    return DFS.combineShadows(IRB.CreateAlignedLoad(ShadowAddr, ShadowAlign),
                              IRB.CreateAlignedLoad(ShadowAddr1, ShadowAlign),
                              Pos);
  }
  }
  if (Size % (64 / DFS.ShadowWidth) == 0) {
    // Fast path for the common case where each byte has identical shadow: load
    // shadow 64 bits at a time, fall out to a __dfsan_union_load call if any
    // shadow is non-equal.
    BasicBlock *FallbackBB = BasicBlock::Create(*DFS.Ctx, "", F);
    IRBuilder<> FallbackIRB(FallbackBB);
    CallInst *FallbackCall = FallbackIRB.CreateCall2(
        DFS.DFSanUnionLoadFn, ShadowAddr, ConstantInt::get(DFS.IntptrTy, Size));
    FallbackCall->addAttribute(AttributeSet::ReturnIndex, Attribute::ZExt);

    // Compare each of the shadows stored in the loaded 64 bits to each other,
    // by computing (WideShadow rotl ShadowWidth) == WideShadow.
    IRBuilder<> IRB(Pos);
    Value *WideAddr =
        IRB.CreateBitCast(ShadowAddr, Type::getInt64PtrTy(*DFS.Ctx));
    Value *WideShadow = IRB.CreateAlignedLoad(WideAddr, ShadowAlign);
    Value *TruncShadow = IRB.CreateTrunc(WideShadow, DFS.ShadowTy);
    Value *ShlShadow = IRB.CreateShl(WideShadow, DFS.ShadowWidth);
    Value *ShrShadow = IRB.CreateLShr(WideShadow, 64 - DFS.ShadowWidth);
    Value *RotShadow = IRB.CreateOr(ShlShadow, ShrShadow);
    Value *ShadowsEq = IRB.CreateICmpEQ(WideShadow, RotShadow);

    BasicBlock *Head = Pos->getParent();
    BasicBlock *Tail = Head->splitBasicBlock(Pos);
    // In the following code LastBr will refer to the previous basic block's
    // conditional branch instruction, whose true successor is fixed up to point
    // to the next block during the loop below or to the tail after the final
    // iteration.
    BranchInst *LastBr = BranchInst::Create(FallbackBB, FallbackBB, ShadowsEq);
    ReplaceInstWithInst(Head->getTerminator(), LastBr);

    for (uint64_t Ofs = 64 / DFS.ShadowWidth; Ofs != Size;
         Ofs += 64 / DFS.ShadowWidth) {
      BasicBlock *NextBB = BasicBlock::Create(*DFS.Ctx, "", F);
      IRBuilder<> NextIRB(NextBB);
      WideAddr = NextIRB.CreateGEP(WideAddr, ConstantInt::get(DFS.IntptrTy, 1));
      Value *NextWideShadow = NextIRB.CreateAlignedLoad(WideAddr, ShadowAlign);
      ShadowsEq = NextIRB.CreateICmpEQ(WideShadow, NextWideShadow);
      LastBr->setSuccessor(0, NextBB);
      LastBr = NextIRB.CreateCondBr(ShadowsEq, FallbackBB, FallbackBB);
    }

    LastBr->setSuccessor(0, Tail);
    FallbackIRB.CreateBr(Tail);
    PHINode *Shadow = PHINode::Create(DFS.ShadowTy, 2, "", &Tail->front());
    Shadow->addIncoming(FallbackCall, FallbackBB);
    Shadow->addIncoming(TruncShadow, LastBr->getParent());
    return Shadow;
  }

  IRBuilder<> IRB(Pos);
  CallInst *FallbackCall = IRB.CreateCall2(
      DFS.DFSanUnionLoadFn, ShadowAddr, ConstantInt::get(DFS.IntptrTy, Size));
  FallbackCall->addAttribute(AttributeSet::ReturnIndex, Attribute::ZExt);
  return FallbackCall;
}

void DFSanVisitor::visitLoadInst(LoadInst &LI) {
  uint64_t Size = DFSF.DFS.DL->getTypeStoreSize(LI.getType());
  uint64_t Align;
  if (ClPreserveAlignment) {
    Align = LI.getAlignment();
    if (Align == 0)
      Align = DFSF.DFS.DL->getABITypeAlignment(LI.getType());
  } else {
    Align = 1;
  }
  IRBuilder<> IRB(&LI);
  Value *LoadedShadow =
      DFSF.loadShadow(LI.getPointerOperand(), Size, Align, &LI);
  Value *PtrShadow = DFSF.getShadow(LI.getPointerOperand());
  DFSF.setShadow(&LI, DFSF.DFS.combineShadows(LoadedShadow, PtrShadow, &LI));
}

void DFSanFunction::storeShadow(Value *Addr, uint64_t Size, uint64_t Align,
                                Value *Shadow, Instruction *Pos) {
  if (AllocaInst *AI = dyn_cast<AllocaInst>(Addr)) {
    llvm::DenseMap<AllocaInst *, AllocaInst *>::iterator i =
        AllocaShadowMap.find(AI);
    if (i != AllocaShadowMap.end()) {
      IRBuilder<> IRB(Pos);
      IRB.CreateStore(Shadow, i->second);
      return;
    }
  }

  uint64_t ShadowAlign = Align * DFS.ShadowWidth / 8;
  IRBuilder<> IRB(Pos);
  Value *ShadowAddr = DFS.getShadowAddress(Addr, Pos);
  if (Shadow == DFS.ZeroShadow) {
    IntegerType *ShadowTy = IntegerType::get(*DFS.Ctx, Size * DFS.ShadowWidth);
    Value *ExtZeroShadow = ConstantInt::get(ShadowTy, 0);
    Value *ExtShadowAddr =
        IRB.CreateBitCast(ShadowAddr, PointerType::getUnqual(ShadowTy));
    IRB.CreateAlignedStore(ExtZeroShadow, ExtShadowAddr, ShadowAlign);
    return;
  }

  const unsigned ShadowVecSize = 128 / DFS.ShadowWidth;
  uint64_t Offset = 0;
  if (Size >= ShadowVecSize) {
    VectorType *ShadowVecTy = VectorType::get(DFS.ShadowTy, ShadowVecSize);
    Value *ShadowVec = UndefValue::get(ShadowVecTy);
    for (unsigned i = 0; i != ShadowVecSize; ++i) {
      ShadowVec = IRB.CreateInsertElement(
          ShadowVec, Shadow, ConstantInt::get(Type::getInt32Ty(*DFS.Ctx), i));
    }
    Value *ShadowVecAddr =
        IRB.CreateBitCast(ShadowAddr, PointerType::getUnqual(ShadowVecTy));
    do {
      Value *CurShadowVecAddr = IRB.CreateConstGEP1_32(ShadowVecAddr, Offset);
      IRB.CreateAlignedStore(ShadowVec, CurShadowVecAddr, ShadowAlign);
      Size -= ShadowVecSize;
      ++Offset;
    } while (Size >= ShadowVecSize);
    Offset *= ShadowVecSize;
  }
  while (Size > 0) {
    Value *CurShadowAddr = IRB.CreateConstGEP1_32(ShadowAddr, Offset);
    IRB.CreateAlignedStore(Shadow, CurShadowAddr, ShadowAlign);
    --Size;
    ++Offset;
  }
}

void DFSanVisitor::visitStoreInst(StoreInst &SI) {
  uint64_t Size =
      DFSF.DFS.DL->getTypeStoreSize(SI.getValueOperand()->getType());
  uint64_t Align;
  if (ClPreserveAlignment) {
    Align = SI.getAlignment();
    if (Align == 0)
      Align = DFSF.DFS.DL->getABITypeAlignment(SI.getValueOperand()->getType());
  } else {
    Align = 1;
  }
  DFSF.storeShadow(SI.getPointerOperand(), Size, Align,
                   DFSF.getShadow(SI.getValueOperand()), &SI);
}

void DFSanVisitor::visitBinaryOperator(BinaryOperator &BO) {
  visitOperandShadowInst(BO);
}

void DFSanVisitor::visitCastInst(CastInst &CI) { visitOperandShadowInst(CI); }

void DFSanVisitor::visitCmpInst(CmpInst &CI) { visitOperandShadowInst(CI); }

void DFSanVisitor::visitGetElementPtrInst(GetElementPtrInst &GEPI) {
  visitOperandShadowInst(GEPI);
}

void DFSanVisitor::visitExtractElementInst(ExtractElementInst &I) {
  visitOperandShadowInst(I);
}

void DFSanVisitor::visitInsertElementInst(InsertElementInst &I) {
  visitOperandShadowInst(I);
}

void DFSanVisitor::visitShuffleVectorInst(ShuffleVectorInst &I) {
  visitOperandShadowInst(I);
}

void DFSanVisitor::visitExtractValueInst(ExtractValueInst &I) {
  visitOperandShadowInst(I);
}

void DFSanVisitor::visitInsertValueInst(InsertValueInst &I) {
  visitOperandShadowInst(I);
}

void DFSanVisitor::visitAllocaInst(AllocaInst &I) {
  bool AllLoadsStores = true;
  for (Instruction::use_iterator i = I.use_begin(), e = I.use_end(); i != e;
       ++i) {
    if (isa<LoadInst>(*i))
      continue;

    if (StoreInst *SI = dyn_cast<StoreInst>(*i)) {
      if (SI->getPointerOperand() == &I)
        continue;
    }

    AllLoadsStores = false;
    break;
  }
  if (AllLoadsStores) {
    IRBuilder<> IRB(&I);
    DFSF.AllocaShadowMap[&I] = IRB.CreateAlloca(DFSF.DFS.ShadowTy);
  }
  DFSF.setShadow(&I, DFSF.DFS.ZeroShadow);
}

void DFSanVisitor::visitSelectInst(SelectInst &I) {
  Value *CondShadow = DFSF.getShadow(I.getCondition());
  Value *TrueShadow = DFSF.getShadow(I.getTrueValue());
  Value *FalseShadow = DFSF.getShadow(I.getFalseValue());

  if (isa<VectorType>(I.getCondition()->getType())) {
    DFSF.setShadow(
        &I, DFSF.DFS.combineShadows(
                CondShadow,
                DFSF.DFS.combineShadows(TrueShadow, FalseShadow, &I), &I));
  } else {
    Value *ShadowSel;
    if (TrueShadow == FalseShadow) {
      ShadowSel = TrueShadow;
    } else {
      ShadowSel =
          SelectInst::Create(I.getCondition(), TrueShadow, FalseShadow, "", &I);
    }
    DFSF.setShadow(&I, DFSF.DFS.combineShadows(CondShadow, ShadowSel, &I));
  }
}

void DFSanVisitor::visitMemTransferInst(MemTransferInst &I) {
  IRBuilder<> IRB(&I);
  Value *DestShadow = DFSF.DFS.getShadowAddress(I.getDest(), &I);
  Value *SrcShadow = DFSF.DFS.getShadowAddress(I.getSource(), &I);
  Value *LenShadow = IRB.CreateMul(
      I.getLength(),
      ConstantInt::get(I.getLength()->getType(), DFSF.DFS.ShadowWidth / 8));
  Value *AlignShadow;
  if (ClPreserveAlignment) {
    AlignShadow = IRB.CreateMul(I.getAlignmentCst(),
                                ConstantInt::get(I.getAlignmentCst()->getType(),
                                                 DFSF.DFS.ShadowWidth / 8));
  } else {
    AlignShadow = ConstantInt::get(I.getAlignmentCst()->getType(),
                                   DFSF.DFS.ShadowWidth / 8);
  }
  Type *Int8Ptr = Type::getInt8PtrTy(*DFSF.DFS.Ctx);
  DestShadow = IRB.CreateBitCast(DestShadow, Int8Ptr);
  SrcShadow = IRB.CreateBitCast(SrcShadow, Int8Ptr);
  IRB.CreateCall5(I.getCalledValue(), DestShadow, SrcShadow, LenShadow,
                  AlignShadow, I.getVolatileCst());
}

void DFSanVisitor::visitReturnInst(ReturnInst &RI) {
  if (RI.getReturnValue()) {
    switch (DFSF.IA) {
    case DataFlowSanitizer::IA_TLS: {
      Value *S = DFSF.getShadow(RI.getReturnValue());
      IRBuilder<> IRB(&RI);
      IRB.CreateStore(S, DFSF.getRetvalTLS());
      break;
    }
    case DataFlowSanitizer::IA_Args: {
      IRBuilder<> IRB(&RI);
      Type *RT = DFSF.F->getFunctionType()->getReturnType();
      Value *InsVal =
          IRB.CreateInsertValue(UndefValue::get(RT), RI.getReturnValue(), 0);
      Value *InsShadow =
          IRB.CreateInsertValue(InsVal, DFSF.getShadow(RI.getReturnValue()), 1);
      RI.setOperand(0, InsShadow);
      break;
    }
    default:
      break;
    }
  }
}

void DFSanVisitor::visitCallSite(CallSite CS) {
  Function *F = CS.getCalledFunction();
  if ((F && F->isIntrinsic()) || isa<InlineAsm>(CS.getCalledValue())) {
    visitOperandShadowInst(*CS.getInstruction());
    return;
  }

  DenseMap<Value *, Function *>::iterator i =
      DFSF.DFS.UnwrappedFnMap.find(CS.getCalledValue());
  if (i != DFSF.DFS.UnwrappedFnMap.end()) {
    CS.setCalledFunction(i->second);
    DFSF.setShadow(CS.getInstruction(), DFSF.DFS.ZeroShadow);
    return;
  }

  IRBuilder<> IRB(CS.getInstruction());

  FunctionType *FT = cast<FunctionType>(
      CS.getCalledValue()->getType()->getPointerElementType());
  if (DFSF.DFS.getDefaultInstrumentedABI() == DataFlowSanitizer::IA_TLS) {
    for (unsigned i = 0, n = FT->getNumParams(); i != n; ++i) {
      IRB.CreateStore(DFSF.getShadow(CS.getArgument(i)),
                      DFSF.getArgTLS(i, CS.getInstruction()));
    }
  }

  Instruction *Next = 0;
  if (!CS.getType()->isVoidTy()) {
    if (InvokeInst *II = dyn_cast<InvokeInst>(CS.getInstruction())) {
      if (II->getNormalDest()->getSinglePredecessor()) {
        Next = II->getNormalDest()->begin();
      } else {
        BasicBlock *NewBB =
            SplitEdge(II->getParent(), II->getNormalDest(), &DFSF.DFS);
        Next = NewBB->begin();
      }
    } else {
      Next = CS->getNextNode();
    }

    if (DFSF.DFS.getDefaultInstrumentedABI() == DataFlowSanitizer::IA_TLS) {
      IRBuilder<> NextIRB(Next);
      LoadInst *LI = NextIRB.CreateLoad(DFSF.getRetvalTLS());
      DFSF.SkipInsts.insert(LI);
      DFSF.setShadow(CS.getInstruction(), LI);
    }
  }

  // Do all instrumentation for IA_Args down here to defer tampering with the
  // CFG in a way that SplitEdge may be able to detect.
  if (DFSF.DFS.getDefaultInstrumentedABI() == DataFlowSanitizer::IA_Args) {
    FunctionType *NewFT = DFSF.DFS.getInstrumentedFunctionType(FT);
    Value *Func =
        IRB.CreateBitCast(CS.getCalledValue(), PointerType::getUnqual(NewFT));
    std::vector<Value *> Args;

    CallSite::arg_iterator i = CS.arg_begin(), e = CS.arg_end();
    for (unsigned n = FT->getNumParams(); n != 0; ++i, --n)
      Args.push_back(*i);

    i = CS.arg_begin();
    for (unsigned n = FT->getNumParams(); n != 0; ++i, --n)
      Args.push_back(DFSF.getShadow(*i));

    if (FT->isVarArg()) {
      unsigned VarArgSize = CS.arg_size() - FT->getNumParams();
      ArrayType *VarArgArrayTy = ArrayType::get(DFSF.DFS.ShadowTy, VarArgSize);
      AllocaInst *VarArgShadow =
          new AllocaInst(VarArgArrayTy, "", DFSF.F->getEntryBlock().begin());
      Args.push_back(IRB.CreateConstGEP2_32(VarArgShadow, 0, 0));
      for (unsigned n = 0; i != e; ++i, ++n) {
        IRB.CreateStore(DFSF.getShadow(*i),
                        IRB.CreateConstGEP2_32(VarArgShadow, 0, n));
        Args.push_back(*i);
      }
    }

    CallSite NewCS;
    if (InvokeInst *II = dyn_cast<InvokeInst>(CS.getInstruction())) {
      NewCS = IRB.CreateInvoke(Func, II->getNormalDest(), II->getUnwindDest(),
                               Args);
    } else {
      NewCS = IRB.CreateCall(Func, Args);
    }
    NewCS.setCallingConv(CS.getCallingConv());
    NewCS.setAttributes(CS.getAttributes().removeAttributes(
        *DFSF.DFS.Ctx, AttributeSet::ReturnIndex,
        AttributeFuncs::typeIncompatible(NewCS.getInstruction()->getType(),
                                         AttributeSet::ReturnIndex)));

    if (Next) {
      ExtractValueInst *ExVal =
          ExtractValueInst::Create(NewCS.getInstruction(), 0, "", Next);
      DFSF.SkipInsts.insert(ExVal);
      ExtractValueInst *ExShadow =
          ExtractValueInst::Create(NewCS.getInstruction(), 1, "", Next);
      DFSF.SkipInsts.insert(ExShadow);
      DFSF.setShadow(ExVal, ExShadow);

      CS.getInstruction()->replaceAllUsesWith(ExVal);
    }

    CS.getInstruction()->eraseFromParent();
  }
}

void DFSanVisitor::visitPHINode(PHINode &PN) {
  PHINode *ShadowPN =
      PHINode::Create(DFSF.DFS.ShadowTy, PN.getNumIncomingValues(), "", &PN);

  // Give the shadow phi node valid predecessors to fool SplitEdge into working.
  Value *UndefShadow = UndefValue::get(DFSF.DFS.ShadowTy);
  for (PHINode::block_iterator i = PN.block_begin(), e = PN.block_end(); i != e;
       ++i) {
    ShadowPN->addIncoming(UndefShadow, *i);
  }

  DFSF.PHIFixups.push_back(std::make_pair(&PN, ShadowPN));
  DFSF.setShadow(&PN, ShadowPN);
}
