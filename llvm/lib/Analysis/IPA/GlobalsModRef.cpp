//===- GlobalsModRef.cpp - Simple Mod/Ref Analysis for Globals ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This simple pass provides alias and mod/ref information for global values
// that do not have their address taken, and keeps track of whether functions
// read or write memory (are "pure").  For this simple (but very common) case,
// we can provide pretty accurate and useful information.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Passes.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Instructions.h"
#include "llvm/Constants.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Support/InstIterator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/SCCIterator.h"
#include <set>
using namespace llvm;

namespace {
  Statistic<>
  NumNonAddrTakenGlobalVars("globalsmodref-aa",
                            "Number of global vars without address taken");
  Statistic<>
  NumNonAddrTakenFunctions("globalsmodref-aa",
                           "Number of functions without address taken");
  Statistic<>
  NumNoMemFunctions("globalsmodref-aa",
                    "Number of functions that do not access memory");
  Statistic<>
  NumReadMemFunctions("globalsmodref-aa",
                      "Number of functions that only read memory");

  /// FunctionRecord - One instance of this structure is stored for every
  /// function in the program.  Later, the entries for these functions are
  /// removed if the function is found to call an external function (in which
  /// case we know nothing about it.
  struct FunctionRecord {
    /// GlobalInfo - Maintain mod/ref info for all of the globals without
    /// addresses taken that are read or written (transitively) by this
    /// function.
    std::map<GlobalValue*, unsigned> GlobalInfo;

    unsigned getInfoForGlobal(GlobalValue *GV) const {
      std::map<GlobalValue*, unsigned>::const_iterator I = GlobalInfo.find(GV);
      if (I != GlobalInfo.end())
        return I->second;
      return 0;
    }

    /// FunctionEffect - Capture whether or not this function reads or writes to
    /// ANY memory.  If not, we can do a lot of aggressive analysis on it.
    unsigned FunctionEffect;

    FunctionRecord() : FunctionEffect(0) {}
  };

  /// GlobalsModRef - The actual analysis pass.
  class GlobalsModRef : public ModulePass, public AliasAnalysis {
    /// NonAddressTakenGlobals - The globals that do not have their addresses
    /// taken.
    std::set<GlobalValue*> NonAddressTakenGlobals;

    /// FunctionInfo - For each function, keep track of what globals are
    /// modified or read.
    std::map<Function*, FunctionRecord> FunctionInfo;

  public:
    bool runOnModule(Module &M) {
      InitializeAliasAnalysis(this);                 // set up super class
      AnalyzeGlobals(M);                          // find non-addr taken globals
      AnalyzeCallGraph(getAnalysis<CallGraph>(), M); // Propagate on CG
      return false;
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AliasAnalysis::getAnalysisUsage(AU);
      AU.addRequired<CallGraph>();
      AU.setPreservesAll();                         // Does not transform code
    }

    //------------------------------------------------
    // Implement the AliasAnalysis API
    //
    AliasResult alias(const Value *V1, unsigned V1Size,
                      const Value *V2, unsigned V2Size);
    ModRefResult getModRefInfo(CallSite CS, Value *P, unsigned Size);
    ModRefResult getModRefInfo(CallSite CS1, CallSite CS2) {
      return AliasAnalysis::getModRefInfo(CS1,CS2);
    }
    bool hasNoModRefInfoForCalls() const { return false; }

    /// getModRefBehavior - Return the behavior of the specified function if
    /// called from the specified call site.  The call site may be null in which
    /// case the most generic behavior of this function should be returned.
    virtual ModRefBehavior getModRefBehavior(Function *F, CallSite CS,
                                         std::vector<PointerAccessInfo> *Info) {
      if (FunctionRecord *FR = getFunctionInfo(F))
        if (FR->FunctionEffect == 0)
          return DoesNotAccessMemory;
        else if ((FR->FunctionEffect & Mod) == 0)
          return OnlyReadsMemory;
      return AliasAnalysis::getModRefBehavior(F, CS, Info);
    }

    virtual void deleteValue(Value *V);
    virtual void copyValue(Value *From, Value *To);

  private:
    /// getFunctionInfo - Return the function info for the function, or null if
    /// the function calls an external function (in which case we don't have
    /// anything useful to say about it).
    FunctionRecord *getFunctionInfo(Function *F) {
      std::map<Function*, FunctionRecord>::iterator I = FunctionInfo.find(F);
      if (I != FunctionInfo.end())
        return &I->second;
      return 0;
    }

    void AnalyzeGlobals(Module &M);
    void AnalyzeCallGraph(CallGraph &CG, Module &M);
    void AnalyzeSCC(std::vector<CallGraphNode *> &SCC);
    bool AnalyzeUsesOfGlobal(Value *V, std::vector<Function*> &Readers,
                             std::vector<Function*> &Writers);
  };

  RegisterPass<GlobalsModRef> X("globalsmodref-aa",
                                "Simple mod/ref analysis for globals");
  RegisterAnalysisGroup<AliasAnalysis> Y(X);
}

Pass *llvm::createGlobalsModRefPass() { return new GlobalsModRef(); }


/// AnalyzeGlobalUses - Scan through the users of all of the internal
/// GlobalValue's in the program.  If none of them have their "Address taken"
/// (really, their address passed to something nontrivial), record this fact,
/// and record the functions that they are used directly in.
void GlobalsModRef::AnalyzeGlobals(Module &M) {
  std::vector<Function*> Readers, Writers;
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
    if (I->hasInternalLinkage()) {
      if (!AnalyzeUsesOfGlobal(I, Readers, Writers)) {
        // Remember that we are tracking this global.
        NonAddressTakenGlobals.insert(I);
        ++NumNonAddrTakenFunctions;
      }
      Readers.clear(); Writers.clear();
    }

  for (Module::global_iterator I = M.global_begin(), E = M.global_end();
       I != E; ++I)
    if (I->hasInternalLinkage()) {
      if (!AnalyzeUsesOfGlobal(I, Readers, Writers)) {
        // Remember that we are tracking this global, and the mod/ref fns
        NonAddressTakenGlobals.insert(I);
        for (unsigned i = 0, e = Readers.size(); i != e; ++i)
          FunctionInfo[Readers[i]].GlobalInfo[I] |= Ref;

        if (!I->isConstant())  // No need to keep track of writers to constants
          for (unsigned i = 0, e = Writers.size(); i != e; ++i)
            FunctionInfo[Writers[i]].GlobalInfo[I] |= Mod;
        ++NumNonAddrTakenGlobalVars;
      }
      Readers.clear(); Writers.clear();
    }
}

/// AnalyzeUsesOfGlobal - Look at all of the users of the specified global value
/// derived pointer.  If this is used by anything complex (i.e., the address
/// escapes), return true.  Also, while we are at it, keep track of those
/// functions that read and write to the value.
bool GlobalsModRef::AnalyzeUsesOfGlobal(Value *V,
                                        std::vector<Function*> &Readers,
                                        std::vector<Function*> &Writers) {
  if (!isa<PointerType>(V->getType())) return true;

  for (Value::use_iterator UI = V->use_begin(), E = V->use_end(); UI != E; ++UI)
    if (LoadInst *LI = dyn_cast<LoadInst>(*UI)) {
      Readers.push_back(LI->getParent()->getParent());
    } else if (StoreInst *SI = dyn_cast<StoreInst>(*UI)) {
      if (V == SI->getOperand(0)) return true;  // Storing the pointer
      Writers.push_back(SI->getParent()->getParent());
    } else if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(*UI)) {
      if (AnalyzeUsesOfGlobal(GEP, Readers, Writers)) return true;
    } else if (CallInst *CI = dyn_cast<CallInst>(*UI)) {
      // Make sure that this is just the function being called, not that it is
      // passing into the function.
      for (unsigned i = 1, e = CI->getNumOperands(); i != e; ++i)
        if (CI->getOperand(i) == V) return true;
    } else if (InvokeInst *II = dyn_cast<InvokeInst>(*UI)) {
      // Make sure that this is just the function being called, not that it is
      // passing into the function.
      for (unsigned i = 3, e = II->getNumOperands(); i != e; ++i)
        if (II->getOperand(i) == V) return true;
    } else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(*UI)) {
      if (CE->getOpcode() == Instruction::GetElementPtr ||
          CE->getOpcode() == Instruction::Cast) {
        if (AnalyzeUsesOfGlobal(CE, Readers, Writers))
          return true;
      } else {
        return true;
      }
    } else if (GlobalValue *GV = dyn_cast<GlobalValue>(*UI)) {
      if (AnalyzeUsesOfGlobal(GV, Readers, Writers)) return true;
    } else {
      return true;
    }
  return false;
}

/// AnalyzeCallGraph - At this point, we know the functions where globals are
/// immediately stored to and read from.  Propagate this information up the call
/// graph to all callers and compute the mod/ref info for all memory for each
/// function.
void GlobalsModRef::AnalyzeCallGraph(CallGraph &CG, Module &M) {
  // We do a bottom-up SCC traversal of the call graph.  In other words, we
  // visit all callees before callers (leaf-first).
  for (scc_iterator<CallGraph*> I = scc_begin(&CG), E = scc_end(&CG); I!=E; ++I)
    if ((*I).size() != 1) {
      AnalyzeSCC(*I);
    } else if (Function *F = (*I)[0]->getFunction()) {
      if (!F->isExternal()) {
        // Nonexternal function.
        AnalyzeSCC(*I);
      } else {
        // Otherwise external function.  Handle intrinsics and other special
        // cases here.
        if (getAnalysis<AliasAnalysis>().doesNotAccessMemory(F))
          // If it does not access memory, process the function, causing us to
          // realize it doesn't do anything (the body is empty).
          AnalyzeSCC(*I);
        else {
          // Otherwise, don't process it.  This will cause us to conservatively
          // assume the worst.
        }
      }
    } else {
      // Do not process the external node, assume the worst.
    }
}

void GlobalsModRef::AnalyzeSCC(std::vector<CallGraphNode *> &SCC) {
  assert(!SCC.empty() && "SCC with no functions?");
  FunctionRecord &FR = FunctionInfo[SCC[0]->getFunction()];

  bool CallsExternal = false;
  unsigned FunctionEffect = 0;

  // Collect the mod/ref properties due to called functions.  We only compute
  // one mod-ref set
  for (unsigned i = 0, e = SCC.size(); i != e && !CallsExternal; ++i)
    for (CallGraphNode::iterator CI = SCC[i]->begin(), E = SCC[i]->end();
         CI != E; ++CI)
      if (Function *Callee = CI->second->getFunction()) {
        if (FunctionRecord *CalleeFR = getFunctionInfo(Callee)) {
          // Propagate function effect up.
          FunctionEffect |= CalleeFR->FunctionEffect;

          // Incorporate callee's effects on globals into our info.
          for (std::map<GlobalValue*, unsigned>::iterator GI =
                 CalleeFR->GlobalInfo.begin(), E = CalleeFR->GlobalInfo.end();
               GI != E; ++GI)
            FR.GlobalInfo[GI->first] |= GI->second;

        } else {
          // Okay, if we can't say anything about it, maybe some other alias
          // analysis can.
          ModRefBehavior MRB =
            AliasAnalysis::getModRefBehavior(Callee, CallSite());
          if (MRB != DoesNotAccessMemory) {
            // FIXME: could make this more aggressive for functions that just
            // read memory.  We should just say they read all globals.
            CallsExternal = true;
            break;
          }
        }
      } else {
        CallsExternal = true;
        break;
      }

  // If this SCC calls an external function, we can't say anything about it, so
  // remove all SCC functions from the FunctionInfo map.
  if (CallsExternal) {
    for (unsigned i = 0, e = SCC.size(); i != e; ++i)
      FunctionInfo.erase(SCC[i]->getFunction());
    return;
  }

  // Otherwise, unless we already know that this function mod/refs memory, scan
  // the function bodies to see if there are any explicit loads or stores.
  if (FunctionEffect != ModRef) {
    for (unsigned i = 0, e = SCC.size(); i != e && FunctionEffect != ModRef;++i)
      for (inst_iterator II = inst_begin(SCC[i]->getFunction()),
             E = inst_end(SCC[i]->getFunction());
           II != E && FunctionEffect != ModRef; ++II)
        if (isa<LoadInst>(*II))
          FunctionEffect |= Ref;
        else if (isa<StoreInst>(*II))
          FunctionEffect |= Mod;
        else if (isa<MallocInst>(*II) || isa<FreeInst>(*II))
          FunctionEffect |= ModRef;
  }

  if ((FunctionEffect & Mod) == 0)
    ++NumReadMemFunctions;
  if (FunctionEffect == 0)
    ++NumNoMemFunctions;
  FR.FunctionEffect = FunctionEffect;

  // Finally, now that we know the full effect on this SCC, clone the
  // information to each function in the SCC.
  for (unsigned i = 1, e = SCC.size(); i != e; ++i)
    FunctionInfo[SCC[i]->getFunction()] = FR;
}



/// getUnderlyingObject - This traverses the use chain to figure out what object
/// the specified value points to.  If the value points to, or is derived from,
/// a global object, return it.
static const GlobalValue *getUnderlyingObject(const Value *V) {
  if (!isa<PointerType>(V->getType())) return 0;

  // If we are at some type of object... return it.
  if (const GlobalValue *GV = dyn_cast<GlobalValue>(V)) return GV;

  // Traverse through different addressing mechanisms...
  if (const Instruction *I = dyn_cast<Instruction>(V)) {
    if (isa<CastInst>(I) || isa<GetElementPtrInst>(I))
      return getUnderlyingObject(I->getOperand(0));
  } else if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(V)) {
    if (CE->getOpcode() == Instruction::Cast ||
        CE->getOpcode() == Instruction::GetElementPtr)
      return getUnderlyingObject(CE->getOperand(0));
  }
  return 0;
}

/// alias - If one of the pointers is to a global that we are tracking, and the
/// other is some random pointer, we know there cannot be an alias, because the
/// address of the global isn't taken.
AliasAnalysis::AliasResult
GlobalsModRef::alias(const Value *V1, unsigned V1Size,
                     const Value *V2, unsigned V2Size) {
  GlobalValue *GV1 = const_cast<GlobalValue*>(getUnderlyingObject(V1));
  GlobalValue *GV2 = const_cast<GlobalValue*>(getUnderlyingObject(V2));

  // If the global's address is taken, pretend we don't know it's a pointer to
  // the global.
  if (GV1 && !NonAddressTakenGlobals.count(GV1)) GV1 = 0;
  if (GV2 && !NonAddressTakenGlobals.count(GV2)) GV2 = 0;

  if ((GV1 || GV2) && GV1 != GV2)
    return NoAlias;

  return AliasAnalysis::alias(V1, V1Size, V2, V2Size);
}

AliasAnalysis::ModRefResult
GlobalsModRef::getModRefInfo(CallSite CS, Value *P, unsigned Size) {
  unsigned Known = ModRef;

  // If we are asking for mod/ref info of a direct call with a pointer to a
  // global we are tracking, return information if we have it.
  if (GlobalValue *GV = const_cast<GlobalValue*>(getUnderlyingObject(P)))
    if (GV->hasInternalLinkage())
      if (Function *F = CS.getCalledFunction())
        if (NonAddressTakenGlobals.count(GV))
          if (FunctionRecord *FR = getFunctionInfo(F))
            Known = FR->getInfoForGlobal(GV);

  if (Known == NoModRef)
    return NoModRef; // No need to query other mod/ref analyses
  return ModRefResult(Known & AliasAnalysis::getModRefInfo(CS, P, Size));
}


//===----------------------------------------------------------------------===//
// Methods to update the analysis as a result of the client transformation.
//
void GlobalsModRef::deleteValue(Value *V) {
  if (GlobalValue *GV = dyn_cast<GlobalValue>(V))
    NonAddressTakenGlobals.erase(GV);
}

void GlobalsModRef::copyValue(Value *From, Value *To) {
}
