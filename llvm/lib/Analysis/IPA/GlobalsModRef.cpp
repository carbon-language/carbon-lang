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
// that do not have their address taken.  For this simple (but very common)
// case, we can provide pretty accurate and useful information.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "globalsmodref"
#include "llvm/Analysis/Passes.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Instructions.h"
#include "llvm/Constants.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/CallGraph.h"
#include "Support/Debug.h"
#include "Support/Statistic.h"
#include "Support/SCCIterator.h"
#include <set>
using namespace llvm;

namespace {
  Statistic<>
  NumNonAddrTakenGlobalVars("globalsmodref-aa",
                            "Number of global vars without address taken");
  Statistic<>
  NumNonAddrTakenFunctions("globalsmodref-aa",
                           "Number of functions without address taken");

  class GlobalsModRef : public Pass, public AliasAnalysis {
    /// ModRefFns - One instance of this record is kept for each global without
    /// its address taken.
    struct ModRefFns {
      /// RefFns/ModFns - Sets of functions that and write globals.
      std::set<Function*> RefFns, ModFns;
    };

    /// NonAddressTakenGlobals - A map of globals that do not have their
    /// addresses taken to their record.
    std::map<GlobalValue*, ModRefFns> NonAddressTakenGlobals;

    /// FunctionInfo - For each function, keep track of what globals are
    /// modified or read.
    std::map<std::pair<Function*, GlobalValue*>, unsigned> FunctionInfo;

  public:
    bool run(Module &M) {
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
    bool hasNoModRefInfoForCalls() const { return false; }

    virtual void deleteValue(Value *V);
    virtual void copyValue(Value *From, Value *To);

  private:
    void AnalyzeGlobals(Module &M);
    void AnalyzeCallGraph(CallGraph &CG, Module &M);
    bool AnalyzeUsesOfGlobal(Value *V, std::vector<Function*> &Readers,
                             std::vector<Function*> &Writers);
  };
  
  RegisterOpt<GlobalsModRef> X("globalsmodref-aa",
                               "Simple mod/ref analysis for globals");
  RegisterAnalysisGroup<AliasAnalysis, GlobalsModRef> Y;
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
        // Remember that we are tracking this global, and the mod/ref fns
        ModRefFns &E = NonAddressTakenGlobals[I];
        E.RefFns.insert(Readers.begin(), Readers.end());
        E.ModFns.insert(Writers.begin(), Writers.end());
        ++NumNonAddrTakenFunctions;
      }
      Readers.clear(); Writers.clear();
    }

  for (Module::giterator I = M.gbegin(), E = M.gend(); I != E; ++I)
    // FIXME: it is kinda dumb to track aliasing properties for constant
    // globals, it will never be particularly useful anyways, 'cause they can
    // never be modified (and the optimizer knows this already)!
    if (I->hasInternalLinkage()) {
      if (!AnalyzeUsesOfGlobal(I, Readers, Writers)) {
        // Remember that we are tracking this global, and the mod/ref fns
        ModRefFns &E = NonAddressTakenGlobals[I];
        E.RefFns.insert(Readers.begin(), Readers.end());
        E.ModFns.insert(Writers.begin(), Writers.end());
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
  //if (!isa<PointerType>(V->getType())) return true;

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
    } else if (ConstantPointerRef *CPR = dyn_cast<ConstantPointerRef>(*UI)) {
      if (AnalyzeUsesOfGlobal(CPR, Readers, Writers)) return true;
    } else {
      return true;
    }
  return false;
}

/// AnalyzeCallGraph - At this point, we know the functions where globals are
/// immediately stored to and read from.  Propagate this information up the call
/// graph to all callers.
void GlobalsModRef::AnalyzeCallGraph(CallGraph &CG, Module &M) {
  if (NonAddressTakenGlobals.empty()) return;  // Don't bother, nothing to do.

  // Invert the NonAddressTakenGlobals map into the FunctionInfo map.
  for (std::map<GlobalValue*, ModRefFns>::iterator I = 
         NonAddressTakenGlobals.begin(), E = NonAddressTakenGlobals.end();
       I != E; ++I) {
    GlobalValue *GV = I->first;
    ModRefFns &MRInfo = I->second;
    for (std::set<Function*>::iterator I = MRInfo.RefFns.begin(), 
           E = MRInfo.RefFns.begin(); I != E; ++I)
      FunctionInfo[std::make_pair(*I, GV)] |= Ref;
    MRInfo.RefFns.clear();
    for (std::set<Function*>::iterator I = MRInfo.ModFns.begin(), 
           E = MRInfo.ModFns.begin(); I != E; ++I)
      FunctionInfo[std::make_pair(*I, GV)] |= Mod;
    MRInfo.ModFns.clear();
  }

  // We do a bottom-up SCC traversal of the call graph.  In other words, we
  // visit all callees before callers (leaf-first).
  for (scc_iterator<CallGraph*> I = scc_begin(&CG), E = scc_end(&CG);
       I != E; ++I) {
    std::map<GlobalValue*, unsigned> ModRefProperties;
    const std::vector<CallGraphNode *> &SCC = *I;

    // Collect the mod/ref properties due to called functions.
    for (unsigned i = 0, e = SCC.size(); i != e; ++i)
      for (CallGraphNode::iterator CI = SCC[i]->begin(), E = SCC[i]->end();
           CI != E; ++CI) {
        if (Function *Callee = (*CI)->getFunction()) {
          // Otherwise, combine the callee properties into our accumulated set.
          std::map<std::pair<Function*, GlobalValue*>, unsigned>::iterator
            CI = FunctionInfo.lower_bound(std::make_pair(Callee,
                                                         (GlobalValue*)0));
          for (;CI != FunctionInfo.end() && CI->first.first == Callee; ++CI)
            ModRefProperties[CI->first.second] |= CI->second;
        } else {
          // For now assume that external functions could mod/ref anything,
          // since they could call into an escaping function that mod/refs an
          // internal.  FIXME: We need better tracking!
          for (std::map<GlobalValue*, ModRefFns>::iterator GI = 
                 NonAddressTakenGlobals.begin(),
                 E = NonAddressTakenGlobals.end(); GI != E; ++GI)
            ModRefProperties[GI->first] = ModRef;
          goto Out;
        }
      }
  Out:
    // Set all functions in the CFG to have these properties.  FIXME: it would
    // be better to use union find to only store these properties once,
    // PARTICULARLY if it's the universal set.
    for (unsigned i = 0, e = SCC.size(); i != e; ++i)
      if (Function *F = SCC[i]->getFunction()) {
        for (std::map<GlobalValue*, unsigned>::iterator I =
               ModRefProperties.begin(), E = ModRefProperties.end();
             I != E; ++I)
          FunctionInfo[std::make_pair(F, I->first)] = I->second;
      }
  }
}



/// getUnderlyingObject - This traverses the use chain to figure out what object
/// the specified value points to.  If the value points to, or is derived from,
/// a global object, return it.
static const GlobalValue *getUnderlyingObject(const Value *V) {
  //if (!isa<PointerType>(V->getType())) return 0;

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
  } else if (const ConstantPointerRef *CPR = dyn_cast<ConstantPointerRef>(V)) {
    return CPR->getValue();
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
  // global, return information if we have it.
  if (GlobalValue *GV = const_cast<GlobalValue*>(getUnderlyingObject(P)))
    if (GV->hasInternalLinkage())
      if (Function *F = CS.getCalledFunction()) {
        std::map<std::pair<Function*, GlobalValue*>, unsigned>::iterator
          it = FunctionInfo.find(std::make_pair(F, GV));
        if (it != FunctionInfo.end())
          Known = it->second;
      }

  if (Known == NoModRef)
    return NoModRef; // No need to query other mod/ref analyses
  return ModRefResult(Known & AliasAnalysis::getModRefInfo(CS, P, Size));
}


//===----------------------------------------------------------------------===//
// Methods to update the analysis as a result of the client transformation.
//
void GlobalsModRef::deleteValue(Value *V) {
  if (GlobalValue *GV = dyn_cast<GlobalValue>(V)) {
    std::map<GlobalValue*, ModRefFns>::iterator I =
      NonAddressTakenGlobals.find(GV);
    if (I != NonAddressTakenGlobals.end())
      NonAddressTakenGlobals.erase(I);
  }
}

void GlobalsModRef::copyValue(Value *From, Value *To) {
  if (GlobalValue *FromGV = dyn_cast<GlobalValue>(From))
    if (GlobalValue *ToGV = dyn_cast<GlobalValue>(To)) {
      std::map<GlobalValue*, ModRefFns>::iterator I =
        NonAddressTakenGlobals.find(FromGV);
      if (I != NonAddressTakenGlobals.end())
        NonAddressTakenGlobals[ToGV] = I->second;
    }
}
