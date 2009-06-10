//===- LazyLiveness.cpp - Lazy, CFG-invariant liveness information --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass implements a lazy liveness analysis as per "Fast Liveness Checking
// for SSA-form Programs," by Boissinot, et al.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "lazyliveness"
#include "llvm/CodeGen/LazyLiveness.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/PostOrderIterator.h"
using namespace llvm;

char LazyLiveness::ID = 0;

void LazyLiveness::computeBackedgeChain(MachineFunction& mf, 
                                        MachineBasicBlock* MBB) {
  SparseBitVector<128> tmp = rv[MBB];
  tmp.set(preorder[MBB]);
  tmp &= backedge_source;
  calculated.set(preorder[MBB]);
  
  for (SparseBitVector<128>::iterator I = tmp.begin(); I != tmp.end(); ++I) {
    MachineBasicBlock* SrcMBB = rev_preorder[*I];
    
    for (MachineBasicBlock::succ_iterator SI = SrcMBB->succ_begin();
         SI != SrcMBB->succ_end(); ++SI) {
      MachineBasicBlock* TgtMBB = *SI;
      
      if (backedges.count(std::make_pair(SrcMBB, TgtMBB)) &&
          !rv[MBB].test(preorder[TgtMBB])) {
        if (!calculated.test(preorder[TgtMBB]))
          computeBackedgeChain(mf, TgtMBB);
        
        tv[MBB].set(preorder[TgtMBB]);
        tv[MBB] |= tv[TgtMBB];
      }
    }
    
    tv[MBB].reset(preorder[MBB]);
  }
}

bool LazyLiveness::runOnMachineFunction(MachineFunction &mf) {
  rv.clear();
  tv.clear();
  backedges.clear();
  backedge_source.clear();
  backedge_target.clear();
  calculated.clear();
  preorder.clear();
  
  MRI = &mf.getRegInfo();
  
  // Step 0: Compute preorder numbering for all MBBs.
  unsigned num = 0;
  for (df_iterator<MachineBasicBlock*> DI = df_begin(&*mf.begin());
       DI != df_end(&*mf.end()); ++DI) {
    preorder[*DI] = num++;
    rev_preorder.push_back(*DI);
  }
  
  // Step 1: Compute the transitive closure of the CFG, ignoring backedges.
  for (po_iterator<MachineBasicBlock*> POI = po_begin(&*mf.begin());
       POI != po_end(&*mf.begin()); ++POI) {
    MachineBasicBlock* MBB = *POI;
    SparseBitVector<128>& entry = rv[MBB];
    entry.set(preorder[MBB]);
    
    for (MachineBasicBlock::succ_iterator SI = MBB->succ_begin();
         SI != MBB->succ_end(); ++SI) {
      DenseMap<MachineBasicBlock*, SparseBitVector<128> >::iterator SII = 
                                                         rv.find(*SI);
      
      // Because we're iterating in postorder, any successor that does not yet
      // have an rv entry must be on a backedge.
      if (SII != rv.end()) {
        entry |= SII->second;
      } else {
        backedges.insert(std::make_pair(MBB, *SI));
        backedge_source.set(preorder[MBB]);
        backedge_target.set(preorder[*SI]);
      }
    }
  }
  
  for (SparseBitVector<128>::iterator I = backedge_source.begin();
       I != backedge_source.end(); ++I)
    computeBackedgeChain(mf, rev_preorder[*I]);
  
  for (po_iterator<MachineBasicBlock*> POI = po_begin(&*mf.begin()),
       POE = po_end(&*mf.begin()); POI != POE; ++POI)
    if (!backedge_target.test(preorder[*POI]))
      for (MachineBasicBlock::succ_iterator SI = (*POI)->succ_begin();
           SI != (*POI)->succ_end(); ++SI)
        if (!backedges.count(std::make_pair(*POI, *SI)) && tv.count(*SI))
          tv[*POI]= tv[*SI];
  
  for (po_iterator<MachineBasicBlock*> POI = po_begin(&*mf.begin()),
       POE = po_end(&*mf.begin()); POI != POE; ++POI)
    tv[*POI].set(preorder[*POI]);
  
  return false;
}

bool LazyLiveness::vregLiveIntoMBB(unsigned vreg, MachineBasicBlock* MBB) {
  MachineDominatorTree& MDT = getAnalysis<MachineDominatorTree>();
  
  MachineBasicBlock* DefMBB = MRI->def_begin(vreg)->getParent();
  unsigned def = preorder[DefMBB];
  unsigned max_dom = 0;
  for (df_iterator<MachineDomTreeNode*> DI = df_begin(MDT[DefMBB]);
       DI != df_end(MDT[DefMBB]); ++DI)
    if (preorder[DI->getBlock()] > max_dom) {
      max_dom = preorder[(*DI)->getBlock()];
    }
  
  if (preorder[MBB] <= def || max_dom < preorder[MBB])
    return false;
  
  SparseBitVector<128>::iterator I = tv[MBB].begin();
  while (I != tv[MBB].end() && *I <= def) ++I;
  while (I != tv[MBB].end() && *I < max_dom) {
    for (MachineRegisterInfo::use_iterator UI = MRI->use_begin(vreg);
         UI != MachineRegisterInfo::use_end(); ++UI) {
      MachineBasicBlock* UseMBB = UI->getParent();
      if (rv[rev_preorder[*I]].test(preorder[UseMBB]))
        return true;
        
      unsigned t_dom = 0;
      for (df_iterator<MachineDomTreeNode*> DI =
           df_begin(MDT[rev_preorder[*I]]);
           DI != df_end(MDT[rev_preorder[*I]]); ++DI)
        if (preorder[DI->getBlock()] > t_dom) {
          max_dom = preorder[(*DI)->getBlock()];
        }
      I = tv[MBB].begin();
      while (I != tv[MBB].end() && *I < t_dom) ++I;
    }
  }
  
  return false;
}

