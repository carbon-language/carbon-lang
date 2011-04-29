//===-- DebugInfoProbe.cpp - DebugInfo Probe ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements DebugInfoProbe. This probe can be used by a pass
// manager to analyze how optimizer is treating debugging information.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "debuginfoprobe"
#include "llvm/DebugInfoProbe.h"
#include "llvm/Function.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Metadata.h"
#include "llvm/PassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLoc.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/StringRef.h"
#include <set>
#include <string>

using namespace llvm;

static cl::opt<bool>
EnableDebugInfoProbe("enable-debug-info-probe", cl::Hidden,
                     cl::desc("Enable debug info probe"));

// CreateInfoOutputFile - Return a file stream to print our output on.
namespace llvm { extern raw_ostream *CreateInfoOutputFile(); }

//===----------------------------------------------------------------------===//
// DebugInfoProbeImpl - This class implements a interface to monitor
// how an optimization pass is preserving debugging information.

namespace llvm {

  class DebugInfoProbeImpl {
  public:
    DebugInfoProbeImpl() : NumDbgLineLost(0),NumDbgValueLost(0) {}
    void initialize(StringRef PName, Function &F);
    void finalize(Function &F);
    void report();
  private:
    unsigned NumDbgLineLost, NumDbgValueLost;
    std::string PassName;
    Function *TheFn;
    std::set<unsigned> LineNos;
    std::set<MDNode *> DbgVariables;
    std::set<Instruction *> MissingDebugLoc;
  };
}

//===----------------------------------------------------------------------===//
// DebugInfoProbeImpl

static void collect(Function &F, std::set<unsigned> &Lines) {
  for (Function::iterator FI = F.begin(), FE = F.end(); FI != FE; ++FI)
    for (BasicBlock::iterator BI = FI->begin(), BE = FI->end(); 
         BI != BE; ++BI) {
      const DebugLoc &DL = BI->getDebugLoc();
      unsigned LineNo = 0;
      if (!DL.isUnknown()) {
        if (MDNode *N = DL.getInlinedAt(F.getContext()))
          LineNo = DebugLoc::getFromDILocation(N).getLine();
        else
          LineNo = DL.getLine();

        Lines.insert(LineNo);
      }
    }
}

/// initialize - Collect information before running an optimization pass.
void DebugInfoProbeImpl::initialize(StringRef PName, Function &F) {
  if (!EnableDebugInfoProbe) return;
  PassName = PName;

  LineNos.clear();
  DbgVariables.clear();
  TheFn = &F;
  collect(F, LineNos);

  for (Function::iterator FI = F.begin(), FE = F.end(); FI != FE; ++FI)
    for (BasicBlock::iterator BI = FI->begin(), BE = FI->end(); 
         BI != BE; ++BI) {
      if (BI->getDebugLoc().isUnknown())
        MissingDebugLoc.insert(BI);
      if (!isa<DbgInfoIntrinsic>(BI)) continue;
      Value *Addr = NULL;
      MDNode *Node = NULL;
      if (DbgDeclareInst *DDI = dyn_cast<DbgDeclareInst>(BI)) {
        Addr = DDI->getAddress();
        Node = DDI->getVariable();
      } else if (DbgValueInst *DVI = dyn_cast<DbgValueInst>(BI)) {
        Addr = DVI->getValue();
        Node = DVI->getVariable();
      }
      if (Addr)
        DbgVariables.insert(Node);
    }
}

/// report - Report findings. This should be invoked after finalize.
void DebugInfoProbeImpl::report() {
  if (!EnableDebugInfoProbe) return;
  if (NumDbgLineLost || NumDbgValueLost) {
    raw_ostream *OutStream = CreateInfoOutputFile();
    if (NumDbgLineLost)
      *OutStream << NumDbgLineLost
                 << "\t times line number info lost by "
                 << PassName << "\n";
    if (NumDbgValueLost)
      *OutStream << NumDbgValueLost
                 << "\t times variable info lost by    "
                 << PassName << "\n";
    delete OutStream;
  }
  NumDbgLineLost = 0;
  NumDbgValueLost = 0;
}

/// finalize - Collect information after running an optimization pass. This
/// must be used after initialization.
void DebugInfoProbeImpl::finalize(Function &F) {
  if (!EnableDebugInfoProbe) return;
  std::set<unsigned> LineNos2;
  collect(F, LineNos2);
  assert (TheFn == &F && "Invalid function to measure!");

  for (std::set<unsigned>::iterator I = LineNos.begin(),
         E = LineNos.end(); I != E; ++I) {
    unsigned LineNo = *I;
    if (LineNos2.count(LineNo) == 0) {
      DEBUG(dbgs() 
            << "DebugInfoProbe("
            << PassName
            << "): Losing dbg info for source line " 
            << LineNo << "\n");
      ++NumDbgLineLost;
    }
  }

  std::set<MDNode *>DbgVariables2;
  for (Function::iterator FI = F.begin(), FE = F.end(); FI != FE; ++FI)
    for (BasicBlock::iterator BI = FI->begin(), BE = FI->end(); 
         BI != BE; ++BI) {
      if (BI->getDebugLoc().isUnknown() &&
          MissingDebugLoc.count(BI) == 0) {
        DEBUG(dbgs() << "DebugInfoProbe(" << PassName << "): --- ");
        DEBUG(BI->print(dbgs()));
        DEBUG(dbgs() << "\n");
      }
      if (!isa<DbgInfoIntrinsic>(BI)) continue;
      Value *Addr = NULL;
      MDNode *Node = NULL;
      if (DbgDeclareInst *DDI = dyn_cast<DbgDeclareInst>(BI)) {
        Addr = DDI->getAddress();
        Node = DDI->getVariable();
      } else if (DbgValueInst *DVI = dyn_cast<DbgValueInst>(BI)) {
        Addr = DVI->getValue();
        Node = DVI->getVariable();
      }
      if (Addr)
        DbgVariables2.insert(Node);
    }

  for (std::set<MDNode *>::iterator I = DbgVariables.begin(), 
         E = DbgVariables.end(); I != E; ++I) {
    if (DbgVariables2.count(*I) == 0 && (*I)->getNumOperands() >= 2) {
      DEBUG(dbgs() 
            << "DebugInfoProbe("
            << PassName
            << "): Losing dbg info for variable: ";
            if (MDString *MDS = dyn_cast_or_null<MDString>(
                (*I)->getOperand(2)))
              dbgs() << MDS->getString();
            else
              dbgs() << "...";
            dbgs() << "\n");
      ++NumDbgValueLost;
    }
  }
}

//===----------------------------------------------------------------------===//
// DebugInfoProbe

DebugInfoProbe::DebugInfoProbe() {
  pImpl = new DebugInfoProbeImpl();
}

DebugInfoProbe::~DebugInfoProbe() {
  delete pImpl;
}

/// initialize - Collect information before running an optimization pass.
void DebugInfoProbe::initialize(StringRef PName, Function &F) {
  pImpl->initialize(PName, F);
}

/// finalize - Collect information after running an optimization pass. This
/// must be used after initialization.
void DebugInfoProbe::finalize(Function &F) {
  pImpl->finalize(F);
}

/// report - Report findings. This should be invoked after finalize.
void DebugInfoProbe::report() {
  pImpl->report();
}

//===----------------------------------------------------------------------===//
// DebugInfoProbeInfo

/// ~DebugInfoProbeInfo - Report data collected by all probes before deleting
/// them.
DebugInfoProbeInfo::~DebugInfoProbeInfo() {
  if (!EnableDebugInfoProbe) return;
    for (StringMap<DebugInfoProbe*>::iterator I = Probes.begin(),
           E = Probes.end(); I != E; ++I) {
      I->second->report();
      delete I->second;
    }
  }

/// initialize - Collect information before running an optimization pass.
void DebugInfoProbeInfo::initialize(Pass *P, Function &F) {
  if (!EnableDebugInfoProbe) return;
  if (P->getAsPMDataManager())
    return;

  StringMapEntry<DebugInfoProbe *> &Entry =
    Probes.GetOrCreateValue(P->getPassName());
  DebugInfoProbe *&Probe = Entry.getValue();
  if (!Probe)
    Probe = new DebugInfoProbe();
  Probe->initialize(P->getPassName(), F);
}

/// finalize - Collect information after running an optimization pass. This
/// must be used after initialization.
void DebugInfoProbeInfo::finalize(Pass *P, Function &F) {
  if (!EnableDebugInfoProbe) return;
  if (P->getAsPMDataManager())
    return;
  StringMapEntry<DebugInfoProbe *> &Entry =
    Probes.GetOrCreateValue(P->getPassName());
  DebugInfoProbe *&Probe = Entry.getValue();
  assert (Probe && "DebugInfoProbe is not initialized!");
  Probe->finalize(F);
}
