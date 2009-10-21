//===-- PIC16FrameOverlay.h - Interface for PIC16 Frame Overlay -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source 
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the PIC16 Overlay infrastructure.
//
//===----------------------------------------------------------------------===//

#ifndef PIC16FRAMEOVERLAY_H
#define PIC16FRAMEOVERLAY_H
 
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Pass.h"
#include "llvm/CallGraphSCCPass.h"

using std::string;
using namespace llvm;

namespace  llvm {
  namespace PIC16Overlay {
    enum OverlayConsts {
      StartInterruptColor = 200,
      StartIndirectCallColor = 300
    }; 
  }
  class PIC16FrameOverlay : public ModulePass {
    std::string OverlayStr;
    unsigned InterruptDepth;
    unsigned IndirectCallColor;
  public:
    static char ID; // Class identification 
    PIC16FrameOverlay() : ModulePass(&ID) {
      OverlayStr = "Overlay=";
      InterruptDepth = PIC16Overlay::StartInterruptColor;
      IndirectCallColor = PIC16Overlay::StartIndirectCallColor;
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const; 
    virtual bool runOnModule(Module &M);

  private: 
    unsigned getColor(Function *Fn);
    void setColor(Function *Fn, unsigned Color);
    unsigned ModifyDepthForInterrupt(CallGraphNode *CGN, unsigned Depth);
    void MarkIndirectlyCalledFunctions(Module &M);
    void DFSTraverse(CallGraphNode *CGN, unsigned Depth);
  };
}  // End of  namespace

#endif
