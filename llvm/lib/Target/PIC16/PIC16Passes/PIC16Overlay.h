//===-- PIC16Overlay.h - Interface for PIC16 Frame Overlay -*- C++ -*-===//
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
 

using std::string;
using namespace llvm;

namespace  llvm {
  // Forward declarations.
  class Function;
  class Module;
  class ModulePass;
  class AnalysisUsage;
  class CallGraphNode;
  class CallGraph;

  namespace PIC16OVERLAY {
    enum OverlayConsts {
      StartInterruptColor = 200,
      StartIndirectCallColor = 300
    }; 
  }
  class PIC16Overlay : public ModulePass {
    std::string OverlayStr;
    unsigned InterruptDepth;
    unsigned IndirectCallColor;
  public:
    static char ID; // Class identification 
    PIC16Overlay() : ModulePass(&ID) {
      OverlayStr = "Overlay=";
      InterruptDepth = PIC16OVERLAY::StartInterruptColor;
      IndirectCallColor = PIC16OVERLAY::StartIndirectCallColor;
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
