//===-- PIC16FrameOverlay.h - Interface for PIC16 Frame Overlay -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source 
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the PIC16 Frame Overlay infrastructure.
//
//===----------------------------------------------------------------------===//

#ifndef PIC16FRAMEOVERLAY_H
#define PIC16FRAMEOVERLAY_H
 
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Pass.h"
#include "llvm/Module.h"
#include "llvm/Support/raw_ostream.h"
#include <vector>
#include <iostream>
using std::vector;
using std::string;
using namespace llvm;

namespace  { 

  class PIC16FrameOverlay : public ModulePass { 
  public:
    static char ID; // Class identification 
    PIC16FrameOverlay() : ModulePass(&ID)  {}

    virtual void getAnalysisUsage(AnalysisUsage &AU) const; 
    virtual bool runOnModule(Module &M); 
  private: 
    void ColorFunction(CallGraphNode *CGN, unsigned Color);
  };
  char PIC16FrameOverlay::ID = 0;
  static RegisterPass<PIC16FrameOverlay>
  Y("pic16overlay", "PIC16 Frame Overlay Analysis");

}  // End of  namespace

#endif
