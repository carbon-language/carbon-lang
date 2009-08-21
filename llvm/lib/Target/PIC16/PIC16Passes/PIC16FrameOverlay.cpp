//===-- PIC16FrameOverlay.cpp - Implementation for PIC16 Frame Overlay===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source 
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the PIC16 Frame Overlay implementation.
//
//===----------------------------------------------------------------------===//


#include "llvm/Analysis/CallGraph.h"
#include "llvm/Pass.h"
#include "llvm/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "PIC16.h"
#include "PIC16FrameOverlay.h"
#include <vector>
#include <iostream>
using namespace llvm;
using std::vector;
using std::string;


void PIC16FrameOverlay::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addRequired<CallGraph>();
}

bool PIC16FrameOverlay::runOnModule(Module &M) {
  CallGraph &CG = getAnalysis<CallGraph>();
  for (CallGraph::iterator it = CG.begin() ; it != CG.end(); it++)
  {
    // External calling node doesn't have any function associated 
    // with it
    if (!it->first)
      continue;

    if (it->second->size() == 0) {
      if (PAN::isInterruptLineFunction(it->first))
        ColorFunction(it->second, PIC16Overlay::GREEN_IL);
      else 
        ColorFunction(it->second, PIC16Overlay::GREEN);
    }
  }
  return false;
}

void PIC16FrameOverlay::ColorFunction(CallGraphNode *CGN, unsigned Color) {
  switch (Color) {
    case PIC16Overlay::GREEN:
    case PIC16Overlay::GREEN_IL: {
      Function *LeafFunct = CGN->getFunction();
      std::string Section = "";
      if (LeafFunct->hasSection()) {
        Section = LeafFunct->getSection();
        Section.append(" ");
      }
      Section.append(PAN::getOverlayStr(Color));
      LeafFunct->setSection(Section);
      break;
    }
    default:
      assert( 0 && "Color not supported");   
  }
}
