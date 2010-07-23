//===-- PIC16Overlay.cpp - Implementation for PIC16 Frame Overlay===//
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
#include "llvm/Instructions.h"
#include "llvm/Value.h"
#include "PIC16Overlay.h"
#include "llvm/Function.h"
#include <cstdlib>
#include <sstream>
using namespace llvm;

namespace llvm {
  char PIC16Overlay::ID = 0;
  ModulePass *createPIC16OverlayPass() { return new PIC16Overlay(); }
}

void PIC16Overlay::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<CallGraph>();
}

void PIC16Overlay::DFSTraverse(CallGraphNode *CGN, unsigned Depth) {
  // Do not set any color for external calling node.
  if (Depth != 0 && CGN->getFunction()) {
    unsigned Color = getColor(CGN->getFunction());

    // Handle indirectly called functions
    if (Color >= PIC16OVERLAY::StartIndirectCallColor || 
        Depth >= PIC16OVERLAY::StartIndirectCallColor) {
      // All functions called from an indirectly called function are given
      // an unique color.
      if (Color < PIC16OVERLAY::StartIndirectCallColor &&
          Depth >= PIC16OVERLAY::StartIndirectCallColor)
        setColor(CGN->getFunction(), Depth);

      for (unsigned int i = 0; i < CGN->size(); i++)
        DFSTraverse((*CGN)[i], ++IndirectCallColor);
      return;
    }
    // Just return if the node already has a color greater than the current 
    // depth. A node must be colored with the maximum depth that it has.
    if (Color >= Depth)
      return;
    
    Depth = ModifyDepthForInterrupt(CGN, Depth);  
    setColor(CGN->getFunction(), Depth);
  }
  
  // Color all children of this node with color depth+1.
  for (unsigned int i = 0; i < CGN->size(); i++)
    DFSTraverse((*CGN)[i], Depth+1);
}

unsigned PIC16Overlay::ModifyDepthForInterrupt(CallGraphNode *CGN,
                                                    unsigned Depth) {
  Function *Fn = CGN->getFunction();

  // Return original Depth if function or section for function do not exist.
  if (!Fn || !Fn->hasSection())
    return Depth;

  // Return original Depth if this function is not marked as interrupt.
  if (Fn->getSection().find("interrupt") == string::npos)
    return Depth;

  Depth = Depth + InterruptDepth;
  return Depth;
}

void PIC16Overlay::setColor(Function *Fn, unsigned Color) {
  std::string Section = "";
  if (Fn->hasSection())
    Section = Fn->getSection();

  size_t Pos = Section.find(OverlayStr);

  // Convert Color to string.
  std::stringstream ss;
  ss << Color;
  std::string ColorString = ss.str();

  // If color is already set then reset it with the new value. Else append 
  // the Color string to section.
  if (Pos != std::string::npos) {
    Pos += OverlayStr.length();
    char c = Section.at(Pos);
    unsigned OldColorLength = 0;  
    while (c >= '0' && c<= '9') {
      OldColorLength++;    
      if (Pos < Section.length() - 1)
        Pos++;
      else
        break;
      c = Section.at(Pos);
    }
    // Replace old color with new one.
    Section.replace(Pos-OldColorLength +1, OldColorLength, ColorString); 
  }
  else {
    // Append Color information to section string.
    if (Fn->hasSection())
      Section.append(" ");
    Section.append(OverlayStr + ColorString);
  }
  Fn->setSection(Section);
}

unsigned PIC16Overlay::getColor(Function *Fn) {
  int Color = 0;
  if (!Fn->hasSection())
    return 0;

  std::string Section = Fn->getSection();
  size_t Pos = Section.find(OverlayStr);
  
  // Return 0 if Color is not set.
  if (Pos == std::string::npos)
    return 0;

  // Set Pos to after "Overlay=".
  Pos += OverlayStr.length();
  char c = Section.at(Pos);
  std::string ColorString = "";

  // Find the string representing Color. A Color can only consist of digits.
  while (c >= '0' && c<= '9') { 
    ColorString.append(1,c);
    if (Pos < Section.length() - 1)
      Pos++;
    else
      break;
    c = Section.at(Pos);
  }
  Color = atoi(ColorString.c_str());
  
  return Color;    
}

bool PIC16Overlay::runOnModule(Module &M) {
  CallGraph &CG = getAnalysis<CallGraph>();
  CallGraphNode *ECN = CG.getExternalCallingNode();

  MarkIndirectlyCalledFunctions(M); 
  // Since External Calling Node is the base function, do a depth first 
  // traversal of CallGraph with ECN as root. Each node with be marked with 
  // a color that is max(color(callers)) + 1.
  if(ECN) {
    DFSTraverse(ECN, 0);
  }
  return false;
}

void PIC16Overlay::MarkIndirectlyCalledFunctions(Module &M) {
  // If the use of a function is not a call instruction then this
  // function might be called indirectly. In that case give it
  // an unique color.
  for (Module::iterator MI = M.begin(), E = M.end(); MI != E; ++MI) {
    for (Value::use_iterator I = MI->use_begin(), E = MI->use_end(); I != E;
         ++I) {
      User *U = *I;
      if ((!isa<CallInst>(U) && !isa<InvokeInst>(U))
          || !CallSite(cast<Instruction>(U)).isCallee(I)) {
        setColor(MI, ++IndirectCallColor);
        break;
      }
    }
  }
}
