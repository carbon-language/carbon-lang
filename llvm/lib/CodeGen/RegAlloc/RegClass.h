//===-- RegClass.h - Machine Independent register coloring ------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//

/* Title:   RegClass.h   -*- C++ -*-
   Author:  Ruchira Sasanka
   Date:    Aug 20, 01
   Purpose: Contains machine independent methods for register coloring.

*/

#ifndef REGCLASS_H
#define REGCLASS_H

#include "llvm/Target/TargetRegInfo.h"
#include "InterferenceGraph.h"
#include <stack>
class TargetRegClassInfo;


//-----------------------------------------------------------------------------
// Class RegClass
//
//   Implements a machine independent register class. 
//
//   This is the class that contains all data structures and common algos
//   for coloring a particular register class (e.g., int class, fp class).  
//   This class is hardware independent. This class accepts a hardware 
//   dependent description of machine registers (TargetRegInfo class) to 
//   get hardware specific info and to color an individual IG node.
//
//   This class contains the InterferenceGraph (IG).
//   Also it contains an IGNode stack that can be used for coloring. 
//   The class provides some easy access methods to the IG methods, since these
//   methods are called thru a register class.
//
//-----------------------------------------------------------------------------
class RegClass {
  const Function *const Meth;           // Function we are working on
  const TargetRegInfo *MRI;             // Machine register information 
  const TargetRegClassInfo *const MRC;  // Machine reg. class for this RegClass
  const unsigned RegClassID;            // my int ID

  InterferenceGraph IG;                 // Interference graph - constructed by
                                        // buildInterferenceGraph
  std::stack<IGNode *> IGNodeStack;     // the stack used for coloring

  // IsColorUsedArr - An array used for coloring each node. This array must be
  // of size MRC->getNumOfAllRegs(). Allocated once in the constructor for
  // efficiency.
  //
  std::vector<bool> IsColorUsedArr;



  //--------------------------- private methods ------------------------------

  void pushAllIGNodes();

  bool  pushUnconstrainedIGNodes();

  IGNode * getIGNodeWithMinSpillCost();

  void colorIGNode(IGNode *const Node);

  // This directly marks the colors used by a particular register number
  // within the register class.  External users should use the public
  // versions of this function below.
  inline void markColorUsed(unsigned classRegNum) {
    assert(classRegNum < IsColorUsedArr.size() && "Invalid register used?");
    IsColorUsedArr[classRegNum] = true;
  }

  inline bool isColorUsed(unsigned regNum) const {
    assert(regNum < IsColorUsedArr.size() && "Invalid register used?");
    return IsColorUsedArr[regNum];
  }

 public:

  RegClass(const Function *M,
	   const TargetRegInfo *_MRI_,
	   const TargetRegClassInfo *_MRC_);

  inline void createInterferenceGraph() { IG.createGraph(); }

  inline InterferenceGraph &getIG() { return IG; }

  inline const unsigned getID() const { return RegClassID; }

  inline const TargetRegClassInfo* getTargetRegClass() const { return MRC; }

  // main method called for coloring regs
  //
  void colorAllRegs();                 

  inline unsigned getNumOfAvailRegs() const 
    { return MRC->getNumOfAvailRegs(); }


  // --- following methods are provided to access the IG contained within this
  // ---- RegClass easilly.

  inline void addLRToIG(LiveRange *const LR) 
    { IG.addLRToIG(LR); }

  inline void setInterference(const LiveRange *const LR1,
			      const LiveRange *const LR2)  
    { IG.setInterference(LR1, LR2); }

  inline unsigned getInterference(const LiveRange *const LR1,
			      const LiveRange *const LR2) const 
    { return IG.getInterference(LR1, LR2); }

  inline void mergeIGNodesOfLRs(const LiveRange *const LR1,
				LiveRange *const LR2) 
    { IG.mergeIGNodesOfLRs(LR1, LR2); }


  inline void clearColorsUsed() {
    IsColorUsedArr.clear();
    IsColorUsedArr.resize(MRC->getNumOfAllRegs());
  }
  inline void markColorsUsed(unsigned ClassRegNum,
                             int UserRegType,
                             int RegTypeWanted) {
    MRC->markColorsUsed(ClassRegNum, UserRegType, RegTypeWanted,IsColorUsedArr);
  }
  inline int getUnusedColor(int machineRegType) const {
    return MRC->findUnusedColor(machineRegType, IsColorUsedArr);
  }

  void printIGNodeList() const;
  void printIG();
};

#endif
