/* Title:   RegClass.h   -*- C++ -*-
   Author:  Ruchira Sasanka
   Date:    Aug 20, 01
   Purpose: Contains machine independent methods for register coloring.

*/

#ifndef REG_CLASS_H
#define REG_CLASS_H

#include "llvm/Target/TargetRegInfo.h"
#include "InterferenceGraph.h"
#include <stack>
class TargetRegClassInfo;

typedef std::vector<unsigned> ReservedColorListType;


//-----------------------------------------------------------------------------
// Class RegClass
//
//   Implements a machine independant register class. 
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
  const TargetRegClassInfo *const MRC; // corresponding MRC
  const unsigned RegClassID;            // my int ID

  InterferenceGraph IG;                 // Interference graph - constructed by
                                        // buildInterferenceGraph
  std::stack<IGNode *> IGNodeStack;     // the stack used for coloring

  // ReservedColorList - for passing registers that are pre-allocated and cannot
  // be used by the register allocator for this function.
  //
  const ReservedColorListType *const ReservedColorList;
  
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


 public:

  RegClass(const Function *M,
	   const TargetRegClassInfo *MRC,
	   const ReservedColorListType *RCL = 0);

  inline void createInterferenceGraph() { IG.createGraph(); }

  inline InterferenceGraph &getIG() { return IG; }

  inline const unsigned getID() const { return RegClassID; }

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


  inline std::vector<bool> &getIsColorUsedArr() { return IsColorUsedArr; }


  void printIGNodeList() const;
  void printIG();
};

#endif
