/* Title:   RegClass.h   -*- C++ -*-
   Author:  Ruchira Sasanka
   Date:    Aug 20, 01
   Purpose: Contains machine independent methods for register coloring.

   This is the class that contains all data structures and common algos
   for coloring a particular register class (e.g., int class, fp class).  
   This class is hardware independent. This class accepts a hardware 
   dependent description of machine registers (MachineRegInfo class) to 
   get hardware specific info and color and indidual IG node.

   This class contains the InterferenceGraph (IG).
   Also it contains an IGNode stack that can be used for coloring. 
   The class provides some easy access methods to the IG methods, since these
   methods are called thru a register class.

*/

#ifndef REG_CLASS_H
#define REG_CLASS_H

#include "llvm/CodeGen/IGNode.h"
#include "llvm/CodeGen/InterferenceGraph.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/MachineRegInfo.h"
#include <stack>

typedef vector<unsigned int> ReservedColorListType;


class RegClass
{

 private:
  const Method *const Meth;             // Method we are working on

  const MachineRegClassInfo *const MRC; // corresponding MRC

  const unsigned RegClassID;            // my int ID

  InterferenceGraph IG;                 // Interference graph - constructed by
                                        // buildInterferenceGraph
  stack <IGNode *> IGNodeStack;         // the stack used for coloring

  // for passing registered that are pre-allocated (e.g., %g's)
  const ReservedColorListType *const ReservedColorList;

  // An array used for coloring each node. This array must be of size 
  // MRC->getNumOfAllRegs(). Allocated once in the constructor
  // for efficiency.
  bool *IsColorUsedArr;


  //------------ private methods ------------------

  void pushAllIGNodes();
  bool  pushUnconstrainedIGNodes();
  IGNode * getIGNodeWithMinSpillCost();
  void colorIGNode(IGNode *const Node);

 public:

  RegClass(const Method *const M, 
	   const MachineRegClassInfo *const MRC, 
	   const ReservedColorListType *const RCL = NULL);

  inline void createInterferenceGraph() 
    { IG.createGraph(); }

  inline InterferenceGraph &getIG() { return IG; }

  inline const unsigned getID() const { return RegClassID; }

  void colorAllRegs();                  // main method called for coloring regs

  inline unsigned getNumOfAvailRegs() const 
    { return MRC->getNumOfAvailRegs(); }

  ~RegClass() { delete[] IsColorUsedArr; };



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


  inline void printIGNodeList() const {
    cout << "IG Nodes for Register Class " << RegClassID << ":" << endl;
    IG.printIGNodeList(); 
  }

  inline void printIG() {  
    cout << "IG for Register Class " << RegClassID << ":" << endl;
    IG.printIG(); 
  }

};







#endif
