//===-- RegClass.cpp -----------------------------------------------------===//
// 
//  class RegClass for coloring-based register allocation for LLVM.
// 
//===----------------------------------------------------------------------===//

#include "RegClass.h"
#include "RegAllocCommon.h"
#include "llvm/CodeGen/IGNode.h"
#include "llvm/Target/TargetRegInfo.h"
using std::cerr;

//----------------------------------------------------------------------------
// This constructor inits IG. The actual matrix is created by a call to 
// createInterferenceGraph() above.
//----------------------------------------------------------------------------
RegClass::RegClass(const Function *M, 
                   const TargetRegInfo *_MRI_,
		   const TargetRegClassInfo *_MRC_)
                  :  Meth(M), MRI(_MRI_), MRC(_MRC_),
                     RegClassID( _MRC_->getRegClassID() ),
                     IG(this), IGNodeStack() {
  if( DEBUG_RA >= RA_DEBUG_Interference)
    cerr << "Created Reg Class: " << RegClassID << "\n";

  IsColorUsedArr.resize(MRC->getNumOfAllRegs());
}



//----------------------------------------------------------------------------
// Main entry point for coloring a register class.
//----------------------------------------------------------------------------
void RegClass::colorAllRegs()
{
  if(DEBUG_RA >= RA_DEBUG_Coloring)
    cerr << "Coloring IG of reg class " << RegClassID << " ...\n";

                                        // pre-color IGNodes
  pushAllIGNodes();                     // push all IG Nodes

  unsigned int StackSize = IGNodeStack.size();    
  IGNode *CurIGNode;

                                        // for all LRs on stack
  for( unsigned int IGN=0; IGN < StackSize; IGN++) {  
  
    CurIGNode = IGNodeStack.top();      // pop the IGNode on top of stack
    IGNodeStack.pop();
    colorIGNode (CurIGNode);            // color it
  }

}



//----------------------------------------------------------------------------
// The method for pushing all IGNodes on to the stack.
//----------------------------------------------------------------------------
void RegClass::pushAllIGNodes()
{
  bool NeedMoreSpills;          


  IG.setCurDegreeOfIGNodes();           // calculate degree of IGNodes

                                        // push non-constrained IGNodes
  bool PushedAll  = pushUnconstrainedIGNodes(); 

  if( DEBUG_RA >= RA_DEBUG_Coloring) {
    cerr << " Puhsed all-unconstrained IGNodes. ";
    if( PushedAll ) cerr << " No constrained nodes left.";
    cerr << "\n";
  }

  if( PushedAll )                       // if NO constrained nodes left
    return;


  // now, we have constrained nodes. So, push one of them (the one with min 
  // spill cost) and try to push the others as unConstrained nodes. 
  // Repeat this.

  do {
    //get node with min spill cost
    //
    IGNode *IGNodeSpill =  getIGNodeWithMinSpillCost(); 
   
    //  push that node on to stack
    //
    IGNodeStack.push(IGNodeSpill);

    // set its OnStack flag and decrement degree of neighs 
    //
    IGNodeSpill->pushOnStack(); 
   
    // now push NON-constrined ones, if any
    //
    NeedMoreSpills = !pushUnconstrainedIGNodes(); 

    if (DEBUG_RA >= RA_DEBUG_Coloring)
      cerr << "\nConstrained IG Node found !@!" << IGNodeSpill->getIndex();

  } while(NeedMoreSpills);            // repeat until we have pushed all 

}




//--------------------------------------------------------------------------
// This method goes thru all IG nodes in the IGNodeList of an IG of a 
// register class and push any unconstrained IG node left (that is not
// already pushed)
//--------------------------------------------------------------------------

bool  RegClass::pushUnconstrainedIGNodes()  
{
  // # of LRs for this reg class 
  unsigned int IGNodeListSize = IG.getIGNodeList().size(); 
  bool pushedall = true;

  // a pass over IGNodeList
  for( unsigned i =0; i  < IGNodeListSize; i++) {

    // get IGNode i from IGNodeList
    IGNode *IGNode = IG.getIGNodeList()[i]; 

    if( !IGNode )                        // can be null due to merging   
      continue;
    
    // if already pushed on stack, continue. This can happen since this
    // method can be called repeatedly until all constrained nodes are
    // pushed
    if( IGNode->isOnStack() )
      continue;
                                        // if the degree of IGNode is lower
    if( (unsigned) IGNode->getCurDegree()  < MRC->getNumOfAvailRegs()) {
      IGNodeStack.push( IGNode );       // push IGNode on to the stack
      IGNode->pushOnStack();            // set OnStack and dec deg of neighs

      if (DEBUG_RA >= RA_DEBUG_Coloring) {
	cerr << " pushed un-constrained IGNode " << IGNode->getIndex() ;
	cerr << " on to stack\n";
      }
    }
    else pushedall = false;             // we didn't push all live ranges
    
  } // for
  
  // returns true if we pushed all live ranges - else false
  return pushedall; 
}



//----------------------------------------------------------------------------
// Get the IGNode withe the minimum spill cost
//----------------------------------------------------------------------------
IGNode * RegClass::getIGNodeWithMinSpillCost()
{

  unsigned int IGNodeListSize = IG.getIGNodeList().size(); 
  double MinSpillCost = 0;
  IGNode *MinCostIGNode = NULL;
  bool isFirstNode = true;

  // pass over IGNodeList to find the IGNode with minimum spill cost
  // among all IGNodes that are not yet pushed on to the stack
  //
  for( unsigned int i =0; i  < IGNodeListSize; i++) {
    IGNode *IGNode = IG.getIGNodeList()[i];
    
    if( ! IGNode )                      // can be null due to merging
      continue;

    if( ! IGNode->isOnStack() ) {

      double SpillCost = (double) IGNode->getParentLR()->getSpillCost() /
	(double) (IGNode->getCurDegree() + 1);
    
      if( isFirstNode ) {         // for the first IG node
	MinSpillCost = SpillCost;
	MinCostIGNode = IGNode;
	isFirstNode = false;
      }

      else if( MinSpillCost > SpillCost) {
	MinSpillCost = SpillCost;
	MinCostIGNode = IGNode;
      }

    }
  }
  
  assert( MinCostIGNode && "No IGNode to spill");
  return MinCostIGNode;
}



//----------------------------------------------------------------------------
// Color the IGNode using the machine specific code.
//----------------------------------------------------------------------------
void RegClass::colorIGNode(IGNode *const Node)
{

  if( ! Node->hasColor() )   {          // not colored as an arg etc.
   
    // init all elements of to  IsColorUsedAr  false;
    clearColorsUsed();

    // initialize all colors used by neighbors of this node to true
    LiveRange *LR = Node->getParentLR();
    unsigned NumNeighbors =  Node->getNumOfNeighbors();
    for (unsigned n=0; n < NumNeighbors; n++) {
      IGNode *NeighIGNode = Node->getAdjIGNode(n);
      LiveRange *NeighLR = NeighIGNode->getParentLR();
      
      // Don't use a color if it is in use by the neighbour,
      // or is suggested for use by the neighbour,
      // markColorsUsed() should be given the color and the reg type for
      // LR, not for NeighLR, because it should mark registers used based on
      // the type we are looking for, not on the regType for the neighbour.
      if (NeighLR->hasColor())
        this->markColorsUsed(NeighLR->getColor(),
                             MRI->getRegTypeForLR(NeighLR),
                             MRI->getRegTypeForLR(LR));  // use LR, not NeighLR
      else if (NeighLR->hasSuggestedColor() &&
               NeighLR->isSuggestedColorUsable())
        this->markColorsUsed(NeighLR->getSuggestedColor(),
                             MRI->getRegTypeForLR(NeighLR),
                             MRI->getRegTypeForLR(LR));  // use LR, not NeighLR
    }

    // call the target specific code for coloring
    //
    MRC->colorIGNode(Node, IsColorUsedArr);
  }
  else {
    if( DEBUG_RA >= RA_DEBUG_Coloring) {
      cerr << " Node " << Node->getIndex();
      cerr << " already colored with color " << Node->getColor() << "\n";
    }
  }


  if( !Node->hasColor() ) {
    if( DEBUG_RA >= RA_DEBUG_Coloring) {
      cerr << " Node " << Node->getIndex();
      cerr << " - could not find a color (needs spilling)\n";
    }
  }

}

void RegClass::printIGNodeList() const {
  std::cerr << "IG Nodes for Register Class " << RegClassID << ":" << "\n";
  IG.printIGNodeList(); 
}

void RegClass::printIG() {  
  std::cerr << "IG for Register Class " << RegClassID << ":" << "\n";
  IG.printIG(); 
}


