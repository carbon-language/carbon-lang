#include "llvm/CodeGen/RegClass.h"



RegClass::RegClass(const Method *const M, 
		   const MachineRegClassInfo *const Mrc, 
		   const ReservedColorListType *const RCL)
                  :  Meth(M), MRC(Mrc), RegClassID( Mrc->getRegClassID() ),
                     IG(this), IGNodeStack(), ReservedColorList(RCL)
{
  if( DEBUG_RA)
    cout << "Created Reg Class: " << RegClassID << endl;

  // This constructor inits IG. The actual matrix is created by a call to 
  // createInterferenceGraph() above.

  IsColorUsedArr = new bool[ Mrc->getNumOfAllRegs() ];
}



void RegClass::colorAllRegs()
{
  if(DEBUG_RA) cout << "Coloring IG of reg class " << RegClassID << " ...\n";

  //preColorIGNodes();                    // pre-color IGNodes
  pushAllIGNodes();                     // push all IG Nodes

  unsigned int StackSize = IGNodeStack.size();    
  IGNode *CurIGNode;

  // for all LRs on stack
  for( unsigned int IGN=0; IGN < StackSize; IGN++) {  
  
    CurIGNode = IGNodeStack.top();      // pop the IGNode on top of stack
    IGNodeStack.pop();
    colorIGNode (CurIGNode);            // color it
  }


  // InsertSpillCode;  ********* TODO ********

}



void RegClass::pushAllIGNodes()
{
  bool NeedMoreSpills;          


  IG.setCurDegreeOfIGNodes();           // calculate degree of IGNodes

  // push non-constrained IGNodes
  bool PushedAll  = pushUnconstrainedIGNodes(); 

  if( DEBUG_RA) {
    cout << " Puhsed all-unconstrained IGNodes. ";
    if( PushedAll ) cout << " No constrained nodes left.";
    cout << endl;
  }

  if( PushedAll )                       // if NO constrained nodes left
    return;


  // now, we have constrained nodes. So, push one of them (the one with min 
  // spill cost) and try to push the others as unConstrained nodes. 
  // Repeat this.

  do{

    //get node with min spill cost
    IGNode *IGNodeSpill =  getIGNodeWithMinSpillCost(); 
   
    //  push that node on to stack
    IGNodeStack.push( IGNodeSpill ); 

    // set its OnStack flag and decrement degree of neighs 
    IGNodeSpill->pushOnStack(); 
   
    // now push NON-constrined ones, if any
    NeedMoreSpills = ! pushUnconstrainedIGNodes(); 

    cerr << "\nConstrained IG Node found !@!" <<  IGNodeSpill->getIndex();

  } while( NeedMoreSpills );            // repeat until we have pushed all 

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
    if( (unsigned) IGNode->getCurDegree()  < MRC->getNumOfAvailRegs() ) {   
      IGNodeStack.push( IGNode );       // push IGNode on to the stack
      IGNode->pushOnStack();            // set OnStack and dec deg of neighs

      if (DEBUG_RA > 1) {
	cout << " pushed un-constrained IGNode " << IGNode->getIndex() ;
	cout << " on to stack" << endl;
      }
    }
    else pushedall = false;             // we didn't push all live ranges
    
  } // for
  
  // returns true if we pushed all live ranges - else false
  return pushedall; 
}




IGNode * RegClass::getIGNodeWithMinSpillCost()
{
  IGNode *IGNode=NULL;
  unsigned int IGNodeListSize = IG.getIGNodeList().size(); 

  // pass over IGNodeList
  for( unsigned int i =0; i  < IGNodeListSize; i++) {
    IGNode = IG.getIGNodeList()[i];
    
    if( ! IGNode )                      // can be null due to merging
      continue;
    
    // return the first IGNode ########## Change this #######
    if( ! IGNode->isOnStack() ) return IGNode;   
  }
  
  assert(0);
  return NULL;
}




void RegClass::colorIGNode(IGNode *const Node)
{

  if( ! Node->hasColor() )   {          // not colored as an arg etc.
   

    // init all elements to  false;
    for( unsigned  i=0; i < MRC->getNumOfAllRegs(); i++) { 
      IsColorUsedArr[ i ] = false;
    }
    
    // init all reserved_regs to true - we can't use them
    for( unsigned i=0; i < ReservedColorList->size() ; i++) {  
      IsColorUsedArr[ (*ReservedColorList)[i] ] = true;
    }

    MRC->colorIGNode(Node, IsColorUsedArr);
  }
  else {
    if( DEBUG_RA ) {
      cout << " Node " << Node->getIndex();
      cout << " already colored with color " << Node->getColor() << endl;
    }
  }


  if( !Node->hasColor() ) {
    if( DEBUG_RA ) {
      cout << " Node " << Node->getIndex();
      cout << " - could not find a color (needs spilling)" << endl;
    }
  }

}



