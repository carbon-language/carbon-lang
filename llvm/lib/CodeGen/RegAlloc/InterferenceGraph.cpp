//===-- InterferenceGraph.cpp ---------------------------------------------===//
// 
//  Interference graph for coloring-based register allocation for LLVM.
// 
//===----------------------------------------------------------------------===//

#include "RegAllocCommon.h"
#include "InterferenceGraph.h"
#include "llvm/CodeGen/IGNode.h"
#include "Support/STLExtras.h"
#include <algorithm>
using std::cerr;

// for asserting this IG node is infact in the IGNodeList of this class
inline static void assertIGNode(const InterferenceGraph *IG,
                                const IGNode *Node) {
  assert(IG->getIGNodeList()[Node->getIndex()] == Node);
}

//-----------------------------------------------------------------------------
// Constructor: Records the RegClass and initalizes IGNodeList.
// The matrix is NOT yet created by the constructor. Call createGraph() 
// to create it after adding all IGNodes to the IGNodeList.
//-----------------------------------------------------------------------------
InterferenceGraph::InterferenceGraph(RegClass *const RC) : RegCl(RC), 
							   IGNodeList() 
{   
  IG = NULL;         
  Size = 0;            
  if( DEBUG_RA >= RA_DEBUG_Interference) {
    cerr << "Interference graph created!\n";
  }
}


//-----------------------------------------------------------------------------
// destructor. Deletes the bit matrix and all IGNodes
//-----------------------------------------------------------------------------
InterferenceGraph:: ~InterferenceGraph() {             

  // delete the matrix
  for(unsigned int r=0; r < IGNodeList.size(); ++r)
    delete[] IG[r];
  delete[] IG;

  // delete all IGNodes in the IGNodeList
  for_each(IGNodeList.begin(), IGNodeList.end(), deleter<IGNode>);
}



//-----------------------------------------------------------------------------
// Creates (dynamically allocates) the bit matrix necessary to hold the
// interference graph.
//-----------------------------------------------------------------------------
void InterferenceGraph::createGraph()   
{ 
    Size = IGNodeList.size();
    IG = new char*[Size]; 
    for( unsigned int r=0; r < Size; ++r)
      IG[r] = new char[Size];

    // init IG matrix
    for(unsigned int i=0; i < Size; i++)     
      for(unsigned int j=0; j < Size; j++)
	IG[i][j] = 0;
}

//-----------------------------------------------------------------------------
// creates a new IGNode for the given live range and add to IG
//-----------------------------------------------------------------------------
void InterferenceGraph::addLRToIG(LiveRange *const LR)
{
  IGNodeList.push_back(new IGNode(LR, IGNodeList.size()));
}


//-----------------------------------------------------------------------------
// set interference for two live ranges
// update both the matrix and AdjLists of nodes.
// If there is already an interference between LR1 and LR2, adj lists
// are not updated. LR1 and LR2 must be distinct since if not, it suggests
// that there is some wrong logic in some other method.
//-----------------------------------------------------------------------------
void InterferenceGraph::setInterference(const LiveRange *const LR1,
					const LiveRange *const LR2 ) {
  assert(LR1 != LR2);   

  IGNode *IGNode1 = LR1->getUserIGNode();
  IGNode *IGNode2 = LR2->getUserIGNode();

  assertIGNode(this, IGNode1);   
  assertIGNode(this, IGNode2);
  
  unsigned row = IGNode1->getIndex();
  unsigned col = IGNode2->getIndex();

  char *val;

  if( DEBUG_RA >= RA_DEBUG_Interference) 
    cerr << "setting intf for: [" << row << "][" <<  col << "]\n"; 

  ( row > col) ?  val = &IG[row][col]: val = &IG[col][row]; 

  if( ! (*val) ) {                      // if this interf is not previously set
    *val = 1;                           // add edges between nodes 
    IGNode1->addAdjIGNode( IGNode2 );   
    IGNode2->addAdjIGNode( IGNode1 );
  }

}


//----------------------------------------------------------------------------
// return whether two live ranges interfere
//----------------------------------------------------------------------------
unsigned InterferenceGraph::getInterference(const LiveRange *const LR1,
                                            const LiveRange *const LR2) const {
  assert(LR1 != LR2);
  assertIGNode(this, LR1->getUserIGNode());  
  assertIGNode(this, LR2->getUserIGNode());

  const unsigned int row = LR1->getUserIGNode()->getIndex();
  const unsigned int col = LR2->getUserIGNode()->getIndex();

  char ret; 
  if (row > col)
    ret = IG[row][col];
  else 
    ret = IG[col][row]; 
  return ret;

}


//----------------------------------------------------------------------------
// Merge 2 IGNodes. The neighbors of the SrcNode will be added to the DestNode.
// Then the IGNode2L  will be deleted. Necessary for coalescing.
// IMPORTANT: The live ranges are NOT merged by this method. Use 
//            LiveRangeInfo::unionAndUpdateLRs for that purpose.
//----------------------------------------------------------------------------

void InterferenceGraph::mergeIGNodesOfLRs(const LiveRange *LR1, 
					  LiveRange *LR2) {

  assert( LR1 != LR2);                  // cannot merge the same live range

  IGNode *const DestNode = LR1->getUserIGNode();
  IGNode *SrcNode = LR2->getUserIGNode();

  assertIGNode(this, DestNode);
  assertIGNode(this, SrcNode);

  if( DEBUG_RA >= RA_DEBUG_Interference) {
    cerr << "Merging LRs: \""; printSet(*LR1); 
    cerr << "\" and \""; printSet(*LR2);
    cerr << "\"\n";
  }

  unsigned SrcDegree = SrcNode->getNumOfNeighbors();
  const unsigned SrcInd = SrcNode->getIndex();


  // for all neighs of SrcNode
  for(unsigned i=0; i < SrcDegree; i++) {        
    IGNode *NeighNode = SrcNode->getAdjIGNode(i); 

    LiveRange *const LROfNeigh = NeighNode->getParentLR();

    // delete edge between src and neigh - even neigh == dest
    NeighNode->delAdjIGNode(SrcNode);  

    // set the matrix posn to 0 betn src and neigh - even neigh == dest
    const unsigned NInd = NeighNode->getIndex();
    ( SrcInd > NInd) ?  (IG[SrcInd][NInd]=0) : (IG[NInd][SrcInd]=0) ; 


    if( LR1 != LROfNeigh) {             // if the neigh != dest 
     
      // add edge betwn Dest and Neigh - if there is no current edge
      setInterference(LR1, LROfNeigh );  
    }
    
  }

  IGNodeList[ SrcInd ] = NULL;

  // SrcNode is no longer necessary - LR2 must be deleted by the caller
  delete( SrcNode );    

}


//----------------------------------------------------------------------------
// must be called after modifications to the graph are over but before
// pushing IGNodes on to the stack for coloring.
//----------------------------------------------------------------------------
void InterferenceGraph::setCurDegreeOfIGNodes()
{
  unsigned Size = IGNodeList.size();

  for( unsigned i=0; i < Size; i++) {
    IGNode *Node = IGNodeList[i];
    if( Node )
      Node->setCurDegree();
  }
}





//--------------------- debugging (Printing) methods -----------------------

//----------------------------------------------------------------------------
// Print the IGnodes 
//----------------------------------------------------------------------------
void InterferenceGraph::printIG() const
{

  for(unsigned int i=0; i < Size; i++) {   

    const IGNode *const Node = IGNodeList[i];
    if(Node) {
      cerr << " [" << i << "] ";

      for( unsigned int j=0; j < Size; j++) {
	if(IG[i][j])
          cerr << "(" << i << "," << j << ") ";
      }
      cerr << "\n";
    }
  }
}

//----------------------------------------------------------------------------
// Print the IGnodes in the IGNode List
//----------------------------------------------------------------------------
void InterferenceGraph::printIGNodeList() const
{
  for(unsigned i=0; i < IGNodeList.size() ; ++i) {
    const IGNode *const Node = IGNodeList[i];

    if (Node) {
      cerr << " [" << Node->getIndex() << "] ";
      printSet(*Node->getParentLR());
      //int Deg = Node->getCurDegree();
      cerr << "\t <# of Neighs: " << Node->getNumOfNeighbors() << ">\n";
    }
  }
}
