#include "llvm/CodeGen/IGNode.h"
#include <algorithm>
#include <iostream>
using std::cerr;

//-----------------------------------------------------------------------------
// Constructor
//-----------------------------------------------------------------------------
IGNode::IGNode(LiveRange *const PLR, unsigned int Ind) : Index(Ind),
                                                         ParentLR(PLR)
{
  OnStack = false;
  CurDegree = -1 ;
  ParentLR->setUserIGNode( this );
}


//-----------------------------------------------------------------------------
// Sets this IGNode on stack and reduce the degree of neighbors  
//-----------------------------------------------------------------------------
void IGNode::pushOnStack()             
{                                     
  OnStack = true; 
  int neighs = AdjList.size();

  if( neighs < 0) {
    cerr << "\nAdj List size = " << neighs;
    assert(0 && "Invalid adj list size");
  }

  for(int i=0; i < neighs; i++)
    AdjList[i]->decCurDegree();
}
 
//-----------------------------------------------------------------------------
// Deletes an adjacency node. IGNodes are deleted when coalescing merges
// two IGNodes together.
//-----------------------------------------------------------------------------
void IGNode::delAdjIGNode(const IGNode *const Node) {
  std::vector<IGNode *>::iterator It = 
    find(AdjList.begin(), AdjList.end(), Node);
  assert( It != AdjList.end() );      // the node must be there
    
  AdjList.erase(It);
}
