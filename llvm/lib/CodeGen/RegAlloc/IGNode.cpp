#include "llvm/CodeGen/IGNode.h"


//-----------------------------------------------------------------------------
// Constructor
//-----------------------------------------------------------------------------
IGNode::IGNode(LiveRange *const PLR, unsigned int Ind): Index(Ind),
							AdjList(),
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
    cout << "\nAdj List size = " << neighs;
    assert(0 && "Invalid adj list size");
  }

  for(int i=0; i < neighs; i++)  (AdjList[i])->decCurDegree();
}
 
//-----------------------------------------------------------------------------
// Deletes an adjacency node. IGNodes are deleted when coalescing merges
// two IGNodes together.
//-----------------------------------------------------------------------------
void IGNode::delAdjIGNode(const IGNode *const Node) {
  vector <IGNode *>::iterator It = AdjList.begin();
    
  // find Node
  for( ; It != AdjList.end() && (*It != Node); It++ ) ;
  assert( It != AdjList.end() );      // the node must be there
  
  AdjList.erase( It );
}
