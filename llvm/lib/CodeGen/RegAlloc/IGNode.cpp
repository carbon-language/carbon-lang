#include "llvm/CodeGen/IGNode.h"


IGNode::IGNode(LiveRange *const PLR, unsigned int Ind): Index(Ind),
                                                        ParentLR(PLR)
{
  OnStack = false;
  CurDegree = -1 ;
  ParentLR->setUserIGNode( this );
}



void IGNode::pushOnStack()            // sets on to stack and 
{                                     // reduce the degree of neighbors  
  OnStack = true; 
  unsigned int neighs = AdjList.size();

  for(unsigned int i=0; i < neighs; i++)  (AdjList[i])->decCurDegree();
}
 

void IGNode::delAdjIGNode(const IGNode *const Node) {
  vector <IGNode *>::iterator It = AdjList.begin();
    
  // find Node
  for( ; It != AdjList.end() && (*It != Node); It++ ) ;
  assert( It != AdjList.end() );      // the node must be there
  
  AdjList.erase( It );
}
