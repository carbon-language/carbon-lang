/* Title:   InterferenceGraph.h
   Author:  Ruchira Sasanka
   Date:    July 20, 01
   Purpose: Interference Graph used for register coloring.

   Notes: 
   Adj Info is  stored in the lower trangular matrix (i.e., row > col ) 

   This class must be used in the following way:

   * Construct class
   * call addLRToIG as many times to add ALL LRs to this IG
   * call createGraph to create the actual matrix
   * Then setInterference, getInterference, mergeIGNodesOfLRs can be 
     called as desired to modify the graph.
   * Once the modifications to the graph are over, call 
     setCurDegreeOfIGNodes() before pushing IGNodes on to stack for coloring.
*/


#ifndef  INTERFERENCE_GRAPH_H
#define  INTERFERENCE_GRAPH_H


#include "llvm/CodeGen/IGNode.h"

typedef vector <IGNode *> IGNodeListType;


class InterferenceGraph
{
  char **IG;                            // a poiner to the interference graph
  unsigned int Size;                    // size of a side of the IG
  RegClass *const RegCl;                // RegCl contains this IG
  IGNodeListType IGNodeList;            // a list of all IGNodes in a reg class
                            
  // for asserting this IG node is infact in the IGNodeList of this class
  inline void assertIGNode(const IGNode *const Node) const {     
    assert( IGNodeList[ Node->getIndex() ] == Node );
  }



 public:

  // the matrix is not yet created by the constructor. Call createGraph() 
  // to create it after adding all IGNodes to the IGNodeList

  InterferenceGraph(RegClass *const RC);
  void createGraph();

  void addLRToIG(LiveRange *const LR);

  void setInterference(const LiveRange *const LR1,
			      const LiveRange *const LR2 );

  unsigned getInterference(const LiveRange *const LR1,
				  const LiveRange *const LR2 ) const ;

  void mergeIGNodesOfLRs(const LiveRange *const LR1, LiveRange *const LR2);

  inline IGNodeListType &getIGNodeList() { return IGNodeList; } 

  void setCurDegreeOfIGNodes();

  void printIG() const;
  void printIGNodeList() const;

  ~InterferenceGraph();
  

};


#endif

