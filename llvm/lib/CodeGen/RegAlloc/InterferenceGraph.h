/* Title:   InterferenceGraph.h   -*- C++ -*-
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

typedef std::vector <IGNode *> IGNodeListType;


class InterferenceGraph {
  char **IG;                            // a poiner to the interference graph
  unsigned int Size;                    // size of a side of the IG
  RegClass *const RegCl;                // RegCl contains this IG
  IGNodeListType IGNodeList;            // a list of all IGNodes in a reg class
                            
 public:
  // the matrix is not yet created by the constructor. Call createGraph() 
  // to create it after adding all IGNodes to the IGNodeList
  InterferenceGraph(RegClass *RC);
  ~InterferenceGraph();

  void createGraph();

  void addLRToIG(LiveRange *LR);

  void setInterference(const LiveRange *LR1,
                       const LiveRange *LR2);

  unsigned getInterference(const LiveRange *LR1,
                           const LiveRange *LR2) const ;

  void mergeIGNodesOfLRs(const LiveRange *LR1, LiveRange *LR2);

  IGNodeListType &getIGNodeList() { return IGNodeList; } 
  const IGNodeListType &getIGNodeList() const { return IGNodeList; } 

  void setCurDegreeOfIGNodes();

  void printIG() const;
  void printIGNodeList() const;
};

#endif
