//===-- InterferenceGraph.h - Interference graph for register coloring -*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//

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

#ifndef INTERFERENCEGRAPH_H
#define INTERFERENCEGRAPH_H

#include <vector>
class LiveRange;
class RegClass;
class IGNode;

class InterferenceGraph {
  char **IG;                            // a poiner to the interference graph
  unsigned int Size;                    // size of a side of the IG
  RegClass *const RegCl;                // RegCl contains this IG
  std::vector<IGNode *> IGNodeList;     // a list of all IGNodes in a reg class
                            
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

  std::vector<IGNode *> &getIGNodeList() { return IGNodeList; } 
  const std::vector<IGNode *> &getIGNodeList() const { return IGNodeList; } 

  void setCurDegreeOfIGNodes();

  void printIG() const;
  void printIGNodeList() const;
};

#endif
