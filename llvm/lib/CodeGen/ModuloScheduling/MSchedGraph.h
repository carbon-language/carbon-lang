//===-- MSchedGraph.h - Scheduling Graph ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// A graph class for dependencies
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MSCHEDGRAPH_H
#define LLVM_MSCHEDGRAPH_H

#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Target/TargetMachine.h"
#include "Support/GraphTraits.h"
#include "Support/STLExtras.h"
#include "Support/iterator"
#include <vector>

namespace llvm {
  class MSchedGraph;
  class MSchedGraphNode;
  template<class IteratorType, class NodeType>
  class MSchedGraphNodeIterator;


  struct MSchedGraphEdge {
    enum DataDepOrderType {
      TrueDep, AntiDep, OutputDep, NonDataDep
    };
    
    enum MSchedGraphEdgeType {
      MemoryDep, ValueDep, MachineRegister
    };

    MSchedGraphNode *getDest() const { return dest; }
    unsigned getIteDiff() { return iteDiff; }
    unsigned getDepOrderType() { return depOrderType; }

  private:
    friend class MSchedGraphNode;
    MSchedGraphEdge(MSchedGraphNode *destination, MSchedGraphEdgeType type, 
		    unsigned deptype, unsigned diff) 
      : dest(destination), depType(type), depOrderType(deptype), iteDiff(diff) {}
    
    MSchedGraphNode *dest;
    MSchedGraphEdgeType depType;
    unsigned depOrderType;
    unsigned iteDiff;
  };

  class MSchedGraphNode {
   
    const MachineInstr* Inst; //Machine Instruction
    MSchedGraph* Parent; //Graph this node belongs to
    unsigned latency; //Latency of Instruction
    
    std::vector<MSchedGraphNode*> Predecessors; //Predecessor Nodes
    std::vector<MSchedGraphEdge> Successors;

  public:
    MSchedGraphNode(const MachineInstr *inst, MSchedGraph *graph, 
		    unsigned late=0);

    //Iterators
    typedef std::vector<MSchedGraphNode*>::iterator pred_iterator;
    pred_iterator pred_begin() { return Predecessors.begin(); }
    pred_iterator pred_end() { return Predecessors.end(); }
    
    typedef std::vector<MSchedGraphNode*>::const_iterator pred_const_iterator;
    pred_const_iterator pred_begin() const { return Predecessors.begin(); }
    pred_const_iterator pred_end() const { return Predecessors.end(); }

    // Successor iterators.
    typedef MSchedGraphNodeIterator<std::vector<MSchedGraphEdge>::const_iterator,
				    const MSchedGraphNode> succ_const_iterator;
    succ_const_iterator succ_begin() const;
    succ_const_iterator succ_end() const;

    typedef MSchedGraphNodeIterator<std::vector<MSchedGraphEdge>::iterator,
				    MSchedGraphNode> succ_iterator;
    succ_iterator succ_begin();
    succ_iterator succ_end();
    

    void addOutEdge(MSchedGraphNode *destination, 
		    MSchedGraphEdge::MSchedGraphEdgeType type, 
		    unsigned deptype, unsigned diff=0) {
      Successors.push_back(MSchedGraphEdge(destination, type, deptype,diff));
      destination->Predecessors.push_back(this);
    }
    const MachineInstr* getInst() { return Inst; }
    MSchedGraph* getParent() { return Parent; }
    bool hasPredecessors() { return (Predecessors.size() > 0); }
    bool hasSuccessors() { return (Successors.size() > 0); }
    int getLatency() { return latency; }
    MSchedGraphEdge getInEdge(MSchedGraphNode *pred);
    unsigned getInEdgeNum(MSchedGraphNode *pred);

    bool isSuccessor(MSchedGraphNode *);
    bool isPredecessor(MSchedGraphNode *);

    //Debug support
    void print(std::ostream &os) const;

  };

  template<class IteratorType, class NodeType>
  class MSchedGraphNodeIterator : public forward_iterator<NodeType*, ptrdiff_t> {
    IteratorType I;   // std::vector<MSchedGraphEdge>::iterator or const_iterator
  public:
    MSchedGraphNodeIterator(IteratorType i) : I(i) {}

    bool operator==(const MSchedGraphNodeIterator RHS) const { return I == RHS.I; }
    bool operator!=(const MSchedGraphNodeIterator RHS) const { return I != RHS.I; }

    const MSchedGraphNodeIterator &operator=(const MSchedGraphNodeIterator &RHS) {
      I = RHS.I;
      return *this;
    }

    NodeType* operator*() const {
      return I->getDest();
    }
    NodeType* operator->() const { return operator*(); }
    
    MSchedGraphNodeIterator& operator++() {                // Preincrement
      ++I;
      return *this;
    }
    MSchedGraphNodeIterator operator++(int) { // Postincrement
      MSchedGraphNodeIterator tmp = *this; ++*this; return tmp; 
    }

    MSchedGraphEdge &getEdge() {
      return *I;
    }
    const MSchedGraphEdge &getEdge() const {
      return *I;
    }
  };

  inline MSchedGraphNode::succ_const_iterator MSchedGraphNode::succ_begin() const {
    return succ_const_iterator(Successors.begin());
  }
  inline MSchedGraphNode::succ_const_iterator MSchedGraphNode::succ_end() const {
    return succ_const_iterator(Successors.end());
  }
  inline MSchedGraphNode::succ_iterator MSchedGraphNode::succ_begin() {
    return succ_iterator(Successors.begin());
  }
  inline MSchedGraphNode::succ_iterator MSchedGraphNode::succ_end() {
    return succ_iterator(Successors.end());
  }

  // ostream << operator for MSGraphNode class
  inline std::ostream &operator<<(std::ostream &os, 
				  const MSchedGraphNode &node) {
    node.print(os);
    return os;
  }



  class MSchedGraph {
    
    const MachineBasicBlock *BB; //Machine basic block
    const TargetMachine &Target; //Target Machine
        
    //Nodes
    std::map<const MachineInstr*, MSchedGraphNode*> GraphMap;

    //Add Nodes and Edges to this graph for our BB
    typedef std::pair<int, MSchedGraphNode*> OpIndexNodePair;
    void buildNodesAndEdges();
    void addValueEdges(std::vector<OpIndexNodePair> &NodesInMap, 
		       MSchedGraphNode *node,
		       bool nodeIsUse, bool nodeIsDef, int diff=0);
    void addMachRegEdges(std::map<int, 
			 std::vector<OpIndexNodePair> >& regNumtoNodeMap);
    void addMemEdges(const std::vector<MSchedGraphNode*>& memInst);

  public:
    MSchedGraph(const MachineBasicBlock *bb, const TargetMachine &targ);
    ~MSchedGraph();
    
    //Add Nodes to the Graph
    void addNode(const MachineInstr* MI, MSchedGraphNode *node);
    
    //iterators 
    typedef std::map<const MachineInstr*, MSchedGraphNode*>::iterator iterator;
    typedef std::map<const MachineInstr*, MSchedGraphNode*>::const_iterator const_iterator;
    typedef std::map<const MachineInstr*, MSchedGraphNode*>::reverse_iterator reverse_iterator;
    iterator find(const MachineInstr* I) { return GraphMap.find(I); }
    iterator end() { return GraphMap.end(); }
    iterator begin() { return GraphMap.begin(); }
    reverse_iterator rbegin() { return GraphMap.rbegin(); }
    reverse_iterator rend() { return GraphMap.rend(); }
    
  };

  
  static MSchedGraphNode& getSecond(std::pair<const MachineInstr* const,
					 MSchedGraphNode*> &Pair) {
    return *Pair.second;
  }



  // Provide specializations of GraphTraits to be able to use graph
  // iterators on the scheduling graph!
  //
  template <> struct GraphTraits<MSchedGraph*> {
    typedef MSchedGraphNode NodeType;
    typedef MSchedGraphNode::succ_iterator ChildIteratorType;
    
    static inline ChildIteratorType child_begin(NodeType *N) { 
      return N->succ_begin(); 
    }
    static inline ChildIteratorType child_end(NodeType *N) { 
      return N->succ_end();
    }

    typedef std::pointer_to_unary_function<std::pair<const MachineInstr* const,
           MSchedGraphNode*>&, MSchedGraphNode&> DerefFun;

    typedef mapped_iterator<MSchedGraph::iterator, DerefFun> nodes_iterator;
    static nodes_iterator nodes_begin(MSchedGraph *G) {
      return map_iterator(((MSchedGraph*)G)->begin(), DerefFun(getSecond));
    }
    static nodes_iterator nodes_end(MSchedGraph *G) {
      return map_iterator(((MSchedGraph*)G)->end(), DerefFun(getSecond));
    }
    

  };
  
  template <> struct GraphTraits<const MSchedGraph*> {
    typedef const MSchedGraphNode NodeType;
    typedef MSchedGraphNode::succ_const_iterator ChildIteratorType;
    
    static inline ChildIteratorType child_begin(NodeType *N) { 
      return N->succ_begin(); 
    }
    static inline ChildIteratorType child_end(NodeType *N) { 
      return N->succ_end();
    }
    typedef std::pointer_to_unary_function<std::pair<const MachineInstr* const,
						     MSchedGraphNode*>&, MSchedGraphNode&> DerefFun;
    
    typedef mapped_iterator<MSchedGraph::iterator, DerefFun> nodes_iterator;
    static nodes_iterator nodes_begin(MSchedGraph *G) {
      return map_iterator(((MSchedGraph*)G)->begin(), DerefFun(getSecond));
    }
    static nodes_iterator nodes_end(MSchedGraph *G) {
      return map_iterator(((MSchedGraph*)G)->end(), DerefFun(getSecond));
    }
  };
  
  template <> struct GraphTraits<Inverse<MSchedGraph*> > {
    typedef MSchedGraphNode NodeType;
    typedef MSchedGraphNode::pred_iterator ChildIteratorType;
    
    static inline ChildIteratorType child_begin(NodeType *N) { 
      return N->pred_begin();
    }
    static inline ChildIteratorType child_end(NodeType *N) { 
      return N->pred_end();
    }
    typedef std::pointer_to_unary_function<std::pair<const MachineInstr* const,
           MSchedGraphNode*>&, MSchedGraphNode&> DerefFun;

    typedef mapped_iterator<MSchedGraph::iterator, DerefFun> nodes_iterator;
    static nodes_iterator nodes_begin(MSchedGraph *G) {
      return map_iterator(((MSchedGraph*)G)->begin(), DerefFun(getSecond));
    }
    static nodes_iterator nodes_end(MSchedGraph *G) {
      return map_iterator(((MSchedGraph*)G)->end(), DerefFun(getSecond));
    }
  };
  
  template <> struct GraphTraits<Inverse<const MSchedGraph*> > {
    typedef const MSchedGraphNode NodeType;
    typedef MSchedGraphNode::pred_const_iterator ChildIteratorType;
    
    static inline ChildIteratorType child_begin(NodeType *N) { 
      return N->pred_begin();
    }
    static inline ChildIteratorType child_end(NodeType *N) { 
      return N->pred_end();
    }

    typedef std::pointer_to_unary_function<std::pair<const MachineInstr* const,
						     MSchedGraphNode*>&, MSchedGraphNode&> DerefFun;
    
    typedef mapped_iterator<MSchedGraph::iterator, DerefFun> nodes_iterator;
    static nodes_iterator nodes_begin(MSchedGraph *G) {
      return map_iterator(((MSchedGraph*)G)->begin(), DerefFun(getSecond));
    }
    static nodes_iterator nodes_end(MSchedGraph *G) {
      return map_iterator(((MSchedGraph*)G)->end(), DerefFun(getSecond));
    }
  };




}

#endif
