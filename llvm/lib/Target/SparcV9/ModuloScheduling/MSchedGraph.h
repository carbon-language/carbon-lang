//===-- MSchedGraph.h - Scheduling Graph ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// A graph class for dependencies. This graph only contains true, anti, and
// output data dependencies for a given MachineBasicBlock. Dependencies
// across iterations are also computed. Unless data dependence analysis
// is provided, a conservative approach of adding dependencies between all
// loads and stores is taken.
//===----------------------------------------------------------------------===//

#ifndef LLVM_MSCHEDGRAPH_H
#define LLVM_MSCHEDGRAPH_H
#include "DependenceAnalyzer.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetData.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/iterator"
#include <vector>

namespace llvm {

  class MSchedGraph;
  class MSchedGraphNode;
  template<class IteratorType, class NodeType>
  class MSchedGraphNodeIterator;

  //MSchedGraphEdge encapsulates the data dependence between nodes. It
  //identifies the dependence type, on what, and the iteration
  //difference
  struct MSchedGraphEdge {
    enum DataDepOrderType {
      TrueDep, AntiDep, OutputDep, NonDataDep
    };

    enum MSchedGraphEdgeType {
      MemoryDep, ValueDep, MachineRegister, BranchDep
    };

    //Get or set edge data
    MSchedGraphNode *getDest() const { return dest; }
    unsigned getIteDiff() { return iteDiff; }
    unsigned getDepOrderType() { return depOrderType; }
    void setDest(MSchedGraphNode *newDest) { dest = newDest; }

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

  //MSchedGraphNode represents a machine instruction and its
  //corresponding latency. Each node also contains a list of its
  //predecessors and sucessors.
  class MSchedGraphNode {

    const MachineInstr* Inst; //Machine Instruction
    MSchedGraph* Parent; //Graph this node belongs to
    unsigned index; //Index in BB
    unsigned latency; //Latency of Instruction
    bool isBranchInstr; //Is this node the branch instr or not

    std::vector<MSchedGraphNode*> Predecessors; //Predecessor Nodes
    std::vector<MSchedGraphEdge> Successors; //Successor edges

  public:
    MSchedGraphNode(const MachineInstr *inst, MSchedGraph *graph,
		    unsigned index, unsigned late=0, bool isBranch=false);

    MSchedGraphNode(const MSchedGraphNode &N);

    //Iterators - Predecessor and Succussor
    typedef std::vector<MSchedGraphNode*>::iterator pred_iterator;
    pred_iterator pred_begin() { return Predecessors.begin(); }
    pred_iterator pred_end() { return Predecessors.end(); }
    unsigned pred_size() { return Predecessors.size(); }

    typedef std::vector<MSchedGraphNode*>::const_iterator pred_const_iterator;
    pred_const_iterator pred_begin() const { return Predecessors.begin(); }
    pred_const_iterator pred_end() const { return Predecessors.end(); }

    typedef MSchedGraphNodeIterator<std::vector<MSchedGraphEdge>::const_iterator,
				    const MSchedGraphNode> succ_const_iterator;
    succ_const_iterator succ_begin() const;
    succ_const_iterator succ_end() const;

    typedef MSchedGraphNodeIterator<std::vector<MSchedGraphEdge>::iterator,
				    MSchedGraphNode> succ_iterator;
    succ_iterator succ_begin();
    succ_iterator succ_end();
    unsigned succ_size() { return Successors.size(); }

    //Get or set predecessor nodes, or successor edges
    void setPredecessor(unsigned index, MSchedGraphNode *dest) {
      Predecessors[index] = dest;
    }

    MSchedGraphNode* getPredecessor(unsigned index) {
      return Predecessors[index];
    }

    MSchedGraphEdge* getSuccessor(unsigned index) {
      return &Successors[index];
    }

    void deleteSuccessor(MSchedGraphNode *node) {
      for (unsigned i = 0; i != Successors.size(); ++i)
	if (Successors[i].getDest() == node) {
	  Successors.erase(Successors.begin()+i);
	  node->Predecessors.erase(std::find(node->Predecessors.begin(),
					     node->Predecessors.end(), this));
	  --i; //Decrease index var since we deleted a node
	}
    }

    void addOutEdge(MSchedGraphNode *destination,
		    MSchedGraphEdge::MSchedGraphEdgeType type,
		    unsigned deptype, unsigned diff=0) {
      Successors.push_back(MSchedGraphEdge(destination, type, deptype,diff));
      destination->Predecessors.push_back(this);
    }

    //General methods to get and set data for the node
    const MachineInstr* getInst() { return Inst; }
    MSchedGraph* getParent() { return Parent; }
    bool hasPredecessors() { return (Predecessors.size() > 0); }
    bool hasSuccessors() { return (Successors.size() > 0); }
    unsigned getLatency() { return latency; }
    unsigned getLatency() const { return latency; }
    unsigned getIndex() { return index; }
    unsigned getIteDiff(MSchedGraphNode *succ);
    MSchedGraphEdge getInEdge(MSchedGraphNode *pred);
    unsigned getInEdgeNum(MSchedGraphNode *pred);
    bool isSuccessor(MSchedGraphNode *);
    bool isPredecessor(MSchedGraphNode *);
    bool isBranch() { return isBranchInstr; }

    //Debug support
    void print(std::ostream &os) const;
    void setParent(MSchedGraph *p) { Parent = p; }
  };

  //Node iterator for graph generation
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


  // Provide specializations of GraphTraits to be able to use graph
  // iterators on the scheduling graph!
  //
  template <> struct GraphTraits<MSchedGraphNode*> {
    typedef MSchedGraphNode NodeType;
    typedef MSchedGraphNode::succ_iterator ChildIteratorType;

    static inline ChildIteratorType child_begin(NodeType *N) {
      return N->succ_begin();
    }
    static inline ChildIteratorType child_end(NodeType *N) {
      return N->succ_end();
    }

    static NodeType *getEntryNode(NodeType* N) { return N; }
  };



  //Graph class to represent dependence graph
  class MSchedGraph {

    const MachineBasicBlock *BB; //Machine basic block
    const TargetMachine &Target; //Target Machine

    //Nodes
    std::map<const MachineInstr*, MSchedGraphNode*> GraphMap;

    //Add Nodes and Edges to this graph for our BB
    typedef std::pair<int, MSchedGraphNode*> OpIndexNodePair;
    void buildNodesAndEdges(std::map<const MachineInstr*, unsigned> &ignoreInstrs, DependenceAnalyzer &DA, std::map<MachineInstr*, Instruction*> &machineTollvm);
    void addValueEdges(std::vector<OpIndexNodePair> &NodesInMap,
		       MSchedGraphNode *node,
		       bool nodeIsUse, bool nodeIsDef, std::vector<const MachineInstr*> &phiInstrs, int diff=0);
    void addMachRegEdges(std::map<int,
			 std::vector<OpIndexNodePair> >& regNumtoNodeMap);
    void addMemEdges(const std::vector<MSchedGraphNode*>& memInst,
		     DependenceAnalyzer &DA, std::map<MachineInstr*, Instruction*> &machineTollvm);
    void addBranchEdges();

  public:
    MSchedGraph(const MachineBasicBlock *bb, const TargetMachine &targ,
		std::map<const MachineInstr*, unsigned> &ignoreInstrs,
		DependenceAnalyzer &DA, std::map<MachineInstr*, Instruction*> &machineTollvm);

    //Copy constructor with maps to link old nodes to new nodes
    MSchedGraph(const MSchedGraph &G, std::map<MSchedGraphNode*, MSchedGraphNode*> &newNodes);
    
    //Print graph
    void print(std::ostream &os) const;

    //Deconstructor!
    ~MSchedGraph();

    //Add or delete nodes from the Graph
    void addNode(const MachineInstr* MI, MSchedGraphNode *node);
    void deleteNode(MSchedGraphNode *node);
    int totalDelay();

    //iterators
    typedef std::map<const MachineInstr*, MSchedGraphNode*>::iterator iterator;
    typedef std::map<const MachineInstr*, MSchedGraphNode*>::const_iterator const_iterator;
    typedef std::map<const MachineInstr*, MSchedGraphNode*>::reverse_iterator reverse_iterator;
    iterator find(const MachineInstr* I) { return GraphMap.find(I); }
    iterator end() { return GraphMap.end(); }
    iterator begin() { return GraphMap.begin(); }
    unsigned size() { return GraphMap.size(); }
    reverse_iterator rbegin() { return GraphMap.rbegin(); }
    reverse_iterator rend() { return GraphMap.rend(); }

    //Get Target or original machine basic block
    const TargetMachine* getTarget() { return &Target; }
    const MachineBasicBlock* getBB() { return BB; }
  };





  // Provide specializations of GraphTraits to be able to use graph
  // iterators on the scheduling graph
  static MSchedGraphNode& getSecond(std::pair<const MachineInstr* const,
				    MSchedGraphNode*> &Pair) {
    return *Pair.second;
  }

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
