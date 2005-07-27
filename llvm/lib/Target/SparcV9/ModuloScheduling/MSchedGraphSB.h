//===-- MSchedGraphSB.h - Scheduling Graph ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// A graph class for dependencies. This graph only contains true, anti, and
// output data dependencies for a vector of MachineBasicBlock. Dependencies
// across iterations are also computed. Unless data dependence analysis
// is provided, a conservative approach of adding dependencies between all
// loads and stores is taken. It also includes pseudo predicate nodes for
// modulo scheduling superblocks.
//===----------------------------------------------------------------------===//

#ifndef LLVM_MSCHEDGRAPHSB_H
#define LLVM_MSCHEDGRAPHSB_H
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

  class MSchedGraphSB;
  class MSchedGraphSBNode;
  template<class IteratorType, class NodeType>
  class MSchedGraphSBNodeIterator;

  //MSchedGraphSBEdge encapsulates the data dependence between nodes. It
  //identifies the dependence type, on what, and the iteration
  //difference
  struct MSchedGraphSBEdge {
    enum DataDepOrderType {
      TrueDep, AntiDep, OutputDep, NonDataDep
    };

    enum MSchedGraphSBEdgeType {
      MemoryDep, ValueDep, MachineRegister, PredDep
    };

    //Get or set edge data
    MSchedGraphSBNode *getDest() const { return dest; }
    unsigned getIteDiff() { return iteDiff; }
    unsigned getDepOrderType() { return depOrderType; }
    void setDest(MSchedGraphSBNode *newDest) { dest = newDest; }

  private:
    friend class MSchedGraphSBNode;
    MSchedGraphSBEdge(MSchedGraphSBNode *destination, MSchedGraphSBEdgeType type,
                    unsigned deptype, unsigned diff)
      : dest(destination), depType(type), depOrderType(deptype), iteDiff(diff) {}

    MSchedGraphSBNode *dest;
    MSchedGraphSBEdgeType depType;
    unsigned depOrderType;
    unsigned iteDiff;
  };

  //MSchedGraphSBNode represents a machine instruction and its
  //corresponding latency. Each node also contains a list of its
  //predecessors and sucessors.
  class MSchedGraphSBNode {

    const MachineInstr* Inst; //Machine Instruction
    std::vector<const MachineInstr*> otherInstrs;

    MSchedGraphSB* Parent; //Graph this node belongs to
    unsigned index; //Index in BB
    unsigned latency; //Latency of Instruction
    bool isBranchInstr; //Is this node the branch instr or not
    bool isPredicateNode; //Indicate if this node should be treated like a predicate

    std::vector<MSchedGraphSBNode*> Predecessors; //Predecessor Nodes
    std::vector<MSchedGraphSBEdge> Successors; //Successor edges

  public:
    MSchedGraphSBNode(const MachineInstr* inst, MSchedGraphSB *graph,
                    unsigned index, unsigned late=0, bool isBranch=false);
    MSchedGraphSBNode(const MachineInstr* inst, std::vector<const MachineInstr*> &other,
                      MSchedGraphSB *graph,
                      unsigned index, unsigned late=0, bool isPNode=true);
    MSchedGraphSBNode(const MSchedGraphSBNode &N);

    //Iterators - Predecessor and Succussor
    typedef std::vector<MSchedGraphSBNode*>::iterator pred_iterator;
    pred_iterator pred_begin() { return Predecessors.begin(); }
    pred_iterator pred_end() { return Predecessors.end(); }
    unsigned pred_size() { return Predecessors.size(); }

    typedef std::vector<MSchedGraphSBNode*>::const_iterator pred_const_iterator;
    pred_const_iterator pred_begin() const { return Predecessors.begin(); }
    pred_const_iterator pred_end() const { return Predecessors.end(); }

    typedef MSchedGraphSBNodeIterator<std::vector<MSchedGraphSBEdge>::const_iterator,
                                    const MSchedGraphSBNode> succ_const_iterator;
    succ_const_iterator succ_begin() const;
    succ_const_iterator succ_end() const;

    typedef MSchedGraphSBNodeIterator<std::vector<MSchedGraphSBEdge>::iterator,
                                    MSchedGraphSBNode> succ_iterator;
    succ_iterator succ_begin();
    succ_iterator succ_end();
    unsigned succ_size() { return Successors.size(); }

    //Get or set predecessor nodes, or successor edges
    void setPredecessor(unsigned index, MSchedGraphSBNode *dest) {
      Predecessors[index] = dest;
    }

    MSchedGraphSBNode* getPredecessor(unsigned index) {
      return Predecessors[index];
    }

    MSchedGraphSBEdge* getSuccessor(unsigned index) {
      return &Successors[index];
    }

    void deleteSuccessor(MSchedGraphSBNode *node) {
      for (unsigned i = 0; i != Successors.size(); ++i)
        if (Successors[i].getDest() == node) {
          Successors.erase(Successors.begin()+i);
          node->Predecessors.erase(std::find(node->Predecessors.begin(),
                                             node->Predecessors.end(), this));
          --i; //Decrease index var since we deleted a node
        }
    }

    void addOutEdge(MSchedGraphSBNode *destination,
                    MSchedGraphSBEdge::MSchedGraphSBEdgeType type,
                    unsigned deptype, unsigned diff=0) {
      Successors.push_back(MSchedGraphSBEdge(destination, type, deptype,diff));
      destination->Predecessors.push_back(this);
    }

    //General methods to get and set data for the node
    const MachineInstr* getInst() { return Inst; }
    MSchedGraphSB* getParent() { return Parent; }
    bool hasPredecessors() { return (Predecessors.size() > 0); }
    bool hasSuccessors() { return (Successors.size() > 0); }
    unsigned getLatency() { return latency; }
    unsigned getLatency() const { return latency; }
    unsigned getIndex() { return index; }
    unsigned getIteDiff(MSchedGraphSBNode *succ);
    MSchedGraphSBEdge getInEdge(MSchedGraphSBNode *pred);
    unsigned getInEdgeNum(MSchedGraphSBNode *pred);
    bool isSuccessor(MSchedGraphSBNode *);
    bool isPredecessor(MSchedGraphSBNode *);
    bool isBranch() { return isBranchInstr; }
    bool isPredicate() { return isPredicateNode; }
    bool isPredicate() const { return isPredicateNode; }
    std::vector<const MachineInstr*> getOtherInstrs() { return otherInstrs; }

    //Debug support
    void print(std::ostream &os) const;
    void setParent(MSchedGraphSB *p) { Parent = p; }
  };

  //Node iterator for graph generation
  template<class IteratorType, class NodeType>
  class MSchedGraphSBNodeIterator : public forward_iterator<NodeType*, ptrdiff_t> {
    IteratorType I;   // std::vector<MSchedGraphSBEdge>::iterator or const_iterator
  public:
    MSchedGraphSBNodeIterator(IteratorType i) : I(i) {}

    bool operator==(const MSchedGraphSBNodeIterator RHS) const { return I == RHS.I; }
    bool operator!=(const MSchedGraphSBNodeIterator RHS) const { return I != RHS.I; }

    const MSchedGraphSBNodeIterator &operator=(const MSchedGraphSBNodeIterator &RHS) {
      I = RHS.I;
      return *this;
    }

    NodeType* operator*() const {
      return I->getDest();
    }
    NodeType* operator->() const { return operator*(); }

    MSchedGraphSBNodeIterator& operator++() {                // Preincrement
      ++I;
      return *this;
    }
    MSchedGraphSBNodeIterator operator++(int) { // Postincrement
      MSchedGraphSBNodeIterator tmp = *this; ++*this; return tmp;
    }

    MSchedGraphSBEdge &getEdge() {
      return *I;
    }
    const MSchedGraphSBEdge &getEdge() const {
      return *I;
    }
  };

  inline MSchedGraphSBNode::succ_const_iterator MSchedGraphSBNode::succ_begin() const {
    return succ_const_iterator(Successors.begin());
  }
  inline MSchedGraphSBNode::succ_const_iterator MSchedGraphSBNode::succ_end() const {
    return succ_const_iterator(Successors.end());
  }
  inline MSchedGraphSBNode::succ_iterator MSchedGraphSBNode::succ_begin() {
    return succ_iterator(Successors.begin());
  }
  inline MSchedGraphSBNode::succ_iterator MSchedGraphSBNode::succ_end() {
    return succ_iterator(Successors.end());
  }

  // ostream << operator for MSGraphNode class
  inline std::ostream &operator<<(std::ostream &os,
                                  const MSchedGraphSBNode &node) {
    node.print(os);
    return os;
  }


  // Provide specializations of GraphTraits to be able to use graph
  // iterators on the scheduling graph!
  //
  template <> struct GraphTraits<MSchedGraphSBNode*> {
    typedef MSchedGraphSBNode NodeType;
    typedef MSchedGraphSBNode::succ_iterator ChildIteratorType;

    static inline ChildIteratorType child_begin(NodeType *N) {
      return N->succ_begin();
    }
    static inline ChildIteratorType child_end(NodeType *N) {
      return N->succ_end();
    }

    static NodeType *getEntryNode(NodeType* N) { return N; }
  };



  //Graph class to represent dependence graph
  class MSchedGraphSB {

    std::vector<const MachineBasicBlock *> BBs; //Machine basic block
    const TargetMachine &Target; //Target Machine

    //Nodes
    std::map<const MachineInstr*, MSchedGraphSBNode*> GraphMap;

    //Add Nodes and Edges to this graph for our BB
    typedef std::pair<int, MSchedGraphSBNode*> OpIndexNodePair;
    void buildNodesAndEdges(std::map<const MachineInstr*, unsigned> &ignoreInstrs, DependenceAnalyzer &DA, std::map<MachineInstr*, Instruction*> &machineTollvm, std::map<MSchedGraphSBNode*, std::set<MachineInstr*> > &liveOutsideTrace);
    void addValueEdges(std::vector<OpIndexNodePair> &NodesInMap,
                       MSchedGraphSBNode *node,
                       bool nodeIsUse, bool nodeIsDef, std::vector<const MachineInstr*> &phiInstrs, int diff=0);
    void addMachRegEdges(std::map<int,
                         std::vector<OpIndexNodePair> >& regNumtoNodeMap);
    void addMemEdges(const std::vector<MSchedGraphSBNode*>& memInst,
                     DependenceAnalyzer &DA, std::map<MachineInstr*, Instruction*> &machineTollvm);


    bool instrCauseException(MachineOpCode opCode);

  public:
    MSchedGraphSB(const MachineBasicBlock *bb, const TargetMachine &targ,
                std::map<const MachineInstr*, unsigned> &ignoreInstrs,
                DependenceAnalyzer &DA, std::map<MachineInstr*, Instruction*> &machineTollvm);

    //Copy constructor with maps to link old nodes to new nodes
    MSchedGraphSB(const MSchedGraphSB &G, std::map<MSchedGraphSBNode*, MSchedGraphSBNode*> &newNodes);

    MSchedGraphSB(std::vector<const MachineBasicBlock*> &bbs,
                const TargetMachine &targ,
                std::map<const MachineInstr*, unsigned> &ignoreInstrs,
                DependenceAnalyzer &DA,
                std::map<MachineInstr*, Instruction*> &machineTollvm);

    //Print graph
    void print(std::ostream &os) const;

    //Deconstructor!
    ~MSchedGraphSB();

    //Add or delete nodes from the Graph
    void addNode(const MachineInstr* MI, MSchedGraphSBNode *node);
    void deleteNode(MSchedGraphSBNode *node);
    int totalDelay();

    //iterators
    typedef std::map<const MachineInstr*, MSchedGraphSBNode*>::iterator iterator;
    typedef std::map<const MachineInstr*, MSchedGraphSBNode*>::const_iterator const_iterator;
    typedef std::map<const MachineInstr*, MSchedGraphSBNode*>::reverse_iterator reverse_iterator;
    iterator find(const MachineInstr* I) { return GraphMap.find(I); }
    iterator end() { return GraphMap.end(); }
    iterator begin() { return GraphMap.begin(); }
    unsigned size() { return GraphMap.size(); }
    reverse_iterator rbegin() { return GraphMap.rbegin(); }
    reverse_iterator rend() { return GraphMap.rend(); }

    //Get Target or original machine basic block
    const TargetMachine* getTarget() { return &Target; }
    std::vector<const MachineBasicBlock*> getBBs() { return BBs; }
  };





  // Provide specializations of GraphTraits to be able to use graph
  // iterators on the scheduling graph
  static MSchedGraphSBNode& getSecond(std::pair<const MachineInstr* const,
                                    MSchedGraphSBNode*> &Pair) {
    return *Pair.second;
  }

  template <> struct GraphTraits<MSchedGraphSB*> {
    typedef MSchedGraphSBNode NodeType;
    typedef MSchedGraphSBNode::succ_iterator ChildIteratorType;

    static inline ChildIteratorType child_begin(NodeType *N) {
      return N->succ_begin();
    }
    static inline ChildIteratorType child_end(NodeType *N) {
      return N->succ_end();
    }

    typedef std::pointer_to_unary_function<std::pair<const MachineInstr* const,
           MSchedGraphSBNode*>&, MSchedGraphSBNode&> DerefFun;

    typedef mapped_iterator<MSchedGraphSB::iterator, DerefFun> nodes_iterator;
    static nodes_iterator nodes_begin(MSchedGraphSB *G) {
      return map_iterator(((MSchedGraphSB*)G)->begin(), DerefFun(getSecond));
    }
    static nodes_iterator nodes_end(MSchedGraphSB *G) {
      return map_iterator(((MSchedGraphSB*)G)->end(), DerefFun(getSecond));
    }

  };

  template <> struct GraphTraits<const MSchedGraphSB*> {
    typedef const MSchedGraphSBNode NodeType;
    typedef MSchedGraphSBNode::succ_const_iterator ChildIteratorType;

    static inline ChildIteratorType child_begin(NodeType *N) {
      return N->succ_begin();
    }
    static inline ChildIteratorType child_end(NodeType *N) {
      return N->succ_end();
    }
    typedef std::pointer_to_unary_function<std::pair<const MachineInstr* const,
                                                     MSchedGraphSBNode*>&, MSchedGraphSBNode&> DerefFun;

    typedef mapped_iterator<MSchedGraphSB::iterator, DerefFun> nodes_iterator;
    static nodes_iterator nodes_begin(MSchedGraphSB *G) {
      return map_iterator(((MSchedGraphSB*)G)->begin(), DerefFun(getSecond));
    }
    static nodes_iterator nodes_end(MSchedGraphSB *G) {
      return map_iterator(((MSchedGraphSB*)G)->end(), DerefFun(getSecond));
    }
  };

  template <> struct GraphTraits<Inverse<MSchedGraphSB*> > {
    typedef MSchedGraphSBNode NodeType;
    typedef MSchedGraphSBNode::pred_iterator ChildIteratorType;

    static inline ChildIteratorType child_begin(NodeType *N) {
      return N->pred_begin();
    }
    static inline ChildIteratorType child_end(NodeType *N) {
      return N->pred_end();
    }
    typedef std::pointer_to_unary_function<std::pair<const MachineInstr* const,
           MSchedGraphSBNode*>&, MSchedGraphSBNode&> DerefFun;

    typedef mapped_iterator<MSchedGraphSB::iterator, DerefFun> nodes_iterator;
    static nodes_iterator nodes_begin(MSchedGraphSB *G) {
      return map_iterator(((MSchedGraphSB*)G)->begin(), DerefFun(getSecond));
    }
    static nodes_iterator nodes_end(MSchedGraphSB *G) {
      return map_iterator(((MSchedGraphSB*)G)->end(), DerefFun(getSecond));
    }
  };

  template <> struct GraphTraits<Inverse<const MSchedGraphSB*> > {
    typedef const MSchedGraphSBNode NodeType;
    typedef MSchedGraphSBNode::pred_const_iterator ChildIteratorType;

    static inline ChildIteratorType child_begin(NodeType *N) {
      return N->pred_begin();
    }
    static inline ChildIteratorType child_end(NodeType *N) {
      return N->pred_end();
    }

    typedef std::pointer_to_unary_function<std::pair<const MachineInstr* const,
                                                     MSchedGraphSBNode*>&, MSchedGraphSBNode&> DerefFun;

    typedef mapped_iterator<MSchedGraphSB::iterator, DerefFun> nodes_iterator;
    static nodes_iterator nodes_begin(MSchedGraphSB *G) {
      return map_iterator(((MSchedGraphSB*)G)->begin(), DerefFun(getSecond));
    }
    static nodes_iterator nodes_end(MSchedGraphSB *G) {
      return map_iterator(((MSchedGraphSB*)G)->end(), DerefFun(getSecond));
    }
  };
}

#endif
