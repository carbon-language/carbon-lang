//===-- Graph.h -------------------------------------------------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// Header file for Graph: This Graph is used by PathProfiles class, and is used
// for detecting proper points in cfg for code insertion
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_GRAPH_H
#define LLVM_GRAPH_H

#include "llvm/BasicBlock.h"
#include <map>
#include <cstdlib>

class Module;
class Function;

//Class Node
//It forms the vertex for the graph
class Node{
public:
  BasicBlock* element;
  int weight;
public:
  inline Node(BasicBlock* x) { element=x; weight=0; }
  inline BasicBlock* &getElement() { return element; }
  inline BasicBlock* const &getElement() const { return element; }
  inline int getWeight() { return weight; }
  inline void setElement(BasicBlock* e) { element=e; }
  inline void setWeight(int w) { weight=w;}
  inline bool operator<(Node& nd) const { return element<nd.element; }
  inline bool operator==(Node& nd) const { return element==nd.element; }
};

//Class Edge
//Denotes an edge in the graph
class Edge{
private:
  Node *first;
  Node *second;
  bool isnull;
  int weight;
  double randId;
public:
  inline Edge(Node *f,Node *s, int wt=0){
    first=f;
    second=s;
    weight=wt;
    randId=rand();
    isnull=false;
  }
  
  inline Edge(Node *f,Node *s, int wt, double rd){
    first=f;
    second=s;
    weight=wt;
    randId=rd;
    isnull=false; 
  }

  inline Edge() { isnull = true; }
  inline double getRandId(){ return randId; }
  inline Node* getFirst() { assert(!isNull()); return first; }
  inline Node* const getFirst() const { assert(!isNull()); return first; }
  inline Node* getSecond() { assert(!isNull()); return second; }
  inline Node* const getSecond() const { assert(!isNull()); return second; }
  
  inline int getWeight() { assert(!isNull());  return weight; }
  inline void setWeight(int n) { assert(!isNull()); weight=n; }
  
  inline void setFirst(Node *&f) { assert(!isNull());  first=f; }
  inline void setSecond(Node *&s) { assert(!isNull()); second=s; }
  
  
  inline bool isNull() const { return isnull;} 
  
  inline bool operator<(const Edge& ed) const{
    // Can't be the same if one is null and the other isn't
    if (isNull() != ed.isNull())
      return true;

    return (*first<*(ed.getFirst()))|| 
      (*first==*(ed.getFirst()) && *second<*(ed.getSecond()));
  }

  inline bool operator==(const Edge& ed) const{
    return !(*this<ed) && !(ed<*this);
  }

  inline bool operator!=(const Edge& ed) const{return !(*this==ed);} 
};


//graphListElement
//This forms the "adjacency list element" of a 
//vertex adjacency list in graph
struct graphListElement{
  Node *element;
  int weight;
  double randId;
  inline graphListElement(Node *n, int w, double rand){ 
    element=n; 
    weight=w;
    randId=rand;
  }
};


namespace std {
  template<>
  struct less<Node *> : public binary_function<Node *, Node *,bool> {
    bool operator()(Node *n1, Node *n2) const {
      return n1->getElement() < n2->getElement();
    }
  };
 
  template<>
  struct less<Edge> : public binary_function<Edge,Edge,bool> {
    bool operator()(Edge e1, Edge e2) const {
      assert(!e1.isNull() && !e2.isNull());
      
      Node *x1=e1.getFirst();
      Node *x2=e1.getSecond();
      Node *y1=e2.getFirst();
      Node *y2=e2.getSecond();
      return (*x1<*y1 ||(*x1==*y1 && *x2<*y2));
    }
  };
}

struct BBSort{
  bool operator()(BasicBlock *BB1, BasicBlock *BB2) const{
    std::string name1=BB1->getName();
    std::string name2=BB2->getName();
    return name1<name2;
  }
};

struct NodeListSort{
  bool operator()(graphListElement BB1, graphListElement BB2) const{
    std::string name1=BB1.element->getElement()->getName();
    std::string name2=BB2.element->getElement()->getName();
    return name1<name2;
  }
};

struct EdgeCompare2{
  bool operator()(Edge e1, Edge e2) const {
    assert(!e1.isNull() && !e2.isNull());
    Node *x1=e1.getFirst();
    Node *x2=e1.getSecond();
    Node *y1=e2.getFirst();
    Node *y2=e2.getSecond();
    int w1=e1.getWeight();
    int w2=e2.getWeight();
    double r1 = e1.getRandId();
    double r2 = e2.getRandId();
    //return (*x1<*y1 || (*x1==*y1 && *x2<*y2) || (*x1==*y1 && *x2==*y2 && w1<w2));
    return (*x1<*y1 || (*x1==*y1 && *x2<*y2) || (*x1==*y1 && *x2==*y2 && w1<w2) || (*x1==*y1 && *x2==*y2 && w1==w2 && r1<r2));
  }
};

//struct EdgeCompare2{
//bool operator()(Edge e1, Edge e2) const {
//  assert(!e1.isNull() && !e2.isNull());
//  return (e1.getRandId()<e2.getRandId());
//}
//};


//this is used to color vertices
//during DFS
enum Color{
  WHITE,
  GREY,
  BLACK
};


//For path profiling,
//We assume that the graph is connected (which is true for
//any method CFG)
//We also assume that the graph has single entry and single exit
//(For this, we make a pass over the graph that ensures this)
//The graph is a construction over any existing graph of BBs
//Its a construction "over" existing cfg: with
//additional features like edges and weights to edges

//graph uses adjacency list representation
class Graph{
public:
  //typedef std::map<Node*, std::list<graphListElement> > nodeMapTy;
  typedef std::map<Node*, std::vector<graphListElement> > nodeMapTy;//chng
private:
  //the adjacency list of a vertex or node
  nodeMapTy nodes;
  
  //the start or root node
  Node *strt;

  //the exit node
  Node *ext;

  //a private method for doing DFS traversal of graph
  //this is used in determining the reverse topological sort 
  //of the graph
  void DFS_Visit(Node *nd, std::vector<Node *> &toReturn);

  //Its a variation of DFS to get the backedges in the graph
  //We get back edges by associating a time
  //and a color with each vertex.
  //The time of a vertex is the time when it was first visited
  //The color of a vertex is initially WHITE,
  //Changes to GREY when it is first visited,
  //and changes to BLACK when ALL its neighbors
  //have been visited
  //So we have a back edge when we meet a successor of
  //a node with smaller time, and GREY color
  void getBackEdgesVisit(Node *u, 
			 std::vector<Edge > &be,
			 std::map<Node *, Color> &clr,
			 std::map<Node *, int> &d, 
			 int &time);

public:
  typedef nodeMapTy::iterator elementIterator;
  typedef nodeMapTy::const_iterator constElementIterator;
  typedef std::vector<graphListElement > nodeList;//chng
  //typedef std::vector<graphListElement > nodeList;

  //graph constructors

  //empty constructor: then add edges and nodes later on
  Graph() {}
  
  //constructor with root and exit node specified
  Graph(std::vector<Node*> n, 
	std::vector<Edge> e, Node *rt, Node *lt);

  //add a node
  void addNode(Node *nd);

  //add an edge
  //this adds an edge ONLY when 
  //the edge to be added doesn not already exist
  //we "equate" two edges here only with their 
  //end points
  void addEdge(Edge ed, int w);

  //add an edge EVEN IF such an edge already exists
  //this may make a multi-graph
  //which does happen when we add dummy edges
  //to the graph, for compensating for back-edges
  void addEdgeForce(Edge ed);

  //set the weight of an edge
  void setWeight(Edge ed);

  //remove an edge
  //Note that it removes just one edge,
  //the first edge that is encountered
  void removeEdge(Edge ed);

  //remove edge with given wt
  void removeEdgeWithWt(Edge ed);

  //check whether graph has an edge
  //having an edge simply means that there is an edge in the graph
  //which has same endpoints as the given edge
  //it may possibly have different weight though
  bool hasEdge(Edge ed);

  //check whether graph has an edge, with a given wt
  bool hasEdgeAndWt(Edge ed);

  //get the list of successor nodes
  std::vector<Node *> getSuccNodes(Node *nd);

  //get the number of outgoing edges
  int getNumberOfOutgoingEdges(Node *nd) const;

  //get the list of predecessor nodes
  std::vector<Node *> getPredNodes(Node *nd);


  //to get the no of incoming edges
  int getNumberOfIncomingEdges(Node *nd);

  //get the list of all the vertices in graph
  std::vector<Node *> getAllNodes() const;
  std::vector<Node *> getAllNodes();

  //get a list of nodes in the graph
  //in r-topological sorted order
  //note that we assumed graph to be connected
  std::vector<Node *> reverseTopologicalSort();
  
  //reverse the sign of weights on edges
  //this way, max-spanning tree could be obtained
  //usin min-spanning tree, and vice versa
  void reverseWts();

  //Ordinarily, the graph is directional
  //this converts the graph into an 
  //undirectional graph
  //This is done by adding an edge
  //v->u for all existing edges u->v
  void makeUnDirectional();

  //print graph: for debugging
  void printGraph();
  
  //get a vector of back edges in the graph
  void getBackEdges(std::vector<Edge> &be, std::map<Node *, int> &d);

  nodeList &sortNodeList(Node *par, nodeList &nl, std::vector<Edge> &be);
 
  //Get the Maximal spanning tree (also a graph)
  //of the graph
  Graph* getMaxSpanningTree();
  
  //get the nodeList adjacent to a node
  //a nodeList element contains a node, and the weight 
  //corresponding to the edge for that element
  inline nodeList &getNodeList(Node *nd) {
    elementIterator nli = nodes.find(nd);
    assert(nli != nodes.end() && "Node must be in nodes map");
    return nodes[nd];//sortNodeList(nd, nli->second);
  }
   
  nodeList &getSortedNodeList(Node *nd, std::vector<Edge> &be) {
    elementIterator nli = nodes.find(nd);
    assert(nli != nodes.end() && "Node must be in nodes map");
    return sortNodeList(nd, nodes[nd], be);
  }
 
  //get the root of the graph
  inline Node *getRoot()                {return strt; }
  inline Node * const getRoot() const   {return strt; }

  //get exit: we assumed there IS a unique exit :)
  inline Node *getExit()                {return ext; }
  inline Node * const getExit() const   {return ext; }
  //Check if a given node is the root
  inline bool isRoot(Node *n) const     {return (*n==*strt); }

  //check if a given node is leaf node
  //here we hv only 1 leaf: which is the exit node
  inline bool isLeaf(Node *n)    const  {return (*n==*ext);  }
};

//This class is used to generate 
//"appropriate" code to be inserted
//along an edge
//The code to be inserted can be of six different types
//as given below
//1: r=k (where k is some constant)
//2: r=0
//3: r+=k
//4: count[k]++
//5: Count[r+k]++
//6: Count[r]++
class getEdgeCode{
 private:
  //cond implies which 
  //"kind" of code is to be inserted
  //(from 1-6 above)
  int cond;
  //inc is the increment: eg k, or 0
  int inc;
  
  //A backedge must carry the code
  //of both incoming "dummy" edge
  //and outgoing "dummy" edge
  //If a->b is a backedge
  //then incoming dummy edge is root->b
  //and outgoing dummy edge is a->exit

  //incoming dummy edge, if any
  getEdgeCode *cdIn;

  //outgoing dummy edge, if any
  getEdgeCode *cdOut;

public:
  getEdgeCode(){
    cdIn=NULL;
    cdOut=NULL;
    inc=0;
    cond=0;
  }

  //set condition: 1-6
  inline void setCond(int n) {cond=n;}

  //get the condition
  inline int getCond() { return cond;}

  //set increment
  inline void setInc(int n) {inc=n;}

  //get increment
  inline int getInc() {return inc;}

  //set CdIn (only used for backedges)
  inline void setCdIn(getEdgeCode *gd){ cdIn=gd;}
  
  //set CdOut (only used for backedges)
  inline void setCdOut(getEdgeCode *gd){ cdOut=gd;}

  //get the code to be inserted on the edge
  //This is determined from cond (1-6)
  void getCode(Instruction *a, Value *b, Function *M, BasicBlock *BB, 
               std::vector<Value *> &retVec);
};


//auxillary functions on graph

//print a given edge in the form BB1Label->BB2Label 
void printEdge(Edge ed);

//Do graph processing: to determine minimal edge increments, 
//appropriate code insertions etc and insert the code at
//appropriate locations
void processGraph(Graph &g, Instruction *rInst, Value *countInst, std::vector<Edge> &be, std::vector<Edge> &stDummy, std::vector<Edge> &exDummy, int n, int MethNo, Value *threshold);

//print the graph (for debugging)
void printGraph(Graph &g);


//void printGraph(const Graph g);
//insert a basic block with appropriate code
//along a given edge
void insertBB(Edge ed, getEdgeCode *edgeCode, Instruction *rInst, Value *countInst, int n, int Methno, Value *threshold);

//Insert the initialization code in the top BB
//this includes initializing r, and count
//r is like an accumulator, that 
//keeps on adding increments as we traverse along a path
//and at the end of the path, r contains the path
//number of that path
//Count is an array, where Count[k] represents
//the number of executions of path k
void insertInTopBB(BasicBlock *front, int k, Instruction *rVar, Value *threshold);

//Add dummy edges corresponding to the back edges
//If a->b is a backedge
//then incoming dummy edge is root->b
//and outgoing dummy edge is a->exit
void addDummyEdges(std::vector<Edge> &stDummy, std::vector<Edge> &exDummy, Graph &g, std::vector<Edge> &be);

//Assign a value to all the edges in the graph
//such that if we traverse along any path from root to exit, and
//add up the edge values, we get a path number that uniquely
//refers to the path we travelled
int valueAssignmentToEdges(Graph& g, std::map<Node *, int> nodePriority, 
                           std::vector<Edge> &be);

void getBBtrace(std::vector<BasicBlock *> &vBB, int pathNo, Function *M);
#endif


