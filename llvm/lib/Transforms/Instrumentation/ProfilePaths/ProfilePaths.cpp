//===-- ProfilePaths.cpp - interface to insert instrumentation ---*- C++ -*--=//
//
// This inserts intrumentation for counting
// execution of paths though a given method
// Its implemented as a "Method" Pass, and called using opt
//
// This pass is implemented by using algorithms similar to 
// 1."Efficient Path Profiling": Ball, T. and Larus, J. R., 
// Proceedings of Micro-29, Dec 1996, Paris, France.
// 2."Efficiently Counting Program events with support for on-line
//   "queries": Ball T., ACM Transactions on Programming Languages
//   and systems, Sep 1994.
//
// The algorithms work on a Graph constructed over the nodes
// made from Basic Blocks: The transformations then take place on
// the constucted graph (implementation in Graph.cpp and GraphAuxillary.cpp)
// and finally, appropriate instrumentation is placed over suitable edges.
// (code inserted through EdgeCode.cpp).
// 
// The algorithm inserts code such that every acyclic path in the CFG
// of a method is identified through a unique number. the code insertion
// is optimal in the sense that its inserted over a minimal set of edges. Also,
// the algorithm makes sure than initialization, path increment and counter
// update can be collapsed into minmimum number of edges.
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/ProfilePaths.h"
#include "llvm/Transforms/UnifyMethodExitNodes.h"
#include "llvm/Support/CFG.h"
#include "llvm/Method.h"
#include "llvm/BasicBlock.h"
#include "llvm/ConstantVals.h"
#include "llvm/DerivedTypes.h"
#include "llvm/iMemory.h"
#include "Graph.h"

using std::vector;

static Node *findBB(std::set<Node *> &st, BasicBlock *BB){
  for(std::set<Node *>::iterator si=st.begin(); si!=st.end(); ++si){
    if(((*si)->getElement())==BB){
      return *si;
    }
  }
  return NULL;
}

//Per method pass for inserting counters and trigger code
bool ProfilePaths::runOnMethod(Method *M){
  //Transform the cfg s.t. we have just one exit node
  BasicBlock *ExitNode = 
    getAnalysis<UnifyMethodExitNodes>().getExitNode();  
  
  //iterating over BBs and making graph
  std::set<Node *> nodes;
  std::set<Edge> edges;
  Node *tmp;
  Node *exitNode, *startNode;

  //The nodes must be uniquesly identified:
  //That is, no two nodes must hav same BB*
  
  //First enter just nodes: later enter edges
  for (Method::iterator BB = M->begin(), BE=M->end(); BB != BE; ++BB){
    Node *nd=new Node(*BB);
    nodes.insert(nd); 
    if(*BB==ExitNode)
      exitNode=nd;
    if(*BB==M->front())
      startNode=nd;
  }

  //now do it againto insert edges
  for (Method::iterator BB = M->begin(), BE=M->end(); BB != BE; ++BB){
    Node *nd=findBB(nodes, *BB);
    assert(nd && "No node for this edge!");
    for(BasicBlock::succ_iterator s=succ_begin(*BB), se=succ_end(*BB); 
	s!=se; ++s){
      Node *nd2=findBB(nodes,*s);
      assert(nd2 && "No node for this edge!");
      Edge ed(nd,nd2,0);
      edges.insert(ed);
    }
  }
  
  Graph g(nodes,edges, startNode, exitNode);

#ifdef DEBUG_PATH_PROFILES  
  printGraph(g);
#endif

  BasicBlock *fr=M->front();
  
  //If only one BB, don't instrument
  if (M->getBasicBlocks().size() == 1) {    
    //The graph is made acyclic: this is done
    //by removing back edges for now, and adding them later on
    vector<Edge> be;
    g.getBackEdges(be);
#ifdef DEBUG_PATH_PROFILES
    cerr<<"Backedges:"<<be.size()<<endl;
#endif
    //Now we need to reflect the effect of back edges
    //This is done by adding dummy edges
    //If a->b is a back edge
    //Then we add 2 back edges for it:
    //1. from root->b (in vector stDummy)
    //and 2. from a->exit (in vector exDummy)
    vector<Edge> stDummy;
    vector<Edge> exDummy;
    addDummyEdges(stDummy, exDummy, g, be);
    
    //Now, every edge in the graph is assigned a weight
    //This weight later adds on to assign path
    //numbers to different paths in the graph
    // All paths for now are acyclic,
    //since no back edges in the graph now
    //numPaths is the number of acyclic paths in the graph
    int numPaths=valueAssignmentToEdges(g);
    
    //create instruction allocation r and count
    //r is the variable that'll act like an accumulator
    //all along the path, we just add edge values to r
    //and at the end, r reflects the path number
    //count is an array: count[x] would store
    //the number of executions of path numbered x
    Instruction *rVar=new 
      AllocaInst(PointerType::get(Type::IntTy), 
		 ConstantUInt::get(Type::UIntTy,1),"R");
    
    Instruction *countVar=new 
      AllocaInst(PointerType::get(Type::IntTy), 
		 ConstantUInt::get(Type::UIntTy, numPaths), "Count");
    
    //insert initialization code in first (entry) BB
    //this includes initializing r and count
    insertInTopBB(M->getEntryNode(),numPaths, rVar, countVar);
    
    //now process the graph: get path numbers,
    //get increments along different paths,
    //and assign "increments" and "updates" (to r and count)
    //"optimally". Finally, insert llvm code along various edges
    processGraph(g, rVar, countVar, be, stDummy, exDummy);
  }

  return true;  // Always modifies method
}

//Before this pass, make sure that there is only one 
//entry and only one exit node for the method in the CFG of the method
void ProfilePaths::getAnalysisUsageInfo(Pass::AnalysisSet &Requires,
					  Pass::AnalysisSet &Destroyed,
					  Pass::AnalysisSet &Provided) {
  Requires.push_back(UnifyMethodExitNodes::ID);
}







