//===-- ProfilePaths.cpp - interface to insert instrumentation ---*- C++ -*--=//
//
// This inserts intrumentation for counting
// execution of paths though a given function
// Its implemented as a "Function" Pass, and called using opt
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
// of a function is identified through a unique number. the code insertion
// is optimal in the sense that its inserted over a minimal set of edges. Also,
// the algorithm makes sure than initialization, path increment and counter
// update can be collapsed into minimum number of edges.
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/UnifyFunctionExitNodes.h"
#include "llvm/Support/CFG.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/iMemory.h"
#include "llvm/iOperators.h"
#include "llvm/iOther.h"
#include "llvm/Module.h"
#include "Graph.h"
#include <fstream>
#include "Config/stdio.h"
using std::vector;

struct ProfilePaths : public FunctionPass {
  bool runOnFunction(Function &F);

  // Before this pass, make sure that there is only one 
  // entry and only one exit node for the function in the CFG of the function
  //
  void ProfilePaths::getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequired<UnifyFunctionExitNodes>();
  }
};

static RegisterOpt<ProfilePaths> X("paths", "Profile Paths");

static Node *findBB(std::vector<Node *> &st, BasicBlock *BB){
  for(std::vector<Node *>::iterator si=st.begin(); si!=st.end(); ++si){
    if(((*si)->getElement())==BB){
      return *si;
    }
  }
  return NULL;
}

//Per function pass for inserting counters and trigger code
bool ProfilePaths::runOnFunction(Function &F){

  static int mn = -1;
  static int CountCounter = 1;
  if(F.isExternal()) {
    return false;
  }
 
  //increment counter for instrumented functions. mn is now function#
  mn++;
  
  // Transform the cfg s.t. we have just one exit node
  BasicBlock *ExitNode = getAnalysis<UnifyFunctionExitNodes>().getExitNode();  

  //iterating over BBs and making graph
  std::vector<Node *> nodes;
  std::vector<Edge> edges;

  Node *tmp;
  Node *exitNode = 0, *startNode = 0;

  // The nodes must be uniquesly identified:
  // That is, no two nodes must hav same BB*
  
  for (Function::iterator BB = F.begin(), BE = F.end(); BB != BE; ++BB) {
    Node *nd=new Node(BB);
    nodes.push_back(nd); 
    if(&*BB == ExitNode)
      exitNode=nd;
    if(BB==F.begin())
      startNode=nd;
  }

  // now do it againto insert edges
  for (Function::iterator BB = F.begin(), BE = F.end(); BB != BE; ++BB){
    Node *nd=findBB(nodes, BB);
    assert(nd && "No node for this edge!");

    for(BasicBlock::succ_iterator s=succ_begin(BB), se=succ_end(BB); 
	s!=se; ++s){
      Node *nd2=findBB(nodes,*s);
      assert(nd2 && "No node for this edge!");
      Edge ed(nd,nd2,0);
      edges.push_back(ed);
    }
  }
  
  Graph g(nodes,edges, startNode, exitNode);

#ifdef DEBUG_PATH_PROFILES  
  std::cerr<<"Original graph\n";
  printGraph(g);
#endif

  BasicBlock *fr = &F.front();
  
  // The graph is made acyclic: this is done
  // by removing back edges for now, and adding them later on
  vector<Edge> be;
  std::map<Node *, int> nodePriority; //it ranks nodes in depth first order traversal
  g.getBackEdges(be, nodePriority);
  
#ifdef DEBUG_PATH_PROFILES
  std::cerr<<"BackEdges-------------\n";
  for(vector<Edge>::iterator VI=be.begin(); VI!=be.end(); ++VI){
    printEdge(*VI);
    cerr<<"\n";
  }
  std::cerr<<"------\n";
#endif

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

#ifdef DEBUG_PATH_PROFILES
  std::cerr<<"After adding dummy edges\n";
  printGraph(g);
#endif

  // Now, every edge in the graph is assigned a weight
  // This weight later adds on to assign path
  // numbers to different paths in the graph
  //  All paths for now are acyclic,
  // since no back edges in the graph now
  // numPaths is the number of acyclic paths in the graph
  int numPaths=valueAssignmentToEdges(g, nodePriority, be);

  //if(numPaths<=1) return false;

  static GlobalVariable *threshold = NULL;
  static bool insertedThreshold = false;

  if(!insertedThreshold){
    threshold = new GlobalVariable(Type::IntTy, false,
                                   GlobalValue::ExternalLinkage, 0,
                                   "reopt_threshold");

    F.getParent()->getGlobalList().push_back(threshold);
    insertedThreshold = true;
  }

  assert(threshold && "GlobalVariable threshold not defined!");


  if(fr->getParent()->getName() == "main"){
    //intialize threshold

    // FIXME: THIS IS HORRIBLY BROKEN.  FUNCTION PASSES CANNOT DO THIS, EXCEPT
    // IN THEIR INITIALIZE METHOD!!
    Function *initialize =
      F.getParent()->getOrInsertFunction("reoptimizerInitialize", Type::VoidTy,
                                         PointerType::get(Type::IntTy), 0);
    
    vector<Value *> trargs;
    trargs.push_back(threshold);
    new CallInst(initialize, trargs, "", fr->begin());
  }


  if(numPaths<=1 || numPaths >5000) return false;
  
#ifdef DEBUG_PATH_PROFILES  
  printGraph(g);
#endif

  //create instruction allocation r and count
  //r is the variable that'll act like an accumulator
  //all along the path, we just add edge values to r
  //and at the end, r reflects the path number
  //count is an array: count[x] would store
  //the number of executions of path numbered x

  Instruction *rVar=new 
    AllocaInst(Type::IntTy, 
               ConstantUInt::get(Type::UIntTy,1),"R");

  //Instruction *countVar=new 
  //AllocaInst(Type::IntTy, 
  //           ConstantUInt::get(Type::UIntTy, numPaths), "Count");

  //initialize counter array!
  std::vector<Constant*> arrayInitialize;
  for(int xi=0; xi<numPaths; xi++)
    arrayInitialize.push_back(ConstantSInt::get(Type::IntTy, 0));

  const ArrayType *ATy = ArrayType::get(Type::IntTy, numPaths);
  Constant *initializer =  ConstantArray::get(ATy, arrayInitialize);
  char tempChar[20];
  sprintf(tempChar, "Count%d", CountCounter);
  CountCounter++;
  std::string countStr = tempChar;
  GlobalVariable *countVar = new GlobalVariable(ATy, false,
                                                GlobalValue::InternalLinkage, 
                                                initializer, countStr,
                                                F.getParent());
  
  // insert initialization code in first (entry) BB
  // this includes initializing r and count
  insertInTopBB(&F.getEntryNode(),numPaths, rVar, threshold);
    
  //now process the graph: get path numbers,
  //get increments along different paths,
  //and assign "increments" and "updates" (to r and count)
  //"optimally". Finally, insert llvm code along various edges
  processGraph(g, rVar, countVar, be, stDummy, exDummy, numPaths, mn, 
               threshold);    
   
  return true;  // Always modifies function
}
