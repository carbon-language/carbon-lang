//===----Instrumentation/ProfilePaths/RetracePath.cppTrigger.cpp--*- C++ -*--=//
//
// Retraces a path of BasicBlock, given a path number and a graph!
//
//===----------------------------------------------------------------------===//

#include "llvm/Module.h"
#include "llvm/iTerminators.h"
#include "llvm/iOther.h"
#include "llvm/Support/CFG.h"
#include "Graph.h"

using std::vector;
using std::map;
using std::cerr;

//Routines to get the path trace!

void getPathFrmNode(Node *n, vector<BasicBlock*> &vBB, int pathNo, Graph &g, 
		    vector<Edge> &stDummy, vector<Edge> &exDummy, 
		    vector<Edge> &be,
		    double strand){
  Graph::nodeList &nlist = g.getNodeList(n);
  
  //printGraph(g);
  //std::cerr<<"Path No: "<<pathNo<<"\n";
  int maxCount=-9999999;
  bool isStart=false;

  if(*n==*g.getRoot())//its root: so first node of path
    isStart=true;

  double edgeRnd=0;
  Node *nextRoot=n;
  for(Graph::nodeList::iterator NLI = nlist.begin(), NLE=nlist.end(); NLI!=NLE;
      ++NLI){
    if(NLI->weight>maxCount && NLI->weight<=pathNo){
      maxCount=NLI->weight;
      nextRoot=NLI->element;
      edgeRnd=NLI->randId;
      if(isStart)
	strand=NLI->randId;
    }
  }

  if(!isStart)
    assert(strand!=-1 && "strand not assigned!"); 

  assert(!(*nextRoot==*n && pathNo>0) && "No more BBs to go");
  assert(!(*nextRoot==*g.getExit() && pathNo-maxCount!=0) && "Reached exit");

  vBB.push_back(n->getElement());

  if(pathNo-maxCount==0 && *nextRoot==*g.getExit()){

    //look for strnd and edgeRnd now:
    bool has1=false, has2=false;
    //check if exit has it
    for(vector<Edge>::iterator VI=exDummy.begin(), VE=exDummy.end(); VI!=VE; 
	++VI){
      if(VI->getRandId()==edgeRnd){
	has2=true;
	break;
      }
    }

    //check if start has it
    for(vector<Edge>::iterator VI=stDummy.begin(), VE=stDummy.end(); VI!=VE; 
	++VI){
      if(VI->getRandId()==strand){
	has1=true;
	break;
      }
    }

    if(has1){
      //find backedge with endpoint vBB[1]
      for(vector<Edge>::iterator VI=be.begin(), VE=be.end(); VI!=VE; ++VI){
	assert(vBB.size()>0 && "vector too small");
	if( VI->getSecond()->getElement() == vBB[1] ){
	  //vBB[0]=VI->getFirst()->getElement();
          vBB.erase(vBB.begin());
	  break;
	}
      }
    }

    if(has2){
      //find backedge with startpoint vBB[vBB.size()-1]
      for(vector<Edge>::iterator VI=be.begin(), VE=be.end(); VI!=VE; ++VI){
	assert(vBB.size()>0 && "vector too small");
	if( VI->getFirst()->getElement() == vBB[vBB.size()-1] && 
            VI->getSecond()->getElement() == vBB[0]){
	  //vBB.push_back(VI->getSecond()->getElement());
	  break;
	}
      }
    }
    else 
      vBB.push_back(nextRoot->getElement());
   
    return;
  }

  assert(pathNo-maxCount>=0);

  return getPathFrmNode(nextRoot, vBB, pathNo-maxCount, g, stDummy, 
			exDummy, be, strand);
}


static Node *findBB(std::vector<Node *> &st, BasicBlock *BB){
  for(std::vector<Node *>::iterator si=st.begin(); si!=st.end(); ++si){
    if(((*si)->getElement())==BB){
      return *si;
    }
  }
  return NULL;
}

void getBBtrace(vector<BasicBlock *> &vBB, int pathNo, Function *M){//,
  //                vector<Instruction *> &instToErase){
  //step 1: create graph
  //Transform the cfg s.t. we have just one exit node
  
  std::vector<Node *> nodes;
  std::vector<Edge> edges;
  Node *tmp;
  Node *exitNode=0, *startNode=0;

  //Creat cfg just once for each function!
  static std::map<Function *, Graph *> graphMap; 

  //get backedges, exit and start edges for the graphs and store them
  static std::map<Function *, vector<Edge> > stMap, exMap, beMap; 
  static std::map<Function *, Value *> pathReg; //path register


  if(!graphMap[M]){
    BasicBlock *ExitNode = 0;
    for (Function::iterator I = M->begin(), E = M->end(); I != E; ++I){
      if (isa<ReturnInst>(I->getTerminator())) {
        ExitNode = &*I;
        break;
      }
    }
  
    assert(ExitNode!=0 && "exitnode not found");

    //iterating over BBs and making graph 
    //The nodes must be uniquely identified:
    //That is, no two nodes must hav same BB*
  
    //keep a map for trigger basicblocks!
    std::map<BasicBlock *, unsigned char> triggerBBs;
    //First enter just nodes: later enter edges
    for(Function::iterator BB = M->begin(), BE=M->end(); BB != BE; ++BB){
      bool cont = false;
      
      if(BB->size()==3 || BB->size() ==2){
        for(BasicBlock::iterator II = BB->begin(), IE = BB->end();
            II != IE; ++II){
          if(CallInst *callInst = dyn_cast<CallInst>(&*II)){
            //std::cerr<<*callInst;
            Function *calledFunction = callInst->getCalledFunction();
            if(calledFunction && calledFunction->getName() == "trigger"){
              triggerBBs[BB] = 9;
              cont = true;
              //std::cerr<<"Found trigger!\n";
              break;
            }
          }
        }
      }
      
      if(cont)
        continue;
      
      // const Instruction *inst = BB->getInstList().begin();
      // if(isa<CallInst>(inst)){
      // Instruction *ii1 = BB->getInstList().begin();
      // CallInst *callInst = dyn_cast<CallInst>(ii1);
      // if(callInst->getCalledFunction()->getName()=="trigger")
      // continue;
      // }
      
      Node *nd=new Node(BB);
      nodes.push_back(nd); 
      if(&*BB==ExitNode)
        exitNode=nd;
      if(&*BB==&M->front())
        startNode=nd;
    }

    assert(exitNode!=0 && startNode!=0 && "Start or exit not found!");
 
    for (Function::iterator BB = M->begin(), BE=M->end(); BB != BE; ++BB){
      if(triggerBBs[BB] == 9) 
        continue;
      
      //if(BB->size()==3)
      //if(CallInst *callInst = dyn_cast<CallInst>(&*BB->getInstList().begin()))
      //if(callInst->getCalledFunction()->getName() == "trigger")
      //continue;
      
      // if(BB->size()==2){
      //         const Instruction *inst = BB->getInstList().begin();
      //         if(isa<CallInst>(inst)){
      //           Instruction *ii1 = BB->getInstList().begin();
      //           CallInst *callInst = dyn_cast<CallInst>(ii1);
      //           if(callInst->getCalledFunction()->getName()=="trigger")
      //             continue;
      //         }
      //       }
      
      Node *nd=findBB(nodes, BB);
      assert(nd && "No node for this edge!");
      
      for(BasicBlock::succ_iterator s=succ_begin(&*BB), se=succ_end(&*BB); 
          s!=se; ++s){
        
        if(triggerBBs[*s] == 9){
          //if(!pathReg[M]){ //Get the path register for this!
          //if(BB->size()>8)
          //  if(LoadInst *ldInst = dyn_cast<LoadInst>(&*BB->getInstList().begin()))
          //    pathReg[M] = ldInst->getPointerOperand();
          //}
          continue;
        }
        //if((*s)->size()==3)
        //if(CallInst *callInst = 
        //   dyn_cast<CallInst>(&*(*s)->getInstList().begin()))
        //  if(callInst->getCalledFunction()->getName() == "trigger")
        //    continue;
        
        //  if((*s)->size()==2){
        //           const Instruction *inst = (*s)->getInstList().begin();
        //           if(isa<CallInst>(inst)){
        //             Instruction *ii1 = (*s)->getInstList().begin();
        //             CallInst *callInst = dyn_cast<CallInst>(ii1);
        //             if(callInst->getCalledFunction()->getName()=="trigger")
        //               continue;
        //           }
        //         }
        
        Node *nd2 = findBB(nodes,*s);
        assert(nd2 && "No node for this edge!");
        Edge ed(nd,nd2,0);
        edges.push_back(ed);
      }
    }
  
    graphMap[M]= new Graph(nodes,edges, startNode, exitNode);
 
    Graph *g = graphMap[M];

    if (M->size() <= 1) return; //uninstrumented 
    
    //step 2: getBackEdges
    //vector<Edge> be;
    std::map<Node *, int> nodePriority;
    g->getBackEdges(beMap[M], nodePriority);
    
    //step 3: add dummy edges
    //vector<Edge> stDummy;
    //vector<Edge> exDummy;
    addDummyEdges(stMap[M], exMap[M], *g, beMap[M]);
    
    //step 4: value assgn to edges
    int numPaths = valueAssignmentToEdges(*g, nodePriority, beMap[M]);
  }
  
  
  //step 5: now travel from root, select max(edge) < pathNo, 
  //and go on until reach the exit
  getPathFrmNode(graphMap[M]->getRoot(), vBB, pathNo, *graphMap[M], 
                 stMap[M], exMap[M], beMap[M], -1);
  

  //post process vBB to locate instructions to be erased
  /*
  if(pathReg[M]){
    for(vector<BasicBlock *>::iterator VBI = vBB.begin(), VBE = vBB.end();
        VBI != VBE; ++VBI){
      for(BasicBlock::iterator BBI = (*VBI)->begin(), BBE = (*VBI)->end();
          BBI != BBE; ++BBI){
        if(LoadInst *ldInst = dyn_cast<LoadInst>(&*BBI)){
          if(pathReg[M] == ldInst->getPointerOperand())
            instToErase.push_back(ldInst);
        }
        else if(StoreInst *stInst = dyn_cast<StoreInst>(&*BBI)){
          if(pathReg[M] == stInst->getPointerOperand())
            instToErase.push_back(stInst);
        }
      }
    }
  }
  */
}
