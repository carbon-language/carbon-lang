//===-- InstLoops.cpp ---------------------------------------- ---*- C++ -*--=//
// Pass to instrument loops
//
// At every backedge, insert a counter for that backedge and a call function
//===----------------------------------------------------------------------===//

#include "llvm/Reoptimizer/InstLoops.h"
#include "llvm/Support/CFG.h"
#include "llvm/Constants.h"
#include "llvm/iMemory.h"
#include "llvm/GlobalVariable.h"
#include "llvm/DerivedTypes.h"
#include "llvm/iOther.h"
#include "llvm/iOperators.h"
#include "llvm/iTerminators.h"
#include "llvm/iPHINode.h"
#include "llvm/Module.h"
#include "llvm/Function.h"
#include "llvm/Pass.h"

//this is used to color vertices
//during DFS

enum Color{
  WHITE,
  GREY,
  BLACK
};

struct InstLoops : public FunctionPass {
  bool runOnFunction(Function &F);
};

static RegisterOpt<InstLoops> X("instloops", "Instrument backedges for profiling");

// createInstLoopsPass - Create a new pass to add path profiling
//
Pass *createInstLoopsPass() {
  return new InstLoops();
}


//helper function to get back edges: it is called by 
//the "getBackEdges" function below
void getBackEdgesVisit(BasicBlock *u,
                       std::map<BasicBlock *, Color > &color,
                       std::map<BasicBlock *, int > &d, 
                       int &time, Value *threshold) {
  
  color[u]=GREY;
  time++;
  d[u]=time;

  for(BasicBlock::succ_iterator vl = succ_begin(u), 
	ve = succ_end(u); vl != ve; ++vl){
    
    BasicBlock *BB = *vl;

    if(color[BB]!=GREY && color[BB]!=BLACK){
      getBackEdgesVisit(BB, color, d, time, threshold);
    }
    
    //now checking for d and f vals
    if(color[BB]==GREY){
      //so v is ancestor of u if time of u > time of v
      if(d[u] >= d[BB]){
        //insert a new basic block: modify terminator accordingly!
        BasicBlock *newBB = new BasicBlock("", u->getParent());
        BranchInst *ti = cast<BranchInst>(u->getTerminator());
        unsigned char index = 1;
        if(ti->getSuccessor(0) == BB){
          index = 0;
        }
        assert(ti->getNumSuccessors() > index && "Not enough successors!");
        ti->setSuccessor(index, newBB);

        //insert global variable of type int
        Constant *initializer = Constant::getNullValue(Type::IntTy);
        GlobalVariable *countVar = new GlobalVariable(Type::IntTy, false, true, 
                                                      initializer, 
                                                      "loopCounter", 
                                                      u->getParent()->getParent());
        
        //load the variable
        Instruction *ldInst = new LoadInst(countVar,"");
        
        //increment
        Instruction *addIn = 
          BinaryOperator::create(Instruction::Add, ldInst, 
                                 ConstantSInt::get(Type::IntTy,1), "");

        //store
        Instruction *stInst = new StoreInst(addIn, countVar);


        Instruction *etr = new LoadInst(threshold, "threshold");
        Instruction *cmpInst = new SetCondInst(Instruction::SetLE, etr, 
                                               addIn, "");
        
        BasicBlock *callTrigger = new BasicBlock("", u->getParent());
        //branch to calltrigger, or *vl
        Instruction *newBr = new BranchInst(callTrigger, BB, cmpInst);

        BasicBlock::InstListType &lt = newBB->getInstList();

        lt.push_back(ldInst);
        lt.push_back(addIn);
        lt.push_back(stInst);
        lt.push_back(etr);
        lt.push_back(cmpInst);
        lt.push_back(newBr);

        //Now add instructions to the triggerCall BB
        //now create a call function
        //call llvm_first_trigger(int *x);
        std::vector<const Type*> inCountArgs;
        inCountArgs.push_back(PointerType::get(Type::IntTy));
 
        const FunctionType *cFty = FunctionType::get(Type::VoidTy, inCountArgs, 
                                                     false);
        Function *inCountMth = 
          u->getParent()->getParent()->getOrInsertFunction("llvm_first_trigger", cFty);
        
        assert(inCountMth && "Initialize method could not be inserted!");

        std::vector<Value *> iniArgs;
        iniArgs.push_back(countVar);
        Instruction *call = new CallInst(inCountMth, iniArgs, "");
        callTrigger->getInstList().push_back(call);
        callTrigger->getInstList().push_back(new BranchInst(BB));
      
        //now iterate over *vl, and set its Phi nodes right
        for(BasicBlock::iterator BB2Inst = BB->begin(), BBend = BB->end(); 
            BB2Inst != BBend; ++BB2Inst){
        
          if(PHINode *phiInst=dyn_cast<PHINode>(&*BB2Inst)){
            int bbIndex = phiInst->getBasicBlockIndex(u);
            if(bbIndex>=0){
              phiInst->setIncomingBlock(bbIndex, newBB);
          
              Value *val = phiInst->getIncomingValue((unsigned int)bbIndex);
              phiInst->addIncoming(val, callTrigger);
            }
          }
        }
      }
    }
  }
  color[u]=BLACK;//done with visiting the node and its neighbors
}


//getting the backedges in a graph
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
void getBackEdges(Function &F, Value *threshold){
  std::map<BasicBlock *, Color > color;
  std::map<BasicBlock *, int> d;
  int time=0;
  getBackEdgesVisit(F.begin(), color, d, time, threshold);
}

//Per function pass for inserting counters and call function
bool InstLoops::runOnFunction(Function &F){
  
  static GlobalVariable *threshold = NULL;
  static bool insertedThreshold = false;
  
   if(!insertedThreshold){
    threshold = new GlobalVariable(Type::IntTy, false, false, 0,
                                                   "reopt_threshold");

    F.getParent()->getGlobalList().push_back(threshold);
    insertedThreshold = true;
  }

  if(F.getName() == "main"){
    //intialize threshold
    std::vector<const Type*> initialize_args;
    initialize_args.push_back(PointerType::get(Type::IntTy));
    
    const FunctionType *Fty = FunctionType::get(Type::VoidTy, initialize_args,
                                                false);
    Function *initialMeth = F.getParent()->getOrInsertFunction("reoptimizerInitialize", Fty);
    assert(initialMeth && "Initialize method could not be inserted!");
    
    std::vector<Value *> trargs;
    trargs.push_back(threshold);
  
    new CallInst(initialMeth, trargs, "", F.begin()->begin());
  }

  assert(threshold && "GlobalVariable threshold not defined!");

  if(F.isExternal()) {
    return false;
  }

  getBackEdges(F, threshold);
  
  return true;
}
