//===-- EdgeCode.cpp - generate LLVM instrumentation code -----------------===//
//It implements the class EdgeCode: which provides 
//support for inserting "appropriate" instrumentation at
//designated points in the graph
//
//It also has methods to insert initialization code in 
//top block of cfg
//===----------------------------------------------------------------------===//

#include "Graph.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/iMemory.h"
#include "llvm/iTerminators.h"
#include "llvm/iOther.h"
#include "llvm/iOperators.h"
#include "llvm/iPHINode.h"
#include "llvm/Module.h"
#include "Config/stdio.h"

#define INSERT_LOAD_COUNT
#define INSERT_STORE

using std::vector;


static void getTriggerCode(Module *M, BasicBlock *BB, int MethNo, Value *pathNo,
                           Value *cnt, Instruction *rInst){ 
  
  vector<Value *> tmpVec;
  tmpVec.push_back(Constant::getNullValue(Type::LongTy));
  tmpVec.push_back(Constant::getNullValue(Type::LongTy));
  Instruction *Idx = new GetElementPtrInst(cnt, tmpVec, "");//,
  BB->getInstList().push_back(Idx);

  const Type *PIntTy = PointerType::get(Type::IntTy);
  Function *trigMeth = M->getOrInsertFunction("trigger", Type::VoidTy, 
                                              Type::IntTy, Type::IntTy,
                                              PIntTy, PIntTy, 0);
  assert(trigMeth && "trigger method could not be inserted!");

  vector<Value *> trargs;

  trargs.push_back(ConstantSInt::get(Type::IntTy,MethNo));
  trargs.push_back(pathNo);
  trargs.push_back(Idx);
  trargs.push_back(rInst);

  Instruction *callInst=new CallInst(trigMeth, trargs, "");//, BB->begin());
  BB->getInstList().push_back(callInst);
  //triggerInst = new CallInst(trigMeth, trargs, "");//, InsertPos);
}


//get the code to be inserted on the edge
//This is determined from cond (1-6)
void getEdgeCode::getCode(Instruction *rInst, Value *countInst, 
			  Function *M, BasicBlock *BB, 
                          vector<Value *> &retVec){
  
  //Instruction *InsertPos = BB->getInstList().begin();
  
  //now check for cdIn and cdOut
  //first put cdOut
  if(cdOut!=NULL){
    cdOut->getCode(rInst, countInst, M, BB, retVec);
  }
  
  if(cdIn!=NULL){
    cdIn->getCode(rInst, countInst, M, BB, retVec);
  }

  //case: r=k code to be inserted
  switch(cond){
  case 1:{
    Value *val=ConstantSInt::get(Type::IntTy,inc);
#ifdef INSERT_STORE
    Instruction *stInst=new StoreInst(val, rInst);//, InsertPos);
    BB->getInstList().push_back(stInst);
#endif
    break;
    }

  //case: r=0 to be inserted
  case 2:{
#ifdef INSERT_STORE
    Instruction *stInst = new StoreInst(ConstantSInt::getNullValue(Type::IntTy), rInst);//, InsertPos);
     BB->getInstList().push_back(stInst);
#endif
    break;
  }
    
  //r+=k
  case 3:{
    Instruction *ldInst = new LoadInst(rInst, "ti1");//, InsertPos);
    BB->getInstList().push_back(ldInst);
    Value *val = ConstantSInt::get(Type::IntTy,inc);
    Instruction *addIn = BinaryOperator::create(Instruction::Add, ldInst, val,
                                          "ti2");//, InsertPos);
    BB->getInstList().push_back(addIn);
#ifdef INSERT_STORE
    Instruction *stInst = new StoreInst(addIn, rInst);//, InsertPos);
    BB->getInstList().push_back(stInst);
#endif
    break;
  }

  //count[inc]++
  case 4:{
    vector<Value *> tmpVec;
    tmpVec.push_back(Constant::getNullValue(Type::LongTy));
    tmpVec.push_back(ConstantSInt::get(Type::LongTy, inc));
    Instruction *Idx = new GetElementPtrInst(countInst, tmpVec, "");//,

    //Instruction *Idx = new GetElementPtrInst(countInst, 
    //           vector<Value*>(1,ConstantSInt::get(Type::LongTy, inc)),
    //                                       "");//, InsertPos);
    BB->getInstList().push_back(Idx);

    Instruction *ldInst=new LoadInst(Idx, "ti1");//, InsertPos);
    BB->getInstList().push_back(ldInst);
 
    Value *val = ConstantSInt::get(Type::IntTy, 1);
    //Instruction *addIn =
    Instruction *newCount =
      BinaryOperator::create(Instruction::Add, ldInst, val,"ti2");
    BB->getInstList().push_back(newCount);
    

#ifdef INSERT_STORE
    //Instruction *stInst=new StoreInst(addIn, Idx, InsertPos);
    Instruction *stInst=new StoreInst(newCount, Idx);//, InsertPos);
    BB->getInstList().push_back(stInst);
#endif
    
    Value *trAddIndex = ConstantSInt::get(Type::IntTy,inc);

    retVec.push_back(newCount);
    retVec.push_back(trAddIndex);
    //insert trigger
    //getTriggerCode(M->getParent(), BB, MethNo, 
    //	   ConstantSInt::get(Type::IntTy,inc), newCount, triggerInst);
    //end trigger code

    assert(inc>=0 && "IT MUST BE POSITIVE NOW");
    break;
  }

  //case: count[r+inc]++
  case 5:{
   
    //ti1=inc+r
    Instruction *ldIndex=new LoadInst(rInst, "ti1");//, InsertPos);
    BB->getInstList().push_back(ldIndex);

    Value *val=ConstantSInt::get(Type::IntTy,inc);
    Instruction *addIndex=BinaryOperator::
      create(Instruction::Add, ldIndex, val,"ti2");//, InsertPos);
    BB->getInstList().push_back(addIndex);
    
    //now load count[addIndex]
    Instruction *castInst=new CastInst(addIndex, 
				       Type::LongTy,"ctin");//, InsertPos);
    BB->getInstList().push_back(castInst);

    vector<Value *> tmpVec;
    tmpVec.push_back(Constant::getNullValue(Type::LongTy));
    tmpVec.push_back(castInst);
    Instruction *Idx = new GetElementPtrInst(countInst, tmpVec, "");//,
    //                                             InsertPos);
    BB->getInstList().push_back(Idx);

    Instruction *ldInst=new LoadInst(Idx, "ti3");//, InsertPos);
    BB->getInstList().push_back(ldInst);

    Value *cons=ConstantSInt::get(Type::IntTy,1);
    //count[addIndex]++
    //std::cerr<<"Type ldInst:"<<ldInst->getType()<<"\t cons:"<<cons->getType()<<"\n";
    Instruction *newCount = BinaryOperator::create(Instruction::Add, ldInst, 
                                                   cons,"");
    BB->getInstList().push_back(newCount);
    
#ifdef INSERT_STORE
    Instruction *stInst = new StoreInst(newCount, Idx);//, InsertPos);
    BB->getInstList().push_back(stInst);
#endif

    retVec.push_back(newCount);
    retVec.push_back(addIndex);
    //insert trigger
    //getTriggerCode(M->getParent(), BB, MethNo, addIndex, newCount, triggerInst);
    //end trigger code

    break;
  }

    //case: count[r]+
  case 6:{
    //ti1=inc+r
    Instruction *ldIndex=new LoadInst(rInst, "ti1");//, InsertPos);
    BB->getInstList().push_back(ldIndex);

    //now load count[addIndex]
    Instruction *castInst2=new CastInst(ldIndex, Type::LongTy,"ctin");
    BB->getInstList().push_back(castInst2);

    vector<Value *> tmpVec;
    tmpVec.push_back(Constant::getNullValue(Type::LongTy));
    tmpVec.push_back(castInst2);
    Instruction *Idx = new GetElementPtrInst(countInst, tmpVec, "");//,

    //Instruction *Idx = new GetElementPtrInst(countInst, 
    //                                       vector<Value*>(1,castInst2), "");
    
    BB->getInstList().push_back(Idx);
    
    Instruction *ldInst=new LoadInst(Idx, "ti2");//, InsertPos);
    BB->getInstList().push_back(ldInst);

    Value *cons=ConstantSInt::get(Type::IntTy,1);

    //count[addIndex]++
    Instruction *newCount = BinaryOperator::create(Instruction::Add, ldInst,
                                                   cons,"ti3");
    BB->getInstList().push_back(newCount);

#ifdef INSERT_STORE
    Instruction *stInst = new StoreInst(newCount, Idx);//, InsertPos);
    BB->getInstList().push_back(stInst);
#endif

    retVec.push_back(newCount);
    retVec.push_back(ldIndex);
    break;
  }
    
  }
}



//Insert the initialization code in the top BB
//this includes initializing r, and count
//r is like an accumulator, that 
//keeps on adding increments as we traverse along a path
//and at the end of the path, r contains the path
//number of that path
//Count is an array, where Count[k] represents
//the number of executions of path k
void insertInTopBB(BasicBlock *front, 
		   int k, 
		   Instruction *rVar, Value *threshold){
  //rVar is variable r, 
  //countVar is count[]

  Value *Int0 = ConstantInt::get(Type::IntTy, 0);
  
  //now push all instructions in front of the BB
  BasicBlock::iterator here=front->begin();
  front->getInstList().insert(here, rVar);
  //front->getInstList().insert(here,countVar);
  
  //Initialize Count[...] with 0

  //for (int i=0;i<k; i++){
  //Value *GEP2 = new GetElementPtrInst(countVar,
  //                      vector<Value *>(1,ConstantSInt::get(Type::LongTy, i)),
  //                                    "", here);
  //new StoreInst(Int0, GEP2, here);
  //}

  //store uint 0, uint *%R
  new StoreInst(Int0, rVar, here);
}


//insert a basic block with appropriate code
//along a given edge
void insertBB(Edge ed,
	      getEdgeCode *edgeCode, 
	      Instruction *rInst, 
	      Value *countInst, 
	      int numPaths, int Methno, Value *threshold){

  BasicBlock* BB1=ed.getFirst()->getElement();
  BasicBlock* BB2=ed.getSecond()->getElement();
  
#ifdef DEBUG_PATH_PROFILES
  //debugging info
  cerr<<"Edges with codes ######################\n";
  cerr<<BB1->getName()<<"->"<<BB2->getName()<<"\n";
  cerr<<"########################\n";
#endif
  
  //We need to insert a BB between BB1 and BB2 
  TerminatorInst *TI=BB1->getTerminator();
  BasicBlock *newBB=new BasicBlock("counter", BB1->getParent());

  //get code for the new BB
  vector<Value *> retVec;

  edgeCode->getCode(rInst, countInst, BB1->getParent(), newBB, retVec);

  BranchInst *BI =  cast<BranchInst>(TI);

  //Is terminator a branch instruction?
  //then we need to change branch destinations to include new BB

  if(BI->isUnconditional()){
    BI->setUnconditionalDest(newBB);
  }
  else{
      if(BI->getSuccessor(0)==BB2)
      BI->setSuccessor(0, newBB);
    
    if(BI->getSuccessor(1)==BB2)
      BI->setSuccessor(1, newBB);
  }

  BasicBlock *triggerBB = NULL;
  if(retVec.size()>0){
    triggerBB = new BasicBlock("trigger", BB1->getParent());
    getTriggerCode(BB1->getParent()->getParent(), triggerBB, Methno, 
                   retVec[1], countInst, rInst);//retVec[0]);

    //Instruction *castInst = new CastInst(retVec[0], Type::IntTy, "");
    Instruction *etr = new LoadInst(threshold, "threshold");
    
    //std::cerr<<"type1: "<<etr->getType()<<" type2: "<<retVec[0]->getType()<<"\n"; 
    Instruction *cmpInst = new SetCondInst(Instruction::SetLE, etr, 
                                           retVec[0], "");
    Instruction *newBI2 = new BranchInst(triggerBB, BB2, cmpInst);
    //newBB->getInstList().push_back(castInst);
    newBB->getInstList().push_back(etr);
    newBB->getInstList().push_back(cmpInst);
    newBB->getInstList().push_back(newBI2);
    
    //triggerBB->getInstList().push_back(triggerInst);
    Instruction *triggerBranch = new BranchInst(BB2);
    triggerBB->getInstList().push_back(triggerBranch);
  }
  else{
    Instruction *newBI2=new BranchInst(BB2);
    newBB->getInstList().push_back(newBI2);
  }

  //now iterate over BB2, and set its Phi nodes right
  for(BasicBlock::iterator BB2Inst = BB2->begin(), BBend = BB2->end(); 
      BB2Inst != BBend; ++BB2Inst){
   
    if(PHINode *phiInst=dyn_cast<PHINode>(BB2Inst)){
      int bbIndex=phiInst->getBasicBlockIndex(BB1);
      assert(bbIndex>=0);
      phiInst->setIncomingBlock(bbIndex, newBB);

      ///check if trigger!=null, then add value corresponding to it too!
      if(retVec.size()>0){
        assert(triggerBB && "BasicBlock with trigger should not be null!");
        Value *vl = phiInst->getIncomingValue((unsigned int)bbIndex);
        phiInst->addIncoming(vl, triggerBB);
      }
    }
  }
}

