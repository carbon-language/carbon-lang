//===-- EdgeCode.cpp - generate LLVM instrumentation code --------*- C++ -*--=//
//It implements the class EdgeCode: which provides 
//support for inserting "appropriate" instrumentation at
//designated points in the graph
//
//It also has methods to insert initialization code in 
//top block of cfg
//===----------------------------------------------------------------------===//

#include "Graph.h"
#include "llvm/BasicBlock.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/iMemory.h"
#include "llvm/iTerminators.h"
#include "llvm/iOther.h"
#include "llvm/iOperators.h"
#include "llvm/iPHINode.h"

using std::vector;

//get the code to be inserted on the edge
//This is determined from cond (1-6)
void getEdgeCode::getCode(Instruction *rInst, 
			  Instruction *countInst, 
			  Function *M, 
			  BasicBlock *BB){
  
  BasicBlock::InstListType& instList=BB->getInstList();
  BasicBlock::iterator here=instList.begin();
  
  //case: r=k code to be inserted
  switch(cond){
  case 1:{
    Value *val=ConstantSInt::get(Type::IntTy,inc);
    Instruction *stInst=new StoreInst(val, rInst);
    here=instList.insert(here,stInst)+1;
    break;
    }

  //case: r=0 to be inserted
  case 2:{
    Value *val=ConstantSInt::get(Type::IntTy,0);
    Instruction *stInst=new StoreInst(val, rInst);
    here=instList.insert(here,stInst)+1;
    break;
  }
    
  //r+=k
  case 3:{
    Instruction *ldInst=new LoadInst(rInst, "ti1");
    Value *val=ConstantSInt::get(Type::IntTy,inc);
    Instruction *addIn=BinaryOperator::
      create(Instruction::Add, ldInst, val,"ti2");
    
    Instruction *stInst=new StoreInst(addIn, rInst);
    here=instList.insert(here,ldInst)+1;
    here=instList.insert(here,addIn)+1;
    here=instList.insert(here,stInst)+1;
    break;
  }

  //count[inc]++
  case 4:{
    Instruction *ldInst=new 
      LoadInst(countInst,vector<Value *>
	       (1,ConstantUInt::get(Type::UIntTy, inc)), "ti1");
    Value *val=ConstantSInt::get(Type::IntTy,1);
    Instruction *addIn=BinaryOperator::
      create(Instruction::Add, ldInst, val,"ti2");

    assert(inc>=0 && "IT MUST BE POSITIVE NOW");
    Instruction *stInst=new 
      StoreInst(addIn, countInst, vector<Value *>
		(1, ConstantUInt::get(Type::UIntTy,inc)));
    
    here=instList.insert(here,ldInst)+1;
    here=instList.insert(here,addIn)+1;
    here=instList.insert(here,stInst)+1;
    break;
  }

  //case: count[r+inc]++
  case 5:{
    //ti1=inc+r
    Instruction *ldIndex=new LoadInst(rInst, "ti1");
    Value *val=ConstantSInt::get(Type::IntTy,inc);
    Instruction *addIndex=BinaryOperator::
      create(Instruction::Add, ldIndex, val,"ti2");
    
    //now load count[addIndex]
    Instruction *castInst=new CastInst(addIndex, 
				       Type::UIntTy,"ctin");
    Instruction *ldInst=new 
      LoadInst(countInst, vector<Value *>(1,castInst), "ti3");
    Value *cons=ConstantSInt::get(Type::IntTy,1);
    
    //count[addIndex]++
    Instruction *addIn=BinaryOperator::
      create(Instruction::Add, ldInst, cons,"ti4");
    Instruction *stInst=new 
      StoreInst(addIn, countInst, 
		vector<Value *>(1,castInst));
    
    here=instList.insert(here,ldIndex)+1;
    here=instList.insert(here,addIndex)+1;
    here=instList.insert(here,castInst)+1;
    here=instList.insert(here,ldInst)+1;
    here=instList.insert(here,addIn)+1;
    here=instList.insert(here,stInst)+1;
    break;
  }

    //case: count[r]+
  case 6:{
    //ti1=inc+r
    Instruction *ldIndex=new LoadInst(rInst, "ti1");

    //now load count[addIndex]
    Instruction *castInst2=new 
      CastInst(ldIndex, Type::UIntTy,"ctin");
    Instruction *ldInst=new 
      LoadInst(countInst, vector<Value *>(1,castInst2), "ti2");
    Value *cons=ConstantSInt::get(Type::IntTy,1);

    //count[addIndex]++
    Instruction *addIn=BinaryOperator::
      create(Instruction::Add, ldInst, cons,"ti3"); 
    Instruction *stInst=new 
      StoreInst(addIn, countInst, vector<Value *>(1,castInst2));
    
    here=instList.insert(here,ldIndex)+1;
    here=instList.insert(here,castInst2)+1;
    here=instList.insert(here,ldInst)+1;
    here=instList.insert(here,addIn)+1;
    here=instList.insert(here,stInst)+1;
    break;
  }
    
  }
  //now check for cdIn and cdOut
  //first put cdOut
  if(cdOut!=NULL){
    cdOut->getCode(rInst, countInst, M, BB);
  }
  if(cdIn!=NULL){
    cdIn->getCode(rInst, countInst, M, BB);
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
		   Instruction *rVar, 
		   Instruction *countVar){
  //rVar is variable r, 
  //countVar is array Count, and these are allocatted outside
  
  //store uint 0, uint *%R, uint 0
  vector<Value *> idx;
  idx.push_back(ConstantUInt::get(Type::UIntTy, 0));
  Instruction *stInstr=new StoreInst(ConstantInt::get(Type::IntTy, 0), rVar, 
				     idx);

  //now push all instructions in front of the BB
  BasicBlock::InstListType& instList=front->getInstList();
  BasicBlock::iterator here=instList.begin();
  here=front->getInstList().insert(here, rVar)+1;
  here=front->getInstList().insert(here,countVar)+1;
  
  //Initialize Count[...] with 0
  for(int i=0;i<k; i++){
    Instruction *stInstrC=new 
      StoreInst(ConstantInt::get(Type::IntTy, 0), 
		countVar, std::vector<Value *>
		(1,ConstantUInt::get(Type::UIntTy, i))); 
    here=front->getInstList().insert(here,stInstrC)+1;
  }
  
  here=front->getInstList().insert(here,stInstr)+1;
}


//insert a basic block with appropriate code
//along a given edge
void insertBB(Edge ed,
	      getEdgeCode *edgeCode, 
	      Instruction *rInst, 
	      Instruction *countInst){

  BasicBlock* BB1=ed.getFirst()->getElement();
  BasicBlock* BB2=ed.getSecond()->getElement();
  
  DEBUG(cerr << "Edges with codes ######################\n";
        cerr << BB1->getName() << "->" << BB2->getName() << "\n";
        cerr << "########################\n");

  //We need to insert a BB between BB1 and BB2 
  TerminatorInst *TI=BB1->getTerminator();
  BasicBlock *newBB=new BasicBlock("counter", BB1->getParent());

  //get code for the new BB
  edgeCode->getCode(rInst, countInst, BB1->getParent(), newBB);
 
  //Is terminator a branch instruction?
  //then we need to change branch destinations to include new BB

  BranchInst *BI=cast<BranchInst>(TI);
 
  if(BI->isUnconditional()){
    BI->setUnconditionalDest(newBB);
    Instruction *newBI2=new BranchInst(BB2);
    newBB->getInstList().push_back(newBI2);
  }
  else{
    Value *cond=BI->getCondition();
    BasicBlock *fB, *tB;
   
    if(BI->getSuccessor(0)==BB2){
      tB=newBB;
      fB=BI->getSuccessor(1);
    }
    else{
      fB=newBB;
      tB=BI->getSuccessor(0);
    }
   
    delete BB1->getInstList().pop_back();
    Instruction *newBI=new BranchInst(tB,fB,cond);
    Instruction *newBI2=new BranchInst(BB2);
    BB1->getInstList().push_back(newBI);
    newBB->getInstList().push_back(newBI2);
  }
  
  //now iterate over BB2, and set its Phi nodes right
  for(BasicBlock::iterator BB2Inst=BB2->begin(), BBend=BB2->end(); 
      BB2Inst!=BBend; ++BB2Inst){
   
    if(PHINode *phiInst=dyn_cast<PHINode>(*BB2Inst)){
      DEBUG(cerr<<"YYYYYYYYYYYYYYYYY\n");

      int bbIndex=phiInst->getBasicBlockIndex(BB1);
      if(bbIndex>=0)
	phiInst->setIncomingBlock(bbIndex, newBB);
    }
  }
}
