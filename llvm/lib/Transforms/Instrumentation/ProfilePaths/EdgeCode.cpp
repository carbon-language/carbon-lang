//===-- EdgeCode.cpp - generate LLVM instrumentation code --------*- C++ -*--=//
//It implements the class EdgeCode: which provides 
//support for inserting "appropriate" instrumentation at
//designated points in the graph
//
//It also has methods to insert initialization code in 
//top block of cfg
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/Graph.h"
#include "llvm/BasicBlock.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/iMemory.h"
#include "llvm/iTerminators.h"
#include "llvm/iOther.h"
#include "llvm/iOperators.h"
#include "llvm/iPHINode.h"
#include "llvm/Module.h"
#include "llvm/SymbolTable.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Constants.h"
#include "llvm/BasicBlock.h"
#include "llvm/Function.h"
#include <string.h>
#include <stdio.h>
#include <iostream>

#define INSERT_LOAD_COUNT
#define INSERT_STORE

using std::vector;


void getTriggerCode(Module *M, BasicBlock *BB, int MethNo, Value *pathNo, 
		    Value *cnt){ 
  static int i=-1;
  i++;
  char gstr[100];
  sprintf(gstr,"globalVar%d",i);
  std::string globalVarName=gstr;
  SymbolTable *ST = M->getSymbolTable();
  vector<const Type*> args;
  //args.push_back(PointerType::get(Type::SByteTy));
  args.push_back(Type::IntTy);
  args.push_back(Type::IntTy);
  args.push_back(Type::IntTy);
  const FunctionType *MTy =
    FunctionType::get(Type::VoidTy, args, false);

  //  Function *triggerMeth = M->getOrInsertFunction("trigger", MTy);
  Function *trigMeth = M->getOrInsertFunction("trigger", MTy);
  assert(trigMeth && "trigger method could not be inserted!");
  //if (Value *triggerMeth = ST->lookup(PointerType::get(MTy), "trigger")) {
  //Function *trigMeth = cast<Function>(triggerMeth);
  vector<Value *> trargs;

  //pred_iterator piter=BB->pred_begin();
  //std::string predName = "uu";//BB->getName();
  //Constant *bbName=ConstantArray::get(predName);//BB->getName());
  //GlobalVariable *gbl=new GlobalVariable(ArrayType::get(Type::SByteTy, 
  //					predName.size()+1), 
  //				 true, true, bbName, gstr);
  
  //M->getGlobalList().push_back(gbl);

  vector<Value *> elargs;
  elargs.push_back(ConstantUInt::get(Type::UIntTy, 0));
  elargs.push_back(ConstantUInt::get(Type::UIntTy, 0));

  // commented out bb name frm which its called
  //Instruction *getElmntInst=new GetElementPtrInst(gbl,elargs,"elmntInst");
  
  //trargs.push_back(ConstantArray::get(BB->getName()));
  
  //trargs.push_back(getElmntInst);
  //trargs.push_back(bbName);

  trargs.push_back(ConstantSInt::get(Type::IntTy,MethNo));
    
  //trargs.push_back(ConstantSInt::get(Type::IntTy,-1));//erase this
  trargs.push_back(pathNo);
  trargs.push_back(cnt);
  Instruction *callInst=new CallInst(trigMeth,trargs);

  BasicBlock::InstListType& instList=BB->getInstList();
  BasicBlock::iterator here=instList.begin();
  //here = ++instList.insert(here, getElmntInst);
  instList.insert(here,callInst);
}


//get the code to be inserted on the edge
//This is determined from cond (1-6)
void getEdgeCode::getCode(Instruction *rInst, 
			  Instruction *countInst, 
			  Function *M, 
			  BasicBlock *BB, int numPaths, int MethNo){
  
  BasicBlock::InstListType& instList=BB->getInstList();
  BasicBlock::iterator here=instList.begin();
  
  //case: r=k code to be inserted
  switch(cond){
  case 1:{
    Value *val=ConstantSInt::get(Type::IntTy,inc);
#ifdef INSERT_STORE
    Instruction *stInst=new StoreInst(val, rInst);
    here = ++instList.insert(here,stInst);
#endif
    break;
    }

  //case: r=0 to be inserted
  case 2:{
    Value *val=ConstantSInt::get(Type::IntTy,0);
#ifdef INSERT_STORE
    Instruction *stInst=new StoreInst(val, rInst);
    here = ++instList.insert(here,stInst);
#endif
    break;
  }
    
  //r+=k
  case 3:{
    
    Instruction *ldInst=new LoadInst(rInst, "ti1");
    Value *val=ConstantSInt::get(Type::IntTy,inc);
    Instruction *addIn=BinaryOperator::
      create(Instruction::Add, ldInst, val,"ti2");
#ifdef INSERT_STORE
    Instruction *stInst=new StoreInst(addIn, rInst);
#endif
    here = ++instList.insert(here,ldInst);
    here = ++instList.insert(here,addIn);
#ifdef INSERT_STORE
    here = ++instList.insert(here,stInst);
#endif
    break;
  }

  //count[inc]++
  case 4:{
    
    assert(inc>=0 && inc<=numPaths && "inc out of bound!");
   
    Instruction *Idx = new GetElementPtrInst(countInst, 
                 vector<Value*>(1,ConstantUInt::get(Type::UIntTy, inc)));

    Instruction *ldInst=new LoadInst(Idx, "ti1");
 
    Value *val = ConstantSInt::get(Type::IntTy, 1);
    Instruction *addIn =
      BinaryOperator::create(Instruction::Add, ldInst, val,"ti2");

    //insert trigger
    getTriggerCode(M->getParent(), BB, MethNo, 
		   ConstantSInt::get(Type::IntTy,inc), addIn);
    here=instList.begin();
    //end trigger code

    assert(inc>=0 && "IT MUST BE POSITIVE NOW");
#ifdef INSERT_STORE
    Instruction *stInst=new StoreInst(addIn, Idx);
#endif
    here = ++instList.insert(here,Idx);
    here = ++instList.insert(here,ldInst);
    here = ++instList.insert(here,addIn);
#ifdef INSERT_STORE
    here = ++instList.insert(here,stInst);
#endif
    break;
  }

  //case: count[r+inc]++
  case 5:{
    
    //ti1=inc+r
    Instruction *ldIndex=new LoadInst(rInst, "ti1");
    Value *val=ConstantSInt::get(Type::IntTy,inc);
    Instruction *addIndex=BinaryOperator::
      create(Instruction::Add, ldIndex, val,"ti2");
    //erase following 1 line
    //Value *valtemp=ConstantSInt::get(Type::IntTy,999);
    //now load count[addIndex]
    
    Instruction *castInst=new CastInst(addIndex, 
				       Type::UIntTy,"ctin");
    Instruction *Idx = new GetElementPtrInst(countInst, 
                                             vector<Value*>(1,castInst));

    Instruction *ldInst=new LoadInst(Idx, "ti3");
    Value *cons=ConstantSInt::get(Type::IntTy,1);
    //count[addIndex]++
    Instruction *addIn=BinaryOperator::
      create(Instruction::Add, ldInst, cons,"ti4");
    
    //insert trigger
    getTriggerCode(M->getParent(), BB, MethNo, addIndex, addIn);
    here=instList.begin();
    //end trigger code
    
#ifdef INSERT_STORE
    ///*
    Instruction *stInst=new StoreInst(addIn, Idx);

    //*/
#endif
    here = ++instList.insert(here,ldIndex);
    here = ++instList.insert(here,addIndex);
    here = ++instList.insert(here,castInst);
    here = ++instList.insert(here,Idx);
    here = ++instList.insert(here,ldInst);
    here = ++instList.insert(here,addIn);
#ifdef INSERT_STORE
    here = ++instList.insert(here,stInst);
#endif
    break;
  }

    //case: count[r]+
  case 6:{
    
    //ti1=inc+r
    Instruction *ldIndex=new LoadInst(rInst, "ti1");
    
    //now load count[addIndex]
    Instruction *castInst2=new CastInst(ldIndex, Type::UIntTy,"ctin");
    Instruction *Idx = new GetElementPtrInst(countInst, 
                                             vector<Value*>(1,castInst2));
    
    Instruction *ldInst=new LoadInst(Idx, "ti2");
    Value *cons=ConstantSInt::get(Type::IntTy,1);

    //count[addIndex]++
    Instruction *addIn=BinaryOperator::create(Instruction::Add, ldInst,
                                              cons,"ti3"); 

    //insert trigger
    getTriggerCode(M->getParent(), BB, MethNo, ldIndex, addIn);
    here=instList.begin();
    //end trigger code
#ifdef INSERT_STORE
    Instruction *stInst=new StoreInst(addIn, Idx);
#endif
    here = ++instList.insert(here,ldIndex);
    here = ++instList.insert(here,castInst2);
    here = ++instList.insert(here,Idx);
    here = instList.insert(here,ldInst);
    here = instList.insert(here,addIn);
#ifdef INSERT_STORE
    here = instList.insert(here,stInst);
#endif
    break;
  }
    
  }
  //now check for cdIn and cdOut
  //first put cdOut
  if(cdIn!=NULL){
    cdIn->getCode(rInst, countInst, M, BB, numPaths, MethNo);
  }
  if(cdOut!=NULL){
    cdOut->getCode(rInst, countInst, M, BB, numPaths, MethNo);
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

  Value *Int0 = ConstantInt::get(Type::IntTy, 0);
  
  //store uint 0, uint *%R, uint 0
  vector<Value *> idx;
  idx.push_back(ConstantUInt::get(Type::UIntTy, 0));

  Instruction *GEP = new GetElementPtrInst(rVar, idx);
  Instruction *stInstr=new StoreInst(Int0, GEP);

  //now push all instructions in front of the BB
  BasicBlock::InstListType& instList=front->getInstList();
  BasicBlock::iterator here=instList.begin();
  here=++front->getInstList().insert(here, rVar);
  here=++front->getInstList().insert(here,countVar);
  
  //Initialize Count[...] with 0

  for(int i=0;i<k; i++){
    Instruction *GEP2 = new GetElementPtrInst(countVar,
                    std::vector<Value *>(1,ConstantUInt::get(Type::UIntTy, i)));
    here=++front->getInstList().insert(here,GEP2);

    Instruction *stInstrC=new StoreInst(Int0, GEP2);
    here=++front->getInstList().insert(here,stInstrC);
  }

  here = ++front->getInstList().insert(here,GEP);
  here = ++front->getInstList().insert(here,stInstr);
}


//insert a basic block with appropriate code
//along a given edge
void insertBB(Edge ed,
	      getEdgeCode *edgeCode, 
	      Instruction *rInst, 
	      Instruction *countInst, 
	      int numPaths, int Methno){
  static int i=-1;
  i++;
  BasicBlock* BB1=ed.getFirst()->getElement();
  BasicBlock* BB2=ed.getSecond()->getElement();
  
#ifdef DEBUG_PATH_PROFILES
  //debugging info
  cerr<<"Edges with codes ######################\n";
  cerr<<BB1->getName()<<"->"<<BB2->getName()<<"\n";
  cerr<<"########################\n";
#endif
  
  char counterstr[100];
  sprintf(counterstr,"counter%d",i);
  std::string ctr=counterstr;

  //We need to insert a BB between BB1 and BB2 
  TerminatorInst *TI=BB1->getTerminator();
  BasicBlock *newBB=new BasicBlock(ctr, BB1->getParent());

  //get code for the new BB
  edgeCode->getCode(rInst, countInst, BB1->getParent(), newBB, numPaths, Methno);
 
  //Is terminator a branch instruction?
  //then we need to change branch destinations to include new BB

  //std::cerr<<"before cast!\n";
  //std::cerr<<"Method no in Edgecode:"<<Methno<<"\n";
  //std::cerr<<"Instruction\n";
  //std::cerr<<*TI;
  BranchInst *BI =  cast<BranchInst>(TI);

  if(BI->isUnconditional()){
    BI->setUnconditionalDest(newBB);
    Instruction *newBI2=new BranchInst(BB2);
    newBB->getInstList().push_back(newBI2);
  }
  else{
      if(BI->getSuccessor(0)==BB2)
      BI->setSuccessor(0, newBB);
    
    if(BI->getSuccessor(1)==BB2)
      BI->setSuccessor(1, newBB);

    Instruction *newBI2=new BranchInst(BB2);
    newBB->getInstList().push_back(newBI2);
  }
  
  //std::cerr<<"After casting\n";
  //get code for the new BB
   //now iterate over BB2, and set its Phi nodes right
  for(BasicBlock::iterator BB2Inst = BB2->begin(), BBend = BB2->end(); 
      BB2Inst != BBend; ++BB2Inst){
   
    if(PHINode *phiInst=dyn_cast<PHINode>(&*BB2Inst)){
      int bbIndex=phiInst->getBasicBlockIndex(BB1);
      assert(bbIndex>=0);
      phiInst->setIncomingBlock(bbIndex, newBB);
    }
  }
}

