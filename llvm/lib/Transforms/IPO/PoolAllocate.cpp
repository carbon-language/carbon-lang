//===-- PoolAllocate.cpp - Pool Allocation Pass ---------------------------===//
//
// This transform changes programs so that disjoint data structures are
// allocated out of different pools of memory, increasing locality.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/PoolAllocate.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Analysis/DataStructure.h"
#include "llvm/Analysis/DSGraph.h"
#include "llvm/Module.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Support/InstVisitor.h"
#include "Support/Statistic.h"
#include "Support/VectorExtras.h"
using std::vector;
using std::map;
using std::multimap;
using namespace PA;

namespace {
  const Type *VoidPtrTy = PointerType::get(Type::SByteTy);

  // The type to allocate for a pool descriptor: { sbyte*, uint, uint }
  // void *Data (the data)
  // unsigned NodeSize  (size of an allocated node)
  // unsigned FreeablePool (are slabs in the pool freeable upon calls to 
  //                        poolfree?)
  const Type *PoolDescType = 
  StructType::get(make_vector<const Type*>(VoidPtrTy, Type::UIntTy, 
					   Type::UIntTy, 0));
  
  const PointerType *PoolDescPtr = PointerType::get(PoolDescType);
  
  RegisterOpt<PoolAllocate>
  X("poolalloc", "Pool allocate disjoint data structures");
}

void PoolAllocate::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<BUDataStructures>();
  AU.addRequired<TDDataStructures>();
  AU.addRequired<TargetData>();
}

// Prints out the functions mapped to the leader of the equivalence class they
// belong to.
void PoolAllocate::printFuncECs() {
  map<Function*, Function*> &leaderMap = FuncECs.getLeaderMap();
  std::cerr << "Indirect Function Map \n";
  for (map<Function*, Function*>::iterator LI = leaderMap.begin(),
	 LE = leaderMap.end(); LI != LE; ++LI) {
    std::cerr << LI->first->getName() << ": leader is "
	      << LI->second->getName() << "\n";
  }
}

void PoolAllocate::buildIndirectFunctionSets(Module &M) {
  // Iterate over the module looking for indirect calls to functions

  // Get top down DSGraph for the functions
  TDDS = &getAnalysis<TDDataStructures>();
  
  for (Module::iterator MI = M.begin(), ME = M.end(); MI != ME; ++MI) {

    DEBUG(std::cerr << "Processing indirect calls function:" <<  MI->getName() << "\n");

    if (MI->isExternal())
      continue;

    DSGraph &TDG = TDDS->getDSGraph(*MI);

    std::vector<DSCallSite> callSites = TDG.getFunctionCalls();

    // For each call site in the function
    // All the functions that can be called at the call site are put in the
    // same equivalence class.
    for (vector<DSCallSite>::iterator CSI = callSites.begin(), 
	   CSE = callSites.end(); CSI != CSE ; ++CSI) {
      if (CSI->isIndirectCall()) {
	DSNode *DSN = CSI->getCalleeNode();
	if (DSN->isIncomplete())
	  std::cerr << "Incomplete node " << CSI->getCallInst();
	// assert(DSN->isGlobalNode());
	std::vector<GlobalValue*> &Callees = DSN->getGlobals();
	if (Callees.size() > 0) {
	  Function *firstCalledF = dyn_cast<Function>(*Callees.begin());
	  FuncECs.addElement(firstCalledF);
	  CallInstTargets.insert(std::pair<CallInst*,Function*>
				 (&CSI->getCallInst(),
				  firstCalledF));
	  if (Callees.size() > 1) {
	    for (std::vector<GlobalValue*>::iterator CalleesI = 
		   Callees.begin()+1, CalleesE = Callees.end(); 
		 CalleesI != CalleesE; ++CalleesI) {
	      Function *calledF = dyn_cast<Function>(*CalleesI);
	      FuncECs.unionSetsWith(firstCalledF, calledF);
	      CallInstTargets.insert(std::pair<CallInst*,Function*>
				     (&CSI->getCallInst(), calledF));
	    }
	  }
	} else {
	  std::cerr << "No targets " << CSI->getCallInst();
	}
      }
    }
  }
  
  // Print the equivalence classes
  DEBUG(printFuncECs());
}

bool PoolAllocate::run(Module &M) {
  if (M.begin() == M.end()) return false;
  CurModule = &M;
  
  AddPoolPrototypes();
  BU = &getAnalysis<BUDataStructures>();

  buildIndirectFunctionSets(M);

  std::map<Function*, Function*> FuncMap;

   // Loop over the functions in the original program finding the pool desc.
  // arguments necessary for each function that is indirectly callable.
  // For each equivalence class, make a list of pool arguments and update
  // the PoolArgFirst and PoolArgLast values for each function.
  Module::iterator LastOrigFunction = --M.end();
  for (Module::iterator I = M.begin(); ; ++I) {
    if (!I->isExternal())
      FindFunctionPoolArgs(*I);
    if (I == LastOrigFunction) break;
  }

  // Now clone a function using the pool arg list obtained in the previous
  // pass over the modules.
  // Loop over only the function initially in the program, don't traverse newly
  // added ones.  If the function uses memory, make its clone.
  for (Module::iterator I = M.begin(); ; ++I) {
    if (!I->isExternal())
      if (Function *R = MakeFunctionClone(*I))
        FuncMap[I] = R;
    if (I == LastOrigFunction) break;
  }
  
  ++LastOrigFunction;

  // Now that all call targets are available, rewrite the function bodies of the
  // clones.
  for (Module::iterator I = M.begin(); I != LastOrigFunction; ++I)
    if (!I->isExternal()) {
      std::map<Function*, Function*>::iterator FI = FuncMap.find(I);
      ProcessFunctionBody(*I, FI != FuncMap.end() ? *FI->second : *I);
    }

  return true;
}


// AddPoolPrototypes - Add prototypes for the pool functions to the specified
// module and update the Pool* instance variables to point to them.
//
void PoolAllocate::AddPoolPrototypes() {
  CurModule->addTypeName("PoolDescriptor", PoolDescType);
  
  // Get poolinit function...
  FunctionType *PoolInitTy =
    FunctionType::get(Type::VoidTy,
                      make_vector<const Type*>(PoolDescPtr, Type::UIntTy, 0),
                      false);
  PoolInit = CurModule->getOrInsertFunction("poolinit", PoolInitTy);

  // Get pooldestroy function...
  std::vector<const Type*> PDArgs(1, PoolDescPtr);
  FunctionType *PoolDestroyTy =
    FunctionType::get(Type::VoidTy, PDArgs, false);
  PoolDestroy = CurModule->getOrInsertFunction("pooldestroy", PoolDestroyTy);
  
  // Get the poolalloc function...
  FunctionType *PoolAllocTy = FunctionType::get(VoidPtrTy, PDArgs, false);
  PoolAlloc = CurModule->getOrInsertFunction("poolalloc", PoolAllocTy);
  
  // Get the poolfree function...
  PDArgs.push_back(VoidPtrTy);       // Pointer to free
  FunctionType *PoolFreeTy = FunctionType::get(Type::VoidTy, PDArgs, false);
  PoolFree = CurModule->getOrInsertFunction("poolfree", PoolFreeTy);
  
  // The poolallocarray function
  FunctionType *PoolAllocArrayTy =
    FunctionType::get(VoidPtrTy,
                      make_vector<const Type*>(PoolDescPtr, Type::UIntTy, 0),
                      false);
  PoolAllocArray = CurModule->getOrInsertFunction("poolallocarray", 
						  PoolAllocArrayTy);
  
}

void PoolAllocate::FindFunctionPoolArgs(Function &F) {
  DSGraph &G = BU->getDSGraph(F);
  std::vector<DSNode*> &Nodes = G.getNodes();
  if (Nodes.empty()) return ;  // No memory activity, nothing is required

  FuncInfo &FI = FunctionInfo[&F];   // Create a new entry for F
  
  FI.Clone = 0;
  
  // Initialize the PoolArgFirst and PoolArgLast for the function depending
  // on whether there have been other functions in the equivalence class
  // that have pool arguments so far in the analysis.
  if (!FuncECs.findClass(&F)) {
    FI.PoolArgFirst = FI.PoolArgLast = 0;
  } else {
    if (EqClass2LastPoolArg.find(FuncECs.findClass(&F)) != 
	EqClass2LastPoolArg.end())
      FI.PoolArgFirst = FI.PoolArgLast = 
	EqClass2LastPoolArg[FuncECs.findClass(&F)] + 1;
    else
      FI.PoolArgFirst = FI.PoolArgLast = 0;
  }
  
  // Find DataStructure nodes which are allocated in pools non-local to the
  // current function.  This set will contain all of the DSNodes which require
  // pools to be passed in from outside of the function.
  hash_set<DSNode*> &MarkedNodes = FI.MarkedNodes;
  
  // Mark globals and incomplete nodes as live... (this handles arguments)
  if (F.getName() != "main")
    for (unsigned i = 0, e = Nodes.size(); i != e; ++i)
      if ((Nodes[i]->isGlobalNode() || Nodes[i]->isIncomplete()) &&
          Nodes[i]->isHeapNode())
        Nodes[i]->markReachableNodes(MarkedNodes);

  // Marked the returned node as alive...
  if (DSNode *RetNode = G.getRetNode().getNode())
    if (RetNode->isHeapNode())
      RetNode->markReachableNodes(MarkedNodes);

  if (MarkedNodes.empty())   // We don't need to clone the function if there
    return;                  // are no incoming arguments to be added.

  for (hash_set<DSNode*>::iterator I = MarkedNodes.begin(),
	 E = MarkedNodes.end(); I != E; ++I)
    FI.PoolArgLast++;

  if (FuncECs.findClass(&F)) {
    // Update the equivalence class last pool argument information
    // only if there actually were pool arguments to the function.
    // Also, there is no entry for the Eq. class in EqClass2LastPoolArg
    // if there are no functions in the equivalence class with pool arguments.
    if (FI.PoolArgLast != FI.PoolArgFirst)
      EqClass2LastPoolArg[FuncECs.findClass(&F)] = FI.PoolArgLast - 1;
  }
  
}

// MakeFunctionClone - If the specified function needs to be modified for pool
// allocation support, make a clone of it, adding additional arguments as
// neccesary, and return it.  If not, just return null.
//
Function *PoolAllocate::MakeFunctionClone(Function &F) {
  
  DSGraph &G = BU->getDSGraph(F);
  std::vector<DSNode*> &Nodes = G.getNodes();
  if (Nodes.empty())
    return 0;
    
  FuncInfo &FI = FunctionInfo[&F];
  
  hash_set<DSNode*> &MarkedNodes = FI.MarkedNodes;
  
  if (!FuncECs.findClass(&F)) {
    // Not in any equivalence class

    if (MarkedNodes.empty())
      return 0;
  } else {
    // No need to clone if there are no pool arguments in any function in the
    // equivalence class
    if (!EqClass2LastPoolArg.count(FuncECs.findClass(&F)))
      return 0;
  }
      
  // Figure out what the arguments are to be for the new version of the function
  const FunctionType *OldFuncTy = F.getFunctionType();
  std::vector<const Type*> ArgTys;
  if (!FuncECs.findClass(&F)) {
    ArgTys.reserve(OldFuncTy->getParamTypes().size() + MarkedNodes.size());
    FI.ArgNodes.reserve(MarkedNodes.size());
    for (hash_set<DSNode*>::iterator I = MarkedNodes.begin(),
	   E = MarkedNodes.end(); I != E; ++I) {
      ArgTys.push_back(PoolDescPtr);      // Add the appropriate # of pool descs
      FI.ArgNodes.push_back(*I);
    }
    if (FI.ArgNodes.empty()) return 0;      // No nodes to be pool allocated!

  }
  else {
    // This function is a member of an equivalence class and needs to be cloned 
    ArgTys.reserve(OldFuncTy->getParamTypes().size() + 
		   EqClass2LastPoolArg[FuncECs.findClass(&F)] + 1);
    FI.ArgNodes.reserve(EqClass2LastPoolArg[FuncECs.findClass(&F)] + 1);
    
    for (int i = 0; i <= EqClass2LastPoolArg[FuncECs.findClass(&F)]; ++i) {
      ArgTys.push_back(PoolDescPtr);      // Add the appropriate # of pool 
                                          // descs
    }

    for (hash_set<DSNode*>::iterator I = MarkedNodes.begin(),
	   E = MarkedNodes.end(); I != E; ++I) {
      FI.ArgNodes.push_back(*I);
    }

    assert ((FI.ArgNodes.size() == (unsigned) (FI.PoolArgLast - 
					       FI.PoolArgFirst)) && 
	    "Number of ArgNodes equal to the number of pool arguments used by this function");
  }
      
      
  ArgTys.insert(ArgTys.end(), OldFuncTy->getParamTypes().begin(),
                OldFuncTy->getParamTypes().end());


  // Create the new function prototype
  FunctionType *FuncTy = FunctionType::get(OldFuncTy->getReturnType(), ArgTys,
                                           OldFuncTy->isVarArg());
  // Create the new function...
  Function *New = new Function(FuncTy, GlobalValue::InternalLinkage,
                               F.getName(), F.getParent());

  // Set the rest of the new arguments names to be PDa<n> and add entries to the
  // pool descriptors map
  std::map<DSNode*, Value*> &PoolDescriptors = FI.PoolDescriptors;
  Function::aiterator NI = New->abegin();
  
  if (FuncECs.findClass(&F)) {
    for (int i = 0; i <= EqClass2LastPoolArg[FuncECs.findClass(&F)]; ++i, 
	   ++NI)
      NI->setName("PDa");
    
    NI = New->abegin();
    if (FI.PoolArgFirst > 0)
      for (int i = 0; i < FI.PoolArgFirst; ++NI, ++i)
	;

    if (FI.ArgNodes.size() > 0)
      for (unsigned i = 0, e = FI.ArgNodes.size(); i != e; ++i, ++NI)
	PoolDescriptors.insert(std::make_pair(FI.ArgNodes[i], NI));

    NI = New->abegin();
    if (EqClass2LastPoolArg.count(FuncECs.findClass(&F)))
      for (int i = 0; i <= EqClass2LastPoolArg[FuncECs.findClass(&F)]; ++i, ++NI)
	;
  } else {
    if (FI.ArgNodes.size())
      for (unsigned i = 0, e = FI.ArgNodes.size(); i != e; ++i, ++NI) {
	NI->setName("PDa");  // Add pd entry
	PoolDescriptors.insert(std::make_pair(FI.ArgNodes[i], NI));
      }
    NI = New->abegin();
    if (FI.ArgNodes.size())
      for (unsigned i = 0; i < FI.ArgNodes.size(); ++NI, ++i)
	;
  }

  // Map the existing arguments of the old function to the corresponding
  // arguments of the new function.
  std::map<const Value*, Value*> ValueMap;
  if (NI != New->aend()) 
    for (Function::aiterator I = F.abegin(), E = F.aend(); I != E; ++I, ++NI) {
      ValueMap[I] = NI;
      NI->setName(I->getName());
    }

  // Populate the value map with all of the globals in the program.
  // FIXME: This should be unneccesary!
  Module &M = *F.getParent();
  for (Module::iterator I = M.begin(), E=M.end(); I!=E; ++I)    ValueMap[I] = I;
  for (Module::giterator I = M.gbegin(), E=M.gend(); I!=E; ++I) ValueMap[I] = I;

  // Perform the cloning.
  std::vector<ReturnInst*> Returns;
  CloneFunctionInto(New, &F, ValueMap, Returns);

  // Invert the ValueMap into the NewToOldValueMap
  std::map<Value*, const Value*> &NewToOldValueMap = FI.NewToOldValueMap;
  for (std::map<const Value*, Value*>::iterator I = ValueMap.begin(),
         E = ValueMap.end(); I != E; ++I)
    NewToOldValueMap.insert(std::make_pair(I->second, I->first));
  
  return FI.Clone = New;
}


// processFunction - Pool allocate any data structures which are contained in
// the specified function...
//
void PoolAllocate::ProcessFunctionBody(Function &F, Function &NewF) {
  DSGraph &G = BU->getDSGraph(F);
  std::vector<DSNode*> &Nodes = G.getNodes();
  if (Nodes.empty()) return;     // Quick exit if nothing to do...
  
  FuncInfo &FI = FunctionInfo[&F];   // Get FuncInfo for F
  hash_set<DSNode*> &MarkedNodes = FI.MarkedNodes;
  
  DEBUG(std::cerr << "[" << F.getName() << "] Pool Allocate: ");
  
  // Loop over all of the nodes which are non-escaping, adding pool-allocatable
  // ones to the NodesToPA vector.
  std::vector<DSNode*> NodesToPA;
  for (unsigned i = 0, e = Nodes.size(); i != e; ++i)
    if (Nodes[i]->isHeapNode() &&   // Pick nodes with heap elems
        !MarkedNodes.count(Nodes[i]))              // Can't be marked
      NodesToPA.push_back(Nodes[i]);
  
  DEBUG(std::cerr << NodesToPA.size() << " nodes to pool allocate\n");
  if (!NodesToPA.empty()) {
    // Create pool construction/destruction code
    std::map<DSNode*, Value*> &PoolDescriptors = FI.PoolDescriptors;
    CreatePools(NewF, NodesToPA, PoolDescriptors);
  }
  
  // Transform the body of the function now...
  TransformFunctionBody(NewF, G, FI);
}


// CreatePools - This creates the pool initialization and destruction code for
// the DSNodes specified by the NodesToPA list.  This adds an entry to the
// PoolDescriptors map for each DSNode.
//
void PoolAllocate::CreatePools(Function &F,
                               const std::vector<DSNode*> &NodesToPA,
                               std::map<DSNode*, Value*> &PoolDescriptors) {
  // Find all of the return nodes in the CFG...
  std::vector<BasicBlock*> ReturnNodes;
  for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I)
    if (isa<ReturnInst>(I->getTerminator()))
      ReturnNodes.push_back(I);

  TargetData &TD = getAnalysis<TargetData>();

  // Loop over all of the pools, inserting code into the entry block of the
  // function for the initialization and code in the exit blocks for
  // destruction.
  //
  Instruction *InsertPoint = F.front().begin();
  for (unsigned i = 0, e = NodesToPA.size(); i != e; ++i) {
    DSNode *Node = NodesToPA[i];
    
    // Create a new alloca instruction for the pool...
    Value *AI = new AllocaInst(PoolDescType, 0, "PD", InsertPoint);
    
    Value *ElSize;
    
    // Void types in DS graph are never used
    if (Node->getType() != Type::VoidTy)
      ElSize = ConstantUInt::get(Type::UIntTy, TD.getTypeSize(Node->getType()));
    else
      ElSize = ConstantUInt::get(Type::UIntTy, 0);
	
    // Insert the call to initialize the pool...
    new CallInst(PoolInit, make_vector(AI, ElSize, 0), "", InsertPoint);
      
    // Update the PoolDescriptors map
    PoolDescriptors.insert(std::make_pair(Node, AI));
    
    // Insert a call to pool destroy before each return inst in the function
    for (unsigned r = 0, e = ReturnNodes.size(); r != e; ++r)
      new CallInst(PoolDestroy, make_vector(AI, 0), "",
		   ReturnNodes[r]->getTerminator());
  }
}


namespace {
  /// FuncTransform - This class implements transformation required of pool
  /// allocated functions.
  struct FuncTransform : public InstVisitor<FuncTransform> {
    PoolAllocate &PAInfo;
    DSGraph &G;
    DSGraph &TDG;
    FuncInfo &FI;

    FuncTransform(PoolAllocate &P, DSGraph &g, DSGraph &tdg, FuncInfo &fi)
      : PAInfo(P), G(g), TDG(tdg), FI(fi) {
    }

    void visitMallocInst(MallocInst &MI);
    void visitFreeInst(FreeInst &FI);
    void visitCallInst(CallInst &CI);
    
    // The following instructions are never modified by pool allocation
    void visitBranchInst(BranchInst &I) { }
    void visitBinaryOperator(Instruction &I) { }
    void visitShiftInst (ShiftInst &I) { }
    void visitSwitchInst (SwitchInst &I) { }
    void visitCastInst (CastInst &I) { }
    void visitAllocaInst(AllocaInst &I) { }
    void visitLoadInst(LoadInst &I) { }
    void visitGetElementPtrInst (GetElementPtrInst &I) { }

    void visitReturnInst(ReturnInst &I);
    void visitStoreInst (StoreInst &I);
    void visitPHINode(PHINode &I);

    void visitInstruction(Instruction &I) {
      std::cerr << "PoolAllocate does not recognize this instruction\n";
      abort();
    }

  private:
    DSNode *getDSNodeFor(Value *V) {
      if (isa<Constant>(V))
	return 0;

      if (!FI.NewToOldValueMap.empty()) {
        // If the NewToOldValueMap is in effect, use it.
        std::map<Value*,const Value*>::iterator I = FI.NewToOldValueMap.find(V);
        if (I != FI.NewToOldValueMap.end())
          V = (Value*)I->second;
      }

      return G.getScalarMap()[V].getNode();
    }
    Value *getPoolHandle(Value *V) {
      DSNode *Node = getDSNodeFor(V);
      // Get the pool handle for this DSNode...
      std::map<DSNode*, Value*>::iterator I = FI.PoolDescriptors.find(Node);
      return I != FI.PoolDescriptors.end() ? I->second : 0;
    }
    
    bool isFuncPtr(Value *V);

    Function* getFuncClass(Value *V);

    Value* retCloneIfFunc(Value *V);
  };
}

void PoolAllocate::TransformFunctionBody(Function &F, DSGraph &G, FuncInfo &FI){
  DSGraph &TDG = TDDS->getDSGraph(G.getFunction());
  FuncTransform(*this, G, TDG, FI).visit(F);
}

// Returns true if V is a function pointer
bool FuncTransform::isFuncPtr(Value *V) {
  if (const PointerType *PTy = dyn_cast<PointerType>(V->getType()))
     return isa<FunctionType>(PTy->getElementType());
  return false;
}

// Given a function pointer, return the function eq. class if one exists
Function* FuncTransform::getFuncClass(Value *V) {
  // Look at DSGraph and see if the set of of functions it could point to
  // are pool allocated.

  if (!isFuncPtr(V))
    return 0;

  // Two cases: 
  // if V is a constant
  if (Function *theFunc = dyn_cast<Function>(V)) {
    if (!PAInfo.FuncECs.findClass(theFunc))
      // If this function does not belong to any equivalence class
      return 0;
    if (PAInfo.EqClass2LastPoolArg.count(PAInfo.FuncECs.findClass(theFunc)))
      return PAInfo.FuncECs.findClass(theFunc);
    else
      return 0;
  }

  // if V is not a constant
  DSNode *DSN = TDG.getNodeForValue(V).getNode();
  if (!DSN) {
    return 0;
  }
  std::vector<GlobalValue*> &Callees = DSN->getGlobals();
  if (Callees.size() > 0) {
    Function *calledF = dyn_cast<Function>(*Callees.begin());
    assert(PAInfo.FuncECs.findClass(calledF) && "should exist in some eq. class");
    if (PAInfo.EqClass2LastPoolArg.count(PAInfo.FuncECs.findClass(calledF)))
      return PAInfo.FuncECs.findClass(calledF);
  }

  return 0;
}

// Returns the clone if  V is a static function (not a pointer) and belongs 
// to an equivalence class i.e. is pool allocated
Value* FuncTransform::retCloneIfFunc(Value *V) {
  if (Function *fixedFunc = dyn_cast<Function>(V))
    if (getFuncClass(V))
      return PAInfo.getFuncInfo(*fixedFunc)->Clone;
  
  return 0;
}

void FuncTransform::visitReturnInst (ReturnInst &I) {
  if (I.getNumOperands())
    if (Value *clonedFunc = retCloneIfFunc(I.getOperand(0))) {
      // Cast the clone of I.getOperand(0) to the non-pool-allocated type
      CastInst *CastI = new CastInst(clonedFunc, I.getOperand(0)->getType(), 
				     "tmp", &I);
      // Insert return instruction that returns the casted value
      new ReturnInst(CastI, &I);

      // Remove original return instruction
      I.getParent()->getInstList().erase(&I);
    }
}

void FuncTransform::visitStoreInst (StoreInst &I) {
  // Check if a constant function is being stored
  if (Value *clonedFunc = retCloneIfFunc(I.getOperand(0))) {
    CastInst *CastI = new CastInst(clonedFunc, I.getOperand(0)->getType(), 
				   "tmp", &I);
    new StoreInst(CastI, I.getOperand(1), &I);
    I.getParent()->getInstList().erase(&I);
  }
}

void FuncTransform::visitPHINode(PHINode &I) {
  // If any of the operands of the PHI node is a constant function pointer
  // that is cloned, the cast instruction has to be inserted at the end of the
  // previous basic block
  
  if (isFuncPtr(&I)) {
    PHINode *V = new PHINode(I.getType(), I.getName(), &I);
    for (unsigned i = 0 ; i < I.getNumIncomingValues(); ++i) {
      if (Value *clonedFunc = retCloneIfFunc(I.getIncomingValue(i))) {
	// Insert CastInst at the end of  I.getIncomingBlock(i)
	BasicBlock::iterator BBI = --I.getIncomingBlock(i)->end();
	// BBI now points to the terminator instruction of the basic block.
	CastInst *CastI = new CastInst(clonedFunc, I.getType(), "tmp", BBI);
	V->addIncoming(CastI, I.getIncomingBlock(i));
      } else {
	V->addIncoming(I.getIncomingValue(i), I.getIncomingBlock(i));
      }
      
    }
    I.replaceAllUsesWith(V);
    I.getParent()->getInstList().erase(&I);
  }
}

void FuncTransform::visitMallocInst(MallocInst &MI) {
  // Get the pool handle for the node that this contributes to...
  Value *PH = getPoolHandle(&MI);
  if (PH == 0) return;
  
  // Insert a call to poolalloc
  Value *V;
  if (MI.isArrayAllocation()) 
    V = new CallInst(PAInfo.PoolAllocArray, 
		     make_vector(PH, MI.getOperand(0), 0),
		     MI.getName(), &MI);
  else
    V = new CallInst(PAInfo.PoolAlloc, make_vector(PH, 0),
		     MI.getName(), &MI);
  
  MI.setName("");  // Nuke MIs name
  
  // Cast to the appropriate type...
  Value *Casted = new CastInst(V, MI.getType(), V->getName(), &MI);
  
  // Update def-use info
  MI.replaceAllUsesWith(Casted);
  
  // Remove old malloc instruction
  MI.getParent()->getInstList().erase(&MI);
  
  hash_map<Value*, DSNodeHandle> &SM = G.getScalarMap();
  hash_map<Value*, DSNodeHandle>::iterator MII = SM.find(&MI);
  
  // If we are modifying the original function, update the DSGraph... 
  if (MII != SM.end()) {
    // V and Casted now point to whatever the original malloc did...
    SM.insert(std::make_pair(V, MII->second));
    SM.insert(std::make_pair(Casted, MII->second));
    SM.erase(MII);                     // The malloc is now destroyed
  } else {             // Otherwise, update the NewToOldValueMap
    std::map<Value*,const Value*>::iterator MII =
      FI.NewToOldValueMap.find(&MI);
    assert(MII != FI.NewToOldValueMap.end() && "MI not found in clone?");
    FI.NewToOldValueMap.insert(std::make_pair(V, MII->second));
    FI.NewToOldValueMap.insert(std::make_pair(Casted, MII->second));
    FI.NewToOldValueMap.erase(MII);
  }
}

void FuncTransform::visitFreeInst(FreeInst &FI) {
  Value *Arg = FI.getOperand(0);
  Value *PH = getPoolHandle(Arg);  // Get the pool handle for this DSNode...
  if (PH == 0) return;
  // Insert a cast and a call to poolfree...
  Value *Casted = new CastInst(Arg, PointerType::get(Type::SByteTy),
                               Arg->getName()+".casted", &FI);
  new CallInst(PAInfo.PoolFree, make_vector(PH, Casted, 0), "", &FI);
  
  // Delete the now obsolete free instruction...
  FI.getParent()->getInstList().erase(&FI);
}

static void CalcNodeMapping(DSNode *Caller, DSNode *Callee,
                            std::map<DSNode*, DSNode*> &NodeMapping) {
  if (Callee == 0) return;
  // assert(Caller && "Callee has node but caller doesn't??");

  // If callee has a node and caller doesn't, then a constant argument was
  // passed by the caller
  if (Caller == 0) {
    NodeMapping.insert(NodeMapping.end(), std::make_pair(Callee, (DSNode*) 0));
  }

  std::map<DSNode*, DSNode*>::iterator I = NodeMapping.find(Callee);
  if (I != NodeMapping.end()) {   // Node already in map...
    assert(I->second == Caller && "Node maps to different nodes on paths?");
  } else {
    NodeMapping.insert(I, std::make_pair(Callee, Caller));
    
    // Recursively add pointed to nodes...
    unsigned numCallerLinks = Caller->getNumLinks();
    unsigned numCalleeLinks = Callee->getNumLinks();

    assert (numCallerLinks <= numCalleeLinks || numCalleeLinks == 0);
    
    for (unsigned i = 0, e = numCalleeLinks; i != e; ++i)
      CalcNodeMapping(Caller->getLink((i%numCallerLinks) << DS::PointerShift).getNode(), Callee->getLink(i << DS::PointerShift).getNode(), NodeMapping);
  }
}

void FuncTransform::visitCallInst(CallInst &CI) {
  Function *CF = CI.getCalledFunction();
  
  // optimization for function pointers that are basically gotten from a cast
  // with only one use and constant expressions with casts in them
  if (!CF) {
    if (CastInst* CastI = dyn_cast<CastInst>(CI.getCalledValue())) {
      if (isa<Function>(CastI->getOperand(0)) && 
	  CastI->getOperand(0)->getType() == CastI->getType())
	CF = dyn_cast<Function>(CastI->getOperand(0));
    } else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(CI.getOperand(0))) {
      if (CE->getOpcode() == Instruction::Cast) {
	if (isa<ConstantPointerRef>(CE->getOperand(0)))
	  return;
	else
	  assert(0 && "Function pointer cast not handled as called function\n");
      }
    }    

  }

  std::vector<Value*> Args;  
  if (!CF) {
    // Indirect call
    
    DEBUG(std::cerr << "  Handling call: " << CI);
    
    std::map<unsigned, Value*> PoolArgs;
    Function *FuncClass;
    
    std::pair<multimap<CallInst*, Function*>::iterator, multimap<CallInst*, Function*>::iterator> Targets = PAInfo.CallInstTargets.equal_range(&CI);
    for (multimap<CallInst*, Function*>::iterator TFI = Targets.first,
	   TFE = Targets.second; TFI != TFE; ++TFI) {
      if (TFI == Targets.first) {
	FuncClass = PAInfo.FuncECs.findClass(TFI->second);
	// Nothing to transform if there are no pool arguments in this
	// equivalence class of functions.
	if (!PAInfo.EqClass2LastPoolArg.count(FuncClass))
	  return;
      }
      
      FuncInfo *CFI = PAInfo.getFuncInfo(*TFI->second);

      if (!CFI->ArgNodes.size()) continue;  // Nothing to transform...
      
      DSGraph &CG = PAInfo.getBUDataStructures().getDSGraph(*TFI->second);  
      std::map<DSNode*, DSNode*> NodeMapping;
      
      Function::aiterator AI = TFI->second->abegin(), AE = TFI->second->aend();
      unsigned OpNum = 1;
      for ( ; AI != AE; ++AI, ++OpNum) {
	if (!isa<Constant>(CI.getOperand(OpNum)))
	  CalcNodeMapping(getDSNodeFor(CI.getOperand(OpNum)), 
			  CG.getScalarMap()[AI].getNode(),
			  NodeMapping);
      }
      assert(OpNum == CI.getNumOperands() && "Varargs calls not handled yet!");
      
      if (CI.getType() != Type::VoidTy)
	CalcNodeMapping(getDSNodeFor(&CI), CG.getRetNode().getNode(), 
			NodeMapping);
      
      unsigned idx = CFI->PoolArgFirst;

      // The following loop determines the pool pointers corresponding to 
      // CFI.
      for (unsigned i = 0, e = CFI->ArgNodes.size(); i != e; ++i, ++idx) {
	if (NodeMapping.count(CFI->ArgNodes[i])) {
	  assert(NodeMapping.count(CFI->ArgNodes[i]) && "Node not in mapping!");
	  DSNode *LocalNode = NodeMapping.find(CFI->ArgNodes[i])->second;
	  if (LocalNode) {
	    assert(FI.PoolDescriptors.count(LocalNode) && "Node not pool allocated?");
	    PoolArgs[idx] = FI.PoolDescriptors.find(LocalNode)->second;
	  }
	  else
	    // LocalNode is null when a constant is passed in as a parameter
	    PoolArgs[idx] = Constant::getNullValue(PoolDescPtr);
	} else {
	  PoolArgs[idx] = Constant::getNullValue(PoolDescPtr);
	}
      }
    }
    
    // Push the pool arguments into Args.
    if (PAInfo.EqClass2LastPoolArg.count(FuncClass)) {
      for (int i = 0; i <= PAInfo.EqClass2LastPoolArg[FuncClass]; ++i) {
	if (PoolArgs.find(i) != PoolArgs.end())
	  Args.push_back(PoolArgs[i]);
	else
	  Args.push_back(Constant::getNullValue(PoolDescPtr));
      }
    
      assert (Args.size()== (unsigned) PAInfo.EqClass2LastPoolArg[FuncClass] + 1 
	      && "Call has same number of pool args as the called function");
    }

    // Add the rest of the arguments (the original arguments of the function)...
    Args.insert(Args.end(), CI.op_begin()+1, CI.op_end());
    
    std::string Name = CI.getName();
    
    Value *NewCall;
    if (Args.size() > CI.getNumOperands() - 1) {
      // If there are any pool arguments
      CastInst *CastI = 
	new CastInst(CI.getOperand(0), 
		     PAInfo.getFuncInfo(*FuncClass)->Clone->getType(), "tmp", 
		     &CI);
      NewCall = new CallInst(CastI, Args, Name, &CI);
    } else {
      NewCall = new CallInst(CI.getOperand(0), Args, Name, &CI);
    }

    CI.replaceAllUsesWith(NewCall);
    DEBUG(std::cerr << "  Result Call: " << *NewCall);
  }
  else {

    FuncInfo *CFI = PAInfo.getFuncInfo(*CF);

    if (CFI == 0 || CFI->Clone == 0) return;  // Nothing to transform...
    
    DEBUG(std::cerr << "  Handling call: " << CI);
    
    DSGraph &CG = PAInfo.getBUDataStructures().getDSGraph(*CF);  // Callee graph
    
    // We need to figure out which local pool descriptors correspond to the pool
    // descriptor arguments passed into the function call.  Calculate a mapping
    // from callee DSNodes to caller DSNodes.  We construct a partial isomophism
    // between the graphs to figure out which pool descriptors need to be passed
    // in.  The roots of this mapping is found from arguments and return values.
    //
    std::map<DSNode*, DSNode*> NodeMapping;
    
    Function::aiterator AI = CF->abegin(), AE = CF->aend();
    unsigned OpNum = 1;
    for (; AI != AE; ++AI, ++OpNum) {
      // Check if the operand of the call is a return of another call
      // for the operand will be transformed in which case.
      // Look up the OldToNewRetValMap to see if the operand is a new value.
      Value *callOp = CI.getOperand(OpNum);
      if (!isa<Constant>(callOp))
	CalcNodeMapping(getDSNodeFor(callOp),CG.getScalarMap()[AI].getNode(), 
			NodeMapping);
    }
    assert(OpNum == CI.getNumOperands() && "Varargs calls not handled yet!");
    
    // Map the return value as well...
    if (CI.getType() != Type::VoidTy)
      CalcNodeMapping(getDSNodeFor(&CI), CG.getRetNode().getNode(), 
		      NodeMapping);

    // Okay, now that we have established our mapping, we can figure out which
    // pool descriptors to pass in...

    // Add an argument for each pool which must be passed in...
    if (CFI->PoolArgFirst != 0) {
      for (int i = 0; i < CFI->PoolArgFirst; ++i)
	Args.push_back(Constant::getNullValue(PoolDescPtr));  
    }

    for (unsigned i = 0, e = CFI->ArgNodes.size(); i != e; ++i) {
      if (NodeMapping.count(CFI->ArgNodes[i])) {
	assert(NodeMapping.count(CFI->ArgNodes[i]) && "Node not in mapping!");
	DSNode *LocalNode = NodeMapping.find(CFI->ArgNodes[i])->second;
	if (LocalNode) {
	  assert(FI.PoolDescriptors.count(LocalNode) && "Node not pool allocated?");
	  Args.push_back(FI.PoolDescriptors.find(LocalNode)->second);
	}
	else
	  Args.push_back(Constant::getNullValue(PoolDescPtr));
      } else {
	Args.push_back(Constant::getNullValue(PoolDescPtr));
      }
    }

    Function *FuncClass = PAInfo.FuncECs.findClass(CF);
    
    if (PAInfo.EqClass2LastPoolArg.count(FuncClass))
      for (unsigned i = CFI->PoolArgLast; 
	   i <= PAInfo.EqClass2LastPoolArg.count(FuncClass); ++i)
	Args.push_back(Constant::getNullValue(PoolDescPtr));

    // Add the rest of the arguments...
    Args.insert(Args.end(), CI.op_begin()+1, CI.op_end());
    
    std::string Name = CI.getName(); 
    Value *NewCall = new CallInst(CFI->Clone, Args, Name, &CI);
    CI.replaceAllUsesWith(NewCall);
    DEBUG(std::cerr << "  Result Call: " << *NewCall);

  }
  
  CI.getParent()->getInstList().erase(&CI);
}
