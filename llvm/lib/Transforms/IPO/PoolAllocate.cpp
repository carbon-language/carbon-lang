//===-- PoolAllocate.cpp - Pool Allocation Pass ---------------------------===//
//
// This transform changes programs so that disjoint data structures are
// allocated out of different pools of memory, increasing locality.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO.h"
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

namespace {
  const Type *VoidPtrTy = PointerType::get(Type::SByteTy);
  // The type to allocate for a pool descriptor: { sbyte*, uint }
  const Type *PoolDescType =
    StructType::get(make_vector<const Type*>(VoidPtrTy, Type::UIntTy, 0));
  const PointerType *PoolDescPtr = PointerType::get(PoolDescType);


  /// PoolInfo - This struct represents a single pool in the context of a
  /// function.  Pools are mapped one to one with nodes in the DSGraph, so this
  /// contains a pointer to the node it corresponds to.  In addition, the pool
  /// is initialized by calling the "poolinit" library function with a chunk of
  /// memory allocated with an alloca instruction.  This entry contains a
  /// pointer to that alloca if the pool is locally allocated or the argument it
  /// is passed in through if not.
  ///
  struct PoolInfo {
    Value  *PoolHandle;      // Pool Handle, an alloca or incoming argument.
    PoolInfo(Value *PH) : PoolHandle(PH) {}
  };

  struct FuncInfo {
    /// MarkedNodes - The set of nodes which are not locally pool allocatable in
    /// the current function.
    ///
    std::set<DSNode*> MarkedNodes;

    /// Clone - The cloned version of the function, if applicable.
    Function *Clone;

    /// ArgNodes - The list of DSNodes which have pools passed in as arguments.
    ///
    std::vector<DSNode*> ArgNodes;

    /// PoolDescriptors - A PoolInfo object for each relevant DSNode in the
    /// current graph.
    std::map<DSNode*, PoolInfo> PoolDescriptors;

    /// NewToOldValueMap - When and if a function needs to be cloned, this map
    /// contains a mapping from all of the values in the new function back to
    /// the values they correspond to in the old function.
    ///
    std::map<Value*, const Value*> NewToOldValueMap;
  };

  /// PA - The main pool allocation pass
  ///
  class PA : public Pass {
    Module *CurModule;
    BUDataStructures *BU;
    
    std::map<Function*, FuncInfo> FunctionInfo;
  public:
    Function *PoolInit, *PoolDestroy, *PoolAlloc, *PoolFree;
  public:
    bool run(Module &M);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<BUDataStructures>();
      AU.addRequired<TargetData>();
    }

    BUDataStructures &getBUDataStructures() const { return *BU; }

    FuncInfo *getFuncInfo(Function &F) {
      std::map<Function*, FuncInfo>::iterator I = FunctionInfo.find(&F);
      return I != FunctionInfo.end() ? &I->second : 0;
    }

  private:

    /// AddPoolPrototypes - Add prototypes for the pool functions to the
    /// specified module and update the Pool* instance variables to point to
    /// them.
    ///
    void AddPoolPrototypes();

    /// MakeFunctionClone - If the specified function needs to be modified for
    /// pool allocation support, make a clone of it, adding additional arguments
    /// as neccesary, and return it.  If not, just return null.
    ///
    Function *MakeFunctionClone(Function &F);

    /// ProcessFunctionBody - Rewrite the body of a transformed function to use
    /// pool allocation where appropriate.
    ///
    void ProcessFunctionBody(Function &Old, Function &New);
    
    /// CreatePools - This creates the pool initialization and destruction code
    /// for the DSNodes specified by the NodesToPA list.  This adds an entry to
    /// the PoolDescriptors map for each DSNode.
    ///
    void CreatePools(Function &F, const std::vector<DSNode*> &NodesToPA,
                     std::map<DSNode*, PoolInfo> &PoolDescriptors);

    void TransformFunctionBody(Function &F, DSGraph &G, FuncInfo &FI);
  };
  RegisterOpt<PA> X("poolalloc", "Pool allocate disjoint data structures");
}

bool PA::run(Module &M) {
  if (M.begin() == M.end()) return false;
  CurModule = &M;
  
  AddPoolPrototypes();
  BU = &getAnalysis<BUDataStructures>();

  std::map<Function*, Function*> FuncMap;

  // Loop over only the function initially in the program, don't traverse newly
  // added ones.  If the function uses memory, make it's clone.
  Module::iterator LastOrigFunction = --M.end();
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

  FunctionInfo.clear();
  return true;
}


// AddPoolPrototypes - Add prototypes for the pool functions to the specified
// module and update the Pool* instance variables to point to them.
//
void PA::AddPoolPrototypes() {
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

#if 0
  Args[0] = Type::UIntTy;            // Number of slots to allocate
  FunctionType *PoolAllocArrayTy = FunctionType::get(VoidPtrTy, Args, true);
  PoolAllocArray = CurModule->getOrInsertFunction("poolallocarray",
                                                  PoolAllocArrayTy);
#endif
}


// MakeFunctionClone - If the specified function needs to be modified for pool
// allocation support, make a clone of it, adding additional arguments as
// neccesary, and return it.  If not, just return null.
//
Function *PA::MakeFunctionClone(Function &F) {
  DSGraph &G = BU->getDSGraph(F);
  std::vector<DSNode*> &Nodes = G.getNodes();
  if (Nodes.empty()) return 0;  // No memory activity, nothing is required

  FuncInfo &FI = FunctionInfo[&F];   // Create a new entry for F
  FI.Clone = 0;

  // Find DataStructure nodes which are allocated in pools non-local to the
  // current function.  This set will contain all of the DSNodes which require
  // pools to be passed in from outside of the function.
  std::set<DSNode*> &MarkedNodes = FI.MarkedNodes;

  // Mark globals and incomplete nodes as live... (this handles arguments)
  if (F.getName() != "main")
    for (unsigned i = 0, e = Nodes.size(); i != e; ++i)
      if (Nodes[i]->NodeType & (DSNode::GlobalNode | DSNode::Incomplete) &&
          Nodes[i]->NodeType & (DSNode::HeapNode))
        Nodes[i]->markReachableNodes(MarkedNodes);

  // Marked the returned node as alive...
  G.getRetNode().getNode()->markReachableNodes(MarkedNodes);

  if (MarkedNodes.empty())   // We don't need to clone the function if there
    return 0;                // are no incoming arguments to be added.

  // Figure out what the arguments are to be for the new version of the function
  const FunctionType *OldFuncTy = F.getFunctionType();
  std::vector<const Type*> ArgTys;
  ArgTys.reserve(OldFuncTy->getParamTypes().size() + MarkedNodes.size());

  FI.ArgNodes.reserve(MarkedNodes.size());
  for (std::set<DSNode*>::iterator I = MarkedNodes.begin(),
         E = MarkedNodes.end(); I != E; ++I)
    if ((*I)->NodeType & DSNode::Incomplete) {
      ArgTys.push_back(PoolDescPtr);      // Add the appropriate # of pool descs
      FI.ArgNodes.push_back(*I);
    }
  if (FI.ArgNodes.empty()) return 0;      // No nodes to be pool allocated!

  ArgTys.insert(ArgTys.end(), OldFuncTy->getParamTypes().begin(),
                OldFuncTy->getParamTypes().end());


  // Create the new function prototype
  FunctionType *FuncTy = FunctionType::get(OldFuncTy->getReturnType(), ArgTys,
                                           OldFuncTy->isVarArg());
  // Create the new function...
  Function *New = new Function(FuncTy, true, F.getName(), F.getParent());

  // Set the rest of the new arguments names to be PDa<n> and add entries to the
  // pool descriptors map
  std::map<DSNode*, PoolInfo> &PoolDescriptors = FI.PoolDescriptors;
  Function::aiterator NI = New->abegin();
  for (unsigned i = 0, e = FI.ArgNodes.size(); i != e; ++i, ++NI) {
    NI->setName("PDa");  // Add pd entry
    PoolDescriptors.insert(std::make_pair(FI.ArgNodes[i], PoolInfo(NI)));
  }

  // Map the existing arguments of the old function to the corresponding
  // arguments of the new function.
  std::map<const Value*, Value*> ValueMap;
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
void PA::ProcessFunctionBody(Function &F, Function &NewF) {
  DSGraph &G = BU->getDSGraph(F);
  std::vector<DSNode*> &Nodes = G.getNodes();
  if (Nodes.empty()) return;     // Quick exit if nothing to do...

  FuncInfo &FI = FunctionInfo[&F];   // Get FuncInfo for F
  std::set<DSNode*> &MarkedNodes = FI.MarkedNodes;
 
  DEBUG(std::cerr << "[" << F.getName() << "] Pool Allocate: ");

  // Loop over all of the nodes which are non-escaping, adding pool-allocatable
  // ones to the NodesToPA vector.
  std::vector<DSNode*> NodesToPA;
  for (unsigned i = 0, e = Nodes.size(); i != e; ++i)
    if (Nodes[i]->NodeType & DSNode::HeapNode &&   // Pick nodes with heap elems
        !(Nodes[i]->NodeType & DSNode::Array) &&   // Doesn't handle arrays yet.
        !MarkedNodes.count(Nodes[i]))              // Can't be marked
      NodesToPA.push_back(Nodes[i]);

  DEBUG(std::cerr << NodesToPA.size() << " nodes to pool allocate\n");
  if (!NodesToPA.empty()) {
    // Create pool construction/destruction code
    std::map<DSNode*, PoolInfo> &PoolDescriptors = FI.PoolDescriptors;
    CreatePools(NewF, NodesToPA, PoolDescriptors);
  }

  // Transform the body of the function now...
  TransformFunctionBody(NewF, G, FI);
}


// CreatePools - This creates the pool initialization and destruction code for
// the DSNodes specified by the NodesToPA list.  This adds an entry to the
// PoolDescriptors map for each DSNode.
//
void PA::CreatePools(Function &F, const std::vector<DSNode*> &NodesToPA,
                     std::map<DSNode*, PoolInfo> &PoolDescriptors) {
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

    Value *ElSize =
      ConstantUInt::get(Type::UIntTy, TD.getTypeSize(Node->getType()));

    // Insert the call to initialize the pool...
    new CallInst(PoolInit, make_vector(AI, ElSize, 0), "", InsertPoint);

    // Update the PoolDescriptors map
    PoolDescriptors.insert(std::make_pair(Node, PoolInfo(AI)));

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
    PA &PAInfo;
    DSGraph &G;
    FuncInfo &FI;

    FuncTransform(PA &P, DSGraph &g, FuncInfo &fi) : PAInfo(P), G(g), FI(fi) {}

    void visitMallocInst(MallocInst &MI);
    void visitFreeInst(FreeInst &FI);
    void visitCallInst(CallInst &CI);

  private:
    DSNode *getDSNodeFor(Value *V) {
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
      std::map<DSNode*, PoolInfo>::iterator I = FI.PoolDescriptors.find(Node);
      return I != FI.PoolDescriptors.end() ? I->second.PoolHandle : 0;
    }
  };
}

void PA::TransformFunctionBody(Function &F, DSGraph &G, FuncInfo &FI) {
  FuncTransform(*this, G, FI).visit(F);
}


void FuncTransform::visitMallocInst(MallocInst &MI) {
  // Get the pool handle for the node that this contributes to...
  Value *PH = getPoolHandle(&MI);
  if (PH == 0) return;
  
  // Insert a call to poolalloc
  Value *V = new CallInst(PAInfo.PoolAlloc, make_vector(PH, 0),
                          MI.getName(), &MI);
  MI.setName("");  // Nuke MIs name
  
  // Cast to the appropriate type...
  Value *Casted = new CastInst(V, MI.getType(), V->getName(), &MI);
  
  // Update def-use info
  MI.replaceAllUsesWith(Casted);
  
  // Remove old malloc instruction
  MI.getParent()->getInstList().erase(&MI);
  
  std::map<Value*, DSNodeHandle> &SM = G.getScalarMap();
  std::map<Value*, DSNodeHandle>::iterator MII = SM.find(&MI);
  
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
  assert(Caller && "Callee has node but caller doesn't??");

  std::map<DSNode*, DSNode*>::iterator I = NodeMapping.find(Callee);
  if (I != NodeMapping.end()) {   // Node already in map...
    assert(I->second == Caller && "Node maps to different nodes on paths?");
  } else {
    NodeMapping.insert(I, std::make_pair(Callee, Caller));
    
    // Recursively add pointed to nodes...
    for (unsigned i = 0, e = Callee->getNumLinks(); i != e; ++i)
      CalcNodeMapping(Caller->getLink(i << DS::PointerShift).getNode(),
                      Callee->getLink(i << DS::PointerShift).getNode(),
                      NodeMapping);
  }
}

void FuncTransform::visitCallInst(CallInst &CI) {
  Function *CF = CI.getCalledFunction();
  assert(CF && "FIXME: Pool allocation doesn't handle indirect calls!");

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
  for (; AI != AE; ++AI, ++OpNum)
    CalcNodeMapping(getDSNodeFor(CI.getOperand(OpNum)),
                    CG.getScalarMap()[AI].getNode(), NodeMapping);
  assert(OpNum == CI.getNumOperands() && "Varargs calls not handled yet!");
  
  // Map the return value as well...
  CalcNodeMapping(getDSNodeFor(&CI), CG.getRetNode().getNode(), NodeMapping);


  // Okay, now that we have established our mapping, we can figure out which
  // pool descriptors to pass in...
  std::vector<Value*> Args;

  // Add an argument for each pool which must be passed in...
  for (unsigned i = 0, e = CFI->ArgNodes.size(); i != e; ++i) {
    if (NodeMapping.count(CFI->ArgNodes[i])) {
      assert(NodeMapping.count(CFI->ArgNodes[i]) && "Node not in mapping!");
      DSNode *LocalNode = NodeMapping.find(CFI->ArgNodes[i])->second;
      assert(FI.PoolDescriptors.count(LocalNode) && "Node not pool allocated?");
      Args.push_back(FI.PoolDescriptors.find(LocalNode)->second.PoolHandle);
    } else {
      Args.push_back(Constant::getNullValue(PoolDescPtr));
    }
  }

  // Add the rest of the arguments...
  Args.insert(Args.end(), CI.op_begin()+1, CI.op_end());

  std::string Name = CI.getName(); CI.setName("");
  Value *NewCall = new CallInst(CFI->Clone, Args, Name, &CI);
  CI.replaceAllUsesWith(NewCall);

  DEBUG(std::cerr << "  Result Call: " << *NewCall);
  CI.getParent()->getInstList().erase(&CI);
}


// createPoolAllocatePass - Global function to access the functionality of this
// pass...
//
Pass *createPoolAllocatePass() { 
  return new PA(); 
}
