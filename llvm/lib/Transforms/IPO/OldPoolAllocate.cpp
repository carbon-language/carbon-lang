//===-- PoolAllocate.cpp - Pool Allocation Pass ---------------------------===//
//
// This transform changes programs so that disjoint data structures are
// allocated out of different pools of memory, increasing locality and shrinking
// pointer size.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/PoolAllocate.h"
#include "llvm/Transforms/CloneFunction.h"
#include "llvm/Analysis/DataStructure.h"
#include "llvm/Analysis/DataStructureGraph.h"
#include "llvm/Pass.h"
#include "llvm/Module.h"
#include "llvm/Function.h"
#include "llvm/BasicBlock.h"
#include "llvm/iMemory.h"
#include "llvm/iTerminators.h"
#include "llvm/iOther.h"
#include "llvm/ConstantVals.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Support/InstVisitor.h"
#include "llvm/Argument.h"
#include "Support/DepthFirstIterator.h"
#include "Support/STLExtras.h"
#include <algorithm>


// FIXME: This is dependant on the sparc backend layout conventions!!
static TargetData TargetData("test");

namespace {
  // ScalarInfo - Information about an LLVM value that we know points to some
  // datastructure we are processing.
  //
  struct ScalarInfo {
    Value  *Val;            // Scalar value in Current Function
    DSNode *Node;           // DataStructure node it points to
    Value  *PoolHandle;     // PoolTy* LLVM value
    
    ScalarInfo(Value *V, DSNode *N, Value *PH)
      : Val(V), Node(N), PoolHandle(PH) {
      assert(V && N && PH && "Null value passed to ScalarInfo ctor!");
    }
  };

  // CallArgInfo - Information on one operand for a call that got expanded.
  struct CallArgInfo {
    int ArgNo;          // Call argument number this corresponds to
    DSNode *Node;       // The graph node for the pool
    Value *PoolHandle;  // The LLVM value that is the pool pointer

    CallArgInfo(int Arg, DSNode *N, Value *PH)
      : ArgNo(Arg), Node(N), PoolHandle(PH) {
      assert(Arg >= -1 && N && PH && "Illegal values to CallArgInfo ctor!");
    }

    // operator< when sorting, sort by argument number.
    bool operator<(const CallArgInfo &CAI) const {
      return ArgNo < CAI.ArgNo;
    }
  };

  // TransformFunctionInfo - Information about how a function eeds to be
  // transformed.
  //
  struct TransformFunctionInfo {
    // ArgInfo - Maintain information about the arguments that need to be
    // processed.  Each pair corresponds to an argument (whose number is the
    // first element) that needs to have a pool pointer (the second element)
    // passed into the transformed function with it.
    //
    // As a special case, "argument" number -1 corresponds to the return value.
    //
    vector<CallArgInfo> ArgInfo;

    // Func - The function to be transformed...
    Function *Func;

    // The call instruction that is used to map CallArgInfo PoolHandle values
    // into the new function values.
    CallInst *Call;

    // default ctor...
    TransformFunctionInfo() : Func(0), Call(0) {}
    
    bool operator<(const TransformFunctionInfo &TFI) const {
      if (Func < TFI.Func) return true;
      if (Func > TFI.Func) return false;
      if (ArgInfo.size() < TFI.ArgInfo.size()) return true;
      if (ArgInfo.size() > TFI.ArgInfo.size()) return false;
      return ArgInfo < TFI.ArgInfo;
    }

    void finalizeConstruction() {
      // Sort the vector so that the return value is first, followed by the
      // argument records, in order.  Note that this must be a stable sort so
      // that the entries with the same sorting criteria (ie they are multiple
      // pool entries for the same argument) are kept in depth first order.
      stable_sort(ArgInfo.begin(), ArgInfo.end());
    }
  };


  // Define the pass class that we implement...
  class PoolAllocate : public Pass {
    // PoolTy - The type of a scalar value that contains a pool pointer.
    PointerType *PoolTy;
  public:

    PoolAllocate() {
      // Initialize the PoolTy instance variable, since the type never changes.
      vector<const Type*> PoolElements;
      PoolElements.push_back(PointerType::get(Type::SByteTy));
      PoolElements.push_back(Type::UIntTy);
      PoolTy = PointerType::get(StructType::get(PoolElements));
      // PoolTy = { sbyte*, uint }*

      CurModule = 0; DS = 0;
      PoolInit = PoolDestroy = PoolAlloc = PoolFree = 0;
    }

    bool run(Module *M);

    // getAnalysisUsageInfo - This function requires data structure information
    // to be able to see what is pool allocatable.
    //
    virtual void getAnalysisUsageInfo(Pass::AnalysisSet &Required,
                                      Pass::AnalysisSet &,Pass::AnalysisSet &) {
      Required.push_back(DataStructure::ID);
    }

  public:
    // CurModule - The module being processed.
    Module *CurModule;

    // DS - The data structure graph for the module being processed.
    DataStructure *DS;

    // Prototypes that we add to support pool allocation...
    Function *PoolInit, *PoolDestroy, *PoolAlloc, *PoolFree;

    // The map of already transformed functions... note that the keys of this
    // map do not have meaningful values for 'Call' or the 'PoolHandle' elements
    // of the ArgInfo elements.
    //
    map<TransformFunctionInfo, Function*> TransformedFunctions;

    // getTransformedFunction - Get a transformed function, or return null if
    // the function specified hasn't been transformed yet.
    //
    Function *getTransformedFunction(TransformFunctionInfo &TFI) const {
      map<TransformFunctionInfo, Function*>::const_iterator I =
        TransformedFunctions.find(TFI);
      if (I != TransformedFunctions.end()) return I->second;
      return 0;
    }


    // addPoolPrototypes - Add prototypes for the pool methods to the specified
    // module and update the Pool* instance variables to point to them.
    //
    void addPoolPrototypes(Module *M);


    // CreatePools - Insert instructions into the function we are processing to
    // create all of the memory pool objects themselves.  This also inserts
    // destruction code.  Add an alloca for each pool that is allocated to the
    // PoolDescriptors map.
    //
    void CreatePools(Function *F, const vector<AllocDSNode*> &Allocs,
                     map<DSNode*, Value*> &PoolDescriptors);

    // processFunction - Convert a function to use pool allocation where
    // available.
    //
    bool processFunction(Function *F);

    // transformFunctionBody - This transforms the instruction in 'F' to use the
    // pools specified in PoolDescriptors when modifying data structure nodes
    // specified in the PoolDescriptors map.  IPFGraph is the closed data
    // structure graph for F, of which the PoolDescriptor nodes come from.
    //
    void transformFunctionBody(Function *F, FunctionDSGraph &IPFGraph,
                               map<DSNode*, Value*> &PoolDescriptors);

    // transformFunction - Transform the specified function the specified way.
    // It we have already transformed that function that way, don't do anything.
    // The nodes in the TransformFunctionInfo come out of callers data structure
    // graph.
    //
    void transformFunction(TransformFunctionInfo &TFI,
                           FunctionDSGraph &CallerIPGraph);

  };
}



// isNotPoolableAlloc - This is a predicate that returns true if the specified
// allocation node in a data structure graph is eligable for pool allocation.
//
static bool isNotPoolableAlloc(const AllocDSNode *DS) {
  if (DS->isAllocaNode()) return true;  // Do not pool allocate alloca's.

  MallocInst *MI = cast<MallocInst>(DS->getAllocation());
  if (MI->isArrayAllocation() && !isa<Constant>(MI->getArraySize()))
    return true;   // Do not allow variable size allocations...

  return false;
}

// processFunction - Convert a function to use pool allocation where
// available.
//
bool PoolAllocate::processFunction(Function *F) {
  // Get the closed datastructure graph for the current function... if there are
  // any allocations in this graph that are not escaping, we need to pool
  // allocate them here!
  //
  FunctionDSGraph &IPGraph = DS->getClosedDSGraph(F);

  // Get all of the allocations that do not escape the current function.  Since
  // they are still live (they exist in the graph at all), this means we must
  // have scalar references to these nodes, but the scalars are never returned.
  // 
  vector<AllocDSNode*> Allocs;
  IPGraph.getNonEscapingAllocations(Allocs);

  // Filter out allocations that we cannot handle.  Currently, this includes
  // variable sized array allocations and alloca's (which we do not want to
  // pool allocate)
  //
  Allocs.erase(remove_if(Allocs.begin(), Allocs.end(), isNotPoolableAlloc),
               Allocs.end());


  if (Allocs.empty()) return false;  // Nothing to do.

  // Insert instructions into the function we are processing to create all of
  // the memory pool objects themselves.  This also inserts destruction code.
  // This fills in the PoolDescriptors map to associate the alloc node with the
  // allocation of the memory pool corresponding to it.
  // 
  map<DSNode*, Value*> PoolDescriptors;
  CreatePools(F, Allocs, PoolDescriptors);

  // Now we need to figure out what called methods we need to transform, and
  // how.  To do this, we look at all of the scalars, seeing which functions are
  // either used as a scalar value (so they return a data structure), or are
  // passed one of our scalar values.
  //
  transformFunctionBody(F, IPGraph, PoolDescriptors);

  return true;
}


class FunctionBodyTransformer : public InstVisitor<FunctionBodyTransformer> {
  PoolAllocate &PoolAllocator;
  vector<ScalarInfo> &Scalars;
  map<CallInst*, TransformFunctionInfo> &CallMap;

  const ScalarInfo &getScalar(const Value *V) {
    for (unsigned i = 0, e = Scalars.size(); i != e; ++i)
      if (Scalars[i].Val == V) return Scalars[i];
    assert(0 && "Scalar not found in getScalar!");
    abort();
    return Scalars[0];
  }

  // updateScalars - Map the scalars array entries that look like 'From' to look
  // like 'To'.
  //
  void updateScalars(Value *From, Value *To) {
    for (unsigned i = 0, e = Scalars.size(); i != e; ++i)
      if (Scalars[i].Val == From) Scalars[i].Val = To;
  }

public:
  FunctionBodyTransformer(PoolAllocate &PA, vector<ScalarInfo> &S,
                          map<CallInst*, TransformFunctionInfo> &C)
    : PoolAllocator(PA), Scalars(S), CallMap(C) {}

  void visitMemAccessInst(MemAccessInst *MAI) {
    // Don't do anything to load, store, or GEP yet...
  }

  // Convert a malloc instruction into a call to poolalloc
  void visitMallocInst(MallocInst *I) {
    const ScalarInfo &SC = getScalar(I);
    BasicBlock *BB = I->getParent();
    BasicBlock::iterator MI = find(BB->begin(), BB->end(), I);
    BB->getInstList().remove(MI);  // Remove the Malloc instruction from the BB

    // Create a new call to poolalloc before the malloc instruction
    vector<Value*> Args;
    Args.push_back(SC.PoolHandle);
    CallInst *Call = new CallInst(PoolAllocator.PoolAlloc, Args, I->getName());
    MI = BB->getInstList().insert(MI, Call)+1;

    // If the type desired is not void*, cast it now...
    Value *Ptr = Call;
    if (Call->getType() != I->getType()) {
      CastInst *CI = new CastInst(Ptr, I->getType(), I->getName());
      BB->getInstList().insert(MI, CI);
      Ptr = CI;
    }

    // Change everything that used the malloc to now use the pool alloc...
    I->replaceAllUsesWith(Ptr);

    // Update the scalars array...
    updateScalars(I, Ptr);

    // Delete the instruction now.
    delete I;
  }

  // Convert the free instruction into a call to poolfree
  void visitFreeInst(FreeInst *I) {
    Value *Ptr = I->getOperand(0);
    const ScalarInfo &SC = getScalar(Ptr);
    BasicBlock *BB = I->getParent();
    BasicBlock::iterator FI = find(BB->begin(), BB->end(), I);

    // If the value is not an sbyte*, convert it now!
    if (Ptr->getType() != PointerType::get(Type::SByteTy)) {
      CastInst *CI = new CastInst(Ptr, PointerType::get(Type::SByteTy),
                                  Ptr->getName());
      FI = BB->getInstList().insert(FI, CI)+1;
      Ptr = CI;
    }

    // Create a new call to poolfree before the free instruction
    vector<Value*> Args;
    Args.push_back(SC.PoolHandle);
    Args.push_back(Ptr);
    CallInst *Call = new CallInst(PoolAllocator.PoolFree, Args);
    FI = BB->getInstList().insert(FI, Call)+1;

    // Remove the old free instruction...
    delete BB->getInstList().remove(FI);
  }

  // visitCallInst - Create a new call instruction with the extra arguments for
  // all of the memory pools that the call needs.
  //
  void visitCallInst(CallInst *I) {
    TransformFunctionInfo &TI = CallMap[I];
    BasicBlock *BB = I->getParent();
    BasicBlock::iterator CI = find(BB->begin(), BB->end(), I);
    BB->getInstList().remove(CI);  // Remove the old call instruction

    // Start with all of the old arguments...
    vector<Value*> Args(I->op_begin()+1, I->op_end());

    // Add all of the pool arguments...
    for (unsigned i = 0, e = TI.ArgInfo.size(); i != e; ++i)
      Args.push_back(TI.ArgInfo[i].PoolHandle);
    
    Function *NF = PoolAllocator.getTransformedFunction(TI);
    CallInst *NewCall = new CallInst(NF, Args, I->getName());
    BB->getInstList().insert(CI, NewCall);

    // Change everything that used the malloc to now use the pool alloc...
    if (I->getType() != Type::VoidTy) {
      I->replaceAllUsesWith(NewCall);

      // Update the scalars array...
      updateScalars(I, NewCall);
    }

    delete I;  // Delete the old call instruction now...
  }

  void visitPHINode(PHINode *PN) {
    // Handle PHI Node
  }

  void visitReturnInst(ReturnInst *I) {
    // Nothing of interest
  }

  void visitSetCondInst(SetCondInst *SCI) {
    // hrm, notice a pattern?
  }

  void visitInstruction(Instruction *I) {
    cerr << "Unknown instruction to FunctionBodyTransformer:\n";
    I->dump();
  }

};


static void addCallInfo(DataStructure *DS,
                        TransformFunctionInfo &TFI, CallInst *CI, int Arg, 
                        DSNode *GraphNode,
                        map<DSNode*, Value*> &PoolDescriptors) {
  assert(CI->getCalledFunction() && "Cannot handle indirect calls yet!");
  assert(TFI.Func == 0 || TFI.Func == CI->getCalledFunction() &&
         "Function call record should always call the same function!");
  assert(TFI.Call == 0 || TFI.Call == CI &&
         "Call element already filled in with different value!");
  TFI.Func = CI->getCalledFunction();
  TFI.Call = CI;
  //FunctionDSGraph &CalledGraph = DS->getClosedDSGraph(TFI.Func);

  // For now, add the entire graph that is pointed to by the call argument.
  // This graph can and should be pruned to only what the function itself will
  // use, because often this will be a dramatically smaller subset of what we
  // are providing.
  //
  for (df_iterator<DSNode*> I = df_begin(GraphNode), E = df_end(GraphNode);
       I != E; ++I) {
    TFI.ArgInfo.push_back(CallArgInfo(Arg, *I, PoolDescriptors[*I]));
  }
}


// transformFunctionBody - This transforms the instruction in 'F' to use the
// pools specified in PoolDescriptors when modifying data structure nodes
// specified in the PoolDescriptors map.  Specifically, scalar values specified
// in the Scalars vector must be remapped.  IPFGraph is the closed data
// structure graph for F, of which the PoolDescriptor nodes come from.
//
void PoolAllocate::transformFunctionBody(Function *F, FunctionDSGraph &IPFGraph,
                                       map<DSNode*, Value*> &PoolDescriptors) {

  // Loop through the value map looking for scalars that refer to nonescaping
  // allocations.  Add them to the Scalars vector.  Note that we may have
  // multiple entries in the Scalars vector for each value if it points to more
  // than one object.
  //
  map<Value*, PointerValSet> &ValMap = IPFGraph.getValueMap();
  vector<ScalarInfo> Scalars;

  cerr << "Building scalar map:\n";

  for (map<Value*, PointerValSet>::iterator I = ValMap.begin(),
         E = ValMap.end(); I != E; ++I) {
    const PointerValSet &PVS = I->second;  // Set of things pointed to by scalar

    cerr << "Scalar Mapping from:"; I->first->dump();
    cerr << "\nScalar Mapping to: "; PVS.print(cerr);

    // Check to see if the scalar points to a data structure node...
    for (unsigned i = 0, e = PVS.size(); i != e; ++i) {
      assert(PVS[i].Index == 0 && "Nonzero not handled yet!");
        
      // If the allocation is in the nonescaping set...
      map<DSNode*, Value*>::iterator AI = PoolDescriptors.find(PVS[i].Node);
      if (AI != PoolDescriptors.end()) // Add it to the list of scalars
        Scalars.push_back(ScalarInfo(I->first, PVS[i].Node, AI->second));
    }
  }



  cerr << "\nIn '" << F->getName()
       << "': Found the following values that point to poolable nodes:\n";

  for (unsigned i = 0, e = Scalars.size(); i != e; ++i)
    Scalars[i].Val->dump();

  // CallMap - Contain an entry for every call instruction that needs to be
  // transformed.  Each entry in the map contains information about what we need
  // to do to each call site to change it to work.
  //
  map<CallInst*, TransformFunctionInfo> CallMap;

  // Now we need to figure out what called methods we need to transform, and
  // how.  To do this, we look at all of the scalars, seeing which functions are
  // either used as a scalar value (so they return a data structure), or are
  // passed one of our scalar values.
  //
  for (unsigned i = 0, e = Scalars.size(); i != e; ++i) {
    Value *ScalarVal = Scalars[i].Val;

    // Check to see if the scalar _IS_ a call...
    if (CallInst *CI = dyn_cast<CallInst>(ScalarVal))
      // If so, add information about the pool it will be returning...
      addCallInfo(DS, CallMap[CI], CI, -1, Scalars[i].Node, PoolDescriptors);

    // Check to see if the scalar is an operand to a call...
    for (Value::use_iterator UI = ScalarVal->use_begin(),
           UE = ScalarVal->use_end(); UI != UE; ++UI) {
      if (CallInst *CI = dyn_cast<CallInst>(*UI)) {
        // Find out which operand this is to the call instruction...
        User::op_iterator OI = find(CI->op_begin(), CI->op_end(), ScalarVal);
        assert(OI != CI->op_end() && "Call on use list but not an operand!?");
        assert(OI != CI->op_begin() && "Pointer operand is call destination?");

        // FIXME: This is broken if the same pointer is passed to a call more
        // than once!  It will get multiple entries for the first pointer.

        // Add the operand number and pool handle to the call table...
        addCallInfo(DS, CallMap[CI], CI, OI-CI->op_begin()-1, Scalars[i].Node,
                    PoolDescriptors);
      }
    }
  }

  // Print out call map...
  for (map<CallInst*, TransformFunctionInfo>::iterator I = CallMap.begin();
       I != CallMap.end(); ++I) {
    cerr << "\nFor call: ";
    I->first->dump();
    I->second.finalizeConstruction();
    cerr << I->second.Func->getName() << " must pass pool pointer for args #";
    for (unsigned i = 0; i < I->second.ArgInfo.size(); ++i)
      cerr << I->second.ArgInfo[i].ArgNo << ", ";
    cerr << "\n";
  }

  // Loop through all of the call nodes, recursively creating the new functions
  // that we want to call...  This uses a map to prevent infinite recursion and
  // to avoid duplicating functions unneccesarily.
  //
  for (map<CallInst*, TransformFunctionInfo>::iterator I = CallMap.begin(),
         E = CallMap.end(); I != E; ++I) {
    // Make sure the entries are sorted.
    I->second.finalizeConstruction();

    // Transform all of the functions we need, or at least ensure there is a
    // cached version available.
    transformFunction(I->second, IPFGraph);
  }

  // Now that all of the functions that we want to call are available, transform
  // the local method so that it uses the pools locally and passes them to the
  // functions that we just hacked up.
  //

  // First step, find the instructions to be modified.
  vector<Instruction*> InstToFix;
  for (unsigned i = 0, e = Scalars.size(); i != e; ++i) {
    Value *ScalarVal = Scalars[i].Val;

    // Check to see if the scalar _IS_ an instruction.  If so, it is involved.
    if (Instruction *Inst = dyn_cast<Instruction>(ScalarVal))
      InstToFix.push_back(Inst);

    // All all of the instructions that use the scalar as an operand...
    for (Value::use_iterator UI = ScalarVal->use_begin(),
           UE = ScalarVal->use_end(); UI != UE; ++UI)
      InstToFix.push_back(dyn_cast<Instruction>(*UI));
  }

  // Eliminate duplicates by sorting, then removing equal neighbors.
  sort(InstToFix.begin(), InstToFix.end());
  InstToFix.erase(unique(InstToFix.begin(), InstToFix.end()), InstToFix.end());

  // Use a FunctionBodyTransformer to transform all of the involved instructions
  FunctionBodyTransformer FBT(*this, Scalars, CallMap);
  for (unsigned i = 0, e = InstToFix.size(); i != e; ++i)
    FBT.visit(InstToFix[i]);


  // Since we have liberally hacked the function to pieces, we want to inform
  // the datastructure pass that its internal representation is out of date.
  //
  DS->invalidateFunction(F);
}

static void addNodeMapping(DSNode *SrcNode, const PointerValSet &PVS,
                           map<DSNode*, PointerValSet> &NodeMapping) {
  for (unsigned i = 0, e = PVS.size(); i != e; ++i)
    if (NodeMapping[SrcNode].add(PVS[i])) {  // Not in map yet?
      assert(PVS[i].Index == 0 && "Node indexing not supported yet!");
      DSNode *DestNode = PVS[i].Node;

      // Loop over all of the outgoing links in the mapped graph
      for (unsigned l = 0, le = DestNode->getNumOutgoingLinks(); l != le; ++l) {
        PointerValSet &SrcSet = SrcNode->getOutgoingLink(l);
        const PointerValSet &DestSet = DestNode->getOutgoingLink(l);

        // Add all of the node mappings now!
        for (unsigned si = 0, se = SrcSet.size(); si != se; ++si) {
          assert(SrcSet[si].Index == 0 && "Can't handle node offset!");
          addNodeMapping(SrcSet[si].Node, DestSet, NodeMapping);
        }
      }
    }
}

// CalculateNodeMapping - There is a partial isomorphism between the graph
// passed in and the graph that is actually used by the function.  We need to
// figure out what this mapping is so that we can transformFunctionBody the
// instructions in the function itself.  Note that every node in the graph that
// we are interested in must be both in the local graph of the called function,
// and in the local graph of the calling function.  Because of this, we only
// define the mapping for these nodes [conveniently these are the only nodes we
// CAN define a mapping for...]
//
// The roots of the graph that we are transforming is rooted in the arguments
// passed into the function from the caller.  This is where we start our
// mapping calculation.
//
// The NodeMapping calculated maps from the callers graph to the called graph.
//
static void CalculateNodeMapping(Function *F, TransformFunctionInfo &TFI,
                                 FunctionDSGraph &CallerGraph,
                                 FunctionDSGraph &CalledGraph, 
                                 map<DSNode*, PointerValSet> &NodeMapping) {
  int LastArgNo = -2;
  for (unsigned i = 0, e = TFI.ArgInfo.size(); i != e; ++i) {
    // Figure out what nodes in the called graph the TFI.ArgInfo[i].Node node
    // corresponds to...
    //
    // Only consider first node of sequence.  Extra nodes may may be added
    // to the TFI if the data structure requires more nodes than just the
    // one the argument points to.  We are only interested in the one the
    // argument points to though.
    //
    if (TFI.ArgInfo[i].ArgNo != LastArgNo) {
      if (TFI.ArgInfo[i].ArgNo == -1) {
        addNodeMapping(TFI.ArgInfo[i].Node, CalledGraph.getRetNodes(),
                       NodeMapping);
      } else {
        // Figure out which node argument # ArgNo points to in the called graph.
        Value *Arg = F->getArgumentList()[TFI.ArgInfo[i].ArgNo];     
        addNodeMapping(TFI.ArgInfo[i].Node, CalledGraph.getValueMap()[Arg],
                       NodeMapping);
      }
      LastArgNo = TFI.ArgInfo[i].ArgNo;
    }
  }
}


// transformFunction - Transform the specified function the specified way.  It
// we have already transformed that function that way, don't do anything.  The
// nodes in the TransformFunctionInfo come out of callers data structure graph.
//
void PoolAllocate::transformFunction(TransformFunctionInfo &TFI,
                                     FunctionDSGraph &CallerIPGraph) {
  if (getTransformedFunction(TFI)) return;  // Function xformation already done?

  cerr << "**********\nEntering transformFunction for "
       << TFI.Func->getName() << ":\n";
  for (unsigned i = 0, e = TFI.ArgInfo.size(); i != e; ++i)
    cerr << "  ArgInfo[" << i << "] = " << TFI.ArgInfo[i].ArgNo << "\n";
  cerr << "\n";


  const FunctionType *OldFuncType = TFI.Func->getFunctionType();

  assert(!OldFuncType->isVarArg() && "Vararg functions not handled yet!");

  // Build the type for the new function that we are transforming
  vector<const Type*> ArgTys;
  for (unsigned i = 0, e = OldFuncType->getNumParams(); i != e; ++i)
    ArgTys.push_back(OldFuncType->getParamType(i));

  // Add one pool pointer for every argument that needs to be supplemented.
  ArgTys.insert(ArgTys.end(), TFI.ArgInfo.size(), PoolTy);

  // Build the new function type...
  const // FIXME when types are not const
  FunctionType *NewFuncType = FunctionType::get(OldFuncType->getReturnType(),
                                                ArgTys,OldFuncType->isVarArg());

  // The new function is internal, because we know that only we can call it.
  // This also helps subsequent IP transformations to eliminate duplicated pool
  // pointers. [in the future when they are implemented].
  //
  Function *NewFunc = new Function(NewFuncType, true,
                                   TFI.Func->getName()+".poolxform");
  CurModule->getFunctionList().push_back(NewFunc);

  // Add the newly formed function to the TransformedFunctions table so that
  // infinite recursion does not occur!
  //
  TransformedFunctions[TFI] = NewFunc;

  // Add arguments to the function... starting with all of the old arguments
  vector<Value*> ArgMap;
  for (unsigned i = 0, e = TFI.Func->getArgumentList().size(); i != e; ++i) {
    const Argument *OFA = TFI.Func->getArgumentList()[i];
    Argument *NFA = new Argument(OFA->getType(), OFA->getName());
    NewFunc->getArgumentList().push_back(NFA);
    ArgMap.push_back(NFA);  // Keep track of the arguments 
  }

  // Now add all of the arguments corresponding to pools passed in...
  for (unsigned i = 0, e = TFI.ArgInfo.size(); i != e; ++i) {
    string Name;
    if (TFI.ArgInfo[i].ArgNo == -1)
      Name = "retpool";
    else
      Name = ArgMap[TFI.ArgInfo[i].ArgNo]->getName();  // Get the arg name
    Argument *NFA = new Argument(PoolTy, Name+".pool");
    NewFunc->getArgumentList().push_back(NFA);
  }

  // Now clone the body of the old function into the new function...
  CloneFunctionInto(NewFunc, TFI.Func, ArgMap);
  
  // Okay, now we have a function that is identical to the old one, except that
  // it has extra arguments for the pools coming in.  Now we have to get the 
  // data structure graph for the function we are replacing, and figure out how
  // our graph nodes map to the graph nodes in the dest function.
  //
  FunctionDSGraph &DSGraph = DS->getClosedDSGraph(NewFunc);  

  // NodeMapping - Multimap from callers graph to called graph.
  //
  map<DSNode*, PointerValSet> NodeMapping;

  CalculateNodeMapping(NewFunc, TFI, CallerIPGraph, DSGraph, 
                       NodeMapping);

  // Print out the node mapping...
  cerr << "\nNode mapping for call of " << NewFunc->getName() << "\n";
  for (map<DSNode*, PointerValSet>::iterator I = NodeMapping.begin();
       I != NodeMapping.end(); ++I) {
    cerr << "Map: "; I->first->print(cerr);
    cerr << "To:  "; I->second.print(cerr);
    cerr << "\n";
  }

  // Fill in the PoolDescriptor information for the transformed function so that
  // it can determine which value holds the pool descriptor for each data
  // structure node that it accesses.
  //
  map<DSNode*, Value*> PoolDescriptors;

  cerr << "\nCalculating the pool descriptor map:\n";

  // All of the pool descriptors must be passed in as arguments...
  for (unsigned i = 0, e = TFI.ArgInfo.size(); i != e; ++i) {
    DSNode *CallerNode = TFI.ArgInfo[i].Node;
    Value  *CallerPool = TFI.ArgInfo[i].PoolHandle;

    cerr << "Mapped caller node: "; CallerNode->print(cerr);
    cerr << "Mapped caller pool: "; CallerPool->dump();

    // Calculate the argument number that the pool is to the function call...
    // The call instruction should not have the pool operands added yet.
    unsigned ArgNo = TFI.Call->getNumOperands()-1+i;
    cerr << "Should be argument #: " << ArgNo << "[i = " << i << "]\n";
    assert(ArgNo < NewFunc->getArgumentList().size() &&
           "Call already has pool arguments added??");

    // Map the pool argument into the called function...
    Value *CalleePool = NewFunc->getArgumentList()[ArgNo];

    // Map the DSNode into the callee's DSGraph
    const PointerValSet &CalleeNodes = NodeMapping[CallerNode];
    for (unsigned n = 0, ne = CalleeNodes.size(); n != ne; ++n) {
      assert(CalleeNodes[n].Index == 0 && "Indexed node not handled yet!");
      DSNode *CalleeNode = CalleeNodes[n].Node;

      cerr << "*** to callee node: "; CalleeNode->print(cerr);
      cerr << "*** to callee pool: "; CalleePool->dump();
      cerr << "\n";
      
      assert(CalleeNode && CalleePool && "Invalid nodes!");
      Value *&PV = PoolDescriptors[CalleeNode];
      //assert((PV == 0 || PV == CalleePool) && "Invalid node remapping!");
      PV = CalleePool;         // Update the pool descriptor map!
    }
  }

  // We must destroy the node mapping so that we don't have latent references
  // into the data structure graph for the new function.  Otherwise we get
  // assertion failures when transformFunctionBody tries to invalidate the
  // graph.
  //
  NodeMapping.clear();

  // Now that we know everything we need about the function, transform the body
  // now!
  //
  transformFunctionBody(NewFunc, DSGraph, PoolDescriptors);

  cerr << "Function after transformation:\n";
  NewFunc->dump();
}


// CreatePools - Insert instructions into the function we are processing to
// create all of the memory pool objects themselves.  This also inserts
// destruction code.  Add an alloca for each pool that is allocated to the
// PoolDescriptors vector.
//
void PoolAllocate::CreatePools(Function *F, const vector<AllocDSNode*> &Allocs,
                               map<DSNode*, Value*> &PoolDescriptors) {
  // FIXME: This should use an IP version of the UnifyAllExits pass!
  vector<BasicBlock*> ReturnNodes;
  for (Function::iterator I = F->begin(), E = F->end(); I != E; ++I)
    if (isa<ReturnInst>((*I)->getTerminator()))
      ReturnNodes.push_back(*I);
  

  // Create the code that goes in the entry and exit nodes for the method...
  vector<Instruction*> EntryNodeInsts;
  for (unsigned i = 0, e = Allocs.size(); i != e; ++i) {
    // Add an allocation and a free for each pool...
    AllocaInst *PoolAlloc = new AllocaInst(PoolTy, 0, "pool");
    EntryNodeInsts.push_back(PoolAlloc);
    PoolDescriptors[Allocs[i]] = PoolAlloc;   // Keep track of pool allocas
    AllocationInst *AI = Allocs[i]->getAllocation();

    // Initialize the pool.  We need to know how big each allocation is.  For
    // our purposes here, we assume we are allocating a scalar, or array of
    // constant size.
    //
    unsigned ElSize = TargetData.getTypeSize(AI->getAllocatedType());
    ElSize *= cast<ConstantUInt>(AI->getArraySize())->getValue();

    vector<Value*> Args;
    Args.push_back(PoolAlloc);    // Pool to initialize
    Args.push_back(ConstantUInt::get(Type::UIntTy, ElSize));
    EntryNodeInsts.push_back(new CallInst(PoolInit, Args));

    // Destroy the pool...
    Args.pop_back();

    for (unsigned EN = 0, ENE = ReturnNodes.size(); EN != ENE; ++EN) {
      Instruction *Destroy = new CallInst(PoolDestroy, Args);

      // Insert it before the return instruction...
      BasicBlock *RetNode = ReturnNodes[EN];
      RetNode->getInstList().insert(RetNode->end()-1, Destroy);
    }
  }

  // Insert the entry node code into the entry block...
  F->getEntryNode()->getInstList().insert(F->getEntryNode()->begin()+1,
                                          EntryNodeInsts.begin(),
                                          EntryNodeInsts.end());
}


// addPoolPrototypes - Add prototypes for the pool methods to the specified
// module and update the Pool* instance variables to point to them.
//
void PoolAllocate::addPoolPrototypes(Module *M) {
  // Get PoolInit function...
  vector<const Type*> Args;
  Args.push_back(PoolTy);           // Pool to initialize
  Args.push_back(Type::UIntTy);     // Num bytes per element
  FunctionType *PoolInitTy = FunctionType::get(Type::VoidTy, Args, false);
  PoolInit = M->getOrInsertFunction("poolinit", PoolInitTy);

  // Get pooldestroy function...
  Args.pop_back();  // Only takes a pool...
  FunctionType *PoolDestroyTy = FunctionType::get(Type::VoidTy, Args, false);
  PoolDestroy = M->getOrInsertFunction("pooldestroy", PoolDestroyTy);

  const Type *PtrVoid = PointerType::get(Type::SByteTy);

  // Get the poolalloc function...
  FunctionType *PoolAllocTy = FunctionType::get(PtrVoid, Args, false);
  PoolAlloc = M->getOrInsertFunction("poolalloc", PoolAllocTy);

  // Get the poolfree function...
  Args.push_back(PtrVoid);
  FunctionType *PoolFreeTy = FunctionType::get(Type::VoidTy, Args, false);
  PoolFree = M->getOrInsertFunction("poolfree", PoolFreeTy);

  // Add the %PoolTy type to the symbol table of the module...
  M->addTypeName("PoolTy", PoolTy->getElementType());
}


bool PoolAllocate::run(Module *M) {
  addPoolPrototypes(M);
  CurModule = M;
  
  DS = &getAnalysis<DataStructure>();
  bool Changed = false;

  // We cannot use an iterator here because it will get invalidated when we add
  // functions to the module later...
  for (unsigned i = 0; i != M->size(); ++i)
    if (!M->getFunctionList()[i]->isExternal()) {
      Changed |= processFunction(M->getFunctionList()[i]);
      if (Changed) {
        cerr << "Only processing one function\n";
        break;
      }
    }

  CurModule = 0;
  DS = 0;
  return false;
}


// createPoolAllocatePass - Global function to access the functionality of this
// pass...
//
Pass *createPoolAllocatePass() { return new PoolAllocate(); }
