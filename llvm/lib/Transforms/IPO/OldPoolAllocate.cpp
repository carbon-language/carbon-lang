//===-- PoolAllocate.cpp - Pool Allocation Pass ---------------------------===//
//
// This transform changes programs so that disjoint data structures are
// allocated out of different pools of memory, increasing locality and shrinking
// pointer size.
//
//===----------------------------------------------------------------------===//

#if 0
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Analysis/DataStructure.h"
#include "llvm/Module.h"
#include "llvm/iMemory.h"
#include "llvm/iTerminators.h"
#include "llvm/iPHINode.h"
#include "llvm/iOther.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Constants.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Support/InstVisitor.h"
#include "Support/DepthFirstIterator.h"
#include "Support/STLExtras.h"
#include <algorithm>
using std::vector;
using std::cerr;
using std::map;
using std::string;
using std::set;

// DEBUG_CREATE_POOLS - Enable this to turn on debug output for the pool
// creation phase in the top level function of a transformed data structure.
//
//#define DEBUG_CREATE_POOLS 1

// DEBUG_TRANSFORM_PROGRESS - Enable this to get lots of debug output on what
// the transformation is doing.
//
//#define DEBUG_TRANSFORM_PROGRESS 1

// DEBUG_POOLBASE_LOAD_ELIMINATOR - Turn this on to get statistics about how
// many static loads were eliminated from a function...
//
#define DEBUG_POOLBASE_LOAD_ELIMINATOR 1

#include "Support/CommandLine.h"
enum PtrSize {
  Ptr8bits, Ptr16bits, Ptr32bits
};

static cl::opt<PtrSize>
ReqPointerSize("poolalloc-ptr-size",
               cl::desc("Set pointer size for -poolalloc pass"),
               cl::values(
  clEnumValN(Ptr32bits, "32", "Use 32 bit indices for pointers"),
  clEnumValN(Ptr16bits, "16", "Use 16 bit indices for pointers"),
  clEnumValN(Ptr8bits ,  "8", "Use 8 bit indices for pointers"),
                          0));

static cl::opt<bool>
DisableRLE("no-pool-load-elim",  cl::Hidden,
           cl::desc("Disable pool load elimination after poolalloc pass"));

const Type *POINTERTYPE;

// FIXME: This is dependant on the sparc backend layout conventions!!
static TargetData TargetData("test");

static const Type *getPointerTransformedType(const Type *Ty) {
  if (const PointerType *PT = dyn_cast<PointerType>(Ty)) {
    return POINTERTYPE;
  } else if (const StructType *STy = dyn_cast<StructType>(Ty)) {
    vector<const Type *> NewElTypes;
    NewElTypes.reserve(STy->getElementTypes().size());
    for (StructType::ElementTypes::const_iterator
           I = STy->getElementTypes().begin(),
           E = STy->getElementTypes().end(); I != E; ++I)
      NewElTypes.push_back(getPointerTransformedType(*I));
    return StructType::get(NewElTypes);
  } else if (const ArrayType *ATy = dyn_cast<ArrayType>(Ty)) {
    return ArrayType::get(getPointerTransformedType(ATy->getElementType()),
                                                    ATy->getNumElements());
  } else {
    assert(Ty->isPrimitiveType() && "Unknown derived type!");
    return Ty;
  }
}

namespace {
  struct PoolInfo {
    DSNode *Node;           // The node this pool allocation represents
    Value  *Handle;         // LLVM value of the pool in the current context
    const Type *NewType;    // The transformed type of the memory objects
    const Type *PoolType;   // The type of the pool

    const Type *getOldType() const { return Node->getType(); }

    PoolInfo() {  // Define a default ctor for map::operator[]
      cerr << "Map subscript used to get element that doesn't exist!\n";
      abort();  // Invalid
    }

    PoolInfo(DSNode *N, Value *H, const Type *NT, const Type *PT)
      : Node(N), Handle(H), NewType(NT), PoolType(PT) {
      // Handle can be null...
      assert(N && NT && PT && "Pool info null!");
    }

    PoolInfo(DSNode *N) : Node(N), Handle(0), NewType(0), PoolType(0) {
      assert(N && "Invalid pool info!");

      // The new type of the memory object is the same as the old type, except
      // that all of the pointer values are replaced with POINTERTYPE values.
      NewType = getPointerTransformedType(getOldType());
    }
  };

  // ScalarInfo - Information about an LLVM value that we know points to some
  // datastructure we are processing.
  //
  struct ScalarInfo {
    Value  *Val;            // Scalar value in Current Function
    PoolInfo Pool;          // The pool the scalar points into
    
    ScalarInfo(Value *V, const PoolInfo &PI) : Val(V), Pool(PI) {
      assert(V && "Null value passed to ScalarInfo ctor!");
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
    // processed.  Each CallArgInfo corresponds to an argument that needs to
    // have a pool pointer passed into the transformed function with it.
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
      std::stable_sort(ArgInfo.begin(), ArgInfo.end());
    }

    // addCallInfo - For a specified function call CI, figure out which pool
    // descriptors need to be passed in as arguments, and which arguments need
    // to be transformed into indices.  If Arg != -1, the specified call
    // argument is passed in as a pointer to a data structure.
    //
    void addCallInfo(DataStructure *DS, CallInst *CI, int Arg,
                     DSNode *GraphNode, map<DSNode*, PoolInfo> &PoolDescs);

    // Make sure that all dependant arguments are added to this transformation
    // info.  For example, if we call foo(null, P) and foo treats it's first and
    // second arguments as belonging to the same data structure, the we MUST add
    // entries to know that the null needs to be transformed into an index as
    // well.
    //
    void ensureDependantArgumentsIncluded(DataStructure *DS,
                                          map<DSNode*, PoolInfo> &PoolDescs);
  };


  // Define the pass class that we implement...
  struct PoolAllocate : public Pass {
    PoolAllocate() {
      switch (ReqPointerSize) {
      case Ptr32bits: POINTERTYPE = Type::UIntTy; break;
      case Ptr16bits: POINTERTYPE = Type::UShortTy; break;
      case Ptr8bits:  POINTERTYPE = Type::UByteTy; break;
      }

      CurModule = 0; DS = 0;
      PoolInit = PoolDestroy = PoolAlloc = PoolFree = 0;
    }

    // getPoolType - Get the type used by the backend for a pool of a particular
    // type.  This pool record is used to allocate nodes of type NodeType.
    //
    // Here, PoolTy = { NodeType*, sbyte*, uint }*
    //
    const StructType *getPoolType(const Type *NodeType) {
      vector<const Type*> PoolElements;
      PoolElements.push_back(PointerType::get(NodeType));
      PoolElements.push_back(PointerType::get(Type::SByteTy));
      PoolElements.push_back(Type::UIntTy);
      StructType *Result = StructType::get(PoolElements);

      // Add a name to the symbol table to correspond to the backend
      // representation of this pool...
      assert(CurModule && "No current module!?");
      string Name = CurModule->getTypeName(NodeType);
      if (Name.empty()) Name = CurModule->getTypeName(PoolElements[0]);
      CurModule->addTypeName(Name+"oolbe", Result);

      return Result;
    }

    bool run(Module &M);

    // getAnalysisUsage - This function requires data structure information
    // to be able to see what is pool allocatable.
    //
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<DataStructure>();
    }

  public:
    // CurModule - The module being processed.
    Module *CurModule;

    // DS - The data structure graph for the module being processed.
    DataStructure *DS;

    // Prototypes that we add to support pool allocation...
    Function *PoolInit, *PoolDestroy, *PoolAlloc, *PoolAllocArray, *PoolFree;

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


    // addPoolPrototypes - Add prototypes for the pool functions to the
    // specified module and update the Pool* instance variables to point to
    // them.
    //
    void addPoolPrototypes(Module &M);


    // CreatePools - Insert instructions into the function we are processing to
    // create all of the memory pool objects themselves.  This also inserts
    // destruction code.  Add an alloca for each pool that is allocated to the
    // PoolDescs map.
    //
    void CreatePools(Function *F, const vector<AllocDSNode*> &Allocs,
                     map<DSNode*, PoolInfo> &PoolDescs);

    // processFunction - Convert a function to use pool allocation where
    // available.
    //
    bool processFunction(Function *F);

    // transformFunctionBody - This transforms the instruction in 'F' to use the
    // pools specified in PoolDescs when modifying data structure nodes
    // specified in the PoolDescs map.  IPFGraph is the closed data structure
    // graph for F, of which the PoolDescriptor nodes come from.
    //
    void transformFunctionBody(Function *F, FunctionDSGraph &IPFGraph,
                               map<DSNode*, PoolInfo> &PoolDescs);

    // transformFunction - Transform the specified function the specified way.
    // It we have already transformed that function that way, don't do anything.
    // The nodes in the TransformFunctionInfo come out of callers data structure
    // graph, and the PoolDescs passed in are the caller's.
    //
    void transformFunction(TransformFunctionInfo &TFI,
                           FunctionDSGraph &CallerIPGraph,
                           map<DSNode*, PoolInfo> &PoolDescs);

  };

  RegisterOpt<PoolAllocate> X("poolalloc",
                              "Pool allocate disjoint datastructures");
}

// isNotPoolableAlloc - This is a predicate that returns true if the specified
// allocation node in a data structure graph is eligable for pool allocation.
//
static bool isNotPoolableAlloc(const AllocDSNode *DS) {
  if (DS->isAllocaNode()) return true;  // Do not pool allocate alloca's.
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
  Allocs.erase(std::remove_if(Allocs.begin(), Allocs.end(), isNotPoolableAlloc),
               Allocs.end());


  if (Allocs.empty()) return false;  // Nothing to do.

#ifdef DEBUG_TRANSFORM_PROGRESS
  cerr << "Transforming Function: " << F->getName() << "\n";
#endif

  // Insert instructions into the function we are processing to create all of
  // the memory pool objects themselves.  This also inserts destruction code.
  // This fills in the PoolDescs map to associate the alloc node with the
  // allocation of the memory pool corresponding to it.
  // 
  map<DSNode*, PoolInfo> PoolDescs;
  CreatePools(F, Allocs, PoolDescs);

#ifdef DEBUG_TRANSFORM_PROGRESS
  cerr << "Transformed Entry Function: \n" << F;
#endif

  // Now we need to figure out what called functions we need to transform, and
  // how.  To do this, we look at all of the scalars, seeing which functions are
  // either used as a scalar value (so they return a data structure), or are
  // passed one of our scalar values.
  //
  transformFunctionBody(F, IPGraph, PoolDescs);

  return true;
}


//===----------------------------------------------------------------------===//
//
// NewInstructionCreator - This class is used to traverse the function being
// modified, changing each instruction visit'ed to use and provide pointer
// indexes instead of real pointers.  This is what changes the body of a
// function to use pool allocation.
//
class NewInstructionCreator : public InstVisitor<NewInstructionCreator> {
  PoolAllocate &PoolAllocator;
  vector<ScalarInfo> &Scalars;
  map<CallInst*, TransformFunctionInfo> &CallMap;
  map<Value*, Value*> &XFormMap;   // Map old pointers to new indexes

  struct RefToUpdate {
    Instruction *I;       // Instruction to update
    unsigned     OpNum;   // Operand number to update
    Value       *OldVal;  // The old value it had

    RefToUpdate(Instruction *i, unsigned o, Value *ov)
      : I(i), OpNum(o), OldVal(ov) {}
  };
  vector<RefToUpdate> ReferencesToUpdate;

  const ScalarInfo &getScalarRef(const Value *V) {
    for (unsigned i = 0, e = Scalars.size(); i != e; ++i)
      if (Scalars[i].Val == V) return Scalars[i];

    cerr << "Could not find scalar " << V << " in scalar map!\n";
    assert(0 && "Scalar not found in getScalar!");
    abort();
    return Scalars[0];
  }
  
  const ScalarInfo *getScalar(const Value *V) {
    for (unsigned i = 0, e = Scalars.size(); i != e; ++i)
      if (Scalars[i].Val == V) return &Scalars[i];
    return 0;
  }

  BasicBlock::iterator ReplaceInstWith(Instruction &I, Instruction *New) {
    BasicBlock *BB = I.getParent();
    BasicBlock::iterator RI = &I;
    BB->getInstList().remove(RI);
    BB->getInstList().insert(RI, New);
    XFormMap[&I] = New;
    return New;
  }

  Instruction *createPoolBaseInstruction(Value *PtrVal) {
    const ScalarInfo &SC = getScalarRef(PtrVal);
    vector<Value*> Args(3);
    Args[0] = ConstantUInt::get(Type::UIntTy, 0);  // No pointer offset
    Args[1] = ConstantUInt::get(Type::UByteTy, 0); // Field #0 of pool descriptr
    Args[2] = ConstantUInt::get(Type::UByteTy, 0); // Field #0 of poolalloc val
    return  new LoadInst(SC.Pool.Handle, Args, PtrVal->getName()+".poolbase");
  }


public:
  NewInstructionCreator(PoolAllocate &PA, vector<ScalarInfo> &S,
                        map<CallInst*, TransformFunctionInfo> &C,
                        map<Value*, Value*> &X)
    : PoolAllocator(PA), Scalars(S), CallMap(C), XFormMap(X) {}


  // updateReferences - The NewInstructionCreator is responsible for creating
  // new instructions to replace the old ones in the function, and then link up
  // references to values to their new values.  For it to do this, however, it
  // keeps track of information about the value mapping of old values to new
  // values that need to be patched up.  Given this value map and a set of
  // instruction operands to patch, updateReferences performs the updates.
  //
  void updateReferences() {
    for (unsigned i = 0, e = ReferencesToUpdate.size(); i != e; ++i) {
      RefToUpdate &Ref = ReferencesToUpdate[i];
      Value *NewVal = XFormMap[Ref.OldVal];

      if (NewVal == 0) {
        if (isa<Constant>(Ref.OldVal) &&  // Refering to a null ptr?
            cast<Constant>(Ref.OldVal)->isNullValue()) {
          // Transform the null pointer into a null index... caching in XFormMap
          XFormMap[Ref.OldVal] = NewVal = Constant::getNullValue(POINTERTYPE);
          //} else if (isa<Argument>(Ref.OldVal)) {
        } else {
          cerr << "Unknown reference to: " << Ref.OldVal << "\n";
          assert(XFormMap[Ref.OldVal] &&
                 "Reference to value that was not updated found!");
        }
      }
        
      Ref.I->setOperand(Ref.OpNum, NewVal);
    }
    ReferencesToUpdate.clear();
  }

  //===--------------------------------------------------------------------===//
  // Transformation methods:
  //   These methods specify how each type of instruction is transformed by the
  // NewInstructionCreator instance...
  //===--------------------------------------------------------------------===//

  void visitGetElementPtrInst(GetElementPtrInst &I) {
    assert(0 && "Cannot transform get element ptr instructions yet!");
  }

  // Replace the load instruction with a new one.
  void visitLoadInst(LoadInst &I) {
    vector<Instruction *> BeforeInsts;

    // Cast our index to be a UIntTy so we can use it to index into the pool...
    CastInst *Index = new CastInst(Constant::getNullValue(POINTERTYPE),
                                   Type::UIntTy, I.getOperand(0)->getName());
    BeforeInsts.push_back(Index);
    ReferencesToUpdate.push_back(RefToUpdate(Index, 0, I.getOperand(0)));
    
    // Include the pool base instruction...
    Instruction *PoolBase = createPoolBaseInstruction(I.getOperand(0));
    BeforeInsts.push_back(PoolBase);

    Instruction *IdxInst =
      BinaryOperator::create(Instruction::Add, *I.idx_begin(), Index,
                             I.getName()+".idx");
    BeforeInsts.push_back(IdxInst);

    vector<Value*> Indices(I.idx_begin(), I.idx_end());
    Indices[0] = IdxInst;
    Instruction *Address = new GetElementPtrInst(PoolBase, Indices,
                                                 I.getName()+".addr");
    BeforeInsts.push_back(Address);

    Instruction *NewLoad = new LoadInst(Address, I.getName());

    // Replace the load instruction with the new load instruction...
    BasicBlock::iterator II = ReplaceInstWith(I, NewLoad);

    // Add all of the instructions before the load...
    NewLoad->getParent()->getInstList().insert(II, BeforeInsts.begin(),
                                               BeforeInsts.end());

    // If not yielding a pool allocated pointer, use the new load value as the
    // value in the program instead of the old load value...
    //
    if (!getScalar(&I))
      I.replaceAllUsesWith(NewLoad);
  }

  // Replace the store instruction with a new one.  In the store instruction,
  // the value stored could be a pointer type, meaning that the new store may
  // have to change one or both of it's operands.
  //
  void visitStoreInst(StoreInst &I) {
    assert(getScalar(I.getOperand(1)) &&
           "Store inst found only storing pool allocated pointer.  "
           "Not imp yet!");

    Value *Val = I.getOperand(0);  // The value to store...

    // Check to see if the value we are storing is a data structure pointer...
    //if (const ScalarInfo *ValScalar = getScalar(I.getOperand(0)))
    if (isa<PointerType>(I.getOperand(0)->getType()))
      Val = Constant::getNullValue(POINTERTYPE);  // Yes, store a dummy

    Instruction *PoolBase = createPoolBaseInstruction(I.getOperand(1));

    // Cast our index to be a UIntTy so we can use it to index into the pool...
    CastInst *Index = new CastInst(Constant::getNullValue(POINTERTYPE),
                                   Type::UIntTy, I.getOperand(1)->getName());
    ReferencesToUpdate.push_back(RefToUpdate(Index, 0, I.getOperand(1)));

    // Instructions to add after the Index...
    vector<Instruction*> AfterInsts;

    Instruction *IdxInst =
      BinaryOperator::create(Instruction::Add, *I.idx_begin(), Index, "idx");
    AfterInsts.push_back(IdxInst);

    vector<Value*> Indices(I.idx_begin(), I.idx_end());
    Indices[0] = IdxInst;
    Instruction *Address = new GetElementPtrInst(PoolBase, Indices,
                                                 I.getName()+"storeaddr");
    AfterInsts.push_back(Address);

    Instruction *NewStore = new StoreInst(Val, Address);
    AfterInsts.push_back(NewStore);
    if (Val != I.getOperand(0))    // Value stored was a pointer?
      ReferencesToUpdate.push_back(RefToUpdate(NewStore, 0, I.getOperand(0)));


    // Replace the store instruction with the cast instruction...
    BasicBlock::iterator II = ReplaceInstWith(I, Index);

    // Add the pool base calculator instruction before the index...
    II = ++Index->getParent()->getInstList().insert(II, PoolBase);
    ++II;

    // Add the instructions that go after the index...
    Index->getParent()->getInstList().insert(II, AfterInsts.begin(),
                                             AfterInsts.end());
  }


  // Create call to poolalloc for every malloc instruction
  void visitMallocInst(MallocInst &I) {
    const ScalarInfo &SCI = getScalarRef(&I);
    vector<Value*> Args;

    CallInst *Call;
    if (!I.isArrayAllocation()) {
      Args.push_back(SCI.Pool.Handle);
      Call = new CallInst(PoolAllocator.PoolAlloc, Args, I.getName());
    } else {
      Args.push_back(I.getArraySize());
      Args.push_back(SCI.Pool.Handle);
      Call = new CallInst(PoolAllocator.PoolAllocArray, Args, I.getName());
    }    

    ReplaceInstWith(I, Call);
  }

  // Convert a call to poolfree for every free instruction...
  void visitFreeInst(FreeInst &I) {
    // Create a new call to poolfree before the free instruction
    vector<Value*> Args;
    Args.push_back(Constant::getNullValue(POINTERTYPE));
    Args.push_back(getScalarRef(I.getOperand(0)).Pool.Handle);
    Instruction *NewCall = new CallInst(PoolAllocator.PoolFree, Args);
    ReplaceInstWith(I, NewCall);
    ReferencesToUpdate.push_back(RefToUpdate(NewCall, 1, I.getOperand(0)));
  }

  // visitCallInst - Create a new call instruction with the extra arguments for
  // all of the memory pools that the call needs.
  //
  void visitCallInst(CallInst &I) {
    TransformFunctionInfo &TI = CallMap[&I];

    // Start with all of the old arguments...
    vector<Value*> Args(I.op_begin()+1, I.op_end());

    for (unsigned i = 0, e = TI.ArgInfo.size(); i != e; ++i) {
      // Replace all of the pointer arguments with our new pointer typed values.
      if (TI.ArgInfo[i].ArgNo != -1)
        Args[TI.ArgInfo[i].ArgNo] = Constant::getNullValue(POINTERTYPE);

      // Add all of the pool arguments...
      Args.push_back(TI.ArgInfo[i].PoolHandle);
    }
    
    Function *NF = PoolAllocator.getTransformedFunction(TI);
    Instruction *NewCall = new CallInst(NF, Args, I.getName());
    ReplaceInstWith(I, NewCall);

    // Keep track of the mapping of operands so that we can resolve them to real
    // values later.
    Value *RetVal = NewCall;
    for (unsigned i = 0, e = TI.ArgInfo.size(); i != e; ++i)
      if (TI.ArgInfo[i].ArgNo != -1)
        ReferencesToUpdate.push_back(RefToUpdate(NewCall, TI.ArgInfo[i].ArgNo+1,
                                        I.getOperand(TI.ArgInfo[i].ArgNo+1)));
      else
        RetVal = 0;   // If returning a pointer, don't change retval...

    // If not returning a pointer, use the new call as the value in the program
    // instead of the old call...
    //
    if (RetVal)
      I.replaceAllUsesWith(RetVal);
  }

  // visitPHINode - Create a new PHI node of POINTERTYPE for all of the old Phi
  // nodes...
  //
  void visitPHINode(PHINode &PN) {
    Value *DummyVal = Constant::getNullValue(POINTERTYPE);
    PHINode *NewPhi = new PHINode(POINTERTYPE, PN.getName());
    for (unsigned i = 0, e = PN.getNumIncomingValues(); i != e; ++i) {
      NewPhi->addIncoming(DummyVal, PN.getIncomingBlock(i));
      ReferencesToUpdate.push_back(RefToUpdate(NewPhi, i*2, 
                                               PN.getIncomingValue(i)));
    }

    ReplaceInstWith(PN, NewPhi);
  }

  // visitReturnInst - Replace ret instruction with a new return...
  void visitReturnInst(ReturnInst &I) {
    Instruction *Ret = new ReturnInst(Constant::getNullValue(POINTERTYPE));
    ReplaceInstWith(I, Ret);
    ReferencesToUpdate.push_back(RefToUpdate(Ret, 0, I.getOperand(0)));
  }

  // visitSetCondInst - Replace a conditional test instruction with a new one
  void visitSetCondInst(SetCondInst &SCI) {
    BinaryOperator &I = (BinaryOperator&)SCI;
    Value *DummyVal = Constant::getNullValue(POINTERTYPE);
    BinaryOperator *New = BinaryOperator::create(I.getOpcode(), DummyVal,
                                                 DummyVal, I.getName());
    ReplaceInstWith(I, New);

    ReferencesToUpdate.push_back(RefToUpdate(New, 0, I.getOperand(0)));
    ReferencesToUpdate.push_back(RefToUpdate(New, 1, I.getOperand(1)));

    // Make sure branches refer to the new condition...
    I.replaceAllUsesWith(New);
  }

  void visitInstruction(Instruction &I) {
    cerr << "Unknown instruction to FunctionBodyTransformer:\n" << I;
  }
};


// PoolBaseLoadEliminator - Every load and store through a pool allocated
// pointer causes a load of the real pool base out of the pool descriptor.
// Iterate through the function, doing a local elimination pass of duplicate
// loads.  This attempts to turn the all too common:
//
// %reg109.poolbase22 = load %root.pool* %root.pool, uint 0, ubyte 0, ubyte 0
// %reg207 = load %root.p* %reg109.poolbase22, uint %reg109, ubyte 0, ubyte 0
// %reg109.poolbase23 = load %root.pool* %root.pool, uint 0, ubyte 0, ubyte 0
// store double %reg207, %root.p* %reg109.poolbase23, uint %reg109, ...
//
// into:
// %reg109.poolbase22 = load %root.pool* %root.pool, uint 0, ubyte 0, ubyte 0
// %reg207 = load %root.p* %reg109.poolbase22, uint %reg109, ubyte 0, ubyte 0
// store double %reg207, %root.p* %reg109.poolbase22, uint %reg109, ...
//
//
class PoolBaseLoadEliminator : public InstVisitor<PoolBaseLoadEliminator> {
  // PoolDescValues - Keep track of the values in the current function that are
  // pool descriptors (loads from which we want to eliminate).
  //
  vector<Value*>      PoolDescValues;

  // PoolDescMap - As we are analyzing a BB, keep track of which load to use
  // when referencing a pool descriptor.
  //
  map<Value*, LoadInst*> PoolDescMap;

  // These two fields keep track of statistics of how effective we are, if
  // debugging is enabled.
  //
  unsigned Eliminated, Remaining;
public:
  // Compact the pool descriptor map into a list of the pool descriptors in the
  // current context that we should know about...
  //
  PoolBaseLoadEliminator(const map<DSNode*, PoolInfo> &PoolDescs) {
    Eliminated = Remaining = 0;
    for (map<DSNode*, PoolInfo>::const_iterator I = PoolDescs.begin(),
           E = PoolDescs.end(); I != E; ++I)
      PoolDescValues.push_back(I->second.Handle);
    
    // Remove duplicates from the list of pool values
    sort(PoolDescValues.begin(), PoolDescValues.end());
    PoolDescValues.erase(unique(PoolDescValues.begin(), PoolDescValues.end()),
                         PoolDescValues.end());
  }

#ifdef DEBUG_POOLBASE_LOAD_ELIMINATOR
  void visitFunction(Function &F) {
    cerr << "Pool Load Elim '" << F.getName() << "'\t";
  }
  ~PoolBaseLoadEliminator() {
    unsigned Total = Eliminated+Remaining;
    if (Total)
      cerr << "removed " << Eliminated << "["
           << Eliminated*100/Total << "%] loads, leaving "
           << Remaining << ".\n";
  }
#endif

  // Loop over the function, looking for loads to eliminate.  Because we are a
  // local transformation, we reset all of our state when we enter a new basic
  // block.
  //
  void visitBasicBlock(BasicBlock &) {
    PoolDescMap.clear();  // Forget state.
  }

  // Starting with an empty basic block, we scan it looking for loads of the
  // pool descriptor.  When we find a load, we add it to the PoolDescMap,
  // indicating that we have a value available to recycle next time we see the
  // poolbase of this instruction being loaded.
  //
  void visitLoadInst(LoadInst &LI) {
    Value *LoadAddr = LI.getPointerOperand();
    map<Value*, LoadInst*>::iterator VIt = PoolDescMap.find(LoadAddr);
    if (VIt != PoolDescMap.end()) {  // We already have a value for this load?
      LI.replaceAllUsesWith(VIt->second);   // Make the current load dead
      ++Eliminated;
    } else {
      // This load might not be a load of a pool pointer, check to see if it is
      if (LI.getNumOperands() == 4 &&  // load pool, uint 0, ubyte 0, ubyte 0
          find(PoolDescValues.begin(), PoolDescValues.end(), LoadAddr) !=
          PoolDescValues.end()) {

        assert("Make sure it's a load of the pool base, not a chaining field" &&
               LI.getOperand(1) == Constant::getNullValue(Type::UIntTy) &&
               LI.getOperand(2) == Constant::getNullValue(Type::UByteTy) &&
               LI.getOperand(3) == Constant::getNullValue(Type::UByteTy));

        // If it is a load of a pool base, keep track of it for future reference
        PoolDescMap.insert(std::make_pair(LoadAddr, &LI));
        ++Remaining;
      }
    }
  }

  // If we run across a function call, forget all state...  Calls to
  // poolalloc/poolfree can invalidate the pool base pointer, so it should be
  // reloaded the next time it is used.  Furthermore, a call to a random
  // function might call one of these functions, so be conservative.  Through
  // more analysis, this could be improved in the future.
  //
  void visitCallInst(CallInst &) {
    PoolDescMap.clear();
  }
};

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
        Function::aiterator AI = F->abegin();
        std::advance(AI, TFI.ArgInfo[i].ArgNo);
        addNodeMapping(TFI.ArgInfo[i].Node, CalledGraph.getValueMap()[AI],
                       NodeMapping);
      }
      LastArgNo = TFI.ArgInfo[i].ArgNo;
    }
  }
}




// addCallInfo - For a specified function call CI, figure out which pool
// descriptors need to be passed in as arguments, and which arguments need to be
// transformed into indices.  If Arg != -1, the specified call argument is
// passed in as a pointer to a data structure.
//
void TransformFunctionInfo::addCallInfo(DataStructure *DS, CallInst *CI,
                                        int Arg, DSNode *GraphNode,
                                        map<DSNode*, PoolInfo> &PoolDescs) {
  assert(CI->getCalledFunction() && "Cannot handle indirect calls yet!");
  assert(Func == 0 || Func == CI->getCalledFunction() &&
         "Function call record should always call the same function!");
  assert(Call == 0 || Call == CI &&
         "Call element already filled in with different value!");
  Func = CI->getCalledFunction();
  Call = CI;
  //FunctionDSGraph &CalledGraph = DS->getClosedDSGraph(Func);

  // For now, add the entire graph that is pointed to by the call argument.
  // This graph can and should be pruned to only what the function itself will
  // use, because often this will be a dramatically smaller subset of what we
  // are providing.
  //
  // FIXME: This should use pool links instead of extra arguments!
  //
  for (df_iterator<DSNode*> I = df_begin(GraphNode), E = df_end(GraphNode);
       I != E; ++I)
    ArgInfo.push_back(CallArgInfo(Arg, *I, PoolDescs[*I].Handle));
}

static void markReachableNodes(const PointerValSet &Vals,
                               set<DSNode*> &ReachableNodes) {
  for (unsigned n = 0, ne = Vals.size(); n != ne; ++n) {
    DSNode *N = Vals[n].Node;
    if (ReachableNodes.count(N) == 0)   // Haven't already processed node?
      ReachableNodes.insert(df_begin(N), df_end(N)); // Insert all
  }
}

// Make sure that all dependant arguments are added to this transformation info.
// For example, if we call foo(null, P) and foo treats it's first and second
// arguments as belonging to the same data structure, the we MUST add entries to
// know that the null needs to be transformed into an index as well.
//
void TransformFunctionInfo::ensureDependantArgumentsIncluded(DataStructure *DS,
                                           map<DSNode*, PoolInfo> &PoolDescs) {
  // FIXME: This does not work for indirect function calls!!!
  if (Func == 0) return;  // FIXME!

  // Make sure argument entries are sorted.
  finalizeConstruction();

  // Loop over the function signature, checking to see if there are any pointer
  // arguments that we do not convert...  if there is something we haven't
  // converted, set done to false.
  //
  unsigned PtrNo = 0;
  bool Done = true;
  if (isa<PointerType>(Func->getReturnType()))    // Make sure we convert retval
    if (PtrNo < ArgInfo.size() && ArgInfo[PtrNo++].ArgNo == -1) {
      // We DO transform the ret val... skip all possible entries for retval
      while (PtrNo < ArgInfo.size() && ArgInfo[PtrNo].ArgNo == -1)
        PtrNo++;
    } else {
      Done = false;
    }

  unsigned i = 0;
  for (Function::aiterator I = Func->abegin(), E = Func->aend(); I!=E; ++I,++i){
    if (isa<PointerType>(I->getType())) {
      if (PtrNo < ArgInfo.size() && ArgInfo[PtrNo++].ArgNo == (int)i) {
        // We DO transform this arg... skip all possible entries for argument
        while (PtrNo < ArgInfo.size() && ArgInfo[PtrNo].ArgNo == (int)i)
          PtrNo++;
      } else {
        Done = false;
        break;
      }
    }
  }

  // If we already have entries for all pointer arguments and retvals, there
  // certainly is no work to do.  Bail out early to avoid building relatively
  // expensive data structures.
  //
  if (Done) return;

#ifdef DEBUG_TRANSFORM_PROGRESS
  cerr << "Must ensure dependant arguments for: " << Func->getName() << "\n";
#endif

  // Otherwise, we MIGHT have to add the arguments/retval if they are part of
  // the same datastructure graph as some other argument or retval that we ARE
  // processing.
  //
  // Get the data structure graph for the called function.
  //
  FunctionDSGraph &CalledDS = DS->getClosedDSGraph(Func);

  // Build a mapping between the nodes in our current graph and the nodes in the
  // called function's graph.  We build it based on our _incomplete_
  // transformation information, because it contains all of the info that we
  // should need.
  //
  map<DSNode*, PointerValSet> NodeMapping;
  CalculateNodeMapping(Func, *this,
                       DS->getClosedDSGraph(Call->getParent()->getParent()),
                       CalledDS, NodeMapping);

  // Build the inverted version of the node mapping, that maps from a node in
  // the called functions graph to a single node in the caller graph.
  // 
  map<DSNode*, DSNode*> InverseNodeMap;
  for (map<DSNode*, PointerValSet>::iterator I = NodeMapping.begin(),
         E = NodeMapping.end(); I != E; ++I) {
    PointerValSet &CalledNodes = I->second;
    for (unsigned i = 0, e = CalledNodes.size(); i != e; ++i)
      InverseNodeMap[CalledNodes[i].Node] = I->first;
  }
  NodeMapping.clear();  // Done with information, free memory
  
  // Build a set of reachable nodes from the arguments/retval that we ARE
  // passing in...
  set<DSNode*> ReachableNodes;

  // Loop through all of the arguments, marking all of the reachable data
  // structure nodes reachable if they are from this pointer...
  //
  for (unsigned i = 0, e = ArgInfo.size(); i != e; ++i) {
    if (ArgInfo[i].ArgNo == -1) {
      if (i == 0)   // Only process retvals once (performance opt)
        markReachableNodes(CalledDS.getRetNodes(), ReachableNodes);
    } else {  // If it's an argument value...
      Function::aiterator AI = Func->abegin();
      std::advance(AI, ArgInfo[i].ArgNo);
      if (isa<PointerType>(AI->getType()))
        markReachableNodes(CalledDS.getValueMap()[AI], ReachableNodes);
    }
  }

  // Now that we know which nodes are already reachable, see if any of the
  // arguments that we are not passing values in for can reach one of the
  // existing nodes...
  //

  // <FIXME> IN THEORY, we should allow arbitrary paths from the argument to
  // nodes we know about.  The problem is that if we do this, then I don't know
  // how to get pool pointers for this head list.  Since we are completely
  // deadline driven, I'll just allow direct accesses to the graph. </FIXME>
  //
  
  PtrNo = 0;
  if (isa<PointerType>(Func->getReturnType()))    // Make sure we convert retval
    if (PtrNo < ArgInfo.size() && ArgInfo[PtrNo++].ArgNo == -1) {
      // We DO transform the ret val... skip all possible entries for retval
      while (PtrNo < ArgInfo.size() && ArgInfo[PtrNo].ArgNo == -1)
        PtrNo++;
    } else {
      // See what the return value points to...

      // FIXME: This should generalize to any number of nodes, just see if any
      // are reachable.
      assert(CalledDS.getRetNodes().size() == 1 &&
             "Assumes only one node is returned");
      DSNode *N = CalledDS.getRetNodes()[0].Node;
      
      // If the return value is not marked as being passed in, but it NEEDS to
      // be transformed, then make it known now.
      //
      if (ReachableNodes.count(N)) {
#ifdef DEBUG_TRANSFORM_PROGRESS
        cerr << "ensure dependant arguments adds return value entry!\n";
#endif
        addCallInfo(DS, Call, -1, InverseNodeMap[N], PoolDescs);

        // Keep sorted!
        finalizeConstruction();
      }
    }

  i = 0;
  for (Function::aiterator I = Func->abegin(), E = Func->aend(); I!=E; ++I, ++i)
    if (isa<PointerType>(I->getType())) {
      if (PtrNo < ArgInfo.size() && ArgInfo[PtrNo++].ArgNo == (int)i) {
        // We DO transform this arg... skip all possible entries for argument
        while (PtrNo < ArgInfo.size() && ArgInfo[PtrNo].ArgNo == (int)i)
          PtrNo++;
      } else {
        // This should generalize to any number of nodes, just see if any are
        // reachable.
        assert(CalledDS.getValueMap()[I].size() == 1 &&
               "Only handle case where pointing to one node so far!");

        // If the arg is not marked as being passed in, but it NEEDS to
        // be transformed, then make it known now.
        //
        DSNode *N = CalledDS.getValueMap()[I][0].Node;
        if (ReachableNodes.count(N)) {
#ifdef DEBUG_TRANSFORM_PROGRESS
          cerr << "ensure dependant arguments adds for arg #" << i << "\n";
#endif
          addCallInfo(DS, Call, i, InverseNodeMap[N], PoolDescs);

          // Keep sorted!
          finalizeConstruction();
        }
      }
    }
}


// transformFunctionBody - This transforms the instruction in 'F' to use the
// pools specified in PoolDescs when modifying data structure nodes specified in
// the PoolDescs map.  Specifically, scalar values specified in the Scalars
// vector must be remapped.  IPFGraph is the closed data structure graph for F,
// of which the PoolDescriptor nodes come from.
//
void PoolAllocate::transformFunctionBody(Function *F, FunctionDSGraph &IPFGraph,
                                         map<DSNode*, PoolInfo> &PoolDescs) {

  // Loop through the value map looking for scalars that refer to nonescaping
  // allocations.  Add them to the Scalars vector.  Note that we may have
  // multiple entries in the Scalars vector for each value if it points to more
  // than one object.
  //
  map<Value*, PointerValSet> &ValMap = IPFGraph.getValueMap();
  vector<ScalarInfo> Scalars;

#ifdef DEBUG_TRANSFORM_PROGRESS
  cerr << "Building scalar map for fn '" << F->getName() << "' body:\n";
#endif

  for (map<Value*, PointerValSet>::iterator I = ValMap.begin(),
         E = ValMap.end(); I != E; ++I) {
    const PointerValSet &PVS = I->second;  // Set of things pointed to by scalar

    // Check to see if the scalar points to a data structure node...
    for (unsigned i = 0, e = PVS.size(); i != e; ++i) {
      if (PVS[i].Index) { cerr << "Problem in " << F->getName() << " for " << I->first << "\n"; }
      assert(PVS[i].Index == 0 && "Nonzero not handled yet!");
        
      // If the allocation is in the nonescaping set...
      map<DSNode*, PoolInfo>::iterator AI = PoolDescs.find(PVS[i].Node);
      if (AI != PoolDescs.end()) {              // Add it to the list of scalars
        Scalars.push_back(ScalarInfo(I->first, AI->second));
#ifdef DEBUG_TRANSFORM_PROGRESS
        cerr << "\nScalar Mapping from:" << I->first
             << "Scalar Mapping to: "; PVS.print(cerr);
#endif
      }
    }
  }

#ifdef DEBUG_TRANSFORM_PROGRESS
  cerr << "\nIn '" << F->getName()
       << "': Found the following values that point to poolable nodes:\n";

  for (unsigned i = 0, e = Scalars.size(); i != e; ++i)
    cerr << Scalars[i].Val;
  cerr << "\n";
#endif

  // CallMap - Contain an entry for every call instruction that needs to be
  // transformed.  Each entry in the map contains information about what we need
  // to do to each call site to change it to work.
  //
  map<CallInst*, TransformFunctionInfo> CallMap;

  // Now we need to figure out what called functions we need to transform, and
  // how.  To do this, we look at all of the scalars, seeing which functions are
  // either used as a scalar value (so they return a data structure), or are
  // passed one of our scalar values.
  //
  for (unsigned i = 0, e = Scalars.size(); i != e; ++i) {
    Value *ScalarVal = Scalars[i].Val;

    // Check to see if the scalar _IS_ a call...
    if (CallInst *CI = dyn_cast<CallInst>(ScalarVal))
      // If so, add information about the pool it will be returning...
      CallMap[CI].addCallInfo(DS, CI, -1, Scalars[i].Pool.Node, PoolDescs);

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
        CallMap[CI].addCallInfo(DS, CI, OI-CI->op_begin()-1,
                                Scalars[i].Pool.Node, PoolDescs);
      }
    }
  }

  // Make sure that all dependant arguments are added as well.  For example, if
  // we call foo(null, P) and foo treats it's first and second arguments as
  // belonging to the same data structure, the we MUST set up the CallMap to
  // know that the null needs to be transformed into an index as well.
  //
  for (map<CallInst*, TransformFunctionInfo>::iterator I = CallMap.begin();
       I != CallMap.end(); ++I)
    I->second.ensureDependantArgumentsIncluded(DS, PoolDescs);

#ifdef DEBUG_TRANSFORM_PROGRESS
  // Print out call map...
  for (map<CallInst*, TransformFunctionInfo>::iterator I = CallMap.begin();
       I != CallMap.end(); ++I) {
    cerr << "For call: " << I->first;
    cerr << I->second.Func->getName() << " must pass pool pointer for args #";
    for (unsigned i = 0; i < I->second.ArgInfo.size(); ++i)
      cerr << I->second.ArgInfo[i].ArgNo << ", ";
    cerr << "\n\n";
  }
#endif

  // Loop through all of the call nodes, recursively creating the new functions
  // that we want to call...  This uses a map to prevent infinite recursion and
  // to avoid duplicating functions unneccesarily.
  //
  for (map<CallInst*, TransformFunctionInfo>::iterator I = CallMap.begin(),
         E = CallMap.end(); I != E; ++I) {
    // Transform all of the functions we need, or at least ensure there is a
    // cached version available.
    transformFunction(I->second, IPFGraph, PoolDescs);
  }

  // Now that all of the functions that we want to call are available, transform
  // the local function so that it uses the pools locally and passes them to the
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
      InstToFix.push_back(cast<Instruction>(*UI));
  }

  // Make sure that we get return instructions that return a null value from the
  // function...
  //
  if (!IPFGraph.getRetNodes().empty()) {
    assert(IPFGraph.getRetNodes().size() == 1 && "Can only return one node?");
    PointerVal RetNode = IPFGraph.getRetNodes()[0];
    assert(RetNode.Index == 0 && "Subindexing not implemented yet!");

    // Only process return instructions if the return value of this function is
    // part of one of the data structures we are transforming...
    //
    if (PoolDescs.count(RetNode.Node)) {
      // Loop over all of the basic blocks, adding return instructions...
      for (Function::iterator I = F->begin(), E = F->end(); I != E; ++I)
        if (ReturnInst *RI = dyn_cast<ReturnInst>(I->getTerminator()))
          InstToFix.push_back(RI);
    }
  }



  // Eliminate duplicates by sorting, then removing equal neighbors.
  sort(InstToFix.begin(), InstToFix.end());
  InstToFix.erase(unique(InstToFix.begin(), InstToFix.end()), InstToFix.end());

  // Loop over all of the instructions to transform, creating the new
  // replacement instructions for them.  This also unlinks them from the
  // function so they can be safely deleted later.
  //
  map<Value*, Value*> XFormMap;  
  NewInstructionCreator NIC(*this, Scalars, CallMap, XFormMap);

  // Visit all instructions... creating the new instructions that we need and
  // unlinking the old instructions from the function...
  //
#ifdef DEBUG_TRANSFORM_PROGRESS
  for (unsigned i = 0, e = InstToFix.size(); i != e; ++i) {
    cerr << "Fixing: " << InstToFix[i];
    NIC.visit(*InstToFix[i]);
  }
#else
  NIC.visit(InstToFix.begin(), InstToFix.end());
#endif

  // Make all instructions we will delete "let go" of their operands... so that
  // we can safely delete Arguments whose types have changed...
  //
  for_each(InstToFix.begin(), InstToFix.end(),
           std::mem_fun(&Instruction::dropAllReferences));

  // Loop through all of the pointer arguments coming into the function,
  // replacing them with arguments of POINTERTYPE to match the function type of
  // the function.
  //
  FunctionType::ParamTypes::const_iterator TI =
    F->getFunctionType()->getParamTypes().begin();
  for (Function::aiterator I = F->abegin(), E = F->aend(); I != E; ++I, ++TI) {
    if (I->getType() != *TI) {
      assert(isa<PointerType>(I->getType()) && *TI == POINTERTYPE);
      Argument *NewArg = new Argument(*TI, I->getName());
      XFormMap[I] = NewArg;  // Map old arg into new arg...

      // Replace the old argument and then delete it...
      I = F->getArgumentList().erase(I);
      I = F->getArgumentList().insert(I, NewArg);
    }
  }

  // Now that all of the new instructions have been created, we can update all
  // of the references to dummy values to be references to the actual values
  // that are computed.
  //
  NIC.updateReferences();

#ifdef DEBUG_TRANSFORM_PROGRESS
  cerr << "TRANSFORMED FUNCTION:\n" << F;
#endif

  // Delete all of the "instructions to fix"
  for_each(InstToFix.begin(), InstToFix.end(), deleter<Instruction>);

  // Eliminate pool base loads that we can easily prove are redundant
  if (!DisableRLE)
    PoolBaseLoadEliminator(PoolDescs).visit(F);

  // Since we have liberally hacked the function to pieces, we want to inform
  // the datastructure pass that its internal representation is out of date.
  //
  DS->invalidateFunction(F);
}



// transformFunction - Transform the specified function the specified way.  It
// we have already transformed that function that way, don't do anything.  The
// nodes in the TransformFunctionInfo come out of callers data structure graph.
//
void PoolAllocate::transformFunction(TransformFunctionInfo &TFI,
                                     FunctionDSGraph &CallerIPGraph,
                                     map<DSNode*, PoolInfo> &CallerPoolDesc) {
  if (getTransformedFunction(TFI)) return;  // Function xformation already done?

#ifdef DEBUG_TRANSFORM_PROGRESS
  cerr << "********** Entering transformFunction for "
       << TFI.Func->getName() << ":\n";
  for (unsigned i = 0, e = TFI.ArgInfo.size(); i != e; ++i)
    cerr << "  ArgInfo[" << i << "] = " << TFI.ArgInfo[i].ArgNo << "\n";
  cerr << "\n";
#endif

  const FunctionType *OldFuncType = TFI.Func->getFunctionType();

  assert(!OldFuncType->isVarArg() && "Vararg functions not handled yet!");

  // Build the type for the new function that we are transforming
  vector<const Type*> ArgTys;
  ArgTys.reserve(OldFuncType->getNumParams()+TFI.ArgInfo.size());
  for (unsigned i = 0, e = OldFuncType->getNumParams(); i != e; ++i)
    ArgTys.push_back(OldFuncType->getParamType(i));

  const Type *RetType = OldFuncType->getReturnType();
  
  // Add one pool pointer for every argument that needs to be supplemented.
  for (unsigned i = 0, e = TFI.ArgInfo.size(); i != e; ++i) {
    if (TFI.ArgInfo[i].ArgNo == -1)
      RetType = POINTERTYPE;  // Return a pointer
    else
      ArgTys[TFI.ArgInfo[i].ArgNo] = POINTERTYPE; // Pass a pointer
    ArgTys.push_back(PointerType::get(CallerPoolDesc.find(TFI.ArgInfo[i].Node)
                                        ->second.PoolType));
  }

  // Build the new function type...
  const FunctionType *NewFuncType = FunctionType::get(RetType, ArgTys,
                                                      OldFuncType->isVarArg());

  // The new function is internal, because we know that only we can call it.
  // This also helps subsequent IP transformations to eliminate duplicated pool
  // pointers (which look like the same value is always passed into a parameter,
  // allowing it to be easily eliminated).
  //
  Function *NewFunc = new Function(NewFuncType, true,
                                   TFI.Func->getName()+".poolxform");
  CurModule->getFunctionList().push_back(NewFunc);


#ifdef DEBUG_TRANSFORM_PROGRESS
  cerr << "Created function prototype: " << NewFunc << "\n";
#endif

  // Add the newly formed function to the TransformedFunctions table so that
  // infinite recursion does not occur!
  //
  TransformedFunctions[TFI] = NewFunc;

  // Add arguments to the function... starting with all of the old arguments
  vector<Value*> ArgMap;
  for (Function::const_aiterator I = TFI.Func->abegin(), E = TFI.Func->aend();
       I != E; ++I) {
    Argument *NFA = new Argument(I->getType(), I->getName());
    NewFunc->getArgumentList().push_back(NFA);
    ArgMap.push_back(NFA);  // Keep track of the arguments 
  }

  // Now add all of the arguments corresponding to pools passed in...
  for (unsigned i = 0, e = TFI.ArgInfo.size(); i != e; ++i) {
    CallArgInfo &AI = TFI.ArgInfo[i];
    string Name;
    if (AI.ArgNo == -1)
      Name = "ret";
    else
      Name = ArgMap[AI.ArgNo]->getName();  // Get the arg name
    const Type *Ty = PointerType::get(CallerPoolDesc[AI.Node].PoolType);
    Argument *NFA = new Argument(Ty, Name+".pool");
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

  // NodeMapping - Multimap from callers graph to called graph.  We are
  // guaranteed that the called function graph has more nodes than the caller,
  // or exactly the same number of nodes.  This is because the called function
  // might not know that two nodes are merged when considering the callers
  // context, but the caller obviously does.  Because of this, a single node in
  // the calling function's data structure graph can map to multiple nodes in
  // the called functions graph.
  //
  map<DSNode*, PointerValSet> NodeMapping;

  CalculateNodeMapping(NewFunc, TFI, CallerIPGraph, DSGraph, 
                       NodeMapping);

  // Print out the node mapping...
#ifdef DEBUG_TRANSFORM_PROGRESS
  cerr << "\nNode mapping for call of " << NewFunc->getName() << "\n";
  for (map<DSNode*, PointerValSet>::iterator I = NodeMapping.begin();
       I != NodeMapping.end(); ++I) {
    cerr << "Map: "; I->first->print(cerr);
    cerr << "To:  "; I->second.print(cerr);
    cerr << "\n";
  }
#endif

  // Fill in the PoolDescriptor information for the transformed function so that
  // it can determine which value holds the pool descriptor for each data
  // structure node that it accesses.
  //
  map<DSNode*, PoolInfo> PoolDescs;

#ifdef DEBUG_TRANSFORM_PROGRESS
  cerr << "\nCalculating the pool descriptor map:\n";
#endif

  // Calculate as much of the pool descriptor map as possible.  Since we have
  // the node mapping between the caller and callee functions, and we have the
  // pool descriptor information of the caller, we can calculate a partical pool
  // descriptor map for the called function.
  //
  // The nodes that we do not have complete information for are the ones that
  // are accessed by loading pointers derived from arguments passed in, but that
  // are not passed in directly.  In this case, we have all of the information
  // except a pool value.  If the called function refers to this pool, the pool
  // value will be loaded from the pool graph and added to the map as neccesary.
  //
  for (map<DSNode*, PointerValSet>::iterator I = NodeMapping.begin();
       I != NodeMapping.end(); ++I) {
    DSNode *CallerNode = I->first;
    PoolInfo &CallerPI = CallerPoolDesc[CallerNode];

    // Check to see if we have a node pointer passed in for this value...
    Value *CalleeValue = 0;
    for (unsigned a = 0, ae = TFI.ArgInfo.size(); a != ae; ++a)
      if (TFI.ArgInfo[a].Node == CallerNode) {
        // Calculate the argument number that the pool is to the function
        // call...  The call instruction should not have the pool operands added
        // yet.
        unsigned ArgNo = TFI.Call->getNumOperands()-1+a;
#ifdef DEBUG_TRANSFORM_PROGRESS
        cerr << "Should be argument #: " << ArgNo << "[i = " << a << "]\n";
#endif
        assert(ArgNo < NewFunc->asize() &&
               "Call already has pool arguments added??");

        // Map the pool argument into the called function...
        Function::aiterator AI = NewFunc->abegin();
        std::advance(AI, ArgNo);
        CalleeValue = AI;
        break;  // Found value, quit loop
      }

    // Loop over all of the data structure nodes that this incoming node maps to
    // Creating a PoolInfo structure for them.
    for (unsigned i = 0, e = I->second.size(); i != e; ++i) {
      assert(I->second[i].Index == 0 && "Doesn't handle subindexing yet!");
      DSNode *CalleeNode = I->second[i].Node;
     
      // Add the descriptor.  We already know everything about it by now, much
      // of it is the same as the caller info.
      // 
      PoolDescs.insert(std::make_pair(CalleeNode,
                                 PoolInfo(CalleeNode, CalleeValue,
                                          CallerPI.NewType,
                                          CallerPI.PoolType)));
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
  transformFunctionBody(NewFunc, DSGraph, PoolDescs);
  
#ifdef DEBUG_TRANSFORM_PROGRESS
  cerr << "Function after transformation:\n" << NewFunc;
#endif
}

static unsigned countPointerTypes(const Type *Ty) {
  if (isa<PointerType>(Ty)) {
    return 1;
  } else if (const StructType *STy = dyn_cast<StructType>(Ty)) {
    unsigned Num = 0;
    for (unsigned i = 0, e = STy->getElementTypes().size(); i != e; ++i)
      Num += countPointerTypes(STy->getElementTypes()[i]);
    return Num;
  } else if (const ArrayType *ATy = dyn_cast<ArrayType>(Ty)) {
    return countPointerTypes(ATy->getElementType());
  } else {
    assert(Ty->isPrimitiveType() && "Unknown derived type!");
    return 0;
  }
}

// CreatePools - Insert instructions into the function we are processing to
// create all of the memory pool objects themselves.  This also inserts
// destruction code.  Add an alloca for each pool that is allocated to the
// PoolDescs vector.
//
void PoolAllocate::CreatePools(Function *F, const vector<AllocDSNode*> &Allocs,
                               map<DSNode*, PoolInfo> &PoolDescs) {
  // Find all of the return nodes in the function...
  vector<BasicBlock*> ReturnNodes;
  for (Function::iterator I = F->begin(), E = F->end(); I != E; ++I)
    if (isa<ReturnInst>(I->getTerminator()))
      ReturnNodes.push_back(I);

#ifdef DEBUG_CREATE_POOLS
  cerr << "Allocs that we are pool allocating:\n";
  for (unsigned i = 0, e = Allocs.size(); i != e; ++i)
    Allocs[i]->dump();
#endif

  map<DSNode*, PATypeHolder> AbsPoolTyMap;

  // First pass over the allocations to process...
  for (unsigned i = 0, e = Allocs.size(); i != e; ++i) {
    // Create the pooldescriptor mapping... with null entries for everything
    // except the node & NewType fields.
    //
    map<DSNode*, PoolInfo>::iterator PI =
      PoolDescs.insert(std::make_pair(Allocs[i], PoolInfo(Allocs[i]))).first;

    // Add a symbol table entry for the new type if there was one for the old
    // type...
    string OldName = CurModule->getTypeName(Allocs[i]->getType());
    if (OldName.empty()) OldName = "node";
    CurModule->addTypeName(OldName+".p", PI->second.NewType);

    // Create the abstract pool types that will need to be resolved in a second
    // pass once an abstract type is created for each pool.
    //
    // Can only handle limited shapes for now...
    const Type *OldNodeTy = Allocs[i]->getType();
    vector<const Type*> PoolTypes;

    // Pool type is the first element of the pool descriptor type...
    PoolTypes.push_back(getPoolType(PoolDescs[Allocs[i]].NewType));

    unsigned NumPointers = countPointerTypes(OldNodeTy);
    while (NumPointers--)   // Add a different opaque type for each pointer
      PoolTypes.push_back(OpaqueType::get());

    assert(Allocs[i]->getNumLinks() == PoolTypes.size()-1 &&
           "Node should have same number of pointers as pool!");

    StructType *PoolType = StructType::get(PoolTypes);

    // Add a symbol table entry for the pooltype if possible...
    CurModule->addTypeName(OldName+".pool", PoolType);

    // Create the pool type, with opaque values for pointers...
    AbsPoolTyMap.insert(std::make_pair(Allocs[i], PoolType));
#ifdef DEBUG_CREATE_POOLS
    cerr << "POOL TY: " << AbsPoolTyMap.find(Allocs[i])->second.get() << "\n";
#endif
  }
  
  // Now that we have types for all of the pool types, link them all together.
  for (unsigned i = 0, e = Allocs.size(); i != e; ++i) {
    PATypeHolder &PoolTyH = AbsPoolTyMap.find(Allocs[i])->second;

    // Resolve all of the outgoing pointer types of this pool node...
    for (unsigned p = 0, pe = Allocs[i]->getNumLinks(); p != pe; ++p) {
      PointerValSet &PVS = Allocs[i]->getLink(p);
      assert(!PVS.empty() && "Outgoing edge is empty, field unused, can"
             " probably just leave the type opaque or something dumb.");
      unsigned Out;
      for (Out = 0; AbsPoolTyMap.count(PVS[Out].Node) == 0; ++Out)
        assert(Out != PVS.size() && "No edge to an outgoing allocation node!?");
      
      assert(PVS[Out].Index == 0 && "Subindexing not implemented yet!");

      // The actual struct type could change each time through the loop, so it's
      // NOT loop invariant.
      const StructType *PoolTy = cast<StructType>(PoolTyH.get());

      // Get the opaque type...
      DerivedType *ElTy = (DerivedType*)(PoolTy->getElementTypes()[p+1].get());

#ifdef DEBUG_CREATE_POOLS
      cerr << "Refining " << ElTy << " of " << PoolTy << " to "
           << AbsPoolTyMap.find(PVS[Out].Node)->second.get() << "\n";
#endif

      const Type *RefPoolTy = AbsPoolTyMap.find(PVS[Out].Node)->second.get();
      ElTy->refineAbstractTypeTo(PointerType::get(RefPoolTy));

#ifdef DEBUG_CREATE_POOLS
      cerr << "Result pool type is: " << PoolTyH.get() << "\n";
#endif
    }
  }

  // Create the code that goes in the entry and exit nodes for the function...
  vector<Instruction*> EntryNodeInsts;
  for (unsigned i = 0, e = Allocs.size(); i != e; ++i) {
    PoolInfo &PI = PoolDescs[Allocs[i]];
    
    // Fill in the pool type for this pool...
    PI.PoolType = AbsPoolTyMap.find(Allocs[i])->second.get();
    assert(!PI.PoolType->isAbstract() &&
           "Pool type should not be abstract anymore!");

    // Add an allocation and a free for each pool...
    AllocaInst *PoolAlloc = new AllocaInst(PI.PoolType, 0,
                                           CurModule->getTypeName(PI.PoolType));
    PI.Handle = PoolAlloc;
    EntryNodeInsts.push_back(PoolAlloc);
    AllocationInst *AI = Allocs[i]->getAllocation();

    // Initialize the pool.  We need to know how big each allocation is.  For
    // our purposes here, we assume we are allocating a scalar, or array of
    // constant size.
    //
    unsigned ElSize = TargetData.getTypeSize(PI.NewType);

    vector<Value*> Args;
    Args.push_back(ConstantUInt::get(Type::UIntTy, ElSize));
    Args.push_back(PoolAlloc);    // Pool to initialize
    EntryNodeInsts.push_back(new CallInst(PoolInit, Args));

    // Add code to destroy the pool in all of the exit nodes of the function...
    Args.clear();
    Args.push_back(PoolAlloc);    // Pool to initialize
    
    for (unsigned EN = 0, ENE = ReturnNodes.size(); EN != ENE; ++EN) {
      Instruction *Destroy = new CallInst(PoolDestroy, Args);

      // Insert it before the return instruction...
      BasicBlock *RetNode = ReturnNodes[EN];
      RetNode->getInstList().insert(RetNode->end()--, Destroy);
    }
  }

  // Now that all of the pool descriptors have been created, link them together
  // so that called functions can get links as neccesary...
  //
  for (unsigned i = 0, e = Allocs.size(); i != e; ++i) {
    PoolInfo &PI = PoolDescs[Allocs[i]];

    // For every pointer in the data structure, initialize a link that
    // indicates which pool to access...
    //
    vector<Value*> Indices(2);
    Indices[0] = ConstantUInt::get(Type::UIntTy, 0);
    for (unsigned l = 0, le = PI.Node->getNumLinks(); l != le; ++l)
      // Only store an entry for the field if the field is used!
      if (!PI.Node->getLink(l).empty()) {
        assert(PI.Node->getLink(l).size() == 1 && "Should have only one link!");
        PointerVal PV = PI.Node->getLink(l)[0];
        assert(PV.Index == 0 && "Subindexing not supported yet!");
        PoolInfo &LinkedPool = PoolDescs[PV.Node];
        Indices[1] = ConstantUInt::get(Type::UByteTy, 1+l);
      
        EntryNodeInsts.push_back(new StoreInst(LinkedPool.Handle, PI.Handle,
                                               Indices));
      }
  }

  // Insert the entry node code into the entry block...
  F->getEntryNode().getInstList().insert(++F->getEntryNode().begin(),
                                          EntryNodeInsts.begin(),
                                          EntryNodeInsts.end());
}


// addPoolPrototypes - Add prototypes for the pool functions to the specified
// module and update the Pool* instance variables to point to them.
//
void PoolAllocate::addPoolPrototypes(Module &M) {
  // Get poolinit function...
  vector<const Type*> Args;
  Args.push_back(Type::UIntTy);     // Num bytes per element
  FunctionType *PoolInitTy = FunctionType::get(Type::VoidTy, Args, true);
  PoolInit = M.getOrInsertFunction("poolinit", PoolInitTy);

  // Get pooldestroy function...
  Args.pop_back();  // Only takes a pool...
  FunctionType *PoolDestroyTy = FunctionType::get(Type::VoidTy, Args, true);
  PoolDestroy = M.getOrInsertFunction("pooldestroy", PoolDestroyTy);

  // Get the poolalloc function...
  FunctionType *PoolAllocTy = FunctionType::get(POINTERTYPE, Args, true);
  PoolAlloc = M.getOrInsertFunction("poolalloc", PoolAllocTy);

  // Get the poolfree function...
  Args.push_back(POINTERTYPE);       // Pointer to free
  FunctionType *PoolFreeTy = FunctionType::get(Type::VoidTy, Args, true);
  PoolFree = M.getOrInsertFunction("poolfree", PoolFreeTy);

  Args[0] = Type::UIntTy;            // Number of slots to allocate
  FunctionType *PoolAllocArrayTy = FunctionType::get(POINTERTYPE, Args, true);
  PoolAllocArray = M.getOrInsertFunction("poolallocarray", PoolAllocArrayTy);
}


bool PoolAllocate::run(Module &M) {
  addPoolPrototypes(M);
  CurModule = &M;
  
  DS = &getAnalysis<DataStructure>();
  bool Changed = false;

  for (Module::iterator I = M.begin(); I != M.end(); ++I)
    if (!I->isExternal()) {
      Changed |= processFunction(I);
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
Pass *createPoolAllocatePass() { 
  assert(0 && "Pool allocator disabled!");
  return 0;
  //return new PoolAllocate(); 
}
#endif
