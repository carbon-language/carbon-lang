//===- Local.cpp - Compute a local data structure graph for a function ----===//
//
// Compute the local version of the data structure graph for a function.  The
// external interface to this file is the DSGraph constructor.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/DataStructure.h"
#include "llvm/Analysis/DSGraph.h"
#include "llvm/iMemory.h"
#include "llvm/iTerminators.h"
#include "llvm/iPHINode.h"
#include "llvm/iOther.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Support/InstVisitor.h"
#include "llvm/Target/TargetData.h"
#include "Support/Statistic.h"
#include "Support/Timer.h"
#include "Support/CommandLine.h"

// FIXME: This should eventually be a FunctionPass that is automatically
// aggregated into a Pass.
//
#include "llvm/Module.h"

static RegisterAnalysis<LocalDataStructures>
X("datastructure", "Local Data Structure Analysis");

namespace DS {
  // FIXME: Do something smarter with target data!
  TargetData TD("temp-td");

  // isPointerType - Return true if this type is big enough to hold a pointer.
  bool isPointerType(const Type *Ty) {
    if (isa<PointerType>(Ty))
      return true;
    else if (Ty->isPrimitiveType() && Ty->isInteger())
      return Ty->getPrimitiveSize() >= PointerSize;
    return false;
  }
}
using namespace DS;


namespace {
  cl::opt<bool>
  DisableDirectCallOpt("disable-direct-call-dsopt", cl::Hidden,
                       cl::desc("Disable direct call optimization in "
                                "DSGraph construction"));
  cl::opt<bool>
  DisableFieldSensitivity("disable-ds-field-sensitivity", cl::Hidden,
                          cl::desc("Disable field sensitivity in DSGraphs"));

  //===--------------------------------------------------------------------===//
  //  GraphBuilder Class
  //===--------------------------------------------------------------------===//
  //
  /// This class is the builder class that constructs the local data structure
  /// graph by performing a single pass over the function in question.
  ///
  class GraphBuilder : InstVisitor<GraphBuilder> {
    DSGraph &G;
    std::vector<DSNode*> &Nodes;
    DSNodeHandle &RetNode;               // Node that gets returned...
    hash_map<Value*, DSNodeHandle> &ScalarMap;
    std::vector<DSCallSite> &FunctionCalls;

  public:
    GraphBuilder(DSGraph &g, std::vector<DSNode*> &nodes, DSNodeHandle &retNode,
                 hash_map<Value*, DSNodeHandle> &SM,
                 std::vector<DSCallSite> &fc)
      : G(g), Nodes(nodes), RetNode(retNode), ScalarMap(SM), FunctionCalls(fc) {

      // Create scalar nodes for all pointer arguments...
      for (Function::aiterator I = G.getFunction().abegin(),
             E = G.getFunction().aend(); I != E; ++I)
        if (isPointerType(I->getType()))
          getValueDest(*I);

      visit(G.getFunction());  // Single pass over the function
    }

  private:
    // Visitor functions, used to handle each instruction type we encounter...
    friend class InstVisitor<GraphBuilder>;
    void visitMallocInst(MallocInst &MI) { handleAlloc(MI, DSNode::HeapNode); }
    void visitAllocaInst(AllocaInst &AI) { handleAlloc(AI, DSNode::AllocaNode);}
    void handleAlloc(AllocationInst &AI, DSNode::NodeTy NT);

    void visitPHINode(PHINode &PN);

    void visitGetElementPtrInst(User &GEP);
    void visitReturnInst(ReturnInst &RI);
    void visitLoadInst(LoadInst &LI);
    void visitStoreInst(StoreInst &SI);
    void visitCallInst(CallInst &CI);
    void visitSetCondInst(SetCondInst &SCI) {}  // SetEQ & friends are ignored
    void visitFreeInst(FreeInst &FI);
    void visitCastInst(CastInst &CI);
    void visitInstruction(Instruction &I);

  private:
    // Helper functions used to implement the visitation functions...

    /// createNode - Create a new DSNode, ensuring that it is properly added to
    /// the graph.
    ///
    DSNode *createNode(DSNode::NodeTy NodeType, const Type *Ty = 0) {
      DSNode *N = new DSNode(NodeType, Ty, &G);   // Create the node
      if (DisableFieldSensitivity)
        N->foldNodeCompletely();
      return N;
    }

    /// setDestTo - Set the ScalarMap entry for the specified value to point to
    /// the specified destination.  If the Value already points to a node, make
    /// sure to merge the two destinations together.
    ///
    void setDestTo(Value &V, const DSNodeHandle &NH);

    /// getValueDest - Return the DSNode that the actual value points to. 
    ///
    DSNodeHandle getValueDest(Value &V);

    /// getLink - This method is used to return the specified link in the
    /// specified node if one exists.  If a link does not already exist (it's
    /// null), then we create a new node, link it, then return it.
    ///
    DSNodeHandle &getLink(const DSNodeHandle &Node, unsigned Link = 0);
  };
}

//===----------------------------------------------------------------------===//
// DSGraph constructor - Simply use the GraphBuilder to construct the local
// graph.
DSGraph::DSGraph(Function &F, DSGraph *GG) : Func(&F), GlobalsGraph(GG) {
  PrintAuxCalls = false;
  // Use the graph builder to construct the local version of the graph
  GraphBuilder B(*this, Nodes, RetNode, ScalarMap, FunctionCalls);
#ifndef NDEBUG
  Timer::addPeakMemoryMeasurement();
#endif

  // Remove all integral constants from the scalarmap!
  for (hash_map<Value*, DSNodeHandle>::iterator I = ScalarMap.begin();
       I != ScalarMap.end();)
    if (isa<ConstantIntegral>(I->first)) {
      hash_map<Value*, DSNodeHandle>::iterator J = I++;
      ScalarMap.erase(J);
    } else
      ++I;

  markIncompleteNodes(DSGraph::MarkFormalArgs);

  // Remove any nodes made dead due to merging...
  removeDeadNodes(DSGraph::KeepUnreachableGlobals);
}


//===----------------------------------------------------------------------===//
// Helper method implementations...
//

/// getValueDest - Return the DSNode that the actual value points to.
///
DSNodeHandle GraphBuilder::getValueDest(Value &Val) {
  Value *V = &Val;
  if (V == Constant::getNullValue(V->getType()))
    return 0;  // Null doesn't point to anything, don't add to ScalarMap!

  DSNodeHandle &NH = ScalarMap[V];
  if (NH.getNode())
    return NH;     // Already have a node?  Just return it...

  // Otherwise we need to create a new node to point to.
  // Check first for constant expressions that must be traversed to
  // extract the actual value.
  if (Constant *C = dyn_cast<Constant>(V))
    if (ConstantPointerRef *CPR = dyn_cast<ConstantPointerRef>(C)) {
      return NH = getValueDest(*CPR->getValue());
    } else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
      if (CE->getOpcode() == Instruction::Cast)
        NH = getValueDest(*CE->getOperand(0));
      else if (CE->getOpcode() == Instruction::GetElementPtr) {
        visitGetElementPtrInst(*CE);
        hash_map<Value*, DSNodeHandle>::iterator I = ScalarMap.find(CE);
        assert(I != ScalarMap.end() && "GEP didn't get processed right?");
        NH = I->second;
      } else {
        // This returns a conservative unknown node for any unhandled ConstExpr
        return NH = createNode(DSNode::UnknownNode);
      }
      if (NH.getNode() == 0) {  // (getelementptr null, X) returns null
        ScalarMap.erase(V);
        return 0;
      }
      return NH;

    } else if (ConstantIntegral *CI = dyn_cast<ConstantIntegral>(C)) {
      // Random constants are unknown mem
      return NH = createNode(DSNode::UnknownNode);
    } else {
      assert(0 && "Unknown constant type!");
    }

  // Otherwise we need to create a new node to point to...
  DSNode *N;
  if (GlobalValue *GV = dyn_cast<GlobalValue>(V)) {
    // Create a new global node for this global variable...
    N = createNode(DSNode::GlobalNode, GV->getType()->getElementType());
    N->addGlobal(GV);
  } else {
    // Otherwise just create a shadow node
    N = createNode(DSNode::ShadowNode);
  }

  NH.setNode(N);      // Remember that we are pointing to it...
  NH.setOffset(0);
  return NH;
}


/// getLink - This method is used to return the specified link in the
/// specified node if one exists.  If a link does not already exist (it's
/// null), then we create a new node, link it, then return it.  We must
/// specify the type of the Node field we are accessing so that we know what
/// type should be linked to if we need to create a new node.
///
DSNodeHandle &GraphBuilder::getLink(const DSNodeHandle &node, unsigned LinkNo) {
  DSNodeHandle &Node = const_cast<DSNodeHandle&>(node);
  DSNodeHandle &Link = Node.getLink(LinkNo);
  if (!Link.getNode()) {
    // If the link hasn't been created yet, make and return a new shadow node
    Link = createNode(DSNode::ShadowNode);
  }
  return Link;
}


/// setDestTo - Set the ScalarMap entry for the specified value to point to the
/// specified destination.  If the Value already points to a node, make sure to
/// merge the two destinations together.
///
void GraphBuilder::setDestTo(Value &V, const DSNodeHandle &NH) {
  DSNodeHandle &AINH = ScalarMap[&V];
  if (AINH.getNode() == 0)   // Not pointing to anything yet?
    AINH = NH;               // Just point directly to NH
  else
    AINH.mergeWith(NH);
}


//===----------------------------------------------------------------------===//
// Specific instruction type handler implementations...
//

/// Alloca & Malloc instruction implementation - Simply create a new memory
/// object, pointing the scalar to it.
///
void GraphBuilder::handleAlloc(AllocationInst &AI, DSNode::NodeTy NodeType) {
  setDestTo(AI, createNode(NodeType));
}

// PHINode - Make the scalar for the PHI node point to all of the things the
// incoming values point to... which effectively causes them to be merged.
//
void GraphBuilder::visitPHINode(PHINode &PN) {
  if (!isPointerType(PN.getType())) return; // Only pointer PHIs

  DSNodeHandle &PNDest = ScalarMap[&PN];
  for (unsigned i = 0, e = PN.getNumIncomingValues(); i != e; ++i)
    PNDest.mergeWith(getValueDest(*PN.getIncomingValue(i)));
}

void GraphBuilder::visitGetElementPtrInst(User &GEP) {
  DSNodeHandle Value = getValueDest(*GEP.getOperand(0));
  if (Value.getNode() == 0) return;

  unsigned Offset = 0;
  const PointerType *PTy = cast<PointerType>(GEP.getOperand(0)->getType());
  const Type *CurTy = PTy->getElementType();

  if (Value.getNode()->mergeTypeInfo(CurTy, Value.getOffset())) {
    // If the node had to be folded... exit quickly
    setDestTo(GEP, Value);  // GEP result points to folded node
    return;
  }

#if 0
  // Handle the pointer index specially...
  if (GEP.getNumOperands() > 1 &&
      GEP.getOperand(1) != ConstantSInt::getNullValue(Type::LongTy)) {

    // If we already know this is an array being accessed, don't do anything...
    if (!TopTypeRec.isArray) {
      TopTypeRec.isArray = true;

      // If we are treating some inner field pointer as an array, fold the node
      // up because we cannot handle it right.  This can come because of
      // something like this:  &((&Pt->X)[1]) == &Pt->Y
      //
      if (Value.getOffset()) {
        // Value is now the pointer we want to GEP to be...
        Value.getNode()->foldNodeCompletely();
        setDestTo(GEP, Value);  // GEP result points to folded node
        return;
      } else {
        // This is a pointer to the first byte of the node.  Make sure that we
        // are pointing to the outter most type in the node.
        // FIXME: We need to check one more case here...
      }
    }
  }
#endif

  // All of these subscripts are indexing INTO the elements we have...
  for (unsigned i = 2, e = GEP.getNumOperands(); i < e; ++i)
    if (GEP.getOperand(i)->getType() == Type::LongTy) {
      // Get the type indexing into...
      const SequentialType *STy = cast<SequentialType>(CurTy);
      CurTy = STy->getElementType();
#if 0
      if (ConstantSInt *CS = dyn_cast<ConstantSInt>(GEP.getOperand(i))) {
        Offset += CS->getValue()*TD.getTypeSize(CurTy);
      } else {
        // Variable index into a node.  We must merge all of the elements of the
        // sequential type here.
        if (isa<PointerType>(STy))
          std::cerr << "Pointer indexing not handled yet!\n";
        else {
          const ArrayType *ATy = cast<ArrayType>(STy);
          unsigned ElSize = TD.getTypeSize(CurTy);
          DSNode *N = Value.getNode();
          assert(N && "Value must have a node!");
          unsigned RawOffset = Offset+Value.getOffset();

          // Loop over all of the elements of the array, merging them into the
          // zero'th element.
          for (unsigned i = 1, e = ATy->getNumElements(); i != e; ++i)
            // Merge all of the byte components of this array element
            for (unsigned j = 0; j != ElSize; ++j)
              N->mergeIndexes(RawOffset+j, RawOffset+i*ElSize+j);
        }
      }
#endif
    } else if (GEP.getOperand(i)->getType() == Type::UByteTy) {
      unsigned FieldNo = cast<ConstantUInt>(GEP.getOperand(i))->getValue();
      const StructType *STy = cast<StructType>(CurTy);
      Offset += TD.getStructLayout(STy)->MemberOffsets[FieldNo];
      CurTy = STy->getContainedType(FieldNo);
    }

  // Add in the offset calculated...
  Value.setOffset(Value.getOffset()+Offset);

  // Value is now the pointer we want to GEP to be...
  setDestTo(GEP, Value);
}

void GraphBuilder::visitLoadInst(LoadInst &LI) {
  DSNodeHandle Ptr = getValueDest(*LI.getOperand(0));
  if (Ptr.getNode() == 0) return;

  // Make that the node is read from...
  Ptr.getNode()->NodeType |= DSNode::Read;

  // Ensure a typerecord exists...
  Ptr.getNode()->mergeTypeInfo(LI.getType(), Ptr.getOffset());

  if (isPointerType(LI.getType()))
    setDestTo(LI, getLink(Ptr));
}

void GraphBuilder::visitStoreInst(StoreInst &SI) {
  const Type *StoredTy = SI.getOperand(0)->getType();
  DSNodeHandle Dest = getValueDest(*SI.getOperand(1));
  if (Dest.getNode() == 0) return;

  // Mark that the node is written to...
  Dest.getNode()->NodeType |= DSNode::Modified;

  // Ensure a typerecord exists...
  Dest.getNode()->mergeTypeInfo(StoredTy, Dest.getOffset());

  // Avoid adding edges from null, or processing non-"pointer" stores
  if (isPointerType(StoredTy))
    Dest.addEdgeTo(getValueDest(*SI.getOperand(0)));
}

void GraphBuilder::visitReturnInst(ReturnInst &RI) {
  if (RI.getNumOperands() && isPointerType(RI.getOperand(0)->getType()))
    RetNode.mergeWith(getValueDest(*RI.getOperand(0)));
}

void GraphBuilder::visitCallInst(CallInst &CI) {
  // Set up the return value...
  DSNodeHandle RetVal;
  if (isPointerType(CI.getType()))
    RetVal = getValueDest(CI);

  DSNode *Callee = 0;
  if (DisableDirectCallOpt || !isa<Function>(CI.getOperand(0)))
    Callee = getValueDest(*CI.getOperand(0)).getNode();

  std::vector<DSNodeHandle> Args;
  Args.reserve(CI.getNumOperands()-1);

  // Calculate the arguments vector...
  for (unsigned i = 1, e = CI.getNumOperands(); i != e; ++i)
    if (isPointerType(CI.getOperand(i)->getType()))
      Args.push_back(getValueDest(*CI.getOperand(i)));

  // Add a new function call entry...
  if (Callee)
    FunctionCalls.push_back(DSCallSite(CI, RetVal, Callee, Args));
  else
    FunctionCalls.push_back(DSCallSite(CI, RetVal,
                                       cast<Function>(CI.getOperand(0)), Args));
}

void GraphBuilder::visitFreeInst(FreeInst &FI) {
  // Mark that the node is written to...
  getValueDest(*FI.getOperand(0)).getNode()->NodeType
    |= DSNode::Modified | DSNode::HeapNode;
}

/// Handle casts...
void GraphBuilder::visitCastInst(CastInst &CI) {
  if (isPointerType(CI.getType()))
    if (isPointerType(CI.getOperand(0)->getType())) {
      // Cast one pointer to the other, just act like a copy instruction
      setDestTo(CI, getValueDest(*CI.getOperand(0)));
    } else {
      // Cast something (floating point, small integer) to a pointer.  We need
      // to track the fact that the node points to SOMETHING, just something we
      // don't know about.  Make an "Unknown" node.
      //
      setDestTo(CI, createNode(DSNode::UnknownNode));
    }
}


// visitInstruction - For all other instruction types, if we have any arguments
// that are of pointer type, make them have unknown composition bits, and merge
// the nodes together.
void GraphBuilder::visitInstruction(Instruction &Inst) {
  DSNodeHandle CurNode;
  if (isPointerType(Inst.getType()))
    CurNode = getValueDest(Inst);
  for (User::op_iterator I = Inst.op_begin(), E = Inst.op_end(); I != E; ++I)
    if (isPointerType((*I)->getType()))
      CurNode.mergeWith(getValueDest(**I));

  if (CurNode.getNode())
    CurNode.getNode()->NodeType |= DSNode::UnknownNode;
}



//===----------------------------------------------------------------------===//
// LocalDataStructures Implementation
//===----------------------------------------------------------------------===//

bool LocalDataStructures::run(Module &M) {
  GlobalsGraph = new DSGraph();

  // Calculate all of the graphs...
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
    if (!I->isExternal())
      DSInfo.insert(std::make_pair(I, new DSGraph(*I, GlobalsGraph)));
  return false;
}

// releaseMemory - If the pass pipeline is done with this pass, we can release
// our memory... here...
//
void LocalDataStructures::releaseMemory() {
  for (hash_map<const Function*, DSGraph*>::iterator I = DSInfo.begin(),
         E = DSInfo.end(); I != E; ++I)
    delete I->second;

  // Empty map so next time memory is released, data structures are not
  // re-deleted.
  DSInfo.clear();
  delete GlobalsGraph;
  GlobalsGraph = 0;
}
