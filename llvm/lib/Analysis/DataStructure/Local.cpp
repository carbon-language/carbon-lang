//===- ComputeLocal.cpp - Compute a local data structure graph for a fn ---===//
//
// Compute the local version of the data structure graph for a function.  The
// external interface to this file is the DSGraph constructor.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/DataStructure.h"
#include "llvm/Function.h"
#include "llvm/iMemory.h"
#include "llvm/iTerminators.h"
#include "llvm/iPHINode.h"
#include "llvm/iOther.h"
#include "llvm/Constants.h"
#include "llvm/GlobalVariable.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Support/InstVisitor.h"
using std::map;
using std::vector;

static RegisterAnalysis<LocalDataStructures>
X("datastructure", "Local Data Structure Analysis");

//===----------------------------------------------------------------------===//
//  GraphBuilder Class
//===----------------------------------------------------------------------===//
//
// This class is the builder class that constructs the local data structure
// graph by performing a single pass over the function in question.
//

namespace {
  class GraphBuilder : InstVisitor<GraphBuilder> {
    DSGraph &G;
    vector<DSNode*> &Nodes;
    DSNodeHandle &RetNode;               // Node that gets returned...
    map<Value*, DSNodeHandle> &ValueMap;
    vector<vector<DSNodeHandle> > &FunctionCalls;

  public:
    GraphBuilder(DSGraph &g, vector<DSNode*> &nodes, DSNodeHandle &retNode,
                 map<Value*, DSNodeHandle> &vm,
                 vector<vector<DSNodeHandle> > &fc)
      : G(g), Nodes(nodes), RetNode(retNode), ValueMap(vm), FunctionCalls(fc) {

      // Create scalar nodes for all pointer arguments...
      for (Function::aiterator I = G.getFunction().abegin(),
             E = G.getFunction().aend(); I != E; ++I)
        if (isa<PointerType>(I->getType()))
          getValueNode(*I);

      visit(G.getFunction());  // Single pass over the function

      // Not inlining, only eliminate trivially dead nodes.
      G.removeTriviallyDeadNodes();
    }

  private:
    // Visitor functions, used to handle each instruction type we encounter...
    friend class InstVisitor<GraphBuilder>;
    void visitMallocInst(MallocInst &MI) { handleAlloc(MI, DSNode::NewNode); }
    void visitAllocaInst(AllocaInst &AI) { handleAlloc(AI, DSNode::AllocaNode);}
    void handleAlloc(AllocationInst &AI, DSNode::NodeTy NT);

    void visitPHINode(PHINode &PN);

    void visitGetElementPtrInst(GetElementPtrInst &GEP);
    void visitReturnInst(ReturnInst &RI);
    void visitLoadInst(LoadInst &LI);
    void visitStoreInst(StoreInst &SI);
    void visitCallInst(CallInst &CI);
    void visitSetCondInst(SetCondInst &SCI) {}  // SetEQ & friends are ignored
    void visitFreeInst(FreeInst &FI) {}         // Ignore free instructions
    void visitInstruction(Instruction &I);      // Visit unsafe ptr instruction

  private:
    // Helper functions used to implement the visitation functions...

    // createNode - Create a new DSNode, ensuring that it is properly added to
    // the graph.
    //
    DSNode *createNode(DSNode::NodeTy NodeType, const Type *Ty);

    // getValueNode - Return a DSNode that corresponds the the specified LLVM
    // value.  This either returns the already existing node, or creates a new
    // one and adds it to the graph, if none exists.
    //
    DSNode *getValueNode(Value &V);

    // getGlobalNode - Just like getValueNode, except the global node itself is
    // returned, not a scalar node pointing to a global.
    //
    DSNode *getGlobalNode(GlobalValue &V);

    // getLink - This method is used to either return the specified link in the
    // specified node if one exists.  If a link does not already exist (it's
    // null), then we create a new node, link it, then return it.
    //
    DSNode *getLink(DSNode *Node, unsigned Link);

    // getSubscriptedNode - Perform the basic getelementptr functionality that
    // must be factored out of gep, load and store while they are all MAI's.
    //
    DSNode *getSubscriptedNode(MemAccessInst &MAI, DSNode *Ptr);
  };
}

//===----------------------------------------------------------------------===//
// DSGraph constructor - Simply use the GraphBuilder to construct the local
// graph.
DSGraph::DSGraph(Function &F, GlobalDSGraph* GlobalsG)
  : Func(F), RetNode(0), GlobalsGraph(GlobalsG) {
  if (GlobalsGraph != this) {
    GlobalsGraph->addReference(this);
    // Use the graph builder to construct the local version of the graph
    GraphBuilder B(*this, Nodes, RetNode, ValueMap, FunctionCalls);
    markIncompleteNodes();
  }
}


//===----------------------------------------------------------------------===//
// Helper method implementations...
//


// createNode - Create a new DSNode, ensuring that it is properly added to the
// graph.
//
DSNode *GraphBuilder::createNode(DSNode::NodeTy NodeType, const Type *Ty) {
  DSNode *N = new DSNode(NodeType, Ty);
  Nodes.push_back(N);
  return N;
}


// getGlobalNode - Just like getValueNode, except the global node itself is
// returned, not a scalar node pointing to a global.
//
DSNode *GraphBuilder::getGlobalNode(GlobalValue &V) {
  DSNodeHandle &NH = ValueMap[&V];
  if (NH) return NH;             // Already have a node?  Just return it...

  // Create a new global node for this global variable...
  DSNode *G = createNode(DSNode::GlobalNode, V.getType()->getElementType());
  G->addGlobal(&V);

  // If this node has outgoing edges, make sure to recycle the same node for
  // each use.  For functions and other global variables, this is unneccesary,
  // so avoid excessive merging by cloning these nodes on demand.
  //
  NH = G;
  return G;
}


// getValueNode - Return a DSNode that corresponds the the specified LLVM value.
// This either returns the already existing node, or creates a new one and adds
// it to the graph, if none exists.
//
DSNode *GraphBuilder::getValueNode(Value &V) {
  assert(isa<PointerType>(V.getType()) && "Should only use pointer scalars!");
  if (!isa<GlobalValue>(V)) {
    DSNodeHandle &NH = ValueMap[&V];
    if (NH) return NH;             // Already have a node?  Just return it...
  }
  
  // Otherwise we need to create a new scalar node...
  DSNode *N = createNode(DSNode::ScalarNode, V.getType());

  // If this is a global value, create the global pointed to.
  if (GlobalValue *GV = dyn_cast<GlobalValue>(&V)) {
    DSNode *G = getGlobalNode(*GV);
    N->addEdgeTo(G);
  } else {
    ValueMap[&V] = N;
  }

  return N;
}


// getLink - This method is used to either return the specified link in the
// specified node if one exists.  If a link does not already exist (it's
// null), then we create a new node, link it, then return it.
//
DSNode *GraphBuilder::getLink(DSNode *Node, unsigned Link) {
  assert(Link < Node->getNumLinks() && "Link accessed out of range!");
  if (Node->getLink(Link) == 0) {
    DSNode::NodeTy NT;
    const Type *Ty;

    switch (Node->getType()->getPrimitiveID()) {
    case Type::PointerTyID:
      Ty = cast<PointerType>(Node->getType())->getElementType();
      NT = DSNode::ShadowNode;
      break;
    case Type::ArrayTyID:
      Ty = cast<ArrayType>(Node->getType())->getElementType();
      NT = DSNode::SubElement;
      break;
    case Type::StructTyID:
      Ty = cast<StructType>(Node->getType())->getContainedType(Link);
      NT = DSNode::SubElement;
      break;
    default:
      assert(0 && "Unexpected type to dereference!");
      abort();
    }

    DSNode *New = createNode(NT, Ty);
    Node->addEdgeTo(Link, New);
  }

  return Node->getLink(Link);
}

// getSubscriptedNode - Perform the basic getelementptr functionality that must
// be factored out of gep, load and store while they are all MAI's.
//
DSNode *GraphBuilder::getSubscriptedNode(MemAccessInst &MAI, DSNode *Ptr) {
  for (unsigned i = MAI.getFirstIndexOperandNumber(), e = MAI.getNumOperands();
       i != e; ++i)
    if (MAI.getOperand(i)->getType() == Type::UIntTy)
      Ptr = getLink(Ptr, 0);
    else if (MAI.getOperand(i)->getType() == Type::UByteTy)
      Ptr = getLink(Ptr, cast<ConstantUInt>(MAI.getOperand(i))->getValue());  

  if (MAI.getFirstIndexOperandNumber() == MAI.getNumOperands())
    Ptr = getLink(Ptr, 0);  // All MAI's have an implicit 0 if nothing else.

  return Ptr;
}

//===----------------------------------------------------------------------===//
// Specific instruction type handler implementations...
//

// Alloca & Malloc instruction implementation - Simply create a new memory
// object, pointing the scalar to it.
//
void GraphBuilder::handleAlloc(AllocationInst &AI, DSNode::NodeTy NodeType) {
  DSNode *Scalar = getValueNode(AI);
  DSNode *New = createNode(NodeType, AI.getAllocatedType());
  Scalar->addEdgeTo(New);   // Make the scalar point to the new node...
}

// PHINode - Make the scalar for the PHI node point to all of the things the
// incoming values point to... which effectively causes them to be merged.
//
void GraphBuilder::visitPHINode(PHINode &PN) {
  if (!isa<PointerType>(PN.getType())) return; // Only pointer PHIs

  DSNode *Scalar     = getValueNode(PN);
  DSNode *ScalarDest = getLink(Scalar, 0);
  for (unsigned i = 0, e = PN.getNumIncomingValues(); i != e; ++i)
    ScalarDest->mergeWith(getLink(getValueNode(*PN.getIncomingValue(i)), 0));
}

void GraphBuilder::visitGetElementPtrInst(GetElementPtrInst &GEP) {
  DSNode *Ptr = getSubscriptedNode(GEP, getValueNode(*GEP.getOperand(0)));
  getValueNode(GEP)->addEdgeTo(Ptr);
}

void GraphBuilder::visitLoadInst(LoadInst &LI) {
  DSNode *Ptr = getSubscriptedNode(LI, getValueNode(*LI.getOperand(0)));
  if (!isa<PointerType>(LI.getType())) return; // Only pointer PHIs
  getValueNode(LI)->addEdgeTo(getLink(Ptr, 0));
}

void GraphBuilder::visitStoreInst(StoreInst &SI) {
  DSNode *DestPtr = getSubscriptedNode(SI, getValueNode(*SI.getOperand(1)));
  if (!isa<PointerType>(SI.getOperand(0)->getType())) return;
  DSNode *Value   = getValueNode(*SI.getOperand(0));
  DestPtr->addEdgeTo(getLink(Value, 0));
}

void GraphBuilder::visitReturnInst(ReturnInst &RI) {
  if (RI.getNumOperands() && isa<PointerType>(RI.getOperand(0)->getType())) {
    DSNode *Value = getLink(getValueNode(*RI.getOperand(0)), 0);
    Value->mergeWith(RetNode);
    RetNode = Value;
  }
}

void GraphBuilder::visitCallInst(CallInst &CI) {
  // Add a new function call entry...
  FunctionCalls.push_back(vector<DSNodeHandle>());
  vector<DSNodeHandle> &Args = FunctionCalls.back();

  // Set up the return value...
  if (isa<PointerType>(CI.getType()))
    Args.push_back(getLink(getValueNode(CI), 0));
  else
    Args.push_back(0);

  unsigned Start = 0;
  // Special case for direct call, avoid creating spurious scalar node...
  if (GlobalValue *GV = dyn_cast<GlobalValue>(CI.getOperand(0))) {
    Args.push_back(getGlobalNode(*GV));
    Start = 1;
  }

  // Pass the arguments in...
  for (unsigned i = Start, e = CI.getNumOperands(); i != e; ++i)
    if (isa<PointerType>(CI.getOperand(i)->getType()))
      Args.push_back(getLink(getValueNode(*CI.getOperand(i)), 0));
}

// visitInstruction - All safe instructions have been processed above, this case
// is where unsafe ptr instructions land.
//
void GraphBuilder::visitInstruction(Instruction &I) {
  // If the return type is a pointer, mark the pointed node as being a cast node
  if (isa<PointerType>(I.getType()))
    getLink(getValueNode(I), 0)->NodeType |= DSNode::CastNode;

  // If any operands are pointers, mark the pointed nodes as being a cast node
  for (Instruction::op_iterator i = I.op_begin(), E = I.op_end(); i!=E; ++i)
    if (isa<PointerType>(i->get()->getType()))
      getLink(getValueNode(*i->get()), 0)->NodeType |= DSNode::CastNode;
}

