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
#include "llvm/DerivedTypes.h"
#include "llvm/Support/InstVisitor.h"
using std::map;
using std::vector;

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
      visit(G.getFunction());  // Single pass over the function
      removeDeadNodes();
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
    void visitInstruction(Instruction &I) {
#ifndef NDEBUG
      bool bad = isa<PointerType>(I.getType());
      for (Instruction::op_iterator i = I.op_begin(), E = I.op_end(); i!=E; ++i)
        bad |= isa<PointerType>(i->get()->getType());
      if (bad) {
        std::cerr << "\n\n\nUNKNOWN PTR INSTRUCTION type: " << I << "\n\n\n";
        assert(0 && "Cannot proceed");
      }
#endif
    }

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

    // getLink - This method is used to either return the specified link in the
    // specified node if one exists.  If a link does not already exist (it's
    // null), then we create a new node, link it, then return it.
    //
    DSNode *getLink(DSNode *Node, unsigned Link);

    // getSubscriptedNode - Perform the basic getelementptr functionality that
    // must be factored out of gep, load and store while they are all MAI's.
    //
    DSNode *getSubscriptedNode(MemAccessInst &MAI, DSNode *Ptr);

    // removeDeadNodes - After the graph has been constructed, this method
    // removes all unreachable nodes that are created because they got merged
    // with other nodes in the graph.
    //
    void removeDeadNodes();
  };
}

//===----------------------------------------------------------------------===//
// DSGraph constructor - Simply use the GraphBuilder to construct the local
// graph.
DSGraph::DSGraph(Function &F) : Func(F), RetNode(0) {
  // Use the graph builder to construct the local version of the graph
  GraphBuilder B(*this, Nodes, RetNode, ValueMap, FunctionCalls);
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


// getValueNode - Return a DSNode that corresponds the the specified LLVM value.
// This either returns the already existing node, or creates a new one and adds
// it to the graph, if none exists.
//
DSNode *GraphBuilder::getValueNode(Value &V) {
  assert(isa<PointerType>(V.getType()) && "Should only use pointer scalars!");
  DSNodeHandle &N = ValueMap[&V];
  if (N) return N;             // Already have a node?  Just return it...
  
  // Otherwise we need to create a new scalar node...
  N = createNode(DSNode::ScalarNode, V.getType());

  if (isa<GlobalValue>(V)) {
    // Traverse the global graph, adding nodes for them all, and marking them
    // all globals.  Be careful to mark functions global as well as the
    // potential graph of global variables.
    //
    DSNode *G = getLink(N, 0);
    G->NodeType |= DSNode::GlobalNode;
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


// removeDeadNodes - After the graph has been constructed, this method removes
// all unreachable nodes that are created because they got merged with other
// nodes in the graph.  These nodes will all be trivially unreachable, so we
// don't have to perform any non-trivial analysis here.
//
void GraphBuilder::removeDeadNodes() {
  for (unsigned i = 0; i != Nodes.size(); )
    if (!Nodes[i]->getReferrers().empty())
      ++i;                                    // This node is alive!
    else {                                    // This node is dead!
      delete Nodes[i];                        // Free memory...
      Nodes.erase(Nodes.begin()+i);           // Remove from node list...
    }
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

  DSNode *Scalar = getValueNode(PN);
  for (unsigned i = 0, e = PN.getNumIncomingValues(); i != e; ++i)
    Scalar->mergeWith(getValueNode(*PN.getIncomingValue(i)));
}

void GraphBuilder::visitGetElementPtrInst(GetElementPtrInst &GEP) {
  DSNode *Scalar = getValueNode(GEP);
  DSNode *Ptr    = getSubscriptedNode(GEP, getValueNode(*GEP.getOperand(0)));
  Scalar->addEdgeTo(Ptr);
}

void GraphBuilder::visitLoadInst(LoadInst &LI) {
  if (!isa<PointerType>(LI.getType())) return; // Only pointer PHIs
  DSNode *Ptr    = getSubscriptedNode(LI, getValueNode(*LI.getOperand(0)));
  getValueNode(LI)->mergeWith(Ptr);
}

void GraphBuilder::visitStoreInst(StoreInst &SI) {
  if (!isa<PointerType>(SI.getOperand(0)->getType())) return;
  DSNode *Value   = getValueNode(*SI.getOperand(0));
  DSNode *DestPtr = getValueNode(*SI.getOperand(1));
  Value->mergeWith(getSubscriptedNode(SI, DestPtr));
}

void GraphBuilder::visitReturnInst(ReturnInst &RI) {
  if (RI.getNumOperands() && isa<PointerType>(RI.getOperand(0)->getType())) {
    DSNode *Value = getValueNode(*RI.getOperand(0));
    Value->mergeWith(RetNode);
    RetNode = Value;
  }
}

void GraphBuilder::visitCallInst(CallInst &CI) {
  FunctionCalls.push_back(vector<DSNodeHandle>());
  vector<DSNodeHandle> &Args = FunctionCalls.back();

  for (unsigned i = 0, e = CI.getNumOperands(); i != e; ++i)
    if (isa<PointerType>(CI.getOperand(i)->getType()))
      Args.push_back(getValueNode(*CI.getOperand(i)));
}
