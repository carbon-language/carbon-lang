//===- Local.cpp - Compute a local data structure graph for a function ----===//
//
// Compute the local version of the data structure graph for a function.  The
// external interface to this file is the DSGraph constructor.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/DataStructure.h"
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

// FIXME: This should eventually be a FunctionPass that is automatically
// aggregated into a Pass.
//
#include "llvm/Module.h"

using std::map;
using std::vector;

static RegisterAnalysis<LocalDataStructures>
X("datastructure", "Local Data Structure Analysis");

using namespace DataStructureAnalysis;

namespace DataStructureAnalysis {
  // FIXME: Do something smarter with target data!
  TargetData TD("temp-td");
  unsigned PointerSize(TD.getPointerSize());

  // isPointerType - Return true if this type is big enough to hold a pointer.
  bool isPointerType(const Type *Ty) {
    if (isa<PointerType>(Ty))
      return true;
    else if (Ty->isPrimitiveType() && Ty->isInteger())
      return Ty->getPrimitiveSize() >= PointerSize;
    return false;
  }
}


namespace {
  //===--------------------------------------------------------------------===//
  //  GraphBuilder Class
  //===--------------------------------------------------------------------===//
  //
  /// This class is the builder class that constructs the local data structure
  /// graph by performing a single pass over the function in question.
  ///
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
        if (isPointerType(I->getType()))
          getValueDest(*I);

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
    void visitCastInst(CastInst &CI);
    void visitInstruction(Instruction &I) {}

  private:
    // Helper functions used to implement the visitation functions...

    /// createNode - Create a new DSNode, ensuring that it is properly added to
    /// the graph.
    ///
    DSNode *createNode(DSNode::NodeTy NodeType, const Type *Ty);

    /// getValueNode - Return a DSNode that corresponds the the specified LLVM
    /// value.  This either returns the already existing node, or creates a new
    /// one and adds it to the graph, if none exists.
    ///
    DSNodeHandle getValueNode(Value &V);

    /// getValueDest - Return the DSNode that the actual value points to.  This
    /// is basically the same thing as: getLink(getValueNode(V), 0)
    ///
    DSNodeHandle &getValueDest(Value &V);

    /// getGlobalNode - Just like getValueNode, except the global node itself is
    /// returned, not a scalar node pointing to a global.
    ///
    DSNodeHandle &getGlobalNode(GlobalValue &V);

    /// getLink - This method is used to return the specified link in the
    /// specified node if one exists.  If a link does not already exist (it's
    /// null), then we create a new node, link it, then return it.  We must
    /// specify the type of the Node field we are accessing so that we know what
    /// type should be linked to if we need to create a new node.
    ///
    DSNodeHandle &getLink(const DSNodeHandle &Node, unsigned Link,
                          const Type *FieldTy);
  };
}

//===----------------------------------------------------------------------===//
// DSGraph constructor - Simply use the GraphBuilder to construct the local
// graph.
DSGraph::DSGraph(Function &F) : Func(&F) {
  // Use the graph builder to construct the local version of the graph
  GraphBuilder B(*this, Nodes, RetNode, ValueMap, FunctionCalls);
  markIncompleteNodes();
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
DSNodeHandle &GraphBuilder::getGlobalNode(GlobalValue &V) {
  DSNodeHandle &NH = ValueMap[&V];
  if (NH.getNode()) return NH;       // Already have a node?  Just return it...

  // Create a new global node for this global variable...
  DSNode *G = createNode(DSNode::GlobalNode, V.getType()->getElementType());
  G->addGlobal(&V);

  // If this node has outgoing edges, make sure to recycle the same node for
  // each use.  For functions and other global variables, this is unneccesary,
  // so avoid excessive merging by cloning these nodes on demand.
  //
  NH.setNode(G);
  return NH;
}


// getValueNode - Return a DSNode that corresponds the the specified LLVM value.
// This either returns the already existing node, or creates a new one and adds
// it to the graph, if none exists.
//
DSNodeHandle GraphBuilder::getValueNode(Value &V) {
  assert(isPointerType(V.getType()) && "Should only use pointer scalars!");
  // Do not share the pointer value to globals... this would cause way too much
  // false merging.
  //
  DSNodeHandle &NH = ValueMap[&V];
  if (!isa<GlobalValue>(V) && NH.getNode())
    return NH;     // Already have a node?  Just return it...
  
  // Otherwise we need to create a new scalar node...
  DSNode *N = createNode(DSNode::ScalarNode, V.getType());

  // If this is a global value, create the global pointed to.
  if (GlobalValue *GV = dyn_cast<GlobalValue>(&V)) {
    N->addEdgeTo(0, getGlobalNode(*GV));
    return DSNodeHandle(N, 0);
  } else {
    NH.setOffset(0);
    NH.setNode(N);
  }

  return NH;
}

/// getValueDest - Return the DSNode that the actual value points to.  This
/// is basically the same thing as: getLink(getValueNode(V), 0)
///
DSNodeHandle &GraphBuilder::getValueDest(Value &V) {
  return getLink(getValueNode(V), 0, V.getType());
}



/// getLink - This method is used to return the specified link in the
/// specified node if one exists.  If a link does not already exist (it's
/// null), then we create a new node, link it, then return it.  We must
/// specify the type of the Node field we are accessing so that we know what
/// type should be linked to if we need to create a new node.
///
DSNodeHandle &GraphBuilder::getLink(const DSNodeHandle &node,
                                    unsigned LinkNo, const Type *FieldTy) {
  DSNodeHandle &Node = const_cast<DSNodeHandle&>(node);

  DSNodeHandle *Link = Node.getLink(LinkNo);
  if (Link) return *Link;
  
  // If the link hasn't been created yet, make and return a new shadow node of
  // the appropriate type for FieldTy...
  //

  // If we are indexing with a typed pointer, then the thing we are pointing
  // to is of the pointed type.  If we are pointing to it with an integer
  // (because of cast to an integer), we represent it with a void type.
  //
  const Type *ReqTy;
  if (const PointerType *Ptr = dyn_cast<PointerType>(FieldTy))
    ReqTy = Ptr->getElementType();
  else
    ReqTy = Type::VoidTy;
  
  DSNode *N = createNode(DSNode::ShadowNode, ReqTy);
  Node.setLink(LinkNo, N);
  return *Node.getLink(LinkNo);
}


//===----------------------------------------------------------------------===//
// Specific instruction type handler implementations...
//

/// Alloca & Malloc instruction implementation - Simply create a new memory
/// object, pointing the scalar to it.
///
void GraphBuilder::handleAlloc(AllocationInst &AI, DSNode::NodeTy NodeType) {
  DSNode *New = createNode(NodeType, AI.getAllocatedType());

  // Make the scalar point to the new node...
  getValueNode(AI).addEdgeTo(New);
}

// PHINode - Make the scalar for the PHI node point to all of the things the
// incoming values point to... which effectively causes them to be merged.
//
void GraphBuilder::visitPHINode(PHINode &PN) {
  if (!isPointerType(PN.getType())) return; // Only pointer PHIs

  DSNodeHandle &ScalarDest = getValueDest(PN);
  for (unsigned i = 0, e = PN.getNumIncomingValues(); i != e; ++i)
    if (!isa<ConstantPointerNull>(PN.getIncomingValue(i)))
      ScalarDest.mergeWith(getValueDest(*PN.getIncomingValue(i)));
}

void GraphBuilder::visitGetElementPtrInst(GetElementPtrInst &GEP) {
  DSNodeHandle Value = getValueDest(*GEP.getOperand(0));

  unsigned Offset = 0;
  const Type *CurTy = GEP.getOperand(0)->getType();

  for (unsigned i = 1, e = GEP.getNumOperands(); i != e; ++i)
    if (GEP.getOperand(i)->getType() == Type::LongTy) {
      // Get the type indexing into...
      const SequentialType *STy = cast<SequentialType>(CurTy);
      CurTy = STy->getElementType();
      if (ConstantSInt *CS = dyn_cast<ConstantSInt>(GEP.getOperand(i))) {
        if (isa<PointerType>(STy))
          std::cerr << "Pointer indexing not handled yet!\n";
        else 
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
    } else if (GEP.getOperand(i)->getType() == Type::UByteTy) {
      unsigned FieldNo = cast<ConstantUInt>(GEP.getOperand(i))->getValue();
      const StructType *STy = cast<StructType>(CurTy);
      Offset += TD.getStructLayout(STy)->MemberOffsets[FieldNo];
      CurTy = STy->getContainedType(FieldNo);
    }

  // Add in the offset calculated...
  Value.setOffset(Value.getOffset()+Offset);

  // Value is now the pointer we want to GEP to be...
  getValueNode(GEP).addEdgeTo(Value);
}

void GraphBuilder::visitLoadInst(LoadInst &LI) {
  DSNodeHandle &Ptr = getValueDest(*LI.getOperand(0));
  if (isPointerType(LI.getType()))
    getValueNode(LI).addEdgeTo(getLink(Ptr, 0, LI.getType()));
}

void GraphBuilder::visitStoreInst(StoreInst &SI) {
  DSNodeHandle &Dest = getValueDest(*SI.getOperand(1));

  // Avoid adding edges from null, or processing non-"pointer" stores
  if (isPointerType(SI.getOperand(0)->getType()) &&
      !isa<ConstantPointerNull>(SI.getOperand(0))) {
    Dest.addEdgeTo(getValueDest(*SI.getOperand(0)));
  }
}

void GraphBuilder::visitReturnInst(ReturnInst &RI) {
  if (RI.getNumOperands() && isPointerType(RI.getOperand(0)->getType()) &&
      !isa<ConstantPointerNull>(RI.getOperand(0))) {
    DSNodeHandle &Value = getValueDest(*RI.getOperand(0));
    Value.mergeWith(RetNode);
    RetNode = Value;
  }
}

void GraphBuilder::visitCallInst(CallInst &CI) {
  // Add a new function call entry...
  FunctionCalls.push_back(vector<DSNodeHandle>());
  vector<DSNodeHandle> &Args = FunctionCalls.back();

  // Set up the return value...
  if (isPointerType(CI.getType()))
    Args.push_back(getLink(getValueNode(CI), 0, CI.getType()));
  else
    Args.push_back(DSNodeHandle());

  unsigned Start = 0;
  // Special case for a direct call, avoid creating spurious scalar node...
  if (GlobalValue *GV = dyn_cast<GlobalValue>(CI.getOperand(0))) {
    Args.push_back(getGlobalNode(*GV));
    Start = 1;
  }

  // Pass the arguments in...
  for (unsigned i = Start, e = CI.getNumOperands(); i != e; ++i)
    if (isPointerType(CI.getOperand(i)->getType()))
      Args.push_back(getLink(getValueNode(*CI.getOperand(i)), 0,
                             CI.getOperand(i)->getType()));
}

/// Handle casts...
void GraphBuilder::visitCastInst(CastInst &CI) {
  if (isPointerType(CI.getType()) && isPointerType(CI.getOperand(0)->getType()))
    getValueNode(CI).addEdgeTo(getLink(getValueNode(*CI.getOperand(0)), 0,
                                       CI.getOperand(0)->getType()));
}




//===----------------------------------------------------------------------===//
// LocalDataStructures Implementation
//===----------------------------------------------------------------------===//

// releaseMemory - If the pass pipeline is done with this pass, we can release
// our memory... here...
//
void LocalDataStructures::releaseMemory() {
  for (std::map<const Function*, DSGraph*>::iterator I = DSInfo.begin(),
         E = DSInfo.end(); I != E; ++I)
    delete I->second;

  // Empty map so next time memory is released, data structures are not
  // re-deleted.
  DSInfo.clear();
}

bool LocalDataStructures::run(Module &M) {
  // Calculate all of the graphs...
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
    if (!I->isExternal())
      DSInfo.insert(std::make_pair(I, new DSGraph(*I)));
  return false;
}

