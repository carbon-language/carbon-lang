//===- NodeImpl.cpp - Implement the data structure analysis nodes ---------===//
//
// Implement the LLVM data structure analysis library.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/DataStructureGraph.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/iMemory.h"
#include "llvm/iOther.h"
#include "Support/STLExtras.h"
#include <algorithm>
#include <sstream>
using std::map;
using std::string;

bool AllocDSNode::isEquivalentTo(DSNode *Node) const {
  if (AllocDSNode *N = dyn_cast<AllocDSNode>(Node))
    return getType() == Node->getType();
  //&& isAllocaNode() == N->isAllocaNode();
  return false;
}

void AllocDSNode::mergeInto(DSNode *Node) const {
  // Make sure the merged node is variable size if this node is var size
  AllocDSNode *N = cast<AllocDSNode>(Node);
  N->isVarSize |= isVarSize;
}

bool GlobalDSNode::isEquivalentTo(DSNode *Node) const {
  if (const GlobalDSNode *G = dyn_cast<GlobalDSNode>(Node)) {
    if (G->Val != Val) return false;

    // Check that the outgoing links are identical...
    assert(getNumLinks() == G->getNumLinks() && "Not identical shape?");
    for (unsigned i = 0, e = getNumLinks(); i != e; ++i)
      if (getLink(i) != G->getLink(i))          // Check links
        return false;
    return true;
  }
  return false;
}

// Call node equivalency - Two call nodes are identical if all of the outgoing
// links are the same, AND if all of the incoming links are identical.
//
bool CallDSNode::isEquivalentTo(DSNode *Node) const {
  if (CallDSNode *C = dyn_cast<CallDSNode>(Node)) {
    if (getReferrers().size() != C->getReferrers().size() ||
        C->getType() != getType())
      return false; // Quick check...

    // Check that the outgoing links are identical...
    assert(getNumLinks() == C->getNumLinks() && "Not identical shape?");
    for (unsigned i = 0, e = getNumLinks(); i != e; ++i)
      if (getLink(i) != C->getLink(i))          // Check links
        return false;


    std::vector<PointerValSet*> Refs1 = C->getReferrers();
    std::vector<PointerValSet*> Refs2 = getReferrers();

    sort(Refs1.begin(), Refs1.end());
    sort(Refs2.begin(), Refs2.end());
    if (Refs1 != Refs2) return false;               // Incoming edges different?

    // Check that all outgoing links are the same...
    return C->ArgLinks == ArgLinks;   // Check that the arguments are identical
  }
  return false;
}

// NodesAreEquivalent - Check to see if the nodes are equivalent in all ways
// except node type.  Since we know N1 is a shadow node, N2 is allowed to be
// any type.
//
bool ShadowDSNode::isEquivalentTo(DSNode *Node) const {
  return getType() == Node->getType();
}




//===----------------------------------------------------------------------===//
//  DSNode Class Implementation
//

static void MapPVS(PointerValSet &PVSOut, const PointerValSet &PVSIn,
                   map<const DSNode*, DSNode*> &NodeMap, bool ReinitOk = false){
  assert((ReinitOk || PVSOut.empty()) && "Value set already initialized!");

  for (unsigned i = 0, e = PVSIn.size(); i != e; ++i)
    PVSOut.add(PointerVal(NodeMap[PVSIn[i].Node], PVSIn[i].Index));
}



unsigned countPointerFields(const Type *Ty) {
  switch (Ty->getPrimitiveID()) {
  case Type::StructTyID: {
    const StructType *ST = cast<StructType>(Ty);
    unsigned Sum = 0;
    for (unsigned i = 0, e = ST->getNumContainedTypes(); i != e; ++i)
      Sum += countPointerFields(ST->getContainedType(i));

    return Sum;
  }

  case Type::ArrayTyID:
    // All array elements are folded together...
    return countPointerFields(cast<ArrayType>(Ty)->getElementType());

  case Type::PointerTyID:
    return 1;
    
  default:                     // Some other type, just treat it like a scalar
    return 0;
  }
}

DSNode::DSNode(enum NodeTy NT, const Type *T) : Ty(T), NodeType(NT) {
  // Create field entries for all of the values in this type...
  FieldLinks.resize(countPointerFields(getType()));
}

void DSNode::removeReferrer(PointerValSet *PVS) {
  std::vector<PointerValSet*>::iterator I = std::find(Referrers.begin(),
                                                      Referrers.end(), PVS);
  assert(I != Referrers.end() && "PVS not pointing to node!");
  Referrers.erase(I);
}


// removeAllIncomingEdges - Erase all edges in the graph that point to this node
void DSNode::removeAllIncomingEdges() {
  while (!Referrers.empty())
    Referrers.back()->removePointerTo(this);
}


static void replaceIn(std::string &S, char From, const std::string &To) {
  for (unsigned i = 0; i < S.size(); )
    if (S[i] == From) {
      S.replace(S.begin()+i, S.begin()+i+1,
                To.begin(), To.end());
      i += To.size();
    } else {
      ++i;
    }
}

static void writeEdges(std::ostream &O, const void *SrcNode,
                       const char *SrcNodePortName, int SrcNodeIdx,
                       const PointerValSet &VS, const string &EdgeAttr = "") {
  for (unsigned j = 0, je = VS.size(); j != je; ++j) {
    O << "\t\tNode" << SrcNode << SrcNodePortName;
    if (SrcNodeIdx != -1) O << SrcNodeIdx;

    O << " -> Node" << VS[j].Node;
    if (VS[j].Index)
      O << ":g" << VS[j].Index;

    if (!EdgeAttr.empty())
      O << "[" << EdgeAttr << "]";
    O << ";\n";
  }
}

static string escapeLabel(const string &In) {
  string Label(In);
  replaceIn(Label, '\\', "\\\\\\\\");  // Escape caption...
  replaceIn(Label, ' ', "\\ ");
  replaceIn(Label, '{', "\\{");
  replaceIn(Label, '}', "\\}");
  return Label;
}

void DSNode::dump() const { print(std::cerr); }

void DSNode::print(std::ostream &O) const {
  string Caption = escapeLabel(getCaption());

  O << "\t\tNode" << (void*)this << " [ label =\"{" << Caption;

  const std::vector<PointerValSet> *Links = getAuxLinks();
  if (Links && !Links->empty()) {
    O << "|{";
    for (unsigned i = 0; i < Links->size(); ++i) {
      if (i) O << "|";
      O << "<f" << i << ">";
    }
    O << "}";
  }

  if (!FieldLinks.empty()) {
    O << "|{";
    for (unsigned i = 0; i < FieldLinks.size(); ++i) {
      if (i) O << "|";
      O << "<g" << i << ">";
    }
    O << "}";
  }
  O << "}\"];\n";

  if (Links)
    for (unsigned i = 0; i < Links->size(); ++i)
      writeEdges(O, this, ":f", i, (*Links)[i]);
  for (unsigned i = 0; i < FieldLinks.size(); ++i)
    writeEdges(O, this, ":g", i, FieldLinks[i]);
}

void DSNode::mapNode(map<const DSNode*, DSNode*> &NodeMap, const DSNode *Old) {
  assert(FieldLinks.size() == Old->FieldLinks.size() &&
         "Cloned nodes do not have the same number of links!");
  for (unsigned j = 0, je = FieldLinks.size(); j != je; ++j)
    MapPVS(FieldLinks[j], Old->FieldLinks[j], NodeMap);

  // Map our SynthNodes...
  assert(SynthNodes.empty() && "Synthnodes already mapped?");
  SynthNodes.reserve(Old->SynthNodes.size());
  for (unsigned i = 0, e = Old->SynthNodes.size(); i != e; ++i)
    SynthNodes.push_back(std::make_pair(Old->SynthNodes[i].first,
                    (ShadowDSNode*)NodeMap[Old->SynthNodes[i].second]));
}

AllocDSNode::AllocDSNode(AllocationInst *V, bool isvarsize)
  : DSNode(NewNode, V->getType()->getElementType()), Allocation(V) {

  // Is variable size if incoming flag says so, or if allocation is var size
  // already.
  isVarSize = isvarsize || !isa<Constant>(V->getArraySize());
}

bool AllocDSNode::isAllocaNode() const {
  return isa<AllocaInst>(Allocation);
}


string AllocDSNode::getCaption() const {
  std::stringstream OS;
  OS << (isMallocNode() ? "new " : "alloca ");

  WriteTypeSymbolic(OS, getType(),
                    Allocation->getParent()->getParent()->getParent());
  if (isVarSize)
    OS << "[ ]";
  return OS.str();
}

GlobalDSNode::GlobalDSNode(GlobalValue *V)
  : DSNode(GlobalNode, V->getType()->getElementType()), Val(V) {
}

string GlobalDSNode::getCaption() const {
  std::stringstream OS;
  if (isa<Function>(Val))
    OS << "fn ";
  else
    OS << "global ";

  WriteTypeSymbolic(OS, getType(), Val->getParent());
  return OS.str() + " %" + Val->getName();
}


ShadowDSNode::ShadowDSNode(const Type *Ty, Module *M) : DSNode(ShadowNode, Ty) {
  Mod = M;
  ShadowParent = 0;
}

ShadowDSNode::ShadowDSNode(const Type *Ty, Module *M, DSNode *ShadParent)
  : DSNode(ShadowNode, Ty) {
  Mod = M;
  ShadowParent = ShadParent;
}

std::string ShadowDSNode::getCaption() const {
  std::stringstream OS;
  OS << "shadow ";
  WriteTypeSymbolic(OS, getType(), Mod);
  return OS.str();
}

CallDSNode::CallDSNode(CallInst *ci) : DSNode(CallNode, ci->getType()), CI(ci) {
  unsigned NumPtrs = 0;
  for (unsigned i = 0, e = ci->getNumOperands(); i != e; ++i)
    if (isa<PointerType>(ci->getOperand(i)->getType()))
      NumPtrs++;
  ArgLinks.resize(NumPtrs);
}

string CallDSNode::getCaption() const {
  std::stringstream OS;
  if (const Function *CM = CI->getCalledFunction())
    OS << "call " << CM->getName();
  else
    OS << "call <indirect>";
  OS << ": ";
  WriteTypeSymbolic(OS, getType(),
                    CI->getParent()->getParent()->getParent());
  return OS.str();
}

void CallDSNode::mapNode(map<const DSNode*, DSNode*> &NodeMap,
                         const DSNode *O) {
  const CallDSNode *Old = cast<CallDSNode>(O);
  DSNode::mapNode(NodeMap, Old);  // Map base portions first...

  assert(ArgLinks.size() == Old->ArgLinks.size() && "# Arguments changed!?");
  for (unsigned i = 0, e = Old->ArgLinks.size(); i != e; ++i)
    MapPVS(ArgLinks[i], Old->ArgLinks[i], NodeMap);
}

void FunctionDSGraph::printFunction(std::ostream &O,
                                    const char *Label) const {
  O << "\tsubgraph cluster_" << Label << "_Function" << (void*)this << " {\n";
  O << "\t\tlabel=\"" << Label << " Function\\ " << Func->getName() << "\";\n";
  for (unsigned i = 0, e = AllocNodes.size(); i != e; ++i)
    AllocNodes[i]->print(O);
  for (unsigned i = 0, e = ShadowNodes.size(); i != e; ++i)
    ShadowNodes[i]->print(O);
  for (unsigned i = 0, e = GlobalNodes.size(); i != e; ++i)
    GlobalNodes[i]->print(O);
  for (unsigned i = 0, e = CallNodes.size(); i != e; ++i)
    CallNodes[i]->print(O);

  if (RetNode.size()) {
    O << "\t\tNode" << (void*)this << Label
      << " [shape=\"ellipse\", label=\"Returns\"];\n";
    writeEdges(O, this, Label, -1, RetNode);
  }

  O << "\n";
  for (std::map<Value*, PointerValSet>::const_iterator I = ValueMap.begin(),
         E = ValueMap.end(); I != E; ++I) {
    if (I->second.size()) {             // Only output nodes with edges...
      std::stringstream OS;
      WriteTypeSymbolic(OS, I->first->getType(), Func->getParent());

      // Create node for I->first
      O << "\t\tNode" << (void*)I->first << Label << " [shape=\""
        << (isa<Argument>(I->first) ? "ellipse" : "box") << "\", label=\""
        << escapeLabel(OS.str()) << "\\n%" << escapeLabel(I->first->getName())
        << "\",fontsize=\"12.0\",color=\"gray70\"];\n";
      
      // add edges from I->first to all pointers in I->second
      writeEdges(O, I->first, Label, -1, I->second,
                 "weight=\"0.9\",color=\"gray70\"");
    }
  }
  
  O << "\t}\n";
}

// Copy constructor - Since we copy the nodes over, we have to be sure to go
// through and fix pointers to point into the new graph instead of into the old
// graph...
//
FunctionDSGraph::FunctionDSGraph(const FunctionDSGraph &DSG) : Func(DSG.Func) {
  std::vector<PointerValSet> Args;
  RetNode = cloneFunctionIntoSelf(DSG, true, Args);
}


// cloneFunctionIntoSelf - Clone the specified method graph into the current
// method graph, returning the Return's set of the graph.   If ValueMap is set
// to true, the ValueMap of the function is cloned into this function as well
// as the data structure graph itself.  Regardless, the arguments value sets
// of DSG are copied into Args.
//
PointerValSet FunctionDSGraph::cloneFunctionIntoSelf(const FunctionDSGraph &DSG,
                                                     bool CloneValueMap,
                                                  std::vector<PointerValSet> &Args) {
  map<const DSNode*, DSNode*> NodeMap;  // Map from old graph to new graph...
  unsigned StartAllocSize = AllocNodes.size();
  AllocNodes.reserve(StartAllocSize+DSG.AllocNodes.size());
  unsigned StartShadowSize = ShadowNodes.size();
  ShadowNodes.reserve(StartShadowSize+DSG.ShadowNodes.size());
  unsigned StartGlobalSize = GlobalNodes.size();
  GlobalNodes.reserve(StartGlobalSize+DSG.GlobalNodes.size());
  unsigned StartCallSize = CallNodes.size();
  CallNodes.reserve(StartCallSize+DSG.CallNodes.size());

  // Clone all of the alloc nodes similarly...
  for (unsigned i = 0, e = DSG.AllocNodes.size(); i != e; ++i) {
    AllocDSNode *New = cast<AllocDSNode>(DSG.AllocNodes[i]->clone());
    NodeMap[DSG.AllocNodes[i]] = New;
    AllocNodes.push_back(New);
  }

  // Clone all of the shadow nodes similarly...
  for (unsigned i = 0, e = DSG.ShadowNodes.size(); i != e; ++i) {
    ShadowDSNode *New = cast<ShadowDSNode>(DSG.ShadowNodes[i]->clone());
    NodeMap[DSG.ShadowNodes[i]] = New;
    ShadowNodes.push_back(New);
  }

  // Clone all of the global nodes...
  for (unsigned i = 0, e = DSG.GlobalNodes.size(); i != e; ++i) {
    GlobalDSNode *New = cast<GlobalDSNode>(DSG.GlobalNodes[i]->clone());
    NodeMap[DSG.GlobalNodes[i]] = New;
    GlobalNodes.push_back(New);
  }

  // Clone all of the call nodes...
  for (unsigned i = 0, e = DSG.CallNodes.size(); i != e; ++i) {
    CallDSNode *New = cast<CallDSNode>(DSG.CallNodes[i]->clone());
    NodeMap[DSG.CallNodes[i]] = New;
    CallNodes.push_back(New);
  }

  // Convert all of the links over in the nodes now that the map has been filled
  // in all the way...
  //
  for (unsigned i = 0, e = DSG.AllocNodes.size(); i != e; ++i)
    AllocNodes[i+StartAllocSize]->mapNode(NodeMap, DSG.AllocNodes[i]);
  for (unsigned i = 0, e = DSG.ShadowNodes.size(); i != e; ++i)
    ShadowNodes[i+StartShadowSize]->mapNode(NodeMap, DSG.ShadowNodes[i]);
  for (unsigned i = 0, e = DSG.GlobalNodes.size(); i != e; ++i)
    GlobalNodes[i+StartGlobalSize]->mapNode(NodeMap, DSG.GlobalNodes[i]);
  for (unsigned i = 0, e = DSG.CallNodes.size(); i != e; ++i)
    CallNodes[i+StartCallSize]->mapNode(NodeMap, DSG.CallNodes[i]);

  // Convert over the arguments...
  Function *OF = DSG.getFunction();
  for (Function::aiterator I = OF->abegin(), E = OF->aend(); I != E; ++I)
    if (isa<PointerType>(I->getType())) {
      PointerValSet ArgPVS;
      assert(DSG.getValueMap().find(I) != DSG.getValueMap().end());
      MapPVS(ArgPVS, DSG.getValueMap().find(I)->second, NodeMap);
      assert(!ArgPVS.empty() && "Argument has no links!");
      Args.push_back(ArgPVS);
    }


  if (CloneValueMap) {
    // Convert value map... the values themselves stay the same, just the nodes
    // have to change...
    //
    for (std::map<Value*,PointerValSet>::const_iterator I =DSG.ValueMap.begin(),
           E = DSG.ValueMap.end(); I != E; ++I)
      MapPVS(ValueMap[I->first], I->second, NodeMap, true);
  }

  // Convert over return node...
  PointerValSet RetVals;
  MapPVS(RetVals, DSG.RetNode, NodeMap);
  return RetVals;
}


FunctionDSGraph::~FunctionDSGraph() {
  RetNode.clear();
  ValueMap.clear();
  for_each(AllocNodes.begin(), AllocNodes.end(),
           std::mem_fun(&DSNode::dropAllReferences));
  for_each(ShadowNodes.begin(), ShadowNodes.end(),
           std::mem_fun(&DSNode::dropAllReferences));
  for_each(GlobalNodes.begin(), GlobalNodes.end(),
           std::mem_fun(&DSNode::dropAllReferences));
  for_each(CallNodes.begin(), CallNodes.end(),
           std::mem_fun(&DSNode::dropAllReferences));
  for_each(AllocNodes.begin(),  AllocNodes.end(),  deleter<DSNode>);
  for_each(ShadowNodes.begin(), ShadowNodes.end(), deleter<DSNode>);
  for_each(GlobalNodes.begin(), GlobalNodes.end(), deleter<DSNode>);
  for_each(CallNodes.begin(),   CallNodes.end(),   deleter<DSNode>);
}

