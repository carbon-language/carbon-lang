//===- NodeImpl.cpp - Implement the data structure analysis nodes ---------===//
//
// Implement the LLVM data structure analysis library.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/DataStructure.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/BasicBlock.h"
#include "llvm/iMemory.h"
#include "llvm/iOther.h"
#include "llvm/Assembly/Writer.h"
#include "Support/STLExtras.h"
#include <algorithm>
#include <sstream>

//===----------------------------------------------------------------------===//
//  DSNode Class Implementation
//

static void MapPVS(PointerValSet &PVSOut, const PointerValSet &PVSIn,
                   map<const DSNode*, DSNode*> &NodeMap) {
  assert(PVSOut.empty() && "Value set already initialized!");

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
  vector<PointerValSet*>::iterator I = std::find(Referrers.begin(),
                                                 Referrers.end(), PVS);
  assert(I != Referrers.end() && "PVS not pointing to node!");
  Referrers.erase(I);
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

void DSNode::print(std::ostream &O) const {
  string Caption = escapeLabel(getCaption());

  O << "\t\tNode" << (void*)this << " [ label =\"{" << Caption;

  const vector<PointerValSet> *Links = getAuxLinks();
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
}

NewDSNode::NewDSNode(AllocationInst *V)
  : DSNode(NewNode, V->getType()->getElementType()), Allocation(V) {
}

string NewDSNode::getCaption() const {
  stringstream OS;
  if (isa<MallocInst>(Allocation))
    OS << "new ";
  else
    OS << "alloca ";

  WriteTypeSymbolic(OS, getType(),
                    Allocation->getParent()->getParent()->getParent());
  if (Allocation->isArrayAllocation())
    OS << "[ ]";
  return OS.str();
}

GlobalDSNode::GlobalDSNode(GlobalValue *V)
  : DSNode(GlobalNode, V->getType()->getElementType()), Val(V) {
}

string GlobalDSNode::getCaption() const {
  stringstream OS;
  WriteTypeSymbolic(OS, getType(), Val->getParent());
  return "global " + OS.str();
}


ShadowDSNode::ShadowDSNode(DSNode *P, Module *M)
  : DSNode(ShadowNode, cast<PointerType>(P->getType())->getElementType()) {
  Parent = P;
  Mod = M;
  ShadowParent = 0;
}

ShadowDSNode::ShadowDSNode(const Type *Ty, Module *M, ShadowDSNode *ShadParent)
  : DSNode(ShadowNode, Ty) {
  Parent = 0;
  Mod = M;
  ShadowParent = ShadParent;
}

std::string ShadowDSNode::getCaption() const {
  stringstream OS;
  WriteTypeSymbolic(OS, getType(), Mod);
  return "shadow " + OS.str();
}

void ShadowDSNode::mapNode(map<const DSNode*, DSNode*> &NodeMap,
                           const DSNode *O) {
  const ShadowDSNode *Old = (ShadowDSNode*)O;
  DSNode::mapNode(NodeMap, Old);  // Map base portions first...

  // Map our SynthNodes...
  assert(SynthNodes.empty() && "Synthnodes already mapped?");
  SynthNodes.reserve(Old->SynthNodes.size());
  for (unsigned i = 0, e = Old->SynthNodes.size(); i != e; ++i)
    SynthNodes.push_back(std::make_pair(Old->SynthNodes[i].first,
                    (ShadowDSNode*)NodeMap[Old->SynthNodes[i].second]));
}


CallDSNode::CallDSNode(CallInst *ci) : DSNode(CallNode, ci->getType()), CI(ci) {
  unsigned NumPtrs = 0;
  if (!isa<Function>(ci->getOperand(0)))
    NumPtrs++;   // Include the method pointer...

  for (unsigned i = 1, e = ci->getNumOperands(); i != e; ++i)
    if (isa<PointerType>(ci->getOperand(i)->getType()))
      NumPtrs++;
  ArgLinks.resize(NumPtrs);
}

string CallDSNode::getCaption() const {
  stringstream OS;
  if (const Function *CM = CI->getCalledFunction())
    OS << "call " << CM->getName();
  else
    OS << "call <indirect>";
  OS << "|Ret: ";
  WriteTypeSymbolic(OS, getType(),
                    CI->getParent()->getParent()->getParent());
  return OS.str();
}

void CallDSNode::mapNode(map<const DSNode*, DSNode*> &NodeMap,
                         const DSNode *O) {
  const CallDSNode *Old = (CallDSNode*)O;
  DSNode::mapNode(NodeMap, Old);  // Map base portions first...

  assert(ArgLinks.size() == Old->ArgLinks.size() && "# Arguments changed!?");
  for (unsigned i = 0, e = Old->ArgLinks.size(); i != e; ++i)
    MapPVS(ArgLinks[i], Old->ArgLinks[i], NodeMap);
}

ArgDSNode::ArgDSNode(FunctionArgument *FA)
  : DSNode(ArgNode, FA->getType()), FuncArg(FA) {
}

string ArgDSNode::getCaption() const {
  stringstream OS;
  OS << "arg %" << FuncArg->getName() << "|Ty: ";
  WriteTypeSymbolic(OS, getType(), FuncArg->getParent()->getParent());
  return OS.str();
}

void FunctionDSGraph::printFunction(std::ostream &O,
                                    const char *Label) const {
  O << "\tsubgraph cluster_" << Label << "_Function" << (void*)this << " {\n";
  O << "\t\tlabel=\"" << Label << " Function\\ " << Func->getName() << "\";\n";
  for (unsigned i = 0, e = Nodes.size(); i != e; ++i)
    Nodes[i]->print(O);
  for (unsigned i = 0, e = ShadowNodes.size(); i != e; ++i)
    ShadowNodes[i]->print(O);

  if (RetNode.size()) {
    O << "\t\tNode" << (void*)this << Label
      << " [shape=\"ellipse\", label=\"Returns\"];\n";
    writeEdges(O, this, Label, -1, RetNode);
  }

  O << "\n";
  for (std::map<Value*, PointerValSet>::const_iterator I = ValueMap.begin(),
         E = ValueMap.end(); I != E; ++I) {
    if (I->second.size()) {  // Only output nodes with edges...
      stringstream OS;
      WriteTypeSymbolic(OS, I->first->getType(), Func->getParent());

      // Create node for I->first
      O << "\t\tNode" << (void*)I->first << Label << " [shape=\"box\", label=\""
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
  RetNode = cloneFunctionIntoSelf(DSG, true);
}


// cloneFunctionIntoSelf - Clone the specified method graph into the current
// method graph, returning the Return's set of the graph.   If ValueMap is set
// to true, the ValueMap of the function is cloned into this function as well
// as the data structure graph itself.
//
PointerValSet FunctionDSGraph::cloneFunctionIntoSelf(const FunctionDSGraph &DSG,
                                                     bool CloneValueMap) {
  map<const DSNode*, DSNode*> NodeMap;  // Map from old graph to new graph...
  unsigned StartSize = Nodes.size();    // We probably already have nodes...
  Nodes.reserve(StartSize+DSG.Nodes.size());
  unsigned StartShadowSize = ShadowNodes.size();
  ShadowNodes.reserve(StartShadowSize+DSG.ShadowNodes.size());

  // Clone all of the nodes, keeping track of the mapping...
  for (unsigned i = 0, e = DSG.Nodes.size(); i != e; ++i)
    Nodes.push_back(NodeMap[DSG.Nodes[i]] = DSG.Nodes[i]->clone());

  // Clone all of the shadow nodes similarly...
  for (unsigned i = 0, e = DSG.ShadowNodes.size(); i != e; ++i)
    ShadowNodes.push_back(cast<ShadowDSNode>(NodeMap[DSG.ShadowNodes[i]] = DSG.ShadowNodes[i]->clone()));


  // Convert all of the links over in the nodes now that the map has been filled
  // in all the way...
  //
  for (unsigned i = 0, e = DSG.Nodes.size(); i != e; ++i)
    Nodes[i+StartSize]->mapNode(NodeMap, DSG.Nodes[i]);

  for (unsigned i = 0, e = DSG.ShadowNodes.size(); i != e; ++i)
    ShadowNodes[i+StartShadowSize]->mapNode(NodeMap, DSG.ShadowNodes[i]);

  if (CloneValueMap) {
    // Convert value map... the values themselves stay the same, just the nodes
    // have to change...
    //
    for (std::map<Value*,PointerValSet>::const_iterator I =DSG.ValueMap.begin(),
           E = DSG.ValueMap.end(); I != E; ++I)
      MapPVS(ValueMap[I->first], I->second, NodeMap);
  }

  // Convert over return node...
  PointerValSet RetVals;
  MapPVS(RetVals, DSG.RetNode, NodeMap);
  return RetVals;
}


FunctionDSGraph::~FunctionDSGraph() {
  RetNode.clear();
  ValueMap.clear();
  for_each(Nodes.begin(), Nodes.end(), mem_fun(&DSNode::dropAllReferences));
  for_each(ShadowNodes.begin(), ShadowNodes.end(),
           mem_fun(&DSNode::dropAllReferences));
  for_each(Nodes.begin(), Nodes.end(), deleter<DSNode>);
  for_each(ShadowNodes.begin(), ShadowNodes.end(), deleter<DSNode>);
}

