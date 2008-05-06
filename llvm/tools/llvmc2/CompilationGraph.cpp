//===--- CompilationGraph.cpp - The LLVM Compiler Driver --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open
// Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Compilation graph - implementation.
//
//===----------------------------------------------------------------------===//

#include "CompilationGraph.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DOTGraphTraits.h"
#include "llvm/Support/GraphWriter.h"

#include <stdexcept>

using namespace llvm;
using namespace llvmcc;

extern cl::list<std::string> InputFilenames;
extern cl::opt<std::string> OutputFilename;

namespace {

  // Choose edge that returns
  template <class C>
  const Edge* ChooseEdge(const C& EdgesContainer,
                         const std::string& NodeName = "root") {
    const Edge* DefaultEdge = 0;

    for (typename C::const_iterator B = EdgesContainer.begin(),
           E = EdgesContainer.end(); B != E; ++B) {
      const Edge* E = B->getPtr();

      if (E->isDefault())
        if (!DefaultEdge)
          DefaultEdge = E;
        else
          throw std::runtime_error("Node " + NodeName
                                   + ": multiple default outward edges found!"
                                   "Most probably a specification error.");
      if (E->isEnabled())
        return E;
    }

    if (DefaultEdge)
      return DefaultEdge;
    else
      throw std::runtime_error("Node " + NodeName
                               + ": no default outward edge found!"
                               "Most probably a specification error.");
  }

}

CompilationGraph::CompilationGraph() {
  NodesMap["root"] = Node(this);
}

Node& CompilationGraph::getNode(const std::string& ToolName) {
  nodes_map_type::iterator I = NodesMap.find(ToolName);
  if (I == NodesMap.end())
    throw std::runtime_error("Node " + ToolName + " is not in the graph");
  return I->second;
}

const Node& CompilationGraph::getNode(const std::string& ToolName) const {
  nodes_map_type::const_iterator I = NodesMap.find(ToolName);
  if (I == NodesMap.end())
    throw std::runtime_error("Node " + ToolName + " is not in the graph!");
  return I->second;
}

const std::string& CompilationGraph::getLanguage(const sys::Path& File) const {
  LanguageMap::const_iterator Lang = ExtsToLangs.find(File.getSuffix());
  if (Lang == ExtsToLangs.end())
    throw std::runtime_error("Unknown suffix: " + File.getSuffix() + '!');
  return Lang->second;
}

const CompilationGraph::tools_vector_type&
CompilationGraph::getToolsVector(const std::string& LangName) const
{
  tools_map_type::const_iterator I = ToolsMap.find(LangName);
  if (I == ToolsMap.end())
    throw std::runtime_error("No tools corresponding to " + LangName
                             + " found!");
  return I->second;
}

void CompilationGraph::insertNode(Tool* V) {
  if (NodesMap.count(V->Name()) == 0) {
    Node N;
    N.OwningGraph = this;
    N.ToolPtr = V;
    NodesMap[V->Name()] = N;
  }
}

void CompilationGraph::insertEdge(const std::string& A, Edge* E) {
  Node& B = getNode(E->ToolName());
  if (A == "root") {
    const std::string& InputLanguage = B.ToolPtr->InputLanguage();
    ToolsMap[InputLanguage].push_back(IntrusiveRefCntPtr<Edge>(E));
    NodesMap["root"].AddEdge(E);
  }
  else {
    Node& N = getNode(A);
    N.AddEdge(E);
  }
  // Increase the inward edge counter.
  B.IncrInEdges();
}

// Pass input file through the chain until we bump into a Join node or
// a node that says that it is the last.
const JoinTool*
CompilationGraph::PassThroughGraph (sys::Path& In,
                                    const Node* StartNode,
                                    const sys::Path& TempDir) const {
  bool Last = false;
  const Node* CurNode = StartNode;
  JoinTool* ret = 0;

  while(!Last) {
    sys::Path Out;
    Tool* CurTool = CurNode->ToolPtr.getPtr();

    if (CurTool->IsJoin()) {
      ret = &dynamic_cast<JoinTool&>(*CurTool);
      ret->AddToJoinList(In);
      break;
    }

    // Is this the last tool?
    if (!CurNode->HasChildren() || CurTool->IsLast()) {
      // Check if the first tool is also the last
      if (Out.empty())
        Out.set(In.getBasename());
      else
        Out.appendComponent(In.getBasename());
      Out.appendSuffix(CurTool->OutputSuffix());
      Last = true;
    }
    else {
      Out = TempDir;
      Out.appendComponent(In.getBasename());
      Out.appendSuffix(CurTool->OutputSuffix());
      Out.makeUnique(true, NULL);
      Out.eraseFromDisk();
    }

    if (CurTool->GenerateAction(In, Out).Execute() != 0)
      throw std::runtime_error("Tool returned error code!");

    CurNode = &getNode(ChooseEdge(CurNode->OutEdges,
                                  CurNode->Name())->ToolName());
    In = Out; Out.clear();
  }

  return ret;
}

int CompilationGraph::Build (const sys::Path& TempDir) const {
  const JoinTool* JT = 0;

  // For each input file:
  for (cl::list<std::string>::const_iterator B = InputFilenames.begin(),
        E = InputFilenames.end(); B != E; ++B) {
    sys::Path In = sys::Path(*B);

    // Get to the head of the toolchain.
    const std::string& InLanguage = getLanguage(In);
    const tools_vector_type& TV = getToolsVector(InLanguage);
    if (TV.empty())
      throw std::runtime_error("No toolchain corresponding to language"
                               + InLanguage + " found!");
    const Node* N = &getNode(ChooseEdge(TV)->ToolName());

    // Pass it through the chain starting at head.
    const JoinTool* NewJoin = PassThroughGraph(In, N, TempDir);
    if (JT && NewJoin && JT != NewJoin)
      throw std::runtime_error("Graphs with multiple Join nodes"
                               "are not yet supported!");
    else if (NewJoin)
      JT = NewJoin;
  }

  // For all join nodes in topological order:
  // TOFIX: implement.
  if (JT) {
    sys::Path Out;
    // If the final output name is empty, set it to "a.out"
    if (!OutputFilename.empty()) {
      Out = sys::Path(OutputFilename);
    }
    else {
      Out = sys::Path("a");
      Out.appendSuffix(JT->OutputSuffix());
    }

    if (JT->GenerateAction(Out).Execute() != 0)
      throw std::runtime_error("Tool returned error code!");
  }

  return 0;
}

// Code related to graph visualization.

namespace llvm {
  template <>
  struct DOTGraphTraits<llvmcc::CompilationGraph*>
    : public DefaultDOTGraphTraits
  {

    template<typename GraphType>
    static std::string getNodeLabel(const Node* N, const GraphType&)
    {
      if (N->ToolPtr)
        if (N->ToolPtr->IsJoin())
          return N->Name() + "\n (join" +
            (N->HasChildren() ? ")"
             : std::string(": ") + N->ToolPtr->OutputLanguage() + ')');
        else
          return N->Name();
      else
        return "root";
    }

    template<typename EdgeIter>
    static std::string getEdgeSourceLabel(const Node* N, EdgeIter I) {
      if (N->ToolPtr)
        return N->ToolPtr->OutputLanguage();
      else
        return I->ToolPtr->InputLanguage();
    }
  };

}

void CompilationGraph::writeGraph() {
  std::ofstream O("CompilationGraph.dot");

  if (O.good()) {
    llvm::WriteGraph(this, "CompilationGraph");
    O.close();
  }
  else {
    throw std::runtime_error("");
  }
}

void CompilationGraph::viewGraph() {
  llvm::ViewGraph(this, "compilation-graph");
}
