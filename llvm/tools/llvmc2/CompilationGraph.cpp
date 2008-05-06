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

CompilationGraph::CompilationGraph() {
  NodesMap["root"] = Node(this);
}

Node& CompilationGraph::getNode(const std::string& ToolName) {
  nodes_map_type::iterator I = NodesMap.find(ToolName);
  if (I == NodesMap.end())
    throw std::runtime_error("Node " + ToolName + " is not in graph");
  return I->second;
}

const Node& CompilationGraph::getNode(const std::string& ToolName) const {
  nodes_map_type::const_iterator I = NodesMap.find(ToolName);
  if (I == NodesMap.end())
    throw std::runtime_error("Node " + ToolName + " is not in graph!");
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
  if (!NodesMap.count(V->Name())) {
    Node N;
    N.OwningGraph = this;
    N.ToolPtr = V;
    NodesMap[V->Name()] = N;
  }
}

void CompilationGraph::insertEdge(const std::string& A,
                                  const std::string& B) {
  // TOTHINK: check this at compile-time?
  if (B == "root")
    throw std::runtime_error("Edges back to the root are not allowed!"
                             "Compilation graph should be acyclic!");

  if (A == "root") {
    const Node& N = getNode(B);
    const std::string& InputLanguage = N.ToolPtr->InputLanguage();
    ToolsMap[InputLanguage].push_back(B);

    // Needed to support iteration via GraphTraits.
    NodesMap["root"].AddEdge(new DefaultEdge(B));
  }
  else {
    Node& N = getNode(A);
    // Check that there is a node at B.
    getNode(B);
    N.AddEdge(new DefaultEdge(B));
  }
}

// TOFIX: extend, add an ability to choose between different
// toolchains, support more interesting graph topologies.
int CompilationGraph::Build (const sys::Path& tempDir) const {
  PathVector JoinList;
  const Tool* JoinTool = 0;
  sys::Path In, Out;

  // For each input file
  for (cl::list<std::string>::const_iterator B = InputFilenames.begin(),
        E = InputFilenames.end(); B != E; ++B) {
    In = sys::Path(*B);

    // Get to the head of the toolchain.
    const tools_vector_type& TV = getToolsVector(getLanguage(In));
    if(TV.empty())
      throw std::runtime_error("Tool names vector is empty!");
    const Node* N = &getNode(*TV.begin());

    // Pass it through the chain until we bump into a Join node or a
    // node that says that it is the last.
    bool Last = false;
    while(!Last) {
      const Tool* CurTool = N->ToolPtr.getPtr();

      if(CurTool->IsJoin()) {
        JoinList.push_back(In);
        JoinTool = CurTool;
        break;
      }

      // Is this the last tool?
      if (!N->HasChildren() || CurTool->IsLast()) {
        Out.appendComponent(In.getBasename());
        Out.appendSuffix(CurTool->OutputSuffix());
        Last = true;
      }
      else {
        Out = tempDir;
        Out.appendComponent(In.getBasename());
        Out.appendSuffix(CurTool->OutputSuffix());
        Out.makeUnique(true, NULL);
        Out.eraseFromDisk();
      }

      if (CurTool->GenerateAction(In, Out).Execute() != 0)
        throw std::runtime_error("Tool returned error code!");

      N = &getNode((*N->EdgesBegin())->ToolName());
      In = Out; Out.clear();
    }
  }

  if(JoinTool) {
    // If the final output name is empty, set it to "a.out"
    if (!OutputFilename.empty()) {
      Out = sys::Path(OutputFilename);
    }
    else {
      Out = sys::Path("a");
      Out.appendSuffix(JoinTool->OutputSuffix());
    }

    if (JoinTool->GenerateAction(JoinList, Out).Execute() != 0)
      throw std::runtime_error("Tool returned error code!");
  }

  return 0;
}

namespace llvm {
  template <>
  struct DOTGraphTraits<llvmcc::CompilationGraph*>
    : public DefaultDOTGraphTraits
  {

  template<typename GraphType>
  static std::string getNodeLabel(const Node* N, const GraphType&) {
    if (N->ToolPtr)
      return N->ToolPtr->Name();
    else
      return "root";
  }

  };
}

void CompilationGraph::writeGraph() {
  std::ofstream O("CompilationGraph.dot");

  if(O.good()) {
    llvm::WriteGraph(this, "CompilationGraph");
    O.close();
  }
  else {
    throw std::runtime_error("");
  }
}

void CompilationGraph::viewGraph() {
  llvm::ViewGraph(this, "CompilationGraph");
}
