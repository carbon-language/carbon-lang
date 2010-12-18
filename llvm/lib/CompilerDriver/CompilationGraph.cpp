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

#include "llvm/CompilerDriver/BuiltinOptions.h"
#include "llvm/CompilerDriver/CompilationGraph.h"
#include "llvm/CompilerDriver/Error.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/DOTGraphTraits.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstring>
#include <iterator>
#include <limits>
#include <queue>

using namespace llvm;
using namespace llvmc;

namespace llvmc {

  const std::string* LanguageMap::GetLanguage(const sys::Path& File) const {
    StringRef suf = sys::path::extension(File.str());
    LanguageMap::const_iterator Lang =
      this->find(suf.empty() ? "*empty*" : suf);
    if (Lang == this->end()) {
      PrintError("File '" + File.str() + "' has unknown suffix '"
                 + suf.str() + '\'');
      return 0;
    }
    return &Lang->second;
  }
}

namespace {

  /// ChooseEdge - Return the edge with the maximum weight. Returns 0 on error.
  template <class C>
  const Edge* ChooseEdge(const C& EdgesContainer,
                         const InputLanguagesSet& InLangs,
                         const std::string& NodeName = "root") {
    const Edge* MaxEdge = 0;
    int MaxWeight = 0;
    bool SingleMax = true;

    // TODO: fix calculation of SingleMax.
    for (typename C::const_iterator B = EdgesContainer.begin(),
           E = EdgesContainer.end(); B != E; ++B) {
      const Edge* e = B->getPtr();
      int EW = e->Weight(InLangs);
      if (EW < 0) {
        // (error) invocation in TableGen -> we don't need to print an error
        // message.
        return 0;
      }
      if (EW > MaxWeight) {
        MaxEdge = e;
        MaxWeight = EW;
        SingleMax = true;
      } else if (EW == MaxWeight) {
        SingleMax = false;
      }
    }

    if (!SingleMax) {
      PrintError("Node " + NodeName + ": multiple maximal outward edges found!"
                 " Most probably a specification error.");
      return 0;
    }
    if (!MaxEdge) {
      PrintError("Node " + NodeName + ": no maximal outward edge found!"
                 " Most probably a specification error.");
      return 0;
    }
    return MaxEdge;
  }

}

void Node::AddEdge(Edge* Edg) {
  // If there already was an edge between two nodes, modify it instead
  // of adding a new edge.
  const std::string& ToolName = Edg->ToolName();
  for (container_type::iterator B = OutEdges.begin(), E = OutEdges.end();
       B != E; ++B) {
    if ((*B)->ToolName() == ToolName) {
      llvm::IntrusiveRefCntPtr<Edge>(Edg).swap(*B);
      return;
    }
  }
  OutEdges.push_back(llvm::IntrusiveRefCntPtr<Edge>(Edg));
}

CompilationGraph::CompilationGraph() {
  NodesMap["root"] = Node(this);
}

Node* CompilationGraph::getNode(const std::string& ToolName) {
  nodes_map_type::iterator I = NodesMap.find(ToolName);
  if (I == NodesMap.end()) {
    PrintError("Node " + ToolName + " is not in the graph");
    return 0;
  }
  return &I->second;
}

const Node* CompilationGraph::getNode(const std::string& ToolName) const {
  nodes_map_type::const_iterator I = NodesMap.find(ToolName);
  if (I == NodesMap.end()) {
    PrintError("Node " + ToolName + " is not in the graph!");
    return 0;
  }
  return &I->second;
}

// Find the tools list corresponding to the given language name.
const CompilationGraph::tools_vector_type*
CompilationGraph::getToolsVector(const std::string& LangName) const
{
  tools_map_type::const_iterator I = ToolsMap.find(LangName);
  if (I == ToolsMap.end()) {
    PrintError("No tool corresponding to the language " + LangName + " found");
    return 0;
  }
  return &I->second;
}

void CompilationGraph::insertNode(Tool* V) {
  if (NodesMap.count(V->Name()) == 0)
    NodesMap[V->Name()] = Node(this, V);
}

int CompilationGraph::insertEdge(const std::string& A, Edge* Edg) {
  Node* B = getNode(Edg->ToolName());
  if (B == 0)
    return 1;

  if (A == "root") {
    const char** InLangs = B->ToolPtr->InputLanguages();
    for (;*InLangs; ++InLangs)
      ToolsMap[*InLangs].push_back(IntrusiveRefCntPtr<Edge>(Edg));
    NodesMap["root"].AddEdge(Edg);
  }
  else {
    Node* N = getNode(A);
    if (N == 0)
      return 1;

    N->AddEdge(Edg);
  }
  // Increase the inward edge counter.
  B->IncrInEdges();

  return 0;
}

// Pass input file through the chain until we bump into a Join node or
// a node that says that it is the last.
int CompilationGraph::PassThroughGraph (const sys::Path& InFile,
                                        const Node* StartNode,
                                        const InputLanguagesSet& InLangs,
                                        const sys::Path& TempDir,
                                        const LanguageMap& LangMap) const {
  sys::Path In = InFile;
  const Node* CurNode = StartNode;

  while(true) {
    Tool* CurTool = CurNode->ToolPtr.getPtr();

    if (CurTool->IsJoin()) {
      JoinTool& JT = static_cast<JoinTool&>(*CurTool);
      JT.AddToJoinList(In);
      break;
    }

    Action CurAction;
    if (int ret = CurTool->GenerateAction(CurAction, In, CurNode->HasChildren(),
                                          TempDir, InLangs, LangMap)) {
      return ret;
    }

    if (int ret = CurAction.Execute())
      return ret;

    if (CurAction.StopCompilation())
      return 0;

    const Edge* Edg = ChooseEdge(CurNode->OutEdges, InLangs, CurNode->Name());
    if (Edg == 0)
      return 1;

    CurNode = getNode(Edg->ToolName());
    if (CurNode == 0)
      return 1;

    In = CurAction.OutFile();
  }

  return 0;
}

// Find the head of the toolchain corresponding to the given file.
// Also, insert an input language into InLangs.
const Node* CompilationGraph::
FindToolChain(const sys::Path& In, const std::string* ForceLanguage,
              InputLanguagesSet& InLangs, const LanguageMap& LangMap) const {

  // Determine the input language.
  const std::string* InLang = (ForceLanguage ? ForceLanguage
                               : LangMap.GetLanguage(In));
  if (InLang == 0)
    return 0;
  const std::string& InLanguage = *InLang;

  // Add the current input language to the input language set.
  InLangs.insert(InLanguage);

  // Find the toolchain for the input language.
  const tools_vector_type* pTV = getToolsVector(InLanguage);
  if (pTV == 0)
    return 0;

  const tools_vector_type& TV = *pTV;
  if (TV.empty()) {
    PrintError("No toolchain corresponding to language "
               + InLanguage + " found");
    return 0;
  }

  const Edge* Edg = ChooseEdge(TV, InLangs);
  if (Edg == 0)
    return 0;

  return getNode(Edg->ToolName());
}

// Helper function used by Build().
// Traverses initial portions of the toolchains (up to the first Join node).
// This function is also responsible for handling the -x option.
int CompilationGraph::BuildInitial (InputLanguagesSet& InLangs,
                                    const sys::Path& TempDir,
                                    const LanguageMap& LangMap) {
  // This is related to -x option handling.
  cl::list<std::string>::const_iterator xIter = Languages.begin(),
    xBegin = xIter, xEnd = Languages.end();
  bool xEmpty = true;
  const std::string* xLanguage = 0;
  unsigned xPos = 0, xPosNext = 0, filePos = 0;

  if (xIter != xEnd) {
    xEmpty = false;
    xPos = Languages.getPosition(xIter - xBegin);
    cl::list<std::string>::const_iterator xNext = llvm::next(xIter);
    xPosNext = (xNext == xEnd) ? std::numeric_limits<unsigned>::max()
      : Languages.getPosition(xNext - xBegin);
    xLanguage = (*xIter == "none") ? 0 : &(*xIter);
  }

  // For each input file:
  for (cl::list<std::string>::const_iterator B = InputFilenames.begin(),
         CB = B, E = InputFilenames.end(); B != E; ++B) {
    sys::Path In = sys::Path(*B);

    // Code for handling the -x option.
    // Output: std::string* xLanguage (can be NULL).
    if (!xEmpty) {
      filePos = InputFilenames.getPosition(B - CB);

      if (xPos < filePos) {
        if (filePos < xPosNext) {
          xLanguage = (*xIter == "none") ? 0 : &(*xIter);
        }
        else { // filePos >= xPosNext
          // Skip xIters while filePos > xPosNext
          while (filePos > xPosNext) {
            ++xIter;
            xPos = xPosNext;

            cl::list<std::string>::const_iterator xNext = llvm::next(xIter);
            if (xNext == xEnd)
              xPosNext = std::numeric_limits<unsigned>::max();
            else
              xPosNext = Languages.getPosition(xNext - xBegin);
            xLanguage = (*xIter == "none") ? 0 : &(*xIter);
          }
        }
      }
    }

    // Find the toolchain corresponding to this file.
    const Node* N = FindToolChain(In, xLanguage, InLangs, LangMap);
    if (N == 0)
      return 1;
    // Pass file through the chain starting at head.
    if (int ret = PassThroughGraph(In, N, InLangs, TempDir, LangMap))
      return ret;
  }

  return 0;
}

// Sort the nodes in topological order.
int CompilationGraph::TopologicalSort(std::vector<const Node*>& Out) {
  std::queue<const Node*> Q;

  Node* Root = getNode("root");
  if (Root == 0)
    return 1;

  Q.push(Root);

  while (!Q.empty()) {
    const Node* A = Q.front();
    Q.pop();
    Out.push_back(A);
    for (Node::const_iterator EB = A->EdgesBegin(), EE = A->EdgesEnd();
         EB != EE; ++EB) {
      Node* B = getNode((*EB)->ToolName());
      if (B == 0)
        return 1;

      B->DecrInEdges();
      if (B->HasNoInEdges())
        Q.push(B);
    }
  }

  return 0;
}

namespace {
  bool NotJoinNode(const Node* N) {
    return N->ToolPtr ? !N->ToolPtr->IsJoin() : true;
  }
}

// Call TopologicalSort and filter the resulting list to include
// only Join nodes.
int CompilationGraph::
TopologicalSortFilterJoinNodes(std::vector<const Node*>& Out) {
  std::vector<const Node*> TopSorted;
  if (int ret = TopologicalSort(TopSorted))
    return ret;
  std::remove_copy_if(TopSorted.begin(), TopSorted.end(),
                      std::back_inserter(Out), NotJoinNode);

  return 0;
}

int CompilationGraph::Build (const sys::Path& TempDir,
                             const LanguageMap& LangMap) {
  InputLanguagesSet InLangs;
  bool WasSomeActionGenerated = !InputFilenames.empty();

  // Traverse initial parts of the toolchains and fill in InLangs.
  if (int ret = BuildInitial(InLangs, TempDir, LangMap))
    return ret;

  std::vector<const Node*> JTV;
  if (int ret = TopologicalSortFilterJoinNodes(JTV))
    return ret;

  // For all join nodes in topological order:
  for (std::vector<const Node*>::iterator B = JTV.begin(), E = JTV.end();
       B != E; ++B) {

    const Node* CurNode = *B;
    JoinTool* JT = &static_cast<JoinTool&>(*CurNode->ToolPtr.getPtr());

    // Are there any files in the join list?
    if (JT->JoinListEmpty() && !(JT->WorksOnEmpty() && InputFilenames.empty()))
      continue;

    WasSomeActionGenerated = true;
    Action CurAction;
    if (int ret = JT->GenerateAction(CurAction, CurNode->HasChildren(),
                                     TempDir, InLangs, LangMap)) {
      return ret;
    }

    if (int ret = CurAction.Execute())
      return ret;

    if (CurAction.StopCompilation())
      return 0;

    const Edge* Edg = ChooseEdge(CurNode->OutEdges, InLangs, CurNode->Name());
    if (Edg == 0)
      return 1;

    const Node* NextNode = getNode(Edg->ToolName());
    if (NextNode == 0)
      return 1;

    if (int ret = PassThroughGraph(sys::Path(CurAction.OutFile()), NextNode,
                                   InLangs, TempDir, LangMap)) {
      return ret;
    }
  }

  if (!WasSomeActionGenerated) {
    PrintError("no input files");
    return 1;
  }

  return 0;
}

int CompilationGraph::CheckLanguageNames() const {
  int ret = 0;

  // Check that names for output and input languages on all edges do match.
  for (const_nodes_iterator B = this->NodesMap.begin(),
         E = this->NodesMap.end(); B != E; ++B) {

    const Node & N1 = B->second;
    if (N1.ToolPtr) {
      for (Node::const_iterator EB = N1.EdgesBegin(), EE = N1.EdgesEnd();
           EB != EE; ++EB) {
        const Node* N2 = this->getNode((*EB)->ToolName());
        if (N2 == 0)
          return 1;

        if (!N2->ToolPtr) {
          ++ret;
          errs() << "Error: there is an edge from '" << N1.ToolPtr->Name()
                 << "' back to the root!\n\n";
          continue;
        }

        const char** OutLangs = N1.ToolPtr->OutputLanguages();
        const char** InLangs = N2->ToolPtr->InputLanguages();
        bool eq = false;
        const char* OutLang = 0;
        for (;*OutLangs; ++OutLangs) {
          OutLang = *OutLangs;
          for (;*InLangs; ++InLangs) {
            if (std::strcmp(OutLang, *InLangs) == 0) {
              eq = true;
              break;
            }
          }
        }

        if (!eq) {
          ++ret;
          errs() << "Error: Output->input language mismatch in the edge '"
                 << N1.ToolPtr->Name() << "' -> '" << N2->ToolPtr->Name()
                 << "'!\n"
                 << "Expected one of { ";

          InLangs = N2->ToolPtr->InputLanguages();
          for (;*InLangs; ++InLangs) {
            errs() << '\'' << *InLangs << (*(InLangs+1) ? "', " : "'");
          }

          errs() << " }, but got '" << OutLang << "'!\n\n";
        }

      }
    }
  }

  return ret;
}

int CompilationGraph::CheckMultipleDefaultEdges() const {
  int ret = 0;
  InputLanguagesSet Dummy;

  // For all nodes, just iterate over the outgoing edges and check if there is
  // more than one edge with maximum weight.
  for (const_nodes_iterator B = this->NodesMap.begin(),
         E = this->NodesMap.end(); B != E; ++B) {
    const Node& N = B->second;
    int MaxWeight = -1024;

    // Ignore the root node.
    if (!N.ToolPtr)
      continue;

    for (Node::const_iterator EB = N.EdgesBegin(), EE = N.EdgesEnd();
         EB != EE; ++EB) {
      int EdgeWeight = (*EB)->Weight(Dummy);
      if (EdgeWeight > MaxWeight) {
        MaxWeight = EdgeWeight;
      }
      else if (EdgeWeight == MaxWeight) {
        ++ret;
        errs() << "Error: there are multiple maximal edges stemming from the '"
               << N.ToolPtr->Name() << "' node!\n\n";
        break;
      }
    }
  }

  return ret;
}

int CompilationGraph::CheckCycles() {
  unsigned deleted = 0;
  std::queue<Node*> Q;

  Node* Root = getNode("root");
  if (Root == 0)
    return 1;

  Q.push(Root);

  // Try to delete all nodes that have no ingoing edges, starting from the
  // root. If there are any nodes left after this operation, then we have a
  // cycle. This relies on '--check-graph' not performing the topological sort.
  while (!Q.empty()) {
    Node* A = Q.front();
    Q.pop();
    ++deleted;

    for (Node::iterator EB = A->EdgesBegin(), EE = A->EdgesEnd();
         EB != EE; ++EB) {
      Node* B = getNode((*EB)->ToolName());
      if (B == 0)
        return 1;

      B->DecrInEdges();
      if (B->HasNoInEdges())
        Q.push(B);
    }
  }

  if (deleted != NodesMap.size()) {
    errs() << "Error: there are cycles in the compilation graph!\n"
           << "Try inspecting the diagram produced by "
           << "'llvmc --view-graph'.\n\n";
    return 1;
  }

  return 0;
}

int CompilationGraph::Check () {
  // We try to catch as many errors as we can in one go.
  int errs = 0;
  int ret = 0;

  // Check that output/input language names match.
  ret = this->CheckLanguageNames();
  if (ret < 0)
    return 1;
  errs += ret;

  // Check for multiple default edges.
  ret = this->CheckMultipleDefaultEdges();
  if (ret < 0)
    return 1;
  errs += ret;

  // Check for cycles.
  ret = this->CheckCycles();
  if (ret < 0)
    return 1;
  errs += ret;

  return errs;
}

// Code related to graph visualization.

namespace {

std::string SquashStrArray (const char** StrArr) {
  std::string ret;

  for (; *StrArr; ++StrArr) {
    if (*(StrArr + 1)) {
      ret += *StrArr;
      ret +=  ", ";
    }
    else {
      ret += *StrArr;
    }
  }

  return ret;
}

} // End anonymous namespace.

namespace llvm {
  template <>
  struct DOTGraphTraits<llvmc::CompilationGraph*>
    : public DefaultDOTGraphTraits
  {
    DOTGraphTraits (bool isSimple=false) : DefaultDOTGraphTraits(isSimple) {}

    template<typename GraphType>
    static std::string getNodeLabel(const Node* N, const GraphType&)
    {
      if (N->ToolPtr)
        if (N->ToolPtr->IsJoin())
          return N->Name() + "\n (join" +
            (N->HasChildren() ? ")"
             : std::string(": ") +
             SquashStrArray(N->ToolPtr->OutputLanguages()) + ')');
        else
          return N->Name();
      else
        return "root";
    }

    template<typename EdgeIter>
    static std::string getEdgeSourceLabel(const Node* N, EdgeIter I) {
      if (N->ToolPtr) {
        return SquashStrArray(N->ToolPtr->OutputLanguages());
      }
      else {
        return SquashStrArray(I->ToolPtr->InputLanguages());
      }
    }
  };

} // End namespace llvm

int CompilationGraph::writeGraph(const std::string& OutputFilename) {
  std::string ErrorInfo;
  raw_fd_ostream O(OutputFilename.c_str(), ErrorInfo);

  if (ErrorInfo.empty()) {
    errs() << "Writing '"<< OutputFilename << "' file...";
    llvm::WriteGraph(O, this);
    errs() << "done.\n";
  }
  else {
    PrintError("Error opening file '" + OutputFilename + "' for writing!");
    return 1;
  }

  return 0;
}

void CompilationGraph::viewGraph() {
  llvm::ViewGraph(this, "compilation-graph");
}
