//==-- SimulEngine.h - Local, Path-Sensitive Dataflow Engine ------*- C++ -*--//
//
//                     The LLVM Compiler Infrastructure
//
//        This file is distributed under the University of Illinois 
//        Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the template class SimulEngine, a generic path-sensitive
//  dataflow engine for intra-procedural path-sensitive analyses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_PS_SIMULENGINE
#define LLVM_CLANG_ANALYSIS_PS_SIMULENGINE

#include "clang/Analysis/PathSensitive/SimulVertex.h"
#include "clang/Analysis/PathSensitive/SimulGraph.h"
#include "clang/AST/CFG.h"
#include "llvm/ADT/OwningPtr.h"

namespace clang {
  namespace simwlist {
    class DFS;
    class BFS;
  }


template <typename VertexTy,
          typename TransferFuncs,
          typename WListTy = simwlist::DFS >
class SimulEngine {
public:
  typedef SimulGraph<VertexTy> SimulationGraph;
  typedef WListTy WorkList; 
  
protected:
  /// func - the FunctionDecl for the function being analyzed.  SimulEngine
  ///  does not own this object.
  FunctionDecl* func;
  
  /// cfg - The cfg for the function being analyzed.  Note that SimulEngine
  ///  only conditionally owns the CFG object.  If no CFG is specified during
  ///  the creation of the SimulEngine object, one is created from
  ///  a provided FunctionDecl* representing the function body.  This cfg
  ///  is then owned by SimulEngine.
  CFG* cfg;
  bool OwnsCFG;
  
  /// SGraph - The simulation graph.
  llvm::OwningPtr<SimulationGraph> SGraph;
  
  /// WList - The simulation worklist.
  WorkList WList;

public:
  /// Construct a SimulEngine object from the   
  SimulEngine(FunctionDecl* fd, CFG* c = NULL)
    : func(fd),
      cfg(c ? c : CFG::buildCFG(fd)), OwnsCFG(c == NULL),
      SGraph(new SimulationGraph()) {
    
    assert (fd && "Cannot provide NULL FunctionDecl for analysis!")
    assert (fd->getBody() && "FunctionDecl must have a body!")        
  }
  
  /// execute - Run the simulation.  If the SimulationGraph contains no
  ///  vertices, the simulation starts from the entrace of the function.
  ///  If the worklist is not empty, the simulation resumes from where it
  ///  left off.  Steps specifies the maximum number of simulation steps
  ///  to take, which is roughly the number of statements visited.
  bool execute(unsigned Steps = 100000) {
    
    if (SGraph->getCounter() == 0) {
      assert (WList.empty() &&
              "Simulation graph is empty but the worklist is not!");
      
      // Enqueue roots onto worklist.
      assert (false && "FIXME");
    }
    else if (WList.empty())
      return false;  // Do nothing.  Nothing left to do.
    
    while (Steps-- > 0 && !WList.empty()) {
      VertexTy* V = static_cast<VertexTy*>(WList.dequeue());
      const ProgramEdge& E = V->getEdge();
      
      switch(E.getKind()) {
        // FIXME: need to handle different edges.        
        
      }
    }
      
    return !WList.empty();
  }
  
  /// getGraph - Return the simulation graph.
  SimulationGraph& getGraph() { 
    assert (SGraph && "Cannot return NULL SimulationGraph.")
    return *SGraph;
  }
  
  const SimulationGraph& getGraph() const {
    assert (SGraph && "Cannot return NULL SimulationGraph.")
    return *SGraph;
  } 
  
  /// takeGraph - Return the simulation graph.  Ownership of the graph is
  ///  transferred to the caller, and later calls to getSimGraph() and
  //   takeSimGraph() will fail.
  SimulationGraph* takeGraph() { return SGraph.take(); }
  
  /// getWorkList - Returns the simulation worklist.
  WorkList& getWorkList() { return WList; }
  const WorkList& getWorkList() const { return WList; }
};

} // end clang namespace

#endif
