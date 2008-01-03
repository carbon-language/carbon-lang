//===-- GRConstantPropagation.cpp --------------------------------*- C++ -*-==//
//             
//              [ Constant Propagation via Graph Reachability ]
//   
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This files defines a simple analysis that performs path-sensitive
//  constant propagation within a function.  An example use of this analysis
//  is to perform simple checks for NULL dereferences.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/PathSensitive/SimulGraph.h"
#include "clang/AST/Expr.h"
#include "clang/AST/CFG.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ImmutableMap.h"

using namespace clang;
using llvm::APInt;
using llvm::APFloat;
using llvm::dyn_cast;
using llvm::cast;

//===----------------------------------------------------------------------===//
// ConstV - Represents a variant over APInt, APFloat, and const char
//===----------------------------------------------------------------------===//

namespace {
class ConstV {
  uintptr_t Data;
public:
  enum VariantType { VTString = 0x0, VTObjCString = 0x1,
                     VTFloat  = 0x2, VTInt = 0x3,
                     Flags    = 0x3 };
  
  ConstV(const StringLiteral* v) 
    : Data(reinterpret_cast<uintptr_t>(v) | VTString) {}
  
  ConstV(const ObjCStringLiteral* v)
    : Data(reinterpret_cast<uintptr_t>(v) | VTObjCString) {} 
           
  ConstV(llvm::APInt* v)
    : Data(reinterpret_cast<uintptr_t>(v) | VTInt) {}
  
  ConstV(llvm::APFloat* v)
    : Data(reinterpret_cast<uintptr_t>(v) | VTFloat) {}
  

  inline void* getData() const { return (void*) (Data & ~Flags); }
  inline VariantType getVT() const { return (VariantType) (Data & Flags); } 
  
  inline void Profile(llvm::FoldingSetNodeID& ID) const {
    ID.AddPointer(getData());
  }
};
} // end anonymous namespace

// Overload machinery for casting from ConstV to contained classes.

namespace llvm {

#define CV_OBJ_CAST(CLASS,FLAG)\
template<> inline bool isa<CLASS,ConstV>(const ConstV& V) {\
  return V.getVT() == FLAG;\
}\
\
template <> struct cast_retty_impl<CLASS, ConstV> {\
  typedef const CLASS* ret_type;\
};
  
CV_OBJ_CAST(APInt,ConstV::VTInt)
CV_OBJ_CAST(APFloat,ConstV::VTFloat)
CV_OBJ_CAST(StringLiteral,ConstV::VTString)
CV_OBJ_CAST(ObjCStringLiteral,ConstV::VTObjCString)  

#undef CV_OBJ_CAST
  
template <> struct simplify_type<ConstV> {
  typedef void* SimpleType;
  static SimpleType getSimplifiedValue(const ConstV &Val) { 
    return Val.getData();
  }
};

} // end llvm namespace

//===----------------------------------------------------------------------===//
/// The checker.
//===----------------------------------------------------------------------===//

namespace {
  
  
class GRCP {
  
  //==---------------------------------==//
  //    Type definitions.
  //==---------------------------------==//
  
public:
  typedef llvm::ImmutableMap<Decl*,ConstV> StateTy;
  typedef SimulVertex<StateTy> VertexTy;  
  typedef SimulGraph<VertexTy> GraphTy;
  typedef llvm::SmallVector<Stmt*,20> StmtStackTy;
  typedef llvm::DenseMap<Stmt*,Stmt*> ParentMapTy;
  
  /// DFSWorkList - A nested class that represents a worklist that processes
  ///  vertices in LIFO order.
  class DFSWorkList {
    llvm::SmallVector<VertexTy*,20> Vertices;
  public:
    bool hasWork() const { return !Vertices.empty(); }
    
    /// Enqueue - Add a vertex to the worklist.
    void Enqueue(VertexTy* V) { Vertices.push_back(V); }
    
    /// Dequeue - Remove a vertex from the worklist.
    VertexTy* Dequeue() {
      assert (hasWork());
      VertexTy* V = Vertices.back();
      Vertices.pop_back();
      return V;
    }
  };
  
  //==---------------------------------==//
  //    Data.
  //==---------------------------------==//
  
private:
  CFG& cfg;
  
  /// StateFactory - States are simply maps from Decls to constants.  This
  ///  object is a collection of all the states (immutable maps) that are
  ///  created by the analysis.  This object owns the created maps.
  StateTy::Factory StateFactory;
  
  /// Graph - The simulation graph.  Each vertex is a (location,state) pair.
  GraphTy Graph;
  
  /// ParentMap - A lazily populated map from a Stmt* to its parent Stmt*.
  ParentMapTy ParentMap;
  
  /// StmtStack - A stack of statements/expressions that records the
  ///  statement hierarchy starting from the Stmt* of the last dequeued
  ///  vertex.  Used to lazily populate ParentMap.
  StmtStackTy StmtStack;
  
  /// WorkList - A set of queued vertices that need to be processed by the
  ///  worklist algorithm.
  DFSWorkList WorkList;
  
  //==---------------------------------==//
  //    Edge processing.
  //==---------------------------------==//
  
  void MakeVertex(const ProgramEdge& Loc, StateTy State, VertexTy* PredV) {
    std::pair<VertexTy*,bool> V = Graph.getVertex(Loc,State);
    V.first->addPredecessor(PredV);
    if (V.second) WorkList.Enqueue(V.first);
  }
  
  void MakeVertex(const ProgramEdge& Loc, VertexTy* PredV) {
    MakeVertex(Loc,PredV->getState(),PredV);
  }
  
  void VisitBlkBlk(const BlkBlkEdge& E, VertexTy* PredV);
  void VisitBlkStmt(const BlkStmtEdge& E, VertexTy* PredV);
  void VisitStmtBlk(const StmtBlkEdge& E, VertexTy* PredV);
  
  void ProcessEOP(CFGBlock* Blk, VertexTy* PredV);
  void ProcessStmt(Stmt* S, VertexTy* PredV);
  void ProcessTerminator(Stmt* Terminator ,VertexTy* PredV);

  //==---------------------------------==//
  //    Disable copying.
  //==---------------------------------==//  
  
  GRCP(const GRCP&); // Do not implement.
  GRCP& operator=(const GRCP&);

  //==--------------------------------==//
  //    Public API.
  //==--------------------------------==//    
  
public:
  GRCP(CFG& c);  
  
  /// getGraph - Returns the simulation graph.
  const GraphTy& getGraph() const { return Graph; }
  
  /// ExecuteWorkList - Run the worklist algorithm for a maximum number of
  ///  steps.  Returns true if there is still simulation state on the worklist.
  bool ExecuteWorkList(unsigned Steps = 1000000);  
};
} // end anonymous namespace


//==--------------------------------------------------------==//
//    Public API.
//==--------------------------------------------------------==//

GRCP::GRCP(CFG& c) : cfg(c) {
  // Get the entry block.  Make sure that it has 1 (and only 1) successor.
  CFGBlock* Entry = &c.getEntry();
  
  assert (Entry->empty() && "Entry block must be empty.");
  assert (Entry->succ_size() == 1 && "Entry block must have 1 successor.");
  
  // Get the first (and only) successor of Entry.
  CFGBlock* Succ = *(Entry->succ_begin());
  
  // Construct an edge representing the starting location in the function.
  BlkBlkEdge StartLoc(Entry,Succ);
  
  // Get the vertex.  Make it a root in the graph.
  VertexTy* Root = Graph.getVertex(StartLoc,StateFactory.GetEmptyMap()).first;
  Graph.addRoot(Root);
  
  // Enqueue the root so that it can be processed by the worklist.
  WorkList.Enqueue(Root);
}


bool GRCP::ExecuteWorkList(unsigned Steps) {
  while (Steps && WorkList.hasWork()) {
    --Steps;
    VertexTy* V = WorkList.Dequeue();
    
    // Dispatch on the location type.
    switch (V->getLocation().getKind()) {
      case ProgramEdge::BlkBlk:
        VisitBlkBlk(cast<BlkBlkEdge>(V->getLocation()),V);
        break;
      
      case ProgramEdge::BlkStmt:
        VisitBlkStmt(cast<BlkStmtEdge>(V->getLocation()),V);
        break;
        
      case ProgramEdge::StmtBlk:
        VisitStmtBlk(cast<StmtBlkEdge>(V->getLocation()),V);
        break;
        
      default:
        assert (false && "Unsupported edge type.");
    }
  }
  
  return WorkList.hasWork();
}

//==--------------------------------------------------------==//
//    Edge processing.
//==--------------------------------------------------------==//

void GRCP::VisitBlkBlk(const BlkBlkEdge& E, GRCP::VertexTy* PredV) {
  
  CFGBlock* Blk = E.Dst();
  
  // Check if we are entering the EXIT block.
  if (Blk == &cfg.getExit()) {
    assert (cfg.getExit().size() == 0 && "EXIT block cannot contain Stmts.");
    // Process the End-Of-Path.
    ProcessEOP(Blk, PredV);
    return;
  }
  
  
  // FIXME: we will dispatch to a function that manipulates the state
  //  at the entrance to a block.
  
  if (!Blk->empty()) {
    // If 'Blk' has at least one statement, create a BlkStmtEdge and create
    // the appropriate vertex.  This is the common case.
    MakeVertex(BlkStmtEdge(Blk,Blk->front()), PredV->getState(), PredV);
  }
  else {
    // Otherwise, create a vertex at the BlkStmtEdge right before the terminator
    // (if any) is evaluated.  
    MakeVertex(StmtBlkEdge(NULL,Blk),PredV->getState(), PredV);
  }
}

void GRCP::VisitBlkStmt(const BlkStmtEdge& E, GRCP::VertexTy* PredV) {
  
  if (Stmt* S = E.Dst())
    ProcessStmt(S,PredV);
  else {
    // No statement.  Create an edge right before the terminator is evaluated.
    MakeVertex(StmtBlkEdge(NULL,E.Src()), PredV->getState(), PredV);
  }
}
  
void GRCP::VisitStmtBlk(const StmtBlkEdge& E, GRCP::VertexTy* PredV) {
  CFGBlock* Blk = E.Dst();
  
  if (Stmt* Terminator = Blk->getTerminator())
    ProcessTerminator(Terminator,PredV);
  else {
    // No terminator.  We should have only 1 successor.
    assert (Blk->succ_size() == 1);    
    MakeVertex(BlkBlkEdge(Blk,*(Blk->succ_begin())), PredV);
  }
}

void GRCP::ProcessEOP(CFGBlock* Blk, GRCP::VertexTy* PredV) {
  // FIXME: Perform dispatch to adjust state.
  VertexTy* V = Graph.getVertex(BlkStmtEdge(Blk,NULL), PredV->getState()).first;  
  V->addPredecessor(PredV);
  Graph.addEndOfPath(V);  
}

void GRCP::ProcessStmt(Stmt* S, GRCP::VertexTy* PredV) {
  assert(false && "Not implemented.");
}

void GRCP::ProcessTerminator(Stmt* Terminator,GRCP::VertexTy* PredV) {
  assert(false && "Not implemented.");  
}
