//===- Andersens.cpp - Andersen's Interprocedural Alias Analysis ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines an implementation of Andersen's interprocedural alias
// analysis
//
// In pointer analysis terms, this is a subset-based, flow-insensitive,
// field-sensitive, and context-insensitive algorithm pointer algorithm.
//
// This algorithm is implemented as three stages:
//   1. Object identification.
//   2. Inclusion constraint identification.
//   3. Inclusion constraint solving.
//
// The object identification stage identifies all of the memory objects in the
// program, which includes globals, heap allocated objects, and stack allocated
// objects.
//
// The inclusion constraint identification stage finds all inclusion constraints
// in the program by scanning the program, looking for pointer assignments and
// other statements that effect the points-to graph.  For a statement like "A =
// B", this statement is processed to indicate that A can point to anything that
// B can point to.  Constraints can handle copies, loads, and stores, and
// address taking.
//
// The inclusion constraint solving phase iteratively propagates the inclusion
// constraints until a fixed point is reached.  This is an O(N^3) algorithm.
//
// Function constraints are handled as if they were structs with X fields.
// Thus, an access to argument X of function Y is an access to node index
// getNode(Y) + X.  This representation allows handling of indirect calls
// without any issues.  To wit, an indirect call Y(a,b) is equivalence to
// *(Y + 1) = a, *(Y + 2) = b.
// The return node for a function is always located at getNode(F) +
// CallReturnPos. The arguments start at getNode(F) + CallArgPos.
//
// Future Improvements:
//   Offline variable substitution, offline detection of online
//   cycles.  Use of BDD's.
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "anders-aa"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/InstIterator.h"
#include "llvm/Support/InstVisitor.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/SparseBitVector.h"
#include <algorithm>
#include <set>
#include <list>
#include <stack>
#include <vector>

using namespace llvm;
STATISTIC(NumIters            , "Number of iterations to reach convergence");
STATISTIC(NumConstraints      , "Number of constraints");
STATISTIC(NumNodes            , "Number of nodes");
STATISTIC(NumUnified          , "Number of variables unified");

namespace {
  const unsigned SelfRep = (unsigned)-1;
  const unsigned Unvisited = (unsigned)-1;
  // Position of the function return node relative to the function node.
  const unsigned CallReturnPos = 2;
  // Position of the function call node relative to the function node.
  const unsigned CallFirstArgPos = 3;

  class VISIBILITY_HIDDEN Andersens : public ModulePass, public AliasAnalysis,
                                      private InstVisitor<Andersens> {
    class Node;

    /// Constraint - Objects of this structure are used to represent the various
    /// constraints identified by the algorithm.  The constraints are 'copy',
    /// for statements like "A = B", 'load' for statements like "A = *B",
    /// 'store' for statements like "*A = B", and AddressOf for statements like
    /// A = alloca;  The Offset is applied as *(A + K) = B for stores,
    /// A = *(B + K) for loads, and A = B + K for copies.  It is
    /// illegal on addressof constraints (Because it is statically
    /// resolvable to A = &C where C = B + K)

    struct Constraint {
      enum ConstraintType { Copy, Load, Store, AddressOf } Type;
      unsigned Dest;
      unsigned Src;
      unsigned Offset;

      Constraint(ConstraintType Ty, unsigned D, unsigned S, unsigned O = 0)
        : Type(Ty), Dest(D), Src(S), Offset(O) {
        assert(Offset == 0 || Ty != AddressOf &&
               "Offset is illegal on addressof constraints");
      }
    };

    // Node class - This class is used to represent a node
    // in the constraint graph.  Due to various optimizations,
    // not always the case that there is a mapping from a Node to a
    // Value.  In particular, we add artificial
    // Node's that represent the set of pointed-to variables
    // shared for each location equivalent Node.
    struct Node {
       Value *Val;
      SparseBitVector<> *Edges;
      SparseBitVector<> *PointsTo;
      SparseBitVector<> *OldPointsTo;
      bool Changed;
      std::list<Constraint> Constraints;

      // Nodes in cycles (or in equivalence classes) are united
      // together using a standard union-find representation with path
      // compression.  NodeRep gives the index into GraphNodes
      // representative for this one.
      unsigned NodeRep;    public:

      Node() : Val(0), Edges(0), PointsTo(0), OldPointsTo(0), Changed(false),
               NodeRep(SelfRep) {
      }

      Node *setValue(Value *V) {
        assert(Val == 0 && "Value already set for this node!");
        Val = V;
        return this;
      }

      /// getValue - Return the LLVM value corresponding to this node.
      ///
      Value *getValue() const { return Val; }

      /// addPointerTo - Add a pointer to the list of pointees of this node,
      /// returning true if this caused a new pointer to be added, or false if
      /// we already knew about the points-to relation.
      bool addPointerTo(unsigned Node) {
        return PointsTo->test_and_set(Node);
      }

      /// intersects - Return true if the points-to set of this node intersects
      /// with the points-to set of the specified node.
      bool intersects(Node *N) const;

      /// intersectsIgnoring - Return true if the points-to set of this node
      /// intersects with the points-to set of the specified node on any nodes
      /// except for the specified node to ignore.
      bool intersectsIgnoring(Node *N, unsigned) const;
    };

    /// GraphNodes - This vector is populated as part of the object
    /// identification stage of the analysis, which populates this vector with a
    /// node for each memory object and fills in the ValueNodes map.
    std::vector<Node> GraphNodes;

    /// ValueNodes - This map indicates the Node that a particular Value* is
    /// represented by.  This contains entries for all pointers.
    std::map<Value*, unsigned> ValueNodes;

    /// ObjectNodes - This map contains entries for each memory object in the
    /// program: globals, alloca's and mallocs.
    std::map<Value*, unsigned> ObjectNodes;

    /// ReturnNodes - This map contains an entry for each function in the
    /// program that returns a value.
    std::map<Function*, unsigned> ReturnNodes;

    /// VarargNodes - This map contains the entry used to represent all pointers
    /// passed through the varargs portion of a function call for a particular
    /// function.  An entry is not present in this map for functions that do not
    /// take variable arguments.
    std::map<Function*, unsigned> VarargNodes;


    /// Constraints - This vector contains a list of all of the constraints
    /// identified by the program.
    std::vector<Constraint> Constraints;

    // Map from graph node to maximum K value that is allowed (For functions,
    // this is equivalent to the number of arguments + CallFirstArgPos)
    std::map<unsigned, unsigned> MaxK;

    /// This enum defines the GraphNodes indices that correspond to important
    /// fixed sets.
    enum {
      UniversalSet = 0,
      NullPtr      = 1,
      NullObject   = 2
    };
    // Stack for Tarjans
    std::stack<unsigned> SCCStack;
    // Topological Index -> Graph node
    std::vector<unsigned> Topo2Node;
    // Graph Node -> Topological Index;
    std::vector<unsigned> Node2Topo;
    // Map from Graph Node to DFS number
    std::vector<unsigned> Node2DFS;
    // Map from Graph Node to Deleted from graph.
    std::vector<bool> Node2Deleted;
    // Current DFS and RPO numbers
    unsigned DFSNumber;
    unsigned RPONumber;

  public:
    static char ID;
    Andersens() : ModulePass((intptr_t)&ID) {}

    bool runOnModule(Module &M) {
      InitializeAliasAnalysis(this);
      IdentifyObjects(M);
      CollectConstraints(M);
      DEBUG(PrintConstraints());
      SolveConstraints();
      DEBUG(PrintPointsToGraph());

      // Free the constraints list, as we don't need it to respond to alias
      // requests.
      ObjectNodes.clear();
      ReturnNodes.clear();
      VarargNodes.clear();
      std::vector<Constraint>().swap(Constraints);
      return false;
    }

    void releaseMemory() {
      // FIXME: Until we have transitively required passes working correctly,
      // this cannot be enabled!  Otherwise, using -count-aa with the pass
      // causes memory to be freed too early. :(
#if 0
      // The memory objects and ValueNodes data structures at the only ones that
      // are still live after construction.
      std::vector<Node>().swap(GraphNodes);
      ValueNodes.clear();
#endif
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AliasAnalysis::getAnalysisUsage(AU);
      AU.setPreservesAll();                         // Does not transform code
    }

    //------------------------------------------------
    // Implement the AliasAnalysis API
    //
    AliasResult alias(const Value *V1, unsigned V1Size,
                      const Value *V2, unsigned V2Size);
    virtual ModRefResult getModRefInfo(CallSite CS, Value *P, unsigned Size);
    virtual ModRefResult getModRefInfo(CallSite CS1, CallSite CS2);
    void getMustAliases(Value *P, std::vector<Value*> &RetVals);
    bool pointsToConstantMemory(const Value *P);

    virtual void deleteValue(Value *V) {
      ValueNodes.erase(V);
      getAnalysis<AliasAnalysis>().deleteValue(V);
    }

    virtual void copyValue(Value *From, Value *To) {
      ValueNodes[To] = ValueNodes[From];
      getAnalysis<AliasAnalysis>().copyValue(From, To);
    }

  private:
    /// getNode - Return the node corresponding to the specified pointer scalar.
    ///
    unsigned getNode(Value *V) {
      if (Constant *C = dyn_cast<Constant>(V))
        if (!isa<GlobalValue>(C))
          return getNodeForConstantPointer(C);

      std::map<Value*, unsigned>::iterator I = ValueNodes.find(V);
      if (I == ValueNodes.end()) {
#ifndef NDEBUG
        V->dump();
#endif
        assert(0 && "Value does not have a node in the points-to graph!");
      }
      return I->second;
    }

    /// getObject - Return the node corresponding to the memory object for the
    /// specified global or allocation instruction.
    unsigned getObject(Value *V) {
      std::map<Value*, unsigned>::iterator I = ObjectNodes.find(V);
      assert(I != ObjectNodes.end() &&
             "Value does not have an object in the points-to graph!");
      return I->second;
    }

    /// getReturnNode - Return the node representing the return value for the
    /// specified function.
    unsigned getReturnNode(Function *F) {
      std::map<Function*, unsigned>::iterator I = ReturnNodes.find(F);
      assert(I != ReturnNodes.end() && "Function does not return a value!");
      return I->second;
    }

    /// getVarargNode - Return the node representing the variable arguments
    /// formal for the specified function.
    unsigned getVarargNode(Function *F) {
      std::map<Function*, unsigned>::iterator I = VarargNodes.find(F);
      assert(I != VarargNodes.end() && "Function does not take var args!");
      return I->second;
    }

    /// getNodeValue - Get the node for the specified LLVM value and set the
    /// value for it to be the specified value.
    unsigned getNodeValue(Value &V) {
      unsigned Index = getNode(&V);
      GraphNodes[Index].setValue(&V);
      return Index;
    }

    unsigned UniteNodes(unsigned First, unsigned Second);
    unsigned FindNode(unsigned Node);

    void IdentifyObjects(Module &M);
    void CollectConstraints(Module &M);
    bool AnalyzeUsesOfFunction(Value *);
    void CreateConstraintGraph();
    void SolveConstraints();
    void QueryNode(unsigned Node);

    unsigned getNodeForConstantPointer(Constant *C);
    unsigned getNodeForConstantPointerTarget(Constant *C);
    void AddGlobalInitializerConstraints(unsigned, Constant *C);

    void AddConstraintsForNonInternalLinkage(Function *F);
    void AddConstraintsForCall(CallSite CS, Function *F);
    bool AddConstraintsForExternalCall(CallSite CS, Function *F);


    void PrintNode(Node *N);
    void PrintConstraints();
    void PrintPointsToGraph();

    //===------------------------------------------------------------------===//
    // Instruction visitation methods for adding constraints
    //
    friend class InstVisitor<Andersens>;
    void visitReturnInst(ReturnInst &RI);
    void visitInvokeInst(InvokeInst &II) { visitCallSite(CallSite(&II)); }
    void visitCallInst(CallInst &CI) { visitCallSite(CallSite(&CI)); }
    void visitCallSite(CallSite CS);
    void visitAllocationInst(AllocationInst &AI);
    void visitLoadInst(LoadInst &LI);
    void visitStoreInst(StoreInst &SI);
    void visitGetElementPtrInst(GetElementPtrInst &GEP);
    void visitPHINode(PHINode &PN);
    void visitCastInst(CastInst &CI);
    void visitICmpInst(ICmpInst &ICI) {} // NOOP!
    void visitFCmpInst(FCmpInst &ICI) {} // NOOP!
    void visitSelectInst(SelectInst &SI);
    void visitVAArg(VAArgInst &I);
    void visitInstruction(Instruction &I);

  };

  char Andersens::ID = 0;
  RegisterPass<Andersens> X("anders-aa",
                            "Andersen's Interprocedural Alias Analysis");
  RegisterAnalysisGroup<AliasAnalysis> Y(X);
}

ModulePass *llvm::createAndersensPass() { return new Andersens(); }

//===----------------------------------------------------------------------===//
//                  AliasAnalysis Interface Implementation
//===----------------------------------------------------------------------===//

AliasAnalysis::AliasResult Andersens::alias(const Value *V1, unsigned V1Size,
                                            const Value *V2, unsigned V2Size) {
  Node *N1 = &GraphNodes[FindNode(getNode(const_cast<Value*>(V1)))];
  Node *N2 = &GraphNodes[FindNode(getNode(const_cast<Value*>(V2)))];

  // Check to see if the two pointers are known to not alias.  They don't alias
  // if their points-to sets do not intersect.
  if (!N1->intersectsIgnoring(N2, NullObject))
    return NoAlias;

  return AliasAnalysis::alias(V1, V1Size, V2, V2Size);
}

AliasAnalysis::ModRefResult
Andersens::getModRefInfo(CallSite CS, Value *P, unsigned Size) {
  // The only thing useful that we can contribute for mod/ref information is
  // when calling external function calls: if we know that memory never escapes
  // from the program, it cannot be modified by an external call.
  //
  // NOTE: This is not really safe, at least not when the entire program is not
  // available.  The deal is that the external function could call back into the
  // program and modify stuff.  We ignore this technical niggle for now.  This
  // is, after all, a "research quality" implementation of Andersen's analysis.
  if (Function *F = CS.getCalledFunction())
    if (F->isDeclaration()) {
      Node *N1 = &GraphNodes[FindNode(getNode(P))];

      if (N1->PointsTo->empty())
        return NoModRef;

      if (!N1->PointsTo->test(UniversalSet))
        return NoModRef;  // P doesn't point to the universal set.
    }

  return AliasAnalysis::getModRefInfo(CS, P, Size);
}

AliasAnalysis::ModRefResult
Andersens::getModRefInfo(CallSite CS1, CallSite CS2) {
  return AliasAnalysis::getModRefInfo(CS1,CS2);
}

/// getMustAlias - We can provide must alias information if we know that a
/// pointer can only point to a specific function or the null pointer.
/// Unfortunately we cannot determine must-alias information for global
/// variables or any other memory memory objects because we do not track whether
/// a pointer points to the beginning of an object or a field of it.
void Andersens::getMustAliases(Value *P, std::vector<Value*> &RetVals) {
  Node *N = &GraphNodes[FindNode(getNode(P))];
  if (N->PointsTo->count() == 1) {
    Node *Pointee = &GraphNodes[N->PointsTo->find_first()];
    // If a function is the only object in the points-to set, then it must be
    // the destination.  Note that we can't handle global variables here,
    // because we don't know if the pointer is actually pointing to a field of
    // the global or to the beginning of it.
    if (Value *V = Pointee->getValue()) {
      if (Function *F = dyn_cast<Function>(V))
        RetVals.push_back(F);
    } else {
      // If the object in the points-to set is the null object, then the null
      // pointer is a must alias.
      if (Pointee == &GraphNodes[NullObject])
        RetVals.push_back(Constant::getNullValue(P->getType()));
    }
  }
  AliasAnalysis::getMustAliases(P, RetVals);
}

/// pointsToConstantMemory - If we can determine that this pointer only points
/// to constant memory, return true.  In practice, this means that if the
/// pointer can only point to constant globals, functions, or the null pointer,
/// return true.
///
bool Andersens::pointsToConstantMemory(const Value *P) {
  Node *N = &GraphNodes[FindNode(getNode((Value*)P))];
  unsigned i;

  for (SparseBitVector<>::iterator bi = N->PointsTo->begin();
       bi != N->PointsTo->end();
       ++bi) {
    i = *bi;
    Node *Pointee = &GraphNodes[i];
    if (Value *V = Pointee->getValue()) {
      if (!isa<GlobalValue>(V) || (isa<GlobalVariable>(V) &&
                                   !cast<GlobalVariable>(V)->isConstant()))
        return AliasAnalysis::pointsToConstantMemory(P);
    } else {
      if (i != NullObject)
        return AliasAnalysis::pointsToConstantMemory(P);
    }
  }

  return true;
}

//===----------------------------------------------------------------------===//
//                       Object Identification Phase
//===----------------------------------------------------------------------===//

/// IdentifyObjects - This stage scans the program, adding an entry to the
/// GraphNodes list for each memory object in the program (global stack or
/// heap), and populates the ValueNodes and ObjectNodes maps for these objects.
///
void Andersens::IdentifyObjects(Module &M) {
  unsigned NumObjects = 0;

  // Object #0 is always the universal set: the object that we don't know
  // anything about.
  assert(NumObjects == UniversalSet && "Something changed!");
  ++NumObjects;

  // Object #1 always represents the null pointer.
  assert(NumObjects == NullPtr && "Something changed!");
  ++NumObjects;

  // Object #2 always represents the null object (the object pointed to by null)
  assert(NumObjects == NullObject && "Something changed!");
  ++NumObjects;

  // Add all the globals first.
  for (Module::global_iterator I = M.global_begin(), E = M.global_end();
       I != E; ++I) {
    ObjectNodes[I] = NumObjects++;
    ValueNodes[I] = NumObjects++;
  }

  // Add nodes for all of the functions and the instructions inside of them.
  for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F) {
    // The function itself is a memory object.
    unsigned First = NumObjects;
    ValueNodes[F] = NumObjects++;
    ObjectNodes[F] = NumObjects++;
    if (isa<PointerType>(F->getFunctionType()->getReturnType()))
      ReturnNodes[F] = NumObjects++;
    if (F->getFunctionType()->isVarArg())
      VarargNodes[F] = NumObjects++;


    // Add nodes for all of the incoming pointer arguments.
    for (Function::arg_iterator I = F->arg_begin(), E = F->arg_end();
         I != E; ++I)
      if (isa<PointerType>(I->getType()))
        ValueNodes[I] = NumObjects++;
    MaxK[First] = NumObjects - First;
    MaxK[First + 1] = NumObjects - First - 1;

    // Scan the function body, creating a memory object for each heap/stack
    // allocation in the body of the function and a node to represent all
    // pointer values defined by instructions and used as operands.
    for (inst_iterator II = inst_begin(F), E = inst_end(F); II != E; ++II) {
      // If this is an heap or stack allocation, create a node for the memory
      // object.
      if (isa<PointerType>(II->getType())) {
        ValueNodes[&*II] = NumObjects++;
        if (AllocationInst *AI = dyn_cast<AllocationInst>(&*II))
          ObjectNodes[AI] = NumObjects++;
      }
    }
  }

  // Now that we know how many objects to create, make them all now!
  GraphNodes.resize(NumObjects);
  NumNodes += NumObjects;
}

//===----------------------------------------------------------------------===//
//                     Constraint Identification Phase
//===----------------------------------------------------------------------===//

/// getNodeForConstantPointer - Return the node corresponding to the constant
/// pointer itself.
unsigned Andersens::getNodeForConstantPointer(Constant *C) {
  assert(isa<PointerType>(C->getType()) && "Not a constant pointer!");

  if (isa<ConstantPointerNull>(C) || isa<UndefValue>(C))
    return NullPtr;
  else if (GlobalValue *GV = dyn_cast<GlobalValue>(C))
    return getNode(GV);
  else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
    switch (CE->getOpcode()) {
    case Instruction::GetElementPtr:
      return getNodeForConstantPointer(CE->getOperand(0));
    case Instruction::IntToPtr:
      return UniversalSet;
    case Instruction::BitCast:
      return getNodeForConstantPointer(CE->getOperand(0));
    default:
      cerr << "Constant Expr not yet handled: " << *CE << "\n";
      assert(0);
    }
  } else {
    assert(0 && "Unknown constant pointer!");
  }
  return 0;
}

/// getNodeForConstantPointerTarget - Return the node POINTED TO by the
/// specified constant pointer.
unsigned Andersens::getNodeForConstantPointerTarget(Constant *C) {
  assert(isa<PointerType>(C->getType()) && "Not a constant pointer!");

  if (isa<ConstantPointerNull>(C))
    return NullObject;
  else if (GlobalValue *GV = dyn_cast<GlobalValue>(C))
    return getObject(GV);
  else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
    switch (CE->getOpcode()) {
    case Instruction::GetElementPtr:
      return getNodeForConstantPointerTarget(CE->getOperand(0));
    case Instruction::IntToPtr:
      return UniversalSet;
    case Instruction::BitCast:
      return getNodeForConstantPointerTarget(CE->getOperand(0));
    default:
      cerr << "Constant Expr not yet handled: " << *CE << "\n";
      assert(0);
    }
  } else {
    assert(0 && "Unknown constant pointer!");
  }
  return 0;
}

/// AddGlobalInitializerConstraints - Add inclusion constraints for the memory
/// object N, which contains values indicated by C.
void Andersens::AddGlobalInitializerConstraints(unsigned NodeIndex,
                                                Constant *C) {
  if (C->getType()->isFirstClassType()) {
    if (isa<PointerType>(C->getType()))
      Constraints.push_back(Constraint(Constraint::Copy, NodeIndex,
                                       getNodeForConstantPointer(C)));
  } else if (C->isNullValue()) {
    Constraints.push_back(Constraint(Constraint::Copy, NodeIndex,
                                     NullObject));
    return;
  } else if (!isa<UndefValue>(C)) {
    // If this is an array or struct, include constraints for each element.
    assert(isa<ConstantArray>(C) || isa<ConstantStruct>(C));
    for (unsigned i = 0, e = C->getNumOperands(); i != e; ++i)
      AddGlobalInitializerConstraints(NodeIndex,
                                      cast<Constant>(C->getOperand(i)));
  }
}

/// AddConstraintsForNonInternalLinkage - If this function does not have
/// internal linkage, realize that we can't trust anything passed into or
/// returned by this function.
void Andersens::AddConstraintsForNonInternalLinkage(Function *F) {
  for (Function::arg_iterator I = F->arg_begin(), E = F->arg_end(); I != E; ++I)
    if (isa<PointerType>(I->getType()))
      // If this is an argument of an externally accessible function, the
      // incoming pointer might point to anything.
      Constraints.push_back(Constraint(Constraint::Copy, getNode(I),
                                       UniversalSet));
}

/// AddConstraintsForCall - If this is a call to a "known" function, add the
/// constraints and return true.  If this is a call to an unknown function,
/// return false.
bool Andersens::AddConstraintsForExternalCall(CallSite CS, Function *F) {
  assert(F->isDeclaration() && "Not an external function!");

  // These functions don't induce any points-to constraints.
  if (F->getName() == "atoi" || F->getName() == "atof" ||
      F->getName() == "atol" || F->getName() == "atoll" ||
      F->getName() == "remove" || F->getName() == "unlink" ||
      F->getName() == "rename" || F->getName() == "memcmp" ||
      F->getName() == "llvm.memset.i32" ||
      F->getName() == "llvm.memset.i64" ||
      F->getName() == "strcmp" || F->getName() == "strncmp" ||
      F->getName() == "execl" || F->getName() == "execlp" ||
      F->getName() == "execle" || F->getName() == "execv" ||
      F->getName() == "execvp" || F->getName() == "chmod" ||
      F->getName() == "puts" || F->getName() == "write" ||
      F->getName() == "open" || F->getName() == "create" ||
      F->getName() == "truncate" || F->getName() == "chdir" ||
      F->getName() == "mkdir" || F->getName() == "rmdir" ||
      F->getName() == "read" || F->getName() == "pipe" ||
      F->getName() == "wait" || F->getName() == "time" ||
      F->getName() == "stat" || F->getName() == "fstat" ||
      F->getName() == "lstat" || F->getName() == "strtod" ||
      F->getName() == "strtof" || F->getName() == "strtold" ||
      F->getName() == "fopen" || F->getName() == "fdopen" ||
      F->getName() == "freopen" ||
      F->getName() == "fflush" || F->getName() == "feof" ||
      F->getName() == "fileno" || F->getName() == "clearerr" ||
      F->getName() == "rewind" || F->getName() == "ftell" ||
      F->getName() == "ferror" || F->getName() == "fgetc" ||
      F->getName() == "fgetc" || F->getName() == "_IO_getc" ||
      F->getName() == "fwrite" || F->getName() == "fread" ||
      F->getName() == "fgets" || F->getName() == "ungetc" ||
      F->getName() == "fputc" ||
      F->getName() == "fputs" || F->getName() == "putc" ||
      F->getName() == "ftell" || F->getName() == "rewind" ||
      F->getName() == "_IO_putc" || F->getName() == "fseek" ||
      F->getName() == "fgetpos" || F->getName() == "fsetpos" ||
      F->getName() == "printf" || F->getName() == "fprintf" ||
      F->getName() == "sprintf" || F->getName() == "vprintf" ||
      F->getName() == "vfprintf" || F->getName() == "vsprintf" ||
      F->getName() == "scanf" || F->getName() == "fscanf" ||
      F->getName() == "sscanf" || F->getName() == "__assert_fail" ||
      F->getName() == "modf")
    return true;


  // These functions do induce points-to edges.
  if (F->getName() == "llvm.memcpy.i32" || F->getName() == "llvm.memcpy.i64" ||
      F->getName() == "llvm.memmove.i32" ||F->getName() == "llvm.memmove.i64" ||
      F->getName() == "memmove") {

    // *Dest = *Src, which requires an artificial graph node to represent the
    // constraint.  It is broken up into *Dest = temp, temp = *Src
    unsigned FirstArg = getNode(CS.getArgument(0));
    unsigned SecondArg = getNode(CS.getArgument(1));
    unsigned TempArg = GraphNodes.size();
    GraphNodes.push_back(Node());
    Constraints.push_back(Constraint(Constraint::Store,
                                     FirstArg, TempArg));
    Constraints.push_back(Constraint(Constraint::Load,
                                     TempArg, SecondArg));
    return true;
  }

  // Result = Arg0
  if (F->getName() == "realloc" || F->getName() == "strchr" ||
      F->getName() == "strrchr" || F->getName() == "strstr" ||
      F->getName() == "strtok") {
    Constraints.push_back(Constraint(Constraint::Copy,
                                     getNode(CS.getInstruction()),
                                     getNode(CS.getArgument(0))));
    return true;
  }

  return false;
}



/// AnalyzeUsesOfFunction - Look at all of the users of the specified function.
/// If this is used by anything complex (i.e., the address escapes), return
/// true.
bool Andersens::AnalyzeUsesOfFunction(Value *V) {

  if (!isa<PointerType>(V->getType())) return true;

  for (Value::use_iterator UI = V->use_begin(), E = V->use_end(); UI != E; ++UI)
    if (dyn_cast<LoadInst>(*UI)) {
      return false;
    } else if (StoreInst *SI = dyn_cast<StoreInst>(*UI)) {
      if (V == SI->getOperand(1)) {
        return false;
      } else if (SI->getOperand(1)) {
        return true;  // Storing the pointer
      }
    } else if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(*UI)) {
      if (AnalyzeUsesOfFunction(GEP)) return true;
    } else if (CallInst *CI = dyn_cast<CallInst>(*UI)) {
      // Make sure that this is just the function being called, not that it is
      // passing into the function.
      for (unsigned i = 1, e = CI->getNumOperands(); i != e; ++i)
        if (CI->getOperand(i) == V) return true;
    } else if (InvokeInst *II = dyn_cast<InvokeInst>(*UI)) {
      // Make sure that this is just the function being called, not that it is
      // passing into the function.
      for (unsigned i = 3, e = II->getNumOperands(); i != e; ++i)
        if (II->getOperand(i) == V) return true;
    } else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(*UI)) {
      if (CE->getOpcode() == Instruction::GetElementPtr ||
          CE->getOpcode() == Instruction::BitCast) {
        if (AnalyzeUsesOfFunction(CE))
          return true;
      } else {
        return true;
      }
    } else if (ICmpInst *ICI = dyn_cast<ICmpInst>(*UI)) {
      if (!isa<ConstantPointerNull>(ICI->getOperand(1)))
        return true;  // Allow comparison against null.
    } else if (dyn_cast<FreeInst>(*UI)) {
      return false;
    } else {
      return true;
    }
  return false;
}

/// CollectConstraints - This stage scans the program, adding a constraint to
/// the Constraints list for each instruction in the program that induces a
/// constraint, and setting up the initial points-to graph.
///
void Andersens::CollectConstraints(Module &M) {
  // First, the universal set points to itself.
  Constraints.push_back(Constraint(Constraint::AddressOf, UniversalSet,
                                   UniversalSet));
  Constraints.push_back(Constraint(Constraint::Store, UniversalSet,
                                   UniversalSet));

  // Next, the null pointer points to the null object.
  Constraints.push_back(Constraint(Constraint::AddressOf, NullPtr, NullObject));

  // Next, add any constraints on global variables and their initializers.
  for (Module::global_iterator I = M.global_begin(), E = M.global_end();
       I != E; ++I) {
    // Associate the address of the global object as pointing to the memory for
    // the global: &G = <G memory>
    unsigned ObjectIndex = getObject(I);
    Node *Object = &GraphNodes[ObjectIndex];
    Object->setValue(I);
    Constraints.push_back(Constraint(Constraint::AddressOf, getNodeValue(*I),
                                     ObjectIndex));

    if (I->hasInitializer()) {
      AddGlobalInitializerConstraints(ObjectIndex, I->getInitializer());
    } else {
      // If it doesn't have an initializer (i.e. it's defined in another
      // translation unit), it points to the universal set.
      Constraints.push_back(Constraint(Constraint::Copy, ObjectIndex,
                                       UniversalSet));
    }
  }

  for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F) {
    // Make the function address point to the function object.
    unsigned ObjectIndex = getObject(F);
    GraphNodes[ObjectIndex].setValue(F);
    Constraints.push_back(Constraint(Constraint::AddressOf, getNodeValue(*F),
                                     ObjectIndex));
    // Set up the return value node.
    if (isa<PointerType>(F->getFunctionType()->getReturnType()))
      GraphNodes[getReturnNode(F)].setValue(F);
    if (F->getFunctionType()->isVarArg())
      GraphNodes[getVarargNode(F)].setValue(F);

    // Set up incoming argument nodes.
    for (Function::arg_iterator I = F->arg_begin(), E = F->arg_end();
         I != E; ++I)
      if (isa<PointerType>(I->getType()))
        getNodeValue(*I);

    // At some point we should just add constraints for the escaping functions
    // at solve time, but this slows down solving. For now, we simply mark
    // address taken functions as escaping and treat them as external.
    if (!F->hasInternalLinkage() || AnalyzeUsesOfFunction(F))
      AddConstraintsForNonInternalLinkage(F);

    if (!F->isDeclaration()) {
      // Scan the function body, creating a memory object for each heap/stack
      // allocation in the body of the function and a node to represent all
      // pointer values defined by instructions and used as operands.
      visit(F);
    } else {
      // External functions that return pointers return the universal set.
      if (isa<PointerType>(F->getFunctionType()->getReturnType()))
        Constraints.push_back(Constraint(Constraint::Copy,
                                         getReturnNode(F),
                                         UniversalSet));

      // Any pointers that are passed into the function have the universal set
      // stored into them.
      for (Function::arg_iterator I = F->arg_begin(), E = F->arg_end();
           I != E; ++I)
        if (isa<PointerType>(I->getType())) {
          // Pointers passed into external functions could have anything stored
          // through them.
          Constraints.push_back(Constraint(Constraint::Store, getNode(I),
                                           UniversalSet));
          // Memory objects passed into external function calls can have the
          // universal set point to them.
          Constraints.push_back(Constraint(Constraint::Copy,
                                           UniversalSet,
                                           getNode(I)));
        }

      // If this is an external varargs function, it can also store pointers
      // into any pointers passed through the varargs section.
      if (F->getFunctionType()->isVarArg())
        Constraints.push_back(Constraint(Constraint::Store, getVarargNode(F),
                                         UniversalSet));
    }
  }
  NumConstraints += Constraints.size();
}


void Andersens::visitInstruction(Instruction &I) {
#ifdef NDEBUG
  return;          // This function is just a big assert.
#endif
  if (isa<BinaryOperator>(I))
    return;
  // Most instructions don't have any effect on pointer values.
  switch (I.getOpcode()) {
  case Instruction::Br:
  case Instruction::Switch:
  case Instruction::Unwind:
  case Instruction::Unreachable:
  case Instruction::Free:
  case Instruction::ICmp:
  case Instruction::FCmp:
    return;
  default:
    // Is this something we aren't handling yet?
    cerr << "Unknown instruction: " << I;
    abort();
  }
}

void Andersens::visitAllocationInst(AllocationInst &AI) {
  unsigned ObjectIndex = getObject(&AI);
  GraphNodes[ObjectIndex].setValue(&AI);
  Constraints.push_back(Constraint(Constraint::AddressOf, getNodeValue(AI),
                                   ObjectIndex));
}

void Andersens::visitReturnInst(ReturnInst &RI) {
  if (RI.getNumOperands() && isa<PointerType>(RI.getOperand(0)->getType()))
    // return V   -->   <Copy/retval{F}/v>
    Constraints.push_back(Constraint(Constraint::Copy,
                                     getReturnNode(RI.getParent()->getParent()),
                                     getNode(RI.getOperand(0))));
}

void Andersens::visitLoadInst(LoadInst &LI) {
  if (isa<PointerType>(LI.getType()))
    // P1 = load P2  -->  <Load/P1/P2>
    Constraints.push_back(Constraint(Constraint::Load, getNodeValue(LI),
                                     getNode(LI.getOperand(0))));
}

void Andersens::visitStoreInst(StoreInst &SI) {
  if (isa<PointerType>(SI.getOperand(0)->getType()))
    // store P1, P2  -->  <Store/P2/P1>
    Constraints.push_back(Constraint(Constraint::Store,
                                     getNode(SI.getOperand(1)),
                                     getNode(SI.getOperand(0))));
}

void Andersens::visitGetElementPtrInst(GetElementPtrInst &GEP) {
  // P1 = getelementptr P2, ... --> <Copy/P1/P2>
  Constraints.push_back(Constraint(Constraint::Copy, getNodeValue(GEP),
                                   getNode(GEP.getOperand(0))));
}

void Andersens::visitPHINode(PHINode &PN) {
  if (isa<PointerType>(PN.getType())) {
    unsigned PNN = getNodeValue(PN);
    for (unsigned i = 0, e = PN.getNumIncomingValues(); i != e; ++i)
      // P1 = phi P2, P3  -->  <Copy/P1/P2>, <Copy/P1/P3>, ...
      Constraints.push_back(Constraint(Constraint::Copy, PNN,
                                       getNode(PN.getIncomingValue(i))));
  }
}

void Andersens::visitCastInst(CastInst &CI) {
  Value *Op = CI.getOperand(0);
  if (isa<PointerType>(CI.getType())) {
    if (isa<PointerType>(Op->getType())) {
      // P1 = cast P2  --> <Copy/P1/P2>
      Constraints.push_back(Constraint(Constraint::Copy, getNodeValue(CI),
                                       getNode(CI.getOperand(0))));
    } else {
      // P1 = cast int --> <Copy/P1/Univ>
#if 0
      Constraints.push_back(Constraint(Constraint::Copy, getNodeValue(CI),
                                       UniversalSet));
#else
      getNodeValue(CI);
#endif
    }
  } else if (isa<PointerType>(Op->getType())) {
    // int = cast P1 --> <Copy/Univ/P1>
#if 0
    Constraints.push_back(Constraint(Constraint::Copy,
                                     UniversalSet,
                                     getNode(CI.getOperand(0))));
#else
    getNode(CI.getOperand(0));
#endif
  }
}

void Andersens::visitSelectInst(SelectInst &SI) {
  if (isa<PointerType>(SI.getType())) {
    unsigned SIN = getNodeValue(SI);
    // P1 = select C, P2, P3   ---> <Copy/P1/P2>, <Copy/P1/P3>
    Constraints.push_back(Constraint(Constraint::Copy, SIN,
                                     getNode(SI.getOperand(1))));
    Constraints.push_back(Constraint(Constraint::Copy, SIN,
                                     getNode(SI.getOperand(2))));
  }
}

void Andersens::visitVAArg(VAArgInst &I) {
  assert(0 && "vaarg not handled yet!");
}

/// AddConstraintsForCall - Add constraints for a call with actual arguments
/// specified by CS to the function specified by F.  Note that the types of
/// arguments might not match up in the case where this is an indirect call and
/// the function pointer has been casted.  If this is the case, do something
/// reasonable.
void Andersens::AddConstraintsForCall(CallSite CS, Function *F) {
  Value *CallValue = CS.getCalledValue();
  bool IsDeref = F == NULL;

  // If this is a call to an external function, try to handle it directly to get
  // some taste of context sensitivity.
  if (F && F->isDeclaration() && AddConstraintsForExternalCall(CS, F))
    return;

  if (isa<PointerType>(CS.getType())) {
    unsigned CSN = getNode(CS.getInstruction());
    if (!F || isa<PointerType>(F->getFunctionType()->getReturnType())) {
      if (IsDeref)
        Constraints.push_back(Constraint(Constraint::Load, CSN,
                                         getNode(CallValue), CallReturnPos));
      else
        Constraints.push_back(Constraint(Constraint::Copy, CSN,
                                         getNode(CallValue) + CallReturnPos));
    } else {
      // If the function returns a non-pointer value, handle this just like we
      // treat a nonpointer cast to pointer.
      Constraints.push_back(Constraint(Constraint::Copy, CSN,
                                       UniversalSet));
    }
  } else if (F && isa<PointerType>(F->getFunctionType()->getReturnType())) {
    Constraints.push_back(Constraint(Constraint::Copy,
                                     UniversalSet,
                                     getNode(CallValue) + CallReturnPos));
  }

  CallSite::arg_iterator ArgI = CS.arg_begin(), ArgE = CS.arg_end();
  if (F) {
    // Direct Call
    Function::arg_iterator AI = F->arg_begin(), AE = F->arg_end();
    for (; AI != AE && ArgI != ArgE; ++AI, ++ArgI)
      if (isa<PointerType>(AI->getType())) {
        if (isa<PointerType>((*ArgI)->getType())) {
          // Copy the actual argument into the formal argument.
          Constraints.push_back(Constraint(Constraint::Copy, getNode(AI),
                                           getNode(*ArgI)));
        } else {
          Constraints.push_back(Constraint(Constraint::Copy, getNode(AI),
                                           UniversalSet));
        }
      } else if (isa<PointerType>((*ArgI)->getType())) {
        Constraints.push_back(Constraint(Constraint::Copy,
                                         UniversalSet,
                                         getNode(*ArgI)));
      }
  } else {
    //Indirect Call
    unsigned ArgPos = CallFirstArgPos;
    for (; ArgI != ArgE; ++ArgI) {
      if (isa<PointerType>((*ArgI)->getType())) {
        // Copy the actual argument into the formal argument.
        Constraints.push_back(Constraint(Constraint::Store,
                                         getNode(CallValue),
                                         getNode(*ArgI), ArgPos++));
      } else {
        Constraints.push_back(Constraint(Constraint::Store,
                                         getNode (CallValue),
                                         UniversalSet, ArgPos++));
      }
    }
  }
  // Copy all pointers passed through the varargs section to the varargs node.
  if (F && F->getFunctionType()->isVarArg())
    for (; ArgI != ArgE; ++ArgI)
      if (isa<PointerType>((*ArgI)->getType()))
        Constraints.push_back(Constraint(Constraint::Copy, getVarargNode(F),
                                         getNode(*ArgI)));
  // If more arguments are passed in than we track, just drop them on the floor.
}

void Andersens::visitCallSite(CallSite CS) {
  if (isa<PointerType>(CS.getType()))
    getNodeValue(*CS.getInstruction());

  if (Function *F = CS.getCalledFunction()) {
    AddConstraintsForCall(CS, F);
  } else {
    AddConstraintsForCall(CS, NULL);
  }
}

//===----------------------------------------------------------------------===//
//                         Constraint Solving Phase
//===----------------------------------------------------------------------===//

/// intersects - Return true if the points-to set of this node intersects
/// with the points-to set of the specified node.
bool Andersens::Node::intersects(Node *N) const {
  return PointsTo->intersects(N->PointsTo);
}

/// intersectsIgnoring - Return true if the points-to set of this node
/// intersects with the points-to set of the specified node on any nodes
/// except for the specified node to ignore.
bool Andersens::Node::intersectsIgnoring(Node *N, unsigned Ignoring) const {
  // TODO: If we are only going to call this with the same value for Ignoring,
  // we should move the special values out of the points-to bitmap.
  bool WeHadIt = PointsTo->test(Ignoring);
  bool NHadIt = N->PointsTo->test(Ignoring);
  bool Result = false;
  if (WeHadIt)
    PointsTo->reset(Ignoring);
  if (NHadIt)
    N->PointsTo->reset(Ignoring);
  Result = PointsTo->intersects(N->PointsTo);
  if (WeHadIt)
    PointsTo->set(Ignoring);
  if (NHadIt)
    N->PointsTo->set(Ignoring);
  return Result;
}

// Create the constraint graph used for solving points-to analysis.
//
void Andersens::CreateConstraintGraph() {
  for (unsigned i = 0, e = Constraints.size(); i != e; ++i) {
    Constraint &C = Constraints[i];
    assert (C.Src < GraphNodes.size() && C.Dest < GraphNodes.size());
    if (C.Type == Constraint::AddressOf)
      GraphNodes[C.Dest].PointsTo->set(C.Src);
    else if (C.Type == Constraint::Load)
      GraphNodes[C.Src].Constraints.push_back(C);
    else if (C.Type == Constraint::Store)
      GraphNodes[C.Dest].Constraints.push_back(C);
    else if (C.Offset != 0)
      GraphNodes[C.Src].Constraints.push_back(C);
    else
      GraphNodes[C.Src].Edges->set(C.Dest);
  }
}

// Perform cycle detection, DFS, and RPO finding.
void Andersens::QueryNode(unsigned Node) {
  assert(GraphNodes[Node].NodeRep == SelfRep && "Querying a non-rep node");
  unsigned OurDFS = ++DFSNumber;
  SparseBitVector<> ToErase;
  SparseBitVector<> NewEdges;
  Node2DFS[Node] = OurDFS;

  for (SparseBitVector<>::iterator bi = GraphNodes[Node].Edges->begin();
       bi != GraphNodes[Node].Edges->end();
       ++bi) {
    unsigned RepNode = FindNode(*bi);
    // If we are going to add an edge to repnode, we have no need for the edge
    // to e anymore.
    if (RepNode != *bi && NewEdges.test(RepNode)){
      ToErase.set(*bi);
      continue;
    }

    // Continue about our DFS.
    if (!Node2Deleted[RepNode]){
      if (Node2DFS[RepNode] == 0) {
        QueryNode(RepNode);
        // May have been changed by query
        RepNode = FindNode(RepNode);
      }
      if (Node2DFS[RepNode] < Node2DFS[Node])
        Node2DFS[Node] = Node2DFS[RepNode];
    }
    // We may have just discovered that e belongs to a cycle, in which case we
    // can also erase it.
    if (RepNode != *bi) {
      ToErase.set(*bi);
      NewEdges.set(RepNode);
    }
  }

  GraphNodes[Node].Edges->intersectWithComplement(ToErase);
  GraphNodes[Node].Edges |= NewEdges;

  // If this node is a root of a non-trivial SCC, place it on our worklist to be
  // processed
  if (OurDFS == Node2DFS[Node]) {
    bool Changed = false;
    while (!SCCStack.empty() && Node2DFS[SCCStack.top()] >= OurDFS) {
      Node = UniteNodes(Node, FindNode(SCCStack.top()));

      SCCStack.pop();
      Changed = true;
    }
    Node2Deleted[Node] = true;
    RPONumber++;

    Topo2Node.at(GraphNodes.size() - RPONumber) = Node;
    Node2Topo[Node] = GraphNodes.size() - RPONumber;
    if (Changed)
      GraphNodes[Node].Changed = true;
  } else {
    SCCStack.push(Node);
  }
}


/// SolveConstraints - This stage iteratively processes the constraints list
/// propagating constraints (adding edges to the Nodes in the points-to graph)
/// until a fixed point is reached.
///
void Andersens::SolveConstraints() {
  bool Changed = true;
  unsigned Iteration = 0;

  // We create the bitmaps here to avoid getting jerked around by the compiler
  // creating objects behind our back and wasting lots of memory.
  for (unsigned i = 0; i < GraphNodes.size(); ++i) {
    Node *N = &GraphNodes[i];
    N->PointsTo = new SparseBitVector<>;
    N->OldPointsTo = new SparseBitVector<>;
    N->Edges = new SparseBitVector<>;
  }
  CreateConstraintGraph();

  Topo2Node.insert(Topo2Node.begin(), GraphNodes.size(), Unvisited);
  Node2Topo.insert(Node2Topo.begin(), GraphNodes.size(), Unvisited);
  Node2DFS.insert(Node2DFS.begin(), GraphNodes.size(), 0);
  Node2Deleted.insert(Node2Deleted.begin(), GraphNodes.size(), false);
  DFSNumber = 0;
  RPONumber = 0;
  // Order graph and mark starting nodes as changed.
  for (unsigned i = 0; i < GraphNodes.size(); ++i) {
    unsigned N = FindNode(i);
    Node *INode = &GraphNodes[i];
    if (Node2DFS[N] == 0) {
      QueryNode(N);
      // Mark as changed if it's a representation and can contribute to the
      // calculation right now.
      if (INode->NodeRep == SelfRep && !INode->PointsTo->empty()
          && (!INode->Edges->empty() || !INode->Constraints.empty()))
        INode->Changed = true;
    }
  }

  do {
     Changed = false;

    ++NumIters;
    DOUT << "Starting iteration #" << Iteration++ << "!\n";
    // TODO: In the microoptimization category, we could just make Topo2Node
    // a fast map and thus only contain the visited nodes.
    for (unsigned i = 0; i < GraphNodes.size(); ++i) {
      unsigned CurrNodeIndex = Topo2Node[i];
      Node *CurrNode;

      // We may not revisit all nodes on every iteration
      if (CurrNodeIndex == Unvisited)
        continue;
      CurrNode = &GraphNodes[CurrNodeIndex];
      // See if this is a node we need to process on this iteration
      if (!CurrNode->Changed || CurrNode->NodeRep != SelfRep)
        continue;
      CurrNode->Changed = false;

      // Figure out the changed points to bits
      SparseBitVector<> CurrPointsTo;
      CurrPointsTo.intersectWithComplement(CurrNode->PointsTo,
                                           CurrNode->OldPointsTo);
      if (CurrPointsTo.empty()){
        continue;
      }
      *(CurrNode->OldPointsTo) |= CurrPointsTo;

      /* Now process the constraints for this node.  */
      for (std::list<Constraint>::iterator li = CurrNode->Constraints.begin();
           li != CurrNode->Constraints.end(); ) {
        li->Src = FindNode(li->Src);
        li->Dest = FindNode(li->Dest);

        // TODO: We could delete redundant constraints here.
        // Src and Dest will be the vars we are going to process.
        // This may look a bit ugly, but what it does is allow us to process
        // both store and load constraints with the same function.
        // Load constraints say that every member of our RHS solution has K
        // added to it, and that variable gets an edge to LHS. We also union
        // RHS+K's solution into the LHS solution.
        // Store constraints say that every member of our LHS solution has K
        // added to it, and that variable gets an edge from RHS. We also union
        // RHS's solution into the LHS+K solution.
        unsigned *Src;
        unsigned *Dest;
        unsigned K = li->Offset;
        unsigned CurrMember;
        if (li->Type == Constraint::Load) {
          Src = &CurrMember;
          Dest = &li->Dest;
        } else if (li->Type == Constraint::Store) {
          Src = &li->Src;
          Dest = &CurrMember;
        } else {
          // TODO Handle offseted copy constraint
          li++;
          continue;
        }
        // TODO: hybrid cycle detection would go here, we should check
        // if it was a statically detected offline equivalence that
        // involves pointers , and if so, remove the redundant constraints.

        const SparseBitVector<> &Solution = CurrPointsTo;

        for (SparseBitVector<>::iterator bi = Solution.begin();
             bi != Solution.end();
             ++bi) {
          CurrMember = *bi;

          // Need to increment the member by K since that is where we are
          // supposed to copy to/from
          // Node that in positive weight cycles, which occur in address taking
          // of fields,  K can go past
          // MaxK[CurrMember] elements, even though that is all it could
          // point to.
          if (K > 0 && K > MaxK[CurrMember])
            continue;
          else
            CurrMember = FindNode(CurrMember + K);

          // Add an edge to the graph, so we can just do regular bitmap ior next
          // time.  It may also let us notice a cycle.
          if (!GraphNodes[*Src].Edges->test_and_set(*Dest)) {
            if (GraphNodes[*Dest].PointsTo |= *(GraphNodes[*Src].PointsTo)) {
              GraphNodes[*Dest].Changed = true;
              // If we changed a node we've already processed, we need another
              // iteration.
              if (Node2Topo[*Dest] <= i)
                Changed = true;
            }
          }
        }
        li++;
      }
      SparseBitVector<> NewEdges;
      SparseBitVector<> ToErase;

      // Now all we have left to do is propagate points-to info along the
      // edges, erasing the redundant edges.


      for (SparseBitVector<>::iterator bi = CurrNode->Edges->begin();
           bi != CurrNode->Edges->end();
           ++bi) {

        unsigned DestVar = *bi;
        unsigned Rep = FindNode(DestVar);

        // If we ended up with this node as our destination, or we've already
        // got an edge for the representative, delete the current edge.
        if (Rep == CurrNodeIndex ||
            (Rep != DestVar && NewEdges.test(Rep))) {
          ToErase.set(DestVar);
          continue;
        }
        // Union the points-to sets into the dest
        if (GraphNodes[Rep].PointsTo |= CurrPointsTo) {
          GraphNodes[Rep].Changed = true;
          if (Node2Topo[Rep] <= i)
            Changed = true;
        }
        // If this edge's destination was collapsed, rewrite the edge.
        if (Rep != DestVar) {
          ToErase.set(DestVar);
          NewEdges.set(Rep);
        }
      }
      CurrNode->Edges->intersectWithComplement(ToErase);
      CurrNode->Edges |= NewEdges;
    }
    if (Changed) {
      DFSNumber = RPONumber = 0;
      Node2Deleted.clear();
      Topo2Node.clear();
      Node2Topo.clear();
      Node2DFS.clear();
      Topo2Node.insert(Topo2Node.begin(), GraphNodes.size(), Unvisited);
      Node2Topo.insert(Node2Topo.begin(), GraphNodes.size(), Unvisited);
      Node2DFS.insert(Node2DFS.begin(), GraphNodes.size(), 0);
      Node2Deleted.insert(Node2Deleted.begin(), GraphNodes.size(), false);
      // Rediscover the DFS/Topo ordering, and cycle detect.
      for (unsigned j = 0; j < GraphNodes.size(); j++) {
        unsigned JRep = FindNode(j);
        if (Node2DFS[JRep] == 0)
          QueryNode(JRep);
      }
    }

  } while (Changed);

  Node2Topo.clear();
  Topo2Node.clear();
  Node2DFS.clear();
  Node2Deleted.clear();
  for (unsigned i = 0; i < GraphNodes.size(); ++i) {
    Node *N = &GraphNodes[i];
    delete N->OldPointsTo;
    delete N->Edges;
  }
}

//===----------------------------------------------------------------------===//
//                               Union-Find
//===----------------------------------------------------------------------===//

// Unite nodes First and Second, returning the one which is now the
// representative node.  First and Second are indexes into GraphNodes
unsigned Andersens::UniteNodes(unsigned First, unsigned Second) {
  assert (First < GraphNodes.size() && Second < GraphNodes.size() &&
          "Attempting to merge nodes that don't exist");
  // TODO: implement union by rank
  Node *FirstNode = &GraphNodes[First];
  Node *SecondNode = &GraphNodes[Second];

  assert (SecondNode->NodeRep == SelfRep && FirstNode->NodeRep == SelfRep &&
          "Trying to unite two non-representative nodes!");
  if (First == Second)
    return First;

  SecondNode->NodeRep = First;
  FirstNode->Changed |= SecondNode->Changed;
  FirstNode->PointsTo |= *(SecondNode->PointsTo);
  FirstNode->Edges |= *(SecondNode->Edges);
  FirstNode->Constraints.splice(FirstNode->Constraints.begin(),
                                SecondNode->Constraints);
  delete FirstNode->OldPointsTo;
  FirstNode->OldPointsTo = new SparseBitVector<>;

  // Destroy interesting parts of the merged-from node.
  delete SecondNode->OldPointsTo;
  delete SecondNode->Edges;
  delete SecondNode->PointsTo;
  SecondNode->Edges = NULL;
  SecondNode->PointsTo = NULL;
  SecondNode->OldPointsTo = NULL;

  NumUnified++;
  DOUT << "Unified Node ";
  DEBUG(PrintNode(FirstNode));
  DOUT << " and Node ";
  DEBUG(PrintNode(SecondNode));
  DOUT << "\n";

  // TODO: Handle SDT
  return First;
}

// Find the index into GraphNodes of the node representing Node, performing
// path compression along the way
unsigned Andersens::FindNode(unsigned NodeIndex) {
  assert (NodeIndex < GraphNodes.size()
          && "Attempting to find a node that can't exist");
  Node *N = &GraphNodes[NodeIndex];
  if (N->NodeRep == SelfRep)
    return NodeIndex;
  else
    return (N->NodeRep = FindNode(N->NodeRep));
}

//===----------------------------------------------------------------------===//
//                               Debugging Output
//===----------------------------------------------------------------------===//

void Andersens::PrintNode(Node *N) {
  if (N == &GraphNodes[UniversalSet]) {
    cerr << "<universal>";
    return;
  } else if (N == &GraphNodes[NullPtr]) {
    cerr << "<nullptr>";
    return;
  } else if (N == &GraphNodes[NullObject]) {
    cerr << "<null>";
    return;
  }
  if (!N->getValue()) {
    cerr << "artificial" << (intptr_t) N;
    return;
  }

  assert(N->getValue() != 0 && "Never set node label!");
  Value *V = N->getValue();
  if (Function *F = dyn_cast<Function>(V)) {
    if (isa<PointerType>(F->getFunctionType()->getReturnType()) &&
        N == &GraphNodes[getReturnNode(F)]) {
      cerr << F->getName() << ":retval";
      return;
    } else if (F->getFunctionType()->isVarArg() &&
               N == &GraphNodes[getVarargNode(F)]) {
      cerr << F->getName() << ":vararg";
      return;
    }
  }

  if (Instruction *I = dyn_cast<Instruction>(V))
    cerr << I->getParent()->getParent()->getName() << ":";
  else if (Argument *Arg = dyn_cast<Argument>(V))
    cerr << Arg->getParent()->getName() << ":";

  if (V->hasName())
    cerr << V->getName();
  else
    cerr << "(unnamed)";

  if (isa<GlobalValue>(V) || isa<AllocationInst>(V))
    if (N == &GraphNodes[getObject(V)])
      cerr << "<mem>";
}

void Andersens::PrintConstraints() {
  cerr << "Constraints:\n";

  for (unsigned i = 0, e = Constraints.size(); i != e; ++i) {
    const Constraint &C = Constraints[i];
    if (C.Type == Constraint::Store) {
      cerr << "*";
      if (C.Offset != 0)
        cerr << "(";
    }
    PrintNode(&GraphNodes[C.Dest]);
    if (C.Type == Constraint::Store && C.Offset != 0)
      cerr << " + " << C.Offset << ")";
    cerr << " = ";
    if (C.Type == Constraint::Load) {
      cerr << "*";
      if (C.Offset != 0)
        cerr << "(";
    }
    else if (C.Type == Constraint::AddressOf)
      cerr << "&";
    PrintNode(&GraphNodes[C.Src]);
    if (C.Offset != 0 && C.Type != Constraint::Store)
      cerr << " + " << C.Offset;
    if (C.Type == Constraint::Load && C.Offset != 0)
      cerr << ")";
    cerr << "\n";
  }
}

void Andersens::PrintPointsToGraph() {
  cerr << "Points-to graph:\n";
  for (unsigned i = 0, e = GraphNodes.size(); i != e; ++i) {
    Node *N = &GraphNodes[i];
    if (FindNode (i) != i) {
      PrintNode(N);
      cerr << "\t--> same as ";
      PrintNode(&GraphNodes[FindNode(i)]);
      cerr << "\n";
    } else {
      cerr << "[" << (N->PointsTo->count()) << "] ";
      PrintNode(N);
      cerr << "\t--> ";

      bool first = true;
      for (SparseBitVector<>::iterator bi = N->PointsTo->begin();
           bi != N->PointsTo->end();
           ++bi) {
        if (!first)
          cerr << ", ";
        PrintNode(&GraphNodes[*bi]);
        first = false;
      }
      cerr << "\n";
    }
  }
}
