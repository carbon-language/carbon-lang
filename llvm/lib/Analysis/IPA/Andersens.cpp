//===- Andersens.cpp - Andersen's Interprocedural Alias Analysis -----------==//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines a very simple implementation of Andersen's interprocedural
// alias analysis.  This implementation does not include any of the fancy
// features that make Andersen's reasonably efficient (like cycle elimination or
// variable substitution), but it should be useful for getting precision
// numbers and can be extended in the future.
//
// In pointer analysis terms, this is a subset-based, flow-insensitive,
// field-insensitive, and context-insensitive algorithm pointer algorithm.
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
// B can point to.  Constraints can handle copies, loads, and stores.
//
// The inclusion constraint solving phase iteratively propagates the inclusion
// constraints until a fixed point is reached.  This is an O(N^3) algorithm.
//
// In the initial pass, all indirect function calls are completely ignored.  As
// the analysis discovers new targets of function pointers, it iteratively
// resolves a precise (and conservative) call graph.  Also related, this
// analysis initially assumes that all internal functions have known incoming
// pointers.  If we find that an internal function's address escapes outside of
// the program, we update this assumption.
//
// Future Improvements:
//   This implementation of Andersen's algorithm is extremely slow.  To make it
//   scale reasonably well, the inclusion constraints could be sorted (easy), 
//   offline variable substitution would be a huge win (straight-forward), and 
//   online cycle elimination (trickier) might help as well.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "anders-aa"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/InstIterator.h"
#include "llvm/Support/InstVisitor.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "Support/Debug.h"
#include "Support/Statistic.h"
#include <set>
using namespace llvm;

namespace {
  Statistic<>
  NumIters("anders-aa", "Number of iterations to reach convergence");
  Statistic<>
  NumConstraints("anders-aa", "Number of constraints");
  Statistic<>
  NumNodes("anders-aa", "Number of nodes");
  Statistic<>
  NumEscapingFunctions("anders-aa", "Number of internal functions that escape");
  Statistic<>
  NumIndirectCallees("anders-aa", "Number of indirect callees found");

  class Andersens : public Pass, public AliasAnalysis,
                    private InstVisitor<Andersens> {
    /// Node class - This class is used to represent a memory object in the
    /// program, and is the primitive used to build the points-to graph.
    class Node {
      std::vector<Node*> Pointees;
      Value *Val;
    public:
      Node() : Val(0) {}
      Node *setValue(Value *V) {
        assert(Val == 0 && "Value already set for this node!");
        Val = V;
        return this;
      }

      /// getValue - Return the LLVM value corresponding to this node.
      Value *getValue() const { return Val; }

      typedef std::vector<Node*>::const_iterator iterator;
      iterator begin() const { return Pointees.begin(); }
      iterator end() const { return Pointees.end(); }

      /// addPointerTo - Add a pointer to the list of pointees of this node,
      /// returning true if this caused a new pointer to be added, or false if
      /// we already knew about the points-to relation.
      bool addPointerTo(Node *N) {
        std::vector<Node*>::iterator I = std::lower_bound(Pointees.begin(),
                                                          Pointees.end(),
                                                          N);
        if (I != Pointees.end() && *I == N)
          return false;
        Pointees.insert(I, N);
        return true;
      }

      /// intersects - Return true if the points-to set of this node intersects
      /// with the points-to set of the specified node.
      bool intersects(Node *N) const;

      /// intersectsIgnoring - Return true if the points-to set of this node
      /// intersects with the points-to set of the specified node on any nodes
      /// except for the specified node to ignore.
      bool intersectsIgnoring(Node *N, Node *Ignoring) const;

      // Constraint application methods.
      bool copyFrom(Node *N);
      bool loadFrom(Node *N);
      bool storeThrough(Node *N);
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

    /// Constraint - Objects of this structure are used to represent the various
    /// constraints identified by the algorithm.  The constraints are 'copy',
    /// for statements like "A = B", 'load' for statements like "A = *B", and
    /// 'store' for statements like "*A = B".
    struct Constraint {
      enum ConstraintType { Copy, Load, Store } Type;
      Node *Dest, *Src;

      Constraint(ConstraintType Ty, Node *D, Node *S)
        : Type(Ty), Dest(D), Src(S) {}
    };
    
    /// Constraints - This vector contains a list of all of the constraints
    /// identified by the program.
    std::vector<Constraint> Constraints;

    /// EscapingInternalFunctions - This set contains all of the internal
    /// functions that are found to escape from the program.  If the address of
    /// an internal function is passed to an external function or otherwise
    /// escapes from the analyzed portion of the program, we must assume that
    /// any pointer arguments can alias the universal node.  This set keeps
    /// track of those functions we are assuming to escape so far.
    std::set<Function*> EscapingInternalFunctions;

    /// IndirectCalls - This contains a list of all of the indirect call sites
    /// in the program.  Since the call graph is iteratively discovered, we may
    /// need to add constraints to our graph as we find new targets of function
    /// pointers.
    std::vector<CallSite> IndirectCalls;

    /// IndirectCallees - For each call site in the indirect calls list, keep
    /// track of the callees that we have discovered so far.  As the analysis
    /// proceeds, more callees are discovered, until the call graph finally
    /// stabilizes.
    std::map<CallSite, std::vector<Function*> > IndirectCallees;

    /// This enum defines the GraphNodes indices that correspond to important
    /// fixed sets.
    enum {
      UniversalSet = 0,
      NullPtr      = 1,
      NullObject   = 2,
    };
    
  public:
    bool run(Module &M) {
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
      EscapingInternalFunctions.clear();
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
    Node *getNode(Value *V) {
      if (Constant *C = dyn_cast<Constant>(V))
        return getNodeForConstantPointer(C);

      std::map<Value*, unsigned>::iterator I = ValueNodes.find(V);
      if (I == ValueNodes.end()) {
        V->dump();
        assert(I != ValueNodes.end() &&
               "Value does not have a node in the points-to graph!");
      }
      return &GraphNodes[I->second];
    }
    
    /// getObject - Return the node corresponding to the memory object for the
    /// specified global or allocation instruction.
    Node *getObject(Value *V) {
      std::map<Value*, unsigned>::iterator I = ObjectNodes.find(V);
      assert(I != ObjectNodes.end() &&
             "Value does not have an object in the points-to graph!");
      return &GraphNodes[I->second];
    }

    /// getReturnNode - Return the node representing the return value for the
    /// specified function.
    Node *getReturnNode(Function *F) {
      std::map<Function*, unsigned>::iterator I = ReturnNodes.find(F);
      assert(I != ReturnNodes.end() && "Function does not return a value!");
      return &GraphNodes[I->second];
    }

    /// getVarargNode - Return the node representing the variable arguments
    /// formal for the specified function.
    Node *getVarargNode(Function *F) {
      std::map<Function*, unsigned>::iterator I = VarargNodes.find(F);
      assert(I != VarargNodes.end() && "Function does not take var args!");
      return &GraphNodes[I->second];
    }

    /// getNodeValue - Get the node for the specified LLVM value and set the
    /// value for it to be the specified value.
    Node *getNodeValue(Value &V) {
      return getNode(&V)->setValue(&V);
    }

    void IdentifyObjects(Module &M);
    void CollectConstraints(Module &M);
    void SolveConstraints();

    Node *getNodeForConstantPointer(Constant *C);
    Node *getNodeForConstantPointerTarget(Constant *C);
    void AddGlobalInitializerConstraints(Node *N, Constant *C);
    void AddConstraintsForNonInternalLinkage(Function *F);
    void AddConstraintsForCall(CallSite CS, Function *F);


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
    void visitSelectInst(SelectInst &SI);
    void visitVANext(VANextInst &I);
    void visitVAArg(VAArgInst &I);
    void visitInstruction(Instruction &I);
  };

  RegisterOpt<Andersens> X("anders-aa",
                           "Andersen's Interprocedural Alias Analysis");
  RegisterAnalysisGroup<AliasAnalysis, Andersens> Y;
}

//===----------------------------------------------------------------------===//
//                  AliasAnalysis Interface Implementation
//===----------------------------------------------------------------------===//

AliasAnalysis::AliasResult Andersens::alias(const Value *V1, unsigned V1Size,
                                            const Value *V2, unsigned V2Size) {
  Node *N1 = getNode((Value*)V1);
  Node *N2 = getNode((Value*)V2);

  // Check to see if the two pointers are known to not alias.  They don't alias
  // if their points-to sets do not intersect.
  if (!N1->intersectsIgnoring(N2, &GraphNodes[NullObject]))
    return NoAlias;

  return AliasAnalysis::alias(V1, V1Size, V2, V2Size);
}

/// getMustAlias - We can provide must alias information if we know that a
/// pointer can only point to a specific function or the null pointer.
/// Unfortunately we cannot determine must-alias information for global
/// variables or any other memory memory objects because we do not track whether
/// a pointer points to the beginning of an object or a field of it.
void Andersens::getMustAliases(Value *P, std::vector<Value*> &RetVals) {
  Node *N = getNode(P);
  Node::iterator I = N->begin();
  if (I != N->end()) {
    // If there is exactly one element in the points-to set for the object...
    ++I;
    if (I == N->end()) {
      Node *Pointee = *N->begin();

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
  }
  
  AliasAnalysis::getMustAliases(P, RetVals);
}

/// pointsToConstantMemory - If we can determine that this pointer only points
/// to constant memory, return true.  In practice, this means that if the
/// pointer can only point to constant globals, functions, or the null pointer,
/// return true.
///
bool Andersens::pointsToConstantMemory(const Value *P) {
  Node *N = getNode((Value*)P);
  for (Node::iterator I = N->begin(), E = N->end(); I != E; ++I) {
    if (Value *V = (*I)->getValue()) {
      if (!isa<GlobalValue>(V) || (isa<GlobalVariable>(V) &&
                                   !cast<GlobalVariable>(V)->isConstant()))
        return AliasAnalysis::pointsToConstantMemory(P);
    } else {
      if (*I != &GraphNodes[NullObject])
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
  for (Module::giterator I = M.gbegin(), E = M.gend(); I != E; ++I) {
    ObjectNodes[I] = NumObjects++;
    ValueNodes[I] = NumObjects++;
  }

  // Add nodes for all of the functions and the instructions inside of them.
  for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F) {
    // The function itself is a memory object.
    ValueNodes[F] = NumObjects++;
    ObjectNodes[F] = NumObjects++;
    if (isa<PointerType>(F->getFunctionType()->getReturnType()))
      ReturnNodes[F] = NumObjects++;
    if (F->getFunctionType()->isVarArg())
      VarargNodes[F] = NumObjects++;

    // Add nodes for all of the incoming pointer arguments.
    for (Function::aiterator I = F->abegin(), E = F->aend(); I != E; ++I)
      if (isa<PointerType>(I->getType()))
        ValueNodes[I] = NumObjects++;

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
Andersens::Node *Andersens::getNodeForConstantPointer(Constant *C) {
  assert(isa<PointerType>(C->getType()) && "Not a constant pointer!");

  if (isa<ConstantPointerNull>(C))
    return &GraphNodes[NullPtr];
  else if (ConstantPointerRef *CPR = dyn_cast<ConstantPointerRef>(C))
    return getNode(CPR->getValue());
  else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
    switch (CE->getOpcode()) {
    case Instruction::GetElementPtr:
      return getNodeForConstantPointer(CE->getOperand(0));
    case Instruction::Cast:
      if (isa<PointerType>(CE->getOperand(0)->getType()))
        return getNodeForConstantPointer(CE->getOperand(0));
      else
        return &GraphNodes[UniversalSet];
    default:
      std::cerr << "Constant Expr not yet handled: " << *CE << "\n";
      assert(0);
    }
  } else {
    assert(0 && "Unknown constant pointer!");
  }
  return 0;
}

/// getNodeForConstantPointerTarget - Return the node POINTED TO by the
/// specified constant pointer.
Andersens::Node *Andersens::getNodeForConstantPointerTarget(Constant *C) {
  assert(isa<PointerType>(C->getType()) && "Not a constant pointer!");

  if (isa<ConstantPointerNull>(C))
    return &GraphNodes[NullObject];
  else if (ConstantPointerRef *CPR = dyn_cast<ConstantPointerRef>(C))
    return getObject(CPR->getValue());
  else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
    switch (CE->getOpcode()) {
    case Instruction::GetElementPtr:
      return getNodeForConstantPointerTarget(CE->getOperand(0));
    case Instruction::Cast:
      if (isa<PointerType>(CE->getOperand(0)->getType()))
        return getNodeForConstantPointerTarget(CE->getOperand(0));
      else
        return &GraphNodes[UniversalSet];
    default:
      std::cerr << "Constant Expr not yet handled: " << *CE << "\n";
      assert(0);
    }
  } else {
    assert(0 && "Unknown constant pointer!");
  }
  return 0;
}

/// AddGlobalInitializerConstraints - Add inclusion constraints for the memory
/// object N, which contains values indicated by C.
void Andersens::AddGlobalInitializerConstraints(Node *N, Constant *C) {
  if (C->getType()->isFirstClassType()) {
    if (isa<PointerType>(C->getType()))
      N->addPointerTo(getNodeForConstantPointer(C));
  } else if (C->isNullValue()) {
    N->addPointerTo(&GraphNodes[NullObject]);
    return;
  } else {
    // If this is an array or struct, include constraints for each element.
    assert(isa<ConstantArray>(C) || isa<ConstantStruct>(C));
    for (unsigned i = 0, e = C->getNumOperands(); i != e; ++i)
      AddGlobalInitializerConstraints(N, cast<Constant>(C->getOperand(i)));
  }
}

void Andersens::AddConstraintsForNonInternalLinkage(Function *F) {
  for (Function::aiterator I = F->abegin(), E = F->aend(); I != E; ++I)
    if (isa<PointerType>(I->getType()))
      // If this is an argument of an externally accessible function, the
      // incoming pointer might point to anything.
      Constraints.push_back(Constraint(Constraint::Copy, getNode(I),
                                       &GraphNodes[UniversalSet]));
}


/// CollectConstraints - This stage scans the program, adding a constraint to
/// the Constraints list for each instruction in the program that induces a
/// constraint, and setting up the initial points-to graph.
///
void Andersens::CollectConstraints(Module &M) {
  // First, the universal set points to itself.
  GraphNodes[UniversalSet].addPointerTo(&GraphNodes[UniversalSet]);

  // Next, the null pointer points to the null object.
  GraphNodes[NullPtr].addPointerTo(&GraphNodes[NullObject]);

  // Next, add any constraints on global variables and their initializers.
  for (Module::giterator I = M.gbegin(), E = M.gend(); I != E; ++I) {
    // Associate the address of the global object as pointing to the memory for
    // the global: &G = <G memory>
    Node *Object = getObject(I);
    Object->setValue(I);
    getNodeValue(*I)->addPointerTo(Object);

    if (I->hasInitializer()) {
      AddGlobalInitializerConstraints(Object, I->getInitializer());
    } else {
      // If it doesn't have an initializer (i.e. it's defined in another
      // translation unit), it points to the universal set.
      Constraints.push_back(Constraint(Constraint::Copy, Object,
                                       &GraphNodes[UniversalSet]));
    }
  }
  
  for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F) {
    // Make the function address point to the function object.
    getNodeValue(*F)->addPointerTo(getObject(F)->setValue(F));

    // Set up the return value node.
    if (isa<PointerType>(F->getFunctionType()->getReturnType()))
      getReturnNode(F)->setValue(F);
    if (F->getFunctionType()->isVarArg())
      getVarargNode(F)->setValue(F);

    // Set up incoming argument nodes.
    for (Function::aiterator I = F->abegin(), E = F->aend(); I != E; ++I)
      if (isa<PointerType>(I->getType()))
        getNodeValue(*I);

    if (!F->hasInternalLinkage())
      AddConstraintsForNonInternalLinkage(F);

    if (!F->isExternal()) {
      // Scan the function body, creating a memory object for each heap/stack
      // allocation in the body of the function and a node to represent all
      // pointer values defined by instructions and used as operands.
      visit(F);
    } else {
      // External functions that return pointers return the universal set.
      if (isa<PointerType>(F->getFunctionType()->getReturnType()))
        Constraints.push_back(Constraint(Constraint::Copy,
                                         getReturnNode(F),
                                         &GraphNodes[UniversalSet]));

      // Any pointers that are passed into the function have the universal set
      // stored into them.
      for (Function::aiterator I = F->abegin(), E = F->aend(); I != E; ++I)
        if (isa<PointerType>(I->getType())) {
          // Pointers passed into external functions could have anything stored
          // through them.
          Constraints.push_back(Constraint(Constraint::Store, getNode(I),
                                           &GraphNodes[UniversalSet]));
          // Memory objects passed into external function calls can have the
          // universal set point to them.
          Constraints.push_back(Constraint(Constraint::Copy,
                                           &GraphNodes[UniversalSet],
                                           getNode(I)));
        }

      // If this is an external varargs function, it can also store pointers
      // into any pointers passed through the varargs section.
      if (F->getFunctionType()->isVarArg())
        Constraints.push_back(Constraint(Constraint::Store, getVarargNode(F),
                                         &GraphNodes[UniversalSet]));
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
  case Instruction::Free:
  case Instruction::Shl:
  case Instruction::Shr:
    return;
  default:
    // Is this something we aren't handling yet?
    std::cerr << "Unknown instruction: " << I;
    abort();
  }
}

void Andersens::visitAllocationInst(AllocationInst &AI) {
  getNodeValue(AI)->addPointerTo(getObject(&AI)->setValue(&AI));
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
    Node *PNN = getNodeValue(PN);
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
      Constraints.push_back(Constraint(Constraint::Copy, getNodeValue(CI),
                                       &GraphNodes[UniversalSet]));
    }
  } else if (isa<PointerType>(Op->getType())) {
    // int = cast P1 --> <Copy/Univ/P1>
    Constraints.push_back(Constraint(Constraint::Copy,
                                     &GraphNodes[UniversalSet],
                                     getNode(CI.getOperand(0))));
  }
}

void Andersens::visitSelectInst(SelectInst &SI) {
  if (isa<PointerType>(SI.getType())) {
    Node *SIN = getNodeValue(SI);
    // P1 = select C, P2, P3   ---> <Copy/P1/P2>, <Copy/P1/P3>
    Constraints.push_back(Constraint(Constraint::Copy, SIN,
                                     getNode(SI.getOperand(1))));
    Constraints.push_back(Constraint(Constraint::Copy, SIN,
                                     getNode(SI.getOperand(2))));
  }
}

void Andersens::visitVANext(VANextInst &I) {
  // FIXME: Implement
  assert(0 && "vanext not handled yet!");
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
  if (isa<PointerType>(CS.getType())) {
    Node *CSN = getNode(CS.getInstruction());
    if (isa<PointerType>(F->getFunctionType()->getReturnType())) {
      Constraints.push_back(Constraint(Constraint::Copy, CSN,
                                       getReturnNode(F)));
    } else {
      // If the function returns a non-pointer value, handle this just like we
      // treat a nonpointer cast to pointer.
      Constraints.push_back(Constraint(Constraint::Copy, CSN,
                                       &GraphNodes[UniversalSet]));
    }
  } else if (isa<PointerType>(F->getFunctionType()->getReturnType())) {
    Constraints.push_back(Constraint(Constraint::Copy,
                                     &GraphNodes[UniversalSet],
                                     getReturnNode(F)));
  }
  
  Function::aiterator AI = F->abegin(), AE = F->aend();
  CallSite::arg_iterator ArgI = CS.arg_begin(), ArgE = CS.arg_end();
  for (; AI != AE && ArgI != ArgE; ++AI, ++ArgI)
    if (isa<PointerType>(AI->getType())) {
      if (isa<PointerType>((*ArgI)->getType())) {
        // Copy the actual argument into the formal argument.
        Constraints.push_back(Constraint(Constraint::Copy, getNode(AI),
                                         getNode(*ArgI)));
      } else {
        Constraints.push_back(Constraint(Constraint::Copy, getNode(AI),
                                         &GraphNodes[UniversalSet]));
      }
    } else if (isa<PointerType>((*ArgI)->getType())) {
      Constraints.push_back(Constraint(Constraint::Copy,
                                       &GraphNodes[UniversalSet],
                                       getNode(*ArgI)));
    }
  
  // Copy all pointers passed through the varargs section to the varargs node.
  if (F->getFunctionType()->isVarArg())
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
    // We don't handle indirect call sites yet.  Keep track of them for when we
    // discover the call graph incrementally.
    IndirectCalls.push_back(CS);
  }
}

//===----------------------------------------------------------------------===//
//                         Constraint Solving Phase
//===----------------------------------------------------------------------===//

/// intersects - Return true if the points-to set of this node intersects
/// with the points-to set of the specified node.
bool Andersens::Node::intersects(Node *N) const {
  iterator I1 = begin(), I2 = N->begin(), E1 = end(), E2 = N->end();
  while (I1 != E1 && I2 != E2) {
    if (*I1 == *I2) return true;
    if (*I1 < *I2)
      ++I1;
    else
      ++I2;
  }
  return false;
}

/// intersectsIgnoring - Return true if the points-to set of this node
/// intersects with the points-to set of the specified node on any nodes
/// except for the specified node to ignore.
bool Andersens::Node::intersectsIgnoring(Node *N, Node *Ignoring) const {
  iterator I1 = begin(), I2 = N->begin(), E1 = end(), E2 = N->end();
  while (I1 != E1 && I2 != E2) {
    if (*I1 == *I2) {
      if (*I1 != Ignoring) return true;
      ++I1; ++I2;
    } else if (*I1 < *I2)
      ++I1;
    else
      ++I2;
  }
  return false;
}

// Copy constraint: all edges out of the source node get copied to the
// destination node.  This returns true if a change is made.
bool Andersens::Node::copyFrom(Node *N) {
  // Use a mostly linear-time merge since both of the lists are sorted.
  bool Changed = false;
  iterator I = N->begin(), E = N->end();
  unsigned i = 0;
  while (I != E && i != Pointees.size()) {
    if (Pointees[i] < *I) {
      ++i;
    } else if (Pointees[i] == *I) {
      ++i; ++I;
    } else {
      // We found a new element to copy over.
      Changed = true;
      Pointees.insert(Pointees.begin()+i, *I);
       ++i; ++I;
    }
  }

  if (I != E) {
    Pointees.insert(Pointees.end(), I, E);
    Changed = true;
  }

  return Changed;
}

bool Andersens::Node::loadFrom(Node *N) {
  bool Changed = false;
  for (iterator I = N->begin(), E = N->end(); I != E; ++I)
    Changed |= copyFrom(*I);
  return Changed;
}

bool Andersens::Node::storeThrough(Node *N) {
  bool Changed = false;
  for (iterator I = begin(), E = end(); I != E; ++I)
    Changed |= (*I)->copyFrom(N);
  return Changed;
}


/// SolveConstraints - This stage iteratively processes the constraints list
/// propagating constraints (adding edges to the Nodes in the points-to graph)
/// until a fixed point is reached.
///
void Andersens::SolveConstraints() {
  bool Changed = true;
  unsigned Iteration = 0;
  while (Changed) {
    Changed = false;
    ++NumIters;
    DEBUG(std::cerr << "Starting iteration #" << Iteration++ << "!\n");

    // Loop over all of the constraints, applying them in turn.
    for (unsigned i = 0, e = Constraints.size(); i != e; ++i) {
      Constraint &C = Constraints[i];
      switch (C.Type) {
      case Constraint::Copy:
        Changed |= C.Dest->copyFrom(C.Src);
        break;
      case Constraint::Load:
        Changed |= C.Dest->loadFrom(C.Src);
        break;
      case Constraint::Store:
        Changed |= C.Dest->storeThrough(C.Src);
        break;
      default:
        assert(0 && "Unknown constraint!");
      }
    }

    if (Changed) {
      // Check to see if any internal function's addresses have been passed to
      // external functions.  If so, we have to assume that their incoming
      // arguments could be anything.  If there are any internal functions in
      // the universal node that we don't know about, we must iterate.
      for (Node::iterator I = GraphNodes[UniversalSet].begin(),
             E = GraphNodes[UniversalSet].end(); I != E; ++I)
        if (Function *F = dyn_cast_or_null<Function>((*I)->getValue()))
          if (F->hasInternalLinkage() &&
              EscapingInternalFunctions.insert(F).second) {
            // We found a function that is just now escaping.  Mark it as if it
            // didn't have internal linkage.
            AddConstraintsForNonInternalLinkage(F);
            DEBUG(std::cerr << "Found escaping internal function: "
                            << F->getName() << "\n");
            ++NumEscapingFunctions;
          }

      // Check to see if we have discovered any new callees of the indirect call
      // sites.  If so, add constraints to the analysis.
      for (unsigned i = 0, e = IndirectCalls.size(); i != e; ++i) {
        CallSite CS = IndirectCalls[i];
        std::vector<Function*> &KnownCallees = IndirectCallees[CS];
        Node *CN = getNode(CS.getCalledValue());

        for (Node::iterator NI = CN->begin(), E = CN->end(); NI != E; ++NI)
          if (Function *F = dyn_cast_or_null<Function>((*NI)->getValue())) {
            std::vector<Function*>::iterator IP =
              std::lower_bound(KnownCallees.begin(), KnownCallees.end(), F);
            if (IP == KnownCallees.end() || *IP != F) {
              // Add the constraints for the call now.
              AddConstraintsForCall(CS, F);
              DEBUG(std::cerr << "Found actual callee '"
                              << F->getName() << "' for call: "
                              << *CS.getInstruction() << "\n");
              ++NumIndirectCallees;
              KnownCallees.insert(IP, F);
            }
          }
      }
    }
  }
}



//===----------------------------------------------------------------------===//
//                               Debugging Output
//===----------------------------------------------------------------------===//

void Andersens::PrintNode(Node *N) {
  if (N == &GraphNodes[UniversalSet]) {
    std::cerr << "<universal>";
    return;
  } else if (N == &GraphNodes[NullPtr]) {
    std::cerr << "<nullptr>";
    return;
  } else if (N == &GraphNodes[NullObject]) {
    std::cerr << "<null>";
    return;
  }

  assert(N->getValue() != 0 && "Never set node label!");
  Value *V = N->getValue();
  if (Function *F = dyn_cast<Function>(V)) {
    if (isa<PointerType>(F->getFunctionType()->getReturnType()) &&
        N == getReturnNode(F)) {
      std::cerr << F->getName() << ":retval";
      return;
    } else if (F->getFunctionType()->isVarArg() && N == getVarargNode(F)) {
      std::cerr << F->getName() << ":vararg";
      return;
    }
  }

  if (Instruction *I = dyn_cast<Instruction>(V))
    std::cerr << I->getParent()->getParent()->getName() << ":";
  else if (Argument *Arg = dyn_cast<Argument>(V))
    std::cerr << Arg->getParent()->getName() << ":";

  if (V->hasName())
    std::cerr << V->getName();
  else
    std::cerr << "(unnamed)";

  if (isa<GlobalValue>(V) || isa<AllocationInst>(V))
    if (N == getObject(V))
      std::cerr << "<mem>";
}

void Andersens::PrintConstraints() {
  std::cerr << "Constraints:\n";
  for (unsigned i = 0, e = Constraints.size(); i != e; ++i) {
    std::cerr << "  #" << i << ":  ";
    Constraint &C = Constraints[i];
    if (C.Type == Constraint::Store)
      std::cerr << "*";
    PrintNode(C.Dest);
    std::cerr << " = ";
    if (C.Type == Constraint::Load)
      std::cerr << "*";
    PrintNode(C.Src);
    std::cerr << "\n";
  }
}

void Andersens::PrintPointsToGraph() {
  std::cerr << "Points-to graph:\n";
  for (unsigned i = 0, e = GraphNodes.size(); i != e; ++i) {
    Node *N = &GraphNodes[i];
    std::cerr << "[" << (N->end() - N->begin()) << "] ";
    PrintNode(N);
    std::cerr << "\t--> ";
    for (Node::iterator I = N->begin(), E = N->end(); I != E; ++I) {
      if (I != N->begin()) std::cerr << ", ";
      PrintNode(*I);
    }
    std::cerr << "\n";
  }
}
