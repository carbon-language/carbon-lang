//===- FunctionRepBuilder.h - Structures for graph building ------*- C++ -*--=//
//
// This file defines the FunctionRepBuilder and InitVisitor classes that are
// used to build the local data structure graph for a method.
//
//===----------------------------------------------------------------------===//

#ifndef DATA_STRUCTURE_METHOD_REP_BUILDER_H
#define DATA_STRUCTURE_METHOD_REP_BUILDER_H

#include "llvm/Analysis/DataStructure.h"
#include "llvm/Support/InstVisitor.h"

// DEBUG_DATA_STRUCTURE_CONSTRUCTION - Define this to 1 if you want debug output
//#define DEBUG_DATA_STRUCTURE_CONSTRUCTION 1

class FunctionRepBuilder;

// InitVisitor - Used to initialize the worklists for data structure analysis.
// Iterate over the instructions in the method, creating nodes for malloc and
// call instructions.  Add all uses of these to the worklist of instructions
// to process.
//
class InitVisitor : public InstVisitor<InitVisitor> {
  FunctionRepBuilder *Rep;
  Function *Func;
public:
  InitVisitor(FunctionRepBuilder *R, Function *F) : Rep(R), Func(F) {}

  void visitCallInst(CallInst &CI);
  void visitAllocationInst(AllocationInst &AI);
  void visitInstruction(Instruction &I);

  // visitOperand - If the specified instruction operand is a global value, add
  // a node for it...
  //
  void visitOperand(Value *V);
};


// FunctionRepBuilder - This builder object creates the datastructure graph for
// a method.
//
class FunctionRepBuilder : InstVisitor<FunctionRepBuilder> {
  friend class InitVisitor;
  FunctionDSGraph *F;
  PointerValSet RetNode;

  // ValueMap - Mapping between values we are processing and the possible
  // datastructures that they may point to...
  map<Value*, PointerValSet> ValueMap;

  // CallMap - Keep track of which call nodes correspond to which call insns.
  // The reverse mapping is stored in the CallDSNodes themselves.
  //
  map<CallInst*, CallDSNode*> CallMap;

  // Worklist - Vector of (pointer typed) instructions to process still...
  std::vector<Instruction *> WorkList;

  // Nodes - Keep track of all of the resultant nodes, because there may not
  // be edges connecting these to anything.
  //
  std::vector<AllocDSNode*>  AllocNodes;
  std::vector<ShadowDSNode*> ShadowNodes;
  std::vector<GlobalDSNode*> GlobalNodes;
  std::vector<CallDSNode*>   CallNodes;

  // addAllUsesToWorkList - Add all of the instructions users of the specified
  // value to the work list for further processing...
  //
  void addAllUsesToWorkList(Value *V);

public:
  FunctionRepBuilder(FunctionDSGraph *f) : F(f) {
    initializeWorkList(F->getFunction());
    processWorkList();
  }

  const std::vector<AllocDSNode*>  &getAllocNodes() const { return AllocNodes; }
  const std::vector<ShadowDSNode*> &getShadowNodes() const {return ShadowNodes;}
  const std::vector<GlobalDSNode*> &getGlobalNodes() const {return GlobalNodes;}
  const std::vector<CallDSNode*>   &getCallNodes() const { return CallNodes; }


  ShadowDSNode *makeSynthesizedShadow(const Type *Ty, DSNode *Parent);

  const PointerValSet &getRetNode() const { return RetNode; }

  const map<Value*, PointerValSet> &getValueMap() const { return ValueMap; }
private:
  static PointerVal getIndexedPointerDest(const PointerVal &InP,
                                          const MemAccessInst &MAI);

  void initializeWorkList(Function *Func);
  void processWorkList() {
    // While the worklist still has instructions to process, process them!
    while (!WorkList.empty()) {
      Instruction *I = WorkList.back(); WorkList.pop_back();
#ifdef DEBUG_DATA_STRUCTURE_CONSTRUCTION
      cerr << "Processing worklist inst: " << I;
#endif
    
      visit(*I); // Dispatch to a visitXXX function based on instruction type...
#ifdef DEBUG_DATA_STRUCTURE_CONSTRUCTION
      if (I->hasName() && ValueMap.count(I)) {
        cerr << "Inst %" << I->getName() << " value is:\n";
        ValueMap[I].print(cerr);
      }
#endif
    }
  }

  //===--------------------------------------------------------------------===//
  // Functions used to process the worklist of instructions...
  //
  // Allow the visitor base class to invoke these methods...
  friend class InstVisitor<FunctionRepBuilder>;

  void visitGetElementPtrInst(GetElementPtrInst &GEP);
  void visitReturnInst(ReturnInst &RI);
  void visitLoadInst(LoadInst &LI);
  void visitStoreInst(StoreInst &SI);
  void visitCallInst(CallInst &CI);
  void visitPHINode(PHINode &PN);
  void visitSetCondInst(SetCondInst &SCI) {}  // SetEQ & friends are ignored
  void visitFreeInst(FreeInst &FI) {}         // Ignore free instructions
  void visitInstruction(Instruction &I) {
    std::cerr << "\n\n\nUNKNOWN INSTRUCTION type: " << I << "\n\n\n";
    assert(0 && "Cannot proceed");
  }
};

#endif
