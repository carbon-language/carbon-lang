//===- FunctionRepBuilder.cpp - Build the local datastructure graph -------===//
//
// Build the local datastructure graph for a single method.
//
//===----------------------------------------------------------------------===//

#include "FunctionRepBuilder.h"
#include "llvm/Function.h"
#include "llvm/BasicBlock.h"
#include "llvm/iMemory.h"
#include "llvm/iPHINode.h"
#include "llvm/iOther.h"
#include "llvm/iTerminators.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Constants.h"
#include "Support/STLExtras.h"
#include <algorithm>

// synthesizeNode - Create a new shadow node that is to be linked into this
// chain..
// FIXME: This should not take a FunctionRepBuilder as an argument!
//
ShadowDSNode *DSNode::synthesizeNode(const Type *Ty,
                                     FunctionRepBuilder *Rep) {
  // If we are a derived shadow node, defer to our parent to synthesize the node
  if (ShadowDSNode *Th = dyn_cast<ShadowDSNode>(this))
    if (Th->getShadowParent())
      return Th->getShadowParent()->synthesizeNode(Ty, Rep);

  // See if we have already synthesized a node of this type...
  for (unsigned i = 0, e = SynthNodes.size(); i != e; ++i)
    if (SynthNodes[i].first == Ty) return SynthNodes[i].second;

  // No we haven't.  Do so now and add it to our list of saved nodes...
  ShadowDSNode *SN = Rep->makeSynthesizedShadow(Ty, this);
  SynthNodes.push_back(make_pair(Ty, SN));
  return SN;
}

ShadowDSNode *FunctionRepBuilder::makeSynthesizedShadow(const Type *Ty,
                                                        DSNode *Parent) {
  ShadowDSNode *Result = new ShadowDSNode(Ty, F->getFunction()->getParent(),
                                          Parent);
  ShadowNodes.push_back(Result);
  return Result;
}



// visitOperand - If the specified instruction operand is a global value, add
// a node for it...
//
void InitVisitor::visitOperand(Value *V) {
  if (!Rep->ValueMap.count(V))                  // Only process it once...
    if (GlobalValue *GV = dyn_cast<GlobalValue>(V)) {
      GlobalDSNode *N = new GlobalDSNode(GV);
      Rep->GlobalNodes.push_back(N);
      Rep->ValueMap[V].add(N);
      Rep->addAllUsesToWorkList(GV);

      // FIXME: If the global variable has fields, we should add critical
      // shadow nodes to represent them!
    }
}


// visitCallInst - Create a call node for the callinst, and create as shadow
// node if the call returns a pointer value.  Check to see if the call node
// uses any global variables...
//
void InitVisitor::visitCallInst(CallInst &CI) {
  CallDSNode *C = new CallDSNode(&CI);
  Rep->CallNodes.push_back(C);
  Rep->CallMap[&CI] = C;
      
  if (const PointerType *PT = dyn_cast<PointerType>(CI.getType())) {
    // Create a critical shadow node to represent the memory object that the
    // return value points to...
    ShadowDSNode *Shad = new ShadowDSNode(PT->getElementType(),
                                          Func->getParent());
    Rep->ShadowNodes.push_back(Shad);
    
    // The return value of the function is a pointer to the shadow value
    // just created...
    //
    C->getLink(0).add(Shad);

    // The call instruction returns a pointer to the shadow block...
    Rep->ValueMap[&CI].add(Shad, &CI);
    
    // If the call returns a value with pointer type, add all of the users
    // of the call instruction to the work list...
    Rep->addAllUsesToWorkList(&CI);
  }

  // Loop over all of the operands of the call instruction (except the first
  // one), to look for global variable references...
  //
  for_each(CI.op_begin(), CI.op_end(),
           bind_obj(this, &InitVisitor::visitOperand));
}


// visitAllocationInst - Create an allocation node for the allocation.  Since
// allocation instructions do not take pointer arguments, they cannot refer to
// global vars...
//
void InitVisitor::visitAllocationInst(AllocationInst &AI) {
  AllocDSNode *N = new AllocDSNode(&AI);
  Rep->AllocNodes.push_back(N);
  
  Rep->ValueMap[&AI].add(N, &AI);
  
  // Add all of the users of the malloc instruction to the work list...
  Rep->addAllUsesToWorkList(&AI);
}


// Visit all other instruction types.  Here we just scan, looking for uses of
// global variables...
//
void InitVisitor::visitInstruction(Instruction &I) {
  for_each(I.op_begin(), I.op_end(),
           bind_obj(this, &InitVisitor::visitOperand));
}


// addAllUsesToWorkList - Add all of the instructions users of the specified
// value to the work list for further processing...
//
void FunctionRepBuilder::addAllUsesToWorkList(Value *V) {
  //cerr << "Adding all uses of " << V << "\n";
  for (Value::use_iterator I = V->use_begin(), E = V->use_end(); I != E; ++I) {
    Instruction *Inst = cast<Instruction>(*I);
    // When processing global values, it's possible that the instructions on
    // the use list are not all in this method.  Only add the instructions
    // that _are_ in this method.
    //
    if (Inst->getParent()->getParent() == F->getFunction())
      // Only let an instruction occur on the work list once...
      if (std::find(WorkList.begin(), WorkList.end(), Inst) == WorkList.end())
        WorkList.push_back(Inst);
  }
}




void FunctionRepBuilder::initializeWorkList(Function *Func) {
  // Add all of the arguments to the method to the graph and add all users to
  // the worklists...
  //
  for (Function::aiterator I = Func->abegin(), E = Func->aend(); I != E; ++I) {
    // Only process arguments that are of pointer type...
    if (const PointerType *PT = dyn_cast<PointerType>(I->getType())) {
      // Add a shadow value for it to represent what it is pointing to and add
      // this to the value map...
      ShadowDSNode *Shad = new ShadowDSNode(PT->getElementType(),
                                            Func->getParent());
      ShadowNodes.push_back(Shad);
      ValueMap[I].add(PointerVal(Shad), I);
      
      // Make sure that all users of the argument are processed...
      addAllUsesToWorkList(I);
    }
  }

  // Iterate over the instructions in the method.  Create nodes for malloc and
  // call instructions.  Add all uses of these to the worklist of instructions
  // to process.
  //
  InitVisitor IV(this, Func);
  IV.visit(Func);
}




PointerVal FunctionRepBuilder::getIndexedPointerDest(const PointerVal &InP,
                                                     const MemAccessInst &MAI) {
  unsigned Index = InP.Index;
  const Type *SrcTy = MAI.getPointerOperand()->getType();

  for (MemAccessInst::const_op_iterator I = MAI.idx_begin(),
         E = MAI.idx_end(); I != E; ++I)
    if ((*I)->getType() == Type::UByteTy) {     // Look for struct indices...
      const StructType *STy = cast<StructType>(SrcTy);
      unsigned StructIdx = cast<ConstantUInt>(I->get())->getValue();
      for (unsigned i = 0; i != StructIdx; ++i)
        Index += countPointerFields(STy->getContainedType(i));

      // Advance SrcTy to be the new element type...
      SrcTy = STy->getContainedType(StructIdx);
    } else {
      // Otherwise, stepping into array or initial pointer, just increment type
      SrcTy = cast<SequentialType>(SrcTy)->getElementType();
    }
  
  return PointerVal(InP.Node, Index);
}

static PointerValSet &getField(const PointerVal &DestPtr) {
  assert(DestPtr.Node != 0);
  return DestPtr.Node->getLink(DestPtr.Index);
}


// Reprocessing a GEP instruction is the result of the pointer operand
// changing.  This means that the set of possible values for the GEP
// needs to be expanded.
//
void FunctionRepBuilder::visitGetElementPtrInst(GetElementPtrInst &GEP) {
  PointerValSet &GEPPVS = ValueMap[&GEP];   // PointerValSet to expand
      
  // Get the input pointer val set...
  const PointerValSet &SrcPVS = ValueMap[GEP.getOperand(0)];
      
  bool Changed = false;  // Process each input value... propogating it.
  for (unsigned i = 0, e = SrcPVS.size(); i != e; ++i) {
    // Calculate where the resulting pointer would point based on an
    // input of 'Val' as the pointer type... and add it to our outgoing
    // value set.  Keep track of whether or not we actually changed
    // anything.
    //
    Changed |= GEPPVS.add(getIndexedPointerDest(SrcPVS[i], GEP));
  }

  // If our current value set changed, notify all of the users of our
  // value.
  //
  if (Changed) addAllUsesToWorkList(&GEP);        
}

void FunctionRepBuilder::visitReturnInst(ReturnInst &RI) {
  RetNode.add(ValueMap[RI.getOperand(0)]);
}

void FunctionRepBuilder::visitLoadInst(LoadInst &LI) {
  // Only loads that return pointers are interesting...
  const PointerType *DestTy = dyn_cast<PointerType>(LI.getType());
  if (DestTy == 0) return;

  const PointerValSet &SrcPVS = ValueMap[LI.getOperand(0)];        
  PointerValSet &LIPVS = ValueMap[&LI];

  bool Changed = false;
  for (unsigned si = 0, se = SrcPVS.size(); si != se; ++si) {
    PointerVal Ptr = getIndexedPointerDest(SrcPVS[si], LI);
    PointerValSet &Field = getField(Ptr);

    if (Field.size()) {             // Field loaded wasn't null?
      Changed |= LIPVS.add(Field);
    } else {
      // If we are loading a null field out of a shadow node, we need to
      // synthesize a new shadow node and link it in...
      //
      ShadowDSNode *SynthNode =
        Ptr.Node->synthesizeNode(DestTy->getElementType(), this);
      Field.add(SynthNode);

      Changed |= LIPVS.add(Field);
    }
  }

  if (Changed) addAllUsesToWorkList(&LI);
}

void FunctionRepBuilder::visitStoreInst(StoreInst &SI) {
  // The only stores that are interesting are stores the store pointers
  // into data structures...
  //
  if (!isa<PointerType>(SI.getOperand(0)->getType())) return;
  if (!ValueMap.count(SI.getOperand(0))) return;  // Src scalar has no values!
        
  const PointerValSet &SrcPVS = ValueMap[SI.getOperand(0)];
  const PointerValSet &PtrPVS = ValueMap[SI.getOperand(1)];

  for (unsigned si = 0, se = SrcPVS.size(); si != se; ++si) {
    const PointerVal &SrcPtr = SrcPVS[si];
    for (unsigned pi = 0, pe = PtrPVS.size(); pi != pe; ++pi) {
      PointerVal Dest = getIndexedPointerDest(PtrPVS[pi], SI);

#if 0
      cerr << "Setting Dest:\n";
      Dest.print(cerr);
      cerr << "to point to Src:\n";
      SrcPtr.print(cerr);
#endif

      // Add SrcPtr into the Dest field...
      if (getField(Dest).add(SrcPtr)) {
        // If we modified the dest field, then invalidate everyone that points
        // to Dest.
        const std::vector<Value*> &Ptrs = Dest.Node->getPointers();
        for (unsigned i = 0, e = Ptrs.size(); i != e; ++i)
          addAllUsesToWorkList(Ptrs[i]);
      }
    }
  }
}

void FunctionRepBuilder::visitCallInst(CallInst &CI) {
  CallDSNode *DSN = CallMap[&CI];
  unsigned PtrNum = 0;
  for (unsigned i = 0, e = CI.getNumOperands(); i != e; ++i)
    if (isa<PointerType>(CI.getOperand(i)->getType()))
      DSN->addArgValue(PtrNum++, ValueMap[CI.getOperand(i)]);
}

void FunctionRepBuilder::visitPHINode(PHINode &PN) {
  assert(isa<PointerType>(PN.getType()) && "Should only update ptr phis");

  PointerValSet &PN_PVS = ValueMap[&PN];
  bool Changed = false;
  for (unsigned i = 0, e = PN.getNumIncomingValues(); i != e; ++i)
    Changed |= PN_PVS.add(ValueMap[PN.getIncomingValue(i)],
                          PN.getIncomingValue(i));

  if (Changed) addAllUsesToWorkList(&PN);
}




// FunctionDSGraph constructor - Perform the global analysis to determine
// what the data structure usage behavior or a method looks like.
//
FunctionDSGraph::FunctionDSGraph(Function *F) : Func(F) {
  FunctionRepBuilder Builder(this);
  AllocNodes  = Builder.getAllocNodes();
  ShadowNodes = Builder.getShadowNodes();
  GlobalNodes = Builder.getGlobalNodes();
  CallNodes   = Builder.getCallNodes();
  RetNode     = Builder.getRetNode();
  ValueMap    = Builder.getValueMap();

  // Remove all entries in the value map that consist of global values pointing
  // at things.  They can only point to their node, so there is no use keeping
  // them.
  //
  for (map<Value*, PointerValSet>::iterator I = ValueMap.begin(),
         E = ValueMap.end(); I != E;)
    if (isa<GlobalValue>(I->first)) {
#if MAP_DOESNT_HAVE_BROKEN_ERASE_MEMBER
      I = ValueMap.erase(I);
#else
      ValueMap.erase(I);            // This is really lame.
      I = ValueMap.begin();         // GCC's stdc++ lib doesn't return an it!
#endif
    } else
      ++I;

  bool Changed = true;
  while (Changed) {
    // Eliminate shadow nodes that are not distinguishable from some other
    // node in the graph...
    //
    Changed = UnlinkUndistinguishableNodes();

    // Eliminate shadow nodes that are now extraneous due to linking...
    Changed |= RemoveUnreachableNodes();
  }
}
