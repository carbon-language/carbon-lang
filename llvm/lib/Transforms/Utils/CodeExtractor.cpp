//===- CodeExtractor.cpp - Pull code region into a new function -----------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements the interface to tear out a code region, such as an
// individual loop or a parallel section, into a new function, replacing it with
// a call to the new function.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/FunctionUtils.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "Support/CommandLine.h"
#include "Support/Debug.h"
#include "Support/StringExtras.h"
#include <algorithm>
#include <set>
using namespace llvm;

// Provide a command-line option to aggregate function arguments into a struct
// for functions produced by the code extrator. This is useful when converting
// extracted functions to pthread-based code, as only one argument (void*) can
// be passed in to pthread_create().
static cl::opt<bool>
AggregateArgsOpt("aggregate-extracted-args", cl::Hidden,
                 cl::desc("Aggregate arguments to code-extracted functions"));

namespace {
  class CodeExtractor {
    typedef std::vector<Value*> Values;
    std::set<BasicBlock*> BlocksToExtract;
    DominatorSet *DS;
    bool AggregateArgs;
  public:
    CodeExtractor(DominatorSet *ds = 0, bool AggArgs = false)
      : DS(ds), AggregateArgs(AggregateArgsOpt) {}

    Function *ExtractCodeRegion(const std::vector<BasicBlock*> &code);

    bool isEligible(const std::vector<BasicBlock*> &code);

  private:
    void findInputsOutputs(Values &inputs, Values &outputs,
                           BasicBlock *newHeader,
                           BasicBlock *newRootNode);

    Function *constructFunction(const Values &inputs,
                                const Values &outputs,
                                BasicBlock *header,
                                BasicBlock *newRootNode, BasicBlock *newHeader,
                                Function *oldFunction, Module *M);

    void moveCodeToFunction(Function *newFunction);

    void emitCallAndSwitchStatement(Function *newFunction,
                                    BasicBlock *newHeader,
                                    Values &inputs,
                                    Values &outputs);

  };
}

void CodeExtractor::findInputsOutputs(Values &inputs, Values &outputs,
                                      BasicBlock *newHeader,
                                      BasicBlock *newRootNode) {
  for (std::set<BasicBlock*>::const_iterator ci = BlocksToExtract.begin(), 
       ce = BlocksToExtract.end(); ci != ce; ++ci) {
    BasicBlock *BB = *ci;
    for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I) {
      // If a used value is defined outside the region, it's an input.  If an
      // instruction is used outside the region, it's an output.
      if (PHINode *PN = dyn_cast<PHINode>(I)) {
        for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
          Value *V = PN->getIncomingValue(i);
          if (!BlocksToExtract.count(PN->getIncomingBlock(i)) &&
              (isa<Instruction>(V) || isa<Argument>(V)))
            inputs.push_back(V);
          else if (Instruction *opI = dyn_cast<Instruction>(V)) {
            if (!BlocksToExtract.count(opI->getParent()))
              inputs.push_back(opI);
          } else if (isa<Argument>(V))
            inputs.push_back(V);
        }
      } else {
        // All other instructions go through the generic input finder
        // Loop over the operands of each instruction (inputs)
        for (User::op_iterator op = I->op_begin(), opE = I->op_end();
             op != opE; ++op)
          if (Instruction *opI = dyn_cast<Instruction>(*op)) {
            // Check if definition of this operand is within the loop
            if (!BlocksToExtract.count(opI->getParent()))
              inputs.push_back(opI);
          } else if (isa<Argument>(*op)) {
            inputs.push_back(*op);
          }
      }
      
      // Consider uses of this instruction (outputs)
      for (Value::use_iterator UI = I->use_begin(), E = I->use_end();
           UI != E; ++UI)
        if (!BlocksToExtract.count(cast<Instruction>(*UI)->getParent())) {
          outputs.push_back(I);
          break;
        }
    } // for: insts
  } // for: basic blocks
}

/// constructFunction - make a function based on inputs and outputs, as follows:
/// f(in0, ..., inN, out0, ..., outN)
///
Function *CodeExtractor::constructFunction(const Values &inputs,
                                           const Values &outputs,
                                           BasicBlock *header,
                                           BasicBlock *newRootNode,
                                           BasicBlock *newHeader,
                                           Function *oldFunction,
                                           Module *M) {
  DEBUG(std::cerr << "inputs: " << inputs.size() << "\n");
  DEBUG(std::cerr << "outputs: " << outputs.size() << "\n");

  // This function returns unsigned, outputs will go back by reference.
  Type *retTy = Type::UShortTy;
  std::vector<const Type*> paramTy;

  // Add the types of the input values to the function's argument list
  for (Values::const_iterator i = inputs.begin(),
         e = inputs.end(); i != e; ++i) {
    const Value *value = *i;
    DEBUG(std::cerr << "value used in func: " << value << "\n");
    paramTy.push_back(value->getType());
  }

  // Add the types of the output values to the function's argument list.
  for (Values::const_iterator I = outputs.begin(), E = outputs.end();
       I != E; ++I) {
    DEBUG(std::cerr << "instr used in func: " << *I << "\n");
    if (AggregateArgs)
      paramTy.push_back((*I)->getType());
    else
      paramTy.push_back(PointerType::get((*I)->getType()));
  }

  DEBUG(std::cerr << "Function type: " << retTy << " f(");
  DEBUG(for (std::vector<const Type*>::iterator i = paramTy.begin(),
               e = paramTy.end(); i != e; ++i) std::cerr << *i << ", ");
  DEBUG(std::cerr << ")\n");

  if (AggregateArgs && (inputs.size() + outputs.size() > 0)) {
    PointerType *StructPtr = PointerType::get(StructType::get(paramTy));
    paramTy.clear();
    paramTy.push_back(StructPtr);
  }
  const FunctionType *funcType = FunctionType::get(retTy, paramTy, false);

  // Create the new function
  Function *newFunction = new Function(funcType,
                                       GlobalValue::InternalLinkage,
                                       oldFunction->getName() + "_code", M);
  newFunction->getBasicBlockList().push_back(newRootNode);

  // Create an iterator to name all of the arguments we inserted.
  Function::aiterator AI = newFunction->abegin();

  // Rewrite all users of the inputs in the extracted region to use the
  // arguments (or appropriate addressing into struct) instead.
  for (unsigned i = 0, e = inputs.size(); i != e; ++i) {
    Value *RewriteVal;
    if (AggregateArgs) {
      std::vector<Value*> Indices;
      Indices.push_back(Constant::getNullValue(Type::UIntTy));
      Indices.push_back(ConstantUInt::get(Type::UIntTy, i));
      std::string GEPname = "gep_" + inputs[i]->getName();
      TerminatorInst *TI = newFunction->begin()->getTerminator();
      GetElementPtrInst *GEP = new GetElementPtrInst(AI, Indices, GEPname, TI);
      RewriteVal = new LoadInst(GEP, "load" + GEPname, TI);
    } else
      RewriteVal = AI++;

    std::vector<User*> Users(inputs[i]->use_begin(), inputs[i]->use_end());
    for (std::vector<User*>::iterator use = Users.begin(), useE = Users.end();
         use != useE; ++use)
      if (Instruction* inst = dyn_cast<Instruction>(*use))
        if (BlocksToExtract.count(inst->getParent()))
          inst->replaceUsesOfWith(inputs[i], RewriteVal);
  }

  // Set names for input and output arguments.
  if (!AggregateArgs) {
    AI = newFunction->abegin();
    for (unsigned i = 0, e = inputs.size(); i != e; ++i, ++AI)
      AI->setName(inputs[i]->getName());
    for (unsigned i = 0, e = outputs.size(); i != e; ++i, ++AI)
      AI->setName(outputs[i]->getName()+".out");  
  }

  // Rewrite branches to basic blocks outside of the loop to new dummy blocks
  // within the new function. This must be done before we lose track of which
  // blocks were originally in the code region.
  std::vector<User*> Users(header->use_begin(), header->use_end());
  for (unsigned i = 0, e = Users.size(); i != e; ++i)
    // The BasicBlock which contains the branch is not in the region
    // modify the branch target to a new block
    if (TerminatorInst *TI = dyn_cast<TerminatorInst>(Users[i]))
      if (!BlocksToExtract.count(TI->getParent()) &&
          TI->getParent()->getParent() == oldFunction)
        TI->replaceUsesOfWith(header, newHeader);

  return newFunction;
}

void CodeExtractor::moveCodeToFunction(Function *newFunction) {
  Function *oldFunc = (*BlocksToExtract.begin())->getParent();
  Function::BasicBlockListType &oldBlocks = oldFunc->getBasicBlockList();
  Function::BasicBlockListType &newBlocks = newFunction->getBasicBlockList();

  for (std::set<BasicBlock*>::const_iterator i = BlocksToExtract.begin(),
         e = BlocksToExtract.end(); i != e; ++i) {
    // Delete the basic block from the old function, and the list of blocks
    oldBlocks.remove(*i);

    // Insert this basic block into the new function
    newBlocks.push_back(*i);
  }
}

void
CodeExtractor::emitCallAndSwitchStatement(Function *newFunction,
                                          BasicBlock *codeReplacer,
                                          Values &inputs,
                                          Values &outputs) {

  // Emit a call to the new function, passing in:
  // *pointer to struct (if aggregating parameters), or 
  // plan inputs and allocated memory for outputs 
  std::vector<Value*> params, StructValues, ReloadOutputs;

  // Add inputs as params, or to be filled into the struct
  for (Values::iterator i = inputs.begin(), e = inputs.end(); i != e; ++i)
    if (AggregateArgs)
      StructValues.push_back(*i);
    else
      params.push_back(*i);

  // Create allocas for the outputs
  for (Values::iterator i = outputs.begin(), e = outputs.end(); i != e; ++i) {
    if (AggregateArgs) {
      StructValues.push_back(*i);
    } else {
      AllocaInst *alloca =
        new AllocaInst((*i)->getType(), 0, (*i)->getName()+".loc",
                       codeReplacer->getParent()->begin()->begin());
      ReloadOutputs.push_back(alloca);
      params.push_back(alloca);
    }
  }

  AllocaInst *Struct = 0;
  if (AggregateArgs && (inputs.size() + outputs.size() > 0)) {
    std::vector<const Type*> ArgTypes;
    for (Values::iterator v = StructValues.begin(),
           ve = StructValues.end(); v != ve; ++v)
      ArgTypes.push_back((*v)->getType());

    // Allocate a struct at the beginning of this function
    Type *StructArgTy = StructType::get(ArgTypes);
    Struct = 
      new AllocaInst(StructArgTy, 0, "structArg", 
                     codeReplacer->getParent()->begin()->begin());
    params.push_back(Struct);

    for (unsigned i = 0, e = inputs.size(); i != e; ++i) {
      std::vector<Value*> Indices;
      Indices.push_back(Constant::getNullValue(Type::UIntTy));
      Indices.push_back(ConstantUInt::get(Type::UIntTy, i));
      GetElementPtrInst *GEP =
        new GetElementPtrInst(Struct, Indices,
                              "gep_" + StructValues[i]->getName(), 0);
      codeReplacer->getInstList().push_back(GEP);
      StoreInst *SI = new StoreInst(StructValues[i], GEP);
      codeReplacer->getInstList().push_back(SI);
    }
  } 

  // Emit the call to the function
  CallInst *call = new CallInst(newFunction, params, "targetBlock");
  codeReplacer->getInstList().push_back(call);

  Function::aiterator OutputArgBegin = newFunction->abegin();
  unsigned FirstOut = inputs.size();
  if (!AggregateArgs)
    std::advance(OutputArgBegin, inputs.size());

  // Reload the outputs passed in by reference
  for (unsigned i = 0, e = outputs.size(); i != e; ++i) {
    Value *Output = 0;
    if (AggregateArgs) {
      std::vector<Value*> Indices;
      Indices.push_back(Constant::getNullValue(Type::UIntTy));
      Indices.push_back(ConstantUInt::get(Type::UIntTy, FirstOut + i));
      GetElementPtrInst *GEP 
        = new GetElementPtrInst(Struct, Indices,
                                "gep_reload_" + outputs[i]->getName(), 0);
      codeReplacer->getInstList().push_back(GEP);
      Output = GEP;
    } else {
      Output = ReloadOutputs[i];
    }
    LoadInst *load = new LoadInst(Output, outputs[i]->getName()+".reload");
    codeReplacer->getInstList().push_back(load);
    std::vector<User*> Users(outputs[i]->use_begin(), outputs[i]->use_end());
    for (unsigned u = 0, e = Users.size(); u != e; ++u) {
      Instruction *inst = cast<Instruction>(Users[u]);
      if (!BlocksToExtract.count(inst->getParent()))
        inst->replaceUsesOfWith(outputs[i], load);
    }
  }

  // Now we can emit a switch statement using the call as a value.
  SwitchInst *TheSwitch = new SwitchInst(call, codeReplacer, codeReplacer);

  // Since there may be multiple exits from the original region, make the new
  // function return an unsigned, switch on that number.  This loop iterates
  // over all of the blocks in the extracted region, updating any terminator
  // instructions in the to-be-extracted region that branch to blocks that are
  // not in the region to be extracted.
  std::map<BasicBlock*, BasicBlock*> ExitBlockMap;

  unsigned switchVal = 0;
  for (std::set<BasicBlock*>::const_iterator i = BlocksToExtract.begin(),
         e = BlocksToExtract.end(); i != e; ++i) {
    TerminatorInst *TI = (*i)->getTerminator();
    for (unsigned i = 0, e = TI->getNumSuccessors(); i != e; ++i)
      if (!BlocksToExtract.count(TI->getSuccessor(i))) {
        BasicBlock *OldTarget = TI->getSuccessor(i);
        // add a new basic block which returns the appropriate value
        BasicBlock *&NewTarget = ExitBlockMap[OldTarget];
        if (!NewTarget) {
          // If we don't already have an exit stub for this non-extracted
          // destination, create one now!
          NewTarget = new BasicBlock(OldTarget->getName() + ".exitStub",
                                     newFunction);

          ConstantUInt *brVal = ConstantUInt::get(Type::UShortTy, switchVal++);
          ReturnInst *NTRet = new ReturnInst(brVal, NewTarget);

          // Update the switch instruction.
          TheSwitch->addCase(brVal, OldTarget);

          // Restore values just before we exit
          Function::aiterator OAI = OutputArgBegin;
          for (unsigned out = 0, e = outputs.size(); out != e; ++out) {
            // For an invoke, the normal destination is the only one that is
            // dominated by the result of the invocation
            BasicBlock *DefBlock = cast<Instruction>(outputs[out])->getParent();
            if (InvokeInst *Invoke = dyn_cast<InvokeInst>(outputs[out]))
              DefBlock = Invoke->getNormalDest();
            if (!DS || DS->dominates(DefBlock, TI->getParent()))
              if (AggregateArgs) {
                std::vector<Value*> Indices;
                Indices.push_back(Constant::getNullValue(Type::UIntTy));
                Indices.push_back(ConstantUInt::get(Type::UIntTy,FirstOut+out));
                GetElementPtrInst *GEP =
                  new GetElementPtrInst(OAI, Indices,
                                        "gep_" + outputs[out]->getName(), 
                                        NTRet);
                new StoreInst(outputs[out], GEP, NTRet);
              } else
                new StoreInst(outputs[out], OAI, NTRet);
            // Advance output iterator even if we don't emit a store
            if (!AggregateArgs) ++OAI;
          }
        }

        // rewrite the original branch instruction with this new target
        TI->setSuccessor(i, NewTarget);
      }
  }

  // Now that we've done the deed, make the default destination of the switch
  // instruction be a block with a call to abort() -- since this path should not
  // be taken, this will abort sooner rather than later.
  if (TheSwitch->getNumSuccessors() > 1) {
    Function *container = codeReplacer->getParent();
    BasicBlock *abortBB = new BasicBlock("abortBlock", container);
    std::vector<const Type*> paramTypes;
    FunctionType *abortTy = FunctionType::get(Type::VoidTy, paramTypes, false);
    Function *abortFunc = 
      container->getParent()->getOrInsertFunction("abort", abortTy);
    abortBB->getInstList().push_back(new CallInst(abortFunc));
    Function *ParentFunc = TheSwitch->getParent()->getParent();
    if (ParentFunc->getReturnType() == Type::VoidTy)
      new ReturnInst(0, abortBB);
    else
      new ReturnInst(Constant::getNullValue(ParentFunc->getReturnType()),
                     abortBB);
    TheSwitch->setSuccessor(0, abortBB);
  } else {
    // There is only 1 successor (the block containing the switch itself), which
    // means that previously this was the last part of the function, and hence
    // this should be rewritten as a `ret'
    
    // Check if the function should return a value
    if (TheSwitch->getParent()->getParent()->getReturnType() != Type::VoidTy &&
        TheSwitch->getParent()->getParent()->getReturnType() ==
        TheSwitch->getCondition()->getType())
      // return what we have
      new ReturnInst(TheSwitch->getCondition(), TheSwitch);
    else
      // just return
      new ReturnInst(0, TheSwitch);

    TheSwitch->getParent()->getInstList().erase(TheSwitch);
  }
}


/// ExtractRegion - Removes a loop from a function, replaces it with a call to
/// new function. Returns pointer to the new function.
///
/// algorithm:
///
/// find inputs and outputs for the region
///
/// for inputs: add to function as args, map input instr* to arg# 
/// for outputs: add allocas for scalars, 
///             add to func as args, map output instr* to arg#
///
/// rewrite func to use argument #s instead of instr*
///
/// for each scalar output in the function: at every exit, store intermediate 
/// computed result back into memory.
///
Function *CodeExtractor::ExtractCodeRegion(const std::vector<BasicBlock*> &code)
{
  if (!isEligible(code))
    return 0;

  // 1) Find inputs, outputs
  // 2) Construct new function
  //  * Add allocas for defs, pass as args by reference
  //  * Pass in uses as args
  // 3) Move code region, add call instr to func
  //
  BlocksToExtract.insert(code.begin(), code.end());

  Values inputs, outputs;

  // Assumption: this is a single-entry code region, and the header is the first
  // block in the region.
  BasicBlock *header = code[0];
  for (unsigned i = 1, e = code.size(); i != e; ++i)
    for (pred_iterator PI = pred_begin(code[i]), E = pred_end(code[i]);
         PI != E; ++PI)
      assert(BlocksToExtract.count(*PI) &&
             "No blocks in this region may have entries from outside the region"
             " except for the first block!");
  
  Function *oldFunction = header->getParent();

  // This takes place of the original loop
  BasicBlock *codeReplacer = new BasicBlock("codeRepl", oldFunction);

  // The new function needs a root node because other nodes can branch to the
  // head of the loop, and the root cannot have predecessors
  BasicBlock *newFuncRoot = new BasicBlock("newFuncRoot");
  newFuncRoot->getInstList().push_back(new BranchInst(header));

  // Find inputs to, outputs from the code region
  //
  // If one of the inputs is coming from a different basic block and it's in a
  // phi node, we need to rewrite the phi node:
  //
  // * All the inputs which involve basic blocks OUTSIDE of this region go into
  //   a NEW phi node that takes care of finding which value really came in.
  //   The result of this phi is passed to the function as an argument. 
  //
  // * All the other phi values stay.
  //
  // FIXME: PHI nodes' incoming blocks aren't being rewritten to accomodate for
  // blocks moving to a new function.
  // SOLUTION: move Phi nodes out of the loop header into the codeReplacer, pass
  // the values as parameters to the function
  findInputsOutputs(inputs, outputs, codeReplacer, newFuncRoot);

  // Step 2: Construct new function based on inputs/outputs,
  // Add allocas for all defs
  Function *newFunction = constructFunction(inputs, outputs, code[0],
                                            newFuncRoot, 
                                            codeReplacer, oldFunction,
                                            oldFunction->getParent());

  emitCallAndSwitchStatement(newFunction, codeReplacer, inputs, outputs);

  moveCodeToFunction(newFunction);

  // Loop over all of the PHI nodes in the entry block (code[0]), and change any
  // references to the old incoming edge to be the new incoming edge.
  for (BasicBlock::iterator I = code[0]->begin();
       PHINode *PN = dyn_cast<PHINode>(I); ++I)
    for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i)
      if (!BlocksToExtract.count(PN->getIncomingBlock(i)))
        PN->setIncomingBlock(i, newFuncRoot);

  // Look at all successors of the codeReplacer block.  If any of these blocks
  // had PHI nodes in them, we need to update the "from" block to be the code
  // replacer, not the original block in the extracted region.
  std::vector<BasicBlock*> Succs(succ_begin(codeReplacer),
                                 succ_end(codeReplacer));
  for (unsigned i = 0, e = Succs.size(); i != e; ++i)
    for (BasicBlock::iterator I = Succs[i]->begin();
         PHINode *PN = dyn_cast<PHINode>(I); ++I)
      for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i)
        if (BlocksToExtract.count(PN->getIncomingBlock(i)))
          PN->setIncomingBlock(i, codeReplacer);
  

  DEBUG(if (verifyFunction(*newFunction)) abort());
  return newFunction;
}

bool CodeExtractor::isEligible(const std::vector<BasicBlock*> &code) {
  // Deny code region if it contains allocas
  for (std::vector<BasicBlock*>::const_iterator BB = code.begin(), e=code.end();
       BB != e; ++BB)
    for (BasicBlock::const_iterator I = (*BB)->begin(), Ie = (*BB)->end();
         I != Ie; ++I)
      if (isa<AllocaInst>(*I))
        return false;
  return true;
}


/// ExtractCodeRegion - slurp a sequence of basic blocks into a brand new
/// function
///
Function* llvm::ExtractCodeRegion(DominatorSet &DS,
                                  const std::vector<BasicBlock*> &code,
                                  bool AggregateArgs) {
  return CodeExtractor(&DS, AggregateArgs).ExtractCodeRegion(code);
}

/// ExtractBasicBlock - slurp a natural loop into a brand new function
///
Function* llvm::ExtractLoop(DominatorSet &DS, Loop *L, bool AggregateArgs) {
  return CodeExtractor(&DS, AggregateArgs).ExtractCodeRegion(L->getBlocks());
}

/// ExtractBasicBlock - slurp a basic block into a brand new function
///
Function* llvm::ExtractBasicBlock(BasicBlock *BB, bool AggregateArgs) {
  std::vector<BasicBlock*> Blocks;
  Blocks.push_back(BB);
  return CodeExtractor(0, AggregateArgs).ExtractCodeRegion(Blocks);  
}
