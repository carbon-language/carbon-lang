//===- SimplifyLibCalls.cpp - Optimize specific well-known library calls --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a variety of small optimizations for calls to specific
// well-known (e.g. runtime library) function calls. For example, a call to the
// function "exit(3)" that occurs within the main() function can be transformed
// into a simple "return 3" instruction. Any optimization that takes this form
// (replace call to library function with simpler code that provides same 
// result) belongs in this file. 
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Instructions.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/hash_map"
using namespace llvm;

namespace {
  Statistic<> SimplifiedLibCalls("simplified-lib-calls", 
      "Number of well-known library calls simplified");

  /// This class is the base class for a set of small but important 
  /// optimizations of calls to well-known functions, such as those in the c
  /// library. This class provides the basic infrastructure for handling 
  /// runOnModule. Subclasses register themselves and provide two methods:
  /// RecognizeCall and OptimizeCall. Whenever this class finds a function call,
  /// it asks the subclasses to recognize the call. If it is recognized, then
  /// the OptimizeCall method is called on that subclass instance. In this way
  /// the subclasses implement the calling conditions on which they trigger and
  /// the action to perform, making it easy to add new optimizations of this
  /// form.
  /// @brief A ModulePass for optimizing well-known function calls
  struct SimplifyLibCalls : public ModulePass {


    /// For this pass, process all of the function calls in the module, calling
    /// RecognizeCall and OptimizeCall as appropriate.
    virtual bool runOnModule(Module &M);

  };

  RegisterOpt<SimplifyLibCalls> 
    X("simplify-libcalls","Simplify well-known library calls");

  struct CallOptimizer
  {
    /// @brief Constructor that registers the optimization
    CallOptimizer(const char * fname );

    virtual ~CallOptimizer();

    /// The implementations of this function in subclasses is the heart of the 
    /// SimplifyLibCalls algorithm. Sublcasses of this class implement 
    /// OptimizeCall to determine if (a) the conditions are right for optimizing
    /// the call and (b) to perform the optimization. If an action is taken 
    /// against ci, the subclass is responsible for returning true and ensuring
    /// that ci is erased from its parent.
    /// @param ci the call instruction under consideration
    /// @param f the function that ci calls.
    /// @brief Optimize a call, if possible.
    virtual bool OptimizeCall(CallInst* ci) const = 0;

    const std::string& getFunctionName() const { return func_name; }
  private:
    std::string func_name;
  };

  /// @brief The list of optimizations deriving from CallOptimizer
  hash_map<std::string,CallOptimizer*> optlist;

  CallOptimizer::CallOptimizer(const char* fname)
    : func_name(fname)
  {
    // Register this call optimizer
    optlist[func_name] = this;
  }

  /// Make sure we get our virtual table in this file.
  CallOptimizer::~CallOptimizer() {}
}

ModulePass *llvm::createSimplifyLibCallsPass() 
{ 
  return new SimplifyLibCalls(); 
}

bool SimplifyLibCalls::runOnModule(Module &M) 
{
  for (Module::iterator FI = M.begin(), FE = M.end(); FI != FE; ++FI)
  {
    // All the "well-known" functions are external because they live in a 
    // runtime library somewhere and were (probably) not compiled by LLVM.
    // So, we only act on external functions that have non-empty uses.
    if (FI->isExternal() && !FI->use_empty())
    {
      // Get the optimization class that pertains to this function
      if (CallOptimizer* CO = optlist[FI->getName()] )
      {
        // Loop over each of the uses of the function
        for (Value::use_iterator UI = FI->use_begin(), UE = FI->use_end(); 
             UI != UE ; )
        {
          // If the use of the function is a call instruction
          if (CallInst* CI = dyn_cast<CallInst>(*UI++))
          {
            // Do the optimization on the CallOptimizer we found earlier.
            if (CO->OptimizeCall(CI))
            {
              ++SimplifiedLibCalls;
              break;
            }
          }
        }
      }
    }
  }
  return true;
}

namespace {

/// This CallOptimizer will find instances of a call to "exit" that occurs
/// within the "main" function and change it to a simple "ret" instruction with
/// the same value as passed to the exit function. It assumes that the 
/// instructions after the call to exit(3) can be deleted since they are 
/// unreachable anyway.
/// @brief Replace calls to exit in main with a simple return
struct ExitInMainOptimization : public CallOptimizer
{
  ExitInMainOptimization() : CallOptimizer("exit") {}
  virtual ~ExitInMainOptimization() {}
  virtual bool OptimizeCall(CallInst* ci) const
  {
    // If the call isn't coming from main or  main doesn't have external linkage
    // or the return type of main is not the same as the type of the exit(3)
    // argument then we don't act
    if (const Function* f = ci->getParent()->getParent())
      if (!(f->hasExternalLinkage() && 
            (f->getReturnType() == ci->getOperand(1)->getType()) &&
            (f->getName() == "main")))
        return false;

    // Okay, time to replace it. Get the basic block of the call instruction
    BasicBlock* bb = ci->getParent();

    // Create a return instruction that we'll replace the call with. Note that
    // the argument of the return is the argument of the call instruction.
    ReturnInst* ri = new ReturnInst(ci->getOperand(1), ci);

    // Erase everything from the call instruction to the end of the block. There
    // really shouldn't be anything other than the call instruction, but just in
    // case there is we delete it all because its now dead.
    bb->getInstList().erase(ci, bb->end());

    return true;
  }
} ExitInMainOptimizer;

/// This CallOptimizer will find instances of a call to "exit" that occurs
/// within the "main" function and change it to a simple "ret" instruction with
/// the same value as passed to the exit function. It assumes that the 
/// instructions after the call to exit(3) can be deleted since they are 
/// unreachable anyway.
/// @brief Replace calls to exit in main with a simple return
struct StrCatOptimization : public CallOptimizer
{
  StrCatOptimization() : CallOptimizer("strcat") {}
  virtual ~StrCatOptimization() {}
  virtual bool OptimizeCall(CallInst* ci) const
  {
    return false;
  }
} StrCatOptimizer;

}
