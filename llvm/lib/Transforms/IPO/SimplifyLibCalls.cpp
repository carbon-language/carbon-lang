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

#define DEBUG_TYPE "simplify-libcalls"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/ADT/hash_map"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Transforms/IPO.h"
#include <iostream>
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

    /// We need some target data for accurate signature details that are
    /// target dependent. So we require target data in our AnalysisUsage.
    virtual void getAnalysisUsage(AnalysisUsage& Info) const;

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

    /// The implementation of this function in subclasses should determine if
    /// \p F is suitable for the optimization. This method is called by 
    /// runOnModule to short circuit visiting all the call sites of such a
    /// function if that function is not suitable in the first place.
    /// If the called function is suitabe, this method should return true;
    /// false, otherwise. This function should also perform any lazy 
    /// initialization that the CallOptimizer needs to do, if its to return 
    /// true. This avoids doing initialization until the optimizer is actually
    /// going to be called upon to do some optimization.
    virtual bool ValidateCalledFunction(
      const Function* F,   ///< The function that is the target of call sites
      const TargetData& TD ///< Information about the target
    ) = 0;

    /// The implementations of this function in subclasses is the heart of the 
    /// SimplifyLibCalls algorithm. Sublcasses of this class implement 
    /// OptimizeCall to determine if (a) the conditions are right for optimizing
    /// the call and (b) to perform the optimization. If an action is taken 
    /// against ci, the subclass is responsible for returning true and ensuring
    /// that ci is erased from its parent.
    /// @param ci the call instruction under consideration
    /// @param f the function that ci calls.
    /// @brief Optimize a call, if possible.
    virtual bool OptimizeCall(
      CallInst* ci,         ///< The call instruction that should be optimized.
      const TargetData& TD  ///< Information about the target
    ) = 0;

    const char * getFunctionName() const { return func_name; }
  private:
    const char* func_name;
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
  CallOptimizer::~CallOptimizer() { }

}

ModulePass *llvm::createSimplifyLibCallsPass() 
{ 
  return new SimplifyLibCalls(); 
}

void SimplifyLibCalls::getAnalysisUsage(AnalysisUsage& Info) const
{
  // Ask that the TargetData analysis be performed before us so we can use
  // the target data.
  Info.addRequired<TargetData>();
}

bool SimplifyLibCalls::runOnModule(Module &M) 
{
  TargetData& TD = getAnalysis<TargetData>();

  bool result = false;

  // The call optimizations can be recursive. That is, the optimization might
  // generate a call to another function which can also be optimized. This way
  // we make the CallOptimizer instances very specific to the case they handle.
  // It also means we need to keep running over the function calls in the module
  // until we don't get any more optimizations possible.
  bool found_optimization = false;
  do
  {
    found_optimization = false;
    for (Module::iterator FI = M.begin(), FE = M.end(); FI != FE; ++FI)
    {
      // All the "well-known" functions are external and have external linkage
      // because they live in a runtime library somewhere and were (probably) 
      // not compiled by LLVM.  So, we only act on external functions that have 
      // external linkage and non-empty uses.
      if (FI->isExternal() && FI->hasExternalLinkage() && !FI->use_empty())
      {
        // Get the optimization class that pertains to this function
        if (CallOptimizer* CO = optlist[FI->getName().c_str()] )
        {
          // Make sure the called function is suitable for the optimization
          if (CO->ValidateCalledFunction(FI,TD))
          {
            // Loop over each of the uses of the function
            for (Value::use_iterator UI = FI->use_begin(), UE = FI->use_end(); 
                 UI != UE ; )
            {
              // If the use of the function is a call instruction
              if (CallInst* CI = dyn_cast<CallInst>(*UI++))
              {
                // Do the optimization on the CallOptimizer.
                if (CO->OptimizeCall(CI,TD))
                {
                  ++SimplifiedLibCalls;
                  found_optimization = result = true;
              DEBUG(std::cerr << "simplify-libcall: " << CO->getFunctionName());
                }
              }
            }
          }
        }
      }
    }
  } while (found_optimization);
  return result;
}

namespace {

  /// Provide some functions for accessing standard library prototypes and
  /// caching them so we don't have to keep recomputing them
  FunctionType* get_strlen(const Type* IntPtrTy)
  {
    static FunctionType* strlen_type = 0;
    if (!strlen_type)
    {
      std::vector<const Type*> args;
      args.push_back(PointerType::get(Type::SByteTy));
      strlen_type = FunctionType::get(IntPtrTy, args, false);
    }
    return strlen_type;
  }

  FunctionType* get_memcpy()
  {
    static FunctionType* memcpy_type = 0;
    if (!memcpy_type)
    {
      // Note: this is for llvm.memcpy intrinsic
      std::vector<const Type*> args;
      args.push_back(PointerType::get(Type::SByteTy));
      args.push_back(PointerType::get(Type::SByteTy));
      args.push_back(Type::IntTy);
      args.push_back(Type::IntTy);
      memcpy_type = FunctionType::get(Type::VoidTy, args, false);
    }
    return memcpy_type;
  }

  /// A function to compute the length of a null-terminated string of integers.
  /// This function can't rely on the size of the constant array because there 
  /// could be a null terminator in the middle of the array. We also have to 
  /// bail out if we find a non-integer constant initializer of one of the 
  /// elements or if there is no null-terminator. The logic below checks
  bool getConstantStringLength(Value* V, uint64_t& len )
  {
    assert(V != 0 && "Invalid args to getConstantStringLength");
    len = 0; // make sure we initialize this 
    User* GEP = 0;
    // If the value is not a GEP instruction nor a constant expression with a 
    // GEP instruction, then return false because ConstantArray can't occur 
    // any other way
    if (GetElementPtrInst* GEPI = dyn_cast<GetElementPtrInst>(V))
      GEP = GEPI;
    else if (ConstantExpr* CE = dyn_cast<ConstantExpr>(V))
      if (CE->getOpcode() == Instruction::GetElementPtr)
        GEP = CE;
      else
        return false;
    else
      return false;

    // Make sure the GEP has exactly three arguments.
    if (GEP->getNumOperands() != 3)
      return false;

    // Check to make sure that the first operand of the GEP is an integer and
    // has value 0 so that we are sure we're indexing into the initializer. 
    if (ConstantInt* op1 = dyn_cast<ConstantInt>(GEP->getOperand(1)))
    {
      if (!op1->isNullValue())
        return false;
    }
    else
      return false;

    // Ensure that the second operand is a ConstantInt. If it isn't then this
    // GEP is wonky and we're not really sure what were referencing into and 
    // better of not optimizing it. While we're at it, get the second index
    // value. We'll need this later for indexing the ConstantArray.
    uint64_t start_idx = 0;
    if (ConstantInt* CI = dyn_cast<ConstantInt>(GEP->getOperand(2)))
      start_idx = CI->getRawValue();
    else
      return false;

    // The GEP instruction, constant or instruction, must reference a global
    // variable that is a constant and is initialized. The referenced constant
    // initializer is the array that we'll use for optimization.
    GlobalVariable* GV = dyn_cast<GlobalVariable>(GEP->getOperand(0));
    if (!GV || !GV->isConstant() || !GV->hasInitializer())
      return false;

    // Get the initializer.
    Constant* INTLZR = GV->getInitializer();

    // Handle the ConstantAggregateZero case
    if (ConstantAggregateZero* CAZ = dyn_cast<ConstantAggregateZero>(INTLZR))
    {
      // This is a degenerate case. The initializer is constant zero so the
      // length of the string must be zero.
      len = 0;
      return true;
    }

    // Must be a Constant Array
    ConstantArray* A = dyn_cast<ConstantArray>(INTLZR);
    if (!A)
      return false;

    // Get the number of elements in the array
    uint64_t max_elems = A->getType()->getNumElements();

    // Traverse the constant array from start_idx (derived above) which is
    // the place the GEP refers to in the array. 
    for ( len = start_idx; len < max_elems; len++)
    {
      if (ConstantInt* CI = dyn_cast<ConstantInt>(A->getOperand(len)))
      {
        // Check for the null terminator
        if (CI->isNullValue())
          break; // we found end of string
      }
      else
        return false; // This array isn't suitable, non-int initializer
    }
    if (len >= max_elems)
      return false; // This array isn't null terminated

    // Subtract out the initial value from the length
    len -= start_idx;
    return true; // success!
  }

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

  // Make sure the called function looks like exit (int argument, int return
  // type, external linkage, not varargs). 
  virtual bool ValidateCalledFunction(const Function* f, const TargetData& TD)
  {
    if (f->arg_size() >= 1)
      if (f->arg_begin()->getType()->isInteger())
        return true;
    return false;
  }

  virtual bool OptimizeCall(CallInst* ci, const TargetData& TD)
  {
    // To be careful, we check that the call to exit is coming from "main", that
    // main has external linkage, and the return type of main and the argument
    // to exit have the same type. 
    Function *from = ci->getParent()->getParent();
    if (from->hasExternalLinkage())
      if (from->getReturnType() == ci->getOperand(1)->getType())
        if (from->getName() == "main")
        {
          // Okay, time to actually do the optimization. First, get the basic 
          // block of the call instruction
          BasicBlock* bb = ci->getParent();

          // Create a return instruction that we'll replace the call with. 
          // Note that the argument of the return is the argument of the call 
          // instruction.
          ReturnInst* ri = new ReturnInst(ci->getOperand(1), ci);

          // Split the block at the call instruction which places it in a new
          // basic block.
          bb->splitBasicBlock(ci);

          // The block split caused a branch instruction to be inserted into
          // the end of the original block, right after the return instruction
          // that we put there. That's not a valid block, so delete the branch
          // instruction.
          bb->getInstList().pop_back();

          // Now we can finally get rid of the call instruction which now lives
          // in the new basic block.
          ci->eraseFromParent();

          // Optimization succeeded, return true.
          return true;
        }
    // We didn't pass the criteria for this optimization so return false
    return false;
  }
} ExitInMainOptimizer;

/// This CallOptimizer will simplify a call to the strcat library function. The
/// simplification is possible only if the string being concatenated is a 
/// constant array or a constant expression that results in a constant array. In
/// this case, if the array is small, we can generate a series of inline store
/// instructions to effect the concatenation without calling strcat.
/// @brief Simplify the strcat library function.
struct StrCatOptimization : public CallOptimizer
{
private:
  Function* strlen_func;
  Function* memcpy_func;
public:
  StrCatOptimization() 
    : CallOptimizer("strcat") 
    , strlen_func(0)
    , memcpy_func(0)
    {}
  virtual ~StrCatOptimization() {}

  inline Function* get_strlen_func(Module*M,const Type* IntPtrTy)
  {
    if (strlen_func)
      return strlen_func;
    return strlen_func = M->getOrInsertFunction("strlen",get_strlen(IntPtrTy));
  }

  inline Function* get_memcpy_func(Module* M) 
  {
    if (memcpy_func)
      return memcpy_func;
    return memcpy_func = M->getOrInsertFunction("llvm.memcpy",get_memcpy());
  }

  /// @brief Make sure that the "strcat" function has the right prototype
  virtual bool ValidateCalledFunction(const Function* f, const TargetData& TD) 
  {
    if (f->getReturnType() == PointerType::get(Type::SByteTy))
      if (f->arg_size() == 2) 
      {
        Function::const_arg_iterator AI = f->arg_begin();
        if (AI++->getType() == PointerType::get(Type::SByteTy))
          if (AI->getType() == PointerType::get(Type::SByteTy))
          {
            // Invalidate the pre-computed strlen_func and memcpy_func Functions
            // because, by definition, this method is only called when a new
            // Module is being traversed. Invalidation causes re-computation for
            // the new Module (if necessary).
            strlen_func = 0;
            memcpy_func = 0;

            // Indicate this is a suitable call type.
            return true;
          }
      }
    return false;
  }

  /// Perform the optimization if the length of the string concatenated
  /// is reasonably short and it is a constant array.
  virtual bool OptimizeCall(CallInst* ci, const TargetData& TD)
  {
    // Extract the initializer (while making numerous checks) from the 
    // source operand of the call to strcat. If we get null back, one of
    // a variety of checks in get_GVInitializer failed
    uint64_t len = 0;
    if (!getConstantStringLength(ci->getOperand(2),len))
      return false;

    // Handle the simple, do-nothing case
    if (len == 0)
    {
      ci->replaceAllUsesWith(ci->getOperand(1));
      ci->eraseFromParent();
      return true;
    }

    // Increment the length because we actually want to memcpy the null
    // terminator as well.
    len++;

    // Extract some information from the instruction
    Module* M = ci->getParent()->getParent()->getParent();

    // We need to find the end of the destination string.  That's where the 
    // memory is to be moved to. We just generate a call to strlen (further 
    // optimized in another pass). Note that the get_strlen_func() call 
    // caches the Function* for us.
    CallInst* strlen_inst = 
      new CallInst(get_strlen_func(M,TD.getIntPtrType()),
                   ci->getOperand(1),"",ci);

    // Now that we have the destination's length, we must index into the 
    // destination's pointer to get the actual memcpy destination (end of
    // the string .. we're concatenating).
    std::vector<Value*> idx;
    idx.push_back(strlen_inst);
    GetElementPtrInst* gep = 
      new GetElementPtrInst(ci->getOperand(1),idx,"",ci);

    // We have enough information to now generate the memcpy call to
    // do the concatenation for us.
    std::vector<Value*> vals;
    vals.push_back(gep); // destination
    vals.push_back(ci->getOperand(2)); // source
    vals.push_back(ConstantSInt::get(Type::IntTy,len)); // length
    vals.push_back(ConstantSInt::get(Type::IntTy,1)); // alignment
    CallInst* memcpy_inst = new CallInst(get_memcpy_func(M), vals, "", ci);

    // Finally, substitute the first operand of the strcat call for the 
    // strcat call itself since strcat returns its first operand; and, 
    // kill the strcat CallInst.
    ci->replaceAllUsesWith(ci->getOperand(1));
    ci->eraseFromParent();
    return true;
  }
} StrCatOptimizer;

/// This CallOptimizer will simplify a call to the strlen library function by
/// replacing it with a constant value if the string provided to it is a 
/// constant array.
/// @brief Simplify the strlen library function.
struct StrLenOptimization : public CallOptimizer
{
  StrLenOptimization() : CallOptimizer("strlen") {}
  virtual ~StrLenOptimization() {}

  /// @brief Make sure that the "strlen" function has the right prototype
  virtual bool ValidateCalledFunction(const Function* f, const TargetData& TD)
  {
    if (f->getReturnType() == TD.getIntPtrType())
      if (f->arg_size() == 1) 
        if (Function::const_arg_iterator AI = f->arg_begin())
          if (AI->getType() == PointerType::get(Type::SByteTy))
            return true;
    return false;
  }

  /// @brief Perform the strlen optimization
  virtual bool OptimizeCall(CallInst* ci, const TargetData& TD)
  {
    // Get the length of the string
    uint64_t len = 0;
    if (!getConstantStringLength(ci->getOperand(1),len))
      return false;

    ci->replaceAllUsesWith(ConstantInt::get(TD.getIntPtrType(),len));
    ci->eraseFromParent();
    return true;
  }
} StrLenOptimizer;

/// This CallOptimizer will simplify a call to the memcpy library function by
/// expanding it out to a small set of stores if the copy source is a constant
/// array. 
/// @brief Simplify the memcpy library function.
struct MemCpyOptimization : public CallOptimizer
{
  MemCpyOptimization() : CallOptimizer("llvm.memcpy") {}
protected:
  MemCpyOptimization(const char* fname) : CallOptimizer(fname) {}
public:
  virtual ~MemCpyOptimization() {}

  /// @brief Make sure that the "memcpy" function has the right prototype
  virtual bool ValidateCalledFunction(const Function* f, const TargetData& TD)
  {
    // Just make sure this has 4 arguments per LLVM spec.
    return (f->arg_size() == 4);
  }

  /// Because of alignment and instruction information that we don't have, we
  /// leave the bulk of this to the code generators. The optimization here just
  /// deals with a few degenerate cases where the length of the string and the
  /// alignment match the sizes of our intrinsic types so we can do a load and
  /// store instead of the memcpy call.
  /// @brief Perform the memcpy optimization.
  virtual bool OptimizeCall(CallInst* ci, const TargetData& TD)
  {
    // Make sure we have constant int values to work with
    ConstantInt* LEN = dyn_cast<ConstantInt>(ci->getOperand(3));
    if (!LEN)
      return false;
    ConstantInt* ALIGN = dyn_cast<ConstantInt>(ci->getOperand(4));
    if (!ALIGN)
      return false;

    // If the length is larger than the alignment, we can't optimize
    uint64_t len = LEN->getRawValue();
    uint64_t alignment = ALIGN->getRawValue();
    if (len > alignment)
      return false;

    Value* dest = ci->getOperand(1);
    Value* src = ci->getOperand(2);
    CastInst* SrcCast = 0;
    CastInst* DestCast = 0;
    switch (len)
    {
      case 0:
        // The memcpy is a no-op so just dump its call.
        ci->eraseFromParent();
        return true;
      case 1:
        SrcCast = new CastInst(src,PointerType::get(Type::SByteTy),"",ci);
        DestCast = new CastInst(dest,PointerType::get(Type::SByteTy),"",ci);
        break;
      case 2:
        SrcCast = new CastInst(src,PointerType::get(Type::ShortTy),"",ci);
        DestCast = new CastInst(dest,PointerType::get(Type::ShortTy),"",ci);
        break;
      case 4:
        SrcCast = new CastInst(src,PointerType::get(Type::IntTy),"",ci);
        DestCast = new CastInst(dest,PointerType::get(Type::IntTy),"",ci);
        break;
      case 8:
        SrcCast = new CastInst(src,PointerType::get(Type::LongTy),"",ci);
        DestCast = new CastInst(dest,PointerType::get(Type::LongTy),"",ci);
        break;
      default:
        return false;
    }
    LoadInst* LI = new LoadInst(SrcCast,"",ci);
    StoreInst* SI = new StoreInst(LI, DestCast, ci);
    ci->eraseFromParent();
    return true;
  }
} MemCpyOptimizer;

/// This CallOptimizer will simplify a call to the memmove library function. It
/// is identical to MemCopyOptimization except for the name of the intrinsic.
/// @brief Simplify the memmove library function.
struct MemMoveOptimization : public MemCpyOptimization
{
  MemMoveOptimization() : MemCpyOptimization("llvm.memmove") {}

} MemMoveOptimizer;

}
