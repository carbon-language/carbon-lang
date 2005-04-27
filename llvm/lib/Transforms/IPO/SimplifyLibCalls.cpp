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

/// This statistic keeps track of the total number of library calls that have
/// been simplified regardless of which call it is.
Statistic<> SimplifiedLibCalls("simplify-libcalls", 
  "Number of well-known library calls simplified");

// Forward declarations
class LibCallOptimization;
class SimplifyLibCalls;

/// @brief The list of optimizations deriving from LibCallOptimization
hash_map<std::string,LibCallOptimization*> optlist;

/// This class is the abstract base class for the set of optimizations that
/// corresponds to one library call. The SimplifyLibCalls pass will call the
/// ValidateCalledFunction method to ask the optimization if a given Function
/// is the kind that the optimization can handle. If the subclass returns true,
/// then SImplifyLibCalls will also call the OptimizeCall method to perform, 
/// or attempt to perform, the optimization(s) for the library call. Otherwise,
/// OptimizeCall won't be called. Subclasses are responsible for providing the
/// name of the library call (strlen, strcpy, etc.) to the LibCallOptimization
/// constructor. This is used to efficiently select which call instructions to
/// optimize. The criteria for a "lib call" is "anything with well known 
/// semantics", typically a library function that is defined by an international
/// standard. Because the semantics are well known, the optimizations can 
/// generally short-circuit actually calling the function if there's a simpler
/// way (e.g. strlen(X) can be reduced to a constant if X is a constant global).
/// @brief Base class for library call optimizations
struct LibCallOptimization
{
  /// The \p fname argument must be the name of the library function being 
  /// optimized by the subclass.
  /// @brief Constructor that registers the optimization.
  LibCallOptimization(const char * fname )
    : func_name(fname)
#ifndef NDEBUG
    , stat_name(std::string("simplify-libcalls:")+fname)
    , occurrences(stat_name.c_str(),"Number of calls simplified") 
#endif
  {
    // Register this call optimizer in the optlist (a hash_map)
    optlist[func_name] = this;
  }

  /// @brief Deregister from the optlist
  virtual ~LibCallOptimization() { optlist.erase(func_name); }

  /// The implementation of this function in subclasses should determine if
  /// \p F is suitable for the optimization. This method is called by 
  /// SimplifyLibCalls::runOnModule to short circuit visiting all the call 
  /// sites of such a function if that function is not suitable in the first 
  /// place.  If the called function is suitabe, this method should return true;
  /// false, otherwise. This function should also perform any lazy 
  /// initialization that the LibCallOptimization needs to do, if its to return 
  /// true. This avoids doing initialization until the optimizer is actually
  /// going to be called upon to do some optimization.
  /// @brief Determine if the function is suitable for optimization
  virtual bool ValidateCalledFunction(
    const Function* F,    ///< The function that is the target of call sites
    SimplifyLibCalls& SLC ///< The pass object invoking us
  ) = 0;

  /// The implementations of this function in subclasses is the heart of the 
  /// SimplifyLibCalls algorithm. Sublcasses of this class implement 
  /// OptimizeCall to determine if (a) the conditions are right for optimizing
  /// the call and (b) to perform the optimization. If an action is taken 
  /// against ci, the subclass is responsible for returning true and ensuring
  /// that ci is erased from its parent.
  /// @brief Optimize a call, if possible.
  virtual bool OptimizeCall(
    CallInst* ci,          ///< The call instruction that should be optimized.
    SimplifyLibCalls& SLC  ///< The pass object invoking us
  ) = 0;

  /// @brief Get the name of the library call being optimized
  const char * getFunctionName() const { return func_name; }

#ifndef NDEBUG
  /// @brief Called by SimplifyLibCalls to update the occurrences statistic.
  void succeeded() { ++occurrences; }
#endif

private:
  const char* func_name; ///< Name of the library call we optimize
#ifndef NDEBUG
  std::string stat_name; ///< Holder for debug statistic name
  Statistic<> occurrences; ///< debug statistic (-debug-only=simplify-libcalls)
#endif
};

/// This class is an LLVM Pass that applies each of the LibCallOptimization 
/// instances to all the call sites in a module, relatively efficiently. The
/// purpose of this pass is to provide optimizations for calls to well-known 
/// functions with well-known semantics, such as those in the c library. The
/// class provides the basic infrastructure for handling runOnModule.  Whenever /// this pass finds a function call, it asks the appropriate optimizer to 
/// validate the call (ValidateLibraryCall). If it is validated, then
/// the OptimizeCall method is also called.
/// @brief A ModulePass for optimizing well-known function calls.
struct SimplifyLibCalls : public ModulePass 
{
  /// We need some target data for accurate signature details that are
  /// target dependent. So we require target data in our AnalysisUsage.
  /// @brief Require TargetData from AnalysisUsage.
  virtual void getAnalysisUsage(AnalysisUsage& Info) const
  {
    // Ask that the TargetData analysis be performed before us so we can use
    // the target data.
    Info.addRequired<TargetData>();
  }

  /// For this pass, process all of the function calls in the module, calling
  /// ValidateLibraryCall and OptimizeCall as appropriate.
  /// @brief Run all the lib call optimizations on a Module.
  virtual bool runOnModule(Module &M)
  {
    reset(M);

    bool result = false;

    // The call optimizations can be recursive. That is, the optimization might
    // generate a call to another function which can also be optimized. This way
    // we make the LibCallOptimization instances very specific to the case they 
    // handle. It also means we need to keep running over the function calls in 
    // the module until we don't get any more optimizations possible.
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
        if (!FI->isExternal() || !FI->hasExternalLinkage() || FI->use_empty())
          continue;

        // Get the optimization class that pertains to this function
        LibCallOptimization* CO = optlist[FI->getName().c_str()];
        if (!CO)
          continue;

        // Make sure the called function is suitable for the optimization
        if (!CO->ValidateCalledFunction(FI,*this))
          continue;

        // Loop over each of the uses of the function
        for (Value::use_iterator UI = FI->use_begin(), UE = FI->use_end(); 
             UI != UE ; )
        {
          // If the use of the function is a call instruction
          if (CallInst* CI = dyn_cast<CallInst>(*UI++))
          {
            // Do the optimization on the LibCallOptimization.
            if (CO->OptimizeCall(CI,*this))
            {
              ++SimplifiedLibCalls;
              found_optimization = result = true;
#ifndef NDEBUG
              CO->succeeded();
#endif
            }
          }
        }
      }
    } while (found_optimization);
    return result;
  }

  /// @brief Return the *current* module we're working on.
  Module* getModule() { return M; }

  /// @brief Return the *current* target data for the module we're working on.
  TargetData* getTargetData() { return TD; }

  /// @brief Return a Function* for the strlen libcall
  Function* get_strlen()
  {
    if (!strlen_func)
    {
      std::vector<const Type*> args;
      args.push_back(PointerType::get(Type::SByteTy));
      FunctionType* strlen_type = 
        FunctionType::get(TD->getIntPtrType(), args, false);
      strlen_func = M->getOrInsertFunction("strlen",strlen_type);
    }
    return strlen_func;
  }

  /// @brief Return a Function* for the memcpy libcall
  Function* get_memcpy()
  {
    if (!memcpy_func)
    {
      // Note: this is for llvm.memcpy intrinsic
      std::vector<const Type*> args;
      args.push_back(PointerType::get(Type::SByteTy));
      args.push_back(PointerType::get(Type::SByteTy));
      args.push_back(Type::IntTy);
      args.push_back(Type::IntTy);
      FunctionType* memcpy_type = FunctionType::get(Type::VoidTy, args, false);
      memcpy_func = M->getOrInsertFunction("llvm.memcpy",memcpy_type);
    }
    return memcpy_func;
  }

private:
  /// @brief Reset our cached data for a new Module
  void reset(Module& mod)
  {
    M = &mod;
    TD = &getAnalysis<TargetData>();
    memcpy_func = 0;
    strlen_func = 0;
  }

private:
  Function* memcpy_func; ///< Cached llvm.memcpy function
  Function* strlen_func; ///< Cached strlen function
  Module* M;             ///< Cached Module
  TargetData* TD;        ///< Cached TargetData
};

// Register the pass
RegisterOpt<SimplifyLibCalls> 
X("simplify-libcalls","Simplify well-known library calls");

} // anonymous namespace

// The only public symbol in this file which just instantiates the pass object
ModulePass *llvm::createSimplifyLibCallsPass() 
{ 
  return new SimplifyLibCalls(); 
}

// Classes below here, in the anonymous namespace, are all subclasses of the
// LibCallOptimization class, each implementing all optimizations possible for a
// single well-known library call. Each has a static singleton instance that
// auto registers it into the "optlist" global above. 
namespace {

// Forward declare a utility function.
bool getConstantStringLength(Value* V, uint64_t& len );

/// This LibCallOptimization will find instances of a call to "exit" that occurs
/// within the "main" function and change it to a simple "ret" instruction with
/// the same value passed to the exit function. When this is done, it splits the
/// basic block at the exit(3) call and deletes the call instruction.
/// @brief Replace calls to exit in main with a simple return
struct ExitInMainOptimization : public LibCallOptimization
{
  ExitInMainOptimization() : LibCallOptimization("exit") {}
  virtual ~ExitInMainOptimization() {}

  // Make sure the called function looks like exit (int argument, int return
  // type, external linkage, not varargs). 
  virtual bool ValidateCalledFunction(const Function* f, SimplifyLibCalls& SLC)
  {
    if (f->arg_size() >= 1)
      if (f->arg_begin()->getType()->isInteger())
        return true;
    return false;
  }

  virtual bool OptimizeCall(CallInst* ci, SimplifyLibCalls& SLC)
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

/// This LibCallOptimization will simplify a call to the strcat library 
/// function. The simplification is possible only if the string being 
/// concatenated is a constant array or a constant expression that results in 
/// a constant string. In this case we can replace it with strlen + llvm.memcpy 
/// of the constant string. Both of these calls are further reduced, if possible
/// on subsequent passes.
/// @brief Simplify the strcat library function.
struct StrCatOptimization : public LibCallOptimization
{
public:
  /// @brief Default constructor
  StrCatOptimization() : LibCallOptimization("strcat") {}

public:
  /// @breif  Destructor
  virtual ~StrCatOptimization() {}

  /// @brief Make sure that the "strcat" function has the right prototype
  virtual bool ValidateCalledFunction(const Function* f, SimplifyLibCalls& SLC) 
  {
    if (f->getReturnType() == PointerType::get(Type::SByteTy))
      if (f->arg_size() == 2) 
      {
        Function::const_arg_iterator AI = f->arg_begin();
        if (AI++->getType() == PointerType::get(Type::SByteTy))
          if (AI->getType() == PointerType::get(Type::SByteTy))
          {
            // Indicate this is a suitable call type.
            return true;
          }
      }
    return false;
  }

  /// @brief Optimize the strcat library function
  virtual bool OptimizeCall(CallInst* ci, SimplifyLibCalls& SLC)
  {
    // Extract some information from the instruction
    Module* M = ci->getParent()->getParent()->getParent();
    Value* dest = ci->getOperand(1);
    Value* src  = ci->getOperand(2);

    // Extract the initializer (while making numerous checks) from the 
    // source operand of the call to strcat. If we get null back, one of
    // a variety of checks in get_GVInitializer failed
    uint64_t len = 0;
    if (!getConstantStringLength(src,len))
      return false;

    // Handle the simple, do-nothing case
    if (len == 0)
    {
      ci->replaceAllUsesWith(dest);
      ci->eraseFromParent();
      return true;
    }

    // Increment the length because we actually want to memcpy the null
    // terminator as well.
    len++;


    // We need to find the end of the destination string.  That's where the 
    // memory is to be moved to. We just generate a call to strlen (further 
    // optimized in another pass).  Note that the SLC.get_strlen() call 
    // caches the Function* for us.
    CallInst* strlen_inst = 
      new CallInst(SLC.get_strlen(), dest, dest->getName()+".len",ci);

    // Now that we have the destination's length, we must index into the 
    // destination's pointer to get the actual memcpy destination (end of
    // the string .. we're concatenating).
    std::vector<Value*> idx;
    idx.push_back(strlen_inst);
    GetElementPtrInst* gep = 
      new GetElementPtrInst(dest,idx,dest->getName()+".indexed",ci);

    // We have enough information to now generate the memcpy call to
    // do the concatenation for us.
    std::vector<Value*> vals;
    vals.push_back(gep); // destination
    vals.push_back(ci->getOperand(2)); // source
    vals.push_back(ConstantSInt::get(Type::IntTy,len)); // length
    vals.push_back(ConstantSInt::get(Type::IntTy,1)); // alignment
    new CallInst(SLC.get_memcpy(), vals, "", ci);

    // Finally, substitute the first operand of the strcat call for the 
    // strcat call itself since strcat returns its first operand; and, 
    // kill the strcat CallInst.
    ci->replaceAllUsesWith(dest);
    ci->eraseFromParent();
    return true;
  }
} StrCatOptimizer;

/// This LibCallOptimization will simplify a call to the strcpy library 
/// function.  Two optimizations are possible: 
/// (1) If src and dest are the same and not volatile, just return dest
/// (2) If the src is a constant then we can convert to llvm.memmove
/// @brief Simplify the strcpy library function.
struct StrCpyOptimization : public LibCallOptimization
{
public:
  StrCpyOptimization() : LibCallOptimization("strcpy") {}
  virtual ~StrCpyOptimization() {}

  /// @brief Make sure that the "strcpy" function has the right prototype
  virtual bool ValidateCalledFunction(const Function* f, SimplifyLibCalls& SLC) 
  {
    if (f->getReturnType() == PointerType::get(Type::SByteTy))
      if (f->arg_size() == 2) 
      {
        Function::const_arg_iterator AI = f->arg_begin();
        if (AI++->getType() == PointerType::get(Type::SByteTy))
          if (AI->getType() == PointerType::get(Type::SByteTy))
          {
            // Indicate this is a suitable call type.
            return true;
          }
      }
    return false;
  }

  /// @brief Perform the strcpy optimization
  virtual bool OptimizeCall(CallInst* ci, SimplifyLibCalls& SLC)
  {
    // First, check to see if src and destination are the same. If they are,
    // then the optimization is to replace the CallInst with the destination
    // because the call is a no-op. Note that this corresponds to the 
    // degenerate strcpy(X,X) case which should have "undefined" results
    // according to the C specification. However, it occurs sometimes and
    // we optimize it as a no-op.
    Value* dest = ci->getOperand(1);
    Value* src = ci->getOperand(2);
    if (dest == src)
    {
      ci->replaceAllUsesWith(dest);
      ci->eraseFromParent();
      return true;
    }
    
    // Get the length of the constant string referenced by the second operand,
    // the "src" parameter. Fail the optimization if we can't get the length
    // (note that getConstantStringLength does lots of checks to make sure this
    // is valid).
    uint64_t len = 0;
    if (!getConstantStringLength(ci->getOperand(2),len))
      return false;

    // If the constant string's length is zero we can optimize this by just
    // doing a store of 0 at the first byte of the destination
    if (len == 0)
    {
      new StoreInst(ConstantInt::get(Type::SByteTy,0),ci->getOperand(1),ci);
      ci->replaceAllUsesWith(dest);
      ci->eraseFromParent();
      return true;
    }

    // Increment the length because we actually want to memcpy the null
    // terminator as well.
    len++;

    // Extract some information from the instruction
    Module* M = ci->getParent()->getParent()->getParent();

    // We have enough information to now generate the memcpy call to
    // do the concatenation for us.
    std::vector<Value*> vals;
    vals.push_back(dest); // destination
    vals.push_back(src); // source
    vals.push_back(ConstantSInt::get(Type::IntTy,len)); // length
    vals.push_back(ConstantSInt::get(Type::IntTy,1)); // alignment
    new CallInst(SLC.get_memcpy(), vals, "", ci);

    // Finally, substitute the first operand of the strcat call for the 
    // strcat call itself since strcat returns its first operand; and, 
    // kill the strcat CallInst.
    ci->replaceAllUsesWith(dest);
    ci->eraseFromParent();
    return true;
  }
} StrCpyOptimizer;

/// This LibCallOptimization will simplify a call to the strlen library 
/// function by replacing it with a constant value if the string provided to 
/// it is a constant array.
/// @brief Simplify the strlen library function.
struct StrLenOptimization : public LibCallOptimization
{
  StrLenOptimization() : LibCallOptimization("strlen") {}
  virtual ~StrLenOptimization() {}

  /// @brief Make sure that the "strlen" function has the right prototype
  virtual bool ValidateCalledFunction(const Function* f, SimplifyLibCalls& SLC)
  {
    if (f->getReturnType() == SLC.getTargetData()->getIntPtrType())
      if (f->arg_size() == 1) 
        if (Function::const_arg_iterator AI = f->arg_begin())
          if (AI->getType() == PointerType::get(Type::SByteTy))
            return true;
    return false;
  }

  /// @brief Perform the strlen optimization
  virtual bool OptimizeCall(CallInst* ci, SimplifyLibCalls& SLC)
  {
    // Get the length of the string
    uint64_t len = 0;
    if (!getConstantStringLength(ci->getOperand(1),len))
      return false;

    ci->replaceAllUsesWith(
        ConstantInt::get(SLC.getTargetData()->getIntPtrType(),len));
    ci->eraseFromParent();
    return true;
  }
} StrLenOptimizer;

/// This LibCallOptimization will simplify a call to the memcpy library 
/// function by expanding it out to a single store of size 0, 1, 2, 4, or 8 
/// bytes depending on the length of the string and the alignment. Additional
/// optimizations are possible in code generation (sequence of immediate store)
/// @brief Simplify the memcpy library function.
struct MemCpyOptimization : public LibCallOptimization
{
  /// @brief Default Constructor
  MemCpyOptimization() : LibCallOptimization("llvm.memcpy") {}
protected:
  /// @brief Subclass Constructor 
  MemCpyOptimization(const char* fname) : LibCallOptimization(fname) {}
public:
  /// @brief Destructor
  virtual ~MemCpyOptimization() {}

  /// @brief Make sure that the "memcpy" function has the right prototype
  virtual bool ValidateCalledFunction(const Function* f, SimplifyLibCalls& TD)
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
  virtual bool OptimizeCall(CallInst* ci, SimplifyLibCalls& TD)
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

    // Get the type we will cast to, based on size of the string
    Value* dest = ci->getOperand(1);
    Value* src = ci->getOperand(2);
    Type* castType = 0;
    switch (len)
    {
      case 0:
        // The memcpy is a no-op so just dump its call.
        ci->eraseFromParent();
        return true;
      case 1: castType = Type::SByteTy; break;
      case 2: castType = Type::ShortTy; break;
      case 4: castType = Type::IntTy; break;
      case 8: castType = Type::LongTy; break;
      default:
        return false;
    }

    // Cast source and dest to the right sized primitive and then load/store
    CastInst* SrcCast = 
      new CastInst(src,PointerType::get(castType),src->getName()+".cast",ci);
    CastInst* DestCast = 
      new CastInst(dest,PointerType::get(castType),dest->getName()+".cast",ci);
    LoadInst* LI = new LoadInst(SrcCast,SrcCast->getName()+".val",ci);
    StoreInst* SI = new StoreInst(LI, DestCast, ci);
    ci->eraseFromParent();
    return true;
  }
} MemCpyOptimizer;

/// This LibCallOptimization will simplify a call to the memmove library 
/// function. It is identical to MemCopyOptimization except for the name of 
/// the intrinsic.
/// @brief Simplify the memmove library function.
struct MemMoveOptimization : public MemCpyOptimization
{
  /// @brief Default Constructor
  MemMoveOptimization() : MemCpyOptimization("llvm.memmove") {}

} MemMoveOptimizer;

/// A function to compute the length of a null-terminated constant array of
/// integers.  This function can't rely on the size of the constant array 
/// because there could be a null terminator in the middle of the array. 
/// We also have to bail out if we find a non-integer constant initializer 
/// of one of the elements or if there is no null-terminator. The logic 
/// below checks each of these conditions and will return true only if all
/// conditions are met. In that case, the \p len parameter is set to the length
/// of the null-terminated string. If false is returned, the conditions were
/// not met and len is set to 0.
/// @brief Get the length of a constant string (null-terminated array).
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

// TODO: Additional cases that we need to add to this file:
// 1. memmove -> memcpy if src is a global constant array
}
