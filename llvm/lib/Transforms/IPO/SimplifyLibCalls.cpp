//===- SimplifyLibCalls.cpp - Optimize specific well-known library calls --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a module pass that applies a variety of small
// optimizations for calls to specific well-known function calls (e.g. runtime
// library functions). For example, a call to the function "exit(3)" that
// occurs within the main() function can be transformed into a simple "return 3"
// instruction. Any optimization that takes this form (replace call to library
// function with simpler code that provides the same result) belongs in this
// file.
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
#include "llvm/Config/config.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Transforms/IPO.h"
using namespace llvm;

namespace {

/// This statistic keeps track of the total number of library calls that have
/// been simplified regardless of which call it is.
Statistic<> SimplifiedLibCalls("simplify-libcalls",
  "Number of library calls simplified");

// Forward declarations
class LibCallOptimization;
class SimplifyLibCalls;

/// This list is populated by the constructor for LibCallOptimization class.
/// Therefore all subclasses are registered here at static initialization time
/// and this list is what the SimplifyLibCalls pass uses to apply the individual
/// optimizations to the call sites.
/// @brief The list of optimizations deriving from LibCallOptimization
static LibCallOptimization *OptList = 0;

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
class LibCallOptimization {
  LibCallOptimization **Prev, *Next;
  const char *FunctionName; ///< Name of the library call we optimize
#ifndef NDEBUG
  Statistic<> occurrences; ///< debug statistic (-debug-only=simplify-libcalls)
#endif
public:
  /// The \p fname argument must be the name of the library function being
  /// optimized by the subclass.
  /// @brief Constructor that registers the optimization.
  LibCallOptimization(const char *FName, const char *Description)
    : FunctionName(FName)
#ifndef NDEBUG
    , occurrences("simplify-libcalls", Description)
#endif
  {
    // Register this optimizer in the list of optimizations.
    Next = OptList;
    OptList = this;
    Prev = &OptList;
    if (Next) Next->Prev = &Next;
  }
  
  /// getNext - All libcall optimizations are chained together into a list,
  /// return the next one in the list.
  LibCallOptimization *getNext() { return Next; }

  /// @brief Deregister from the optlist
  virtual ~LibCallOptimization() {
    *Prev = Next;
    if (Next) Next->Prev = Prev;
  }

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
  const char *getFunctionName() const { return FunctionName; }

  /// @brief Called by SimplifyLibCalls to update the occurrences statistic.
  void succeeded() {
#ifndef NDEBUG
    DEBUG(++occurrences);
#endif
  }
};

/// This class is an LLVM Pass that applies each of the LibCallOptimization
/// instances to all the call sites in a module, relatively efficiently. The
/// purpose of this pass is to provide optimizations for calls to well-known
/// functions with well-known semantics, such as those in the c library. The
/// class provides the basic infrastructure for handling runOnModule.  Whenever
/// this pass finds a function call, it asks the appropriate optimizer to
/// validate the call (ValidateLibraryCall). If it is validated, then
/// the OptimizeCall method is also called.
/// @brief A ModulePass for optimizing well-known function calls.
class SimplifyLibCalls : public ModulePass {
public:
  /// We need some target data for accurate signature details that are
  /// target dependent. So we require target data in our AnalysisUsage.
  /// @brief Require TargetData from AnalysisUsage.
  virtual void getAnalysisUsage(AnalysisUsage& Info) const {
    // Ask that the TargetData analysis be performed before us so we can use
    // the target data.
    Info.addRequired<TargetData>();
  }

  /// For this pass, process all of the function calls in the module, calling
  /// ValidateLibraryCall and OptimizeCall as appropriate.
  /// @brief Run all the lib call optimizations on a Module.
  virtual bool runOnModule(Module &M) {
    reset(M);

    bool result = false;
    hash_map<std::string, LibCallOptimization*> OptznMap;
    for (LibCallOptimization *Optzn = OptList; Optzn; Optzn = Optzn->getNext())
      OptznMap[Optzn->getFunctionName()] = Optzn;

    // The call optimizations can be recursive. That is, the optimization might
    // generate a call to another function which can also be optimized. This way
    // we make the LibCallOptimization instances very specific to the case they
    // handle. It also means we need to keep running over the function calls in
    // the module until we don't get any more optimizations possible.
    bool found_optimization = false;
    do {
      found_optimization = false;
      for (Module::iterator FI = M.begin(), FE = M.end(); FI != FE; ++FI) {
        // All the "well-known" functions are external and have external linkage
        // because they live in a runtime library somewhere and were (probably)
        // not compiled by LLVM.  So, we only act on external functions that
        // have external linkage and non-empty uses.
        if (!FI->isExternal() || !FI->hasExternalLinkage() || FI->use_empty())
          continue;

        // Get the optimization class that pertains to this function
        hash_map<std::string, LibCallOptimization*>::iterator OMI =
          OptznMap.find(FI->getName());
        if (OMI == OptznMap.end()) continue;
        
        LibCallOptimization *CO = OMI->second;

        // Make sure the called function is suitable for the optimization
        if (!CO->ValidateCalledFunction(FI, *this))
          continue;

        // Loop over each of the uses of the function
        for (Value::use_iterator UI = FI->use_begin(), UE = FI->use_end();
             UI != UE ; ) {
          // If the use of the function is a call instruction
          if (CallInst* CI = dyn_cast<CallInst>(*UI++)) {
            // Do the optimization on the LibCallOptimization.
            if (CO->OptimizeCall(CI, *this)) {
              ++SimplifiedLibCalls;
              found_optimization = result = true;
              CO->succeeded();
            }
          }
        }
      }
    } while (found_optimization);
    
    return result;
  }

  /// @brief Return the *current* module we're working on.
  Module* getModule() const { return M; }

  /// @brief Return the *current* target data for the module we're working on.
  TargetData* getTargetData() const { return TD; }

  /// @brief Return the size_t type -- syntactic shortcut
  const Type* getIntPtrType() const { return TD->getIntPtrType(); }

  /// @brief Return a Function* for the putchar libcall
  Function* get_putchar() {
    if (!putchar_func)
      putchar_func = M->getOrInsertFunction("putchar", Type::IntTy, Type::IntTy,
                                            NULL);
    return putchar_func;
  }

  /// @brief Return a Function* for the puts libcall
  Function* get_puts() {
    if (!puts_func)
      puts_func = M->getOrInsertFunction("puts", Type::IntTy,
                                         PointerType::get(Type::SByteTy),
                                         NULL);
    return puts_func;
  }

  /// @brief Return a Function* for the fputc libcall
  Function* get_fputc(const Type* FILEptr_type) {
    if (!fputc_func)
      fputc_func = M->getOrInsertFunction("fputc", Type::IntTy, Type::IntTy,
                                          FILEptr_type, NULL);
    return fputc_func;
  }

  /// @brief Return a Function* for the fputs libcall
  Function* get_fputs(const Type* FILEptr_type) {
    if (!fputs_func)
      fputs_func = M->getOrInsertFunction("fputs", Type::IntTy,
                                          PointerType::get(Type::SByteTy),
                                          FILEptr_type, NULL);
    return fputs_func;
  }

  /// @brief Return a Function* for the fwrite libcall
  Function* get_fwrite(const Type* FILEptr_type) {
    if (!fwrite_func)
      fwrite_func = M->getOrInsertFunction("fwrite", TD->getIntPtrType(),
                                           PointerType::get(Type::SByteTy),
                                           TD->getIntPtrType(),
                                           TD->getIntPtrType(),
                                           FILEptr_type, NULL);
    return fwrite_func;
  }

  /// @brief Return a Function* for the sqrt libcall
  Function* get_sqrt() {
    if (!sqrt_func)
      sqrt_func = M->getOrInsertFunction("sqrt", Type::DoubleTy, 
                                         Type::DoubleTy, NULL);
    return sqrt_func;
  }

  /// @brief Return a Function* for the strlen libcall
  Function* get_strcpy() {
    if (!strcpy_func)
      strcpy_func = M->getOrInsertFunction("strcpy",
                                           PointerType::get(Type::SByteTy),
                                           PointerType::get(Type::SByteTy),
                                           PointerType::get(Type::SByteTy),
                                           NULL);
    return strcpy_func;
  }

  /// @brief Return a Function* for the strlen libcall
  Function* get_strlen() {
    if (!strlen_func)
      strlen_func = M->getOrInsertFunction("strlen", TD->getIntPtrType(),
                                           PointerType::get(Type::SByteTy),
                                           NULL);
    return strlen_func;
  }

  /// @brief Return a Function* for the memchr libcall
  Function* get_memchr() {
    if (!memchr_func)
      memchr_func = M->getOrInsertFunction("memchr",
                                           PointerType::get(Type::SByteTy),
                                           PointerType::get(Type::SByteTy),
                                           Type::IntTy, TD->getIntPtrType(),
                                           NULL);
    return memchr_func;
  }

  /// @brief Return a Function* for the memcpy libcall
  Function* get_memcpy() {
    if (!memcpy_func) {
      const Type *SBP = PointerType::get(Type::SByteTy);
      const char *N = TD->getIntPtrType() == Type::UIntTy ?
                            "llvm.memcpy.i32" : "llvm.memcpy.i64";
      memcpy_func = M->getOrInsertFunction(N, Type::VoidTy, SBP, SBP,
                                           TD->getIntPtrType(), Type::UIntTy,
                                           NULL);
    }
    return memcpy_func;
  }

  Function *getUnaryFloatFunction(const char *Name, Function *&Cache) {
    if (!Cache)
      Cache = M->getOrInsertFunction(Name, Type::FloatTy, Type::FloatTy, NULL);
    return Cache;
  }
  
  Function *get_floorf() { return getUnaryFloatFunction("floorf", floorf_func);}
  Function *get_ceilf()  { return getUnaryFloatFunction( "ceilf",  ceilf_func);}
  Function *get_roundf() { return getUnaryFloatFunction("roundf", roundf_func);}
  Function *get_rintf()  { return getUnaryFloatFunction( "rintf",  rintf_func);}
  Function *get_nearbyintf() { return getUnaryFloatFunction("nearbyintf",
                                                            nearbyintf_func); }
private:
  /// @brief Reset our cached data for a new Module
  void reset(Module& mod) {
    M = &mod;
    TD = &getAnalysis<TargetData>();
    putchar_func = 0;
    puts_func = 0;
    fputc_func = 0;
    fputs_func = 0;
    fwrite_func = 0;
    memcpy_func = 0;
    memchr_func = 0;
    sqrt_func   = 0;
    strcpy_func = 0;
    strlen_func = 0;
    floorf_func = 0;
    ceilf_func = 0;
    roundf_func = 0;
    rintf_func = 0;
    nearbyintf_func = 0;
  }

private:
  /// Caches for function pointers.
  Function *putchar_func, *puts_func;
  Function *fputc_func, *fputs_func, *fwrite_func;
  Function *memcpy_func, *memchr_func;
  Function* sqrt_func;
  Function *strcpy_func, *strlen_func;
  Function *floorf_func, *ceilf_func, *roundf_func;
  Function *rintf_func, *nearbyintf_func;
  Module *M;             ///< Cached Module
  TargetData *TD;        ///< Cached TargetData
};

// Register the pass
RegisterPass<SimplifyLibCalls>
X("simplify-libcalls", "Simplify well-known library calls");

} // anonymous namespace

// The only public symbol in this file which just instantiates the pass object
ModulePass *llvm::createSimplifyLibCallsPass() {
  return new SimplifyLibCalls();
}

// Classes below here, in the anonymous namespace, are all subclasses of the
// LibCallOptimization class, each implementing all optimizations possible for a
// single well-known library call. Each has a static singleton instance that
// auto registers it into the "optlist" global above.
namespace {

// Forward declare utility functions.
bool getConstantStringLength(Value* V, uint64_t& len, ConstantArray** A = 0 );
Value *CastToCStr(Value *V, Instruction &IP);

/// This LibCallOptimization will find instances of a call to "exit" that occurs
/// within the "main" function and change it to a simple "ret" instruction with
/// the same value passed to the exit function. When this is done, it splits the
/// basic block at the exit(3) call and deletes the call instruction.
/// @brief Replace calls to exit in main with a simple return
struct ExitInMainOptimization : public LibCallOptimization {
  ExitInMainOptimization() : LibCallOptimization("exit",
      "Number of 'exit' calls simplified") {}

  // Make sure the called function looks like exit (int argument, int return
  // type, external linkage, not varargs).
  virtual bool ValidateCalledFunction(const Function *F, SimplifyLibCalls &SLC){
    return F->arg_size() >= 1 && F->arg_begin()->getType()->isInteger();
  }

  virtual bool OptimizeCall(CallInst* ci, SimplifyLibCalls& SLC) {
    // To be careful, we check that the call to exit is coming from "main", that
    // main has external linkage, and the return type of main and the argument
    // to exit have the same type.
    Function *from = ci->getParent()->getParent();
    if (from->hasExternalLinkage())
      if (from->getReturnType() == ci->getOperand(1)->getType())
        if (from->getName() == "main") {
          // Okay, time to actually do the optimization. First, get the basic
          // block of the call instruction
          BasicBlock* bb = ci->getParent();

          // Create a return instruction that we'll replace the call with.
          // Note that the argument of the return is the argument of the call
          // instruction.
          new ReturnInst(ci->getOperand(1), ci);

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
struct StrCatOptimization : public LibCallOptimization {
public:
  /// @brief Default constructor
  StrCatOptimization() : LibCallOptimization("strcat",
      "Number of 'strcat' calls simplified") {}

public:

  /// @brief Make sure that the "strcat" function has the right prototype
  virtual bool ValidateCalledFunction(const Function* f, SimplifyLibCalls& SLC){
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
  virtual bool OptimizeCall(CallInst* ci, SimplifyLibCalls& SLC) {
    // Extract some information from the instruction
    Value* dest = ci->getOperand(1);
    Value* src  = ci->getOperand(2);

    // Extract the initializer (while making numerous checks) from the
    // source operand of the call to strcat. If we get null back, one of
    // a variety of checks in get_GVInitializer failed
    uint64_t len = 0;
    if (!getConstantStringLength(src,len))
      return false;

    // Handle the simple, do-nothing case
    if (len == 0) {
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
    vals.push_back(ConstantUInt::get(SLC.getIntPtrType(),len)); // length
    vals.push_back(ConstantUInt::get(Type::UIntTy,1)); // alignment
    new CallInst(SLC.get_memcpy(), vals, "", ci);

    // Finally, substitute the first operand of the strcat call for the
    // strcat call itself since strcat returns its first operand; and,
    // kill the strcat CallInst.
    ci->replaceAllUsesWith(dest);
    ci->eraseFromParent();
    return true;
  }
} StrCatOptimizer;

/// This LibCallOptimization will simplify a call to the strchr library
/// function.  It optimizes out cases where the arguments are both constant
/// and the result can be determined statically.
/// @brief Simplify the strcmp library function.
struct StrChrOptimization : public LibCallOptimization {
public:
  StrChrOptimization() : LibCallOptimization("strchr",
      "Number of 'strchr' calls simplified") {}

  /// @brief Make sure that the "strchr" function has the right prototype
  virtual bool ValidateCalledFunction(const Function* f, SimplifyLibCalls& SLC){
    if (f->getReturnType() == PointerType::get(Type::SByteTy) &&
        f->arg_size() == 2)
      return true;
    return false;
  }

  /// @brief Perform the strchr optimizations
  virtual bool OptimizeCall(CallInst *ci, SimplifyLibCalls &SLC) {
    // If there aren't three operands, bail
    if (ci->getNumOperands() != 3)
      return false;

    // Check that the first argument to strchr is a constant array of sbyte.
    // If it is, get the length and data, otherwise return false.
    uint64_t len = 0;
    ConstantArray* CA;
    if (!getConstantStringLength(ci->getOperand(1),len,&CA))
      return false;

    // Check that the second argument to strchr is a constant int, return false
    // if it isn't
    ConstantSInt* CSI = dyn_cast<ConstantSInt>(ci->getOperand(2));
    if (!CSI) {
      // Just lower this to memchr since we know the length of the string as
      // it is constant.
      Function* f = SLC.get_memchr();
      std::vector<Value*> args;
      args.push_back(ci->getOperand(1));
      args.push_back(ci->getOperand(2));
      args.push_back(ConstantUInt::get(SLC.getIntPtrType(),len));
      ci->replaceAllUsesWith( new CallInst(f,args,ci->getName(),ci));
      ci->eraseFromParent();
      return true;
    }

    // Get the character we're looking for
    int64_t chr = CSI->getValue();

    // Compute the offset
    uint64_t offset = 0;
    bool char_found = false;
    for (uint64_t i = 0; i < len; ++i) {
      if (ConstantSInt* CI = dyn_cast<ConstantSInt>(CA->getOperand(i))) {
        // Check for the null terminator
        if (CI->isNullValue())
          break; // we found end of string
        else if (CI->getValue() == chr) {
          char_found = true;
          offset = i;
          break;
        }
      }
    }

    // strchr(s,c)  -> offset_of_in(c,s)
    //    (if c is a constant integer and s is a constant string)
    if (char_found) {
      std::vector<Value*> indices;
      indices.push_back(ConstantUInt::get(Type::ULongTy,offset));
      GetElementPtrInst* GEP = new GetElementPtrInst(ci->getOperand(1),indices,
          ci->getOperand(1)->getName()+".strchr",ci);
      ci->replaceAllUsesWith(GEP);
    } else {
      ci->replaceAllUsesWith(
          ConstantPointerNull::get(PointerType::get(Type::SByteTy)));
    }
    ci->eraseFromParent();
    return true;
  }
} StrChrOptimizer;

/// This LibCallOptimization will simplify a call to the strcmp library
/// function.  It optimizes out cases where one or both arguments are constant
/// and the result can be determined statically.
/// @brief Simplify the strcmp library function.
struct StrCmpOptimization : public LibCallOptimization {
public:
  StrCmpOptimization() : LibCallOptimization("strcmp",
      "Number of 'strcmp' calls simplified") {}

  /// @brief Make sure that the "strcmp" function has the right prototype
  virtual bool ValidateCalledFunction(const Function *F, SimplifyLibCalls &SLC){
    return F->getReturnType() == Type::IntTy && F->arg_size() == 2;
  }

  /// @brief Perform the strcmp optimization
  virtual bool OptimizeCall(CallInst* ci, SimplifyLibCalls& SLC) {
    // First, check to see if src and destination are the same. If they are,
    // then the optimization is to replace the CallInst with a constant 0
    // because the call is a no-op.
    Value* s1 = ci->getOperand(1);
    Value* s2 = ci->getOperand(2);
    if (s1 == s2) {
      // strcmp(x,x)  -> 0
      ci->replaceAllUsesWith(ConstantInt::get(Type::IntTy,0));
      ci->eraseFromParent();
      return true;
    }

    bool isstr_1 = false;
    uint64_t len_1 = 0;
    ConstantArray* A1;
    if (getConstantStringLength(s1,len_1,&A1)) {
      isstr_1 = true;
      if (len_1 == 0) {
        // strcmp("",x) -> *x
        LoadInst* load =
          new LoadInst(CastToCStr(s2,*ci), ci->getName()+".load",ci);
        CastInst* cast =
          new CastInst(load,Type::IntTy,ci->getName()+".int",ci);
        ci->replaceAllUsesWith(cast);
        ci->eraseFromParent();
        return true;
      }
    }

    bool isstr_2 = false;
    uint64_t len_2 = 0;
    ConstantArray* A2;
    if (getConstantStringLength(s2, len_2, &A2)) {
      isstr_2 = true;
      if (len_2 == 0) {
        // strcmp(x,"") -> *x
        LoadInst* load =
          new LoadInst(CastToCStr(s1,*ci),ci->getName()+".val",ci);
        CastInst* cast =
          new CastInst(load,Type::IntTy,ci->getName()+".int",ci);
        ci->replaceAllUsesWith(cast);
        ci->eraseFromParent();
        return true;
      }
    }

    if (isstr_1 && isstr_2) {
      // strcmp(x,y)  -> cnst  (if both x and y are constant strings)
      std::string str1 = A1->getAsString();
      std::string str2 = A2->getAsString();
      int result = strcmp(str1.c_str(), str2.c_str());
      ci->replaceAllUsesWith(ConstantSInt::get(Type::IntTy,result));
      ci->eraseFromParent();
      return true;
    }
    return false;
  }
} StrCmpOptimizer;

/// This LibCallOptimization will simplify a call to the strncmp library
/// function.  It optimizes out cases where one or both arguments are constant
/// and the result can be determined statically.
/// @brief Simplify the strncmp library function.
struct StrNCmpOptimization : public LibCallOptimization {
public:
  StrNCmpOptimization() : LibCallOptimization("strncmp",
      "Number of 'strncmp' calls simplified") {}

  /// @brief Make sure that the "strncmp" function has the right prototype
  virtual bool ValidateCalledFunction(const Function* f, SimplifyLibCalls& SLC){
    if (f->getReturnType() == Type::IntTy && f->arg_size() == 3)
      return true;
    return false;
  }

  /// @brief Perform the strncpy optimization
  virtual bool OptimizeCall(CallInst *ci, SimplifyLibCalls &SLC) {
    // First, check to see if src and destination are the same. If they are,
    // then the optimization is to replace the CallInst with a constant 0
    // because the call is a no-op.
    Value* s1 = ci->getOperand(1);
    Value* s2 = ci->getOperand(2);
    if (s1 == s2) {
      // strncmp(x,x,l)  -> 0
      ci->replaceAllUsesWith(ConstantInt::get(Type::IntTy,0));
      ci->eraseFromParent();
      return true;
    }

    // Check the length argument, if it is Constant zero then the strings are
    // considered equal.
    uint64_t len_arg = 0;
    bool len_arg_is_const = false;
    if (ConstantInt* len_CI = dyn_cast<ConstantInt>(ci->getOperand(3))) {
      len_arg_is_const = true;
      len_arg = len_CI->getRawValue();
      if (len_arg == 0) {
        // strncmp(x,y,0)   -> 0
        ci->replaceAllUsesWith(ConstantInt::get(Type::IntTy,0));
        ci->eraseFromParent();
        return true;
      }
    }

    bool isstr_1 = false;
    uint64_t len_1 = 0;
    ConstantArray* A1;
    if (getConstantStringLength(s1, len_1, &A1)) {
      isstr_1 = true;
      if (len_1 == 0) {
        // strncmp("",x) -> *x
        LoadInst* load = new LoadInst(s1,ci->getName()+".load",ci);
        CastInst* cast =
          new CastInst(load,Type::IntTy,ci->getName()+".int",ci);
        ci->replaceAllUsesWith(cast);
        ci->eraseFromParent();
        return true;
      }
    }

    bool isstr_2 = false;
    uint64_t len_2 = 0;
    ConstantArray* A2;
    if (getConstantStringLength(s2,len_2,&A2)) {
      isstr_2 = true;
      if (len_2 == 0) {
        // strncmp(x,"") -> *x
        LoadInst* load = new LoadInst(s2,ci->getName()+".val",ci);
        CastInst* cast =
          new CastInst(load,Type::IntTy,ci->getName()+".int",ci);
        ci->replaceAllUsesWith(cast);
        ci->eraseFromParent();
        return true;
      }
    }

    if (isstr_1 && isstr_2 && len_arg_is_const) {
      // strncmp(x,y,const) -> constant
      std::string str1 = A1->getAsString();
      std::string str2 = A2->getAsString();
      int result = strncmp(str1.c_str(), str2.c_str(), len_arg);
      ci->replaceAllUsesWith(ConstantSInt::get(Type::IntTy,result));
      ci->eraseFromParent();
      return true;
    }
    return false;
  }
} StrNCmpOptimizer;

/// This LibCallOptimization will simplify a call to the strcpy library
/// function.  Two optimizations are possible:
/// (1) If src and dest are the same and not volatile, just return dest
/// (2) If the src is a constant then we can convert to llvm.memmove
/// @brief Simplify the strcpy library function.
struct StrCpyOptimization : public LibCallOptimization {
public:
  StrCpyOptimization() : LibCallOptimization("strcpy",
      "Number of 'strcpy' calls simplified") {}

  /// @brief Make sure that the "strcpy" function has the right prototype
  virtual bool ValidateCalledFunction(const Function* f, SimplifyLibCalls& SLC){
    if (f->getReturnType() == PointerType::get(Type::SByteTy))
      if (f->arg_size() == 2) {
        Function::const_arg_iterator AI = f->arg_begin();
        if (AI++->getType() == PointerType::get(Type::SByteTy))
          if (AI->getType() == PointerType::get(Type::SByteTy)) {
            // Indicate this is a suitable call type.
            return true;
          }
      }
    return false;
  }

  /// @brief Perform the strcpy optimization
  virtual bool OptimizeCall(CallInst* ci, SimplifyLibCalls& SLC) {
    // First, check to see if src and destination are the same. If they are,
    // then the optimization is to replace the CallInst with the destination
    // because the call is a no-op. Note that this corresponds to the
    // degenerate strcpy(X,X) case which should have "undefined" results
    // according to the C specification. However, it occurs sometimes and
    // we optimize it as a no-op.
    Value* dest = ci->getOperand(1);
    Value* src = ci->getOperand(2);
    if (dest == src) {
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
    if (len == 0) {
      new StoreInst(ConstantInt::get(Type::SByteTy,0),ci->getOperand(1),ci);
      ci->replaceAllUsesWith(dest);
      ci->eraseFromParent();
      return true;
    }

    // Increment the length because we actually want to memcpy the null
    // terminator as well.
    len++;

    // We have enough information to now generate the memcpy call to
    // do the concatenation for us.
    std::vector<Value*> vals;
    vals.push_back(dest); // destination
    vals.push_back(src); // source
    vals.push_back(ConstantUInt::get(SLC.getIntPtrType(),len)); // length
    vals.push_back(ConstantUInt::get(Type::UIntTy,1)); // alignment
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
struct StrLenOptimization : public LibCallOptimization {
  StrLenOptimization() : LibCallOptimization("strlen",
      "Number of 'strlen' calls simplified") {}

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
    // Make sure we're dealing with an sbyte* here.
    Value* str = ci->getOperand(1);
    if (str->getType() != PointerType::get(Type::SByteTy))
      return false;

    // Does the call to strlen have exactly one use?
    if (ci->hasOneUse())
      // Is that single use a binary operator?
      if (BinaryOperator* bop = dyn_cast<BinaryOperator>(ci->use_back()))
        // Is it compared against a constant integer?
        if (ConstantInt* CI = dyn_cast<ConstantInt>(bop->getOperand(1)))
        {
          // Get the value the strlen result is compared to
          uint64_t val = CI->getRawValue();

          // If its compared against length 0 with == or !=
          if (val == 0 &&
              (bop->getOpcode() == Instruction::SetEQ ||
               bop->getOpcode() == Instruction::SetNE))
          {
            // strlen(x) != 0 -> *x != 0
            // strlen(x) == 0 -> *x == 0
            LoadInst* load = new LoadInst(str,str->getName()+".first",ci);
            BinaryOperator* rbop = BinaryOperator::create(bop->getOpcode(),
              load, ConstantSInt::get(Type::SByteTy,0),
              bop->getName()+".strlen", ci);
            bop->replaceAllUsesWith(rbop);
            bop->eraseFromParent();
            ci->eraseFromParent();
            return true;
          }
        }

    // Get the length of the constant string operand
    uint64_t len = 0;
    if (!getConstantStringLength(ci->getOperand(1),len))
      return false;

    // strlen("xyz") -> 3 (for example)
    const Type *Ty = SLC.getTargetData()->getIntPtrType();
    if (Ty->isSigned())
      ci->replaceAllUsesWith(ConstantSInt::get(Ty, len));
    else
      ci->replaceAllUsesWith(ConstantUInt::get(Ty, len));
     
    ci->eraseFromParent();
    return true;
  }
} StrLenOptimizer;

/// IsOnlyUsedInEqualsComparison - Return true if it only matters that the value
/// is equal or not-equal to zero. 
static bool IsOnlyUsedInEqualsZeroComparison(Instruction *I) {
  for (Value::use_iterator UI = I->use_begin(), E = I->use_end();
       UI != E; ++UI) {
    Instruction *User = cast<Instruction>(*UI);
    if (User->getOpcode() == Instruction::SetNE ||
        User->getOpcode() == Instruction::SetEQ) {
      if (isa<Constant>(User->getOperand(1)) && 
          cast<Constant>(User->getOperand(1))->isNullValue())
        continue;
    } else if (CastInst *CI = dyn_cast<CastInst>(User))
      if (CI->getType() == Type::BoolTy)
        continue;
    // Unknown instruction.
    return false;
  }
  return true;
}

/// This memcmpOptimization will simplify a call to the memcmp library
/// function.
struct memcmpOptimization : public LibCallOptimization {
  /// @brief Default Constructor
  memcmpOptimization()
    : LibCallOptimization("memcmp", "Number of 'memcmp' calls simplified") {}
  
  /// @brief Make sure that the "memcmp" function has the right prototype
  virtual bool ValidateCalledFunction(const Function *F, SimplifyLibCalls &TD) {
    Function::const_arg_iterator AI = F->arg_begin();
    if (F->arg_size() != 3 || !isa<PointerType>(AI->getType())) return false;
    if (!isa<PointerType>((++AI)->getType())) return false;
    if (!(++AI)->getType()->isInteger()) return false;
    if (!F->getReturnType()->isInteger()) return false;
    return true;
  }
  
  /// Because of alignment and instruction information that we don't have, we
  /// leave the bulk of this to the code generators.
  ///
  /// Note that we could do much more if we could force alignment on otherwise
  /// small aligned allocas, or if we could indicate that loads have a small
  /// alignment.
  virtual bool OptimizeCall(CallInst *CI, SimplifyLibCalls &TD) {
    Value *LHS = CI->getOperand(1), *RHS = CI->getOperand(2);

    // If the two operands are the same, return zero.
    if (LHS == RHS) {
      // memcmp(s,s,x) -> 0
      CI->replaceAllUsesWith(Constant::getNullValue(CI->getType()));
      CI->eraseFromParent();
      return true;
    }
    
    // Make sure we have a constant length.
    ConstantInt *LenC = dyn_cast<ConstantInt>(CI->getOperand(3));
    if (!LenC) return false;
    uint64_t Len = LenC->getRawValue();
      
    // If the length is zero, this returns 0.
    switch (Len) {
    case 0:
      // memcmp(s1,s2,0) -> 0
      CI->replaceAllUsesWith(Constant::getNullValue(CI->getType()));
      CI->eraseFromParent();
      return true;
    case 1: {
      // memcmp(S1,S2,1) -> *(ubyte*)S1 - *(ubyte*)S2
      const Type *UCharPtr = PointerType::get(Type::UByteTy);
      CastInst *Op1Cast = new CastInst(LHS, UCharPtr, LHS->getName(), CI);
      CastInst *Op2Cast = new CastInst(RHS, UCharPtr, RHS->getName(), CI);
      Value *S1V = new LoadInst(Op1Cast, LHS->getName()+".val", CI);
      Value *S2V = new LoadInst(Op2Cast, RHS->getName()+".val", CI);
      Value *RV = BinaryOperator::createSub(S1V, S2V, CI->getName()+".diff",CI);
      if (RV->getType() != CI->getType())
        RV = new CastInst(RV, CI->getType(), RV->getName(), CI);
      CI->replaceAllUsesWith(RV);
      CI->eraseFromParent();
      return true;
    }
    case 2:
      if (IsOnlyUsedInEqualsZeroComparison(CI)) {
        // TODO: IF both are aligned, use a short load/compare.
      
        // memcmp(S1,S2,2) -> S1[0]-S2[0] | S1[1]-S2[1] iff only ==/!= 0 matters
        const Type *UCharPtr = PointerType::get(Type::UByteTy);
        CastInst *Op1Cast = new CastInst(LHS, UCharPtr, LHS->getName(), CI);
        CastInst *Op2Cast = new CastInst(RHS, UCharPtr, RHS->getName(), CI);
        Value *S1V1 = new LoadInst(Op1Cast, LHS->getName()+".val1", CI);
        Value *S2V1 = new LoadInst(Op2Cast, RHS->getName()+".val1", CI);
        Value *D1 = BinaryOperator::createSub(S1V1, S2V1,
                                              CI->getName()+".d1", CI);
        Constant *One = ConstantInt::get(Type::IntTy, 1);
        Value *G1 = new GetElementPtrInst(Op1Cast, One, "next1v", CI);
        Value *G2 = new GetElementPtrInst(Op2Cast, One, "next2v", CI);
        Value *S1V2 = new LoadInst(G1, LHS->getName()+".val2", CI);
        Value *S2V2 = new LoadInst(G2, RHS->getName()+".val2", CI);
        Value *D2 = BinaryOperator::createSub(S1V2, S2V2,
                                              CI->getName()+".d1", CI);
        Value *Or = BinaryOperator::createOr(D1, D2, CI->getName()+".res", CI);
        if (Or->getType() != CI->getType())
          Or = new CastInst(Or, CI->getType(), Or->getName(), CI);
        CI->replaceAllUsesWith(Or);
        CI->eraseFromParent();
        return true;
      }
      break;
    default:
      break;
    }
    
    return false;
  }
} memcmpOptimizer;


/// This LibCallOptimization will simplify a call to the memcpy library
/// function by expanding it out to a single store of size 0, 1, 2, 4, or 8
/// bytes depending on the length of the string and the alignment. Additional
/// optimizations are possible in code generation (sequence of immediate store)
/// @brief Simplify the memcpy library function.
struct LLVMMemCpyMoveOptzn : public LibCallOptimization {
  LLVMMemCpyMoveOptzn(const char* fname, const char* desc)
  : LibCallOptimization(fname, desc) {}

  /// @brief Make sure that the "memcpy" function has the right prototype
  virtual bool ValidateCalledFunction(const Function* f, SimplifyLibCalls& TD) {
    // Just make sure this has 4 arguments per LLVM spec.
    return (f->arg_size() == 4);
  }

  /// Because of alignment and instruction information that we don't have, we
  /// leave the bulk of this to the code generators. The optimization here just
  /// deals with a few degenerate cases where the length of the string and the
  /// alignment match the sizes of our intrinsic types so we can do a load and
  /// store instead of the memcpy call.
  /// @brief Perform the memcpy optimization.
  virtual bool OptimizeCall(CallInst* ci, SimplifyLibCalls& TD) {
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
    if (alignment == 0)
      alignment = 1; // Alignment 0 is identity for alignment 1
    if (len > alignment)
      return false;

    // Get the type we will cast to, based on size of the string
    Value* dest = ci->getOperand(1);
    Value* src = ci->getOperand(2);
    Type* castType = 0;
    switch (len)
    {
      case 0:
        // memcpy(d,s,0,a) -> noop
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
};

/// This LibCallOptimization will simplify a call to the memcpy/memmove library
/// functions.
LLVMMemCpyMoveOptzn LLVMMemCpyOptimizer32("llvm.memcpy.i32",
                                    "Number of 'llvm.memcpy' calls simplified");
LLVMMemCpyMoveOptzn LLVMMemCpyOptimizer64("llvm.memcpy.i64",
                                   "Number of 'llvm.memcpy' calls simplified");
LLVMMemCpyMoveOptzn LLVMMemMoveOptimizer32("llvm.memmove.i32",
                                   "Number of 'llvm.memmove' calls simplified");
LLVMMemCpyMoveOptzn LLVMMemMoveOptimizer64("llvm.memmove.i64",
                                   "Number of 'llvm.memmove' calls simplified");

/// This LibCallOptimization will simplify a call to the memset library
/// function by expanding it out to a single store of size 0, 1, 2, 4, or 8
/// bytes depending on the length argument.
struct LLVMMemSetOptimization : public LibCallOptimization {
  /// @brief Default Constructor
  LLVMMemSetOptimization(const char *Name) : LibCallOptimization(Name,
      "Number of 'llvm.memset' calls simplified") {}

  /// @brief Make sure that the "memset" function has the right prototype
  virtual bool ValidateCalledFunction(const Function *F, SimplifyLibCalls &TD) {
    // Just make sure this has 3 arguments per LLVM spec.
    return F->arg_size() == 4;
  }

  /// Because of alignment and instruction information that we don't have, we
  /// leave the bulk of this to the code generators. The optimization here just
  /// deals with a few degenerate cases where the length parameter is constant
  /// and the alignment matches the sizes of our intrinsic types so we can do
  /// store instead of the memcpy call. Other calls are transformed into the
  /// llvm.memset intrinsic.
  /// @brief Perform the memset optimization.
  virtual bool OptimizeCall(CallInst *ci, SimplifyLibCalls &TD) {
    // Make sure we have constant int values to work with
    ConstantInt* LEN = dyn_cast<ConstantInt>(ci->getOperand(3));
    if (!LEN)
      return false;
    ConstantInt* ALIGN = dyn_cast<ConstantInt>(ci->getOperand(4));
    if (!ALIGN)
      return false;

    // Extract the length and alignment
    uint64_t len = LEN->getRawValue();
    uint64_t alignment = ALIGN->getRawValue();

    // Alignment 0 is identity for alignment 1
    if (alignment == 0)
      alignment = 1;

    // If the length is zero, this is a no-op
    if (len == 0) {
      // memset(d,c,0,a) -> noop
      ci->eraseFromParent();
      return true;
    }

    // If the length is larger than the alignment, we can't optimize
    if (len > alignment)
      return false;

    // Make sure we have a constant ubyte to work with so we can extract
    // the value to be filled.
    ConstantUInt* FILL = dyn_cast<ConstantUInt>(ci->getOperand(2));
    if (!FILL)
      return false;
    if (FILL->getType() != Type::UByteTy)
      return false;

    // memset(s,c,n) -> store s, c (for n=1,2,4,8)

    // Extract the fill character
    uint64_t fill_char = FILL->getValue();
    uint64_t fill_value = fill_char;

    // Get the type we will cast to, based on size of memory area to fill, and
    // and the value we will store there.
    Value* dest = ci->getOperand(1);
    Type* castType = 0;
    switch (len) {
      case 1:
        castType = Type::UByteTy;
        break;
      case 2:
        castType = Type::UShortTy;
        fill_value |= fill_char << 8;
        break;
      case 4:
        castType = Type::UIntTy;
        fill_value |= fill_char << 8 | fill_char << 16 | fill_char << 24;
        break;
      case 8:
        castType = Type::ULongTy;
        fill_value |= fill_char << 8 | fill_char << 16 | fill_char << 24;
        fill_value |= fill_char << 32 | fill_char << 40 | fill_char << 48;
        fill_value |= fill_char << 56;
        break;
      default:
        return false;
    }

    // Cast dest to the right sized primitive and then load/store
    CastInst* DestCast =
      new CastInst(dest,PointerType::get(castType),dest->getName()+".cast",ci);
    new StoreInst(ConstantUInt::get(castType,fill_value),DestCast, ci);
    ci->eraseFromParent();
    return true;
  }
};

LLVMMemSetOptimization MemSet32Optimizer("llvm.memset.i32");
LLVMMemSetOptimization MemSet64Optimizer("llvm.memset.i64");


/// This LibCallOptimization will simplify calls to the "pow" library
/// function. It looks for cases where the result of pow is well known and
/// substitutes the appropriate value.
/// @brief Simplify the pow library function.
struct PowOptimization : public LibCallOptimization {
public:
  /// @brief Default Constructor
  PowOptimization() : LibCallOptimization("pow",
      "Number of 'pow' calls simplified") {}

  /// @brief Make sure that the "pow" function has the right prototype
  virtual bool ValidateCalledFunction(const Function* f, SimplifyLibCalls& SLC){
    // Just make sure this has 2 arguments
    return (f->arg_size() == 2);
  }

  /// @brief Perform the pow optimization.
  virtual bool OptimizeCall(CallInst *ci, SimplifyLibCalls &SLC) {
    const Type *Ty = cast<Function>(ci->getOperand(0))->getReturnType();
    Value* base = ci->getOperand(1);
    Value* expn = ci->getOperand(2);
    if (ConstantFP *Op1 = dyn_cast<ConstantFP>(base)) {
      double Op1V = Op1->getValue();
      if (Op1V == 1.0) {
        // pow(1.0,x) -> 1.0
        ci->replaceAllUsesWith(ConstantFP::get(Ty,1.0));
        ci->eraseFromParent();
        return true;
      }
    }  else if (ConstantFP* Op2 = dyn_cast<ConstantFP>(expn)) {
      double Op2V = Op2->getValue();
      if (Op2V == 0.0) {
        // pow(x,0.0) -> 1.0
        ci->replaceAllUsesWith(ConstantFP::get(Ty,1.0));
        ci->eraseFromParent();
        return true;
      } else if (Op2V == 0.5) {
        // pow(x,0.5) -> sqrt(x)
        CallInst* sqrt_inst = new CallInst(SLC.get_sqrt(), base,
            ci->getName()+".pow",ci);
        ci->replaceAllUsesWith(sqrt_inst);
        ci->eraseFromParent();
        return true;
      } else if (Op2V == 1.0) {
        // pow(x,1.0) -> x
        ci->replaceAllUsesWith(base);
        ci->eraseFromParent();
        return true;
      } else if (Op2V == -1.0) {
        // pow(x,-1.0)    -> 1.0/x
        BinaryOperator* div_inst= BinaryOperator::createDiv(
          ConstantFP::get(Ty,1.0), base, ci->getName()+".pow", ci);
        ci->replaceAllUsesWith(div_inst);
        ci->eraseFromParent();
        return true;
      }
    }
    return false; // opt failed
  }
} PowOptimizer;

/// This LibCallOptimization will simplify calls to the "printf" library
/// function. It looks for cases where the result of printf is not used and the
/// operation can be reduced to something simpler.
/// @brief Simplify the printf library function.
struct PrintfOptimization : public LibCallOptimization {
public:
  /// @brief Default Constructor
  PrintfOptimization() : LibCallOptimization("printf",
      "Number of 'printf' calls simplified") {}

  /// @brief Make sure that the "printf" function has the right prototype
  virtual bool ValidateCalledFunction(const Function* f, SimplifyLibCalls& SLC){
    // Just make sure this has at least 1 arguments
    return (f->arg_size() >= 1);
  }

  /// @brief Perform the printf optimization.
  virtual bool OptimizeCall(CallInst* ci, SimplifyLibCalls& SLC) {
    // If the call has more than 2 operands, we can't optimize it
    if (ci->getNumOperands() > 3 || ci->getNumOperands() <= 2)
      return false;

    // If the result of the printf call is used, none of these optimizations
    // can be made.
    if (!ci->use_empty())
      return false;

    // All the optimizations depend on the length of the first argument and the
    // fact that it is a constant string array. Check that now
    uint64_t len = 0;
    ConstantArray* CA = 0;
    if (!getConstantStringLength(ci->getOperand(1), len, &CA))
      return false;

    if (len != 2 && len != 3)
      return false;

    // The first character has to be a %
    if (ConstantInt* CI = dyn_cast<ConstantInt>(CA->getOperand(0)))
      if (CI->getRawValue() != '%')
        return false;

    // Get the second character and switch on its value
    ConstantInt* CI = dyn_cast<ConstantInt>(CA->getOperand(1));
    switch (CI->getRawValue()) {
      case 's':
      {
        if (len != 3 ||
            dyn_cast<ConstantInt>(CA->getOperand(2))->getRawValue() != '\n')
          return false;

        // printf("%s\n",str) -> puts(str)
        Function* puts_func = SLC.get_puts();
        if (!puts_func)
          return false;
        std::vector<Value*> args;
        args.push_back(CastToCStr(ci->getOperand(2), *ci));
        new CallInst(puts_func,args,ci->getName(),ci);
        ci->replaceAllUsesWith(ConstantSInt::get(Type::IntTy,len));
        break;
      }
      case 'c':
      {
        // printf("%c",c) -> putchar(c)
        if (len != 2)
          return false;

        Function* putchar_func = SLC.get_putchar();
        if (!putchar_func)
          return false;
        CastInst* cast = new CastInst(ci->getOperand(2), Type::IntTy,
                                      CI->getName()+".int", ci);
        new CallInst(putchar_func, cast, "", ci);
        ci->replaceAllUsesWith(ConstantSInt::get(Type::IntTy, 1));
        break;
      }
      default:
        return false;
    }
    ci->eraseFromParent();
    return true;
  }
} PrintfOptimizer;

/// This LibCallOptimization will simplify calls to the "fprintf" library
/// function. It looks for cases where the result of fprintf is not used and the
/// operation can be reduced to something simpler.
/// @brief Simplify the fprintf library function.
struct FPrintFOptimization : public LibCallOptimization {
public:
  /// @brief Default Constructor
  FPrintFOptimization() : LibCallOptimization("fprintf",
      "Number of 'fprintf' calls simplified") {}

  /// @brief Make sure that the "fprintf" function has the right prototype
  virtual bool ValidateCalledFunction(const Function* f, SimplifyLibCalls& SLC){
    // Just make sure this has at least 2 arguments
    return (f->arg_size() >= 2);
  }

  /// @brief Perform the fprintf optimization.
  virtual bool OptimizeCall(CallInst* ci, SimplifyLibCalls& SLC) {
    // If the call has more than 3 operands, we can't optimize it
    if (ci->getNumOperands() > 4 || ci->getNumOperands() <= 2)
      return false;

    // If the result of the fprintf call is used, none of these optimizations
    // can be made.
    if (!ci->use_empty())
      return false;

    // All the optimizations depend on the length of the second argument and the
    // fact that it is a constant string array. Check that now
    uint64_t len = 0;
    ConstantArray* CA = 0;
    if (!getConstantStringLength(ci->getOperand(2), len, &CA))
      return false;

    if (ci->getNumOperands() == 3) {
      // Make sure there's no % in the constant array
      for (unsigned i = 0; i < len; ++i) {
        if (ConstantInt* CI = dyn_cast<ConstantInt>(CA->getOperand(i))) {
          // Check for the null terminator
          if (CI->getRawValue() == '%')
            return false; // we found end of string
        } else {
          return false;
        }
      }

      // fprintf(file,fmt) -> fwrite(fmt,strlen(fmt),file)
      const Type* FILEptr_type = ci->getOperand(1)->getType();
      Function* fwrite_func = SLC.get_fwrite(FILEptr_type);
      if (!fwrite_func)
        return false;

      // Make sure that the fprintf() and fwrite() functions both take the
      // same type of char pointer.
      if (ci->getOperand(2)->getType() !=
          fwrite_func->getFunctionType()->getParamType(0))
        return false;

      std::vector<Value*> args;
      args.push_back(ci->getOperand(2));
      args.push_back(ConstantUInt::get(SLC.getIntPtrType(),len));
      args.push_back(ConstantUInt::get(SLC.getIntPtrType(),1));
      args.push_back(ci->getOperand(1));
      new CallInst(fwrite_func,args,ci->getName(),ci);
      ci->replaceAllUsesWith(ConstantSInt::get(Type::IntTy,len));
      ci->eraseFromParent();
      return true;
    }

    // The remaining optimizations require the format string to be length 2
    // "%s" or "%c".
    if (len != 2)
      return false;

    // The first character has to be a %
    if (ConstantInt* CI = dyn_cast<ConstantInt>(CA->getOperand(0)))
      if (CI->getRawValue() != '%')
        return false;

    // Get the second character and switch on its value
    ConstantInt* CI = dyn_cast<ConstantInt>(CA->getOperand(1));
    switch (CI->getRawValue()) {
      case 's':
      {
        uint64_t len = 0;
        ConstantArray* CA = 0;
        if (getConstantStringLength(ci->getOperand(3), len, &CA)) {
          // fprintf(file,"%s",str) -> fwrite(str,strlen(str),1,file)
          const Type* FILEptr_type = ci->getOperand(1)->getType();
          Function* fwrite_func = SLC.get_fwrite(FILEptr_type);
          if (!fwrite_func)
            return false;
          std::vector<Value*> args;
          args.push_back(CastToCStr(ci->getOperand(3), *ci));
          args.push_back(ConstantUInt::get(SLC.getIntPtrType(),len));
          args.push_back(ConstantUInt::get(SLC.getIntPtrType(),1));
          args.push_back(ci->getOperand(1));
          new CallInst(fwrite_func,args,ci->getName(),ci);
          ci->replaceAllUsesWith(ConstantSInt::get(Type::IntTy,len));
        } else {
          // fprintf(file,"%s",str) -> fputs(str,file)
          const Type* FILEptr_type = ci->getOperand(1)->getType();
          Function* fputs_func = SLC.get_fputs(FILEptr_type);
          if (!fputs_func)
            return false;
          std::vector<Value*> args;
          args.push_back(CastToCStr(ci->getOperand(3), *ci));
          args.push_back(ci->getOperand(1));
          new CallInst(fputs_func,args,ci->getName(),ci);
          ci->replaceAllUsesWith(ConstantSInt::get(Type::IntTy,len));
        }
        break;
      }
      case 'c':
      {
        // fprintf(file,"%c",c) -> fputc(c,file)
        const Type* FILEptr_type = ci->getOperand(1)->getType();
        Function* fputc_func = SLC.get_fputc(FILEptr_type);
        if (!fputc_func)
          return false;
        CastInst* cast = new CastInst(ci->getOperand(3), Type::IntTy,
                                      CI->getName()+".int", ci);
        new CallInst(fputc_func,cast,ci->getOperand(1),"",ci);
        ci->replaceAllUsesWith(ConstantSInt::get(Type::IntTy,1));
        break;
      }
      default:
        return false;
    }
    ci->eraseFromParent();
    return true;
  }
} FPrintFOptimizer;

/// This LibCallOptimization will simplify calls to the "sprintf" library
/// function. It looks for cases where the result of sprintf is not used and the
/// operation can be reduced to something simpler.
/// @brief Simplify the sprintf library function.
struct SPrintFOptimization : public LibCallOptimization {
public:
  /// @brief Default Constructor
  SPrintFOptimization() : LibCallOptimization("sprintf",
      "Number of 'sprintf' calls simplified") {}

  /// @brief Make sure that the "fprintf" function has the right prototype
  virtual bool ValidateCalledFunction(const Function *f, SimplifyLibCalls &SLC){
    // Just make sure this has at least 2 arguments
    return (f->getReturnType() == Type::IntTy && f->arg_size() >= 2);
  }

  /// @brief Perform the sprintf optimization.
  virtual bool OptimizeCall(CallInst *ci, SimplifyLibCalls &SLC) {
    // If the call has more than 3 operands, we can't optimize it
    if (ci->getNumOperands() > 4 || ci->getNumOperands() < 3)
      return false;

    // All the optimizations depend on the length of the second argument and the
    // fact that it is a constant string array. Check that now
    uint64_t len = 0;
    ConstantArray* CA = 0;
    if (!getConstantStringLength(ci->getOperand(2), len, &CA))
      return false;

    if (ci->getNumOperands() == 3) {
      if (len == 0) {
        // If the length is 0, we just need to store a null byte
        new StoreInst(ConstantInt::get(Type::SByteTy,0),ci->getOperand(1),ci);
        ci->replaceAllUsesWith(ConstantSInt::get(Type::IntTy,0));
        ci->eraseFromParent();
        return true;
      }

      // Make sure there's no % in the constant array
      for (unsigned i = 0; i < len; ++i) {
        if (ConstantInt* CI = dyn_cast<ConstantInt>(CA->getOperand(i))) {
          // Check for the null terminator
          if (CI->getRawValue() == '%')
            return false; // we found a %, can't optimize
        } else {
          return false; // initializer is not constant int, can't optimize
        }
      }

      // Increment length because we want to copy the null byte too
      len++;

      // sprintf(str,fmt) -> llvm.memcpy(str,fmt,strlen(fmt),1)
      Function* memcpy_func = SLC.get_memcpy();
      if (!memcpy_func)
        return false;
      std::vector<Value*> args;
      args.push_back(ci->getOperand(1));
      args.push_back(ci->getOperand(2));
      args.push_back(ConstantUInt::get(SLC.getIntPtrType(),len));
      args.push_back(ConstantUInt::get(Type::UIntTy,1));
      new CallInst(memcpy_func,args,"",ci);
      ci->replaceAllUsesWith(ConstantSInt::get(Type::IntTy,len));
      ci->eraseFromParent();
      return true;
    }

    // The remaining optimizations require the format string to be length 2
    // "%s" or "%c".
    if (len != 2)
      return false;

    // The first character has to be a %
    if (ConstantInt* CI = dyn_cast<ConstantInt>(CA->getOperand(0)))
      if (CI->getRawValue() != '%')
        return false;

    // Get the second character and switch on its value
    ConstantInt* CI = dyn_cast<ConstantInt>(CA->getOperand(1));
    switch (CI->getRawValue()) {
    case 's': {
      // sprintf(dest,"%s",str) -> llvm.memcpy(dest, str, strlen(str)+1, 1)
      Function* strlen_func = SLC.get_strlen();
      Function* memcpy_func = SLC.get_memcpy();
      if (!strlen_func || !memcpy_func)
        return false;
      
      Value *Len = new CallInst(strlen_func, CastToCStr(ci->getOperand(3), *ci),
                                ci->getOperand(3)->getName()+".len", ci);
      Value *Len1 = BinaryOperator::createAdd(Len,
                                            ConstantInt::get(Len->getType(), 1),
                                              Len->getName()+"1", ci);
      if (Len1->getType() != SLC.getIntPtrType())
        Len1 = new CastInst(Len1, SLC.getIntPtrType(), Len1->getName(), ci);
      std::vector<Value*> args;
      args.push_back(CastToCStr(ci->getOperand(1), *ci));
      args.push_back(CastToCStr(ci->getOperand(3), *ci));
      args.push_back(Len1);
      args.push_back(ConstantUInt::get(Type::UIntTy,1));
      new CallInst(memcpy_func, args, "", ci);
      
      // The strlen result is the unincremented number of bytes in the string.
      if (!ci->use_empty()) {
        if (Len->getType() != ci->getType())
          Len = new CastInst(Len, ci->getType(), Len->getName(), ci);
        ci->replaceAllUsesWith(Len);
      }
      ci->eraseFromParent();
      return true;
    }
    case 'c': {
      // sprintf(dest,"%c",chr) -> store chr, dest
      CastInst* cast = new CastInst(ci->getOperand(3),Type::SByteTy,"char",ci);
      new StoreInst(cast, ci->getOperand(1), ci);
      GetElementPtrInst* gep = new GetElementPtrInst(ci->getOperand(1),
        ConstantUInt::get(Type::UIntTy,1),ci->getOperand(1)->getName()+".end",
        ci);
      new StoreInst(ConstantInt::get(Type::SByteTy,0),gep,ci);
      ci->replaceAllUsesWith(ConstantSInt::get(Type::IntTy,1));
      ci->eraseFromParent();
      return true;
    }
    }
    return false;
  }
} SPrintFOptimizer;

/// This LibCallOptimization will simplify calls to the "fputs" library
/// function. It looks for cases where the result of fputs is not used and the
/// operation can be reduced to something simpler.
/// @brief Simplify the puts library function.
struct PutsOptimization : public LibCallOptimization {
public:
  /// @brief Default Constructor
  PutsOptimization() : LibCallOptimization("fputs",
      "Number of 'fputs' calls simplified") {}

  /// @brief Make sure that the "fputs" function has the right prototype
  virtual bool ValidateCalledFunction(const Function *F, SimplifyLibCalls &SLC){
    // Just make sure this has 2 arguments
    return F->arg_size() == 2;
  }

  /// @brief Perform the fputs optimization.
  virtual bool OptimizeCall(CallInst* ci, SimplifyLibCalls& SLC) {
    // If the result is used, none of these optimizations work
    if (!ci->use_empty())
      return false;

    // All the optimizations depend on the length of the first argument and the
    // fact that it is a constant string array. Check that now
    uint64_t len = 0;
    if (!getConstantStringLength(ci->getOperand(1), len))
      return false;

    switch (len) {
      case 0:
        // fputs("",F) -> noop
        break;
      case 1:
      {
        // fputs(s,F)  -> fputc(s[0],F)  (if s is constant and strlen(s) == 1)
        const Type* FILEptr_type = ci->getOperand(2)->getType();
        Function* fputc_func = SLC.get_fputc(FILEptr_type);
        if (!fputc_func)
          return false;
        LoadInst* loadi = new LoadInst(ci->getOperand(1),
          ci->getOperand(1)->getName()+".byte",ci);
        CastInst* casti = new CastInst(loadi,Type::IntTy,
          loadi->getName()+".int",ci);
        new CallInst(fputc_func,casti,ci->getOperand(2),"",ci);
        break;
      }
      default:
      {
        // fputs(s,F)  -> fwrite(s,1,len,F) (if s is constant and strlen(s) > 1)
        const Type* FILEptr_type = ci->getOperand(2)->getType();
        Function* fwrite_func = SLC.get_fwrite(FILEptr_type);
        if (!fwrite_func)
          return false;
        std::vector<Value*> parms;
        parms.push_back(ci->getOperand(1));
        parms.push_back(ConstantUInt::get(SLC.getIntPtrType(),len));
        parms.push_back(ConstantUInt::get(SLC.getIntPtrType(),1));
        parms.push_back(ci->getOperand(2));
        new CallInst(fwrite_func,parms,"",ci);
        break;
      }
    }
    ci->eraseFromParent();
    return true; // success
  }
} PutsOptimizer;

/// This LibCallOptimization will simplify calls to the "isdigit" library
/// function. It simply does range checks the parameter explicitly.
/// @brief Simplify the isdigit library function.
struct isdigitOptimization : public LibCallOptimization {
public:
  isdigitOptimization() : LibCallOptimization("isdigit",
      "Number of 'isdigit' calls simplified") {}

  /// @brief Make sure that the "isdigit" function has the right prototype
  virtual bool ValidateCalledFunction(const Function* f, SimplifyLibCalls& SLC){
    // Just make sure this has 1 argument
    return (f->arg_size() == 1);
  }

  /// @brief Perform the toascii optimization.
  virtual bool OptimizeCall(CallInst *ci, SimplifyLibCalls &SLC) {
    if (ConstantInt* CI = dyn_cast<ConstantInt>(ci->getOperand(1))) {
      // isdigit(c)   -> 0 or 1, if 'c' is constant
      uint64_t val = CI->getRawValue();
      if (val >= '0' && val <='9')
        ci->replaceAllUsesWith(ConstantSInt::get(Type::IntTy,1));
      else
        ci->replaceAllUsesWith(ConstantSInt::get(Type::IntTy,0));
      ci->eraseFromParent();
      return true;
    }

    // isdigit(c)   -> (unsigned)c - '0' <= 9
    CastInst* cast =
      new CastInst(ci->getOperand(1),Type::UIntTy,
        ci->getOperand(1)->getName()+".uint",ci);
    BinaryOperator* sub_inst = BinaryOperator::createSub(cast,
        ConstantUInt::get(Type::UIntTy,0x30),
        ci->getOperand(1)->getName()+".sub",ci);
    SetCondInst* setcond_inst = new SetCondInst(Instruction::SetLE,sub_inst,
        ConstantUInt::get(Type::UIntTy,9),
        ci->getOperand(1)->getName()+".cmp",ci);
    CastInst* c2 =
      new CastInst(setcond_inst,Type::IntTy,
        ci->getOperand(1)->getName()+".isdigit",ci);
    ci->replaceAllUsesWith(c2);
    ci->eraseFromParent();
    return true;
  }
} isdigitOptimizer;

struct isasciiOptimization : public LibCallOptimization {
public:
  isasciiOptimization()
    : LibCallOptimization("isascii", "Number of 'isascii' calls simplified") {}
  
  virtual bool ValidateCalledFunction(const Function *F, SimplifyLibCalls &SLC){
    return F->arg_size() == 1 && F->arg_begin()->getType()->isInteger() && 
           F->getReturnType()->isInteger();
  }
  
  /// @brief Perform the isascii optimization.
  virtual bool OptimizeCall(CallInst *CI, SimplifyLibCalls &SLC) {
    // isascii(c)   -> (unsigned)c < 128
    Value *V = CI->getOperand(1);
    if (V->getType()->isSigned())
      V = new CastInst(V, V->getType()->getUnsignedVersion(), V->getName(), CI);
    Value *Cmp = BinaryOperator::createSetLT(V, ConstantUInt::get(V->getType(),
                                                                  128),
                                             V->getName()+".isascii", CI);
    if (Cmp->getType() != CI->getType())
      Cmp = new CastInst(Cmp, CI->getType(), Cmp->getName(), CI);
    CI->replaceAllUsesWith(Cmp);
    CI->eraseFromParent();
    return true;
  }
} isasciiOptimizer;


/// This LibCallOptimization will simplify calls to the "toascii" library
/// function. It simply does the corresponding and operation to restrict the
/// range of values to the ASCII character set (0-127).
/// @brief Simplify the toascii library function.
struct ToAsciiOptimization : public LibCallOptimization {
public:
  /// @brief Default Constructor
  ToAsciiOptimization() : LibCallOptimization("toascii",
      "Number of 'toascii' calls simplified") {}

  /// @brief Make sure that the "fputs" function has the right prototype
  virtual bool ValidateCalledFunction(const Function* f, SimplifyLibCalls& SLC){
    // Just make sure this has 2 arguments
    return (f->arg_size() == 1);
  }

  /// @brief Perform the toascii optimization.
  virtual bool OptimizeCall(CallInst *ci, SimplifyLibCalls &SLC) {
    // toascii(c)   -> (c & 0x7f)
    Value* chr = ci->getOperand(1);
    BinaryOperator* and_inst = BinaryOperator::createAnd(chr,
        ConstantInt::get(chr->getType(),0x7F),ci->getName()+".toascii",ci);
    ci->replaceAllUsesWith(and_inst);
    ci->eraseFromParent();
    return true;
  }
} ToAsciiOptimizer;

/// This LibCallOptimization will simplify calls to the "ffs" library
/// calls which find the first set bit in an int, long, or long long. The
/// optimization is to compute the result at compile time if the argument is
/// a constant.
/// @brief Simplify the ffs library function.
struct FFSOptimization : public LibCallOptimization {
protected:
  /// @brief Subclass Constructor
  FFSOptimization(const char* funcName, const char* description)
    : LibCallOptimization(funcName, description) {}

public:
  /// @brief Default Constructor
  FFSOptimization() : LibCallOptimization("ffs",
      "Number of 'ffs' calls simplified") {}

  /// @brief Make sure that the "ffs" function has the right prototype
  virtual bool ValidateCalledFunction(const Function *F, SimplifyLibCalls &SLC){
    // Just make sure this has 2 arguments
    return F->arg_size() == 1 && F->getReturnType() == Type::IntTy;
  }

  /// @brief Perform the ffs optimization.
  virtual bool OptimizeCall(CallInst *TheCall, SimplifyLibCalls &SLC) {
    if (ConstantInt *CI = dyn_cast<ConstantInt>(TheCall->getOperand(1))) {
      // ffs(cnst)  -> bit#
      // ffsl(cnst) -> bit#
      // ffsll(cnst) -> bit#
      uint64_t val = CI->getRawValue();
      int result = 0;
      if (val) {
        ++result;
        while ((val & 1) == 0) {
          ++result;
          val >>= 1;
        }
      }
      TheCall->replaceAllUsesWith(ConstantSInt::get(Type::IntTy, result));
      TheCall->eraseFromParent();
      return true;
    }

    // ffs(x)   -> x == 0 ? 0 : llvm.cttz(x)+1
    // ffsl(x)  -> x == 0 ? 0 : llvm.cttz(x)+1
    // ffsll(x) -> x == 0 ? 0 : llvm.cttz(x)+1
    const Type *ArgType = TheCall->getOperand(1)->getType();
    ArgType = ArgType->getUnsignedVersion();
    const char *CTTZName;
    switch (ArgType->getTypeID()) {
    default: assert(0 && "Unknown unsigned type!");
    case Type::UByteTyID : CTTZName = "llvm.cttz.i8" ; break;
    case Type::UShortTyID: CTTZName = "llvm.cttz.i16"; break;
    case Type::UIntTyID  : CTTZName = "llvm.cttz.i32"; break;
    case Type::ULongTyID : CTTZName = "llvm.cttz.i64"; break;
    }
    
    Function *F = SLC.getModule()->getOrInsertFunction(CTTZName, ArgType,
                                                       ArgType, NULL);
    Value *V = new CastInst(TheCall->getOperand(1), ArgType, "tmp", TheCall);
    Value *V2 = new CallInst(F, V, "tmp", TheCall);
    V2 = new CastInst(V2, Type::IntTy, "tmp", TheCall);
    V2 = BinaryOperator::createAdd(V2, ConstantSInt::get(Type::IntTy, 1),
                                   "tmp", TheCall);
    Value *Cond = 
      BinaryOperator::createSetEQ(V, Constant::getNullValue(V->getType()),
                                  "tmp", TheCall);
    V2 = new SelectInst(Cond, ConstantInt::get(Type::IntTy, 0), V2,
                        TheCall->getName(), TheCall);
    TheCall->replaceAllUsesWith(V2);
    TheCall->eraseFromParent();
    return true;
  }
} FFSOptimizer;

/// This LibCallOptimization will simplify calls to the "ffsl" library
/// calls. It simply uses FFSOptimization for which the transformation is
/// identical.
/// @brief Simplify the ffsl library function.
struct FFSLOptimization : public FFSOptimization {
public:
  /// @brief Default Constructor
  FFSLOptimization() : FFSOptimization("ffsl",
      "Number of 'ffsl' calls simplified") {}

} FFSLOptimizer;

/// This LibCallOptimization will simplify calls to the "ffsll" library
/// calls. It simply uses FFSOptimization for which the transformation is
/// identical.
/// @brief Simplify the ffsl library function.
struct FFSLLOptimization : public FFSOptimization {
public:
  /// @brief Default Constructor
  FFSLLOptimization() : FFSOptimization("ffsll",
      "Number of 'ffsll' calls simplified") {}

} FFSLLOptimizer;

/// This optimizes unary functions that take and return doubles.
struct UnaryDoubleFPOptimizer : public LibCallOptimization {
  UnaryDoubleFPOptimizer(const char *Fn, const char *Desc)
  : LibCallOptimization(Fn, Desc) {}
  
  // Make sure that this function has the right prototype
  virtual bool ValidateCalledFunction(const Function *F, SimplifyLibCalls &SLC){
    return F->arg_size() == 1 && F->arg_begin()->getType() == Type::DoubleTy &&
           F->getReturnType() == Type::DoubleTy;
  }

  /// ShrinkFunctionToFloatVersion - If the input to this function is really a
  /// float, strength reduce this to a float version of the function,
  /// e.g. floor((double)FLT) -> (double)floorf(FLT).  This can only be called
  /// when the target supports the destination function and where there can be
  /// no precision loss.
  static bool ShrinkFunctionToFloatVersion(CallInst *CI, SimplifyLibCalls &SLC,
                                           Function *(SimplifyLibCalls::*FP)()){
    if (CastInst *Cast = dyn_cast<CastInst>(CI->getOperand(1)))
      if (Cast->getOperand(0)->getType() == Type::FloatTy) {
        Value *New = new CallInst((SLC.*FP)(), Cast->getOperand(0),
                                  CI->getName(), CI);
        New = new CastInst(New, Type::DoubleTy, CI->getName(), CI);
        CI->replaceAllUsesWith(New);
        CI->eraseFromParent();
        if (Cast->use_empty())
          Cast->eraseFromParent();
        return true;
      }
    return false;
  }
};


struct FloorOptimization : public UnaryDoubleFPOptimizer {
  FloorOptimization()
    : UnaryDoubleFPOptimizer("floor", "Number of 'floor' calls simplified") {}
  
  virtual bool OptimizeCall(CallInst *CI, SimplifyLibCalls &SLC) {
#ifdef HAVE_FLOORF
    // If this is a float argument passed in, convert to floorf.
    if (ShrinkFunctionToFloatVersion(CI, SLC, &SimplifyLibCalls::get_floorf))
      return true;
#endif
    return false; // opt failed
  }
} FloorOptimizer;

struct CeilOptimization : public UnaryDoubleFPOptimizer {
  CeilOptimization()
  : UnaryDoubleFPOptimizer("ceil", "Number of 'ceil' calls simplified") {}
  
  virtual bool OptimizeCall(CallInst *CI, SimplifyLibCalls &SLC) {
#ifdef HAVE_CEILF
    // If this is a float argument passed in, convert to ceilf.
    if (ShrinkFunctionToFloatVersion(CI, SLC, &SimplifyLibCalls::get_ceilf))
      return true;
#endif
    return false; // opt failed
  }
} CeilOptimizer;

struct RoundOptimization : public UnaryDoubleFPOptimizer {
  RoundOptimization()
  : UnaryDoubleFPOptimizer("round", "Number of 'round' calls simplified") {}
  
  virtual bool OptimizeCall(CallInst *CI, SimplifyLibCalls &SLC) {
#ifdef HAVE_ROUNDF
    // If this is a float argument passed in, convert to roundf.
    if (ShrinkFunctionToFloatVersion(CI, SLC, &SimplifyLibCalls::get_roundf))
      return true;
#endif
    return false; // opt failed
  }
} RoundOptimizer;

struct RintOptimization : public UnaryDoubleFPOptimizer {
  RintOptimization()
  : UnaryDoubleFPOptimizer("rint", "Number of 'rint' calls simplified") {}
  
  virtual bool OptimizeCall(CallInst *CI, SimplifyLibCalls &SLC) {
#ifdef HAVE_RINTF
    // If this is a float argument passed in, convert to rintf.
    if (ShrinkFunctionToFloatVersion(CI, SLC, &SimplifyLibCalls::get_rintf))
      return true;
#endif
    return false; // opt failed
  }
} RintOptimizer;

struct NearByIntOptimization : public UnaryDoubleFPOptimizer {
  NearByIntOptimization()
  : UnaryDoubleFPOptimizer("nearbyint",
                           "Number of 'nearbyint' calls simplified") {}
  
  virtual bool OptimizeCall(CallInst *CI, SimplifyLibCalls &SLC) {
#ifdef HAVE_NEARBYINTF
    // If this is a float argument passed in, convert to nearbyintf.
    if (ShrinkFunctionToFloatVersion(CI, SLC,&SimplifyLibCalls::get_nearbyintf))
      return true;
#endif
    return false; // opt failed
  }
} NearByIntOptimizer;

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
bool getConstantStringLength(Value *V, uint64_t &len, ConstantArray **CA) {
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
  if (ConstantInt* op1 = dyn_cast<ConstantInt>(GEP->getOperand(1))) {
    if (!op1->isNullValue())
      return false;
  } else
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
  if (ConstantAggregateZero *CAZ = dyn_cast<ConstantAggregateZero>(INTLZR)) {
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
  for (len = start_idx; len < max_elems; len++) {
    if (ConstantInt *CI = dyn_cast<ConstantInt>(A->getOperand(len))) {
      // Check for the null terminator
      if (CI->isNullValue())
        break; // we found end of string
    } else
      return false; // This array isn't suitable, non-int initializer
  }
  
  if (len >= max_elems)
    return false; // This array isn't null terminated

  // Subtract out the initial value from the length
  len -= start_idx;
  if (CA)
    *CA = A;
  return true; // success!
}

/// CastToCStr - Return V if it is an sbyte*, otherwise cast it to sbyte*,
/// inserting the cast before IP, and return the cast.
/// @brief Cast a value to a "C" string.
Value *CastToCStr(Value *V, Instruction &IP) {
  const Type *SBPTy = PointerType::get(Type::SByteTy);
  if (V->getType() != SBPTy)
    return new CastInst(V, SBPTy, V->getName(), &IP);
  return V;
}

// TODO:
//   Additional cases that we need to add to this file:
//
// cbrt:
//   * cbrt(expN(X))  -> expN(x/3)
//   * cbrt(sqrt(x))  -> pow(x,1/6)
//   * cbrt(sqrt(x))  -> pow(x,1/9)
//
// cos, cosf, cosl:
//   * cos(-x)  -> cos(x)
//
// exp, expf, expl:
//   * exp(log(x))  -> x
//
// log, logf, logl:
//   * log(exp(x))   -> x
//   * log(x**y)     -> y*log(x)
//   * log(exp(y))   -> y*log(e)
//   * log(exp2(y))  -> y*log(2)
//   * log(exp10(y)) -> y*log(10)
//   * log(sqrt(x))  -> 0.5*log(x)
//   * log(pow(x,y)) -> y*log(x)
//
// lround, lroundf, lroundl:
//   * lround(cnst) -> cnst'
//
// memcmp:
//   * memcmp(x,y,l)   -> cnst
//      (if all arguments are constant and strlen(x) <= l and strlen(y) <= l)
//
// memmove:
//   * memmove(d,s,l,a) -> memcpy(d,s,l,a)
//       (if s is a global constant array)
//
// pow, powf, powl:
//   * pow(exp(x),y)  -> exp(x*y)
//   * pow(sqrt(x),y) -> pow(x,y*0.5)
//   * pow(pow(x,y),z)-> pow(x,y*z)
//
// puts:
//   * puts("") -> fputc("\n",stdout) (how do we get "stdout"?)
//
// round, roundf, roundl:
//   * round(cnst) -> cnst'
//
// signbit:
//   * signbit(cnst) -> cnst'
//   * signbit(nncst) -> 0 (if pstv is a non-negative constant)
//
// sqrt, sqrtf, sqrtl:
//   * sqrt(expN(x))  -> expN(x*0.5)
//   * sqrt(Nroot(x)) -> pow(x,1/(2*N))
//   * sqrt(pow(x,y)) -> pow(|x|,y*0.5)
//
// stpcpy:
//   * stpcpy(str, "literal") ->
//           llvm.memcpy(str,"literal",strlen("literal")+1,1)
// strrchr:
//   * strrchr(s,c) -> reverse_offset_of_in(c,s)
//      (if c is a constant integer and s is a constant string)
//   * strrchr(s1,0) -> strchr(s1,0)
//
// strncat:
//   * strncat(x,y,0) -> x
//   * strncat(x,y,0) -> x (if strlen(y) = 0)
//   * strncat(x,y,l) -> strcat(x,y) (if y and l are constants an l > strlen(y))
//
// strncpy:
//   * strncpy(d,s,0) -> d
//   * strncpy(d,s,l) -> memcpy(d,s,l,1)
//      (if s and l are constants)
//
// strpbrk:
//   * strpbrk(s,a) -> offset_in_for(s,a)
//      (if s and a are both constant strings)
//   * strpbrk(s,"") -> 0
//   * strpbrk(s,a) -> strchr(s,a[0]) (if a is constant string of length 1)
//
// strspn, strcspn:
//   * strspn(s,a)   -> const_int (if both args are constant)
//   * strspn("",a)  -> 0
//   * strspn(s,"")  -> 0
//   * strcspn(s,a)  -> const_int (if both args are constant)
//   * strcspn("",a) -> 0
//   * strcspn(s,"") -> strlen(a)
//
// strstr:
//   * strstr(x,x)  -> x
//   * strstr(s1,s2) -> offset_of_s2_in(s1)
//       (if s1 and s2 are constant strings)
//
// tan, tanf, tanl:
//   * tan(atan(x)) -> x
//
// trunc, truncf, truncl:
//   * trunc(cnst) -> cnst'
//
//
}
