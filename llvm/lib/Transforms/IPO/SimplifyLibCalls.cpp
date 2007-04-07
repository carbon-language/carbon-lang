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
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Transforms/IPO.h"
using namespace llvm;

/// This statistic keeps track of the total number of library calls that have
/// been simplified regardless of which call it is.
STATISTIC(SimplifiedLibCalls, "Number of library calls simplified");

namespace {
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
class VISIBILITY_HIDDEN LibCallOptimization {
  LibCallOptimization **Prev, *Next;
  const char *FunctionName; ///< Name of the library call we optimize
#ifndef NDEBUG
  Statistic occurrences; ///< debug statistic (-debug-only=simplify-libcalls)
#endif
public:
  /// The \p fname argument must be the name of the library function being
  /// optimized by the subclass.
  /// @brief Constructor that registers the optimization.
  LibCallOptimization(const char *FName, const char *Description)
    : FunctionName(FName) {
      
#ifndef NDEBUG
    occurrences.construct("simplify-libcalls", Description);
#endif
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

  bool ReplaceCallWith(CallInst *CI, Value *V) {
    if (!CI->use_empty())
      CI->replaceAllUsesWith(V);
    CI->eraseFromParent();
    return true;
  }
  
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
class VISIBILITY_HIDDEN SimplifyLibCalls : public ModulePass {
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
        // have external or dllimport linkage and non-empty uses.
        if (!FI->isDeclaration() ||
            !(FI->hasExternalLinkage() || FI->hasDLLImportLinkage()) ||
            FI->use_empty())
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
  Constant *get_putchar() {
    if (!putchar_func)
      putchar_func = 
        M->getOrInsertFunction("putchar", Type::Int32Ty, Type::Int32Ty, NULL);
    return putchar_func;
  }

  /// @brief Return a Function* for the puts libcall
  Constant *get_puts() {
    if (!puts_func)
      puts_func = M->getOrInsertFunction("puts", Type::Int32Ty,
                                         PointerType::get(Type::Int8Ty),
                                         NULL);
    return puts_func;
  }

  /// @brief Return a Function* for the fputc libcall
  Constant *get_fputc(const Type* FILEptr_type) {
    if (!fputc_func)
      fputc_func = M->getOrInsertFunction("fputc", Type::Int32Ty, Type::Int32Ty,
                                          FILEptr_type, NULL);
    return fputc_func;
  }

  /// @brief Return a Function* for the fputs libcall
  Constant *get_fputs(const Type* FILEptr_type) {
    if (!fputs_func)
      fputs_func = M->getOrInsertFunction("fputs", Type::Int32Ty,
                                          PointerType::get(Type::Int8Ty),
                                          FILEptr_type, NULL);
    return fputs_func;
  }

  /// @brief Return a Function* for the fwrite libcall
  Constant *get_fwrite(const Type* FILEptr_type) {
    if (!fwrite_func)
      fwrite_func = M->getOrInsertFunction("fwrite", TD->getIntPtrType(),
                                           PointerType::get(Type::Int8Ty),
                                           TD->getIntPtrType(),
                                           TD->getIntPtrType(),
                                           FILEptr_type, NULL);
    return fwrite_func;
  }

  /// @brief Return a Function* for the sqrt libcall
  Constant *get_sqrt() {
    if (!sqrt_func)
      sqrt_func = M->getOrInsertFunction("sqrt", Type::DoubleTy, 
                                         Type::DoubleTy, NULL);
    return sqrt_func;
  }

  /// @brief Return a Function* for the strcpy libcall
  Constant *get_strcpy() {
    if (!strcpy_func)
      strcpy_func = M->getOrInsertFunction("strcpy",
                                           PointerType::get(Type::Int8Ty),
                                           PointerType::get(Type::Int8Ty),
                                           PointerType::get(Type::Int8Ty),
                                           NULL);
    return strcpy_func;
  }

  /// @brief Return a Function* for the strlen libcall
  Constant *get_strlen() {
    if (!strlen_func)
      strlen_func = M->getOrInsertFunction("strlen", TD->getIntPtrType(),
                                           PointerType::get(Type::Int8Ty),
                                           NULL);
    return strlen_func;
  }

  /// @brief Return a Function* for the memchr libcall
  Constant *get_memchr() {
    if (!memchr_func)
      memchr_func = M->getOrInsertFunction("memchr",
                                           PointerType::get(Type::Int8Ty),
                                           PointerType::get(Type::Int8Ty),
                                           Type::Int32Ty, TD->getIntPtrType(),
                                           NULL);
    return memchr_func;
  }

  /// @brief Return a Function* for the memcpy libcall
  Constant *get_memcpy() {
    if (!memcpy_func) {
      const Type *SBP = PointerType::get(Type::Int8Ty);
      const char *N = TD->getIntPtrType() == Type::Int32Ty ?
                            "llvm.memcpy.i32" : "llvm.memcpy.i64";
      memcpy_func = M->getOrInsertFunction(N, Type::VoidTy, SBP, SBP,
                                           TD->getIntPtrType(), Type::Int32Ty,
                                           NULL);
    }
    return memcpy_func;
  }

  Constant *getUnaryFloatFunction(const char *Name, Constant *&Cache) {
    if (!Cache)
      Cache = M->getOrInsertFunction(Name, Type::FloatTy, Type::FloatTy, NULL);
    return Cache;
  }
  
  Constant *get_floorf() { return getUnaryFloatFunction("floorf", floorf_func);}
  Constant *get_ceilf()  { return getUnaryFloatFunction( "ceilf",  ceilf_func);}
  Constant *get_roundf() { return getUnaryFloatFunction("roundf", roundf_func);}
  Constant *get_rintf()  { return getUnaryFloatFunction( "rintf",  rintf_func);}
  Constant *get_nearbyintf() { return getUnaryFloatFunction("nearbyintf",
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
  Constant *putchar_func, *puts_func;
  Constant *fputc_func, *fputs_func, *fwrite_func;
  Constant *memcpy_func, *memchr_func;
  Constant *sqrt_func;
  Constant *strcpy_func, *strlen_func;
  Constant *floorf_func, *ceilf_func, *roundf_func;
  Constant *rintf_func, *nearbyintf_func;
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
static bool GetConstantStringInfo(Value *V, ConstantArray *&Array,
                                  uint64_t &Length, uint64_t &StartIdx);
static Value *CastToCStr(Value *V, Instruction *IP);

/// This LibCallOptimization will find instances of a call to "exit" that occurs
/// within the "main" function and change it to a simple "ret" instruction with
/// the same value passed to the exit function. When this is done, it splits the
/// basic block at the exit(3) call and deletes the call instruction.
/// @brief Replace calls to exit in main with a simple return
struct VISIBILITY_HIDDEN ExitInMainOptimization : public LibCallOptimization {
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
struct VISIBILITY_HIDDEN StrCatOptimization : public LibCallOptimization {
public:
  /// @brief Default constructor
  StrCatOptimization() : LibCallOptimization("strcat",
      "Number of 'strcat' calls simplified") {}

public:

  /// @brief Make sure that the "strcat" function has the right prototype
  virtual bool ValidateCalledFunction(const Function* f, SimplifyLibCalls& SLC){
    if (f->getReturnType() == PointerType::get(Type::Int8Ty))
      if (f->arg_size() == 2)
      {
        Function::const_arg_iterator AI = f->arg_begin();
        if (AI++->getType() == PointerType::get(Type::Int8Ty))
          if (AI->getType() == PointerType::get(Type::Int8Ty))
          {
            // Indicate this is a suitable call type.
            return true;
          }
      }
    return false;
  }

  /// @brief Optimize the strcat library function
  virtual bool OptimizeCall(CallInst *CI, SimplifyLibCalls &SLC) {
    // Extract some information from the instruction
    Value *Dst = CI->getOperand(1);
    Value *Src = CI->getOperand(2);

    // Extract the initializer (while making numerous checks) from the
    // source operand of the call to strcat.
    uint64_t SrcLength, StartIdx;
    ConstantArray *Arr;
    if (!GetConstantStringInfo(Src, Arr, SrcLength, StartIdx))
      return false;

    // Handle the simple, do-nothing case
    if (SrcLength == 0)
      return ReplaceCallWith(CI, Dst);

    // We need to find the end of the destination string.  That's where the
    // memory is to be moved to. We just generate a call to strlen (further
    // optimized in another pass).
    CallInst *DstLen = new CallInst(SLC.get_strlen(), Dst,
                                    Dst->getName()+".len", CI);

    // Now that we have the destination's length, we must index into the
    // destination's pointer to get the actual memcpy destination (end of
    // the string .. we're concatenating).
    Dst = new GetElementPtrInst(Dst, DstLen, Dst->getName()+".indexed", CI);

    // We have enough information to now generate the memcpy call to
    // do the concatenation for us.
    Value *Vals[] = {
      Dst, Src,
      ConstantInt::get(SLC.getIntPtrType(), SrcLength+1), // copy nul term.
      ConstantInt::get(Type::Int32Ty, 1)  // alignment
    };
    new CallInst(SLC.get_memcpy(), Vals, 4, "", CI);

    return ReplaceCallWith(CI, Dst);
  }
} StrCatOptimizer;

/// This LibCallOptimization will simplify a call to the strchr library
/// function.  It optimizes out cases where the arguments are both constant
/// and the result can be determined statically.
/// @brief Simplify the strcmp library function.
struct VISIBILITY_HIDDEN StrChrOptimization : public LibCallOptimization {
public:
  StrChrOptimization() : LibCallOptimization("strchr",
      "Number of 'strchr' calls simplified") {}

  /// @brief Make sure that the "strchr" function has the right prototype
  virtual bool ValidateCalledFunction(const Function *F, SimplifyLibCalls &SLC){
    const FunctionType *FT = F->getFunctionType();
    return FT->getNumParams() == 2 &&
           FT->getReturnType() == PointerType::get(Type::Int8Ty) &&
           FT->getParamType(0) == FT->getReturnType() &&
           isa<IntegerType>(FT->getParamType(1));
  }

  /// @brief Perform the strchr optimizations
  virtual bool OptimizeCall(CallInst *CI, SimplifyLibCalls &SLC) {
    // Check that the first argument to strchr is a constant array of sbyte.
    // If it is, get the length and data, otherwise return false.
    uint64_t StrLength, StartIdx;
    ConstantArray *CA = 0;
    if (!GetConstantStringInfo(CI->getOperand(1), CA, StrLength, StartIdx))
      return false;

    // If the second operand is not constant, just lower this to memchr since we
    // know the length of the input string.
    ConstantInt *CSI = dyn_cast<ConstantInt>(CI->getOperand(2));
    if (!CSI) {
      Value *Args[3] = {
        CI->getOperand(1),
        CI->getOperand(2),
        ConstantInt::get(SLC.getIntPtrType(), StrLength+1)
      };
      return ReplaceCallWith(CI, new CallInst(SLC.get_memchr(), Args, 3,
                                              CI->getName(), CI));
    }

    // Get the character we're looking for
    int64_t CharValue = CSI->getSExtValue();

    if (StrLength == 0) {
      // If the length of the string is zero, and we are searching for zero,
      // return the input pointer.
      if (CharValue == 0)
        return ReplaceCallWith(CI, CI->getOperand(1));
      // Otherwise, char wasn't found.
      return ReplaceCallWith(CI, Constant::getNullValue(CI->getType()));
    }
    
    // Compute the offset
    uint64_t i = 0;
    while (1) {
      assert(i <= StrLength && "Didn't find null terminator?");
      if (ConstantInt *C = dyn_cast<ConstantInt>(CA->getOperand(i+StartIdx))) {
        // Did we find our match?
        if (C->getSExtValue() == CharValue)
          break;
        if (C->isZero()) // We found the end of the string. strchr returns null.
          return ReplaceCallWith(CI, Constant::getNullValue(CI->getType()));
      }
      ++i;
    }

    // strchr(s+n,c)  -> gep(s+n+i,c)
    //    (if c is a constant integer and s is a constant string)
    Value *Idx = ConstantInt::get(Type::Int64Ty, i);
    Value *GEP = new GetElementPtrInst(CI->getOperand(1), Idx, 
                                       CI->getOperand(1)->getName() +
                                       ".strchr", CI);
    return ReplaceCallWith(CI, GEP);
  }
} StrChrOptimizer;

/// This LibCallOptimization will simplify a call to the strcmp library
/// function.  It optimizes out cases where one or both arguments are constant
/// and the result can be determined statically.
/// @brief Simplify the strcmp library function.
struct VISIBILITY_HIDDEN StrCmpOptimization : public LibCallOptimization {
public:
  StrCmpOptimization() : LibCallOptimization("strcmp",
      "Number of 'strcmp' calls simplified") {}

  /// @brief Make sure that the "strcmp" function has the right prototype
  virtual bool ValidateCalledFunction(const Function *F, SimplifyLibCalls &SLC){
    const FunctionType *FT = F->getFunctionType();
    return FT->getReturnType() == Type::Int32Ty && FT->getNumParams() == 2 &&
           FT->getParamType(0) == FT->getParamType(1) &&
           FT->getParamType(0) == PointerType::get(Type::Int8Ty);
  }

  /// @brief Perform the strcmp optimization
  virtual bool OptimizeCall(CallInst *CI, SimplifyLibCalls &SLC) {
    // First, check to see if src and destination are the same. If they are,
    // then the optimization is to replace the CallInst with a constant 0
    // because the call is a no-op.
    Value *Str1P = CI->getOperand(1);
    Value *Str2P = CI->getOperand(2);
    if (Str1P == Str2P)      // strcmp(x,x)  -> 0
      return ReplaceCallWith(CI, ConstantInt::get(CI->getType(), 0));

    uint64_t Str1Len, Str1StartIdx;
    ConstantArray *A1;
    bool Str1IsCst = GetConstantStringInfo(Str1P, A1, Str1Len, Str1StartIdx);
    if (Str1IsCst && Str1Len == 0) {
      // strcmp("", x) -> *x
      Value *V = new LoadInst(Str2P, CI->getName()+".load", CI);
      V = new ZExtInst(V, CI->getType(), CI->getName()+".int", CI);
      return ReplaceCallWith(CI, V);
    }

    uint64_t Str2Len, Str2StartIdx;
    ConstantArray* A2;
    bool Str2IsCst = GetConstantStringInfo(Str2P, A2, Str2Len, Str2StartIdx);
    if (Str2IsCst && Str2Len == 0) {
      // strcmp(x,"") -> *x
      Value *V = new LoadInst(Str1P, CI->getName()+".load", CI);
      V = new ZExtInst(V, CI->getType(), CI->getName()+".int", CI);
      return ReplaceCallWith(CI, V);
    }

    if (Str1IsCst && Str2IsCst && A1->isCString() && A2->isCString()) {
      // strcmp(x, y)  -> cnst  (if both x and y are constant strings)
      std::string S1 = A1->getAsString();
      std::string S2 = A2->getAsString();
      int R = strcmp(S1.c_str()+Str1StartIdx, S2.c_str()+Str2StartIdx);
      return ReplaceCallWith(CI, ConstantInt::get(CI->getType(), R));
    }
    return false;
  }
} StrCmpOptimizer;

/// This LibCallOptimization will simplify a call to the strncmp library
/// function.  It optimizes out cases where one or both arguments are constant
/// and the result can be determined statically.
/// @brief Simplify the strncmp library function.
struct VISIBILITY_HIDDEN StrNCmpOptimization : public LibCallOptimization {
public:
  StrNCmpOptimization() : LibCallOptimization("strncmp",
      "Number of 'strncmp' calls simplified") {}

  /// @brief Make sure that the "strncmp" function has the right prototype
  virtual bool ValidateCalledFunction(const Function *F, SimplifyLibCalls &SLC){
    const FunctionType *FT = F->getFunctionType();
    return FT->getReturnType() == Type::Int32Ty && FT->getNumParams() == 3 &&
           FT->getParamType(0) == FT->getParamType(1) &&
           FT->getParamType(0) == PointerType::get(Type::Int8Ty) &&
           isa<IntegerType>(FT->getParamType(2));
    return false;
  }

  /// @brief Perform the strncmp optimization
  virtual bool OptimizeCall(CallInst *CI, SimplifyLibCalls &SLC) {
    // First, check to see if src and destination are the same. If they are,
    // then the optimization is to replace the CallInst with a constant 0
    // because the call is a no-op.
    Value *Str1P = CI->getOperand(1);
    Value *Str2P = CI->getOperand(2);
    if (Str1P == Str2P)  // strncmp(x,x)  -> 0
      return ReplaceCallWith(CI, ConstantInt::get(CI->getType(), 0));
    
    // Check the length argument, if it is Constant zero then the strings are
    // considered equal.
    uint64_t Length;
    if (ConstantInt *LengthArg = dyn_cast<ConstantInt>(CI->getOperand(3)))
      Length = LengthArg->getZExtValue();
    else
      return false;
    
    if (Length == 0) {
      // strncmp(x,y,0)   -> 0
      return ReplaceCallWith(CI, ConstantInt::get(CI->getType(), 0));
    }
    
    uint64_t Str1Len, Str1StartIdx;
    ConstantArray *A1;
    bool Str1IsCst = GetConstantStringInfo(Str1P, A1, Str1Len, Str1StartIdx);
    if (Str1IsCst && Str1Len == 0) {
      // strncmp("", x) -> *x
      Value *V = new LoadInst(Str2P, CI->getName()+".load", CI);
      V = new ZExtInst(V, CI->getType(), CI->getName()+".int", CI);
      return ReplaceCallWith(CI, V);
    }
    
    uint64_t Str2Len, Str2StartIdx;
    ConstantArray* A2;
    bool Str2IsCst = GetConstantStringInfo(Str2P, A2, Str2Len, Str2StartIdx);
    if (Str2IsCst && Str2Len == 0) {
      // strncmp(x,"") -> *x
      Value *V = new LoadInst(Str1P, CI->getName()+".load", CI);
      V = new ZExtInst(V, CI->getType(), CI->getName()+".int", CI);
      return ReplaceCallWith(CI, V);
    }
    
    if (Str1IsCst && Str2IsCst && A1->isCString() &&
        A2->isCString()) {
      // strncmp(x, y)  -> cnst  (if both x and y are constant strings)
      std::string S1 = A1->getAsString();
      std::string S2 = A2->getAsString();
      int R = strncmp(S1.c_str()+Str1StartIdx, S2.c_str()+Str2StartIdx, Length);
      return ReplaceCallWith(CI, ConstantInt::get(CI->getType(), R));
    }
    return false;
  }
} StrNCmpOptimizer;

/// This LibCallOptimization will simplify a call to the strcpy library
/// function.  Two optimizations are possible:
/// (1) If src and dest are the same and not volatile, just return dest
/// (2) If the src is a constant then we can convert to llvm.memmove
/// @brief Simplify the strcpy library function.
struct VISIBILITY_HIDDEN StrCpyOptimization : public LibCallOptimization {
public:
  StrCpyOptimization() : LibCallOptimization("strcpy",
      "Number of 'strcpy' calls simplified") {}

  /// @brief Make sure that the "strcpy" function has the right prototype
  virtual bool ValidateCalledFunction(const Function *F, SimplifyLibCalls &SLC){
    const FunctionType *FT = F->getFunctionType();
    return FT->getNumParams() == 2 &&
           FT->getParamType(0) == FT->getParamType(1) &&
           FT->getReturnType() == FT->getParamType(0) &&
           FT->getParamType(0) == PointerType::get(Type::Int8Ty);
  }

  /// @brief Perform the strcpy optimization
  virtual bool OptimizeCall(CallInst *CI, SimplifyLibCalls &SLC) {
    // First, check to see if src and destination are the same. If they are,
    // then the optimization is to replace the CallInst with the destination
    // because the call is a no-op. Note that this corresponds to the
    // degenerate strcpy(X,X) case which should have "undefined" results
    // according to the C specification. However, it occurs sometimes and
    // we optimize it as a no-op.
    Value *Dst = CI->getOperand(1);
    Value *Src = CI->getOperand(2);
    if (Dst == Src) {
      // strcpy(x, x) -> x
      return ReplaceCallWith(CI, Dst);
    }
    
    // Get the length of the constant string referenced by the Src operand.
    uint64_t SrcLen, SrcStartIdx;
    ConstantArray *SrcArr;
    if (!GetConstantStringInfo(Src, SrcArr, SrcLen, SrcStartIdx))
      return false;

    // If the constant string's length is zero we can optimize this by just
    // doing a store of 0 at the first byte of the destination
    if (SrcLen == 0) {
      new StoreInst(ConstantInt::get(Type::Int8Ty, 0), Dst, CI);
      return ReplaceCallWith(CI, Dst);
    }

    // We have enough information to now generate the memcpy call to
    // do the concatenation for us.
    Value *MemcpyOps[] = {
      Dst, Src,
      ConstantInt::get(SLC.getIntPtrType(), SrcLen+1), // length including nul.
      ConstantInt::get(Type::Int32Ty, 1) // alignment
    };
    new CallInst(SLC.get_memcpy(), MemcpyOps, 4, "", CI);

    return ReplaceCallWith(CI, Dst);
  }
} StrCpyOptimizer;

/// This LibCallOptimization will simplify a call to the strlen library
/// function by replacing it with a constant value if the string provided to
/// it is a constant array.
/// @brief Simplify the strlen library function.
struct VISIBILITY_HIDDEN StrLenOptimization : public LibCallOptimization {
  StrLenOptimization() : LibCallOptimization("strlen",
      "Number of 'strlen' calls simplified") {}

  /// @brief Make sure that the "strlen" function has the right prototype
  virtual bool ValidateCalledFunction(const Function *F, SimplifyLibCalls &SLC){
    const FunctionType *FT = F->getFunctionType();
    return FT->getNumParams() == 1 &&
           FT->getParamType(0) == PointerType::get(Type::Int8Ty) &&
           isa<IntegerType>(FT->getReturnType());
  }

  /// @brief Perform the strlen optimization
  virtual bool OptimizeCall(CallInst *CI, SimplifyLibCalls &SLC) {
    // Make sure we're dealing with an sbyte* here.
    Value *Str = CI->getOperand(1);

    // Does the call to strlen have exactly one use?
    if (CI->hasOneUse()) {
      // Is that single use a icmp operator?
      if (ICmpInst *Cmp = dyn_cast<ICmpInst>(CI->use_back()))
        // Is it compared against a constant integer?
        if (ConstantInt *Cst = dyn_cast<ConstantInt>(Cmp->getOperand(1))) {
          // If its compared against length 0 with == or !=
          if (Cst->getZExtValue() == 0 && Cmp->isEquality()) {
            // strlen(x) != 0 -> *x != 0
            // strlen(x) == 0 -> *x == 0
            Value *V = new LoadInst(Str, Str->getName()+".first", CI);
            V = new ICmpInst(Cmp->getPredicate(), V, 
                             ConstantInt::get(Type::Int8Ty, 0),
                             Cmp->getName()+".strlen", CI);
            Cmp->replaceAllUsesWith(V);
            Cmp->eraseFromParent();
            return ReplaceCallWith(CI, 0);  // no uses.
          }
        }
    }

    // Get the length of the constant string operand
    uint64_t StrLen = 0, StartIdx;
    ConstantArray *A;
    if (!GetConstantStringInfo(CI->getOperand(1), A, StrLen, StartIdx))
      return false;

    // strlen("xyz") -> 3 (for example)
    return ReplaceCallWith(CI, ConstantInt::get(CI->getType(), StrLen));
  }
} StrLenOptimizer;

/// IsOnlyUsedInEqualsComparison - Return true if it only matters that the value
/// is equal or not-equal to zero. 
static bool IsOnlyUsedInEqualsZeroComparison(Instruction *I) {
  for (Value::use_iterator UI = I->use_begin(), E = I->use_end();
       UI != E; ++UI) {
    if (ICmpInst *IC = dyn_cast<ICmpInst>(*UI))
      if (IC->isEquality())
        if (Constant *C = dyn_cast<Constant>(IC->getOperand(1)))
          if (C->isNullValue())
            continue;
    // Unknown instruction.
    return false;
  }
  return true;
}

/// This memcmpOptimization will simplify a call to the memcmp library
/// function.
struct VISIBILITY_HIDDEN memcmpOptimization : public LibCallOptimization {
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
      return ReplaceCallWith(CI, Constant::getNullValue(CI->getType()));
    }
    
    // Make sure we have a constant length.
    ConstantInt *LenC = dyn_cast<ConstantInt>(CI->getOperand(3));
    if (!LenC) return false;
    uint64_t Len = LenC->getZExtValue();
      
    // If the length is zero, this returns 0.
    switch (Len) {
    case 0:
      // memcmp(s1,s2,0) -> 0
      return ReplaceCallWith(CI, Constant::getNullValue(CI->getType()));
    case 1: {
      // memcmp(S1,S2,1) -> *(ubyte*)S1 - *(ubyte*)S2
      const Type *UCharPtr = PointerType::get(Type::Int8Ty);
      CastInst *Op1Cast = CastInst::create(
          Instruction::BitCast, LHS, UCharPtr, LHS->getName(), CI);
      CastInst *Op2Cast = CastInst::create(
          Instruction::BitCast, RHS, UCharPtr, RHS->getName(), CI);
      Value *S1V = new LoadInst(Op1Cast, LHS->getName()+".val", CI);
      Value *S2V = new LoadInst(Op2Cast, RHS->getName()+".val", CI);
      Value *RV = BinaryOperator::createSub(S1V, S2V, CI->getName()+".diff",CI);
      if (RV->getType() != CI->getType())
        RV = CastInst::createIntegerCast(RV, CI->getType(), false, 
                                         RV->getName(), CI);
      return ReplaceCallWith(CI, RV);
    }
    case 2:
      if (IsOnlyUsedInEqualsZeroComparison(CI)) {
        // TODO: IF both are aligned, use a short load/compare.
      
        // memcmp(S1,S2,2) -> S1[0]-S2[0] | S1[1]-S2[1] iff only ==/!= 0 matters
        const Type *UCharPtr = PointerType::get(Type::Int8Ty);
        CastInst *Op1Cast = CastInst::create(
            Instruction::BitCast, LHS, UCharPtr, LHS->getName(), CI);
        CastInst *Op2Cast = CastInst::create(
            Instruction::BitCast, RHS, UCharPtr, RHS->getName(), CI);
        Value *S1V1 = new LoadInst(Op1Cast, LHS->getName()+".val1", CI);
        Value *S2V1 = new LoadInst(Op2Cast, RHS->getName()+".val1", CI);
        Value *D1 = BinaryOperator::createSub(S1V1, S2V1,
                                              CI->getName()+".d1", CI);
        Constant *One = ConstantInt::get(Type::Int32Ty, 1);
        Value *G1 = new GetElementPtrInst(Op1Cast, One, "next1v", CI);
        Value *G2 = new GetElementPtrInst(Op2Cast, One, "next2v", CI);
        Value *S1V2 = new LoadInst(G1, LHS->getName()+".val2", CI);
        Value *S2V2 = new LoadInst(G2, RHS->getName()+".val2", CI);
        Value *D2 = BinaryOperator::createSub(S1V2, S2V2,
                                              CI->getName()+".d1", CI);
        Value *Or = BinaryOperator::createOr(D1, D2, CI->getName()+".res", CI);
        if (Or->getType() != CI->getType())
          Or = CastInst::createIntegerCast(Or, CI->getType(), false /*ZExt*/, 
                                           Or->getName(), CI);
        return ReplaceCallWith(CI, Or);
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
struct VISIBILITY_HIDDEN LLVMMemCpyMoveOptzn : public LibCallOptimization {
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
    uint64_t len = LEN->getZExtValue();
    uint64_t alignment = ALIGN->getZExtValue();
    if (alignment == 0)
      alignment = 1; // Alignment 0 is identity for alignment 1
    if (len > alignment)
      return false;

    // Get the type we will cast to, based on size of the string
    Value* dest = ci->getOperand(1);
    Value* src = ci->getOperand(2);
    const Type* castType = 0;
    switch (len) {
      case 0:
        // memcpy(d,s,0,a) -> d
        return ReplaceCallWith(ci, 0);
      case 1: castType = Type::Int8Ty; break;
      case 2: castType = Type::Int16Ty; break;
      case 4: castType = Type::Int32Ty; break;
      case 8: castType = Type::Int64Ty; break;
      default:
        return false;
    }

    // Cast source and dest to the right sized primitive and then load/store
    CastInst* SrcCast = CastInst::create(Instruction::BitCast,
        src, PointerType::get(castType), src->getName()+".cast", ci);
    CastInst* DestCast = CastInst::create(Instruction::BitCast,
        dest, PointerType::get(castType),dest->getName()+".cast", ci);
    LoadInst* LI = new LoadInst(SrcCast,SrcCast->getName()+".val",ci);
    new StoreInst(LI, DestCast, ci);
    return ReplaceCallWith(ci, 0);
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
struct VISIBILITY_HIDDEN LLVMMemSetOptimization : public LibCallOptimization {
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
    uint64_t len = LEN->getZExtValue();
    uint64_t alignment = ALIGN->getZExtValue();

    // Alignment 0 is identity for alignment 1
    if (alignment == 0)
      alignment = 1;

    // If the length is zero, this is a no-op
    if (len == 0) {
      // memset(d,c,0,a) -> noop
      return ReplaceCallWith(ci, 0);
    }

    // If the length is larger than the alignment, we can't optimize
    if (len > alignment)
      return false;

    // Make sure we have a constant ubyte to work with so we can extract
    // the value to be filled.
    ConstantInt* FILL = dyn_cast<ConstantInt>(ci->getOperand(2));
    if (!FILL)
      return false;
    if (FILL->getType() != Type::Int8Ty)
      return false;

    // memset(s,c,n) -> store s, c (for n=1,2,4,8)

    // Extract the fill character
    uint64_t fill_char = FILL->getZExtValue();
    uint64_t fill_value = fill_char;

    // Get the type we will cast to, based on size of memory area to fill, and
    // and the value we will store there.
    Value* dest = ci->getOperand(1);
    const Type* castType = 0;
    switch (len) {
      case 1:
        castType = Type::Int8Ty;
        break;
      case 2:
        castType = Type::Int16Ty;
        fill_value |= fill_char << 8;
        break;
      case 4:
        castType = Type::Int32Ty;
        fill_value |= fill_char << 8 | fill_char << 16 | fill_char << 24;
        break;
      case 8:
        castType = Type::Int64Ty;
        fill_value |= fill_char << 8 | fill_char << 16 | fill_char << 24;
        fill_value |= fill_char << 32 | fill_char << 40 | fill_char << 48;
        fill_value |= fill_char << 56;
        break;
      default:
        return false;
    }

    // Cast dest to the right sized primitive and then load/store
    CastInst* DestCast = new BitCastInst(dest, PointerType::get(castType), 
                                         dest->getName()+".cast", ci);
    new StoreInst(ConstantInt::get(castType,fill_value),DestCast, ci);
    return ReplaceCallWith(ci, 0);
  }
};

LLVMMemSetOptimization MemSet32Optimizer("llvm.memset.i32");
LLVMMemSetOptimization MemSet64Optimizer("llvm.memset.i64");


/// This LibCallOptimization will simplify calls to the "pow" library
/// function. It looks for cases where the result of pow is well known and
/// substitutes the appropriate value.
/// @brief Simplify the pow library function.
struct VISIBILITY_HIDDEN PowOptimization : public LibCallOptimization {
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
      if (Op1V == 1.0) // pow(1.0,x) -> 1.0
        return ReplaceCallWith(ci, ConstantFP::get(Ty, 1.0));
    }  else if (ConstantFP* Op2 = dyn_cast<ConstantFP>(expn)) {
      double Op2V = Op2->getValue();
      if (Op2V == 0.0) {
        // pow(x,0.0) -> 1.0
        return ReplaceCallWith(ci, ConstantFP::get(Ty,1.0));
      } else if (Op2V == 0.5) {
        // pow(x,0.5) -> sqrt(x)
        CallInst* sqrt_inst = new CallInst(SLC.get_sqrt(), base,
            ci->getName()+".pow",ci);
        return ReplaceCallWith(ci, sqrt_inst);
      } else if (Op2V == 1.0) {
        // pow(x,1.0) -> x
        return ReplaceCallWith(ci, base);
      } else if (Op2V == -1.0) {
        // pow(x,-1.0)    -> 1.0/x
        Value *div_inst = 
          BinaryOperator::createFDiv(ConstantFP::get(Ty, 1.0), base,
                                     ci->getName()+".pow", ci);
        return ReplaceCallWith(ci, div_inst);
      }
    }
    return false; // opt failed
  }
} PowOptimizer;

/// This LibCallOptimization will simplify calls to the "printf" library
/// function. It looks for cases where the result of printf is not used and the
/// operation can be reduced to something simpler.
/// @brief Simplify the printf library function.
struct VISIBILITY_HIDDEN PrintfOptimization : public LibCallOptimization {
public:
  /// @brief Default Constructor
  PrintfOptimization() : LibCallOptimization("printf",
      "Number of 'printf' calls simplified") {}

  /// @brief Make sure that the "printf" function has the right prototype
  virtual bool ValidateCalledFunction(const Function *F, SimplifyLibCalls &SLC){
    // Just make sure this has at least 1 arguments
    return F->arg_size() >= 1;
  }

  /// @brief Perform the printf optimization.
  virtual bool OptimizeCall(CallInst *CI, SimplifyLibCalls &SLC) {
    // If the call has more than 2 operands, we can't optimize it
    if (CI->getNumOperands() != 3)
      return false;

    // If the result of the printf call is used, none of these optimizations
    // can be made.
    if (!CI->use_empty())
      return false;

    // All the optimizations depend on the length of the first argument and the
    // fact that it is a constant string array. Check that now
    uint64_t FormatLen, FormatIdx;
    ConstantArray *CA = 0;
    if (!GetConstantStringInfo(CI->getOperand(1), CA, FormatLen, FormatIdx))
      return false;

    if (FormatLen != 2 && FormatLen != 3)
      return false;

    // The first character has to be a %
    if (cast<ConstantInt>(CA->getOperand(FormatIdx))->getZExtValue() != '%')
      return false;

    // Get the second character and switch on its value
    switch (cast<ConstantInt>(CA->getOperand(FormatIdx+1))->getZExtValue()) {
    default:  return false;
    case 's': {
      if (FormatLen != 3 ||
          cast<ConstantInt>(CA->getOperand(FormatIdx+2))->getZExtValue() !='\n')
        return false;

      // printf("%s\n",str) -> puts(str)
      new CallInst(SLC.get_puts(), CastToCStr(CI->getOperand(2), CI),
                   CI->getName(), CI);
      return ReplaceCallWith(CI, 0);
    }
    case 'c': {
      // printf("%c",c) -> putchar(c)
      if (FormatLen != 2)
        return false;
      
      Value *V = CI->getOperand(2);
      if (!isa<IntegerType>(V->getType()) ||
          cast<IntegerType>(V->getType())->getBitWidth() < 32)
        return false;

      V = CastInst::createSExtOrBitCast(V, Type::Int32Ty, CI->getName()+".int",
                                        CI);
      new CallInst(SLC.get_putchar(), V, "", CI);
      return ReplaceCallWith(CI, 0);
    }
    }
  }
} PrintfOptimizer;

/// This LibCallOptimization will simplify calls to the "fprintf" library
/// function. It looks for cases where the result of fprintf is not used and the
/// operation can be reduced to something simpler.
/// @brief Simplify the fprintf library function.
struct VISIBILITY_HIDDEN FPrintFOptimization : public LibCallOptimization {
public:
  /// @brief Default Constructor
  FPrintFOptimization() : LibCallOptimization("fprintf",
      "Number of 'fprintf' calls simplified") {}

  /// @brief Make sure that the "fprintf" function has the right prototype
  virtual bool ValidateCalledFunction(const Function *F, SimplifyLibCalls &SLC){
    const FunctionType *FT = F->getFunctionType();
    return FT->getNumParams() == 2 &&  // two fixed arguments.
           FT->getParamType(1) == PointerType::get(Type::Int8Ty) &&
           isa<PointerType>(FT->getParamType(0)) &&
           isa<IntegerType>(FT->getReturnType());
  }

  /// @brief Perform the fprintf optimization.
  virtual bool OptimizeCall(CallInst *CI, SimplifyLibCalls &SLC) {
    // If the call has more than 3 operands, we can't optimize it
    if (CI->getNumOperands() != 3 && CI->getNumOperands() != 4)
      return false;

    // All the optimizations depend on the format string.
    uint64_t FormatLen, FormatStartIdx;
    ConstantArray *CA = 0;
    if (!GetConstantStringInfo(CI->getOperand(2), CA, FormatLen,FormatStartIdx))
      return false;

    // IF fthis is just a format string, turn it into fwrite.
    if (CI->getNumOperands() == 3) {
      if (!CA->isCString()) return false;
      
      // Make sure there's no % in the constant array
      std::string S = CA->getAsString();

      for (unsigned i = FormatStartIdx, e = S.size(); i != e; ++i)
        if (S[i] == '%')
          return false; // we found a format specifier

      // fprintf(file,fmt) -> fwrite(fmt,strlen(fmt),file)
      const Type *FILEty = CI->getOperand(1)->getType();

      Value *FWriteArgs[] = {
        CI->getOperand(2),
        ConstantInt::get(SLC.getIntPtrType(), FormatLen),
        ConstantInt::get(SLC.getIntPtrType(), 1),
        CI->getOperand(1)
      };
      new CallInst(SLC.get_fwrite(FILEty), FWriteArgs, 4, CI->getName(), CI);
      return ReplaceCallWith(CI, ConstantInt::get(CI->getType(), FormatLen));
    }
    
    // The remaining optimizations require the format string to be length 2:
    // "%s" or "%c".
    if (FormatLen != 2)
      return false;

    // The first character has to be a % for us to handle it.
    if (cast<ConstantInt>(CA->getOperand(FormatStartIdx))->getZExtValue() !='%')
      return false;

    // Get the second character and switch on its value
    switch(cast<ConstantInt>(CA->getOperand(FormatStartIdx+1))->getZExtValue()){
    case 'c': {
      // fprintf(file,"%c",c) -> fputc(c,file)
      const Type *FILETy = CI->getOperand(1)->getType();
      Value *C = CastInst::createZExtOrBitCast(CI->getOperand(3), Type::Int32Ty,
                                               CI->getName()+".int", CI);
      new CallInst(SLC.get_fputc(FILETy), C, CI->getOperand(1), "", CI);
      return ReplaceCallWith(CI, ConstantInt::get(CI->getType(), 1));
    }
    case 's': {
      const Type *FILETy = CI->getOperand(1)->getType();
      uint64_t LitStrLen, LitStartIdx;
      ConstantArray *CA = 0;
      if (GetConstantStringInfo(CI->getOperand(3), CA, LitStrLen, LitStartIdx)){
        // fprintf(file,"%s",str) -> fwrite(str,strlen(str),1,file)
        Value *FWriteArgs[] = {
          CastToCStr(CI->getOperand(3), CI),
          ConstantInt::get(SLC.getIntPtrType(), LitStrLen),
          ConstantInt::get(SLC.getIntPtrType(), 1),
          CI->getOperand(1)
        };
        new CallInst(SLC.get_fwrite(FILETy), FWriteArgs, 4, CI->getName(), CI);
        return ReplaceCallWith(CI, ConstantInt::get(Type::Int32Ty, LitStrLen));
      }
      
      // If the result of the fprintf call is used, we can't do this.
      // TODO: we could insert a strlen call.
      if (!CI->use_empty())
        return false;
      
      // fprintf(file,"%s",str) -> fputs(str,file)
      new CallInst(SLC.get_fputs(FILETy), CastToCStr(CI->getOperand(3), CI),
                   CI->getOperand(1), CI->getName(), CI);
      return ReplaceCallWith(CI, 0);
    }
    default:
      return false;
    }
  }
} FPrintFOptimizer;

/// This LibCallOptimization will simplify calls to the "sprintf" library
/// function. It looks for cases where the result of sprintf is not used and the
/// operation can be reduced to something simpler.
/// @brief Simplify the sprintf library function.
struct VISIBILITY_HIDDEN SPrintFOptimization : public LibCallOptimization {
public:
  /// @brief Default Constructor
  SPrintFOptimization() : LibCallOptimization("sprintf",
      "Number of 'sprintf' calls simplified") {}

  /// @brief Make sure that the "sprintf" function has the right prototype
  virtual bool ValidateCalledFunction(const Function *F, SimplifyLibCalls &SLC){
    const FunctionType *FT = F->getFunctionType();
    return FT->getNumParams() == 2 &&  // two fixed arguments.
           FT->getParamType(1) == PointerType::get(Type::Int8Ty) &&
           FT->getParamType(0) == FT->getParamType(1) &&
           isa<IntegerType>(FT->getReturnType());
  }

  /// @brief Perform the sprintf optimization.
  virtual bool OptimizeCall(CallInst *CI, SimplifyLibCalls &SLC) {
    // If the call has more than 3 operands, we can't optimize it
    if (CI->getNumOperands() != 3 && CI->getNumOperands() != 4)
      return false;

    uint64_t FormatLen, FormatStartIdx;
    ConstantArray *CA = 0;
    if (!GetConstantStringInfo(CI->getOperand(2), CA, FormatLen,FormatStartIdx))
      return false;
    
    if (CI->getNumOperands() == 3) {
      if (!CA->isCString()) return false;
      
      // Make sure there's no % in the constant array
      std::string S = CA->getAsString();
      for (unsigned i = FormatStartIdx, e = S.size(); i != e; ++i)
        if (S[i] == '%')
          return false; // we found a format specifier
      
      // sprintf(str,fmt) -> llvm.memcpy(str,fmt,strlen(fmt),1)
      Value *MemCpyArgs[] = {
        CI->getOperand(1), CI->getOperand(2),
        ConstantInt::get(SLC.getIntPtrType(), FormatLen+1), // Copy the nul byte
        ConstantInt::get(Type::Int32Ty, 1)
      };
      new CallInst(SLC.get_memcpy(), MemCpyArgs, 4, "", CI);
      return ReplaceCallWith(CI, ConstantInt::get(CI->getType(), FormatLen));
    }

    // The remaining optimizations require the format string to be "%s" or "%c".
    if (FormatLen != 2 ||
        cast<ConstantInt>(CA->getOperand(FormatStartIdx))->getZExtValue() !='%')
      return false;

    // Get the second character and switch on its value
    switch (cast<ConstantInt>(CA->getOperand(1))->getZExtValue()) {
    case 'c': {
      // sprintf(dest,"%c",chr) -> store chr, dest
      Value *V = CastInst::createTruncOrBitCast(CI->getOperand(3),
                                                Type::Int8Ty, "char", CI);
      new StoreInst(V, CI->getOperand(1), CI);
      Value *Ptr = new GetElementPtrInst(CI->getOperand(1),
                                         ConstantInt::get(Type::Int32Ty, 1),
                                         CI->getOperand(1)->getName()+".end",
                                         CI);
      new StoreInst(ConstantInt::get(Type::Int8Ty,0), Ptr, CI);
      return ReplaceCallWith(CI, ConstantInt::get(Type::Int32Ty, 1));
    }
    case 's': {
      // sprintf(dest,"%s",str) -> llvm.memcpy(dest, str, strlen(str)+1, 1)
      Value *Len = new CallInst(SLC.get_strlen(),
                                CastToCStr(CI->getOperand(3), CI),
                                CI->getOperand(3)->getName()+".len", CI);
      Value *UnincLen = Len;
      Len = BinaryOperator::createAdd(Len, ConstantInt::get(Len->getType(), 1),
                                      Len->getName()+"1", CI);
      Value *MemcpyArgs[4] = {
        CI->getOperand(1),
        CastToCStr(CI->getOperand(3), CI),
        Len,
        ConstantInt::get(Type::Int32Ty, 1)
      };
      new CallInst(SLC.get_memcpy(), MemcpyArgs, 4, "", CI);
      
      // The strlen result is the unincremented number of bytes in the string.
      if (!CI->use_empty()) {
        if (UnincLen->getType() != CI->getType())
          UnincLen = CastInst::createIntegerCast(UnincLen, CI->getType(), false, 
                                                 Len->getName(), CI);
        CI->replaceAllUsesWith(UnincLen);
      }
      return ReplaceCallWith(CI, 0);
    }
    }
    return false;
  }
} SPrintFOptimizer;

/// This LibCallOptimization will simplify calls to the "fputs" library
/// function. It looks for cases where the result of fputs is not used and the
/// operation can be reduced to something simpler.
/// @brief Simplify the puts library function.
struct VISIBILITY_HIDDEN PutsOptimization : public LibCallOptimization {
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
    uint64_t len, StartIdx;
    ConstantArray *CA;
    if (!GetConstantStringInfo(ci->getOperand(1), CA, len, StartIdx))
      return false;

    switch (len) {
      case 0:
        // fputs("",F) -> noop
        break;
      case 1:
      {
        // fputs(s,F)  -> fputc(s[0],F)  (if s is constant and strlen(s) == 1)
        const Type* FILEptr_type = ci->getOperand(2)->getType();
        LoadInst* loadi = new LoadInst(ci->getOperand(1),
          ci->getOperand(1)->getName()+".byte",ci);
        CastInst* casti = new SExtInst(loadi, Type::Int32Ty, 
                                       loadi->getName()+".int", ci);
        new CallInst(SLC.get_fputc(FILEptr_type), casti,
                     ci->getOperand(2), "", ci);
        break;
      }
      default:
      {
        // fputs(s,F)  -> fwrite(s,1,len,F) (if s is constant and strlen(s) > 1)
        const Type* FILEptr_type = ci->getOperand(2)->getType();
        Value *parms[4] = {
          ci->getOperand(1),
          ConstantInt::get(SLC.getIntPtrType(),len),
          ConstantInt::get(SLC.getIntPtrType(),1),
          ci->getOperand(2)
        };
        new CallInst(SLC.get_fwrite(FILEptr_type), parms, 4, "", ci);
        break;
      }
    }
    return ReplaceCallWith(ci, 0);  // Known to have no uses (see above).
  }
} PutsOptimizer;

/// This LibCallOptimization will simplify calls to the "isdigit" library
/// function. It simply does range checks the parameter explicitly.
/// @brief Simplify the isdigit library function.
struct VISIBILITY_HIDDEN isdigitOptimization : public LibCallOptimization {
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
      uint64_t val = CI->getZExtValue();
      if (val >= '0' && val <= '9')
        return ReplaceCallWith(ci, ConstantInt::get(Type::Int32Ty, 1));
      else
        return ReplaceCallWith(ci, ConstantInt::get(Type::Int32Ty, 0));
    }

    // isdigit(c)   -> (unsigned)c - '0' <= 9
    CastInst* cast = CastInst::createIntegerCast(ci->getOperand(1),
        Type::Int32Ty, false/*ZExt*/, ci->getOperand(1)->getName()+".uint", ci);
    BinaryOperator* sub_inst = BinaryOperator::createSub(cast,
        ConstantInt::get(Type::Int32Ty,0x30),
        ci->getOperand(1)->getName()+".sub",ci);
    ICmpInst* setcond_inst = new ICmpInst(ICmpInst::ICMP_ULE,sub_inst,
        ConstantInt::get(Type::Int32Ty,9),
        ci->getOperand(1)->getName()+".cmp",ci);
    CastInst* c2 = new ZExtInst(setcond_inst, Type::Int32Ty, 
        ci->getOperand(1)->getName()+".isdigit", ci);
    return ReplaceCallWith(ci, c2);
  }
} isdigitOptimizer;

struct VISIBILITY_HIDDEN isasciiOptimization : public LibCallOptimization {
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
    Value *Cmp = new ICmpInst(ICmpInst::ICMP_ULT, V, 
                              ConstantInt::get(V->getType(), 128), 
                              V->getName()+".isascii", CI);
    if (Cmp->getType() != CI->getType())
      Cmp = new BitCastInst(Cmp, CI->getType(), Cmp->getName(), CI);
    return ReplaceCallWith(CI, Cmp);
  }
} isasciiOptimizer;


/// This LibCallOptimization will simplify calls to the "toascii" library
/// function. It simply does the corresponding and operation to restrict the
/// range of values to the ASCII character set (0-127).
/// @brief Simplify the toascii library function.
struct VISIBILITY_HIDDEN ToAsciiOptimization : public LibCallOptimization {
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
    Value *chr = ci->getOperand(1);
    Value *and_inst = BinaryOperator::createAnd(chr,
        ConstantInt::get(chr->getType(),0x7F),ci->getName()+".toascii",ci);
    return ReplaceCallWith(ci, and_inst);
  }
} ToAsciiOptimizer;

/// This LibCallOptimization will simplify calls to the "ffs" library
/// calls which find the first set bit in an int, long, or long long. The
/// optimization is to compute the result at compile time if the argument is
/// a constant.
/// @brief Simplify the ffs library function.
struct VISIBILITY_HIDDEN FFSOptimization : public LibCallOptimization {
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
    return F->arg_size() == 1 && F->getReturnType() == Type::Int32Ty;
  }

  /// @brief Perform the ffs optimization.
  virtual bool OptimizeCall(CallInst *TheCall, SimplifyLibCalls &SLC) {
    if (ConstantInt *CI = dyn_cast<ConstantInt>(TheCall->getOperand(1))) {
      // ffs(cnst)  -> bit#
      // ffsl(cnst) -> bit#
      // ffsll(cnst) -> bit#
      uint64_t val = CI->getZExtValue();
      int result = 0;
      if (val) {
        ++result;
        while ((val & 1) == 0) {
          ++result;
          val >>= 1;
        }
      }
      return ReplaceCallWith(TheCall, ConstantInt::get(Type::Int32Ty, result));
    }

    // ffs(x)   -> x == 0 ? 0 : llvm.cttz(x)+1
    // ffsl(x)  -> x == 0 ? 0 : llvm.cttz(x)+1
    // ffsll(x) -> x == 0 ? 0 : llvm.cttz(x)+1
    const Type *ArgType = TheCall->getOperand(1)->getType();
    const char *CTTZName;
    assert(ArgType->getTypeID() == Type::IntegerTyID &&
           "llvm.cttz argument is not an integer?");
    unsigned BitWidth = cast<IntegerType>(ArgType)->getBitWidth();
    if (BitWidth == 8)
      CTTZName = "llvm.cttz.i8";
    else if (BitWidth == 16)
      CTTZName = "llvm.cttz.i16"; 
    else if (BitWidth == 32)
      CTTZName = "llvm.cttz.i32";
    else {
      assert(BitWidth == 64 && "Unknown bitwidth");
      CTTZName = "llvm.cttz.i64";
    }
    
    Constant *F = SLC.getModule()->getOrInsertFunction(CTTZName, ArgType,
                                                       ArgType, NULL);
    Value *V = CastInst::createIntegerCast(TheCall->getOperand(1), ArgType, 
                                           false/*ZExt*/, "tmp", TheCall);
    Value *V2 = new CallInst(F, V, "tmp", TheCall);
    V2 = CastInst::createIntegerCast(V2, Type::Int32Ty, false/*ZExt*/, 
                                     "tmp", TheCall);
    V2 = BinaryOperator::createAdd(V2, ConstantInt::get(Type::Int32Ty, 1),
                                   "tmp", TheCall);
    Value *Cond = new ICmpInst(ICmpInst::ICMP_EQ, V, 
                               Constant::getNullValue(V->getType()), "tmp", 
                               TheCall);
    V2 = new SelectInst(Cond, ConstantInt::get(Type::Int32Ty, 0), V2,
                        TheCall->getName(), TheCall);
    return ReplaceCallWith(TheCall, V2);
  }
} FFSOptimizer;

/// This LibCallOptimization will simplify calls to the "ffsl" library
/// calls. It simply uses FFSOptimization for which the transformation is
/// identical.
/// @brief Simplify the ffsl library function.
struct VISIBILITY_HIDDEN FFSLOptimization : public FFSOptimization {
public:
  /// @brief Default Constructor
  FFSLOptimization() : FFSOptimization("ffsl",
      "Number of 'ffsl' calls simplified") {}

} FFSLOptimizer;

/// This LibCallOptimization will simplify calls to the "ffsll" library
/// calls. It simply uses FFSOptimization for which the transformation is
/// identical.
/// @brief Simplify the ffsl library function.
struct VISIBILITY_HIDDEN FFSLLOptimization : public FFSOptimization {
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
                                           Constant *(SimplifyLibCalls::*FP)()){
    if (FPExtInst *Cast = dyn_cast<FPExtInst>(CI->getOperand(1)))
      if (Cast->getOperand(0)->getType() == Type::FloatTy) {
        Value *New = new CallInst((SLC.*FP)(), Cast->getOperand(0),
                                  CI->getName(), CI);
        New = new FPExtInst(New, Type::DoubleTy, CI->getName(), CI);
        CI->replaceAllUsesWith(New);
        CI->eraseFromParent();
        if (Cast->use_empty())
          Cast->eraseFromParent();
        return true;
      }
    return false;
  }
};


struct VISIBILITY_HIDDEN FloorOptimization : public UnaryDoubleFPOptimizer {
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

struct VISIBILITY_HIDDEN CeilOptimization : public UnaryDoubleFPOptimizer {
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

struct VISIBILITY_HIDDEN RoundOptimization : public UnaryDoubleFPOptimizer {
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

struct VISIBILITY_HIDDEN RintOptimization : public UnaryDoubleFPOptimizer {
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

struct VISIBILITY_HIDDEN NearByIntOptimization : public UnaryDoubleFPOptimizer {
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

/// GetConstantStringInfo - This function computes the length of a
/// null-terminated constant array of integers.  This function can't rely on the
/// size of the constant array because there could be a null terminator in the
/// middle of the array.
///
/// We also have to bail out if we find a non-integer constant initializer
/// of one of the elements or if there is no null-terminator. The logic
/// below checks each of these conditions and will return true only if all
/// conditions are met.  If the conditions aren't met, this returns false.
///
/// If successful, the \p Array param is set to the constant array being
/// indexed, the \p Length parameter is set to the length of the null-terminated
/// string pointed to by V, the \p StartIdx value is set to the first
/// element of the Array that V points to, and true is returned.
static bool GetConstantStringInfo(Value *V, ConstantArray *&Array,
                                  uint64_t &Length, uint64_t &StartIdx) {
  assert(V != 0 && "Invalid args to GetConstantStringInfo");
  // Initialize results.
  Length = 0;
  StartIdx = 0;
  Array = 0;
  
  User *GEP = 0;
  // If the value is not a GEP instruction nor a constant expression with a
  // GEP instruction, then return false because ConstantArray can't occur
  // any other way
  if (GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(V)) {
    GEP = GEPI;
  } else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(V)) {
    if (CE->getOpcode() != Instruction::GetElementPtr)
      return false;
    GEP = CE;
  } else {
    return false;
  }

  // Make sure the GEP has exactly three arguments.
  if (GEP->getNumOperands() != 3)
    return false;

  // Check to make sure that the first operand of the GEP is an integer and
  // has value 0 so that we are sure we're indexing into the initializer.
  if (ConstantInt* op1 = dyn_cast<ConstantInt>(GEP->getOperand(1))) {
    if (!op1->isZero())
      return false;
  } else
    return false;

  // If the second index isn't a ConstantInt, then this is a variable index
  // into the array.  If this occurs, we can't say anything meaningful about
  // the string.
  StartIdx = 0;
  if (ConstantInt *CI = dyn_cast<ConstantInt>(GEP->getOperand(2)))
    StartIdx = CI->getZExtValue();
  else
    return false;

  // The GEP instruction, constant or instruction, must reference a global
  // variable that is a constant and is initialized. The referenced constant
  // initializer is the array that we'll use for optimization.
  GlobalVariable* GV = dyn_cast<GlobalVariable>(GEP->getOperand(0));
  if (!GV || !GV->isConstant() || !GV->hasInitializer())
    return false;
  Constant *GlobalInit = GV->getInitializer();

  // Handle the ConstantAggregateZero case
  if (isa<ConstantAggregateZero>(GlobalInit)) {
    // This is a degenerate case. The initializer is constant zero so the
    // length of the string must be zero.
    Length = 0;
    return true;
  }

  // Must be a Constant Array
  Array = dyn_cast<ConstantArray>(GlobalInit);
  if (!Array) return false;

  // Get the number of elements in the array
  uint64_t NumElts = Array->getType()->getNumElements();

  // Traverse the constant array from start_idx (derived above) which is
  // the place the GEP refers to in the array.
  Length = StartIdx;
  while (1) {
    if (Length >= NumElts)
      return false; // The array isn't null terminated.
    
    Constant *Elt = Array->getOperand(Length);
    if (ConstantInt *CI = dyn_cast<ConstantInt>(Elt)) {
      // Check for the null terminator.
      if (CI->isZero())
        break; // we found end of string
    } else
      return false; // This array isn't suitable, non-int initializer
    ++Length;
  }
  
  // Subtract out the initial value from the length
  Length -= StartIdx;
  return true; // success!
}

/// CastToCStr - Return V if it is an sbyte*, otherwise cast it to sbyte*,
/// inserting the cast before IP, and return the cast.
/// @brief Cast a value to a "C" string.
static Value *CastToCStr(Value *V, Instruction *IP) {
  assert(isa<PointerType>(V->getType()) && 
         "Can't cast non-pointer type to C string type");
  const Type *SBPTy = PointerType::get(Type::Int8Ty);
  if (V->getType() != SBPTy)
    return new BitCastInst(V, SBPTy, V->getName(), IP);
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
