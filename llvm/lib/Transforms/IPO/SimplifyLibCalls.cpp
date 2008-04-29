//===- SimplifyLibCalls.cpp - Optimize specific well-known library calls --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "llvm/Intrinsics.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Config/config.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Transforms/IPO.h"
#include <cstring>
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
  static char ID; // Pass identification, replacement for typeid
  SimplifyLibCalls() : ModulePass((intptr_t)&ID) {}

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
    StringMap<LibCallOptimization*> OptznMap;
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
        StringMap<LibCallOptimization*>::iterator OMI =
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
                                         PointerType::getUnqual(Type::Int8Ty),
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
                                          PointerType::getUnqual(Type::Int8Ty),
                                          FILEptr_type, NULL);
    return fputs_func;
  }

  /// @brief Return a Function* for the fwrite libcall
  Constant *get_fwrite(const Type* FILEptr_type) {
    if (!fwrite_func)
      fwrite_func = M->getOrInsertFunction("fwrite", TD->getIntPtrType(),
                                           PointerType::getUnqual(Type::Int8Ty),
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
                                           PointerType::getUnqual(Type::Int8Ty),
                                           PointerType::getUnqual(Type::Int8Ty),
                                           PointerType::getUnqual(Type::Int8Ty),
                                           NULL);
    return strcpy_func;
  }

  /// @brief Return a Function* for the strlen libcall
  Constant *get_strlen() {
    if (!strlen_func)
      strlen_func = M->getOrInsertFunction("strlen", TD->getIntPtrType(),
                                           PointerType::getUnqual(Type::Int8Ty),
                                           NULL);
    return strlen_func;
  }

  /// @brief Return a Function* for the memchr libcall
  Constant *get_memchr() {
    if (!memchr_func)
      memchr_func = M->getOrInsertFunction("memchr",
                                           PointerType::getUnqual(Type::Int8Ty),
                                           PointerType::getUnqual(Type::Int8Ty),
                                           Type::Int32Ty, TD->getIntPtrType(),
                                           NULL);
    return memchr_func;
  }

  /// @brief Return a Function* for the memcpy libcall
  Constant *get_memcpy() {
    if (!memcpy_func) {
      Intrinsic::ID IID = (TD->getIntPtrType() == Type::Int32Ty) ?
        Intrinsic::memcpy_i32 : Intrinsic::memcpy_i64;
      memcpy_func = Intrinsic::getDeclaration(M, IID);
    }
    return memcpy_func;
  }

  Constant *getUnaryFloatFunction(const char *BaseName, const Type *T = 0) {
    if (T == 0) T = Type::FloatTy;
    
    char NameBuffer[20];
    const char *Name;
    if (T == Type::DoubleTy)
      Name = BaseName;              // floor
    else {
      Name = NameBuffer;
      unsigned NameLen = strlen(BaseName);
      assert(NameLen < sizeof(NameBuffer)-2 && "Buffer too small");
      memcpy(NameBuffer, BaseName, NameLen);
      if (T == Type::FloatTy)
        NameBuffer[NameLen] = 'f';  // floorf
      else
        NameBuffer[NameLen] = 'l';  // floorl
      NameBuffer[NameLen+1] = 0;
    }
    
    return M->getOrInsertFunction(Name, T, T, NULL);
  }
  
  Constant *get_floorf() { return getUnaryFloatFunction("floor"); }
  Constant *get_ceilf()  { return getUnaryFloatFunction( "ceil"); }
  Constant *get_roundf() { return getUnaryFloatFunction("round"); }
  Constant *get_rintf()  { return getUnaryFloatFunction( "rint"); }
  Constant *get_nearbyintf() { return getUnaryFloatFunction("nearbyint"); }

  
  
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
  }

private:
  /// Caches for function pointers.
  Constant *putchar_func, *puts_func;
  Constant *fputc_func, *fputs_func, *fwrite_func;
  Constant *memcpy_func, *memchr_func;
  Constant *sqrt_func;
  Constant *strcpy_func, *strlen_func;
  Module *M;             ///< Cached Module
  TargetData *TD;        ///< Cached TargetData
};

char SimplifyLibCalls::ID = 0;
// Register the pass
RegisterPass<SimplifyLibCalls>
X("simplify-libcalls", "Simplify well-known library calls");

} // anonymous namespace

// The only public symbol in this file which just instantiates the pass object
ModulePass *llvm::createSimplifyLibCallsPass() {
  return new SimplifyLibCalls();
}

// Forward declare utility functions.
static bool GetConstantStringInfo(Value *V, std::string &Str);
static Value *CastToCStr(Value *V, Instruction *IP);
static uint64_t GetStringLength(Value *V);


// Classes below here, in the anonymous namespace, are all subclasses of the
// LibCallOptimization class, each implementing all optimizations possible for a
// single well-known library call. Each has a static singleton instance that
// auto registers it into the "optlist" global above.
namespace {

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
      if (from->getReturnType() == ci->getOperand(1)->getType()
          && !isa<StructType>(from->getReturnType()))
        if (from->getName() == "main") {
          // Okay, time to actually do the optimization. First, get the basic
          // block of the call instruction
          BasicBlock* bb = ci->getParent();

          // Create a return instruction that we'll replace the call with.
          // Note that the argument of the return is the argument of the call
          // instruction.
          ReturnInst::Create(ci->getOperand(1), ci);

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
  virtual bool ValidateCalledFunction(const Function *F, SimplifyLibCalls &SLC){
    const FunctionType *FT = F->getFunctionType();
    return FT->getNumParams() == 2 &&
           FT->getReturnType() == PointerType::getUnqual(Type::Int8Ty) &&
           FT->getParamType(0) == FT->getReturnType() &&
           FT->getParamType(1) == FT->getReturnType();
  }

  /// @brief Optimize the strcat library function
  virtual bool OptimizeCall(CallInst *CI, SimplifyLibCalls &SLC) {
    // Extract some information from the instruction
    Value *Dst = CI->getOperand(1);
    Value *Src = CI->getOperand(2);

    // Extract the initializer (while making numerous checks) from the
    // source operand of the call to strcat.
    std::string SrcStr;
    if (!GetConstantStringInfo(Src, SrcStr))
      return false;

    // Handle the simple, do-nothing case
    if (SrcStr.empty())
      return ReplaceCallWith(CI, Dst);

    // We need to find the end of the destination string.  That's where the
    // memory is to be moved to. We just generate a call to strlen.
    CallInst *DstLen = CallInst::Create(SLC.get_strlen(), Dst,
                                        Dst->getName()+".len", CI);

    // Now that we have the destination's length, we must index into the
    // destination's pointer to get the actual memcpy destination (end of
    // the string .. we're concatenating).
    Dst = GetElementPtrInst::Create(Dst, DstLen, Dst->getName()+".indexed", CI);

    // We have enough information to now generate the memcpy call to
    // do the concatenation for us.
    Value *Vals[] = {
      Dst, Src,
      ConstantInt::get(SLC.getIntPtrType(), SrcStr.size()+1), // copy nul byte.
      ConstantInt::get(Type::Int32Ty, 1)  // alignment
    };
    CallInst::Create(SLC.get_memcpy(), Vals, Vals + 4, "", CI);

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
           FT->getReturnType() == PointerType::getUnqual(Type::Int8Ty) &&
           FT->getParamType(0) == FT->getReturnType() &&
           isa<IntegerType>(FT->getParamType(1));
  }

  /// @brief Perform the strchr optimizations
  virtual bool OptimizeCall(CallInst *CI, SimplifyLibCalls &SLC) {
    // Check that the first argument to strchr is a constant array of sbyte.
    std::string Str;
    if (!GetConstantStringInfo(CI->getOperand(1), Str))
      return false;

    // If the second operand is not constant, just lower this to memchr since we
    // know the length of the input string.
    ConstantInt *CSI = dyn_cast<ConstantInt>(CI->getOperand(2));
    if (!CSI) {
      Value *Args[3] = {
        CI->getOperand(1),
        CI->getOperand(2),
        ConstantInt::get(SLC.getIntPtrType(), Str.size()+1)
      };
      return ReplaceCallWith(CI, CallInst::Create(SLC.get_memchr(), Args, Args + 3,
                                                  CI->getName(), CI));
    }

    // strchr can find the nul character.
    Str += '\0';
    
    // Get the character we're looking for
    char CharValue = CSI->getSExtValue();

    // Compute the offset
    uint64_t i = 0;
    while (1) {
      if (i == Str.size())    // Didn't find the char.  strchr returns null.
        return ReplaceCallWith(CI, Constant::getNullValue(CI->getType()));
      // Did we find our match?
      if (Str[i] == CharValue)
        break;
      ++i;
    }

    // strchr(s+n,c)  -> gep(s+n+i,c)
    //    (if c is a constant integer and s is a constant string)
    Value *Idx = ConstantInt::get(Type::Int64Ty, i);
    Value *GEP = GetElementPtrInst::Create(CI->getOperand(1), Idx, 
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
           FT->getParamType(0) == PointerType::getUnqual(Type::Int8Ty);
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

    std::string Str1;
    if (!GetConstantStringInfo(Str1P, Str1))
      return false;
    if (Str1.empty()) {
      // strcmp("", x) -> *x
      Value *V = new LoadInst(Str2P, CI->getName()+".load", CI);
      V = new ZExtInst(V, CI->getType(), CI->getName()+".int", CI);
      return ReplaceCallWith(CI, V);
    }

    std::string Str2;
    if (!GetConstantStringInfo(Str2P, Str2))
      return false;
    if (Str2.empty()) {
      // strcmp(x,"") -> *x
      Value *V = new LoadInst(Str1P, CI->getName()+".load", CI);
      V = new ZExtInst(V, CI->getType(), CI->getName()+".int", CI);
      return ReplaceCallWith(CI, V);
    }

    // strcmp(x, y)  -> cnst  (if both x and y are constant strings)
    int R = strcmp(Str1.c_str(), Str2.c_str());
    return ReplaceCallWith(CI, ConstantInt::get(CI->getType(), R));
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
           FT->getParamType(0) == PointerType::getUnqual(Type::Int8Ty) &&
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
    if (Str1P == Str2P)  // strncmp(x,x, n)  -> 0
      return ReplaceCallWith(CI, ConstantInt::get(CI->getType(), 0));
    
    // Check the length argument, if it is Constant zero then the strings are
    // considered equal.
    uint64_t Length;
    if (ConstantInt *LengthArg = dyn_cast<ConstantInt>(CI->getOperand(3)))
      Length = LengthArg->getZExtValue();
    else
      return false;
    
    if (Length == 0) // strncmp(x,y,0)   -> 0
      return ReplaceCallWith(CI, ConstantInt::get(CI->getType(), 0));
    
    std::string Str1;
    if (!GetConstantStringInfo(Str1P, Str1))
      return false;
    if (Str1.empty()) {
      // strncmp("", x, n) -> *x
      Value *V = new LoadInst(Str2P, CI->getName()+".load", CI);
      V = new ZExtInst(V, CI->getType(), CI->getName()+".int", CI);
      return ReplaceCallWith(CI, V);
    }
    
    std::string Str2;
    if (!GetConstantStringInfo(Str2P, Str2))
      return false;
    if (Str2.empty()) {
      // strncmp(x, "", n) -> *x
      Value *V = new LoadInst(Str1P, CI->getName()+".load", CI);
      V = new ZExtInst(V, CI->getType(), CI->getName()+".int", CI);
      return ReplaceCallWith(CI, V);
    }
    
    // strncmp(x, y, n)  -> cnst  (if both x and y are constant strings)
    int R = strncmp(Str1.c_str(), Str2.c_str(), Length);
    return ReplaceCallWith(CI, ConstantInt::get(CI->getType(), R));
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
           FT->getParamType(0) == PointerType::getUnqual(Type::Int8Ty);
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
    std::string SrcStr;
    if (!GetConstantStringInfo(Src, SrcStr))
      return false;
    
    // If the constant string's length is zero we can optimize this by just
    // doing a store of 0 at the first byte of the destination
    if (SrcStr.empty()) {
      new StoreInst(ConstantInt::get(Type::Int8Ty, 0), Dst, CI);
      return ReplaceCallWith(CI, Dst);
    }

    // We have enough information to now generate the memcpy call to
    // do the concatenation for us.
    Value *MemcpyOps[] = {
      Dst, Src, // Pass length including nul byte.
      ConstantInt::get(SLC.getIntPtrType(), SrcStr.size()+1),
      ConstantInt::get(Type::Int32Ty, 1) // alignment
    };
    CallInst::Create(SLC.get_memcpy(), MemcpyOps, MemcpyOps + 4, "", CI);

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
           FT->getParamType(0) == PointerType::getUnqual(Type::Int8Ty) &&
           isa<IntegerType>(FT->getReturnType());
  }

  /// @brief Perform the strlen optimization
  virtual bool OptimizeCall(CallInst *CI, SimplifyLibCalls &SLC) {
    // Make sure we're dealing with an sbyte* here.
    Value *Src = CI->getOperand(1);

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
            Value *V = new LoadInst(Src, Src->getName()+".first", CI);
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
    // strlen("xyz") -> 3 (for example)
    if (uint64_t Len = GetStringLength(Src))
      return ReplaceCallWith(CI, ConstantInt::get(CI->getType(), Len-1));
    return false;
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
      const Type *UCharPtr = PointerType::getUnqual(Type::Int8Ty);
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
        const Type *UCharPtr = PointerType::getUnqual(Type::Int8Ty);
        CastInst *Op1Cast = CastInst::create(
            Instruction::BitCast, LHS, UCharPtr, LHS->getName(), CI);
        CastInst *Op2Cast = CastInst::create(
            Instruction::BitCast, RHS, UCharPtr, RHS->getName(), CI);
        Value *S1V1 = new LoadInst(Op1Cast, LHS->getName()+".val1", CI);
        Value *S2V1 = new LoadInst(Op2Cast, RHS->getName()+".val1", CI);
        Value *D1 = BinaryOperator::createSub(S1V1, S2V1,
                                              CI->getName()+".d1", CI);
        Constant *One = ConstantInt::get(Type::Int32Ty, 1);
        Value *G1 = GetElementPtrInst::Create(Op1Cast, One, "next1v", CI);
        Value *G2 = GetElementPtrInst::Create(Op2Cast, One, "next2v", CI);
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
/// function.  It simply converts them into calls to llvm.memcpy.*;
/// the resulting call should be optimized later.
/// @brief Simplify the memcpy library function.
struct VISIBILITY_HIDDEN MemCpyOptimization : public LibCallOptimization {
public:
  MemCpyOptimization() : LibCallOptimization("memcpy",
      "Number of 'memcpy' calls simplified") {}

  /// @brief Make sure that the "memcpy" function has the right prototype
  virtual bool ValidateCalledFunction(const Function *F, SimplifyLibCalls &SLC){
    const FunctionType *FT = F->getFunctionType();
    const Type* voidPtr = PointerType::getUnqual(Type::Int8Ty);
    return FT->getReturnType() == voidPtr && FT->getNumParams() == 3 &&
           FT->getParamType(0) == voidPtr &&
           FT->getParamType(1) == voidPtr &&
           FT->getParamType(2) == SLC.getIntPtrType();
  }

  /// @brief Perform the memcpy optimization
  virtual bool OptimizeCall(CallInst *CI, SimplifyLibCalls &SLC) {
    Value *MemcpyOps[] = {
      CI->getOperand(1), CI->getOperand(2), CI->getOperand(3),
      ConstantInt::get(Type::Int32Ty, 1)   // align = 1 always.
    };
    CallInst::Create(SLC.get_memcpy(), MemcpyOps, MemcpyOps + 4, "", CI);
    // memcpy always returns the destination
    return ReplaceCallWith(CI, CI->getOperand(1));
  }
} MemCpyOptimizer;

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
        src, PointerType::getUnqual(castType), src->getName()+".cast", ci);
    CastInst* DestCast = CastInst::create(Instruction::BitCast,
        dest, PointerType::getUnqual(castType),dest->getName()+".cast", ci);
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
    CastInst* DestCast = new BitCastInst(dest, PointerType::getUnqual(castType), 
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
  PowOptimization(const char *Name) : LibCallOptimization(Name,
      "Number of 'pow' calls simplified") {}

  /// @brief Make sure that the "pow" function has the right prototype
  virtual bool ValidateCalledFunction(const Function *F, SimplifyLibCalls &SLC){
    // Just make sure this has 2 arguments of the same FP type, which match the
    // result type.
    const FunctionType *FT = F->getFunctionType();
    return FT->getNumParams() == 2 && 
          FT->getParamType(0) == FT->getParamType(1) &&
          FT->getParamType(0) == FT->getReturnType() &&
          FT->getParamType(0)->isFloatingPoint();
  }

  /// @brief Perform the pow optimization.
  virtual bool OptimizeCall(CallInst *CI, SimplifyLibCalls &SLC) {
    Value *Op1 = CI->getOperand(1);
    Value *Op2 = CI->getOperand(2);
    if (ConstantFP *Op1C = dyn_cast<ConstantFP>(Op1)) {
      if (Op1C->isExactlyValue(1.0)) // pow(1.0, x) -> 1.0
        return ReplaceCallWith(CI, Op1C);
      if (Op1C->isExactlyValue(2.0)) {// pow(2.0, x) -> exp2(x)
        Value *Exp2 = SLC.getUnaryFloatFunction("exp2", CI->getType());
        Value *Res = CallInst::Create(Exp2, Op2, CI->getName()+"exp2", CI);
        return ReplaceCallWith(CI, Res);
      }
    }
    
    ConstantFP *Op2C = dyn_cast<ConstantFP>(Op2);
    if (Op2C == 0) return false;
    
    if (Op2C->getValueAPF().isZero()) {
      // pow(x, 0.0) -> 1.0
      return ReplaceCallWith(CI, ConstantFP::get(CI->getType(), 1.0));
    } else if (Op2C->isExactlyValue(0.5)) {
      // FIXME: This is not safe for -0.0 and -inf.  This can only be done when
      // 'unsafe' math optimizations are allowed.
      // x    pow(x, 0.5)  sqrt(x)
      // ---------------------------------------------
      // -0.0    +0.0       -0.0
      // -inf    +inf       NaN
#if 0
      // pow(x, 0.5) -> sqrt(x)
      Value *Sqrt = CallInst::Create(SLC.get_sqrt(), Op1, "sqrt", CI);
      return ReplaceCallWith(CI, Sqrt);
#endif
    } else if (Op2C->isExactlyValue(1.0)) {
      // pow(x, 1.0) -> x
      return ReplaceCallWith(CI, Op1);
    } else if (Op2C->isExactlyValue(2.0)) {
      // pow(x, 2.0) -> x*x
      Value *Sq = BinaryOperator::createMul(Op1, Op1, "pow2", CI);
      return ReplaceCallWith(CI, Sq);
    } else if (Op2C->isExactlyValue(-1.0)) {
      // pow(x, -1.0) -> 1.0/x
      Value *R = BinaryOperator::createFDiv(ConstantFP::get(CI->getType(), 1.0),
                                            Op1, CI->getName()+".pow", CI);
      return ReplaceCallWith(CI, R);
    }
    return false; // opt failed
  }
};

PowOptimization PowFOptimizer("powf");
PowOptimization PowOptimizer("pow");
PowOptimization PowLOptimizer("powl");
 
  
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
    // Just make sure this has at least 1 argument and returns an integer or
    // void type.
    const FunctionType *FT = F->getFunctionType();
    return FT->getNumParams() >= 1 &&
          (isa<IntegerType>(FT->getReturnType()) ||
           FT->getReturnType() == Type::VoidTy);
  }

  /// @brief Perform the printf optimization.
  virtual bool OptimizeCall(CallInst *CI, SimplifyLibCalls &SLC) {
    // All the optimizations depend on the length of the first argument and the
    // fact that it is a constant string array. Check that now
    std::string FormatStr;
    if (!GetConstantStringInfo(CI->getOperand(1), FormatStr))
      return false;
    
    // If this is a simple constant string with no format specifiers that ends
    // with a \n, turn it into a puts call.
    if (FormatStr.empty()) {
      // Tolerate printf's declared void.
      if (CI->use_empty()) return ReplaceCallWith(CI, 0);
      return ReplaceCallWith(CI, ConstantInt::get(CI->getType(), 0));
    }
    
    if (FormatStr.size() == 1) {
      // Turn this into a putchar call, even if it is a %.
      Value *V = ConstantInt::get(Type::Int32Ty, FormatStr[0]);
      CallInst::Create(SLC.get_putchar(), V, "", CI);
      if (CI->use_empty()) return ReplaceCallWith(CI, 0);
      return ReplaceCallWith(CI, ConstantInt::get(CI->getType(), 1));
    }

    // Check to see if the format str is something like "foo\n", in which case
    // we convert it to a puts call.  We don't allow it to contain any format
    // characters.
    if (FormatStr[FormatStr.size()-1] == '\n' &&
        FormatStr.find('%') == std::string::npos) {
      // Create a string literal with no \n on it.  We expect the constant merge
      // pass to be run after this pass, to merge duplicate strings.
      FormatStr.erase(FormatStr.end()-1);
      Constant *Init = ConstantArray::get(FormatStr, true);
      Constant *GV = new GlobalVariable(Init->getType(), true,
                                        GlobalVariable::InternalLinkage,
                                        Init, "str",
                                     CI->getParent()->getParent()->getParent());
      // Cast GV to be a pointer to char.
      GV = ConstantExpr::getBitCast(GV, PointerType::getUnqual(Type::Int8Ty));
      CallInst::Create(SLC.get_puts(), GV, "", CI);

      if (CI->use_empty()) return ReplaceCallWith(CI, 0);
      // The return value from printf includes the \n we just removed, so +1.
      return ReplaceCallWith(CI,
                             ConstantInt::get(CI->getType(), 
                                              FormatStr.size()+1));
    }
    
    
    // Only support %c or "%s\n" for now.
    if (FormatStr.size() < 2 || FormatStr[0] != '%')
      return false;

    // Get the second character and switch on its value
    switch (FormatStr[1]) {
    default:  return false;
    case 's':
      if (FormatStr != "%s\n" || CI->getNumOperands() < 3 ||
          // TODO: could insert strlen call to compute string length.
          !CI->use_empty())
        return false;

      // printf("%s\n",str) -> puts(str)
      CallInst::Create(SLC.get_puts(), CastToCStr(CI->getOperand(2), CI),
                       CI->getName(), CI);
      return ReplaceCallWith(CI, 0);
    case 'c': {
      // printf("%c",c) -> putchar(c)
      if (FormatStr.size() != 2 || CI->getNumOperands() < 3)
        return false;
      
      Value *V = CI->getOperand(2);
      if (!isa<IntegerType>(V->getType()) ||
          cast<IntegerType>(V->getType())->getBitWidth() > 32)
        return false;

      V = CastInst::createZExtOrBitCast(V, Type::Int32Ty, CI->getName()+".int",
                                        CI);
      CallInst::Create(SLC.get_putchar(), V, "", CI);
      return ReplaceCallWith(CI, ConstantInt::get(CI->getType(), 1));
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
           FT->getParamType(1) == PointerType::getUnqual(Type::Int8Ty) &&
           isa<PointerType>(FT->getParamType(0)) &&
           isa<IntegerType>(FT->getReturnType());
  }

  /// @brief Perform the fprintf optimization.
  virtual bool OptimizeCall(CallInst *CI, SimplifyLibCalls &SLC) {
    // If the call has more than 3 operands, we can't optimize it
    if (CI->getNumOperands() != 3 && CI->getNumOperands() != 4)
      return false;

    // All the optimizations depend on the format string.
    std::string FormatStr;
    if (!GetConstantStringInfo(CI->getOperand(2), FormatStr))
      return false;

    // If this is just a format string, turn it into fwrite.
    if (CI->getNumOperands() == 3) {
      for (unsigned i = 0, e = FormatStr.size(); i != e; ++i)
        if (FormatStr[i] == '%')
          return false; // we found a format specifier

      // fprintf(file,fmt) -> fwrite(fmt,strlen(fmt),file)
      const Type *FILEty = CI->getOperand(1)->getType();

      Value *FWriteArgs[] = {
        CI->getOperand(2),
        ConstantInt::get(SLC.getIntPtrType(), FormatStr.size()),
        ConstantInt::get(SLC.getIntPtrType(), 1),
        CI->getOperand(1)
      };
      CallInst::Create(SLC.get_fwrite(FILEty), FWriteArgs, FWriteArgs + 4, CI->getName(), CI);
      return ReplaceCallWith(CI, ConstantInt::get(CI->getType(), 
                                                  FormatStr.size()));
    }
    
    // The remaining optimizations require the format string to be length 2:
    // "%s" or "%c".
    if (FormatStr.size() != 2 || FormatStr[0] != '%')
      return false;

    // Get the second character and switch on its value
    switch (FormatStr[1]) {
    case 'c': {
      // fprintf(file,"%c",c) -> fputc(c,file)
      const Type *FILETy = CI->getOperand(1)->getType();
      Value *C = CastInst::createZExtOrBitCast(CI->getOperand(3), Type::Int32Ty,
                                               CI->getName()+".int", CI);
      SmallVector<Value *, 2> Args;
      Args.push_back(C);
      Args.push_back(CI->getOperand(1));
      CallInst::Create(SLC.get_fputc(FILETy), Args.begin(), Args.end(), "", CI);
      return ReplaceCallWith(CI, ConstantInt::get(CI->getType(), 1));
    }
    case 's': {
      const Type *FILETy = CI->getOperand(1)->getType();
      
      // If the result of the fprintf call is used, we can't do this.
      // TODO: we should insert a strlen call.
      if (!CI->use_empty() || !isa<PointerType>(CI->getOperand(3)->getType()))
        return false;
      
      // fprintf(file,"%s",str) -> fputs(str,file)
      SmallVector<Value *, 2> Args;
      Args.push_back(CastToCStr(CI->getOperand(3), CI));
      Args.push_back(CI->getOperand(1));
      CallInst::Create(SLC.get_fputs(FILETy), Args.begin(),
                       Args.end(), CI->getName(), CI);
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
           FT->getParamType(1) == PointerType::getUnqual(Type::Int8Ty) &&
           FT->getParamType(0) == FT->getParamType(1) &&
           isa<IntegerType>(FT->getReturnType());
  }

  /// @brief Perform the sprintf optimization.
  virtual bool OptimizeCall(CallInst *CI, SimplifyLibCalls &SLC) {
    // If the call has more than 3 operands, we can't optimize it
    if (CI->getNumOperands() != 3 && CI->getNumOperands() != 4)
      return false;

    std::string FormatStr;
    if (!GetConstantStringInfo(CI->getOperand(2), FormatStr))
      return false;
    
    if (CI->getNumOperands() == 3) {
      // Make sure there's no % in the constant array
      for (unsigned i = 0, e = FormatStr.size(); i != e; ++i)
        if (FormatStr[i] == '%')
          return false; // we found a format specifier
      
      // sprintf(str,fmt) -> llvm.memcpy(str,fmt,strlen(fmt),1)
      Value *MemCpyArgs[] = {
        CI->getOperand(1), CI->getOperand(2),
        ConstantInt::get(SLC.getIntPtrType(), 
                         FormatStr.size()+1), // Copy the nul byte.
        ConstantInt::get(Type::Int32Ty, 1)
      };
      CallInst::Create(SLC.get_memcpy(), MemCpyArgs, MemCpyArgs + 4, "", CI);
      return ReplaceCallWith(CI, ConstantInt::get(CI->getType(), 
                                                  FormatStr.size()));
    }

    // The remaining optimizations require the format string to be "%s" or "%c".
    if (FormatStr.size() != 2 || FormatStr[0] != '%')
      return false;

    // Get the second character and switch on its value
    switch (FormatStr[1]) {
    case 'c': {
      // sprintf(dest,"%c",chr) -> store chr, dest
      Value *V = CastInst::createTruncOrBitCast(CI->getOperand(3),
                                                Type::Int8Ty, "char", CI);
      new StoreInst(V, CI->getOperand(1), CI);
      Value *Ptr = GetElementPtrInst::Create(CI->getOperand(1),
                                             ConstantInt::get(Type::Int32Ty, 1),
                                             CI->getOperand(1)->getName()+".end",
                                             CI);
      new StoreInst(ConstantInt::get(Type::Int8Ty,0), Ptr, CI);
      return ReplaceCallWith(CI, ConstantInt::get(Type::Int32Ty, 1));
    }
    case 's': {
      // sprintf(dest,"%s",str) -> llvm.memcpy(dest, str, strlen(str)+1, 1)
      Value *Len = CallInst::Create(SLC.get_strlen(),
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
      CallInst::Create(SLC.get_memcpy(), MemcpyArgs, MemcpyArgs + 4, "", CI);
      
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
/// @brief Simplify the fputs library function.
struct VISIBILITY_HIDDEN FPutsOptimization : public LibCallOptimization {
public:
  /// @brief Default Constructor
  FPutsOptimization() : LibCallOptimization("fputs",
      "Number of 'fputs' calls simplified") {}

  /// @brief Make sure that the "fputs" function has the right prototype
  virtual bool ValidateCalledFunction(const Function *F, SimplifyLibCalls &SLC){
    // Just make sure this has 2 arguments
    return F->arg_size() == 2;
  }

  /// @brief Perform the fputs optimization.
  virtual bool OptimizeCall(CallInst *CI, SimplifyLibCalls &SLC) {
    // If the result is used, none of these optimizations work.
    if (!CI->use_empty())
      return false;

    // All the optimizations depend on the length of the first argument and the
    // fact that it is a constant string array. Check that now
    std::string Str;
    if (!GetConstantStringInfo(CI->getOperand(1), Str))
      return false;

    const Type *FILETy = CI->getOperand(2)->getType();
    // fputs(s,F)  -> fwrite(s,1,len,F) (if s is constant and strlen(s) > 1)
    Value *FWriteParms[4] = {
      CI->getOperand(1),
      ConstantInt::get(SLC.getIntPtrType(), Str.size()),
      ConstantInt::get(SLC.getIntPtrType(), 1),
      CI->getOperand(2)
    };
    CallInst::Create(SLC.get_fwrite(FILETy), FWriteParms, FWriteParms + 4, "", CI);
    return ReplaceCallWith(CI, 0);  // Known to have no uses (see above).
  }
} FPutsOptimizer;

/// This LibCallOptimization will simplify calls to the "fwrite" function.
struct VISIBILITY_HIDDEN FWriteOptimization : public LibCallOptimization {
public:
  /// @brief Default Constructor
  FWriteOptimization() : LibCallOptimization("fwrite",
                                       "Number of 'fwrite' calls simplified") {}
  
  /// @brief Make sure that the "fputs" function has the right prototype
  virtual bool ValidateCalledFunction(const Function *F, SimplifyLibCalls &SLC){
    const FunctionType *FT = F->getFunctionType();
    return FT->getNumParams() == 4 && 
           FT->getParamType(0) == PointerType::getUnqual(Type::Int8Ty) &&
           FT->getParamType(1) == FT->getParamType(2) &&
           isa<IntegerType>(FT->getParamType(1)) &&
           isa<PointerType>(FT->getParamType(3)) &&
           isa<IntegerType>(FT->getReturnType());
  }
  
  virtual bool OptimizeCall(CallInst *CI, SimplifyLibCalls &SLC) {
    // Get the element size and count.
    uint64_t EltSize, EltCount;
    if (ConstantInt *C = dyn_cast<ConstantInt>(CI->getOperand(2)))
      EltSize = C->getZExtValue();
    else
      return false;
    if (ConstantInt *C = dyn_cast<ConstantInt>(CI->getOperand(3)))
      EltCount = C->getZExtValue();
    else
      return false;
    
    // If this is writing zero records, remove the call (it's a noop).
    if (EltSize * EltCount == 0)
      return ReplaceCallWith(CI, ConstantInt::get(CI->getType(), 0));
    
    // If this is writing one byte, turn it into fputc.
    if (EltSize == 1 && EltCount == 1) {
      SmallVector<Value *, 2> Args;
      // fwrite(s,1,1,F) -> fputc(s[0],F)
      Value *Ptr = CI->getOperand(1);
      Value *Val = new LoadInst(Ptr, Ptr->getName()+".byte", CI);
      Args.push_back(new ZExtInst(Val, Type::Int32Ty, Val->getName()+".int", CI));
      Args.push_back(CI->getOperand(4));
      const Type *FILETy = CI->getOperand(4)->getType();
      CallInst::Create(SLC.get_fputc(FILETy), Args.begin(), Args.end(), "", CI);
      return ReplaceCallWith(CI, ConstantInt::get(CI->getType(), 1));
    }
    return false;
  }
} FWriteOptimizer;

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
      Cmp = new ZExtInst(Cmp, CI->getType(), Cmp->getName(), CI);
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
    assert(ArgType->getTypeID() == Type::IntegerTyID &&
           "llvm.cttz argument is not an integer?");
    Constant *F = Intrinsic::getDeclaration(SLC.getModule(),
                                            Intrinsic::cttz, &ArgType, 1);

    Value *V = CastInst::createIntegerCast(TheCall->getOperand(1), ArgType, 
                                           false/*ZExt*/, "tmp", TheCall);
    Value *V2 = CallInst::Create(F, V, "tmp", TheCall);
    V2 = CastInst::createIntegerCast(V2, Type::Int32Ty, false/*ZExt*/, 
                                     "tmp", TheCall);
    V2 = BinaryOperator::createAdd(V2, ConstantInt::get(Type::Int32Ty, 1),
                                   "tmp", TheCall);
    Value *Cond = new ICmpInst(ICmpInst::ICMP_EQ, V, 
                               Constant::getNullValue(V->getType()), "tmp", 
                               TheCall);
    V2 = SelectInst::Create(Cond, ConstantInt::get(Type::Int32Ty, 0), V2,
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
        Value *New = CallInst::Create((SLC.*FP)(), Cast->getOperand(0),
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
} // end anon namespace

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
static bool GetConstantStringInfo(Value *V, std::string &Str) {
  // Look through noop bitcast instructions.
  if (BitCastInst *BCI = dyn_cast<BitCastInst>(V)) {
    if (BCI->getType() == BCI->getOperand(0)->getType())
      return GetConstantStringInfo(BCI->getOperand(0), Str);
    return false;
  }
  
  // If the value is not a GEP instruction nor a constant expression with a
  // GEP instruction, then return false because ConstantArray can't occur
  // any other way
  User *GEP = 0;
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
  if (ConstantInt *Idx = dyn_cast<ConstantInt>(GEP->getOperand(1))) {
    if (!Idx->isZero())
      return false;
  } else
    return false;

  // If the second index isn't a ConstantInt, then this is a variable index
  // into the array.  If this occurs, we can't say anything meaningful about
  // the string.
  uint64_t StartIdx = 0;
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
    Str.clear();
    return true;
  }

  // Must be a Constant Array
  ConstantArray *Array = dyn_cast<ConstantArray>(GlobalInit);
  if (!Array) return false;

  // Get the number of elements in the array
  uint64_t NumElts = Array->getType()->getNumElements();

  // Traverse the constant array from StartIdx (derived above) which is
  // the place the GEP refers to in the array.
  for (unsigned i = StartIdx; i < NumElts; ++i) {
    Constant *Elt = Array->getOperand(i);
    ConstantInt *CI = dyn_cast<ConstantInt>(Elt);
    if (!CI) // This array isn't suitable, non-int initializer.
      return false;
    if (CI->isZero())
      return true; // we found end of string, success!
    Str += (char)CI->getZExtValue();
  }
  
  return false; // The array isn't null terminated.
}

/// GetStringLengthH - If we can compute the length of the string pointed to by
/// the specified pointer, return 'len+1'.  If we can't, return 0.
static uint64_t GetStringLengthH(Value *V, SmallPtrSet<PHINode*, 32> &PHIs) {
  // Look through noop bitcast instructions.
  if (BitCastInst *BCI = dyn_cast<BitCastInst>(V))
    return GetStringLengthH(BCI->getOperand(0), PHIs);
  
  // If this is a PHI node, there are two cases: either we have already seen it
  // or we haven't.
  if (PHINode *PN = dyn_cast<PHINode>(V)) {
    if (!PHIs.insert(PN))
      return ~0ULL;  // already in the set.

    // If it was new, see if all the input strings are the same length.
    uint64_t LenSoFar = ~0ULL;
    for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
      uint64_t Len = GetStringLengthH(PN->getIncomingValue(i), PHIs);
      if (Len == 0) return 0; // Unknown length -> unknown.
      
      if (Len == ~0ULL) continue;
      
      if (Len != LenSoFar && LenSoFar != ~0ULL)
        return 0;    // Disagree -> unknown.
      LenSoFar = Len;
    }
    
    // Success, all agree.
    return LenSoFar;
  }
  
  // strlen(select(c,x,y)) -> strlen(x) ^ strlen(y)
  if (SelectInst *SI = dyn_cast<SelectInst>(V)) {
    uint64_t Len1 = GetStringLengthH(SI->getTrueValue(), PHIs);
    if (Len1 == 0) return 0;
    uint64_t Len2 = GetStringLengthH(SI->getFalseValue(), PHIs);
    if (Len2 == 0) return 0;
    if (Len1 == ~0ULL) return Len2;
    if (Len2 == ~0ULL) return Len1;
    if (Len1 != Len2) return 0;
    return Len1;
  }
  
  // If the value is not a GEP instruction nor a constant expression with a
  // GEP instruction, then return unknown.
  User *GEP = 0;
  if (GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(V)) {
    GEP = GEPI;
  } else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(V)) {
    if (CE->getOpcode() != Instruction::GetElementPtr)
      return 0;
    GEP = CE;
  } else {
    return 0;
  }
  
  // Make sure the GEP has exactly three arguments.
  if (GEP->getNumOperands() != 3)
    return 0;
  
  // Check to make sure that the first operand of the GEP is an integer and
  // has value 0 so that we are sure we're indexing into the initializer.
  if (ConstantInt *Idx = dyn_cast<ConstantInt>(GEP->getOperand(1))) {
    if (!Idx->isZero())
      return 0;
  } else
    return 0;
  
  // If the second index isn't a ConstantInt, then this is a variable index
  // into the array.  If this occurs, we can't say anything meaningful about
  // the string.
  uint64_t StartIdx = 0;
  if (ConstantInt *CI = dyn_cast<ConstantInt>(GEP->getOperand(2)))
    StartIdx = CI->getZExtValue();
  else
    return 0;
  
  // The GEP instruction, constant or instruction, must reference a global
  // variable that is a constant and is initialized. The referenced constant
  // initializer is the array that we'll use for optimization.
  GlobalVariable* GV = dyn_cast<GlobalVariable>(GEP->getOperand(0));
  if (!GV || !GV->isConstant() || !GV->hasInitializer())
    return 0;
  Constant *GlobalInit = GV->getInitializer();
  
  // Handle the ConstantAggregateZero case, which is a degenerate case. The
  // initializer is constant zero so the length of the string must be zero.
  if (isa<ConstantAggregateZero>(GlobalInit))
    return 1;  // Len = 0 offset by 1.
  
  // Must be a Constant Array
  ConstantArray *Array = dyn_cast<ConstantArray>(GlobalInit);
  if (!Array || Array->getType()->getElementType() != Type::Int8Ty)
    return false;
  
  // Get the number of elements in the array
  uint64_t NumElts = Array->getType()->getNumElements();
  
  // Traverse the constant array from StartIdx (derived above) which is
  // the place the GEP refers to in the array.
  for (unsigned i = StartIdx; i != NumElts; ++i) {
    Constant *Elt = Array->getOperand(i);
    ConstantInt *CI = dyn_cast<ConstantInt>(Elt);
    if (!CI) // This array isn't suitable, non-int initializer.
      return 0;
    if (CI->isZero())
      return i-StartIdx+1; // We found end of string, success!
  }
  
  return 0; // The array isn't null terminated, conservatively return 'unknown'.
}
  
/// GetStringLength - If we can compute the length of the string pointed to by
/// the specified pointer, return 'len+1'.  If we can't, return 0.
static uint64_t GetStringLength(Value *V) {
  if (!isa<PointerType>(V->getType())) return 0;

  SmallPtrSet<PHINode*, 32> PHIs;
  uint64_t Len = GetStringLengthH(V, PHIs);
  // If Len is ~0ULL, we had an infinite phi cycle: this is dead code, so return
  // an empty string as a length.
  return Len == ~0ULL ? 1 : Len;
}

  
  
/// CastToCStr - Return V if it is an sbyte*, otherwise cast it to sbyte*,
/// inserting the cast before IP, and return the cast.
/// @brief Cast a value to a "C" string.
static Value *CastToCStr(Value *V, Instruction *IP) {
  assert(isa<PointerType>(V->getType()) && 
         "Can't cast non-pointer type to C string type");
  const Type *SBPTy = PointerType::getUnqual(Type::Int8Ty);
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
//   * puts("") -> putchar("\n")
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
