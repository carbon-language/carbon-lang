//===- llvm/Pass.h - Base class for Passes ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a base class that indicates that a specified class is a
// transformation pass implementation.
//
// Passes are designed this way so that it is possible to run passes in a cache
// and organizationally optimal order without having to specify it at the front
// end.  This allows arbitrary passes to be strung together and have them
// executed as effeciently as possible.
//
// Passes should extend one of the classes below, depending on the guarantees
// that it can make about what will be modified as it is run.  For example, most
// global optimizations should derive from FunctionPass, because they do not add
// or delete functions, they operate on the internals of the function.
//
// Note that this file #includes PassSupport.h and PassAnalysisSupport.h (at the
// bottom), so the APIs exposed by these files are also automatically available
// to all users of this file.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASS_H
#define LLVM_PASS_H

#include "llvm/Support/DataTypes.h"
#include "llvm/Support/Streams.h"
#include <vector>
#include <deque>
#include <map>
#include <iosfwd>
#include <cassert>

namespace llvm {

class Value;
class BasicBlock;
class Function;
class Module;
class AnalysisUsage;
class PassInfo;
class ImmutablePass;
class PMStack;
class AnalysisResolver;
class PMDataManager;

// AnalysisID - Use the PassInfo to identify a pass...
typedef const PassInfo* AnalysisID;

/// Different types of internal pass managers. External pass managers
/// (PassManager and FunctionPassManager) are not represented here.
/// Ordering of pass manager types is important here.
enum PassManagerType {
  PMT_Unknown = 0,
  PMT_ModulePassManager = 1, /// MPPassManager 
  PMT_CallGraphPassManager,  /// CGPassManager
  PMT_FunctionPassManager,   /// FPPassManager
  PMT_LoopPassManager,       /// LPPassManager
  PMT_BasicBlockPassManager, /// BBPassManager
  PMT_Last
};

typedef enum PassManagerType PassManagerType;

//===----------------------------------------------------------------------===//
/// Pass interface - Implemented by all 'passes'.  Subclass this if you are an
/// interprocedural optimization or you do not fit into any of the more
/// constrained passes described below.
///
class Pass {
  AnalysisResolver *Resolver;  // Used to resolve analysis
  intptr_t PassID;

  // AnalysisImpls - This keeps track of which passes implement the interfaces
  // that are required by the current pass (to implement getAnalysis()).
  //
  std::vector<std::pair<const PassInfo*, Pass*> > AnalysisImpls;

  void operator=(const Pass&);  // DO NOT IMPLEMENT
  Pass(const Pass &);           // DO NOT IMPLEMENT
public:
  explicit Pass(intptr_t pid) : Resolver(0), PassID(pid) {}
  explicit Pass(const void *pid) : Resolver(0), PassID((intptr_t)pid) {}
  virtual ~Pass();

  /// getPassName - Return a nice clean name for a pass.  This usually
  /// implemented in terms of the name that is registered by one of the
  /// Registration templates, but can be overloaded directly, and if nothing
  /// else is available, C++ RTTI will be consulted to get a SOMEWHAT
  /// intelligible name for the pass.
  ///
  virtual const char *getPassName() const;

  /// getPassInfo - Return the PassInfo data structure that corresponds to this
  /// pass...  If the pass has not been registered, this will return null.
  ///
  const PassInfo *getPassInfo() const;

  /// runPass - Run this pass, returning true if a modification was made to the
  /// module argument.  This should be implemented by all concrete subclasses.
  ///
  virtual bool runPass(Module &M) { return false; }
  virtual bool runPass(BasicBlock&) { return false; }

  /// print - Print out the internal state of the pass.  This is called by
  /// Analyze to print out the contents of an analysis.  Otherwise it is not
  /// necessary to implement this method.  Beware that the module pointer MAY be
  /// null.  This automatically forwards to a virtual function that does not
  /// provide the Module* in case the analysis doesn't need it it can just be
  /// ignored.
  ///
  virtual void print(std::ostream &O, const Module *M) const;
  void print(std::ostream *O, const Module *M) const { if (O) print(*O, M); }
  void dump() const; // dump - call print(std::cerr, 0);

  /// Each pass is responsible for assigning a pass manager to itself.
  /// PMS is the stack of available pass manager. 
  virtual void assignPassManager(PMStack &PMS, 
                                 PassManagerType T = PMT_Unknown) {}
  /// Check if available pass managers are suitable for this pass or not.
  virtual void preparePassManager(PMStack &PMS) {}
  
  ///  Return what kind of Pass Manager can manage this pass.
  virtual PassManagerType getPotentialPassManagerType() const {
    return PMT_Unknown; 
  }

  // Access AnalysisResolver
  inline void setResolver(AnalysisResolver *AR) { 
    assert (!Resolver && "Resolver is already set");
    Resolver = AR; 
  }
  inline AnalysisResolver *getResolver() { 
    assert (Resolver && "Resolver is not set");
    return Resolver; 
  }

  /// getAnalysisUsage - This function should be overriden by passes that need
  /// analysis information to do their job.  If a pass specifies that it uses a
  /// particular analysis result to this function, it can then use the
  /// getAnalysis<AnalysisType>() function, below.
  ///
  virtual void getAnalysisUsage(AnalysisUsage &Info) const {
    // By default, no analysis results are used, all are invalidated.
  }

  /// releaseMemory() - This member can be implemented by a pass if it wants to
  /// be able to release its memory when it is no longer needed.  The default
  /// behavior of passes is to hold onto memory for the entire duration of their
  /// lifetime (which is the entire compile time).  For pipelined passes, this
  /// is not a big deal because that memory gets recycled every time the pass is
  /// invoked on another program unit.  For IP passes, it is more important to
  /// free memory when it is unused.
  ///
  /// Optionally implement this function to release pass memory when it is no
  /// longer used.
  ///
  virtual void releaseMemory() {}

  /// verifyAnalysis() - This member can be implemented by a analysis pass to
  /// check state of analysis information. 
  virtual void verifyAnalysis() const {}

  // dumpPassStructure - Implement the -debug-passes=PassStructure option
  virtual void dumpPassStructure(unsigned Offset = 0);

  template<typename AnalysisClass>
  static const PassInfo *getClassPassInfo() {
    return lookupPassInfo(intptr_t(&AnalysisClass::ID));
  }

  // lookupPassInfo - Return the pass info object for the specified pass class,
  // or null if it is not known.
  static const PassInfo *lookupPassInfo(intptr_t TI);

  /// getAnalysisToUpdate<AnalysisType>() - This function is used by subclasses
  /// to get to the analysis information that might be around that needs to be
  /// updated.  This is different than getAnalysis in that it can fail (ie the
  /// analysis results haven't been computed), so should only be used if you
  /// provide the capability to update an analysis that exists.  This method is
  /// often used by transformation APIs to update analysis results for a pass
  /// automatically as the transform is performed.
  ///
  template<typename AnalysisType>
  AnalysisType *getAnalysisToUpdate() const; // Defined in PassAnalysisSupport.h

  /// mustPreserveAnalysisID - This method serves the same function as
  /// getAnalysisToUpdate, but works if you just have an AnalysisID.  This
  /// obviously cannot give you a properly typed instance of the class if you
  /// don't have the class name available (use getAnalysisToUpdate if you do),
  /// but it can tell you if you need to preserve the pass at least.
  ///
  bool mustPreserveAnalysisID(const PassInfo *AnalysisID) const;

  /// getAnalysis<AnalysisType>() - This function is used by subclasses to get
  /// to the analysis information that they claim to use by overriding the
  /// getAnalysisUsage function.
  ///
  template<typename AnalysisType>
  AnalysisType &getAnalysis() const; // Defined in PassAnalysisSupport.h

  template<typename AnalysisType>
  AnalysisType &getAnalysis(Function &F); // Defined in PassanalysisSupport.h

  template<typename AnalysisType>
  AnalysisType &getAnalysisID(const PassInfo *PI) const;

  template<typename AnalysisType>
  AnalysisType &getAnalysisID(const PassInfo *PI, Function &F);
};

inline std::ostream &operator<<(std::ostream &OS, const Pass &P) {
  P.print(OS, 0); return OS;
}

//===----------------------------------------------------------------------===//
/// ModulePass class - This class is used to implement unstructured
/// interprocedural optimizations and analyses.  ModulePasses may do anything
/// they want to the program.
///
class ModulePass : public Pass {
public:
  /// runOnModule - Virtual method overriden by subclasses to process the module
  /// being operated on.
  virtual bool runOnModule(Module &M) = 0;

  virtual bool runPass(Module &M) { return runOnModule(M); }
  virtual bool runPass(BasicBlock&) { return false; }

  virtual void assignPassManager(PMStack &PMS, 
                                 PassManagerType T = PMT_ModulePassManager);

  ///  Return what kind of Pass Manager can manage this pass.
  virtual PassManagerType getPotentialPassManagerType() const {
    return PMT_ModulePassManager;
  }

  explicit ModulePass(intptr_t pid) : Pass(pid) {}
  explicit ModulePass(const void *pid) : Pass(pid) {}
  // Force out-of-line virtual method.
  virtual ~ModulePass();
};


//===----------------------------------------------------------------------===//
/// ImmutablePass class - This class is used to provide information that does
/// not need to be run.  This is useful for things like target information and
/// "basic" versions of AnalysisGroups.
///
class ImmutablePass : public ModulePass {
public:
  /// initializePass - This method may be overriden by immutable passes to allow
  /// them to perform various initialization actions they require.  This is
  /// primarily because an ImmutablePass can "require" another ImmutablePass,
  /// and if it does, the overloaded version of initializePass may get access to
  /// these passes with getAnalysis<>.
  ///
  virtual void initializePass() {}

  /// ImmutablePasses are never run.
  ///
  bool runOnModule(Module &M) { return false; }

  explicit ImmutablePass(intptr_t pid) : ModulePass(pid) {}
  explicit ImmutablePass(const void *pid) : ModulePass(pid) {}
  
  // Force out-of-line virtual method.
  virtual ~ImmutablePass();
};

//===----------------------------------------------------------------------===//
/// FunctionPass class - This class is used to implement most global
/// optimizations.  Optimizations should subclass this class if they meet the
/// following constraints:
///
///  1. Optimizations are organized globally, i.e., a function at a time
///  2. Optimizing a function does not cause the addition or removal of any
///     functions in the module
///
class FunctionPass : public Pass {
public:
  explicit FunctionPass(intptr_t pid) : Pass(pid) {}
  explicit FunctionPass(const void *pid) : Pass(pid) {}

  /// doInitialization - Virtual method overridden by subclasses to do
  /// any necessary per-module initialization.
  ///
  virtual bool doInitialization(Module &M) { return false; }

  /// runOnFunction - Virtual method overriden by subclasses to do the
  /// per-function processing of the pass.
  ///
  virtual bool runOnFunction(Function &F) = 0;

  /// doFinalization - Virtual method overriden by subclasses to do any post
  /// processing needed after all passes have run.
  ///
  virtual bool doFinalization(Module &M) { return false; }

  /// runOnModule - On a module, we run this pass by initializing,
  /// ronOnFunction'ing once for every function in the module, then by
  /// finalizing.
  ///
  virtual bool runOnModule(Module &M);

  /// run - On a function, we simply initialize, run the function, then
  /// finalize.
  ///
  bool run(Function &F);

  virtual void assignPassManager(PMStack &PMS, 
                                 PassManagerType T = PMT_FunctionPassManager);

  ///  Return what kind of Pass Manager can manage this pass.
  virtual PassManagerType getPotentialPassManagerType() const {
    return PMT_FunctionPassManager;
  }
};



//===----------------------------------------------------------------------===//
/// BasicBlockPass class - This class is used to implement most local
/// optimizations.  Optimizations should subclass this class if they
/// meet the following constraints:
///   1. Optimizations are local, operating on either a basic block or
///      instruction at a time.
///   2. Optimizations do not modify the CFG of the contained function, or any
///      other basic block in the function.
///   3. Optimizations conform to all of the constraints of FunctionPasses.
///
class BasicBlockPass : public Pass {
public:
  explicit BasicBlockPass(intptr_t pid) : Pass(pid) {}
  explicit BasicBlockPass(const void *pid) : Pass(pid) {}

  /// doInitialization - Virtual method overridden by subclasses to do
  /// any necessary per-module initialization.
  ///
  virtual bool doInitialization(Module &M) { return false; }

  /// doInitialization - Virtual method overridden by BasicBlockPass subclasses
  /// to do any necessary per-function initialization.
  ///
  virtual bool doInitialization(Function &F) { return false; }

  /// runOnBasicBlock - Virtual method overriden by subclasses to do the
  /// per-basicblock processing of the pass.
  ///
  virtual bool runOnBasicBlock(BasicBlock &BB) = 0;

  /// doFinalization - Virtual method overriden by BasicBlockPass subclasses to
  /// do any post processing needed after all passes have run.
  ///
  virtual bool doFinalization(Function &F) { return false; }

  /// doFinalization - Virtual method overriden by subclasses to do any post
  /// processing needed after all passes have run.
  ///
  virtual bool doFinalization(Module &M) { return false; }


  // To run this pass on a function, we simply call runOnBasicBlock once for
  // each function.
  //
  bool runOnFunction(Function &F);

  /// To run directly on the basic block, we initialize, runOnBasicBlock, then
  /// finalize.
  ///
  virtual bool runPass(Module &M) { return false; }
  virtual bool runPass(BasicBlock &BB);

  virtual void assignPassManager(PMStack &PMS, 
                                 PassManagerType T = PMT_BasicBlockPassManager);

  ///  Return what kind of Pass Manager can manage this pass.
  virtual PassManagerType getPotentialPassManagerType() const {
    return PMT_BasicBlockPassManager; 
  }
};

/// PMStack
/// Top level pass manager (see PasManager.cpp) maintains active Pass Managers 
/// using PMStack. Each Pass implements assignPassManager() to connect itself
/// with appropriate manager. assignPassManager() walks PMStack to find
/// suitable manager.
///
/// PMStack is just a wrapper around standard deque that overrides pop() and
/// push() methods.
class PMStack {
public:
  typedef std::deque<PMDataManager *>::reverse_iterator iterator;
  iterator begin() { return S.rbegin(); }
  iterator end() { return S.rend(); }

  void handleLastUserOverflow();

  void pop();
  inline PMDataManager *top() { return S.back(); }
  void push(Pass *P);
  inline bool empty() { return S.empty(); }

  void dump();
private:
  std::deque<PMDataManager *> S;
};


/// If the user specifies the -time-passes argument on an LLVM tool command line
/// then the value of this boolean will be true, otherwise false.
/// @brief This is the storage for the -time-passes option.
extern bool TimePassesIsEnabled;

} // End llvm namespace

// Include support files that contain important APIs commonly used by Passes,
// but that we want to separate out to make it easier to read the header files.
//
#include "llvm/PassSupport.h"
#include "llvm/PassAnalysisSupport.h"

#endif
