//===- llvm/Pass.h - Base class for XForm Passes -----------------*- C++ -*--=//
//
// This file defines a base class that indicates that a specified class is a
// transformation pass implementation.
//
// Pass's are designed this way so that it is possible to run passes in a cache
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

#include <vector>
#include <map>
#include <iosfwd>
#include <typeinfo>
#include <cassert>
class Value;
class BasicBlock;
class Function;
class Module;
class AnalysisUsage;
class PassInfo;
class ImmutablePass;
template<class UnitType> class PassManagerT;
struct AnalysisResolver;

// AnalysisID - Use the PassInfo to identify a pass...
typedef const PassInfo* AnalysisID;

//===----------------------------------------------------------------------===//
/// Pass interface - Implemented by all 'passes'.  Subclass this if you are an
/// interprocedural optimization or you do not fit into any of the more
/// constrained passes described below.
///
class Pass {
  friend class AnalysisResolver;
  AnalysisResolver *Resolver;  // AnalysisResolver this pass is owned by...
  const PassInfo *PassInfoCache;

  // AnalysisImpls - This keeps track of which passes implement the interfaces
  // that are required by the current pass (to implement getAnalysis()).
  //
  std::vector<std::pair<const PassInfo*, Pass*> > AnalysisImpls;

  void operator=(const Pass&);  // DO NOT IMPLEMENT
  Pass(const Pass &);           // DO NOT IMPLEMENT
public:
  Pass() : Resolver(0), PassInfoCache(0) {}
  virtual ~Pass() {} // Destructor is virtual so we can be subclassed

  /// getPassName - Return a nice clean name for a pass.  This usually
  /// implemented in terms of the name that is registered by one of the
  /// Registration templates, but can be overloaded directly, and if nothing
  /// else is available, C++ RTTI will be consulted to get a SOMEWHAT
  /// intelligable name for the pass.
  ///
  virtual const char *getPassName() const;

  /// getPassInfo - Return the PassInfo data structure that corresponds to this
  /// pass...  If the pass has not been registered, this will return null.
  ///
  const PassInfo *getPassInfo() const;

  /// run - Run this pass, returning true if a modification was made to the
  /// module argument.  This should be implemented by all concrete subclasses.
  ///
  virtual bool run(Module &M) = 0;

  /// print - Print out the internal state of the pass.  This is called by
  /// Analyze to print out the contents of an analysis.  Otherwise it is not
  /// necessary to implement this method.  Beware that the module pointer MAY be
  /// null.  This automatically forwards to a virtual function that does not
  /// provide the Module* in case the analysis doesn't need it it can just be
  /// ignored.
  ///
  virtual void print(std::ostream &O, const Module *M) const { print(O); }
  virtual void print(std::ostream &O) const;
  void dump() const; // dump - call print(std::cerr, 0);


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

  // dumpPassStructure - Implement the -debug-passes=PassStructure option
  virtual void dumpPassStructure(unsigned Offset = 0);


  // getPassInfo - Static method to get the pass information from a class name.
  template<typename AnalysisClass>
  static const PassInfo *getClassPassInfo() {
    return lookupPassInfo(typeid(AnalysisClass));
  }

  // lookupPassInfo - Return the pass info object for the specified pass class,
  // or null if it is not known.
  static const PassInfo *lookupPassInfo(const std::type_info &TI);

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
  AnalysisType &getAnalysis() const {
    assert(Resolver && "Pass has not been inserted into a PassManager object!");
    const PassInfo *PI = getClassPassInfo<AnalysisType>();
    return getAnalysisID<AnalysisType>(PI);
  }

  template<typename AnalysisType>
  AnalysisType &getAnalysisID(const PassInfo *PI) const {
    assert(Resolver && "Pass has not been inserted into a PassManager object!");
    assert(PI && "getAnalysis for unregistered pass!");

    // PI *must* appear in AnalysisImpls.  Because the number of passes used
    // should be a small number, we just do a linear search over a (dense)
    // vector.
    Pass *ResultPass = 0;
    for (unsigned i = 0; ; ++i) {
      assert(i != AnalysisImpls.size() &&
             "getAnalysis*() called on an analysis that we not "
             "'required' by pass!");
      if (AnalysisImpls[i].first == PI) {
        ResultPass = AnalysisImpls[i].second;
        break;
      }
    }

    // Because the AnalysisType may not be a subclass of pass (for
    // AnalysisGroups), we must use dynamic_cast here to potentially adjust the
    // return pointer (because the class may multiply inherit, once from pass,
    // once from AnalysisType).
    //
    AnalysisType *Result = dynamic_cast<AnalysisType*>(ResultPass);
    assert(Result && "Pass does not implement interface required!");
    return *Result;
  }

private:
  friend class PassManagerT<Module>;
  friend class PassManagerT<Function>;
  friend class PassManagerT<BasicBlock>;
  virtual void addToPassManager(PassManagerT<Module> *PM, AnalysisUsage &AU);
};

inline std::ostream &operator<<(std::ostream &OS, const Pass &P) {
  P.print(OS, 0); return OS;
}



//===----------------------------------------------------------------------===//
/// ImmutablePass class - This class is used to provide information that does
/// not need to be run.  This is useful for things like target information and
/// "basic" versions of AnalysisGroups.
///
struct ImmutablePass : public Pass {
  /// initializePass - This method may be overriden by immutable passes to allow
  /// them to perform various initialization actions they require.  This is
  /// primarily because an ImmutablePass can "require" another ImmutablePass,
  /// and if it does, the overloaded version of initializePass may get access to
  /// these passes with getAnalysis<>.
  ///
  virtual void initializePass() {}

  /// ImmutablePasses are never run.
  ///
  virtual bool run(Module &M) { return false; }

private:
  friend class PassManagerT<Module>;
  virtual void addToPassManager(PassManagerT<Module> *PM, AnalysisUsage &AU);
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
struct FunctionPass : public Pass {
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

  /// run - On a module, we run this pass by initializing, ronOnFunction'ing
  /// once for every function in the module, then by finalizing.
  ///
  virtual bool run(Module &M);

  /// run - On a function, we simply initialize, run the function, then
  /// finalize.
  ///
  bool run(Function &F);

private:
  friend class PassManagerT<Module>;
  friend class PassManagerT<Function>;
  friend class PassManagerT<BasicBlock>;
  virtual void addToPassManager(PassManagerT<Module> *PM, AnalysisUsage &AU);
  virtual void addToPassManager(PassManagerT<Function> *PM, AnalysisUsage &AU);
};



//===----------------------------------------------------------------------===//
/// BasicBlockPass class - This class is used to implement most local
/// optimizations.  Optimizations should subclass this class if they
/// meet the following constraints:
///   1. Optimizations are local, operating on either a basic block or
///      instruction at a time.
///   2. Optimizations do not modify the CFG of the contained function, or any
///      other basic block in the function.
///   3. Optimizations conform to all of the constraints of FunctionPass's.
///
struct BasicBlockPass : public FunctionPass {
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
  bool run(BasicBlock &BB);

private:
  friend class PassManagerT<Function>;
  friend class PassManagerT<BasicBlock>;
  virtual void addToPassManager(PassManagerT<Function> *PM, AnalysisUsage &AU);
  virtual void addToPassManager(PassManagerT<BasicBlock> *PM,AnalysisUsage &AU);
};

// Include support files that contain important APIs commonly used by Passes,
// but that we want to separate out to make it easier to read the header files.
//
#include "llvm/PassSupport.h"
#include "llvm/PassAnalysisSupport.h"

#endif
