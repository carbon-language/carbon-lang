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
class Value;
class BasicBlock;
class Function;
class Module;
class AnalysisUsage;
class PassInfo;
template<class UnitType> class PassManagerT;
struct AnalysisResolver;

// AnalysisID - Use the PassInfo to identify a pass...
typedef const PassInfo* AnalysisID;


//===----------------------------------------------------------------------===//
// Pass interface - Implemented by all 'passes'.  Subclass this if you are an
// interprocedural optimization or you do not fit into any of the more
// constrained passes described below.
//
class Pass {
  friend class AnalysisResolver;
  AnalysisResolver *Resolver;  // AnalysisResolver this pass is owned by...
public:
  Pass(AnalysisResolver *AR = 0) : Resolver(AR) {}
  virtual ~Pass() {} // Destructor is virtual so we can be subclassed

  // getPassName - Return a nice clean name for a pass.  This should be
  // overloaded by the pass, but if it is not, C++ RTTI will be consulted to get
  // a SOMEWHAT intelligable name for the pass.
  //
  virtual const char *getPassName() const;

  // getPassInfo - Return the PassInfo data structure that corresponds to this
  // pass...
  const PassInfo *getPassInfo() const;

  // run - Run this pass, returning true if a modification was made to the
  // module argument.  This should be implemented by all concrete subclasses.
  //
  virtual bool run(Module &M) = 0;

  // print - Print out the internal state of the pass.  This is called by
  // Analyze to print out the contents of an analysis.  Otherwise it is not
  // neccesary to implement this method.  Beware that the module pointer MAY be
  // null.  This automatically forwards to a virtual function that does not
  // provide the Module* in case the analysis doesn't need it it can just be
  // ignored.
  //
  virtual void print(std::ostream &O, const Module *M) const { print(O); }
  virtual void print(std::ostream &O) const;
  void dump() const; // dump - call print(std::cerr, 0);


  // getAnalysisUsage - This function should be overriden by passes that need
  // analysis information to do their job.  If a pass specifies that it uses a
  // particular analysis result to this function, it can then use the
  // getAnalysis<AnalysisType>() function, below.
  //
  virtual void getAnalysisUsage(AnalysisUsage &Info) const {
    // By default, no analysis results are used, all are invalidated.
  }

  // releaseMemory() - This member can be implemented by a pass if it wants to
  // be able to release its memory when it is no longer needed.  The default
  // behavior of passes is to hold onto memory for the entire duration of their
  // lifetime (which is the entire compile time).  For pipelined passes, this
  // is not a big deal because that memory gets recycled every time the pass is
  // invoked on another program unit.  For IP passes, it is more important to
  // free memory when it is unused.
  //
  // Optionally implement this function to release pass memory when it is no
  // longer used.
  //
  virtual void releaseMemory() {}

  // dumpPassStructure - Implement the -debug-passes=PassStructure option
  virtual void dumpPassStructure(unsigned Offset = 0);

protected:
  // getAnalysis<AnalysisType>() - This function is used by subclasses to get to
  // the analysis information that they claim to use by overriding the
  // getAnalysisUsage function.
  //
  template<typename AnalysisType>
  AnalysisType &getAnalysis(AnalysisID AID = AnalysisType::ID) {
    assert(Resolver && "Pass not resident in a PassManager object!");
    return *(AnalysisType*)Resolver->getAnalysis(AID);
  }

  // getAnalysisToUpdate<AnalysisType>() - This function is used by subclasses
  // to get to the analysis information that might be around that needs to be
  // updated.  This is different than getAnalysis in that it can fail (ie the
  // analysis results haven't been computed), so should only be used if you
  // provide the capability to update an analysis that exists.
  //
  template<typename AnalysisType>
  AnalysisType *getAnalysisToUpdate(AnalysisID AID = AnalysisType::ID) {
    assert(Resolver && "Pass not resident in a PassManager object!");
    return (AnalysisType*)Resolver->getAnalysisToUpdate(AID);
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
// FunctionPass class - This class is used to implement most global
// optimizations.  Optimizations should subclass this class if they meet the
// following constraints:
//
//  1. Optimizations are organized globally, ie a function at a time
//  2. Optimizing a function does not cause the addition or removal of any
//     functions in the module
//
struct FunctionPass : public Pass {
  // doInitialization - Virtual method overridden by subclasses to do
  // any neccesary per-module initialization.
  //
  virtual bool doInitialization(Module &M) { return false; }

  // runOnFunction - Virtual method overriden by subclasses to do the
  // per-function processing of the pass.
  //
  virtual bool runOnFunction(Function &F) = 0;

  // doFinalization - Virtual method overriden by subclasses to do any post
  // processing needed after all passes have run.
  //
  virtual bool doFinalization(Module &M) { return false; }

  // run - On a module, we run this pass by initializing, ronOnFunction'ing once
  // for every function in the module, then by finalizing.
  //
  virtual bool run(Module &M);

  // run - On a function, we simply initialize, run the function, then finalize.
  //
  bool run(Function &F);

private:
  friend class PassManagerT<Module>;
  friend class PassManagerT<Function>;
  friend class PassManagerT<BasicBlock>;
  virtual void addToPassManager(PassManagerT<Module> *PM, AnalysisUsage &AU);
  virtual void addToPassManager(PassManagerT<Function> *PM, AnalysisUsage &AU);
};



//===----------------------------------------------------------------------===//
// BasicBlockPass class - This class is used to implement most local
// optimizations.  Optimizations should subclass this class if they
// meet the following constraints:
//   1. Optimizations are local, operating on either a basic block or
//      instruction at a time.
//   2. Optimizations do not modify the CFG of the contained function, or any
//      other basic block in the function.
//   3. Optimizations conform to all of the contstraints of FunctionPass's.
//
struct BasicBlockPass : public FunctionPass {
  // runOnBasicBlock - Virtual method overriden by subclasses to do the
  // per-basicblock processing of the pass.
  //
  virtual bool runOnBasicBlock(BasicBlock &BB) = 0;

  // To run this pass on a function, we simply call runOnBasicBlock once for
  // each function.
  //
  virtual bool runOnFunction(Function &F);

  // To run directly on the basic block, we initialize, runOnBasicBlock, then
  // finalize.
  //
  bool run(BasicBlock &BB);

private:
  friend class PassManagerT<Function>;
  friend class PassManagerT<BasicBlock>;
  virtual void addToPassManager(PassManagerT<Function> *PM, AnalysisUsage &AU);
  virtual void addToPassManager(PassManagerT<BasicBlock> *PM,AnalysisUsage &AU);
};

// Include support files that contain important APIs commonly used by Passes,
// but that we want to seperate out to make it easier to read the header files.
//
#include "llvm/PassSupport.h"
#include "llvm/PassAnalysisSupport.h"

#endif
