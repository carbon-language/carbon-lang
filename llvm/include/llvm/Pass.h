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
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASS_H
#define LLVM_PASS_H

#include <vector>
#include <map>
class Value;
class BasicBlock;
class Function;
class Module;
class AnalysisUsage;
class AnalysisID;
template<class UnitType> class PassManagerT;
struct AnalysisResolver;

//===----------------------------------------------------------------------===//
// Pass interface - Implemented by all 'passes'.  Subclass this if you are an
// interprocedural optimization or you do not fit into any of the more
// constrained passes described below.
//
class Pass {
  friend class AnalysisResolver;
  AnalysisResolver *Resolver;  // AnalysisResolver this pass is owned by...
public:
  inline Pass(AnalysisResolver *AR = 0) : Resolver(AR) {}
  inline virtual ~Pass() {} // Destructor is virtual so we can be subclassed

  // getPassName - Return a nice clean name for a pass.  This should be
  // overloaded by the pass, but if it is not, C++ RTTI will be consulted to get
  // a SOMEWHAT intelligable name for the pass.
  //
  virtual const char *getPassName() const;

  // run - Run this pass, returning true if a modification was made to the
  // module argument.  This should be implemented by all concrete subclasses.
  //
  virtual bool run(Module &M) = 0;

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


// CreatePass - Helper template to invoke the constructor for the AnalysisID
// class. Note that this should be a template internal to AnalysisID, but
// GCC 2.95.3 crashes if we do that, doh.
//
template<class AnalysisType>
static Pass *CreatePass(AnalysisID ID) { return new AnalysisType(ID); }

//===----------------------------------------------------------------------===//
// AnalysisID - This class is used to uniquely identify an analysis pass that
//              is referenced by a transformation.
//
class AnalysisID {
  static unsigned NextID;               // Next ID # to deal out...
  unsigned ID;                          // Unique ID for this analysis
  Pass *(*Constructor)(AnalysisID);     // Constructor to return the Analysis

  AnalysisID();                         // Disable default ctor
  AnalysisID(unsigned id, Pass *(*Ct)(AnalysisID)) : ID(id), Constructor(Ct) {}
public:
  // create - the only way to define a new AnalysisID.  This static method is
  // supposed to be used to define the class static AnalysisID's that are
  // provided by analysis passes.  In the implementation (.cpp) file for the
  // class, there should be a line that looks like this (using CallGraph as an
  // example):
  //
  //  AnalysisID CallGraph::ID(AnalysisID::create<CallGraph>());
  //
  template<class AnalysisType>
  static AnalysisID create() {
    return AnalysisID(NextID++, CreatePass<AnalysisType>);
  }

  // Special Copy Constructor - This is how analysis passes declare that they
  // only depend on the CFG of the function they are working on, so they are not
  // invalidated by other passes that do not modify the CFG.  This should be
  // used like this:
  // AnalysisID DominatorSet::ID(AnalysisID::create<DominatorSet>(), true);
  //
  AnalysisID(const AnalysisID &AID, bool DependsOnlyOnCFG = false);


  inline Pass *createPass() const { return Constructor(*this); }

  inline bool operator==(const AnalysisID &A) const {
    return A.ID == ID;
  }
  inline bool operator!=(const AnalysisID &A) const {
    return A.ID != ID;
  }
  inline bool operator<(const AnalysisID &A) const {
    return ID < A.ID;
  }
};

//===----------------------------------------------------------------------===//
// AnalysisUsage - Represent the analysis usage information of a pass.  This
// tracks analyses that the pass REQUIRES (must available when the pass runs),
// and analyses that the pass PRESERVES (the pass does not invalidate the
// results of these analyses).  This information is provided by a pass to the
// Pass infrastructure through the getAnalysisUsage virtual function.
//
class AnalysisUsage {
  // Sets of analyses required and preserved by a pass
  std::vector<AnalysisID> Required, Preserved, Provided;
  bool PreservesAll;
public:
  AnalysisUsage() : PreservesAll(false) {}
  
  // addRequires - Add the specified ID to the required set of the usage info
  // for a pass.
  //
  AnalysisUsage &addRequired(AnalysisID ID) {
    Required.push_back(ID);
    return *this;
  }

  // addPreserves - Add the specified ID to the set of analyses preserved by
  // this pass
  //
  AnalysisUsage &addPreserved(AnalysisID ID) {
    Preserved.push_back(ID);
    return *this;
  }

  void addProvided(AnalysisID ID) {
    Provided.push_back(ID);
  }

  // PreservesAll - Set by analyses that do not transform their input at all
  void setPreservesAll() { PreservesAll = true; }
  bool preservesAll() const { return PreservesAll; }

  // preservesCFG - This function should be called to by the pass, iff they do
  // not:
  //
  //  1. Add or remove basic blocks from the function
  //  2. Modify terminator instructions in any way.
  //
  // This function annotates the AnalysisUsage info object to say that analyses
  // that only depend on the CFG are preserved by this pass.
  //
  void preservesCFG();

  const std::vector<AnalysisID> &getRequiredSet() const { return Required; }
  const std::vector<AnalysisID> &getPreservedSet() const { return Preserved; }
  const std::vector<AnalysisID> &getProvidedSet() const { return Provided; }
};



//===----------------------------------------------------------------------===//
// AnalysisResolver - Simple interface implemented by PassManagers objects that
// is used to pull analysis information out of them.
//
struct AnalysisResolver {
  virtual Pass *getAnalysisOrNullUp(AnalysisID ID) const = 0;
  virtual Pass *getAnalysisOrNullDown(AnalysisID ID) const = 0;
  Pass *getAnalysis(AnalysisID ID) {
    Pass *Result = getAnalysisOrNullUp(ID);
    assert(Result && "Pass has an incorrect analysis uses set!");
    return Result;
  }

  // getAnalysisToUpdate - Return an analysis result or null if it doesn't exist
  Pass *getAnalysisToUpdate(AnalysisID ID) {
    Pass *Result = getAnalysisOrNullUp(ID);
    return Result;
  }

  virtual unsigned getDepth() const = 0;

  virtual void markPassUsed(AnalysisID P, Pass *User) = 0;

  void startPass(Pass *P) {}
  void endPass(Pass *P) {}
protected:
  void setAnalysisResolver(Pass *P, AnalysisResolver *AR);
};

#endif
