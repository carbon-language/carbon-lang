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
// global optimizations should derive from MethodPass, because they do not add
// or delete methods, they operate on the internals of the method.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASS_H
#define LLVM_PASS_H

#include <vector>
#include <map>
class Value;
class BasicBlock;
class Method;
class Module;
class AnalysisID;
class Pass;
template<class UnitType> class PassManagerT;
struct AnalysisResolver;

// PassManager - Top level PassManagerT instantiation intended to be used.
// Implemented in PassManager.h
typedef PassManagerT<Module> PassManager;


//===----------------------------------------------------------------------===//
// Pass interface - Implemented by all 'passes'.  Subclass this if you are an
// interprocedural optimization or you do not fit into any of the more
// constrained passes described below.
//
class Pass {
  friend class AnalysisResolver;
  AnalysisResolver *Resolver;  // AnalysisResolver this pass is owned by...
public:
  typedef std::vector<AnalysisID> AnalysisSet;

  inline Pass(AnalysisResolver *AR = 0) : Resolver(AR) {}
  inline virtual ~Pass() {} // Destructor is virtual so we can be subclassed


  // run - Run this pass, returning true if a modification was made to the
  // module argument.  This should be implemented by all concrete subclasses.
  //
  virtual bool run(Module *M) = 0;

  // getAnalysisUsageInfo - This function should be overriden by passes that
  // need analysis information to do their job.  If a pass specifies that it
  // uses a particular analysis result to this function, it can then use the
  // getAnalysis<AnalysisType>() function, below.
  //
  // The Destroyed vector is used to communicate what analyses are invalidated
  // by this pass.  This is critical to specify so that the PassManager knows
  // which analysis must be rerun after this pass has proceeded.  Analysis are
  // only invalidated if run() returns true.
  //
  // The Provided vector is used for passes that provide analysis information,
  // these are the analysis passes themselves.  All analysis passes should
  // override this method to return themselves in the provided set.
  //
  virtual void getAnalysisUsageInfo(AnalysisSet &Required,
                                    AnalysisSet &Destroyed,
                                    AnalysisSet &Provided) {
    // By default, no analysis results are used or destroyed.
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

#ifndef NDEBUG
  // dumpPassStructure - Implement the -debug-passes=PassStructure option
  virtual void dumpPassStructure(unsigned Offset = 0);
#endif

protected:
  // getAnalysis<AnalysisType>() - This function is used by subclasses to get to
  // the analysis information that they claim to use by overriding the
  // getAnalysisUsageInfo function.
  //
  template<typename AnalysisType>
  AnalysisType &getAnalysis(AnalysisID AID = AnalysisType::ID) {
    assert(Resolver && "Pass not resident in a PassManager object!");
    return *(AnalysisType*)Resolver->getAnalysis(AID);
  }

private:
  friend class PassManagerT<Module>;
  friend class PassManagerT<Method>;
  friend class PassManagerT<BasicBlock>;
  virtual void addToPassManager(PassManagerT<Module> *PM, AnalysisSet &Req,
                                AnalysisSet &Destroyed, AnalysisSet &Provided);
};


//===----------------------------------------------------------------------===//
// MethodPass class - This class is used to implement most global optimizations.
// Optimizations should subclass this class if they meet the following
// constraints:
//  1. Optimizations are organized globally, ie a method at a time
//  2. Optimizing a method does not cause the addition or removal of any methods
//     in the module
//
struct MethodPass : public Pass {
  // doInitialization - Virtual method overridden by subclasses to do
  // any neccesary per-module initialization.
  //
  virtual bool doInitialization(Module *M) { return false; }

  // runOnMethod - Virtual method overriden by subclasses to do the per-method
  // processing of the pass.
  //
  virtual bool runOnMethod(Method *M) = 0;

  // doFinalization - Virtual method overriden by subclasses to do any post
  // processing needed after all passes have run.
  //
  virtual bool doFinalization(Module *M) { return false; }

  // run - On a module, we run this pass by initializing, ronOnMethod'ing once
  // for every method in the module, then by finalizing.
  //
  virtual bool run(Module *M);

  // run - On a method, we simply initialize, run the method, then finalize.
  //
  bool run(Method *M);

private:
  friend class PassManagerT<Module>;
  friend class PassManagerT<Method>;
  friend class PassManagerT<BasicBlock>;
  virtual void addToPassManager(PassManagerT<Module> *PM, AnalysisSet &Req,
                                AnalysisSet &Dest, AnalysisSet &Prov);
  virtual void addToPassManager(PassManagerT<Method> *PM,AnalysisSet &Req,
                                AnalysisSet &Dest, AnalysisSet &Prov);
};



//===----------------------------------------------------------------------===//
// BasicBlockPass class - This class is used to implement most local
// optimizations.  Optimizations should subclass this class if they
// meet the following constraints:
//   1. Optimizations are local, operating on either a basic block or
//      instruction at a time.
//   2. Optimizations do not modify the CFG of the contained method, or any
//      other basic block in the method.
//   3. Optimizations conform to all of the contstraints of MethodPass's.
//
struct BasicBlockPass : public MethodPass {
  // runOnBasicBlock - Virtual method overriden by subclasses to do the
  // per-basicblock processing of the pass.
  //
  virtual bool runOnBasicBlock(BasicBlock *M) = 0;

  // To run this pass on a method, we simply call runOnBasicBlock once for each
  // method.
  //
  virtual bool runOnMethod(Method *BB);

  // To run directly on the basic block, we initialize, runOnBasicBlock, then
  // finalize.
  //
  bool run(BasicBlock *BB);

private:
  friend class PassManagerT<Method>;
  friend class PassManagerT<BasicBlock>;
  virtual void addToPassManager(PassManagerT<Method> *PM, AnalysisSet &,
                                AnalysisSet &, AnalysisSet &);
  virtual void addToPassManager(PassManagerT<BasicBlock> *PM, AnalysisSet &,
                                AnalysisSet &, AnalysisSet &);
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
  virtual unsigned getDepth() const = 0;

  virtual void markPassUsed(AnalysisID P, Pass *User) = 0;
protected:
  void setAnalysisResolver(Pass *P, AnalysisResolver *AR);
};



#endif
