//===- llvm/PassAnalysisSupport.h - Analysis Pass Support code ---*- C++ -*-==//
//
// This file defines stuff that is used to define and "use" Analysis Passes.
// This file is automatically #included by Pass.h, so:
//
//           NO .CPP FILES SHOULD INCLUDE THIS FILE DIRECTLY
//
// Instead, #include Pass.h
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASS_ANALYSIS_SUPPORT_H
#define LLVM_PASS_ANALYSIS_SUPPORT_H

// No need to include Pass.h, we are being included by it!


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
