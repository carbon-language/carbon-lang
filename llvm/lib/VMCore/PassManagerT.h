//===- llvm/PassManager.h - Container for Passes -----------------*- C++ -*--=//
//
// This file defines the PassManager class.  This class is used to hold,
// maintain, and optimize execution of Pass's.  The PassManager class ensures
// that analysis results are available before a pass runs, and that Pass's are
// destroyed when the PassManager is destroyed.
//
// The PassManagerT template is instantiated three times to do its job.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSMANAGER_H
#define LLVM_PASSMANAGER_H

#include "llvm/Pass.h"
#include <string>

// PassManager - Top level PassManagerT instantiation intended to be used.
typedef PassManagerT<Module> PassManager;


//===----------------------------------------------------------------------===//
// PMDebug class - a set of debugging functions that are enabled when compiling
// with -g on.  If compiling at -O, all functions are inlined noops.
//
struct PMDebug {
#ifdef NDEBUG
  inline static void PrintPassStructure(Pass *) {}
  inline static void PrintPassInformation(unsigned,const char*,Pass*,Value*) {}
  inline static void PrintAnalysisSetInfo(unsigned,const char*,
                                          const Pass::AnalysisSet &) {}
#else
  // If compiled in debug mode, these functions can be enabled by setting
  // -debug-pass on the command line of the tool being used.
  //
  static void PrintPassStructure(Pass *P);
  static void PrintPassInformation(unsigned,const char*,Pass *, Value *);
  static void PrintAnalysisSetInfo(unsigned,const char*,const Pass::AnalysisSet&);
#endif
};



//===----------------------------------------------------------------------===//
// Declare the PassManagerTraits which will be specialized...
//
template<class UnitType> class PassManagerTraits;   // Do not define.


//===----------------------------------------------------------------------===//
// PassManagerT - Container object for passes.  The PassManagerT destructor
// deletes all passes contained inside of the PassManagerT, so you shouldn't 
// delete passes manually, and all passes should be dynamically allocated.
//
template<typename UnitType>
class PassManagerT : public PassManagerTraits<UnitType>,public AnalysisResolver{
  typedef typename PassManagerTraits<UnitType>::PassClass       PassClass;
  typedef typename PassManagerTraits<UnitType>::SubPassClass SubPassClass;
  typedef typename PassManagerTraits<UnitType>::BatcherClass BatcherClass;
  typedef typename PassManagerTraits<UnitType>::ParentClass   ParentClass;
  typedef          PassManagerTraits<UnitType>                     Traits;

  friend typename PassManagerTraits<UnitType>::PassClass;
  friend typename PassManagerTraits<UnitType>::SubPassClass;  
  friend class PassManagerTraits<UnitType>;

  std::vector<PassClass*> Passes;    // List of pass's to run

  // The parent of this pass manager...
  const ParentClass *Parent;

  // The current batcher if one is in use, or null
  BatcherClass *Batcher;

  // CurrentAnalyses - As the passes are being run, this map contains the
  // analyses that are available to the current pass for use.  This is accessed
  // through the getAnalysis() function in this class and in Pass.
  //
  std::map<AnalysisID, Pass*> CurrentAnalyses;

public:
  PassManagerT(ParentClass *Par = 0) : Parent(Par), Batcher(0) {}
  ~PassManagerT() {
    // Delete all of the contained passes...
    for (std::vector<PassClass*>::iterator I = Passes.begin(), E = Passes.end();
         I != E; ++I)
      delete *I;
  }

  // run - Run all of the queued passes on the specified module in an optimal
  // way.
  virtual bool runOnUnit(UnitType *M) {
    bool MadeChanges = false;
    closeBatcher();
    CurrentAnalyses.clear();

    // Output debug information...
    if (Parent == 0) PMDebug::PrintPassStructure(this);

    // Run all of the passes
    for (unsigned i = 0, e = Passes.size(); i < e; ++i) {
      PassClass *P = Passes[i];
      
      PMDebug::PrintPassInformation(getDepth(), "Executing Pass", P, (Value*)M);

      // Get information about what analyses the pass uses...
      std::vector<AnalysisID> Required, Destroyed, Provided;
      P->getAnalysisUsageInfo(Required, Destroyed, Provided);
      
      PMDebug::PrintAnalysisSetInfo(getDepth(), "Required", Required);

#ifndef NDEBUG
      // All Required analyses should be available to the pass as it runs!
      for (Pass::AnalysisSet::iterator I = Required.begin(), 
                                       E = Required.end(); I != E; ++I) {
        assert(getAnalysisOrNullUp(*I) && "Analysis used but not available!");
      }
#endif

      // Run the sub pass!
      MadeChanges |= Traits::runPass(P, M);

      PMDebug::PrintAnalysisSetInfo(getDepth(), "Destroyed", Destroyed);
      PMDebug::PrintAnalysisSetInfo(getDepth(), "Provided", Provided);

      // Erase all analyses in the destroyed set...
      for (Pass::AnalysisSet::iterator I = Destroyed.begin(), 
             E = Destroyed.end(); I != E; ++I)
        CurrentAnalyses.erase(*I);
      
      // Add all analyses in the provided set...
      for (Pass::AnalysisSet::iterator I = Provided.begin(),
             E = Provided.end(); I != E; ++I)
        CurrentAnalyses[*I] = P;
    }
    return MadeChanges;
  }

  // add - Add a pass to the queue of passes to run.  This passes ownership of
  // the Pass to the PassManager.  When the PassManager is destroyed, the pass
  // will be destroyed as well, so there is no need to delete the pass.  Also,
  // all passes MUST be new'd.
  //
  void add(PassClass *P) {
    // Get information about what analyses the pass uses...
    std::vector<AnalysisID> Required, Destroyed, Provided;
    P->getAnalysisUsageInfo(Required, Destroyed, Provided);

    // Loop over all of the analyses used by this pass,
    for (std::vector<AnalysisID>::iterator I = Required.begin(),
                                           E = Required.end(); I != E; ++I) {
      if (getAnalysisOrNullDown(*I) == 0)
        add((PassClass*)I->createPass());
    }

    // Tell the pass to add itself to this PassManager... the way it does so
    // depends on the class of the pass, and is critical to laying out passes in
    // an optimal order..
    //
    P->addToPassManager(this, Destroyed, Provided);
  }

#ifndef NDEBUG
  // dumpPassStructure - Implement the -debug-passes=PassStructure option
  virtual void dumpPassStructure(unsigned Offset = 0) {
    std::cerr << std::string(Offset*2, ' ') << "Pass Manager\n";
    for (std::vector<PassClass*>::iterator I = Passes.begin(), E = Passes.end();
         I != E; ++I)
      (*I)->dumpPassStructure(Offset+1);
  }
#endif

public:
  Pass *getAnalysisOrNullDown(AnalysisID ID) {
    std::map<AnalysisID, Pass*>::iterator I = CurrentAnalyses.find(ID);
    if (I == CurrentAnalyses.end()) {
      if (Batcher)
        return ((AnalysisResolver*)Batcher)->getAnalysisOrNullDown(ID);
      return 0;
    }
    return I->second;
  }

  Pass *getAnalysisOrNullUp(AnalysisID ID) {
    std::map<AnalysisID, Pass*>::iterator I = CurrentAnalyses.find(ID);
    if (I == CurrentAnalyses.end()) {
      if (Parent)
        return ((AnalysisResolver*)Parent)->getAnalysisOrNullUp(ID);
      return 0;
    }
    return I->second;
  }

  virtual unsigned getDepth() const {
    if (Parent == 0) return 0;
    return 1 + ((AnalysisResolver*)Parent)->getDepth();
  }

private:

  // addPass - These functions are used to implement the subclass specific
  // behaviors present in PassManager.  Basically the add(Pass*) method ends up
  // reflecting its behavior into a Pass::addToPassManager call.  Subclasses of
  // Pass override it specifically so that they can reflect the type
  // information inherent in "this" back to the PassManager.
  //
  // For generic Pass subclasses (which are interprocedural passes), we simply
  // add the pass to the end of the pass list and terminate any accumulation of
  // MethodPasses that are present.
  //
  void addPass(PassClass *P, Pass::AnalysisSet &Destroyed,
               Pass::AnalysisSet &Provided) {
    // Providers are analysis classes which are forbidden to modify the module
    // they are operating on, so they are allowed to be reordered to before the
    // batcher...
    //
    if (Batcher && Provided.empty())
      closeBatcher();                     // This pass cannot be batched!
    
    // Set the Resolver instance variable in the Pass so that it knows where to 
    // find this object...
    //
    setAnalysisResolver(P, this);
    Passes.push_back(P);

    // Erase all analyses in the destroyed set...
    for (std::vector<AnalysisID>::iterator I = Destroyed.begin(), 
           E = Destroyed.end(); I != E; ++I)
      CurrentAnalyses.erase(*I);

    // Add all analyses in the provided set...
    for (std::vector<AnalysisID>::iterator I = Provided.begin(),
           E = Provided.end(); I != E; ++I)
      CurrentAnalyses[*I] = P;
  }
  
  // For MethodPass subclasses, we must be sure to batch the MethodPasses
  // together in a MethodPassBatcher object so that all of the analyses are run
  // together a method at a time.
  //
  void addPass(SubPassClass *MP, Pass::AnalysisSet &Destroyed,
               Pass::AnalysisSet &Provided) {
    if (Batcher == 0) // If we don't have a batcher yet, make one now.
      Batcher = new BatcherClass(this);
    // The Batcher will queue them passes up
    MP->addToPassManager(Batcher, Destroyed, Provided);
  }

  // closeBatcher - Terminate the batcher that is being worked on.
  void closeBatcher() {
    if (Batcher) {
      Passes.push_back(Batcher);
      Batcher = 0;
    }
  }
};



//===----------------------------------------------------------------------===//
// PassManagerTraits<BasicBlock> Specialization
//
// This pass manager is used to group together all of the BasicBlockPass's
// into a single unit.
//
template<> struct PassManagerTraits<BasicBlock> : public BasicBlockPass {
  // PassClass - The type of passes tracked by this PassManager
  typedef BasicBlockPass PassClass;

  // SubPassClass - The types of classes that should be collated together
  // This is impossible to match, so BasicBlock instantiations of PassManagerT
  // do not collate.
  //
  typedef PassManagerT<Module> SubPassClass;

  // BatcherClass - The type to use for collation of subtypes... This class is
  // never instantiated for the PassManager<BasicBlock>, but it must be an 
  // instance of PassClass to typecheck.
  //
  typedef PassClass BatcherClass;

  // ParentClass - The type of the parent PassManager...
  typedef PassManagerT<Method> ParentClass;

  // PMType - The type of the passmanager that subclasses this class
  typedef PassManagerT<BasicBlock> PMType;

  // runPass - Specify how the pass should be run on the UnitType
  static bool runPass(PassClass *P, BasicBlock *M) {
    // todo, init and finalize
    return P->runOnBasicBlock(M);
  }

  // Implement the BasicBlockPass interface...
  virtual bool doInitialization(Module *M);
  virtual bool runOnBasicBlock(BasicBlock *BB);
  virtual bool doFinalization(Module *M);
};



//===----------------------------------------------------------------------===//
// PassManagerTraits<Method> Specialization
//
// This pass manager is used to group together all of the MethodPass's
// into a single unit.
//
template<> struct PassManagerTraits<Method> : public MethodPass {
  // PassClass - The type of passes tracked by this PassManager
  typedef MethodPass PassClass;

  // SubPassClass - The types of classes that should be collated together
  typedef BasicBlockPass SubPassClass;

  // BatcherClass - The type to use for collation of subtypes...
  typedef PassManagerT<BasicBlock> BatcherClass;

  // ParentClass - The type of the parent PassManager...
  typedef PassManagerT<Module> ParentClass;

  // PMType - The type of the passmanager that subclasses this class
  typedef PassManagerT<Method> PMType;

  // runPass - Specify how the pass should be run on the UnitType
  static bool runPass(PassClass *P, Method *M) {
    return P->runOnMethod(M);
  }

  // Implement the MethodPass interface...
  virtual bool doInitialization(Module *M);
  virtual bool runOnMethod(Method *M);
  virtual bool doFinalization(Module *M);
};



//===----------------------------------------------------------------------===//
// PassManagerTraits<Module> Specialization
//
// This is the top level PassManager implementation that holds generic passes.
//
template<> struct PassManagerTraits<Module> : public Pass {
  // PassClass - The type of passes tracked by this PassManager
  typedef Pass PassClass;

  // SubPassClass - The types of classes that should be collated together
  typedef MethodPass SubPassClass;

  // BatcherClass - The type to use for collation of subtypes...
  typedef PassManagerT<Method> BatcherClass;

  // ParentClass - The type of the parent PassManager...
  typedef void ParentClass;

  // runPass - Specify how the pass should be run on the UnitType
  static bool runPass(PassClass *P, Module *M) { return P->run(M); }

  // run - Implement the Pass interface...
  virtual bool run(Module *M) {
    return ((PassManagerT<Module>*)this)->runOnUnit(M);
  }
};



//===----------------------------------------------------------------------===//
// PassManagerTraits Method Implementations
//

// PassManagerTraits<BasicBlock> Implementations
//
inline bool PassManagerTraits<BasicBlock>::doInitialization(Module *M) {
  bool Changed = false;
  for (unsigned i = 0, e = ((PMType*)this)->Passes.size(); i != e; ++i)
    ((PMType*)this)->Passes[i]->doInitialization(M);
  return Changed;
}

inline bool PassManagerTraits<BasicBlock>::runOnBasicBlock(BasicBlock *BB) {
  return ((PMType*)this)->runOnUnit(BB);
}

inline bool PassManagerTraits<BasicBlock>::doFinalization(Module *M) {
  bool Changed = false;
  for (unsigned i = 0, e = ((PMType*)this)->Passes.size(); i != e; ++i)
    ((PMType*)this)->Passes[i]->doFinalization(M);
  return Changed;
}


// PassManagerTraits<Method> Implementations
//
inline bool PassManagerTraits<Method>::doInitialization(Module *M) {
  bool Changed = false;
  for (unsigned i = 0, e = ((PMType*)this)->Passes.size(); i != e; ++i)
    ((PMType*)this)->Passes[i]->doInitialization(M);
  return Changed;
}

inline bool PassManagerTraits<Method>::runOnMethod(Method *M) {
  return ((PMType*)this)->runOnUnit(M);
}

inline bool PassManagerTraits<Method>::doFinalization(Module *M) {
  bool Changed = false;
  for (unsigned i = 0, e = ((PMType*)this)->Passes.size(); i != e; ++i)
    ((PMType*)this)->Passes[i]->doFinalization(M);
  return Changed;
}

#endif
