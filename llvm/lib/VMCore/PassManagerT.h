//===- PassManagerT.h - Container for Passes --------------------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines the PassManagerT class.  This class is used to hold,
// maintain, and optimize execution of Pass's.  The PassManager class ensures
// that analysis results are available before a pass runs, and that Pass's are
// destroyed when the PassManager is destroyed.
//
// The PassManagerT template is instantiated three times to do its job.  The
// public PassManager class is a Pimpl around the PassManagerT<Module> interface
// to avoid having all of the PassManager clients being exposed to the
// implementation details herein.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSMANAGER_T_H
#define LLVM_PASSMANAGER_T_H

#include "llvm/Pass.h"
#include "Support/CommandLine.h"
#include "Support/LeakDetector.h"
#include "Support/Timer.h"
#include <algorithm>
#include <iostream>
class Annotable;

//===----------------------------------------------------------------------===//
// Pass debugging information.  Often it is useful to find out what pass is
// running when a crash occurs in a utility.  When this library is compiled with
// debugging on, a command line option (--debug-pass) is enabled that causes the
// pass name to be printed before it executes.
//

// Different debug levels that can be enabled...
enum PassDebugLevel {
  None, Arguments, Structure, Executions, Details
};

static cl::opt<enum PassDebugLevel>
PassDebugging("debug-pass", cl::Hidden,
              cl::desc("Print PassManager debugging information"),
              cl::values(
  clEnumVal(None      , "disable debug output"),
  clEnumVal(Arguments , "print pass arguments to pass to 'opt'"),
  clEnumVal(Structure , "print pass structure before run()"),
  clEnumVal(Executions, "print pass name before it is executed"),
  clEnumVal(Details   , "print pass details when it is executed"),
                         0));

//===----------------------------------------------------------------------===//
// PMDebug class - a set of debugging functions, that are not to be
// instantiated by the template.
//
struct PMDebug {
  static void PerformPassStartupStuff(Pass *P) {
    // If debugging is enabled, print out argument information...
    if (PassDebugging >= Arguments) {
      std::cerr << "Pass Arguments: ";
      PrintArgumentInformation(P);
      std::cerr << "\n";

      // Print the pass execution structure
      if (PassDebugging >= Structure)
        P->dumpPassStructure();
    }
  }

  static void PrintArgumentInformation(const Pass *P);
  static void PrintPassInformation(unsigned,const char*,Pass *, Annotable *);
  static void PrintAnalysisSetInfo(unsigned,const char*,Pass *P,
                                   const std::vector<AnalysisID> &);
};


//===----------------------------------------------------------------------===//
// TimingInfo Class - This class is used to calculate information about the
// amount of time each pass takes to execute.  This only happens when
// -time-passes is enabled on the command line.
//

class TimingInfo {
  std::map<Pass*, Timer> TimingData;
  TimerGroup TG;

  // Private ctor, must use 'create' member
  TimingInfo() : TG("... Pass execution timing report ...") {}
public:
  // TimingDtor - Print out information about timing information
  ~TimingInfo() {
    // Delete all of the timers...
    TimingData.clear();
    // TimerGroup is deleted next, printing the report.
  }

  // createTheTimeInfo - This method either initializes the TheTimeInfo pointer
  // to a non null value (if the -time-passes option is enabled) or it leaves it
  // null.  It may be called multiple times.
  static void createTheTimeInfo();

  void passStarted(Pass *P) {
    if (dynamic_cast<AnalysisResolver*>(P)) return;
    std::map<Pass*, Timer>::iterator I = TimingData.find(P);
    if (I == TimingData.end())
      I=TimingData.insert(std::make_pair(P, Timer(P->getPassName(), TG))).first;
    I->second.startTimer();
  }
  void passEnded(Pass *P) {
    if (dynamic_cast<AnalysisResolver*>(P)) return;
    std::map<Pass*, Timer>::iterator I = TimingData.find(P);
    assert (I != TimingData.end() && "passStarted/passEnded not nested right!");
    I->second.stopTimer();
  }
};

static TimingInfo *TheTimeInfo;

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
  typedef PassManagerTraits<UnitType> Traits;
  typedef typename Traits::PassClass       PassClass;
  typedef typename Traits::SubPassClass SubPassClass;
  typedef typename Traits::BatcherClass BatcherClass;
  typedef typename Traits::ParentClass   ParentClass;

  friend class PassManagerTraits<UnitType>::PassClass;
  friend class PassManagerTraits<UnitType>::SubPassClass;  
  friend class PassManagerTraits<UnitType>;
  friend class ImmutablePass;

  std::vector<PassClass*> Passes;    // List of passes to run
  std::vector<ImmutablePass*> ImmutablePasses;  // List of immutable passes

  // The parent of this pass manager...
  ParentClass * const Parent;

  // The current batcher if one is in use, or null
  BatcherClass *Batcher;

  // CurrentAnalyses - As the passes are being run, this map contains the
  // analyses that are available to the current pass for use.  This is accessed
  // through the getAnalysis() function in this class and in Pass.
  //
  std::map<AnalysisID, Pass*> CurrentAnalyses;

  // LastUseOf - This map keeps track of the last usage in our pipeline of a
  // particular pass.  When executing passes, the memory for .first is free'd
  // after .second is run.
  //
  std::map<Pass*, Pass*> LastUseOf;

public:
  PassManagerT(ParentClass *Par = 0) : Parent(Par), Batcher(0) {}
  ~PassManagerT() {
    // Delete all of the contained passes...
    for (typename std::vector<PassClass*>::iterator
           I = Passes.begin(), E = Passes.end(); I != E; ++I)
      delete *I;

    for (std::vector<ImmutablePass*>::iterator
           I = ImmutablePasses.begin(), E = ImmutablePasses.end(); I != E; ++I)
      delete *I;
  }

  // run - Run all of the queued passes on the specified module in an optimal
  // way.
  virtual bool runOnUnit(UnitType *M) {
    bool MadeChanges = false;
    closeBatcher();
    CurrentAnalyses.clear();

    TimingInfo::createTheTimeInfo();

    // Add any immutable passes to the CurrentAnalyses set...
    for (unsigned i = 0, e = ImmutablePasses.size(); i != e; ++i) {
      ImmutablePass *IPass = ImmutablePasses[i];
      if (const PassInfo *PI = IPass->getPassInfo()) {
        CurrentAnalyses[PI] = IPass;

        const std::vector<const PassInfo*> &II = PI->getInterfacesImplemented();
        for (unsigned i = 0, e = II.size(); i != e; ++i)
          CurrentAnalyses[II[i]] = IPass;
      }
    }

    // LastUserOf - This contains the inverted LastUseOfMap...
    std::map<Pass *, std::vector<Pass*> > LastUserOf;
    for (std::map<Pass*, Pass*>::iterator I = LastUseOf.begin(),
                                          E = LastUseOf.end(); I != E; ++I)
      LastUserOf[I->second].push_back(I->first);


    // Output debug information...
    if (Parent == 0) PMDebug::PerformPassStartupStuff(this);

    // Run all of the passes
    for (unsigned i = 0, e = Passes.size(); i < e; ++i) {
      PassClass *P = Passes[i];
      
      PMDebug::PrintPassInformation(getDepth(), "Executing Pass", P,
                                    (Annotable*)M);

      // Get information about what analyses the pass uses...
      AnalysisUsage AnUsage;
      P->getAnalysisUsage(AnUsage);
      PMDebug::PrintAnalysisSetInfo(getDepth(), "Required", P,
                                    AnUsage.getRequiredSet());

      // All Required analyses should be available to the pass as it runs!  Here
      // we fill in the AnalysisImpls member of the pass so that it can
      // successfully use the getAnalysis() method to retrieve the
      // implementations it needs.
      //
      P->AnalysisImpls.clear();
      P->AnalysisImpls.reserve(AnUsage.getRequiredSet().size());
      for (std::vector<const PassInfo *>::const_iterator
             I = AnUsage.getRequiredSet().begin(), 
             E = AnUsage.getRequiredSet().end(); I != E; ++I) {
        Pass *Impl = getAnalysisOrNullUp(*I);
        if (Impl == 0) {
          std::cerr << "Analysis '" << (*I)->getPassName()
                    << "' used but not available!";
          assert(0 && "Analysis used but not available!");
        } else if (PassDebugging == Details) {
          if ((*I)->getPassName() != std::string(Impl->getPassName()))
            std::cerr << "    Interface '" << (*I)->getPassName()
                    << "' implemented by '" << Impl->getPassName() << "'\n";
        }
        P->AnalysisImpls.push_back(std::make_pair(*I, Impl));
      }

      // Run the sub pass!
      if (TheTimeInfo) TheTimeInfo->passStarted(P);
      bool Changed = runPass(P, M);
      if (TheTimeInfo) TheTimeInfo->passEnded(P);
      MadeChanges |= Changed;

      // Check for memory leaks by the pass...
      LeakDetector::checkForGarbage(std::string("after running pass '") +
                                    P->getPassName() + "'");

      if (Changed)
        PMDebug::PrintPassInformation(getDepth()+1, "Made Modification", P,
                                      (Annotable*)M);
      PMDebug::PrintAnalysisSetInfo(getDepth(), "Preserved", P,
                                    AnUsage.getPreservedSet());


      // Erase all analyses not in the preserved set...
      if (!AnUsage.getPreservesAll()) {
        const std::vector<AnalysisID> &PreservedSet = AnUsage.getPreservedSet();
        for (std::map<AnalysisID, Pass*>::iterator I = CurrentAnalyses.begin(),
               E = CurrentAnalyses.end(); I != E; )
          if (std::find(PreservedSet.begin(), PreservedSet.end(), I->first) !=
              PreservedSet.end())
            ++I; // This analysis is preserved, leave it in the available set...
          else {
            if (!dynamic_cast<ImmutablePass*>(I->second)) {
              std::map<AnalysisID, Pass*>::iterator J = I++;
              CurrentAnalyses.erase(J);   // Analysis not preserved!
            } else {
              ++I;
            }
          }
      }

      // Add the current pass to the set of passes that have been run, and are
      // thus available to users.
      //
      if (const PassInfo *PI = P->getPassInfo()) {
        CurrentAnalyses[PI] = P;

        // This pass is the current implementation of all of the interfaces it
        // implements as well.
        //
        const std::vector<const PassInfo*> &II = PI->getInterfacesImplemented();
        for (unsigned i = 0, e = II.size(); i != e; ++i)
          CurrentAnalyses[II[i]] = P;
      }

      // Free memory for any passes that we are the last use of...
      std::vector<Pass*> &DeadPass = LastUserOf[P];
      for (std::vector<Pass*>::iterator I = DeadPass.begin(),E = DeadPass.end();
           I != E; ++I) {
        PMDebug::PrintPassInformation(getDepth()+1, "Freeing Pass", *I,
                                      (Annotable*)M);
        (*I)->releaseMemory();
      }

      // Make sure to remove dead passes from the CurrentAnalyses list...
      for (std::map<AnalysisID, Pass*>::iterator I = CurrentAnalyses.begin();
           I != CurrentAnalyses.end(); ) {
        std::vector<Pass*>::iterator DPI = std::find(DeadPass.begin(),
                                                     DeadPass.end(), I->second);
        if (DPI != DeadPass.end()) {    // This pass is dead now... remove it
          std::map<AnalysisID, Pass*>::iterator IDead = I++;
          CurrentAnalyses.erase(IDead);
        } else {
          ++I;  // Move on to the next element...
        }
      }
    }

    return MadeChanges;
  }

  // dumpPassStructure - Implement the -debug-passes=PassStructure option
  virtual void dumpPassStructure(unsigned Offset = 0) {
    // Print out the immutable passes...
    for (unsigned i = 0, e = ImmutablePasses.size(); i != e; ++i)
      ImmutablePasses[i]->dumpPassStructure(0);

    std::cerr << std::string(Offset*2, ' ') << Traits::getPMName()
              << " Pass Manager\n";
    for (typename std::vector<PassClass*>::iterator
           I = Passes.begin(), E = Passes.end(); I != E; ++I) {
      PassClass *P = *I;
      P->dumpPassStructure(Offset+1);

      // Loop through and see which classes are destroyed after this one...
      for (std::map<Pass*, Pass*>::iterator I = LastUseOf.begin(),
                                            E = LastUseOf.end(); I != E; ++I) {
        if (P == I->second) {
          std::cerr << "--" << std::string(Offset*2, ' ');
          I->first->dumpPassStructure(0);
        }
      }
    }
  }

  Pass *getImmutablePassOrNull(const PassInfo *ID) const {
    for (unsigned i = 0, e = ImmutablePasses.size(); i != e; ++i) {
      const PassInfo *IPID = ImmutablePasses[i]->getPassInfo();
      if (IPID == ID)
        return ImmutablePasses[i];
      
      // This pass is the current implementation of all of the interfaces it
      // implements as well.
      //
      const std::vector<const PassInfo*> &II =
        IPID->getInterfacesImplemented();
      for (unsigned j = 0, e = II.size(); j != e; ++j)
        if (II[j] == ID) return ImmutablePasses[i];
    }
    return 0;
  }

  Pass *getAnalysisOrNullDown(const PassInfo *ID) const {
    std::map<AnalysisID, Pass*>::const_iterator I = CurrentAnalyses.find(ID);

    if (I != CurrentAnalyses.end())
      return I->second;  // Found it.

    if (Pass *P = getImmutablePassOrNull(ID))
      return P;

    if (Batcher)
      return ((AnalysisResolver*)Batcher)->getAnalysisOrNullDown(ID);
    return 0;
  }

  Pass *getAnalysisOrNullUp(const PassInfo *ID) const {
    std::map<AnalysisID, Pass*>::const_iterator I = CurrentAnalyses.find(ID);
    if (I != CurrentAnalyses.end())
      return I->second;  // Found it.

    if (Parent)          // Try scanning...
      return Parent->getAnalysisOrNullUp(ID);
    else if (!ImmutablePasses.empty())
      return getImmutablePassOrNull(ID);
    return 0;
  }

  // markPassUsed - Inform higher level pass managers (and ourselves)
  // that these analyses are being used by this pass.  This is used to
  // make sure that analyses are not free'd before we have to use
  // them...
  //
  void markPassUsed(const PassInfo *P, Pass *User) {
    std::map<AnalysisID, Pass*>::const_iterator I = CurrentAnalyses.find(P);

    if (I != CurrentAnalyses.end()) {
      LastUseOf[I->second] = User;    // Local pass, extend the lifetime
    } else {
      // Pass not in current available set, must be a higher level pass
      // available to us, propagate to parent pass manager...  We tell the
      // parent that we (the passmanager) are using the analysis so that it
      // frees the analysis AFTER this pass manager runs.
      //
      if (Parent) {
        Parent->markPassUsed(P, this);
      } else {
        assert(getAnalysisOrNullUp(P) && 
               dynamic_cast<ImmutablePass*>(getAnalysisOrNullUp(P)) &&
               "Pass available but not found! "
               "Perhaps this is a module pass requiring a function pass?");
      }
    }
  }
  
  // Return the number of parent PassManagers that exist
  virtual unsigned getDepth() const {
    if (Parent == 0) return 0;
    return 1 + Parent->getDepth();
  }

  virtual unsigned getNumContainedPasses() const { return Passes.size(); }
  virtual const Pass *getContainedPass(unsigned N) const {
    assert(N < Passes.size() && "Pass number out of range!");
    return Passes[N];
  }

  // add - Add a pass to the queue of passes to run.  This gives ownership of
  // the Pass to the PassManager.  When the PassManager is destroyed, the pass
  // will be destroyed as well, so there is no need to delete the pass.  This
  // implies that all passes MUST be new'd.
  //
  void add(PassClass *P) {
    // Get information about what analyses the pass uses...
    AnalysisUsage AnUsage;
    P->getAnalysisUsage(AnUsage);
    const std::vector<AnalysisID> &Required = AnUsage.getRequiredSet();

    // Loop over all of the analyses used by this pass,
    for (std::vector<AnalysisID>::const_iterator I = Required.begin(),
           E = Required.end(); I != E; ++I) {
      if (getAnalysisOrNullDown(*I) == 0)
        add((PassClass*)(*I)->createPass());
    }

    // Tell the pass to add itself to this PassManager... the way it does so
    // depends on the class of the pass, and is critical to laying out passes in
    // an optimal order..
    //
    P->addToPassManager(this, AnUsage);
  }

  // add - H4x0r an ImmutablePass into a PassManager that might not be
  // expecting one.
  //
  void add(ImmutablePass *P) {
    // Get information about what analyses the pass uses...
    AnalysisUsage AnUsage;
    P->getAnalysisUsage(AnUsage);
    const std::vector<AnalysisID> &Required = AnUsage.getRequiredSet();

    // Loop over all of the analyses used by this pass,
    for (std::vector<AnalysisID>::const_iterator I = Required.begin(),
           E = Required.end(); I != E; ++I) {
      if (getAnalysisOrNullDown(*I) == 0)
        add((PassClass*)(*I)->createPass());
    }

    // Add the ImmutablePass to this PassManager.
    addPass(P, AnUsage);
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
  // FunctionPass's that are present.
  //
  void addPass(PassClass *P, AnalysisUsage &AnUsage) {
    const std::vector<AnalysisID> &RequiredSet = AnUsage.getRequiredSet();

    // FIXME: If this pass being added isn't killed by any of the passes in the
    // batcher class then we can reorder to pass to execute before the batcher
    // does, which will potentially allow us to batch more passes!
    //
    //const std::vector<AnalysisID> &ProvidedSet = AnUsage.getProvidedSet();
    if (Batcher /*&& ProvidedSet.empty()*/)
      closeBatcher();                     // This pass cannot be batched!
    
    // Set the Resolver instance variable in the Pass so that it knows where to 
    // find this object...
    //
    setAnalysisResolver(P, this);
    Passes.push_back(P);

    // Inform higher level pass managers (and ourselves) that these analyses are
    // being used by this pass.  This is used to make sure that analyses are not
    // free'd before we have to use them...
    //
    for (std::vector<AnalysisID>::const_iterator I = RequiredSet.begin(),
           E = RequiredSet.end(); I != E; ++I)
      markPassUsed(*I, P);     // Mark *I as used by P

    // Erase all analyses not in the preserved set...
    if (!AnUsage.getPreservesAll()) {
      const std::vector<AnalysisID> &PreservedSet = AnUsage.getPreservedSet();
      for (std::map<AnalysisID, Pass*>::iterator I = CurrentAnalyses.begin(),
             E = CurrentAnalyses.end(); I != E; ) {
        if (std::find(PreservedSet.begin(), PreservedSet.end(), I->first) ==
            PreservedSet.end()) {             // Analysis not preserved!
          CurrentAnalyses.erase(I);           // Remove from available analyses
          I = CurrentAnalyses.begin();
        } else {
          ++I;
        }
      }
    }

    // Add this pass to the currently available set...
    if (const PassInfo *PI = P->getPassInfo()) {
      CurrentAnalyses[PI] = P;

      // This pass is the current implementation of all of the interfaces it
      // implements as well.
      //
      const std::vector<const PassInfo*> &II = PI->getInterfacesImplemented();
      for (unsigned i = 0, e = II.size(); i != e; ++i)
        CurrentAnalyses[II[i]] = P;
    }

    // For now assume that our results are never used...
    LastUseOf[P] = P;
  }
  
  // For FunctionPass subclasses, we must be sure to batch the FunctionPass's
  // together in a BatcherClass object so that all of the analyses are run
  // together a function at a time.
  //
  void addPass(SubPassClass *MP, AnalysisUsage &AnUsage) {
    if (Batcher == 0) // If we don't have a batcher yet, make one now.
      Batcher = new BatcherClass(this);
    // The Batcher will queue the passes up
    MP->addToPassManager(Batcher, AnUsage);
  }

  // closeBatcher - Terminate the batcher that is being worked on.
  void closeBatcher() {
    if (Batcher) {
      Passes.push_back(Batcher);
      Batcher = 0;
    }
  }

public:
  // When an ImmutablePass is added, it gets added to the top level pass
  // manager.
  void addPass(ImmutablePass *IP, AnalysisUsage &AU) {
    if (Parent) { // Make sure this request goes to the top level passmanager...
      Parent->addPass(IP, AU);
      return;
    }

    // Set the Resolver instance variable in the Pass so that it knows where to 
    // find this object...
    //
    setAnalysisResolver(IP, this);
    ImmutablePasses.push_back(IP);

    // All Required analyses should be available to the pass as it initializes!
    // Here we fill in the AnalysisImpls member of the pass so that it can
    // successfully use the getAnalysis() method to retrieve the implementations
    // it needs.
    //
    IP->AnalysisImpls.clear();
    IP->AnalysisImpls.reserve(AU.getRequiredSet().size());
    for (std::vector<const PassInfo *>::const_iterator 
           I = AU.getRequiredSet().begin(),
           E = AU.getRequiredSet().end(); I != E; ++I) {
      Pass *Impl = getAnalysisOrNullUp(*I);
      if (Impl == 0) {
        std::cerr << "Analysis '" << (*I)->getPassName()
                  << "' used but not available!";
        assert(0 && "Analysis used but not available!");
      } else if (PassDebugging == Details) {
        if ((*I)->getPassName() != std::string(Impl->getPassName()))
          std::cerr << "    Interface '" << (*I)->getPassName()
                    << "' implemented by '" << Impl->getPassName() << "'\n";
      }
      IP->AnalysisImpls.push_back(std::make_pair(*I, Impl));
    }
    
    // Initialize the immutable pass...
    IP->initializePass();
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
  typedef PassManagerT<Function> ParentClass;

  // PMType - The type of the passmanager that subclasses this class
  typedef PassManagerT<BasicBlock> PMType;

  // runPass - Specify how the pass should be run on the UnitType
  static bool runPass(PassClass *P, BasicBlock *M) {
    // todo, init and finalize
    return P->runOnBasicBlock(*M);
  }

  // getPMName() - Return the name of the unit the PassManager operates on for
  // debugging.
  const char *getPMName() const { return "BasicBlock"; }
  virtual const char *getPassName() const { return "BasicBlock Pass Manager"; }

  // Implement the BasicBlockPass interface...
  virtual bool doInitialization(Module &M);
  virtual bool doInitialization(Function &F);
  virtual bool runOnBasicBlock(BasicBlock &BB);
  virtual bool doFinalization(Function &F);
  virtual bool doFinalization(Module &M);

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
  }
};



//===----------------------------------------------------------------------===//
// PassManagerTraits<Function> Specialization
//
// This pass manager is used to group together all of the FunctionPass's
// into a single unit.
//
template<> struct PassManagerTraits<Function> : public FunctionPass {
  // PassClass - The type of passes tracked by this PassManager
  typedef FunctionPass PassClass;

  // SubPassClass - The types of classes that should be collated together
  typedef BasicBlockPass SubPassClass;

  // BatcherClass - The type to use for collation of subtypes...
  typedef PassManagerT<BasicBlock> BatcherClass;

  // ParentClass - The type of the parent PassManager...
  typedef PassManagerT<Module> ParentClass;

  // PMType - The type of the passmanager that subclasses this class
  typedef PassManagerT<Function> PMType;

  // runPass - Specify how the pass should be run on the UnitType
  static bool runPass(PassClass *P, Function *F) {
    return P->runOnFunction(*F);
  }

  // getPMName() - Return the name of the unit the PassManager operates on for
  // debugging.
  const char *getPMName() const { return "Function"; }
  virtual const char *getPassName() const { return "Function Pass Manager"; }

  // Implement the FunctionPass interface...
  virtual bool doInitialization(Module &M);
  virtual bool runOnFunction(Function &F);
  virtual bool doFinalization(Module &M);

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
  }
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
  typedef FunctionPass SubPassClass;

  // BatcherClass - The type to use for collation of subtypes...
  typedef PassManagerT<Function> BatcherClass;

  // ParentClass - The type of the parent PassManager...
  typedef AnalysisResolver ParentClass;

  // runPass - Specify how the pass should be run on the UnitType
  static bool runPass(PassClass *P, Module *M) { return P->run(*M); }

  // getPMName() - Return the name of the unit the PassManager operates on for
  // debugging.
  const char *getPMName() const { return "Module"; }
  virtual const char *getPassName() const { return "Module Pass Manager"; }

  // run - Implement the PassManager interface...
  bool run(Module &M) {
    return ((PassManagerT<Module>*)this)->runOnUnit(&M);
  }
};



//===----------------------------------------------------------------------===//
// PassManagerTraits Method Implementations
//

// PassManagerTraits<BasicBlock> Implementations
//
inline bool PassManagerTraits<BasicBlock>::doInitialization(Module &M) {
  bool Changed = false;
  for (unsigned i = 0, e = ((PMType*)this)->Passes.size(); i != e; ++i)
    ((PMType*)this)->Passes[i]->doInitialization(M);
  return Changed;
}

inline bool PassManagerTraits<BasicBlock>::doInitialization(Function &F) {
  bool Changed = false;
  for (unsigned i = 0, e = ((PMType*)this)->Passes.size(); i != e; ++i)
    ((PMType*)this)->Passes[i]->doInitialization(F);
  return Changed;
}

inline bool PassManagerTraits<BasicBlock>::runOnBasicBlock(BasicBlock &BB) {
  return ((PMType*)this)->runOnUnit(&BB);
}

inline bool PassManagerTraits<BasicBlock>::doFinalization(Function &F) {
  bool Changed = false;
  for (unsigned i = 0, e = ((PMType*)this)->Passes.size(); i != e; ++i)
    ((PMType*)this)->Passes[i]->doFinalization(F);
  return Changed;
}

inline bool PassManagerTraits<BasicBlock>::doFinalization(Module &M) {
  bool Changed = false;
  for (unsigned i = 0, e = ((PMType*)this)->Passes.size(); i != e; ++i)
    ((PMType*)this)->Passes[i]->doFinalization(M);
  return Changed;
}


// PassManagerTraits<Function> Implementations
//
inline bool PassManagerTraits<Function>::doInitialization(Module &M) {
  bool Changed = false;
  for (unsigned i = 0, e = ((PMType*)this)->Passes.size(); i != e; ++i)
    ((PMType*)this)->Passes[i]->doInitialization(M);
  return Changed;
}

inline bool PassManagerTraits<Function>::runOnFunction(Function &F) {
  return ((PMType*)this)->runOnUnit(&F);
}

inline bool PassManagerTraits<Function>::doFinalization(Module &M) {
  bool Changed = false;
  for (unsigned i = 0, e = ((PMType*)this)->Passes.size(); i != e; ++i)
    ((PMType*)this)->Passes[i]->doFinalization(M);
  return Changed;
}

#endif
