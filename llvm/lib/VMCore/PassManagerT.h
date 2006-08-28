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
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/LeakDetector.h"
#include "llvm/Support/Timer.h"
#include <algorithm>
#include <iostream>

namespace llvm {

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
                         clEnumValEnd));

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
  static void PrintPassInformation(unsigned,const char*,Pass *, Module *);
  static void PrintPassInformation(unsigned,const char*,Pass *, Function *);
  static void PrintPassInformation(unsigned,const char*,Pass *, BasicBlock *);
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

struct BBTraits {
  typedef BasicBlock UnitType;
  
  // PassClass - The type of passes tracked by this PassManager
  typedef BasicBlockPass PassClass;

  // SubPassClass - The types of classes that should be collated together
  // This is impossible to match, so BasicBlock instantiations of PassManagerT
  // do not collate.
  //
  typedef BasicBlockPassManager SubPassClass;

  // BatcherClass - The type to use for collation of subtypes... This class is
  // never instantiated for the BasicBlockPassManager, but it must be an
  // instance of PassClass to typecheck.
  //
  typedef PassClass BatcherClass;

  // ParentClass - The type of the parent PassManager...
  typedef FunctionPassManagerT ParentClass;

  // PMType - The type of this passmanager
  typedef BasicBlockPassManager PMType;
};

struct FTraits {
  typedef Function UnitType;
  
  // PassClass - The type of passes tracked by this PassManager
  typedef FunctionPass PassClass;

  // SubPassClass - The types of classes that should be collated together
  typedef BasicBlockPass SubPassClass;

  // BatcherClass - The type to use for collation of subtypes...
  typedef BasicBlockPassManager BatcherClass;

  // ParentClass - The type of the parent PassManager...
  typedef ModulePassManager ParentClass;

  // PMType - The type of this passmanager
  typedef FunctionPassManagerT PMType;
};

struct MTraits {
  typedef Module UnitType;
  
  // PassClass - The type of passes tracked by this PassManager
  typedef ModulePass PassClass;

  // SubPassClass - The types of classes that should be collated together
  typedef FunctionPass SubPassClass;

  // BatcherClass - The type to use for collation of subtypes...
  typedef FunctionPassManagerT BatcherClass;

  // ParentClass - The type of the parent PassManager...
  typedef AnalysisResolver ParentClass;
  
  // PMType - The type of this passmanager
  typedef ModulePassManager PMType;
};


//===----------------------------------------------------------------------===//
// PassManagerT - Container object for passes.  The PassManagerT destructor
// deletes all passes contained inside of the PassManagerT, so you shouldn't
// delete passes manually, and all passes should be dynamically allocated.
//
template<typename Trait> class PassManagerT : public AnalysisResolver {
  
  typedef typename Trait::PassClass    PassClass;
  typedef typename Trait::UnitType     UnitType;
  typedef typename Trait::ParentClass  ParentClass;
  typedef typename Trait::SubPassClass SubPassClass;
  typedef typename Trait::BatcherClass BatcherClass;
  typedef typename Trait::PMType       PMType;
  
  friend class ModulePass;
  friend class FunctionPass;
  friend class BasicBlockPass;
  
  friend class ImmutablePass;
  
  friend class BasicBlockPassManager;
  friend class FunctionPassManagerT;
  friend class ModulePassManager;

  std::vector<PassClass*> Passes;               // List of passes to run
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
  
  // getPMName() - Return the name of the unit the PassManager operates on for
  // debugging.
  virtual const char *getPMName() const =0;
  
  virtual const char *getPassName() const =0;

  virtual bool runPass(PassClass *P, UnitType *M) =0;
  
  // TODO:Figure out what pure virtuals remain.
  
  
  PassManagerT(ParentClass *Par = 0) : Parent(Par), Batcher(0) {}
  virtual ~PassManagerT() {
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
    closeBatcher();
    CurrentAnalyses.clear();

    TimingInfo::createTheTimeInfo();

    addImmutablePasses();

    // LastUserOf - This contains the inverted LastUseOfMap...
    std::map<Pass *, std::vector<Pass*> > LastUserOf;
    for (std::map<Pass*, Pass*>::iterator I = LastUseOf.begin(),
                                          E = LastUseOf.end(); I != E; ++I)
      LastUserOf[I->second].push_back(I->first);

    // Output debug information...
    assert(dynamic_cast<PassClass*>(this) && 
           "It wasn't the PassClass I thought it was");
    if (Parent == 0) 
      PMDebug::PerformPassStartupStuff((dynamic_cast<PMType*>(this)));

    return runPasses(M, LastUserOf);
  }

  // dumpPassStructure - Implement the -debug-passes=PassStructure option
  inline void dumpPassStructure(unsigned Offset = 0) {
    // Print out the immutable passes...
    
    for (unsigned i = 0, e = ImmutablePasses.size(); i != e; ++i)
      ImmutablePasses[i]->dumpPassStructure(0);

    std::cerr << std::string(Offset*2, ' ') << this->getPMName()
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

      // Prolong live range of analyses that are needed after an analysis pass
      // is destroyed, for querying by subsequent passes
      AnalysisUsage AnUsage;
      I->second->getAnalysisUsage(AnUsage);
      const std::vector<AnalysisID> &IDs = AnUsage.getRequiredTransitiveSet();
      for (std::vector<AnalysisID>::const_iterator i = IDs.begin(),
             e = IDs.end(); i != e; ++i)
        markPassUsed(*i, User);

    } else {
      // Pass not in current available set, must be a higher level pass
      // available to us, propagate to parent pass manager...  We tell the
      // parent that we (the passmanager) are using the analysis so that it
      // frees the analysis AFTER this pass manager runs.
      //
      if (Parent) {
        assert(dynamic_cast<Pass*>(this) && 
               "It wasn't the Pass type I thought it was.");
        Parent->markPassUsed(P, dynamic_cast<Pass*>(this));
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
    
    addRequiredPasses(AnUsage.getRequiredSet());
    
    // Tell the pass to add itself to this PassManager... the way it does so
    // depends on the class of the pass, and is critical to laying out passes in
    // an optimal order..
    //
    assert(dynamic_cast<PMType*>(this) && 
        "It wasn't the right passmanager type.");
    P->addToPassManager(static_cast<PMType*>(this), AnUsage);
  }

  // add - H4x0r an ImmutablePass into a PassManager that might not be
  // expecting one.
  //
  void add(ImmutablePass *P) {
    // Get information about what analyses the pass uses...
    AnalysisUsage AnUsage;
    P->getAnalysisUsage(AnUsage);
    
    addRequiredPasses(AnUsage.getRequiredSet());
    
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
    // batcher class then we can reorder the pass to execute before the batcher
    // does, which will potentially allow us to batch more passes!
    //
    if (Batcher)
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

    removeNonPreservedAnalyses(AnUsage);
    
    makeCurrentlyAvailable(P);
    
    // For now assume that our results are never used...
    LastUseOf[P] = P;
  }
  
  // For FunctionPass subclasses, we must be sure to batch the FunctionPass's
  // together in a BatcherClass object so that all of the analyses are run
  // together a function at a time.
  //
  void addPass(SubPassClass *MP, AnalysisUsage &AnUsage) {

    if (Batcher == 0) { // If we don't have a batcher yet, make one now.
      assert(dynamic_cast<PMType*>(this) && 
             "It wasn't the PassManager type I thought it was");
      Batcher = new BatcherClass((static_cast<PMType*>(this)));
    }

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

  void addRequiredPasses(const std::vector<AnalysisID> &Required) {
    for (std::vector<AnalysisID>::const_iterator I = Required.begin(),
         E = Required.end(); I != E; ++I) {
      if (getAnalysisOrNullDown(*I) == 0) {
        Pass *AP = (*I)->createPass();
        if (ImmutablePass *IP = dynamic_cast<ImmutablePass *> (AP)) add(IP);
        else if (PassClass *RP = dynamic_cast<PassClass *> (AP)) add(RP);
        else assert (0 && "Wrong kind of pass for this PassManager");
      }
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
private:
  
  // Add any immutable passes to the CurrentAnalyses set...
  inline void addImmutablePasses() { 
    for (unsigned i = 0, e = ImmutablePasses.size(); i != e; ++i) {
      ImmutablePass *IPass = ImmutablePasses[i];
      if (const PassInfo *PI = IPass->getPassInfo()) {
        CurrentAnalyses[PI] = IPass;

        const std::vector<const PassInfo*> &II = PI->getInterfacesImplemented();
        for (unsigned i = 0, e = II.size(); i != e; ++i)
          CurrentAnalyses[II[i]] = IPass;
      }
    }
  }
  
  // Run all of the passes
  inline bool runPasses(UnitType *M, 
                 std::map<Pass *, std::vector<Pass*> > &LastUserOf) { 
    bool MadeChanges = false;
    
    for (unsigned i = 0, e = Passes.size(); i < e; ++i) {
      PassClass *P = Passes[i];

      PMDebug::PrintPassInformation(getDepth(), "Executing Pass", P, M);

      // Get information about what analyses the pass uses...
      AnalysisUsage AnUsage;
      P->getAnalysisUsage(AnUsage);
      PMDebug::PrintAnalysisSetInfo(getDepth(), "Required", P,
                                    AnUsage.getRequiredSet());
      
      initialiseAnalysisImpl(P, AnUsage);
      
      // Run the sub pass!
      if (TheTimeInfo) TheTimeInfo->passStarted(P);
      bool Changed = runPass(P, M);
      if (TheTimeInfo) TheTimeInfo->passEnded(P);
      MadeChanges |= Changed;

      // Check for memory leaks by the pass...
      LeakDetector::checkForGarbage(std::string("after running pass '") +
                                    P->getPassName() + "'");

      if (Changed)
        PMDebug::PrintPassInformation(getDepth()+1, "Made Modification", P, M);
      PMDebug::PrintAnalysisSetInfo(getDepth(), "Preserved", P,
                                    AnUsage.getPreservedSet());
      
      // Erase all analyses not in the preserved set
      removeNonPreservedAnalyses(AnUsage);
      
      makeCurrentlyAvailable(P);
      
      // free memory and remove dead passes from the CurrentAnalyses list...
      removeDeadPasses(P, M, LastUserOf);
    }
    
    return MadeChanges;
  }
  
  // All Required analyses should be available to the pass as it runs!  Here
  // we fill in the AnalysisImpls member of the pass so that it can
  // successfully use the getAnalysis() method to retrieve the
  // implementations it needs.
  //
  inline void initialiseAnalysisImpl(PassClass *P, AnalysisUsage &AnUsage) { 
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
  }
  
  inline void removeNonPreservedAnalyses(AnalysisUsage &AnUsage) { 
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
  }
  
  inline void removeDeadPasses(Pass* P, UnitType *M, 
              std::map<Pass *, std::vector<Pass*> > &LastUserOf) { 
    std::vector<Pass*> &DeadPass = LastUserOf[P];
    for (std::vector<Pass*>::iterator I = DeadPass.begin(),E = DeadPass.end();
          I != E; ++I) {
      PMDebug::PrintPassInformation(getDepth()+1, "Freeing Pass", *I, M);
      (*I)->releaseMemory();
    }
    
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
  
  inline void makeCurrentlyAvailable(Pass* P) { 
    if (const PassInfo *PI = P->getPassInfo()) {
      CurrentAnalyses[PI] = P;

      // This pass is the current implementation of all of the interfaces it
      // implements as well.
      //
      const std::vector<const PassInfo*> &II = PI->getInterfacesImplemented();
      for (unsigned i = 0, e = II.size(); i != e; ++i)
        CurrentAnalyses[II[i]] = P;
    }
  }
};



//===----------------------------------------------------------------------===//
// BasicBlockPassManager
//
// This pass manager is used to group together all of the BasicBlockPass's
// into a single unit.
//
class BasicBlockPassManager : public BasicBlockPass, 
                              public BBTraits, 
                              public PassManagerT<BBTraits> {
public:
  BasicBlockPassManager(BBTraits::ParentClass* PC) : 
    PassManagerT<BBTraits>(PC) {
  }
  
  BasicBlockPassManager(BasicBlockPassManager* BBPM) : 
    PassManagerT<BBTraits>(BBPM->Parent) {
  }
  
  virtual bool runPass(Module &M) { return false; }

  virtual bool runPass(BasicBlock &BB) { return BasicBlockPass::runPass(BB); }

  // runPass - Specify how the pass should be run on the UnitType
  virtual bool runPass(BBTraits::PassClass *P, BasicBlock *M) {
    // TODO: init and finalize
    return P->runOnBasicBlock(*M);
  }
  
  virtual ~BasicBlockPassManager() {}
  
  virtual void dumpPassStructure(unsigned Offset = 0) { 
    PassManagerT<BBTraits>::dumpPassStructure(Offset);
  }
  
  // getPMName() - Return the name of the unit the PassManager operates on for
  // debugging.
  virtual const char *getPMName() const { return "BasicBlock"; }
  
  virtual const char *getPassName() const { return "BasicBlock Pass Manager"; }
  
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
// FunctionPassManager
//
// This pass manager is used to group together all of the FunctionPass's
// into a single unit.
//
class FunctionPassManagerT : public FunctionPass, 
                             public FTraits, 
                             public PassManagerT<FTraits> {
public:
  FunctionPassManagerT() : PassManagerT<FTraits>(0) {}
  
  // Parent constructor
  FunctionPassManagerT(FTraits::ParentClass* PC) : PassManagerT<FTraits>(PC) {}
  
  FunctionPassManagerT(FunctionPassManagerT* FPM) : 
    PassManagerT<FTraits>(FPM->Parent) {
  }
  
  virtual ~FunctionPassManagerT() {}
  
  virtual void dumpPassStructure(unsigned Offset = 0) { 
    PassManagerT<FTraits>::dumpPassStructure(Offset);
  }
  
  // getPMName() - Return the name of the unit the PassManager operates on for
  // debugging.
  virtual const char *getPMName() const { return "Function"; }
  
  virtual const char *getPassName() const { return "Function Pass Manager"; }
  
  virtual bool runOnFunction(Function &F);
  
  virtual bool doInitialization(Module &M);
  
  virtual bool doFinalization(Module &M);
  
  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
  }
  
  virtual bool runPass(Module &M) { return FunctionPass::runPass(M); }
  virtual bool runPass(BasicBlock &BB) { return FunctionPass::runPass(BB); }

  // runPass - Specify how the pass should be run on the UnitType
  virtual bool runPass(FTraits::PassClass *P, Function *F) {
    return P->runOnFunction(*F);
  }
};


//===----------------------------------------------------------------------===//
// ModulePassManager
//
// This is the top level PassManager implementation that holds generic passes.
//
class ModulePassManager : public ModulePass, 
                          public MTraits, 
                          public PassManagerT<MTraits> {
public:
  ModulePassManager() : PassManagerT<MTraits>(0) {}
  
  // Batcher Constructor
  ModulePassManager(MTraits::ParentClass* PC) : PassManagerT<MTraits>(PC) {}
  
  ModulePassManager(ModulePassManager* MPM) : 
    PassManagerT<MTraits>((MPM->Parent)) {
  }
  
  virtual ~ModulePassManager() {}
  
  virtual void dumpPassStructure(unsigned Offset = 0) { 
    PassManagerT<MTraits>::dumpPassStructure(Offset);
  }
  
  // getPMName() - Return the name of the unit the PassManager operates on for
  // debugging.
  virtual const char *getPassName() const { return "Module Pass Manager"; }
  
  // getPMName() - Return the name of the unit the PassManager operates on for
  // debugging.
  virtual const char *getPMName() const { return "Module"; }
  
  // runOnModule - Implement the PassManager interface.
  virtual bool runOnModule(Module &M);

  virtual bool runPass(Module &M) { return ModulePass::runPass(M); }
  virtual bool runPass(BasicBlock &BB) { return ModulePass::runPass(BB); }

  // runPass - Specify how the pass should be run on the UnitType
  virtual bool runPass(MTraits::PassClass *P, Module *M) {
    return P->runOnModule(*M);
  }
};

//===----------------------------------------------------------------------===//
// PassManager Method Implementations
//

// BasicBlockPassManager Implementations
//

inline bool BasicBlockPassManager::runOnBasicBlock(BasicBlock &BB) {
  return ((BBTraits::PMType*)this)->runOnUnit(&BB);
}

inline bool BasicBlockPassManager::doInitialization(Module &M) {
  bool Changed = false;
  for (unsigned i = 0, e =((BBTraits::PMType*)this)->Passes.size(); i != e; ++i)
    ((BBTraits::PMType*)this)->Passes[i]->doInitialization(M);
  return Changed;
}

inline bool BasicBlockPassManager::doInitialization(Function &F) {
  bool Changed = false;
  for (unsigned i = 0, e =((BBTraits::PMType*)this)->Passes.size(); i != e; ++i)
    ((BBTraits::PMType*)this)->Passes[i]->doInitialization(F);
  return Changed;
}

inline bool BasicBlockPassManager::doFinalization(Function &F) {
  bool Changed = false;
  for (unsigned i = 0, e =((BBTraits::PMType*)this)->Passes.size(); i != e; ++i)
    ((BBTraits::PMType*)this)->Passes[i]->doFinalization(F);
  return Changed;
}

inline bool BasicBlockPassManager::doFinalization(Module &M) {
  bool Changed = false;
  for (unsigned i=0, e = ((BBTraits::PMType*)this)->Passes.size(); i != e; ++i)
    ((BBTraits::PMType*)this)->Passes[i]->doFinalization(M);
  return Changed;
}

// FunctionPassManagerT Implementations
//

inline bool FunctionPassManagerT::runOnFunction(Function &F) {
  return ((FTraits::PMType*)this)->runOnUnit(&F);
}

inline bool FunctionPassManagerT::doInitialization(Module &M) {
  bool Changed = false;
  for (unsigned i=0, e = ((FTraits::PMType*)this)->Passes.size(); i != e; ++i)
    ((FTraits::PMType*)this)->Passes[i]->doInitialization(M);
  return Changed;
}

inline bool FunctionPassManagerT::doFinalization(Module &M) {
  bool Changed = false;
  for (unsigned i=0, e = ((FTraits::PMType*)this)->Passes.size(); i != e; ++i)
    ((FTraits::PMType*)this)->Passes[i]->doFinalization(M);
  return Changed;
}

// ModulePassManager Implementations
//

bool ModulePassManager::runOnModule(Module &M) {
  return ((PassManagerT<MTraits>*)this)->runOnUnit(&M);
}

} // End llvm namespace

#endif
