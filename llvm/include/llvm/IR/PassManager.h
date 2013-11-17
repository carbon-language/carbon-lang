//===- PassManager.h - Pass management infrastructure -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This header defines various interfaces for pass management in LLVM. There
/// is no "pass" interface in LLVM per se. Instead, an instance of any class
/// which supports a method to 'run' it over a unit of IR can be used as
/// a pass. A pass manager is generally a tool to collect a sequence of passes
/// which run over a particular IR construct, and run each of them in sequence
/// over each such construct in the containing IR construct. As there is no
/// containing IR construct for a Module, a manager for passes over modules
/// forms the base case which runs its managed passes in sequence over the
/// single module provided.
///
/// The core IR library provides managers for running passes over
/// modules and functions.
///
/// * FunctionPassManager can run over a Module, runs each pass over
///   a Function.
/// * ModulePassManager must be directly run, runs each pass over the Module.
///
/// Note that the implementations of the pass managers use concept-based
/// polymorphism as outlined in the "Value Semantics and Concept-based
/// Polymorphism" talk (or its abbreviated sibling "Inheritance Is The Base
/// Class of Evil") by Sean Parent:
/// * http://github.com/sean-parent/sean-parent.github.com/wiki/Papers-and-Presentations
/// * http://www.youtube.com/watch?v=_BpMYeUFXv8
/// * http://channel9.msdn.com/Events/GoingNative/2013/Inheritance-Is-The-Base-Class-of-Evil
///
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/polymorphic_ptr.h"
#include "llvm/Support/type_traits.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include <list>
#include <vector>

namespace llvm {

class Module;
class Function;

/// \brief Implementation details of the pass manager interfaces.
namespace detail {

/// \brief Template for the abstract base class used to dispatch
/// polymorphically over pass objects.
template <typename T> struct PassConcept {
  // Boiler plate necessary for the container of derived classes.
  virtual ~PassConcept() {}
  virtual PassConcept *clone() = 0;

  /// \brief The polymorphic API which runs the pass over a given IR entity.
  virtual bool run(T Arg) = 0;
};

/// \brief A template wrapper used to implement the polymorphic API.
///
/// Can be instantiated for any object which provides a \c run method
/// accepting a \c T. It requires the pass to be a copyable
/// object.
template <typename T, typename PassT> struct PassModel : PassConcept<T> {
  PassModel(PassT Pass) : Pass(llvm_move(Pass)) {}
  virtual PassModel *clone() { return new PassModel(Pass); }
  virtual bool run(T Arg) { return Pass.run(Arg); }
  PassT Pass;
};

}

class AnalysisManager;

class ModulePassManager {
public:
  ModulePassManager(Module *M, AnalysisManager *AM = 0) : M(M), AM(AM) {}

  template <typename ModulePassT> void addPass(ModulePassT Pass) {
    Passes.push_back(new ModulePassModel<ModulePassT>(llvm_move(Pass)));
  }

  void run();

private:
  // Pull in the concept type and model template specialized for modules.
  typedef detail::PassConcept<Module *> ModulePassConcept;
  template <typename PassT>
  struct ModulePassModel : detail::PassModel<Module *, PassT> {
    ModulePassModel(PassT Pass) : detail::PassModel<Module *, PassT>(Pass) {}
  };

  Module *M;
  AnalysisManager *AM;
  std::vector<polymorphic_ptr<ModulePassConcept> > Passes;
};

class FunctionPassManager {
public:
  FunctionPassManager(AnalysisManager *AM = 0) : AM(AM) {}

  template <typename FunctionPassT> void addPass(FunctionPassT Pass) {
    Passes.push_back(new FunctionPassModel<FunctionPassT>(llvm_move(Pass)));
  }

  bool run(Module *M);

private:
  // Pull in the concept type and model template specialized for functions.
  typedef detail::PassConcept<Function *> FunctionPassConcept;
  template <typename PassT>
  struct FunctionPassModel : detail::PassModel<Function *, PassT> {
    FunctionPassModel(PassT Pass)
        : detail::PassModel<Function *, PassT>(Pass) {}
  };

  AnalysisManager *AM;
  std::vector<polymorphic_ptr<FunctionPassConcept> > Passes;
};


/// \brief An analysis manager to coordinate and cache analyses run over
/// a module.
///
/// The analysis manager is typically used by passes in a pass pipeline
/// (consisting potentially of several individual pass managers) over a module
/// of IR. It provides registration of available analyses, declaring
/// requirements on support for specific analyses, running of an specific
/// analysis over a specific unit of IR to compute an analysis result, and
/// caching of the analysis results to reuse them across multiple passes.
///
/// It is the responsibility of callers to use the invalidation API to
/// invalidate analysis results when the IR they correspond to changes. The
/// \c ModulePassManager and \c FunctionPassManager do this automatically.
class AnalysisManager {
public:
  AnalysisManager(Module *M) : M(M) {}

  /// \brief Get the result of an analysis pass for this module.
  ///
  /// If there is not a valid cached result in the manager already, this will
  /// re-run the analysis to produce a valid result.
  ///
  /// The module passed in must be the same module as the analysis manager was
  /// constructed around.
  template <typename PassT>
  const typename PassT::Result &getResult(Module *M) {
    assert(ModuleAnalysisPasses.count(PassT::ID()) &&
           "This analysis pass was not registered prior to being queried");

    const AnalysisResultConcept<Module> &ResultConcept =
        getResultImpl(PassT::ID(), M);
    typedef AnalysisResultModel<Module, typename PassT::Result> ResultModelT;
    return static_cast<const ResultModelT &>(ResultConcept).Result;
  }

  /// \brief Get the result of an analysis pass for a function.
  ///
  /// If there is not a valid cached result in the manager already, this will
  /// re-run the analysis to produce a valid result.
  template <typename PassT>
  const typename PassT::Result &getResult(Function *F) {
    assert(FunctionAnalysisPasses.count(PassT::ID()) &&
           "This analysis pass was not registered prior to being queried");

    const AnalysisResultConcept<Function> &ResultConcept =
        getResultImpl(PassT::ID(), F);
    typedef AnalysisResultModel<Function, typename PassT::Result> ResultModelT;
    return static_cast<const ResultModelT &>(ResultConcept).Result;
  }

  /// \brief Register an analysis pass with the manager.
  ///
  /// This provides an initialized and set-up analysis pass to the
  /// analysis
  /// manager. Whomever is setting up analysis passes must use this to
  /// populate
  /// the manager with all of the analysis passes available.
  template <typename PassT> void registerAnalysisPass(PassT Pass) {
    registerAnalysisPassImpl<PassT>(llvm_move(Pass));
  }

  /// \brief Invalidate a specific analysis pass for an IR module.
  ///
  /// Note that the analysis result can disregard invalidation.
  template <typename PassT> void invalidate(Module *M) {
    invalidateImpl(PassT::ID(), M);
  }

  /// \brief Invalidate a specific analysis pass for an IR function.
  ///
  /// Note that the analysis result can disregard invalidation.
  template <typename PassT> void invalidate(Function *F) {
    invalidateImpl(PassT::ID(), F);
  }

  /// \brief Invalidate analyses cached for an IR Module.
  ///
  /// Note that specific analysis results can disregard invalidation by
  /// overriding their invalidate method.
  ///
  /// The module must be the module this analysis manager was constructed
  /// around.
  void invalidateAll(Module *M);

  /// \brief Invalidate analyses cached for an IR Function.
  ///
  /// Note that specific analysis results can disregard invalidation by
  /// overriding the invalidate method.
  void invalidateAll(Function *F);

private:
  /// \brief Abstract concept of an analysis result.
  ///
  /// This concept is parameterized over the IR unit that this result pertains
  /// to.
  template <typename IRUnitT> struct AnalysisResultConcept {
    virtual ~AnalysisResultConcept() {}
    virtual AnalysisResultConcept *clone() = 0;

    /// \brief Method to try and mark a result as invalid.
    ///
    /// When the outer \c AnalysisManager detects a change in some underlying
    /// unit of the IR, it will call this method on all of the results cached.
    ///
    /// \returns true if the result should indeed be invalidated (the default).
    virtual bool invalidate(IRUnitT *IR) = 0;
  };

  /// \brief Wrapper to model the analysis result concept.
  ///
  /// Can wrap any type which implements a suitable invalidate member and model
  /// the AnalysisResultConcept for the AnalysisManager.
  template <typename IRUnitT, typename ResultT>
  struct AnalysisResultModel : AnalysisResultConcept<IRUnitT> {
    AnalysisResultModel(ResultT Result) : Result(llvm_move(Result)) {}
    virtual AnalysisResultModel *clone() {
      return new AnalysisResultModel(Result);
    }

    /// \brief The model delegates to the \c ResultT method.
    virtual bool invalidate(IRUnitT *IR) { return Result.invalidate(IR); }

    ResultT Result;
  };

  /// \brief Abstract concept of an analysis pass.
  ///
  /// This concept is parameterized over the IR unit that it can run over and
  /// produce an analysis result.
  template <typename IRUnitT> struct AnalysisPassConcept {
    virtual ~AnalysisPassConcept() {}
    virtual AnalysisPassConcept *clone() = 0;

    /// \brief Method to run this analysis over a unit of IR.
    /// \returns The analysis result object to be queried by users, the caller
    /// takes ownership.
    virtual AnalysisResultConcept<IRUnitT> *run(IRUnitT *IR) = 0;
  };

  /// \brief Wrapper to model the analysis pass concept.
  ///
  /// Can wrap any type which implements a suitable \c run method. The method
  /// must accept the IRUnitT as an argument and produce an object which can be
  /// wrapped in a \c AnalysisResultModel.
  template <typename PassT>
  struct AnalysisPassModel : AnalysisPassConcept<typename PassT::IRUnitT> {
    AnalysisPassModel(PassT Pass) : Pass(llvm_move(Pass)) {}
    virtual AnalysisPassModel *clone() { return new AnalysisPassModel(Pass); }

    // FIXME: Replace PassT::IRUnitT with type traits when we use C++11.
    typedef typename PassT::IRUnitT IRUnitT;

    // FIXME: Replace PassT::Result with type traits when we use C++11.
    typedef AnalysisResultModel<IRUnitT, typename PassT::Result> ResultModelT;

    /// \brief The model delegates to the \c PassT::run method.
    ///
    /// The return is wrapped in an \c AnalysisResultModel.
    virtual ResultModelT *run(IRUnitT *IR) {
      return new ResultModelT(Pass.run(IR));
    }

    PassT Pass;
  };


  /// \brief Get a module pass result, running the pass if necessary.
  const AnalysisResultConcept<Module> &getResultImpl(void *PassID, Module *M);

  /// \brief Get a function pass result, running the pass if necessary.
  const AnalysisResultConcept<Function> &getResultImpl(void *PassID,
                                                       Function *F);

  /// \brief Invalidate a module pass result.
  void invalidateImpl(void *PassID, Module *M);

  /// \brief Invalidate a function pass result.
  void invalidateImpl(void *PassID, Function *F);


  /// \brief Module pass specific implementation of registration.
  template <typename PassT>
  typename enable_if<is_same<typename PassT::IRUnitT, Module> >::type
  registerAnalysisPassImpl(PassT Pass) {
    assert(!ModuleAnalysisPasses.count(PassT::ID()) &&
           "Registered the same analysis pass twice!");
    ModuleAnalysisPasses[PassT::ID()] =
        new AnalysisPassModel<PassT>(llvm_move(Pass));
  }

  /// \brief Function pass specific implementation of registration.
  template <typename PassT>
  typename enable_if<is_same<typename PassT::IRUnitT, Function> >::type
  registerAnalysisPassImpl(PassT Pass) {
    assert(!FunctionAnalysisPasses.count(PassT::ID()) &&
           "Registered the same analysis pass twice!");
    FunctionAnalysisPasses[PassT::ID()] =
        new AnalysisPassModel<PassT>(llvm_move(Pass));
  }


  /// \brief Map type from module analysis pass ID to pass concept pointer.
  typedef DenseMap<void *, polymorphic_ptr<AnalysisPassConcept<Module> > >
  ModuleAnalysisPassMapT;

  /// \brief Collection of module analysis passes, indexed by ID.
  ModuleAnalysisPassMapT ModuleAnalysisPasses;

  /// \brief Map type from module analysis pass ID to pass result concept pointer.
  typedef DenseMap<void *, polymorphic_ptr<AnalysisResultConcept<Module> > >
  ModuleAnalysisResultMapT;

  /// \brief Cache of computed module analysis results for this module.
  ModuleAnalysisResultMapT ModuleAnalysisResults;


  /// \brief Map type from function analysis pass ID to pass concept pointer.
  typedef DenseMap<void *, polymorphic_ptr<AnalysisPassConcept<Function> > >
  FunctionAnalysisPassMapT;

  /// \brief Collection of function analysis passes, indexed by ID.
  FunctionAnalysisPassMapT FunctionAnalysisPasses;

  /// \brief List of function analysis pass IDs and associated concept pointers.
  ///
  /// Requires iterators to be valid across appending new entries and arbitrary
  /// erases. Provides both the pass ID and concept pointer such that it is
  /// half of a bijection and provides storage for the actual result concept.
  typedef std::list<
      std::pair<void *, polymorphic_ptr<AnalysisResultConcept<Function> > > >
  FunctionAnalysisResultListT;

  /// \brief Map type from function pointer to our custom list type.
  typedef DenseMap<Function *, FunctionAnalysisResultListT> FunctionAnalysisResultListMapT;

  /// \brief Map from function to a list of function analysis results.
  ///
  /// Provides linear time removal of all analysis results for a function and
  /// the ultimate storage for a particular cached analysis result.
  FunctionAnalysisResultListMapT FunctionAnalysisResultLists;

  /// \brief Map type from a pair of analysis ID and function pointer to an
  /// iterator into a particular result list.
  typedef DenseMap<std::pair<void *, Function *>,
                   FunctionAnalysisResultListT::iterator>
  FunctionAnalysisResultMapT;

  /// \brief Map from an analysis ID and function to a particular cached
  /// analysis result.
  FunctionAnalysisResultMapT FunctionAnalysisResults;

  /// \brief Module handle for the \c AnalysisManager.
  Module *M;
};

}
