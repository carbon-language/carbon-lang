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

#ifndef LLVM_IR_PASS_MANAGER_H
#define LLVM_IR_PASS_MANAGER_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/type_traits.h"
#include <list>
#include <memory>
#include <vector>

namespace llvm {

class Module;
class Function;

/// \brief An abstract set of preserved analyses following a transformation pass
/// run.
///
/// When a transformation pass is run, it can return a set of analyses whose
/// results were preserved by that transformation. The default set is "none",
/// and preserving analyses must be done explicitly.
///
/// There is also an explicit all state which can be used (for example) when
/// the IR is not mutated at all.
class PreservedAnalyses {
public:
  // We have to explicitly define all the special member functions because MSVC
  // refuses to generate them.
  PreservedAnalyses() {}
  PreservedAnalyses(const PreservedAnalyses &Arg)
      : PreservedPassIDs(Arg.PreservedPassIDs) {}
  PreservedAnalyses(PreservedAnalyses &&Arg)
      : PreservedPassIDs(Arg.PreservedPassIDs) {}
  PreservedAnalyses &operator=(PreservedAnalyses RHS) {
    std::swap(*this, RHS);
    return *this;
  }

  /// \brief Convenience factory function for the empty preserved set.
  static PreservedAnalyses none() { return PreservedAnalyses(); }

  /// \brief Construct a special preserved set that preserves all passes.
  static PreservedAnalyses all() {
    PreservedAnalyses PA;
    PA.PreservedPassIDs.insert((void *)AllPassesID);
    return PA;
  }

  /// \brief Mark a particular pass as preserved, adding it to the set.
  template <typename PassT> void preserve() {
    if (!areAllPreserved())
      PreservedPassIDs.insert(PassT::ID());
  }

  /// \brief Intersect this set with another in place.
  ///
  /// This is a mutating operation on this preserved set, removing all
  /// preserved passes which are not also preserved in the argument.
  void intersect(const PreservedAnalyses &Arg) {
    if (Arg.areAllPreserved())
      return;
    if (areAllPreserved()) {
      PreservedPassIDs = Arg.PreservedPassIDs;
      return;
    }
    for (SmallPtrSet<void *, 2>::const_iterator I = PreservedPassIDs.begin(),
                                                E = PreservedPassIDs.end();
         I != E; ++I)
      if (!Arg.PreservedPassIDs.count(*I))
        PreservedPassIDs.erase(*I);
  }

  /// \brief Intersect this set with a temporary other set in place.
  ///
  /// This is a mutating operation on this preserved set, removing all
  /// preserved passes which are not also preserved in the argument.
  void intersect(PreservedAnalyses &&Arg) {
    if (Arg.areAllPreserved())
      return;
    if (areAllPreserved()) {
      PreservedPassIDs = std::move(Arg.PreservedPassIDs);
      return;
    }
    for (SmallPtrSet<void *, 2>::const_iterator I = PreservedPassIDs.begin(),
                                                E = PreservedPassIDs.end();
         I != E; ++I)
      if (!Arg.PreservedPassIDs.count(*I))
        PreservedPassIDs.erase(*I);
  }

  /// \brief Query whether a pass is marked as preserved by this set.
  template <typename PassT> bool preserved() const {
    return preserved(PassT::ID());
  }

  /// \brief Query whether an abstract pass ID is marked as preserved by this
  /// set.
  bool preserved(void *PassID) const {
    return PreservedPassIDs.count((void *)AllPassesID) ||
           PreservedPassIDs.count(PassID);
  }

private:
  // Note that this must not be -1 or -2 as those are already used by the
  // SmallPtrSet.
  static const uintptr_t AllPassesID = (intptr_t)-3;

  bool areAllPreserved() const { return PreservedPassIDs.count((void *)AllPassesID); }

  SmallPtrSet<void *, 2> PreservedPassIDs;
};

/// \brief Implementation details of the pass manager interfaces.
namespace detail {

/// \brief Template for the abstract base class used to dispatch
/// polymorphically over pass objects.
template <typename IRUnitT, typename AnalysisManagerT> struct PassConcept {
  // Boiler plate necessary for the container of derived classes.
  virtual ~PassConcept() {}

  /// \brief The polymorphic API which runs the pass over a given IR entity.
  ///
  /// Note that actual pass object can omit the analysis manager argument if
  /// desired. Also that the analysis manager may be null if there is no
  /// analysis manager in the pass pipeline.
  virtual PreservedAnalyses run(IRUnitT IR, AnalysisManagerT *AM) = 0;

  /// \brief Polymorphic method to access the name of a pass.
  virtual StringRef name() = 0;
};

/// \brief SFINAE metafunction for computing whether \c PassT has a run method
/// accepting an \c AnalysisManagerT.
template <typename IRUnitT, typename AnalysisManagerT, typename PassT,
          typename ResultT>
class PassRunAcceptsAnalysisManager {
  typedef char SmallType;
  struct BigType { char a, b; };

  template <typename T, ResultT (T::*)(IRUnitT, AnalysisManagerT *)>
  struct Checker;

  template <typename T> static SmallType f(Checker<T, &T::run> *);
  template <typename T> static BigType f(...);

public:
  enum { Value = sizeof(f<PassT>(0)) == sizeof(SmallType) };
};

/// \brief A template wrapper used to implement the polymorphic API.
///
/// Can be instantiated for any object which provides a \c run method accepting
/// an \c IRUnitT. It requires the pass to be a copyable object. When the
/// \c run method also accepts an \c AnalysisManagerT*, we pass it along.
template <typename IRUnitT, typename AnalysisManagerT, typename PassT,
          bool AcceptsAnalysisManager = PassRunAcceptsAnalysisManager<
              IRUnitT, AnalysisManagerT, PassT, PreservedAnalyses>::Value>
struct PassModel;

/// \brief Specialization of \c PassModel for passes that accept an analyis
/// manager.
template <typename IRUnitT, typename AnalysisManagerT, typename PassT>
struct PassModel<IRUnitT, AnalysisManagerT, PassT,
                 true> : PassConcept<IRUnitT, AnalysisManagerT> {
  explicit PassModel(PassT Pass) : Pass(std::move(Pass)) {}
  // We have to explicitly define all the special member functions because MSVC
  // refuses to generate them.
  PassModel(const PassModel &Arg) : Pass(Arg.Pass) {}
  PassModel(PassModel &&Arg) : Pass(Arg.Pass) {}
  PassModel &operator=(PassModel RHS) {
    std::swap(*this, RHS);
    return *this;
  }

  PreservedAnalyses run(IRUnitT IR, AnalysisManagerT *AM) override {
    return Pass.run(IR, AM);
  }
  StringRef name() override { return PassT::name(); }
  PassT Pass;
};

/// \brief Specialization of \c PassModel for passes that accept an analyis
/// manager.
template <typename IRUnitT, typename AnalysisManagerT, typename PassT>
struct PassModel<IRUnitT, AnalysisManagerT, PassT,
                 false> : PassConcept<IRUnitT, AnalysisManagerT> {
  explicit PassModel(PassT Pass) : Pass(std::move(Pass)) {}
  // We have to explicitly define all the special member functions because MSVC
  // refuses to generate them.
  PassModel(const PassModel &Arg) : Pass(Arg.Pass) {}
  PassModel(PassModel &&Arg) : Pass(Arg.Pass) {}
  PassModel &operator=(PassModel RHS) {
    std::swap(*this, RHS);
    return *this;
  }

  PreservedAnalyses run(IRUnitT IR, AnalysisManagerT *AM) override {
    return Pass.run(IR);
  }
  StringRef name() override { return PassT::name(); }
  PassT Pass;
};

/// \brief Abstract concept of an analysis result.
///
/// This concept is parameterized over the IR unit that this result pertains
/// to.
template <typename IRUnitT> struct AnalysisResultConcept {
  virtual ~AnalysisResultConcept() {}

  /// \brief Method to try and mark a result as invalid.
  ///
  /// When the outer analysis manager detects a change in some underlying
  /// unit of the IR, it will call this method on all of the results cached.
  ///
  /// This method also receives a set of preserved analyses which can be used
  /// to avoid invalidation because the pass which changed the underlying IR
  /// took care to update or preserve the analysis result in some way.
  ///
  /// \returns true if the result is indeed invalid (the default).
  virtual bool invalidate(IRUnitT IR, const PreservedAnalyses &PA) = 0;
};

/// \brief SFINAE metafunction for computing whether \c ResultT provides an
/// \c invalidate member function.
template <typename IRUnitT, typename ResultT> class ResultHasInvalidateMethod {
  typedef char SmallType;
  struct BigType { char a, b; };

  template <typename T, bool (T::*)(IRUnitT, const PreservedAnalyses &)>
  struct Checker;

  template <typename T> static SmallType f(Checker<T, &T::invalidate> *);
  template <typename T> static BigType f(...);

public:
  enum { Value = sizeof(f<ResultT>(0)) == sizeof(SmallType) };
};

/// \brief Wrapper to model the analysis result concept.
///
/// By default, this will implement the invalidate method with a trivial
/// implementation so that the actual analysis result doesn't need to provide
/// an invalidation handler. It is only selected when the invalidation handler
/// is not part of the ResultT's interface.
template <typename IRUnitT, typename PassT, typename ResultT,
          bool HasInvalidateHandler =
              ResultHasInvalidateMethod<IRUnitT, ResultT>::Value>
struct AnalysisResultModel;

/// \brief Specialization of \c AnalysisResultModel which provides the default
/// invalidate functionality.
template <typename IRUnitT, typename PassT, typename ResultT>
struct AnalysisResultModel<IRUnitT, PassT, ResultT,
                           false> : AnalysisResultConcept<IRUnitT> {
  explicit AnalysisResultModel(ResultT Result) : Result(std::move(Result)) {}
  // We have to explicitly define all the special member functions because MSVC
  // refuses to generate them.
  AnalysisResultModel(const AnalysisResultModel &Arg) : Result(Arg.Result) {}
  AnalysisResultModel(AnalysisResultModel &&Arg) : Result(Arg.Result) {}
  AnalysisResultModel &operator=(AnalysisResultModel RHS) {
    std::swap(*this, RHS);
    return *this;
  }

  /// \brief The model bases invalidation solely on being in the preserved set.
  //
  // FIXME: We should actually use two different concepts for analysis results
  // rather than two different models, and avoid the indirect function call for
  // ones that use the trivial behavior.
  bool invalidate(IRUnitT, const PreservedAnalyses &PA) override {
    return !PA.preserved(PassT::ID());
  }

  ResultT Result;
};

/// \brief Specialization of \c AnalysisResultModel which delegates invalidate
/// handling to \c ResultT.
template <typename IRUnitT, typename PassT, typename ResultT>
struct AnalysisResultModel<IRUnitT, PassT, ResultT,
                           true> : AnalysisResultConcept<IRUnitT> {
  explicit AnalysisResultModel(ResultT Result) : Result(std::move(Result)) {}
  // We have to explicitly define all the special member functions because MSVC
  // refuses to generate them.
  AnalysisResultModel(const AnalysisResultModel &Arg) : Result(Arg.Result) {}
  AnalysisResultModel(AnalysisResultModel &&Arg) : Result(Arg.Result) {}
  AnalysisResultModel &operator=(AnalysisResultModel RHS) {
    std::swap(*this, RHS);
    return *this;
  }

  /// \brief The model delegates to the \c ResultT method.
  bool invalidate(IRUnitT IR, const PreservedAnalyses &PA) override {
    return Result.invalidate(IR, PA);
  }

  ResultT Result;
};

/// \brief Abstract concept of an analysis pass.
///
/// This concept is parameterized over the IR unit that it can run over and
/// produce an analysis result.
template <typename IRUnitT, typename AnalysisManagerT>
struct AnalysisPassConcept {
  virtual ~AnalysisPassConcept() {}

  /// \brief Method to run this analysis over a unit of IR.
  /// \returns A unique_ptr to the analysis result object to be queried by
  /// users.
  virtual std::unique_ptr<AnalysisResultConcept<IRUnitT>>
      run(IRUnitT IR, AnalysisManagerT *AM) = 0;
};

/// \brief Wrapper to model the analysis pass concept.
///
/// Can wrap any type which implements a suitable \c run method. The method
/// must accept the IRUnitT as an argument and produce an object which can be
/// wrapped in a \c AnalysisResultModel.
template <typename IRUnitT, typename AnalysisManagerT, typename PassT,
          bool AcceptsAnalysisManager = PassRunAcceptsAnalysisManager<
              IRUnitT, AnalysisManagerT, PassT,
              typename PassT::Result>::Value> struct AnalysisPassModel;

/// \brief Specialization of \c AnalysisPassModel which passes an
/// \c AnalysisManager to PassT's run method.
template <typename IRUnitT, typename AnalysisManagerT, typename PassT>
struct AnalysisPassModel<IRUnitT, AnalysisManagerT, PassT,
                         true> : AnalysisPassConcept<IRUnitT,
                                                     AnalysisManagerT> {
  explicit AnalysisPassModel(PassT Pass) : Pass(std::move(Pass)) {}
  // We have to explicitly define all the special member functions because MSVC
  // refuses to generate them.
  AnalysisPassModel(const AnalysisPassModel &Arg) : Pass(Arg.Pass) {}
  AnalysisPassModel(AnalysisPassModel &&Arg) : Pass(Arg.Pass) {}
  AnalysisPassModel &operator=(AnalysisPassModel RHS) {
    std::swap(*this, RHS);
    return *this;
  }

  // FIXME: Replace PassT::Result with type traits when we use C++11.
  typedef AnalysisResultModel<IRUnitT, PassT, typename PassT::Result>
      ResultModelT;

  /// \brief The model delegates to the \c PassT::run method.
  ///
  /// The return is wrapped in an \c AnalysisResultModel.
  std::unique_ptr<AnalysisResultConcept<IRUnitT>>
  run(IRUnitT IR, AnalysisManagerT *AM) override {
    return make_unique<ResultModelT>(Pass.run(IR, AM));
  }

  PassT Pass;
};

/// \brief Specialization of \c AnalysisPassModel which does not pass an
/// \c AnalysisManager to PassT's run method.
template <typename IRUnitT, typename AnalysisManagerT, typename PassT>
struct AnalysisPassModel<IRUnitT, AnalysisManagerT, PassT,
                         false> : AnalysisPassConcept<IRUnitT,
                                                     AnalysisManagerT> {
  explicit AnalysisPassModel(PassT Pass) : Pass(std::move(Pass)) {}
  // We have to explicitly define all the special member functions because MSVC
  // refuses to generate them.
  AnalysisPassModel(const AnalysisPassModel &Arg) : Pass(Arg.Pass) {}
  AnalysisPassModel(AnalysisPassModel &&Arg) : Pass(Arg.Pass) {}
  AnalysisPassModel &operator=(AnalysisPassModel RHS) {
    std::swap(*this, RHS);
    return *this;
  }

  // FIXME: Replace PassT::Result with type traits when we use C++11.
  typedef AnalysisResultModel<IRUnitT, PassT, typename PassT::Result>
      ResultModelT;

  /// \brief The model delegates to the \c PassT::run method.
  ///
  /// The return is wrapped in an \c AnalysisResultModel.
  std::unique_ptr<AnalysisResultConcept<IRUnitT>>
  run(IRUnitT IR, AnalysisManagerT *) override {
    return make_unique<ResultModelT>(Pass.run(IR));
  }

  PassT Pass;
};

}

class ModuleAnalysisManager;

class ModulePassManager {
public:
  // We have to explicitly define all the special member functions because MSVC
  // refuses to generate them.
  ModulePassManager() {}
  ModulePassManager(ModulePassManager &&Arg) : Passes(std::move(Arg.Passes)) {}
  ModulePassManager &operator=(ModulePassManager &&RHS) {
    Passes = std::move(RHS.Passes);
    return *this;
  }

  /// \brief Run all of the module passes in this module pass manager over
  /// a module.
  ///
  /// This method should only be called for a single module as there is the
  /// expectation that the lifetime of a pass is bounded to that of a module.
  PreservedAnalyses run(Module *M, ModuleAnalysisManager *AM = 0);

  template <typename ModulePassT> void addPass(ModulePassT Pass) {
    Passes.emplace_back(new ModulePassModel<ModulePassT>(std::move(Pass)));
  }

  static StringRef name() { return "ModulePassManager"; }

private:
  // Pull in the concept type and model template specialized for modules.
  typedef detail::PassConcept<Module *, ModuleAnalysisManager> ModulePassConcept;
  template <typename PassT>
  struct ModulePassModel
      : detail::PassModel<Module *, ModuleAnalysisManager, PassT> {
    ModulePassModel(PassT Pass)
        : detail::PassModel<Module *, ModuleAnalysisManager, PassT>(
              std::move(Pass)) {}
  };

  ModulePassManager(const ModulePassManager &) LLVM_DELETED_FUNCTION;
  ModulePassManager &operator=(const ModulePassManager &) LLVM_DELETED_FUNCTION;

  std::vector<std::unique_ptr<ModulePassConcept>> Passes;
};

class FunctionAnalysisManager;

class FunctionPassManager {
public:
  // We have to explicitly define all the special member functions because MSVC
  // refuses to generate them.
  FunctionPassManager() {}
  FunctionPassManager(FunctionPassManager &&Arg) : Passes(std::move(Arg.Passes)) {}
  FunctionPassManager &operator=(FunctionPassManager &&RHS) {
    Passes = std::move(RHS.Passes);
    return *this;
  }

  template <typename FunctionPassT> void addPass(FunctionPassT Pass) {
    Passes.emplace_back(new FunctionPassModel<FunctionPassT>(std::move(Pass)));
  }

  PreservedAnalyses run(Function *F, FunctionAnalysisManager *AM = 0);

  static StringRef name() { return "FunctionPassManager"; }

private:
  // Pull in the concept type and model template specialized for functions.
  typedef detail::PassConcept<Function *, FunctionAnalysisManager>
      FunctionPassConcept;
  template <typename PassT>
  struct FunctionPassModel
      : detail::PassModel<Function *, FunctionAnalysisManager, PassT> {
    FunctionPassModel(PassT Pass)
        : detail::PassModel<Function *, FunctionAnalysisManager, PassT>(
              std::move(Pass)) {}
  };

  FunctionPassManager(const FunctionPassManager &) LLVM_DELETED_FUNCTION;
  FunctionPassManager &
  operator=(const FunctionPassManager &) LLVM_DELETED_FUNCTION;

  std::vector<std::unique_ptr<FunctionPassConcept>> Passes;
};

namespace detail {

/// \brief A CRTP base used to implement analysis managers.
///
/// This class template serves as the boiler plate of an analysis manager. Any
/// analysis manager can be implemented on top of this base class. Any
/// implementation will be required to provide specific hooks:
///
/// - getResultImpl
/// - getCachedResultImpl
/// - invalidateImpl
///
/// The details of the call pattern are within.
template <typename DerivedT, typename IRUnitT>
class AnalysisManagerBase {
  DerivedT *derived_this() { return static_cast<DerivedT *>(this); }
  const DerivedT *derived_this() const { return static_cast<const DerivedT *>(this); }

  AnalysisManagerBase(const AnalysisManagerBase &) LLVM_DELETED_FUNCTION;
  AnalysisManagerBase &
  operator=(const AnalysisManagerBase &) LLVM_DELETED_FUNCTION;

protected:
  typedef detail::AnalysisResultConcept<IRUnitT> ResultConceptT;
  typedef detail::AnalysisPassConcept<IRUnitT, DerivedT> PassConceptT;

  // FIXME: Provide template aliases for the models when we're using C++11 in
  // a mode supporting them.

  // We have to explicitly define all the special member functions because MSVC
  // refuses to generate them.
  AnalysisManagerBase() {}
  AnalysisManagerBase(AnalysisManagerBase &&Arg)
      : AnalysisPasses(std::move(Arg.AnalysisPasses)) {}
  AnalysisManagerBase &operator=(AnalysisManagerBase &&RHS) {
    AnalysisPasses = std::move(RHS.AnalysisPasses);
    return *this;
  }

public:
  /// \brief Get the result of an analysis pass for this module.
  ///
  /// If there is not a valid cached result in the manager already, this will
  /// re-run the analysis to produce a valid result.
  template <typename PassT> typename PassT::Result &getResult(IRUnitT IR) {
    assert(AnalysisPasses.count(PassT::ID()) &&
           "This analysis pass was not registered prior to being queried");

    ResultConceptT &ResultConcept =
        derived_this()->getResultImpl(PassT::ID(), IR);
    typedef detail::AnalysisResultModel<IRUnitT, PassT, typename PassT::Result>
        ResultModelT;
    return static_cast<ResultModelT &>(ResultConcept).Result;
  }

  /// \brief Get the cached result of an analysis pass for this module.
  ///
  /// This method never runs the analysis.
  ///
  /// \returns null if there is no cached result.
  template <typename PassT>
  typename PassT::Result *getCachedResult(IRUnitT IR) const {
    assert(AnalysisPasses.count(PassT::ID()) &&
           "This analysis pass was not registered prior to being queried");

    ResultConceptT *ResultConcept =
        derived_this()->getCachedResultImpl(PassT::ID(), IR);
    if (!ResultConcept)
      return 0;

    typedef detail::AnalysisResultModel<IRUnitT, PassT, typename PassT::Result>
        ResultModelT;
    return &static_cast<ResultModelT *>(ResultConcept)->Result;
  }

  /// \brief Register an analysis pass with the manager.
  ///
  /// This provides an initialized and set-up analysis pass to the analysis
  /// manager. Whomever is setting up analysis passes must use this to populate
  /// the manager with all of the analysis passes available.
  template <typename PassT> void registerPass(PassT Pass) {
    assert(!AnalysisPasses.count(PassT::ID()) &&
           "Registered the same analysis pass twice!");
    typedef detail::AnalysisPassModel<IRUnitT, DerivedT, PassT> PassModelT;
    AnalysisPasses[PassT::ID()].reset(new PassModelT(std::move(Pass)));
  }

  /// \brief Invalidate a specific analysis pass for an IR module.
  ///
  /// Note that the analysis result can disregard invalidation.
  template <typename PassT> void invalidate(Module *M) {
    assert(AnalysisPasses.count(PassT::ID()) &&
           "This analysis pass was not registered prior to being invalidated");
    derived_this()->invalidateImpl(PassT::ID(), M);
  }

  /// \brief Invalidate analyses cached for an IR unit.
  ///
  /// Walk through all of the analyses pertaining to this unit of IR and
  /// invalidate them unless they are preserved by the PreservedAnalyses set.
  void invalidate(IRUnitT IR, const PreservedAnalyses &PA) {
    derived_this()->invalidateImpl(IR, PA);
  }

protected:
  /// \brief Lookup a registered analysis pass.
  PassConceptT &lookupPass(void *PassID) {
    typename AnalysisPassMapT::iterator PI = AnalysisPasses.find(PassID);
    assert(PI != AnalysisPasses.end() &&
           "Analysis passes must be registered prior to being queried!");
    return *PI->second;
  }

  /// \brief Lookup a registered analysis pass.
  const PassConceptT &lookupPass(void *PassID) const {
    typename AnalysisPassMapT::const_iterator PI = AnalysisPasses.find(PassID);
    assert(PI != AnalysisPasses.end() &&
           "Analysis passes must be registered prior to being queried!");
    return *PI->second;
  }

private:
  /// \brief Map type from module analysis pass ID to pass concept pointer.
  typedef DenseMap<void *, std::unique_ptr<PassConceptT>> AnalysisPassMapT;

  /// \brief Collection of module analysis passes, indexed by ID.
  AnalysisPassMapT AnalysisPasses;
};

}

/// \brief A module analysis pass manager with lazy running and caching of
/// results.
class ModuleAnalysisManager
    : public detail::AnalysisManagerBase<ModuleAnalysisManager, Module *> {
  friend class detail::AnalysisManagerBase<ModuleAnalysisManager, Module *>;
  typedef detail::AnalysisManagerBase<ModuleAnalysisManager, Module *> BaseT;
  typedef BaseT::ResultConceptT ResultConceptT;
  typedef BaseT::PassConceptT PassConceptT;

public:
  // We have to explicitly define all the special member functions because MSVC
  // refuses to generate them.
  ModuleAnalysisManager() {}
  ModuleAnalysisManager(ModuleAnalysisManager &&Arg)
      : BaseT(std::move(static_cast<BaseT &>(Arg))),
        ModuleAnalysisResults(std::move(Arg.ModuleAnalysisResults)) {}
  ModuleAnalysisManager &operator=(ModuleAnalysisManager &&RHS) {
    BaseT::operator=(std::move(static_cast<BaseT &>(RHS)));
    ModuleAnalysisResults = std::move(RHS.ModuleAnalysisResults);
    return *this;
  }

private:
  ModuleAnalysisManager(const ModuleAnalysisManager &) LLVM_DELETED_FUNCTION;
  ModuleAnalysisManager &
  operator=(const ModuleAnalysisManager &) LLVM_DELETED_FUNCTION;

  /// \brief Get a module pass result, running the pass if necessary.
  ResultConceptT &getResultImpl(void *PassID, Module *M);

  /// \brief Get a cached module pass result or return null.
  ResultConceptT *getCachedResultImpl(void *PassID, Module *M) const;

  /// \brief Invalidate a module pass result.
  void invalidateImpl(void *PassID, Module *M);

  /// \brief Invalidate results across a module.
  void invalidateImpl(Module *M, const PreservedAnalyses &PA);

  /// \brief Map type from module analysis pass ID to pass result concept pointer.
  typedef DenseMap<void *,
                   std::unique_ptr<detail::AnalysisResultConcept<Module *>>>
      ModuleAnalysisResultMapT;

  /// \brief Cache of computed module analysis results for this module.
  ModuleAnalysisResultMapT ModuleAnalysisResults;
};

/// \brief A function analysis manager to coordinate and cache analyses run over
/// a module.
class FunctionAnalysisManager
    : public detail::AnalysisManagerBase<FunctionAnalysisManager, Function *> {
  friend class detail::AnalysisManagerBase<FunctionAnalysisManager, Function *>;
  typedef detail::AnalysisManagerBase<FunctionAnalysisManager, Function *> BaseT;
  typedef BaseT::ResultConceptT ResultConceptT;
  typedef BaseT::PassConceptT PassConceptT;

public:
  // Most public APIs are inherited from the CRTP base class.

  // We have to explicitly define all the special member functions because MSVC
  // refuses to generate them.
  FunctionAnalysisManager() {}
  FunctionAnalysisManager(FunctionAnalysisManager &&Arg)
      : BaseT(std::move(static_cast<BaseT &>(Arg))),
        FunctionAnalysisResults(std::move(Arg.FunctionAnalysisResults)) {}
  FunctionAnalysisManager &operator=(FunctionAnalysisManager &&RHS) {
    BaseT::operator=(std::move(static_cast<BaseT &>(RHS)));
    FunctionAnalysisResults = std::move(RHS.FunctionAnalysisResults);
    return *this;
  }

  /// \brief Returns true if the analysis manager has an empty results cache.
  bool empty() const;

  /// \brief Clear the function analysis result cache.
  ///
  /// This routine allows cleaning up when the set of functions itself has
  /// potentially changed, and thus we can't even look up a a result and
  /// invalidate it directly. Notably, this does *not* call invalidate
  /// functions as there is nothing to be done for them.
  void clear();

private:
  FunctionAnalysisManager(const FunctionAnalysisManager &) LLVM_DELETED_FUNCTION;
  FunctionAnalysisManager &
  operator=(const FunctionAnalysisManager &) LLVM_DELETED_FUNCTION;

  /// \brief Get a function pass result, running the pass if necessary.
  ResultConceptT &getResultImpl(void *PassID, Function *F);

  /// \brief Get a cached function pass result or return null.
  ResultConceptT *getCachedResultImpl(void *PassID, Function *F) const;

  /// \brief Invalidate a function pass result.
  void invalidateImpl(void *PassID, Function *F);

  /// \brief Invalidate the results for a function..
  void invalidateImpl(Function *F, const PreservedAnalyses &PA);

  /// \brief List of function analysis pass IDs and associated concept pointers.
  ///
  /// Requires iterators to be valid across appending new entries and arbitrary
  /// erases. Provides both the pass ID and concept pointer such that it is
  /// half of a bijection and provides storage for the actual result concept.
  typedef std::list<std::pair<
      void *, std::unique_ptr<detail::AnalysisResultConcept<Function *>>>>
  FunctionAnalysisResultListT;

  /// \brief Map type from function pointer to our custom list type.
  typedef DenseMap<Function *, FunctionAnalysisResultListT>
  FunctionAnalysisResultListMapT;

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
};

/// \brief A module analysis which acts as a proxy for a function analysis
/// manager.
///
/// This primarily proxies invalidation information from the module analysis
/// manager and module pass manager to a function analysis manager. You should
/// never use a function analysis manager from within (transitively) a module
/// pass manager unless your parent module pass has received a proxy result
/// object for it.
class FunctionAnalysisManagerModuleProxy {
public:
  class Result;

  static void *ID() { return (void *)&PassID; }

  explicit FunctionAnalysisManagerModuleProxy(FunctionAnalysisManager &FAM)
      : FAM(FAM) {}
  FunctionAnalysisManagerModuleProxy(
      const FunctionAnalysisManagerModuleProxy &Arg)
      : FAM(Arg.FAM) {}
  FunctionAnalysisManagerModuleProxy(FunctionAnalysisManagerModuleProxy &&Arg)
      : FAM(Arg.FAM) {}
  FunctionAnalysisManagerModuleProxy &
  operator=(FunctionAnalysisManagerModuleProxy RHS) {
    std::swap(*this, RHS);
    return *this;
  }

  /// \brief Run the analysis pass and create our proxy result object.
  ///
  /// This doesn't do any interesting work, it is primarily used to insert our
  /// proxy result object into the module analysis cache so that we can proxy
  /// invalidation to the function analysis manager.
  ///
  /// In debug builds, it will also assert that the analysis manager is empty
  /// as no queries should arrive at the function analysis manager prior to
  /// this analysis being requested.
  Result run(Module *M);

private:
  static char PassID;

  FunctionAnalysisManager &FAM;
};

/// \brief The result proxy object for the
/// \c FunctionAnalysisManagerModuleProxy.
///
/// See its documentation for more information.
class FunctionAnalysisManagerModuleProxy::Result {
public:
  explicit Result(FunctionAnalysisManager &FAM) : FAM(FAM) {}
  // We have to explicitly define all the special member functions because MSVC
  // refuses to generate them.
  Result(const Result &Arg) : FAM(Arg.FAM) {}
  Result(Result &&Arg) : FAM(Arg.FAM) {}
  Result &operator=(Result RHS) {
    std::swap(*this, RHS);
    return *this;
  }
  ~Result();

  /// \brief Accessor for the \c FunctionAnalysisManager.
  FunctionAnalysisManager &getManager() { return FAM; }

  /// \brief Handler for invalidation of the module.
  ///
  /// If this analysis itself is preserved, then we assume that the set of \c
  /// Function objects in the \c Module hasn't changed and thus we don't need
  /// to invalidate *all* cached data associated with a \c Function* in the \c
  /// FunctionAnalysisManager.
  ///
  /// Regardless of whether this analysis is marked as preserved, all of the
  /// analyses in the \c FunctionAnalysisManager are potentially invalidated
  /// based on the set of preserved analyses.
  bool invalidate(Module *M, const PreservedAnalyses &PA);

private:
  FunctionAnalysisManager &FAM;
};

/// \brief A function analysis which acts as a proxy for a module analysis
/// manager.
///
/// This primarily provides an accessor to a parent module analysis manager to
/// function passes. Only the const interface of the module analysis manager is
/// provided to indicate that once inside of a function analysis pass you
/// cannot request a module analysis to actually run. Instead, the user must
/// rely on the \c getCachedResult API.
///
/// This proxy *doesn't* manage the invalidation in any way. That is handled by
/// the recursive return path of each layer of the pass manager and the
/// returned PreservedAnalysis set.
class ModuleAnalysisManagerFunctionProxy {
public:
  /// \brief Result proxy object for \c ModuleAnalysisManagerFunctionProxy.
  class Result {
  public:
    explicit Result(const ModuleAnalysisManager &MAM) : MAM(MAM) {}
    // We have to explicitly define all the special member functions because
    // MSVC refuses to generate them.
    Result(const Result &Arg) : MAM(Arg.MAM) {}
    Result(Result &&Arg) : MAM(Arg.MAM) {}
    Result &operator=(Result RHS) {
      std::swap(*this, RHS);
      return *this;
    }

    const ModuleAnalysisManager &getManager() const { return MAM; }

    /// \brief Handle invalidation by ignoring it, this pass is immutable.
    bool invalidate(Function *) { return false; }

  private:
    const ModuleAnalysisManager &MAM;
  };

  static void *ID() { return (void *)&PassID; }

  ModuleAnalysisManagerFunctionProxy(const ModuleAnalysisManager &MAM)
      : MAM(MAM) {}

  /// \brief Run the analysis pass and create our proxy result object.
  /// Nothing to see here, it just forwards the \c MAM reference into the
  /// result.
  Result run(Function *) { return Result(MAM); }

private:
  static char PassID;

  const ModuleAnalysisManager &MAM;
};

/// \brief Trivial adaptor that maps from a module to its functions.
///
/// Designed to allow composition of a FunctionPass(Manager) and
/// a ModulePassManager. Note that if this pass is constructed with a pointer
/// to a \c ModuleAnalysisManager it will run the
/// \c FunctionAnalysisManagerModuleProxy analysis prior to running the function
/// pass over the module to enable a \c FunctionAnalysisManager to be used
/// within this run safely.
template <typename FunctionPassT>
class ModuleToFunctionPassAdaptor {
public:
  explicit ModuleToFunctionPassAdaptor(FunctionPassT Pass)
      : Pass(std::move(Pass)) {}
  // We have to explicitly define all the special member functions because MSVC
  // refuses to generate them.
  ModuleToFunctionPassAdaptor(const ModuleToFunctionPassAdaptor &Arg)
      : Pass(Arg.Pass) {}
  ModuleToFunctionPassAdaptor(ModuleToFunctionPassAdaptor &&Arg)
      : Pass(std::move(Arg.Pass)) {}
  ModuleToFunctionPassAdaptor &operator=(ModuleToFunctionPassAdaptor RHS) {
    std::swap(*this, RHS);
    return *this;
  }

  /// \brief Runs the function pass across every function in the module.
  PreservedAnalyses run(Module *M, ModuleAnalysisManager *AM) {
    FunctionAnalysisManager *FAM = 0;
    if (AM)
      // Setup the function analysis manager from its proxy.
      FAM = &AM->getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();

    PreservedAnalyses PA = PreservedAnalyses::all();
    for (Module::iterator I = M->begin(), E = M->end(); I != E; ++I) {
      PreservedAnalyses PassPA = Pass.run(I, FAM);

      // We know that the function pass couldn't have invalidated any other
      // function's analyses (that's the contract of a function pass), so
      // directly handle the function analysis manager's invalidation here.
      if (FAM)
        FAM->invalidate(I, PassPA);

      // Then intersect the preserved set so that invalidation of module
      // analyses will eventually occur when the module pass completes.
      PA.intersect(std::move(PassPA));
    }

    // By definition we preserve the proxy. This precludes *any* invalidation
    // of function analyses by the proxy, but that's OK because we've taken
    // care to invalidate analyses in the function analysis manager
    // incrementally above.
    PA.preserve<FunctionAnalysisManagerModuleProxy>();
    return PA;
  }

  static StringRef name() { return "ModuleToFunctionPassAdaptor"; }

private:
  FunctionPassT Pass;
};

/// \brief A function to deduce a function pass type and wrap it in the
/// templated adaptor.
template <typename FunctionPassT>
ModuleToFunctionPassAdaptor<FunctionPassT>
createModuleToFunctionPassAdaptor(FunctionPassT Pass) {
  return std::move(ModuleToFunctionPassAdaptor<FunctionPassT>(std::move(Pass)));
}

}

#endif
