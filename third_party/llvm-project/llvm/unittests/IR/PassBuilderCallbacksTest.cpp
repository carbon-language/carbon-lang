//===- unittests/IR/PassBuilderCallbacksTest.cpp - PB Callback Tests --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Testing/Support/Error.h"
#include <functional>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <llvm/ADT/Any.h>
#include <llvm/Analysis/CGSCCPassManager.h>
#include <llvm/Analysis/LoopAnalysisManager.h>
#include <llvm/AsmParser/Parser.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/PassInstrumentation.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/Regex.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Transforms/Scalar/LoopPassManager.h>

using namespace llvm;

namespace {
using testing::AnyNumber;
using testing::DoAll;
using testing::Not;
using testing::Return;
using testing::WithArgs;
using testing::_;

/// A CRTP base for analysis mock handles
///
/// This class reconciles mocking with the value semantics implementation of the
/// AnalysisManager. Analysis mock handles should derive from this class and
/// call \c setDefault() in their constroctur for wiring up the defaults defined
/// by this base with their mock run() and invalidate() implementations.
template <typename DerivedT, typename IRUnitT,
          typename AnalysisManagerT = AnalysisManager<IRUnitT>,
          typename... ExtraArgTs>
class MockAnalysisHandleBase {
public:
  class Analysis : public AnalysisInfoMixin<Analysis> {
    friend AnalysisInfoMixin<Analysis>;
    friend MockAnalysisHandleBase;
    static AnalysisKey Key;

    DerivedT *Handle;

    Analysis(DerivedT &Handle) : Handle(&Handle) {
      static_assert(std::is_base_of<MockAnalysisHandleBase, DerivedT>::value,
                    "Must pass the derived type to this template!");
    }

  public:
    class Result {
      friend MockAnalysisHandleBase;

      DerivedT *Handle;

      Result(DerivedT &Handle) : Handle(&Handle) {}

    public:
      // Forward invalidation events to the mock handle.
      bool invalidate(IRUnitT &IR, const PreservedAnalyses &PA,
                      typename AnalysisManagerT::Invalidator &Inv) {
        return Handle->invalidate(IR, PA, Inv);
      }
    };

    Result run(IRUnitT &IR, AnalysisManagerT &AM, ExtraArgTs... ExtraArgs) {
      return Handle->run(IR, AM, ExtraArgs...);
    }
  };

  Analysis getAnalysis() { return Analysis(static_cast<DerivedT &>(*this)); }
  typename Analysis::Result getResult() {
    return typename Analysis::Result(static_cast<DerivedT &>(*this));
  }
  static StringRef getName() { return llvm::getTypeName<DerivedT>(); }

protected:
  // FIXME: MSVC seems unable to handle a lambda argument to Invoke from within
  // the template, so we use a boring static function.
  static bool invalidateCallback(IRUnitT &IR, const PreservedAnalyses &PA,
                                 typename AnalysisManagerT::Invalidator &Inv) {
    auto PAC = PA.template getChecker<Analysis>();
    return !PAC.preserved() &&
           !PAC.template preservedSet<AllAnalysesOn<IRUnitT>>();
  }

  /// Derived classes should call this in their constructor to set up default
  /// mock actions. (We can't do this in our constructor because this has to
  /// run after the DerivedT is constructed.)
  void setDefaults() {
    ON_CALL(static_cast<DerivedT &>(*this),
            run(_, _, testing::Matcher<ExtraArgTs>(_)...))
        .WillByDefault(Return(this->getResult()));
    ON_CALL(static_cast<DerivedT &>(*this), invalidate(_, _, _))
        .WillByDefault(&invalidateCallback);
  }
};

/// A CRTP base for pass mock handles
///
/// This class reconciles mocking with the value semantics implementation of the
/// PassManager. Pass mock handles should derive from this class and
/// call \c setDefault() in their constroctur for wiring up the defaults defined
/// by this base with their mock run() and invalidate() implementations.
template <typename DerivedT, typename IRUnitT, typename AnalysisManagerT,
          typename... ExtraArgTs>
AnalysisKey MockAnalysisHandleBase<DerivedT, IRUnitT, AnalysisManagerT,
                                   ExtraArgTs...>::Analysis::Key;

template <typename DerivedT, typename IRUnitT,
          typename AnalysisManagerT = AnalysisManager<IRUnitT>,
          typename... ExtraArgTs>
class MockPassHandleBase {
public:
  class Pass : public PassInfoMixin<Pass> {
    friend MockPassHandleBase;

    DerivedT *Handle;

    Pass(DerivedT &Handle) : Handle(&Handle) {
      static_assert(std::is_base_of<MockPassHandleBase, DerivedT>::value,
                    "Must pass the derived type to this template!");
    }

  public:
    PreservedAnalyses run(IRUnitT &IR, AnalysisManagerT &AM,
                          ExtraArgTs... ExtraArgs) {
      return Handle->run(IR, AM, ExtraArgs...);
    }
  };

  static StringRef getName() { return llvm::getTypeName<DerivedT>(); }

  Pass getPass() { return Pass(static_cast<DerivedT &>(*this)); }

protected:
  /// Derived classes should call this in their constructor to set up default
  /// mock actions. (We can't do this in our constructor because this has to
  /// run after the DerivedT is constructed.)
  void setDefaults() {
    ON_CALL(static_cast<DerivedT &>(*this),
            run(_, _, testing::Matcher<ExtraArgTs>(_)...))
        .WillByDefault(Return(PreservedAnalyses::all()));
  }
};

/// Mock handles for passes for the IRUnits Module, CGSCC, Function, Loop.
/// These handles define the appropriate run() mock interface for the respective
/// IRUnit type.
template <typename IRUnitT> struct MockPassHandle;
template <>
struct MockPassHandle<Loop>
    : MockPassHandleBase<MockPassHandle<Loop>, Loop, LoopAnalysisManager,
                         LoopStandardAnalysisResults &, LPMUpdater &> {
  MOCK_METHOD4(run,
               PreservedAnalyses(Loop &, LoopAnalysisManager &,
                                 LoopStandardAnalysisResults &, LPMUpdater &));
  static void invalidateLoop(Loop &L, LoopAnalysisManager &,
                             LoopStandardAnalysisResults &,
                             LPMUpdater &Updater) {
    Updater.markLoopAsDeleted(L, L.getName());
  }
  MockPassHandle() { setDefaults(); }
};

template <>
struct MockPassHandle<LoopNest>
    : MockPassHandleBase<MockPassHandle<LoopNest>, LoopNest,
                         LoopAnalysisManager, LoopStandardAnalysisResults &,
                         LPMUpdater &> {
  MOCK_METHOD4(run,
               PreservedAnalyses(LoopNest &, LoopAnalysisManager &,
                                 LoopStandardAnalysisResults &, LPMUpdater &));
  static void invalidateLoopNest(LoopNest &L, LoopAnalysisManager &,
                                 LoopStandardAnalysisResults &,
                                 LPMUpdater &Updater) {
    Updater.markLoopAsDeleted(L.getOutermostLoop(), L.getName());
  }
  MockPassHandle() { setDefaults(); }
};

template <>
struct MockPassHandle<Function>
    : MockPassHandleBase<MockPassHandle<Function>, Function> {
  MOCK_METHOD2(run, PreservedAnalyses(Function &, FunctionAnalysisManager &));

  MockPassHandle() { setDefaults(); }
};

template <>
struct MockPassHandle<LazyCallGraph::SCC>
    : MockPassHandleBase<MockPassHandle<LazyCallGraph::SCC>, LazyCallGraph::SCC,
                         CGSCCAnalysisManager, LazyCallGraph &,
                         CGSCCUpdateResult &> {
  MOCK_METHOD4(run,
               PreservedAnalyses(LazyCallGraph::SCC &, CGSCCAnalysisManager &,
                                 LazyCallGraph &G, CGSCCUpdateResult &UR));

  static void invalidateSCC(LazyCallGraph::SCC &C, CGSCCAnalysisManager &,
                            LazyCallGraph &, CGSCCUpdateResult &UR) {
    UR.InvalidatedSCCs.insert(&C);
  }

  MockPassHandle() { setDefaults(); }
};

template <>
struct MockPassHandle<Module>
    : MockPassHandleBase<MockPassHandle<Module>, Module> {
  MOCK_METHOD2(run, PreservedAnalyses(Module &, ModuleAnalysisManager &));

  MockPassHandle() { setDefaults(); }
};

/// Mock handles for analyses for the IRUnits Module, CGSCC, Function, Loop.
/// These handles define the appropriate run() and invalidate() mock interfaces
/// for the respective IRUnit type.
template <typename IRUnitT> struct MockAnalysisHandle;
template <>
struct MockAnalysisHandle<Loop>
    : MockAnalysisHandleBase<MockAnalysisHandle<Loop>, Loop,
                             LoopAnalysisManager,
                             LoopStandardAnalysisResults &> {

  MOCK_METHOD3_T(run, typename Analysis::Result(Loop &, LoopAnalysisManager &,
                                                LoopStandardAnalysisResults &));

  MOCK_METHOD3_T(invalidate, bool(Loop &, const PreservedAnalyses &,
                                  LoopAnalysisManager::Invalidator &));

  MockAnalysisHandle<Loop>() { this->setDefaults(); }
};

template <>
struct MockAnalysisHandle<Function>
    : MockAnalysisHandleBase<MockAnalysisHandle<Function>, Function> {
  MOCK_METHOD2(run, Analysis::Result(Function &, FunctionAnalysisManager &));

  MOCK_METHOD3(invalidate, bool(Function &, const PreservedAnalyses &,
                                FunctionAnalysisManager::Invalidator &));

  MockAnalysisHandle<Function>() { setDefaults(); }
};

template <>
struct MockAnalysisHandle<LazyCallGraph::SCC>
    : MockAnalysisHandleBase<MockAnalysisHandle<LazyCallGraph::SCC>,
                             LazyCallGraph::SCC, CGSCCAnalysisManager,
                             LazyCallGraph &> {
  MOCK_METHOD3(run, Analysis::Result(LazyCallGraph::SCC &,
                                     CGSCCAnalysisManager &, LazyCallGraph &));

  MOCK_METHOD3(invalidate, bool(LazyCallGraph::SCC &, const PreservedAnalyses &,
                                CGSCCAnalysisManager::Invalidator &));

  MockAnalysisHandle<LazyCallGraph::SCC>() { setDefaults(); }
};

template <>
struct MockAnalysisHandle<Module>
    : MockAnalysisHandleBase<MockAnalysisHandle<Module>, Module> {
  MOCK_METHOD2(run, Analysis::Result(Module &, ModuleAnalysisManager &));

  MOCK_METHOD3(invalidate, bool(Module &, const PreservedAnalyses &,
                                ModuleAnalysisManager::Invalidator &));

  MockAnalysisHandle<Module>() { setDefaults(); }
};

static std::unique_ptr<Module> parseIR(LLVMContext &C, const char *IR) {
  SMDiagnostic Err;
  return parseAssemblyString(IR, Err, C);
}

/// Helper for HasName matcher that returns getName both for IRUnit and
/// for IRUnit pointer wrapper into llvm::Any (wrapped by PassInstrumentation).
template <typename IRUnitT> std::string getName(const IRUnitT &IR) {
  return std::string(IR.getName());
}

template <> std::string getName(const StringRef &name) {
  return std::string(name);
}

template <> std::string getName(const llvm::Any &WrappedIR) {
  if (any_isa<const Module *>(WrappedIR))
    return any_cast<const Module *>(WrappedIR)->getName().str();
  if (any_isa<const Function *>(WrappedIR))
    return any_cast<const Function *>(WrappedIR)->getName().str();
  if (any_isa<const Loop *>(WrappedIR))
    return any_cast<const Loop *>(WrappedIR)->getName().str();
  if (any_isa<const LoopNest *>(WrappedIR))
    return any_cast<const LoopNest *>(WrappedIR)->getName().str();
  if (any_isa<const LazyCallGraph::SCC *>(WrappedIR))
    return any_cast<const LazyCallGraph::SCC *>(WrappedIR)->getName();
  return "<UNKNOWN>";
}
/// Define a custom matcher for objects which support a 'getName' method.
///
/// LLVM often has IR objects or analysis objects which expose a name
/// and in tests it is convenient to match these by name for readability.
/// Usually, this name is either a StringRef or a plain std::string. This
/// matcher supports any type exposing a getName() method of this form whose
/// return value is compatible with an std::ostream. For StringRef, this uses
/// the shift operator defined above.
///
/// It should be used as:
///
///   HasName("my_function")
///
/// No namespace or other qualification is required.
MATCHER_P(HasName, Name, "") {
  *result_listener << "has name '" << getName(arg) << "'";
  return Name == getName(arg);
}

MATCHER_P(HasNameRegex, Name, "") {
  *result_listener << "has name '" << getName(arg) << "'";
  llvm::Regex r(Name);
  return r.match(getName(arg));
}

struct MockPassInstrumentationCallbacks {
  PassInstrumentationCallbacks Callbacks;

  MockPassInstrumentationCallbacks() {
    ON_CALL(*this, runBeforePass(_, _)).WillByDefault(Return(true));
  }
  MOCK_METHOD2(runBeforePass, bool(StringRef PassID, llvm::Any));
  MOCK_METHOD2(runBeforeSkippedPass, void(StringRef PassID, llvm::Any));
  MOCK_METHOD2(runBeforeNonSkippedPass, void(StringRef PassID, llvm::Any));
  MOCK_METHOD3(runAfterPass,
               void(StringRef PassID, llvm::Any, const PreservedAnalyses &PA));
  MOCK_METHOD2(runAfterPassInvalidated,
               void(StringRef PassID, const PreservedAnalyses &PA));
  MOCK_METHOD2(runBeforeAnalysis, void(StringRef PassID, llvm::Any));
  MOCK_METHOD2(runAfterAnalysis, void(StringRef PassID, llvm::Any));

  void registerPassInstrumentation() {
    Callbacks.registerShouldRunOptionalPassCallback(
        [this](StringRef P, llvm::Any IR) {
          return this->runBeforePass(P, IR);
        });
    Callbacks.registerBeforeSkippedPassCallback(
        [this](StringRef P, llvm::Any IR) {
          this->runBeforeSkippedPass(P, IR);
        });
    Callbacks.registerBeforeNonSkippedPassCallback(
        [this](StringRef P, llvm::Any IR) {
          this->runBeforeNonSkippedPass(P, IR);
        });
    Callbacks.registerAfterPassCallback(
        [this](StringRef P, llvm::Any IR, const PreservedAnalyses &PA) {
          this->runAfterPass(P, IR, PA);
        });
    Callbacks.registerAfterPassInvalidatedCallback(
        [this](StringRef P, const PreservedAnalyses &PA) {
          this->runAfterPassInvalidated(P, PA);
        });
    Callbacks.registerBeforeAnalysisCallback([this](StringRef P, llvm::Any IR) {
      return this->runBeforeAnalysis(P, IR);
    });
    Callbacks.registerAfterAnalysisCallback(
        [this](StringRef P, llvm::Any IR) { this->runAfterAnalysis(P, IR); });
  }

  void ignoreNonMockPassInstrumentation(StringRef IRName) {
    // Generic EXPECT_CALLs are needed to match instrumentation on unimportant
    // parts of a pipeline that we do not care about (e.g. various passes added
    // by default by PassBuilder - Verifier pass etc).
    // Make sure to avoid ignoring Mock passes/analysis, we definitely want
    // to check these explicitly.
    EXPECT_CALL(*this,
                runBeforePass(Not(HasNameRegex("Mock")), HasName(IRName)))
        .Times(AnyNumber());
    EXPECT_CALL(
        *this, runBeforeSkippedPass(Not(HasNameRegex("Mock")), HasName(IRName)))
        .Times(AnyNumber());
    EXPECT_CALL(*this, runBeforeNonSkippedPass(Not(HasNameRegex("Mock")),
                                               HasName(IRName)))
        .Times(AnyNumber());
    EXPECT_CALL(*this,
                runAfterPass(Not(HasNameRegex("Mock")), HasName(IRName), _))
        .Times(AnyNumber());
    EXPECT_CALL(*this,
                runBeforeAnalysis(Not(HasNameRegex("Mock")), HasName(IRName)))
        .Times(AnyNumber());
    EXPECT_CALL(*this,
                runAfterAnalysis(Not(HasNameRegex("Mock")), HasName(IRName)))
        .Times(AnyNumber());
  }
};

template <typename IRUnitT>
using ExtraMockPassHandle =
    std::conditional_t<std::is_same<IRUnitT, Loop>::value,
                       MockPassHandle<LoopNest>, MockPassHandle<IRUnitT>>;

template <typename PassManagerT> class PassBuilderCallbacksTest;

/// This test fixture is shared between all the actual tests below and
/// takes care of setting up appropriate defaults.
///
/// The template specialization serves to extract the IRUnit and AM types from
/// the given PassManagerT.
template <typename TestIRUnitT, typename... ExtraPassArgTs,
          typename... ExtraAnalysisArgTs>
class PassBuilderCallbacksTest<PassManager<
    TestIRUnitT, AnalysisManager<TestIRUnitT, ExtraAnalysisArgTs...>,
    ExtraPassArgTs...>> : public testing::Test {
protected:
  using IRUnitT = TestIRUnitT;
  using AnalysisManagerT = AnalysisManager<TestIRUnitT, ExtraAnalysisArgTs...>;
  using PassManagerT =
      PassManager<TestIRUnitT, AnalysisManagerT, ExtraPassArgTs...>;
  using AnalysisT = typename MockAnalysisHandle<IRUnitT>::Analysis;

  LLVMContext Context;
  std::unique_ptr<Module> M;

  MockPassInstrumentationCallbacks CallbacksHandle;

  PassBuilder PB;
  ModulePassManager PM;
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager AM;

  MockPassHandle<IRUnitT> PassHandle;
  ExtraMockPassHandle<IRUnitT> ExtraPassHandle;

  MockAnalysisHandle<IRUnitT> AnalysisHandle;

  static PreservedAnalyses getAnalysisResult(IRUnitT &U, AnalysisManagerT &AM,
                                             ExtraAnalysisArgTs &&... Args) {
    (void)AM.template getResult<AnalysisT>(
        U, std::forward<ExtraAnalysisArgTs>(Args)...);
    return PreservedAnalyses::all();
  }

  PassBuilderCallbacksTest()
      : M(parseIR(Context,
                  "declare void @bar()\n"
                  "define void @foo(i32 %n) {\n"
                  "entry:\n"
                  "  br label %loop\n"
                  "loop:\n"
                  "  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]\n"
                  "  %iv.next = add i32 %iv, 1\n"
                  "  tail call void @bar()\n"
                  "  %cmp = icmp eq i32 %iv, %n\n"
                  "  br i1 %cmp, label %exit, label %loop\n"
                  "exit:\n"
                  "  ret void\n"
                  "}\n")),
        CallbacksHandle(),
        PB(nullptr, PipelineTuningOptions(), None, &CallbacksHandle.Callbacks),
        PM(), LAM(), FAM(), CGAM(), AM() {

    EXPECT_TRUE(&CallbacksHandle.Callbacks ==
                PB.getPassInstrumentationCallbacks());

    /// Register a callback for analysis registration.
    ///
    /// The callback is a function taking a reference to an AnalyisManager
    /// object. When called, the callee gets to register its own analyses with
    /// this PassBuilder instance.
    PB.registerAnalysisRegistrationCallback([this](AnalysisManagerT &AM) {
      // Register our mock analysis
      AM.registerPass([this] { return AnalysisHandle.getAnalysis(); });
    });

    /// Register a callback for pipeline parsing.
    ///
    /// During parsing of a textual pipeline, the PassBuilder will call these
    /// callbacks for each encountered pass name that it does not know. This
    /// includes both simple pass names as well as names of sub-pipelines. In
    /// the latter case, the InnerPipeline is not empty.
    PB.registerPipelineParsingCallback(
        [this](StringRef Name, PassManagerT &PM,
               ArrayRef<PassBuilder::PipelineElement> InnerPipeline) {
          /// Handle parsing of the names of analysis utilities such as
          /// require<test-analysis> and invalidate<test-analysis> for our
          /// analysis mock handle
          if (parseAnalysisUtilityPasses<AnalysisT>("test-analysis", Name, PM))
            return true;

          /// Parse the name of our pass mock handle
          if (Name == "test-transform") {
            PM.addPass(PassHandle.getPass());
            if (std::is_same<IRUnitT, Loop>::value)
              PM.addPass(ExtraPassHandle.getPass());
            return true;
          }
          return false;
        });

    /// Register builtin analyses and cross-register the analysis proxies
    PB.registerModuleAnalyses(AM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, AM);
  }
};

using ModuleCallbacksTest = PassBuilderCallbacksTest<ModulePassManager>;
using CGSCCCallbacksTest = PassBuilderCallbacksTest<CGSCCPassManager>;
using FunctionCallbacksTest = PassBuilderCallbacksTest<FunctionPassManager>;
using LoopCallbacksTest = PassBuilderCallbacksTest<LoopPassManager>;

/// Test parsing of the name of our mock pass for all IRUnits.
///
/// The pass should by default run our mock analysis and then preserve it.
TEST_F(ModuleCallbacksTest, Passes) {
  EXPECT_CALL(AnalysisHandle, run(HasName("<string>"), _));
  EXPECT_CALL(PassHandle, run(HasName("<string>"), _))
      .WillOnce(&getAnalysisResult);

  StringRef PipelineText = "test-transform";
  ASSERT_THAT_ERROR(PB.parsePassPipeline(PM, PipelineText), Succeeded())
      << "Pipeline was: " << PipelineText;

  PM.run(*M, AM);
}

TEST_F(ModuleCallbacksTest, InstrumentedPasses) {
  EXPECT_CALL(AnalysisHandle, run(HasName("<string>"), _));
  EXPECT_CALL(PassHandle, run(HasName("<string>"), _))
      .WillOnce(&getAnalysisResult);

  CallbacksHandle.registerPassInstrumentation();
  // Non-mock instrumentation not specifically mentioned below can be ignored.
  CallbacksHandle.ignoreNonMockPassInstrumentation("<string>");

  // PassInstrumentation calls should happen in-sequence, in the same order
  // as passes/analyses are scheduled.
  ::testing::Sequence PISequence;
  EXPECT_CALL(CallbacksHandle, runBeforePass(HasNameRegex("MockPassHandle"),
                                             HasName("<string>")))
      .InSequence(PISequence);
  EXPECT_CALL(CallbacksHandle,
              runBeforeNonSkippedPass(HasNameRegex("MockPassHandle"),
                                      HasName("<string>")))
      .InSequence(PISequence);
  EXPECT_CALL(CallbacksHandle,
              runBeforeAnalysis(HasNameRegex("MockAnalysisHandle"),
                                HasName("<string>")))
      .InSequence(PISequence);
  EXPECT_CALL(
      CallbacksHandle,
      runAfterAnalysis(HasNameRegex("MockAnalysisHandle"), HasName("<string>")))
      .InSequence(PISequence);
  EXPECT_CALL(CallbacksHandle, runAfterPass(HasNameRegex("MockPassHandle"),
                                            HasName("<string>"), _))
      .InSequence(PISequence);

  // No passes are skipped, so there should be no calls to
  // runBeforeSkippedPass().
  EXPECT_CALL(
      CallbacksHandle,
      runBeforeSkippedPass(HasNameRegex("MockPassHandle"), HasName("<string>")))
      .Times(0);

  StringRef PipelineText = "test-transform";
  ASSERT_THAT_ERROR(PB.parsePassPipeline(PM, PipelineText), Succeeded())
      << "Pipeline was: " << PipelineText;

  PM.run(*M, AM);
}

TEST_F(ModuleCallbacksTest, InstrumentedSkippedPasses) {
  CallbacksHandle.registerPassInstrumentation();
  // Non-mock instrumentation run here can safely be ignored.
  CallbacksHandle.ignoreNonMockPassInstrumentation("<string>");
  CallbacksHandle.ignoreNonMockPassInstrumentation("foo");

  // Skip all passes by returning false. Pass managers and adaptor passes are
  // also passes that observed by the callbacks.
  EXPECT_CALL(CallbacksHandle, runBeforePass(_, _))
      .WillRepeatedly(Return(false));

  EXPECT_CALL(CallbacksHandle,
              runBeforeSkippedPass(HasNameRegex("MockPassHandle"), _))
      .Times(3);

  EXPECT_CALL(AnalysisHandle, run(HasName("<string>"), _)).Times(0);
  EXPECT_CALL(PassHandle, run(HasName("<string>"), _)).Times(0);

  // As the pass is skipped there is no nonskippedpass/afterPass,
  // beforeAnalysis/afterAnalysis as well.
  EXPECT_CALL(CallbacksHandle,
              runBeforeNonSkippedPass(HasNameRegex("MockPassHandle"), _))
      .Times(0);
  EXPECT_CALL(CallbacksHandle,
              runAfterPass(HasNameRegex("MockPassHandle"), _, _))
      .Times(0);
  EXPECT_CALL(CallbacksHandle,
              runBeforeAnalysis(HasNameRegex("MockAnalysisHandle"), _))
      .Times(0);
  EXPECT_CALL(CallbacksHandle,
              runAfterAnalysis(HasNameRegex("MockAnalysisHandle"), _))
      .Times(0);

  // Order is important here. `Adaptor` expectations should be checked first
  // because the its argument contains 'PassManager' (for example:
  // ModuleToFunctionPassAdaptor{{.*}}PassManager{{.*}}). Check
  // `runBeforeNonSkippedPass` and `runAfterPass` to show that they are not
  // skipped.
  //
  // Pass managers are not ignored.
  // 5 = (1) ModulePassManager + (2) FunctionPassMangers + (1) LoopPassManager +
  //     (1) CGSCCPassManager
  EXPECT_CALL(CallbacksHandle,
              runBeforeNonSkippedPass(HasNameRegex("PassManager"), _))
      .Times(5);
  EXPECT_CALL(
      CallbacksHandle,
      runBeforeNonSkippedPass(HasNameRegex("ModuleToFunctionPassAdaptor"), _))
      .Times(1);
  EXPECT_CALL(CallbacksHandle,
              runBeforeNonSkippedPass(
                  HasNameRegex("ModuleToPostOrderCGSCCPassAdaptor"), _))
      .Times(1);
  EXPECT_CALL(
      CallbacksHandle,
      runBeforeNonSkippedPass(HasNameRegex("CGSCCToFunctionPassAdaptor"), _))
      .Times(1);
  EXPECT_CALL(
      CallbacksHandle,
      runBeforeNonSkippedPass(HasNameRegex("FunctionToLoopPassAdaptor"), _))
      .Times(1);

  // The `runAfterPass` checks are the same as these of
  // `runBeforeNonSkippedPass`.
  EXPECT_CALL(CallbacksHandle, runAfterPass(HasNameRegex("PassManager"), _, _))
      .Times(5);
  EXPECT_CALL(CallbacksHandle,
              runAfterPass(HasNameRegex("ModuleToFunctionPassAdaptor"), _, _))
      .Times(1);
  EXPECT_CALL(
      CallbacksHandle,
      runAfterPass(HasNameRegex("ModuleToPostOrderCGSCCPassAdaptor"), _, _))
      .Times(1);
  EXPECT_CALL(CallbacksHandle,
              runAfterPass(HasNameRegex("CGSCCToFunctionPassAdaptor"), _, _))
      .Times(1);
  EXPECT_CALL(CallbacksHandle,
              runAfterPass(HasNameRegex("FunctionToLoopPassAdaptor"), _, _))
      .Times(1);

  // Ignore analyses introduced by adaptor passes.
  EXPECT_CALL(CallbacksHandle,
              runBeforeAnalysis(Not(HasNameRegex("MockAnalysisHandle")), _))
      .Times(AnyNumber());
  EXPECT_CALL(CallbacksHandle,
              runAfterAnalysis(Not(HasNameRegex("MockAnalysisHandle")), _))
      .Times(AnyNumber());

  // Register Funtion and Loop version of "test-transform" for testing
  PB.registerPipelineParsingCallback(
      [](StringRef Name, FunctionPassManager &FPM,
         ArrayRef<PassBuilder::PipelineElement>) {
        if (Name == "test-transform") {
          FPM.addPass(MockPassHandle<Function>().getPass());
          return true;
        }
        return false;
      });
  PB.registerPipelineParsingCallback(
      [](StringRef Name, LoopPassManager &LPM,
         ArrayRef<PassBuilder::PipelineElement>) {
        if (Name == "test-transform") {
          LPM.addPass(MockPassHandle<Loop>().getPass());
          return true;
        }
        return false;
      });

  StringRef PipelineText = "test-transform,function(test-transform),cgscc("
                           "function(loop(test-transform)))";
  ASSERT_THAT_ERROR(PB.parsePassPipeline(PM, PipelineText), Succeeded())
      << "Pipeline was: " << PipelineText;

  PM.run(*M, AM);
}

TEST_F(FunctionCallbacksTest, Passes) {
  EXPECT_CALL(AnalysisHandle, run(HasName("foo"), _));
  EXPECT_CALL(PassHandle, run(HasName("foo"), _)).WillOnce(&getAnalysisResult);

  StringRef PipelineText = "test-transform";
  ASSERT_THAT_ERROR(PB.parsePassPipeline(PM, PipelineText), Succeeded())
      << "Pipeline was: " << PipelineText;
  PM.run(*M, AM);
}

TEST_F(FunctionCallbacksTest, InstrumentedPasses) {
  CallbacksHandle.registerPassInstrumentation();
  // Non-mock instrumentation not specifically mentioned below can be ignored.
  CallbacksHandle.ignoreNonMockPassInstrumentation("<string>");
  CallbacksHandle.ignoreNonMockPassInstrumentation("foo");

  EXPECT_CALL(AnalysisHandle, run(HasName("foo"), _));
  EXPECT_CALL(PassHandle, run(HasName("foo"), _)).WillOnce(&getAnalysisResult);

  // PassInstrumentation calls should happen in-sequence, in the same order
  // as passes/analyses are scheduled.
  ::testing::Sequence PISequence;
  EXPECT_CALL(CallbacksHandle,
              runBeforePass(HasNameRegex("MockPassHandle"), HasName("foo")))
      .InSequence(PISequence);
  EXPECT_CALL(
      CallbacksHandle,
      runBeforeNonSkippedPass(HasNameRegex("MockPassHandle"), HasName("foo")))
      .InSequence(PISequence);
  EXPECT_CALL(
      CallbacksHandle,
      runBeforeAnalysis(HasNameRegex("MockAnalysisHandle"), HasName("foo")))
      .InSequence(PISequence);
  EXPECT_CALL(
      CallbacksHandle,
      runAfterAnalysis(HasNameRegex("MockAnalysisHandle"), HasName("foo")))
      .InSequence(PISequence);
  EXPECT_CALL(CallbacksHandle,
              runAfterPass(HasNameRegex("MockPassHandle"), HasName("foo"), _))
      .InSequence(PISequence);

  // No passes are skipped, so there should be no calls to
  // runBeforeSkippedPass().
  EXPECT_CALL(
      CallbacksHandle,
      runBeforeSkippedPass(HasNameRegex("MockPassHandle"), HasName("foo")))
      .Times(0);

  // Our mock pass does not invalidate IR.
  EXPECT_CALL(CallbacksHandle,
              runAfterPassInvalidated(HasNameRegex("MockPassHandle"), _))
      .Times(0);

  StringRef PipelineText = "test-transform";
  ASSERT_THAT_ERROR(PB.parsePassPipeline(PM, PipelineText), Succeeded())
      << "Pipeline was: " << PipelineText;
  PM.run(*M, AM);
}

TEST_F(FunctionCallbacksTest, InstrumentedSkippedPasses) {
  CallbacksHandle.registerPassInstrumentation();
  // Non-mock instrumentation run here can safely be ignored.
  CallbacksHandle.ignoreNonMockPassInstrumentation("<string>");
  CallbacksHandle.ignoreNonMockPassInstrumentation("foo");

  // Skip the pass by returning false.
  EXPECT_CALL(CallbacksHandle,
              runBeforePass(HasNameRegex("MockPassHandle"), HasName("foo")))
      .WillOnce(Return(false));

  EXPECT_CALL(
      CallbacksHandle,
      runBeforeSkippedPass(HasNameRegex("MockPassHandle"), HasName("foo")))
      .Times(1);

  EXPECT_CALL(AnalysisHandle, run(HasName("foo"), _)).Times(0);
  EXPECT_CALL(PassHandle, run(HasName("foo"), _)).Times(0);

  // As the pass is skipped there is no afterPass, beforeAnalysis/afterAnalysis
  // as well.
  EXPECT_CALL(CallbacksHandle,
              runBeforeNonSkippedPass(HasNameRegex("MockPassHandle"), _))
      .Times(0);
  EXPECT_CALL(CallbacksHandle,
              runAfterPass(HasNameRegex("MockPassHandle"), _, _))
      .Times(0);
  EXPECT_CALL(CallbacksHandle,
              runAfterPassInvalidated(HasNameRegex("MockPassHandle"), _))
      .Times(0);
  EXPECT_CALL(CallbacksHandle,
              runAfterPass(HasNameRegex("MockPassHandle"), _, _))
      .Times(0);
  EXPECT_CALL(CallbacksHandle,
              runBeforeAnalysis(HasNameRegex("MockAnalysisHandle"), _))
      .Times(0);
  EXPECT_CALL(CallbacksHandle,
              runAfterAnalysis(HasNameRegex("MockAnalysisHandle"), _))
      .Times(0);

  StringRef PipelineText = "test-transform";
  ASSERT_THAT_ERROR(PB.parsePassPipeline(PM, PipelineText), Succeeded())
      << "Pipeline was: " << PipelineText;
  PM.run(*M, AM);
}

TEST_F(LoopCallbacksTest, Passes) {
  EXPECT_CALL(AnalysisHandle, run(HasName("loop"), _, _));
  EXPECT_CALL(PassHandle, run(HasName("loop"), _, _, _))
      .WillOnce(WithArgs<0, 1, 2>(&getAnalysisResult));
  EXPECT_CALL(ExtraPassHandle, run(HasName("loop"), _, _, _));

  StringRef PipelineText = "test-transform";
  ASSERT_THAT_ERROR(PB.parsePassPipeline(PM, PipelineText), Succeeded())
      << "Pipeline was: " << PipelineText;
  PM.run(*M, AM);
}

TEST_F(LoopCallbacksTest, InstrumentedPasses) {
  CallbacksHandle.registerPassInstrumentation();
  // Non-mock instrumentation not specifically mentioned below can be ignored.
  CallbacksHandle.ignoreNonMockPassInstrumentation("<string>");
  CallbacksHandle.ignoreNonMockPassInstrumentation("foo");
  CallbacksHandle.ignoreNonMockPassInstrumentation("loop");

  EXPECT_CALL(AnalysisHandle, run(HasName("loop"), _, _));
  EXPECT_CALL(PassHandle, run(HasName("loop"), _, _, _))
      .WillOnce(WithArgs<0, 1, 2>(&getAnalysisResult));
  EXPECT_CALL(ExtraPassHandle, run(HasName("loop"), _, _, _));

  // PassInstrumentation calls should happen in-sequence, in the same order
  // as passes/analyses are scheduled.
  ::testing::Sequence PISequence;
  EXPECT_CALL(CallbacksHandle,
              runBeforePass(HasNameRegex("MockPassHandle"), HasName("loop")))
      .InSequence(PISequence);
  EXPECT_CALL(
      CallbacksHandle,
      runBeforeNonSkippedPass(HasNameRegex("MockPassHandle"), HasName("loop")))
      .InSequence(PISequence);
  EXPECT_CALL(
      CallbacksHandle,
      runBeforeAnalysis(HasNameRegex("MockAnalysisHandle"), HasName("loop")))
      .InSequence(PISequence);
  EXPECT_CALL(
      CallbacksHandle,
      runAfterAnalysis(HasNameRegex("MockAnalysisHandle"), HasName("loop")))
      .InSequence(PISequence);
  EXPECT_CALL(CallbacksHandle,
              runAfterPass(HasNameRegex("MockPassHandle"), HasName("loop"), _))
      .InSequence(PISequence);

  EXPECT_CALL(CallbacksHandle,
              runBeforePass(HasNameRegex("MockPassHandle<.*LoopNest>"),
                            HasName("loop")))
      .InSequence(PISequence);
  EXPECT_CALL(CallbacksHandle,
              runBeforeNonSkippedPass(
                  HasNameRegex("MockPassHandle<.*LoopNest>"), HasName("loop")))
      .InSequence(PISequence);
  EXPECT_CALL(CallbacksHandle,
              runAfterPass(HasNameRegex("MockPassHandle<.*LoopNest>"),
                           HasName("loop"), _))
      .InSequence(PISequence);

  // Our mock pass does not invalidate IR.
  EXPECT_CALL(CallbacksHandle,
              runAfterPassInvalidated(HasNameRegex("MockPassHandle"), _))
      .Times(0);

  // No passes are skipped, so there should be no calls to
  // runBeforeSkippedPass().
  EXPECT_CALL(
      CallbacksHandle,
      runBeforeSkippedPass(HasNameRegex("MockPassHandle"), HasName("loop")))
      .Times(0);

  StringRef PipelineText = "test-transform";
  ASSERT_THAT_ERROR(PB.parsePassPipeline(PM, PipelineText), Succeeded())
      << "Pipeline was: " << PipelineText;
  PM.run(*M, AM);
}

TEST_F(LoopCallbacksTest, InstrumentedInvalidatingPasses) {
  CallbacksHandle.registerPassInstrumentation();
  // Non-mock instrumentation not specifically mentioned below can be ignored.
  CallbacksHandle.ignoreNonMockPassInstrumentation("<string>");
  CallbacksHandle.ignoreNonMockPassInstrumentation("foo");
  CallbacksHandle.ignoreNonMockPassInstrumentation("loop");

  EXPECT_CALL(AnalysisHandle, run(HasName("loop"), _, _));
  EXPECT_CALL(PassHandle, run(HasName("loop"), _, _, _))
      .WillOnce(DoAll(WithArgs<0, 1, 2, 3>(&PassHandle.invalidateLoop),
                      WithArgs<0, 1, 2>(&getAnalysisResult)));

  // PassInstrumentation calls should happen in-sequence, in the same order
  // as passes/analyses are scheduled.
  ::testing::Sequence PISequence;
  EXPECT_CALL(CallbacksHandle,
              runBeforePass(HasNameRegex("MockPassHandle"), HasName("loop")))
      .InSequence(PISequence);
  EXPECT_CALL(
      CallbacksHandle,
      runBeforeNonSkippedPass(HasNameRegex("MockPassHandle"), HasName("loop")))
      .InSequence(PISequence);
  EXPECT_CALL(
      CallbacksHandle,
      runBeforeAnalysis(HasNameRegex("MockAnalysisHandle"), HasName("loop")))
      .InSequence(PISequence);
  EXPECT_CALL(
      CallbacksHandle,
      runAfterAnalysis(HasNameRegex("MockAnalysisHandle"), HasName("loop")))
      .InSequence(PISequence);
  EXPECT_CALL(CallbacksHandle,
              runAfterPassInvalidated(HasNameRegex("MockPassHandle"), _))
      .InSequence(PISequence);
  EXPECT_CALL(CallbacksHandle,
              runAfterPassInvalidated(HasNameRegex("^PassManager"), _))
      .InSequence(PISequence);

  // Our mock pass invalidates IR, thus normal runAfterPass is never called.
  EXPECT_CALL(CallbacksHandle,
              runAfterPass(HasNameRegex("MockPassHandle"), HasName("loop"), _))
      .Times(0);

  StringRef PipelineText = "test-transform";
  ASSERT_THAT_ERROR(PB.parsePassPipeline(PM, PipelineText), Succeeded())
      << "Pipeline was: " << PipelineText;
  PM.run(*M, AM);
}

TEST_F(LoopCallbacksTest, InstrumentedInvalidatingLoopNestPasses) {
  CallbacksHandle.registerPassInstrumentation();
  // Non-mock instrumentation not specifically mentioned below can be ignored.
  CallbacksHandle.ignoreNonMockPassInstrumentation("<string>");
  CallbacksHandle.ignoreNonMockPassInstrumentation("foo");
  CallbacksHandle.ignoreNonMockPassInstrumentation("loop");

  EXPECT_CALL(AnalysisHandle, run(HasName("loop"), _, _));
  EXPECT_CALL(PassHandle, run(HasName("loop"), _, _, _))
      .WillOnce(WithArgs<0, 1, 2>(&getAnalysisResult));
  EXPECT_CALL(ExtraPassHandle, run(HasName("loop"), _, _, _))
      .WillOnce(DoAll(&ExtraPassHandle.invalidateLoopNest,
                      [&](LoopNest &, LoopAnalysisManager &,
                          LoopStandardAnalysisResults &,
                          LPMUpdater &) { return PreservedAnalyses::all(); }));

  // PassInstrumentation calls should happen in-sequence, in the same order
  // as passes/analyses are scheduled.
  ::testing::Sequence PISequence;
  EXPECT_CALL(CallbacksHandle,
              runBeforePass(HasNameRegex("MockPassHandle"), HasName("loop")))
      .InSequence(PISequence);
  EXPECT_CALL(
      CallbacksHandle,
      runBeforeNonSkippedPass(HasNameRegex("MockPassHandle"), HasName("loop")))
      .InSequence(PISequence);
  EXPECT_CALL(
      CallbacksHandle,
      runBeforeAnalysis(HasNameRegex("MockAnalysisHandle"), HasName("loop")))
      .InSequence(PISequence);
  EXPECT_CALL(
      CallbacksHandle,
      runAfterAnalysis(HasNameRegex("MockAnalysisHandle"), HasName("loop")))
      .InSequence(PISequence);
  EXPECT_CALL(CallbacksHandle,
              runAfterPass(HasNameRegex("MockPassHandle"), HasName("loop"), _))
      .InSequence(PISequence);

  EXPECT_CALL(CallbacksHandle,
              runBeforePass(HasNameRegex("MockPassHandle<.*LoopNest>"),
                            HasName("loop")))
      .InSequence(PISequence);
  EXPECT_CALL(CallbacksHandle,
              runBeforeNonSkippedPass(
                  HasNameRegex("MockPassHandle<.*LoopNest>"), HasName("loop")))
      .InSequence(PISequence);
  EXPECT_CALL(
      CallbacksHandle,
      runAfterPassInvalidated(HasNameRegex("MockPassHandle<.*LoopNest>"), _))
      .InSequence(PISequence);

  EXPECT_CALL(CallbacksHandle,
              runAfterPassInvalidated(HasNameRegex("^PassManager"), _))
      .InSequence(PISequence);

  // Our mock pass invalidates IR, thus normal runAfterPass is never called.
  EXPECT_CALL(CallbacksHandle, runAfterPassInvalidated(
                                   HasNameRegex("MockPassHandle<.*Loop>"), _))
      .Times(0);
  EXPECT_CALL(CallbacksHandle,
              runAfterPass(HasNameRegex("MockPassHandle<.*LoopNest>"),
                           HasName("loop"), _))
      .Times(0);

  StringRef PipelineText = "test-transform";
  ASSERT_THAT_ERROR(PB.parsePassPipeline(PM, PipelineText), Succeeded())
      << "Pipeline was: " << PipelineText;
  PM.run(*M, AM);
}

TEST_F(LoopCallbacksTest, InstrumentedSkippedPasses) {
  CallbacksHandle.registerPassInstrumentation();
  // Non-mock instrumentation run here can safely be ignored.
  CallbacksHandle.ignoreNonMockPassInstrumentation("<string>");
  CallbacksHandle.ignoreNonMockPassInstrumentation("foo");
  CallbacksHandle.ignoreNonMockPassInstrumentation("loop");

  // Skip the pass by returning false.
  EXPECT_CALL(
      CallbacksHandle,
      runBeforePass(HasNameRegex("MockPassHandle<.*Loop>"), HasName("loop")))
      .WillOnce(Return(false));

  EXPECT_CALL(CallbacksHandle,
              runBeforeSkippedPass(HasNameRegex("MockPassHandle<.*Loop>"),
                                   HasName("loop")))
      .Times(1);

  EXPECT_CALL(CallbacksHandle,
              runBeforePass(HasNameRegex("MockPassHandle<.*LoopNest>"),
                            HasName("loop")))
      .WillOnce(Return(false));

  EXPECT_CALL(CallbacksHandle,
              runBeforeSkippedPass(HasNameRegex("MockPassHandle<.*LoopNest>"),
                                   HasName("loop")))
      .Times(1);

  EXPECT_CALL(AnalysisHandle, run(HasName("loop"), _, _)).Times(0);
  EXPECT_CALL(PassHandle, run(HasName("loop"), _, _, _)).Times(0);
  EXPECT_CALL(ExtraPassHandle, run(HasName("loop"), _, _, _)).Times(0);

  // As the pass is skipped there is no afterPass, beforeAnalysis/afterAnalysis
  // as well.
  EXPECT_CALL(CallbacksHandle, runBeforeNonSkippedPass(
                                   HasNameRegex("MockPassHandle<.*Loop>"), _))
      .Times(0);
  EXPECT_CALL(CallbacksHandle,
              runAfterPass(HasNameRegex("MockPassHandle<.*Loop>"), _, _))
      .Times(0);
  EXPECT_CALL(CallbacksHandle, runAfterPassInvalidated(
                                   HasNameRegex("MockPassHandle<.*Loop>"), _))
      .Times(0);
  EXPECT_CALL(
      CallbacksHandle,
      runBeforeNonSkippedPass(HasNameRegex("MockPassHandle<.*LoopNest>"), _))
      .Times(0);
  EXPECT_CALL(CallbacksHandle,
              runAfterPass(HasNameRegex("MockPassHandle<.*LoopNest>"), _, _))
      .Times(0);
  EXPECT_CALL(
      CallbacksHandle,
      runAfterPassInvalidated(HasNameRegex("MockPassHandle<.*LoopNest>"), _))
      .Times(0);
  EXPECT_CALL(CallbacksHandle,
              runBeforeAnalysis(HasNameRegex("MockAnalysisHandle"), _))
      .Times(0);
  EXPECT_CALL(CallbacksHandle,
              runAfterAnalysis(HasNameRegex("MockAnalysisHandle"), _))
      .Times(0);

  StringRef PipelineText = "test-transform";
  ASSERT_THAT_ERROR(PB.parsePassPipeline(PM, PipelineText), Succeeded())
      << "Pipeline was: " << PipelineText;
  PM.run(*M, AM);
}

TEST_F(CGSCCCallbacksTest, Passes) {
  EXPECT_CALL(AnalysisHandle, run(HasName("(foo)"), _, _));
  EXPECT_CALL(PassHandle, run(HasName("(foo)"), _, _, _))
      .WillOnce(WithArgs<0, 1, 2>(&getAnalysisResult));

  StringRef PipelineText = "test-transform";
  ASSERT_THAT_ERROR(PB.parsePassPipeline(PM, PipelineText), Succeeded())
      << "Pipeline was: " << PipelineText;
  PM.run(*M, AM);
}

TEST_F(CGSCCCallbacksTest, InstrumentedPasses) {
  CallbacksHandle.registerPassInstrumentation();
  // Non-mock instrumentation not specifically mentioned below can be ignored.
  CallbacksHandle.ignoreNonMockPassInstrumentation("<string>");
  CallbacksHandle.ignoreNonMockPassInstrumentation("foo");
  CallbacksHandle.ignoreNonMockPassInstrumentation("(foo)");

  EXPECT_CALL(AnalysisHandle, run(HasName("(foo)"), _, _));
  EXPECT_CALL(PassHandle, run(HasName("(foo)"), _, _, _))
      .WillOnce(WithArgs<0, 1, 2>(&getAnalysisResult));

  // PassInstrumentation calls should happen in-sequence, in the same order
  // as passes/analyses are scheduled.
  ::testing::Sequence PISequence;
  EXPECT_CALL(CallbacksHandle,
              runBeforePass(HasNameRegex("MockPassHandle"), HasName("(foo)")))
      .InSequence(PISequence);
  EXPECT_CALL(
      CallbacksHandle,
      runBeforeNonSkippedPass(HasNameRegex("MockPassHandle"), HasName("(foo)")))
      .InSequence(PISequence);
  EXPECT_CALL(
      CallbacksHandle,
      runBeforeAnalysis(HasNameRegex("MockAnalysisHandle"), HasName("(foo)")))
      .InSequence(PISequence);
  EXPECT_CALL(
      CallbacksHandle,
      runAfterAnalysis(HasNameRegex("MockAnalysisHandle"), HasName("(foo)")))
      .InSequence(PISequence);
  EXPECT_CALL(CallbacksHandle,
              runAfterPass(HasNameRegex("MockPassHandle"), HasName("(foo)"), _))
      .InSequence(PISequence);

  // Our mock pass does not invalidate IR.
  EXPECT_CALL(CallbacksHandle,
              runAfterPassInvalidated(HasNameRegex("MockPassHandle"), _))
      .Times(0);

  // No passes are skipped, so there should be no calls to
  // runBeforeSkippedPass().
  EXPECT_CALL(
      CallbacksHandle,
      runBeforeSkippedPass(HasNameRegex("MockPassHandle"), HasName("(foo)")))
      .Times(0);

  StringRef PipelineText = "test-transform";
  ASSERT_THAT_ERROR(PB.parsePassPipeline(PM, PipelineText), Succeeded())
      << "Pipeline was: " << PipelineText;
  PM.run(*M, AM);
}

TEST_F(CGSCCCallbacksTest, InstrumentedInvalidatingPasses) {
  CallbacksHandle.registerPassInstrumentation();
  // Non-mock instrumentation not specifically mentioned below can be ignored.
  CallbacksHandle.ignoreNonMockPassInstrumentation("<string>");
  CallbacksHandle.ignoreNonMockPassInstrumentation("foo");
  CallbacksHandle.ignoreNonMockPassInstrumentation("(foo)");

  EXPECT_CALL(AnalysisHandle, run(HasName("(foo)"), _, _));
  EXPECT_CALL(PassHandle, run(HasName("(foo)"), _, _, _))
      .WillOnce(DoAll(WithArgs<0, 1, 2, 3>(&PassHandle.invalidateSCC),
                      WithArgs<0, 1, 2>(&getAnalysisResult)));

  // PassInstrumentation calls should happen in-sequence, in the same order
  // as passes/analyses are scheduled.
  ::testing::Sequence PISequence;
  EXPECT_CALL(CallbacksHandle,
              runBeforePass(HasNameRegex("MockPassHandle"), HasName("(foo)")))
      .InSequence(PISequence);
  EXPECT_CALL(
      CallbacksHandle,
      runBeforeNonSkippedPass(HasNameRegex("MockPassHandle"), HasName("(foo)")))
      .InSequence(PISequence);
  EXPECT_CALL(
      CallbacksHandle,
      runBeforeAnalysis(HasNameRegex("MockAnalysisHandle"), HasName("(foo)")))
      .InSequence(PISequence);
  EXPECT_CALL(
      CallbacksHandle,
      runAfterAnalysis(HasNameRegex("MockAnalysisHandle"), HasName("(foo)")))
      .InSequence(PISequence);
  EXPECT_CALL(CallbacksHandle,
              runAfterPassInvalidated(HasNameRegex("MockPassHandle"), _))
      .InSequence(PISequence);
  EXPECT_CALL(CallbacksHandle,
              runAfterPassInvalidated(HasNameRegex("^PassManager"), _))
      .InSequence(PISequence);

  // Our mock pass does invalidate IR, thus normal runAfterPass is never called.
  EXPECT_CALL(CallbacksHandle,
              runAfterPass(HasNameRegex("MockPassHandle"), HasName("(foo)"), _))
      .Times(0);

  StringRef PipelineText = "test-transform";
  ASSERT_THAT_ERROR(PB.parsePassPipeline(PM, PipelineText), Succeeded())
      << "Pipeline was: " << PipelineText;
  PM.run(*M, AM);
}

TEST_F(CGSCCCallbacksTest, InstrumentedSkippedPasses) {
  CallbacksHandle.registerPassInstrumentation();
  // Non-mock instrumentation run here can safely be ignored.
  CallbacksHandle.ignoreNonMockPassInstrumentation("<string>");
  CallbacksHandle.ignoreNonMockPassInstrumentation("foo");
  CallbacksHandle.ignoreNonMockPassInstrumentation("(foo)");

  // Skip the pass by returning false.
  EXPECT_CALL(CallbacksHandle,
              runBeforePass(HasNameRegex("MockPassHandle"), HasName("(foo)")))
      .WillOnce(Return(false));

  EXPECT_CALL(
      CallbacksHandle,
      runBeforeSkippedPass(HasNameRegex("MockPassHandle"), HasName("(foo)")))
      .Times(1);

  // neither Analysis nor Pass are called.
  EXPECT_CALL(AnalysisHandle, run(HasName("(foo)"), _, _)).Times(0);
  EXPECT_CALL(PassHandle, run(HasName("(foo)"), _, _, _)).Times(0);

  // As the pass is skipped there is no afterPass, beforeAnalysis/afterAnalysis
  // as well.
  EXPECT_CALL(CallbacksHandle,
              runBeforeNonSkippedPass(HasNameRegex("MockPassHandle"), _))
      .Times(0);
  EXPECT_CALL(CallbacksHandle,
              runAfterPass(HasNameRegex("MockPassHandle"), _, _))
      .Times(0);
  EXPECT_CALL(CallbacksHandle,
              runAfterPassInvalidated(HasNameRegex("MockPassHandle"), _))
      .Times(0);
  EXPECT_CALL(CallbacksHandle,
              runBeforeAnalysis(HasNameRegex("MockAnalysisHandle"), _))
      .Times(0);
  EXPECT_CALL(CallbacksHandle,
              runAfterAnalysis(HasNameRegex("MockAnalysisHandle"), _))
      .Times(0);

  StringRef PipelineText = "test-transform";
  ASSERT_THAT_ERROR(PB.parsePassPipeline(PM, PipelineText), Succeeded())
      << "Pipeline was: " << PipelineText;
  PM.run(*M, AM);
}

/// Test parsing of the names of analysis utilities for our mock analysis
/// for all IRUnits.
///
/// We first require<>, then invalidate<> it, expecting the analysis to be run
/// once and subsequently invalidated.
TEST_F(ModuleCallbacksTest, AnalysisUtilities) {
  EXPECT_CALL(AnalysisHandle, run(HasName("<string>"), _));
  EXPECT_CALL(AnalysisHandle, invalidate(HasName("<string>"), _, _));

  StringRef PipelineText = "require<test-analysis>,invalidate<test-analysis>";
  ASSERT_THAT_ERROR(PB.parsePassPipeline(PM, PipelineText), Succeeded())
      << "Pipeline was: " << PipelineText;
  PM.run(*M, AM);
}

TEST_F(CGSCCCallbacksTest, PassUtilities) {
  EXPECT_CALL(AnalysisHandle, run(HasName("(foo)"), _, _));
  EXPECT_CALL(AnalysisHandle, invalidate(HasName("(foo)"), _, _));

  StringRef PipelineText = "require<test-analysis>,invalidate<test-analysis>";
  ASSERT_THAT_ERROR(PB.parsePassPipeline(PM, PipelineText), Succeeded())
      << "Pipeline was: " << PipelineText;
  PM.run(*M, AM);
}

TEST_F(FunctionCallbacksTest, AnalysisUtilities) {
  EXPECT_CALL(AnalysisHandle, run(HasName("foo"), _));
  EXPECT_CALL(AnalysisHandle, invalidate(HasName("foo"), _, _));

  StringRef PipelineText = "require<test-analysis>,invalidate<test-analysis>";
  ASSERT_THAT_ERROR(PB.parsePassPipeline(PM, PipelineText), Succeeded())
      << "Pipeline was: " << PipelineText;
  PM.run(*M, AM);
}

TEST_F(LoopCallbacksTest, PassUtilities) {
  EXPECT_CALL(AnalysisHandle, run(HasName("loop"), _, _));
  EXPECT_CALL(AnalysisHandle, invalidate(HasName("loop"), _, _));

  StringRef PipelineText = "require<test-analysis>,invalidate<test-analysis>";

  ASSERT_THAT_ERROR(PB.parsePassPipeline(PM, PipelineText), Succeeded())
      << "Pipeline was: " << PipelineText;
  PM.run(*M, AM);
}

/// Test parsing of the top-level pipeline.
///
/// The ParseTopLevelPipeline callback takes over parsing of the entire pipeline
/// from PassBuilder if it encounters an unknown pipeline entry at the top level
/// (i.e., the first entry on the pipeline).
/// This test parses a pipeline named 'another-pipeline', whose only elements
/// may be the test-transform pass or the analysis utilities
TEST_F(ModuleCallbacksTest, ParseTopLevelPipeline) {
  PB.registerParseTopLevelPipelineCallback(
      [this](ModulePassManager &MPM,
             ArrayRef<PassBuilder::PipelineElement> Pipeline) {
        auto &FirstName = Pipeline.front().Name;
        auto &InnerPipeline = Pipeline.front().InnerPipeline;
        if (FirstName == "another-pipeline") {
          for (auto &E : InnerPipeline) {
            if (parseAnalysisUtilityPasses<AnalysisT>("test-analysis", E.Name,
                                                      PM))
              continue;

            if (E.Name == "test-transform") {
              PM.addPass(PassHandle.getPass());
              continue;
            }
            return false;
          }
        }
        return true;
      });

  EXPECT_CALL(AnalysisHandle, run(HasName("<string>"), _));
  EXPECT_CALL(PassHandle, run(HasName("<string>"), _))
      .WillOnce(&getAnalysisResult);
  EXPECT_CALL(AnalysisHandle, invalidate(HasName("<string>"), _, _));

  StringRef PipelineText =
      "another-pipeline(test-transform,invalidate<test-analysis>)";
  ASSERT_THAT_ERROR(PB.parsePassPipeline(PM, PipelineText), Succeeded())
      << "Pipeline was: " << PipelineText;
  PM.run(*M, AM);

  /// Test the negative case
  PipelineText = "another-pipeline(instcombine)";
  ASSERT_THAT_ERROR(PB.parsePassPipeline(PM, PipelineText), Failed())
      << "Pipeline was: " << PipelineText;
}
} // end anonymous namespace
