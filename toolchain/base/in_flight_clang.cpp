// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/base/in_flight_clang.h"

#include <memory>
#include <mutex>

#include "clang/Basic/Stack.h"
#include "clang/CodeGen/CodeGenAction.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/MultiplexConsumer.h"
#include "clang/Frontend/TextDiagnostic.h"
#include "clang/Sema/Lookup.h"
#include "clang/include/clang/FrontendTool/Utils.h"
#include "common/check.h"
#include "llvm/IR/Module.h"

namespace Carbon {

// Facilitates communication between a dedicated producer thread that runs Clang
// to build AST and do code generation, and a consumer thread that
// interacts with AST during 'Check' and later consumes the resulting
// `llvm::Module` during 'Lower'.
//
// The channel is essentially a state machine that moves forward, but not
// backwards. Calls to functions in the API normally happens in the order of
// declarations in the code, but some intermediate calls can be skipped in case
// of errors.
//
// At any point after the first call, only one of the threads is actively
// running and the data passed around can be used without extra synchronization,
// as long as the API of the channel is being followed.
struct InFlightClang::AstChannel {
  // Called by producer thread when AST is ready.
  // Can be skipped if AST could not be produced due to an error.
  void reportASTAndWaitForCodeGenRequest(clang::CodeGenAction* action,
                                         clang::Sema* sema) {
    std::unique_lock<std::mutex> lock(mut_);
    CARBON_CHECK(state_ == State::Created);

    action_ = action;
    sema_ = sema;

    progressTo(State::AstReady);
    waitUntil(lock, State::CodeGenFinishRequested);
  }

  struct AstAndAction {
    clang::CodeGenAction* action = nullptr;
    clang::Sema* sema = nullptr;
  };
  // Called by the consumer thread.
  auto waitForAST() -> AstAndAction {
    std::unique_lock<std::mutex> lock(mut_);
    waitUntil(lock, State::AstReady);
    return {.action = action_, .sema = sema_};
  }

  // Called by producer thread when CodeGen is finished and results can be
  // consumed from the CodeGenAction passed in previous callbacks.
  void notifyCodeGenFinished() {
    std::unique_lock<std::mutex> lock(mut_);
    progressTo(State::CodeGenFinished);
    waitUntil(lock, State::WorkerFinishRequested);
  }

  // Called by consumer thread.
  void waitForCodeGenFinished() {
    std::unique_lock<std::mutex> lock(mut_);
    progressTo(State::CodeGenFinishRequested);
    waitUntil(lock, State::CodeGenFinished);
  }

  // Called by the consumer thread. AST and other data structures will be
  // cleaned up after this call returns and should not be used.
  void waitForWorkerFinished() {
    std::unique_lock<std::mutex> lock(mut_);
    progressTo(State::WorkerFinishRequested);
    waitUntil(lock, State::WorkedFinished);
  }

  // Called by producer thread.
  void nofityWorkerFinished() {
    std::unique_lock<std::mutex> lock(mut_);
    progressTo(State::WorkedFinished);
  }

 private:
  std::mutex mut_;
  std::condition_variable cond_;

  clang::CodeGenAction* action_ = nullptr;
  clang::Sema* sema_ = nullptr;

  enum State {
    Created = 0,
    AstReady,
    CodeGenFinishRequested,
    CodeGenFinished,
    WorkerFinishRequested,
    WorkedFinished,
  };
  State state_ = State::Created;

  void progressTo(State s) {
    if (state_ < s) {
      state_ = s;
      cond_.notify_all();
    }
  }

  void waitUntil(std::unique_lock<std::mutex>& lock, State s) {
    cond_.wait(lock, [&] { return s <= state_; });
  }
};

InFlightClang::InFlightClang(clang::Sema* sema, clang::CodeGenAction* action,
                             std::unique_ptr<AstChannel> chan,
                             std::thread worker)
    : sema_(sema),
      action_(action),
      chan_(std::move(chan)),
      worker_(std::move(worker)) {}

auto InFlightClang::getASTContext() -> clang::ASTContext& {
  return getSema().getASTContext();
}

auto InFlightClang::getSourceManager() -> clang::SourceManager& {
  return getSema().getSourceManager();
}

auto InFlightClang::getSourceManager() const -> const clang::SourceManager& {
  CARBON_CHECK(sema_ != nullptr);
  return sema_->getSourceManager();
}

auto InFlightClang::getSema() -> clang::Sema& {
  CARBON_CHECK(sema_ != nullptr);
  return *sema_;
}

auto InFlightClang::takeLLVMContext() -> std::unique_ptr<llvm::LLVMContext> {
  CARBON_CHECK(action_ != nullptr);
  CARBON_CHECK(!llvm_context_taken_);
  llvm_context_taken_ = true;
  return std::unique_ptr<llvm::LLVMContext>(action_->takeLLVMContext());
}

auto InFlightClang::getCodeGenerator() const -> clang::CodeGenerator& {
  CARBON_CHECK(action_ != nullptr, "action is null");
  return *action_->getCodeGenerator();
}

auto InFlightClang::finishCompilation() && -> std::unique_ptr<llvm::Module> {
  chan_->waitForCodeGenFinished();
  return action_->takeModule();
}

namespace {
// The minimal set of extensions points of Clang that Carbon needs.
class ClangCallbacks {
 public:
  virtual void onOriginalFrontendActionCreated(
      clang::FrontendAction& action) = 0;
  virtual void handleTranslationUnit(clang::Sema& sema) = 0;
  virtual void onCompilationFinished() = 0;
};
}  // namespace

static auto PrepareCompilation(
    llvm::ArrayRef<const char*> argv, llvm::StringRef target,
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs,
    std::unique_ptr<clang::DiagnosticConsumer> consumer)
    -> std::unique_ptr<clang::CompilerInstance> {
  auto compiler = std::make_unique<clang::CompilerInstance>();
  compiler->createDiagnostics(*fs, consumer.release(), /*ShouldOwn=*/true);

  clang::driver::Driver driver(argv[0], target, compiler->getDiagnostics(),
                               "clang LLVM compiler", fs);
  std::unique_ptr<clang::driver::Compilation> compilation(
      driver.BuildCompilation(argv));
  if (!compilation) {
    return nullptr;
  }
  const clang::driver::JobList& jobs = compilation->getJobs();
  if (jobs.size() != 1) {
    // CARBON_VLOG("got multiple jobs with '{}'", argv);
    return nullptr;
  }
  auto& job = *jobs.begin();
  if (job.getSource().getKind() != clang::driver::Action::AssembleJobClass) {
    // CARBON_VLOG("expected an assemble job class with '{}'", argv);
    return nullptr;
  }
  if (!clang::CompilerInvocation::CreateFromArgs(
          compiler->getInvocation(), job.getArguments(),
          compiler->getDiagnostics(), argv[0])) {
    return nullptr;
  }
  if (!compiler->createFileManager(clang::createVFSFromCompilerInvocation(
          compiler->getInvocation(), compiler->getDiagnostics(), fs))) {
    // CARBON_VLOG("failed to create a file manager with '{}'", argv);
    return nullptr;
  }
  // Prevent compiler from outputing various status messages, e.g. '3 errors
  // generated'.
  compiler->setVerboseOutputStream(llvm::nulls());
  // Do not produce the output, we will process the llvm::Module ourselves.
  // TODO: prevent machine code generation from running completely.
  compiler->setOutputStream(std::make_unique<llvm::raw_null_ostream>());
  return compiler;
}

static auto RunCompilation(clang::CompilerInstance& compiler,
                           ClangCallbacks& callbacks) {
  class InFlightClangASTConsumer : public clang::SemaConsumer {
   public:
    explicit InFlightClangASTConsumer(ClangCallbacks& callbacks)
        : callbacks_(callbacks), active_sema_(nullptr) {}

    void InitializeSema(clang::Sema& sema) override { active_sema_ = &sema; }
    void ForgetSema() override { active_sema_ = nullptr; }

    void HandleTranslationUnit(clang::ASTContext& /*ast*/) override {
      CARBON_CHECK(active_sema_ != nullptr);
      callbacks_.handleTranslationUnit(*active_sema_);
    }

   private:
    ClangCallbacks& callbacks_;
    clang::Sema* active_sema_;
  };

  class InFlightClangFrontendAction : public clang::WrapperFrontendAction {
   public:
    explicit InFlightClangFrontendAction(
        ClangCallbacks& callbacks, std::unique_ptr<FrontendAction> wrapped)
        : clang::WrapperFrontendAction(std::move(wrapped)),
          callbacks_(callbacks) {}

    auto CreateASTConsumer(clang::CompilerInstance& compiler,
                           llvm::StringRef file)
        -> std::unique_ptr<clang::ASTConsumer> override {
      std::vector<std::unique_ptr<clang::ASTConsumer>> consumers;
      consumers.push_back(
          std::make_unique<InFlightClangASTConsumer>(callbacks_));
      if (auto c = WrapperFrontendAction::CreateASTConsumer(compiler, file)) {
        consumers.push_back(std::move(c));
      }
      return std::make_unique<clang::MultiplexConsumer>(std::move(consumers));
    }

   private:
    ClangCallbacks& callbacks_;
  };

  auto original_action = clang::CreateFrontendAction(compiler);
  if (!original_action) {
    return;
  }
  callbacks.onOriginalFrontendActionCreated(*original_action);

  auto action = std::make_unique<InFlightClangFrontendAction>(
      callbacks, std::move(original_action));
  compiler.ExecuteAction(*action);

  callbacks.onCompilationFinished();
}

auto InFlightClang::CompileFromArguments(
    llvm::ArrayRef<const char*> argv, llvm::StringRef target,
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs,
    std::unique_ptr<clang::DiagnosticConsumer> consumer)
    -> std::unique_ptr<InFlightClang> {
  auto compiler = PrepareCompilation(argv, target, fs, std::move(consumer));
  if (!compiler) {
    return nullptr;
  }

  class Callbacks : public ClangCallbacks {
   public:
    explicit Callbacks(AstChannel& data) : chan_(data) {}

    void onOriginalFrontendActionCreated(
        clang::FrontendAction& action) override {
      CARBON_CHECK(action_ == nullptr, "expected a single call");
      action_ = dynamic_cast<clang::CodeGenAction*>(&action);
      CARBON_CHECK(action_ != nullptr, "expected a CodeGenAction");
    }

    void handleTranslationUnit(clang::Sema& sema) override {
      chan_.reportASTAndWaitForCodeGenRequest(action_, &sema);
    }

    virtual void onCompilationFinished() override {
      chan_.notifyCodeGenFinished();
    }

   private:
    AstChannel& chan_;
    clang::CodeGenAction* action_ = nullptr;
  };

  auto data = std::make_unique<AstChannel>();
  // Bridge the Clang's callback-based API into staged API by using a worker
  // thread. The compilation on the worker will:
  // 1. produce an AST and pass it back to us,
  // 2. wait for signal from InFlightClang's destructor,
  // 3. finish any necessary cleanups and terminate.
  std::thread worker(
      [compiler = std::move(compiler), data = data.get()]() mutable {
        // TODO: ensure the created thread gets a large stack.
        clang::noteBottomOfStack();

        Callbacks callbacks(*data);
        // In that call, callbacks will pass control to consumer thread.
        RunCompilation(*compiler, callbacks);

        // This cleans up AST and other Clang data structures.
        // Note: Clang sets DisableFree by default, so most memory is leaked.
        compiler.reset();

        data->nofityWorkerFinished();
      });

  auto [action, sema] = data->waitForAST();
  // The cleanup of the worker thread is in destructor of InFlightClang.
  auto ret = std::unique_ptr<InFlightClang>(
      new InFlightClang(sema, action, std::move(data), std::move(worker)));
  if (!sema) {
    // CARBON_VLOG("failed to create the AST");
    return nullptr;
  }
  return ret;
}

InFlightClang::~InFlightClang() {
  chan_->waitForWorkerFinished();
  worker_.join();
}

}  // namespace Carbon
