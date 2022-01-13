//===- MLIRContext.h - MLIR Global Context Class ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_MLIRCONTEXT_H
#define MLIR_IR_MLIRCONTEXT_H

#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include <functional>
#include <memory>
#include <vector>

namespace llvm {
class ThreadPool;
} // end namespace llvm

namespace mlir {
class AbstractOperation;
class DebugActionManager;
class DiagnosticEngine;
class Dialect;
class DialectRegistry;
class InFlightDiagnostic;
class Location;
class MLIRContextImpl;
class StorageUniquer;

/// MLIRContext is the top-level object for a collection of MLIR operations. It
/// holds immortal uniqued objects like types, and the tables used to unique
/// them.
///
/// MLIRContext gets a redundant "MLIR" prefix because otherwise it ends up with
/// a very generic name ("Context") and because it is uncommon for clients to
/// interact with it.
///
/// The context wrap some multi-threading facilities, and in particular by
/// default it will implicitly create a thread pool.
/// This can be undesirable if multiple context exists at the same time or if a
/// process will be long-lived and create and destroy contexts.
/// To control better thread spawning, an externally owned ThreadPool can be
/// injected in the context. For example:
///
///  llvm::ThreadPool myThreadPool;
///  while (auto *request = nextCompilationRequests()) {
///    MLIRContext ctx(registry, MLIRContext::Threading::DISABLED);
///    ctx.setThreadPool(myThreadPool);
///    processRequest(request, cxt);
///  }
///
class MLIRContext {
public:
  enum class Threading { DISABLED, ENABLED };
  /// Create a new Context.
  explicit MLIRContext(Threading multithreading = Threading::ENABLED);
  explicit MLIRContext(const DialectRegistry &registry,
                       Threading multithreading = Threading::ENABLED);
  ~MLIRContext();

  /// Return information about all IR dialects loaded in the context.
  std::vector<Dialect *> getLoadedDialects();

  /// Return the dialect registry associated with this context.
  const DialectRegistry &getDialectRegistry();

  /// Append the contents of the given dialect registry to the registry
  /// associated with this context.
  void appendDialectRegistry(const DialectRegistry &registry);

  /// Return information about all available dialects in the registry in this
  /// context.
  std::vector<StringRef> getAvailableDialects();

  /// Get a registered IR dialect with the given namespace. If an exact match is
  /// not found, then return nullptr.
  Dialect *getLoadedDialect(StringRef name);

  /// Get a registered IR dialect for the given derived dialect type. The
  /// derived type must provide a static 'getDialectNamespace' method.
  template <typename T>
  T *getLoadedDialect() {
    return static_cast<T *>(getLoadedDialect(T::getDialectNamespace()));
  }

  /// Get (or create) a dialect for the given derived dialect type. The derived
  /// type must provide a static 'getDialectNamespace' method.
  template <typename T>
  T *getOrLoadDialect() {
    return static_cast<T *>(
        getOrLoadDialect(T::getDialectNamespace(), TypeID::get<T>(), [this]() {
          std::unique_ptr<T> dialect(new T(this));
          return dialect;
        }));
  }

  /// Load a dialect in the context.
  template <typename Dialect>
  void loadDialect() {
    getOrLoadDialect<Dialect>();
  }

  /// Load a list dialects in the context.
  template <typename Dialect, typename OtherDialect, typename... MoreDialects>
  void loadDialect() {
    getOrLoadDialect<Dialect>();
    loadDialect<OtherDialect, MoreDialects...>();
  }

  /// Load all dialects available in the registry in this context.
  void loadAllAvailableDialects();

  /// Get (or create) a dialect for the given derived dialect name.
  /// The dialect will be loaded from the registry if no dialect is found.
  /// If no dialect is loaded for this name and none is available in the
  /// registry, returns nullptr.
  Dialect *getOrLoadDialect(StringRef name);

  /// Return true if we allow to create operation for unregistered dialects.
  bool allowsUnregisteredDialects();

  /// Enables creating operations in unregistered dialects.
  void allowUnregisteredDialects(bool allow = true);

  /// Return true if multi-threading is enabled by the context.
  bool isMultithreadingEnabled();

  /// Set the flag specifying if multi-threading is disabled by the context.
  void disableMultithreading(bool disable = true);
  void enableMultithreading(bool enable = true) {
    disableMultithreading(!enable);
  }

  /// Set a new thread pool to be used in this context. This method requires
  /// that multithreading is disabled for this context prior to the call. This
  /// allows to share a thread pool across multiple contexts, as well as
  /// decoupling the lifetime of the threads from the contexts. The thread pool
  /// must outlive the context. Multi-threading will be enabled as part of this
  /// method.
  void setThreadPool(llvm::ThreadPool &pool);

  /// Return the thread pool used by this context. This method requires that
  /// multithreading be enabled within the context, and should generally not be
  /// used directly. Users should instead prefer the threading utilities within
  /// Threading.h.
  llvm::ThreadPool &getThreadPool();

  /// Return true if we should attach the operation to diagnostics emitted via
  /// Operation::emit.
  bool shouldPrintOpOnDiagnostic();

  /// Set the flag specifying if we should attach the operation to diagnostics
  /// emitted via Operation::emit.
  void printOpOnDiagnostic(bool enable);

  /// Return true if we should attach the current stacktrace to diagnostics when
  /// emitted.
  bool shouldPrintStackTraceOnDiagnostic();

  /// Set the flag specifying if we should attach the current stacktrace when
  /// emitting diagnostics.
  void printStackTraceOnDiagnostic(bool enable);

  /// Return information about all registered operations.  This isn't very
  /// efficient: typically you should ask the operations about their properties
  /// directly.
  std::vector<AbstractOperation *> getRegisteredOperations();

  /// Return true if this operation name is registered in this context.
  bool isOperationRegistered(StringRef name);

  // This is effectively private given that only MLIRContext.cpp can see the
  // MLIRContextImpl type.
  MLIRContextImpl &getImpl() { return *impl; }

  /// Returns the diagnostic engine for this context.
  DiagnosticEngine &getDiagEngine();

  /// Returns the storage uniquer used for creating affine constructs.
  StorageUniquer &getAffineUniquer();

  /// Returns the storage uniquer used for constructing type storage instances.
  /// This should not be used directly.
  StorageUniquer &getTypeUniquer();

  /// Returns the storage uniquer used for constructing attribute storage
  /// instances. This should not be used directly.
  StorageUniquer &getAttributeUniquer();

  /// Returns the manager of debug actions within the context.
  DebugActionManager &getDebugActionManager();

  /// These APIs are tracking whether the context will be used in a
  /// multithreading environment: this has no effect other than enabling
  /// assertions on misuses of some APIs.
  void enterMultiThreadedExecution();
  void exitMultiThreadedExecution();

  /// Get a dialect for the provided namespace and TypeID: abort the program if
  /// a dialect exist for this namespace with different TypeID. If a dialect has
  /// not been loaded for this namespace/TypeID yet, use the provided ctor to
  /// create one on the fly and load it. Returns a pointer to the dialect owned
  /// by the context.
  /// The use of this method is in general discouraged in favor of
  /// 'getOrLoadDialect<DialectClass>()'.
  Dialect *getOrLoadDialect(StringRef dialectNamespace, TypeID dialectID,
                            function_ref<std::unique_ptr<Dialect>()> ctor);

  /// Returns a hash of the registry of the context that may be used to give
  /// a rough indicator of if the state of the context registry has changed. The
  /// context registry correlates to loaded dialects and their entities
  /// (attributes, operations, types, etc.).
  llvm::hash_code getRegistryHash();

private:
  const std::unique_ptr<MLIRContextImpl> impl;

  MLIRContext(const MLIRContext &) = delete;
  void operator=(const MLIRContext &) = delete;
};

//===----------------------------------------------------------------------===//
// MLIRContext CommandLine Options
//===----------------------------------------------------------------------===//

/// Register a set of useful command-line options that can be used to configure
/// various flags within the MLIRContext. These flags are used when constructing
/// an MLIR context for initialization.
void registerMLIRContextCLOptions();

} // end namespace mlir

#endif // MLIR_IR_MLIRCONTEXT_H
