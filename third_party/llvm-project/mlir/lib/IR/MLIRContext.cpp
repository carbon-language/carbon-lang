//===- MLIRContext.cpp - MLIR Type Classes --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/MLIRContext.h"
#include "AffineExprDetail.h"
#include "AffineMapDetail.h"
#include "AttributeDetail.h"
#include "IntegerSetDetail.h"
#include "TypeDetail.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Identifier.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/DebugAction.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/RWMutex.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

#define DEBUG_TYPE "mlircontext"

using namespace mlir;
using namespace mlir::detail;

using llvm::hash_combine;
using llvm::hash_combine_range;

//===----------------------------------------------------------------------===//
// MLIRContext CommandLine Options
//===----------------------------------------------------------------------===//

namespace {
/// This struct contains command line options that can be used to initialize
/// various bits of an MLIRContext. This uses a struct wrapper to avoid the need
/// for global command line options.
struct MLIRContextOptions {
  llvm::cl::opt<bool> disableThreading{
      "mlir-disable-threading",
      llvm::cl::desc("Disabling multi-threading within MLIR")};

  llvm::cl::opt<bool> printOpOnDiagnostic{
      "mlir-print-op-on-diagnostic",
      llvm::cl::desc("When a diagnostic is emitted on an operation, also print "
                     "the operation as an attached note"),
      llvm::cl::init(true)};

  llvm::cl::opt<bool> printStackTraceOnDiagnostic{
      "mlir-print-stacktrace-on-diagnostic",
      llvm::cl::desc("When a diagnostic is emitted, also print the stack trace "
                     "as an attached note")};
};
} // end anonymous namespace

static llvm::ManagedStatic<MLIRContextOptions> clOptions;

/// Register a set of useful command-line options that can be used to configure
/// various flags within the MLIRContext. These flags are used when constructing
/// an MLIR context for initialization.
void mlir::registerMLIRContextCLOptions() {
  // Make sure that the options struct has been initialized.
  *clOptions;
}

//===----------------------------------------------------------------------===//
// Locking Utilities
//===----------------------------------------------------------------------===//

namespace {
/// Utility reader lock that takes a runtime flag that specifies if we really
/// need to lock.
struct ScopedReaderLock {
  ScopedReaderLock(llvm::sys::SmartRWMutex<true> &mutexParam, bool shouldLock)
      : mutex(shouldLock ? &mutexParam : nullptr) {
    if (mutex)
      mutex->lock_shared();
  }
  ~ScopedReaderLock() {
    if (mutex)
      mutex->unlock_shared();
  }
  llvm::sys::SmartRWMutex<true> *mutex;
};
/// Utility writer lock that takes a runtime flag that specifies if we really
/// need to lock.
struct ScopedWriterLock {
  ScopedWriterLock(llvm::sys::SmartRWMutex<true> &mutexParam, bool shouldLock)
      : mutex(shouldLock ? &mutexParam : nullptr) {
    if (mutex)
      mutex->lock();
  }
  ~ScopedWriterLock() {
    if (mutex)
      mutex->unlock();
  }
  llvm::sys::SmartRWMutex<true> *mutex;
};
} // end anonymous namespace.

//===----------------------------------------------------------------------===//
// AffineMap and IntegerSet hashing
//===----------------------------------------------------------------------===//

/// A utility function to safely get or create a uniqued instance within the
/// given set container.
template <typename ValueT, typename DenseInfoT, typename KeyT,
          typename ConstructorFn>
static ValueT safeGetOrCreate(DenseSet<ValueT, DenseInfoT> &container,
                              KeyT &&key, llvm::sys::SmartRWMutex<true> &mutex,
                              bool threadingIsEnabled,
                              ConstructorFn &&constructorFn) {
  // Check for an existing instance in read-only mode.
  if (threadingIsEnabled) {
    llvm::sys::SmartScopedReader<true> instanceLock(mutex);
    auto it = container.find_as(key);
    if (it != container.end())
      return *it;
  }

  // Acquire a writer-lock so that we can safely create the new instance.
  ScopedWriterLock instanceLock(mutex, threadingIsEnabled);

  // Check for an existing instance again here, because another writer thread
  // may have already created one. Otherwise, construct a new instance.
  auto existing = container.insert_as(ValueT(), key);
  if (existing.second)
    return *existing.first = constructorFn();
  return *existing.first;
}

namespace {
struct AffineMapKeyInfo : DenseMapInfo<AffineMap> {
  // Affine maps are uniqued based on their dim/symbol counts and affine
  // expressions.
  using KeyTy = std::tuple<unsigned, unsigned, ArrayRef<AffineExpr>>;
  using DenseMapInfo<AffineMap>::isEqual;

  static unsigned getHashValue(const AffineMap &key) {
    return getHashValue(
        KeyTy(key.getNumDims(), key.getNumSymbols(), key.getResults()));
  }

  static unsigned getHashValue(KeyTy key) {
    return hash_combine(
        std::get<0>(key), std::get<1>(key),
        hash_combine_range(std::get<2>(key).begin(), std::get<2>(key).end()));
  }

  static bool isEqual(const KeyTy &lhs, AffineMap rhs) {
    if (rhs == getEmptyKey() || rhs == getTombstoneKey())
      return false;
    return lhs == std::make_tuple(rhs.getNumDims(), rhs.getNumSymbols(),
                                  rhs.getResults());
  }
};

struct IntegerSetKeyInfo : DenseMapInfo<IntegerSet> {
  // Integer sets are uniqued based on their dim/symbol counts, affine
  // expressions appearing in the LHS of constraints, and eqFlags.
  using KeyTy =
      std::tuple<unsigned, unsigned, ArrayRef<AffineExpr>, ArrayRef<bool>>;
  using DenseMapInfo<IntegerSet>::isEqual;

  static unsigned getHashValue(const IntegerSet &key) {
    return getHashValue(KeyTy(key.getNumDims(), key.getNumSymbols(),
                              key.getConstraints(), key.getEqFlags()));
  }

  static unsigned getHashValue(KeyTy key) {
    return hash_combine(
        std::get<0>(key), std::get<1>(key),
        hash_combine_range(std::get<2>(key).begin(), std::get<2>(key).end()),
        hash_combine_range(std::get<3>(key).begin(), std::get<3>(key).end()));
  }

  static bool isEqual(const KeyTy &lhs, IntegerSet rhs) {
    if (rhs == getEmptyKey() || rhs == getTombstoneKey())
      return false;
    return lhs == std::make_tuple(rhs.getNumDims(), rhs.getNumSymbols(),
                                  rhs.getConstraints(), rhs.getEqFlags());
  }
};
} // end anonymous namespace.

//===----------------------------------------------------------------------===//
// MLIRContextImpl
//===----------------------------------------------------------------------===//

namespace mlir {
/// This is the implementation of the MLIRContext class, using the pImpl idiom.
/// This class is completely private to this file, so everything is public.
class MLIRContextImpl {
public:
  //===--------------------------------------------------------------------===//
  // Debugging
  //===--------------------------------------------------------------------===//

  /// An action manager for use within the context.
  DebugActionManager debugActionManager;

  //===--------------------------------------------------------------------===//
  // Identifier uniquing
  //===--------------------------------------------------------------------===//

  // Identifier allocator and mutex for thread safety.
  llvm::BumpPtrAllocator identifierAllocator;
  llvm::sys::SmartRWMutex<true> identifierMutex;

  //===--------------------------------------------------------------------===//
  // Diagnostics
  //===--------------------------------------------------------------------===//
  DiagnosticEngine diagEngine;

  //===--------------------------------------------------------------------===//
  // Options
  //===--------------------------------------------------------------------===//

  /// In most cases, creating operation in unregistered dialect is not desired
  /// and indicate a misconfiguration of the compiler. This option enables to
  /// detect such use cases
  bool allowUnregisteredDialects = false;

  /// Enable support for multi-threading within MLIR.
  bool threadingIsEnabled = true;

  /// Track if we are currently executing in a threaded execution environment
  /// (like the pass-manager): this is only a debugging feature to help reducing
  /// the chances of data races one some context APIs.
#ifndef NDEBUG
  std::atomic<int> multiThreadedExecutionContext{0};
#endif

  /// If the operation should be attached to diagnostics printed via the
  /// Operation::emit methods.
  bool printOpOnDiagnostic = true;

  /// If the current stack trace should be attached when emitting diagnostics.
  bool printStackTraceOnDiagnostic = false;

  //===--------------------------------------------------------------------===//
  // Other
  //===--------------------------------------------------------------------===//

  /// This points to the ThreadPool used when processing MLIR tasks in parallel.
  /// It can't be nullptr when multi-threading is enabled. Otherwise if
  /// multi-threading is disabled, and the threadpool wasn't externally provided
  /// using `setThreadPool`, this will be nullptr.
  llvm::ThreadPool *threadPool = nullptr;

  /// In case where the thread pool is owned by the context, this ensures
  /// destruction with the context.
  std::unique_ptr<llvm::ThreadPool> ownedThreadPool;

  /// This is a list of dialects that are created referring to this context.
  /// The MLIRContext owns the objects.
  DenseMap<StringRef, std::unique_ptr<Dialect>> loadedDialects;
  DialectRegistry dialectsRegistry;

  /// This is a mapping from operation name to AbstractOperation for registered
  /// operations.
  llvm::StringMap<AbstractOperation> registeredOperations;

  /// Identifiers are uniqued by string value and use the internal string set
  /// for storage.
  llvm::StringMap<PointerUnion<Dialect *, MLIRContext *>,
                  llvm::BumpPtrAllocator &>
      identifiers;

  /// An allocator used for AbstractAttribute and AbstractType objects.
  llvm::BumpPtrAllocator abstractDialectSymbolAllocator;

  //===--------------------------------------------------------------------===//
  // Affine uniquing
  //===--------------------------------------------------------------------===//

  // Affine allocator and mutex for thread safety.
  llvm::BumpPtrAllocator affineAllocator;
  llvm::sys::SmartRWMutex<true> affineMutex;

  // Affine map uniquing.
  using AffineMapSet = DenseSet<AffineMap, AffineMapKeyInfo>;
  AffineMapSet affineMaps;

  // Integer set uniquing.
  using IntegerSets = DenseSet<IntegerSet, IntegerSetKeyInfo>;
  IntegerSets integerSets;

  // Affine expression uniquing.
  StorageUniquer affineUniquer;

  //===--------------------------------------------------------------------===//
  // Type uniquing
  //===--------------------------------------------------------------------===//

  DenseMap<TypeID, AbstractType *> registeredTypes;
  StorageUniquer typeUniquer;

  /// Cached Type Instances.
  BFloat16Type bf16Ty;
  Float16Type f16Ty;
  Float32Type f32Ty;
  Float64Type f64Ty;
  Float80Type f80Ty;
  Float128Type f128Ty;
  IndexType indexTy;
  IntegerType int1Ty, int8Ty, int16Ty, int32Ty, int64Ty, int128Ty;
  NoneType noneType;

  //===--------------------------------------------------------------------===//
  // Attribute uniquing
  //===--------------------------------------------------------------------===//

  DenseMap<TypeID, AbstractAttribute *> registeredAttributes;
  StorageUniquer attributeUniquer;

  /// Cached Attribute Instances.
  BoolAttr falseAttr, trueAttr;
  UnitAttr unitAttr;
  UnknownLoc unknownLocAttr;
  DictionaryAttr emptyDictionaryAttr;
  StringAttr emptyStringAttr;

public:
  MLIRContextImpl(bool threadingIsEnabled)
      : threadingIsEnabled(threadingIsEnabled),
        identifiers(identifierAllocator) {
    if (threadingIsEnabled) {
      ownedThreadPool = std::make_unique<llvm::ThreadPool>();
      threadPool = ownedThreadPool.get();
    }
  }
  ~MLIRContextImpl() {
    for (auto typeMapping : registeredTypes)
      typeMapping.second->~AbstractType();
    for (auto attrMapping : registeredAttributes)
      attrMapping.second->~AbstractAttribute();
  }
};
} // end namespace mlir

MLIRContext::MLIRContext(Threading setting)
    : MLIRContext(DialectRegistry(), setting) {}

MLIRContext::MLIRContext(const DialectRegistry &registry, Threading setting)
    : impl(new MLIRContextImpl(setting == Threading::ENABLED)) {
  // Initialize values based on the command line flags if they were provided.
  if (clOptions.isConstructed()) {
    disableMultithreading(clOptions->disableThreading);
    printOpOnDiagnostic(clOptions->printOpOnDiagnostic);
    printStackTraceOnDiagnostic(clOptions->printStackTraceOnDiagnostic);
  }

  // Pre-populate the registry.
  registry.appendTo(impl->dialectsRegistry);

  // Ensure the builtin dialect is always pre-loaded.
  getOrLoadDialect<BuiltinDialect>();

  // Initialize several common attributes and types to avoid the need to lock
  // the context when accessing them.

  //// Types.
  /// Floating-point Types.
  impl->bf16Ty = TypeUniquer::get<BFloat16Type>(this);
  impl->f16Ty = TypeUniquer::get<Float16Type>(this);
  impl->f32Ty = TypeUniquer::get<Float32Type>(this);
  impl->f64Ty = TypeUniquer::get<Float64Type>(this);
  impl->f80Ty = TypeUniquer::get<Float80Type>(this);
  impl->f128Ty = TypeUniquer::get<Float128Type>(this);
  /// Index Type.
  impl->indexTy = TypeUniquer::get<IndexType>(this);
  /// Integer Types.
  impl->int1Ty = TypeUniquer::get<IntegerType>(this, 1, IntegerType::Signless);
  impl->int8Ty = TypeUniquer::get<IntegerType>(this, 8, IntegerType::Signless);
  impl->int16Ty =
      TypeUniquer::get<IntegerType>(this, 16, IntegerType::Signless);
  impl->int32Ty =
      TypeUniquer::get<IntegerType>(this, 32, IntegerType::Signless);
  impl->int64Ty =
      TypeUniquer::get<IntegerType>(this, 64, IntegerType::Signless);
  impl->int128Ty =
      TypeUniquer::get<IntegerType>(this, 128, IntegerType::Signless);
  /// None Type.
  impl->noneType = TypeUniquer::get<NoneType>(this);

  //// Attributes.
  //// Note: These must be registered after the types as they may generate one
  //// of the above types internally.
  /// Unknown Location Attribute.
  impl->unknownLocAttr = AttributeUniquer::get<UnknownLoc>(this);
  /// Bool Attributes.
  impl->falseAttr = IntegerAttr::getBoolAttrUnchecked(impl->int1Ty, false);
  impl->trueAttr = IntegerAttr::getBoolAttrUnchecked(impl->int1Ty, true);
  /// Unit Attribute.
  impl->unitAttr = AttributeUniquer::get<UnitAttr>(this);
  /// The empty dictionary attribute.
  impl->emptyDictionaryAttr = DictionaryAttr::getEmptyUnchecked(this);
  /// The empty string attribute.
  impl->emptyStringAttr = StringAttr::getEmptyStringAttrUnchecked(this);

  // Register the affine storage objects with the uniquer.
  impl->affineUniquer
      .registerParametricStorageType<AffineBinaryOpExprStorage>();
  impl->affineUniquer
      .registerParametricStorageType<AffineConstantExprStorage>();
  impl->affineUniquer.registerParametricStorageType<AffineDimExprStorage>();
}

MLIRContext::~MLIRContext() {}

/// Copy the specified array of elements into memory managed by the provided
/// bump pointer allocator.  This assumes the elements are all PODs.
template <typename T>
static ArrayRef<T> copyArrayRefInto(llvm::BumpPtrAllocator &allocator,
                                    ArrayRef<T> elements) {
  auto result = allocator.Allocate<T>(elements.size());
  std::uninitialized_copy(elements.begin(), elements.end(), result);
  return ArrayRef<T>(result, elements.size());
}

//===----------------------------------------------------------------------===//
// Debugging
//===----------------------------------------------------------------------===//

DebugActionManager &MLIRContext::getDebugActionManager() {
  return getImpl().debugActionManager;
}

//===----------------------------------------------------------------------===//
// Diagnostic Handlers
//===----------------------------------------------------------------------===//

/// Returns the diagnostic engine for this context.
DiagnosticEngine &MLIRContext::getDiagEngine() { return getImpl().diagEngine; }

//===----------------------------------------------------------------------===//
// Dialect and Operation Registration
//===----------------------------------------------------------------------===//

void MLIRContext::appendDialectRegistry(const DialectRegistry &registry) {
  registry.appendTo(impl->dialectsRegistry);

  // For the already loaded dialects, register the interfaces immediately.
  for (const auto &kvp : impl->loadedDialects)
    registry.registerDelayedInterfaces(kvp.second.get());
}

const DialectRegistry &MLIRContext::getDialectRegistry() {
  return impl->dialectsRegistry;
}

/// Return information about all registered IR dialects.
std::vector<Dialect *> MLIRContext::getLoadedDialects() {
  std::vector<Dialect *> result;
  result.reserve(impl->loadedDialects.size());
  for (auto &dialect : impl->loadedDialects)
    result.push_back(dialect.second.get());
  llvm::array_pod_sort(result.begin(), result.end(),
                       [](Dialect *const *lhs, Dialect *const *rhs) -> int {
                         return (*lhs)->getNamespace() < (*rhs)->getNamespace();
                       });
  return result;
}
std::vector<StringRef> MLIRContext::getAvailableDialects() {
  std::vector<StringRef> result;
  for (auto dialect : impl->dialectsRegistry.getDialectNames())
    result.push_back(dialect);
  return result;
}

/// Get a registered IR dialect with the given namespace. If none is found,
/// then return nullptr.
Dialect *MLIRContext::getLoadedDialect(StringRef name) {
  // Dialects are sorted by name, so we can use binary search for lookup.
  auto it = impl->loadedDialects.find(name);
  return (it != impl->loadedDialects.end()) ? it->second.get() : nullptr;
}

Dialect *MLIRContext::getOrLoadDialect(StringRef name) {
  Dialect *dialect = getLoadedDialect(name);
  if (dialect)
    return dialect;
  DialectAllocatorFunctionRef allocator =
      impl->dialectsRegistry.getDialectAllocator(name);
  return allocator ? allocator(this) : nullptr;
}

/// Get a dialect for the provided namespace and TypeID: abort the program if a
/// dialect exist for this namespace with different TypeID. Returns a pointer to
/// the dialect owned by the context.
Dialect *
MLIRContext::getOrLoadDialect(StringRef dialectNamespace, TypeID dialectID,
                              function_ref<std::unique_ptr<Dialect>()> ctor) {
  auto &impl = getImpl();
  // Get the correct insertion position sorted by namespace.
  std::unique_ptr<Dialect> &dialect = impl.loadedDialects[dialectNamespace];

  if (!dialect) {
    LLVM_DEBUG(llvm::dbgs()
               << "Load new dialect in Context " << dialectNamespace << "\n");
#ifndef NDEBUG
    if (impl.multiThreadedExecutionContext != 0)
      llvm::report_fatal_error(
          "Loading a dialect (" + dialectNamespace +
          ") while in a multi-threaded execution context (maybe "
          "the PassManager): this can indicate a "
          "missing `dependentDialects` in a pass for example.");
#endif
    dialect = ctor();
    assert(dialect && "dialect ctor failed");

    // Refresh all the identifiers dialect field, this catches cases where a
    // dialect may be loaded after identifier prefixed with this dialect name
    // were already created.
    llvm::SmallString<32> dialectPrefix(dialectNamespace);
    dialectPrefix.push_back('.');
    for (auto &identifierEntry : impl.identifiers)
      if (identifierEntry.second.is<MLIRContext *>() &&
          identifierEntry.first().startswith(dialectPrefix))
        identifierEntry.second = dialect.get();

    // Actually register the interfaces with delayed registration.
    impl.dialectsRegistry.registerDelayedInterfaces(dialect.get());
    return dialect.get();
  }

  // Abort if dialect with namespace has already been registered.
  if (dialect->getTypeID() != dialectID)
    llvm::report_fatal_error("a dialect with namespace '" + dialectNamespace +
                             "' has already been registered");

  return dialect.get();
}

void MLIRContext::loadAllAvailableDialects() {
  for (StringRef name : getAvailableDialects())
    getOrLoadDialect(name);
}

llvm::hash_code MLIRContext::getRegistryHash() {
  llvm::hash_code hash(0);
  // Factor in number of loaded dialects, attributes, operations, types.
  hash = llvm::hash_combine(hash, impl->loadedDialects.size());
  hash = llvm::hash_combine(hash, impl->registeredAttributes.size());
  hash = llvm::hash_combine(hash, impl->registeredOperations.size());
  hash = llvm::hash_combine(hash, impl->registeredTypes.size());
  return hash;
}

bool MLIRContext::allowsUnregisteredDialects() {
  return impl->allowUnregisteredDialects;
}

void MLIRContext::allowUnregisteredDialects(bool allowing) {
  impl->allowUnregisteredDialects = allowing;
}

/// Return true if multi-threading is enabled by the context.
bool MLIRContext::isMultithreadingEnabled() {
  return impl->threadingIsEnabled && llvm::llvm_is_multithreaded();
}

/// Set the flag specifying if multi-threading is disabled by the context.
void MLIRContext::disableMultithreading(bool disable) {
  impl->threadingIsEnabled = !disable;

  // Update the threading mode for each of the uniquers.
  impl->affineUniquer.disableMultithreading(disable);
  impl->attributeUniquer.disableMultithreading(disable);
  impl->typeUniquer.disableMultithreading(disable);

  // Destroy thread pool (stop all threads) if it is no longer needed, or create
  // a new one if multithreading was re-enabled.
  if (disable) {
    // If the thread pool is owned, explicitly set it to nullptr to avoid
    // keeping a dangling pointer around. If the thread pool is externally
    // owned, we don't do anything.
    if (impl->ownedThreadPool) {
      assert(impl->threadPool);
      impl->threadPool = nullptr;
      impl->ownedThreadPool.reset();
    }
  } else if (!impl->threadPool) {
    // The thread pool isn't externally provided.
    assert(!impl->ownedThreadPool);
    impl->ownedThreadPool = std::make_unique<llvm::ThreadPool>();
    impl->threadPool = impl->ownedThreadPool.get();
  }
}

void MLIRContext::setThreadPool(llvm::ThreadPool &pool) {
  assert(!isMultithreadingEnabled() &&
         "expected multi-threading to be disabled when setting a ThreadPool");
  impl->threadPool = &pool;
  impl->ownedThreadPool.reset();
  enableMultithreading();
}

llvm::ThreadPool &MLIRContext::getThreadPool() {
  assert(isMultithreadingEnabled() &&
         "expected multi-threading to be enabled within the context");
  assert(impl->threadPool &&
         "multi-threading is enabled but threadpool not set");
  return *impl->threadPool;
}

void MLIRContext::enterMultiThreadedExecution() {
#ifndef NDEBUG
  ++impl->multiThreadedExecutionContext;
#endif
}
void MLIRContext::exitMultiThreadedExecution() {
#ifndef NDEBUG
  --impl->multiThreadedExecutionContext;
#endif
}

/// Return true if we should attach the operation to diagnostics emitted via
/// Operation::emit.
bool MLIRContext::shouldPrintOpOnDiagnostic() {
  return impl->printOpOnDiagnostic;
}

/// Set the flag specifying if we should attach the operation to diagnostics
/// emitted via Operation::emit.
void MLIRContext::printOpOnDiagnostic(bool enable) {
  impl->printOpOnDiagnostic = enable;
}

/// Return true if we should attach the current stacktrace to diagnostics when
/// emitted.
bool MLIRContext::shouldPrintStackTraceOnDiagnostic() {
  return impl->printStackTraceOnDiagnostic;
}

/// Set the flag specifying if we should attach the current stacktrace when
/// emitting diagnostics.
void MLIRContext::printStackTraceOnDiagnostic(bool enable) {
  impl->printStackTraceOnDiagnostic = enable;
}

/// Return information about all registered operations.  This isn't very
/// efficient, typically you should ask the operations about their properties
/// directly.
std::vector<AbstractOperation *> MLIRContext::getRegisteredOperations() {
  // We just have the operations in a non-deterministic hash table order. Dump
  // into a temporary array, then sort it by operation name to get a stable
  // ordering.
  llvm::StringMap<AbstractOperation> &registeredOps =
      impl->registeredOperations;

  std::vector<AbstractOperation *> result;
  result.reserve(registeredOps.size());
  for (auto &elt : registeredOps)
    result.push_back(&elt.second);
  llvm::array_pod_sort(
      result.begin(), result.end(),
      [](AbstractOperation *const *lhs, AbstractOperation *const *rhs) {
        return (*lhs)->name.compare((*rhs)->name);
      });

  return result;
}

bool MLIRContext::isOperationRegistered(StringRef name) {
  return impl->registeredOperations.count(name);
}

void Dialect::addType(TypeID typeID, AbstractType &&typeInfo) {
  auto &impl = context->getImpl();
  assert(impl.multiThreadedExecutionContext == 0 &&
         "Registering a new type kind while in a multi-threaded execution "
         "context");
  auto *newInfo =
      new (impl.abstractDialectSymbolAllocator.Allocate<AbstractType>())
          AbstractType(std::move(typeInfo));
  if (!impl.registeredTypes.insert({typeID, newInfo}).second)
    llvm::report_fatal_error("Dialect Type already registered.");
}

void Dialect::addAttribute(TypeID typeID, AbstractAttribute &&attrInfo) {
  auto &impl = context->getImpl();
  assert(impl.multiThreadedExecutionContext == 0 &&
         "Registering a new attribute kind while in a multi-threaded execution "
         "context");
  auto *newInfo =
      new (impl.abstractDialectSymbolAllocator.Allocate<AbstractAttribute>())
          AbstractAttribute(std::move(attrInfo));
  if (!impl.registeredAttributes.insert({typeID, newInfo}).second)
    llvm::report_fatal_error("Dialect Attribute already registered.");
}

//===----------------------------------------------------------------------===//
// AbstractAttribute
//===----------------------------------------------------------------------===//

/// Get the dialect that registered the attribute with the provided typeid.
const AbstractAttribute &AbstractAttribute::lookup(TypeID typeID,
                                                   MLIRContext *context) {
  const AbstractAttribute *abstract = lookupMutable(typeID, context);
  if (!abstract)
    llvm::report_fatal_error("Trying to create an Attribute that was not "
                             "registered in this MLIRContext.");
  return *abstract;
}

AbstractAttribute *AbstractAttribute::lookupMutable(TypeID typeID,
                                                    MLIRContext *context) {
  auto &impl = context->getImpl();
  auto it = impl.registeredAttributes.find(typeID);
  if (it == impl.registeredAttributes.end())
    return nullptr;
  return it->second;
}

//===----------------------------------------------------------------------===//
// AbstractOperation
//===----------------------------------------------------------------------===//

ParseResult AbstractOperation::parseAssembly(OpAsmParser &parser,
                                             OperationState &result) const {
  return parseAssemblyFn(parser, result);
}

/// Look up the specified operation in the operation set and return a pointer
/// to it if present. Otherwise, return a null pointer.
AbstractOperation *AbstractOperation::lookupMutable(StringRef opName,
                                                    MLIRContext *context) {
  auto &impl = context->getImpl();
  auto it = impl.registeredOperations.find(opName);
  if (it != impl.registeredOperations.end())
    return &it->second;
  return nullptr;
}

void AbstractOperation::insert(
    StringRef name, Dialect &dialect, TypeID typeID,
    ParseAssemblyFn &&parseAssembly, PrintAssemblyFn &&printAssembly,
    VerifyInvariantsFn &&verifyInvariants, FoldHookFn &&foldHook,
    GetCanonicalizationPatternsFn &&getCanonicalizationPatterns,
    detail::InterfaceMap &&interfaceMap, HasTraitFn &&hasTrait,
    ArrayRef<StringRef> attrNames) {
  MLIRContext *ctx = dialect.getContext();
  auto &impl = ctx->getImpl();
  assert(impl.multiThreadedExecutionContext == 0 &&
         "Registering a new operation kind while in a multi-threaded execution "
         "context");

  // Register the attribute names of this operation.
  MutableArrayRef<Identifier> cachedAttrNames;
  if (!attrNames.empty()) {
    cachedAttrNames = MutableArrayRef<Identifier>(
        impl.identifierAllocator.Allocate<Identifier>(attrNames.size()),
        attrNames.size());
    for (unsigned i : llvm::seq<unsigned>(0, attrNames.size()))
      new (&cachedAttrNames[i]) Identifier(Identifier::get(attrNames[i], ctx));
  }

  // Register the information for this operation.
  AbstractOperation opInfo(
      name, dialect, typeID, std::move(parseAssembly), std::move(printAssembly),
      std::move(verifyInvariants), std::move(foldHook),
      std::move(getCanonicalizationPatterns), std::move(interfaceMap),
      std::move(hasTrait), cachedAttrNames);
  if (!impl.registeredOperations.insert({name, std::move(opInfo)}).second) {
    llvm::errs() << "error: operation named '" << name
                 << "' is already registered.\n";
    abort();
  }
}

AbstractOperation::AbstractOperation(
    StringRef name, Dialect &dialect, TypeID typeID,
    ParseAssemblyFn &&parseAssembly, PrintAssemblyFn &&printAssembly,
    VerifyInvariantsFn &&verifyInvariants, FoldHookFn &&foldHook,
    GetCanonicalizationPatternsFn &&getCanonicalizationPatterns,
    detail::InterfaceMap &&interfaceMap, HasTraitFn &&hasTrait,
    ArrayRef<Identifier> attrNames)
    : name(Identifier::get(name, dialect.getContext())), dialect(dialect),
      typeID(typeID), interfaceMap(std::move(interfaceMap)),
      foldHookFn(std::move(foldHook)),
      getCanonicalizationPatternsFn(std::move(getCanonicalizationPatterns)),
      hasTraitFn(std::move(hasTrait)),
      parseAssemblyFn(std::move(parseAssembly)),
      printAssemblyFn(std::move(printAssembly)),
      verifyInvariantsFn(std::move(verifyInvariants)),
      attributeNames(attrNames) {}

//===----------------------------------------------------------------------===//
// AbstractType
//===----------------------------------------------------------------------===//

const AbstractType &AbstractType::lookup(TypeID typeID, MLIRContext *context) {
  const AbstractType *type = lookupMutable(typeID, context);
  if (!type)
    llvm::report_fatal_error(
        "Trying to create a Type that was not registered in this MLIRContext.");
  return *type;
}

AbstractType *AbstractType::lookupMutable(TypeID typeID, MLIRContext *context) {
  auto &impl = context->getImpl();
  auto it = impl.registeredTypes.find(typeID);
  if (it == impl.registeredTypes.end())
    return nullptr;
  return it->second;
}

//===----------------------------------------------------------------------===//
// Identifier uniquing
//===----------------------------------------------------------------------===//

/// Return an identifier for the specified string.
Identifier Identifier::get(const Twine &string, MLIRContext *context) {
  SmallString<32> tempStr;
  StringRef str = string.toStringRef(tempStr);

  // Check invariants after seeing if we already have something in the
  // identifier table - if we already had it in the table, then it already
  // passed invariant checks.
  assert(!str.empty() && "Cannot create an empty identifier");
  assert(str.find('\0') == StringRef::npos &&
         "Cannot create an identifier with a nul character");

  auto getDialectOrContext = [&]() {
    PointerUnion<Dialect *, MLIRContext *> dialectOrContext = context;
    auto dialectNamePair = str.split('.');
    if (!dialectNamePair.first.empty())
      if (Dialect *dialect = context->getLoadedDialect(dialectNamePair.first))
        dialectOrContext = dialect;
    return dialectOrContext;
  };

  auto &impl = context->getImpl();
  if (!context->isMultithreadingEnabled()) {
    auto insertedIt = impl.identifiers.insert({str, nullptr});
    if (insertedIt.second)
      insertedIt.first->second = getDialectOrContext();
    return Identifier(&*insertedIt.first);
  }

  // Check for an existing identifier in read-only mode.
  {
    llvm::sys::SmartScopedReader<true> contextLock(impl.identifierMutex);
    auto it = impl.identifiers.find(str);
    if (it != impl.identifiers.end())
      return Identifier(&*it);
  }

  // Acquire a writer-lock so that we can safely create the new instance.
  llvm::sys::SmartScopedWriter<true> contextLock(impl.identifierMutex);
  auto it = impl.identifiers.insert({str, getDialectOrContext()}).first;
  return Identifier(&*it);
}

Dialect *Identifier::getDialect() {
  return entry->second.dyn_cast<Dialect *>();
}

MLIRContext *Identifier::getContext() {
  if (Dialect *dialect = getDialect())
    return dialect->getContext();
  return entry->second.get<MLIRContext *>();
}

//===----------------------------------------------------------------------===//
// Type uniquing
//===----------------------------------------------------------------------===//

/// Returns the storage uniquer used for constructing type storage instances.
/// This should not be used directly.
StorageUniquer &MLIRContext::getTypeUniquer() { return getImpl().typeUniquer; }

BFloat16Type BFloat16Type::get(MLIRContext *context) {
  return context->getImpl().bf16Ty;
}
Float16Type Float16Type::get(MLIRContext *context) {
  return context->getImpl().f16Ty;
}
Float32Type Float32Type::get(MLIRContext *context) {
  return context->getImpl().f32Ty;
}
Float64Type Float64Type::get(MLIRContext *context) {
  return context->getImpl().f64Ty;
}
Float80Type Float80Type::get(MLIRContext *context) {
  return context->getImpl().f80Ty;
}
Float128Type Float128Type::get(MLIRContext *context) {
  return context->getImpl().f128Ty;
}

/// Get an instance of the IndexType.
IndexType IndexType::get(MLIRContext *context) {
  return context->getImpl().indexTy;
}

/// Return an existing integer type instance if one is cached within the
/// context.
static IntegerType
getCachedIntegerType(unsigned width,
                     IntegerType::SignednessSemantics signedness,
                     MLIRContext *context) {
  if (signedness != IntegerType::Signless)
    return IntegerType();

  switch (width) {
  case 1:
    return context->getImpl().int1Ty;
  case 8:
    return context->getImpl().int8Ty;
  case 16:
    return context->getImpl().int16Ty;
  case 32:
    return context->getImpl().int32Ty;
  case 64:
    return context->getImpl().int64Ty;
  case 128:
    return context->getImpl().int128Ty;
  default:
    return IntegerType();
  }
}

IntegerType IntegerType::get(MLIRContext *context, unsigned width,
                             IntegerType::SignednessSemantics signedness) {
  if (auto cached = getCachedIntegerType(width, signedness, context))
    return cached;
  return Base::get(context, width, signedness);
}

IntegerType
IntegerType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                        MLIRContext *context, unsigned width,
                        SignednessSemantics signedness) {
  if (auto cached = getCachedIntegerType(width, signedness, context))
    return cached;
  return Base::getChecked(emitError, context, width, signedness);
}

/// Get an instance of the NoneType.
NoneType NoneType::get(MLIRContext *context) {
  if (NoneType cachedInst = context->getImpl().noneType)
    return cachedInst;
  // Note: May happen when initializing the singleton attributes of the builtin
  // dialect.
  return Base::get(context);
}

//===----------------------------------------------------------------------===//
// Attribute uniquing
//===----------------------------------------------------------------------===//

/// Returns the storage uniquer used for constructing attribute storage
/// instances. This should not be used directly.
StorageUniquer &MLIRContext::getAttributeUniquer() {
  return getImpl().attributeUniquer;
}

/// Initialize the given attribute storage instance.
void AttributeUniquer::initializeAttributeStorage(AttributeStorage *storage,
                                                  MLIRContext *ctx,
                                                  TypeID attrID) {
  storage->initialize(AbstractAttribute::lookup(attrID, ctx));

  // If the attribute did not provide a type, then default to NoneType.
  if (!storage->getType())
    storage->setType(NoneType::get(ctx));
}

BoolAttr BoolAttr::get(MLIRContext *context, bool value) {
  return value ? context->getImpl().trueAttr : context->getImpl().falseAttr;
}

UnitAttr UnitAttr::get(MLIRContext *context) {
  return context->getImpl().unitAttr;
}

UnknownLoc UnknownLoc::get(MLIRContext *context) {
  return context->getImpl().unknownLocAttr;
}

/// Return empty dictionary.
DictionaryAttr DictionaryAttr::getEmpty(MLIRContext *context) {
  return context->getImpl().emptyDictionaryAttr;
}

/// Return an empty string.
StringAttr StringAttr::get(MLIRContext *context) {
  return context->getImpl().emptyStringAttr;
}

//===----------------------------------------------------------------------===//
// AffineMap uniquing
//===----------------------------------------------------------------------===//

StorageUniquer &MLIRContext::getAffineUniquer() {
  return getImpl().affineUniquer;
}

AffineMap AffineMap::getImpl(unsigned dimCount, unsigned symbolCount,
                             ArrayRef<AffineExpr> results,
                             MLIRContext *context) {
  auto &impl = context->getImpl();
  auto key = std::make_tuple(dimCount, symbolCount, results);

  // Safely get or create an AffineMap instance.
  return safeGetOrCreate(
      impl.affineMaps, key, impl.affineMutex, impl.threadingIsEnabled, [&] {
        auto *res = impl.affineAllocator.Allocate<detail::AffineMapStorage>();

        // Copy the results into the bump pointer.
        results = copyArrayRefInto(impl.affineAllocator, results);

        // Initialize the memory using placement new.
        new (res)
            detail::AffineMapStorage{dimCount, symbolCount, results, context};
        return AffineMap(res);
      });
}

AffineMap AffineMap::get(MLIRContext *context) {
  return getImpl(/*dimCount=*/0, /*symbolCount=*/0, /*results=*/{}, context);
}

AffineMap AffineMap::get(unsigned dimCount, unsigned symbolCount,
                         MLIRContext *context) {
  return getImpl(dimCount, symbolCount, /*results=*/{}, context);
}

AffineMap AffineMap::get(unsigned dimCount, unsigned symbolCount,
                         AffineExpr result) {
  return getImpl(dimCount, symbolCount, {result}, result.getContext());
}

AffineMap AffineMap::get(unsigned dimCount, unsigned symbolCount,
                         ArrayRef<AffineExpr> results, MLIRContext *context) {
  return getImpl(dimCount, symbolCount, results, context);
}

//===----------------------------------------------------------------------===//
// Integer Sets: these are allocated into the bump pointer, and are immutable.
// Unlike AffineMap's, these are uniqued only if they are small.
//===----------------------------------------------------------------------===//

IntegerSet IntegerSet::get(unsigned dimCount, unsigned symbolCount,
                           ArrayRef<AffineExpr> constraints,
                           ArrayRef<bool> eqFlags) {
  // The number of constraints can't be zero.
  assert(!constraints.empty());
  assert(constraints.size() == eqFlags.size());

  auto &impl = constraints[0].getContext()->getImpl();

  // A utility function to construct a new IntegerSetStorage instance.
  auto constructorFn = [&] {
    auto *res = impl.affineAllocator.Allocate<detail::IntegerSetStorage>();

    // Copy the results and equality flags into the bump pointer.
    constraints = copyArrayRefInto(impl.affineAllocator, constraints);
    eqFlags = copyArrayRefInto(impl.affineAllocator, eqFlags);

    // Initialize the memory using placement new.
    new (res)
        detail::IntegerSetStorage{dimCount, symbolCount, constraints, eqFlags};
    return IntegerSet(res);
  };

  // If this instance is uniqued, then we handle it separately so that multiple
  // threads may simultaneously access existing instances.
  if (constraints.size() < IntegerSet::kUniquingThreshold) {
    auto key = std::make_tuple(dimCount, symbolCount, constraints, eqFlags);
    return safeGetOrCreate(impl.integerSets, key, impl.affineMutex,
                           impl.threadingIsEnabled, constructorFn);
  }

  // Otherwise, acquire a writer-lock so that we can safely create the new
  // instance.
  ScopedWriterLock affineLock(impl.affineMutex, impl.threadingIsEnabled);
  return constructorFn();
}

//===----------------------------------------------------------------------===//
// StorageUniquerSupport
//===----------------------------------------------------------------------===//

/// Utility method to generate a callback that can be used to generate a
/// diagnostic when checking the construction invariants of a storage object.
/// This is defined out-of-line to avoid the need to include Location.h.
llvm::unique_function<InFlightDiagnostic()>
mlir::detail::getDefaultDiagnosticEmitFn(MLIRContext *ctx) {
  return [ctx] { return emitError(UnknownLoc::get(ctx)); };
}
llvm::unique_function<InFlightDiagnostic()>
mlir::detail::getDefaultDiagnosticEmitFn(const Location &loc) {
  return [=] { return emitError(loc); };
}
