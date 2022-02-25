//===- TypeSupport.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines support types for registering dialect extended types.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_TYPE_SUPPORT_H
#define MLIR_IR_TYPE_SUPPORT_H

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/StorageUniquerSupport.h"
#include "llvm/ADT/Twine.h"

namespace mlir {
class Dialect;
class MLIRContext;

//===----------------------------------------------------------------------===//
// AbstractType
//===----------------------------------------------------------------------===//

/// This class contains all of the static information common to all instances of
/// a registered Type.
class AbstractType {
public:
  /// Look up the specified abstract type in the MLIRContext and return a
  /// reference to it.
  static const AbstractType &lookup(TypeID typeID, MLIRContext *context);

  /// This method is used by Dialect objects when they register the list of
  /// types they contain.
  template <typename T> static AbstractType get(Dialect &dialect) {
    return AbstractType(dialect, T::getInterfaceMap(), T::getTypeID());
  }

  /// This method is used by Dialect objects to register types with
  /// custom TypeIDs.
  /// The use of this method is in general discouraged in favor of
  /// 'get<CustomType>(dialect)';
  static AbstractType get(Dialect &dialect, detail::InterfaceMap &&interfaceMap,
                          TypeID typeID) {
    return AbstractType(dialect, std::move(interfaceMap), typeID);
  }

  /// Return the dialect this type was registered to.
  Dialect &getDialect() const { return const_cast<Dialect &>(dialect); }

  /// Returns an instance of the concept object for the given interface if it
  /// was registered to this type, null otherwise. This should not be used
  /// directly.
  template <typename T> typename T::Concept *getInterface() const {
    return interfaceMap.lookup<T>();
  }

  /// Returns true if the type has the interface with the given ID.
  bool hasInterface(TypeID interfaceID) const {
    return interfaceMap.contains(interfaceID);
  }

  /// Return the unique identifier representing the concrete type class.
  TypeID getTypeID() const { return typeID; }

private:
  AbstractType(Dialect &dialect, detail::InterfaceMap &&interfaceMap,
               TypeID typeID)
      : dialect(dialect), interfaceMap(std::move(interfaceMap)),
        typeID(typeID) {}

  /// Give StorageUserBase access to the mutable lookup.
  template <typename ConcreteT, typename BaseT, typename StorageT,
            typename UniquerT, template <typename T> class... Traits>
  friend class detail::StorageUserBase;

  /// Look up the specified abstract type in the MLIRContext and return a
  /// (mutable) pointer to it. Return a null pointer if the type could not
  /// be found in the context.
  static AbstractType *lookupMutable(TypeID typeID, MLIRContext *context);

  /// This is the dialect that this type was registered to.
  const Dialect &dialect;

  /// This is a collection of the interfaces registered to this type.
  detail::InterfaceMap interfaceMap;

  /// The unique identifier of the derived Type class.
  const TypeID typeID;
};

//===----------------------------------------------------------------------===//
// TypeStorage
//===----------------------------------------------------------------------===//

namespace detail {
struct TypeUniquer;
} // end namespace detail

/// Base storage class appearing in a Type.
class TypeStorage : public StorageUniquer::BaseStorage {
  friend detail::TypeUniquer;
  friend StorageUniquer;

public:
  /// Return the abstract type descriptor for this type.
  const AbstractType &getAbstractType() {
    assert(abstractType && "Malformed type storage object.");
    return *abstractType;
  }

protected:
  /// This constructor is used by derived classes as part of the TypeUniquer.
  TypeStorage() : abstractType(nullptr) {}

private:
  /// Set the abstract type for this storage instance. This is used by the
  /// TypeUniquer when initializing a newly constructed type storage object.
  void initialize(const AbstractType &abstractTy) {
    abstractType = const_cast<AbstractType *>(&abstractTy);
  }

  /// The abstract description for this type.
  AbstractType *abstractType;
};

/// Default storage type for types that require no additional initialization or
/// storage.
using DefaultTypeStorage = TypeStorage;

//===----------------------------------------------------------------------===//
// TypeStorageAllocator
//===----------------------------------------------------------------------===//

/// This is a utility allocator used to allocate memory for instances of derived
/// Types.
using TypeStorageAllocator = StorageUniquer::StorageAllocator;

//===----------------------------------------------------------------------===//
// TypeUniquer
//===----------------------------------------------------------------------===//
namespace detail {
/// A utility class to get, or create, unique instances of types within an
/// MLIRContext. This class manages all creation and uniquing of types.
struct TypeUniquer {
  /// Get an uniqued instance of a parametric type T.
  template <typename T, typename... Args>
  static typename std::enable_if_t<
      !std::is_same<typename T::ImplType, TypeStorage>::value, T>
  get(MLIRContext *ctx, Args &&...args) {
#ifndef NDEBUG
    if (!ctx->getTypeUniquer().isParametricStorageInitialized(T::getTypeID()))
      llvm::report_fatal_error(llvm::Twine("can't create type '") +
                               llvm::getTypeName<T>() +
                               "' because storage uniquer isn't initialized: "
                               "the dialect was likely not loaded.");
#endif
    return ctx->getTypeUniquer().get<typename T::ImplType>(
        [&](TypeStorage *storage) {
          storage->initialize(AbstractType::lookup(T::getTypeID(), ctx));
        },
        T::getTypeID(), std::forward<Args>(args)...);
  }
  /// Get an uniqued instance of a singleton type T.
  template <typename T>
  static typename std::enable_if_t<
      std::is_same<typename T::ImplType, TypeStorage>::value, T>
  get(MLIRContext *ctx) {
#ifndef NDEBUG
    if (!ctx->getTypeUniquer().isSingletonStorageInitialized(T::getTypeID()))
      llvm::report_fatal_error(llvm::Twine("can't create type '") +
                               llvm::getTypeName<T>() +
                               "' because storage uniquer isn't initialized: "
                               "the dialect was likely not loaded.");
#endif
    return ctx->getTypeUniquer().get<typename T::ImplType>(T::getTypeID());
  }

  /// Change the mutable component of the given type instance in the provided
  /// context.
  template <typename T, typename... Args>
  static LogicalResult mutate(MLIRContext *ctx, typename T::ImplType *impl,
                              Args &&...args) {
    assert(impl && "cannot mutate null type");
    return ctx->getTypeUniquer().mutate(T::getTypeID(), impl,
                                        std::forward<Args>(args)...);
  }

  /// Register a parametric type instance T with the uniquer.
  template <typename T>
  static typename std::enable_if_t<
      !std::is_same<typename T::ImplType, TypeStorage>::value>
  registerType(MLIRContext *ctx) {
    ctx->getTypeUniquer().registerParametricStorageType<typename T::ImplType>(
        T::getTypeID());
  }
  /// Register a singleton type instance T with the uniquer.
  template <typename T>
  static typename std::enable_if_t<
      std::is_same<typename T::ImplType, TypeStorage>::value>
  registerType(MLIRContext *ctx) {
    ctx->getTypeUniquer().registerSingletonStorageType<TypeStorage>(
        T::getTypeID(), [&](TypeStorage *storage) {
          storage->initialize(AbstractType::lookup(T::getTypeID(), ctx));
        });
  }
};
} // namespace detail

} // end namespace mlir

#endif
