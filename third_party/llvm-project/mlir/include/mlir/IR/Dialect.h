//===- Dialect.h - IR Dialect Description -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the 'dialect' abstraction.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_DIALECT_H
#define MLIR_IR_DIALECT_H

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/TypeID.h"

#include <map>
#include <tuple>

namespace mlir {
class DialectAsmParser;
class DialectAsmPrinter;
class DialectInterface;
class OpBuilder;
class Type;

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

/// Dialects are groups of MLIR operations, types and attributes, as well as
/// behavior associated with the entire group.  For example, hooks into other
/// systems for constant folding, interfaces, default named types for asm
/// printing, etc.
///
/// Instances of the dialect object are loaded in a specific MLIRContext.
///
class Dialect {
public:
  /// Type for a callback provided by the dialect to parse a custom operation.
  /// This is used for the dialect to provide an alternative way to parse custom
  /// operations, including unregistered ones.
  using ParseOpHook =
      function_ref<ParseResult(OpAsmParser &parser, OperationState &result)>;

  virtual ~Dialect();

  /// Utility function that returns if the given string is a valid dialect
  /// namespace
  static bool isValidNamespace(StringRef str);

  MLIRContext *getContext() const { return context; }

  StringRef getNamespace() const { return name; }

  /// Returns the unique identifier that corresponds to this dialect.
  TypeID getTypeID() const { return dialectID; }

  /// Returns true if this dialect allows for unregistered operations, i.e.
  /// operations prefixed with the dialect namespace but not registered with
  /// addOperation.
  bool allowsUnknownOperations() const { return unknownOpsAllowed; }

  /// Return true if this dialect allows for unregistered types, i.e., types
  /// prefixed with the dialect namespace but not registered with addType.
  /// These are represented with OpaqueType.
  bool allowsUnknownTypes() const { return unknownTypesAllowed; }

  /// Register dialect-wide canonicalization patterns. This method should only
  /// be used to register canonicalization patterns that do not conceptually
  /// belong to any single operation in the dialect. (In that case, use the op's
  /// canonicalizer.) E.g., canonicalization patterns for op interfaces should
  /// be registered here.
  virtual void getCanonicalizationPatterns(RewritePatternSet &results) const {}

  /// Registered hook to materialize a single constant operation from a given
  /// attribute value with the desired resultant type. This method should use
  /// the provided builder to create the operation without changing the
  /// insertion position. The generated operation is expected to be constant
  /// like, i.e. single result, zero operands, non side-effecting, etc. On
  /// success, this hook should return the value generated to represent the
  /// constant value. Otherwise, it should return null on failure.
  virtual Operation *materializeConstant(OpBuilder &builder, Attribute value,
                                         Type type, Location loc) {
    return nullptr;
  }

  //===--------------------------------------------------------------------===//
  // Parsing Hooks
  //===--------------------------------------------------------------------===//

  /// Parse an attribute registered to this dialect. If 'type' is nonnull, it
  /// refers to the expected type of the attribute.
  virtual Attribute parseAttribute(DialectAsmParser &parser, Type type) const;

  /// Print an attribute registered to this dialect. Note: The type of the
  /// attribute need not be printed by this method as it is always printed by
  /// the caller.
  virtual void printAttribute(Attribute, DialectAsmPrinter &) const {
    llvm_unreachable("dialect has no registered attribute printing hook");
  }

  /// Parse a type registered to this dialect.
  virtual Type parseType(DialectAsmParser &parser) const;

  /// Print a type registered to this dialect.
  virtual void printType(Type, DialectAsmPrinter &) const {
    llvm_unreachable("dialect has no registered type printing hook");
  }

  /// Return the hook to parse an operation registered to this dialect, if any.
  /// By default this will lookup for registered operations and return the
  /// `parse()` method registered on the RegisteredOperationName. Dialects can
  /// override this behavior and handle unregistered operations as well.
  virtual Optional<ParseOpHook> getParseOperationHook(StringRef opName) const;

  /// Print an operation registered to this dialect.
  /// This hook is invoked for registered operation which don't override the
  /// `print()` method to define their own custom assembly.
  virtual llvm::unique_function<void(Operation *, OpAsmPrinter &printer)>
  getOperationPrinter(Operation *op) const;

  //===--------------------------------------------------------------------===//
  // Verification Hooks
  //===--------------------------------------------------------------------===//

  /// Verify an attribute from this dialect on the argument at 'argIndex' for
  /// the region at 'regionIndex' on the given operation. Returns failure if
  /// the verification failed, success otherwise. This hook may optionally be
  /// invoked from any operation containing a region.
  virtual LogicalResult verifyRegionArgAttribute(Operation *,
                                                 unsigned regionIndex,
                                                 unsigned argIndex,
                                                 NamedAttribute);

  /// Verify an attribute from this dialect on the result at 'resultIndex' for
  /// the region at 'regionIndex' on the given operation. Returns failure if
  /// the verification failed, success otherwise. This hook may optionally be
  /// invoked from any operation containing a region.
  virtual LogicalResult verifyRegionResultAttribute(Operation *,
                                                    unsigned regionIndex,
                                                    unsigned resultIndex,
                                                    NamedAttribute);

  /// Verify an attribute from this dialect on the given operation. Returns
  /// failure if the verification failed, success otherwise.
  virtual LogicalResult verifyOperationAttribute(Operation *, NamedAttribute) {
    return success();
  }

  //===--------------------------------------------------------------------===//
  // Interfaces
  //===--------------------------------------------------------------------===//

  /// Lookup an interface for the given ID if one is registered, otherwise
  /// nullptr.
  const DialectInterface *getRegisteredInterface(TypeID interfaceID) {
    auto it = registeredInterfaces.find(interfaceID);
    return it != registeredInterfaces.end() ? it->getSecond().get() : nullptr;
  }
  template <typename InterfaceT> const InterfaceT *getRegisteredInterface() {
    return static_cast<const InterfaceT *>(
        getRegisteredInterface(InterfaceT::getInterfaceID()));
  }

  /// Lookup an op interface for the given ID if one is registered, otherwise
  /// nullptr.
  virtual void *getRegisteredInterfaceForOp(TypeID interfaceID,
                                            OperationName opName) {
    return nullptr;
  }
  template <typename InterfaceT>
  typename InterfaceT::Concept *
  getRegisteredInterfaceForOp(OperationName opName) {
    return static_cast<typename InterfaceT::Concept *>(
        getRegisteredInterfaceForOp(InterfaceT::getInterfaceID(), opName));
  }

  /// Register a dialect interface with this dialect instance.
  void addInterface(std::unique_ptr<DialectInterface> interface);

  /// Register a set of dialect interfaces with this dialect instance.
  template <typename... Args>
  void addInterfaces() {
    (void)std::initializer_list<int>{
        0, (addInterface(std::make_unique<Args>(this)), 0)...};
  }

protected:
  /// The constructor takes a unique namespace for this dialect as well as the
  /// context to bind to.
  /// Note: The namespace must not contain '.' characters.
  /// Note: All operations belonging to this dialect must have names starting
  ///       with the namespace followed by '.'.
  /// Example:
  ///       - "tf" for the TensorFlow ops like "tf.add".
  Dialect(StringRef name, MLIRContext *context, TypeID id);

  /// This method is used by derived classes to add their operations to the set.
  ///
  template <typename... Args> void addOperations() {
    (void)std::initializer_list<int>{
        0, (RegisteredOperationName::insert<Args>(*this), 0)...};
  }

  /// Register a set of type classes with this dialect.
  template <typename... Args> void addTypes() {
    (void)std::initializer_list<int>{0, (addType<Args>(), 0)...};
  }

  /// Register a type instance with this dialect.
  /// The use of this method is in general discouraged in favor of
  /// 'addTypes<CustomType>()'.
  void addType(TypeID typeID, AbstractType &&typeInfo);

  /// Register a set of attribute classes with this dialect.
  template <typename... Args> void addAttributes() {
    (void)std::initializer_list<int>{0, (addAttribute<Args>(), 0)...};
  }

  /// Register an attribute instance with this dialect.
  /// The use of this method is in general discouraged in favor of
  /// 'addAttributes<CustomAttr>()'.
  void addAttribute(TypeID typeID, AbstractAttribute &&attrInfo);

  /// Enable support for unregistered operations.
  void allowUnknownOperations(bool allow = true) { unknownOpsAllowed = allow; }

  /// Enable support for unregistered types.
  void allowUnknownTypes(bool allow = true) { unknownTypesAllowed = allow; }

private:
  Dialect(const Dialect &) = delete;
  void operator=(Dialect &) = delete;

  /// Register an attribute instance with this dialect.
  template <typename T> void addAttribute() {
    // Add this attribute to the dialect and register it with the uniquer.
    addAttribute(T::getTypeID(), AbstractAttribute::get<T>(*this));
    detail::AttributeUniquer::registerAttribute<T>(context);
  }

  /// Register a type instance with this dialect.
  template <typename T> void addType() {
    // Add this type to the dialect and register it with the uniquer.
    addType(T::getTypeID(), AbstractType::get<T>(*this));
    detail::TypeUniquer::registerType<T>(context);
  }

  /// The namespace of this dialect.
  StringRef name;

  /// The unique identifier of the derived Op class, this is used in the context
  /// to allow registering multiple times the same dialect.
  TypeID dialectID;

  /// This is the context that owns this Dialect object.
  MLIRContext *context;

  /// Flag that specifies whether this dialect supports unregistered operations,
  /// i.e. operations prefixed with the dialect namespace but not registered
  /// with addOperation.
  bool unknownOpsAllowed = false;

  /// Flag that specifies whether this dialect allows unregistered types, i.e.
  /// types prefixed with the dialect namespace but not registered with addType.
  /// These types are represented with OpaqueType.
  bool unknownTypesAllowed = false;

  /// A collection of registered dialect interfaces.
  DenseMap<TypeID, std::unique_ptr<DialectInterface>> registeredInterfaces;

  friend class DialectRegistry;
  friend void registerDialect();
  friend class MLIRContext;
};

} // namespace mlir

namespace llvm {
/// Provide isa functionality for Dialects.
template <typename T>
struct isa_impl<T, ::mlir::Dialect,
                std::enable_if_t<std::is_base_of<::mlir::Dialect, T>::value>> {
  static inline bool doit(const ::mlir::Dialect &dialect) {
    return mlir::TypeID::get<T>() == dialect.getTypeID();
  }
};
template <typename T>
struct isa_impl<
    T, ::mlir::Dialect,
    std::enable_if_t<std::is_base_of<::mlir::DialectInterface, T>::value>> {
  static inline bool doit(const ::mlir::Dialect &dialect) {
    return const_cast<::mlir::Dialect &>(dialect).getRegisteredInterface<T>();
  }
};
template <typename T>
struct cast_retty_impl<T, ::mlir::Dialect *> {
  using ret_type =
      std::conditional_t<std::is_base_of<::mlir::Dialect, T>::value, T *,
                         const T *>;
};
template <typename T>
struct cast_retty_impl<T, ::mlir::Dialect> {
  using ret_type =
      std::conditional_t<std::is_base_of<::mlir::Dialect, T>::value, T &,
                         const T &>;
};

template <typename T>
struct cast_convert_val<T, ::mlir::Dialect, ::mlir::Dialect> {
  template <typename To>
  static std::enable_if_t<std::is_base_of<::mlir::Dialect, To>::value, To &>
  doitImpl(::mlir::Dialect &dialect) {
    return static_cast<To &>(dialect);
  }
  template <typename To>
  static std::enable_if_t<std::is_base_of<::mlir::DialectInterface, To>::value,
                          const To &>
  doitImpl(::mlir::Dialect &dialect) {
    return *dialect.getRegisteredInterface<To>();
  }

  static auto &doit(::mlir::Dialect &dialect) { return doitImpl<T>(dialect); }
};
template <class T>
struct cast_convert_val<T, ::mlir::Dialect *, ::mlir::Dialect *> {
  static auto doit(::mlir::Dialect *dialect) {
    return &cast_convert_val<T, ::mlir::Dialect, ::mlir::Dialect>::doit(
        *dialect);
  }
};

} // namespace llvm

#endif
