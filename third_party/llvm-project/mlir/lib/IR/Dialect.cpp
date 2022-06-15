//===- Dialect.cpp - Dialect implementation -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Regex.h"

#define DEBUG_TYPE "dialect"

using namespace mlir;
using namespace detail;

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

Dialect::Dialect(StringRef name, MLIRContext *context, TypeID id)
    : name(name), dialectID(id), context(context) {
  assert(isValidNamespace(name) && "invalid dialect namespace");
}

Dialect::~Dialect() = default;

/// Verify an attribute from this dialect on the argument at 'argIndex' for
/// the region at 'regionIndex' on the given operation. Returns failure if
/// the verification failed, success otherwise. This hook may optionally be
/// invoked from any operation containing a region.
LogicalResult Dialect::verifyRegionArgAttribute(Operation *, unsigned, unsigned,
                                                NamedAttribute) {
  return success();
}

/// Verify an attribute from this dialect on the result at 'resultIndex' for
/// the region at 'regionIndex' on the given operation. Returns failure if
/// the verification failed, success otherwise. This hook may optionally be
/// invoked from any operation containing a region.
LogicalResult Dialect::verifyRegionResultAttribute(Operation *, unsigned,
                                                   unsigned, NamedAttribute) {
  return success();
}

/// Parse an attribute registered to this dialect.
Attribute Dialect::parseAttribute(DialectAsmParser &parser, Type type) const {
  parser.emitError(parser.getNameLoc())
      << "dialect '" << getNamespace()
      << "' provides no attribute parsing hook";
  return Attribute();
}

/// Parse a type registered to this dialect.
Type Dialect::parseType(DialectAsmParser &parser) const {
  // If this dialect allows unknown types, then represent this with OpaqueType.
  if (allowsUnknownTypes()) {
    StringAttr ns = StringAttr::get(getContext(), getNamespace());
    return OpaqueType::get(ns, parser.getFullSymbolSpec());
  }

  parser.emitError(parser.getNameLoc())
      << "dialect '" << getNamespace() << "' provides no type parsing hook";
  return Type();
}

Optional<Dialect::ParseOpHook>
Dialect::getParseOperationHook(StringRef opName) const {
  return None;
}

llvm::unique_function<void(Operation *, OpAsmPrinter &printer)>
Dialect::getOperationPrinter(Operation *op) const {
  assert(op->getDialect() == this &&
         "Dialect hook invoked on non-dialect owned operation");
  return nullptr;
}

/// Utility function that returns if the given string is a valid dialect
/// namespace
bool Dialect::isValidNamespace(StringRef str) {
  llvm::Regex dialectNameRegex("^[a-zA-Z_][a-zA-Z_0-9\\$]*$");
  return dialectNameRegex.match(str);
}

/// Register a set of dialect interfaces with this dialect instance.
void Dialect::addInterface(std::unique_ptr<DialectInterface> interface) {
  auto it = registeredInterfaces.try_emplace(interface->getID(),
                                             std::move(interface));
  (void)it;
  LLVM_DEBUG({
    if (!it.second) {
      llvm::dbgs() << "[" DEBUG_TYPE
                      "] repeated interface registration for dialect "
                   << getNamespace();
    }
  });
}

//===----------------------------------------------------------------------===//
// Dialect Interface
//===----------------------------------------------------------------------===//

DialectInterface::~DialectInterface() = default;

DialectInterfaceCollectionBase::DialectInterfaceCollectionBase(
    MLIRContext *ctx, TypeID interfaceKind) {
  for (auto *dialect : ctx->getLoadedDialects()) {
    if (auto *interface = dialect->getRegisteredInterface(interfaceKind)) {
      interfaces.insert(interface);
      orderedInterfaces.push_back(interface);
    }
  }
}

DialectInterfaceCollectionBase::~DialectInterfaceCollectionBase() = default;

/// Get the interface for the dialect of given operation, or null if one
/// is not registered.
const DialectInterface *
DialectInterfaceCollectionBase::getInterfaceFor(Operation *op) const {
  return getInterfaceFor(op->getDialect());
}

//===----------------------------------------------------------------------===//
// DialectExtension
//===----------------------------------------------------------------------===//

DialectExtensionBase::~DialectExtensionBase() = default;

//===----------------------------------------------------------------------===//
// DialectRegistry
//===----------------------------------------------------------------------===//

DialectRegistry::DialectRegistry() { insert<BuiltinDialect>(); }

DialectAllocatorFunctionRef
DialectRegistry::getDialectAllocator(StringRef name) const {
  auto it = registry.find(name.str());
  if (it == registry.end())
    return nullptr;
  return it->second.second;
}

void DialectRegistry::insert(TypeID typeID, StringRef name,
                             const DialectAllocatorFunction &ctor) {
  auto inserted = registry.insert(
      std::make_pair(std::string(name), std::make_pair(typeID, ctor)));
  if (!inserted.second && inserted.first->second.first != typeID) {
    llvm::report_fatal_error(
        "Trying to register different dialects for the same namespace: " +
        name);
  }
}

void DialectRegistry::applyExtensions(Dialect *dialect) const {
  MLIRContext *ctx = dialect->getContext();
  StringRef dialectName = dialect->getNamespace();

  // Functor used to try to apply the given extension.
  auto applyExtension = [&](const DialectExtensionBase &extension) {
    ArrayRef<StringRef> dialectNames = extension.getRequiredDialects();

    // Handle the simple case of a single dialect name. In this case, the
    // required dialect should be the current dialect.
    if (dialectNames.size() == 1) {
      if (dialectNames.front() == dialectName)
        extension.apply(ctx, dialect);
      return;
    }

    // Otherwise, check to see if this extension requires this dialect.
    const StringRef *nameIt = llvm::find(dialectNames, dialectName);
    if (nameIt == dialectNames.end())
      return;

    // If it does, ensure that all of the other required dialects have been
    // loaded.
    SmallVector<Dialect *> requiredDialects;
    requiredDialects.reserve(dialectNames.size());
    for (auto it = dialectNames.begin(), e = dialectNames.end(); it != e;
         ++it) {
      // The current dialect is known to be loaded.
      if (it == nameIt) {
        requiredDialects.push_back(dialect);
        continue;
      }
      // Otherwise, check if it is loaded.
      Dialect *loadedDialect = ctx->getLoadedDialect(*it);
      if (!loadedDialect)
        return;
      requiredDialects.push_back(loadedDialect);
    }
    extension.apply(ctx, requiredDialects);
  };

  for (const auto &extension : extensions)
    applyExtension(*extension);
}

void DialectRegistry::applyExtensions(MLIRContext *ctx) const {
  // Functor used to try to apply the given extension.
  auto applyExtension = [&](const DialectExtensionBase &extension) {
    ArrayRef<StringRef> dialectNames = extension.getRequiredDialects();

    // Check to see if all of the dialects for this extension are loaded.
    SmallVector<Dialect *> requiredDialects;
    requiredDialects.reserve(dialectNames.size());
    for (StringRef dialectName : dialectNames) {
      Dialect *loadedDialect = ctx->getLoadedDialect(dialectName);
      if (!loadedDialect)
        return;
      requiredDialects.push_back(loadedDialect);
    }
    extension.apply(ctx, requiredDialects);
  };

  for (const auto &extension : extensions)
    applyExtension(*extension);
}

bool DialectRegistry::isSubsetOf(const DialectRegistry &rhs) const {
  // Treat any extensions conservatively.
  if (!extensions.empty())
    return false;
  // Check that the current dialects fully overlap with the dialects in 'rhs'.
  return llvm::all_of(
      registry, [&](const auto &it) { return rhs.registry.count(it.first); });
}
