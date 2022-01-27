//===- Interfaces.h - Interface wrapper classes -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_INTERFACES_H_
#define MLIR_TABLEGEN_INTERFACES_H_

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {
class Init;
class Record;
} // namespace llvm

namespace mlir {
namespace tblgen {

// Wrapper class with helper methods for accessing InterfaceMethod defined
// in TableGen.
class InterfaceMethod {
public:
  // This struct represents a single method argument.
  struct Argument {
    StringRef type;
    StringRef name;
  };

  explicit InterfaceMethod(const llvm::Record *def);

  // Return the return type of this method.
  StringRef getReturnType() const;

  // Return the name of this method.
  StringRef getName() const;

  // Return if this method is static.
  bool isStatic() const;

  // Return the body for this method if it has one.
  llvm::Optional<StringRef> getBody() const;

  // Return the default implementation for this method if it has one.
  llvm::Optional<StringRef> getDefaultImplementation() const;

  // Return the description of this method if it has one.
  llvm::Optional<StringRef> getDescription() const;

  // Arguments.
  ArrayRef<Argument> getArguments() const;
  bool arg_empty() const;

private:
  // The TableGen definition of this method.
  const llvm::Record *def;

  // The arguments of this method.
  SmallVector<Argument, 2> arguments;
};

//===----------------------------------------------------------------------===//
// Interface
//===----------------------------------------------------------------------===//

// Wrapper class with helper methods for accessing Interfaces defined in
// TableGen.
class Interface {
public:
  explicit Interface(const llvm::Record *def);

  // Return the name of this interface.
  StringRef getName() const;

  // Return the C++ namespace of this interface.
  StringRef getCppNamespace() const;

  // Return the methods of this interface.
  ArrayRef<InterfaceMethod> getMethods() const;

  // Return the description of this method if it has one.
  llvm::Optional<StringRef> getDescription() const;

  // Return the interfaces extra class declaration code.
  llvm::Optional<StringRef> getExtraClassDeclaration() const;

  // Return the traits extra class declaration code.
  llvm::Optional<StringRef> getExtraTraitClassDeclaration() const;

  // Return the verify method body if it has one.
  llvm::Optional<StringRef> getVerify() const;

  // Returns the Tablegen definition this interface was constructed from.
  const llvm::Record &getDef() const { return *def; }

private:
  // The TableGen definition of this interface.
  const llvm::Record *def;

  // The methods of this interface.
  SmallVector<InterfaceMethod, 8> methods;
};

// An interface that is registered to an Attribute.
struct AttrInterface : public Interface {
  using Interface::Interface;

  static bool classof(const Interface *interface);
};
// An interface that is registered to an Operation.
struct OpInterface : public Interface {
  using Interface::Interface;

  static bool classof(const Interface *interface);
};
// An interface that is registered to a Type.
struct TypeInterface : public Interface {
  using Interface::Interface;

  static bool classof(const Interface *interface);
};
} // namespace tblgen
} // namespace mlir

#endif // MLIR_TABLEGEN_INTERFACES_H_
