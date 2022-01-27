//===- SPIRVConversion.h - SPIR-V Conversion Utilities ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines utilities to use while converting to the SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPIRV_TRANSFORMS_SPIRVCONVERSION_H
#define MLIR_DIALECT_SPIRV_TRANSFORMS_SPIRVCONVERSION_H

#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallSet.h"

namespace mlir {

//===----------------------------------------------------------------------===//
// Type Converter
//===----------------------------------------------------------------------===//

/// Type conversion from builtin types to SPIR-V types for shader interface.
///
/// For memref types, this converter additionally performs type wrapping to
/// satisfy shader interface requirements: shader interface types must be
/// pointers to structs.
class SPIRVTypeConverter : public TypeConverter {
public:
  struct Options {
    /// Whether to emulate non-32-bit scalar types with 32-bit scalar types if
    /// no native support.
    ///
    /// Non-32-bit scalar types require special hardware support that may not
    /// exist on all GPUs. This is reflected in SPIR-V as that non-32-bit scalar
    /// types require special capabilities or extensions. This option controls
    /// whether to use 32-bit types to emulate, if a scalar type of a certain
    /// bitwidth is not supported in the target environment. This requires the
    /// runtime to also feed in data with a matched bitwidth and layout for
    /// interface types. The runtime can do that by inspecting the SPIR-V
    /// module.
    ///
    /// If the original scalar type has less than 32-bit, a multiple of its
    /// values will be packed into one 32-bit value to be memory efficient.
    bool emulateNon32BitScalarTypes{true};

    /// Use 64-bit integers to convert index types.
    bool use64bitIndex{false};

    /// The number of bits to store a boolean value. It is eight bits by
    /// default.
    unsigned boolNumBits{8};

    // Note: we need this instead of inline initializers because of
    // https://bugs.llvm.org/show_bug.cgi?id=36684
    Options()

    {}
  };

  explicit SPIRVTypeConverter(spirv::TargetEnvAttr targetAttr,
                              Options options = {});

  /// Gets the SPIR-V correspondence for the standard index type.
  Type getIndexType() const;

  /// Returns the corresponding memory space for memref given a SPIR-V storage
  /// class.
  static unsigned getMemorySpaceForStorageClass(spirv::StorageClass);

  /// Returns the SPIR-V storage class given a memory space for memref. Return
  /// llvm::None if the memory space does not map to any SPIR-V storage class.
  static Optional<spirv::StorageClass>
  getStorageClassForMemorySpace(unsigned space);

  /// Returns the options controlling the SPIR-V type converter.
  const Options &getOptions() const;

private:
  spirv::TargetEnv targetEnv;
  Options options;

  MLIRContext *getContext() const;
};

//===----------------------------------------------------------------------===//
// Conversion Target
//===----------------------------------------------------------------------===//

// The default SPIR-V conversion target.
//
// It takes a SPIR-V target environment and controls operation legality based on
// the their availability in the target environment.
class SPIRVConversionTarget : public ConversionTarget {
public:
  /// Creates a SPIR-V conversion target for the given target environment.
  static std::unique_ptr<SPIRVConversionTarget>
  get(spirv::TargetEnvAttr targetAttr);

private:
  explicit SPIRVConversionTarget(spirv::TargetEnvAttr targetAttr);

  // Be explicit that instance of this class cannot be copied or moved: there
  // are lambdas capturing fields of the instance.
  SPIRVConversionTarget(const SPIRVConversionTarget &) = delete;
  SPIRVConversionTarget(SPIRVConversionTarget &&) = delete;
  SPIRVConversionTarget &operator=(const SPIRVConversionTarget &) = delete;
  SPIRVConversionTarget &operator=(SPIRVConversionTarget &&) = delete;

  /// Returns true if the given `op` is legal to use under the current target
  /// environment.
  bool isLegalOp(Operation *op);

  spirv::TargetEnv targetEnv;
};

//===----------------------------------------------------------------------===//
// Patterns and Utility Functions
//===----------------------------------------------------------------------===//

/// Appends to a pattern list additional patterns for translating the builtin
/// `func` op to the SPIR-V dialect. These patterns do not handle shader
/// interface/ABI; they convert function parameters to be of SPIR-V allowed
/// types.
void populateBuiltinFuncToSPIRVPatterns(SPIRVTypeConverter &typeConverter,
                                        RewritePatternSet &patterns);

namespace spirv {
class AccessChainOp;

/// Returns the value for the given `builtin` variable. This function gets or
/// inserts the global variable associated for the builtin within the nearest
/// symbol table enclosing `op`. Returns null Value on error.
Value getBuiltinVariableValue(Operation *op, BuiltIn builtin, Type integerType,
                              OpBuilder &builder);

/// Gets the value at the given `offset` of the push constant storage with a
/// total of `elementCount` `integerType` integers. A global variable will be
/// created in the nearest symbol table enclosing `op` for the push constant
/// storage if not existing. Load ops will be created via the given `builder` to
/// load values from the push constant. Returns null Value on error.
Value getPushConstantValue(Operation *op, unsigned elementCount,
                           unsigned offset, Type integerType,
                           OpBuilder &builder);

/// Generates IR to perform index linearization with the given `indices` and
/// their corresponding `strides`, adding an initial `offset`.
Value linearizeIndex(ValueRange indices, ArrayRef<int64_t> strides,
                     int64_t offset, Type integerType, Location loc,
                     OpBuilder &builder);

/// Performs the index computation to get to the element at `indices` of the
/// memory pointed to by `basePtr`, using the layout map of `baseType`.
/// Returns null if index computation cannot be performed.

// TODO: This method assumes that the `baseType` is a MemRefType with AffineMap
// that has static strides. Extend to handle dynamic strides.
spirv::AccessChainOp getElementPtr(SPIRVTypeConverter &typeConverter,
                                   MemRefType baseType, Value basePtr,
                                   ValueRange indices, Location loc,
                                   OpBuilder &builder);

} // namespace spirv
} // namespace mlir

#endif // MLIR_DIALECT_SPIRV_TRANSFORMS_SPIRVCONVERSION_H
