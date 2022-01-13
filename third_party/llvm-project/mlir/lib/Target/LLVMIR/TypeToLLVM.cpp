//===- TypeToLLVM.cpp - type translation from MLIR to LLVM IR -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVMIR/TypeToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Type.h"

using namespace mlir;

namespace mlir {
namespace LLVM {
namespace detail {
/// Support for translating MLIR LLVM dialect types to LLVM IR.
class TypeToLLVMIRTranslatorImpl {
public:
  /// Constructs a class creating types in the given LLVM context.
  TypeToLLVMIRTranslatorImpl(llvm::LLVMContext &context) : context(context) {}

  /// Translates a single type.
  llvm::Type *translateType(Type type) {
    // If the conversion is already known, just return it.
    if (knownTranslations.count(type))
      return knownTranslations.lookup(type);

    // Dispatch to an appropriate function.
    llvm::Type *translated =
        llvm::TypeSwitch<Type, llvm::Type *>(type)
            .Case([this](LLVM::LLVMVoidType) {
              return llvm::Type::getVoidTy(context);
            })
            .Case(
                [this](Float16Type) { return llvm::Type::getHalfTy(context); })
            .Case([this](BFloat16Type) {
              return llvm::Type::getBFloatTy(context);
            })
            .Case(
                [this](Float32Type) { return llvm::Type::getFloatTy(context); })
            .Case([this](Float64Type) {
              return llvm::Type::getDoubleTy(context);
            })
            .Case([this](Float80Type) {
              return llvm::Type::getX86_FP80Ty(context);
            })
            .Case([this](Float128Type) {
              return llvm::Type::getFP128Ty(context);
            })
            .Case([this](LLVM::LLVMPPCFP128Type) {
              return llvm::Type::getPPC_FP128Ty(context);
            })
            .Case([this](LLVM::LLVMX86MMXType) {
              return llvm::Type::getX86_MMXTy(context);
            })
            .Case([this](LLVM::LLVMTokenType) {
              return llvm::Type::getTokenTy(context);
            })
            .Case([this](LLVM::LLVMLabelType) {
              return llvm::Type::getLabelTy(context);
            })
            .Case([this](LLVM::LLVMMetadataType) {
              return llvm::Type::getMetadataTy(context);
            })
            .Case<LLVM::LLVMArrayType, IntegerType, LLVM::LLVMFunctionType,
                  LLVM::LLVMPointerType, LLVM::LLVMStructType,
                  LLVM::LLVMFixedVectorType, LLVM::LLVMScalableVectorType,
                  VectorType>(
                [this](auto type) { return this->translate(type); })
            .Default([](Type t) -> llvm::Type * {
              llvm_unreachable("unknown LLVM dialect type");
            });

    // Cache the result of the conversion and return.
    knownTranslations.try_emplace(type, translated);
    return translated;
  }

private:
  /// Translates the given array type.
  llvm::Type *translate(LLVM::LLVMArrayType type) {
    return llvm::ArrayType::get(translateType(type.getElementType()),
                                type.getNumElements());
  }

  /// Translates the given function type.
  llvm::Type *translate(LLVM::LLVMFunctionType type) {
    SmallVector<llvm::Type *, 8> paramTypes;
    translateTypes(type.getParams(), paramTypes);
    return llvm::FunctionType::get(translateType(type.getReturnType()),
                                   paramTypes, type.isVarArg());
  }

  /// Translates the given integer type.
  llvm::Type *translate(IntegerType type) {
    return llvm::IntegerType::get(context, type.getWidth());
  }

  /// Translates the given pointer type.
  llvm::Type *translate(LLVM::LLVMPointerType type) {
    return llvm::PointerType::get(translateType(type.getElementType()),
                                  type.getAddressSpace());
  }

  /// Translates the given structure type, supports both identified and literal
  /// structs. This will _create_ a new identified structure every time, use
  /// `convertType` if a structure with the same name must be looked up instead.
  llvm::Type *translate(LLVM::LLVMStructType type) {
    SmallVector<llvm::Type *, 8> subtypes;
    if (!type.isIdentified()) {
      translateTypes(type.getBody(), subtypes);
      return llvm::StructType::get(context, subtypes, type.isPacked());
    }

    llvm::StructType *structType =
        llvm::StructType::create(context, type.getName());
    // Mark the type we just created as known so that recursive calls can pick
    // it up and use directly.
    knownTranslations.try_emplace(type, structType);
    if (type.isOpaque())
      return structType;

    translateTypes(type.getBody(), subtypes);
    structType->setBody(subtypes, type.isPacked());
    return structType;
  }

  /// Translates the given built-in vector type compatible with LLVM.
  llvm::Type *translate(VectorType type) {
    assert(LLVM::isCompatibleVectorType(type) &&
           "expected compatible with LLVM vector type");
    return llvm::FixedVectorType::get(translateType(type.getElementType()),
                                      type.getNumElements());
  }

  /// Translates the given fixed-vector type.
  llvm::Type *translate(LLVM::LLVMFixedVectorType type) {
    return llvm::FixedVectorType::get(translateType(type.getElementType()),
                                      type.getNumElements());
  }

  /// Translates the given scalable-vector type.
  llvm::Type *translate(LLVM::LLVMScalableVectorType type) {
    return llvm::ScalableVectorType::get(translateType(type.getElementType()),
                                         type.getMinNumElements());
  }

  /// Translates a list of types.
  void translateTypes(ArrayRef<Type> types,
                      SmallVectorImpl<llvm::Type *> &result) {
    result.reserve(result.size() + types.size());
    for (auto type : types)
      result.push_back(translateType(type));
  }

  /// Reference to the context in which the LLVM IR types are created.
  llvm::LLVMContext &context;

  /// Map of known translation. This serves a double purpose: caches translation
  /// results to avoid repeated recursive calls and makes sure identified
  /// structs with the same name (that is, equal) are resolved to an existing
  /// type instead of creating a new type.
  llvm::DenseMap<Type, llvm::Type *> knownTranslations;
};
} // end namespace detail
} // end namespace LLVM
} // end namespace mlir

LLVM::TypeToLLVMIRTranslator::TypeToLLVMIRTranslator(llvm::LLVMContext &context)
    : impl(new detail::TypeToLLVMIRTranslatorImpl(context)) {}

LLVM::TypeToLLVMIRTranslator::~TypeToLLVMIRTranslator() {}

llvm::Type *LLVM::TypeToLLVMIRTranslator::translateType(Type type) {
  return impl->translateType(type);
}

unsigned LLVM::TypeToLLVMIRTranslator::getPreferredAlignment(
    Type type, const llvm::DataLayout &layout) {
  return layout.getPrefTypeAlignment(translateType(type));
}
