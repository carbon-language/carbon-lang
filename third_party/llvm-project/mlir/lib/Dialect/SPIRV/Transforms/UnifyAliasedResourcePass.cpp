//===- UnifyAliasedResourcePass.cpp - Pass to Unify Aliased Resources -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that unifies access of multiple aliased resources
// into access of one single resource.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <iterator>

#define DEBUG_TYPE "spirv-unify-aliased-resource"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

using Descriptor = std::pair<uint32_t, uint32_t>; // (set #, binding #)
using AliasedResourceMap =
    DenseMap<Descriptor, SmallVector<spirv::GlobalVariableOp>>;

/// Collects all aliased resources in the given SPIR-V `moduleOp`.
static AliasedResourceMap collectAliasedResources(spirv::ModuleOp moduleOp) {
  AliasedResourceMap aliasedResources;
  moduleOp->walk([&aliasedResources](spirv::GlobalVariableOp varOp) {
    if (varOp->getAttrOfType<UnitAttr>("aliased")) {
      Optional<uint32_t> set = varOp.descriptor_set();
      Optional<uint32_t> binding = varOp.binding();
      if (set && binding)
        aliasedResources[{*set, *binding}].push_back(varOp);
    }
  });
  return aliasedResources;
}

/// Returns the element type if the given `type` is a runtime array resource:
/// `!spv.ptr<!spv.struct<!spv.rtarray<...>>>`. Returns null type otherwise.
static Type getRuntimeArrayElementType(Type type) {
  auto ptrType = type.dyn_cast<spirv::PointerType>();
  if (!ptrType)
    return {};

  auto structType = ptrType.getPointeeType().dyn_cast<spirv::StructType>();
  if (!structType || structType.getNumElements() != 1)
    return {};

  auto rtArrayType =
      structType.getElementType(0).dyn_cast<spirv::RuntimeArrayType>();
  if (!rtArrayType)
    return {};

  return rtArrayType.getElementType();
}

/// Given a list of resource element `types`, returns the index of the canonical
/// resource that all resources should be unified into. Returns llvm::None if
/// unable to unify.
static Optional<int> deduceCanonicalResource(ArrayRef<spirv::SPIRVType> types) {
  SmallVector<int> scalarNumBits, totalNumBits;
  scalarNumBits.reserve(types.size());
  totalNumBits.reserve(types.size());
  bool hasVector = false;

  for (spirv::SPIRVType type : types) {
    assert(type.isScalarOrVector());
    if (auto vectorType = type.dyn_cast<VectorType>()) {
      if (vectorType.getNumElements() % 2 != 0)
        return llvm::None; // Odd-sized vector has special layout requirements.

      Optional<int64_t> numBytes = type.getSizeInBytes();
      if (!numBytes)
        return llvm::None;

      scalarNumBits.push_back(
          vectorType.getElementType().getIntOrFloatBitWidth());
      totalNumBits.push_back(*numBytes * 8);
      hasVector = true;
    } else {
      scalarNumBits.push_back(type.getIntOrFloatBitWidth());
      totalNumBits.push_back(scalarNumBits.back());
    }
  }

  if (hasVector) {
    // If there are vector types, require all element types to be the same for
    // now to simplify the transformation.
    if (!llvm::is_splat(scalarNumBits))
      return llvm::None;

    // Choose the one with the largest bitwidth as the canonical resource, so
    // that we can still keep vectorized load/store.
    auto *maxVal = std::max_element(totalNumBits.begin(), totalNumBits.end());
    // Make sure that the canonical resource's bitwidth is divisible by others.
    // With out this, we cannot properly adjust the index later.
    if (llvm::any_of(totalNumBits,
                     [maxVal](int64_t bits) { return *maxVal % bits != 0; }))
      return llvm::None;

    return std::distance(totalNumBits.begin(), maxVal);
  }

  // All element types are scalars. Then choose the smallest bitwidth as the
  // cannonical resource to avoid subcomponent load/store.
  auto *minVal = std::min_element(scalarNumBits.begin(), scalarNumBits.end());
  if (llvm::any_of(scalarNumBits,
                   [minVal](int64_t bit) { return bit % *minVal != 0; }))
    return llvm::None;
  return std::distance(scalarNumBits.begin(), minVal);
}

static bool areSameBitwidthScalarType(Type a, Type b) {
  return a.isIntOrFloat() && b.isIntOrFloat() &&
         a.getIntOrFloatBitWidth() == b.getIntOrFloatBitWidth();
}

//===----------------------------------------------------------------------===//
// Analysis
//===----------------------------------------------------------------------===//

namespace {
/// A class for analyzing aliased resources.
///
/// Resources are expected to be spv.GlobalVarible that has a descriptor set and
/// binding number. Such resources are of the type `!spv.ptr<!spv.struct<...>>`
/// per Vulkan requirements.
///
/// Right now, we only support the case that there is a single runtime array
/// inside the struct.
class ResourceAliasAnalysis {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ResourceAliasAnalysis)

  explicit ResourceAliasAnalysis(Operation *);

  /// Returns true if the given `op` can be rewritten to use a canonical
  /// resource.
  bool shouldUnify(Operation *op) const;

  /// Returns all descriptors and their corresponding aliased resources.
  const AliasedResourceMap &getResourceMap() const { return resourceMap; }

  /// Returns the canonical resource for the given descriptor/variable.
  spirv::GlobalVariableOp
  getCanonicalResource(const Descriptor &descriptor) const;
  spirv::GlobalVariableOp
  getCanonicalResource(spirv::GlobalVariableOp varOp) const;

  /// Returns the element type for the given variable.
  spirv::SPIRVType getElementType(spirv::GlobalVariableOp varOp) const;

private:
  /// Given the descriptor and aliased resources bound to it, analyze whether we
  /// can unify them and record if so.
  void recordIfUnifiable(const Descriptor &descriptor,
                         ArrayRef<spirv::GlobalVariableOp> resources);

  /// Mapping from a descriptor to all aliased resources bound to it.
  AliasedResourceMap resourceMap;

  /// Mapping from a descriptor to the chosen canonical resource.
  DenseMap<Descriptor, spirv::GlobalVariableOp> canonicalResourceMap;

  /// Mapping from an aliased resource to its descriptor.
  DenseMap<spirv::GlobalVariableOp, Descriptor> descriptorMap;

  /// Mapping from an aliased resource to its element (scalar/vector) type.
  DenseMap<spirv::GlobalVariableOp, spirv::SPIRVType> elementTypeMap;
};
} // namespace

ResourceAliasAnalysis::ResourceAliasAnalysis(Operation *root) {
  // Collect all aliased resources first and put them into different sets
  // according to the descriptor.
  AliasedResourceMap aliasedResources =
      collectAliasedResources(cast<spirv::ModuleOp>(root));

  // For each resource set, analyze whether we can unify; if so, try to identify
  // a canonical resource, whose element type has the largest bitwidth.
  for (const auto &descriptorResource : aliasedResources) {
    recordIfUnifiable(descriptorResource.first, descriptorResource.second);
  }
}

bool ResourceAliasAnalysis::shouldUnify(Operation *op) const {
  if (auto varOp = dyn_cast<spirv::GlobalVariableOp>(op)) {
    auto canonicalOp = getCanonicalResource(varOp);
    return canonicalOp && varOp != canonicalOp;
  }
  if (auto addressOp = dyn_cast<spirv::AddressOfOp>(op)) {
    auto moduleOp = addressOp->getParentOfType<spirv::ModuleOp>();
    auto *varOp = SymbolTable::lookupSymbolIn(moduleOp, addressOp.variable());
    return shouldUnify(varOp);
  }

  if (auto acOp = dyn_cast<spirv::AccessChainOp>(op))
    return shouldUnify(acOp.base_ptr().getDefiningOp());
  if (auto loadOp = dyn_cast<spirv::LoadOp>(op))
    return shouldUnify(loadOp.ptr().getDefiningOp());
  if (auto storeOp = dyn_cast<spirv::StoreOp>(op))
    return shouldUnify(storeOp.ptr().getDefiningOp());

  return false;
}

spirv::GlobalVariableOp ResourceAliasAnalysis::getCanonicalResource(
    const Descriptor &descriptor) const {
  auto varIt = canonicalResourceMap.find(descriptor);
  if (varIt == canonicalResourceMap.end())
    return {};
  return varIt->second;
}

spirv::GlobalVariableOp ResourceAliasAnalysis::getCanonicalResource(
    spirv::GlobalVariableOp varOp) const {
  auto descriptorIt = descriptorMap.find(varOp);
  if (descriptorIt == descriptorMap.end())
    return {};
  return getCanonicalResource(descriptorIt->second);
}

spirv::SPIRVType
ResourceAliasAnalysis::getElementType(spirv::GlobalVariableOp varOp) const {
  auto it = elementTypeMap.find(varOp);
  if (it == elementTypeMap.end())
    return {};
  return it->second;
}

void ResourceAliasAnalysis::recordIfUnifiable(
    const Descriptor &descriptor, ArrayRef<spirv::GlobalVariableOp> resources) {
  // Collect the element types for all resources in the current set.
  SmallVector<spirv::SPIRVType> elementTypes;
  for (spirv::GlobalVariableOp resource : resources) {
    Type elementType = getRuntimeArrayElementType(resource.type());
    if (!elementType)
      return; // Unexpected resource variable type.

    auto type = elementType.cast<spirv::SPIRVType>();
    if (!type.isScalarOrVector())
      return; // Unexpected resource element type.

    elementTypes.push_back(type);
  }

  Optional<int> index = deduceCanonicalResource(elementTypes);
  if (!index)
    return;

  // Update internal data structures for later use.
  resourceMap[descriptor].assign(resources.begin(), resources.end());
  canonicalResourceMap[descriptor] = resources[*index];
  for (const auto &resource : llvm::enumerate(resources)) {
    descriptorMap[resource.value()] = descriptor;
    elementTypeMap[resource.value()] = elementTypes[resource.index()];
  }
}

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//

template <typename OpTy>
class ConvertAliasResource : public OpConversionPattern<OpTy> {
public:
  ConvertAliasResource(const ResourceAliasAnalysis &analysis,
                       MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern<OpTy>(context, benefit), analysis(analysis) {}

protected:
  const ResourceAliasAnalysis &analysis;
};

struct ConvertVariable : public ConvertAliasResource<spirv::GlobalVariableOp> {
  using ConvertAliasResource::ConvertAliasResource;

  LogicalResult
  matchAndRewrite(spirv::GlobalVariableOp varOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Just remove the aliased resource. Users will be rewritten to use the
    // canonical one.
    rewriter.eraseOp(varOp);
    return success();
  }
};

struct ConvertAddressOf : public ConvertAliasResource<spirv::AddressOfOp> {
  using ConvertAliasResource::ConvertAliasResource;

  LogicalResult
  matchAndRewrite(spirv::AddressOfOp addressOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Rewrite the AddressOf op to get the address of the canoncical resource.
    auto moduleOp = addressOp->getParentOfType<spirv::ModuleOp>();
    auto srcVarOp = cast<spirv::GlobalVariableOp>(
        SymbolTable::lookupSymbolIn(moduleOp, addressOp.variable()));
    auto dstVarOp = analysis.getCanonicalResource(srcVarOp);
    rewriter.replaceOpWithNewOp<spirv::AddressOfOp>(addressOp, dstVarOp);
    return success();
  }
};

struct ConvertAccessChain : public ConvertAliasResource<spirv::AccessChainOp> {
  using ConvertAliasResource::ConvertAliasResource;

  LogicalResult
  matchAndRewrite(spirv::AccessChainOp acOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto addressOp = acOp.base_ptr().getDefiningOp<spirv::AddressOfOp>();
    if (!addressOp)
      return rewriter.notifyMatchFailure(acOp, "base ptr not addressof op");

    auto moduleOp = acOp->getParentOfType<spirv::ModuleOp>();
    auto srcVarOp = cast<spirv::GlobalVariableOp>(
        SymbolTable::lookupSymbolIn(moduleOp, addressOp.variable()));
    auto dstVarOp = analysis.getCanonicalResource(srcVarOp);

    spirv::SPIRVType srcElemType = analysis.getElementType(srcVarOp);
    spirv::SPIRVType dstElemType = analysis.getElementType(dstVarOp);

    if (srcElemType == dstElemType ||
        areSameBitwidthScalarType(srcElemType, dstElemType)) {
      // We have the same bitwidth for source and destination element types.
      // Thie indices keep the same.
      rewriter.replaceOpWithNewOp<spirv::AccessChainOp>(
          acOp, adaptor.base_ptr(), adaptor.indices());
      return success();
    }

    Location loc = acOp.getLoc();
    auto i32Type = rewriter.getI32Type();

    if (srcElemType.isIntOrFloat() && dstElemType.isa<VectorType>()) {
      // The source indices are for a buffer with scalar element types. Rewrite
      // them into a buffer with vector element types. We need to scale the last
      // index for the vector as a whole, then add one level of index for inside
      // the vector.
      int srcNumBits = *srcElemType.getSizeInBytes();
      int dstNumBits = *dstElemType.getSizeInBytes();
      assert(dstNumBits > srcNumBits && dstNumBits % srcNumBits == 0);
      int ratio = dstNumBits / srcNumBits;
      auto ratioValue = rewriter.create<spirv::ConstantOp>(
          loc, i32Type, rewriter.getI32IntegerAttr(ratio));

      auto indices = llvm::to_vector<4>(acOp.indices());
      Value oldIndex = indices.back();
      indices.back() =
          rewriter.create<spirv::SDivOp>(loc, i32Type, oldIndex, ratioValue);
      indices.push_back(
          rewriter.create<spirv::SModOp>(loc, i32Type, oldIndex, ratioValue));

      rewriter.replaceOpWithNewOp<spirv::AccessChainOp>(
          acOp, adaptor.base_ptr(), indices);
      return success();
    }

    if (srcElemType.isIntOrFloat() && dstElemType.isIntOrFloat()) {
      // The source indices are for a buffer with larger bitwidth scalar element
      // types. Rewrite them into a buffer with smaller bitwidth element types.
      // We only need to scale the last index.
      int srcNumBits = *srcElemType.getSizeInBytes();
      int dstNumBits = *dstElemType.getSizeInBytes();
      assert(srcNumBits > dstNumBits && srcNumBits % dstNumBits == 0);
      int ratio = srcNumBits / dstNumBits;
      auto ratioValue = rewriter.create<spirv::ConstantOp>(
          loc, i32Type, rewriter.getI32IntegerAttr(ratio));

      auto indices = llvm::to_vector<4>(acOp.indices());
      Value oldIndex = indices.back();
      indices.back() =
          rewriter.create<spirv::IMulOp>(loc, i32Type, oldIndex, ratioValue);

      rewriter.replaceOpWithNewOp<spirv::AccessChainOp>(
          acOp, adaptor.base_ptr(), indices);
      return success();
    }

    return rewriter.notifyMatchFailure(acOp, "unsupported src/dst types");
  }
};

struct ConvertLoad : public ConvertAliasResource<spirv::LoadOp> {
  using ConvertAliasResource::ConvertAliasResource;

  LogicalResult
  matchAndRewrite(spirv::LoadOp loadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcElemType =
        loadOp.ptr().getType().cast<spirv::PointerType>().getPointeeType();
    auto dstElemType =
        adaptor.ptr().getType().cast<spirv::PointerType>().getPointeeType();
    if (!srcElemType.isIntOrFloat() || !dstElemType.isIntOrFloat())
      return rewriter.notifyMatchFailure(loadOp, "not scalar type");

    Location loc = loadOp.getLoc();
    auto newLoadOp = rewriter.create<spirv::LoadOp>(loc, adaptor.ptr());
    if (srcElemType == dstElemType) {
      rewriter.replaceOp(loadOp, newLoadOp->getResults());
      return success();
    }

    if (areSameBitwidthScalarType(srcElemType, dstElemType)) {
      auto castOp = rewriter.create<spirv::BitcastOp>(loc, srcElemType,
                                                      newLoadOp.value());
      rewriter.replaceOp(loadOp, castOp->getResults());

      return success();
    }

    // The source and destination have scalar types of different bitwidths.
    // For such cases, we need to load multiple smaller bitwidth values and
    // construct a larger bitwidth one.

    int srcNumBits = srcElemType.getIntOrFloatBitWidth();
    int dstNumBits = dstElemType.getIntOrFloatBitWidth();
    assert(srcNumBits > dstNumBits && srcNumBits % dstNumBits == 0);
    int ratio = srcNumBits / dstNumBits;
    if (ratio > 4)
      return rewriter.notifyMatchFailure(loadOp, "more than 4 components");

    SmallVector<Value> components;
    components.reserve(ratio);
    components.push_back(newLoadOp);

    auto acOp = adaptor.ptr().getDefiningOp<spirv::AccessChainOp>();
    if (!acOp)
      return rewriter.notifyMatchFailure(loadOp, "ptr not spv.AccessChain");

    auto i32Type = rewriter.getI32Type();
    Value oneValue = spirv::ConstantOp::getOne(i32Type, loc, rewriter);
    auto indices = llvm::to_vector<4>(acOp.indices());
    for (int i = 1; i < ratio; ++i) {
      // Load all subsequent components belonging to this element.
      indices.back() = rewriter.create<spirv::IAddOp>(loc, i32Type,
                                                      indices.back(), oneValue);
      auto componentAcOp =
          rewriter.create<spirv::AccessChainOp>(loc, acOp.base_ptr(), indices);
      components.push_back(rewriter.create<spirv::LoadOp>(loc, componentAcOp));
    }
    std::reverse(components.begin(), components.end()); // For little endian..

    // Create a vector of the components and then cast back to the larger
    // bitwidth element type.
    auto vectorType = VectorType::get({ratio}, dstElemType);
    Value vectorValue = rewriter.create<spirv::CompositeConstructOp>(
        loc, vectorType, components);
    rewriter.replaceOpWithNewOp<spirv::BitcastOp>(loadOp, srcElemType,
                                                  vectorValue);
    return success();
  }
};

struct ConvertStore : public ConvertAliasResource<spirv::StoreOp> {
  using ConvertAliasResource::ConvertAliasResource;

  LogicalResult
  matchAndRewrite(spirv::StoreOp storeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcElemType =
        storeOp.ptr().getType().cast<spirv::PointerType>().getPointeeType();
    auto dstElemType =
        adaptor.ptr().getType().cast<spirv::PointerType>().getPointeeType();
    if (!srcElemType.isIntOrFloat() || !dstElemType.isIntOrFloat())
      return rewriter.notifyMatchFailure(storeOp, "not scalar type");
    if (!areSameBitwidthScalarType(srcElemType, dstElemType))
      return rewriter.notifyMatchFailure(storeOp, "different bitwidth");

    Location loc = storeOp.getLoc();
    Value value = adaptor.value();
    if (srcElemType != dstElemType)
      value = rewriter.create<spirv::BitcastOp>(loc, dstElemType, value);
    rewriter.replaceOpWithNewOp<spirv::StoreOp>(storeOp, adaptor.ptr(), value,
                                                storeOp->getAttrs());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

namespace {
class UnifyAliasedResourcePass final
    : public SPIRVUnifyAliasedResourcePassBase<UnifyAliasedResourcePass> {
public:
  void runOnOperation() override;
};
} // namespace

void UnifyAliasedResourcePass::runOnOperation() {
  spirv::ModuleOp moduleOp = getOperation();
  MLIRContext *context = &getContext();

  // Analyze aliased resources first.
  ResourceAliasAnalysis &analysis = getAnalysis<ResourceAliasAnalysis>();

  ConversionTarget target(*context);
  target.addDynamicallyLegalOp<spirv::GlobalVariableOp, spirv::AddressOfOp,
                               spirv::AccessChainOp, spirv::LoadOp,
                               spirv::StoreOp>(
      [&analysis](Operation *op) { return !analysis.shouldUnify(op); });
  target.addLegalDialect<spirv::SPIRVDialect>();

  // Run patterns to rewrite usages of non-canonical resources.
  RewritePatternSet patterns(context);
  patterns.add<ConvertVariable, ConvertAddressOf, ConvertAccessChain,
               ConvertLoad, ConvertStore>(analysis, context);
  if (failed(applyPartialConversion(moduleOp, target, std::move(patterns))))
    return signalPassFailure();

  // Drop aliased attribute if we only have one single bound resource for a
  // descriptor. We need to re-collect the map here given in the above the
  // conversion is best effort; certain sets may not be converted.
  AliasedResourceMap resourceMap =
      collectAliasedResources(cast<spirv::ModuleOp>(moduleOp));
  for (const auto &dr : resourceMap) {
    const auto &resources = dr.second;
    if (resources.size() == 1)
      resources.front()->removeAttr("aliased");
  }
}

std::unique_ptr<mlir::OperationPass<spirv::ModuleOp>>
spirv::createUnifyAliasedResourcePass() {
  return std::make_unique<UnifyAliasedResourcePass>();
}
