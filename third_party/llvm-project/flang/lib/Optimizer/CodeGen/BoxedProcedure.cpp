//===-- BoxedProcedure.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/LowLevelIntrinsics.h"
#include "flang/Optimizer/CodeGen/CodeGen.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Support/FIRContext.h"
#include "flang/Optimizer/Support/FatalError.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "flang-procedure-pointer"

using namespace fir;

namespace {
/// Options to the procedure pointer pass.
struct BoxedProcedureOptions {
  // Lower the boxproc abstraction to function pointers and thunks where
  // required.
  bool useThunks = true;
};

/// This type converter rewrites all `!fir.boxproc<Func>` types to `Func` types.
class BoxprocTypeRewriter : public mlir::TypeConverter {
public:
  using mlir::TypeConverter::convertType;

  /// Does the type \p ty need to be converted?
  /// Any type that is a `!fir.boxproc` in whole or in part will need to be
  /// converted to a function type to lower the IR to function pointer form in
  /// the default implementation performed in this pass. Other implementations
  /// are possible, so those may convert `!fir.boxproc` to some other type or
  /// not at all depending on the implementation target's characteristics and
  /// preference.
  bool needsConversion(mlir::Type ty) {
    if (ty.isa<BoxProcType>())
      return true;
    if (auto funcTy = ty.dyn_cast<mlir::FunctionType>()) {
      for (auto t : funcTy.getInputs())
        if (needsConversion(t))
          return true;
      for (auto t : funcTy.getResults())
        if (needsConversion(t))
          return true;
      return false;
    }
    if (auto tupleTy = ty.dyn_cast<mlir::TupleType>()) {
      for (auto t : tupleTy.getTypes())
        if (needsConversion(t))
          return true;
      return false;
    }
    if (auto recTy = ty.dyn_cast<RecordType>()) {
      if (llvm::any_of(visitedTypes,
                       [&](mlir::Type rt) { return rt == recTy; }))
        return false;
      bool result = false;
      visitedTypes.push_back(recTy);
      for (auto t : recTy.getTypeList()) {
        if (needsConversion(t.second)) {
          result = true;
          break;
        }
      }
      visitedTypes.pop_back();
      return result;
    }
    if (auto boxTy = ty.dyn_cast<BoxType>())
      return needsConversion(boxTy.getEleTy());
    if (isa_ref_type(ty))
      return needsConversion(unwrapRefType(ty));
    if (auto t = ty.dyn_cast<SequenceType>())
      return needsConversion(unwrapSequenceType(ty));
    return false;
  }

  BoxprocTypeRewriter(mlir::Location location) : loc{location} {
    addConversion([](mlir::Type ty) { return ty; });
    addConversion(
        [&](BoxProcType boxproc) { return convertType(boxproc.getEleTy()); });
    addConversion([&](mlir::TupleType tupTy) {
      llvm::SmallVector<mlir::Type> memTys;
      for (auto ty : tupTy.getTypes())
        memTys.push_back(convertType(ty));
      return mlir::TupleType::get(tupTy.getContext(), memTys);
    });
    addConversion([&](mlir::FunctionType funcTy) {
      llvm::SmallVector<mlir::Type> inTys;
      llvm::SmallVector<mlir::Type> resTys;
      for (auto ty : funcTy.getInputs())
        inTys.push_back(convertType(ty));
      for (auto ty : funcTy.getResults())
        resTys.push_back(convertType(ty));
      return mlir::FunctionType::get(funcTy.getContext(), inTys, resTys);
    });
    addConversion([&](ReferenceType ty) {
      return ReferenceType::get(convertType(ty.getEleTy()));
    });
    addConversion([&](PointerType ty) {
      return PointerType::get(convertType(ty.getEleTy()));
    });
    addConversion(
        [&](HeapType ty) { return HeapType::get(convertType(ty.getEleTy())); });
    addConversion(
        [&](BoxType ty) { return BoxType::get(convertType(ty.getEleTy())); });
    addConversion([&](SequenceType ty) {
      // TODO: add ty.getLayoutMap() as needed.
      return SequenceType::get(ty.getShape(), convertType(ty.getEleTy()));
    });
    addConversion([&](RecordType ty) -> mlir::Type {
      if (!needsConversion(ty))
        return ty;
      // FIR record types can have recursive references, so conversion is a bit
      // more complex than the other types. This conversion is not needed
      // presently, so just emit a TODO message. Need to consider the uniqued
      // name of the record, etc. Also, fir::RecordType::get returns the
      // existing type being translated. So finalize() will not change it, and
      // the translation would not do anything. So the type needs to be mutated,
      // and this might require special care to comply with MLIR infrastructure.

      // TODO: this will be needed to support derived type containing procedure
      // pointer components.
      fir::emitFatalError(
          loc, "not yet implemented: record type with a boxproc type");
      return RecordType::get(ty.getContext(), "*fixme*");
    });
    addArgumentMaterialization(materializeProcedure);
    addSourceMaterialization(materializeProcedure);
    addTargetMaterialization(materializeProcedure);
  }

  static mlir::Value materializeProcedure(mlir::OpBuilder &builder,
                                          BoxProcType type,
                                          mlir::ValueRange inputs,
                                          mlir::Location loc) {
    assert(inputs.size() == 1);
    return builder.create<ConvertOp>(loc, unwrapRefType(type.getEleTy()),
                                     inputs[0]);
  }

  void setLocation(mlir::Location location) { loc = location; }

private:
  llvm::SmallVector<mlir::Type> visitedTypes;
  mlir::Location loc;
};

/// A `boxproc` is an abstraction for a Fortran procedure reference. Typically,
/// Fortran procedures can be referenced directly through a function pointer.
/// However, Fortran has one-level dynamic scoping between a host procedure and
/// its internal procedures. This allows internal procedures to directly access
/// and modify the state of the host procedure's variables.
///
/// There are any number of possible implementations possible.
///
/// The implementation used here is to convert `boxproc` values to function
/// pointers everywhere. If a `boxproc` value includes a frame pointer to the
/// host procedure's data, then a thunk will be created at runtime to capture
/// the frame pointer during execution. In LLVM IR, the frame pointer is
/// designated with the `nest` attribute. The thunk's address will then be used
/// as the call target instead of the original function's address directly.
class BoxedProcedurePass : public BoxedProcedurePassBase<BoxedProcedurePass> {
public:
  BoxedProcedurePass() { options = {true}; }
  BoxedProcedurePass(bool useThunks) { options = {useThunks}; }

  inline mlir::ModuleOp getModule() { return getOperation(); }

  void runOnOperation() override final {
    if (options.useThunks) {
      auto *context = &getContext();
      mlir::IRRewriter rewriter(context);
      BoxprocTypeRewriter typeConverter(mlir::UnknownLoc::get(context));
      mlir::Dialect *firDialect = context->getLoadedDialect("fir");
      getModule().walk([&](mlir::Operation *op) {
        typeConverter.setLocation(op->getLoc());
        if (auto addr = mlir::dyn_cast<BoxAddrOp>(op)) {
          auto ty = addr.getVal().getType();
          if (typeConverter.needsConversion(ty) ||
              ty.isa<mlir::FunctionType>()) {
            // Rewrite all `fir.box_addr` ops on values of type `!fir.boxproc`
            // or function type to be `fir.convert` ops.
            rewriter.setInsertionPoint(addr);
            rewriter.replaceOpWithNewOp<ConvertOp>(
                addr, typeConverter.convertType(addr.getType()), addr.getVal());
          }
        } else if (auto func = mlir::dyn_cast<mlir::func::FuncOp>(op)) {
          mlir::FunctionType ty = func.getFunctionType();
          if (typeConverter.needsConversion(ty)) {
            rewriter.startRootUpdate(func);
            auto toTy =
                typeConverter.convertType(ty).cast<mlir::FunctionType>();
            if (!func.empty())
              for (auto e : llvm::enumerate(toTy.getInputs())) {
                unsigned i = e.index();
                auto &block = func.front();
                block.insertArgument(i, e.value(), func.getLoc());
                block.getArgument(i + 1).replaceAllUsesWith(
                    block.getArgument(i));
                block.eraseArgument(i + 1);
              }
            func.setType(toTy);
            rewriter.finalizeRootUpdate(func);
          }
        } else if (auto embox = mlir::dyn_cast<EmboxProcOp>(op)) {
          // Rewrite all `fir.emboxproc` ops to either `fir.convert` or a thunk
          // as required.
          mlir::Type toTy = embox.getType().cast<BoxProcType>().getEleTy();
          rewriter.setInsertionPoint(embox);
          if (embox.getHost()) {
            // Create the thunk.
            auto module = embox->getParentOfType<mlir::ModuleOp>();
            FirOpBuilder builder(rewriter, getKindMapping(module));
            auto loc = embox.getLoc();
            mlir::Type i8Ty = builder.getI8Type();
            mlir::Type i8Ptr = builder.getRefType(i8Ty);
            mlir::Type buffTy = SequenceType::get({32}, i8Ty);
            auto buffer = builder.create<AllocaOp>(loc, buffTy);
            mlir::Value closure =
                builder.createConvert(loc, i8Ptr, embox.getHost());
            mlir::Value tramp = builder.createConvert(loc, i8Ptr, buffer);
            mlir::Value func =
                builder.createConvert(loc, i8Ptr, embox.getFunc());
            builder.create<fir::CallOp>(
                loc, factory::getLlvmInitTrampoline(builder),
                llvm::ArrayRef<mlir::Value>{tramp, func, closure});
            auto adjustCall = builder.create<fir::CallOp>(
                loc, factory::getLlvmAdjustTrampoline(builder),
                llvm::ArrayRef<mlir::Value>{tramp});
            rewriter.replaceOpWithNewOp<ConvertOp>(embox, toTy,
                                                   adjustCall.getResult(0));
          } else {
            // Just forward the function as a pointer.
            rewriter.replaceOpWithNewOp<ConvertOp>(embox, toTy,
                                                   embox.getFunc());
          }
        } else if (auto mem = mlir::dyn_cast<AllocaOp>(op)) {
          auto ty = mem.getType();
          if (typeConverter.needsConversion(ty)) {
            rewriter.setInsertionPoint(mem);
            auto toTy = typeConverter.convertType(unwrapRefType(ty));
            bool isPinned = mem.getPinned();
            llvm::StringRef uniqName;
            if (mem.getUniqName().hasValue())
              uniqName = mem.getUniqName().getValue();
            llvm::StringRef bindcName;
            if (mem.getBindcName().hasValue())
              bindcName = mem.getBindcName().getValue();
            rewriter.replaceOpWithNewOp<AllocaOp>(
                mem, toTy, uniqName, bindcName, isPinned, mem.getTypeparams(),
                mem.getShape());
          }
        } else if (auto mem = mlir::dyn_cast<AllocMemOp>(op)) {
          auto ty = mem.getType();
          if (typeConverter.needsConversion(ty)) {
            rewriter.setInsertionPoint(mem);
            auto toTy = typeConverter.convertType(unwrapRefType(ty));
            llvm::StringRef uniqName;
            if (mem.getUniqName().hasValue())
              uniqName = mem.getUniqName().getValue();
            llvm::StringRef bindcName;
            if (mem.getBindcName().hasValue())
              bindcName = mem.getBindcName().getValue();
            rewriter.replaceOpWithNewOp<AllocMemOp>(
                mem, toTy, uniqName, bindcName, mem.getTypeparams(),
                mem.getShape());
          }
        } else if (auto coor = mlir::dyn_cast<CoordinateOp>(op)) {
          auto ty = coor.getType();
          mlir::Type baseTy = coor.getBaseType();
          if (typeConverter.needsConversion(ty) ||
              typeConverter.needsConversion(baseTy)) {
            rewriter.setInsertionPoint(coor);
            auto toTy = typeConverter.convertType(ty);
            auto toBaseTy = typeConverter.convertType(baseTy);
            rewriter.replaceOpWithNewOp<CoordinateOp>(coor, toTy, coor.getRef(),
                                                      coor.getCoor(), toBaseTy);
          }
        } else if (auto index = mlir::dyn_cast<FieldIndexOp>(op)) {
          auto ty = index.getType();
          mlir::Type onTy = index.getOnType();
          if (typeConverter.needsConversion(ty) ||
              typeConverter.needsConversion(onTy)) {
            rewriter.setInsertionPoint(index);
            auto toTy = typeConverter.convertType(ty);
            auto toOnTy = typeConverter.convertType(onTy);
            rewriter.replaceOpWithNewOp<FieldIndexOp>(
                index, toTy, index.getFieldId(), toOnTy, index.getTypeparams());
          }
        } else if (auto index = mlir::dyn_cast<LenParamIndexOp>(op)) {
          auto ty = index.getType();
          mlir::Type onTy = index.getOnType();
          if (typeConverter.needsConversion(ty) ||
              typeConverter.needsConversion(onTy)) {
            rewriter.setInsertionPoint(index);
            auto toTy = typeConverter.convertType(ty);
            auto toOnTy = typeConverter.convertType(onTy);
            rewriter.replaceOpWithNewOp<LenParamIndexOp>(
                mem, toTy, index.getFieldId(), toOnTy);
          }
        } else if (op->getDialect() == firDialect) {
          rewriter.startRootUpdate(op);
          for (auto i : llvm::enumerate(op->getResultTypes()))
            if (typeConverter.needsConversion(i.value())) {
              auto toTy = typeConverter.convertType(i.value());
              op->getResult(i.index()).setType(toTy);
            }
          rewriter.finalizeRootUpdate(op);
        }
      });
    }
    // TODO: any alternative implementation. Note: currently, the default code
    // gen will not be able to handle boxproc and will give an error.
  }

private:
  BoxedProcedureOptions options;
};
} // namespace

std::unique_ptr<mlir::Pass> fir::createBoxedProcedurePass() {
  return std::make_unique<BoxedProcedurePass>();
}

std::unique_ptr<mlir::Pass> fir::createBoxedProcedurePass(bool useThunks) {
  return std::make_unique<BoxedProcedurePass>(useThunks);
}
