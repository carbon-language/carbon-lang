//===-- Command.cpp -- generate command line runtime API calls ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/Runtime/Command.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Runtime/command.h"

using namespace Fortran::runtime;

// Certain runtime intrinsics should only be run when select parameters of the
// intrisic are supplied. In certain cases one of these parameters may not be
// given, however the intrinsic needs to be run due to another required
// parameter being supplied. In this case the missing parameter is assigned to
// have an "absent" value. This typically happens in IntrinsicCall.cpp. For this
// reason the extra indirection with `isAbsent` is needed for testing whether a
// given parameter is actually present (so that parameters with "value" absent
// are not considered as present).
inline bool isAbsent(mlir::Value val) {
  return mlir::isa_and_nonnull<fir::AbsentOp>(val.getDefiningOp());
}

mlir::Value fir::runtime::genCommandArgumentCount(fir::FirOpBuilder &builder,
                                                  mlir::Location loc) {
  auto argumentCountFunc =
      fir::runtime::getRuntimeFunc<mkRTKey(ArgumentCount)>(loc, builder);
  return builder.create<fir::CallOp>(loc, argumentCountFunc).getResult(0);
}

mlir::Value fir::runtime::genArgumentValue(fir::FirOpBuilder &builder,
                                           mlir::Location loc,
                                           mlir::Value number,
                                           mlir::Value value,
                                           mlir::Value errmsg) {
  auto argumentValueFunc =
      fir::runtime::getRuntimeFunc<mkRTKey(ArgumentValue)>(loc, builder);
  llvm::SmallVector<mlir::Value> args = fir::runtime::createArguments(
      builder, loc, argumentValueFunc.getFunctionType(), number, value, errmsg);
  return builder.create<fir::CallOp>(loc, argumentValueFunc, args).getResult(0);
}

mlir::Value fir::runtime::genArgumentLength(fir::FirOpBuilder &builder,
                                            mlir::Location loc,
                                            mlir::Value number) {
  auto argumentLengthFunc =
      fir::runtime::getRuntimeFunc<mkRTKey(ArgumentLength)>(loc, builder);
  llvm::SmallVector<mlir::Value> args = fir::runtime::createArguments(
      builder, loc, argumentLengthFunc.getFunctionType(), number);
  return builder.create<fir::CallOp>(loc, argumentLengthFunc, args)
      .getResult(0);
}

mlir::Value fir::runtime::genEnvVariableValue(
    fir::FirOpBuilder &builder, mlir::Location loc, mlir::Value name,
    mlir::Value value, mlir::Value trimName, mlir::Value errmsg) {
  auto valueFunc =
      fir::runtime::getRuntimeFunc<mkRTKey(EnvVariableValue)>(loc, builder);
  mlir::FunctionType valueFuncTy = valueFunc.getFunctionType();
  mlir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
  mlir::Value sourceLine =
      fir::factory::locationToLineNo(builder, loc, valueFuncTy.getInput(5));
  llvm::SmallVector<mlir::Value> args =
      fir::runtime::createArguments(builder, loc, valueFuncTy, name, value,
                                    trimName, errmsg, sourceFile, sourceLine);
  return builder.create<fir::CallOp>(loc, valueFunc, args).getResult(0);
}

mlir::Value fir::runtime::genEnvVariableLength(fir::FirOpBuilder &builder,
                                               mlir::Location loc,
                                               mlir::Value name,
                                               mlir::Value trimName) {
  auto lengthFunc =
      fir::runtime::getRuntimeFunc<mkRTKey(EnvVariableLength)>(loc, builder);
  mlir::FunctionType lengthFuncTy = lengthFunc.getFunctionType();
  mlir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
  mlir::Value sourceLine =
      fir::factory::locationToLineNo(builder, loc, lengthFuncTy.getInput(3));
  llvm::SmallVector<mlir::Value> args = fir::runtime::createArguments(
      builder, loc, lengthFuncTy, name, trimName, sourceFile, sourceLine);
  return builder.create<fir::CallOp>(loc, lengthFunc, args).getResult(0);
}
