//===- mlir-pdll-lsp-server.cpp - PDLL Language Server main ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-pdll-lsp-server/MlirPdllLspServerMain.h"

using namespace mlir;

int main(int argc, char **argv) {
  return failed(MlirPdllLspServerMain(argc, argv));
}
