//===- LSPServer.h - PDLL LSP Server ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIB_MLIR_TOOLS_MLIRPDLLLSPSERVER_LSPSERVER_H
#define LIB_MLIR_TOOLS_MLIRPDLLLSPSERVER_LSPSERVER_H

#include <memory>

namespace mlir {
struct LogicalResult;

namespace lsp {
class JSONTransport;
class PDLLServer;

/// Run the main loop of the LSP server using the given PDLL server and
/// transport.
LogicalResult runPdllLSPServer(PDLLServer &server, JSONTransport &transport);

} // namespace lsp
} // namespace mlir

#endif // LIB_MLIR_TOOLS_MLIRPDLLLSPSERVER_LSPSERVER_H
