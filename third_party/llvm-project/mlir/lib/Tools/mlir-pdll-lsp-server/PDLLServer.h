//===- PDLLServer.h - PDL General Language Server ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIB_MLIR_TOOLS_MLIRPDLLSPSERVER_SERVER_H_
#define LIB_MLIR_TOOLS_MLIRPDLLSPSERVER_SERVER_H_

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <string>

namespace mlir {
namespace lsp {
struct Diagnostic;
class CompilationDatabase;
struct PDLLViewOutputResult;
enum class PDLLViewOutputKind;
struct CompletionList;
struct DocumentLink;
struct DocumentSymbol;
struct Hover;
struct InlayHint;
struct Location;
struct Position;
struct Range;
struct SignatureHelp;
struct TextDocumentContentChangeEvent;
class URIForFile;

/// This class implements all of the PDLL related functionality necessary for a
/// language server. This class allows for keeping the PDLL specific logic
/// separate from the logic that involves LSP server/client communication.
class PDLLServer {
public:
  struct Options {
    Options(const std::vector<std::string> &compilationDatabases,
            const std::vector<std::string> &extraDirs)
        : compilationDatabases(compilationDatabases), extraDirs(extraDirs) {}

    /// The filenames for databases containing compilation commands for PDLL
    /// files passed to the server.
    const std::vector<std::string> &compilationDatabases;

    /// Additional list of include directories to search.
    const std::vector<std::string> &extraDirs;
  };

  PDLLServer(const Options &options);
  ~PDLLServer();

  /// Add the document, with the provided `version`, at the given URI. Any
  /// diagnostics emitted for this document should be added to `diagnostics`.
  void addDocument(const URIForFile &uri, StringRef contents, int64_t version,
                   std::vector<Diagnostic> &diagnostics);

  /// Update the document, with the provided `version`, at the given URI. Any
  /// diagnostics emitted for this document should be added to `diagnostics`.
  void updateDocument(const URIForFile &uri,
                      ArrayRef<TextDocumentContentChangeEvent> changes,
                      int64_t version, std::vector<Diagnostic> &diagnostics);

  /// Remove the document with the given uri. Returns the version of the removed
  /// document, or None if the uri did not have a corresponding document within
  /// the server.
  Optional<int64_t> removeDocument(const URIForFile &uri);

  /// Return the locations of the object pointed at by the given position.
  void getLocationsOf(const URIForFile &uri, const Position &defPos,
                      std::vector<Location> &locations);

  /// Find all references of the object pointed at by the given position.
  void findReferencesOf(const URIForFile &uri, const Position &pos,
                        std::vector<Location> &references);

  /// Return the document links referenced by the given file.
  void getDocumentLinks(const URIForFile &uri,
                        std::vector<DocumentLink> &documentLinks);

  /// Find a hover description for the given hover position, or None if one
  /// couldn't be found.
  Optional<Hover> findHover(const URIForFile &uri, const Position &hoverPos);

  /// Find all of the document symbols within the given file.
  void findDocumentSymbols(const URIForFile &uri,
                           std::vector<DocumentSymbol> &symbols);

  /// Get the code completion list for the position within the given file.
  CompletionList getCodeCompletion(const URIForFile &uri,
                                   const Position &completePos);

  /// Get the signature help for the position within the given file.
  SignatureHelp getSignatureHelp(const URIForFile &uri,
                                 const Position &helpPos);

  /// Get the inlay hints for the range within the given file.
  void getInlayHints(const URIForFile &uri, const Range &range,
                     std::vector<InlayHint> &inlayHints);

  /// Get the output of the given PDLL file, or None if there is no valid
  /// output.
  Optional<PDLLViewOutputResult> getPDLLViewOutput(const URIForFile &uri,
                                                   PDLLViewOutputKind kind);

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace lsp
} // namespace mlir

#endif // LIB_MLIR_TOOLS_MLIRPDLLSPSERVER_SERVER_H_
