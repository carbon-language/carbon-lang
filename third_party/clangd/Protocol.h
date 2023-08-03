//===--- Protocol.h - Language Server Protocol Implementation ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains structs based on the LSP specification at
// https://github.com/Microsoft/language-server-protocol/blob/main/protocol.md
//
// This is not meant to be a complete implementation, new interfaces are added
// when they're needed.
//
// Each struct has a toJSON and fromJSON function, that converts between
// the struct and a JSON representation. (See JSON.h)
//
// Some structs also have operator<< serialization. This is for debugging and
// tests, and is not generally machine-readable.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_PROTOCOL_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_PROTOCOL_H

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/JSON.h>

#include <bitset>

#include "URI.h"

namespace clang::clangd {
// URI in "file" scheme for a file.
struct URIForFile {
  URIForFile() = default;

  /// Canonicalizes \p AbsPath via URI.
  ///
  /// File paths in URIForFile can come from index or local AST. Path from
  /// index goes through URI transformation, and the final path is resolved by
  /// URI scheme and could potentially be different from the original path.
  /// Hence, we do the same transformation for all paths.
  ///
  /// Files can be referred to by several paths (e.g. in the presence of links).
  /// Which one we prefer may depend on where we're coming from. \p TUPath is a
  /// hint, and should usually be the main entrypoint file we're processing.
  static URIForFile canonicalize(llvm::StringRef AbsPath,
                                 llvm::StringRef TUPath);

  static llvm::Expected<URIForFile> fromURI(const URI& U,
                                            llvm::StringRef HintPath);

  /// Retrieves absolute path to the file.
  llvm::StringRef file() const { return File; }

  explicit operator bool() const { return !File.empty(); }
  std::string uri() const { return URI::createFile(File).toString(); }

  friend bool operator==(const URIForFile& LHS, const URIForFile& RHS) {
    return LHS.File == RHS.File;
  }

  friend bool operator!=(const URIForFile& LHS, const URIForFile& RHS) {
    return !(LHS == RHS);
  }

  friend bool operator<(const URIForFile& LHS, const URIForFile& RHS) {
    return LHS.File < RHS.File;
  }

 private:
  explicit URIForFile(std::string&& File) : File(std::move(File)) {}

  std::string File;
};

/// Serialize/deserialize \p URIForFile to/from a string URI.
llvm::json::Value toJSON(const URIForFile& U);
bool fromJSON(const llvm::json::Value&, URIForFile&, llvm::json::Path);

struct TextDocumentIdentifier {
  /// The text document's URI.
  URIForFile uri;
};

llvm::json::Value toJSON(const TextDocumentIdentifier&);
bool fromJSON(const llvm::json::Value&, TextDocumentIdentifier&,
              llvm::json::Path);

struct VersionedTextDocumentIdentifier : public TextDocumentIdentifier {
  /// The version number of this document. If a versioned text document
  /// identifier is sent from the server to the client and the file is not open
  /// in the editor (the server has not received an open notification before)
  /// the server can send `null` to indicate that the version is known and the
  /// content on disk is the master (as speced with document content ownership).
  ///
  /// The version number of a document will increase after each change,
  /// including undo/redo. The number doesn't need to be consecutive.
  ///
  /// clangd extension: versions are optional, and synthesized if missing.
  std::optional<std::int64_t> version;
};
llvm::json::Value toJSON(const VersionedTextDocumentIdentifier&);
bool fromJSON(const llvm::json::Value&, VersionedTextDocumentIdentifier&,
              llvm::json::Path);

struct Position {
  /// Line position in a document (zero-based).
  int line = 0;

  /// Character offset on a line in a document (zero-based).
  /// WARNING: this is in UTF-16 codepoints, not bytes or characters!
  /// Use the functions in SourceCode.h to construct/interpret Positions.
  int character = 0;

  friend bool operator==(const Position& LHS, const Position& RHS) {
    return std::tie(LHS.line, LHS.character) ==
           std::tie(RHS.line, RHS.character);
  }
  friend bool operator!=(const Position& LHS, const Position& RHS) {
    return !(LHS == RHS);
  }
  friend bool operator<(const Position& LHS, const Position& RHS) {
    return std::tie(LHS.line, LHS.character) <
           std::tie(RHS.line, RHS.character);
  }
  friend bool operator<=(const Position& LHS, const Position& RHS) {
    return std::tie(LHS.line, LHS.character) <=
           std::tie(RHS.line, RHS.character);
  }
};
bool fromJSON(const llvm::json::Value&, Position&, llvm::json::Path);
llvm::json::Value toJSON(const Position&);
llvm::raw_ostream& operator<<(llvm::raw_ostream&, const Position&);

struct Range {
  /// The range's start position.
  Position start;

  /// The range's end position.
  Position end;

  friend bool operator==(const Range& LHS, const Range& RHS) {
    return std::tie(LHS.start, LHS.end) == std::tie(RHS.start, RHS.end);
  }
  friend bool operator!=(const Range& LHS, const Range& RHS) {
    return !(LHS == RHS);
  }
  friend bool operator<(const Range& LHS, const Range& RHS) {
    return std::tie(LHS.start, LHS.end) < std::tie(RHS.start, RHS.end);
  }

  bool contains(Position Pos) const { return start <= Pos && Pos < end; }
  bool contains(Range Rng) const {
    return start <= Rng.start && Rng.end <= end;
  }
};
bool fromJSON(const llvm::json::Value&, Range&, llvm::json::Path);
llvm::json::Value toJSON(const Range&);
llvm::raw_ostream& operator<<(llvm::raw_ostream&, const Range&);

struct Location {
  /// The text document's URI.
  URIForFile uri;
  Range range;

  friend bool operator==(const Location& LHS, const Location& RHS) {
    return LHS.uri == RHS.uri && LHS.range == RHS.range;
  }

  friend bool operator!=(const Location& LHS, const Location& RHS) {
    return !(LHS == RHS);
  }

  friend bool operator<(const Location& LHS, const Location& RHS) {
    return std::tie(LHS.uri, LHS.range) < std::tie(RHS.uri, RHS.range);
  }
};
llvm::json::Value toJSON(const Location&);
llvm::raw_ostream& operator<<(llvm::raw_ostream&, const Location&);

struct TextDocumentItem {
  /// The text document's URI.
  URIForFile uri;

  /// The text document's language identifier.
  std::string languageId;

  /// The version number of this document (it will strictly increase after each
  /// change, including undo/redo.
  ///
  /// clangd extension: versions are optional, and synthesized if missing.
  std::optional<int64_t> version;

  /// The content of the opened text document.
  std::string text;
};
bool fromJSON(const llvm::json::Value&, TextDocumentItem&, llvm::json::Path);

struct DidOpenTextDocumentParams {
  /// The document that was opened.
  TextDocumentItem textDocument;
};
bool fromJSON(const llvm::json::Value&, DidOpenTextDocumentParams&,
              llvm::json::Path);

struct DidCloseTextDocumentParams {
  /// The document that was closed.
  TextDocumentIdentifier textDocument;
};
bool fromJSON(const llvm::json::Value&, DidCloseTextDocumentParams&,
              llvm::json::Path);

struct DidSaveTextDocumentParams {
  /// The document that was saved.
  TextDocumentIdentifier textDocument;
};
bool fromJSON(const llvm::json::Value&, DidSaveTextDocumentParams&,
              llvm::json::Path);

struct TextDocumentContentChangeEvent {
  /// The range of the document that changed.
  std::optional<Range> range;

  /// The length of the range that got replaced.
  std::optional<int> rangeLength;

  /// The new text of the range/document.
  std::string text;
};
bool fromJSON(const llvm::json::Value&, TextDocumentContentChangeEvent&,
              llvm::json::Path);

struct DidChangeTextDocumentParams {
  /// The document that did change. The version number points
  /// to the version after all provided content changes have
  /// been applied.
  VersionedTextDocumentIdentifier textDocument;

  /// The actual content changes.
  std::vector<TextDocumentContentChangeEvent> contentChanges;

  /// Forces diagnostics to be generated, or to not be generated, for this
  /// version of the file. If not set, diagnostics are eventually consistent:
  /// either they will be provided for this version or some subsequent one.
  /// This is a clangd extension.
  std::optional<bool> wantDiagnostics;

  /// Force a complete rebuild of the file, ignoring all cached state. Slow!
  /// This is useful to defeat clangd's assumption that missing headers will
  /// stay missing.
  /// This is a clangd extension.
  bool forceRebuild = false;
};
bool fromJSON(const llvm::json::Value&, DidChangeTextDocumentParams&,
              llvm::json::Path);

struct DocumentSymbolParams {
  // The text document to find symbols in.
  TextDocumentIdentifier textDocument;
};
bool fromJSON(const llvm::json::Value&, DocumentSymbolParams&,
              llvm::json::Path);

/// A symbol kind.
enum class SymbolKind {
  File = 1,
  Module = 2,
  Namespace = 3,
  Package = 4,
  Class = 5,
  Method = 6,
  Property = 7,
  Field = 8,
  Constructor = 9,
  Enum = 10,
  Interface = 11,
  Function = 12,
  Variable = 13,
  Constant = 14,
  String = 15,
  Number = 16,
  Boolean = 17,
  Array = 18,
  Object = 19,
  Key = 20,
  Null = 21,
  EnumMember = 22,
  Struct = 23,
  Event = 24,
  Operator = 25,
  TypeParameter = 26
};
bool fromJSON(const llvm::json::Value&, SymbolKind&, llvm::json::Path);
constexpr auto SymbolKindMin = static_cast<size_t>(SymbolKind::File);
constexpr auto SymbolKindMax = static_cast<size_t>(SymbolKind::TypeParameter);
using SymbolKindBitset = std::bitset<SymbolKindMax + 1>;
bool fromJSON(const llvm::json::Value&, SymbolKindBitset&, llvm::json::Path);

/// Represents programming constructs like variables, classes, interfaces etc.
/// that appear in a document. Document symbols can be hierarchical and they
/// have two ranges: one that encloses its definition and one that points to its
/// most interesting range, e.g. the range of an identifier.
struct DocumentSymbol {
  /// The name of this symbol.
  std::string name;

  /// More detail for this symbol, e.g the signature of a function.
  std::string detail;

  /// The kind of this symbol.
  SymbolKind kind;

  /// Indicates if this symbol is deprecated.
  bool deprecated = false;

  /// The range enclosing this symbol not including leading/trailing whitespace
  /// but everything else like comments. This information is typically used to
  /// determine if the clients cursor is inside the symbol to reveal in the
  /// symbol in the UI.
  Range range;

  /// The range that should be selected and revealed when this symbol is being
  /// picked, e.g the name of a function. Must be contained by the `range`.
  Range selectionRange;

  /// Children of this symbol, e.g. properties of a class.
  std::vector<DocumentSymbol> children;
};
llvm::raw_ostream& operator<<(llvm::raw_ostream& O, const DocumentSymbol& S);
llvm::json::Value toJSON(const DocumentSymbol& S);
}  // namespace clang::clangd

namespace llvm::json {
template <typename T>
llvm::json::Value toJSON(const T& val) {
  return clang::clangd::toJSON(val);
}
template <typename T>
bool fromJSON(const llvm::json::Value& json, T& val, llvm::json::Path path) {
  return clang::clangd::fromJSON(json, val, path);
}
}  // namespace llvm::json

#endif
