//===--- Protocol.cpp - Language Server Protocol Implementation -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the serialization code for the LSP structs.
//
//===----------------------------------------------------------------------===//

#include "Protocol.h"

#include "URI.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "support/Logger.h"

namespace clang::clangd {
// Helper that doesn't treat `null` and absent fields as failures.
template <typename T>
bool mapOptOrNull(const llvm::json::Value& Params, llvm::StringLiteral Prop,
                  T& Out, llvm::json::Path P) {
  auto* O = Params.getAsObject();
  assert(O);
  auto* V = O->get(Prop);
  // Field is missing or null.
  if (!V || V->getAsNull())
    return true;
  return fromJSON(*V, Out, P.field(Prop));
}

URIForFile URIForFile::canonicalize(llvm::StringRef AbsPath,
                                    llvm::StringRef TUPath) {
  assert(llvm::sys::path::is_absolute(AbsPath) && "the path is relative");
  auto Resolved = URI::resolvePath(AbsPath, TUPath);
  if (!Resolved) {
    elog(
        "URIForFile: failed to resolve path {0} with TU path {1}: "
        "{2}.\nUsing unresolved path.",
        AbsPath, TUPath, Resolved.takeError());
    return URIForFile(std::string(AbsPath));
  }
  return URIForFile(std::move(*Resolved));
}

llvm::Expected<URIForFile> URIForFile::fromURI(const URI& U,
                                               llvm::StringRef HintPath) {
  auto Resolved = URI::resolve(U, HintPath);
  if (!Resolved)
    return Resolved.takeError();
  return URIForFile(std::move(*Resolved));
}

bool fromJSON(const llvm::json::Value& E, URIForFile& R, llvm::json::Path P) {
  if (auto S = E.getAsString()) {
    auto Parsed = URI::parse(*S);
    if (!Parsed) {
      consumeError(Parsed.takeError());
      P.report("failed to parse URI");
      return false;
    }
    if (Parsed->scheme() != "file" && Parsed->scheme() != "test") {
      P.report("clangd only supports 'file' URI scheme for workspace files");
      return false;
    }
    // "file" and "test" schemes do not require hint path.
    auto U = URIForFile::fromURI(*Parsed, /*HintPath=*/"");
    if (!U) {
      P.report("unresolvable URI");
      consumeError(U.takeError());
      return false;
    }
    R = std::move(*U);
    return true;
  }
  return false;
}

llvm::json::Value toJSON(const URIForFile& U) { return U.uri(); }

llvm::raw_ostream& operator<<(llvm::raw_ostream& OS, const URIForFile& U) {
  return OS << U.uri();
}

llvm::json::Value toJSON(const TextDocumentIdentifier& R) {
  return llvm::json::Object{{"uri", R.uri}};
}

bool fromJSON(const llvm::json::Value& Params, TextDocumentIdentifier& R,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O && O.map("uri", R.uri);
}

llvm::json::Value toJSON(const VersionedTextDocumentIdentifier& R) {
  auto Result = toJSON(static_cast<const TextDocumentIdentifier&>(R));
  Result.getAsObject()->try_emplace("version", R.version);
  return Result;
}

bool fromJSON(const llvm::json::Value& Params,
              VersionedTextDocumentIdentifier& R, llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return fromJSON(Params, static_cast<TextDocumentIdentifier&>(R), P) && O &&
         O.map("version", R.version);
}

bool fromJSON(const llvm::json::Value& Params, Position& R,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O && O.map("line", R.line) && O.map("character", R.character);
}

llvm::json::Value toJSON(const Position& P) {
  return llvm::json::Object{
      {"line", P.line},
      {"character", P.character},
  };
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& OS, const Position& P) {
  return OS << P.line << ':' << P.character;
}

bool fromJSON(const llvm::json::Value& Params, Range& R, llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O && O.map("start", R.start) && O.map("end", R.end);
}

llvm::json::Value toJSON(const Range& P) {
  return llvm::json::Object{
      {"start", P.start},
      {"end", P.end},
  };
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& OS, const Range& R) {
  return OS << R.start << '-' << R.end;
}

llvm::json::Value toJSON(const Location& P) {
  return llvm::json::Object{
      {"uri", P.uri},
      {"range", P.range},
  };
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& OS, const Location& L) {
  return OS << L.range << '@' << L.uri;
}

bool fromJSON(const llvm::json::Value& Params, TextDocumentItem& R,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O && O.map("uri", R.uri) && O.map("languageId", R.languageId) &&
         O.map("version", R.version) && O.map("text", R.text);
}

bool fromJSON(const llvm::json::Value& Params, DidOpenTextDocumentParams& R,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O && O.map("textDocument", R.textDocument);
}

bool fromJSON(const llvm::json::Value& Params, DidCloseTextDocumentParams& R,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O && O.map("textDocument", R.textDocument);
}

bool fromJSON(const llvm::json::Value& Params, DidSaveTextDocumentParams& R,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O && O.map("textDocument", R.textDocument);
}

bool fromJSON(const llvm::json::Value& Params,
              TextDocumentContentChangeEvent& R, llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O && O.map("range", R.range) && O.map("rangeLength", R.rangeLength) &&
         O.map("text", R.text);
}

bool fromJSON(const llvm::json::Value& Params, DidChangeTextDocumentParams& R,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O && O.map("textDocument", R.textDocument) &&
         O.map("contentChanges", R.contentChanges) &&
         O.map("wantDiagnostics", R.wantDiagnostics) &&
         mapOptOrNull(Params, "forceRebuild", R.forceRebuild, P);
}

bool fromJSON(const llvm::json::Value& Params, DocumentSymbolParams& R,
              llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O && O.map("textDocument", R.textDocument);
}

bool fromJSON(const llvm::json::Value& E, SymbolKind& Out,
              llvm::json::Path /* P */) {
  if (auto T = E.getAsInteger()) {
    if (*T < static_cast<int>(SymbolKind::File) ||
        *T > static_cast<int>(SymbolKind::TypeParameter))
      return false;
    Out = static_cast<SymbolKind>(*T);
    return true;
  }
  return false;
}

bool fromJSON(const llvm::json::Value& E, SymbolKindBitset& Out,
              llvm::json::Path P) {
  if (auto* A = E.getAsArray()) {
    for (size_t I = 0; I < A->size(); ++I) {
      SymbolKind KindOut;
      if (fromJSON((*A)[I], KindOut, P.index(I)))
        Out.set(size_t(KindOut));
    }
    return true;
  }
  return false;
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& O, const DocumentSymbol& S) {
  return O << S.name << " - " << toJSON(S);
}

llvm::json::Value toJSON(const DocumentSymbol& S) {
  llvm::json::Object Result{{"name", S.name},
                            {"kind", static_cast<int>(S.kind)},
                            {"range", S.range},
                            {"selectionRange", S.selectionRange}};

  if (!S.detail.empty())
    Result["detail"] = S.detail;
  if (!S.children.empty())
    Result["children"] = S.children;
  if (S.deprecated)
    Result["deprecated"] = true;
  // FIXME: workaround for older gcc/clang
  return std::move(Result);
}

}  // namespace clang::clangd
