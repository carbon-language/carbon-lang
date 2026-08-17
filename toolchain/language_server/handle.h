// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LANGUAGE_SERVER_HANDLE_H_
#define CARBON_TOOLCHAIN_LANGUAGE_SERVER_HANDLE_H_

#include "clang-tools-extra/clangd/Protocol.h"
#include "toolchain/language_server/context.h"

namespace Carbon::LanguageServer {

// Locates the entity named at a position.
auto HandleDefinition(
    Context& context, const clang::clangd::TextDocumentPositionParams& params,
    llvm::function_ref<
        auto(llvm::Expected<std::vector<clang::clangd::Location>>)->void>
        on_done) -> void;

// Stores the content of newly-opened documents.
auto HandleDidChangeTextDocument(
    Context& context, const clang::clangd::DidChangeTextDocumentParams& params)
    -> void;

// Closes a document.
auto HandleDidCloseTextDocument(
    Context& context, const clang::clangd::DidCloseTextDocumentParams& params)
    -> void;

// Acknowledges that a document was saved.
auto HandleDidSaveTextDocument(
    Context& context, const clang::clangd::DidSaveTextDocumentParams& params)
    -> void;

// Updates the content of already-open documents.
auto HandleDidOpenTextDocument(
    Context& context, const clang::clangd::DidOpenTextDocumentParams& params)
    -> void;

// Provides information about document symbols.
auto HandleDocumentSymbol(
    Context& context, const clang::clangd::DocumentSymbolParams& params,
    llvm::function_ref<
        auto(llvm::Expected<std::vector<clang::clangd::DocumentSymbol>>)->void>
        on_done) -> void;

// Provides the type of the entity at a position.
auto HandleHover(
    Context& context, const clang::clangd::TextDocumentPositionParams& params,
    llvm::function_ref<auto(llvm::Expected<clang::clangd::Hover>)->void>
        on_done) -> void;

// Tells the client what features are supported, and negotiates the position
// encoding.
auto HandleInitialize(
    Context& context, const clang::clangd::InitializeParams& params,
    llvm::function_ref<auto(llvm::Expected<llvm::json::Object>)->void> on_done)
    -> void;

// Acknowledges that the client finished initializing.
auto HandleInitialized(Context& context, const clang::clangd::NoParams& params)
    -> void;

// Finds references to the entity named at a position, within this file only.
auto HandleReferences(
    Context& context, const clang::clangd::ReferenceParams& params,
    llvm::function_ref<
        auto(llvm::Expected<std::vector<clang::clangd::Location>>)->void>
        on_done) -> void;

// Prepares LSP for shutdown.
auto HandleShutdown(
    Context& /*context*/,
    const clang::clangd::NoParams& /*client_capabilities*/,
    llvm::function_ref<auto(llvm::Expected<std::nullptr_t>)->void> on_done)
    -> void;

// Locates the type of the entity named at a position.
auto HandleTypeDefinition(
    Context& context, const clang::clangd::TextDocumentPositionParams& params,
    llvm::function_ref<
        auto(llvm::Expected<std::vector<clang::clangd::Location>>)->void>
        on_done) -> void;

}  // namespace Carbon::LanguageServer

#endif  // CARBON_TOOLCHAIN_LANGUAGE_SERVER_HANDLE_H_
