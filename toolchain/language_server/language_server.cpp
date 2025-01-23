// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/language_server/language_server.h"

#include "clang-tools-extra/clangd/LSPBinder.h"
#include "clang-tools-extra/clangd/Transport.h"
#include "common/raw_string_ostream.h"
#include "toolchain/language_server/context.h"
#include "toolchain/language_server/incoming_messages.h"
#include "toolchain/language_server/outgoing_messages.h"

namespace Carbon::LanguageServer {

auto Run(FILE* input_stream, llvm::raw_ostream& output_stream,
         llvm::raw_ostream& /*error_stream*/) -> ErrorOr<Success> {
  // Set up the connection.
  std::unique_ptr<clang::clangd::Transport> transport(
      clang::clangd::newJSONTransport(input_stream, output_stream,
                                      /*InMirror=*/nullptr,
                                      /*Pretty=*/true));
  Context context;
  // TODO: Use error_stream in IncomingMessages to report dropped errors.
  IncomingMessages incoming(transport.get(), &context);
  OutgoingMessages outgoing(transport.get());

  // Run the server loop.
  llvm::Error err = transport->loop(incoming);
  if (err) {
    RawStringOstream out;
    out << err;
    return Error(out.TakeStr());
  } else {
    return Success();
  }
}

}  // namespace Carbon::LanguageServer
