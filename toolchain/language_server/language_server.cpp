// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/language_server/language_server.h"

#include "clang-tools-extra/clangd/LSPBinder.h"
#include "clang-tools-extra/clangd/Transport.h"
#include "clang-tools-extra/clangd/support/Logger.h"
#include "common/raw_string_ostream.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/language_server/context.h"
#include "toolchain/language_server/incoming_messages.h"
#include "toolchain/language_server/outgoing_messages.h"

namespace Carbon::LanguageServer {

// An adapter for clangd's logging that sends most messages to `error_stream`.
// Verbose logging is only printed if `vlog_stream` is provided, and will be
// sent there.
class Logger : public clang::clangd::Logger {
 public:
  explicit Logger(llvm::raw_ostream* error_stream,
                  llvm::raw_ostream* vlog_stream)
      : error_logger_(*error_stream, clang::clangd::Logger::Info),
        vlog_logger_(vlog_stream
                         ? std::make_unique<clang::clangd::StreamLogger>(
                               *vlog_stream, clang::clangd::Logger::Verbose)
                         : nullptr) {}

  auto log(Level level, const char* format,
           const llvm::formatv_object_base& message) -> void override {
    if (level != clang::clangd::Logger::Verbose) {
      error_logger_.log(level, format, message);
    } else if (vlog_logger_) {
      vlog_logger_->log(level, format, message);
    }
  }

 private:
  clang::clangd::StreamLogger error_logger_;
  std::unique_ptr<clang::clangd::StreamLogger> vlog_logger_;
};

auto Run(FILE* input_stream, llvm::raw_ostream& output_stream,
         llvm::raw_ostream& error_stream, llvm::raw_ostream* vlog_stream,
         Diagnostics::Consumer& consumer) -> bool {
  // The language server internally uses diagnostics for logging issues, but the
  // clangd parts have their own logging system. We intercept that here.
  Logger logger(&error_stream, vlog_stream);
  clang::clangd::LoggingSession logging_session(logger);

  // Set up the connection.
  std::unique_ptr<clang::clangd::Transport> transport(
      clang::clangd::newJSONTransport(input_stream, output_stream,
                                      /*InMirror=*/nullptr,
                                      /*Pretty=*/true));
  OutgoingMessages outgoing(transport.get());
  Context context(vlog_stream, &consumer, &outgoing);
  IncomingMessages incoming(transport.get(), &context);

  // Run the server loop.
  llvm::Error err = transport->loop(incoming);
  if (err) {
    RawStringOstream out;
    out << err;
    CARBON_DIAGNOSTIC(LanguageServerTransportError, Error, "{0}", std::string);
    Diagnostics::NoLocEmitter emitter(&consumer);
    emitter.Emit(LanguageServerTransportError, out.TakeStr());
    return false;
  }
  return true;
}

}  // namespace Carbon::LanguageServer
