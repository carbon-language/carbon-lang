//===-- ClangFuzzer.cpp - Fuzz Clang --------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file implements a function that runs Clang on a single
///  input. This function is then linked into the Fuzzer library.
///
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Tooling.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/CompilerInstance.h"

using namespace clang;

extern "C" void TestOneInput(uint8_t *data, size_t size) {
  std::string s((const char *)data, size);
  llvm::IntrusiveRefCntPtr<FileManager> Files(
      new FileManager(FileSystemOptions()));
  tooling::ToolInvocation Invocation({"clang", "-c", "test.cc"},
                                     new clang::SyntaxOnlyAction, Files.get());
  IgnoringDiagConsumer Diags;
  Invocation.setDiagnosticConsumer(&Diags);
  Invocation.mapVirtualFile("test.cc", s);
  Invocation.run();
}
