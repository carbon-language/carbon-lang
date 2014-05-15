//===- yaml2obj - Convert YAML to a binary object file --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This program takes a YAML description of an object file and outputs the
// binary equivalent.
//
// This is used for writing tests that require binary files.
//
//===----------------------------------------------------------------------===//

#include "yaml2obj.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace llvm;

static cl::opt<std::string>
  Input(cl::Positional, cl::desc("<input>"), cl::init("-"));

// TODO: The "right" way to tell what kind of object file a given YAML file
// corresponds to is to look at YAML "tags" (e.g. `!Foo`). Then, different
// tags (`!ELF`, `!COFF`, etc.) would be used to discriminate between them.
// Interpreting the tags is needed eventually for when writing test cases,
// so that we can e.g. have `!Archive` contain a sequence of `!ELF`, and
// just Do The Right Thing. However, interpreting these tags and acting on
// them appropriately requires some work in the YAML parser and the YAMLIO
// library.
enum YAMLObjectFormat {
  YOF_COFF,
  YOF_ELF
};

cl::opt<YAMLObjectFormat> Format(
  "format",
  cl::desc("Interpret input as this type of object file"),
  cl::values(
    clEnumValN(YOF_COFF, "coff", "COFF object file format"),
    clEnumValN(YOF_ELF, "elf", "ELF object file format"),
  clEnumValEnd));

static cl::opt<std::string> OutputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"));

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv);
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);
  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.

  if (OutputFilename.empty())
    OutputFilename = "-";

  std::string ErrorInfo;
  std::unique_ptr<tool_output_file> Out(
      new tool_output_file(OutputFilename.c_str(), ErrorInfo, sys::fs::F_None));
  if (!ErrorInfo.empty()) {
    errs() << ErrorInfo << '\n';
    return 1;
  }

  std::unique_ptr<MemoryBuffer> Buf;
  if (MemoryBuffer::getFileOrSTDIN(Input, Buf))
    return 1;

  int Res = 1;
  if (Format == YOF_COFF)
    Res = yaml2coff(Out->os(), Buf.get());
  else if (Format == YOF_ELF)
    Res = yaml2elf(Out->os(), Buf.get());
  else
    errs() << "Not yet implemented\n";

  if (Res == 0)
    Out->keep();

  return Res;
}
