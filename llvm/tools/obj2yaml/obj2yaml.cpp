//===------ utils/obj2yaml.cpp - obj2yaml conversion tool -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Error.h"
#include "obj2yaml.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/COFF.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"

using namespace llvm;
using namespace llvm::object;

static std::error_code dumpObject(const ObjectFile &Obj) {
  if (Obj.isCOFF())
    return coff2yaml(outs(), cast<COFFObjectFile>(Obj));
  if (Obj.isELF())
    return elf2yaml(outs(), Obj);

  return obj2yaml_error::unsupported_obj_file_format;
}

static std::error_code dumpInput(StringRef File) {
  if (File != "-" && !sys::fs::exists(File))
    return obj2yaml_error::file_not_found;

  ErrorOr<Binary *> BinaryOrErr = createBinary(File);
  if (std::error_code EC = BinaryOrErr.getError())
    return EC;

  std::unique_ptr<Binary> Binary(BinaryOrErr.get());
  // TODO: If this is an archive, then burst it and dump each entry
  if (ObjectFile *Obj = dyn_cast<ObjectFile>(Binary.get()))
    return dumpObject(*Obj);

  return obj2yaml_error::unrecognized_file_format;
}

cl::opt<std::string> InputFilename(cl::Positional, cl::desc("<input file>"),
                                   cl::init("-"));

int main(int argc, char *argv[]) {
  cl::ParseCommandLineOptions(argc, argv);
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);
  llvm_shutdown_obj Y; // Call llvm_shutdown() on exit.

  if (std::error_code EC = dumpInput(InputFilename)) {
    errs() << "Error: '" << EC.message() << "'\n";
    return 1;
  }

  return 0;
}
