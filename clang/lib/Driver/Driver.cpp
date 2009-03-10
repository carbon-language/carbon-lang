//===--- Driver.cpp - Clang GCC Compatible Driver -----------------------*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/Driver.h"

#include "clang/Driver/Arg.h"
#include "clang/Driver/ArgList.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/HostInfo.h"
#include "clang/Driver/Option.h"
#include "clang/Driver/Options.h"

#include "llvm/Support/raw_ostream.h"
using namespace clang::driver;

Driver::Driver(const char *_Name, const char *_Dir,
               const char *_DefaultHostTriple) 
  : Opts(new OptTable()),
    Name(_Name), Dir(_Dir), DefaultHostTriple(_DefaultHostTriple),
    Host(0),
    CCCIsCXX(false), CCCEcho(false), 
    CCCNoClang(false), CCCNoClangCXX(false), CCCNoClangCPP(false)
{
  
}

Driver::~Driver() {
  delete Opts;
}

ArgList *Driver::ParseArgStrings(const char **ArgBegin, const char **ArgEnd) {
  ArgList *Args = new ArgList(ArgBegin, ArgEnd);
  
  unsigned Index = 0, End = ArgEnd - ArgBegin;
  while (Index < End) {
    unsigned Prev = Index;
    Arg *A = getOpts().ParseOneArg(*Args, Index, End);
    if (A)
      Args->append(A);

    assert(Index > Prev && "Parser failed to consume argument.");
  }

  return Args;
}

Compilation *Driver::BuildCompilation(int argc, const char **argv) {
  // FIXME: This stuff needs to go into the Compilation, not the
  // driver.
  bool CCCPrintOptions = false, CCCPrintPhases = false;

  const char **Start = argv + 1, **End = argv + argc;
  const char *HostTriple = DefaultHostTriple.c_str();

  // Read -ccc args. 
  //
  // FIXME: We need to figure out where this behavior should
  // live. Most of it should be outside in the client; the parts that
  // aren't should have proper options, either by introducing new ones
  // or by overloading gcc ones like -V or -b.
  for (; Start != End && memcmp(*Start, "-ccc-", 5) == 0; ++Start) {
    const char *Opt = *Start + 5;
    
    if (!strcmp(Opt, "print-options")) {
      CCCPrintOptions = true;
    } else if (!strcmp(Opt, "print-phases")) {
      CCCPrintPhases = true;
    } else if (!strcmp(Opt, "cxx")) {
      CCCIsCXX = true;
    } else if (!strcmp(Opt, "echo")) {
      CCCEcho = true;
      
    } else if (!strcmp(Opt, "no-clang")) {
      CCCNoClang = true;
    } else if (!strcmp(Opt, "no-clang-cxx")) {
      CCCNoClangCXX = true;
    } else if (!strcmp(Opt, "no-clang-cpp")) {
      CCCNoClangCPP = true;
    } else if (!strcmp(Opt, "clang-archs")) {
      assert(Start+1 < End && "FIXME: -ccc- argument handling.");
      const char *Cur = *++Start;
    
      for (;;) {
        const char *Next = strchr(Cur, ',');

        if (Next) {
          CCCClangArchs.insert(std::string(Cur, Next));
          Cur = Next + 1;
        } else {
          CCCClangArchs.insert(std::string(Cur));
          break;
        }
      }

    } else if (!strcmp(Opt, "host-triple")) {
      assert(Start+1 < End && "FIXME: -ccc- argument handling.");
      HostTriple = *++Start;

    } else {
      // FIXME: Error handling.
      llvm::errs() << "invalid option: " << *Start << "\n";
      exit(1);
    }
  }

  Host = Driver::GetHostInfo(HostTriple);

  ArgList *Args = ParseArgStrings(Start, End);

  // FIXME: This behavior shouldn't be here.
  if (CCCPrintOptions) {
    PrintOptions(Args);
    exit(0);
  }
  
  assert(0 && "FIXME: Implement");

  return new Compilation();
}

void Driver::PrintOptions(const ArgList *Args) {
  unsigned i = 0;
  for (ArgList::const_iterator it = Args->begin(), ie = Args->end(); 
       it != ie; ++it, ++i) {
    Arg *A = *it;
    llvm::errs() << "Option " << i << " - "
                 << "Name: \"" << A->getOption().getName() << "\", "
                 << "Values: {";
    for (unsigned j = 0; j < A->getNumValues(); ++j) {
      if (j)
        llvm::errs() << ", ";
      llvm::errs() << '"' << A->getValue(*Args, j) << '"';
    }
    llvm::errs() << "}\n";
  }
}

HostInfo *Driver::GetHostInfo(const char *Triple) {
  // Dice into arch, platform, and OS. This matches 
  //  arch,platform,os = '(.*?)-(.*?)-(.*?)'
  // and missing fields are left empty.
  std::string Arch, Platform, OS;

  if (const char *ArchEnd = strchr(Triple, '-')) {
    Arch = std::string(Triple, ArchEnd);

    if (const char *PlatformEnd = strchr(ArchEnd+1, '-')) {
      Platform = std::string(ArchEnd+1, PlatformEnd);
      OS = PlatformEnd+1;
    } else
      Platform = ArchEnd+1;
  } else
    Arch = Triple;

  if (memcmp(&Platform[0], "darwin", 6) == 0)
    return new DarwinHostInfo(Arch.c_str(), Platform.c_str(), OS.c_str());
    
  return new UnknownHostInfo(Arch.c_str(), Platform.c_str(), OS.c_str());
}
