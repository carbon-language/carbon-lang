//===- SampleProfReader.cpp - Read LLVM sample profile data ---------------===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the class that reads LLVM sample profiles. It
// supports two file formats: text and bitcode. The textual representation
// is useful for debugging and testing purposes. The bitcode representation
// is more compact, resulting in smaller file sizes. However, they can
// both be used interchangeably.
//
// NOTE: If you are making changes to the file format, please remember
//       to document them in the Clang documentation at
//       tools/clang/docs/UsersManual.rst.
//
// Text format
// -----------
//
// Sample profiles are written as ASCII text. The file is divided into
// sections, which correspond to each of the functions executed at runtime.
// Each section has the following format
//
//     function1:total_samples:total_head_samples
//     offset1[.discriminator]: number_of_samples [fn1:num fn2:num ... ]
//     offset2[.discriminator]: number_of_samples [fn3:num fn4:num ... ]
//     ...
//     offsetN[.discriminator]: number_of_samples [fn5:num fn6:num ... ]
//
// The file may contain blank lines between sections and within a
// section. However, the spacing within a single line is fixed. Additional
// spaces will result in an error while reading the file.
//
// Function names must be mangled in order for the profile loader to
// match them in the current translation unit. The two numbers in the
// function header specify how many total samples were accumulated in the
// function (first number), and the total number of samples accumulated
// in the prologue of the function (second number). This head sample
// count provides an indicator of how frequently the function is invoked.
//
// Each sampled line may contain several items. Some are optional (marked
// below):
//
// a. Source line offset. This number represents the line number
//    in the function where the sample was collected. The line number is
//    always relative to the line where symbol of the function is
//    defined. So, if the function has its header at line 280, the offset
//    13 is at line 293 in the file.
//
//    Note that this offset should never be a negative number. This could
//    happen in cases like macros. The debug machinery will register the
//    line number at the point of macro expansion. So, if the macro was
//    expanded in a line before the start of the function, the profile
//    converter should emit a 0 as the offset (this means that the optimizers
//    will not be able to associate a meaningful weight to the instructions
//    in the macro).
//
// b. [OPTIONAL] Discriminator. This is used if the sampled program
//    was compiled with DWARF discriminator support
//    (http://wiki.dwarfstd.org/index.php?title=Path_Discriminators).
//    DWARF discriminators are unsigned integer values that allow the
//    compiler to distinguish between multiple execution paths on the
//    same source line location.
//
//    For example, consider the line of code ``if (cond) foo(); else bar();``.
//    If the predicate ``cond`` is true 80% of the time, then the edge
//    into function ``foo`` should be considered to be taken most of the
//    time. But both calls to ``foo`` and ``bar`` are at the same source
//    line, so a sample count at that line is not sufficient. The
//    compiler needs to know which part of that line is taken more
//    frequently.
//
//    This is what discriminators provide. In this case, the calls to
//    ``foo`` and ``bar`` will be at the same line, but will have
//    different discriminator values. This allows the compiler to correctly
//    set edge weights into ``foo`` and ``bar``.
//
// c. Number of samples. This is an integer quantity representing the
//    number of samples collected by the profiler at this source
//    location.
//
// d. [OPTIONAL] Potential call targets and samples. If present, this
//    line contains a call instruction. This models both direct and
//    number of samples. For example,
//
//      130: 7  foo:3  bar:2  baz:7
//
//    The above means that at relative line offset 130 there is a call
//    instruction that calls one of ``foo()``, ``bar()`` and ``baz()``,
//    with ``baz()`` being the relatively more frequently called target.
//
//===----------------------------------------------------------------------===//

#include "llvm/ProfileData/SampleProfReader.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/Regex.h"

using namespace sampleprof;
using namespace llvm;

/// \brief Print the samples collected for a function on stream \p OS.
///
/// \param OS Stream to emit the output to.
void FunctionSamples::print(raw_ostream &OS) {
  OS << TotalSamples << ", " << TotalHeadSamples << ", " << BodySamples.size()
     << " sampled lines\n";
  for (BodySampleMap::const_iterator SI = BodySamples.begin(),
                                     SE = BodySamples.end();
       SI != SE; ++SI)
    OS << "\tline offset: " << SI->first.LineOffset
       << ", discriminator: " << SI->first.Discriminator
       << ", number of samples: " << SI->second << "\n";
  OS << "\n";
}

/// \brief Print the function profile for \p FName on stream \p OS.
///
/// \param OS Stream to emit the output to.
/// \param FName Name of the function to print.
void SampleProfileReader::printFunctionProfile(raw_ostream &OS,
                                               StringRef FName) {
  OS << "Function: " << FName << ":\n";
  Profiles[FName].print(OS);
}

/// \brief Dump the function profile for \p FName.
///
/// \param FName Name of the function to print.
void SampleProfileReader::dumpFunctionProfile(StringRef FName) {
  printFunctionProfile(dbgs(), FName);
}

/// \brief Dump all the function profiles found.
void SampleProfileReader::dump() {
  for (StringMap<FunctionSamples>::const_iterator I = Profiles.begin(),
                                                  E = Profiles.end();
       I != E; ++I)
    dumpFunctionProfile(I->getKey());
}

/// \brief Load samples from a text file.
///
/// See the documentation at the top of the file for an explanation of
/// the expected format.
///
/// \returns true if the file was loaded successfully, false otherwise.
bool SampleProfileReader::loadText() {
  ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr =
      MemoryBuffer::getFile(Filename);
  if (std::error_code EC = BufferOrErr.getError()) {
    std::string Msg(EC.message());
    M.getContext().diagnose(DiagnosticInfoSampleProfile(Filename.data(), Msg));
    return false;
  }
  MemoryBuffer &Buffer = *BufferOrErr.get();
  line_iterator LineIt(Buffer, /*SkipBlanks=*/true, '#');

  // Read the profile of each function. Since each function may be
  // mentioned more than once, and we are collecting flat profiles,
  // accumulate samples as we parse them.
  Regex HeadRE("^([^0-9].*):([0-9]+):([0-9]+)$");
  Regex LineSample("^([0-9]+)\\.?([0-9]+)?: ([0-9]+)(.*)$");
  while (!LineIt.is_at_eof()) {
    // Read the header of each function.
    //
    // Note that for function identifiers we are actually expecting
    // mangled names, but we may not always get them. This happens when
    // the compiler decides not to emit the function (e.g., it was inlined
    // and removed). In this case, the binary will not have the linkage
    // name for the function, so the profiler will emit the function's
    // unmangled name, which may contain characters like ':' and '>' in its
    // name (member functions, templates, etc).
    //
    // The only requirement we place on the identifier, then, is that it
    // should not begin with a number.
    SmallVector<StringRef, 3> Matches;
    if (!HeadRE.match(*LineIt, &Matches)) {
      reportParseError(LineIt.line_number(),
                       "Expected 'mangled_name:NUM:NUM', found " + *LineIt);
      return false;
    }
    assert(Matches.size() == 4);
    StringRef FName = Matches[1];
    unsigned NumSamples, NumHeadSamples;
    Matches[2].getAsInteger(10, NumSamples);
    Matches[3].getAsInteger(10, NumHeadSamples);
    Profiles[FName] = FunctionSamples();
    FunctionSamples &FProfile = Profiles[FName];
    FProfile.addTotalSamples(NumSamples);
    FProfile.addHeadSamples(NumHeadSamples);
    ++LineIt;

    // Now read the body. The body of the function ends when we reach
    // EOF or when we see the start of the next function.
    while (!LineIt.is_at_eof() && isdigit((*LineIt)[0])) {
      if (!LineSample.match(*LineIt, &Matches)) {
        reportParseError(
            LineIt.line_number(),
            "Expected 'NUM[.NUM]: NUM[ mangled_name:NUM]*', found " + *LineIt);
        return false;
      }
      assert(Matches.size() == 5);
      unsigned LineOffset, NumSamples, Discriminator = 0;
      Matches[1].getAsInteger(10, LineOffset);
      if (Matches[2] != "")
        Matches[2].getAsInteger(10, Discriminator);
      Matches[3].getAsInteger(10, NumSamples);

      // FIXME: Handle called targets (in Matches[4]).

      // When dealing with instruction weights, we use the value
      // zero to indicate the absence of a sample. If we read an
      // actual zero from the profile file, return it as 1 to
      // avoid the confusion later on.
      if (NumSamples == 0)
        NumSamples = 1;
      FProfile.addBodySamples(LineOffset, Discriminator, NumSamples);
      ++LineIt;
    }
  }

  return true;
}

/// \brief Load execution samples from a file.
///
/// This function examines the header of the given file to determine
/// whether to use the text or the bitcode loader.
bool SampleProfileReader::load() {
  // TODO Actually detect the file format.
  return loadText();
}
