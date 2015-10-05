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
// supports two file formats: text and binary. The textual representation
// is useful for debugging and testing purposes. The binary representation
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
//      offset1[.discriminator]: number_of_samples [fn1:num fn2:num ... ]
//      offset2[.discriminator]: number_of_samples [fn3:num fn4:num ... ]
//      ...
//      offsetN[.discriminator]: number_of_samples [fn5:num fn6:num ... ]
//      offsetA[.discriminator]: fnA:num_of_total_samples
//       offsetA1[.discriminator]: number_of_samples [fn7:num fn8:num ... ]
//       ...
//
// This is a nested tree in which the identation represent the nest level
// of the inline stack. There is no blank line in the file. And the spacing
// within a single line is fixed. Additional spaces will result in an error
// while reading the file.
//
// Inline stack is a stack of source locations in which the top of the stack
// represents the leaf function, and the bottom of the stack represents the
// actual symbol in which the instruction belongs.
//
// Function names must be mangled in order for the profile loader to
// match them in the current translation unit. The two numbers in the
// function header specify how many total samples were accumulated in the
// function (first number), and the total number of samples accumulated
// in the prologue of the function (second number). This head sample
// count provides an indicator of how frequently the function is invoked.
//
// There are two types of lines in the function body.
//
// * Sampled line represents the profile information of a source location.
// * Callsite line represents the profile inofrmation of a callsite.
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
// Each callsite line may contain several items. Some are optional.
//
// a. Source line offset. This number represents the line number of the
//    callsite that is inlined in the profiled binary.
//
// b. [OPTIONAL] Discriminator. Same as the discriminator for sampled line.
//
// c. Number of samples. This is an integer quantity representing the
//    total number of samples collected for the inlined instance at this
//    callsite
//===----------------------------------------------------------------------===//

#include "llvm/ProfileData/SampleProfReader.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

using namespace llvm::sampleprof;
using namespace llvm;

/// \brief Print the samples collected for a function on stream \p OS.
///
/// \param OS Stream to emit the output to.
void FunctionSamples::print(raw_ostream &OS) {
  OS << TotalSamples << ", " << TotalHeadSamples << ", " << BodySamples.size()
     << " sampled lines\n";
  for (const auto &SI : BodySamples) {
    LineLocation Loc = SI.first;
    const SampleRecord &Sample = SI.second;
    OS << "\tline offset: " << Loc.LineOffset
       << ", discriminator: " << Loc.Discriminator
       << ", number of samples: " << Sample.getSamples();
    if (Sample.hasCalls()) {
      OS << ", calls:";
      for (const auto &I : Sample.getCallTargets())
        OS << " " << I.first() << ":" << I.second;
    }
    OS << "\n";
  }
  OS << "\n";
}

/// \brief Dump the function profile for \p FName.
///
/// \param FName Name of the function to print.
/// \param OS Stream to emit the output to.
void SampleProfileReader::dumpFunctionProfile(StringRef FName,
                                              raw_ostream &OS) {
  OS << "Function: " << FName << ": ";
  Profiles[FName].print(OS);
}

/// \brief Dump all the function profiles found on stream \p OS.
void SampleProfileReader::dump(raw_ostream &OS) {
  for (const auto &I : Profiles)
    dumpFunctionProfile(I.getKey(), OS);
}

/// \brief Parse \p Input as function head.
///
/// Parse one line of \p Input, and update function name in \p FName,
/// function's total sample count in \p NumSamples, function's entry
/// count in \p NumHeadSamples.
///
/// \returns true if parsing is successful.
static bool ParseHead(const StringRef &Input, StringRef &FName,
                      unsigned &NumSamples, unsigned &NumHeadSamples) {
  if (Input[0] == ' ')
    return false;
  size_t n2 = Input.rfind(':');
  size_t n1 = Input.rfind(':', n2 - 1);
  FName = Input.substr(0, n1);
  if (Input.substr(n1 + 1, n2 - n1 - 1).getAsInteger(10, NumSamples))
    return false;
  if (Input.substr(n2 + 1).getAsInteger(10, NumHeadSamples))
    return false;
  return true;
}

/// \brief Parse \p Input as line sample.
///
/// \param Input input line.
/// \param IsCallsite true if the line represents an inlined callsite.
/// \param Depth the depth of the inline stack.
/// \param NumSamples total samples of the line/inlined callsite.
/// \param LineOffset line offset to the start of the function.
/// \param Discriminator discriminator of the line.
/// \param TargetCountMap map from indirect call target to count.
///
/// returns true if parsing is successful.
static bool ParseLine(const StringRef &Input, bool &IsCallsite, unsigned &Depth,
                      unsigned &NumSamples, unsigned &LineOffset,
                      unsigned &Discriminator, StringRef &CalleeName,
                      DenseMap<StringRef, unsigned> &TargetCountMap) {
  for (Depth = 0; Input[Depth] == ' '; Depth++)
    ;
  if (Depth == 0)
    return false;

  size_t n1 = Input.find(':');
  StringRef Loc = Input.substr(Depth, n1 - Depth);
  size_t n2 = Loc.find('.');
  if (n2 == StringRef::npos) {
    if (Loc.getAsInteger(10, LineOffset))
      return false;
    Discriminator = 0;
  } else {
    if (Loc.substr(0, n2).getAsInteger(10, LineOffset))
      return false;
    if (Loc.substr(n2 + 1).getAsInteger(10, Discriminator))
      return false;
  }

  StringRef Rest = Input.substr(n1 + 2);
  if (Rest[0] >= '0' && Rest[0] <= '9') {
    IsCallsite = false;
    size_t n3 = Rest.find(' ');
    if (n3 == StringRef::npos) {
      if (Rest.getAsInteger(10, NumSamples))
        return false;
    } else {
      if (Rest.substr(0, n3).getAsInteger(10, NumSamples))
        return false;
    }
    while (n3 != StringRef::npos) {
      n3 += Rest.substr(n3).find_first_not_of(' ');
      Rest = Rest.substr(n3);
      n3 = Rest.find(' ');
      StringRef pair = Rest;
      if (n3 != StringRef::npos) {
        pair = Rest.substr(0, n3);
      }
      int n4 = pair.find(':');
      unsigned count;
      if (pair.substr(n4 + 1).getAsInteger(10, count))
        return false;
      TargetCountMap[pair.substr(0, n4)] = count;
    }
  } else {
    IsCallsite = true;
    int n3 = Rest.find_last_of(':');
    CalleeName = Rest.substr(0, n3);
    if (Rest.substr(n3 + 1).getAsInteger(10, NumSamples))
      return false;
  }
  return true;
}

/// \brief Load samples from a text file.
///
/// See the documentation at the top of the file for an explanation of
/// the expected format.
///
/// \returns true if the file was loaded successfully, false otherwise.
std::error_code SampleProfileReaderText::read() {
  line_iterator LineIt(*Buffer, /*SkipBlanks=*/true, '#');

  SmallVector<FunctionSamples *, 10> InlineStack;

  for (; !LineIt.is_at_eof(); ++LineIt) {
    if ((*LineIt)[(*LineIt).find_first_not_of(' ')] == '#')
      continue;
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
    if ((*LineIt)[0] != ' ') {
      unsigned NumSamples, NumHeadSamples;
      StringRef FName;
      if (!ParseHead(*LineIt, FName, NumSamples, NumHeadSamples)) {
        reportError(LineIt.line_number(),
                    "Expected 'mangled_name:NUM:NUM', found " + *LineIt);
        return sampleprof_error::malformed;
      }
      Profiles[FName] = FunctionSamples();
      FunctionSamples &FProfile = Profiles[FName];
      FProfile.addTotalSamples(NumSamples);
      FProfile.addHeadSamples(NumHeadSamples);
      InlineStack.clear();
      InlineStack.push_back(&FProfile);
    } else {
      unsigned NumSamples;
      StringRef FName;
      DenseMap<StringRef, unsigned> TargetCountMap;
      bool IsCallsite;
      unsigned Depth, LineOffset, Discriminator;
      if (!ParseLine(*LineIt, IsCallsite, Depth, NumSamples, LineOffset,
                     Discriminator, FName, TargetCountMap)) {
        reportError(LineIt.line_number(),
                    "Expected 'NUM[.NUM]: NUM[ mangled_name:NUM]*', found " +
                        *LineIt);
        return sampleprof_error::malformed;
      }
      if (IsCallsite) {
        while (InlineStack.size() > Depth) {
          InlineStack.pop_back();
        }
        FunctionSamples &FSamples = InlineStack.back()->functionSamplesAt(
            CallsiteLocation(LineOffset, Discriminator, FName));
        FSamples.addTotalSamples(NumSamples);
        InlineStack.push_back(&FSamples);
      } else {
        while (InlineStack.size() > Depth) {
          InlineStack.pop_back();
        }
        FunctionSamples &FProfile = *InlineStack.back();
        for (const auto &name_count : TargetCountMap) {
          FProfile.addCalledTargetSamples(LineOffset, Discriminator,
                                          name_count.first, name_count.second);
        }
        FProfile.addBodySamples(LineOffset, Discriminator, NumSamples);
      }
    }
  }

  return sampleprof_error::success;
}

template <typename T> ErrorOr<T> SampleProfileReaderBinary::readNumber() {
  unsigned NumBytesRead = 0;
  std::error_code EC;
  uint64_t Val = decodeULEB128(Data, &NumBytesRead);

  if (Val > std::numeric_limits<T>::max())
    EC = sampleprof_error::malformed;
  else if (Data + NumBytesRead > End)
    EC = sampleprof_error::truncated;
  else
    EC = sampleprof_error::success;

  if (EC) {
    reportError(0, EC.message());
    return EC;
  }

  Data += NumBytesRead;
  return static_cast<T>(Val);
}

ErrorOr<StringRef> SampleProfileReaderBinary::readString() {
  std::error_code EC;
  StringRef Str(reinterpret_cast<const char *>(Data));
  if (Data + Str.size() + 1 > End) {
    EC = sampleprof_error::truncated;
    reportError(0, EC.message());
    return EC;
  }

  Data += Str.size() + 1;
  return Str;
}

std::error_code SampleProfileReaderBinary::read() {
  while (!at_eof()) {
    auto FName(readString());
    if (std::error_code EC = FName.getError())
      return EC;

    Profiles[*FName] = FunctionSamples();
    FunctionSamples &FProfile = Profiles[*FName];

    auto Val = readNumber<unsigned>();
    if (std::error_code EC = Val.getError())
      return EC;
    FProfile.addTotalSamples(*Val);

    Val = readNumber<unsigned>();
    if (std::error_code EC = Val.getError())
      return EC;
    FProfile.addHeadSamples(*Val);

    // Read the samples in the body.
    auto NumRecords = readNumber<unsigned>();
    if (std::error_code EC = NumRecords.getError())
      return EC;
    for (unsigned I = 0; I < *NumRecords; ++I) {
      auto LineOffset = readNumber<uint64_t>();
      if (std::error_code EC = LineOffset.getError())
        return EC;

      auto Discriminator = readNumber<uint64_t>();
      if (std::error_code EC = Discriminator.getError())
        return EC;

      auto NumSamples = readNumber<uint64_t>();
      if (std::error_code EC = NumSamples.getError())
        return EC;

      auto NumCalls = readNumber<unsigned>();
      if (std::error_code EC = NumCalls.getError())
        return EC;

      for (unsigned J = 0; J < *NumCalls; ++J) {
        auto CalledFunction(readString());
        if (std::error_code EC = CalledFunction.getError())
          return EC;

        auto CalledFunctionSamples = readNumber<uint64_t>();
        if (std::error_code EC = CalledFunctionSamples.getError())
          return EC;

        FProfile.addCalledTargetSamples(*LineOffset, *Discriminator,
                                        *CalledFunction,
                                        *CalledFunctionSamples);
      }

      FProfile.addBodySamples(*LineOffset, *Discriminator, *NumSamples);
    }
  }

  return sampleprof_error::success;
}

std::error_code SampleProfileReaderBinary::readHeader() {
  Data = reinterpret_cast<const uint8_t *>(Buffer->getBufferStart());
  End = Data + Buffer->getBufferSize();

  // Read and check the magic identifier.
  auto Magic = readNumber<uint64_t>();
  if (std::error_code EC = Magic.getError())
    return EC;
  else if (*Magic != SPMagic())
    return sampleprof_error::bad_magic;

  // Read the version number.
  auto Version = readNumber<uint64_t>();
  if (std::error_code EC = Version.getError())
    return EC;
  else if (*Version != SPVersion())
    return sampleprof_error::unsupported_version;

  return sampleprof_error::success;
}

bool SampleProfileReaderBinary::hasFormat(const MemoryBuffer &Buffer) {
  const uint8_t *Data =
      reinterpret_cast<const uint8_t *>(Buffer.getBufferStart());
  uint64_t Magic = decodeULEB128(Data);
  return Magic == SPMagic();
}

bool SourceInfo::operator<(const SourceInfo &P) const {
  if (Line != P.Line)
    return Line < P.Line;
  if (StartLine != P.StartLine)
    return StartLine < P.StartLine;
  if (Discriminator != P.Discriminator)
    return Discriminator < P.Discriminator;
  return FuncName < P.FuncName;
}

std::error_code SampleProfileReaderGCC::skipNextWord() {
  uint32_t dummy;
  if (!GcovBuffer.readInt(dummy))
    return sampleprof_error::truncated;
  return sampleprof_error::success;
}

template <typename T> ErrorOr<T> SampleProfileReaderGCC::readNumber() {
  if (sizeof(T) <= sizeof(uint32_t)) {
    uint32_t Val;
    if (GcovBuffer.readInt(Val) && Val <= std::numeric_limits<T>::max())
      return static_cast<T>(Val);
  } else if (sizeof(T) <= sizeof(uint64_t)) {
    uint64_t Val;
    if (GcovBuffer.readInt64(Val) && Val <= std::numeric_limits<T>::max())
      return static_cast<T>(Val);
  }

  std::error_code EC = sampleprof_error::malformed;
  reportError(0, EC.message());
  return EC;
}

ErrorOr<StringRef> SampleProfileReaderGCC::readString() {
  StringRef Str;
  if (!GcovBuffer.readString(Str))
    return sampleprof_error::truncated;
  return Str;
}

std::error_code SampleProfileReaderGCC::readHeader() {
  // Read the magic identifier.
  if (!GcovBuffer.readGCDAFormat())
    return sampleprof_error::unrecognized_format;

  // Read the version number. Note - the GCC reader does not validate this
  // version, but the profile creator generates v704.
  GCOV::GCOVVersion version;
  if (!GcovBuffer.readGCOVVersion(version))
    return sampleprof_error::unrecognized_format;

  if (version != GCOV::V704)
    return sampleprof_error::unsupported_version;

  // Skip the empty integer.
  if (std::error_code EC = skipNextWord())
    return EC;

  return sampleprof_error::success;
}

std::error_code SampleProfileReaderGCC::readSectionTag(uint32_t Expected) {
  uint32_t Tag;
  if (!GcovBuffer.readInt(Tag))
    return sampleprof_error::truncated;

  if (Tag != Expected)
    return sampleprof_error::malformed;

  if (std::error_code EC = skipNextWord())
    return EC;

  return sampleprof_error::success;
}

std::error_code SampleProfileReaderGCC::readNameTable() {
  if (std::error_code EC = readSectionTag(GCOVTagAFDOFileNames))
    return EC;

  uint32_t Size;
  if (!GcovBuffer.readInt(Size))
    return sampleprof_error::truncated;

  for (uint32_t I = 0; I < Size; ++I) {
    StringRef Str;
    if (!GcovBuffer.readString(Str))
      return sampleprof_error::truncated;
    Names.push_back(Str);
  }

  return sampleprof_error::success;
}

std::error_code SampleProfileReaderGCC::readFunctionProfiles() {
  if (std::error_code EC = readSectionTag(GCOVTagAFDOFunction))
    return EC;

  uint32_t NumFunctions;
  if (!GcovBuffer.readInt(NumFunctions))
    return sampleprof_error::truncated;

  SourceStack Stack;
  for (uint32_t I = 0; I < NumFunctions; ++I)
    if (std::error_code EC = readOneFunctionProfile(Stack, true))
      return EC;

  return sampleprof_error::success;
}

std::error_code SampleProfileReaderGCC::addSourceCount(StringRef Name,
                                                       const SourceStack &Src,
                                                       uint64_t Count) {
  if (Src.size() == 0 || Src[0].Malformed())
    return sampleprof_error::malformed;
  FunctionSamples &FProfile = Profiles[Name];
  FProfile.addTotalSamples(Count);
  // FIXME(dnovillo) - Properly update inline stack for FnName.
  FProfile.addBodySamples(Src[0].Line, Src[0].Discriminator, Count);
  return sampleprof_error::success;
}

std::error_code
SampleProfileReaderGCC::readOneFunctionProfile(const SourceStack &Stack,
                                               bool Update) {
  uint64_t HeadCount = 0;
  if (Stack.size() == 0)
    if (!GcovBuffer.readInt64(HeadCount))
      return sampleprof_error::truncated;

  uint32_t NameIdx;
  if (!GcovBuffer.readInt(NameIdx))
    return sampleprof_error::truncated;

  StringRef Name(Names[NameIdx]);

  uint32_t NumPosCounts;
  if (!GcovBuffer.readInt(NumPosCounts))
    return sampleprof_error::truncated;

  uint32_t NumCallSites;
  if (!GcovBuffer.readInt(NumCallSites))
    return sampleprof_error::truncated;

  if (Stack.size() == 0) {
    FunctionSamples &FProfile = Profiles[Name];
    FProfile.addHeadSamples(HeadCount);
    if (FProfile.getTotalSamples() > 0)
      Update = false;
  }

  for (uint32_t I = 0; I < NumPosCounts; ++I) {
    uint32_t Offset;
    if (!GcovBuffer.readInt(Offset))
      return sampleprof_error::truncated;

    uint32_t NumTargets;
    if (!GcovBuffer.readInt(NumTargets))
      return sampleprof_error::truncated;

    uint64_t Count;
    if (!GcovBuffer.readInt64(Count))
      return sampleprof_error::truncated;

    SourceInfo Info(Name, "", "", 0, Offset >> 16, Offset & 0xffff);
    SourceStack NewStack;
    NewStack.push_back(Info);
    NewStack.insert(NewStack.end(), Stack.begin(), Stack.end());
    if (Update)
      addSourceCount(NewStack[NewStack.size() - 1].FuncName, NewStack, Count);

    for (uint32_t J = 0; J < NumTargets; J++) {
      uint32_t HistVal;
      if (!GcovBuffer.readInt(HistVal))
        return sampleprof_error::truncated;

      if (HistVal != HIST_TYPE_INDIR_CALL_TOPN)
        return sampleprof_error::malformed;

      uint64_t TargetIdx;
      if (!GcovBuffer.readInt64(TargetIdx))
        return sampleprof_error::truncated;
      StringRef TargetName(Names[TargetIdx]);

      uint64_t TargetCount;
      if (!GcovBuffer.readInt64(TargetCount))
        return sampleprof_error::truncated;

      if (Update) {
        FunctionSamples &TargetProfile = Profiles[TargetName];
        TargetProfile.addBodySamples(NewStack[0].Line,
                                     NewStack[0].Discriminator, TargetCount);
      }
    }
  }

  for (uint32_t I = 0; I < NumCallSites; I++) {
    // The offset is encoded as:
    //   high 16 bits: line offset to the start of the function.
    //   low 16 bits: discriminator.
    uint32_t Offset;
    if (!GcovBuffer.readInt(Offset))
      return sampleprof_error::truncated;
    SourceInfo Info(Name, "", "", 0, Offset >> 16, Offset & 0xffff);
    SourceStack NewStack;
    NewStack.push_back(Info);
    NewStack.insert(NewStack.end(), Stack.begin(), Stack.end());
    if (std::error_code EC = readOneFunctionProfile(NewStack, Update))
      return EC;
  }

  return sampleprof_error::success;
}

/// \brief Read a GCC AutoFDO profile.
///
/// This format is generated by the Linux Perf conversion tool at
/// https://github.com/google/autofdo.
std::error_code SampleProfileReaderGCC::read() {
  // Read the string table.
  if (std::error_code EC = readNameTable())
    return EC;

  // Read the source profile.
  if (std::error_code EC = readFunctionProfiles())
    return EC;

  return sampleprof_error::success;
}

bool SampleProfileReaderGCC::hasFormat(const MemoryBuffer &Buffer) {
  StringRef Magic(reinterpret_cast<const char *>(Buffer.getBufferStart()));
  return Magic == "adcg*704";
}

/// \brief Prepare a memory buffer for the contents of \p Filename.
///
/// \returns an error code indicating the status of the buffer.
static ErrorOr<std::unique_ptr<MemoryBuffer>>
setupMemoryBuffer(std::string Filename) {
  auto BufferOrErr = MemoryBuffer::getFileOrSTDIN(Filename);
  if (std::error_code EC = BufferOrErr.getError())
    return EC;
  auto Buffer = std::move(BufferOrErr.get());

  // Sanity check the file.
  if (Buffer->getBufferSize() > std::numeric_limits<unsigned>::max())
    return sampleprof_error::too_large;

  return std::move(Buffer);
}

/// \brief Create a sample profile reader based on the format of the input file.
///
/// \param Filename The file to open.
///
/// \param Reader The reader to instantiate according to \p Filename's format.
///
/// \param C The LLVM context to use to emit diagnostics.
///
/// \returns an error code indicating the status of the created reader.
ErrorOr<std::unique_ptr<SampleProfileReader>>
SampleProfileReader::create(StringRef Filename, LLVMContext &C) {
  auto BufferOrError = setupMemoryBuffer(Filename);
  if (std::error_code EC = BufferOrError.getError())
    return EC;

  auto Buffer = std::move(BufferOrError.get());
  std::unique_ptr<SampleProfileReader> Reader;
  if (SampleProfileReaderBinary::hasFormat(*Buffer))
    Reader.reset(new SampleProfileReaderBinary(std::move(Buffer), C));
  else if (SampleProfileReaderGCC::hasFormat(*Buffer))
    Reader.reset(new SampleProfileReaderGCC(std::move(Buffer), C));
  else
    Reader.reset(new SampleProfileReaderText(std::move(Buffer), C));

  if (std::error_code EC = Reader->readHeader())
    return EC;

  return std::move(Reader);
}
