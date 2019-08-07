//===- Delta.cpp - Delta Debugging Algorithm Implementation ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation for the Delta Debugging Algorithm:
// it splits a given set of Targets (i.e. Functions, Instructions, BBs, etc.)
// into chunks and tries to reduce the number chunks that are interesting.
//
//===----------------------------------------------------------------------===//

#include "Delta.h"
#include "llvm/Transforms/Utils/Cloning.h"

/// Writes IR code to the given Filepath
static bool writeProgramToFile(StringRef Filepath, int FD, const Module &M) {
  ToolOutputFile Out(Filepath, FD);
  M.print(Out.os(), /*AnnotationWriter=*/nullptr);
  Out.os().close();

  if (!Out.os().has_error()) {
    Out.keep();
    return false;
  }
  return true;
}

/// Creates a temporary (and unique) file inside the tmp folder and writes
/// the given module IR.
static SmallString<128> createTmpFile(Module *M, StringRef TmpDir) {
  SmallString<128> UniqueFilepath;
  int UniqueFD;

  std::error_code EC = sys::fs::createUniqueFile(TmpDir + "/tmp-%%%.ll",
                                                 UniqueFD, UniqueFilepath);
  if (EC) {
    errs() << "Error making unique filename: " << EC.message() << "!\n";
    exit(1);
  }

  if (writeProgramToFile(UniqueFilepath, UniqueFD, *M)) {
    errs() << "Error emitting bitcode to file '" << UniqueFilepath << "'!\n";
    exit(1);
  }
  return UniqueFilepath;
}

/// Prints the Chunk Indexes with the following format: [start, end], if
/// chunk is at minimum size (1), then it just displays [start].
static void printChunks(std::vector<Chunk> Chunks, bool Oneline = false) {
  if (Chunks.empty()) {
    outs() << "No Chunks";
    return;
  }

  for (auto C : Chunks) {
    if (!Oneline)
      outs() << '\t';
    C.print();
    if (!Oneline)
      outs() << '\n';
  }
}

/// Counts the amount of lines for a given file
static unsigned getLines(StringRef Filepath) {
  unsigned Lines = 0;
  std::string CurrLine;
  std::ifstream FileStream(Filepath);

  while (std::getline(FileStream, CurrLine))
    ++Lines;

  return Lines;
}

/// Splits Chunks in half and prints them.
/// If unable to split (when chunk size is 1) returns false.
static bool increaseGranularity(std::vector<Chunk> &Chunks) {
  outs() << "Increasing granularity...";
  std::vector<Chunk> NewChunks;
  bool SplitOne = false;

  for (auto &C : Chunks) {
    if (C.end - C.begin == 0)
      NewChunks.push_back(C);
    else {
      unsigned Half = (C.begin + C.end) / 2;
      NewChunks.push_back({C.begin, Half});
      NewChunks.push_back({Half + 1, C.end});
      SplitOne = true;
    }
  }
  if (SplitOne) {
    Chunks = NewChunks;
    outs() << "Success! New Chunks:\n";
    printChunks(Chunks);
  }
  return SplitOne;
}

bool llvm::runTestWithoutChunks(
    TestRunner &Test, std::function<void(const std::vector<Chunk> &, Module *)>
                          ExtractChunksFromModule) {
  std::unique_ptr<Module> Clone = CloneModule(*Test.getProgram());

  // Generate Module with only Targets inside Current Chunks
  ExtractChunksFromModule({}, Clone.get());
  // Write Module to tmp file
  SmallString<128> CurrentFilepath =
      createTmpFile(Clone.get(), Test.getTmpDir());
  outs() << " | " << sys::path::filename(CurrentFilepath);

  // Completely reduced Program isn't interesting
  if (!Test.run(CurrentFilepath))
    return false;

  // Hooray! We reduced the testcase completely
  Test.setReducedFilepath(CurrentFilepath);
  Test.setProgram(std::move(Clone));
  return true;
}

/// Runs the Delta Debugging algorithm, splits the code into chunks and
/// reduces the amount of chunks that are considered interesting by the
/// given test.
void llvm::runDeltaPass(
    TestRunner &Test, unsigned Targets,
    std::function<void(const std::vector<Chunk> &, Module *)>
        ExtractChunksFromModule) {
  if (!Targets)
    return;

  std::vector<Chunk> Chunks = {{1, Targets}};
  std::set<Chunk> UninterestingChunks;
  std::unique_ptr<Module> ReducedProgram;

  if (!Test.run(Test.getReducedFilepath())) {
    outs() << "\nInput isn't interesting! Verify interesting-ness test\n";
    outs() << "----------------------------\n";
    return;
  }

  if (!increaseGranularity(Chunks)) {
    outs() << "\nCan't reduce anymore\n";
    outs() << "----------------------------\n";
    return;
  }

  do {
    UninterestingChunks = {};
    for (int I = Chunks.size() - 1; I >= 0; --I) {
      std::vector<Chunk> CurrentChunks;

      for (auto C : Chunks)
        if (!UninterestingChunks.count(C) && C != Chunks[I])
          CurrentChunks.push_back(C);

      if (CurrentChunks.empty())
        continue;

      // Clone module before hacking it up..
      std::unique_ptr<Module> CurrentProgram = CloneModule(*Test.getProgram());
      // Generate Module with only Targets inside Current Chunks
      ExtractChunksFromModule(CurrentChunks, CurrentProgram.get());
      // Write Module to tmp file
      SmallString<128> CurrentFilepath =
          createTmpFile(CurrentProgram.get(), Test.getTmpDir());

      outs() << "Testing with: ";
      printChunks(CurrentChunks, /*Oneline=*/true);
      outs() << " | " << sys::path::filename(CurrentFilepath);

      // Current Chunks aren't interesting
      if (!Test.run(CurrentFilepath)) {
        outs() << "\n";
        continue;
      }

      UninterestingChunks.insert(Chunks[I]);
      Test.setReducedFilepath(CurrentFilepath);
      ReducedProgram = std::move(CurrentProgram);
      outs() << " **** SUCCESS | lines: " << getLines(CurrentFilepath) << "\n";
    }
    // Delete uninteresting chunks
    auto UnwantedChunks = Chunks.end();
    UnwantedChunks = std::remove_if(Chunks.begin(), Chunks.end(),
                                    [UninterestingChunks](const Chunk &C) {
                                      return UninterestingChunks.count(C);
                                    });
    Chunks.erase(UnwantedChunks, Chunks.end());

  } while (!UninterestingChunks.empty() || increaseGranularity(Chunks));

  // If we reduced the testcase replace it
  if (ReducedProgram)
    Test.setProgram(std::move(ReducedProgram));
  outs() << "Couldn't increase anymore.\n";
  outs() << "----------------------------\n";
}