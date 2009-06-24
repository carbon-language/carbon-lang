//===- AsmStreamerTest.cpp - Triple unit tests ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

namespace {

// Helper class.
class StringAsmStreamer {
  std::string Str;
  raw_string_ostream OS;
  MCContext Context;
  MCStreamer *Streamer;

public:
  StringAsmStreamer() : OS(Str), Streamer(createAsmStreamer(Context, OS)) {}
  ~StringAsmStreamer() { 
    delete Streamer;
  }

  MCContext &getContext() { return Context; }
  MCStreamer &getStreamer() { return *Streamer; }

  const std::string &getString() {
    Streamer->Finish();
    return Str;
  }
};

TEST(AsmStreamer, EmptyOutput) {
  StringAsmStreamer S;
  EXPECT_EQ(S.getString(), "");
}

TEST(AsmStreamer, Sections) {
  StringAsmStreamer S;
  MCSection *Sec0 = S.getContext().GetSection("foo");
  S.getStreamer().SwitchSection(Sec0);
  EXPECT_EQ(S.getString(), ".section foo\n");
}

}
