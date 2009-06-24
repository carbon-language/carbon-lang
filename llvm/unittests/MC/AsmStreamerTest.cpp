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
#include "llvm/MC/MCValue.h"
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

TEST(AsmStreamer, Values) {
  StringAsmStreamer S;
  MCSection *Sec0 = S.getContext().GetSection("foo");
  MCSymbol *A = S.getContext().CreateSymbol("a");
  MCSymbol *B = S.getContext().CreateSymbol("b");
  S.getStreamer().SwitchSection(Sec0);
  S.getStreamer().EmitLabel(A);
  S.getStreamer().EmitLabel(B);
  S.getStreamer().EmitValue(MCValue::get(A, B, 10), 1);
  S.getStreamer().EmitValue(MCValue::get(A, B, 10), 2);
  S.getStreamer().EmitValue(MCValue::get(A, B, 10), 4);
  S.getStreamer().EmitValue(MCValue::get(A, B, 10), 8);
  EXPECT_EQ(S.getString(), ".section foo\n\
a:\n\
b:\n\
.byte a - b + 10\n\
.short a - b + 10\n\
.long a - b + 10\n\
.quad a - b + 10\n\
");
}

}
