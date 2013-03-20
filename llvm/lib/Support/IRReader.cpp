//===- IRReader.cpp - Reader for LLVM IR files ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/IRReader.h"
using namespace llvm;

const char *llvm::TimeIRParsingGroupName = "LLVM IR Parsing";
const char *llvm::TimeIRParsingName = "Parse IR";

bool llvm::TimeIRParsingIsEnabled = false;
static cl::opt<bool,true>
EnableTimeIRParsing("time-ir-parsing", cl::location(TimeIRParsingIsEnabled),
                    cl::desc("Measure the time IR parsing takes"));

