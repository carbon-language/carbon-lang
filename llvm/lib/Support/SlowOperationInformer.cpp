//===-- SlowOperationInformer.cpp - Keep the user informed ----------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
// 
// This file implements the SlowOperationInformer class for the LLVM debugger.
//
//===----------------------------------------------------------------------===//

#include "Support/SlowOperationInformer.h"
#include "Config/config.h"     // Get the signal handler return type
#include <iostream>
#include <sstream>
#include <signal.h>
#include <unistd.h>
#include <cassert>
using namespace llvm;

/// OperationCancelled - This flag is set by the SIGINT signal handler if the
/// user presses CTRL-C.
static volatile bool OperationCancelled;

/// ShouldShowStatus - This flag gets set if the operation takes a long time.
///
static volatile bool ShouldShowStatus;

/// NestedSOI - Sanity check.  SlowOperationInformers cannot be nested or run in
/// parallel.  This ensures that they never do.
static bool NestedSOI = false;

static RETSIGTYPE SigIntHandler(int Sig) {
  OperationCancelled = true;
  signal(SIGINT, SigIntHandler);
}

static RETSIGTYPE SigAlarmHandler(int Sig) {
  ShouldShowStatus = true;
}

static void (*OldSigIntHandler) (int);


SlowOperationInformer::SlowOperationInformer(const std::string &Name)
  : OperationName(Name), LastPrintAmount(0) {
  assert(!NestedSOI && "SlowerOperationInformer objects cannot be nested!");
  NestedSOI = true;

  OperationCancelled = 0;
  ShouldShowStatus = 0;

  signal(SIGALRM, SigAlarmHandler);
  OldSigIntHandler = signal(SIGINT, SigIntHandler);
  alarm(1);
}

SlowOperationInformer::~SlowOperationInformer() {
  NestedSOI = false;
  if (LastPrintAmount) {
    // If we have printed something, make _sure_ we print the 100% amount, and
    // also print a newline.
    std::cout << std::string(LastPrintAmount, '\b') << "Progress "
              << OperationName << ": 100%  \n";
  }

  alarm(0);
  signal(SIGALRM, SIG_DFL);
  signal(SIGINT, OldSigIntHandler);
}

/// progress - Clients should periodically call this method when they are in
/// an exception-safe state.  The Amount variable should indicate how far
/// along the operation is, given in 1/10ths of a percent (in other words,
/// Amount should range from 0 to 1000).
void SlowOperationInformer::progress(unsigned Amount) {
  if (OperationCancelled) {
    std::cout << "\n";
    LastPrintAmount = 0;
    throw "While " + OperationName + ", operation cancelled.";
  }

  // If we haven't spent enough time in this operation to warrant displaying the
  // progress bar, don't do so yet.
  if (!ShouldShowStatus)
    return;

  // Delete whatever we printed last time.
  std::string ToPrint = std::string(LastPrintAmount, '\b');

  std::ostringstream OS;
  OS << "Progress " << OperationName << ": " << Amount/10;
  if (unsigned Rem = Amount % 10)
    OS << "." << Rem << "%";
  else
    OS << "%  ";

  LastPrintAmount = OS.str().size();
  std::cout << ToPrint+OS.str() << std::flush;
}
