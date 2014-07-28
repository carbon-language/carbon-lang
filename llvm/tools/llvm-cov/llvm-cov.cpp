//===- llvm-cov.cpp - LLVM coverage tool ----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// llvm-cov is a command line tools to analyze and report coverage information.
//
//===----------------------------------------------------------------------===//

/// \brief The main function for the gcov compatible coverage tool
int gcov_main(int argc, const char **argv);

int main(int argc, const char **argv) {
  return gcov_main(argc, argv);
}
