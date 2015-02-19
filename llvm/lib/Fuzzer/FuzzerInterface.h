//===- FuzzerInterface.h - Interface header for the Fuzzer ------*- C++ -* ===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Define the interface between the Fuzzer and the library being tested.
//===----------------------------------------------------------------------===//

#ifndef LLVM_FUZZER_INTERFACE_H
#define LLVM_FUZZER_INTERFACE_H

#include <cstddef>
#include <cstdint>

namespace fuzzer {

typedef void (*UserCallback)(const uint8_t *data, size_t size);
int FuzzerDriver(int argc, char **argv, UserCallback Callback);

}  // namespace fuzzer

#endif  // LLVM_FUZZER_INTERFACE_H
