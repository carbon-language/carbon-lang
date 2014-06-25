//==- llvm/Support/RandomNumberGenerator.h - RNG for diversity ---*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines an abstraction for random number generation (RNG).
// Note that the current implementation is not cryptographically secure
// as it uses the C++11 <random> facilities.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_RANDOMNUMBERGENERATOR_H_
#define LLVM_SUPPORT_RANDOMNUMBERGENERATOR_H_

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/DataTypes.h" // Needed for uint64_t on Windows.
#include <random>

namespace llvm {

/// A random number generator.
/// Instances of this class should not be shared across threads.
class RandomNumberGenerator {
public:
  /// Seeds and salts the underlying RNG engine. The salt of type StringRef
  /// is passed into the constructor. The seed can be set on the command
  /// line via -rng-seed=<uint64>.
  /// The reason for the salt is to ensure different random streams even if
  /// the same seed is used for multiple invocations of the compiler.
  /// A good salt value should add additional entropy and be constant across
  /// different machines (i.e., no paths) to allow for reproducible builds.
  /// An instance of this class can be retrieved from the current Module.
  /// \see Module::getRNG
  RandomNumberGenerator(StringRef Salt);

  /// Returns a random number in the range [0, Max).
  uint64_t next(uint64_t Max);

private:
  // 64-bit Mersenne Twister by Matsumoto and Nishimura, 2000
  // http://en.cppreference.com/w/cpp/numeric/random/mersenne_twister_engine
  std::mt19937_64 Generator;

  // Noncopyable.
  RandomNumberGenerator(const RandomNumberGenerator &other)
      LLVM_DELETED_FUNCTION;
  RandomNumberGenerator &
  operator=(const RandomNumberGenerator &other) LLVM_DELETED_FUNCTION;
};
}

#endif
