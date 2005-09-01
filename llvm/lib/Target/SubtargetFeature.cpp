//===- SubtargetFeature.cpp - CPU characteristics Implementation ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Jim Laskey and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the SubtargetFeature interface.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/SubtargetFeature.h"

#include <string>
#include <algorithm>
#include <vector>
#include <cassert>


using namespace llvm;

/// Splits a string of comma separated items in to a vector of strings.
void SubtargetFeatures::Split(std::vector<std::string> &V,
                              const std::string &S) {
  // Start at beginning of string.
  size_t Pos = 0;
  while (true) {
    // Find the next comma
    size_t Comma = S.find(',', Pos);
    // If no comma found then the the rest of the string is used
    if (Comma == std::string::npos) {
      // Add string to vector
      V.push_back(S.substr(Pos));
      break;
    }
    // Otherwise add substring to vector
    V.push_back(S.substr(Pos, Comma - Pos));
    // Advance to next item
    Pos = Comma + 1;
  }
}

/// Join a vector of strings to a string with a comma separating each element.
std::string SubtargetFeatures::Join(const std::vector<std::string> &V) {
  // Start with empty string.
  std::string Result;
  // If the vector is not empty 
  if (!V.empty()) {
    // Start with the CPU feature
    Result = V[0];
    // For each successive feature
    for (size_t i = 1; i < V.size(); i++) {
      // Add a comma
      Result += ",";
      // Add the feature
      Result += V[i];
    }
  }
  // Return the features string 
  return Result;
}

/// Convert a string to lowercase.
std::string SubtargetFeatures::toLower(const std::string &S) {
  // Copy the string
  std::string Result = S;
  // For each character in string
  for (size_t i = 0; i < Result.size(); i++) {
    // Convert character to lowercase
    Result[i] = std::tolower(Result[i]);
  }
  // Return the lowercased string
  return Result;
}

/// Adding features.
void SubtargetFeatures::AddFeature(const std::string &String,
                                   bool IsEnabled) {
  // Don't add empty features
  if (!String.empty()) {
    // Convert to lowercase, prepend flag and add to vector
    Features.push_back(PrependFlag(toLower(String), IsEnabled));
  }
}

/// Find item in array using binary search.
const SubtargetFeatureKV *
SubtargetFeatures::Find(const std::string &S,
                        const SubtargetFeatureKV *A, size_t L) {
  // Determine the end of the array
  const SubtargetFeatureKV *Hi = A + L;
  // Binary search the array
  const SubtargetFeatureKV *F = std::lower_bound(A, Hi, S);
  // If not found then return NULL
  if (F == Hi || std::string(F->Key) != S) return NULL;
  // Return the found array item
  return F;
}

/// Parse feature string for quick usage.
uint32_t SubtargetFeatures::Parse(const std::string &String,
                                  const std::string &DefaultCPU,
                                  const SubtargetFeatureKV *CPUTable,
                                  size_t CPUTableSize,
                                  const SubtargetFeatureKV *FeatureTable,
                                  size_t FeatureTableSize) {
  assert(CPUTable && "missing CPU table");
  assert(FeatureTable && "missing features table");
#ifndef NDEBUG
  for (size_t i = 1; i < CPUTableSize; i++) {
    assert(strcmp(CPUTable[i - 1].Key, CPUTable[i].Key) < 0 &&
           "CPU table is not sorted");
  }
  for (size_t i = 1; i < FeatureTableSize; i++) {
    assert(strcmp(FeatureTable[i - 1].Key, FeatureTable[i].Key) < 0 &&
          "CPU features table is not sorted");
  }
#endif
  std::vector<std::string> Features;    // Subtarget features as a vector
  uint32_t Bits = 0;                    // Resulting bits
  // Split up features
  Split(Features, String);
  // Check if default is needed
  if (Features[0].empty()) Features[0] = DefaultCPU;
  // Find CPU entry
  const SubtargetFeatureKV *CPUEntry =
                            Find(Features[0], CPUTable, CPUTableSize);
  // If there is a match
  if (CPUEntry) {
    // Set base feature bits
    Bits = CPUEntry->Value;
  } else {
    std::cerr << Features[0]
              << " is not a recognized processor for this target"
              << " (ignoring processor)"
              << "\n";
  }
  // Iterate through each feature
  for (size_t i = 1; i < Features.size(); i++) {
    // Get next feature
    const std::string &Feature = Features[i];
    // Find feature in table.
    const SubtargetFeatureKV *FeatureEntry =
                       Find(StripFlag(Feature), FeatureTable, FeatureTableSize);
    // If there is a match
    if (FeatureEntry) {
      // Enable/disable feature in bits
      if (isEnabled(Feature)) Bits |=  FeatureEntry->Value;
      else                    Bits &= ~FeatureEntry->Value;
    } else {
      std::cerr << Feature
                << " is not a recognized feature for this target"
                << " (ignoring feature)"
                << "\n";
    }
  }
  return Bits;
}

/// Print feature string.
void SubtargetFeatures::print(std::ostream &OS) const {
  for (size_t i = 0; i < Features.size(); i++) {
    OS << Features[i] << "  ";
  }
  OS << "\n";
}

/// Dump feature info.
void SubtargetFeatures::dump() const {
  print(std::cerr);
}
