//===-- llvm/Target/SubtargetFeature.h - CPU characteristics ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Jim Laskey and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines and manages user or tool specified CPU characteristics.
// The intent is to be able to package specific features that should or should
// not be used on a specific target processor.  A tool, such as llc, could, as
// as example, gather chip info from the command line, a long with features
// that should be used on that chip.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_SUBTARGETFEATURE_H
#define LLVM_TARGET_SUBTARGETFEATURE_H

#include <string>
#include <vector>
#include <iosfwd>
#include "llvm/Support/DataTypes.h"

namespace llvm {

//===----------------------------------------------------------------------===//
///
/// SubtargetFeatureKV - Used to provide key value pairs for feature and
/// CPU bit flags.
//
struct SubtargetFeatureKV {
  const char *Key;                      // K-V key string
  const char *Desc;                     // Help descriptor
  uint32_t Value;                       // K-V integer value
  
  // Compare routine for std binary search
  bool operator<(const std::string &S) const {
    return strcmp(Key, S.c_str()) < 0;
  }
};
  
//===----------------------------------------------------------------------===//
///
/// SubtargetFeatures - Manages the enabling and disabling of subtarget 
/// specific features.  Features are encoded as a string of the form
///   "cpu,+attr1,+attr2,-attr3,...,+attrN"
/// A comma separates each feature from the next (all lowercase.)
/// The first feature is always the CPU subtype (eg. pentiumm).  If the CPU
/// value is "generic" then the CPU subtype should be generic for the target.
/// Each of the remaining features is prefixed with + or - indicating whether
/// that feature should be enabled or disabled contrary to the cpu
/// specification.
///

class SubtargetFeatures {
  std::vector<std::string> Features;    // Subtarget features as a vector
public:
  SubtargetFeatures(const std::string &Initial = std::string());

  /// Features string accessors.
  std::string getString() const;
  void setString(const std::string &Initial);

  /// Setting CPU string.  Replaces previous setting.  Setting to "" clears CPU.
  ///
  void setCPU(const std::string &String);
  
  /// Adding Features.
  void AddFeature(const std::string &String, bool IsEnabled = true);
           
  /// Parse feature string for quick usage.
  static uint32_t Parse(const std::string &String,
                        const std::string &DefaultCPU,
                        const SubtargetFeatureKV *CPUTable,
                        size_t CPUTableSize,
                        const SubtargetFeatureKV *FeatureTable,
                        size_t FeatureTableSize);
  
  /// Print feature string.
  void print(std::ostream &OS) const;
  
  // Dump feature info.
  void dump() const;
};

} // End namespace llvm

#endif
