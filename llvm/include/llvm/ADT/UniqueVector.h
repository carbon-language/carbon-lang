
//===-- llvm/ADT/UniqueVector.h ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by James M. Laskey and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_UNIQUEVECTOR_H
#define LLVM_ADT_UNIQUEVECTOR_H

#include <map>
#include <vector>

namespace llvm {

//===----------------------------------------------------------------------===//
/// UniqueVector - This class produces a sequential ID number (base 1) for each
/// unique entry that is added.  T is the type of entries in the vector. This
/// class should have an implementation of operator== and of operator<.
/// Entries can be fetched using operator[] with the entry ID. 
template<class T> class UniqueVector {
private:
  // Map - Used to handle the correspondence of entry to ID.
  std::map<T, unsigned> Map;

  // Vector - ID ordered vector of entries. Entries can be indexed by ID - 1.
  //
  std::vector<const T *> Vector;
  
public:
  /// insert - Append entry to the vector if it doesn't already exist.  Returns
  /// the entry's index + 1 to be used as a unique ID.
  unsigned insert(const T &Entry) {
    // Check if the entry is already in the map.
    typename std::map<T, unsigned>::iterator MI = Map.lower_bound(Entry);
    
    // See if entry exists, if so return prior ID.
    if (MI != Map.end() && MI->first == Entry) return MI->second;

    // Compute ID for entry.
    unsigned ID = Vector.size() + 1;
    
    // Insert in map.
    MI = Map.insert(MI, std::make_pair(Entry, ID));
    
    // Insert in vector.
    Vector.push_back(&MI->first);

    return ID;
  }
  
  /// idFor - return the ID for an existing entry.  Returns 0 if the entry is
  /// not found.
  unsigned idFor(const T &Entry) const {
    // Search for entry in the map.
    typename std::map<T, unsigned>::iterator MI = Map.find(Entry);
    
    // See if entry exists, if so return ID.
    if (MI != Map.end()) return MI->second;
    
    // No luck.
    return 0;
  }

  /// operator[] - Returns a reference to the entry with the specified ID.
  ///
  const T &operator[](unsigned ID) const { return *Vector[ID - 1]; }
  
  /// size - Returns the number of entries in the vector.
  ///
  size_t size() const { return Vector.size(); }
  
  /// empty - Returns true if the vector is empty.
  ///
  bool empty() const { return Vector.empty(); }
  
  /// reset - Clears all the entries.
  ///
  void reset() {
    Map.clear();
    Vector.resize(0, 0);
  }
};

} // End of namespace llvm

#endif // LLVM_ADT_UNIQUEVECTOR_H
