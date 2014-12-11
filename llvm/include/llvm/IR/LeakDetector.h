//===- LeakDetector.h - Provide leak detection ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a class that can be used to provide very simple memory leak
// checks for an API.  Basically LLVM uses this to make sure that Instructions,
// for example, are deleted when they are supposed to be, and not leaked away.
//
// When compiling with NDEBUG (Release build), this class does nothing, thus
// adding no checking overhead to release builds.  Note that this class is
// implemented in a very simple way, requiring completely manual manipulation
// and checking for garbage, but this is intentional: users should not be using
// this API, only other APIs should.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_LEAKDETECTOR_H
#define LLVM_IR_LEAKDETECTOR_H

#include <string>

namespace llvm {

class LLVMContext;
class Value;
class MDNode;

struct LeakDetector {
  /// addGarbageObject - Add a pointer to the internal set of "garbage" object
  /// pointers.  This should be called when objects are created, or if they are
  /// taken out of an owning collection.
  ///
  template <class T> static void addGarbageObject(T *Object) {
#ifndef NDEBUG
    addGarbageObjectImpl(Object);
#endif
  }

  /// removeGarbageObject - Remove a pointer from our internal representation of
  /// our "garbage" objects.  This should be called when an object is added to
  /// an "owning" collection.
  ///
  template <class T> static void removeGarbageObject(T *Object) {
#ifndef NDEBUG
    removeGarbageObjectImpl(Object);
#endif
  }

  /// checkForGarbage - Traverse the internal representation of garbage
  /// pointers.  If there are any pointers that have been add'ed, but not
  /// remove'd, big obnoxious warnings about memory leaks are issued.
  ///
  /// The specified message will be printed indicating when the check was
  /// performed.
  ///
  static void checkForGarbage(LLVMContext &C, const std::string &Message) {
#ifndef NDEBUG
    checkForGarbageImpl(C, Message);
#endif
  }

private:
  /// Overload the normal methods to work better with Value* because they are
  /// by far the most common in LLVM.
  ///
  /// Besides making the warning messages nicer, this hides errors by storing
  /// Value* in a different leak-detection container than other classes.
  static void addGarbageObjectImpl(const Value *Object);
  static void removeGarbageObjectImpl(const Value *Object);

  /// Overload the normal methods to work better with MDNode* to improve error
  /// messages.
  ///
  /// For better or worse, this hides errors when other types are added as
  /// garbage, deleted without being removed, and an MDNode is allocated in the
  /// same spot.
  ///
  /// \note Only handle \a MDNode for now, since we can't always get access to
  /// an \a LLVMContext for other \a Metadata types.
  static void addGarbageObjectImpl(const MDNode *Object);
  static void removeGarbageObjectImpl(const MDNode *Object);

  static void addGarbageObjectImpl(void *Object);
  static void removeGarbageObjectImpl(void *Object);
  static void checkForGarbageImpl(LLVMContext &C, const std::string &Message);
};

} // End llvm namespace

#endif
