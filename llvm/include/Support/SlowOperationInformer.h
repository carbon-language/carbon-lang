//===- SlowOperationInformer.h - Keep the user informed ---------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines a simple object which can be used to let the user know what
// is going on when a slow operation is happening, and gives them the ability to
// cancel it.  Potentially slow operations can stack allocate one of these
// objects, and periodically call the "progress" method to update the progress
// bar.  If the operation takes more than 1 second to complete, the progress bar
// is automatically shown and updated.  As such, the slow operation should not
// print stuff to the screen, and should not be confused if an extra line
// appears on the screen (ie, the cursor should be at the start of the line).
//
// If the user presses CTRL-C during the operation, the next invocation of the
// progress method with throw an std::string object indicating that the
// operation was cancelled.  As such, client code must be exception safe around
// the progress method.
//
// Because SlowOperationInformers fiddle around with signals, they cannot be
// nested, and interact poorly with threads.  The SIGALRM handler is set back to
// SIGDFL, but the SIGINT signal handler is restored when the
// SlowOperationInformer is destroyed.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_SLOW_OPERATION_INFORMER_H
#define SUPPORT_SLOW_OPERATION_INFORMER_H

#include <string>
#include <cassert>

namespace llvm {
  class SlowOperationInformer {
    std::string OperationName;
    unsigned LastPrintAmount;
    
    SlowOperationInformer(const SlowOperationInformer&);   // DO NOT IMPLEMENT
    void operator=(const SlowOperationInformer&);          // DO NOT IMPLEMENT
  public:
    SlowOperationInformer(const std::string &Name);
    ~SlowOperationInformer();
    
    /// progress - Clients should periodically call this method when they are in
    /// an exception-safe state.  The Amount variable should indicate how far
    /// along the operation is, given in 1/10ths of a percent (in other words,
    /// Amount should range from 0 to 1000).
    void progress(unsigned Amount);

    /// progress - Same as the method above, but this performs the division for
    /// you, and helps you avoid overflow if you are dealing with largish
    /// numbers.
    void progress(unsigned Current, unsigned Maximum) {
      assert(Maximum != 0 &&
             "Shouldn't be doing work if there is nothing to do!");
      progress(Current*1000ULL/Maximum);
    }
  };
} // end namespace llvm

#endif /* SLOW_OPERATION_INFORMER_H */
