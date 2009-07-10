//===-- llvm/CodeGen/AsmFormatter.h - Formatted asm framework ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains various I/O manipulators to pretty-print asm.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/FormattedStream.h"
#include "llvm/Target/TargetAsmInfo.h"

namespace llvm 
{
  /// AsmComment - An I/O manipulator to output an end-of-line comment
  ///
  class AsmComment : public Column {
  private:
    /// CommentColumn - The column at which to output a comment
    ///
    static const int CommentColumn = 60;
    /// Text - The comment to output
    ///
    std::string Text;
    /// TAI - Target information from the code generator
    ///
    const TargetAsmInfo &TAI;
    
  public:
    AsmComment(const TargetAsmInfo &T) 
        : Column(CommentColumn), Text(""), TAI(T) {}

    AsmComment(const std::string &Cmnt,
            const TargetAsmInfo &T) 
        : Column(CommentColumn), Text(Cmnt), TAI(T) {}

    /// operator() - Store a comments tring for later processing.
    ///
    AsmComment &operator()(const std::string &Cmnt) {
      Text = Cmnt;
      return *this;
    }

    /// operator() - Make Comment a functor invoktable by a stream
    /// output operator.  This intentially hides Column's operator().
    ///
    formatted_raw_ostream &operator()(formatted_raw_ostream &Out) const {
      Column::operator()(Out);
      Out << TAI.getCommentString() << " " << Text;
      return(Out);
    }
  };

  /// operator<< - Support comment formatting in formatted streams.
  ///
  inline formatted_raw_ostream &operator<<(formatted_raw_ostream &Out,
                                           const AsmComment &Func)
  {
    return(Func(Out));
  }
}
