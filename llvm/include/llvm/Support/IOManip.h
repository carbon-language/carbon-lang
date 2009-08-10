//===----------------- IOManip.h - iostream manipulators ---------*- C++ -*===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Manipulators to do special-purpose formatting.
//
//===----------------------------------------------------------------------===//

namespace llvm {
  /// Indent - Insert spaces into the character output stream.  The
  /// "level" is multiplied by the "scale" to calculate the number of
  /// spaces to insert.  "level" can represent something like loop
  /// nesting level, for example.
  ///
  class Indent {
  public:
    explicit Indent(int lvl, int amt = 2)
        : level(lvl), scale(amt) {}

    template<typename OStream>
    OStream &operator()(OStream &out) const {
      for(int i = 0; i < level*scale; ++i) {
        out << " ";
      }
      return out;
    }

  private:
    int level;
    int scale;
  };

  template<typename OStream>
  OStream &operator<<(OStream &out, const Indent &indent)
  {
    return(indent(out));
  }
}
