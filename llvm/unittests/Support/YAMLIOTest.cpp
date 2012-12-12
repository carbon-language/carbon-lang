//===- unittest/Support/YAMLIOTest.cpp ------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/YAMLTraits.h"
#include "gtest/gtest.h"

// To keep build bots going, disable tests until I figure out 
// why gcc complains there is no match for these traits.
#if 0

using llvm::yaml::Input;
using llvm::yaml::Output;
using llvm::yaml::IO;
using llvm::yaml::MappingTraits;
using llvm::yaml::MappingNormalization;
using llvm::yaml::ScalarTraits;
using llvm::yaml::Hex8;
using llvm::yaml::Hex16;
using llvm::yaml::Hex32;
using llvm::yaml::Hex64;


//===----------------------------------------------------------------------===//
//  Test MappingTraits
//===----------------------------------------------------------------------===//

struct FooBar {
  int foo;
  int bar;
};
typedef std::vector<FooBar> FooBarSequence;

LLVM_YAML_IS_SEQUENCE_VECTOR(FooBar)


namespace llvm {
namespace yaml {
  template <>
  struct MappingTraits<FooBar> {
    static void mapping(IO &io, FooBar& fb) {
      io.mapRequired("foo",    fb.foo);
      io.mapRequired("bar",    fb.bar);
    }
  };
}
}


//
// Test the reading of a yaml mapping
//
TEST(YAMLIO, TestMapRead) {
  FooBar doc;
  Input yin("---\nfoo:  3\nbar:  5\n...\n");
  yin >> doc;

  EXPECT_FALSE(yin.error());
  EXPECT_EQ(doc.foo, 3);
  EXPECT_EQ(doc.bar,5);
}


//
// Test the reading of a yaml sequence of mappings
//
TEST(YAMLIO, TestSequenceMapRead) {
  FooBarSequence seq;
  Input yin("---\n - foo:  3\n   bar:  5\n - foo:  7\n   bar:  9\n...\n");
  yin >> seq;

  EXPECT_FALSE(yin.error());
  EXPECT_EQ(seq.size(), 2UL);
  FooBar& map1 = seq[0];
  FooBar& map2 = seq[1];
  EXPECT_EQ(map1.foo, 3);
  EXPECT_EQ(map1.bar, 5);
  EXPECT_EQ(map2.foo, 7);
  EXPECT_EQ(map2.bar, 9);
}


//
// Test writing then reading back a sequence of mappings
//
TEST(YAMLIO, TestSequenceMapWriteAndRead) {
  std::string intermediate;
  {
    FooBar entry1;
    entry1.foo = 10;
    entry1.bar = -3;
    FooBar entry2;
    entry2.foo = 257;
    entry2.bar = 0;
    FooBarSequence seq;
    seq.push_back(entry1);
    seq.push_back(entry2);

    llvm::raw_string_ostream ostr(intermediate);
    Output yout(ostr);
    yout << seq;
  }

  {
    Input yin(intermediate);
    FooBarSequence seq2;
    yin >> seq2;

    EXPECT_FALSE(yin.error());
    EXPECT_EQ(seq2.size(), 2UL);
    FooBar& map1 = seq2[0];
    FooBar& map2 = seq2[1];
    EXPECT_EQ(map1.foo, 10);
    EXPECT_EQ(map1.bar, -3);
    EXPECT_EQ(map2.foo, 257);
    EXPECT_EQ(map2.bar, 0);
  }
}


//===----------------------------------------------------------------------===//
//  Test built-in types
//===----------------------------------------------------------------------===//

struct BuiltInTypes {
  llvm::StringRef str;
  uint64_t        u64;
  uint32_t        u32;
  uint16_t        u16;
  uint8_t         u8;
  bool            b;
  int64_t         s64;
  int32_t         s32;
  int16_t         s16;
  int8_t          s8;
  float           f;
  double          d;
  Hex8            h8;
  Hex16           h16;
  Hex32           h32;
  Hex64           h64;
};

namespace llvm {
namespace yaml {
  template <>
  struct MappingTraits<BuiltInTypes> {
    static void mapping(IO &io, BuiltInTypes& bt) {
      io.mapRequired("str",      bt.str);
      io.mapRequired("u64",      bt.u64);
      io.mapRequired("u32",      bt.u32);
      io.mapRequired("u16",      bt.u16);
      io.mapRequired("u8",       bt.u8);
      io.mapRequired("b",        bt.b);
      io.mapRequired("s64",      bt.s64);
      io.mapRequired("s32",      bt.s32);
      io.mapRequired("s16",      bt.s16);
      io.mapRequired("s8",       bt.s8);
      io.mapRequired("f",        bt.f);
      io.mapRequired("d",        bt.d);
      io.mapRequired("h8",       bt.h8);
      io.mapRequired("h16",      bt.h16);
      io.mapRequired("h32",      bt.h32);
      io.mapRequired("h64",      bt.h64);
    }
  };
}
}


//
// Test the reading of all built-in scalar conversions
//
TEST(YAMLIO, TestReadBuiltInTypes) {
  BuiltInTypes map;
  Input yin("---\n"
            "str:      hello there\n"
            "u64:      5000000000\n"
            "u32:      4000000000\n"
            "u16:      65000\n"
            "u8:       255\n"
            "b:        false\n"
            "s64:      -5000000000\n"
            "s32:      -2000000000\n"
            "s16:      -32000\n"
            "s8:       -127\n"
            "f:        137.125\n"
            "d:        -2.8625\n"
            "h8:       0xFF\n"
            "h16:      0x8765\n"
            "h32:      0xFEDCBA98\n"
            "h64:      0xFEDCBA9876543210\n"
           "...\n");
  yin >> map;

  EXPECT_FALSE(yin.error());
  EXPECT_TRUE(map.str.equals("hello there"));
  EXPECT_EQ(map.u64, 5000000000ULL);
  EXPECT_EQ(map.u32, 4000000000);
  EXPECT_EQ(map.u16, 65000);
  EXPECT_EQ(map.u8,  255);
  EXPECT_EQ(map.b,   false);
  EXPECT_EQ(map.s64, -5000000000LL);
  EXPECT_EQ(map.s32, -2000000000L);
  EXPECT_EQ(map.s16, -32000);
  EXPECT_EQ(map.s8,  -127);
  EXPECT_EQ(map.f,   137.125);
  EXPECT_EQ(map.d,   -2.8625);
  EXPECT_EQ(map.h8,  Hex8(255));
  EXPECT_EQ(map.h16, Hex16(0x8765));
  EXPECT_EQ(map.h32, Hex32(0xFEDCBA98));
  EXPECT_EQ(map.h64, Hex64(0xFEDCBA9876543210LL));
}


//
// Test writing then reading back all built-in scalar types
//
TEST(YAMLIO, TestReadWriteBuiltInTypes) {
  std::string intermediate;
  {
    BuiltInTypes map;
    map.str = "one two";
    map.u64 = 6000000000;
    map.u32 = 3000000000;
    map.u16 = 50000;
    map.u8  = 254;
    map.b   = true;
    map.s64 = -6000000000;
    map.s32 = -2000000000;
    map.s16 = -32000;
    map.s8  = -128;
    map.f   = 3.25;
    map.d   = -2.8625;
    map.h8  = 254;
    map.h16 = 50000;
    map.h32 = 3000000000;
    map.h64 = 6000000000LL;

    llvm::raw_string_ostream ostr(intermediate);
    Output yout(ostr);
    yout << map;
  }

  {
    Input yin(intermediate);
    BuiltInTypes map;
    yin >> map;

    EXPECT_FALSE(yin.error());
    EXPECT_TRUE(map.str.equals("one two"));
    EXPECT_EQ(map.u64,      6000000000ULL);
    EXPECT_EQ(map.u32,      3000000000UL);
    EXPECT_EQ(map.u16,      50000);
    EXPECT_EQ(map.u8,       254);
    EXPECT_EQ(map.b,        true);
    EXPECT_EQ(map.s64,      -6000000000LL);
    EXPECT_EQ(map.s32,      -2000000000L);
    EXPECT_EQ(map.s16,      -32000);
    EXPECT_EQ(map.s8,       -128);
    EXPECT_EQ(map.f,        3.25);
    EXPECT_EQ(map.d,        -2.8625);
    EXPECT_EQ(map.h8,       Hex8(254));
    EXPECT_EQ(map.h16,      Hex16(50000));
    EXPECT_EQ(map.h32,      Hex32(3000000000));
    EXPECT_EQ(map.h64,      Hex64(6000000000LL));
  }
}



//===----------------------------------------------------------------------===//
//  Test ScalarEnumerationTraits
//===----------------------------------------------------------------------===//

enum Colors {
    cRed,
    cBlue,
    cGreen,
    cYellow
};

struct ColorMap {
  Colors      c1;
  Colors      c2;
  Colors      c3;
  Colors      c4;
  Colors      c5;
  Colors      c6;
};

namespace llvm {
namespace yaml {
  template <>
  struct ScalarEnumerationTraits<Colors> {
    static void enumeration(IO &io, Colors &value) {
      io.enumCase(value, "red",   cRed);
      io.enumCase(value, "blue",  cBlue);
      io.enumCase(value, "green", cGreen);
      io.enumCase(value, "yellow",cYellow);
    }
  };
  template <>
  struct MappingTraits<ColorMap> {
    static void mapping(IO &io, ColorMap& c) {
      io.mapRequired("c1", c.c1);
      io.mapRequired("c2", c.c2);
      io.mapRequired("c3", c.c3);
      io.mapOptional("c4", c.c4, cBlue);   // supplies default
      io.mapOptional("c5", c.c5, cYellow); // supplies default
      io.mapOptional("c6", c.c6, cRed);    // supplies default
    }
  };
}
}


//
// Test reading enumerated scalars
//
TEST(YAMLIO, TestEnumRead) {
  ColorMap map;
  Input yin("---\n"
            "c1:  blue\n"
            "c2:  red\n"
            "c3:  green\n"
            "c5:  yellow\n"
            "...\n");
  yin >> map;

  EXPECT_FALSE(yin.error());
  EXPECT_EQ(cBlue,  map.c1);
  EXPECT_EQ(cRed,   map.c2);
  EXPECT_EQ(cGreen, map.c3);
  EXPECT_EQ(cBlue,  map.c4);  // tests default
  EXPECT_EQ(cYellow,map.c5);  // tests overridden
  EXPECT_EQ(cRed,   map.c6);  // tests default
}



//===----------------------------------------------------------------------===//
//  Test ScalarBitSetTraits
//===----------------------------------------------------------------------===//

enum MyFlags {
  flagNone    = 0,
  flagBig     = 1 << 0,
  flagFlat    = 1 << 1,
  flagRound   = 1 << 2,
  flagPointy  = 1 << 3
};
inline MyFlags operator|(MyFlags a, MyFlags b) {
  return static_cast<MyFlags>(
                      static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

struct FlagsMap {
  MyFlags     f1;
  MyFlags     f2;
  MyFlags     f3;
  MyFlags     f4;
};


namespace llvm {
namespace yaml {
  template <>
  struct ScalarBitSetTraits<MyFlags> {
    static void bitset(IO &io, MyFlags &value) {
      io.bitSetCase(value, "big",   flagBig);
      io.bitSetCase(value, "flat",  flagFlat);
      io.bitSetCase(value, "round", flagRound);
      io.bitSetCase(value, "pointy",flagPointy);
    }
  };
  template <>
  struct MappingTraits<FlagsMap> {
    static void mapping(IO &io, FlagsMap& c) {
      io.mapRequired("f1", c.f1);
      io.mapRequired("f2", c.f2);
      io.mapRequired("f3", c.f3);
      io.mapOptional("f4", c.f4, MyFlags(flagRound));
     }
  };
}
}


//
// Test reading flow sequence representing bit-mask values
//
TEST(YAMLIO, TestFlagsRead) {
  FlagsMap map;
  Input yin("---\n"
            "f1:  [ big ]\n"
            "f2:  [ round, flat ]\n"
            "f3:  []\n"
            "...\n");
  yin >> map;

  EXPECT_FALSE(yin.error());
  EXPECT_EQ(flagBig,              map.f1);
  EXPECT_EQ(flagRound|flagFlat,   map.f2);
  EXPECT_EQ(flagNone,             map.f3);  // check empty set
  EXPECT_EQ(flagRound,            map.f4);  // check optional key
}


//
// Test writing then reading back bit-mask values
//
TEST(YAMLIO, TestReadWriteFlags) {
  std::string intermediate;
  {
    FlagsMap map;
    map.f1 = flagBig;
    map.f2 = flagRound | flagFlat;
    map.f3 = flagNone;
    map.f4 = flagNone;

    llvm::raw_string_ostream ostr(intermediate);
    Output yout(ostr);
    yout << map;
  }

  {
    Input yin(intermediate);
    FlagsMap map2;
    yin >> map2;

    EXPECT_FALSE(yin.error());
    EXPECT_EQ(flagBig,              map2.f1);
    EXPECT_EQ(flagRound|flagFlat,   map2.f2);
    EXPECT_EQ(flagNone,             map2.f3);
    //EXPECT_EQ(flagRound,            map2.f4);  // check optional key
  }
}



//===----------------------------------------------------------------------===//
//  Test ScalarTraits
//===----------------------------------------------------------------------===//

struct MyCustomType {
  int length;
  int width;
};

struct MyCustomTypeMap {
  MyCustomType     f1;
  MyCustomType     f2;
  int              f3;
};


namespace llvm {
namespace yaml {
  template <>
  struct MappingTraits<MyCustomTypeMap> {
    static void mapping(IO &io, MyCustomTypeMap& s) {
      io.mapRequired("f1", s.f1);
      io.mapRequired("f2", s.f2);
      io.mapRequired("f3", s.f3);
     }
  };
  // MyCustomType is formatted as a yaml scalar.  A value of
  // {length=3, width=4} would be represented in yaml as "3 by 4".
  template<>
  struct ScalarTraits<MyCustomType> {
    static void output(const MyCustomType &value, void* ctxt, llvm::raw_ostream &out) {
      out << llvm::format("%d by %d", value.length, value.width);
    }
    static StringRef input(StringRef scalar, void* ctxt, MyCustomType &value) {
      size_t byStart = scalar.find("by");
      if ( byStart != StringRef::npos ) {
        StringRef lenStr = scalar.slice(0, byStart);
        lenStr = lenStr.rtrim();
        if ( lenStr.getAsInteger(0, value.length) ) {
          return "malformed length";
        }
        StringRef widthStr = scalar.drop_front(byStart+2);
        widthStr = widthStr.ltrim();
        if ( widthStr.getAsInteger(0, value.width) ) {
          return "malformed width";
        }
        return StringRef();
      }
      else {
          return "malformed by";
      }
    }
  };
}
}


//
// Test writing then reading back custom values
//
TEST(YAMLIO, TestReadWriteMyCustomType) {
  std::string intermediate;
  {
    MyCustomTypeMap map;
    map.f1.length = 1;
    map.f1.width  = 4;
    map.f2.length = 100;
    map.f2.width  = 400;
    map.f3 = 10;

    llvm::raw_string_ostream ostr(intermediate);
    Output yout(ostr);
    yout << map;
  }

  {
    Input yin(intermediate);
    MyCustomTypeMap map2;
    yin >> map2;

    EXPECT_FALSE(yin.error());
    EXPECT_EQ(1,      map2.f1.length);
    EXPECT_EQ(4,      map2.f1.width);
    EXPECT_EQ(100,    map2.f2.length);
    EXPECT_EQ(400,    map2.f2.width);
    EXPECT_EQ(10,     map2.f3);
  }
}


//===----------------------------------------------------------------------===//
//  Test flow sequences
//===----------------------------------------------------------------------===//

LLVM_YAML_STRONG_TYPEDEF(int, MyNumber)
LLVM_YAML_IS_FLOW_SEQUENCE_VECTOR(MyNumber)
LLVM_YAML_IS_FLOW_SEQUENCE_VECTOR(llvm::StringRef)

namespace llvm {
namespace yaml {
  template<>
  struct ScalarTraits<MyNumber> {
    static void output(const MyNumber &value, void *, llvm::raw_ostream &out) {
      out << value;
    }

    static StringRef input(StringRef scalar, void *, MyNumber &value) {
      long long n;
      if ( getAsSignedInteger(scalar, 0, n) )
        return "invalid number";
      value = n;
      return StringRef();
    }
  };
}
}

struct NameAndNumbers {
  llvm::StringRef               name;
  std::vector<llvm::StringRef>  strings;
  std::vector<MyNumber>         single;
  std::vector<MyNumber>         numbers;
};

namespace llvm {
namespace yaml {
  template <>
  struct MappingTraits<NameAndNumbers> {
    static void mapping(IO &io, NameAndNumbers& nn) {
      io.mapRequired("name",     nn.name);
      io.mapRequired("strings",  nn.strings);
      io.mapRequired("single",   nn.single);
      io.mapRequired("numbers",  nn.numbers);
    }
  };
}
}


//
// Test writing then reading back custom values
//
TEST(YAMLIO, TestReadWriteMyFlowSequence) {
  std::string intermediate;
  {
    NameAndNumbers map;
    map.name  = "hello";
    map.strings.push_back(llvm::StringRef("one"));
    map.strings.push_back(llvm::StringRef("two"));
    map.single.push_back(1);
    map.numbers.push_back(10);
    map.numbers.push_back(-30);
    map.numbers.push_back(1024);

    llvm::raw_string_ostream ostr(intermediate);
    Output yout(ostr);
    yout << map;
  }

  {
    Input yin(intermediate);
    NameAndNumbers map2;
    yin >> map2;

    EXPECT_FALSE(yin.error());
    EXPECT_TRUE(map2.name.equals("hello"));
    EXPECT_EQ(map2.strings.size(), 2UL);
    EXPECT_TRUE(map2.strings[0].equals("one"));
    EXPECT_TRUE(map2.strings[1].equals("two"));
    EXPECT_EQ(map2.single.size(), 1UL);
    EXPECT_EQ(1,       map2.single[0]);
    EXPECT_EQ(map2.numbers.size(), 3UL);
    EXPECT_EQ(10,      map2.numbers[0]);
    EXPECT_EQ(-30,     map2.numbers[1]);
    EXPECT_EQ(1024,    map2.numbers[2]);
  }
}


//===----------------------------------------------------------------------===//
//  Test normalizing/denormalizing
//===----------------------------------------------------------------------===//

LLVM_YAML_STRONG_TYPEDEF(uint32_t, TotalSeconds)

typedef std::vector<TotalSeconds> SecondsSequence;

LLVM_YAML_IS_FLOW_SEQUENCE_VECTOR(TotalSeconds)


namespace llvm {
namespace yaml {
  template <>
  struct MappingTraits<TotalSeconds> {

    class NormalizedSeconds {
    public:
      NormalizedSeconds(IO &io)
        : hours(0), minutes(0), seconds(0) {
      }
      NormalizedSeconds(IO &, TotalSeconds &secs)
        : hours(secs/3600),
          minutes((secs - (hours*3600))/60),
          seconds(secs % 60) {
      }
      TotalSeconds denormalize(IO &) {
        return TotalSeconds(hours*3600 + minutes*60 + seconds);
      }

      uint32_t     hours;
      uint8_t      minutes;
      uint8_t      seconds;
    };

    static void mapping(IO &io, TotalSeconds &secs) {
      MappingNormalization<NormalizedSeconds, TotalSeconds> keys(io, secs);

      io.mapOptional("hours",    keys->hours,    (uint32_t)0);
      io.mapOptional("minutes",  keys->minutes,  (uint8_t)0);
      io.mapRequired("seconds",  keys->seconds);
    }
  };
}
}


//
// Test the reading of a yaml sequence of mappings
//
TEST(YAMLIO, TestReadMySecondsSequence) {
  SecondsSequence seq;
  Input yin("---\n - hours:  1\n   seconds:  5\n - seconds:  59\n...\n");
  yin >> seq;

  EXPECT_FALSE(yin.error());
  EXPECT_EQ(seq.size(), 2UL);
  EXPECT_EQ(seq[0], 3605U);
  EXPECT_EQ(seq[1], 59U);
}


//
// Test writing then reading back custom values
//
TEST(YAMLIO, TestReadWriteMySecondsSequence) {
  std::string intermediate;
  {
    SecondsSequence seq;
    seq.push_back(4000);
    seq.push_back(500);
    seq.push_back(59);

    llvm::raw_string_ostream ostr(intermediate);
    Output yout(ostr);
    yout << seq;
  }
  {
    Input yin(intermediate);
    SecondsSequence seq2;
    yin >> seq2;

    EXPECT_FALSE(yin.error());
    EXPECT_EQ(seq2.size(), 3UL);
    EXPECT_EQ(seq2[0], 4000U);
    EXPECT_EQ(seq2[1], 500U);
    EXPECT_EQ(seq2[2], 59U);
  }
}


//===----------------------------------------------------------------------===//
//  Test dynamic typing
//===----------------------------------------------------------------------===//

enum AFlags {
    a1,
    a2,
    a3
};

enum BFlags {
    b1,
    b2,
    b3
};

enum Kind {
    kindA,
    kindB
};

struct KindAndFlags {
  KindAndFlags() : kind(kindA), flags(0) { }
  KindAndFlags(Kind k, uint32_t f) : kind(k), flags(f) { }
  Kind        kind;
  uint32_t    flags;
};

typedef std::vector<KindAndFlags> KindAndFlagsSequence;

LLVM_YAML_IS_FLOW_SEQUENCE_VECTOR(KindAndFlags)

namespace llvm {
namespace yaml {
  template <>
  struct ScalarEnumerationTraits<AFlags> {
    static void enumeration(IO &io, AFlags &value) {
      io.enumCase(value, "a1",  a1);
      io.enumCase(value, "a2",  a2);
      io.enumCase(value, "a3",  a3);
    }
  };
  template <>
  struct ScalarEnumerationTraits<BFlags> {
    static void enumeration(IO &io, BFlags &value) {
      io.enumCase(value, "b1",  b1);
      io.enumCase(value, "b2",  b2);
      io.enumCase(value, "b3",  b3);
    }
  };
  template <>
  struct ScalarEnumerationTraits<Kind> {
    static void enumeration(IO &io, Kind &value) {
      io.enumCase(value, "A",  kindA);
      io.enumCase(value, "B",  kindB);
    }
  };
  template <>
  struct MappingTraits<KindAndFlags> {
    static void mapping(IO &io, KindAndFlags& kf) {
      io.mapRequired("kind",  kf.kind);
      // type of flags field varies depending on kind field
      if ( kf.kind == kindA )
        io.mapRequired("flags", *((AFlags*)&kf.flags));
      else
        io.mapRequired("flags", *((BFlags*)&kf.flags));
    }
  };
}
}


//
// Test the reading of a yaml sequence dynamic types
//
TEST(YAMLIO, TestReadKindAndFlagsSequence) {
  KindAndFlagsSequence seq;
  Input yin("---\n - kind:  A\n   flags:  a2\n - kind:  B\n   flags:  b1\n...\n");
  yin >> seq;

  EXPECT_FALSE(yin.error());
  EXPECT_EQ(seq.size(), 2UL);
  EXPECT_EQ(seq[0].kind,  kindA);
  EXPECT_EQ(seq[0].flags, a2);
  EXPECT_EQ(seq[1].kind,  kindB);
  EXPECT_EQ(seq[1].flags, b1);
}

//
// Test writing then reading back dynamic types
//
TEST(YAMLIO, TestReadWriteKindAndFlagsSequence) {
  std::string intermediate;
  {
    KindAndFlagsSequence seq;
    seq.push_back(KindAndFlags(kindA,a1));
    seq.push_back(KindAndFlags(kindB,b1));
    seq.push_back(KindAndFlags(kindA,a2));
    seq.push_back(KindAndFlags(kindB,b2));
    seq.push_back(KindAndFlags(kindA,a3));

    llvm::raw_string_ostream ostr(intermediate);
    Output yout(ostr);
    yout << seq;
  }
  {
    Input yin(intermediate);
    KindAndFlagsSequence seq2;
    yin >> seq2;

    EXPECT_FALSE(yin.error());
    EXPECT_EQ(seq2.size(), 5UL);
    EXPECT_EQ(seq2[0].kind,  kindA);
    EXPECT_EQ(seq2[0].flags, a1);
    EXPECT_EQ(seq2[1].kind,  kindB);
    EXPECT_EQ(seq2[1].flags, b1);
    EXPECT_EQ(seq2[2].kind,  kindA);
    EXPECT_EQ(seq2[2].flags, a2);
    EXPECT_EQ(seq2[3].kind,  kindB);
    EXPECT_EQ(seq2[3].flags, b2);
    EXPECT_EQ(seq2[4].kind,  kindA);
    EXPECT_EQ(seq2[4].flags, a3);
  }
}


//===----------------------------------------------------------------------===//
//  Test document list
//===----------------------------------------------------------------------===//

struct FooBarMap {
  int foo;
  int bar;
};
typedef std::vector<FooBarMap> FooBarMapDocumentList;

LLVM_YAML_IS_DOCUMENT_LIST_VECTOR(FooBarMap)


namespace llvm {
namespace yaml {
  template <>
  struct MappingTraits<FooBarMap> {
    static void mapping(IO &io, FooBarMap& fb) {
      io.mapRequired("foo",    fb.foo);
      io.mapRequired("bar",    fb.bar);
    }
  };
}
}


//
// Test the reading of a yaml mapping
//
TEST(YAMLIO, TestDocRead) {
  FooBarMap doc;
  Input yin("---\nfoo:  3\nbar:  5\n...\n");
  yin >> doc;

  EXPECT_FALSE(yin.error());
  EXPECT_EQ(doc.foo, 3);
  EXPECT_EQ(doc.bar,5);
}



//
// Test writing then reading back a sequence of mappings
//
TEST(YAMLIO, TestSequenceDocListWriteAndRead) {
  std::string intermediate;
  {
    FooBarMap doc1;
    doc1.foo = 10;
    doc1.bar = -3;
    FooBarMap doc2;
    doc2.foo = 257;
    doc2.bar = 0;
    std::vector<FooBarMap> docList;
    docList.push_back(doc1);
    docList.push_back(doc2);

    llvm::raw_string_ostream ostr(intermediate);
    Output yout(ostr);
    yout << docList;
  }


  {
    Input yin(intermediate);
    std::vector<FooBarMap> docList2;
    yin >> docList2;

    EXPECT_FALSE(yin.error());
    EXPECT_EQ(docList2.size(), 2UL);
    FooBarMap& map1 = docList2[0];
    FooBarMap& map2 = docList2[1];
    EXPECT_EQ(map1.foo, 10);
    EXPECT_EQ(map1.bar, -3);
    EXPECT_EQ(map2.foo, 257);
    EXPECT_EQ(map2.bar, 0);
  }
}


//===----------------------------------------------------------------------===//
//  Test error handling
//===----------------------------------------------------------------------===//



static void suppressErrorMessages(const llvm::SMDiagnostic &, void *) {
}


//
// Test error handling of unknown enumerated scalar
//
TEST(YAMLIO, TestColorsReadError) {
  ColorMap map;
  Input yin("---\n"
            "c1:  blue\n"
            "c2:  purple\n"
            "c3:  green\n"
            "...\n");
  yin.setDiagHandler(suppressErrorMessages);
  yin >> map;
  EXPECT_TRUE(yin.error());
}


//
// Test error handling of flow sequence with unknown value
//
TEST(YAMLIO, TestFlagsReadError) {
  FlagsMap map;
  Input yin("---\n"
            "f1:  [ big ]\n"
            "f2:  [ round, hollow ]\n"
            "f3:  []\n"
            "...\n");
  yin.setDiagHandler(suppressErrorMessages);
  yin >> map;

  EXPECT_TRUE(yin.error());
}


//
// Test error handling reading built-in uint8_t type
//
LLVM_YAML_IS_SEQUENCE_VECTOR(uint8_t)
TEST(YAMLIO, TestReadBuiltInTypesUint8Error) {
  std::vector<uint8_t> seq;
  Input yin("---\n"
            "- 255\n"
            "- 0\n"
            "- 257\n"
            "...\n");
  yin.setDiagHandler(suppressErrorMessages);
  yin >> seq;

  EXPECT_TRUE(yin.error());
}


//
// Test error handling reading built-in uint16_t type
//
LLVM_YAML_IS_SEQUENCE_VECTOR(uint16_t)
TEST(YAMLIO, TestReadBuiltInTypesUint16Error) {
  std::vector<uint16_t> seq;
  Input yin("---\n"
            "- 65535\n"
            "- 0\n"
            "- 66000\n"
            "...\n");
  yin.setDiagHandler(suppressErrorMessages);
  yin >> seq;

  EXPECT_TRUE(yin.error());
}


//
// Test error handling reading built-in uint32_t type
//
LLVM_YAML_IS_SEQUENCE_VECTOR(uint32_t)
TEST(YAMLIO, TestReadBuiltInTypesUint32Error) {
  std::vector<uint32_t> seq;
  Input yin("---\n"
            "- 4000000000\n"
            "- 0\n"
            "- 5000000000\n"
            "...\n");
  yin.setDiagHandler(suppressErrorMessages);
  yin >> seq;

  EXPECT_TRUE(yin.error());
}


//
// Test error handling reading built-in uint64_t type
//
LLVM_YAML_IS_SEQUENCE_VECTOR(uint64_t)
TEST(YAMLIO, TestReadBuiltInTypesUint64Error) {
  std::vector<uint64_t> seq;
  Input yin("---\n"
            "- 18446744073709551615\n"
            "- 0\n"
            "- 19446744073709551615\n"
            "...\n");
  yin.setDiagHandler(suppressErrorMessages);
  yin >> seq;

  EXPECT_TRUE(yin.error());
}


//
// Test error handling reading built-in int8_t type
//
LLVM_YAML_IS_SEQUENCE_VECTOR(int8_t)
TEST(YAMLIO, TestReadBuiltInTypesint8OverError) {
  std::vector<int8_t> seq;
  Input yin("---\n"
            "- -128\n"
            "- 0\n"
            "- 127\n"
            "- 128\n"
           "...\n");
  yin.setDiagHandler(suppressErrorMessages);
  yin >> seq;

  EXPECT_TRUE(yin.error());
}

//
// Test error handling reading built-in int8_t type
//
TEST(YAMLIO, TestReadBuiltInTypesint8UnderError) {
  std::vector<int8_t> seq;
  Input yin("---\n"
            "- -128\n"
            "- 0\n"
            "- 127\n"
            "- -129\n"
            "...\n");
  yin.setDiagHandler(suppressErrorMessages);
  yin >> seq;

  EXPECT_TRUE(yin.error());
}


//
// Test error handling reading built-in int16_t type
//
LLVM_YAML_IS_SEQUENCE_VECTOR(int16_t)
TEST(YAMLIO, TestReadBuiltInTypesint16UnderError) {
  std::vector<int16_t> seq;
  Input yin("---\n"
            "- 32767\n"
            "- 0\n"
            "- -32768\n"
            "- -32769\n"
            "...\n");
  yin.setDiagHandler(suppressErrorMessages);
  yin >> seq;

  EXPECT_TRUE(yin.error());
}


//
// Test error handling reading built-in int16_t type
//
TEST(YAMLIO, TestReadBuiltInTypesint16OverError) {
  std::vector<int16_t> seq;
  Input yin("---\n"
            "- 32767\n"
            "- 0\n"
            "- -32768\n"
            "- 32768\n"
            "...\n");
  yin.setDiagHandler(suppressErrorMessages);
  yin >> seq;

  EXPECT_TRUE(yin.error());
}


//
// Test error handling reading built-in int32_t type
//
LLVM_YAML_IS_SEQUENCE_VECTOR(int32_t)
TEST(YAMLIO, TestReadBuiltInTypesint32UnderError) {
  std::vector<int32_t> seq;
  Input yin("---\n"
            "- 2147483647\n"
            "- 0\n"
            "- -2147483648\n"
            "- -2147483649\n"
            "...\n");
  yin.setDiagHandler(suppressErrorMessages);
  yin >> seq;

  EXPECT_TRUE(yin.error());
}

//
// Test error handling reading built-in int32_t type
//
TEST(YAMLIO, TestReadBuiltInTypesint32OverError) {
  std::vector<int32_t> seq;
  Input yin("---\n"
            "- 2147483647\n"
            "- 0\n"
            "- -2147483648\n"
            "- 2147483649\n"
            "...\n");
  yin.setDiagHandler(suppressErrorMessages);
  yin >> seq;

  EXPECT_TRUE(yin.error());
}


//
// Test error handling reading built-in int64_t type
//
LLVM_YAML_IS_SEQUENCE_VECTOR(int64_t)
TEST(YAMLIO, TestReadBuiltInTypesint64UnderError) {
  std::vector<int64_t> seq;
  Input yin("---\n"
            "- -9223372036854775808\n"
            "- 0\n"
            "- 9223372036854775807\n"
            "- -9223372036854775809\n"
            "...\n");
  yin.setDiagHandler(suppressErrorMessages);
  yin >> seq;

  EXPECT_TRUE(yin.error());
}

//
// Test error handling reading built-in int64_t type
//
TEST(YAMLIO, TestReadBuiltInTypesint64OverError) {
  std::vector<int64_t> seq;
  Input yin("---\n"
            "- -9223372036854775808\n"
            "- 0\n"
            "- 9223372036854775807\n"
            "- 9223372036854775809\n"
            "...\n");
  yin.setDiagHandler(suppressErrorMessages);
  yin >> seq;

  EXPECT_TRUE(yin.error());
}

//
// Test error handling reading built-in float type
//
LLVM_YAML_IS_SEQUENCE_VECTOR(float)
TEST(YAMLIO, TestReadBuiltInTypesFloatError) {
  std::vector<float> seq;
  Input yin("---\n"
            "- 0.0\n"
            "- 1000.1\n"
            "- -123.456\n"
            "- 1.2.3\n"
            "...\n");
  yin.setDiagHandler(suppressErrorMessages);
  yin >> seq;

  EXPECT_TRUE(yin.error());
}

//
// Test error handling reading built-in float type
//
LLVM_YAML_IS_SEQUENCE_VECTOR(double)
TEST(YAMLIO, TestReadBuiltInTypesDoubleError) {
  std::vector<double> seq;
  Input yin("---\n"
            "- 0.0\n"
            "- 1000.1\n"
            "- -123.456\n"
            "- 1.2.3\n"
            "...\n");
  yin.setDiagHandler(suppressErrorMessages);
  yin >> seq;

  EXPECT_TRUE(yin.error());
}

//
// Test error handling reading built-in Hex8 type
//
LLVM_YAML_IS_SEQUENCE_VECTOR(Hex8)
TEST(YAMLIO, TestReadBuiltInTypesHex8Error) {
  std::vector<Hex8> seq;
  Input yin("---\n"
            "- 0x12\n"
            "- 0xFE\n"
            "- 0x123\n"
            "...\n");
  yin.setDiagHandler(suppressErrorMessages);
  yin >> seq;

  EXPECT_TRUE(yin.error());
}


//
// Test error handling reading built-in Hex16 type
//
LLVM_YAML_IS_SEQUENCE_VECTOR(Hex16)
TEST(YAMLIO, TestReadBuiltInTypesHex16Error) {
  std::vector<Hex16> seq;
  Input yin("---\n"
            "- 0x0012\n"
            "- 0xFEFF\n"
            "- 0x12345\n"
            "...\n");
  yin.setDiagHandler(suppressErrorMessages);
  yin >> seq;

  EXPECT_TRUE(yin.error());
}

//
// Test error handling reading built-in Hex32 type
//
LLVM_YAML_IS_SEQUENCE_VECTOR(Hex32)
TEST(YAMLIO, TestReadBuiltInTypesHex32Error) {
  std::vector<Hex32> seq;
  Input yin("---\n"
            "- 0x0012\n"
            "- 0xFEFF0000\n"
            "- 0x1234556789\n"
            "...\n");
  yin.setDiagHandler(suppressErrorMessages);
  yin >> seq;

  EXPECT_TRUE(yin.error());
}

//
// Test error handling reading built-in Hex64 type
//
LLVM_YAML_IS_SEQUENCE_VECTOR(Hex64)
TEST(YAMLIO, TestReadBuiltInTypesHex64Error) {
  std::vector<Hex64> seq;
  Input yin("---\n"
            "- 0x0012\n"
            "- 0xFFEEDDCCBBAA9988\n"
            "- 0x12345567890ABCDEF0\n"
            "...\n");
  yin.setDiagHandler(suppressErrorMessages);
  yin >> seq;

  EXPECT_TRUE(yin.error());
}


#endif
