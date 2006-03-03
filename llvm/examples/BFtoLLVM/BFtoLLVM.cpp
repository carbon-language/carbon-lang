//===-- BFtoLLVM.cpp - BF language Front End for LLVM ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a simple front end for the BF language.  It is compatible with the
// language as described in "The BrainF*** Language Specification (01 January
// 2002)", which is available from http://esoteric.sange.fi/ENSI . It does not
// implement the optional keyword # ("Output partial tape state").
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <vector>
#include <fstream>
#include <cerrno>
#include <cstring>
#include <string>
#include <cstdio>
#include <cassert>

void emitDeclarations(std::ofstream &dest) {
  dest << "; This assembly code brought to you by BFtoLLVM\n"
       << "\nimplementation\n"
       << "\n; Declarations\n"
       << "\ndeclare int %getchar()\n"
       << "declare int %putchar(int)\n"
       << "declare void %llvm.memset.i32(sbyte*, ubyte, uint, uint)\n"
       << "\n";
}

void emitMainFunctionProlog(std::ofstream &dest) {
  dest << "\n; Main function\n"
       << "int %main(int %argc, sbyte** %argv) {\n"
       << "\nentry:\n"
       << "%arr = alloca sbyte, uint 30000\n"
       << "call void (sbyte*, ubyte, uint, uint)* %llvm.memset.i32"
       << "(sbyte* %arr, ubyte 0, uint 30000, uint 1)\n"
       << "%ptrbox = alloca sbyte*\n"
       << "store sbyte* %arr, sbyte **%ptrbox\n"
       << "\n";
}

void emitMainFunctionEpilog(std::ofstream &dest) {
  dest << "ret int 0\n"
       << "}\n";
}

std::string gensym (const std::string varName, bool percent = true) {
  char buf[80];
  static unsigned int SymbolCounter = 0;
  sprintf (buf, "%s%s%u", percent ? "%" : "", varName.c_str(), SymbolCounter++);
  return std::string (buf);
}

void emitArith (std::string op, char delta, std::ofstream &dest) {
  std::string ptr = gensym (op + "ptr"),
              val = gensym (op + "val"),
              result = gensym (op + "result");
  dest << ptr << " = load sbyte** %ptrbox\n"
       << val << " = load sbyte* " << ptr << "\n"
       << result << " = add sbyte " << val << ", " << (int)delta << "\n"
       << "store sbyte " << result << ", sbyte* " << ptr << "\n";
}

// + becomes ++*p; and - becomes --*p;
void emitPlus  (std::ofstream &dest, int ct) { emitArith ("plus",  +ct, dest); }
void emitMinus (std::ofstream &dest, int ct) { emitArith ("minus", -ct, dest); }

void emitLoadAndCast (std::string ptr, std::string val, std::string cast,
                      std::string type, std::ofstream &dest) {
  dest << ptr << " = load sbyte** %ptrbox\n"
       << val << " = load sbyte* " << ptr << "\n"
       << cast << " = cast sbyte " << val << " to " << type << "\n";
}

// , becomes *p = getchar();
void emitComma(std::ofstream &dest, int ct) {
  assert (ct == 1);
  std::string ptr = gensym("commaptr"), read = gensym("commaread"),
              cast = gensym("commacast");
  dest << ptr << " = load sbyte** %ptrbox\n"
       << read << " = call int %getchar()\n"
       << cast << " = cast int " << read << " to sbyte\n"
       << "store sbyte " << cast << ", sbyte* " << ptr << "\n";
}

// . becomes putchar(*p);
void emitDot(std::ofstream &dest, int ct) {
  assert (ct == 1);
  std::string ptr = gensym("dotptr"), val = gensym("dotval"),
              cast = gensym("dotcast");
  emitLoadAndCast (ptr, val, cast, "int", dest);
  dest << "call int %putchar(int " << cast << ")\n";
}

void emitPointerArith(std::string opname, int delta, std::ofstream &dest) {
  std::string ptr = gensym(opname + "ptr"), result = gensym(opname + "result");
  dest << ptr << " = load sbyte** %ptrbox\n"
       << result << " = getelementptr sbyte* " << ptr << ", int " << delta
       << "\n"
       << "store sbyte* " << result << ", sbyte** %ptrbox\n";
}

// < becomes --p; and > becomes ++p;
void emitLT(std::ofstream &dest, int ct) { emitPointerArith ("lt", -ct, dest); }
void emitGT(std::ofstream &dest, int ct) { emitPointerArith ("gt", +ct, dest); }

static std::vector<std::string> whileStack;

// [ becomes while (*p) {
void emitLeftBracket(std::ofstream &dest, int ct) {
  assert (ct == 1);
  std::string whileName = gensym ("While", false);
  whileStack.push_back (whileName);
  dest << "br label %testFor" << whileName << "\n"
       << "\ninside" << whileName << ":\n";
}

// ] becomes }
void emitRightBracket(std::ofstream &dest, int ct) {
  assert (ct == 1);
  std::string whileName = whileStack.back (),
              ptr = gensym("bracketptr"),
              val = gensym("bracketval"),
              cast = gensym("bracketcast");
  whileStack.pop_back ();
  dest << "br label %testFor" << whileName << "\n"
       << "\ntestFor" << whileName << ":\n";
  emitLoadAndCast (ptr, val, cast, "bool", dest);
  dest << "br bool " << cast << ", label %inside" << whileName << ", "
       << "label %after" << whileName << "\n"
       << "\nafter" << whileName << ":\n";
}

typedef void (*FuncTy)(std::ofstream &, int);
static FuncTy table[256];
static bool multi[256];

void consume (int ch, int repeatCount, std::ofstream &dest) {
  FuncTy func = table[ch];
  if (!func)
    return;
  else if (multi[ch])
    func (dest, repeatCount);
  else
    for (int i = 0; i < repeatCount; ++i)
      func (dest, 1);
}

void initializeTable() {
  memset (table, 0, 256);
  memset (multi, 0, 256);
  table[(int)'+'] = emitPlus;          multi[(int)'+'] = true;
  table[(int)'-'] = emitMinus;         multi[(int)'-'] = true;
  table[(int)','] = emitComma;         multi[(int)','] = false;
  table[(int)'.'] = emitDot;           multi[(int)'.'] = false;
  table[(int)'<'] = emitLT;            multi[(int)'<'] = true;
  table[(int)'>'] = emitGT;            multi[(int)'>'] = true;
  table[(int)'['] = emitLeftBracket;   multi[(int)'['] = false;
  table[(int)']'] = emitRightBracket;  multi[(int)']'] = false;
}

int main (int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0] << " input-source output-llvm\n";
    return 1;
  }

  char *sourceFileName = argv[1];
  char *destFileName = argv[2];

  std::ifstream src (sourceFileName);
  if (!src.good()) {
    std::cerr << sourceFileName << ": " << strerror(errno) << "\n";
    return 1;
  }

  std::ofstream dest (destFileName);
  if (!dest.good()) {
    std::cerr << destFileName << ": " << strerror(errno) << "\n";
    return 1;
  }

  emitDeclarations(dest);
  emitMainFunctionProlog(dest);

  initializeTable();
  char ch, lastCh;
  src >> lastCh;
  int repeatCount = 1;
  for (src >> ch; !src.eof (); src >> ch, ++repeatCount)
    if (ch != lastCh) {
      consume (lastCh, repeatCount, dest);
      lastCh = ch;
      repeatCount = 0;
    }
  consume (lastCh, repeatCount, dest);

  emitMainFunctionEpilog(dest);

  src.close();
  dest.close();
  return 0;
}
