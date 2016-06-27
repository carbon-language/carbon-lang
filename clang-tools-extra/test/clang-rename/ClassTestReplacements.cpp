// RUN: mkdir -p %T/fixes
// RUN: cat %s > %t.cpp
// RUN: clang-rename -offset=225 -new-name=Hector -export-fixes=%T/fixes.yaml %t.cpp --
// RUN: clang-apply-replacements %T
// RUN: sed 's,//.*,,' %t.cpp | FileCheck %s
class Cla  // CHECK: class Hector
{
};

// Use grep -FUbo 'Cla' <file> to get the correct offset of Cla when changing
// this file.
