//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <string>

int main (int argc, char const *argv[])
{
    std::string stdstring("Hello\t\tWorld\nI am here\t\tto say hello\n"); //%self.addTearDownHook(lambda x: x.runCmd("setting set escape-non-printables true"))
    const char* constcharstar = stdstring.c_str();
    return 0; //% self.assertTrue(self.frame().FindVariable('stdstring').GetSummary() == '"Hello\\t\\tWorld\\nI am here\\t\\tto say hello\\n"')
     //% self.assertTrue(self.frame().FindVariable('constcharstar').GetSummary() == '"Hello\\t\\tWorld\\nI am here\\t\\tto say hello\\n"')
     //% self.runCmd("setting set escape-non-printables false")
     //% self.assertTrue(self.frame().FindVariable('stdstring').GetSummary() == '"Hello\t\tWorld\nI am here\t\tto say hello\n"')
     //% self.assertTrue(self.frame().FindVariable('constcharstar').GetSummary() == '"Hello\t\tWorld\nI am here\t\tto say hello\n"')
}
