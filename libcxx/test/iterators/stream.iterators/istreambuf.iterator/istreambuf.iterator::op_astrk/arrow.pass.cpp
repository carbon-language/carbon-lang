//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <iterator>

// istreambuf_iterator

// pointer operator->() const;

#include <iostream> 
#include <sstream>
#include <streambuf> 

typedef char C;
int main ()
{
   std::istringstream s("filename");
   std::istreambuf_iterator<char> i(s);

   (*i).~C();  // This is well-formed...
   i->~C();  // ... so this should be supported!
}
