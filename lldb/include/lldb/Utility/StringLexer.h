//===--------------------- StringLexer.h -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef utility_StringLexer_h_
#define utility_StringLexer_h_

#include <string>
#include <list>

namespace lldb_utility {
    
class StringLexer
{
public:
    typedef std::string::size_type Position;
    typedef std::string::size_type Size;

    typedef std::string::value_type Character;
    
    StringLexer (std::string s);
    
    StringLexer (const StringLexer& rhs);
    
    Character
    Peek ();
    
    bool
    NextIf (Character c);
    
    Character
    Next ();
    
    bool
    HasAtLeast (Size s);
    
    bool
    HasAny (Character c);
    
    void
    PutBack (Character c);
    
    StringLexer&
    operator = (const StringLexer& rhs);
    
private:
    std::string m_data;
    Position m_position;
    std::list<Character> m_putback_data;
    
    void
    Consume();
};

} // namespace lldb_private

#endif // #ifndef utility_StringLexer_h_
