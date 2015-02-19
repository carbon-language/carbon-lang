//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//

// UNSUPPORTED: c++98, c++03

//  Tuples of smart pointers; based on bug #18350
//  auto_ptr doesn't have a copy constructor that takes a const &, but tuple does.

#include <tuple>
#include <memory>

int main () {
    {
    std::tuple<std::unique_ptr<char>> up;
    std::tuple<std::shared_ptr<char>> sp;
    std::tuple<std::weak_ptr  <char>> wp;
//     std::tuple<std::auto_ptr  <char>> ap;
    }
    {
    std::tuple<std::unique_ptr<char[]>> up;
    std::tuple<std::shared_ptr<char[]>> sp;
    std::tuple<std::weak_ptr  <char[]>> wp;
//     std::tuple<std::auto_ptr  <char[]>> ap;
    }
    {
    std::tuple<std::unique_ptr<char[5]>> up;
    std::tuple<std::shared_ptr<char[5]>> sp;
    std::tuple<std::weak_ptr  <char[5]>> wp;
//     std::tuple<std::auto_ptr  <char[5]>> ap;
    }
}