//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//


extern "C" void __offload_target_main(void);

int main(int argc, char ** argv)
{
    __offload_target_main();
    return 0;
}
