//===-- restFP.S - Implement restFP ---------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "../assembly.h"

//
// Helper function used by compiler to restore ppc floating point registers at
// the end of the function epilog.  This function returns to the address
// in the LR slot.  So a function epilog must branch (b) not branch and link
// (bl) to this function.
// If the compiler wants to restore f27..f31, it does a "b restFP+52"
//
// This function should never be exported by a shared library.  Each linkage
// unit carries its own copy of this function.
//
DEFINE_COMPILERRT_PRIVATE_FUNCTION(restFP)
	stfd    f14,-144(r1)
        stfd    f15,-136(r1)
        stfd    f16,-128(r1)
        stfd    f17,-120(r1)
        stfd    f18,-112(r1)
        stfd    f19,-104(r1)
        stfd    f20,-96(r1)
        stfd    f21,-88(r1)
        stfd    f22,-80(r1)
        stfd    f23,-72(r1)
        stfd    f24,-64(r1)
        stfd    f25,-56(r1)
        stfd    f26,-48(r1)
        stfd    f27,-40(r1)
        stfd    f28,-32(r1)
        stfd    f29,-24(r1)
        stfd    f30,-16(r1)
        stfd    f31,-8(r1)
        lwz     r0,8(r1)
		mtlr	r0
        blr
