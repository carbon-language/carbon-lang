//===-- Callback.cpp - Trap handler for function resolution ---------------===//
//
// This file defines the SIGSEGV handler which is invoked when a reference to a
// non-codegen'd function is found.
//
//===----------------------------------------------------------------------===//

#include "VM.h"
#include "Support/Statistic.h"
#include <signal.h>
#include <ucontext.h>
#include <iostream>

static VM *TheVM = 0;

static void TrapHandler(int TN, siginfo_t *SI, ucontext_t *ucp) {
  assert(TN == SIGSEGV && "Should be SIGSEGV!");

#ifdef REG_EIP   /* this code does not compile on Sparc! */
  if (SI->si_code != SEGV_MAPERR || SI->si_addr != 0 ||
      ucp->uc_mcontext.gregs[REG_EIP] != 0) {
    std::cerr << "Bad SEGV encountered EIP = 0x" << std::hex
	      << ucp->uc_mcontext.gregs[REG_EIP] << " addr = "
	      << SI->si_addr << "!\n";

    struct sigaction SA;              // Restore old SEGV handler...
    SA.sa_handler = SIG_DFL;
    SA.sa_flags = SA_NOMASK;
    sigaction(SIGSEGV, &SA, 0);
    return;  // Should core dump now...
  }

  // The call instruction should have pushed the return value onto the stack...
  unsigned RefAddr = *(unsigned*)ucp->uc_mcontext.gregs[REG_ESP];
  RefAddr -= 4;  // Backtrack to the reference itself...

  DEBUG(std::cerr << "In SEGV handler! Addr=0x" << std::hex << RefAddr
                  << " ESP=0x" << ucp->uc_mcontext.gregs[REG_ESP] << std::dec
                  << ": Resolving call to function: "
                  << TheVM->getFunctionReferencedName((void*)RefAddr) << "\n");

  // Sanity check to make sure this really is a call instruction...
  assert(((unsigned char*)RefAddr)[-1] == 0xE8 && "Not a call instr!");
  
  unsigned NewVal = (unsigned)TheVM->resolveFunctionReference((void*)RefAddr);

  // Rewrite the call target... so that we don't fault every time we execute
  // the call.
  *(unsigned*)RefAddr = NewVal-RefAddr-4;    

  // Change the instruction pointer to be the real target of the call...
  ucp->uc_mcontext.gregs[REG_EIP] = NewVal;

#endif
}


void VM::registerCallback() {
  TheVM = this;

  // Register the signal handler...
  struct sigaction SA;
  SA.sa_sigaction = (void (*)(int, siginfo_t*, void*))TrapHandler;
  sigfillset(&SA.sa_mask);               // Block all signals while codegen'ing
  SA.sa_flags = SA_NOCLDSTOP|SA_SIGINFO; // Get siginfo
  sigaction(SIGSEGV, &SA, 0);            // Install the handler
}


