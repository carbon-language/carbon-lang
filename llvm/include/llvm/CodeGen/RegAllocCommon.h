

#ifndef REG_ALLOC_COMMON_H
#define REG_ALLOC_COMMON_H

// set DEBUG_RA for printing out debug messages
// if DEBUG_RA is 1 normal output messages
// if DEBUG_RA is 2 extensive debug info for each instr

enum RegAllocDebugLevel_t {
  RA_DEBUG_None    = 0,
  RA_DEBUG_Normal  = 1,
  RA_DEBUG_Verbose = 2,
};

extern RegAllocDebugLevel_t DEBUG_RA;

#endif
